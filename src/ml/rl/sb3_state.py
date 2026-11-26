"""What it takes to continue a Stable-Baselines3 agent where it stopped, and how much of that SB3 itself provides.

SB3's documented ``save`` / ``load`` keep the policy and optimizer parameters, the hyperparameters, the number of timesteps, the updates counter, the last observation, the episode statistics and (for the recurrent
policy) nothing of the LSTM's running state; ``save_replay_buffer`` / ``load_replay_buffer`` keep an off-policy agent's experience. They do **not** keep the random streams, the environment, or the recurrent state, and
``load`` re-seeds every generator from the model's seed, so a model that is merely reloaded and trained on follows a different trajectory from the one that was never stopped. This module is the application's manifest of
the rest:

* the global generators (PyTorch, NumPy, Python ``random``) and the action space's own generator (SAC's warm-up actions are sampled from it), saved after every unit and re-applied AFTER ``load``;
* the environment, mid-episode (a pickle of the wrapped environment: its ledger, clock and generator), which a reloaded model otherwise replaces by a freshly built one;
* the observation the next step acts on and the episode-start flags (``_last_obs``, ``_last_episode_starts``: private attributes that ``load`` leaves unset, so that a reloaded agent would reset its environment and start a new episode);
* the LSTM hidden and cell states of the recurrent policy, which SB3 re-zeroes on ``load`` (read from ``_last_lstm_states``, a private attribute: its absence caps the guarantee at ``epoch_boundary``);
* the metric history and the per-unit counters.

Continuation is exact only when all of this is restored AND the run is a single environment on one device with constant learning-rate and clip schedules (a callable schedule would depend on the total number of timesteps).
Anything less is reported at a lower level; ``describe_limits`` lists them. Vectorised environments, ``VecNormalize`` (not used by the platform's agent) and GPU runs are not covered.
"""

from __future__ import annotations

import hashlib
import io
import pickle
import random

import numpy as np
import torch

ALGOS = ("ppo", "sac", "recurrent_ppo")
#: the library versions the exact-recovery claim was verified on (``tests/test_fresh_process_sb3.py`` and ``tests/test_sb3_recovery_validation.py``). Another version may well work; it is not claimed.
TESTED = {"stable_baselines3": ("2.9",), "sb3_contrib": ("2.9",)}

#: Every private (underscore) member of Stable-Baselines3 this module touches, why, and what happens when it is missing. Everything else is public API (``save``, ``load``, ``save_replay_buffer``, ``num_timesteps``,
#: ``policy``, ``actor``, ``critic``, ``replay_buffer``, ``get_env``, ``n_envs``, ``action_space``, ``ep_info_buffer``, ``logger``).
PRIVATE_USED = {
    "_last_obs": "the observation the next step acts on; SB3's own load() leaves it None, which would reset the environment and start a new episode. Missing: the exactness claim is withdrawn.",
    "_last_episode_starts": "the episode-start flags that go with it. Missing: as above.",
    "_last_lstm_states": "RecurrentPPO's running LSTM state (re-zeroed by load()). Missing: the exactness claim is withdrawn and the run says so.",
    "_n_updates": "the number of gradient updates; read only to cross-check a restore (a mismatch lowers the guarantee) and for the metrics row.",
    "_episode_num": "the number of finished episodes; read only to cross-check a restore and for the metrics row.",
}


def versions() -> dict:
    out = {}
    for name in ("stable_baselines3", "sb3_contrib", "gymnasium"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "?"))
        except ImportError:
            pass
    return out


class GlobalSchedule:
    """A learning-rate or clip-range schedule that is a function of the agent's own ``num_timesteps`` and of the whole run's length, not of SB3's ``progress_remaining``. SB3 computes ``progress_remaining`` from the
    timesteps of the current ``learn`` call, so training in checkpointed blocks would restart a linear schedule at every block and no longer equal one uninterrupted call. This schedule ignores that argument. The clock is set
    by :class:`Clock` from the public ``num_timesteps``; the object is picklable (SB3 stores it in the model zip) and ``kind="constant"`` is exactly the float."""

    def __init__(self, base: float, kind: str, total: int, floor: float = 0.1):
        self.base, self.kind, self.total, self.floor, self.timesteps = float(base), kind, max(int(total), 1), float(floor), 0

    def __call__(self, progress_remaining: float) -> float:
        if self.kind == "constant":
            return self.base
        done = min(max(self.timesteps / self.total, 0.0), 1.0)
        return self.base * (1.0 - (1.0 - self.floor) * done)

    def __repr__(self) -> str:
        return f"GlobalSchedule({self.base}, {self.kind}, total={self.total})"


def make_schedule(base: float, kind: str, total: int):
    """A plain float for a constant schedule (what SB3 gets without this module), a :class:`GlobalSchedule` otherwise."""
    if kind in (None, "", "constant"):
        return float(base)
    if kind != "linear":
        raise ValueError(f"unknown schedule {kind!r} (constant or linear)")
    return GlobalSchedule(base, kind, total)


def schedules_of(agent) -> list:
    """The :class:`GlobalSchedule` objects the agent uses (found on its public ``learning_rate`` and ``clip_range`` attributes), so that their clock can be set."""
    return [s for s in (getattr(agent, "learning_rate", None), getattr(agent, "clip_range", None)) if isinstance(s, GlobalSchedule)]


def set_clock(agent) -> None:
    for s in schedules_of(agent):
        s.timesteps = float(agent.num_timesteps)


def ceiling_reasons(agent, kind: str, device: str, chunk: int, *, tested: dict | None = None) -> list[str]:
    """What prevents an EXACT continuation of this agent. An empty list means nothing known does. These are not guesses about the library: each is a configuration this platform's tests did not cover (or a member that is
    missing), so the guarantee is capped at ``epoch_boundary`` and the reason is shown in the monitor."""
    out = []
    if device not in ("cpu", "auto") or str(getattr(getattr(agent, "device", None), "type", "cpu")) != "cpu":
        out.append(f"device {device}: GPU kernels and CUDA generators were not verified")
    n_envs = int(getattr(agent, "n_envs", 1) or 1)
    if n_envs != 1:
        out.append(f"{n_envs} vectorised environments: only one environment's state is saved")
    try:
        if agent.get_vec_normalize_env() is not None:
            out.append("VecNormalize: the running observation and reward statistics are not saved")
    except AttributeError:
        pass
    for name in ("_last_obs", "_last_episode_starts"):
        if not hasattr(agent, name):
            out.append(f"SB3 no longer has {name}: the next observation cannot be restored")
    if kind == "recurrent_ppo" and not hasattr(agent, "_last_lstm_states"):
        out.append("SB3 no longer has _last_lstm_states: the recurrent state cannot be restored")
    freq = getattr(agent, "train_freq", None)
    if freq is not None:
        unit = getattr(getattr(freq, "unit", None), "value", getattr(freq, "unit", "step"))
        f = int(getattr(freq, "frequency", 1))
        if str(unit) == "episode":
            out.append("train_freq in episodes: a block boundary can fall inside a collection phase")
        elif f > 1 and chunk % f:
            out.append(f"train_freq {f} does not divide the block of {chunk} timesteps: a block boundary falls inside a collection phase")
    if kind in ("ppo", "recurrent_ppo"):
        n_steps = int(getattr(agent, "n_steps", 1))
        if chunk % n_steps:
            out.append(f"the block of {chunk} timesteps is not a whole number of {n_steps}-step rollouts")
    now = versions()
    for lib, ok in (tested or TESTED).items():
        v = now.get(lib)
        if v is not None and not any(v == o or v.startswith(o + ".") for o in ok):
            out.append(f"{lib} {v} was not verified (verified: {', '.join(ok)})")
    return out


def _digest_tensors(tensors) -> str:
    h = hashlib.sha256()
    for name, x in tensors:
        h.update(name.encode())
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _feed(h, obj, depth: int = 0) -> None:
    """Hash the content of ``obj`` in a canonical way (a pickle of the same object is not byte-stable across a round trip, a canonical walk is)."""
    if depth > 12:
        h.update(b"<deep>")
    elif isinstance(obj, np.random.Generator):
        _feed(h, obj.bit_generator.state, depth + 1)
    elif isinstance(obj, np.ndarray):
        h.update(str(obj.dtype).encode() + str(obj.shape).encode() + np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            h.update(str(k).encode())
            _feed(h, obj[k], depth + 1)
    elif isinstance(obj, (list, tuple)):
        h.update(b"[%d" % len(obj))
        for x in obj:
            _feed(h, x, depth + 1)
    elif isinstance(obj, (int, float, str, bool, bytes)) or obj is None:
        h.update(repr(obj).encode())
    elif isinstance(obj, np.generic):
        h.update(repr(obj.item()).encode())
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        _feed(h, vars(obj), depth + 1)
    else:
        h.update(type(obj).__name__.encode())


def env_digest(env) -> str:
    """A digest of the environment's training state: the unwrapped environment's attributes and generator, and the monitor's episode record (wall-clock timings excluded)."""
    h = hashlib.sha256()
    base = env.unwrapped
    _feed(h, {k: v for k, v in vars(base).items() if k not in ("np_random", "_np_random")})
    _feed(h, base.np_random.bit_generator.state)
    for name in ("get_episode_rewards", "get_episode_lengths"):
        if hasattr(env, name):
            _feed(h, list(getattr(env, name)()))
    for name in ("rewards", "needs_reset", "total_steps"):
        if hasattr(env, name):
            _feed(h, getattr(env, name))
    return h.hexdigest()


def _optimizers(agent) -> dict:
    out = {}
    for name, obj in (("policy", getattr(getattr(agent, "policy", None), "optimizer", None)), ("actor", getattr(getattr(agent, "actor", None), "optimizer", None)),
                      ("critic", getattr(getattr(agent, "critic", None), "optimizer", None)), ("ent_coef", getattr(agent, "ent_coef_optimizer", None))):
        if obj is not None:
            out[name] = obj
    return out


def fingerprint_state(agent, env=None) -> dict:
    """Digests of the parts of a live agent whose agreement after a restore is what "exact" means: the counters, the policy parameters, every optimizer's state, the replay buffer's position, the next observation and the
    environment. Compared by :func:`validate`."""
    sd = agent.policy.state_dict()
    out = {"counters": {"num_timesteps": int(agent.num_timesteps), "n_updates": int(getattr(agent, "_n_updates", 0)), "episodes": int(getattr(agent, "_episode_num", 0))},
           "policy": _digest_tensors(sorted(sd.items())), "optimizers": {}}
    for name, opt in _optimizers(agent).items():
        buf = io.BytesIO()
        torch.save(opt.state_dict(), buf)
        out["optimizers"][name] = hashlib.sha256(buf.getvalue()).hexdigest()
    rb = getattr(agent, "replay_buffer", None)
    if rb is not None:
        out["replay"] = {"size": int(rb.size()), "pos": int(rb.pos), "full": bool(rb.full), "digest": _digest_tensors([("o", torch.as_tensor(rb.observations[: rb.size()])), ("a", torch.as_tensor(rb.actions[: rb.size()])), ("r", torch.as_tensor(rb.rewards[: rb.size()]))])}
    if getattr(agent, "_last_obs", None) is not None:
        out["obs"] = hashlib.sha256(np.ascontiguousarray(agent._last_obs).tobytes()).hexdigest()
    if env is not None:
        out["env"] = env_digest(env)
    return out


# What a disagreement means. A counter or a payload that shapes the next update is BLOCKING (continuing would silently train something else); the others only withdraw the exactness claim.
BLOCKING = ("num_timesteps", "policy", "optimizers", "replay", "obs", "env")


def validate(agent, manifest: dict, env=None) -> dict:
    """Compare a restored agent with the digests recorded when the checkpoint was written. Returns ``{"checks": {name: bool}, "blocking": [..], "downgrade": [..]}``: ``blocking`` must stop the continuation,
    ``downgrade`` caps the guarantee at ``epoch_boundary``. A manifest without digests (an older checkpoint) cannot be validated and is downgraded."""
    saved = manifest.get("digests")
    if not saved:
        return {"checks": {}, "blocking": [], "downgrade": ["the checkpoint carries no digests to validate the restore against"]}
    now = fingerprint_state(agent, env)
    checks, blocking, downgrade = {}, [], []

    def check(name, ok, message, is_blocking):
        checks[name] = bool(ok)
        if not ok:
            (blocking if is_blocking else downgrade).append(message)
    for key in ("num_timesteps", "n_updates", "episodes"):
        check(f"counter:{key}", saved["counters"][key] == now["counters"][key], f"{key}: saved {saved['counters'][key]}, restored {now['counters'][key]}", key == "num_timesteps")
    check("policy", saved["policy"] == now["policy"], "the restored policy parameters differ from the saved ones", True)
    for name in sorted(set(saved["optimizers"]) | set(now["optimizers"])):
        check(f"optimizer:{name}", saved["optimizers"].get(name) == now["optimizers"].get(name), f"the {name} optimizer's state differs from the saved one", True)
    if "replay" in saved or "replay" in now:
        a, b = saved.get("replay"), now.get("replay")
        check("replay_buffer", a == b, f"the replay buffer differs (saved {None if a is None else (a['size'], a['pos'], a['full'])}, restored {None if b is None else (b['size'], b['pos'], b['full'])})", True)
    check("next_observation", saved.get("obs") == now.get("obs"), "the observation the next step acts on differs", True)
    if env is not None and "env" in saved:
        check("environment", saved["env"] == now["env"], "the restored environment differs from the saved one", True)
    return {"checks": checks, "blocking": blocking, "downgrade": downgrade}


def describe_limits(kind: str, device: str) -> list[str]:
    notes = ["a single environment only: a vectorised run's per-environment state is not saved", "learning rate and clip range must be constants (they are, in this platform)", "observation and reward normalisation are not used by this agent, so there is no running statistic to restore"]
    if device not in ("cpu", "auto"):
        notes.append(f"device {device}: GPU kernels and the CUDA generators were not verified")
    if kind == "recurrent_ppo":
        notes.append("the LSTM state is read from a private SB3 attribute (_last_lstm_states)")
    if kind == "sac":
        notes.append("the replay buffer is saved whole: the checkpoint grows with it")
    return notes


def _numpy_state():
    s = np.random.get_state()
    return (s[0], s[1].copy(), int(s[2]), int(s[3]), float(s[4]))


def capture(agent, kind: str) -> dict:
    """Everything needed to rebuild ``agent`` and continue it. Call it between units (after a whole rollout/update for PPO; any step for SAC)."""
    buf = io.BytesIO()
    agent.save(buf)                                                                           # policy, optimizer(s), hyperparameters, timesteps, last observation, episode statistics
    out = {"model_zip": buf.getvalue(), "kind": kind, "versions": versions()}
    if getattr(agent, "replay_buffer", None) is not None:
        rb = io.BytesIO()
        agent.save_replay_buffer(rb)
        out["replay"] = rb.getvalue()
    out["env"] = pickle.dumps(agent.get_env().envs[0])                                         # the wrapped environment, mid-episode
    sp = agent.action_space.np_random
    out["rng"] = {"torch": torch.get_rng_state(), "numpy": _numpy_state(), "python": random.getstate(), "action_space": sp.bit_generator.state}
    lstm = getattr(agent, "_last_lstm_states", None)
    if lstm is not None:
        lb = io.BytesIO()
        torch.save(lstm, lb)
        out["lstm"] = lb.getvalue()
    out["last"] = {"obs": np.array(agent._last_obs, copy=True), "starts": np.array(agent._last_episode_starts, copy=True)}      # the observation the next step acts on (SB3's ``load`` leaves it unset)
    out["counters"] = {"num_timesteps": int(agent.num_timesteps), "n_updates": int(getattr(agent, "_n_updates", 0)), "episodes": int(getattr(agent, "_episode_num", 0))}
    out["digests"] = fingerprint_state(agent, agent.get_env().envs[0])                                      # what "restored exactly" will be checked against
    return out


def restore(algo, manifest: dict, device: str):
    """``(agent, env, verdict)`` rebuilt from a manifest produced by :func:`capture`, with every generator re-applied after SB3's own re-seeding. ``verdict`` is :func:`validate`'s comparison of the rebuilt agent with
    the digests recorded at the checkpoint (before anything is stepped)."""
    env = pickle.loads(manifest["env"])
    agent = algo.load(io.BytesIO(manifest["model_zip"]), env=env, device=device)
    set_clock(agent)
    if "replay" in manifest:
        agent.load_replay_buffer(io.BytesIO(manifest["replay"]))
    if "lstm" in manifest:
        agent._last_lstm_states = torch.load(io.BytesIO(manifest["lstm"]), map_location="cpu", weights_only=False)
    agent._last_obs = np.array(manifest["last"]["obs"], copy=True)
    agent._last_episode_starts = np.array(manifest["last"]["starts"], copy=True)
    apply_rng(agent, manifest["rng"])
    verdict = validate(agent, manifest, agent.get_env().envs[0])
    cur = versions()
    for lib, v in (manifest.get("versions") or {}).items():
        if cur.get(lib) != v:
            verdict["downgrade"].append(f"{lib} {v} wrote the checkpoint, {cur.get(lib)} is reading it")
    return agent, env, verdict


def apply_rng(agent, rng: dict) -> None:
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    agent.action_space.np_random.bit_generator.state = rng["action_space"]
