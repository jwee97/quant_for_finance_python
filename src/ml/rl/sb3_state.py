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

import io
import pickle
import random

import numpy as np
import torch

ALGOS = ("ppo", "sac", "recurrent_ppo")


def versions() -> dict:
    out = {}
    for name in ("stable_baselines3", "sb3_contrib", "gymnasium"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "?"))
        except ImportError:
            pass
    return out


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
    return out


def restore(algo, manifest: dict, device: str):
    """``(agent, env)`` rebuilt from a manifest produced by :func:`capture`, with every generator re-applied after SB3's own re-seeding."""
    env = pickle.loads(manifest["env"])
    agent = algo.load(io.BytesIO(manifest["model_zip"]), env=env, device=device)
    if "replay" in manifest:
        agent.load_replay_buffer(io.BytesIO(manifest["replay"]))
    if "lstm" in manifest:
        agent._last_lstm_states = torch.load(io.BytesIO(manifest["lstm"]), map_location="cpu", weights_only=False)
    agent._last_obs = np.array(manifest["last"]["obs"], copy=True)
    agent._last_episode_starts = np.array(manifest["last"]["starts"], copy=True)
    apply_rng(agent, manifest["rng"])
    return agent, env


def apply_rng(agent, rng: dict) -> None:
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    agent.action_space.np_random.bit_generator.state = rng["action_space"]


def check_counters(agent, manifest: dict) -> list[str]:
    """Differences between what was saved and what the rebuilt agent reports: an empty list means the counters agree."""
    c, bad = manifest["counters"], []
    for key, now in (("num_timesteps", int(agent.num_timesteps)), ("n_updates", int(getattr(agent, "_n_updates", 0))), ("episodes", int(getattr(agent, "_episode_num", 0)))):
        if c[key] != now:
            bad.append(f"{key}: saved {c[key]}, restored {now}")
    return bad
