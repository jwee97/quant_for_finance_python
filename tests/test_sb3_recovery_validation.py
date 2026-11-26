"""Stable-Baselines3 recovery is only called exact when the restored agent has been checked against what was saved.

* **Schedules**: a constant schedule and a linear one (learning rate and PPO clip range) are tested separately. Each trained in checkpointed blocks must equal one plain ``learn`` call bit for bit, and a killed and
  resumed run must equal the uninterrupted one. (Without the global clock of ``sb3_state.GlobalSchedule`` a linear schedule would restart in every block.)
* **Validation**: the checkpoint records digests of the counters, the policy, every optimizer, the replay buffer, the next observation and the environment. A resume rebuilds the agent and compares. A counter that
  shapes training (timesteps), the policy, an optimizer, the replay buffer, the observation or the environment that disagrees BLOCKS the continuation; the update and episode counters, a missing digest block or another
  library version only lower the guarantee to ``epoch_boundary``. These tests tamper with a real checkpoint and resume it in a new process.
* **Ceiling**: configurations nobody verified (a GPU, vectorised environments, ``VecNormalize``, a block that splits a rollout, an unverified library version) cap the guarantee; the cap logic is tested with stand-in agents.
  That the cap is applied is verified; that these configurations are exact or inexact is NOT: there is no direct evidence either way.
* **Private API**: every underscore member of SB3 that the module reads exists on the installed version (a canary for a library update).

They need stable-baselines3 and sb3-contrib: skipped as dependency-blocked without ``$QUANT_SB3_PYTHON`` or ``/tmp/sb3env`` (``scripts/make_sb3_env.sh``)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep.checkpoint import CheckpointManager                      # noqa: E402
from src.ml.rl import sb3_state                                           # noqa: E402
from tests.fresh_process_scenarios import was_killed                       # noqa: E402
from tests.test_fresh_process_sb3 import PY, blob, fit_dir, rows, run     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
needs_sb3 = pytest.mark.skipif(PY is None, reason="dependency-blocked: stable-baselines3 and sb3-contrib are not installed (scripts/make_sb3_env.sh)")


# ------------------------------------------------------------------------------------------------------------------------------------------ schedules
@needs_sb3
@pytest.mark.parametrize("name", ["sb3_ppo_constant", "sb3_ppo_linear", "sb3_sac_linear"])
def test_a_schedule_trained_in_blocks_equals_one_call_and_survives_a_kill(name, tmp_path):
    plain = run(name, tmp_path / "plain", "plain", tmp_path / "plain.json")
    ref = run(name, tmp_path / "ref", "reference", tmp_path / "ref.json")
    dead = run(name, tmp_path / "run", "interrupted", tmp_path / "dead.json", "--kill-after", "3")
    assert plain.returncode == 0 and ref.returncode == 0, (plain.stderr[-800:], ref.stderr[-800:])
    assert was_killed(dead.returncode)
    res = run(name, tmp_path / "run", "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-1200:]
    a, b, c = (json.loads((tmp_path / f).read_text()) for f in ("plain.json", "ref.json", "res.json"))
    np.testing.assert_array_equal(a["params"], b["params"])                                   # blocks == one plain call
    np.testing.assert_array_equal(c["params"], b["params"])                                   # kill + resume == uninterrupted
    np.testing.assert_array_equal(c["loss"], b["loss"])
    log = rows(fit_dir(tmp_path / "run") / "metrics.jsonl")
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["level"] == "exact"
    checks = [r for r in log if r["type"] == "recovery_checks"][-1]
    assert checks["blocking"] == [] and checks["downgrade"] == [] and checks["checks"] and all(checks["checks"].values()), checks
    limits = [r for r in log if r["type"] == "recovery_limits"][0]
    assert limits["exactness_blocked_by"] == [] and "_last_obs" in limits["private_sb3_members_used"]


@needs_sb3
def test_a_linear_schedule_really_changes_what_is_trained(tmp_path):
    out = {}
    for name in ("sb3_ppo_constant", "sb3_ppo_linear"):
        assert run(name, tmp_path / name, "plain", tmp_path / f"{name}.json").returncode == 0
        out[name] = np.array(json.loads((tmp_path / f"{name}.json").read_text())["params"])
    assert not np.array_equal(out["sb3_ppo_constant"], out["sb3_ppo_linear"]), "the schedule must have an effect, or the equality above proves nothing"


# ------------------------------------------------------------------------------------------------------------------------------------------ validation
def tamper(run_dir, edit):
    """Rewrite the newest checkpoint with ``edit(manifest)`` applied to the SB3 manifest (and re-seal it with a new checksum)."""
    fit = fit_dir(run_dir)
    b = blob(run_dir)
    edit(b["history"]["extra"]["sb3"])
    keep = {k: b.pop(k) for k in ("schema", "saved_at", "versions") if k in b}
    for f in fit.glob("*.pt*"):
        f.unlink()
    CheckpointManager(fit).save(b)
    return b


def flip(h):
    return ("0" if h[0] != "0" else "1") + h[1:]


BLOCKING = {
    "timesteps": ("sb3_ppo", lambda m: m["digests"]["counters"].__setitem__("num_timesteps", m["digests"]["counters"]["num_timesteps"] + 64), "num_timesteps"),
    "policy": ("sb3_ppo", lambda m: m["digests"].__setitem__("policy", flip(m["digests"]["policy"])), "policy parameters"),
    "optimizer": ("sb3_ppo", lambda m: m["digests"]["optimizers"].__setitem__("policy", flip(m["digests"]["optimizers"]["policy"])), "optimizer"),
    "replay_buffer": ("sb3_sac", lambda m: m["digests"]["replay"].__setitem__("pos", m["digests"]["replay"]["pos"] + 1), "replay buffer"),
    "observation": ("sb3_ppo", lambda m: m["digests"].__setitem__("obs", flip(m["digests"]["obs"])), "observation"),
    "environment": ("sb3_ppo", lambda m: m["digests"].__setitem__("env", flip(m["digests"]["env"])), "environment"),
}


@needs_sb3
@pytest.mark.parametrize("what", sorted(BLOCKING))
def test_a_disagreement_that_shapes_training_blocks_the_continuation(what, tmp_path):
    name, edit, phrase = BLOCKING[what]
    run_dir = tmp_path / "run"
    assert was_killed(run(name, run_dir, "interrupted", tmp_path / "dead.json", "--kill-after", "3").returncode)
    tamper(run_dir, edit)
    res = run(name, run_dir, "resume", tmp_path / "res.json")
    assert res.returncode != 0 and "CheckpointIncompatible" in res.stderr and "does not match the checkpoint" in res.stderr and phrase in res.stderr, res.stderr[-900:]
    log = rows(fit_dir(run_dir) / "metrics.jsonl")
    assert log[-1]["type"] == "interrupted" and not any(r["type"] == "epoch" and r["epoch"] > 3 and not r.get("recovered") for r in log), "nothing was trained on a state that failed validation"
    assert not any(r["type"] == "resume" for r in log[-3:]), "no resume row for a continuation that never happened"


DOWNGRADE = {
    "updates counter": lambda m: m["digests"]["counters"].__setitem__("n_updates", m["digests"]["counters"]["n_updates"] + 1),
    "episodes counter": lambda m: m["digests"]["counters"].__setitem__("episodes", m["digests"]["counters"]["episodes"] + 1),
    "no digests": lambda m: m.pop("digests"),
    "other library version": lambda m: m["versions"].__setitem__("stable_baselines3", "0.0.1"),
}


@needs_sb3
@pytest.mark.parametrize("what", sorted(DOWNGRADE))
def test_a_disagreement_that_only_weakens_the_claim_lowers_the_guarantee_and_says_why(what, tmp_path):
    name = "sb3_ppo"
    run_dir = tmp_path / "run"
    assert was_killed(run(name, run_dir, "interrupted", tmp_path / "dead.json", "--kill-after", "3").returncode)
    tamper(run_dir, DOWNGRADE[what])
    res = run(name, run_dir, "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-900:]
    log = rows(fit_dir(run_dir) / "metrics.jsonl")
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["level"] == "epoch_boundary", resume
    text = " ".join(resume["reasons"])
    assert {"updates counter": "n_updates", "episodes counter": "episodes", "no digests": "no digests", "other library version": "0.0.1"}[what] in text, resume["reasons"]
    assert [r["epoch"] for r in log if r["type"] == "epoch"] == list(range(1, 11)) and log[-1]["type"] == "finish"


# ------------------------------------------------------------------------------------------------------------------------------------------ ceiling (stand-in agents)
def fake(**kw):
    base = dict(device=SimpleNamespace(type="cpu"), n_envs=1, get_vec_normalize_env=lambda: None, _last_obs=np.zeros(1), _last_episode_starts=np.zeros(1), n_steps=64, train_freq=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_cap_is_applied_for_every_configuration_that_was_not_verified(monkeypatch):
    monkeypatch.setattr(sb3_state, "versions", lambda: {"stable_baselines3": "2.9.2", "sb3_contrib": "2.9.2"})
    assert sb3_state.ceiling_reasons(fake(), "ppo", "cpu", 128) == []
    cases = {
        "GPU": (fake(device=SimpleNamespace(type="cuda")), "ppo", "cuda", 128, "GPU"),
        "vectorised": (fake(n_envs=4), "ppo", "cpu", 128, "vectorised"),
        "VecNormalize": (fake(get_vec_normalize_env=lambda: object()), "ppo", "cpu", 128, "VecNormalize"),
        "no _last_obs": (SimpleNamespace(**{k: v for k, v in vars(fake()).items() if k != "_last_obs"}), "ppo", "cpu", 128, "_last_obs"),
        "no lstm state": (fake(), "recurrent_ppo", "cpu", 128, "_last_lstm_states"),
        "block splits a rollout": (fake(), "ppo", "cpu", 100, "whole number"),
        "train_freq in episodes": (fake(train_freq=SimpleNamespace(frequency=1, unit="episode")), "sac", "cpu", 100, "episodes"),
        "train_freq not dividing the block": (fake(train_freq=SimpleNamespace(frequency=8, unit="step")), "sac", "cpu", 100, "does not divide"),
    }
    for label, (agent, kind, device, chunk, phrase) in cases.items():
        reasons = sb3_state.ceiling_reasons(agent, kind, device, chunk)
        assert reasons and any(phrase in r for r in reasons), (label, reasons)
    monkeypatch.setattr(sb3_state, "versions", lambda: {"stable_baselines3": "3.0.0"})
    assert any("3.0.0" in r and "not verified" in r for r in sb3_state.ceiling_reasons(fake(), "ppo", "cpu", 128))


def test_validate_names_each_kind_of_disagreement_and_never_passes_a_missing_digest_block():
    torch.manual_seed(0)
    policy = torch.nn.Linear(2, 2)
    agent = SimpleNamespace(policy=policy, num_timesteps=128, _n_updates=4, _episode_num=2, _last_obs=np.ones((1, 3)), replay_buffer=None)
    saved = {"digests": sb3_state.fingerprint_state(agent)}
    ok = sb3_state.validate(agent, saved)
    assert ok["blocking"] == [] and ok["downgrade"] == [] and all(ok["checks"].values())
    agent.num_timesteps = 129
    assert any("num_timesteps" in m for m in sb3_state.validate(agent, saved)["blocking"])
    agent.num_timesteps, agent._n_updates = 128, 5
    v = sb3_state.validate(agent, saved)
    assert v["blocking"] == [] and any("n_updates" in m for m in v["downgrade"])
    agent._n_updates = 4
    with torch.no_grad():
        policy.weight += 1e-6
    assert any("policy" in m for m in sb3_state.validate(agent, saved)["blocking"])
    assert sb3_state.validate(agent, {})["downgrade"] == ["the checkpoint carries no digests to validate the restore against"]


# ------------------------------------------------------------------------------------------------------------------------------------------ private API canary
@needs_sb3
def test_every_private_sb3_member_the_module_reads_exists_on_the_installed_version():
    code = r"""
import json, sys
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from src.ml.rl import sb3_state
out = {}
for name, make, env in (("ppo", lambda e: PPO("MlpPolicy", e, n_steps=32, batch_size=32, verbose=0, device="cpu"), "CartPole-v1"),
                        ("sac", lambda e: SAC("MlpPolicy", e, learning_starts=10, buffer_size=200, verbose=0, device="cpu"), "Pendulum-v1"),
                        ("recurrent_ppo", lambda e: RecurrentPPO("MlpLstmPolicy", e, n_steps=32, batch_size=32, verbose=0, device="cpu"), "CartPole-v1")):
    agent = make(env)
    agent.learn(64)
    out[name] = {m: hasattr(agent, m) for m in sb3_state.PRIVATE_USED}
print(json.dumps(out))
"""
    res = subprocess.run([PY, "-W", "ignore", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-800:]
    got = json.loads(res.stdout.strip().splitlines()[-1])
    for kind, members in got.items():
        for m, present in members.items():
            if m == "_last_lstm_states" and kind != "recurrent_ppo":
                continue
            assert present, f"{kind}: SB3 no longer has {m}; sb3_state.PRIVATE_USED documents why it is read and the cap {sb3_state.PRIVATE_USED[m]}"
