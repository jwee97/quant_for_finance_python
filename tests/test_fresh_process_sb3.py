"""Stable-Baselines3 agents (PPO, SAC, recurrent PPO) behind the platform's model interface, killed with SIGKILL and resumed in NEW processes. They need stable-baselines3, gymnasium and sb3-contrib, which the main
environment does not install: the tests run the scenarios with the Python found in ``$QUANT_SB3_PYTHON`` or ``/tmp/sb3env/bin/python`` (``scripts/make_sb3_env.sh`` creates it) and are skipped, as dependency-blocked,
when there is none.

What is checked, per agent: chunking the training into checkpointed blocks does not change what is trained (bit for bit against a plain ``learn`` call); a worker killed after a committed block is resumed in a new process
and ends with the same policy, the same reward history and the same final training state as an uninterrupted run (so the random streams, the environment and, for SAC, the replay buffer, for the recurrent agent the LSTM
state were all restored); the logs and gradient diagnostics are contiguous; a damaged newest checkpoint is reported as approximate; a checkpoint of another configuration is refused."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.fresh_process_scenarios import maxdiff, record, was_killed   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [os.environ.get("QUANT_SB3_PYTHON"), "/tmp/sb3env/bin/python", sys.executable]


def sb3_python():
    for exe in CANDIDATES:
        if exe and Path(exe).exists():
            ok = subprocess.run([exe, "-c", "import stable_baselines3, sb3_contrib, gymnasium"], capture_output=True).returncode == 0
            if ok:
                return exe
    return None


PY = sb3_python()
pytestmark = pytest.mark.skipif(PY is None, reason="dependency-blocked: stable-baselines3, sb3-contrib and gymnasium are not installed (scripts/make_sb3_env.sh builds an isolated environment)")
UNITS = {"sb3_ppo": 10, "sb3_sac": 10, "sb3_recurrent_ppo": 10}


def run(name, folder, mode, out, *extra):
    cmd = [PY, "-W", "ignore", "-m", "tests.fresh_process_scenarios", name, str(folder), mode, str(out), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fit_dir(folder):
    return next(Path(folder).rglob("metrics.jsonl")).parent


def blob(folder, name="last.pt"):
    return torch.load(io.BytesIO((fit_dir(folder) / name).read_bytes()), map_location="cpu", weights_only=False)


def same(a, b, path="x"):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    else:
        assert a == b, path


def unzip(raw):
    """The members of an SB3 zip: policy and optimizer parameters, as tensors."""
    import zipfile
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return {n: torch.load(io.BytesIO(z.read(n)), map_location="cpu", weights_only=False) for n in z.namelist() if n.endswith(".pth")}


@pytest.mark.parametrize("name", sorted(UNITS))
def test_chunked_checkpointed_training_equals_a_plain_learn_call(name, tmp_path):
    plain = run(name, tmp_path / "plain", "plain", tmp_path / "plain.json")
    ref = run(name, tmp_path / "ref", "reference", tmp_path / "ref.json")
    assert plain.returncode == 0 and ref.returncode == 0, (plain.stderr[-800:], ref.stderr[-800:])
    a, b = json.loads((tmp_path / "plain.json").read_text()), json.loads((tmp_path / "ref.json").read_text())
    np.testing.assert_array_equal(a["params"], b["params"])                                                        # the policy: bit for bit
    np.testing.assert_array_equal(a["loss"], b["loss"])                                                            # the reward history
    assert a["extra"]["num_timesteps"] == b["extra"]["num_timesteps"]


@pytest.mark.parametrize("name", sorted(UNITS))
def test_a_killed_agent_resumes_in_a_new_process_to_the_uninterrupted_result(name, tmp_path):
    ref_dir, run_dir = tmp_path / "ref", tmp_path / "run"
    ref = run(name, ref_dir, "reference", tmp_path / "ref.json")
    assert ref.returncode == 0, ref.stderr[-1500:]
    dead = run(name, run_dir, "interrupted", tmp_path / "dead.json", "--kill-after", "3")
    assert was_killed(dead.returncode) and not (tmp_path / "dead.json").exists(), (dead.returncode, dead.stderr[-600:])
    killed = blob(run_dir)
    committed = killed["epoch"] + 1
    man = killed["history"]["extra"]["sb3"]
    assert 3 <= committed < UNITS[name] and man["counters"]["num_timesteps"] > 0 and man["kind"] == name[4:]
    assert ("replay" in man) == (name == "sb3_sac") and ("lstm" in man) == (name == "sb3_recurrent_ppo")           # what each algorithm needs beyond the model: a replay buffer, a recurrent state
    res = run(name, run_dir, "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-1500:]
    a, b = json.loads((tmp_path / "ref.json").read_text()), json.loads((tmp_path / "res.json").read_text())
    np.testing.assert_array_equal(b["params"], a["params"])                                                        # policy weights, bit for bit
    np.testing.assert_array_equal(b["loss"], a["loss"])
    np.testing.assert_array_equal(b["outputs"], a["outputs"])                                                      # the trained policy played over the training window
    assert b["extra"]["timesteps"] == a["extra"]["timesteps"] and b["extra"]["num_timesteps"] == a["extra"]["num_timesteps"]
    # the final training state on disk: policy and optimizer parameters, counters, generators, environment
    fa, fb = blob(ref_dir), blob(run_dir)
    ma, mb = fa["history"]["extra"]["sb3"], fb["history"]["extra"]["sb3"]
    za, zb = unzip(ma["model_zip"]), unzip(mb["model_zip"])
    assert za.keys() == zb.keys()
    for k in za:
        same(za[k], zb[k], k)                                                                                      # every state dict SB3 saves: policy and each optimizer
    same(ma["counters"], mb["counters"], "counters")
    same(ma["rng"]["action_space"], mb["rng"]["action_space"], "action_space")
    same(ma["rng"]["torch"], mb["rng"]["torch"], "torch")
    out = subprocess.run([PY, "-W", "ignore", "-m", "tests.sb3_compare", str(fit_dir(ref_dir) / "last.pt"), str(fit_dir(run_dir) / "last.pt")], cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == "same", (out.stdout, out.stderr[-500:])                                              # the environment, mid-episode: ledger, clock, generator, episode rewards
    if "replay" in ma:
        assert ma["replay"] == mb["replay"]
    # logs and diagnostics
    log = rows(fit_dir(run_dir) / "metrics.jsonl")
    epochs = [r for r in log if r["type"] == "epoch"]
    assert [r["epoch"] for r in epochs] == list(range(1, UNITS[name] + 1)) and log[-1]["type"] == "finish"
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["level"] == "exact" and resume["epoch"] == committed and all(r["unit"] == "timestep block" for r in epochs)
    assert [r["metrics"]["timesteps"] for r in epochs] == [r["metrics"]["timesteps"] for r in rows(fit_dir(ref_dir) / "metrics.jsonl") if r["type"] == "epoch"]
    g = rows(fit_dir(run_dir) / "gradients.jsonl")
    by = {}
    for r in g:
        by.setdefault(r["optimizer"], []).append(r["epoch"])
    record(family="sb3", scenario=name, killed_after_units=committed, resumed_from_unit=resume["epoch"], units_total=len(epochs), recovery_level=resume["level"], max_abs_diff_policy=maxdiff(a["params"], b["params"]),
           max_abs_diff_rewards=maxdiff(a["loss"], b["loss"]), timesteps=b["extra"]["num_timesteps"], replay_buffer_saved="replay" in ma, lstm_state_saved="lstm" in ma)
    assert by and all(eps == sorted(set(eps)) for eps in by.values()), by                                          # no unit twice; units without an update are simply absent
    assert {"sac": {"actor", "critic"}, "sb3_ppo": {"policy"}, "sb3_recurrent_ppo": {"policy"}}.get(name, {"actor", "critic"}) >= set(by) or name == "sb3_sac"


def test_a_damaged_newest_checkpoint_is_reported_as_approximate_and_another_configuration_is_refused(tmp_path):
    name = "sb3_ppo"
    run_dir = tmp_path / "run"
    assert was_killed(run(name, run_dir, "interrupted", tmp_path / "dead.json", "--kill-after", "4").returncode)
    fit = fit_dir(run_dir)
    keep = (fit / "last.pt").read_bytes()
    (fit / "last.pt").write_bytes(keep[:300])
    res = run(name, run_dir, "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-1500:]
    resume = [r for r in rows(fit / "metrics.jsonl") if r["type"] == "resume"][-1]
    assert resume["file"] == "last.prev.pt" and resume["level"] == "epoch_boundary" and any("damaged" in x for x in resume["reasons"])
    assert [r["epoch"] for r in rows(fit / "metrics.jsonl") if r["type"] == "epoch"] == list(range(1, 11))
    b = blob(run_dir)
    b["config_fp"] = "another-configuration"
    for f in fit.glob("*.pt*"):
        f.unlink()
    torch.save(b, fit / "last.pt")
    bad = run(name, run_dir, "resume", tmp_path / "res2.json")
    assert bad.returncode != 0 and "CheckpointIncompatible" in bad.stderr


def test_with_no_checkpoint_a_resume_trains_from_the_start(tmp_path):
    out = run("sb3_ppo", tmp_path / "fresh", "resume", tmp_path / "res.json")
    assert out.returncode == 0, out.stderr[-1500:]
    log = rows(fit_dir(tmp_path / "fresh") / "metrics.jsonl")
    assert not any(r["type"] == "resume" for r in log) and [r["epoch"] for r in log if r["type"] == "epoch"] == list(range(1, 11))
