"""Double DQN, on its own: a training process is killed with SIGKILL after a committed checkpoint, a NEW process restores it from disk and continues, and the continuation is compared with an uninterrupted
reference process: online and target Q-networks, Adam state (the number of updates), the replay buffer, the environment's state, the generators, the exploration schedule, the target-synchronisation counter and the
per-unit metrics. Deterministic continuation is demonstrated here because the environment's state is saved with the agent; for an environment that cannot save its state the level would be ``epoch_boundary``."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import integrity  # noqa: E402

from tests.fresh_process_scenarios import maxdiff, record, was_killed   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEPS, BLOCK, WARMUP, TARGET_EVERY = 2000, 250, 100, 50


def run(folder, mode, out, *extra):
    cmd = [sys.executable, "-W", "ignore", "-m", "tests.fresh_process_scenarios", "rl_double_dqn", str(folder), mode, str(out), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint(folder, name="last.pt"):
    return torch.load(io.BytesIO((next(folder.rglob("metrics.jsonl")).parent / name).read_bytes()), map_location="cpu", weights_only=False)


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    else:
        assert a == b


def test_double_dqn_killed_after_a_checkpoint_continues_in_a_new_process_exactly(tmp_path):
    ref = run(tmp_path / "ref", "reference", tmp_path / "ref.json")
    assert ref.returncode == 0, ref.stderr[-1500:]
    dead = run(tmp_path / "run", "interrupted", tmp_path / "dead.json", "--kill-after", "4")
    assert was_killed(dead.returncode) and not (tmp_path / "dead.json").exists()
    killed = checkpoint(tmp_path / "run")
    k = killed["epoch"] + 1                                                                                         # units committed when the process died
    assert 4 <= k < 8
    st = killed["history"]["extra"]["S"]
    assert killed["history"]["extra"]["buf"]["n"] == min(k * BLOCK, 50000) and st["updates"] == k * BLOCK - WARMUP + 1           # replay size and update count at the checkpoint
    assert st["syncs"] == len([t for t in range(WARMUP - 1, k * BLOCK) if t % TARGET_EVERY == 0])                    # the target network was synchronised on schedule
    assert killed["optimizer"]["adam"]["state"][0]["step"] == st["updates"]                                          # Adam's step counter is the number of updates
    assert killed["history"]["extra"]["env"] is not None and killed["rng"]["gens"]["rng"]["type"] == "numpy"
    res = run(tmp_path / "run", "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-1500:]
    a, b = json.loads((tmp_path / "ref.json").read_text()), json.loads((tmp_path / "res.json").read_text())
    np.testing.assert_allclose(b["params"], a["params"], rtol=0, atol=0)                                             # both Q-networks' weights: bit for bit
    np.testing.assert_allclose(b["outputs"], a["outputs"], rtol=0, atol=0)                                           # the greedy policy in every regime
    # the whole final training state, from the two checkpoints on disk (the reference never saw an interruption)
    fa, fb = checkpoint(tmp_path / "ref"), checkpoint(tmp_path / "run")
    assert fa["epoch"] == fb["epoch"] == 7
    for key in ("model", "optimizer", "scheduler"):
        same(fa[key], fb[key])
    same(fa["history"]["extra"], fb["history"]["extra"])                                                              # replay buffer, current observation, environment state, counters
    same(fa["rng"]["gens"], fb["rng"]["gens"])
    assert torch.equal(torch.tensor(list(bytes.fromhex(fa["rng"]["torch"]))), torch.tensor(list(bytes.fromhex(fb["rng"]["torch"]))))
    final = fb["history"]["extra"]["S"]
    assert final["updates"] == STEPS - WARMUP + 1 and final["syncs"] == len([t for t in range(WARMUP - 1, STEPS) if t % TARGET_EVERY == 0])
    # the per-unit training record: exploration schedule, updates, syncs, rewards identical to the reference's, in order, with the resume marked
    mr = [r for r in rows(next((tmp_path / "ref").rglob("metrics.jsonl"))) if r["type"] == "epoch"]
    mb_log = rows(next((tmp_path / "run").rglob("metrics.jsonl")))
    mb = [r for r in mb_log if r["type"] == "epoch"]
    assert [r["epoch"] for r in mb] == list(range(1, 9))
    for x, y in zip(mr, mb):
        for key in ("epsilon", "updates", "target_syncs", "env_steps", "replay_size", "mean_reward"):
            assert x["metrics"][key] == pytest.approx(y["metrics"][key], rel=0, abs=0), (x["epoch"], key)
        assert x["train_loss"] == y["train_loss"]
    eps0, eps1 = 1.0, 0.05
    resume = [r for r in mb_log if r["type"] == "resume"][-1]
    assert resume["level"] == "exact" and resume["epoch"] == k
    record(family="double_dqn", scenario="rl_double_dqn", killed_after_units=k, resumed_from_unit=resume["epoch"], units_total=8, recovery_level=resume["level"], max_abs_diff_q_params=maxdiff(a["params"], b["params"]), updates=final["updates"], target_syncs=final["syncs"])
    assert mb[-1]["metrics"]["epsilon"] == pytest.approx(eps0 + (eps1 - eps0) * min(1.0, (STEPS - 1) / (0.7 * STEPS)))      # the schedule continued where it stopped


def test_a_corrupted_newest_checkpoint_falls_back_to_the_previous_one_and_is_reported_as_approximate(tmp_path):
    """The previous checkpoint is a complete, older state, but the platform's contract is conservative: continuing from a fallback is reported as ``epoch_boundary``, the generators are not restored, and the
    continuation is approximate RL training (the same schedule, counters and replay history, but a different random stream), not the reference's trajectory."""
    ref = run(tmp_path / "ref", "reference", tmp_path / "ref.json")
    dead = run(tmp_path / "run", "interrupted", tmp_path / "dead.json", "--kill-after", "4")
    assert ref.returncode == 0 and was_killed(dead.returncode)
    fit = next((tmp_path / "run").rglob("last.pt")).parent
    assert (fit / "last.prev.pt").exists()
    (fit / "last.pt").write_bytes((fit / "last.pt").read_bytes()[:200])                                              # the newest checkpoint is damaged
    res = run(tmp_path / "run", "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-1500:]
    log = rows(fit / "metrics.jsonl")
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["file"] == "last.prev.pt" and resume["level"] == "epoch_boundary" and any("damaged" in x for x in resume["reasons"])     # honest: not claimed exact
    a, b = json.loads((tmp_path / "ref.json").read_text()), json.loads((tmp_path / "res.json").read_text())
    assert not np.allclose(b["params"], a["params"], rtol=0, atol=1e-9)                                              # a different random stream: the trajectory is not the reference's
    assert np.isfinite(b["params"]).all() and np.isfinite(b["outputs"]).all()
    epochs = [r for r in log if r["type"] == "epoch"]
    assert [r["epoch"] for r in epochs] == list(range(1, 9))
    final = epochs[-1]["metrics"]
    assert final["updates"] == STEPS - WARMUP + 1 and final["target_syncs"] == len([t for t in range(WARMUP - 1, STEPS) if t % TARGET_EVERY == 0]) and final["replay_size"] == STEPS      # the counters and the schedule are the same
    assert final["epsilon"] == pytest.approx(1.0 + (0.05 - 1.0) * min(1.0, (STEPS - 1) / (0.7 * STEPS)))
    ck = checkpoint(tmp_path / "run")
    assert ck["epoch"] == 7 and ck["history"]["extra"]["S"]["updates"] == STEPS - WARMUP + 1


def test_with_no_checkpoint_a_resume_starts_fresh_and_a_changed_environment_is_refused(tmp_path):
    first = run(tmp_path / "run", "resume", tmp_path / "res.json")                                                    # resume requested, nothing on disk
    assert first.returncode == 0, first.stderr[-1500:]
    log = rows(next((tmp_path / "run").rglob("metrics.jsonl")))
    assert not any(r["type"] == "resume" for r in log) and [r["epoch"] for r in log if r["type"] == "epoch"] == list(range(1, 9))
    # a checkpoint of another problem (here: another seed's fingerprint) is refused outright
    fit = next((tmp_path / "run").rglob("last.pt")).parent
    blob = torch.load(io.BytesIO((fit / "last.pt").read_bytes()), map_location="cpu", weights_only=False)
    blob["config_fp"] = "someone-elses-config"
    (fit / "metrics.jsonl").unlink()
    torch.save(blob, fit / "last.pt")
    (fit / "last.pt.sha256").unlink(missing_ok=True)
    (fit / "last.pt.hmac").write_text(integrity.sign((fit / "last.pt").read_bytes()))                    # signed by this installation, so it is read, and refused because it is another problem's
    (fit / "last.prev.pt").unlink(missing_ok=True)
    bad = run(tmp_path / "run", "resume", tmp_path / "res2.json")
    assert bad.returncode != 0 and "CheckpointIncompatible" in bad.stderr
