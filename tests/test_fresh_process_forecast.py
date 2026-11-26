"""Every forecast network (`models.deep_forecast.build_network`: PatchTST, TSMixer, N-BEATS, N-HiTS, TimeMixer, TFT, FNN, CNN, LSTM, GRU, Transformer) on its own: trained for several epochs, the worker killed with SIGKILL
after a committed checkpoint, resumed in a NEW process. For each architecture the final network and optimizer state of the resumed run equal those of an uninterrupted reference (read from the checkpoints on disk),
the resumed run continues from the right epoch with contiguous metrics and gradient diagnostics, and the Training monitor's API shows the resumed run as completed with its resume marker. A damaged newest checkpoint and a
missing checkpoint are handled for a representative network."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import runs                                   # noqa: E402
from tests.fresh_process_scenarios import FORECAST_KINDS, maxdiff, record, was_killed        # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
UNITS = 10


def run(folder, mode, out, kind, *extra):
    cmd = [sys.executable, "-W", "ignore", "-m", "tests.fresh_process_scenarios", f"forecast_{kind}", str(folder), mode, str(out), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fit_dir(folder):
    return next(Path(folder).rglob("metrics.jsonl")).parent


def checkpoint(folder, name="last.pt"):
    return torch.load(io.BytesIO((fit_dir(folder) / name).read_bytes()), map_location="cpu", weights_only=False)


def same(a, b, path="state"):
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


def setup(tmp_path, kind):
    root = tmp_path / "root"
    base = root / "data" / "user" / "training"
    return root, base / "ref", base / "run"


@pytest.mark.parametrize("kind", FORECAST_KINDS)
def test_this_architecture_resumes_in_a_fresh_process_with_its_network_and_optimizer_restored(kind, tmp_path):
    root, ref_dir, run_dir = setup(tmp_path, kind)
    ref = run(ref_dir, "reference", tmp_path / "ref.json", kind)
    assert ref.returncode == 0, ref.stderr[-1500:]
    dead = run(run_dir, "interrupted", tmp_path / "dead.json", kind, "--kill-after", "3")
    assert was_killed(dead.returncode) and not (tmp_path / "dead.json").exists()
    killed = checkpoint(run_dir)
    committed = killed["epoch"] + 1
    assert 3 <= committed < UNITS and killed["model"]["net"] and killed["optimizer"]["adam"]["state"]          # a durable checkpoint holds the network and Adam's moments
    res = run(run_dir, "resume", tmp_path / "res.json", kind)
    assert res.returncode == 0, res.stderr[-1500:]
    a, b = json.loads((tmp_path / "ref.json").read_text()), json.loads((tmp_path / "res.json").read_text())
    np.testing.assert_allclose(b["loss"], a["loss"], rtol=1e-6, atol=1e-8)                                         # validation loss history
    np.testing.assert_allclose(b["train_loss"], a["train_loss"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(b["outputs"], a["outputs"], rtol=1e-6, atol=1e-7)                                   # test-window predictions of the best network
    fa, fb = checkpoint(ref_dir), checkpoint(run_dir)
    assert fa["epoch"] == fb["epoch"]
    for key in ("model", "optimizer"):
        same(fa[key], fb[key], key)                                                                               # network and Adam state
    same(fa["rng"]["torch"], fb["rng"]["torch"], "rng.torch")                                                      # the generators the loop draws from: PyTorch's and the shuffling generator
    same(fa["rng"]["gens"], fb["rng"]["gens"], "rng.gens")                                                         # (NumPy's global generator is never drawn from here: each process starts it from its own entropy)
    log = rows(fit_dir(run_dir) / "metrics.jsonl")
    epochs = [r["epoch"] for r in log if r["type"] == "epoch"]
    assert epochs == list(range(1, len(epochs) + 1)) and log[-1]["type"] == "finish"
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["level"] == "exact" and resume["epoch"] >= committed - 0 and resume["epoch"] <= committed + 1       # it continued from the committed epoch, not from zero
    grads = [r["epoch"] for r in rows(fit_dir(run_dir) / "gradients.jsonl")]
    assert grads == list(range(len(grads)))
    record(family="forecast_network", scenario=f"forecast_{kind}", killed_after_units=committed, resumed_from_unit=resume["epoch"], units_total=len(epochs), recovery_level=resume["level"], max_abs_diff_val_loss=maxdiff(a["loss"], b["loss"]),
           max_abs_diff_outputs=maxdiff(a["outputs"], b["outputs"]))
    # the Training monitor's view of the recovered run
    d = runs.detail(root, "run")
    assert d["summary"]["status"] in ("completed", "stopped_early") and d["series"]["epoch"] == epochs
    assert any(e["type"] == "resume" and e["level"] == "exact" for e in d["events"]) and d["series"]["train_loss"][-1] == pytest.approx(a["train_loss"][len(epochs) - 1], rel=1e-6)
    g = runs.gradients(root, "run")
    assert g["optimizers"] == ["adam"] and g["epochs"] == list(range(len(grads))) and g["series"]["l2_median"][0] > 0 and g["flow"]["rows"]


def test_a_damaged_newest_checkpoint_and_a_missing_checkpoint_are_handled_for_a_forecast_network(tmp_path):
    kind = "lstm"
    _, ref_dir, run_dir = setup(tmp_path, kind)
    assert run(ref_dir, "reference", tmp_path / "ref.json", kind).returncode == 0
    assert was_killed(run(run_dir, "interrupted", tmp_path / "dead.json", kind, "--kill-after", "4").returncode)
    fit = fit_dir(run_dir)
    (fit / "last.pt").write_bytes((fit / "last.pt").read_bytes()[:300])                                           # damaged
    res = run(run_dir, "resume", tmp_path / "res.json", kind)
    assert res.returncode == 0, res.stderr[-1500:]
    log = rows(fit / "metrics.jsonl")
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["file"] == "last.prev.pt" and resume["level"] == "epoch_boundary" and any("damaged" in x for x in resume["reasons"])
    assert [r["epoch"] for r in log if r["type"] == "epoch"] == list(range(1, UNITS + 1))
    assert np.isfinite(json.loads((tmp_path / "res.json").read_text())["outputs"]).all()
    # no checkpoint at all: a resume starts the run from the beginning, and says nothing was restored
    fresh = tmp_path / "root" / "data" / "user" / "training" / "empty"
    out = run(fresh, "resume", tmp_path / "res3.json", kind)
    assert out.returncode == 0, out.stderr[-1500:]
    log = rows(fit_dir(fresh) / "metrics.jsonl")
    assert not any(r["type"] == "resume" for r in log) and [r["epoch"] for r in log if r["type"] == "epoch"] == list(range(1, UNITS + 1))
