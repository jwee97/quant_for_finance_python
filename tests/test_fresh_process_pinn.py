"""Physics-informed networks killed and resumed in fresh processes at every phase of the optimisation: during Adam, immediately before the switch to L-BFGS, inside an L-BFGS closure evaluation (a line search is
running), immediately after a completed L-BFGS step, and after the polish finished but before the run was closed. Each of the four pricing networks (Black-Scholes-Merton, Vasicek, Heston, Bates) is tested on its own
with the configuration of its accuracy test, and the recovered network is held to that test's pricing tolerance against the independent reference (closed form, or the Fourier price)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from src.derivatives.pricing import bsm_price, heston_price     # noqa: E402
from src.models import pinn                                      # noqa: E402
from tests.fresh_process_scenarios import record, maxdiff        # noqa: E402

from tests.fresh_process_scenarios import was_killed   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ADAM_UNITS = {"pinn_bsm": 6, "pinn_vasicek": 8, "pinn_heston": 5, "pinn_bates": 5}


def run(name, folder, mode, out, *extra):
    cmd = [sys.executable, "-W", "ignore", "-m", "tests.fresh_process_scenarios", name, str(folder), mode, str(out), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def reference_error(name, prices):
    p = np.asarray(prices)
    if name == "pinn_bsm":
        S = np.array([80, 90, 100, 110, 120.0])
        ref = np.array([bsm_price(s, 100, tau, 0.03, 0.01, 0.25) for tau in (0.5, 1.0) for s in S])
        return float(np.abs(p - ref).max()), 0.7
    if name == "pinn_vasicek":
        r = np.array([0.0, 0.03, 0.05, 0.08, 0.10])
        return float(np.abs(p - pinn.vasicek_bond(r, 3.0, 0.8, 0.05, 0.02)).max()), 0.005
    S, v = np.array([90.0, 100.0, 110.0]), np.array([0.03, 0.04, 0.05])
    if name == "pinn_bates":
        ref = [pinn.bates_price(s, 100, 1.0, 0.03, 0.0, vv, 2.0, 0.04, 0.3, -0.7, 0.5, -0.1, 0.15) for s, vv in zip(S, v)]
    else:
        ref = [heston_price(s, 100, 1.0, 0.03, 0.0, vv, 2.0, 0.04, 0.3, -0.7) for s, vv in zip(S, v)]
    return float(np.abs(p - np.array(ref)).max()), 1.5


CASES = [("pinn_bsm", "adam", ["--kill-after", "3"]), ("pinn_bsm", "before_switch", ["--kill-after", "6"]), ("pinn_bsm", "lbfgs_closure", ["--kill-at", "closure:40"]),
         ("pinn_bsm", "after_lbfgs_step", ["--kill-after", "8"]), ("pinn_bsm", "after_polish", ["--kill-at", "finish"]),
         ("pinn_vasicek", "adam", ["--kill-after", "4"]), ("pinn_vasicek", "lbfgs_closure", ["--kill-at", "closure:25"]), ("pinn_vasicek", "after_polish", ["--kill-at", "finish"]),
         ("pinn_heston", "adam", ["--kill-after", "2"]), ("pinn_heston", "lbfgs_closure", ["--kill-at", "closure:30"]), ("pinn_heston", "after_polish", ["--kill-at", "finish"]),
         ("pinn_bates", "lbfgs_closure", ["--kill-at", "closure:30"])]


@pytest.mark.parametrize("name, point, args", CASES, ids=[f"{n}-{p}" for n, p, _ in CASES])
def test_a_pinn_killed_at_this_point_resumes_in_a_new_process_to_the_uninterrupted_result(name, point, args, tmp_path):
    ref = run(name, tmp_path / "ref", "reference", tmp_path / "ref.json")
    assert ref.returncode == 0, ref.stderr[-1500:]
    dead = run(name, tmp_path / "run", "interrupted", tmp_path / "dead.json", *args)
    assert was_killed(dead.returncode) and not (tmp_path / "dead.json").exists(), (dead.returncode, dead.stderr[-600:])
    fit = next((tmp_path / "run").rglob("metrics.jsonl")).parent
    killed_rows = rows(fit / "metrics.jsonl")
    assert (fit / "last.pt").exists() and all(r["type"] != "finish" for r in killed_rows)
    if point == "lbfgs_closure":
        assert (fit / "inflight.json").exists(), "a kill inside an L-BFGS step must leave the in-flight marker"
    res = run(name, tmp_path / "run", "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-1500:]
    a, b = json.loads((tmp_path / "ref.json").read_text()), json.loads((tmp_path / "res.json").read_text())
    for key in ("loss", "outputs", "params"):
        np.testing.assert_allclose(b[key], a[key], rtol=1e-5, atol=1e-6, err_msg=key)
    error, tolerance = reference_error(name, b["outputs"])
    assert error < tolerance, (error, tolerance)                                                      # the existing accuracy test's tolerance, unchanged
    log = rows(fit / "metrics.jsonl")
    epochs = [r for r in log if r["type"] == "epoch"]
    assert [r["epoch"] for r in epochs] == list(range(1, len(epochs) + 1)) and log[-1]["type"] == "finish"
    record(family="pinn", scenario=name, kill_point=point, killed_after_units=len([r for r in killed_rows if r["type"] == "epoch"]), units_total=len(epochs), recovery_level=[r for r in log if r["type"] == "resume"][-1]["level"],
           recomputed_work=any(r["type"] == "recomputed_work" for r in log), max_abs_diff_loss=maxdiff(a["loss"], b["loss"]), max_abs_diff_prices=maxdiff(a["outputs"], b["outputs"]), max_abs_diff_params=maxdiff(a["params"], b["params"]),
           pricing_error=error, pricing_tolerance=tolerance)
    phases = [r["phase"] for r in epochs]
    adam = ADAM_UNITS[name]
    assert phases[:adam] == ["adam"] * adam and set(phases[adam:]) == {"lbfgs"}, phases               # the two phases are distinct in the record
    resume = [r for r in log if r["type"] == "resume"]
    assert resume and resume[-1]["level"] == "exact"
    if point == "lbfgs_closure":
        lost = [r for r in log if r["type"] == "recomputed_work"]
        assert lost and lost[-1]["phase"] == "lbfgs", log                                              # the unfinished L-BFGS step was detected and its recomputation recorded
    grads = rows(fit / "gradients.jsonl")
    by = {}
    for r in grads:
        by.setdefault(r["optimizer"], []).append(r["epoch"])
    assert by["adam"] == list(range(adam)) and by["lbfgs"] == list(range(adam, adam + len(by["lbfgs"]))), by      # contiguous per optimizer: no unit twice, none missing
    ref_grads = rows(next((tmp_path / "ref").rglob("gradients.jsonl")))
    for x, y in zip(grads, ref_grads):
        assert x["global"]["l2_median"] == pytest.approx(y["global"]["l2_median"], rel=1e-4), (x["optimizer"], x["epoch"])     # the gradient history is the uninterrupted run's


def test_a_damaged_newest_checkpoint_inside_the_polish_falls_back_and_a_missing_one_restarts(tmp_path):
    """Mid-polish the newest checkpoint is damaged: the previous L-BFGS boundary is used, the continuation is reported as approximate (the platform's fallback rule), and the network still ends valid and within the
    pricing tolerance. With no checkpoint at all a resume simply trains from the start."""
    name = "pinn_vasicek"
    assert was_killed(run(name, tmp_path / "run", "interrupted", tmp_path / "dead.json", "--kill-at", "closure:25").returncode)
    fit = next((tmp_path / "run").rglob("metrics.jsonl")).parent
    (fit / "last.pt").write_bytes((fit / "last.pt").read_bytes()[:300])
    res = run(name, tmp_path / "run", "resume", tmp_path / "res.json")
    assert res.returncode == 0, res.stderr[-1200:]
    log = rows(fit / "metrics.jsonl")
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["file"] == "last.prev.pt" and resume["level"] == "epoch_boundary" and any("damaged" in x for x in resume["reasons"])
    out = json.loads((tmp_path / "res.json").read_text())
    error, tolerance = reference_error(name, out["outputs"])
    assert np.isfinite(out["params"]).all() and error < tolerance, (error, tolerance)
    assert [r["epoch"] for r in log if r["type"] == "epoch"] == list(range(1, len([r for r in log if r["type"] == "epoch"]) + 1))
    fresh = run(name, tmp_path / "empty", "resume", tmp_path / "res2.json")
    assert fresh.returncode == 0, fresh.stderr[-1200:]
    log2 = rows(next((tmp_path / "empty").rglob("metrics.jsonl")))
    assert not any(r["type"] == "resume" for r in log2) and log2[-1]["type"] == "finish"
