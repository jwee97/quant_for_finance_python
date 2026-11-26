"""The gradient diagnostics on the model families added in the second round: the residual gradients of a physics-informed network (second-order automatic differentiation) against an independent computation, the
two optimizers of a PINN as separate phases, a Q-network under Double DQN, and the Stable-Baselines3 agents through their public interface only (marked as limited: no pre-clip norms, no update ratios)."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import context, gradients as gd              # noqa: E402
from tests.test_fresh_process_sb3 import PY as SB3_PY         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_the_gradient_of_a_pde_residual_loss_is_measured_exactly():
    """u_t = u_xx on collocation points: the loss is a function of second derivatives, so its gradient needs ``create_graph``. The monitor reads what ``backward`` left; an independent ``autograd.grad`` must agree."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 12), torch.nn.Tanh(), torch.nn.Linear(12, 12), torch.nn.Tanh(), torch.nn.Linear(12, 1))
    pts = torch.rand(64, 2).requires_grad_(True)
    u = net(pts)
    g = torch.autograd.grad(u.sum(), pts, create_graph=True)[0]
    u_t, u_x = g[:, 0:1], g[:, 1:2]
    u_xx = torch.autograd.grad(u_x.sum(), pts, create_graph=True)[0][:, 1:2]
    loss = ((u_t - u_xx) ** 2).mean()
    ref = torch.autograd.grad(loss, list(net.parameters()), retain_graph=True, allow_unused=True)      # independent of backward(); the last bias has no effect on a second derivative: its gradient is absent, not zero
    assert ref[5] is None
    loss.backward()
    mon = gd.GradientMonitor("adam", dict(net.named_parameters()), "standard")
    mon.begin_epoch(0, 1)
    mon.observe(0)
    s, ev = mon.end_epoch(0, lr=1e-3, train_loss=float(loss), val_loss=None)
    expected = math.sqrt(sum(float((r.double() ** 2).sum()) for r in ref if r is not None))
    assert s["global"]["l2_median"] == pytest.approx(expected, rel=1e-6)
    by_layer = {"0": ref[0:2], "2": ref[2:4], "4": ref[4:6]}
    for row in s["layers"]:
        got = [r for r in by_layer[row["name"]] if r is not None]
        exp = math.sqrt(sum(float((r.double() ** 2).sum()) for r in got))
        n = sum(r.numel() for r in got)                                                                    # an absent gradient is outside the denominator, not a zero inside it
        assert row["l2"] == pytest.approx(exp, rel=1e-6) and row["rms"] == pytest.approx(exp / math.sqrt(n), rel=1e-6)
    assert s["global"]["missing_gradients"] == 1
    assert not [e for e in ev if e["severity"] == "critical"]


def test_a_pinn_reports_adam_and_lbfgs_as_separate_optimizers_with_finite_stable_gradients(tmp_path):
    from src.models import pinn

    with context.use(str(tmp_path), grad_level="standard"):
        pinn.VasicekPINN(0.8, 0.05, 0.02, maturity=5.0, steps=400, lbfgs_steps=30, hidden=12, layers=2, batch=128, seed=1).fit()
    fit = next(tmp_path.rglob("gradients.jsonl")).parent
    g = rows(fit / "gradients.jsonl")
    by = {}
    for r in g:
        by.setdefault(r["optimizer"], []).append(r)
    assert set(by) == {"adam", "lbfgs"} and by["adam"][-1]["epoch"] + 1 == by["lbfgs"][0]["epoch"]                  # the phases follow each other without overlap
    for r in g:
        assert r["global"]["nonfinite_steps"] == 0 and r["global"]["l2_median"] > 0 and math.isfinite(r["global"]["l2_median"])
        assert all(math.isfinite(l["rms"]) for l in r["layers"])
    assert all(any(l.get("update_ratio") for l in r["layers"]) for r in by["lbfgs"])                                 # the real displacement of each L-BFGS step is measured
    assert by["lbfgs"][-1]["global"]["l2_median"] < by["adam"][0]["global"]["l2_median"]                            # the polish ends near a stationary point
    events = [json.loads(line) for line in (fit / "metrics.jsonl").read_text().splitlines() if '"gradient_event"' in line]
    assert not [e for e in events if e["severity"] == "critical"]


def test_double_dqn_exposes_its_q_network_gradients(tmp_path):
    from src.rl import deep
    from src.rl.envs import RegimeFractionEnv

    with context.use(str(tmp_path), grad_level="detailed"):
        deep.DQN(double=True, hidden=16, warmup=100, target_every=50).train(RegimeFractionEnv(seed=0), steps=1000, seed=3)
    g = rows(next(tmp_path.rglob("gradients.jsonl")))
    assert {r["optimizer"] for r in g} == {"adam"} and len(g) == 4
    first = g[0]
    assert [l["name"] for l in first["layers"]] == ["0", "2", "4"] and all(l["g_w"] > 0 and l["n_params"] > 0 for l in first["layers"])
    assert first["histogram"]["bins"] and first["global"]["sampled"]["p99"] >= first["global"]["sampled"]["p50"] >= 0


@pytest.mark.skipif(SB3_PY is None, reason="dependency-blocked: no Python with stable-baselines3, sb3-contrib and gymnasium")
@pytest.mark.parametrize("name, optimizers", [("sb3_ppo", {"policy"}), ("sb3_sac", {"actor", "critic"})])
def test_sb3_agents_expose_policy_and_value_gradients_through_public_hooks_and_say_what_is_limited(name, optimizers, tmp_path):
    out = subprocess.run([SB3_PY, "-W", "ignore", "-m", "tests.fresh_process_scenarios", name, str(tmp_path / "run"), "reference", str(tmp_path / "r.json")], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-1000:]
    fit = next((tmp_path / "run").rglob("gradients.jsonl")).parent
    g = rows(fit / "gradients.jsonl")
    assert {r["optimizer"] for r in g} == optimizers
    for r in g:
        assert math.isfinite(r["global"]["l2_median"]) and r["global"]["l2_median"] > 0 and r["layers"]
        assert not any("update_ratio" in l for l in r["layers"])                                                     # not measurable through public SB3 interfaces: marked limited, not faked
    log = [json.loads(line) for line in (fit / "metrics.jsonl").read_text().splitlines()]
    note = [r for r in log if r["type"] == "diagnostics_note"]
    assert note and "limited" in note[0]["message"] and "update-to-weight" in note[0]["message"]
