"""The gradient diagnostics checked against numbers worked out independently: every statistic of a gradient that is set by hand, the handling of frozen, missing, sparse and non-finite gradients, accumulation,
mixed-precision unscaling, clipping, several optimizers, the measured update-to-weight ratio, and the warnings (a single spike is not an explosion; a frozen layer is not a dead one)."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import gradients as gd          # noqa: E402

torch.set_num_threads(1)


def monitor(params, level="standard", **kw):
    m = gd.GradientMonitor("opt", params, level, **kw)
    m.begin_epoch(0, steps_per_epoch=1)
    return m


def epoch(m, **kw):
    return m.end_epoch(0, lr=0.1, train_loss=1.0, val_loss=1.0, **kw)


def param(values, grad=None, requires_grad=True):
    p = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32), requires_grad=requires_grad)
    if grad is not None:
        p.grad = torch.tensor(grad, dtype=torch.float32)
    return p


def test_global_statistics_match_numpy_on_a_hand_set_gradient():
    g = np.array([3.0, -4.0, 0.0, 0.0, 1e-12, 12.0, -0.5, 2.0, 0.0, 1.0], dtype=np.float32)
    p = param(np.ones(10), g)
    m = monitor({"fc.weight": p})
    m.observe(0)
    s, ev = epoch(m)
    glob, smp = s["global"], s["global"]["sampled"]
    assert glob["l2_median"] == pytest.approx(float(np.linalg.norm(g.astype(np.float64))), rel=1e-6)
    assert smp["linf"] == pytest.approx(12.0) and smp["min"] == pytest.approx(-4.0) and smp["max"] == pytest.approx(12.0)
    assert smp["mean_abs"] == pytest.approx(np.abs(g).mean(), rel=1e-6)
    assert smp["rms"] == pytest.approx(np.sqrt((g.astype(np.float64) ** 2).mean()), rel=1e-6)
    assert smp["std"] == pytest.approx(g.astype(np.float64).std(), rel=1e-5)
    assert smp["zero_frac"] == pytest.approx(0.3) and smp["near_zero_frac"] == pytest.approx(0.4)               # the 1e-12 entry is near zero (< 1e-8) but not zero
    for q in (50, 90, 95, 99):
        assert smp[f"p{q}"] == pytest.approx(np.percentile(np.abs(g), q, method="nearest"))
    assert smp["median"] == smp["p50"]
    assert glob["missing_gradients"] == 0 and glob["trainable_params"] == 1


def test_frozen_parameters_are_left_out_and_a_missing_gradient_is_not_a_zero_gradient():
    live, frozen = param([1.0, 1.0], [3.0, 4.0]), param([1.0, 1.0], [100.0, 100.0], requires_grad=False)
    unused = param([1.0, 1.0])                                                                        # trainable, but no gradient reached it
    m = monitor({"a.weight": live, "b.weight": frozen, "c.weight": unused})
    m.observe(0)
    s, _ = epoch(m)
    g = s["global"]
    assert g["l2_median"] == pytest.approx(5.0) and g["frozen_params"] == 1 and g["trainable_params"] == 2 and g["missing_gradients"] == 1
    assert [r["name"] for r in s["layers"]] == ["a"]                                                  # neither the frozen nor the unused layer is in the active-gradient table
    assert s["global"]["sampled"]["zero_frac"] == 0.0                                                 # the unused parameter is not counted as zeros


def test_nonfinite_gradients_are_counted_and_excluded_from_the_other_statistics():
    p = param(np.ones(5), [1.0, float("nan"), 2.0, float("inf"), -2.0])
    m = monitor({"fc.weight": p})
    m.observe(0)
    s, events = epoch(m)
    smp = s["global"]["sampled"]
    assert smp["nonfinite"] == 2 and s["global"]["nonfinite_steps"] == 1
    assert smp["linf"] == pytest.approx(2.0) and smp["mean_abs"] == pytest.approx(5.0 / 3.0) and smp["min"] == -2.0
    assert any(e["kind"] == "nonfinite_gradient" and e["severity"] == "critical" for e in events)
    assert s["layers"][0]["nonfinite"] == 2


def test_a_sparse_gradient_is_measured_on_its_stored_values_and_its_implicit_zeros_count():
    emb = torch.nn.Embedding(100, 4, sparse=True)
    out = emb(torch.tensor([3, 7])).sum()
    out.backward()
    m = monitor({"emb.weight": emb.weight})
    m.observe(0)
    s, _ = epoch(m)
    row = s["layers"][0]
    assert row["l2"] == pytest.approx(math.sqrt(8))                                                    # eight stored values equal to one
    assert row["zero_frac"] == pytest.approx(1 - 8 / 400)                                              # the 392 implicit zeros are zeros


def test_accumulated_gradient_is_measured_once_after_all_microbatches():
    p = param([1.0, 1.0, 1.0])
    m = monitor({"fc.weight": p}, accumulation=2)
    for micro in (torch.tensor([1.0, 2.0, 2.0]), torch.tensor([2.0, 2.0, 1.0])):
        (p * micro).sum().backward()                                                                  # microbatch gradients accumulate in .grad
    m.observe(0)                                                                                      # called once, at the optimizer step
    s, _ = epoch(m)
    assert s["global"]["l2_median"] == pytest.approx(np.linalg.norm([3.0, 4.0, 3.0]))
    assert s["global"]["optimizer_steps"] == 1


def test_mixed_precision_gradients_are_measured_after_unscaling():
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    p = param([1.0, 1.0])
    opt = torch.optim.SGD([p], lr=0.1)
    loss = (p * torch.tensor([3.0, 4.0])).sum()
    scaler.scale(loss).backward()
    assert p.grad.tolist() == [3.0 * 1024, 4.0 * 1024]                                                # as the backward pass leaves them
    m = monitor({"fc.weight": p})
    scaler.unscale_(opt)                                                                              # the training loop does this before clipping and before the monitor
    m.observe(0)
    s, _ = epoch(m)
    assert s["global"]["l2_median"] == pytest.approx(5.0)


def test_clipping_is_counted_and_pre_and_post_clip_norms_are_reported_apart():
    p = param([1.0, 1.0], [6.0, 8.0])
    m = monitor({"fc.weight": p}, clip=1.0)
    m.observe(0)
    torch.nn.utils.clip_grad_norm_([p], 1.0)
    m.after_clip(0)
    s, _ = epoch(m)
    g = s["global"]
    assert g["l2_median"] == pytest.approx(10.0) and g["post_clip_l2_mean"] == pytest.approx(1.0, rel=1e-5)
    assert g["clip_frequency"] == 1.0 and g["clipped_steps"] == 1


def test_the_update_to_weight_ratio_is_the_one_the_optimizer_applied():
    p = param([3.0, 4.0], [1.0, 2.0])
    opt = torch.optim.SGD([p], lr=0.5)
    m = monitor({"fc.weight": p})
    m.observe(0)
    m.before_step(0)
    opt.step()
    m.after_step(0)
    s, _ = epoch(m)
    row = s["layers"][0]
    assert row["update_ratio"] == pytest.approx(0.5 * math.sqrt(5) / 5.0, rel=1e-6)                   # |lr g| / |w|
    assert row["g_w"] == pytest.approx(math.sqrt(5) / 5.0, rel=1e-6) and row["param_norm"] == pytest.approx(5.0)


def test_two_optimizers_are_monitored_separately():
    g_p, d_p = param([1.0, 1.0], [3.0, 4.0]), param([1.0, 1.0], [6.0, 8.0])
    mg, md = gd.GradientMonitor("generator", {"G.weight": g_p}, "standard"), gd.GradientMonitor("critic", {"D.weight": d_p}, "standard")
    for m in (mg, md):
        m.begin_epoch(0, 1)
        m.observe(0)
    assert epoch(mg)[0]["global"]["l2_median"] == pytest.approx(5.0) and epoch(md)[0]["global"]["l2_median"] == pytest.approx(10.0)
    assert epoch(mg)[0]["optimizer"] == "generator"


def test_layers_are_grouped_by_module_in_depth_order_and_compared_by_rms():
    net = torch.nn.Sequential(torch.nn.Linear(3, 100), torch.nn.Linear(100, 1))
    for q in net.parameters():
        q.grad = torch.ones_like(q)
    m = monitor(dict(net.named_parameters()))
    m.observe(0)
    s, _ = epoch(m)
    names = [r["name"] for r in s["layers"]]
    assert names == ["0", "1"] and s["layers"][0]["n_params"] == 400 and s["layers"][1]["n_params"] == 101
    assert s["layers"][0]["rms"] == pytest.approx(1.0) and s["layers"][1]["rms"] == pytest.approx(1.0)       # equal per-element gradient: equal RMS although the layers differ 4x in size
    assert s["layers"][0]["l2"] == pytest.approx(20.0)


def test_detailed_level_adds_histograms_and_medians_and_lightweight_adds_neither():
    p = param(np.ones(8), [1e-3, 1e-3, 1e-1, 1.0, 0.0, 0.0, 10.0, 1e-6])
    d = monitor({"fc.weight": p}, "detailed")
    d.observe(0)
    sd, _ = epoch(d)
    h = sd["histogram"]
    assert h["zeros"] == 2 and sum(h["bins"]) == 6 and sd["layers"][0]["median_abs"] == pytest.approx(np.median(np.abs(p.grad.numpy())))
    lw = monitor({"fc.weight": p}, "lightweight")
    lw.observe(0)
    sl, _ = epoch(lw)
    assert "sampled" not in sl["global"] and not sl["layers"] and "histogram" not in sl and sl["global"]["l2_median"] is not None


def test_monitoring_never_changes_gradients_parameters_or_graph():
    p = param([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    before, gbefore = p.detach().clone(), p.grad.clone()
    m = monitor({"fc.weight": p}, "detailed")
    m.observe(0)
    m.after_clip(0)
    m.before_step(0)
    m.after_step(0)
    epoch(m)
    assert torch.equal(p.detach(), before) and torch.equal(p.grad, gbefore) and not any(t.requires_grad for t in m._norms)


# ----------------------------------------------------------------------------------------------------------- the warnings
def run_units(m, units, lr=0.1, losses=None):
    out = []
    for e, norm in enumerate(units):
        m.begin_epoch(e, 10, lr)
        p = next(iter(m.active.values()))
        for k, n in enumerate(norm if isinstance(norm, (list, tuple)) else [norm] * 10):
            p.grad = torch.full_like(p, n / math.sqrt(p.numel()))
            m.observe(k)
            if m.clip:
                torch.nn.utils.clip_grad_norm_([p], m.clip)
        tl, vl = (losses[e] if losses else (1.0, 1.0))
        out += m.end_epoch(e, lr=lr, train_loss=tl, val_loss=vl)[1]
    return out


def kinds(events):
    return [e["kind"] for e in events]


def test_one_large_gradient_is_a_spike_not_an_explosion():
    m = monitor({"fc.weight": param(np.ones(4))}, "lightweight", clip=1.0)
    ev = run_units(m, [[1.0] * 9 + [500.0]] + [1.0] * 5)
    assert "gradient_spike" in kinds(ev) and "exploding_gradient" not in kinds(ev) and "repeated_clipping" not in kinds(ev)
    assert "exploding_gradient" not in kinds(ev)


def test_a_gradient_that_keeps_growing_and_is_clipped_every_step_is_reported_after_it_persists():
    m = monitor({"fc.weight": param(np.ones(4))}, "lightweight", clip=1.0)
    ev = run_units(m, [1.0, 1.0, 1.0, 1e3, 1e3, 1e3, 1e3, 1e3])
    k = kinds(ev)
    assert "repeated_clipping" in k and "exploding_gradient" in k
    first = next(e for e in ev if e["kind"] == "exploding_gradient")
    assert first["epoch"] >= 5 and first["action"] == "continued" and first["thresholds"]["explode_growth"] == 100.0 and first["lr"] == 0.1


def test_a_layer_with_negligible_gradient_is_flagged_only_when_it_persists_and_a_frozen_layer_never():
    net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for q in net[0].parameters():
        q.requires_grad_(False)                                                                         # frozen on purpose
    m = monitor(dict(net.named_parameters()), "standard")
    ev = []
    for e in range(5):
        m.begin_epoch(e, 1, 0.1)
        for i, layer in enumerate(net):
            for q in layer.parameters():
                if q.requires_grad:
                    q.grad = torch.full_like(q, 1.0 if i == 2 else 1e-10)                             # layer 1 starves, layer 2 is healthy
        m.observe(0)
        ev += m.end_epoch(e, lr=0.1, train_loss=1.0, val_loss=1.0)[1]
    vanish = [x for x in ev if x["kind"] == "vanishing_gradient"]
    assert vanish and vanish[0]["epoch"] == 2 and vanish[0]["layers"] == ["1"]                          # third unit in a row, not the first
    assert all("0" not in x["layers"] for x in ev)                                                      # the frozen layer is never named


def test_a_trainable_layer_with_exactly_zero_gradient_is_dead_but_a_sparse_one_is_not():
    a, b = param(np.ones(4), np.zeros(4)), param(np.ones(4), [1.0, 0.0, 0.0, 0.0])
    m = monitor({"a.weight": a, "b.weight": b}, "standard")
    ev = []
    for e in range(4):
        m.begin_epoch(e, 1, 0.1)
        m.observe(0)
        ev += m.end_epoch(e, lr=0.1, train_loss=1.0, val_loss=1.0)[1]
    dead = [x for x in ev if x["kind"] == "dead_layer"]
    assert dead and dead[0]["layers"] == ["a"]


def test_overfitting_needs_several_units_of_falling_training_and_rising_validation_loss():
    m = monitor({"fc.weight": param(np.ones(4))}, "lightweight")
    losses = [(1.0, 1.0), (0.9, 1.0), (0.8, 1.1), (0.7, 1.2), (0.6, 1.3)]
    ev = run_units(m, [1.0] * 5, losses=losses)
    assert "potential_overfitting" in kinds(ev) and next(e for e in ev if e["kind"] == "potential_overfitting")["epoch"] == 4
    m2 = monitor({"fc.weight": param(np.ones(4))}, "lightweight")
    assert "potential_overfitting" not in kinds(run_units(m2, [1.0] * 5, losses=[(1.0, 1.0), (0.9, 1.1), (0.95, 1.2), (0.7, 1.1), (0.6, 1.3)]))


def test_the_detector_state_survives_a_checkpoint():
    m = monitor({"fc.weight": param(np.ones(4))}, "lightweight", clip=1.0)
    run_units(m, [1.0, 1.0, 1.0, 1e3, 1e3])
    clone = monitor({"fc.weight": param(np.ones(4))}, "lightweight", clip=1.0)
    clone.load_state_dict(m.state_dict())
    ev = run_units_from(clone, [1e3, 1e3, 1e3], start=5)
    assert "exploding_gradient" in kinds(ev)


def run_units_from(m, units, start):
    out = []
    for e, norm in enumerate(units, start=start):
        m.begin_epoch(e, 10, 0.1)
        p = next(iter(m.active.values()))
        for k in range(10):
            p.grad = torch.full_like(p, norm / math.sqrt(p.numel()))
            m.observe(k)
            torch.nn.utils.clip_grad_norm_([p], 1.0)
        out += m.end_epoch(e, lr=0.1, train_loss=1.0, val_loss=1.0)[1]
    return out


def test_levels_are_validated_and_auto_depends_on_where_it_runs():
    assert gd.resolve_level("auto", True, True) == "lightweight" and gd.resolve_level("auto", False, True) == "standard" and gd.resolve_level("auto", False, False) == "off"
    assert gd.resolve_level("detailed", True, True) == "detailed"
    with pytest.raises(ValueError):
        gd.resolve_level("loud", False, True)
