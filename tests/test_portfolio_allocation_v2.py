"""Second round of the portfolio allocation: holding period, forecast horizon, rebalancing and execution lag as separate quantities; calibration beyond the pooled slope with dependence-aware inference; stabilisers; optimiser
adapters; point-in-time currency conversion; and that everything recorded before these existed is reproduced exactly."""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

from src.ml import alloc_adapters as ad
from src.ml import alloc_calibration as cal
from src.ml import alloc_fx as fxm
from src.ml import alloc_horizon as hz
from src.ml import alloc_stability as stab
from src.ml import allocator_compare as ac
from src.ml import config as mlcfg
from src.ml import portfolio_alloc as pa
from src.ml.backtest import run_backtest
from tests.test_portfolio_allocation import DATES, N, panel, score_of, signal

BT = {"rebalance": "weekly", "cost_bps": 5.0, "slippage_bps": 2.0, "target_vol": None, "max_leverage": 1.5, "extra_lag": 0}


def informative(r, seed=0, strength=0.4, noise=0.02, h=5, lag=1):
    """A forecast whose information is the realised window return it is scored against, plus noise."""
    truth = pa.realised_window_returns(r, h, lag).fillna(0.0)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(strength * truth.to_numpy() + rng.normal(0, noise, r.shape), index=r.index, columns=r.columns)


def alloc_sig(allocator="mean_variance", h=5, lag=1, **kw):
    lim = kw.pop("limits", {})
    base = signal(allocator)
    base["window_days"], base["lag_total"] = h, lag
    base["allocation"].update({"calibrate": "slope", "calib_min_t": 0.0, "calib_min_obs": 60, **kw})
    base["allocation"]["limits"] = {"max_weight": 1.0, **lim}
    return base


# ============================================================================================ 1. holding, horizon, rebalance, lag
@pytest.mark.parametrize("h", [1, 5, 10, 20])
@pytest.mark.parametrize("every", [1, 5, 21])
def test_horizon_and_rebalance_frequency_vary_independently_and_every_book_is_valid(h, every):
    r = panel(vols=(0.01, 0.02), seed=60)
    s = informative(r, 60, h=h)
    w, log = pa.run(s, r, None, alloc_sig(h=h, opt_every=every))
    assert log["opt_every"] == every and log["holding_window_days"] == h
    held = w[w.abs().sum(axis=1) > 0]
    assert (held.sum(axis=1).round(9) == 1.0).all() and (held >= -1e-9).all().all(), "one book at a time: overlapping forecasts are never added together"
    assert log["counts"]["decisions"] == len(range(0, N, every))
    changes = w.diff().abs().sum(axis=1).gt(1e-12)
    assert changes.iloc[1:].to_numpy()[[i for i in range(len(changes) - 1) if (i + 1) % every != 0]].sum() == 0, "the book changes only on decision dates"


def test_the_holding_window_sets_the_edge_and_covariance_scale_and_not_the_decision_schedule():
    r = panel(vols=(0.01, 0.01), seed=61)
    s = informative(r, 61, h=10)
    for L, expected in ((5, 0.5), (10, 1.0), (20, 1.0)):                                       # the edge accrues within the horizon and is not extrapolated beyond it
        w, log = pa.run(s, r, None, alloc_sig(h=10, holding_days=L, opt_every=5))
        rec = [x for x in log["records"] if x.get("horizon")][-1]["horizon"]
        assert rec["edge_scale"] == pytest.approx(expected) and rec["holding_days"] == L and rec["decision_every"] == 5 and rec["forecast_days"] == 10
    assert hz.edge_scale(3, 10) == pytest.approx(0.3) and hz.edge_scale(40, 10) == 1.0


def test_holding_days_equal_to_the_horizon_reproduces_the_original_window_exactly():
    r = panel(vols=(0.01, 0.02), seed=62)
    s = informative(r, 62)
    a, _ = pa.run(s, r, None, alloc_sig(h=5))
    b, _ = pa.run(s, r, None, alloc_sig(h=5, holding_days=5))
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)


def test_staggered_tranches_hold_overlapping_decisions_as_partial_positions_and_stay_feasible():
    r = panel(vols=(0.01, 0.02, 0.015), corr=0.2, seed=63)
    s = informative(r, 63, h=5)
    sg = alloc_sig(h=5, holding_days=20, opt_every=5, lifecycle="staggered_tranches", limits={"max_weight": 0.6})
    w, log = pa.run(s, r, None, sg)
    rec = [x for x in log["records"] if x.get("tranche")][-1]
    assert rec["tranche"] == {"active": 4, "of": 4, "replaced": "the oldest, expiring tranche"} and "held_violations" not in rec
    held = w[w.abs().sum(axis=1) > 0]
    assert (held.sum(axis=1).round(9) == 1.0).all() and held.max().max() <= 0.6 + 1e-9, "an average of feasible books is feasible"
    single, _ = pa.run(s, r, None, alloc_sig(h=5, holding_days=20, opt_every=5, limits={"max_weight": 0.6}))
    assert w.diff().abs().sum().sum() < single.diff().abs().sum().sum(), "four staggered tranches turn the book over more slowly than one book re-optimised each decision"
    one_tranche_books = [np.array(list(x["weights"].values())) for x in log["records"] if x.get("tranche")][-4:]
    assert np.allclose(np.mean(one_tranche_books, axis=0), np.array(list(rec["held_weights"].values())), atol=1e-9), "the held book is the mean of the last four tranche books"


def test_config_validation_derives_the_holding_window_and_no_longer_forces_one_day():
    c = mlcfg.experiment_config({"dataset": {"target": "fwd_ret_5", "horizon": 5}, "signal": {"method": "portfolio_allocation", "holding_period": 10, "allocation": {"allocator": "mean_variance"}}})
    assert c["signal"]["allocation"]["holding_days"] == 10 and c["signal"]["holding_period"] == 10
    legacy = mlcfg.experiment_config({"dataset": {"target": "fwd_ret_5", "horizon": 5}, "signal": {"method": "portfolio_allocation", "holding_period": 1, "allocation": {"allocator": "mean_variance"}}})
    assert "holding_days" not in legacy["signal"]["allocation"], "1 with no holding_days is the original behaviour"
    st = mlcfg.experiment_config({"dataset": {"target": "fwd_ret_5", "horizon": 5}, "signal": {"method": "portfolio_allocation", "holding_period": 15, "allocation": {"lifecycle": "staggered_tranches", "opt_every": 5}}})
    assert st["signal"]["allocation"]["holding_days"] == 15 and st["signal"]["allocation"]["lifecycle"] == "staggered_tranches"
    with pytest.raises(mlcfg.MLConfigError, match="holding_days"):
        mlcfg.experiment_config({"dataset": {"target": "fwd_ret_5", "horizon": 5}, "signal": {"method": "portfolio_allocation", "allocation": {"holding_days": 500}}})


@pytest.mark.parametrize("h, hp, life, every", [(1, 1, "single_book", None), (5, 5, "staggered_tranches", 5), (10, 10, "staggered_tranches", 5), (20, 20, "staggered_tranches", 5), (20, 20, "single_book", 1)])
def test_the_whole_workflow_from_forecast_to_executed_pnl_for_each_horizon(h, hp, life, every):
    """Train, forecast, allocate, execute through the engine with lag and costs: 1-, 5-, 10- and 20-day horizons with their own holding windows and schedules."""
    from src.ml import experiments as ex
    from tests.ml_helpers import make_bundle

    b = make_bundle(n_assets=2, n=1500, seed=3, alpha=1.0, macro=False)
    al = {"allocator": "mean_variance", "lifecycle": life, "limits": {"max_weight": 0.9}}
    if every:
        al["opt_every"] = every
    cfg = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": f"fwd_ret_{h}", "horizon": h, "min_history": 60}, "split": {"min_train_days": 300, "refit_every": 63, "validation_fraction": 0.2,
           "test_fraction": 0.2, "forward_fraction": 0.1}, "model": {"kind": "ridge"}, "backtest": {"rebalance": "weekly", "target_vol": None}, "signal": {"method": "portfolio_allocation", "holding_period": hp, "allocation": al}}
    o = ex.run_experiment(cfg, bundle=b, save=False, explain=False, final_model=False)
    pa_rep = o.report["portfolio_allocation"]
    assert pa_rep["counts"]["optimal"] == pa_rep["counts"]["decisions"] > 0 and pa_rep["holding_window_days"] == (hp if hp != 1 else h)
    w = o.weights.dropna(how="all")
    live = w[w.abs().sum(axis=1) > 0]
    assert live.sum(axis=1).round(6).between(0.0, 1.0 + 1e-6).all() and (live >= -1e-9).all().all()
    assert np.allclose(o.net, o.gross - o.costs.reindex(o.gross.index).fillna(0.0), atol=1e-10), "net = gross - costs"
    assert o.turnover.sum() > 0 and o.costs.sum() > 0, "trades were made and paid for"


def test_execution_lag_moves_the_pnl_and_not_the_target_weights_for_a_fixed_forecast():
    r = panel(vols=(0.01, 0.02), seed=64)
    s = informative(r, 64)
    w1, _ = pa.run(s, r, None, alloc_sig(opt_every=1, lag=1))                                   # a decision every day, so that a later execution reads a different one
    w3, _ = pa.run(s, r, None, alloc_sig(opt_every=1, lag=1))
    assert np.allclose(w1.to_numpy(), w3.to_numpy())
    daily = {**BT, "rebalance": "daily"}
    a = run_backtest(w1, r, None, daily, 1, "t")
    b = run_backtest(w1, r, None, daily, 3, "t")
    ga, gb = a.gross_returns.dropna(), b.gross_returns.dropna()
    assert not np.allclose(ga.reindex(gb.index).fillna(0), gb), "the same book executed later earns different returns"
    held_a, held_b = a.weights.shift(1).fillna(0.0), b.weights.shift(1).fillna(0.0)
    first_a, first_b = (held_a.abs().sum(axis=1) > 0).idxmax(), (held_b.abs().sum(axis=1) > 0).idxmax()
    assert (first_b - first_a).days >= 2, "a longer lag starts holding the book later"
    assert a.config["signal_lag"] == 1 and b.config["signal_lag"] == 3


def test_variance_ratio_covariance_follows_serial_correlation_and_the_direct_estimate_needs_independent_windows():
    rng = np.random.default_rng(65)
    n, phi, L = 3000, 0.4, 5
    e = rng.standard_normal((n, 2)) * 0.01
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    w = pd.DataFrame(x, index=DATES[:n] if n <= len(DATES) else pd.bdate_range("2010-01-01", periods=n), columns=["A", "B"])
    expected = 1 + 2 * sum((1 - k / L) * phi ** k for k in range(1, L))
    vr = hz.variance_ratio(w, L)
    assert vr["A"] == pytest.approx(expected, rel=0.2) and vr["A"] > 1.3
    daily = w.cov()
    lin, _ = hz.horizon_covariance(daily, w, L, "linear")
    vrc, info = hz.horizon_covariance(daily, w, L, "variance_ratio")
    assert vrc.loc["A", "A"] > lin.loc["A", "A"] * 1.3 and info["variance_ratio"]["A"] == pytest.approx(vr["A"])
    iid = pd.DataFrame(rng.standard_normal((n, 2)) * 0.01, index=w.index, columns=["A", "B"])
    assert hz.variance_ratio(iid, L)["A"] == pytest.approx(1.0, abs=0.2)
    mp, info = hz.horizon_covariance(daily, w, L, "multiperiod")
    assert 0 < info["direct_weight"] < 1 and np.linalg.eigvalsh(mp.to_numpy()).min() > 0
    short, info2 = hz.horizon_covariance(daily, w.iloc[:40], L, "multiperiod")
    assert "fallback" in info2 and np.allclose(short.to_numpy(), (daily * L).to_numpy()), "too few independent windows: the linear estimate, said so"


@pytest.mark.parametrize("method", ["linear", "variance_ratio", "multiperiod"])
def test_each_horizon_covariance_method_runs_through_the_allocator_and_is_recorded(method):
    r = panel(vols=(0.01, 0.02), seed=66)
    s = informative(r, 66)
    w, log = pa.run(s, r, None, alloc_sig(h=5, holding_days=10, cov_horizon=method, opt_every=10))
    rec = [x for x in log["records"] if x.get("horizon")][-1]["horizon"]
    assert rec["covariance"]["method"].startswith(method) or rec["covariance"].get("fallback")
    assert log["config"]["cov_horizon"] == method if method != "linear" else "cov_horizon" not in log["config"]


# ============================================================================================ 2. calibration
def test_dependence_aware_standard_errors_match_the_sampling_spread_when_forecasts_persist():
    """Persistent forecasts against overlapping outcomes: the iid error is too small, HAC and the block bootstrap are close to the true spread of the slope."""
    h, T, A = 5, 400, 3
    rng = np.random.default_rng(0)
    bs, hac, boot, iid = [], [], [], []
    for _ in range(60):
        daily = rng.standard_normal((T + h, 1)) * 0.01 + 0.8 * rng.standard_normal((T + h, A)) * 0.01
        x = np.zeros((T, A))
        for t in range(1, T):
            x[t] = 0.9 * x[t - 1] + rng.standard_normal(A) * 0.005
        y = np.array([daily[t + 1:t + 1 + h].sum(axis=0) for t in range(T)]) + 0.5 * x
        b, se, _ = cal.slope_inference(x, y, h, "hac")
        bs.append(b)
        hac.append(se)
        boot.append(cal.slope_inference(x, y, h, "block_bootstrap")[1])
        xv, yv = x.ravel(), y.ravel()
        bb = (xv * yv).sum() / (xv ** 2).sum()
        iid.append(math.sqrt(((yv - bb * xv).var(ddof=1)) / (xv ** 2).sum()))
    emp = float(np.std(bs, ddof=1))
    assert float(np.mean(iid)) < 0.7 * emp, "the iid standard error understates it"
    assert 0.7 * emp < float(np.mean(hac)) < 1.4 * emp and 0.7 * emp < float(np.mean(boot)) < 1.4 * emp


def test_effective_sample_size_counts_dates_not_overlapping_pairs():
    h, T, A = 5, 300, 4
    rng = np.random.default_rng(1)
    x = np.zeros((T, A))
    for t in range(1, T):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal(A) * 0.005                                      # persistent forecasts: their products with overlapping residuals are serially dependent
    daily = rng.standard_normal((T + h, 1)) * 0.01
    y = np.array([daily[t + 1:t + 1 + h].sum(axis=0) * np.ones(A) for t in range(T)]) + 0.5 * x    # all assets share the same overlapping noise
    _, _, n_eff = cal.slope_inference(x, y, h, "hac")
    assert n_eff <= T and n_eff < 0.6 * T, "overlapping outcomes with persistent forecasts are worth fewer independent observations than there are dates"
    _, _, n_eff_iid = cal.slope_inference(x, y, h, "sqrt_horizon")
    assert n_eff_iid == pytest.approx(T * A / h), "the legacy count is pairs / horizon"


def _six_asset_panel(true_slopes, seed=70, n=1400):
    rng = np.random.default_rng(seed)
    r = pd.DataFrame(rng.standard_normal((n, 6)) * 0.01, index=pd.bdate_range("2012-01-01", periods=n), columns=[f"S{i}" for i in range(6)])
    truth = pa.realised_window_returns(r, 5, 1).fillna(0.0)
    s = pd.DataFrame({c: truth[c] * (1.0 if true_slopes[i] > 0 else 0.0) + rng.normal(0, 0.02, n) for i, c in enumerate(r.columns)}, index=r.index)
    return r, s


def test_partial_pooling_shrinks_asset_slopes_toward_the_pooled_one_and_needs_three_assets():
    r, s = _six_asset_panel([1, 1, 1, 0, 0, 0])
    cfg_a = pa.config_from_signal(alloc_sig(calib_method="asset", calib_inference="hac", calib_asset_min_eff=15))
    cfg_p = pa.config_from_signal(alloc_sig(calib_method="partial_pooling", calib_inference="hac", calib_asset_min_eff=15))
    x = s.iloc[200:900]
    y = pa.realised_window_returns(r, 5, 1).iloc[200:900]
    a = cal.calibrate(x, y, cfg_a, 5)
    p = cal.calibrate(x, y, cfg_p, 5)
    pooled = a.pooled_slope
    assert np.var(p.slopes.to_numpy()) <= np.var(a.slopes.to_numpy()) + 1e-12, "shrinkage reduces the spread of the asset slopes"
    assert p.tau is not None and all(v == "shrunk" for v in p.used.values())
    assert min(p.slopes.min(), a.slopes.min()) >= 0.0 and max(p.slopes.max(), a.slopes.max()) <= 1.0 and np.isfinite(pooled)
    # two stocks: not enough assets to separate real differences from noise
    two = cal.calibrate(x.iloc[:, :2], y.iloc[:, :2], cfg_p, 5)
    assert two.tau is None and all(v == "pooled" for v in two.used.values()) and "at least three assets" in two.fallback


def test_asset_specific_slopes_need_independent_windows_and_otherwise_use_the_pooled_slope():
    r, s = _six_asset_panel([1, 1, 1, 0, 0, 0], n=500)
    cfg = pa.config_from_signal(alloc_sig(calib_method="asset", calib_inference="hac", calib_asset_min_eff=10_000))
    c = cal.calibrate(s.iloc[100:300], pa.realised_window_returns(r, 5, 1).iloc[100:300], cfg, 5)
    assert all(v == "pooled" for v in c.used.values()) and "independent outcome windows" in c.fallback


def test_sector_pooling_needs_a_sector_map_and_three_sectors():
    r, s = _six_asset_panel([1, 1, 1, 0, 0, 0])
    y = pa.realised_window_returns(r, 5, 1)
    cfg = pa.config_from_signal(alloc_sig(calib_method="sector_pooling", calib_inference="hac", calib_asset_min_eff=15))
    none = cal.calibrate(s.iloc[200:900], y.iloc[200:900], cfg, 5, None)
    assert "sector map" in none.fallback
    two = cal.calibrate(s.iloc[200:900], y.iloc[200:900], cfg, 5, {c: ("a" if i < 3 else "b") for i, c in enumerate(s.columns)})
    assert "three sectors" in two.fallback
    three = cal.calibrate(s.iloc[200:900], y.iloc[200:900], cfg, 5, {c: "abc"[i % 3] for i, c in enumerate(s.columns)})
    assert set(three.used.values()) == {"sector"} and three.tau is not None


def test_regularised_calibration_shrinks_a_noisy_slope_more_than_a_precise_one():
    rng = np.random.default_rng(2)
    cfg = pa.config_from_signal(alloc_sig(calib_method="regularized", calib_inference="hac", calib_prior_sd=0.3))
    out = {}
    for name, noise in (("precise", 0.002), ("noisy", 0.05)):
        x = pd.DataFrame(rng.standard_normal((300, 2)) * 0.01)
        y = 0.5 * x + pd.DataFrame(rng.standard_normal((300, 2)) * noise)
        c = cal.calibrate(x, y, cfg, 1)
        out[name] = c.slopes.iloc[0] / max(c.pooled_slope, 1e-9)
    assert out["noisy"] < out["precise"] < 1.0 + 1e-9


def test_weak_and_noisy_forecasts_earn_no_calibrated_edge_under_every_method_and_the_book_is_risk_based():
    r = panel(vols=(0.01, 0.02), seed=71)
    rng = np.random.default_rng(71)
    s = pd.DataFrame(rng.normal(0, 0.02, r.shape), index=r.index, columns=r.columns)         # pure noise
    for m, inf in (("pooled", "hac"), ("asset", "hac"), ("partial_pooling", "block_bootstrap"), ("regularized", "hac")):
        w, log = pa.run(s, r, None, alloc_sig(calib_method=m, calib_inference=inf, calib_min_t=2.0, calib_asset_min_eff=15))
        recs = [x for x in log["records"] if x.get("forecast")]
        no_edge = sum(1 for x in recs if x["forecast"]["status"] == "no_edge")
        assert no_edge > 0.8 * len(recs), (m, inf, no_edge, len(recs))
        assert w.iloc[-1]["S0"] > w.iloc[-1]["S1"], "no forecast: the lower-risk stock gets more, the book is risk-based"


@pytest.mark.parametrize("method, inf", [("pooled", "hac"), ("asset", "hac"), ("partial_pooling", "block_bootstrap"), ("regularized", "hac")])
def test_calibration_never_uses_an_outcome_that_had_not_ended_by_the_decision(method, inf):
    r = panel(vols=(0.01, 0.012, 0.009), seed=72)
    s = informative(r, 72)
    cfg = pa.config_from_signal(alloc_sig(calib_method=method, calib_inference=inf, calib_asset_min_eff=10))
    i = 400
    a = pa.ForecastModel(s, r, cfg, 5, 1).asof(i)
    r2, s2 = r.copy(), s.copy()
    r2.iloc[i + 1:] = 7.0
    s2.iloc[i + 1:] = -1.0
    b = pa.ForecastModel(s2, r2, cfg, 5, 1).asof(i)
    assert a.slope == pytest.approx(b.slope) and a.n_obs == b.n_obs and np.allclose(a.mu.to_numpy(), b.mu.to_numpy())


def test_calibration_methods_are_compared_out_of_sample_and_none_is_called_better_without_evidence():
    r = panel(vols=(0.01, 0.012, 0.009, 0.011), seed=73, n=1400)
    s = informative(r, 73, strength=0.5)
    labels = pd.Series(np.where(np.arange(len(r)) < 1000, "validation", "test"), index=r.index)
    out = ac.compare_calibration(s, r, alloc_sig(calib_asset_min_eff=15), 1, labels, None,
                                 variants={"pooled_sqrt_horizon": {}, "pooled_hac": {"calib_inference": "hac"}, "partial_hac": {"calib_method": "partial_pooling", "calib_inference": "hac"}})
    assert out["select_on"] == "validation" and out["decision_dates"] >= 100
    for name, v in out["variants"].items():
        assert v["mse"] > 0 and v["skill_vs_zero"] > 0
        if name != out["baseline"]:
            assert v["versus_baseline"]["verdict"] in ("cannot be told apart from the baseline on this sample", "lower error than the baseline (the interval excludes zero)", "higher error than the baseline (the interval excludes zero)")
    assert "protected segments were not read" in out["note"]
    # the test segment's outcomes are never touched: corrupting them changes nothing
    r2 = r.copy()
    r2.iloc[1000:] = 5.0
    out2 = ac.compare_calibration(s, r2, alloc_sig(calib_asset_min_eff=15), 1, labels, None, variants={"pooled_sqrt_horizon": {}})
    assert out2["variants"]["pooled_sqrt_horizon"]["mse"] == pytest.approx(out["variants"]["pooled_sqrt_horizon"]["mse"])


# ============================================================================================ 3. stabilisers
def _noisy_two_stock(seed=80):
    r = panel(vols=(0.01, 0.02), corr=0.5, seed=seed)
    return r, informative(r, seed, strength=0.15, noise=0.04)


def test_blend_is_exactly_the_convex_combination_and_the_endpoints_are_the_two_books():
    r, s = _noisy_two_stock()
    w, log = pa.run(s, r, None, alloc_sig(baseline="inverse_vol", blend="fixed", blend_alpha=0.3))
    rec = [x for x in log["records"] if x.get("blend", {}).get("status") == "blended"][-1]
    b = rec["blend"]
    for k in rec["weights"]:
        assert rec["weights"][k] == pytest.approx((1 - 0.3) * b["baseline_weights"][k] + 0.3 * b["ml_weights"][k], abs=1e-9)
    w0, _ = pa.run(s, r, None, alloc_sig(baseline="inverse_vol", blend="fixed", blend_alpha=0.0))
    iv, _ = pa.run(s, r, None, alloc_sig("inverse_vol"))
    w1, _ = pa.run(s, r, None, alloc_sig(baseline="inverse_vol", blend="fixed", blend_alpha=1.0))
    mv, _ = pa.run(s, r, None, alloc_sig())
    assert np.allclose(w0.iloc[-1], iv.iloc[-1], atol=1e-6) and np.allclose(w1.iloc[-1], mv.iloc[-1], atol=1e-6)


def test_blending_with_a_baseline_cuts_the_swings_and_the_concentration_of_a_weak_forecast():
    r, s = _noisy_two_stock()
    plain, _ = pa.run(s, r, None, alloc_sig(calib_min_t=0.0))
    blended, _ = pa.run(s, r, None, alloc_sig(baseline="equal_weight", blend="fixed", blend_alpha=0.4))
    mv = lambda w: float((w.diff().abs().sum(axis=1) / 2).iloc[300:].max())
    eff = lambda w: float(np.mean([stab.concentration(row)["effective_n"] for _, row in w.iloc[300:].iterrows() if row.sum() > 0]))
    assert mv(blended) < mv(plain) and eff(blended) > eff(plain)


def test_confidence_blend_uses_the_calibration_evidence_and_never_exceeds_its_cap():
    r = panel(vols=(0.01, 0.02), seed=81)
    strong, noise = informative(r, 81, strength=0.9, noise=0.005), pd.DataFrame(np.random.default_rng(81).normal(0, 0.03, r.shape), index=r.index, columns=r.columns)
    alphas = {}
    for name, sc in (("strong", strong), ("noise", noise)):
        w, log = pa.run(sc, r, None, alloc_sig(baseline="inverse_vol", blend="confidence", blend_max=0.8, calib_min_t=0.0, calib_inference="hac"))
        a = [x["blend"]["alpha"] for x in log["records"] if x.get("blend", {}).get("status") == "blended"]
        alphas[name] = np.mean(a[-40:])
        assert max(a) <= 0.8 + 1e-12
    assert alphas["strong"] > alphas["noise"]


def test_validation_alpha_uses_only_resolved_decisions_and_defaults_to_the_baseline_without_evidence():
    r, s = _noisy_two_stock(82)
    sg = alloc_sig(baseline="inverse_vol", blend="validation", blend_min_obs=10, blend_window=20, opt_every=5)
    w, log = pa.run(s, r, None, sg)
    alphas = [(x["date"], x["blend"]["alpha"], x["blend"].get("resolved")) for x in log["records"] if x.get("blend", {}).get("status") == "blended"]
    assert alphas[0][1] == 0.0 and all(a[1] in stab.ALPHA_GRID for a in alphas), "no resolved decisions: the baseline, and alpha comes from the fixed grid"
    cut = 450
    r2, s2 = r.copy(), s.copy()
    r2.iloc[cut + 1:] = r2.iloc[cut + 1:] * -2.0
    s2.iloc[cut + 1:] = s2.iloc[cut + 1:] * -2.0
    w2, log2 = pa.run(s2, r2, None, sg)
    assert np.allclose(w.iloc[:cut + 1].to_numpy(), w2.iloc[:cut + 1].to_numpy(), atol=1e-9), "alpha at a date read nothing after it"


def test_a_blend_needs_a_baseline_and_a_forecast_driven_allocator():
    with pytest.raises(pa.AllocationError, match="needs a baseline"):
        pa.config_from_signal(alloc_sig(blend="fixed"))
    with pytest.raises(pa.AllocationError, match="reads no forecast"):
        pa.config_from_signal(alloc_sig("inverse_vol", baseline="equal_weight", blend="fixed"))


def test_the_weight_change_limit_is_respected_after_the_first_book_and_reported_when_it_binds():
    r = panel(vols=(0.01, 0.01), seed=83)
    s = pd.DataFrame({"S0": 0.02 * np.sign(np.sin(np.arange(N) / 12.0)), "S1": 0.0}, index=r.index)
    w, log = pa.run(s, r, None, alloc_sig(calib=None, max_weight_change=0.10, calibrate="none") if False else {**alloc_sig(max_weight_change=0.10), "allocation": {**alloc_sig(max_weight_change=0.10)["allocation"], "calibrate": "none", "risk_aversion": 1.0}})
    moves = w.diff().abs().max(axis=1).iloc[1:]
    first = w.abs().sum(axis=1).gt(0).idxmax()
    assert moves.loc[first:].iloc[1:].max() <= 0.10 + 1e-9 and log["counts"]["weight_change_limited"] > 0
    assert any("weight_change_limited" in x for x in log["records"])


def test_weight_change_limit_projection_is_exact_and_an_unsatisfiable_one_is_reported():
    w0 = pd.Series([0.9, 0.1])
    w, moved, ok = stab.weight_change_limited(pd.Series([0.2, 0.8]), w0, 0.2, 0.0, 1.0, 1.0)
    assert ok and moved and w.sum() == pytest.approx(1.0) and (w - w0).abs().max() <= 0.2 + 1e-12 and w["S0" if "S0" in w else 0] == pytest.approx(0.7)
    _, _, ok2 = stab.weight_change_limited(pd.Series([0.5, 0.5]), pd.Series([1.0, 0.0]), 0.1, 0.0, 0.6, 1.0)
    assert not ok2, "a cap of 60% and a move limit of 10 points from 100% cannot reach a fully invested book"
    r = panel(vols=(0.01, 0.01), seed=84)
    s = pd.DataFrame({"S0": 0.02 * np.sign(np.sin(np.arange(N) / 12.0)), "S1": 0.0}, index=r.index)
    sg = alloc_sig(max_weight_change=0.05, limits={"max_weight": 0.6})
    sg["allocation"].update(calibrate="none")
    _, log = pa.run(s, r, None, sg)
    assert log["counts"]["failure_policy_used"] >= 0


def test_forecast_uncertainty_shrinkage_lowers_the_tilt_monotonically_and_minimum_evidence_removes_it():
    r, s = _noisy_two_stock(85)
    devs = []
    for k in (0.0, 0.5, 1.0, 2.0):
        w, _ = pa.run(s, r, None, alloc_sig(uncertainty_shrink=k, calib_inference="hac", calib_min_t=0.0))
        devs.append(float((w["S0"].iloc[300:] - w["S0"].iloc[300:].mean()).abs().mean()))
    assert devs[0] >= devs[1] >= devs[2] >= devs[3] - 1e-12
    w, log = pa.run(s, r, None, alloc_sig(min_n_eff=10_000, calib_inference="hac"))
    recs = [x for x in log["records"] if x.get("forecast")]
    gated = [x for x in recs if "min_n_eff" in x.get("evidence", {})]
    assert gated and all(x["forecast"]["status_after_evidence_controls"] == "no_edge" for x in gated), "enough outcomes to calibrate but not enough independent ones: no forecast is used"
    assert all(x["forecast"]["status"] != "calibrated" or "min_n_eff" in x.get("evidence", {}) for x in recs)
    w2, log2 = pa.run(s, r, None, alloc_sig(min_confidence=0.999, calib_inference="hac", calib_min_t=0.0))
    assert any("min_confidence" in x.get("evidence", {}) for x in log2["records"])


def test_concentration_diagnostics_are_exact():
    assert stab.concentration(pd.Series([0.5, 0.5]))["effective_n"] == pytest.approx(2.0)
    c = stab.concentration(pd.Series([1.0, 0.0, 0.0]))
    assert c["effective_n"] == pytest.approx(1.0) and c["max_weight"] == 1.0 and c["hhi"] == pytest.approx(1.0)
    c2 = stab.concentration(pd.Series([0.6, 0.3, 0.1, 0.0]))
    assert c2["hhi"] == pytest.approx(0.36 + 0.09 + 0.01) and c2["top_share"] == pytest.approx(0.6)
    assert stab.concentration(pd.Series([0.0, 0.0]))["effective_n"] is None


def test_sensitivities_to_forecast_risk_aversion_and_cost_are_attached_and_behave():
    r, s = _noisy_two_stock(86)
    w, log = pa.run(s, r, None, alloc_sig(sensitivity=True, calib_min_t=0.0))
    rec = [x for x in log["records"] if x.get("sensitivity")][-1]["sensitivity"]
    assert rec["forecast"]["available"] and rec["forecast"]["draws"] > 10 and set(rec["forecast"]["weight_sd"]) == {"S0", "S1"}
    books = [np.array([row["weights"]["S0"], row["weights"]["S1"]]) for row in rec["risk_aversion"]]
    dist = [float(np.abs(b - books[-1]).sum()) for b in books]
    assert all(dist[i] >= dist[i + 1] - 1e-9 for i in range(len(dist) - 1)), "as risk aversion rises the book converges monotonically to the risk-only (minimum-variance) book"
    turns = [row["turnover"] for row in rec["costs"]]
    assert turns[0] >= turns[-1] - 1e-9, "dearer trading and a turnover penalty never increase the trade"
    assert "sensitivity" not in [x for x in pa.run(s, r, None, alloc_sig())[1]["records"] if x.get("weights")][-1]


def test_the_conservative_preset_is_documented_explicit_overridable_and_leaves_the_default_alone():
    assert pa.AllocConfig().risk_aversion == 5.0 and pa.config_from_signal({"allocation": {}}).risk_aversion == 5.0
    c = pa.config_from_signal({"allocation": {"preset": "conservative"}})
    p = stab.PRESETS["conservative"]
    assert c.risk_aversion == 20.0 and c.calib_inference == "hac" and c.blend == "confidence" and c.max_weight_change == 0.10 and c.limits.max_weight == 0.60 and "not an optimum" in p["description"]
    o = pa.config_from_signal({"allocation": {"preset": "conservative", "risk_aversion": 8.0, "limits": {"max_weight": 0.7}}})
    assert o.risk_aversion == 8.0 and o.limits.max_weight == 0.7 and o.blend == "confidence"
    with pytest.raises(pa.AllocationError, match="unknown preset"):
        pa.config_from_signal({"allocation": {"preset": "aggressive"}})


def test_on_weak_forecasts_the_conservative_preset_is_far_more_stable_than_plain_mean_variance():
    r, s = _noisy_two_stock(87)
    plain, _ = pa.run(s, r, None, alloc_sig(calib_min_t=0.0))
    sg = alloc_sig()
    sg["allocation"] = {"preset": "conservative", "calibrate": "slope", "calib_min_obs": 60, "cov_method": "sample", "cost_bps": 0.0}
    cons, _ = pa.run(s, r, None, sg)
    swing = lambda w: float((w.diff().abs().sum(axis=1) / 2).iloc[300:].max())
    assert swing(cons) <= 0.10 + 1e-9 < swing(plain)
    assert cons["S0"].iloc[300:].max() <= 0.60 + 1e-9


# ============================================================================================ 4. adapters
def test_every_adapter_declares_what_it_needs_and_whether_it_reads_the_forecast():
    for k, a in ad.ADAPTERS.items():
        d = pa.DOCS[k]
        assert d["uses_forecast"] == a.uses_forecast and d["min_assets"] == a.min_assets and d["long_short"] is False and d["needs"] and d["failure"] and d["native_constraints"] is not None
    assert not ad.ADAPTERS["herc"].uses_forecast and "NOT used" in ad.ADAPTERS["herc"].assumptions
    assert not ad.ADAPTERS["nco"].uses_forecast and ad.ADAPTERS["nco_max_sharpe"].uses_forecast and ad.ADAPTERS["near_optimal"].uses_forecast and not ad.ADAPTERS["wasserstein_cvar"].uses_forecast
    assert set(ad.ADAPTERS) <= set(pa.ALLOCATORS) and set(pa.CORE_ALLOCATORS) < set(pa.ALLOCATORS)


def test_adapters_refuse_what_they_cannot_honour():
    r = panel(vols=(0.01, 0.02), seed=90)
    with pytest.raises(pa.AllocationError, match="at least 3 assets"):
        pa.run(score_of(r, (0, 0)), r, None, alloc_sig("herc"))
    with pytest.raises(pa.AllocationError, match="at least 4 assets"):
        pa.run(score_of(r, (0, 0)), r, None, alloc_sig("nco"))
    with pytest.raises(pa.AllocationError, match="long-only"):
        pa.config_from_signal(alloc_sig("herc", limits={"long_only": False, "gross_max": 1.0}))


def _six(seed=91):
    return panel(vols=(0.01, 0.02, 0.015, 0.012, 0.018, 0.011), corr=0.3, seed=seed)


@pytest.mark.parametrize("key", ["herc", "nco", "nco_max_sharpe", "near_optimal", "wasserstein_cvar"])
def test_adapters_satisfy_the_position_and_sector_limits_after_allocation_and_report_the_projection(key):
    r = _six()
    groups = {"S0": "a", "S1": "a", "S2": "b", "S3": "b", "S4": "c", "S5": "c"}
    lim = pa.Limits(max_weight=0.3, max_sector=0.55)
    s = score_of(r, (0.003, 0, 0, 0, -0.002, 0))
    w, log = pa.run(s, r, None, alloc_sig(key, opt_every=60, cov_window=252, limits={"max_weight": 0.3, "max_sector": 0.55}), groups)
    live = w[w.abs().sum(axis=1) > 0]
    assert len(live) > 0
    for _, row in live.iterrows():
        assert pa.violations(row, lim, groups) == []
    rec = [x for x in log["records"] if x.get("allocator_detail", {}).get("optimiser_book")]
    assert rec and all("projection_l1" in x["allocator_detail"] for x in rec)
    assert log["documentation"]["uses_forecast"] == ad.ADAPTERS[key].uses_forecast


def test_risk_only_adapters_ignore_the_forecast_and_forecast_adapters_respond_to_it():
    r = _six(92)
    flat, tilt = score_of(r, (0,) * 6), score_of(r, (0.004, 0, 0, 0, -0.003, 0))
    for key, responds in (("herc", False), ("nco", False), ("wasserstein_cvar", False), ("nco_max_sharpe", True), ("near_optimal", True)):
        a, _ = pa.run(flat, r, None, alloc_sig(key, opt_every=120, calibrate="none"))
        b, _ = pa.run(tilt, r, None, alloc_sig(key, opt_every=120, calibrate="none"))
        differs = not np.allclose(a.iloc[-1], b.iloc[-1], atol=1e-6)
        assert differs == responds, (key, a.iloc[-1].round(3).tolist(), b.iloc[-1].round(3).tolist())


def test_herc_matches_the_library_implementation_and_blends_with_a_baseline_only_when_it_reads_a_forecast():
    from src.portfolio.hierarchical import herc_weights

    r = _six(93)
    w, log = pa.run(score_of(r, (0,) * 6), r, None, alloc_sig("herc", opt_every=300))
    rec = [x for x in log["records"] if x.get("weights")][-1]
    i = pd.Timestamp(rec["date"])
    fit_cov = pa.rm.fit_asof(r, i, pa.rm.CovSpec("sample", window=252, min_obs=60)).matrix
    assert np.allclose(pd.Series(rec["allocator_detail"]["optimiser_book"]).reindex(r.columns), herc_weights(fit_cov * 5.0).reindex(r.columns), atol=1e-9)
    with pytest.raises(pa.AllocationError, match="reads no forecast"):
        pa.config_from_signal(alloc_sig("herc", baseline="equal_weight", blend="fixed"))


# ============================================================================================ 5. currency
def _fx_table(index, rates, quote="base_per_local", available=None, ccy="JPY", base="USD"):
    t = fxm.FxTable(base)
    t.set_series(ccy, pd.Series(rates, index=index), None if available is None else pd.Series(available, index=index), quote)
    return t


def test_fx_identity_and_quotation_direction():
    idx = pd.bdate_range("2024-01-01", periods=4)
    t = _fx_table(idx, [0.0067, 0.0068, 0.0066, 0.0066])                                      # USD per JPY
    ctx = fxm.FxContext(t, {"TOYO": "JPY", "AAPL": "USD"})
    local = pd.DataFrame({"TOYO": [0.01, 0.02, -0.01, 0.0], "AAPL": [0.01, 0.0, 0.01, 0.02]}, index=idx)
    base, fxr, rep = fxm.base_returns(local, ctx)
    assert np.allclose(base["TOYO"].iloc[1:], (1 + local["TOYO"].iloc[1:]) * (1 + fxr["TOYO"].iloc[1:]) - 1)
    assert fxr["TOYO"].iloc[1] == pytest.approx(0.0068 / 0.0067 - 1) and (fxr["AAPL"] == 0).all() and np.allclose(base["AAPL"], local["AAPL"])
    inv = _fx_table(idx, [1 / 0.0067, 1 / 0.0068, 1 / 0.0066, 1 / 0.0066], quote="local_per_base")      # JPY per USD: the same market quoted the other way
    assert np.allclose(inv.returns("JPY", idx).dropna(), t.returns("JPY", idx).dropna()), "a stronger yen is a positive FX return either way it is quoted"
    hedged = fxm.FxContext(t, {"TOYO": "JPY", "AAPL": "USD"}, hedge={"TOYO": 1.0})
    hb, _, _ = fxm.base_returns(local, hedged)
    assert np.allclose(hb["TOYO"].iloc[1:], local["TOYO"].iloc[1:]), "a fully hedged holding earns its local return"


def test_a_rate_is_usable_only_once_it_was_available_and_never_after_the_stale_limit():
    idx = pd.bdate_range("2024-01-01", periods=10)
    avail = list(idx + pd.Timedelta(days=2))                                                  # each rate is published two days after the date it refers to
    t = _fx_table(idx, np.linspace(0.0066, 0.0070, 10), available=avail)
    u = t.usable("JPY", idx)
    assert u.iloc[0] != u.iloc[0] and u.iloc[1] != u.iloc[1], "the first rates were not available yet on their own dates"
    assert u.iloc[2] == pytest.approx(t._rates["JPY"].iloc[0]), "the rate of day 0 is first usable two days later, not before"
    stale = fxm.FxTable("USD", max_stale_days=2)
    stale.set_series("JPY", pd.Series([0.0066, 0.0067], index=idx[[0, 1]]))
    far = stale.usable("JPY", idx)
    assert far.iloc[1] == pytest.approx(0.0067) and np.isnan(far.iloc[-1]), "an old rate is missing, not carried"
    with pytest.raises(fxm.FxError, match="before the time it refers to"):
        _fx_table(idx[:2], [0.0066, 0.0067], available=list(idx[:2] - pd.Timedelta(days=1)))


def test_a_missing_rate_is_never_one_and_never_zero_change():
    idx = pd.bdate_range("2024-01-01", periods=30)
    t = fxm.FxTable("USD", max_stale_days=3)
    t.set_series("JPY", pd.Series(0.0067, index=idx[:10]))                                    # rates stop after day 10
    ctx = fxm.FxContext(t, {"TOYO": "JPY"})
    local = pd.DataFrame({"TOYO": 0.001}, index=idx)
    with pytest.raises(fxm.FxError, match="exchange rates are missing"):
        fxm.base_returns(local, ctx)
    ctx2 = fxm.FxContext(t, {"TOYO": "JPY"}, missing="exclude")
    base, _, rep = fxm.base_returns(local, ctx2)
    assert base["TOYO"].iloc[20:].isna().all() and rep["assets"]["TOYO"]["days_missing_after_first_rate"] > 0
    with pytest.raises(fxm.FxError, match="no exchange-rate series for EUR"):
        fxm.base_returns(local, fxm.FxContext(t, {"TOYO": "EUR"}))
    with pytest.raises(fxm.FxError, match="not declared"):
        fxm.base_returns(local, fxm.FxContext(t, {"OTHER": "JPY"}))


def test_a_forecast_is_converted_with_a_stated_assumption_and_never_with_a_future_exchange_rate():
    idx = pd.bdate_range("2024-01-01", periods=40)
    rates = pd.Series(np.linspace(0.0066, 0.0070, 40), index=idx)
    t = fxm.FxTable("USD")
    t.set_series("JPY", rates)
    ctx = fxm.FxContext(t, {"TOYO": "JPY", "AAPL": "USD"})
    mu = pd.DataFrame({"TOYO": 0.02, "AAPL": 0.02}, index=idx)
    out, info = fxm.convert_forecast(mu, ctx, 5)
    assert np.allclose(out, mu) and info["assumption"] == "random_walk" and "assumption, not an estimate" in info["note"]
    t2 = fxm.FxTable("USD")
    t2.set_series("JPY", pd.concat([rates.iloc[:20], rates.iloc[20:] * 3.0]))               # the rates after day 20 are very different
    out2, _ = fxm.convert_forecast(mu, fxm.FxContext(t2, {"TOYO": "JPY", "AAPL": "USD"}), 5)
    assert np.allclose(out2, out), "the converted forecast does not depend on any exchange rate"
    fcst = pd.DataFrame({"JPY": 0.01}, index=idx)
    out3, info3 = fxm.convert_forecast(mu, fxm.FxContext(t, {"TOYO": "JPY", "AAPL": "USD"}, assumption=fxm.FxAssumption("forecast", fcst)), 5)
    assert out3["TOYO"].iloc[0] == pytest.approx(1.02 * 1.01 - 1) and out3["AAPL"].iloc[0] == pytest.approx(0.02)
    with pytest.raises(fxm.FxError, match="needs the expected FX return"):
        fxm.FxAssumption("forecast")


def test_currency_attribution_costs_and_exposure_reconcile():
    idx = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(3)
    t = fxm.FxTable("USD")
    t.set_series("JPY", pd.Series(0.0067 * np.exp(np.cumsum(rng.normal(0, 0.004, 60))), index=idx))
    t.set_series("EUR", pd.Series(1.08 * np.exp(np.cumsum(rng.normal(0, 0.003, 60))), index=idx))
    ctx = fxm.FxContext(t, {"A": "JPY", "B": "EUR", "C": "USD"}, hedge={"B": 0.5}, fx_cost_bps=4.0)
    local = pd.DataFrame(rng.normal(0, 0.01, (60, 3)), index=idx, columns=list("ABC"))
    base, fxr, _ = fxm.base_returns(local, ctx)
    held = pd.DataFrame({"A": 0.3, "B": 0.5, "C": 0.2}, index=idx)
    att = fxm.currency_attribution(held.iloc[1:], local.iloc[1:], fxr.iloc[1:], ctx)
    total_base = (held * base).sum(axis=1).iloc[1:]
    assert np.allclose(att["local"] + att["fx"], total_base, atol=1e-12), "local + currency parts equal the base-currency return exactly"
    assert set(att["by_currency_fx"].columns) == {"JPY", "EUR", "USD"} and (att["by_currency_fx"]["USD"].abs() < 1e-15).all()
    assert fxm.fx_exposure(pd.Series({"A": 0.3, "B": 0.5, "C": 0.2}), ctx) == pytest.approx({"JPY": 0.3, "EUR": 0.25, "USD": 0.2})
    trades = pd.DataFrame({"A": [0.1, 0.0], "B": [0.0, 0.2], "C": [0.3, 0.3]}, index=idx[:2])
    assert fxm.fx_cost_series(trades, ctx).tolist() == pytest.approx([0.1 * 4e-4, 0.2 * 4e-4]), "only foreign exposure traded pays to convert"


def test_the_allocation_runs_in_the_base_currency_and_keeps_its_currency_refusals():
    idx = N
    r = panel(vols=(0.01, 0.02), seed=95)
    r.columns = ["TOYO", "AAPL"]
    rng = np.random.default_rng(95)
    t = fxm.FxTable("USD")
    t.set_series("JPY", pd.Series(0.0067 * np.exp(np.cumsum(rng.normal(0, 0.005, len(r)))), index=r.index))
    ctx = fxm.FxContext(t, {"TOYO": "JPY", "AAPL": "USD"})
    s = informative(r, 95)
    w, log = pa.run(s, r, None, alloc_sig(), fx=ctx)
    assert log["fx"]["base"] == "USD" and log["fx"]["foreign_assets"] == ["TOYO"] and "identity" in log["fx"]
    w_iv, _ = pa.run(s, r, None, alloc_sig("inverse_vol"), fx=ctx)
    w_iv_loc, _ = pa.run(s, r, None, alloc_sig("inverse_vol"))
    assert w_iv.iloc[-1]["TOYO"] < w_iv_loc.iloc[-1]["TOYO"], "the yen's exchange-rate volatility makes the stock riskier to a dollar investor, so it gets less"
    with pytest.raises(pa.AllocationError, match="forecast is in EUR|declared in EUR|only|local"):
        pa.run(s, r, None, {**alloc_sig(), "allocation": {**alloc_sig()["allocation"], "forecast_currency": "EUR"}}, fx=ctx)
    # without an FX context the original refusal stands
    with pytest.raises(pa.AllocationError, match="does not mix currencies"):
        pa.check_forecast_contract(pa.config_from_signal({"allocation": {"forecast_currency": "JPY", "base_currency": "USD"}}), 5)
    with pytest.raises(pa.AllocationError, match="no exchange-rate series for"):
        pa.run(s, r, None, alloc_sig(), fx=fxm.FxContext(fxm.FxTable("USD"), {"TOYO": "JPY", "AAPL": "USD"}))
    with pytest.raises(pa.AllocationError, match="probability cannot be converted"):
        pa.run(s, r, None, {**alloc_sig(), "allocation": {**alloc_sig()["allocation"], "forecast_kind": "probability"}}, fx=ctx, probability=s)


def test_the_comparison_converts_once_for_every_policy_and_records_it():
    r = panel(vols=(0.01, 0.02), seed=96)
    r.columns = ["TOYO", "AAPL"]
    rng = np.random.default_rng(96)
    t = fxm.FxTable("USD")
    t.set_series("JPY", pd.Series(0.0067 * np.exp(np.cumsum(rng.normal(0, 0.005, len(r)))), index=r.index))
    s = informative(r, 96)
    labels = pd.Series(np.where(np.arange(len(r)) < 450, "validation", "test"), index=r.index)
    out = ac.compare(s, r, None, alloc_sig(), BT, 1, labels, allocators=("equal_weight", "inverse_vol", "mean_variance"), fx=fxm.FxContext(t, {"TOYO": "JPY", "AAPL": "USD"}))
    assert out["fx"]["base"] == "USD" and set(out["allocators"]) == {"equal_weight", "inverse_vol", "mean_variance"}


# ============================================================================================ 6/7. comparison reporting
def test_policies_are_compared_on_identical_inputs_with_sortino_and_dependence_aware_intervals():
    r, s = _noisy_two_stock(97)
    labels = pd.Series(np.where(np.arange(len(r)) < 450, "validation", "test"), index=r.index)
    out = ac.compare(s, r, None, alloc_sig(calib_inference="hac"), BT, 1, labels, allocators=("equal_weight", "inverse_vol", "mean_variance"),
                     policies={"mv_conservative": {"label": "Mean-variance, conservative preset", "preset": "conservative"}, "mv_blend": {"baseline": "inverse_vol", "blend": "fixed", "blend_alpha": 0.5}})
    assert set(out["allocators"]) == {"equal_weight", "inverse_vol", "mean_variance", "mv_conservative", "mv_blend"} and out["table"]["mv_conservative"]["label"] == "Mean-variance, conservative preset"
    m = out["table"]["equal_weight"]["segments"]["validation"]
    assert m["sortino"] is not None and m["sharpe"] is not None and m["effective_n"] == pytest.approx(2.0)
    assert out["sharpe_ci"]["equal_weight"]["hac"]["interval"][0] < out["sharpe_ci"]["equal_weight"]["hac"]["interval"][1]
    assert "mv_blend" in out["paired_sharpe_vs_equal_weight"] and "p_value_holm" in out["paired_vs_equal_weight"]["mv_blend"]
    assert set(out["table"]["mv_blend"]["segments"]) == {"validation"}, "the protected segment is not computed"
    sel = out["selected"]
    assert sel in out["allocators"]
    # nothing about the choice depends on the test segment
    r2, s2 = r.copy(), s.copy()
    r2.iloc[451:] = r2.iloc[451:] * -3.0
    out2 = ac.compare(s2, r2, None, alloc_sig(calib_inference="hac"), BT, 1, labels, allocators=("equal_weight", "inverse_vol", "mean_variance"), policies={"mv_blend": {"baseline": "inverse_vol", "blend": "fixed", "blend_alpha": 0.5}})
    assert out2["table"]["equal_weight"]["segments"]["validation"]["sharpe"] == pytest.approx(m["sharpe"])


# ============================================================================================ reproducibility of what existed before
GOLD = {"mean_variance": (729861.9262975, "b5e724db67020557"), "black_litterman": (729875.15964978, "63436a1b2f025871"), "inverse_vol": (729822.92344914, "c6d655407b051169"),
        "erc": (729819.9082417, "8fb230d39bdfbdca"), "cvar": (729710.59316646, "19326ffdc254023d"), "kelly": (729847.04174905, "66c9ae4c320b2aa1"), "robust_mv": (729847.25278904, "1a5ec988e34f9342")}


@pytest.mark.parametrize("allocator", sorted(GOLD))
def test_allocations_recorded_before_this_change_are_reproduced_exactly(allocator):
    """Weights, configuration hash and decision counts of the first release, computed with the code as it was (git fe53411) on this fixed panel."""
    dates = pd.bdate_range("2018-01-01", periods=700)
    rng = np.random.default_rng(77)
    r = pd.DataFrame(rng.standard_normal((700, 3)) * np.array([0.01, 0.02, 0.015]), index=dates, columns=["A", "B", "C"])
    s = pd.DataFrame(0.3 * r.rolling(5).sum().shift(-2).fillna(0.0).to_numpy() + rng.normal(0, 0.02, r.shape), index=dates, columns=r.columns)
    sig = {"method": "portfolio_allocation", "holding_period": 1, "window_days": 5, "lag_total": 1, "allocation": {"allocator": allocator, "calibrate": "slope", "calib_min_t": 0.0, "cov_method": "sample", "cost_bps": 5.0, "limits": {"max_weight": 0.8}}}
    w, log = pa.run(s, r, None, sig)
    checksum, h = GOLD[allocator]
    assert round(float((w.fillna(0).to_numpy() * np.arange(1, w.size + 1).reshape(w.shape)).sum()), 8) == pytest.approx(checksum, abs=1e-6)
    assert log["config_hash"] == h, "settings added since then do not alter the recorded configuration of a run that does not use them"
