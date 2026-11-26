"""The Portfolio Lab's comparison harness: identical windows and costs for every method, honest bookkeeping (fallbacks, the cap, drift between rebalances), and every method and covariance estimator running."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio import lab
from src.portfolio.covariance import estimate_covariance


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(0)
    T, N = 700, 8
    f = rng.standard_normal((T, 2)) * 0.006
    B = rng.standard_normal((N, 2)) * 0.8 + np.array([1.0, 0.0])
    return pd.DataFrame(f @ B.T + rng.standard_normal((T, N)) * 0.01 + 0.0003, index=pd.bdate_range("2018-01-01", periods=T), columns=[f"A{i}" for i in range(N)])


def test_every_method_runs_without_a_fallback_and_the_table_has_what_was_promised(returns):
    res = lab.compare(returns, list(lab.METHODS), lookback=252, step=42, perturb=2)
    t = res["table"]
    assert set(t.index) == set(lab.METHODS) and (t["fallbacks"] == 0).all(), t["fallbacks"].to_dict()
    need = {"ann_return", "ann_vol", "sharpe", "sharpe_se", "max_drawdown", "cdar_95", "turnover_per_year", "effective_n", "diversification_ratio", "capped", "instability", "sharpe_first_half", "sharpe_second_half"}
    assert need <= set(t.columns) and (t["effective_n"].between(1.0, 8.0 + 1e-9)).all()
    for k, w in res["weights"].items():
        assert np.allclose(w.sum(axis=1), 1.0) and (w >= -1e-9).all().all(), k
    assert t.loc["equal_weight", "instability"] == pytest.approx(0.0) and t.loc["max_sharpe", "instability"] > t.loc["min_variance", "instability"] * 0.5
    assert res["equity"].shape[1] == len(lab.METHODS) and (res["equity"].iloc[0] > 0).all() and res["config"]["assets"] == 8


def test_the_accounting_of_drift_and_cost_is_the_obvious_one(returns):
    """Equal weight on a hand-computed schedule: the first rebalance buys from cash (cost on all of it), later ones trade only the drift."""
    lookback, step, bps = 100, 5, 20.0
    res = lab.compare(returns, ["equal_weight"], lookback=lookback, step=step, cost_bps=bps)
    R = returns.to_numpy()
    n = R.shape[1]
    held, net = np.zeros(n), []
    for p in range(lookback, len(R) - 1, step):
        w = np.full(n, 1.0 / n)
        traded = np.abs(w - held).sum()
        cur = w.copy()
        for t in range(p, min(p + step, len(R))):
            g = cur @ R[t]
            net.append(g - (bps * 1e-4 * traded if t == p else 0.0))
            cur = cur * (1 + R[t]) / (1 + g)
        held = cur
    eq = res["equity"]["equal_weight"]
    series = eq.pct_change().fillna(eq.iloc[0] - 1).to_numpy()
    assert len(series) == len(net) and series == pytest.approx(np.array(net), abs=1e-12), "net returns equal the hand-computed drift and cost schedule"
    assert res["table"].loc["equal_weight", "turnover_per_rebalance"] < 0.2


def test_a_failing_method_is_equal_weight_and_the_failures_are_counted(returns, monkeypatch):
    def boom(window, ctx):
        raise RuntimeError("solver failed")

    monkeypatch.setitem(lab.METHODS, "broken", lab.Method("broken", "Broken", "test", False, boom))
    res = lab.compare(returns, ["broken", "equal_weight"], lookback=120, step=21)
    t = res["table"]
    assert t.loc["broken", "fallbacks"] == len(res["rebalances"]) and t.loc["equal_weight", "fallbacks"] == 0
    assert t.loc["broken", "sharpe"] == pytest.approx(t.loc["equal_weight", "sharpe"]), "a method that always fails IS equal weight, and the table says how often it did"


def test_the_cap_binds_the_same_way_for_everyone_and_is_reported(returns):
    res = lab.compare(returns, ["equal_weight", "max_sharpe", "mean_variance"], lookback=252, step=42, max_weight=0.2)
    for w in res["weights"].values():
        assert w.max().max() <= 0.2 + 1e-9 and np.allclose(w.sum(axis=1), 1)
    t = res["table"]
    assert t.loc["equal_weight", "capped"] == 0 and t.loc["mean_variance", "capped"] > 0
    with pytest.raises(lab.LabError, match="cannot be fully invested"):
        lab.compare(returns, ["equal_weight"], max_weight=0.1)


def test_cap_weights_redistributes_exactly():
    w = lab.cap_weights(np.array([0.7, 0.2, 0.05, 0.05]), 0.4)
    assert w.sum() == pytest.approx(1.0) and w.max() == pytest.approx(0.4) and w[0] == pytest.approx(0.4)
    assert w[1] > 0.2 and w[2] == pytest.approx(w[3]), "the excess goes to the others in proportion"
    assert lab.cap_weights(np.array([0.25] * 4), 0.25) == pytest.approx([0.25] * 4)
    with pytest.raises(lab.LabError):
        lab.cap_weights(np.array([0.5, 0.5]), 0.4)


def test_bad_requests_say_what_to_change(returns):
    for kw, word in (({"methods": ["oracle"]}, "unknown method"), ({"cov_method": "magic"}, "cov_method"), ({"lookback": 10}, "lookback"), ({"lookback": 600, "step": 63}, "common rows"), ({"mu_shrink": 2}, "mu_shrink"),
                     ({"cost_bps": -1}, "cost_bps")):
        with pytest.raises(lab.LabError, match=word):
            lab.compare(returns, **{"methods": ["equal_weight"], **kw})
    with pytest.raises(lab.LabError, match="at least 3 assets"):
        lab.compare(returns.iloc[:, :2], ["equal_weight"])


def test_every_covariance_estimator_gives_a_usable_matrix(returns):
    for m in lab.COV_METHODS:
        c = estimate_covariance(returns.iloc[:300], m, 252)
        assert c.shape == (8, 8) and np.linalg.eigvalsh(c.to_numpy()).min() > 0 and np.allclose(c, c.T), m
    assert not np.allclose(estimate_covariance(returns.iloc[:300], "factor", 252), estimate_covariance(returns.iloc[:300], "sample", 252))


# ------------------------------------------------------------------------------------------------------------------ the allocators the Backtest tab can now run
def test_the_new_allocators_build_long_only_books_within_the_configured_limits():
    from src.framework import ALLOCATORS, load_library
    from src.framework.allocation import Context
    from src.utils.config import load_config
    from tests.test_allocators_book_exact import last_origin, market

    load_library()
    bundle, config = market(), load_config()
    cap = float(config.get("portfolio.constraints.max_weight", 0.25))
    for name, kw in (("nco", {}), ("near_optimal", {"tolerance": 0.3}), ("cdar", {"alpha": 0.9})):
        weights = ALLOCATORS.create(name, lookback=252, **kw).build(Context(bundle, config))
        origin, w = last_origin(weights, bundle)
        assert w.sum() == pytest.approx(1.0, abs=1e-6) and (w >= -1e-9).all() and w.max() <= cap + 1e-6, (name, w.round(3).to_dict())
        assert weights.loc[:bundle.index[250]].abs().sum().sum() == 0, "nothing is held before the first full window"
    centre = last_origin(ALLOCATORS.create("near_optimal", lookback=252, tolerance=0.3).build(Context(bundle, config)), bundle)[1]
    assert (centre > 0).all(), "the centre holds every asset"
    for name, kw in (("near_optimal", {"tolerance": 0.0}), ("cdar", {"alpha": 1.0})):
        with pytest.raises(ValueError):
            ALLOCATORS.create(name, **kw)


def test_diagnostics_numbers_are_the_obvious_ones_and_use_the_comparisons_own_conditions(returns):
    """Equal weight is the closed-form case (effective N = 8, HHI = 1/8, no weight turnover, no sensitivity to the inputs); the others satisfy the identities the numbers must obey."""
    keys = ["equal_weight", "min_variance", "max_sharpe", "inverse_vol"]
    res = lab.compare(returns, keys, lookback=252, step=42, max_weight=0.4, cost_bps=15.0)
    d = lab.diagnostics(res, returns, n_draws=8, frontier_points=8)
    m = d["methods"]
    eq = m["equal_weight"]
    assert eq["effective_n"] == pytest.approx(8.0) and eq["hhi"] == pytest.approx(1 / 8) and eq["max_weight"] == pytest.approx(1 / 8) and eq["top3_share"] == pytest.approx(3 / 8)
    assert eq["weight_turnover"] == pytest.approx(0.0, abs=1e-12) and eq["cov_sensitivity"] == pytest.approx(0.0, abs=1e-9) and eq["mu_sensitivity"] is None and eq["uses_expected_returns"] is False
    for k, r in m.items():
        assert 1.0 - 1e-9 <= r["effective_n"] <= 8.0 + 1e-9 and r["hhi"] <= 1.0 and r["max_weight"] <= 0.4 + 1e-6, k   # the cap is honoured in what is reported
        assert sum(r["last_weights"].values()) == pytest.approx(1.0) and r["cost_drag_per_year"] == pytest.approx(r["turnover_per_year"] * 15e-4)
    assert m["max_sharpe"]["mu_sensitivity"] is not None and m["max_sharpe"]["mu_sensitivity"] > 0
    assert m["max_sharpe"]["effective_n"] <= m["equal_weight"]["effective_n"] and m["min_variance"]["weight_turnover"] > 0
    f = d["frontier"]
    vols = [x["vol"] for x in f]
    top = int(np.argmin(vols))
    assert len(f) >= 4 and all(b >= a - 1e-9 for a, b in zip(vols[top:-1], vols[top + 1:])), "above the minimum-variance point, volatility never falls as the required return rises"
    assert [x["return"] for x in f] == sorted(x["return"] for x in f)
    assert d["placed"]["min_variance"]["vol"] <= min(vols) + 1e-6, "the minimum-variance method sits at or below every point of the frontier built under the same cap and covariance"
    assert d["conditions"]["cost_bps"] == 15.0 and d["conditions"]["max_weight"] == 0.4 and d["conditions"]["lookback"] == 252
    p = d["placed"]["min_variance"]
    assert p["vol"] <= d["placed"]["equal_weight"]["vol"] + 1e-9, "the minimum-variance portfolio is no more volatile than equal weight on the window it was fitted to"
