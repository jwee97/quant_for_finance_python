"""Scenario generators and their yardstick, scenario backtests, and combinatorial purged cross-validation."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from src.scenarios import GENERATORS, compare, generate, quality_report
from src.scenarios import backtest as sb
from src.validation import cpcv


def garch_world(seed=0, T=1500, N=4):
    """A common factor and idiosyncratic noise whose volatility clusters: the thing the generators are asked to reproduce."""
    rng = np.random.default_rng(seed)
    h = np.full(T, 1.0)
    z = rng.standard_t(6, size=T) / np.sqrt(1.5)
    e = np.empty(T)
    for t in range(T):
        e[t] = np.sqrt(h[t]) * z[t]
        if t + 1 < T:
            h[t + 1] = 0.1 + 0.08 * e[t] ** 2 + 0.88 * h[t]
    common = 0.01 * e / np.sqrt(h.mean())
    r = common[:, None] * np.linspace(0.6, 1.4, N) + 0.005 * rng.standard_t(5, size=(T, N)) / np.sqrt(5 / 3)
    return pd.DataFrame(r, columns=list("ABCDEFGH")[:N], index=pd.bdate_range("2015-01-01", periods=T))


@pytest.fixture(scope="module")
def world():
    return garch_world()


# ------------------------------------------------------------------------------------------------------------------ generators
@pytest.mark.parametrize("name", ["historical", "bootstrap", "copula", "risk_factor", "arima_garch"])
def test_classical_generators_have_the_right_shape_are_finite_and_repeatable(world, name):
    a = generate(name, world, 50, 10, seed=3)
    b = generate(name, world, 50, 10, seed=3)
    c = generate(name, world, 50, 10, seed=4)
    assert a.shape == (50, 10, world.shape[1]) and np.isfinite(a).all()
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_historical_scenarios_are_windows_of_the_history(world):
    S = generate("historical", world, 20, 15, seed=1)
    R = world.to_numpy()
    for path in S:
        starts = [s for s in range(len(R) - 14) if np.array_equal(R[s], path[0])]
        assert starts and np.array_equal(R[starts[0]:starts[0] + 15], path)


def test_iid_bootstrap_draws_whole_rows_and_block_bootstrap_keeps_runs(world):
    R = world.to_numpy()
    rows = {tuple(r) for r in R}
    iid = generate("bootstrap", world, 10, 30, seed=0, kind="iid")
    assert all(tuple(r) in rows for path in iid for r in path)                                        # rows are resampled whole: cross-sectional dependence survives
    blk = generate("bootstrap", world, 10, 60, seed=0, kind="stationary", block=20.0)
    lookup = {tuple(r): i for i, r in enumerate(R)}
    runs = np.mean([lookup[tuple(path[t + 1])] == lookup[tuple(path[t])] + 1 for path in blk for t in range(59)])
    assert runs > 0.8                                                                                  # mean block 20: about 95% of steps continue a run


def test_generators_preserve_what_they_claim_to(world):
    for name, kw in (("copula", {}), ("risk_factor", {}), ("arima_garch", {}), ("bootstrap", {"kind": "iid"})):
        q = quality_report(world, generate(name, world, 400, 21, seed=0, **kw))
        assert q["corr_error"] < 0.1 and q["vol_error"] < 0.25, name


def test_garch_scenarios_cluster_volatility_and_iid_ones_do_not(world):
    iid = quality_report(world, generate("bootstrap", world, 300, 40, seed=0, kind="iid"))
    garch = quality_report(world, generate("arima_garch", world, 300, 40, seed=0))
    blocks = quality_report(world, generate("bootstrap", world, 300, 40, seed=0, kind="stationary", block=20.0))
    assert iid["clustering_error"] > 0.05                                                              # the history clusters; independent draws cannot
    assert garch["clustering_error"] < iid["clustering_error"] and blocks["clustering_error"] < iid["clustering_error"]


def test_generator_validation(world):
    with pytest.raises(ValueError):
        generate("nope", world, 10, 5)
    with pytest.raises(ValueError):
        generate("historical", world, 0, 5)
    with pytest.raises(ValueError):
        generate("historical", world, 10, 5000)
    with pytest.raises(ValueError):
        generate("historical", world.iloc[:20], 10, 5)
    assert set(GENERATORS) == {"historical", "bootstrap", "copula", "risk_factor", "arima_garch", "wgan_gp", "factor_vae", "diffusion"}


# ------------------------------------------------------------------------------------------------------------------ yardstick
def test_quality_report_flags_a_generator_that_breaks_the_dependence_and_the_tails(world):
    good = quality_report(world, generate("bootstrap", world, 400, 21, seed=0, kind="iid"))
    rng = np.random.default_rng(0)
    sd = world.to_numpy().std(axis=0)
    gauss_indep = rng.normal(size=(400, 21, world.shape[1])) * sd                                       # right volatilities, no dependence, no fat tails
    bad = quality_report(world, gauss_indep)
    assert bad["corr_error"] > 5 * good["corr_error"] and bad["corr_error"] > 0.3
    assert bad["kurtosis_ratio"] < 0.8 < good["kurtosis_ratio"]
    assert bad["tail_dep_error"] > good["tail_dep_error"]
    assert good["marginal_ks"] < 0.03 and 0.8 < good["es_ratio"] < 1.25


def test_compare_returns_one_row_per_generator(world):
    sets = {"a": generate("historical", world, 100, 10), "b": generate("copula", world, 100, 10)}
    table = compare(world, sets)
    assert list(table.index) == ["a", "b"] and "marginal_ks" in table.columns and np.isfinite(table.to_numpy(dtype=float)).all()


def test_quality_report_single_asset_and_single_day_paths():
    r = pd.DataFrame({"A": np.random.default_rng(0).normal(size=500)})
    q = quality_report(r, np.random.default_rng(1).normal(size=(200, 1, 1)))
    assert "corr_error" not in q and "clustering_error" not in q and q["marginal_ks"] < 0.15


# ------------------------------------------------------------------------------------------------------------------ scenario backtests
def test_scenario_backtest_distribution_and_costs(world):
    equal = lambda r: pd.DataFrame(1.0 / r.shape[1], index=r.index, columns=r.columns)                    # noqa: E731
    flat = lambda r: pd.DataFrame(0.0, index=r.index, columns=r.columns)                                  # noqa: E731
    a = sb.scenario_backtest(equal, world, "bootstrap", n_paths=30, horizon=250, seed=1, kind="stationary")
    assert len(a) == 30 and {"return", "volatility", "sharpe", "max_drawdown", "terminal_wealth"} <= set(a.columns)
    assert a["volatility"].between(0.02, 0.5).all() and (a["max_drawdown"] <= 0).all() and a["terminal_wealth"].std() > 0
    assert a.equals(sb.scenario_backtest(equal, world, "bootstrap", n_paths=30, horizon=250, seed=1, kind="stationary"))
    z = sb.scenario_backtest(flat, world, "historical", n_paths=5, horizon=100)
    assert np.allclose(z["terminal_wealth"], 1.0) and np.allclose(z["max_drawdown"], 0.0)

    def flip(r):                                                                                        # trades every day: pays the cost every day
        w = pd.DataFrame(0.0, index=r.index, columns=r.columns)
        w.iloc[::2, 0], w.iloc[1::2, 1] = 1.0, 1.0
        return w

    free = sb.scenario_backtest(flip, world, "historical", n_paths=10, horizon=100, seed=2)
    dear = sb.scenario_backtest(flip, world, "historical", n_paths=10, horizon=100, seed=2, cost_bps=50)
    assert (dear["terminal_wealth"] < free["terminal_wealth"]).all()


def test_evaluate_path_uses_yesterdays_weights_for_todays_return():
    idx = pd.bdate_range("2020-01-01", periods=3)
    r = pd.DataFrame({"A": [0.0, 0.10, -0.05]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    net = sb.evaluate_path(r, w)
    assert net.tolist() == pytest.approx([0.0, 0.10, -0.05])
    w0 = pd.DataFrame({"A": [0.0, 1.0, 0.0]}, index=idx)                                               # decided at day 1's close: earns day 2's return only
    assert sb.evaluate_path(r, w0).tolist() == pytest.approx([0.0, 0.0, -0.05])


# ------------------------------------------------------------------------------------------------------------------ CPCV
def test_cpcv_splits_are_disjoint_purged_and_embargoed():
    n, G, k, h, emb = 600, 6, 2, 10, 5
    splits = cpcv.cpcv_splits(n, G, k, horizon=h, embargo=emb)
    assert len(splits) == comb(G, k)
    bounds = cpcv.group_bounds(n, G)
    for sp in splits:
        assert not set(sp.train) & set(sp.test) and len(set(sp.test_groups)) == k
        assert len(sp.test) == sum(bounds[g + 1] - bounds[g] for g in sp.test_groups)
        for g in sp.test_groups:
            lo, hi = bounds[g], bounds[g + 1]
            banned = set(range(max(lo - (h - 1), 0), min(hi + emb, n)))
            assert not banned & set(sp.train)                                                          # purge before, embargo after
        assert len(sp.train) > 0


def test_cpcv_paths_cover_the_history_once_each_and_use_predictions_from_the_right_split():
    n, G, k = 300, 6, 2
    splits = cpcv.cpcv_splits(n, G, k, horizon=3)
    preds = [sp.test.astype(float) for sp in splits]                                                    # predict each observation's own index
    paths = cpcv.assemble_paths(n, splits, preds, G, k)
    assert paths.shape == (cpcv.n_paths(G, k), n) and cpcv.n_paths(G, k) == comb(G - 1, k - 1) == 5
    assert not np.isnan(paths).any() and all(np.array_equal(p, np.arange(n)) for p in paths)


def test_cpcv_each_group_is_tested_in_the_same_number_of_splits():
    splits = cpcv.cpcv_splits(120, 5, 2)
    counts = np.zeros(5, dtype=int)
    for sp in splits:
        for g in sp.test_groups:
            counts[g] += 1
    assert (counts == comb(4, 1)).all()


def test_cpcv_validation():
    for bad in ((100, 6, 0), (100, 6, 6), (5, 6, 2)):
        with pytest.raises(ValueError):
            cpcv.cpcv_splits(*bad)
    with pytest.raises(ValueError):
        cpcv.cpcv_splits(100, 5, 2, horizon=0)
