"""Nested clustered optimisation, drawdown and entropic risk, and near-optimal centring: each against a closed form, a brute-force search, or the property that defines it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.portfolio import drawdown as dd
from src.portfolio import near_optimal as no
from src.portfolio import nco


def block_cov(sizes=(5, 5, 5), within=0.6, vols=(0.2, 0.2, 0.2), between=0.05):
    n = sum(sizes)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    corr = np.where(labels[:, None] == labels[None, :], within, between)
    np.fill_diagonal(corr, 1.0)
    v = np.repeat(vols, sizes)
    cols = [f"X{i:02d}" for i in range(n)]
    return pd.DataFrame(corr * np.outer(v, v), index=cols, columns=cols), labels


def test_drawdown_measures_on_a_path_worked_by_hand():
    r = np.array([0.10, -0.05, -0.05, 0.02])                          # cumulative .10 .05 .00 .02 against a peak of .10: drawdowns 0 .05 .10 .08
    assert dd.drawdown_path(r) == pytest.approx([0, 0.05, 0.10, 0.08])
    assert dd.max_drawdown(r) == pytest.approx(0.10)
    assert dd.cdar(r, 0.5) == pytest.approx((0.10 + 0.08) / 2), "the average of the worst half"
    assert dd.cdar(r, 0.0) == pytest.approx(np.mean([0, 0.05, 0.10, 0.08])) and dd.cdar(r, 0.75) == pytest.approx(0.10)
    assert dd.drawdown_path(np.array([-0.02, -0.01, 0.0]))[2] == pytest.approx(0.03), "the starting equity is the first peak"
    s = pd.Series(r, index=pd.RangeIndex(4))
    assert isinstance(dd.drawdown_path(s), pd.Series)


def test_cdar_weights_are_the_best_on_a_grid_of_two_assets():
    rng = np.random.default_rng(1)
    T = 400
    a = rng.standard_normal(T) * 0.01 + 0.0004
    b = 0.6 * a + 0.8 * rng.standard_normal(T) * 0.01 + 0.0003
    R = pd.DataFrame({"a": a, "b": b})
    res = dd.cdar_weights(R, alpha=0.9)
    assert res["ok"] and res["weights"].sum() == pytest.approx(1.0) and (res["weights"] >= -1e-9).all()
    grid = np.linspace(0, 1, 401)
    values = [dd.cdar(R.to_numpy() @ np.array([w, 1 - w]), 0.9) for w in grid]
    assert res["cdar"] == pytest.approx(min(values), rel=2e-3) and res["cdar"] <= min(values) + 1e-9, "the LP finds the grid's minimum"
    assert res["max_drawdown"] >= res["cdar"] - 1e-12


def test_max_drawdown_weights_and_the_return_floor():
    rng = np.random.default_rng(2)
    T = 400
    R = pd.DataFrame(rng.standard_normal((T, 3)) * [0.005, 0.01, 0.02] + [0.0002, 0.0004, 0.0009])
    m = dd.max_drawdown_weights(R)
    grid = [(w1, w2, 1 - w1 - w2) for w1 in np.linspace(0, 1, 51) for w2 in np.linspace(0, 1 - w1, max(int(50 * (1 - w1)), 1) + 1)]
    best = min(dd.max_drawdown(R.to_numpy() @ np.array(w)) for w in grid)
    assert m["ok"] and m["max_drawdown"] <= best + 1e-9
    target = 0.8 * float(R.mean().max())                                           # reachable, but only by leaning on the best asset
    floor = dd.max_drawdown_weights(R, target_return=target)
    assert floor["ok"] and float(R.mean().to_numpy() @ floor["weights"].to_numpy()) >= target - 1e-9 and floor["max_drawdown"] >= m["max_drawdown"] - 1e-9, "a return floor can only cost drawdown"
    assert not dd.max_drawdown_weights(R, target_return=0.5)["ok"], "an unreachable return is reported, not silently equal-weighted"
    with pytest.raises(ValueError, match="at least 60"):
        dd.cdar_weights(R.iloc[:20])
    with pytest.raises(ValueError, match="alpha"):
        dd.cdar_weights(R, alpha=1.0)


def test_evar_matches_the_gaussian_closed_form_and_dominates_cvar():
    rng = np.random.default_rng(3)
    mu, s, alpha = 0.001, 0.02, 0.95
    r = rng.normal(mu, s, 400000)
    expected = -mu + s * np.sqrt(-2 * np.log(1 - alpha))              # EVaR of a normal LOSS with mean -mu and std s
    assert dd.evar(r, alpha) == pytest.approx(expected, rel=0.03)
    loss = -r
    var = np.quantile(loss, alpha)
    cvar = loss[loss >= var].mean()
    assert dd.evar(r, alpha) >= cvar >= var and cvar == pytest.approx(-mu + s * norm.pdf(norm.ppf(alpha)) / (1 - alpha), rel=0.02)


def test_evar_weights_beat_equal_weight_and_match_a_grid():
    rng = np.random.default_rng(4)
    T = 600
    a = rng.standard_normal(T) * 0.01
    b = 0.3 * a + rng.standard_t(4, T) * 0.012
    R = pd.DataFrame({"a": a, "b": b})
    res = dd.evar_weights(R, alpha=0.9)
    grid = np.linspace(0, 1, 201)
    vals = [dd.evar(R.to_numpy() @ np.array([w, 1 - w]), 0.9) for w in grid]
    assert res["ok"] and res["evar"] <= dd.evar(R.to_numpy() @ np.array([0.5, 0.5]), 0.9) + 1e-12 and res["evar"] == pytest.approx(min(vals), rel=0.01)
    floor = dd.evar_weights(R, alpha=0.9, target_return=float(R.mean().max()))
    assert floor["ok"] and floor["weights"].sum() == pytest.approx(1.0)


def test_nco_recovers_the_clusters_and_allocates_between_them():
    cov, labels = block_cov((5, 5, 5), vols=(0.15, 0.20, 0.40))
    w, info = nco.nco_weights(cov, details=True, k=3)
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-12).all()
    assert len(set(zip(info["labels"], labels))) == 3, "the three blocks are the three clusters"
    by_block = w.groupby(labels).sum()
    assert by_block.iloc[0] > by_block.iloc[1] > by_block.iloc[2], "the volatile block gets the least"
    inside = w[labels == 1] / w[labels == 1].sum()
    assert inside.max() - inside.min() < 1e-6, "identical assets share a cluster equally"
    auto = nco.nco_weights(cov)
    assert auto.sum() == pytest.approx(1.0)
    rp = nco.nco_weights(cov, objective="risk_parity", k=3)
    ms = nco.nco_weights(cov, mu=pd.Series(np.where(labels == 0, 0.10, 0.02), index=cov.index), objective="max_sharpe", k=3)
    assert rp.sum() == pytest.approx(1.0) and ms.groupby(labels).sum().iloc[0] > by_block.iloc[0]
    with pytest.raises(ValueError, match="objective"):
        nco.nco_weights(cov, objective="oracle")
    tiny = nco.nco_weights(cov.iloc[:3, :3])
    assert tiny.sum() == pytest.approx(1.0)


def test_nco_is_more_stable_than_the_plain_minimum_variance_optimum_when_the_covariance_is_noisy():
    cov, _ = block_cov((8, 8, 8, 8), within=0.7, vols=(0.2, 0.25, 0.3, 0.35), between=0.1)
    rng = np.random.default_rng(6)
    R = pd.DataFrame(rng.multivariate_normal(np.zeros(len(cov)), cov.to_numpy() / 252, size=70), columns=cov.columns)       # 70 observations of 32 assets
    plain = no.weight_dispersion(R, lambda r: nco.long_only(r.cov().to_numpy(), None, "min_variance"), n_boot=25, seed=1)
    nested = no.weight_dispersion(R, lambda r: nco.nco_weights(r.cov()), n_boot=25, seed=1)
    assert nested["turnover"] < plain["turnover"], (nested["turnover"], plain["turnover"])


def test_near_optimal_centre_spans_optimum_to_equal_weight():
    cov, _ = block_cov((4, 4), within=0.8, vols=(0.15, 0.30))
    mu = pd.Series(np.linspace(0.02, 0.10, 8), index=cov.index)
    zero = no.near_optimal_center(mu, cov, 3.0, tolerance=0.0)
    assert np.allclose(zero["weights"], zero["optimum"]) and zero["loss_share"] == 0
    mid = no.near_optimal_center(mu, cov, 3.0, tolerance=0.2)
    assert mid["converged"] and mid["weights"].sum() == pytest.approx(1.0) and (mid["weights"] > 0).all(), "the centre holds every asset"
    assert 0 <= mid["loss_share"] <= 0.2 + 1e-6, "it gives up no more than the tolerance"
    assert mid["n_eff_center"] > mid["n_eff_optimum"]
    wide = no.near_optimal_center(mu, cov, 3.0, tolerance=0.8)
    assert wide["n_eff_center"] >= mid["n_eff_center"] - 1e-6 and wide["loss_share"] >= mid["loss_share"] - 1e-9
    assert wide["u_center"] >= wide["u_equal"] - 1e-9, "never worse than equal weight at the widest tolerance"
    capped = no.near_optimal_center(mu, cov, 3.0, tolerance=0.3, ub=0.25)
    assert capped["weights"].max() <= 0.25 + 1e-6
    with pytest.raises(ValueError, match="tolerance"):
        no.near_optimal_center(mu, cov, tolerance=1.5)
    with pytest.raises(ValueError, match="no fully invested"):
        no.near_optimal_center(mu, cov, ub=0.05)
    assert no.near_optimal_center(None, cov, tolerance=0.1)["weights"].sum() == pytest.approx(1.0), "no expected returns: the minimum-variance centre"


def test_the_centre_is_more_stable_than_the_optimum_under_resampling():
    cov, _ = block_cov((6, 6, 6), within=0.75, vols=(0.18, 0.22, 0.28), between=0.2)
    rng = np.random.default_rng(9)
    R = pd.DataFrame(rng.multivariate_normal(np.full(len(cov), 0.0003), cov.to_numpy() / 252, size=120), columns=cov.columns)
    def best(r):
        return no.optimum(r.mean().to_numpy() * 252, r.cov().to_numpy() * 252, 3.0)

    def centred(r):
        return no.near_optimal_center(r.mean() * 252, r.cov() * 252, 3.0, tolerance=0.3)["weights"]
    a, b = no.weight_dispersion(R, best, n_boot=15, seed=2), no.weight_dispersion(R, centred, n_boot=15, seed=2)
    assert b["turnover"] < a["turnover"], (b["turnover"], a["turnover"])
