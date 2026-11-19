"""The Risk Engine: factor models recover the structure they were generated from, the risk decomposition adds up exactly, neutralising and hedging do what they say, the bias statistic catches a wrong
forecast, and the two new covariance estimators have the properties they are defined by."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.riskengine import covariance as rc
from src.riskengine import decompose as dc
from src.riskengine import factor_model as fm
from src.riskengine import hedge as hg
from src.riskengine import monitor as mo

T, N, K = 1200, 40, 3


@pytest.fixture(scope="module")
def world():
    """Returns ``r = B f + e`` with three known factors (a market, a spread, a style premium), static betas, and residual volatility of 1% a day."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2016-01-04", periods=T)
    cols = [f"A{i:02d}" for i in range(N)]
    B = np.column_stack([1.0 + 0.3 * rng.standard_normal(N), rng.standard_normal(N), rng.standard_normal(N)])
    f = rng.standard_normal((T, K)) * np.array([0.010, 0.006, 0.004])
    e = rng.standard_normal((T, N)) * 0.010
    returns = pd.DataFrame(f @ B.T + e, index=dates, columns=cols)
    factors = pd.DataFrame(f, index=dates, columns=["mkt", "spread", "style"])
    true_cov = B @ np.diag(np.array([0.010, 0.006, 0.004]) ** 2) @ B.T + np.eye(N) * 0.010 ** 2
    return returns, factors, pd.DataFrame(B, index=cols, columns=["mkt", "spread", "style"]), pd.DataFrame(true_cov, index=cols, columns=cols)


def test_time_series_model_recovers_the_exposures_and_the_covariance(world):
    returns, factors, B, true_cov = world
    m = fm.fit_time_series(returns, factors)
    assert np.abs(m.exposures.to_numpy() - B.to_numpy()).max() < 0.15 and m.kind == "time_series" and m.n_obs == T
    rel = np.linalg.norm(m.covariance().to_numpy() - true_cov.to_numpy()) / np.linalg.norm(true_cov.to_numpy())
    sample = returns.cov().to_numpy()
    assert rel < 0.07 and rel < np.linalg.norm(sample - true_cov.to_numpy()) / np.linalg.norm(true_cov.to_numpy()), "the factor structure beats the sample covariance (narrowly with 30 observations per asset)"
    assert m.specific_var.mean() == pytest.approx(0.010 ** 2, rel=0.08) and 0.4 < m.r2.median() < 0.95
    assert np.allclose(m.covariance(252).to_numpy(), m.covariance().to_numpy() * 252) and m.volatility(252).min() > 0
    s = m.summary()
    assert 0 < s["systematic_share"] < 1 and s["factors"] == 3


def test_time_series_model_refuses_too_little_data(world):
    returns, factors, *_ = world
    with pytest.raises(fm.FactorModelError, match="at least 60"):
        fm.fit_time_series(returns.iloc[:30], factors.iloc[:30])


def test_cross_sectional_model_recovers_sector_and_style_factor_returns():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2017-01-02", periods=500)
    n = 48
    cols = [f"S{i:02d}" for i in range(n)]
    sector = pd.Series(np.repeat(["tech", "bank", "energy", "health"], n // 4), index=cols)
    style = pd.DataFrame(np.tile(rng.standard_normal(n), (len(dates), 1)), index=dates, columns=cols)
    true_mkt = rng.standard_normal(len(dates)) * 0.008
    true_sec = rng.standard_normal((len(dates), 4)) * 0.004
    true_sec -= true_sec.mean(axis=1, keepdims=True)                              # equal-sized sectors: the constraint is a zero mean
    true_style = rng.standard_normal(len(dates)) * 0.003
    dummies = pd.get_dummies(sector).astype(float)[["bank", "energy", "health", "tech"]]
    ret = true_mkt[:, None] + true_sec @ dummies.to_numpy().T + true_style[:, None] * style.shift(0).to_numpy() + rng.standard_normal((len(dates), n)) * 0.006
    returns = pd.DataFrame(ret, index=dates, columns=cols)
    m = fm.fit_cross_sectional(returns, {"value": style}, sector)
    assert m.kind == "cross_sectional" and set(m.groups.values()) == {"market", "sector", "style"} and m.exposures.shape == (n, 6)
    fr = m.factor_returns
    # the characteristic is static, so lagging it changes nothing here; the style premium and the market are recovered
    truth = pd.DataFrame({"market": true_mkt, "value": true_style, "tech": true_sec[:, 3]}, index=dates).reindex(fr.index)       # the first date has no lagged exposure and is skipped
    assert all(np.corrcoef(fr[c], truth[c])[0, 1] > t for c, t in (("market", 0.95), ("value", 0.9), ("tech", 0.9)))
    sizes = dummies.sum()
    assert np.allclose((fr[["bank", "energy", "health", "tech"]] * sizes[["bank", "energy", "health", "tech"]]).sum(axis=1), 0, atol=1e-6), "sector returns are constrained to net to zero"
    assert m.covariance().shape == (n, n) and np.linalg.eigvalsh(m.covariance().to_numpy()).min() > 0


def test_cross_sectional_model_uses_the_exposure_of_the_day_before():
    """A characteristic that is the NEXT period's return would explain it perfectly if it were not lagged; the model must not see it."""
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2018-01-01", periods=200)
    cols = [f"S{i}" for i in range(30)]
    noise = pd.DataFrame(rng.standard_normal((len(dates), 30)) * 0.01, index=dates, columns=cols)
    leak = noise.sub(noise.mean(axis=1), axis=0).div(noise.std(axis=1), axis=0)          # leak[t] is the standardised return OF t: unlagged, its slope would be 0.01 on every date
    m = fm.fit_cross_sectional(noise, {"leak": leak}, None)
    slope = m.factor_returns["leak"]
    assert abs(slope.mean()) < 0.0008 and slope.abs().max() < 0.008, "lagged by one period, yesterday's characteristic says nothing about today's return"


def test_statistical_model_finds_the_number_of_factors(world):
    returns, *_ = world
    m = fm.fit_statistical(returns)
    assert 3 <= len(m.factors) <= 5 and m.exposures.shape[0] == N and np.allclose(m.exposures.T.to_numpy() @ m.exposures.to_numpy(), np.eye(len(m.factors)), atol=1e-8)
    fixed = fm.fit_statistical(returns, n_factors=3)
    assert len(fixed.factors) == 3 and fixed.r2.median() > 0.3
    full = returns.cov().to_numpy()
    assert np.linalg.norm(fixed.covariance().to_numpy() - full) / np.linalg.norm(full) < 0.2


def test_risk_decomposition_adds_up_exactly(world):
    returns, factors, *_ = world
    m = fm.fit_time_series(returns, factors, groups={"mkt": "market", "spread": "style", "style": "style"})
    w = pd.Series(np.random.default_rng(1).standard_normal(N), index=returns.columns)
    w /= w.abs().sum()
    d = dc.risk_decomposition(w, m)
    vol = float(np.sqrt(w @ m.covariance() @ w))
    assert d["volatility"] == pytest.approx(vol)
    assert d["factors"].sum() + d["specific"] == pytest.approx(vol) and d["groups"].sum() == pytest.approx(vol) and d["assets"].sum() == pytest.approx(vol)
    assert d["factor_share"] + d["specific_share"] == pytest.approx(1.0) and set(d["groups"].index) == {"market", "style", "specific"}
    eps = 1e-6                                                      # the marginal contribution is the derivative of volatility
    w2 = w.copy()
    w2.iloc[3] += eps
    assert (np.sqrt(w2 @ m.covariance() @ w2) - vol) / eps == pytest.approx(d["marginal"].iloc[3], rel=1e-3)
    assert dc.risk_decomposition(w * 0, m)["volatility"] == 0
    assert dc.risk_decomposition(w, m, annualise=252)["volatility"] == pytest.approx(vol * np.sqrt(252))


def test_active_risk_of_the_benchmark_is_zero_and_of_a_tilt_is_not(world):
    returns, factors, *_ = world
    m = fm.fit_time_series(returns, factors)
    bench = pd.Series(1.0 / N, index=returns.columns)
    assert dc.active_risk(bench, bench, m)["volatility"] == 0
    tilt = bench.copy()
    tilt.iloc[:5] += 0.02
    tilt.iloc[5:10] -= 0.02
    a = dc.active_risk(tilt, bench, m)
    assert a["volatility"] > 0 and a["active_weights"].sum() == pytest.approx(0, abs=1e-12)


def test_factor_risk_budget_moves_the_portfolio_toward_the_budget():
    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2017-01-02", periods=400)
    n = 24
    cols = [f"S{i:02d}" for i in range(n)]
    sector = pd.Series(np.repeat(["a", "b", "c"], n // 3), index=cols)
    style = pd.DataFrame(np.tile(rng.standard_normal(n), (len(dates), 1)), index=dates, columns=cols)
    returns = pd.DataFrame(rng.standard_normal((len(dates), n)) * 0.01 + rng.standard_normal((len(dates), 1)) * 0.008, index=dates, columns=cols)
    m = fm.fit_cross_sectional(returns, {"value": style}, sector)
    budget = {"market": 0.4, "sector": 0.2, "style": 0.1, "specific": 0.3}
    r = dc.factor_risk_budget_weights(m, budget)
    equal = dc.risk_decomposition(pd.Series(1.0 / n, index=cols), m)
    eq_shares = equal["groups"] / equal["volatility"]
    eq_loss = float(((eq_shares.reindex(r["target"].index).fillna(0) - r["target"]) ** 2).sum())
    assert r["weights"].sum() == pytest.approx(1.0) and (r["weights"] >= -1e-9).all() and r["loss"] <= eq_loss + 1e-12
    with pytest.raises(ValueError, match="no factors in group"):
        dc.factor_risk_budget_weights(fm.fit_statistical(returns), {"market": 1.0})


def test_neutralising_zeroes_the_chosen_exposures_with_the_smallest_change_in_risk(world):
    returns, factors, *_ = world
    m = fm.fit_time_series(returns, factors)
    w = pd.Series(1.0 / N, index=returns.columns)
    out = hg.neutralise(w, m, ["mkt"])
    assert abs(out["exposures_after"]["mkt"]) < 1e-9 and abs(out["exposures_before"]["mkt"]) > 0.5
    assert out["volatility_after"] < out["volatility_before"]
    both = hg.neutralise(w, m, ["mkt", "spread"])
    assert abs(both["exposures_after"][["mkt", "spread"]]).max() < 1e-9
    norm = hg.neutralise(w, m, ["mkt"], metric="norm")
    S = m.covariance().to_numpy()
    risk_change = float(out["trade"] @ S @ out["trade"])
    assert risk_change <= float(norm["trade"] @ S @ norm["trade"]) + 1e-12 and norm["trade"].pow(2).sum() <= out["trade"].pow(2).sum() + 1e-12
    with pytest.raises(ValueError, match="unknown factor"):
        hg.neutralise(w, m, ["momentum"])
    with pytest.raises(ValueError, match="metric"):
        hg.neutralise(w, m, ["mkt"], metric="x")


def test_a_minimum_variance_hedge_removes_the_hedgeable_variance(world):
    returns, _, _, true_cov = world
    idx = returns.columns
    index_fund = idx[0]
    w = pd.Series(0.0, index=idx)
    w.iloc[10:20] = 0.1
    h = hg.min_variance_hedge(w, true_cov, [index_fund])
    assert h["variance_after"] < h["variance_before"] and 0 < h["effectiveness"] < 1
    h2 = hg.min_variance_hedge(w, true_cov, list(idx[:5]))
    assert h2["effectiveness"] > h["effectiveness"], "more instruments hedge more"
    # the closed form: the hedge position is minus the regression slope of the portfolio on the instrument
    S = true_cov.to_numpy()
    assert h["hedge"].iloc[0] == pytest.approx(-(S[0] @ w.to_numpy()) / S[0, 0], rel=1e-6)
    with pytest.raises(ValueError, match="hedge assets"):
        hg.min_variance_hedge(w, true_cov, ["nope"])


def test_bias_statistic_is_one_for_the_true_volatility_and_catches_a_wrong_one():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2019-01-01", periods=1500)
    sigma = pd.Series(0.01 * (1 + 0.5 * np.sin(np.arange(1500) / 80)), index=idx)
    r = pd.Series(rng.standard_normal(1500) * sigma.to_numpy(), index=idx)
    ok = mo.bias_statistic(r, sigma)
    assert ok["verdict"] == "ok" and abs(ok["latest"] - 1) < 0.15 and abs(ok["full"] - 1) < 0.05
    low = mo.bias_statistic(r, 0.6 * sigma)
    assert low["verdict"].startswith("risk understated") and low["latest"] == pytest.approx(1 / 0.6, rel=0.1)
    assert mo.bias_statistic(r, 1.7 * sigma)["verdict"].startswith("risk overstated")
    assert mo.bias_statistic(r.iloc[:10], sigma.iloc[:10])["verdict"] == "not enough data"


def test_forecast_check_compares_forecast_with_what_followed(world):
    returns, *_ = world
    w = pd.Series(1.0 / N, index=returns.columns)
    out = mo.portfolio_forecast_check(w, returns, lambda window: window.cov().to_numpy(), lookback=250, step=21)
    assert len(out) > 30 and out["ratio"].median() == pytest.approx(1.0, abs=0.2) and (out["forecast"] > 0).all()


def test_exposure_and_factor_vol_drift_flags_a_changed_world(world):
    returns, factors, *_ = world
    a = fm.fit_time_series(returns.iloc[:600], factors.iloc[:600])
    b = fm.fit_time_series(returns.iloc[:600], factors.iloc[:600] * np.array([1.0, 2.0, 1.0]))
    d = mo.exposure_drift(a, b)
    assert d.loc["spread", "factor_vol_change"] == pytest.approx(1.0, abs=0.01) and abs(d.loc["mkt", "factor_vol_change"]) < 1e-9
    assert d.loc["spread", "mean_abs_exposure_change"] > 0.2 and d.loc["mkt", "mean_abs_exposure_change"] < 1e-9


def test_detoning_removes_the_market_component(world):
    returns, *_ = world
    corr = returns.corr()
    top = np.linalg.eigvalsh(corr.to_numpy()).max()
    flat = rc.detone_correlation(corr, 1)
    assert np.allclose(np.diag(flat), 1.0) and np.linalg.eigvalsh(flat.to_numpy()).max() < top and np.linalg.eigvalsh(flat.to_numpy()).min() < 1e-6, "rank-deficient by construction"
    assert flat.to_numpy()[np.triu_indices(N, 1)].mean() < corr.to_numpy()[np.triu_indices(N, 1)].mean()
    assert rc.detone_correlation(corr, 0).equals(corr) or np.allclose(rc.detone_correlation(corr, 0), corr)
    assert rc.detoned_covariance(returns, 1).shape == (N, N)


def test_graphical_lasso_finds_the_conditional_independence_structure():
    rng = np.random.default_rng(4)
    n, t = 8, 1500
    prec = np.eye(n) * 2.0
    for i in range(n - 1):
        prec[i, i + 1] = prec[i + 1, i] = -0.9                                 # a chain: only neighbours are conditionally dependent
    x = rng.multivariate_normal(np.zeros(n), np.linalg.inv(prec), size=t) * 0.01
    r = pd.DataFrame(x, columns=list("ABCDEFGH"))
    g = rc.graphical_lasso_covariance(r, alpha=0.08, annualise=False)
    P = g["precision"].to_numpy()
    scaled = P * np.outer(r.std().to_numpy(), r.std().to_numpy())
    assert abs(scaled[0, 4]) < 1e-6 and abs(scaled[0, 7]) < 1e-6 and abs(scaled[0, 1]) > 0.05 and g["sparsity"] > 0.4
    assert np.linalg.eigvalsh(g["covariance"].to_numpy()).min() > 0
    cv = rc.graphical_lasso_covariance(r, annualise=False)
    assert cv["alpha"] > 0 and cv["covariance"].shape == (n, n)
    with pytest.raises(ValueError, match="at least 30"):
        rc.graphical_lasso_covariance(r.iloc[:10])
