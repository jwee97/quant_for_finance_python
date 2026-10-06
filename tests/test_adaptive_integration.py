"""Regime -> forecast confidence -> risk limits, calibrated confidence and decay-aware combination: planted truths, edge cases, invalid input, no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import DETECTORS, Pipeline, RegimeSeries, bundle_from_prices, register_model
from src.framework.adaptive import (RegimeRiskLimits, calibrate_confidence, decay_trust_weights, fit_decay, holding_period_ic, regime_spread_scale)
from src.framework.forecasting import ForecastModel
from src.framework.types import ForecastPanel
from src.framework.validate import check_causality
from src.utils.config import load_config

H = 21


def _panel(mean: pd.DataFrame, std: float | pd.DataFrame = 0.04) -> ForecastPanel:
    s = std if isinstance(std, pd.DataFrame) else pd.DataFrame(std, index=mean.index, columns=mean.columns)
    return ForecastPanel.from_mean_std(mean, s, H, "t")


@pytest.fixture(scope="module")
def world():
    """Two regimes (A calm, B stressed); forecasts are right about the spread in A and 2x over-confident in B."""
    rng = np.random.default_rng(5)
    n, k = 3200, 4
    idx = pd.bdate_range("2005-01-03", periods=n)
    state = (np.arange(n) // 160) % 2                                  # alternate every 160 days
    daily_sd = np.where(state == 1, 0.02, 0.01)
    r = pd.DataFrame(rng.normal(0, 1, (n, k)) * daily_sd[:, None], index=idx, columns=list("WXYZ"))
    probs = pd.DataFrame({"A": (state == 0).astype(float), "B": (state == 1).astype(float)}, index=idx)
    std = pd.DataFrame(0.01 * np.sqrt(H), index=idx, columns=r.columns)          # a constant forecast spread: right in A, half the truth in B
    mean = pd.DataFrame(0.0, index=idx, columns=r.columns)
    return r, probs, std, mean


# ------------------------------------------------------------------------------------------ regime -> forecast spread
def test_regime_spread_scale_learns_that_the_forecast_is_overconfident_in_the_stressed_regime(world):
    r, probs, std, mean = world
    panel = ForecastPanel.from_mean_std(mean, std, H, "t")
    adjusted, scale = regime_spread_scale(panel, r, RegimeSeries(probs), min_obs=2000, shrink=0.0)
    late = scale.iloc[2500:]
    in_b, in_a = late[probs["B"].iloc[2500:] == 1.0], late[probs["A"].iloc[2500:] == 1.0]
    assert in_b.mean() > 1.5 and in_b.mean() > in_a.mean() + 0.5              # stressed spread widened toward the true 2x
    assert 0.8 < in_a.mean() < 1.3                                           # calm spread left alone
    assert (adjusted.std.iloc[2500:] > 0).all().all() and adjusted.mean.equals(panel.mean)
    assert scale.iloc[:500].eq(1.0).all()                                     # until a regime has enough matured evidence the spread is untouched


def test_regime_spread_scale_shrink_one_is_the_identity_and_bad_input_is_rejected(world):
    r, probs, std, mean = world
    panel = ForecastPanel.from_mean_std(mean, std, H, "t")
    same, scale = regime_spread_scale(panel, r, RegimeSeries(probs), min_obs=500, shrink=1.0)
    assert np.allclose(scale.to_numpy(), 1.0)
    with pytest.raises(ValueError, match="regime detector"):
        regime_spread_scale(panel, r, None)


def test_regime_spread_scale_uses_only_matured_outcomes(world):
    r, probs, std, mean = world
    panel = ForecastPanel.from_mean_std(mean, std, H, "t")
    base = regime_spread_scale(panel, r, RegimeSeries(probs), min_obs=500, shrink=0.0)[1]
    cut = 2000
    shocked = r.copy()
    shocked.iloc[cut + 1:] = shocked.iloc[cut + 1:] * 50.0                    # change every return AFTER the cutoff day (the cutoff day's own return is known at its close)
    changed = regime_spread_scale(panel, shocked, RegimeSeries(probs), min_obs=500, shrink=0.0)[1]
    assert np.allclose(base.iloc[: cut + 1].to_numpy(), changed.iloc[: cut + 1].to_numpy())      # nothing on or before the cutoff can see it


# ------------------------------------------------------------------------------------------ calibration of confidence
@pytest.fixture(scope="module")
def overconfident():
    rng = np.random.default_rng(9)
    n, k = 3400, 5
    idx = pd.bdate_range("2004-01-05", periods=n)
    r = pd.DataFrame(rng.normal(0, 0.01, (n, k)), index=idx, columns=list("ABCDE"))
    fwd_sd = 0.01 * np.sqrt(H)
    mean = pd.DataFrame(rng.normal(0, 1, (n, k)) * fwd_sd * 1.5, index=idx, columns=r.columns)       # forecasts that claim far more than they know
    return r, _panel(mean, fwd_sd)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_pulls_overconfident_probabilities_toward_the_truth(overconfident, method):
    r, panel = overconfident
    cal = calibrate_confidence(panel, r, method, min_obs=2000, refit_every=21)
    fwd = (1 + r).cumprod().shift(-H) / (1 + r).cumprod() - 1
    up = (fwd > 0).astype(float).where(fwd.notna())
    late = slice(2600, 3300)
    ok = up.iloc[late].notna() & cal.p_up.iloc[late].notna()
    raw = panel.probability_up().iloc[late].where(ok).stack()
    new = cal.p_up.iloc[late].where(ok).stack()
    y = up.iloc[late].where(ok).stack()
    brier = lambda p: float(((p - y) ** 2).mean())
    assert brier(new) < brier(raw)                                            # calibration improved the proper score
    assert abs(float((new - 0.5).abs().mean())) < abs(float((raw - 0.5).abs().mean()))   # and shrank the claims toward a coin flip
    assert cal.confidence.stack().max() <= 1.0 and cal.mean.equals(panel.mean) and cal.std.equals(panel.std)


def test_calibration_edge_cases_and_invalid_input(overconfident):
    r, panel = overconfident
    assert calibrate_confidence(panel, r, "none") is panel
    early = calibrate_confidence(panel, r, "platt", min_obs=10 ** 9)
    assert np.allclose(early.p_up.fillna(0).to_numpy(), panel.probability_up().fillna(0).to_numpy())     # not enough matured pairs: raw probability kept
    with pytest.raises(ValueError, match="calibration method"):
        calibrate_confidence(panel, r, "magic")


def test_calibration_uses_only_matured_outcomes(overconfident):
    r, panel = overconfident
    base = calibrate_confidence(panel, r, "platt", min_obs=1500, refit_every=21)
    cut = 2800
    shocked = r.copy()
    shocked.iloc[cut:] = -shocked.iloc[cut:]                                  # flip every outcome from the cutoff
    changed = calibrate_confidence(panel, shocked, "platt", min_obs=1500, refit_every=21)
    # the calibrator in force at row t was fitted on outcomes complete by t, so rows up to cut + H are unaffected only up to the last refit before cut + H
    safe = cut + 0
    assert np.allclose(base.p_up.iloc[:safe].fillna(-1).to_numpy(), changed.p_up.iloc[:safe].fillna(-1).to_numpy())


# ------------------------------------------------------------------------------------------ alpha decay
def test_fit_decay_recovers_an_exponential_and_the_holding_period_formula_matches_the_average():
    tau = 12.0
    curve = {d: 0.05 * np.exp(-d / tau) for d in (1, 5, 10, 21, 42, 63)}
    ic0, got = fit_decay(curve)
    assert ic0 == pytest.approx(0.05, rel=0.1) and got == pytest.approx(tau, rel=0.1)
    direct = np.mean([0.05 * np.exp(-d / tau) for d in np.arange(1, 22)])
    assert holding_period_ic(0.05, tau, 21) == pytest.approx(direct, rel=0.08)          # continuous formula against the discrete mean
    assert holding_period_ic(-0.01, tau, 21) == 0.0 and holding_period_ic(0.05, float("nan"), 21) == 0.0
    assert fit_decay({1: -0.02, 5: -0.01, 10: -0.02})[0] == 0.0 and fit_decay({1: 0.02, 5: 0.01})[0] == 0.0   # no positive information, or too few lags: no claim


def test_decay_weights_favour_the_alpha_whose_information_persists_and_use_only_the_past():
    rng = np.random.default_rng(3)
    n, k = 3000, 8
    idx = pd.bdate_range("2006-01-02", periods=n)
    fast, slow = rng.normal(size=(n, k)), rng.normal(size=(n, k))
    base = rng.normal(0, 0.01, (n, k))
    ret = base.copy()
    for lag in range(1, 4):                                                    # the fast signal moves returns on days 1-3 after it is seen only
        ret[lag:] += 0.004 * fast[:-lag]
    for lag in range(1, 61):                                                   # the slow one keeps moving them for 60 days, at a smaller daily size
        ret[lag:] += 0.0016 * slow[:-lag]
    returns = pd.DataFrame(ret, index=idx, columns=[f"a{i}" for i in range(k)])
    panels = {"fast": _panel(pd.DataFrame(fast, index=idx, columns=returns.columns)), "slow": _panel(pd.DataFrame(slow, index=idx, columns=returns.columns))}
    w, table = decay_trust_weights(panels, returns, lags=(1, 5, 10, 21, 42, 63), window=1500, min_obs=500, holding=21)
    late = w.iloc[2500:]
    assert np.allclose(late.sum(axis=1), 1.0) and (late >= 0).all().all()
    tau = table.xs("slow", level="model")["tau"].dropna().iloc[-1], table.xs("fast", level="model")["tau"].dropna().iloc[-1]
    assert tau[0] > tau[1]                                                      # the slow alpha is estimated to decay more slowly
    cut = 2200
    shocked = returns.copy()
    shocked.iloc[cut:] = np.random.default_rng(0).normal(0, 0.05, shocked.iloc[cut:].shape)
    w2, _ = decay_trust_weights(panels, shocked, lags=(1, 5, 10, 21, 42, 63), window=1500, min_obs=500, holding=21)
    assert np.allclose(w.iloc[: cut - 70].to_numpy(), w2.iloc[: cut - 70].to_numpy())  # weights up to the cutoff (less the longest lag) cannot see later returns


# ------------------------------------------------------------------------------------------ regime -> risk limits
def test_gross_caps_bind_in_the_capped_regime_and_blend_with_probability():
    idx = pd.bdate_range("2020-01-01", periods=60)
    w = pd.DataFrame(0.5, index=idx, columns=list("AB"))                       # gross 1.0
    r = pd.DataFrame(0.0, index=idx, columns=list("AB"))
    crisis = RegimeSeries(pd.DataFrame({"Crisis": 1.0, "Calm": 0.0}, index=idx))
    out, diag = RegimeRiskLimits(gross_caps={"Crisis": 0.4}).apply(w, r, crisis)
    assert np.allclose(out.abs().sum(axis=1), 0.4) and np.allclose(diag["cap_factor"], 0.4)
    half = RegimeSeries(pd.DataFrame({"Crisis": 0.5, "Calm": 0.5}, index=idx))
    out_half, _ = RegimeRiskLimits(gross_caps={"Crisis": 0.4}).apply(w, r, half)
    assert out_half.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)            # the cap is probability-weighted with an unconstrained calm: it does not bind at p = 0.5
    untouched, _ = RegimeRiskLimits().apply(w, r, crisis)
    assert untouched.equals(w)


def test_drawdown_limit_derisks_after_a_loss_and_releases_after_recovery():
    n = 120
    idx = pd.bdate_range("2020-01-01", periods=n)
    w = pd.DataFrame(1.0, index=idx, columns=["A"])
    ret = np.zeros(n)
    ret[10:15] = -0.03                                                           # a 14% drawdown
    ret[60:100] = 0.01                                                           # then a long recovery
    r = pd.DataFrame({"A": ret}, index=idx)
    out, diag = RegimeRiskLimits(drawdown={"trigger": 0.10, "derisk": 0.25, "recover": 0.05}).apply(w, r)
    assert diag["drawdown_limit_on"].iloc[:10].eq(0).all() and diag["drawdown_limit_on"].iloc[14:30].eq(1).all()
    assert out["A"].iloc[20] == pytest.approx(0.25) and out["A"].iloc[-1] == pytest.approx(1.0)
    # no look-ahead: returns after day t cannot change the factor on day t
    r2 = r.copy()
    r2.iloc[40:] = -0.05
    out2, _ = RegimeRiskLimits(drawdown={"trigger": 0.10, "derisk": 0.25, "recover": 0.05}).apply(w, r2)
    assert out.iloc[:40].equals(out2.iloc[:40])


@pytest.mark.parametrize("bad", [{"trigger": 0.1, "recover": 0.2}, {"trigger": 1.5}, {"trigger": 0.1, "derisk": 1.2}])
def test_invalid_drawdown_limits_are_rejected(bad):
    idx = pd.bdate_range("2020-01-01", periods=5)
    with pytest.raises(ValueError, match="drawdown limits"):
        RegimeRiskLimits(drawdown=bad).apply(pd.DataFrame(1.0, index=idx, columns=["A"]), pd.DataFrame(0.0, index=idx, columns=["A"]))


# ------------------------------------------------------------------------------------------ detectors
@pytest.mark.parametrize("name", ["gmm", "bocpd"])
def test_new_detectors_are_registered_causal_probabilities(name):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2012-01-01", periods=1800)
    vol = np.where(np.arange(1800) % 600 > 450, 0.03, 0.008)
    r = pd.DataFrame(rng.normal(0, 1, (1800, 4)) * vol[:, None], index=idx, columns=list("ABCD"))
    bundle = bundle_from_prices(100 * (1 + r).cumprod(), min_history=60)
    series = DETECTORS.create(name).detect(bundle)
    series.validate()
    assert series.probabilities.dropna().shape[0] > 500
    assert check_causality(DETECTORS.create(name), bundle)["ok"]


# ------------------------------------------------------------------------------------------ pipeline integration
@register_model("test_adaptive_fast", "test", "a short-memory trend used by the adaptive integration test")
class _Fast(ForecastModel):
    name, family = "test_adaptive_fast", "test"

    def score(self, data):
        return data.prices.pct_change(10).where(data.investable)


@register_model("test_adaptive_slow", "test", "a long-memory trend used by the adaptive integration test")
class _Slow(ForecastModel):
    name, family = "test_adaptive_slow", "test"

    def score(self, data):
        return data.prices.pct_change(120).where(data.investable)


@pytest.fixture(scope="module")
def pipeline_bundle():
    rng = np.random.default_rng(11)
    n = 3000
    idx = pd.bdate_range("2008-01-01", periods=n)
    vol = np.where((np.arange(n) // 250) % 3 == 2, 0.02, 0.008)
    drift = np.sin(np.arange(n) / 60.0) * 0.0008
    r = pd.DataFrame(rng.normal(0, 1, (n, 6)) * vol[:, None] + drift[:, None], index=idx, columns=list("ABCDEF"))
    return bundle_from_prices(100 * (1 + r).cumprod(), min_history=60, name="adaptive")


def test_pipeline_wires_regime_confidence_calibration_decay_weights_and_risk_limits(pipeline_bundle):
    config = load_config()
    spec = {"name": "adaptive", "models": [{"name": "test_adaptive_fast"}, {"name": "test_adaptive_slow"}], "regime": {"detector": "vol_state"},
            "combination": {"rule": "decay_weighted", "min_obs": 126, "window": 504},
            "forecast": {"regime_spread": {"min_obs": 500}, "confidence": {"method": "platt", "min_obs": 1500}},
            "risk": {"mode": "regime", "limits": {"gross_caps": {"Crisis": 0.5}, "drawdown": {"trigger": 0.08, "derisk": 0.5, "recover": 0.04}}},
            "evaluation": {"benchmarks": ["equal_weight"]}}
    result = Pipeline(spec, config, pipeline_bundle).run()
    assert {"alpha_decay", "regime_spread_scale", "risk_limits", "by_regime", "attribution"} <= set(result.tables)
    assert result.forecasts.p_up is not None and result.forecasts.confidence.stack().dropna().between(0, 1).all()
    assert "risk_limit_days_share" in result.metrics and 0.0 <= result.metrics["risk_limit_days_share"] <= 1.0
    assert np.isfinite(result.metrics["sharpe"]) and "reliability" in result.tables
    assert result.validation["causality"]["test_adaptive_fast"]["ok"]


def test_pipeline_rejects_unknown_forecast_and_risk_options(pipeline_bundle):
    config = load_config()
    base = {"name": "x", "models": [{"name": "test_adaptive_fast"}]}
    with pytest.raises(ValueError, match="unknown forecast options"):
        Pipeline({**base, "forecast": {"magic": 1}}, config, pipeline_bundle).run(validate=False)
    with pytest.raises(ValueError, match="unknown risk limits"):
        Pipeline({**base, "risk": {"mode": "none", "limits": {"nope": 1}}}, config, pipeline_bundle).run(validate=False)
    with pytest.raises(ValueError, match="regime detector"):
        Pipeline({**base, "forecast": {"regime_spread": {}}}, config, pipeline_bundle).run(validate=False)
