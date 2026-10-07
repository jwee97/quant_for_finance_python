"""Strategies and allocators built on the econometrics / probability layers: causal, sensible on planted structure, and wired into the pipeline."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.framework import ALLOCATORS, MODELS, Pipeline, PipelineSpec, bundle_from_prices, load_library
from src.framework.allocation import Context
from src.framework.types import ForecastPanel
from src.framework.validate import check_causality
from src.strategies.econometric import alpha_beta_slope, steady_state_gains
from src.utils.config import load_config

warnings.filterwarnings("ignore")
load_library()


def _bundle(n=2000, k=6, seed=11, drift=0.0003):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    common = rng.normal(0, 0.006, (n, 1))
    ret = common * rng.uniform(0.4, 1.2, (1, k)) + rng.normal(0, 0.007, (n, k)) + drift
    return bundle_from_prices(pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=list("ABCDEFGHIJ")[:k]), name="synthetic")


@pytest.fixture(scope="module")
def bundle():
    return _bundle()


def test_alpha_beta_filter_equals_the_steady_state_kalman_filter():
    rng = np.random.default_rng(0)
    n = 600
    y = np.cumsum(0.05 + rng.normal(0, 0.1, n)) + rng.normal(0, 1.0, n)
    q_level, q_slope = 0.02, 1e-4
    g1, g2 = steady_state_gains(q_level, q_slope)
    assert 0 < g1 < 1 and 0 < g2 < g1
    mine = alpha_beta_slope(pd.DataFrame({"x": y}), g1, g2)["x"].to_numpy()
    from src.econometrics.statespace import StateSpace

    m = StateSpace([[1.0, 1.0], [0.0, 1.0]], [[1.0, 0.0]], [[1.0]], np.diag([q_level, q_slope]), P0=np.eye(2) * 1e3)
    ref = m.filter(y).filtered_state[:, 1]
    assert mine[200:] == pytest.approx(ref[200:], abs=1e-4)


def test_alpha_beta_filter_handles_a_late_start_and_missing_prices():
    y = pd.DataFrame({"a": np.arange(100.0) * 0.1, "b": np.where(np.arange(100) < 30, np.nan, np.arange(100.0) * 0.1)})
    y.loc[60:62, "a"] = np.nan
    s = alpha_beta_slope(y, *steady_state_gains(0.02, 1e-3))
    assert s["b"].iloc[:30].isna().all() and s["b"].iloc[30:].notna().all() and s["a"].notna().all()
    assert s["a"].iloc[-1] == pytest.approx(0.1, abs=0.02)


def test_kalman_trend_follows_the_direction_of_the_trend_and_is_causal(bundle):
    rng = np.random.default_rng(3)
    n = 1500
    drift = np.where(np.arange(n) < 750, 0.002, -0.002)
    px = pd.DataFrame({"A": 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))}, index=pd.bdate_range("2012", periods=n))
    b = bundle_from_prices(px, name="regime")
    s = MODELS.create("kalman_trend").score(b)["A"]
    assert s.iloc[300:700].mean() > 0.1 and s.iloc[1000:1450].mean() < -0.1 and s.dropna().abs().max() <= 1.0
    for name in ("kalman_trend", "bvar_lead_lag"):
        assert check_causality(MODELS.create(name), bundle, cutoff=bundle.index[1500])["ok"]
    with pytest.raises(ValueError):
        MODELS.create("kalman_trend", level_noise=-1.0)


def test_garch_managed_exposure_is_inverse_to_forecast_volatility_and_is_causal():
    from src.econometrics.garch import walk_forward_volatility
    from src.probability.montecarlo import simulate_garch

    n = 1800
    idx = pd.bdate_range("2010", periods=n)
    r = simulate_garch(n, 2, 1e-6, 0.1, 0.85, mu=0.0003, seed=5)
    b = bundle_from_prices(pd.DataFrame(100 * np.cumprod(1 + r.T, axis=0), index=idx, columns=["A", "B"]), name="garch-world")
    model = MODELS.create("garch_vol_managed", min_train=500, refit_every=600)
    s = model.score(b)
    assert s["A"].dropna().between(0, 3.0).all() and 0.7 < s["A"].dropna().mean() < 1.4
    realised = b.returns["A"].rolling(21).std()
    both = pd.concat([s["A"], realised], axis=1).dropna()
    assert both.corr().iloc[0, 1] < -0.4                                                  # exposure is low exactly when volatility is high
    wf = walk_forward_volatility(b.returns["A"].dropna(), "garch", "normal", 500, 600)
    typical = ((wf["forecast"] ** 2).reindex(idx).shift(-1)).expanding(min_periods=252).mean()
    expected = (typical / (wf["forecast"] ** 2).reindex(idx).shift(-1)).clip(upper=3.0)
    assert s["A"].dropna().to_numpy() == pytest.approx(expected.reindex(s["A"].dropna().index).to_numpy())
    assert check_causality(model, b, cutoff=b.index[1500])["ok"]
    with pytest.raises(ValueError):
        MODELS.create("garch_vol_managed", model="x")


def test_evt_managed_exposure_responds_to_the_tail_and_is_causal():
    rng = np.random.default_rng(6)
    n = 1800
    scale = np.where((np.arange(n) > 1200) & (np.arange(n) < 1300), 3.0, 1.0)
    r = rng.standard_t(4, n) * 0.006 * scale / np.sqrt(2)
    b = bundle_from_prices(pd.DataFrame({"A": 100 * np.cumprod(1 + r)}, index=pd.bdate_range("2010", periods=n)), name="tail")
    model = MODELS.create("evt_risk_managed", min_obs=600, refit_every=126)
    s = model.score(b)["A"]
    assert s.iloc[1290:1320].mean() < s.iloc[1100:1190].mean() and s.dropna().between(0, 3.0).all()
    assert check_causality(model, b, cutoff=b.index[1500])["ok"]
    with pytest.raises(ValueError):
        MODELS.create("evt_risk_managed", p=0.5)


def test_bvar_lead_lag_finds_a_planted_lead():
    rng = np.random.default_rng(7)
    months = 240
    idx = pd.bdate_range("2000-01-03", periods=months * 21)
    month_id = np.arange(len(idx)) // 21
    m_ret = rng.normal(0, 0.04, (months, 6))
    m_ret[1:, 1] = 1.0 * m_ret[:-1, 0] + rng.normal(0, 0.015, months - 1)          # B follows A by one month
    daily = m_ret[month_id] / 21
    px = pd.DataFrame(100 * np.cumprod(1 + daily, axis=0), index=idx, columns=list("ABCDEF"))
    b = bundle_from_prices(px, name="leadlag")
    model = MODELS.create("bvar_lead_lag", tightness=0.3, min_months=60)
    score = model.score(b)
    ends = [idx[(m + 1) * 21 - 1] for m in range(months - 1)]
    fc = score.loc[ends, "B"].dropna()
    realised = pd.Series(m_ret[1:, 1], index=ends).reindex(fc.index)
    assert np.corrcoef(fc, realised)[0, 1] > 0.45
    quiet = score.loc[ends, "C"].dropna()
    assert abs(np.corrcoef(quiet, pd.Series(m_ret[1:, 2], index=ends).reindex(quiet.index))[0, 1]) < 0.4


def _context(bundle, mean_value, confidence_value, horizon=21):
    cfg = load_config()
    mean = pd.DataFrame(mean_value, index=bundle.index, columns=bundle.assets) if np.isscalar(mean_value) else mean_value
    std = pd.DataFrame(0.05, index=bundle.index, columns=bundle.assets)
    conf = pd.DataFrame(confidence_value, index=bundle.index, columns=bundle.assets)
    return Context(bundle, cfg, forecasts=ForecastPanel(mean, std, conf, horizon, "test"))


def test_black_litterman_allocator_moves_from_the_market_toward_confident_views(bundle):
    low = ALLOCATORS.create("black_litterman").build(_context(bundle, 0.0, 0.05))
    flat_row = low.iloc[-1]
    assert flat_row.sum() == pytest.approx(1.0, abs=1e-3) and flat_row.std() < 0.05           # no conviction: roughly the equal-weight market
    view = pd.DataFrame(0.0, index=bundle.index, columns=bundle.assets)
    view["A"] = 0.03
    sure = ALLOCATORS.create("black_litterman").build(_context(bundle, view, 0.95)).iloc[-1]
    unsure = ALLOCATORS.create("black_litterman").build(_context(bundle, view, 0.10)).iloc[-1]
    assert sure["A"] >= unsure["A"] - 1e-9 and unsure["A"] >= flat_row["A"] - 1e-6 and sure["A"] > flat_row["A"] + 0.05 and sure["A"] <= 0.25 + 1e-6
    out = ALLOCATORS.create("black_litterman").build(_context(bundle, 0.0, 0.5))
    changes = out.diff().abs().sum(axis=1)
    stamped = changes[changes > 1e-12].index
    month_ends = set(pd.Series(bundle.index, index=bundle.index).groupby(bundle.index.to_period("M")).max())
    assert set(stamped) <= month_ends and (out.iloc[:252] == 0).all().all()                                  # only at month-ends; nothing before a year of history


def test_kelly_allocator_scales_with_fraction_and_respects_caps(bundle):
    view = pd.DataFrame(0.0, index=bundle.index, columns=bundle.assets)
    view["A"], view["B"] = 0.02, -0.02
    q = ALLOCATORS.create("kelly", fraction=0.25, long_only=False, max_gross=2.0).build(_context(bundle, view, 1.0)).iloc[-1]
    h = ALLOCATORS.create("kelly", fraction=0.5, long_only=False, max_gross=2.0).build(_context(bundle, view, 1.0)).iloc[-1]
    assert q["A"] > 0 > q["B"] and h["A"] >= q["A"] and h.abs().sum() <= 2.0 + 1e-9
    longs = ALLOCATORS.create("kelly", fraction=0.5).build(_context(bundle, view, 1.0)).iloc[-1]
    assert (longs >= 0).all() and longs.abs().sum() <= 1.0 + 1e-9 and longs["B"] == 0
    weighted = ALLOCATORS.create("kelly", fraction=0.5, long_only=False).build(_context(bundle, view, 0.2)).iloc[-1]
    assert weighted.abs().sum() < h.abs().sum()
    with pytest.raises(ValueError):
        ALLOCATORS.create("kelly", fraction=2.0)
    with pytest.raises(ValueError):
        ALLOCATORS.create("kelly").build(Context(bundle, load_config()))


def test_econometric_models_and_allocators_run_through_the_pipeline(bundle):
    cfg = load_config()
    for model, alloc in [("kalman_trend", {"allocator": "sleeves"}), ("bvar_lead_lag", {"allocator": "kelly", "params": {"fraction": 0.5}}),
                         ("tsmom", {"allocator": "black_litterman"})]:
        spec = PipelineSpec.from_dict({"name": f"{model}-{alloc['allocator']}", "models": [{"name": model}], "allocation": alloc})
        result = Pipeline(spec, cfg, bundle).run()
        assert np.isfinite(result.net_returns.dropna()).all() and len(result.net_returns.dropna()) > 500
