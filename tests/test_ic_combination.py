"""The optimal-IC and orthogonal-IC combination rules: planted truth (two copies, one independent source, one noise), causality, and the pipeline wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity.alpha_model import information_coefficients
from src.framework import Pipeline, bundle_from_prices, register_model
from src.framework.forecasting import ForecastModel, combine_forecasts, ic_trust_weights
from src.framework.ic_combination import optimal_ic_weights, orthogonal_ic_forecast
from src.framework.types import ForecastPanel
from src.signals.alpha_engine import forward_returns
from src.utils.config import load_config


def _world(seed, n=1800, k=20):
    """Next-day returns load on a shared source ``s`` and an independent source ``u``. A and B are two noisy copies of ``s``, C is ``u``, D is noise; all have forecast horizon one day."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    cols = [f"S{i:02d}" for i in range(k)]
    s, u, e = rng.normal(size=(n, k)), rng.normal(size=(n, k)), rng.normal(size=(n, k))
    r = np.zeros((n, k))
    r[1:] = 0.1 * s[:-1] + 0.1 * u[:-1]
    frame = lambda x: pd.DataFrame(x, index=idx, columns=cols)
    signals = {"A": 0.8 * s + 0.6 * rng.normal(size=(n, k)), "B": 0.8 * s + 0.6 * rng.normal(size=(n, k)), "C": u, "D": rng.normal(size=(n, k))}
    spread = frame(np.full((n, k), 0.01))
    panels = {name: ForecastPanel.from_mean_std(frame(0.0005 * x), spread, 1, name) for name, x in signals.items()}
    return panels, frame((r + 0.99 * e) * 0.01)


def _ic(panel, returns, start=800):
    return information_coefficients({"x": panel.mean}, forward_returns(returns, 1), "pearson", 5)["x"].iloc[start:]


@pytest.mark.parametrize("seed", [0, 1])
def test_the_independent_source_gets_more_than_either_copy_and_noise_gets_nothing(seed):
    panels, returns = _world(seed)
    raw, expected = optimal_ic_weights(panels, returns)
    share = raw.div(raw.sum(axis=1), axis=0)
    late = share.iloc[800:].mean()
    assert abs(late.sum() - 1) < 1e-9 and (raw >= 0).all().all()
    copies = late[["A", "B"]].mean()
    assert late["C"] > 1.4 * copies and late["C"] > late[["A", "B"]].max()      # two copies of one source are worth less each than one independent source
    assert late["D"] < 0.05                                                    # a model with no information is left out
    trusted = ic_trust_weights(panels, returns, window=504, min_obs=126)       # the existing rule sees only each model's own IC, whose levels differ by 25%: the covariance is what separates the copies
    trusted = trusted.div(trusted.sum(axis=1), axis=0).iloc[800:].mean()
    assert late["C"] / copies > 1.15 * trusted["C"] / trusted[["A", "B"]].mean()
    assert expected.iloc[:250].isna().all() and expected.iloc[-1] > 0.3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_combination_beats_equal_weights_and_delivers_the_ir_it_expected(seed):
    panels, returns = _world(seed)
    raw, expected = optimal_ic_weights(panels, returns)
    best, equal = combine_forecasts(panels, "given", raw), combine_forecasts(panels, "equal")
    ic_best, ic_equal = _ic(best, returns), _ic(equal, returns)
    ir = lambda s: s.mean() / s.std()
    assert ir(ic_best) > ir(ic_equal) and ic_best.mean() > ic_equal.mean()
    assert abs(ir(ic_best) - expected.iloc[800:].mean()) < 0.2 * expected.iloc[800:].mean()          # the ratio the data promised is the ratio it delivered


def test_the_orthogonal_rule_returns_one_forecast_that_is_just_as_good_and_depends_on_the_order():
    panels, returns = _world(3)
    forecast, w, expected = orthogonal_ic_forecast(panels, returns, "gram_schmidt")
    assert isinstance(forecast, ForecastPanel) and forecast.mean.shape == panels["A"].mean.shape and (forecast.std.iloc[300:] > 0).all().all()
    equal = combine_forecasts(panels, "equal")
    ir = lambda s: s.mean() / s.std()
    assert ir(_ic(forecast, returns)) > ir(_ic(equal, returns))
    assert abs(w.sum(axis=1) - 1).max() < 1e-9 and (w >= 0).all().all()
    assert w["D"].iloc[800:].mean() < 0.05
    # in return units: the composite is as dispersed as the models are
    assert 0.3 < forecast.mean.iloc[800:].std(axis=1).mean() / panels["A"].mean.iloc[800:].std(axis=1).mean() < 3.0
    # the model listed first keeps what it shares with the others, so the copy listed second is left with little and swapping the copies swaps their weights
    late = slice(800, None)
    assert w["A"].iloc[late].mean() > w["B"].iloc[late].mean() + 0.05
    _, w2, _ = orthogonal_ic_forecast({k: panels[k] for k in ("B", "A", "C", "D")}, returns, "gram_schmidt")
    assert w2["B"].iloc[late].mean() > w2["A"].iloc[late].mean() + 0.05
    sym, ws, _ = orthogonal_ic_forecast(panels, returns, "symmetric")
    assert ir(_ic(sym, returns)) > ir(_ic(equal, returns)) and ws["D"].iloc[800:].mean() < 0.05


def test_the_weights_on_a_date_never_use_returns_that_had_not_happened():
    panels, returns = _world(4, n=1200)
    j = 900
    base, _ = optimal_ic_weights(panels, returns)
    other = returns.copy()
    other.iloc[j:] = np.random.default_rng(0).normal(0, 0.02, other.iloc[j:].shape)
    changed, _ = optimal_ic_weights(panels, other)
    assert base.iloc[:j].equals(changed.iloc[:j])
    assert np.abs(base.iloc[j + 40:].to_numpy() - changed.iloc[j + 40:].to_numpy()).max() > 1e-6
    f0, _, _ = orthogonal_ic_forecast(panels, returns)
    f1, _, _ = orthogonal_ic_forecast(panels, other)
    assert f0.mean.iloc[:j].equals(f1.mean.iloc[:j])


def test_unknown_orthogonalisation_and_too_little_history_are_handled():
    panels, returns = _world(5, n=400)
    with pytest.raises(ValueError, match="orthogonalize"):
        orthogonal_ic_forecast(panels, returns, "nope")
    raw, expected = optimal_ic_weights(panels, returns, min_obs=500)             # more matured ICs than there are days: equal weights of the standardised forecasts, no promise made
    share = raw.div(raw.sum(axis=1), axis=0)
    assert expected.isna().all()
    spread = {n: p.mean.std(axis=1) for n, p in panels.items()}
    assert np.allclose((raw * pd.DataFrame(spread)).to_numpy()[10:], 1.0 / 4)
    assert np.allclose(share.sum(axis=1), 1.0)


# ------------------------------------------------------------------------------------------------------------------ the pipeline
@register_model("test_ic_trend_fast", "test", "a short trend used by the IC combination tests")
class _Fast(ForecastModel):
    name, family = "test_ic_trend_fast", "test"

    def score(self, data):
        return data.prices.pct_change(10).where(data.investable)


@register_model("test_ic_trend_slow", "test", "a long trend used by the IC combination tests")
class _Slow(ForecastModel):
    name, family = "test_ic_trend_slow", "test"

    def score(self, data):
        return data.prices.pct_change(120).where(data.investable)


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(11)
    n = 3000
    idx = pd.bdate_range("2008-01-01", periods=n)
    drift = np.sin(np.arange(n) / 60.0) * 0.0008
    r = pd.DataFrame(rng.normal(0, 0.01, (n, 8)) + drift[:, None], index=idx, columns=list("ABCDEFGH"))
    return bundle_from_prices(100 * (1 + r).cumprod(), min_history=60, name="ic")


@pytest.mark.parametrize("rule", ["optimal_ic", "orthogonal_ic"])
def test_the_pipeline_runs_both_rules_and_reports_what_they_did(bundle, rule):
    spec = {"name": rule, "models": [{"name": "test_ic_trend_fast"}, {"name": "test_ic_trend_slow"}], "combination": {"rule": rule, "min_obs": 126, "window": 504}}
    result = Pipeline(spec, load_config(), bundle).run()
    assert np.isfinite(result.metrics["sharpe"])
    table = result.tables["ic_combination"]
    assert list(table.index) == ["test_ic_trend_fast", "test_ic_trend_slow"] and abs(table["mean_weight"].sum() - 1) < 1e-9
    assert result.validation["causality"]["test_ic_trend_fast"]["ok"]
    assert "attribution_by_model" in result.tables and len(result.tables["attribution_by_model"]) >= 4          # the model-level attribution uses the shares the rule reports
    assert result.forecasts.mean.loc[result.start:].notna().any().any()


def test_the_rules_are_offered_by_the_web_api_and_accepted_in_a_spec():
    from src.webapp.api import COMBINATIONS

    assert {"optimal_ic", "orthogonal_ic"} <= set(COMBINATIONS)
    assert all(isinstance(text, str) and text for text in COMBINATIONS.values())
