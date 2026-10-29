"""The verdicts a backtest result carries, checked against the definitions.

``framework/validate.py`` is what turns a return stream and a forecast panel into the figures a reader trusts: the probability that the Sharpe ratio is real, the rank correlation and hit rate of the
forecasts, whether a model looks at the future, and the tables of an evaluation (by period, by asset, against benchmarks). Each is recomputed here from its definition on data small enough to work through.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, norm, skew

from src.framework import ForecastModel, Pipeline, bundle_from_prices
from src.framework import validate
from src.framework.registry import MODELS, register_model
from src.framework.types import ForecastPanel
from src.framework.validate import check_causality, deflated_probability, forecast_quality
from src.utils.config import load_config


class Spy:
    def __init__(self, result=None):
        self.result, self.calls = result, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


# ------------------------------------------------------------------------------------------------------------------ the deflated Sharpe ratio
def test_with_one_trial_it_is_the_probability_that_the_true_sharpe_is_positive_given_the_sample_length_skew_and_kurtosis():
    net = pd.Series(np.random.default_rng(3).normal(0.0006, 0.01, 500))
    sr = net.mean() / net.std(ddof=1)                                                                  # the daily Sharpe ratio
    g3, g4 = skew(net, bias=False), kurtosis(net, fisher=False, bias=False)
    expected = norm.cdf(sr / np.sqrt((1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2) / (len(net) - 1)))
    assert deflated_probability(net, 1) == pytest.approx(expected) and deflated_probability(net, 0) == pytest.approx(expected), "nothing to deflate for"
    assert 0.5 < expected < 1.0
    assert deflated_probability(pd.concat([pd.Series([np.nan] * 5), net], ignore_index=True), 1) == pytest.approx(expected), "missing days are not days"
    assert deflated_probability(net[::-1].reset_index(drop=True) * -1.0, 1) < 0.5, "a losing strategy is probably not a winning one"


def test_it_needs_thirty_days_and_some_variation_and_with_two_or_more_trials_it_is_the_deflated_ratio(monkeypatch):
    net = pd.Series(np.random.default_rng(3).normal(0.0006, 0.01, 500))
    assert np.isnan(deflated_probability(net[:29], 1)) and np.isfinite(deflated_probability(net[:30], 1)), "thirty days are the least"
    assert np.isnan(deflated_probability(pd.Series(0.001, index=range(100)), 1)) and np.isnan(deflated_probability(pd.Series(0.0, index=range(100)), 1)), "a series that never moves has no Sharpe ratio"
    spy = Spy(0.42)
    monkeypatch.setattr(validate, "deflated_sharpe_ratio", spy)
    assert deflated_probability(net, 1) != 0.42 and spy.calls == []
    assert deflated_probability(net, 2) == 0.42 and deflated_probability(net, 50) == 0.42
    (args, _), = spy.calls[:1]
    assert args[0] == pytest.approx(np.sqrt(252.0) * net.mean() / net.std(ddof=1)) and args[1:3] == (2, 500) and args[3] == pytest.approx(net.skew()) and args[4] == pytest.approx(net.kurt() + 3.0), "the annualised ratio, the trials, the days, the skewness and the kurtosis"
    assert spy.calls[1][0][1] == 50


# ------------------------------------------------------------------------------------------------------------------ the quality of the forecasts
RETURNS = np.array([-0.002, -0.001, 0.0005, 0.001, 0.002, 0.003])                                       # constant daily returns of six assets, so the forward return of five days is known
HORIZON = 5


def panel(scale: float = 1.0, days: int = 60):
    index = pd.bdate_range("2021-01-04", periods=days)
    returns = pd.DataFrame(np.tile(RETURNS, (days, 1)), index=index, columns=list("ABCDEF"))
    mean = pd.DataFrame(np.tile(RETURNS * HORIZON * scale, (days, 1)), index=index, columns=returns.columns)
    return ForecastPanel.from_mean_std(mean, mean * 0 + 0.02, horizon=HORIZON), returns, index


def test_a_forecast_that_ranks_the_assets_as_they_turn_out_has_a_rank_correlation_of_one_and_every_hit():
    forecasts, returns, index = panel()
    out = forecast_quality(forecasts, returns, index[10])
    p = norm.cdf(RETURNS * HORIZON / 0.02)                                                              # the Gaussian probability of a rise, asset by asset
    up = (RETURNS > 0).astype(float)
    assert out["mean_rank_ic"] == pytest.approx(1.0) and out["ic_observations"] == 9, "a decision every five days from the 11th, the last whose five days are over is the 51st"
    assert out["hit_rate"] == 1.0 and out["mean_confidence"] == pytest.approx(np.abs(2.0 * p - 1.0).mean())
    assert out["brier"] == pytest.approx(((p - up) ** 2).mean()) and out["brier_climatology"] == pytest.approx((4 / 6) * (2 / 6)), "the Brier score, and what a forecast of the base rate would score"
    table = out["reliability"]
    bins = np.ceil(p * 10).astype(int)
    expected = [(float(p[bins == b].mean()), float(up[bins == b].mean()), int(45 * (bins == b).sum())) for b in sorted(set(bins))]
    assert list(table.columns) == ["predicted", "realised", "n"] and len(table) == len(expected) == 5
    assert table["predicted"].tolist() == pytest.approx([e[0] for e in expected]) and table["realised"].tolist() == pytest.approx([e[1] for e in expected]), "each tenth of the probability: the mean forecast and how often it rose"
    assert table["n"].tolist() == [e[2] for e in expected], "and the cells in it: 45 days of each asset"
    assert set(out) == {"mean_rank_ic", "ic_observations", "hit_rate", "mean_confidence", "brier", "brier_climatology", "reliability"}


def test_the_evaluation_starts_where_it_is_told_and_a_reversed_forecast_scores_minus_one_and_no_hits():
    forecasts, returns, index = panel()
    late = forecast_quality(forecasts, returns, index[40])
    assert late["ic_observations"] == 3 and late["reliability"]["n"].sum() == 6 * 15, "fifteen days from the 41st to the 55th whose five days are over"
    flipped, _, _ = panel(scale=-1.0)
    out = forecast_quality(flipped, returns, index[10])
    assert out["mean_rank_ic"] == pytest.approx(-1.0) and out["hit_rate"] == 0.0 and out["ic_observations"] == 9


def test_a_forecast_of_nothing_is_not_a_hit_and_one_with_no_view_has_no_scores():
    forecasts, returns, index = panel()
    zero = ForecastPanel.from_mean_std(forecasts.mean * 0.0, forecasts.std, horizon=HORIZON)
    out = forecast_quality(zero, returns, index[10])
    assert out["hit_rate"] == 0.0 and out["ic_observations"] == 0 and np.isnan(out["mean_rank_ic"]), "no sign, no ranking"
    partly = forecasts.mean.copy()
    partly["A"] = 0.0
    out = forecast_quality(ForecastPanel.from_mean_std(partly, forecasts.std, horizon=HORIZON), returns, index[10])
    assert out["hit_rate"] == 1.0, "a forecast of exactly zero says nothing: it is neither a hit nor a miss"
    blank = ForecastPanel.from_mean_std(forecasts.mean * np.nan, forecasts.std, horizon=HORIZON)
    empty = forecast_quality(blank, returns, index[10])
    assert np.isnan(empty["brier"]) and np.isnan(empty["brier_climatology"]) and np.isnan(empty["mean_confidence"]) and len(empty["reliability"]) == 0 and empty["hit_rate"] == 0.0


def test_a_calibrated_probability_replaces_the_gaussian_one_in_the_score_and_the_reliability():
    forecasts, returns, index = panel()
    forecasts.p_up = forecasts.mean * 0.0 + 0.9
    out = forecast_quality(forecasts, returns, index[10])
    assert out["brier"] == pytest.approx((4 * 0.01 + 2 * 0.81) / 6) and len(out["reliability"]) == 1
    row = out["reliability"].iloc[0]
    assert row["predicted"] == pytest.approx(0.9) and row["realised"] == pytest.approx(4 / 6) and row["n"] == 270
    assert out["mean_confidence"] == pytest.approx(np.abs(2.0 * norm.cdf(RETURNS * HORIZON / 0.02) - 1.0).mean()), "the confidence is the forecast's own"


# ------------------------------------------------------------------------------------------------------------------ the look-ahead test
def bundle_of(days: int = 300, seed: int = 4):
    index = pd.bdate_range("2019-01-02", periods=days)
    steps = np.random.default_rng(seed).normal(0.0004, 0.01, (days, 6))
    return bundle_from_prices(pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=list("ABCDEF")), name="causal", min_history=40)


class Causal:
    def forecast(self, bundle):
        return SimpleNamespace(mean=bundle.returns.rolling(20).mean())


class Peeking:
    def forecast(self, bundle):
        return SimpleNamespace(mean=bundle.returns.shift(-1))


def test_a_model_that_uses_only_the_past_is_unchanged_when_the_future_is_noise_and_one_that_peeks_is_not():
    bundle = bundle_of()
    cutoff = bundle.index[int(0.7 * len(bundle.index))]
    good = check_causality(Causal(), bundle)
    assert good == {"ok": True, "max_abs_difference": 0.0, "availability_mismatches": 0, "cutoff": str(cutoff.date()), "compared": 6 * (bundle.index.get_loc(cutoff) - 19)}, "from the 21st day (the first return is missing, and the mean needs twenty) to the cutoff"
    bad = check_causality(Peeking(), bundle)
    assert bad["ok"] is False and bad["max_abs_difference"] > 0.01 and bad["availability_mismatches"] == 0
    early = bundle.index[100]
    assert check_causality(Causal(), bundle, cutoff=early)["cutoff"] == str(early.date()) and check_causality(Causal(), bundle, cutoff=str(early.date()))["compared"] == 6 * (100 - 19)
    assert check_causality(Peeking(), bundle, cutoff=early)["ok"] is False


def test_a_difference_is_forgiven_up_to_the_tolerance_and_a_gap_that_depends_on_the_future_is_not():
    bundle = bundle_of()

    class Faint:
        def forecast(self, bundle):
            return SimpleNamespace(mean=bundle.returns.shift(-1) * 1e-12)

    assert check_causality(Faint(), bundle)["ok"] is True and 0.0 < check_causality(Faint(), bundle)["max_abs_difference"] < 1e-9
    assert check_causality(Faint(), bundle, tolerance=1e-15)["ok"] is False

    real_tomorrow = bundle.returns.shift(-1)

    class Selective:
        def forecast(self, other):
            return SimpleNamespace(mean=other.returns.where(other.returns.shift(-1) == real_tomorrow.reindex(other.index)))

    result = check_causality(Selective(), bundle)
    assert result["availability_mismatches"] == 6 and result["max_abs_difference"] == 0.0 and result["ok"] is False, "the values agree where both exist, and six of them exist on one side only"


def test_a_regime_detector_is_judged_on_its_probabilities():
    bundle = bundle_of()

    class Causal_detector:
        def detect(self, data):
            return SimpleNamespace(probabilities=data.returns.rolling(10).mean())

    class Peeking_detector:
        def detect(self, data):
            return SimpleNamespace(probabilities=data.returns.shift(-1))

    assert check_causality(Causal_detector(), bundle)["ok"] and not check_causality(Peeking_detector(), bundle)["ok"]


# ------------------------------------------------------------------------------------------------------------------ the tables of an evaluation
class Trend(ForecastModel):
    name = "test_validate_trend"

    def score(self, data):
        return data.prices.pct_change(40)


class Overrides:
    """The platform's configuration with a few keys answered differently."""

    def __init__(self, base, values):
        self.base, self.values = base, values

    def get(self, key, default=None):
        return self.values[key] if key in self.values else self.base.get(key, default)

    def __getattr__(self, name):
        return getattr(self.base, name)


@pytest.fixture(autouse=True)
def registered():
    register_model("test_validate_trend")(Trend)
    yield
    MODELS.unregister("test_validate_trend")


@pytest.fixture(scope="module")
def market():
    steps = np.random.default_rng(5).normal(0.0, 0.01, (1300, 6)) * np.linspace(0.6, 1.6, 6) + np.linspace(-0.0006, 0.0010, 6)
    index = pd.bdate_range("2017-01-02", periods=1300)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=list("ABCDEF"))
    return bundle_from_prices(prices, name="classes", min_history=40, asset_class={"A": "equity", "B": "equity", "C": "bond", "D": "bond", "E": "commodity", "F": "commodity"})


@pytest.fixture(scope="module")
def config():
    return load_config()


def run(market, config, **evaluation):
    spec = {"name": "t", "models": [{"name": "test_validate_trend"}], "evaluation": {"benchmarks": [], "n_trials": 1, "causality": False, "brinson": False, "model_attribution": False, **evaluation}}
    return Pipeline(spec, config, market).run(validate=False)


def test_the_return_is_attributed_to_each_asset_each_class_and_the_costs_and_the_rows_add_to_the_net_return(market, config):
    out = run(market, config)
    net = out.net_returns.loc[out.start:].dropna()
    held = out.weights.shift(1).reindex(net.index)
    contribution = 100.0 * (held * market.returns.reindex(net.index)).sum()
    by_asset = out.tables["attribution_by_asset"]
    assert list(by_asset.columns) == ["contribution_pct", "average_weight", "asset_class"] and by_asset["contribution_pct"].is_monotonic_decreasing
    assert by_asset["contribution_pct"].to_dict() == pytest.approx(contribution.to_dict()) and by_asset["average_weight"].to_dict() == pytest.approx(held.mean().to_dict())
    assert by_asset["asset_class"].to_dict() == {"A": "equity", "B": "equity", "C": "bond", "D": "bond", "E": "commodity", "F": "commodity"}
    table = out.tables["attribution"]
    assert table.index.tolist() == ["bond", "commodity", "equity", "costs", "other (compounding, lag alignment)"] and list(table.columns) == ["return_points"]
    assert table.loc["bond", "return_points"] == pytest.approx(contribution[["C", "D"]].sum()) and table.loc["equity", "return_points"] == pytest.approx(contribution[["A", "B"]].sum())
    assert table.loc["costs", "return_points"] == pytest.approx(-100.0 * out.costs.reindex(net.index).sum()) and table.loc["costs", "return_points"] < 0.0
    assert table["return_points"].sum() == pytest.approx(100.0 * net.sum()), "the rows add up to the sum of the daily net returns"


def test_the_return_is_described_in_each_sample_window_that_has_more_than_twenty_days(market, config):
    samples = {"early": {"start": "2019-01-01", "end": "2019-12-31"}, "late": {"start": "2020-01-01", "end": None}, "tiny": {"start": "2021-12-01", "end": "2021-12-20"}, "before": {"start": "2010-01-01", "end": "2010-12-31"}}
    out = run(market, Overrides(config, {"backtest.samples": samples}))
    table = out.tables["by_sample"]
    net = out.net_returns.loc[out.start:].dropna()
    assert table.index.tolist() == ["early", "late"] and list(table.columns) == ["sharpe", "cagr", "ann_vol", "max_drawdown", "days"], "a window of twenty days or fewer, or none, is left out"
    early, late = net.loc["2019-01-01":"2019-12-31"], net.loc["2020-01-01":]
    assert table.loc["early", "days"] == len(early) and table.loc["late", "days"] == len(late)
    assert table.loc["early", "ann_vol"] == pytest.approx(early.std(ddof=1) * np.sqrt(252.0)) and table.loc["late", "sharpe"] == pytest.approx(np.sqrt(252.0) * late.mean() / late.std(ddof=1))
    assert table.loc["late", "max_drawdown"] == pytest.approx(((1 + late).cumprod() / (1 + late).cumprod().cummax() - 1.0).min())


def test_each_benchmark_is_run_by_its_name_and_a_name_that_is_not_one_or_a_book_with_nothing_in_it_is_a_row_that_says_so(market, config):
    bootstrap = {"framework.tearsheet.bootstrap": {"n_samples": 100, "block_length": 21, "seed": 7}}
    out = run(market, Overrides(config, bootstrap), benchmarks=["equal_weight", "M2_risk_parity", "bogus"])
    table = out.tables["benchmarks"]
    assert table["benchmark"].tolist() == ["equal_weight", "M2_risk_parity", "bogus"] and set(out.benchmarks) == {"equal_weight", "M2_risk_parity"}, "listed under the name asked for; the prefix of the platform's ladder is only how the book is found"
    good = table.iloc[0]
    assert good["difference"] == pytest.approx(good["sharpe_strategy"] - good["sharpe_benchmark"]) and 0.0 <= good["p_value"] <= 1.0
    net, bench = out.net_returns.loc[out.start:].dropna(), out.benchmarks["equal_weight"]
    common = net.index.intersection(bench.index)
    assert good["sharpe_strategy"] == pytest.approx(np.sqrt(252.0) * net.loc[common].mean() / net.loc[common].std(ddof=1)), "both measured on the days they share"
    assert table.iloc[2]["error"] == "\"unknown benchmark 'bogus'\"" and table.iloc[2][["sharpe_strategy", "p_value"]].isna().all()
    small = bundle_from_prices(market.prices[["A", "B", "C", "D"]], name="four", min_history=40)
    four = run(small, config, benchmarks=["equal_weight"])
    assert four.tables["benchmarks"].iloc[0]["error"] == "the benchmark held nothing on these dates (the portfolio builders need at least five investable assets)" and four.benchmarks == {}
    default = Pipeline({"name": "t", "models": [{"name": "test_validate_trend"}], "evaluation": {"causality": False, "brinson": False, "model_attribution": False}}, Overrides(config, {"framework.tearsheet.benchmarks": ["inverse_vol"], **bootstrap}), market).run(validate=False)
    assert default.tables["benchmarks"]["benchmark"].tolist() == ["inverse_vol"], "the platform's list when the specification names none"
