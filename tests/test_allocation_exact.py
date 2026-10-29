"""The allocators that turn forecasts and scores into target weights, checked against numbers worked out by hand.

Every strategy of the Backtest tab goes through one of these (the forecast stack, the score stack, per-asset sleeves, confidence sizing, a model's own weights, or a blend of them by regime), and the
pipeline-level tests that touched them take minutes each, which is why a planted-bug audit could not be pointed at them. These run in a second and state, for a small market, what each allocator holds on
each day and why: the clip and the equal slice of a sleeve, the volatility scale and its ceiling, a confidence floor, a gross and a cap, the blend and its fallback before a detector has a view.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.features.volatility import rolling_volatility
from src.framework.allocation import (
    ALLOCATORS,
    BOOKS,
    LADDER,
    Allocator,
    ConfidenceSized,
    Context,
    ForecastStack,
    ModelWeights,
    RegimeSwitch,
    ScoreStack,
    Sleeves,
    StaticAllocator,
    _tilted,
    book,
)
from src.framework.registry import register_allocator
from src.framework.types import ForecastPanel, RegimeSeries
from src.signals.transform import signal_to_positions


class Config:
    """The two things an allocator asks of a configuration."""

    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class Model:
    """A forecast model reduced to what an allocator reads: a score and, for a structured model, its own weights."""

    name = "stub"

    def __init__(self, score=None, weights=None):
        self._score, self._weights = score, weights

    def score(self, data):
        return self._score

    def weights(self, data):
        return self._weights


def market(n: int = 120, columns: str = "ABCD", seed: int = 5, investable=None) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-04", periods=n)
    returns = pd.DataFrame(rng.normal(0.0, 0.01, (n, len(columns))) * np.linspace(1.0, 3.0, len(columns)), index=index, columns=list(columns))
    mask = pd.DataFrame(True, index=index, columns=list(columns)) if investable is None else investable
    return SimpleNamespace(returns=returns, investable=mask, index=index, assets=list(columns), group={}, market=None, name="test")


def table(rows, columns="ABCD", start="2021-01-04") -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.bdate_range(start, periods=len(rows)), columns=list(columns), dtype=float)


# ------------------------------------------------------------------------------------------------------------------ the context and the books
def test_the_position_stack_settings_are_read_from_the_configuration_with_the_documented_defaults():
    ctx = Context(market(), Config())
    assert ctx.transform() == {"winsorize_quantile": 0.02, "cross_sectional": True, "scale": "zscore", "clip": 3.0, "target_vol": 0.10, "max_abs_weight": 0.25, "gross_leverage": 1.0, "long_only_book": False}
    custom = Context(market(), Config(**{"strategies.transform": {"winsorize_quantile": 0.05, "cross_sectional": False, "scale": "rank", "clip": None, "max_abs_weight": 0.4, "gross_leverage": 1.5, "long_only": True},
                                         "portfolio.volatility.target_vol": 0.15}))
    assert custom.transform() == {"winsorize_quantile": 0.05, "cross_sectional": False, "scale": "rank", "clip": None, "target_vol": 0.15, "max_abs_weight": 0.4, "gross_leverage": 1.5, "long_only_book": True}
    assert ctx.models == [] and ctx.forecasts is None and ctx.regimes is None and ctx.cache == {}


def test_the_limits_come_from_the_configuration_and_the_bundles_own_groups_win_over_the_configurations():
    class Limits(Config):
        group_map = {"A": "config_group"}

    config = Limits(**{"portfolio.constraints": {"max_weight": 0.4, "group_limits": {"eq": 0.5}}})
    own = Context(SimpleNamespace(**{**market().__dict__, "group": {"A": "eq", "B": "eq"}}), config).constraints()
    assert own.max_weight == 0.4 and own.group_limits == {"eq": 0.5} and own.group_map == {"A": "eq", "B": "eq"}, "the groups of the bundle's own tickers"
    fallback = Context(market(), config).constraints()
    assert fallback.group_map == {"A": "config_group"}, "a bundle with no groups takes the configuration's"


def test_a_book_is_built_once_per_context_from_the_ladder_or_from_the_optimiser_and_a_users_universe_never_touches_the_platforms_cache(monkeypatch):
    import experiments.strategies as strategies
    import src.portfolio.bayesian as bayesian

    b = market(n=30)
    built = pd.DataFrame(0.25, index=b.index[:20], columns=b.assets)
    calls = []

    def build_ladder(m, config, include):
        calls.append(("build", include))
        return {include[0]: built}

    def cached_ladder(m, config, directory, include):
        calls.append(("cached", include, directory))
        return {include[0]: built * 2.0}

    monkeypatch.setattr(strategies, "build_ladder", build_ladder)
    monkeypatch.setattr(strategies, "cached_ladder", cached_ladder)
    own = SimpleNamespace(**{**b.__dict__, "as_market": lambda: "the market"})

    class Paths(Config):
        group_map: dict = {}

        def path(self, name):
            return f"data/{name}"

    ctx = Context(own, Paths())
    first = book("equal_weight", ctx)
    assert calls == [("build", ["M0_equal_weight"])], "your own prices are built, never cached"
    assert first.index.equals(b.index) and first.iloc[:20].eq(0.25).all().all() and first.iloc[20:].eq(0.25).all().all(), "laid on the bundle's dates and carried forward where the book stops"
    assert book("equal_weight", ctx) is first and len(calls) == 1, "once per context"
    platform = SimpleNamespace(**{**own.__dict__, "market": "the platform's"})
    cached = book("inverse_vol", Context(platform, Paths()))
    assert calls[-1] == ("cached", ["M1_inverse_vol"], "data/processed") and cached.iloc[0].tolist() == [0.5] * 4, "the platform's own data uses the cache"

    def broken(*args, **kwargs):
        raise OSError("cache unreadable")

    monkeypatch.setattr(strategies, "cached_ladder", broken)
    again = book("risk_parity", Context(platform, Paths()))
    assert calls[-1] == ("build", ["M2_risk_parity"]) and again.iloc[0].tolist() == [0.25] * 4, "a cache that cannot be read is built again"
    seen = {}

    def bayesian_book(returns, investable, kind, constraints, dates):
        seen.update(kind=kind, rows=len(returns), dates=len(dates), constraints=constraints)
        return built

    monkeypatch.setattr(bayesian, "bayesian_book", bayesian_book)
    mvo = book("mvo", Context(own, Paths(**{"portfolio.constraints": {"max_weight": 0.3}})))
    assert seen["kind"] == "mvo_sample" and seen["rows"] == 30 and seen["dates"] == 2 and seen["constraints"].max_weight == 0.3, "the mean-variance book is solved at each month's first day"
    assert mvo.index.equals(b.index)


def test_a_book_is_one_of_the_named_ladder_and_an_unknown_one_says_what_there_is():
    assert set(LADDER) == {"equal_weight", "inverse_vol", "risk_parity", "mean_cvar", "hrp", "herc"} and BOOKS == tuple(LADDER) + ("mvo",)
    with pytest.raises(KeyError, match="unknown book 'momentum'; choose from"):
        book("momentum", Context(market(), Config()))
    assert StaticAllocator().book == "risk_parity" and StaticAllocator("hrp").book == "hrp"
    assert [StaticAllocator(b).required_assets() for b in ("equal_weight", "inverse_vol", "risk_parity", "hrp", "mvo")] == [1, 1, 2, 2, 2], "a book that weighs assets by their covariance needs two to weigh between"


def test_the_base_allocator_builds_nothing_and_asks_for_one_asset_unless_told_otherwise():
    allocator = Allocator()
    assert allocator.required_assets() == 1 and allocator.name == "allocator"
    with pytest.raises(NotImplementedError):
        allocator.build(Context(market(), Config()))
    allocator.min_assets = 3
    assert allocator.required_assets() == 3


# ------------------------------------------------------------------------------------------------------------------ the forecast stack and the score stack
def forecasts(n: int = 120, seed: int = 6) -> ForecastPanel:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-04", periods=n)
    mean = pd.DataFrame(rng.normal(0.0, 0.01, (n, 4)), index=index, columns=list("ABCD"))
    std = pd.DataFrame(np.tile([0.01, 0.02, 0.03, 0.04], (n, 1)), index=index, columns=list("ABCD"))        # a different uncertainty for each asset, so dividing by it is not the same as multiplying
    return ForecastPanel.from_mean_std(mean, std)


def test_the_forecast_stack_applies_the_position_stack_to_the_mean_over_the_standard_deviation():
    b, fc = market(), forecasts()
    ctx = Context(b, Config(), forecasts=fc)
    signal = fc.mean / fc.std.where(fc.std > 0)
    vol = rolling_volatility(b.returns, 63)
    stack = dict(winsorize_quantile=0.02, cross_sectional=True, scale="zscore", clip=3.0, target_vol=0.10, max_abs_weight=0.25, gross_leverage=1.0, long_only_book=False)
    default = ForecastStack().build(ctx)
    pd.testing.assert_frame_equal(default, signal_to_positions(signal, vol, investable=b.investable, **stack))
    assert default.iloc[-1].abs().sum() == pytest.approx(1.0) and default.iloc[-1].abs().max() <= 0.25 + 1e-9, "a book of gross one, capped at a quarter"
    big = ForecastStack(gross=2.0, max_weight=0.6).build(ctx)
    pd.testing.assert_frame_equal(big, signal_to_positions(signal, vol, investable=b.investable, **{**stack, "gross_leverage": 2.0, "max_abs_weight": 0.6}))
    assert big.iloc[-1].abs().sum() == pytest.approx(2.0)
    longs = ForecastStack(long_only=True).build(ctx)
    pd.testing.assert_frame_equal(longs, signal_to_positions(signal, vol, investable=b.investable, **{**stack, "long_only_book": True}))
    assert (longs >= 0.0).all().all() and longs.iloc[-1].sum() <= 1.0 + 1e-9, "no shorts, and no more than a whole book"
    vol_30 = ForecastStack().build(Context(b, Config(**{"portfolio.volatility.lookback": 30}), forecasts=fc))
    pd.testing.assert_frame_equal(vol_30, signal_to_positions(signal, rolling_volatility(b.returns, 30), investable=b.investable, **stack))


def test_the_time_series_mode_trades_each_asset_on_its_own_signal_and_the_modes_and_errors_are_checked():
    b, fc = market(), forecasts()
    ctx = Context(b, Config(), forecasts=fc)
    signal = fc.mean / fc.std.where(fc.std > 0)
    own = ForecastStack(mode="time_series").build(ctx)
    pd.testing.assert_frame_equal(own, signal_to_positions(signal, rolling_volatility(b.returns, 63), investable=b.investable, winsorize_quantile=0.02, cross_sectional=False, scale="none", clip=3.0, target_vol=0.10,
                                                          max_abs_weight=0.25, gross_leverage=1.0, long_only_book=False))
    assert not own.equals(ForecastStack().build(ctx)) and (np.sign(own.iloc[-1]) == np.sign(signal.iloc[-1])).all(), "no demeaning across assets: the sign is the sign of the forecast"
    assert ForecastStack().required_assets() == 2 and ForecastStack(mode="time_series").required_assets() == 1 and ScoreStack().required_assets() == 2 and ScoreStack("time_series").required_assets() == 1
    with pytest.raises(ValueError, match="forecast_stack needs forecasts"):
        ForecastStack().build(Context(b, Config()))
    with pytest.raises(ValueError, match="mode must be 'cross_sectional' or 'time_series'"):
        ForecastStack(mode="panel").build(ctx)
    quiet = fc.std.copy()
    quiet["D"] = 0.0
    silent = ForecastStack().build(Context(b, Config(), forecasts=ForecastPanel(fc.mean, quiet, fc.confidence)))
    assert (silent["D"] == 0.0).all(), "a forecast with no standard deviation has no signal and no position"
    pd.testing.assert_frame_equal(silent, signal_to_positions(fc.mean / quiet.where(quiet > 0), rolling_volatility(b.returns, 63), investable=b.investable, **Context(b, Config()).transform()))
    assert (silent[["A", "B", "C"]].abs().sum(axis=1).iloc[-5:] > 0.5).all(), "and it takes nothing from the other three"


def test_the_score_stack_applies_the_position_stack_to_one_models_score_where_it_can_be_held():
    b = market()
    rng = np.random.default_rng(8)
    score = pd.DataFrame(rng.normal(size=b.returns.shape), index=b.index, columns=b.returns.columns)
    held = b.investable.copy()
    held.iloc[:, 3] = False
    b = market(investable=held)
    ctx = Context(b, Config(), models=[Model(score=score)])
    vol = rolling_volatility(b.returns, 63)
    stack = Context(b, Config()).transform()
    expected = signal_to_positions(score.where(b.investable), vol, investable=b.investable, **stack)
    pd.testing.assert_frame_equal(ScoreStack().build(ctx), expected)
    assert (ScoreStack().build(ctx)["D"] == 0.0).all(), "what cannot be held has no position"
    own = ScoreStack("time_series").build(ctx)
    stack.update(cross_sectional=False, scale="none")
    pd.testing.assert_frame_equal(own, signal_to_positions(score.where(b.investable), vol, investable=b.investable, **stack))
    with pytest.raises(ValueError, match="score_stack takes exactly one model"):
        ScoreStack().build(Context(b, Config(), models=[Model(score=score), Model(score=score)]))
    with pytest.raises(ValueError, match="score_stack takes exactly one model"):
        ScoreStack().build(Context(b, Config()))


# ------------------------------------------------------------------------------------------------------------------ sleeves
def test_a_sleeve_is_an_equal_slice_of_capital_times_its_own_signal_clipped_and_nothing_is_renormalised():
    b = market()
    signal = table([[1.0, -2.0, 0.5, np.nan], [np.nan] * 4, [4.0, -4.0, 0.0, 1.0]])
    padded = pd.concat([signal, pd.DataFrame(np.nan, index=b.index[3:], columns=list("ABCD"))])
    ctx = Context(b, Config(), models=[Model(score=padded)])
    out = Sleeves().build(ctx)
    assert out.iloc[0].tolist() == pytest.approx([0.25, -0.5, 0.125, 0.0]), "each signal times a quarter of the capital; a missing signal is cash"
    assert out.iloc[2].tolist() == pytest.approx([0.75, -0.75, 0.0, 0.25]), "a signal is clipped at 3 before it is sliced"
    assert out.iloc[1].isna().all() and out.iloc[3:].isna().all().all(), "a day with no signal anywhere is missing, not flat"
    twice = Sleeves(gross=2.0, clip=1.0).build(ctx)
    assert twice.iloc[0].tolist() == pytest.approx([0.5, -0.5, 0.25, 0.0]) and twice.iloc[2].tolist() == pytest.approx([0.5, -0.5, 0.0, 0.5]), "a gross of two and a clip of one"
    longs = Sleeves(long_only=True).build(ctx)
    assert longs.iloc[0].tolist() == pytest.approx([0.25, 0.0, 0.125, 0.0]) and longs.iloc[2].tolist() == pytest.approx([0.75, 0.0, 0.0, 0.25])
    held = b.investable.copy()
    held.iloc[:3, 3] = False
    fewer = Sleeves().build(Context(market(investable=held), Config(), models=[Model(score=padded)]))
    assert fewer.iloc[0].tolist() == pytest.approx([1 / 3, -2 / 3, 1 / 6, 0.0]), "the capital is split over the assets that can be held that day"


def test_sleeves_can_be_scaled_to_a_volatility_target_with_a_ceiling_and_nothing_before_there_is_history():
    b = market(n=200)
    score = pd.DataFrame(1.0, index=b.index, columns=list("ABCD"))
    ctx = Context(b, Config(**{"portfolio.volatility.lookback": 40}), models=[Model(score=score)])
    flat = Sleeves().build(ctx)
    assert (flat == 0.25).all().all(), "unscaled: a quarter in each"
    scaled = Sleeves(vol_scale=True, target_vol=0.10, max_scale=3.0).build(ctx)
    vol = rolling_volatility(b.returns, 40)
    k = 150
    expected = 0.25 * (0.10 / vol.iloc[k]).clip(0.0, 3.0)
    assert scaled.iloc[k].tolist() == pytest.approx(expected.tolist()) and (scaled.iloc[:23] == 0.0).all().all(), "a quarter times the target over the asset's own volatility; no volatility yet, no position"
    assert scaled.iloc[k]["A"] > scaled.iloc[k]["D"], "the quieter asset is levered up"
    capped = Sleeves(vol_scale=True, target_vol=5.0, max_scale=1.5).build(ctx)
    assert capped.iloc[k].tolist() == pytest.approx([0.25 * 1.5] * 4), "a ceiling on the scale"
    default = Sleeves(vol_scale=True).build(Context(b, Config(), models=[Model(score=score)]))
    vol63 = rolling_volatility(b.returns, 63)
    assert default.iloc[k].tolist() == pytest.approx((0.25 * (0.10 / vol63.iloc[k]).clip(0.0, 3.0)).tolist()), "63 days, a target of 10% and a ceiling of three, by default"


def test_a_sleeve_needs_one_model_and_positive_settings():
    ctx = Context(market(), Config(), models=[])
    with pytest.raises(ValueError, match="sleeves takes exactly one model"):
        Sleeves().build(ctx)
    for kwargs in ({"gross": 0.0}, {"gross": -1.0}, {"target_vol": 0.0}, {"target_vol": -0.1}, {"max_scale": 0.0}, {"max_scale": -1.0}, {"clip": 0.0}, {"clip": -3.0}):
        with pytest.raises(ValueError, match="gross, target_vol, max_scale and clip must be positive"):
            Sleeves(**kwargs)
    sleeves = Sleeves()
    assert (sleeves.gross, sleeves.vol_scale, sleeves.target_vol, sleeves.max_scale, sleeves.clip, sleeves.long_only) == (1.0, False, 0.10, 3.0, 3.0, False)


# ------------------------------------------------------------------------------------------------------------------ a model's own weights, and confidence
def test_a_structured_model_supplies_its_own_weights_which_are_laid_on_the_markets_dates_and_assets():
    b = market(n=6)
    own = pd.DataFrame({"B": [0.5, 0.5, np.nan], "A": [-0.5, -0.5, 0.2], "Z": [1.0, 1.0, 1.0]}, index=b.index[2:5])
    out = ModelWeights().build(Context(b, Config(), models=[Model(weights=own)]))
    assert list(out.columns) == list("ABCD") and out.index.equals(b.index)
    assert out.loc[b.index[2], "A"] == -0.5 and out.loc[b.index[2], "B"] == 0.5 and out.loc[b.index[4], "B"] == 0.0 and out.loc[b.index[4], "A"] == 0.2, "a missing weight is flat"
    assert (out.loc[b.index[[0, 1, 5]]] == 0.0).all().all() and "Z" not in out and not out.isna().any().any()
    with pytest.raises(ValueError, match="model_weights takes exactly one model"):
        ModelWeights().build(Context(b, Config(), models=[]))
    with pytest.raises(ValueError, match="stub does not define target weights"):
        ModelWeights().build(Context(b, Config(), models=[Model()]))


def test_confidence_sizing_takes_the_sign_of_the_forecast_times_its_confidence_above_a_floor_then_a_gross_and_a_cap():
    b = market(n=3)
    mean = table([[0.02, -0.01, 0.03, 0.0], [0.02, -0.01, 0.03, 0.01], [-0.02, 0.01, 0.03, 0.01]])
    confidence = table([[0.8, 0.4, 0.2, 0.9], [0.8, 0.4, 0.2, 0.9], [0.5, 0.5, 0.5, 0.5]])
    fc = ForecastPanel(mean, mean * 0 + 0.02, confidence)
    out = ConfidenceSized(min_confidence=0.3, max_weight=0.7).build(Context(b, Config(), forecasts=fc))
    assert out.iloc[0].tolist() == pytest.approx([0.8 / 1.2, -0.4 / 1.2, 0.0, 0.0]), "the third is under the floor and the fourth has no sign: 0.8 and -0.4, scaled to a gross of one"
    assert out.iloc[1].tolist() == pytest.approx([0.8 / 2.1, -0.4 / 2.1, 0.0, 0.9 / 2.1]) and out.iloc[2].tolist() == pytest.approx([-0.25, 0.25, 0.25, 0.25])
    default = ConfidenceSized().build(Context(b, Config(), forecasts=fc))
    assert default.iloc[0].tolist() == pytest.approx([0.25, -0.25, 0.25, 0.0]), "no floor, a quarter at most: 0.8, -0.4 and 0.2 are cut to a quarter and what is cut goes to the third; the fourth has no sign, so the book stays short of a whole"
    assert default.iloc[1].tolist() == pytest.approx([0.25, -0.25, 0.25, 0.25]) and default.iloc[2].tolist() == pytest.approx([-0.25, 0.25, 0.25, 0.25])
    longs = ConfidenceSized(long_only=True, max_weight=0.7).build(Context(b, Config(), forecasts=fc))
    assert longs.iloc[1].tolist() == pytest.approx([0.8 / 1.9, 0.0, 0.2 / 1.9, 0.9 / 1.9]) and (longs >= 0.0).all().all(), "a short forecast gets nothing in a book that cannot short"
    twice = ConfidenceSized(gross=2.0, max_weight=0.9).build(Context(b, Config(), forecasts=fc))
    assert twice.iloc[2].abs().sum() == pytest.approx(2.0)
    held = b.investable.copy()
    held.iloc[:, 0] = False
    masked = ConfidenceSized(max_weight=0.9).build(Context(market(n=3, investable=held), Config(), forecasts=fc))
    assert (masked["A"] == 0.0).all() and masked.iloc[1].abs().sum() == pytest.approx(1.0), "what cannot be held has no position and the book is scaled over the rest"
    at_floor = ConfidenceSized(min_confidence=0.5, max_weight=0.9).build(Context(b, Config(), forecasts=fc))
    assert at_floor.iloc[2].tolist() == pytest.approx([-0.25, 0.25, 0.25, 0.25]), "a confidence exactly at the floor is enough"
    short_mask = b.investable.drop(columns="D")
    unlisted = ConfidenceSized(max_weight=0.9).build(Context(market(n=3, investable=short_mask), Config(), forecasts=fc))
    assert (unlisted["D"] == 0.0).all() and unlisted.iloc[2].abs().sum() == pytest.approx(1.0), "an asset the investable table does not list cannot be held"
    cc = ConfidenceSized()
    assert (cc.min_confidence, cc.gross, cc.max_weight, cc.long_only) == (0.0, 1.0, 0.25, False)


# ------------------------------------------------------------------------------------------------------------------ tilts and the blend by regime
def test_a_tilt_adds_to_the_base_book_and_the_sum_is_brought_back_to_the_base_books_own_gross():
    base = table([[0.4, 0.3, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0], [0.25, 0.25, -0.25, -0.25]])
    out = _tilted(base, {"A": 0.1, "Z": 0.5})
    assert out.iloc[0].tolist() == pytest.approx([0.5 / 1.1, 0.3 / 1.1, 0.2 / 1.1, 0.1 / 1.1]), "A gets a tenth more, then the book is scaled back to a gross of one"
    assert out.iloc[1].tolist() == [0.0, 0.0, 0.0, 0.0], "an empty book stays empty: a tilt cannot make a book out of nothing"
    assert out.iloc[2].tolist() == pytest.approx([0.35 / 1.1, 0.25 / 1.1, -0.25 / 1.1, -0.25 / 1.1]), "gross is the sum of the absolute weights, and the base book's own is one"
    assert base.iloc[0].tolist() == [0.4, 0.3, 0.2, 0.1], "the base book is not changed"
    assert _tilted(base, {}).equals(base) and (_tilted(base, {"Z": 1.0}).to_numpy() == base.to_numpy()).all()


def regimes_for(index, calm, stress) -> RegimeSeries:
    return RegimeSeries(pd.DataFrame({"calm": calm, "stress": stress}, index=index))


class Fixed(Allocator):
    """An allocator that holds one fixed book, to see how a blend weighs them."""

    def __init__(self, weights=(1.0, 0.0, 0.0, 0.0)):
        self.weights = list(weights)

    def build(self, ctx):
        return pd.DataFrame([self.weights] * len(ctx.bundle.index), index=ctx.bundle.index, columns=ctx.bundle.assets)


def test_a_blend_weighs_the_books_of_the_regimes_by_their_probability_and_falls_back_before_the_detector_has_a_view():
    register_allocator("test_fixed")(Fixed)
    try:
        b = market(n=5)
        calm = [0.7, 0.2, 0.5, np.nan, np.nan]
        stress = [0.3, 0.8, 0.5, np.nan, 0.4]
        ctx = Context(b, Config(), regimes=regimes_for(b.index, calm, stress))
        rules = {"calm": {"allocator": "test_fixed", "params": {"weights": [1.0, 0.0, 0.0, 0.0]}}, "stress": {"allocator": "test_fixed", "params": {"weights": [0.0, 0.0, 0.0, 1.0]}}}
        default = {"allocator": "test_fixed", "params": {"weights": [0.0, 0.5, 0.5, 0.0]}}
        soft = RegimeSwitch(rules, default).build(ctx)
        assert soft.iloc[0].tolist() == pytest.approx([0.7, 0.0, 0.0, 0.3]) and soft.iloc[1].tolist() == pytest.approx([0.2, 0.0, 0.0, 0.8]) and soft.iloc[2].tolist() == pytest.approx([0.5, 0.0, 0.0, 0.5])
        assert soft.iloc[3].tolist() == pytest.approx([0.0, 0.5, 0.5, 0.0]), "no view at all: the default book"
        assert soft.iloc[4].tolist() == pytest.approx([0.0, 0.3, 0.3, 0.4]), "a view that covers 0.4 of the probability: that share to its book, the rest to the default"
        hard = RegimeSwitch(rules, default, mode="hard").build(ctx)
        assert hard.iloc[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0]) and hard.iloc[1].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0]), "the most probable regime alone"
        assert hard.iloc[3].tolist() == pytest.approx([0.0, 0.5, 0.5, 0.0])
        with pytest.raises(ValueError, match="mode must be 'soft' or 'hard'"):
            RegimeSwitch(rules, default, mode="blend").build(ctx)
        with pytest.raises(ValueError, match="regime_switch needs regimes"):
            RegimeSwitch(rules, default).build(Context(b, Config()))
        fallback = RegimeSwitch({"calm": rules["calm"]}, default).build(ctx)
        assert fallback.iloc[0].tolist() == pytest.approx([0.7, 0.15, 0.15, 0.0]), "a regime with no rule uses the default"
        assert RegimeSwitch({}).default == {"book": "risk_parity"} and RegimeSwitch({}).mode == "soft"
        tilted = RegimeSwitch({"calm": {**rules["calm"], "tilt": {"B": 0.5}}, "stress": rules["stress"]}, default).build(ctx)
        assert tilted.iloc[0].tolist() == pytest.approx([0.7 * (1.0 / 1.5), 0.7 * (0.5 / 1.5), 0.0, 0.3]), "a tilt reshapes that regime's book before it is weighed"
    finally:
        ALLOCATORS.unregister("test_fixed")
