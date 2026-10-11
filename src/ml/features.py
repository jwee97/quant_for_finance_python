"""The feature library: named, documented, causal feature functions in ten groups.

Every feature is a function of the bundle's data up to and including the date it is stamped with, so a feature row can be computed at the close of ``t`` and used to decide a trade at ``t``. That is not a
promise made in a docstring: ``tests/test_ml_data.py`` replaces every price, volume and macro value after a cutoff with noise and requires every feature up to the cutoff to be unchanged
(``MarketBundle.perturbed_after``, the same test the platform applies to its strategies).

A feature function takes ``Inputs`` and returns a wide frame (date x asset). ``broadcast`` features describe the market, not the asset (the same value for every asset on a date): they are not normalised
across assets, because across assets they are constant, and are standardised inside each training fold instead.

Groups. ``price`` trailing returns and position against moving averages and extremes; ``momentum`` the classic 12-1 / 6-1 / 3-1, RSI, MACD, breakouts, trend strength; ``mean_reversion`` short-term
reversal, z-scores, Bollinger position, autocorrelation, internal bar strength; ``volatility`` realised, downside, range-based, EWMA, skew and kurtosis, beta and idiosyncratic volatility, drawdown;
``volume`` dollar volume, volume surge, Amihud illiquidity (needs volume); ``relative`` strength against the universe and the asset class; ``macro`` levels and changes of the macro series in the bundle;
``futures`` carry, curve slope and basis momentum (needs ``CARRY_<asset>`` in the bundle's macro panel); ``options`` the implied-volatility index of the underlying and its premium over realised volatility
(needs VIX-family series); ``crypto`` funding, perpetual basis and stablecoin supply growth (needs the crypto bundle). A group whose inputs are absent is skipped and the skip is reported, never faked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

ANN = 252.0
GROUPS = ("price", "momentum", "mean_reversion", "volatility", "volume", "relative", "macro", "futures", "options", "crypto")
# The implied-volatility index of the underlying, where the platform has one (the same mapping the options workspace uses).
IV_INDEX = {"SPY": "VIX", "QQQ": "VXN", "GLD": "GVZ", "DBC": "OVX"}


@dataclass
class Inputs:
    """What a feature function may look at."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    volume: pd.DataFrame | None
    high: pd.DataFrame | None
    low: pd.DataFrame | None
    raw_close: pd.DataFrame | None
    macro: pd.DataFrame
    asset_class: dict
    group: dict
    market: pd.Series                    # equal-weight return of the universe, known at each close (the cross-sectional mean of that day's returns)

    @property
    def assets(self) -> list[str]:
        return list(self.prices.columns)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    description: str
    fn: Callable[[Inputs], pd.DataFrame]
    lookback: int = 1                    # trading days of history the value needs
    needs: tuple = ()                    # "volume", "ohlc", "raw_close", "macro", "carry", "iv", "crypto"
    broadcast: bool = False

    def info(self) -> dict:
        return {"name": self.name, "group": self.group, "description": self.description, "lookback": self.lookback, "needs": list(self.needs), "broadcast": self.broadcast}


FEATURES: dict[str, FeatureSpec] = {}


def feature(name: str, group: str, description: str, lookback: int = 1, needs: tuple = (), broadcast: bool = False):
    """Register a feature function under ``name``."""
    if group not in GROUPS:
        raise ValueError(f"unknown feature group '{group}'")

    def wrap(fn):
        if name in FEATURES:
            raise ValueError(f"feature '{name}' is already registered")
        FEATURES[name] = FeatureSpec(name, group, description, fn, lookback, needs, broadcast)
        return fn
    return wrap


# ------------------------------------------------------------------------------------------------------------------ helpers
def rolling_slope(y: pd.DataFrame, window: int) -> pd.DataFrame:
    """OLS slope of each column against time over a trailing ``window`` (NaN if any value in the window is missing), by convolution with fixed weights."""
    t = np.arange(window, dtype=float)
    w = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    vals = y.to_numpy(dtype=float)
    out = np.full_like(vals, np.nan)
    for j in range(vals.shape[1]):
        col = vals[:, j]
        bad = np.isnan(col)
        conv = np.convolve(np.where(bad, 0.0, col), w[::-1], mode="valid")
        n_bad = pd.Series(bad.astype(float)).rolling(window).sum().to_numpy()[window - 1:]
        res = np.where(n_bad == 0, conv, np.nan)
        out[window - 1:, j] = res
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def _vol(r: pd.DataFrame, window: int) -> pd.DataFrame:
    return r.rolling(window, min_periods=max(10, window // 2)).std(ddof=1) * np.sqrt(ANN)


def _z(x: pd.DataFrame, window: int) -> pd.DataFrame:
    m = x.rolling(window, min_periods=max(20, window // 3))
    return (x - m.mean()) / m.std(ddof=1).replace(0.0, np.nan)


def _broadcast(series: pd.Series, assets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(np.repeat(series.to_numpy(dtype=float)[:, None], len(assets), axis=1), index=series.index, columns=assets)


# ------------------------------------------------------------------------------------------------------------------ price
for _h in (1, 5, 21, 63, 126, 252):
    feature(f"ret_{_h}", "price", f"{_h}-day trailing return", lookback=_h + 1)(lambda x, h=_h: x.prices.pct_change(h, fill_method=None))
for _k in (2, 3, 4, 5):
    feature(f"ret_lag_{_k}", "price", f"the daily return {_k - 1} day(s) before the latest one (the autoregression's regressors)", lookback=_k + 1)(lambda x, k=_k: x.returns.shift(k - 1))
for _w in (20, 50, 200):
    feature(f"ma_ratio_{_w}", "price", f"price over its {_w}-day average, minus 1", lookback=_w)(lambda x, w=_w: x.prices / x.prices.rolling(w, min_periods=int(w * 0.8)).mean() - 1.0)
feature("dist_high_252", "price", "price over its 252-day high, minus 1 (0 at a new high)", lookback=252)(lambda x: x.prices / x.prices.rolling(252, min_periods=126).max() - 1.0)
feature("dist_low_252", "price", "price over its 252-day low, minus 1 (0 at a new low)", lookback=252)(lambda x: x.prices / x.prices.rolling(252, min_periods=126).min() - 1.0)


# ------------------------------------------------------------------------------------------------------------------ momentum
feature("mom_12_1", "momentum", "return from 12 months ago to 1 month ago (skips the last month)", lookback=253)(lambda x: x.prices.shift(21) / x.prices.shift(252) - 1.0)
feature("mom_6_1", "momentum", "return from 6 months ago to 1 month ago", lookback=127)(lambda x: x.prices.shift(21) / x.prices.shift(126) - 1.0)
feature("mom_3_1", "momentum", "return from 3 months ago to 1 month ago", lookback=64)(lambda x: x.prices.shift(21) / x.prices.shift(63) - 1.0)


@feature("rsi_14", "momentum", "Wilder's relative strength index over 14 days, centred at 0 (range -0.5 to 0.5)", lookback=60)
def _rsi(x: Inputs) -> pd.DataFrame:
    d = x.prices.diff()
    up, dn = d.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean(), (-d.clip(upper=0.0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return up / (up + dn).replace(0.0, np.nan) - 0.5


@feature("macd_hist", "momentum", "MACD histogram (12/26/9 exponential averages) as a fraction of price", lookback=80)
def _macd(x: Inputs) -> pd.DataFrame:
    p = x.prices
    macd = p.ewm(span=12, adjust=False, min_periods=26).mean() - p.ewm(span=26, adjust=False, min_periods=26).mean()
    return (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / p


@feature("breakout_20", "momentum", "distance above the prior 20-day high in units of 20-day price volatility (negative inside the range)", lookback=22)
def _breakout(x: Inputs) -> pd.DataFrame:
    p = x.prices
    return (p - p.rolling(20).max().shift(1)) / p.rolling(20).std(ddof=1).replace(0.0, np.nan)


feature("trend_strength_63", "momentum", "slope of log price over 63 days, in daily-volatility units per day, scaled to a t-like number", lookback=64)(
    lambda x: rolling_slope(np.log(x.prices.where(x.prices > 0)), 63) * 63 / (x.returns.rolling(63, min_periods=40).std(ddof=1).replace(0.0, np.nan) * np.sqrt(63.0)))


# ------------------------------------------------------------------------------------------------------------------ mean reversion
feature("rev_1", "mean_reversion", "minus yesterday's return (short-term reversal)", lookback=2)(lambda x: -x.returns)
feature("rev_5", "mean_reversion", "minus the 5-day return", lookback=6)(lambda x: -x.prices.pct_change(5, fill_method=None))
feature("zscore_20", "mean_reversion", "price against its 20-day mean in 20-day standard deviations", lookback=20)(
    lambda x: (x.prices - x.prices.rolling(20).mean()) / x.prices.rolling(20).std(ddof=1).replace(0.0, np.nan))
feature("zscore_60", "mean_reversion", "price against its 60-day mean in 60-day standard deviations", lookback=60)(
    lambda x: (x.prices - x.prices.rolling(60).mean()) / x.prices.rolling(60).std(ddof=1).replace(0.0, np.nan))
feature("bollinger_pos_20", "mean_reversion", "position inside the 20-day, two-sigma Bollinger band (0 at the lower band, 1 at the upper)", lookback=20)(
    lambda x: (x.prices - (x.prices.rolling(20).mean() - 2 * x.prices.rolling(20).std(ddof=1))) / (4 * x.prices.rolling(20).std(ddof=1)).replace(0.0, np.nan))
feature("autocorr_63", "mean_reversion", "first-order autocorrelation of daily returns over 63 days", lookback=64)(lambda x: x.returns.rolling(63, min_periods=40).corr(x.returns.shift(1)))


@feature("ibs", "mean_reversion", "internal bar strength: where the close sits in the day's range (0 at the low, 1 at the high); needs unadjusted OHLC", lookback=1, needs=("ohlc", "raw_close"))
def _ibs(x: Inputs) -> pd.DataFrame:
    rng = (x.high - x.low).replace(0.0, np.nan)
    return ((x.raw_close - x.low) / rng).clip(0.0, 1.0)


# ------------------------------------------------------------------------------------------------------------------ volatility
for _w in (21, 63, 252):
    feature(f"vol_{_w}", "volatility", f"annualised standard deviation of daily returns over {_w} days", lookback=_w)(lambda x, w=_w: _vol(x.returns, w))
feature("vol_ratio_21_63", "volatility", "21-day volatility over 63-day volatility (rising when above 1)", lookback=63)(lambda x: _vol(x.returns, 21) / _vol(x.returns, 63).replace(0.0, np.nan))
feature("vol_of_vol_63", "volatility", "standard deviation over 63 days of the 21-day volatility, relative to its mean", lookback=84)(
    lambda x: _vol(x.returns, 21).rolling(63, min_periods=40).std(ddof=1) / _vol(x.returns, 21).rolling(63, min_periods=40).mean().replace(0.0, np.nan))
feature("downside_vol_63", "volatility", "annualised root-mean-square of negative returns over 63 days", lookback=63)(
    lambda x: np.sqrt((x.returns.clip(upper=0.0) ** 2).rolling(63, min_periods=40).mean()) * np.sqrt(ANN))
feature("skew_63", "volatility", "skewness of daily returns over 63 days", lookback=63)(lambda x: x.returns.rolling(63, min_periods=40).skew())
feature("kurt_63", "volatility", "excess kurtosis of daily returns over 63 days", lookback=63)(lambda x: x.returns.rolling(63, min_periods=40).kurt())
feature("drawdown_63", "volatility", "price over its 63-day high, minus 1", lookback=63)(lambda x: x.prices / x.prices.rolling(63, min_periods=40).max() - 1.0)
feature("ewma_vol_30", "volatility", "annualised exponentially weighted volatility, half-life 30 days", lookback=60)(lambda x: np.sqrt((x.returns ** 2).ewm(halflife=30, min_periods=30).mean()) * np.sqrt(ANN))


@feature("parkinson_21", "volatility", "annualised range-based (Parkinson) volatility over 21 days from daily high and low", lookback=21, needs=("ohlc",))
def _parkinson(x: Inputs) -> pd.DataFrame:
    hl = np.log(x.high / x.low) ** 2
    return np.sqrt(hl.rolling(21, min_periods=15).mean() / (4.0 * np.log(2.0)) * ANN)


@feature("beta_63", "volatility", "beta to the equal-weight market over 63 days", lookback=63)
def _beta(x: Inputs) -> pd.DataFrame:
    m = x.market
    cov = x.returns.rolling(63, min_periods=40).cov(m)
    return cov.div(m.rolling(63, min_periods=40).var(ddof=1).replace(0.0, np.nan), axis=0)


@feature("idio_vol_63", "volatility", "annualised volatility of returns left after removing the 63-day beta to the market", lookback=63)
def _idio(x: Inputs) -> pd.DataFrame:
    beta = _beta(x)
    resid = x.returns.sub(beta.shift(1).mul(x.market, axis=0))                    # beta known at t-1 applied to the market return of t: the residual is a clean out-of-sample measure
    return resid.rolling(63, min_periods=40).std(ddof=1) * np.sqrt(ANN)


# ------------------------------------------------------------------------------------------------------------------ volume
feature("log_dollar_vol_21", "volume", "log of the 21-day average dollar volume", lookback=21, needs=("volume",))(
    lambda x: np.log((x.prices * x.volume).rolling(21, min_periods=15).mean().clip(lower=1.0)))
feature("vol_surge_21", "volume", "volume over its 21-day average, minus 1", lookback=21, needs=("volume",))(
    lambda x: x.volume / x.volume.rolling(21, min_periods=15).mean().replace(0.0, np.nan) - 1.0)
feature("volume_trend_63", "volume", "log of the 21-day over the 63-day average volume", lookback=63, needs=("volume",))(
    lambda x: np.log(x.volume.rolling(21, min_periods=15).mean().clip(lower=1.0) / x.volume.rolling(63, min_periods=40).mean().clip(lower=1.0)))
feature("amihud_21", "volume", "Amihud illiquidity: 21-day average of |return| over dollar volume, in millionths", lookback=21, needs=("volume",))(
    lambda x: (x.returns.abs() / (x.prices * x.volume).replace(0.0, np.nan)).rolling(21, min_periods=15).mean() * 1e6)


@feature("up_down_volume_21", "volume", "share of the last 21 days' volume traded on up days minus the share on down days", lookback=21, needs=("volume",))
def _updown(x: Inputs) -> pd.DataFrame:
    sign = np.sign(x.returns)
    num = (x.volume * sign).rolling(21, min_periods=15).sum()
    return num / x.volume.rolling(21, min_periods=15).sum().replace(0.0, np.nan)


# ------------------------------------------------------------------------------------------------------------------ relative
def _xs_demean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0)


feature("rel_ret_21", "relative", "21-day return minus the universe average", lookback=22)(lambda x: _xs_demean(x.prices.pct_change(21, fill_method=None)))
feature("rel_ret_63", "relative", "63-day return minus the universe average", lookback=64)(lambda x: _xs_demean(x.prices.pct_change(63, fill_method=None)))
feature("rank_ret_63", "relative", "cross-sectional percentile rank of the 63-day return", lookback=64)(lambda x: x.prices.pct_change(63, fill_method=None).rank(axis=1, pct=True))
feature("rel_vol_21", "relative", "21-day volatility over the universe median", lookback=21)(lambda x: _vol(x.returns, 21).div(_vol(x.returns, 21).median(axis=1), axis=0))
feature("corr_market_63", "relative", "63-day correlation with the equal-weight market", lookback=63)(lambda x: x.returns.rolling(63, min_periods=40).corr(x.market))


@feature("class_rel_ret_21", "relative", "21-day return minus the average of the asset's own class", lookback=22)
def _class_rel(x: Inputs) -> pd.DataFrame:
    r21 = x.prices.pct_change(21, fill_method=None)
    out = pd.DataFrame(np.nan, index=r21.index, columns=r21.columns)
    classes: dict[str, list[str]] = {}
    for a in r21.columns:
        classes.setdefault(x.asset_class.get(a, "unknown"), []).append(a)
    for members in classes.values():
        out[members] = r21[members].sub(r21[members].mean(axis=1), axis=0) if len(members) > 1 else 0.0
    return out


# ------------------------------------------------------------------------------------------------------------------ macro
MACRO_LEVELS = ("DGS10", "T10Y2Y", "T10Y3M", "BAA10Y", "VIX", "MOVE", "DFF", "DTWEXBGS", "T10YIE")


def _macro_present(x: Inputs, names: tuple) -> list[str]:
    return [c for c in names if c in x.macro.columns and x.macro[c].notna().sum() > 300]


def _macro_feature(col: str, kind: str):
    def fn(x: Inputs) -> pd.DataFrame:
        s = x.macro[col].astype(float).reindex(x.prices.index)
        series = _z(s.to_frame(col), 252)[col] if kind == "z" else (s - s.shift(21)) / s.rolling(252, min_periods=120).std(ddof=1).replace(0.0, np.nan)
        return _broadcast(series, x.assets)
    return fn


for _c in MACRO_LEVELS:
    feature(f"macro_{_c}_z", "macro", f"{_c}: level in standard deviations of its trailing year", lookback=252, needs=("macro",), broadcast=True)(_macro_feature(_c, "z"))
    feature(f"macro_{_c}_chg21", "macro", f"{_c}: 21-day change in units of its trailing-year standard deviation", lookback=252, needs=("macro",), broadcast=True)(_macro_feature(_c, "chg"))


# ------------------------------------------------------------------------------------------------------------------ futures
def _per_asset_macro(x: Inputs, prefix: str) -> pd.DataFrame:
    cols = {a: x.macro[f"{prefix}_{a}"] for a in x.assets if f"{prefix}_{a}" in x.macro.columns}
    return pd.DataFrame(cols).reindex(index=x.prices.index, columns=x.assets)


feature("carry", "futures", "annualised roll yield of the front two contracts (positive in backwardation)", lookback=1, needs=("carry",))(lambda x: _per_asset_macro(x, "CARRY"))
feature("carry_smooth_21", "futures", "21-day average of the roll yield", lookback=21, needs=("carry",))(lambda x: _per_asset_macro(x, "CARRY").rolling(21, min_periods=10).mean())
feature("carry_to_vol", "futures", "roll yield over 63-day volatility", lookback=63, needs=("carry",))(lambda x: _per_asset_macro(x, "CARRY") / _vol(x.returns, 63).replace(0.0, np.nan))
feature("curve_slope", "futures", "log price of the third contract over the first (positive in contango)", lookback=1, needs=("carry",))(lambda x: _per_asset_macro(x, "SLOPE"))
feature("basis_momentum", "futures", "Boons-Prado basis momentum: one-year mean return of the front contract minus the second", lookback=252, needs=("carry",))(
    lambda x: _per_asset_macro(x, "BASISMOM"))


# ------------------------------------------------------------------------------------------------------------------ options (implied-volatility indices)
def _iv_frame(x: Inputs) -> pd.DataFrame:
    cols = {a: x.macro[IV_INDEX[a]].astype(float) / 100.0 for a in x.assets if a in IV_INDEX and IV_INDEX[a] in x.macro.columns}
    return pd.DataFrame(cols).reindex(index=x.prices.index, columns=x.assets)


feature("iv_level", "options", "the underlying's implied-volatility index (VIX, VXN, GVZ, OVX) as a decimal; empty for assets without one", lookback=1, needs=("iv",))(_iv_frame)
feature("iv_rank_252", "options", "where the implied-volatility index sits between its one-year low and high", lookback=252, needs=("iv",))(
    lambda x: (_iv_frame(x) - _iv_frame(x).rolling(252, min_periods=126).min()) / (_iv_frame(x).rolling(252, min_periods=126).max() - _iv_frame(x).rolling(252, min_periods=126).min()).replace(0.0, np.nan))
feature("iv_minus_rv_21", "options", "implied-volatility index minus 21-day realised volatility (the variance risk premium, in volatility points)", lookback=21, needs=("iv",))(
    lambda x: _iv_frame(x) - _vol(x.returns, 21))
feature("iv_chg_5", "options", "5-day change in the implied-volatility index", lookback=6, needs=("iv",))(lambda x: _iv_frame(x) - _iv_frame(x).shift(5))


# ------------------------------------------------------------------------------------------------------------------ crypto
def _crypto_base(asset: str) -> str:
    return asset.split("_")[0]


def _crypto_frame(x: Inputs, prefix: str) -> pd.DataFrame:
    cols = {a: x.macro[f"{prefix}_{_crypto_base(a)}"] for a in x.assets if f"{prefix}_{_crypto_base(a)}" in x.macro.columns}
    return pd.DataFrame(cols).reindex(index=x.prices.index, columns=x.assets)


feature("funding_ann", "crypto", "trailing 7-day perpetual funding earned by a short, annualised", lookback=7, needs=("crypto",))(lambda x: _crypto_frame(x, "FUNDING"))
feature("funding_z_90", "crypto", "funding in standard deviations of its trailing 90 days", lookback=90, needs=("crypto",))(lambda x: _z(_crypto_frame(x, "FUNDING"), 90))
feature("perp_basis", "crypto", "perpetual price over spot, minus 1", lookback=1, needs=("crypto",))(lambda x: _crypto_frame(x, "BASIS"))
feature("perp_basis_z_90", "crypto", "perpetual basis in standard deviations of its trailing 90 days", lookback=90, needs=("crypto",))(lambda x: _z(_crypto_frame(x, "BASIS"), 90))


@feature("stable_growth_30", "crypto", "30-day log growth of the total stablecoin supply (market-wide)", lookback=31, needs=("crypto",), broadcast=True)
def _stable(x: Inputs) -> pd.DataFrame:
    return _broadcast(np.log(x.macro["STABLE_SUPPLY"].astype(float)).diff(30).reindex(x.prices.index), x.assets)


# ------------------------------------------------------------------------------------------------------------------ computing a set of features
def inputs_from_bundle(bundle) -> Inputs:
    """The data a feature may use, taken from a ``MarketBundle``; the unadjusted close is taken from the underlying market data when there is any (needed only for the bar-position feature)."""
    returns = bundle.returns.reindex(index=bundle.prices.index, columns=bundle.prices.columns)
    market = returns.mean(axis=1, skipna=True)
    mkt = getattr(bundle, "market", None)
    raw_close = getattr(mkt, "close", None) if mkt is not None else None
    return Inputs(prices=bundle.prices.astype(float), returns=returns.astype(float), volume=None if bundle.volume is None else bundle.volume.reindex_like(bundle.prices).astype(float),
                  high=None if bundle.high is None else bundle.high.reindex_like(bundle.prices).astype(float), low=None if bundle.low is None else bundle.low.reindex_like(bundle.prices).astype(float),
                  raw_close=None if raw_close is None else raw_close.reindex_like(bundle.prices).astype(float), macro=bundle.macro.reindex(bundle.prices.index) if len(bundle.macro) else pd.DataFrame(index=bundle.prices.index),
                  asset_class=dict(bundle.asset_class or {}), group=dict(bundle.group or {}), market=market.astype(float))


def available(spec: FeatureSpec, x: Inputs) -> str | None:
    """``None`` if the feature can be computed from ``x``, else the reason it cannot."""
    for need in spec.needs:
        if need == "volume" and (x.volume is None or x.volume.notna().sum().sum() < 100):
            return "the data has no volume"
        if need == "ohlc" and (x.high is None or x.low is None):
            return "the data has no high and low"
        if need == "raw_close" and x.raw_close is None:
            return "the data has no unadjusted close"
        if need == "macro" and not _macro_present(x, (spec.name.split("_")[1],)):
            return f"the macro series {spec.name.split('_')[1]} is not in the data"
        if need == "carry" and not any(c.startswith("CARRY_") for c in x.macro.columns):
            return "the data has no CARRY_<asset> series (a futures bundle has)"
        if need == "iv" and not any(IV_INDEX.get(a) in x.macro.columns for a in x.assets):
            return "none of the assets has an implied-volatility index (SPY: VIX, QQQ: VXN, GLD: GVZ, DBC: OVX)"
        if need == "crypto" and not any(c.startswith(("FUNDING_", "BASIS_", "STABLE_")) for c in x.macro.columns):
            return "the data has no funding, basis or stablecoin series (the crypto bundle has)"
    return None


@dataclass
class FeatureSet:
    """Computed features: ``frames`` (name -> wide frame), what was skipped and why, and the features that are market-wide."""

    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    broadcast: set = field(default_factory=set)

    @property
    def names(self) -> list[str]:
        return list(self.frames)


def compute_features(x: Inputs, groups: list[str] | tuple, names: list[str] | None = None) -> FeatureSet:
    """Compute the features of ``groups`` (or exactly ``names``) that the data supports. Unsupported ones are skipped with a reason in ``FeatureSet.skipped``."""
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        raise ValueError(f"unknown feature groups {bad}; choose from {list(GROUPS)}")
    chosen = [FEATURES[n] for n in names] if names else [f for f in FEATURES.values() if f.group in groups]
    out = FeatureSet()
    for spec in chosen:
        reason = available(spec, x)
        if reason is not None:
            out.skipped[spec.name] = reason
            continue
        frame = spec.fn(x).reindex(index=x.prices.index, columns=x.assets).replace([np.inf, -np.inf], np.nan)
        if frame.notna().sum().sum() == 0:
            out.skipped[spec.name] = "no valid values (the history is shorter than the feature's lookback)"
            continue
        out.frames[spec.name] = frame.astype(np.float32)
        if spec.broadcast:
            out.broadcast.add(spec.name)
    return out


def list_features() -> list[dict]:
    """Every registered feature with its group, description, lookback and requirements: what the dataset builder's picker shows."""
    return [f.info() for f in FEATURES.values()]


def group_summary() -> list[dict]:
    return [{"group": g, "features": [f.name for f in FEATURES.values() if f.group == g]} for g in GROUPS]
