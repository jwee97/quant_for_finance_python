"""Trend following and momentum, the best-documented premia in liquid markets.

Time-series momentum (Moskowitz, Ooi and Pedersen 2012; Hurst, Ooi and Pedersen 2017), the exponentially weighted crossover ensemble of Baz et al. (2015),
Carver-style breakouts, Kaufman's adaptive average, Wilder's directional movement, SuperTrend, Keltner and Ichimoku channels, MACD; and on the cross-sectional side
the 52-week-high effect (George and Hwang 2004), residual momentum (Blitz, Huij and Martens 2011), "frog in the pan" smooth momentum (Da, Gurun and Warachka 2014)
and Keller's 13612W momentum. Every score on day ``t`` uses data through ``t`` only; indicators that need a high and a low fall back to the close when the bundle has none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.regimes import market_proxy
from ..framework.registry import register_model
from ._common import atr_pct, daily_vol, high_low, stateful, wilder


def _valid(score: pd.DataFrame, data) -> pd.DataFrame:
    return score.where(data.investable)


@register_model("tsmom", "time-series", "Time-series momentum (Moskowitz-Ooi-Pedersen): the sign of the 12-month return, sized by 40% over each asset's own volatility")
class TimeSeriesMomentum(ForecastModel):
    """An asset that rose over the past year tends to keep rising, and one that fell tends to keep falling (under-reaction, hedging demand, herding)."""

    name, family, position_mode = "tsmom", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, lookback: int = 252, target_vol: float = 0.40, max_scale: float = 4.0, halflife: float = 60.0):
        if lookback < 5 or target_vol <= 0 or max_scale <= 0:
            raise ValueError("lookback >= 5, target_vol > 0, max_scale > 0")
        self.lookback, self.target_vol, self.max_scale, self.halflife = lookback, target_vol, max_scale, halflife

    def score(self, data):
        ret = data.prices / data.prices.shift(self.lookback) - 1.0
        vol = daily_vol(data.returns, self.halflife) * np.sqrt(252.0)
        scale = (self.target_vol / vol).clip(upper=self.max_scale)
        return _valid(np.sign(ret) * scale, data).where(ret.notna() & vol.notna())


@register_model("tsmom_multi", "time-series", "Multi-horizon time-series momentum: the average of the signs of the 1-, 3- and 12-month returns (a century of evidence)")
class MultiHorizonTSMOM(ForecastModel):
    """Averaging trend signals over several horizons keeps most of the premium and halves the turnover of any single lookback."""

    name, family, position_mode = "tsmom_multi", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, horizons: tuple = (21, 63, 252)):
        if not horizons or min(horizons) < 2:
            raise ValueError("horizons must be a non-empty tuple of windows of at least 2 days")
        self.horizons = tuple(int(h) for h in horizons)

    def score(self, data):
        signs = [np.sign(data.prices / data.prices.shift(h) - 1.0) for h in self.horizons]
        stack = np.stack([s.to_numpy() for s in signs])
        score = np.where(np.isfinite(stack).all(axis=0), stack.mean(axis=0), np.nan)
        return _valid(pd.DataFrame(score, index=data.index, columns=data.prices.columns), data)


@register_model("ma_ensemble", "time-series", "Exponential moving-average crossover ensemble (Baz et al.): three speeds, volatility-normalised, passed through a response that fades extreme trends")
class MovingAverageEnsemble(ForecastModel):
    """Crossovers at several speeds, normalised by price volatility and the typical size of the signal, capture trends of different lengths without choosing one."""

    name, family, position_mode = "ma_ensemble", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, pairs: tuple = ((8, 24), (16, 48), (32, 96)), price_window: int = 63, signal_window: int = 252):
        if not pairs or any(f >= s or f < 1 for f, s in pairs):
            raise ValueError("each pair is (fast half-life, slow half-life) with 1 <= fast < slow")
        self.pairs, self.price_window, self.signal_window = tuple((int(f), int(s)) for f, s in pairs), price_window, signal_window

    def score(self, data):
        close = data.prices
        std_price = close.rolling(self.price_window, min_periods=self.price_window).std()
        out = []
        for fast, slow in self.pairs:
            x = (close.ewm(halflife=fast, adjust=False, min_periods=slow).mean() - close.ewm(halflife=slow, adjust=False, min_periods=slow).mean()) / std_price
            y = x / x.rolling(self.signal_window, min_periods=self.signal_window).std()
            out.append(y * np.exp(-(y ** 2) / 4.0) / 0.89)                 # the response peaks at |y| = sqrt(2) and decays for very extended trends
        return _valid(sum(out) / len(out), data)


@register_model("breakout_ensemble", "time-series", "Breakout ensemble (Carver): where the price sits in its 10- to 320-day range, smoothed and averaged across horizons")
class BreakoutEnsemble(ForecastModel):
    """Price near the top of a long range is a trend in progress; averaging many range lengths gives a smooth, low-turnover signal."""

    name, family, position_mode = "breakout_ensemble", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, lookbacks: tuple = (10, 20, 40, 80, 160, 320), min_horizons: int = 3):
        if not lookbacks or min(lookbacks) < 2 or min_horizons < 1:
            raise ValueError("lookbacks of at least 2 days and min_horizons >= 1")
        self.lookbacks, self.min_horizons = tuple(int(x) for x in lookbacks), int(min_horizons)

    def score(self, data):
        close = data.prices
        parts = []
        for n in self.lookbacks:
            hi, lo = close.rolling(n, min_periods=n).max(), close.rolling(n, min_periods=n).min()
            raw = 40.0 * (close - (hi + lo) / 2.0) / (hi - lo).replace(0.0, np.nan)
            parts.append(raw.ewm(span=max(2, int(np.ceil(n / 4))), adjust=False, min_periods=1).mean().where(raw.notna()))
        stack = np.stack([p.to_numpy() for p in parts])
        count = np.isfinite(stack).sum(axis=0)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=0)
        score = pd.DataFrame(np.where(count >= self.min_horizons, mean, np.nan), index=data.index, columns=close.columns).clip(-20, 20) / 20.0
        return _valid(score, data)


def _kama(prices: np.ndarray, n: int, fast: int, slow: int) -> np.ndarray:
    """Kaufman's adaptive moving average, one column at a time (the recursion is inherently sequential)."""
    out = np.full(prices.shape, np.nan)
    fast_sc, slow_sc = 2.0 / (fast + 1.0), 2.0 / (slow + 1.0)
    for j in range(prices.shape[1]):
        p = prices[:, j]
        finite = np.flatnonzero(np.isfinite(p))
        if len(finite) <= n:
            continue
        start = finite[0] + n
        change = np.abs(p[n:] - p[:-n])
        path = np.abs(np.diff(p, prepend=np.nan))
        csum = np.concatenate([np.zeros(n), np.nancumsum(path)[n:] - np.nancumsum(path)[:-n]]) if len(p) > n else np.zeros_like(p)
        k = np.nan
        for t in range(start, len(p)):
            if not np.isfinite(p[t]):
                out[t, j] = k
                continue
            if not np.isfinite(k):
                k = p[t]
            denom = csum[t]
            er = abs(p[t] - p[t - n]) / denom if denom > 0 and np.isfinite(p[t - n]) else 0.0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            k = k + sc * (p[t] - k)
            out[t, j] = k
    return out


@register_model("kama_trend", "time-series", "Kaufman adaptive moving average trend: long above the adaptive average, short below, sized by distance in volatility units")
class KaufmanTrend(ForecastModel):
    """An average that speeds up in clean trends and slows in noise gives fewer whipsaws than a fixed-length average."""

    name, family, position_mode = "kama_trend", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, er_window: int = 10, fast: int = 2, slow: int = 30, vol_window: int = 20):
        if er_window < 2 or not 1 <= fast < slow:
            raise ValueError("er_window >= 2 and 1 <= fast < slow")
        self.er_window, self.fast, self.slow, self.vol_window = er_window, fast, slow, vol_window

    def score(self, data):
        close = data.prices
        kama = pd.DataFrame(_kama(close.to_numpy(dtype=float), self.er_window, self.fast, self.slow), index=close.index, columns=close.columns)
        vol = data.returns.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(self.er_window)
        return _valid(np.tanh((close / kama - 1.0) / vol.replace(0.0, np.nan)), data)


@register_model("adx_trend", "time-series", "Wilder's directional movement: follow the stronger of +DI and -DI, but only while the ADX says a trend exists (default above 25)")
class ADXTrend(ForecastModel):
    """Trend-following loses in trendless markets; the ADX measures how directional the recent range has been and switches the strategy off in chop."""

    name, family, position_mode = "adx_trend", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, window: int = 14, threshold: float = 25.0):
        if window < 2 or threshold < 0:
            raise ValueError("window >= 2 and threshold >= 0")
        self.window, self.threshold = window, threshold

    def indicators(self, data) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        high, low = high_low(data)
        close = data.prices
        up, down = high.diff(), -low.diff()
        plus = up.where((up > down) & (up > 0), 0.0)
        minus = down.where((down > up) & (down > 0), 0.0)
        prev = close.shift(1)
        tr = pd.concat([(high - low).stack(), (high - prev).abs().stack(), (low - prev).abs().stack()], axis=1).max(axis=1).unstack().reindex_like(close)
        atr = wilder(tr, self.window)
        plus_di, minus_di = 100.0 * wilder(plus, self.window) / atr, 100.0 * wilder(minus, self.window) / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        return plus_di, minus_di, wilder(dx.fillna(0.0).where(plus_di.notna()), self.window)

    def score(self, data):
        plus_di, minus_di, adx = self.indicators(data)
        score = np.sign(plus_di - minus_di) * (adx >= self.threshold)
        return _valid(score.where(adx.notna()), data)


@register_model("supertrend", "time-series", "SuperTrend: an ATR band that trails price and flips the position when the close crosses it (10-day ATR, multiplier 3)")
class SuperTrend(ForecastModel):
    """A stop that only ever tightens in the direction of the trend turns 'is the trend intact?' into a yes or a no."""

    name, family, position_mode = "supertrend", "trend", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, atr_window: int = 10, multiplier: float = 3.0):
        if atr_window < 2 or multiplier <= 0:
            raise ValueError("atr_window >= 2 and multiplier > 0")
        self.atr_window, self.multiplier = atr_window, multiplier

    def score(self, data):
        close = data.prices
        high, low = high_low(data)
        mid = ((high + low) / 2.0).to_numpy(dtype=float)
        atr = (atr_pct(data, self.atr_window) * close).to_numpy(dtype=float)
        c = close.to_numpy(dtype=float)
        upper_basic, lower_basic = mid + self.multiplier * atr, mid - self.multiplier * atr
        n, k = c.shape
        upper, lower, direction = np.full((n, k), np.nan), np.full((n, k), np.nan), np.zeros((n, k))
        for t in range(n):
            ok = np.isfinite(upper_basic[t]) & np.isfinite(c[t])
            if not ok.any():
                continue
            if t == 0 or not np.isfinite(upper[t - 1]).any():
                upper[t], lower[t] = np.where(ok, upper_basic[t], np.nan), np.where(ok, lower_basic[t], np.nan)
                direction[t] = np.where(ok, np.where(c[t] >= mid[t], 1.0, -1.0), 0.0)
                continue
            pu, pl, pc, pd_ = upper[t - 1], lower[t - 1], c[t - 1], direction[t - 1]
            fresh = ~np.isfinite(pu) & ok
            tighter_u = (upper_basic[t] < pu) | (pc > pu)
            tighter_l = (lower_basic[t] > pl) | (pc < pl)
            upper[t] = np.where(ok, np.where(fresh | tighter_u, upper_basic[t], pu), np.nan)
            lower[t] = np.where(ok, np.where(fresh | tighter_l, lower_basic[t], pl), np.nan)
            flip_up = (pd_ <= 0) & (c[t] > pu)
            flip_down = (pd_ >= 0) & (c[t] < pl)
            direction[t] = np.where(ok, np.where(fresh, np.where(c[t] >= mid[t], 1.0, -1.0), np.where(flip_up, 1.0, np.where(flip_down, -1.0, pd_))), 0.0)
        out = pd.DataFrame(direction, index=close.index, columns=close.columns)
        return _valid(out.where(pd.DataFrame(np.isfinite(upper), index=close.index, columns=close.columns)), data)


@register_model("keltner_breakout", "time-series", "Keltner channel breakout: enter beyond the 20-day EMA plus or minus two ATRs, hold until the close crosses the EMA back")
class KeltnerBreakout(ForecastModel):
    """A close outside a volatility-scaled envelope is a move bigger than noise; the middle line is the exit."""

    name, family, position_mode = "keltner_breakout", "trend", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, window: int = 20, k: float = 2.0, atr_window: int = 14):
        if window < 2 or k <= 0 or atr_window < 2:
            raise ValueError("window >= 2, k > 0, atr_window >= 2")
        self.window, self.k, self.atr_window = window, k, atr_window

    def score(self, data):
        close = data.prices
        ema = close.ewm(span=self.window, adjust=False, min_periods=self.window).mean()
        band = self.k * atr_pct(data, self.atr_window) * close
        out = stateful(close > ema + band, close < ema, close < ema - band, close > ema)
        return _valid(out.where(ema.notna() & band.notna()), data)


@register_model("macd_trend", "time-series", "MACD trend: the 12/26-day EMA difference (the MACD line) scaled by the price's own volatility; optionally its histogram against the 9-day signal line")
class MACDTrend(ForecastModel):
    """The gap between a fast and a slow average is positive in up-trends and negative in down-trends; its histogram (the gap minus its own average) measures whether the trend is strengthening."""

    name, family, position_mode = "macd_trend", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, vol_window: int = 20, use: str = "line"):
        if not 1 <= fast < slow or signal < 1 or use not in ("line", "histogram"):
            raise ValueError("1 <= fast < slow, signal >= 1 and use is 'line' or 'histogram'")
        self.fast, self.slow, self.signal, self.vol_window, self.use = fast, slow, signal, vol_window, use

    def score(self, data):
        close = data.prices
        macd = close.ewm(span=self.fast, adjust=False, min_periods=self.slow).mean() - close.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
        value = macd if self.use == "line" else macd - macd.ewm(span=self.signal, adjust=False, min_periods=self.signal).mean()
        scale = close * data.returns.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(self.slow)
        return _valid(np.tanh(value / scale.replace(0.0, np.nan)), data)


@register_model("ichimoku_trend", "time-series", "Ichimoku cloud: long above the cloud with the conversion line over the base line, short below the cloud with it under, flat inside")
class IchimokuTrend(ForecastModel):
    """The cloud is a support and resistance zone built from three ranges; trading only when price is clear of it avoids the middle of a range."""

    name, family, position_mode = "ichimoku_trend", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
        if not 1 <= tenkan < kijun < senkou:
            raise ValueError("1 <= tenkan < kijun < senkou")
        self.tenkan, self.kijun, self.senkou = tenkan, kijun, senkou

    def score(self, data):
        high, low = high_low(data)
        close = data.prices

        def mid(n):
            return (high.rolling(n, min_periods=n).max() + low.rolling(n, min_periods=n).min()) / 2.0

        conversion, base = mid(self.tenkan), mid(self.kijun)
        span_a = ((conversion + base) / 2.0).shift(self.kijun)              # the cloud drawn today was computed `kijun` days ago
        span_b = mid(self.senkou).shift(self.kijun)
        top, bottom = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
        long = (close > top) & (conversion > base)
        short = (close < bottom) & (conversion < base)
        score = long.astype(float) - short.astype(float)
        return _valid(score.where(top.notna()), data)


# ---------------------------------------------------------------------------------------------------------- cross-sectional momentum
@register_model("high_52w", "cross-sectional", "52-week-high proximity (George-Hwang): favour assets trading closest to their trailing 252-day high")
class High52Week(ForecastModel):
    """Investors anchor on the 52-week high and under-react to news that pushes a price toward it, so assets near their highs keep outperforming."""

    name, family = "high_52w", "momentum"

    def __init__(self, window: int = 252):
        if window < 20:
            raise ValueError("window must be at least 20 days")
        self.window = window

    def score(self, data):
        high = data.prices.rolling(self.window, min_periods=self.window).max()
        return _valid(data.prices / high - 1.0, data)


@register_model("smooth_momentum", "cross-sectional", "Frog-in-the-pan momentum (Da-Gurun-Warachka): 12-1 month return, boosted when the gain came in many small steps rather than a few jumps")
class SmoothMomentum(ForecastModel):
    """Investors under-react to information that arrives gradually, so momentum built from many small moves lasts longer than momentum from a few big ones."""

    name, family = "smooth_momentum", "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21):
        if lookback <= skip + 20:
            raise ValueError("lookback must exceed skip by at least 20 days")
        self.lookback, self.skip = lookback, skip

    def score(self, data):
        ret = data.prices.shift(self.skip) / data.prices.shift(self.lookback) - 1.0
        daily = data.returns.shift(self.skip)
        window = self.lookback - self.skip
        pos = (daily > 0).astype(float).where(daily.notna()).rolling(window, min_periods=window).mean()
        neg = (daily < 0).astype(float).where(daily.notna()).rolling(window, min_periods=window).mean()
        information_discreteness = np.sign(ret) * (neg - pos)             # low (negative) = the move came in many same-signed days
        return _valid(ret * (1.0 - information_discreteness), data)


@register_model("momentum_13612w", "cross-sectional", "Keller's 13612W momentum: the weighted average of the 1-, 3-, 6- and 12-month returns (weights 12, 4, 2, 1)")
class Momentum13612W(ForecastModel):
    """Weighting recent months heavily makes the signal react faster to turning points than a plain 12-month return, which is why the Keller allocations use it."""

    name, family = "momentum_13612w", "momentum"

    def __init__(self, month: int = 21):
        if month < 5:
            raise ValueError("month must be at least 5 trading days")
        self.month = month

    def score(self, data):
        m, p = self.month, data.prices
        parts = [(12, 1), (4, 3), (2, 6), (1, 12)]
        score = sum(w * (p / p.shift(m * k) - 1.0) for w, k in parts) / 19.0
        return _valid(score, data)


@register_model("accelerating_dual_momentum", "time-series", "Accelerating dual momentum: hold the top-k assets by the average 1-, 3- and 6-month return, but only those that beat cash")
class AcceleratingDualMomentum(ForecastModel):
    """A shorter, averaged lookback reacts to a turn sooner than 12-month dual momentum, at the price of more switching."""

    name, family, position_mode = "accelerating_dual_momentum", "momentum", "time_series"
    rebalance = "weekly"

    def __init__(self, top_k: int = 2, cash: str = "SHY", month: int = 21):
        if top_k < 1 or month < 5:
            raise ValueError("top_k >= 1 and month >= 5")
        self.top_k, self.cash, self.month = top_k, cash, month

    def score(self, data):
        m, p = self.month, data.prices
        ret = ((p / p.shift(m) - 1.0) + (p / p.shift(3 * m) - 1.0) + (p / p.shift(6 * m) - 1.0)) / 3.0
        cash = ret[self.cash] if self.cash in ret.columns else pd.Series(0.0, index=ret.index)
        excess = ret.sub(cash, axis=0).where(data.investable)
        eligible = excess.drop(columns=[self.cash], errors="ignore")
        rank = eligible.rank(axis=1, ascending=False)
        chosen = ((rank <= self.top_k) & (eligible > 0)).astype(float).where(eligible.notna())
        return chosen.reindex(columns=p.columns).fillna(0.0).where(excess.notna().any(axis=1), np.nan)


@register_model("residual_momentum", "cross-sectional", "Residual momentum (Blitz-Huij-Martens): 12-1 month momentum of the part of each return that the market does not explain, scaled by its volatility")
class ResidualMomentum(ForecastModel):
    """Stripping out market beta leaves the asset-specific trend, which is steadier and less exposed to momentum crashes than raw momentum."""

    name, family = "residual_momentum", "momentum"

    def __init__(self, beta_window: int = 252, lookback: int = 252, skip: int = 21):
        if beta_window < 60 or lookback <= skip + 20:
            raise ValueError("beta_window >= 60 and lookback must exceed skip by at least 20 days")
        self.beta_window, self.lookback, self.skip = beta_window, lookback, skip

    def score(self, data):
        market = market_proxy(data)
        cov = data.returns.rolling(self.beta_window, min_periods=self.beta_window // 2).cov(market)
        beta = cov.div(market.rolling(self.beta_window, min_periods=self.beta_window // 2).var(), axis=0)
        resid = data.returns - beta.shift(1).mul(market, axis=0)               # the beta is known before the day it is applied to
        window = self.lookback - self.skip
        total = resid.shift(self.skip).rolling(window, min_periods=window).sum()
        sd = resid.shift(self.skip).rolling(window, min_periods=window).std() * np.sqrt(window)
        return _valid(total / sd.replace(0.0, np.nan), data)
