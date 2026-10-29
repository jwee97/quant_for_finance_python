"""Rules from the practitioner books: Carver's forecast scaling, Clenow's regression momentum, Baltas and Kosowski's t-statistic trend, the Lo-MacKinlay variance-ratio switch,
Connors' RSI and "Double 7s", the realised-skewness premium and portfolio insurance (CPPI, TIPP).

Every score on day ``t`` uses data through ``t`` only (the rolling statistics, the expanding scalar estimates and the account of the insurance rule all look backwards).

``carver_ewmac``          Carver (2015, 2019): six exponentially weighted crossovers, each scaled to an average absolute forecast of 10 (the scalar is ESTIMATED from the data's own history,
                          as Carver advises, not copied from a table), capped at +-20, averaged and multiplied by the forecast diversification multiplier
``clenow_momentum``       Clenow (2015): annualised slope of the exponential regression of the last 90 days times its R-squared, above the 100-day average and without a 15% gap
``tstat_momentum``        Baltas and Kosowski (2013, 2020): the t-statistic of a year of daily log returns, capped at one, in place of the sign of the return
``variance_ratio_switch`` Lo and MacKinlay (1988), Campbell-Lo-MacKinlay (1997), Chan (2013): follow the last moves where the heteroscedasticity-robust variance ratio says they persist,
                          fade them where it says they reverse, stay out where it cannot tell
``connors_rsi``           Connors and Alvarez: the average of RSI(3), RSI(2) of the up/down streak and the percent rank of the day's return, bought when very low in an up-trend
``double_seven``          Connors and Alvarez (2008): buy a 7-day closing low above the 200-day average, sell a 7-day closing high
``skew_premium``          Amaya, Christoffersen, Jacobs and Vasquez (2015); Fernandez-Perez, Frijns, Fuertes and Miffre (2018): favour the most negatively skewed assets
``cppi``                  Black and Jones (1987), Perold (1986); with ``ratchet`` the TIPP of Estep and Kritzman (1988): the share at risk is a multiple of the cushion above a floor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import stateful, wilder

ANN = 252.0


def _valid(score: pd.DataFrame, data) -> pd.DataFrame:
    return score.where(data.investable)


# ------------------------------------------------------------------------------------------------------------------ Carver
def ewmac_raw(prices: pd.DataFrame, fast: int, slow: int, vol_span: int = 35) -> pd.DataFrame:
    """The raw EWMAC forecast: the gap between the fast and the slow exponentially weighted average, in units of the standard deviation of daily price changes (an exponentially weighted
    estimate with span ``vol_span``). ``fast`` and ``slow`` are spans, ``alpha = 2 / (span + 1)``."""
    gap = prices.ewm(span=fast, adjust=False, min_periods=slow).mean() - prices.ewm(span=slow, adjust=False, min_periods=slow).mean()
    sigma = prices.diff().ewm(span=vol_span, adjust=False, min_periods=vol_span).std()
    return gap / sigma.replace(0.0, np.nan)


def forecast_diversification_multiplier(corr: np.ndarray, floor: float = 0.0, cap: float = 2.5) -> float:
    """Carver's multiplier for equally weighted forecasts: ``1 / sqrt(w' C w)`` with ``w = 1/K`` and the correlations ``C`` floored at ``floor`` (he sets negative ones to zero), capped at ``cap``.
    Averaging K imperfectly correlated forecasts shrinks their average size; the multiplier puts it back."""
    k = corr.shape[0]
    c = np.maximum(corr, floor).copy()
    np.fill_diagonal(c, 1.0)
    value = 1.0 / np.sqrt(c.sum() / k ** 2)
    return float(min(value, cap))


@register_model("carver_ewmac", "time-series", "Carver's EWMAC forecast: six exponentially weighted crossovers (2/8 to 64/256 days), each scaled to an average absolute forecast of 10 from the data's own history, capped at 20 and averaged with a diversification multiplier")
class CarverEWMAC(ForecastModel):
    """Trend rules of different speeds earn different amounts and cost different turnover; scaling every rule to the same average size, capping the extremes and averaging them with a multiplier for
    their imperfect correlation gives one smooth forecast (+20 is the strongest buy, 0 no view, 10 an average-sized position)."""

    name, family, position_mode = "carver_ewmac", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, fast_spans: tuple = (2, 4, 8, 16, 32, 64), slow_multiple: float = 4.0, vol_span: int = 35, target_abs: float = 10.0, cap: float = 20.0, min_history: int = 500,
                 diversification: bool = True, fdm_cap: float = 2.5):
        if not fast_spans or min(fast_spans) < 1 or slow_multiple <= 1 or vol_span < 5 or target_abs <= 0 or cap < target_abs or min_history < 20 or fdm_cap < 1:
            raise ValueError("fast_spans non-empty and >= 1, slow_multiple > 1, vol_span >= 5, 0 < target_abs <= cap, min_history >= 20, fdm_cap >= 1")
        self.fast_spans, self.slow_multiple, self.vol_span = tuple(int(s) for s in fast_spans), float(slow_multiple), int(vol_span)
        self.target_abs, self.cap, self.min_history, self.diversification, self.fdm_cap = float(target_abs), float(cap), int(min_history), bool(diversification), float(fdm_cap)

    def raw_forecasts(self, data) -> list[pd.DataFrame]:
        return [ewmac_raw(data.prices, f, max(f + 1, int(round(f * self.slow_multiple))), self.vol_span) for f in self.fast_spans]

    def capped_forecasts(self, data) -> tuple[list[pd.DataFrame], pd.Series]:
        """The scaled and capped forecast of each speed, and the number of dates (so far) on which every speed had a value."""
        raw = self.raw_forecasts(data)
        everything = np.stack([r.to_numpy() for r in raw])                         # (speeds, dates, assets)
        finite = np.isfinite(everything)
        dates_ready = pd.Series(finite.all(axis=0).any(axis=1).cumsum(), index=data.index)
        out = []
        for r in raw:
            absolute = r.abs()
            pooled = absolute.sum(axis=1).cumsum() / absolute.notna().sum(axis=1).cumsum()          # the average absolute raw forecast of all assets through each date
            scalar = self.target_abs / pooled.replace(0.0, np.nan)
            out.append((r.mul(scalar, axis=0)).clip(-self.cap, self.cap))
        return out, dates_ready

    def multiplier(self, capped: list[pd.DataFrame]) -> pd.Series:
        """The forecast diversification multiplier through each date, from the expanding correlations of the capped forecasts pooled over assets."""
        k = len(capped)
        if k == 1 or not self.diversification:
            return pd.Series(1.0, index=capped[0].index)
        stack = np.stack([c.to_numpy() for c in capped])
        ok = np.isfinite(stack).all(axis=0)
        x = np.where(ok, stack, 0.0)
        n = ok.sum(axis=1).cumsum().astype(float)                                   # pooled observations through each date
        mean = np.stack([x[i].sum(axis=1).cumsum() for i in range(k)]) / np.where(n > 0, n, np.nan)
        out = np.full(len(n), np.nan)
        cross = {(i, j): (x[i] * x[j]).sum(axis=1).cumsum() for i in range(k) for j in range(i, k)}
        for t in range(len(n)):
            if n[t] < 3:
                continue
            c = np.eye(k)
            for i in range(k):
                for j in range(i + 1, k):
                    cov = cross[(i, j)][t] / n[t] - mean[i, t] * mean[j, t]
                    vi = cross[(i, i)][t] / n[t] - mean[i, t] ** 2
                    vj = cross[(j, j)][t] / n[t] - mean[j, t] ** 2
                    c[i, j] = c[j, i] = cov / np.sqrt(vi * vj) if vi > 0 and vj > 0 else 0.0
            out[t] = forecast_diversification_multiplier(c, 0.0, self.fdm_cap)
        return pd.Series(out, index=capped[0].index)

    def score(self, data):
        capped, ready = self.capped_forecasts(data)
        everything = np.stack([c.to_numpy() for c in capped])
        combined = pd.DataFrame(np.where(np.isfinite(everything).all(axis=0), everything.mean(axis=0), np.nan), index=data.index, columns=data.prices.columns)
        fdm = self.multiplier(capped)
        forecast = combined.mul(fdm, axis=0).clip(-self.cap, self.cap)
        forecast = forecast.mul(np.where(ready >= self.min_history, 1.0, np.nan), axis=0)
        return _valid(forecast / self.target_abs, data)                            # 1 = the average position, +-cap/target_abs at the extremes


# ------------------------------------------------------------------------------------------------------------------ Clenow
def exponential_regression(prices: pd.DataFrame, window: int, annual_days: int = 250) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The annualised slope of the least-squares line through ``log(price)`` over the last ``window`` days, as a simple return per year (``exp(slope * annual_days) - 1``), and the R-squared of the fit."""
    y = np.log(prices.where(prices > 0))
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    slope = y.rolling(window, min_periods=window).cov(t).div(t.rolling(window, min_periods=window).var(), axis=0)
    r2 = y.rolling(window, min_periods=window).corr(t) ** 2
    return np.expm1(slope * annual_days), r2


@register_model("clenow_momentum", "cross-sectional", "Clenow's momentum rank (Stocks on the Move): annualised slope of the exponential regression of the last 90 days times its R-squared, held only above the 100-day average and with no daily gap above 15%")
class ClenowMomentum(ForecastModel):
    """A steep, straight climb scores high; a steep but jagged one is discounted by the R-squared. Names below their 100-day average or with a large recent gap are not eligible."""

    name, family, position_mode = "clenow_momentum", "momentum", "cross_sectional"
    rebalance = "weekly"

    def __init__(self, window: int = 90, trend_window: int = 100, gap: float = 0.15, annual_days: int = 250, use_r2: bool = True):
        if window < 10 or trend_window < 5 or gap <= 0 or annual_days < 1:
            raise ValueError("window >= 10, trend_window >= 5, gap > 0, annual_days >= 1")
        self.window, self.trend_window, self.gap, self.annual_days, self.use_r2 = int(window), int(trend_window), float(gap), int(annual_days), bool(use_r2)

    def score(self, data):
        annual, r2 = exponential_regression(data.prices, self.window, self.annual_days)
        momentum = annual * r2 if self.use_r2 else annual
        trend = data.prices > data.prices.rolling(self.trend_window, min_periods=self.trend_window).mean()
        gapped = data.returns.abs().rolling(self.window, min_periods=1).max() > self.gap
        return _valid(momentum.where(trend & ~gapped), data)


# ------------------------------------------------------------------------------------------------------------------ Baltas-Kosowski
@register_model("tstat_momentum", "time-series", "t-statistic trend (Baltas-Kosowski): the t-statistic of a year of daily log returns, capped at plus or minus one, in place of the sign of the year's return")
class TStatMomentum(ForecastModel):
    """The sign of a return throws away how reliable the trend was. The t-statistic of the mean daily return weighs a steady drift above a lucky jump and moves in and out of positions gradually."""

    name, family, position_mode = "tstat_momentum", "trend", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, lookback: int = 252, scale: float = 1.0):
        if lookback < 20 or scale <= 0:
            raise ValueError("lookback >= 20 and scale > 0")
        self.lookback, self.scale = int(lookback), float(scale)

    def score(self, data):
        log_ret = np.log(data.prices).diff()
        mean, sd = log_ret.rolling(self.lookback, min_periods=self.lookback).mean(), log_ret.rolling(self.lookback, min_periods=self.lookback).std()
        t = mean / (sd / np.sqrt(self.lookback)).replace(0.0, np.nan)
        return _valid((t / self.scale).clip(-1.0, 1.0), data)


# ------------------------------------------------------------------------------------------------------------------ Lo-MacKinlay
def variance_ratio_stats(window_returns: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Lo and MacKinlay's variance ratio ``VR(q)`` and its heteroscedasticity-robust statistic ``z*(q)`` for each row of ``window_returns`` (rows are windows of ``n`` one-period log returns).

    ``VR(q) = sigma_c^2(q) / sigma_a^2`` with the unbiased estimators of the paper (overlapping q-sums, divisor ``m = q (n - q + 1)(1 - q / n)``), and
    ``z*(q) = sqrt(n) (VR - 1) / sqrt(theta)``, ``theta = 4 sum_{j=1}^{q-1} (1 - j / q)^2 delta_j``, ``delta_j = n sum_t d_t^2 d_{t-j}^2 / (sum_t d_t^2)^2``, ``d = r - mean``.
    Under a random walk (even a heteroscedastic one) ``z*`` is close to N(0, 1); a positive value means the moves persist, a negative one that they reverse."""
    w = np.asarray(window_returns, dtype=float)
    n = w.shape[1]
    if q < 2 or n < 2 * q:
        raise ValueError("q >= 2 and the window must hold at least 2q returns")
    mu = w.mean(axis=1, keepdims=True)
    d = w - mu
    d2 = d ** 2
    sigma_a = d2.sum(axis=1) / (n - 1)
    cs = np.cumsum(w, axis=1)
    sums = cs[:, q - 1:] - np.concatenate([np.zeros((w.shape[0], 1)), cs[:, :n - q]], axis=1)          # the n - q + 1 overlapping q-period returns
    m = q * (n - q + 1) * (1.0 - q / n)
    sigma_c = ((sums - q * mu) ** 2).sum(axis=1) / m
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = sigma_c / sigma_a
        denominator = d2.sum(axis=1) ** 2
        theta = np.zeros(w.shape[0])
        for j in range(1, q):
            delta = n * (d2[:, j:] * d2[:, :n - j]).sum(axis=1) / denominator
            theta += 4.0 * (1.0 - j / q) ** 2 * delta
        z = np.sqrt(n) * (vr - 1.0) / np.sqrt(theta)
    return vr, z


def rolling_variance_ratio(log_returns: pd.Series, window: int, q: int) -> tuple[pd.Series, pd.Series]:
    """``variance_ratio_stats`` over the trailing ``window`` returns ending on each date (NaN while the window is incomplete or holds a missing return)."""
    r = log_returns.to_numpy(dtype=float)
    vr, z = np.full(len(r), np.nan), np.full(len(r), np.nan)
    if len(r) >= window:
        w = sliding_window_view(r, window)
        ok = np.isfinite(w).all(axis=1)
        if ok.any():
            a, b = variance_ratio_stats(w[ok], q)
            idx = np.flatnonzero(ok) + window - 1
            vr[idx], z[idx] = a, b
    return pd.Series(vr, index=log_returns.index), pd.Series(z, index=log_returns.index)


@register_model("variance_ratio_switch", "time-series", "Variance-ratio switch (Lo-MacKinlay): follow the last q-day move where the robust variance ratio says moves persist, fade it where it says they reverse, stay out where it cannot tell")
class VarianceRatioSwitch(ForecastModel):
    """Whether a market trends or reverts is itself measurable. A variance ratio above one (autocorrelation of returns positive) favours momentum, below one favours mean reversion, and
    a statistic inside +-``z_crit`` is no evidence either way."""

    name, family, position_mode = "variance_ratio_switch", "regime", "time_series"
    rebalance = "weekly"
    book = "sleeves"

    def __init__(self, window: int = 252, q: int = 5, z_crit: float = 1.645, smooth: bool = False):
        if q < 2 or window < 4 * q or z_crit < 0:
            raise ValueError("q >= 2, window >= 4q and z_crit >= 0")
        self.window, self.q, self.z_crit, self.smooth = int(window), int(q), float(z_crit), bool(smooth)

    def statistics(self, data) -> tuple[pd.DataFrame, pd.DataFrame]:
        log_ret = np.log(data.prices).diff()
        vr, z = {}, {}
        for col in log_ret.columns:
            vr[col], z[col] = rolling_variance_ratio(log_ret[col], self.window, self.q)
        return pd.DataFrame(vr, index=data.index), pd.DataFrame(z, index=data.index)

    def score(self, data):
        _, z = self.statistics(data)
        last = np.sign(data.prices / data.prices.shift(self.q) - 1.0)
        strength = (z / (2.0 * self.z_crit)).clip(-1.0, 1.0) if self.smooth and self.z_crit > 0 else np.sign(z)
        return _valid((last * strength).where(z.abs() >= self.z_crit, 0.0).where(z.notna()), data)


# ------------------------------------------------------------------------------------------------------------------ Connors
def rsi(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder's relative strength index of ``x`` over ``n`` periods (0 to 100); 100 when there were gains and no losses, 50 when nothing moved."""
    change = x.diff()
    gain, loss = wilder(change.clip(lower=0.0), n), wilder((-change).clip(lower=0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + gain / loss)
    out = out.where(loss > 0, np.where(gain > 0, 100.0, 50.0))
    return out.where(gain.notna() & loss.notna())


def streak(close: pd.DataFrame) -> pd.DataFrame:
    """Consecutive higher closes (+1, +2, ...) or lower closes (-1, -2, ...); a close that equals the last resets it to zero."""
    c = close.to_numpy(dtype=float)
    out = np.full(c.shape, np.nan)
    run = np.zeros(c.shape[1])
    for t in range(1, c.shape[0]):
        step = np.sign(c[t] - c[t - 1])
        run = np.where(step > 0, np.where(run > 0, run + 1.0, 1.0), np.where(step < 0, np.where(run < 0, run - 1.0, -1.0), 0.0))
        out[t] = np.where(np.isfinite(step), run, np.nan)
        run = np.where(np.isfinite(step), run, 0.0)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def percent_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """The share (0 to 100) of the ``window`` values before each date that are below the value on that date."""
    a = x.to_numpy(dtype=float)
    out = np.full(a.shape, np.nan)
    if a.shape[0] > window:
        for j in range(a.shape[1]):
            w = sliding_window_view(a[:, j], window + 1)
            ok = np.isfinite(w).all(axis=1)
            below = (w[:, :-1] < w[:, -1:]).mean(axis=1) * 100.0
            out[window:, j] = np.where(ok, below, np.nan)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def connors_rsi(close: pd.DataFrame, rsi_len: int = 3, streak_len: int = 2, rank_len: int = 100) -> pd.DataFrame:
    """ConnorsRSI: the average of the RSI of the price, the RSI of the up/down streak and the percent rank of the one-day return."""
    return (rsi(close, rsi_len) + rsi(streak(close), streak_len) + percent_rank(close / close.shift(1) - 1.0, rank_len)) / 3.0


@register_model("connors_rsi", "time-series", "ConnorsRSI pullback (Connors-Alvarez): buy when the average of RSI(3), the RSI of the up/down streak and the percent rank of the day's return falls below 10 in an up-trend, sell above 70")
class ConnorsRSI(ForecastModel):
    """Three short-term measures of how stretched a fall is (the price's own RSI, the length of the losing streak, how large the day's loss is against the last 100) agree more often at real extremes than any alone."""

    name, family, position_mode = "connors_rsi", "reversion", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, rsi_len: int = 3, streak_len: int = 2, rank_len: int = 100, buy_below: float = 10.0, exit_above: float = 70.0, trend_len: int = 200, allow_short: bool = False,
                 sell_above: float = 90.0, cover_below: float = 30.0):
        if min(rsi_len, streak_len, rank_len) < 2 or trend_len < 5 or not 0 < buy_below < exit_above < 100 or not 0 < cover_below < sell_above < 100:
            raise ValueError("lengths >= 2, trend_len >= 5, 0 < buy_below < exit_above < 100 and 0 < cover_below < sell_above < 100")
        self.rsi_len, self.streak_len, self.rank_len, self.trend_len = int(rsi_len), int(streak_len), int(rank_len), int(trend_len)
        self.buy_below, self.exit_above, self.allow_short, self.sell_above, self.cover_below = float(buy_below), float(exit_above), bool(allow_short), float(sell_above), float(cover_below)

    def indicator(self, data) -> pd.DataFrame:
        return connors_rsi(data.prices, self.rsi_len, self.streak_len, self.rank_len)

    def score(self, data):
        crsi = self.indicator(data)
        up = data.prices > data.prices.rolling(self.trend_len, min_periods=self.trend_len).mean()
        down = data.prices < data.prices.rolling(self.trend_len, min_periods=self.trend_len).mean()
        position = stateful(long_in=(crsi < self.buy_below) & up, long_out=crsi > self.exit_above,
                            short_in=((crsi > self.sell_above) & down) if self.allow_short else None, short_out=(crsi < self.cover_below) if self.allow_short else None)
        return _valid(position.where(crsi.notna()), data)


@register_model("double_seven", "time-series", "Double 7s (Connors-Alvarez): buy a 7-day closing low while the price is above its 200-day average, sell a 7-day closing high")
class DoubleSeven(ForecastModel):
    """In an up-trend a close at a one-week low is a pullback that has usually reversed within days; the exit at a one-week high takes the bounce."""

    name, family, position_mode = "double_seven", "reversion", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, window: int = 7, trend_len: int = 200, allow_short: bool = False):
        if window < 2 or trend_len < 5:
            raise ValueError("window >= 2 and trend_len >= 5")
        self.window, self.trend_len, self.allow_short = int(window), int(trend_len), bool(allow_short)

    def score(self, data):
        c = data.prices
        low, high = c.rolling(self.window, min_periods=self.window).min(), c.rolling(self.window, min_periods=self.window).max()
        ma = c.rolling(self.trend_len, min_periods=self.trend_len).mean()
        position = stateful(long_in=(c <= low) & (c > ma), long_out=c >= high,
                            short_in=((c >= high) & (c < ma)) if self.allow_short else None, short_out=(c <= low) if self.allow_short else None)
        return _valid(position.where(ma.notna() & high.notna()), data)


# ------------------------------------------------------------------------------------------------------------------ skewness
@register_model("skew_premium", "cross-sectional", "Realised-skewness premium (Amaya et al., Fernandez-Perez et al.): favour the assets whose recent daily returns are the most negatively skewed, avoid the most positively skewed")
class SkewPremium(ForecastModel):
    """Investors overpay for lottery-like positive skewness, so assets with positively skewed returns are expensive and those with negative skewness earn a premium for it."""

    name, family, position_mode = "skew_premium", "factor", "cross_sectional"
    rebalance = "weekly"

    def __init__(self, window: int = 252):
        if window < 21:
            raise ValueError("window >= 21")
        self.window = int(window)

    def score(self, data):
        return _valid(-data.returns.rolling(self.window, min_periods=self.window).skew(), data)


# ------------------------------------------------------------------------------------------------------------------ portfolio insurance
def cppi_exposure(returns: np.ndarray, multiplier: float, floor: float, max_exposure: float, cash_daily: float, ratchet: bool) -> np.ndarray:
    """The share of the account at risk on each date, decided from the account's value through that date. The account starts at 1 on the first finite return; the floor is ``floor`` times
    the starting value grown at the cash rate (with ``ratchet``, the larger of that and ``floor`` times the highest value so far: TIPP); the share is ``multiplier x cushion / value``
    clipped to ``[0, max_exposure]``; the next day the account earns ``share x return + (1 - share) x cash``."""
    out = np.full(len(returns), np.nan)
    value, start, peak = 1.0, None, 1.0
    for t, r in enumerate(returns):
        if start is None:
            if not np.isfinite(r):
                continue
            start = t
            floor_value = floor
        else:
            if np.isfinite(r):
                value *= 1.0 + out[t - 1] * r + (1.0 - out[t - 1]) * cash_daily
            else:
                value *= 1.0 + cash_daily
            floor_value = floor * (1.0 + cash_daily) ** (t - start)
            peak = max(peak, value)
            if ratchet:
                floor_value = max(floor_value, floor * peak)
        cushion = value - floor_value
        out[t] = float(np.clip(multiplier * cushion / value, 0.0, max_exposure)) if value > 0 else 0.0
    return out


@register_model("cppi", "risk-managed", "Constant-proportion portfolio insurance (Black-Jones, Perold): hold a multiple of the cushion above a floor in the asset and the rest in cash; the ratchet option raises the floor with the account's high (TIPP)")
class CPPI(ForecastModel):
    """The multiple times the cushion is the amount at risk, so the account cannot fall through the floor unless the asset gaps by more than one over the multiple in a day. Each asset runs its own account."""

    name, family, position_mode = "cppi", "risk-managed", "time_series"
    rebalance = "daily"
    book = "sleeves"

    def __init__(self, multiplier: float = 4.0, floor: float = 0.85, max_exposure: float = 1.0, cash_rate: float = 0.0, ratchet: bool = False):
        if multiplier <= 0 or not 0.0 < floor < 1.0 or max_exposure <= 0 or cash_rate < 0:
            raise ValueError("multiplier > 0, 0 < floor < 1, max_exposure > 0, cash_rate >= 0")
        self.multiplier, self.floor, self.max_exposure, self.cash_rate, self.ratchet = float(multiplier), float(floor), float(max_exposure), float(cash_rate), bool(ratchet)

    def score(self, data):
        cash_daily = (1.0 + self.cash_rate) ** (1.0 / ANN) - 1.0
        out = {c: cppi_exposure(data.returns[c].to_numpy(dtype=float), self.multiplier, self.floor, self.max_exposure, cash_daily, self.ratchet) for c in data.returns.columns}
        return _valid(pd.DataFrame(out, index=data.index), data)
