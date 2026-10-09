"""Factor timing: tilt a factor up, down or off according to the calendar, the macroeconomy or the market's own state, learning from history how it has paid in each state.

    calendar_factor_timing    the January effect, the month of the year, the month within the quarter, "sell in May": each price-based factor is forecast with the average premium it earned in the same
                              calendar state in earlier years
    macro_factor_timing       the same with a macroeconomic state: the direction of Fed policy, M1 growth, GDP growth, inflation, producer-price inflation, or whether the market is up or down
    earnings_season_premium   the seasonal earnings-announcement effect: stocks expected to announce this month (they announced in the same month a year ago) earn a premium, from your earnings dates
                              or, failing that, from the spikes in their trading volume

A factor's payoff is not constant. A *timing* model forecasts it from the state of the world, and the forecast of an asset is its exposure to each factor times the factor's forecast premium. Every
premium forecast uses only months whose returns are known (the month ``s`` return is known at ``s + 1``): in each state the mean of earlier premiums, shrunk toward the overall mean by ``n / (n + shrink)`` when
the state has been seen only ``n`` times, so a state seen twice does not flip a factor. Because it is a *forecast*, a factor can be switched off, or reversed, in a state where it has not paid. No sign is built in:
the January effect of momentum (a loss) is something the model finds in the data or does not.

All three are monthly decisions on data through the month-end. The macroeconomic series come with their publication lags from the platform's point-in-time macro panel; M1, GDP and producer prices are
revised after they are published and the files hold the latest versions, so the lags reduce that look-ahead and do not remove it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..equity.alpha_model import standardize
from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..utils.dates import rebalance_dates
from . import alpha_styles
from .factor_models import CHARACTERISTICS, _characteristic_list, _grid, _to_daily, price_characteristics


# ------------------------------------------------------------------------------------------------------------------ the premium forecast
def factor_slopes(z: np.ndarray, forward: np.ndarray, min_assets: int = 8) -> np.ndarray:
    """For each month-end, the cross-sectional slope of the next month's return on the exposure ``z`` (in standard deviations, with an intercept): the premium the factor earned that month, per unit of
    exposure. ``z`` and ``forward`` are (months, assets); NaN where a month has fewer than ``min_assets`` assets with both."""
    out = np.full(z.shape[0], np.nan)
    for s in range(z.shape[0]):
        ok = np.isfinite(z[s]) & np.isfinite(forward[s])
        if ok.sum() < min_assets:
            continue
        zz, yy = z[s, ok] - z[s, ok].mean(), forward[s, ok] - forward[s, ok].mean()
        den = float(zz @ zz)
        if den > 1e-12:
            out[s] = float(zz @ yy) / den
    return out


def conditional_premium(slopes: np.ndarray, state: np.ndarray, shrink: float = 12.0, min_obs: int = 24, window: int = 0) -> np.ndarray:
    """The forecast premium of month ``t``: the mean of the premiums earned in earlier months (those with ``s <= t - 1``, whose returns are known at ``t``), moved toward the mean of the earlier months in
    the same state as ``t`` by ``n / (n + shrink)``, where ``n`` is how many there were. ``state`` gives each month's state (an integer, ``-1`` if unknown), known at the decision date. ``window`` limits the
    history to that many months (``0``: all of it). NaN until ``min_obs`` earlier months exist."""
    T = len(slopes)
    out = np.full(T, np.nan)
    for t in range(T):
        if state[t] < 0:
            continue
        lo = 0 if window <= 0 else max(0, t - window)
        y, st = slopes[lo:t], state[lo:t]
        ok = np.isfinite(y)
        if ok.sum() < min_obs:
            continue
        m = float(y[ok].mean())
        sel = ok & (st == state[t])
        n = int(sel.sum())
        out[t] = m + (n / (n + shrink)) * (float(y[sel].mean()) - m) if n else m
    return out


# ------------------------------------------------------------------------------------------------------------------ states
def _target_month(grid: pd.DatetimeIndex) -> np.ndarray:
    """The calendar month (1 to 12) in which the return after each month-end is earned."""
    return ((grid.month.to_numpy() % 12) + 1).astype(int)


CALENDAR_STATES = ("january", "month", "quarter", "halloween")


def calendar_state(grid: pd.DatetimeIndex, kind: str) -> np.ndarray:
    """The state of the month a decision is for: ``january`` (1 if it is January), ``month`` (0 to 11), ``quarter`` (0, 1, 2: the month within the calendar quarter, the quarter-end month being 2) or
    ``halloween`` (1 for November to April, 0 for May to October)."""
    m = _target_month(grid)
    if kind == "january":
        return (m == 1).astype(int)
    if kind == "month":
        return m - 1
    if kind == "quarter":
        return (m - 1) % 3
    if kind == "halloween":
        return ((m >= 11) | (m <= 4)).astype(int)
    raise ValueError(f"state must be one of {', '.join(CALENDAR_STATES)}")


MACRO_STATES = {
    "fed": ("DFF", "Fed policy: the six-month change in the federal funds rate (easing, flat, tightening)"),
    "m1": ("M1SL", "M1 growth over a year, above or below its own median so far"),
    "gdp": ("GDPC1", "real GDP growth over a year, above or below its own median so far"),
    "inflation": ("CPIAUCNS", "CPI inflation over a year, above or below its own median so far"),
    "ppi": ("PPIACO", "producer-price inflation over a year, above or below its own median so far"),
    "market": (None, "the market's own state: the equal-weighted return of the universe over the last months, up or down"),
}


def _above_median(measure: pd.Series, min_history: int) -> np.ndarray:
    """1 where the measure is at or above its median over everything up to and including the date, 0 below it, -1 until ``min_history`` observations exist."""
    median = measure.expanding(min_periods=min_history).median()
    state = np.where(measure.notna() & median.notna(), (measure >= median).astype(int), -1)
    return state


def macro_state(data, kind: str, grid: pd.DatetimeIndex, market_months: int = 12, min_history: int = 60, flat_band: float = 0.25) -> np.ndarray:
    """The macroeconomic state at each month-end of ``grid``, from what had been published by then. ``fed``: 0 easing, 1 flat, 2 tightening (the funds rate changed by more than ``flat_band`` points over six
    months); ``m1``, ``gdp``, ``inflation``, ``ppi``: 1 if the year-on-year growth is at or above its median so far, else 0; ``market``: 1 if the universe's equal-weighted return over ``market_months`` is
    positive. -1 where the state is not yet defined."""
    if kind not in MACRO_STATES:
        raise ValueError(f"state must be one of {', '.join(MACRO_STATES)}")
    series, _ = MACRO_STATES[kind]
    if kind == "market":
        r = data.returns.where(data.investable).mean(axis=1).fillna(0.0)
        trailing = (1.0 + r).rolling(21 * market_months, min_periods=21 * market_months).apply(np.prod, raw=True) - 1.0
        v = trailing.reindex(grid)
        return np.where(v.notna(), (v > 0).astype(int), -1)
    s = data.macro[series].astype(float).ffill()
    if kind == "fed":
        now, before = s.reindex(grid), s.reindex(grid - pd.DateOffset(months=6), method="ffill")
        delta = pd.Series(now.to_numpy() - before.to_numpy(), index=grid)
        return np.where(delta.notna(), np.where(delta > flat_band, 2, np.where(delta < -flat_band, 0, 1)), -1)
    months = rebalance_dates(data.index, "monthly")                                                      # the median is over every month-end so far, whatever dates are asked for
    now, before = s.reindex(months), s.reindex(months - pd.DateOffset(years=1), method="ffill")
    growth = pd.Series(now.to_numpy() / before.to_numpy() - 1.0, index=months).replace([np.inf, -np.inf], np.nan)
    state = pd.Series(_above_median(growth, min_history), index=months)
    return state.reindex(grid, method="ffill").fillna(-1).astype(int).to_numpy()


# ------------------------------------------------------------------------------------------------------------------ the timing models
class _FactorTiming(ForecastModel):
    """Shared machinery: forecast each factor's premium from the state, multiply by the asset's exposure, sum over factors."""

    position_mode = "cross_sectional"
    min_assets = 8

    def _setup(self, factor: str, shrink: float, min_obs: int, window: int) -> None:
        names = list(CHARACTERISTICS) if factor.strip().lower() == "all" else _characteristic_list(factor)
        if shrink < 0 or min_obs < 6 or window < 0:
            raise ValueError("shrink >= 0, min_obs >= 6, window >= 0")
        self.factor, self.names, self.shrink, self.min_obs, self.window = factor, names, float(shrink), int(min_obs), int(window)

    def states(self, data, grid) -> np.ndarray:
        raise NotImplementedError

    def exposures(self, data, grid) -> dict:
        chars = price_characteristics(data)
        investable = data.investable.loc[grid]
        return {n: standardize(chars[n].loc[grid], investable).to_numpy() for n in self.names}

    def premia(self, data) -> pd.DataFrame:
        """The forecast premium of each factor at each month-end (months by factors): what the model believes the factors will pay next month given the state."""
        grid, forward = _grid(data)
        st = self.states(data, grid)
        fwd = forward.to_numpy()
        return pd.DataFrame({n: conditional_premium(factor_slopes(z, fwd), st, self.shrink, self.min_obs, self.window) for n, z in self.exposures(data, grid).items()}, index=grid)

    def score(self, data):
        grid, forward = _grid(data)
        st = self.states(data, grid)
        fwd = forward.to_numpy()
        total, seen = np.zeros(fwd.shape), np.zeros(fwd.shape, dtype=bool)
        for n, z in self.exposures(data, grid).items():
            f = conditional_premium(factor_slopes(z, fwd), st, self.shrink, self.min_obs, self.window)
            ok = np.isfinite(z) & np.isfinite(f)[:, None]
            total += np.where(ok, f[:, None] * np.nan_to_num(z), 0.0)
            seen |= ok
        return _to_daily(pd.DataFrame(np.where(seen, total, np.nan), index=grid, columns=data.returns.columns), data)


@register_model("calendar_factor_timing", "seasonal", "Calendar factor timing: forecast each price factor with the premium it earned in the same calendar state before (January or not, the month, the month within the quarter, Nov-Apr or May-Oct)")
class CalendarFactorTiming(_FactorTiming):
    """The calendar is known in advance, so a rule that depends on it is not look-ahead; it is, however, among the most data-mined ideas in finance (the January effect, turn of the quarter, sell in May),
    which is why the premium in each state is learned from earlier years only and shrunk toward the factor's overall premium until the state has been seen often. ``state`` is ``january``,
    ``month`` (twelve states, most to learn), ``quarter`` (the month within the quarter) or ``halloween``; ``factor`` is one price-based factor (``mom``, ``rev``, ``lowvol``, ``lowbeta``, ``nomax``,
    ``high``), a comma-separated list, or ``all``."""

    name, family = "calendar_factor_timing", "seasonal"

    def __init__(self, factor: str = "all", state: str = "january", shrink: float = 12.0, min_obs: int = 24, window: int = 0):
        if state not in CALENDAR_STATES:
            raise ValueError(f"state must be one of {', '.join(CALENDAR_STATES)}")
        self._setup(factor, shrink, min_obs, window)
        self.state = state

    def states(self, data, grid):
        return calendar_state(grid, self.state)


@register_model("macro_factor_timing", "macro", "Macro factor timing: forecast each price factor with the premium it earned before in the same macroeconomic state (Fed easing or tightening, M1, GDP, CPI or PPI growth, an up or down market)")
class MacroFactorTiming(_FactorTiming):
    """Factors pay differently in different economies: momentum in rising and falling markets (Cooper, Gutierrez and Hameed 2004), value in expansions and contractions, low-risk stocks when policy
    tightens. ``state`` is ``market`` (the default: needs no series), ``fed`` (the direction of the federal funds rate, series ``DFF``), ``m1`` (``M1SL``), ``gdp`` (``GDPC1``), ``inflation``
    (``CPIAUCNS``) or ``ppi`` (``PPIACO``); the series come with their publication lags from the macro panel. The premium forecast in a state is what the factor earned in the same state in earlier
    months, shrunk toward its overall mean; states are classified from data available at the date (growth against its own median so far, never against the full sample)."""

    name, family = "macro_factor_timing", "macro"

    def __init__(self, factor: str = "all", state: str = "market", shrink: float = 12.0, min_obs: int = 36, window: int = 0, market_months: int = 12, min_history: int = 60, flat_band: float = 0.25):
        if state not in MACRO_STATES:
            raise ValueError(f"state must be one of {', '.join(MACRO_STATES)}")
        if market_months < 1 or min_history < 12 or flat_band < 0:
            raise ValueError("market_months >= 1, min_history >= 12, flat_band >= 0")
        self._setup(factor, shrink, min_obs, window)
        self.state, self.market_months, self.min_history, self.flat_band = state, int(market_months), int(min_history), float(flat_band)

    def require(self, data) -> None:
        series = MACRO_STATES[self.state][0]
        if series is not None and (series not in data.macro.columns or data.macro[series].dropna().empty):
            raise KeyError(f"macro_factor_timing with state '{self.state}' needs macro series ['{series}'], which this bundle does not have")

    def states(self, data, grid):
        self.require(data)
        return macro_state(data, self.state, grid, self.market_months, self.min_history, self.flat_band)


# ------------------------------------------------------------------------------------------------------------------ the earnings announcement premium
def announcement_events(data, path: str = "earnings_dates.csv", source: str = "auto", volume_multiple: float = 3.0, spacing: int = 45) -> pd.DataFrame:
    """A (days by assets) table of 1.0 on the days a company announced earnings. ``source``: ``file`` (``date, ticker`` rows in ``data/user/earnings_dates.csv``), ``volume`` (the proxy: the day of a
    trading-volume spike, ``volume_multiple`` times the median of the previous 60 days and the largest within ``spacing`` days either side), or ``auto`` (the file if it exists, else the proxy)."""
    idx, cols = data.index, list(data.assets)
    p = Path(path)
    p = p if p.is_absolute() else alpha_styles.USER_DATA / p
    if source in ("file", "auto") and p.exists():
        table = alpha_styles._read(p)
        if not {"date", "ticker"} <= set(table.columns):
            raise ValueError(f"{p} needs the columns date and ticker")
        table["date"] = pd.to_datetime(table["date"])
        table = table[table["ticker"].isin(cols)]
        pos = idx.searchsorted(table["date"].to_numpy())                                           # the first trading day on or after the announcement
        keep = pos < len(idx)
        flags = np.zeros((len(idx), len(cols)))
        flags[pos[keep], [cols.index(t) for t in table["ticker"][keep]]] = 1.0
        return pd.DataFrame(flags, index=idx, columns=cols)
    if source == "file":
        alpha_styles._resolve(path, "the earnings announcement premium", "Columns: date, ticker (one row per announcement; the date is the first day the news could be traded on).")
    if data.volume is None:
        raise KeyError(f"the earnings announcement premium needs the file {p}. Columns: date, ticker (one row per announcement), or a bundle with trading volume, from which announcements are inferred as volume spikes.")
    vol = data.volume.reindex(idx)
    spike = vol / vol.rolling(60, min_periods=40).median().shift(1)
    peak = spike.rolling(2 * spacing + 1, center=True, min_periods=1).max()
    return ((spike >= volume_multiple) & (spike >= peak)).astype(float)


@register_model("earnings_season_premium", "event-driven", "Seasonal earnings announcement premium: favour stocks expected to announce earnings next month (they announced in the same month a year ago), with the premium learned from earlier months")
class EarningsSeasonPremium(ForecastModel):
    """Stocks earn more in the months in which they announce earnings (Frazzini and Lamont 2007, Barber, De George, Lehavy and Trueman 2013), and the months are predictable: companies keep their
    reporting calendar. The model flags, at each month-end, the stocks that announced in the next calendar month a year earlier, and forecasts with the average premium the flag has earned in earlier
    months (it is learned, so a flag that has not paid is switched off or reversed). Announcements come from ``data/user/earnings_dates.csv`` (``date, ticker``) if it exists, else are inferred from spikes in trading volume,
    which needs a bundle with volume and works only for companies, not funds."""

    name, family, position_mode = "earnings_season_premium", "event-driven", "cross_sectional"
    min_assets = 8

    def __init__(self, source: str = "auto", path: str = "earnings_dates.csv", window: int = 0, min_obs: int = 24, volume_multiple: float = 3.0, shrink: float = 0.0):
        if source not in ("auto", "file", "volume") or min_obs < 6 or window < 0 or volume_multiple <= 1:
            raise ValueError("source in auto, file, volume; min_obs >= 6; window >= 0; volume_multiple > 1")
        self.source, self.path, self.window, self.min_obs, self.volume_multiple, self.shrink = source, path, int(window), int(min_obs), float(volume_multiple), float(shrink)

    def expected(self, events: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
        """1 for the stocks that announced in the calendar month a year before the month after each month-end of ``grid``."""
        month_of = events.index.to_period("M")
        by_month = events.groupby(month_of).max()                                                      # announced at all in that month
        target = grid.to_period("M") + 1 - 12
        return by_month.reindex(target).fillna(0.0).set_axis(grid, axis=0).astype(float)

    def score(self, data):
        events = announcement_events(data, self.path, self.source, self.volume_multiple)
        grid, forward = _grid(data)
        flag = self.expected(events, grid)
        z = standardize(flag.where(data.investable.loc[grid]), data.investable.loc[grid]).to_numpy()
        slopes = factor_slopes(z, forward.to_numpy())
        f = conditional_premium(slopes, np.zeros(len(grid), dtype=int), self.shrink, self.min_obs, self.window)
        out = pd.DataFrame(f[:, None] * z, index=grid, columns=data.returns.columns)
        return _to_daily(out, data)
