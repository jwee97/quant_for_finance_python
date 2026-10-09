"""Alpha-generating styles: long-term, short-term, company news, company outlook and corporate-action strategies.

Two of the styles need information that prices and volume do not carry (what a company announced, what analysts expect, which index a stock joined). This repository has no such feed, so those strategies come
in two forms and say which: a *proxy* built from prices and volume alone (a big move on heavy volume is what news looks like in a price series), and a *data-driven* model that reads a file you supply and treats it
with the same discipline as a price: it is usable only after its date plus a publication lag, and nothing about it is assumed beyond what past observations show.

    jensen_alpha           long-term: the persistent, market-adjusted return of the last three years, in units of its own noise (the appraisal ratio)
    adaptive_autocorrelation   short-term: yesterday's move, continued where an asset's own returns have positive first-order autocorrelation and faded where it is negative, only when that autocorrelation is significant
    squeeze_breakout       short-term: a Bollinger squeeze (volatility at the low end of its own range) followed by a close outside the band
    abnormal_volume_drift  news proxy: a big move on heavy volume carries on for days, a big move on ordinary volume is faded
    event_study_drift      news, corporate actions, index changes, anything with a date: a walk-forward event study that learns each event type's abnormal-return path from COMPLETED past events only and
                           trades it when its t-statistic is large enough (events from price shocks by default, or from your events file)
    news_sentiment         company news from headlines you supply: a small finance word list scores each headline, scores decay with a half-life and are usable the day after
    panel_signal           company outlook (analyst revisions, earnings surprises, guidance, any score you have): a date-by-ticker file turned into a signal with a publication lag and an expiry

Files go in ``data/user/`` (ignored by git); relative paths are read from there. ``event_study_drift`` reads ``date, ticker, type`` (and optionally ``size``); ``news_sentiment`` reads ``date, ticker, headline``;
``panel_signal`` reads a wide table (``date`` plus one column per ticker) or a long one (``date, ticker, value``). Dates are the first day the information could have been traded on: for news that
broke after the close, use the next day.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import daily_vol, stateful

USER_DATA = Path(__file__).resolve().parents[2] / "data" / "user"
MISSING = ["", "NaN", "nan", "N/A", "n/a", "#N/A"]                                                          # NOT "NA" or "NULL": those are tickers and event types in real files


def _read(path: Path) -> pd.DataFrame:
    """A user file as a table. Only the usual spellings of "missing" are treated as missing, so a ticker called NA or an event type called NULL survives."""
    return pd.read_csv(path, keep_default_na=False, na_values=MISSING)


def _resolve(path: str, what: str, columns: str) -> Path:
    """The file a data-driven model reads; a relative name is looked for in ``data/user/``. A missing file is explained, not guessed at."""
    p = Path(path)
    p = p if p.is_absolute() else USER_DATA / p
    if not p.exists():
        raise KeyError(f"{what} needs the file {p}. {columns} Put it there (data/user/ is not part of the repository) or pass its path.")
    return p


def _market(data) -> pd.Series:
    """The equal-weight average return of the investable assets: the benchmark abnormal returns and betas are measured against. With a single asset there is no market to subtract
    (the average would be the asset itself), so the benchmark is zero and abnormal returns are plain returns."""
    if data.returns.shape[1] < 2:
        return pd.Series(0.0, index=data.returns.index)
    return data.returns.where(data.investable).mean(axis=1)


# ------------------------------------------------------------------------------------------------------------------------------- long term
@register_model("jensen_alpha", "cross-sectional", "Persistent alpha (Jensen): favour assets whose market-adjusted return over the last three years was large relative to its noise, skipping the last month")
class JensenAlpha(ForecastModel):
    """The part of an asset's return that its exposure to the market does not explain, if it has been steady, tends to persist: managers and assets with a record of alpha keep a little of it, and
    ranking on the appraisal ratio (alpha over residual volatility) rather than on alpha alone demotes the lucky streaks. The market is the equal-weight average of the assets here; the last month is skipped as
    in 12-1 momentum, to keep short-term reversal out of a long-term signal."""

    name, family, position_mode = "jensen_alpha", "cross-sectional", "cross_sectional"

    def __init__(self, window: int = 756, skip: int = 21):
        if window < 120 or skip < 0:
            raise ValueError("window >= 120 and skip >= 0")
        self.window, self.skip = window, skip

    def score(self, data):
        r = data.returns.where(data.investable)
        m = _market(data)
        w, mp = self.window, max(self.window // 2, 60)
        beta = r.rolling(w, min_periods=mp).cov(m).div(m.rolling(w, min_periods=mp).var(), axis=0)
        alpha = r.rolling(w, min_periods=mp).mean() - beta.mul(m.rolling(w, min_periods=mp).mean(), axis=0)
        resid = (r.rolling(w, min_periods=mp).var() - beta.pow(2).mul(m.rolling(w, min_periods=mp).var(), axis=0)).clip(lower=1e-12)
        return (alpha / np.sqrt(resid)).shift(self.skip).where(data.investable)


# ------------------------------------------------------------------------------------------------------------------------------- short term
@register_model("adaptive_autocorrelation", "time-series", "Adaptive autocorrelation: continue yesterday's move where an asset's own returns have significant positive autocorrelation, fade it where it is significantly negative")
class AdaptiveAutocorrelation(ForecastModel):
    """Short-horizon returns are weakly predictable from their own past, with a sign that differs by asset and period (momentum in some, bid-ask bounce and overreaction in others). Measuring the
    first-order autocorrelation over the last year and acting only when its t-statistic says it is not noise lets each asset choose between continuing and fading, and stay out when there is nothing to find."""

    name, family, position_mode = "adaptive_autocorrelation", "time-series", "time_series"
    book, rebalance, horizon = "sleeves", "daily", 5

    def __init__(self, window: int = 250, scale: float = 2.0):
        if window < 60 or scale <= 0:
            raise ValueError("window >= 60 and scale > 0")
        self.window, self.scale = window, scale

    def score(self, data):
        r = data.returns
        rho = r.rolling(self.window, min_periods=self.window // 2).corr(r.shift(1))
        t = rho * np.sqrt(self.window)
        move = (r / daily_vol(r)).clip(-3.0, 3.0)
        return (np.tanh(t / self.scale) * move).where(data.investable)


@register_model("squeeze_breakout", "time-series", "Volatility squeeze breakout: when Bollinger bandwidth has been in the lowest fifth of its range, go with the first close outside the band and hold until the close crosses the middle band")
class SqueezeBreakout(ForecastModel):
    """Volatility clusters in time, so a spell of unusually narrow ranges is followed by a wider one; the direction of the first decisive close out of the band is the best available guess for the direction
    of the expansion. The squeeze must have been present the day BEFORE the breakout, so the band that is broken is not itself widened by the breakout bar."""

    name, family, position_mode = "squeeze_breakout", "time-series", "time_series"
    book, rebalance, horizon = "sleeves", "daily", 10

    def __init__(self, window: int = 20, k: float = 2.0, lookback: int = 126, quantile: float = 0.2):
        if window < 5 or k <= 0 or lookback < 40 or not 0 < quantile < 1:
            raise ValueError("window >= 5, k > 0, lookback >= 40, 0 < quantile < 1")
        self.window, self.k, self.lookback, self.quantile = window, k, lookback, quantile

    def score(self, data):
        c = data.prices
        mid, sd = c.rolling(self.window, min_periods=self.window).mean(), c.rolling(self.window, min_periods=self.window).std()
        upper, lower = mid + self.k * sd, mid - self.k * sd
        width = (upper - lower) / mid
        squeezed = width < width.rolling(self.lookback, min_periods=self.lookback // 2).quantile(self.quantile)
        was_squeezed = squeezed.shift(1, fill_value=False)
        out = stateful(was_squeezed & (c > upper.shift(1)), c < mid, was_squeezed & (c < lower.shift(1)), c > mid)
        return out.where(upper.notna() & width.rolling(self.lookback, min_periods=self.lookback // 2).mean().notna() & data.investable)


# ------------------------------------------------------------------------------------------------------------------------------- news and events
@register_model("abnormal_volume_drift", "time-series", "News proxy: a move of more than two standard deviations on at least twice the usual volume drifts on for days; the same move on ordinary volume is faded")
class AbnormalVolumeDrift(ForecastModel):
    """News shows in a price series as a large move that other traders confirm by trading: that kind of move tends to continue as the information spreads (under-reaction), while a large move nobody trades
    on tends to be noise that reverts (Chan 2003; Campbell, Grossman and Wang 1993 on volume and reversals). The signal is the sign of the move, decaying linearly over ``horizon`` days. A proxy: it
    cannot tell good news from bad news in a headline, only that something happened."""

    name, family, position_mode = "abnormal_volume_drift", "time-series", "time_series"
    book, rebalance, horizon = "sleeves", "daily", 10

    def __init__(self, k: float = 2.0, volume_ratio: float = 2.0, hold: int = 10, reversal: float = 0.5, vol_window: int = 60):
        if k <= 0 or volume_ratio <= 0 or hold < 2 or reversal < 0 or vol_window < 20:
            raise ValueError("k, volume_ratio > 0; hold >= 2; reversal >= 0; vol_window >= 20")
        self.k, self.volume_ratio, self.hold, self.reversal, self.vol_window = k, volume_ratio, hold, reversal, vol_window

    def score(self, data):
        if data.volume is None:
            raise KeyError("abnormal_volume_drift needs trading volume, which this bundle does not have")
        r = data.returns
        z = r / r.rolling(self.vol_window, min_periods=20).std().shift(1)
        volume = data.volume.reindex_like(r)
        heavy = volume > self.volume_ratio * volume.rolling(20, min_periods=10).median().shift(1)
        big = z.abs() > self.k
        impulse = (np.sign(r) * ((big & heavy).astype(float) - self.reversal * (big & ~heavy).astype(float))).fillna(0.0)
        out = sum((1.0 - j / self.hold) * impulse.shift(j, fill_value=0.0) for j in range(self.hold))
        return out.where(data.investable)


def _price_events(data, z: float, volume_ratio: float, vol_window: int = 60) -> pd.DataFrame:
    """Events found in the prices: ``pos`` (row), ``asset`` (column) and ``kind`` (``up``/``down``, and ``heavy``/``quiet`` volume when there is volume) of every move beyond ``z`` standard deviations."""
    r = data.returns.where(data.investable)
    shock = r / r.rolling(vol_window, min_periods=20).std().shift(1)
    big = shock.abs() >= z
    pos, asset = np.nonzero(big.to_numpy())
    kind = np.where(shock.to_numpy()[pos, asset] > 0, "up", "down").astype(object)
    if data.volume is not None:
        volume = data.volume.reindex_like(r)
        heavy = (volume > volume_ratio * volume.rolling(20, min_periods=10).median().shift(1)).to_numpy()[pos, asset]
        kind = np.array([f"{k}_{'heavy' if h else 'quiet'}" for k, h in zip(kind, heavy)], dtype=object)
    return pd.DataFrame({"pos": pos, "asset": asset, "kind": kind})


def _file_events(data, path: str) -> pd.DataFrame:
    """Events from a file of ``date, ticker, type`` (and optionally ``size``, whose sign splits each type in two): the first trading day on or after each date, for tickers in the universe."""
    table = _read(_resolve(path, "event_study_drift", "Columns: date, ticker, type (and optionally size)."))
    missing = {"date", "ticker", "type"} - set(table.columns)
    if missing:
        raise KeyError(f"the events file needs the columns date, ticker and type; missing {sorted(missing)}")
    table["date"] = pd.to_datetime(table["date"])
    columns = {c: i for i, c in enumerate(data.returns.columns)}
    table = table[table["ticker"].isin(columns)]
    pos = data.index.searchsorted(table["date"].to_numpy())
    keep = (pos < len(data.index))
    table, pos = table[keep], pos[keep]
    gap = (data.index[pos] - table["date"].to_numpy()).days if len(pos) else np.array([], int)
    keep = gap <= 5                                                                                           # an event on a day with no trading is rolled forward, but not across a week
    table, pos = table[keep], pos[keep]
    kind = table["type"].astype(str)
    if "size" in table.columns:
        kind = kind + np.where(table["size"].to_numpy() >= 0, "_up", "_down")
    return pd.DataFrame({"pos": pos, "asset": [columns[t] for t in table["ticker"]], "kind": kind.to_numpy(dtype=object)})


@register_model("event_study_drift", "event-driven", "Walk-forward event study: learn the abnormal-return path that follows each type of event from completed past events only, and trade it when its t-statistic is large (events from price shocks, or from your file)")
class EventStudyDrift(ForecastModel):
    """Corporate actions, index changes, announcements and price shocks all have the same structure: a date, an asset and a type, followed by a stretch of returns that is, on average, not what the market did.
    The strategy measures that average path by event TYPE, using only events whose whole window has finished by the decision date, and holds the expected remaining drift of every event in its window
    when (and only when) the average cumulative abnormal return of the type is at least ``tstat`` standard errors from zero on at least ``min_events`` events. It therefore learns whether a type
    continues or reverts, and stays flat on noise. Without a file the events are moves beyond ``z`` standard deviations, split by direction and by heavy or ordinary volume. The signal is the expected
    remaining abnormal return over its own standard deviation (squashed by tanh)."""

    name, family, position_mode = "event_study_drift", "event-driven", "time_series"
    book, rebalance, horizon = "sleeves", "daily", 10

    def __init__(self, path: str = "", z: float = 2.0, volume_ratio: float = 2.0, window: int = 10, min_events: int = 30, tstat: float = 2.0):
        if z <= 0 or volume_ratio <= 0 or window < 2 or min_events < 5 or tstat < 0:
            raise ValueError("z, volume_ratio > 0; window >= 2; min_events >= 5; tstat >= 0")
        self.path, self.z, self.volume_ratio, self.window, self.min_events, self.tstat = path, z, volume_ratio, window, min_events, tstat

    def events(self, data) -> pd.DataFrame:
        return _file_events(data, self.path) if self.path else _price_events(data, self.z, self.volume_ratio)

    def score(self, data):
        r = data.returns.where(data.investable)
        ar = r.sub(_market(data), axis=0).to_numpy()                                                          # abnormal return: the asset against the market
        T, K = ar.shape
        H = self.window
        ev = self.events(data).sort_values("pos", kind="stable").reset_index(drop=True)
        out = np.zeros((T, K))
        if len(ev) == 0:
            return pd.DataFrame(out, index=data.index, columns=data.returns.columns).where(data.investable)
        kinds = {k: i for i, k in enumerate(sorted(ev["kind"].unique()))}
        pos, asset, kind = ev["pos"].to_numpy(), ev["asset"].to_numpy(), np.array([kinds[k] for k in ev["kind"]])
        E = len(ev)
        path = np.full((E, H), np.nan)                                                                         # abnormal return h days after the event, h = 1 .. H
        for h in range(1, H + 1):
            ok = pos + h < T
            path[ok, h - 1] = ar[pos[ok] + h, asset[ok]]
        finished = pos + H                                                                                     # the day the last return of the window is known
        usable = np.isfinite(path).all(axis=1) & (finished < T)
        cum = np.where(usable, np.nansum(path, axis=1), 0.0)
        by_finish = np.argsort(finished, kind="stable")
        sig = daily_vol(data.returns).to_numpy()
        n = len(kinds)
        s_path, s_cum, s_sq, count = np.zeros((n, H)), np.zeros(n), np.zeros(n), np.zeros(n)
        nxt = 0
        sorted_pos = pos
        for t in range(T):
            while nxt < E and finished[by_finish[nxt]] <= t:                                                    # events whose whole window has been observed by the close of t
                e = by_finish[nxt]
                nxt += 1
                if usable[e]:
                    k = kind[e]
                    s_path[k] += path[e]
                    s_cum[k] += cum[e]
                    s_sq[k] += cum[e] ** 2
                    count[k] += 1
            lo, hi = np.searchsorted(sorted_pos, t - H + 1, "left"), np.searchsorted(sorted_pos, t, "right")  # events dated in (t - H, t]: still inside their window
            if hi <= lo:
                continue
            c = np.maximum(count, 1.0)
            mean_cum = s_cum / c
            var_cum = np.maximum((s_sq - c * mean_cum ** 2) / np.maximum(c - 1.0, 1.0), 1e-18)
            tstat = mean_cum / np.sqrt(var_cum / c)
            trade = (count >= self.min_events) & (np.abs(tstat) >= self.tstat)
            if not trade.any():
                continue
            profile = s_path / c[:, None]                                                                       # the average abnormal return by day, per type
            remaining = np.cumsum(profile[:, ::-1], axis=1)[:, ::-1]                                            # expected abnormal return from day h to the end of the window
            ids = np.arange(lo, hi)
            lag = t - pos[ids]                                                                                  # 0 on the event day: the whole window is still ahead
            k = kind[ids]
            live = trade[k]
            if not live.any():
                continue
            ids, lag, k = ids[live], lag[live], k[live]
            expected = remaining[k, lag]
            days_left = H - lag
            vol = sig[t, asset[ids]]
            z = expected / (np.where(np.isfinite(vol) & (vol > 0), vol, np.nan) * np.sqrt(days_left))
            np.add.at(out[t], asset[ids], np.nan_to_num(z))
        return pd.DataFrame(np.tanh(out), index=data.index, columns=data.returns.columns).where(data.investable)


# ------------------------------------------------------------------------------------------------------------------------------- headlines and scores you supply
POSITIVE = ("beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies", "gain", "gains", "profit", "profits", "growth", "record", "upgrade", "upgraded", "outperform", "raises",
            "raised", "strong", "exceeds", "exceeded", "wins", "approval", "approved", "buyback", "dividend", "expands", "expansion", "recovery", "improves", "improved", "optimistic", "boost",
            "boosts", "breakthrough", "partnership", "acquires", "tops", "robust", "rebound", "rebounds")
NEGATIVE = ("miss", "misses", "missed", "plunge", "plunges", "slump", "slumps", "fall", "falls", "loss", "losses", "decline", "declines", "downgrade", "downgraded", "underperform", "cuts", "cut",
            "weak", "warns", "warning", "lawsuit", "probe", "investigation", "recall", "default", "bankruptcy", "fraud", "layoffs", "shortfall", "delay", "delays", "fine", "fined", "penalty",
            "slows", "slowdown", "concern", "concerns", "pessimistic", "halts", "halted", "restates", "restatement", "dilution")
_WORD = re.compile(r"[a-z']+")


def headline_score(text: str) -> float:
    """A headline's tone from a small finance word list: ``(positive - negative) / (positive + negative)`` among the words it uses, 0 when none match. Crude by design: it shows the plumbing (dated
    text in, a decayed signal out) and is meant to be replaced by your own scorer or a ``score`` column."""
    words = _WORD.findall(str(text).lower())
    pos, neg = sum(w in POSITIVE for w in words), sum(w in NEGATIVE for w in words)
    return (pos - neg) / (pos + neg) if pos + neg else 0.0


def _decayed_events(data, table: pd.DataFrame, half_life: float, lag: int, expiry: int | None = None) -> pd.DataFrame:
    """Dated scores per ticker, rolled to the first trading day on or after their date, delayed by ``lag`` trading days, then decayed by ``half_life`` days (the sum of the live ones)."""
    columns = {c: i for i, c in enumerate(data.returns.columns)}
    table = table[table["ticker"].isin(columns)]
    pos = data.index.searchsorted(pd.to_datetime(table["date"]).to_numpy()) + lag
    keep = pos < len(data.index)
    impulse = np.zeros((len(data.index), len(columns)))
    np.add.at(impulse, (pos[keep], [columns[t] for t in table["ticker"].to_numpy()[keep]]), table["value"].to_numpy()[keep])
    decay = 0.5 ** (1.0 / half_life)
    level = np.zeros(len(columns))
    out = np.empty_like(impulse)
    for t in range(len(impulse)):
        level = level * decay + impulse[t]
        out[t] = level
    return pd.DataFrame(out, index=data.index, columns=data.returns.columns)


@register_model("news_sentiment", "event-driven", "Headline sentiment from a file you supply (date, ticker, headline): a finance word list scores each headline, scores decay with a half-life and act from the day after")
class NewsSentiment(ForecastModel):
    """Prices under-react to news for days, so the tone of recent headlines predicts the next few days' returns of the company they are about. Each headline gets a score in [-1, 1] from a word list (or
    from a ``score`` column of your own), the scores for a ticker add up and decay with a half-life, and a headline dated D is used from the trading day ``lag`` after D (the default of one respects
    news that arrives after the close). Needs ``data/user/headlines.csv`` or the path you give; without headlines it refuses to run rather than invent a signal."""

    name, family, position_mode = "news_sentiment", "event-driven", "time_series"
    book, rebalance, horizon = "sleeves", "daily", 5

    def __init__(self, path: str = "headlines.csv", half_life: float = 3.0, lag: int = 1, scale: float = 1.5):
        if half_life <= 0 or lag < 0 or scale <= 0:
            raise ValueError("half_life > 0, lag >= 0, scale > 0")
        self.path, self.half_life, self.lag, self.scale = path, half_life, lag, scale

    def score(self, data):
        table = _read(_resolve(self.path, "news_sentiment", "Columns: date, ticker, headline (or a numeric score column)."))
        if not {"date", "ticker"} <= set(table.columns) or not ({"headline"} <= set(table.columns) or "score" in table.columns):
            raise KeyError("the headlines file needs the columns date, ticker and headline (or score)")
        table["value"] = table["score"].astype(float) if "score" in table.columns else table["headline"].map(headline_score)
        return np.tanh(_decayed_events(data, table, self.half_life, self.lag) / self.scale).where(data.investable)


@register_model("panel_signal", "event-driven", "Company outlook from a score you supply (analyst revisions, earnings surprises, guidance): a date-by-ticker file, usable after a publication lag and for a limited time")
class PanelSignal(ForecastModel):
    """Any forward-looking company measure with a date (the change in analyst earnings estimates, the standardised earnings surprise, a guidance flag) is a signal once it is handled like a price: it may
    not be used before it was published (``lag``), it goes stale (``expiry`` trading days), and it is compared across companies (``standardise``). ``change`` replaces the level by its change over that many
    days, which is what a revision is. Needs ``data/user/signals.csv`` (wide: ``date`` and one column per ticker; or long: ``date, ticker, value``) or the path you give."""

    name, family, position_mode = "panel_signal", "event-driven", "cross_sectional"

    def __init__(self, path: str = "signals.csv", lag: int = 1, expiry: int = 63, change: int = 0, standardise: bool = True, direction: float = 1.0):
        if lag < 0 or expiry < 1 or change < 0 or direction == 0:
            raise ValueError("lag >= 0, expiry >= 1, change >= 0, direction non-zero")
        self.path, self.lag, self.expiry, self.change, self.standardise, self.direction = path, lag, expiry, change, standardise, direction

    def score(self, data):
        table = _read(_resolve(self.path, "panel_signal", "Wide: date plus one column per ticker; long: date, ticker, value."))
        table["date"] = pd.to_datetime(table["date"])
        if {"ticker", "value"} <= set(table.columns):
            table = table.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
        else:
            table = table.set_index("date")
        table = table.sort_index()
        panel = table.reindex(data.index.union(table.index)).ffill(limit=self.expiry).reindex(data.index)            # valid from its date, for `expiry` trading days
        panel = panel.shift(self.lag)
        if self.change:
            panel = panel - panel.shift(self.change)
        panel = panel.reindex(columns=data.returns.columns)
        if self.standardise:
            panel = panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1).replace(0.0, np.nan), axis=0)
        return (self.direction * panel).where(data.investable)
