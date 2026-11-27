"""Point-in-time multi-currency support for the portfolio allocation.

Quotation. Every series is held as **base currency units per one unit of the instrument's currency** (``base_per_local``): a USD-based investor holding a JPY stock has a series of USD per JPY, about 0.0067. A series quoted
the other way (JPY per USD, ``local_per_base``) is inverted on the way in, so a rise in the stored series always means the foreign currency strengthened against the base. The FX return of a day is ``R_fx = S_t / S_{t-1} - 1``.

Identity. For an unhedged foreign holding ``1 + R_base = (1 + R_local) * (1 + R_fx)``. With a hedge ratio ``k`` of the currency exposure (forward hedge, carry ignored) ``1 + R_base = (1 + R_local) * (1 + (1 - k) R_fx)``.

Time. Each observation has the time it refers to and the time it became *available* (``available_at``; the observation time when none is given). A rate is usable on a date only if it was available before that
date ended; the rate in force is the latest usable one, and one older than ``max_stale_days`` is missing. A missing rate is never 1, never carried past the stale limit and never replaced by a later one: the conversion
raises (``missing="raise"``, the default) or removes that asset's return on that day (``missing="exclude"``), and says how many days.

Observed versus expected. Converting an *observed* local return uses the rates of the day it happened. Converting a *forecast* made at the decision date needs an assumption about the exchange rate over the forecast
horizon, which is not yet known: ``random_walk`` (expected FX return 0, the default and a stated assumption) or ``forecast`` (a caller-supplied expected FX return per decision date, read only on that date). No exchange rate
after the decision date is read to convert a forecast made at it.

Costs and attribution. Converting currency costs ``fx_cost_bps`` of the foreign exposure traded: ``fx_cost_series`` computes it from the trades (applied beside the engine's own cost model, which has one rate for every asset).
``currency_attribution`` splits the portfolio's return exactly into the local-currency part and the currency part (including the cross term), by currency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class FxError(ValueError):
    """A currency conversion that cannot be done honestly, with what is missing."""


QUOTES = ("base_per_local", "local_per_base")
ASSUMPTIONS = ("random_walk", "forecast")


@dataclass
class FxAssumption:
    kind: str = "random_walk"
    forecast: pd.DataFrame | None = None        # expected FX return over the forecast horizon, decision date x currency (the value on a date is what was known on it)

    def __post_init__(self):
        if self.kind not in ASSUMPTIONS:
            raise FxError(f"the FX assumption must be one of {list(ASSUMPTIONS)}")
        if self.kind == "forecast" and self.forecast is None:
            raise FxError("an FX forecast assumption needs the expected FX return per decision date")


class FxTable:
    """Rates and when they became available, per currency, quoted as base units per unit of the currency."""

    def __init__(self, base: str = "USD", max_stale_days: int = 5):
        self.base = base.upper()
        self.max_stale_days = int(max_stale_days)
        self._rates: dict[str, pd.Series] = {}
        self._avail: dict[str, pd.Series] = {}

    def set_series(self, ccy: str, rates: pd.Series, available_at: pd.Series | None = None, quote: str = "base_per_local") -> None:
        if quote not in QUOTES:
            raise FxError(f"quote must be one of {list(QUOTES)}")
        r = pd.Series(rates).astype(float).sort_index()
        r = r[(r > 0) & np.isfinite(r)]
        if quote == "local_per_base":
            r = 1.0 / r
        av = pd.Series(r.index, index=r.index) if available_at is None else pd.Series(pd.to_datetime(available_at)).reindex(r.index)
        if av.isna().any():
            raise FxError(f"{ccy}: every rate needs an availability time")
        if (av < pd.Series(pd.to_datetime(r.index), index=r.index)).any():
            raise FxError(f"{ccy}: a rate cannot be available before the time it refers to")
        self._rates[ccy.upper()], self._avail[ccy.upper()] = r, av

    def currencies(self) -> list[str]:
        return sorted(self._rates)

    def usable(self, ccy: str, dates: pd.DatetimeIndex) -> pd.Series:
        """The rate in force on each date (base per unit of ``ccy``): the latest observation available before the date ended and not older than the stale limit; NaN otherwise."""
        ccy = ccy.upper()
        if ccy == self.base:
            return pd.Series(1.0, index=dates)
        if ccy not in self._rates:
            raise FxError(f"no exchange-rate series for {ccy} against {self.base}: a rate is not guessed")
        r, av = self._rates[ccy], self._avail[ccy]
        eff = pd.Series(r.values, index=pd.DatetimeIndex(pd.to_datetime(av.values))).sort_index()      # the observation is in force from the moment it became available
        eff = eff[~eff.index.duplicated(keep="last")]
        cutoff = pd.DatetimeIndex(dates) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)            # what was known by the end of the day
        pos = eff.index.searchsorted(cutoff, side="right") - 1
        vals = np.where(pos >= 0, eff.to_numpy()[np.clip(pos, 0, None)], np.nan)
        when = np.where(pos >= 0, eff.index.to_numpy()[np.clip(pos, 0, None)], np.datetime64("NaT"))
        age = (cutoff.to_numpy() - when) / np.timedelta64(1, "D")
        vals = np.where(np.isfinite(age) & (age <= self.max_stale_days + 1), vals, np.nan)
        return pd.Series(vals, index=dates)

    def returns(self, ccy: str, dates: pd.DatetimeIndex) -> pd.Series:
        s = self.usable(ccy, dates)
        return s / s.shift(1) - 1.0


@dataclass
class FxContext:
    table: FxTable
    currencies: dict                      # asset -> the currency it trades in
    assumption: FxAssumption = field(default_factory=FxAssumption)
    hedge: dict | None = None             # asset -> share of its currency exposure hedged (0..1), default 0
    missing: str = "raise"                # raise | exclude
    fx_cost_bps: float = 0.0

    def __post_init__(self):
        if self.missing not in ("raise", "exclude"):
            raise FxError("missing must be raise or exclude")
        self.currencies = {a: str(c).upper() for a, c in self.currencies.items()}

    def hedge_of(self, asset: str) -> float:
        k = float((self.hedge or {}).get(asset, 0.0))
        if not 0.0 <= k <= 1.0:
            raise FxError(f"the hedge ratio of {asset} must be between 0 and 1")
        return k

    def foreign(self, assets) -> list[str]:
        return [a for a in assets if self.currencies.get(a, self.table.base) != self.table.base]

    def check(self, assets) -> None:
        unknown = [a for a in assets if a not in self.currencies]
        if unknown:
            raise FxError(f"the trading currency of {', '.join(map(str, unknown))} is not declared: a currency is never assumed to be the base")
        for ccy in sorted({self.currencies[a] for a in self.foreign(assets)}):
            if ccy not in self.table.currencies():
                raise FxError(f"no exchange-rate series for {ccy} against {self.table.base}: the portfolio cannot be valued in {self.table.base} without it")


def fx_returns(ctx: FxContext, assets, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The FX return of each asset's currency on each date (0 for the base currency), NaN where a rate is missing or stale."""
    out = {}
    cache: dict = {}
    for a in assets:
        ccy = ctx.currencies[a]
        if ccy == ctx.table.base:
            out[a] = pd.Series(0.0, index=dates)
            continue
        if ccy not in cache:
            cache[ccy] = ctx.table.returns(ccy, dates)
        out[a] = cache[ccy]
    return pd.DataFrame(out, index=dates)[list(assets)]


def base_returns(local: pd.DataFrame, ctx: FxContext) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Daily returns in the base currency, ``(1 + R_local)(1 + (1 - k) R_fx) - 1``, the FX return used for each asset-day, and a report of the missing observations."""
    ctx.check(list(local.columns))
    fxr = fx_returns(ctx, list(local.columns), local.index)
    k = pd.Series({a: ctx.hedge_of(a) for a in local.columns})
    foreign = ctx.foreign(list(local.columns))
    first = {}
    missing = {}
    for a in foreign:
        s = ctx.table.usable(ctx.currencies[a], local.index)
        valid = s.dropna()
        first[a] = valid.index[0] if len(valid) else None
        gap = fxr[a].isna() & local[a].notna()
        gap.iloc[0] = False                                                      # the first row has no earlier rate to form a change from, whatever the data
        if first[a] is not None:
            gap = gap & (local.index > first[a])                                  # the first day a rate exists has no change either; days before it are reported separately below
        n_gap = int(gap.sum())
        n_pre = int((local[a].notna() & (local.index < (first[a] if first[a] is not None else local.index[-1] + pd.Timedelta(days=1)))).sum())
        if n_gap or n_pre:
            missing[a] = {"currency": ctx.currencies[a], "days_missing_after_first_rate": n_gap, "days_before_first_rate": n_pre}
    if missing and ctx.missing == "raise":
        raise FxError("exchange rates are missing: " + "; ".join(f"{a} ({m['currency']}): {m['days_missing_after_first_rate']} day(s) with no usable rate, {m['days_before_first_rate']} before the first one" for a, m in missing.items())
                      + ". Provide the rates, or set missing='exclude' to drop those asset-days; a missing rate is never taken as no change")
    eff = fxr.mul(1.0 - k.reindex(fxr.columns), axis=1)
    base = (1.0 + local) * (1.0 + eff) - 1.0                                       # NaN fx -> NaN: the asset-day is excluded, not valued at zero FX change
    return base, fxr, {"assets": missing, "policy": ctx.missing}


def convert_forecast(score_local: pd.DataFrame, ctx: FxContext, horizon: int) -> tuple[pd.DataFrame, dict]:
    """Expected base-currency return over the horizon from an expected local return at each decision date: ``(1 + mu_local)(1 + (1 - k) E[R_fx]) - 1``. ``E[R_fx]`` is the assumption (0 for a random walk, or the supplied
    forecast for that date): no exchange rate after the decision date is read."""
    ctx.check(list(score_local.columns))
    ex = pd.DataFrame(0.0, index=score_local.index, columns=score_local.columns)
    note = "random walk: the expected exchange-rate change over the horizon is zero (an assumption, not an estimate)"
    if ctx.assumption.kind == "forecast":
        f = ctx.assumption.forecast.reindex(index=score_local.index)
        for a in ctx.foreign(list(score_local.columns)):
            c = ctx.currencies[a]
            if c not in f.columns:
                raise FxError(f"the FX forecast has no column for {c}")
            ex[a] = f[c]
        note = "a supplied expected exchange-rate change per decision date, read only on that date"
    k = pd.Series({a: ctx.hedge_of(a) for a in score_local.columns})
    out = (1.0 + score_local) * (1.0 + ex.mul(1.0 - k.reindex(ex.columns), axis=1)) - 1.0
    for a in ctx.foreign(list(score_local.columns)):                                # a forecast of a foreign asset that needs a non-zero FX forecast it does not have is NaN, not local
        out[a] = out[a].where(ex[a].notna())
    return out, {"assumption": ctx.assumption.kind, "note": note, "horizon_days": int(horizon)}


def fx_exposure(weights: pd.Series, ctx: FxContext) -> dict:
    """Net unhedged exposure by currency (weights summed over the assets that trade in it, times one minus the hedge ratio), including the base currency."""
    out: dict = {}
    for a, w in weights.items():
        c = ctx.currencies[a]
        out[c] = out.get(c, 0.0) + float(w) * (1.0 - (ctx.hedge_of(a) if c != ctx.table.base else 0.0))
    return out


def fx_cost_series(trades: pd.DataFrame, ctx: FxContext) -> pd.Series:
    """Daily cost of converting currency: ``fx_cost_bps`` of the foreign (unhedged-weighted) exposure traded that day, as a fraction of capital. ``trades`` is the engine's date x asset traded weight."""
    foreign = ctx.foreign(list(trades.columns))
    if not foreign or ctx.fx_cost_bps <= 0:
        return pd.Series(0.0, index=trades.index)
    return trades[foreign].abs().sum(axis=1) * ctx.fx_cost_bps / 1e4


def currency_attribution(held: pd.DataFrame, local: pd.DataFrame, fxr: pd.DataFrame, ctx: FxContext) -> dict:
    """Exact split of the portfolio's daily base-currency return into the local-currency part ``w R_local`` and the currency part ``w (1 + R_local)(1 - k) R_fx`` (the cross term belongs to the currency), by currency.
    ``held`` is the book held through each day. Returns the daily frames and their totals; ``local + fx`` equals ``w R_base`` to rounding."""
    assets = list(held.columns)
    k = pd.Series({a: ctx.hedge_of(a) for a in assets})
    loc = held * local[assets]
    fxp = held * (1.0 + local[assets]) * fxr[assets].mul(1.0 - k, axis=1)
    by_ccy_local = loc.T.groupby(lambda a: ctx.currencies[a]).sum().T
    by_ccy_fx = fxp.T.groupby(lambda a: ctx.currencies[a]).sum().T
    return {"local": loc.sum(axis=1), "fx": fxp.sum(axis=1), "by_currency_local": by_ccy_local, "by_currency_fx": by_ccy_fx,
            "total_local": float(loc.sum().sum()), "total_fx": float(fxp.sum().sum())}


def prepare(score: pd.DataFrame, returns: pd.DataFrame, ctx: FxContext, horizon: int, kind: str = "return", prices: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict]:
    """Everything the allocation reads, in the base currency: the forecasts (expected local return -> expected base return under the FX assumption), the observed returns, the asset-days left out where a rate is
    missing (``missing='exclude'``), and a report. ``kind`` is the forecast kind: a price forecast is turned into a local return first; a directional probability is refused (its payoff model would have to be
    estimated in the base currency)."""
    if kind == "probability":
        raise FxError("a directional probability cannot be converted to the base currency: its payoff model is estimated from base-currency outcomes only when the model's own target is in that currency; train on a return target")
    local_forecast = score
    if kind == "price":
        if prices is None:
            raise FxError("a price forecast needs the decision-date prices to become a return before it is converted")
        p = prices.reindex(index=score.index, columns=score.columns)
        local_forecast = score / p.where(p > 0) - 1.0
    base_ret, fxr, miss = base_returns(returns, ctx)
    base_score, conv = convert_forecast(local_forecast, ctx, horizon)
    exclude = (returns.notna() & base_ret.isna())
    inv = None if not exclude.any().any() else ~exclude                               # an asset-day with a local return but no usable rate is not investable
    report = {"base": ctx.table.base, "currencies": dict(ctx.currencies), "foreign_assets": ctx.foreign(list(returns.columns)), "hedge": {a: ctx.hedge_of(a) for a in returns.columns if ctx.currencies[a] != ctx.table.base},
              "quote": "base_per_local (base units per one unit of the instrument's currency)", "identity": "1 + R_base = (1 + R_local)(1 + (1 - k) R_fx)", "forecast_conversion": conv, "missing": miss, "fx_cost_bps": ctx.fx_cost_bps,
              "excluded_asset_days": int(exclude.sum().sum())}
    return base_score, base_ret, inv, report
