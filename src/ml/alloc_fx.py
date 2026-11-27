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
    fx_cost_bps: float = 0.0              # cost of converting currency on every trade of a foreign asset, bps of the notional traded (a purchase converts into the currency, a sale back)
    hedge_cost_bps: float = 0.0           # cost of changing the notional of the forward hedge, bps of the notional changed

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


def base_returns(local: pd.DataFrame, ctx: FxContext, apply_hedge: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Daily returns in the base currency, the FX return used for each asset-day, and a report of the missing observations.

    ``apply_hedge=True`` (the allocator's risk estimation): ``(1 + R_local)(1 + (1 - k) R_fx) - 1``, the return of a holding whose currency exposure is hedged to the ratio ``k``: a device for estimating the risk
    of the hedged book, not a P&L. ``apply_hedge=False`` (what the backtest values the book with): the unhedged ``(1 + R_local)(1 + R_fx) - 1``; the hedge is then an explicit overlay of forward positions
    (``hedge_overlay``) whose P&L and costs are computed from notionals."""
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
    eff = fxr.mul(1.0 - k.reindex(fxr.columns), axis=1) if apply_hedge else fxr
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


# ------------------------------------------------------------------------------------------------------------ configuration, costs, explicit hedge positions
SPEC_KEYS = {"base", "currencies", "hedge", "cost_bps", "hedge_cost_bps", "missing", "assumption", "available_lag_days", "max_stale_days"}


def check_spec(spec: dict) -> dict:
    """Validate a ``backtest.fx`` section and return it with its defaults filled (the base currency is explicit; nothing about an asset's currency is assumed)."""
    if not isinstance(spec, dict):
        raise FxError("fx must be an object")
    extra = set(spec) - SPEC_KEYS
    if extra:
        raise FxError(f"fx has no setting {sorted(extra)}; the settings are {sorted(SPEC_KEYS)}")
    base = str(spec.get("base") or "").upper()
    if len(base) != 3 or not base.isalpha():
        raise FxError("fx.base is the base (reporting) currency as a three-letter code, for example USD")
    ccy = spec.get("currencies")
    if not isinstance(ccy, dict) or not ccy:
        raise FxError("fx.currencies maps every ticker to the currency it trades in, for example {\"NVDA\": \"USD\", \"SAP.DE\": \"EUR\"}: a currency is never assumed")
    ccy = {str(a).upper(): str(c).upper() for a, c in ccy.items()}
    if any(len(c) != 3 or not c.isalpha() for c in ccy.values()):
        raise FxError("every currency is a three-letter code")
    hedge = spec.get("hedge", 0.0)
    if isinstance(hedge, dict):
        hedge = {str(a).upper(): float(v) for a, v in hedge.items()}
        ok = all(0.0 <= v <= 1.0 for v in hedge.values())
    else:
        hedge = float(hedge or 0.0)
        ok = 0.0 <= hedge <= 1.0
    if not ok:
        raise FxError("a hedge ratio is between 0 and 1")
    out = {"base": base, "currencies": ccy, "hedge": hedge, "cost_bps": float(spec.get("cost_bps", 0.0)), "hedge_cost_bps": float(spec.get("hedge_cost_bps", 0.0)), "missing": str(spec.get("missing", "raise")),
           "assumption": str(spec.get("assumption", "random_walk")), "available_lag_days": int(spec.get("available_lag_days", 0)), "max_stale_days": int(spec.get("max_stale_days", 5))}
    if out["cost_bps"] < 0 or out["hedge_cost_bps"] < 0 or out["available_lag_days"] < 0 or out["max_stale_days"] < 0:
        raise FxError("costs, the availability lag and the stale limit are not negative")
    if out["missing"] not in ("raise", "exclude"):
        raise FxError("fx.missing is raise or exclude")
    if out["assumption"] != "random_walk":
        raise FxError("fx.assumption: a forecast is converted with a random-walk exchange rate (expected FX change 0); a supplied FX forecast is available through the Python interface only")
    return out


def context_from_spec(spec: dict, table: FxTable) -> FxContext:
    s = check_spec(spec)
    hedge = s["hedge"] if isinstance(s["hedge"], dict) else {a: s["hedge"] for a, c in s["currencies"].items() if c != s["base"]}
    return FxContext(table, s["currencies"], FxAssumption(s["assumption"]), hedge, s["missing"], s["cost_bps"], s["hedge_cost_bps"])


def per_asset_bps(ctx: FxContext, base_bps: float, assets) -> dict:
    """The one-way cost of each asset for the backtest's linear cost model: the trading cost, plus the currency-conversion cost on every foreign asset (a purchase converts base into the currency and a sale back)."""
    foreign = set(ctx.foreign(list(assets)))
    return {a: float(base_bps) + (ctx.fx_cost_bps if a in foreign else 0.0) for a in assets}


def hedge_overlay(held: pd.DataFrame, trades: pd.DataFrame, fxr: pd.DataFrame, ctx: FxContext) -> dict:
    """Explicit forward hedge positions on the book's currency exposure, with their own cash flows.

    At each close the notional hedged in currency ``c`` is ``sum_i k_i w_i`` over the assets trading in ``c`` (``w`` the book held at that close, ``k`` the hedge ratio): it is struck on days the book traded and
    left fixed in between (the exposure then drifts away from the hedge, and the residual is reported). The overlay sells ``c`` forward: it earns ``-N_{t-1} R_fx,t`` on day ``t`` (settled daily, as a futures-style
    mark: no discounting) and pays ``hedge_cost_bps`` of the notional changed. Forward points (the interest-rate differential) are NOT modelled: no financing rates are available to the platform, so the carry of
    the hedge is zero here and the overlay is not exchange-accurate; the report says so. Returns the daily P&L, cost, notionals by currency and the residual unhedged exposure."""
    assets = list(held.columns)
    foreign = ctx.foreign(assets)
    ccys = sorted({ctx.currencies[a] for a in foreign})
    if not foreign or not any(ctx.hedge_of(a) > 0 for a in foreign):
        z = pd.Series(0.0, index=held.index)
        return {"pnl": z, "cost": z, "notional": pd.DataFrame(0.0, index=held.index, columns=ccys), "residual_exposure": pd.DataFrame(0.0, index=held.index, columns=ccys), "active": False}
    k = pd.Series({a: ctx.hedge_of(a) for a in foreign})
    exposure = pd.DataFrame({c: held[[a for a in foreign if ctx.currencies[a] == c]].mul(k[[a for a in foreign if ctx.currencies[a] == c]], axis=1).sum(axis=1) for c in ccys}, index=held.index)
    traded = trades.reindex(index=held.index, columns=assets).abs().sum(axis=1) > 1e-12
    notional = exposure.where(traded).ffill().fillna(0.0)
    notional.iloc[0] = exposure.iloc[0] if bool(traded.iloc[0]) else 0.0
    fx_by_ccy = pd.DataFrame({c: fxr[[a for a in foreign if ctx.currencies[a] == c][0]] for c in ccys}, index=held.index).fillna(0.0)
    pnl = -(notional.shift(1).fillna(0.0) * fx_by_ccy).sum(axis=1)
    cost = notional.diff().abs().fillna(notional.abs()).sum(axis=1) * ctx.hedge_cost_bps / 1e4
    total_exposure = pd.DataFrame({c: held[[a for a in foreign if ctx.currencies[a] == c]].sum(axis=1) for c in ccys}, index=held.index)
    return {"pnl": pnl, "cost": cost, "notional": notional, "residual_exposure": total_exposure - notional, "active": True}


def coverage(table: FxTable, assets_ccy: dict, dates: pd.DatetimeIndex) -> dict:
    """Missing-data diagnostics per currency: how much of the asset calendar has a usable rate, the longest run without one, and the first and last usable date."""
    out = {}
    for c in sorted({c for c in assets_ccy.values() if c != table.base}):
        u = table.usable(c, dates)
        ok = u.notna()
        run = longest = 0
        for v in ok.to_numpy():
            run = 0 if v else run + 1
            longest = max(longest, run)
        out[c] = {"days": int(len(dates)), "usable": int(ok.sum()), "coverage": float(ok.mean()) if len(dates) else None, "longest_gap_days": int(longest),
                  "first": None if not ok.any() else str(ok[ok].index[0].date()), "last": None if not ok.any() else str(ok[ok].index[-1].date())}
    return out


def report_block(ctx: FxContext, local_returns: pd.DataFrame, fx_ret: pd.DataFrame, held: pd.DataFrame, trades: pd.DataFrame, overlay: dict, missing: dict, provenance: dict | None) -> dict:
    """Everything the page shows about currency for one run: the assumptions, exposure and its hedge, the split of return into local and currency parts, the costs, and the data diagnostics."""
    assets = list(held.columns)
    att = currency_attribution(held.shift(1).fillna(0.0), local_returns, fx_ret, ctx)
    ccys = sorted({ctx.currencies[a] for a in assets})
    exposure = pd.DataFrame({c: held[[a for a in assets if ctx.currencies[a] == c]].sum(axis=1) for c in ccys}, index=held.index)
    step = max(1, len(held) // 300)
    idx = held.index[::step]
    conv_cost = float(fx_cost_series(trades.reindex(columns=assets).fillna(0.0), ctx).sum())
    hedge_pnl, hedge_cost = float(overlay["pnl"].sum()), float(overlay["cost"].sum())
    reasons = ["forward points and financing rates are not available: the hedge earns no carry in this model", "rates are daily closes, not executable quotes"]
    return {"base": ctx.table.base, "currencies": dict(ctx.currencies), "foreign_assets": ctx.foreign(assets), "hedge_ratio": {a: ctx.hedge_of(a) for a in ctx.foreign(assets)},
            "quote": "base currency per one unit of the instrument's currency", "identity": "1 + R_base = (1 + R_local)(1 + R_fx); hedged overlay P&L = -(notional hedged) x R_fx, settled daily",
            "forecast_conversion": {"assumption": ctx.assumption.kind, "note": "the expected exchange-rate change over the forecast horizon is zero (a random walk): an assumption, not an estimate"},
            "exposure": {"latest": {c: float(v) for c, v in exposure.iloc[-1].items()}, "mean": {c: float(v) for c, v in exposure.mean().items()},
                         "series": {"dates": [str(d.date()) for d in idx], **{c: [float(v) for v in exposure[c].reindex(idx)] for c in ccys}}},
            "hedge": {"active": bool(overlay["active"]), "notional_latest": {c: float(v) for c, v in overlay["notional"].iloc[-1].items()}, "residual_exposure_latest": {c: float(v) for c, v in overlay["residual_exposure"].iloc[-1].items()},
                      "pnl_total": hedge_pnl, "cost_total": hedge_cost, "carry": "not modelled", "instrument": "forward positions struck on days the book traded, settled daily"},
            "attribution": {"local_return_total": att["total_local"], "currency_return_total": att["total_fx"],
                            "by_currency_local": {c: float(v) for c, v in att["by_currency_local"].sum().items()}, "by_currency_fx": {c: float(v) for c, v in att["by_currency_fx"].sum().items()},
                            "note": "sums of daily contributions (not compounded); the currency part includes the local-times-currency cross term"},
            "costs": {"conversion_cost_total": conv_cost, "conversion_cost_bps": ctx.fx_cost_bps, "hedge_cost_total": hedge_cost, "hedge_cost_bps": ctx.hedge_cost_bps,
                      "note": "the conversion cost is inside the engine's costs (a per-asset rate); the hedge cost and P&L are added to the net result beside it"},
            "data": {"missing": missing, "coverage": coverage(ctx.table, ctx.currencies, local_returns.index), "series": (provenance or {}).get("series", {}), "assumptions": (provenance or {}).get("assumptions", [])},
            "exchange_accurate": False, "not_exchange_accurate_because": reasons}
