"""Portfolio risk at mixed-asset level: exposures, volatility, VaR and CVaR, Greeks, DV01, scenarios, factor exposure, funding and rollover risk, tail contributions.

:class:`PortfolioRisk` turns the engine's current positions into one :class:`RiskReport`:

* **exposures**: gross and net notional in the base currency, by position, asset class, currency and underlying; options are also shown delta-adjusted;
* **Greeks**: per option (from the implied volatility of its mark) delta, gamma, vega and theta, summed per underlying in base-currency terms (delta in currency, gamma per 1% move,
  vega per volatility point); **rates risk**: PV01 and key-rate PV01 of swaps from full repricing under bumped curves;
* **volatility, VaR and CVaR**: historical simulation of the current positions over the last ``lookback`` days of point-in-time returns (options through delta and gamma, swaps through
  key-rate PV01 times the historical changes of their curves), and a parametric volatility from the same P&L series. Tail risk contributions are the Euler allocation of the expected
  shortfall: each position's average P&L on the portfolio's worst days;
* **factor exposure**: beta-weighted net exposure to factors, from supplied betas or the regression of each position's returns on a benchmark;
* **scenarios**: P&L of named shocks (instrument or asset-class returns, a parallel curve shift in basis points, a volatility shift), by FULL REPRICING of options and swaps;
* **funding and rollover risk**: the daily funding bill of perpetual positions at the latest rate, and the notional of futures positions expiring within ``rollover_days``.

All inputs are point-in-time through the engine's data access; nothing here sees the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..instruments.crypto import CryptoPerp
from ..instruments.futures import Future
from ..instruments.options import Option

DEFAULT_SCENARIOS = {
    "equity -10%": {"returns": {"equity": -0.10, "option": 0.0}, "vol": 0.30},
    "rates +100bp": {"curve_bp": 100.0},
    "rates -100bp": {"curve_bp": -100.0},
    "usd +10%": {"fx_base": ("USD", 0.10)},
    "crypto -30%": {"returns": {"crypto": -0.30}},
    "vol shock +50%": {"vol": 0.50},
}


@dataclass
class RiskReport:
    ts: pd.Timestamp
    equity: float
    gross: float
    net: float
    leverage: float
    positions: pd.DataFrame
    by_asset_class: pd.DataFrame
    by_currency: pd.Series
    greeks: pd.DataFrame
    pv01: float
    key_rate_pv01: pd.Series
    vol_annual: float
    var: float
    cvar: float
    var_alpha: float
    tail_contribution: pd.Series
    factor_exposure: dict
    scenarios: pd.Series
    funding_per_day: float
    rollover: pd.DataFrame
    notes: list = field(default_factory=list)

    def summary(self) -> pd.Series:
        return pd.Series({"equity": self.equity, "gross": self.gross, "net": self.net, "leverage": self.leverage, "vol_annual": self.vol_annual, f"var_{self.var_alpha:.0%}": self.var,
                          f"cvar_{self.var_alpha:.0%}": self.cvar, "pv01": self.pv01, "funding_per_day": self.funding_per_day})


class PortfolioRisk:
    def __init__(self, lookback: int = 252, alpha: float = 0.99, factor_betas: dict | None = None, benchmark: str | None = None, scenarios: dict | None = None, rollover_days: int = 10,
                 min_history: int = 30):
        self.lookback, self.alpha, self.factor_betas, self.benchmark = lookback, alpha, factor_betas or {}, benchmark
        self.scenarios = DEFAULT_SCENARIOS if scenarios is None else scenarios
        self.rollover_days, self.min_history = rollover_days, min_history

    # ------------------------------------------------------------------------------------------------------------------------- building blocks
    def _under_price(self, engine, inst):
        from .lifecycle import LifecycleError, underlying_mark

        try:
            return underlying_mark(engine, inst, engine.clock.now)
        except (LifecycleError, KeyError):
            return float("nan")

    def _option_greeks(self, engine, inst: Option, qty: float, mark: float, ts) -> dict | None:
        from ..derivatives.iv import implied_vol

        S = self._under_price(engine, inst)
        T = inst.time_to_expiry(ts)
        if not (np.isfinite(S) and T > 0 and mark > 0):
            return None
        if inst.underlying_kind == "future":
            iv = self._iv_black76(mark, S, inst, T)
        else:
            iv = float(np.asarray(implied_vol(mark, S, inst.strike, T, 0.0, 0.0, inst.is_call)).ravel()[0])
        if not np.isfinite(iv):
            return None
        g = inst.greeks(S, iv, ts)
        m = inst.contract_multiplier * qty * engine.ledger.rate(inst.currency)
        return {"S": S, "iv": iv, "delta_base": g["delta"] * S * m, "gamma_base": g["gamma"] * S * S * 0.01 * m, "vega_base": g["vega"] * 0.01 * m, "theta_base": g["theta"] / 365.0 * m,
                "delta_units": g["delta"] * inst.contract_multiplier * qty}

    @staticmethod
    def _iv_black76(price, F, inst, T):
        from scipy import optimize

        from ..derivatives.pricing import black76_price
        f = lambda s: float(np.asarray(black76_price(F, inst.strike, T, 0.0, s, inst.is_call)).ravel()[0]) - price      # noqa: E731
        try:
            return float(optimize.brentq(f, 1e-4, 5.0))
        except ValueError:
            return float("nan")

    def _swap_curves(self, engine, inst, ts):
        from ..swaps.integration import _curveset

        cs, _ = _curveset(engine.marks, inst, ts)
        return cs

    # -------------------------------------------------------------------------------------------------------------------------------- report
    def report(self, engine) -> RiskReport:
        led, ts = engine.ledger, engine.clock.now
        marks = engine.current_marks
        expo = led.exposures(marks)
        equity = led.equity()
        notes: list[str] = []
        rows, greek_rows = {}, []
        pnl_cols: dict[str, pd.Series] = {}
        pv01_total, krp = 0.0, pd.Series(dtype=float)
        funding = 0.0
        rollover = []
        base = engine.config.base_currency
        for iid, pos in led.positions.items():
            inst = engine.registry.get(iid)
            mark = marks.get(iid, pos.last_mark)
            row = {"asset_class": inst.asset_class, "type": inst.instrument_type, "currency": inst.currency, "quantity": pos.quantity, "mark": mark,
                   "notional_base": float(expo.loc[iid, "notional_base"]) if iid in expo.index else np.nan,
                   "signed_notional_base": float(expo.loc[iid, "signed_notional_base"]) if iid in expo.index else np.nan}
            delta_base = row["signed_notional_base"]
            if isinstance(inst, Option):
                g = self._option_greeks(engine, inst, pos.quantity, mark, ts)
                if g is not None:
                    delta_base = g["delta_base"]
                    greek_rows.append({"instrument_id": iid, "underlying": inst.underlying_id, **{k: g[k] for k in ("S", "delta_base", "gamma_base", "vega_base", "theta_base", "iv")}})
                else:
                    notes.append(f"no Greeks for {iid}")
            row["delta_base"] = delta_base
            rows[iid] = row
            if inst.instrument_type in ("irs", "basis_swap", "xccy_basis_swap"):
                from ..swaps.risk import key_rate_pv01, pv01

                cs = self._swap_curves(engine, inst, ts)
                if cs is not None:
                    val = max(c.valuation for c in cs.discount.values())
                    p = pv01(inst, cs, val) * pos.quantity * led.rate(inst.currency)
                    pv01_total += p
                    k = key_rate_pv01(inst, cs, val) * pos.quantity * led.rate(inst.currency)
                    krp = krp.add(k, fill_value=0.0)
                    row["pv01_base"] = p
            if isinstance(inst, CryptoPerp):
                rate = engine.data.funding(iid)
                if np.isfinite(rate) and np.isfinite(mark):
                    funding += inst.funding_payment(pos.quantity, mark, rate) * (24.0 / inst.funding_interval_hours) * led.rate(inst.currency)
            if isinstance(inst, Future) and (inst.expiry - ts).days <= self.rollover_days:
                rollover.append({"instrument_id": iid, "expiry": inst.expiry, "days": (inst.expiry - ts).days, "notional_base": row["notional_base"]})
            r = engine.data.returns(inst.underlying_id if isinstance(inst, Option) and inst.underlying_id in engine.registry else iid, self.lookback)
            if len(r) >= self.min_history:
                pnl_cols[iid] = r * delta_base
                if isinstance(inst, Option) and greek_rows and greek_rows[-1]["instrument_id"] == iid:
                    pnl_cols[iid] = pnl_cols[iid] + 0.5 * greek_rows[-1]["gamma_base"] * 100.0 * r ** 2
            elif inst.instrument_type not in ("irs", "basis_swap", "xccy_basis_swap"):
                notes.append(f"{iid}: fewer than {self.min_history} returns, excluded from VaR")
        positions = pd.DataFrame.from_dict(rows, orient="index")
        # rates P&L from key-rate PV01 and historical curve changes
        if len(krp):
            rate_pnl = self._rate_pnl_series(engine, krp)
            if rate_pnl is not None:
                pnl_cols["rates"] = rate_pnl
        pnl = pd.DataFrame(pnl_cols).dropna(how="all").fillna(0.0)
        total = pnl.sum(axis=1) if len(pnl) else pd.Series(dtype=float)
        if len(total) >= self.min_history:
            q = float(np.quantile(total, 1 - self.alpha))
            var = -q
            tail = total <= q
            cvar = float(-total[tail].mean()) if tail.any() else var
            contrib = -pnl[tail].mean() if tail.any() else pd.Series(0.0, index=pnl.columns)
            vol = float(total.std(ddof=1) * np.sqrt(252.0))
        else:
            var = cvar = vol = float("nan")
            contrib = pd.Series(dtype=float)
            notes.append("not enough history for VaR")
        by_class = positions.groupby("asset_class").agg(gross=("notional_base", "sum"), net=("signed_notional_base", "sum"), delta=("delta_base", "sum")) if len(positions) else pd.DataFrame()
        by_ccy: dict[str, float] = {}
        for iid, row in rows.items():
            inst = engine.registry.get(iid)
            n = row["signed_notional_base"]
            if hasattr(inst, "base_currency") and inst.cash_style == "currency_exchange":
                by_ccy[inst.base_currency] = by_ccy.get(inst.base_currency, 0.0) + n
                by_ccy[inst.quote_currency] = by_ccy.get(inst.quote_currency, 0.0) - n
            else:
                by_ccy[inst.currency] = by_ccy.get(inst.currency, 0.0) + n
        greeks = pd.DataFrame(greek_rows)
        greeks_u = greeks.groupby("underlying")[["delta_base", "gamma_base", "vega_base", "theta_base"]].sum() if len(greeks) else pd.DataFrame()
        gross = float(positions["notional_base"].sum()) if len(positions) else 0.0
        net = float(positions["signed_notional_base"].sum()) if len(positions) else 0.0
        factors = self._factors(engine, positions)
        scen = self._scenarios(engine, positions, greek_rows, ts)
        return RiskReport(ts, equity, gross, net, gross / equity if equity else float("nan"), positions, by_class, pd.Series(by_ccy, dtype=float), greeks_u, pv01_total, krp, vol, var, cvar,
                          self.alpha, contrib, factors, scen, funding, pd.DataFrame(rollover), notes)

    # ---------------------------------------------------------------------------------------------------------------------------- pieces
    def _rate_pnl_series(self, engine, krp: pd.Series):
        hist = {}
        for cid in {c for c in (f"{engine.config.base_currency}-DISCOUNT",)}:
            o = engine.store._groups.get((cid, "curve"))
            if o is None:
                continue
            k = int(np.searchsorted(o.avail, engine.clock.now.value, side="right"))
            rows = o.rows[max(0, k - self.lookback - 1):k]
            curves = [engine.store._cols["curve_values"][r] for r in rows]
            times = [pd.Timestamp(engine.store._cols["timestamp"][r]) for r in rows]
            if len(curves) < 5:
                continue
            df = pd.DataFrame(curves, index=times).sort_index()
            hist[cid] = df
        if not hist:
            return None
        df = next(iter(hist.values()))
        change_bp = df.diff().dropna() * 1e4
        nodes = [t for t in krp.index if t in change_bp.columns]
        if not nodes:
            return None
        return (change_bp[nodes] * krp[nodes]).sum(axis=1)

    def _factors(self, engine, positions: pd.DataFrame) -> dict:
        if positions.empty:
            return {}
        out: dict[str, float] = {}
        for iid, row in positions.iterrows():
            for f, b in self.factor_betas.get(iid, {}).items():
                out[f] = out.get(f, 0.0) + b * row["delta_base"]
        if self.benchmark and self.benchmark in engine.registry:
            bench = engine.data.returns(self.benchmark, self.lookback)
            beta_sum = 0.0
            for iid, row in positions.iterrows():
                r = engine.data.returns(iid, self.lookback)
                j = pd.concat([r, bench], axis=1).dropna()
                if len(j) >= self.min_history and j.iloc[:, 1].var() > 0:
                    beta_sum += float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()) * row["delta_base"]
            out[f"beta to {self.benchmark}"] = beta_sum
        return out

    def _scenarios(self, engine, positions: pd.DataFrame, greek_rows: list, ts) -> pd.Series:
        led = engine.ledger
        out = {}
        greek = {g["instrument_id"]: g for g in greek_rows}
        for name, sc in self.scenarios.items():
            total = 0.0
            for iid, row in positions.iterrows():
                inst = engine.registry.get(iid)
                ret = sc.get("returns", {}).get(iid, sc.get("returns", {}).get(inst.asset_class, 0.0))
                if "fx_base" in sc and hasattr(inst, "base_currency") and inst.cash_style == "currency_exchange":
                    ccy, mv = sc["fx_base"]
                    ret = -mv if inst.base_currency == ccy else (mv if inst.quote_currency == ccy else 0.0)
                if isinstance(inst, Option) and iid in greek:
                    g = greek[iid]
                    total += self._reprice_option(engine, inst, positions.loc[iid], g, ret, sc.get("vol", 0.0), ts)
                    continue
                if inst.instrument_type in ("irs", "basis_swap", "xccy_basis_swap") and "curve_bp" in sc:
                    from ..swaps.pricing import price

                    cs = self._swap_curves(engine, inst, ts)
                    if cs is not None:
                        val = max(c.valuation for c in cs.discount.values())
                        total += (price(inst, cs.bumped(sc["curve_bp"]), val).pv - price(inst, cs, val).pv) * row["quantity"] * led.rate(inst.currency)
                    continue
                if np.isfinite(row["delta_base"]):
                    total += row["delta_base"] * ret
            out[name] = total
        return pd.Series(out)

    def _reprice_option(self, engine, inst: Option, row, g: dict, ret: float, vol_shift: float, ts) -> float:
        S0, iv = g["S"], g["iv"]
        p0 = inst.model_price(S0, iv, ts)
        p1 = inst.model_price(S0 * (1.0 + ret), iv * (1.0 + vol_shift), ts)
        return (p1 - p0) * inst.contract_multiplier * row["quantity"] * engine.ledger.rate(inst.currency)
