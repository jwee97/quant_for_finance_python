"""An options backtester: positions in contracts, daily marks, realistic fills, delta hedging, expiry settlement and Greek P&L attribution.

The engine replays a dictionary ``{date: chain}`` (see ``schema``) together with the underlying's closing price and asks a strategy, once per date, what trades it wants.

Conventions that matter for honesty:

* **No look-ahead.** A strategy is given only the chain, the spot and the position as of the close of date ``t`` plus the underlying history up to ``t``. Orders are filled at the
  quotes of date ``t + execution_lag`` (default 1), so a decision made on today's close never trades at today's price.
* **Fills cross the spread.** A buy fills at ``mid + fill_fraction * half_spread`` and a sell at ``mid - fill_fraction * half_spread`` (``fill_fraction = 1`` is a full cross). An
  order for a contract not quoted on the execution date is skipped and recorded.
* **Costs.** Commission per contract; the underlying hedge pays ``hedge_cost_bps`` of the traded notional.
* **Marks and expiry.** Positions are marked at the mid. At expiry a contract is cash-settled at intrinsic value against the underlying close (European style); contracts still open
  on the last date stay marked to market. If a held contract has no quote on a date it is marked with Black-Scholes at its last implied volatility.
* **Returns** are daily P&L divided by ``capital`` (a fixed base: margin and funding are NOT modelled, so choose ``capital`` to be what the strategy would actually tie up, e.g.
  the strike for a cash-secured put). Cash earns nothing, which understates the short-premium strategies' interest income and does not penalise the long-underlying ones for funding.
* **Attribution.** Each day's P&L on positions held overnight is split with the previous close's Greeks into delta, gamma, vega, theta and an unexplained remainder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .greeks import bsm_greeks
from .pricing import bsm_price


@dataclass(frozen=True)
class Order:
    """Buy (positive) or sell (negative) ``quantity`` contracts of ``(expiry, strike, right)``."""

    expiry: pd.Timestamp
    strike: float
    right: str
    quantity: float

    @property
    def key(self) -> tuple:
        return (pd.Timestamp(self.expiry), float(self.strike), self.right)


@dataclass(frozen=True)
class Hedge:
    """Set the underlying position to ``target_units`` shares (a delta hedge or a covered-call stock leg)."""

    target_units: float


@dataclass
class StrategyContext:
    date: pd.Timestamp
    chain: pd.DataFrame
    spot: float
    history: pd.Series                 # underlying close up to and including ``date``
    positions: dict                    # key -> signed contracts
    underlying_units: float
    equity: float
    capital: float
    greeks: dict                       # net position Greeks in underlying units: delta, gamma, vega, theta


class OptionStrategy:
    name = "strategy"

    def on_date(self, ctx: StrategyContext) -> list:
        raise NotImplementedError


@dataclass
class OptionBacktestResult:
    equity: pd.Series
    pnl: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame            # per date: number of open contracts, underlying units
    greeks: pd.DataFrame               # net Greeks at each close (units of the underlying; vega per 1.00, theta per year)
    attribution: pd.DataFrame          # delta, gamma, vega, theta, hedge, unexplained, costs
    costs: pd.Series
    skipped_orders: int = 0

    def summary(self, periods_per_year: int = 252) -> pd.Series:
        r = self.returns.dropna()
        sd = r.std(ddof=1)
        eq = (1 + r).cumprod()
        return pd.Series({"total_return": float(eq.iloc[-1] - 1), "annual_return": float(r.mean() * periods_per_year), "annual_vol": float(sd * np.sqrt(periods_per_year)),
                          "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"), "max_drawdown": float((eq / eq.cummax() - 1).min()),
                          "worst_day": float(r.min()), "costs_total": float(self.costs.sum()), "n_trades": int(len(self.trades))})


def _quote_lookup(chain: pd.DataFrame) -> dict:
    cols = ["bid", "ask", "mid", "iv", "T", "underlying", "rate", "dividend"] if "iv" in chain else ["bid", "ask", "mid", "T", "underlying", "rate", "dividend"]
    sub = chain.set_index(["expiry", "strike", "right"])[cols]
    return {(e, float(k), r): row for (e, k, r), row in zip(sub.index, sub.to_dict("records"))}


@dataclass
class _Book:
    """The mutable state of a backtest: cash, the underlying position, option positions, last implied vols and the trade log."""

    cash: float = 0.0
    units: float = 0.0
    positions: dict = field(default_factory=dict)
    last_iv: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    skipped: int = 0


class OptionBacktester:
    def __init__(self, chains: dict, spot: pd.Series, capital: float, multiplier: float = 100.0, commission: float = 0.65, hedge_cost_bps: float = 1.0,
                 fill_fraction: float = 1.0, execution_lag: int = 1):
        if capital <= 0 or multiplier <= 0 or not 0.0 <= fill_fraction <= 1.0 or execution_lag < 0:
            raise ValueError("capital and multiplier must be positive, fill_fraction in [0, 1], execution_lag >= 0")
        self.chains = {pd.Timestamp(k): v for k, v in chains.items()}
        self.dates = sorted(self.chains)
        self.spot = spot.reindex(self.dates)
        self.capital, self.mult, self.commission, self.hedge_bps = capital, multiplier, commission, hedge_cost_bps
        self.fill_fraction, self.lag = fill_fraction, execution_lag
        self._quotes = {d: _quote_lookup(c) for d, c in self.chains.items()}

    # ------------------------------------------------------------------------------------------------------------------ helpers
    def _price_position(self, d, key, last_iv) -> float:
        q = self._quotes[d].get(key)
        if q is not None:
            return float(q["mid"])
        iv = last_iv.get(key)
        if iv is None:
            return 0.0
        expiry, strike, right = key
        T = max((expiry - d).days / 365.0, 0.0)
        ref = next(iter(self._quotes[d].values()))
        return float(bsm_price(self.spot[d], strike, T, ref["rate"], ref["dividend"], iv, right == "C"))

    def _equity(self, d, book: _Book) -> float:
        return book.cash + book.units * float(self.spot[d]) + sum(qty * self.mult * self._price_position(d, key, book.last_iv) for key, qty in book.positions.items())

    def _greeks(self, d, book: _Book) -> tuple[dict, dict]:
        totals = {"delta": book.units, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        per_contract = {}
        for key, qty in book.positions.items():
            q = self._quotes[d].get(key)
            iv = float(q["iv"]) if q is not None and "iv" in q else book.last_iv.get(key)
            if iv is None or not np.isfinite(iv):
                continue
            expiry, strike, right = key
            T = max((expiry - d).days / 365.0, 1e-6)
            ref = q if q is not None else next(iter(self._quotes[d].values()))
            g = bsm_greeks(self.spot[d], strike, T, ref["rate"], ref["dividend"], iv, right == "C")
            per_contract[key] = {k: float(v) for k, v in g.items()} | {"iv": iv}
            for name in totals:
                totals[name] += qty * self.mult * float(g[name])
        return totals, per_contract

    def _settle(self, d, book: _Book) -> None:
        spot = float(self.spot[d])
        for key in [k for k in book.positions if k[0] <= d]:
            qty = book.positions.pop(key)
            intrinsic = max(spot - key[1], 0.0) if key[2] == "C" else max(key[1] - spot, 0.0)
            book.cash += qty * self.mult * intrinsic
            book.last_iv.pop(key, None)

    def _execute(self, d, orders, book: _Book) -> tuple[float, float]:
        """Fill ``orders`` at the quotes of date ``d``; returns ``(fees paid, slippage paid)``. Slippage is the cost of crossing the spread: the fill price against the mid
        the position is then marked at."""
        spot = float(self.spot[d])
        paid = slip = 0.0
        for o in orders:
            if isinstance(o, Hedge):
                change = o.target_units - book.units
                if abs(change) > 1e-9:
                    fee = abs(change) * spot * self.hedge_bps / 1e4
                    book.cash -= change * spot + fee
                    book.units = o.target_units
                    paid += fee
                    book.trades.append({"date": d, "kind": "hedge", "expiry": pd.NaT, "strike": np.nan, "right": "", "quantity": change, "price": spot, "cost": fee})
                continue
            q = self._quotes[d].get(o.key)
            if q is None or o.quantity == 0:
                book.skipped += 1
                continue
            half = 0.5 * (q["ask"] - q["bid"])
            price = q["mid"] + np.sign(o.quantity) * self.fill_fraction * half
            fee = abs(o.quantity) * self.commission
            book.cash -= o.quantity * self.mult * price + fee
            paid += fee
            slip += o.quantity * self.mult * (price - q["mid"])
            book.positions[o.key] = book.positions.get(o.key, 0.0) + o.quantity
            if abs(book.positions[o.key]) < 1e-12:
                book.positions.pop(o.key)
            if "iv" in q and np.isfinite(q["iv"]):
                book.last_iv[o.key] = float(q["iv"])
            book.trades.append({"date": d, "kind": "option", "expiry": o.expiry, "strike": o.strike, "right": o.right, "quantity": o.quantity, "price": price, "cost": fee})
        return paid, slip

    # ---------------------------------------------------------------------------------------------------------------------- run
    def run(self, strategy: OptionStrategy) -> OptionBacktestResult:
        book = _Book()
        pending: list = []                                  # (index at which the batch executes, orders)
        equity_hist, costs, rows_pos, rows_greeks, rows_attr = {}, {}, [], [], []
        prev_equity, prev_state = 0.0, None
        for i, d in enumerate(self.dates):
            spot = float(self.spot[d])
            self._settle(d, book)
            day_cost = day_slip = 0.0
            due = [o for (t, o) in pending if t <= i]
            pending = [(t, o) for (t, o) in pending if t > i]
            for batch in due:
                fee, slip = self._execute(d, batch, book)
                day_cost, day_slip = day_cost + fee, day_slip + slip
            for key in list(book.positions):                # refresh last implied vols from today's quotes
                q = self._quotes[d].get(key)
                if q is not None and "iv" in q and np.isfinite(q["iv"]):
                    book.last_iv[key] = float(q["iv"])
            equity = self._equity(d, book)
            if prev_state is not None:                      # attribute the overnight P&L of the positions held at the previous close
                p_pos, p_units, p_spot, p_greeks, p_date = prev_state
                dS, dt_days = spot - p_spot, (d - p_date).days
                a = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
                for key, qty in p_pos.items():
                    g = p_greeks.get(key)
                    if g is None:
                        continue
                    q = self._quotes[d].get(key)
                    d_iv = (float(q["iv"]) - g["iv"]) if q is not None and "iv" in q and np.isfinite(q["iv"]) else 0.0
                    a["delta"] += qty * self.mult * g["delta"] * dS
                    a["gamma"] += qty * self.mult * 0.5 * g["gamma"] * dS ** 2
                    a["vega"] += qty * self.mult * g["vega"] * d_iv
                    a["theta"] += qty * self.mult * g["theta"] * dt_days / 365.0
                hedge = p_units * dS
                pnl_before_costs = equity - prev_equity + day_cost + day_slip
                rows_attr.append({"date": d, **a, "hedge": hedge, "unexplained": pnl_before_costs - sum(a.values()) - hedge, "slippage": -day_slip, "costs": -day_cost})
            orders = strategy.on_date(StrategyContext(d, self.chains[d], spot, self.spot.loc[:d], dict(book.positions), book.units, equity, self.capital, self._greeks(d, book)[0]))
            if orders:
                if self.lag == 0:
                    fee, slip = self._execute(d, orders, book)
                    day_cost, day_slip = day_cost + fee, day_slip + slip
                    equity = self._equity(d, book)
                else:
                    pending.append((i + self.lag, list(orders)))
            totals, per_contract = self._greeks(d, book)
            equity_hist[d], costs[d] = equity, day_cost
            rows_pos.append({"date": d, "open_contracts": float(sum(abs(v) for v in book.positions.values())), "underlying_units": book.units})
            rows_greeks.append({"date": d, **totals})
            prev_state, prev_equity = (dict(book.positions), book.units, spot, per_contract, d), equity
        equity_s = pd.Series(equity_hist)
        pnl = equity_s.diff().fillna(0.0)
        return OptionBacktestResult(equity_s, pnl, pnl / self.capital, pd.DataFrame(book.trades), pd.DataFrame(rows_pos).set_index("date"), pd.DataFrame(rows_greeks).set_index("date"),
                                    pd.DataFrame(rows_attr).set_index("date") if rows_attr else pd.DataFrame(), pd.Series(costs), book.skipped)


# --------------------------------------------------------------------------------------------------------------------- selection
def dte_of(chain: pd.DataFrame, expiry) -> int:
    return int((pd.Timestamp(expiry) - chain["date"].iloc[0]).days)


def pick_expiry(chain: pd.DataFrame, target_dte: int, min_dte: int = 0):
    """The listed expiry whose days-to-expiry is closest to ``target_dte`` (and at least ``min_dte``)."""
    exp = chain.drop_duplicates("expiry")[["expiry", "T"]]
    exp = exp[exp["T"] * 365.0 >= min_dte]
    if exp.empty:
        return None
    return exp.iloc[(exp["T"] * 365.0 - target_dte).abs().argsort().iloc[0]]["expiry"]


def pick_by_delta(chain: pd.DataFrame, expiry, right: str, target_delta: float):
    """The listed strike at ``expiry`` of the given ``right`` whose absolute Black-Scholes delta (from the quote's implied vol) is closest to ``target_delta``. Returns ``(strike, delta)``."""
    g = chain[(chain["expiry"] == expiry) & (chain["right"] == right) & (chain["bid"] > 0)]
    if g.empty:
        return None
    gk = bsm_greeks(g["underlying"].to_numpy(), g["strike"].to_numpy(), g["T"].to_numpy(), g["rate"].to_numpy(), g["dividend"].to_numpy(), g["iv"].to_numpy(), right == "C")
    delta = np.abs(gk["delta"])
    i = int(np.argmin(np.abs(delta - target_delta)))
    return float(g["strike"].iloc[i]), float(delta[i])


def pick_atm(chain: pd.DataFrame, expiry):
    """The strike nearest the forward at ``expiry``."""
    g = chain[chain["expiry"] == expiry]
    return float(g.iloc[(g["strike"] - g["F"]).abs().argsort().iloc[0]]["strike"])


def quoted(chain: pd.DataFrame, expiry, strike: float, right: str) -> bool:
    return bool(((chain["expiry"] == expiry) & (chain["strike"] == strike) & (chain["right"] == right) & (chain["bid"] > 0)).any())
