"""A quantity-and-cash ledger for the portfolio environment, and the two reward formulas that need exact arithmetic.

The ledger holds cash, a quantity per asset and a mark per asset, so that equity, weights and costs are DERIVED from what is held (``equity = cash + sum(quantity * mark)``), never carried along as a target.
Weights drift with the marks, a trade is solved for cash feasibility (the target weights are those of equity AFTER the trading cost), and a portfolio whose equity reaches zero is dead for good.

This is one currency, cash-equity accounting. It does not model futures margin, FX or corporate actions; the event-driven engine and ledger in ``src/engine`` and ``src/ledger`` do (see
``docs/ml_correctness_v3/ASSUMPTIONS_AND_LIMITATIONS.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS_EQUITY = 1e-9
SOLVE_ITERATIONS = 100


@dataclass
class Ledger:
    cash: float
    qty: np.ndarray
    price: np.ndarray
    peak: float
    dead: bool = False

    @classmethod
    def start(cls, n_assets: int, cash: float = 1.0, price: np.ndarray | None = None) -> "Ledger":
        marks = np.ones(n_assets) if price is None else np.asarray(price, dtype=float).copy()
        return cls(float(cash), np.zeros(n_assets), marks, float(cash))

    @property
    def equity(self) -> float:
        return 0.0 if self.dead else float(self.cash + self.qty @ self.price)

    def weights(self) -> np.ndarray:
        e = self.equity
        return np.zeros(len(self.qty)) if e <= EPS_EQUITY else self.qty * self.price / e

    def gross(self) -> float:
        e = self.equity
        return 0.0 if e <= EPS_EQUITY else float(np.abs(self.qty * self.price).sum() / e)

    def drawdown(self) -> float:
        return -1.0 if self.dead else self.equity / self.peak - 1.0

    def _bury(self) -> None:
        self.dead, self.cash, self.qty = True, 0.0, np.zeros_like(self.qty)

    def mark(self, ret: np.ndarray, financing: float = 0.0, borrow: float = 0.0) -> None:
        """One session passes: the marks move by ``ret`` (a missing return is no move), debit cash pays ``financing`` (a daily rate) and short notional pays ``borrow`` (a daily fee)."""
        if self.dead:
            return
        self.price = self.price * (1.0 + np.maximum(np.nan_to_num(np.asarray(ret, dtype=float)), -1.0))      # a return below -100% is a data error: the price stops at zero
        self.qty = np.where(self.price > 0.0, self.qty, 0.0)                                                 # a holding in an asset worth nothing is written off (its value was already zero)
        if financing and self.cash < 0.0:
            self.cash += self.cash * financing
        if borrow:
            self.cash -= borrow * float(np.maximum(-self.qty * self.price, 0.0).sum())
        if self.equity <= EPS_EQUITY:
            self._bury()
        else:
            self.peak = max(self.peak, self.equity)

    def trade_to(self, target: np.ndarray, cost_rate: float = 0.0, impact: float = 0.0) -> dict:
        """Trade to ``target`` weights of the equity that is left AFTER the cost, at the current marks.

        With ``k = equity_after / equity_before`` and ``dw = target * k - weights_before`` (in units of the equity before), the cost is ``cost_rate * sum|dw| + impact * sum(dw^2)`` of the equity before and
        ``k = 1 - cost``. That fixed point is solved by iteration; quantities, cash and the cost then follow, and ``cash_after = cash - fills * marks - cost`` leaves equity exactly ``equity * k``.
        """
        if self.dead:
            return {"fill": np.zeros_like(self.qty), "cost": 0.0, "turnover": 0.0, "dead": True}
        tradable = self.price > 0.0
        e0, w0, target = self.equity, self.weights(), np.where(tradable, np.asarray(target, dtype=float), 0.0)    # nothing can be bought in an asset that is worth nothing
        k = 1.0
        for _ in range(SOLVE_ITERATIONS):
            dw = target * k - w0
            k_new = 1.0 - cost_rate * float(np.abs(dw).sum()) - impact * float((dw ** 2).sum())
            if abs(k_new - k) < 1e-15:
                k = k_new
                break
            k = 0.5 * (k + k_new) if abs(k_new - k) > 0.2 else k_new                           # damped when a large impact makes the map steep
        if k <= 0.0:                                                                           # the trade costs more than the portfolio is worth
            self._bury()
            return {"fill": np.zeros_like(self.qty), "cost": 1.0, "turnover": float(np.abs(target * 0.0 - w0).sum()), "dead": True}
        dw = target * k - w0
        new_qty = np.divide(target * (e0 * k), self.price, out=np.zeros_like(self.qty), where=tradable)
        fill = new_qty - self.qty
        cost = e0 * (1.0 - k)
        self.cash = self.cash - float(fill @ self.price) - cost
        self.qty = new_qty
        self.peak = max(self.peak, self.equity)
        return {"fill": fill, "cost": 1.0 - k, "turnover": float(np.abs(dw).sum()), "dead": False}


# ------------------------------------------------------------------------------------------------------------ rewards
def drawdown_increase(previous_dd: float, dd: float) -> float:
    """How much deeper the drawdown got in one step, under the negative convention ``dd = equity / peak - 1``: ``max(previous_dd - dd, 0)``. A recovery or a new peak is zero."""
    return max(previous_dd - dd, 0.0)


def differential_sharpe(x: float, A: float, B: float, eta: float, clip: float = 10.0, floor: float = 1e-8) -> tuple[float, float, float, float]:
    """Moody and Saffell's differential Sharpe ratio for one return ``x`` (percent, in this environment), from the moments BEFORE this return.

    ``dA = x - A; dB = x*x - B; D = (B*dA - A*dB/2) / (B - A*A)^(3/2)``; the moments then move ``A += eta*dA; B += eta*dB``. The reward is ``D`` itself, clipped to ``[-clip, clip]``; where the variance
    ``B - A*A`` is not above ``floor`` (the first steps, a constant return) ``D`` is undefined and the reward is 0. Returns ``(reward, raw D or nan, A_next, B_next)``.
    """
    dA, dB = x - A, x * x - B
    var = B - A * A
    if var > floor:
        raw = (B * dA - 0.5 * A * dB) / var ** 1.5
        reward = float(np.clip(raw, -clip, clip))
    else:
        raw, reward = float("nan"), 0.0
    return reward, raw, A + eta * dA, B + eta * dB
