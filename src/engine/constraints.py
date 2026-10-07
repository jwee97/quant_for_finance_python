"""Portfolio constraints: limits on what a strategy may hold, applied to its target notionals before anything is traded.

Targets arrive as SIGNED NOTIONALS in the base currency (position x multiplier x price x exchange rate); :meth:`ConstraintSet.apply` returns notionals that satisfy the limits and the
names of the constraints that bound, so a result can say *why* the portfolio is smaller than the signals wanted. The limits, applied in this order:

1. per-instrument weight (a fraction of capital) and explicit per-instrument caps;
2. liquidity budget: at most a fraction of average daily volume per instrument;
3. per-asset-class gross caps;
4. per-currency net exposure caps (an FX position is long one currency and short the other);
5. factor exposure bands (beta-weighted net exposure);
6. gross exposure / leverage, then net exposure;
7. concentration: no instrument above a share of gross exposure;
8. volatility target: scale all notionals so the portfolio's annualised volatility (from the supplied covariance) does not exceed the target;
9. drawdown control: a step schedule scaling exposure down as the drawdown deepens;
10. margin utilisation: scale so that estimated initial margin stays below a fraction of equity;
11. turnover: limit the total change from the current holdings to a fraction of capital by moving only part of the way to the target.

Every limit is optional. The order matters (later scalings act on earlier results) and is fixed so that results are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ConstraintInputs:
    capital: float
    current: dict                                   # instrument id -> current signed notional (base)
    asset_class: dict = field(default_factory=dict)
    currency_exposure: dict = field(default_factory=dict)   # instrument id -> {currency: sign} for 1 unit of notional
    adv_notional: dict = field(default_factory=dict)        # instrument id -> average daily traded notional (base)
    cov: pd.DataFrame | None = None                         # annualised covariance of instrument returns
    betas: dict = field(default_factory=dict)               # instrument id -> {factor: beta}
    drawdown: float = 0.0
    margin_rate: dict = field(default_factory=dict)         # instrument id -> initial margin as a fraction of notional


def _shrink(n: np.ndarray, contrib: np.ndarray, low: float, high: float) -> bool:
    """Scale down (in place) the positions whose contribution pushes the total ``contrib.sum()`` outside ``[low, high]`` by a common factor, just enough to bring it to the bound.
    Returns True if anything changed. Only positions on the offending side are reduced, so hedges are never cut."""
    total = contrib.sum()
    if low <= total <= high:
        return False
    side = 1.0 if total > high else -1.0
    limit = high if side > 0 else low
    same = contrib * side > 0
    s = abs(contrib[same].sum())
    if s <= 0:
        return False
    f = 1.0 - (abs(total - limit)) / s
    n[same] *= max(f, 0.0)
    return True


@dataclass
class ConstraintSet:
    max_instrument_weight: float | None = None
    instrument_caps: dict = field(default_factory=dict)         # id -> max |notional| as a fraction of capital
    liquidity_participation: float | None = None                # max |notional| / ADV notional
    asset_class_caps: dict = field(default_factory=dict)        # class -> max gross as a fraction of capital
    currency_caps: dict = field(default_factory=dict)           # currency -> max |net exposure| as a fraction of capital
    factor_bands: dict = field(default_factory=dict)            # factor -> (low, high) net beta exposure as a fraction of capital
    max_gross: float | None = None                              # leverage cap (gross / capital)
    max_net: float | None = None
    max_concentration: float | None = None                      # max share of gross in one instrument
    vol_target: float | None = None                             # annualised
    drawdown_scaling: tuple = ()                                 # ((drawdown threshold, scale), ...) e.g. ((0.10, 0.75), (0.20, 0.5), (0.30, 0.0))
    max_margin_utilization: float | None = None
    max_turnover: float | None = None                           # |change| / capital per rebalance

    def apply(self, targets: dict, inputs: ConstraintInputs) -> tuple[dict, list[str]]:
        ids = list(targets)
        n = np.array([targets[i] for i in ids], dtype=float)
        cap = inputs.capital
        bound: list[str] = []

        def note(name):
            if name not in bound:
                bound.append(name)

        if cap <= 0:
            return {i: 0.0 for i in ids}, ["no_capital"]
        # 1 per instrument
        for k, i in enumerate(ids):
            lim = min(self.max_instrument_weight if self.max_instrument_weight is not None else np.inf, self.instrument_caps.get(i, np.inf)) * cap
            if abs(n[k]) > lim:
                n[k] = np.sign(n[k]) * lim
                note("instrument_cap")
        # 2 liquidity
        if self.liquidity_participation is not None:
            for k, i in enumerate(ids):
                adv = inputs.adv_notional.get(i, np.nan)
                if np.isfinite(adv) and adv > 0 and abs(n[k]) > self.liquidity_participation * adv:
                    n[k] = np.sign(n[k]) * self.liquidity_participation * adv
                    note("liquidity")
        # 3 asset-class gross caps
        for cls, lim in self.asset_class_caps.items():
            idx = [k for k, i in enumerate(ids) if inputs.asset_class.get(i) == cls]
            g = np.abs(n[idx]).sum() if idx else 0.0
            if g > lim * cap and g > 0:
                n[idx] *= lim * cap / g
                note(f"asset_class:{cls}")
        # 4 currency net exposure
        for ccy, lim in self.currency_caps.items():
            contrib = np.array([n[k] * inputs.currency_exposure.get(i, {}).get(ccy, 0.0) for k, i in enumerate(ids)])
            if _shrink(n, contrib, -lim * cap, lim * cap):
                note(f"currency:{ccy}")
        # 5 factor bands
        for factor, (lo, hi) in self.factor_bands.items():
            contrib = np.array([n[k] * inputs.betas.get(i, {}).get(factor, 0.0) for k, i in enumerate(ids)])
            if _shrink(n, contrib, lo * cap, hi * cap):
                note(f"factor:{factor}")
        # 6 gross and net
        gross = np.abs(n).sum()
        if self.max_gross is not None and gross > self.max_gross * cap and gross > 0:
            n *= self.max_gross * cap / gross
            note("gross")
        if self.max_net is not None:
            net = n.sum()
            if abs(net) > self.max_net * cap:
                longs, shorts = n[n > 0].sum(), -n[n < 0].sum()
                excess = abs(net) - self.max_net * cap
                if net > 0 and longs > 0:
                    n[n > 0] *= (longs - excess) / longs
                elif shorts > 0:
                    n[n < 0] *= (shorts - excess) / shorts
                note("net")
        # 7 concentration: no instrument above ``max_concentration`` of the gross AFTER the cap (capping lowers the gross, so iterate; infeasible when fewer instruments than 1 / limit)
        if self.max_concentration is not None:
            active = int((np.abs(n) > 0).sum())
            if active and self.max_concentration * active >= 1.0:
                for _ in range(60):
                    gross = np.abs(n).sum()
                    big = np.abs(n) > self.max_concentration * gross * (1.0 + 1e-9)
                    if gross <= 0 or not big.any():
                        break
                    n[big] = np.sign(n[big]) * self.max_concentration * gross
                    note("concentration")
        # 8 volatility target
        if self.vol_target is not None and inputs.cov is not None and len(ids):
            c = inputs.cov.reindex(index=ids, columns=ids).fillna(0.0).to_numpy()
            w = n / cap
            var = float(w @ c @ w)
            vol = np.sqrt(max(var, 0.0))
            if vol > self.vol_target > 0:
                n *= self.vol_target / vol
                note("vol_target")
        # 9 drawdown control
        if self.drawdown_scaling:
            scale = 1.0
            for threshold, s in sorted(self.drawdown_scaling):
                if inputs.drawdown >= threshold:
                    scale = s
            if scale < 1.0:
                n *= scale
                note("drawdown")
        # 10 margin utilisation
        if self.max_margin_utilization is not None:
            req = sum(abs(n[k]) * inputs.margin_rate.get(i, 0.0) for k, i in enumerate(ids))
            if req > self.max_margin_utilization * cap and req > 0:
                n *= self.max_margin_utilization * cap / req
                note("margin")
        # 11 turnover
        if self.max_turnover is not None:
            cur = np.array([inputs.current.get(i, 0.0) for i in ids])
            change = np.abs(n - cur).sum()
            if change > self.max_turnover * cap and change > 0:
                n = cur + (n - cur) * self.max_turnover * cap / change
                note("turnover")
        return {i: float(v) for i, v in zip(ids, n)}, bound
