"""Cross-asset trend following on the common API: time-series momentum, breakout and moving-average signals blended with a confidence, sized to a volatility target.

For every instrument (a plain instrument id or a future chain id, in which case a back-adjusted continuous history is used) three signals in ``[-1, 1]`` are computed from
point-in-time prices:

* **time-series momentum** over several lookbacks: ``tanh(r_k / (sigma sqrt(k)))``, the return in units of its own standard error, averaged;
* **breakout**: the position of the price in its ``breakout``-day range, rescaled to ``[-1, 1]``;
* **moving-average crossover**: ``tanh`` of the fast-minus-slow average in volatility units.

The blend is their mean; the CONFIDENCE is the share of signals that agree with its sign (full agreement: 1). An optional regime filter halves exposure when realised volatility is in
its top quartile of its own history. Weights are inversely proportional to volatility so each market contributes similar risk, scaled to ``target_vol`` and capped by ``max_leverage``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import Schedule, Signal, Strategy, Target
from .common import ann_vol, continuous_history, risk_weights


class TrendStrategy(Strategy):
    name = "trend"
    schedule = Schedule("weekly", "16:30", 4, "US")

    def __init__(self, instruments, lookbacks=(21, 63, 126, 252), breakout: int = 126, ma_pair=(50, 200), target_vol: float = 0.10, vol_window: int = 60, max_leverage: float = 3.0,
                 regime_filter: bool = False, name: str | None = None, min_confidence: float = 0.0):
        self.instruments = list(instruments)
        self.lookbacks, self.breakout, self.ma_pair = tuple(lookbacks), breakout, tuple(ma_pair)
        self.target_vol, self.vol_window, self.max_leverage, self.regime_filter, self.min_confidence = target_vol, vol_window, max_leverage, regime_filter, min_confidence
        if name:
            self.name = name

    def _prices(self, ctx, iid: str) -> pd.Series:
        n = max(max(self.lookbacks), self.breakout, self.ma_pair[1]) + 5
        return continuous_history(ctx, iid, n) if ctx.registry.is_chain(iid) else ctx.data.history(iid, n)

    def components(self, p: pd.Series) -> dict[str, float]:
        lp = np.log(p.dropna())
        r = lp.diff().dropna()
        sigma = float(r.tail(self.vol_window).std(ddof=1))
        out: dict[str, float] = {}
        if not sigma > 0:
            return out
        mom = [np.tanh((lp.iloc[-1] - lp.iloc[-1 - k]) / (sigma * np.sqrt(k))) for k in self.lookbacks if len(lp) > k + 1]
        if mom:
            out["momentum"] = float(np.mean(mom))
        if len(p) > self.breakout:
            w = p.tail(self.breakout)
            lo, hi = float(w.min()), float(w.max())
            out["breakout"] = float(2.0 * (p.iloc[-1] - lo) / (hi - lo) - 1.0) if hi > lo else 0.0
        fast, slow = self.ma_pair
        if len(p) > slow:
            out["ma"] = float(np.tanh((p.tail(fast).mean() - p.tail(slow).mean()) / (p.iloc[-1] * sigma * np.sqrt(slow / 2.0))))
        return out

    def generate_signals(self, ctx):
        out = []
        for iid in self.instruments:
            p = self._prices(ctx, iid)
            if len(p) < 40:
                continue
            comp = self.components(p)
            if not comp:
                continue
            vals = np.array(list(comp.values()))
            blend = float(vals.mean())
            conf = float((np.sign(vals) == np.sign(blend)).mean()) if blend != 0 else 0.0
            if self.regime_filter:
                r = np.log(p).diff().dropna()
                rv = r.rolling(20).std().dropna()
                if len(rv) > 60 and rv.iloc[-1] > rv.quantile(0.75):
                    blend *= 0.5
            out.append(Signal(iid, blend, conf, meta=comp))
        return out

    def map_to_targets(self, ctx, signals):
        sig = {s.instrument_id: s.value * (s.confidence if s.confidence >= self.min_confidence else 0.0) for s in signals}
        vols = {i: ann_vol(ctx, i, self.vol_window) for i in sig}
        w = risk_weights(sig, vols, self.target_vol, self.max_leverage)
        return [Target(i, weight=v) for i, v in w.items()]
