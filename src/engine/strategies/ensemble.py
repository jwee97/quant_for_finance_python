"""A regime-aware ensemble of strategies on the common API: trend, carry and volatility sleeves blended by a regime filter, a risk overlay and a drawdown brake.

:class:`EnsembleStrategy` owns a set of MEMBER strategies (any ``Strategy``: trend, carry, basis, relative value, volatility). Each member runs unchanged against a
:class:`_MemberContext` that hands it the same point-in-time data but captures what it would trade instead of sending orders. The ensemble then

1. asks a **regime detector** for the state probabilities now (:class:`VolRegime`: realised volatility against its own history; :class:`HMMRegime`: a two-state Gaussian HMM
   refit on past returns only and filtered to today), and blends the per-regime multipliers of each member by those probabilities;
2. applies a **risk overlay** across members from their SHADOW returns (what each member's targets earned on the prices that followed, before costs): ``inverse_vol`` or ``erc``
   (equal risk contribution), mixed with the base weights by ``overlay_strength``;
3. scales the combined book to a **volatility target** using the realised volatility of the combined shadow return (scale capped by ``max_scale``);
4. applies a **drawdown brake**: exposure is cut linearly from full at ``dd_start`` to ``dd_floor`` of normal at ``dd_full``, using the account drawdown;
5. sums the scaled member targets by instrument and sends them through the engine, so one set of constraints, costs and margin rules governs the whole book.

Member targets given as weights are added as weights; quantities (options and their hedges) are scaled and passed through. If one instrument is targeted both ways the quantity is converted to
a weight at today's price. Instruments a member stopped targeting are flattened explicitly. Shadow returns use only prices known at each decision, so nothing here looks ahead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..strategy import Schedule, Signal, Strategy, Target
from .common import continuous_history


# ------------------------------------------------------------------------------------------------------------------------------------ regime detectors
class VolRegime:
    """Stress when the reference's ``window``-day realised volatility is above ``quantile`` of its own past (needs ``min_obs`` days). ``probabilities`` -> ``{"calm", "stress"}``."""

    states = ("calm", "stress")

    def __init__(self, reference: str, window: int = 21, quantile: float = 0.75, min_obs: int = 120):
        self.reference, self.window, self.quantile, self.min_obs = reference, window, quantile, min_obs

    def probabilities(self, ctx) -> dict[str, float]:
        p = continuous_history(ctx, self.reference, 800) if ctx.registry.is_chain(self.reference) else ctx.data.history(self.reference, 800)
        r = np.log(p).diff().dropna()
        rv = r.rolling(self.window).std().dropna()
        if len(rv) < self.min_obs:
            return {"calm": 1.0, "stress": 0.0}
        z = float((rv.iloc[-1] > rv.iloc[:-1].quantile(self.quantile)))
        return {"calm": 1.0 - z, "stress": z}


class HMMRegime:
    """A two-state Gaussian HMM on the reference's daily returns, refit every ``refit_every`` decisions on data known now and filtered forward; state 1 is the high-variance one."""

    states = ("calm", "stress")

    def __init__(self, reference: str, min_obs: int = 250, refit_every: int = 63, seed: int = 0):
        self.reference, self.min_obs, self.refit_every, self.seed = reference, min_obs, refit_every, seed
        self._fit = None
        self._since = 0

    def probabilities(self, ctx) -> dict[str, float]:
        from ...models.regimes import fit_hmm, forward_filter, log_emission

        p = continuous_history(ctx, self.reference, 1500) if ctx.registry.is_chain(self.reference) else ctx.data.history(self.reference, 1500)
        r = np.log(p).diff().dropna().to_numpy()
        if len(r) < self.min_obs:
            return {"calm": 1.0, "stress": 0.0}
        if self._fit is None or self._since >= self.refit_every:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fit = fit_hmm(r.reshape(-1, 1), 2, n_init=2, n_iter=100, seed=self.seed, init_from=self._fit)
            self._since = 0
        self._since += 1
        alpha, _ = forward_filter(log_emission(r.reshape(-1, 1), self._fit), self._fit.startprob, self._fit.transmat)
        s = float(alpha[-1][1])
        return {"calm": 1.0 - s, "stress": s}


# -------------------------------------------------------------------------------------------------------------------------------- member plumbing
class _MemberPortfolio:
    def __init__(self, ctx, share: float):
        self._ctx, self._share = ctx, share

    def __getattr__(self, name):
        return getattr(self._ctx.portfolio, name)

    @property
    def capital(self) -> float:
        return self._ctx.portfolio.capital * self._share


class _MemberContext:
    """What a member strategy sees: the ensemble's data and portfolio view, its own scratch store, and a ``set_targets`` that records instead of trading."""

    def __init__(self, ctx, store: dict, name: str, share: float):
        self._ctx = ctx
        self.name, self.store = name, store
        self.portfolio = _MemberPortfolio(ctx, share)
        self.captured: list[Target] | None = None

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    @property
    def ts(self):
        return self._ctx.ts

    def set_targets(self, targets, **kwargs):
        if isinstance(targets, dict):
            targets = [t if isinstance(t, Target) else Target(k, quantity=float(t)) for k, t in targets.items()]
        self.captured = list(targets)

    def _refuse(self, *a, **k):
        raise RuntimeError("an ensemble member may only emit targets (ctx.set_targets); direct orders would bypass the ensemble's risk overlay")

    submit = order = buy = sell = cancel = cancel_all = close_all = exercise = _refuse


def _price_now(ctx, iid: str) -> float:
    if ctx.registry.is_chain(iid):
        s = continuous_history(ctx, iid, 5)
        return float(s.iloc[-1]) if len(s) else float("nan")
    return ctx.data.mid(iid)


class EnsembleStrategy(Strategy):
    name = "ensemble"
    schedule = Schedule("daily", "16:30", 4, "US")

    def __init__(self, members: dict, detector=None, regime_multipliers: dict | None = None, base_weights: dict | None = None, rebalance_days: dict | int = 5,
                 target_vol: float | None = 0.10, vol_window: int = 40, max_scale: float = 3.0, min_scale: float = 0.2, overlay: str = "inverse_vol", overlay_strength: float = 0.5,
                 dd_start: float = 0.05, dd_full: float = 0.15, dd_floor: float = 0.25, band_abs: float = 0.002, band_rel: float = 0.15, name: str | None = None):
        if overlay not in ("none", "inverse_vol", "erc"):
            raise ValueError("overlay must be none, inverse_vol or erc")
        self.members = dict(members)
        self.detector = detector
        self.regime_multipliers = regime_multipliers or {}
        self.base_weights = base_weights or {k: 1.0 / len(self.members) for k in self.members}
        self.rebalance_days = rebalance_days
        self.target_vol, self.vol_window, self.max_scale, self.min_scale = target_vol, vol_window, max_scale, min_scale
        self.overlay, self.overlay_strength = overlay, overlay_strength
        self.dd_start, self.dd_full, self.dd_floor = dd_start, dd_full, dd_floor
        self.band_abs, self.band_rel = band_abs, band_rel
        self._sent: dict[str, float] = {}
        if name:
            self.name = name
        self._stores = {k: {} for k in self.members}
        self._last_run: dict[str, pd.Timestamp] = {}
        self._cached: dict[str, list[Target]] = {}
        self._shadow_pos: dict[str, tuple[pd.Timestamp, dict]] = {}
        self._shadow_ret: dict[str, list] = {k: [] for k in self.members}
        self._combined: tuple | None = None
        self._combined_ret: list = []
        self._prev_ids: set = set()
        self.history: list[dict] = []                                  # one row per decision: the regime, the member weights and the scales (for inspection)

    # ------------------------------------------------------------------------------------------------------------------------------- hooks
    def on_start(self, ctx):
        for k, m in self.members.items():
            m.on_start(_MemberContext(ctx, self._stores[k], k, self.base_weights.get(k, 0.0)))

    def on_instrument_event(self, ctx, event):
        for k, m in self.members.items():
            m.on_instrument_event(_MemberContext(ctx, self._stores[k], k, self.base_weights.get(k, 0.0)), event)

    # ------------------------------------------------------------------------------------------------------------------------- the decision
    def _every(self, k: str) -> int:
        return self.rebalance_days.get(k, 5) if isinstance(self.rebalance_days, dict) else int(self.rebalance_days)

    def _run_member(self, ctx, k: str, share: float) -> list[Target]:
        last = self._last_run.get(k)
        if last is not None and (ctx.ts - last).days < self._every(k) and k in self._cached:
            return self._cached[k]
        mc = _MemberContext(ctx, self._stores[k], k, share)
        self.members[k].on_schedule(mc)
        out = mc.captured if mc.captured is not None else self._cached.get(k, [])
        self._last_run[k], self._cached[k] = ctx.ts, out
        return out

    def _to_weights(self, ctx, targets: list[Target]) -> dict[str, float]:
        """Weights of the weight-type targets only (quantity targets are handled separately)."""
        return {t.instrument_id: float(t.weight) for t in targets if t.weight is not None}

    def _update_shadow(self, ctx, k: str, targets: list[Target]) -> None:
        """Record the member's shadow return since its previous decision and store its new weights with today's prices."""
        prev = self._shadow_pos.get(k)
        if prev is not None:
            ts0, pos = prev
            r = 0.0
            for iid, (w, p0) in pos.items():
                p1 = _price_now(ctx, iid)
                if np.isfinite(p1) and np.isfinite(p0) and p0 > 0:
                    r += w * (p1 / p0 - 1.0)
            self._shadow_ret[k].append((ctx.ts, r / max((ctx.ts - ts0).days, 1) * 1.0, (ctx.ts - ts0).days))
        pos = {}
        for iid, w in self._to_weights(ctx, targets).items():
            p = _price_now(ctx, iid)
            if np.isfinite(p) and w != 0.0:
                pos[iid] = (w, p)
        self._shadow_pos[k] = (ctx.ts, pos)

    def _member_vol(self, k: str) -> float:
        rows = self._shadow_ret[k][-self.vol_window:]
        if len(rows) < 10:
            return float("nan")
        daily = np.array([r for _, r, _ in rows], float)                  # return per calendar day between decisions
        return float(daily.std(ddof=1) * np.sqrt(365.0))

    def _regime_blend(self, probs: dict[str, float]) -> dict[str, float]:
        out = {}
        for k in self.members:
            out[k] = sum(probs.get(state, 0.0) * self.regime_multipliers.get(state, {}).get(k, 1.0) for state in probs)
        return out

    def _overlay_weights(self, weights: dict[str, float]) -> dict[str, float]:
        if self.overlay == "none" or self.overlay_strength <= 0:
            return weights
        vols = {k: self._member_vol(k) for k in weights}
        ok = {k: v for k, v in vols.items() if np.isfinite(v) and v > 1e-6}
        if len(ok) < len(weights) or len(ok) < 2:
            return weights
        if self.overlay == "erc":
            from ...portfolio.risk_parity import risk_parity_weights

            n = min(len(self._shadow_ret[k]) for k in ok)
            frame = pd.DataFrame({k: [r for _, r, _ in self._shadow_ret[k][-n:]] for k in ok})
            if n < 20:
                return weights
            rp = np.asarray(risk_parity_weights(frame.cov() * 365.0), float)
            tilt = {k: float(v) * len(ok) for k, v in zip(frame.columns, rp)}
        else:
            inv = {k: 1.0 / v for k, v in ok.items()}
            mean = np.mean(list(inv.values()))
            tilt = {k: v / mean for k, v in inv.items()}
        a = self.overlay_strength
        return {k: w * ((1 - a) + a * tilt.get(k, 1.0)) for k, w in weights.items()}

    def _vol_scale(self) -> float:
        if self.target_vol is None:
            return 1.0
        rows = self._combined_ret[-self.vol_window:]
        if len(rows) < 10:
            return 1.0
        daily = np.array(rows, float)
        vol = float(daily.std(ddof=1) * np.sqrt(365.0))
        if not vol > 1e-6:
            return 1.0
        return float(np.clip(self.target_vol / vol, self.min_scale, self.max_scale))

    def _drawdown_scale(self, ctx) -> float:
        dd = abs(float(ctx.portfolio.drawdown or 0.0))
        if dd <= self.dd_start:
            return 1.0
        if dd >= self.dd_full:
            return self.dd_floor
        return float(1.0 - (1.0 - self.dd_floor) * (dd - self.dd_start) / (self.dd_full - self.dd_start))

    def on_schedule(self, ctx):
        probs = self.detector.probabilities(ctx) if self.detector is not None else {"calm": 1.0}
        mult = self._regime_blend(probs)
        base = {k: self.base_weights.get(k, 0.0) * mult[k] for k in self.members}
        raw: dict[str, list[Target]] = {}
        for k in self.members:
            raw[k] = self._run_member(ctx, k, base[k] or self.base_weights.get(k, 0.0))
            self._update_shadow(ctx, k, raw[k])
        weights = self._overlay_weights(base)
        # the combined shadow return from the previous decision's blended weights
        if self._combined is not None:
            ts0, pos = self._combined
            r = 0.0
            for iid, (w, p0) in pos.items():
                p1 = _price_now(ctx, iid)
                if np.isfinite(p1) and np.isfinite(p0) and p0 > 0:
                    r += w * (p1 / p0 - 1.0)
            self._combined_ret.append(r / max((ctx.ts - ts0).days, 1))
        vscale, dscale = self._vol_scale(), self._drawdown_scale(ctx)
        scale = vscale * dscale
        w_sum: dict[str, float] = {}
        q_sum: dict[str, float] = {}
        cap = ctx.portfolio.capital
        for k, targets in raw.items():
            share = weights[k]                                           # the member's effective share of capital: base weight x regime multiplier x overlay
            ratio = share / self.base_weights[k] if self.base_weights.get(k) else 0.0
            for t in targets:
                if t.weight is not None:
                    w_sum[t.instrument_id] = w_sum.get(t.instrument_id, 0.0) + t.weight * share * scale
                elif t.quantity is not None:
                    q_sum[t.instrument_id] = q_sum.get(t.instrument_id, 0.0) + t.quantity * ratio * scale
                elif cap > 0:
                    w_sum[t.instrument_id] = w_sum.get(t.instrument_id, 0.0) + t.notional / cap * ratio * scale
        for iid in list(q_sum):                                          # one instrument targeted both ways: express the quantity as a weight
            if iid in w_sum and not ctx.registry.is_chain(iid):
                inst, px, cap = ctx.registry.get(iid), ctx.data.mid(iid), ctx.portfolio.capital
                if np.isfinite(px) and cap > 0:
                    w_sum[iid] += q_sum.pop(iid) * inst.notional(px, 1.0) / cap
        for iid, w in list(w_sum.items()):                                # hysteresis: leave a weight where it is unless it moved by more than the band (saves turnover)
            old = self._sent.get(iid)
            if old is not None and abs(w - old) < max(self.band_abs, self.band_rel * abs(old)) and (w != 0.0):
                w_sum[iid] = old
        self._sent = dict(w_sum)
        final = [Target(i, weight=float(w)) for i, w in w_sum.items()] + [Target(i, quantity=float(q)) for i, q in q_sum.items()]
        now_ids = {t.instrument_id for t in final}
        for iid in self._prev_ids - now_ids:
            if not ctx.registry.is_chain(iid) and ctx.registry.has_instrument(iid):
                inst = ctx.registry.get(iid)
                if inst.expiry is not None and inst.expiry < ctx.ts:
                    continue
            final.append(Target(iid, weight=0.0))
        self._prev_ids = now_ids
        pos = {}
        for iid, w in w_sum.items():
            p = _price_now(ctx, iid)
            if np.isfinite(p) and w != 0:
                pos[iid] = (w, p)
        self._combined = (ctx.ts, pos)
        self.history.append({"ts": ctx.ts, **{f"p_{s}": v for s, v in probs.items()}, **{f"w_{k}": float(v) for k, v in weights.items()}, "vol_scale": vscale, "dd_scale": dscale})
        ctx.set_targets(final)

    def generate_signals(self, ctx) -> list[Signal]:
        return []
