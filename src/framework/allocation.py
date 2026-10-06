"""Turning forecasts and regimes into portfolio weights.

An allocator returns TARGET weights stamped at the decision date (the backtest engine applies the execution lag).

    static          one of the existing books: equal weight, inverse volatility, risk parity, mean-CVaR, HRP, HERC or plug-in MVO
    forecast_stack  the Generation 1 position stack applied to the combined forecast (mean / std)
    confidence      sign of the forecast times its confidence: low-confidence forecasts get small positions
    regime_switch   a probability-weighted blend of other allocators, one per regime: "in a crisis use mean-CVaR, in calm use MVO,
                    in an inflation shock tilt to commodities"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.volatility import rolling_volatility
from ..portfolio.constraints import Constraints
from ..signals.transform import apply_weight_cap, normalise_gross, signal_to_positions
from ..utils.dates import rebalance_dates
from .registry import ALLOCATORS, register_allocator

LADDER = {"equal_weight": "M0_equal_weight", "inverse_vol": "M1_inverse_vol", "risk_parity": "M2_risk_parity",
          "mean_cvar": "M9_mean_cvar", "hrp": "M11_hrp", "herc": "M12_herc"}
BOOKS = tuple(LADDER) + ("mvo",)


@dataclass
class Context:
    """What an allocator may use: the bundle, the config, and (when the pipeline built them) forecasts and regimes."""

    bundle: object
    config: object
    forecasts: object | None = None
    regimes: object | None = None
    models: list = field(default_factory=list)
    cache: dict = field(default_factory=dict)

    def constraints(self) -> Constraints:
        return Constraints.from_config(self.config, group_map=self.bundle.group or None)

    def transform(self) -> dict:
        node = self.config.get("strategies.transform", {}) or {}
        return {"winsorize_quantile": float(node.get("winsorize_quantile", 0.02)), "cross_sectional": bool(node.get("cross_sectional", True)),
                "scale": str(node.get("scale", "zscore")), "clip": node.get("clip", 3.0),
                "target_vol": float(self.config.get("portfolio.volatility.target_vol", 0.10)),
                "max_abs_weight": float(node.get("max_abs_weight", 0.25)), "gross_leverage": float(node.get("gross_leverage", 1.0)),
                "long_only_book": bool(node.get("long_only", False))}


def book(name: str, ctx: Context) -> pd.DataFrame:
    """A named book (see ``BOOKS``), built once per context. Platform bundles reuse the on-disk ladder cache."""
    if name not in BOOKS:
        raise KeyError(f"unknown book '{name}'; choose from {BOOKS}")
    key = ("book", name)
    if key in ctx.cache:
        return ctx.cache[key]
    market = ctx.bundle.as_market()
    if name == "mvo":
        from ..portfolio.bayesian import bayesian_book

        weights = bayesian_book(ctx.bundle.returns, ctx.bundle.investable, "mvo_sample", ctx.constraints(), rebalance_dates(ctx.bundle.index, "monthly"))
    else:
        from experiments.strategies import build_ladder, cached_ladder          # the Generation 1-2 builders live with their stage code

        try:
            weights = cached_ladder(market, ctx.config, ctx.config.path("processed"), [LADDER[name]])[LADDER[name]]
        except Exception:
            weights = build_ladder(market, ctx.config, [LADDER[name]])[LADDER[name]]
    ctx.cache[key] = weights.reindex(ctx.bundle.index).ffill()
    return ctx.cache[key]


class Allocator:
    name = "allocator"

    def build(self, ctx: Context) -> pd.DataFrame:
        raise NotImplementedError


@register_allocator("static", "One of the existing books: equal_weight, inverse_vol, risk_parity, mean_cvar, hrp, herc, mvo")
class StaticAllocator(Allocator):
    def __init__(self, book: str = "risk_parity"):
        self.book = book

    def build(self, ctx: Context) -> pd.DataFrame:
        return book(self.book, ctx)


@register_allocator("forecast_stack", "The Generation 1 position stack (winsorise, z-score, clip, risk-scale, cap) applied to mean / std of the combined forecast")
class ForecastStack(Allocator):
    def __init__(self, mode: str = "cross_sectional", long_only: bool = False, gross: float = 1.0, max_weight: float = 0.25, target_vol: float | None = None):
        self.mode, self.long_only, self.gross, self.max_weight, self.target_vol = mode, long_only, gross, max_weight, target_vol

    def build(self, ctx: Context) -> pd.DataFrame:
        if ctx.forecasts is None:
            raise ValueError("forecast_stack needs forecasts")
        fc = ctx.forecasts
        signal = fc.mean / fc.std.where(fc.std > 0)
        transform = ctx.transform()
        transform.update(gross_leverage=self.gross, max_abs_weight=self.max_weight, long_only_book=self.long_only)
        if self.target_vol is not None:
            transform["target_vol"] = self.target_vol
        if self.mode == "time_series":
            transform.update(cross_sectional=False, scale="none")
        elif self.mode != "cross_sectional":
            raise ValueError("mode must be 'cross_sectional' or 'time_series'")
        volatility = rolling_volatility(ctx.bundle.returns, int(ctx.config.get("portfolio.volatility.lookback", 63)))
        return signal_to_positions(signal, volatility, investable=ctx.bundle.investable, **transform)


@register_allocator("score_stack", "The Generation 1 position stack applied to a single model's raw score (no forecast layer): reproduces the Generation 1 signal books")
class ScoreStack(Allocator):
    def __init__(self, mode: str = "cross_sectional"):
        self.mode = mode

    def build(self, ctx: Context) -> pd.DataFrame:
        if len(ctx.models) != 1:
            raise ValueError("score_stack takes exactly one model")
        signal = ctx.models[0].score(ctx.bundle).where(ctx.bundle.investable)
        transform = ctx.transform()
        if self.mode == "time_series":
            transform.update(cross_sectional=False, scale="none")
        volatility = rolling_volatility(ctx.bundle.returns, int(ctx.config.get("portfolio.volatility.lookback", 63)))
        return signal_to_positions(signal, volatility, investable=ctx.bundle.investable, **transform)


@register_allocator("confidence", "Sign of the forecast times its confidence, gross-normalised and capped; forecasts below a confidence floor get nothing")
class ConfidenceSized(Allocator):
    def __init__(self, min_confidence: float = 0.0, gross: float = 1.0, max_weight: float = 0.25, long_only: bool = False):
        self.min_confidence, self.gross, self.max_weight, self.long_only = min_confidence, gross, max_weight, long_only

    def build(self, ctx: Context) -> pd.DataFrame:
        fc = ctx.forecasts
        raw = np.sign(fc.mean) * fc.confidence.where(fc.confidence >= self.min_confidence)
        raw = raw.where(ctx.bundle.investable.reindex_like(raw).fillna(False))
        if self.long_only:
            raw = raw.clip(lower=0.0)
        weights = normalise_gross(raw.fillna(0.0), self.gross)
        return apply_weight_cap(weights, self.max_weight, self.gross).fillna(0.0)


def _tilted(base: pd.DataFrame, tilt: dict, cap: float | None = None) -> pd.DataFrame:
    """Add the tilt (assets -> weight) to a base book and rescale to the base book's own gross exposure."""
    out = base.copy()
    gross = base.abs().sum(axis=1)
    for asset, extra in tilt.items():
        if asset in out.columns:
            out[asset] = out[asset] + float(extra)
    rescaled = out.div(out.abs().sum(axis=1).replace(0.0, np.nan), axis=0).mul(gross, axis=0).fillna(0.0)
    return rescaled


@register_allocator("regime_switch", "A probability-weighted blend of other allocators, one per regime, with optional commodity/asset tilts")
class RegimeSwitch(Allocator):
    """``rules`` maps a regime name to ``{"book": name}`` (a static book), ``{"allocator": name, "params": {...}}``, or either of those with
    ``"tilt": {asset: extra weight}``. ``default`` is used for regimes with no rule. ``mode='soft'`` blends by probability,
    ``'hard'`` uses the single most probable regime."""

    def __init__(self, rules: dict, default: dict | None = None, mode: str = "soft"):
        self.rules, self.default, self.mode = rules, default or {"book": "risk_parity"}, mode

    def _source(self, spec: dict, ctx: Context) -> pd.DataFrame:
        if "allocator" in spec:
            base = ALLOCATORS.create(spec["allocator"], **(spec.get("params") or {})).build(ctx)
        else:
            base = book(spec.get("book", "risk_parity"), ctx)
        return _tilted(base, spec["tilt"]) if spec.get("tilt") else base

    def build(self, ctx: Context) -> pd.DataFrame:
        if ctx.regimes is None:
            raise ValueError("regime_switch needs regimes")
        probs = ctx.regimes.probabilities.reindex(ctx.bundle.index)
        if self.mode == "hard":
            hard = probs.eq(probs.max(axis=1), axis=0).astype(float).where(probs.notna())
            probs = hard.div(hard.sum(axis=1), axis=0)
        elif self.mode != "soft":
            raise ValueError("mode must be 'soft' or 'hard'")
        columns = ctx.bundle.assets
        index = ctx.bundle.index
        out = pd.DataFrame(0.0, index=index, columns=columns)
        covered = pd.Series(0.0, index=index)
        sources: dict[str, pd.DataFrame] = {}
        for regime in probs.columns:
            spec = self.rules.get(regime, self.default)
            key = repr(sorted(spec.items(), key=lambda kv: kv[0]))
            if key not in sources:
                sources[key] = self._source(spec, ctx).reindex(index=index, columns=columns).fillna(0.0)
            p = probs[regime].fillna(0.0)
            out = out + sources[key].mul(p, axis=0)
            covered = covered + p
        # Before the detector has a view (its burn-in), fall back to the default rule rather than holding nothing.
        fallback_key = repr(sorted(self.default.items(), key=lambda kv: kv[0]))
        if fallback_key not in sources:
            sources[fallback_key] = self._source(self.default, ctx).reindex(index=index, columns=columns).fillna(0.0)
        out = out + sources[fallback_key].mul((1.0 - covered).clip(0.0, 1.0), axis=0)
        return out
