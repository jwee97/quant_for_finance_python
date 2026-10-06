"""The pipeline: market -> regime -> forecasts -> confidence -> combination -> portfolio -> risk -> execution -> validation -> attribution.

One ``PipelineSpec`` (a plain dict or a YAML file) describes a strategy; ``Pipeline.run`` produces a ``PipelineResult``
with the net returns, the weights, the regime path, the forecasts, the metrics and the validation battery. Adding a strategy
is writing a ``ForecastModel`` and naming it in ``models``.

Every step is causal. ``validate.check_causality`` proves it for each model by replacing all data after a cutoff with noise
and requiring the forecasts up to the cutoff to be unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..backtest.engine import BacktestEngine
from ..backtest.metrics import deflated_sharpe_ratio, performance_summary
from ..signals.alpha_engine import forward_returns, matured_ic, trailing_sharpe, trust_weights
from ..utils.dates import DateWindow, rebalance_dates
from ..utils.logging import get_logger
from ..validation.robustness import paired_sharpe_test
from .allocation import ALLOCATORS, BOOKS, Context
from .data import MarketBundle
from .forecasting import combine_forecasts, ic_trust_weights, regime_trust_weights
from .registry import DETECTORS, MODELS
from .risk import RegimeRiskPolicy
from .types import ForecastPanel

LOGGER = get_logger(__name__)
ANN = 252.0


@dataclass
class PipelineSpec:
    name: str = "strategy"
    models: list = field(default_factory=list)                 # [{"name": ..., "params": {...}}]
    combination: dict = field(default_factory=lambda: {"rule": "equal"})
    regime: dict | None = None                                  # {"detector": ..., "params": {...}}
    allocation: dict = field(default_factory=dict)              # {"allocator": ..., "params": {...}}
    risk: dict = field(default_factory=dict)                    # {"mode": none|constant|regime, ...}
    execution: dict = field(default_factory=dict)               # {"rebalance": ..., "signal_lag": ..., "aum": ...}
    evaluation: dict = field(default_factory=dict)              # {"benchmarks": [...], "n_trials": 1, "causality": True, "start": None}
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "PipelineSpec":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown spec keys {sorted(unknown)}; allowed: {sorted(known)}")
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineSpec":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class PipelineResult:
    spec: PipelineSpec
    bundle_name: str
    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.Series
    turnover: pd.Series
    regimes: object | None = None
    forecasts: ForecastPanel | None = None
    model_forecasts: dict = field(default_factory=dict)
    benchmarks: dict = field(default_factory=dict)             # name -> net returns on the evaluation window
    metrics: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)                 # name -> DataFrame (attribution, by-regime, by-sample, forecast quality)
    risk_scalar: pd.Series | None = None
    start: pd.Timestamp | None = None
    timings: dict = field(default_factory=dict)

    @property
    def window(self) -> pd.Series:
        return self.net_returns.loc[self.start:] if self.start is not None else self.net_returns


def _engine(config, spec: PipelineSpec) -> BacktestEngine:
    engine = BacktestEngine.from_config(config)
    overrides = {}
    if spec.execution.get("rebalance"):
        overrides["rebalance"] = spec.execution["rebalance"]
    if spec.execution.get("signal_lag") is not None:
        overrides["signal_lag"] = int(spec.execution["signal_lag"])
    return replace(engine, **overrides) if overrides else engine


def _position_mode(models: list) -> str:
    modes = {getattr(m, "position_mode", "cross_sectional") for m in models}
    return modes.pop() if len(modes) == 1 else "cross_sectional"


class Pipeline:
    def __init__(self, spec: PipelineSpec | dict, config, bundle: MarketBundle):
        self.spec = spec if isinstance(spec, PipelineSpec) else PipelineSpec.from_dict(spec)
        self.config, self.bundle = config, bundle

    # ------------------------------------------------------------------------------------------ steps
    def _models(self):
        out = []
        for item in self.spec.models:
            item = {"name": item} if isinstance(item, str) else item
            model = MODELS.create(item["name"], **(item.get("params") or {}))
            model.require(self.bundle)
            out.append(model)
        return out

    def _forecasts(self, models) -> dict:
        node = self.config.get("framework.forecast", {}) or {}
        calibration = node.get("calibration", {}) or {}
        return {m.name: m.forecast(self.bundle, int(calibration.get("min_observations", 504)), bool(calibration.get("allow_negative_slope", False)),
                                   float(node.get("volatility_halflife_days", 40.0))) for m in models}

    def _standalone(self, panels: dict, models, ctx_base: dict, engine: BacktestEngine) -> pd.DataFrame:
        streams = {}
        for m in models:
            ctx = Context(self.bundle, self.config, forecasts=panels[m.name])
            w = ALLOCATORS.create("forecast_stack", mode=getattr(m, "position_mode", "cross_sectional")).build(ctx)
            streams[m.name] = engine.run(w, self.bundle.returns, m.name, self.bundle.investable, apply_vol_target=True).net_returns
        return pd.DataFrame(streams)

    def _combine(self, panels: dict, models, engine: BacktestEngine, regimes=None) -> tuple[ForecastPanel, pd.DataFrame | None]:
        rule = str(self.spec.combination.get("rule", "equal"))
        if len(panels) == 1:
            return next(iter(panels.values())), None
        if rule in ("equal", "confidence", "precision"):
            return combine_forecasts(panels, rule), None
        if rule == "ic_weighted":
            w = ic_trust_weights(panels, self.bundle.returns)
            return combine_forecasts(panels, "given", w), w
        if rule == "regime_conditional":
            if regimes is None:
                raise ValueError("the regime_conditional rule needs a regime detector in the spec")
            standalone = self._standalone(panels, models, {}, engine)
            node = self.spec.combination
            w = regime_trust_weights(standalone, regimes.hard_labels(), rebalance_dates(self.bundle.index, "monthly"), int(node.get("min_regime_days", 126)),
                                     float(node.get("shrink", 0.5)))
            return combine_forecasts(panels, "given", w), w
        if rule == "cost_aware":
            standalone = self._standalone(panels, models, {}, engine)
            window, min_hist = int(self.spec.combination.get("window", 504)), int(self.spec.combination.get("min_history", 252))
            scores = pd.DataFrame({n: trailing_sharpe(standalone[n], window, min_hist) for n in standalone})
            w = trust_weights(scores, rebalance_dates(self.bundle.index, "monthly"), shrink=float(self.spec.combination.get("shrink", 0.5)))
            return combine_forecasts(panels, "given", w), w
        raise ValueError(f"unknown combination rule '{rule}'")

    def _allocator(self, models):
        node = dict(self.spec.allocation or {})
        structured = len(models) == 1 and getattr(models[0], "structured", False)
        name = node.get("allocator") or ("model_weights" if structured else "forecast_stack" if models else "static")
        params = dict(node.get("params") or {})
        if name in ("forecast_stack", "score_stack"):
            params.setdefault("mode", _position_mode(models))
        if name == "static":
            params.setdefault("book", "risk_parity")
        return name, ALLOCATORS.create(name, **params)

    # ------------------------------------------------------------------------------------------- run
    def run(self, validate: bool = True, precomputed: dict | None = None) -> PipelineResult:
        """Run the whole pipeline. ``precomputed`` lets a caller that has already computed the models' forecasts reuse them (e.g. to compare combination rules)."""
        t0 = time.perf_counter()
        spec, bundle, config = self.spec, self.bundle, self.config
        timings: dict = {}
        engine = _engine(config, spec)
        models = self._models()
        regimes = None
        if spec.regime:
            regimes = DETECTORS.create(spec.regime["detector"], **(spec.regime.get("params") or {})).detect(bundle)
        timings["regime"] = time.perf_counter() - t0

        panels, combined, trust = {}, None, None
        if models:
            panels = precomputed if precomputed is not None else self._forecasts(models)
            combined, trust = self._combine(panels, models, engine, regimes)
        timings["forecast"] = time.perf_counter() - t0

        allocator_name, allocator = self._allocator(models)
        ctx = Context(bundle, config, forecasts=combined, regimes=regimes, models=models)
        targets = allocator.build(ctx)
        timings["allocation"] = time.perf_counter() - t0

        risk = dict(spec.risk or {})
        mode = risk.get("mode") or ("constant" if allocator_name in ("forecast_stack", "confidence", "score_stack", "model_weights") else "none")
        scalar = None
        apply_vol_target = False
        if mode == "constant":
            if risk.get("target_vol") is not None:
                engine = replace(engine, target_vol=float(risk["target_vol"]))
            apply_vol_target = True
        elif mode == "regime":
            policy = RegimeRiskPolicy(targets=risk.get("regime_targets") or RegimeRiskPolicy().targets, default_target=float(risk.get("default_target", 0.10)),
                                      lookback=int(risk.get("lookback", 63)), max_leverage=float(risk.get("max_leverage", 1.5)))
            targets, scalar, _ = policy.apply(targets, bundle.returns, regimes)
        elif mode != "none":
            raise ValueError("risk.mode must be none, constant or regime")

        result = engine.run(targets, bundle.returns, spec.name, bundle.investable, apply_vol_target=apply_vol_target)
        if spec.execution.get("aum"):
            from ..backtest.impact import ImpactSettings, impact_net_returns, market_state

            sigma, adv = market_state(bundle.prices if bundle.volume is not None else bundle.prices, bundle.volume if bundle.volume is not None else bundle.prices * 0.0 + 1e9,
                                      bundle.returns)
            net, daily, _ = impact_net_returns(result.gross_returns, result.trades, float(spec.execution["aum"]), sigma, adv,
                                               engine.cost_model.rates(result.trades.columns), ImpactSettings(float(spec.execution.get("impact_coefficient", 1.0))))
            net_returns, costs = net.dropna(), daily
        else:
            net_returns, costs = result.net_returns, result.costs
        timings["backtest"] = time.perf_counter() - t0

        start = self._evaluation_start(combined, targets, net_returns)
        out = PipelineResult(spec, bundle.name, net_returns, result.gross_returns, result.weights, result.trades, costs, result.turnover, regimes, combined,
                             panels, {}, {}, {}, {}, scalar, start, timings)
        from .validate import evaluate
        evaluate(out, self, engine, ctx, models, validate, trust)
        timings["total"] = time.perf_counter() - t0
        return out

    def _evaluation_start(self, forecasts, targets, net_returns) -> pd.Timestamp:
        candidates = [net_returns.index[0]]
        if forecasts is not None and forecasts.first_valid() is not None:
            candidates.append(forecasts.first_valid())
        defined = targets.notna().any(axis=1)                  # the first date the allocator has a view; a flat book earns zero, it is not "missing"
        if defined.any():
            candidates.append(defined.idxmax())
        if self.spec.evaluation.get("start"):
            candidates.append(pd.Timestamp(self.spec.evaluation["start"]))
        return max(candidates)
