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
from ..backtest.impact import market_state
from ..signals.alpha_engine import trailing_sharpe, trust_weights
from ..utils.dates import rebalance_dates
from ..utils.logging import get_logger
from .allocation import ALLOCATORS, Context
from .data import MarketBundle
from .adaptive import RegimeRiskLimits, calibrate_confidence, decay_trust_weights, regime_spread_scale
from .forecasting import combine_forecasts, ic_trust_weights, regime_trust_weights
from .ic_combination import optimal_ic_weights, orthogonal_ic_forecast
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
    forecast: dict = field(default_factory=dict)                # {"regime_spread": {...}, "confidence": {"method": "platt"|"isotonic"|"none"}}
    allocation: dict = field(default_factory=dict)              # {"allocator": ..., "params": {...}}
    risk: dict = field(default_factory=dict)                    # {"mode": none|constant|regime, "limits": {"gross_caps": {...}, "drawdown": {...}}}
    execution: dict = field(default_factory=dict)               # {"rebalance": ..., "signal_lag": ..., "aum": ...}
    evaluation: dict = field(default_factory=dict)              # {"benchmarks": [...], "n_trials": 1, "causality": True, "start": None, "explain": False, "brinson": True, "model_attribution": True}
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


_FREQUENCY_ORDER = {"daily": 0, "weekly": 1, "monthly": 2}


def _engine(config, spec: PipelineSpec, models=()) -> BacktestEngine:
    """The backtest engine for a spec. A rule that acts on a daily or weekly signal declares so (``ForecastModel.rebalance``); the fastest such hint is used unless the spec says otherwise."""
    engine = BacktestEngine.from_config(config)
    overrides = {}
    hints = [m.rebalance for m in models if getattr(m, "rebalance", None) in _FREQUENCY_ORDER]
    if spec.execution.get("rebalance"):
        overrides["rebalance"] = spec.execution["rebalance"]
    elif hints:
        overrides["rebalance"] = min(hints, key=_FREQUENCY_ORDER.get)
    if spec.execution.get("signal_lag") is not None:
        overrides["signal_lag"] = int(spec.execution["signal_lag"])
    if spec.execution.get("min_assets") is not None:                       # the platform default of 5 zeroes a book of fewer than five assets
        overrides["min_assets"] = int(spec.execution["min_assets"])
    return replace(engine, **overrides) if overrides else engine


def _position_mode(models: list) -> str:
    modes = {getattr(m, "position_mode", "cross_sectional") for m in models}
    return modes.pop() if len(modes) == 1 else "cross_sectional"


def allocator_choice(allocation: dict | None, models: list) -> tuple[str, dict]:
    """The allocator a spec ends up with, and its parameters: the one named in ``allocation``, else the model's own weights (a structured model), else the forecast stack in the models' position mode."""
    node = dict(allocation or {})
    structured = len(models) == 1 and getattr(models[0], "structured", False)
    name = node.get("allocator") or ("model_weights" if structured else "forecast_stack" if models else "static")
    params = dict(node.get("params") or {})
    if name in ("forecast_stack", "score_stack"):
        params.setdefault("mode", _position_mode(models))
    if name == "static":
        params.setdefault("book", "risk_parity")
    return name, params


class Pipeline:
    def __init__(self, spec: PipelineSpec | dict, config, bundle: MarketBundle):
        self.spec = spec if isinstance(spec, PipelineSpec) else PipelineSpec.from_dict(spec)
        self.config, self.bundle = config, bundle
        self._decay_table = None
        self._ic_table = None
        from . import load_library

        load_library()                                          # idempotent: the shipped strategies are always available to a Pipeline
        self._rule = None

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

    def _standalone(self, panels: dict, models, ctx_base: dict, engine: BacktestEngine, gross: bool = False) -> pd.DataFrame:
        streams = {}
        for m in models:
            ctx = Context(self.bundle, self.config, forecasts=panels[m.name])
            w = ALLOCATORS.create("forecast_stack", mode=getattr(m, "position_mode", "cross_sectional")).build(ctx)
            res = engine.run(w, self.bundle.returns, m.name, self.bundle.investable, apply_vol_target=True)
            streams[m.name] = res.gross_returns if gross else res.net_returns
        return pd.DataFrame(streams)

    def _combine(self, panels: dict, models, engine: BacktestEngine, regimes=None) -> tuple[ForecastPanel, pd.DataFrame | None]:
        rule = str(self.spec.combination.get("rule", "equal"))
        self._rule = rule
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
        if rule == "decay_weighted":
            node = self.spec.combination
            w, decay = decay_trust_weights(panels, self.bundle.returns, tuple(node.get("lags", (1, 5, 10, 21, 42, 63))), int(node.get("window", 756)),
                                           int(node.get("min_obs", 252)), int(node.get("holding", 21)))
            self._decay_table = decay
            return combine_forecasts(panels, "given", w), w
        if rule in ("optimal_ic", "orthogonal_ic"):
            node = self.spec.combination
            window, min_obs, shrink = int(node.get("window", 504)), int(node.get("min_obs", 252)), float(node.get("shrink", 0.3))
            if rule == "optimal_ic":
                w, expected = optimal_ic_weights(panels, self.bundle.returns, window, min_obs, shrink)
                share = w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / w.shape[1])
                self._ic_table = (share, expected)
                return combine_forecasts(panels, "given", w), share
            panel, w, expected = orthogonal_ic_forecast(panels, self.bundle.returns, str(node.get("orthogonalize", "gram_schmidt")), window, min_obs, shrink)
            self._ic_table = (w, expected)
            return panel, w
        if rule == "cost_aware":
            standalone = self._standalone(panels, models, {}, engine)
            window, min_hist = int(self.spec.combination.get("window", 504)), int(self.spec.combination.get("min_history", 252))
            scores = pd.DataFrame({n: trailing_sharpe(standalone[n], window, min_hist) for n in standalone})
            w = trust_weights(scores, rebalance_dates(self.bundle.index, "monthly"), shrink=float(self.spec.combination.get("shrink", 0.5)))
            return combine_forecasts(panels, "given", w), w
        raise ValueError(f"unknown combination rule '{rule}'")

    def _adjust_forecast(self, combined: ForecastPanel, regimes) -> tuple[ForecastPanel, dict]:
        """Regime-conditional spread, then calibrated confidence, as declared in ``spec.forecast`` (both optional)."""
        node = dict(self.spec.forecast or {})
        unknown = set(node) - {"regime_spread", "confidence"}
        if unknown:
            raise ValueError(f"unknown forecast options {sorted(unknown)}; allowed: regime_spread, confidence")
        out: dict = {}
        if node.get("regime_spread") is not None:
            opts = dict(node["regime_spread"] or {})
            combined, scale = regime_spread_scale(combined, self.bundle.returns, regimes, int(opts.get("min_obs", 1000)), float(opts.get("shrink", 0.5)))
            out["regime_spread_scale"] = scale
        if node.get("confidence") is not None:
            opts = dict(node["confidence"] or {})
            combined = calibrate_confidence(combined, self.bundle.returns, str(opts.get("method", "platt")), int(opts.get("min_obs", 2000)), int(opts.get("refit_every", 21)))
        return combined, out

    def _allocator(self, models):
        name, params = allocator_choice(self.spec.allocation, models)
        return name, ALLOCATORS.create(name, **params)

    # ------------------------------------------------------------------------------------------- run
    def run(self, validate: bool = True, precomputed: dict | None = None) -> PipelineResult:
        """Run the whole pipeline. ``precomputed`` lets a caller that has already computed the models' forecasts reuse them (e.g. to compare combination rules)."""
        t0 = time.perf_counter()
        spec, bundle, config = self.spec, self.bundle, self.config
        timings: dict = {}
        models = self._models()
        engine = _engine(config, spec, models)
        regimes = None
        if spec.regime:
            regimes = DETECTORS.create(spec.regime["detector"], **(spec.regime.get("params") or {})).detect(bundle)
        timings["regime"] = time.perf_counter() - t0

        panels, combined, trust = {}, None, None
        if models:
            panels = precomputed if precomputed is not None else self._forecasts(models)
            combined, trust = self._combine(panels, models, engine, regimes)
            combined, adjustments = self._adjust_forecast(combined, regimes)
        else:
            adjustments = {}
        timings["forecast"] = time.perf_counter() - t0

        allocator_name, allocator = self._allocator(models)
        ctx = Context(bundle, config, forecasts=combined, regimes=regimes, models=models)
        targets = allocator.build(ctx)
        timings["allocation"] = time.perf_counter() - t0

        risk = dict(spec.risk or {})
        unknown = set(risk) - {"mode", "target_vol", "regime_targets", "default_target", "lookback", "max_leverage", "limits"}
        if unknown:
            raise ValueError(f"unknown risk options {sorted(unknown)}")
        mode = risk.get("mode") or ("constant" if allocator_name in ("forecast_stack", "confidence", "score_stack", "model_weights") else "none")
        limits = risk.get("limits")
        scalar, limit_diag = None, None
        apply_vol_target = False
        if mode == "constant" and limits:
            # limits act on FINAL weights, so the constant volatility target is applied explicitly here instead of inside the engine
            constant = float(risk["target_vol"]) if risk.get("target_vol") is not None else float(config.get("portfolio.volatility.target_vol", 0.10))
            policy = RegimeRiskPolicy(targets={}, default_target=constant, lookback=int(risk.get("lookback", 63)), max_leverage=float(risk.get("max_leverage", 1.5)))
            targets, scalar, _ = policy.apply(targets, bundle.returns, None)
        elif mode == "constant":
            if risk.get("target_vol") is not None:
                engine = replace(engine, target_vol=float(risk["target_vol"]))
            apply_vol_target = True
        elif mode == "regime":
            policy = RegimeRiskPolicy(targets=risk.get("regime_targets") or RegimeRiskPolicy().targets, default_target=float(risk.get("default_target", 0.10)),
                                      lookback=int(risk.get("lookback", 63)), max_leverage=float(risk.get("max_leverage", 1.5)))
            targets, scalar, _ = policy.apply(targets, bundle.returns, regimes)
        elif mode != "none":
            raise ValueError("risk.mode must be none, constant or regime")
        if limits:
            unknown_limits = set(limits) - {"gross_caps", "drawdown"}
            if unknown_limits:
                raise ValueError(f"unknown risk limits {sorted(unknown_limits)}; allowed: gross_caps, drawdown")
            targets, limit_diag = RegimeRiskLimits(dict(limits.get("gross_caps") or {}), dict(limits.get("drawdown") or {})).apply(targets, bundle.returns, regimes)

        result = engine.run(targets, bundle.returns, spec.name, bundle.investable, apply_vol_target=apply_vol_target)
        if spec.execution.get("aum"):
            from ..backtest.impact import ImpactSettings, impact_net_returns

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
        if adjustments.get("regime_spread_scale") is not None:
            out.tables["regime_spread_scale"] = adjustments["regime_spread_scale"].loc[start:].describe().to_frame("regime_spread_scale")
        if getattr(self, "_decay_table", None) is not None:
            out.tables["alpha_decay"] = self._decay_table.groupby("model").agg(ic0_mean=("ic0", "mean"), tau_median=("tau", "median"))
        if getattr(self, "_ic_table", None) is not None:
            shares, expected = self._ic_table
            late = shares.loc[start:]
            out.tables["ic_combination"] = pd.DataFrame({"mean_weight": late.mean(), "last_weight": late.iloc[-1]})
            if expected.loc[start:].notna().any():
                out.metrics["expected_ic_ir"] = float(expected.loc[start:].mean())
        self._attribution_extras(out, models, panels, trust, engine, start)
        if spec.evaluation.get("explain"):
            for model in models:
                table = model.explain(bundle)
                if table is not None and len(table):
                    out.tables[f"explain_{model.name}"] = table
        if limit_diag is not None:
            ld = limit_diag.loc[start:]
            out.tables["risk_limits"] = pd.DataFrame({"share_of_days": {"gross cap binding": float((ld["cap_factor"] < 1.0 - 1e-9).mean()),
                                                                          "drawdown limit on": float(ld["drawdown_limit_on"].mean())},
                                                       "mean_exposure_factor": {"gross cap binding": float(ld["cap_factor"].mean()), "drawdown limit on": float(ld["drawdown_factor"].mean())}})
            out.metrics["risk_limit_days_share"] = float(((ld["cap_factor"] < 1.0 - 1e-9) | (ld["drawdown_limit_on"] > 0)).mean())
        timings["total"] = time.perf_counter() - t0
        return out

    def _attribution_extras(self, out, models, panels, trust, engine, start) -> None:
        """Brinson against equal weight, model-level attribution, cost split and capacity: the integrated attribution and execution reports."""
        from .analytics import DEFAULT_GRID, brinson_vs_benchmark, capacity_by_aum, cost_breakdown, model_attribution, standalone_shares

        opts = self.spec.evaluation
        if opts.get("brinson", True):
            table = brinson_vs_benchmark(out, self.bundle)
            if len(table):
                out.tables["attribution_vs_benchmark"] = table
        if len(models) > 1 and opts.get("model_attribution", True):
            try:
                shares = standalone_shares(panels, self._rule or "equal", trust)
            except ValueError:
                shares = None
            if shares is not None:
                out.tables["attribution_by_model"] = model_attribution(out, self._standalone(panels, models, {}, engine, gross=True), shares)
        aum = self.spec.execution.get("aum")
        if aum or self.spec.execution.get("capacity"):
            from ..backtest.impact import ImpactSettings

            volume = self.bundle.volume if self.bundle.volume is not None else self.bundle.prices * 0.0 + 1e9
            sigma, adv = market_state(self.bundle.prices, volume, self.bundle.returns)
            coefficient = float(self.spec.execution.get("impact_coefficient", 1.0))
            if aum:
                out.tables["cost_breakdown"] = cost_breakdown(out, engine.cost_model.rates(out.trades.columns), sigma, adv, float(aum), ImpactSettings(coefficient))
            if self.spec.execution.get("capacity"):
                grid = self.spec.execution["capacity"] if isinstance(self.spec.execution["capacity"], (list, tuple)) else DEFAULT_GRID
                curve, cap = capacity_by_aum(out, self.bundle, engine, tuple(float(g) for g in grid), coefficient)
                out.tables["capacity"] = curve.rename("net_sharpe").to_frame()
                out.metrics["capacity_usd"] = float(cap) if isinstance(cap, (int, float)) else float("nan")
                out.metrics["capacity_note"] = cap if isinstance(cap, str) else ""

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
