"""The Machine learning workspace's endpoints.

Nothing here trains a model on its own. Every action goes through ``src.ml``: a dataset is built without look-ahead, a model is fitted walk-forward, its predictions become positions, the positions are
backtested with costs by the platform's engine, and the result is split into train / validation / test / forward segments, with the number of times the same data has been looked at recorded beside it.

Each method takes a plain dictionary (the query string of a GET or the JSON body of a POST) and returns a JSON-able one. Anything that takes more than a moment (building a dataset, training, searching,
re-backtesting, meta-labelling, recording a forward signal) is submitted as a job and polled through ``/api/job/<id>`` like the Backtest tab's. Experiments, models and datasets are kept in the registry
under ``<root>/artifacts/ml`` (SQLite for the records, files for the artifacts); see ``src/ml/registry.py``.
"""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from ..assets import common
from ..ml import diagnostics as dx
from ..ml import ensembles as en
from ..ml import execution as xexec
from ..ml import explainability as xp
from ..ml import frozen_eval as fe
from ..ml import gates
from ..ml import governance as gv
from ..ml import monitoring as mon
from ..ml import null_controls as nc
from ..ml import optimization as opt
from ..ml import policy as pol
from ..ml import research_controls as rc
from ..ml import selection as sel
from ..ml import strategy_search as ss
from ..ml.bundle import InferenceBundle
from ..ml.config import build_key, experiment_config, load_ml_config
from ..ml.datasets import UNIVERSES, build_dataset, feature_catalog, resolve_bundle
from ..ml.experiments import RL_KINDS, load_run, rebacktest, run_experiment
from ..ml.features import GROUPS
from ..ml.models import Model, ModelUnavailable, default_refit, model_catalog
from ..ml.registry import STAGES, Registry
from ..ml.signals import METHODS as SIGNAL_METHODS
from ..ml import comparison as cmpn, r6_reports as r6
from ..ml import delay_report as dly
from ..ml.ensemble_models import parse_members
from ..ml.targets import CLASSIFICATION
from ..ml.targets import KINDS as TARGET_KINDS
from ..ml.training import plan_for
from ..ml.visualization import jsonable, leaderboard_payload, render_html, shap_payload
from ..utils.optional import unavailable_reason
from . import tuning
from .api import ApiError
from .ml_advanced_api import AdvancedRoutes
from .payload import clean
from .universe import MAX_TICKERS, UniverseError, normalise

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
KEY = re.compile(r"^[a-z_]{1,20}\.[a-z_0-9]{1,30}$")
PARAM = re.compile(r"^[a-z_][a-z_0-9]{0,30}$")
LIMITS = {"null_replications": 1000, "null_trials": 12, "plan_candidates": 6, "search_random": 30, "search_bayes": 30, "search_generations": 4, "search_population": 10, "search_top_k": 4, "search_kinds": 8, "tune_trials": 40, "compare": 8, "combine": 12}
# How long to look when tuning one model, for a person who does not want to choose a number of trials: each is a full walk-forward experiment, so a try takes tens of seconds, not one.
TUNE_EFFORT = {"quick": (8, "8 tries: a first look, a few minutes"), "standard": (16, "16 tries: enough for a model with two or three settings"), "thorough": (30, "30 tries: for several settings; this can take half an hour")}
TUNE_TEXT = {"alpha": "How strongly the model is held back from fitting noise: higher gives simpler, steadier fits", "l1_ratio": "How much of the penalty switches features off altogether (1) rather than shrinking them (0)",
             "max_depth": "How deep each tree may grow: deeper trees fit more detail and overfit more easily", "min_samples_leaf": "How many examples a leaf of a tree must hold: more makes steadier, blunter trees",
             "max_features": "How many features each split or each filter step may look at", "learning_rate": "How big each learning step is: lower is slower and steadier", "max_iter": "How many trees are added",
             "n_estimators": "How many trees are added", "l2_regularization": "A penalty on large leaf values: higher is steadier", "reg_lambda": "A penalty on large leaf values: higher is steadier",
             "num_leaves": "How many leaves each tree may have", "q": "How fast the Kalman filter lets the model's coefficients drift: higher follows the data closely, lower assumes steadier relationships",
             "forgetting": "How fast old data is forgotten: 0.99 remembers about 100 days, 0.9999 about 10,000", "eta": "How big each learning step is", "l2": "A penalty on large coefficients",
             "horizon": "How many days ahead the model predicts", "method": "How predictions become positions", "n": "How many assets are held on each side", "holding_period": "Days a position is held before it is reconsidered",
             "rebalance": "How often positions are changed"}

GROUP_TEXT = {"price": "Trailing returns, distance from highs and lows, moving-average gaps", "momentum": "Return over the last 1, 3, 6 and 12 months, skipping the most recent month", "mean_reversion": "Short-term reversal and distance from the mean in standard deviations",
              "volatility": "Realised volatility over several windows, its ratio, range-based estimates, downside volatility", "volume": "Dollar volume and its surprise; only where the data has volume",
              "relative": "Each asset's return and volatility relative to the universe", "macro": "Rates, spreads and the volatility index at the date, lagged by their publication delay",
              "futures": "Carry, roll yield and curve slope; only for the futures universe", "options": "Implied volatility level, skew and term structure; only where the platform has the index",
              "crypto": "Funding, basis and stablecoin growth; only for the crypto universe"}
SIGNAL_TEXT = {"top_n_long_short": "Long the n highest scores, short the n lowest, dollar neutral", "long_only_top_quantile": "Long the top quantile of scores, equal weight", "proportional": "Weight each asset in proportion to its score",
               "vol_scaled": "Proportional to score divided by volatility, so each position carries similar risk", "rank_normalised": "Weight by the cross-sectional rank of the score (robust to outliers)",
               "threshold": "Long above the threshold, short below its negative", "regime_filtered": "Trade the book only in the calm regime of the dataset's regime detector",
               "confidence_weighted": "Proportional to score times the model's confidence (for a classifier, |2p - 1|)",
               "uncertainty_weighted": "Score divided by the model's own predictive standard deviation (needs a model that reports one), with views on unusually uncertain assets shrunk by kappa",
               "uncertainty_mean_variance": "The same uncertainty-adjusted views through the inverse covariance of trailing returns: the covariance-aware version",
               "cost_aware_mean_variance": "The book that maximises the forecast return less risk, trading cost and turnover from the book actually held (needs a return forecast; solver status is reported)",
               "calibrated_sizing": "The score calibrated to a return by regression on settled outcomes, sized by edge, its uncertainty, risk and signal stability; flat when no edge has been shown",
               "target_weights": "The scores ARE the target weights (for the reinforcement-learning agents, which size their own positions)"}
ENSEMBLE_TEXT = {"equal": "One over N", "inverse_vol": "Weight by one over trailing volatility", "risk_parity": "Equal risk contributions from a shrunk covariance matrix", "mean_variance": "Long-only mean-variance with means shrunk halfway to their average",
                 "bayesian": "Mean-variance with Bayes-Stein shrunk means", "regime_conditioned": "Mean-variance on the days in the same volatility regime as today", "online_learning": "Multiplicative weights (Hedge): winners gain weight",
                 "bandit": "Thompson sampling: the chance, under the posterior, that each strategy is the best one", "best_trailing": "All in the strategy with the best trailing Sharpe ratio",
                 "regime_selector": "All in the strategy with the best Sharpe ratio in the current regime"}

# Which settings the browser may send for each section: ("int"|"float", low, high), ("choice", options), ("bool",) and a few special kinds.
SPEC = {
    "dataset": {"universe": ("universe",), "feature_groups": ("groups",), "target": ("target",), "horizon": ("int", 1, 252), "lag": ("int", 0, 20), "min_history": ("int", 60, 3000), "winsor": ("float", 0.0, 0.2),
                "normalize": ("choice", ("none", "cross_sectional_z", "rank")), "start": ("date",), "end": ("date",), "regime_detector": ("choice", ("vol_state", "hmm", "gmm")), "benchmark": ("benchmark",)},
    "split": {"scheme": ("choice", ("walk_forward", "expanding", "rolling")), "window": ("choice", ("expanding", "rolling")), "rolling_days": ("int", 252, 5000), "min_train_days": ("int", 126, 4000), "refit_every": ("int", 1, 252),
              "embargo": ("int", 0, 60), "validation_fraction": ("float", 0.05, 0.4), "test_fraction": ("float", 0.05, 0.4), "forward_fraction": ("float", 0.05, 0.4)},
    "signal": {"method": ("choice", SIGNAL_METHODS), "n": ("int", 1, 20), "quantile": ("float", 0.05, 0.5), "threshold": ("float", -10.0, 10.0), "gross": ("float", 0.1, 3.0), "net": ("float", -1.0, 1.0), "max_weight": ("float", 0.02, 1.0),
               "sector_cap": ("float", 0.05, 1.0), "holding_period": ("int", 1, 60), "regime_on_states": ("states",), "reinterpret_outputs": ("bool",), "kappa": ("float", 0.0, 10.0), "uncertainty_power": ("float", 0.0, 3.0), "mv_lookback": ("int", 40, 504),
               "mv_rebalance": ("int", 1, 60), "risk_aversion": ("float", 0.0, 100.0), "cov_method": ("choice", ("sample", "ewma", "lw_constant_correlation", "lw_identity", "pca_denoised", "factor")),
               "cov_window": ("int", 60, 1000), "opt_every": ("int", 1, 60), "turnover_penalty": ("float", 0.0, 10.0), "alloc_fallback": ("choice", ("none", "hold", "equal_weight")),
               "no_trade_margin": ("float", 0.0, 50.0), "no_trade_hold_margin": ("float", 0.0, 50.0), "trade_filter_k": ("float", 0.0, 10.0), "trade_filter_cost_margin": ("float", 0.0, 50.0),
               "trade_filter_window": ("int", 40, 1000), "calib_window": ("int", 60, 2000), "calib_min_t": ("float", 0.0, 10.0), "stability_window": ("int", 5, 250),
               "regime_exposure": ("choice", ("vol_quantile", "hmm", "markov_switching")), "regime_states": ("int", 2, 5), "regime_floor": ("float", 0.0, 1.0), "regime_cap": ("float", 0.05, 1.0),
               "regime_refit_every": ("int", 5, 252), "regime_min_days": ("int", 20, 500), "regime_min_train": ("int", 60, 1000)},
    "backtest": {"rebalance": ("choice", ("daily", "weekly", "monthly")), "cost_bps": ("float", 0.0, 250.0), "slippage_bps": ("float", 0.0, 250.0), "target_vol": ("float", 0.01, 1.0), "max_leverage": ("float", 0.5, 5.0),
                 "extra_lag": ("int", 0, 20)},
    "execution": {"mode": ("choice", ("vectorized_legacy", "realistic_cash_equity")), "capital": ("float", 1_000.0, 1e12), "max_participation": ("float", 0.001, 1.0), "adv_window": ("int", 5, 250),
                  "commission_bps": ("float", 0.0, 250.0), "half_spread_bps": ("float", 0.0, 250.0), "slippage_bps": ("float", 0.0, 250.0), "impact_coefficient": ("float", 0.0, 5.0),
                  "sizing": ("choice", ("decision_close", "fill_time")), "min_trade_weight": ("float", 0.0, 0.2), "expiry_sessions": ("int", 1, 20), "cash_rate": ("float", 0.0, 0.5),
                  "financing_spread": ("float", 0.0, 0.5), "lot": ("int", 1, 100_000), "min_adv_dollars": ("float", 0.0, 1e12), "min_listing_days": ("int", 0, 5000),
                  "shorts": ("choice", ("unconstrained", "refuse"))},
}
NULLABLE = {"start", "end", "regime_detector", "sector_cap", "target_vol", "max_participation", "no_trade_margin", "no_trade_hold_margin", "trade_filter_k", "regime_exposure"}                    # an explicit null means "none", not "use the default"


def _id(q: dict, key: str = "id") -> str:
    value = str(q.get(key) or "")
    if not ID.match(value) or ".." in value:
        raise ApiError(f"{key} is not a valid identifier")
    return value


def _study(raw) -> str:
    value = str(raw or "")
    if not rc.ID.match(value):
        raise ApiError("a study name is 1 to 80 letters, digits, '.', '_', ':' or '-'")
    return value


def _only(body: dict, allowed: set, route: str) -> None:
    """Refuse a request that carries anything the route does not take: an unknown option is an error, not something to ignore."""
    extra = set(body) - allowed
    if extra:
        raise ApiError(f"{route} does not take {sorted(extra)}; it takes {sorted(allowed)}")


def _ids(q: dict, key: str = "ids", lo: int = 2, hi: int = 8) -> list[str]:
    raw = q.get(key)
    items = raw if isinstance(raw, (list, tuple)) else str(raw or "").split(",")
    out = [str(x).strip() for x in items if str(x).strip()]
    if not lo <= len(out) <= hi:
        raise ApiError(f"choose between {lo} and {hi} experiments")
    for i in out:
        if not ID.match(i) or ".." in i:
            raise ApiError(f"'{i}' is not a valid experiment id")
    return list(dict.fromkeys(out))


def _num(path: str, raw, lo: float, hi: float, integer: bool):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ApiError(f"{path} must be a number") from None
    if not np.isfinite(value) or not lo <= value <= hi:
        raise ApiError(f"{path} must be between {lo:g} and {hi:g}")
    return int(round(value)) if integer else value


def _int(q: dict, key: str, default: int, lo: int, hi: int) -> int:
    raw = q.get(key)
    return default if raw in (None, "") else _num(key, raw, lo, hi, True)


def _float(q: dict, key: str, default: float, lo: float, hi: float) -> float:
    raw = q.get(key)
    return default if raw in (None, "") else _num(key, raw, lo, hi, False)


def _date(path: str, raw) -> str:
    try:
        return str(pd.Timestamp(str(raw)).date())
    except (ValueError, TypeError):
        raise ApiError(f"{path} must be a date such as 2015-01-31") from None


@contextmanager
def _user_errors():
    """Whatever the ML modules refuse because of the request (an unknown experiment, a model that cannot run on this target, too little data) reaches the page as a message."""
    try:
        yield
    except ApiError:
        raise
    except FileNotFoundError as exc:
        raise ApiError(f"a saved file is missing ({Path(str(exc.filename or exc)).name}); the experiment may have been moved or partly deleted") from None
    except ModelUnavailable as exc:
        raise ApiError(str(exc)) from None
    except (ValueError, KeyError) as exc:
        raise ApiError(str(exc.args[0] if isinstance(exc, KeyError) and exc.args else exc)) from None


class MlApi(AdvancedRoutes):
    def __init__(self, app):
        self.app = app
        self.root = Path(app.root)
        self._registry: Registry | None = None
        self._lock = threading.Lock()
        self._heavy = threading.Lock()                       # explanations and combinations: numpy work of a few seconds, one request at a time
        self.stops: dict[str, threading.Event] = {}          # job id -> the flag that ends that tuning early

    # ------------------------------------------------------------------------------------------------------------ routes
    def get_routes(self) -> dict:
        return {"/api/ml/catalog": self.catalog, "/api/ml/experiments": self.experiments, "/api/ml/experiment": self.experiment, "/api/ml/compare": self.compare, "/api/ml/explain": self.explain,
                "/api/ml/feature-sets": self.feature_sets, "/api/ml/verification": self.verification, "/api/ml/models": self.models, "/api/ml/trials": self.trials, "/api/ml/forward": self.forward, "/api/ml/report-html": self.report_html,
                "/api/ml/lineage": self.lineage, "/api/ml/holdout": self.holdout, "/api/ml/promotion-status": self.promotion_status, "/api/ml/statistical-diagnostics": self.statistical_diagnostics,
                "/api/ml/null-controls": self.null_controls, "/api/ml/trading-diagnostics": self.trading_diagnostics, "/api/ml/validation-policy": self.validation_policy, "/api/ml/governance": self.governance_status,
                "/api/ml/alpha-diagnostics": self.alpha_diagnostics, "/api/ml/portfolio-diagnostics": self.portfolio_diagnostics, "/api/ml/improvement-experiment": self.improvement_get,
                "/api/ml/delay-report": self.delay_report, **self.advanced_get_routes()}

    def post_routes(self) -> dict:
        return {"/api/ml/dataset/build": self.dataset_build, "/api/ml/train": self.train, "/api/ml/rebacktest": self.rebacktest, "/api/ml/search": self.search, "/api/ml/tune": self.tune, "/api/ml/tune/stop": self.tune_stop,
                "/api/ml/combine": self.combine, "/api/ml/meta-label": self.meta_label, "/api/ml/forward/step": self.forward_step, "/api/ml/promote": self.promote, "/api/ml/verification/run": self.verification_run, "/api/ml/delete": self.delete,
                "/api/ml/evaluation-plan": self.evaluation_plan, "/api/ml/evaluation-plan/freeze": self.evaluation_plan_freeze, "/api/ml/evaluate": self.evaluate,
                "/api/ml/null-controls/run": self.null_controls_run, "/api/ml/selection-tests/run": self.selection_tests_run, "/api/ml/validation-policy": self.validation_policy_adopt,
                "/api/ml/statistical-diagnostics/run": self.diagnostics_run, "/api/ml/multiseed/run": self.multiseed_run, "/api/ml/governance/version": self.governance_version,
                "/api/ml/governance/action": self.governance_action, "/api/ml/improvement-experiment": self.improvement_run, **self.advanced_post_routes()}

    # ------------------------------------------------------------------------------------------------------------ plumbing
    @property
    def registry_root(self) -> Path:
        return self.root / str(load_ml_config()["registry"]["root"])

    @property
    def registry(self) -> Registry:
        with self._lock:
            if self._registry is None:
                self._registry = Registry(self.registry_root)
            return self._registry

    def _universe(self, raw):
        if isinstance(raw, str):
            if raw not in UNIVERSES:
                raise ApiError(f"universe must be one of {list(UNIVERSES)} or a list of tickers")
            return raw
        if not isinstance(raw, (list, tuple)):
            raise ApiError("universe must be a name or a list of tickers")
        try:
            tickers = normalise(raw)
            if not 2 <= len(tickers) <= MAX_TICKERS:
                raise ApiError(f"a model needs between 2 and {MAX_TICKERS} tickers")
            self.app.builder.preflight(tickers)
            waiting = self.app.builder.waiting(tickers)
        except UniverseError as exc:
            raise ApiError(str(exc)) from None
        if waiting:                                                             # a rate-limited source: the same rule as the Backtest tab, which downloads them within the limit
            raise ApiError(f"{', '.join(waiting)} {'has' if len(waiting) == 1 else 'have'} not been downloaded yet. Run a backtest once with these tickers (it downloads them), then train here.")
        return tickers

    def _clean_section(self, section: str, raw) -> dict:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ApiError(f"{section} must be an object")
        out = {}
        for key, value in raw.items():
            if key not in SPEC[section]:
                raise ApiError(f"{section} has no setting '{key}'; the settings are {sorted(SPEC[section])}")
            path, spec = f"{section}.{key}", SPEC[section][key]
            if value is None or value == "":
                if key in NULLABLE:
                    out[key] = None
                continue
            kind = spec[0]
            if kind in ("int", "float"):
                out[key] = _num(path, value, spec[1], spec[2], kind == "int")
            elif kind == "choice":
                if value not in spec[1]:
                    raise ApiError(f"{path} must be one of {list(spec[1])}")
                out[key] = value
            elif kind == "universe":
                out[key] = self._universe(value)
            elif kind == "groups":
                if not isinstance(value, list) or not value or any(g not in GROUPS for g in value):
                    raise ApiError(f"{path} must be a non-empty list drawn from {list(GROUPS)}")
                out[key] = list(dict.fromkeys(value))
            elif kind == "target":
                from ..ml.targets import parse_target

                try:
                    parse_target(str(value))
                except ValueError as exc:
                    raise ApiError(str(exc)) from None
                out[key] = str(value)
            elif kind == "date":
                out[key] = _date(path, value)
            elif kind == "benchmark":
                if not re.match(r"^[A-Za-z0-9._^=-]{1,12}$", str(value)):
                    raise ApiError(f"{path} must be equal_weight or a ticker of the universe")
                out[key] = str(value)
            elif kind == "states":
                if not isinstance(value, list) or not value or len(value) > 5:
                    raise ApiError(f"{path} must be a list of regime state numbers")
                out[key] = [_num(path, v, 0, 4, True) for v in value]
        return out

    def _overrides(self, body: dict, need_model: bool = True) -> dict:
        """The experiment configuration a request describes, as overrides of the platform's defaults: only the settings of ``SPEC`` are accepted, each checked for type and range, and the whole is then
        validated by ``ml.config.experiment_config``."""
        over: dict = {}
        for section in SPEC:
            clean_section = self._clean_section(section, body.get(section))
            if clean_section:
                over[section] = clean_section
        if "impact_coefficient" in over.get("execution", {}):                          # the one nested setting of the section, offered flat
            over["execution"]["impact"] = {"coefficient": over["execution"].pop("impact_coefficient")}
        model = body.get("model")
        if model is not None:
            if not isinstance(model, dict):
                raise ApiError("model must be an object with kind and params")
            kind = str(model.get("kind") or "ridge")
            params = model.get("params") or {}
            if not isinstance(params, dict) or len(params) > 20:
                raise ApiError("model.params must be an object of at most 20 settings")
            for k, v in params.items():
                if not PARAM.match(str(k)) or not (v is None or isinstance(v, (bool, int, float, str))):
                    raise ApiError(f"model.params.{k} must be a number, text, true/false or null")
            over["model"] = {"kind": kind, "params": dict(params)}
            known = {m["kind"]: m for m in model_catalog()}
            if kind in known and not known[kind]["available"]:
                raise ApiError(f"the {known[kind]['label']} model cannot be used here: {known[kind]['reason']}")      # refused before a job is queued, with the reason
            if kind in known and "refit_every" not in over.get("split", {}):
                over.setdefault("split", {})["refit_every"] = default_refit(kind)           # trees refit every quarter, linear models every month: the platform's own default per kind
            if kind in RL_KINDS:                                                          # an agent sizes its own book, and trains against the costs the backtest will charge
                over.setdefault("signal", {}).setdefault("method", "target_weights")
                bt = {**load_ml_config()["backtest"], **over.get("backtest", {})}
                over["model"]["params"].setdefault("cost_bps", float(bt["cost_bps"]) + float(bt["slippage_bps"]))
        elif need_model:
            over["model"] = {"kind": "ridge", "params": {}}
        if body.get("study"):
            over["study"] = _study(body["study"])
        if body.get("name"):
            over["name"] = str(body["name"])[:80]
        if body.get("notes"):
            over["notes"] = str(body["notes"])[:500]
        if body.get("seed") not in (None, ""):
            over["seed"] = _num("seed", body["seed"], 0, 1_000_000, True)
        if body.get("robustness"):
            over["robustness"] = {"enabled": True}
        try:
            experiment_config(over)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return over

    def _submit(self, label: str, work) -> dict:
        return self.app.submit_task(label, work, "ml")

    # ------------------------------------------------------------------------------------------------------------ catalogue
    def catalog(self, q: dict | None = None) -> dict:
        """Everything the workspace's forms offer: models and their settings, feature groups and features, targets, position rules, combination methods, the platform's defaults and what is already saved."""
        cfg = load_ml_config()
        features = feature_catalog()
        groups = []
        for g in GROUPS:
            names = [f for f in features if f["group"] == g]
            groups.append({"id": g, "text": GROUP_TEXT.get(g, ""), "n": len(names), "features": [{"name": f["name"], "text": f["description"], "needs": f["needs"], "broadcast": f["broadcast"]} for f in names]})
        root = self.registry_root
        counts = {"experiments": 0, "models": 0, "feature_sets": 0}
        if (root / "registry.sqlite").exists():
            reg = self.registry
            counts = {"experiments": len(reg.list_experiments(10_000)), "models": len(reg.list_models()), "feature_sets": len(reg.list_feature_sets())}
        return clean({
            "models": [{**m, "refit_every": m["refit_every"]} for m in model_catalog()], "search_kinds": list(ss.DEFAULT_KINDS), "feature_groups": groups,
            "targets": [{"kind": k, "text": v, "classification": k in CLASSIFICATION} for k, v in TARGET_KINDS.items()], "horizons": [1, 5, 10, 20, 60],
            "signals": [{"id": s, "text": SIGNAL_TEXT.get(s, "")} for s in SIGNAL_METHODS], "ensembles": [{"id": k, "text": v} for k, v in ENSEMBLE_TEXT.items()],
            "universes": [{"id": "etf", "text": "The platform's ETFs (real prices)", "source": "platform"}, {"id": "crypto", "text": "Deribit BTC and ETH with their carry indices (real data)", "source": "platform"},
                          {"id": "futures", "text": "Simulated futures roots (a Schwartz-Smith model; checks the machinery, not a market)", "source": "simulated"}],
            "tune_spaces": {m["kind"]: self._space_rows(m["kind"]) for m in model_catalog() if m["available"] and m["params"]}, "tune_efforts": [{"id": k, "trials": n, "text": t} for k, (n, t) in TUNE_EFFORT.items()],
            "defaults": {k: cfg[k] for k in ("dataset", "split", "model", "signal", "backtest", "robustness", "search")}, "stages": list(STAGES), "limits": LIMITS,
            "engines": [{"id": "builtin", "text": "The platform's own random, TPE and Bayesian searches", "available": True, "reason": None},
                        {"id": "optuna", "text": "Optuna's multivariate TPE sampler", "available": unavailable_reason("optuna") is None, "reason": unavailable_reason("optuna")}],
            "registry": {"root": str(Path(cfg["registry"]["root"])), **counts}, "as_of": str(self.app.builder.default_bundle().index[-1].date()),
            "timing": {"decision": "the close of day t", "trade": "lag days later (the engine's signal lag, 1)", "target": "from the close of t + lag to the close of t + lag + horizon",
                       "purge": "training rows whose label window overlaps a test day are dropped (lag + horizon + embargo)", "segments": "train (warm-up) / validation (selection) / test and forward (reserved, reported once)"}})

    @staticmethod
    def _space_rows(kind: str) -> list[dict]:
        """What tuning one model of this kind varies, for the form: each setting with its range or its choices and a line on what it does."""
        try:
            space = ss.kind_space(kind)
        except ValueError:
            return []
        rows = []
        for path, p in space.items():
            leaf = path.split(".")[-1]
            rows.append({"path": path, "label": leaf.replace("_", " "), "kind": p.kind, "low": p.low, "high": p.high, "choices": list(p.choices), "text": TUNE_TEXT.get(leaf, "A setting of the model"),
                         "model": path.startswith("model.params.")})
        return rows

    # ------------------------------------------------------------------------------------------------------------ datasets
    def dataset_build(self, body: dict) -> dict:
        """``POST /api/ml/dataset/build`` (a job): build and save the panel of causal features and forward-looking targets, and describe it: coverage, features skipped and why, and each feature's rank IC with
        the target over the WARM-UP segment only (looking at later segments to choose features would leak them)."""
        over = self._overrides({"dataset": body.get("dataset"), "split": body.get("split"), "name": body.get("name")}, need_model=False)
        name = str(body.get("name") or "dataset")[:60]
        extra = [t for t in (body.get("extra_targets") or []) if isinstance(t, str)][:6]
        for t in extra:
            self._clean_section("dataset", {"target": t})

        def work(progress):
            cfg = experiment_config(over)
            progress("loading the universe")
            with _user_errors():
                bundle, info = resolve_bundle(self.app.config, cfg["dataset"]["universe"], self.app.builder)
                progress("building features and targets")
                ds = build_dataset(bundle, cfg["dataset"], name, extra, info)
                fid = self.registry.register_feature_set(ds, build_key(cfg))
                return self._describe_dataset(ds, cfg, fid, info, progress)

        return self._submit(f"dataset: {name}", work)

    def _register_feature_screen(self, ds, cfg: dict, fid: str, fic, until) -> dict:
        """The look the dataset page takes at the features is adaptive feature discovery (``selection.register_screen``)."""
        return sel.register_screen(self.registry, rc.study_of(cfg), fid, fic)

    def _describe_dataset(self, ds, cfg: dict, fid: str, info: dict, progress) -> dict:
        cat = {f["name"]: f for f in feature_catalog()}
        cov = ds.X.notna().mean()
        until, plan_info, scope = None, None, "warm-up segment"
        try:
            plan, _ = plan_for(ds, cfg["split"], ds.target)
            until = plan.dates[plan.bounds["train"][1] - 1]
            plan_info = plan.to_dict()
        except ValueError as exc:                                             # too short for the split: look at the first half of the dates only, never at the later ones
            plan_info = {"error": str(exc)}
            until, scope = ds.dates[len(ds.dates) // 2], "first half of the dates"
        progress("measuring each feature's information coefficient on the warm-up segment")
        try:
            fic = xp.feature_ic(ds, until=until) if until is not None else pd.DataFrame()
        except Exception:                                                                              # a diagnostic must not lose the dataset
            fic = pd.DataFrame()
        screen = self._register_feature_screen(ds, cfg, fid, fic, until)
        rows = []
        for f in ds.features:
            c = cat.get(f, {"group": "regime", "description": "The market's regime state from the chosen detector (all assets share it)", "broadcast": True})
            r = fic.loc[f] if f in fic.index else None
            rows.append({"name": f, "group": c["group"], "text": c["description"], "broadcast": bool(c["broadcast"]), "coverage": float(cov[f]),
                         "ic": None if r is None else float(r["mean_ic"]), "ic_ir": None if r is None else float(r["ic_ir"]), "ic_t": None if r is None else float(r["t"])})
        y = ds.Y[ds.target].dropna()
        if until is not None:
            y = y[y.index.get_level_values(0) <= until]
        per_asset = ds.X.groupby(level="asset").size().reindex(ds.assets).fillna(0).astype(int)
        first = ds.X.reset_index().groupby("asset")["date"].min().reindex(ds.assets)
        out = {"feature_set": fid, "dataset": ds.summary(), "features": rows, "skipped": {k: str(v) for k, v in ds.meta.get("skipped", {}).items()},
               "assets": [{"asset": a, "rows": int(per_asset[a]), "first": None if pd.isna(first[a]) else str(pd.Timestamp(first[a]).date())} for a in ds.assets], "plan": plan_info,
               "target": {"name": ds.target, "n": int(len(y)), "mean": float(y.mean()) if len(y) else None, "std": float(y.std()) if len(y) > 1 else None, "share_positive": float((y > 0).mean()) if len(y) else None,
                          "scope": scope},
               "groups": {g: int(sum(1 for r in rows if r["group"] == g)) for g in dict.fromkeys(r["group"] for r in rows)}, "provenance": {"source": info.get("source"), "label": info.get("label")},
               "timing": {"lag": ds.lag, "horizon": ds.horizon, "purge": ds.lag + ds.horizon + int(cfg["split"]["embargo"])}, "audit": ds.meta.get("audit"), "feature_screen": screen}
        return common.envelope("ml_dataset", out, {"data": info.get("source", "platform") if info.get("source") in common.SOURCES else "mixed"},
                               ([f"{len(ds.meta.get('skipped', {}))} features could not be built from this data (see skipped)."] if ds.meta.get("skipped") else []), ds.dates[-1],
                               ["Features at the close of day t use data up to t only (every feature is tested for this); targets look forward from t + lag.", f"Information coefficients and target statistics are measured on the {scope} only."])

    def feature_sets(self, q: dict | None = None) -> dict:
        """``GET /api/ml/feature-sets``: the saved panels."""
        return clean({"feature_sets": [{k: v for k, v in f.items() if k != "path"} for f in self.registry.list_feature_sets()]})

    # ------------------------------------------------------------------------------------------------------------ training
    def train(self, body: dict) -> dict:
        """``POST /api/ml/train`` (a job): the whole experiment. Returns the experiment id and its report.

        ``mode`` is ``full`` (the default: every segment is predicted; where no reservation protects the history this is recorded as an exposure and the report says ``exposed_exploratory``) or ``development``
        (it stops at the end of the validation segment: the test and forward outcomes are not read, and nothing about the run can depend on them). A history that a frozen plan has reserved is refused in
        ``full`` mode with the way forward: ``development``, or ``POST /api/ml/evaluate`` for the plan."""
        mode = str(body.get("mode") or "full")
        if mode not in ("full", "development"):
            raise ApiError("mode must be full or development; a final evaluation of a reserved history goes through POST /api/ml/evaluation-plan and /api/ml/evaluate")
        over = self._overrides(body)
        explain, robust = bool(body.get("explain", True)), bool(body.get("robustness", False))
        label = over.get("name") or f"{over['model']['kind']} on {over.get('dataset', {}).get('universe', 'etf')}"

        def work(progress):
            with _user_errors():
                exp = run_experiment(over, config=self.app.config, builder=self.app.builder, registry=self.registry, progress=progress, explain=explain, robustness=robust or None,
                                     stop_after="validation" if mode == "development" else None, entry_point="api")
            return {"id": exp.id, "report": exp.report}

        return self._submit(f"ml: {label}" + (" (development)" if mode == "development" else ""), work)

    def rebacktest(self, body: dict) -> dict:
        """``POST /api/ml/rebacktest`` (a job): a saved experiment's predictions under different position rules, rebalancing, costs or with the stress tests, with no refit. Recorded as a child run."""
        exp_id = _id(body)
        extra = set(body) - {"id", "signal", "backtest", "name", "notes", "robustness", "execution"}
        if extra:
            raise ApiError(f"a re-backtest can change only the signal, backtest, execution and robustness settings, not {sorted(extra)}: they would change the predictions (train a new model instead)")
        over = self._overrides({k: body.get(k) for k in ("signal", "backtest", "name", "notes", "robustness", "execution")}, need_model=False)
        over = {k: v for k, v in over.items() if k in ("signal", "backtest", "name", "notes", "robustness", "execution")}

        def work(progress):
            with _user_errors():
                exp = rebacktest(self.registry, exp_id, over, progress, config=self.app.config, builder=self.app.builder)
            return {"id": exp.id, "parent": exp_id, "report": exp.report}

        return self._submit(f"ml: re-backtest {exp_id[-6:]}", work)

    def meta_label(self, body: dict) -> dict:
        """``POST /api/ml/meta-label`` (a job): train a secondary model on a saved experiment's out-of-sample bets and size the book with it."""
        exp_id = _id(body)
        kind = str(body.get("kind") or "logistic")
        if kind not in {m["kind"] for m in model_catalog() if "classification" in m["tasks"]}:
            raise ApiError("the secondary model must be a classifier: logistic, random_forest, hist_gb ...")
        signal = self._clean_section("signal", body.get("signal"))

        def work(progress):
            with _user_errors():
                exp, meta = en.meta_label_experiment(self.registry, exp_id, kind, None, signal or None, progress)
                parent = self.registry.report(exp_id)
            return {"id": exp.id, "parent": exp_id, "meta": meta, "report": exp.report, "parent_segments": parent["segments"]["stats"]}

        return self._submit(f"ml: meta-label {exp_id[-6:]}", work)

    # ------------------------------------------------------------------------------------------------------------ search
    def search(self, body: dict) -> dict:
        """``POST /api/ml/search`` (a job): the staged strategy search (random, then Bayesian, then genetic) over model kinds, hyperparameters, horizon, position rule and rebalancing. Trials see only the
        warm-up and validation segments and stop there. With ``finalize: true`` (off by default) a plan is frozen from the best few and evaluated once on the test and forward segments; every trial counts against
        their deflated Sharpe ratio."""
        over = self._overrides(body, need_model=False)
        kinds = body.get("kinds") or list(ss.DEFAULT_KINDS)
        valid = {m["kind"] for m in model_catalog() if m["available"]}
        if not isinstance(kinds, list) or not 1 <= len(kinds) <= LIMITS["search_kinds"] or any(k not in valid for k in kinds):
            raise ApiError(f"kinds must be 1 to {LIMITS['search_kinds']} of {sorted(valid)}")
        budget = {"n_random": _int(body, "n_random", 8, 0, LIMITS["search_random"]), "n_bayes": _int(body, "n_bayes", 6, 0, LIMITS["search_bayes"]),
                  "generations": _int(body, "generations", 2, 0, LIMITS["search_generations"]), "population": _int(body, "population", 6, 2, LIMITS["search_population"]),
                  "top_k": _int(body, "top_k", 2, 1, LIMITS["search_top_k"])}
        if budget["n_random"] + budget["n_bayes"] < 2:
            raise ApiError("a search needs at least two trials")
        seed = _int(body, "seed", 0, 0, 1_000_000)
        finalise = bool(body.get("finalize", False))
        key = str(body["idempotency_key"]) if body.get("idempotency_key") else None
        if key is not None and not rc.ID.match(key):
            raise ApiError("idempotency_key is 1 to 80 letters, digits, '.', '_', ':' or '-'")

        def work(progress):
            with _user_errors():
                result = ss.search(over, config=self.app.config, builder=self.app.builder, registry=self.registry, kinds=tuple(kinds), seed=seed, progress=progress, finalize_top=finalise, idempotency_key=key,
                                   **budget)
            return jsonable({**result, "budget": budget, "finalize": {"requested": finalise, "note": None if finalise else "the search stopped at the end of the validation segment; to evaluate the best on the test and "
                                                                                                                       "forward segments, freeze a plan (POST /api/ml/evaluation-plan) and evaluate it (POST /api/ml/evaluate), or repeat the search with finalize: true "
                                                                                                                       "(deprecated: it does the same, once, and records it)"}})

        return self._submit(f"ml: search {sum(budget[k] for k in ('n_random', 'n_bayes'))}+ trials", work)

    def tune(self, body: dict) -> dict:
        """``POST /api/ml/tune`` (a job): the settings of ONE model kind, validation-only, and (optionally) the best set finalised on every segment. With ``effort`` (``quick``, ``standard`` or ``thorough``)
        the search is a TPE one that starts from the settings in the request, as its first trial, so the answer comes with a comparison against them: the best and the starting settings are both run on the
        test and forward segments once, at the end, and ``summary`` says in words whether the improvement held up there. Without ``effort`` it is the plain search of ``method`` and ``n_trials``."""
        over = self._overrides(body)
        kind = over["model"]["kind"]
        effort = str(body.get("effort") or "")
        if effort:
            if effort not in TUNE_EFFORT:
                raise ApiError(f"effort must be one of {', '.join(TUNE_EFFORT)}")
            trials, method = TUNE_EFFORT[effort][0], "tpe"
        else:
            method = str(body.get("method") or "random")
            if method not in ("random", "tpe", "bayes"):
                raise ApiError("method must be random, tpe or bayes")
            trials = _int(body, "n_trials", 12, 4, LIMITS["tune_trials"])
        engine = str(body.get("engine") or "builtin")
        if engine not in ("builtin", "optuna"):
            raise ApiError("engine must be builtin or optuna")
        if engine == "optuna":
            why = unavailable_reason("optuna")
            if why:
                raise ApiError(f"the Optuna engine is not available here: {why}")
            method = "optuna"
        if kind not in {m["kind"] for m in model_catalog() if m["available"]}:
            raise ApiError(f"the {kind} model is not available here")
        space = ss.kind_space(kind)
        if not any(k.startswith("model.params.") for k in space):
            raise ApiError(f"the {kind} model has no hyperparameters to tune; tune its signal settings with a search instead")
        finalise, seed = bool(body.get("finalize", False)), _int(body, "seed", 0, 0, 1_000_000)
        key = str(body["idempotency_key"]) if body.get("idempotency_key") else None
        if key is not None and not rc.ID.match(key):
            raise ApiError("idempotency_key is 1 to 80 letters, digits, '.', '_', ':' or '-'")
        begin = None
        if effort:                                                                   # the settings in the form are the first trial: the space is stretched to hold them
            try:
                begin = opt.current_settings(over, space)
                space = opt.include(space, begin)
            except ValueError as exc:
                raise ApiError(str(exc)) from None
        stop = threading.Event()

        def work(progress):
            with _user_errors():
                out = opt.tune(over, space, trials, method, seed, config=self.app.config, builder=self.app.builder, registry=self.registry, study_name=f"tune-{kind}", progress=progress, start=begin,
                               stop=stop.is_set)
                final = final_start = hold_t = bench = evaluation = None
                if finalise and out.get("best"):
                    progress("freezing an evaluation plan from the best validation settings and evaluating it once")
                    wants_comparison = begin is not None and out["best"]["number"] != 0
                    params_list = [out["best"]["params"]] + ([begin] if wants_comparison else [])
                    names = [f"{kind} tuned" if begin is not None else None] + ([f"{kind} starting settings"] if wants_comparison else [])
                    ev = fe.compat_finalize(self.registry, rc.study_of(experiment_config(over)), over, params_list, names=names, selection_adjustment="dsr_all_trials",
                                            idempotency_key=key or f"tune-{kind}-{__import__('uuid').uuid4().hex[:12]}", actor="api", entry_point="api", config=self.app.config, builder=self.app.builder,
                                            progress=progress)
                    evaluation = {k: ev.get(k) for k in ("status", "plan", "deprecated", "idempotency_key", "untouched")}
                    if ev["status"] == "evaluated":
                        exp = ev["_experiments"][0]
                        final = {"id": exp.id, "report": exp.report}
                        held_days = exp.labels.reindex(exp.net.index).isin(["test", "forward"])                              # the days no search saw
                        bench = {"holdout": {"sharpe": tuning.metrics(exp.benchmark[held_days])["sharpe"]}}                  # holding the whole universe in equal amounts: a long-only winner may only be riding it
                        if wants_comparison:
                            start_exp = ev["_experiments"][1]
                            final_start = {"id": start_exp.id, "report": start_exp.report}
                            hold = lambda e: e.net[e.labels.reindex(e.net.index).isin(["test", "forward"])]                 # noqa: E731  the days no search saw
                            hold_t = tuning.paired_t(hold(exp), hold(start_exp))
                    else:                                                                                                        # a retry of a finished evaluation: the stored result, no second look
                        evaluation["result"] = ev.get("result") or ev.get("detail")
                summary = self._tune_summary(out, final, final_start, hold_t, bench)
            return jsonable({**out, "kind": kind, "space": sorted(space), "order": list(space), "final": final, "final_start": final_start, "summary": summary, "start": begin, "effort": effort or None, "evaluation": evaluation,
                             "finalize": {"requested": finalise, "note": None if finalise else "the tuning stopped at the end of the validation segment: the test and forward outcomes were not read. To evaluate the "
                                                                                                "best settings, freeze a plan (POST /api/ml/evaluation-plan) and evaluate it (POST /api/ml/evaluate), or tune with finalize: true (deprecated: the same, done once and recorded)"},
                             "texts": {path: TUNE_TEXT.get(path.split(".")[-1], "") for path in space}})

        started = self._submit(f"ml: tune {kind}", work)
        for old in [j for j in self.stops if j not in self.app.jobs or self.app.jobs[j].status not in ("queued", "running")]:
            self.stops.pop(old, None)
        self.stops[started["job"]] = stop
        return started

    def tune_stop(self, body: dict) -> dict:
        """``POST /api/ml/tune/stop {"job": id}``: end a tuning after the trial it is on; the trials so far are kept, and the best of them is still run on every segment."""
        flag = self.stops.get(str(body.get("job") or ""))
        if flag is None:
            raise ApiError("that tuning is not running in this app")
        flag.set()
        return {"ok": True}

    @staticmethod
    def _validation_only_verdict(out: dict) -> dict:
        """The verdict of a tuning that did not evaluate the test and forward segments (the default since the holdout is protected): the validation Sharpe of the starting settings and of the winner, the deflated
        probability when there is one, and the way to find out whether the gain holds up. It never calls the gain good or overfitted: the data that could say so was not read."""
        fmt = lambda v: "undefined" if v is None or not np.isfinite(v) else f"{v:.2f}"                      # noqa: E731
        first = next((t for t in out["trials"] if t["number"] == 0), {})
        d, b, tried = first.get("validation_sharpe"), out["best"].get("validation_sharpe"), out["n_trials"]
        if out["best"]["number"] == 0:
            level, headline = "keep", f"Keep what you have: none of the {tried - 1} other settings beat it on the validation segment"
            reasons = ["Nothing tried did better than the starting settings on the validation segment, so there is nothing to carry over."]
        else:
            level, headline = "mixed", "Better on the validation segment; the test and forward segments have not been looked at"
            reasons = [f"On the validation segment the Sharpe ratio went from {fmt(d)} to {fmt(b)}. That is the part the search could see, so some of it is luck."]
        reasons.append("The test and forward segments were not read: they stay protected until a frozen plan evaluates them once. To find out whether the improvement holds up there, freeze a plan "
                       "(POST /api/ml/evaluation-plan) and evaluate it (POST /api/ml/evaluate), or tick 'Also evaluate the best on the test and forward segments, once' (a deprecated shortcut that does the same and is recorded).")
        sel = out.get("selection") if isinstance(out.get("selection"), dict) else None
        if sel and sel.get("deflated_sharpe_probability") is not None:
            total = sel.get("n_trials_counted", tried)
            reasons.append(f"After {total} tries, the chance that the winner's Sharpe ratio on the validation segment is more than luck is {sel['deflated_sharpe_probability'] * 100:.0f}% (95% or more is convincing).")
        return {"level": level, "headline": headline, "reasons": reasons}

    @staticmethod
    def _tune_summary(out: dict, final: dict | None, final_start: dict | None, hold_t: float | None = None, benchmark: dict | None = None) -> dict | None:
        """The verdict of a tuning that started from the person's own settings: the validation Sharpe ratio (where the search chose) and the test and forward Sharpe ratios (days-weighted, where it did not) for
        the starting settings and for the winner, put in words by the same rules as the Tune tab."""
        if out.get("start_number") is None or not out.get("best"):
            return None
        if final is None:                                                                                    # the protected segments were not read: the verdict says what the validation shows and what it cannot
            return MlApi._validation_only_verdict(out)

        def held(report: dict):
            stats = report["segments"]["stats"]
            parts = [(stats[s]["sharpe"], stats[s].get("n_days") or 1) for s in ("test", "forward") if s in stats and stats[s].get("sharpe") is not None]
            return sum(v * n for v, n in parts) / sum(n for _, n in parts) if parts else None

        same = out["best"]["number"] == 0
        first = next((t for t in out["trials"] if t["number"] == 0), {})
        best = {"tuning": {"sharpe": out["best"].get("validation_sharpe")}, "holdout": {"sharpe": held(final["report"])}}
        start = {"tuning": {"sharpe": first.get("validation_sharpe")}, "holdout": {"sharpe": held(final["report"]) if same else (held(final_start["report"]) if final_start else None)}}
        selection = out.get("selection") if isinstance(out.get("selection"), dict) and out["selection"].get("deflated_sharpe_probability") is not None else None
        return tuning.judge("sharpe", start, best, "", "", same, selection, out["n_trials"], 0, None, benchmark, ("the validation segment", "the test and forward segments", "those segments"), hold_t=hold_t)

    # ------------------------------------------------------------------------------------------------------------ the registry
    def verification(self, q: dict | None = None) -> dict:
        """``GET /api/ml/verification[?id=P0-01]``: the latest persisted check of each model-correctness requirement, whether the source it checked has changed since (``stale``), and every status in the evidence
        record. Nothing here computes a status: only a run (``POST /api/ml/verification/run``) writes one."""
        from . import ml_verify as mv
        return clean(mv.report(self.registry if (self.registry_root / "registry.sqlite").exists() else None, (q or {}).get("id")))

    def verification_run(self, body: dict) -> dict:
        """``POST /api/ml/verification/run {"requirements": ["P0-01", ...]}`` (a job): run the tests behind the requirements and record the result."""
        from . import ml_verify as mv
        requirements = body.get("requirements")
        if requirements is not None and (not isinstance(requirements, list) or len(requirements) > 40 or not all(isinstance(r, str) for r in requirements)):
            raise ApiError("requirements must be a list of requirement ids")
        mv.validate(requirements)

        def work(progress):
            with _user_errors():
                return clean({"checks": mv.run(self.registry, requirements, progress)})

        return self._submit("ml: verification", work)

    def experiments(self, q: dict) -> dict:
        """``GET /api/ml/experiments?sort=validation.sharpe&limit=50&universe=&model=&stage=``: the leaderboard, with the number of looks at the data when each result was produced."""
        sort = str(q.get("sort") or "validation.sharpe")
        if not KEY.match(sort):
            raise ApiError("sort must look like validation.sharpe")
        limit = _int(q, "limit", 50, 1, 500)
        rows = self.registry.leaderboard(sort, 1000, str(q["universe"]) if q.get("universe") else None)
        if q.get("model"):
            rows = [r for r in rows if r["model_kind"] == q["model"]]
        if q.get("stage"):
            rows = [r for r in rows if r.get("stage", "none") == q["stage"]]
        trials = {}
        for r in rows:
            trials[r["dataset_key"]] = self.registry.trial_count(r["dataset_key"])
        payload, parents = leaderboard_payload(rows[:limit]), {r["id"]: r.get("parent_id") for r in rows}
        labels = rc.holdout_labels(self.registry, [r["id"] for r in payload])
        for row in payload:                                                                     # a re-backtest or a meta-labelled run has no model of its own: forward tests and explanations need the parent's
            row["has_model"] = (self.registry.root / "experiments" / row["id"] / "model.pkl").exists()
            row["parent"] = parents.get(row["id"])
            row["holdout"] = labels.get(row["id"], "development_or_unrecorded")
        data = {"rows": payload, "sort": sort, "n": len(rows), "dataset_trials": trials,
                "note": "Sharpe ratios are net of costs. 'trials' is how many looks at the same sample (experiments and search trials) had been recorded when the run was made; the deflated Sharpe probability in each report corrects for it."}
        return clean(common.envelope("ml_experiments", data, {"experiments": "platform"}, [], None, []))

    def experiment(self, q: dict) -> dict:
        """``GET /api/ml/experiment?id=``: one experiment in full: its report, its registry record and its children (re-backtests, meta-labelled runs)."""
        exp_id = _id(q)
        with _user_errors():
            reg = self.registry
            record, report = reg.get_experiment(exp_id), reg.report(exp_id)
            kids = [{"id": r["id"], "name": r["name"], "created": r["created"]} for r in reg.list_experiments(1000) if r.get("parent_id") == exp_id]
            models = [m for m in reg.list_models() if m["experiment_id"] == exp_id]
        record.pop("artifact_dir", None)
        return clean({"id": exp_id, "report": report, "record": {k: record[k] for k in ("stage", "parent_id", "created", "seconds", "source", "notes") if k in record},
                      "artifacts": [{k: a[k] for k in ("name", "kind", "bytes")} for a in record.get("artifacts", [])], "children": kids,
                      "models": [{"id": m["id"], "stage": m["stage"], "kind": m["kind"], "created": m["created"]} for m in models]})

    def report_html(self, q: dict) -> dict:
        """``GET /api/ml/report-html?id=``: the experiment's self-contained HTML report (the page opens it in a new tab)."""
        exp_id = _id(q)
        with _user_errors():
            report = self.registry.report(exp_id)
        return {"id": exp_id, "html": render_html(report)}

    def compare(self, q: dict) -> dict:
        """``GET /api/ml/compare?ids=a,b,c``: settings and segment results side by side, equity curves over the shared days, correlations and a similarity map."""
        ids = _ids(q, "ids", 2, LIMITS["compare"])
        with _user_errors(), self._heavy:
            out = en.compare_experiments(self.registry, ids)
        return clean(common.envelope("ml_compare", out, {"experiments": "platform"}, [out["note"]] if out.get("note") else [], out["end"], []))

    def models(self, q: dict | None = None) -> dict:
        """``GET /api/ml/models?stage=``: the model registry."""
        stage = (q or {}).get("stage") or None
        if stage and stage not in STAGES:
            raise ApiError(f"stage must be one of {list(STAGES)}")
        return clean({"models": [{k: v for k, v in m.items() if k not in ("path", "features")} | {"n_features": len(m["features"])} for m in self.registry.list_models(stage)], "stages": list(STAGES)})

    def promote(self, body: dict) -> dict:
        """``POST /api/ml/promote {"model_id", "stage", "approval": {"actor", "note"}}``: move a registered model to a stage (at most one production model per universe; the previous one is archived).

        The stage change is decided here from the persisted evidence, never from the request: ``staging`` needs Gate A (Algorithm Verified) and Gate B (Strategy Validated); ``production`` needs all three gates and a
        recorded human approval. Taking a model out of use (``none``, ``archived``) needs nothing. A refusal carries the gate, its status and the first reason."""
        _only(body, {"model_id", "stage", "approval", "note"}, "/api/ml/promote")
        model_id, stage = _id(body, "model_id"), str(body.get("stage") or "")
        if stage not in STAGES:
            raise ApiError(f"stage must be one of {list(STAGES)}")
        approval = body.get("approval")
        if approval is not None:
            if not isinstance(approval, dict) or set(approval) - {"actor", "note"} or not all(isinstance(v, str) and len(v) <= 300 for v in approval.values()):
                raise ApiError("approval is {\"actor\": ..., \"note\": ...}, two short texts")
        with _user_errors():
            decision = self.registry.promote(model_id, stage, actor="api", note=str(body.get("note") or "")[:300], approval=approval)
        return {"ok": True, "model_id": model_id, "stage": stage, "decision": {k: decision.get(k) for k in ("allowed", "gate_seqs", "policy_version", "reason")}}

    def delete(self, body: dict) -> dict:
        """``POST /api/ml/delete`` with ``{"id": ..., "confirm": true}``: remove an experiment and its files."""
        exp_id = _id(body)
        if body.get("confirm") is not True:
            raise ApiError("deleting an experiment needs confirm: true")
        with _user_errors():
            self.registry.delete_experiment(exp_id)
        return {"ok": True, "id": exp_id}

    def trials(self, q: dict) -> dict:
        """``GET /api/ml/trials?study=&dataset_key=&limit=``: the recorded search trials."""
        study = str(q["study"]) if q.get("study") else None
        key = str(q["dataset_key"]) if q.get("dataset_key") else None
        for v in (study, key):
            if v is not None and not re.match(r"^[A-Za-z0-9_.:-]{1,80}$", v):
                raise ApiError("study and dataset_key are short identifiers")
        rows = self.registry.list_trials(study, key, _int(q, "limit", 200, 1, 2000))
        return clean({"trials": [{k: r[k] for k in ("study", "dataset_key", "number", "params", "value", "seconds", "created", "stage")} for r in rows],
                      "n_total": {k: self.registry.trial_count(k) for k in {r["dataset_key"] for r in rows}},
                      "lineage": {"note": "Failed and cancelled tries, feature searches and derived runs are in the study's lineage (GET /api/ml/lineage), which is where the raw look counts live."}})

    # ------------------------------------------------------------------------------------------------------------ research controls (Release 3)
    def lineage(self, q: dict) -> dict:
        """``GET /api/ml/lineage?id=<experiment>`` or ``?study=<study>``: the study's operations (every try, with its state, kind, parent and configuration), the raw look counts, the feature sets and configurations
        tried, the evaluation plans, the holdout access history, the tombstones of deleted experiments and the selection evidence. Counts are counts of recorded tries, not estimates of independent trials."""
        if q.get("id") and q.get("study"):
            raise ApiError("give id or study, not both")
        with _user_errors():
            if q.get("id"):
                exp_id = _id(q)
                self.registry.get_experiment(exp_id)
                if rc.study_of_experiment(self.registry, exp_id) is None:                      # saved before lineage was kept: say so, do not invent a study
                    data = {"study": {"study_id": None}, "studies": [], "operations": [], "raw_looks": 0, "by_kind": {}, "by_state": {}, "failed": 0, "cancelled": 0, "feature_sets": [], "configurations": [],
                            "plans": [], "holdout_access": [], "tombstones": [], "selection_evidence": [], "parents": [],
                            "note": "This experiment ran before study lineage was recorded: its tries are not in any study and none is reconstructed."}
                else:
                    data = rc.lineage(self.registry, experiment_id=exp_id)
            elif q.get("study"):
                data = rc.lineage(self.registry, study_id=_study(q["study"]))
            else:
                raise ApiError("give id (an experiment) or study")
        data["operations"] = data["operations"][-_int(q, "limit", 500, 1, 2000):]
        return clean(common.envelope("ml_lineage", data, {"registry": "platform"}, [], None, [data["note"]]))

    def holdout(self, q: dict) -> dict:
        """``GET /api/ml/holdout?id=<plan id>`` or ``?study=``: the plans, their reservations (with the overlaps found and whether the history is untouched), the evaluations, and the access ledger."""
        with _user_errors():
            data = rc.holdout_status(self.registry, _study(q["study"]) if q.get("study") else None, _id(q) if q.get("id") else None)
        return clean(common.envelope("ml_holdout", data, {"registry": "platform"}, [], None, ["Reading a result that was already computed is not a new look. A plan's holdout is consumed by its one evaluation event; a new evaluation is a re-use and is never untouched."]))

    def evaluation_plan(self, body: dict) -> dict:
        """``POST /api/ml/evaluation-plan`` (a job): create an evaluation plan from candidate configurations, build the data to learn which outcomes its test and forward segments reveal, reserve them in the
        ledger (any overlap with an earlier look is reported and the plan is then not untouched) and, with ``freeze: true``, freeze it. A frozen plan can be evaluated once (``POST /api/ml/evaluate``)."""
        _only(body, {"study", "candidates", "baseline", "selection_adjustment", "metrics", "uncertainty_method", "decision_policy", "notes", "freeze"}, "/api/ml/evaluation-plan")
        cands = body.get("candidates")
        if not isinstance(cands, list) or not 1 <= len(cands) <= LIMITS["plan_candidates"]:
            raise ApiError(f"candidates is a list of 1 to {LIMITS['plan_candidates']} experiment configurations")
        overs = [self._overrides(c) for c in cands]
        base = self._overrides(body["baseline"]) if body.get("baseline") else None
        study = _study(body.get("study") or rc.default_study_id(experiment_config(overs[0])))
        adjustment = str(body.get("selection_adjustment") or "none")
        if adjustment not in ("none", "bonferroni", "holm", "reality_check", "spa", "dsr_all_trials"):
            raise ApiError("selection_adjustment must be none, bonferroni, holm, reality_check, spa or dsr_all_trials")
        metrics = body.get("metrics")
        if metrics is not None and (not isinstance(metrics, list) or len(metrics) > 12 or not all(isinstance(m, str) and PARAM.match(m) for m in metrics)):
            raise ApiError("metrics is a short list of metric names")
        policy = body.get("decision_policy")
        if policy is not None and (not isinstance(policy, dict) or len(str(policy)) > 2000):
            raise ApiError("decision_policy is a small object")
        freeze = bool(body.get("freeze", False))

        def work(progress):
            with _user_errors():
                progress("building the data to learn which outcomes the plan would reserve")
                state = fe.plan_and_reserve(self.registry, study, overs, base, freeze=freeze, actor="api", config=self.app.config, builder=self.app.builder, selection_adjustment=adjustment,
                                            metrics=metrics, uncertainty_method=str(body.get("uncertainty_method") or "not_specified")[:120], decision_policy=policy, notes=str(body.get("notes") or "")[:500])
            return clean(state)

        return self._submit(f"ml: evaluation plan ({len(overs)} candidate{'s' if len(overs) != 1 else ''})", work)

    def evaluation_plan_freeze(self, body: dict) -> dict:
        """``POST /api/ml/evaluation-plan/freeze {"plan_id"}``: the explicit transition after which a plan can be evaluated. A plan with no reservation cannot be frozen."""
        _only(body, {"plan_id"}, "/api/ml/evaluation-plan/freeze")
        with _user_errors():
            return clean(rc.freeze_plan(self.registry, _id(body, "plan_id"), "api"))

    def evaluate(self, body: dict) -> dict:
        """``POST /api/ml/evaluate {"plan_id", "idempotency_key", "allow_reuse"?, "resume"?}`` (a job): the one evaluation event of a frozen plan. The same key returns the same result and reads nothing again; a
        different key on a consumed plan is refused unless ``allow_reuse`` says it is a deliberate re-use, which is recorded as one and is never untouched. ``resume`` takes over a claim whose process died."""
        _only(body, {"plan_id", "idempotency_key", "allow_reuse", "resume"}, "/api/ml/evaluate")
        plan_id, key = _id(body, "plan_id"), str(body.get("idempotency_key") or "")
        if not rc.ID.match(key):
            raise ApiError("idempotency_key is required: 1 to 80 letters, digits, '.', '_', ':' or '-'. A retry with the same key returns the same result")
        allow_reuse, resume = bool(body.get("allow_reuse", False)), bool(body.get("resume", False))

        def work(progress):
            with _user_errors():
                try:
                    out = fe.evaluate_plan(self.registry, plan_id, idempotency_key=key, actor="api", entry_point="api", allow_reuse=allow_reuse, resume=resume, config=self.app.config,
                                           builder=self.app.builder, progress=progress)
                except rc.ControlError as exc:
                    raise ValueError(str(exc)) from None
            out.pop("_experiments", None)
            return clean(out)

        return self._submit(f"ml: evaluate {plan_id[-8:]}", work)

    def statistical_diagnostics(self, q: dict) -> dict:
        """``GET /api/ml/statistical-diagnostics?id=<experiment>`` (or ``?study=``): the selection-bias evidence recorded for the study, procedure by procedure, with its status (``applied``,
        ``not_applicable``, ``unsupported``, ``insufficient``, ``failed``), assumptions, scope, dependence treatment and limitations, and the null-control studies. A procedure that was not applied is never shown as passed."""
        with _user_errors():
            study = _study(q["study"]) if q.get("study") else (rc.study_of_experiment(self.registry, _id(q)) if q.get("id") else None)
            if not q.get("study") and not q.get("id"):
                raise ApiError("give id (an experiment) or study")
            diagnostics = self._diagnostics_block(_id(q)) if q.get("id") else None
            if not study:                                                                                   # an experiment saved before lineage: no study, no evidence, said plainly
                return clean(common.envelope("ml_statistical_diagnostics", {"study": None, "evidence": [], "latest": [], "null_studies": [], "raw_looks": 0, "diagnostics": diagnostics,
                                                                           "statement": "this experiment is in no study (it ran before lineage was recorded), so no selection test can be attributed to it"},
                                             {"registry": "platform"}, [], None, []))
            family = rc.study_family(self.registry, study)
            with self.registry._conn() as c:
                rows = [dict(r) for r in c.execute(f"SELECT * FROM selection_evidence WHERE study_id IN ({','.join('?' * len(family))}) ORDER BY id DESC LIMIT 200", family)]
                nulls = [dict(r) for r in c.execute("SELECT null_id, study_id, state, planned, done, spec_hash, result, created FROM null_studies WHERE study_id=? ORDER BY rowid DESC LIMIT 50", (study,))]
        for r in rows:
            for k in ("assumptions", "scope", "dependence", "limitations", "statistic"):
                r[k] = __import__("json").loads(r[k]) if r[k] else None
        for n in nulls:
            n["result"] = __import__("json").loads(n["result"]) if n["result"] else None
        latest = {}
        for r in rows:
            latest.setdefault(r["method"], r)
        data = {"study": study, "evidence": rows, "latest": list(latest.values()), "null_studies": nulls, "raw_looks": rc.raw_looks(self.registry, study), "diagnostics": diagnostics,
                "statement": sel.statement([sel.Evidence(r["method"], r["status"], statistic=r["statistic"] or {}) for r in latest.values()]) if latest else "no selection test has been run for this study"}
        return clean(common.envelope("ml_statistical_diagnostics", data, {"registry": "platform"}, [], None, ["A procedure with status other than applied was not applied; it is not a pass."]))

    def selection_tests_run(self, body: dict) -> dict:
        """``POST /api/ml/selection-tests/run {"study", "procedures"?, "n_boot"?}`` (a job): apply the selection-bias procedures (deflated Sharpe, PBO, Reality Check, SPA, FDR) to the study's recorded trials
        where they apply, and persist each result or the reason it was not applied."""
        _only(body, {"study", "procedures", "n_boot", "seed"}, "/api/ml/selection-tests/run")
        study = _study(body.get("study"))
        procs = body.get("procedures") or list(sel.PROCEDURES)
        if not isinstance(procs, list) or not procs or not all(p in sel.PROCEDURES for p in procs):
            raise ApiError(f"procedures is a non-empty list drawn from {list(sel.PROCEDURES)}")
        n_boot, seed = _int(body, "n_boot", 500, 50, 2000), _int(body, "seed", 0, 0, 1_000_000)

        def work(progress):
            with _user_errors():
                progress("applying the selection procedures to the study's recorded trials")
                adopted = pol.adopted(self.registry)
                return clean(sel.evaluate_study(self.registry, study, procedures=tuple(procs), n_boot=n_boot, seed=seed, criteria=sel.criteria_from_policy(adopted[0] if adopted else None)))

        return self._submit(f"ml: selection tests for {study}", work)

    def null_controls(self, q: dict) -> dict:
        """``GET /api/ml/null-controls[?id=<null id>]``: null-control studies (the design that was frozen, the replications, the false-promotion rate with its intervals and the verdict)."""
        with _user_errors():
            if q.get("id"):
                data = nc.load(self.registry, _id(q))
                data["replication_records"] = data["replication_records"][-200:]
                return clean(common.envelope("ml_null_study", data, {"registry": "platform"}, [], None, []))
            with self.registry._conn() as c:
                rows = [dict(r) for r in c.execute("SELECT null_id, study_id, state, planned, done, spec_hash, result, created, finished FROM null_studies ORDER BY rowid DESC LIMIT 100")]
        for r in rows:
            r["result"] = __import__("json").loads(r["result"]) if r["result"] else None
        return clean(common.envelope("ml_null_studies", {"studies": rows}, {"registry": "platform"}, [], None, ["A study with too few replications validates the plumbing only; it says nothing about the error rate."]))

    def null_controls_run(self, body: dict) -> dict:
        """``POST /api/ml/null-controls/run`` (a job): freeze the design, then run the whole search pipeline on placebo data ``replications`` times and report the false-promotion rate with its binomial
        interval. The design (budget, kinds, generator, gate, tolerance, precision, minimum replications, seed) is hashed and stored before the first replication."""
        _only(body, {"study", "replications", "kinds", "n_random", "n_bayes", "seed", "tolerance", "precision", "min_replications", "min_shift", "dataset", "split", "signal", "backtest", "model"}, "/api/ml/null-controls/run")
        over = self._overrides(body, need_model=True)
        kinds = body.get("kinds") or ["ridge", "lasso"]
        valid = {m["kind"] for m in model_catalog() if m["available"]}
        if not isinstance(kinds, list) or not 1 <= len(kinds) <= 4 or any(k not in valid for k in kinds):
            raise ApiError(f"kinds must be 1 to 4 of {sorted(valid)}")
        spec = nc.NullSpec(study_id=_study(body.get("study") or rc.default_study_id(experiment_config(over))), base=over, kinds=tuple(kinds), n_random=_int(body, "n_random", 4, 1, LIMITS["null_trials"]),
                           n_bayes=_int(body, "n_bayes", 0, 0, LIMITS["null_trials"]), replications=_int(body, "replications", 20, 2, LIMITS["null_replications"]), seed=_int(body, "seed", 0, 0, 1_000_000),
                           tolerance=_float(body, "tolerance", 0.10, 0.005, 0.5), precision=_float(body, "precision", 0.10, 0.005, 0.5), min_replications=_int(body, "min_replications", 100, 2, LIMITS["null_replications"]),
                           min_shift=_int(body, "min_shift", 200, 60, 1000))
        stop = threading.Event()

        def work(progress):
            with _user_errors():
                made = nc.register(self.registry, spec)
                progress("design frozen: " + made["null_id"])
                replicate = nc.search_pipeline(spec, config=self.app.config, builder=self.app.builder)
                out = nc.run(self.registry, made["null_id"], replicate, progress=progress, stop=stop.is_set)
            return clean(out)

        started = self._submit(f"ml: null controls ({spec.replications} replications)", work)
        self.stops[started["job"]] = stop
        return started

    def trading_diagnostics(self, q: dict) -> dict:
        """``GET /api/ml/trading-diagnostics?id=<experiment>``: the evaluation mode of a result and what it assumes: the timeline from decision to fill, the price convention, gross and net, the itemised costs,
        turnover, capacity at predeclared capitals, the accounting reconciliation, the comparison with the vectorized engine and the limitations that qualify the claim. An experiment saved before execution
        modes were recorded is read as ``vectorized_legacy`` and says so."""
        exp_id = _id(q)
        with _user_errors():
            reg = self.registry
            record, report = reg.get_experiment(exp_id), reg.report(exp_id)
        block = report.get("execution") or xexec.legacy_block(report["config"], recorded=False)
        siblings = [r["id"] for r in reg.list_experiments(1000) if r.get("parent_id") == (record.get("parent_id") or exp_id) and r["id"] != exp_id][:20]
        data = {"id": exp_id, "parent": record.get("parent_id"), "siblings": siblings, **block}
        notes = ["Every realistic result is a child run: its parent is untouched and keeps its own mode.", *(block.get("limitations") or block.get("not_modelled") or [])]
        return clean(common.envelope("ml_trading_diagnostics", data, {"registry": "platform"}, [], None, notes))

    def promotion_status(self, q: dict) -> dict:
        """``GET /api/ml/promotion-status?id=<experiment>``: the three independent gates (Algorithm Verified, Strategy Validated, Operationally Ready), each ``passed``, ``failed``, ``pending`` or ``blocked``
        with ``stale`` when its evidence has changed since it was decided, the item-by-item reasons, the policy version and the evidence behind each. Computed here from the persisted evidence (and recorded
        when it changes); nothing in a request can set a status."""
        exp_id = _id(q)
        with _user_errors():
            res = gates.decide(self.registry, exp_id, "api")
            models = [{"id": m["id"], "stage": m["stage"], "history": gates.stage_history(self.registry, m["id"])} for m in self.registry.list_models() if m["experiment_id"] == exp_id]
        res["models"] = models
        res["history"] = gates.history(self.registry, exp_id, _int(q, "history", 12, 0, 100))
        res["next"] = {n: (g["reasons"][0] if g["reasons"] else "nothing outstanding") for n, g in res["gates"].items()}
        return clean(common.envelope("ml_promotion_status", res, {"registry": "platform"}, [], None, [res["note"], "A gate that is not passed is shown with its reason. A stale decision is shown as pending."]))

    # ------------------------------------------------------------------------------------------------------------ Release 5: the validation policy, the diagnostics, governance and the ledger
    def _diagnostics_block(self, exp_id: str) -> dict | None:
        """The latest stored statistical diagnostics of an experiment, read against the adopted policy. ``stale`` when they were computed under a different policy (then no item is shown as a verdict)."""
        rec = dx.latest(self.registry, exp_id, "diagnostics")
        adopted = pol.adopted(self.registry)
        seeds = dx.latest(self.registry, exp_id, "multiseed")
        if rec is None:
            return {"record": None, "policy": None if adopted is None else adopted[0].content(), "items": [], "stale": False, "multiseed": None if seeds is None else seeds["content"],
                    "note": "no diagnostics have been computed for this experiment"}
        stale = adopted is None or rec["policy_hash"] != adopted[0].content_hash()
        items = [] if stale or adopted is None else dx.evaluate_policy(adopted[0], rec["content"])
        return {"record": rec["content"], "seq": rec["seq"], "created": rec["created"], "content_hash": rec["content_hash"], "policy_hash": rec["policy_hash"], "stale": stale, "items": items,
                "multiseed": None if seeds is None else seeds["content"],
                "policy": None if adopted is None else adopted[0].content(),
                "note": "computed under another policy than the one adopted now: compute them again" if stale and adopted is not None else "no policy is adopted: the numbers are shown, no verdict is read from them" if adopted is None else ""}

    def validation_policy(self, q: dict) -> dict:
        """``GET /api/ml/validation-policy``: the policy in force (or none), who adopted it and why, the adoption history, the criteria a policy may state, and the illustrative example. The platform implies no
        Sharpe, deflated-Sharpe, PBO, p-value or paper-trading-duration threshold: until a person adopts a policy Gates B and C are blocked."""
        with _user_errors():
            adopted, history = pol.adopted(self.registry), pol.history(self.registry, 20)
        crit = {k: ({"kind": v[0], "options": list(v[1]), "text": v[2]} if v[0] == "choice" else {"kind": v[0], "min": v[1], "max": v[2], "text": v[3]}) for k, v in pol.CRITERIA.items()}
        data = {"adopted": None if adopted is None else {"policy": adopted[0].content(), "policy_hash": adopted[0].content_hash(), "adoption": adopted[1]}, "history": history, "criteria": crit,
                "example": pol.example_policy().content()}
        return clean(common.envelope("ml_validation_policy", data, {"registry": "platform"}, [], None,
                                     ["A policy is a declaration made before the evaluation. The example is illustrative: its numbers are not calibrated to any market and are not a recommendation."]))

    def validation_policy_adopt(self, body: dict) -> dict:
        """``POST /api/ml/validation-policy {"policy": {...} | "example": true, "actor", "note"}``: adopt a policy. Needs a named person and a reason; the record is append-only and the latest adoption is in force."""
        _only(body, {"policy", "example", "actor", "note"}, "/api/ml/validation-policy")
        if bool(body.get("example")) == bool(body.get("policy")):
            raise ApiError("give either policy (an object) or example: true")
        raw = body.get("policy")
        if not body.get("example") and (not isinstance(raw, dict) or set(raw) - {"name", "version", "criteria", "description"} or not isinstance(raw.get("criteria"), dict)):
            raise ApiError("policy is an object with name, version, criteria (an object) and an optional description")
        with _user_errors():
            policy = pol.example_policy() if body.get("example") else pol.ValidationPolicy(str(raw.get("name") or "")[:80], str(raw.get("version") or "")[:40], dict(raw["criteria"]), False,
                                                                                             str(raw.get("description") or "")[:1000])
            rec = pol.adopt(self.registry, policy, str(body.get("actor") or "")[:80], str(body.get("note") or "")[:500])
        return clean(common.envelope("ml_validation_policy_adopted", rec, {"registry": "platform"}, [], None, ["Diagnostics computed under an earlier policy are stale under this one and must be computed again."]))

    def diagnostics_run(self, body: dict) -> dict:
        """``POST /api/ml/statistical-diagnostics/run {"id", "n_boot"?, "seed"?, "neighborhood"?, "delays"?, "capitals"?}`` (a job): compute and store the statistical diagnostics of one experiment under the adopted
        policy: the three Sharpe estimates, matched benchmarks, regimes, calibration, the study's selection evidence, and (when asked) the neighbourhood, delay and capacity analyses, which re-run the saved
        predictions on the validation segment and count as looks."""
        _only(body, {"id", "n_boot", "seed", "neighborhood", "delays", "capitals"}, "/api/ml/statistical-diagnostics/run")
        exp_id = _id(body)
        n_boot, seed = _int(body, "n_boot", 500, 200, 2000), _int(body, "seed", 0, 0, 1_000_000)
        grid = body.get("neighborhood")
        if grid is not None:
            if not isinstance(grid, dict) or not 1 <= len(grid) <= 4 or any(not isinstance(v, list) or not 1 <= len(v) <= 8 for v in grid.values()):
                raise ApiError("neighborhood is an object of 1 to 4 settings (signal.*, backtest.*, execution.*), each a list of 1 to 8 values")
        delays, capitals = body.get("delays"), body.get("capitals")
        if delays is not None and (not isinstance(delays, list) or not 1 <= len(delays) <= 8 or any(not isinstance(d, int) or isinstance(d, bool) or not 0 <= d <= 20 for d in delays)):
            raise ApiError("delays is a list of 1 to 8 integers between 0 and 20 (extra sessions between decision and fill)")
        if capitals is not None and (not isinstance(capitals, list) or not 2 <= len(capitals) <= 6 or any(not isinstance(c, (int, float)) or isinstance(c, bool) or not 1e3 <= c <= 1e12 for c in capitals)):
            raise ApiError("capitals is a list of 2 to 6 amounts between 1e3 and 1e12")

        def work(progress):
            with _user_errors():
                self.registry.get_experiment(exp_id)
                progress("computing the dependence-aware estimates, benchmarks, regimes and calibration")
                out = dx.run(self.registry, exp_id, actor="api", n_boot=n_boot, seed=seed, neighborhood_grid=grid, delays=delays, capitals=capitals, config=self.app.config, builder=self.app.builder)
            return clean({"experiment_id": exp_id, "record_seq": out.get("record_seq"), "policy_hash": out["policy_hash"], "diagnostics": self._diagnostics_block(exp_id)})

        return self._submit(f"ml: diagnostics {exp_id[-6:]}", work)

    def _stored_record(self, kind: str, exp_id: str, record_kind: str) -> dict:
        rec = dx.latest(self.registry, exp_id, record_kind)
        data = ({"eligibility": "not_computed", "reason": "these diagnostics have not been computed for this experiment", "experiment_id": exp_id} if rec is None
                else {**rec["content"], "record_seq": rec["seq"], "computed_at": rec["created"], "stored": True})
        return clean(common.envelope(kind, data, {"registry": "platform"}, [], None, []))

    def alpha_diagnostics(self, q: dict) -> dict:
        """``GET /api/ml/alpha-diagnostics?id=<experiment>[&segment=validation][&baseline=<experiment>]``: the rank IC of the saved scores with a dependence-aware interval, its rolling value, its decay with delay and
        horizon, coverage, stability across blocks, the redundancy of the feature panel (development rows only) and, given a baseline experiment, the conditional IC and the paired blend. The test and forward segments
        are refused unless a frozen plan evaluated them. Each request is appended to ``analysis_records`` as a look (diagnostics that drive feature choices are logged)."""
        _only(q, {"id", "segment", "baseline", "stored"}, "/api/ml/alpha-diagnostics")
        with _user_errors():
            exp_id = _id(q)
            self.registry.get_experiment(exp_id)
            if q.get("stored"):                                                                    # the latest record of a request made earlier: reading it computes and records nothing
                return self._stored_record("ml_alpha_diagnostics", exp_id, "alpha_diagnostics")
            baseline = _id(q, "baseline") if q.get("baseline") else None
            with self._heavy:
                out = r6.alpha_diagnostics(self.registry, exp_id, segment=str(q.get("segment") or "validation"), baseline_id=baseline)
            seq = dx.store(self.registry, "alpha_diagnostics", exp_id, out, actor="api")
        return clean(common.envelope("ml_alpha_diagnostics", {**out, "record_seq": seq}, {"registry": "platform"}, [], None, []))

    def portfolio_diagnostics(self, q: dict) -> dict:
        """``GET /api/ml/portfolio-diagnostics?id=<experiment>[&segment=validation][&horizon=21][&dates=24]``: the book the experiment held, the covariance estimators compared on the risk it realised, the risk
        contributions of its last book and, for a return forecast only, how an optimal cost-aware book moves under a predeclared perturbation grid. Appended to ``analysis_records`` as a look."""
        _only(q, {"id", "segment", "horizon", "dates", "stored"}, "/api/ml/portfolio-diagnostics")
        with _user_errors():
            exp_id = _id(q)
            self.registry.get_experiment(exp_id)
            if q.get("stored"):
                return self._stored_record("ml_portfolio_diagnostics", exp_id, "portfolio_diagnostics")
            with self._heavy:
                out = r6.portfolio_diagnostics(self.registry, exp_id, segment=str(q.get("segment") or "validation"), horizon=_int(q, "horizon", 21, 5, 63), max_dates=_int(q, "dates", 24, 6, 60))
            seq = dx.store(self.registry, "portfolio_diagnostics", exp_id, out, actor="api")
        return clean(common.envelope("ml_portfolio_diagnostics", {**out, "record_seq": seq}, {"registry": "platform"}, [], None, []))

    def delay_report(self, q: dict) -> dict:
        """``GET /api/ml/delay-report?id=<experiment>[&segment=validation][&delays=0,1,2,5][&participation=0.1][&stored=1]``: the saved experiment's rank IC, expected edge, net result and capacity when its
        book is executed 0, 1, 2, 5 ... sessions later (the delays are declared in the request and 0 must be among them), the timing in real calendar days, and the stale-book rule: a delayed book cannot see
        a newer signal. Capacity needs ``participation`` (a limit on the share of a day's volume) and the universe's volume; without them it is said to be not established. Each request is appended to
        ``analysis_records``. The test and forward segments are refused unless a frozen plan evaluated them."""
        _only(q, {"id", "segment", "delays", "participation", "stored"}, "/api/ml/delay-report")
        exp_id = _id(q)
        with _user_errors():
            if str(q.get("stored") or "") in ("1", "true"):
                return self._stored_record("ml_delay_report", exp_id, "delay_report")
            try:
                delays = [int(x) for x in str(q.get("delays") or "0,1,2,5").split(",") if x.strip() != ""]
            except ValueError:
                raise ApiError("delays is a comma-separated list of whole numbers of sessions, for example 0,1,2,5") from None
            participation = _float(q, "participation", 0.0, 0.0, 1.0) or None
            volume = None
            if participation:
                cfg = self.registry.report(exp_id)["config"]
                try:
                    volume = getattr(resolve_bundle(self.app.config, cfg["dataset"]["universe"], self.app.builder)[0], "volume", None)
                except (FileNotFoundError, KeyError, ValueError):
                    volume = None                                                                                 # capacity is then reported as not established, not assumed
            with self._heavy:
                out = dly.report(self.registry, exp_id, segment=str(q.get("segment") or "validation"), declaration=dly.declare(delays), volume=volume, max_participation=participation)
            seq = dx.store(self.registry, "delay_report", exp_id, out, actor="api")
        return clean(common.envelope("ml_delay_report", {**out, "record_seq": seq}, {"registry": "platform"}, [], None, []))

    def improvement_get(self, q: dict) -> dict:
        """``GET /api/ml/improvement-experiment?id=<card>`` (the frozen card with its results, negative findings and gate decision) or, without an id, the list of cards (``&study=`` narrows it)."""
        _only(q, {"id", "study"}, "/api/ml/improvement-experiment")
        with _user_errors():
            if q.get("id"):
                data = cmpn.get(self.registry, str(q["id"]))
            else:
                data = {"cards": cmpn.list_cards(self.registry, _study(q["study"]) if q.get("study") else None)}
        return clean(common.envelope("ml_improvement_experiment", data, {"registry": "platform"}, [], None, []))

    IMPROVEMENT_FAMILIES = ("target", "feature", "horizon", "portfolio", "rebalancing", "risk_control", "ensemble", "execution", "combination")

    def _saved_baseline(self, exp_id: str) -> dict:
        """A saved experiment's configuration in the shape a request carries it (the derived signal settings are left out: validation derives them again)."""
        with _user_errors():
            cfg = self.registry.report(exp_id)["config"]
        body = {k: dict(cfg[k]) for k in ("dataset", "split", "model", "signal", "backtest") if cfg.get(k)}
        for k in ("cost_bps", "window_days"):
            body.get("signal", {}).pop(k, None)
        if cfg.get("execution"):
            ex = dict(cfg["execution"])
            if isinstance(ex.get("impact"), dict):
                ex["impact_coefficient"] = ex.pop("impact").get("coefficient", 0.0)
            body["execution"] = ex
        return body

    def improvement_run(self, body: dict) -> dict:
        """``POST /api/ml/improvement-experiment`` (a job): freeze a comparison card and run it once. ``family`` is ``target``, ``feature``, ``horizon`` or ``portfolio``; ``baseline`` is a configuration as for
        ``/api/ml/train``; ``candidates`` are 1 to 6 CHANGES laid over the baseline (a change that alters anything the family does not allow is refused); ``hypothesis`` and ``one_change`` are required.
        Optional: ``study``, ``primary_metric`` (``ic`` or ``net_sharpe``), ``parsimony`` (a declared tie-break rule), ``min_effect``, ``level``, ``minimum_coverage``, ``n_boot``, ``seed``. The arms are development
        runs (the test and forward outcomes are not predicted); nothing here promotes anything."""
        allowed = {"family", "hypothesis", "one_change", "baseline", "from_experiment", "candidates", "presets", "study", "primary_metric", "parsimony", "min_effect", "level", "minimum_coverage", "n_boot",
                   "seed", "members", "select_baseline", "combination_of", "ablation", "options"}
        _only(body, allowed, "/api/ml/improvement-experiment")
        family = str(body.get("family") or "")
        if family not in self.IMPROVEMENT_FAMILIES:
            raise ApiError(f"family must be one of {list(self.IMPROVEMENT_FAMILIES)}")
        ablation = bool(body.get("ablation"))
        for key in ("hypothesis", "one_change"):
            if not str(body.get(key) or "").strip() and not (ablation and family == "combination"):
                raise ApiError(f"{key} is required: state it before the numbers exist")
        if family == "combination":
            return self._combination_run(body, ablation)
        options = body.get("options") or {}
        if not isinstance(options, dict) or set(options) - {"capital", "max_participation"}:
            raise ApiError("options holds capital and max_participation (the realistic execution preset)")
        members = body.get("members")
        if members is not None:
            text = ",".join(str(m) for m in members) if isinstance(members, list) else str(members)
            try:
                parsed = parse_members(text)
            except ValueError as exc:
                raise ApiError(str(exc)) from None
            if not 2 <= len(parsed) <= 8:
                raise ApiError("members: between two and eight models")
            options = {**options, "members": text}
        if family == "ensemble" and not options.get("members"):
            raise ApiError("an ensemble comparison needs members, for example \"ridge,lasso,elasticnet\"")
        select = bool(body.get("select_baseline"))
        if select and family != "ensemble":
            raise ApiError("select_baseline applies to the ensemble family: the baseline is the best single member on the development segment")
        cands, presets = body.get("candidates") or [], body.get("presets") or []
        if not isinstance(cands, list) or not isinstance(presets, list) or any(not isinstance(c, dict) for c in cands) or not 1 <= len(cands) + len(presets) <= 6:
            raise ApiError("candidates is a list of 1 to 6 objects, each a change laid over the baseline (presets may stand in for some or all of them)")
        if (body.get("baseline") is None) == (body.get("from_experiment") is None):
            raise ApiError("give exactly one of baseline (a configuration object, as for /api/ml/train) and from_experiment (a saved experiment whose configuration is the baseline)")
        if body.get("baseline") is not None and not isinstance(body["baseline"], dict):
            raise ApiError("baseline is a configuration object, as for /api/ml/train")
        rule = body.get("parsimony")
        if rule is not None and not isinstance(rule, dict):
            raise ApiError("parsimony is a rule object, for example {\"rule\": \"one_se\", \"k\": 1, \"complexity\": \"n_features\"}")
        metric = body.get("primary_metric")
        if metric not in (None, "ic", "net_sharpe"):
            raise ApiError("primary_metric is ic or net_sharpe")
        min_effect, level = _float(body, "min_effect", 0.0, 0.0, 10.0), _float(body, "level", 0.05, 0.001, 0.2)
        coverage, n_boot, seed = _float(body, "minimum_coverage", 0.8, 0.0, 1.0), _int(body, "n_boot", 1000, 200, 3000), _int(body, "seed", 0, 0, 1_000_000)
        base = self._overrides(body["baseline"] if body.get("baseline") is not None else self._saved_baseline(_id(body, "from_experiment")))

        def change(c: dict) -> dict:
            ch = self._overrides(c, need_model=False)
            if "refit_every" not in (c.get("split") or {}) and "refit_every" in ch.get("split", {}):
                ch["split"].pop("refit_every")                                         # the default refit period of a model kind is not a change the user asked for
                if not ch["split"]:
                    ch.pop("split")
            return ch

        changes = []
        if presets:
            with _user_errors():
                changes.extend(change(c) for c in cmpn.preset_changes(family, base, presets, options))
        changes.extend(change(c) for c in cands)
        study = _study(body["study"]) if body.get("study") else f"improve-{family}-{cmpn.digest(base, 6)}"

        def work(progress):
            with _user_errors():
                baseline, selection = base, None
                if select:                                                                                    # the single model development would choose: each member alone, then the ensemble against the winner
                    selection = cmpn.select_baseline(self.registry, base, parse_members(options["members"]), study_id=study, config=self.app.config, builder=self.app.builder, progress=progress,
                                                     metric=metric or "ic")
                    baseline = cmpn.merge(base, {"model": {"kind": selection["selected"]["kind"], "params": selection["selected"]["params"]}})
                card = cmpn.new_card(str(body["hypothesis"]), family, baseline, changes, study_id=study, one_change_description=str(body["one_change"]), primary_metric=metric, parsimony=rule,
                                     min_effect=min_effect, level=level, minimum_coverage=coverage, n_boot=n_boot, seed=seed, baseline_selection=selection)
                cmpn.freeze(self.registry, card, actor="api")
                out = cmpn.run(self.registry, card["card_id"], config=self.app.config, builder=self.app.builder, progress=progress)
            return clean({"card_id": card["card_id"], "card": out})

        return self._submit(f"ml: {family} comparison", work)

    def _combination_run(self, body: dict, ablation: bool) -> dict:
        """The combination (or its leave-one-out ablation) of components that were each compared on their own: ``combination_of`` is a list of ``{card_id, candidate}``. The baseline, the candidates and the
        presets are the components' and cannot be given again."""
        for key in ("baseline", "from_experiment", "candidates", "presets", "members", "select_baseline", "options"):
            if body.get(key):
                raise ApiError(f"{key} does not apply to a combination: it is built from the components, which were each compared against one baseline")
        parents = body.get("combination_of")
        if not isinstance(parents, list) or not 2 <= len(parents) <= 6 or any(not isinstance(x, dict) or set(x) != {"card_id", "candidate"} for x in parents):
            raise ApiError("combination_of is a list of 2 to 6 objects {card_id, candidate}, each a candidate of a card that was run")
        metric = body.get("primary_metric")
        if metric not in (None, "ic", "net_sharpe"):
            raise ApiError("primary_metric is ic or net_sharpe")
        kw = dict(primary_metric=metric, min_effect=_float(body, "min_effect", 0.0, 0.0, 10.0), level=_float(body, "level", 0.05, 0.001, 0.2), minimum_coverage=_float(body, "minimum_coverage", 0.8, 0.0, 1.0),
                  n_boot=_int(body, "n_boot", 1000, 200, 3000), seed=_int(body, "seed", 0, 0, 1_000_000))
        if body.get("study"):
            kw["study_id"] = _study(body["study"])
        if body.get("parsimony") is not None:
            raise ApiError("a combination is compared against its baseline, not against a simpler candidate: parsimony does not apply")
        make = cmpn.ablation_card if ablation else cmpn.combination_card
        text = {"hypothesis": str(body.get("hypothesis") or "").strip(), "one_change_description": str(body.get("one_change") or "").strip()}
        if ablation:
            text = {k: v for k, v in text.items() if v}

        def work(progress):
            with _user_errors():
                card = make(self.registry, parents, **text, **kw)
                cmpn.freeze(self.registry, card, actor="api")
                out = cmpn.run(self.registry, card["card_id"], config=self.app.config, builder=self.app.builder, progress=progress)
            return clean({"card_id": card["card_id"], "card": out})

        return self._submit(f"ml: {'ablation' if ablation else 'combination'} comparison", work)

    def multiseed_run(self, body: dict) -> dict:
        """``POST /api/ml/multiseed/run {"id", "seeds", "n_boot"?}`` (a job): train the experiment's own configuration once per predeclared seed (three or more, distinct) in development mode, as looks of the same
        study, and store the spread across seeds beside each seed's time-series uncertainty. A seed is never chosen by its result; the point estimate is the median."""
        _only(body, {"id", "seeds", "n_boot"}, "/api/ml/multiseed/run")
        exp_id = _id(body)
        seeds = body.get("seeds")
        if not isinstance(seeds, list) or not 3 <= len(seeds) <= 12 or any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x <= 1_000_000 for x in seeds) or len(set(seeds)) != len(seeds):
            raise ApiError("seeds is a list of 3 to 12 distinct integers between 0 and 1000000, declared before the runs")
        n_boot = _int(body, "n_boot", 300, 200, 1000)

        def work(progress):
            with _user_errors():
                self.registry.get_experiment(exp_id)
                progress(f"training {len(seeds)} seeds in development mode")
                out = dx.multiseed(self.registry, exp_id, seeds, n_boot=n_boot, actor="api", config=self.app.config, builder=self.app.builder)
            return clean(out)

        return self._submit(f"ml: {len(seeds)} seeds for {exp_id[-6:]}", work)

    def governance_status(self, q: dict) -> dict:
        """``GET /api/ml/governance?id=<experiment>``: the strategy versions registered for an experiment (identity: model artifact, signal, execution, dataset and risk settings), the current one's risk limits,
        operating state, latest monitoring check, written recovery procedure, recorded approvals and audit trail, and the prospective ledger's counts."""
        exp_id = _id(q)
        with _user_errors():
            self.registry.get_experiment(exp_id)
            versions = gv.versions_of(self.registry, exp_id)
            current = gv.summary(self.registry, versions[-1]["version_id"]) if versions else None
            ledger = mon.ledger_view(self.registry, exp_id)
        data = {"experiment_id": exp_id, "versions": [{"version_id": v["version_id"], "created": v["created"], "actor": v["actor"], "risk_limits_hash": v["risk_limits_hash"], "settings": v["content"]["settings"]} for v in versions],
                "current": current, "ledger": ledger, "limits": {k: {"kind": v[0], "min": v[1], "max": v[2], "text": v[3]} for k, v in gv.LIMITS.items()}}
        return clean(common.envelope("ml_governance", data, {"registry": "platform"}, [], None,
                                     ["Changing the signal, execution or risk settings is a new strategy version: its evidence starts again. A deployment approval is a person's record and is tied to one version."]))

    def governance_version(self, body: dict) -> dict:
        """``POST /api/ml/governance/version {"id", "actor", "risk_limits"?}``: register the identity of an experiment as a strategy version under the given risk limits (idempotent)."""
        _only(body, {"id", "actor", "risk_limits"}, "/api/ml/governance/version")
        exp_id = _id(body)
        raw = body.get("risk_limits")
        if raw is not None and not isinstance(raw, dict):
            raise ApiError("risk_limits is an object of limits: " + ", ".join(sorted(gv.LIMITS)))
        with _user_errors():
            self.registry.get_experiment(exp_id)
            out = gv.register_version(self.registry, exp_id, actor=str(body.get("actor") or "")[:80], risk_limits=None if raw is None else gv.RiskLimits(dict(raw), "api"))
        return clean(common.envelope("ml_governance_version", out, {"registry": "platform"}, [], None, ["The platform's code hash is not part of the identity; each commitment records it."]))

    def governance_action(self, body: dict) -> dict:
        """``POST /api/ml/governance/action {"version", "action", "actor", "reason"|"note"|"procedure"|"to"}``: halt, resume, roll back, document the recovery procedure or record the separate human deployment
        approval. Every action is appended to the audit with the person's name."""
        _only(body, {"version", "action", "actor", "reason", "note", "procedure", "to", "stage"}, "/api/ml/governance/action")
        version, action, actor = _id(body, "version"), str(body.get("action") or ""), str(body.get("actor") or "")[:80]
        with _user_errors():
            if action == "halt":
                out = gv.halt(self.registry, version, str(body.get("reason") or "")[:500], actor)
            elif action == "resume":
                out = gv.resume(self.registry, version, str(body.get("note") or "")[:500], actor)
            elif action == "rollback":
                out = gv.rollback(self.registry, version, _id(body, "to"), str(body.get("reason") or "")[:500], actor)
            elif action == "recovery":
                out = {"event_seq": gv.document_recovery(self.registry, version, str(body.get("procedure") or "")[:4000], actor), "version_id": version}
            elif action == "approve":
                if body.get("stage", "production") not in ("staging", "production"):
                    raise ApiError("stage is staging or production")
                out = gv.approve_deployment(self.registry, version, actor=actor, note=str(body.get("note") or "")[:500], stage=str(body.get("stage") or "production"))
            else:
                raise ApiError("action is one of halt, resume, rollback, recovery, approve")
        return clean(common.envelope("ml_governance_action", out, {"registry": "platform"}, [], None, ["An approval records a person's decision; it does not move a model. Promotion re-checks the gates."]))

    # ------------------------------------------------------------------------------------------------------------ explanation
    def explain(self, q: dict) -> dict:
        """``GET /api/ml/explain?id=&asset=&date=``: why the saved final model scored an asset as it did on a date (Shapley values, an exact decomposition of the score), with the day's scores for every asset."""
        exp_id = _id(q)
        n_perm = _int(q, "permutations", 40, 10, 200)
        with _user_errors(), self._heavy:
            reg = self.registry
            reg.get_experiment(exp_id)
            path = reg.root / "experiments" / exp_id / "model.pkl"
            if not path.exists():
                raise ApiError("this experiment has no saved final model (it was run without one)")
            _, ds, _, _ = load_run(reg, exp_id)
            model = Model.from_bytes(path.read_bytes())
            date = pd.Timestamp(_date("date", q["date"])) if q.get("date") else ds.dates[-1]
            if date not in ds.dates:
                date = ds.dates[ds.dates.searchsorted(date, side="right") - 1] if date > ds.dates[0] else ds.dates[0]
            rows = ds.X.xs(date, level=0, drop_level=False)
            scores = pd.Series(model.predict(rows), index=rows.index.get_level_values("asset")).sort_values(ascending=False)
            asset = str(q["asset"]) if q.get("asset") else str(scores.abs().idxmax())
            if asset not in ds.assets:
                raise ApiError(f"unknown asset '{asset}'; the experiment's assets are {ds.assets}")
            shap = shap_payload(xp.explain_prediction(model, ds, date, asset, n_perm=n_perm))
        data = {"id": exp_id, "asset": asset, "date": str(date.date()), "assets": ds.assets, "scores": [{"asset": a, "score": float(v)} for a, v in scores.items()], "explanation": shap,
                "model": model.kind, "target": ds.target}
        return clean(common.envelope("ml_explain", data, {"model": "platform"}, [], date, ["Shapley values decompose this one score exactly: base + contributions = prediction. They describe the model, not the market: a feature that drives the score has not been shown to drive returns."]))

    # ------------------------------------------------------------------------------------------------------------ combination
    def combine(self, body: dict) -> dict:
        """``POST /api/ml/combine``: the strategy combination engine over saved experiments' out-of-sample returns, every method side by side."""
        ids = _ids(body, "ids", 2, LIMITS["combine"])
        methods = body.get("methods") or list(en.METHODS)
        if not isinstance(methods, list) or not methods or any(m not in en.METHODS for m in methods):
            raise ApiError(f"methods must be drawn from {list(en.METHODS)}")
        kw = {"lookback": _int(body, "lookback", 252, 60, 1000), "rebalance": _int(body, "rebalance", 21, 5, 126), "min_history": _int(body, "min_history", 126, 40, 750),
              "risk_aversion": _float(body, "risk_aversion", 5.0, 0.5, 50.0), "eta": _float(body, "eta", 5.0, 0.5, 50.0), "seed": _int(body, "seed", 0, 0, 1_000_000)}
        if body.get("development_end"):                                                                  # the best single member is chosen up to this date and each mixture is judged after it
            try:
                kw["development_end"] = str(pd.Timestamp(str(body["development_end"])).date())
            except (ValueError, TypeError):
                raise ApiError("development_end is a date, for example 2021-12-31") from None
        with _user_errors(), self._heavy:
            out = en.combine_experiments(self.registry, ids, tuple(methods), **kw)
        return clean(common.envelope("ml_combine", out, {"experiments": "platform"}, [], out["start"], [out["note"]]))

    # ------------------------------------------------------------------------------------------------------------ forward testing
    def forward(self, q: dict) -> dict:
        """``GET /api/ml/forward?id=``: the signals recorded for an experiment and their realised outcomes (read from the registry; no data is loaded)."""
        exp_id = _id(q)
        with _user_errors():
            reg = self.registry
            record = reg.get_experiment(exp_id)
            report = reg.report(exp_id)
            signals = reg.forward_signals(exp_id)
        saved = Path(reg.root) / "experiments" / exp_id / "model.qfb"
        seen = InferenceBundle.load(saved).header.get("fit_info", {}).get("label_end_max") if saved.exists() else None             # the day the saved model's labels reach: signals up to it are replays
        summary = mon.forward_record(signals, record["name"], exp_id, int(report["config"]["dataset"]["horizon"]), seen)
        rows = [{"id": s["id"], "date": s["signal_date"], "recorded": s["created"], "turnover": s["turnover"], "realised": s["realised"], "settled_through": s["realised_date"], "n_positions": len(s["weights"]),
                 "weights": s["weights"], "scores": s["scores"]} for s in signals]
        ledger = mon.ledger_view(reg, exp_id)
        return clean(common.envelope("ml_forward", {"summary": summary, "signals": rows, "backtest_ic": report["ic"]["all"].get("mean"), "rebalance": report["config"]["backtest"]["rebalance"], "ledger": ledger},
                                     {"signals": "platform"}, summary["alerts"], None, ["A signal is a paper order list: nothing is sent anywhere. 'Recorded' is when the signal was made, so a signal recorded after its date is marked as replayed."]))

    def forward_step(self, body: dict) -> dict:
        """``POST /api/ml/forward/step`` (a job): record the saved model's signal for the newest date (or ``as_of``), commit it to the immutable ledger (the ledger classifies it prospective or replay from the
        cutoffs it computes; a request cannot say), settle the commitments whose window has completed, run a monitoring check and report the live record, drift and the equity of the committed holdings."""
        _only(body, {"id", "capital", "as_of", "record"}, "/api/ml/forward/step")                    # a commitment's class, creation time and cutoffs are computed here; a request cannot state them
        exp_id = _id(body)
        capital = _float(body, "capital", 100_000.0, 1e3, 1e10)
        as_of = _date("as_of", body["as_of"]) if body.get("as_of") else None
        record_today = body.get("record", True) is not False

        def work(progress):
            with _user_errors():
                tracker = mon.ForwardTracker(self.registry, exp_id, self.app.config, self.app.builder)
                signal = None
                if record_today:
                    progress("making the signal for the newest date")
                    signal = tracker.step(None, as_of, capital)
                progress("settling matured signals")
                settled = tracker.settle(None, as_of)
                progress("measuring drift")
                summary = tracker.summary(None, as_of, monitor=True)
            return {"signal": signal, "settled": settled, "summary": summary}

        return self._submit(f"ml: forward {exp_id[-6:]}", work)
