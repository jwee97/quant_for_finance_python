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
from ..ml import ensembles as en
from ..ml import explainability as xp
from ..ml import monitoring as mon
from ..ml import optimization as opt
from ..ml import strategy_search as ss
from ..ml.config import build_key, experiment_config, load_ml_config
from ..ml.datasets import UNIVERSES, build_dataset, feature_catalog, resolve_bundle
from ..ml.experiments import load_run, rebacktest, run_experiment
from ..ml.features import GROUPS
from ..ml.models import Model, default_refit, model_catalog
from ..ml.registry import STAGES, Registry
from ..ml.signals import METHODS as SIGNAL_METHODS
from ..ml.targets import CLASSIFICATION
from ..ml.targets import KINDS as TARGET_KINDS
from ..ml.training import plan_for
from ..ml.visualization import jsonable, leaderboard_payload, render_html, shap_payload
from .api import ApiError
from .payload import clean
from .universe import MAX_TICKERS, UniverseError, normalise

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
KEY = re.compile(r"^[a-z_]{1,20}\.[a-z_0-9]{1,30}$")
PARAM = re.compile(r"^[a-z_][a-z_0-9]{0,30}$")
LIMITS = {"search_random": 30, "search_bayes": 30, "search_generations": 4, "search_population": 10, "search_top_k": 4, "search_kinds": 8, "tune_trials": 40, "compare": 8, "combine": 12}

GROUP_TEXT = {"price": "Trailing returns, distance from highs and lows, moving-average gaps", "momentum": "Return over the last 1, 3, 6 and 12 months, skipping the most recent month", "mean_reversion": "Short-term reversal and distance from the mean in standard deviations",
              "volatility": "Realised volatility over several windows, its ratio, range-based estimates, downside volatility", "volume": "Dollar volume and its surprise; only where the data has volume",
              "relative": "Each asset's return and volatility relative to the universe", "macro": "Rates, spreads and the volatility index at the date, lagged by their publication delay",
              "futures": "Carry, roll yield and curve slope; only for the futures universe", "options": "Implied volatility level, skew and term structure; only where the platform has the index",
              "crypto": "Funding, basis and stablecoin growth; only for the crypto universe"}
SIGNAL_TEXT = {"top_n_long_short": "Long the n highest scores, short the n lowest, dollar neutral", "long_only_top_quantile": "Long the top quantile of scores, equal weight", "proportional": "Weight each asset in proportion to its score",
               "vol_scaled": "Proportional to score divided by volatility, so each position carries similar risk", "rank_normalised": "Weight by the cross-sectional rank of the score (robust to outliers)",
               "threshold": "Long above the threshold, short below its negative", "regime_filtered": "Trade the book only in the calm regime of the dataset's regime detector",
               "confidence_weighted": "Proportional to score times the model's confidence (for a classifier, |2p - 1|)"}
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
               "sector_cap": ("float", 0.05, 1.0), "holding_period": ("int", 1, 60), "regime_on_states": ("states",)},
    "backtest": {"rebalance": ("choice", ("daily", "weekly", "monthly")), "cost_bps": ("float", 0.0, 250.0), "slippage_bps": ("float", 0.0, 250.0), "target_vol": ("float", 0.01, 1.0), "max_leverage": ("float", 0.5, 5.0),
                 "extra_lag": ("int", 0, 20)},
}
NULLABLE = {"start", "end", "regime_detector", "sector_cap", "target_vol"}                    # an explicit null means "none", not "use the default"


def _id(q: dict, key: str = "id") -> str:
    value = str(q.get(key) or "")
    if not ID.match(value) or ".." in value:
        raise ApiError(f"{key} is not a valid identifier")
    return value


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
    except (ValueError, KeyError) as exc:
        raise ApiError(str(exc.args[0] if isinstance(exc, KeyError) and exc.args else exc)) from None


class MlApi:
    def __init__(self, app):
        self.app = app
        self.root = Path(app.root)
        self._registry: Registry | None = None
        self._lock = threading.Lock()
        self._heavy = threading.Lock()                       # explanations and combinations: numpy work of a few seconds, one request at a time

    # ------------------------------------------------------------------------------------------------------------ routes
    def get_routes(self) -> dict:
        return {"/api/ml/catalog": self.catalog, "/api/ml/experiments": self.experiments, "/api/ml/experiment": self.experiment, "/api/ml/compare": self.compare, "/api/ml/explain": self.explain,
                "/api/ml/feature-sets": self.feature_sets, "/api/ml/models": self.models, "/api/ml/trials": self.trials, "/api/ml/forward": self.forward, "/api/ml/report-html": self.report_html}

    def post_routes(self) -> dict:
        return {"/api/ml/dataset/build": self.dataset_build, "/api/ml/train": self.train, "/api/ml/rebacktest": self.rebacktest, "/api/ml/search": self.search, "/api/ml/tune": self.tune,
                "/api/ml/combine": self.combine, "/api/ml/meta-label": self.meta_label, "/api/ml/forward/step": self.forward_step, "/api/ml/promote": self.promote, "/api/ml/delete": self.delete}

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
            if kind in {m["kind"] for m in model_catalog()} and "refit_every" not in over.get("split", {}):
                over.setdefault("split", {})["refit_every"] = default_refit(kind)           # trees refit every quarter, linear models every month: the platform's own default per kind
        elif need_model:
            over["model"] = {"kind": "ridge", "params": {}}
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
        return self.app.submit_task(label, work)

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
            "defaults": {k: cfg[k] for k in ("dataset", "split", "model", "signal", "backtest", "robustness", "search")}, "stages": list(STAGES), "limits": LIMITS,
            "registry": {"root": str(Path(cfg["registry"]["root"])), **counts}, "as_of": str(self.app.builder.default_bundle().index[-1].date()),
            "timing": {"decision": "the close of day t", "trade": "lag days later (the engine's signal lag, 1)", "target": "from the close of t + lag to the close of t + lag + horizon",
                       "purge": "training rows whose label window overlaps a test day are dropped (lag + horizon + embargo)", "segments": "train (warm-up) / validation (selection) / test and forward (reserved, reported once)"}})

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
               "timing": {"lag": ds.lag, "horizon": ds.horizon, "purge": ds.lag + ds.horizon + int(cfg["split"]["embargo"])}}
        return common.envelope("ml_dataset", out, {"data": info.get("source", "platform") if info.get("source") in common.SOURCES else "mixed"},
                               ([f"{len(ds.meta.get('skipped', {}))} features could not be built from this data (see skipped)."] if ds.meta.get("skipped") else []), ds.dates[-1],
                               ["Features at the close of day t use data up to t only (every feature is tested for this); targets look forward from t + lag.", f"Information coefficients and target statistics are measured on the {scope} only."])

    def feature_sets(self, q: dict | None = None) -> dict:
        """``GET /api/ml/feature-sets``: the saved panels."""
        return clean({"feature_sets": [{k: v for k, v in f.items() if k != "path"} for f in self.registry.list_feature_sets()]})

    # ------------------------------------------------------------------------------------------------------------ training
    def train(self, body: dict) -> dict:
        """``POST /api/ml/train`` (a job): the whole experiment. Returns the experiment id and its report."""
        over = self._overrides(body)
        explain, robust = bool(body.get("explain", True)), bool(body.get("robustness", False))
        label = over.get("name") or f"{over['model']['kind']} on {over.get('dataset', {}).get('universe', 'etf')}"

        def work(progress):
            with _user_errors():
                exp = run_experiment(over, config=self.app.config, builder=self.app.builder, registry=self.registry, progress=progress, explain=explain, robustness=robust or None)
            return {"id": exp.id, "report": exp.report}

        return self._submit(f"ml: {label}", work)

    def rebacktest(self, body: dict) -> dict:
        """``POST /api/ml/rebacktest`` (a job): a saved experiment's predictions under different position rules, rebalancing, costs or with the stress tests, with no refit. Recorded as a child run."""
        exp_id = _id(body)
        extra = set(body) - {"id", "signal", "backtest", "name", "notes", "robustness"}
        if extra:
            raise ApiError(f"a re-backtest can change only the signal, backtest and robustness settings, not {sorted(extra)}: they would change the predictions (train a new model instead)")
        over = self._overrides({k: body.get(k) for k in ("signal", "backtest", "name", "notes", "robustness")}, need_model=False)
        over = {k: v for k, v in over.items() if k in ("signal", "backtest", "name", "notes", "robustness")}

        def work(progress):
            with _user_errors():
                exp = rebacktest(self.registry, exp_id, over, progress)
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
        warm-up and validation segments; the best few are finalised on every segment once, at the end, and every trial counts against their deflated Sharpe ratio."""
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

        def work(progress):
            with _user_errors():
                result = ss.search(over, config=self.app.config, builder=self.app.builder, registry=self.registry, kinds=tuple(kinds), seed=seed, progress=progress, **budget)
            return jsonable({**result, "budget": budget})

        return self._submit(f"ml: search {sum(budget[k] for k in ('n_random', 'n_bayes'))}+ trials", work)

    def tune(self, body: dict) -> dict:
        """``POST /api/ml/tune`` (a job): hyperparameters of ONE model kind, random or TPE, validation-only, and (optionally) the best set finalised on every segment."""
        over = self._overrides(body)
        kind = over["model"]["kind"]
        method = str(body.get("method") or "random")
        if method not in ("random", "tpe"):
            raise ApiError("method must be random or tpe")
        if kind not in {m["kind"] for m in model_catalog() if m["available"]}:
            raise ApiError(f"the {kind} model is not available here")
        space = ss.kind_space(kind)
        if not any(k.startswith("model.params.") for k in space):
            raise ApiError(f"the {kind} model has no hyperparameters to tune; tune its signal settings with a search instead")
        trials, finalise, seed = _int(body, "n_trials", 12, 4, LIMITS["tune_trials"]), bool(body.get("finalize", True)), _int(body, "seed", 0, 0, 1_000_000)

        def work(progress):
            with _user_errors():
                out = opt.tune(over, space, trials, method, seed, config=self.app.config, builder=self.app.builder, registry=self.registry, study_name=f"tune-{kind}", progress=progress)
                final = None
                if finalise and out.get("best"):
                    progress("running the best settings on every segment")
                    exp = opt.finalize(over, out["best"]["params"], config=self.app.config, builder=self.app.builder, registry=self.registry, trials_extra=out["n_trials"], robustness=False, progress=progress)
                    final = {"id": exp.id, "report": exp.report}
            return jsonable({**out, "space": sorted(space), "final": final})

        return self._submit(f"ml: tune {kind}", work)

    # ------------------------------------------------------------------------------------------------------------ the registry
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
        data = {"rows": leaderboard_payload(rows[:limit]), "sort": sort, "n": len(rows), "dataset_trials": trials,
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
        """``POST /api/ml/promote``: move a registered model to a stage (at most one production model per universe; the previous one is archived)."""
        model_id, stage = _id(body, "model_id"), str(body.get("stage") or "")
        if stage not in STAGES:
            raise ApiError(f"stage must be one of {list(STAGES)}")
        with _user_errors():
            self.registry.promote(model_id, stage)
        return {"ok": True, "model_id": model_id, "stage": stage}

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
                      "n_total": {k: self.registry.trial_count(k) for k in {r["dataset_key"] for r in rows}}})

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
        summary = mon.forward_record(signals, record["name"], exp_id, int(report["config"]["dataset"]["horizon"]))
        rows = [{"id": s["id"], "date": s["signal_date"], "recorded": s["created"], "turnover": s["turnover"], "realised": s["realised"], "settled_through": s["realised_date"], "n_positions": len(s["weights"]),
                 "weights": s["weights"], "scores": s["scores"]} for s in signals]
        return clean(common.envelope("ml_forward", {"summary": summary, "signals": rows, "backtest_ic": report["ic"]["all"].get("mean"), "rebalance": report["config"]["backtest"]["rebalance"]},
                                     {"signals": "platform"}, summary["alerts"], None, ["A signal is a paper order list: nothing is sent anywhere. 'Recorded' is when the signal was made, so a signal recorded after its date is marked as replayed."]))

    def forward_step(self, body: dict) -> dict:
        """``POST /api/ml/forward/step`` (a job): record the saved model's signal for the newest date (or ``as_of``), settle the signals whose window has completed, and report the live record and drift."""
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
                summary = tracker.summary(None, as_of)
            return {"signal": signal, "settled": settled, "summary": summary}

        return self._submit(f"ml: forward {exp_id[-6:]}", work)
