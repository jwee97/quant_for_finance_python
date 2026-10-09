"""The dashboard's application logic, independent of HTTP so it can be tested directly.

Everything a browser can do is a method here: list what can be run, check tickers, run a backtest as a background job, vet a formula, read a guide.
Strategies are never taken as code from the browser: formulas are parsed by ``src.strategies.expression`` and Python strategies are files you put in ``user_strategies/``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..framework import ALLOCATORS, DETECTORS, MODELS, Pipeline, PipelineSpec, load_library
from ..framework.requirements import factory_min_assets, minimum_assets, universe_problem
from ..framework.tearsheet import red_flags
from ..strategies import expression
from ..strategies.user import TEMPLATE, load_user_strategies
from .payload import clean, result_payload
from ..equity.factors import names as factor_names
from ..equity.portfolio import PRESETS as book_presets
from ..strategies.factor_models import KINDS as learner_kinds
from ..strategies.factor_timing import CALENDAR_STATES, MACRO_STATES
from .universe import CLASSES, MAX_TICKERS, TickerStore, UniverseBuilder, UniverseError, itick_source, normalise

COMBINATIONS = {
    "equal": "Average the models' forecasts",
    "confidence": "Weight each model by how confident its forecast is",
    "precision": "Weight each model by the inverse of its forecast variance",
    "ic_weighted": "Weight by each model's recent information coefficient (needs more than one model and a long history)",
    "decay_weighted": "Weight by information coefficient at the holding period, using each model's fitted alpha decay",
    "optimal_ic": "Weight to maximise the information ratio of the combination: the inverse covariance of the models' information coefficients times their means (needs more than one model and a long history)",
    "orthogonal_ic": "The same after making the models' forecasts independent of each other, each keeping only what the earlier ones did not say (needs more than one model and a long history)",
    "cost_aware": "Weight by each model's trailing net Sharpe ratio",
    "regime_conditional": "Weight by how each model has done in the current regime (needs a regime detector)",
}
CHOICES = {"mode": ["cross_sectional", "time_series"], "features": ["price", "price_macro"], "updater": ["ridge", "nlms", "kalman"],
           ("constrained_long_short", "book"): list(book_presets), ("multi_period", "book"): list(book_presets), ("ml_factor_model", "kind"): list(learner_kinds),
           ("calendar_factor_timing", "state"): list(CALENDAR_STATES), ("macro_factor_timing", "state"): list(MACRO_STATES), ("earnings_season_premium", "source"): ["auto", "file", "volume"], ("fundamental_dcf", "factor"): ["dcf_upside", "mdcf_upside", "mdcf_prob"],
           ("fundamental_value", "factor"): ["composite"] + factor_names("value"), ("fundamental_quality", "factor"): ["composite"] + factor_names("quality"),
           ("fundamental_momentum", "factor"): ["composite"] + factor_names("momentum") + factor_names("estimates"),
           ("fundamental_alpha", "weights"): ["optimal", "equal"], ("fundamental_alpha", "orthogonalize"): ["gram_schmidt", "symmetric", "none"],
           ("fundamental_alpha", "context"): ["", "value", "growth", "earnings_variability", "size"],
           ("deep_window", "kind"): ["nbeats", "nhits", "patchtst", "tsmixer", "timemixer"], ("bayesian", "kind"): ["mvo_sample", "bayes_stein", "bayes_predictive"],
           ("dynamic_cov", "model"): ["dcc", "ogarch", "static"], ("static", "book"): ["equal_weight", "inverse_vol", "risk_parity", "mean_cvar", "hrp", "herc", "mvo"],
           ("policy_portfolio", "preset"): ["60_40", "permanent", "all_weather", "bogleheads"], ("policy_portfolio", "calendar"): ["monthly", "quarterly", "semiannual", "annual", "never"]}
ALLOCATOR_TEXT = {"sleeves": "Independent sleeves: every asset is its own small strategy with an equal slice of capital times its signal, cash when flat (needs exactly one strategy; best for pullback, trend-filter and calendar rules)",
                  "score_stack": "Trade the signal as written: rank and scale the raw score, with no calibration against history (needs exactly one strategy; best for your own formulas)",
                  "forecast_stack": "Calibrated forecast: learn from matured history how much a unit of the score has paid, then size positions (a signal that has not paid gets no position)"}
SLOW = {"multi_period": "plans the next few months as a quadratic programme every month: several seconds on 15 assets, longer on 60", "constrained_long_short": "solves a quadratic programme every month: about ten seconds on 15 assets, a minute on 60", "mv_cvar": "solves a CVaR-constrained programme every month from 400 scenarios: about a minute",
        "qubo_select": "anneals a QUBO every month from 16 random starts: a minute or two", "ml_factor_model": "refits a learner every month (every third for trees and the network): seconds, longer for forests",
        "fundamental_dcf": "values every company every month, with a Monte Carlo of 400 paths for the multipath version: a few seconds", "fundamental_alpha": "builds the factors and the walk-forward weights: several seconds",
        "tca_mvo": "solves a cost-aware optimisation every month: a few seconds", "deep_window": "trains a neural network every year of history: about 10-20 seconds", "chronos": "downloads and runs a foundation model: about a minute on CPU",
        "timesfm": "runs a 200M-parameter model on CPU: many minutes on a long history", "es_policy": "trains a policy: about 10 seconds"}
DOC_ROOT_FILES = {"dashboard": "dashboard.md", "how_to_add_a_strategy": "how_to_add_a_strategy.md", "start_here": "START_HERE.md", "glossary": "glossary.md", "tour_of_a_backtest_day": "tour_of_a_backtest_day.md",
                  "roadmap_coverage": "roadmap_coverage.md", "algorithmic_trading": "algorithmic_trading.md", "platform_integration": "platform_integration.md", "feature_audit": "feature_audit.md"}
SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,80}$")

# What the page needs from this server. 1 was the first catalogue; 2 added data sources (``sources``), the ticker-count check (``/api/requirements``), earnings and trades; 3 added the Execution and
# Cash flows tabs (``/api/exec``, ``/api/cash``). The page compares it with its own number and says so when the server behind it is older (the files were updated while the app was running, so it
# still runs the old code): the whole page needs 2, and the two new tabs ask for 3 on their own, so an older app still runs the Backtest tab.
API_VERSION = 3
CODE_ROOT = Path(__file__).resolve().parents[1]


def code_signature(root: Path = CODE_ROOT) -> tuple[int, float]:
    """``(number of Python files, newest modification time)`` under ``root``. Compared with the value taken when this process started, a difference means the code changed on disk after the
    app loaded it, so the app is running something older than what the files now say (comparing the files with each other, not with the clock, so a skewed clock cannot fake it)."""
    count, newest = 0, 0.0
    for path in Path(root).rglob("*.py"):
        try:
            newest = max(newest, path.stat().st_mtime)
            count += 1
        except OSError:
            continue
    return count, newest


LAUNCH_SIGNATURE = code_signature()


class ApiError(ValueError):
    """A request the user can fix; the message is shown in the page."""


def _kind_of(default) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    return "json"


def parameters(registry_kind: str, name: str, factory) -> list[dict]:
    """The constructor arguments of a plug-in as form fields: name, kind, default, choices."""
    try:
        signature = inspect.signature(factory.__init__ if inspect.isclass(factory) else factory)
    except (TypeError, ValueError):
        return []
    out = []
    for p in signature.parameters.values():
        if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        default = None if p.default is inspect.Parameter.empty else p.default
        choices = CHOICES.get((name, p.name)) or CHOICES.get(p.name)
        out.append({"name": p.name, "kind": "choice" if choices else _kind_of(default), "default": list(default) if isinstance(default, tuple) else default,
                    "choices": choices, "required": p.default is inspect.Parameter.empty})
    return out


class Job:
    def __init__(self, label: str):
        self.id, self.label, self.status, self.message = uuid.uuid4().hex[:12], label, "queued", "waiting for a free worker"
        self.created, self.result, self.error = time.time(), None, None

    def view(self) -> dict:
        return {"id": self.id, "label": self.label, "status": self.status, "message": self.message, "error": self.error, "result": self.result,
                "elapsed": round(time.time() - self.created, 1)}


class App:
    def __init__(self, config, store: TickerStore | None = None, builder: UniverseBuilder | None = None, root: Path | None = None):
        self.config, self.root = config, Path(root or config.root)
        load_library()
        self.builder = builder or UniverseBuilder(config, store or TickerStore(self.root / "data" / "user" / "prices"), sources={"itick": itick_source(self.root)})
        self.jobs: dict[str, Job] = {}
        self.trials: dict[str, set] = {}
        self._pool = ThreadPoolExecutor(max_workers=1)             # one at a time: the models set process-wide torch flags and share the data caches
        self._lock = threading.Lock()
        self._lab_lock = threading.Lock()                          # the Execution and Cash flows tabs: numpy work of a few seconds, one request at a time so a click storm cannot use every core

    # ------------------------------------------------------------------------------------------------------ catalogue
    def catalog(self) -> dict:
        models = []
        for e in MODELS.entries():
            if e.family == "crypto":
                continue
            instance_requires = list(getattr(e.factory, "requires", ()) or ())
            models.append({"name": e.name, "family": e.family or "other", "description": e.description, "params": parameters("model", e.name, e.factory), "requires": instance_requires,
                           "slow": SLOW.get(e.name), "user": not getattr(e.factory, "__module__", "").startswith("src.strategies"), "structured": bool(getattr(e.factory, "structured", False)),
                           "book": getattr(e.factory, "book", None), "rebalance": getattr(e.factory, "rebalance", None), "position_mode": getattr(e.factory, "position_mode", None),
                           "min_assets": factory_min_assets(e.factory)})
        allocators = [{"name": e.name, "description": e.description, "params": parameters("allocator", e.name, e.factory), "slow": SLOW.get(e.name)} for e in ALLOCATORS.entries()
                      if e.name not in ("model_weights", "regime_switch")]
        for a in allocators:
            a["description"] = ALLOCATOR_TEXT.get(a["name"], a["description"])
        detectors = [{"name": e.name, "description": e.description} for e in DETECTORS.entries() if e.name != "static"]
        default = self.builder.default_bundle()
        return clean({
            "api_version": API_VERSION, "restart_needed": code_signature() != LAUNCH_SIGNATURE,
            "models": models, "allocators": allocators, "detectors": detectors, "combinations": [{"name": k, "description": v} for k, v in COMBINATIONS.items()],
            "default_tickers": list(default.assets), "default_classes": dict(default.asset_class), "classes": list(CLASSES), "max_tickers": MAX_TICKERS,
            "sources": [src.status() for src in self.builder.sources.values()], "default_source": "yahoo",
            "data_range": [str(default.index[0].date()), str(default.index[-1].date())],
            "formula": {"functions": [{"name": k, "signature": v[0], "help": v[1]} for k, v in expression.FUNCTIONS.items()],
                        "examples": [{"title": t, "expr": e, "mode": m} for t, e, m in expression.EXAMPLES]},
            "template": TEMPLATE.format(title="My idea", name="my_idea", cls="MyIdea"), "user_folder": "user_strategies",
        })

    def reload_strategies(self) -> dict:
        return {"files": load_user_strategies(self.root, reload=True), "models": [e.name for e in MODELS.entries()]}

    # ------------------------------------------------------------------------------------------------------ tickers
    def check_tickers(self, body: dict) -> dict:
        try:
            return self.builder.check(body.get("tickers") or [], body.get("start") or None, bool(body.get("refresh")), body.get("source"))
        except UniverseError as error:
            raise ApiError(str(error)) from None

    def test_source(self, body: dict) -> dict:
        """Check that a data source works with one cheap call (for iTick: the key, the host and the daily interval). Never returns the key."""
        try:
            src = self.builder.source(body.get("source"))
        except UniverseError as error:
            raise ApiError(str(error)) from None
        test = getattr(src.provider, "test", None)
        return clean(test() if test else {"ok": True, "message": f"{src.label} needs no key"})

    # ------------------------------------------------------------------------------------------------------ formulas
    def check_formula(self, body: dict) -> dict:
        formula, mode = str(body.get("expr") or ""), str(body.get("mode") or "cross_sectional")
        try:
            model = MODELS.create("expression", expr=formula, mode=mode)
            tickers, source = body.get("tickers") or self.builder.default_tickers(), body.get("source")
            waiting = self.builder.waiting(tickers, body.get("start"), source)
            if waiting:
                label = self.builder.source(source).label
                return {"ok": False, "error": f"{', '.join(waiting)} {'has' if len(waiting) == 1 else 'have'} not been downloaded from {label} yet. Run a backtest once (it downloads them, within the rate limit), then check the formula."}
            bundle, info = self.builder.resolve(tickers, body.get("classes"), body.get("start"), False, source)
            score = model.score(bundle).dropna(how="all")
        except (expression.FormulaError, ValueError, UniverseError) as error:
            return {"ok": False, "error": str(error)}
        if score.empty:
            return {"ok": False, "error": "the formula produced no values (the windows may be longer than the history)"}
        latest = score.iloc[-1].dropna().sort_values(ascending=False)
        return clean({"ok": True, "as_of": str(score.index[-1].date()), "first": str(score.index[0].date()), "coverage": float(score.notna().mean().mean()),
                      "latest": [{"ticker": t, "score": v} for t, v in latest.items()], "universe": info})

    # ------------------------------------------------------------------------------------------------------ backtests
    def _models(self, body: dict) -> tuple[list[dict], list]:
        """The strategies a request names as ``[{"name", "params"}]`` and as built instances; raises ``ApiError`` for one that does not exist or does not accept its settings."""
        models = body.get("models") or []
        if not 1 <= len(models) <= 4:
            raise ApiError("choose between one and four strategies")
        spec_models, instances = [], []
        for m in models:
            name, params = str(m.get("name") or ""), dict(m.get("params") or {})
            if name not in MODELS:
                raise ApiError(f"unknown strategy '{name}'")
            try:
                instances.append(MODELS.create(name, **params))
            except (TypeError, ValueError, KeyError) as error:
                raise ApiError(f"{name}: {error}") from None
            spec_models.append({"name": name, "params": params})
        return spec_models, instances

    def _allocation(self, body: dict, n_models: int) -> dict | None:
        if not body.get("allocator"):
            return None
        name = str(body["allocator"])
        if name not in ALLOCATORS:
            raise ApiError(f"unknown allocator '{name}'")
        params = dict(body.get("allocator_params") or {})
        if name in ("score_stack", "sleeves") and n_models != 1:
            raise ApiError(f"'{name}' needs exactly one strategy")
        try:
            ALLOCATORS.create(name, **params)
        except (TypeError, ValueError, KeyError) as error:
            raise ApiError(f"{name}: {error}") from None
        return {"allocator": name, "params": params}

    def requirements(self, body: dict) -> dict:
        """How many tickers the chosen strategies and allocation need and whether ``body["tickers"]`` (a count) is enough: the page uses it to explain a disabled Run button."""
        _, instances = self._models(body)
        allocation = self._allocation(body, len(instances))
        need, reasons = minimum_assets(instances, allocation)
        try:
            have = int(body.get("tickers") or 0)
        except (TypeError, ValueError):
            raise ApiError("tickers must be a count") from None
        problem = "Choose at least one ticker." if have < 1 else universe_problem(instances, allocation, have)
        return {"need": need, "have": have, "ok": problem is None, "message": problem, "reasons": reasons}

    def build_spec(self, body: dict, n_tickers: int | None = None) -> dict:
        """The pipeline specification for a request. With ``n_tickers`` it also refuses a universe too small for the chosen strategies (a ranking needs at least two tickers)."""
        spec_models, instances = self._models(body)
        spec: dict = {"name": str(body.get("label") or " + ".join(m["name"] for m in spec_models))[:80], "models": spec_models}
        if len(spec_models) > 1:
            rule = str(body.get("combination") or "equal")
            if rule not in COMBINATIONS:
                raise ApiError(f"unknown combination rule '{rule}'")
            spec["combination"] = {"rule": rule}
        allocation = self._allocation(body, len(spec_models))
        if allocation:
            spec["allocation"] = allocation
        if n_tickers is not None:
            problem = universe_problem(instances, allocation, n_tickers)
            if problem:
                raise ApiError(problem)
        if body.get("regime"):
            if str(body["regime"]) not in DETECTORS:
                raise ApiError(f"unknown regime detector '{body['regime']}'")
            spec["regime"] = {"detector": str(body["regime"])}
            if body.get("regime_risk"):
                spec["risk"] = {"mode": "regime"}
        elif body.get("regime_risk"):
            raise ApiError("a volatility target by regime needs a regime detector")
        if spec.get("combination", {}).get("rule") == "regime_conditional" and "regime" not in spec:
            raise ApiError("the regime-conditional rule needs a regime detector")
        if body.get("aum"):
            aum = float(body["aum"])
            if not 1e4 <= aum <= 1e12:
                raise ApiError("assets under management must be between $10 thousand and $1 trillion")
            spec["execution"] = {"aum": aum}
        spec["evaluation"] = {"benchmarks": ["equal_weight", "risk_parity"], "causality": bool(body.get("validate", False))}
        if body.get("explain"):
            spec["evaluation"]["explain"] = True
        return spec

    @staticmethod
    def _capital(body: dict) -> float:
        """The starting capital the earnings are shown on: the request's ``capital``, else the assets under management if given, else $100,000."""
        raw = body.get("capital") or body.get("aum") or 100_000
        try:
            capital = float(raw)
        except (TypeError, ValueError):
            raise ApiError("starting capital must be a number") from None
        if not 1e3 <= capital <= 1e12:
            raise ApiError("starting capital must be between $1 thousand and $1 trillion")
        return capital

    def submit(self, body: dict) -> dict:
        try:
            tickers = normalise(body.get("tickers") or self.builder.default_tickers())
            self.builder.preflight(tickers, body.get("start"), body.get("source"))
        except UniverseError as error:
            raise ApiError(str(error)) from None
        spec = self.build_spec(body, n_tickers=len(tickers))
        capital = self._capital(body)
        job = Job(spec["name"])
        with self._lock:
            self.jobs[job.id] = job
            for old in list(self.jobs)[:-60]:
                self.jobs.pop(old, None)
        self._pool.submit(self._run, job, spec, tickers, body, capital)
        return {"job": job.id}

    def _run(self, job: Job, spec: dict, tickers: list[str], body: dict, capital: float = 100_000.0) -> None:
        def progress(text: str) -> None:
            job.message = text

        try:
            job.status, job.message = "running", "building the universe (downloading any new tickers)"
            started = time.perf_counter()
            bundle, universe = self.builder.resolve(tickers, body.get("classes"), body.get("start"), bool(body.get("refresh")), body.get("source"), progress)
            key = hashlib.sha1(json.dumps([universe["tickers"], universe["first"]], sort_keys=True).encode()).hexdigest()[:10]
            digest = hashlib.sha1(json.dumps({k: v for k, v in spec.items() if k not in ("name", "evaluation")}, sort_keys=True, default=str).encode()).hexdigest()[:12]
            with self._lock:
                seen = self.trials.setdefault(key, set())
                seen.add(digest)
                spec["evaluation"]["n_trials"] = len(seen)
            if len(universe["tickers"]) < 5:                       # the platform default of five zeroes a smaller book, and the risk-parity benchmark needs five assets
                spec.setdefault("execution", {})["min_assets"] = 1
                spec["evaluation"]["benchmarks"] = ["equal_weight"]
            job.message = "running the pipeline"
            pipeline = Pipeline(PipelineSpec.from_dict(spec), self.config, bundle)
            result = pipeline.run(validate=bool(spec["evaluation"]["causality"]))
            spec_view = PipelineSpec.from_dict(spec).to_dict()
            job.result = result_payload(result, bundle, red_flags(result), universe, spec_view, yaml.safe_dump(spec_view, sort_keys=False), time.perf_counter() - started, capital)
            job.result["trials_in_universe"] = len(seen)
            if job.result["held_days"] < 20 or not np.isfinite(result.metrics.get("sharpe", np.nan)):
                job.result["warning"] = ("This strategy held (almost) no positions. With the calibrated forecast the framework holds nothing when a signal has not paid in the past; "
                                         "choose 'Trade the signal as written' under Portfolio, or check that the signal has enough history.")
            job.status, job.message = "done", "finished"
        except (ApiError, UniverseError, ValueError, KeyError) as error:
            job.status, job.error = "error", str(error.args[0] if isinstance(error, KeyError) and error.args else error)
        except Exception as error:                                   # a model bug must reach the user as a message, not a hung page
            traceback.print_exc()
            job.status, job.error = "error", f"{type(error).__name__}: {error}"

    def job(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise ApiError("unknown job")
        return job.view()

    def reset_trials(self) -> dict:
        with self._lock:
            self.trials.clear()
        return {"ok": True}


    # ------------------------------------------------------------------------------------------------------ the execution lab
    EXEC_ACTIONS = ("run", "frontier", "basket", "hft")

    def exec_lab(self, body: dict) -> dict:
        """The Execution tab: run algorithms on a simulated day (``run``), trace the cost-risk frontier (``frontier``), work a demonstration basket (``basket``) or run a high-frequency simulation
        (``hft``). Everything is computed on a stylised market from the numbers in the request; no prices are downloaded."""
        from ..algo import factory

        action = str(body.get("action") or "run")
        if action not in self.EXEC_ACTIONS:
            raise ApiError(f"action must be one of {list(self.EXEC_ACTIONS)}")
        seed = _number(body, "seed", 0, 0, 1_000_000, integer=True)
        market = dict(side=_choice(body, "side", "buy", ("buy", "sell")), shares=_number(body, "shares", 200_000, 1, 1e9), adv=_number(body, "adv", 2_000_000, 1e3, 1e10),
                      sigma=_number(body, "sigma", 0.02, 0.001, 0.2), price=_number(body, "price", 50.0, 0.01, 1e5), spread_bps=_number(body, "spread_bps", 4.0, 0, 200),
                      intervals=_number(body, "intervals", 26, 4, 100, integer=True))
        with self._lab_lock:
            try:
                if action == "run":
                    algos = body.get("algos") or ["twap", "vwap"]
                    if not isinstance(algos, (str, list, tuple)):
                        raise ApiError("algos must be a list of algorithm names")
                    algos = algos.replace(",", " ").split() if isinstance(algos, str) else [str(a) for a in algos]
                    if not 1 <= len(algos) <= 8 or any(len(a) > 80 for a in algos):
                        raise ApiError("choose between one and eight algorithms")
                    target = body.get("target_bps")
                    out = factory.run_order(algos, style=_choice(body, "style", "aggressive", ("aggressive", "working", "passive")), paths=_number(body, "paths", 300, 20, 2000, integer=True), seed=seed,
                                            scenario=_choice(body, "scenario", "normal", ("normal", "trend_up", "trend_down", "mean_reverting", "crisis")),
                                            max_participation=_number(body, "max_participation", 0.35, 0.01, 1.0), target_bps=None if target in (None, "") else _number(body, "target_bps", 0, -1000, 1000), **market)
                    out["inputs"] = {**market, "seed": seed}
                elif action == "frontier":
                    out = factory.efficient_frontier(**market)
                    out["inputs"] = dict(market)
                elif action == "basket":
                    inputs = dict(size=_number(body, "size", 8, 2, 16, integer=True), seed=seed, risk_aversion=_number(body, "risk_aversion", 1e-3, 1e-6, 10.0),
                                  block_threshold=_number(body, "block_threshold", 0.005, 1e-4, 1.0), share=_number(body, "share", 0.5, 0.0, 1.0))
                    out = factory.basket_report(**inputs)
                    out["inputs"] = inputs
                else:
                    out = factory.hft_report(_choice(body, "sim", "etf", factory.HFT_SIMS), seed)
                    out["inputs"] = {"sim": out["sim"], "seed": seed}
            except (ValueError, KeyError) as error:
                raise ApiError(str(error.args[0] if isinstance(error, KeyError) and error.args else error)) from None
        return clean(out)

    # ------------------------------------------------------------------------------------------------------ the cash-flow lab
    CASH_ACTIONS = ("simulate", "spending", "redeem", "ldi")

    def _mix(self, body: dict, bundle) -> pd.Series:
        """The portfolio the cash-flow tools work on: the weights in the request, else an equal weight of the tickers."""
        tickers = [t for t in normalise(body.get("tickers") or self.builder.default_tickers()) if t in bundle.prices.columns]
        given = body.get("weights") or {}
        if not isinstance(given, dict):
            raise ApiError("weights must be a mapping from ticker to weight")
        if given:
            try:
                w = pd.Series({str(k).upper(): float(v) for k, v in given.items() if str(k).upper() in tickers})
            except (TypeError, ValueError):
                raise ApiError("weights must be numbers") from None
            if w.empty or (w < 0).any() or w.sum() <= 0:
                raise ApiError("weights must be non-negative, add up to more than zero, and name tickers that are in the list")
        else:
            w = pd.Series(1.0, index=tickers)
        return w / w.sum()

    def cash_lab(self, body: dict) -> dict:
        """The Cash flows tab: a portfolio followed through deposits, withdrawals and dividends under each policy (``simulate``), payments out of it under spending rules (``spending``), the cost of a
        redemption (``redeem``) and liability-driven investing (``ldi``). The portfolio is made of the tickers already downloaded for the Backtest tab."""
        from ..cashflow import flows, liabilities, redemption, simulate, spending
        from ..cashflow.policies import POLICIES

        action = str(body.get("action") or "simulate")
        if action not in self.CASH_ACTIONS:
            raise ApiError(f"action must be one of {list(self.CASH_ACTIONS)}")
        source = body.get("source")
        try:
            tickers = normalise(body.get("tickers") or self.builder.default_tickers())
            waiting = self.builder.waiting(tickers, body.get("start"), source)
            if waiting:
                label = self.builder.source(source).label
                raise ApiError(f"{', '.join(waiting)} {'has' if len(waiting) == 1 else 'have'} not been downloaded from {label} yet. Run a backtest once (it downloads them, within the rate limit), then use this tab.")
            bundle, info = self.builder.resolve(tickers, body.get("classes"), body.get("start"), False, source)
        except UniverseError as error:
            raise ApiError(str(error)) from None
        mix = self._mix(body, bundle)
        prices = bundle.prices[list(mix.index)].dropna()
        if len(prices) < 260:
            raise ApiError("the chosen tickers share less than a year of prices")
        week = lambda s: [None if not math.isfinite(v) else round(float(v), 4) for v in s.iloc[::5].to_numpy(dtype=float)]            # noqa: E731  (weekly points keep the page light)
        head = {"mix": {k: float(v) for k, v in mix.items()}, "period": [str(prices.index[0].date()), str(prices.index[-1].date())], "action": action}
        with self._lab_lock:
            try:
                if action == "simulate":
                    chosen = body.get("policies") or ["pro_rata", "correct_drift", "rebalance", "cash"]
                    if not isinstance(chosen, list) or not 1 <= len(chosen) <= 5 or any(p not in POLICIES for p in chosen):
                        raise ApiError(f"policies must be 1 to 5 of {list(POLICIES)}")
                    deposit, withdraw, growth = _number(body, "deposit", 0, 0, 1e9), _number(body, "withdraw", 0, 0, 1e9), _number(body, "growth", 0.0, -0.5, 1.0)
                    initial, rebalance = _number(body, "initial", 100_000, 1e3, 1e12), _choice(body, "rebalance", "none", simulate.REBALANCE)
                    dividend_yield, dividend_policy = _number(body, "dividend_yield", 0.0, 0.0, 0.2), _choice(body, "dividend_policy", "flow", simulate.DIVIDEND_POLICIES)
                    cost_bps, dca_months = _number(body, "cost_bps", 5.0, 0.0, 200.0), _number(body, "dca_months", 1, 1, 36, integer=True)
                    schedule = flows.flow_schedule(prices.index, deposit=deposit, withdraw=withdraw, growth=growth)
                    rows, nav, dev = [], {}, {}
                    for policy in chosen:
                        r = simulate.simulate_cashflows(prices, mix, schedule, initial=initial, inflow_policy=policy, outflow_policy=policy, rebalance=rebalance, dividend_yield=dividend_yield or None,
                                                        dividend_policy=dividend_policy, cost_bps=cost_bps, dca_months=dca_months)
                        keep = ("final_value", "profit", "irr", "twr_annual", "turnover", "costs", "dividends", "mean_deviation", "max_deviation", "mean_cash", "deposits", "withdrawals")
                        rows.append({"policy": policy, **{k: r.summary[k] for k in keep}})
                        nav[policy], dev[policy] = week(r.nav), week(r.deviation)
                    out = {**head, "rows": rows, "dates": [str(d.date()) for d in prices.index[::5]], "nav": nav, "deviation": dev,
                           "inputs": {"initial": initial, "deposit": deposit, "withdraw": withdraw, "growth": growth, "rebalance": rebalance, "dividend_yield": dividend_yield, "dividend_policy": dividend_policy,
                                      "cost_bps": cost_bps, "dca_months": dca_months}}
                elif action == "spending":
                    rules = body.get("rules") or list(spending.RULES)
                    if not isinstance(rules, list) or not 1 <= len(rules) <= 4 or any(x not in spending.RULES for x in rules):
                        raise ApiError(f"rules must be 1 to 4 of {list(spending.RULES)}")
                    daily = (prices.pct_change().dropna() @ mix).to_numpy()
                    # The futures are drawn once and every rule faces the same ones (the draw is the slow part: paths x years x 252 numbers, so the limits keep a request to a few seconds and about a gigabyte).
                    args = dict(initial=_number(body, "initial", 1_000_000, 1e3, 1e12), years=_number(body, "years", 30, 1, 50, integer=True), paths=_number(body, "paths", 1000, 100, 2000, integer=True),
                                inflation=_number(body, "inflation", 0.02, -0.05, 0.2), seed=_number(body, "seed", 0, 0, 1_000_000, integer=True))
                    rate, target_ruin = _number(body, "rate", 0.04, 0.001, 0.5), _number(body, "target_ruin", 0.05, 0.001, 0.5)
                    annual = spending.bootstrap_annual_returns(daily, args["years"], args["paths"], seed=args["seed"])
                    shared = dict(annual_returns=annual, initial=args["initial"], years=args["years"], inflation=args["inflation"])
                    results = {rule: spending.simulate_spending(rule=rule, rate=rate, **shared) for rule in rules}
                    safe = spending.sustainable_rate(target_ruin=target_ruin, rule="fixed_real", **shared)
                    out = {**head, "rate": rate, "years": args["years"], "inflation": args["inflation"], "initial": args["initial"], "paths": args["paths"], "days": int(len(daily)), "sustainable_rate": safe,
                           "target_ruin": target_ruin,
                           "rows": [{"rule": k, **{m: v[m] for m in ("ruin_probability", "spending_cut_probability", "mean_real_spending", "years_funded")},
                                     "median_final_real_wealth": v["final_real_wealth"][0.5], "p05_final_real_wealth": v["final_real_wealth"][0.05]} for k, v in results.items()],
                           "bands": {k: v["bands"] for k, v in results.items()}}
                elif action == "redeem":
                    cash = _number(body, "cash", 0.05, 0.0, 0.9)
                    weights = mix * (1.0 - cash)
                    if bundle.volume is not None:
                        dollars = (bundle.prices[list(mix.index)] * bundle.volume[list(mix.index)]).tail(60).median()
                        note = "Dollar volume is the median of the last 60 days in the data."
                    else:
                        dollars, note = pd.Series(1e8, index=mix.index), "The data has no volume; every asset is assumed to trade $100 million a day."
                    sigma = bundle.returns[list(mix.index)].tail(252).std()
                    aum, share, participation = _number(body, "aum", 1e9, 1e5, 1e13), _number(body, "redemption", 0.1, 0.001, 0.9), _number(body, "participation", 0.1, 0.01, 1.0)
                    table = redemption.compare_redemption_policies(weights, aum, share, dollars, sigma, policies=("pro_rata", "liquid", "cash"), cash=cash, participation=participation)
                    out = {**head, "cash": cash, "note": note, "rows": [{"policy": str(i), **{c: float(v) for c, v in row.items()}} for i, row in table.iterrows()],
                           "inputs": {"aum": aum, "redemption": share, "cash": cash, "participation": participation}}
                else:
                    if "DGS10" not in bundle.macro.columns or bundle.macro["DGS10"].dropna().empty:
                        raise ApiError("liability-driven investing needs the 10-year Treasury yield (DGS10), which this data does not have")
                    if not all(isinstance(body.get(k) or [], (list, tuple)) for k in ("hedge", "seeking")):
                        raise ApiError("hedge and seeking are lists of tickers")
                    hedge, seeking = [str(t).upper() for t in body.get("hedge") or []], [str(t).upper() for t in body.get("seeking") or []]
                    if not hedge or not seeking or any(t not in bundle.prices.columns for t in hedge + seeking):
                        raise ApiError("name at least one bond fund to hedge with and one return-seeking fund, from the tickers in the list")
                    years = _number(body, "liability_years", 25, 5, 60, integer=True)
                    liab = liabilities.Liability.level(100.0, years)
                    fr0 = _number(body, "funding_ratio", 0.85, 0.3, 3.0)
                    frames, rows = {}, []
                    for label, ratio in (("unhedged", 0.0), ("glide path", "glide"), ("fully hedged", 1.0)):
                        o = liabilities.simulate_ldi(bundle.prices, bundle.macro["DGS10"], liab, hedge, seeking, initial_funding_ratio=fr0, hedge_ratio=ratio, pay=False)
                        frames[label] = o["frame"]
                        rows.append({"plan": label, **o["summary"]})
                    first = next(iter(frames.values()))
                    out = {**head, "hedge": hedge, "seeking": seeking, "rows": rows, "dates": [str(d.date()) for d in first.index[::5]], "funding_ratio": {k: week(v["funding_ratio"]) for k, v in frames.items()},
                           "hedge_weight": {k: week(v["w_hedge"]) for k, v in frames.items()}, "inputs": {"funding_ratio": fr0, "liability_years": years, "payment": 100.0}}
            except (ValueError, KeyError) as error:
                raise ApiError(str(error.args[0] if isinstance(error, KeyError) and error.args else error)) from None
        out["universe"] = {"tickers": info["tickers"], "source": info.get("source")} if isinstance(info, dict) else None
        return clean(out)

    # ------------------------------------------------------------------------------------------------------ guides
    def docs_index(self) -> dict:
        docs = self.root / "docs"
        pinned = [{"slug": slug, "title": _title(docs / f), "group": "Start"} for slug, f in DOC_ROOT_FILES.items() if (docs / f).exists()]
        guides = []
        for path in sorted((docs / "techniques").glob("*.md")):
            if path.stem == "index":
                continue
            meta = _front(path)
            guides.append({"slug": f"technique-{path.stem}", "title": meta.get("title", path.stem), "group": "Technique guides", "difficulty": meta.get("difficulty", 3)})
        guides.sort(key=lambda g: (g["difficulty"], g["title"]))
        cards = [{"slug": f"strategy-{p.stem}", "title": p.stem, "group": "Strategy cards"} for p in sorted((docs / "strategies").glob("*.md")) if p.stem != "index"]
        return {"docs": pinned + guides + cards}

    def doc(self, slug: str) -> dict:
        if not SAFE_SLUG.match(slug):
            raise ApiError("unknown guide")
        docs = self.root / "docs"
        if slug in DOC_ROOT_FILES:
            path = docs / DOC_ROOT_FILES[slug]
        elif slug.startswith("technique-"):
            path = docs / "techniques" / f"{slug.removeprefix('technique-')}.md"
        elif slug.startswith("strategy-"):
            path = docs / "strategies" / f"{slug.removeprefix('strategy-')}.md"
        else:
            raise ApiError("unknown guide")
        if not path.is_file() or docs.resolve() not in path.resolve().parents:
            raise ApiError("unknown guide")
        text = path.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            text = text.split("\n---\n", 1)[1] if "\n---\n" in text else text
        return {"slug": slug, "title": _title(path), "markdown": text}


def _number(body: dict, key: str, default, lo: float, hi: float, integer: bool = False) -> float:
    """A number from a request, with its default and its range; the message says which field is wrong."""
    raw = body.get(key, default)
    raw = default if raw is None or raw == "" else raw
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ApiError(f"{key.replace('_', ' ')} must be a number") from None
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ApiError(f"{key.replace('_', ' ')} must be between {lo:g} and {hi:g}")
    return int(round(value)) if integer else value


def _choice(body: dict, key: str, default: str, allowed) -> str:
    value = body.get(key)
    value = default if value in (None, "") else str(value)
    if value not in allowed:
        raise ApiError(f"{key.replace('_', ' ')} must be one of {sorted(allowed)}")
    return value


def _front(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text:
        return {}
    try:
        return yaml.safe_load(text.split("\n---\n", 1)[0].removeprefix("---\n")) or {}
    except yaml.YAMLError:
        return {}


def _title(path: Path) -> str:
    meta = _front(path)
    if meta.get("title"):
        return str(meta["title"])
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem

