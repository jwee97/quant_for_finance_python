"""The Tune tab's endpoints: find good settings for a strategy on the tickers listed in the Backtest tab, and say whether to believe them.

``GET /api/tune/plan?strategy=kalman_trend`` lists the strategy's settings with a range for each (what the form shows). ``POST /api/tune/run`` starts the search as a job (kind ``tune``, polled
through ``/api/job/<id>`` like a backtest) and ``POST /api/tune/stop`` ends one early, keeping the tries so far. The search itself, the hold-out and the verdict are in ``webapp.tuning``; this module
runs the real pipeline for it, on the Backtest tab's universe, data source and start date.

Every try is a backtest of the same universe, so it is counted in the trial counter that the Backtest tab's deflated Sharpe ratio uses (``App.trials``), under the same fingerprint a backtest of those
settings would have: tuning a strategy and then running the winner there does not count it twice, and does not hide the dozens of looks it took to find it.
"""

from __future__ import annotations

import copy
import threading

import numpy as np

from ..framework import Pipeline, PipelineSpec
from ..framework.registry import MODELS
from . import tuning
from .api import ACTIVE, SLOW, ApiError, parameters
from .payload import clean
from .universe import normalise

MAX_SETTINGS = 12
HOLDOUTS = (0.2, 0.3, 0.4)


def _flag(body: dict, key: str, default: bool) -> bool:
    raw = body.get(key, default)
    if not isinstance(raw, bool):
        raise ApiError(f"{key} must be true or false")
    return raw


def _number(raw, label: str, lo: float, hi: float, integer: bool = False):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ApiError(f"{label} must be a number") from None
    if not np.isfinite(value) or not lo <= value <= hi:
        raise ApiError(f"{label} must be between {lo:g} and {hi:g}")
    return int(round(value)) if integer else value


class TuneApi:
    def __init__(self, app):
        self.app = app
        self.stops: dict[str, threading.Event] = {}                      # job id -> the flag that ends that search early

    def get_routes(self) -> dict:
        return {"/api/tune/plan": self.plan}

    def post_routes(self) -> dict:
        return {"/api/tune/run": self.run, "/api/tune/stop": self.stop}

    # ------------------------------------------------------------------------------------------------------------ the form
    @staticmethod
    def _entry(name: str):
        entry = next((e for e in MODELS.entries() if e.name == name), None)
        if entry is None:
            raise ApiError(f"unknown strategy '{name}'")
        return entry

    def _settings(self, name: str) -> tuple[list[tuning.Setting], dict]:
        entry = self._entry(name)
        params = parameters("model", name, entry.factory)
        return tuning.settings_for(name, params), {p["name"]: p["default"] for p in params}

    def plan(self, q: dict | None = None) -> dict:
        """``GET /api/tune/plan?strategy=name``: the settings the person can tune for this strategy, each with its standard value, range, help text and whether it is ticked at first, and the choices
        the form offers (how long to look, what to maximise, how much of the history to keep back)."""
        name = str((q or {}).get("strategy") or "")
        entry = self._entry(name)
        settings, _ = self._settings(name)
        return clean({
            "strategy": name, "description": entry.description, "slow": SLOW.get(name), "settings": [s.view() for s in settings],
            "note": None if settings else f"{name} has no numeric or yes/no settings that can be tuned: its settings are lists or text. Choose another strategy.",
            "efforts": [{"id": k, **{f: v[f] for f in ("label", "trials", "seconds", "text")}} for k, v in tuning.EFFORTS.items()],
            "objectives": [{"id": k, "label": v["label"], "text": v["text"], "unit": v["unit"]} for k, v in tuning.OBJECTIVES.items()],
            "holdouts": list(HOLDOUTS), "default_holdout": 0.3, "max_settings": MAX_SETTINGS, "max_trials": tuning.MAX_TRIALS, "min_trials": tuning.MIN_TRIALS,
        })

    # ------------------------------------------------------------------------------------------------------------ a search
    def _chosen(self, name: str, body: dict, available: dict[str, tuning.Setting]) -> tuple[list[tuning.Setting], dict, list[str]]:
        """The settings to search (with the person's ranges and starting values) and the ones that stay fixed; ``notes`` says what was adjusted."""
        picked = body.get("tune")
        if not isinstance(picked, list) or not picked:
            raise ApiError("tick at least one setting to tune")
        if len(picked) > MAX_SETTINGS:
            raise ApiError(f"tune at most {MAX_SETTINGS} settings at a time: more cannot be searched in a few dozen tries")
        values = {} if body.get("values") is None else body["values"]
        if not isinstance(values, dict):
            raise ApiError("values must be a mapping of setting name to value")
        notes: list[str] = []
        tuned: list[tuning.Setting] = []
        for item in picked:
            if not isinstance(item, dict):
                raise ApiError('each setting to tune must be an object with a name, such as {"name": "lookback", "low": 21, "high": 252}')
            sname = str(item.get("name") or "")
            if sname not in available:
                raise ApiError(f"'{sname}' is not a setting of {name} that can be tuned")
            if any(t.name == sname for t in tuned):
                raise ApiError(f"{sname} is listed twice")
            s = copy.copy(available[sname])
            start = self._value(s, values.get(sname, s.default), notes)
            if s.kind in ("int", "float"):
                low = _number(item.get("low", s.low), f"{s.label}: the lowest value to try", -1e12, 1e12, s.kind == "int")
                high = _number(item.get("high", s.high), f"{s.label}: the highest value to try", -1e12, 1e12, s.kind == "int")
                if not low < high:
                    raise ApiError(f"{s.label}: the lowest value to try must be below the highest")
                scale = str(item.get("scale") or s.scale)
                if scale not in ("linear", "log"):
                    raise ApiError(f"{s.label}: the scale must be linear or log")
                if s.kind == "float" and scale == "log" and (low <= 0 or start <= 0):
                    raise ApiError(f"{s.label}: a logarithmic range, and its starting value, must be above zero")
                if start < low or start > high:
                    widened = (min(low, start), max(high, start))
                    notes.append(f"The range of {s.label} was widened to {widened[0]:g} to {widened[1]:g} to include its starting value, {start:g}.")
                    low, high = widened
                s.low, s.high, s.scale = low, high, ("linear" if s.kind == "int" else scale)
            s.start, s.tune = start, True
            tuned.append(s)
        base = {}
        try:                                                                      # the starting settings are a real try: the strategy must accept them
            MODELS.create(name, **{t.name: t.start for t in tuned if t.start != t.default})
        except (TypeError, ValueError, KeyError) as error:
            raise ApiError(f"{name}: {error.args[0] if isinstance(error, KeyError) and error.args else error}") from None
        for sname, raw in values.items():
            if sname not in available:
                raise ApiError(f"'{sname}' is not a setting of {name} that can be changed here")
            if any(t.name == sname for t in tuned):
                continue
            value = self._value(available[sname], raw, notes)
            if value != available[sname].default:
                base[sname] = value
        return tuned, base, notes

    @staticmethod
    def _value(s: tuning.Setting, raw, notes: list[str]):
        """A starting or fixed value as the setting's own type, or an error that names the setting."""
        if s.kind == "bool":
            if not isinstance(raw, bool):
                raise ApiError(f"{s.label} must be true or false")
            return raw
        if s.kind == "choice":
            if raw not in s.choices:
                raise ApiError(f"{s.label} must be one of {', '.join(map(str, s.choices))}")
            return raw
        return _number(raw, s.label, -1e12, 1e12, s.kind == "int")

    def run(self, body: dict) -> dict:
        """``POST /api/tune/run`` (a job). ``body``: the Backtest tab's universe (``tickers``, ``classes``, ``start``, ``source``, ``refresh``), the ``strategy``, the settings to search as ``tune``
        (``[{name, low, high, scale}]``) and every changed setting as ``values``, ``allocator`` and ``allocator_params``, and how to search: ``objective``, ``effort`` (``quick``, ``standard``,
        ``thorough`` or ``custom`` with ``trials`` and ``seconds``), ``holdout`` (the share of days kept back), ``stable`` and ``seed``. Returns ``{"job": id}``."""
        app = self.app
        name = str(body.get("strategy") or "")
        entry = self._entry(name)
        settings, library = self._settings(name)
        tuned, base, notes = self._chosen(name, body, {s.name: s for s in settings})
        objective = str(body.get("objective") or "sharpe")
        if objective not in tuning.OBJECTIVES:
            raise ApiError(f"the measure must be one of {', '.join(tuning.OBJECTIVES)}")
        effort = str(body.get("effort") or "standard")
        if effort == "custom":
            trials = _number(body.get("trials"), "the number of tries", 2, tuning.MAX_TRIALS, True)
            seconds = _number(body.get("seconds", 600), "the time limit in seconds", 10, 3600)
        elif effort in tuning.EFFORTS:
            trials, seconds = tuning.EFFORTS[effort]["trials"], float(tuning.EFFORTS[effort]["seconds"])
        else:
            raise ApiError(f"effort must be one of {', '.join(tuning.EFFORTS)} or custom")
        holdout = _number(body.get("holdout", 0.3), "the hold-out share", 0.1, 0.5)
        stable, seed = _flag(body, "stable", True), _number(body.get("seed", 0), "the seed", 0, 1_000_000, True)
        from .universe import UniverseError

        try:
            tickers = normalise(body.get("tickers") or app.builder.default_tickers())
            app.builder.preflight(tickers, body.get("start"), body.get("source"))
        except UniverseError as error:
            raise ApiError(str(error)) from None
        spec = app.build_spec({"models": [{"name": name, "params": base}], "allocator": body.get("allocator") or None, "allocator_params": body.get("allocator_params") or {},
                               "label": f"tune {name}"}, n_tickers=len(tickers))
        spec["evaluation"]["benchmarks"] = []                                 # the comparison tables of a full run are not needed to score a try, and cost about as much as the run
        stop = threading.Event()

        def work(progress) -> dict:
            progress("building the universe (downloading any new tickers)")
            bundle, universe = app.builder.resolve(tickers, body.get("classes"), body.get("start"), bool(body.get("refresh")), body.get("source"), progress)
            key = app.universe_key(universe)
            with app._lock:
                prior = len(app.trials.get(key, ()))
            small = len(universe["tickers"]) < 5

            def trial(params: dict) -> tuning.Run:
                changed = {k: v for k, v in params.items() if k not in library or v != library[k]}      # as the Backtest tab would send them: only what differs from the standard
                one = copy.deepcopy(spec)
                one["models"][0]["params"] = changed
                digest = app.spec_digest(one)
                with app._lock:
                    app.trials.setdefault(key, set()).add(digest)
                if small:                                                       # a book of fewer than five tickers is otherwise zeroed by the platform's minimum
                    one.setdefault("execution", {})["min_assets"] = 1
                result = Pipeline(PipelineSpec.from_dict(one), app.config, bundle).run(validate=False)
                return tuning.Run(result.window, result.weights)

            with tuning.quiet("src.backtest.engine"):
                out = tuning.tune(trial, tuned, base, objective, holdout, trials, seconds, stable, seed, progress, stop.is_set, bundle.returns, prior)
            with app._lock:
                total = len(app.trials.get(key, ()))
            best = {**base, **{k: v for k, v in out["best"]["params"].items() if k not in library or v != library[k]}}
            out.update(strategy=name, description=entry.description, universe=universe, fixed=base, trials_in_universe=total, notes=notes,
                       use={"models": [{"name": name, "params": best}]}, effort=effort, label=f"{name} · {len(universe['tickers'])} tickers")
            return out

        for old in [j for j in self.stops if j not in app.jobs or app.jobs[j].status not in ACTIVE]:
            self.stops.pop(old, None)
        started = app.submit_task(f"tune: {name} on {len(tickers)} ticker{'s' if len(tickers) != 1 else ''}", work, "tune")
        self.stops[started["job"]] = stop
        return started

    def stop(self, body: dict) -> dict:
        """``POST /api/tune/stop {"job": id}``: end a search after the try it is on. The tries so far are kept, scored and reported."""
        flag = self.stops.get(str(body.get("job") or ""))
        if flag is None:
            raise ApiError("that search is not running in this app")
        flag.set()
        return {"ok": True}
