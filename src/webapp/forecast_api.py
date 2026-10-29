"""The Forecast tab's endpoints: forecast one ticker's price as a distribution with several methods, and find out honestly how good each method has been.

``GET /api/forecast/plan`` lists the methods (what each assumes and where it breaks), the efforts and the horizons the form offers. ``POST /api/forecast/run`` starts the work as a job (kind ``forecast``, polled
through ``/api/job/<id>`` like a backtest) and ``POST /api/forecast/stop`` ends one early, keeping the forecasts scored so far. The methods, the walk-forward evaluation and the verdict are in ``src/forecast``; this module
fetches the ticker's prices from the Backtest tab's data source and puts the pieces into one report.

A forecast is not a trade, so nothing here touches the trial counter of the Backtest tab.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from ..forecast import BASELINE, BLEND, BLENDS, DEFAULT, METHODS, catalog
from ..references import FORECAST_EVALUATION, references
from ..forecast import evaluate as ev
from ..forecast.base import ForecastError, History
from ..forecast.registry import FAMILIES
from ..forecast.verdict import judge
from .api import ACTIVE, ApiError
from .universe import UniverseError, normalise

HORIZONS = (5, 10, 21, 63)
HISTORY_DAYS = 300                                         # days of past prices drawn beside the fan
MIN_DAYS = 800


def _number(raw, label: str, lo: float, hi: float, integer: bool = False):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ApiError(f"{label} must be a number") from None
    if not np.isfinite(value) or not lo <= value <= hi:
        raise ApiError(f"{label} must be between {lo:g} and {hi:g}")
    return int(round(value)) if integer else value


class ForecastApi:
    def __init__(self, app):
        self.app = app
        self.stops: dict[str, threading.Event] = {}                      # job id -> the flag that ends that evaluation early

    def get_routes(self) -> dict:
        return {"/api/forecast/plan": self.plan}

    def post_routes(self) -> dict:
        return {"/api/forecast/run": self.run, "/api/forecast/stop": self.stop}

    def plan(self, q: dict | None = None) -> dict:
        """``GET /api/forecast/plan``: the methods, grouped as the form shows them, and what the form chooses among."""
        return {"methods": [{**m, "references": references("forecaster", m["id"])} for m in catalog()], "evaluation_references": [r.view() for r in FORECAST_EVALUATION], "families": list(FAMILIES), "default": list(DEFAULT), "baseline": BASELINE, "blend": BLEND, "blends": list(BLENDS), "horizons": list(HORIZONS), "default_horizon": 21,
                "efforts": [{"id": k, "label": v["label"], "text": v["text"]} for k, v in ev.EFFORTS.items()], "default_effort": "standard", "scored_horizons": list(ev.HORIZONS), "min_days": MIN_DAYS}

    def run(self, body: dict) -> dict:
        """``POST /api/forecast/run`` (a job). ``body``: the ``ticker`` and the Backtest tab's data choices (``classes``, ``start``, ``source``, ``refresh``), the ``methods`` (ids; the random walk is always run as the
        yardstick), the ``horizon`` in trading days (5, 10, 21 or 63), ``effort`` (``quick``, ``standard`` or ``thorough``) and ``seed``. Returns ``{"job": id}``."""
        app = self.app
        raw = body.get("ticker")
        if not raw:
            tickers = body.get("tickers")
            raw = (tickers[0] if isinstance(tickers, list) and tickers else None) or app.builder.default_tickers()[0]
        try:
            ticker = normalise([raw])[0]
        except UniverseError as error:
            raise ApiError(str(error)) from None
        methods = body.get("methods", list(DEFAULT))
        if not isinstance(methods, list) or not all(isinstance(m, str) for m in methods):
            raise ApiError("methods must be a list of method names")
        unknown = [m for m in methods if m not in METHODS and m not in BLENDS]
        if unknown:
            raise ApiError(f"unknown forecasting method {unknown[0]!r}; choose from {', '.join(list(METHODS) + list(BLENDS))}")
        chosen = [m for m in dict.fromkeys(methods) if m != BASELINE]
        if not chosen:
            raise ApiError("choose at least one method besides the random walk, which is always run as the yardstick")
        horizon = _number(body.get("horizon", 21), "the horizon", 1, ev.MAX_HORIZON, True)
        if horizon not in HORIZONS:
            raise ApiError(f"the horizon must be one of {', '.join(map(str, HORIZONS))} trading days")
        effort = str(body.get("effort") or "standard")
        if effort not in ev.EFFORTS:
            raise ApiError(f"effort must be one of {', '.join(ev.EFFORTS)}")
        seed = _number(body.get("seed", 0), "the seed", 0, 1_000_000, True)
        try:
            app.builder.preflight([ticker], body.get("start"), body.get("source"))
        except UniverseError as error:
            raise ApiError(str(error)) from None
        stop = threading.Event()

        def work(progress) -> dict:
            progress("getting the prices (downloading them if they are new)")
            try:
                bundle, universe = app.builder.resolve([ticker], body.get("classes"), body.get("start"), bool(body.get("refresh")), body.get("source"), progress)
            except UniverseError as error:
                raise ApiError(str(error)) from None
            col = bundle.prices[ticker]
            history = History.from_frame(col, None if bundle.high is None else bundle.high.get(ticker), None if bundle.low is None else bundle.low.get(ticker))
            if history.n < MIN_DAYS:
                raise ApiError(f"{ticker} has {history.n} days of prices from {history.index[0].date()}; forecasting needs at least {MIN_DAYS} (about three years). Choose an earlier start date or another ticker.")
            try:
                evaluation = ev.evaluate(history, chosen, horizon, effort, seed, progress, stop.is_set)
                progress("forecasting from the last day")
                now = ev.forecast_now(history, [BASELINE] + chosen, horizon, seed=seed, progress=progress, blend_weights=evaluation.get("blend_weights"))
            except ForecastError as error:
                raise ApiError(str(error)) from None
            verdict = judge(evaluation, horizon)
            asof = history.index[-1]
            future = pd.bdate_range(asof + pd.Timedelta(days=1), periods=horizon)
            tail = history.tail(HISTORY_DAYS)
            notes = ["The future dates skip weekends and ignore exchange holidays, so a date can be a day out.",
                     "Every forecast here uses only prices up to the date it was made. The prices are adjusted for splits and dividends.",
                     "The random walk (zero average return, the spread of the last year) is the yardstick; scores are in log-return units and a method's 'skill' is how far below the random walk's average CRPS its own is."]
            if history.high is None:
                notes.append("This data has no daily highs and lows for every day, so the HAR model reads each day's variance from its squared return instead of its range, which is a noisier measure.")
            for name, why in now["errors"].items():
                notes.append(f"{METHODS[name].label if name in METHODS else name} could not forecast from the last day: {why}")
            return {"ticker": ticker, "universe": universe, "asof": str(asof.date()), "last_price": float(history.close[-1]), "horizon": horizon, "effort": effort, "methods": chosen,
                    "history": {"dates": [str(d.date()) for d in tail.index], "close": [float(v) for v in tail.close]}, "future_dates": [str(d.date()) for d in future],
                    "evaluation": evaluation, "now": now, "verdict": verdict, "notes": notes, "stopped": bool(stop.is_set()), "history_days": history.n, "first_date": str(history.index[0].date())}

        for old in [j for j in self.stops if j not in app.jobs or app.jobs[j].status not in ACTIVE]:
            self.stops.pop(old, None)
        started = app.submit_task(f"forecast: {ticker}, {len(chosen)} method{'s' if len(chosen) != 1 else ''}, {horizon} days", work, "forecast")
        self.stops[started["job"]] = stop
        return started

    def stop(self, body: dict) -> dict:
        """``POST /api/forecast/stop {"job": id}``: end an evaluation after the forecast it is on. What is scored so far is kept and reported."""
        flag = self.stops.get(str(body.get("job") or ""))
        if flag is None:
            raise ApiError("that forecast is not running in this app")
        flag.set()
        return {"ok": True}
