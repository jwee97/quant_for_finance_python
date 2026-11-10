"""The Scenarios tab's endpoints: make thousands of possible futures for a set of tickers with several generators, from simple resampling to a GAN, a variational autoencoder and a diffusion model, and check how
closely each one's scenarios resemble the history they were made from.

``GET /api/scenarios/plan`` lists the generators (what each assumes and where it breaks) and what the form offers. ``POST /api/scenarios/run`` starts the work as a job (kind ``scenarios``, polled through
``/api/job/<id>``). The generators and the scores are in ``src/scenarios``; this module fetches the tickers' prices from the Backtest tab's data source and puts the pieces into one record.

Scenarios are not trades, so nothing here touches the trial counter of the Backtest tab.
"""

from __future__ import annotations

import threading

from ..scenarios import report
from .api import ACTIVE, ApiError
from .universe import UniverseError, normalise


def _integer(raw, label: str, lo: int, hi: int) -> int:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ApiError(f"{label} must be a number") from None
    if value != value or not lo <= value <= hi or value != int(value):
        raise ApiError(f"{label} must be a whole number between {lo} and {hi}")
    return int(value)


class ScenariosApi:
    def __init__(self, app):
        self.app = app
        self.stops: dict[str, threading.Event] = {}

    def get_routes(self) -> dict:
        return {"/api/scenarios/plan": self.plan}

    def post_routes(self) -> dict:
        return {"/api/scenarios/run": self.run, "/api/scenarios/stop": self.stop}

    def plan(self, q: dict | None = None) -> dict:
        """``GET /api/scenarios/plan``: the generators, grouped as the form shows them, and what the form chooses among."""
        return {"generators": report.catalog(), "families": report.FAMILIES, "default": report.DEFAULT, "horizons": list(report.HORIZONS), "n_limits": list(report.N_LIMITS), "default_n": 500,
                "min_days": report.MIN_DAYS, "max_assets": report.MAX_ASSETS, "metrics": [{"id": k, **v} for k, v in report.METRICS.items()], "levels": list(report.LEVELS)}

    def run(self, body: dict) -> dict:
        """``POST /api/scenarios/run`` (a job). ``body``: the ``tickers`` (the Backtest tab's if left out) and its data choices (``classes``, ``start``, ``source``, ``refresh``), the ``generators`` (ids), the number of
        scenarios ``n``, the ``horizon`` in trading days and ``seed``. Returns ``{"job": id}``."""
        app = self.app
        raw = body.get("tickers")
        if raw is None:
            raw = app.builder.default_tickers()
        if not isinstance(raw, list) or not raw:
            raise ApiError("choose at least one ticker")
        try:
            tickers = normalise(raw)
        except UniverseError as error:
            raise ApiError(str(error)) from None
        if len(tickers) > report.MAX_ASSETS:
            raise ApiError(f"scenarios use at most {report.MAX_ASSETS} tickers at once; this has {len(tickers)}. Take some out in the Backtest tab")
        names = body.get("generators", list(report.DEFAULT))
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise ApiError("generators must be a list of generator names")
        n = _integer(body.get("n", 500), "the number of scenarios", report.N_LIMITS[0], report.N_LIMITS[1])
        horizon = _integer(body.get("horizon", 21), "the horizon", 1, 126)
        if horizon not in report.HORIZONS:
            raise ApiError(f"the horizon must be one of {', '.join(map(str, report.HORIZONS))} trading days")
        seed = _integer(body.get("seed", 0), "the seed", 0, 1_000_000)
        unknown = [x for x in names if x not in {g["id"] for g in report.catalog()}]
        if unknown:
            raise ApiError(f"unknown generator {unknown[0]!r}; choose from {', '.join(g['id'] for g in report.catalog())}")
        if not names:
            raise ApiError("choose at least one generator")
        try:
            app.builder.preflight(tickers, body.get("start"), body.get("source"))
        except UniverseError as error:
            raise ApiError(str(error)) from None
        stop = threading.Event()

        def work(progress) -> dict:
            progress("getting the prices (downloading them if they are new)")
            try:
                bundle, universe = app.builder.resolve(tickers, body.get("classes"), body.get("start"), bool(body.get("refresh")), body.get("source"), progress)
            except UniverseError as error:
                raise ApiError(str(error)) from None
            prices = bundle.prices[[t for t in tickers if t in bundle.prices.columns]]
            returns = prices.pct_change().dropna()
            try:
                result = report.run(returns, names, n, horizon, seed, progress, stop.is_set)
            except report.ScenarioError as error:
                raise ApiError(str(error)) from None
            result["universe"] = universe
            result["stopped"] = bool(stop.is_set())
            return result

        for old in [j for j in self.stops if j not in app.jobs or app.jobs[j].status not in ACTIVE]:
            self.stops.pop(old, None)
        started = app.submit_task(f"scenarios: {len(tickers)} ticker{'s' if len(tickers) != 1 else ''}, {len(names)} generator{'s' if len(names) != 1 else ''}, {horizon} days", work, "scenarios")
        self.stops[started["job"]] = stop
        return started

    def stop(self, body: dict) -> dict:
        """``POST /api/scenarios/stop {"job": id}``: end a run after the generator it is on. The generators finished so far are kept and reported."""
        flag = self.stops.get(str(body.get("job") or ""))
        if flag is None:
            raise ApiError("that run is not going in this app")
        flag.set()
        return {"ok": True}
