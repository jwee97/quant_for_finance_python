"""The Stock Universe Explorer's endpoint: ``POST /api/universe/explore`` (a job) and ``GET /api/universe/plan``.

The request names tickers (and optionally a membership table of dated listings), the screens, construction methods and lookbacks to search, and the validation design. The job loads prices (and volume, where
the data has it), runs the nested chronological search and returns the whole study. Nothing here chooses a configuration for the person: the grid is theirs, and its size is reported with every result.
"""

from __future__ import annotations

from ..portfolio import lab as pl
from ..universelab import discover as dc, membership as mb, screens as sc, study
from .api import ApiError
from .lab_api import _tickers
from .universe import UniverseError

LIMITS = {"folds": (2, 8), "min_train_days": (252, 2520), "embargo": (0, 60), "step": (5, 63), "cost_bps": (0.0, 200.0), "max_weight": (0.05, 1.0), "bootstrap": (0, 1000)}
DEFAULTS = {"folds": 4, "min_train_days": 504, "embargo": 5, "step": 21, "cost_bps": 10.0, "max_weight": 1.0, "bootstrap": 300}
DEFAULT_METHODS = ["equal_weight", "min_variance", "risk_parity", "hrp"]


class UniverseApi:
    def __init__(self, app):
        self.app = app

    def get_routes(self) -> dict:
        return {"/api/universe/plan": self.plan}

    def post_routes(self) -> dict:
        return {"/api/universe/explore": self.explore}

    def plan(self, q: dict | None = None) -> dict:
        return {"screens": [{"key": k, "label": sc.LABELS[k]} for k in sc.KINDS], "methods": [{"key": m.key, "label": m.label, "family": m.family} for m in pl.METHODS.values()], "defaults": DEFAULTS,
                "default_methods": DEFAULT_METHODS, "lookbacks": [126, 252, 504], "max_candidates": dc.MAX_CANDIDATES, "limits": {k: list(v) for k, v in LIMITS.items()},
                "membership_columns": list(mb.COLUMNS),
                "notes": ["Without a membership table the universe is today's list of names and the result carries the survivorship warning.",
                          "The screen, the construction method and the lookback are chosen together on past data only, then judged on data they never saw.",
                          "Cap-weight is not available (shares outstanding are not in the data); where volume exists a dollar-volume-weighted benchmark is shown and labelled a proxy."]}

    def _settings(self, body: dict) -> dict:
        cfg = dict(DEFAULTS)
        for k, v in (body.get("design") or {}).items():
            if k not in LIMITS:
                raise ApiError(f"design has no setting {k!r}; the settings are {sorted(LIMITS)}")
            lo, hi = LIMITS[k]
            try:
                x = float(v)
            except (TypeError, ValueError):
                raise ApiError(f"{k} must be a number") from None
            if not lo <= x <= hi:
                raise ApiError(f"{k} must be between {lo} and {hi}")
            cfg[k] = int(x) if isinstance(DEFAULTS[k], int) else x
        return cfg

    def explore(self, body: dict) -> dict:
        tickers = _tickers(body, 6)
        cfg = self._settings(body)
        try:
            screens = [sc.validate(s) for s in (body.get("screens") or [{"kind": "all"}])]
        except (sc.ScreenError, TypeError, ValueError) as exc:
            raise ApiError(str(exc)) from None
        methods = body.get("methods") or DEFAULT_METHODS
        bad = [m for m in methods if m not in pl.METHODS]
        if not isinstance(methods, list) or bad:
            raise ApiError(f"unknown method(s): {', '.join(map(str, bad))}" if bad else "methods must be a list")
        lookbacks = body.get("lookbacks") or [252]
        if not isinstance(lookbacks, list) or any(not isinstance(x, (int, float)) or not 60 <= x <= 1000 for x in lookbacks):
            raise ApiError("lookbacks are whole numbers of days between 60 and 1000")
        lookbacks = [int(x) for x in lookbacks]
        try:
            dc.candidates(screens, methods, lookbacks)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        member = None
        if body.get("membership"):
            try:
                member = mb.from_csv(str(body["membership"]), "complete" if body.get("membership_complete") else "partial")
            except mb.MembershipError as exc:
                raise ApiError(str(exc)) from None
        try:
            self.app.builder.preflight(tickers, body.get("start") or None, body.get("source"))
        except UniverseError as error:
            raise ApiError(str(error)) from None

        def work(progress):
            try:
                progress("loading the prices")
                bundle, universe = self.app.builder.resolve(tickers, None, body.get("start") or None, bool(body.get("refresh")), body.get("source"), progress)
            except UniverseError as error:
                raise ApiError(str(error)) from None
            cols = [t for t in tickers if t in bundle.prices.columns]
            if len(cols) < 6:
                raise ApiError("fewer than 6 of the tickers have prices")
            groups = {c: (bundle.group or {}).get(c) or (bundle.asset_class or {}).get(c) or "other" for c in cols}
            try:
                out = study.run_study(bundle.prices[cols], screens, methods, lookbacks, volume=None if getattr(bundle, "volume", None) is None else bundle.volume[cols], membership=member,
                                      groups=groups, progress=progress, **cfg)
            except (ValueError, sc.ScreenError) as exc:
                raise ApiError(str(exc)) from None
            out["universe"] = {"source": universe.get("source"), "tickers": cols}
            return out

        return self.app.submit_task("Stock universe study", work, kind="lab")
