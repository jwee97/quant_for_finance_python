"""The Alpha, Risk and Portfolio Lab's endpoints (``src/alphalab``, ``src/riskengine``, ``src/portfolio.lab``).

``GET  /api/lab/plan``          what each study does, the portfolio methods and their settings, the signals, the policies, and what the lab does not do
``POST /api/lab/alpha``         IC, decay, calibration, breadth and transfer of one signal over several tickers, as a job
``POST /api/lab/risk``          a factor model, the risk of a portfolio by factor, and the bias statistic, as a job
``POST /api/lab/portfolio``     the chosen optimisers on identical windows, costs and caps, as a job
``POST /api/lab/trading``       the calibrated alpha traded under four policies and the implementation waterfall, as a job
``GET  /api/lab/signals``       the versioned signals in the registry; ``POST /api/lab/signals/register`` and ``/api/lab/signals/delete``

A run is not saved to a research session yet. Signals here are built from prices; a signal from a model or from fundamentals is registered through ``src.alphalab.registry`` and studied with the same functions.
"""

from __future__ import annotations

import numpy as np

from ..alphalab import studies as st
from ..alphalab.registry import RegistryError, SignalRegistry
from ..portfolio import lab as pl
from .api import ApiError
from .universe import UniverseError, normalise

LIMITS = {"lookback": (int, 60, 1000), "step": (int, 1, 126), "cost_bps": (float, 0.0, 200.0), "mu_shrink": (float, 0.0, 1.0), "max_weight": (float, 0.02, 1.0), "risk_aversion": (float, 0.1, 100.0), "perturb": (int, 0, 12),
          "seed": (int, 0, 2 ** 31 - 1)}
PORTFOLIO_SETTINGS = [
    ("lookback", "Window (days)", "The trailing history every method sees at each rebalance."),
    ("step", "Rebalance every (days)", "Holding period between rebalances; weights drift with prices in between."),
    ("cost_bps", "Cost (bp per unit traded)", "Charged on every change of weights, including the first purchase from cash."),
    ("cov_method", "Covariance estimator", "The same estimate for every method. Shrinkage, EWMA, random-matrix denoising, graphical lasso (a sparse inverse), or a PCA factor model."),
    ("mu_shrink", "Shrink expected returns toward the average", "0 uses each asset's own sample mean; 1 gives every asset the same expected return. Only the methods that use expected returns see it."),
    ("max_weight", "Cap per asset", "Applied to every method after it chooses, by clipping and redistributing. The table shows how much weight each method needed moved."),
    ("risk_aversion", "Risk aversion", "For the mean-variance, near-optimal and robust methods."),
    ("perturb", "Resamples for weight stability", "How many bootstrap resamples of the window to re-optimise on (at up to six dates) to see how much the weights jump. 0 skips it; each costs time."),
]


def _number(raw: dict, spec: dict) -> dict:
    cfg = {}
    for key, value in (raw or {}).items():
        if key not in spec:
            raise ApiError(f"unknown setting {key!r}")
        kind, lo, hi = spec[key]
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ApiError(f"{key} must be a number") from None
        if v != v or not lo <= v <= hi or (kind is int and v != int(v)):
            raise ApiError(f"{key} must be {'a whole number ' if kind is int else 'a number '}between {lo} and {hi}")
        cfg[key] = kind(v)
    return cfg


def _tickers(body: dict, minimum: int, maximum: int = 40) -> list:
    try:
        tickers = normalise(body.get("tickers") or body.get("ticker") or [])
    except UniverseError as error:
        raise ApiError(str(error)) from None
    if len(tickers) < minimum:
        raise ApiError(f"choose at least {minimum} tickers")
    if len(tickers) > maximum:
        raise ApiError(f"choose at most {maximum} tickers")
    return tickers


class LabApi:
    def __init__(self, app):
        self.app = app
        self.registry = SignalRegistry(app.root / "data" / "user" / "alpha_signals")

    def get_routes(self) -> dict:
        return {"/api/lab/plan": self.plan, "/api/lab/signals": self.signals}

    def post_routes(self) -> dict:
        return {"/api/lab/alpha": self.alpha, "/api/lab/risk": self.risk, "/api/lab/portfolio": self.portfolio, "/api/lab/trading": self.trading, "/api/lab/signals/register": self.register,
                "/api/lab/signals/delete": self.delete}

    # ------------------------------------------------------------------------------------------------------ plan
    def plan(self, q: dict | None = None) -> dict:
        return {"signals": [{"key": k, "label": v[0], "help": v[1]} for k, v in st.SIGNALS.items()],
                "methods": [{"key": m.key, "label": m.label, "family": m.family, "uses_mu": m.uses_mu, "note": m.note} for m in pl.METHODS.values()], "defaults_methods": ["equal_weight", "min_variance", "risk_parity", "hrp", "nco", "max_sharpe"],
                "cov_methods": list(pl.COV_METHODS), "settings": [{"key": k, "label": label, "help": help_} for k, label, help_ in PORTFOLIO_SETTINGS],
                "defaults": {"lookback": 252, "step": 21, "cost_bps": 10, "cov_method": "shrinkage", "mu_shrink": 0.8, "max_weight": 1.0, "risk_aversion": 3.0, "perturb": 0},
                "limits": {k: [v[1], v[2]] for k, v in LIMITS.items()}, "risk_kinds": ["statistical", "time_series"], "policies": ["full", "partial", "band", "threshold"],
                "studies": [{"key": "alpha", "label": "Alpha", "what": "How good a signal is, how fast it fades, what it is worth as an expected return, how many independent bets it makes, and what a long-only book keeps of it."},
                            {"key": "risk", "label": "Risk", "what": "A factor model of the universe, a portfolio's risk by factor, and whether yesterday's risk forecast held."},
                            {"key": "portfolio", "label": "Portfolio", "what": "Optimisers compared on identical windows, costs and caps: return, risk, drawdown, turnover, concentration, and how much their weights jump."},
                            {"key": "trading", "label": "Trading", "what": "The same alpha traded under four policies, and where the paper alpha goes: limits, costs, delay."}],
                "notes": ["Signals are built from prices. A model's or a fundamentals signal goes through the same functions from the library (src.alphalab).",
                          "Differences between optimisers over a few years are usually inside the noise: the Sharpe standard error is in the table.",
                          "A method that fails on a date falls back to equal weight there and the count is shown.",
                          "Nothing here is saved to a research session yet."]}

    # ------------------------------------------------------------------------------------------------------ data
    def _load(self, tickers: list, body: dict, progress):
        start, source, refresh = body.get("start") or None, body.get("source"), bool(body.get("refresh"))
        try:
            self.app.builder.preflight(tickers, start, source)
            progress("loading the prices")
            bundle, universe = self.app.builder.resolve(tickers, None, start, refresh, source, progress)
        except UniverseError as error:
            raise ApiError(str(error)) from None
        px = bundle.prices[[t for t in tickers if t in bundle.prices.columns]].dropna(how="any")
        if px.shape[1] < len(tickers):
            raise ApiError("some tickers have no prices: " + ", ".join(sorted(set(tickers) - set(px.columns))))
        return px, universe

    def _check(self, tickers: list, body: dict) -> None:
        try:
            self.app.builder.preflight(tickers, body.get("start") or None, body.get("source"))
        except UniverseError as error:
            raise ApiError(str(error)) from None

    # ------------------------------------------------------------------------------------------------------ studies
    def alpha(self, body: dict) -> dict:
        tickers = _tickers(body, 6)
        signal, horizon = body.get("signal") or "momentum", int(body.get("horizon") or 5)
        if signal not in st.SIGNALS:
            raise ApiError(f"signal must be one of {', '.join(st.SIGNALS)}")
        if not 1 <= horizon <= 63:
            raise ApiError("horizon must be 1 to 63 days")
        self._check(tickers, body)

        def work(progress):
            px, universe = self._load(tickers, body, progress)
            progress("information coefficients, decay and calibration")
            try:
                out = st.alpha_study(px, signal, horizon)
            except st.StudyError as error:
                raise ApiError(str(error)) from None
            out["universe"] = {"source": universe.get("source"), "tickers": tickers}
            return out

        return self.app.submit_task(f"Alpha study: {signal}", work, kind="lab")

    def risk(self, body: dict) -> dict:
        tickers = _tickers(body, 4)
        kind = body.get("kind") or "statistical"
        if kind not in ("statistical", "time_series"):
            raise ApiError("kind must be statistical or time_series")
        n = body.get("n_factors")
        if n not in (None, ""):
            try:
                n = int(n)
            except (TypeError, ValueError):
                raise ApiError("n_factors must be a whole number") from None
            if not 1 <= n <= min(10, len(tickers) - 1):
                raise ApiError(f"n_factors must be 1 to {min(10, len(tickers) - 1)}")
        else:
            n = None
        self._check(tickers, body)

        def work(progress):
            px, universe = self._load(tickers, body, progress)
            progress("fitting the factor model")
            try:
                out = st.risk_study(px.pct_change().iloc[1:], kind, n)
            except (st.StudyError, ValueError) as error:
                raise ApiError(str(error)) from None
            out["universe"] = {"source": universe.get("source"), "tickers": tickers}
            return out

        return self.app.submit_task("Risk model", work, kind="lab")

    def portfolio(self, body: dict) -> dict:
        tickers = _tickers(body, 4)
        raw = dict(body.get("config") or {})
        cov_method = raw.pop("cov_method", "shrinkage")
        if cov_method not in pl.COV_METHODS:
            raise ApiError(f"cov_method must be one of {', '.join(pl.COV_METHODS)}")
        cfg = _number(raw, LIMITS)
        methods = body.get("methods")
        if methods is None:
            methods = ["equal_weight", "min_variance", "risk_parity", "hrp", "nco", "max_sharpe"]
        if not isinstance(methods, list) or not methods:
            raise ApiError("choose at least one method")
        bad = [m for m in methods if m not in pl.METHODS]
        if bad:
            raise ApiError(f"unknown method(s): {', '.join(map(str, bad))}")
        if len(methods) > len(pl.METHODS):
            raise ApiError("too many methods")
        if cfg.get("max_weight", 1.0) * len(tickers) < 1.0 - 1e-12:
            raise ApiError(f"a cap of {cfg['max_weight']} on {len(tickers)} tickers cannot be fully invested")
        self._check(tickers, body)

        def work(progress):
            px, universe = self._load(tickers, body, progress)
            try:
                res = pl.compare(px.pct_change().iloc[1:], methods, cov_method=cov_method, progress=lambda text, frac=None: progress(f"{text}"), **cfg)
            except pl.LabError as error:
                raise ApiError(str(error)) from None
            eq = res["equity"]
            table = res["table"].copy()
            rows = []
            for key, r in table.iterrows():
                d = {"key": key}
                for c, v in r.items():
                    d[c] = v if isinstance(v, str) else (None if not np.isfinite(v) else float(v))
                d["note"] = res["notes"].get(key, "")
                rows.append(d)
            last = {k: {a: float(x) for a, x in w.iloc[-1].items()} for k, w in res["weights"].items()}
            return {"table": rows, "equity": {"dates": [str(d.date()) for d in eq.index], "series": {k: [round(float(x), 6) for x in eq[k].to_numpy()] for k in eq.columns}}, "last_weights": last,
                    "mean_weights": {k: {a: float(x) for a, x in w.mean().items()} for k, w in res["weights"].items()}, "rebalances": res["rebalances"], "config": res["config"], "universe": {"source": universe.get("source"), "tickers": tickers},
                    "notes": ["Every method saw the same windows, covariance estimate, caps and costs.",
                              "Sharpe standard errors are large: a difference smaller than about two of them is not evidence that one method is better.",
                              "'Fallbacks' counts the dates a method failed and equal weight was used instead; 'capped' is the average weight moved by the cap."]}

        return self.app.submit_task("Portfolio lab", work, kind="lab")

    def trading(self, body: dict) -> dict:
        tickers = _tickers(body, 6, 30)
        signal, horizon = body.get("signal") or "momentum", int(body.get("horizon") or 5)
        if signal not in st.SIGNALS:
            raise ApiError(f"signal must be one of {', '.join(st.SIGNALS)}")
        if not 1 <= horizon <= 63:
            raise ApiError("horizon must be 1 to 63 days")
        cfg = _number(body.get("config") or {}, {"cost_bps": (float, 0.0, 200.0), "target_gross": (float, 0.1, 5.0), "lag": (int, 0, 5)})
        self._check(tickers, body)

        def work(progress):
            px, universe = self._load(tickers, body, progress)
            progress("calibrating the alpha and trading it under four policies (about half a minute)")
            try:
                out = st.trading_study(px, signal, horizon, long_only=bool(body.get("long_only", True)), **cfg)
            except st.StudyError as error:
                raise ApiError(str(error)) from None
            out["universe"] = {"source": universe.get("source"), "tickers": tickers}
            return out

        return self.app.submit_task(f"Trading study: {signal}", work, kind="lab")

    # ------------------------------------------------------------------------------------------------------ registry
    def signals(self, q: dict | None = None) -> dict:
        return {"signals": self.registry.listing()}

    def register(self, body: dict) -> dict:
        tickers = _tickers(body, 6)
        signal = body.get("signal") or "momentum"
        if signal not in st.SIGNALS:
            raise ApiError(f"signal must be one of {', '.join(st.SIGNALS)}")
        name = str(body.get("name") or signal).strip()
        self._check(tickers, body)

        def work(progress):
            px, _ = self._load(tickers, body, progress)
            panel = st.build_signal(px, signal).dropna(how="all")
            try:
                return self.registry.register(name, panel, lag=0, horizon=int(body.get("horizon") or 5), description=st.SIGNALS[signal][0], source="prices")
            except RegistryError as error:
                raise ApiError(str(error)) from None

        return self.app.submit_task(f"Register signal {name}", work, kind="lab")

    def delete(self, body: dict) -> dict:
        try:
            self.registry.delete(str(body.get("name") or ""))
        except RegistryError as error:
            raise ApiError(str(error)) from None
        return {"deleted": body.get("name")}
