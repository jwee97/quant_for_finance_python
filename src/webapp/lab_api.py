"""The Alpha, Risk and Portfolio Lab's endpoints (``src/alphalab``, ``src/riskengine``, ``src/portfolio.lab``).

``GET  /api/lab/plan``          what each study does, the portfolio methods and their settings, the signals, the policies, and what the lab does not do
``POST /api/lab/alpha``         IC, decay, calibration, breadth and transfer of one signal over several tickers, as a job
``POST /api/lab/risk``          a factor model, the risk of a portfolio by factor, and the bias statistic, as a job
``POST /api/lab/portfolio``     the chosen optimisers on identical windows, costs and caps, as a job
``POST /api/lab/trading``       the calibrated alpha traded under four policies and the implementation waterfall, as a job
``GET  /api/lab/signals``       the versioned signals in the registry; ``POST /api/lab/signals/register`` and ``/api/lab/signals/delete``

Saving to a research session: a factor study (``/api/lab/factors``), a portfolio comparison (``/api/lab/portfolio``, one method at a time) and a trading study (``/api/lab/trading``) are kept with
``POST /api/research/artifact/save`` (the Alpha research workflow and the Portfolio Lab have a Save to session button), then versioned, run again and compared in the Research comparison. The single-signal alpha study
(``/api/lab/alpha``) and the risk study (``/api/lab/risk``) have no saved form yet. Signals here are built from prices; a signal from a model or from fundamentals is registered through ``src.alphalab.registry`` and studied with the same functions.
"""

from __future__ import annotations

import numpy as np

from ..alphalab.fundamental import FundamentalError, panels as fundamental_panels
from ..alphalab import studies as st
from ..alphalab.registry import RegistryError, SignalRegistry
from ..alphalab import expected as ex
from ..forecast import alpha as fa
from ..portfolio import attribution as pat
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


def _f(v):
    v = float(v)
    return v if np.isfinite(v) else None


class LabApi:
    def __init__(self, app):
        self.app = app
        self.registry = SignalRegistry(app.root / "data" / "user" / "alpha_signals")

    def get_routes(self) -> dict:
        return {"/api/lab/plan": self.plan, "/api/lab/signals": self.signals}

    def post_routes(self) -> dict:
        return {"/api/lab/alpha": self.alpha, "/api/lab/risk": self.risk, "/api/lab/portfolio": self.portfolio, "/api/lab/trading": self.trading, "/api/lab/factors": self.factors, "/api/lab/signals/register": self.register,
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
                          "A factor study, a portfolio comparison (one method at a time) and a trading study can be saved to a research session and compared there; the single-signal alpha study and the risk study cannot yet."]}

    # ------------------------------------------------------------------------------------------------------ data
    def _load(self, tickers: list, body: dict, progress):
        start, source, refresh = body.get("start") or None, body.get("source"), bool(body.get("refresh"))
        try:
            self.app.builder.preflight(tickers, start, source)
            progress("loading the prices")
            bundle, universe = self.app.builder.resolve(tickers, None, start, refresh, source, progress)
        except UniverseError as error:
            raise ApiError(str(error)) from None
        universe = {**universe, "groups": {str(k): str(v) for k, v in dict(getattr(bundle, "group", None) or getattr(bundle, "asset_class", None) or {}).items() if k in tickers}}
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

    def factors(self, body: dict) -> dict:
        """``POST /api/lab/factors {tickers, signals, horizon, fundamental_factors?, fundamentals_db?}`` (a job): evaluate the chosen factors, correct for the number compared and combine them from the past only.
        ``fundamental_factors`` are derived metrics of the Fundamentals tab (``roe``, ``gross_margin`` ...) read as of each date from the point-in-time tables; they are admitted only after the point-in-time
        audit of those companies is not ``blocked`` (the audit and the coverage come back with the result)."""
        tickers = _tickers(body, 6)
        horizon = int(body.get("horizon") or 5)
        fundamental = body.get("fundamental_factors") or []
        if not isinstance(fundamental, list) or not all(isinstance(s, str) and s for s in fundamental) or len(set(fundamental)) != len(fundamental):
            raise ApiError("fundamental_factors must be a list of distinct metric names from the Fundamentals tab")
        signals = body.get("signals")
        signals = list(st.SIGNALS) if signals is None and not fundamental else (signals or [])
        if not isinstance(signals, list) or not all(isinstance(s, str) and s in st.SIGNALS for s in signals) or len(set(signals)) != len(signals):
            raise ApiError(f"signals must be distinct names from: {', '.join(st.SIGNALS)}")
        if not 1 <= horizon <= 63:
            raise ApiError("horizon must be 1 to 63 days")
        db = str(body.get("fundamentals_db") or "real")
        if db not in ("real", "demo"):
            raise ApiError("fundamentals_db must be real or demo")
        self._check(tickers, body)

        def work(progress):
            px, universe = self._load(tickers, body, progress)
            extra, info = {}, None
            if fundamental:
                progress("checking when the fundamentals could have been known (point in time)")
                try:
                    store = self.app.fundamentals.bench(db, create=db == "real").store
                    got = fundamental_panels(store, tickers, fundamental, px.index)
                except FundamentalError as error:
                    raise ApiError(str(error)) from None
                extra = got["panels"]
                info = {"database": db, "basis": got["basis"], "pit": got["pit"], "coverage": got["coverage"], "missing_tickers": got["missing_tickers"], "settings": got["settings"]}
            progress("evaluating, correcting for multiple testing and combining the factors")
            try:
                out = st.factor_study(px, signals, horizon, extra=extra, extra_info=info)
            except st.StudyError as error:
                raise ApiError(str(error)) from None
            out["universe"] = {"source": universe.get("source"), "tickers": tickers}
            return out

        return self.app.submit_task("Factor study", work, kind="lab")

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
        step = int(cfg.get("step", 21))
        specs = self._mu_specs(body, step)
        attr_cfg = body.get("attribution", {"kind": "market"})
        if attr_cfg not in (None, False) and (not isinstance(attr_cfg, dict) or attr_cfg.get("kind", "market") not in ("market", "statistical")):
            raise ApiError("attribution.kind must be market or statistical")
        sectors = body.get("sectors")
        if sectors is not None and not (isinstance(sectors, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in sectors.items())):
            raise ApiError("sectors must map each ticker to a group name")
        sector_source = str(body.get("sector_source") or "auto")
        if sector_source not in ("auto", "fundamentals", "none"):
            raise ApiError("sector_source must be auto, fundamentals or none")
        self._check(tickers, body)

        def work(progress):
            px, universe = self._load(tickers, body, progress)
            data = px.pct_change().iloc[1:]
            runs = {}
            for i, spec in enumerate(specs):
                mu_fn, label, meta = self._mu_provider(spec, px, data, cfg, step, progress)
                try:
                    runs[label] = (pl.compare(data, methods, cov_method=cov_method, progress=lambda text, frac=None: progress(f"{text}"), mu_fn=mu_fn, mu_label=label, horizon=int(spec.get("horizon") or step), **cfg), meta)
                except pl.LabError as error:
                    raise ApiError(str(error)) from None
            primary = next(iter(runs))
            res, mu_meta = runs[primary]
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
            try:
                diag = pl.diagnostics(res, px.pct_change().iloc[1:])
            except Exception as error:                                                                   # the comparison stands without the extra view; say that it is missing and why
                diag = {"error": f"{type(error).__name__}: {error}"}
            attribution = None
            if attr_cfg not in (None, False):
                progress("attributing the returns to factors and, where there is a classification, to groups (Brinson-Fachler)")
                groups, why = self._sectors(tickers, sectors, sector_source, universe)
                attribution = pat.compact(pat.attribute_lab_result(res, data, kind=attr_cfg.get("kind", "market"), n_factors=int(attr_cfg.get("n_factors", 2)), sectors=groups))
                if groups is None and attribution.get("brinson_reason") is None:
                    attribution["brinson_reason"] = why
                elif groups is None:
                    attribution["brinson_reason"] = why or attribution["brinson_reason"]
            comparison = None
            if len(runs) > 1:
                comparison = {lbl: {"table": [{"key": k, "sharpe": _f(r["table"].loc[k, "sharpe"]), "ann_return": _f(r["table"].loc[k, "ann_return"]), "ann_vol": _f(r["table"].loc[k, "ann_vol"]), "turnover_per_year": _f(r["table"].loc[k, "turnover_per_year"]),
                                              "max_drawdown": _f(r["table"].loc[k, "max_drawdown"])} for k in r["table"].index], "equity": {k: [round(float(x), 6) for x in r["equity"][k].to_numpy()] for k in r["equity"].columns},
                                    "mu_report": {"filled_with_historical": [v["filled_with_historical"] for v in r["mu_report"].values()], "assets_with_estimate": [v["assets_with_estimate"] for v in r["mu_report"].values()]}, "meta": m}
                              for lbl, (r, m) in runs.items()}
            expected = {"source": primary, "meta": mu_meta, "report": {d: {k: v for k, v in rep.items() if k != "info"} for d, rep in res["mu_report"].items()},
                        "last_info": (list(res["mu_report"].values())[-1]["info"] if res["mu_report"] else {}), "horizon_days": int(specs[0].get("horizon") or step)}
            return {"table": rows, "equity": {"dates": [str(d.date()) for d in eq.index], "series": {k: [round(float(x), 6) for x in eq[k].to_numpy()] for k in eq.columns}}, "last_weights": last,
                    "mean_weights": {k: {a: float(x) for a, x in w.mean().items()} for k, w in res["weights"].items()}, "diagnostics": diag, "expected_returns": expected, "attribution": attribution, "source_comparison": comparison, "rebalances": res["rebalances"], "config": res["config"], "universe": {"source": universe.get("source"), "tickers": tickers},
                    "notes": ["Every method saw the same windows, covariance estimate, caps and costs.",
                              "Sharpe standard errors are large: a difference smaller than about two of them is not evidence that one method is better.",
                              "'Fallbacks' counts the dates a method failed and equal weight was used instead; 'capped' is the average weight moved by the cap."]}

        return self.app.submit_task("Portfolio lab", work, kind="lab")

    def _mu_specs(self, body: dict, step: int) -> list[dict]:
        """The sources of expected returns the request asks for (the first is the one the page draws in full; any others are run on identical settings and compared)."""
        raw = body.get("expected_returns")
        specs = [{"source": "historical"}] if raw in (None, "", {}) else raw if isinstance(raw, list) else [raw]
        if not 1 <= len(specs) <= 4 or not all(isinstance(s, dict) for s in specs):
            raise ApiError("expected_returns is a source, or a list of up to four")
        seen = set()
        for s in specs:
            src = s.get("source", "historical")
            if src not in ("historical", "factor", "forecast", "combined"):
                raise ApiError("an expected-return source is historical, factor, forecast or combined")
            if src in ("factor", "combined") and s.get("factor", "momentum") not in st.SIGNALS:
                raise ApiError(f"factor must be one of: {', '.join(st.SIGNALS)}")
            if src in ("forecast", "combined") and s.get("forecast_method", "historical") not in fa.MEAN_METHODS:
                raise ApiError(f"forecast_method must be one of: {', '.join(fa.MEAN_METHODS)}")
            if s.get("calibrate", "past_slope") not in ("past_slope", "none"):
                raise ApiError("calibrate is past_slope or none")
            if s.get("combine", "equal") not in ("equal", "inverse_error"):
                raise ApiError("combine is equal or inverse_error")
            if src == "historical":                                                           # the window's own mean has no forecast horizon to check
                label = self._mu_label(s)
                if label in seen:
                    raise ApiError("the same source of expected returns is listed twice")
                seen.add(label)
                continue
            try:
                h = int(s.get("horizon") or step)
                age = int(s.get("max_age_days", 10))
            except (TypeError, ValueError):
                raise ApiError("horizon and max_age_days are whole numbers") from None
            if not 1 <= h <= 63 or not 1 <= age <= 120:
                raise ApiError("horizon is 1 to 63 days and max_age_days 1 to 120" + (f" (the rebalance step is {step} days: a forecast source needs a step of at most 63)" if step > 63 else ""))
            if h != step and not s.get("allow_horizon_scaling"):
                raise ApiError(f"the forecast horizon is {h} days and the portfolio is rebalanced every {step}: a forecast is used only at its own horizon (set the horizon to the rebalance step, or allow scaling, a constant-drift assumption)")
            label = self._mu_label(s)
            if label in seen:
                raise ApiError("the same source of expected returns is listed twice")
            seen.add(label)
        return specs

    @staticmethod
    def _mu_label(s: dict) -> str:
        src = s.get("source", "historical")
        return {"historical": "historical mean", "factor": f"factor: {s.get('factor', 'momentum')}", "forecast": f"forecast: {s.get('forecast_method', 'historical')}",
                "combined": f"combined: {s.get('factor', 'momentum')} + {s.get('forecast_method', 'historical')}"}[src]

    def _mu_provider(self, s: dict, px, data, cfg: dict, step: int, progress):
        src, h = s.get("source", "historical"), int(s.get("horizon") or step)
        label = self._mu_label(s)
        if src == "historical":
            return None, label, {"source": src, "note": "the mean return of the window, the Lab's default"}
        scaling = bool(s.get("allow_horizon_scaling"))
        lookback = int(cfg.get("lookback", 252))
        parts, meta = {}, {"source": src, "horizon_days": h, "allow_horizon_scaling": scaling}
        if src in ("factor", "combined"):
            parts["factor"] = ex.factor_provider(px, s.get("factor", "momentum"), h)
            meta["factor"] = s.get("factor", "momentum")
        if src in ("forecast", "combined"):
            method = s.get("forecast_method", "historical")
            origins = [data.index[p - 1] for p in range(lookback, len(data) - 1, step)]
            progress(f"forecasting every asset at {len(origins)} rebalance dates with {method}")
            records = fa.generate(px, method, origins, h, n_paths=int(s.get("n_paths", 300)), seed=int(s.get("seed", 0)))
            if not records:
                raise ApiError(f"{method} needs more history than the prices hold: no forecast could be made")
            parts["forecast"] = ex.forecast_provider(records, px, h, int(s.get("max_age_days", 10)), s.get("calibrate", "past_slope"), scaling)
            meta.update({"forecast_method": method, "forecasts": len(records), "calibrate": s.get("calibrate", "past_slope"), "max_age_days": int(s.get("max_age_days", 10)),
                         "training_fingerprint_example": records[-1].training_fingerprint, "model_version": records[-1].model_version, "forecast_contract": records[-1].to_dict()})
        fn = parts["factor"] if src == "factor" else parts["forecast"] if src == "forecast" else ex.combined_provider(parts, s.get("combine", "equal"), step, h)
        return fn, label, meta

    def _sectors(self, tickers: list, given, source: str, universe: dict):
        if source == "none":
            return None, "classification switched off"
        if given:
            return {k: v for k, v in given.items()}, None
        if source == "fundamentals":
            try:
                sec = self.app.fundamentals.bench("real", create=False).store.securities()
            except Exception:
                return None, "the fundamentals database could not be read"
            m = {str(r.ticker).upper(): str(r.sector) for r in sec.itertuples() if getattr(r, "sector", None)}
            groups = {t: m[t.upper()] for t in tickers if t.upper() in m}
            return (groups, None) if groups else (None, "the fundamentals database holds no sector for these tickers")
        groups = (universe or {}).get("groups") or {}
        groups = {t: g for t, g in groups.items() if g and g != "unknown"}
        if groups:
            return groups, None
        return None, "the data carry no classification of these tickers (give sectors, or take them from the Fundamentals tab)"

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
