"""The Fundamentals workbench: the operations the page and the command line share, over one service and one database.

Everything the Fundamentals tab shows and every ``quant fundamentals`` subcommand does is a method here, so the two cannot drift apart and neither needs the web server to be tested. A method takes
plain values (dictionaries from a form or a file), checks them, does the work on the local database and returns plain JSON-safe values; it never reads a provider (only :meth:`FundamentalsService.sync`
and the price loaders do, and only when asked).

A research context (the slow part: it builds the point-in-time tables of every company) is kept for the last few universes asked for and thrown away when the database changes, so moving between
the Factor Lab, a combination and a backtest on one universe builds it once.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import asdict
from dataclasses import fields as dataclass_fields

import numpy as np
import pandas as pd
from scipy import stats

from . import analytics, bridge, calibration, combine, metrics, prices, transforms
from . import expr as X
from . import factors as factor_library
from .discovery import DEFAULT_STRICTNESS, FITNESS, PRESETS, STAGES, STRICTNESS, Discovery, DiscoverySpec, Thresholds, default_seeds, splits
from .errors import FundamentalsError
from .evaluate import CONTROLS, PERIODS_PER_YEAR, TYPICAL_IC, EvalSpec, detectable_effect, evaluate, evaluate_factor, ic_stats, look_ahead_audit, rank_ic
from .jsonutil import clean, series_points
from .library import MONITOR_STATES, STATUSES, TRANSITIONS, AlphaLibrary
from .library import build_signal as build_alpha_signal
from .pit import PeriodTables, PitSettings
from .providers import PROVIDER_NAMES
from .research import FREQ_DAYS, GROUPS, HORIZONS, ResearchContext, UniverseSpec, horizon_days
from .service import FundamentalsService

KEEP_CONTEXTS = 2
MAX_SYMBOLS = 200
MAX_SCAN = 120
DIRECTIONS = ("theory", "auto", 1, -1)
POLICY = ("OpenFilings' usage policy does not allow rotating accounts or API keys to get past its quotas, and several keys of one account share one allowance. The first working key is used; a later "
          "key takes over only if the first is rejected as invalid. When the allowance is spent the work stops until it resets. More credits come from a larger plan or from prepaid top-ups.")
KEY_NAMES = ["OPENFILINGS_API_KEY"] + [f"OPENFILINGS_API_KEY_{i}" for i in range(1, 6)]


# ------------------------------------------------------------------------------------------------------------------ checking what a form sends
def _strict(cls, raw: dict | None, label: str) -> dict:
    raw = dict(raw or {})
    if not isinstance(raw, dict):
        raise FundamentalsError(f"{label} must be a dictionary")
    known = {f.name for f in dataclass_fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise FundamentalsError(f"unknown {label} setting(s) {unknown}; the settings are {sorted(known)}")
    return raw


def _number(raw, label: str, low: float, high: float, integer: bool = False):
    try:
        x = float(raw)
    except (TypeError, ValueError):
        raise FundamentalsError(f"{label} must be a number") from None
    if not np.isfinite(x) or not low <= x <= high:
        raise FundamentalsError(f"{label} must be between {low:g} and {high:g}")
    return int(round(x)) if integer else x


def _date(raw, label: str) -> str | None:
    if raw in (None, ""):
        return None
    try:
        return str(pd.Timestamp(str(raw)).date())
    except (ValueError, TypeError):
        raise FundamentalsError(f"{label} must be a date such as 2015-01-31") from None


def universe_spec(raw: dict | None) -> UniverseSpec:
    raw = _strict(UniverseSpec, raw, "universe")
    out = {}
    for key in ("countries", "regions", "sectors", "markets"):
        if key in raw:
            value = raw[key]
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value) or len(value) > 100:
                raise FundamentalsError(f"universe {key} must be a list of names")
            out[key] = tuple(value)
    for key in ("min_market_cap_usd", "min_dollar_volume_usd"):
        if key in raw:
            out[key] = _number(raw[key], key.replace("_", " "), 0, 1e15)
    for key in ("developed_only", "include_inactive"):
        if key in raw:
            out[key] = bool(raw[key])
    if "name" in raw:
        out["name"] = str(raw["name"])[:60]
    return UniverseSpec(**out)


def pit_settings(raw: dict | None) -> PitSettings:
    raw = dict(raw or {})
    unknown = sorted(set(raw) - {"info_lag_days", "use_estimated", "naive_availability"})
    if unknown:
        raise FundamentalsError(f"unknown point-in-time setting(s) {unknown}; the settings are info_lag_days, use_estimated and naive_availability")
    out = {}
    if "info_lag_days" in raw:
        out["info_lag_days"] = _number(raw["info_lag_days"], "the information lag", 0, 60, True)
    for key in ("use_estimated", "naive_availability"):
        if key in raw:
            out[key] = bool(raw[key])
    return PitSettings(**out)


def eval_spec(raw: dict | None) -> EvalSpec:
    raw = _strict(EvalSpec, raw, "evaluation")
    spec = EvalSpec.from_dict(raw)
    if spec.direction not in DIRECTIONS:
        raise FundamentalsError("direction must be 'theory', 'auto', 1 or -1")
    if spec.horizon not in HORIZONS and not (isinstance(spec.horizon, (int, float)) and 1 <= spec.horizon <= 504):
        raise FundamentalsError(f"horizon must be one of {sorted(HORIZONS, key=HORIZONS.get)} or a number of trading days")
    spec.quantiles = _number(spec.quantiles, "quantiles", 2, 10, True)
    if spec.top_fraction is not None:
        spec.top_fraction = _number(spec.top_fraction, "the top fraction", 0.02, 0.5)
    if spec.weighting not in ("equal", "signal", "inv_vol"):
        raise FundamentalsError("weighting must be equal, signal or inv_vol")
    spec.cost_bps, spec.borrow_bps = _number(spec.cost_bps, "cost", 0, 500), _number(spec.borrow_bps, "borrow cost", 0, 2000)
    spec.min_assets = _number(spec.min_assets, "the minimum number of companies", 5, 2000, True)
    spec.start, spec.end = _date(spec.start, "start"), _date(spec.end, "end")
    if not isinstance(spec.steps, list) or len(spec.steps) > 12 or not all(isinstance(s, dict) and s.get("op") in transforms.OPS for s in spec.steps):
        raise FundamentalsError(f"the steps are a list of at most 12 operations, each one of {list(transforms.OPS)}")
    bad = [b for b in spec.breakdowns if b not in ("country", "sector", "region", "regime")]
    if bad:
        raise FundamentalsError("breakdowns are country, sector, region and regime")
    spec.premium = bool(spec.premium)
    if not isinstance(spec.controls, (list, tuple)) or len(spec.controls) > 8 or not all(isinstance(c, str) for c in spec.controls):
        raise FundamentalsError("controls are a list of at most 8 names")
    allowed = set(CONTROLS) | set(factor_library.names())
    unknown = [c for c in spec.controls if c not in allowed]
    if unknown:
        raise FundamentalsError(f"unknown control(s) {unknown}: size, beta, volatility, momentum or the name of a library factor")
    spec.controls = tuple(dict.fromkeys(spec.controls))
    return spec


class Workbench:
    def __init__(self, root=".", service: FundamentalsService | None = None):
        self.service = service or FundamentalsService(root)
        self.store = self.service.store
        self.library = AlphaLibrary(self.store)
        self._contexts: OrderedDict = OrderedDict()
        self._size: tuple | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------------------------------------------------ contexts
    def _signature(self) -> tuple:
        c = self.store.counts()
        return (c["companies"], c["observations"], c["price_rows"], c["annual_periods"], c["quarterly_periods"], c["restated_vintages"])

    def _context_key(self, universe, freq, entry_lag, start, end, pit) -> tuple:
        if freq not in FREQ_DAYS:
            raise FundamentalsError("freq must be W (weekly), M (monthly) or Q (quarterly)")
        spec, settings = universe_spec(universe), pit_settings(pit)
        lag = _number(entry_lag, "the entry lag", 0, 10, True)
        return (spec.key(), freq, lag, _date(start, "start"), _date(end, "end"), repr(settings), self._signature()), spec, settings

    def context(self, universe: dict | None = None, freq: str = "M", entry_lag: int = 1, start: str | None = None, end: str | None = None, pit: dict | None = None) -> ResearchContext:
        """The research context for a universe, from a cache that forgets the least recently used and everything when the database changes."""
        key, spec, settings = self._context_key(universe, freq, entry_lag, start, end, pit)
        with self._lock:
            hit = self._contexts.get(key)
            if hit is not None:
                self._contexts.move_to_end(key)
                return hit
            ctx = ResearchContext(self.store, spec, start=key[3], end=key[4], freq=freq, entry_lag=key[2], settings=settings)
            self._contexts[key] = ctx
            while len(self._contexts) > KEEP_CONTEXTS:
                self._contexts.popitem(last=False)
            return ctx

    def peek(self, universe: dict | None = None, freq: str = "M", entry_lag: int = 1, start: str | None = None, end: str | None = None, pit: dict | None = None) -> ResearchContext | None:
        """The context for these choices if it has already been built, else ``None`` (never builds one: a quick check can use the data when it is at hand and do without when it is not)."""
        key = self._context_key(universe, freq, entry_lag, start, end, pit)[0]
        with self._lock:
            return self._contexts.get(key)

    def forget(self) -> None:
        with self._lock:
            self._contexts.clear()

    # ------------------------------------------------------------------------------------------------------------ what exists and what is missing
    def readiness(self) -> dict:
        c = self.store.counts()
        problems = []
        if c["companies"] == 0:
            problems.append("There are no companies yet: add some in the Data Manager, or load the simulated world to try the tab.")
        elif c["with_financials"] == 0:
            problems.append("No company has financial statements yet: fetch them in the Data Manager (a plan shows what it will cost first).")
        if c["companies"] and c["price_rows"] == 0:
            problems.append("No company has prices yet: load them in the Data Manager. A factor study needs the price history to measure what a factor would have earned.")
        return {"ready": not problems, "problems": problems, "companies": c["companies"], "with_financials": c["with_financials"], "price_rows": c["price_rows"]}

    def status(self) -> dict:
        s = self.service.data_status()
        s["readiness"] = self.readiness()
        s["simulated"] = self.store.get_meta("simulated_world")
        s["policy"] = POLICY
        s["key_setup"] = {"names": KEY_NAMES, "text": "Keys are never typed into this page. On Replit open Tools, then Secrets, and add OPENFILINGS_API_KEY_1 (and _2 ... _5 for more) with the key as the value; then stop and run the app. "
                          "Elsewhere put them in the environment or in the git-ignored .env file."}
        return clean(s)

    def overview(self) -> dict:
        counts = self.store.counts()
        securities = self.store.securities()
        by_sector = securities["sector"].fillna("unknown").value_counts()
        by_market = securities["market"].fillna("unknown").value_counts()
        coverage = self.store.coverage()
        metric_cov = self.store.metric_coverage()
        lib = self.library.counts()
        top = [{"alpha_id": a["alpha_id"], "text": a["text"], "status": a["status"], "ic": a["ic"], "oos_ic": a["oos_ic"], "sharpe": a["sharpe"], "fitness": a["fitness"]} for a in self.library.list(limit=5)]
        runs = Discovery(self.store).runs(5)
        backtests = self.experiments("backtest", 5)
        return clean({"readiness": self.readiness(), "study_size": self.study_size(), "counts": counts, "by_country": coverage.to_dict("records"), "by_sector": [{"sector": k, "companies": int(v)} for k, v in by_sector.items()],
                      "by_market": [{"market": k, "companies": int(v)} for k, v in by_market.items()], "metric_coverage": metric_cov.head(30).to_dict("records"), "quality": self.store.quality_counts(),
                      "library": {**lib, "top": top}, "runs": runs, "backtests": backtests, "simulated": self.store.get_meta("simulated_world"),
                      "pipeline": ["providers", "database", "point-in-time", "metrics", "factors", "tests", "discovery", "library", "combination", "backtest"]})

    def experiments(self, kind: str | None = None, limit: int = 20) -> list[dict]:
        sql = "SELECT experiment_id, kind, created_at, spec, data_version, code_version, summary, status FROM experiments" + (" WHERE kind = ?" if kind else "") + " ORDER BY created_at DESC LIMIT ?"
        with self.store.db.transaction() as conn:
            rows = conn.execute(sql, ([kind] if kind else []) + [int(limit)]).fetchall()
        import json as _json
        return [{**dict(r), "spec": _json.loads(r["spec"]) if r["spec"] else {}, "summary": _json.loads(r["summary"]) if r["summary"] else {}} for r in rows]

    def quality(self, limit: int = 100) -> dict:
        with self.store.db.transaction() as conn:
            rows = conn.execute("SELECT s.ticker, s.exchange, p.fiscal_year, p.fiscal_period, p.period_end, o.metric_code, o.value, o.quality_flag FROM fundamental_observations o "
                                "JOIN financial_periods p ON p.id = o.financial_period_id JOIN securities s ON s.security_id = p.security_id WHERE o.quality_flag <> 'ok' "
                                "ORDER BY p.period_end DESC LIMIT ?", (int(limit),)).fetchall()
        return clean({"counts": self.store.quality_counts(), "examples": [dict(r) for r in rows],
                      "explain": {"identity_error": "the figures do not add up (for example assets differ from liabilities plus equity)", "unit_error": "a figure is orders of magnitude away from the company's own neighbours",
                                  "ok": "no problem found", "estimated": "a filing date had to be estimated"}})

    # ------------------------------------------------------------------------------------------------------------ companies
    def companies(self, query: str = "", country: str = "", sector: str = "", limit: int = 50, offset: int = 0) -> dict:
        limit, offset = _number(limit, "limit", 1, 500, True), _number(offset, "offset", 0, 10_000_000, True)
        where, args = ["1=1"], []
        if query.strip():
            where.append("(s.ticker LIKE ? COLLATE NOCASE OR s.company_name LIKE ? COLLATE NOCASE OR s.isin = ?)")
            args += [f"%{query.strip()}%", f"%{query.strip()}%", query.strip().upper()]
        if country:
            where.append("s.country = ?")
            args.append(country)
        if sector:
            where.append("s.sector = ?")
            args.append(sector)
        sql = (f"FROM securities s WHERE {' AND '.join(where)}")
        with self.store.db.transaction() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) {sql}", args).fetchone()[0])
            rows = conn.execute(
                "SELECT s.security_id, s.ticker, s.exchange, s.company_name, s.country, s.currency, s.sector, s.industry, s.active, s.delisted_date, s.source, "
                "(SELECT COUNT(*) FROM financial_periods p WHERE p.security_id = s.security_id AND p.fiscal_period = 'FY' AND p.vintage = 0) AS annual, "
                "(SELECT COUNT(*) FROM financial_periods p WHERE p.security_id = s.security_id AND p.fiscal_period <> 'FY' AND p.vintage = 0) AS quarterly, "
                "(SELECT MAX(p.period_end) FROM financial_periods p WHERE p.security_id = s.security_id) AS last_period_end, "
                "(SELECT COUNT(*) FROM prices_daily d WHERE d.security_id = s.security_id) AS price_days "
                f"{sql} ORDER BY s.ticker, s.exchange LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
            countries = [r[0] for r in conn.execute("SELECT DISTINCT country FROM securities WHERE country IS NOT NULL ORDER BY country")]
            sectors = [r[0] for r in conn.execute("SELECT DISTINCT sector FROM securities WHERE sector IS NOT NULL ORDER BY sector")]
        return clean({"total": total, "offset": offset, "limit": limit, "companies": [dict(r) for r in rows], "countries": countries, "sectors": sectors})

    FIGURES = ("revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow", "capital_expenditure", "free_cash_flow", "total_assets", "total_debt", "shareholders_equity",
               "cash", "shares_outstanding")
    RATIOS = ("gross_margin", "operating_margin", "net_margin", "roe", "roa", "roic", "revenue_growth", "fcf_margin", "debt_to_equity", "interest_coverage", "accruals_ratio", "asset_turnover",
              "piotroski_f", "accounting_anomaly_score")

    def company(self, security_id: int, years: int = 12) -> dict:
        """One company as the database holds it: the profile, what was fetched and when, the annual figures with the date each became public, the ratios built from them, the DuPont decomposition of
        the change in return on equity, the Piotroski F-score and the accounting-anomaly score, and the figures the quality checks flagged."""
        info = self.store.get_security(int(security_id))
        if info is None:
            raise FundamentalsError(f"no company with id {security_id}")
        sid = int(security_id)
        tables = PeriodTables(self.store, [sid], PitSettings())
        metrics.compute(tables, "fy")
        fy = tables.table("fy")
        fy = fy[fy["security_id"] == sid].sort_values("period_end").tail(int(years)).copy()
        keep_figures = [c for c in self.FIGURES if c in fy.columns]
        keep_ratios = [c for c in self.RATIOS if c in fy.columns]
        annual = []
        for _, r in fy.iterrows():
            annual.append({"fiscal_year": int(r["fiscal_year"]), "period_end": r["period_end"], "available": r["available_date"], "estimated": bool(r["pit_estimated"]), "quality": r["pit_quality"],
                           "provider": r["provider"], "currency": r["reporting_currency"], **{c: r[c] for c in keep_figures + keep_ratios}})
        dup = analytics.dupont(tables, "fy")
        dup = dup.loc[fy.index]
        sig = analytics.signals(tables, "fy").loc[fy.index]
        parts = analytics.anomaly_parts(tables, "fy").loc[fy.index]
        latest_anomaly = fy["accounting_anomaly_score"].dropna().iloc[-1] if "accounting_anomaly_score" in fy and fy["accounting_anomaly_score"].notna().any() else None
        ids = self.store.identifiers(sid)
        price = self.store.prices([sid])
        close = self.store.closes([sid])
        sync = {r: self.store.get_sync(sid, info["source"], f"financials:{r}") for r in ("annual", "quarter")} if info.get("source") else {}
        flagged = self.store.inspect(sid)
        flagged = flagged[flagged["quality_flag"] != "ok"].head(40)
        dates = [str(d.date()) for d in fy["period_end"]]
        series = {name: [None if pd.isna(v) else float(v) for v in fy[name]] for name in keep_figures + keep_ratios if name in fy.columns and not str(fy[name].dtype).startswith("object")}
        return clean({
            "profile": info, "identifiers": ids.to_dict("records"), "sync": sync, "summary": self.store.period_summary(sid),
            "prices": None if price.empty else {"first": str(price.index[0].date()), "last": str(price.index[-1].date()), "days": int(len(price)),
                                                 "last_close": None if close.empty else float(close.iloc[-1, 0]), "last_adjusted": float(price.iloc[-1, 0])},
            "annual": annual, "figures": keep_figures, "ratios": keep_ratios, "dates": dates, "series": series,
            "dupont": [{"fiscal_year": int(y), **{k: v for k, v in row.items() if k not in ("roe_from_steps",)}} for y, (_, row) in zip(fy["fiscal_year"], dup.iterrows())],
            "dupont_steps": [{"key": k, "label": label} for k, label in analytics.DUPONT_STEPS],
            "piotroski": [{"fiscal_year": int(y), **row.to_dict()} for y, (_, row) in zip(fy["fiscal_year"], sig.iterrows())], "piotroski_signals": [{"key": k, "label": label} for k, label in analytics.PIOTROSKI],
            "anomaly": [{"fiscal_year": int(y), **row.to_dict()} for y, (_, row) in zip(fy["fiscal_year"], parts.iterrows())],
            "anomaly_reading": None if latest_anomaly is None else analytics.interpret_anomaly(float(latest_anomaly)),
            "flagged": flagged.to_dict("records"),
            "notes": ["Each row appears in research from the day the figures were public (the 'available' date), never from its period end.",
                      "The Piotroski F-score and the anomaly score are screens, not verdicts: a high anomaly score means 'look closer' and is never evidence of fraud."]})

    def set_price_symbol(self, security_id: int, symbol: str) -> dict:
        if self.store.get_security(int(security_id)) is None:
            raise FundamentalsError(f"no company with id {security_id}")
        prices.set_price_symbol(self.store, int(security_id), symbol)
        return {"security_id": int(security_id), "symbol": str(symbol).strip().upper()}

    # ------------------------------------------------------------------------------------------------------------ the catalogue of choices
    def catalogue(self) -> dict:
        factors = factor_library.catalogue()
        ops = {"winsorize": "cut each tail at a percentile (0.5, 1, 2.5 or 5)", "zscore": "subtract the date's mean, divide by its standard deviation", "rank": "rank the companies on each date",
               "percentile": "the rank as a share, 0 to 1", "sector_rank": "rank within the company's sector", "industry_rank": "rank within the industry", "country_rank": "rank within the country",
               "group_demean": "subtract the mean of the group (sector, industry, country, region or market)", "group_zscore": "z-score within a group", "delta": "change over some periods",
               "pct_change": "percentage change over some periods", "hist_zscore": "the company's own history: today against its past mean", "neutralize": "remove what the factor shares with sector, industry, country, market, size, beta or volatility",
               "fill": "fill a missing value by a stated rule (neutral, sector median or similar; never silently zero)", "sign": "multiply by +1 or -1 (turn 'low is good' into 'high is good')",
               "clip": "cut the values at plus and minus a limit"}
        return clean({
            "factors": factors, "families": list(factor_library.FAMILIES), "transforms": [{"op": o, "text": ops.get(o, "")} for o in transforms.OPS], "groups": list(GROUPS), "winsor_levels": list(transforms.WINSOR_LEVELS),
            "horizons": [{"id": k, "days": v} for k, v in sorted(HORIZONS.items(), key=lambda kv: kv[1])], "frequencies": [{"id": k, "days": v} for k, v in FREQ_DAYS.items()],
            "weightings": ["equal", "signal", "inv_vol"], "directions": ["theory", "auto", 1, -1], "breakdowns": ["country", "sector", "region", "regime"],
            "metrics": metrics.catalogue(), "operators": {"binary": list(X.BINARY), "unary": list(X.UNARY), "time_series": list(X.TIME_SERIES), "cross_sectional": list(X.CROSS_SECTIONAL)},
            "formula": {"max_nodes": X.MAX_NODES, "max_depth": X.MAX_DEPTH, "examples": ["div(fcf,market_cap)", "delta(roic,12)", "neg(growth(total_assets,12))", "sector_rank(div(operating_income,total_assets))"]},
            "discovery": {"defaults": {**DiscoverySpec().describe(), "strictness": DEFAULT_STRICTNESS}, "stages": list(STAGES), "thresholds": Thresholds().__dict__, "fitness": FITNESS,
                          "strictness": list(STRICTNESS), "presets": {name: {**Thresholds().__dict__, **PRESETS[name]} for name in STRICTNESS}, "placebo_repeats": calibration.MAX_REPEATS},
            "combination": {"methods": list(combine.METHODS), "models": list(combine.MODELS), "grids": combine.GRIDS, "limits": combine.PARAM_LIMITS},
            "portfolio": {"defaults": bridge.PortfolioSpec().describe(), "sides": list(bridge.SIDES), "selections": list(bridge.SELECTIONS), "weightings": list(bridge.WEIGHTINGS), "technical": bridge.technical_names()},
            "library": {"statuses": list(STATUSES), "transitions": {k: list(v) for k, v in TRANSITIONS.items()}, "monitor": list(MONITOR_STATES)}, "providers": list(PROVIDER_NAMES),
            "rule": "An interesting factor is not a validated alpha: only a formula that passed the staged tests on dates it was not chosen on, and survived the correction for every idea tried, is called a Candidate, and nothing is called Validated without time."})

    # ------------------------------------------------------------------------------------------------------------ factor studies
    def evaluate(self, ctx: ResearchContext, spec: EvalSpec) -> dict:
        return clean(evaluate_factor(ctx, spec))

    def scan(self, ctx: ResearchContext, names: list[str] | None, spec: EvalSpec, progress=None, stop=None) -> dict:
        """Every factor (or the ones named) evaluated under one set of conditions and set side by side: how many companies each covers, its information coefficient, the long-short portfolio net of costs, and
        whether its buckets rise in order. Evaluating many factors is multiple testing: the table says how many were looked at."""
        say = progress or (lambda text: None)
        names = list(names) if names else factor_library.names()
        if not names or len(names) > MAX_SCAN:
            raise FundamentalsError(f"scan between 1 and {MAX_SCAN} factors at a time")
        for n in names:
            factor_library.get(n)
        short = EvalSpec.from_dict({**spec.describe(), "breakdowns": (), "premium": False})
        rows = []
        for i, name in enumerate(names):
            if stop is not None and stop():
                break
            say(f"{name} ({i + 1} of {len(names)})")
            f = factor_library.get(name)
            try:
                r = evaluate_factor(ctx, EvalSpec.from_dict({**short.describe(), "factor": name}))
            except FundamentalsError as error:
                rows.append({"name": name, "label": f.label, "family": f.family, "error": str(error)[:200]})
                continue
            rows.append({"name": name, "label": f.label, "family": f.family, "direction": f.direction, "coverage": factor_library.coverage(ctx, name), "mean_ic": r["ic"].get("mean_ic"),
                         "t_stat": r["ic"].get("t_stat"), "icir": r["ic"].get("icir"), "positive_share": r["ic"].get("positive_share"), "spread": r["quantiles"].get("spread_top_minus_bottom"),
                         "monotonicity": (r["quantiles"].get("monotonicity") or {}).get("spearman"), "net_sharpe": (r["portfolio"].get("net") or {}).get("sharpe"),
                         "gross_sharpe": (r["portfolio"].get("gross") or {}).get("sharpe"), "turnover": r["portfolio"].get("avg_turnover"), "n_dates": r.get("n_dates")})
        done = [r for r in rows if "error" not in r]
        done.sort(key=lambda r: -abs(r["t_stat"] if r.get("t_stat") is not None else 0.0))
        return clean({"rows": done + [r for r in rows if "error" in r], "tested": len(done), "universe": ctx.describe(), "spec": short.describe(),
                      "note": f"{len(done)} factors were looked at under one set of conditions. With that many, a few show |t| above 2 by luck alone (about one in twenty at the 5% level): treat this table as "
                              "a map of where to look, not as a list of alphas."})

    def check_formula(self, text: str, ctx: ResearchContext | None = None) -> dict:
        """Parse a formula and say what it is. With a context it also checks the types against the data's fields and reports how many companies it covers."""
        try:
            tree = X.parse(str(text))
        except FundamentalsError as error:
            return {"ok": False, "error": str(error)}
        out: dict = {"ok": True, "text": X.to_string(tree), "canonical": X.canonical_string(tree), "hash": X.expression_hash(tree), "nodes": X.size(X.canonical(tree)),
                     "fields": sorted(set(X.fields_used(X.canonical(tree)))), "tree": tree}
        if ctx is None:
            return clean(out)
        try:
            info = X.describe(tree, ctx)
        except FundamentalsError as error:
            return {"ok": False, "error": str(error), "text": out["text"]}
        out.update(well_typed=info["well_typed"], problem=info["problem"], kind=info["kind"])
        if info["well_typed"]:
            panel, _ = X.evaluate(tree, ctx)
            mask = ctx.investable()
            out["coverage"] = float(panel.where(mask).notna().to_numpy().sum() / max(int(mask.to_numpy().sum()), 1))
        else:
            out["ok"] = False
            out["error"] = info["problem"]
        return clean(out)

    def evaluate_formula(self, ctx: ResearchContext, text: str, spec: EvalSpec) -> dict:
        tree = X.parse(str(text))
        sign = 1.0
        if spec.direction in (1, -1):
            sign = float(spec.direction)
        signal = build_alpha_signal(ctx, tree, {"steps": spec.steps, "direction": sign})
        notes = []
        if spec.direction == "auto":
            ic = rank_ic(signal, ctx.forward(spec.horizon).reindex(signal.index), spec.min_assets)
            if len(ic) and ic.mean() < 0:
                signal = -signal
                notes.append("direction 'auto': the sign was flipped because the in-sample IC was negative. That is a choice made on the data; count it as a second trial.")
            else:
                notes.append("direction 'auto': the formula's own sign was kept. Choosing the sign that worked is a choice made on the data; count it as a second trial.")
        result = evaluate(ctx, signal, spec, label=X.to_string(tree))
        result["warnings"] = notes + list(result["warnings"])
        result["formula"] = self.check_formula(text)
        return clean(result)

    def lookahead(self, spec: EvalSpec, universe: dict | None, freq: str, pit: dict | None) -> dict:
        return look_ahead_audit(self.store, spec, universe_spec(universe), freq, pit_settings(pit))

    # ------------------------------------------------------------------------------------------------------------ combining alphas
    def signals(self, ctx: ResearchContext, parts: list[dict]) -> tuple[dict, list[dict]]:
        """Each part of a combination resolved to a signal panel (a factor, a technical signal, a library alpha or a formula) with what it is."""
        if not isinstance(parts, list) or not 2 <= len(parts) <= bridge.MAX_PARTS:
            raise FundamentalsError(f"choose between 2 and {bridge.MAX_PARTS} signals to put together")
        out, infos = {}, []
        for j, part in enumerate(parts):
            panel, info = bridge.resolve(ctx, part, self.library)
            out[f"{j + 1}:{info['label']}"] = panel
            infos.append(info)
        return out, infos

    def correlation(self, ctx: ResearchContext, parts: list[dict], cut: float = 0.5) -> dict:
        signals, infos = self.signals(ctx, parts)
        out = combine.analyse(ctx, signals, _number(cut, "the cluster distance", 0.05, 0.95))
        out["warnings"] = sorted({w for i in infos for w in i["warnings"]})
        return clean(out)

    def combination(self, ctx: ResearchContext, parts: list[dict], method: str, horizon="1m", options: dict | None = None, spec: EvalSpec | None = None) -> dict:
        signals, infos = self.signals(ctx, parts)
        opts = {k: v for k, v in (options or {}).items() if k in ("window", "min_obs", "shrink", "orthogonalize", "nonnegative", "cut")}
        result = combine.combine(ctx, signals, method, horizon, **opts)
        spec = spec or EvalSpec(horizon=horizon)
        spec = EvalSpec.from_dict({**spec.describe(), "horizon": horizon, "breakdowns": (), "premium": False})
        fwd, h = ctx.forward(horizon), horizon_days(horizon)
        stats_ = {}
        for name, panel in {**signals, "combined": result.signal}.items():
            s = combine.standardise({name: panel}, ctx)[name] if name != "combined" else panel
            stats_[name] = ic_stats(rank_ic(s, fwd.reindex(s.index), spec.min_assets), h, FREQ_DAYS[ctx.freq], PERIODS_PER_YEAR[ctx.freq])
        equal = combine.combine(ctx, signals, "equal", horizon).signal
        stats_["equal weight"] = ic_stats(rank_ic(equal, fwd.reindex(equal.index), spec.min_assets), h, FREQ_DAYS[ctx.freq], PERIODS_PER_YEAR[ctx.freq])
        names = list(signals)
        weights = result.weights.reindex(columns=names)
        step = max(len(weights) // 120, 1)
        return clean({"method": method, "notes": result.notes, "parts": [{"label": i["label"], "kind": i["kind"], **stats_[n]} for n, i in zip(names, infos)], "combined": stats_["combined"],
                      "equal_weight": stats_["equal weight"], "weights": {"dates": [str(d.date()) for d in weights.index[::step]], "names": [i["label"] for i in infos],
                                                                         "values": weights.iloc[::step].round(4).to_numpy().tolist()},
                      "evaluation": evaluate(ctx, result.signal, spec, label=f"{method} combination"), "warnings": sorted({w for i in infos for w in i["warnings"]}),
                      "ic": series_points(rank_ic(result.signal, fwd.reindex(result.signal.index), spec.min_assets))})

    def machine_learning(self, ctx: ResearchContext, parts: list[dict], model: str, horizon="1m", options: dict | None = None, spec: EvalSpec | None = None) -> dict:
        signals, infos = self.signals(ctx, parts)
        opts = {k: v for k, v in (options or {}).items() if k in ("min_train", "retrain_every", "window", "embargo", "tune", "seed", "permutation_repeats", "params")}
        result = combine.ml_combine(ctx, signals, model, horizon, **opts)
        spec = EvalSpec.from_dict({**(spec or EvalSpec()).describe(), "horizon": horizon, "breakdowns": (), "premium": False})
        comparison = combine.compare_with_rules(ctx, result, signals, horizon)
        return clean({"model": model, "notes": result.notes, "folds": result.folds, "importance": result.importance, "comparison": comparison,
                      "parts": [i["label"] for i in infos], "evaluation": evaluate(ctx, result.signal, spec, label=f"{model} model"),
                      "warnings": sorted({w for i in infos for w in i["warnings"]})})

    def backtest(self, ctx: ResearchContext, signal_spec, portfolio: dict | None, name: str | None = None, n_trials: int | None = None) -> dict:
        return clean(bridge.tear(ctx, signal_spec, portfolio, self.library, name, n_trials))

    # ------------------------------------------------------------------------------------------------------------ the library and its monitor
    def alpha(self, alpha_id: str) -> dict:
        rec = self.library.get(alpha_id)
        return clean({**rec, "status_log": self.library.status_log(alpha_id), "monitor": self.library.monitor_history(alpha_id), "transitions": list(TRANSITIONS.get(rec["status"], ()))})

    def monitor(self, alpha_ids: list[str] | None = None, progress=None, stop=None) -> dict:
        """Recompute alphas on the data now in the database and judge each: Healthy, Weakening, Decaying or Broken. By default every alpha that is not Retired or Rejected."""
        say = progress or (lambda text: None)
        if alpha_ids is None:
            alpha_ids = [a["alpha_id"] for a in self.library.list(limit=1000) if a["status"] not in ("Retired", "Rejected")]
        results = []
        for i, alpha_id in enumerate(alpha_ids):
            if stop is not None and stop():
                break
            rec = self.library.get(alpha_id)
            s = rec["settings"]
            say(f"{rec['text']} ({i + 1} of {len(alpha_ids)})")
            try:
                ctx = self.context(s.get("universe"), s.get("freq", "M"), int(s.get("entry_lag", 1)))
                results.append({**self.library.monitor(alpha_id, ctx), "text": rec["text"]})
            except FundamentalsError as error:
                results.append({"alpha_id": alpha_id, "text": rec["text"], "monitor": "Broken", "detail": {"error": str(error)[:200]}, "status": rec["status"]})
        return clean({"results": results, "checked": len(results),
                      "note": "A validated or production alpha judged Decaying or Broken is moved to Decaying automatically; nothing is ever deleted."})

    # ------------------------------------------------------------------------------------------------------------ discovery
    def discovery_spec(self, raw: dict | None) -> DiscoverySpec:
        raw = _strict(DiscoverySpec, raw, "discovery")
        raw.setdefault("strictness", DEFAULT_STRICTNESS)                          # a request that does not say starts from the standard bars, not from the first-look ones
        spec = DiscoverySpec.from_dict(raw)
        if spec.strictness not in STRICTNESS:
            raise FundamentalsError(f"strictness must be one of {list(STRICTNESS)}")
        universe_spec(spec.universe)
        if spec.freq not in FREQ_DAYS:
            raise FundamentalsError("freq must be W, M or Q")
        horizon_days(spec.horizon)
        spec.n_candidates = _number(spec.n_candidates, "the number of formulas", 1, 20_000, True)
        spec.finalists = _number(spec.finalists, "the number of finalists", 1, 200, True)
        spec.max_nodes = _number(spec.max_nodes, "the largest formula", 3, X.MAX_NODES, True)
        spec.seed = _number(spec.seed, "the seed", 0, 1_000_000, True)
        spec.train_fraction = _number(spec.train_fraction, "the training share", 0.3, 0.8)
        spec.validation_fraction = _number(spec.validation_fraction, "the validation share", 0.1, 0.4)
        if spec.train_fraction + spec.validation_fraction > 0.9:
            raise FundamentalsError("training and validation together must leave at least a tenth of the dates for the locked test")
        if spec.seeds is not None:
            if not isinstance(spec.seeds, list) or len(spec.seeds) > 200:
                raise FundamentalsError("seeds is a list of formulas (at most 200)")
            for text in spec.seeds:
                X.parse(str(text))
        if not isinstance(spec.thresholds, dict):
            raise FundamentalsError("thresholds is a dictionary of bars")
        unknown = sorted(set(spec.thresholds) - {f.name for f in dataclass_fields(Thresholds)})
        if unknown:
            raise FundamentalsError(f"unknown threshold(s) {unknown}")
        spec.thresholds = {k: _number(v, f"the bar {k}", -1e6, 1e6) for k, v in spec.thresholds.items()}
        return spec

    def discovery(self) -> Discovery:
        return Discovery(self.store)

    def seeds_for(self, ctx: ResearchContext) -> list[str]:
        return default_seeds(ctx)

    def splits_for(self, ctx: ResearchContext, spec: DiscoverySpec) -> dict:
        return splits(ctx, spec)

    def preflight(self, ctx: ResearchContext, spec: DiscoverySpec) -> dict:
        """What a search could find, before it is run: for each of the three blocks of dates the smallest average IC that block can tell from luck, how many formulas luck alone is expected to push
        through the first screen, and the chance that a real alpha of each typical strength would get through the first screen, the validation block and the locked test (the screens before and
        between those only lower it). A search that could not find a real alpha of the strength it hopes for is not worth the afternoon, and says so here."""
        blocks, limits = splits(ctx, spec), spec.limits()
        h, step = horizon_days(spec.horizon), FREQ_DAYS[ctx.freq]
        forward = ctx.forward(spec.horizon)
        power = {}
        for name in ("train", "validation", "oos"):
            first, last = blocks[name]
            companies = float(np.median(forward.loc[first:last].notna().sum(axis=1)))
            power[name] = detectable_effect(companies, blocks["dates"][name], h, step)
        if not all(p.get("available") for p in power.values()):
            raise FundamentalsError("too few companies or dates in one of the blocks to say what a search could find")
        se = {name: p["se_of_mean_ic"] for name, p in power.items()}
        norm = stats.norm
        table = []
        for ic in TYPICAL_IC:
            first = float(norm.cdf(ic / se["train"] - limits.t_ic) + norm.cdf(-ic / se["train"] - limits.t_ic))
            validated = float(norm.cdf(ic / se["validation"] - limits.val_t))
            locked = float(norm.cdf(ic / se["oos"] - limits.oos_t))
            table.append({"ic": ic, "first_screen": first, "validation": validated, "locked_test": locked, "found": first * validated * locked})
        luck = float(spec.n_candidates * 2.0 * norm.sf(limits.t_ic))
        found_at = {row["ic"]: row["found"] for row in table}
        adequate = bool(found_at[0.05] >= 0.5)
        z80 = float(norm.ppf(0.8))
        reading = (f"With about {power['train']['companies']:.0f} companies on {blocks['dates']['train']} training dates, the first screen (|t| of {limits.t_ic:g}) catches a real mean IC of "
                   f"{(limits.t_ic + z80) * se['train']:.3f} or more four times in five, and the locked test, on {blocks['dates']['oos']} dates, one of {(limits.oos_t + z80) * se['oos']:.3f} or more. "
                   f"A real alpha with an IC of 0.03 would get through the first screen, the validation block and the locked test about {found_at[0.03]:.0%} of the time, and one of 0.05 about "
                   f"{found_at[0.05]:.0%} (before the screens in between, which lower it). Luck alone is expected to pass the first screen with about {luck:.0f} of the {spec.n_candidates} formulas.")
        if not adequate:
            reading += (" This search is underpowered for the alphas that are worth having: add companies or history, or lower the bars (exploratory), knowing that lower bars let more luck through.")
        return clean({"strictness": spec.strictness, "limits": asdict(limits), "blocks": blocks, "formulas": spec.n_candidates, "power": power, "table": table, "luck_first_screen": luck,
                      "adequate": adequate, "reading": reading})

    def placebo(self, spec: DiscoverySpec, repeats: int = 5, real: dict | None = None, progress=None, stop=None) -> dict:
        """Run a search ``repeats`` times on returns shuffled among the companies (so nothing can be found) and report what it finds anyway, how much of ``real`` (the counts of a real run) that
        makes luck, and whether the t-statistics can be believed. Nothing is written to the library or the graveyard; the result is kept with the experiments."""
        ctx = self.context(spec.universe, spec.freq, spec.entry_lag)
        repeats = _number(repeats, "the number of shuffles", 1, calibration.MAX_REPEATS, True)
        result = calibration.calibrate_search(ctx, spec, repeats, progress=progress, stop=stop, real=real)
        if result["complete"]:
            result["experiment_id"] = calibration.save(self.store, result)
        return clean(result)

    def study_size(self, horizon: str = "1m", min_assets: int = 20) -> dict:
        """What a factor study on the database as it stands could tell apart from luck: roughly how many companies have both a statement and prices at the same time, on how many month-ends, and the
        smallest average IC that many companies and dates can see. It reads two columns from the database and builds nothing, so the Overview can show it; a study that follows may differ a
        little (it also drops companies that are not investable)."""
        h, step = horizon_days(horizon), FREQ_DAYS["M"]
        key = (self._signature(), h, min_assets)
        with self._lock:
            if self._size and self._size[0] == key:
                return self._size[1]
        result = self._study_size(h, step, min_assets)
        with self._lock:
            self._size = (key, result)
        return result

    def _study_size(self, h: int, step: int, min_assets: int) -> dict:
        with self.store.db.transaction() as conn:
            span = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT security_id, MIN(date), MAX(date) FROM prices_daily GROUP BY security_id")}
            filed = {r[0]: r[1] for r in conn.execute("SELECT security_id, MIN(COALESCE(filing_date, publication_date, accepted_date, period_end)) FROM financial_periods "
                                                       "WHERE fiscal_period = 'FY' AND vintage = 0 GROUP BY security_id")}
        both = [sid for sid in span if sid in filed and filed[sid]]
        base = {"companies_with_both": len(both), "min_assets": min_assets}
        if len(both) < min_assets:
            return {**base, "available": False, "reading": f"A factor study needs at least {min_assets} companies that have statements and prices at the same time; the database has {len(both)}."}
        start = np.array([max(pd.Timestamp(span[s][0]), pd.Timestamp(filed[s]) + pd.Timedelta(days=45)).value for s in both])
        end = np.array([(pd.Timestamp(span[s][1]) - pd.Timedelta(days=h * 7 // 5)).value for s in both])
        months = pd.period_range(pd.Timestamp(int(start.min())), pd.Timestamp(int(end.max())), freq="M").to_timestamp(how="end")
        counts = np.array([int(((start <= m.value) & (m.value <= end)).sum()) for m in months])
        usable = counts >= min_assets
        if usable.sum() < 12:
            return {**base, "available": False, "dates": int(usable.sum()), "reading": f"There are {int(usable.sum())} month-ends with at least {min_assets} companies that have statements and prices at "
                                                                                      "once; a study needs at least a year of them, and three or more to say anything."}
        n, t = float(np.median(counts[usable])), int(usable.sum())
        out = detectable_effect(n, t, h, step)
        return clean({**base, **out, "companies": n, "dates": t})

    # ------------------------------------------------------------------------------------------------------------ prices
    def load_prices(self, security_ids: list[int] | None = None, start: str = prices.DEFAULT_START, fetch=None, progress=None, stop=None, refresh: bool = False, fx: bool = True) -> dict:
        fetch = fetch or prices.yahoo_download
        report = prices.load_prices(self.store, security_ids, _date(start, "start") or prices.DEFAULT_START, fetch, progress, stop, refresh)
        if fx and not report.get("stopped"):
            report["fx"] = prices.load_fx(self.store, _date(start, "start") or prices.DEFAULT_START, fetch, progress)
        self.forget()
        return clean(report)


__all__ = ["Workbench", "universe_spec", "pit_settings", "eval_spec", "KEY_NAMES", "POLICY", "MAX_SYMBOLS"]
