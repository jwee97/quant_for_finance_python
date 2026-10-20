"""The Fundamentals tab's endpoints: company fundamentals in a local database, factor studies, automated alpha discovery, the alpha library and its monitor, combinations of alphas, and the backtest of an
alpha through the platform's own engine.

Everything here is a thin wrapper over :class:`fundamentals.workbench.Workbench`, which the ``quant fundamentals`` command line uses as well. Reading endpoints answer at once from the database;
anything that can take more than a moment (fetching statements, loading prices, a study, a discovery run, a model, a backtest) starts a job (kind ``fundamentals``) that the page polls through
``/api/job/<id>`` like a backtest, and a long one can be stopped.

**Two databases.** ``db`` (a query field or a body field) is ``real`` (the default) or ``demo``. The demo database holds a simulated world with alphas planted in it, to try the tab and to test it;
it is a separate file, so a simulated company can never be mixed into a study of real ones, and an alpha found in it never reaches the real library. Fetching from a provider is refused on the demo
database, and the simulated world is refused on the real one.

**Keys.** A provider key is read from the environment (on Replit: Secrets) and never passes through the page: nothing here accepts a key, and nothing that leaves contains one. Fetching statements
spends the provider's credits, so ``/api/fund/sync`` will not start without a ``budget`` (the most credits this run may spend); the plan endpoint says what a run would cost first.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from ..fundamentals import prices as price_module
from ..fundamentals.discovery import Discovery, DiscoverySpec
from ..fundamentals.errors import FundamentalsError
from ..fundamentals.jsonutil import clean
from ..fundamentals.library import STATUSES
from ..fundamentals.providers.openfilings import SEARCH_KINDS
from ..fundamentals.providers.simulated import PLANTED
from ..fundamentals.service import FundamentalsService
from ..fundamentals.workbench import MAX_SYMBOLS, Workbench, eval_spec
from .api import ACTIVE, ApiError

DBS = ("real", "demo")
DEMO_FILE = Path("data") / "user" / "fundamentals_demo.db"
SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.:\-^=]{0,23}$")
MAX_BUDGET = 1_000_000
CONTEXT_FIELDS = ("universe", "freq", "entry_lag", "start", "end", "pit")


def _number(raw, label: str, lo: float, hi: float, integer: bool = False):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ApiError(f"{label} must be a number") from None
    if not value == value or not lo <= value <= hi:                                     # NaN fails both tests
        raise ApiError(f"{label} must be between {lo:g} and {hi:g}")
    return int(round(value)) if integer else value


def _db(source: dict) -> str:
    db = str(source.get("db") or "real")
    if db not in DBS:
        raise ApiError(f"db must be one of {', '.join(DBS)}")
    return db


def _symbols(raw) -> list[str]:
    if isinstance(raw, str):
        raw = re.split(r"[\s,;]+", raw.strip())
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        raise ApiError("symbols must be a list of tickers such as us:AAPL or jp:7203")
    out = [s.strip() for s in raw if s.strip()]
    if not out:
        raise ApiError("name at least one company")
    if len(out) > MAX_SYMBOLS:
        raise ApiError(f"at most {MAX_SYMBOLS} companies at a time")
    bad = [s for s in out if not SYMBOL.match(s)]
    if bad:
        raise ApiError(f"'{bad[0]}' is not a valid ticker (letters, digits and . : - only, up to 24 characters)")
    return list(dict.fromkeys(out))


def _ids(raw, label: str = "security_ids") -> list[int] | None:
    if raw in (None, []):
        return None
    if not isinstance(raw, list) or len(raw) > 100_000:
        raise ApiError(f"{label} must be a list of company ids")
    try:
        return [int(i) for i in raw]
    except (TypeError, ValueError):
        raise ApiError(f"{label} must be a list of company ids") from None


class FundamentalsApi:
    def __init__(self, app):
        self.app = app
        self.benches: dict[str, Workbench] = {}
        self.stops: dict[str, threading.Event] = {}
        self.fetch = None                                                # replaced in tests; the default asks Yahoo Finance
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------------------------------ routes
    def get_routes(self) -> dict:
        return self._clean({"/api/fund/status": self.status, "/api/fund/overview": self.overview, "/api/fund/catalogue": self.catalogue, "/api/fund/companies": self.companies,
                            "/api/fund/company": self.company, "/api/fund/quality": self.quality, "/api/fund/library": self.library, "/api/fund/alpha": self.alpha,
                            "/api/fund/graveyard": self.graveyard, "/api/fund/runs": self.runs, "/api/fund/run": self.run_summary, "/api/fund/experiments": self.experiments})

    def post_routes(self) -> dict:
        return self._clean({"/api/fund/companies/add": self.add_companies, "/api/fund/companies/search": self.search_companies, "/api/fund/sync/plan": self.sync_plan, "/api/fund/provider/test": self.provider_test,
                            "/api/fund/price-symbol": self.price_symbol, "/api/fund/formula/check": self.formula_check, "/api/fund/library/status": self.library_status,
                            "/api/fund/library/validate": self.library_validate, "/api/fund/sync": self.sync, "/api/fund/simulate": self.simulate, "/api/fund/prices": self.load_prices,
                            "/api/fund/scan": self.scan, "/api/fund/evaluate": self.evaluate, "/api/fund/evaluate/formula": self.evaluate_formula, "/api/fund/lookahead": self.lookahead,
                            "/api/fund/discover": self.discover, "/api/fund/monitor": self.monitor, "/api/fund/correlation": self.correlation, "/api/fund/combine": self.combine,
                            "/api/fund/ml": self.ml, "/api/fund/backtest": self.backtest, "/api/fund/stop": self.stop})

    @staticmethod
    def _clean(routes: dict) -> dict:
        """Every answer goes out as plain JSON (no NaN, no numpy): the server refuses to send anything else."""
        def wrapped(fn):
            return lambda arg=None: clean(fn(arg))
        return {path: wrapped(fn) for path, fn in routes.items()}

    # ------------------------------------------------------------------------------------------------------------ the two databases
    def path_of(self, db: str) -> Path:
        return self.app.root / DEMO_FILE if db == "demo" else self.app.root / "data" / "user" / "fundamentals.db"

    def bench(self, db: str = "real", create: bool = True) -> Workbench:
        with self._lock:
            if db not in self.benches:
                path = self.path_of(db)
                if db == "demo" and not create and not path.exists():
                    raise ApiError("the demo data has not been made yet: press 'Make the simulated world' in the Data Manager")
                self.benches[db] = Workbench(service=FundamentalsService(self.app.root, db_path=path))
            return self.benches[db]

    def _real_only(self, db: str, what: str) -> None:
        if db != "real":
            raise ApiError(f"{what} is for real data: switch to Real data (the demo database is a simulated world and never calls a provider)")

    def _guard(self, fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except FundamentalsError as error:
            raise ApiError(str(error)) from None

    def _ready(self, wb: Workbench) -> None:
        r = wb.readiness()
        if not r["ready"]:
            raise ApiError(r["problems"][0])

    def _job(self, label: str, work, stoppable: bool = False) -> dict:
        stop = threading.Event() if stoppable else None

        def run(progress):
            try:
                return work(progress, stop)
            except FundamentalsError as error:
                raise ApiError(str(error)) from None

        for old in [j for j in self.stops if j not in self.app.jobs or self.app.jobs[j].status not in ACTIVE]:
            self.stops.pop(old, None)
        started = self.app.submit_task(label, run, "fundamentals")
        if stop is not None:
            self.stops[started["job"]] = stop
        return started

    def _context(self, wb: Workbench, body: dict):
        return wb.context(**{k: body[k] for k in CONTEXT_FIELDS if body.get(k) is not None})

    # ------------------------------------------------------------------------------------------------------------ reading
    def status(self, q: dict | None = None) -> dict:
        """``GET /api/fund/status?db=``: everything the Data Manager shows: counts, the providers with their allowances, the key slots and what each has used (never a key), the policy on keys, readiness."""
        q = q or {}
        db = _db(q)
        out = self._guard(self.bench(db).status)               # reading a demo database that is not there yet makes an empty one: harmless, and the page shows what to do next
        out["database"] = db
        out["databases"] = {name: self._summary(name) for name in DBS}
        return out

    def _summary(self, db: str) -> dict:
        if not self.path_of(db).exists() and db not in self.benches:
            return {"exists": False, "companies": 0}
        counts = self.bench(db).store.counts()
        return {"exists": True, "companies": counts["companies"], "price_rows": counts["price_rows"], "alphas": counts["alphas"]}

    def overview(self, q: dict | None = None) -> dict:
        q = q or {}
        db = _db(q)
        return self._guard(self.bench(db, create=db == "real").overview)

    def catalogue(self, q: dict | None = None) -> dict:
        return self._guard(self.bench("real").catalogue)

    def companies(self, q: dict | None = None) -> dict:
        """``GET /api/fund/companies?db=&q=&country=&sector=&limit=&offset=``: the companies in this database. On the real one it also says whether the provider's directory can be searched now
        (``lookup``: a key is set, what a search and a first fetch cost, what is left today), so the page can offer the search only when it can work."""
        q = q or {}
        db = _db(q)
        out = self._guard(self.bench(db, create=db == "real").companies, q.get("q", ""), q.get("country", ""), q.get("sector", ""), q.get("limit", 50), q.get("offset", 0))
        if db == "real":
            out["lookup"] = self._lookup_state(self.bench(db))
        return out

    @staticmethod
    def _lookup_state(wb: Workbench) -> dict:
        svc = wb.service
        prov, pool = svc.provider("openfilings"), svc.pool("openfilings")
        ok, why = prov.available()
        return {"available": ok, "reason": why, "cost": prov.cost("search"), "fetch_cost": prov.cost("financials") + (prov.cost("profile") if prov.has_lookup else 0), "unit": pool.allow.unit,
                "remaining_today": pool.remaining_today() if ok else None}

    def company(self, q: dict | None = None) -> dict:
        q = q or {}
        db = _db(q)
        try:
            sid = int(q.get("id"))
        except (TypeError, ValueError):
            raise ApiError("say which company: ?id=<security id>") from None
        return self._guard(self.bench(db, create=db == "real").company, sid)

    def quality(self, q: dict | None = None) -> dict:
        q = q or {}
        db = _db(q)
        return self._guard(self.bench(db, create=db == "real").quality, _number(q.get("limit", 100), "limit", 1, 1000, True))

    def library(self, q: dict | None = None) -> dict:
        """``GET /api/fund/library?db=&status=&order=&limit=``: the alphas, best first, with the count of each status and of the graveyard."""
        q = q or {}
        db = _db(q)
        wb = self.bench(db, create=db == "real")
        status = q.get("status") or None
        if status is not None and status not in STATUSES:
            raise ApiError(f"status must be one of {', '.join(STATUSES)}")
        rows = self._guard(wb.library.list, status, q.get("order", "fitness"), _number(q.get("limit", 200), "limit", 1, 1000, True))
        keep = ("alpha_id", "name", "text", "status", "status_reason", "created_at", "updated_at", "universe", "rebalance_frequency", "forward_horizon", "ic", "icir", "sharpe", "sortino", "max_drawdown",
                "turnover", "oos_ic", "oos_sharpe", "complexity", "fitness", "q_value", "deflated_sharpe")
        return {"alphas": [{**{k: r.get(k) for k in keep}, "direction": (r.get("metrics") or {}).get("direction")} for r in rows], "counts": wb.library.counts(), "statuses": list(STATUSES)}

    def alpha(self, q: dict | None = None) -> dict:
        q = q or {}
        db = _db(q)
        return self._guard(self.bench(db, create=db == "real").alpha, str(q.get("id") or ""))

    def graveyard(self, q: dict | None = None) -> dict:
        """``GET /api/fund/graveyard``: every idea that failed, with the stage it failed at and why. Nothing here is ever deleted: it is the record of what was tried."""
        q = q or {}
        db = _db(q)
        wb = self.bench(db, create=db == "real")
        rows = self._guard(wb.library.graveyard, _number(q.get("limit", 200), "limit", 1, 2000, True), q.get("stage") or None)
        return {"rows": [{k: r.get(k) for k in ("id", "text", "reason_rejected", "stage", "ic", "oos_ic", "sharpe", "date_tested", "run_id")} for r in rows], "counts": wb.library.counts()}

    def runs(self, q: dict | None = None) -> dict:
        q = q or {}
        db = _db(q)
        return {"runs": Discovery(self.bench(db, create=db == "real").store).runs(_number(q.get("limit", 30), "limit", 1, 200, True))}

    def run_summary(self, q: dict | None = None) -> dict:
        q = q or {}
        db = _db(q)
        d = Discovery(self.bench(db, create=db == "real").store)
        out = self._guard(d.summary, str(q.get("id") or ""))
        out["candidates"] = [{k: c.get(k) for k in ("text", "status", "stage_reached", "reason")} | {"metrics": {m: c["metrics"].get(m) for m in ("ic", "icir", "oos_ic", "oos_t", "sharpe", "q_value", "deflated_sharpe", "turnover", "direction")}}
                             for c in d.candidates(out["run_id"], status=q.get("status") or None, limit=_number(q.get("limit", 100), "limit", 1, 1000, True))]
        return out

    def experiments(self, q: dict | None = None) -> dict:
        q = q or {}
        db = _db(q)
        return {"experiments": self.bench(db, create=db == "real").experiments(q.get("kind") or None, _number(q.get("limit", 20), "limit", 1, 200, True))}

    # ------------------------------------------------------------------------------------------------------------ small writes
    def add_companies(self, body: dict) -> dict:
        """``POST /api/fund/companies/add {symbols, market?}``: register companies in the security master. Costs nothing: only what was typed is stored; the first sync fills in the rest."""
        db = _db(body)
        self._real_only(db, "adding companies")
        wb = self.bench(db)
        symbols = _symbols(body.get("symbols"))
        market = body.get("market")
        if market is not None and not re.match(r"^[A-Za-z]{2,4}$", str(market)):
            raise ApiError("market is a short code such as us, jp or uk")
        ids = self._guard(wb.service.add_companies, "openfilings", symbols, None, market)
        rows = [wb.store.get_security(i) or {} for i in ids]
        return {"added": len(ids), "security_ids": ids, "symbols": symbols, "companies": [{"security_id": i, "ticker": r.get("ticker"), "name": r.get("company_name")} for i, r in zip(ids, rows)]}

    def search_companies(self, body: dict) -> dict:
        """``POST /api/fund/companies/search {q, kind?, market?, refresh?}``: look a company up in OpenFilings' directory by name, ticker or ISIN. It spends one credit of the day's allowance (the same
        search again is answered from what was saved, for nothing; ``refresh`` asks again) and registers nothing: adding a company and fetching its statements are separate steps."""
        db = _db(body)
        self._real_only(db, "searching OpenFilings")
        wb = self.bench(db)
        q = body.get("q")
        if not isinstance(q, str) or len(q.strip()) < 2 or len(q) > 120:
            raise ApiError("type at least two characters (up to 120) of a name, a ticker or an ISIN to search for")
        kind = str(body.get("kind") or "auto")
        if kind not in SEARCH_KINDS:
            raise ApiError(f"kind must be one of {', '.join(SEARCH_KINDS)}")
        market = body.get("market") or None
        if market is not None and not re.match(r"^[A-Za-z]{2,4}$", str(market)):
            raise ApiError("market is a short code such as us, jp or uk")
        return self._guard(wb.service.search_companies, "openfilings", q, kind, str(market) if market else None, 10, bool(body.get("refresh")))

    def sync_plan(self, body: dict) -> dict:
        """``POST /api/fund/sync/plan``: what a sync would fetch and what it would cost against the allowance. Nothing is spent."""
        db = _db(body)
        self._real_only(db, "planning a sync")
        wb = self.bench(db)
        provider = str(body.get("provider") or "openfilings")
        ids = _ids(body.get("security_ids"))
        periods = tuple(body.get("periods") or ("annual",))
        start_year = _number(body["start_year"], "the first year", 1990, 2100, True) if body.get("start_year") else None
        if not set(periods) <= {"annual", "quarter"}:
            raise ApiError("periods are annual and quarter")
        svc = wb.service
        plan = self._guard(svc.plan, provider, ids, periods, start_year)
        return {"plan": plan.summary(), "affordability": svc.affordability(provider, plan), "tasks": [{"security_id": t.security_id, "symbol": t.symbol, "reason": t.reason, "cost": t.cost} for t in plan.tasks[:200]]}

    def provider_test(self, body: dict) -> dict:
        """``POST /api/fund/provider/test {provider}``: one cheap call (one credit) to prove the key works and learn the plan, which sets the allowance the quota manager enforces."""
        db = _db(body)
        self._real_only(db, "testing a provider")
        return self._guard(self.bench(db).service.test_provider, str(body.get("provider") or "openfilings"))

    def price_symbol(self, body: dict) -> dict:
        db = _db(body)
        self._real_only(db, "setting a price symbol")
        try:
            sid = int(body.get("security_id"))
        except (TypeError, ValueError):
            raise ApiError("say which company (security_id)") from None
        return self._guard(self.bench(db).set_price_symbol, sid, str(body.get("symbol") or ""))

    def formula_check(self, body: dict) -> dict:
        """``POST /api/fund/formula/check {formula}``: parse a formula and say what it is; with the data already at hand (a study has built it) it also checks the types and the coverage."""
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        text = str(body.get("formula") or "")
        if not text.strip() or len(text) > 2000:
            raise ApiError("write a formula, up to 2000 characters")
        ctx = self._guard(wb.peek, **{k: body[k] for k in CONTEXT_FIELDS if body.get(k) is not None})
        return self._guard(wb.check_formula, text, ctx)

    def library_status(self, body: dict) -> dict:
        """``POST /api/fund/library/status {alpha_id, status, reason, force?}``: move an alpha along the allowed paths. A move the table does not allow needs ``force`` and a reason, and is recorded as forced."""
        wb = self.bench(_db(body), create=False)
        status = str(body.get("status") or "")
        return self._guard(wb.library.set_status, str(body.get("alpha_id") or ""), status, str(body.get("reason") or ""), bool(body.get("force")))

    def library_validate(self, body: dict) -> dict:
        """``POST /api/fund/library/validate {alpha_id, reason?}``: Candidate to Validated if the evidence holds after the correction for the number of ideas tried; ``reason`` approves it anyway, recorded as forced."""
        wb = self.bench(_db(body), create=False)
        reason = body.get("reason")
        return self._guard(wb.library.validate, str(body.get("alpha_id") or ""), str(reason) if reason else None)

    # ------------------------------------------------------------------------------------------------------------ jobs: the data
    def sync(self, body: dict) -> dict:
        """``POST /api/fund/sync (a job)``: fetch what is missing or stale. ``budget`` (required) is the most credits this run may spend; the run stops cleanly when it is reached, when the allowance is spent or on
        ``/api/fund/stop``. Statements cost credits: look at ``/api/fund/sync/plan`` first."""
        db = _db(body)
        self._real_only(db, "fetching statements")
        wb = self.bench(db)
        provider = str(body.get("provider") or "openfilings")
        if provider != "openfilings":
            raise ApiError("the real database is filled from OpenFilings (provider: openfilings)")
        if body.get("budget") in (None, ""):
            raise ApiError("say how many credits this run may spend (budget): fetching statements uses the account's allowance")
        budget = _number(body["budget"], "the budget", 1, MAX_BUDGET, True)
        symbols = _symbols(body["symbols"]) if body.get("symbols") else None
        ids = _ids(body.get("security_ids"))
        periods = tuple(body.get("periods") or ("annual",))
        if not set(periods) <= {"annual", "quarter"}:
            raise ApiError("periods are annual and quarter")
        start_year = _number(body["start_year"], "the first year", 1990, 2100, True) if body.get("start_year") else None
        market = body.get("market")

        def work(progress, stop):
            report = wb.service.sync(provider, symbols, ids, periods, start_year, budget, progress, stop.is_set, market)
            wb.forget()
            return report
        return self._job(f"fundamentals: fetch statements (up to {budget} credits)", work, True)

    def simulate(self, body: dict) -> dict:
        """``POST /api/fund/simulate (a job)``: make the demo database, a simulated world of companies with a few alphas planted in it. ``reset`` starts the file again."""
        if _db(body) != "demo":
            raise ApiError("the simulated world is made in the demo database (db: demo); the real database never holds simulated companies")
        n = _number(body.get("companies", 120), "the number of companies", 30, 400, True)
        first = _number(body.get("start_year", 2009), "the first year", 1995, 2020, True)
        last = _number(body.get("end_year", 2022), "the last year", first + 6, 2024, True)
        seed = _number(body.get("seed", 7), "the seed", 0, 1_000_000, True)
        reset = bool(body.get("reset"))
        planted = body.get("planted")
        if planted is not None:
            if not isinstance(planted, dict) or set(planted) - set(PLANTED):
                raise ApiError(f"planted is a dictionary of the strengths of {', '.join(PLANTED)}")
            planted = {k: _number(v, f"the planted strength of {k}", 0, 1 if k == "anticipation" else 0.05) for k, v in planted.items()}

        def work(progress, stop):
            with self._lock:
                if reset:
                    self.benches.pop("demo", None)
                    for suffix in ("", "-wal", "-shm"):
                        Path(str(self.path_of("demo")) + suffix).unlink(missing_ok=True)
            wb = self.bench("demo")
            report = wb.service.load_simulated(n, first, last, seed, progress, planted)
            wb.forget()
            return report
        return self._job(f"fundamentals: make the simulated world ({n} companies)", work)

    def load_prices(self, body: dict) -> dict:
        """``POST /api/fund/prices (a job)``: daily prices and exchange rates for the companies, from Yahoo Finance. ``security_ids`` limits it; ``refresh`` fetches the whole history again."""
        db = _db(body)
        self._real_only(db, "loading prices")
        wb = self.bench(db)
        ids = _ids(body.get("security_ids"))
        start = body.get("start") or price_module.DEFAULT_START

        def work(progress, stop):
            return wb.load_prices(ids, start, self.fetch, progress, stop.is_set, bool(body.get("refresh")), body.get("fx", True) is not False)
        return self._job("fundamentals: load prices", work, True)

    # ------------------------------------------------------------------------------------------------------------ jobs: research
    def scan(self, body: dict) -> dict:
        """``POST /api/fund/scan (a job)``: evaluate many factors (``names``; all by default) under one set of conditions and set them side by side."""
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        self._ready(wb)
        spec = self._guard(eval_spec, body.get("spec"))
        names = body.get("names")
        if names is not None and (not isinstance(names, list) or not all(isinstance(n, str) for n in names)):
            raise ApiError("names must be a list of factor names")

        def work(progress, stop):
            progress("building the research context")
            ctx = self._context(wb, body)
            return wb.scan(ctx, names, spec, progress, stop.is_set)
        return self._job(f"fundamentals: scan {len(names) if names else 'all'} factors", work, True)

    def evaluate(self, body: dict) -> dict:
        """``POST /api/fund/evaluate (a job)``: the full study of one library factor: information coefficient, quantiles, long-short portfolio, decay, countries, sectors, regimes."""
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        self._ready(wb)
        spec = self._guard(eval_spec, body.get("spec"))
        if not spec.factor:
            raise ApiError("name a factor to evaluate (spec.factor)")

        def work(progress, stop):
            progress("building the research context")
            ctx = self._context(wb, body)
            progress("evaluating " + spec.factor)
            return wb.evaluate(ctx, spec)
        return self._job(f"fundamentals: study {spec.factor}", work)

    def evaluate_formula(self, body: dict) -> dict:
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        self._ready(wb)
        spec = self._guard(eval_spec, {**(body.get("spec") or {}), "factor": None})
        text = str(body.get("formula") or "")
        if not text.strip() or len(text) > 2000:
            raise ApiError("write a formula, up to 2000 characters")

        def work(progress, stop):
            progress("building the research context")
            ctx = self._context(wb, body)
            progress("evaluating the formula")
            return wb.evaluate_formula(ctx, text, spec)
        return self._job("fundamentals: study a formula", work)

    def lookahead(self, body: dict) -> dict:
        """``POST /api/fund/lookahead (a job)``: the same factor study twice, once with every figure known from the day it was public and once as if known at its period end."""
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        self._ready(wb)
        spec = self._guard(eval_spec, body.get("spec"))
        if not spec.factor:
            raise ApiError("name a factor to audit (spec.factor)")

        def work(progress, stop):
            progress("running the study twice")
            return wb.lookahead(spec, body.get("universe"), body.get("freq", "M"), body.get("pit"))
        return self._job(f"fundamentals: look-ahead audit of {spec.factor}", work)

    def discover(self, body: dict) -> dict:
        """``POST /api/fund/discover (a job)``: search for alphas. ``{"spec": {...}}`` starts a run; ``{"run_id": ...}`` resumes one that was stopped or interrupted, exactly where it left off. Every formula
        tried is counted and kept, the best are tested once on dates they were never chosen on, and nothing is ever called validated."""
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        self._ready(wb)
        d = Discovery(wb.store)
        if body.get("run_id"):
            run_id = str(body["run_id"])
            record = self._guard(d.get_run, run_id)
            if record["status"] == "finished":
                raise ApiError("that run has finished: its locked test has been looked at, and looking again would spend it. Start a new run instead")
            spec = DiscoverySpec.from_dict(record["spec"])
        else:
            spec = self._guard(wb.discovery_spec, body.get("spec"))
            run_id = d.create(spec)

        def work(progress, stop):
            progress("building the research context")
            ctx = wb.context(spec.universe, spec.freq, spec.entry_lag)
            runner = Discovery(wb.store, ctx)
            return runner.run(run_id, lambda done, total, text: progress(f"{done} of {total} formulas: {text}"), stop.is_set)
        started = self._job(f"fundamentals: discovery run {run_id}", work, True)
        return {**started, "run_id": run_id}

    def monitor(self, body: dict) -> dict:
        """``POST /api/fund/monitor (a job)``: recompute alphas on the data now in the database and judge each Healthy, Weakening, Decaying or Broken."""
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        ids = body.get("alpha_ids")
        if ids is not None and (not isinstance(ids, list) or not all(isinstance(i, str) for i in ids)):
            raise ApiError("alpha_ids must be a list of alpha ids")
        return self._job("fundamentals: monitor the alphas", lambda progress, stop: wb.monitor(ids, progress, stop.is_set), True)

    def correlation(self, body: dict) -> dict:
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        self._ready(wb)
        parts, cut = body.get("parts"), body.get("cut", 0.5)

        def work(progress, stop):
            progress("building the research context")
            ctx = self._context(wb, body)
            progress("correlating the alphas")
            return wb.correlation(ctx, parts, cut)
        return self._job("fundamentals: correlation and clusters of alphas", work)

    def combine(self, body: dict) -> dict:
        """``POST /api/fund/combine (a job)``: put alphas together by a rule (equal, ic, icir, diversified, cluster) using only information that was known on each date."""
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        self._ready(wb)
        parts, method, horizon, options = body.get("parts"), str(body.get("method") or "icir"), body.get("horizon", "1m"), body.get("options")
        spec = self._guard(eval_spec, body.get("spec"))
        if options is not None and not isinstance(options, dict):
            raise ApiError("options must be a dictionary")

        def work(progress, stop):
            progress("building the research context")
            ctx = self._context(wb, body)
            progress(f"combining by {method}")
            return wb.combination(ctx, parts, method, horizon, options, spec)
        return self._job(f"fundamentals: combine alphas ({method})", work)

    def ml(self, body: dict) -> dict:
        """``POST /api/fund/ml (a job)``: fit a model walk-forward (no training row whose return was still unfolding) to predict returns from alphas, and compare it with the simple rules."""
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        self._ready(wb)
        parts, model, horizon, options = body.get("parts"), str(body.get("model") or "ridge"), body.get("horizon", "1m"), body.get("options")
        spec = self._guard(eval_spec, body.get("spec"))
        if options is not None and not isinstance(options, dict):
            raise ApiError("options must be a dictionary")

        def work(progress, stop):
            progress("building the research context")
            ctx = self._context(wb, body)
            progress(f"fitting {model} walk-forward")
            return wb.machine_learning(ctx, parts, model, horizon, options, spec)
        return self._job(f"fundamentals: {model} model on alphas", work)

    def backtest(self, body: dict) -> dict:
        """``POST /api/fund/backtest (a job)``: BACKTEST THIS ALPHA. ``signal`` describes it (a factor, a technical signal, a library alpha, a formula, a combination, a model, or a composite of these);
        ``portfolio`` says how it becomes a book. The weights go through the platform's backtest engine, net of costs."""
        db = _db(body)
        wb = self.bench(db, create=db == "real")
        self._ready(wb)
        signal, portfolio = body.get("signal"), body.get("portfolio")
        if signal is None:
            raise ApiError("say what to backtest (signal)")
        n_trials = _number(body["n_trials"], "the number of ideas tried", 1, 10_000_000, True) if body.get("n_trials") else None
        name = str(body["name"])[:80] if body.get("name") else None

        def work(progress, stop):
            progress("building the research context")
            ctx = self._context(wb, body)
            progress("building the book and running the engine")
            return wb.backtest(ctx, signal, portfolio, name, n_trials)
        return self._job("fundamentals: backtest an alpha", work)

    def stop(self, body: dict) -> dict:
        """``POST /api/fund/stop {"job": id}``: end a job after the step it is on; what is done is kept."""
        flag = self.stops.get(str(body.get("job") or ""))
        if flag is None:
            raise ApiError("that job is not running in this app")
        flag.set()
        return {"ok": True}


__all__ = ["FundamentalsApi", "DBS"]
