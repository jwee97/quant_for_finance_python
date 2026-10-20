"""The database-first sync: ask the local database what it already has, ask a provider only for what is missing or new, store everything that comes back, and count every request.

The order of work for a company is the one the specification fixes: (1) look in the database; (2) decide whether what is there is complete and current; (3) request only the missing or new periods;
(4) keep the raw answer once; (5) normalise and store; (6) check the quality; and only then (7) do research on the local data. Nothing in the research reads a provider.

"Current" is judged from the calendar, not from a clock on the data: once the latest stored period is old enough that the next filing should exist (the period plus its length plus a typical filing
lag), the company is *stale* and is looked at again, at most once a week. Until then a company is *fresh* and costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import quality
from .errors import FundamentalsError, MissingKey, NotFound, PlanError, ProviderError, QuotaExhausted
from .keys import KeyPool
from .providers import PROVIDER_NAMES, make_provider
from .providers.base import FundamentalDataProvider
from .store import DEFAULT_PATH, FundamentalsStore
from .usage import LIMIT_KINDS, UsageLog

CADENCE = {"annual": 365, "quarter": 91}          # days between periods
FILING_LAG = {"annual": 150, "quarter": 75}       # a generous typical lag, so "stale" means a filing really is overdue
RECHECK_DAYS = 7                                   # never look again sooner than this
NOT_FOUND_RETRY_DAYS = 60
REQUESTS_PER_TASK = {"fmp": 3, "openfilings": 1, "simulated": 1}


def _today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).strftime("%Y-%m-%d"))


@dataclass
class SyncTask:
    security_id: int
    symbol: str
    provider: str
    period: str
    reason: str                       # missing | stale | history
    requests: int


@dataclass
class SyncPlan:
    tasks: list[SyncTask] = field(default_factory=list)
    fresh: int = 0
    skipped: list[str] = field(default_factory=list)

    @property
    def requests(self) -> int:
        return sum(t.requests for t in self.tasks)

    def summary(self) -> dict:
        reasons: dict[str, int] = {}
        for t in self.tasks:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
        return {"to_fetch": len(self.tasks), "fresh": self.fresh, "reasons": reasons, "requests": self.requests, "skipped": self.skipped[:20]}


class FundamentalsService:
    def __init__(self, root: str | Path = ".", db_path: str | Path | None = None, providers: dict | None = None):
        self.root = Path(root)
        self.store = FundamentalsStore(Path(db_path) if db_path else self.root / DEFAULT_PATH)
        self.usage = UsageLog(self.store.db)
        self._providers: dict[str, FundamentalDataProvider] = dict(providers or {})
        self._pools: dict[str, KeyPool] = {}

    # ------------------------------------------------------------------------------------------------------------ providers and keys
    def pool(self, name: str) -> KeyPool:
        if name not in self._pools:
            if name == "fmp":
                self._pools[name] = KeyPool.from_environment(name, self.usage, self.store.salt(), self.root)
            else:
                self._pools[name] = KeyPool.without_key(name, self.usage)
        return self._pools[name]

    def provider(self, name: str) -> FundamentalDataProvider:
        if name not in self._providers:
            if name == "fmp":
                self._providers[name] = make_provider("fmp", pool=self.pool("fmp"))
            elif name == "openfilings":
                self._providers[name] = make_provider("openfilings")
            elif name == "simulated":
                self._providers[name] = make_provider("simulated")
            else:
                raise FundamentalsError(f"unknown provider {name!r}; choose from {', '.join(PROVIDER_NAMES)}")
        return self._providers[name]

    def register(self, name: str, provider: FundamentalDataProvider) -> None:
        self._providers[name] = provider

    def _call(self, name: str, fn, endpoint: str, symbol: str = ""):
        """Run a provider call that does not do its own keyed accounting (every keyless provider) through the budget, logging the request."""
        prov = self.provider(name)
        if prov.keyed:
            return fn()
        return self.pool(name).call(lambda _key: fn(), endpoint=endpoint, symbol=symbol)

    # ------------------------------------------------------------------------------------------------------------ planning
    def _state_key(self, period: str) -> str:
        return f"financials:{period}"

    def reason_to_sync(self, security_id: int, provider: str, period: str, start_year: int | None = None, today: pd.Timestamp | None = None) -> str | None:
        """``missing``, ``history`` (older periods were asked for and never looked for), ``stale`` (a newer filing should exist) or ``None`` (fresh)."""
        today = today or _today()
        state = self.store.get_sync(security_id, provider, self._state_key(period))
        if state and state.get("status") == "not_found":
            checked = pd.Timestamp(state["last_checked"]) if state.get("last_checked") else None
            return None if checked is not None and (today - checked).days < NOT_FOUND_RETRY_DAYS else "missing"
        if not state or not state.get("last_success"):
            return "missing"
        earliest = state.get("earliest_period_end")
        if start_year and earliest and int(str(earliest)[:4]) > start_year and not state.get("history_exhausted"):
            return "history"
        last_end = pd.Timestamp(state["last_period_end"]) if state.get("last_period_end") else None
        checked = pd.Timestamp(state["last_checked"]) if state.get("last_checked") else None
        if last_end is not None and today >= last_end + pd.Timedelta(days=CADENCE[period] + FILING_LAG[period]):
            if checked is None or (today - checked).days >= RECHECK_DAYS:
                return "stale"
        return None

    def _symbols(self, provider: str) -> dict[int, str]:
        """``security_id -> the symbol this provider knows the company by`` (a company another provider added may not be known to this one)."""
        with self.store.db.transaction() as conn:
            rows = conn.execute("SELECT security_id, provider_id FROM security_identifiers WHERE provider = ? AND id_type = 'symbol'", (provider,)).fetchall()
        return {int(r[0]): str(r[1]) for r in rows}

    def plan(self, provider: str, security_ids: list[int] | None = None, periods: tuple = ("annual",), start_year: int | None = None, today: pd.Timestamp | None = None) -> SyncPlan:
        """What a sync would fetch, without fetching anything."""
        frame = self.store.securities()
        if security_ids is not None:
            frame = frame[frame["security_id"].isin(security_ids)]
        known = self._symbols(provider)
        out = SyncPlan()
        for row in frame.itertuples():
            symbol = known.get(int(row.security_id))
            if symbol is None:
                out.skipped.append(f"{row.ticker}: {provider} does not know this company yet")
                continue
            for period in periods:
                if period not in CADENCE:
                    raise FundamentalsError(f"period must be one of {sorted(CADENCE)}")
                reason = self.reason_to_sync(int(row.security_id), provider, period, start_year, today)
                if reason is None:
                    out.fresh += 1
                else:
                    out.tasks.append(SyncTask(int(row.security_id), symbol, provider, period, reason, REQUESTS_PER_TASK.get(provider, 1)))
        return out

    # ------------------------------------------------------------------------------------------------------------ doing the work
    def add_companies(self, provider: str, symbols: list[str], progress=None) -> list[int]:
        """Make sure each symbol is in the security master, asking the provider about only the ones the database has never seen."""
        prov = self.provider(provider)
        ids = []
        for i, symbol in enumerate(symbols):
            existing = self._lookup(provider, symbol)
            if existing is not None:
                ids.append(existing)
                continue
            if progress:
                progress(f"looking up {symbol} ({i + 1} of {len(symbols)})")
            profile = self._call(provider, lambda s=symbol: prov.get_company(s), "profile", symbol)
            ids.append(self.store.upsert_security(profile))
        return ids

    def _lookup(self, provider: str, symbol: str) -> int | None:
        with self.store.db.transaction() as conn:
            row = conn.execute("SELECT security_id FROM security_identifiers WHERE provider = ? AND id_type = 'symbol' AND provider_id = ?", (provider, symbol)).fetchone()
        return int(row[0]) if row else None

    def sync(self, provider: str, symbols: list[str] | None = None, security_ids: list[int] | None = None, periods: tuple = ("annual",), start_year: int | None = None,
             max_requests: int | None = None, progress=None, stop=None) -> dict:
        """Fetch what is missing or stale for ``symbols`` (new companies are added first) or for ``security_ids``. Stops cleanly when the budget, the key allowance or ``max_requests`` runs out."""
        say = progress or (lambda text: None)
        prov = self.provider(provider)
        ok, why = prov.available()
        if not ok:
            raise MissingKey(why)
        report = {"provider": provider, "fetched": 0, "fresh": 0, "failed": [], "not_found": [], "requests": 0, "stopped": "", "periods": 0, "vintages": 0, "unmapped": []}
        ids = list(security_ids or [])
        if symbols:
            try:
                ids += self.add_companies(provider, symbols, say)
            except QuotaExhausted as error:
                report["stopped"] = str(error)
        plan = self.plan(provider, ids if (symbols or security_ids) else None, periods, start_year)          # nothing named means everything in the database
        report["fresh"] = plan.fresh
        spent = 0
        for n, task in enumerate(plan.tasks):
            if stop is not None and stop():
                report["stopped"] = "stopped on request"
                break
            if max_requests is not None and spent + task.requests > max_requests:
                report["stopped"] = f"reached the limit of {max_requests} requests for this run; {len(plan.tasks) - n} companies remain"
                break
            say(f"{task.symbol} ({n + 1} of {len(plan.tasks)}): {task.reason}, {task.period}")
            try:
                done = self._fetch(prov, task, start_year)
            except QuotaExhausted as error:
                report["stopped"] = f"{error}; {len(plan.tasks) - n} companies remain for tomorrow"
                break
            except NotFound:
                self.store.set_sync(task.security_id, provider, self._state_key(task.period), last_checked=_today().strftime("%Y-%m-%d"), status="not_found", error="not found")
                report["not_found"].append(task.symbol)
                continue
            except (PlanError, ProviderError) as error:
                self.store.set_sync(task.security_id, provider, self._state_key(task.period), last_checked=_today().strftime("%Y-%m-%d"), status="error", error=str(error)[:200])
                report["failed"].append({"symbol": task.symbol, "error": str(error)[:200]})
                if isinstance(error, PlanError) and len([f for f in report["failed"] if "plan" in f["error"].lower()]) >= 5:
                    report["stopped"] = "five companies in a row are outside the provider's plan; stopping"
                    break
                continue
            spent += done["requests"]
            report["fetched"] += 1
            report["periods"] += done["periods"]
            report["vintages"] += done["vintages"]
            report["unmapped"] = sorted(set(report["unmapped"]) | set(done["unmapped"]))
        report["requests"] = spent
        return report

    def _fetch(self, prov: FundamentalDataProvider, task: SyncTask, start_year: int | None) -> dict:
        label = "quarter" if task.period == "quarter" else "annual"
        incremental = task.reason == "stale"                                  # a company that is only due its newest filing costs the same requests but a small answer
        newest = (6 if label == "quarter" else 3) if incremental else None
        result = self._call(task.provider, lambda: prov.get_financials(task.symbol, start_year=None if incremental else start_year, period=label, limit=newest),
                            f"financials:{label}", task.symbol)
        self.store.save_raw(task.provider, task.security_id, f"financials:{label}", result.request, result.raw)
        counts = self.store.store_financials(task.security_id, task.provider, result.periods)
        quality.check_security(self.store, task.security_id)
        summary = self.store.period_summary(task.security_id, task.provider)
        old = self.store.get_sync(task.security_id, task.provider, self._state_key(task.period)) or {}
        earliest = min([d for d in (summary.get("first_end"), old.get("earliest_period_end")) if d], default=None)
        self.store.set_sync(task.security_id, task.provider, self._state_key(task.period), last_checked=_today().strftime("%Y-%m-%d"), last_success=_today().strftime("%Y-%m-%d"),
                            last_period_end=summary.get("last_end"), earliest_period_end=earliest, history_exhausted=1 if (result.exhausted or old.get("history_exhausted")) else 0,
                            n_periods=int(summary.get("n") or 0), status="ok", error=None)
        return {"requests": result.requests, "periods": counts["periods"], "vintages": counts["vintages"], "unmapped": counts["unmapped"]}

    # ------------------------------------------------------------------------------------------------------------ status for the Data Manager
    def test_provider(self, name: str) -> dict:
        """Check a provider now (OpenFilings runs a subprocess, so this is only done when asked) and say what is wrong if it is not usable."""
        prov = self.provider(name)
        if hasattr(prov, "_checked"):
            prov._checked = None
        ok, why = prov.available()
        return {**prov.describe(), "available": ok, "reason": why}

    def data_status(self) -> dict:
        """Everything the Data Manager's DATA STATUS panel shows, from the database and the usage log (no provider is contacted)."""
        counts = self.store.counts()
        providers = []
        for name in PROVIDER_NAMES:
            prov = self.provider(name)
            pool = self.pool(name)
            if name == "openfilings" and getattr(prov, "_checked", None) is None:                  # checking it runs a subprocess: only when asked
                ok, why = False, "not checked yet: use Test on the Data Manager page"
            else:
                ok, why = prov.available()
            keys = pool.slots()
            usage = self.usage.summary(name, keys)
            providers.append({**prov.describe(), "available": ok, "reason": why, "requests_today": usage.requests, "successful": usage.successful, "failed": usage.failed,
                              "limit": usage.limit, "limit_kind": LIMIT_KINDS.get(name, ""), "keys": [{k: v for k, v in s.items() if k != "key_id"} for s in keys],
                              "key_notes": pool.notes, "remaining_today": pool.remaining_today()})
        today = _today()
        stale = 0
        sources = {int(r.security_id): str(r.source) for r in self.store.securities().itertuples() if r.source}
        for sid, source in sources.items():
            if self.reason_to_sync(sid, source, "annual", None, today) in ("stale", "missing"):
                stale += 1
        return {"counts": counts, "providers": providers, "companies_cached": counts["with_financials"], "companies_stale": stale, "last_update": self.usage.last_update(),
                "database_bytes": self.store.db.size_bytes(), "quality": self.store.quality_counts()}

    # ------------------------------------------------------------------------------------------------------------ the demonstration world
    def load_simulated(self, n_companies: int = 120, start_year: int = 2009, end_year: int = 2022, seed: int = 7, progress=None) -> dict:
        """Fill the database from the simulated world through the same path a real provider takes: security master, raw payloads, statements, quality, daily prices and exchange rates."""
        from .providers.simulated import SimulatedProvider
        say = progress or (lambda text: None)
        prov = SimulatedProvider(n_companies=n_companies, start_year=start_year, end_year=end_year, seed=seed)
        self.register("simulated", prov)
        self._pools.pop("simulated", None)
        say("building the simulated companies")
        symbols = prov.symbols()
        ids = self.add_companies("simulated", symbols, None)
        say(f"storing statements for {len(ids)} companies")
        report = self.sync("simulated", security_ids=ids, periods=("annual", "quarter"), start_year=None, progress=None)
        profiles = {s: i for s, i in zip(symbols, ids)}
        say("storing daily prices")
        prices = prov.prices()
        prices["security_id"] = prices["ticker"].map(profiles)
        self.store.store_prices(prices[["security_id", "date", "adj_close", "close", "volume"]])
        for sym, sid in profiles.items():
            if not prov.get_company(sym).active:
                last = prices.loc[prices["ticker"] == sym, "date"].max()
                self.store.set_active(sid, False, str(pd.Timestamp(last).date()))
        fx = prov.fx()
        self.store.store_fx(fx, "simulated")
        self.store.set_meta("simulated_world", f"{n_companies}:{start_year}:{end_year}:{seed}")
        report["companies"] = len(ids)
        report["price_rows"] = int(len(prices))
        return report
