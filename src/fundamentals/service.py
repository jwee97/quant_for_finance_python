"""The database-first sync: ask the local database what it already has, ask a provider only for what is missing or new, store everything that comes back, and count every request.

The order of work for a company is the one the specification fixes: (1) look in the database; (2) decide whether what is there is complete and current; (3) request only the missing or new periods;
(4) keep the raw answer once; (5) normalise and store; (6) check the quality; and only then (7) do research on the local data. Nothing in the research reads a provider.

"Current" is judged from the calendar, not from a clock on the data: once the latest stored period is old enough that the next filing should exist (the period plus its length plus a typical filing
lag), the company is *stale* and is looked at again, at most once a week. Until then a company is *fresh* and costs nothing.

Money. A provider's calls are metered (OpenFilings: credits), so every plan states its cost before anything is spent, a run can be capped (``budget``), and a run stops cleanly the moment the
provider's allowance runs out; what is left waits for the allowance to reset. Registering a company costs nothing: only the ticker and market the user typed are stored, and the first sync fills in
the rest (a lookup for sector and identifiers, then the statements).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import quality
from .errors import FundamentalsError, InvalidKey, MissingKey, NotFound, PlanError, ProviderError, QuotaExhausted, SyncPending
from .keys import KeyPool
from .providers import PROVIDER_NAMES, make_provider
from .providers.base import FundamentalDataProvider
from .store import DEFAULT_PATH, FundamentalsStore
from .usage import UsageLog

CADENCE = {"annual": 365, "quarter": 91}          # days between periods
FILING_LAG = {"annual": 150, "quarter": 75}       # a generous typical lag, so "stale" means a filing really is overdue
RECHECK_DAYS = 7                                   # never look again sooner than this
NOT_FOUND_RETRY_DAYS = 60


@dataclass
class SyncTask:
    security_id: int
    symbol: str
    provider: str
    period: str
    reason: str                       # missing | stale | history
    cost: int                         # in the provider's unit (credits for OpenFilings)
    profile: bool = False             # the company has no sector or identifiers yet: look it up first


@dataclass
class SyncPlan:
    tasks: list[SyncTask] = field(default_factory=list)
    fresh: int = 0
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unit: str = "requests"

    @property
    def cost(self) -> int:
        return sum(t.cost for t in self.tasks)

    def summary(self) -> dict:
        reasons: dict[str, int] = {}
        for t in self.tasks:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
        return {"to_fetch": len(self.tasks), "fresh": self.fresh, "reasons": reasons, "cost": self.cost, "unit": self.unit, "skipped": self.skipped[:20], "notes": self.notes}


class FundamentalsService:
    def __init__(self, root: str | Path = ".", db_path: str | Path | None = None, providers: dict | None = None, clock=None):
        self.root = Path(root)
        self.store = FundamentalsStore(Path(db_path) if db_path else self.root / DEFAULT_PATH)
        self.usage = UsageLog(self.store.db, clock)
        self._providers: dict[str, FundamentalDataProvider] = dict(providers or {})
        self._pools: dict[str, KeyPool] = {}

    # ------------------------------------------------------------------------------------------------------------ providers and keys
    def plan_of(self, name: str) -> str | None:
        """The plan the account last reported when tested (stored, so the quota manager enforces the right allowance after a restart)."""
        return self.store.get_meta(f"plan:{name}")

    def pool(self, name: str) -> KeyPool:
        if name not in self._pools:
            if name == "openfilings":
                self._pools[name] = KeyPool.from_environment(name, self.usage, self.store.salt(), self.root, plan=self.plan_of(name))
            else:
                self._pools[name] = KeyPool.without_key(name, self.usage)
        return self._pools[name]

    def provider(self, name: str) -> FundamentalDataProvider:
        if name not in self._providers:
            if name == "openfilings":
                self._providers[name] = make_provider("openfilings", pool=self.pool("openfilings"))
            elif name == "simulated":
                self._providers[name] = make_provider("simulated")
            else:
                raise FundamentalsError(f"unknown provider {name!r}; choose from {', '.join(PROVIDER_NAMES)}")
        return self._providers[name]

    def register(self, name: str, provider: FundamentalDataProvider) -> None:
        """Use ``provider`` under ``name``; a provider that carries a key pool brings it along, so its calls and the quota manager see the same keys."""
        self._providers[name] = provider
        pool = getattr(provider, "pool", None)
        if pool is not None:
            self._pools[name] = pool

    def _today(self) -> pd.Timestamp:
        return pd.Timestamp(self.usage.today())

    def _call(self, name: str, fn, endpoint: str, symbol: str = ""):
        """Run a provider call through the budget. A keyed provider does its own accounting (it knows what each call costs); any other is counted here, one unit a call."""
        prov = self.provider(name)
        if prov.keyed:
            return fn()
        return self.pool(name).call(lambda _key: fn(), endpoint=endpoint, symbol=symbol)

    # ------------------------------------------------------------------------------------------------------------ planning
    def _state_key(self, period: str) -> str:
        return f"financials:{period}"

    def reason_to_sync(self, security_id: int, provider: str, period: str, start_year: int | None = None, today: pd.Timestamp | None = None) -> str | None:
        """``missing``, ``history`` (older periods were asked for and never looked for), ``stale`` (a newer filing should exist) or ``None`` (fresh)."""
        today = today or self._today()
        state = self.store.get_sync(security_id, provider, self._state_key(period))
        checked = pd.Timestamp(state["last_checked"]) if state and state.get("last_checked") else None
        if state and state.get("status") == "not_found":
            return None if checked is not None and (today - checked).days < NOT_FOUND_RETRY_DAYS else "missing"
        if state and state.get("status") == "pending":                                    # the provider is still collecting the filings: ask again, but not within the same day
            return None if checked is not None and (today - checked).days < 1 else "missing"
        if not state or not state.get("last_success"):
            return "missing"
        earliest = state.get("earliest_period_end")
        if start_year and earliest and int(str(earliest)[:4]) > start_year and not state.get("history_exhausted"):
            return "history"
        last_end = pd.Timestamp(state["last_period_end"]) if state.get("last_period_end") else None
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
        """What a sync would fetch and what it would cost, without fetching anything."""
        prov = self.provider(provider)
        frame = self.store.securities()
        if security_ids is not None:
            frame = frame[frame["security_id"].isin(security_ids)]
        known = self._symbols(provider)
        unit = self.pool(provider).allow.unit
        out = SyncPlan(unit=unit)
        usable = []
        for period in periods:
            if period not in CADENCE:
                raise FundamentalsError(f"period must be one of {sorted(CADENCE)}")
            if period in prov.periods_available:
                usable.append(period)
            else:
                out.notes.append(f"{provider} has no {period} statements in this application, so none are planned")
        for row in frame.itertuples():
            symbol = known.get(int(row.security_id))
            if symbol is None:
                out.skipped.append(f"{row.ticker}: {provider} does not know this company yet")
                continue
            needs_profile = prov.has_lookup and not (isinstance(row.sector, str) and row.sector) and not (isinstance(row.industry, str) and row.industry)
            first = True
            for period in usable:
                reason = self.reason_to_sync(int(row.security_id), provider, period, start_year, today)
                if reason is None:
                    out.fresh += 1
                    continue
                lookup = needs_profile and first
                out.tasks.append(SyncTask(int(row.security_id), symbol, provider, period, reason, prov.cost("financials", period) + (prov.cost("profile") if lookup else 0), lookup))
                first = False
        return out

    def affordability(self, provider: str, plan: SyncPlan) -> dict:
        """How the plan sits against what the key in use may still spend: how many tasks fit today, and how many days the rest needs at a full allowance."""
        pool = self.pool(provider)
        left, daily = pool.remaining_today(), pool.allow.daily
        fit, spent = 0, 0
        for task in plan.tasks:
            if left is not None and spent + task.cost > left:
                break
            spent += task.cost
            fit += 1
        days = None
        if daily and plan.tasks:
            days = max(math.ceil(plan.cost / daily), 1)
        return {"unit": plan.unit, "cost": plan.cost, "remaining_today": left, "remaining_month": pool.remaining_month(), "fit_today": fit, "tasks": len(plan.tasks), "days_needed": days,
                "monthly_limit": pool.allow.monthly, "daily_limit": daily}

    # ------------------------------------------------------------------------------------------------------------ doing the work
    def add_companies(self, provider: str, symbols: list[str], progress=None, market: str | None = None) -> list[int]:
        """Make sure each symbol is in the security master. Costs nothing: only the ticker and market the user typed are stored; the first sync fills in the rest."""
        prov = self.provider(provider)
        ids = []
        for symbol in symbols:
            if not str(symbol).strip():
                continue
            normal = prov.normalise_symbol(symbol, market)
            existing = self._lookup(provider, normal)
            if existing is not None:
                ids.append(existing)
                continue
            profile = prov.offline_profile(normal, market)
            ids.append(self.store.upsert_security(profile))
        if progress:
            progress(f"{len(ids)} companies registered")
        return ids

    def _lookup(self, provider: str, symbol: str) -> int | None:
        with self.store.db.transaction() as conn:
            row = conn.execute("SELECT security_id FROM security_identifiers WHERE provider = ? AND id_type = 'symbol' AND provider_id = ?", (provider, symbol)).fetchone()
        return int(row[0]) if row else None

    def sync(self, provider: str, symbols: list[str] | None = None, security_ids: list[int] | None = None, periods: tuple = ("annual",), start_year: int | None = None,
             budget: int | None = None, progress=None, stop=None, market: str | None = None) -> dict:
        """Fetch what is missing or stale for ``symbols`` (new companies are registered first) or for ``security_ids``.

        Stops cleanly when ``budget`` (in the provider's unit, for this run), the key's allowance or ``stop()`` says so; what is left is planned again next time and costs nothing until then.
        """
        say = progress or (lambda text: None)
        prov = self.provider(provider)
        ok, why = prov.available()
        if not ok:
            raise MissingKey(why)
        unit = self.pool(provider).allow.unit
        report = {"provider": provider, "unit": unit, "fetched": 0, "fresh": 0, "failed": [], "not_found": [], "pending": [], "cost": 0, "stopped": "", "periods": 0, "vintages": 0,
                  "unmapped": [], "notes": []}
        ids = list(security_ids or [])
        if symbols:
            ids += self.add_companies(provider, symbols, say, market)
        plan = self.plan(provider, ids if (symbols or security_ids) else None, periods, start_year)          # nothing named means everything in the database
        report["fresh"], report["notes"] = plan.fresh, list(plan.notes)
        spent = 0
        for n, task in enumerate(plan.tasks):
            if stop is not None and stop():
                report["stopped"] = "stopped on request"
                break
            if budget is not None and spent + task.cost > budget:
                report["stopped"] = f"reached the budget of {budget} {unit} for this run; {len(plan.tasks) - n} companies remain"
                break
            say(f"{task.symbol} ({n + 1} of {len(plan.tasks)}): {task.reason}, {task.period}, up to {task.cost} {unit}")
            try:
                self.pool(provider).ensure(task.cost)                          # the lookup and the statements are started together or not at all
                done = self._fetch(prov, task, start_year)
            except QuotaExhausted as error:
                report["stopped"] = f"{error}; {len(plan.tasks) - n} companies remain"
                break
            except InvalidKey as error:
                report["stopped"] = str(error)
                break
            except NotFound:
                self.store.set_sync(task.security_id, provider, self._state_key(task.period), last_checked=self.usage.today(), status="not_found", error="not found")
                report["not_found"].append(task.symbol)
                continue
            except SyncPending as error:
                self.store.set_sync(task.security_id, provider, self._state_key(task.period), last_checked=self.usage.today(), status="pending", error=str(error)[:200])
                report["pending"].append(task.symbol)
                continue
            except (PlanError, ProviderError) as error:
                self.store.set_sync(task.security_id, provider, self._state_key(task.period), last_checked=self.usage.today(), status="error", error=str(error)[:200])
                report["failed"].append({"symbol": task.symbol, "error": str(error)[:200]})
                if isinstance(error, PlanError) and len([f for f in report["failed"] if "plan" in f["error"].lower()]) >= 5:
                    report["stopped"] = "five companies in a row are outside the provider's plan; stopping"
                    break
                continue
            spent += done["cost"]
            report["fetched"] += 1
            report["periods"] += done["periods"]
            report["vintages"] += done["vintages"]
            report["unmapped"] = sorted(set(report["unmapped"]) | set(done["unmapped"]))
        report["cost"] = spent
        return report

    def _fetch(self, prov: FundamentalDataProvider, task: SyncTask, start_year: int | None) -> dict:
        label = "quarter" if task.period == "quarter" else "annual"
        incremental = task.reason == "stale"                                  # a company that is only due its newest filing costs the same but returns a small answer
        newest = (6 if label == "quarter" else 3) if incremental else None
        cost = 0
        if task.profile:
            profile = self._call(task.provider, lambda: prov.get_company(task.symbol), "lookup", task.symbol)
            cost += prov.cost("profile")
            self.store.enrich_security(task.security_id, profile)
        result = self._call(task.provider, lambda: prov.get_financials(task.symbol, start_year=None if incremental else start_year, period=label, limit=newest),
                            f"financials:{label}", task.symbol)
        cost += result.cost
        self.store.save_raw(task.provider, task.security_id, f"financials:{label}", result.request, result.raw)
        if result.profile is not None:
            self.store.enrich_security(task.security_id, result.profile)
        counts = self.store.store_financials(task.security_id, task.provider, result.periods)
        quality.check_security(self.store, task.security_id)
        summary = self.store.period_summary(task.security_id, task.provider)
        old = self.store.get_sync(task.security_id, task.provider, self._state_key(task.period)) or {}
        earliest = min([d for d in (summary.get("first_end"), old.get("earliest_period_end")) if d], default=None)
        self.store.set_sync(task.security_id, task.provider, self._state_key(task.period), last_checked=self.usage.today(), last_success=self.usage.today(),
                            last_period_end=summary.get("last_end"), earliest_period_end=earliest, history_exhausted=1 if (result.exhausted or old.get("history_exhausted")) else 0,
                            n_periods=int(summary.get("n") or 0), status="ok", error=None)
        return {"cost": cost, "requests": result.requests, "periods": counts["periods"], "vintages": counts["vintages"], "unmapped": counts["unmapped"]}

    # ------------------------------------------------------------------------------------------------------------ status for the Data Manager
    def test_provider(self, name: str) -> dict:
        """Check a provider now: one credit for a keyed one (the call that lists the markets on the account's plan), which also tells the quota manager which plan the account is on."""
        prov = self.provider(name)
        ok, why = prov.available()
        info = {**prov.describe(), "available": ok, "reason": why}
        if not ok or not hasattr(prov, "test"):
            return info
        try:
            result = prov.test()
        except QuotaExhausted as error:
            return {**info, "ok": False, "error": str(error)}
        except ProviderError as error:
            return {**info, "ok": False, "error": str(error)}
        if result.get("plan"):
            self.store.set_meta(f"plan:{name}", result["plan"])
            self.pool(name).plan = result["plan"]
        return {**info, "ok": True, **result}

    def data_status(self) -> dict:
        """Everything the Data Manager's DATA STATUS panel shows, from the database and the usage log (no provider is contacted)."""
        counts = self.store.counts()
        providers = []
        for name in PROVIDER_NAMES:
            prov = self.provider(name)
            pool = self.pool(name)
            ok, why = prov.available()
            keys = pool.slots()
            plan = self.plan_of(name)
            usage = self.usage.summary(name, keys, plan)
            allow = pool.allow
            providers.append({**prov.describe(), "available": ok, "reason": why, "unit": allow.unit, "plan": plan, "allowance_source": allow.source, "requests_today": usage.requests,
                              "successful": usage.successful, "failed": usage.failed, "used_today": usage.used_today, "used_month": usage.used_month, "daily_limit": allow.daily,
                              "monthly_limit": allow.monthly, "remaining_today": pool.remaining_today(), "remaining_month": pool.remaining_month(),
                              "resets": {"daily": self.usage.next_day(), "monthly": self.usage.next_month()}, "cost_per_company": prov.cost("financials") + (prov.cost("profile") if prov.has_lookup else 0),
                              "keys": [{k: v for k, v in s.items() if k != "key_id"} for s in keys], "key_notes": pool.notes, "recent": self.usage.recent(name, 12)})
        today = self._today()
        stale = 0
        sources = {int(r.security_id): str(r.source) for r in self.store.securities().itertuples() if r.source}
        for sid, source in sources.items():
            if self.reason_to_sync(sid, source, "annual", None, today) in ("stale", "missing"):
                stale += 1
        return {"counts": counts, "providers": providers, "companies_cached": counts["with_financials"], "companies_stale": stale, "last_update": self.usage.last_update(),
                "database_bytes": self.store.db.size_bytes(), "quality": self.store.quality_counts()}

    # ------------------------------------------------------------------------------------------------------------ the demonstration world
    def load_simulated(self, n_companies: int = 120, start_year: int = 2009, end_year: int = 2022, seed: int = 7, progress=None, planted: dict | None = None) -> dict:
        """Fill the database from the simulated world through the same path a real provider takes: security master, raw payloads, statements, quality, daily prices and exchange rates.

        ``planted`` overrides the strength of the alphas hidden in the world (see :data:`providers.simulated.PLANTED`); a small test world needs stronger ones to be conclusive."""
        from .providers.simulated import PLANTED, SimulatedProvider
        say = progress or (lambda text: None)
        with self.store.db.transaction() as conn:
            real = int(conn.execute("SELECT COUNT(*) FROM securities WHERE source IS NOT NULL AND source <> 'simulated'").fetchone()[0])
        if real:
            raise FundamentalsError(f"this database holds {real} real companies: a simulated world would be mixed into studies of them. Make it in a separate database (the Fundamentals tab keeps one for the demo)")
        extra = {"planted": {**PLANTED, **planted}} if planted else {}
        prov = SimulatedProvider(n_companies=n_companies, start_year=start_year, end_year=end_year, seed=seed, **extra)
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
