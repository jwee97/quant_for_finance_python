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

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from . import coverage, quality
from .errors import FundamentalsError, InvalidKey, MissingKey, NotFound, PlanError, ProviderError, QuotaExhausted, SyncPending
from .keys import KeyPool
from .providers import PROVIDER_NAMES, make_provider
from .providers.base import FundamentalDataProvider
from .store import DEFAULT_PATH, CompanyProfile, FundamentalsStore
from .usage import UsageLog

CADENCE = {"annual": 365, "quarter": 91}          # days between periods
FILING_LAG = {"annual": 150, "quarter": 75}       # a generous typical lag, so "stale" means a filing really is overdue
RECHECK_DAYS = 7                                   # never look again sooner than this
NOT_FOUND_RETRY_DAYS = 60
SEARCH_KEEP_DAYS = 30                              # a saved search answer is used again for this long (a company's listing hardly ever changes) ...
SEARCH_EMPTY_KEEP_DAYS = 1                         # ... and one that found nothing for a day, so a second press does not spend another credit
KIND_WORDS = {"name": "a company name", "symbol": "a ticker", "isin": "an ISIN", "lei": "a LEI"}
PROVIDER_WORDS = {"openfilings": "OpenFilings", "simulated": "the simulated world"}
NO_QUARTERS = {"openfilings": "OpenFilings quarterly and half-year figures cannot be requested yet: its documentation does not say how to ask for them, and a guess would cost 10 credits a try."}
_PROFILE_TEXT = ("provider_symbol", "ticker", "name", "exchange", "market", "country", "currency", "sector", "industry", "isin", "lei")
_PROFILE_EXTRA = ("also_listed", "score", "registry", "hint", "primary", "has_financials")


def _year_of(statement) -> int:
    """The fiscal year a statement belongs to: the provider's own label, else the year its period ended."""
    return statement.fiscal_year if statement.fiscal_year is not None else int(statement.period_end[:4])


def _answer_of(raw, request) -> dict:
    """What an answer held (:func:`coverage.summarise_answer`) with the note the provider put in it, as the adapter kept it with the request (cut to length, with no key in it)."""
    return {**coverage.summarise_answer(raw), "message": str((request or {}).get("message") or "")}


def _freeze(profile: CompanyProfile) -> dict:
    """A search hit as plain JSON, for the saved answer."""
    out = {k: getattr(profile, k) for k in _PROFILE_TEXT}
    out["extra"] = {k: v for k, v in (profile.extra or {}).items() if k in _PROFILE_EXTRA}
    return out


def _thaw(provider: str, raw) -> CompanyProfile | None:
    """A saved hit as a profile again. What was saved is data from the outside: only known fields of the expected type come back, cut to a sensible length."""
    if not isinstance(raw, dict):
        return None

    def text(key: str) -> str | None:
        value = raw.get(key)
        return value.strip()[:200] if isinstance(value, str) and value.strip() else None
    symbol, ticker = text("provider_symbol"), text("ticker")
    if not symbol or not ticker:
        return None
    extra = raw.get("extra") if isinstance(raw.get("extra"), dict) else {}
    score = extra.get("score")
    return CompanyProfile(provider, symbol, ticker, text("name") or "", exchange=text("exchange") or "", market=text("market"), country=text("country"), currency=text("currency"),
                          sector=text("sector"), industry=text("industry"), isin=text("isin"), lei=text("lei"),
                          extra={"also_listed": [a.strip()[:30] for a in (extra.get("also_listed") or []) if isinstance(a, str)][:6], "score": score if isinstance(score, (int, float)) else None,
                                 "registry": str(extra.get("registry") or "")[:40], "hint": str(extra.get("hint") or "")[:200], "primary": bool(extra.get("primary")),
                                 "has_financials": bool(extra.get("has_financials", True))})


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
                self._take_up_choice(name)
            else:
                self._pools[name] = KeyPool.without_key(name, self.usage)
        return self._pools[name]

    def _take_up_choice(self, name: str) -> None:
        """The key the user chose to be in use is kept in the database as the key's salted hash (never the key) and applied whenever the pool is made, so a restart does not undo it."""
        saved = self.store.get_meta(f"key_choice:{name}")
        if saved:
            self._pools[name].prefer(saved)

    def use_key(self, name: str, slot) -> dict:
        """The user's choice of which key makes the requests (the *Use this key* button), named by its slot (``key 2``). Nothing is spent and nothing switches by itself: the choice is kept, as the key's
        salted hash and never the key, until another is made. It is refused while the key in use has spent its allowance, because OpenFilings' usage policy forbids rotating keys to get past a quota
        (:meth:`KeyPool.choose`)."""
        pool = self.pool(name)
        chosen = pool.choose(slot)
        self.store.set_meta(f"key_choice:{name}", pool.chosen_id or "")
        return {"provider": name, "active": chosen, "keys": pool.slots()}

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
            self._take_up_choice(name)

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
            hit = self._hit(provider, normal)                                           # a search has already paid for the name and the identifiers: keep them
            if hit is not None:
                profile.name, profile.isin, profile.lei = hit.name or profile.name, hit.isin or profile.isin, hit.lei or profile.lei
            ids.append(self.store.upsert_security(profile))
        if progress:
            progress(f"{len(ids)} companies registered")
        return ids

    def _lookup(self, provider: str, symbol: str) -> int | None:
        with self.store.db.transaction() as conn:
            row = conn.execute("SELECT security_id FROM security_identifiers WHERE provider = ? AND id_type = 'symbol' AND provider_id = ?", (provider, symbol)).fetchone()
        return int(row[0]) if row else None

    # ------------------------------------------------------------------------------------------------------------ finding a company
    def _hit(self, provider: str, symbol: str) -> CompanyProfile | None:
        try:
            return _thaw(provider, json.loads(self.store.get_meta(f"hit:{provider}:{symbol}") or "null"))
        except ValueError:
            return None

    def _saved_search(self, key: str, provider: str, today: pd.Timestamp) -> dict | None:
        """The saved answer to this search if there is one and it is recent enough to trust."""
        try:
            saved = json.loads(self.store.get_meta(key) or "null")
        except ValueError:
            return None
        if not isinstance(saved, dict) or not isinstance(saved.get("profiles"), list):
            return None
        try:
            age = (today - pd.Timestamp(str(saved.get("at"))[:10])).days
        except (ValueError, TypeError):
            return None
        profiles = [p for p in (_thaw(provider, raw) for raw in saved["profiles"]) if p is not None]
        keep = SEARCH_KEEP_DAYS if profiles else SEARCH_EMPTY_KEEP_DAYS
        return {**saved, "profiles": profiles} if 0 <= age < keep else None

    def allowed_markets(self, provider: str) -> tuple[set[str], bool] | None:
        """``(markets, all)`` the account's plan includes, as the last connection test reported them; ``None`` before there has been one."""
        try:
            saved = json.loads(self.store.get_meta(f"markets:{provider}") or "null")
        except ValueError:
            return None
        if not isinstance(saved, dict) or not isinstance(saved.get("allowed"), list):
            return None
        return {str(m).lower() for m in saved["allowed"]}, bool(saved.get("all"))

    def search_companies(self, provider: str, query: str, kind: str = "auto", market: str | None = None, limit: int = 10, refresh: bool = False) -> dict:
        """Look a company up in the provider's directory, by name, ticker or ISIN. Costs one credit (OpenFilings); the same search again is answered from the saved answer for nothing.

        Nothing is registered: adding a company is a separate, free step, and fetching its statements another, so a search never spends more than it says. What comes back says, for each hit,
        whether the company is already in the database and how many annual periods it has.
        """
        prov = self.provider(provider)
        ok, why = prov.available()
        if not ok:
            raise MissingKey(why)
        text = " ".join(str(query or "").split())
        if len(text) < 2:
            raise FundamentalsError("type at least two characters of a name, a ticker or an ISIN to search for")
        if len(text) > 120:
            raise FundamentalsError("a search is at most 120 characters")
        try:
            plan = prov.search_plan(text, kind, market)
        except ProviderError as error:
            raise FundamentalsError(str(error)) from None
        key = f"search:{provider}:{plan['kind']}:{plan['market'] or ''}:{plan['value'].lower()}"
        today = self._today()
        saved = None if refresh else self._saved_search(key, provider, today)
        if saved is not None:
            profiles, message, ambiguous, credits, at = saved["profiles"], str(saved.get("message") or ""), bool(saved.get("ambiguous")), 0, str(saved.get("at"))[:10]
            skipped = [str(m)[:12] for m in (saved.get("skipped") or []) if isinstance(m, str)][:20]
        else:
            detail = self._call(provider, lambda: prov.search_detail(text, limit, plan["kind"], plan["market"]), "search", text)
            profiles, message, ambiguous, credits, at = detail["profiles"], str(detail.get("message") or ""), bool(detail.get("ambiguous")), int(detail.get("credits", 0)), today.strftime("%Y-%m-%d")
            skipped = [str(m)[:12] for m in (detail.get("markets_skipped") or []) if isinstance(m, str)][:20]
            self.store.set_meta(key, json.dumps({"at": at, "message": message, "ambiguous": ambiguous, "skipped": skipped, "profiles": [_freeze(p) for p in profiles]}, separators=(",", ":")))
            for p in profiles:
                self.store.set_meta(f"hit:{provider}:{p.provider_symbol}", json.dumps(_freeze(p), separators=(",", ":")))
        allowed = self.allowed_markets(provider)
        rows = []
        for p in profiles:
            sid = self._lookup(provider, p.provider_symbol)
            market_id = p.provider_symbol.partition(":")[0]
            rows.append({"symbol": p.provider_symbol, "ticker": p.ticker, "market": market_id, "name": p.name, "country": p.country, "isin": p.isin, "lei": p.lei,
                         "primary": bool(p.extra.get("primary")), "also_listed": list(p.extra.get("also_listed") or []), "score": p.extra.get("score"), "hint": p.extra.get("hint") or "",
                         "has_financials": bool(p.extra.get("has_financials", True)), "plan_allows": None if allowed is None else (allowed[1] or market_id in allowed[0]),
                         "in_database": sid is not None, "security_id": sid, "periods": int(self.store.period_summary(sid, provider).get("n") or 0) if sid is not None else 0})
        pool = self.pool(provider)
        return {"query": text, "kind": plan["kind"], "kind_word": KIND_WORDS.get(plan["kind"], plan["kind"]), "market": plan["market"] or "", "alternatives": list(plan["alternatives"]),
                "results": rows, "count": len(rows), "cached": saved is not None, "credits": credits, "message": message, "ambiguous": ambiguous, "markets_skipped": skipped, "searched_at": at,
                "remaining_today": pool.remaining_today(), "unit": pool.allow.unit, "cost_to_fetch": prov.cost("financials") + (prov.cost("profile") if prov.has_lookup else 0)}

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
        try:
            result = self._call(task.provider, lambda: prov.get_financials(task.symbol, start_year=None if incremental else start_year, period=label, limit=newest),
                                f"financials:{label}", task.symbol)
        except NotFound as error:
            self._remember_empty(task.security_id, task.provider, label, error, task.reason)
            raise
        cost += result.cost
        self.store.save_raw(task.provider, task.security_id, f"financials:{label}", result.request, result.raw)
        if result.profile is not None:
            self.store.enrich_security(task.security_id, result.profile)
        counts = self.store.store_financials(task.security_id, task.provider, result.periods)
        asked_from, asked_to = coverage.asked_years(result.request)
        self.store.log_fetch(task.security_id, task.provider, f"financials:{label}", asked_from, asked_to, _answer_of(result.raw, result.request), len(result.periods), counts["periods"], result.cost,
                             task.reason, self.usage.today())
        quality.check_security(self.store, task.security_id)
        summary = self.store.period_summary(task.security_id, task.provider)
        old = self.store.get_sync(task.security_id, task.provider, self._state_key(task.period)) or {}
        earliest = min([d for d in (summary.get("first_end"), old.get("earliest_period_end")) if d], default=None)
        self.store.set_sync(task.security_id, task.provider, self._state_key(task.period), last_checked=self.usage.today(), last_success=self.usage.today(),
                            last_period_end=summary.get("last_end"), earliest_period_end=earliest, history_exhausted=1 if (result.exhausted or old.get("history_exhausted")) else 0,
                            n_periods=int(summary.get("n") or 0), status="ok", error=None)
        return {"cost": cost, "requests": result.requests, "periods": counts["periods"], "vintages": counts["vintages"], "unmapped": counts["unmapped"]}

    def _remember_empty(self, security_id: int, provider: str, label: str, error: NotFound, reason: str) -> None:
        """A series that came back without one annual statement in it was paid for: keep the answer, and which years the request covered, so that those years are not paid for again unasked."""
        if error.raw is None:
            return                                                       # the provider does not know the company: nothing was charged and nothing is learnt about any year
        request = error.request or {}
        asked_from, asked_to = coverage.asked_years(request)
        self.store.save_raw(provider, security_id, f"financials:{label}", request, error.raw)
        self.store.log_fetch(security_id, provider, f"financials:{label}", asked_from, asked_to, _answer_of(error.raw, request), 0, 0, int(error.charged or 0), reason, self.usage.today())

    # ------------------------------------------------------------------------------------------------------------ which years are held
    def _known(self, security_id: int, provider: str | None) -> tuple[dict, str, FundamentalDataProvider]:
        info = self.store.get_security(int(security_id))
        if info is None:
            raise FundamentalsError(f"no company with id {security_id}")
        name = provider or info.get("source") or "openfilings"
        return info, name, self.provider(name)

    def coverage(self, security_id: int, provider: str | None = None, today: date | str | None = None) -> dict:
        """The fiscal years of one company: each held, missing, found empty when asked for, due (a filing that should exist by now) or not yet due; the quarterly statements held; the last request in words
        (why the company has the years it has); and whether and at what price more can be asked for. Nothing is spent and no provider is contacted."""
        info, name, prov = self._known(security_id, provider)
        sid = int(security_id)
        day = date.fromisoformat(str(today or self.usage.today())[:10])
        held = self.store.held_periods(sid, name)
        request_type = "financials:annual"
        log = self.store.fetch_log(sid, name, request_type)
        derived = False
        if not log:
            log = coverage.derive_log(self.store.raw_answers(sid, name, request_type))
            derived = bool(log)
        annual = coverage.grid([r for r in held if r["fiscal_period"] == "FY"], log, day, FILING_LAG["annual"])
        pool = self.pool(name)
        ok, why = prov.available()
        word = PROVIDER_WORDS.get(name, name)
        quarters: dict[int, dict] = {}
        for r in held:
            if r["fiscal_period"] != "FY":
                quarters.setdefault(int(r["fiscal_year"]), {})[r["fiscal_period"]] = str(r["period_end"])[:10]
        return {"security_id": int(security_id), "ticker": info["ticker"], "name": info.get("company_name"), "provider": name, "provider_word": word, "symbol": self._symbols(name).get(sid),
                "annual": annual, "quarterly": {"supported": "quarter" in prov.periods_available, "held": sum(len(v) for v in quarters.values()),
                                                "rows": [{"year": y, "periods": quarters[y]} for y in sorted(quarters)], "note": "" if "quarter" in prov.periods_available else NO_QUARTERS.get(name, f"{word} has no quarterly figures here.")},
                "last_request": log[0] if log else None, "reading": coverage.reading(log[0] if log else None, word, pool.allow.unit), "derived": derived,
                "log": [{**r, "asked": coverage.describe_window({"from": r.get("asked_from"), "to": r.get("asked_to")})} for r in log[:8]],
                "unit": pool.allow.unit, "request_cost": prov.cost("financials"), "remaining_today": pool.remaining_today() if ok else None, "can_fetch": ok and "annual" in prov.periods_available,
                "why_not": "" if ok else why, "sync": self.store.get_sync(sid, name, request_type), "today": day.isoformat()}

    def plan_years(self, security_id: int, years, provider: str | None = None, period: str = "annual", ask_again: bool = False, today: date | str | None = None) -> dict:
        """What asking for ``years`` would do and cost, without asking: which are held already (never asked for), which are left out (found empty before, not yet due) and the one request that covers the rest."""
        if period != "annual":
            raise FundamentalsError("years are picked for the annual statements; quarterly figures are not requested one year at a time")
        try:
            picked = sorted({int(y) for y in years})
        except (TypeError, ValueError):
            raise FundamentalsError("years must be whole numbers such as 2014") from None
        if len(picked) > coverage.MAX_PICK:
            raise FundamentalsError(f"pick at most {coverage.MAX_PICK} years at a time")
        cov = self.coverage(security_id, provider, today)
        p = coverage.plan(cov["annual"]["cells"], picked, ask_again, cov["annual"]["latest_held"], cov["request_cost"])
        p.update({"security_id": cov["security_id"], "provider": cov["provider"], "symbol": cov["symbol"], "unit": cov["unit"], "remaining_today": cov["remaining_today"], "can_fetch": cov["can_fetch"],
                  "why_not": cov["why_not"], "fits_today": cov["remaining_today"] is None or p["cost"] <= cov["remaining_today"], "text": coverage.plan_text(p, cov["unit"], cov["remaining_today"])})
        return p

    def fetch_years(self, security_id: int, years, budget: int, provider: str | None = None, period: str = "annual", ask_again: bool = False, progress=None,
                    today: date | str | None = None) -> dict:
        """Ask the provider once for the picked years that are not held, store what is new and leave what is held alone. ``budget`` is the most this may spend (the plan's cost is what the user was shown)."""
        say = progress or (lambda text: None)
        p = self.plan_years(security_id, years, provider, period, ask_again, today)
        if not p["to_fetch"]:
            raise FundamentalsError(coverage.nothing_to_fetch(p))
        if not p["can_fetch"]:
            raise MissingKey(p["why_not"] or f"{p['provider']} cannot be asked now")
        if p["cost"] > budget:
            raise FundamentalsError(f"this asks for {coverage.amount(p['cost'], p['unit'])} but the budget is {budget}")
        name, sid, window = p["provider"], int(security_id), p["window"]
        prov, word = self.provider(name), PROVIDER_WORDS.get(name, name)
        if not p["symbol"]:
            raise FundamentalsError(f"{word} does not know this company yet: fetch its statements first")
        self.pool(name).ensure(p["cost"])
        say(f"{p['symbol']}: asking for {coverage.describe_window(window)}, up to {coverage.amount(p['cost'], p['unit'])}")
        before = {(r["fiscal_period"], str(r["period_end"])[:10]) for r in self.store.held_periods(sid, name)}
        label, request_type, day = "annual", "financials:annual", self.usage.today()
        try:
            result = self._call(name, lambda: prov.get_financials(p["symbol"], start_year=window["from"], end_year=window["to"], period=label), request_type, p["symbol"])
        except NotFound as error:
            if error.raw is None:
                raise
            self._remember_empty(sid, name, label, error, "chosen")
            held_years = [int(r["fiscal_year"]) for r in self.store.held_periods(sid, name) if r["fiscal_period"] == "FY"]
            self.store.set_sync(sid, name, request_type, last_checked=day, **({"history_exhausted": 1} if held_years and window["from"] < min(held_years) else {}))      # nothing from before the earliest year held
            report = {"security_id": sid, "symbol": p["symbol"], "provider": name, "window": window, "cost": int(error.charged or 0), "unit": p["unit"], "added": [], "left_alone": 0, "outside": 0,
                      "nothing_for": p["to_fetch"], "entries": coverage.summarise_answer(error.raw)["entries"]}
            return {**report, "text": coverage.report_text(report, word)}
        inside = [q for q in result.periods if _year_of(q) >= window["from"] and (window["to"] is None or _year_of(q) <= window["to"])]
        fresh = [q for q in inside if (q.fiscal_period, q.period_end[:10]) not in before]
        self.store.save_raw(name, sid, request_type, result.request, result.raw)                        # what was paid for is kept before anything is made of it
        if result.profile is not None:
            self.store.enrich_security(sid, result.profile)
        counts = self.store.store_financials(sid, name, fresh)
        quality.check_security(self.store, sid)
        held = self.store.held_periods(sid, name)
        added = sorted({(int(r["fiscal_year"]), str(r["period_end"])[:10]) for r in held if r["fiscal_period"] == "FY" and (r["fiscal_period"], str(r["period_end"])[:10]) not in before})
        first_held = min((int(r["fiscal_year"]) for r in held if r["fiscal_period"] == "FY"), default=None)
        summary = self.store.period_summary(sid, name)
        old = self.store.get_sync(sid, name, request_type) or {}
        earliest = min([d for d in (summary.get("first_end"), old.get("earliest_period_end")) if d], default=None)
        asked_from, asked_to = coverage.asked_years(result.request)
        self.store.log_fetch(sid, name, request_type, asked_from, asked_to, _answer_of(result.raw, result.request), len(result.periods), counts["periods"], result.cost, "chosen", day)
        self.store.set_sync(sid, name, request_type, last_checked=day, last_success=day, last_period_end=summary.get("last_end"), earliest_period_end=earliest,
                            history_exhausted=1 if (old.get("history_exhausted") or (first_held is not None and window["from"] < first_held)) else 0,        # asked from before anything held
                            n_periods=int(summary.get("n") or 0), status="ok", error=None)
        report = {"security_id": sid, "symbol": p["symbol"], "provider": name, "window": window, "cost": result.cost, "unit": p["unit"], "added": [{"year": y, "period_end": e} for y, e in added],
                  "left_alone": len(inside) - len(fresh), "outside": len(result.periods) - len(inside), "nothing_for": [y for y in p["to_fetch"] if y not in {y for y, _ in added}],
                  "entries": coverage.summarise_answer(result.raw)["entries"]}
        return {**report, "text": coverage.report_text(report, word)}

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
        if result.get("allowed_markets") is not None:
            self.store.set_meta(f"markets:{name}", json.dumps({"allowed": [str(m).lower() for m in result["allowed_markets"]], "all": bool(result.get("all_markets"))}))
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
