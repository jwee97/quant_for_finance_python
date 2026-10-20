"""Every external request, counted: the quota manager behind the Data Manager's DATA STATUS panel and behind the rule "never make a hidden or duplicate request".

A provider call goes through :meth:`UsageLog.record` (success or failure, which endpoint, which security, which key *slot*, how many credits it cost; the key itself is never stored, only a salted
one-way identifier so a key keeps its count when the list is reordered). Allowances are per provider and per key, in the provider's own unit: OpenFilings meters **credits** (a search costs 1, a
multi-year financial statement series 10) against a daily and a monthly allowance; the simulator meters plain requests. Days and months are UTC. A failed request costs nothing (OpenFilings refunds
it), so only successful calls add to the count.
"""

from __future__ import annotations

import os
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .errors import QuotaExhausted  # noqa: F401  (re-exported: the quota error belongs with the quota manager)
from .schema import Database, now_iso

# OpenFilings plans: credits per UTC day and per calendar month, one pool shared by the REST API and the MCP server (https://openfilings.org/docs/rate-limits). Top-up credits are a prepaid balance
# on top of these and never reset.
PLANS = {"free": (100, 1_000), "pro": (2_500, 25_000), "business": (10_000, 100_000)}
UNITS = {"openfilings": "credits", "simulated": "requests"}
DEFAULT_PLAN = "free"
SIMULATED_DAILY = 1_000_000


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def today() -> str:
    return utc_now().strftime("%Y-%m-%d")


def _env_int(name: str) -> tuple[bool, int | None]:
    """``(set, value)``: a positive integer, or ``None`` (no limit) for 0 or a negative number."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return False, None
    try:
        value = int(raw)
    except ValueError:
        return False, None
    return True, (value if value > 0 else None)


@dataclass(frozen=True)
class Allowance:
    """What one key may use: ``daily`` and ``monthly`` amounts of ``unit`` (``None`` means no limit), and where the numbers came from so the Data Manager can say how sure they are."""

    unit: str
    daily: int | None
    monthly: int | None
    source: str


def allowance(provider: str, plan: str | None = None) -> Allowance:
    """The allowance of one key of ``provider``.

    ``<PROVIDER>_DAILY_LIMIT`` and ``<PROVIDER>_MONTHLY_LIMIT`` win; otherwise OpenFilings uses its plan (``OPENFILINGS_PLAN``, else the plan the account reported when last tested, else the free
    plan). A wrong number here only makes the manager stop early (too low) or find out from the provider (too high: it answers HTTP 429 and the key rests until the allowance resets).
    """
    unit = UNITS.get(provider, "requests")
    set_d, daily = _env_int(f"{provider.upper()}_DAILY_LIMIT")
    set_m, monthly = _env_int(f"{provider.upper()}_MONTHLY_LIMIT")
    if provider == "openfilings":
        name = (os.environ.get("OPENFILINGS_PLAN", "").strip().lower() or (plan or "").lower())
        known = name in PLANS
        base_d, base_m = PLANS[name if known else DEFAULT_PLAN]
        source = f"plan {name} (set by OPENFILINGS_PLAN)" if known and os.environ.get("OPENFILINGS_PLAN", "").strip() else (
            f"plan {name} (reported by the account)" if known else f"plan {DEFAULT_PLAN} (assumed: use Test on the Data Manager page, or set OPENFILINGS_PLAN)")
        if set_d or set_m:
            source = "environment limits"
        return Allowance(unit, daily if set_d else base_d, monthly if set_m else base_m, source)
    if provider == "simulated":
        return Allowance(unit, daily if set_d else SIMULATED_DAILY, monthly if set_m else None, "environment limits" if set_d or set_m else "none")
    return Allowance(unit, daily if set_d else None, monthly if set_m else None, "environment limits" if set_d or set_m else "none")


@dataclass(frozen=True)
class Block:
    """Why a key may not be used now, and when that ends (``until`` is ``None`` when only the user can end it: topping up credits, replacing a bad key)."""

    kind: str                 # day | month | balance | auth
    until: str | None
    message: str


@dataclass(frozen=True)
class Usage:
    provider: str
    date: str
    unit: str
    requests: int                # calls made today
    successful: int
    failed: int
    used_today: int              # in ``unit``
    used_month: int
    daily_limit: int | None
    monthly_limit: int | None
    keys: tuple                  # per key: {"slot", "used_today", "used_month", "state", "role"}


class UsageLog:
    def __init__(self, db: Database, clock=None):
        self.db = db
        self._clock = clock or utc_now

    # ------------------------------------------------------------------------------------------------------------ the calendar
    def today(self) -> str:
        return self._clock().strftime("%Y-%m-%d")

    def month_start(self) -> str:
        return self._clock().strftime("%Y-%m-01")

    def next_day(self) -> str:
        return (self._clock().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).strftime("%Y-%m-%d 00:00 UTC")

    def next_month(self) -> str:
        now = self._clock()
        days = monthrange(now.year, now.month)[1]
        return (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days)).strftime("%Y-%m-%d 00:00 UTC")

    # ------------------------------------------------------------------------------------------------------------ writing
    def record(self, provider: str, endpoint: str = "", success: bool = True, symbol: str = "", market: str = "", key_slot: str = "", key_id: str = "", requests: int = 1,
               credits: int | None = None, error: str = "") -> None:
        """One external call. ``credits`` defaults to 1 for a success and 0 for a failure (a failed call is not charged)."""
        spent = int(credits) if credits is not None else (1 if success else 0)
        with self.db.transaction() as conn:
            conn.execute("INSERT INTO api_usage (provider, date, endpoint, requests, credits_used, success, symbol, market, timestamp, key_slot, key_id, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (provider, self.today(), endpoint, int(requests), spent, 1 if success else 0, symbol, market, now_iso(), key_slot, key_id, (error or "")[:300]))

    # ------------------------------------------------------------------------------------------------------------ reading
    def _sum(self, column: str, provider: str, key_id: str | None, since: str) -> int:
        sql, args = f"SELECT COALESCE(SUM({column}), 0) FROM api_usage WHERE provider = ? AND date >= ?", [provider, since]
        if key_id is not None:
            sql += " AND key_id = ?"
            args.append(key_id)
        with self.db.transaction() as conn:
            return int(conn.execute(sql, args).fetchone()[0])

    def used_today(self, provider: str, key_id: str | None = None) -> int:
        """Credits (or requests, for a provider that counts those) spent today."""
        return self._sum("credits_used", provider, key_id, self.today())

    def used_month(self, provider: str, key_id: str | None = None) -> int:
        return self._sum("credits_used", provider, key_id, self.month_start())

    def requests_today(self, provider: str, key_id: str | None = None) -> int:
        return self._sum("requests", provider, key_id, self.today())

    def blocked(self, provider: str, key_id: str) -> Block | None:
        """Why ``key_id`` may not be used now (the provider's last answer to it said its allowance is gone, its credits are gone, or it is not a valid key), else ``None``.

        Only the key's latest call counts, so one that works again (the day rolled over, the user topped up and pressed Test) is free at once.
        """
        with self.db.transaction() as conn:
            row = conn.execute("SELECT success, date, error FROM api_usage WHERE provider = ? AND key_id = ? ORDER BY id DESC LIMIT 1", (provider, key_id)).fetchone()
        if row is None or row["success"]:
            return None
        error, day = row["error"] or "", row["date"]
        if error.startswith("limit:day") and day == self.today():
            return Block("day", self.next_day(), error)
        if error.startswith("limit:month") and day[:7] == self.today()[:7]:
            return Block("month", self.next_month(), error)
        if error.startswith("balance"):
            return Block("balance", None, error)
        if error.startswith("auth") and day == self.today():
            return Block("auth", None, error)
        return None

    def summary(self, provider: str, keys: list[dict] | None = None, plan: str | None = None) -> Usage:
        """Today's counts for ``provider``; ``keys`` (from the key pool) adds a line per key."""
        allow = allowance(provider, plan)
        with self.db.transaction() as conn:
            row = conn.execute("SELECT COALESCE(SUM(requests),0), COALESCE(SUM(CASE WHEN success=1 THEN requests ELSE 0 END),0), COALESCE(SUM(CASE WHEN success=0 THEN requests ELSE 0 END),0) "
                               "FROM api_usage WHERE provider = ? AND date = ?", (provider, self.today())).fetchone()
        lines = [{"slot": k["slot"], "used_today": self.used_today(provider, k["key_id"]), "used_month": self.used_month(provider, k["key_id"]), "state": k.get("state", "ready"),
                  "role": k.get("role", "")} for k in keys or []]
        return Usage(provider, self.today(), allow.unit, int(row[0]), int(row[1]), int(row[2]), self.used_today(provider), self.used_month(provider), allow.daily, allow.monthly, tuple(lines))

    def history(self, provider: str, days: int = 14) -> list[dict]:
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT date, SUM(requests) AS requests, SUM(CASE WHEN success=1 THEN requests ELSE 0 END) AS ok, SUM(credits_used) AS credits FROM api_usage WHERE provider = ? "
                                "GROUP BY date ORDER BY date DESC LIMIT ?", (provider, int(days))).fetchall()
        return [dict(r) for r in rows][::-1]

    def last_update(self, provider: str | None = None) -> str | None:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT MAX(timestamp) FROM api_usage WHERE success = 1" + (" AND provider = ?" if provider else ""), (provider,) if provider else ()).fetchone()
        return row[0] if row and row[0] else None

    def recent(self, provider: str, limit: int = 30) -> list[dict]:
        """The latest calls (no key material), newest first: what the Data Manager's activity log shows."""
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT timestamp, endpoint, symbol, market, key_slot, requests, credits_used, success, error FROM api_usage WHERE provider = ? ORDER BY id DESC LIMIT ?",
                                (provider, int(limit))).fetchall()
        return [dict(r) for r in rows]
