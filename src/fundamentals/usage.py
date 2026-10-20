"""Every external request, counted: the quota manager behind the Data Manager's DATA STATUS panel and behind the rule "never make a hidden or duplicate request".

A provider call goes through :meth:`UsageLog.record` (success or failure, which endpoint, which security, which key *slot*; the key itself is never stored, only a salted one-way identifier so a
key keeps its count when the list is reordered). Budgets are per provider: a keyed provider's daily allowance is per key (so five keys give five allowances), a keyless one has a politeness budget
the user can change. Days are UTC days.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from .errors import QuotaExhausted  # noqa: F401  (re-exported: the quota error belongs with the quota manager)
from .schema import Database, now_iso

# per-key daily allowances; ``<PROVIDER>_DAILY_LIMIT`` overrides. FMP's free plan is documented at 250 requests a day: check the plan of your own keys, a wrong number here only makes the manager
# stop early (too low) or find out from the provider (too high, a rate-limit answer ends that key's day).
DEFAULT_DAILY_LIMITS = {"fmp": 250, "openfilings": 300, "simulated": 1_000_000}
LIMIT_KINDS = {"fmp": "per key", "openfilings": "politeness budget", "simulated": "none"}


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def daily_limit(provider: str) -> int | None:
    """The requests a day allowed for one key of ``provider`` (or for the provider when it has no keys); ``None`` when the provider has no limit."""
    raw = os.environ.get(f"{provider.upper()}_DAILY_LIMIT", "").strip()
    if raw:
        try:
            value = int(raw)
            return value if value > 0 else None
        except ValueError:
            pass
    return DEFAULT_DAILY_LIMITS.get(provider)


@dataclass(frozen=True)
class Usage:
    provider: str
    date: str
    requests: int
    successful: int
    failed: int
    limit: int | None            # the allowance across every key
    keys: tuple                  # per key: {"slot", "used", "limit", "state"}


class UsageLog:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------------------------------------------------ writing
    def record(self, provider: str, endpoint: str = "", success: bool = True, symbol: str = "", market: str = "", key_slot: str = "", key_id: str = "", requests: int = 1,
               credits: int = 1, error: str = "") -> None:
        with self.db.transaction() as conn:
            conn.execute("INSERT INTO api_usage (provider, date, endpoint, requests, credits_used, success, symbol, market, timestamp, key_slot, key_id, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (provider, today(), endpoint, int(requests), int(credits), 1 if success else 0, symbol, market, now_iso(), key_slot, key_id, (error or "")[:300]))

    # ------------------------------------------------------------------------------------------------------------ reading
    def used_today(self, provider: str, key_id: str | None = None) -> int:
        sql, args = "SELECT COALESCE(SUM(requests), 0) FROM api_usage WHERE provider = ? AND date = ?", [provider, today()]
        if key_id is not None:
            sql += " AND key_id = ?"
            args.append(key_id)
        with self.db.transaction() as conn:
            return int(conn.execute(sql, args).fetchone()[0])

    def blocked_today(self, provider: str, key_id: str) -> str | None:
        """Why ``key_id`` may not be used for the rest of today (an earlier answer said its allowance is gone or that it is not a valid key), else ``None``."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT error FROM api_usage WHERE provider = ? AND key_id = ? AND date = ? AND (error LIKE 'limit%' OR error LIKE 'auth%') ORDER BY id DESC LIMIT 1",
                               (provider, key_id, today())).fetchone()
        return row[0] if row else None

    def summary(self, provider: str, keys: list[dict] | None = None, days: int = 1) -> Usage:
        """Today's counts for ``provider``; ``keys`` (from the key pool: ``slot``, ``key_id``, ``state``) adds a line per key."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT COALESCE(SUM(requests),0), COALESCE(SUM(CASE WHEN success=1 THEN requests ELSE 0 END),0), COALESCE(SUM(CASE WHEN success=0 THEN requests ELSE 0 END),0) "
                               "FROM api_usage WHERE provider = ? AND date = ?", (provider, today())).fetchone()
        limit = daily_limit(provider)
        lines = []
        for k in keys or []:
            lines.append({"slot": k["slot"], "used": self.used_today(provider, k["key_id"]), "limit": limit, "state": k.get("state", "ready")})
        total_limit = None if limit is None else limit * max(len(lines), 1)
        return Usage(provider, today(), int(row[0]), int(row[1]), int(row[2]), total_limit, tuple(lines))

    def history(self, provider: str, days: int = 14) -> list[dict]:
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT date, SUM(requests) AS requests, SUM(CASE WHEN success=1 THEN requests ELSE 0 END) AS ok FROM api_usage WHERE provider = ? "
                                "GROUP BY date ORDER BY date DESC LIMIT ?", (provider, int(days))).fetchall()
        return [dict(r) for r in rows][::-1]

    def last_update(self, provider: str | None = None) -> str | None:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT MAX(timestamp) FROM api_usage WHERE success = 1" + (" AND provider = ?" if provider else ""), (provider,) if provider else ()).fetchone()
        return row[0] if row and row[0] else None
