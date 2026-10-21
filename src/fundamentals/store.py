"""The repository: everything the application reads and writes in the fundamentals database goes through here.

Writers are idempotent (running the same download twice changes nothing) and keep what the point-in-time engine needs: when a figure first became public, and every later restatement as a new
``vintage`` that is public only from the day it was first seen. Readers return pandas frames in the long layout the engine pivots, so research never loops over securities in Python.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import canonical
from .errors import FundamentalsError
from .schema import REGIONS, Database, now_iso

DEFAULT_PATH = Path("data") / "user" / "fundamentals.db"
RESTATEMENT_TOLERANCE = 1e-6


@dataclass
class CompanyProfile:
    """What a provider says about a company (the security master's raw material)."""

    provider: str
    provider_symbol: str
    ticker: str
    name: str = ""
    exchange: str = ""
    market: str | None = None             # an exchange code from the markets table, when the provider names one
    country: str | None = None            # ISO 3166 alpha-2
    currency: str | None = None           # the trading currency
    sector: str | None = None
    industry: str | None = None
    isin: str | None = None
    figi: str | None = None
    lei: str | None = None
    active: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class PeriodStatement:
    """One fiscal period of one company as a provider reported it: the source fields, untouched, and the dates that say when it became public."""

    fiscal_period: str                    # FY, Q1..Q4, H1, H2
    period_end: str
    values: dict                          # source field -> number
    fiscal_year: int | None = None
    period_start: str | None = None
    filing_date: str | None = None
    accepted_date: str | None = None
    publication_date: str | None = None
    reporting_currency: str | None = None
    accounting_standard: str | None = None
    provider_record_id: str | None = None


def _date(value) -> str | None:
    if value in (None, "", "0000-00-00"):
        return None
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


class FundamentalsStore:
    def __init__(self, db: Database | str | Path):
        self.db = db if isinstance(db, Database) else Database(db)

    # ------------------------------------------------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))

    def salt(self) -> str:
        """A random string created once per database, mixed into the hash that identifies an API key in the usage log."""
        value = self.get_meta("key_salt")
        if value is None:
            value = hashlib.sha256(np.random.default_rng().bytes(32)).hexdigest()
            with self.db.transaction() as conn:
                conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('key_salt', ?)", (value,))
            value = self.get_meta("key_salt")
        return value

    # ------------------------------------------------------------------------------------------------------------ the security master
    def ensure_market(self, conn, code: str | None, country: str | None, currency: str | None, name: str | None = None) -> int | None:
        if not code:
            return None
        row = conn.execute("SELECT market_id FROM markets WHERE code = ?", (code,)).fetchone()
        if row:
            return int(row[0])
        cur = conn.execute("INSERT INTO markets (code, name, country, region, currency) VALUES (?,?,?,?,?)", (code, name or code, country, REGIONS.get(country or ""), currency))
        return int(cur.lastrowid)

    def upsert_security(self, profile: CompanyProfile) -> int:
        """The ``security_id`` for this company, creating it if ``(ticker, exchange)`` is new. Fields the provider leaves empty never erase what another provider supplied."""
        if not profile.ticker:
            raise FundamentalsError("a security needs a ticker")
        ticker, exchange = profile.ticker.strip().upper(), (profile.exchange or "").strip().upper()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.db.transaction() as conn:
            market_id = self.ensure_market(conn, profile.market or exchange or None, profile.country, profile.currency)
            row = conn.execute("SELECT security_id FROM securities WHERE ticker = ? AND exchange = ?", (ticker, exchange)).fetchone()
            fields = {"company_name": profile.name or None, "country": profile.country, "currency": profile.currency, "sector": profile.sector, "industry": profile.industry,
                      "isin": profile.isin, "figi": profile.figi, "lei": profile.lei, "provider_symbol": profile.provider_symbol or None, "source": profile.provider, "market_id": market_id}
            if row is None:
                cols = ["ticker", "exchange", "active", "first_seen", "last_seen"] + list(fields)
                vals = [ticker, exchange, 1 if profile.active else 0, today, today] + list(fields.values())
                cur = conn.execute(f"INSERT INTO securities ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
                security_id = int(cur.lastrowid)
            else:
                security_id = int(row[0])
                sets = [f"{k} = COALESCE(?, {k})" for k in fields]
                conn.execute(f"UPDATE securities SET {', '.join(sets)}, last_seen = ?, active = ? WHERE security_id = ?",
                             list(fields.values()) + [today, 1 if profile.active else 0, security_id])
            for id_type, value in (("symbol", profile.provider_symbol), ("isin", profile.isin), ("lei", profile.lei), ("figi", profile.figi)):
                if value:
                    conn.execute("INSERT OR IGNORE INTO security_identifiers (security_id, provider, id_type, provider_id) VALUES (?,?,?,?)", (security_id, profile.provider, id_type, str(value)))
        return security_id

    def enrich_security(self, security_id: int, profile: CompanyProfile) -> None:
        """Fill what is still blank on an existing security from what a provider has just said about it. Nothing already known is overwritten and ``(ticker, exchange)`` never changes, so an answer
        that spells the ticker differently cannot create a second copy of the company."""
        fields = {"company_name": profile.name or None, "country": profile.country, "currency": profile.currency, "sector": profile.sector, "industry": profile.industry,
                  "isin": profile.isin, "figi": profile.figi, "lei": profile.lei}
        with self.db.transaction() as conn:
            sets = ", ".join(f"{k} = COALESCE({k}, ?)" for k in fields)
            conn.execute(f"UPDATE securities SET {sets}, last_seen = ? WHERE security_id = ?", [*fields.values(), datetime.now(timezone.utc).strftime("%Y-%m-%d"), int(security_id)])
            for id_type, value in (("isin", profile.isin), ("lei", profile.lei), ("figi", profile.figi)):
                if value:
                    conn.execute("INSERT OR IGNORE INTO security_identifiers (security_id, provider, id_type, provider_id) VALUES (?,?,?,?)", (int(security_id), profile.provider, id_type, str(value)))

    def set_active(self, security_id: int, active: bool, delisted_date: str | None = None) -> None:
        with self.db.transaction() as conn:
            conn.execute("UPDATE securities SET active = ?, delisted_date = ? WHERE security_id = ?", (1 if active else 0, None if active else _date(delisted_date), int(security_id)))

    def securities(self, active_only: bool = False, countries: list[str] | None = None, sectors: list[str] | None = None) -> pd.DataFrame:
        sql = ("SELECT s.security_id, s.ticker, s.exchange, s.company_name, s.country, s.currency, s.sector, s.industry, s.isin, s.provider_symbol, s.source, s.active, s.delisted_date, "
               "s.first_seen, s.last_seen, m.code AS market, m.region FROM securities s LEFT JOIN markets m ON m.market_id = s.market_id WHERE 1=1")
        args: list = []
        if active_only:
            sql += " AND s.active = 1"
        if countries:
            sql += f" AND s.country IN ({','.join('?' * len(countries))})"
            args += list(countries)
        if sectors:
            sql += f" AND s.sector IN ({','.join('?' * len(sectors))})"
            args += list(sectors)
        with self.db.transaction() as conn:
            return pd.read_sql_query(sql + " ORDER BY s.security_id", conn, params=args)

    def get_security(self, security_id: int) -> dict | None:
        frame = self.securities()
        row = frame[frame["security_id"] == int(security_id)]
        return None if row.empty else {k: (None if pd.isna(v) else v) for k, v in row.iloc[0].to_dict().items()}

    def search(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Companies whose ticker or name contains ``query``, with the exchange and country that tell same-ticker listings apart."""
        like = f"%{query.strip()}%"
        with self.db.transaction() as conn:
            return pd.read_sql_query(
                "SELECT s.security_id, s.ticker, s.exchange, s.company_name, s.country, s.currency, s.sector, s.industry FROM securities s "
                "WHERE s.ticker LIKE ? COLLATE NOCASE OR s.company_name LIKE ? COLLATE NOCASE OR s.isin = ? "
                "ORDER BY (s.ticker = ? COLLATE NOCASE) DESC, s.company_name LIMIT ?", conn, params=[like, like, query.strip().upper(), query.strip(), int(limit)])

    def identifiers(self, security_id: int) -> pd.DataFrame:
        with self.db.transaction() as conn:
            return pd.read_sql_query("SELECT provider, id_type, provider_id FROM security_identifiers WHERE security_id = ?", conn, params=[int(security_id)])

    # ------------------------------------------------------------------------------------------------------------ raw payloads
    def save_raw(self, provider: str, security_id: int | None, request_type: str, parameters: dict, payload) -> tuple[int, bool]:
        """Keep the provider's answer once. Returns ``(id, is_new)``; the same answer again (same hash) is not stored a second time."""
        digest = _hash(payload)
        blob = zlib.compress(json.dumps(payload, default=str, separators=(",", ":")).encode(), 6)
        with self.db.transaction() as conn:
            row = conn.execute("SELECT id FROM raw_fundamental_payloads WHERE provider = ? AND response_hash = ?", (provider, digest)).fetchone()
            if row:
                return int(row[0]), False
            cur = conn.execute("INSERT INTO raw_fundamental_payloads (provider, security_id, request_type, request_parameters, response_payload, response_hash, retrieved_at) VALUES (?,?,?,?,?,?,?)",
                               (provider, security_id, request_type, json.dumps(parameters, sort_keys=True, default=str), blob, digest, now_iso()))
            return int(cur.lastrowid), True

    def load_raw(self, raw_id: int):
        with self.db.transaction() as conn:
            row = conn.execute("SELECT response_payload FROM raw_fundamental_payloads WHERE id = ?", (int(raw_id),)).fetchone()
        if row is None:
            raise KeyError(f"no stored payload {raw_id}")
        return json.loads(zlib.decompress(row[0]).decode())

    # ------------------------------------------------------------------------------------------------------------ periods and observations
    def store_financials(self, security_id: int, provider: str, periods: list[PeriodStatement], retrieved: str | None = None) -> dict:
        """Normalise and store ``periods``. Returns counts: ``periods`` new, ``vintages`` (restatements), ``unchanged``, ``observations`` written and ``unmapped`` source fields."""
        retrieved = _date(retrieved) or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        counts = {"periods": 0, "vintages": 0, "unchanged": 0, "observations": 0}
        unmapped: set[str] = set()
        with self.db.transaction() as conn:
            for p in periods:
                observations, missing = canonical.normalise_values(provider, p.values)
                unmapped.update(missing)
                if not observations:
                    continue
                end = _date(p.period_end)
                if end is None:
                    continue
                fiscal_year = p.fiscal_year if p.fiscal_year is not None else int(end[:4])
                row = conn.execute("SELECT id, vintage, filing_date FROM financial_periods WHERE security_id = ? AND provider = ? AND fiscal_period = ? AND period_end = ? "
                                   "ORDER BY vintage DESC LIMIT 1", (security_id, provider, p.fiscal_period, end)).fetchone()
                dates = (_date(p.filing_date), _date(p.accepted_date), _date(p.publication_date))
                if row is None:
                    period_id = self._insert_period(conn, security_id, provider, p, fiscal_year, end, dates, 0, "filed" if any(dates) else "estimated")
                    self._insert_observations(conn, period_id, provider, observations, p.reporting_currency, restated=False)
                    counts["periods"] += 1
                    counts["observations"] += len(observations)
                    continue
                period_id, vintage = int(row[0]), int(row[1])
                old = {r[0]: r[1] for r in conn.execute("SELECT metric_code, value FROM fundamental_observations WHERE financial_period_id = ?", (period_id,))}
                changed = any(code in old and not _close(old[code], o["value"]) for code, o in observations.items())
                if changed:
                    new_id = self._insert_period(conn, security_id, provider, p, fiscal_year, end, (dates[0], dates[1], retrieved), vintage + 1, "restated")
                    self._insert_observations(conn, new_id, provider, observations, p.reporting_currency, restated=True)
                    counts["vintages"] += 1
                    counts["observations"] += len(observations)
                else:
                    added = [(c, o) for c, o in observations.items() if c not in old]
                    if added:
                        self._insert_observations(conn, period_id, provider, dict(added), p.reporting_currency, restated=False)
                        counts["observations"] += len(added)
                    first = [d for d in (row[2], dates[0]) if d]
                    conn.execute("UPDATE financial_periods SET updated_at = ?, filing_date = COALESCE(?, filing_date), accepted_date = COALESCE(accepted_date, ?), "
                                 "publication_date = COALESCE(publication_date, ?) WHERE id = ?", (now_iso(), min(first) if first else None, dates[1], dates[2], period_id))
                    counts["unchanged"] += 1
        counts["unmapped"] = sorted(unmapped)
        return counts

    @staticmethod
    def _insert_period(conn, security_id, provider, p: PeriodStatement, fiscal_year, end, dates, vintage, pit_quality) -> int:
        stamp = now_iso()
        cur = conn.execute(
            "INSERT INTO financial_periods (security_id, fiscal_year, fiscal_period, period_start, period_end, filing_date, accepted_date, publication_date, reporting_currency, "
            "accounting_standard, statement_type, provider, provider_record_id, vintage, pit_quality, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (security_id, fiscal_year, p.fiscal_period, _date(p.period_start), end, dates[0], dates[1], dates[2], p.reporting_currency, p.accounting_standard, "combined", provider,
             p.provider_record_id, vintage, pit_quality, stamp, stamp))
        return int(cur.lastrowid)

    @staticmethod
    def _insert_observations(conn, period_id: int, provider: str, observations: dict, currency: str | None, restated: bool) -> None:
        stamp = now_iso()
        rows = []
        for code, o in observations.items():
            m = canonical.METRICS[code]
            unit = "currency" if m.kind in ("flow", "stock") else ("currency_per_share" if m.kind == "per_share" else "shares")
            rows.append((period_id, code, float(o["value"]), unit, currency if unit != "shares" else None, o["source_metric"], provider, 1 if restated else 0, o["flag"], stamp))
        conn.executemany("INSERT OR IGNORE INTO fundamental_observations (financial_period_id, metric_code, value, unit, currency, source_metric, provider, is_restated, quality_flag, created_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)

    def set_quality(self, updates: list[tuple[str, int, str]]) -> None:
        """``[(flag, period_id, metric_code)]``: flag individual observations (the data-quality engine's output)."""
        with self.db.transaction() as conn:
            conn.executemany("UPDATE fundamental_observations SET quality_flag = ? WHERE financial_period_id = ? AND metric_code = ?", updates)

    # ------------------------------------------------------------------------------------------------------------ sync state
    def get_sync(self, security_id: int, provider: str, request_type: str = "financials") -> dict | None:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM sync_state WHERE security_id = ? AND provider = ? AND request_type = ?", (int(security_id), provider, request_type)).fetchone()
        return dict(row) if row else None

    def set_sync(self, security_id: int, provider: str, request_type: str = "financials", **fields) -> None:
        allowed = {"last_checked", "last_success", "last_period_end", "earliest_period_end", "history_exhausted", "n_periods", "status", "error"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"unknown sync fields {sorted(bad)}")
        with self.db.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO sync_state (security_id, provider, request_type) VALUES (?,?,?)", (int(security_id), provider, request_type))
            if fields:
                conn.execute(f"UPDATE sync_state SET {', '.join(k + ' = ?' for k in fields)} WHERE security_id = ? AND provider = ? AND request_type = ?", [*fields.values(), int(security_id), provider, request_type])

    # ------------------------------------------------------------------------------------------------------------ what is held, and what was asked
    def held_periods(self, security_id: int, provider: str | None = None) -> list[dict]:
        """Every statement held for a company, oldest first: ``fiscal_year``, ``fiscal_period``, ``period_end``, ``filing_date``, ``provider``, ``vintage``. A restatement is a later vintage of a period."""
        sql = ("SELECT COALESCE(fiscal_year, CAST(substr(period_end, 1, 4) AS INTEGER)) AS fiscal_year, fiscal_period, period_end, filing_date, provider, vintage FROM financial_periods "
               "WHERE security_id = ?")
        args: list = [int(security_id)]
        if provider:
            sql += " AND provider = ?"
            args.append(provider)
        with self.db.transaction() as conn:
            rows = conn.execute(sql + " ORDER BY period_end, vintage", args).fetchall()
        return [dict(r) for r in rows]

    def log_fetch(self, security_id: int, provider: str, request_type: str, asked_from: int | None, asked_to: int | None, answer: dict, kept: int, new: int, cost: int,
                  reason: str = "", at: str | None = None) -> None:
        """Remember one request: the fiscal years it asked for (``None`` is open), what the answer held (``answer`` is :func:`coverage.summarise_answer` of it), how many annual statements this
        application read from it and how many of those were new, and what it cost. A year inside the window that came back empty is not asked for again unless the user says so."""
        detail = {"labels": answer.get("labels") or {}, "echoed": answer.get("echoed") or {}}
        with self.db.transaction() as conn:
            conn.execute("INSERT INTO fetch_log (security_id, provider, request_type, asked_from, asked_to, at, entries, kept, new, first_year, last_year, cost, reason, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (int(security_id), provider, request_type, asked_from, asked_to, at or datetime.now(timezone.utc).strftime("%Y-%m-%d"), int(answer.get("entries") or 0), int(kept), int(new),
                          answer.get("first_year"), answer.get("last_year"), int(cost), reason, json.dumps(detail, sort_keys=True)))

    def fetch_log(self, security_id: int, provider: str, request_type: str = "financials:annual", limit: int = 20) -> list[dict]:
        """The requests made for a company, newest first."""
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT * FROM fetch_log WHERE security_id = ? AND provider = ? AND request_type = ? ORDER BY id DESC LIMIT ?", (int(security_id), provider, request_type, int(limit))).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d["detail"]) if d.get("detail") else {}
            out.append(d)
        return out

    def raw_answers(self, security_id: int, provider: str, request_type: str, limit: int = 20) -> list[dict]:
        """The answers kept for a company, newest first, each with the parameters of the request that brought it and the day it was kept."""
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT id, request_parameters, response_payload, retrieved_at FROM raw_fundamental_payloads WHERE security_id = ? AND provider = ? AND request_type = ? "
                                "ORDER BY id DESC LIMIT ?", (int(security_id), provider, request_type, int(limit))).fetchall()
        out = []
        for r in rows:
            try:
                payload = json.loads(zlib.decompress(r[2]).decode())
                parameters = json.loads(r[1]) if r[1] else {}
            except (ValueError, zlib.error):
                continue
            out.append({"id": int(r[0]), "parameters": parameters, "payload": payload, "retrieved_at": r[3]})
        return out

    def period_summary(self, security_id: int, provider: str | None = None) -> dict:
        """What is stored for a company: how many periods, the earliest and latest period end, the latest filing date."""
        sql = ("SELECT COUNT(*) AS n, MIN(period_end) AS first_end, MAX(period_end) AS last_end, MAX(COALESCE(publication_date, accepted_date, filing_date)) AS last_filed, "
               "SUM(CASE WHEN fiscal_period = 'FY' THEN 1 ELSE 0 END) AS annual FROM financial_periods WHERE security_id = ?")
        args: list = [int(security_id)]
        if provider:
            sql += " AND provider = ?"
            args.append(provider)
        with self.db.transaction() as conn:
            row = conn.execute(sql, args).fetchone()
        return {k: row[k] for k in row.keys()}

    # ------------------------------------------------------------------------------------------------------------ reading for research
    def periods(self, security_ids: list[int] | None = None) -> pd.DataFrame:
        sql = ("SELECT p.id AS period_id, p.security_id, p.fiscal_year, p.fiscal_period, p.period_start, p.period_end, p.filing_date, p.accepted_date, p.publication_date, "
               "p.reporting_currency, p.provider, p.vintage, p.pit_quality FROM financial_periods p")
        args: list = []
        if security_ids is not None:
            sql += f" WHERE p.security_id IN ({','.join('?' * len(security_ids))})"
            args = [int(i) for i in security_ids]
        with self.db.transaction() as conn:
            frame = pd.read_sql_query(sql, conn, params=args)
        for c in ("period_start", "period_end", "filing_date", "accepted_date", "publication_date"):
            frame[c] = pd.to_datetime(frame[c])
        return frame

    def observations(self, security_ids: list[int] | None = None, metrics: list[str] | None = None) -> pd.DataFrame:
        """Long frame: ``period_id, security_id, metric_code, value, quality_flag, provider, is_restated, currency, source_metric`` (the periods come from :meth:`periods`)."""
        sql = ("SELECT o.financial_period_id AS period_id, p.security_id, o.metric_code, o.value, o.quality_flag, o.provider, o.is_restated, o.currency, o.source_metric "
               "FROM fundamental_observations o JOIN financial_periods p ON p.id = o.financial_period_id WHERE 1=1")
        args: list = []
        if security_ids is not None:
            sql += f" AND p.security_id IN ({','.join('?' * len(security_ids))})"
            args += [int(i) for i in security_ids]
        if metrics is not None:
            sql += f" AND o.metric_code IN ({','.join('?' * len(metrics))})"
            args += list(metrics)
        with self.db.transaction() as conn:
            return pd.read_sql_query(sql, conn, params=args)

    # ------------------------------------------------------------------------------------------------------------ prices and FX
    def store_prices(self, frame: pd.DataFrame) -> int:
        """``frame``: columns ``security_id, date, adj_close`` and optionally ``close`` (unadjusted, what a market capitalisation is built from) and ``volume`` (shares). Existing rows are replaced."""
        need = {"security_id", "date", "adj_close"}
        if not need <= set(frame.columns):
            raise FundamentalsError(f"prices need the columns {sorted(need)}")
        blank = pd.Series(np.nan, index=frame.index)
        close = frame["close"] if "close" in frame.columns else blank
        vol = frame["volume"] if "volume" in frame.columns else blank

        def opt(x):
            return None if pd.isna(x) else float(x)

        rows = [(int(s), pd.Timestamp(d).strftime("%Y-%m-%d"), float(p), opt(c), opt(v))
                for s, d, p, c, v in zip(frame["security_id"], frame["date"], frame["adj_close"], close, vol) if np.isfinite(p) and p > 0]
        with self.db.transaction() as conn:
            conn.executemany("INSERT OR REPLACE INTO prices_daily (security_id, date, adj_close, close, volume) VALUES (?,?,?,?,?)", rows)
        return len(rows)

    def _price_field(self, field: str, security_ids: list[int], start: str | None, end: str | None) -> pd.DataFrame:
        if not security_ids:
            return pd.DataFrame()
        sql = f"SELECT security_id, date, {field} AS v FROM prices_daily WHERE {field} IS NOT NULL AND security_id IN ({','.join('?' * len(security_ids))})"
        args: list = [int(i) for i in security_ids]
        if start:
            sql += " AND date >= ?"
            args.append(_date(start))
        if end:
            sql += " AND date <= ?"
            args.append(_date(end))
        with self.db.transaction() as conn:
            long = pd.read_sql_query(sql, conn, params=args)
        if long.empty:
            return pd.DataFrame()
        wide = long.pivot(index="date", columns="security_id", values="v")
        wide.index = pd.DatetimeIndex(wide.index)
        return wide.sort_index()

    def prices(self, security_ids: list[int], start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Adjusted closes (total-return basis, what returns are computed from), dates by ``security_id``."""
        return self._price_field("adj_close", security_ids, start, end)

    def closes(self, security_ids: list[int], start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Unadjusted closes, the basis of a market capitalisation (empty where only adjusted prices were stored)."""
        return self._price_field("close", security_ids, start, end)

    def volumes(self, security_ids: list[int], start: str | None = None, end: str | None = None) -> pd.DataFrame:
        return self._price_field("volume", security_ids, start, end)

    def store_fx(self, frame: pd.DataFrame, source: str = "") -> int:
        """``frame``: columns ``date, base, quote, rate`` (units of ``quote`` per one ``base``)."""
        rows = [(pd.Timestamp(d).strftime("%Y-%m-%d"), str(b).upper(), str(q).upper(), float(r), source) for d, b, q, r in zip(frame["date"], frame["base"], frame["quote"], frame["rate"]) if np.isfinite(r) and r > 0]
        with self.db.transaction() as conn:
            conn.executemany("INSERT OR REPLACE INTO fx_rates (date, base, quote, rate, source) VALUES (?,?,?,?,?)", rows)
        return len(rows)

    def fx(self) -> pd.DataFrame:
        with self.db.transaction() as conn:
            frame = pd.read_sql_query("SELECT date, base, quote, rate FROM fx_rates", conn)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    # ------------------------------------------------------------------------------------------------------------ reporting (Data Manager)
    def counts(self) -> dict:
        with self.db.transaction() as conn:
            one = lambda sql: int(conn.execute(sql).fetchone()[0])  # noqa: E731
            return {"companies": one("SELECT COUNT(*) FROM securities"), "markets": one("SELECT COUNT(DISTINCT market_id) FROM securities WHERE market_id IS NOT NULL"),
                    "countries": one("SELECT COUNT(DISTINCT country) FROM securities WHERE country IS NOT NULL"),
                    "with_financials": one("SELECT COUNT(DISTINCT security_id) FROM financial_periods"),
                    "annual_periods": one("SELECT COUNT(*) FROM financial_periods WHERE fiscal_period = 'FY'"),
                    "quarterly_periods": one("SELECT COUNT(*) FROM financial_periods WHERE fiscal_period <> 'FY'"),
                    "observations": one("SELECT COUNT(*) FROM fundamental_observations"), "derived_metrics": one("SELECT COUNT(*) FROM fundamental_metrics"),
                    "restated_vintages": one("SELECT COUNT(*) FROM financial_periods WHERE vintage > 0"), "raw_payloads": one("SELECT COUNT(*) FROM raw_fundamental_payloads"),
                    "price_rows": one("SELECT COUNT(*) FROM prices_daily"), "alphas": one("SELECT COUNT(*) FROM alpha_library"), "rejected": one("SELECT COUNT(*) FROM alpha_graveyard")}

    def coverage(self) -> pd.DataFrame:
        """Per country: companies, and how many have annual and quarterly statements, with the median years of history. Computed from what is in the database."""
        with self.db.transaction() as conn:
            frame = pd.read_sql_query(
                "SELECT s.security_id, COALESCE(s.country, '??') AS country, "
                "SUM(CASE WHEN p.fiscal_period = 'FY' THEN 1 ELSE 0 END) AS annual, SUM(CASE WHEN p.fiscal_period <> 'FY' THEN 1 ELSE 0 END) AS quarterly, "
                "MIN(p.period_end) AS first_end, MAX(p.period_end) AS last_end "
                "FROM securities s LEFT JOIN financial_periods p ON p.security_id = s.security_id AND p.vintage = 0 GROUP BY s.security_id", conn)
        if frame.empty:
            return pd.DataFrame(columns=["country", "companies", "annual_coverage", "quarterly_coverage", "median_history_years"])
        years = (pd.to_datetime(frame["last_end"]) - pd.to_datetime(frame["first_end"])).dt.days / 365.25
        frame = frame.assign(years=years)
        out = frame.groupby("country").agg(companies=("security_id", "count"), annual=("annual", lambda x: (x > 0).mean()), quarterly=("quarterly", lambda x: (x > 0).mean()),
                                          median_history_years=("years", "median")).reset_index()
        return out.rename(columns={"annual": "annual_coverage", "quarterly": "quarterly_coverage"}).sort_values("companies", ascending=False).reset_index(drop=True)

    def metric_coverage(self) -> pd.DataFrame:
        """For each canonical metric: how many companies and periods have it, and the share of annual periods that do."""
        with self.db.transaction() as conn:
            frame = pd.read_sql_query(
                "SELECT o.metric_code, COUNT(DISTINCT p.security_id) AS companies, COUNT(*) AS observations FROM fundamental_observations o "
                "JOIN financial_periods p ON p.id = o.financial_period_id GROUP BY o.metric_code", conn)
            total = int(conn.execute("SELECT COUNT(DISTINCT security_id) FROM financial_periods").fetchone()[0])
        frame["company_share"] = frame["companies"] / max(total, 1)
        return frame.sort_values("company_share", ascending=False).reset_index(drop=True)

    def quality_counts(self) -> dict:
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT quality_flag, COUNT(*) FROM fundamental_observations GROUP BY quality_flag").fetchall()
            pit = conn.execute("SELECT pit_quality, COUNT(*) FROM financial_periods GROUP BY pit_quality").fetchall()
        return {"observations": {r[0]: int(r[1]) for r in rows}, "periods": {r[0]: int(r[1]) for r in pit}}

    def inspect(self, security_id: int, metric: str | None = None) -> pd.DataFrame:
        """Every stored observation of a company with the period, dates, provider, source field and flags: the auditing view behind any number."""
        sql = ("SELECT p.fiscal_year, p.fiscal_period, p.period_start, p.period_end, p.filing_date, p.accepted_date, p.publication_date, p.pit_quality, p.vintage, p.reporting_currency, "
               "o.metric_code, o.value, o.unit, o.currency, o.source_metric, o.provider, o.is_restated, o.quality_flag FROM fundamental_observations o "
               "JOIN financial_periods p ON p.id = o.financial_period_id WHERE p.security_id = ?")
        args: list = [int(security_id)]
        if metric:
            sql += " AND o.metric_code = ?"
            args.append(metric)
        with self.db.transaction() as conn:
            return pd.read_sql_query(sql + " ORDER BY p.period_end DESC, p.fiscal_period, o.metric_code", conn, params=args)


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= RESTATEMENT_TOLERANCE * max(abs(a), abs(b), 1e-12)


__all__ = ["CompanyProfile", "PeriodStatement", "FundamentalsStore", "DEFAULT_PATH"]
