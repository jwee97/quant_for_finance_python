"""The fundamentals database: one SQLite file (``data/user/fundamentals.db``, git-ignored, persistent) and the tables in it.

The design rules are the ones the research depends on:

* **The database is the source of truth, the providers are not.** Everything a provider returns is normalised and stored; research reads only the database.
* **A security is a ``security_id``**, never a bare ticker: ``(ticker, exchange)`` is unique, and every provider's own identifier is kept in ``security_identifiers``.
* **A financial period remembers when it became public** (``filing_date``, ``accepted_date``, ``publication_date``) and how sure we are of it (``pit_quality``): ``filed`` when a provider gave a
  real date, ``estimated`` when only the period end is known and a conservative lag is assumed. A restated figure is a new ``vintage`` of the same period, public from the day we first saw it.
* **Observations are long and canonical**: one row per period and canonical metric code, with the source field it came from and a quality flag. Derived ratios that many studies reuse are kept in
  ``fundamental_metrics`` with the version of the formula; experimental factor matrices are never stored.
* **Every external request is counted** (``api_usage``), what each one asked for and what came back is kept (``fetch_log``: so that a year already asked for and not found is not paid for again),
  every raw response kept once (``raw_fundamental_payloads``, deduplicated by hash), and every tested alpha remembered
  (``alpha_library`` for survivors, ``alpha_graveyard`` for the rest) so the same idea is not paid for twice.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS markets (
    market_id INTEGER PRIMARY KEY, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL, country TEXT, region TEXT, currency TEXT, timezone TEXT);

CREATE TABLE IF NOT EXISTS securities (
    security_id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, exchange TEXT NOT NULL DEFAULT '', market_id INTEGER REFERENCES markets(market_id),
    company_name TEXT, country TEXT, currency TEXT, sector TEXT, industry TEXT, isin TEXT, figi TEXT, lei TEXT, provider_symbol TEXT, source TEXT,
    active INTEGER NOT NULL DEFAULT 1, delisted_date TEXT, first_seen TEXT, last_seen TEXT,
    UNIQUE (ticker, exchange));
CREATE INDEX IF NOT EXISTS ix_securities_country ON securities(country);
CREATE INDEX IF NOT EXISTS ix_securities_sector ON securities(sector, industry);
CREATE INDEX IF NOT EXISTS ix_securities_market ON securities(market_id);
CREATE INDEX IF NOT EXISTS ix_securities_symbol ON securities(provider_symbol);

CREATE TABLE IF NOT EXISTS security_identifiers (
    security_id INTEGER NOT NULL REFERENCES securities(security_id) ON DELETE CASCADE, provider TEXT NOT NULL, id_type TEXT NOT NULL, provider_id TEXT NOT NULL,
    PRIMARY KEY (provider, id_type, provider_id));
CREATE INDEX IF NOT EXISTS ix_identifiers_security ON security_identifiers(security_id);

CREATE TABLE IF NOT EXISTS financial_periods (
    id INTEGER PRIMARY KEY, security_id INTEGER NOT NULL REFERENCES securities(security_id) ON DELETE CASCADE,
    fiscal_year INTEGER, fiscal_period TEXT NOT NULL, period_start TEXT, period_end TEXT NOT NULL,
    filing_date TEXT, accepted_date TEXT, publication_date TEXT,
    reporting_currency TEXT, accounting_standard TEXT, statement_type TEXT NOT NULL DEFAULT 'combined',
    provider TEXT NOT NULL, provider_record_id TEXT, vintage INTEGER NOT NULL DEFAULT 0, pit_quality TEXT NOT NULL DEFAULT 'filed',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE (security_id, provider, fiscal_period, period_end, vintage));
CREATE INDEX IF NOT EXISTS ix_periods_security ON financial_periods(security_id, period_end);
CREATE INDEX IF NOT EXISTS ix_periods_filing ON financial_periods(filing_date);
CREATE INDEX IF NOT EXISTS ix_periods_end ON financial_periods(period_end);

CREATE TABLE IF NOT EXISTS fundamental_observations (
    id INTEGER PRIMARY KEY, financial_period_id INTEGER NOT NULL REFERENCES financial_periods(id) ON DELETE CASCADE,
    metric_code TEXT NOT NULL, value REAL NOT NULL, unit TEXT, currency TEXT, source_metric TEXT, provider TEXT,
    is_restated INTEGER NOT NULL DEFAULT 0, quality_flag TEXT NOT NULL DEFAULT 'ok', created_at TEXT NOT NULL,
    UNIQUE (financial_period_id, metric_code));
CREATE INDEX IF NOT EXISTS ix_obs_metric ON fundamental_observations(metric_code, financial_period_id);

CREATE TABLE IF NOT EXISTS raw_fundamental_payloads (
    id INTEGER PRIMARY KEY, provider TEXT NOT NULL, security_id INTEGER REFERENCES securities(security_id) ON DELETE SET NULL, request_type TEXT NOT NULL,
    request_parameters TEXT, response_payload BLOB NOT NULL, response_hash TEXT NOT NULL, retrieved_at TEXT NOT NULL,
    UNIQUE (provider, response_hash));
CREATE INDEX IF NOT EXISTS ix_raw_security ON raw_fundamental_payloads(security_id, request_type);

CREATE TABLE IF NOT EXISTS fundamental_metrics (
    security_id INTEGER NOT NULL REFERENCES securities(security_id) ON DELETE CASCADE, financial_period_id INTEGER NOT NULL REFERENCES financial_periods(id) ON DELETE CASCADE,
    basis TEXT NOT NULL, metric_name TEXT NOT NULL, metric_value REAL, calculation_version TEXT NOT NULL, calculated_at TEXT NOT NULL,
    PRIMARY KEY (financial_period_id, basis, metric_name, calculation_version));
CREATE INDEX IF NOT EXISTS ix_metrics_name ON fundamental_metrics(metric_name, basis);
CREATE INDEX IF NOT EXISTS ix_metrics_security ON fundamental_metrics(security_id);

CREATE TABLE IF NOT EXISTS api_usage (
    id INTEGER PRIMARY KEY, provider TEXT NOT NULL, date TEXT NOT NULL, endpoint TEXT, requests INTEGER NOT NULL DEFAULT 1, credits_used INTEGER NOT NULL DEFAULT 1,
    success INTEGER NOT NULL, symbol TEXT, market TEXT, timestamp TEXT NOT NULL, key_slot TEXT, key_id TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS ix_usage_day ON api_usage(provider, date);
CREATE INDEX IF NOT EXISTS ix_usage_key ON api_usage(provider, key_id, date);

CREATE TABLE IF NOT EXISTS sync_state (
    security_id INTEGER NOT NULL REFERENCES securities(security_id) ON DELETE CASCADE, provider TEXT NOT NULL, request_type TEXT NOT NULL,
    last_checked TEXT, last_success TEXT, last_period_end TEXT, earliest_period_end TEXT, history_exhausted INTEGER NOT NULL DEFAULT 0,
    n_periods INTEGER NOT NULL DEFAULT 0, status TEXT, error TEXT,
    PRIMARY KEY (security_id, provider, request_type));

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY, security_id INTEGER NOT NULL REFERENCES securities(security_id) ON DELETE CASCADE, provider TEXT NOT NULL, request_type TEXT NOT NULL,
    asked_from INTEGER, asked_to INTEGER, at TEXT NOT NULL, entries INTEGER NOT NULL DEFAULT 0, kept INTEGER NOT NULL DEFAULT 0, new INTEGER NOT NULL DEFAULT 0,
    first_year INTEGER, last_year INTEGER, cost INTEGER NOT NULL DEFAULT 0, reason TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS ix_fetch_log ON fetch_log(security_id, provider, request_type);

CREATE TABLE IF NOT EXISTS fx_rates (date TEXT NOT NULL, base TEXT NOT NULL, quote TEXT NOT NULL, rate REAL NOT NULL, source TEXT, PRIMARY KEY (date, base, quote));

CREATE TABLE IF NOT EXISTS prices_daily (
    security_id INTEGER NOT NULL REFERENCES securities(security_id) ON DELETE CASCADE, date TEXT NOT NULL, adj_close REAL NOT NULL, close REAL, volume REAL,
    PRIMARY KEY (security_id, date)) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS alpha_library (
    alpha_id TEXT PRIMARY KEY, name TEXT NOT NULL, expression TEXT NOT NULL, expression_hash TEXT NOT NULL, settings_hash TEXT NOT NULL, settings TEXT NOT NULL,
    description TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, universe TEXT, rebalance_frequency TEXT, forward_horizon INTEGER,
    ic REAL, icir REAL, sharpe REAL, sortino REAL, max_drawdown REAL, turnover REAL, oos_ic REAL, oos_sharpe REAL, complexity REAL, fitness REAL, q_value REAL, deflated_sharpe REAL,
    status TEXT NOT NULL, status_reason TEXT, metrics TEXT, UNIQUE (expression_hash, settings_hash));
CREATE INDEX IF NOT EXISTS ix_alpha_status ON alpha_library(status);

CREATE TABLE IF NOT EXISTS alpha_graveyard (
    id INTEGER PRIMARY KEY, expression_hash TEXT NOT NULL, settings_hash TEXT NOT NULL, expression TEXT NOT NULL, settings TEXT, reason_rejected TEXT NOT NULL, stage TEXT,
    ic REAL, oos_ic REAL, sharpe REAL, date_tested TEXT NOT NULL, run_id TEXT, UNIQUE (expression_hash, settings_hash));

CREATE TABLE IF NOT EXISTS alpha_status_log (
    id INTEGER PRIMARY KEY, alpha_id TEXT NOT NULL REFERENCES alpha_library(alpha_id) ON DELETE CASCADE, at TEXT NOT NULL, from_status TEXT, to_status TEXT NOT NULL, reason TEXT, forced INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_status_log_alpha ON alpha_status_log(alpha_id, at);

CREATE TABLE IF NOT EXISTS alpha_monitor (
    id INTEGER PRIMARY KEY, alpha_id TEXT NOT NULL REFERENCES alpha_library(alpha_id) ON DELETE CASCADE, checked_at TEXT NOT NULL, as_of TEXT, status TEXT NOT NULL,
    recent_ic REAL, historical_ic REAL, recent_sharpe REAL, recent_spread REAL, coverage REAL, turnover REAL, detail TEXT);
CREATE INDEX IF NOT EXISTS ix_monitor_alpha ON alpha_monitor(alpha_id, checked_at);

CREATE TABLE IF NOT EXISTS discovery_runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, spec TEXT NOT NULL, stage_counts TEXT, error TEXT);

CREATE TABLE IF NOT EXISTS discovery_candidates (
    run_id TEXT NOT NULL REFERENCES discovery_runs(run_id) ON DELETE CASCADE, expression_hash TEXT NOT NULL, expression TEXT NOT NULL, stage_reached INTEGER NOT NULL,
    status TEXT NOT NULL, reason TEXT, metrics TEXT, PRIMARY KEY (run_id, expression_hash));

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL, spec TEXT NOT NULL, data_version TEXT, code_version TEXT, summary TEXT, status TEXT NOT NULL);
"""

# a short list that gives a region to every market a provider names; anything else is added when a provider reports it
MARKETS = [
    ("NYSE", "New York Stock Exchange", "US", "North America", "USD", "America/New_York"), ("NASDAQ", "Nasdaq", "US", "North America", "USD", "America/New_York"),
    ("TSX", "Toronto Stock Exchange", "CA", "North America", "CAD", "America/Toronto"), ("LSE", "London Stock Exchange", "GB", "Europe", "GBP", "Europe/London"),
    ("XETRA", "Deutsche Boerse Xetra", "DE", "Europe", "EUR", "Europe/Berlin"), ("EPA", "Euronext Paris", "FR", "Europe", "EUR", "Europe/Paris"),
    ("AMS", "Euronext Amsterdam", "NL", "Europe", "EUR", "Europe/Amsterdam"), ("BME", "Bolsa de Madrid", "ES", "Europe", "EUR", "Europe/Madrid"),
    ("MIL", "Borsa Italiana", "IT", "Europe", "EUR", "Europe/Rome"), ("SIX", "SIX Swiss Exchange", "CH", "Europe", "CHF", "Europe/Zurich"),
    ("STO", "Nasdaq Stockholm", "SE", "Europe", "SEK", "Europe/Stockholm"), ("CPH", "Nasdaq Copenhagen", "DK", "Europe", "DKK", "Europe/Copenhagen"),
    ("OSL", "Oslo Bors", "NO", "Europe", "NOK", "Europe/Oslo"), ("HEL", "Nasdaq Helsinki", "FI", "Europe", "EUR", "Europe/Helsinki"),
    ("TSE", "Tokyo Stock Exchange", "JP", "Asia-Pacific", "JPY", "Asia/Tokyo"), ("HKEX", "Hong Kong Exchanges", "HK", "Asia-Pacific", "HKD", "Asia/Hong_Kong"),
    ("SGX", "Singapore Exchange", "SG", "Asia-Pacific", "SGD", "Asia/Singapore"), ("ASX", "Australian Securities Exchange", "AU", "Asia-Pacific", "AUD", "Australia/Sydney"),
    ("NZX", "New Zealand Exchange", "NZ", "Asia-Pacific", "NZD", "Pacific/Auckland"), ("KRX", "Korea Exchange", "KR", "Asia-Pacific", "KRW", "Asia/Seoul"),
    ("NSE", "National Stock Exchange of India", "IN", "Asia-Pacific", "INR", "Asia/Kolkata"), ("SSE", "Shanghai Stock Exchange", "CN", "Asia-Pacific", "CNY", "Asia/Shanghai"),
    ("TWSE", "Taiwan Stock Exchange", "TW", "Asia-Pacific", "TWD", "Asia/Taipei"), ("B3", "B3 Brasil Bolsa Balcao", "BR", "Latin America", "BRL", "America/Sao_Paulo"),
    ("BMV", "Bolsa Mexicana de Valores", "MX", "Latin America", "MXN", "America/Mexico_City"), ("JSE", "Johannesburg Stock Exchange", "ZA", "Africa", "ZAR", "Africa/Johannesburg"),
]

# countries the application calls developed markets for its "Global Developed" universe
DEVELOPED = {"US", "CA", "GB", "DE", "FR", "NL", "ES", "IT", "CH", "SE", "DK", "NO", "FI", "BE", "AT", "PT", "IE", "LU", "JP", "HK", "SG", "AU", "NZ", "IL"}
REGIONS = {"US": "North America", "CA": "North America", "GB": "Europe", "DE": "Europe", "FR": "Europe", "NL": "Europe", "ES": "Europe", "IT": "Europe", "CH": "Europe", "SE": "Europe",
           "DK": "Europe", "NO": "Europe", "FI": "Europe", "BE": "Europe", "AT": "Europe", "PT": "Europe", "IE": "Europe", "LU": "Europe", "PL": "Europe", "TR": "Europe",
           "JP": "Asia-Pacific", "HK": "Asia-Pacific", "SG": "Asia-Pacific", "AU": "Asia-Pacific", "NZ": "Asia-Pacific", "KR": "Asia-Pacific", "IN": "Asia-Pacific",
           "CN": "Asia-Pacific", "TW": "Asia-Pacific", "BR": "Latin America", "MX": "Latin America", "CO": "Latin America", "PE": "Latin America", "ZA": "Africa", "IL": "Middle East"}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: str | Path, timeout: float = 30.0) -> sqlite3.Connection:
    """A connection with the settings the application relies on (foreign keys, write-ahead logging so the dashboard can read while a download writes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -65536")
    return conn


def initialise(conn: sqlite3.Connection) -> None:
    """Create the tables (idempotent), seed the market list and record the schema version."""
    conn.executescript(DDL)
    conn.executemany("INSERT OR IGNORE INTO markets (code, name, country, region, currency, timezone) VALUES (?, ?, ?, ?, ?, ?)", MARKETS)
    conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('created_at', ?)", (now_iso(),))
    conn.commit()
    found = int(conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0])
    if found > SCHEMA_VERSION:
        raise RuntimeError(f"the fundamentals database was written by a newer version of the application (schema {found}, this code knows {SCHEMA_VERSION})")


class Database:
    """A path to the fundamentals database and a way to open connections to it (one per use: SQLite connections are not shared between threads)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._ready = False

    def connect(self) -> sqlite3.Connection:
        conn = connect(self.path)
        if not self._ready:
            initialise(conn)
            self._ready = True
        return conn

    def transaction(self):
        """``with db.transaction() as conn:`` commits on success, rolls back on an error, and always closes."""
        return _Transaction(self)

    def size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self.path) + suffix)
            if p.exists():
                total += p.stat().st_size
        return total


class _Transaction:
    def __init__(self, db: Database):
        self.db, self.conn = db, None

    def __enter__(self) -> sqlite3.Connection:
        self.conn = self.db.connect()
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()
        return False
