"""The point-in-time store: every stored figure with when it became available, as Parquet files queried with DuckDB.

SQLite remains the source of truth (the providers, the importer and the research engine all write and read it); this module is the analytic, reproducible *view* of it. ``export`` writes one long table (a
row per company, period, vintage and line) to Parquet, partitioned by country and fiscal year, with the columns research needs to answer "what was known on day D":

* the figure as stored (base units), its currency, its quality flag, whether it is a restatement and its ``vintage`` (0 is the original; a restatement is a later vintage, never an overwrite);
* ``filing_date``, ``accepted_date``, ``publication_date`` as the source gave them, and ``available_at``: the day it could be used, computed by the same function as the research engine
  (:func:`pit.availability`: the earliest known date plus the information lag; for a restatement the day it was first seen; a conservative lag after the period end, flagged ``estimated``, when no date is known);
* the provenance of the figure where an import recorded one (file or link, accession, page, taxonomy tag, the figure as reported, extraction method and status).

``PitStore.as_of(day)`` returns, for each company, period and line, the figure that was public on ``day``: the latest vintage available by then, from the provider that ranks first. A manifest next to the files
records when and from what the export was made and the availability settings used, so a result can be reproduced. DuckDB and PyArrow are optional (``pip install ".[fundamentals-extras]"``).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .errors import FundamentalsError
from .pit import PitSettings, availability

SCHEMA = 1
COLUMNS = ["security_id", "ticker", "exchange", "country", "sector", "fiscal_year", "fiscal_period", "period_start", "period_end", "provider", "vintage", "metric_code", "value", "currency", "quality_flag",
           "is_restated", "pit_quality", "filing_date", "accepted_date", "publication_date", "available_at", "availability_estimated", "source_metric", "source_ref", "accession", "page", "taxonomy_tag",
           "reported_value", "reported_unit", "extraction_method", "extraction_status", "edited"]


def _need():
    try:
        import duckdb
        import pyarrow  # noqa: F401
    except ImportError:
        raise FundamentalsError("the point-in-time store needs DuckDB and PyArrow (pip install duckdb pyarrow, or pip install \".[fundamentals-extras]\")") from None
    return duckdb


def frame(store, settings: PitSettings | None = None) -> pd.DataFrame:
    """The long point-in-time table of a store, as a DataFrame (see the module docstring for the columns)."""
    settings = settings or PitSettings()
    periods, obs = store.periods(), store.observations()
    if periods.empty or obs.empty:
        return pd.DataFrame(columns=COLUMNS)
    available, estimated = availability(periods, settings)
    periods = periods.assign(available_at=available, availability_estimated=estimated)
    secs = store.securities()[["security_id", "ticker", "exchange", "country", "sector"]]
    with store.db.transaction() as conn:
        prov = pd.read_sql_query("SELECT financial_period_id AS period_id, metric_code, source_ref, accession, page, taxonomy_tag, reported_value, reported_unit, extraction_method, extraction_status, edited "
                                 "FROM observation_provenance", conn)
    df = obs.drop(columns=["provider"]).merge(periods.drop(columns=["security_id"]), on="period_id", how="inner").merge(secs, on="security_id", how="left")
    df = df.merge(prov.drop_duplicates(["period_id", "metric_code"], keep="last"), on=["period_id", "metric_code"], how="left")
    df["fiscal_year"] = df["fiscal_year"].fillna(df["period_end"].dt.year).astype(int)
    df["country"] = df["country"].fillna("NA")
    df["is_restated"] = df["is_restated"].astype(bool)
    return df[[c for c in COLUMNS if c in df.columns]].reset_index(drop=True)


def export(store, out_dir: str | Path, settings: PitSettings | None = None) -> dict:
    """Write the store's point-in-time table to ``out_dir`` as Parquet (partitioned by country and fiscal year) with a manifest; returns the manifest."""
    _need()
    import pyarrow as pa
    import pyarrow.parquet as pq
    settings = settings or PitSettings()
    out = Path(out_dir)
    data = out / "observations"
    out.mkdir(parents=True, exist_ok=True)
    df = frame(store, settings)
    if data.exists():
        for f in data.rglob("*.parquet"):
            f.unlink()
    if len(df):
        for c in ("period_start", "period_end", "filing_date", "accepted_date", "publication_date", "available_at"):
            df[c] = pd.to_datetime(df[c])
        pq.write_to_dataset(pa.Table.from_pandas(df, preserve_index=False), root_path=str(data), partition_cols=["country", "fiscal_year"], existing_data_behavior="delete_matching")
    db_path = getattr(store.db, "path", None)
    manifest = {"schema": SCHEMA, "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "rows": int(len(df)), "securities": int(df["security_id"].nunique()) if len(df) else 0,
                "source_db": str(db_path) if db_path else None, "source_mtime": os.path.getmtime(db_path) if db_path and os.path.exists(db_path) else None, "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(settings).items()},
                "columns": COLUMNS, "partitions": sorted({f"{c}/{y}" for c, y in zip(df["country"], df["fiscal_year"])}) if len(df) else []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


class PitStore:
    def __init__(self, directory: str | Path, priority: tuple = PitSettings().provider_priority):
        self.dir = Path(directory)
        self.priority = tuple(priority)
        if not (self.dir / "manifest.json").exists():
            raise FundamentalsError(f"no point-in-time export in {self.dir}: run the export first")
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self._con = None

    def con(self):
        if self._con is None:
            duckdb = _need()
            self._con = duckdb.connect()
            glob = str(self.dir / "observations" / "**" / "*.parquet").replace("'", "''")
            if self.manifest["rows"]:
                self._con.execute(f"CREATE VIEW obs AS SELECT * FROM read_parquet('{glob}', hive_partitioning = true, union_by_name = true)")
            else:
                self._con.execute("CREATE VIEW obs AS SELECT NULL::BIGINT security_id WHERE false")
        return self._con

    def _rank_sql(self) -> str:
        cases = " ".join(f"WHEN provider = '{p}' THEN {i}" for i, p in enumerate(self.priority))
        return f"CASE {cases} ELSE {len(self.priority)} END"

    def as_of(self, day: str | date, security_ids: list[int] | None = None, metrics: list[str] | None = None, countries: list[str] | None = None, include_estimated: bool = True) -> pd.DataFrame:
        """For each company, period and line, the figure that was public on ``day``: the highest-ranked provider's latest vintage available by then."""
        if not self.manifest["rows"]:
            return pd.DataFrame(columns=COLUMNS)
        where, args = ["available_at <= ?"], [pd.Timestamp(day).to_pydatetime()]
        if security_ids:
            where.append(f"security_id IN ({','.join('?' * len(security_ids))})")
            args += [int(i) for i in security_ids]
        if metrics:
            where.append(f"metric_code IN ({','.join('?' * len(metrics))})")
            args += list(metrics)
        if countries:
            where.append(f"country IN ({','.join('?' * len(countries))})")
            args += list(countries)
        if not include_estimated:
            where.append("NOT availability_estimated")
        sql = (f"SELECT * EXCLUDE (rn, rank) FROM (SELECT *, {self._rank_sql()} AS rank, ROW_NUMBER() OVER (PARTITION BY security_id, fiscal_period, period_end, metric_code "
               f"ORDER BY {self._rank_sql()}, vintage DESC, available_at DESC) AS rn FROM obs WHERE {' AND '.join(where)}) WHERE rn = 1 ORDER BY security_id, period_end, fiscal_period, metric_code")
        return self.con().execute(sql, args).df()

    def vintages(self, security_id: int, metric: str) -> pd.DataFrame:
        """Every vintage of one line of one company with its availability date: the original and each restatement, side by side."""
        if not self.manifest["rows"]:
            return pd.DataFrame(columns=COLUMNS)
        return self.con().execute("SELECT fiscal_period, period_end, provider, vintage, value, is_restated, filing_date, available_at, availability_estimated, accession, extraction_method FROM obs "
                                  "WHERE security_id = ? AND metric_code = ? ORDER BY period_end, fiscal_period, provider, vintage", [int(security_id), metric]).df()

    def summary(self) -> pd.DataFrame:
        """Rows, companies and the first and latest availability by country and provider."""
        if not self.manifest["rows"]:
            return pd.DataFrame()
        return self.con().execute("SELECT country, provider, COUNT(*) AS rows, COUNT(DISTINCT security_id) AS companies, MIN(period_end) AS first_period, MAX(period_end) AS last_period, MAX(available_at) AS last_available, "
                                  "SUM(CASE WHEN availability_estimated THEN 1 ELSE 0 END) AS estimated FROM obs GROUP BY country, provider ORDER BY country, provider").df()
