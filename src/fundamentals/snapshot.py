"""A fingerprint of the research data, so a result can say exactly what it was computed on and a later run can tell whether the data has changed.

The earlier data version was three counts (companies, observations, price rows): a corrected figure, a restatement or a different source choice changes none of them, so two runs on different data
looked identical. The fingerprint hashes the content that research reads: every observation (period, vintage, filing date, metric, value, quality flag), every confirmed custom observation, the
source preferences, and the prices and exchange rates (row counts and sums). ``check`` compares a recorded version with the data now.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .schema import now_iso


def fingerprint(store) -> dict:
    with store.db.transaction() as c:
        obs = pd.read_sql_query("SELECT p.security_id, p.fiscal_period, p.period_end, p.vintage, p.filing_date, p.provider, o.metric_code, ROUND(o.value, 6) AS value, o.quality_flag "
                                "FROM fundamental_observations o JOIN financial_periods p ON p.id = o.financial_period_id ORDER BY 1, 2, 3, 4, 6, 7", c)
        cust = pd.read_sql_query("SELECT security_id, metric_code, period_end, vintage, filing_date, ROUND(value, 6) AS value, currency, scope FROM custom_observations WHERE superseded = 0 ORDER BY 1, 2, 3, 4", c)
        pref = pd.read_sql_query("SELECT * FROM source_preferences ORDER BY 1, 2, 3", c)
        px = c.execute("SELECT COUNT(*), ROUND(COALESCE(SUM(adj_close), 0), 4), ROUND(COALESCE(SUM(COALESCE(close, 0)), 0), 4) FROM prices_daily").fetchone()
        fx = c.execute("SELECT COUNT(*), ROUND(COALESCE(SUM(rate), 0), 6) FROM fx_rates").fetchone()
    parts = {"observations": int(pd.util.hash_pandas_object(obs, index=False).sum()) if len(obs) else 0, "custom": int(pd.util.hash_pandas_object(cust, index=False).sum()) if len(cust) else 0,
             "preferences": int(pd.util.hash_pandas_object(pref, index=False).sum()) if len(pref) else 0, "prices": list(px), "fx": list(fx)}
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
    return {"hash": digest, "observations": len(obs), "custom_observations": len(cust), "price_rows": px[0], "fx_rows": fx[0], "parts": parts}


def version(store) -> str:
    """The data version recorded with an experiment: counts, the content hash and the day."""
    f = fingerprint(store)
    n = store.counts()
    return f"{n['companies']}c/{f['observations']}o/{f['price_rows']}p#{f['hash']}@{now_iso()[:10]}"


def check(store, recorded: str | None) -> dict:
    """Is the data what an experiment ran on? ``{"same": bool, "recorded": ..., "now": ..., "reason": ...}``."""
    now = fingerprint(store)["hash"]
    if not recorded or "#" not in recorded:
        return {"same": None, "recorded": recorded, "now": now, "reason": "the experiment was recorded before content fingerprints existed: it cannot be checked"}
    was = recorded.split("#", 1)[1].split("@", 1)[0]
    return {"same": was == now, "recorded": was, "now": now, "reason": None if was == now else "the data has changed since this experiment ran (a correction, a restatement, a new filing, another source choice or new prices)"}
