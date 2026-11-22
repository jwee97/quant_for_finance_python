"""Year-to-date figures and the standalone quarters derived from them.

A company that reports cumulatively (operating cash flow 100 for the first quarter, 250 for six months, 390 for nine months, 600 for the year) has standalone quarters of 100, 150, 140 and 210. The cumulative
figures are the observations; the quarters are **derived**, and every derived quarter says how (``derivation``), from which inputs (period, end date, filing date, vintage), when it became public
(``available_from``: the latest filing among its inputs, never earlier) and what looked wrong. Nothing is estimated: a quarter whose input is missing is not derived. Stocks (balance sheet items) are
instants, not cumulative: they are never differenced.

52/53-week years: the length of a derived quarter is recorded in days (91 or 98) and only a length far outside a quarter is queried; the dates are never forced onto calendar quarter ends.
"""

from __future__ import annotations

from datetime import date, timedelta

from . import canonical
from .schema import now_iso

ORDER = ("Q1", "H1", "Q3YTD", "FY")                    # cumulative to the end of quarter 1, 2, 3 and 4 (H1 is the first half, Q3YTD the nine months)
CUM_TO = {"Q1": 1, "H1": 2, "Q3YTD": 3, "FY": 4}
LABEL = {"Q1": "Q1", "H1": "H1 (six months)", "Q3YTD": "nine months", "FY": "full year"}


def decumulate(cum: dict[str, dict]) -> dict:
    """``cum``: cumulative observations of one flow in one fiscal year, keyed by ``Q1``, ``H1``, ``Q3YTD``, ``FY``, each ``{value, period_end, filing_date?, vintage?}``. Returns ``{quarter: {...}}`` for the
    quarters that can be derived exactly: Q1 is the first cumulative figure; each later quarter is a cumulative figure minus the one before it. If the half-year is missing, Q2 and Q3 are not separable but
    the second half is (``H2`` = year minus half-year)."""
    out: dict[str, dict] = {}

    def make(name, cur_key, prev_key):
        cur, prev = cum[cur_key], (cum[prev_key] if prev_key else None)
        end, start = cur["period_end"], (prev["period_end"] if prev else None)
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days if start else None
        inputs = [{"period": k, "label": LABEL[k], "period_end": cum[k]["period_end"], "value": cum[k]["value"], "filing_date": cum[k].get("filing_date"), "vintage": cum[k].get("vintage", 0)}
                  for k in ([cur_key] + ([prev_key] if prev_key else []))]
        filed = [i["filing_date"] for i in inputs if i["filing_date"]]
        vint = {i["vintage"] for i in inputs}
        flags = []
        if len(vint) > 1:
            flags.append("inputs are from different vintages (one may be restated): check the derived figure")
        if days is not None and not 75 <= days <= 110 and name in ("Q2", "Q3", "Q4"):
            flags.append(f"the derived period is {days} days long, not a quarter")
        if len(filed) < len(inputs):
            flags.append("an input has no filing date: availability is estimated")
        value = cur["value"] - (prev["value"] if prev else 0.0)
        out[name] = {"value": value, "period_end": end, "period_start_exclusive": start, "days": days, "inputs": inputs, "available_from": max(filed) if len(filed) == len(inputs) else None,
                     "derivation": f"{LABEL[cur_key]} {cur['value']:,.6g}" + (f" − {LABEL[prev_key]} {prev['value']:,.6g}" if prev else "") + f" = {value:,.6g}", "flags": flags}

    if "Q1" in cum:
        make("Q1", "Q1", None)
    if "H1" in cum and "Q1" in cum:
        make("Q2", "H1", "Q1")
    if "Q3YTD" in cum and "H1" in cum:
        make("Q3", "Q3YTD", "H1")
    if "FY" in cum and "Q3YTD" in cum:
        make("Q4", "FY", "Q3YTD")
    elif "FY" in cum and "H1" in cum and "Q3" not in out:
        make("H2", "FY", "H1")
    return out


def fiscal_year_of_start(period_start: str) -> int:
    """The fiscal year a cumulative period belongs to, named by the year the fiscal year *ends* (a year starting 2023-04-01 is 2024): every cumulative figure of one year shares its start date."""
    d = date.fromisoformat(period_start)
    try:
        nxt = d.replace(year=d.year + 1)
    except ValueError:                                                                                   # 29 February
        nxt = d.replace(year=d.year + 1, day=28)
    return (nxt - timedelta(days=1)).year


def save_cumulative(store, security_id: int, metric: str, fiscal_year: int, cum_period: str, period_end: str, value: float, currency: str | None, filing_date: str | None = None,
                    source: str = "manual", vintage: int = 0, note: str = "") -> None:
    """Keep one cumulative observation as reported (never overwritten by a derived quarter)."""
    if cum_period not in CUM_TO:
        raise ValueError(f"cum_period is one of {', '.join(ORDER)}")
    if metric not in canonical.METRICS or canonical.METRICS[metric].statement not in ("income", "cashflow") or canonical.METRICS[metric].kind not in ("flow", "per_share"):
        raise ValueError(f"{metric} is not a cumulative flow line")
    with store.db.transaction() as c:
        c.execute("INSERT OR REPLACE INTO ytd_observations (security_id, metric_code, fiscal_year, cum_period, period_end, value, currency, filing_date, source, vintage, note, created_at) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (int(security_id), metric, int(fiscal_year), cum_period, period_end, float(value), currency, filing_date, source, int(vintage), note[:200], now_iso()))


def derive(store, security_id: int, write: bool = False) -> list[dict]:
    """Derive the standalone quarters of every flow with cumulative observations. With ``write`` they are stored as periods of provider ``derived`` (flag ``derived``, source text = the derivation) so
    that TTM and factors can use them from the day ``available_from``; the cumulative observations stay as they were."""
    with store.db.transaction() as c:
        rows = c.execute("SELECT metric_code, fiscal_year, cum_period, period_end, value, currency, filing_date, vintage FROM ytd_observations WHERE security_id = ? "
                         "ORDER BY metric_code, fiscal_year, vintage", (int(security_id),)).fetchall()
    grouped: dict[tuple, dict] = {}
    for m, fy, cp, end, v, ccy, filed, vint in rows:
        grouped.setdefault((m, fy), {})[cp] = {"value": v, "period_end": end, "filing_date": filed, "vintage": vint, "currency": ccy}      # the latest vintage of each input
    with store.db.transaction() as c:                                                                      # the full year, from the annual figures held (not derived ones): the fourth quarter is the year less nine months
        annual = c.execute("SELECT o.metric_code, p.period_end, o.value, o.currency, p.filing_date FROM fundamental_observations o JOIN financial_periods p ON p.id = o.financial_period_id "
                           "WHERE p.security_id = ? AND p.fiscal_period = 'FY' AND p.provider != 'derived' AND p.vintage = 0", (int(security_id),)).fetchall()
    for m, end, v, ccy, filed in annual:
        key = (m, int(str(end)[:4]))
        if key in grouped and "FY" not in grouped[key]:
            grouped[key]["FY"] = {"value": v, "period_end": str(end)[:10], "filing_date": filed, "vintage": 0, "currency": ccy}
    derived = []
    for (m, fy), cum in sorted(grouped.items()):
        ccys = {x["currency"] for x in cum.values() if x["currency"]}
        if len(ccys) > 1:
            derived.append({"metric": m, "fiscal_year": fy, "quarter": None, "value": None, "flags": [f"the cumulative figures are in different currencies ({', '.join(sorted(ccys))}): not derived"]})
            continue
        for q, d in decumulate(cum).items():
            derived.append({"metric": m, "fiscal_year": fy, "quarter": q, "currency": next(iter(ccys), None), **d})
    if write:
        _write(store, security_id, derived)
    return derived


def _write(store, security_id: int, derived: list[dict]) -> None:
    now = now_iso()
    with store.db.transaction() as c:
        for d in derived:
            if d.get("value") is None:
                continue
            avail = d.get("available_from")
            c.execute("INSERT OR IGNORE INTO financial_periods (security_id, fiscal_year, fiscal_period, period_end, filing_date, reporting_currency, provider, vintage, pit_quality, created_at, updated_at) "
                      "VALUES (?, ?, ?, ?, ?, ?, 'derived', 0, ?, ?, ?)", (int(security_id), d["fiscal_year"], d["quarter"], d["period_end"], avail, d.get("currency"), "filed" if avail else "estimated", now, now))
            pid = c.execute("SELECT id FROM financial_periods WHERE security_id = ? AND provider = 'derived' AND fiscal_period = ? AND period_end = ? AND vintage = 0",
                            (int(security_id), d["quarter"], d["period_end"])).fetchone()[0]
            c.execute("INSERT OR REPLACE INTO fundamental_observations (financial_period_id, metric_code, value, currency, source_metric, provider, quality_flag, created_at) VALUES (?, ?, ?, ?, ?, 'derived', ?, ?)",
                      (pid, d["metric"], d["value"], d.get("currency"), "derived: " + d["derivation"], "derived", now))
