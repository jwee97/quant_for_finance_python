"""Where a figure came from, side by side: the providers and the manual imports that hold the same line of the same period, what each says, and which one research will use.

Nothing is overwritten. Every source keeps its own row; the comparison shows the differences, who is preferred (a person's explicit choice first, else the provider priority, in which figures a person confirmed
come first) and whether a person has confirmed the figure. :meth:`Store.set_source_preference <src.fundamentals.store.FundamentalsStore.set_source_preference>` records a choice.
"""

from __future__ import annotations

import pandas as pd

from . import canonical
from .pit import PitSettings

TOLERANCE = 0.005                                                  # sources agree when they are within half a per cent


def compare(store, security_id: int, settings: PitSettings | None = None, only_conflicts: bool = False) -> list[dict]:
    s = settings or PitSettings()
    periods = store.periods([security_id])
    obs = store.observations([security_id])
    if periods.empty or obs.empty:
        return []
    prefs = {(p["metric_code"], p["period_end"]): p for p in store.source_preferences([security_id])}
    with store.db.transaction() as c:
        confirmed = {(r[0], r[1]) for r in c.execute("SELECT financial_period_id, metric_code FROM observation_provenance WHERE verified = 1").fetchall()}
    frame = obs.drop(columns=["provider"]).merge(periods, on=["period_id", "security_id"], how="inner")
    out = []
    for (fp, end, code), g in frame.groupby(["fiscal_period", "period_end", "metric_code"]):
        rows = []
        for _, r in g.sort_values(["provider", "vintage"]).iterrows():
            rows.append({"provider": r["provider"], "value": float(r["value"]), "currency": r["currency"] or r["reporting_currency"], "filing_date": str(r["filing_date"])[:10] if pd.notna(r["filing_date"]) else None,
                         "vintage": int(r["vintage"]), "quality_flag": r["quality_flag"], "restated": bool(r["is_restated"]), "confirmed_by_person": (int(r["period_id"]), code) in confirmed or r["provider"] == "manual"})
        providers = sorted({x["provider"] for x in rows}, key=s.rank)
        if len(providers) < 2 and only_conflicts:
            continue
        latest = {p: [x for x in rows if x["provider"] == p][-1] for p in providers}
        values = [x["value"] for x in latest.values()]
        spread = (max(values) - min(values)) / max(abs(max(values)), abs(min(values)), 1e-12) if len(values) > 1 else 0.0
        end_s = str(end)[:10]
        pref = prefs.get((code, end_s))
        chosen = pref["provider"] if pref and pref["provider"] in latest else providers[0]
        conflict = spread > TOLERANCE
        if only_conflicts and not conflict:
            continue
        out.append({"fiscal_period": fp, "period_end": end_s, "metric": code, "label": canonical.METRICS[code].label if code in canonical.METRICS else code, "sources": rows, "providers": providers,
                    "used": chosen, "used_because": "your choice" if pref and pref["provider"] in latest else "provider priority", "max_relative_difference": round(spread, 6), "conflict": conflict})
    return sorted(out, key=lambda r: (r["period_end"], r["metric"]))
