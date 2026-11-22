"""Where a figure came from, side by side: the providers and the manual imports that hold the same line of the same period, what each says, and which one research will use.

Nothing is overwritten. Every source keeps its own row; the comparison shows the differences, who is preferred (a person's explicit choice first, else the provider priority, in which figures a person confirmed
come first) and whether a person has confirmed the figure. :meth:`Store.set_source_preference <src.fundamentals.store.FundamentalsStore.set_source_preference>` records a choice.
"""

from __future__ import annotations

import pandas as pd

from . import canonical
from .pit import PitSettings

TOLERANCE = 0.005                                                  # sources agree when they are within half a per cent


def compare(store, security_id: int, settings: PitSettings | None = None, only_conflicts: bool = False, frames: dict | None = None) -> list[dict]:
    """``frames`` (``periods``, ``obs``, ``prefs``, ``confirmed``) lets a caller that compares many companies load the tables once instead of once per company."""
    s = settings or PitSettings()
    if frames is None:
        periods = store.periods([security_id])
        obs = store.observations([security_id])
        prefs_rows = store.source_preferences([security_id])
        with store.db.transaction() as c:
            confirmed = {(r[0], r[1]) for r in c.execute("SELECT v.financial_period_id, v.metric_code FROM observation_provenance v JOIN financial_periods p ON p.id = v.financial_period_id "
                                                         "WHERE v.verified = 1 AND p.security_id = ?", (int(security_id),)).fetchall()}
    else:
        periods, obs, prefs_rows, confirmed = frames["periods"], frames["obs"], frames["prefs"], frames["confirmed"]
    if periods.empty or obs.empty:
        return []
    prefs = {(p["metric_code"], p["period_end"]): p for p in prefs_rows if p["security_id"] == security_id}
    frame = obs.drop(columns=["provider"]).merge(periods, on=["period_id", "security_id"], how="inner")
    out = []
    frame = frame.sort_values(["provider", "vintage"])
    cols = ["fiscal_period", "period_end", "metric_code", "provider", "value", "currency", "reporting_currency", "filing_date", "vintage", "quality_flag", "is_restated", "period_id"]
    groups: dict[tuple, list] = {}
    for rec in zip(*[frame[c].to_numpy() for c in cols]):                                              # one pass over the rows, not a sort and a row iterator per group
        groups.setdefault((rec[0], rec[1], rec[2]), []).append(rec)
    for (fp, end, code), recs in groups.items():
        rows = []
        for (_, _, _, provider, value, ccy, rccy, filed, vintage, flag, restated, pid) in recs:
            rows.append({"provider": provider, "value": float(value), "currency": ccy or rccy, "filing_date": str(filed)[:10] if pd.notna(filed) else None,
                         "vintage": int(vintage), "quality_flag": flag, "restated": bool(restated), "confirmed_by_person": (int(pid), code) in confirmed or provider == "manual"})
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
