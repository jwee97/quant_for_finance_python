"""Point-in-time inputs: join on availability, keep the revision record, and say how good the evidence is (P0-15).

A published number has two dates, the period it describes and the instant anyone could first have known it, and a restatement is a new vintage of the same period that is public only from its own release. A
join on the period (``June CPI`` on 1 June) hands a model a number six weeks before it existed. ``vintage_asof`` joins on availability instead, at the instant of the decision (session-aware: ``sessions.py``),
and takes the value from the platform's fundamentals engine (``src/fundamentals/pit.step_panel``): on each decision the value of the latest period that was public, with the latest vintage of it that was public, so
a late restatement of an old period never replaces a newer one. ``known_value`` asks the narrower question, what one period was believed to be at one instant.

How good the evidence is depends on what was kept. ``observed`` means the availability of each value is a recorded instant (a market print at its close, a filing date, a vintage log); ``estimated`` means it is
a fixed release lag applied to the latest vintage, so a later revision is invisible and the validation looks a little further ahead than the world did; ``unavailable`` means there is no record at all and the
value is joined on its own date. ``pit_report`` names the quality of every macro input a dataset uses and says whether validation claims on it are restricted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sessions import decision_instants

QUALITIES = ("observed", "estimated", "unavailable")
# Market prints that exist at the close of their date and are not revised afterwards (the release lag of the platform's loader covers their publication). Everything else the platform serves from FRED is the
# latest vintage, so its availability is an estimate.
NEVER_REVISED = frozenset({"VIX", "VIX3M", "VXN", "GVZ", "OVX", "MOVE", "DGS10", "DGS2", "DGS3MO", "T10Y2Y", "T10Y3M"})
DERIVED_AT_CLOSE = ("CARRY_", "SLOPE_", "BASISMOM_", "BASIS_")                         # computed from the market prices known at each close
MACRO_FEATURE_PREFIX = "macro_"


class PITError(ValueError):
    """Records that cannot be joined on availability."""


def _instants(available) -> np.ndarray:
    """UTC nanoseconds of each availability stamp. A stamp with a zone is taken as it is; a date with no time of day is public from the END of that UTC day (the conservative reading); any other naive time is UTC."""
    stamps = pd.to_datetime(pd.Series(available).reset_index(drop=True))
    out = np.empty(len(stamps), dtype=np.int64)
    for i, t in enumerate(stamps):
        if t.tzinfo is None:
            if t == t.normalize():
                t = t + pd.Timedelta(days=1)
            t = t.tz_localize("UTC")
        out[i] = t.tz_convert("UTC").value
    return out


def vintage_asof(records: pd.DataFrame, decision, decision_dates, latency_minutes: int = 0, max_age_days: int | None = None) -> pd.Series:
    """The series as it was known at each decision: ``records`` has ``reference_date``, ``value``, ``available_at`` and optionally ``vintage`` (0 for the first release, 1 for the first restatement ...).

    On a decision, the value is that of the latest reference period with any vintage public at the decision instant, and of the latest vintage of it public by then. A value released after the decision is not there
    yet; a value released late is first seen on the first decision after its release."""
    missing = {"reference_date", "value", "available_at"} - set(records.columns)
    if missing:
        raise PITError(f"the records lack {sorted(missing)}: a series cannot be joined on availability without when each value became public")
    from ..fundamentals.pit import step_panel

    dates = pd.DatetimeIndex(decision_dates)
    when = decision_instants(decision, dates, latency_minutes).asi8
    available = _instants(records["available_at"])
    vintage = records["vintage"].to_numpy(dtype=int) if "vintage" in records.columns else np.zeros(len(records), dtype=int)
    order = np.lexsort((vintage, available))                                                         # by when public, then by vintage: the later of two on the same decision wins
    first_seen = np.searchsorted(when, available[order], side="left")                                # the first decision at or after the release
    keep = first_seen < len(dates)
    events = pd.DataFrame({"security_id": 0, "available_date": dates[first_seen[keep]], "period_end": pd.to_datetime(records["reference_date"]).to_numpy()[order][keep],
                           "value": records["value"].to_numpy(dtype=float)[order][keep]})
    return step_panel(events, dates, [0], max_age_days=max_age_days)[0].rename("value")


def known_value(records: pd.DataFrame, reference_date, at) -> float:
    """What one period's value was believed to be at the instant ``at`` (a timestamp): the latest vintage of it public by then, or NaN if none was."""
    rec = records[pd.to_datetime(records["reference_date"]) == pd.Timestamp(reference_date)]
    if rec.empty:
        return float("nan")
    available = _instants(rec["available_at"])
    stamp = pd.Timestamp(at)
    now = (stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")).value
    vintage = rec["vintage"].to_numpy(dtype=int) if "vintage" in rec.columns else np.zeros(len(rec), dtype=int)
    ok = available <= now
    if not ok.any():
        return float("nan")
    pick = np.lexsort((vintage[ok], available[ok]))[-1]
    return float(rec["value"].to_numpy(dtype=float)[ok][pick])


def join_on_reference(records: pd.DataFrame, dates) -> pd.Series:
    """The join that leaks: each date gets the latest value whose REFERENCE date is not after it, whenever it was published. Kept to measure how much look-ahead it would add (a diagnostic, never a feature)."""
    ref = pd.to_datetime(records["reference_date"]).to_numpy()
    order = np.argsort(ref, kind="stable")
    k = np.searchsorted(ref[order], np.asarray(pd.DatetimeIndex(dates), dtype="datetime64[ns]"), side="right") - 1
    values = records["value"].to_numpy(dtype=float)[order]
    return pd.Series(np.where(k >= 0, values[np.maximum(k, 0)], np.nan), index=pd.DatetimeIndex(dates))


# ------------------------------------------------------------------------------------------------------------------ how good is the evidence
def series_quality(name: str, meta: dict | None = None) -> dict:
    """``{quality, basis, release_lag_days}`` of one macro series. ``meta`` is what the loader recorded about it (``release_lag_days``, ``source``, ``frequency``, and ``quality`` if it knows better)."""
    meta = meta or {}
    lag = meta.get("release_lag_days")
    if meta.get("quality") in QUALITIES:
        return {"quality": meta["quality"], "basis": str(meta.get("basis") or "declared by the loader"), "release_lag_days": lag}
    if name in NEVER_REVISED:
        return {"quality": "observed", "basis": "a market print available at the close of its date and not revised afterwards", "release_lag_days": lag}
    if name.startswith(DERIVED_AT_CLOSE):
        return {"quality": "observed", "basis": "computed from market prices known at each close", "release_lag_days": lag}
    if lag is not None:
        return {"quality": "estimated", "basis": f"a fixed release lag of {int(lag)} days applied to the latest vintage: a later revision is not visible", "release_lag_days": int(lag)}
    return {"quality": "unavailable", "basis": "no publication or vintage record: the value is joined on its own date", "release_lag_days": None}


def fundamentals_quality(pit_quality: str) -> str:
    """The vocabulary of ``src/fundamentals`` in this module's: a figure with a real filing date is ``observed`` (a restatement is a recorded vintage), one whose date is an assumed lag is ``estimated``."""
    return {"filed": "observed", "restated": "observed", "estimated": "estimated"}.get(str(pit_quality), "unavailable")


def macro_columns_used(feature_names, macro_columns, assets) -> list[str]:
    """The macro columns the chosen features read, from their names and the series conventions of ``features.py`` (a macro level, a per-asset carry or implied-volatility series, funding and basis)."""
    from .features import IV_INDEX

    have, names = set(macro_columns), list(feature_names)
    used: set = set()
    for n in names:
        if n.startswith(MACRO_FEATURE_PREFIX):
            used.add(n[len(MACRO_FEATURE_PREFIX):].rsplit("_", 1)[0])
        elif n in ("carry", "carry_smooth_21", "carry_to_vol", "curve_slope", "basis_momentum"):
            used |= {c for c in have if c.startswith(("CARRY_", "SLOPE_", "BASISMOM_")) and c.split("_", 1)[1] in set(assets)}
        elif n.startswith("iv_"):
            used |= {IV_INDEX[a] for a in assets if a in IV_INDEX}
        elif n.startswith(("funding_", "perp_basis")):
            used |= {c for c in have if c.startswith(("FUNDING_", "BASIS_"))}
        elif n == "stable_growth_30":
            used.add("STABLE_SUPPLY")
    return sorted(used & have)


def pit_report(used: list[str], meta: dict | None = None) -> dict:
    """The point-in-time quality of every macro input a dataset uses, and whether validation claims on it are restricted (any input that is not ``observed`` restricts them)."""
    meta = meta or {}
    series = {name: series_quality(name, meta.get(name)) for name in used}
    counts = {q: sum(1 for s in series.values() if s["quality"] == q) for q in QUALITIES}
    restricted = bool(counts["estimated"] or counts["unavailable"])
    if not series:
        statement = "no macro or fundamental input is used, so there is no point-in-time evidence to qualify"
    elif restricted:
        statement = (f"{counts['estimated']} macro input(s) use a fixed release lag on the latest vintage and {counts['unavailable']} have no availability record: later revisions are invisible here, so validation claims "
                     "that rest on them are restricted (the result may look a little further ahead than the world did)")
    else:
        statement = "every macro input is a recorded market print or is computed from one"
    return {"series": series, "counts": counts, "claims_restricted": restricted, "statement": statement}
