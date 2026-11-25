"""Data provenance, one schema for the Options and Futures workspaces.

Every payload says what kind of data its numbers are, where they came from, when they were observed, how old that is, which instrument they describe, which assumptions they rest on and whether the
model behind them was calibrated to anything. The kinds:

``observed_live``       quotes observed now (a live feed). Nothing here receives one yet, so nothing is labelled live.
``observed_delayed``    quotes observed at a stated time and published late (the free chains are about 15 minutes late).
``historical_observed`` real prices or quotes of past dates (downloaded histories, the platform's price history, files you supplied).
``calibrated_model``    a model's output after it was fitted to observed quotes (the fit date and quality are given).
``synthetic``           simulated: a model quoted on real spot prices, or a pure simulation. It demonstrates mechanics and is not evidence about markets.
``mixed``               parts of several kinds (each part is listed).

Nothing is ever substituted silently: a request for a real source that is not available is an error, never a synthetic answer; where a payload mixes real and synthetic parts it says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd

KINDS = ("observed_live", "observed_delayed", "historical_observed", "calibrated_model", "synthetic", "mixed")
OF_SOURCE = {"platform": "historical_observed", "user_file": "historical_observed", "downloaded": "observed_delayed", "simulated": "synthetic", "model": "synthetic", "mixed": "mixed"}
HELP = {"observed_live": "Quotes observed now.", "observed_delayed": "Real quotes, observed at the time shown and published late.", "historical_observed": "Real prices or quotes of past dates.",
        "calibrated_model": "A model fitted to observed quotes: only as good as the fit shown beside it.", "synthetic": "Simulated. It shows how the mathematics works; it is not evidence about any market.",
        "mixed": "Parts of several kinds: see the list of parts."}
FRESH_AFTER_HOURS = {"observed_live": 0.1, "observed_delayed": 36.0}                        # beyond this a live or delayed observation is stale; histories are dated, not stale


@dataclass
class Provenance:
    kind: str
    source: str
    parts: dict = field(default_factory=dict)             # part -> kind
    observed_at: str | None = None
    as_of: str | None = None
    age_hours: float | None = None
    freshness: str = "unknown"                            # fresh | stale | historical | not_applicable | unknown
    instrument: str | None = None
    provider: str | None = None
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    calibration: dict | None = None                       # {"status": "calibrated|not_calibrated|failed", "fit_date", "rmse", "model", ...}
    help: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def kind_of(source: str, calibrated: bool = False) -> str:
    k = OF_SOURCE.get(source, "synthetic")
    return "calibrated_model" if calibrated and k == "synthetic" else k


def build(parts: dict[str, str], as_of=None, observed_at=None, instrument: str | None = None, provider: str | None = None, assumptions=None, warnings=None, calibration: dict | None = None, now=None) -> dict:
    """The provenance block of a payload. ``parts`` maps each part of the payload to its source word (``platform``, ``user_file``, ``downloaded``, ``simulated``, ``model``, ``mixed``)."""
    if parts and all(s == "unavailable" for s in parts.values()):
        return Provenance(kind="unavailable", source="unavailable", parts={p: "unavailable" for p in parts}, as_of=None if as_of is None else str(pd.Timestamp(as_of).date()), freshness="not_applicable",
                          warnings=list(warnings or []) + ["Nothing could be computed from observed data."], help="No observed data: the page says what is needed.").to_dict()
    parts = {p: s for p, s in parts.items() if s != "unavailable"}
    kinds = {p: kind_of(s, calibration is not None and calibration.get("status") == "calibrated" and s in ("model", "simulated") and False) for p, s in parts.items() if s}
    if calibration and calibration.get("status") == "calibrated":
        kinds = {p: ("calibrated_model" if k == "synthetic" and parts[p] == "model" else k) for p, k in kinds.items()}
    values = set(kinds.values())
    kind = values.pop() if len(values) == 1 else ("mixed" if values else "synthetic")
    src = {s for s in parts.values() if s}
    source = src.pop() if len(src) == 1 else ("mixed" if src else "simulated")
    now = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    when = pd.Timestamp(observed_at) if observed_at is not None else None
    if when is not None and when.tzinfo is not None:
        when = when.tz_convert("UTC").tz_localize(None)
    ref = when if when is not None else (pd.Timestamp(as_of) + pd.Timedelta(hours=21) if as_of is not None else None)       # a date alone: the close of that day
    age = None if ref is None else max(0.0, (now - ref).total_seconds() / 3600.0)
    warn = list(warnings or [])
    if kind in ("synthetic",):
        fresh = "not_applicable"
        warn.append("Synthetic data: a demonstration of the mathematics, not evidence about a market.")
    elif kind in FRESH_AFTER_HOURS:
        fresh = "unknown" if age is None else ("fresh" if age <= FRESH_AFTER_HOURS[kind] else "stale")
        if fresh == "stale":
            warn.append(f"The observation is {age:.0f} hours old: quotes may have moved.")
    elif kind == "mixed":
        fresh = "unknown" if age is None else "historical"
        if "synthetic" in kinds.values():
            warn.append("Part of this view is synthetic (" + ", ".join(p for p, k in kinds.items() if k == "synthetic") + "): do not read it as one empirical dataset.")
    else:
        fresh = "historical"
    if calibration is not None and calibration.get("status") != "calibrated":
        warn.append(f"The model is not calibrated ({calibration.get('status', 'unknown')}): its output is illustrative.")
    p = Provenance(kind=kind, source=source, parts=kinds, observed_at=None if when is None else str(when), as_of=None if as_of is None else str(pd.Timestamp(as_of).date()),
                   age_hours=None if age is None else round(age, 2), freshness=fresh, instrument=instrument, provider=provider, assumptions=list(assumptions or []), warnings=warn, calibration=calibration, help=HELP[kind])
    return p.to_dict()
