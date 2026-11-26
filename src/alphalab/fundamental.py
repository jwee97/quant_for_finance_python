"""Fundamental factors for the Alpha Lab, admitted only after a point-in-time check.

A fundamentals figure is a step function: it changes on the day a filing becomes public. The Alpha Lab's factor study reads it exactly like a price signal, which is only honest if the day each figure becomes
visible is the day it was public. So before any figure is read, :func:`src.fundamentals.audit.pit_audit` runs on the companies asked for, and a ``blocked`` verdict (a figure dated before the period it describes,
or the look-ahead switch on) stops the study with the reasons. A ``ready_with_assumptions`` verdict lets it through and the assumptions travel with the result, so a reader sees that some dates were assumed.

What this module adds is the join: tickers to securities, the derived metric panel on the study's own calendar, and the coverage report (how many of the requested companies and dates have a value). It computes no
factor of its own: the metrics are those of :mod:`src.fundamentals.metrics`, built from the same tables the Fundamentals tab shows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fundamentals import audit, metrics
from ..fundamentals.pit import PeriodTables, PitSettings
from ..fundamentals.store import FundamentalsStore

MIN_COMPANIES = 6


class FundamentalError(ValueError):
    """A fundamental factor cannot be admitted (and why)."""


def available_metrics(store: FundamentalsStore, basis: str = "fy") -> list[str]:
    return sorted(d["name"] for d in metrics.catalogue())


def panels(store: FundamentalsStore, tickers: list[str], names: list[str], dates: pd.DatetimeIndex, basis: str = "fy", settings: PitSettings | None = None) -> dict:
    if not names:
        raise FundamentalError("choose at least one fundamental factor")
    settings = settings or PitSettings()
    sec = store.securities()
    by_ticker = {str(r.ticker).upper(): int(r.security_id) for r in sec.itertuples()} if len(sec) else {}
    wanted = [t for t in tickers]
    ids = {t: by_ticker.get(t.upper()) for t in wanted}
    missing = [t for t, i in ids.items() if i is None]
    held = {t: i for t, i in ids.items() if i is not None}
    if len(held) < MIN_COMPANIES:
        raise FundamentalError(f"fundamental factors need statements for at least {MIN_COMPANIES} of the chosen tickers; the fundamentals database holds {len(held)}"
                               + (f" (no company called {', '.join(missing[:8])}{'…' if len(missing) > 8 else ''})" if missing else ""))
    pit = audit.pit_audit(store, settings, list(held.values()))
    if pit["verdict"]["status"] == "blocked":
        raise FundamentalError("the point-in-time check blocks these figures: " + "; ".join(pit["verdict"]["reasons"]))
    tables = PeriodTables(store, list(held.values()), settings)
    metrics.compute(tables, basis)
    table = tables.table(basis)
    unknown = [n for n in names if n not in table.columns]
    if unknown:
        raise FundamentalError(f"no fundamental factor called {', '.join(unknown)}")
    out, coverage = {}, {}
    inv = {i: t for t, i in held.items()}
    for n in names:
        panel = tables.panel(n, pd.DatetimeIndex(dates), basis, list(held.values())).rename(columns=inv)
        panel = panel.reindex(columns=[t for t in wanted])
        out[f"fund:{n}"] = panel
        per_date = panel.notna().sum(axis=1)
        coverage[n] = {"tickers_with_a_value": int(panel.notna().any().sum()), "tickers_requested": len(wanted), "share_of_cells": float(panel.notna().to_numpy().mean()),
                       "median_assets_per_date": float(per_date.median()), "first_value": (str(panel.dropna(how="all").index[0].date()) if panel.notna().any().any() else None)}
        if coverage[n]["tickers_with_a_value"] < MIN_COMPANIES:
            raise FundamentalError(f"{n} has a value for only {coverage[n]['tickers_with_a_value']} of the {len(wanted)} tickers; at least {MIN_COMPANIES} are needed to rank")
    return {"panels": out, "pit": pit, "coverage": coverage, "missing_tickers": missing, "basis": basis, "settings": settings.describe()}
