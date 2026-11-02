"""Forecast horizons of 1, 5, 10 and 20 sessions, compared without rescaling overlapping returns (spec v3, Release 6, P2-10).

A forecast of horizon ``h`` produced every day does not define ``h`` independent daily portfolios, and a longer horizon cannot be compared with a shorter one by dividing its ``h``-day return by ``h`` or by
comparing information coefficients measured against labels of different lengths. What is comparable is the **actual daily equity of the book the horizon implies**, on the same dates, with the same costs, lag and
risk settings. This module supplies the three pieces that make that statement checkable:

* ``tranche_books`` - the staggered holding the platform's ``signals.holding_period`` implements, worked out independently as ``h`` sleeves of ``1/h`` of capital, each re-set to the day's book every ``h`` days; the
  combined book, its gross exposure (never more than one daily book's, so nothing is counted twice) and the ``h``-times over-count that adding the overlapping books instead of averaging them would give;
* ``comparable_conventions`` - the settings two configurations must share for their results to be compared across horizons, and the ones that are allowed to differ (the target, the horizon and the holding period);
* ``horizon_changes`` - the candidates of a horizon study as changes to a baseline, each moving the target, the horizon and the holding period together, for ``comparison.new_card(family="horizon")``.

The actual per-row label interval of every horizon is the dataset's (``intervals.label_frame``): a decision on ``t`` is traded ``lag`` sessions later and its label runs ``h`` sessions after that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .comparison import diff_paths
from .config import experiment_config
from .targets import parse_target

VERSION = "horizons-v1"
HORIZONS = (1, 5, 10, 20)
ALLOWED_TO_DIFFER = ("dataset.target", "dataset.horizon", "signal.holding_period")


def tranche_books(daily_books: pd.DataFrame, h: int) -> dict:
    """The combined book of ``h`` sleeves. Sleeve ``k`` holds, on day ``t``, the book of the latest day ``s <= t`` with ``s = k (mod h)``; capital is split equally among the sleeves. During the first ``h - 1`` days
    only the sleeves that already exist are funded and the book is their average (the platform's convention: full gross in fewer tranches), which ``warmup_days`` counts."""
    if h < 1:
        raise ValueError("the holding period is at least one day")
    b = daily_books.fillna(0.0)
    n = len(b)
    combined = np.zeros(b.shape)
    sleeves = {}
    values = b.to_numpy()
    for k in range(min(h, n)):
        held = np.zeros(b.shape)
        for t in range(k, n):
            s = t - ((t - k) % h)                                                                 # the latest day not after t whose index is k modulo h
            held[t] = values[s]
        sleeves[k] = held
    for t in range(n):
        live = [k for k in sleeves if k <= t]                                                     # a sleeve exists once its first day has come
        combined[t] = np.mean([sleeves[k][t] for k in live], axis=0)
    combined = pd.DataFrame(combined, index=b.index, columns=b.columns)
    naive = pd.DataFrame(sum(sleeves[k] for k in sleeves), index=b.index, columns=b.columns)    # adding the overlapping tranches instead of splitting capital among them
    gross, one = combined.abs().sum(axis=1), b.abs().sum(axis=1)
    return {"version": VERSION, "horizon": int(h), "combined": combined, "sleeves": {k: pd.DataFrame(v, index=b.index, columns=b.columns) for k, v in sleeves.items()}, "warmup_days": int(min(h - 1, n)),
            "max_gross": float(gross.max()) if n else 0.0, "max_single_book_gross": float(one.max()) if n else 0.0, "never_more_than_one_book": bool((gross <= one.rolling(h, min_periods=1).max() + 1e-12).all()),
            "naive_sum_gross_max": float(naive.abs().sum(axis=1).max()) if n else 0.0,
            "note": "capital is split equally among the sleeves, so the combined gross never exceeds the largest daily book's; adding the overlapping books instead would multiply exposure by up to h"}


def comparable_conventions(a: dict, b: dict) -> dict:
    """Whether two experiment configurations differ only where a horizon comparison allows (the target, the horizon and the holding period). Anything else that differs (lag, costs, rebalancing, the position rule,
    execution settings, the model, the splits) would make a difference in results a difference in conventions, and is listed."""
    ca, cb = experiment_config(a), experiment_config(b)
    paths = diff_paths(ca, cb)
    stray = [p for p in paths if p not in ALLOWED_TO_DIFFER]
    problems = []
    ta, ha = parse_target(ca["dataset"]["target"])[1], int(ca["dataset"]["horizon"])
    tb, hb = parse_target(cb["dataset"]["target"])[1], int(cb["dataset"]["horizon"])
    if ta != ha or tb != hb:
        problems.append("a target's horizon and dataset.horizon disagree")
    if int(ca["signal"]["holding_period"]) != ha or int(cb["signal"]["holding_period"]) != hb:
        problems.append("the holding period is not the horizon: the book would be held for a different time than the forecast covers (allowed, but then it is a second change)")
    return {"version": VERSION, "matched": not stray, "differences": paths, "unmatched": stray, "problems": problems,
            "horizons": [ha, hb], "note": "results of different horizons are compared on the daily net returns of the book each implies, on the same dates; ICs of different horizons are not commensurable"}


def horizon_changes(baseline: dict, horizons=HORIZONS) -> list[dict]:
    """The candidates of a horizon study: each changes the target's horizon, ``dataset.horizon`` and the holding period together (they are one change: a forecast of ``h`` sessions held for ``h`` sessions)."""
    cfg = experiment_config(baseline)
    kind, base_h = parse_target(cfg["dataset"]["target"])
    out = []
    for h in horizons:
        if int(h) == base_h:
            continue
        out.append({"dataset": {"target": f"{kind}_{int(h)}", "horizon": int(h)}, "signal": {"holding_period": int(h)}})
    return out
