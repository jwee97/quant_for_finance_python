"""Stock splits, found in the stored prices, and what they do to share counts and per-share figures.

A market capitalisation is the unadjusted close times a share count taken from a filing. If the company splits 4:1 *after* that filing, the close drops to a quarter and the filing's share count does not,
so the market capitalisation (and every yield built on it) is a quarter of the truth until the next filing. The same holds for per-share figures (book value, net current assets, Graham number) set
against a post-split price.

Splits are inferred, not looked up: on a split day the unadjusted close falls by the ratio while the adjusted close does not, so ``(close_t / close_{t-1}) / (adj_t / adj_{t-1})`` is about 1/ratio; on a
dividend day it is only slightly below 1. A move of the quotient outside [0.8, 1.25] that is close to a usual ratio (2:1, 3:2, 5:1, 1:10 …) is taken as a split; any other large move is reported as
*unexplained* and is **not** applied (a merger, a spin-off or a data error is not a split). Spin-offs and mergers are not detected here: they change the company, not the share count, and are listed as a
limitation. A source that stores only adjusted prices gives no splits at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOW, HIGH = 0.8, 1.25
USUAL = sorted({float(a) for a in range(2, 21)} | {1.5, 2.5, 4 / 3, 5 / 4} | {1.0 / a for a in range(2, 21)} | {2 / 3, 4 / 5, 2 / 5})
TOLERANCE = 0.03


def _snap(ratio: float) -> float | None:
    best = min(USUAL, key=lambda u: abs(np.log(ratio / u)))
    return best if abs(np.log(ratio / best)) <= TOLERANCE else None


def detect(close: pd.DataFrame, adj: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """``(splits, unexplained)``: each ``{security_id, date, ratio, quotient}``; ``ratio`` is shares after per share before (4.0 for a 4-for-1 split, 0.1 for a 1-for-10 reverse split)."""
    quotient = (close / close.shift(1)) / (adj / adj.shift(1))
    splits, other = [], []
    for sid in quotient.columns:
        q = quotient[sid].dropna()
        for date, v in q[(q < LOW) | (q > HIGH)].items():
            ratio = 1.0 / float(v)
            snapped = _snap(ratio)
            rec = {"security_id": int(sid), "date": date, "quotient": float(v)}
            (splits if snapped else other).append({**rec, "ratio": snapped if snapped else ratio})
    return splits, other


def cumulative_log(close: pd.DataFrame, adj: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Daily cumulative log split ratio per company (0 before the first split) and the events."""
    splits, other = detect(close, adj)
    cum = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for s in splits:
        cum.loc[cum.index >= s["date"], s["security_id"]] += np.log(s["ratio"])
    return cum, splits, other


def since_update(panel: pd.DataFrame, cum_at_dates: pd.DataFrame) -> pd.DataFrame:
    """For each company and date, the cumulative split ratio between the date the panel's value last changed (a new filing) and the date: the factor a share count must be multiplied by (and a per-share
    figure divided by) to be on the basis of the date's price. Where a company has no value the factor is 1."""
    out = pd.DataFrame(1.0, index=panel.index, columns=panel.columns)
    cum = cum_at_dates.reindex(index=panel.index, columns=panel.columns).fillna(0.0)
    for c in panel.columns:
        v = panel[c].to_numpy(float)
        changed = np.r_[True, ~np.isclose(v[1:], v[:-1], rtol=1e-12, atol=0, equal_nan=True)]
        last = np.maximum.accumulate(np.where(changed, np.arange(len(v)), 0))
        base = cum[c].to_numpy()[last]
        out[c] = np.exp(cum[c].to_numpy() - base)
    return out.where(panel.notna(), 1.0)
