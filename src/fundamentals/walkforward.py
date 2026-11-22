"""Walk-forward validation of a factor: what is chosen from the past is judged on the future it never saw.

Each fold looks at a *training* window of dates, chooses the sign of the factor from the mean rank IC there (the only thing chosen: no parameter search, so nothing else can be overfitted), skips an
*embargo* of dates (the forward return of a training date overlaps the next ``horizon`` dates; testing right after it would leak), and measures the signed IC on the following *test* window. The
out-of-sample series is the concatenation of the test windows; the report compares it with the in-sample IC of the same dates, because a factor that only works in the sample it was chosen on shows a large
gap. With many candidate factors, correct the p-values for the number tried (:func:`bh_adjust`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import newey_west, rank_ic


def folds(n_dates: int, train: int, test: int, embargo: int, step: int | None = None) -> list[tuple[int, int, int, int]]:
    """``(train_start, train_end, test_start, test_end)`` as positions, rolling by ``step`` (default the test length)."""
    step = step or test
    out, start = [], 0
    while start + train + embargo + test <= n_dates:
        out.append((start, start + train, start + train + embargo, start + train + embargo + test))
        start += step
    return out


def run_panels(signal: pd.DataFrame, forward: pd.DataFrame, horizon_dates: int = 1, train: int = 36, test: int = 12, min_assets: int = 20) -> dict:
    ic = rank_ic(signal, forward, min_assets).reindex(signal.index)
    fl = folds(len(ic), train, test, embargo=max(0, int(horizon_dates)))
    if not fl:
        return {"folds": [], "reason": f"not enough dates: {len(ic)} available, {train} train + {horizon_dates} embargo + {test} test needed", "oos": None}
    rows, oos, ins = [], [], []
    for a, b, c, d in fl:
        tr, te = ic.iloc[a:b].dropna(), ic.iloc[c:d].dropna()
        if len(tr) < max(6, train // 3) or len(te) < max(3, test // 3):
            continue
        sign = 1.0 if tr.mean() >= 0 else -1.0
        rows.append({"train": [str(ic.index[a])[:10], str(ic.index[b - 1])[:10]], "test": [str(ic.index[c])[:10], str(ic.index[d - 1])[:10]], "chosen_sign": sign, "train_ic": float(tr.mean()),
                     "test_ic_signed": float(sign * te.mean()), "n_train": int(len(tr)), "n_test": int(len(te))})
        oos.append(sign * te)
        ins.append(sign * tr)
    if not rows:
        return {"folds": [], "reason": "no fold had enough dates with a measurable IC", "oos": None}
    o, i = pd.concat(oos), pd.concat(ins)
    nw = newey_west(o.to_numpy(), max(1, int(horizon_dates)))
    mean_o, mean_i = float(o.mean()), float(i.mean())
    return {"folds": rows, "oos": {"mean_ic": mean_o, "t_stat": nw["t"], "p_value": nw["p"], "hit_rate": float((o > 0).mean()), "n_dates": int(len(o)), "folds_positive": float(np.mean([r["test_ic_signed"] > 0 for r in rows]))},
            "in_sample": {"mean_ic": mean_i}, "degradation": (None if abs(mean_i) < 1e-12 else float(mean_o / mean_i)),
            "reading": ("holds out of sample" if mean_o > 0 and mean_o / mean_i >= 0.7 and np.mean([r["test_ic_signed"] > 0 for r in rows]) >= 0.6 else "weak out of sample" if mean_o > 0 else "does not hold out of sample") if abs(mean_i) > 1e-12 else "no in-sample signal"}


def run(ctx, signal: pd.DataFrame, horizon: str | int = "1m", train: int = 36, test: int = 12) -> dict:
    """Walk forward on a research context: the forward return of each date over ``horizon`` (its overlap with later dates is embargoed)."""
    from .research import FREQ_DAYS, horizon_days
    h = max(1, int(np.ceil(horizon_days(horizon) / FREQ_DAYS[ctx.freq])))
    return run_panels(signal, ctx.forward(horizon).reindex(signal.index), h, train, test)


def bh_adjust(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values for the number of factors tried."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(adj, 0, 1)
    return out.tolist()
