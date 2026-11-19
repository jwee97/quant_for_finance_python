"""Overlapping labels and what they do to a sample (Lopez de Prado, chapter 4).

A label is a statement about the returns between its start and its end. Two labels whose windows overlap share returns, so they are not independent observations, and treating them as if they were
overstates how much data there is. ``sample_quality`` says by how much, ``sequential_bootstrap`` draws training samples that are less redundant than an ordinary bootstrap, and the weights below tell a model how
much each label contributes that nobody else's already does.

An event covers the rows ``pos_entry + 1 .. pos1`` (inclusive; ``pos0 + 1`` without an entry column): the returns earned after the entry and up to the row at which the label was decided. These are the intervals ``src/ml/intervals.py`` counts concurrency over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml import intervals as iv


def spans(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The first and last row of the returns each label depends on."""
    first = labels["pos_entry"] if "pos_entry" in labels.columns else labels["pos0"]
    return first.to_numpy() + 1, labels["pos1"].to_numpy()


def sample_quality(labels: pd.DataFrame, n_rows: int) -> dict:
    """How redundant the labels are: the average number of labels alive on a day they cover, the average uniqueness, and the *effective* sample size ``(sum u)^2 / sum u^2`` (Kish): the number of independent labels the sample is worth."""
    if labels.empty:
        return {"events": 0}
    s, e = spans(labels)
    c = iv.concurrency(s, e, n_rows)
    u = iv.average_uniqueness(s, e, n_rows)
    live = c[c > 0]
    return {"events": int(len(labels)), "mean_concurrency": float(live.mean()), "max_concurrency": int(live.max()), "mean_uniqueness": float(u.mean()), "min_uniqueness": float(u.min()),
            "effective_sample_size": float(u.sum() ** 2 / (u ** 2).sum()), "uniqueness_deciles": [float(x) for x in np.quantile(u, np.linspace(0, 1, 11))],
            "note": "the effective sample size counts the labels as if each were weighted by its uniqueness; it is a guide to how much independent information there is, not a degrees-of-freedom correction"}


def uniqueness(labels: pd.DataFrame, n_rows: int) -> pd.Series:
    """Average uniqueness of each label (1 if it overlaps nobody)."""
    s, e = spans(labels)
    return pd.Series(iv.average_uniqueness(s, e, n_rows), index=labels.index, name="uniqueness")


def return_attribution_weights(labels: pd.DataFrame, returns: pd.Series) -> pd.Series:
    """Sample weights from the size of the return a label is responsible for: ``|sum over its days of r_t / c_t|``, scaled to mean one. A label that spans a big move *alone* counts for more than one that shares it."""
    s, e = spans(labels)
    n = len(returns)
    c = iv.concurrency(s, e, n)
    share = np.divide(returns.to_numpy(dtype=float), c, out=np.zeros(n), where=c > 0)
    cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(share))])
    w = np.abs(cum[np.clip(e, 0, n - 1) + 1] - cum[np.clip(s, 0, n - 1)])
    w = w * len(w) / w.sum() if w.sum() > 0 else np.ones(len(w))
    return pd.Series(w, index=labels.index, name="attribution_weight")


def time_decay(weights: pd.Series, last_weight: float = 1.0) -> pd.Series:
    """Decay sample weights with the cumulative uniqueness of the sample, oldest to newest. ``last_weight = 1`` is no decay; ``0`` makes the oldest weight zero (linear); between 0 and 1 the oldest keeps that share of the newest; below 0 the oldest
    ``|last_weight|`` of the sample (by cumulative weight) gets zero weight."""
    if not -1.0 <= last_weight <= 1.0:
        raise ValueError("last_weight must be between -1 and 1")
    cum = weights.cumsum()
    total = cum.iloc[-1]
    slope = (1.0 - last_weight) / total if last_weight >= 0 else 1.0 / ((last_weight + 1.0) * total)
    const = 1.0 - slope * total
    f = const + slope * cum
    f[f < 0] = 0.0
    return f.rename("decay")


def sequential_bootstrap(labels: pd.DataFrame, n_rows: int, n_draws: int | None = None, seed: int = 0) -> np.ndarray:
    """Draw ``n_draws`` labels (default: as many as there are) one at a time, each with probability proportional to the uniqueness it *would add* to the labels already drawn. Overlapping labels are drawn less often together,
    so the average uniqueness of the bootstrap sample is higher than that of an ordinary bootstrap. Returns the integer positions drawn (with repeats), to index ``labels`` with."""
    k = len(labels)
    if k == 0:
        return np.array([], dtype=int)
    s, e = spans(labels)
    s, e = np.clip(s, 0, n_rows - 1), np.clip(e, 0, n_rows - 1)
    rng = np.random.default_rng(seed)
    count = np.zeros(n_rows)
    drawn = []
    for _ in range(int(n_draws or k)):
        inv = 1.0 / (count + 1.0)                                                           # the concurrency each day would have if the candidate were added
        cum = np.concatenate([[0.0], np.cumsum(inv)])
        added = (cum[e + 1] - cum[s]) / (e - s + 1)
        p = added / added.sum()
        j = int(rng.choice(k, p=p))
        drawn.append(j)
        count[s[j]:e[j] + 1] += 1.0
    return np.array(drawn, dtype=int)


def bootstrap_uniqueness(labels: pd.DataFrame, n_rows: int, draws: np.ndarray) -> float:
    """Average uniqueness of a bootstrap sample (repeats count as separate labels), to compare ordinary and sequential draws."""
    s, e = spans(labels.iloc[draws])
    return float(iv.average_uniqueness(s, e, n_rows).mean())
