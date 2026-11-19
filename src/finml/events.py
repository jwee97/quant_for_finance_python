"""Event sampling and triple-barrier labelling (Lopez de Prado, *Advances in Financial Machine Learning*, chapters 2 and 3).

Two pieces. A **CUSUM filter** chooses *when* to look: it emits an event when the price has moved by more than a threshold since the last event, in either direction, rather than every day. The **triple-barrier
method** then labels each event by what happens first on the path after it: the profit-taking barrier, the stop-loss barrier or the vertical barrier (a maximum holding period). Every label carries the
timestamp at which it was decided (``t1``), which is what purging, embargo and sample uniqueness need.

Conventions, all chosen so that a label can never use information the trade could not have had:

* An event at row ``i`` with ``entry_lag = L`` is entered at the close of row ``i + L`` (``L = 1`` is "the signal is read at the close and filled at the next close"). Barriers are watched from row ``i + L + 1``.
* Barriers sit ``pt`` and ``sl`` times the *target* (a volatility known at ``i``, as a fraction of the entry price) above and below it. For a short (``side = -1``) the profit barrier is the lower one.
* With only closes, a barrier is touched when a close is beyond it, and the exit is that close. With ``high`` and ``low``, a barrier is touched when the bar's range reaches it and the exit is the barrier level.
  A bar whose range reaches both barriers cannot be ordered from OHLC data: ``same_bar`` says what to assume (``"stop"``, the default, takes the loss; ``"profit"`` takes the gain; ``"drop"`` removes the event).
  Use intraday data to resolve it if the choice matters.
* An event is labelled only if its whole vertical window exists, whether or not a barrier was touched early: which events are labelled never depends on what happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SAME_BAR = ("stop", "profit", "drop")


def daily_vol(close: pd.Series, span: int = 100, min_periods: int = 20) -> pd.Series:
    """Exponentially weighted standard deviation of daily simple returns: the usual *target* for barrier widths. Known at the end of each row."""
    r = close.astype(float).pct_change()
    return r.ewm(span=span, min_periods=min_periods).std()


def cusum_events(series: pd.Series, threshold) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter on the changes of ``series`` (a price or a log price). ``threshold`` is a number or a Series aligned to ``series`` (a volatility, say), in the units of the changes.

    ``s+ = max(0, s+ + dy)`` and ``s- = min(0, s- + dy)``; an event is emitted, and both sums reset, when ``s+ > h`` or ``s- < -h``. The threshold at the row where the sum is checked is the one used, and it must
    be known at that row. Returns the event times."""
    y = series.astype(float)
    dy = y.diff().to_numpy()
    h = threshold.reindex(y.index).to_numpy(dtype=float) if isinstance(threshold, pd.Series) else np.full(len(y), float(threshold))
    out, pos, neg = [], 0.0, 0.0
    for i in range(1, len(y)):
        if not (np.isfinite(dy[i]) and np.isfinite(h[i]) and h[i] > 0):
            continue
        pos, neg = max(0.0, pos + dy[i]), min(0.0, neg + dy[i])
        if neg < -h[i]:
            neg = 0.0
            out.append(y.index[i])
        elif pos > h[i]:
            pos = 0.0
            out.append(y.index[i])
    return pd.DatetimeIndex(out) if out else pd.DatetimeIndex([], dtype=y.index.dtype)


def triple_barrier_events(close: pd.Series, events, target: pd.Series, *, pt: float = 1.0, sl: float = 1.0, vertical_bars: int = 10, side: pd.Series | None = None, entry_lag: int = 0,
                          high: pd.Series | None = None, low: pd.Series | None = None, same_bar: str = "stop", min_target: float = 0.0, cost: float = 0.0) -> pd.DataFrame:
    """Label ``events`` by the first barrier touched. Returns one row per labelled event, indexed by the event time ``t0``:

    ``t1`` the time the label was decided (the touch, or the vertical barrier), ``entry``/``exit`` prices, ``ret`` the return of the position (``side`` x price change, less ``cost`` as a fraction of the
    entry price), ``barrier`` ("pt", "sl" or "vertical"), ``target`` the barrier width used, ``side`` (+1 / -1; +1 when ``side`` is not given), ``bin`` the label, ``pos0`` / ``pos_entry`` / ``pos1`` the integer rows of ``t0``, of the entry and of ``t1`` (what purging and uniqueness use), and ``ambiguous``: True where the bar that ended the label touched both barriers
    (OHLC data cannot say which came first; ``same_bar`` decided).

    ``bin`` is, with a ``side`` given (a primary signal: this is the *meta-label*), 1 if the position made money after ``cost`` and 0 if not. Without ``side``, it is the sign of the return of the touched barrier
    (+1 / -1), and 0 for a vertical barrier whose return is smaller in size than half a barrier."""
    if same_bar not in SAME_BAR:
        raise ValueError(f"same_bar must be one of {SAME_BAR}")
    if (high is None) != (low is None):
        raise ValueError("give both high and low, or neither")
    if vertical_bars < 1 or pt < 0 or sl < 0 or (pt == 0 and sl == 0 and vertical_bars is None):
        raise ValueError("vertical_bars must be at least 1 and the barrier multiples non-negative")
    px = close.astype(float)
    idx = px.index
    c = px.to_numpy()
    hi = c if high is None else high.reindex(idx).to_numpy(dtype=float)
    lo = c if low is None else low.reindex(idx).to_numpy(dtype=float)
    ev = pd.DatetimeIndex(events) if not isinstance(events, pd.DatetimeIndex) else events
    ev = ev[ev.isin(idx)]
    tgt = target.reindex(idx).to_numpy(dtype=float)
    sd = None if side is None else side.reindex(idx).to_numpy(dtype=float)
    rows = []
    n = len(c)
    for t0 in ev:
        i = int(idx.get_loc(t0))
        e = i + entry_lag
        end = e + vertical_bars
        w = tgt[i]
        s = 1.0 if sd is None else sd[i]
        if end >= n or not np.isfinite(w) or w <= min_target or not np.isfinite(s) or s == 0 or not np.isfinite(c[e]):
            continue
        entry = c[e]
        up, dn = entry * (1 + (pt if s > 0 else sl) * w), entry * (1 - (sl if s > 0 else pt) * w)
        up_on, dn_on = (pt if s > 0 else sl) > 0, (sl if s > 0 else pt) > 0                 # a multiple of zero switches that barrier off
        j_exit, how, exit_px, both = end, "vertical", c[end], False
        for j in range(e + 1, end + 1):
            hit_up = up_on and hi[j] >= up
            hit_dn = dn_on and lo[j] <= dn
            if not (hit_up or hit_dn):
                continue
            if hit_up and hit_dn:
                both = True
                if same_bar == "drop":
                    j_exit = -1
                    break
                take_up = (same_bar == "profit") == (s > 0)                                   # "stop": the barrier that loses money for this side; "profit": the one that makes it
                hit_up, hit_dn = take_up, not take_up
            j_exit = j
            if high is None:
                exit_px = c[j]
            else:
                exit_px = up if hit_up else dn
            how = ("pt" if s > 0 else "sl") if hit_up else ("sl" if s > 0 else "pt")
            break
        if j_exit < 0 or not np.isfinite(exit_px):
            continue
        ret = s * (exit_px / entry - 1.0) - cost
        if sd is not None:
            lab = 1.0 if ret > 0 else 0.0
        else:
            lab = float(np.sign(ret)) if how != "vertical" or abs(ret) >= 0.5 * w else 0.0
        rows.append((t0, idx[j_exit], entry, exit_px, ret, how, w, s, lab, i, e, j_exit, both))
    cols = ["t0", "t1", "entry", "exit", "ret", "barrier", "target", "side", "bin", "pos0", "pos_entry", "pos1", "ambiguous"]
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out.set_index("t0")
    out["side"] = out["side"].astype(int)
    out["pos0"], out["pos_entry"], out["pos1"] = out["pos0"].astype(int), out["pos_entry"].astype(int), out["pos1"].astype(int)
    return out.set_index("t0")


def label_summary(labels: pd.DataFrame) -> dict:
    """What the labels look like: how many, which barrier ended them, the share of each class, and the holding period."""
    if labels.empty:
        return {"events": 0}
    hold = (labels["pos1"] - labels["pos0"]).to_numpy()
    return {"events": int(len(labels)), "barrier": {k: int(v) for k, v in labels["barrier"].value_counts().items()},
            "classes": {str(k): int(v) for k, v in labels["bin"].value_counts().sort_index().items()},
            "mean_return": float(labels["ret"].mean()), "mean_holding_bars": float(hold.mean()), "max_holding_bars": int(hold.max())}
