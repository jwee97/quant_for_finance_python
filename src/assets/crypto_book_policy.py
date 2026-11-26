"""One rule for which recorded order book a historical decision may see, used by the store and by the replay alike.

The rule, in this order, for a decision at time ``t``:

1. take the LATEST row received at or before ``t`` (receipt time, the moment the system could have acted on it; nothing received later is ever visible, and the exchange's own event time is never used to look ahead);
2. check that row: its stored quality (``ok``, ``empty``, ``crossed``, ``gap``), that every price and quantity is finite and positive, that the best bid is below the best ask (a *locked* book, bid equal to ask, is
   refused like a crossed one), that its sequence does not go backwards or skip (``sequence_gap`` for an update whose ``prev_sequence`` is not the last sequence seen), that its exchange timestamp does not
   run backwards, is not older than ``max_exchange_lag_s`` before receipt (a stale feed) or ahead of receipt by more than ``max_clock_skew_s`` (a bad clock), and that it belongs to the instrument asked for;
3. check its age: ``t - recv`` at most ``max_age_s``;
4. if it fails, the answer is **no book**, with the reason. An older valid book is NOT substituted. Only when ``allow_last_valid_fallback`` is set is the latest earlier valid row used, and then only if it is
   within ``fallback_max_age_s``; the answer says so (``fallback_used``) together with the rejected newer row, so the experiment and each trade can record it.

Rows are ordered by receipt time and, for equal receipt times, by arrival order in the store, so an out-of-order write cannot move a row into the past of another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BookPolicy:
    max_age_s: float = 5.0
    max_exchange_lag_s: float | None = 30.0         # receipt time minus exchange time beyond this: a stale feed (None: not checked)
    max_clock_skew_s: float = 2.0                   # exchange time ahead of receipt by more than this: a clock problem
    allow_last_valid_fallback: bool = False
    fallback_max_age_s: float | None = None         # how old a fallback book may be (default: ``max_age_s``)

    def describe(self) -> dict:
        d = asdict(self)
        d["strict"] = not self.allow_last_valid_fallback
        return d


def _level_problem(px, qty) -> str | None:
    p, q = np.asarray(px, dtype=float), np.asarray(qty, dtype=float)
    if len(p) == 0 or len(q) == 0 or len(p) != len(q):
        return "empty_side"
    if not (np.isfinite(p).all() and np.isfinite(q).all()) or (p <= 0).any() or (q <= 0).any():
        return "invalid_level"
    return None


def annotate(df: pd.DataFrame, venue: str | None, instrument: str | None, policy: BookPolicy) -> pd.DataFrame:
    """The rows ordered by receipt (arrival order breaking ties) with ``valid`` and ``reason`` columns: whether each row, by itself and against the rows before it, may be used at all (age is checked at the decision)."""
    if df is None or not len(df):
        return pd.DataFrame(columns=list(df.columns if df is not None else []) + ["valid", "reason"])
    d = df.reset_index(drop=True).copy()
    d["_arrival"] = np.arange(len(d))
    d = d.sort_values(["recv_ts_ms", "_arrival"], kind="mergesort").reset_index(drop=True)
    reasons: list[str] = []
    max_seq: dict[str, int] = {}
    last_seq: int | None = None
    max_exch = None
    for _, r in d.iterrows():
        why = ""
        q = str(r.get("quality", "ok"))
        if q != "ok":
            why = f"quality_{q}"
        if not why and venue is not None and (str(r.get("venue", venue)).upper() != venue.upper() or str(r.get("instrument", instrument)).upper() != str(instrument).upper()):
            why = "identity_mismatch"
        if not why:
            why = _level_problem(r["bids_px"], r["bids_qty"]) or _level_problem(r["asks_px"], r["asks_qty"]) or ""
        if not why:
            bid, ask = float(r["bids_px"][0]), float(r["asks_px"][0])
            why = "crossed" if bid > ask else "locked" if bid == ask else ""
        recv = int(r["recv_ts_ms"])
        ex = None if pd.isna(r.get("exchange_ts_ms")) else int(r["exchange_ts_ms"])
        if not why and ex is not None:
            if policy.max_exchange_lag_s is not None and recv - ex > policy.max_exchange_lag_s * 1000:
                why = "stale_exchange_timestamp"
            elif ex - recv > policy.max_clock_skew_s * 1000:
                why = "exchange_timestamp_in_future"
            elif max_exch is not None and ex < max_exch:
                why = "exchange_timestamp_regression"
        if ex is not None:
            max_exch = ex if max_exch is None else max(max_exch, ex)
        seq = None if pd.isna(r.get("sequence")) else int(r["sequence"])
        prev = None if pd.isna(r.get("prev_sequence")) else int(r["prev_sequence"])
        if not why and seq is not None:
            key = str(r.get("kind", ""))
            if key in max_seq and seq < max_seq[key]:
                why = "out_of_order_sequence"
            elif r.get("kind") == "update" and prev is not None and last_seq is not None and prev != last_seq:
                why = "sequence_gap"
        if seq is not None:
            max_seq[str(r.get("kind", ""))] = max(seq, max_seq.get(str(r.get("kind", "")), seq))
            last_seq = seq
        reasons.append(why)
    d["reason"] = reasons
    d["valid"] = [x == "" for x in reasons]
    return d.drop(columns=["_arrival"])


def _book(r) -> dict:
    return {"recv_ts_ms": int(r["recv_ts_ms"]), "exchange_ts_ms": None if pd.isna(r["exchange_ts_ms"]) else int(r["exchange_ts_ms"]), "sequence": None if pd.isna(r["sequence"]) else int(r["sequence"]),
            "bids": list(zip(r["bids_px"], r["bids_qty"])), "asks": list(zip(r["asks_px"], r["asks_qty"])), "kind": r.get("kind"), "venue": r.get("venue"), "instrument": r.get("instrument")}


def decide(d: pd.DataFrame, t_ms: int, policy: BookPolicy, max_age_s: float | None = None) -> dict:
    """The decision at ``t_ms`` from an annotated frame: ``{"book", "status", "reason", "fallback_used", "rejected"}``. ``status`` is ``ok``, ``fallback`` or ``none``."""
    age_limit = (policy.max_age_s if max_age_s is None else max_age_s) * 1000
    if d is None or not len(d):
        return {"book": None, "status": "none", "reason": "no_book_recorded", "fallback_used": False, "rejected": None}
    t = np.asarray(d["recv_ts_ms"].to_numpy(), dtype=np.int64)
    i = int(np.searchsorted(t, int(t_ms), side="right")) - 1                         # the latest row at or before the decision: nothing later is visible
    if i < 0:
        return {"book": None, "status": "none", "reason": "no_book_before_decision", "fallback_used": False, "rejected": None}
    r = d.iloc[i]
    age = int(t_ms) - int(r["recv_ts_ms"])
    why = "" if r["valid"] else str(r["reason"])
    if not why and age > age_limit:
        why = "stale"
    if not why:
        return {"book": {**_book(r), "age_s": age / 1000.0}, "status": "ok", "reason": "", "fallback_used": False, "rejected": None}
    rejected = {"recv_ts_ms": int(r["recv_ts_ms"]), "reason": why, "age_s": age / 1000.0, "sequence": None if pd.isna(r["sequence"]) else int(r["sequence"])}
    if policy.allow_last_valid_fallback:
        limit = (policy.fallback_max_age_s if policy.fallback_max_age_s is not None else policy.max_age_s if max_age_s is None else max_age_s) * 1000
        for j in range(i - 1, -1, -1):
            if d.iloc[j]["valid"]:
                a2 = int(t_ms) - int(d.iloc[j]["recv_ts_ms"])
                if a2 <= limit:
                    return {"book": {**_book(d.iloc[j]), "age_s": a2 / 1000.0, "fallback": True}, "status": "fallback", "reason": why, "fallback_used": True, "rejected": rejected}
                break
    return {"book": None, "status": "none", "reason": why, "fallback_used": False, "rejected": rejected}
