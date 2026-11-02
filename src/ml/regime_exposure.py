"""Causal regime exposure: a risk budget by market state, fitted on training evidence only (spec v3, Release 7, P2-11).

A static book takes the same risk in a calm market and in a turbulent one. Here the exposure of the book follows the market's state, and every part of the rule is causal and walk-forward:

* **State.** The state on date ``t`` is built from returns up to the close of ``t`` (the equal-weight return, its trailing volatility and the cross-sectional dispersion). States are **filtered**: the
  probability of each regime given the data up to ``t`` and parameters fitted *before* ``t``. Smoothed probabilities (which let later observations rewrite earlier ones) and calendars of crises chosen
  with hindsight are not available as engines, and the configuration refuses them by name.
* **Numbering.** At every refit the states are put in the order of their mean volatility (0 is the calmest), so a number means the same thing before and after a refit; a refit that reshuffles the
  engine's own labels cannot change what a state is mapped to. Whether the labels were reshuffled is recorded.
* **Budget.** At each refit the parameters are fitted on the rows before it, and each state's budget is read from what the *unscaled book* earned the day after the state was observed over those same rows:
  ``exposure_s = clip(vol_all / vol_s, floor, cap)`` (the same risk in every state, and never more than ``cap`` <= 1: the overlay lowers the exposure of a book, it does not lever it past the limits
  the book was built within). A state seen on fewer than ``min_days`` rows has insufficient evidence and gets the neutral exposure 1, which is recorded.
* **Use.** From a refit to the next, the book of decision date ``t`` is multiplied by ``sum_s P(s | data to t) * exposure_s``. Before the first refit (``regime_min_train`` rows of the book's own history, 252
  by default) the multiplier is 1. The book exists only where the experiment predicted, so on a short development-mode sample the overlay may never act; ``warmup_days`` in the log says so.

The multiplier is applied last, after the trade filter and the no-trade region: it is a risk control on the whole book, not a forecast, and a risk cut must not be vetoed by a rule that weighs expected return
against cost. Its effect on risk is reported with the experiment (and the comparison harness matches risk before it compares drawdowns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "regime-exposure-v1"
ENGINES = ("vol_quantile", "hmm", "markov_switching")
MIN_TRAIN = 252                                                      # rows of evidence before the first fit
COLUMNS = ("mkt_ret", "log_vol", "log_disp")
LOG_VOL = COLUMNS.index("log_vol")


def market_features(returns: pd.DataFrame, investable: pd.DataFrame | None = None) -> pd.DataFrame:
    """The market's state variables on every date, each row from that date's returns and earlier ones only (the same construction the regime models use)."""
    r = returns if investable is None else returns.where(investable.reindex_like(returns).fillna(False).astype(bool))
    mkt = r.mean(axis=1)
    vol = mkt.rolling(21, min_periods=15).std(ddof=1) * np.sqrt(252.0)
    disp = r.std(axis=1, ddof=0).rolling(5, min_periods=3).mean()
    out = pd.DataFrame({"mkt_ret": mkt, "log_vol": np.log(vol.clip(lower=1e-4)), "log_disp": np.log(disp.clip(lower=1e-5))}, index=returns.index)
    return out.replace([np.inf, -np.inf], np.nan)


class VolQuantileEngine:
    """States are the bands of the market's trailing volatility, with the band edges taken from the training rows (terciles for three states). The state of a date is known at its close: one-hot."""

    name = "vol_quantile"

    def __init__(self, n_states: int, seed: int = 0):
        self.k = int(n_states)

    def fit(self, Z: np.ndarray) -> "VolQuantileEngine":
        v = Z[:, LOG_VOL]
        v = v[np.isfinite(v)]
        if len(v) < 60:
            raise ValueError("fewer than 60 rows of market volatility to fit the bands on")
        self.edges_ = np.quantile(v, np.arange(1, self.k) / self.k)
        return self

    def filter(self, Z: np.ndarray) -> np.ndarray:
        v = Z[:, LOG_VOL]
        out = np.zeros((len(Z), self.k))
        ok = np.isfinite(v)
        out[np.flatnonzero(ok), np.searchsorted(self.edges_, v[ok], side="right")] = 1.0
        for t in range(len(Z)):                                      # a date without a state keeps the previous belief (nothing after it is used)
            if not ok[t]:
                out[t] = out[t - 1] if t else 1.0 / self.k
        return out

    def facts(self) -> dict:
        return {"edges": [float(e) for e in self.edges_]}


def make_engine(name: str, n_states: int, seed: int = 0):
    if name == "vol_quantile":
        return VolQuantileEngine(n_states, seed)
    from . import regimes

    if name == "hmm":
        try:
            import hmmlearn  # noqa: F401
        except ImportError:
            raise ValueError("the hmm engine needs the hmmlearn package, which is not installed here") from None
        return regimes.HMMEngine(n_states, seed=seed)
    if name == "markov_switching":
        return regimes.MarkovSwitchingEngine(n_states, seed=seed)
    raise ValueError(f"unknown regime engine '{name}'; the causal engines are {list(ENGINES)}")


def order_by_volatility(P: np.ndarray, log_vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The permutation that puts the columns of ``P`` (state probabilities on the training rows) in the order of their probability-weighted mean volatility, calmest first, with those means."""
    ok = np.isfinite(log_vol)
    w = np.nan_to_num(P[ok])
    mean = (w * log_vol[ok, None]).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-12)
    mean = np.where(w.sum(axis=0) > 1e-9, mean, np.inf)              # a state that never occurs is the last
    return np.argsort(mean, kind="stable"), mean


def exposures(P: np.ndarray, nxt: np.ndarray, floor: float, cap: float, min_days: int) -> tuple[np.ndarray, list[dict]]:
    """The exposure of each state from the training rows: ``clip(vol_all / vol_state, floor, cap)`` of the unscaled book's next-day returns, and neutral (1) where the state has fewer than ``min_days``
    observations or no spread. Returns the exposures and a table of the evidence behind each."""
    ok = np.isfinite(nxt)
    vol_all = float(np.std(nxt[ok], ddof=1)) if ok.sum() > 2 else float("nan")
    state = P.argmax(axis=1)
    out, table = np.ones(P.shape[1]), []
    for s in range(P.shape[1]):
        m = ok & (state == s)
        n = int(m.sum())
        vol = float(np.std(nxt[m], ddof=1)) if n >= 2 else float("nan")
        if n < min_days or not np.isfinite(vol) or vol <= 0 or not vol_all > 0:
            table.append({"state": s, "n": n, "vol": vol if np.isfinite(vol) else None, "exposure": 1.0, "status": "insufficient evidence: the neutral exposure 1"})
            continue
        out[s] = float(np.clip(vol_all / vol, floor, cap))
        table.append({"state": s, "n": n, "vol": vol, "exposure": float(out[s]), "status": "estimated"})
    return out, table


def apply(book: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Scale ``book`` (weights stamped at their decision dates) by the causal regime exposure; returns the scaled book and the log."""
    engine = str(cfg["regime_exposure"])
    k, floor, cap = int(cfg.get("regime_states", 3)), float(cfg.get("regime_floor", 0.25)), float(cfg.get("regime_cap", 1.0))
    every, min_days, min_train = int(cfg.get("regime_refit_every", 63)), int(cfg.get("regime_min_days", 60)), int(cfg.get("regime_min_train", MIN_TRAIN))
    rets = returns.reindex(index=book.index, columns=book.columns)
    Z = market_features(rets, None if investable is None else investable.reindex(index=book.index, columns=book.columns)).to_numpy()
    active = (book.abs().sum(axis=1) > 0).to_numpy()
    nxt = (book * rets.shift(-1)).sum(axis=1, min_count=1).to_numpy()                            # the unscaled book's return the day after each decision: known one day later
    nxt = np.where(active, nxt, np.nan)                                                         # days with no book carry no evidence
    usable = np.isfinite(Z).all(axis=1)
    T = len(book)
    mult = np.ones(T)
    ready = np.cumsum(usable & np.isfinite(nxt))
    start = int(np.argmax(ready >= min_train)) if ready[-1] >= min_train else T
    log = {"version": VERSION, "engine": engine, "states": k, "floor": floor, "cap": cap, "refit_every": every, "min_days": min_days, "min_train_rows": min_train, "warmup_days": int(min(start, T)),
           "refits": [], "failed_refits": 0, "smoothing": "none: filtered probabilities only", "retrospective_labels": "none"}
    r = start
    while r < T:
        end = min(r + every, T)
        try:
            eng = make_engine(engine, k, 0).fit(Z[:r][usable[:r]])
            P = eng.filter(Z[:end])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:      # a model that cannot be fitted leaves the exposure neutral for that stretch, and says so
            log["failed_refits"] += 1
            log["refits"].append({"date": str(book.index[r].date()), "status": "failed", "reason": str(exc)[:200]})
            r = end
            continue
        order, mean_lv = order_by_volatility(P[:r][usable[:r]], Z[:r, LOG_VOL][usable[:r]])
        P = P[:, order]
        e, table = exposures(P[:r], nxt[:r], floor, cap, min_days)
        mult[r:end] = P[r:end] @ e
        for row, lv in zip(table, mean_lv[order]):
            row["mean_log_vol"] = float(lv) if np.isfinite(lv) else None
        log["refits"].append({"date": str(book.index[r].date()), "status": "fitted", "labels_reordered": bool((order != np.arange(len(order))).any()), "train_rows": int(usable[:r].sum()),
                              "ordering_calm_to_turbulent": bool(np.all(np.diff(mean_lv[order][np.isfinite(mean_lv[order])]) >= -1e-12)), "states": table, "engine_facts": eng.facts()})
        r = end
    scaled = book.mul(mult, axis=0)
    live = active & (np.arange(T) >= start)
    log["mean_multiplier"] = float(mult[live].mean()) if live.any() else None
    log["share_at_floor"] = float((mult[live] <= floor + 1e-9).mean()) if live.any() else None
    log["min_multiplier"] = float(mult[live].min()) if live.any() else None
    log["n_refits"] = int(sum(1 for x in log["refits"] if x["status"] == "fitted"))
    log["note"] = ("the exposure follows the filtered market state with budgets read from the unscaled book's own past; it can lower a book and never lever it; before the first refit and wherever a state has too "
                   "little evidence the exposure is 1")
    return scaled, log
