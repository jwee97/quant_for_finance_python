"""The Market Regime Lab: what state the market is in, how long states last, how a return stream behaves in each, and how far to trust the labels.

Two kinds of regime estimate are kept apart everywhere, because mixing them is the commonest way to flatter a regime strategy:

* **filtered** (causal): the probability of each state on day *t* uses only data up to *t*, and the model parameters come from a fit that stopped before *t* (an expanding window refitted every
  ``refit_every`` days; ``models.regimes.walk_forward_hmm``). A rule could have traded on it as it stood.
* **smoothed** (retrospective): one fit on the whole sample and the backward pass that lets tomorrow's data decide today's label. It is the better *description* of what happened and it cannot be
  traded on. It is returned only so the two can be compared, and every use of it carries the label ``retrospective``.

Performance by regime uses the label known at the close of day *t* against the return of day *t + 1* (``lag=1``, the default): the return that a position taken on the label could earn. ``lag=0``
shows the return of the same day, which for a volatility regime is partly the day that defined it; it is labelled descriptive. A regime with fewer than ``min_days`` days has *insufficient evidence*
and no number, never a zero.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..ml import inference as inf
from ..models import regimes as mr

ANN = 252.0
UNCERTAIN = 0.6                                                                   # a day whose most likely state has less probability than this is an uncertain day
STATE_NAMES = {2: ("LowVol", "HighVol"), 3: ("LowVol", "MidVol", "HighVol")}
MIN_DAYS = 60


def state_names(k: int) -> list[str]:
    return list(STATE_NAMES.get(k, tuple(f"State{i}" for i in range(k))))


def _num(x):
    return None if x is None or (isinstance(x, (float, np.floating)) and not math.isfinite(float(x))) else float(x)


# ------------------------------------------------------------------------------------------------------ the estimates
def filtered_hmm(proxy: pd.Series, n_states: int = 2, min_train: int = 750, refit_every: int = 126, seed: int = 11, n_init: int = 3) -> dict:
    """The causal estimate: probabilities on every day from ``min_train`` on (earlier days have no estimate), and the parameters of every refit, which say how stable the states are."""
    if n_states not in (2, 3):
        raise ValueError("choose 2 or 3 states")
    if len(proxy) < min_train + 60:
        raise ValueError(f"{len(proxy)} days of returns: at least {min_train + 60} are needed ({min_train} to fit the first model and 60 to read it)")
    wf = mr.walk_forward_hmm(proxy.to_frame("proxy"), n_states, min_train, refit_every, n_init, 1, seed=seed)
    names = state_names(n_states)
    probs = wf.prob.copy()
    probs.columns = names
    fits = []
    for w in wf.windows:
        fit = w["fit"]
        fits.append({"date": str(w["date"].date()), "vol": [float(math.sqrt(c[0, 0] * ANN)) for c in fit.covars], "mean": [float(m[0] * ANN) for m in fit.means],
                     "stay": [float(fit.transmat[i, i]) for i in range(len(names))], "degenerate": bool(fit.is_degenerate)})
    last = wf.windows[-1]["fit"] if wf.windows else None
    return {"probabilities": probs, "fits": fits, "transmat": None if last is None else np.asarray(last.transmat), "names": names,
            "settings": {"n_states": n_states, "min_train": min_train, "refit_every": refit_every, "seed": seed}}


def smoothed_hmm(proxy: pd.Series, n_states: int = 2, seed: int = 11, n_init: int = 3) -> pd.DataFrame:
    """The retrospective estimate: one fit on the whole sample and the backward pass. Never to be traded on."""
    out = mr.full_sample_hmm(proxy.to_frame("proxy"), n_states, n_init=n_init, seed=seed)["smoothed"].copy()
    out.columns = state_names(n_states)
    return out


def labels_from(probs: pd.DataFrame) -> pd.Series:
    """The most likely state of each day, ``NaN`` where there is no estimate."""
    valid = probs.dropna(how="all")
    lab = valid.idxmax(axis=1)
    return lab.reindex(probs.index)


# ------------------------------------------------------------------------------------------------------ what the labels say
def transitions(labels: pd.Series, names: list[str]) -> dict:
    """The observed day-to-day transitions between the hard labels: counts, and the share of each state's days that were followed by each state."""
    lab = labels.dropna()
    idx = {n: i for i, n in enumerate(names)}
    counts = np.zeros((len(names), len(names)), dtype=int)
    for a, b in zip(lab.iloc[:-1], lab.iloc[1:]):
        counts[idx[a], idx[b]] += 1
    rows = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, rows, out=np.full(counts.shape, np.nan), where=rows > 0)
    return {"states": names, "counts": counts.tolist(), "probabilities": [[_num(round(float(v), 4)) for v in row] for row in share]}


def durations(labels: pd.Series, names: list[str]) -> list[dict]:
    """How long each state lasts when it occurs: the number of spells and the mean, median and longest spell in trading days."""
    lab = labels.dropna()
    out = []
    if len(lab):
        run = (lab != lab.shift()).cumsum()
        spells = lab.groupby(run).agg(["first", "size"])
    for n in names:
        sizes = spells.loc[spells["first"] == n, "size"].to_numpy() if len(lab) else np.array([])
        out.append({"state": n, "spells": int(len(sizes)), "mean_days": _num(sizes.mean()) if len(sizes) else None, "median_days": _num(np.median(sizes)) if len(sizes) else None,
                    "longest_days": int(sizes.max()) if len(sizes) else None, "days": int((lab == n).sum()), "share": _num((lab == n).mean()) if len(lab) else None})
    return out


def performance_by_regime(returns: pd.Series, labels: pd.Series, *, lag: int = 1, min_days: int = MIN_DAYS, frequency: str = "daily", n_boot: int = 300) -> dict:
    """The behaviour of ``returns`` in each regime. ``lag=1``: the label at the close of day *t* is paired with the return of day *t + 1*, so the number is what a position taken on the label could earn.
    Each regime with at least ``min_days`` days gets its days, share, annualised mean, volatility, Sharpe (naive, with a dependence-aware interval), hit rate and worst day; the rest say *insufficient*."""
    if lag not in (0, 1):
        raise ValueError("lag must be 0 (the same day, descriptive) or 1 (the next day, what a rule could earn)")
    r = returns.dropna().astype(float)
    lab = labels.dropna()
    lab = lab.shift(lag).reindex(r.index) if lag else lab.reindex(r.index)                # the label of the previous close for each return day
    rows = []
    both = lab.notna()
    for state in sorted({s for s in lab.dropna().unique()}):
        part = r[lab == state]
        row = {"state": state, "n_days": int(len(part)), "share": _num(len(part) / max(int(both.sum()), 1))}
        if len(part) < min_days:
            row.update(eligibility="insufficient", reason=f"{len(part)} days: fewer than the {min_days} needed for a reading")
        else:
            sd = float(part.std(ddof=1))
            mean = inf.hac_mean(part, frequency=frequency, label=f"regime={state}")
            sh = inf.sharpe_inference(part, frequency=frequency, label=f"regime={state}", n_boot=n_boot)
            row.update(eligibility="applied", ann_mean=_num(part.mean() * ANN), ann_vol=_num(sd * math.sqrt(ANN)), sharpe=_num(sh["naive"].value), sharpe_interval=list(map(_num, sh["hac"].interval)) if sh["hac"].interval else None,
                       mean_p=_num(mean.p_value), hit_rate=_num((part > 0).mean()), worst_day=_num(part.min()), best_day=_num(part.max()))
        rows.append(row)
    return {"lag": lag, "min_days": min_days, "basis": "next-day return after the label (what a rule could earn)" if lag else "same-day return (descriptive: includes the day that defined the label)",
            "rows": rows, "n_days": int(both.sum()), "n_unlabelled": int((~both).sum())}


def stability(probs: pd.DataFrame, labels: pd.Series, fits: list[dict] | None = None, smoothed_labels: pd.Series | None = None) -> dict:
    """How far to trust the labels: how confident they are, how often they flip, how much the parameters move between refits, and how different the retrospective view would be."""
    valid = probs.dropna(how="all")
    top = valid.max(axis=1)
    lab = labels.dropna()
    changes = int((lab != lab.shift()).sum() - 1) if len(lab) else 0
    years = max((lab.index[-1] - lab.index[0]).days / 365.25, 1e-9) if len(lab) > 1 else float("nan")
    swing = (lab != lab.shift()).to_numpy()                                           # a switch followed by another within 5 days is a flip-flop, not a regime
    flip = 0
    pos = np.flatnonzero(swing[1:]) + 1 if len(lab) > 1 else np.array([], dtype=int)
    for a, b in zip(pos[:-1], pos[1:]):
        if b - a <= 5:
            flip += 1
    out = {"n_days": int(len(valid)), "mean_confidence": _num(top.mean()) if len(top) else None, "uncertain_share": _num((top < UNCERTAIN).mean()) if len(top) else None, "uncertain_below": UNCERTAIN,
           "switches": max(changes, 0), "switches_per_year": _num(max(changes, 0) / years) if len(lab) > 1 else None, "short_spells_share": _num(flip / max(changes, 1)) if changes > 0 else 0.0}
    if fits:
        vols = np.array([f["vol"] for f in fits])
        out["parameter_drift"] = {"refits": len(fits), "vol_min": [_num(v) for v in vols.min(axis=0)], "vol_max": [_num(v) for v in vols.max(axis=0)], "degenerate_fits": int(sum(f["degenerate"] for f in fits)),
                                  "note": "the volatility each state stands for, across the refits: a wide range means the states are not the same thing from one refit to the next"}
    if smoothed_labels is not None:
        both = pd.concat([lab, smoothed_labels.reindex(lab.index)], axis=1).dropna()
        out["agreement_with_retrospective"] = _num((both.iloc[:, 0] == both.iloc[:, 1]).mean()) if len(both) else None
    return out


def analyse(proxy: pd.Series, filtered: dict, smoothed: pd.DataFrame | None = None, *, min_days: int = MIN_DAYS) -> dict:
    """Everything the lab page shows for one market proxy: the timeline of the filtered regime, its transitions, durations, the proxy's own behaviour by regime (next-day and same-day), the stability
    diagnostics, the latest reading and, if given, the retrospective estimate with how much it differs."""
    probs, names = filtered["probabilities"], filtered["names"]
    labels = labels_from(probs)
    valid = probs.dropna(how="all")
    sm_labels = labels_from(smoothed) if smoothed is not None else None
    last = valid.iloc[-1] if len(valid) else None
    dates = [str(d.date()) for d in valid.index]
    out = {"names": names, "settings": filtered["settings"], "first_estimate": dates[0] if dates else None, "as_of": dates[-1] if dates else None,
           "timeline": {"dates": dates, "label": [names.index(x) for x in labels.loc[valid.index]], "confidence": [_num(round(float(v), 3)) for v in valid.max(axis=1)],
                        "probabilities": {n: [_num(round(float(v), 3)) for v in valid[n]] for n in names}},
           "transitions": transitions(labels, names), "durations": durations(labels, names),
           "proxy_by_regime": {"next_day": performance_by_regime(proxy, labels, lag=1, min_days=min_days), "same_day": performance_by_regime(proxy, labels, lag=0, min_days=min_days)},
           "stability": stability(probs, labels, filtered.get("fits"), sm_labels),
           "current": None if last is None else {"as_of": dates[-1], "state": str(last.idxmax()), "probabilities": {n: _num(round(float(last[n]), 3)) for n in names}, "basis": "filtered: uses data up to and including this date"},
           "model_transition": None if filtered.get("transmat") is None else [[_num(round(float(v), 4)) for v in row] for row in filtered["transmat"]],
           "fits": filtered.get("fits"), "evidence": "walk_forward", "label_basis": "filtered (causal): each day's probabilities use only data up to that day and a model fitted before it"}
    if smoothed is not None:
        sv = smoothed.loc[valid.index]
        out["retrospective"] = {"label": "retrospective: uses the whole sample, including days after each label. Not tradeable.", "dates": dates, "label_index": [names.index(x) for x in sm_labels.loc[valid.index]],
                                "probabilities": {n: [_num(round(float(v), 3)) for v in sv[n]] for n in names}}
    return out


def market_proxy_stats(proxy: pd.Series) -> dict:
    """Plain facts about the series the regimes were estimated on, so the page can say what 'the market' means here."""
    return {"n_days": int(len(proxy)), "first": str(proxy.index[0].date()), "last": str(proxy.index[-1].date()), "ann_vol": _num(proxy.std(ddof=1) * math.sqrt(ANN)), "ann_mean": _num(proxy.mean() * ANN),
            "definition": "equal-weight daily return of the chosen assets"}
