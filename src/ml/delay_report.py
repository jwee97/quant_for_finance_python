"""Delay report: IC, expected edge, net return and capacity at predeclared observation-to-execution delays (spec v3, Release 7, P2-15).

A signal observed at the close of day ``t`` is worth less the longer it waits to be traded. The report measures how much less, at delays that were **declared before the numbers were seen** (``declare`` fixes
them and returns a hash the result cites), on the validation segment of a saved experiment:

* **Timing from the actual timestamps.** A delay is a number of sessions on the dataset's own date index, and the report says how long that was on the calendar: a decision on a Friday filled one
  session later waits three days, not one. The IC is measured from the same fill the portfolio gets: the forward return starts ``lag + extra_lag + delay`` sessions after the observation.
* **The stale-book rule.** A delayed portfolio executes the book that was *decided at the observation close*, ``delay`` sessions later. No rule re-reads the signal at the later time: a delayed book cannot see
  a newer signal unless a rule that does so is defined explicitly, and none is defined here. (The tests build a book whose every position earns only at its undelayed fill, and check that a delay of one
  session earns nothing.)
* **Edge, net result, capacity.** The expected edge is the gross return the executed book earned per year and per unit traded (``breakeven_cost_bps``: the one-way cost at which the net result would be
  zero, next to the cost the experiment declared); the net result is the same engine's net return with its costs. Capacity is the share of the book one session's participation limit lets through at each
  predeclared capital, at the delayed execution dates; without a volume record it is *not established*, not assumed.

Nothing here refits a model or re-runs the experiment's pipeline: the saved scores and the saved book are shifted in time through the same engine. Nothing is chosen from the table, and the protected
segments are not read unless a frozen plan evaluated them.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import alpha as al
from . import evaluation as ev
from .backtest import annualised_turnover, label_days, run_backtest, trim_to_signal
from .targets import forward_return

VERSION = "delay-report-v1"
DEFAULT_DELAYS = (0, 1, 2, 5)
DEFAULT_CAPITALS = (1e5, 1e6, 1e7, 1e8)
MAX_DELAY = 20
ANN = 252.0


def declare(delays=DEFAULT_DELAYS, capitals=DEFAULT_CAPITALS) -> dict:
    """The predeclared delays (sessions, including 0) and capital levels, with the hash the result will cite."""
    ds = sorted({int(d) for d in delays})
    if not ds or ds[0] != 0 or ds[-1] > MAX_DELAY or any(int(d) != d for d in delays):
        raise ValueError(f"the delays are whole numbers of sessions from 0 (no delay) to {MAX_DELAY}, and 0 must be among them: it is what the others are compared with")
    cs = sorted({float(c) for c in capitals})
    if any(c <= 0 for c in cs):
        raise ValueError("capital levels are positive")
    body = {"delays": ds, "capitals": cs, "version": VERSION}
    return {**body, "hash": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]}


def timing(index: pd.DatetimeIndex, decisions: pd.DatetimeIndex, execution_lag: int) -> dict:
    """When positions decided on ``decisions`` are filled, on the real date index: the session ``execution_lag`` sessions later (the position is established at that session's close and earns from the
    next one), and the calendar days that took. A decision with no such session in the data (the last few) is counted and left out, not extrapolated."""
    pos = index.get_indexer(pd.DatetimeIndex(decisions))
    ok = pos >= 0
    target = pos[ok] + int(execution_lag)
    inside = target < len(index)
    gaps = (index[target[inside]] - index[pos[ok][inside]]).days.to_numpy().astype(float)
    n_dec = int(len(decisions))
    return {"execution_lag_sessions": int(execution_lag), "decisions": n_dec, "with_a_fill_session": int(inside.sum()), "without": int(n_dec - inside.sum()),
            "calendar_days": {"mean": float(gaps.mean()) if len(gaps) else None, "median": float(np.median(gaps)) if len(gaps) else None, "max": float(gaps.max()) if len(gaps) else None},
            "longer_than_sessions": float((gaps > execution_lag + 1e-9).mean()) if len(gaps) else None,
            "note": "the share of decisions whose wait spans more calendar days than sessions (weekends, holidays, closures)"}


def ic_at_delay(scores: pd.DataFrame, prices: pd.DataFrame, horizon: int, entry_lag: int, mask: np.ndarray, min_assets: int = 5) -> dict:
    """Rank IC of the scores of the masked dates against the ``horizon``-session return that starts ``entry_lag`` sessions after the observation: mean, Newey-West t (lag at least ``horizon - 1``) and count."""
    fwd = forward_return(prices, horizon, entry_lag).reindex(index=scores.index, columns=scores.columns)
    sc = scores[mask].dropna(how="all")
    if len(sc) == 0:
        return {"eligibility": "insufficient", "reason": "no scores on the segment"}
    tab = al.ic_table(sc, fwd.loc[sc.index], min_assets)
    out = al.ic_inference(tab, horizon)
    ic = tab["ic"].dropna()
    return {**out, "mean_ic": float(ic.mean()) if len(ic) else None, "entry_lag": int(entry_lag),
            "mean_ic_note": "the plain mean of the daily rank IC (descriptive: it exists even when the interval in `mean` cannot be estimated, for instance for a constant IC)"}


def book_at_delay(positions: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, bt_cfg: dict, lag: int, extra_lag: int, first_signal, plan_labels: pd.Series, segment: str) -> dict:
    """The stale book (decided at the observation close) executed ``extra_lag`` sessions later through the backtest engine: its gross, net, costs and turnover on the segment, and the executed weights."""
    res = run_backtest(positions, returns, investable, bt_cfg, lag, name="delay", extra_lag=int(extra_lag))
    parts = trim_to_signal(res, first_signal, lag)
    seg = label_days(parts["net"].index, plan_labels) == segment
    net, gross, costs, turn = (parts[k][seg.to_numpy()] for k in ("net", "gross", "costs", "turnover"))
    n = int(len(net.dropna()))
    out: dict = {"n_days": n, "engine": "vectorized backtest engine on the saved book"}
    if n < 30:
        return {**out, "eligibility": "insufficient", "reason": f"{n} days on the {segment} segment", "weights": parts["weights"]}
    eq = (1.0 + net.fillna(0.0)).cumprod()
    out.update(eligibility="applied", gross_ann_return=float(gross.mean() * ANN), net_ann_return=float(net.mean() * ANN), costs_ann=float(costs.mean() * ANN), sharpe=ev.sharpe_of(net.dropna()),
               max_drawdown=float((eq / eq.cummax() - 1.0).min()), ann_turnover=float(annualised_turnover(turn)),
               breakeven_cost_bps=float(1e4 * gross.sum() / turn.sum()) if float(turn.sum()) > 0 else None)
    out["weights"] = parts["weights"]
    return out


def capacity_at_delay(positions: pd.DataFrame, prices: pd.DataFrame, volume: pd.DataFrame | None, decisions: pd.DatetimeIndex, execution_lag: int, capitals, max_participation: float, adv_window: int,
                      max_dates: int = 12) -> dict:
    """The share of the book that one session's participation limit lets through at each capital, at the delayed execution dates (the median over a spread of decisions); not established without volume."""
    from . import realism as rz

    if volume is None:
        return {"status": "not_established", "reason": "no volume record: capacity is not assumed", "levels": []}
    adv = rz.pit_adv(volume.reindex(index=prices.index, columns=prices.columns), prices, adv_window)
    pos = prices.index.get_indexer(pd.DatetimeIndex(decisions))
    held = [(p, d) for p, d in zip(pos, decisions) if p >= 0 and p + execution_lag < len(prices) and positions.loc[d].abs().sum() > 0]
    if not held:
        return {"status": "not_established", "reason": "no decision held a position with a later session", "levels": []}
    step = max(1, len(held) // max_dates)
    rows = []
    for p, d in held[::step][:max_dates]:
        fill = prices.index[p + execution_lag]                                                       # the book decided at d is traded at this session, against that session's ADV
        cur = rz.capacity_curve(positions.loc[d].abs(), prices.loc[fill], adv.loc[fill], capitals, max_participation)
        rows.append(cur.set_index("capital")["fill_share"])
    tab = pd.concat(rows, axis=1)
    return {"status": "measured", "dates": int(tab.shape[1]), "max_participation": float(max_participation),
            "levels": [{"capital": float(c), "median_fill_share": float(tab.loc[c].median()), "worst_fill_share": float(tab.loc[c].min())} for c in tab.index],
            "note": "the share of the target one session's participation limit lets through, at the delayed execution date's ADV; the median and the worst of the sampled decisions"}


def report(registry, exp_id: str, *, segment: str = "validation", declaration: dict | None = None, volume: pd.DataFrame | None = None, max_participation: float | None = None, adv_window: int = 20,
           min_assets: int = 5) -> dict:
    """The delay report of a saved experiment on one segment (validation by default; a protected segment only if a frozen plan evaluated it)."""
    from . import diagnostics as dx
    from . import experiments as xp
    from .r6_reports import _guard

    _guard(registry, exp_id, segment)
    decl = declaration or declare()
    cfg, ds, train, _ = xp.load_run(registry, exp_id)
    scores = train.scores_wide(ds)
    mask = (label_days(scores.index, train.plan.labels()) == segment).to_numpy()
    positions = pd.read_parquet(registry.root / "experiments" / exp_id / "positions.parquet").reindex(ds.dates).fillna(0.0)
    lag, base_extra, h = int(ds.lag), int(cfg["backtest"].get("extra_lag", 0)), int(ds.horizon)
    first_signal = scores.dropna(how="all").index[0]
    decisions = scores.index[mask & scores.notna().any(axis=1).to_numpy()]
    out: dict = {"version": VERSION, "experiment_id": exp_id, "segment": segment, "declaration": decl, "horizon": h, "label_lag": lag, "extra_lag": base_extra,
                 "model": cfg["model"]["kind"], "target": cfg["dataset"]["target"],
                 "rule": "stale book: the book decided at the observation close is executed `delay` sessions later; no rule re-reads the signal at execution time",
                 "declared_cost_bps": float(cfg["backtest"]["cost_bps"]) + float(cfg["backtest"]["slippage_bps"]), "rows": []}
    if len(decisions) == 0:
        return dx._jsonable({**out, "eligibility": "insufficient", "reason": f"the experiment has no scores on the {segment} segment"})      # noqa: SLF001
    base: dict = {}
    for d in decl["delays"]:
        entry = lag + base_extra + int(d)
        row: dict = {"delay": int(d), "timing": timing(ds.dates, decisions, entry), "ic": ic_at_delay(scores, ds.prices, h, entry, mask, min_assets)}
        port = book_at_delay(positions, ds.returns, ds.investable, cfg["backtest"], lag, base_extra + int(d), first_signal, train.plan.labels(), segment)
        port.pop("weights", None)
        row["portfolio"] = port
        row["capacity"] = (capacity_at_delay(positions, ds.prices, volume, decisions, entry, decl["capitals"], max_participation, adv_window)
                           if max_participation is not None else {"status": "not_established", "reason": "no participation limit was given, so capacity is not assumed", "levels": []})
        if d == 0:
            base = row
        out["rows"].append(row)
    for row in out["rows"]:
        ic0, ic = (base.get("ic") or {}).get("mean_ic"), (row.get("ic") or {}).get("mean_ic")
        s0, s = (base.get("portfolio") or {}).get("sharpe"), (row.get("portfolio") or {}).get("sharpe")
        row["ic_ratio_to_undelayed"] = float(ic / ic0) if ic is not None and ic0 is not None and ic0 > 0 else None
        row["sharpe_ratio_to_undelayed"] = float(s / s0) if s is not None and s0 is not None and s0 > 0 else None
        row["ratio_note"] = "undefined (not 0, not 1) when the undelayed result is not positive: a ratio to a loss or to nothing means nothing"
    out["eligibility"] = "applied"
    out["caveats"] = ["a description of the validation segment at predeclared delays, not a selection of a delay", "net results are the vectorized engine's on the saved book; a realistic-execution experiment's own fills are not re-simulated here"]
    return dx._jsonable(out)                                                                                       # noqa: SLF001
