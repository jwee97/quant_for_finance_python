"""Alpha diagnostics (spec v3, Release 6, P2-02, P2-13 and P2-18).

What a signal is worth is a statement about its rank correlation with what followed, how long that lasts, how much of the universe it covers, whether it is stable across the sample, whether it says anything
the rest of the book does not, and how sure we can be given that the labels it is judged on overlap. Each piece here returns numbers *and the states in which a number does not exist*:

* ``ic_table`` - the cross-sectional rank IC of each date with the number of assets that entered it and a state: ``ok``, ``too_few`` (fewer assets than the declared minimum), ``tied_signal`` or ``tied_target`` (a
  constant cross-section has no rank correlation: it is unavailable, not zero). Ties get average ranks;
* ``ic_inference`` - the mean IC with a Newey-West standard error whose lag is at least ``horizon - 1``, because the IC of consecutive dates shares most of its label window (``inference.hac_mean``); an
  information ratio and a hit rate beside it, labelled as descriptive;
* ``decay_grid`` - the IC of the same signal against the return that starts ``k`` extra sessions later and runs ``h`` sessions, for a grid of ``(k, h)``: how fast it goes stale;
* ``coverage_report``, ``stability_report`` and ``redundancy_report`` - where the signal exists, whether its IC is the same in every part of the sample, and how much of a feature set is the same information
  twice (average cross-sectional rank correlation and the participation-ratio count of independent features);
* ``correlation_report`` (P2-13) - prediction correlation, portfolio-return correlation on the dates both exist (with the number dropped from each side, never silently), exposure overlap, concentration and the
  diversification a second book adds;
* ``conditional_ic`` and ``blend_value`` (P2-18) - the rank IC of a new signal *after* removing, date by date, what a set of existing signals explain of both the signal and the target (a partial rank
  correlation; a signal the existing ones reproduce exactly is ``explained_signal``, not zero), and the paired difference of a blend that includes the new book against the base book on aligned dates. These measure **predictive incremental value on the sample given**; they are not a causal
  claim and not a feature importance.

``development_only`` is the guard for diagnostics that drive a choice of features: it refuses the protected segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import inference as inf
from .targets import forward_return

VERSION = "alpha-v1"
DEVELOPMENT = ("train", "validation")
PROTECTED = ("test", "forward")


def development_only(frame, labels: pd.Series, segments=DEVELOPMENT):
    """The rows of ``frame`` (indexed by date, or by (date, asset)) whose segment is a development segment. Asking for a protected segment is refused: a diagnostic that picks features must not read them."""
    bad = [s for s in segments if s in PROTECTED]
    if bad:
        raise ValueError(f"diagnostics that drive a choice of features are computed on development data only; {bad} are protected until a frozen plan evaluates them")
    dates = frame.index.get_level_values(0) if isinstance(frame.index, pd.MultiIndex) else frame.index
    lab = labels.reindex(pd.DatetimeIndex(dates)).to_numpy()
    return frame[np.isin(lab, list(segments))]


def ic_table(signal: pd.DataFrame, target: pd.DataFrame, min_assets: int = 5) -> pd.DataFrame:
    """Per date: the Spearman correlation (average ranks for ties), the number of assets with both values and the state."""
    both = signal.notna() & target.notna() & np.isfinite(signal) & np.isfinite(target)
    n = both.sum(axis=1)
    s, t = signal.where(both), target.where(both)
    sr, tr = s.rank(axis=1), t.rank(axis=1)
    sc, tc = sr.sub(sr.mean(axis=1), axis=0), tr.sub(tr.mean(axis=1), axis=0)
    ss, tt = (sc ** 2).sum(axis=1), (tc ** 2).sum(axis=1)
    cov = (sc * tc).sum(axis=1)
    ic = cov / np.sqrt(ss * tt)
    state = pd.Series("ok", index=signal.index, dtype=object)
    state[n < min_assets] = "too_few"
    state[(n >= min_assets) & (ss <= 1e-12)] = "tied_signal"
    state[(n >= min_assets) & (ss > 1e-12) & (tt <= 1e-12)] = "tied_target"
    return pd.DataFrame({"ic": ic.where(state == "ok"), "n": n.astype(int), "state": state})


def rolling_ic(table: pd.DataFrame, window: int = 63, min_periods: int | None = None) -> pd.DataFrame:
    """The mean IC over the last ``window`` dates that have one, with how many that was. A window with fewer than ``min_periods`` (default a third of it) has no value."""
    mp = min_periods or max(window // 3, 5)
    ic = table["ic"]
    return pd.DataFrame({"mean": ic.rolling(window, min_periods=mp).mean(), "n_ok": ic.rolling(window, min_periods=1).count()})


def ic_inference(table: pd.DataFrame, horizon: int, *, level: float = 0.95, label: str = "") -> dict:
    """Mean IC with a HAC interval that accounts for the overlap of the labels (lag at least ``horizon - 1``), plus the information ratio and hit rate as descriptive numbers, and a count of the dates that had no IC by
    reason. The IC of consecutive dates is not independent, so the naive standard deviation over the square root of the count is not offered."""
    ic = table["ic"].dropna()
    res = inf.hac_mean(ic, horizon=int(horizon), level=level, label=label or "rank IC")
    states = table["state"].value_counts().to_dict()
    return {"version": VERSION, "mean": res.as_dict(), "n_dates": int(len(table)), "n_with_ic": int(len(ic)), "unavailable_by_reason": {k: int(v) for k, v in states.items() if k != "ok"},
            "information_ratio_per_date": None if len(ic) < 5 or ic.std(ddof=1) == 0 else float(ic.mean() / ic.std(ddof=1)), "hit_rate": None if not len(ic) else float((ic > 0).mean()),
            "mean_assets_per_date": float(table["n"].mean()) if len(table) else None,
            "note": "the standard error is Newey-West with a lag of at least horizon - 1 because consecutive labels overlap; the information ratio and hit rate are descriptive"}


def decay_grid(signal: pd.DataFrame, prices: pd.DataFrame, *, base_lag: int = 1, extra_lags=(0, 1, 2, 5, 10), horizons=(1, 5, 10, 20), min_assets: int = 5) -> dict:
    """The IC against ``forward_return(prices, h, base_lag + k)`` for every ``(k, h)``. ``k = 0`` is the tradeable return of the declared timeline; a larger ``k`` is the same signal acted on ``k`` sessions late. Each cell
    carries a HAC interval with the horizon's overlap."""
    cells = []
    for k in extra_lags:
        for h in horizons:
            tgt = forward_return(prices, int(h), int(base_lag) + int(k)).reindex(index=signal.index, columns=signal.columns)
            tab = ic_table(signal, tgt, min_assets)
            res = inf.hac_mean(tab["ic"].dropna(), horizon=int(h), label=f"IC lag+{k} h{h}")
            cells.append({"extra_lag": int(k), "horizon": int(h), "mean_ic": res.value, "interval": None if res.interval is None else list(res.interval), "n": int(tab["ic"].notna().sum()), "eligibility": res.eligibility})
    base = {c["horizon"]: c["mean_ic"] for c in cells if c["extra_lag"] == min(extra_lags)}
    for c in cells:
        b = base.get(c["horizon"])
        c["share_of_undelayed"] = None if b is None or c["mean_ic"] is None or not b > 0 else float(c["mean_ic"] / b)
    return {"version": VERSION, "base_lag": int(base_lag), "cells": cells, "note": "a share of the undelayed IC is undefined (None) when the undelayed IC is missing or not positive"}


def coverage_report(signal: pd.DataFrame, investable: pd.DataFrame | None = None, *, min_asset_coverage: float = 0.8) -> dict:
    """Where the signal exists among the (date, asset) cells that were investable: coverage by asset and by date, the cells that were investable but empty, and the longest run of missing dates per asset."""
    inv = investable.reindex(index=signal.index, columns=signal.columns).fillna(False).astype(bool) if investable is not None else pd.DataFrame(True, index=signal.index, columns=signal.columns)
    have = signal.notna() & np.isfinite(signal.to_numpy(dtype=float, na_value=np.nan)) & inv
    cells = int(inv.to_numpy().sum())
    if cells == 0:
        return {"version": VERSION, "eligibility": "insufficient", "reason": "no investable (date, asset) cell"}
    by_asset = (have.sum() / inv.sum().replace(0, np.nan)).astype(float)
    per_date = have.sum(axis=1)
    missing = inv & ~have
    runs = {}
    for a in signal.columns:
        m = missing[a].to_numpy()
        best = cur = 0
        for v in m:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        runs[str(a)] = int(best)
    thin = sorted(str(a) for a, c in by_asset.items() if np.isfinite(c) and c < min_asset_coverage)
    return {"version": VERSION, "eligibility": "applied", "cells": cells, "coverage": float(have.to_numpy().sum() / cells), "missing_cells": int(missing.to_numpy().sum()),
            "by_asset": {str(a): None if not np.isfinite(c) else float(c) for a, c in by_asset.items()}, "assets_below_threshold": thin, "threshold": float(min_asset_coverage),
            "assets_per_date": {"min": int(per_date.min()), "median": float(per_date.median()), "max": int(per_date.max())}, "longest_missing_run": runs,
            "note": "missing cells are listed; they are not dropped to improve the picture"}


def stability_report(table: pd.DataFrame, blocks: int = 4, signal: pd.DataFrame | None = None, persistence_lags=(1, 5, 21)) -> dict:
    """The IC in ``blocks`` consecutive parts of the dates that have one (equal counts), how often a block has the sign of the whole, and the spread of the block means. With ``signal``, the rank persistence of the
    signal itself (the cross-sectional rank correlation between date t and t - lag, averaged over dates)."""
    ic = table["ic"].dropna()
    if len(ic) < blocks * 10:
        out = {"version": VERSION, "eligibility": "insufficient", "reason": f"{len(ic)} dates with an IC: fewer than {blocks * 10} needed for {blocks} blocks"}
    else:
        parts = np.array_split(np.arange(len(ic)), blocks)
        means = [float(ic.iloc[p].mean()) for p in parts]
        overall = float(ic.mean())
        out = {"version": VERSION, "eligibility": "applied", "blocks": [{"from": str(ic.index[p[0]].date()), "to": str(ic.index[p[-1]].date()), "n": int(len(p)), "mean_ic": m} for p, m in zip(parts, means)],
               "overall": overall, "same_sign_share": float(np.mean([np.sign(m) == np.sign(overall) for m in means])), "spread_of_block_means": float(np.std(means, ddof=1)),
               "note": "blocks are consecutive parts of the sample; they describe variation over time and are not independent tests"}
    if signal is not None:
        pers = {}
        ranks = signal.rank(axis=1)
        for lag in persistence_lags:
            a, b = ranks, ranks.shift(lag)
            both = a.notna() & b.notna()
            ac = a.where(both).sub(a.where(both).mean(axis=1), axis=0)
            bc = b.where(both).sub(b.where(both).mean(axis=1), axis=0)
            rho = (ac * bc).sum(axis=1) / np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1)).replace(0.0, np.nan)
            pers[int(lag)] = None if rho.dropna().empty else float(rho.mean())
        out["rank_persistence"] = pers
    return out


def redundancy_report(X: pd.DataFrame, *, threshold: float = 0.9, every: int = 1, min_assets: int = 5) -> dict:
    """How much of a feature panel (rows (date, asset), columns features) is the same information twice. Each feature is ranked within its date, then the features are correlated across all rows: the average
    cross-sectional rank correlation. Reported: the pairs above ``threshold``, the participation-ratio count of independent features ``(sum of eigenvalues)^2 / sum of squared eigenvalues`` of that matrix, and the
    features that cannot be assessed (constant or empty). Dates with fewer than ``min_assets`` assets are left out and counted. Computed on whatever rows the caller passes: pass development rows only."""
    dates = X.index.get_level_values(0)
    keep = pd.Series(1, index=X.index).groupby(level=0).transform("sum") >= min_assets
    sub = X[keep.to_numpy()]
    if every > 1:
        picked = pd.Index(sub.index.get_level_values(0).unique()[::every])
        sub = sub[sub.index.get_level_values(0).isin(picked)]
    ranked = sub.groupby(level=0).rank(pct=True)
    usable = [c for c in ranked.columns if ranked[c].notna().sum() >= 50 and ranked[c].nunique() > 1]
    unavailable = {str(c): "constant or fewer than 50 ranked rows" for c in ranked.columns if c not in usable}
    if len(usable) < 2:
        return {"version": VERSION, "eligibility": "insufficient", "reason": "fewer than two assessable features", "unavailable": unavailable}
    corr = ranked[usable].corr()
    values = np.linalg.eigvalsh(corr.fillna(0.0).to_numpy())
    values = np.clip(values, 0.0, None)
    eff = float(values.sum() ** 2 / (values ** 2).sum())
    iu = np.triu_indices(len(usable), 1)
    pairs = [(usable[i], usable[j], float(corr.iloc[i, j])) for i, j in zip(*iu) if abs(corr.iloc[i, j]) >= threshold]
    return {"version": VERSION, "eligibility": "applied", "n_features": len(usable), "effective_independent_features": eff, "pairs_above_threshold": sorted(pairs, key=lambda t: -abs(t[2]))[:50], "threshold": float(threshold),
            "unavailable": unavailable, "rows_used": int(len(sub)), "dates_dropped_for_small_cross_section": int(len(set(dates)) - len(set(sub.index.get_level_values(0)))),
            "note": "average cross-sectional rank correlation, pooled over the rows given; the participation ratio counts how many features' worth of information the panel carries"}


def _align(series: dict[str, pd.Series]) -> tuple[pd.DataFrame, dict]:
    frame = pd.concat(series, axis=1)
    both = frame.dropna()
    dropped = {k: int(series[k].dropna().shape[0] - len(both)) for k in series}
    return both, dropped


def correlation_report(predictions: dict[str, pd.DataFrame], returns: dict[str, pd.Series], weights: dict[str, pd.DataFrame], *, min_assets: int = 5) -> dict:
    """Pairwise: the mean cross-sectional rank correlation of two prediction panels, the correlation of the two portfolios' returns on the dates both exist, the overlap of their books, and each book's
    concentration. The number of dates that exist for one side only is reported for every pair. For two books, the Sharpe ratio of an equal-risk blend and its diversification ratio."""
    names = list(returns)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = {"a": a, "b": b}
            if a in predictions and b in predictions:
                tab = ic_table(predictions[a], predictions[b], min_assets)
                row["prediction_rank_correlation"] = None if tab["ic"].dropna().empty else float(tab["ic"].mean())
                row["prediction_dates"] = int(tab["ic"].notna().sum())
            both, dropped = _align({a: returns[a], b: returns[b]})
            row["return_correlation"] = None if len(both) < 20 else float(both[a].corr(both[b]))
            row["aligned_dates"], row["dropped_dates"] = int(len(both)), dropped
            if a in weights and b in weights:
                wa, wb = weights[a].fillna(0.0), weights[b].reindex(index=weights[a].index, columns=weights[a].columns).fillna(0.0)
                same = np.minimum(wa.abs(), wb.abs()).where(np.sign(wa) == np.sign(wb), 0.0).sum(axis=1)
                gross = np.minimum(wa.abs().sum(axis=1), wb.abs().sum(axis=1))
                cos = (wa * wb).sum(axis=1) / np.sqrt((wa ** 2).sum(axis=1) * (wb ** 2).sum(axis=1)).replace(0.0, np.nan)
                row["exposure_overlap"] = None if (gross > 0).sum() == 0 else float((same / gross.replace(0.0, np.nan)).mean())
                row["weight_cosine"] = None if cos.dropna().empty else float(cos.mean())
            if len(both) >= 60:
                va, vb = both[a].std(ddof=1), both[b].std(ddof=1)
                if va > 0 and vb > 0:
                    blend = 0.5 * both[a] / va + 0.5 * both[b] / vb
                    sh = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(252))                  # noqa: E731
                    row["blend"] = {"sharpe_a": sh(both[a]), "sharpe_b": sh(both[b]), "sharpe_blend": sh(blend), "diversification_ratio": float(0.5 * (va + vb) / np.sqrt(0.25 * va ** 2 + 0.25 * vb ** 2 + 0.5 * va * vb * both[a].corr(both[b])))}
            pairs.append(row)
    conc = {}
    for name, w in weights.items():
        g = w.fillna(0.0).abs().sum(axis=1)
        live = g > 0
        hhi = (w.fillna(0.0) ** 2).sum(axis=1)[live] / (g[live] ** 2)
        top3 = w.fillna(0.0).abs().apply(lambda r: r.nlargest(3).sum(), axis=1)[live] / g[live]
        conc[name] = {"mean_hhi": float(hhi.mean()) if len(hhi) else None, "effective_positions": float((1.0 / hhi).mean()) if len(hhi) else None, "top3_share": float(top3.mean()) if len(top3) else None, "days_with_a_book": int(live.sum())}
    return {"version": VERSION, "pairs": pairs, "concentration": conc, "note": "returns are aligned on the dates both exist and the counts dropped from each side are reported; no period is removed because it is unflattering"}


def conditional_ic(new: pd.DataFrame, existing: dict[str, pd.DataFrame], target: pd.DataFrame, *, min_assets: int = 8) -> pd.DataFrame:
    """The rank IC of ``new`` with ``target`` after removing, on each date, the part of both that the ranks of the ``existing`` signals explain (a cross-sectional least-squares regression of each on the existing
    ranks and a constant; the correlation of the two residuals). A date needs at least ``max(min_assets, regressors + 4)`` assets (so that three residual degrees of freedom remain); otherwise its state is ``too_few``. A constant signal or target is ``tied_signal``/``tied_target``; a signal (or target) that the existing signals reproduce exactly leaves no residual and is ``explained_signal``/``explained_target``: a different condition, and in neither is the IC zero, it is unavailable. The result is the same table shape as ``ic_table``."""
    names = list(existing)
    k = len(names)
    out = []
    for date in new.index:
        cols = [new.loc[date], target.loc[date]] + [existing[n].loc[date] if date in existing[n].index else pd.Series(np.nan, index=new.columns) for n in names]
        frame = pd.concat(cols, axis=1).dropna()
        n = len(frame)
        if n < max(min_assets, k + 4):
            out.append((date, np.nan, n, "too_few"))
            continue
        r = frame.rank()
        if r.iloc[:, 0].nunique() <= 1:                                                              # a constant cross-section: no ranks to explain
            out.append((date, np.nan, n, "tied_signal"))
            continue
        if r.iloc[:, 1].nunique() <= 1:
            out.append((date, np.nan, n, "tied_target"))
            continue
        design = np.column_stack([np.ones(n)] + [r.iloc[:, 2 + j].to_numpy() for j in range(k)])
        res = []
        for col in (0, 1):
            y = r.iloc[:, col].to_numpy()
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            res.append(y - design @ beta)
        if res[0].std() < 1e-9 * max(r.iloc[:, 0].std(), 1.0):                                       # the existing signals reproduce it exactly: nothing is left to correlate
            out.append((date, np.nan, n, "explained_signal"))
        elif res[1].std() < 1e-9 * max(r.iloc[:, 1].std(), 1.0):
            out.append((date, np.nan, n, "explained_target"))
        else:
            out.append((date, float(np.corrcoef(res[0], res[1])[0, 1]), n, "ok"))
    t = pd.DataFrame(out, columns=["date", "ic", "n", "state"]).set_index("date")
    return t


def blend_value(base: pd.Series, new: pd.Series, *, n_boot: int = 500, seed: int = 0, label: str = "") -> dict:
    """The paired Sharpe difference between an equal-risk blend of the base book and the new book and the base book alone, on the dates both exist (joint block resampling). The blend divides each book by its own
    volatility, so a book is not rewarded for merely being larger. Predictive incremental value on this sample; not a causal claim and not a feature importance."""
    both = pd.concat([base, new], axis=1, keys=["base", "new"]).dropna()
    dropped = {"base": int(base.dropna().shape[0] - len(both)), "new": int(new.dropna().shape[0] - len(both))}
    if len(both) < inf.MIN_OBS or both["base"].std(ddof=1) == 0 or both["new"].std(ddof=1) == 0:
        return {"version": VERSION, "eligibility": "insufficient", "reason": f"{len(both)} common dates, or a book that never varies", "aligned": int(len(both)), "dropped": dropped}
    vb, vn = both["base"].std(ddof=1), both["new"].std(ddof=1)
    blend = (0.5 * both["base"] / vb + 0.5 * both["new"] / vn) * vb                              # scaled back to the base book's volatility
    res = inf.paired_difference(blend, both["base"], statistic="sharpe", n_boot=n_boot, seed=seed, label=label or "blend vs base")
    return {"version": VERSION, "eligibility": res.eligibility, "difference": res.as_dict(), "aligned": int(len(both)), "dropped": dropped, "correlation": float(both["base"].corr(both["new"])),
            "interpretation": "predictive incremental value of the new book given the base book on these dates; it is not a causal claim and not a feature importance"}
