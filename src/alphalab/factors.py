"""Factor evaluation, comparison and combination: the validation and combination stages of the Alpha Lab as one tested library.

Everything is computed from a signal panel (dates x assets, known at the close of each date) and the forward returns it is meant to predict (dates x assets, the return AFTER the date). Observed data only; nothing
is filled with synthetic values, and a cross-section too thin to rank is refused, not scored.

* :func:`evaluate_factor`: rank IC and Pearson IC with a Newey-West (HAC) interval, quantile spreads and the long-short return, turnover, coverage by date and by asset, extreme-value share, stability across sub-periods,
  decay, and the IC left after neutralising controls (size, beta, momentum, sector dummies you supply).
* :func:`compare_factors`: the same for several factors, with the p-values corrected for having looked at several (Holm and Benjamini-Hochberg).
* :func:`combine_factors`: equal weight, IC weighted, ridge and orthogonalised combinations, each trained on the PAST only (the weight at a date uses ICs whose forward windows had ended), judged on a period none of
  them was fitted on, against the best single factor and against equal weight: a combination that does not beat those is not an improvement.

None of this says a factor makes money. It says how well it ranks assets, how reliably, how fast the information fades, and what it costs in turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..models.regression import MIN_ASSETS_RANK_IC, cross_sectional_ic
from . import ic as icm


class FactorError(ValueError):
    """A factor evaluation that cannot be done honestly (too few assets or too little history)."""


def _breadth(signal: pd.DataFrame, forward: pd.DataFrame, min_assets: int) -> int:
    if min_assets < MIN_ASSETS_RANK_IC:
        raise FactorError(f"a rank IC needs at least {MIN_ASSETS_RANK_IC} assets on a date; min_assets={min_assets} is not allowed")
    both = signal.notna() & forward.reindex_like(signal).notna()
    n = int(both.sum(axis=1).median()) if len(both) else 0
    if n < min_assets:
        raise FactorError(f"the cross-section is too thin to rank: a median of {n} assets have both a signal and an outcome on a date, and at least {min_assets} are required")
    return n


def _hac(ic: pd.Series, horizon: int, level: float = 0.95) -> dict:
    x = ic.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 20:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan"), "low": float("nan"), "high": float("nan")}
    mean = float(x.mean())
    se = icm._newey_west_se(x, max(int(horizon) - 1, 0))                                  # overlapping forward windows make neighbouring ICs dependent: horizon - 1 lags
    z = stats.norm.ppf(0.5 + level / 2)
    t = mean / se if se > 0 else float("nan")
    return {"n": n, "mean": mean, "se": se, "t": t, "p": float(2 * (1 - stats.norm.cdf(abs(t)))) if np.isfinite(t) else float("nan"), "low": mean - z * se, "high": mean + z * se}


def _neutralise(signal: pd.DataFrame, controls: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The signal with its cross-sectional linear dependence on the controls removed, date by date (OLS residual). A date with fewer assets than regressors + 2 is left missing."""
    out = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)
    names = list(controls)
    for t in signal.index:
        y = signal.loc[t]
        X = pd.concat([controls[c].loc[t].rename(c) for c in names if t in controls[c].index], axis=1) if names else pd.DataFrame(index=signal.columns)
        ok = y.notna() & X.notna().all(axis=1)
        if ok.sum() < len(names) + 3:
            continue
        A = np.column_stack([np.ones(int(ok.sum())), X[ok].to_numpy(dtype=float)])
        yv = y[ok].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        resid = yv - A @ beta
        if yv.std() > 0 and resid.std() < 1e-9 * yv.std():                                 # the signal is a linear combination of the controls: nothing is left, and rounding noise is not a signal
            continue
        out.loc[t, ok[ok].index] = resid
    return out


def evaluate_factor(signal: pd.DataFrame, forward: pd.DataFrame, horizon: int = 1, returns: pd.DataFrame | None = None, controls: dict[str, pd.DataFrame] | None = None, min_assets: int = 5,
                    buckets: int = 5, decay_lags=(1, 2, 3, 5, 10, 21)) -> dict:
    """Evaluate one factor. ``forward`` is the return that follows each date over ``horizon`` periods; ``returns`` (one-period returns) enables the decay curve; ``controls`` are panels (dates x assets) to neutralise
    against. Raises :class:`FactorError` on a cross-section too thin to rank."""
    signal, forward = signal.align(forward, join="inner")
    n_med = _breadth(signal, forward, min_assets)
    rank_ic = cross_sectional_ic(signal, forward, "spearman", min_assets)
    pearson = cross_sectional_ic(signal, forward, "pearson", min_assets)
    if len(rank_ic) < 30:
        raise FactorError(f"only {len(rank_ic)} dates have enough assets: at least 30 are required to say anything about a factor")
    out = {"n_assets_median": n_med, "n_dates": int(len(rank_ic)), "horizon": int(horizon), "rank_ic": _hac(rank_ic, horizon), "pearson_ic": _hac(pearson, horizon)}
    # quantile spreads and the long-short return (top bucket minus bottom bucket, per date)
    pct = signal.rank(axis=1, pct=True)
    q = max(2, min(buckets, n_med // 2))
    bucket = np.ceil(pct * q).clip(1, q).where(signal.notna() & forward.notna())
    per = {b: forward.where(bucket == b).mean(axis=1) for b in range(1, q + 1)}
    ls = (per[q] - per[1]).dropna()
    out["quantiles"] = {"buckets": q, "mean_return": {str(b): float(per[b].mean()) for b in per}, "long_short_mean": float(ls.mean()) if len(ls) else float("nan"),
                        "long_short_t": float(ls.mean() / (icm._newey_west_se(ls.to_numpy(), max(int(horizon) - 1, 0)) or np.nan)) if len(ls) > 20 else float("nan"),
                        "monotonicity": float(stats.spearmanr(np.arange(1, q + 1), [per[b].mean() for b in per]).statistic) if q >= 3 else float("nan")}
    # turnover of the ranking, coverage, extremes
    ranks = signal.rank(axis=1, pct=True)
    out["turnover"] = float((ranks.diff().abs().mean(axis=1)).mean())                    # mean absolute change in percentile rank per period: 0 = frozen, ~0.33 = reshuffled each period
    cov_date = signal.notna().mean(axis=1)
    cov_asset = signal.notna().mean(axis=0)
    z = (signal.sub(signal.mean(axis=1), axis=0)).div(signal.std(axis=1).replace(0, np.nan), axis=0)
    out["coverage"] = {"by_date_min": float(cov_date.min()), "by_date_mean": float(cov_date.mean()), "by_asset_min": float(cov_asset.min()), "assets_below_half": [str(c) for c in cov_asset.index[cov_asset < 0.5]],
                       "extreme_share": float((z.abs() > 4).sum().sum() / max(int(z.notna().sum().sum()), 1))}
    # stability: sign and size of the IC in thirds of the sample
    thirds = [part for part in np.array_split(rank_ic.to_numpy(), 3)]
    out["stability"] = {"thirds_mean": [float(p.mean()) for p in thirds], "same_sign_share": float(np.mean([np.sign(p.mean()) == np.sign(rank_ic.mean()) for p in thirds]))}
    if returns is not None:
        lags = [k for k in decay_lags if k < len(returns) // 4]
        curve = icm.lagged_ic(signal, returns, lags=lags, min_assets=min_assets) if lags else pd.DataFrame()
        out["decay"] = {} if curve.empty else {str(int(k)): float(v) for k, v in curve["ic"].items()} if "ic" in curve else {}
    if controls:
        neutral = _neutralise(signal, controls)
        nic = cross_sectional_ic(neutral, forward, "spearman", min_assets)
        out["neutralised"] = {"controls": list(controls), "rank_ic": _hac(nic, horizon), "share_of_ic_kept": float(nic.mean() / rank_ic.mean()) if len(nic) and rank_ic.mean() else float("nan"), "nothing_left": bool(len(nic) < 20),
                              "dates_with_residual": int(len(nic))}
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    items = sorted(((k, p) for k, p in pvalues.items() if np.isfinite(p)), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return {**{k: float("nan") for k in pvalues}, **out}


def benjamini_hochberg(pvalues: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg adjusted p-values (false-discovery-rate control)."""
    items = sorted(((k, p) for k, p in pvalues.items() if np.isfinite(p)), key=lambda kv: kv[1])
    m, adj, prev = len(items), {}, 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        prev = min(prev, p * m / (i + 1))
        adj[k] = prev
    return {**{k: float("nan") for k in pvalues}, **adj}


def compare_factors(signals: dict[str, pd.DataFrame], forward: pd.DataFrame, horizon: int = 1, alpha: float = 0.05, **kw) -> dict:
    """Evaluate several factors and correct the p-values for the number looked at. ``survives`` is Holm-adjusted p below ``alpha``: a factor that clears only its raw p-value is not called predictive."""
    evals = {k: evaluate_factor(s, forward, horizon, **kw) for k, s in signals.items()}
    raw = {k: e["rank_ic"]["p"] for k, e in evals.items()}
    h, bh = holm(raw), benjamini_hochberg(raw)
    rows = [{"factor": k, "rank_ic": e["rank_ic"]["mean"], "ci_low": e["rank_ic"]["low"], "ci_high": e["rank_ic"]["high"], "t": e["rank_ic"]["t"], "p_raw": raw[k], "p_holm": h[k], "p_bh": bh[k],
             "survives_holm": bool(np.isfinite(h[k]) and h[k] < alpha), "survives_bh": bool(np.isfinite(bh[k]) and bh[k] < alpha), "long_short": e["quantiles"]["long_short_mean"], "turnover": e["turnover"],
             "stable": e["stability"]["same_sign_share"], "thirds": e["stability"]["thirds_mean"], "n_assets_median": e["n_assets_median"], "pearson_ic": e["pearson_ic"]["mean"],
             "ic_neutralised": (e.get("neutralised") or {}).get("rank_ic", {}).get("mean"), "share_kept": (e.get("neutralised") or {}).get("share_of_ic_kept"),
             "nothing_left": (e.get("neutralised") or {}).get("nothing_left")} for k, e in evals.items()]
    rows.sort(key=lambda r: -abs(r["rank_ic"]) if np.isfinite(r["rank_ic"]) else 0)
    return {"table": rows, "n_tested": len(rows), "alpha": alpha, "details": evals,
            "note": "p-values are corrected for the number of factors compared (Holm controls the family-wise error; Benjamini-Hochberg the false-discovery rate). A factor chosen after looking at many needs the corrected value."}


def _zs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1).replace(0, np.nan), axis=0)


def combine_factors(signals: dict[str, pd.DataFrame], forward: pd.DataFrame, horizon: int = 1, test_fraction: float = 0.4, ridge: float = 10.0, min_assets: int = 5, min_train: int = 120) -> dict:
    """Combine factors four ways using only the past, and judge the result where nothing was fitted.

    The weight at date ``t`` uses information whose forward window had ended by ``t`` (IC weights: the mean IC of dates ``<= t - horizon``; ridge: a regression on dates ``<= t - horizon``). The evaluation
    period is the last ``test_fraction`` of the dates; the earlier ones only train. A combination is reported as an improvement only if its out-of-sample IC beats BOTH the best single factor (chosen on the
    training period, not the test) and equal weight, by more than the standard error of the difference."""
    names = list(signals)
    if len(names) < 2:
        raise FactorError("combining needs at least two factors")
    z = {k: _zs(s) for k, s in signals.items()}
    idx = z[names[0]].index
    for k in names[1:]:
        idx = idx.intersection(z[k].index)
    idx = idx.intersection(forward.index)
    cols = z[names[0]].columns
    z = {k: v.reindex(index=idx, columns=cols) for k, v in z.items()}
    fwd = forward.reindex(index=idx, columns=cols)
    _breadth(z[names[0]], fwd, min_assets)
    n = len(idx)
    split = int(n * (1 - test_fraction))
    if split < min_train or n - split < 60:
        raise FactorError(f"not enough history: {n} dates, {split} to train and {n - split} to test; at least {min_train} and 60 are required")
    ics = {k: cross_sectional_ic(z[k], fwd, "spearman", min_assets).reindex(idx) for k in names}
    ic_frame = pd.DataFrame(ics)
    h = max(int(horizon), 1)

    def known(frame: pd.DataFrame) -> pd.DataFrame:                                      # what was known at date t: ICs of dates <= t - h
        return frame.shift(h)

    ew = sum(z[k] for k in names) / len(names)
    rolling = known(ic_frame).expanding(min_periods=min_train // 2).mean()               # IC weights at t from the past only
    icw = sum(z[k].mul(rolling[k].clip(lower=0).reindex(idx), axis=0) for k in names)    # a factor whose past IC is negative gets no weight (it is not flipped on the evidence of the past alone)
    # ridge: weights from a pooled regression of the forward return z-score on the factor z-scores over dates up to t - h, refreshed monthly
    ridge_sig = pd.DataFrame(np.nan, index=idx, columns=cols)
    ort_sig = pd.DataFrame(np.nan, index=idx, columns=cols)
    y_all = _zs(fwd)
    past_ic = known(ic_frame).expanding(min_periods=min_train // 2).mean()
    last_w = None
    start = max(min_train // 2, 1)
    for i in range(start, n):
        if (i - start) % 21 == 0 or last_w is None:
            hi = i - h + 1
            if hi > min_train // 2:
                rows = idx[max(0, hi - 756):hi]
                X = np.column_stack([z[k].loc[rows].to_numpy().ravel() for k in names])
                y = y_all.loc[rows].to_numpy().ravel()
                ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
                if ok.sum() > 50 * len(names):
                    Xo, yo = X[ok], y[ok]
                    last_w = np.linalg.solve(Xo.T @ Xo + ridge * len(yo) / 1000.0 * np.eye(len(names)), Xo.T @ yo)
        if last_w is not None:
            ridge_sig.iloc[i] = sum(z[k].iloc[i].fillna(0.0) * last_w[m] for m, k in enumerate(names))
        # orthogonalised: Gram-Schmidt across the factors on this date's cross-section, strongest past IC first; each factor contributes only what the stronger ones do not already say, signed by its past IC
        order = [m for m in np.argsort(-np.nan_to_num(past_ic.iloc[i].abs().to_numpy()))]
        basis, parts = [], np.zeros(len(cols))
        for m in order:
            v = z[names[m]].iloc[i].fillna(0.0).to_numpy().copy()
            for q_ in basis:
                den = float(q_ @ q_)
                if den > 0:
                    v = v - float(v @ q_) / den * q_
            basis.append(v)
            sgn = np.sign(past_ic.iloc[i].iloc[m]) if np.isfinite(past_ic.iloc[i].iloc[m]) else 0.0
            sd = v.std()
            parts += sgn * (v / sd if sd > 0 else v)
        ort_sig.iloc[i] = parts
    candidates = {"equal_weight": ew, "ic_weighted": icw, "ridge": ridge_sig, "orthogonalised": ort_sig}
    test = idx[split:]
    train = idx[:split]
    single_train = {k: float(ic_frame[k].loc[train].mean()) for k in names}
    best = max(single_train, key=lambda k: abs(single_train[k]))                          # chosen on the training period only
    results = {}
    for k, s in {**{f"single:{n_}": z[n_] for n_ in names}, **candidates}.items():
        ic_t = cross_sectional_ic(s.loc[test], fwd.loc[test], "spearman", min_assets).dropna()
        results[k] = {**_hac(ic_t, horizon), "n_test_dates": int(len(ic_t))}
    best_ic = cross_sectional_ic(z[best].loc[test], fwd.loc[test], "spearman", min_assets).dropna()
    ew_ic = cross_sectional_ic(ew.loc[test], fwd.loc[test], "spearman", min_assets).dropna()
    verdicts = {}
    for k in candidates:
        ic_k = cross_sectional_ic(candidates[k].loc[test], fwd.loc[test], "spearman", min_assets).dropna()
        row = {}
        for label, ref in (("best_single", best_ic), ("equal_weight", ew_ic)):
            if k == "equal_weight" and label == "equal_weight":
                continue
            d = (ic_k.reindex(ref.index) - ref).dropna()
            se = icm._newey_west_se(d.to_numpy(), max(h - 1, 0)) if len(d) > 20 else float("nan")
            row[label] = {"difference": float(d.mean()) if len(d) else float("nan"), "se": se, "better": bool(len(d) > 20 and np.isfinite(se) and d.mean() > 1.96 * se)}
        verdicts[k] = {**row, "improves_on_baselines": bool(all(v["better"] for v in row.values()))}
    return {"factors": names, "train_dates": [str(train[0].date()), str(train[-1].date())], "test_dates": [str(test[0].date()), str(test[-1].date())], "best_single_on_training": best,
            "training_ic": single_train, "out_of_sample": results, "verdicts": verdicts,
            "note": "Weights use only the past; the test period was never used to choose or fit anything. 'improves_on_baselines' needs the combination to beat both the best single factor (chosen on the training period) and equal weight by more than 1.96 standard errors of the difference."}


def discovery(signals: dict[str, pd.DataFrame], min_assets: int = 5, cluster_threshold: float = 0.7) -> dict:
    """What a person should know about a set of candidate signals before testing any of them: for each, coverage by date and by asset (and the months with the most missing values), the share of extreme values,
    whether the cross-sectional distribution drifts between the first and second half of the history, how persistent the ranking is (so how much it will turn over), and how redundant the signals are with each other
    (the average daily rank correlation, and greedy clusters of signals with |correlation| above ``cluster_threshold``). Observed data only; a signal with a cross-section too thin to rank is reported as such, not scored."""
    names = list(signals)
    if not names:
        raise FactorError("choose at least one signal")
    out = {"signals": {}, "min_assets": min_assets, "rank_ic_min_assets": MIN_ASSETS_RANK_IC}
    zs = {k: _zs(s) for k, s in signals.items()}
    for k, s in signals.items():
        n_med = int(s.notna().sum(axis=1).median()) if len(s) else 0
        cov_date, cov_asset = s.notna().mean(axis=1), s.notna().mean(axis=0)
        miss_month = (1 - cov_date).groupby(s.index.to_period("M")).mean().sort_values(ascending=False)
        z = zs[k]
        half = len(z) // 2
        first, second = s.iloc[:half].to_numpy(float).ravel(), s.iloc[half:].to_numpy(float).ravel()   # raw values: a cross-sectional z-score would hide a change of level or scale
        first, second = first[np.isfinite(first)], second[np.isfinite(second)]
        ks = stats.ks_2samp(first, second) if len(first) > 50 and len(second) > 50 else None
        def auto(lag: int) -> float:                                                         # the average daily rank correlation of the signal with itself ``lag`` days later
            c = cross_sectional_ic(s, s.shift(-lag), "spearman", max(min_assets, MIN_ASSETS_RANK_IC)) if n_med >= max(min_assets, MIN_ASSETS_RANK_IC) else pd.Series(dtype=float)
            return float(c.mean()) if len(c) else float("nan")
        out["signals"][k] = {"n_assets_median": n_med, "rankable": n_med >= max(min_assets, MIN_ASSETS_RANK_IC),
                             "reason": None if n_med >= max(min_assets, MIN_ASSETS_RANK_IC) else f"a median of {n_med} assets have a value on a date; at least {max(min_assets, MIN_ASSETS_RANK_IC)} are needed to rank",
                             "coverage": {"by_date_mean": float(cov_date.mean()), "by_date_min": float(cov_date.min()), "by_asset": {str(c): float(v) for c, v in cov_asset.items()},
                                          "worst_months": [{"month": str(m), "missing": float(v)} for m, v in miss_month.head(3).items() if v > 0]},
                             "extreme_share": float((z.abs() > 4).sum().sum() / max(int(z.notna().sum().sum()), 1)),
                             "drift": None if ks is None else {"ks": float(ks.statistic), "p": float(ks.pvalue), "mean_first": float(first.mean()), "mean_second": float(second.mean()), "std_first": float(first.std()), "std_second": float(second.std())},
                             "rank_persistence": {"lag1": auto(1), "lag21": auto(21)}}
    corr = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            if a == b:
                corr.loc[a, b] = 1.0
            elif np.isnan(corr.loc[a, b]):
                c = cross_sectional_ic(zs[a], zs[b], "spearman", max(min_assets, MIN_ASSETS_RANK_IC)) if out["signals"][a]["rankable"] and out["signals"][b]["rankable"] else pd.Series(dtype=float)
                corr.loc[a, b] = corr.loc[b, a] = float(c.mean()) if len(c) else float("nan")
    clusters, used = [], set()
    for a in names:
        if a in used:
            continue
        grp = [a] + [b for b in names if b != a and b not in used and np.isfinite(corr.loc[a, b]) and abs(corr.loc[a, b]) >= cluster_threshold]
        used.update(grp)
        clusters.append(grp)
    out["correlation"] = {"names": names, "matrix": [[None if not np.isfinite(v) else float(v) for v in row] for row in corr.to_numpy()]}
    out["clusters"] = clusters
    out["redundant"] = [grp for grp in clusters if len(grp) > 1]
    return out
