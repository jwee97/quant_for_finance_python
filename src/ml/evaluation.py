"""Judging a model: how well it ranks, how fast the edge fades, whether the strategy survives its segments, and whether the result looks like an overfit.

Rank information coefficient (IC). The cross-sectional Spearman correlation between the scores of a date and the realised forward returns of the same date; its mean over dates is the model's skill at
choosing among assets, its mean over its standard deviation (the IC information ratio) says how steadily. Forward returns overlap by the target horizon, so the t-statistic is computed with
``n / horizon`` independent observations. IC decay is the same IC against longer and longer horizons: an edge that is gone after two days cannot be traded weekly.

The overfitting report compares the Sharpe ratio of the strategy in each segment of the timeline (train: in-sample; validation: where choices were made; test and forward: reserved) and turns the gaps into
a generalisation score and plain-language warnings. It is a set of diagnostics, not a proof: a strategy can pass every one of them and still be a fluke, which is why the number of trials tried is part
of the report.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ..backtest.metrics import deflated_sharpe_ratio, performance_summary, probabilistic_sharpe_ratio
from .targets import forward_return

ANN = 252.0


# ------------------------------------------------------------------------------------------------------------------ information coefficient
def rank_ic_wide(score: pd.DataFrame, target: pd.DataFrame, min_assets: int = 3, method: str = "spearman") -> pd.Series:
    """The cross-sectional correlation between ``score`` and ``target`` on each date (rows of two wide frames), over the assets with both. Dates with fewer than ``min_assets`` are missing."""
    both = score.notna() & target.notna()
    s, t = score.where(both), target.where(both)
    if method == "spearman":
        s, t = s.rank(axis=1), t.rank(axis=1)
    sc, tc = s.sub(s.mean(axis=1), axis=0), t.sub(t.mean(axis=1), axis=0)
    cov = (sc * tc).sum(axis=1)
    den = np.sqrt((sc ** 2).sum(axis=1) * (tc ** 2).sum(axis=1))
    ic = cov / den.replace(0.0, np.nan)
    return ic.where(both.sum(axis=1) >= min_assets)


def ic_summary(ic: pd.Series, horizon: int = 1) -> dict:
    """Mean IC, its standard deviation, the IC information ratio, the share of positive dates, and a t-statistic that counts one independent observation per ``horizon`` days."""
    x = ic.dropna()
    n = len(x)
    if n < 5:
        return {"n": int(n), "mean": float("nan"), "std": float("nan"), "ir": float("nan"), "t": float("nan"), "hit_rate": float("nan"), "p_value": float("nan")}
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    t = mean / sd * math.sqrt(n / max(horizon, 1)) if sd > 0 else float("nan")
    return {"n": int(n), "mean": mean, "std": sd, "ir": mean / sd if sd > 0 else float("nan"), "t": t, "hit_rate": float((x > 0).mean()), "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(t)))) if np.isfinite(t) else float("nan")}


def ic_decay(score: pd.DataFrame, prices: pd.DataFrame, lag: int = 1, horizons: tuple = (1, 2, 3, 5, 10, 21, 42, 63)) -> list[dict]:
    """The mean rank IC of ``score`` against the forward return over each horizon (measured from the same ``lag`` after the decision as the targets)."""
    out = []
    for h in horizons:
        ic = rank_ic_wide(score, forward_return(prices, h, lag).reindex_like(score))
        s = ic_summary(ic, h)
        out.append({"horizon": int(h), "mean_ic": s["mean"], "t": s["t"], "n": s["n"]})
    return out


def quantile_spread(score: pd.DataFrame, fwd: pd.DataFrame, buckets: int = 5) -> dict:
    """The average forward return of the assets in each score bucket (1 = lowest scores), the top-minus-bottom spread, and how monotonic the buckets are (Spearman correlation of bucket and return)."""
    n_assets = int(score.notna().sum(axis=1).median()) if len(score) else 0
    q = max(2, min(buckets, n_assets // 2)) if n_assets >= 4 else 2
    pct = score.rank(axis=1, pct=True)
    bucket = np.ceil(pct * q).clip(1, q).where(score.notna() & fwd.reindex_like(score).notna())
    rows = []
    for b in range(1, q + 1):
        sel = (bucket == b)
        vals = fwd.reindex_like(score).where(sel).to_numpy()
        rows.append({"bucket": b, "mean_return": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"), "n": int(np.isfinite(vals).sum())})
    means = np.array([r["mean_return"] for r in rows])
    ok = np.isfinite(means)
    mono = float(stats.spearmanr(np.arange(q)[ok], means[ok]).statistic) if ok.sum() >= 3 else float("nan")
    return {"buckets": rows, "spread": float(means[-1] - means[0]) if ok[0] and ok[-1] else float("nan"), "monotonicity": mono}


# ------------------------------------------------------------------------------------------------------------------ prediction metrics
def regression_metrics(pred: pd.Series, y: pd.Series) -> dict:
    """Error and fit of the forecasts against the realised target on the rows that have both: mean squared and absolute error, the correlation, the out-of-sample R^2 against a zero forecast
    (Gu, Kelly and Xiu 2020: positive only if the model beats saying nothing), and how often the sign is right."""
    d = pd.concat([pred.rename("p"), y.rename("y")], axis=1).dropna()
    if len(d) < 30:
        return {"n": int(len(d))}
    e = d["y"] - d["p"]
    return {"n": int(len(d)), "mse": float((e ** 2).mean()), "mae": float(e.abs().mean()), "corr": float(d["p"].corr(d["y"])), "r2_oos_zero": float(1.0 - (e ** 2).sum() / (d["y"] ** 2).sum()),
            "sign_accuracy": float((np.sign(d["p"]) == np.sign(d["y"])).mean())}


def classification_metrics(prob: pd.Series, y: pd.Series) -> dict:
    """Accuracy at 0.5, the area under the ROC curve (0.5 is a coin flip), the Brier score against the base rate's, precision and recall of the positive class."""
    from sklearn.metrics import roc_auc_score

    d = pd.concat([prob.rename("p"), y.rename("y")], axis=1).dropna()
    if len(d) < 30 or d["y"].nunique() < 2:
        return {"n": int(len(d))}
    pred = (d["p"] > 0.5).astype(int)
    base = float(d["y"].mean())
    tp = float(((pred == 1) & (d["y"] == 1)).sum())
    return {"n": int(len(d)), "accuracy": float((pred == d["y"]).mean()), "auc": float(roc_auc_score(d["y"], d["p"])), "brier": float(((d["p"] - d["y"]) ** 2).mean()),
            "brier_base_rate": float(((base - d["y"]) ** 2).mean()), "base_rate": base, "precision": tp / max(float((pred == 1).sum()), 1.0), "recall": tp / max(float((d["y"] == 1).sum()), 1.0)}


# ------------------------------------------------------------------------------------------------------------------ strategy by segment
def sharpe_of(r: pd.Series) -> float:
    x = r.dropna()
    sd = x.std(ddof=1) if len(x) > 2 else float("nan")
    return float(x.mean() / sd * math.sqrt(ANN)) if np.isfinite(sd) and sd > 1e-12 else float("nan")                 # a constant series has no Sharpe ratio


def segment_stats(net: pd.Series, labels: pd.Series, insample_net: pd.Series | None = None) -> dict:
    """Headline numbers of the strategy within each segment: days, annualised return, volatility, Sharpe, maximum drawdown and hit rate. ``insample_net`` supplies the train segment (the strategy built from the
    first model's in-sample predictions), because the out-of-sample run starts at the validation segment."""
    out = {}
    for seg in ("train", "validation", "test", "forward"):
        r = insample_net if (seg == "train" and insample_net is not None) else net[labels.reindex(net.index) == seg]
        r = r.dropna()
        if len(r) < 5:
            continue
        eq = (1.0 + r).cumprod()
        out[seg] = {"n_days": int(len(r)), "ann_return": float(r.mean() * ANN), "ann_vol": float(r.std(ddof=1) * math.sqrt(ANN)), "sharpe": sharpe_of(r), "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
                    "hit_rate": float((r > 0).mean()), "total_return": float(eq.iloc[-1] - 1.0)}
    return out


def overfit_report(segments: dict, n_trials: int = 1, ic_by_fold: list | None = None, gross_sharpe: float | None = None, net_sharpe: float | None = None, n_features: int | None = None,
                   n_rows: int | None = None, trial_sharpes: list | None = None, test_forward_returns: pd.Series | None = None) -> dict:
    """The Overfitting Center: Sharpe by segment, a generalisation score in [0, 1] and warnings.

    ``generalization_score`` is the worst out-of-sample Sharpe (validation, test, forward) over the larger of the train and validation Sharpe (floored at 0.25), clipped to [0, 1], and zero if any
    out-of-sample Sharpe is not positive: how much of the best performance seen survives where it was not fitted. ``psr_test_forward`` is the probability that the true Sharpe over the test and forward
    segments (which no choice has seen) is positive; ``deflated_sharpe_probability`` deflates the validation Sharpe by the number of trials.
    """
    sr = {k: v["sharpe"] for k, v in segments.items() if np.isfinite(v.get("sharpe", np.nan))}
    oos = [sr[s] for s in ("validation", "test", "forward") if s in sr]
    best_seen = max([sr.get(s, -np.inf) for s in ("train", "validation")] + [0.25])
    score = float(np.clip(min(oos) / best_seen, 0.0, 1.0)) if oos and min(oos) > 0 else 0.0
    warns: list[str] = []
    n = lambda seg: segments.get(seg, {}).get("n_days", 0)                                                   # noqa: E731
    if "train" in sr and "validation" in sr and sr["train"] > 1.0 and sr["validation"] < 0.5 * sr["train"]:
        warns.append(f"In-sample Sharpe {sr['train']:.2f} falls to {sr['validation']:.2f} out of sample: the model fits its training data better than it predicts.")
    if "validation" in sr and "test" in sr and sr["validation"] > 0.3 and sr["test"] < 0.0:
        warns.append(f"Positive in validation ({sr['validation']:.2f}) but negative in the test segment ({sr['test']:.2f}): the settings were probably chosen on noise.")
    if "test" in sr and "forward" in sr and sr["test"] > 0.3 and sr["forward"] < 0.0:
        warns.append(f"The forward segment ({sr['forward']:.2f}) disagrees with the test segment ({sr['test']:.2f}): the edge may have been a fluke or may have decayed.")
    if "validation" in sr and "test" in sr and sr["validation"] > 0 and sr["test"] < 0.5 * sr["validation"]:
        warns.append(f"The test Sharpe ({sr['test']:.2f}) is less than half the validation Sharpe ({sr['validation']:.2f}).")
    for seg in ("validation", "test", "forward"):
        if seg in segments and n(seg) < 120:
            warns.append(f"The {seg} segment has only {n(seg)} days: a Sharpe ratio estimated on so little has a standard error of about {1.0 / math.sqrt(max(n(seg), 1) / ANN):.1f}.")
    dsr = float("nan")
    if "validation" in sr and n("validation") > 30:
        dsr = deflated_sharpe_ratio(sr["validation"], max(int(n_trials), 1), n("validation"))
        if n_trials > 5 and np.isfinite(dsr) and dsr < 0.95:
            warns.append(f"{n_trials} configurations have been tried on this data; the probability that the validation Sharpe beats what selection alone would produce is {dsr:.0%}.")
    if ic_by_fold:
        neg = float(np.mean([x < 0 for x in ic_by_fold if np.isfinite(x)])) if any(np.isfinite(x) for x in ic_by_fold) else float("nan")
        if np.isfinite(neg) and neg > 0.4:
            warns.append(f"The information coefficient is negative in {neg:.0%} of the refits: the edge is unstable.")
    if gross_sharpe is not None and net_sharpe is not None and np.isfinite(gross_sharpe) and np.isfinite(net_sharpe) and gross_sharpe - net_sharpe > 0.5:
        warns.append(f"Costs take {gross_sharpe - net_sharpe:.2f} of Sharpe (gross {gross_sharpe:.2f}, net {net_sharpe:.2f}): the edge is thin against the turnover.")
    if n_features and n_rows and n_features > 0.1 * math.sqrt(n_rows):
        warns.append(f"{n_features} features on {n_rows:,} rows: with that many inputs a flexible model can fit noise; prefer regularised or simpler models.")
    psr = float("nan")
    if test_forward_returns is not None and len(test_forward_returns.dropna()) > 30:
        r = test_forward_returns.dropna()
        psr = probabilistic_sharpe_ratio(sharpe_of(r), 0.0, len(r), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)))
        if np.isfinite(psr) and psr < 0.9:
            warns.append(f"On the segments no choice has seen (test and forward) the probability that the true Sharpe is positive is only {psr:.0%}.")
    if not warns and oos:
        warns.append("No overfitting warning fired. That is not evidence the edge is real: check the robustness tests and keep the trial count honest.")
    return {"sharpe": sr, "generalization_score": score, "psr_test_forward": psr, "deflated_sharpe_probability": dsr, "n_trials": int(n_trials), "warnings": warns,
            "gaps": {"train_to_validation": sr.get("train", np.nan) - sr.get("validation", np.nan), "validation_to_test": sr.get("validation", np.nan) - sr.get("test", np.nan),
                     "test_to_forward": sr.get("test", np.nan) - sr.get("forward", np.nan)}}


def strategy_summary(net: pd.Series, gross: pd.Series | None = None, turnover: pd.Series | None = None, costs: pd.Series | None = None, benchmark: pd.Series | None = None) -> dict:
    """The platform's standard metric block for a return stream (see ``backtest.metrics.performance_summary``), trimmed to the numbers the dashboard shows."""
    s = performance_summary(net.dropna(), turnover=turnover, costs=costs, benchmark=benchmark)
    keep = ("start", "end", "n_obs", "cagr", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown", "hit_rate", "skew", "var_95", "cvar_95", "total_return", "ann_turnover", "ann_cost_drag", "tracking_error",
            "information_ratio", "beta_to_benchmark", "alpha_ann")
    out = {k: s[k] for k in keep if k in s}
    if gross is not None:
        out["gross_sharpe"] = sharpe_of(gross)
    return out
