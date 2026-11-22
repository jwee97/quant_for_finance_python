"""Evidence about a backtest, beyond its headline numbers, for options and futures alike.

* ``bootstrap`` puts confidence intervals on the Sharpe ratio, the mean return and the worst drawdown with the **stationary block bootstrap** (Politis and Romano 1994): returns are resampled in blocks of
  random length, so the serial dependence of daily P&L (overlapping positions, volatility clustering) is kept instead of assumed away. A short sample gives a wide interval: that is the finding.
* ``walk_forward`` is the honest version of "choose the best setting": for each fold the setting is chosen on a training window and judged on the window *after* it, which it never saw. The out-of-sample
  series is the stitched test windows. The in-sample winner of the whole period is reported beside it, so the gap between them (the cost of choosing) is visible.
* ``realism`` lists what a backtest models and what it does not, and labels it ``simplified`` unless every essential friction of the instrument is modelled. A backtest is never called realistic by
  default: the label is earned by the checklist, and the missing items are printed with the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ESSENTIAL = {
    "options": ["bid-ask spread crossed on every fill", "commissions", "no look-ahead: signal at one close, trade at a later quote", "expiry settlement", "assignment and early exercise (American options)",
                "margin requirements and funding", "liquidity limits (volume, open interest, size)", "contract multiplier", "dividends and discounting in valuation"],
    "futures": ["bid-ask spread crossed on every fill", "commissions", "no look-ahead: signal at one close, trade at a later price", "contract multiplier and tick size", "daily variation margin",
                "initial margin and collateral financing", "contract rolling on a stated rule", "roll execution cost (calendar-spread half-spread)", "liquidity limits (volume, open interest, size)", "expiry and delivery conventions"],
}


def _sharpe(r: np.ndarray, periods: float) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def _maxdd(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1.0).min())


def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap resample of ``range(n)``: blocks start at random places and continue with probability ``1 - 1/mean_block``, wrapping around."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new = rng.random(n) < p
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if new[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap(returns: pd.Series, n: int = 1000, mean_block: float | None = None, periods: float = 252.0, level: float = 0.95, seed: int = 0) -> dict:
    r = returns.dropna().to_numpy(float)
    if len(r) < 30:
        return {"available": False, "reason": f"only {len(r)} returns: too few to bootstrap"}
    block = mean_block or max(2.0, round(len(r) ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)
    sh, mu, dd = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        x = r[stationary_indices(len(r), block, rng)]
        sh[i], mu[i], dd[i] = _sharpe(x, periods), x.mean() * periods, _maxdd(x)
    lo = (1 - level) / 2 * 100
    q = lambda a: [float(np.nanpercentile(a, lo)), float(np.nanpercentile(a, 100 - lo))]  # noqa: E731
    point = {"sharpe": _sharpe(r, periods), "annual_mean_return": float(r.mean() * periods), "max_drawdown": _maxdd(r)}
    ci = {"sharpe": q(sh), "annual_mean_return": q(mu), "max_drawdown": q(dd)}
    return {"available": True, "method": "stationary block bootstrap (Politis and Romano 1994)", "resamples": n, "mean_block_length": float(block), "level": level, "n_returns": int(len(r)), "point": point, "interval": ci,
            "sharpe_interval_contains_zero": bool(ci["sharpe"][0] <= 0 <= ci["sharpe"][1]),
            "reading": "the Sharpe ratio is not distinguishable from zero in this sample" if ci["sharpe"][0] <= 0 <= ci["sharpe"][1] else "the Sharpe ratio is above zero at this confidence level in this sample (not a claim about any other sample)"}


def walk_forward(returns_by_setting: dict, train: int = 126, test: int = 42, embargo: int = 1, periods: float = 252.0) -> dict:
    """``returns_by_setting``: ``{setting: daily return Series}``, every setting run over the same dates. Each fold picks the setting with the best Sharpe ratio on its training days and takes that setting's
    returns on the test days that follow, after an ``embargo`` of days (the positions at the end of training overlap the start of the test)."""
    frame = pd.DataFrame(returns_by_setting).dropna(how="all")
    if frame.shape[1] < 2:
        raise ValueError("walk-forward needs at least two settings to choose between")
    if len(frame) < train + embargo + test:
        raise ValueError(f"not enough days: {len(frame)} available, {train} + {embargo} + {test} needed")
    folds, oos = [], []
    start = 0
    while start + train + embargo + test <= len(frame):
        tr = frame.iloc[start:start + train]
        te = frame.iloc[start + train + embargo:start + train + embargo + test]
        score = {c: _sharpe(tr[c].dropna().to_numpy(), periods) for c in frame.columns if tr[c].notna().sum() > 10}
        best = max(score, key=lambda c: score[c] if np.isfinite(score[c]) else -np.inf)
        folds.append({"train": [str(tr.index[0].date()), str(tr.index[-1].date())], "test": [str(te.index[0].date()), str(te.index[-1].date())], "chosen": best, "train_sharpe": float(score[best]),
                      "test_sharpe": _sharpe(te[best].dropna().to_numpy(), periods), "test_mean": float(te[best].mean())})
        oos.append(te[best])
        start += test
    out = pd.concat(oos)
    in_sample = {c: _sharpe(frame[c].dropna().to_numpy(), periods) for c in frame.columns}
    best_all = max(in_sample, key=lambda c: in_sample[c] if np.isfinite(in_sample[c]) else -np.inf)
    s_oos, s_in = _sharpe(out.to_numpy(), periods), in_sample[best_all]
    return {"folds": folds, "oos_sharpe": s_oos, "oos_days": int(len(out)), "in_sample_best": best_all, "in_sample_sharpe": s_in, "gap": float(s_in - s_oos) if np.isfinite(s_oos) and np.isfinite(s_in) else None,
            "settings_chosen": {k: int(v) for k, v in pd.Series([f["chosen"] for f in folds]).value_counts().items()}, "oos_bootstrap": bootstrap(out, n=400, periods=periods),
            "reading": ("the best setting of the whole period looks better than what choosing one in advance achieved: part of that gap is the cost of choosing, not a property of the strategy"
                        if np.isfinite(s_oos) and s_in - s_oos > 0.15 else "choosing the setting in advance gave about what the best setting of the whole period did")}


def realism(instrument: str, modelled: dict[str, bool], notes: dict[str, str] | None = None) -> dict:
    """``modelled`` maps checklist items (``ESSENTIAL[instrument]``) to whether the backtest models them. The label is ``simplified`` unless all of them are, in which case it is ``covers the listed
    frictions``; it is never "realistic". Unknown items in ``modelled`` are ignored, missing ones count as not modelled."""
    items = ESSENTIAL[instrument]
    rows = [{"item": i, "modelled": bool(modelled.get(i, False)), "note": (notes or {}).get(i)} for i in items]
    missing = [r["item"] for r in rows if not r["modelled"]]
    return {"label": "simplified" if missing else "covers the listed frictions", "modelled": [r["item"] for r in rows if r["modelled"]], "not_modelled": missing, "items": rows,
            "statement": ("Not a realistic simulation of trading: " + "; ".join(missing) + " are not modelled." if missing else "Every listed friction is modelled; unlisted ones may not be.")}
