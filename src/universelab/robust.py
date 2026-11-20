"""What makes a portfolio result believable, or not.

All of it is computed on the stitched out-of-sample returns of ``discover`` (days on which the choice of configuration was made without seeing them), against a benchmark on the same days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def sharpe(x: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(ANN)) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def summary(net: pd.Series) -> dict:
    x = net.dropna()
    if len(x) < 5:
        return {"n": int(len(x))}
    eq = (1 + x).cumprod()
    years = len(x) / ANN
    dd = eq / eq.cummax() - 1
    q = x.quantile(0.05)
    return {"n": int(len(x)), "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else float("nan"), "ann_vol": float(x.std(ddof=1) * np.sqrt(ANN)), "sharpe": sharpe(x),
            "max_drawdown": float(dd.min()), "es95": float(x[x <= q].mean()) if (x <= q).any() else float("nan"), "total_return": float(eq.iloc[-1] - 1)}


def beta(net: pd.Series, bench: pd.Series) -> float:
    d = pd.concat([net, bench], axis=1, keys=["a", "b"]).dropna()
    return float(np.cov(d["a"], d["b"])[0, 1] / d["b"].var()) if len(d) > 5 and d["b"].var() > 0 else float("nan")


def stability(weights: pd.DataFrame) -> dict:
    """How much the holdings change from one rebalance to the next: the Jaccard overlap of the sets held, and the average one-way weight change."""
    if len(weights) < 2:
        return {"jaccard": float("nan"), "weight_change": float("nan")}
    sets = [set(weights.columns[(row > 1e-9).to_numpy()]) for _, row in weights.iterrows()]
    jac = [len(a & b) / len(a | b) for a, b in zip(sets[:-1], sets[1:]) if a | b]
    ch = (weights.diff().abs().sum(axis=1) / 2.0).iloc[1:]
    return {"jaccard": float(np.mean(jac)) if jac else float("nan"), "weight_change": float(ch.mean())}


def concentration(weights: pd.DataFrame, groups: dict | None = None) -> dict:
    """The effective number of names (1 / sum of squared weights) and, with groups, the largest group's share, averaged over rebalances."""
    w = weights.fillna(0.0)
    hhi = (w ** 2).sum(axis=1).replace(0, np.nan)
    out = {"effective_names": float((1.0 / hhi).mean()), "max_weight": float(w.max(axis=1).mean())}
    if groups:
        g = pd.Series({c: groups.get(c, "other") for c in w.columns})
        share = w.T.groupby(g).sum().T
        out["largest_group_share"] = float(share.max(axis=1).mean())
        out["largest_group"] = str(share.mean().idxmax())
    return out


def block_bootstrap(net: pd.Series, bench: pd.Series | None = None, n: int = 500, block: int = 21, seed: int = 0) -> dict:
    """A stationary-style block bootstrap of the daily returns: the spread of the Sharpe ratio, the chance it is above zero and (given a benchmark) above the benchmark's, resampling the pair together."""
    d = pd.concat([net, bench], axis=1).dropna() if bench is not None else net.dropna().to_frame()
    a = d.iloc[:, 0].to_numpy()
    b = d.iloc[:, 1].to_numpy() if bench is not None else None
    m = len(a)
    if m < 3 * block:
        return {"n": int(m), "note": "too few days to bootstrap"}
    rng = np.random.default_rng(seed)
    starts_per = int(np.ceil(m / block))
    sh, diff = [], []
    for _ in range(n):
        idx = np.concatenate([(s + np.arange(block)) % m for s in rng.integers(0, m, starts_per)])[:m]
        sh.append(sharpe(a[idx]))
        if b is not None:
            diff.append(sharpe(a[idx]) - sharpe(b[idx]))
    sh = np.array(sh)
    out = {"n": int(m), "draws": n, "block": block, "sharpe_lo": float(np.nanpercentile(sh, 5)), "sharpe_hi": float(np.nanpercentile(sh, 95)), "p_sharpe_positive": float(np.nanmean(sh > 0))}
    if b is not None:
        diff = np.array(diff)
        out.update({"p_beats_benchmark": float(np.nanmean(diff > 0)), "diff_lo": float(np.nanpercentile(diff, 5)), "diff_hi": float(np.nanpercentile(diff, 95))})
    return out


def regimes(net: pd.Series, bench: pd.Series, window: int = 63) -> list[dict]:
    """The result in rising and falling markets (by the benchmark's trailing return) and in calm and turbulent ones (by its trailing volatility, split at the median). The state on a day uses only days before it."""
    d = pd.concat([net, bench], axis=1, keys=["net", "bench"]).dropna()
    trail = bench.rolling(window).sum().shift(1).reindex(d.index)
    vol = bench.rolling(window).std().shift(1).reindex(d.index)
    rows = []
    for name, mask in (("market rising", trail > 0), ("market falling", trail <= 0), ("calm", vol <= vol.median()), ("turbulent", vol > vol.median())):
        s = d[mask.fillna(False)]
        if len(s) >= 30:
            rows.append({"regime": name, "days": int(len(s)), "strategy_sharpe": sharpe(s["net"]), "benchmark_sharpe": sharpe(s["bench"]), "strategy_ann_return": float(s["net"].mean() * ANN), "benchmark_ann_return": float(s["bench"].mean() * ANN)})
    return rows
