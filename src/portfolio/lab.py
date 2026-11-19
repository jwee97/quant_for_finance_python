"""The Portfolio Lab: optimisers compared like for like.

Every method is given the same data: the same trailing window of returns on every rebalance date, the same covariance estimator, the same expected-return estimate, the same cap on a position, the same
trading cost, and the same holding period (weights drift with prices between rebalances). A method is judged on what a person would ask of it: return, risk, drawdown, turnover, concentration,
diversification, and how much its weights jump when the data are resampled.

* A method that fails on a date (the solver did not converge, a window too short) falls back to equal weight on that date, and the count of such dates is reported (``fallbacks``). A table that hid them would
  flatter the methods that fail most.
* The cap is applied to all methods the same way, by clipping and redistributing, after the method has chosen: ``capped`` is the average weight it moved, so a method that needs the cap to be a sane
  portfolio can be seen to need it.
* Expected returns for the methods that use them are the sample mean shrunk toward the grand mean (``mu_shrink``); the methods that do not (minimum variance, risk parity, HRP, HERC, NCO's
  minimum-variance form) ignore them.

This is an evaluation harness, not a recommendation: with a few years of data per asset the differences between methods are usually inside the noise, and the table says so in its ``sharpe_se`` column.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import drawdown as ddm
from . import near_optimal as noc
from .covariance import estimate_covariance
from .nco import long_only, nco_weights

ANN = 252.0


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    family: str
    uses_mu: bool
    fn: object            # fn(window: DataFrame, ctx: dict) -> weights (Series or array in column order)
    note: str = ""


def _equal(w, c):
    return np.full(w.shape[1], 1.0 / w.shape[1])


def _inverse_vol(w, c):
    v = 1.0 / np.sqrt(np.diag(c["cov"]))
    return v / v.sum()


def _min_variance(w, c):
    return long_only(c["cov"], None, "min_variance")


def _max_sharpe(w, c):
    return long_only(c["cov"], c["mu"], "max_sharpe")


def _risk_parity(w, c):
    return long_only(c["cov"], None, "risk_parity")


def _hrp(w, c):
    from .hierarchical import hrp_weights

    return hrp_weights(pd.DataFrame(c["cov"], index=w.columns, columns=w.columns))


def _herc(w, c):
    from .hierarchical import herc_weights

    return herc_weights(pd.DataFrame(c["cov"], index=w.columns, columns=w.columns))


def _nco_min(w, c):
    return nco_weights(pd.DataFrame(c["cov"], index=w.columns, columns=w.columns))


def _nco_sharpe(w, c):
    return nco_weights(pd.DataFrame(c["cov"], index=w.columns, columns=w.columns), pd.Series(c["mu"], index=w.columns), "max_sharpe")


def _mean_cvar(w, c):
    from .cvar_optimize import mean_cvar_weights

    return mean_cvar_weights(w, None, alpha=0.95)


def _cdar(w, c):
    r = ddm.cdar_weights(w, 0.95)
    if not r["ok"]:
        raise RuntimeError(r["message"])
    return r["weights"]


def _max_dd(w, c):
    r = ddm.max_drawdown_weights(w)
    if not r["ok"]:
        raise RuntimeError(r["message"])
    return r["weights"]


def _evar(w, c):
    r = ddm.evar_weights(w, 0.95)
    if not r["ok"]:
        raise RuntimeError(r["message"])
    return r["weights"]


def _mean_variance(w, c):
    return noc.optimum(c["mu"], c["cov"], c["lam"])


def _near_optimal(w, c):
    r = noc.near_optimal_center(pd.Series(c["mu"], index=w.columns), pd.DataFrame(c["cov"], index=w.columns, columns=w.columns), c["lam"], tolerance=0.2)
    if not r["converged"]:
        raise RuntimeError("the centring did not converge")
    return r["weights"]


def _robust(w, c):
    from .mean_variance import robust_weights

    r = robust_weights(pd.Series(c["mu"], index=w.columns), pd.DataFrame(c["cov"], index=w.columns, columns=w.columns), risk_aversion=c["lam"], kappa=1.0)
    if not r.success:
        raise RuntimeError(r.message)
    return r.weights


def _black_litterman(w, c):
    """Equilibrium returns from equal-weight 'market' weights, tilted by relative views from the trailing return of each asset over the window (the extremes of the cross-section)."""
    from .black_litterman import black_litterman_weights

    cov = pd.DataFrame(c["cov"], index=w.columns, columns=w.columns)
    r = black_litterman_weights(cov, pd.Series(1.0 / w.shape[1], index=w.columns), signal=(1.0 + w).prod() - 1.0, risk_aversion=c["lam"])
    if not r.success:
        raise RuntimeError(r.message)
    return r.weights


METHODS = {m.key: m for m in [
    Method("equal_weight", "Equal weight", "heuristic", False, _equal, "1/N: the benchmark every optimiser must beat"),
    Method("inverse_vol", "Inverse volatility", "heuristic", False, _inverse_vol),
    Method("min_variance", "Minimum variance", "risk", False, _min_variance, "long-only"),
    Method("max_sharpe", "Maximum Sharpe", "mean-variance", True, _max_sharpe, "long-only; uses the shrunk expected returns"),
    Method("mean_variance", "Mean-variance (plug-in)", "mean-variance", True, _mean_variance, "risk aversion 3, long-only"),
    Method("near_optimal", "Near-optimal centre", "mean-variance", True, _near_optimal, "centre of the portfolios within 20% of the best utility advantage over equal weight (Cajas)"),
    Method("robust", "Robust mean-variance", "mean-variance", True, _robust, "ellipsoidal uncertainty in the returns"),
    Method("black_litterman", "Black-Litterman", "mean-variance", False, _black_litterman, "equilibrium prior, momentum views"),
    Method("risk_parity", "Risk parity", "risk", False, _risk_parity),
    Method("hrp", "Hierarchical risk parity", "hierarchical", False, _hrp),
    Method("herc", "HERC", "hierarchical", False, _herc),
    Method("nco", "NCO (minimum variance)", "hierarchical", False, _nco_min, "nested clustered optimisation"),
    Method("nco_sharpe", "NCO (maximum Sharpe)", "hierarchical", True, _nco_sharpe),
    Method("mean_cvar", "CVaR (95%)", "tail", False, _mean_cvar, "minimise expected shortfall on the window's scenarios"),
    Method("cdar", "CDaR (95%)", "drawdown", False, _cdar, "minimise the conditional drawdown at risk of the window's path"),
    Method("max_drawdown", "Minimum max drawdown", "drawdown", False, _max_dd),
    Method("evar", "EVaR (95%)", "tail", False, _evar, "entropic value at risk"),
]}
COV_METHODS = ("sample", "shrinkage", "ewma", "pca_denoised", "graphical_lasso", "factor")


class LabError(ValueError):
    pass


def cap_weights(w: np.ndarray, cap: float, iterations: int = 60) -> np.ndarray:
    """Clip every weight at ``cap`` and spread the excess over the uncapped weights in proportion (repeated until none exceeds it). Raises if ``cap * n < 1``."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    n = len(w)
    if cap * n < 1.0 - 1e-12:
        raise LabError(f"a cap of {cap} on {n} assets cannot be fully invested")
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    for _ in range(iterations):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over & (w < cap - 1e-12)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum() if w[free].sum() > 0 else excess / free.sum()
    return w / w.sum()


def _metrics(net: pd.Series, gross: pd.Series, turnovers: list, weights: pd.DataFrame, covs: dict, caps: list, fallbacks: int, instab: list, runtime: float) -> dict:
    n = len(net)
    mean, sd = float(net.mean()), float(net.std(ddof=1))
    ann_ret, ann_vol = mean * ANN, sd * np.sqrt(ANN)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    eq = (1.0 + net).cumprod()
    dd = (eq / eq.cummax() - 1.0)
    half = n // 2

    def sh(x):
        s = x.std(ddof=1)
        return float(x.mean() / s * np.sqrt(ANN)) if len(x) > 5 and s > 0 else float("nan")

    div = []
    for t, w in weights.iterrows():
        c = covs[t]
        v = float(np.sqrt(max(w.to_numpy() @ c @ w.to_numpy(), 1e-18)))
        div.append(float(w.to_numpy() @ np.sqrt(np.diag(c))) / v)
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "sharpe_se": float(np.sqrt((1 + 0.5 * sharpe ** 2) / (n / ANN))) if np.isfinite(sharpe) else float("nan"),
            "gross_sharpe": float(gross.mean() / gross.std(ddof=1) * np.sqrt(ANN)) if gross.std(ddof=1) > 0 else float("nan"), "max_drawdown": float(dd.min()), "calmar": ann_ret / abs(float(dd.min())) if dd.min() < 0 else float("nan"),
            "cdar_95": ddm.cdar(net.to_numpy(), 0.95), "turnover_per_rebalance": float(np.mean(turnovers)), "turnover_per_year": float(np.sum(turnovers) / (n / ANN)),
            "effective_n": float(np.mean([1.0 / (w ** 2).sum() for w in weights.to_numpy()])), "max_weight": float(weights.max(axis=1).mean()), "diversification_ratio": float(np.mean(div)),
            "capped": float(np.mean(caps)), "fallbacks": int(fallbacks), "instability": float(np.mean(instab)) if instab else float("nan"), "sharpe_first_half": sh(net.iloc[:half]), "sharpe_second_half": sh(net.iloc[half:]),
            "seconds": runtime}


def compare(returns: pd.DataFrame, methods: list[str] | None = None, lookback: int = 252, step: int = 21, cost_bps: float = 10.0, cov_method: str = "shrinkage", mu_shrink: float = 0.8, max_weight: float = 1.0,
            risk_aversion: float = 3.0, perturb: int = 0, seed: int = 0, progress=None) -> dict:
    """Run the methods over the same walk-forward schedule and tabulate them. ``perturb`` is the number of bootstrap resamples of the window used (at up to six rebalance dates) to measure how much a
    method's weights move when the data do; 0 skips it."""
    keys = list(methods or ["equal_weight", "min_variance", "risk_parity", "hrp", "nco", "max_sharpe"])
    unknown = [k for k in keys if k not in METHODS]
    if unknown:
        raise LabError(f"unknown method(s): {', '.join(unknown)}")
    if cov_method not in COV_METHODS:
        raise LabError(f"cov_method must be one of {', '.join(COV_METHODS)}")
    if lookback < 60 or step < 1 or not 0.0 <= mu_shrink <= 1.0 or cost_bps < 0 or not 0.0 < max_weight <= 1.0:
        raise LabError("lookback >= 60, step >= 1, 0 <= mu_shrink <= 1, cost_bps >= 0 and 0 < max_weight <= 1")
    data = returns.dropna(how="any")
    n_assets = data.shape[1]
    if n_assets < 3:
        raise LabError("need at least 3 assets with a common history")
    if max_weight * n_assets < 1.0 - 1e-12:
        raise LabError(f"a cap of {max_weight} on {n_assets} assets cannot be fully invested")
    if len(data) < lookback + 2 * step:
        raise LabError(f"need at least {lookback + 2 * step} common rows, have {len(data)}")
    R = data.to_numpy(dtype=float)
    dates = list(range(lookback, len(data) - 1, step))
    rng = np.random.default_rng(seed)
    pert_at = set(dates[:: max(len(dates) // 6, 1)][:6]) if perturb else set()
    cost = cost_bps * 1e-4
    ctxs = {}
    for p in dates:
        win = data.iloc[p - lookback:p]
        cov = estimate_covariance(win, cov_method, lookback, annualise=True)
        mu = win.mean() * ANN
        mu = (mu_shrink * mu.mean() + (1 - mu_shrink) * mu)
        ctxs[p] = {"cov": cov.to_numpy(dtype=float), "mu": mu.to_numpy(dtype=float), "lam": risk_aversion}
    out = {"net": {}, "weights": {}, "metrics": {}, "notes": {}}
    for mi, key in enumerate(keys):
        m = METHODS[key]
        t0 = time.time()
        target_rows, covs, caps, turnovers, instab = {}, {}, [], [], []
        fallbacks = 0
        net = np.zeros(len(data))
        gross = np.zeros(len(data))
        held = np.zeros(n_assets)
        for p in dates:
            win = data.iloc[p - lookback:p]
            ctx = ctxs[p]
            try:
                w = np.asarray(pd.Series(m.fn(win, ctx), index=win.columns).reindex(win.columns).to_numpy(dtype=float))
                if not np.isfinite(w).all() or w.sum() <= 0:
                    raise RuntimeError("non-finite weights")
                w = np.clip(w, 0.0, None) / np.clip(w, 0.0, None).sum()
            except Exception:
                w, fallbacks = np.full(n_assets, 1.0 / n_assets), fallbacks + 1
            capped = cap_weights(w, max_weight)
            caps.append(float(np.abs(capped - w).sum() / 2))
            w = capped
            if p in pert_at:
                for _ in range(perturb):
                    idx = rng.integers(0, len(win), len(win))
                    boot = win.iloc[idx].reset_index(drop=True)
                    try:
                        bc = {"cov": estimate_covariance(boot, cov_method, lookback, annualise=True).to_numpy(dtype=float), "lam": risk_aversion}
                        bm = boot.mean() * ANN
                        bc["mu"] = (mu_shrink * bm.mean() + (1 - mu_shrink) * bm).to_numpy(dtype=float)
                        bw = cap_weights(np.asarray(pd.Series(m.fn(boot, bc), index=win.columns).to_numpy(dtype=float)), max_weight)
                        instab.append(float(np.abs(bw - w).sum() / 2))
                    except Exception:
                        pass
            traded = float(np.abs(w - held).sum())
            turnovers.append(traded / 2.0)
            target_rows[data.index[p]] = w
            covs[data.index[p]] = ctx["cov"] / ANN
            end = min(p + step, len(data))
            cur = w.copy()
            for t in range(p, end):
                gr = float(cur @ R[t])
                gross[t] = gr
                net[t] = gr - (cost * traded if t == p else 0.0)
                cur = cur * (1.0 + R[t]) / (1.0 + gr)
            held = cur
        start = dates[0]
        idx = data.index[start:min(dates[-1] + step, len(data))]
        netS = pd.Series(net[start:start + len(idx)], index=idx)
        grossS = pd.Series(gross[start:start + len(idx)], index=idx)
        W = pd.DataFrame(target_rows, index=data.columns).T
        out["net"][key] = netS
        out["weights"][key] = W
        out["metrics"][key] = _metrics(netS, grossS, turnovers, W, covs, caps, fallbacks, instab, time.time() - t0)
        out["notes"][key] = m.note
        if progress:
            progress(f"{m.label} ({mi + 1}/{len(keys)})", (mi + 1) / len(keys))
    table = pd.DataFrame(out["metrics"]).T
    table.insert(0, "method", [METHODS[k].label for k in table.index])
    table.insert(1, "family", [METHODS[k].family for k in table.index])
    equity = pd.DataFrame({k: (1.0 + s).cumprod() for k, s in out["net"].items()})
    return {"table": table, "equity": equity, "weights": out["weights"], "notes": out["notes"], "rebalances": [str(data.index[p].date()) for p in dates],
            "config": {"methods": keys, "lookback": lookback, "step": step, "cost_bps": cost_bps, "cov_method": cov_method, "mu_shrink": mu_shrink, "max_weight": max_weight, "risk_aversion": risk_aversion,
                       "perturb": perturb, "seed": seed, "assets": n_assets, "rows": len(data), "first": str(data.index[dates[0]].date()), "last": str(equity.index[-1].date())}}
