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


def _wasserstein(w, c):
    from .robust_wasserstein import wasserstein_cvar_weights

    r = wasserstein_cvar_weights(w, eps=0.0005, alpha=0.95)
    if not r["ok"]:
        raise RuntimeError(r["message"])
    return r["weights"]


def _graph_capped(w, c):
    from .graph_constrained import graph_capped_weights

    return graph_capped_weights(pd.DataFrame(c["cov"], index=w.columns, columns=w.columns), "mst", cap=0.5, hops=1)["weights"]


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
    Method("wasserstein_cvar", "Wasserstein-robust CVaR", "tail", False, _wasserstein, "worst CVaR over a ball of radius 5 bp around the window's scenarios: CVaR + eps ||w|| / 5%"),
    Method("graph_capped", "Graph-capped minimum variance", "hierarchical", False, _graph_capped, "minimum variance with at most 50% inside any asset's neighbourhood on the minimum spanning tree"),
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


def diagnostics(res: dict, returns: pd.DataFrame, n_draws: int = 30, mu_noise: float = 1.0, frontier_points: int = 12, seed: int = 0) -> dict:
    """What a :func:`compare` result does not say: how concentrated each method's portfolio is, how stable its weights are from one rebalance to the next, how much they move when the return forecast or the
    covariance estimate is changed a little, what trading costs it, and where the methods sit against the long-only efficient frontier of the last window. Everything uses the SAME window, covariance estimate,
    return forecast, cap and costs as the comparison (read from ``res["config"]``); nothing is re-chosen per method.

    * concentration: effective number of positions (1 / HHI, averaged over rebalances), mean largest weight, share of the three largest, zero-weight positions, and how often the cap bound;
    * stability: mean one-way turnover between consecutive target weights (before drift) and the mean rank correlation of consecutive weight vectors;
    * sensitivity (at the last rebalance): the one-way change in weights (half the L1 distance) when the expected returns get noise of ``mu_noise`` cross-sectional standard deviations (only for methods that use them) and when
      the covariance is estimated by another method (sample versus shrinkage); the median over ``n_draws`` seeded draws;
    * cost attribution: yearly turnover times the cost per unit traded, against the gross-to-net return gap;
    * frontier: minimum-variance long-only portfolios under the same cap for a ladder of expected returns, with each method's last weights placed on it (volatility and expected return from the last window)."""
    from scipy.optimize import minimize

    cfg = res["config"]
    keys, cap, lookback, cov_method, mu_shrink, cost = cfg["methods"], cfg["max_weight"], cfg["lookback"], cfg["cov_method"], cfg["mu_shrink"], cfg["cost_bps"] * 1e-4
    data = returns.dropna(how="any")
    last_p = len(data) - 1
    rebal_dates = [pd.Timestamp(d) for d in res["rebalances"]]
    p = int(data.index.get_indexer([rebal_dates[-1]])[0])
    win = data.iloc[p - lookback:p]
    cov = estimate_covariance(win, cov_method, lookback, annualise=True).to_numpy(dtype=float)
    mu_raw = (win.mean() * ANN).to_numpy(dtype=float)
    mu = mu_shrink * mu_raw.mean() + (1 - mu_shrink) * mu_raw
    rng = np.random.default_rng(seed)
    ctx0 = {"cov": cov, "mu": mu, "lam": cfg["risk_aversion"]}

    def solve(key: str, ctx: dict) -> np.ndarray:
        w = np.asarray(pd.Series(METHODS[key].fn(win, ctx), index=win.columns).to_numpy(dtype=float))
        w = np.clip(w, 0.0, None)
        return cap_weights(w / w.sum(), cap)

    rows = {}
    for key in keys:
        W = res["weights"][key]
        w2 = (W ** 2).sum(axis=1)
        srt = np.sort(W.to_numpy(), axis=1)[:, ::-1]
        diffs = W.diff().dropna()
        corr = [float(W.iloc[i].rank().corr(W.iloc[i + 1].rank())) for i in range(len(W) - 1)]
        base = solve(key, ctx0)
        mu_moves, cov_moves = [], []
        if METHODS[key].uses_mu:
            sd = float(np.std(mu)) or float(np.std(mu_raw)) or 1e-6
            for _ in range(n_draws):
                try:
                    mu_moves.append(float(np.abs(solve(key, {**ctx0, "mu": mu + mu_noise * sd * rng.normal(size=len(mu))}) - base).sum() / 2))
                except Exception:
                    continue
        for alt in [m for m in ("sample", "shrinkage") if m != cov_method]:
            try:
                c2 = estimate_covariance(win, alt, lookback, annualise=True).to_numpy(dtype=float)
                cov_moves.append(float(np.abs(solve(key, {**ctx0, "cov": c2}) - base).sum() / 2))
            except Exception:
                continue
        row = res["table"].loc[key]
        rows[key] = {"label": METHODS[key].label, "effective_n": float(np.mean(1.0 / w2)), "hhi": float(w2.mean()), "max_weight": float(srt[:, 0].mean()), "top3_share": float(srt[:, :3].sum(axis=1).mean()),
                     "zero_weight_positions": float((W.to_numpy() < 1e-4).sum(axis=1).mean()), "cap_binding_share": float(row["capped"]),
                     "weight_turnover": float(diffs.abs().sum(axis=1).mean() / 2) if len(diffs) else float("nan"), "weight_rank_correlation": float(np.nanmean(corr)) if corr and np.isfinite(corr).any() else float("nan"),
                     "mu_sensitivity": float(np.median(mu_moves)) if mu_moves else None, "cov_sensitivity": float(np.median(cov_moves)) if cov_moves else None,
                     "uses_expected_returns": bool(METHODS[key].uses_mu), "turnover_per_year": float(row["turnover_per_year"]), "cost_drag_per_year": float(row["turnover_per_year"] * cost),
                     "gross_net_sharpe": [float(row["gross_sharpe"]), float(row["sharpe"])], "last_weights": {str(a): float(x) for a, x in zip(win.columns, base)}}
    # the long-only frontier under the same cap, from the same last-window cov and mu
    n = len(mu)
    lo, hi = float(mu.min()), float(min(mu.max(), np.sort(mu)[::-1][: max(int(np.ceil(1 / cap)), 1)].mean()))
    pts = []
    w0 = np.full(n, 1.0 / n)
    for target in np.linspace(lo, hi, frontier_points):
        r = minimize(lambda w: w @ cov @ w, w0, jac=lambda w: 2 * cov @ w, bounds=[(0.0, cap)] * n, method="SLSQP",
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}, {"type": "eq", "fun": lambda w, t=target: w @ mu - t}], options={"maxiter": 200})
        if r.success:
            pts.append({"return": float(target), "vol": float(np.sqrt(max(r.x @ cov @ r.x, 0.0)))})
    placed = {k: {"return": float(np.array(list(v["last_weights"].values())) @ mu), "vol": float(np.sqrt(max(np.array(list(v["last_weights"].values())) @ cov @ np.array(list(v["last_weights"].values())), 0.0)))}
              for k, v in rows.items()}
    return {"methods": rows, "frontier": pts, "placed": placed, "window": [str(win.index[0].date()), str(win.index[-1].date())], "as_of_rebalance": str(rebal_dates[-1].date()),
            "conditions": {"lookback": lookback, "cov_method": cov_method, "mu_shrink": mu_shrink, "max_weight": cap, "cost_bps": cfg["cost_bps"], "risk_aversion": cfg["risk_aversion"], "step": cfg["step"], "rebalances": len(rebal_dates)},
            "notes": ["Every method was given the same windows, covariance, expected-return forecast, cap and costs.",
                      "Sensitivity is at the last rebalance only: half the L1 change in weights (0 = unchanged, 1 = a completely different portfolio) when the expected returns get noise, or the covariance is estimated another way.",
                      "A method with a higher Sharpe on the history is not better if its weights are concentrated, unstable or collapse under small changes of the inputs: read the Sharpe with these columns."]}
