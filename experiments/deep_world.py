"""Deep networks, generative scenarios, physics-informed pricing, graph portfolios and combinatorial cross-validation, measured on worlds with a known truth and on the platform's 15 ETFs.

Simulated worlds check that each tool does its job; the 15-ETF runs are reported as they came out (a short, small sample). Nothing is tuned.

    python -m experiments.deep_world          # about fifteen minutes
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.framework import ALLOCATORS, MODELS, bundle_from_prices, load_library
from src.framework.allocation import Context
from src.scenarios import compare, generate
from src.scenarios import backtest as sb
from src.utils.config import load_config
from src.validation import cpcv


def etf_bundle():
    from src.framework.data import load_default_bundle

    return load_default_bundle(load_config())


# ------------------------------------------------------------------------------------------------------------------ 1. network families
def planted_world(seed=0, n=3000, k=12):
    """Returns with a nonlinear effect planted: the next month tilts by tanh of the last month's cumulative return (the world of the deep-model tests, longer)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.012, size=(n, k))
    for t in range(22, n):
        r[t] += 0.06 * np.tanh(r[t - 21:t].sum(axis=0) / 0.2) / 21
    idx = pd.bdate_range("2005-01-03", periods=n)
    return bundle_from_prices(pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=[f"A{i}" for i in range(k)]), name="planted")


def monthly_ic(bundle, score, skip=0):
    from src.equity import alpha_model as am
    from src.utils.dates import rebalance_dates

    grid = rebalance_dates(bundle.index, "monthly")
    px = bundle.prices.loc[grid]
    ic = am.information_coefficients({"x": score.loc[grid]}, px.shift(-1) / px - 1.0, "spearman", 5)["x"].dropna().iloc[skip:]
    return {"mean IC": float(ic.mean()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic))), "months": len(ic)}


def network_table(kinds=("fnn", "cnn", "lstm", "gru", "transformer", "patchtst", "nbeats")) -> pd.DataFrame:
    rows = {}
    planted = planted_world()
    for kind in kinds:
        t0 = time.time()
        model = MODELS.create("deep_window", kind=kind, min_train=1000, max_epochs=8, patience=3)
        planted_ic = monthly_ic(planted, model.score(planted))
        rows[kind] = {"IC, planted nonlinear world": planted_ic["mean IC"], "t": planted_ic["t"], "seconds": time.time() - t0}
    return pd.DataFrame(rows).T


def etf_network_table(kinds=("fnn", "cnn", "lstm", "transformer")) -> pd.DataFrame:
    bundle = etf_bundle()
    rows = {}
    for kind in kinds:
        model = MODELS.create("deep_window", kind=kind, max_epochs=8, patience=3)
        ic = monthly_ic(bundle, model.score(bundle))
        rows[kind] = {"IC on the 15 ETFs": ic["mean IC"], "t": ic["t"], "months": ic["months"]}
    ridge = MODELS.create("ml_factor_model", kind="ridge")
    ic = monthly_ic(bundle, ridge.score(bundle))
    rows["ridge on price characteristics (for scale)"] = {"IC on the 15 ETFs": ic["mean IC"], "t": ic["t"], "months": ic["months"]}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------------------ 2. scenarios
GENS = (("historical", {}), ("bootstrap", {"kind": "iid"}), ("bootstrap", {"kind": "stationary", "block": 20.0}), ("copula", {}), ("risk_factor", {}), ("arima_garch", {}),
        ("factor_vae", {"factors": 3, "epochs": 150}), ("wgan_gp", {}), ("diffusion", {"epochs": 200}))


def label(name, kw):
    return name if "kind" not in kw else f"{name} ({kw['kind']})"


def scenario_table() -> pd.DataFrame:
    r = etf_bundle().returns[["SPY", "TLT", "GLD", "EEM", "IEF"]].dropna().iloc[-2500:]
    sets = {}
    for name, kw in GENS:
        sets[label(name, kw)] = generate(name, r, 500, 21, 0, **kw)
    return compare(r, sets)


def var_backtest_table() -> pd.DataFrame:
    """Fit each generator on the first half of five ETFs' history and ask for the 99% VaR of the equal-weight portfolio over one day; count the exceedances in the second half (Kupiec's test: the rate should be 1%)."""
    r = etf_bundle().returns[["SPY", "TLT", "GLD", "EEM", "IEF"]].dropna().iloc[-3000:]
    train, test = r.iloc[:1500], r.iloc[1500:]
    port = test.mean(axis=1).to_numpy()
    rows = {}
    for name, kw in GENS:
        S = generate(name, train, 4000, 1, 0, **kw)[:, 0].mean(axis=1)
        var = -np.quantile(S, 0.01)
        k = int((port < -var).sum())
        n = len(port)
        phat = k / n
        lr = -2 * (np.log(0.99 ** (n - k) * 0.01 ** k) - np.log((1 - phat) ** (n - k) * phat ** k)) if 0 < k < n else float("nan")
        rows[label(name, kw)] = {"99% VaR": var, "exceedance rate": phat, "Kupiec p-value": float(1 - stats.chi2.cdf(lr, 1)) if np.isfinite(lr) else float("nan")}
    return pd.DataFrame(rows).T


def scenario_backtest_table() -> pd.DataFrame:
    r = etf_bundle().returns[["SPY", "TLT", "GLD", "EEM", "IEF"]].dropna().iloc[-3000:]
    equal = lambda x: pd.DataFrame(1.0 / x.shape[1], index=x.index, columns=x.columns)                    # noqa: E731

    def trend(x):
        px = (1 + x).cumprod()
        on = (px / px.shift(126) > 1).astype(float).shift(1).fillna(0.0)                                 # 6-month trend, decided at the previous close
        return on.div(x.shape[1])

    rows = {}
    for gen, kw in (("bootstrap", {"kind": "iid"}), ("bootstrap", {"kind": "stationary", "block": 20.0}), ("arima_garch", {})):
        for sname, strat in (("equal weight", equal), ("6-month trend", trend)):
            t = sb.scenario_backtest(strat, r, gen, n_paths=100, horizon=756, seed=1, cost_bps=5, **kw)
            rows[(label(gen, kw), sname)] = {"median Sharpe": t["sharpe"].median(), "5th percentile Sharpe": t["sharpe"].quantile(0.05), "median max drawdown": t["max_drawdown"].median(),
                                              "5th percentile terminal wealth": t["terminal_wealth"].quantile(0.05)}
    out = pd.DataFrame(rows).T
    out.index.names = ["scenarios", "rule"]
    return out


# ------------------------------------------------------------------------------------------------------------------ 3. PINNs
def pinn_table() -> pd.DataFrame:
    from src.derivatives.pricing import bsm_price, heston_price
    from src.models import pinn

    rows = {}
    t0 = time.time()
    net = pinn.BSMPINN(0.03, 0.01, 0.25, 1.0).fit()
    S = np.array([80, 90, 100, 110, 120.0])
    err = [np.abs(net.price(S, 100, tau) - np.array([bsm_price(s, 100, tau, 0.03, 0.01, 0.25) for s in S])).max() for tau in (0.25, 0.5, 1.0)]
    rows["Black-Scholes-Merton call (strike 100)"] = {"max error, shortest maturity": err[0], "middle": err[1], "longest": err[2], "seconds": time.time() - t0}
    t0 = time.time()
    net = pinn.VasicekPINN(0.8, 0.05, 0.02, 5.0, steps=2000).fit()
    r = np.array([0.0, 0.03, 0.05, 0.08, 0.10])
    err = [np.abs(net.price(r, tau) - pinn.vasicek_bond(r, tau, 0.8, 0.05, 0.02)).max() for tau in (1.0, 3.0, 5.0)]
    rows["Vasicek zero-coupon bond (price per 1)"] = {"max error, shortest maturity": err[0], "middle": err[1], "longest": err[2], "seconds": time.time() - t0}
    S, v = np.array([85, 95, 100, 105, 115.0]), np.array([0.02, 0.04, 0.04, 0.04, 0.08])
    for name, kw, ref in (("Heston call", {}, lambda s, vv, tau: heston_price(s, 100, tau, 0.03, 0.0, vv, 2.0, 0.04, 0.3, -0.7)),
                          ("Bates call", dict(lam=0.5, mu_j=-0.1, sigma_j=0.15), lambda s, vv, tau: pinn.bates_price(s, 100, tau, 0.03, 0.0, vv, 2.0, 0.04, 0.3, -0.7, 0.5, -0.1, 0.15))):
        t0 = time.time()
        net = pinn.HestonPINN(0.03, 0.0, 2.0, 0.04, 0.3, -0.7, **kw).fit()
        err = [np.abs(net.price(S, 100, tau, v) - np.array([ref(s, vv, tau) for s, vv in zip(S, v)])).max() for tau in (0.25, 0.5, 1.0)]
        rows[name + " (strike 100)"] = {"max error, shortest maturity": err[0], "middle": err[1], "longest": err[2], "seconds": time.time() - t0}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------------------ 4. graph portfolios
def stats_of(w, r, start="2010-01-01"):
    x = (w.shift(1).fillna(0.0) * r).sum(axis=1)
    x = x[x.index >= start]
    eq = (1 + x).cumprod()
    hhi = (w[w.index >= start] ** 2).sum(axis=1).mean()
    turnover = w[w.index >= start].diff().abs().sum(axis=1).mean() * 252
    return {"return": 252 * x.mean(), "volatility": x.std() * np.sqrt(252), "Sharpe": x.mean() / x.std() * np.sqrt(252), "max drawdown": float((eq / eq.cummax()).min() - 1),
            "effective N": 1.0 / hhi, "turnover/yr": turnover}


def graph_table() -> pd.DataFrame:
    bundle = etf_bundle()
    cfg = load_config()
    ctx = Context(bundle, cfg)
    r = bundle.returns.fillna(0.0)
    rows = {}
    for label_, name, kw in (("equal weight", "static", {"book": "equal_weight"}), ("inverse volatility", "static", {"book": "inverse_vol"}), ("risk parity", "static", {"book": "risk_parity"}),
                             ("HRP", "static", {"book": "hrp"}),
                             ("HSP, risk sensitivity", "hierarchical_sensitivity_parity", {"sensitivity": "risk"}), ("HSP, factor sensitivity", "hierarchical_sensitivity_parity", {"sensitivity": "factor"}),
                             ("TMFG, clique centrality, peripheral", "graph_centrality", {}), ("TMFG, clique centrality, central", "graph_centrality", {"tilt": "central"}),
                             ("MST, strength, peripheral", "graph_centrality", {"graph": "mst", "centrality": "strength"}),
                             ("TMFG, eigenvector, peripheral", "graph_centrality", {"centrality": "eigenvector"})):
        rows[label_] = stats_of(ALLOCATORS.create(name, **kw).build(ctx), r)
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------------------ 5. CPCV
def cpcv_table(seed=0) -> pd.DataFrame:
    """A ridge regression of next-day return of an equal-weighted ETF book on its last five days, trained on the other groups, trading the sign. The CPCV paths show how much the Sharpe ratio of the same rule
    changes with the part of history it is tested on; walk-forward gives one number."""
    r = etf_bundle().returns.dropna().mean(axis=1).to_numpy()
    n = len(r)
    X = np.column_stack([np.r_[np.zeros(k), r[:-k]] for k in range(1, 6)])
    y = r
    splits = cpcv.cpcv_splits(n, 8, 2, horizon=1, embargo=5)
    preds = []
    for sp in splits:
        A = X[sp.train]
        coef = np.linalg.solve(A.T @ A + 1e-4 * np.eye(5) * len(A), A.T @ y[sp.train])
        preds.append(np.sign(X[sp.test] @ coef))
    pos = cpcv.assemble_paths(n, splits, preds, 8, 2)
    sharpe = lambda x: x.mean() / x.std() * np.sqrt(252)                                              # noqa: E731
    paths = [sharpe(p * y) for p in pos]
    wf = []
    start = n // 2
    A = X[:start]
    coef = np.linalg.solve(A.T @ A + 1e-4 * np.eye(5) * len(A), A.T @ y[:start])
    wf = sharpe(np.sign(X[start:] @ coef) * y[start:])
    buy = sharpe(y)
    return pd.DataFrame({"CPCV paths (8 groups, 2 tested: 7 paths)": {"mean Sharpe": float(np.mean(paths)), "std over paths": float(np.std(paths, ddof=1)), "min": float(np.min(paths)), "max": float(np.max(paths))},
                         "one walk-forward split (second half)": {"mean Sharpe": float(wf)}, "buy and hold, whole history": {"mean Sharpe": float(buy)}}).T


def run() -> dict:
    load_library()
    warnings.simplefilter("ignore")
    return {"networks_planted": network_table(), "networks_etf": etf_network_table(), "scenarios": scenario_table(), "var_backtest": var_backtest_table(),
            "scenario_backtest": scenario_backtest_table(), "pinn": pinn_table(), "graph": graph_table(), "cpcv": cpcv_table()}


def main() -> None:
    pd.options.display.float_format = "{:.4g}".format
    pd.options.display.width = 240
    t0 = time.time()
    for name, table in run().items():
        print(f"== {name} ==")
        print(table.to_string(), "\n", flush=True)
    print(f"{time.time() - t0:.0f} seconds")


if __name__ == "__main__":
    main()
