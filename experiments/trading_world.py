"""Multi-period trading, limit orders and smart order routing on simulated worlds with a known truth, and what they find.

Everything here is simulated: returns whose predictable part follows a known process, a market whose order flow follows a known law, and venues whose hidden liquidity is drawn from known distributions. So the
question is whether each method does what its theory says and how the methods compare, not whether anything would make money or save money in a real market. Nothing is tuned; sizes and seeds are fixed.

    python -m experiments.trading_world          # about three minutes
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.algo import limit_orders as lo
from src.algo import routing as rt
from src.equity import multiperiod as mp


# ------------------------------------------------------------------------------------------------------------------ 1. multi-period portfolios
def predictable_world(n: int = 6, periods: int = 240, persistence: float = 0.8, seed: int = 0):
    """Returns ``r_(t+1) = alpha_t + e_(t+1)`` where the expected return ``alpha_t`` of each asset is a persistent AR(1) process and ``e`` has a factor covariance; the investor sees ``alpha_t`` when trading."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 2))
    S = (F @ F.T / 2 + np.diag(rng.uniform(0.5, 1.0, n))) * 0.004                               # monthly covariance: volatilities of 6% to 8% a month
    stat_sd = 0.012                                                                               # stationary sd of the expected monthly return
    alpha = np.zeros((periods + 1, n))
    for t in range(1, periods + 1):
        alpha[t] = persistence * alpha[t - 1] + stat_sd * np.sqrt(1 - persistence ** 2) * rng.normal(size=n)
    noise = rng.multivariate_normal(np.zeros(n), S, size=periods)
    returns = alpha[:periods] + noise                                                             # the return of period t is earned by the holdings chosen at t
    return alpha[:periods], returns, S


def run_policy(policy, alpha, returns, S, gamma: float, Lam, kappa, x0=None) -> dict:
    """Average realised utility per period of a policy ``(t, x_prev) -> x`` : the return of the holdings less the risk charge and the trading costs; plus turnover."""
    T, n = returns.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    total, traded = 0.0, 0.0
    for t in range(T):
        new = policy(t, x)
        d = new - x
        total += new @ returns[t] - 0.5 * gamma * new @ S @ new - 0.5 * d @ Lam @ d - kappa * np.abs(d).sum()
        traded += np.abs(d).sum()
        x = new
    return {"utility per month": total / T, "turnover per month": traded / T}


def _paired(rows: dict, base: str, scale: float = 1e4) -> pd.DataFrame:
    """Mean utility per month (basis points) of each policy over the trials, and its difference from ``base`` on the same paths with the standard error of that difference."""
    base_v = np.asarray(rows[base])
    out = {}
    for name, v in rows.items():
        v = np.asarray(v)
        d = v - base_v
        out[name] = {"utility (bp/month)": scale * v.mean(), "minus the baseline": scale * d.mean(), "s.e. of the difference": scale * d.std(ddof=1) / np.sqrt(len(d)) if name != base else 0.0}
    return pd.DataFrame(out).T


def policy_comparison(trials: int = 10, n: int = 6, periods: int = 240, gamma: float = 5.0, quad: float = 20.0, persistence: float = 0.9) -> pd.DataFrame:
    """The same forecast on the same paths, four ways: ignore costs and hold the Markowitz portfolio, account for the cost one period at a time, plan with the closed-form dynamic programme (the forecast
    fades at the known rate), and the same plan with a short horizon. Costs are quadratic, so the dynamic programme is exact and the unconstrained optimum."""
    rows = {"ignore costs (hold the Markowitz portfolio)": [], "one period at a time, with costs": [], "multi-period: plan 3 months, rest summarised": [], "multi-period: full horizon (closed form)": []}
    traded = {k: [] for k in rows}
    for seed in range(trials):
        alpha, returns, S = predictable_world(n, periods, persistence, seed)
        Lam = mp.cost_matrix(S, quad, "risk")
        paths = {t: mp.decaying_path(alpha[t], persistence, 40) for t in range(periods)}
        policies = (lambda t, x: np.linalg.solve(gamma * S, alpha[t]),
                    lambda t, x: mp.lq_solve(alpha[t][None, :], S, gamma, Lam, x).holdings[0],
                    lambda t, x: mp.mpc_step(paths[t], S, gamma, x, Lam, None, 1.0, plan_horizon=3)[0],
                    lambda t, x: mp.lq_solve(paths[t], S, gamma, Lam, x).holdings[0])
        for name, pol in zip(rows, policies):
            out = run_policy(pol, alpha, returns, S, gamma, Lam, 0.0)
            rows[name].append(out["utility per month"])
            traded[name].append(out["turnover per month"])
    table = _paired(rows, "one period at a time, with costs")
    table["turnover (x NAV/month)"] = pd.Series({k: float(np.mean(v)) for k, v in traded.items()})
    return table


def constrained_comparison(trials: int = 6, n: int = 6, periods: int = 120, gamma: float = 5.0, quad: float = 20.0, persistence: float = 0.9) -> pd.DataFrame:
    """The same four-asset-style comparison under a book with limits (each position within 30%, gross exposure at most 100%, net within 30%), solved by model-predictive control, and the unconstrained
    optimum as an upper bound that the limits keep out of reach."""
    rows = {"one period at a time (MPC, 1 month)": [], "MPC, 3 months, rest summarised": [], "MPC, 6 months, rest summarised": [], "upper bound: no limits (closed form)": []}
    traded = {k: [] for k in rows}
    limits = dict(lb=-0.3, ub=0.3, max_gross=1.0, net=(-0.3, 0.3))
    for seed in range(trials):
        alpha, returns, S = predictable_world(n, periods, persistence, seed)
        Lam = mp.cost_matrix(S, quad, "risk")
        paths = {t: mp.decaying_path(alpha[t], persistence, 40) for t in range(periods)}
        policies = (lambda t, x: mp.mpc_step(paths[t][:1], S, gamma, x, Lam, None, 1.0, terminal=False, **limits)[0],
                    lambda t, x: mp.mpc_step(paths[t], S, gamma, x, Lam, None, 1.0, plan_horizon=3, **limits)[0],
                    lambda t, x: mp.mpc_step(paths[t], S, gamma, x, Lam, None, 1.0, plan_horizon=6, **limits)[0],
                    lambda t, x: mp.lq_solve(paths[t], S, gamma, Lam, x).holdings[0])
        for name, pol in zip(rows, policies):
            out = run_policy(pol, alpha, returns, S, gamma, Lam, 0.0)
            rows[name].append(out["utility per month"])
            traded[name].append(out["turnover per month"])
    table = _paired(rows, "one period at a time (MPC, 1 month)")
    table["turnover (x NAV/month)"] = pd.Series({k: float(np.mean(v)) for k, v in traded.items()})
    return table


def trade_rate_table(gamma: float = 5.0, persistence_discount: float = 0.99) -> pd.DataFrame:
    """The long-run share of the gap to the aim that is closed each month, for one asset, by trading cost and risk aversion."""
    S = np.array([[0.004]])
    rows = {}
    for lam in (0.0005, 0.002, 0.01, 0.05, 0.25):
        rows[f"impact {lam}"] = {f"risk aversion {g}": mp.stationary_trade_rate(S, g, np.array([[lam]]), persistence_discount)[0][0, 0] for g in (2.0, 5.0, 10.0)}
    return pd.DataFrame(rows).T


def no_trade_width(gamma: float = 5.0) -> pd.DataFrame:
    """Width of the no-trade region around the aim for one asset, as a share of the aim, by proportional cost: a cost of 10 bp a unit against a forecast of 1% a month."""
    S = np.array([[0.004]])
    alpha = 0.01
    rows = {}
    for kappa in (0.0005, 0.001, 0.0025, 0.005):
        aim, _ = mp.no_trade_region([alpha], S, gamma, [kappa])
        rows[f"cost {kappa * 1e4:.0f} bp"] = {"aim": aim[0], "no-trade region": f"[{(alpha - kappa) / (gamma * S[0, 0]):.3f}, {(alpha + kappa) / (gamma * S[0, 0]):.3f}]", "width / aim": 2 * kappa / alpha}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------------------ 2. limit orders
def limit_order_table() -> pd.DataFrame:
    """Buy 100,000 shares in ten five-minute intervals: the best mix of limit and market orders against all-market and limit-then-market, as the urgency (the price drifting away) rises."""
    rows = {}
    for drift_bps in (0.0, 2.0, 5.0, 12.0):
        model = lo.LimitOrderModel(shares=100_000, intervals=10, half_spread_bps=3.0, impact_bps=8.0, drift_bps=drift_bps, adverse_selection_bps=1.0, fill_first=0.5, fill_decay=0.97, risk_bps=0.15, units=50)
        sol = lo.solve(model)
        sim = {name: lo.simulate(model, pol, paths=4000, seed=1) for name, pol in (("optimal mix", sol.policy), ("all market now", lo.all_market_policy(model)), ("limit orders, market at the end", lo.limit_then_market_policy(model)))}
        rows[f"drift {drift_bps:g} bp/interval"] = {**{f"{k} (bp)": v["cost_bps"] for k, v in sim.items()}, "share by limit orders": sim["optimal mix"]["limit_share"], "dp value (bp)": sol.cost_bps}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------------------ 3. smart order routing
def routing_table(rounds: int = 500, trials: int = 12) -> pd.DataFrame:
    """Route 1,000 shares a round across five venues whose hidden liquidity differs, learning only from what fills (a fill of the full amount says nothing about how much more was there)."""
    venues = rt.example_venues()
    policies = ("uniform", "best first-unit probability", "learned (Kaplan-Meier, optimistic)", "oracle (true distributions)")
    out = {p: [] for p in policies}
    for seed in range(trials):
        for p in policies:
            out[p].append(rt.simulate_routing(venues, rounds=rounds, shares=1000, policy=p, seed=seed)["expected_fill_rate_last_third"])
    return pd.DataFrame({p: {"expected fill rate, last third of the rounds": np.mean(v), "s.e.": np.std(v, ddof=1) / np.sqrt(trials)} for p, v in out.items()}).T


# ------------------------------------------------------------------------------------------------------------------ 4. the multi_period allocator in the pipeline
def pipeline_comparison(seeds: tuple = (1, 2, 3, 4), n_assets: int = 30, n_years: int = 9) -> pd.DataFrame:
    """The same forecast (a monthly cross-sectional regression on momentum and reversal, planted) through a 130/30 book that trades one month at a time and through the multi-period book, both told the same cost
    (10 bp a unit traded, which is what the engine charges), in several worlds."""
    from src.equity.synthetic import simulate_fundamental_world
    from src.framework import Pipeline, bundle_from_prices, load_library
    from src.utils.config import load_config

    load_library()
    config = load_config()
    books = (("one month at a time (constrained_long_short)", "constrained_long_short", {"book": "130_30", "cost_bps": 10.0}),
             ("planning 3 of 6 months (multi_period)", "multi_period", {"book": "130_30", "cost_bps": 10.0, "horizon": 6, "plan_horizon": 3}),
             ("planning 3 of 6 months, impact cost too", "multi_period", {"book": "130_30", "cost_bps": 10.0, "horizon": 6, "plan_horizon": 3, "impact": 0.5}))
    runs = {}
    for seed in seeds:
        world = simulate_fundamental_world(n_assets=n_assets, n_years=n_years, seed=seed, premia={"value": 0.0, "quality": 0.0, "investment": 0.0, "momentum": 0.008, "reversal": 0.006, "revision": 0.0})
        bundle = bundle_from_prices(world.prices, min_history=60, name="trading")
        for label, allocator, params in books:
            t0 = time.time()
            spec = {"name": label, "models": [{"name": "characteristic_regression"}], "allocation": {"allocator": allocator, "params": params}, "risk": {"mode": "none"}, "evaluation": {"causality": False}}
            r = Pipeline(spec, config, bundle).run(validate=False)
            w = r.weights.loc[r.start:]
            w = w[w.abs().sum(axis=1) > 1e-9]
            runs[(label, seed)] = {"net Sharpe": r.metrics["sharpe"], "gross exposure": w.abs().sum(axis=1).mean(), "turnover (x/yr)": r.metrics["ann_turnover"], "cost (bp/yr)": r.metrics["ann_cost_bps"],
                                   "years evaluated": len(r.window) / 252.0, "seconds": time.time() - t0}
    frame = pd.DataFrame(runs).T
    frame.index.names = ["book", "seed"]
    mean = frame.groupby(level="book", sort=False).mean()
    sharpe = frame["net Sharpe"].unstack("seed")
    base = sharpe.iloc[0]
    paired = sharpe.sub(base, axis=1)
    mean.insert(1, "minus one-month book", paired.mean(axis=1))
    mean.insert(2, "... s.e. (paired)", paired.std(axis=1) / np.sqrt(len(seeds)))
    return mean


def run() -> dict:
    return {"policy_comparison": policy_comparison(), "constrained_comparison": constrained_comparison(), "trade_rate": trade_rate_table(), "no_trade_region": no_trade_width(), "limit_orders": limit_order_table(),
            "routing": routing_table(), "pipeline": pipeline_comparison()}


def main() -> None:
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 200
    t0 = time.time()
    for name, table in run().items():
        print(f"== {name} ==")
        print(table.to_string(), "\n")
    print(f"{time.time() - t0:.0f} seconds")


if __name__ == "__main__":
    main()
