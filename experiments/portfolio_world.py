"""Portfolio theory, constrained books, CVaR and QUBO portfolios and factor models on simulated worlds with a known truth, and what they find.

Everything here is simulated, so the question is whether the tools recover what was planted and how the methods compare with one another, not whether anything would make money in a real market.
Nothing is tuned; sizes and seeds are fixed.

    python -m experiments.portfolio_world          # about four minutes
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from src.equity import alpha_model as am
from src.equity import qubo as qb
from src.equity import theory as th
from src.equity.cvar_portfolio import mean_variance_cvar
from src.equity.synthetic import simulate_fundamental_world
from src.framework import MODELS, Pipeline, bundle_from_prices, load_library
from src.strategies import factor_models as fm
from src.utils.config import load_config
from src.utils.dates import rebalance_dates


# ------------------------------------------------------------------------------------------------------------------ 1. mean-variance and estimation error
def estimation_error(n_assets: int = 12, months: int = 60, trials: int = 300, seed: int = 0) -> pd.DataFrame:
    """The tangency portfolio from a few years of data against equal weights, minimum variance and a shrunk version, all judged on the true means and covariance."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n_assets, 3))
    cov = (F @ F.T / 3 * 0.02 + np.diag(rng.uniform(0.01, 0.03, n_assets))) / 12.0
    mu = rng.uniform(0.02, 0.10, n_assets) / 12.0
    true_sharpe = lambda w: float((w @ mu) / np.sqrt(w @ cov @ w) * np.sqrt(12))
    best = th.tangency_portfolio(pd.Series(mu), pd.DataFrame(cov), 0.0)
    rows = {k: [] for k in ("equal weight", "minimum variance", "tangency (estimated)", "tangency, long-only (estimated)", "tangency with means shrunk to the grand mean")}
    L = np.linalg.cholesky(cov)
    for _ in range(trials):
        sample = mu + (L @ rng.normal(size=(n_assets, months))).T
        m_hat, c_hat = sample.mean(axis=0), np.cov(sample, rowvar=False)
        rows["equal weight"].append(true_sharpe(np.full(n_assets, 1.0 / n_assets)))
        rows["minimum variance"].append(true_sharpe(th.global_minimum_variance(pd.Series(m_hat), pd.DataFrame(c_hat)).weights.to_numpy()))
        rows["tangency (estimated)"].append(true_sharpe(th.tangency_portfolio(pd.Series(m_hat), pd.DataFrame(c_hat), 0.0).weights.to_numpy()))
        try:
            rows["tangency, long-only (estimated)"].append(true_sharpe(th.tangency_portfolio(pd.Series(m_hat), pd.DataFrame(c_hat), 0.0, long_only=True).weights.to_numpy()))
        except ValueError:
            rows["tangency, long-only (estimated)"].append(np.nan)
        shrunk = 0.5 * m_hat + 0.5 * m_hat.mean()
        rows["tangency with means shrunk to the grand mean"].append(true_sharpe(th.tangency_portfolio(pd.Series(shrunk), pd.DataFrame(c_hat), 0.0).weights.to_numpy()))
    table = pd.DataFrame({k: pd.Series(v).describe()[["mean", "25%", "50%", "75%"]] for k, v in rows.items()}).T
    table.loc["the true tangency portfolio"] = [best.sharpe * np.sqrt(12)] * 4
    return table


# ------------------------------------------------------------------------------------------------------------------ 2. CAPM and APT
def pricing_tests(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = {}
    for label, zero_beta in (("standard CAPM world", 0.0), ("zero-beta world (intercept 0.4% a month)", 0.004)):
        n, T = 60, 2400
        beta = rng.uniform(0.3, 1.7, n)
        market = pd.Series(0.005 + 0.04 * rng.normal(size=T), name="mkt")
        r = pd.DataFrame(zero_beta * (1 - beta) + np.outer(market, beta) + 0.03 * rng.normal(size=(T, n)), columns=[f"S{i}" for i in range(n)])
        sml = th.security_market_line(r, market)
        rows[label] = {"intercept": sml.intercept, "t intercept": sml.t_intercept, "slope": sml.slope, "t slope": sml.t_slope, "market premium": sml.market_premium, "cross-sectional R2": sml.r2}
    return pd.DataFrame(rows).T


def apt_arbitrage(seed: int = 1) -> pd.DataFrame:
    """Plant an APT with two factors and mispricing in twenty names, estimate it on one sample and trade the pricing errors out of sample, with zero net investment and zero factor exposure."""
    rng = np.random.default_rng(seed)
    n, T, k = 50, 3000, 2
    B = rng.normal(size=(n, k))
    lam = np.array([0.003, -0.001])
    alpha = np.zeros(n)
    alpha[:20] = 0.0015 * rng.choice([-1, 1], 20)
    names = [f"S{i}" for i in range(n)]

    def draw(T_):
        f = lam + 0.03 * rng.normal(size=(T_, k))
        return pd.DataFrame(f, columns=["f1", "f2"]), pd.DataFrame(alpha + f @ B.T + 0.03 * rng.normal(size=(T_, n)), columns=names)

    (train_f, train_r), (_, test_r) = draw(T), draw(T)
    fit = th.apt_two_pass(train_r, train_f)
    cov = pd.DataFrame(np.cov(train_r.to_numpy(), rowvar=False), index=names, columns=names)
    w = th.apt_arbitrage_portfolio(fit["alpha"], fit["betas"].to_numpy(), cov, gross=2.0)
    out_sample = test_r.to_numpy() @ w.to_numpy()
    return pd.DataFrame({"value": {
        "factor premia recovered (true 0.003, -0.001)": float(fit["lambda"].iloc[0]),
        "second premium": float(fit["lambda"].iloc[1]),
        "pricing-error chi2 (df %d)" % fit["alpha_df"]: fit["alpha_chi2"],
        "net investment": float(w.sum()),
        "largest factor exposure": float(np.abs(fit["betas"].to_numpy().T @ w.to_numpy()).max()),
        "out-of-sample mean per month": float(out_sample.mean()),
        "out-of-sample t": float(out_sample.mean() / out_sample.std() * np.sqrt(len(out_sample)))}})


# ------------------------------------------------------------------------------------------------------------------ 3. constrained books
BOOKS = (("long only", {"book": "long_only"}), ("130/30", {"book": "130_30"}), ("130/30, sector neutral", {"book": "130_30", "sector_neutral": True}),
         ("market neutral, beta neutral", {"book": "market_neutral", "beta_neutral": True}), ("130/30, at most 30% turnover a month", {"book": "130_30", "max_turnover": 0.3}),
         ("130/30, 20 bp trading cost in the optimiser", {"book": "130_30", "cost_bps": 20.0}))


def constrained_books(n_assets: int = 40, n_years: int = 12, seeds: tuple = (1, 2, 3, 4)) -> pd.DataFrame:
    """The same alpha (a monthly cross-sectional regression on momentum and reversal, planted in the world) through six books, in several simulated worlds. The exposures, beta, turnover and cost are
    structural and barely move from world to world; the Sharpe ratios do, so they come with their standard error across worlds."""
    load_library()
    config = load_config()
    model = [{"name": "characteristic_regression"}]
    runs = {}
    for seed in seeds:
        world = simulate_fundamental_world(n_assets=n_assets, n_years=n_years, seed=seed, premia={"value": 0.0, "quality": 0.0, "investment": 0.0, "momentum": 0.008, "reversal": 0.006, "revision": 0.0})
        groups = {a: f"S{i % 5}" for i, a in enumerate(world.prices.columns)}
        bundle = bundle_from_prices(world.prices, min_history=60, name="books", asset_class=groups)
        market = bundle.returns.mean(axis=1)
        for label, params in BOOKS:
            t0 = time.time()
            spec = {"name": label, "models": model, "allocation": {"allocator": "constrained_long_short", "params": params}, "risk": {"mode": "none"}, "evaluation": {"causality": False}}
            r = Pipeline(spec, config, bundle).run(validate=False)
            w, book, m = r.weights.loc[r.start:], r.gross_returns.loc[r.start:], market.loc[r.start:]
            w = w[w.abs().sum(axis=1) > 1e-9]                                   # the first days of the window are flat, before the first monthly rebalance
            runs[(label, seed)] = {"net Sharpe": r.metrics["sharpe"], "gross exposure": w.abs().sum(axis=1).mean(), "net exposure": w.sum(axis=1).mean(), "turnover (x/yr)": r.metrics["ann_turnover"],
                                   "cost (bp/yr)": r.metrics["ann_cost_bps"], "beta to equal weight": float(book.cov(m) / m.var()), "years evaluated": len(r.window) / 252.0, "seconds": time.time() - t0}
    frame = pd.DataFrame(runs).T
    frame.index.names = ["book", "seed"]
    mean = frame.groupby(level="book", sort=False).mean()
    sharpe = frame["net Sharpe"].unstack("seed")
    paired = sharpe.sub(sharpe.loc["long only"], axis=1)                            # every book against long only, in the same world
    mean.insert(1, "Sharpe s.e.", sharpe.std(axis=1) / np.sqrt(len(seeds)))
    mean.insert(2, "minus long only", paired.mean(axis=1))
    mean.insert(3, "... s.e. (paired)", paired.std(axis=1) / np.sqrt(len(seeds)))
    return mean


# ------------------------------------------------------------------------------------------------------------------ 4. CVaR
def cvar_frontier(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    S, n = 800, 6
    base = rng.normal(size=(S, n)) * np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.05])
    crash = rng.random((S, n)) < 0.03
    R = base - crash * np.array([0.0, 0.0, 0.02, 0.05, 0.10, 0.15]) + np.array([0.004, 0.005, 0.006, 0.0075, 0.0095, 0.012])
    mu = np.array([0.004, 0.005, 0.006, 0.0075, 0.0095, 0.012])
    rows = {}
    free = mean_variance_cvar(mu, R, None, risk_aversion=2.0, fully_invested=False)
    rows["no limit"] = {"CVaR limit": np.nan, "CVaR 95%": free.cvar, "expected return": free.expected_return, "volatility": free.volatility, "cash": free.cash, "weight on the fat-tailed asset": free.weights.iloc[-1]}
    for limit in (0.08, 0.05, 0.03, 0.02):
        r = mean_variance_cvar(mu, R, limit, risk_aversion=2.0, fully_invested=False)
        rows[f"limit {limit:.0%}"] = {"CVaR limit": limit, "CVaR 95%": r.cvar, "expected return": r.expected_return, "volatility": r.volatility, "cash": r.cash, "weight on the fat-tailed asset": r.weights.iloc[-1]}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------------------ 5. QUBO
def annealing(trials: int = 12) -> pd.DataFrame:
    hits, gaps = 0, []
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        F = rng.normal(size=(14, 3))
        cov = F @ F.T / 3 * 0.04 + np.diag(rng.uniform(0.01, 0.03, 14))
        mu = rng.uniform(0.02, 0.12, 14)
        mask, obj = qb.anneal_selection(mu, cov, 4, 6.0, sweeps=400, restarts=32, seed=seed)
        import itertools
        best = min(qb.selection_objective(mu, cov, np.isin(np.arange(14), c), 6.0) for c in itertools.combinations(range(14), 4))
        hits += int(obj <= best + 1e-12)
        gaps.append(obj - best)
    timing = {}
    for n, k in ((40, 10), (80, 15)):
        rng = np.random.default_rng(n)
        F = rng.normal(size=(n, 3))
        cov = F @ F.T / 3 * 0.04 + np.diag(rng.uniform(0.01, 0.03, n))
        mu = rng.uniform(0.02, 0.12, n)
        t0 = time.time()
        mask, obj = qb.anneal_selection(mu, cov, k, 6.0, sweeps=300, restarts=16, seed=0)
        sel = []
        for _ in range(k):
            sel.append(min((i for i in range(n) if i not in sel), key=lambda i: qb.selection_objective(mu, cov, np.isin(np.arange(n), sel + [i]), 6.0)))
        timing[f"best {k} of {n}"] = {"annealing objective": obj, "greedy objective": qb.selection_objective(mu, cov, np.isin(np.arange(n), sel), 6.0), "seconds": time.time() - t0}
    out = pd.DataFrame(timing).T
    out.loc["best 4 of 14 vs exhaustive search"] = [float(np.mean(gaps)), np.nan, np.nan]
    out.loc["... found the optimum in"] = [hits / trials, np.nan, np.nan]
    return out


# ------------------------------------------------------------------------------------------------------------------ 6. factor models
def factor_model_table(seed: int = 1) -> pd.DataFrame:
    load_library()
    world = simulate_fundamental_world(n_assets=60, n_years=12, seed=seed, premia={"value": 0.0, "quality": 0.0, "investment": 0.0, "momentum": 0.008, "reversal": 0.006, "revision": 0.0})
    bundle = bundle_from_prices(world.prices, min_history=60, name="characteristics")
    grid = rebalance_dates(bundle.index, "monthly")
    px = bundle.prices.loc[grid]
    fwd = px.shift(-1) / px - 1.0

    def summary(score):
        ic = am.information_coefficients({"x": score.loc[grid]}, fwd, "spearman", 15)["x"].iloc[36:]
        return {"mean IC": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(len(ic))}
    rows = {"characteristic_regression": summary(MODELS.create("characteristic_regression").score(bundle)), "apt_alpha (3 components)": summary(MODELS.create("apt_alpha").score(bundle))}
    for kind in fm.KINDS:
        t0 = time.time()
        rows[f"ml_factor_model: {kind}"] = {**summary(MODELS.create("ml_factor_model", kind=kind).score(bundle)), "seconds": time.time() - t0}
    return pd.DataFrame(rows).T


def _monthly_ic(bundle, score, skip: int) -> dict:
    grid = rebalance_dates(bundle.index, "monthly")
    px = bundle.prices.loc[grid]
    ic = am.information_coefficients({"x": score.loc[grid]}, px.shift(-1) / px - 1.0, "spearman", 15)["x"].iloc[skip:]
    return {"mean IC": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(len(ic)), "months": len(ic)}


def apt_components(seeds: tuple = tuple(range(6)), components: tuple = (1, 2, 3, 5)) -> pd.DataFrame:
    """The statistical APT on worlds with three planted common factors (each with a positive premium) and a planted mispricing, by the number of components removed, averaged over worlds. Removing too few
    leaves factor premia in the score: they predict returns, so the mean IC is higher, but they are not alpha and make the IC swing with the factors."""
    load_library()
    rows = {c: [] for c in components}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        T, n, k = 3000, 40, 3
        idx = pd.bdate_range("2004-01-05", periods=T)
        B = rng.normal(size=(n, k))
        f = 0.0004 + 0.008 * rng.normal(size=(T, k))
        alpha = 0.0006 * rng.normal(size=n)
        r = pd.DataFrame(alpha + f @ B.T + 0.01 * rng.normal(size=(T, n)), index=idx, columns=[f"S{i:02d}" for i in range(n)])
        bundle = bundle_from_prices(100 * (1 + r).cumprod(), min_history=60, name="mispricing")
        for c in components:
            rows[c].append(_monthly_ic(bundle, MODELS.create("apt_alpha", factors=c).score(bundle), 24))
    table = pd.DataFrame({f"{c} component{'s' if c > 1 else ''} removed": pd.DataFrame(v).mean() for c, v in rows.items()}).T
    table["worlds"] = len(seeds)
    return table[["mean IC", "t", "months"]].assign(worlds=len(seeds))


def macro_worlds(seed: int = 1) -> pd.DataFrame:
    """The macro factor model on a world where exposure to two macroeconomic changes is priced, and on the same world with nothing priced."""
    load_library()
    T, n = 3600, 40
    idx = pd.bdate_range("2004-01-05", periods=T)
    rows = {}
    for label, premium in (("exposures priced (0.07% a day per unit)", 0.0007), ("nothing priced (control)", 0.0)):
        rng = np.random.default_rng(seed)
        g = rng.normal(size=(T, 2))
        L = rng.normal(size=(n, 2))
        r = L @ np.full(2, premium) + 0.004 * g @ L.T + 0.012 * rng.normal(size=(T, n))
        macro = pd.DataFrame({"DGS10": 3 + np.cumsum(g[:, 0] * 0.03), "T10Y3M": 1 + np.cumsum(g[:, 1] * 0.03)}, index=idx)
        world = bundle_from_prices(100 * (1 + pd.DataFrame(r, index=idx, columns=[f"S{i:02d}" for i in range(n)])).cumprod(), macro=macro, min_history=60, name="macro")
        rows[label] = _monthly_ic(world, MODELS.create("macro_factor_model", series="DGS10,T10Y3M", innovations=False).score(world), 48)
    return pd.DataFrame(rows).T


def nonlinear_targets(seed: int = 2) -> pd.DataFrame:
    """Out-of-sample R-squared of each learner when the return depends on a feature through a U shape, and through a U shape plus an interaction: what only the nonlinear learners can see."""
    rng = np.random.default_rng(seed)
    X, Xt = rng.normal(size=(4000, 6)), rng.normal(size=(2000, 6))
    r2 = lambda yt, p: 1 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    rows = {}
    for label, target in (("U shape in one feature", lambda Z: Z[:, 0] ** 2 - 1.0), ("U shape plus an interaction", lambda Z: Z[:, 0] ** 2 - 1.0 + Z[:, 0] * Z[:, 1])):
        y = target(X) + 0.3 * rng.normal(size=len(X))
        rows[label] = {k: r2(target(Xt), fm.fit_learner(k, X, y, depth=6, trees=60)[0](Xt)) for k in ("ols", "lasso", "pls", "cart", "forest", "mlp")}
    return pd.DataFrame(rows).T


def run() -> dict:
    warnings.simplefilter("ignore")
    return {"estimation_error": estimation_error(), "pricing": pricing_tests(), "apt": apt_arbitrage(), "books": constrained_books(), "cvar": cvar_frontier(), "annealing": annealing(), "factor_models": factor_model_table(), "apt_components": apt_components(), "macro_worlds": macro_worlds(), "nonlinear_targets": nonlinear_targets()}


def main() -> None:
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 200
    for name, table in run().items():
        print(f"== {name} ==")
        print(table.to_string(), "\n")


if __name__ == "__main__":
    main()
