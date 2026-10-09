"""Payments out of a portfolio: how much to spend each year, and what that does to the chance of running out.

A retiree, an endowment or a foundation takes money out every year. The rule that sets the amount matters as much as the portfolio: a fixed real amount is predictable and fragile (it keeps being
paid from a shrinking pot), a fixed share of the pot never runs out and varies a great deal, and the rules in between smooth one into the other.

    fixed_real        the first year's spending (``rate`` of the starting value) raised with inflation every year, whatever the portfolio does (the "4% rule")
    percent_of_nav    ``rate`` of the portfolio's value at the start of each year: never ruined, but spending follows the market
    endowment         the Yale / Tobin rule: ``smoothing`` of last year's spending raised with inflation plus ``1 - smoothing`` of ``rate`` times the current value
    guardrails        Guyton and Klinger: last year's spending raised with inflation, cut by ``adjust`` if that is more than ``upper`` times the initial withdrawal rate of the current
                      value, raised by ``adjust`` if it is less than ``lower`` times that rate

:func:`simulate_spending` runs a rule over many future markets made by resampling the portfolio's own daily history in blocks (the stationary bootstrap, so volatility clusters and short-run
dependence are kept) and reports the chance of running out, the spread of the final wealth and of the spending that was possible; :func:`sustainable_rate` finds the highest starting rate whose chance of
ruin stays under a limit. Spending is paid at the start of each year, and amounts are nominal unless called real.
"""

from __future__ import annotations

import numpy as np

RULES = ("fixed_real", "percent_of_nav", "endowment", "guardrails")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def bootstrap_annual_returns(returns, years: int, paths: int, periods_per_year: int = 252, mean_block: float = 21.0, seed=0) -> np.ndarray:
    """``paths`` simulated sequences of ``years`` annual returns made by compounding resampled periods: blocks of the history of average length ``mean_block`` joined end to end (the stationary
    bootstrap), wrapping round the end of the history."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 2 * mean_block:
        raise ValueError("the return history is too short for the block length")
    rng = np.random.default_rng(seed)
    steps = years * periods_per_year
    restart = rng.random((paths, steps)) < 1.0 / max(mean_block, 1.0)
    restart[:, 0] = True
    where = np.maximum.accumulate(np.where(restart, np.arange(steps)[None, :], 0), axis=1)
    start = np.take_along_axis(rng.integers(0, len(r), size=(paths, steps)), where, axis=1)
    picked = r[(start + np.arange(steps)[None, :] - where) % len(r)]
    return (1.0 + picked).reshape(paths, years, periods_per_year).prod(axis=2) - 1.0


def simulate_spending(returns=None, *, annual_returns: np.ndarray | None = None, rule: str = "fixed_real", initial: float = 1_000_000.0, rate: float = 0.04, years: int = 30, paths: int = 2000,
                      inflation: float = 0.02, periods_per_year: int = 252, mean_block: float = 21.0, smoothing: float = 0.7, upper: float = 1.2, lower: float = 0.8, adjust: float = 0.10,
                      seed: int = 0) -> dict:
    """Run a spending rule over simulated markets (from ``returns``, a history of the portfolio's periodic returns, or given directly as ``annual_returns``, a paths-by-years array).

    Returns ``ruin_probability`` (the share of paths in which the portfolio could not pay the year's spending in full), ``years_funded`` (the average number of years paid in full), the quantiles of
    final wealth and of average real spending, ``spending_cut_probability`` (the share of paths where real spending fell below 80% of the first year's at some point) and year-by-year ``bands``
    of the quantiles of wealth and of real spending, for charting. Wealth and spending are also given in real terms (deflated by ``inflation``)."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    if initial <= 0 or rate <= 0 or years < 1 or paths < 1 or not 0 <= smoothing <= 1 or not 0 < lower < 1 < upper or not 0 < adjust < 1:
        raise ValueError("initial, rate > 0; years, paths >= 1; 0 <= smoothing <= 1; 0 < lower < 1 < upper; 0 < adjust < 1")
    if annual_returns is None:
        if returns is None:
            raise ValueError("give returns or annual_returns")
        annual_returns = bootstrap_annual_returns(returns, years, paths, periods_per_year, mean_block, seed)
    R = np.asarray(annual_returns, float)
    if R.ndim != 2 or R.shape[1] < years:
        raise ValueError("annual_returns must be a paths-by-years array covering the horizon")
    paths = R.shape[0]
    nav = np.full(paths, float(initial))
    spend = np.zeros(paths)
    ruined = np.zeros(paths, bool)
    funded = np.zeros(paths)
    wealth = np.zeros((paths, years + 1))
    wealth[:, 0] = nav
    paid = np.zeros((paths, years))
    for y in range(years):
        if rule == "fixed_real":
            want = np.full(paths, rate * initial * (1.0 + inflation) ** y)
        elif rule == "percent_of_nav":
            want = rate * nav
        elif rule == "endowment":
            want = np.full(paths, rate * initial) if y == 0 else smoothing * spend * (1.0 + inflation) + (1.0 - smoothing) * rate * nav
        else:
            if y == 0:
                want = np.full(paths, rate * initial)
            else:
                want = spend * (1.0 + inflation)
                with np.errstate(divide="ignore", invalid="ignore"):
                    current = np.where(nav > 0, want / nav, np.inf)
                want = np.where(current > upper * rate, want * (1.0 - adjust), np.where(current < lower * rate, want * (1.0 + adjust), want))
        spend = want
        pay = np.minimum(want, nav)
        ruined |= want > nav + 1e-9
        funded += (want <= nav + 1e-9)
        paid[:, y] = pay
        nav = np.maximum(nav - pay, 0.0) * (1.0 + R[:, y])
        wealth[:, y + 1] = nav
    deflator = (1.0 + inflation) ** np.arange(years + 1)
    real_wealth = wealth / deflator[None, :]
    real_paid = paid / deflator[None, :years]
    first_real = real_paid[:, 0]
    q = lambda a, axis=0: np.quantile(a, QUANTILES, axis=axis)                                              # noqa: E731
    cut = (real_paid < 0.8 * first_real[:, None] - 1e-9).any(axis=1)
    return {"rule": rule, "ruin_probability": float(ruined.mean()), "years_funded": float(funded.mean()), "final_wealth": dict(zip(QUANTILES, q(wealth[:, -1]).tolist())),
            "final_real_wealth": dict(zip(QUANTILES, q(real_wealth[:, -1]).tolist())), "real_spending": dict(zip(QUANTILES, q(real_paid.mean(axis=1)).tolist())),
            "mean_real_spending": float(real_paid.mean()), "spending_cut_probability": float(cut.mean()),
            "bands": {"wealth": q(real_wealth, axis=0).tolist(), "spending": q(real_paid, axis=0).tolist(), "quantiles": list(QUANTILES), "years": list(range(years + 1))}}


def sustainable_rate(returns=None, *, annual_returns: np.ndarray | None = None, target_ruin: float = 0.05, rule: str = "fixed_real", lo: float = 0.005, hi: float = 0.15, tol: float = 1e-4, **kwargs) -> float:
    """The highest starting spending rate whose probability of ruin is at most ``target_ruin``, found by bisection over the same simulated markets (a higher rate never ruins fewer paths).

    Returns ``lo`` if even the lowest rate is too risky and ``hi`` if the highest is safe enough."""
    if not 0 < target_ruin < 1 or lo >= hi:
        raise ValueError("0 < target_ruin < 1 and lo < hi")
    if annual_returns is None:
        years, paths = int(kwargs.get("years", 30)), int(kwargs.get("paths", 2000))
        annual_returns = bootstrap_annual_returns(returns, years, paths, kwargs.get("periods_per_year", 252), kwargs.get("mean_block", 21.0), kwargs.get("seed", 0))

    def ruin(r: float) -> float:
        return simulate_spending(annual_returns=annual_returns, rule=rule, rate=r, **{k: v for k, v in kwargs.items() if k in ("initial", "years", "inflation", "smoothing", "upper", "lower", "adjust")})["ruin_probability"]

    if ruin(lo) > target_ruin:
        return lo
    if ruin(hi) <= target_ruin:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ruin(mid) <= target_ruin else (lo, mid)
    return lo
