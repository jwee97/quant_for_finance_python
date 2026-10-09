"""Portfolio theory and asset pricing in closed form: the efficient frontier, the tangency portfolio and the capital market line, CAPM (standard and zero-beta) and APT.

**Modern portfolio theory.** With expected returns ``mu`` and covariance ``Sigma`` and no constraints, everything is linear algebra. Let ``A = 1'Sigma^-1 1``, ``B = 1'Sigma^-1 mu``, ``C = mu'Sigma^-1 mu`` and ``D = AC - B^2``.
The minimum-variance portfolio is ``Sigma^-1 1 / A``; the frontier portfolio with expected return ``m`` is ``[(C - B m) Sigma^-1 1 + (A m - B) Sigma^-1 mu] / D`` and has variance ``(A m^2 - 2 B m + C) / D``
(Merton 1972). Any two frontier portfolios span the frontier. With a risk-free rate ``rf`` the best risky portfolio is the tangency portfolio ``Sigma^-1 (mu - rf)``, normalised; the capital market line
``E[R] = rf + Sharpe * sd`` joins ``rf`` to it, and an investor with risk aversion ``gamma`` puts ``(mu_T - rf) / (gamma sd_T^2)`` of wealth in it and the rest in the risk-free asset (more than all of it means borrowing).

**CAPM.** If everyone holds the tangency portfolio, it is the market and ``E[R_i] - rf = beta_i (E[R_m] - rf)``. :func:`capm_regression` estimates alpha and beta of each asset; :func:`security_market_line` is the
cross-sectional test of the pricing line (a two-pass regression with Shanken's correction): the standard CAPM says its intercept is the risk-free rate and its slope the market premium, Black's (1972)
zero-beta CAPM, which holds without a risk-free asset, lets the intercept be the expected return of the *zero-beta portfolio*, the minimum-variance portfolio uncorrelated with the market
(:func:`zero_beta_portfolio`).

**APT.** Ross's arbitrage pricing theory needs no market portfolio: if returns follow a factor model ``R = rf + B f + e`` and no zero-cost, zero-risk portfolio earns anything, then ``E[R] = rf + B lambda``. :func:`apt_two_pass`
estimates the loadings and the premia, the pricing errors are the mispricings, and :func:`apt_arbitrage_portfolio` builds the cheapest portfolio that is long the underpriced, short the overpriced, with
zero net investment and zero exposure to every factor, which is how a mispricing would be traded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..stats.panel import fama_macbeth_two_pass
from ..stats.regression import newey_west_lag
from .qp import solve_qp


# ------------------------------------------------------------------------------------------------------------------ modern portfolio theory
@dataclass
class Portfolio:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


def _as_arrays(mu, cov):
    names = list(mu.index) if isinstance(mu, pd.Series) else (list(cov.index) if isinstance(cov, pd.DataFrame) else list(range(len(np.asarray(mu)))))
    m, S = np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)
    if S.shape != (len(m), len(m)) or not (np.isfinite(m).all() and np.isfinite(S).all()):
        raise ValueError("mu and cov must be finite and conformable")
    return names, m, 0.5 * (S + S.T)


def _portfolio(names, w, m, S, rf: float = 0.0) -> Portfolio:
    ret, vol = float(w @ m), float(np.sqrt(max(w @ S @ w, 0.0)))
    return Portfolio(pd.Series(w, index=names), ret, vol, (ret - rf) / vol if vol > 0 else float("nan"))


def frontier_constants(mu, cov) -> dict:
    """``A, B, C, D`` of the frontier and the two vectors ``Sigma^-1 1`` and ``Sigma^-1 mu`` they come from."""
    _, m, S = _as_arrays(mu, cov)
    one = np.ones(len(m))
    s1, sm = np.linalg.solve(S, one), np.linalg.solve(S, m)
    A, B, C = float(one @ s1), float(one @ sm), float(m @ sm)
    return {"A": A, "B": B, "C": C, "D": A * C - B * B, "inv_one": s1, "inv_mu": sm}


def global_minimum_variance(mu, cov) -> Portfolio:
    names, m, S = _as_arrays(mu, cov)
    k = frontier_constants(mu, cov)
    return _portfolio(names, k["inv_one"] / k["A"], m, S)


def frontier_portfolio(mu, cov, target_return: float) -> Portfolio:
    """The minimum-variance portfolio (shorts allowed, fully invested) with expected return ``target_return``."""
    names, m, S = _as_arrays(mu, cov)
    k = frontier_constants(mu, cov)
    if k["D"] <= 1e-14:
        raise ValueError("expected returns are all equal (or the covariance is singular): there is no frontier")
    w = ((k["C"] - k["B"] * target_return) * k["inv_one"] + (k["A"] * target_return - k["B"]) * k["inv_mu"]) / k["D"]
    return _portfolio(names, w, m, S)


def frontier_variance(mu, cov, target_return) -> np.ndarray:
    """The frontier's variance at each expected return, from the closed form ``(A m^2 - 2 B m + C) / D``."""
    k = frontier_constants(mu, cov)
    t = np.asarray(target_return, dtype=float)
    return (k["A"] * t ** 2 - 2.0 * k["B"] * t + k["C"]) / k["D"]


def frontier_weights_between(first: Portfolio, second: Portfolio, share: float) -> pd.Series:
    """Two-fund separation: a mix of two frontier portfolios is on the frontier."""
    return share * first.weights + (1.0 - share) * second.weights


def tangency_portfolio(mu, cov, risk_free: float = 0.0, long_only: bool = False) -> Portfolio:
    """The portfolio with the highest Sharpe ratio: ``Sigma^-1 (mu - rf)`` normalised to sum to one. With ``long_only`` the same problem under ``w >= 0``, solved as the quadratic programme
    ``min y'Sigma y  s.t.  (mu - rf)'y = 1, y >= 0``, ``w = y / sum(y)``."""
    names, m, S = _as_arrays(mu, cov)
    excess = m - risk_free
    if long_only:
        if not (excess > 1e-12).any():
            raise ValueError("no asset has an expected return above the risk-free rate: there is no long-only tangency portfolio")
        r = solve_qp(S, np.zeros(len(m)), A=excess[None, :], l=[1.0], u=[1.0], lb=0.0)
        if not r.ok:
            raise ValueError(f"the long-only tangency problem did not solve ({r.status})")
        y = np.clip(r.x, 0.0, None)
        w = y / y.sum()
    else:
        raw = np.linalg.solve(S, excess)
        if abs(raw.sum()) < 1e-12:
            raise ValueError("the tangency portfolio does not exist: the weights of Sigma^-1 (mu - rf) sum to zero")
        w = raw / raw.sum()
    return _portfolio(names, w, m, S, risk_free)


def capital_market_line(mu, cov, risk_free: float = 0.0, volatilities=None) -> dict:
    """The line ``E[R] = rf + Sharpe * sd`` through the risk-free asset and the tangency portfolio: its intercept, slope, the tangency portfolio and the expected return at each volatility asked for."""
    t = tangency_portfolio(mu, cov, risk_free)
    out = {"intercept": float(risk_free), "slope": t.sharpe, "tangency": t}
    if volatilities is not None:
        v = np.asarray(volatilities, dtype=float)
        out["expected_return"] = risk_free + t.sharpe * v
    return out


def cml_allocation(tangency: Portfolio, risk_free: float, risk_aversion: float) -> dict:
    """Mean-variance utility ``mu - gamma sigma^2 / 2`` over mixes of the risk-free asset and the tangency portfolio is maximised by holding ``y = (mu_T - rf) / (gamma sigma_T^2)`` of wealth in the
    tangency portfolio (above one is borrowing at the risk-free rate); returns that share, the weights it implies and the expected return and volatility of the mix."""
    if risk_aversion <= 0 or tangency.volatility <= 0:
        raise ValueError("risk_aversion and the tangency volatility must be positive")
    y = (tangency.expected_return - risk_free) / (risk_aversion * tangency.volatility ** 2)
    return {"risky_share": float(y), "cash": float(1.0 - y), "weights": tangency.weights * y, "expected_return": float(risk_free + y * (tangency.expected_return - risk_free)),
            "volatility": float(abs(y) * tangency.volatility)}


# ------------------------------------------------------------------------------------------------------------------ CAPM
def capm_regression(excess_returns: pd.DataFrame, market_excess: pd.Series, lags: int | None = None, periods_per_year: int = 1) -> pd.DataFrame:
    """Time-series regression of each asset's excess return on the market's: ``alpha``, ``beta``, their t-statistics (Newey-West with ``lags`` lags, the rule of thumb if None; ``lags=0`` for plain OLS),
    R-squared and the residual volatility. ``periods_per_year`` annualises alpha and volatility."""
    frame = pd.concat([excess_returns, market_excess.rename("__market__")], axis=1).dropna()
    y, m = frame[excess_returns.columns].to_numpy(), frame["__market__"].to_numpy()
    T = len(m)
    if T < 10:
        raise ValueError("at least ten observations are needed")
    X = np.column_stack([np.ones(T), m])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    lags = newey_west_lag(T) if lags is None else int(lags)
    XtXi = np.linalg.inv(X.T @ X)
    t_alpha, t_beta = np.empty(y.shape[1]), np.empty(y.shape[1])
    for j in range(y.shape[1]):
        u = X * resid[:, [j]]
        S = u.T @ u
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            G = u[k:].T @ u[:-k]
            S = S + w * (G + G.T)
        V = XtXi @ S @ XtXi
        t_alpha[j], t_beta[j] = coef[0, j] / np.sqrt(V[0, 0]), coef[1, j] / np.sqrt(V[1, 1])
    tss = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return pd.DataFrame({"alpha": coef[0] * periods_per_year, "beta": coef[1], "t_alpha": t_alpha, "t_beta": t_beta, "r2": 1.0 - (resid ** 2).sum(axis=0) / tss,
                         "resid_vol": resid.std(axis=0, ddof=2) * np.sqrt(periods_per_year)}, index=excess_returns.columns)


def capm_expected_returns(betas, risk_free: float, market_premium: float):
    """The CAPM's expected returns ``rf + beta * premium``."""
    return risk_free + np.asarray(betas, dtype=float) * market_premium if not isinstance(betas, pd.Series) else risk_free + betas * market_premium


@dataclass
class SMLResult:
    intercept: float           # the cross-sectional intercept: the zero-beta rate (standard CAPM: the risk-free rate, i.e. zero for excess returns)
    slope: float               # the price of beta
    t_intercept: float         # Shanken-corrected
    t_slope: float
    market_premium: float      # the mean of the market factor over the sample
    r2: float
    betas: pd.Series
    pricing_errors: pd.Series


def security_market_line(returns: pd.DataFrame, market: pd.Series, allow_zero_beta_rate: bool = True) -> SMLResult:
    """Two-pass test of the pricing line (Fama-MacBeth with Shanken's correction): betas from time-series regressions on the market, then the cross-section of mean returns on the betas.

    Pass ``excess`` returns to test the standard CAPM, whose intercept should be zero and whose slope the market's mean excess return. ``allow_zero_beta_rate`` keeps the intercept (Black's zero-beta
    version, in which it is the expected return of a portfolio with no market risk); ``False`` forces the line through the origin."""
    out = fama_macbeth_two_pass(returns, market.to_frame(market.name or "market"), add_constant=allow_zero_beta_rate)
    lam, t = out["lambda"], out["t_shanken"]
    name = market.name or "market"
    return SMLResult(float(lam["const"]) if allow_zero_beta_rate else 0.0, float(lam[name]), float(t["const"]) if allow_zero_beta_rate else float("nan"), float(t[name]), float(market.mean()),
                     float(out["cross_sectional_r2"]), out["betas"][name], out["alpha"])


def zero_beta_portfolio(cov, benchmark, mu=None) -> Portfolio:
    """The minimum-variance fully invested portfolio uncorrelated with ``benchmark`` (weights): minimise ``w'Sigma w`` subject to ``1'w = 1`` and ``w'Sigma b = 0``. Black's zero-beta CAPM says the
    expected return of this portfolio is the intercept of the pricing line. With ``mu`` the portfolio reports its expected return."""
    names = list(cov.index) if isinstance(cov, pd.DataFrame) else list(range(len(cov)))
    S = np.asarray(cov, dtype=float)
    S = 0.5 * (S + S.T)
    b = np.asarray(benchmark, dtype=float)
    n = len(S)
    C = np.vstack([np.ones(n), S @ b])
    SiCt = np.linalg.solve(S, C.T)
    w = SiCt @ np.linalg.solve(C @ SiCt, np.array([1.0, 0.0]))
    return _portfolio(names, w, np.asarray(mu, dtype=float) if mu is not None else np.zeros(n), S)


# ------------------------------------------------------------------------------------------------------------------ APT
def apt_two_pass(returns: pd.DataFrame, factors: pd.DataFrame, add_constant: bool = False) -> dict:
    """Estimate an APT: loadings from time-series regressions on the factors, premia from the cross-section of mean returns on the loadings (Shanken-corrected), and the pricing errors. The pricing errors
    are the mispricings the theory says should be zero; ``alpha_chi2`` tests that they are jointly zero."""
    return fama_macbeth_two_pass(returns, factors, add_constant=add_constant)


def apt_expected_returns(loadings: pd.DataFrame, premia: pd.Series, risk_free: float = 0.0) -> pd.Series:
    """The APT's expected returns ``rf + B lambda``."""
    return risk_free + loadings @ premia.reindex(loadings.columns)


def statistical_factors(returns: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The first ``k`` principal components of the returns as factors, with each asset's loading on them (a statistical APT needs no named factors, only that the common variation is low-dimensional).

    The factor returns are the *raw* returns of the portfolios the components define (weights = the loadings), not the demeaned scores: their means are the factor premia, and a factor with zero mean by
    construction could price nothing."""
    R = returns.dropna()
    X = R.to_numpy() - R.to_numpy().mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    factors = pd.DataFrame(R.to_numpy() @ Vt[:k].T, index=R.index, columns=[f"pc{i + 1}" for i in range(k)])
    loadings = pd.DataFrame(Vt[:k].T, index=R.columns, columns=factors.columns)
    return factors, loadings


def apt_arbitrage_portfolio(alpha, loadings, cov, target_alpha: float = 1.0, gross: float | None = None) -> pd.Series:
    """The minimum-variance zero-investment portfolio with no factor exposure and alpha exposure ``target_alpha``: minimise ``w'Sigma w`` subject to ``1'w = 0``, ``B'w = 0`` and ``alpha'w = target``.
    The APT says its expected return is zero; if the estimated alphas are not, this is the cheapest way to trade them. ``gross`` rescales it to that sum of absolute weights."""
    names = list(alpha.index) if isinstance(alpha, pd.Series) else list(range(len(alpha)))
    a = np.asarray(alpha, dtype=float)
    B = np.atleast_2d(np.asarray(loadings, dtype=float))
    if B.shape[0] != len(a):
        B = B.T
    S = np.asarray(cov, dtype=float)
    S = 0.5 * (S + S.T)
    n = len(a)
    C = np.vstack([np.ones(n), B.T, a])
    SiCt = np.linalg.solve(S, C.T)
    d = np.zeros(C.shape[0])
    d[-1] = target_alpha
    w = SiCt @ np.linalg.solve(C @ SiCt, d)
    if gross is not None and np.abs(w).sum() > 0:
        w = w * gross / np.abs(w).sum()
    return pd.Series(w, index=names)
