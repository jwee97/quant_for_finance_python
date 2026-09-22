"""Value at Risk (Ch. 21 §21.2-§21.3, spec §34-§36).

``VaR_alpha`` is the loss exceeded with probability ``1 - alpha``. All VaR
figures here are **positive numbers representing losses**, which removes the
most common source of sign confusion in risk code.

Four methods, deliberately kept side by side rather than choosing one:

``historical``          empirical quantile. No distributional assumption; the
                        tail is whatever happened.
``parametric_normal``   mu + z sigma. Fast, and demonstrably wrong on this
                        universe -- Stage 2 rejected normality for all 15
                        assets, and the backtest below quantifies the cost.
``parametric_t``        Student-t with estimated degrees of freedom.
``monte_carlo``         simulate from a fitted multivariate distribution, or
                        bootstrap the joint history.

And, more importantly, ``rolling_var_backtest`` with Kupiec and Christoffersen
tests: a risk model is validated by counting breaches, not by the plausibility
of its number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Point estimators
# ---------------------------------------------------------------------------
def historical_var(returns: pd.Series, alpha: float = 0.95, horizon: int = 1) -> float:
    """Empirical quantile of the loss distribution."""
    clean = returns.dropna()
    if len(clean) < 20:
        return float("nan")
    value = -float(np.quantile(clean, 1.0 - alpha))
    return value * np.sqrt(horizon)


def parametric_var(returns: pd.Series, alpha: float = 0.95, horizon: int = 1,
                   distribution: str = "normal") -> float:
    """Closed-form VaR under an assumed distribution."""
    clean = returns.dropna()
    if len(clean) < 20:
        return float("nan")
    mu, sigma = float(clean.mean()), float(clean.std(ddof=1))
    if distribution == "normal":
        quantile = stats.norm.ppf(1.0 - alpha)
    elif distribution in {"t", "student_t"}:
        df, loc, scale = stats.t.fit(clean.to_numpy())
        # Standardise so the quantile is comparable to the normal case.
        quantile = stats.t.ppf(1.0 - alpha, df)
        sigma = scale * np.sqrt(df / (df - 2.0)) if df > 2 else scale
        mu = loc
        return -(mu + quantile * scale) * np.sqrt(horizon)
    else:
        raise ValueError(f"unknown distribution '{distribution}'")
    return -(mu + quantile * sigma) * np.sqrt(horizon)


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.95, horizon: int = 1) -> float:
    """Normal VaR adjusted for skewness and excess kurtosis.

    A cheap middle ground between the parametric and historical methods: it
    keeps the closed form but uses the third and fourth moments the Stage 2
    EDA showed are large.
    """
    clean = returns.dropna()
    if len(clean) < 30:
        return float("nan")
    mu, sigma = float(clean.mean()), float(clean.std(ddof=1))
    skew = float(stats.skew(clean, bias=False))
    excess = float(stats.kurtosis(clean, bias=False))
    z = stats.norm.ppf(1.0 - alpha)
    adjusted = (z + (z ** 2 - 1) * skew / 6.0 + (z ** 3 - 3 * z) * excess / 24.0
                - (2 * z ** 3 - 5 * z) * skew ** 2 / 36.0)
    return -(mu + adjusted * sigma) * np.sqrt(horizon)


def monte_carlo_var(returns: pd.DataFrame | pd.Series, weights: pd.Series | None = None,
                    alpha: float = 0.95, horizon: int = 1, n_scenarios: int = 20000,
                    distribution: str = "student_t", df: float = 5.0,
                    seed: int = 20240101) -> tuple[float, np.ndarray]:
    """Simulate the portfolio loss distribution.

    For a multi-asset book the simulation is on the **joint** distribution:
    scenarios are drawn from a multivariate normal or multivariate-t with the
    estimated covariance, or bootstrapped as whole rows of history. Drawing
    each asset independently would destroy the correlation structure, which is
    the entire reason a portfolio has less risk than its parts.

    Returns ``(var, simulated_portfolio_returns)``.
    """
    rng = np.random.default_rng(seed)

    if isinstance(returns, pd.Series) or weights is None:
        series = returns if isinstance(returns, pd.Series) else (returns @ weights)
        clean = series.dropna()
        mu, sigma = float(clean.mean()), float(clean.std(ddof=1))
        if distribution == "normal":
            draws = rng.normal(mu, sigma, n_scenarios)
        elif distribution in {"t", "student_t"}:
            scale = sigma / np.sqrt(df / (df - 2.0)) if df > 2 else sigma
            draws = mu + scale * rng.standard_t(df, n_scenarios)
        elif distribution == "bootstrap":
            draws = rng.choice(clean.to_numpy(), size=n_scenarios, replace=True)
        else:
            raise ValueError(f"unknown distribution '{distribution}'")
    else:
        clean = returns.dropna(how="any")
        w = weights.reindex(clean.columns).fillna(0.0).to_numpy(dtype=float)
        if distribution == "bootstrap":
            rows = rng.integers(0, len(clean), size=n_scenarios)
            draws = clean.to_numpy()[rows] @ w
        else:
            mean = clean.mean().to_numpy()
            cov = clean.cov(ddof=1).to_numpy()
            if distribution == "normal":
                simulated = rng.multivariate_normal(mean, cov, n_scenarios)
            else:
                # Multivariate t: normal mixture with an inverse-chi-square scale.
                scale = cov * (df - 2.0) / df if df > 2 else cov
                normal = rng.multivariate_normal(np.zeros(len(mean)), scale, n_scenarios)
                chi = rng.chisquare(df, n_scenarios) / df
                simulated = mean + normal / np.sqrt(chi)[:, None]
            draws = simulated @ w

    if horizon > 1:
        draws = draws * np.sqrt(horizon)
    return -float(np.quantile(draws, 1.0 - alpha)), draws


def var_comparison(returns: pd.Series, alphas=(0.95, 0.99), horizon: int = 1,
                   n_scenarios: int = 20000, df: float = 5.0, seed: int = 20240101) -> pd.DataFrame:
    """Every method at every confidence level, in one table."""
    rows = []
    for alpha in alphas:
        mc, _ = monte_carlo_var(returns, None, alpha, horizon, n_scenarios, "student_t", df, seed)
        mc_normal, _ = monte_carlo_var(returns, None, alpha, horizon, n_scenarios, "normal", df, seed)
        rows.append(
            {
                "alpha": alpha,
                "historical": historical_var(returns, alpha, horizon),
                "parametric_normal": parametric_var(returns, alpha, horizon, "normal"),
                "parametric_t": parametric_var(returns, alpha, horizon, "t"),
                "cornish_fisher": cornish_fisher_var(returns, alpha, horizon),
                "monte_carlo_normal": mc_normal,
                "monte_carlo_t": mc,
            }
        )
    return pd.DataFrame(rows).set_index("alpha")


# ---------------------------------------------------------------------------
# Rolling VaR and breach testing (spec §36, Ch. 21 §21.3.5-6)
# ---------------------------------------------------------------------------
def rolling_var(returns: pd.Series, alpha: float = 0.95, lookback: int = 500,
                method: str = "historical", min_lookback: int = 250,
                df: float = 5.0) -> pd.Series:
    """VaR forecast for ``t+1`` using information through ``t`` only.

    The ``shift(1)`` at the end is the whole point: a VaR computed on a window
    that includes tomorrow would pass any backtest.
    """
    clean = returns.dropna()
    if method == "historical":
        forecast = clean.rolling(lookback, min_periods=min_lookback).quantile(1.0 - alpha).mul(-1.0)
    elif method == "parametric_normal":
        mu = clean.rolling(lookback, min_periods=min_lookback).mean()
        sigma = clean.rolling(lookback, min_periods=min_lookback).std(ddof=1)
        forecast = -(mu + stats.norm.ppf(1.0 - alpha) * sigma)
    elif method == "ewma_normal":
        # RiskMetrics-style: exponentially weighted variance, normal quantile.
        variance = (clean ** 2).ewm(halflife=lookback / 8.0, min_periods=min_lookback).mean()
        forecast = -(stats.norm.ppf(1.0 - alpha) * np.sqrt(variance))
    elif method == "parametric_t":
        mu = clean.rolling(lookback, min_periods=min_lookback).mean()
        sigma = clean.rolling(lookback, min_periods=min_lookback).std(ddof=1)
        quantile = stats.t.ppf(1.0 - alpha, df) / np.sqrt(df / (df - 2.0))
        forecast = -(mu + quantile * sigma)
    elif method == "cornish_fisher":
        forecast = clean.rolling(lookback, min_periods=min_lookback).apply(
            lambda w: cornish_fisher_var(pd.Series(w), alpha), raw=False
        )
    else:
        raise ValueError(f"unknown rolling VaR method '{method}'")
    return forecast.shift(1).rename(f"var_{method}_{alpha}")


@dataclass
class VaRBacktest:
    """Breach statistics and formal coverage tests."""

    alpha: float
    method: str
    n_obs: int
    n_breaches: int
    expected_breaches: float
    breach_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    christoffersen_stat: float
    christoffersen_pvalue: float
    joint_stat: float
    joint_pvalue: float
    mean_breach_size: float
    max_breach_size: float
    breaches: pd.Series

    def verdict(self, significance: float = 0.05) -> str:
        if not np.isfinite(self.kupiec_pvalue):
            return "inconclusive"
        coverage_ok = self.kupiec_pvalue > significance
        independence_ok = (not np.isfinite(self.christoffersen_pvalue)
                           or self.christoffersen_pvalue > significance)
        if coverage_ok and independence_ok:
            return "pass"
        if not coverage_ok and self.breach_rate > (1.0 - self.alpha):
            return "fail: understates risk"
        if not coverage_ok:
            return "fail: overstates risk"
        return "fail: breaches are clustered"

    def to_row(self) -> dict:
        return {
            "alpha": self.alpha,
            "method": self.method,
            "n_obs": self.n_obs,
            "n_breaches": self.n_breaches,
            "expected_breaches": self.expected_breaches,
            "breach_rate": self.breach_rate,
            "expected_rate": 1.0 - self.alpha,
            "kupiec_pvalue": self.kupiec_pvalue,
            "christoffersen_pvalue": self.christoffersen_pvalue,
            "joint_pvalue": self.joint_pvalue,
            "mean_breach_size": self.mean_breach_size,
            "max_breach_size": self.max_breach_size,
            "verdict": self.verdict(),
        }


def kupiec_test(n_breaches: int, n_obs: int, alpha: float) -> tuple[float, float]:
    """Unconditional coverage: are there the right *number* of breaches?

    Likelihood ratio against the null that the true breach probability equals
    ``1 - alpha``. Chi-square with one degree of freedom.
    """
    p = 1.0 - alpha
    if n_obs == 0 or n_breaches == 0:
        if n_obs == 0:
            return float("nan"), float("nan")
        statistic = -2.0 * n_obs * np.log(1.0 - p)
        return float(statistic), float(1.0 - stats.chi2.cdf(statistic, 1))
    rate = n_breaches / n_obs
    if rate in (0.0, 1.0):
        return float("nan"), float("nan")
    log_null = (n_obs - n_breaches) * np.log(1.0 - p) + n_breaches * np.log(p)
    log_alt = (n_obs - n_breaches) * np.log(1.0 - rate) + n_breaches * np.log(rate)
    statistic = -2.0 * (log_null - log_alt)
    return float(statistic), float(1.0 - stats.chi2.cdf(statistic, 1))


def christoffersen_test(breaches: pd.Series) -> tuple[float, float]:
    """Independence: are breaches *clustered*?

    A model can have exactly the right number of breaches and still be useless
    if they all arrive in the same week -- which is precisely what happens when
    a model ignores volatility clustering. Tests a first-order Markov chain
    against the independence restriction.
    """
    indicator = breaches.dropna().astype(int).to_numpy()
    if len(indicator) < 10:
        return float("nan"), float("nan")
    previous, current = indicator[:-1], indicator[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return float("nan"), float("nan")
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi in (0.0, 1.0) or pi01 in (0.0,) or pi11 in (0.0, 1.0):
        return float("nan"), float("nan")

    log_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    log_alt = (n00 * np.log(1 - pi01) + n01 * np.log(pi01)
               + n10 * np.log(1 - pi11) + n11 * np.log(pi11))
    statistic = -2.0 * (log_null - log_alt)
    return float(statistic), float(1.0 - stats.chi2.cdf(statistic, 1))


def rolling_var_backtest(returns: pd.Series, alpha: float = 0.95, lookback: int = 500,
                         method: str = "historical", min_lookback: int = 250,
                         df: float = 5.0) -> VaRBacktest:
    """Forecast VaR through time, count the breaches, test them."""
    forecast = rolling_var(returns, alpha, lookback, method, min_lookback, df)
    aligned = pd.concat([returns.rename("r"), forecast.rename("var")], axis=1).dropna()
    if aligned.empty:
        raise ValueError("no overlapping VaR forecasts and returns")

    losses = -aligned["r"]
    breaches = (losses > aligned["var"]).astype(int)
    n_obs, n_breaches = int(len(breaches)), int(breaches.sum())

    kupiec_stat, kupiec_p = kupiec_test(n_breaches, n_obs, alpha)
    christ_stat, christ_p = christoffersen_test(breaches)
    joint_stat = kupiec_stat + christ_stat if np.isfinite(christ_stat) else kupiec_stat
    joint_p = float(1.0 - stats.chi2.cdf(joint_stat, 2)) if np.isfinite(joint_stat) else float("nan")

    excess = (losses - aligned["var"])[breaches.astype(bool)]
    return VaRBacktest(
        alpha=alpha, method=method, n_obs=n_obs, n_breaches=n_breaches,
        expected_breaches=n_obs * (1.0 - alpha),
        breach_rate=n_breaches / n_obs if n_obs else float("nan"),
        kupiec_stat=kupiec_stat, kupiec_pvalue=kupiec_p,
        christoffersen_stat=christ_stat, christoffersen_pvalue=christ_p,
        joint_stat=joint_stat, joint_pvalue=joint_p,
        mean_breach_size=float(excess.mean()) if len(excess) else 0.0,
        max_breach_size=float(excess.max()) if len(excess) else 0.0,
        breaches=breaches,
    )


def compare_var_methods(returns: pd.Series, alphas=(0.95, 0.99),
                        methods=("historical", "parametric_normal", "parametric_t", "ewma_normal"),
                        lookback: int = 500, min_lookback: int = 250, df: float = 5.0) -> pd.DataFrame:
    """Backtest every method at every confidence level.

    This table is the answer to "which VaR method should we use?" -- and it is
    an empirical answer, not a theoretical preference.
    """
    rows = []
    for alpha in alphas:
        for method in methods:
            try:
                result = rolling_var_backtest(returns, alpha, lookback, method, min_lookback, df)
            except ValueError:
                continue
            rows.append(result.to_row())
    return pd.DataFrame(rows)
