"""Conditional Value at Risk (Ch. 21 §21.2.3-§21.2.4).

    CVaR_alpha = E[L | L > VaR_alpha]

The expected loss *given* that the VaR threshold is breached. Two properties
make it the better headline number for this project:

1. It is **coherent** (sub-additive): merging two books can never raise CVaR
   above the sum of their CVaRs. VaR can, which means VaR can penalise
   diversification.
2. It uses the whole tail rather than one quantile, so it distinguishes "you
   lose 3% on a bad day" from "you lose 3% on a bad day and 12% on a terrible
   one". Given the excess kurtosis measured in Stage 2, that distinction is
   the risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical_cvar(returns: pd.Series, alpha: float = 0.95, horizon: int = 1) -> float:
    clean = returns.dropna()
    if len(clean) < 20:
        return float("nan")
    cutoff = np.quantile(clean, 1.0 - alpha)
    tail = clean[clean <= cutoff]
    if tail.empty:
        return float("nan")
    return -float(tail.mean()) * np.sqrt(horizon)


def parametric_cvar(returns: pd.Series, alpha: float = 0.95, horizon: int = 1,
                    distribution: str = "normal") -> float:
    """Closed-form CVaR.

    Normal:  ``CVaR = -mu + sigma * phi(z) / (1 - alpha)``
    Student-t has the analogous expression with the t density.
    """
    clean = returns.dropna()
    if len(clean) < 20:
        return float("nan")
    mu, sigma = float(clean.mean()), float(clean.std(ddof=1))
    tail = 1.0 - alpha
    if distribution == "normal":
        z = stats.norm.ppf(tail)
        return float(-(mu - sigma * stats.norm.pdf(z) / tail)) * np.sqrt(horizon)
    if distribution in {"t", "student_t"}:
        df, loc, scale = stats.t.fit(clean.to_numpy())
        q = stats.t.ppf(tail, df)
        density = stats.t.pdf(q, df)
        expectation = -scale * density / tail * (df + q ** 2) / (df - 1.0)
        return float(-(loc + expectation)) * np.sqrt(horizon)
    raise ValueError(f"unknown distribution '{distribution}'")


def monte_carlo_cvar(returns, weights: pd.Series | None = None, alpha: float = 0.95,
                     horizon: int = 1, n_scenarios: int = 20000,
                     distribution: str = "student_t", df: float = 5.0,
                     seed: int = 20240101) -> float:
    from .var import monte_carlo_var

    _, draws = monte_carlo_var(returns, weights, alpha, horizon, n_scenarios, distribution, df, seed)
    cutoff = np.quantile(draws, 1.0 - alpha)
    tail = draws[draws <= cutoff]
    return -float(tail.mean()) if len(tail) else float("nan")


def cvar_comparison(returns: pd.Series, alphas=(0.95, 0.99), horizon: int = 1,
                    n_scenarios: int = 20000, df: float = 5.0, seed: int = 20240101) -> pd.DataFrame:
    from .var import historical_var, parametric_var

    rows = []
    for alpha in alphas:
        var_hist = historical_var(returns, alpha, horizon)
        cvar_hist = historical_cvar(returns, alpha, horizon)
        rows.append(
            {
                "alpha": alpha,
                "var_historical": var_hist,
                "cvar_historical": cvar_hist,
                "cvar_var_ratio": cvar_hist / var_hist if var_hist else np.nan,
                "var_normal": parametric_var(returns, alpha, horizon, "normal"),
                "cvar_normal": parametric_cvar(returns, alpha, horizon, "normal"),
                "cvar_t": parametric_cvar(returns, alpha, horizon, "t"),
                "cvar_monte_carlo": monte_carlo_cvar(returns, None, alpha, horizon,
                                                     n_scenarios, "student_t", df, seed),
            }
        )
    return pd.DataFrame(rows).set_index("alpha")


def rolling_cvar(returns: pd.Series, alpha: float = 0.95, lookback: int = 500,
                 min_lookback: int = 250) -> pd.Series:
    """Rolling historical CVaR, lagged so it is a genuine forecast."""
    def tail_mean(window: np.ndarray) -> float:
        cutoff = np.quantile(window, 1.0 - alpha)
        tail = window[window <= cutoff]
        return -float(tail.mean()) if len(tail) else np.nan

    clean = returns.dropna()
    return clean.rolling(lookback, min_periods=min_lookback).apply(tail_mean, raw=True) \
        .shift(1).rename(f"cvar_{alpha}")


def cvar_backtest(returns: pd.Series, alpha: float = 0.95, lookback: int = 500,
                  min_lookback: int = 250) -> dict:
    """Is the realised tail loss close to the forecast CVaR?

    The natural CVaR analogue of a VaR breach test: among the days that
    breached VaR, compare the average realised loss with the predicted
    conditional mean. A ratio far above 1 means the model understates how bad
    the bad days are.
    """
    from .var import rolling_var

    var_forecast = rolling_var(returns, alpha, lookback, "historical", min_lookback)
    cvar_forecast = rolling_cvar(returns, alpha, lookback, min_lookback)
    frame = pd.concat(
        [returns.rename("r"), var_forecast.rename("var"), cvar_forecast.rename("cvar")], axis=1
    ).dropna()
    if frame.empty:
        return {}
    losses = -frame["r"]
    breached = losses > frame["var"]
    if not breached.any():
        return {"n_breaches": 0}
    realised_tail = float(losses[breached].mean())
    predicted_tail = float(frame.loc[breached, "cvar"].mean())
    return {
        "n_obs": int(len(frame)),
        "n_breaches": int(breached.sum()),
        "realised_tail_loss": realised_tail,
        "predicted_cvar": predicted_tail,
        "ratio_realised_to_predicted": realised_tail / predicted_tail if predicted_tail else np.nan,
        "worst_realised_loss": float(losses.max()),
        "mean_cvar_forecast": float(frame["cvar"].mean()),
    }
