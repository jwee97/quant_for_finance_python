"""Risk decomposition (Ch. 21 §21.2.2, §21.2.4; spec §27).

Where does portfolio risk actually come from? Three decompositions, each
answering a different question:

``volatility``  RC_i = w_i (Sigma w)_i / sigma_p, summing to sigma_p
``VaR``         marginal and component VaR under a normal approximation
``CVaR``        component CVaR from the empirical tail, no distribution assumed
``PCA``         risk attributed to statistical factors rather than to assets

The last one matters on a multi-asset book: a portfolio can look diversified
across fifteen tickers while holding one enormous bet on PC1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..portfolio.covariance import portfolio_volatility


def volatility_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    from ..portfolio.risk_parity import risk_contribution_frame

    assets = [a for a in weights.index if a in covariance.columns]
    return risk_contribution_frame(weights.reindex(assets), covariance.loc[assets, assets])


def marginal_var(weights: pd.Series, covariance: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
    """``dVaR/dw_i`` under a normal approximation.

    ``MVaR_i = z_alpha * (Sigma w)_i / sigma_p``. The normal assumption is
    acknowledged rather than hidden: it is used here because the *derivative*
    is far less sensitive to the distributional assumption than the level is,
    and the empirical alternative is in ``component_cvar`` below.
    """
    from scipy import stats

    assets = [a for a in weights.index if a in covariance.columns]
    w = weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    cov = covariance.loc[assets, assets].to_numpy(dtype=float)
    sigma = portfolio_volatility(w, cov)
    if sigma <= 1e-14:
        return pd.Series(0.0, index=assets)
    z = abs(stats.norm.ppf(1.0 - alpha))
    return pd.Series(z * (cov @ w) / sigma, index=assets, name="marginal_var")


def component_var(weights: pd.Series, covariance: pd.DataFrame, alpha: float = 0.95) -> pd.DataFrame:
    """Component VaR: ``w_i * MVaR_i``, summing to the portfolio VaR."""
    assets = [a for a in weights.index if a in covariance.columns]
    marginal = marginal_var(weights, covariance, alpha)
    w = weights.reindex(assets)
    component = w * marginal
    total = component.sum()
    return pd.DataFrame(
        {
            "weight": w,
            "marginal_var": marginal,
            "component_var": component,
            "var_share": component / total if abs(total) > 1e-14 else np.nan,
        }
    )


def component_cvar(returns: pd.DataFrame, weights: pd.Series, alpha: float = 0.95) -> pd.DataFrame:
    """Component CVaR from the empirical tail.

    ``CCVaR_i = w_i * E[-r_i | portfolio in its worst (1-alpha) tail]``. No
    distributional assumption at all: it asks what each asset actually did on
    the portfolio's worst days, which is the question a risk committee is
    really asking.
    """
    assets = [a for a in weights.index if a in returns.columns]
    w = weights.reindex(assets).fillna(0.0)
    data = returns[assets].dropna(how="any")
    if data.empty:
        return pd.DataFrame()
    portfolio = data @ w
    cutoff = np.quantile(portfolio, 1.0 - alpha)
    tail = data[portfolio <= cutoff]
    if tail.empty:
        return pd.DataFrame()
    contribution = -(tail.mean() * w)
    total = contribution.sum()
    return pd.DataFrame(
        {
            "weight": w,
            "mean_return_in_tail": tail.mean(),
            "component_cvar": contribution,
            "cvar_share": contribution / total if abs(total) > 1e-14 else np.nan,
            "n_tail_days": int(len(tail)),
        }
    )


def factor_risk_contributions(weights: pd.Series, returns: pd.DataFrame,
                              n_components: int = 5) -> pd.DataFrame:
    """Attribute portfolio variance to principal components.

    The check that a fifteen-ticker book is not one bet wearing a disguise.
    """
    from ..features.pca import pca_decomposition

    assets = [a for a in weights.index if a in returns.columns]
    data = returns[assets].dropna(how="any")
    if data.empty or len(data) < len(assets) + 5:
        return pd.DataFrame()

    result = pca_decomposition(data, use_correlation=False)
    w = weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    loadings = result.eigenvectors.reindex(assets).to_numpy(dtype=float)
    exposures = loadings.T @ w                      # portfolio's exposure to each PC
    variances = exposures ** 2 * result.eigenvalues.to_numpy()
    total = variances.sum()

    n = min(n_components, len(variances))
    return pd.DataFrame(
        {
            "factor_exposure": exposures[:n],
            "variance_contribution": variances[:n],
            "variance_share": (variances / total)[:n] if total > 0 else np.nan,
            "factor_explained_variance": result.explained_variance.to_numpy()[:n],
        },
        index=result.eigenvalues.index[:n],
    )


def diversification_ratio(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """``sum_i w_i sigma_i / sigma_p``.

    One means no diversification benefit at all; higher is better. A useful
    single-number complement to the risk-contribution table.
    """
    assets = [a for a in weights.index if a in covariance.columns]
    w = weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    cov = covariance.loc[assets, assets].to_numpy(dtype=float)
    weighted_vol = float(np.abs(w) @ np.sqrt(np.diag(cov)))
    sigma = portfolio_volatility(w, cov)
    return weighted_vol / sigma if sigma > 1e-14 else float("nan")


def risk_summary(weights: pd.Series, returns: pd.DataFrame, covariance: pd.DataFrame,
                 alpha: float = 0.95) -> dict:
    """Headline risk numbers for one book."""
    from .cvar import historical_cvar
    from .var import historical_var

    assets = [a for a in weights.index if a in returns.columns]
    w = weights.reindex(assets).fillna(0.0)
    portfolio = (returns[assets] @ w).dropna()
    volatility_frame = volatility_contributions(w, covariance)
    factor = factor_risk_contributions(w, returns)
    return {
        "portfolio_vol_ann": portfolio_volatility(w.to_numpy(), covariance.loc[assets, assets].to_numpy()),
        "diversification_ratio": diversification_ratio(w, covariance),
        "effective_n_positions": float(1.0 / (w ** 2).sum()) if float((w ** 2).sum()) > 0 else np.nan,
        "max_risk_share": float(volatility_frame["risk_share"].max()),
        "top_risk_contributor": str(volatility_frame["risk_share"].idxmax()),
        "pc1_variance_share": float(factor["variance_share"].iloc[0]) if len(factor) else np.nan,
        "var_95_daily": historical_var(portfolio, alpha),
        "cvar_95_daily": historical_cvar(portfolio, alpha),
    }
