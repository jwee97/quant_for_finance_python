"""Bayesian portfolio construction (Generation 3, Priority 3).

A plug-in optimiser takes the sample mean and covariance as the truth and
maximises over them, which is Chapter 19's error-maximisation problem. Here the
parameters are uncertain, and the decision is taken under that uncertainty.

Model. Daily returns ``r ~ N(mu, Sigma)`` with the conjugate
normal-inverse-Wishart prior

    Sigma ~ IW(nu0, Psi0),      mu | Sigma ~ N(m0, Sigma / kappa0),

and posterior (n observations, mean xbar, scatter S)

    kappa_n = kappa0 + n,   nu_n = nu0 + n,   m_n = (kappa0 m0 + n xbar) / kappa_n,
    Psi_n = Psi0 + S + (kappa0 n / kappa_n) (xbar - m0)(xbar - m0)'.

Prior. The mean is shrunk toward a COMMON value (the minimum-variance
portfolio's mean, Jorion's Bayes-Stein target), with a strength chosen by
empirical Bayes rather than a free constant: for ``N`` assets and ``T``
observations the shrinkage factor is

    w = (N + 2) / ((N + 2) + T (xbar - m0 1)' S^-1 (xbar - m0 1)),   kappa0 = T w / (1 - w).

The covariance prior is centred on the constant-correlation target with the
equivalent of ``nu0`` days of data.

Decision. Weights are a distribution: one constrained mean-variance optimum per
posterior draw. The posterior-mean weights average those optima, which is the
Bayes decision under estimation risk for a risk-averse investor who cannot
short; the spread of the optima is the uncertainty about the allocation itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import invwishart

from .constraints import Constraints
from .covariance import constant_correlation_target
from .mean_variance import mean_variance_weights

ANN = 252.0


@dataclass
class Posterior:
    """Normal-inverse-Wishart posterior over daily (mu, Sigma)."""

    assets: list[str]
    mean: np.ndarray           # m_n
    kappa: float               # kappa_n
    nu: float                  # nu_n
    scale: np.ndarray          # Psi_n
    prior_weight: float        # Jorion shrinkage factor w
    kappa0: float

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def mean_covariance(self) -> np.ndarray:
        """E[Sigma | data] = Psi_n / (nu_n - N - 1), daily."""
        return self.scale / (self.nu - self.n_assets - 1.0)

    def draw(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """``n`` joint draws of (mu, Sigma), daily units: arrays (n, N) and (n, N, N)."""
        sigmas = invwishart(df=self.nu, scale=self.scale).rvs(size=n, random_state=rng)
        sigmas = np.asarray(sigmas).reshape(n, self.n_assets, self.n_assets)
        chol = np.linalg.cholesky(sigmas / self.kappa)
        mus = self.mean + np.einsum("nij,nj->ni", chol, rng.standard_normal((n, self.n_assets)))
        return mus, sigmas


def jorion_prior(xbar: np.ndarray, sample_cov: np.ndarray, n_obs: int) -> tuple[float, float]:
    """(common prior mean m0, shrinkage factor w) by empirical Bayes."""
    n = len(xbar)
    inverse = np.linalg.inv(sample_cov)
    ones = np.ones(n)
    m0 = float(ones @ inverse @ xbar / (ones @ inverse @ ones))
    deviation = xbar - m0
    distance = float(n_obs * deviation @ inverse @ deviation)
    weight = (n + 2.0) / ((n + 2.0) + distance)
    return m0, float(np.clip(weight, 0.0, 0.999))


def niw_posterior(window: pd.DataFrame, nu0: float = 126.0) -> Posterior:
    """Posterior from a window of daily returns (complete rows only)."""
    clean = window.dropna(how="any")
    n_obs, n = clean.shape
    values = clean.to_numpy(dtype=float)
    xbar = values.mean(axis=0)
    sample_cov = np.cov(values, rowvar=False, ddof=1)
    m0, weight = jorion_prior(xbar, sample_cov, n_obs)
    kappa0 = n_obs * weight / (1.0 - weight)
    target = constant_correlation_target(clean).to_numpy(dtype=float)
    psi0 = (nu0 - n - 1.0) * target
    scatter = (values - xbar).T @ (values - xbar)
    kappa_n, nu_n = kappa0 + n_obs, nu0 + n_obs
    m_vec = np.full(n, m0)
    mean_n = (kappa0 * m_vec + n_obs * xbar) / kappa_n
    gap = (xbar - m_vec)[:, None]
    psi_n = psi0 + scatter + (kappa0 * n_obs / kappa_n) * (gap @ gap.T)
    psi_n = 0.5 * (psi_n + psi_n.T)
    return Posterior(list(clean.columns), mean_n, kappa_n, nu_n, psi_n, weight, kappa0)


def _frame(cov: np.ndarray, assets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(cov, index=assets, columns=assets)


def posterior_weight_draws(post: Posterior, constraints: Constraints, risk_aversion: float,
                           n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Constrained mean-variance optimum for each posterior draw: array (draws, N); failed solves are dropped."""
    mus, sigmas = post.draw(n_draws, rng)
    out = []
    start = np.full(post.n_assets, 1.0 / post.n_assets)
    for mu, sigma in zip(mus, sigmas):
        result = mean_variance_weights(pd.Series(ANN * mu, index=post.assets), _frame(ANN * sigma, post.assets),
                                       risk_aversion, constraints)
        if result.success:
            out.append(result.weights.reindex(post.assets).to_numpy())
            start = out[-1]
    if not out:
        raise RuntimeError("no posterior draw produced a feasible optimum")
    return np.vstack(out)


def bayesian_weights(window: pd.DataFrame, kind: str, constraints: Constraints, risk_aversion: float = 5.0,
                     nu0: float = 126.0, n_draws: int = 100, seed: int = 11) -> pd.Series:
    """Weights for one window. ``kind``: ``mvo_sample`` (control), ``bayes_stein`` or ``bayes_predictive``."""
    clean = window.dropna(how="any")
    assets = list(clean.columns)
    if kind == "mvo_sample":
        from .covariance import estimate_covariance
        cov = estimate_covariance(clean, "shrinkage", len(clean), annualise=True)
        mu = pd.Series(ANN * clean.mean().to_numpy(), index=assets)
        result = mean_variance_weights(mu, cov, risk_aversion, constraints)
        if not result.success:
            raise RuntimeError("mvo_sample did not converge")
        return result.weights
    post = niw_posterior(clean, nu0)
    if kind == "bayes_stein":
        mu = pd.Series(ANN * post.mean, index=assets)
        result = mean_variance_weights(mu, _frame(ANN * post.mean_covariance(), assets), risk_aversion, constraints)
        if not result.success:
            raise RuntimeError("bayes_stein did not converge")
        return result.weights
    if kind == "bayes_predictive":
        draws = posterior_weight_draws(post, constraints, risk_aversion, n_draws, np.random.default_rng(seed))
        return pd.Series(draws.mean(axis=0), index=assets)
    raise ValueError(f"unknown kind '{kind}'")


def bayesian_book(returns: pd.DataFrame, investable: pd.DataFrame, kind: str, constraints: Constraints,
                  rebalance_index: pd.DatetimeIndex, lookback: int = 252, min_assets: int = 5,
                  risk_aversion: float = 5.0, nu0: float = 126.0, n_draws: int = 100, seed: int = 11) -> pd.DataFrame:
    """Target weights on each rebalance date from trailing data only; forward-filled like the other books."""
    index = pd.DatetimeIndex(returns.index)
    book = pd.DataFrame(np.nan, index=index, columns=returns.columns)
    for stamp in rebalance_index:
        position = index.get_loc(stamp)
        if position < lookback:
            continue
        window = returns.iloc[position - lookback + 1:position + 1]
        live = [c for c in returns.columns if bool(investable.loc[stamp, c]) and window[c].notna().sum() > lookback * 0.8]
        if len(live) < min_assets:
            continue
        sample = window[live].dropna(how="any")
        if len(sample) < lookback * 0.6:
            continue
        try:
            weights = bayesian_weights(sample, kind, constraints, risk_aversion, nu0, n_draws, seed + position)
        except Exception:
            continue
        book.loc[stamp, live] = weights.reindex(live).to_numpy()
    return book.ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# Weights as a distribution, and the predictive check
# ---------------------------------------------------------------------------
def weight_distribution(window: pd.DataFrame, constraints: Constraints, risk_aversion: float = 5.0,
                        nu0: float = 126.0, n_draws: int = 200, seed: int = 11) -> pd.DataFrame:
    """Per asset: posterior mean, 5th and 95th percentile of the optimal weight, share of draws holding it."""
    post = niw_posterior(window, nu0)
    draws = posterior_weight_draws(post, constraints, risk_aversion, n_draws, np.random.default_rng(seed))
    return pd.DataFrame({
        "mean": draws.mean(axis=0), "p05": np.percentile(draws, 5, axis=0), "p95": np.percentile(draws, 95, axis=0),
        "share_holding": (draws > 1e-3).mean(axis=0),
    }, index=post.assets)


def predictive_quantiles(window: pd.DataFrame, weights: pd.Series, horizon: int = 21, levels=(0.5, 0.9),
                         nu0: float = 126.0, n_draws: int = 400, seed: int = 11) -> dict:
    """Central predictive intervals for the book's ``horizon``-day return, integrating over (mu, Sigma)."""
    post = niw_posterior(window, nu0)
    rng = np.random.default_rng(seed)
    mus, sigmas = post.draw(n_draws, rng)
    w = weights.reindex(post.assets).fillna(0.0).to_numpy()
    mean = horizon * mus @ w
    var = horizon * np.einsum("i,nij,j->n", w, sigmas, w)
    sample = mean + np.sqrt(var) * rng.standard_normal(n_draws)
    out = {"median": float(np.median(sample))}
    for level in levels:
        lo, hi = np.percentile(sample, [50 * (1 - level), 100 - 50 * (1 - level)])
        out[f"lo_{level}"], out[f"hi_{level}"] = float(lo), float(hi)
    return out


def weight_stability_bayes(window: pd.DataFrame, methods: dict, constraints: Constraints, n_bootstrap: int = 60,
                           block_length: int = 21, seed: int = 5, risk_aversion: float = 5.0) -> pd.DataFrame:
    """Dispersion of each method's weights when the window is block-bootstrapped (cf. Stage 18).

    ``methods`` maps a name to a function ``returns -> weights``. Reported: mean pairwise L1 distance between
    bootstrap weight vectors, mean maximum weight and mean effective number of positions.
    """
    from ..validation.robustness import stationary_bootstrap_indices

    clean = window.dropna(how="any")
    values = clean.to_numpy()
    columns = list(clean.columns)
    n_obs = len(clean)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[np.ndarray]] = {name: [] for name in methods}
    for _ in range(n_bootstrap):
        sample = pd.DataFrame(values[stationary_bootstrap_indices(n_obs, block_length, rng)], columns=columns)
        for name, fn in methods.items():
            try:
                draws[name].append(fn(sample).reindex(columns).to_numpy())
            except Exception:
                continue
    rows = {}
    for name, vectors in draws.items():
        if len(vectors) < 5:
            continue
        stack = np.vstack(vectors)
        pairwise = [np.abs(stack[i] - stack[j]).sum() for i in range(len(stack)) for j in range(i + 1, min(i + 20, len(stack)))]
        rows[name] = {
            "mean_pairwise_l1": float(np.mean(pairwise)),
            "mean_max_weight": float(stack.max(axis=1).mean()),
            "mean_effective_n": float((1.0 / np.square(stack).sum(axis=1)).mean()),
            "n_draws": len(vectors),
        }
    return pd.DataFrame(rows).T
