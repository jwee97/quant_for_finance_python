"""Monte Carlo: path simulators, variance reduction and importance sampling for rare events.

Simulators (all take a ``seed`` and return arrays of shape ``(paths, steps + 1)`` or ``(paths, steps + 1, assets)``):
``simulate_gbm`` (exact lognormal steps, antithetic option), ``simulate_correlated_gbm`` (Cholesky), ``simulate_ou`` (exact discretisation),
``simulate_merton_jump`` (jump diffusion) and ``simulate_garch`` (GARCH(1,1) with Gaussian or Student-t shocks).

Variance reduction in ``mc_estimate``: antithetic variates, a control variate with a known mean, and randomised quasi-Monte Carlo (scrambled Sobol,
``scipy.stats.qmc``), whose error is estimated from independent randomisations.

Rare events (a 1-in-10,000 loss) are hit by about one path in 10,000, so plain Monte Carlo needs millions of paths. Importance sampling draws from a
distribution that makes the event common and corrects with the likelihood ratio: ``portfolio_tail_probability_is`` shifts the mean of the risk factors to the
most likely failure point (the exponential-tilting solution for a linear loss), and ``cross_entropy_is`` learns the shifted Gaussian adaptively (Rubinstein 1999).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as sps
from scipy.stats import qmc


def _rng(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


# --------------------------------------------------------------------------------------------------------------------- simulators
def simulate_gbm(s0: float, mu: float, sigma: float, horizon: float, steps: int, n_paths: int, antithetic: bool = False, seed=0) -> np.ndarray:
    """Geometric Brownian motion with exact steps: ``S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)``. With ``antithetic`` the second half of the
    paths uses ``-Z`` (``n_paths`` is rounded up to even)."""
    rng = _rng(seed)
    dt = horizon / steps
    half = (n_paths + 1) // 2 if antithetic else n_paths
    z = rng.standard_normal((half, steps))
    if antithetic:
        z = np.vstack([z, -z])[:n_paths + (n_paths % 2)]
    log_inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    return s0 * np.exp(np.concatenate([np.zeros((len(z), 1)), np.cumsum(log_inc, axis=1)], axis=1))


def simulate_correlated_gbm(s0, mu, cov, horizon: float, steps: int, n_paths: int, seed=0) -> np.ndarray:
    """Correlated GBM: ``cov`` is the ANNUAL covariance matrix of log returns. Output ``(paths, steps + 1, assets)``."""
    rng = _rng(seed)
    s0, mu, cov = np.asarray(s0, float), np.asarray(mu, float), np.asarray(cov, float)
    dt = horizon / steps
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_paths, steps, len(s0))) @ L.T
    drift = (mu - 0.5 * np.diag(cov)) * dt
    log_inc = drift + np.sqrt(dt) * z
    out = np.concatenate([np.zeros((n_paths, 1, len(s0))), np.cumsum(log_inc, axis=1)], axis=1)
    return s0 * np.exp(out)


def simulate_ou(x0: float, kappa: float, theta: float, sigma: float, horizon: float, steps: int, n_paths: int, seed=0) -> np.ndarray:
    """Ornstein-Uhlenbeck ``dX = kappa (theta - X) dt + sigma dW`` with the exact transition ``X_{t+dt} | X_t ~ N(theta + (X_t - theta) e^{-kappa dt}, sigma^2 (1 - e^{-2 kappa dt}) / 2 kappa)``."""
    rng = _rng(seed)
    dt = horizon / steps
    a = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1.0 - a ** 2) / (2.0 * kappa))
    x = np.empty((n_paths, steps + 1))
    x[:, 0] = x0
    z = rng.standard_normal((n_paths, steps))
    for t in range(steps):
        x[:, t + 1] = theta + (x[:, t] - theta) * a + sd * z[:, t]
    return x


def simulate_merton_jump(s0: float, mu: float, sigma: float, lam: float, jump_mean: float, jump_std: float, horizon: float, steps: int, n_paths: int, seed=0) -> np.ndarray:
    """Merton (1976) jump diffusion: GBM plus Poisson(``lam``) jumps whose log size is ``N(jump_mean, jump_std^2)``. The drift is compensated so that
    ``E[S_T] = s0 e^{mu T}``."""
    rng = _rng(seed)
    dt = horizon / steps
    kbar = np.exp(jump_mean + 0.5 * jump_std ** 2) - 1.0
    n_jumps = rng.poisson(lam * dt, size=(n_paths, steps))
    jumps = n_jumps * jump_mean + np.sqrt(n_jumps) * jump_std * rng.standard_normal((n_paths, steps))
    inc = (mu - lam * kbar - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, steps)) + jumps
    return s0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1))


def simulate_garch(n_steps: int, n_paths: int, omega: float, alpha: float, beta: float, mu: float = 0.0, df: float | None = None, seed=0, burn: int = 200) -> np.ndarray:
    """GARCH(1,1) returns ``r_t = mu + sqrt(h_t) e_t``, ``h_t = omega + alpha (r_{t-1} - mu)^2 + beta h_{t-1}``; ``e`` Gaussian or unit-variance Student-t with ``df``."""
    rng = _rng(seed)
    total = n_steps + burn
    e = rng.standard_normal((n_paths, total)) if df is None else rng.standard_t(df, size=(n_paths, total)) * np.sqrt((df - 2.0) / df)
    h = np.full(n_paths, omega / max(1e-12, 1.0 - alpha - beta))
    r = np.empty((n_paths, total))
    for t in range(total):
        r[:, t] = mu + np.sqrt(h) * e[:, t]
        h = omega + alpha * (r[:, t] - mu) ** 2 + beta * h
    return r[:, burn:]


# ---------------------------------------------------------------------------------------------------------------- estimation
def mc_estimate(f: Callable[[np.ndarray], np.ndarray], dim: int, n: int = 100000, method: str = "plain", control: Callable | None = None,
                control_mean: float = 0.0, n_rand: int = 10, seed: int = 0) -> dict:
    """``E[f(Z)]`` for ``Z ~ N(0, I_dim)`` by Monte Carlo. ``f`` takes ``(m, dim)`` and returns ``(m,)``.

    ``method``: ``plain``; ``antithetic`` (averages ``f(Z)`` and ``f(-Z)``); ``control`` (needs ``control``, a function of ``Z`` with known mean ``control_mean``,
    and subtracts its optimally scaled deviation); ``sobol`` (scrambled Sobol with ``n_rand`` independent randomisations, standard error from their spread).
    Returns the estimate, its standard error, a 95% interval and the variance-reduction factor against plain sampling when it can be computed.
    """
    rng = np.random.default_rng(seed)
    if method == "plain":
        y = f(rng.standard_normal((n, dim)))
        est, se = y.mean(), y.std(ddof=1) / np.sqrt(n)
    elif method == "antithetic":
        z = rng.standard_normal((n // 2, dim))
        y = 0.5 * (f(z) + f(-z))
        est, se = y.mean(), y.std(ddof=1) / np.sqrt(len(y))
    elif method == "control":
        if control is None:
            raise ValueError("method='control' needs a control function")
        z = rng.standard_normal((n, dim))
        y, c = f(z), control(z)
        beta = np.cov(y, c)[0, 1] / c.var(ddof=1)
        adj = y - beta * (c - control_mean)
        est, se = adj.mean(), adj.std(ddof=1) / np.sqrt(n)
    elif method == "sobol":
        m = int(2 ** np.ceil(np.log2(max(n // n_rand, 2))))
        means = []
        for k in range(n_rand):
            u = qmc.Sobol(d=dim, scramble=True, seed=int(rng.integers(1 << 31))).random(m)
            means.append(f(sps.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))).mean())
        est, se = float(np.mean(means)), float(np.std(means, ddof=1) / np.sqrt(n_rand))
    else:
        raise ValueError("method must be plain, antithetic, control or sobol")
    return {"estimate": float(est), "se": float(se), "ci": (float(est - 1.96 * se), float(est + 1.96 * se)), "method": method}


def weighted_quantile(x, q, weights=None) -> np.ndarray:
    """Quantile(s) of a weighted sample (the importance-sampling estimate of a quantile)."""
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cum, x)


def effective_sample_size(weights) -> float:
    """Kish's effective sample size ``(sum w)^2 / sum w^2`` of importance weights; small relative to ``n`` means the proposal is poor."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


# --------------------------------------------------------------------------------------------------------- importance sampling
def importance_sampling(f: Callable, sampler: Callable, log_target: Callable, log_proposal: Callable, n: int = 100000, seed: int = 0, self_normalised: bool = False) -> dict:
    """``E_target[f(X)]`` using draws from the proposal. ``sampler(rng, n)`` draws, ``log_target``/``log_proposal`` evaluate the (log) densities.
    ``self_normalised=True`` allows densities known only up to a constant (biased but consistent)."""
    rng = np.random.default_rng(seed)
    x = sampler(rng, n)
    logw = log_target(x) - log_proposal(x)
    w = np.exp(logw - logw.max()) if self_normalised else np.exp(logw)
    fx = f(x)
    if self_normalised:
        est = float((w * fx).sum() / w.sum())
        se = float(np.sqrt(np.sum((w / w.sum()) ** 2 * (fx - est) ** 2)))
    else:
        y = w * fx
        est, se = float(y.mean()), float(y.std(ddof=1) / np.sqrt(n))
    return {"estimate": est, "se": se, "ess": effective_sample_size(w), "n": n}


def portfolio_tail_probability_is(weights, mean, cov, loss_level: float, n: int = 100000, seed: int = 0) -> dict:
    """``P(portfolio loss > loss_level)`` for Gaussian risk factors, loss ``= -w' R``, by importance sampling.

    The proposal shifts the mean of ``R`` to the most likely point on the failure boundary, ``m* = mean - Sigma w (loss_level + w' mean) / (w' Sigma w)``;
    the likelihood ratio is exact. For a rare event this has a far smaller standard error than plain sampling at the same ``n``, and the exact answer
    ``1 - Phi((loss_level + w' mean) / sqrt(w' Sigma w))`` is returned for checking.
    """
    w, mean, cov = np.asarray(weights, float), np.asarray(mean, float), np.asarray(cov, float)
    s2 = float(w @ cov @ w)
    shift = cov @ w * (loss_level + w @ mean) / s2
    proposal_mean = mean - shift
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)
    x = proposal_mean + rng.standard_normal((n, len(w))) @ L.T
    # log-likelihood ratio of N(mean, cov) to N(proposal_mean, cov): only the mean differs
    Ci = np.linalg.inv(cov)
    d_t, d_p = x - mean, x - proposal_mean
    logw = -0.5 * (np.einsum("ni,ij,nj->n", d_t, Ci, d_t) - np.einsum("ni,ij,nj->n", d_p, Ci, d_p))
    ind = (-(x @ w) > loss_level).astype(float)
    y = np.exp(logw) * ind
    exact = float(sps.norm.sf((loss_level + w @ mean) / np.sqrt(s2)))
    plain_z = rng.multivariate_normal(mean, cov, size=n)
    plain = ((-(plain_z @ w)) > loss_level).astype(float)
    return {"estimate": float(y.mean()), "se": float(y.std(ddof=1) / np.sqrt(n)), "exact": exact, "plain_estimate": float(plain.mean()),
            "plain_se": float(plain.std(ddof=1) / np.sqrt(n)), "ess": effective_sample_size(np.exp(logw)[ind > 0]) if ind.sum() else 0.0}


def cross_entropy_is(score: Callable[[np.ndarray], np.ndarray], level: float, dim: int, n: int = 20000, rho: float = 0.1, max_iter: int = 30, final_n: int = 100000,
                     seed: int = 0) -> dict:
    """Cross-entropy method (Rubinstein 1999) for ``P(score(Z) > level)``, ``Z ~ N(0, I)``: iteratively move the proposal mean toward the elite ``rho`` share
    of samples until the level itself is reached, then estimate with the learned proposal. Useful when the failure region is not linear."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(dim)
    for _ in range(max_iter):
        z = mu + rng.standard_normal((n, dim))
        s = score(z)
        gamma = min(level, float(np.quantile(s, 1.0 - rho)))
        elite = s >= gamma
        logw = -0.5 * (z ** 2).sum(axis=1) + 0.5 * ((z - mu) ** 2).sum(axis=1)
        w = np.exp(logw)[elite]
        mu = (w[:, None] * z[elite]).sum(axis=0) / w.sum()
        if gamma >= level:
            break
    z = mu + rng.standard_normal((final_n, dim))
    logw = -0.5 * (z ** 2).sum(axis=1) + 0.5 * ((z - mu) ** 2).sum(axis=1)
    y = np.exp(logw) * (score(z) > level)
    return {"estimate": float(y.mean()), "se": float(y.std(ddof=1) / np.sqrt(final_n)), "proposal_mean": mu, "ess": effective_sample_size(np.exp(logw)[score(z) > level]) if y.sum() else 0.0}
