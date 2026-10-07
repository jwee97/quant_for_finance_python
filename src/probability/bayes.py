"""Bayesian inference for research questions: how sure are we, given this much data?

* ``bayes_linear_regression``: the conjugate normal-inverse-gamma posterior for a regression, its Student-t predictive distribution and the log marginal
  likelihood (for comparing models without a hold-out set).
* ``bayesian_sharpe`` / ``compare_sharpe``: the posterior distribution of a Sharpe ratio (and of the difference between two strategies) from a Student-t
  likelihood sampled by MCMC, which answers "what is the probability this strategy's true Sharpe is positive / beats the benchmark?" directly.
* ``beta_binomial``: the posterior of a hit rate with a Beta prior.
* ``metropolis_hastings``: a random-walk Metropolis sampler with adaptive proposal scale and the diagnostics that decide whether to trust it: split R-hat
  (Gelman-Rubin) and effective sample size; ``hpd_interval`` gives highest-posterior-density intervals.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import special, stats as sps


# ------------------------------------------------------------------------------------------------------------------------ MCMC
def metropolis_hastings(log_posterior: Callable[[np.ndarray], float], x0, n_samples: int = 5000, n_chains: int = 4, burn: int = 1000, scale: float = 0.5, adapt: bool = True,
                        seed: int = 0) -> dict:
    """Random-walk Metropolis. Several chains start from dispersed points around ``x0``; during burn-in each chain's proposal covariance is adapted toward
    ``2.38^2 / d`` times the sample covariance (Roberts-Gelman-Gilks). Returns the post-burn chains ``(chains, samples, dim)``, acceptance rates, R-hat and ESS."""
    rng = np.random.default_rng(seed)
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    d = len(x0)
    chains = np.empty((n_chains, n_samples, d))
    acc = np.zeros(n_chains)
    for c in range(n_chains):
        x = x0 + 0.1 * scale * rng.standard_normal(d)
        lp = log_posterior(x)
        cov = (scale ** 2) * np.eye(d) / d
        hist = []
        for t in range(burn + n_samples):
            prop = rng.multivariate_normal(x, cov)
            lq = log_posterior(prop)
            if np.log(rng.random()) < lq - lp:
                x, lp = prop, lq
                if t >= burn:
                    acc[c] += 1
            hist.append(x)
            if adapt and t < burn and t >= 100 and t % 50 == 0:
                h = np.array(hist[t // 2:])
                cov = (2.38 ** 2 / d) * (np.cov(h.T).reshape(d, d) + 1e-9 * np.eye(d))
            if t >= burn:
                chains[c, t - burn] = x
    return {"chains": chains, "acceptance": acc / n_samples, "rhat": rhat(chains), "ess": ess(chains), "samples": chains.reshape(-1, d)}


def rhat(chains: np.ndarray) -> np.ndarray:
    """Split R-hat (Vehtari et al. 2021): chains are halved and the between/within variance ratio compared. Values above ~1.01 mean the chains have not mixed."""
    m, n, d = chains.shape
    half = n // 2
    split = np.concatenate([chains[:, :half], chains[:, half: 2 * half]], axis=0)
    mm, nn = split.shape[0], split.shape[1]
    means, var = split.mean(axis=1), split.var(axis=1, ddof=1)
    W, B = var.mean(axis=0), nn * means.var(axis=0, ddof=1)
    return np.sqrt(((nn - 1) / nn * W + B / nn) / W)


def ess(chains: np.ndarray) -> np.ndarray:
    """Effective sample size from the autocorrelation of the pooled chains (initial positive sequence of Geyer 1992)."""
    m, n, d = chains.shape
    out = np.empty(d)
    for j in range(d):
        x = chains[:, :, j] - chains[:, :, j].mean(axis=1, keepdims=True)
        var = x.var(axis=1, ddof=0).mean()
        rho = []
        for k in range(1, n // 2):
            rho_k = np.mean([(x[c, k:] @ x[c, :-k]) / (n * var) for c in range(m)])
            if rho_k < 0.0:
                break
            rho.append(rho_k)
        out[j] = m * n / (1.0 + 2.0 * np.sum(rho))
    return out


def hpd_interval(samples, mass: float = 0.95) -> tuple[float, float]:
    """Narrowest interval containing ``mass`` of the posterior draws."""
    s = np.sort(np.asarray(samples, dtype=float))
    k = int(np.ceil(mass * len(s)))
    widths = s[k - 1:] - s[: len(s) - k + 1]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + k - 1])


# --------------------------------------------------------------------------------------------------------------------- regression
def bayes_linear_regression(y, X, prior_precision: float = 1e-6, a0: float = 1e-3, b0: float = 1e-3, prior_mean=None, add_constant: bool = True) -> dict:
    """Conjugate Bayesian regression ``y = X b + e``, ``e ~ N(0, s2)``, ``b | s2 ~ N(m0, s2 V0)``, ``s2 ~ InvGamma(a0, b0)`` with ``V0 = I / prior_precision``.

    Returns the posterior mean and scale matrix of ``b`` (a multivariate Student-t with ``2 a_n`` degrees of freedom), the inverse-gamma posterior of ``s2``, the
    log marginal likelihood (higher = better supported model, and automatically penalises complexity), and a ``predict`` function giving the Student-t predictive
    mean and scale.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    if add_constant:
        X = np.column_stack([np.ones(len(y)), X])
    n, k = X.shape
    m0 = np.zeros(k) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    V0i = prior_precision * np.eye(k)
    Vni = V0i + X.T @ X
    Vn = np.linalg.inv(Vni)
    mn = Vn @ (V0i @ m0 + X.T @ y)
    an = a0 + n / 2.0
    bn = b0 + 0.5 * (y @ y + m0 @ V0i @ m0 - mn @ Vni @ mn)
    logml = (special.gammaln(an) - special.gammaln(a0) + a0 * np.log(b0) - an * np.log(bn) + 0.5 * (np.linalg.slogdet(Vn)[1] + k * np.log(prior_precision))
             - 0.5 * n * np.log(2 * np.pi))
    scale = bn / an * Vn

    def predict(Xnew):
        Xn = np.asarray(Xnew, dtype=float).reshape(-1, k - int(add_constant))
        if add_constant:
            Xn = np.column_stack([np.ones(len(Xn)), Xn])
        mean = Xn @ mn
        sd = np.sqrt(bn / an * (1.0 + np.einsum("ij,jk,ik->i", Xn, Vn, Xn)))
        return mean, sd, 2 * an

    return {"mean": mn, "scale": scale, "df": 2 * an, "sd": np.sqrt(np.diag(scale) * (2 * an) / (2 * an - 2)) if an > 1 else np.full(k, np.nan), "a_n": an, "b_n": bn,
            "sigma2_mean": float(bn / (an - 1)) if an > 1 else float("nan"), "log_marginal_likelihood": float(logml), "predict": predict}


def bayes_factor(logml_a: float, logml_b: float) -> float:
    """``exp(log ML_a - log ML_b)``: above 3 is positive, above 20 strong evidence for model A (Kass & Raftery 1995)."""
    return float(np.exp(logml_a - logml_b))


# ------------------------------------------------------------------------------------------------------------------------ Sharpe
def bayesian_sharpe(returns, n_samples: int = 4000, annualisation: float = 252.0, seed: int = 0, burn: int = 2500) -> dict:
    """Posterior of the Sharpe ratio under a Student-t likelihood (heavy tails) with weak priors, sampled by MCMC on ``(mu, log sigma, log (nu - 1))``.

    The returns are standardised first so the sampler works on an O(1) scale whatever the units. Priors: ``mu ~ N(0, 10^2)`` and ``log sigma ~ N(0, 5^2)`` on
    the standardised scale, ``nu - 1 ~ Exponential(mean 29)``. Returns the draws of the ANNUALISED Sharpe ratio, its HPD interval, ``P(Sharpe > 0)`` and the
    convergence diagnostics (check ``rhat < 1.05`` and ``ess``).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    m0, s0 = r.mean(), r.std(ddof=1)
    z = (r - m0) / s0

    def logpost(p):
        mu, ls, lnu = p
        if not -3 < lnu < 6 or not -5 < ls < 5:
            return -np.inf
        nu = 1.0 + np.exp(lnu)
        return sps.t.logpdf(z, nu, mu, np.exp(ls)).sum() + sps.norm.logpdf(mu, 0, 10.0) + sps.norm.logpdf(ls, 0.0, 5.0) + sps.expon.logpdf(nu - 1.0, scale=29.0) + lnu

    out = metropolis_hastings(logpost, [0.0, 0.0, np.log(29.0)], n_samples=n_samples // 4 + 1, n_chains=4, burn=burn, scale=0.15, seed=seed)
    d = out["samples"]
    mu = d[:, 0] * s0 + m0
    sigma = np.exp(d[:, 1]) * s0
    sr = mu / sigma * np.sqrt(annualisation)
    return {"draws": sr, "mean": float(sr.mean()), "median": float(np.median(sr)), "hpd95": hpd_interval(sr), "prob_positive": float((sr > 0).mean()), "rhat": out["rhat"],
            "ess": out["ess"], "nu_median": float(1.0 + np.median(np.exp(d[:, 2])))}


def compare_sharpe(a, b, n_samples: int = 4000, annualisation: float = 252.0, seed: int = 0) -> dict:
    """Posterior of ``Sharpe(a) - Sharpe(b)`` for two return series on the SAME dates (both posteriors are independent draws; paired dependence is ignored,
    so this is conservative when the strategies are positively correlated)."""
    pa = bayesian_sharpe(a, n_samples, annualisation, seed)
    pb = bayesian_sharpe(b, n_samples, annualisation, seed + 1)
    k = min(len(pa["draws"]), len(pb["draws"]))
    diff = pa["draws"][:k] - pb["draws"][:k]
    return {"diff_draws": diff, "mean": float(diff.mean()), "hpd95": hpd_interval(diff), "prob_a_better": float((diff > 0).mean())}


def beta_binomial(successes: int, trials: int, prior=(1.0, 1.0), mass: float = 0.95) -> dict:
    """Posterior of a probability (hit rate) with a ``Beta(a, b)`` prior: ``Beta(a + k, b + n - k)``, with its mean, credible interval and ``P(p > 0.5)``."""
    a, b = prior[0] + successes, prior[1] + trials - successes
    d = sps.beta(a, b)
    lo, hi = d.ppf([(1 - mass) / 2, 1 - (1 - mass) / 2])
    return {"a": a, "b": b, "mean": float(d.mean()), "interval": (float(lo), float(hi)), "prob_above_half": float(d.sf(0.5))}
