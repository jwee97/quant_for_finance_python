"""A hierarchical Bayesian model of expected returns: partial pooling of assets within groups, and of groups toward the whole.

Sample means of returns are noisy, and the noise is of the same order as the differences between assets, so an optimiser that trusts them builds a portfolio on error. Shrinking every mean toward the grand
mean (Jorion's Bayes-Stein, already in :mod:`src.portfolio.bayesian`) helps, but it treats a bond and a stock as exchangeable. A *hierarchical* prior lets the data say how alike the assets in a group are,
and how alike the groups are::

    xbar | mu      ~  N(mu, V)                       V = Sigma / T       (the sample mean of T returns, with the covariance taken as known)
    mu_i           =  theta_g(i) + delta_i           delta_i ~ N(0, tau_a^2)    (an asset's deviation from its group)
    theta_g        =  m + eta_g                      eta_g   ~ N(0, tau_g^2)    (a group's deviation from the common mean m; m is flat)

Integrating out ``mu`` gives ``xbar ~ N(m 1, Omega + V)`` with ``Omega = tau_a^2 I + tau_g^2 G G'`` (``G`` is the asset-by-group membership matrix), so the two scales have a marginal likelihood in closed
form. Here the posterior over ``(tau_a, tau_g)`` is computed on a grid (flat on the log scale, which is the standard weakly informative choice), the exact conditional posterior of ``mu`` is Gaussian,
``mean = Omega (Omega + V)^-1 (xbar - m 1) + m 1``, ``cov = Omega - Omega (Omega + V)^-1 Omega``, and mixing over the grid gives the full posterior of the means without sampling chains. When the
groups are alike the group scale shrinks to zero and the model pools everything (as Bayes-Stein does); when assets within a group are alike but groups differ, it pools within groups and leaves the groups apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HierarchicalPosterior:
    assets: list
    mean: np.ndarray                 # posterior mean of mu, mixed over the scales
    grid: np.ndarray                 # (points, 2): tau_a, tau_g
    weights: np.ndarray              # posterior probability of each grid point
    means: np.ndarray                # (points, N): conditional posterior means
    covs: np.ndarray                 # (points, N, N): conditional posterior covariances
    common_mean: np.ndarray          # (points,): the posterior mean of m at each grid point

    @property
    def tau_a(self) -> float:
        return float(self.weights @ self.grid[:, 0])

    @property
    def tau_g(self) -> float:
        return float(self.weights @ self.grid[:, 1])

    def covariance(self) -> np.ndarray:
        """The posterior covariance of mu, including the uncertainty about the scales."""
        within = np.tensordot(self.weights, self.covs, axes=1)
        d = self.means - self.mean[None, :]
        between = (d * self.weights[:, None]).T @ d
        return within + between

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` draws of mu: a grid point by its posterior probability, then the Gaussian conditional."""
        which = rng.choice(len(self.weights), size=n, p=self.weights)
        out = np.empty((n, len(self.mean)))
        for k in np.unique(which):
            rows = np.flatnonzero(which == k)
            L = np.linalg.cholesky(self.covs[k] + 1e-12 * np.eye(len(self.mean)))
            out[rows] = self.means[k] + rng.normal(size=(len(rows), len(self.mean))) @ L.T
        return out


def membership(assets: list, groups: dict) -> np.ndarray:
    """The asset-by-group 0/1 matrix; an asset with no group is a group of its own."""
    labels = [groups.get(a, f"__alone_{a}") for a in assets]
    names = list(dict.fromkeys(labels))
    G = np.zeros((len(assets), len(names)))
    for i, lab in enumerate(labels):
        G[i, names.index(lab)] = 1.0
    return G


def hierarchical_posterior(xbar, cov, n_obs: int, groups: dict, assets: list | None = None, points: int = 11, span: tuple = (0.02, 4.0)) -> HierarchicalPosterior:
    """The posterior of the expected returns ``mu`` given the sample means ``xbar`` of ``n_obs`` observations, their covariance ``cov`` (of one observation) and the group of each asset.

    The scales ``tau_a`` and ``tau_g`` are searched on a ``points x points`` log grid from ``span[0]`` to ``span[1]`` times the cross-sectional standard deviation of ``xbar``, plus (nearly) zero, which means
    complete pooling."""
    xbar = np.asarray(xbar, dtype=float)
    assets = list(assets) if assets is not None else (list(xbar.index) if isinstance(xbar, pd.Series) else list(range(len(xbar))))
    n = len(xbar)
    S = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    if S.shape != (n, n) or n_obs < 2 or points < 3:
        raise ValueError("cov must match xbar, n_obs >= 2, points >= 3")
    V = S / n_obs
    if isinstance(groups, dict) and groups and not set(groups) & set(assets):
        raise ValueError("none of the assets appears in groups: pass assets=[...] (or a Series for xbar) named as in the groups")
    G = membership(assets, groups) if not isinstance(groups, np.ndarray) else groups
    GG = G @ G.T
    scale = float(np.std(xbar)) or float(np.sqrt(np.mean(np.diag(V))))
    taus = np.concatenate([[1e-4], np.geomspace(span[0], span[1], points)]) * scale
    s0 = 10.0 * scale + 10.0 * float(np.sqrt(np.mean(np.diag(V))))                    # a flat prior on the common mean
    one = np.ones(n)
    m0 = float(xbar.mean())
    grid, logml, means, covs, ms = [], [], [], [], []
    for ta in taus:
        for tg in taus:
            Omega = ta ** 2 * np.eye(n) + tg ** 2 * GG + s0 ** 2 * np.outer(one, one)
            M = Omega + V
            try:
                c = np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                continue
            r = xbar - m0 * one
            z = np.linalg.solve(c, r)
            logml.append(-0.5 * float(z @ z) - float(np.log(np.diag(c)).sum()))
            K = Omega @ np.linalg.inv(M)
            means.append(m0 * one + K @ r)
            covs.append(0.5 * ((Omega - K @ Omega) + (Omega - K @ Omega).T))
            grid.append((ta, tg))
            ms.append(float(m0 + s0 ** 2 * one @ np.linalg.solve(M, r)))
    logml = np.array(logml)
    w = np.exp(logml - logml.max())
    w /= w.sum()
    means, covs = np.array(means), np.array(covs)
    return HierarchicalPosterior(assets, w @ means, np.array(grid), w, means, covs, np.array(ms))


def hierarchical_means(window: pd.DataFrame, groups: dict, points: int = 9) -> tuple[pd.Series, HierarchicalPosterior]:
    """Posterior mean returns (per period) for the columns of a window of returns, using the sample covariance shrunk to its diagonal for stability."""
    clean = window.dropna(how="any")
    n = len(clean)
    S = np.cov(clean.to_numpy(), rowvar=False)
    S = 0.8 * S + 0.2 * np.diag(np.diag(S))
    post = hierarchical_posterior(clean.mean().to_numpy(), S, n, groups, list(clean.columns), points)
    return pd.Series(post.mean, index=clean.columns), post
