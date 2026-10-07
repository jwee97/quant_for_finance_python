"""Markov chains and Markov-switching models.

``MarkovChain`` holds a transition matrix ``P`` and answers the questions a regime or state model raises: the stationary distribution (long-run share of
time in each state), the expected duration of a state (``1 / (1 - p_ii)``), mean first-passage times between states, n-step probabilities and simulation.
``estimate_transition_matrix`` counts transitions (with optional Dirichlet smoothing and a posterior to show how uncertain a rare transition is).
``test_markov_order`` and ``runs_test`` test whether yesterday's state helps predict today's at all: for daily up/down returns it usually does not, which
is the honest baseline against which any regime model should be judged.

``fit_markov_switching`` wraps the Hamilton (1989) regime-switching regression from ``statsmodels`` and returns FILTERED probabilities (using data up to
``t``), which are usable in a backtest; the smoothed probabilities use the whole sample and are not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


class MarkovChain:
    def __init__(self, P, states=None):
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1] or not np.allclose(P.sum(axis=1), 1.0, atol=1e-8) or (P < -1e-12).any():
            raise ValueError("P must be a square row-stochastic matrix")
        self.P = P
        self.states = list(states) if states is not None else list(range(len(P)))

    @property
    def n(self) -> int:
        return len(self.P)

    def stationary(self) -> pd.Series:
        """The left eigenvector of ``P`` for eigenvalue 1, normalised to sum to one (unique for an irreducible chain)."""
        w, v = np.linalg.eig(self.P.T)
        i = int(np.argmin(np.abs(w - 1.0)))
        pi = np.real(v[:, i])
        pi = pi / pi.sum()
        return pd.Series(np.clip(pi, 0, None) / np.clip(pi, 0, None).sum(), index=self.states)

    def n_step(self, k: int) -> pd.DataFrame:
        return pd.DataFrame(np.linalg.matrix_power(self.P, k), index=self.states, columns=self.states)

    def expected_duration(self) -> pd.Series:
        """Mean number of consecutive periods in each state: ``1 / (1 - p_ii)``."""
        with np.errstate(divide="ignore"):
            return pd.Series(1.0 / (1.0 - np.diag(self.P)), index=self.states)

    def mean_first_passage(self) -> pd.DataFrame:
        """``M[i, j]``: expected steps to reach ``j`` for the first time starting from ``i`` (``M[j, j]`` is the mean return time ``1 / pi_j``)."""
        n = self.n
        M = np.zeros((n, n))
        for j in range(n):
            keep = [i for i in range(n) if i != j]
            Q = self.P[np.ix_(keep, keep)]
            M[keep, j] = np.linalg.solve(np.eye(n - 1) - Q, np.ones(n - 1))
            M[j, j] = 1.0 / self.stationary().iloc[j]
        return pd.DataFrame(M, index=self.states, columns=self.states)

    def mixing_time(self, eps: float = 0.01) -> int:
        """Steps until every row of ``P^k`` is within total-variation distance ``eps`` of the stationary distribution."""
        pi = self.stationary().to_numpy()
        Pk = np.eye(self.n)
        for k in range(1, 10000):
            Pk = Pk @ self.P
            if 0.5 * np.abs(Pk - pi).sum(axis=1).max() < eps:
                return k
        return 10000

    def simulate(self, n: int, start: int = 0, seed=0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = np.empty(n, dtype=int)
        out[0] = start
        cum = np.cumsum(self.P, axis=1)
        u = rng.random(n)
        for t in range(1, n):
            out[t] = int(np.searchsorted(cum[out[t - 1]], u[t]))
        return np.array(self.states)[out] if self.states != list(range(self.n)) else out


def transition_counts(labels, states=None) -> pd.DataFrame:
    s = pd.Series(np.asarray(labels))
    states = list(states) if states is not None else sorted(s.unique())
    c = pd.crosstab(s.iloc[:-1].to_numpy(), s.iloc[1:].to_numpy()).reindex(index=states, columns=states, fill_value=0)
    return c


def estimate_transition_matrix(labels, states=None, smoothing: float = 0.0) -> MarkovChain:
    """Maximum-likelihood transition matrix from a label sequence; ``smoothing`` adds a Dirichlet pseudo-count to every cell (Laplace = 1)."""
    c = transition_counts(labels, states) + smoothing
    P = c.div(c.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / c.shape[1])
    return MarkovChain(P.to_numpy(), list(c.index))


def transition_posterior(labels, states=None, prior: float = 1.0, n_draws: int = 2000, seed=0) -> np.ndarray:
    """Posterior draws of the transition matrix: each row ``~ Dirichlet(prior + counts)``. Returns ``(draws, n, n)``; the spread shows how little a rare transition is known."""
    c = transition_counts(labels, states).to_numpy() + prior
    rng = np.random.default_rng(seed)
    return np.stack([np.stack([rng.dirichlet(row) for row in c]) for _ in range(n_draws)])


def test_markov_order(labels) -> dict:
    """Likelihood-ratio test of a first-order chain against independent draws (zeroth order). H0: the next state does not depend on the current one."""
    c = transition_counts(labels).to_numpy().astype(float)
    n = c.sum()
    p0 = c.sum(axis=0) / n
    ll0 = float(np.sum(c * np.log(np.where(p0 > 0, p0, 1.0))))
    P = c / np.maximum(c.sum(axis=1, keepdims=True), 1)
    ll1 = float(np.sum(c * np.log(np.where(P > 0, P, 1.0))))
    k = c.shape[0]
    stat = 2.0 * (ll1 - ll0)
    df = (k - 1) ** 2
    return {"statistic": stat, "df": df, "pvalue": float(sps.chi2.sf(stat, df))}


test_markov_order.__test__ = False                       # not a pytest test


def runs_test(signs) -> dict:
    """Wald-Wolfowitz runs test on a binary sequence. Too few runs = persistence (momentum), too many = reversal."""
    s = np.asarray(signs).astype(int)
    n1, n0 = int(s.sum()), int((1 - s).sum())
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    n = n0 + n1
    mean = 2.0 * n0 * n1 / n + 1.0
    var = 2.0 * n0 * n1 * (2.0 * n0 * n1 - n) / (n ** 2 * (n - 1.0))
    z = (runs - mean) / np.sqrt(var)
    return {"runs": runs, "expected": float(mean), "z": float(z), "pvalue": float(2 * sps.norm.sf(abs(z)))}


def discretize(x, n_states: int = 3, edges=None) -> np.ndarray:
    """Quantile bins of a series (``edges`` fixes the cut points, e.g. ones computed on a training window)."""
    x = np.asarray(x, dtype=float)
    cuts = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, n_states + 1)[1:-1]) if edges is None else np.asarray(edges)
    return np.searchsorted(cuts, x)


def fit_markov_switching(returns: pd.Series, k_regimes: int = 2, switching_variance: bool = True, order: int = 0, trend: str = "c"):
    """Hamilton (1989) Markov-switching mean (and variance) model via ``statsmodels``. Returns ``{'result', 'filtered', 'smoothed', 'transition', 'durations'}``
    where ``filtered`` is causal (uses data up to each date) and ``smoothed`` is not."""
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    y = returns.dropna()
    if order:
        model = MarkovAutoregression(y, k_regimes=k_regimes, order=order, switching_ar=False, switching_variance=switching_variance, trend=trend)
    else:
        model = MarkovRegression(y, k_regimes=k_regimes, trend=trend, switching_variance=switching_variance)
    res = model.fit(disp=False)
    filt = pd.DataFrame(res.filtered_marginal_probabilities, index=y.index[-len(res.filtered_marginal_probabilities):])
    smooth = pd.DataFrame(res.smoothed_marginal_probabilities, index=filt.index)
    P = np.asarray(res.regime_transition)[..., 0] if np.asarray(res.regime_transition).ndim == 3 else np.asarray(res.regime_transition)
    chain = MarkovChain(P.T if P.shape[0] == k_regimes else P)
    return {"result": res, "filtered": filt, "smoothed": smooth, "transition": chain, "durations": pd.Series(res.expected_durations)}
