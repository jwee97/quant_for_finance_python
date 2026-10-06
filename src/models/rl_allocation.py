"""A reinforcement-learning allocator trained by evolution strategies (Generation 5, Stage 38).

The policy is deliberately small and legible: ``score_i = theta . x_i + b_i`` over standardised asset features, ``weights = softmax(score)`` (long-only,
fully invested). Zero parameters is the equal-weight policy, so any tilt has to be earned. It is trained by the evolution-strategies gradient estimator
of Salimans et al. (2017) with antithetic sampling and centred-rank fitness, on a reward that is mean-variance utility net of linear trading costs. ES needs no gradient of the
simulator and no value function, which is why it suits a reward (a backtest) that is cheap to evaluate but awkward to differentiate.
"""

from __future__ import annotations

import numpy as np


def softmax_weights(X: np.ndarray, theta: np.ndarray, bias: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """``X``: (months, assets, features) -> (months, assets) weights. Non-finite features are treated as zero (the training mean after standardising).
    ``mask`` (months, assets) marks assets that exist; the others get exactly zero weight."""
    score = np.nan_to_num(X) @ theta + bias
    if mask is not None:
        score = np.where(mask, score, -np.inf)
    score = score - score.max(axis=-1, keepdims=True)
    e = np.exp(score)
    return e / e.sum(axis=-1, keepdims=True)


def utility(params: np.ndarray, X: np.ndarray, R: np.ndarray, n_features: int, risk_aversion: float, cost: float, mask: np.ndarray | None = None) -> float:
    """Sum over months of (net return - 0.5 * risk_aversion * net return^2). ``R`` is the next-month return of each asset; the first month pays costs from equal weight."""
    theta, bias = params[:n_features], params[n_features:]
    w = softmax_weights(X, theta, bias, mask)
    first = np.full((1, w.shape[1]), 1.0 / w.shape[1]) if mask is None else mask[:1] / mask[:1].sum()
    prev = np.vstack([first, w[:-1]])
    net = (w * R).sum(axis=1) - cost * np.abs(w - prev).sum(axis=1)
    return float(np.mean(net - 0.5 * risk_aversion * net ** 2))


def evolution_strategies(fitness, dim: int, seed: int, pairs: int = 32, sigma: float = 0.1, lr: float = 0.05, iterations: int = 60) -> tuple[np.ndarray, list[float]]:
    """Maximise ``fitness(params)`` starting from zero. Returns the final parameters and the fitness of the mean policy per iteration."""
    rng = np.random.default_rng(seed)
    params = np.zeros(dim)
    history = []
    for _ in range(iterations):
        eps = rng.normal(size=(pairs, dim))
        plus = np.array([fitness(params + sigma * e) for e in eps])
        minus = np.array([fitness(params - sigma * e) for e in eps])
        both = np.concatenate([plus, minus])
        ranks = np.empty(len(both))
        ranks[both.argsort()] = np.arange(len(both))
        centred = ranks / (len(both) - 1) - 0.5
        diff = centred[:pairs] - centred[pairs:]
        params = params + lr / (pairs * sigma) * (diff @ eps)
        history.append(float(fitness(params)))
    return params, history
