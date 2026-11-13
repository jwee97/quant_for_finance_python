"""A ridge regression on random Fourier features: the model of "The Virtue of Complexity in Return Prediction" (Kelly, Malamud and Zhou, Journal of Finance 2024).

The features are passed through ``components / 2`` random projections and each projection through a sine and a cosine. A ridge regression on those columns is a kernel regression with a Gaussian kernel,
computed at a cost that is linear in the rows. The paper's claim is that out-of-sample performance keeps improving as the number of random features grows past the number of observations, provided the
ridge penalty is kept: the shrinkage, not a small model, is what controls the variance. The claim is disputed (Nagel 2025 shows that with few observations such a model largely re-learns volatility-timed
momentum; Buncic 2025 traces part of the gain to design choices), which is a reason to run it beside a plain ridge and never alone. Both are in the catalogue for that.

Each projection is drawn as ``w ~ Normal(0, bandwidth^2 / d)`` for ``d`` standardised inputs, so ``w'x`` has a spread of about ``bandwidth`` whatever the number of inputs (the paper fixes 15 predictors and
tunes the bandwidth for them). ``draws`` independent sets of projections are fitted and their predictions averaged, which removes most of the luck of one draw.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

CHUNK = 8192


class RandomFeatureRidge(RegressorMixin, BaseEstimator):
    def __init__(self, components: int = 1024, bandwidth: float = 1.0, alpha: float = 100.0, draws: int = 3, random_state: int = 0):
        self.components = components
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.draws = draws
        self.random_state = random_state

    def _features(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        proj = X @ W
        return np.concatenate([np.sin(proj), np.cos(proj)], axis=1) / np.sqrt(W.shape[1])

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        half = int(self.components) // 2
        if half < 1 or float(self.bandwidth) <= 0 or float(self.alpha) < 0 or int(self.draws) < 1:
            raise ValueError("components must be at least 2, bandwidth positive, alpha non-negative and draws at least 1")
        n, d = X.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        rng = np.random.default_rng(int(self.random_state))
        self.projections_, self.coef_parts_, self.intercepts_ = [], [], []
        for _ in range(int(self.draws)):
            W = rng.normal(0.0, float(self.bandwidth) / np.sqrt(d), size=(d, half))
            gram, moment, mean_z, mean_y = np.zeros((2 * half, 2 * half)), np.zeros(2 * half), np.zeros(2 * half), float(np.average(y, weights=w))
            for i in range(0, n, CHUNK):
                mean_z += (self._features(X[i:i + CHUNK], W) * w[i:i + CHUNK, None]).sum(axis=0)
            mean_z /= w.sum()
            for i in range(0, n, CHUNK):                                                              # the Gram matrix is accumulated in pieces: the feature matrix of a large panel never exists whole
                Z = self._features(X[i:i + CHUNK], W) - mean_z
                Zw = Z * w[i:i + CHUNK, None]
                gram += Zw.T @ Z
                moment += Zw.T @ (y[i:i + CHUNK] - mean_y)
            beta = np.linalg.solve(gram + (float(self.alpha) + 1e-10) * np.eye(2 * half), moment)
            self.projections_.append(W)
            self.coef_parts_.append(beta)
            self.intercepts_.append(mean_y - float(mean_z @ beta))
        self.n_features_in_ = d
        self.complexity_ = 2 * half / n                                                               # the paper's c: random features per observation
        self.feature_importances_ = self._sensitivity(X[:: max(1, n // 512)][:512])
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(len(X))
        for W, beta, b0 in zip(self.projections_, self.coef_parts_, self.intercepts_):
            for i in range(0, len(X), CHUNK):
                out[i:i + CHUNK] += self._features(X[i:i + CHUNK], W) @ beta + b0
        return out / len(self.projections_)

    def _sensitivity(self, X: np.ndarray) -> np.ndarray:
        """Mean absolute derivative of the prediction with respect to each input over a sample of rows, as shares that sum to one: what the fitted function responds to."""
        grad = np.zeros_like(X)
        for W, beta in zip(self.projections_, self.coef_parts_):
            half = W.shape[1]
            proj = X @ W
            grad += ((np.cos(proj) * beta[:half] - np.sin(proj) * beta[half:]) / np.sqrt(half)) @ W.T
        size = np.abs(grad / len(self.projections_)).mean(axis=0)
        return size / (size.sum() + 1e-300)
