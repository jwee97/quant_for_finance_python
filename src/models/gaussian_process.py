"""Gaussian process regression: a prior over functions, updated by data, with an honest statement of how unsure it is.

A Gaussian process is a distribution over functions ``f`` such that any finite set of values ``f(x_1) ... f(x_n)`` is multivariate normal with mean zero and covariance ``k(x_i, x_j)``. Observing
``y = f(x) + noise`` with ``noise ~ N(0, s^2)``, the posterior at new points is normal as well, in closed form::

    mean(x*)  = k(x*, X) (K + s^2 I)^-1 y
    var(x*)   = k(x*, x*) - k(x*, X) (K + s^2 I)^-1 k(X, x*)

The kernel ``k`` is the whole model. A squared-exponential kernel says that nearby inputs have similar outputs and lets each input have its own length scale (*automatic relevance determination*: an input that does not
matter gets a long length scale and drops out); a Matern kernel allows rougher functions; a linear kernel makes the process a Bayesian linear regression; kernels add. The kernel's few hyperparameters are set by
maximising the log marginal likelihood ``-1/2 y'(K + s^2 I)^-1 y - 1/2 log|K + s^2 I| - n/2 log 2pi``, which balances fit against complexity without a validation set; its gradient is analytic here.

The cost is a Cholesky factorisation of an ``n x n`` matrix, so ``n`` of a few thousand is the limit; :class:`GaussianProcess` is exact, and callers that have more data keep a subsample.
The implementation is checked against scikit-learn's in the tests (same kernel, same hyperparameters, same answer to rounding).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

LOG_2PI = float(np.log(2.0 * np.pi))


def _scaled_sqdist(X: np.ndarray, Y: np.ndarray, ls: np.ndarray) -> np.ndarray:
    A, B = X / ls, Y / ls
    return np.maximum((A * A).sum(axis=1)[:, None] + (B * B).sum(axis=1)[None, :] - 2.0 * A @ B.T, 0.0)


class Kernel:
    """A covariance function with parameters kept on a log scale (so they stay positive and optimisation is well conditioned)."""

    def __call__(self, X, Y=None) -> np.ndarray:
        raise NotImplementedError

    def diag(self, X) -> np.ndarray:
        """``k(x, x)`` for every row of ``X``."""
        raise NotImplementedError

    def gradients(self, X) -> list:
        """``dK(X, X) / d theta_j`` for every log parameter."""
        raise NotImplementedError

    @property
    def theta(self) -> np.ndarray:
        raise NotImplementedError

    @theta.setter
    def theta(self, value) -> None:
        raise NotImplementedError

    @property
    def bounds(self) -> list:
        raise NotImplementedError

    def __add__(self, other: "Kernel") -> "Kernel":
        return SumKernel(self, other)

    def clone(self) -> "Kernel":
        import copy

        return copy.deepcopy(self)


class Stationary(Kernel):
    """``variance * g(r)`` of the scaled distance ``r``: ``rbf`` (squared exponential), ``matern32`` or ``matern52``. With ``ard`` there is a length scale for every input, else one for all."""

    def __init__(self, kind: str = "rbf", length_scale=1.0, variance: float = 1.0, ard: bool = False, dim: int | None = None):
        if kind not in ("rbf", "matern32", "matern52"):
            raise ValueError("kind must be rbf, matern32 or matern52")
        ls = np.atleast_1d(np.asarray(length_scale, dtype=float))
        if ard and dim is not None and len(ls) == 1:
            ls = np.full(dim, float(ls[0]))
        if (ls <= 0).any() or variance <= 0:
            raise ValueError("length scales and the variance must be positive")
        self.kind, self.ard = kind, bool(ard)
        self.length_scale, self.variance = ls if ard else ls[:1], float(variance)

    def _ls(self, d: int) -> np.ndarray:
        return self.length_scale if self.ard else np.full(d, self.length_scale[0])

    def _profile(self, r2: np.ndarray) -> np.ndarray:
        if self.kind == "rbf":
            return np.exp(-0.5 * r2)
        r = np.sqrt(r2)
        if self.kind == "matern32":
            return (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
        return (1.0 + np.sqrt(5.0) * r + 5.0 * r2 / 3.0) * np.exp(-np.sqrt(5.0) * r)

    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        return self.variance * self._profile(_scaled_sqdist(X, Y, self._ls(X.shape[1])))

    def diag(self, X):
        return np.full(len(np.atleast_2d(X)), self.variance)                              # every profile equals one at zero distance

    def gradients(self, X):
        X = np.atleast_2d(X)
        d = X.shape[1]
        ls = self._ls(d)
        r2 = _scaled_sqdist(X, X, ls)
        K = self.variance * self._profile(r2)
        if self.kind == "rbf":
            weight = K                                                                   # dK/dlog l_d = K * delta_d^2 / l_d^2
        else:
            r = np.sqrt(r2)
            if self.kind == "matern32":
                weight = 3.0 * self.variance * np.exp(-np.sqrt(3.0) * r)
            else:
                weight = (5.0 / 3.0) * self.variance * (1.0 + np.sqrt(5.0) * r) * np.exp(-np.sqrt(5.0) * r)
        per_dim = [weight * (X[:, [k]] - X[:, [k]].T) ** 2 / ls[k] ** 2 for k in range(d)]
        grads = [sum(per_dim)] if not self.ard else per_dim
        return grads + [K]                                                               # then dK/dlog variance = K

    @property
    def theta(self):
        return np.log(np.concatenate([self.length_scale, [self.variance]]))

    @theta.setter
    def theta(self, value):
        value = np.exp(np.asarray(value, dtype=float))
        self.length_scale, self.variance = value[:-1], float(value[-1])

    @property
    def bounds(self):
        return [(np.log(0.03), np.log(100.0))] * len(self.length_scale) + [(np.log(1e-4), np.log(1e3))]


class Linear(Kernel):
    """``variance * x'y``: a Bayesian linear regression with a Gaussian prior on the weights."""

    def __init__(self, variance: float = 1.0):
        if variance <= 0:
            raise ValueError("variance must be positive")
        self.variance = float(variance)

    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        return self.variance * X @ Y.T

    def diag(self, X):
        return self.variance * (np.atleast_2d(X) ** 2).sum(axis=1)

    def gradients(self, X):
        X = np.atleast_2d(X)
        return [self.variance * X @ X.T]

    @property
    def theta(self):
        return np.array([np.log(self.variance)])

    @theta.setter
    def theta(self, value):
        self.variance = float(np.exp(np.asarray(value, dtype=float)[0]))

    @property
    def bounds(self):
        return [(np.log(1e-4), np.log(1e3))]


class SumKernel(Kernel):
    def __init__(self, a: Kernel, b: Kernel):
        self.a, self.b = a, b

    def __call__(self, X, Y=None):
        return self.a(X, Y) + self.b(X, Y)

    def diag(self, X):
        return self.a.diag(X) + self.b.diag(X)

    def gradients(self, X):
        return self.a.gradients(X) + self.b.gradients(X)

    @property
    def theta(self):
        return np.concatenate([self.a.theta, self.b.theta])

    @theta.setter
    def theta(self, value):
        value = np.asarray(value, dtype=float)
        k = len(self.a.theta)
        self.a.theta, self.b.theta = value[:k], value[k:]

    @property
    def bounds(self):
        return self.a.bounds + self.b.bounds


def make_kernel(spec: str, dim: int) -> Kernel:
    """A kernel from a short name: ``rbf``, ``matern32``, ``matern52`` (each with a length scale per input), ``linear``, or sums such as ``rbf+linear``."""
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError("an empty kernel")
    kernels = []
    for p in parts:
        if p in ("rbf", "matern32", "matern52"):
            kernels.append(Stationary(p, 1.0, 1.0, ard=True, dim=dim))
        elif p == "linear":
            kernels.append(Linear(0.5))
        else:
            raise ValueError(f"unknown kernel '{p}': rbf, matern32, matern52, linear")
    out = kernels[0]
    for k in kernels[1:]:
        out = out + k
    return out


@dataclass
class _Fit:
    L: tuple
    alpha: np.ndarray
    X: np.ndarray


class GaussianProcess:
    """Exact GP regression with Gaussian noise. ``noise`` is the noise *variance* (after the targets are standardised if ``normalize_y``); ``optimize`` maximises the log marginal likelihood over the kernel's
    hyperparameters and the noise from the current values and from ``restarts`` random starts, and keeps the best."""

    def __init__(self, kernel: Kernel | None = None, noise: float = 0.1, normalize_y: bool = True, optimize: bool = True, restarts: int = 0, seed: int = 0, jitter: float = 1e-8):
        if noise <= 0 or restarts < 0 or jitter < 0:
            raise ValueError("noise > 0, restarts >= 0, jitter >= 0")
        self.kernel, self.noise, self.normalize_y, self.optimize, self.restarts, self.seed, self.jitter = kernel, float(noise), bool(normalize_y), bool(optimize), int(restarts), int(seed), float(jitter)
        self._fit: _Fit | None = None
        self.y_mean, self.y_scale = 0.0, 1.0

    # ------------------------------------------------------------------------------------------------------------------ likelihood
    def _factor(self, K: np.ndarray):
        n = len(K)
        jitter = self.jitter
        for _ in range(8):
            try:
                return cho_factor(K + jitter * np.eye(n), lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                jitter = max(jitter * 10.0, 1e-10)
        raise np.linalg.LinAlgError("the covariance matrix is not positive definite")

    def _objective(self, params: np.ndarray, X: np.ndarray, y: np.ndarray):
        """Negative log marginal likelihood and its gradient with respect to the log kernel parameters and the log noise."""
        k = len(self.kernel.theta)
        self.kernel.theta = params[:k]
        noise = float(np.exp(params[k]))
        K = self.kernel(X) + noise * np.eye(len(X))
        try:
            c = self._factor(K)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(params)
        alpha = cho_solve(c, y, check_finite=False)
        n = len(y)
        lml = -0.5 * y @ alpha - np.log(np.diag(c[0])).sum() - 0.5 * n * LOG_2PI
        Kinv = cho_solve(c, np.eye(n), check_finite=False)
        inner = np.outer(alpha, alpha) - Kinv
        grads = [0.5 * np.sum(inner * dK) for dK in self.kernel.gradients(X)]
        grads.append(0.5 * noise * np.trace(inner))                                       # d/dlog(noise) of K is noise * I
        return -float(lml), -np.array(grads)

    def log_marginal_likelihood(self, X=None, y=None) -> float:
        """The log marginal likelihood at the current hyperparameters (of the training data by default)."""
        if X is None:
            if self._fit is None:
                raise RuntimeError("fit the process first")
            X, y = self._fit.X, self._y_train
        return -self._objective(np.concatenate([self.kernel.theta, [np.log(self.noise)]]), np.atleast_2d(X), np.asarray(y, dtype=float))[0]

    # ------------------------------------------------------------------------------------------------------------------ fit and predict
    def fit(self, X, y) -> "GaussianProcess":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y) or len(X) < 2:
            raise ValueError("X and y need the same number (at least two) of rows")
        if self.kernel is None:
            self.kernel = make_kernel("rbf", X.shape[1])
        self.y_mean = float(y.mean()) if self.normalize_y else 0.0
        self.y_scale = float(y.std()) if self.normalize_y and y.std() > 0 else 1.0
        z = (y - self.y_mean) / self.y_scale
        if self.optimize:
            k = len(self.kernel.theta)
            bounds = self.kernel.bounds + [(np.log(1e-5), np.log(10.0))]
            start = np.clip(np.concatenate([self.kernel.theta, [np.log(self.noise)]]), [b[0] for b in bounds], [b[1] for b in bounds])
            starts = [start]
            rng = np.random.default_rng(self.seed)
            for _ in range(self.restarts):
                starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))
            best = None
            for s in starts:
                res = minimize(self._objective, s, args=(X, z), jac=True, method="L-BFGS-B", bounds=bounds)
                if best is None or res.fun < best.fun:
                    best = res
            self.kernel.theta = best.x[:k]
            self.noise = float(np.exp(best.x[k]))
        K = self.kernel(X) + self.noise * np.eye(len(X))
        c = self._factor(K)
        self._fit = _Fit(c, cho_solve(c, z, check_finite=False), X)
        self._y_train = z
        return self

    def predict(self, Xs, return_std: bool = False, return_cov: bool = False, include_noise: bool = False):
        """The posterior mean at ``Xs`` (and its standard deviation or covariance; ``include_noise`` adds the observation noise to get the spread of a new observation rather than of the function)."""
        if self._fit is None:
            raise RuntimeError("fit the process first")
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = self.kernel(Xs, self._fit.X)
        mean = Ks @ self._fit.alpha * self.y_scale + self.y_mean
        if not (return_std or return_cov):
            return mean
        v = cho_solve(self._fit.L, Ks.T, check_finite=False)
        if return_cov:
            cov = (self.kernel(Xs) - Ks @ v) * self.y_scale ** 2
            if include_noise:
                cov = cov + self.noise * self.y_scale ** 2 * np.eye(len(Xs))
            return mean, cov
        var = (self.kernel.diag(Xs) - np.einsum("ij,ji->i", Ks, v)) * self.y_scale ** 2
        if include_noise:
            var = var + self.noise * self.y_scale ** 2
        return mean, np.sqrt(np.maximum(var, 0.0))

    def sample_y(self, Xs, n: int = 1, seed: int = 0) -> np.ndarray:
        """``n`` functions drawn from the posterior at ``Xs`` (rows)."""
        mean, cov = self.predict(Xs, return_cov=True)
        rng = np.random.default_rng(seed)
        try:
            L = np.linalg.cholesky(cov + 1e-9 * self.y_scale ** 2 * np.eye(len(mean)))
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(cov)
            L = V * np.sqrt(np.maximum(w, 0.0))
        return mean[None, :] + rng.normal(size=(n, len(mean))) @ L.T

    @property
    def length_scales(self):
        """The length scales of the first stationary component (one per input with ARD): a long one means the output hardly depends on that input."""
        k = self.kernel
        while isinstance(k, SumKernel):
            k = k.a
        return getattr(k, "length_scale", None)
