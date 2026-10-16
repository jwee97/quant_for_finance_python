"""Exponential Lévy models and the Heston model: characteristic functions, exact samplers, densities and maximum-likelihood fits.

The Black-Scholes model says a log price moves by a normal amount in any interval. Real returns have fatter tails and a skew, and an option market shows it as a smile.
An *exponential Lévy* model keeps the independent, stationary increments of Brownian motion and replaces the normal law by a richer one. The log price is a Lévy process ``L`` plus a
drift chosen so that the discounted price is a martingale::

    S_t = S_0 exp((r - q - w) t + L_t),       w = log E[exp(L_1)]            (the convexity correction)
    E[exp(i u L_t)] = exp(t psi(u))           psi is the characteristic exponent

Everything a pricer needs follows from ``psi``: the characteristic function of the log return ``X_T = ln(S_T / S_0)`` is ``exp(i u (r - q - w) T + T psi(u))``.

``BlackScholes``    ``psi(u) = -sigma^2 u^2 / 2``.
``Merton``          Brownian motion plus compound-Poisson jumps with normal sizes (Merton 1976): intensity ``lam``, jump mean ``mu_j`` and standard deviation ``sigma_j``.
``Kou``             jumps with a double-exponential size (Kou 2002): up jumps with probability ``p`` and rate ``eta1`` (which must exceed 1 so that the price has a mean), down jumps with rate ``eta2``.
``VarianceGamma``   Brownian motion with drift ``theta`` and volatility ``sigma`` run on a gamma clock of variance ``nu`` per unit time (Madan, Carr and Chang 1998): infinite activity,
                    no Brownian part needed, ``theta < 0`` gives the equity skew and ``nu`` the kurtosis.
``NIG``             the same with an inverse-Gaussian clock of variance ``kappa`` (Barndorff-Nielsen 1997): heavier tails than the variance gamma for the same variance.
``Heston``          stochastic variance (Heston 1993), not a Lévy process: only the characteristic function and, in ``probability.sde``, a simulator.

``Model.sample`` draws an exact increment of ``L`` over any interval (no discretisation error), so it serves both Monte Carlo pricing (risk-neutral) and forecasting (fitted to
returns, physical). ``Model.pdf`` is the density of an increment: closed-form for the normal, Merton, variance gamma and NIG, by Fourier inversion otherwise. ``fit_returns`` fits a model to a
series of returns by maximum likelihood and ``compare_fits`` ranks the models by the information criteria.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

_TOL_MARTINGALE = 1e-9


class Model:
    """A model of the log price with a known characteristic function. Subclasses set ``param_names``, ``bounds`` (for calibration and fitting) and ``start``."""

    name = "model"
    label = "model"
    param_names: tuple[str, ...] = ()
    bounds: dict[str, tuple[float, float]] = {}
    start: dict[str, float] = {}
    is_levy = True

    def __init__(self, **params):
        extra, missing = set(params) - set(self.param_names), set(self.param_names) - set(params)
        if extra or missing:
            raise TypeError(f"{self.name} takes the parameters {', '.join(self.param_names)} (unknown: {sorted(extra)}, missing: {sorted(missing)})")
        self.params = {k: float(params[k]) for k in self.param_names}
        self._check()

    def _check(self) -> None:
        raise NotImplementedError

    def replace(self, **changes) -> "Model":
        return type(self)(**{**self.params, **changes})

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v:.6g}' for k, v in self.params.items())})"

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.params == other.params

    def __hash__(self) -> int:
        return hash((type(self).__name__, tuple(self.params.items())))

    # ----------------------------------------------------------------------------------------------------------------------- characteristic function
    def psi(self, u) -> np.ndarray:
        """The characteristic exponent: ``log E[exp(i u L_1)]`` for complex ``u``."""
        raise NotImplementedError

    def omega(self) -> float:
        """``log E[exp(L_1)] = psi(-i)``: the convexity correction that makes the discounted price a martingale."""
        return float(np.real(self.psi(-1j)))

    def cf(self, u, T: float, r: float = 0.0, q: float = 0.0) -> np.ndarray:
        """The risk-neutral characteristic function of ``X_T = ln(S_T / S_0)``: ``exp(i u (r - q - w) T + T psi(u))``."""
        u = np.asarray(u, dtype=complex)
        return np.exp(1j * u * (r - q - self.omega()) * T + T * self.psi(u))

    def cf_centered(self, u, T: float, r: float = 0.0, q: float = 0.0) -> np.ndarray:
        """The characteristic function of ``ln(S_T / F_T)`` with ``F_T`` the forward. For a Lévy model it does not depend on the rate or the dividend yield (the arguments are accepted
        so that every model has the same signature)."""
        u = np.asarray(u, dtype=complex)
        return np.exp(T * (self.psi(u) - 1j * u * self.omega()))

    def exponent_limit(self) -> float:
        """The supremum of ``s`` for which ``E[exp(s L_1)]`` is finite. The price has a mean only if it exceeds 1, and a Fourier method that damps the call with ``exp(alpha k)``
        needs ``alpha + 1`` below it."""
        return float("inf")

    # --------------------------------------------------------------------------------------------------------------------------------- moments
    def cumulants(self) -> tuple[float, float, float, float]:
        """The first four cumulants of ``L_1``."""
        raise NotImplementedError

    def moments(self) -> dict:
        """Mean, variance, skewness and excess kurtosis of ``L_1`` (per unit time). The skewness and excess kurtosis of an increment over ``dt`` are these over ``sqrt(dt)`` and ``dt``."""
        k1, k2, k3, k4 = self.cumulants()
        return {"mean": k1, "variance": k2, "skewness": k3 / k2 ** 1.5, "excess_kurtosis": k4 / k2 ** 2}

    # ---------------------------------------------------------------------------------------------------------------------------------- sampling
    def sample(self, dt: float, size, rng: np.random.Generator) -> np.ndarray:
        """Exact draws of ``L_dt`` (the increment of the Lévy process over ``dt``, without the drift of the price)."""
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------------------------ density
    def pdf(self, x, dt: float) -> np.ndarray:
        """The density of ``L_dt`` at ``x`` (closed form where there is one, Fourier inversion of ``exp(dt psi)`` otherwise)."""
        return pdf_fft(self, dt, x)


def _positive(model: Model, *names: str) -> None:
    for n in names:
        if not model.params[n] > 0:
            raise ValueError(f"{model.name}: {n} must be positive")


class BlackScholes(Model):
    name, label = "bs", "Black-Scholes"
    param_names = ("sigma",)
    bounds = {"sigma": (0.01, 3.0)}
    start = {"sigma": 0.2}

    def _check(self):
        _positive(self, "sigma")

    def psi(self, u):
        u = np.asarray(u, dtype=complex)
        return -0.5 * self.params["sigma"] ** 2 * u ** 2

    def cumulants(self):
        return 0.0, self.params["sigma"] ** 2, 0.0, 0.0

    def sample(self, dt, size, rng):
        return self.params["sigma"] * np.sqrt(dt) * rng.standard_normal(size)

    def pdf(self, x, dt):
        return stats.norm.pdf(np.asarray(x, dtype=float), 0.0, self.params["sigma"] * np.sqrt(dt))


class Merton(Model):
    name, label = "merton", "Merton jump diffusion"
    param_names = ("sigma", "lam", "mu_j", "sigma_j")
    bounds = {"sigma": (0.01, 1.5), "lam": (0.01, 50.0), "mu_j": (-0.5, 0.3), "sigma_j": (0.005, 0.6)}
    start = {"sigma": 0.15, "lam": 1.0, "mu_j": -0.1, "sigma_j": 0.1}

    def _check(self):
        _positive(self, "sigma", "sigma_j")
        if self.params["lam"] < 0:
            raise ValueError("merton: lam must not be negative")

    def psi(self, u):
        u = np.asarray(u, dtype=complex)
        p = self.params
        return -0.5 * p["sigma"] ** 2 * u ** 2 + p["lam"] * (np.exp(1j * u * p["mu_j"] - 0.5 * p["sigma_j"] ** 2 * u ** 2) - 1.0)

    def cumulants(self):
        p = self.params
        m, s, lam = p["mu_j"], p["sigma_j"], p["lam"]
        return (lam * m, p["sigma"] ** 2 + lam * (m ** 2 + s ** 2), lam * (m ** 3 + 3 * m * s ** 2), lam * (m ** 4 + 6 * m ** 2 * s ** 2 + 3 * s ** 4))

    def sample(self, dt, size, rng):
        p = self.params
        n = rng.poisson(p["lam"] * dt, size)
        return p["sigma"] * np.sqrt(dt) * rng.standard_normal(size) + n * p["mu_j"] + np.sqrt(n) * p["sigma_j"] * rng.standard_normal(size)

    def pdf(self, x, dt):
        """A Poisson mixture of normals: ``sum_n Poisson(n; lam dt) N(x; n mu_j, sigma^2 dt + n sigma_j^2)``."""
        p = self.params
        x = np.asarray(x, dtype=float)
        lam_t = p["lam"] * dt
        n_max = int(min(max(lam_t + 12.0 * np.sqrt(lam_t) + 12, 20), 400))
        out = np.zeros_like(x)
        for n in range(n_max + 1):
            out += stats.poisson.pmf(n, lam_t) * stats.norm.pdf(x, n * p["mu_j"], np.sqrt(p["sigma"] ** 2 * dt + n * p["sigma_j"] ** 2))
        return out

    def levy_density(self, z):
        """The Lévy measure ``nu(dz) / dz``: ``lam`` times a normal density."""
        p = self.params
        return p["lam"] * stats.norm.pdf(np.asarray(z, dtype=float), p["mu_j"], p["sigma_j"])


class Kou(Model):
    name, label = "kou", "Kou double-exponential jumps"
    param_names = ("sigma", "lam", "p", "eta1", "eta2")
    bounds = {"sigma": (0.01, 1.5), "lam": (0.01, 50.0), "p": (0.01, 0.99), "eta1": (1.05, 200.0), "eta2": (1.0, 200.0)}
    start = {"sigma": 0.15, "lam": 3.0, "p": 0.4, "eta1": 15.0, "eta2": 12.0}

    def _check(self):
        _positive(self, "sigma", "eta2")
        p = self.params
        if p["lam"] < 0 or not 0.0 <= p["p"] <= 1.0:
            raise ValueError("kou: lam >= 0 and 0 <= p <= 1")
        if not p["eta1"] > 1.0:
            raise ValueError("kou: eta1 must exceed 1, or the price has no finite mean")

    def psi(self, u):
        u = np.asarray(u, dtype=complex)
        p = self.params
        jump = p["p"] * p["eta1"] / (p["eta1"] - 1j * u) + (1.0 - p["p"]) * p["eta2"] / (p["eta2"] + 1j * u) - 1.0
        return -0.5 * p["sigma"] ** 2 * u ** 2 + p["lam"] * jump

    def cumulants(self):
        p = self.params
        pr, e1, e2, lam = p["p"], p["eta1"], p["eta2"], p["lam"]
        m1 = pr / e1 - (1 - pr) / e2
        m2 = 2 * pr / e1 ** 2 + 2 * (1 - pr) / e2 ** 2
        m3 = 6 * pr / e1 ** 3 - 6 * (1 - pr) / e2 ** 3
        m4 = 24 * pr / e1 ** 4 + 24 * (1 - pr) / e2 ** 4
        return lam * m1, p["sigma"] ** 2 + lam * m2, lam * m3, lam * m4

    def sample(self, dt, size, rng):
        p = self.params
        shape = (size,) if np.isscalar(size) else tuple(size)
        n = rng.poisson(p["lam"] * dt, shape)
        total = int(n.sum())
        out = p["sigma"] * np.sqrt(dt) * rng.standard_normal(shape)
        if total:
            up = rng.random(total) < p["p"]
            jumps = np.where(up, rng.exponential(1.0 / p["eta1"], total), -rng.exponential(1.0 / p["eta2"], total))
            owner = np.repeat(np.arange(int(np.prod(shape))), n.ravel())
            out = out + np.bincount(owner, weights=jumps, minlength=int(np.prod(shape))).reshape(shape)
        return out

    def levy_density(self, z):
        p = self.params
        z = np.asarray(z, dtype=float)
        return p["lam"] * np.where(z > 0, p["p"] * p["eta1"] * np.exp(-p["eta1"] * z), (1.0 - p["p"]) * p["eta2"] * np.exp(p["eta2"] * z))

    def exponent_limit(self) -> float:
        return self.params["eta1"]


class VarianceGamma(Model):
    name, label = "vg", "Variance gamma"
    param_names = ("sigma", "theta", "nu")
    bounds = {"sigma": (0.01, 1.0), "theta": (-1.0, 1.0), "nu": (1e-4, 2.0)}
    start = {"sigma": 0.2, "theta": -0.1, "nu": 0.2}

    def _check(self):
        _positive(self, "sigma", "nu")

    def valid(self) -> bool:
        """Whether ``E[exp(L_1)]`` is finite: ``1 - theta nu - sigma^2 nu / 2 > 0``."""
        p = self.params
        return 1.0 - p["theta"] * p["nu"] - 0.5 * p["sigma"] ** 2 * p["nu"] > 0.0

    def psi(self, u):
        u = np.asarray(u, dtype=complex)
        p = self.params
        return -np.log(1.0 - 1j * u * p["theta"] * p["nu"] + 0.5 * p["sigma"] ** 2 * p["nu"] * u ** 2) / p["nu"]

    def cumulants(self):
        s, th, nu = self.params["sigma"], self.params["theta"], self.params["nu"]
        return th, s ** 2 + th ** 2 * nu, 3 * s ** 2 * th * nu + 2 * th ** 3 * nu ** 2, 3 * s ** 4 * nu + 12 * s ** 2 * th ** 2 * nu ** 2 + 6 * th ** 4 * nu ** 3

    def sample(self, dt, size, rng):
        p = self.params
        g = rng.gamma(dt / p["nu"], p["nu"], size)
        return p["theta"] * g + p["sigma"] * np.sqrt(g) * rng.standard_normal(size)

    def pdf(self, x, dt):
        """The closed form of Madan, Carr and Chang (1998), in terms of a modified Bessel function of the second kind."""
        s, th, nu = self.params["sigma"], self.params["theta"], self.params["nu"]
        x = np.asarray(x, dtype=float)
        t_nu = dt / nu
        ax = np.maximum(np.abs(x), 1e-12)
        c = 2.0 * s ** 2 / nu + th ** 2
        arg = np.sqrt(ax ** 2 * c) / s ** 2
        log_pdf = (np.log(2.0) + th * x / s ** 2 - t_nu * np.log(nu) - 0.5 * np.log(2.0 * np.pi) - np.log(s) - special.gammaln(t_nu) + (t_nu / 2.0 - 0.25) * np.log(ax ** 2 / c)
                   + np.log(np.maximum(special.kve(t_nu - 0.5, arg), 1e-300)) - arg)
        return np.exp(log_pdf)

    def exponent_limit(self) -> float:
        """The positive root of ``1 - theta nu s - sigma^2 nu s^2 / 2``."""
        s, th, nu = self.params["sigma"], self.params["theta"], self.params["nu"]
        return float((-th + np.sqrt(th ** 2 + 2.0 * s ** 2 / nu)) / s ** 2)

    def levy_density(self, z):
        """``nu(dz)/dz = exp(theta z / sigma^2) exp(-sqrt(2/nu + theta^2/sigma^2) |z| / sigma) / (nu |z|)``."""
        s, th, nu = self.params["sigma"], self.params["theta"], self.params["nu"]
        z = np.asarray(z, dtype=float)
        return np.exp(th * z / s ** 2 - np.sqrt(2.0 / nu + th ** 2 / s ** 2) * np.abs(z) / s) / (nu * np.abs(z))


class NIG(Model):
    name, label = "nig", "Normal inverse Gaussian"
    param_names = ("sigma", "theta", "kappa")
    bounds = {"sigma": (0.01, 1.0), "theta": (-1.0, 1.0), "kappa": (1e-4, 2.0)}
    start = {"sigma": 0.2, "theta": -0.1, "kappa": 0.2}

    def _check(self):
        _positive(self, "sigma", "kappa")

    def valid(self) -> bool:
        """Whether ``E[exp(L_1)]`` is finite: ``1 - 2 theta kappa - sigma^2 kappa > 0``."""
        p = self.params
        return 1.0 - 2.0 * p["theta"] * p["kappa"] - p["sigma"] ** 2 * p["kappa"] > 0.0

    def psi(self, u):
        u = np.asarray(u, dtype=complex)
        p = self.params
        return (1.0 - np.sqrt(1.0 - 2j * u * p["theta"] * p["kappa"] + p["sigma"] ** 2 * p["kappa"] * u ** 2)) / p["kappa"]

    def cumulants(self):
        s, th, k = self.params["sigma"], self.params["theta"], self.params["kappa"]
        return th, s ** 2 + th ** 2 * k, 3 * s ** 2 * th * k + 3 * th ** 3 * k ** 2, 3 * s ** 4 * k + 18 * s ** 2 * th ** 2 * k ** 2 + 15 * th ** 4 * k ** 3

    def sample(self, dt, size, rng):
        p = self.params
        clock = rng.wald(dt, dt ** 2 / p["kappa"], size)          # inverse Gaussian with mean dt and variance kappa dt
        return p["theta"] * clock + p["sigma"] * np.sqrt(clock) * rng.standard_normal(size)

    def classical_parameters(self) -> dict:
        """The usual ``(alpha, beta, delta)`` of the NIG law: ``alpha = sqrt(theta^2/sigma^4 + 1/(kappa sigma^2))``, ``beta = theta/sigma^2``, ``delta = sigma/sqrt(kappa)``."""
        s, th, k = self.params["sigma"], self.params["theta"], self.params["kappa"]
        return {"alpha": float(np.sqrt(th ** 2 / s ** 4 + 1.0 / (k * s ** 2))), "beta": th / s ** 2, "delta": s / np.sqrt(k)}

    def pdf(self, x, dt):
        """``alpha delta_t K_1(alpha sqrt(delta_t^2 + x^2)) exp(delta_t sqrt(alpha^2 - beta^2) + beta x) / (pi sqrt(delta_t^2 + x^2))``, with ``delta_t = delta dt``."""
        c = self.classical_parameters()
        a, b, d = c["alpha"], c["beta"], c["delta"] * dt
        x = np.asarray(x, dtype=float)
        root = np.sqrt(d ** 2 + x ** 2)
        gamma = np.sqrt(a ** 2 - b ** 2)
        log_pdf = np.log(a * d / np.pi) + np.log(np.maximum(special.kve(1.0, a * root), 1e-300)) - a * root - np.log(root) + d * gamma + b * x
        return np.exp(log_pdf)

    def exponent_limit(self) -> float:
        """The positive root of ``1 - 2 theta kappa s - sigma^2 kappa s^2``."""
        s, th, k = self.params["sigma"], self.params["theta"], self.params["kappa"]
        return float((-th + np.sqrt(th ** 2 + s ** 2 / k)) / s ** 2)

    def levy_density(self, z):
        c = self.classical_parameters()
        z = np.asarray(z, dtype=float)
        az = np.maximum(np.abs(z), 1e-300)
        return c["delta"] * c["alpha"] / np.pi * np.exp(c["beta"] * z) * special.kv(1.0, c["alpha"] * az) / az


class Heston(Model):
    """The Heston (1993) stochastic-variance model; only its characteristic function (in the Albrecher et al. 2007 form, free of branch cuts) lives here. ``probability.sde.simulate_heston``
    simulates it."""

    name, label = "heston", "Heston stochastic volatility"
    param_names = ("v0", "kappa", "theta", "xi", "rho")
    bounds = {"v0": (1e-4, 2.0), "kappa": (0.05, 20.0), "theta": (1e-4, 2.0), "xi": (0.05, 3.0), "rho": (-0.99, 0.99)}
    start = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7}
    is_levy = False

    def _check(self):
        _positive(self, "v0", "kappa", "theta", "xi")
        if not -1.0 < self.params["rho"] < 1.0:
            raise ValueError("heston: rho must be strictly between -1 and 1")

    @property
    def feller(self) -> bool:
        p = self.params
        return 2.0 * p["kappa"] * p["theta"] >= p["xi"] ** 2

    def psi(self, u):
        raise NotImplementedError("the Heston model is not a Levy process: use cf")

    def omega(self) -> float:
        return 0.0

    def cf(self, u, T, r=0.0, q=0.0):
        u = np.asarray(u, dtype=complex)
        p = self.params
        v0, kappa, theta, xi, rho = p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"]
        b = kappa - rho * xi * 1j * u
        d = np.sqrt(b ** 2 + xi ** 2 * (1j * u + u ** 2))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = (r - q) * 1j * u * T + kappa * theta / xi ** 2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
        D = (b - d) / xi ** 2 * (1.0 - e) / (1.0 - g * e)
        return np.exp(C + D * v0)

    def cf_centered(self, u, T, r=0.0, q=0.0):
        u = np.asarray(u, dtype=complex)
        return self.cf(u, T, r, q) * np.exp(-1j * u * (r - q) * T)

    def cumulants(self):
        raise NotImplementedError("the Heston model has no single set of cumulants: the distribution of a return depends on the horizon")

    def moments(self):
        raise NotImplementedError("see cumulants")

    def sample(self, dt, size, rng):
        raise NotImplementedError("use probability.sde.simulate_heston")


MODELS: dict[str, type[Model]] = {m.name: m for m in (BlackScholes, Merton, Kou, VarianceGamma, NIG, Heston)}


def create(name: str, **params) -> Model:
    """A model by its short name (``bs``, ``merton``, ``kou``, ``vg``, ``nig``, ``heston``)."""
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'; choose from {', '.join(MODELS)}")
    return MODELS[name](**params)


# ---------------------------------------------------------------------------------------------------------------------------------- the density
def pdf_fft(model: Model, dt: float, x, n: int = 2 ** 13, span_std: float = 14.0) -> np.ndarray:
    """The density of the increment ``L_dt`` by Fourier inversion of ``exp(dt psi(u))`` on a regular grid, interpolated to ``x``. The grid covers ``span_std`` standard deviations
    each side of the mean with ``n`` points. It exists for the models with no closed form (Kou) and to check the closed forms."""
    k1, k2, _, _ = model.cumulants()
    sd = np.sqrt(k2 * dt)
    half = span_std * sd
    dx = 2.0 * half / n
    centre = k1 * dt
    grid = centre + (np.arange(n) - n / 2) * dx
    du = 2.0 * np.pi / (n * dx)
    u = (np.arange(n) - n / 2) * du
    phi = np.exp(dt * model.psi(u) - 1j * u * centre)                      # the cf of the centred increment
    # p(y_j) = (du / 2 pi) sum_k phi(u_k) exp(-i u_k y_j), y_j = (j - n/2) dx
    shift = np.exp(1j * np.pi * np.arange(n))                               # (-1)^k from the half-grid offset
    dens = np.real(np.fft.fft(phi * shift) * shift) * du / (2.0 * np.pi)
    dens = np.maximum(dens, 0.0)
    x = np.asarray(x, dtype=float)
    return np.interp(x, grid, dens, left=0.0, right=0.0)


# ------------------------------------------------------------------------------------------------------------------------------- fitting
@dataclass
class ReturnsFit:
    """A model fitted to returns by maximum likelihood. ``drift`` is the annualised drift of the log price (the location, not part of the shape)."""

    model: Model
    drift: float
    loglik: float
    n: int
    k: int
    converged: bool
    message: str = ""

    @property
    def aic(self) -> float:
        return 2.0 * self.k - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        return self.k * np.log(self.n) - 2.0 * self.loglik

    def summary(self) -> dict:
        return {"model": self.model.name, "label": self.model.label, "params": dict(self.model.params), "drift": self.drift, "loglik": self.loglik, "aic": self.aic, "bic": self.bic,
                "n": self.n, "converged": self.converged}


def _loglik(model: Model, r: np.ndarray, dt: float, drift: float) -> float:
    d = model.pdf(r - drift * dt, dt)
    if not np.all(np.isfinite(d)):
        return -np.inf
    return float(np.sum(np.log(np.maximum(d, 1e-300))))


def fit_returns(returns, model: str | type[Model], dt: float = 1.0 / 252.0, starts: list[dict] | None = None, maxiter: int = 600) -> ReturnsFit:
    """Maximum-likelihood fit of a Lévy model's shape and a drift to a series of returns observed every ``dt`` years.

    The parameters are annualised (``sigma = 0.2`` is a 20% volatility, ``lam = 5`` is five jumps a year). The likelihood is the product of the densities of the increments over
    ``dt``, from the exact density of the model; each fit is started from a few moment-based points and the best is kept. ``returns`` are log returns.
    """
    cls = MODELS[model] if isinstance(model, str) else model
    if not cls.is_levy:
        raise ValueError("only Levy models can be fitted to a series of returns this way")
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 100:
        raise ValueError("at least 100 returns are needed")
    names = cls.param_names
    lo = np.array([cls.bounds[k][0] for k in names])
    hi = np.array([cls.bounds[k][1] for k in names])
    var_rate = float(np.var(r) / dt)
    ex_kurt = float(stats.kurtosis(r))
    sd_rate = np.sqrt(var_rate)
    base = dict(cls.start)
    proposals = list(starts or [])
    for scale in (0.9, 0.7):
        s = dict(base)
        if "sigma" in s:
            s["sigma"] = float(np.clip(sd_rate * np.sqrt(scale), *cls.bounds["sigma"]))
        if cls is VarianceGamma:
            s["nu"] = float(np.clip(max(ex_kurt, 0.3) * dt / 3.0 * 0.8, *cls.bounds["nu"]))
        if cls is NIG:
            s["kappa"] = float(np.clip(max(ex_kurt, 0.3) * dt / 3.0 * 0.8, *cls.bounds["kappa"]))
        if cls is Merton:
            s["lam"], s["sigma_j"] = 5.0 + 20.0 * (scale < 0.8), float(np.clip(sd_rate * 0.15, *cls.bounds["sigma_j"]))
        if cls is Kou:
            s["lam"] = 5.0 + 20.0 * (scale < 0.8)
        proposals.append(s)
    best = None
    mu0 = float(np.mean(r) / dt)
    for s in proposals:
        x0 = np.clip(np.array([s[k] for k in names] + [mu0]), np.append(lo, -5.0), np.append(hi, 5.0))

        def nll(z, names=names):
            try:
                model_ = cls(**dict(zip(names, z[:-1])))
            except ValueError:
                return 1e12
            if hasattr(model_, "valid") and not model_.valid():
                return 1e12
            v = _loglik(model_, r, dt, z[-1])
            return 1e12 if not np.isfinite(v) else -v / n

        res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=list(zip(np.append(lo, -5.0), np.append(hi, 5.0))), options={"maxiter": maxiter, "ftol": 1e-12})
        res2 = optimize.minimize(nll, res.x, method="Nelder-Mead", options={"maxiter": 2 * maxiter, "xatol": 1e-7, "fatol": 1e-11})
        z, fun = (res2.x, res2.fun) if res2.fun <= res.fun else (res.x, res.fun)
        z = np.clip(z, np.append(lo, -5.0), np.append(hi, 5.0))
        cand = (fun, z, bool(res.success or res2.success), str(res.message))
        if best is None or cand[0] < best[0]:
            best = cand
    fun, z, ok, msg = best
    model_ = cls(**dict(zip(names, z[:-1])))
    return ReturnsFit(model_, float(z[-1]), float(-fun * n), n, len(names) + 1, ok, msg)


def compare_fits(returns, models=("bs", "merton", "kou", "vg", "nig"), dt: float = 1.0 / 252.0) -> list[ReturnsFit]:
    """Fit each model and rank by AIC (lowest first): which shape of daily return the data prefer, after paying for the parameters."""
    fits = []
    for m in models:
        try:
            fits.append(fit_returns(returns, m, dt))
        except (ValueError, FloatingPointError):
            continue
    return sorted(fits, key=lambda f: f.aic)


# ----------------------------------------------------------------------------------------------------------------------------------- paths
def simulate_paths(model: Model, S0: float, r: float, q: float, T: float, steps: int, n_paths: int, seed=0, antithetic: bool = False) -> np.ndarray:
    """Risk-neutral paths of the spot under ``model``, shape ``(n_paths, steps + 1)``: for a Lévy model the exact increments (so any number of steps is exact at the grid dates), for the
    Heston model Andersen's QE scheme. ``antithetic`` mirrors the Gaussian draw of a normal-driven model (the other models ignore it)."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    if isinstance(model, Heston):
        from ..probability.sde import simulate_heston

        p = model.params
        return simulate_heston(S0, p["v0"], r, q, p["kappa"], p["theta"], p["xi"], p["rho"], T, steps, n_paths, "qe", rng, antithetic=antithetic)[0]
    drift = (r - q - model.omega()) * dt
    if antithetic and isinstance(model, BlackScholes):
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, steps))
        inc = model.params["sigma"] * np.sqrt(dt) * np.vstack([z, -z])[:n_paths]
    else:
        inc = model.sample(dt, (n_paths, steps), rng)
    logs = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(drift + inc, axis=1)], axis=1)
    return S0 * np.exp(logs)
