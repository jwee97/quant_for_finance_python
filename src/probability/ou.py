r"""The Ornstein-Uhlenbeck process: simulation, estimation, hitting times and the best levels to trade a spread that follows it.

A spread that is pulled back to a level ``mu`` at speed ``theta`` while noise of size ``sigma`` pushes it away is an Ornstein-Uhlenbeck (OU) process::

    dX = -theta (X - mu) dt + sigma dW             X_{t+dt} = mu + (X_t - mu) e^{-theta dt} + sigma sqrt((1 - e^{-2 theta dt}) / (2 theta)) Z

It is stationary (mean ``mu``, standard deviation ``sigma / sqrt(2 theta)``), and a sample taken every ``dt`` is an AR(1) series with coefficient ``phi = e^{-theta dt}``. Everything below is exact
for the process, none of it an Euler approximation.

``simulate_ou``        exact paths.
``fit_ou``             the maximum-likelihood fit from the exact transition density, which is an ordinary regression of ``X_{t+1}`` on ``X_t`` followed by the map ``theta = -ln(phi) / dt``,
                       ``mu = c / (1 - phi)``, ``sigma^2 = 2 theta s^2 / (1 - phi^2)`` (``s^2`` the residual variance); the standard errors follow by the delta method. The usual shortcut
                       ``theta = (1 - phi) / dt`` is the Euler value and is biased when ``theta dt`` is not small.
``half_life``          ``ln 2 / theta``, the time for a deviation to halve in expectation.
``dickey_fuller_t``    the t statistic that tells a mean-reverting series from a random walk (with the right critical values: -2.86 for a series, -3.4 for a pair's spread).
``hitting_time``       the expected time to first reach a level, ``(sqrt(pi) / theta) * Int erfcx(-w) dw`` over the standardised path, and ``hitting_time_moments`` its standard deviation too
                       (from the second moment, by the same method).
``cycle_time``         the expected time for the spread to go from a lower level to an upper one and back, ``(pi / theta) (erfi(w_m) - erfi(w_a))`` (Bertram 2010).
``optimal_thresholds`` the levels at which to buy and sell to earn the most per unit of time when each round trip costs ``c`` (Bertram 2010). A trade from ``mu - d`` to ``mu + d`` earns
                       ``2d - c`` and the cycle takes ``(2 pi / theta) erfi(d sqrt(theta) / sigma)``; for a fixed width the cycle is shortest when the levels are symmetric about ``mu`` (the
                       exponential in ``erfi'`` is convex), so the problem is one-dimensional in ``d``. Waiting for a wide gap earns more per trade but trades less often: the optimum is where
                       the two balance, and it moves out as the cost rises.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize, signal, special, stats

from .sde import FitResult, _ci


def _rng(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


# --------------------------------------------------------------------------------------------------------------------------------- simulation
def simulate_ou(x0: float, theta: float, mu: float, sigma: float, horizon: float, steps: int, n_paths: int = 1, seed=0) -> np.ndarray:
    """Exact paths of the OU process, shape ``(n_paths, steps + 1)``: each step is drawn from the exact transition law, so any step size is exact at the grid dates."""
    if theta <= 0 or sigma <= 0 or steps < 1 or horizon <= 0:
        raise ValueError("theta, sigma and the horizon must be positive and there must be a step")
    rng = _rng(seed)
    dt = horizon / steps
    phi = np.exp(-theta * dt)
    sd = sigma * np.sqrt((1.0 - phi ** 2) / (2.0 * theta))
    z = rng.standard_normal((n_paths, steps))
    out = np.empty((n_paths, steps + 1))
    out[:, 0] = x0
    # y_{t+1} = phi y_t + sd z_t is a first-order recursive filter: the loop runs in compiled code, whatever the number of steps
    out[:, 1:] = mu + signal.lfilter([1.0], [1.0, -phi], sd * z, axis=1, zi=np.full((n_paths, 1), phi * (x0 - mu)))[0]
    return out


def stationary(theta: float, mu: float, sigma: float) -> tuple[float, float]:
    """The mean and standard deviation of the stationary distribution."""
    return float(mu), float(sigma / np.sqrt(2.0 * theta))


def half_life(theta: float) -> float:
    return float(np.log(2.0) / theta)


# ------------------------------------------------------------------------------------------------------------------------------------ fitting
def ou_loglik(params, x: np.ndarray, dt: float) -> float:
    """The exact conditional log-likelihood ``Sum log p(x_{t+1} | x_t)`` of a sample under ``(theta, mu, sigma)``."""
    theta, mu, sigma = params
    if theta <= 0 or sigma <= 0:
        return -np.inf
    phi = np.exp(-theta * dt)
    var = sigma ** 2 * (1.0 - phi ** 2) / (2.0 * theta)
    resid = x[1:] - mu - (x[:-1] - mu) * phi
    return float(np.sum(stats.norm.logpdf(resid, 0.0, np.sqrt(var))))


def fit_ou(x, dt: float = 1.0 / 252.0) -> FitResult:
    """The maximum-likelihood fit of ``(theta, mu, sigma)`` to a series observed every ``dt`` (a log spread, say). Raises ``ValueError`` when the series shows no mean reversion
    (the autoregressive coefficient is at least one: the maximum likelihood estimate of ``theta`` is then zero) or none of the persistence an OU process needs (coefficient not positive).
    The standard errors are those of the regression carried through the map to ``(theta, mu, sigma)`` by the delta method."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x) - 1
    if n < 20:
        raise ValueError("an OU fit needs at least 21 observations")
    y, z = x[1:], x[:-1]
    zc = z - z.mean()
    sxx = float(zc @ zc)
    if sxx <= 0:
        raise ValueError("the series is constant")
    phi = float(zc @ (y - y.mean()) / sxx)
    if phi >= 1.0:
        raise ValueError("no mean reversion: the autoregressive coefficient is at least one (a random walk or an explosive series)")
    if phi <= 0.0:
        raise ValueError("no persistence: the autoregressive coefficient is not positive, which an Ornstein-Uhlenbeck process sampled this slowly cannot produce")
    c = float(y.mean() - phi * z.mean())
    resid = y - c - phi * z
    s2 = float(resid @ resid / n)
    theta = -np.log(phi) / dt
    mu = c / (1.0 - phi)
    sigma2 = 2.0 * theta * s2 / (1.0 - phi ** 2)
    sigma = float(np.sqrt(sigma2))
    # delta method: the regression gives Cov(c, phi) = s2 (X'X)^-1 and Var(s2) = 2 s2^2 / (n - 2), independent of each other
    cov_cp = s2 * np.linalg.inv(np.array([[n, z.sum()], [z.sum(), z @ z]]))
    var_s2 = 2.0 * s2 ** 2 / max(n - 2, 1)
    dtheta_dphi = -1.0 / (phi * dt)
    dsig2_dphi = 2.0 * s2 * (dtheta_dphi * (1.0 - phi ** 2) + theta * 2.0 * phi) / (1.0 - phi ** 2) ** 2
    dsig2_ds2 = 2.0 * theta / (1.0 - phi ** 2)
    jac = np.array([[0.0, dtheta_dphi, 0.0],                                   # d theta / d (c, phi, s2)
                    [1.0 / (1.0 - phi), c / (1.0 - phi) ** 2, 0.0],            # d mu
                    [0.0, dsig2_dphi / (2.0 * sigma), dsig2_ds2 / (2.0 * sigma)]])   # d sigma
    cov = np.zeros((3, 3))
    cov[:2, :2] = cov_cp
    cov[2, 2] = var_s2
    se = np.sqrt(np.maximum(np.diag(jac @ cov @ jac.T), 0.0))
    est = {"theta": float(theta), "mu": float(mu), "sigma": sigma}
    se_d = dict(zip(("theta", "mu", "sigma"), map(float, se)))
    ll = ou_loglik((theta, mu, sigma), x, dt)
    return FitResult(est, se_d, _ci(est, se_d), ll, n, True)


def dickey_fuller_t(x) -> float:
    """The Dickey-Fuller t statistic of ``phi - 1`` in ``x_{t+1} = c + phi x_t + e``: strongly negative when the series reverts, near zero for a random walk. Its distribution under a unit
    root is not Student's: the 5% critical value is about -2.86 for a series and about -3.4 for the residual of a regression of one random walk on another (Engle and Granger), which is
    what a pair's spread is, so a spread should be required to beat -3.4, not -2."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        raise ValueError("at least 20 observations are needed")
    y, z = x[1:], x[:-1]
    zc = z - z.mean()
    sxx = float(zc @ zc)
    if sxx <= 0:
        raise ValueError("the series is constant")
    phi = float(zc @ (y - y.mean()) / sxx)
    resid = y - (y.mean() - phi * z.mean()) - phi * z
    s2 = float(resid @ resid / max(len(y) - 2, 1))
    return float((phi - 1.0) / np.sqrt(s2 / sxx))


# --------------------------------------------------------------------------------------------------------------------------- hitting times
def _standardised(x: float, level: float, theta: float, mu: float, sigma: float) -> tuple[float, float]:
    """The start and the target in the units in which the process has unit rate and equilibrium variance one half: ``w = (x - mu) sqrt(theta) / sigma``. A target below the start is
    reflected about ``mu`` (the process is symmetric), so the target is always above."""
    wx, wb = (x - mu) * np.sqrt(theta) / sigma, (level - mu) * np.sqrt(theta) / sigma
    return (wx, wb) if wb >= wx else (-wx, -wb)


def hitting_time(x: float, level: float, theta: float, mu: float, sigma: float) -> float:
    """The expected time for the process started at ``x`` to first reach ``level``: ``(sqrt(pi) / theta) Int_{w_x}^{w_b} erfcx(-w) dw`` in the standardised units (``erfcx`` is the
    scaled complementary error function ``e^{w^2} erfc(w)``, which keeps the integrand finite where ``e^{w^2}`` and ``erfc`` are not)."""
    if theta <= 0 or sigma <= 0:
        raise ValueError("theta and sigma must be positive")
    wx, wb = _standardised(x, level, theta, mu, sigma)
    if wb == wx:
        return 0.0
    val = integrate.quad(lambda w: special.erfcx(-w), wx, wb, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
    return float(np.sqrt(np.pi) / theta * val)


def hitting_time_moments(x: float, level: float, theta: float, mu: float, sigma: float, step: float = 2e-3) -> tuple[float, float]:
    """The mean and standard deviation of the first-passage time from ``x`` to ``level``. In the standardised units (time scaled by ``theta``) the mean is ``m1(w) = sqrt(pi) Int_w^{w_b}
    erfcx(-v) dv`` and the second moment ``m2(w) = 4 Int_w^{w_b} e^{y^2} Int_{-inf}^{y} m1(z) e^{-z^2} dz dy``; both are integrated on a grid of spacing ``step`` (the variance of the
    time is ``m2 - m1^2`` over ``theta^2``)."""
    if theta <= 0 or sigma <= 0:
        raise ValueError("theta and sigma must be positive")
    wx, wb = _standardised(x, level, theta, mu, sigma)
    if wb == wx:
        return 0.0, 0.0
    lo = min(wx, wb) - 9.0
    n = int(np.ceil((wb - lo) / step))
    w = np.linspace(lo, wb, n + 1)
    h = w[1] - w[0]
    f = special.erfcx(-w)
    # m1(w_i) = sqrt(pi) * (integral from w_i to w_b of erfcx(-v) dv): a cumulative sum from the right
    seg = 0.5 * (f[:-1] + f[1:]) * h
    m1 = np.sqrt(np.pi) * np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    inner_f = m1 * np.exp(-w ** 2)
    inner = np.concatenate([[0.0], np.cumsum(0.5 * (inner_f[:-1] + inner_f[1:]) * h)])            # Int_{-inf}^{w_i} m1 e^{-z^2} dz (the tail below the grid is negligible)
    outer_f = np.exp(np.minimum(w ** 2, 700.0)) * inner
    outer = 0.5 * (outer_f[:-1] + outer_f[1:]) * h
    m2 = 4.0 * np.concatenate([np.cumsum(outer[::-1])[::-1], [0.0]])
    i = int(np.clip(np.searchsorted(w, wx), 0, n))
    frac = (wx - w[i - 1]) / h if i > 0 else 0.0
    m1x = (1 - frac) * m1[i - 1] + frac * m1[i] if i > 0 else m1[0]
    m2x = (1 - frac) * m2[i - 1] + frac * m2[i] if i > 0 else m2[0]
    var = max(m2x - m1x ** 2, 0.0)
    return float(m1x / theta), float(np.sqrt(var) / theta)


def cycle_time(lower: float, upper: float, theta: float, mu: float, sigma: float) -> float:
    """The expected time for the process to go from ``lower`` to ``upper`` and back to ``lower``: ``(pi / theta) (erfi(w_upper) - erfi(w_lower))`` (Bertram 2010). Equals the sum of the two
    ``hitting_time``s."""
    if not lower < upper:
        raise ValueError("the lower level must be below the upper one")
    wa, wm = (lower - mu) * np.sqrt(theta) / sigma, (upper - mu) * np.sqrt(theta) / sigma
    return float(np.pi / theta * (special.erfi(wm) - special.erfi(wa)))


# -------------------------------------------------------------------------------------------------------------------------------- thresholds
def optimal_thresholds(theta: float, mu: float, sigma: float, cost: float) -> dict:
    """The buy and sell levels that maximise the expected profit per unit of time when a round trip costs ``cost`` (in the units of the spread). Buy at ``entry = mu - d``, sell at ``exit =
    mu + d``; the profit per cycle is ``2d - cost`` and the cycle takes ``cycle_time``. Returns the levels, the half-width ``d`` (also in standard deviations of the stationary law), the profit
    per unit of time ``rate``, the expected cycle and holding times and the number of round trips a year of the long leg (the short leg, trading the same levels in reverse, doubles the profit
    without more waiting)."""
    if theta <= 0 or sigma <= 0 or cost < 0:
        raise ValueError("theta and sigma must be positive and the cost not negative")
    scale = sigma / np.sqrt(theta)                                    # the unit of w

    def negative_rate(w: float) -> float:
        d = w * scale
        return -(2.0 * d - cost) / (2.0 * np.pi / theta * special.erfi(w))

    lo = max(cost / (2.0 * scale), 1e-6)                              # below this the trade loses money
    if lo > 12.0:
        raise ValueError("the cost is many times the range the spread moves in: no pair of levels earns a profit")
    res = optimize.minimize_scalar(negative_rate, bounds=(lo, lo + 6.0), method="bounded", options={"xatol": 1e-10})
    w = float(res.x)
    d = w * scale
    cycle = cycle_time(mu - d, mu + d, theta, mu, sigma)
    hold = hitting_time(mu - d, mu + d, theta, mu, sigma)
    return {"entry": float(mu - d), "exit": float(mu + d), "half_width": float(d), "half_width_sd": float(d / (sigma / np.sqrt(2.0 * theta))), "rate": float(-res.fun),
            "cycle": cycle, "hold": float(hold), "wait": float(cycle - hold), "round_trips_per_year": float(1.0 / cycle)}
