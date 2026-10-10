"""Physics-informed neural networks (Raissi, Perdikaris and Karniadakis 2019) for pricing equations: Black-Scholes-Merton, Vasicek bonds, Heston and Bates options.

A PINN represents the solution of a partial differential equation by a neural network ``u(tau, x)`` of time to maturity and the state, and trains it by minimising the equation's *residual* at randomly drawn
collocation points (derivatives by automatic differentiation) together with the boundary conditions, instead of fitting data. Here the terminal condition (the payoff at ``tau = 0``) is imposed exactly by
writing ``u = payoff + tau * network``, and the far boundaries by a penalty. There is no data and no simulation: the network *is* the price, for every state at once, and its derivatives give the Greeks.

    bsm_pinn      u_tau = 1/2 sigma^2 u_yy + (r - q - sigma^2/2) u_y - r u in log-moneyness ``y = ln(S/K)``, a call; exact answer: the Black-Scholes formula
    vasicek_pinn  a zero-coupon bond: P_tau = kappa (theta - r) P_r + 1/2 sigma^2 P_rr - r P, P(0) = 1; exact answer: the affine formula
    heston_pinn   a call under stochastic variance (three state variables: tau, y, v); reference: the Fourier price
    bates_pinn    Heston with lognormal jumps in the spot, which adds the integral term ``lambda E[u(y + J) - u(y)] - lambda m u_y``, evaluated by Gauss-Hermite quadrature on the network itself; reference: the Fourier price

Accuracy is limited: with a few thousand steps a PINN prices to a fraction of a percent of the strike in the middle of its domain and is worse near the edges and for short maturities; it is a way of solving a
PDE without a grid (attractive as the dimension grows), not a replacement for a closed form where one exists. Heston and Bates are the interesting uses, and cost minutes to train.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate


# ------------------------------------------------------------------------------------------------------------------ references
def vasicek_bond(r0, tau, kappa, theta, sigma):
    """The Vasicek zero-coupon bond price ``exp(A - B r)``."""
    r0, tau = np.asarray(r0, dtype=float), np.asarray(tau, dtype=float)
    B = (1.0 - np.exp(-kappa * tau)) / kappa
    A = (theta - sigma ** 2 / (2 * kappa ** 2)) * (B - tau) - sigma ** 2 * B ** 2 / (4 * kappa)
    return np.exp(A - B * r0)


def fourier_call(cf, S, K, T, r, q=0.0):
    """European call by Gil-Pelaez inversion from the characteristic function ``cf(u)`` of ``ln S_T``."""
    lk = math.log(K)
    cf_m_i = cf(-1j)

    def i1(u):
        return np.real(np.exp(-1j * u * lk) * cf(u - 1j) / (1j * u * cf_m_i)) if u > 1e-12 else 0.0

    def i2(u):
        return np.real(np.exp(-1j * u * lk) * cf(u) / (1j * u)) if u > 1e-12 else 0.0

    p1 = 0.5 + integrate.quad(i1, 1e-10, 200.0, limit=400)[0] / math.pi
    p2 = 0.5 + integrate.quad(i2, 1e-10, 200.0, limit=400)[0] / math.pi
    return S * math.exp(-q * T) * p1 - K * math.exp(-r * T) * p2


def bates_price(S, K, T, r, q, v0, kappa, theta, xi, rho, lam, mu_j, sigma_j):
    """Bates (1996): Heston with lognormal jumps (log jump size ``N(mu_j, sigma_j^2)``, intensity ``lam``); the spot's drift is compensated by ``lam * m`` with ``m = exp(mu_j + sigma_j^2/2) - 1``."""
    from ..derivatives.pricing import heston_characteristic

    m = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

    def cf(u):
        u = np.asarray(u, dtype=complex)
        jump = np.exp(lam * T * (np.exp(1j * u * mu_j - 0.5 * sigma_j ** 2 * u ** 2) - 1.0))
        return heston_characteristic(u, T, r, q + lam * m, v0, kappa, theta, xi, rho, S) * jump

    return fourier_call(cf, S, K, T, r, q)


# ------------------------------------------------------------------------------------------------------------------ the networks
def _net(inp: int, hidden: int = 64, layers: int = 4):
    from torch import nn

    parts, size = [], inp
    for _ in range(layers):
        parts += [nn.Linear(size, hidden), nn.Tanh()]
        size = hidden
    parts.append(nn.Linear(size, 1))
    return nn.Sequential(*parts)


def _payoff(y, sharpness: float = 30.0):
    """The call payoff per unit strike, ``max(e^y - 1, 0)``, smoothed (softplus) so that automatic differentiation sees its curvature: with the exact kink the second derivative is zero everywhere the
    computer looks and the network can leave the payoff untouched, which is not a solution of the equation. The smoothing is about ``ln 2 / sharpness`` of the strike at the kink at maturity and fades with time."""
    import torch

    return torch.nn.functional.softplus(torch.exp(y) - 1.0, beta=sharpness)


def _train(loss_fn, params, steps: int, lr: float, seed: int, report: int = 200, lbfgs_steps: int = 0):
    """Adam with a cosine learning-rate decay on freshly drawn collocation points, then optionally ``lbfgs_steps`` iterations of L-BFGS on one fixed draw (which polishes the fit far more than more Adam steps do)."""
    import torch

    params = list(params)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1), eta_min=lr * 0.02)
    history = []
    for step in range(steps):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % report == 0 or step == steps - 1:
            history.append(float(loss.detach()))
    if lbfgs_steps:
        torch.manual_seed(seed + 1)
        state = torch.get_rng_state()
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_steps, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            torch.set_rng_state(state)                                                                  # the same collocation points at every evaluation
            lb.zero_grad()
            value = loss_fn()
            value.backward()
            return value

        lb.step(closure)
        history.append(float(closure().detach()))
    return history


class _PINN:
    def predict_numpy(self, *args):
        import torch

        with torch.no_grad():
            return self._eval(*[torch.as_tensor(np.asarray(a, dtype=np.float32)) for a in args]).numpy().astype(float).reshape(-1)


class BSMPINN(_PINN):
    """A European call (per unit strike) under Black-Scholes-Merton. ``price(S, K, tau)`` rescales: the price is ``K * u(tau, ln(S/K))``."""

    def __init__(self, r: float, q: float, sigma: float, maturity: float = 1.0, y_range: float = 1.5, steps: int = 1500, batch: int = 1024, lr: float = 3e-3, hidden: int = 48, layers: int = 3, sharpness: float = 30.0, lbfgs_steps: int = 400, seed: int = 0):
        self.sharpness, self.lbfgs_steps = sharpness, lbfgs_steps
        self.r, self.q, self.sigma, self.T, self.ymax, self.steps, self.batch, self.lr, self.hidden, self.layers, self.seed = r, q, sigma, maturity, y_range, steps, batch, lr, hidden, layers, seed

    def _u(self, tau, y):
        import torch

        x = torch.cat([tau / self.T, y / self.ymax], dim=1)
        return _payoff(y, self.sharpness) + tau * self.net(x)                                                  # smoothed payoff + tau * network: exact at tau = 0 up to the smoothing

    def fit(self) -> "BSMPINN":
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        self.net = _net(2, self.hidden, self.layers)
        r, q, s = self.r, self.q, self.sigma

        def loss():
            tau = (torch.rand(self.batch, 1) * self.T).requires_grad_(True)
            y = ((torch.rand(self.batch, 1) * 2 - 1) * self.ymax).requires_grad_(True)
            u = self._u(tau, y)
            u_t = torch.autograd.grad(u.sum(), tau, create_graph=True)[0]
            u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
            res = u_t - 0.5 * s ** 2 * u_yy - (r - q - 0.5 * s ** 2) * u_y + r * u
            tb = torch.rand(256, 1) * self.T
            lo = self._u(tb, torch.full_like(tb, -self.ymax))
            hi = self._u(tb, torch.full_like(tb, self.ymax))
            bc = (lo ** 2).mean() + ((hi - (torch.exp(torch.full_like(tb, self.ymax) - q * tb) - torch.exp(-r * tb))) ** 2).mean()
            return (res ** 2).mean() + bc

        self.history = _train(loss, self.net.parameters(), self.steps, self.lr, self.seed, lbfgs_steps=self.lbfgs_steps)
        return self

    def _eval(self, tau, y):
        return self._u(tau.reshape(-1, 1), y.reshape(-1, 1))

    def price(self, S, K, tau):
        S, tau = np.broadcast_arrays(np.asarray(S, dtype=float), np.asarray(tau, dtype=float))
        return K * self.predict_numpy(tau, np.log(S / K))


class VasicekPINN(_PINN):
    """A zero-coupon bond ``P(tau, r)`` under Vasicek ``dr = kappa (theta - r) dt + sigma dW``."""

    def __init__(self, kappa: float, theta: float, sigma: float, maturity: float = 5.0, r_range: tuple = (-0.02, 0.14), steps: int = 2000, batch: int = 1024, lr: float = 3e-3, hidden: int = 32, layers: int = 3, lbfgs_steps: int = 0, seed: int = 0):
        self.lbfgs_steps = lbfgs_steps
        self.kappa, self.theta, self.sigma, self.T, self.r_range, self.steps, self.batch, self.lr, self.hidden, self.layers, self.seed = kappa, theta, sigma, maturity, r_range, steps, batch, lr, hidden, layers, seed

    def _u(self, tau, r):
        import torch

        lo, hi = self.r_range
        x = torch.cat([tau / self.T, (r - 0.5 * (lo + hi)) / (0.5 * (hi - lo))], dim=1)
        return 1.0 + tau * self.net(x)

    def fit(self) -> "VasicekPINN":
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        self.net = _net(2, self.hidden, self.layers)
        k, th, s = self.kappa, self.theta, self.sigma
        lo, hi = self.r_range

        def loss():
            tau = (torch.rand(self.batch, 1) * self.T).requires_grad_(True)
            r = (lo + torch.rand(self.batch, 1) * (hi - lo)).requires_grad_(True)
            u = self._u(tau, r)
            u_t = torch.autograd.grad(u.sum(), tau, create_graph=True)[0]
            u_r = torch.autograd.grad(u.sum(), r, create_graph=True)[0]
            u_rr = torch.autograd.grad(u_r.sum(), r, create_graph=True)[0]
            res = u_t - k * (th - r) * u_r - 0.5 * s ** 2 * u_rr + r * u
            return (res ** 2).mean()

        self.history = _train(loss, self.net.parameters(), self.steps, self.lr, self.seed, lbfgs_steps=self.lbfgs_steps)
        return self

    def _eval(self, tau, r):
        return self._u(tau.reshape(-1, 1), r.reshape(-1, 1))

    def price(self, r, tau):
        r, tau = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(tau, dtype=float))
        return self.predict_numpy(tau, r)


class HestonPINN(_PINN):
    """A European call (per unit strike) under Heston, ``u(tau, y, v)``, with optional lognormal jumps in the spot (Bates) when ``lam > 0``."""

    def __init__(self, r: float, q: float, kappa: float, theta: float, xi: float, rho: float, maturity: float = 1.0, y_range: float = 1.2, v_max: float = 0.6, lam: float = 0.0,
                 mu_j: float = 0.0, sigma_j: float = 0.0, steps: int = 1500, batch: int = 1024, lr: float = 3e-3, hidden: int = 48, layers: int = 3, quad: int = 9, sharpness: float = 30.0, lbfgs_steps: int = 300, seed: int = 0):
        self.sharpness, self.lbfgs_steps = sharpness, lbfgs_steps
        self.r, self.q, self.kappa, self.theta, self.xi, self.rho, self.T, self.ymax, self.vmax = r, q, kappa, theta, xi, rho, maturity, y_range, v_max
        self.lam, self.mu_j, self.sigma_j, self.steps, self.batch, self.lr, self.hidden, self.layers, self.quad, self.seed = lam, mu_j, sigma_j, steps, batch, lr, hidden, layers, quad, seed

    def _u(self, tau, y, v):
        import torch

        x = torch.cat([tau / self.T, y / self.ymax, v / self.vmax], dim=1)
        return _payoff(y, self.sharpness) + tau * self.net(x)

    def fit(self) -> "HestonPINN":
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        self.net = _net(3, self.hidden, self.layers)
        r, q, kap, th, xi, rho = self.r, self.q, self.kappa, self.theta, self.xi, self.rho
        lam, mu_j, sj = self.lam, self.mu_j, self.sigma_j
        m = math.exp(mu_j + 0.5 * sj ** 2) - 1.0
        nodes, weights = np.polynomial.hermite_e.hermegauss(self.quad)
        nodes_t, w_t = torch.as_tensor(nodes, dtype=torch.float32), torch.as_tensor(weights / weights.sum(), dtype=torch.float32)

        def loss():
            tau = (torch.rand(self.batch, 1) * self.T).requires_grad_(True)
            y = ((torch.rand(self.batch, 1) * 2 - 1) * self.ymax).requires_grad_(True)
            v = (torch.rand(self.batch, 1) ** 1.5 * self.vmax).requires_grad_(True)               # more points at low variance, where the equation degenerates
            u = self._u(tau, y, v)
            g = lambda out, x: torch.autograd.grad(out.sum(), x, create_graph=True)[0]            # noqa: E731
            u_t, u_y, u_v = g(u, tau), g(u, y), g(u, v)
            u_yy, u_vv, u_yv = g(u_y, y), g(u_v, v), g(u_y, v)
            drift_y = r - q - 0.5 * v - lam * m
            res = u_t - 0.5 * v * u_yy - drift_y * u_y - rho * xi * v * u_yv - 0.5 * xi ** 2 * v * u_vv - kap * (th - v) * u_v + r * u
            if lam > 0:
                shifted = y.unsqueeze(1) + mu_j + sj * nodes_t.view(1, -1, 1)                       # (batch, quad, 1)
                u_jump = self._u(tau.unsqueeze(1).expand(-1, self.quad, -1).reshape(-1, 1), shifted.reshape(-1, 1), v.unsqueeze(1).expand(-1, self.quad, -1).reshape(-1, 1)).view(-1, self.quad)
                res = res - lam * ((u_jump * w_t).sum(dim=1, keepdim=True) - u)
            tb = torch.rand(256, 1) * self.T
            vb = torch.rand(256, 1) * self.vmax
            top = torch.full_like(tb, self.ymax)
            bc = (self._u(tb, -top, vb) ** 2).mean() + ((self._u(tb, top, vb) - (torch.exp(top - q * tb) - torch.exp(-r * tb))) ** 2).mean()
            return (res ** 2).mean() + bc

        self.history = _train(loss, self.net.parameters(), self.steps, self.lr, self.seed, lbfgs_steps=self.lbfgs_steps)
        return self

    def _eval(self, tau, y, v):
        return self._u(tau.reshape(-1, 1), y.reshape(-1, 1), v.reshape(-1, 1))

    def price(self, S, K, tau, v):
        S, tau, v = np.broadcast_arrays(np.asarray(S, dtype=float), np.asarray(tau, dtype=float), np.asarray(v, dtype=float))
        return K * self.predict_numpy(tau, np.log(S / K), v)
