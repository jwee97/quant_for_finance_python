"""Copulas: the dependence structure of a joint distribution, separated from the marginals (Sklar 1959; Nelsen 2006; McNeil et al. 2015, ch. 7).

Correlation measures linear dependence and says nothing about how assets behave TOGETHER IN A CRASH. A copula ``C`` joins marginals into a joint law,
``F(x, y) = C(F_X(x), F_Y(y))``, so the marginals (fitted with EVT, Student-t, or the empirical distribution) and the dependence can be modelled separately.

=========  ===================================  ==========================  ==========================
copula     parameters                           lower-tail dependence       upper-tail dependence
=========  ===================================  ==========================  ==========================
gaussian   correlation matrix                   0                           0
student    correlation matrix, degrees of fr.   ``2 t_{v+1}(-sqrt((v+1)(1-rho)/(1+rho)))``   same
clayton    theta > 0                            ``2^(-1/theta)``            0
gumbel     theta >= 1                           0                           ``2 - 2^(1/theta)``
frank      theta (not 0)                        0                           0
=========  ===================================  ==========================  ==========================

Fitting uses the **pseudo-observations** (ranks / (n + 1)) so that no marginal is assumed (canonical maximum likelihood, Genest et al. 1995). Archimedean copulas
are bivariate here (one parameter); ``gaussian`` and ``student`` handle any dimension. ``compare_copulas`` ranks them by AIC/BIC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps
from scipy.special import gammaln


def pseudo_observations(x) -> np.ndarray:
    """Ranks scaled into (0, 1): ``rank / (n + 1)`` per column. The empirical-marginal transform used to fit a copula."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return np.column_stack([sps.rankdata(a[:, j]) / (len(a) + 1.0) for j in range(a.shape[1])])


def kendall_tau(x, y=None) -> float:
    if y is None:
        a = np.asarray(x)
        x, y = a[:, 0], a[:, 1]
    return float(sps.kendalltau(x, y)[0])


def empirical_tail_dependence(u: np.ndarray, q: float = 0.05) -> dict:
    """Empirical ``lambda_L(q) = P(V <= q | U <= q)`` and ``lambda_U(q) = P(V > 1 - q | U > 1 - q)`` of a bivariate pseudo-sample."""
    u1, u2 = u[:, 0], u[:, 1]
    lo = ((u1 <= q) & (u2 <= q)).sum() / max((u1 <= q).sum(), 1)
    hi = ((u1 > 1 - q) & (u2 > 1 - q)).sum() / max((u1 > 1 - q).sum(), 1)
    return {"lower": float(lo), "upper": float(hi), "q": q}


@dataclass
class Copula:
    family: str
    params: dict
    dim: int
    loglik: float = np.nan
    n_obs: int = 0

    # ------------------------------------------------------------------------------------------------------------- density
    def logpdf(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 1e-10, 1 - 1e-10)
        f = self.family
        if f == "gaussian":
            z = sps.norm.ppf(u)
            R = self.params["corr"]
            Ri = np.linalg.inv(R)
            quad = np.einsum("ni,ij,nj->n", z, Ri - np.eye(self.dim), z)
            return -0.5 * np.linalg.slogdet(R)[1] - 0.5 * quad
        if f == "student":
            nu, R = self.params["df"], self.params["corr"]
            x = sps.t.ppf(u, nu)
            Ri = np.linalg.inv(R)
            d = self.dim
            quad = np.einsum("ni,ij,nj->n", x, Ri, x)
            joint = gammaln((nu + d) / 2) - gammaln(nu / 2) - d / 2 * np.log(nu * np.pi) - 0.5 * np.linalg.slogdet(R)[1] - (nu + d) / 2 * np.log1p(quad / nu)
            marg = np.sum(sps.t.logpdf(x, nu), axis=1)
            return joint - marg
        th = self.params["theta"]
        a, b = u[:, 0], u[:, 1]
        if f == "clayton":
            return np.log1p(th) - (1 + th) * (np.log(a) + np.log(b)) - (2 + 1 / th) * np.log(a ** -th + b ** -th - 1)
        if f == "gumbel":
            la, lb = -np.log(a), -np.log(b)
            s = la ** th + lb ** th
            C = np.exp(-s ** (1 / th))
            return np.log(C) + (th - 1) * (np.log(la) + np.log(lb)) - np.log(a) - np.log(b) + (1 / th - 2) * np.log(s) + np.log(s ** (1 / th) + th - 1)
        if f == "frank":
            e = np.expm1(-th)
            num = -th * e * np.exp(-th * (a + b))
            den = (e + np.expm1(-th * a) * np.expm1(-th * b)) ** 2
            return np.log(np.abs(num / den))
        raise ValueError(f"unknown copula {f}")

    def cdf(self, u: np.ndarray) -> np.ndarray:
        """The copula ``C(u)``. Gaussian and Student use their multivariate distribution functions (bivariate Student by scipy's ``multivariate_t``)."""
        u = np.clip(np.asarray(u, dtype=float), 1e-10, 1 - 1e-10)
        f = self.family
        if f == "gaussian":
            return sps.multivariate_normal(mean=np.zeros(self.dim), cov=self.params["corr"]).cdf(sps.norm.ppf(u))
        if f == "student":
            return sps.multivariate_t(loc=np.zeros(self.dim), shape=self.params["corr"], df=self.params["df"]).cdf(sps.t.ppf(u, self.params["df"]))
        th = self.params["theta"]
        a, b = u[:, 0], u[:, 1]
        if f == "clayton":
            return (a ** -th + b ** -th - 1) ** (-1 / th)
        if f == "gumbel":
            return np.exp(-((-np.log(a)) ** th + (-np.log(b)) ** th) ** (1 / th))
        if f == "frank":
            return -np.log1p(np.expm1(-th * a) * np.expm1(-th * b) / np.expm1(-th)) / th
        raise ValueError(f)

    # ----------------------------------------------------------------------------------------------------------- simulation
    def simulate(self, n: int, seed=0) -> np.ndarray:
        """``n`` draws of the uniform marginals with this dependence."""
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        f = self.family
        if f == "gaussian":
            z = rng.multivariate_normal(np.zeros(self.dim), self.params["corr"], size=n)
            return sps.norm.cdf(z)
        if f == "student":
            nu = self.params["df"]
            z = rng.multivariate_normal(np.zeros(self.dim), self.params["corr"], size=n)
            w = rng.chisquare(nu, size=n) / nu
            return sps.t.cdf(z / np.sqrt(w)[:, None], nu)
        th = self.params["theta"]
        if f == "clayton":                                           # Marshall-Olkin with a gamma frailty
            v = rng.gamma(1.0 / th, 1.0, size=n)
            e = rng.exponential(size=(n, 2))
            return (1.0 + e / v[:, None]) ** (-1.0 / th)
        if f == "gumbel":                                            # Marshall-Olkin with a positive-stable frailty (Chambers-Mallows-Stuck)
            alpha = 1.0 / th
            w, e = rng.uniform(0, np.pi, n), rng.exponential(size=n)
            s = np.sin(alpha * w) / np.sin(w) ** (1 / alpha) * (np.sin((1 - alpha) * w) / e) ** ((1 - alpha) / alpha) if alpha < 1 else np.ones(n)
            ee = rng.exponential(size=(n, 2))
            return np.exp(-(ee / s[:, None]) ** alpha)
        if f == "frank":                                             # conditional inversion
            a, t = rng.uniform(size=n), rng.uniform(size=n)
            b = -np.log1p(t * np.expm1(-th) / (t + (1 - t) * np.exp(-th * a))) / th
            return np.column_stack([a, b])
        raise ValueError(f)

    # --------------------------------------------------------------------------------------------------------- dependence
    def tail_dependence(self) -> dict:
        f = self.family
        if f == "gaussian":
            return {"lower": 0.0, "upper": 0.0}
        if f == "student":
            nu, R = self.params["df"], self.params["corr"]
            rho = R[0, 1]
            lam = 2.0 * sps.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
            return {"lower": float(lam), "upper": float(lam)}
        th = self.params["theta"]
        if f == "clayton":
            return {"lower": float(2 ** (-1 / th)), "upper": 0.0}
        if f == "gumbel":
            return {"lower": 0.0, "upper": float(2 - 2 ** (1 / th))}
        return {"lower": 0.0, "upper": 0.0}

    def kendall_tau(self) -> float:
        f = self.family
        if f in ("gaussian", "student"):
            return float(2.0 / np.pi * np.arcsin(self.params["corr"][0, 1]))
        th = self.params["theta"]
        if f == "clayton":
            return th / (th + 2.0)
        if f == "gumbel":
            return 1.0 - 1.0 / th
        from scipy.integrate import quad

        def debye1(x):                                              # D1(x) = (1/x) int_0^x t / (e^t - 1) dt, for x > 0
            return quad(lambda t: t / np.expm1(t) if t > 1e-12 else 1.0, 0.0, x)[0] / x

        d = debye1(th) if th > 0 else debye1(-th) - th / 2.0       # D1(-x) = D1(x) + x / 2
        return float(1.0 - 4.0 / th * (1.0 - d))

    @property
    def n_params(self) -> int:
        d = self.dim
        if self.family == "gaussian":
            return d * (d - 1) // 2
        if self.family == "student":
            return d * (d - 1) // 2 + 1
        return 1

    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    def bic(self) -> float:
        return np.log(self.n_obs) * self.n_params - 2 * self.loglik


def _corr_from_tau(u: np.ndarray) -> np.ndarray:
    d = u.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            R[i, j] = R[j, i] = np.sin(np.pi / 2.0 * sps.kendalltau(u[:, i], u[:, j])[0])
    w, v = np.linalg.eigh(R)
    if w.min() < 1e-6:                                          # project to a valid correlation matrix
        R = (v * np.clip(w, 1e-6, None)) @ v.T
        s = np.sqrt(np.diag(R))
        R = R / np.outer(s, s)
    return R


def fit_copula(data, family: str = "gaussian", pseudo: bool = True) -> Copula:
    """Fit by canonical maximum likelihood on the pseudo-observations of ``data`` (rows = observations). ``pseudo=False`` treats ``data`` as already uniform."""
    u = pseudo_observations(data) if pseudo else np.asarray(data, dtype=float)
    d = u.shape[1]
    n = len(u)
    if family == "gaussian":
        z = sps.norm.ppf(u)
        R = np.corrcoef(z, rowvar=False).reshape(d, d)
        cop = Copula("gaussian", {"corr": R}, d)
    elif family == "student":
        R0 = _corr_from_tau(u)

        def nll(nu):
            return -Copula("student", {"corr": R0, "df": nu}, d).logpdf(u).sum()

        res = optimize.minimize_scalar(nll, bounds=(2.1, 60.0), method="bounded")
        cop = Copula("student", {"corr": R0, "df": float(res.x)}, d)
    elif family in ("clayton", "gumbel", "frank"):
        if d != 2:
            raise ValueError(f"{family} copula is bivariate here")
        tau = sps.kendalltau(u[:, 0], u[:, 1])[0]
        if family == "clayton":
            lo, hi = 1e-3, 30.0
        elif family == "gumbel":
            lo, hi = 1.0001, 30.0
        else:
            lo, hi = (1e-3, 60.0) if tau > 0 else (-60.0, -1e-3)
        res = optimize.minimize_scalar(lambda th: -Copula(family, {"theta": th}, 2).logpdf(u).sum(), bounds=(lo, hi), method="bounded")
        cop = Copula(family, {"theta": float(res.x)}, 2)
    else:
        raise ValueError("family must be gaussian, student, clayton, gumbel or frank")
    cop.loglik = float(cop.logpdf(u).sum())
    cop.n_obs = n
    return cop


def compare_copulas(data, families=("gaussian", "student", "clayton", "gumbel", "frank")) -> pd.DataFrame:
    """Fit each family to a bivariate (or, for gaussian/student, multivariate) sample and rank by AIC; includes the implied tail dependence."""
    rows = []
    d = np.asarray(data).shape[1]
    for f in families:
        if d != 2 and f not in ("gaussian", "student"):
            continue
        c = fit_copula(data, f)
        td = c.tail_dependence()
        rows.append({"family": f, "loglik": c.loglik, "aic": c.aic(), "bic": c.bic(), "lower_tail": td["lower"], "upper_tail": td["upper"], "params": {k: v for k, v in c.params.items() if k != "corr"}})
    return pd.DataFrame(rows).set_index("family").sort_values("aic")


# --------------------------------------------------------------------------------------------------------- marginals + portfolio
def fit_marginal(x, kind: str = "empirical"):
    """A callable pair ``(cdf, ppf)`` for a marginal: ``empirical`` (with linear interpolation), ``student`` (location-scale t), ``normal`` or ``evt``
    (empirical body, GPD tails above the 95th and below the 5th percentiles; McNeil & Frey semi-parametric)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if kind == "empirical":
        grid = (np.arange(1, n + 1) - 0.5) / n
        return (lambda v: np.interp(v, x, grid)), (lambda p: np.interp(p, grid, x))
    if kind == "normal":
        mu, sd = x.mean(), x.std(ddof=1)
        return (lambda v: sps.norm.cdf(v, mu, sd)), (lambda p: sps.norm.ppf(p, mu, sd))
    if kind == "student":
        nu, mu, sd = sps.t.fit(x)
        return (lambda v: sps.t.cdf(v, nu, mu, sd)), (lambda p: sps.t.ppf(p, nu, mu, sd))
    if kind == "evt":
        from .evt import fit_gpd
        up, lo = fit_gpd(x, quantile=0.95), fit_gpd(-x, quantile=0.95)
        u_hi, u_lo = up.threshold, -lo.threshold
        grid = (np.arange(1, n + 1) - 0.5) / n

        def ppf(p):
            p = np.asarray(p, dtype=float)
            body = np.interp(p, grid, x)
            hi = np.where(p > 1 - up.tail_fraction, up.quantile(np.clip(p, 0, 1 - 1e-12)) if np.isscalar(p) else np.array([up.quantile(min(v, 1 - 1e-12)) for v in np.atleast_1d(p)]), body)
            lw = np.where(p < lo.tail_fraction, -np.array([lo.quantile(min(1 - v, 1 - 1e-12)) for v in np.atleast_1d(p)]).reshape(np.shape(p)), hi)
            return np.where(p < lo.tail_fraction, lw, hi)

        def cdf(v):
            v = np.asarray(v, dtype=float)
            body = np.interp(v, x, grid)
            exc_hi = np.clip(v - u_hi, 0, None)
            tail_hi = 1 - up.tail_fraction * (1 + up.xi * exc_hi / up.beta) ** (-1 / up.xi) if abs(up.xi) > 1e-8 else 1 - up.tail_fraction * np.exp(-exc_hi / up.beta)
            exc_lo = np.clip(u_lo - v, 0, None)
            tail_lo = lo.tail_fraction * (1 + lo.xi * exc_lo / lo.beta) ** (-1 / lo.xi) if abs(lo.xi) > 1e-8 else lo.tail_fraction * np.exp(-exc_lo / lo.beta)
            return np.where(v > u_hi, tail_hi, np.where(v < u_lo, tail_lo, body))

        return cdf, ppf
    raise ValueError("kind must be empirical, normal, student or evt")


def simulate_joint(returns: pd.DataFrame, n: int = 100000, copula: str = "student", marginal: str = "evt", seed: int = 0) -> pd.DataFrame:
    """Simulate scenarios from the fitted copula and marginals: the input to a copula-based portfolio VaR/ES or a mean-CVaR optimisation."""
    r = returns.dropna()
    cop = fit_copula(r.to_numpy(), copula)
    u = cop.simulate(n, seed)
    cols = {}
    for j, c in enumerate(r.columns):
        _, ppf = fit_marginal(r[c].to_numpy(), marginal)
        cols[c] = np.asarray(ppf(u[:, j]), dtype=float).reshape(-1)
    return pd.DataFrame(cols)


def portfolio_var_es(scenarios: pd.DataFrame, weights, alpha: float = 0.99) -> dict:
    """Empirical VaR and ES of a portfolio over simulated scenarios (loss = -portfolio return)."""
    w = np.asarray(weights if not isinstance(weights, pd.Series) else weights.reindex(scenarios.columns), dtype=float)
    loss = -(scenarios.to_numpy() @ w)
    var = float(np.quantile(loss, alpha))
    return {"var": var, "es": float(loss[loss >= var].mean())}
