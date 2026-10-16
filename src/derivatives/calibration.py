"""Calibrating an option-pricing model to a chain of quotes.

Calibration is the inverse problem of pricing: given today's option quotes, find the parameters under which the model reprices them. The result is a *risk-neutral* description
of the market (it says what the quotes imply, not what will happen), and the only honest summary of it is how well the model reproduces the quotes and what it then implies for
products that are not quoted.

``Quotes``                  the out-of-the-money implied volatilities of several expiries (puts below the forward, calls above: the side that carries the information).
``quotes_from_chain``       pick them from a chain table of the platform: liquid out-of-the-money quotes, a few expiries spread over the curve, a limited number of strikes each.
``calibrate``               a bounded least-squares fit of a model's parameters, shared by all expiries, to the quotes. The residual of each quote is its price error divided by its Black-Scholes
                            vega, which equals the implied-volatility error to first order, so the fit weighs every strike evenly in volatility terms (a plain price fit is dominated by the
                            at-the-money options and ignores the wings). Prices come from the Lewis (2001) integral, so one evaluation costs a few thousand characteristic-function values.
``flat_smile_rmse``         the benchmark: one Black-Scholes volatility per expiry (the best constant), which a model must beat to claim that it captures the smile.
``compare_models``          calibrate several models and rank them.

The error is reported in volatility points (0.01 is one point): the root mean square, the mean absolute and the largest error over the quotes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

from . import fourier, levy
from .iv import implied_vol
from .pricing import bsm_price


@dataclass
class Quotes:
    """Out-of-the-money implied volatilities: ``T`` (years), ``K`` (strike) and ``iv`` for each quote, with the market's ``S``, ``r``, ``q``. ``weight`` scales a quote's residual."""

    S: float
    r: float
    q: float
    T: np.ndarray
    K: np.ndarray
    iv: np.ndarray
    weight: np.ndarray | None = None
    expiries: list = field(default_factory=list)

    def __post_init__(self):
        self.T, self.K, self.iv = (np.asarray(x, dtype=float) for x in (self.T, self.K, self.iv))
        if not (len(self.T) == len(self.K) == len(self.iv)) or len(self.T) < 3:
            raise ValueError("at least three quotes with a time, a strike and a volatility each are needed")
        if np.any(self.T <= 0) or np.any(self.K <= 0) or np.any(self.iv <= 0) or not np.all(np.isfinite(self.iv)):
            raise ValueError("times, strikes and implied volatilities must be positive and finite")
        if self.weight is not None:
            self.weight = np.asarray(self.weight, dtype=float)

    @property
    def forward(self) -> np.ndarray:
        return self.S * np.exp((self.r - self.q) * self.T)

    @property
    def is_call(self) -> np.ndarray:
        return self.K >= self.forward

    @property
    def maturities(self) -> np.ndarray:
        return np.unique(self.T)

    def market_prices(self) -> np.ndarray:
        """The Black-Scholes price of each quote at its implied volatility (a call above the forward, a put below)."""
        out = np.empty(len(self.T))
        c = self.is_call
        out[c] = bsm_price(self.S, self.K[c], self.T[c], self.r, self.q, self.iv[c], True)
        out[~c] = bsm_price(self.S, self.K[~c], self.T[~c], self.r, self.q, self.iv[~c], False)
        return out

    def vegas(self) -> np.ndarray:
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.iv ** 2) * self.T) / (self.iv * np.sqrt(self.T))
        return self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)

    def atm_iv(self) -> float:
        """The implied volatility nearest the forward on the shortest expiry that has quotes within 5%: a scale for the starting values."""
        k = np.abs(np.log(self.K / self.forward))
        near = k <= 0.05
        pool = near if near.any() else np.ones(len(k), dtype=bool)
        idx = np.flatnonzero(pool)
        return float(np.median(self.iv[idx[np.argsort(k[idx])[:5]]]))


def quotes_from_chain(chain, max_expiries: int = 6, min_dte: int = 7, window: float = 0.3, min_points: int = 6, max_strikes: int = 24, min_mid: float = 0.05) -> Quotes:
    """The calibration quotes of a chain table (the platform's schema: ``expiry``, ``strike``, ``right``, ``underlying``, ``rate``, ``bid``, ``ask`` and, when the file lacks them, ``T``, ``dividend``, ``mid``,
    ``iv``, which are derived):
    the out-of-the-money options with a bid, a mid of at least ``min_mid`` and a finite implied volatility, within ``window`` of the forward in log strike, for the expiries at least
    ``min_dte`` days out that keep ``min_points`` quotes, ``max_expiries`` of them spread evenly along the curve, at most ``max_strikes`` strikes each."""
    g = chain.copy()
    if "dividend" not in g:
        g["dividend"] = 0.0
    if "mid" not in g:
        g["mid"] = 0.5 * (g["bid"] + g["ask"])
    if "T" not in g:
        g["T"] = (pd.to_datetime(g["expiry"]) - pd.to_datetime(g["date"])).dt.days / 365.0
    if "F" not in g:
        g["F"] = g["underlying"] * np.exp((g["rate"] - g["dividend"]) * g["T"])
    if "iv" not in g or g["iv"].isna().all():                                  # a vendor file without volatilities: invert the mids
        iv = np.full(len(g), np.nan)
        for right, call in (("C", True), ("P", False)):
            sel = (g["right"] == right).to_numpy()
            if sel.any():
                h = g[sel]
                iv[sel] = implied_vol(h["mid"].to_numpy(), h["underlying"].to_numpy(), h["strike"].to_numpy(), h["T"].to_numpy(), h["rate"].to_numpy(), h["dividend"].to_numpy(), call)
        g["iv"] = iv
    k = np.log(g["strike"] / g["F"])
    otm = ((g["right"] == "P") & (g["strike"] <= g["F"])) | ((g["right"] == "C") & (g["strike"] > g["F"]))
    ok = otm & np.isfinite(g["iv"]) & (g["iv"] > 0) & (g["bid"] > 0) & (g["mid"] >= min_mid) & (k.abs() <= window) & (g["T"] * 365.0 >= min_dte)
    g = g[ok].assign(k=k[ok])
    groups = [(e, s.sort_values("strike")) for e, s in g.groupby("expiry") if len(s) >= min_points]
    if not groups:
        raise ValueError(f"no expiry has {min_points} usable out-of-the-money quotes at least {min_dte} days out: the chain is too thin to calibrate a model")
    groups.sort(key=lambda t: float(t[1]["T"].iloc[0]))
    if len(groups) > max_expiries:
        keep = np.unique(np.round(np.linspace(0, len(groups) - 1, max_expiries)).astype(int))
        groups = [groups[i] for i in keep]
    frames, expiries = [], []
    for e, s in groups:
        if len(s) > max_strikes:
            s = s.iloc[np.unique(np.round(np.linspace(0, len(s) - 1, max_strikes)).astype(int))]
        frames.append(s)
        expiries.append(str(np.datetime64(e, "D")))
    use = pd.concat(frames)
    return Quotes(float(use["underlying"].iloc[0]), float(use["rate"].median()), float(use["dividend"].median()), use["T"].to_numpy(), use["strike"].to_numpy(), use["iv"].to_numpy(),
                  expiries=expiries)


@dataclass
class Calibration:
    """A fitted model and how well it reprices the quotes (errors in volatility points: 0.01 = one point). ``std_errors`` are the usual least-squares standard errors of the free
    parameters (``sqrt(diag(s^2 (J'J)^-1))`` with the residual scale ``s``; NaN when the problem is singular), ``condition`` the condition number of the Jacobian (large: some combination
    of the parameters is not pinned down by the quotes) and ``at_bounds`` the parameters that ended on the edge of their allowed range (the model is straining to fit)."""

    model: levy.Model
    rmse: float
    mae: float
    max_error: float
    n_quotes: int
    n_expiries: int
    model_iv: np.ndarray
    success: bool
    message: str
    seconds: float
    n_evals: int
    n_failed: int = 0
    by_expiry: list = field(default_factory=list)
    std_errors: dict = field(default_factory=dict)
    at_bounds: list = field(default_factory=list)
    condition: float = float("nan")

    def summary(self) -> dict:
        return {"model": self.model.name, "label": self.model.label, "params": dict(self.model.params), "rmse": self.rmse, "mae": self.mae, "max_error": self.max_error, "n_quotes": self.n_quotes,
                "n_expiries": self.n_expiries, "success": self.success, "seconds": self.seconds, "n_evals": self.n_evals, "n_failed": self.n_failed, "by_expiry": self.by_expiry,
                "std_errors": dict(self.std_errors), "at_bounds": list(self.at_bounds), "condition": self.condition}


def model_otm_prices(model: levy.Model, quotes: Quotes, step: float = 0.15, max_upper: float = 2000.0) -> np.ndarray:
    """The model's price of each quote (calls above the forward, puts below), expiry by expiry through the Lewis integral (``step`` 0.15 is accurate to about 1e-9 in price; capping the
    integral at ``max_upper`` 2000 costs under 2e-4 in price for the slowest-decaying case, a few days of a variance-gamma model, and keeps one evaluation to a few milliseconds)."""
    out = np.empty(len(quotes.T))
    for T in quotes.maturities:
        idx = np.flatnonzero(quotes.T == T)
        calls = fourier.lewis_call(model, quotes.S, quotes.K[idx], T, quotes.r, quotes.q, step=step, max_upper=max_upper)
        puts = calls - quotes.S * np.exp(-quotes.q * T) + quotes.K[idx] * np.exp(-quotes.r * T)
        out[idx] = np.where(quotes.is_call[idx], calls, puts)
    return out


def model_ivs(model: levy.Model, quotes: Quotes, **kw) -> np.ndarray:
    """The Black-Scholes implied volatility of the model's price of each quote (NaN where the price is indistinguishable from intrinsic value)."""
    px = np.maximum(model_otm_prices(model, quotes, **kw), 0.0)
    out = np.full(len(px), np.nan)
    c = quotes.is_call
    T, K = quotes.T, quotes.K
    if c.any():
        out[c] = implied_vol(px[c], quotes.S, K[c], T[c], quotes.r, quotes.q, True)
    if (~c).any():
        out[~c] = implied_vol(px[~c], quotes.S, K[~c], T[~c], quotes.r, quotes.q, False)
    return out


def flat_smile_rmse(quotes: Quotes) -> float:
    """The root-mean-square error of the best flat-volatility description: each expiry gets the average of its own implied volatilities (the best constant for the squared error)."""
    err = np.empty(len(quotes.T))
    for T in quotes.maturities:
        idx = quotes.T == T
        err[idx] = quotes.iv[idx] - quotes.iv[idx].mean()
    return float(np.sqrt(np.mean(err ** 2)))


def starting_points(cls: type[levy.Model], quotes: Quotes) -> list[dict]:
    """Starting parameters scaled to the market's volatility: the model's defaults with the volatility parameters set from the at-the-money level, and a variant with less diffusion."""
    atm = quotes.atm_iv()
    base = dict(cls.start)
    out = []
    for share in (0.8, 0.5):
        s = dict(base)
        if "sigma" in s and cls is not levy.Heston:
            s["sigma"] = float(np.clip(atm * np.sqrt(share), *cls.bounds["sigma"]))
        if cls is levy.Heston:
            lo, hi = cls.bounds["v0"]
            s["v0"] = float(np.clip(atm ** 2, lo, hi))
            s["theta"] = float(np.clip(atm ** 2 * (1.1 if share > 0.6 else 0.8), *cls.bounds["theta"]))
            s["xi"] = 0.5 if share > 0.6 else 0.9
        if cls is levy.Merton:
            s["lam"], s["sigma_j"] = (0.5, 0.1) if share > 0.6 else (2.0, 0.15)
        if cls is levy.VarianceGamma or cls is levy.NIG:
            key = "nu" if cls is levy.VarianceGamma else "kappa"
            s[key] = 0.2 if share > 0.6 else 0.5
        if cls is levy.Kou:
            s["lam"], s["eta1"], s["eta2"] = (2.0, 12.0, 8.0) if share > 0.6 else (6.0, 25.0, 15.0)
        out.append(s)
    return out


def calibrate(model: str | type[levy.Model], quotes: Quotes, start: dict | None = None, fixed: dict | None = None, step: float = 0.15, max_nfev: int = 120, n_starts: int = 2) -> Calibration:
    """Fit ``model``'s parameters (the same for every expiry) to ``quotes`` by bounded least squares on the vega-scaled price errors.

    ``start`` overrides the starting point, ``fixed`` holds some parameters at given values (say ``rho`` for Heston), ``n_starts`` tries that many starting points and keeps the best.
    A parameter set outside the model's validity region (no exponential moment of order one, the price not a martingale) counts as a large error, which steers the search back.
    """
    cls = levy.MODELS[model] if isinstance(model, str) else model
    fixed = dict(fixed or {})
    names = [k for k in cls.param_names if k not in fixed]
    if not names:
        raise ValueError("every parameter is fixed: nothing to calibrate")
    lo = np.array([cls.bounds[k][0] for k in names])
    hi = np.array([cls.bounds[k][1] for k in names])
    target = quotes.market_prices()
    vega = np.maximum(quotes.vegas(), 1e-4 * quotes.S)
    w = np.ones(len(target)) if quotes.weight is None else quotes.weight
    counter = {"n": 0}
    big = 1.0

    def build(x) -> levy.Model | None:
        try:
            m = cls(**fixed, **dict(zip(names, x)))
        except ValueError:
            return None
        if hasattr(m, "valid") and not m.valid():
            return None
        if hasattr(m, "exponent_limit") and m.exponent_limit() <= 1.05:
            return None
        return m

    def resid(x):
        counter["n"] += 1
        m = build(x)
        if m is None:
            return np.full(len(target), big)
        try:
            with np.errstate(all="ignore"):
                px = model_otm_prices(m, quotes, step)
        except (FloatingPointError, ValueError, OverflowError):
            return np.full(len(target), big)
        r = w * (px - target) / vega
        return np.where(np.isfinite(r), r, big)

    t0 = time.perf_counter()
    starts = [start] if start is not None else starting_points(cls, quotes)[:max(1, n_starts)]
    best = None
    for s in starts:
        merged = {**s, **fixed}
        x0 = np.clip(np.array([merged[k] for k in names]), lo + 1e-9, hi - 1e-9)
        sol = optimize.least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale=np.maximum(np.abs(x0), 1e-2), max_nfev=max_nfev, xtol=1e-10, ftol=1e-10)
        if best is None or sol.cost < best.cost:
            best = sol
    fitted = build(best.x)
    if fitted is None:
        raise ValueError("the search ended outside the valid region of the model")
    iv = model_ivs(fitted, quotes, step=step)
    err = iv - quotes.iv
    good = np.isfinite(err)
    if good.sum() < 3:
        raise ValueError("the fitted model reprices fewer than three quotes to a usable volatility")
    by = []
    for T in quotes.maturities:
        idx = (quotes.T == T) & good
        by.append({"T": float(T), "dte": float(T * 365.0), "n": int(idx.sum()), "rmse": float(np.sqrt(np.mean(err[idx] ** 2))) if idx.any() else float("nan"), "bias": float(np.mean(err[idx])) if idx.any() else float("nan")})
    at_bounds = [k for k, v, a, b in zip(names, best.x, lo, hi) if min(v - a, b - v) <= 1e-3 * (b - a)]
    J = np.asarray(best.jac, dtype=float)
    cond = float(np.linalg.cond(J)) if np.all(np.isfinite(J)) else float("nan")
    se = np.full(len(names), np.nan)
    if np.isfinite(cond) and cond < 1e12:
        se = np.sqrt(np.maximum(np.diag(np.linalg.inv(J.T @ J)), 0.0) * 2.0 * best.cost / max(J.shape[0] - J.shape[1], 1))
    return Calibration(fitted, float(np.sqrt(np.mean(err[good] ** 2))), float(np.mean(np.abs(err[good]))), float(np.max(np.abs(err[good]))), len(err), len(quotes.maturities), iv, bool(best.success),
                       str(best.message), time.perf_counter() - t0, counter["n"], int((~good).sum()), by, {k: float(v) for k, v in zip(names, se)}, at_bounds, cond)


def compare_models(quotes: Quotes, models=("merton", "kou", "vg", "nig", "heston"), **kw) -> list[Calibration]:
    """Calibrate each model and rank by the root-mean-square volatility error (best first). A model whose search fails is left out."""
    out = []
    for m in models:
        try:
            out.append(calibrate(m, quotes, **kw))
        except ValueError:
            continue
    return sorted(out, key=lambda c: c.rmse)
