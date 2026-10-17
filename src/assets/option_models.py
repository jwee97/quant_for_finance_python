"""Reports on the option-pricing models of ``derivatives``: fit them to a chain, price a product under each, and cost the hedging of an option.

The Options tab (``webapp.models_api``) and the ``quant models`` command both call these functions, so the page and the terminal give the same numbers. Each report takes a chain table of the platform's
schema (``assets.options_analytics.prepare_chain``) and returns ``(data, warnings, notes)``: the numbers, the things the reader must know, and how they were made.

``fit_report``      the calibrations of the five models (``calibration``) ranked against the best flat volatility, the smile each draws beside the market's, the density of the price at expiry each implies
                    (``fourier.risk_neutral_density``) and its odds of a fall.
``price_report``    one product (a European, digital, barrier, average-price or American option) under every fitted model and under Black-Scholes. On a vanilla the models differ by how well each fits the quotes;
                    on another product they differ by that and by what the quotes do not determine, so the spread between them, set beside their spread on the vanilla at the same strike, is the
                    product's model risk.
``hedge_report``    what hedging an option costs when every trade costs money (``costs``): Leland's price at several rebalancing frequencies, the Whalley-Wilmott band, a simulated comparison of the rules and
                    the utility-based bid and ask of a binomial market.

A calibration is a *risk-neutral* description of the quotes: it says what they imply, not what will happen. When the chain is the platform's simulated one the models are fitted to another model, so the fit
shows which family resembles that surface and nothing about a market: the reports say so.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import integrate, stats

from ..derivatives import calibration as cal
from ..derivatives import costs, exotics, fourier, levy, pde, pide
from ..derivatives.pricing import bsm_price
from . import options_analytics as oa

FIT_MODELS = ("merton", "kou", "vg", "nig", "heston")
BARRIERS = ("down-out", "up-out", "down-in", "up-in")
LIMITS = {"max_expiries": (2, 8), "min_dte": (1, 120), "window": (0.05, 0.6), "dte": (7, 730), "strike_pct": (20.0, 300.0), "barrier_pct": (10.0, 400.0), "paths": (5_000, 200_000), "seed": (0, 1_000_000),
          "vol_pct": (1.0, 300.0), "cost_bp": (0.0, 500.0), "steps": (2, 504), "risk_aversion": (0.01, 100.0), "hedge_paths": (1_000, 40_000)}
DEFAULTS = {"max_expiries": 6, "min_dte": 7, "window": 0.30, "paths": 50_000, "dte": 90, "cost_bp": 5.0, "risk_aversion": 1.0, "hedge_paths": 10_000}
MODEL_INFO = {
    "merton": {"text": "Black-Scholes with rare jumps of normally distributed size: a few large moves a year on top of a diffusion.",
               "assumes": "Finite activity: jumps are rare and of one typical size; between them the price diffuses.", "fails": "Jumps of one fixed average size cannot give both a steep short-dated skew and a flat long-dated one."},
    "kou": {"text": "Like Merton, but jump sizes are double-exponential: a different, usually fatter, tail on each side.",
            "assumes": "Finite-activity jumps whose sizes have exponential tails, up and down separately.", "fails": "Its tail parameters are poorly pinned down by a chain: it often ends at the edge of its range."},
    "vg": {"text": "The variance gamma model: pure jumps, infinitely many small ones, with a skew parameter and a kurtosis parameter and no diffusion.",
           "assumes": "Prices move only by jumps (no continuous part); skew and tails each have one dial.", "fails": "Its skew fades with maturity as in a random walk, faster than most markets' do."},
    "nig": {"text": "The normal inverse Gaussian model: infinite-activity jumps like the variance gamma, with somewhat fatter tails.",
            "assumes": "Pure jumps with semi-heavy tails; skew and tails each have one dial.", "fails": "Like all Lévy models its smile flattens with maturity as for independent increments."},
    "heston": {"text": "Stochastic volatility: variance is itself random, mean-reverting, and moves against the price (correlation rho). It is the only model here whose smile does not flatten as fast with maturity.",
               "assumes": "Variance follows a square-root process; no jumps in the price.", "fails": "Short-dated skew is too flat unless the volatility of variance is extreme; it cannot price early exercise here (two state variables)."},
}
PRODUCTS = {
    "vanilla": {"label": "European option", "text": "A call or put exercised only at expiry: priced from the model's characteristic function by the Lewis integral. The models were fitted to these, so each should agree with the market's smile to within its fit error."},
    "digital": {"label": "Digital (cash-or-nothing)", "text": "Pays 1 if the price ends above (call) or below (put) the strike. Its price is a risk-neutral probability, which is the slope of the smile in disguise: a flat volatility cannot price it."},
    "barrier": {"label": "Barrier option", "text": "A call or put that is cancelled (knock-out) or only born (knock-in) if the price touches a level; monitored on each trading day. Depends on the paths, which the vanilla quotes do not pin down."},
    "asian": {"label": "Average-price option", "text": "Pays on the average of the price on equally spaced dates against a fixed strike. Averaging damps the tails, so the models disagree less than on a barrier."},
    "american": {"label": "American option", "text": "Can be exercised at any time. Listed options on ETFs are American. Solved on a grid with the early-exercise constraint (jump models only)."},
}
HEDGE_RULES = {
    "delta": "Trade to the Black-Scholes delta on every date.",
    "leland": "The same, sold at Leland's price and hedged with the adjusted volatility, so that the premium pays for the hedging cost.",
    "band": "Trade only when the holding leaves the Whalley-Wilmott no-trade band, and then only to its edge.",
}


class Unavailable(Exception):
    """A product a model cannot price: reported in the table, not an error of the request."""


def plan() -> dict:
    """What the forms (and the ``--list`` of the command) choose among: the models with their parameters and ranges, the products, the barrier kinds, the hedging rules and the defaults."""
    models = []
    for mid in FIT_MODELS:
        cls = levy.MODELS[mid]
        models.append({"id": mid, "label": cls.label, "params": [{"name": k, "start": cls.start[k], "low": cls.bounds[k][0], "high": cls.bounds[k][1]} for k in cls.param_names], **MODEL_INFO[mid], "american": cls.is_levy})
    return {"models": models, "default_models": list(FIT_MODELS), "products": [{"id": k, **v} for k, v in PRODUCTS.items()], "barriers": list(BARRIERS), "hedge_rules": [{"id": k, "text": v} for k, v in HEDGE_RULES.items()],
            "defaults": dict(DEFAULTS)}


def check_range(name: str, value, lo, hi):
    """``value`` if it is a finite number within ``[lo, hi]``, else a ``ValueError`` that names the field."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name.replace('_', ' ')} must be a number") from None
    if not np.isfinite(v) or not lo <= v <= hi:
        raise ValueError(f"{name.replace('_', ' ')} must be between {lo:g} and {hi:g}")
    return value


def check_models(ids) -> list[str]:
    """The model ids as a list without repeats (all five when none are given)."""
    if ids in (None, "", ()):
        return list(FIT_MODELS)
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError("models must be a list of model names")
    bad = [m for m in ids if m not in FIT_MODELS]
    if bad:
        raise ValueError(f"unknown model {bad[0]!r}; choose from {', '.join(FIT_MODELS)}")
    return list(dict.fromkeys(ids))


# ------------------------------------------------------------------------------------------------------------------------------------ fitting
def fingerprint(chain: pd.DataFrame) -> tuple:
    """A cheap identity of a chain's content (size, date, price and the sums of its volatilities and mids), so that a fit is reused only for the same quotes."""
    return (len(chain), str(pd.Timestamp(chain["date"].iloc[0]).date()), round(float(chain["underlying"].iloc[0]), 4), round(float(np.nansum(chain["iv"])), 6), round(float(np.nansum(chain["mid"])), 4))


def select_quotes(chain: pd.DataFrame, max_expiries: int = 6, min_dte: int = 7, window: float = 0.30) -> cal.Quotes:
    """The out-of-the-money quotes the models are fitted to (see ``calibration.quotes_from_chain``). Raises ``ValueError`` when the chain is too thin."""
    return cal.quotes_from_chain(chain, max_expiries=max_expiries, min_dte=min_dte, window=window)


class FitCache:
    """Calibrations kept for a session, keyed by what was fitted to what (the chain's fingerprint, the quote settings and the model), the least recently used dropped first."""

    def __init__(self, size: int = 24):
        self.size = size
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple, cal.Calibration] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def calibrate(self, quotes: cal.Quotes, ids: list[str], key: tuple | None = None, progress=None) -> tuple[dict[str, cal.Calibration], dict[str, str]]:
        """The calibrations of ``ids`` (from the cache when ``key`` names a fit already made) and the models whose search failed, with the reason."""
        done, failed = {}, {}
        for i, mid in enumerate(ids):
            hit = None
            if key is not None:
                with self._lock:
                    hit = self._items.get((key, mid))
                    if hit is not None:
                        self._items.move_to_end((key, mid))
            if hit is None:
                if progress:
                    progress(f"fitting {levy.MODELS[mid].label} ({i + 1} of {len(ids)}): about {'10' if mid in ('heston', 'vg', 'kou') else '5'} seconds")
                try:
                    hit = cal.calibrate(mid, quotes)
                except ValueError as exc:
                    failed[mid] = str(exc)
                    continue
                if key is not None:
                    with self._lock:
                        self._items[(key, mid)] = hit
                        while len(self._items) > self.size:
                            self._items.popitem(last=False)
            done[mid] = hit
        return done, failed


def _atm_of(quotes: cal.Quotes, T: float) -> float:
    """The at-the-money volatility of the quotes of one expiry (the smile interpolated at the forward)."""
    on = quotes.T == T
    return float(np.interp(0.0, np.log(quotes.K[on] / quotes.forward[on]), quotes.iv[on]))


def fit_report(name: str, quotes: cal.Quotes, results: dict[str, cal.Calibration], failed: dict[str, str], settings: dict, simulated: bool = False) -> tuple[dict, list[str], list[str]]:
    """The fit of the models to ``quotes`` as ``(data, warnings, notes)``: the models ranked by their error (volatility points) with parameters, standard errors, the edges they hit and the conditioning,
    the flat-volatility benchmark, the smiles and the implied density. ``simulated`` says the chain is itself a model."""
    flat = cal.flat_smile_rmse(quotes)
    rows = []
    for mid, c in sorted(results.items(), key=lambda kv: kv[1].rmse):
        s = c.summary()
        s.update(id=mid, n_params=len(c.model.param_names), explained=float(max(0.0, 1.0 - (c.rmse / flat) ** 2)) if flat > 0 else None, params=dict(c.model.params), bounds={k: list(c.model.bounds[k]) for k in c.model.param_names})
        rows.append(s)
    warnings = []
    if simulated:
        warnings.append("This chain is itself a model (an arbitrage-free SSVI surface on the ETF's real price), so the models are fitted to another model. The fit shows which family resembles that surface, not what a "
                        "market thinks: download a real chain (the Free data card) to calibrate to quotes.")
    for r in rows:
        label = levy.MODELS[r["id"]].label
        if r["at_bounds"]:
            warnings.append(f"{label}: {', '.join(r['at_bounds'])} ended at the edge of the allowed range, so the model is straining to fit and that parameter is not a measurement.")
        if np.isfinite(r["condition"]) and r["condition"] > 1e6:
            warnings.append(f"{label}: the quotes do not pin all its parameters down (condition number {r['condition']:.1e}); read the standard errors.")
    if rows[0]["rmse"] > 0.02:
        warnings.append(f"The best model is still {rows[0]['rmse'] * 100:.1f} volatility points from the quotes on average: none of these families reproduces this smile.")
    if flat < 0.003:
        warnings.append(f"The smile is almost flat ({flat * 100:.2f} points from a constant volatility): there is little for a model to explain.")
    for mid, why in failed.items():
        warnings.append(f"{levy.MODELS[mid].label} could not be fitted: {why}")
    data = {"underlying": name, "spot": quotes.S, "rate": quotes.r, "dividend": quotes.q, "settings": settings,
            "quotes": {"n": int(len(quotes.T)), "expiries": [{"expiry": e, "dte": float(T * 365.0), "n": int((quotes.T == T).sum()), "atm_iv": _atm_of(quotes, T)} for e, T in zip(quotes.expiries, quotes.maturities)]},
            "flat_rmse": flat, "models": rows, "best": rows[0]["id"], "smiles": _smiles(quotes, results), "density": _density(quotes, results)}
    notes = ["Each model's parameters are shared by every expiry and fitted by bounded least squares on the vega-scaled price errors, which equal the implied-volatility errors to first order, so every strike counts evenly in volatility terms.",
             "Prices come from the Lewis (2001) integral of the characteristic function (exact to about 1e-9); the Heston model's characteristic function uses the stable 'little trap' form.",
             "Errors are in volatility points. The flat-volatility benchmark is one Black-Scholes volatility per expiry, the best constant: a model that does not beat it has not captured the smile.",
             "A calibration describes the risk-neutral measure (what the quotes imply), including the market's price for crash risk. It is not a forecast."]
    return data, warnings, notes


def _smiles(quotes: cal.Quotes, results: dict[str, cal.Calibration], max_shown: int = 4) -> list[dict]:
    """The market's smile and each model's, for up to ``max_shown`` expiries spread along the curve (and always the one the density is drawn for)."""
    mats = quotes.maturities
    spread = set(np.round(np.linspace(0, len(mats) - 1, min(max_shown, len(mats)))).astype(int).tolist())
    out = []
    for i in sorted(spread | {int(np.argmin(np.abs(mats * 365.0 - 30.0)))}):
        T = mats[i]
        idx = np.flatnonzero(quotes.T == T)
        idx = idx[np.argsort(quotes.K[idx])]
        f = float(quotes.forward[idx[0]])
        out.append({"expiry": quotes.expiries[i], "dte": float(T * 365.0), "forward": f, "strike": quotes.K[idx], "log_moneyness": np.log(quotes.K[idx] / f), "market": quotes.iv[idx],
                    "models": {mid: c.model_iv[idx] for mid, c in results.items()}})
    return out


def _density(quotes: cal.Quotes, results: dict[str, cal.Calibration]) -> dict:
    """The risk-neutral density of the log return to the expiry nearest 30 days under each model and under Black-Scholes at the at-the-money volatility, with the odds of falls and rises."""
    mats = quotes.maturities
    i = int(np.argmin(np.abs(mats * 365.0 - 30.0)))
    T, expiry = float(mats[i]), quotes.expiries[i]
    S, r, q = quotes.S, quotes.r, quotes.q
    atm = _atm_of(quotes, mats[i])
    sd = atm * np.sqrt(T)
    grid = np.linspace(-5.0 * sd, 5.0 * sd, 161) + (r - q) * T
    drops = np.arange(1, 31, dtype=float)                                       # the left tail: the chance of falling at least this many per cent
    curves, tails, drop_curves = {}, [], {}
    for mid, c in results.items():
        try:
            dens = fourier.risk_neutral_density(c.model, S, T, r, q, s_grid=S * np.exp(grid))
            full = fourier.risk_neutral_density(c.model, S, T, r, q, points=1201)
        except (ValueError, FloatingPointError):
            continue
        x, px = full["x"], full["density"] * full["s"]
        mass = float(integrate.trapezoid(px, x))
        if not np.isfinite(mass) or abs(mass - 1.0) > 0.02:
            continue
        px = px / mass
        mean = float(integrate.trapezoid(x * px, x))
        var = float(integrate.trapezoid((x - mean) ** 2 * px, x))
        skew = float(integrate.trapezoid((x - mean) ** 3 * px, x) / var ** 1.5)
        kurt = float(integrate.trapezoid((x - mean) ** 4 * px, x) / var ** 2 - 3.0)

        def below(level: float, x=x, px=px) -> float:
            keep = x <= level
            return float(integrate.trapezoid(px[keep], x[keep])) if keep.sum() > 1 else 0.0

        curves[mid] = dens["density"] * dens["s"] * 0.01                       # probability in a one-per-cent-wide slice of the log return
        drop_curves[mid] = [below(np.log1p(-d / 100.0)) for d in drops]
        tails.append({"id": mid, "label": c.model.label, "p_down_10": below(np.log(0.9)), "p_down_20": below(np.log(0.8)), "p_up_10": 1.0 - below(np.log(1.1)), "stdev": float(np.sqrt(var)), "skewness": skew, "excess_kurtosis": kurt})
    norm = stats.norm((r - q - 0.5 * atm ** 2) * T, sd)
    tails.append({"id": "bs", "label": "Black-Scholes", "p_down_10": float(norm.cdf(np.log(0.9))), "p_down_20": float(norm.cdf(np.log(0.8))), "p_up_10": float(norm.sf(np.log(1.1))), "stdev": float(sd), "skewness": 0.0, "excess_kurtosis": 0.0})
    return {"expiry": expiry, "dte": T * 365.0, "atm_iv": atm, "x": grid * 100.0, "curves": curves, "bs": norm.pdf(grid) * 0.01, "drop_pct": drops, "drop_curves": drop_curves,
            "bs_drop": [float(norm.cdf(np.log1p(-d / 100.0))) for d in drops], "tails": tails}


# ------------------------------------------------------------------------------------------------------------------------------------ pricing
def atm_vol(chain: pd.DataFrame, expiry) -> float:
    """The at-the-money volatility of one expiry (from its own smile, else interpolated along the term structure)."""
    g = chain[chain["expiry"] == pd.Timestamp(expiry)]
    vol = oa.atm_iv(g)
    return vol if np.isfinite(vol) else oa.interpolate_iv(oa.term_structure(chain), float(g["T"].iloc[0]) * 365.0)


def _digital_call(model: levy.Model, S: float, K: float, T: float, r: float, q: float) -> float:
    """The price of 1 paid if the price ends above ``K``: minus the strike-derivative of the call, ``-dC/dK = e^{-rT} P(S_T > K)``, by a central difference of the (smooth) Lewis price."""
    h = 1e-3 * K
    lo, hi = fourier.lewis_call(model, S, np.array([K - h, K + h]), T, r, q)
    return float(-(hi - lo) / (2.0 * h))


def product_spec(chain: pd.DataFrame, product: str = "vanilla", expiry=None, dte: float = 90.0, strike_pct: float = 100.0, call: bool | None = None, barrier_kind: str = "down-out",
                 barrier_pct: float | None = None, fixings: int = 12, paths: int = 50_000, seed: int = 0) -> dict:
    """What is to be priced, checked against the chain: the listed ``expiry`` (or the one nearest ``dte`` days), the strike as a per cent of the price, call or put (default a call, a put for the American),
    and for a barrier its kind and level (a down barrier below the price, an up barrier above), for an average the number of dates, for the simulated products the paths and the seed. Raises ``ValueError``."""
    if product not in PRODUCTS:
        raise ValueError(f"product must be one of {sorted(PRODUCTS)}")
    if expiry not in (None, ""):
        try:
            when = pd.Timestamp(str(expiry))
        except ValueError:
            raise ValueError("expiry must be a date like 2026-12-18") from None
        if not (chain["expiry"] == when).any():
            raise ValueError(f"expiry {when.date()} is not on the chain; choose one of {[e['expiry'] for e in oa.expiry_list(chain)]}")
    else:
        when = oa.nearest_expiry_to(chain, float(check_range("dte", dte, *LIMITS["dte"])))
    g = chain[chain["expiry"] == when]
    T = float(g["T"].iloc[0])
    if T <= 0:
        raise ValueError("that expiry has passed")
    spot = float(chain["underlying"].iloc[0])
    spec = {"product": product, "expiry": str(pd.Timestamp(when).date()), "T": T, "dte": T * 365.0, "call": bool(product != "american" if call is None else call), "S": spot,
            "K": spot * float(check_range("strike_pct", strike_pct, *LIMITS["strike_pct"])) / 100.0, "forward": float(g["F"].iloc[0]), "steps": int(np.clip(round(T * 252), 10, 252)), "paths": 0, "seed": 0}
    if product in ("barrier", "asian"):
        spec.update(paths=int(check_range("paths", paths, *LIMITS["paths"])), seed=int(check_range("seed", seed, *LIMITS["seed"])))
    if product == "barrier":
        if barrier_kind not in BARRIERS:
            raise ValueError(f"barrier kind must be one of {sorted(BARRIERS)}")
        level = float(check_range("barrier_pct", (90.0 if barrier_kind.startswith("down") else 110.0) if barrier_pct is None else barrier_pct, *LIMITS["barrier_pct"]))
        if barrier_kind.startswith("down") and level >= 100.0:
            raise ValueError("a down barrier must be below the price (under 100% of it)")
        if barrier_kind.startswith("up") and level <= 100.0:
            raise ValueError("an up barrier must be above the price (over 100% of it)")
        spec.update(kind=barrier_kind, H=spot * level / 100.0)
    if product == "asian":
        spec["fixings"] = int(check_range("fixings", fixings, 2, spec["steps"]))
    return spec


def _row(mid: str, label: str, model: levy.Model, spec: dict, fit_error: float | None, want_boundary: bool) -> dict:
    """One line of the price table: the product under ``model``."""
    S, K, T, call, r, q = spec["S"], spec["K"], spec["T"], spec["call"], spec["r"], spec["q"]
    row = {"id": mid, "label": label, "fit_error": fit_error, "price": None, "se": None, "method": "", "note": ""}
    t0 = time.perf_counter()
    try:
        p = spec["product"]
        if p == "vanilla":
            row.update(price=float(fourier.price(model, S, K, T, r, q, call)), method="Fourier (Lewis) integral of the characteristic function")
        elif p == "digital":
            dc = _digital_call(model, S, K, T, r, q)
            row.update(price=dc if call else float(np.exp(-r * T)) - dc, method="Fourier: minus the strike-derivative of the call price")
        elif p == "barrier":
            res = exotics.barrier_mc(model, S, K, spec["H"], T, r, q, call, spec["kind"], spec["steps"], spec["paths"], spec["seed"], bridge=False)
            row.update(price=res["price"], se=res["se"], method=f"Monte Carlo, {spec['paths']:,} paths, monitored on {spec['steps']} dates")
        elif p == "asian":
            res = exotics.asian_arithmetic_mc(model, S, K, T, r, q, call, spec["fixings"], spec["paths"], spec["seed"])
            row.update(price=res["price"], se=res["se"], method=f"Monte Carlo, {spec['paths']:,} paths, {spec['fixings']} fixings, control variate: {res['control']}")
        else:
            if not model.is_levy:
                raise Unavailable("not offered: the Heston model has a second state variable (its variance), so the one-dimensional early-exercise solver does not apply")
            try:
                am = pide.american(model, S, K, T, r, q, call, boundary=want_boundary)
            except ValueError as exc:
                raise Unavailable(str(exc)) from None
            eu = pide.european(model, S, K, T, r, q, call)
            row.update(price=am.price, european=eu.price, premium=am.price - eu.price, method="Integro-differential equation on a grid with the early-exercise constraint", delta=am.delta,
                       boundary={"dte": am.tau * 365.0, "price": am.boundary} if want_boundary else None)
    except Unavailable as why:
        row["note"] = str(why)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        row["note"] = f"could not be priced: {exc}"
    row["seconds"] = time.perf_counter() - t0
    return row


def _bs_rows(spec: dict, sigma: float, smile_vol: float | None, want_boundary: bool) -> list[dict]:
    """The Black-Scholes benchmarks at the at-the-money volatility (and, for a vanilla, the market's own smile volatility at this strike)."""
    S, K, T, call, r, q = spec["S"], spec["K"], spec["T"], spec["call"], spec["r"], spec["q"]
    rows = [_row("bs", f"Black-Scholes at the at-the-money volatility {sigma:.1%}", levy.BlackScholes(sigma=sigma), spec, None, want_boundary)]
    if spec["product"] == "vanilla" and smile_vol is not None:
        rows.insert(0, {"id": "market", "label": f"The market's smile: Black-Scholes at {smile_vol:.1%}, this strike's own volatility", "fit_error": None, "se": None, "price": float(bsm_price(S, K, T, r, q, smile_vol, call)),
                        "method": "Black-Scholes formula at the volatility read off the quoted smile", "note": "", "seconds": 0.0})
    if spec["product"] == "american":
        ref = pde.vanilla(S, K, T, r, q, sigma, call, american=True, boundary=want_boundary)
        rows[-1].update(price=float(ref.price), method="Crank-Nicolson finite differences with the early-exercise constraint (Brennan-Schwartz)", european=float(bsm_price(S, K, T, r, q, sigma, call)))
        rows[-1]["premium"] = rows[-1]["price"] - rows[-1]["european"]
        if want_boundary:
            rows[-1]["boundary"] = {"dte": ref.tau * 365.0, "price": ref.boundary}
    if spec["product"] == "barrier":
        try:
            rows.append({"id": "bs_continuous", "label": f"Black-Scholes, barrier watched continuously (closed form, {sigma:.1%})", "fit_error": None, "se": None,
                         "price": float(exotics.barrier_price(S, K, spec["H"], T, r, q, sigma, call, spec["kind"])), "method": "Reiner-Rubinstein formula",
                         "note": "Daily monitoring is a weaker barrier than continuous monitoring: a knock-out is worth more, a knock-in less.", "seconds": 0.0})
        except ValueError:
            pass
    return rows


def price_report(name: str, chain: pd.DataFrame, spec: dict, quotes: cal.Quotes, results: dict[str, cal.Calibration], failed: dict[str, str], simulated: bool = False, progress=None) -> tuple[dict, list[str], list[str]]:
    """``spec`` (from ``product_spec``) priced under every model of ``results`` and under Black-Scholes, as ``(data, warnings, notes)``: a row per model with its price, standard error where simulated and fit
    error, the benchmark rows, the range across the models (the model risk), the price by strike for a vanilla or digital, and the early-exercise boundary of an American option."""
    say = progress or (lambda text: None)
    spec = {**spec, "r": quotes.r, "q": quotes.q}
    sigma = atm_vol(chain, spec["expiry"])
    smile_vol = oa.iv_lookup(chain, spec["expiry"], spec["K"]) if spec["product"] == "vanilla" else None
    want_boundary = spec["product"] == "american"
    pool = {m: c for m, c in results.items() if c.model.is_levy} if want_boundary else results                       # the early-exercise solver prices the jump models only
    best = min(pool, key=lambda m: pool[m].rmse) if pool else min(results, key=lambda m: results[m].rmse)
    rows = []
    for i, (mid, c) in enumerate(sorted(results.items(), key=lambda kv: kv[1].rmse)):
        say(f"pricing under {c.model.label} ({i + 1} of {len(results)})")
        rows.append(_row(mid, c.model.label, c.model, spec, c.rmse, want_boundary and mid == best))
    say("pricing under Black-Scholes")
    benchmarks = _bs_rows(spec, sigma, smile_vol, want_boundary)
    priced = [r for r in rows if r["price"] is not None]
    bs = next(b for b in benchmarks if b["id"] == "bs")
    summary = None
    if priced:
        lo, hi = min(r["price"] for r in priced), max(r["price"] for r in priced)
        summary = {"low": lo, "high": hi, "range": hi - lo, "range_share": (hi - lo) / abs(bs["price"]) if bs["price"] else None, "best": best, "best_price": next((r["price"] for r in priced if r["id"] == best), None),
                   "bs_price": bs["price"], "vanilla_range_share": None}
        if spec["product"] != "vanilla":                                       # the yardstick for model risk: how far apart the same models are on the vanilla at this strike, which is only how well each fits
            try:
                S, K, T, call, r, q = spec["S"], spec["K"], spec["T"], spec["call"], spec["r"], spec["q"]
                prices = [float(fourier.price(results[row["id"]].model, S, K, T, r, q, call)) for row in priced]
                reference = float(bsm_price(S, K, T, r, q, sigma, call))
                summary["vanilla_range_share"] = (max(prices) - min(prices)) / reference if reference > 0 else None
            except (ValueError, FloatingPointError):
                pass
    warnings = []
    if simulated:
        warnings.append("The chain is itself a model: the fitted models reproduce a simulated surface, so the spread between their prices measures how much the product depends on the model family, and says nothing about a market.")
    for r in rows:
        if r.get("fit_error") is not None and r["fit_error"] > 0.02:
            warnings.append(f"{r['label']} fits the quotes only to {r['fit_error'] * 100:.1f} volatility points: weigh its price accordingly.")
    for r in rows:
        if r["price"] is None and r["note"]:
            warnings.append(f"{r['label']}: {r['note']}")
    for mid, why in failed.items():
        warnings.append(f"{levy.MODELS[mid].label} could not be fitted: {why}")
    if summary and summary["vanilla_range_share"] is not None and summary["range_share"] is not None:
        if summary["range_share"] > 1.5 * summary["vanilla_range_share"] + 0.01:
            warnings.append(f"The models differ on this product by {summary['range_share']:.1%} of the Black-Scholes price against {summary['vanilla_range_share']:.1%} on the vanilla at this strike: part of the spread is model risk beyond how well they fit.")
        else:
            warnings.append(f"The models differ on this product by {summary['range_share']:.1%} of the Black-Scholes price and by {summary['vanilla_range_share']:.1%} on the vanilla at this strike: here the spread is mostly how well each fits the quotes, not extra model risk.")
    if spec["product"] == "barrier":
        warnings.append(f"Every simulated price is monitored on {spec['steps']} trading dates; a barrier that is touched between two dates is missed, which makes a knock-out dearer and a knock-in cheaper than watching it continuously.")
    notes = [PRODUCTS[spec["product"]]["text"],
             "Every model was fitted to the same out-of-the-money quotes. On a vanilla the models differ by how well each fits them (the fit error is in the table); on another product they differ by that and by what the quotes do not determine. "
             "The range between them is this product's model risk, fit error included: compare it with the range of the same models on the vanilla at this strike, which is the part that is only fit.",
             "Simulated prices carry a standard error (one sigma); the models are priced on the same seed, which makes their differences less noisy than their levels."]
    curve = _strike_curve(spec, results, sigma) if spec["product"] in ("vanilla", "digital") else None
    data = {"underlying": name, "spec": {k: v for k, v in spec.items() if k != "S"} | {"spot": spec["S"]}, "sigma_atm": sigma, "models": rows, "benchmarks": benchmarks, "summary": summary, "curve": curve,
            "boundary": _boundary(rows, benchmarks)}
    return data, warnings, notes


def _boundary(rows: list[dict], benchmarks: list[dict], points: int = 40) -> dict | None:
    """The early-exercise boundary of the best-fitting jump model and of Black-Scholes, as functions of the days left, on one grid of days (the two solvers step time differently)."""
    lines = {}
    for r in rows + benchmarks:
        b = r.get("boundary")
        if b is None:
            continue
        days, px = np.asarray(b["dte"], dtype=float), np.asarray(b["price"], dtype=float)
        ok = np.isfinite(px) & (days > 0)
        if ok.sum() >= 3:
            lines[r["id"]] = (r["label"], days[ok], px[ok])
    if not lines:
        return None
    grid = np.linspace(max(v[1][0] for v in lines.values()), min(v[1][-1] for v in lines.values()), points)
    return {mid: {"label": label, "dte": grid, "price": np.interp(grid, days, px)} for mid, (label, days, px) in lines.items()}


def _strike_curve(spec: dict, results: dict, sigma: float) -> dict:
    """The price of the vanilla or digital across strikes (80 to 120 per cent of the price) under each model and Black-Scholes."""
    S, T, r, q, call = spec["S"], spec["T"], spec["r"], spec["q"], spec["call"]
    pct = np.linspace(80.0, 120.0, 21)
    ks = S * pct / 100.0
    curves, models = {}, {mid: c.model for mid, c in results.items()} | {"bs": levy.BlackScholes(sigma=sigma)}
    for mid, m in models.items():
        try:
            if spec["product"] == "vanilla":
                curves[mid] = np.asarray(fourier.price(m, S, ks, T, r, q, call), dtype=float)
            else:
                dc = np.array([_digital_call(m, S, k, T, r, q) for k in ks])
                curves[mid] = dc if call else float(np.exp(-r * T)) - dc
        except (ValueError, FloatingPointError):
            continue
    return {"strike_pct": pct, "strike": ks, "curves": curves}


# ------------------------------------------------------------------------------------------------------------------------------------ hedging
def hedge_inputs(chain: pd.DataFrame, spec: dict, vol_pct: float | None = None, cost_bp: float = 5.0, steps: int | None = None, risk_aversion: float = 1.0, paths: int = 10_000, seed: int = 0) -> dict:
    """The settings of a hedging report checked against the chain: the rate and dividend yield of the expiry, the volatility (default the at-the-money one), the cost of one trade (basis points of the value
    traded), the rebalancing dates (default every trading day), the risk aversion (in units of the option's price), the simulated paths and the seed. Raises ``ValueError``."""
    g = chain[chain["expiry"] == pd.Timestamp(spec["expiry"])]
    sigma = atm_vol(chain, spec["expiry"]) if vol_pct in (None, "") else float(check_range("vol", vol_pct, *LIMITS["vol_pct"])) / 100.0
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("the chain has no at-the-money volatility for that expiry: type one in")
    return {"r": float(g["rate"].median()), "q": float(g["dividend"].median()), "sigma": sigma, "cost": float(check_range("cost_bp", cost_bp, *LIMITS["cost_bp"])) / 1e4,
            "steps": int(check_range("steps", spec["steps"] if steps in (None, "") else steps, *LIMITS["steps"])), "ra": float(check_range("risk_aversion", risk_aversion, *LIMITS["risk_aversion"])),
            "paths": int(check_range("paths", paths, *LIMITS["hedge_paths"])), "seed": int(check_range("seed", seed, *LIMITS["seed"]))}


def hedge_report(name: str, spec: dict, inputs: dict, progress=None) -> tuple[dict, list[str], list[str]]:
    """What hedging the option of ``spec`` costs under ``inputs`` (from ``hedge_inputs``), as ``(data, warnings, notes)``: Leland's price at several rebalancing frequencies, the Whalley-Wilmott band
    around the delta, three hedging rules simulated on the same paths (with the distribution of their profit and loss) and the utility-based writer's and buyer's prices."""
    say = progress or (lambda text: None)
    S, K, T, call = spec["S"], spec["K"], spec["T"], spec["call"]
    r, q, sigma, cost, steps, ra, paths, seed = (inputs[k] for k in ("r", "q", "sigma", "cost", "steps", "ra", "paths", "seed"))
    premium = float(bsm_price(S, K, T, r, q, sigma, call))
    gamma = ra / max(premium, 1e-6)                                            # risk aversion per dollar: a loss of the whole premium is worth `ra`
    say("Leland's price at several rebalancing frequencies")
    leland = []
    for n in sorted({2, 4, 8, 13, 26, 52, 104, 252, 504} | {steps}):
        dt = T / n
        le = costs.leland_number(sigma, dt, cost)
        row = {"steps": n, "le": le, "seller_vol": costs.leland_sigma(sigma, dt, cost, "sell"), "seller_price": costs.leland_price(S, K, T, r, q, sigma, cost, dt, call, "sell"), "buyer_vol": None, "buyer_price": None}
        if le < 1.0:
            row.update(buyer_vol=costs.leland_sigma(sigma, dt, cost, "buy"), buyer_price=costs.leland_price(S, K, T, r, q, sigma, cost, dt, call, "buy"))
        leland.append(row)
    say("the Whalley-Wilmott band")
    spots = np.linspace(0.85 * S, 1.15 * S, 61)
    delta, half = costs.whalley_wilmott_band(spots, K, T, r, q, sigma, cost, gamma, call)
    band = {"spot": spots, "delta": delta, "low": delta - half, "high": delta + half, "half_width_at_spot": float(np.interp(S, spots, half))}
    sims, raw = [], {}
    for i, rule in enumerate(("delta", "leland", "band")):
        say(f"simulating the {rule} hedge ({i + 1} of 3): {paths:,} paths, {steps} dates")
        res = costs.simulate_hedging(S, K, T, r, q, sigma, cost, steps, paths, rule, call, gamma, seed=seed)
        raw[rule] = res.pop("pnl")
        res.update(q05=float(np.quantile(raw[rule], 0.05)), q50=float(np.quantile(raw[rule], 0.5)), q95=float(np.quantile(raw[rule], 0.95)), text=HEDGE_RULES[rule])
        sims.append(res)
    allp = np.concatenate(list(raw.values()))
    lo, hi = float(np.quantile(allp, 0.001)), float(np.quantile(allp, 0.999))
    edges, counts, outside = np.linspace(lo, hi, 31), {}, {}
    for rule, pnl in raw.items():                                              # the extremes are left out (and counted), not piled into the end bins, where they would look like a second hump
        inside = (pnl >= lo) & (pnl <= hi)
        n_in = np.histogram(pnl[inside], bins=edges)[0]
        counts[rule] = n_in / max(int(n_in.sum()), 1)
        outside[rule] = float(1.0 - inside.mean())
    say("the utility-based price in a binomial market")
    dpz = costs.dpz_prices(S * np.exp(-q * T), K, T, r, sigma, cost, gamma, n_steps=40, call=call, bands=True)         # the tree has no dividend: it hedges with the prepaid forward S e^{-qT}, which pays none
    warnings = []
    if leland[-1]["le"] >= 1.0:
        warnings.append(f"Leland's number is {leland[-1]['le']:.2f} at the most frequent rebalancing shown: at this cost a buyer cannot replicate the option at a profit.")
    if cost * 1e4 > 50:
        warnings.append("A cost above 50 basis points a trade is a wide spread for a listed ETF: the results will look extreme.")
    notes = ["Cost is the proportional cost of one trade, in basis points of the value traded (half the relative bid-ask spread). Leland's own round-trip cost is twice it.",
             "Leland (1985): rebalancing every dt turns the cost into extra volatility, sigma^2 (1 + Le) for the seller and sigma^2 (1 - Le) for the buyer, Le = 2 sqrt(2/pi) cost / (sigma sqrt(dt)). It prices a convex payoff correctly to first order only.",
             "Whalley and Wilmott (1997): with exponential utility the best hedge keeps the holding within a band around the Black-Scholes delta, of half-width (3 e^{-r tau} cost S Gamma^2 / (2 risk aversion))^(1/3), and trades only at its edge.",
             "The simulation sells the option at its Black-Scholes price (Leland's own for that rule) and hedges at the chosen dates on paths of the model with the typed volatility, paying the cost on every trade, the first purchase and the final liquidation.",
             f"Risk aversion {ra:g} is in units of the option's price ({premium:.2f}): the exponent's coefficient is {gamma:.3g} per dollar. Hodges and Neuberger (1989) and Davis, Panas and Zariphopoulou (1993) give the bid and ask of the writer and the buyer.",
             "The binomial tree has no dividend: it hedges with the prepaid forward S e^(-qT), which pays none, so its frictionless price is the Cox-Ross-Rubinstein price of the same option (about the Black-Scholes price, to within the 40-step tree's error)."]
    data = {"underlying": name, "inputs": {"spot": S, "strike": K, "expiry": spec["expiry"], "dte": spec["dte"], "call": call, "rate": r, "dividend": q, "vol": sigma, "cost": cost, "cost_bp": cost * 1e4, "steps": steps,
                                          "risk_aversion": ra, "gamma": gamma, "paths": paths, "seed": seed},
            "premium": premium, "leland": leland, "band": band, "simulation": sims, "histogram": {"edges": edges, "share": counts, "outside": outside}, "dpz": dpz}
    return data, warnings, notes


# ------------------------------------------------------------------------------------------------------------------------------------ text
def _money(v, d: int = 3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:,.{d}f}"


def _pts(v) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:.2f}"


def _param_text(m: dict) -> str:
    out = []
    for k, v in m["params"].items():
        se = m["std_errors"].get(k)
        tag = " (at the edge of its range)" if k in m["at_bounds"] else (" (not pinned down)" if se is not None and np.isfinite(se) and abs(se) > 5 * abs(v) else "")
        out.append(f"{k} {v:.4g}" + (f" ±{se:.2g}" if se is not None and np.isfinite(se) and not tag else "") + tag)
    return ", ".join(out)


def format_fit(data: dict, warnings: list[str] = ()) -> str:
    """The fit as text: the models ranked, their parameters, and what each implies about a fall."""
    d, q = data["density"], data["quotes"]
    lines = [f"{data['underlying']}: option models fitted to {q['n']} out-of-the-money quotes, {len(q['expiries'])} expiries from {q['expiries'][0]['dte']:.0f} to {q['expiries'][-1]['dte']:.0f} days (price {data['spot']:,.2f})", "",
             f"{'model':32s} {'error':>7s} {'largest':>8s} {'explained':>10s}  parameters (± one standard error)", f"{'':32s} {'(pts)':>7s} {'(pts)':>8s}"]
    for m in data["models"]:
        lines.append(f"{levy.MODELS[m['id']].label:32s} {_pts(m['rmse']):>7s} {_pts(m['max_error']):>8s} {m['explained'] * 100 if m['explained'] is not None else float('nan'):9.0f}%  {_param_text(m)}")
    lines += [f"{'flat volatility (to beat)':32s} {_pts(data['flat_rmse']):>7s}", "", f"What each model implies for the price in {d['dte']:.0f} days (risk-neutral, not a forecast):",
              f"{'model':32s} {'fall 10%':>9s} {'fall 20%':>9s} {'rise 10%':>9s} {'skew':>7s} {'ex.kurt':>8s}"]
    for t in d["tails"]:
        lines.append(f"{t['label']:32s} {t['p_down_10'] * 100:8.2f}% {t['p_down_20'] * 100:8.2f}% {t['p_up_10'] * 100:8.2f}% {t['skewness']:7.2f} {t['excess_kurtosis']:8.2f}")
    return "\n".join(lines + [""] + [f"Watch: {w}" for w in warnings])


def format_price(data: dict, warnings: list[str] = ()) -> str:
    """The prices of one product under every model as text, with the range across the models."""
    sp = data["spec"]
    head = f"{PRODUCTS[sp['product']]['label']}: {'call' if sp['call'] else 'put'}, strike {sp['K']:,.2f} ({sp['K'] / sp['spot'] * 100:.0f}% of the price {sp['spot']:,.2f}), expiry {sp['expiry']} ({sp['dte']:.0f} days)"
    if sp["product"] == "barrier":
        head += f", {sp['kind'].replace('-', ' and ')} at {sp['H']:,.2f}"
    if sp["product"] == "asian":
        head += f", averaged on {sp['fixings']} dates"
    american = sp["product"] == "american"
    lines = [head, "", f"{'source':58s} {'price':>10s} {'± s.e.':>8s} {'fit (pts)':>9s}" + (f" {'European':>10s} {'premium':>9s}" if american else "")]
    for r in data["benchmarks"] + data["models"]:
        price = "not priced" if r["price"] is None else _money(r["price"], 4 if sp["product"] == "digital" else 3)
        lines.append(f"{r['label'][:58]:58s} {price:>10s} {_money(r['se']) if r['se'] else '':>8s} {_pts(r['fit_error']) if r['fit_error'] is not None else '':>9s}"
                     + (f" {_money(r.get('european')):>10s} {_money(r.get('premium')):>9s}" if american and r["price"] is not None else ""))
    s = data["summary"]
    if s:
        lines += ["", f"Range across the fitted models: {_money(s['low'])} to {_money(s['high'])}" + (f" ({s['range_share'] * 100:.1f}% of the Black-Scholes price {_money(s['bs_price'])})" if s["range_share"] is not None else "")
                  + (f"; the same models differ by {s['vanilla_range_share'] * 100:.1f}% on the vanilla at this strike" if s.get("vanilla_range_share") is not None else "")
                  + f"; the best-fitting model that prices it is {levy.MODELS[s['best']].label}."]
    return "\n".join(lines + [""] + [f"Watch: {w}" for w in warnings])


def format_hedge(data: dict, warnings: list[str] = ()) -> str:
    """The hedging report as text."""
    i, dpz = data["inputs"], data["dpz"]
    lines = [f"Hedging a {'call' if i['call'] else 'put'}: strike {i['strike']:,.2f} ({i['strike'] / i['spot'] * 100:.0f}% of {i['spot']:,.2f}), expiry {i['expiry']} ({i['dte']:.0f} days), volatility {i['vol'] * 100:.1f}%, "
             f"{i['cost_bp']:.1f} bp a trade, {i['steps']} rebalancing dates", "", f"Black-Scholes price with free trading: {_money(data['premium'])}", "",
             f"{'rebalancing dates':>18s} {'Leland no.':>10s} {'seller pays':>12s} {'buyer pays':>11s}"]
    for x in data["leland"]:
        lines.append(f"{x['steps']:18d} {x['le']:10.3f} {_money(x['seller_price']):>12s} {_money(x['buyer_price']) if x['buyer_price'] is not None else 'cannot hedge':>11s}" + ("   <- yours" if x["steps"] == i["steps"] else ""))
    lines += ["", f"Whalley-Wilmott no-trade band at today's price: ±{data['band']['half_width_at_spot']:.3f} shares around the delta", "", "Simulated hedging of one sold option:",
              f"{'rule':14s} {'premium':>9s} {'mean P&L':>9s} {'std':>7s} {'5th pct':>8s} {'95th pct':>9s} {'cost':>7s} {'trades':>7s}"]
    for x in data["simulation"]:
        lines.append(f"{x['strategy']:14s} {_money(x['premium'], 2):>9s} {x['mean']:9.2f} {x['std']:7.2f} {x['q05']:8.2f} {x['q95']:9.2f} {x['cost']:7.2f} {x['trades']:7.1f}")
    lines += ["", f"Utility-based prices (binomial market): writer {_money(dpz['writer'])}, buyer {_money(dpz['buyer'])}, frictionless {_money(dpz['crr'])}; the writer's best holding is not to trade between {dpz['band'][0]:.3f} and {dpz['band'][1]:.3f} shares"]
    return "\n".join(lines + [""] + [f"Watch: {w}" for w in warnings])
