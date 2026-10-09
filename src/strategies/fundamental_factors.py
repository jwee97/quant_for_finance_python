"""Fundamental-factor strategies: value, quality, momentum and estimate revisions, discounted cash flow, the optimal alpha model, and its contextual and nonlinear versions.

These are the equity strategies of Qian, Hua and Sorensen, *Quantitative Equity Portfolio Management*, built on :mod:`src.equity`. They read company statements from a file you supply
(``data/user/fundamentals.csv``, see :mod:`src.equity.fundamentals` for the columns) and treat them as point-in-time data: a figure is used only after its filing date, and goes stale. A model that needs
the file refuses to run without it, and says which columns a factor wants. The price-based momentum factors (``ret1``, ``ret9``, ``adj_ret9``) need no file.

    fundamental_value       cash flow, EBITDA and earnings to value, payout, net financing, book to price, sales to enterprise value: one factor, or the equal-weighted composite
    fundamental_quality     RNOA, CFROI, operating leverage, accruals, capital expenditure, external financing and share issuance
    fundamental_momentum    the one-month reversal, nine-month momentum, risk-adjusted momentum, and (with analyst columns) earnings revisions, diffusion and growth revisions
    fundamental_dcf         discounted cash flow and its multipath (Monte Carlo) version: value per share against the price
    fundamental_alpha       every factor you ask for, weighed walk-forward by the covariance of their ICs (optionally made orthogonal first, optionally per bucket of a context)
    fundamental_nonlinear   linear factors plus squares, products and conditional terms, forecast with trailing Fama-MacBeth slopes

Factors are oriented by the direction the literature expects (a factor whose high end is bad is turned round), standardised across stocks on every date and averaged. The orientation is a prior: the alpha model
learns the weights, signs included, from what the factors have earned.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..equity import dcf as _dcf                                     # noqa: F401  (registers the valuation factors)
from ..equity.alpha_model import combine_factors, gram_schmidt, optimal_alpha, standardize, symmetric_orthogonalize
from ..equity.contextual import buckets, context_variable, contextual_alpha, fama_macbeth_forecast, nonlinear_features
from ..equity.factors import FACTORS, compute, names as factor_names
from ..equity.fundamentals import USER_DATA, FactorInputs, Fundamentals
from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..signals.transform import cross_sectional_zscore
from ..utils.dates import rebalance_dates

_CACHE: dict = {}


def load_fundamentals(path: str, lag_days: int, expiry: int) -> Fundamentals:
    """The statements file, read once per change of the file."""
    p = Path(path)
    p = p if p.is_absolute() else USER_DATA / p
    if not p.exists():
        return Fundamentals.from_csv(path, lag_days, expiry)                              # raises the explanation of what the file should contain
    key = (str(p), p.stat().st_mtime_ns, lag_days, expiry)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = Fundamentals.from_csv(p, lag_days, expiry)
    return _CACHE[key]


def _no_fundamentals(prices: pd.DataFrame) -> Fundamentals:
    """An empty table: the price-only factors need nothing else."""
    return Fundamentals(pd.DataFrame({"ticker": [], "available": pd.to_datetime([])}))


class _FundamentalModel(ForecastModel):
    family = "fundamental"
    position_mode = "cross_sectional"
    horizon = 21
    min_assets = 6                                                                           # a cross-section of fewer stocks says nothing about which are cheap

    path, lag_days, expiry = "fundamentals.csv", 60, 400

    def inputs(self, data, optional: bool = False) -> FactorInputs:
        try:
            fund = load_fundamentals(self.path, self.lag_days, self.expiry)
        except KeyError:
            if not optional:
                raise
            fund = _no_fundamentals(data.prices)
        return FactorInputs(data.prices, fund)

    def _check(self, lag_days, expiry):
        if lag_days < 0 or expiry < 1:
            raise ValueError("lag_days >= 0 and expiry >= 1")


def _computable(x: FactorInputs, names: list[str]) -> tuple[list[str], dict]:
    ok, lacking = [], {}
    for n in names:
        gap = x.fund.missing(*FACTORS[n].needs)
        (lacking.__setitem__(n, gap) if gap else ok.append(n))
    return ok, lacking


def _oriented_composite(x: FactorInputs, names: list[str], investable: pd.DataFrame, oriented: bool = True, rows=None) -> pd.DataFrame:
    """Each factor standardised across stocks (and turned round if the literature expects its high end to be bad), averaged over the factors a stock has, standardised again. ``rows`` limits the dates."""
    z = []
    for n in names:
        f = FACTORS[n].fn(x)
        f = f.loc[rows] if rows is not None else f
        z.append(standardize(f, investable.reindex(f.index)) * (FACTORS[n].sign if oriented else 1.0))
    stack = np.stack([f.to_numpy() for f in z])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)                                                    # a stock missing some factors is judged on the rest
    return cross_sectional_zscore(pd.DataFrame(mean, index=z[0].index, columns=z[0].columns))


def _style_score(model: _FundamentalModel, data, style_names: list[str], factor: str, oriented: bool, price_only: tuple = ()) -> pd.DataFrame:
    if factor != "composite" and factor not in style_names:
        raise ValueError(f"factor must be 'composite' or one of {style_names}")
    wanted = style_names if factor == "composite" else [factor]
    x = model.inputs(data, optional=all(n in price_only for n in wanted) or (factor == "composite" and bool(price_only)))
    ok, lacking = _computable(x, wanted)
    if not ok:
        detail = "; ".join(f"{n} needs {gap}" for n, gap in lacking.items())
        raise KeyError(f"{model.name}: nothing can be computed from the fundamentals file ({detail})")
    if factor != "composite":
        return (FACTORS[factor].fn(x) * (FACTORS[factor].sign if oriented else 1.0)).where(data.investable)
    return _oriented_composite(x, ok, data.investable, oriented).where(data.investable)


VALUE_FACTORS = factor_names("value")
QUALITY_FACTORS = factor_names("quality")
MOMENTUM_FACTORS = factor_names("momentum") + factor_names("estimates")
PRICE_FACTORS = tuple(factor_names("momentum"))
VALUATION_FACTORS = ("dcf_upside", "mdcf_upside", "mdcf_prob")


@register_model("fundamental_value", "fundamental", "Value from company statements: cash flow, EBITDA, earnings, payout, net financing, book value or sales against what the market asks (one factor or the composite)")
class FundamentalValue(_FundamentalModel):
    """Cheap stocks, measured by what a dollar of market value buys of the company's cash flow, EBITDA, earnings, payout, book value or sales, have earned more than expensive ones, a premium that is either
    compensation for distress or the market extrapolating bad news too far. Enterprise value (equity plus debt and preferred less cash) is the base for the operating measures, market value for the
    shareholder measures. A factor is the ratio as it stood on the date, from the latest filing public by then. ``factor`` picks one of the eight or the equal-weighted composite. Needs
    ``data/user/fundamentals.csv``."""

    name = "fundamental_value"

    def __init__(self, factor: str = "composite", path: str = "fundamentals.csv", lag_days: int = 60, expiry: int = 400, oriented: bool = True):
        self._check(lag_days, expiry)
        self.factor, self.path, self.lag_days, self.expiry, self.oriented = factor, path, lag_days, expiry, oriented

    def score(self, data):
        return _style_score(self, data, VALUE_FACTORS, self.factor, self.oriented)


@register_model("fundamental_quality", "fundamental", "Quality from company statements: return on operating assets, CFROI, operating leverage, accruals, capital expenditure, external financing and share issuance")
class FundamentalQuality(_FundamentalModel):
    """Companies that earn high returns on the capital they use, and fund their growth from inside, have earned more than those that grow by absorbing capital: accruals and capital expenditure
    above the firm's own history, new debt and new shares all signal a firm that is spending more than it earns, which the market is slow to price. ``factor`` picks one of ten or the composite, each
    turned round where the literature expects the high end to be bad. Needs ``data/user/fundamentals.csv``."""

    name = "fundamental_quality"

    def __init__(self, factor: str = "composite", path: str = "fundamentals.csv", lag_days: int = 60, expiry: int = 400, oriented: bool = True):
        self._check(lag_days, expiry)
        self.factor, self.path, self.lag_days, self.expiry, self.oriented = factor, path, lag_days, expiry, oriented

    def score(self, data):
        return _style_score(self, data, QUALITY_FACTORS, self.factor, self.oriented)


@register_model("fundamental_momentum", "fundamental", "Momentum and revisions: the one-month reversal, nine-month price momentum (raw and per unit of risk) and, with analyst columns, earnings revisions and diffusion")
class FundamentalMomentum(_FundamentalModel):
    """Stocks keep moving the way they have for about nine months (skipping the latest month, which reverses), and analysts' earnings estimates drift the same way for months because they are revised a
    little at a time. The three price factors need no file; the three estimate factors need ``eps_fy1``, ``n_up``, ``n_down``, ``n_estimates`` and ``ltg`` in ``data/user/fundamentals.csv``. The composite
    uses the price factors, plus the estimate factors when the file has their columns."""

    name = "fundamental_momentum"

    def __init__(self, factor: str = "composite", path: str = "fundamentals.csv", lag_days: int = 60, expiry: int = 400, oriented: bool = True):
        self._check(lag_days, expiry)
        self.factor, self.path, self.lag_days, self.expiry, self.oriented = factor, path, lag_days, expiry, oriented

    def score(self, data):
        return _style_score(self, data, MOMENTUM_FACTORS, self.factor, self.oriented, price_only=PRICE_FACTORS)


@register_model("fundamental_dcf", "fundamental", "Discounted cash flow value per share against the price, as a point estimate or as the median of a Monte Carlo of the uncertain inputs (multipath DCF)")
class FundamentalDCF(_FundamentalModel):
    """A stock is worth the cash it will pay out, discounted. Free cash flow grows from a first-stage rate (the analysts' long-term growth, or the last three years' sales growth) fading to a terminal rate,
    and is discounted at a rate built from the stock's own beta. The multipath version draws growth, starting cash flow, discount rate and terminal rate from distributions centred on those inputs, values
    every draw and uses the median value, or the share of draws above the price, so a stock whose value is high only if everything goes right ranks below one that is cheap across the range.
    ``factor`` is ``dcf_upside``, ``mdcf_upside`` or ``mdcf_prob``. Needs ``data/user/fundamentals.csv``; values are refreshed monthly."""

    name = "fundamental_dcf"

    def __init__(self, factor: str = "mdcf_upside", path: str = "fundamentals.csv", lag_days: int = 60, expiry: int = 400, risk_free: float = 0.03, equity_premium: float = 0.05,
                 terminal_growth: float = 0.025, years: int = 10, paths: int = 400):
        self._check(lag_days, expiry)
        if factor not in VALUATION_FACTORS or risk_free < 0 or equity_premium <= 0 or not 0 <= terminal_growth < 0.08 or years < 2 or paths < 50:
            raise ValueError(f"factor in {VALUATION_FACTORS}, risk_free >= 0, equity_premium > 0, 0 <= terminal_growth < 0.08, years >= 2, paths >= 50")
        self.factor, self.path, self.lag_days, self.expiry = factor, path, lag_days, expiry
        self.risk_free, self.equity_premium, self.terminal_growth, self.years, self.paths = risk_free, equity_premium, terminal_growth, years, paths

    def score(self, data):
        x = self.inputs(data)
        params = dict(risk_free=self.risk_free, equity_premium=self.equity_premium, terminal_growth=self.terminal_growth, years=self.years)
        if self.factor != "dcf_upside":
            params["paths"] = self.paths
        return compute(self.factor, x, **params).where(data.investable)


STYLES = ("value", "quality", "momentum", "estimates", "valuation")


def _build_factors(spec: str, x: FactorInputs, grid: pd.DatetimeIndex, investable: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The factors of a comma-separated list, each standardised across stocks on the decision dates and turned round where the literature expects its high end to be bad.

    A token that names a factor gives that factor; a token that names a style (``value``, ``quality``, ``momentum``, ``estimates``, ``valuation``) or ``all`` gives the equal-weighted composite of the
    style's factors that the file supports (the factors it lacks columns for are left out). Several tokens give several factors, which is what the alpha model then weighs."""
    out: dict[str, pd.DataFrame] = {}
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token == "all" or token in STYLES:
            for style in (STYLES if token == "all" else (token,)):
                ok, _ = _computable(x, factor_names(style))
                if ok:
                    out[style] = _oriented_composite(x, ok, investable, True, grid)
        elif token in FACTORS:
            ok, lacking = _computable(x, [token])
            if not ok:
                raise KeyError(f"the factor {token} needs the columns {lacking[token]}, which the fundamentals file does not have")
            out[token] = standardize(FACTORS[token].fn(x).loc[grid], investable) * FACTORS[token].sign
        else:
            raise ValueError(f"unknown factor or style '{token}'; styles: {', '.join(STYLES)}, all")
    if not out:
        raise KeyError("none of the requested factors can be computed from the fundamentals file")
    return out


def _grid(data) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """The monthly decision dates and the return from each to the next (missing for the last)."""
    grid = rebalance_dates(data.index, "monthly")
    px = data.prices.loc[grid]
    return grid, px.shift(-1) / px - 1.0


def _to_daily(grid_frame: pd.DataFrame, data) -> pd.DataFrame:
    """A table on the decision dates held until the next one."""
    return grid_frame.reindex(data.index).ffill().where(data.investable)


@register_model("fundamental_alpha", "fundamental", "The optimal alpha model on fundamental factors: weights from the covariance of their information coefficients, factors made orthogonal first, optionally per bucket of value, growth or earnings variability")
class FundamentalAlpha(_FundamentalModel):
    """Qian, Hua and Sorensen's alpha model: every month, standardise each factor across stocks, make them independent of each other (Gram-Schmidt in the order listed, or symmetrically), measure each factor's
    information coefficient against the next month's return, and weigh the factors by ``Sigma^-1 mu`` of those ICs over the last ``window`` months (the weights that maximise the information ratio of the
    composite). Only ICs whose month has ended are used. ``factors`` lists factors and styles: a style (``value``, ``quality``, ``momentum``, ``estimates``, ``valuation`` or ``all``) is the composite of its
    factors, so the default weighs three composites. With ``context`` set the whole model runs separately in each tercile of that variable (``value``, ``growth``, ``earnings_variability`` or ``size``), so
    the factors can matter differently for cheap and dear stocks. ``weights='equal'`` skips the learning. Needs ``data/user/fundamentals.csv``."""

    name = "fundamental_alpha"
    horizon = 21

    def __init__(self, factors: str = "value,quality,momentum", weights: str = "optimal", orthogonalize: str = "gram_schmidt", context: str = "", window: int = 36, min_obs: int = 12,
                 shrink: float = 0.3, nonnegative: bool = False, path: str = "fundamentals.csv", lag_days: int = 60, expiry: int = 400):
        self._check(lag_days, expiry)
        if weights not in ("optimal", "equal") or orthogonalize not in ("none", "gram_schmidt", "symmetric") or window < 12 or min_obs < 6 or not 0 <= shrink <= 1:
            raise ValueError("weights in (optimal, equal), orthogonalize in (none, gram_schmidt, symmetric), window >= 12, min_obs >= 6, 0 <= shrink <= 1")
        if context and context not in ("value", "growth", "earnings_variability", "size"):
            raise ValueError("context must be empty or one of value, growth, earnings_variability, size")
        self.factors, self.weights, self.orthogonalize, self.context = factors, weights, orthogonalize, context
        self.window, self.min_obs, self.shrink, self.nonnegative = window, min_obs, shrink, nonnegative
        self.path, self.lag_days, self.expiry = path, lag_days, expiry

    def score(self, data):
        x = self.inputs(data)
        grid, forward = _grid(data)
        investable = data.investable.loc[grid]
        std = _build_factors(self.factors, x, grid, investable)
        min_assets = int(min(10, max(4, data.returns.shape[1] // 2)))
        order = {"none": False, "gram_schmidt": "gram_schmidt", "symmetric": "symmetric"}[self.orthogonalize]
        if self.weights == "equal":
            std = gram_schmidt(std) if order == "gram_schmidt" else symmetric_orthogonalize(std) if order == "symmetric" else std
            alpha = combine_factors(std, pd.Series(1.0 / len(std), index=list(std)))
        elif self.context:
            std = gram_schmidt(std) if order == "gram_schmidt" else symmetric_orthogonalize(std) if order == "symmetric" else std
            context = buckets(context_variable(self.context, x).loc[grid], 3, investable)
            alpha = contextual_alpha(std, forward, context, 1, self.window, self.min_obs, self.shrink, self.nonnegative, None, max(min_assets, 5)).alpha
        else:
            alpha = optimal_alpha(std, forward, 1, self.window, self.min_obs, self.shrink, order, None, self.nonnegative, None, min_assets=min_assets).alpha
        return _to_daily(alpha, data)


def _parse_terms(spec: str) -> list[tuple]:
    terms = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        parts = token.split(":")
        kind, args = parts[0], parts[1:]
        if kind == "quadratic" and len(args) == 1 or kind == "interaction" and len(args) == 2 or kind == "conditional" and len(args) in (2, 3):
            terms.append((kind, *args))
        else:
            raise ValueError(f"cannot read the term '{token}': use quadratic:f, interaction:f:g or conditional:f:g[:high|low]")
    return terms


@register_model("fundamental_nonlinear", "fundamental", "Nonlinear effects: factors plus their squares, products and conditional versions, forecast with the trailing average cross-sectional (Fama-MacBeth) slopes")
class FundamentalNonlinear(_FundamentalModel):
    """Returns are not always a straight line in a factor (capital expenditure is the textbook case: both starved and gorged firms do worse than the middle). Each month the next month's returns are
    regressed on the linear factors and the terms given (``quadratic:f``, ``interaction:f:g``, ``conditional:f:g:high``) across stocks, and a stock's forecast is the trailing mean of those slopes (those
    whose month has ended) times its terms. A term that earns nothing gets a slope near zero and so little weight. Needs ``data/user/fundamentals.csv``."""

    name = "fundamental_nonlinear"
    horizon = 21

    def __init__(self, factors: str = "b2p,rnoa,icapx,ret9", terms: str = "quadratic:icapx", window: int = 60, min_obs: int = 24, path: str = "fundamentals.csv", lag_days: int = 60, expiry: int = 400):
        self._check(lag_days, expiry)
        if window < 12 or min_obs < 6:
            raise ValueError("window >= 12 and min_obs >= 6")
        self.factors, self.terms, self.window, self.min_obs = factors, terms, window, min_obs
        self.path, self.lag_days, self.expiry = path, lag_days, expiry
        _parse_terms(terms)

    def score(self, data):
        x = self.inputs(data)
        grid, forward = _grid(data)
        investable = data.investable.loc[grid]
        linear = _build_factors(self.factors, x, grid, investable)
        terms = _parse_terms(self.terms)
        for term in terms:
            for referred in (term[1], term[2]) if term[0] != "quadratic" else (term[1],):
                if referred not in linear:
                    raise ValueError(f"the term {':'.join(term)} refers to {referred}, which is not in factors ({', '.join(linear)})")
        features = nonlinear_features(linear, terms, investable)
        result = fama_macbeth_forecast(features, forward, 1, self.window, self.min_obs, None, min_assets=max(len(features) + 6, 12))
        return _to_daily(result.forecast, data)
