"""Forecast-driven, risk-aware portfolio allocation for the machine-learning workspace.

A model's output is a forecast, not a portfolio. Rank-based position rules (``signals.py``) turn a forecast into weights that depend only on the ORDER of the scores, so with two stocks they are
always the same two numbers (long the better, short the worse, or 50/50 when both are held) whatever the forecasts, the risks and the correlation are. This module is the other path: the forecasts become
per-asset expected returns with an uncertainty, a covariance is estimated as of each decision date, and one of nine allocators turns them into weights under one set of constraints. It is selected with
``signal.method = "portfolio_allocation"``; no existing method, default or saved experiment changes (the configuration hash of an experiment that does not use it is unchanged).

Pipeline at every decision date ``t`` (nothing after ``t`` is read; the backtest engine applies the execution lag, drift, costs and the ledger exactly as for every other rule):

1. **Forecast to expected return** (``ForecastModel``). Return forecasts are read as simple returns over the dataset's horizon. A price forecast becomes ``F / P_t - 1``. A directional probability becomes
   ``p * E[r | up] + (1 - p) * E[r | down]`` with the two conditional means and the probability's own reliability estimated from outcomes already known at ``t``; a probability is never read as a return. A
   forecast whose horizon, unit or currency differs from the portfolio's is refused, not rescaled.
2. **Calibration and uncertainty, from the past only.** The forecasts made at dates whose outcome was complete by ``t`` are regressed on what followed: ``realised = b * forecast``. ``b`` is clipped to
   ``[0, 1]`` (a forecast is never trusted beyond its own size, and a forecast that has not predicted gets none), the standard error of ``b`` and the out-of-sample forecast-error standard deviation are
   kept. With too little evidence the calibrated forecast is zero and the allocation is the risk-based one, labelled ``no_edge``.
3. **Risk** (``risk_models.fit_asof``): sample, EWMA, Ledoit-Wolf, PCA-denoised or factor covariance of the returns up to ``t``, scaled to the holding window. One asset has its variance only.
4. **Constraints** (``Limits``) are checked for feasibility before anything is solved and a problem that cannot be satisfied is reported with the reason; it is never relaxed silently.
5. **Allocator** (``ALLOCATORS``): equal weight, inverse volatility, equal risk contribution, mean-variance, Black-Litterman, hierarchical risk parity, CVaR, robust mean-variance, fractional Kelly. The
   optimisers reuse ``portfolio.*`` and ``ml.allocation.solve``; nothing is re-implemented except the CVaR linear programme, which needs the same constraints as the others.
6. **Explanation.** Every weight is split exactly into an equal-weight base, a risk effect (the allocator with a zero forecast minus the base) and a forecast effect (the allocator minus the risk-only
   book), with the constraints that bind. ``weights = equal + risk + forecast`` holds to rounding.

References: Markowitz (1952); Black and Litterman (1992); Ledoit and Wolf (2004); DeMiguel, Garlappi and Uppal (2009); Lopez de Prado (2016); Rockafellar and Uryasev (2000); Gu, Kelly and Xiu (2020).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from . import allocation as alloc
from . import alloc_adapters as ad
from . import alloc_calibration as cal
from . import alloc_fx as fxm
from . import alloc_horizon as hz
from . import alloc_stability as stab
from . import risk_models as rm

VERSION = "portfolio-allocation-v1"
CORE_ALLOCATORS = ("equal_weight", "inverse_vol", "erc", "mean_variance", "black_litterman", "hrp", "cvar", "robust_mv", "kelly")
ALLOCATORS = CORE_ALLOCATORS + tuple(ad.ADAPTERS)          # the adapters (alloc_adapters) are listed after the original nine
FORECAST_KINDS = ("return", "price", "probability", "distribution")
FAILURE_POLICIES = ("risk_only", "hold", "equal_weight", "cash")
LIFECYCLES = ("single_book", "staggered_tranches")
USES_FORECAST = {"mean_variance", "black_litterman", "cvar", "robust_mv", "kelly"} | ad.FORECAST_ADAPTERS
LONG_ONLY_ONLY = {"equal_weight", "inverse_vol", "erc", "hrp"} | set(ad.ADAPTERS)
TOL = 1e-8


class AllocationError(ValueError):
    """A request the allocation cannot honour, with what to change."""


class InfeasibleConstraints(AllocationError):
    """The constraints cannot all hold at once."""


# ------------------------------------------------------------------------------------------------------------ documentation of every allocator
DOCS = {
    "equal_weight": {"label": "Equal weight", "objective": "w_i = 1/N (within the limits)", "assumptions": "no forecast or risk estimate is informative (DeMiguel, Garlappi and Uppal 2009 find it hard to beat out of sample)",
                     "needs": "nothing", "failure": "ignores risk and forecasts altogether: a very volatile name gets the same weight as a calm one", "uses_forecast": False, "long_short": False, "min_assets": 1},
    "inverse_vol": {"label": "Inverse volatility", "objective": "w_i proportional to 1/sigma_i", "assumptions": "volatility estimates are informative, correlations and forecasts are not",
                    "needs": "a volatility per asset", "failure": "ignores correlation: two near-identical stocks are both fully weighted", "uses_forecast": False, "long_short": False, "min_assets": 1},
    "erc": {"label": "Equal risk contribution", "objective": "each asset contributes the same share of portfolio variance (or the given risk budget)", "assumptions": "a covariance matrix is estimable; no forecast is used",
            "needs": "a positive-definite covariance", "failure": "needs long-only weights; the solution is unstable when the covariance is near singular", "uses_forecast": False, "long_short": False, "min_assets": 1},
    "mean_variance": {"label": "Mean-variance (Markowitz 1952)", "objective": "maximise mu'w - (lambda/2) w'Sigma w - trading cost", "assumptions": "the calibrated forecast is the conditional mean; Sigma is the covariance of the holding window",
                      "needs": "calibrated expected returns and a covariance", "failure": "estimation error in mu is amplified: with a weak forecast the weights jump (the calibration shrinks the forecast toward zero for that reason)", "uses_forecast": True,
                      "long_short": True, "min_assets": 1},
    "black_litterman": {"label": "Black-Litterman (1992)", "objective": "mean-variance on the posterior of an equilibrium prior and the model's forecasts as uncertain absolute views", "assumptions": "the prior is the return that makes equal weights optimal; each view's variance is the model's past out-of-sample forecast error variance divided by the number of independent forecasts",
                        "needs": "forecasts with out-of-sample errors; a covariance", "failure": "with no measured view uncertainty the posterior is the prior (equal weights); the prior is an assumption, not data", "uses_forecast": True, "long_short": True, "min_assets": 1},
    "hrp": {"label": "Hierarchical risk parity (Lopez de Prado 2016)", "objective": "recursive bisection of a correlation tree, inverse-variance within clusters", "assumptions": "a meaningful correlation structure between at least three assets",
            "needs": "at least three assets and a covariance", "failure": "with two assets it is inverse variance and with one it is trivial, so it is refused for fewer than three; long-only", "uses_forecast": False, "long_short": False, "min_assets": 3},
    "cvar": {"label": "CVaR (Rockafellar and Uryasev 2000)", "objective": "maximise mu'w - kappa CVaR_alpha(w) over the trailing window's scenarios of window returns", "assumptions": "the past scenarios, de-meaned, describe the tail; the forecast supplies the mean",
             "needs": "at least 60 scenarios and a forecast (or zero)", "failure": "a short window gives a noisy tail; overlapping scenarios are not independent", "uses_forecast": True, "long_short": True, "min_assets": 1},
    "robust_mv": {"label": "Robust mean-variance", "objective": "mean-variance with each forecast reduced by kappa times its parameter uncertainty (box uncertainty set)", "assumptions": "the forecast's scale is known only to within the standard error of the calibration slope",
                  "needs": "calibration with a standard error", "failure": "conservative: a forecast that is not clearly significant is shrunk to nothing and the book becomes risk-based", "uses_forecast": True, "long_short": True, "min_assets": 1},
    "kelly": {"label": "Fractional Kelly", "objective": "mean-variance with risk aversion 1/f (f = Kelly fraction): the growth-optimal book scaled by f", "assumptions": "the forecast is calibrated (slope significantly positive over enough outcomes); returns are not strongly serially dependent",
              "needs": "a calibration t-statistic above the threshold and enough resolved forecasts", "failure": "full Kelly is aggressive and very sensitive to an overstated edge: it is used only where the calibration justifies it and otherwise reported as not justified (risk-only book)", "uses_forecast": True, "long_short": True, "min_assets": 1},
}
DOCS.update(ad.docs())                                                           # the adapters declare their own documentation (alloc_adapters.ADAPTERS)


# ------------------------------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class Limits:
    long_only: bool = True
    cash_allowed: bool = False
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_sector: float | None = None
    gross_max: float = 1.0
    net_min: float | None = None
    net_max: float | None = None
    turnover_max: float | None = None           # one-way per rebalance: sum |w - w_prev| / 2
    vol_target: float | None = None             # annualised
    max_leverage: float = 1.0
    risk_budget: tuple | None = None            # ((name, share), ...) for the equal-risk-contribution allocator


@dataclass(frozen=True)
class AllocConfig:
    allocator: str = "mean_variance"
    forecast_kind: str = "return"
    forecast_horizon: int | None = None         # declared horizon of the forecast, in trading days; must equal the dataset's
    forecast_currency: str | None = None        # declared currency of the forecast; must equal the portfolio's
    base_currency: str | None = None
    calibrate: str = "slope"                    # slope | none
    calib_window: int = 504
    calib_min_obs: int = 60
    calib_min_t: float = 1.0
    cov_method: str = "lw_constant_correlation"
    cov_window: int = 252
    cov_halflife: float = 60.0
    cov_min_obs: int = 60
    risk_aversion: float = 5.0
    tau: float = 0.05
    kelly_fraction: float = 0.5
    kelly_min_t: float = 2.0
    cvar_alpha: float = 0.95
    robust_kappa: float = 1.0
    opt_every: int | None = None
    cost_bps: float = 5.0
    turnover_penalty: float = 0.0
    on_failure: str = "risk_only"
    limits: Limits = field(default_factory=Limits)
    # position lifecycle (see alloc_horizon): None / defaults reproduce the one-window behaviour recorded before these settings existed
    holding_days: int | None = None            # intended holding window L; None: the forecast horizon
    lifecycle: str = "single_book"             # single_book: one book re-optimised at each decision | staggered_tranches: overlapping tranches, each held L sessions
    cov_horizon: str = "linear"                # linear | variance_ratio | multiperiod
    forecast_source: str = "scaled"            # scaled: the h-day forecast is scaled to the holding window | direct: the model's target IS the holding window (no scaling; refused if they differ)
    edge_model: str = "linear"                 # linear | compound: how a forecast is scaled to another window
    edge_beyond_horizon: str = "zero"          # zero | persist: whether edge is credited for days beyond the forecast horizon
    # forecast calibration beyond the pooled baseline (see alloc_calibration)
    calib_method: str = "pooled"               # pooled | asset | partial_pooling | sector_pooling | regularized
    calib_inference: str = "sqrt_horizon"      # sqrt_horizon (legacy) | hac | block_bootstrap
    calib_asset_min_eff: float = 30.0          # independent outcome windows an asset needs before it gets a slope of its own
    calib_prior_sd: float = 0.5                # prior standard deviation of the slope around zero for the regularised method
    # keeping a weak forecast from becoming a concentrated book (see alloc_stability); the defaults leave the allocator's own book untouched
    baseline: str = "none"                     # none | equal_weight | inverse_vol | erc: the risk-based book the ML book is blended with
    blend: str = "none"                        # none | fixed | confidence | validation: how alpha in (1 - alpha) w_baseline + alpha w_ml is chosen
    blend_alpha: float = 0.5
    blend_max: float = 1.0
    blend_window: int = 24
    blend_min_obs: int = 12
    uncertainty_shrink: float = 0.0            # forecasts use the slope minus this many standard errors of the slope
    min_n_eff: float = 0.0                     # independent outcome windows the calibration must rest on before the forecast is used at all
    min_confidence: float = 0.0                # an asset whose slope is positive with less than this probability gets no forecast
    max_weight_change: float | None = None     # no weight moves more than this in one decision (not applied to the first book)
    sensitivity: bool = False                  # attach forecast, risk-aversion and cost sensitivities of the last decision to the record
    preset: str | None = None                  # a documented set of the above (alloc_stability.PRESETS); explicit settings win

    def content(self) -> dict:
        d = asdict(self)
        d["schema"] = VERSION
        for k, v in _NEW_DEFAULTS.items():                                       # settings added after the first release appear only when they are used, so that an existing record keeps its hash
            if d.get(k) == v:
                d.pop(k, None)
        return d

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.content(), sort_keys=True, default=str).encode()).hexdigest()[:16]


_NEW_DEFAULTS = {"holding_days": None, "lifecycle": "single_book", "cov_horizon": "linear", "forecast_source": "scaled", "edge_model": "linear", "edge_beyond_horizon": "zero", "calib_method": "pooled", "calib_inference": "sqrt_horizon", "calib_asset_min_eff": 30.0, "calib_prior_sd": 0.5,
                 "baseline": "none", "blend": "none", "blend_alpha": 0.5, "blend_max": 1.0, "blend_window": 24, "blend_min_obs": 12, "uncertainty_shrink": 0.0, "min_n_eff": 0.0, "min_confidence": 0.0,
                 "max_weight_change": None, "sensitivity": False, "preset": None}
_LIMIT_KEYS = set(Limits.__dataclass_fields__)
_CFG_KEYS = set(AllocConfig.__dataclass_fields__) - {"limits"}


def config_from_signal(sig: dict) -> AllocConfig:
    """The ``signal.allocation`` block as a validated :class:`AllocConfig`. Unknown keys are refused (a typo must not become a default)."""
    raw = dict(sig.get("allocation") or {})
    lim_raw = dict(raw.pop("limits", {}) or {})
    if raw.get("preset") is not None:                                           # a preset fills in what the request did not set; the request wins
        if raw["preset"] not in stab.PRESETS:
            raise AllocationError(f"unknown preset '{raw['preset']}'; the presets are {sorted(stab.PRESETS)}")
        pr = stab.PRESETS[raw["preset"]]
        raw = {**pr["allocation"], **raw}
        lim_raw = {**pr.get("limits", {}), **lim_raw}
    unknown = (set(raw) - _CFG_KEYS) | (set(lim_raw) - _LIMIT_KEYS)
    if unknown:
        raise AllocationError(f"unknown allocation setting(s) {sorted(unknown)}; the settings are {sorted(_CFG_KEYS | _LIMIT_KEYS)}")
    if lim_raw.get("risk_budget") is not None and isinstance(lim_raw["risk_budget"], dict):
        lim_raw["risk_budget"] = tuple(sorted((str(k), float(v)) for k, v in lim_raw["risk_budget"].items()))
    limits = Limits(**lim_raw)
    cfg = AllocConfig(**raw, limits=limits)
    check_config(cfg)
    return cfg


def check_config(c: AllocConfig) -> None:
    if c.allocator not in ALLOCATORS:
        raise AllocationError(f"unknown allocator '{c.allocator}'; the allocators are {list(ALLOCATORS)}")
    if c.forecast_kind not in FORECAST_KINDS:
        raise AllocationError(f"forecast_kind must be one of {list(FORECAST_KINDS)}")
    if c.calibrate not in ("slope", "none"):
        raise AllocationError("calibrate must be 'slope' (past outcomes) or 'none' (the raw forecast, labelled uncalibrated)")
    if c.on_failure not in FAILURE_POLICIES:
        raise AllocationError(f"on_failure must be one of {list(FAILURE_POLICIES)}")
    if c.cov_method not in rm.METHODS:
        raise AllocationError(f"cov_method must be one of {list(rm.METHODS)}")
    if not (c.risk_aversion >= 0 and 0 < c.kelly_fraction <= 1 and 0 < c.cvar_alpha < 1 and c.robust_kappa >= 0 and c.cost_bps >= 0 and c.turnover_penalty >= 0 and c.tau > 0):
        raise AllocationError("risk_aversion, cost_bps, turnover_penalty and robust_kappa are not negative; 0 < kelly_fraction <= 1; 0 < cvar_alpha < 1; tau > 0")
    if c.calib_min_obs < 20 or c.calib_window < c.calib_min_obs or c.cov_window < c.cov_min_obs or c.cov_min_obs < 20:
        raise AllocationError("calib_min_obs and cov_min_obs are at least 20 and no larger than their windows")
    if c.opt_every is not None and c.opt_every < 1:
        raise AllocationError("opt_every is at least 1 day")
    if c.holding_days is not None and not 1 <= int(c.holding_days) <= 252:
        raise AllocationError("holding_days is between 1 and 252 sessions")
    if c.baseline not in stab.BASELINES or c.blend not in stab.BLENDS:
        raise AllocationError(f"baseline must be one of {list(stab.BASELINES)} and blend one of {list(stab.BLENDS)}")
    if c.blend != "none" and c.baseline == "none":
        raise AllocationError("blending needs a baseline: choose equal_weight, inverse_vol or erc")
    if c.blend != "none" and c.allocator not in USES_FORECAST:
        raise AllocationError(f"{DOCS[c.allocator]['label']} reads no forecast, so there is nothing to blend with a baseline; choose a forecast-driven allocator or no blend")
    if not (0 <= c.blend_alpha <= 1 and 0 <= c.blend_max <= 1 and c.blend_window >= 5 and c.blend_min_obs >= 3 and c.uncertainty_shrink >= 0 and c.min_n_eff >= 0 and 0 <= c.min_confidence <= 1):
        raise AllocationError("blend_alpha and blend_max are in [0, 1]; blend_window at least 5 and blend_min_obs at least 3; uncertainty_shrink, min_n_eff non-negative; min_confidence in [0, 1]")
    if c.max_weight_change is not None and not 0 < c.max_weight_change <= 1:
        raise AllocationError("max_weight_change is a share of capital in (0, 1]")
    if c.calib_method not in cal.METHODS:
        raise AllocationError(f"calib_method must be one of {list(cal.METHODS)}")
    if c.calib_inference not in cal.INFERENCE:
        raise AllocationError(f"calib_inference must be one of {list(cal.INFERENCE)}")
    if not (c.calib_asset_min_eff >= 5 and c.calib_prior_sd > 0):
        raise AllocationError("calib_asset_min_eff is at least 5 independent windows and calib_prior_sd is positive")
    if c.lifecycle not in LIFECYCLES:
        raise AllocationError(f"lifecycle must be one of {list(LIFECYCLES)}")
    if c.forecast_source not in hz.FORECAST_SOURCES or c.edge_model not in hz.EDGE_MODELS or c.edge_beyond_horizon not in hz.BEYOND:
        raise AllocationError(f"forecast_source must be one of {list(hz.FORECAST_SOURCES)}, edge_model one of {list(hz.EDGE_MODELS)} and edge_beyond_horizon one of {list(hz.BEYOND)}")
    if c.cov_horizon not in hz.COV_HORIZON_METHODS:
        raise AllocationError(f"cov_horizon must be one of {list(hz.COV_HORIZON_METHODS)}")
    lim = c.limits
    if c.allocator in LONG_ONLY_ONLY and not lim.long_only:
        raise AllocationError(f"{DOCS[c.allocator]['label']} builds long-only books; choose mean-variance, Black-Litterman, CVaR, robust mean-variance or Kelly for a long-short allocation")
    if lim.risk_budget is not None and c.allocator != "erc":
        raise AllocationError("a risk budget applies to the equal-risk-contribution allocator only")


# ------------------------------------------------------------------------------------------------------------ constraints: feasibility and projection
def feasibility(lim: Limits, names: list[str], groups: dict | None = None, allocator: str | None = None) -> list[str]:
    """What is wrong with the constraints for this universe, as readable problems; an empty list means a feasible book exists. Checked before any optimisation."""
    n = len(names)
    out: list[str] = []
    if n == 0:
        return ["no asset is investable"]
    if lim.min_weight < 0 or lim.max_weight <= 0 or lim.min_weight > lim.max_weight:
        out.append(f"the weight limits [{lim.min_weight}, {lim.max_weight}] are not a valid interval")
    if lim.long_only:
        full = not lim.cash_allowed
        lo, hi = n * lim.min_weight, n * lim.max_weight
        if full and hi < 1.0 - 1e-9:
            out.append(f"a maximum weight of {lim.max_weight:g} on {n} asset(s) cannot be fully invested (the most that can be held is {hi:.2f}); allow cash or raise the maximum")
        if lo > min(lim.gross_max, 1.0) + 1e-9:
            out.append(f"{n} asset(s) at the minimum weight {lim.min_weight:g} need {lo:.2f} of capital, more than the {min(lim.gross_max, 1.0):g} available")
        if full and lim.gross_max < 1.0 - 1e-9:
            out.append("a fully invested long-only book needs a gross limit of at least 1; allow cash or raise the gross limit")
    else:
        if lim.gross_max <= 0:
            out.append("the gross limit is not positive")
        if lim.net_min is not None and lim.net_max is not None and lim.net_min > lim.net_max + 1e-12:
            out.append(f"the net exposure minimum {lim.net_min:g} is above its maximum {lim.net_max:g}")
        for tag, v in (("minimum", lim.net_min), ("maximum", lim.net_max)):
            if v is not None and abs(v) > lim.gross_max + 1e-12:
                out.append(f"the net exposure {tag} {v:g} exceeds the gross limit {lim.gross_max:g}")
        if lim.min_weight > 0:
            out.append("a minimum weight above zero is a long-only constraint; set it to 0 for a long-short book")
    if lim.max_sector is not None:
        if not 0 < lim.max_sector <= 1.0:
            out.append("the sector limit is a share of capital in (0, 1]")
        elif groups is None:
            out.append("a sector cap needs a sector map for the universe and none is available: it is refused rather than ignored")
        elif groups is not None and lim.long_only and not lim.cash_allowed:
            sectors = {groups.get(a, "_none") for a in names}
            if len(sectors) * lim.max_sector < 1.0 - 1e-9:
                out.append(f"{len(sectors)} sector(s) capped at {lim.max_sector:g} cannot hold a fully invested book (the most is {len(sectors) * lim.max_sector:.2f}); allow cash or raise the cap")
        elif groups is not None and lim.long_only:
            if lim.max_sector * len({groups.get(a, "_none") for a in names}) < lim.min_weight * n - 1e-12:
                out.append("the sector cap is below the minimum weights it must hold")
    if lim.turnover_max is not None and lim.turnover_max < 0:
        out.append("the turnover limit is not negative")
    if lim.vol_target is not None:
        if lim.vol_target <= 0:
            out.append("the volatility target is positive")
        if lim.long_only and not lim.cash_allowed and lim.max_leverage <= 1.0 + 1e-12:
            out.append("a volatility target needs room to scale: allow cash (to reduce risk) or a maximum leverage above 1 (to add it); a fully invested, unlevered long-only book has a fixed gross exposure")
    if lim.max_leverage < min(lim.gross_max, 1.0) - 1e-12:
        out.append("the maximum leverage is below the gross limit")
    if lim.risk_budget is not None:
        budget = dict(lim.risk_budget)
        missing = [a for a in names if a not in budget]
        if missing or any(v < 0 for v in budget.values()) or sum(budget.values()) <= 0:
            out.append("a risk budget names every asset with non-negative shares that sum to a positive number" + (f" (missing: {', '.join(missing[:5])})" if missing else ""))
    if allocator in LONG_ONLY_ONLY and not lim.long_only:
        out.append(f"{allocator} builds long-only books")
    return out


def require_feasible(lim: Limits, names: list[str], groups: dict | None = None, allocator: str | None = None) -> None:
    problems = feasibility(lim, names, groups, allocator)
    if problems:
        raise InfeasibleConstraints("; ".join(problems))


def _bounds(lim: Limits, n: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(n, lim.min_weight if lim.long_only else -lim.max_weight)
    hi = np.full(n, lim.max_weight)
    return lo, hi


def project(x: pd.Series, lim: Limits, groups: dict | None = None) -> pd.Series:
    """The nearest (Euclidean) book to ``x`` that satisfies the box, the sum (fully invested or cash allowed), the gross and net limits and the sector caps. Exact for the box and the sum; the other limits are
    enforced as constraints of the same projection. Raises :class:`InfeasibleConstraints` when no such book exists."""
    names = list(x.index)
    n = len(names)
    require_feasible(lim, names, groups)
    lo, hi = _bounds(lim, n)
    v = x.to_numpy(dtype=float)
    full = lim.long_only and not lim.cash_allowed
    cons = []
    if lim.long_only:
        s_hi = min(lim.gross_max, 1.0) if lim.cash_allowed else 1.0
        s_lo = 1.0 if full else 0.0
        cons.append({"type": "ineq", "fun": lambda w: s_hi - w.sum()})
        cons.append({"type": "ineq", "fun": lambda w: w.sum() - s_lo})
    else:
        cons.append({"type": "ineq", "fun": lambda w: lim.gross_max - np.abs(w).sum()})
        if lim.net_min is not None:
            cons.append({"type": "ineq", "fun": lambda w: w.sum() - lim.net_min})
        if lim.net_max is not None:
            cons.append({"type": "ineq", "fun": lambda w: lim.net_max - w.sum()})
    if lim.max_sector is not None and groups is not None:
        arr = np.array([groups.get(a, "_none") for a in names])
        for g in sorted(set(arr.tolist())):
            m = (arr == g)
            cons.append({"type": "ineq", "fun": (lambda w, m=m: lim.max_sector - np.abs(w[m]).sum())})
    if full and not cons[2:] and lim.max_sector is None:
        w = _box_sum_projection(v, lo, hi, 1.0)                                       # the common case has an exact answer
        return pd.Series(w, index=names)
    start = np.clip(v, lo, hi)
    res = minimize(lambda w: 0.5 * float(((w - v) ** 2).sum()), start, jac=lambda w: w - v, bounds=list(zip(lo, hi)), constraints=cons, method="SLSQP", options={"maxiter": 300, "ftol": 1e-14})
    w = np.clip(res.x, lo, hi)
    pd_ = pd.Series(w, index=names)
    problems = violations(pd_, lim, groups)
    if problems:
        raise InfeasibleConstraints("the projection could not satisfy: " + "; ".join(problems))
    return pd_


def _box_sum_projection(v: np.ndarray, lo: np.ndarray, hi: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection onto {lo <= w <= hi, sum w = total} by bisection on the multiplier."""
    a, b = float((v - hi).min()) - 1.0, float((v - lo).max()) + 1.0
    for _ in range(200):
        mid = 0.5 * (a + b)
        s = float(np.clip(v - mid, lo, hi).sum())
        if s > total:
            a = mid
        else:
            b = mid
    return np.clip(v - 0.5 * (a + b), lo, hi)


def violations(w: pd.Series, lim: Limits, groups: dict | None = None, tol: float = 1e-6) -> list[str]:
    """Every limit the book breaks (empty when it satisfies them all): the final check that makes a silent relaxation impossible."""
    out = []
    a = w.to_numpy(dtype=float)
    if (a > lim.max_weight + tol).any():
        out.append(f"a weight above the maximum {lim.max_weight:g}")
    if lim.long_only:
        if (a < lim.min_weight - tol).any():
            out.append(f"a weight below the minimum {lim.min_weight:g}")
        if not lim.cash_allowed and abs(a.sum() - 1.0) > tol and lim.vol_target is None:
            out.append(f"not fully invested (the weights sum to {a.sum():.6f})")
        if a.sum() > max(min(lim.gross_max, 1.0), lim.max_leverage if lim.vol_target is not None else 0.0) + tol:
            out.append(f"the weights sum to {a.sum():.4f}, above the limit")
    else:
        if (a < -lim.max_weight - tol).any():
            out.append(f"a short position beyond -{lim.max_weight:g}")
        if np.abs(a).sum() > lim.gross_max + tol:
            out.append(f"gross exposure {np.abs(a).sum():.4f} above {lim.gross_max:g}")
        if lim.net_min is not None and a.sum() < lim.net_min - tol:
            out.append(f"net exposure {a.sum():.4f} below {lim.net_min:g}")
        if lim.net_max is not None and a.sum() > lim.net_max + tol:
            out.append(f"net exposure {a.sum():.4f} above {lim.net_max:g}")
    if lim.max_sector is not None and groups is not None:
        sect = pd.Series(np.abs(a), index=w.index).groupby(lambda k: groups.get(k, "_none")).sum()
        if (sect > lim.max_sector + tol).any():
            out.append(f"sector {sect.idxmax()} holds {sect.max():.4f}, above {lim.max_sector:g}")
    return out


def binding(w: pd.Series, lim: Limits, groups: dict | None = None, tol: float = 1e-6) -> list[str]:
    """The constraints the book sits on (active limits): what the weights are being held back by."""
    a = w.to_numpy(dtype=float)
    out = []
    for name, x in zip(w.index, a):
        if x >= lim.max_weight - tol:
            out.append(f"{name} at the maximum weight")
        elif lim.long_only and lim.min_weight > 0 and x <= lim.min_weight + tol:
            out.append(f"{name} at the minimum weight")
        elif not lim.long_only and x <= -lim.max_weight + tol:
            out.append(f"{name} at the maximum short")
    if lim.max_sector is not None and groups is not None:
        sect = pd.Series(np.abs(a), index=w.index).groupby(lambda k: groups.get(k, "_none")).sum()
        out += [f"sector {g} at its cap" for g, v in sect.items() if v >= lim.max_sector - tol]
    if not lim.long_only and np.abs(a).sum() >= lim.gross_max - tol:
        out.append("gross exposure at its limit")
    return out


def limit_turnover(w: pd.Series, prev: pd.Series, lim: Limits) -> tuple[pd.Series, bool]:
    """If the one-way turnover to ``w`` exceeds the limit, move only as far toward it as the limit allows. The result is a convex combination of two books that both satisfy the other constraints, so it satisfies them too."""
    if lim.turnover_max is None:
        return w, False
    d = (w - prev).abs().sum() / 2.0
    if d <= lim.turnover_max + 1e-12:
        return w, False
    a = lim.turnover_max / d
    return prev + a * (w - prev), True


def scale_to_vol(w: pd.Series, cov_window: pd.DataFrame, horizon_days: int, lim: Limits) -> tuple[pd.Series, float | None, float]:
    """Scale the book so its forecast annualised volatility is the target, never above the maximum leverage and (when cash is allowed) never beyond the base book's own gross when that would lever it. Returns the book, the
    volatility before scaling and the factor used."""
    if lim.vol_target is None:
        return w, None, 1.0
    a = w.to_numpy(dtype=float)
    var_w = float(a @ cov_window.reindex(index=w.index, columns=w.index).to_numpy(dtype=float) @ a)
    ann = math.sqrt(max(var_w, 0.0) * 252.0 / max(horizon_days, 1))
    if ann <= 1e-12:
        return w, ann, 1.0
    f = lim.vol_target / ann
    gross = float(np.abs(a).sum())
    f = min(f, lim.max_leverage / gross) if gross > 0 else f
    if lim.long_only and not lim.cash_allowed:
        f = max(f, 1.0) if lim.max_leverage > 1.0 else 1.0
    return w * f, ann, float(f)


# ------------------------------------------------------------------------------------------------------------ forecasts
@dataclass
class Estimate:
    mu: pd.Series                      # calibrated expected simple return over the holding window
    mu_raw: pd.Series
    sd: pd.Series                      # out-of-sample forecast error standard deviation of the window return
    slope: float | None
    slope_se: float | None
    t_stat: float | None
    n_obs: int
    n_eff: float
    status: str                        # calibrated | no_edge | uncalibrated | insufficient_history
    note: str = ""
    slopes: pd.Series | None = None    # the slope applied to each asset (asset-specific and partially pooled calibration); None: one pooled slope
    confidence: pd.Series | None = None  # per asset: probability the true slope is positive given the evidence (0 where there is no edge)
    method: str = "pooled"
    inference: str = "sqrt_horizon"
    fallback: str = ""
    tau: float | None = None
    n_dates: int = 0

    def info(self) -> dict:
        out = {"status": self.status, "slope": self.slope, "slope_se": self.slope_se, "t_stat": self.t_stat, "n_obs": int(self.n_obs), "n_eff": float(self.n_eff), "note": self.note}
        if self.method != "pooled" or self.inference != "sqrt_horizon":
            out.update(method=self.method, inference=self.inference, fallback=self.fallback, tau=self.tau, n_dates=int(self.n_dates),
                       slopes=None if self.slopes is None else {str(k): float(v) for k, v in self.slopes.items()},
                       confidence=None if self.confidence is None else {str(k): float(v) for k, v in self.confidence.items()})
        return out


def realised_window_returns(returns: pd.DataFrame, horizon: int, lag: int) -> pd.DataFrame:
    """The return each asset earned over the window that a decision made at row ``s`` is exposed to: the compounded daily returns of rows ``s + lag + 1 .. s + lag + horizon``. Row ``s`` carries a value only once
    ``s + lag + horizon`` rows exist; whoever reads it at row ``i`` must use ``s <= i - lag - horizon``."""
    gross = np.log1p(returns.where(returns > -1.0))
    csum = gross.cumsum()
    out = np.expm1(csum.shift(-(lag + horizon)) - csum.shift(-lag))
    return out


def check_forecast_contract(cfg: AllocConfig, dataset_horizon: int, base_currency: str | None = None, fx: "fxm.FxContext | None" = None) -> None:
    """Refuse a forecast whose horizon or currency is not the portfolio's, and say which. With an FX context the forecast may be in each asset's own (local) currency or the base: it is converted at the decision date
    under the stated FX assumption (alloc_fx); any other currency, or a foreign asset with no exchange-rate series, is refused."""
    if cfg.forecast_horizon is not None and int(cfg.forecast_horizon) != int(dataset_horizon):
        raise AllocationError(f"the forecast is for {int(cfg.forecast_horizon)} trading days but the portfolio's holding window is {int(dataset_horizon)}: forecasts of another horizon are not rescaled; re-train the model on the matching target")
    if fx is not None:
        if cfg.forecast_currency is not None and str(cfg.forecast_currency).upper() not in ("LOCAL", fx.table.base):
            raise AllocationError(f"the forecast is declared in {cfg.forecast_currency}: with an FX context a forecast is in each asset's own currency ('local') or in the base currency {fx.table.base}")
        if cfg.base_currency is not None and str(cfg.base_currency).upper() != fx.table.base:
            raise AllocationError(f"the allocation's base currency is {cfg.base_currency} but the exchange-rate table is quoted against {fx.table.base}")
        return
    base = cfg.base_currency or base_currency
    if cfg.forecast_currency is not None and base is not None and str(cfg.forecast_currency).upper() != str(base).upper():
        raise AllocationError(f"the forecast is in {cfg.forecast_currency} and the portfolio in {base}: convert the forecast with an exchange-rate forecast first; the allocation does not mix currencies")


def to_expected_return(score: pd.DataFrame, kind: str, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-asset expected simple return from a model output of the given kind. A return is itself; a price is ``F / P - 1`` at the decision date; a probability and a distribution need their own estimators and are
    handled in :class:`ForecastModel` (a probability is never read as a return)."""
    if kind == "return" or kind == "distribution":
        return score
    if kind == "price":
        if prices is None:
            raise AllocationError("a price forecast needs the decision-date prices to become a return")
        p = prices.reindex(index=score.index, columns=score.columns)
        return score / p.where(p > 0) - 1.0
    if kind == "probability":
        raise AllocationError("a directional probability is not an expected return: it needs the conditional payoff of an up and a down move, estimated from past outcomes (ForecastModel does that)")
    raise AllocationError(f"unknown forecast kind '{kind}'")


class ForecastModel:
    """Calibrated expected returns and their uncertainty, as of any row, from outcomes known by then."""

    def __init__(self, score: pd.DataFrame, returns: pd.DataFrame, cfg: AllocConfig, horizon: int, lag: int, prices: pd.DataFrame | None = None, probability: pd.DataFrame | None = None,
                 predictive_sd: pd.DataFrame | None = None, groups: dict | None = None):
        self.cfg, self.h, self.lag = cfg, int(horizon), int(lag)
        self.groups = groups
        self.returns = returns.reindex(index=score.index, columns=score.columns)
        self.predictive_sd = predictive_sd
        self.kind = cfg.forecast_kind
        if self.kind == "probability":
            if probability is None:
                raise AllocationError("a probability forecast needs the model's predicted probabilities; this model does not provide them")
            self.p = probability.reindex(index=score.index, columns=score.columns)
            self.x = None
        else:
            self.x = to_expected_return(score, self.kind, prices)
            self.p = None
        self.real = realised_window_returns(self.returns, self.h, self.lag)

    def asof(self, i: int) -> Estimate:
        cfg = self.cfg
        cols = self.returns.columns
        last = i - self.lag - self.h                                                      # the latest forecast row whose outcome is complete by row i
        lo = max(0, last - cfg.calib_window + 1)
        if self.kind == "probability":
            return self._probability(i, lo, last)
        raw = self.x.iloc[i]
        if cfg.calibrate == "none":
            return Estimate(raw, raw, self._sd(i, lo, last, raw.index, None), None, None, None, 0, 0.0, "uncalibrated", "the raw forecast is used as it is: nothing was checked against outcomes")
        if last < lo or last < 0:
            return Estimate(raw * 0.0, raw, pd.Series(np.nan, index=cols), None, None, None, 0, 0.0, "insufficient_history", "no forecast has a known outcome yet")
        xs = self.x.iloc[lo:last + 1]
        ys = self.real.iloc[lo:last + 1]
        pair = pd.concat([xs.stack(future_stack=True), ys.stack(future_stack=True)], axis=1, keys=["x", "y"]).dropna()
        n = len(pair)
        if n < cfg.calib_min_obs:
            return Estimate(raw * 0.0, raw, pd.Series(np.nan, index=cols), None, None, None, n, n / self.h, "insufficient_history", f"{n} resolved forecasts; {cfg.calib_min_obs} are needed")
        if cfg.calib_method != "pooled" or cfg.calib_inference != "sqrt_horizon":
            return self._calibrated(i, lo, last, xs, ys, raw, n)
        x, y = pair["x"].to_numpy(), pair["y"].to_numpy()
        sxx = float((x ** 2).sum())
        if sxx <= 0:
            return Estimate(raw * 0.0, raw, self._sd(i, lo, last, raw.index, 0.0), 0.0, None, None, n, n / self.h, "no_edge", "the forecasts were all zero")
        b = float((x * y).sum() / sxx)
        resid = y - b * x
        dof = max(n - 1, 1)
        se = float(math.sqrt(resid.var(ddof=1) / sxx * self.h)) if n > 2 else float("nan")      # overlapping windows are dependent: the standard error is inflated by sqrt(horizon)
        t = b / se if se and np.isfinite(se) and se > 0 else float("nan")
        n_eff = n / self.h
        sd = self._sd(i, lo, last, raw.index, b)
        if not np.isfinite(t) or b <= 0 or t < cfg.calib_min_t:
            return Estimate(raw * 0.0, raw, sd, max(b, 0.0), se, t, n, n_eff, "no_edge", f"the forecast has not predicted: slope {b:.2f}, t {t:.2f} (below {cfg.calib_min_t:g})")
        bc = min(b, 1.0)
        return Estimate(raw * bc, raw, sd, bc, se, t, n, n_eff, "calibrated", f"calibrated by a slope of {bc:.2f} (t {t:.1f}, {n} outcomes)")

    def _calibrated(self, i: int, lo: int, last: int, xs: pd.DataFrame, ys: pd.DataFrame, raw: pd.Series, n: int) -> Estimate:
        """Calibration by the method and dependence-aware inference in the configuration (alloc_calibration), from the same resolved forecasts as the baseline."""
        cfg, cols = self.cfg, list(raw.index)
        c = cal.calibrate(xs, ys, cfg, self.h, self.groups)
        slopes = c.slopes.reindex(cols).fillna(0.0)
        sd = self._sd_vec(i, lo, last, cols, slopes)
        pooled_b = float(min(max(c.pooled_slope, 0.0), 1.0))
        if not (slopes > 0).any():
            return Estimate(raw * 0.0, raw, sd, pooled_b, c.pooled_se, c.pooled_t, n, c.n_eff, "no_edge", f"no demonstrated edge: pooled slope {c.pooled_slope:.2f}, t {c.pooled_t:.2f} (below {cfg.calib_min_t:g}); {c.note}",
                            slopes, c.confidence.reindex(cols).fillna(0.0), c.method, c.inference, c.fallback, c.tau, c.n_dates)
        note = c.note + (f"; {c.fallback}" if c.fallback else "")
        return Estimate(raw * slopes, raw, sd, pooled_b, c.pooled_se, c.pooled_t, n, c.n_eff, "calibrated", note, slopes, c.confidence.reindex(cols).fillna(0.0), c.method, c.inference, c.fallback, c.tau, c.n_dates)

    def _sd_vec(self, i: int, lo: int, last: int, cols, slopes: pd.Series) -> pd.Series:
        """Forecast-error standard deviation per asset when each asset has its own slope."""
        xs, ys = self.x.iloc[lo:last + 1], self.real.iloc[lo:last + 1]
        err = ys - xs * slopes.reindex(xs.columns).fillna(0.0)
        pooled = float(np.nanstd(err.to_numpy(), ddof=1)) if np.isfinite(err.to_numpy()).sum() > 2 else float("nan")
        per = err.std(ddof=1).where(err.count() >= self.cfg.calib_min_obs, pooled)
        sd = per.reindex(cols).fillna(pooled)
        if self.predictive_sd is not None:
            own = self.predictive_sd.iloc[i].reindex(cols)
            sd = own.where(own > 0, sd)
        return sd

    def _sd(self, i: int, lo: int, last: int, index, b: float | None) -> pd.Series:
        """Out-of-sample forecast-error standard deviation of the window return per asset (pooled when an asset has too few outcomes); the model's own predictive std replaces it where it has one."""
        cols = list(index)
        if last < lo or last < 0:
            return pd.Series(np.nan, index=cols)
        xs, ys = self.x.iloc[lo:last + 1], self.real.iloc[lo:last + 1]
        err = ys - (0.0 if b is None else b) * xs
        pooled = float(np.nanstd(err.to_numpy(), ddof=1)) if np.isfinite(err.to_numpy()).sum() > 2 else float("nan")
        per = err.std(ddof=1).where(err.count() >= self.cfg.calib_min_obs, pooled)
        sd = per.reindex(cols).fillna(pooled)
        if self.predictive_sd is not None:
            own = self.predictive_sd.iloc[i].reindex(cols)
            sd = own.where(own > 0, sd)
        return sd

    def _probability(self, i: int, lo: int, last: int) -> Estimate:
        cfg, cols = self.cfg, self.returns.columns
        p_now = self.p.iloc[i]
        if last < lo or last < 0:
            return Estimate(p_now * 0.0, p_now, pd.Series(np.nan, index=cols), None, None, None, 0, 0.0, "insufficient_history", "no forecast has a known outcome yet")
        ps = self.p.iloc[lo:last + 1].stack(future_stack=True)
        ys = self.real.iloc[lo:last + 1].stack(future_stack=True)
        pair = pd.concat([ps, ys], axis=1, keys=["p", "y"]).dropna()
        up, dn = pair[pair["y"] > 0], pair[pair["y"] <= 0]
        n = len(pair)
        if n < cfg.calib_min_obs or len(up) < 10 or len(dn) < 10:
            return Estimate(p_now * 0.0, p_now, pd.Series(np.nan, index=cols), None, None, None, n, n / self.h, "insufficient_history",
                            f"a probability becomes a return only through the average gain after an up move and loss after a down move; {len(up)} up and {len(dn)} down outcomes are known, 10 of each and {cfg.calib_min_obs} in all are needed")
        m_up, m_dn = float(up["y"].mean()), float(dn["y"].mean())
        z = (pair["y"] > 0).astype(float).to_numpy()
        pbar, zbar = float(pair["p"].mean()), float(z.mean())
        dp = pair["p"].to_numpy() - pbar
        sxx = float((dp ** 2).sum())
        slope = float((dp * (z - zbar)).sum() / sxx) if sxx > 0 else 0.0
        resid = (z - zbar) - slope * dp
        se = float(math.sqrt(resid.var(ddof=1) / sxx * self.h)) if sxx > 0 and n > 2 else float("nan")
        t = slope / se if se and np.isfinite(se) and se > 0 else float("nan")
        b = float(np.clip(slope, 0.0, 1.0)) if np.isfinite(t) and t >= cfg.calib_min_t else 0.0
        p_cal = (zbar + b * (p_now - pbar)).clip(0.0, 1.0)
        mu = p_cal * m_up + (1.0 - p_cal) * m_dn
        sd = pd.Series(float(math.sqrt(max(zbar * m_up ** 2 + (1 - zbar) * m_dn ** 2 - (zbar * m_up + (1 - zbar) * m_dn) ** 2, 0.0) + float(pair["y"].var(ddof=1)))), index=cols)
        status = "calibrated" if b > 0 else "no_edge"
        note = (f"the probability was calibrated by a reliability slope of {b:.2f} (t {t:.1f}); the payoff model is +{m_up:.2%} after an up move and {m_dn:.2%} after a down move, from {n} outcomes" if b > 0 else
                f"the probability has not predicted the direction (reliability slope t {t:.2f}); the expected return is the unconditional average of {zbar * m_up + (1 - zbar) * m_dn:.2%}")
        return Estimate(mu, p_now, sd, b, se, t, n, n / self.h, status, note)


# ------------------------------------------------------------------------------------------------------------ allocators
@dataclass
class Result:
    weights: pd.Series | None
    status: str                         # optimal | failed | infeasible | not_justified | unsuitable
    reason: str = ""
    detail: dict = field(default_factory=dict)


def _spec(cfg: AllocConfig, risk_aversion: float | None = None) -> alloc.AllocationSpec:
    lim = cfg.limits
    ra = cfg.risk_aversion if risk_aversion is None else risk_aversion
    if lim.long_only:
        net_min, net_max = (1.0, 1.0) if not lim.cash_allowed else (None, None)
        gross = 1.0 if not lim.cash_allowed else min(lim.gross_max, 1.0)
    else:
        net_min, net_max, gross = lim.net_min, lim.net_max, lim.gross_max
    return alloc.AllocationSpec(risk_aversion=float(ra), cost=cfg.cost_bps / 1e4, turnover_penalty=cfg.turnover_penalty, long_only=lim.long_only, gross_max=float(gross), net_min=net_min, net_max=net_max,
                                position_max=float(lim.max_weight), group_max=lim.max_sector)


def _solve(mu: pd.Series, cov: pd.DataFrame, w0: pd.Series, cfg: AllocConfig, groups: dict | None, risk_aversion: float | None = None, penalty: pd.Series | None = None) -> Result:
    sol = alloc.solve(mu, cov, w0, _spec(cfg, risk_aversion), groups=groups, holding_penalty=penalty)
    if sol.weights is None:
        return Result(None, sol.status, sol.reason)
    return Result(sol.weights, "optimal", "", {"components": sol.components, "solver": sol.solver})


def _cvar_lp(mu: pd.Series, scenarios: np.ndarray, w0: pd.Series, cfg: AllocConfig, groups: dict | None) -> Result:
    """Maximise ``mu'w - kappa * CVaR_alpha(w) - cost |w - w0|`` over the scenarios (Rockafellar and Uryasev 2000) as one linear programme under the shared limits."""
    lim = cfg.limits
    names = list(mu.index)
    n, K = len(names), scenarios.shape[0]
    a = cfg.cvar_alpha
    kappa = cfg.risk_aversion / 2.0
    c_cost = cfg.cost_bps / 1e4
    # variables: w+ (n), w- (n), zeta (1), u (K), t (n)  with t >= |w - w0|
    nv = 2 * n + 1 + K + n
    c = np.zeros(nv)
    c[:n] = -mu.to_numpy() + c_cost * 0.0
    c[n:2 * n] = mu.to_numpy()
    c[2 * n] = kappa
    c[2 * n + 1:2 * n + 1 + K] = kappa / ((1.0 - a) * K)
    c[2 * n + 1 + K:] = c_cost
    A, b = [], []
    for k in range(K):                                                            # u_k >= -s_k'w - zeta
        row = np.zeros(nv)
        row[:n] = -scenarios[k]
        row[n:2 * n] = scenarios[k]
        row[2 * n] = -1.0
        row[2 * n + 1 + k] = -1.0
        A.append(row)
        b.append(0.0)
    w0v = w0.reindex(names).fillna(0.0).to_numpy()
    for i in range(n):                                                            # t_i >= w_i - w0_i and t_i >= w0_i - w_i
        r1, r2 = np.zeros(nv), np.zeros(nv)
        r1[i], r1[n + i], r1[2 * n + 1 + K + i] = 1.0, -1.0, -1.0
        r2[i], r2[n + i], r2[2 * n + 1 + K + i] = -1.0, 1.0, -1.0
        A += [r1, r2]
        b += [w0v[i], -w0v[i]]
    long_only = lim.long_only
    Aeq, beq = [], []
    row = np.zeros(nv)
    row[:n], row[n:2 * n] = 1.0, -1.0
    if long_only:
        if not lim.cash_allowed:
            Aeq.append(row)
            beq.append(1.0)
        else:
            A.append(row)
            b.append(min(lim.gross_max, 1.0))
    else:
        g = np.zeros(nv)
        g[:2 * n] = 1.0
        A.append(g)
        b.append(lim.gross_max)
        if lim.net_min is not None:
            A.append(-row)
            b.append(-lim.net_min)
        if lim.net_max is not None:
            A.append(row)
            b.append(lim.net_max)
    if lim.max_sector is not None and groups is not None:
        arr = np.array([groups.get(x, "_none") for x in names])
        for gname in sorted(set(arr.tolist())):
            m = (arr == gname).astype(float)
            r = np.zeros(nv)
            r[:n], r[n:2 * n] = m, m
            A.append(r)
            b.append(lim.max_sector)
    bounds = [(lim.min_weight if long_only else 0.0, lim.max_weight)] * n + [(0.0, 0.0 if long_only else lim.max_weight)] * n + [(None, None)] + [(0.0, None)] * K + [(0.0, None)] * n
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=np.array(Aeq) if Aeq else None, b_eq=np.array(beq) if beq else None, bounds=bounds, method="highs")
    if res.status == 2:
        return Result(None, "infeasible", "no book satisfies the constraints")
    if not res.success:
        return Result(None, "failed", str(res.message))
    w = pd.Series(res.x[:n] - res.x[n:2 * n], index=names)
    w = w.where(w.abs() > 1e-12, 0.0)
    return Result(w, "optimal", "", {"cvar_alpha": a, "scenarios": int(K), "objective": float(res.fun)})


def scenario_matrix(returns: pd.DataFrame, asof_pos: int, window: int, horizon: int, names: list[str]) -> np.ndarray:
    """Overlapping ``horizon``-day compounded returns over the ``window`` days to row ``asof_pos`` inclusive, de-meaned (the forecast supplies the mean)."""
    past = returns[names].iloc[max(0, asof_pos - window + 1):asof_pos + 1].dropna(how="any")
    if len(past) < horizon + 20:
        return np.empty((0, len(names)))
    comp = np.expm1(np.log1p(past.clip(lower=-0.999999)).rolling(horizon).sum()).dropna()
    s = comp.to_numpy(dtype=float)
    return s - s.mean(axis=0, keepdims=True)


def allocate(allocator: str, mu: pd.Series, sd: pd.Series, cov_w: pd.DataFrame, vol_daily: pd.Series, w0: pd.Series, cfg: AllocConfig, groups: dict | None, est: Estimate | None,
             scenarios: np.ndarray | None = None) -> Result:
    """One allocation. ``mu`` and ``cov_w`` are for the holding window; ``mu`` is the calibrated forecast. Heuristic allocators ignore ``mu`` and are projected onto the limits; the optimisers solve under the limits."""
    lim = cfg.limits
    names = list(mu.index)
    n = len(names)
    zero = pd.Series(0.0, index=names)
    if allocator == "equal_weight":
        return Result(project(pd.Series(1.0 / n, index=names), lim, groups), "optimal")
    if allocator == "inverse_vol":
        v = vol_daily.reindex(names)
        if not (v > 0).all():
            return Result(None, "failed", "a volatility is zero or missing")
        w = (1.0 / v) / (1.0 / v).sum()
        return Result(project(w, lim, groups), "optimal")
    if allocator == "erc":
        from ..portfolio.risk_parity import risk_parity_weights

        if n == 1:
            return Result(project(pd.Series(1.0, index=names), lim, groups), "optimal")
        budget = None if lim.risk_budget is None else pd.Series(dict(lim.risk_budget)).reindex(names)
        try:
            w = risk_parity_weights(cov_w, budget)
        except Exception as exc:                                                             # noqa: BLE001
            return Result(None, "failed", f"the risk-parity solver failed: {exc}")
        return Result(project(w, lim, groups), "optimal")
    if allocator == "hrp":
        if n < 3:
            return Result(None, "unsuitable", f"hierarchical risk parity needs at least three assets to build a tree; {n} given (with two it is inverse variance: use inverse volatility)")
        from ..portfolio.hierarchical import hrp_weights

        try:
            w = hrp_weights(cov_w)
        except Exception as exc:                                                             # noqa: BLE001
            return Result(None, "failed", f"HRP failed: {exc}")
        return Result(project(w, lim, groups), "optimal")
    if allocator == "mean_variance":
        return _solve(mu, cov_w, w0, cfg, groups)
    if allocator == "robust_mv":
        slope_se = est.slope_se if est is not None and est.slope_se is not None and np.isfinite(est.slope_se) else 0.0
        raw = est.mu_raw if est is not None else mu
        pen = cfg.robust_kappa * raw.abs().reindex(names) * slope_se
        return _solve(mu, cov_w, w0, cfg, groups, penalty=pen)
    if allocator == "kelly":
        if est is None or est.status not in ("calibrated",) or est.t_stat is None or not np.isfinite(est.t_stat) or est.t_stat < cfg.kelly_min_t:
            return Result(None, "not_justified", f"fractional Kelly needs a calibrated forecast with t >= {cfg.kelly_min_t:g}; this one is {'unavailable' if est is None else est.status} (t {None if est is None else est.t_stat})")
        return _solve(mu, cov_w, w0, cfg, groups, risk_aversion=1.0 / cfg.kelly_fraction)
    if allocator == "black_litterman":
        from ..portfolio.black_litterman import black_litterman_posterior, implied_equilibrium_returns

        cov = cov_w
        prior = implied_equilibrium_returns(cov, pd.Series(1.0 / n, index=names), cfg.risk_aversion)
        if est is None or est.status != "calibrated" or sd.isna().any():
            post = prior                                                                      # no view the past supports: the posterior is the equilibrium
        else:
            ne = max(est.n_eff, 1.0)
            view_var = (sd.reindex(names) ** 2) / ne
            views = pd.DataFrame(np.eye(n), index=names, columns=names)
            try:
                post = black_litterman_posterior(cov, prior, views, mu.reindex(names), tau=cfg.tau, view_variance=view_var + 1e-12)
            except Exception as exc:                                                         # noqa: BLE001
                return Result(None, "failed", f"the Black-Litterman posterior failed: {exc}")
        out = _solve(post, cov, w0, cfg, groups)
        out.detail["posterior"] = {str(k): float(v) for k, v in post.items()}
        out.detail["prior"] = {str(k): float(v) for k, v in prior.items()}
        return out
    if allocator in ad.ADAPTERS:
        try:
            raw = ad.raw_book(allocator, mu, cov_w, scenarios, cfg.risk_aversion, lim.min_weight if lim.long_only else 0.0, lim.max_weight, names)
        except Exception as exc:                                                             # noqa: BLE001 - an optimiser failure is a failed decision with its reason
            return Result(None, "failed", f"{DOCS[allocator]['label']} failed: {exc}")
        if raw.isna().any() or abs(float(raw.sum()) - 1.0) > 1e-6:
            return Result(None, "failed", f"{DOCS[allocator]['label']} returned an invalid book")
        w = project(raw, lim, groups)                                                        # what the optimiser cannot honour itself (caps, floors, sector caps) is enforced by the nearest feasible book
        return Result(w, "optimal", "", {"projection_l1": float((w - raw).abs().sum()), "optimiser_book": {str(k): float(v) for k, v in raw.items()}, "uses_forecast": ad.ADAPTERS[allocator].uses_forecast})
    if allocator == "cvar":
        if scenarios is None or len(scenarios) < 60:
            return Result(None, "failed", f"CVaR needs at least 60 scenarios of window returns; {0 if scenarios is None else len(scenarios)} available")
        return _cvar_lp(mu, scenarios, w0, cfg, groups)
    raise AllocationError(f"unknown allocator '{allocator}'")


# ------------------------------------------------------------------------------------------------------------ the walk-forward
def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> dict:
    df = rm.risk_contributions(w, cov)
    return {str(k): {"weight": float(r["weight"]), "share": None if not np.isfinite(r["share"]) else float(r["share"])} for k, r in df.iterrows()}


def _explain(final: pd.Series, risk_only: pd.Series, equal: pd.Series, lim: Limits, groups: dict | None) -> dict:
    risk_eff = risk_only - equal
    fc_eff = final - risk_only
    r1, f1 = float(risk_eff.abs().sum()), float(fc_eff.abs().sum())
    share = f1 / (r1 + f1) if (r1 + f1) > 1e-12 else 0.0
    bind = binding(final, lim, groups)
    if (r1 + f1) <= 1e-9:
        driver = "equal"
    elif bind and share < 0.35:
        driver = "constraints"
    else:
        driver = "forecast" if share >= 0.5 else "risk"
    return {"equal": {str(k): float(v) for k, v in equal.items()}, "risk_only": {str(k): float(v) for k, v in risk_only.items()}, "risk_effect": {str(k): float(v) for k, v in risk_eff.items()},
            "forecast_effect": {str(k): float(v) for k, v in fc_eff.items()}, "forecast_share": float(share), "driver": driver, "binding": bind}


def _apply_evidence(est: Estimate, cfg: AllocConfig) -> tuple[Estimate, dict]:
    """Shrink and gate a calibrated forecast by the evidence behind it: the slope minus ``uncertainty_shrink`` standard errors, no forecast at all below ``min_n_eff`` independent windows, and none for an asset whose
    slope is positive with less than ``min_confidence`` probability. Returns the estimate to use and what was done (empty when nothing applied)."""
    info: dict = {}
    if est.status != "calibrated":
        return est, info
    cols = list(est.mu_raw.index)
    slopes = est.slopes.reindex(cols).fillna(0.0) if est.slopes is not None else pd.Series(float(est.slope or 0.0), index=cols)
    if cfg.min_n_eff > 0 and est.n_eff < cfg.min_n_eff:
        info["min_n_eff"] = f"{est.n_eff:.1f} independent outcome windows; {cfg.min_n_eff:g} are required: no forecast is used"
        return replace(est, mu=est.mu * 0.0, status="no_edge", note=info["min_n_eff"]), info
    new = slopes.copy()
    if cfg.uncertainty_shrink > 0:
        new = stab.uncertainty_shrunk_slopes(slopes, est.slope_se, cfg.uncertainty_shrink)
        info["uncertainty_shrink"] = {"kappa": cfg.uncertainty_shrink, "slope_before": {str(k): float(v) for k, v in slopes.items()}, "slope_after": {str(k): float(v) for k, v in new.items()}}
    conf = est.confidence if est.confidence is not None else pd.Series(float(_norm_cdf(est.t_stat)) if est.t_stat is not None and np.isfinite(est.t_stat) else 0.0, index=cols)
    if cfg.min_confidence > 0:
        low = conf.reindex(cols).fillna(0.0) < cfg.min_confidence
        if low.any():
            new = new.where(~low, 0.0)
            info["min_confidence"] = {"threshold": cfg.min_confidence, "dropped": [str(c) for c in low.index[low]]}
    if not info:
        return (est if est.confidence is not None else replace(est, confidence=conf)), info
    mu = est.mu_raw * new
    status = "calibrated" if (new > 0).any() else "no_edge"
    return replace(est, mu=mu, slopes=new, status=status, confidence=conf), info


def _norm_cdf(t: float) -> float:
    return 0.5 * (1.0 + math.erf(float(t) / math.sqrt(2.0)))


def decision_sensitivity(inp: dict, cfg: AllocConfig, groups: dict | None, seed: int = 0) -> dict:
    """How much the last decision's book would move under perturbations: the forecast (resampled within the slope's standard error), the risk aversion (x1/4 .. x4), the transaction cost and the turnover penalty."""
    names, mu, sd, cov_w, vol, w0, est, scen = inp["names"], inp["mu"], inp["sd"], inp["cov_w"], inp["vol"], inp["w0"], inp["est"], inp["scen"]

    def book(c: AllocConfig, m: pd.Series) -> pd.Series | None:
        r = allocate(c.allocator, m, sd, cov_w, vol, w0, c, groups, est, scen)
        return r.weights.reindex(names).fillna(0.0) if r.weights is not None else None

    base = book(cfg, mu)
    out: dict = {"base": None if base is None else {str(k): float(v) for k, v in base.items()}}
    if base is None:
        return out
    rel = None
    if est is not None and est.slope and est.slope > 0 and est.slope_se is not None and np.isfinite(est.slope_se):
        rel = float(est.slope_se / est.slope)
    if rel is None:
        out["forecast"] = {"available": False, "note": "the forecast has no calibrated slope to perturb (uncalibrated, no edge or too little history)"}
    else:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(40):
            b = book(cfg, mu * max(0.0, 1.0 + float(rng.standard_normal()) * rel))
            if b is not None:
                draws.append(b.to_numpy())
        arr = np.array(draws)
        lo, hi = book(cfg, mu * max(0.0, 1.0 - rel)), book(cfg, mu * (1.0 + rel))
        out["forecast"] = {"available": True, "relative_se": rel, "draws": int(len(arr)), "weight_mean": {n: float(v) for n, v in zip(names, arr.mean(axis=0))}, "weight_sd": {n: float(v) for n, v in zip(names, arr.std(axis=0, ddof=1))},
                           "weight_min": {n: float(v) for n, v in zip(names, arr.min(axis=0))}, "weight_max": {n: float(v) for n, v in zip(names, arr.max(axis=0))},
                           "weight_at_minus_1se": None if lo is None else {str(k): float(v) for k, v in lo.items()}, "weight_at_plus_1se": None if hi is None else {str(k): float(v) for k, v in hi.items()}}
    ra = []
    for f in (0.25, 0.5, 1.0, 2.0, 4.0):
        b = book(replace(cfg, risk_aversion=cfg.risk_aversion * f), mu)
        ra.append({"risk_aversion": cfg.risk_aversion * f, "weights": None if b is None else {str(k): float(v) for k, v in b.items()}, "concentration": None if b is None else stab.concentration(b)})
    out["risk_aversion"] = ra
    costs = []
    for c_bps, pen in ((0.0, 0.0), (cfg.cost_bps, cfg.turnover_penalty), (max(cfg.cost_bps * 5.0, 25.0), cfg.turnover_penalty), (100.0, cfg.turnover_penalty + 5.0)):
        b = book(replace(cfg, cost_bps=c_bps, turnover_penalty=pen), mu)
        costs.append({"cost_bps": c_bps, "turnover_penalty": pen, "weights": None if b is None else {str(k): float(v) for k, v in b.items()}, "turnover": None if b is None else float((b - w0).abs().sum() / 2.0)})
    out["costs"] = costs
    return out


def run(score: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, sig: dict, groups: dict | None = None, *, prices: pd.DataFrame | None = None, probability: pd.DataFrame | None = None,
        predictive_sd: pd.DataFrame | None = None, allocator: str | None = None, fx: "fxm.FxContext | None" = None) -> tuple[pd.DataFrame, dict]:
    """Target weights for every row of ``score`` plus the log of every decision. ``sig`` is the ``signal`` section of an experiment configuration (``signal.allocation`` holds the settings, ``window_days`` the
    dataset's horizon and ``lag_total`` the lag between a decision and the first return it earns). Row ``t`` uses the scores, returns and eligibility of rows up to ``t``."""
    cfg = config_from_signal(sig)
    if allocator is not None:
        cfg = AllocConfig(**{**{k: getattr(cfg, k) for k in _CFG_KEYS}, "allocator": allocator, "limits": cfg.limits})
        check_config(cfg)
    h = int(sig["window_days"])
    lag = int(sig.get("lag_total", 1))
    check_forecast_contract(cfg, h, None, fx)
    fx_report = None
    if fx is not None:                                                                 # everything below is then read in the base currency
        try:
            score, returns, fx_excluded, fx_report = fxm.prepare(score, returns, fx, h, cfg.forecast_kind, prices)
        except fxm.FxError as exc:
            raise AllocationError(str(exc)) from None
        if fx_excluded is not None:
            investable = fx_excluded if investable is None else (investable.reindex_like(fx_excluded).fillna(False).astype(bool) & fx_excluded)
        if cfg.forecast_kind == "price":
            cfg = replace(cfg, forecast_kind="return")
    L = int(cfg.holding_days) if cfg.holding_days else h                              # the window the optimisation is posed over: the holding window, or the forecast horizon when none is set
    if cfg.forecast_source == "direct" and L != h:
        raise AllocationError(f"a direct holding-horizon forecast must be a forecast of the holding window: the model's target is {h} trading days and the holding window is {L}. Train on fwd_ret_{L} "
                              f"(dataset.horizon {L}) or use forecast_source 'scaled', which states its scaling assumption")
    every = int(cfg.opt_every or L)
    K = hz.tranche_count(L, every) if cfg.lifecycle == "staggered_tranches" else 1   # overlapping tranches alive at once
    tranches: list = []                                                                # the last K tranche books (staggered lifecycle only)
    cols = list(score.columns)
    need = int(DOCS[cfg.allocator]["min_assets"])
    if len(cols) < need:
        raise AllocationError(f"{DOCS[cfg.allocator]['label']} needs at least {need} assets and this universe has {len(cols)}: {DOCS[cfg.allocator]['failure']}")
    ret = returns.reindex(index=score.index, columns=cols)
    fm = ForecastModel(score, ret, cfg, h, lag, prices, probability, predictive_sd, groups)
    cov_spec = rm.CovSpec(cfg.cov_method, window=cfg.cov_window, halflife=cfg.cov_halflife, min_obs=cfg.cov_min_obs)
    out = np.zeros(score.shape)
    last = pd.Series(0.0, index=cols)

    def commit(book: pd.Series) -> pd.Series:
        """The book held after a decision. One book: the decision itself, superseding the last (overlapping forecasts are never added). Staggered tranches: the average of the last K tranche books (each a full
        book of its own, held for L sessions), so that overlapping forecasts are K partial positions and not K independent full-size trades."""
        if K == 1:
            return book
        tranches.append(book.copy())
        del tranches[:-K]
        return sum(tranches) / float(len(tranches))

    def start_book() -> pd.Series:
        """What the tranche being opened replaces: the tranche that is expiring, or nothing while the first K are being opened."""
        return tranches[0].copy() if K > 1 and len(tranches) == K else (pd.Series(0.0, index=cols) if K > 1 else last)
    records: list[dict] = []
    blend_hist: list = []                                                              # (row, baseline book, ML book) of each past decision, for the validation rule for alpha
    real_L = realised_window_returns(ret, L, lag) if cfg.blend == "validation" else None
    last_inputs: dict | None = None
    started = False
    counts = {"no_risk_estimate": 0, "decisions": 0, "optimal": 0, "failed": 0, "infeasible": 0, "not_justified": 0, "unsuitable": 0, "insufficient_history": 0, "no_edge": 0, "failure_policy_used": 0, "turnover_limited": 0, "vol_scaled": 0,
              "single_asset": 0, "repaired_covariance": 0}
    for i, date in enumerate(score.index):
        if i % every != 0:
            out[i] = last.to_numpy()
            continue
        row = score.iloc[i]
        names = [a for a in cols if pd.notna(row[a]) and (investable is None or bool(investable.reindex(index=[date], columns=[a]).iloc[0, 0]))]
        if not names:
            out[i] = last.to_numpy()
            continue
        counts["decisions"] += 1
        rec: dict = {"date": str(pd.Timestamp(date).date()), "assets": names, "allocator": cfg.allocator}
        try:
            require_feasible(cfg.limits, names, groups, cfg.allocator)
        except InfeasibleConstraints as exc:
            counts["infeasible"] += 1
            rec.update(status="infeasible", reason=str(exc))
            records.append(rec)
            w = commit(_failure_book(cfg, last, names, cols, None, groups))
            counts["failure_policy_used"] += 1
            rec["failure_policy"] = cfg.on_failure
            out[i] = w.to_numpy()
            last = w
            continue
        est = fm.asof(i)
        if est.status in ("insufficient_history", "no_edge"):
            counts[est.status] += 1
        rec["forecast"] = est.info()
        est, evid = _apply_evidence(est, cfg)
        if evid:
            rec["evidence"] = evid
            rec["forecast"]["status_after_evidence_controls"] = est.status
        try:
            fit = rm.fit_asof(ret, date, cov_spec, investable) if len(names) >= 2 else None
        except ValueError as exc:
            fit = None
            rec["risk_error"] = str(exc)
        if len(names) == 1:
            counts["single_asset"] += 1
            past = ret[names[0]].iloc[max(0, i - cfg.cov_window + 1):i + 1].dropna()
            if len(past) < cfg.cov_min_obs:
                fit_cov = None
            else:
                fit_cov = pd.DataFrame([[float(past.var(ddof=1))]], index=names, columns=names)
                rec["risk"] = {"method": "single-asset sample variance", "n_obs": int(len(past))}
        elif fit is not None:
            fit_cov = fit.matrix.reindex(index=names, columns=names) if all(a in fit.matrix.index for a in names) else None
            if fit_cov is not None:
                rec["risk"] = fit.record()
                counts["repaired_covariance"] += int(fit.repaired)
        else:
            fit_cov = None
        if fit_cov is None or fit_cov.isna().any().any():
            why = rec.get("risk_error", "no covariance could be estimated as of this date (too few observations)")
            if not started:                                                                    # before the first risk estimate there is nothing to allocate on: the book is flat, not a guess
                rec.update(status="no_risk_estimate", reason=why, weights={})
                counts["no_risk_estimate"] += 1
                records.append(rec)
                out[i] = last.to_numpy()
                continue
            rec.update(status="failed", reason=why)
            counts["failed"] += 1
            counts["failure_policy_used"] += 1
            records.append(rec)
            w = commit(_failure_book(cfg, last, names, cols, None, groups))
            rec["failure_policy"] = cfg.on_failure
            out[i] = w.to_numpy()
            last = w
            continue
        started = True
        window_ret = ret[names].loc[:date].tail(cfg.cov_window)
        cov_w, hinfo = hz.horizon_covariance(fit_cov, window_ret, L, cfg.cov_horizon)
        rec["horizon"] = {"forecast_days": h, "holding_days": L, "decision_every": every, "lag": lag, "lifecycle": cfg.lifecycle, "tranches": K, "covariance": hinfo}
        rec["horizon"].update(forecast_source=cfg.forecast_source, edge_model=cfg.edge_model, beyond_horizon=cfg.edge_beyond_horizon,
                              assumptions=hz.assumptions(L, h, cfg.edge_model, cfg.edge_beyond_horizon, cfg.cov_horizon, cfg.forecast_source))
        if cfg.holding_days and cfg.forecast_source == "scaled":               # an explicit holding window: the h-day forecast is scaled to it under the stated assumption
            sc = hz.edge_scale(L, h, cfg.edge_beyond_horizon)
            est = replace(est, mu=hz.scale_expected_return(est.mu, L, h, cfg.edge_model, cfg.edge_beyond_horizon), mu_raw=hz.scale_expected_return(est.mu_raw, L, h, cfg.edge_model, cfg.edge_beyond_horizon),
                          sd=est.sd * math.sqrt(sc))
            rec["horizon"]["edge_scale"] = sc
        vol_daily = pd.Series(np.sqrt(np.diag(fit_cov.to_numpy())), index=names)
        mu = est.mu.reindex(names).fillna(0.0)
        sd = est.sd.reindex(names)
        extra_bad: list = []
        blend_alpha_used, blend_base_book = None, None
        w0 = start_book().reindex(names).fillna(0.0)
        scen = scenario_matrix(ret, i, cfg.cov_window, L, names) if (cfg.allocator == "cvar" or cfg.allocator in ad.NEEDS_SCENARIOS) else None
        res = allocate(cfg.allocator, mu, sd, cov_w, vol_daily, w0, cfg, groups, est, scen)
        rec["allocator_detail"] = res.detail
        if res.weights is None:
            rec.update(status=res.status, reason=res.reason)
            counts[res.status] = counts.get(res.status, 0) + 1
            counts["failure_policy_used"] += 1
            rec["failure_policy"] = cfg.on_failure
            w = _failure_book(cfg, last, names, cols, (cov_w, vol_daily, cfg.limits), groups)
        else:
            counts["optimal"] += 1
            rec["status"] = "optimal"
            w = res.weights.reindex(names).fillna(0.0)
            if cfg.limits.min_weight > 0 and cfg.limits.long_only and cfg.allocator in USES_FORECAST:
                w = project(w, cfg.limits, groups)                                         # the optimiser has no floor of its own: enforce it by the nearest feasible book
            w_ml = w.copy()
            if cfg.blend != "none":
                base_res = allocate(cfg.baseline, mu * 0.0, sd, cov_w, vol_daily, w0, cfg, groups, None, scen)
                if base_res.weights is None:
                    rec["blend"] = {"status": "baseline_failed", "reason": base_res.reason, "alpha": 1.0}
                else:
                    wb = base_res.weights.reindex(names).fillna(0.0)
                    if cfg.blend == "fixed":
                        alpha, ainfo = cfg.blend_alpha, {"note": "the configured alpha"}
                    elif cfg.blend == "confidence":
                        alpha, note = stab.confidence_alpha(est.confidence, names, cfg.blend_max)
                        ainfo = {"note": note}
                    else:
                        alpha, ainfo = stab.validation_alpha(blend_hist, real_L, i, lag, L, cfg.blend_window, cfg.blend_min_obs, cfg.risk_aversion)
                    full_b, full_m = pd.Series(0.0, index=cols), pd.Series(0.0, index=cols)
                    full_b[names], full_m[names] = wb, w_ml
                    blend_hist.append((i, full_b, full_m))
                    w = (1.0 - alpha) * wb + alpha * w_ml
                    blend_base_book, blend_alpha_used = wb, alpha
                    rec["blend"] = {"status": "blended", "mode": cfg.blend, "baseline": cfg.baseline, "alpha": float(alpha), "baseline_weights": {str(k): float(v) for k, v in wb.items()}, "ml_weights": {str(k): float(v) for k, v in w_ml.items()}, **ainfo}
                    bad_b = violations(w, cfg.limits, groups)
                    if bad_b:                                                              # a blend of two feasible books breaks nothing the limits make convex; if it does, the nearest feasible book is used and the move is reported
                        try:
                            wp = project(w, cfg.limits, groups)
                            rec["blend"]["projection"] = {"reason": bad_b, "l1_deviation": float((wp - w).abs().sum())}
                            w = wp
                        except InfeasibleConstraints as exc:
                            rec["blend"]["projection"] = {"reason": bad_b, "failed": str(exc)}
            w, limited = limit_turnover(w, w0, cfg.limits)
            counts["turnover_limited"] += int(limited)
            rec["turnover_limited"] = bool(limited)
            if cfg.max_weight_change is not None and float(w0.abs().sum()) > 1e-12:           # the first book is built from nothing: the limit governs the moves after it
                lo_b, hi_b = _bounds(cfg.limits, len(names))
                total = 1.0 if (cfg.limits.long_only and not cfg.limits.cash_allowed) else None
                w2, moved, ok = stab.weight_change_limited(w, w0, cfg.max_weight_change, float(lo_b[0]), float(hi_b[0]), total)
                if not ok:
                    extra_bad.append(f"no book within {cfg.max_weight_change:g} of the previous weights can satisfy the limits")
                elif moved:
                    rec["weight_change_limited"] = {"limit": cfg.max_weight_change, "l1_change": float((w2 - w).abs().sum())}
                    counts["weight_change_limited"] = counts.get("weight_change_limited", 0) + 1
                    w = w2
            w, ann, f = scale_to_vol(w, cov_w, L, cfg.limits)
            if cfg.limits.vol_target is not None:
                counts["vol_scaled"] += int(abs(f - 1.0) > 1e-9)
                rec["vol_scaling"] = {"forecast_vol": ann, "factor": f, "target": cfg.limits.vol_target}
        bad = violations(w, cfg.limits, groups) + extra_bad
        if bad:                                                                                # never hand a book that breaks a limit to the backtest
            rec.update(status="infeasible", reason="the book broke: " + "; ".join(bad))
            counts["infeasible"] += 1
            w = _failure_book(cfg, last, names, cols, (cov_w, vol_daily, cfg.limits), groups)
            rec["failure_policy"] = cfg.on_failure
        # the explanation: equal base, risk-only book, final book
        eq = project(pd.Series(1.0 / len(names), index=names), cfg.limits, groups) if cfg.limits.long_only else pd.Series(1.0 / len(names), index=names) * min(1.0, cfg.limits.gross_max)
        ro = _risk_only(cfg, res, mu, sd, cov_w, vol_daily, w0, groups, scen)
        risk_only = (ro if ro is not None else eq).reindex(names).fillna(0.0)
        if blend_base_book is not None:                                                    # the same blend with no forecast: what the policy would hold on risk alone
            risk_only = (1.0 - blend_alpha_used) * blend_base_book + blend_alpha_used * risk_only
        wf = w.reindex(names).fillna(0.0)
        rec["weights"] = {str(k): float(v) for k, v in wf.items()}
        rec["mu"] = {str(k): float(v) for k, v in mu.items()}
        rec["mu_raw"] = {str(k): None if not np.isfinite(v) else float(v) for k, v in est.mu_raw.reindex(names).items()}
        rec["sd"] = {str(k): None if not np.isfinite(v) else float(v) for k, v in sd.items()}
        rec["volatility_daily"] = {str(k): float(v) for k, v in vol_daily.items()}
        rec["expected_return"] = float(wf @ mu)
        rec["forecast_vol_window"] = float(math.sqrt(max(wf.to_numpy() @ cov_w.to_numpy() @ wf.to_numpy(), 0.0)))
        rec["risk_contributions"] = risk_contributions(wf, cov_w)
        rec["explain"] = _explain(wf, risk_only, eq, cfg.limits, groups)
        rec["concentration"] = stab.concentration(wf)
        rec["covariance_window"] = {str(a): {str(b): float(cov_w.loc[a, b]) for b in names} for a in names}
        rec["correlation"] = {str(a): {str(b): float(cov_w.loc[a, b] / math.sqrt(cov_w.loc[a, a] * cov_w.loc[b, b])) if cov_w.loc[a, a] > 0 and cov_w.loc[b, b] > 0 else None for b in names} for a in names}
        if cfg.sensitivity:
            last_inputs = {"names": names, "mu": mu, "sd": sd, "cov_w": cov_w, "vol": vol_daily, "w0": w0, "est": est, "scen": scen, "rec": rec}
        rec["turnover_from_previous"] = float((wf - w0).abs().sum() / 2.0)
        records.append(rec)
        full = pd.Series(0.0, index=cols)
        full[names] = wf
        held = commit(full)
        if K > 1:                                                                              # the held book is an average of books that each satisfy the limits: check it as well, and keep it visible
            bad_held = violations(held.reindex(names).fillna(0.0), cfg.limits, groups)
            rec["held_weights"] = {str(k): float(v) for k, v in held.reindex(names).fillna(0.0).items()}
            rec["tranche"] = {"active": len(tranches), "of": K, "replaced": None if len(tranches) < K else "the oldest, expiring tranche"}
            if bad_held:
                rec["held_violations"] = bad_held
        out[i] = held.to_numpy()
        last = held
    weights = pd.DataFrame(out, index=score.index, columns=cols)
    if last_inputs is not None:
        try:
            last_inputs["rec"]["sensitivity"] = decision_sensitivity(last_inputs, cfg, groups)
        except Exception as exc:                                                                 # noqa: BLE001 - a diagnostic must never take the allocation down
            last_inputs["rec"]["sensitivity"] = {"error": str(exc)}
    shown = [r for r in records if r.get("weights") and r.get("explain")]
    if shown:                                                                                    # the plain-language account of the latest decision, from its own record
        shown[-1]["narrative"] = stab.narrative(shown[-1], cfg.content())
    log = {"version": VERSION, "config": cfg.content(), "config_hash": cfg.content_hash(), "allocator": cfg.allocator, "documentation": DOCS[cfg.allocator], "holding_window_days": L, "forecast_horizon_days": h, "lag": lag, "opt_every": every,
           "counts": counts, "records": records, **({"fx": fx_report} if fx_report is not None else {}), "note": "weights at row t use the scores, returns and eligibility known at t; the backtest applies the execution lag, drift, costs and the ledger"}
    return weights, log


def _risk_only(cfg: AllocConfig, res: Result, mu: pd.Series, sd: pd.Series, cov_w: pd.DataFrame, vol_daily: pd.Series, w0: pd.Series, groups: dict | None, scen) -> pd.Series | None:
    """The same allocator with a zero forecast: what the book would be on risk and the constraints alone. An allocator that never reads a forecast is its own risk-only book."""
    if cfg.allocator not in USES_FORECAST:
        return res.weights
    if cfg.allocator == "kelly":
        r = _solve(mu * 0.0, cov_w, w0, cfg, groups)
    else:
        r = allocate(cfg.allocator, mu * 0.0, sd, cov_w, vol_daily, w0, cfg, groups, None, scen)
    return r.weights


def _failure_book(cfg: AllocConfig, last: pd.Series, names: list[str], cols: list[str], risk, groups: dict | None) -> pd.Series:
    """The book used when a decision has no solution, as the configuration said (never silent: the record carries ``failure_policy``)."""
    if cfg.on_failure == "hold":
        return last.copy()
    if cfg.on_failure == "cash":
        return pd.Series(0.0, index=cols)
    out = pd.Series(0.0, index=cols)
    eq = pd.Series(1.0 / len(names), index=names)
    try:
        if cfg.on_failure == "risk_only" and risk is not None:
            cov_w, vol, lim = risk
            w = project((1.0 / vol) / (1.0 / vol).sum(), lim, groups)
        else:
            w = project(eq, cfg.limits, groups)
    except InfeasibleConstraints:
        return pd.Series(0.0, index=cols)
    out[names] = w
    return out
