"""Dependence-aware statistical inference for strategy returns, with one output contract (spec v3, Release 5, P1-16 and section 8.1).

Every function returns a ``StatResult``: the metric or test, its version, the hypothesis, the identity of the sample, the frequency and units, the risk-free convention, the estimator, the dependence
assumptions with their lag or block settings, the confidence level, the seed and replications, the selection-family scope, the value, interval and p-value *when they are valid*, an eligibility state
(``applied``, ``not_applicable``, ``unsupported``, ``insufficient`` or ``failed``, the vocabulary of ``selection.py``) and the limitations. A result that was not applied has no number in it.

Two things this module refuses to do:

* use a HAC standard error of the *mean* as if it were the standard error of a Sharpe ratio. The Sharpe ratio is a ratio of two moments; ``sharpe_inference`` takes the HAC long-run covariance of the
  moment conditions ``(x, x^2)`` and applies the delta method to the ratio, and it also gives a stationary-bootstrap interval. The naive annualised Sharpe and its iid standard error are reported
  separately, labelled naive, so the two can be compared and neither is mistaken for the other;
* treat overlapping horizons as independent: a series whose observations share a forecast horizon ``h`` is analysed with at least ``h - 1`` HAC lags (``hac_mean(horizon=h)``), and the result says so.

The references are independent: statsmodels' HAC for the mean, closed-form AR(1) factors, and simulation checks of interval coverage whose own uncertainty is stated.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..stats.inference import bootstrap_indices, optimal_block_length
from ..stats.regression import newey_west_lag

VERSION = "inference-v1"
ELIGIBILITY = ("applied", "not_applicable", "unsupported", "insufficient", "failed")
PERIODS = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4, "annual": 1, "daily_24x7": 365}
MIN_BOOT = 200                                                                                          # fewer resamples than this cannot place the tails of a percentile interval
MIN_OBS = 20                                                                                           # fewer observations than this and no dependence-aware estimate is attempted


@dataclass
class StatResult:
    """The statistical output contract. ``value``, ``interval`` and ``p_value`` are ``None`` unless ``eligibility`` is ``applied``."""

    name: str
    version: str
    hypothesis: str
    sample: dict
    frequency: str
    units: str
    risk_free: str
    estimator: str
    dependence: str
    settings: dict
    level: float
    seed: int | None
    replications: int | None
    scope: str
    value: float | None = None
    interval: tuple | None = None
    p_value: float | None = None
    eligibility: str = "applied"
    reason: str = ""
    limitations: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.eligibility not in ELIGIBILITY:
            raise ValueError(f"eligibility must be one of {ELIGIBILITY}")
        if self.eligibility != "applied":
            self.value, self.interval, self.p_value = None, None, None                                  # no number accompanies a procedure that was not applied

    def as_dict(self) -> dict:
        d = asdict(self)
        d["interval"] = None if self.interval is None else [float(self.interval[0]), float(self.interval[1])]
        return d


def sample_identity(x: pd.Series | np.ndarray, label: str = "") -> dict:
    """Which observations a result is about: the segment label, the count, the first and last date and a hash of the values."""
    s = x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, dtype=float))
    s = s.dropna()
    h = hashlib.sha256(np.ascontiguousarray(s.to_numpy(dtype=float)).tobytes()).hexdigest()[:16]
    first, last = (str(s.index[0]), str(s.index[-1])) if len(s) else (None, None)
    return {"segment": label, "n": int(len(s)), "first": first, "last": last, "values_hash": h}


def periods_per_year(frequency: str) -> int:
    if frequency not in PERIODS:
        raise ValueError(f"frequency must be one of {sorted(PERIODS)}: the annualisation factor is not guessed")
    return PERIODS[frequency]


def overlap_lags(n: int, horizon: int = 1, lags: int | None = None) -> int:
    """The HAC lag to use: the Newey-West rule of thumb, never fewer than ``horizon - 1`` when observations share a forecast horizon, or the lag the caller asked for if it is not smaller than that."""
    base = newey_west_lag(n) if lags is None else int(lags)
    return max(base, int(horizon) - 1)


def is_constant(a: np.ndarray) -> bool:
    """A series that does not vary beyond floating-point rounding (its spread is within 1e-12 of its own scale): ratios and t statistics of it are undefined, not huge."""
    a = np.asarray(a, dtype=float)
    return bool(len(a) == 0 or np.ptp(a) <= 1e-12 * max(float(np.abs(a).max()), 1e-300))


def _clean(x) -> tuple[np.ndarray, pd.Series]:
    s = x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, dtype=float))
    s = s.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return s.to_numpy(), s


def _insufficient(name: str, hypothesis: str, series: pd.Series, label: str, frequency: str, level: float, why: str, scope: str = "single pre-declared comparison", **kw) -> StatResult:
    return StatResult(name, VERSION, hypothesis, sample_identity(series, label), frequency, kw.get("units", "return per period"), kw.get("risk_free", "zero"), kw.get("estimator", "not computed"), "not assessed", {},
                      level, None, None, scope, eligibility="insufficient", reason=why, limitations=[why])


# ---------------------------------------------------------------------------------------------------------------------------------------- HAC for a mean
def long_run_variance(u: np.ndarray, lags: int) -> float:
    """The Bartlett-kernel (Newey-West) long-run variance of a mean-zero series: ``gamma_0 + 2 sum_k (1 - k / (L + 1)) gamma_k``."""
    n = len(u)
    out = float(u @ u) / n
    for k in range(1, min(lags, n - 1) + 1):
        out += 2.0 * (1.0 - k / (lags + 1.0)) * float(u[k:] @ u[:-k]) / n
    return out


def hac_mean(x, *, frequency: str = "daily", label: str = "", lags: int | None = None, horizon: int = 1, level: float = 0.95, scope: str = "single pre-declared comparison") -> StatResult:
    """The mean of a return series with a Newey-West standard error: ``Var(mean) = lrv / n``. ``horizon`` is the forecast horizon that the observations overlap by (their dependence is then at least
    ``horizon - 1`` lags long). The interval and p-value use Student's t with ``n - 1`` degrees of freedom: the HAC approximation is asymptotic and the t is a mild guard in small samples, not a correction."""
    a, s = _clean(x)
    hyp = "the mean return per period is zero"
    if len(a) < MIN_OBS:
        return _insufficient("mean_return_hac", hyp, s, label, frequency, level, f"{len(a)} observations: fewer than {MIN_OBS}, so no dependence-aware standard error is attempted")
    L = overlap_lags(len(a), horizon, lags)
    mu = float(a.mean())
    lrv = long_run_variance(a - mu, L)
    se = float(np.sqrt(max(lrv, 0.0) / len(a)))
    if se <= 0.0 or is_constant(a):
        return _insufficient("mean_return_hac", hyp, s, label, frequency, level, "the long-run variance is zero (a constant series): the t statistic is undefined")
    t = mu / se
    df = len(a) - 1
    crit = float(sps.t.ppf(0.5 + level / 2.0, df))
    return StatResult("mean_return_hac", VERSION, hyp, sample_identity(s, label), frequency, "return per period (fraction)", "none: raw returns", "Newey-West (Bartlett kernel) long-run variance of the mean",
                      f"serial dependence up to lag {L}" + (f" (overlapping horizon {horizon}: at least {horizon - 1})" if horizon > 1 else ""), {"lags": L, "horizon": int(horizon), "kernel": "bartlett", "dof": df},
                      level, None, None, scope, value=mu, interval=(mu - crit * se, mu + crit * se), p_value=float(2.0 * sps.t.sf(abs(t), df)),
                      limitations=["asymptotic: the HAC estimate is biased downward in short samples with strong dependence", "this is the standard error of the MEAN, not of a Sharpe ratio"],
                      extras={"se": se, "t": t, "naive_se": float(a.std(ddof=1) / np.sqrt(len(a)))})


# ---------------------------------------------------------------------------------------------------------------------------------------- the Sharpe ratio
def _sharpe_per_period(a: np.ndarray) -> float:
    sd = a.std(ddof=1)
    return float(a.mean() / sd) if sd > 0 else float("nan")


def sharpe_delta_hac(a: np.ndarray, lags: int) -> tuple[float, float]:
    """``(sharpe per period, standard error)``: the delta method on the moment conditions ``(x, x^2)`` with their HAC long-run covariance. ``SR = mu / sqrt(m2 - mu^2)``; the gradient is
    ``(1/s + mu^2/s^3, -mu/(2 s^3))`` with ``s`` the standard deviation. Uses the maximum-likelihood variance ``m2 - mu^2`` in the gradient, which differs from ``ddof=1`` by O(1/n)."""
    n = len(a)
    mu, m2 = float(a.mean()), float((a ** 2).mean())
    var = m2 - mu ** 2
    s = float(np.sqrt(var)) if var > 0 else float("nan")
    psi = np.column_stack([a - mu, a ** 2 - m2])
    omega = np.zeros((2, 2))
    omega += psi.T @ psi / n
    for k in range(1, min(lags, n - 1) + 1):
        g = psi[k:].T @ psi[:-k] / n
        omega += (1.0 - k / (lags + 1.0)) * (g + g.T)
    grad = np.array([1.0 / s + mu ** 2 / s ** 3, -mu / (2.0 * s ** 3)])
    return mu / s, float(np.sqrt(max(grad @ omega @ grad, 0.0) / n))


def sharpe_inference(returns, *, frequency: str = "daily", label: str = "", rf_per_period: float = 0.0, lags: int | None = None, horizon: int = 1, level: float = 0.95, n_boot: int = 2000,
                     seed: int = 0, scope: str = "single pre-declared comparison") -> dict:
    """The annualised Sharpe ratio of ``returns`` (excess of ``rf_per_period``) in three clearly separated parts: ``naive`` (the point estimate and the iid standard error), ``hac`` (the delta method with HAC
    moments) and ``bootstrap`` (a stationary-bootstrap percentile interval). Each is a ``StatResult``; the naive one says what it ignores."""
    a0, s0 = _clean(returns)
    ppy = periods_per_year(frequency)
    hyp = "the annualised Sharpe ratio is zero"
    rf = f"{rf_per_period:g} per period, subtracted before the ratio" if rf_per_period else "zero"
    if len(a0) < MIN_OBS:
        why = f"{len(a0)} observations: fewer than {MIN_OBS}"
        return {"naive": _insufficient("sharpe_naive", hyp, s0, label, frequency, level, why), "hac": _insufficient("sharpe_hac", hyp, s0, label, frequency, level, why),
                "bootstrap": _insufficient("sharpe_stationary_bootstrap", hyp, s0, label, frequency, level, why)}
    a = a0 - rf_per_period
    sd = a.std(ddof=1)
    if not sd > 0 or is_constant(a):
        why = "the series is constant: the Sharpe ratio is undefined"
        return {k: _insufficient(f"sharpe_{k}", hyp, s0, label, frequency, level, why) for k in ("naive", "hac", "bootstrap")}
    n, sr_p = len(a), float(a.mean() / sd)
    ann = float(np.sqrt(ppy))
    z = float(sps.norm.ppf(0.5 + level / 2.0))
    ident = sample_identity(s0, label)
    common = dict(hypothesis=hyp, sample=ident, frequency=frequency, units=f"annualised ({ppy} periods per year), excess of the risk-free convention", risk_free=rf, level=level, scope=scope)
    se_iid = float(np.sqrt((1.0 + 0.5 * sr_p ** 2) / n))
    naive = StatResult("sharpe_naive", VERSION, estimator="sample mean over sample standard deviation, times sqrt(periods per year); iid standard error sqrt((1 + SR^2/2)/n)", dependence="NONE assumed: observations "
                       "treated as independent and identically distributed", settings={"periods_per_year": ppy}, seed=None, replications=None, value=sr_p * ann,
                       interval=((sr_p - z * se_iid) * ann, (sr_p + z * se_iid) * ann), p_value=float(2.0 * sps.norm.sf(abs(sr_p / se_iid))),
                       limitations=["ignores serial dependence, overlapping horizons, skewness and kurtosis: too narrow for most strategy returns", "a point estimate chosen from many looks is inflated (see the selection tests)"],
                       extras={"sharpe_per_period": sr_p, "se_per_period": se_iid}, **common)
    L = overlap_lags(n, horizon, lags)
    _, se_h = sharpe_delta_hac(a, L)                                                                    # the delta method supplies the standard error; the point estimate is the one number all three share
    sr_h = sr_p
    hac = StatResult("sharpe_hac", VERSION, estimator="delta method on the moment conditions (x, x^2) with a Newey-West (Bartlett) long-run covariance", dependence=f"serial dependence up to lag {L}"
                     + (f" (overlapping horizon {horizon})" if horizon > 1 else ""), settings={"lags": L, "horizon": int(horizon), "periods_per_year": ppy}, seed=None, replications=None,
                     value=sr_h * ann, interval=((sr_h - z * se_h) * ann, (sr_h + z * se_h) * ann), p_value=float(2.0 * sps.norm.sf(abs(sr_h / se_h))) if se_h > 0 else None,
                     limitations=["asymptotic normal interval", "the HAC covariance of the squared returns needs a long sample for fourth moments to be estimated well"],
                     extras={"sharpe_per_period": sr_h, "se_per_period": se_h, "se_ratio_to_naive": se_h / se_iid}, **common)
    block = optimal_block_length(a, kind="stationary")
    block = max(float(block), float(horizon))                                                           # a block shorter than the overlap would break the dependence being estimated
    rng = np.random.default_rng(seed)
    draws = np.array([_sharpe_per_period(a[bootstrap_indices(n, "stationary", block, rng)]) for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    if len(draws) < MIN_BOOT:
        boot = _insufficient("sharpe_stationary_bootstrap", hyp, s0, label, frequency, level, f"{len(draws)} usable bootstrap replications: fewer than {MIN_BOOT}, too few to place the tails of a percentile interval")
        return {"naive": naive, "hac": hac, "bootstrap": boot}
    lo, hi = np.quantile(draws, [0.5 - level / 2.0, 0.5 + level / 2.0])
    boot = StatResult("sharpe_stationary_bootstrap", VERSION, estimator="Politis-Romano stationary bootstrap, percentile interval of the resampled Sharpe ratio", dependence=f"serial dependence preserved by "
                      f"resampling blocks of mean length {block:.1f}", settings={"mean_block": block, "periods_per_year": ppy}, seed=seed, replications=int(len(draws)), value=sr_p * ann,
                      interval=(float(lo) * ann, float(hi) * ann), p_value=float(min(1.0, 2.0 * min((draws <= 0).mean(), (draws >= 0).mean()))) if len(draws) else None,
                      limitations=["percentile interval: not bias-corrected", f"Monte Carlo error from {len(draws)} replications; the p-value is the two-sided share of resampled ratios on one side of zero"],
                      extras={"bootstrap_se_per_period": float(draws.std(ddof=1)) if len(draws) > 1 else None}, **common)
    return {"naive": naive, "hac": hac, "bootstrap": boot}


# ---------------------------------------------------------------------------------------------------------------------------------------- paired differences
def paired_difference(a, b, *, statistic: str = "sharpe", frequency: str = "daily", label: str = "", n_boot: int = 2000, seed: int = 0, level: float = 0.95, horizon: int = 1, lags: int | None = None,
                      scope: str = "single pre-declared comparison") -> StatResult:
    """The difference ``statistic(a) - statistic(b)`` of two return series on their COMMON dates, with the same resampled indices applied to both (so the dependence between them is kept) in a stationary
    bootstrap. ``statistic`` is ``sharpe`` (annualised) or ``mean``. The result records how many dates were aligned and how many were dropped from each series: nothing is aligned silently."""
    sa, sb = pd.Series(a).astype(float), pd.Series(b).astype(float)
    both = pd.concat([sa, sb], axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    hyp = f"{statistic}(a) - {statistic}(b) is zero"
    dropped = {"only_in_a": int(len(sa.dropna()) - len(both)), "only_in_b": int(len(sb.dropna()) - len(both))}
    ident = {**sample_identity(both.iloc[:, 0] - both.iloc[:, 1], label), "aligned": int(len(both)), "dropped": dropped}
    if statistic not in ("sharpe", "mean"):
        return StatResult("paired_difference", VERSION, hyp, ident, frequency, "-", "zero", "-", "-", {}, level, None, None, scope, eligibility="unsupported", reason="statistic must be sharpe or mean")
    if len(both) < MIN_OBS:
        r = _insufficient("paired_difference", hyp, both.iloc[:, 0], label, frequency, level, f"{len(both)} common observations: fewer than {MIN_OBS}", scope)
        r.sample = ident
        return r
    x, y = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    ppy = periods_per_year(frequency)
    f = (lambda u: _sharpe_per_period(u) * np.sqrt(ppy)) if statistic == "sharpe" else (lambda u: float(u.mean()))
    est = f(x) - f(y)
    block = max(float(optimal_block_length(x - y, kind="stationary")), float(horizon))
    rng = np.random.default_rng(seed)
    n = len(both)
    draws = []
    for _ in range(n_boot):
        idx = bootstrap_indices(n, "stationary", block, rng)
        draws.append(f(x[idx]) - f(y[idx]))
    draws = np.array([d for d in draws if np.isfinite(d)])
    if len(draws) < MIN_BOOT:
        r = _insufficient("paired_difference", hyp, both.iloc[:, 0], label, frequency, level, f"{len(draws)} usable bootstrap replications: fewer than {MIN_BOOT}", scope)
        r.sample = ident
        return r
    lo, hi = np.quantile(draws, [0.5 - level / 2.0, 0.5 + level / 2.0])
    centred = draws - draws.mean()                                                                          # the null distribution: the resampled differences centred on zero
    p = float((np.abs(centred) >= abs(est)).mean())
    extras = {"estimate_a": float(f(x)), "estimate_b": float(f(y)), "bootstrap_se": float(draws.std(ddof=1)), "correlation": float(np.corrcoef(x, y)[0, 1])}
    if statistic == "mean":
        extras["hac"] = hac_mean(x - y, frequency=frequency, label=label, lags=lags, horizon=horizon, level=level, scope=scope).as_dict()
    return StatResult("paired_difference", VERSION, hyp, ident, frequency, "annualised Sharpe ratio" if statistic == "sharpe" else "return per period", "zero", "joint stationary bootstrap of the aligned pairs",
                      f"serial dependence and cross-dependence preserved by resampling the same blocks of mean length {block:.1f} in both series", {"mean_block": block, "statistic": statistic}, level, seed,
                      int(len(draws)), scope, value=float(est), interval=(float(lo), float(hi)), p_value=p,
                      limitations=["the economic effect size is the value itself; the p-value says only whether the difference is distinguishable from resampling noise", "centred-bootstrap p-value: approximate in short samples"],
                      extras=extras)


def economic_effect(res: StatResult) -> dict:
    """The effect size beside the significance: the value, its interval and whether the interval excludes zero (statistical), so that neither is reported alone."""
    if res.eligibility != "applied":
        return {"value": None, "excludes_zero": None, "state": res.eligibility}
    lo, hi = res.interval
    return {"value": res.value, "interval": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0), "p_value": res.p_value, "state": "applied"}
