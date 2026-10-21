"""Factor evaluation: does a signal, known on each date, order the companies by what they earn next?

Everything here takes a **signal panel** (dates by securities, higher meaning "expect more", already transformed and signed) and the research context, and measures it against what happened after:

* the **information coefficient** (IC) on every date, the rank correlation between the signal and the forward return across companies: its mean, median, spread, share of positive dates, the
  information ratio of the IC, and a Newey-West t-statistic that allows for overlapping horizons;
* **quantile analysis**: the average forward return of each bucket (quintiles or deciles) and whether the buckets rise in order (monotonicity);
* a **long-short portfolio** (top bucket against bottom bucket, held one rebalance period) with its return, volatility, Sharpe, Sortino, drawdown, Calmar, hit rate and turnover, gross and after costs;
* **decay**: the IC at one week, one, three, six and twelve months;
* **breakdowns** by country, sector, region and market regime (bull or bear, high or low volatility, each known at the start of the period it describes);
* **stability** by calendar year and in halves;
* how far the result can be trusted: a 95% interval for the mean IC and for the Sharpe ratio, what the study could have detected at all (:func:`detectable_effect`), and the premium the signal
  earns after controlling for size, beta, volatility and momentum, with a Newey-West error (:func:`fama_macbeth`).

The verdict never says "validated". A factor that looks good here is "interesting": whether it survives out-of-sample data, the number of ideas tried, and costs is for :mod:`discovery`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..backtest.metrics import performance_summary
from ..models.regression import cross_sectional_ic
from . import factors as factor_library
from . import transforms
from .errors import FundamentalsError
from .jsonutil import clean, series_points
from .research import FREQ_DAYS, HORIZONS, ResearchContext, horizon_days

PERIODS_PER_YEAR = {"W": 52, "M": 12, "Q": 4}
DECAY_HORIZONS = ("1w", "1m", "3m", "6m", "12m")
CONTROLS = ("size", "beta", "volatility", "momentum")          # what a premium is measured after controlling for, unless asked otherwise
MIN_DATES = 24


@dataclass
class EvalSpec:
    """Everything that defines a factor study besides the data: stored with the result so it can be shown, hashed and run again."""

    factor: str | None = None                          # a library factor, or None when a panel is passed in directly
    steps: list = field(default_factory=lambda: [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}])
    direction: int | str = "theory"                    # +1, -1, or "theory" (the factor's own direction); "auto" picks the sign that worked, which the report counts as a second trial
    horizon: str | int = "1m"
    quantiles: int = 5
    top_fraction: float | None = None                  # the long and short legs; default 1 / quantiles
    weighting: str = "equal"                           # equal | signal | inv_vol
    cost_bps: float = 10.0                             # one-way, per unit of notional traded
    borrow_bps: float = 50.0                           # per year, on the short notional
    start: str | None = None
    end: str | None = None
    min_assets: int = 20
    breakdowns: tuple = ("country", "sector", "region", "regime")
    premium: bool = True                               # also run the Fama-MacBeth regression (a scan of many factors turns it off)
    controls: tuple = CONTROLS                         # what the premium is measured after controlling for; library factors may be added

    def describe(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict | None) -> "EvalSpec":
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in EvalSpec.__dataclass_fields__}
        for key in ("breakdowns", "controls"):
            if key in known:
                known[key] = tuple(known[key])
        return EvalSpec(**known)


# ------------------------------------------------------------------------------------------------------------------ statistics
def newey_west(x, lags: int | None = None) -> dict:
    """Mean of ``x`` with a Newey-West (Bartlett kernel) standard error: ``{mean, se, t, p, lags, n}``. ``lags`` defaults to the rule of thumb ``floor(4 (n/100)^(2/9))``."""
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(v)
    if n < 3:
        return {"mean": float(v.mean()) if n else float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan"), "lags": 0, "n": n}
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if lags is None else int(lags)
    lags = min(max(lags, 0), n - 2)
    d = v - v.mean()
    var = float(d @ d) / n
    for k in range(1, lags + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * float(d[k:] @ d[:-k]) / n
    se = math.sqrt(max(var, 0.0) / n)
    t = float(v.mean() / se) if se > 0 else float("nan")
    p = float(2.0 * stats.t.sf(abs(t), df=n - 1)) if math.isfinite(t) else float("nan")
    return {"mean": float(v.mean()), "se": se, "t": t, "p": p, "lags": lags, "n": n}


def overlap_lags(horizon: int, step: int) -> int:
    """Rebalance dates whose forward windows overlap: a horizon of ``h`` days sampled every ``step`` days overlaps its neighbours in ``ceil(h / step) - 1`` of them."""
    return max(int(math.ceil(horizon / step)) - 1, 0)


def ic_stats(ic: pd.Series, horizon: int, step: int, periods_per_year: int) -> dict:
    ic = ic.dropna()
    if ic.empty:
        return {"n_dates": 0}
    nw = newey_west(ic, max(overlap_lags(horizon, step), int(np.floor(4.0 * (len(ic) / 100.0) ** (2.0 / 9.0)))))
    sd = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    return {"n_dates": int(len(ic)), "mean_ic": float(ic.mean()), "median_ic": float(ic.median()), "std_ic": sd, "positive_share": float((ic > 0).mean()),
            "icir": float(ic.mean() / sd) if sd and sd > 0 else float("nan"), "icir_annualised": float(ic.mean() / sd * math.sqrt(periods_per_year)) if sd and sd > 0 else float("nan"),
            "t_stat": nw["t"], "p_value": nw["p"], "hac_lags": nw["lags"], "skew": float(stats.skew(ic, bias=False)) if len(ic) > 3 and sd and sd > 0 else float("nan"),
            "se": nw["se"], "ci95": [float(ic.mean() - 1.96 * nw["se"]), float(ic.mean() + 1.96 * nw["se"])] if np.isfinite(nw["se"]) else [float("nan"), float("nan")]}


def rank_ic(signal: pd.DataFrame, forward: pd.DataFrame, min_assets: int = 20) -> pd.Series:
    return cross_sectional_ic(signal, forward, method="spearman", min_assets=min_assets)


def pearson_ic(signal: pd.DataFrame, forward: pd.DataFrame, min_assets: int = 20) -> pd.Series:
    return cross_sectional_ic(signal, forward, method="pearson", min_assets=min_assets)


# ------------------------------------------------------------------------------------------------------------------ quantiles
def buckets(signal: pd.DataFrame, q: int) -> pd.DataFrame:
    """Bucket number 1 (lowest) to ``q`` (highest) by percentile rank on each date; missing where the signal is. Ties share a rank, so a signal with few distinct values fills fewer buckets."""
    pct = signal.rank(axis=1, pct=True, method="average")
    return np.ceil(pct * q).clip(1, q).where(signal.notna())


def quantile_returns(signal: pd.DataFrame, returns: pd.DataFrame, q: int) -> pd.DataFrame:
    b = buckets(signal, q)
    both = returns.reindex_like(signal)
    return pd.DataFrame({k: both.where(b == k).mean(axis=1) for k in range(1, q + 1)})


def monotonicity(means: pd.Series) -> dict:
    """How well the buckets rise in order: the rank correlation of bucket number with its mean return, and the share of adjacent buckets that rise."""
    m = means.dropna()
    if len(m) < 3:
        return {"spearman": float("nan"), "rising_pairs": float("nan"), "buckets": int(len(m))}
    rho = float(stats.spearmanr(np.arange(len(m)), m.to_numpy()).statistic)
    rising = float((np.diff(m.to_numpy()) > 0).mean())
    return {"spearman": rho, "rising_pairs": rising, "buckets": int(len(m))}


# ------------------------------------------------------------------------------------------------------------------ the portfolio
def holding_returns(ctx: ResearchContext) -> pd.DataFrame:
    """The return of holding a company from one rebalance date's entry (a day after the date) to the next one's: contiguous periods, so a compounded curve is honest. A company that stops
    trading is carried at its last price. The final date has no next one and is missing."""
    key = ("holding",)
    if key not in ctx._cache:
        entry = ctx.pos + ctx.entry_lag
        entry = np.minimum(entry, len(ctx.adj) - 1)
        held, carried = ctx.adj.to_numpy(float), ctx.adj.ffill().to_numpy(float)
        values = np.full((len(entry), ctx.n), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            values[:-1] = carried[entry[1:]] / held[entry[:-1]] - 1.0
        ctx._cache[key] = pd.DataFrame(np.where(np.isfinite(values), values, np.nan), index=ctx.dates, columns=ctx.ids)
    return ctx._cache[key]


def leg_weights(signal: pd.DataFrame, frac: float, weighting: str, ctx: ResearchContext, min_names: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long and short weights (each leg sums to one, a leg with fewer than ``min_names`` companies is empty), from the top and bottom ``frac`` of the signal on each date."""
    if weighting not in ("equal", "signal", "inv_vol"):
        raise FundamentalsError("weighting must be equal, signal or inv_vol")
    pct = signal.rank(axis=1, pct=True, method="average")
    valid = signal.notna().to_numpy()
    top, bottom = valid & (pct.to_numpy(float) > 1.0 - frac), valid & (pct.to_numpy(float) <= frac)
    if weighting == "equal":
        base = np.ones(signal.shape)
    elif weighting == "signal":
        base = np.abs(pct.to_numpy(float) - 0.5)
    else:
        vol = ctx.volatility(63).reindex_like(signal).to_numpy(float)
        base = np.where(vol > 0, 1.0 / np.where(vol > 0, vol, 1.0), 0.0)
    out = []
    for mask in (top, bottom):
        raw = np.where(mask, np.nan_to_num(base), 0.0)
        total = raw.sum(axis=1, keepdims=True)
        usable = (mask.sum(axis=1, keepdims=True) >= min_names) & (total > 0)
        out.append(pd.DataFrame(np.where(usable, raw / np.where(total > 0, total, 1.0), 0.0), index=signal.index, columns=signal.columns))
    return out[0], out[1]


def _traded(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Sum of absolute weight changes at each rebalance, measured against the weights as they drifted with the previous period's returns."""
    prev = weights.shift(1).fillna(0.0)
    drifted = prev * (1.0 + returns.shift(1).fillna(0.0))
    scale = drifted.sum(axis=1).replace(0.0, np.nan)
    drifted = drifted.div(scale, axis=0).fillna(0.0)
    return (weights - drifted).abs().sum(axis=1)


def long_short(signal: pd.DataFrame, ctx: ResearchContext, spec: EvalSpec) -> dict:
    """The top ``frac`` against the bottom ``frac``, rebalanced on every date and held to the next, with its costs."""
    frac = spec.top_fraction or 1.0 / spec.quantiles
    hold = holding_returns(ctx).reindex_like(signal)
    wl, ws = leg_weights(signal, frac, spec.weighting, ctx)
    long_ret, short_ret = (wl * hold.fillna(0.0)).sum(axis=1), (ws * hold.fillna(0.0)).sum(axis=1)
    live = (wl.sum(axis=1) > 0) & (ws.sum(axis=1) > 0) & hold.notna().any(axis=1)
    gross = (long_ret - short_ret).where(live)
    traded = _traded(wl, hold) + _traded(ws, hold)
    step_days = FREQ_DAYS[ctx.freq]
    cost = spec.cost_bps / 1e4 * traded
    borrow = spec.borrow_bps / 1e4 * step_days / 252.0
    net = (gross - cost - borrow).where(live)
    return {"gross": gross, "net": net, "long": long_ret.where(live), "short": short_ret.where(live), "turnover": (traded / 2.0).where(live), "weights_long": wl, "weights_short": ws, "frac": frac}


def portfolio_stats(returns: pd.Series, periods_per_year: int) -> dict:
    clean_returns = returns.dropna()
    if len(clean_returns) < 6:
        return {"n_periods": int(len(clean_returns))}
    summary = performance_summary(clean_returns, periods_per_year=periods_per_year)
    keep = ("n_obs", "cagr", "ann_return_arith", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown", "max_drawdown_date", "hit_rate", "total_return", "skew", "excess_kurtosis")
    out = {k: summary.get(k) for k in keep}
    out["t_stat"] = float(clean_returns.mean() / (clean_returns.std(ddof=1) / math.sqrt(len(clean_returns)))) if clean_returns.std(ddof=1) > 0 else float("nan")
    out["sharpe_ci95"] = sharpe_interval(clean_returns, periods_per_year)
    return out


def sharpe_interval(returns: pd.Series, periods_per_year: int, z: float = 1.96) -> list[float]:
    """A 95% interval for the annualised Sharpe ratio, from its large-sample standard error allowing for skew and fat tails (Mertens): ``se = sqrt((1 - skew * sr + (kurt - 1) / 4 * sr^2) / (n - 1))``
    for the per-period ratio ``sr`` and the raw kurtosis ``kurt`` (3 for a normal distribution)."""
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 8 or not r.std(ddof=1) > 1e-12:                                           # a flat series has no ratio (and its moments are all rounding)
        return [float("nan"), float("nan")]
    sr = float(r.mean() / r.std(ddof=1))
    skew, kurt = float(stats.skew(r, bias=False)), float(stats.kurtosis(r, bias=False, fisher=False))
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / (n - 1)
    se = math.sqrt(var) if var > 0 else float("nan")
    scale = math.sqrt(periods_per_year)
    return [float((sr - z * se) * scale), float((sr + z * se) * scale)] if np.isfinite(se) else [float("nan"), float("nan")]


# ------------------------------------------------------------------------------------------------------------------ regimes and groups
def regimes(ctx: ResearchContext) -> pd.DataFrame:
    """Two labels per rebalance date, each known on the date: ``trend`` (``bull`` when the equal-weighted market rose over the last twelve months, else ``bear``) and ``vol`` (``high`` when its
    three-month volatility is above the median of all earlier dates, else ``low``)."""
    key = ("regimes",)
    if key not in ctx._cache:
        m = ctx.market_return().fillna(0.0)
        level = (1.0 + m).cumprod()
        trailing = pd.Series(np.nan, index=ctx.dates)
        ok = ctx.pos >= 252
        trailing[ok] = level.to_numpy()[ctx.pos[ok]] / level.to_numpy()[ctx.pos[ok] - 252] - 1.0
        vol = (m.rolling(63, min_periods=40).std() * math.sqrt(252.0)).iloc[ctx.pos].set_axis(ctx.dates)
        median = vol.expanding(min_periods=12).median().shift(1)
        trend = pd.Series(np.where(trailing.isna(), None, np.where(trailing > 0, "bull", "bear")), index=ctx.dates)
        level_vol = pd.Series(np.where(median.isna() | vol.isna(), None, np.where(vol > median, "high", "low")), index=ctx.dates)
        ctx._cache[key] = pd.DataFrame({"trend": trend, "vol": level_vol})
    return ctx._cache[key]


def _group_ic(signal: pd.DataFrame, forward: pd.DataFrame, labels: pd.Series, horizon: int, step: int, ppy: int, min_names: int = 8) -> dict:
    out = {}
    for label, ids in labels.groupby(labels):
        cols = [c for c in ids.index if c in signal.columns]
        if len(cols) < min_names:
            continue
        ic = cross_sectional_ic(signal[cols], forward[cols], "spearman", min_names).dropna()
        if len(ic) < 12:
            continue
        s = ic_stats(ic, horizon, step, ppy)
        out[str(label)] = {"companies": len(cols), "n_dates": s["n_dates"], "mean_ic": s["mean_ic"], "t_stat": s["t_stat"], "positive_share": s["positive_share"]}
    return out


def _regime_ic(ic: pd.Series, labels: pd.Series, horizon: int, step: int, ppy: int) -> dict:
    out = {}
    for label in sorted({str(v) for v in labels.dropna()}):
        sub = ic[labels.reindex(ic.index).astype(object) == label].dropna()
        if len(sub) >= 12:
            s = ic_stats(sub, horizon, step, ppy)
            out[label] = {"n_dates": s["n_dates"], "mean_ic": s["mean_ic"], "t_stat": s["t_stat"], "positive_share": s["positive_share"]}
    return out


# ------------------------------------------------------------------------------------------------------------------ power
TYPICAL_IC = (0.01, 0.02, 0.03, 0.05, 0.08)


def detectable_effect(n_companies: float, n_dates: int, horizon: int, step: int, ic_sd: float | None = None, alpha: float = 0.05, power: float = 0.80) -> dict:
    """What a study of this size could tell apart from luck.

    On one date the rank correlation of a signal that predicts nothing has a standard deviation of about ``1 / sqrt(N - 1)`` across ``N`` companies (or the measured spread of the IC, ``ic_sd``, when
    it is known). The mean over ``T`` dates has that spread over ``sqrt(T_eff)``, where ``T_eff`` is the number of dates whose forward windows do not overlap (a horizon of ``h`` days sampled every
    ``step`` days shares its window with the ``ceil(h / step) - 1`` dates around it). The smallest mean IC that reaches ``|t| > z(1 - alpha/2)`` with probability ``power`` is
    ``(z(1 - alpha/2) + z(power)) * se``; and a true mean IC of ``ic`` is expected to show a t-statistic of ``ic / se``.
    """
    n, t = float(n_companies), int(n_dates)
    if not (n > 3 and t >= 2):
        return {"available": False, "reason": "too few companies or dates to say what could be detected"}
    overlap = max(int(math.ceil(horizon / max(step, 1))), 1)
    effective = max(t / overlap, 1.0)
    sd = float(ic_sd) if ic_sd is not None and np.isfinite(ic_sd) and ic_sd > 0 else 1.0 / math.sqrt(n - 1.0)
    se = sd / math.sqrt(effective)
    z_a, z_b = float(stats.norm.ppf(1.0 - alpha / 2.0)), float(stats.norm.ppf(power))
    mde = (z_a + z_b) * se
    table = []
    for ic in TYPICAL_IC:
        expected = ic / se
        table.append({"ic": ic, "expected_t": float(expected), "power_t2": float(stats.norm.cdf(expected - z_a) + stats.norm.cdf(-expected - z_a)),
                      "power_t3": float(stats.norm.cdf(expected - 3.0) + stats.norm.cdf(-expected - 3.0))})
    if mde <= 0.025:
        reading = f"With about {n:.0f} companies on each of {t} dates this study can tell a mean IC of {mde:.3f} or more from luck, which covers the effects practitioners treat as useful (about 0.02 to 0.05)."
    elif mde <= 0.05:
        reading = (f"With about {n:.0f} companies on each of {t} dates this study can reliably see a mean IC of {mde:.3f} or more. A signal with a true IC of 0.02 would show a t-statistic of about "
                   f"{0.02 / se:.1f} on average, so a real effect of that size could easily be missed here, and a result near the threshold is as likely to be luck.")
    else:
        reading = (f"With about {n:.0f} companies on each of {t} dates this study can only see a mean IC of {mde:.3f} or more. Practitioners treat 0.02 to 0.05 as a useful signal, so it cannot "
                   "confirm one: a result this small would be luck, and a real effect of that size would not show. Add companies (each one helps) or history before trusting any result.")
    return {"available": True, "companies": n, "dates": t, "effective_dates": float(effective), "null_sd_per_date": sd, "se_of_mean_ic": float(se), "mde_ic": float(mde), "alpha": alpha, "power": power,
            "table": table, "reading": reading, "adequate": bool(mde <= 0.05)}


def control_panels(ctx: ResearchContext, names) -> dict:
    """The controls of a premium regression as dates-by-companies panels known on each date: ``size`` (the log of market capitalisation in US dollars), ``beta``, ``volatility`` (three months of
    daily returns), ``momentum`` (twelve months less the last one), or any factor of the library by name (known as of the date like every other panel)."""
    out = {}
    for name in names:
        if name == "size":
            cap = ctx.market_cap_usd()
            out[name] = np.log(cap.where(cap > 0))
        elif name == "beta":
            out[name] = ctx.beta(252)
        elif name == "volatility":
            out[name] = ctx.volatility(63)
        elif name == "momentum":
            out[name] = ctx.past(252, 21)
        elif name in factor_library.names():
            out[name] = factor_library.compute(ctx, name)
        else:
            raise FundamentalsError(f"unknown control {name!r}: size, beta, volatility, momentum or the name of a library factor")
    return out


def _z(x: np.ndarray) -> np.ndarray:
    """Each column on a common scale: the normal score of its ranks (the standard-normal value at the rank's place), so that a few extreme values of a skewed ratio cannot decide a slope. A column
    that takes one value everywhere has no scale and comes out as NaN."""
    n = x.shape[0]
    ranks = np.column_stack([stats.rankdata(x[:, j]) for j in range(x.shape[1])])
    scores = stats.norm.ppf((ranks - 0.5) / n)
    flat = np.ptp(x, axis=0) <= 0
    scores[:, flat] = np.nan
    return scores


def fama_macbeth(ctx: ResearchContext, signal: pd.DataFrame, forward: pd.DataFrame, controls: dict, horizon: int, step: int, min_assets: int = 20) -> dict:
    """The premium a signal earns, from a cross-sectional regression on every date (Fama and MacBeth, 1973).

    On each date the forward returns (cut at the 1st and 99th percentile of the date) are regressed on the signal and the controls, each put on a common scale across the companies of that date
    (the normal score of its rank, which is the scale the rank IC measures on and which a few extreme values of a skewed ratio cannot distort). The coefficient of the signal is the return, over
    the forward window, of one standard deviation more of the signal *holding the controls fixed*. Its mean over
    the dates is the premium and its Newey-West error allows for the overlap of the windows. It is also run without the controls, so the two can be compared: a premium that vanishes once size,
    beta, volatility and momentum are held fixed was one of those in disguise.
    """
    mask = ctx.investable().reindex_like(signal).fillna(False).to_numpy(bool)
    sig, fwd = signal.to_numpy(float), forward.reindex_like(signal).to_numpy(float)
    ctl = {k: v.reindex_like(signal).to_numpy(float) for k, v in controls.items()}
    k = len(ctl)
    plain, full, r2s, counts = [], [], [], []
    for i in range(len(signal)):
        ok = mask[i] & np.isfinite(sig[i]) & np.isfinite(fwd[i])
        for panel in ctl.values():
            ok &= np.isfinite(panel[i])
        if ok.sum() < max(min_assets, k + 12):
            plain.append(np.nan)
            full.append(np.nan)
            continue
        y = fwd[i][ok]
        lo, hi = np.percentile(y, [1, 99])
        y = np.clip(y, lo, hi)
        columns = [sig[i][ok]] + [panel[i][ok] for panel in ctl.values()]
        z = _z(np.column_stack(columns))
        if not np.isfinite(z).all():
            plain.append(np.nan)
            full.append(np.nan)
            continue
        ones = np.ones(len(y))
        b1 = np.linalg.lstsq(np.column_stack([ones, z[:, 0]]), y, rcond=None)[0]
        design = np.column_stack([ones, z])
        b2, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ b2
        plain.append(float(b1[1]))
        full.append(float(b2[1]))
        r2s.append(1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum()) if ((y - y.mean()) ** 2).sum() > 0 else float("nan"))
        counts.append(int(ok.sum()))
    index = signal.index

    def summary(values) -> dict:
        s = pd.Series(values, index=index).dropna()
        if len(s) < 12:
            return {"n_dates": int(len(s))}
        lags = max(overlap_lags(horizon, step), int(np.floor(4.0 * (len(s) / 100.0) ** (2.0 / 9.0))))
        nw = newey_west(s, lags)
        return {"premium": nw["mean"], "se": nw["se"], "t_stat": nw["t"], "n_dates": int(len(s)), "positive_share": float((s > 0).mean()),
                "ci95": [nw["mean"] - 1.96 * nw["se"], nw["mean"] + 1.96 * nw["se"]] if np.isfinite(nw["se"]) else [float("nan"), float("nan")]}
    with_controls, without = summary(full), summary(plain)
    explained = summary([a - b for a, b in zip(plain, full)])                       # per date, what holding the controls fixed took away: its own error says whether the two versions really differ
    out = {"controls": list(controls), "with_controls": with_controls, "without_controls": without, "explained_by_controls": explained, "average_r2": float(np.nanmean(r2s)) if r2s else float("nan"),
           "average_companies": float(np.mean(counts)) if counts else float("nan"), "series": series_points(pd.Series(full, index=index).dropna())}
    out["reading"] = premium_reading(out)
    return out


def premium_reading(fm: dict) -> str:
    """What the premium regression says, in words. The two versions (with and without the controls) are compared through their per-date difference and its error, never by whether each one is
    "significant": a premium that is clear on its own and not clear with the controls may differ from it by no more than noise."""
    a, b, gap = fm.get("with_controls", {}), fm.get("without_controls", {}), fm.get("explained_by_controls", {})
    if "premium" not in a or "premium" not in b:
        return "Too few dates to run the regression."
    controls = list(fm["controls"])
    names = ", ".join(controls[:-1]) + (" and " + controls[-1] if len(controls) > 1 else (controls[0] if controls else ""))
    held = f" holding {names} fixed" if controls else ""
    text = f"One standard deviation more of the signal earned {a['premium'] * 100:+.2f}% over the forward window{held} (t = {a['t_stat']:.1f})"
    if not controls:
        return text + "."
    text += f"; on its own, {b['premium'] * 100:+.2f}% (t = {b['t_stat']:.1f})."
    ta, tb, td = a.get("t_stat"), b.get("t_stat"), gap.get("t_stat")
    if ta is None or tb is None or not (np.isfinite(ta) and np.isfinite(tb)):
        return text
    share = gap["premium"] / b["premium"] if gap.get("premium") is not None and b["premium"] else float("nan")
    different = td is not None and np.isfinite(td) and abs(td) >= 2.0
    if abs(ta) >= 2.0:
        if different and np.isfinite(share) and share > 0.25:
            return text + f" The premium is still there with the controls held fixed, but they account for about {share:.0%} of it (the difference has t = {td:.1f})."
        return text + " The premium survives the controls."
    if abs(tb) >= 2.0:
        if different:
            return text + f" The premium does not survive the controls: they account for about {share:.0%} of it (the difference has t = {td:.1f})."
        return text + (f" With the controls the premium can no longer be told from luck, but the two versions differ by no more than noise (t = {td:.1f}), so the data cannot say whether the controls "
                       "explain it.") if td is not None and np.isfinite(td) else " With the controls the premium can no longer be told from luck."
    return text + " Neither version can be told from luck."


# ------------------------------------------------------------------------------------------------------------------ the study
def evaluate(ctx: ResearchContext, signal: pd.DataFrame, spec: EvalSpec | None = None, label: str = "signal") -> dict:
    """Evaluate a ready signal panel. See the module docstring for what is measured; the result is plain JSON (numbers, strings, lists)."""
    spec = spec or EvalSpec()
    h = horizon_days(spec.horizon)
    step = FREQ_DAYS[ctx.freq]
    ppy = PERIODS_PER_YEAR[ctx.freq]
    window = pd.Series(True, index=signal.index)
    if spec.start:
        window &= signal.index >= pd.Timestamp(spec.start)
    if spec.end:
        window &= signal.index <= pd.Timestamp(spec.end)
    signal = signal.loc[window.to_numpy()]
    fwd = ctx.forward(spec.horizon).reindex(signal.index)
    warnings: list[str] = []
    ic = rank_ic(signal, fwd, spec.min_assets)
    if len(ic) < MIN_DATES:
        warnings.append(f"only {len(ic)} dates have enough companies to measure an IC; the statistics below are unreliable")
    ic_pearson = pearson_ic(signal, fwd, spec.min_assets)
    stats_ic = ic_stats(ic, h, step, ppy)
    stats_ic["pearson_mean_ic"] = float(ic_pearson.mean()) if len(ic_pearson) else float("nan")
    both = (signal.notna() & fwd.reindex_like(signal).notna()).sum(axis=1)
    usable = both[both >= spec.min_assets]
    power = detectable_effect(float(usable.median()) if len(usable) else 0.0, len(ic), h, step, stats_ic.get("std_ic"))
    if power.get("available") and not power["adequate"]:
        warnings.append(f"this study is small: it can only tell a mean IC of {power['mde_ic']:.3f} or more from luck, so it cannot confirm a signal of the usual size (0.02 to 0.05)")
    q = int(spec.quantiles)
    qret = quantile_returns(signal, fwd, q)
    qmean = qret.mean()
    quant = {"buckets": q, "mean_forward_return": {str(k): float(v) for k, v in qmean.items()}, "spread_top_minus_bottom": float(qmean.iloc[-1] - qmean.iloc[0]) if q > 1 else float("nan"),
             "monotonicity": monotonicity(qmean), "horizon": spec.horizon}
    populated = int((qmean.notna()).sum())
    if populated < q:
        warnings.append(f"only {populated} of {q} buckets are ever populated: the signal has too few distinct values to split that finely")
    ls = long_short(signal, ctx, spec)
    stats_gross, stats_net = portfolio_stats(ls["gross"], ppy), portfolio_stats(ls["net"], ppy)
    long_only = (ls["long"] - holding_returns(ctx).reindex_like(signal).where(ctx.investable().reindex_like(signal)).mean(axis=1)).where(ls["net"].notna())
    portfolio = {"top_fraction": ls["frac"], "weighting": spec.weighting, "cost_bps": spec.cost_bps, "borrow_bps": spec.borrow_bps, "gross": stats_gross, "net": stats_net,
                 "avg_turnover": float(ls["turnover"].mean()) if ls["turnover"].notna().any() else float("nan"), "long_leg_mean": float(ls["long"].mean()), "short_leg_mean": float(ls["short"].mean()),
                 "long_minus_universe_mean": float(long_only.mean()) if long_only.notna().any() else float("nan"), "periods_per_year": ppy}
    gross_sharpe, net_sharpe = stats_gross.get("sharpe"), stats_net.get("sharpe")
    if gross_sharpe is not None and net_sharpe is not None and np.isfinite(gross_sharpe) and np.isfinite(net_sharpe):
        portfolio["cost_drag_sharpe"] = float(gross_sharpe - net_sharpe)
    decay = []
    for hz in DECAY_HORIZONS:
        days = HORIZONS[hz]
        s = ic_stats(rank_ic(signal, ctx.forward(days).reindex(signal.index), spec.min_assets), days, step, ppy)
        decay.append({"horizon": hz, "days": days, "mean_ic": s.get("mean_ic"), "t_stat": s.get("t_stat"), "n_dates": s.get("n_dates", 0)})
    coverage = signal.where(ctx.investable().reindex_like(signal)).notna().sum(axis=1)
    investable = ctx.investable().reindex_like(signal).sum(axis=1)
    cover_share = (coverage / investable.replace(0, np.nan)).dropna()
    if len(cover_share) and float(cover_share.mean()) < 0.5:
        warnings.append(f"the signal exists for only {cover_share.mean():.0%} of the investable companies on average: the result describes a subset")
    breakdown: dict = {}
    if "country" in spec.breakdowns:
        breakdown["country"] = _group_ic(signal, fwd, ctx.group("country"), h, step, ppy)
    if "sector" in spec.breakdowns:
        breakdown["sector"] = _group_ic(signal, fwd, ctx.group("sector"), h, step, ppy)
    if "region" in spec.breakdowns:
        breakdown["region"] = _group_ic(signal, fwd, ctx.group("region"), h, step, ppy, 15)
    if "regime" in spec.breakdowns:
        reg = regimes(ctx).reindex(signal.index)
        breakdown["regime"] = {"trend": _regime_ic(ic, reg["trend"], h, step, ppy), "volatility": _regime_ic(ic, reg["vol"], h, step, ppy)}
    years = ic.groupby(ic.index.year).mean()
    half = len(ic) // 2
    stability = {"by_year": {str(k): float(v) for k, v in years.items()}, "positive_years": float((years > 0).mean()) if len(years) else float("nan"),
                 "first_half_ic": float(ic.iloc[:half].mean()) if half else float("nan"), "second_half_ic": float(ic.iloc[half:].mean()) if half else float("nan")}
    stability["halves_agree"] = bool(np.sign(stability["first_half_ic"]) == np.sign(stability["second_half_ic"])) if half and np.isfinite(stability["first_half_ic"]) and np.isfinite(stability["second_half_ic"]) else None
    distinct = signal.nunique(axis=1)
    typical = float(distinct[distinct > 0].median()) if (distinct > 0).any() else float("nan")
    if np.isfinite(typical) and typical < 15:
        warnings.append(f"the signal takes only {int(typical)} distinct values on a typical date, so ranks have many ties and the buckets are uneven")
    surv = ctx.survivorship()
    if not surv["unbiased"]:
        warnings.append("survivorship: " + surv["verdict"])
    charts = {
        "ic": series_points(ic), "ic_rolling": series_points(ic.rolling(12, min_periods=6).mean()), "ic_hist": _hist(ic),
        "quantile_mean": [{"bucket": int(k), "return": float(v)} for k, v in qmean.items() if np.isfinite(v)],
        "quantile_cumulative": {str(k): series_points((1.0 + holding_returns(ctx).reindex_like(signal).where(buckets(signal, q) == k).mean(axis=1).fillna(0.0)).cumprod()) for k in range(1, q + 1)},
        "ls_cumulative_gross": series_points((1.0 + ls["gross"].fillna(0.0)).cumprod()), "ls_cumulative_net": series_points((1.0 + ls["net"].fillna(0.0)).cumprod()),
        "ls_drawdown": series_points(_drawdown(ls["net"])), "turnover": series_points(ls["turnover"]), "decay": decay, "coverage": series_points(cover_share),
    }
    premium = None
    if spec.premium:
        try:
            premium = fama_macbeth(ctx, signal, fwd, control_panels(ctx, [c for c in spec.controls if c != spec.factor]), h, step, spec.min_assets)
        except (FundamentalsError, ValueError, np.linalg.LinAlgError, KeyError) as error:
            warnings.append(f"the premium regression could not be run: {error}")
    result = {"label": label, "spec": spec.describe(), "universe": ctx.describe(), "ic": stats_ic, "quantiles": quant, "portfolio": portfolio, "decay": decay, "breakdown": breakdown,
              "stability": stability, "warnings": warnings, "charts": charts, "n_dates": int(len(signal)), "average_companies": float(coverage.mean()) if len(coverage) else 0.0,
              "power": power, "premium": premium}
    result["verdict"] = verdict(result)
    return clean(result)


def _hist(ic: pd.Series, bins: int = 20) -> dict:
    v = ic.dropna().to_numpy(float)
    if len(v) < 5:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(v, bins=bins)
    return {"edges": [round(float(e), 5) for e in edges], "counts": [int(c) for c in counts]}


def _drawdown(returns: pd.Series) -> pd.Series:
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    return curve / curve.cummax() - 1.0


def verdict(result: dict) -> str:
    """The sentence under the numbers. Never "validated": that needs out-of-sample data, a count of the ideas tried and costs."""
    ic = result["ic"]
    n, t = ic.get("n_dates", 0), ic.get("t_stat")
    if n < MIN_DATES or t is None or not np.isfinite(t):
        return "Too little data to judge: fewer than two years of dates with enough companies."
    mean = ic.get("mean_ic") or 0.0
    net = result["portfolio"].get("net", {}).get("sharpe")
    mono = result["quantiles"]["monotonicity"].get("spearman")
    direction = "in the expected direction" if mean > 0 else "AGAINST the direction given"
    if abs(t) < 1.65:
        return f"No evidence of a relationship in this sample (mean IC {mean:+.3f}, t = {t:.1f}). Treat it as noise."
    if abs(t) < 3.0:
        return (f"Some evidence, {direction} (mean IC {mean:+.3f}, t = {t:.1f}). With the number of ideas usually tried, a t below 3 is weak: this is interesting, not validated. "
                "It needs out-of-sample data and a count of everything tried.")
    cost_note = "" if net is None or not np.isfinite(net) else (f" The long-short portfolio has a net Sharpe of {net:.2f}." if net > 0 else f" But the long-short portfolio loses money after costs (net Sharpe {net:.2f}).")
    mono_note = "" if mono is None or not np.isfinite(mono) else (" The buckets rise in order." if mono > 0.8 else (" The buckets do not rise cleanly, which is a warning." if mono < 0.5 else ""))
    return (f"Strong in-sample evidence, {direction} (mean IC {mean:+.3f}, t = {t:.1f}).{cost_note}{mono_note} Interesting is not validated: check it out of sample, against the number of ideas "
            "tried, and after costs before trusting it.")


# ------------------------------------------------------------------------------------------------------------------ from a library factor
def build_signal(ctx: ResearchContext, spec: EvalSpec) -> pd.DataFrame:
    """The signal panel a library factor and a spec describe: raw values, the pipeline of transformations, then the sign."""
    if not spec.factor:
        raise FundamentalsError("name a factor to evaluate")
    f = factor_library.get(spec.factor)
    raw = factor_library.compute(ctx, spec.factor)
    out = transforms.apply(raw, spec.steps, ctx)
    sign = f.direction if spec.direction == "theory" else (1 if spec.direction == "auto" else int(spec.direction))
    return out * (1.0 if sign >= 0 else -1.0)


def evaluate_factor(ctx: ResearchContext, spec: EvalSpec) -> dict:
    """Evaluate a library factor. With ``direction='auto'`` the sign that earned is used and the result says so (it is a choice made on the data)."""
    spec = spec if isinstance(spec, EvalSpec) else EvalSpec.from_dict(spec)
    signal = build_signal(ctx, spec)
    notes = []
    if spec.direction == "auto":
        ic = rank_ic(signal, ctx.forward(spec.horizon).reindex(signal.index), spec.min_assets)
        if len(ic) and ic.mean() < 0:
            signal = -signal
            notes.append("direction 'auto': the sign was flipped because the in-sample IC was negative. That is a choice made on the data; count it as a second trial.")
        else:
            notes.append("direction 'auto': kept the factor's own sign. Choosing the sign that worked is a choice made on the data; count it as a second trial.")
    result = evaluate(ctx, signal, spec, label=spec.factor)
    result["warnings"] = notes + list(result["warnings"])
    f = factor_library.get(spec.factor)
    result["factor"] = {"name": f.name, "label": f.label, "family": f.family, "direction": f.direction, "needs": list(f.needs), "text": f.text}
    return result


# ------------------------------------------------------------------------------------------------------------------ the look-ahead audit
LAG_SWEEP = (1, 30, 60, 90)


def look_ahead_audit(store, spec: EvalSpec, universe=None, freq: str = "M", settings=None, lags: tuple = LAG_SWEEP) -> dict:
    """The same study twice: with every figure available from the day it was public (the honest way) and with every figure treated as known on its period end. If the second is much better
    than the first, the result leans on knowing figures before anyone could have, and the honest number is the one to believe.

    It also waits: the study is repeated with each figure used only ``lags`` days after its filing (``by_lag``). A signal whose information has gone within a month of the filing was never
    for a slow investor, and one that is still there after three months is not about being first to read the filing. Where filing dates are estimated and not known, a long wait is also the safe
    reading of how stale a figure may be."""
    from dataclasses import replace

    from .pit import PitSettings
    base = settings or PitSettings()
    out, panels = {}, {}
    for name, s in (("honest", base), ("naive", replace(base, naive_availability=True))):
        ctx = ResearchContext(store, universe, freq=freq, settings=s)
        signal = build_signal(ctx, spec)
        h = horizon_days(spec.horizon)
        ic = rank_ic(signal, ctx.forward(spec.horizon).reindex(signal.index), spec.min_assets)
        out[name] = ic_stats(ic, h, FREQ_DAYS[freq], PERIODS_PER_YEAR[freq])
        panels[name] = ic
    by_lag = []
    for lag in sorted({int(x) for x in lags}):
        if lag == base.info_lag_days:
            stats_lag = out["honest"]
        else:
            ctx = ResearchContext(store, universe, freq=freq, settings=replace(base, info_lag_days=lag))
            signal = build_signal(ctx, spec)
            stats_lag = ic_stats(rank_ic(signal, ctx.forward(spec.horizon).reindex(signal.index), spec.min_assets), horizon_days(spec.horizon), FREQ_DAYS[freq], PERIODS_PER_YEAR[freq])
        by_lag.append({"lag_days": lag, "mean_ic": stats_lag.get("mean_ic"), "t_stat": stats_lag.get("t_stat"), "n_dates": stats_lag.get("n_dates", 0)})
    gap = (out["naive"].get("mean_ic") or 0.0) - (out["honest"].get("mean_ic") or 0.0)
    honest_ic = out["honest"].get("mean_ic") or 0.0
    flattered = abs(honest_ic) > 1e-12 and gap * np.sign(honest_ic) > 0.5 * abs(honest_ic) or (abs(honest_ic) < 0.01 and abs(out["naive"].get("mean_ic") or 0.0) > 0.03)
    waits = [r for r in by_lag if r["lag_days"] >= 30 and r["mean_ic"] is not None and np.isfinite(r["mean_ic"])]
    first = next((r for r in by_lag if r["lag_days"] == base.info_lag_days and r["mean_ic"] is not None), None) or (by_lag[0] if by_lag else None)
    fades = bool(waits and first and first["mean_ic"] and np.isfinite(first["mean_ic"]) and abs(first["mean_ic"]) > 0.01 and waits[0]["mean_ic"] * np.sign(first["mean_ic"]) < 0.5 * abs(first["mean_ic"]))
    return clean({"factor": spec.factor, "honest": out["honest"], "naive": out["naive"], "naive_minus_honest_ic": gap, "flattered": bool(flattered), "by_lag": by_lag, "fades_when_waiting": fades,
                  "verdict": ("Treating figures as known on their period end flatters this factor a great deal: the honest result is the one to believe." if flattered else
                              "Treating figures as known earlier changes little: the result does not lean on early knowledge."),
                  "lag_verdict": ("The IC falls to less than half within a month of waiting: the signal lives in the days right after a filing, so it is only as good as how fast it can be acted on." if fades
                                  else "Waiting a month or more after the filing leaves most of the IC in place: the signal is not about being first to read a filing.") if by_lag else ""})
