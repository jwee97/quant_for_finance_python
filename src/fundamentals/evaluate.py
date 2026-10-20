"""Factor evaluation: does a signal, known on each date, order the companies by what they earn next?

Everything here takes a **signal panel** (dates by securities, higher meaning "expect more", already transformed and signed) and the research context, and measures it against what happened after:

* the **information coefficient** (IC) on every date, the rank correlation between the signal and the forward return across companies: its mean, median, spread, share of positive dates, the
  information ratio of the IC, and a Newey-West t-statistic that allows for overlapping horizons;
* **quantile analysis**: the average forward return of each bucket (quintiles or deciles) and whether the buckets rise in order (monotonicity);
* a **long-short portfolio** (top bucket against bottom bucket, held one rebalance period) with its return, volatility, Sharpe, Sortino, drawdown, Calmar, hit rate and turnover, gross and after costs;
* **decay**: the IC at one week, one, three, six and twelve months;
* **breakdowns** by country, sector, region and market regime (bull or bear, high or low volatility, each known at the start of the period it describes);
* **stability** by calendar year and in halves.

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

    def describe(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict | None) -> "EvalSpec":
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in EvalSpec.__dataclass_fields__}
        if "breakdowns" in known:
            known["breakdowns"] = tuple(known["breakdowns"])
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
            "t_stat": nw["t"], "p_value": nw["p"], "hac_lags": nw["lags"], "skew": float(stats.skew(ic, bias=False)) if len(ic) > 3 else float("nan")}


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
    return out


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
    result = {"label": label, "spec": spec.describe(), "universe": ctx.describe(), "ic": stats_ic, "quantiles": quant, "portfolio": portfolio, "decay": decay, "breakdown": breakdown,
              "stability": stability, "warnings": warnings, "charts": charts, "n_dates": int(len(signal)), "average_companies": float(coverage.mean()) if len(coverage) else 0.0}
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
def look_ahead_audit(store, spec: EvalSpec, universe=None, freq: str = "M", settings=None) -> dict:
    """The same study twice: with every figure available from the day it was public (the honest way) and with every figure treated as known on its period end. If the second is much better
    than the first, the result leans on knowing figures before anyone could have, and the honest number is the one to believe."""
    from dataclasses import replace

    from .pit import PitSettings
    base = settings or PitSettings()
    out = {}
    for name, s in (("honest", base), ("naive", replace(base, naive_availability=True))):
        ctx = ResearchContext(store, universe, freq=freq, settings=s)
        signal = build_signal(ctx, spec)
        h = horizon_days(spec.horizon)
        ic = rank_ic(signal, ctx.forward(spec.horizon).reindex(signal.index), spec.min_assets)
        out[name] = ic_stats(ic, h, FREQ_DAYS[freq], PERIODS_PER_YEAR[freq])
    gap = (out["naive"].get("mean_ic") or 0.0) - (out["honest"].get("mean_ic") or 0.0)
    honest_ic = out["honest"].get("mean_ic") or 0.0
    flattered = abs(honest_ic) > 1e-12 and gap * np.sign(honest_ic) > 0.5 * abs(honest_ic) or (abs(honest_ic) < 0.01 and abs(out["naive"].get("mean_ic") or 0.0) > 0.03)
    return clean({"factor": spec.factor, "honest": out["honest"], "naive": out["naive"], "naive_minus_honest_ic": gap, "flattered": bool(flattered),
                  "verdict": ("Treating figures as known on their period end flatters this factor a great deal: the honest result is the one to believe." if flattered else
                              "Treating figures as known earlier changes little: the result does not lean on early knowledge.")})
