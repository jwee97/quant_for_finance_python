"""The studies the Alpha Lab page runs: each takes prices or returns and returns plain numbers, lists and dictionaries (JSON-ready), so the web layer only loads data and calls them.

Signals here are built from prices alone, because every universe has prices. They are examples of the interface, not recommendations: a signal registered from a fundamentals file or a model goes through the
same functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..portfolio.covariance import estimate_covariance
from ..riskengine import decompose as dc
from ..riskengine import factor_model as fm
from ..riskengine import monitor as mo
from . import breadth as br
from . import calibrate as cal
from . import dynamic as dy
from . import ic as icm

SIGNALS = {
    "momentum": ("Momentum (12 months, skipping the last month)", "The return from 252 to 21 days ago: what has gone up keeps going up, until it reverses."),
    "reversal": ("Short-term reversal (1 month)", "Minus the last month's return: what fell last month rebounds."),
    "low_vol": ("Low volatility (3 months)", "Minus the 63-day volatility: calmer names are expected to do relatively better than their risk suggests."),
    "trend": ("Trend (price over 200-day average)", "How far the price is above its 200-day average."),
}


class StudyError(ValueError):
    pass


def build_signal(prices: pd.DataFrame, kind: str) -> pd.DataFrame:
    """A signal panel from prices, known at the close of each date."""
    if kind not in SIGNALS:
        raise StudyError(f"signal must be one of {', '.join(SIGNALS)}")
    r = prices.pct_change()
    if kind == "momentum":
        return prices.shift(21) / prices.shift(252) - 1.0
    if kind == "reversal":
        return -(prices / prices.shift(21) - 1.0)
    if kind == "low_vol":
        return -r.rolling(63, min_periods=40).std()
    return prices / prices.rolling(200, min_periods=150).mean() - 1.0


def _series(s: pd.Series, step: int = 1, digits: int = 5) -> list:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in s.iloc[::step].to_numpy(dtype=float)]


def _dates(idx, step: int = 1) -> list:
    return [str(d.date()) for d in idx[::step]]


def _frame_rows(df: pd.DataFrame, digits: int = 5) -> list:
    out = []
    for k, row in df.iterrows():
        d = {"index": str(k)}
        for c, v in row.items():
            d[str(c)] = None if (isinstance(v, float) and not np.isfinite(v)) else (round(float(v), digits) if isinstance(v, (float, np.floating, int, np.integer)) else v)
        out.append(d)
    return out


def alpha_study(prices: pd.DataFrame, signal: str = "momentum", horizon: int = 5, window: int = 252, min_obs: int = 60, lags=(1, 2, 3, 5, 10, 21, 42, 63), cov_lookback: int = 252) -> dict:
    """IC statistics, decay and the information horizon, calibration to expected residual return with its reliability, breadth, and what a long-only portfolio keeps of the latest alpha."""
    if prices.shape[1] < 6:
        raise StudyError("an alpha study ranks assets against each other: choose at least 6 tickers")
    if not 1 <= horizon <= 63:
        raise StudyError("horizon must be 1 to 63 days")
    returns = prices.pct_change().iloc[1:]
    sig = build_signal(prices, signal).reindex(returns.index)
    if returns.shape[0] < 500:
        raise StudyError("need at least 500 days of history")
    z = cal.standardise(sig)
    fwd = cal.forward_residual(returns, horizon)
    ic = icm.ic_series(z, fwd)
    summary = icm.ic_summary(ic, horizon)
    curve = icm.lagged_ic(z, returns, lags=[k for k in lags if k < len(returns) // 4])
    fit = icm.fit_decay(curve)
    forecast = cal.calibrate(sig, returns, horizon=horizon, window=window, min_obs=min_obs)
    rel = cal.reliability(forecast.alpha, fwd)
    resid = cal.residual_returns(returns).dropna(how="all")
    b = br.effective_breadth(resid.dropna(how="any").iloc[-756:] if resid.dropna(how="any").shape[0] > 100 else resid.fillna(0.0), z, horizon)
    latest = forecast.alpha.dropna(how="all").iloc[-1] if forecast.alpha.notna().any().any() else None
    tc = None
    if latest is not None:
        cols = [c for c in latest.dropna().index]
        cov = estimate_covariance(returns[cols].tail(cov_lookback).fillna(0.0), "shrinkage", cov_lookback, annualise=False)
        # alpha is over `horizon` days: put the covariance on the same horizon
        c = constrained = br.constrained_transfer(latest[cols].to_numpy(dtype=float), cov.to_numpy() * horizon, lb=0.0, risk_aversion=1.0)
        tc = {"long_only_tc": c["tc"], "ir_unconstrained": c["ir_unconstrained"], "ir_long_only": c["ir_constrained"], "ok": c["ok"],
              "assets": [{"asset": a, "alpha": float(latest[a]), "long_only_weight": float(constrained["holdings"][i])} for i, a in enumerate(cols)]}
    roll = icm.rolling_ic(ic, 126)
    step = max(1, len(ic) // 400)
    ic_cum = ic.fillna(0.0).cumsum()
    law = None
    if np.isfinite(summary["mean"]) and np.isfinite(b["breadth"]):
        law = {"ic": summary["mean"], "breadth": b["breadth"], "ir_at_tc_1": br.fundamental_law(summary["mean"], b["breadth"], 1.0),
               "ir_at_tc": br.fundamental_law(summary["mean"], b["breadth"], tc["long_only_tc"] if tc and np.isfinite(tc["long_only_tc"]) else 1.0)}
    return {"signal": signal, "label": SIGNALS[signal][0], "horizon": int(horizon), "assets": list(prices.columns), "days": int(len(returns)), "first": str(returns.index[0].date()), "last": str(returns.index[-1].date()),
            "ic": summary, "ic_series": {"dates": _dates(ic.index, step), "cumulative": _series(ic_cum, step, 4), "rolling": _series(roll["mean"], step, 4)},
            "decay": {"table": _frame_rows(curve, 5), "fit": {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in fit.items() if k != "fitted"}, "fitted": [] if fit["fitted"].empty else _frame_rows(fit["fitted"].to_frame("ic"), 5)},
            "calibration": {"slope": rel["slope"], "se": rel["se"], "t_vs_one": rel["t_vs_one"], "n": rel["n"], "verdict": rel["verdict"], "bins": [] if rel["bins"].empty else _frame_rows(rel["bins"], 6)},
            "ic_hat": {"dates": _dates(forecast.ic_hat.dropna().index, max(1, len(forecast.ic_hat.dropna()) // 300)), "values": _series(forecast.ic_hat.dropna(), max(1, len(forecast.ic_hat.dropna()) // 300), 4)},
            "breadth": b, "law": law, "transfer": tc,
            "notes": ["The IC is a rank correlation across assets on each date; forward returns are residuals after the trailing beta to the equal-weighted market.",
                      "The calibrated alpha uses only ICs whose forward windows had ended, shrunk toward zero by the signal-to-noise of their own average.",
                      "Breadth is an estimate: independent bets in the cross-section times non-overlapping horizons a year times the share of the signal that is new at each one.",
                      "A signal built from prices alone is an example of the interface, not a recommendation."]}


def risk_study(returns: pd.DataFrame, kind: str = "statistical", n_factors: int | None = None, weights: pd.Series | None = None, lookback: int = 504, factor_cov_halflife: float | None = 126) -> dict:
    """A factor model of the universe, the risk of a portfolio (equal weight by default) by factor, whether the model's risk forecast has held, and what neutralising the largest factor does."""
    data = returns.dropna(how="any")
    if data.shape[1] < 4:
        raise StudyError("a risk model needs at least 4 assets with a common history")
    if len(data) < lookback // 2 + 100:
        raise StudyError(f"need at least {lookback // 2 + 100} common days")
    w = (weights if weights is not None else pd.Series(1.0 / data.shape[1], index=data.columns)).reindex(data.columns).fillna(0.0)
    window = data.tail(lookback)
    if kind == "statistical":
        model = fm.fit_statistical(window, n_factors, halflife=factor_cov_halflife)
    elif kind == "time_series":
        market = data.mean(axis=1).rename("market")
        model = fm.fit_time_series(window, market.loc[window.index].to_frame(), halflife=factor_cov_halflife, groups={"market": "market"})
    else:
        raise StudyError("kind must be statistical or time_series")
    d = dc.risk_decomposition(w, model, annualise=252)
    # the bias statistic: forecast volatility of the portfolio from a trailing covariance, lagged one day, against the day's return
    pr = (data * w.to_numpy()).sum(axis=1)
    fc = pd.Series(np.nan, index=data.index)
    for t in range(252, len(data), 21):
        cov = data.iloc[t - 252:t].cov().to_numpy()
        fc.iloc[t:t + 21] = np.sqrt(max(w.to_numpy() @ cov @ w.to_numpy(), 0.0))
    bias = mo.bias_statistic(pr.iloc[252:], fc.iloc[252:], window=126)
    step = max(1, len(bias["series"]) // 300)
    top = d["factors"].abs().idxmax()
    neutral = None
    if len(model.factors) > 1 or kind == "time_series":
        try:
            from ..riskengine import hedge as hg

            n = hg.neutralise(w, model, [top])
            neutral = {"factor": str(top), "exposure_before": float(n["exposures_before"][top]), "exposure_after": float(n["exposures_after"][top]), "volatility_before": n["volatility_before"] * np.sqrt(252),
                       "volatility_after": n["volatility_after"] * np.sqrt(252), "turnover": n["turnover"]}
        except (ValueError, np.linalg.LinAlgError):
            neutral = None
    corr = window.corr()
    ev = np.sort(np.linalg.eigvalsh(corr.to_numpy()))[::-1]
    from ..features.pca import marchenko_pastur_bounds

    _, upper = marchenko_pastur_bounds(len(window), window.shape[1])
    contributions = [{"factor": str(k), "group": model.groups.get(k, "other"), "exposure": float(d["exposures"][k]), "contribution": float(d["factors"][k])} for k in model.factors] + [{"factor": "specific", "group": "specific", "exposure": None, "contribution": float(d["specific"])}]
    return {"kind": kind, "assets": list(data.columns), "window": {"first": str(window.index[0].date()), "last": str(window.index[-1].date()), "days": int(len(window))}, "model": model.summary(),
            "risk": {"volatility": d["volatility"], "factor_share": d["factor_share"], "specific_share": d["specific_share"], "groups": {k: float(v) for k, v in d["groups"].items()}},
            "contributions": contributions, "assets_risk": [{"asset": a, "weight": float(w[a]), "contribution": float(d["assets"][a]), "marginal": float(d["marginal"][a])} for a in data.columns],
            "bias": {"latest": bias["latest"], "full": bias["full"], "verdict": bias["verdict"], "dates": _dates(bias["series"].index, step), "values": _series(bias["series"], step, 4), "low": _series(bias["low"].reindex(bias["series"].index), step, 4), "high": _series(bias["high"].reindex(bias["series"].index), step, 4)},
            "neutral": neutral, "spectrum": {"eigenvalues": [round(float(x), 4) for x in ev[:20]], "noise_bound": float(upper), "above_noise": int((ev > upper).sum())},
            "notes": ["Factors are the leading principal components of the returns (statistical) or the equal-weighted market (time series); a fundamental model needs exposures such as sector and value, which the Fundamentals data supplies in the library (src.riskengine.factor_model.fit_cross_sectional).",
                      "Contributions to volatility add up exactly to the portfolio's volatility.",
                      "The bias statistic is the standard deviation of the portfolio's daily return divided by the volatility forecast made the day before: 1 when the forecast is right, outside the band when it is not."]}


def trading_study(prices: pd.DataFrame, signal: str = "momentum", horizon: int = 5, cost_bps: float = 5.0, impact: float = 1e-3, target_gross: float = 1.0, lag: int = 1, warmup: int = 252, long_only: bool = True,
                  policies=("full", "partial", "band", "threshold"), max_days: int = 1500) -> dict:
    """Trade the calibrated alpha of a signal under each policy and show where the paper alpha goes."""
    if prices.shape[1] < 6:
        raise StudyError("choose at least 6 tickers")
    returns = prices.pct_change().iloc[1:]
    sig = build_signal(prices, signal).reindex(returns.index)
    f = cal.calibrate(sig, returns, horizon=horizon, window=252, min_obs=60)
    alpha = (f.alpha / float(horizon)).dropna(how="all").fillna(0.0)
    if len(alpha) < warmup + 250:
        raise StudyError("not enough history after the warm-up for a trading study")
    cov = estimate_covariance(returns.iloc[:warmup].fillna(0.0), "shrinkage", warmup, annualise=False)
    alpha, ret = alpha.iloc[warmup:].iloc[-max_days:], returns.loc[alpha.index].iloc[warmup:].iloc[-max_days:].fillna(0.0)
    unit = np.mean([np.abs(np.linalg.solve(cov.to_numpy(), a)).sum() for a in alpha.to_numpy()[::10]])
    gamma = float(max(unit / target_gross, 1e-6))
    lb = 0.0 if long_only else None
    table = dy.compare_policies(alpha, ret, cov, policies=policies, risk_aversion=gamma, cost_bps=cost_bps, impact=impact, lag=0, lb=lb)
    waterfall = dy.implementation_waterfall(alpha, ret, cov, risk_aversion=gamma, cost_bps=cost_bps, lag=lag, lb=lb, impact=impact)
    return {"signal": signal, "label": SIGNALS[signal][0], "horizon": int(horizon), "risk_aversion": gamma, "days": int(len(alpha)), "first": str(alpha.index[0].date()), "last": str(alpha.index[-1].date()),
            "policies": _frame_rows(table.rename_axis("policy"), 5), "waterfall": _frame_rows(waterfall.rename_axis("step"), 5),
            "config": {"cost_bps": cost_bps, "impact": impact, "target_gross": target_gross, "lag": lag, "long_only": long_only},
            "notes": ["The same calibrated alpha, covariance and costs go through every policy. The covariance is fixed from the first year; trading starts after it.",
                      "Full trades to the aim every day. Partial adjustment closes the stationary share of the gap (Garleanu-Pedersen). The band holds while the gradient is inside the proportional cost. Threshold trades only when far from the aim.",
                      "The waterfall re-runs the same forecasts with each real-world step added: position limits, trading costs, a policy that respects them, and one day's delay."]}


def factor_study(prices: pd.DataFrame, signals: list[str], horizon: int = 5, min_assets: int = 5) -> dict:
    """The validation and combination stages on price-built signals: each factor evaluated (rank and Pearson IC with a HAC interval, quantile spread, turnover, coverage, stability), the family corrected for
    multiple testing, and the factors combined four ways from the past only and judged on the last 40% of the dates. Forward returns are residuals after the equal-weighted market, as in :func:`alpha_study`."""
    from . import factors as fc

    if prices.shape[1] < 6:
        raise StudyError("a factor study ranks assets against each other: choose at least 6 tickers")
    if not 1 <= horizon <= 63:
        raise StudyError("horizon must be 1 to 63 days")
    if len(signals) < 2:
        raise StudyError("choose at least two signals to compare and combine")
    returns = prices.pct_change().iloc[1:]
    if returns.shape[0] < 500:
        raise StudyError("need at least 500 days of history")
    fwd = cal.forward_residual(returns, horizon)
    z = {k: cal.standardise(build_signal(prices, k).reindex(returns.index)) for k in signals}
    try:
        compared = fc.compare_factors(z, fwd, horizon, min_assets=min_assets, returns=returns)
        combined = fc.combine_factors(z, fwd, horizon, min_assets=min_assets)
    except fc.FactorError as error:
        raise StudyError(str(error)) from None
    compared.pop("details")
    return {"horizon": int(horizon), "assets": list(prices.columns), "days": int(len(returns)), "first": str(returns.index[0].date()), "last": str(returns.index[-1].date()), "comparison": compared, "combination": combined,
            "notes": ["Each factor's rank IC is against the residual return over the horizon; the interval is Newey-West with horizon - 1 lags.",
                      "p-values are corrected for the number of factors (Holm, Benjamini-Hochberg); a factor is called predictive only if it survives the correction.",
                      "Combinations are weighted from the past only and judged on a test period nothing was fitted on; a combination is an improvement only if it beats the best single factor and equal weight."]}
