"""Expected returns for the portfolio optimisers from four sources, every one using only what was known when the portfolio was set.

A provider is a function ``mu(p, data) -> Estimate`` called by :func:`src.portfolio.lab.compare` at each rebalance row ``p`` (the weights set there earn the returns of rows ``p`` onward and are decided from
rows before ``p``): it returns the horizon-``h`` expected simple return of each asset (``NaN`` where it has none), the spread of that estimate and a note about what it did. ``compare`` fills a missing asset with
the historical mean, counts it, and puts every source in the optimiser's units with the same scaling (:func:`src.forecast.alpha.to_optimiser_units`).

* ``historical``  the window's own mean return, the Lab's default; the baseline.
* ``factor``      Grinold's rule of thumb, ``mu_i = IC x sigma_i x z_i``: the factor's information coefficient measured on dates whose forward return was already complete, times the asset's volatility over the
                  horizon, times the factor's cross-sectional z-score on the last known day. With no usable IC the source has no opinion.
* ``forecast``    the forecasting methods' expected returns (:mod:`src.forecast.alpha`), each from prices up to the origin, converted explicitly, used only from the day after the forecast was made and only if not stale,
                  and shrunk by the slope of realised on forecast over forecasts whose outcome was already known (below one the forecasts overstate; with too few resolved forecasts the weight is zero).
* ``combined``    the factor and forecast estimates blended: equal weights, or weights proportional to the inverse of each source's squared error over rebalances whose outcome had been observed.

The same ``mu_shrink``, covariance, caps, costs and rebalance schedule apply to every source: only the expected returns differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..forecast import alpha as fa
from . import studies as st

MIN_IC_DATES = 60
MIN_PAIRS = 20


@dataclass
class Estimate:
    mu: pd.Series                         # horizon-h expected simple return per asset (NaN: no opinion)
    sd: pd.Series | None = None           # the uncertainty of that estimate, per asset
    info: dict = field(default_factory=dict)


def _z(row: pd.Series) -> pd.Series:
    s = row.std(ddof=0)
    return (row - row.mean()) / s if s and np.isfinite(s) and s > 0 else row * np.nan


def factor_provider(prices: pd.DataFrame, signal: str, horizon: int):
    """The factor-implied expected returns (see the module docstring). ``prices`` covers the same dates as the returns passed to ``compare`` plus the first one."""
    sig = st.build_signal(prices, signal)
    fwd = prices.shift(-horizon) / prices - 1.0                                      # the return from t to t+h: complete only at t+h
    ic = pd.Series(index=prices.index, dtype=float)
    for t in prices.index:
        s, f = sig.loc[t], fwd.loc[t]
        ok = s.notna() & f.notna()
        ic.loc[t] = s[ok].rank().corr(f[ok].rank()) if ok.sum() >= 3 else np.nan

    def mu(p: int, data: pd.DataFrame) -> Estimate:
        last = data.index[p - 1]                                                     # the last day known when the weights are set
        pos = prices.index.get_loc(last)
        done = ic.iloc[:max(pos - horizon + 1, 0)].dropna()                          # dates whose forward return was complete on or before ``last``
        if len(done) < MIN_IC_DATES:
            return Estimate(pd.Series(np.nan, index=data.columns), info={"note": f"too few completed dates for an information coefficient ({len(done)})"})
        ic_hat = float(done.mean())
        z = _z(sig.loc[last].reindex(data.columns))
        vol = data.iloc[max(p - 252, 0):p].std() * np.sqrt(horizon)
        est = ic_hat * vol * z
        return Estimate(est, (1 - ic_hat ** 2) ** 0.5 * vol, {"ic": ic_hat, "ic_dates": int(len(done))})
    return mu


def forecast_provider(records: list[fa.ForecastRecord], prices: pd.DataFrame, horizon: int, max_age_days: int = 10, calibrate: str = "past_slope", allow_scaling: bool = False, base_currency: str | None = None):
    def mu(p: int, data: pd.DataFrame) -> Estimate:
        on, last = data.index[p], data.index[p - 1]
        vals, sds, stale, rejected = {}, {}, 0, []
        for a in data.columns:
            pool = [r for r in records if r.instrument == a and pd.Timestamp(r.available_at) < on]
            rec = fa.latest_usable(pool, a, on, None, max_age_days)
            if rec is None:
                stale += 1 if pool else 0
                continue
            try:
                er = fa.align_horizon(fa.expected_simple_return(rec, base_currency), horizon, allow_scaling)
            except fa.ForecastConversionError as error:
                rejected.append(str(error))
                continue
            vals[a], sds[a] = er.value, er.uncertainty if er.uncertainty is not None else np.nan
        est = pd.Series(vals, dtype=float).reindex(data.columns)
        sd = pd.Series(sds, dtype=float).reindex(data.columns)
        rel = fa.reliability([r for r in records if pd.Timestamp(r.origin) < last], prices, last, base_currency)
        weight, why = 1.0, "uncalibrated"
        if calibrate == "past_slope":
            if rel["slope"] is None or rel["n"] < MIN_PAIRS:
                weight, why = 0.0, f"{rel['n']} resolved forecasts: too few to trust the forecasts yet, so they get no weight"
            else:
                weight, why = float(min(max(rel["slope"], 0.0), 1.0)), f"past slope of realised on forecast {rel['slope']:.2f}"
        return Estimate(est * weight if weight > 0 else est * np.nan, sd, {"reliability": rel, "weight": weight, "calibration": why, "stale": stale, "rejected": rejected[:3], "forecasts_used": int(est.notna().sum())})
    return mu


def combined_provider(parts: dict, method: str = "equal", step: int = 21, horizon: int = 21):
    """Blend the sources' estimates. ``inverse_error`` weights each source by one over its mean squared error on rebalances whose horizon was complete before the current one; with fewer than three such
    rebalances it falls back to equal weights and says so."""
    history: dict[str, list[tuple[int, pd.Series]]] = {k: [] for k in parts}

    def mu(p: int, data: pd.DataFrame) -> Estimate:
        ests = {k: f(p, data) for k, f in parts.items()}
        for k, e in ests.items():
            history[k].append((p, e.mu))
        w = {k: 1.0 for k in parts}
        how = "equal weights"
        if method == "inverse_error":
            errs = {}
            for k in parts:
                se = []
                for p0, m0 in history[k][:-1]:
                    if p0 + horizon > p:                                              # that horizon was not complete by the day this decision is made
                        continue
                    real = (1.0 + data.iloc[p0:p0 + horizon]).prod() - 1.0
                    ok = m0.notna()
                    if ok.sum() >= 3:
                        se.append(float(((m0[ok] - real[ok]) ** 2).mean()))
                if len(se) >= 3:
                    errs[k] = float(np.mean(se))
            if len(errs) == len(parts) and all(v > 0 for v in errs.values()):
                w = {k: 1.0 / v for k, v in errs.items()}
                how = "inverse of past squared error " + ", ".join(f"{k} {v:.2e}" for k, v in errs.items())
            else:
                how = "equal weights (fewer than three completed rebalances to weigh the sources by)"
        stack = pd.DataFrame({k: ests[k].mu for k in parts})
        weights = pd.DataFrame({k: stack[k].notna() * w[k] for k in parts})
        den = weights.sum(axis=1).replace(0, np.nan)
        out = (stack.fillna(0.0) * weights).sum(axis=1) / den
        return Estimate(out, None, {"weights": {k: float(v) / sum(w.values()) for k, v in w.items()}, "how": how, "sources": {k: e.info for k, e in ests.items()}})
    return mu
