"""Where regimes, calibration and alpha decay change what a forecast is worth (Generation 5 integration layer).

    regime_spread_scale     the forecast spread is re-scaled by how wrong the forecasts have been in the regime the market is in now,
                            so confidence falls where the model has been over-confident (Regime -> Forecast confidence)
    calibrate_confidence    P(up) is recalibrated (Platt or isotonic) on matured outcomes, and confidence is |2 P(up) - 1| of the calibrated
                            probability: probability -> calibration -> trade
    decay_trust_weights     alpha-combination weights from each model's IC-versus-horizon curve (its decay), not only its IC at one horizon
    RegimeRiskLimits        gross-exposure caps by regime and a drawdown de-risking rule (Regime -> Risk limits)

Everything here uses only information that was available on the date it acts: labels enter an estimate only once their horizon has elapsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..signals.alpha_engine import forward_returns, matured_ic
from ..utils.dates import rebalance_dates
from .types import ForecastPanel, confidence_from


# ---------------------------------------------------------------------------------------------- regime -> forecast spread
def regime_spread_scale(panel: ForecastPanel, returns: pd.DataFrame, regimes, min_obs: int = 1000, shrink: float = 0.5,
                        bounds: tuple[float, float] = (0.5, 2.0)) -> tuple[ForecastPanel, pd.Series]:
    """Re-scale ``std`` by the realised dispersion of the standardised forecast errors in the current regime.

    On date ``t`` the error of an earlier forecast is ``z = (return over s+1..s+h - mean_s) / std_s``, known once ``s <= t - h``. For regime ``r``,
    ``kappa_r`` is the root of the probability-weighted mean of ``z^2`` over matured pairs (weights = the regime probability on ``s``). A forecaster
    whose spread is right has ``kappa = 1``; ``kappa > 1`` means over-confidence in that regime. The scale on ``t`` is ``sum_r p_r(t) kappa_r``, with
    ``kappa`` shrunk toward 1 by ``shrink`` and fixed at 1 until a regime has ``min_obs`` matured (probability-weighted) observations.
    Returns the adjusted panel and the scale series.
    """
    if regimes is None:
        raise ValueError("regime_spread needs a regime detector in the spec")
    h = panel.horizon
    fwd = forward_returns(returns, h).reindex_like(panel.mean)
    z = ((fwd - panel.mean) / panel.std.where(panel.std > 0))
    ok = z.notna()
    z2 = (z ** 2).where(ok)
    n_s = ok.sum(axis=1).astype(float)
    sq_s = z2.sum(axis=1)
    probs = regimes.probabilities.reindex(panel.mean.index).fillna(0.0)
    scale = pd.Series(0.0, index=panel.mean.index)
    covered = pd.Series(0.0, index=panel.mean.index)
    lo, hi = bounds
    for name in probs.columns:
        p = probs[name]
        num = (p * sq_s).cumsum().shift(h)
        den = (p * n_s).cumsum().shift(h)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = np.sqrt(num / den.where(den > 0))
        kappa = kappa.where(den >= min_obs, 1.0).fillna(1.0).clip(lo, hi)
        kappa = shrink * 1.0 + (1.0 - shrink) * kappa
        scale = scale + p * kappa
        covered = covered + p
    scale = scale + (1.0 - covered).clip(0.0, 1.0)                  # no regime view yet: leave the spread alone
    std = panel.std.mul(scale, axis=0)
    adjusted = ForecastPanel(panel.mean, std, confidence_from(panel.mean, std), h, panel.name, None)
    return adjusted, scale


# ---------------------------------------------------------------------------------------------- calibration of P(up)
def calibrate_confidence(panel: ForecastPanel, returns: pd.DataFrame, method: str = "platt", min_obs: int = 2000, refit_every: int = 21) -> ForecastPanel:
    """Recalibrate ``P(up) = Phi(mean/std)`` walk-forward and recompute confidence as ``|2 P(up) - 1|``.

    ``method``: ``platt`` (sigmoid of the logit), ``isotonic`` (monotone steps) or ``none`` (returns the panel unchanged). Every ``refit_every`` days the
    calibrator is fitted on all (probability, outcome) pairs whose outcome was complete by then (``s <= t - h``) and applied until the next refit;
    until ``min_obs`` such pairs exist the raw probability is used. ``mean`` and ``std`` are untouched: only the probability, and so the confidence, change.
    """
    if method == "none":
        return panel
    if method not in ("platt", "isotonic"):
        raise ValueError("calibration method must be 'platt', 'isotonic' or 'none'")
    from ..models.probabilistic import PlattCalibrator, clip_probability

    h = panel.horizon
    raw = panel.probability_up()
    fwd = forward_returns(returns, h).reindex_like(raw)
    up = (fwd > 0).astype(float).where(fwd.notna())
    index = raw.index
    out = raw.copy()
    cut = np.cumsum((raw.notna().to_numpy() & up.notna().to_numpy()).sum(axis=1))             # cumulative usable (probability, outcome) pairs by row
    refit_rows = np.arange(0, len(index), int(refit_every))
    raw_v, up_v = raw.to_numpy(), up.to_numpy()
    for r in refit_rows:
        upto = r - h                                                   # rows <= upto have a complete outcome by row r
        if upto <= 0 or cut[upto] < min_obs:
            continue
        x, y = raw_v[: upto + 1].ravel(), up_v[: upto + 1].ravel()
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = clip_probability(x[keep]), y[keep]
        if len(np.unique(y)) < 2:
            continue
        end = min(r + int(refit_every), len(index))
        block = raw_v[r:end]
        flat = block.ravel()
        good = np.isfinite(flat)
        mapped = np.full(flat.shape, np.nan)
        if method == "platt":
            mapped[good] = PlattCalibrator().fit(x, y).transform(flat[good])
        else:
            from sklearn.isotonic import IsotonicRegression

            mapped[good] = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y).predict(flat[good])
        out.iloc[r:end] = mapped.reshape(block.shape)
    conf = (2.0 * out - 1.0).abs()
    return ForecastPanel(panel.mean, panel.std, conf.where(panel.mean.notna()), h, panel.name, out.where(panel.mean.notna()))


# ---------------------------------------------------------------------------------------------- alpha decay -> combination
_TAU_GRID = np.concatenate([np.geomspace(0.5, 400.0, 60), [np.inf]])


def fit_decay(ic_by_lag: dict[int, float]) -> tuple[float, float]:
    """Fit ``IC(d) = IC0 * exp(-d / tau)`` to the mean incremental ICs by non-negative least squares over a grid of ``tau``.

    All lags are used, including the noisy or negative ones (dropping them would bias the fit toward whatever happened to be positive). ``IC0`` is the
    least-squares amplitude for each ``tau`` clipped at zero, and the ``tau`` with the smallest squared error wins (``inf`` = no decay across the lags).
    Returns ``(IC0, tau)``; ``(0, nan)`` when no positive amplitude fits or fewer than three lags are available.
    """
    pts = [(d, v) for d, v in ic_by_lag.items() if np.isfinite(v)]
    if len(pts) < 3:
        return 0.0, float("nan")
    d = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    best = (float("inf"), 0.0, float("nan"))
    for tau in _TAU_GRID:
        basis = np.ones_like(d) if np.isinf(tau) else np.exp(-d / tau)
        amp = max(0.0, float(basis @ y / (basis @ basis)))
        sse = float(((y - amp * basis) ** 2).sum())
        if amp > 0 and sse < best[0]:
            best = (sse, amp, float(tau))
    return (best[1], best[2]) if best[1] > 0 else (0.0, float("nan"))


def holding_period_ic(ic0: float, tau: float, holding: int) -> float:
    """Mean incremental IC over a holding window of ``holding`` days for an alpha whose IC decays as ``IC0 * exp(-d / tau)``: ``IC0 * tau * (1 - exp(-H / tau)) / H``."""
    if ic0 <= 0:
        return 0.0
    if not np.isfinite(tau):
        return float(ic0) if tau == float("inf") else 0.0
    return float(ic0 * tau * (1.0 - np.exp(-holding / tau)) / holding)


def decay_trust_weights(panels: dict[str, ForecastPanel], returns: pd.DataFrame, lags=(1, 5, 10, 21, 42, 63), window: int = 756, min_obs: int = 252,
                        holding: int = 21) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly combination weights proportional to each model's decay-adjusted holding-period IC.

    The *incremental* IC at lag ``d`` is the cross-sectional rank correlation between a forecast on date ``t`` and the SINGLE-day return on ``t + d``
    (not the compounded return over ``d`` days), so it measures how much information the forecast still carries ``d`` days later. For each model it is
    averaged over the trailing ``window`` matured days (needs ``min_obs``) and an exponential decay is fitted across lags; the weight is the mean
    incremental IC over the ``holding`` days a position is held. Models with no identified positive information get nothing; if no model has any, the
    weights are equal. Returns ``(weights, decay_table)`` where the table holds the fitted ``IC0`` and ``tau`` through time.
    """
    index = pd.DatetimeIndex(returns.index)
    updates = rebalance_dates(index, "monthly")
    names = list(panels)
    ic = {n: {d: matured_ic(panels[n].mean.reindex(index), returns.shift(-d), d) for d in lags} for n in names}
    rows, decay_rows = {}, {}
    for date in updates:
        value = {}
        for n in names:
            curve = {}
            for d in lags:
                s_ = ic[n][d].loc[:date].dropna().iloc[-window:]
                curve[d] = float(s_.mean()) if len(s_) >= min_obs else np.nan
            ic0, tau = fit_decay(curve)
            value[n] = holding_period_ic(ic0, tau, holding)
            decay_rows[(date, n)] = {"ic0": ic0, "tau": tau}
        total = sum(value.values())
        rows[date] = [value[n] / total for n in names] if total > 0 else [1.0 / len(names)] * len(names)
    weights = pd.DataFrame.from_dict(rows, orient="index", columns=names).reindex(index).ffill()
    weights = weights.fillna(1.0 / len(names))
    table = pd.DataFrame.from_dict(decay_rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["date", "model"])
    return weights, table


# ---------------------------------------------------------------------------------------------- regime -> risk limits
@dataclass
class RegimeRiskLimits:
    """Hard limits applied to FINAL target weights.

    ``gross_caps``: regime name -> maximum gross exposure; the cap on a date is the probability-weighted average (regimes without a cap are unconstrained).
    ``drawdown``: ``{"trigger": 0.10, "derisk": 0.5, "recover": 0.05}``. When the drawdown of the LIMITED book's own realised equity from its running peak
    reaches ``trigger``, exposure is multiplied by ``derisk`` until the drawdown has recovered to ``recover``.
    The limited book's equity path uses weights lagged one day, as the engine does, so the rule uses only returns already realised.
    """

    gross_caps: dict = field(default_factory=dict)
    drawdown: dict = field(default_factory=dict)

    def cap_series(self, regimes, index: pd.DatetimeIndex) -> pd.Series:
        if not self.gross_caps or regimes is None:
            return pd.Series(np.inf, index=index)
        probs = regimes.probabilities.reindex(index).fillna(0.0)
        cap = pd.Series(0.0, index=index)
        covered = pd.Series(0.0, index=index)
        for name, c in self.gross_caps.items():
            if name in probs.columns:
                cap = cap + probs[name] * float(c)
                covered = covered + probs[name]
        return cap + (1.0 - covered).clip(0.0, 1.0) * 1e6              # an effectively unbounded cap where no capped regime is probable

    def apply(self, weights: pd.DataFrame, returns: pd.DataFrame, regimes=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        w = weights.copy()
        gross = w.abs().sum(axis=1)
        cap = self.cap_series(regimes, w.index)
        cap_factor = (cap / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        w = w.mul(cap_factor, axis=0)
        dd_factor = pd.Series(1.0, index=w.index)
        drawdown_state = pd.Series(0.0, index=w.index)
        if self.drawdown:
            trigger, derisk = float(self.drawdown["trigger"]), float(self.drawdown.get("derisk", 0.5))
            recover = float(self.drawdown.get("recover", trigger / 2.0))
            if not (0.0 < recover < trigger < 1.0 and 0.0 <= derisk < 1.0):
                raise ValueError("drawdown limits need 0 < recover < trigger < 1 and 0 <= derisk < 1")
            r = returns.reindex(index=w.index, columns=w.columns).fillna(0.0).to_numpy()
            wv = w.to_numpy()
            equity, peak, on = 1.0, 1.0, False
            prev = np.zeros(wv.shape[1])
            factors, states = np.ones(len(w)), np.zeros(len(w))
            for t in range(len(w)):
                equity *= 1.0 + float(prev @ r[t])                        # return on day t from the limited weights stamped at t-1
                peak = max(peak, equity)
                dd = 1.0 - equity / peak
                if not on and dd >= trigger:
                    on = True
                elif on and dd <= recover:
                    on = False
                f = derisk if on else 1.0
                factors[t], states[t] = f, float(on)
                prev = wv[t] * f
            dd_factor = pd.Series(factors, index=w.index)
            drawdown_state = pd.Series(states, index=w.index)
            w = w.mul(dd_factor, axis=0)
        diag = pd.DataFrame({"cap_factor": cap_factor, "drawdown_factor": dd_factor, "drawdown_limit_on": drawdown_state})
        return w, diag
