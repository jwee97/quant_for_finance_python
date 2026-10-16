"""More forecasters that are well known in the quant literature, each answering a different question.

``kalman``   the Kalman filter on a local linear trend (Harvey 1989): the log price is a level that drifts with a slowly changing slope, observed with a little noise; the filter estimates the level and the slope
             day by day and the forecast is that slope carried forward, with the uncertainty of the state added. It is an adaptive drift estimate: how fast the slope may change is fitted by maximum likelihood.
``arma``     an ARMA(1,1) on daily returns (Box and Jenkins 1970) with no constant: whether yesterday's return and yesterday's surprise predict today's. Normal, constant-variance shocks.
``har``      the heterogeneous autoregression of realised volatility (Corsi 2009): tomorrow's volatility regressed on the volatility of the last day, week and month, each over its own horizon, with the range-based
             (Parkinson 1980) estimate of a day's variance where the data have highs and lows. Shocks are Student-t and the forecast of the variance is itself uncertain, which fattens the tails of longer horizons.
``analog``   nearest-neighbour analogues: find the days in the last five years that looked most like today (the same recent moves, volatility and distance from the high) and use what followed them,
             scaled to today's volatility. Pattern matching with its assumptions in the open.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize

from ..econometrics.arima import fit_arima
from .base import Forecast, ForecastError, Forecaster, History, cumulate, fill_between, standard_t
from .features import FEATURES, WARMUP, feature_table

TRADING_DAYS = 252
LN2 = math.log(2.0)


# ------------------------------------------------------------------------------------------------------------------------------------- Kalman local linear trend
def _filter(y: np.ndarray, ve: float, vl: float, vs: float, state: bool = False):
    """Kalman filter of ``y_t = level_t + e_t``, ``level_{t+1} = level_t + slope_t + n_t``, ``slope_{t+1} = slope_t + z_t`` in scalar arithmetic (fast for a few hundred days). Returns the
    log-likelihood, and with ``state`` the filtered level, slope and their covariance after the last day."""
    level, slope = float(y[0]), float(np.mean(np.diff(y[:20]))) if len(y) > 21 else 0.0
    p11, p12, p22 = 1.0, 0.0, 1e-4
    ll = 0.0
    for t in range(1, len(y)):
        level, p11, p12, p22 = level + slope, p11 + 2.0 * p12 + p22 + vl, p12 + p22, p22 + vs
        v = y[t] - level
        S = p11 + ve
        k1, k2 = p11 / S, p12 / S
        level, slope = level + k1 * v, slope + k2 * v
        p11, p12, p22 = p11 - k1 * p11, p12 - k1 * p12, p22 - k2 * p12
        ll -= 0.5 * (math.log(2.0 * math.pi * S) + v * v / S)
    return (ll, level, slope, p11, p12, p22) if state else ll


class KalmanTrend(Forecaster):
    name = "kalman"
    short = "Kalman trend"
    label = "Kalman filter, local linear trend"
    family = "State space"
    text = "The log price is a level with a slowly changing slope; the filter tracks both day by day and the forecast carries the slope forward, with the uncertainty of the state added."
    assumes = "A market drifts, and the drift itself changes slowly, at a pace the data can estimate."
    fails = "The drift of a stock is too small to see in two years of daily data, so the slope it finds is mostly noise; the constant-variance noise misses calm and stormy spells."
    min_history = 400
    mean_signal = True
    window = 500

    def fit(self, h: History, horizons):
        y = np.log(h.close[-self.window:])
        y = y - y[0]
        if len(y) < 250:
            raise ForecastError("too few days for a trend filter")
        base = float(np.var(np.diff(y)))
        if base <= 0:
            raise ForecastError("the price is constant")
        start = np.log([0.1 * base, base, 1e-4 * base])
        bounds = [(np.log(1e-10), np.log(1e-2)), (np.log(1e-8), np.log(1e-2)), (np.log(1e-14), np.log(1e-5))]
        res = optimize.minimize(lambda th: -_filter(y, *np.exp(th)), start, method="L-BFGS-B", bounds=bounds)
        ve, vl, vs = np.exp(res.x)
        return {"ve": float(ve), "vl": float(vl), "vs": float(vs)}

    def predict(self, state, h: History, horizons, n, rng):
        y = np.log(h.close[-self.lookback:])
        y = y - y[0]
        ve, vl, vs = state["ve"], state["vl"], state["vs"]
        _, level, slope, p11, p12, p22 = _filter(y, ve, vl, vs, state=True)
        P = np.array([[p11, p12], [p12, p22]])
        P = 0.5 * (P + P.T) + 1e-18 * np.eye(2)
        try:
            chol = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            raise ForecastError("the filter's state covariance is not positive definite") from None
        init = np.array([level, slope]) + rng.standard_normal((n, 2)) @ chol.T
        lv, sl = init[:, 0], init[:, 1]
        steps = max(horizons)
        out = np.empty((n, steps))
        for k in range(steps):
            lv = lv + sl + math.sqrt(vl) * rng.standard_normal(n)
            sl = sl + math.sqrt(vs) * rng.standard_normal(n)
            out[:, k] = lv + math.sqrt(ve) * rng.standard_normal(n) - y[-1]
        return Forecast(out, {"slope_per_day_annual": slope * TRADING_DAYS, "slope_uncertainty_annual": math.sqrt(max(p22, 0.0)) * TRADING_DAYS, "level_noise_share": ve / (ve + vl),
                              "slope_variability": vs / max(vl, 1e-18)})


# ------------------------------------------------------------------------------------------------------------------------------------- ARMA
class ARMAReturns(Forecaster):
    name = "arma"
    short = "ARMA(1,1)"
    label = "ARMA(1,1) on daily returns"
    family = "Time series"
    text = "Whether yesterday's return and yesterday's surprise help predict today's, with normal shocks of constant variance and no average return."
    assumes = "Daily returns have a little serial dependence of the ARMA kind, and nothing else."
    fails = "Daily returns have almost none, so it is close to a random walk with a spread from its own residuals; the constant variance misses volatility clustering."
    min_history = 600
    mean_signal = True
    ORDER = (1, 0, 1)

    def fit(self, h: History, horizons):
        r = pd.Series(h.r[-self.window:])
        if len(r) < 500:
            raise ForecastError("too few days for an ARMA model")
        try:
            return fit_arima(r, self.ORDER, trend="n")
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
            raise ForecastError(f"the ARMA model could not be fitted: {error}") from None

    def predict(self, state, h: History, horizons, n, rng):
        model = state
        steps = max(horizons)
        model._y = pd.Series(h.r[-self.lookback:])                           # bring the filter up to the origin with the fitted numbers held fixed
        fc = model.forecast(steps)
        mean = np.cumsum(fc["mean"].to_numpy())
        psi = model.psi_weights(steps)
        c = np.cumsum(psi)
        C = np.zeros((steps, steps))
        for k in range(steps):
            C[k, :k + 1] = c[:k + 1][::-1]                                    # the weight of the shock of step m on the cumulative return of step k is the sum of the first k - m + 1 psi weights
        eps = math.sqrt(model.sigma2) * rng.standard_normal((n, steps))
        info = {"ar": [float(v) for v in model.ar], "ma": [float(v) for v in model.ma], "residual_volatility_annual": math.sqrt(model.sigma2 * TRADING_DAYS), "next_day_mean": float(fc["mean"].iloc[0])}
        return Forecast(mean[None, :] + eps @ C.T, info)


# ------------------------------------------------------------------------------------------------------------------------------------- HAR
class HAR(Forecaster):
    name = "har"
    short = "HAR volatility"
    label = "HAR volatility model (day, week and month)"
    family = "Volatility"
    text = "Forecasts the volatility of the next days from the volatility of the last day, week and month, using the day's high and low where there are some; shocks are Student-t and the volatility forecast is itself uncertain."
    assumes = "Traders act on different horizons (days, weeks, months), so volatility has memory at all three, and the log of volatility is linear in them."
    fails = "Without highs and lows the day's variance is read from one squared return, which is a noisy proxy; it forecasts the size of moves, never their direction; a regime unlike the five years in the fit is a surprise."
    needs_range = True
    min_history = 600
    WINDOWS = (1, 5, 22)
    FLOOR = 0.05                                       # a one-day variance below this share of the average is treated as this share before taking logs

    def _proxy(self, h: History) -> tuple[np.ndarray, str]:
        """The variance of each day (aligned with the returns): the Parkinson estimator from the high and low where there are some, else the squared return."""
        if h.high is not None and h.low is not None:
            hl = np.log(h.high / h.low)
            return (hl[1:] ** 2) / (4.0 * LN2), "range"
        return h.r ** 2, "returns"

    @staticmethod
    def _means(v: np.ndarray, w: int) -> np.ndarray:
        """The mean of the last ``w`` values after each position (NaN until there are ``w``)."""
        c = np.concatenate([[0.0], np.cumsum(v)])
        out = np.full(len(v), np.nan)
        out[w - 1:] = (c[w:] - c[:-w]) / w
        return out

    def _design(self, v: np.ndarray, floor: float) -> np.ndarray:
        cols = [np.log(np.maximum(self._means(v, self.WINDOWS[0]), floor))] + [np.log(np.maximum(self._means(v, w), 1e-18)) for w in self.WINDOWS[1:]]
        return np.column_stack([np.ones(len(v))] + cols)

    def fit(self, h: History, horizons):
        w = h.tail(self.window)
        v, kind = self._proxy(w)
        r = w.r
        scale = float(np.mean(r * r) / max(np.mean(v), 1e-18)) if kind == "range" else 1.0         # a range misses the overnight gap: scale it to close-to-close variance
        v = v * scale
        floor = self.FLOOR * float(v.mean())
        X = self._design(v, floor)
        models = {}
        for hz in sorted(set(horizons) | {1}):
            ahead = self._means(v, hz)                                            # mean of days t-hz+1 .. t: shifted to be the days after t
            y = np.full(len(v), np.nan)
            y[:-hz] = np.log(np.maximum(ahead[hz:], 1e-18))
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
            if ok.sum() < 200:
                raise ForecastError("too few days for the volatility regression")
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            resid = y[ok] - X[ok] @ beta
            models[hz] = {"beta": beta, "s2": float(resid @ resid / (ok.sum() - X.shape[1]))}
        # tail weight of the one-day shocks from the fit's own standardised residuals
        idx = np.flatnonzero(np.isfinite(X).all(axis=1))
        idx = idx[idx + 1 < len(r)]
        one = models[1]
        z = r[idx + 1] / np.sqrt(np.exp(X[idx] @ one["beta"] + 0.5 * one["s2"]))        # the forecast made after day t is for the return of day t + 1
        z = z / z.std()
        kurt = float(np.mean(z ** 4) - 3.0)
        nu = float(np.clip(4.0 + 6.0 / kurt, 5.0, 100.0)) if kurt > 0.2 else 100.0
        return {"models": models, "scale": scale, "floor": floor, "kind": kind, "nu": nu}

    def predict(self, state, h: History, horizons, n, rng):
        v, kind = self._proxy(h.tail(self.lookback))
        if kind != state["kind"]:
            raise ForecastError("the data no longer have highs and lows where the fit used them")
        v = v * state["scale"]
        x = self._design(v, state["floor"])[-1]
        if not np.isfinite(x).all():
            raise ForecastError("too few recent days for the volatility features")
        steps = max(horizons)
        anchors = sorted(set(horizons) | {1})
        cum = [0.0]
        for hz in anchors:
            m = state["models"][hz]
            cum.append(hz * math.exp(float(x @ m["beta"]) + 0.5 * m["s2"]))      # the expected total variance over hz days
        grid = np.array([0] + anchors, dtype=float)
        V = np.maximum.accumulate(np.interp(np.arange(0, steps + 1), grid, np.array(cum)))
        rate = np.maximum(np.diff(V), 1e-18)                                       # the variance of each day
        sd_log = np.interp(np.arange(1, steps + 1), [1] + anchors, [math.sqrt(state["models"][1]["s2"])] + [math.sqrt(state["models"][hz]["s2"]) for hz in anchors])
        u = rng.standard_normal((n, 1))
        mult = np.exp(sd_log[None, :] * u - 0.5 * sd_log[None, :] ** 2)            # the variance forecast is uncertain: a lognormal factor of mean one, shared along a path
        z = standard_t(rng, (n, steps), state["nu"])
        daily = np.sqrt(rate[None, :] * mult) * z
        return Forecast(cumulate(daily), {"volatility_tomorrow_annual": math.sqrt(rate[0] * TRADING_DAYS), "volatility_next_month_annual": math.sqrt(V[min(21, steps)] / min(21, steps) * TRADING_DAYS),
                                          "proxy": state["kind"], "tail_degrees_of_freedom": state["nu"]})


# ------------------------------------------------------------------------------------------------------------------------------------- analogues
class Analogues(Forecaster):
    name = "analog"
    short = "Analogues"
    label = "Nearest-neighbour analogues"
    family = "Pattern matching"
    text = "Finds the days in the last five years that looked most like today (recent moves, volatility, distance from the high) and uses what followed them, scaled to today's volatility."
    assumes = "Similar situations have similar sequels, and 'similar' is well captured by five numbers."
    fails = "The closest of 1,200 days are close only by luck in five dimensions; the sequels of neighbouring days overlap, so there are fewer independent examples than it looks."
    min_history = 800
    mean_signal = True
    K = 60

    def fit(self, h: History, horizons):
        X, sigma = feature_table(h.close)
        L = np.log(h.close)
        rows = np.arange(WARMUP, h.n)
        mu, sd = X[rows].mean(axis=0), X[rows].std(axis=0) + 1e-9
        Z = np.clip((X - mu) / sd, -4.0, 4.0)
        lib = {}
        for hz in sorted(set(horizons)):
            idx = np.arange(WARMUP, h.n - hz)
            if len(idx) < 200:
                raise ForecastError("too few days to look for analogues")
            lib[hz] = {"Z": Z[idx], "y": (L[idx + hz] - L[idx]) / (sigma[idx] * np.sqrt(hz))}
        return {"mu": mu, "sd": sd, "lib": lib}

    def predict(self, state, h: History, horizons, n, rng):
        X, sigma = feature_table(h.close)
        x, sg = X[-1], sigma[-1]
        if not np.all(np.isfinite(x)) or not np.isfinite(sg):
            raise ForecastError("too few recent days for the features")
        z = np.clip((x - state["mu"]) / state["sd"], -4.0, 4.0)
        anchors, info = {}, {"analogues_used": self.K, "typical_distance": {}, "mean_standardised_outcome": {}}
        for hz in sorted(set(horizons)):
            lib = state["lib"][hz]
            d = ((lib["Z"] - z) ** 2).sum(axis=1)
            near = np.argpartition(d, self.K)[:self.K]
            outcomes = lib["y"][near]
            anchors[hz] = sg * np.sqrt(hz) * outcomes[rng.integers(0, len(outcomes), n)]
            info["typical_distance"][str(hz)] = float(np.sqrt(d[near].mean()))
            info["mean_standardised_outcome"][str(hz)] = float(outcomes.mean())
        info["features_today"] = dict(zip(FEATURES, (float(v) for v in x)))
        return Forecast(fill_between(anchors, max(horizons)), info)
