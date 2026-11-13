"""Path-dependent volatility (Guyon and Lekeufack, "Volatility is (mostly) path-dependent", Quantitative Finance 2023).

The paper's finding is that most of the level of volatility, realised or implied, is explained by two things one can read from the past returns alone: a weighted average of the returns themselves (a
trend: after falls volatility is higher, the leverage effect) and the root of a weighted average of the squared returns (recent activity: volatility clusters). The weights fade with the age of the return,
quickly at first and then slowly. The model is ``sigma_t = b0 + b1 * trend_t + b2 * activity_t`` with ``b1 < 0``.

Here each of the two is measured on two clocks (a fast and a slow exponential average, the paper's Markovian form of its power-law kernel), and the four loadings are fitted by least squares to the next
day's volatility proxy (the day's high-low range where the data have one, else the absolute return). A forecast is a simulation: each simulated day draws a shock from the model's own past standardised
shocks, scales it by the volatility the path so far implies, and updates the four averages with the new return. So a path that falls raises its own volatility, which a GARCH(1,1) cannot do and which is
where the fat left tail of multi-day returns comes from.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from .base import Forecast, Forecaster, ForecastError, History, cumulate

LN2 = math.log(2.0)


class PathDependentVol(Forecaster):
    name = "pdv"
    short = "Path-dependent vol"
    label = "Path-dependent volatility (trend and activity of past returns, simulated)"
    family = "Volatility"
    text = ("Volatility is read from the path of past returns: it is higher after falls and after busy days, on a fast and a slow clock. Each simulated path updates its own volatility as it goes, "
            "so a path that falls becomes more volatile.")
    assumes = "The level of volatility is a fixed linear function of a weighted average of past returns and of past squared returns; shocks, once scaled by it, are draws from one distribution."
    fails = ("The loadings are fitted to a noisy one-day proxy and held fixed, so a market whose reaction to falls changes is a surprise; it forecasts the size of moves, never their direction; "
             "volatility that moves for reasons outside the price path (an election, an earnings date) is invisible to it.")
    needs_range = True
    min_history = 600
    TREND_HALFLIVES = (3.0, 20.0)                      # days: the fast and the slow clock of the trend of returns
    ACTIVITY_HALFLIVES = (8.0, 60.0)                   # and of the squared returns
    FLOOR = 0.25                                       # the fitted volatility is never below this share of its median in the fit
    CAP = 4.0                                          # nor above this multiple of its largest value in the fit: a simulated path cannot feed on itself without limit

    @staticmethod
    def _decay(halflife: float) -> float:
        return 0.5 ** (1.0 / halflife)

    def _features(self, r: np.ndarray) -> np.ndarray:
        """``(len(r), 5)``: a constant, the trend on the two clocks, and the activity (root mean squared return) on the two clocks, each using the returns up to and including that day."""
        cols = [np.ones(len(r))]
        for hl in self.TREND_HALFLIVES:
            a = self._decay(hl)
            cols.append(lfilter([1.0 - a], [1.0, -a], r))
        for hl in self.ACTIVITY_HALFLIVES:
            a = self._decay(hl)
            cols.append(np.sqrt(lfilter([1.0 - a], [1.0, -a], r * r)))
        return np.column_stack(cols)

    @staticmethod
    def _proxy(h: History) -> tuple[np.ndarray, str]:
        """The volatility of each day, aligned with the returns: from the high-low range (Parkinson) where there is one, else the absolute return scaled so that its mean is the standard deviation of a normal."""
        if h.high is not None and h.low is not None:
            return np.abs(np.log(h.high / h.low))[1:] / math.sqrt(4.0 * LN2), "range"
        return np.abs(h.r) * math.sqrt(math.pi / 2.0), "returns"

    def fit(self, h: History, horizons):
        w = h.tail(self.window)
        r = w.r
        if len(r) < 400:
            raise ForecastError("too few days for the volatility regression")
        v, kind = self._proxy(w)
        burn = int(4 * max(self.ACTIVITY_HALFLIVES))                                          # the slow average needs this long to forget where it started
        X = self._features(r)
        Xf, y = X[burn:-1], v[burn + 1:]                                                      # the features after day t explain the volatility of day t + 1
        ok = np.isfinite(Xf).all(axis=1) & np.isfinite(y)
        if ok.sum() < 250:
            raise ForecastError("too few days for the volatility regression")
        beta, *_ = np.linalg.lstsq(Xf[ok], y[ok], rcond=None)
        fitted = Xf @ beta
        floor = self.FLOOR * float(np.median(fitted[ok]))
        if not np.isfinite(floor) or floor <= 0:
            raise ForecastError("the fitted volatility is not positive")
        cap = self.CAP * float(np.max(fitted[ok]))
        sigma = np.clip(fitted, floor, cap)
        z = r[burn + 1:][ok] / sigma[ok]
        scale = float(np.sqrt(np.mean(z * z)))                                                # the proxy measures the typical move; the shocks are rescaled to unit variance and the volatility with them
        resid = y[ok] - fitted[ok]
        r2 = 1.0 - float(resid @ resid) / float(((y[ok] - y[ok].mean()) ** 2).sum())
        return {"beta": beta, "floor": floor, "cap": cap, "scale": scale, "shocks": z / scale, "kind": kind, "r2": r2}

    def _sigma(self, state, trend, activity) -> np.ndarray:
        b = state["beta"]
        raw = b[0] + b[1] * trend[0] + b[2] * trend[1] + b[3] * np.sqrt(activity[0]) + b[4] * np.sqrt(activity[1])
        return np.clip(raw, state["floor"], state["cap"]) * state["scale"]

    def predict(self, state, h: History, horizons, n, rng):
        r = h.tail(self.lookback).r
        if len(r) < int(4 * max(self.ACTIVITY_HALFLIVES)):
            raise ForecastError("too few recent days for the volatility features")
        x = self._features(r)[-1]
        steps = max(horizons)
        ta, aa = [self._decay(hl) for hl in self.TREND_HALFLIVES], [self._decay(hl) for hl in self.ACTIVITY_HALFLIVES]
        trend = [np.full(n, x[1]), np.full(n, x[2])]
        activity = [np.full(n, x[3] ** 2), np.full(n, x[4] ** 2)]
        now = float(self._sigma(state, [x[1], x[2]], [x[3] ** 2, x[4] ** 2]))
        out = np.empty((n, steps))
        shocks = rng.choice(state["shocks"], size=(n, steps))
        for k in range(steps):
            ret = self._sigma(state, trend, activity) * shocks[:, k]
            out[:, k] = ret
            for i in range(2):
                trend[i] = ta[i] * trend[i] + (1.0 - ta[i]) * ret
                activity[i] = aa[i] * activity[i] + (1.0 - aa[i]) * ret * ret
        b = state["beta"]
        info = {"volatility_now_annual": now * math.sqrt(252.0), "trend_loading_fast": float(b[1]), "trend_loading_slow": float(b[2]), "activity_loading_fast": float(b[3]), "activity_loading_slow": float(b[4]),
                "falls_raise_volatility": bool(b[1] + b[2] < 0), "share_of_proxy_explained": float(state["r2"]), "volatility_proxy": state["kind"]}
        return Forecast(cumulate(out), info)
