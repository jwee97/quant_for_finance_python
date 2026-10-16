"""Gaussian process forecasters.

A Gaussian process is a prior over functions in which any handful of values is jointly normal with a covariance that says how alike nearby inputs are. Conditioning on data gives, at every new input, a
mean and an honest variance that widens where the data are thin (Rasmussen and Williams 2006). The two ways of using one for a stock are very different, and both are here:

``gp_returns``  *regression*. Learn how the return over the next ``h`` days, in units of the volatility of the day, depends on a few features of where the market is now: the moves of the last week, month
                and quarter, whether volatility is above or below its usual level, how far the price sits below its 126-day high. One input length scale per feature is learned (automatic relevance
                determination), so a feature that does not matter is switched off by the data. The prediction is a mean (usually close to the average return, because there is little signal) and an
                uncertainty; the noise around it is resampled from the fit's own residuals, so the tails are the data's.
``gp_drift``    *a smooth drift*. Treat the daily return, in units of volatility, as a slowly changing average (a Gaussian process in time with a Matern kernel) plus noise, and carry the average forward. The
                length scale says how fast the drift may change; the noise is most of the variance, as it is in real returns, so the forecast stays close to the average of the last year and moves
                only where the data insist.
``gp_price``    *extrapolation of the price*. Fit a smooth curve through the log price of the last 120 days (a Matern kernel plus a linear trend) and continue it. Draws from the posterior give coherent
                price paths with a spread that widens with distance. It is the picture people have in mind for "a Gaussian process forecast" and it is here as the control: a price is not a smooth curve
                plus noise but a sum of its own shocks, and the curve fitted to it extrapolates those shocks as if they were a trend. It is expected to come last, and the evaluation says by how much.
"""

from __future__ import annotations

import numpy as np

from ..models.gaussian_process import GaussianProcess, make_kernel
from .base import Forecast, ForecastError, Forecaster, History, fill_between
from .features import FEATURES, WARMUP, ewma_variance, feature_table

MAX_POINTS = 220                                      # training points of a regression: the exact process costs a cube of this
NOISE_BOUNDS = (0.75, 3.0)                            # the prior for returns: at most a quarter of their variance is predictable
LENGTH_BOUNDS = (0.7, 50.0)                           # smooth in the features: a length scale below the spread of the data would interpolate the noise
WINSOR_X, WINSOR_Y = 4.0, 6.0                         # features and outcomes (in units of volatility) are clipped here so that one crash does not set a length scale


class GPReturns(Forecaster):
    name = "gp_returns"
    short = "GP regression"
    label = "Gaussian process on recent moves and volatility"
    family = "Gaussian process"
    text = "Learns how the next return, in units of volatility, depends on the recent moves, on whether volatility is high or low, and on the distance from the 126-day high; one uncertainty comes with every forecast."
    assumes = "Where the market stands now (its recent moves and volatility) shapes the next weeks' return, smoothly, in a way that was the same in the last five years."
    fails = "There is little signal: the fit usually settles on a mean close to the average return and a noise that is almost all of the variance. A relationship that held for five years and not afterwards is learned and then wrong."
    min_history = 800
    mean_signal = True
    heavy = True

    def fit(self, h: History, horizons):
        X, sigma = feature_table(h.close)
        L = np.log(h.close)
        models = {}
        for hz in sorted(set(horizons)):
            idx = np.arange(WARMUP, h.n - hz)
            if len(idx) < 150:
                raise ForecastError("too few days to learn from")
            y = (L[idx + hz] - L[idx]) / (sigma[idx] * np.sqrt(hz))
            pick = idx[::-1][::max(1, len(idx) // MAX_POINTS)][::-1]               # evenly spaced, ending at the most recent day with a known outcome
            keep = np.searchsorted(idx, pick)
            Xtr, ytr = X[pick], np.clip(y[keep], -WINSOR_Y, WINSOR_Y)
            mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
            Z = np.clip((Xtr - mu) / sd, -WINSOR_X, WINSOR_X)
            gp = GaussianProcess(make_kernel("matern52", Z.shape[1], LENGTH_BOUNDS), noise=1.0, normalize_y=True, restarts=0, seed=0, noise_bounds=NOISE_BOUNDS).fit(Z, ytr)
            resid = ytr - gp.predict(Z)
            models[hz] = {"gp": gp, "mu": mu, "sd": sd, "resid": resid - resid.mean(), "y_var": float(ytr.var()), "n": len(pick)}
        return models

    def predict(self, state, h: History, horizons, n, rng):
        X, sigma = feature_table(h.close)
        x, sg = X[-1], sigma[-1]
        if not np.all(np.isfinite(x)) or not np.isfinite(sg):
            raise ForecastError("too few recent days for the features")
        anchors, info = {}, {"features_today": dict(zip(FEATURES, (float(v) for v in x))), "standardised_forecast": {}, "unexplained_share": {}}
        for hz in sorted(set(horizons)):
            m = state[hz]
            z = np.clip((x - m["mu"]) / m["sd"], -WINSOR_X, WINSOR_X)[None, :]
            mean, sd_f = m["gp"].predict(z, return_std=True)
            draws = mean[0] + sd_f[0] * rng.standard_normal(n) + m["resid"][rng.integers(0, len(m["resid"]), n)]
            anchors[hz] = sg * np.sqrt(hz) * draws
            info["standardised_forecast"][str(hz)] = float(mean[0])
            info["unexplained_share"][str(hz)] = float(min(1.0, m["resid"].var() / m["y_var"]))
        top = state[max(state)]
        ls = top["gp"].length_scales
        if ls is not None:
            info["feature_relevance"] = dict(zip(FEATURES, (float(v) for v in 1.0 / np.asarray(ls))))        # a short length scale means the feature matters
        return Forecast(fill_between(anchors, max(horizons)), info)


class GPDrift(Forecaster):
    name = "gp_drift"
    short = "GP drift"
    label = "Gaussian process on the drift"
    family = "Gaussian process"
    text = "Treats the daily return, in units of volatility, as a slowly changing average plus noise (a Gaussian process in time) and carries the average forward, fading back to the last year's average at the pace the fit finds."
    assumes = "A stock has a drift that changes smoothly over weeks and months, with daily noise around it, and the pace of the changes can be learned from the last year."
    fails = "The drift is a small part of the noise: a year of daily returns pins it down to within several times its own size, so most of what the process finds is noise. The forecast stays close to the year's average and moves only where the data insist."
    min_history = 500
    mean_signal = True
    LOOK = 250
    window = 500
    NOISE = (0.7, 3.0)                                 # in units of the variance of the standardised returns: at most 30% of it is a drift
    LENGTH = (0.1, 3.0)                                # in hundreds of days: from ten days to three hundred

    def _inputs(self, h: History):
        r = h.r
        v = ewma_variance(r)
        z = r[31:] / np.sqrt(v[30:-1])                 # each return over the volatility known before it
        z = np.clip(z[-self.LOOK:], -WINSOR_Y, WINSOR_Y)
        return (np.arange(1 - len(z), 1, dtype=float) / 100.0)[:, None], z, float(np.sqrt(v[-1]))

    def fit(self, h: History, horizons):
        if h.n < self.LOOK + 60:
            raise ForecastError("too few days for a drift")
        x, z, _ = self._inputs(h)
        gp = GaussianProcess(make_kernel("matern32", 1, self.LENGTH), noise=1.0, normalize_y=True, restarts=0, seed=0, noise_bounds=self.NOISE).fit(x, z)
        return {"kernel": gp.kernel.clone(), "noise": gp.noise}

    def predict(self, state, h: History, horizons, n, rng):
        x, z, sigma = self._inputs(h)
        gp = GaussianProcess(state["kernel"].clone(), noise=state["noise"], normalize_y=True, optimize=False, noise_bounds=self.NOISE).fit(x, z)
        steps = max(horizons)
        xs = (np.arange(1, steps + 1, dtype=float) / 100.0)[:, None]
        f = gp.sample_y(xs, n=n, seed=int(rng.integers(2 ** 31 - 1)))
        draws = f + np.sqrt(gp.noise) * gp.y_scale * rng.standard_normal(f.shape)
        mean = gp.predict(xs)
        return Forecast(np.cumsum(sigma * draws, axis=1), {"drift_tomorrow_in_volatilities": float(mean[0]), "drift_in_a_month_in_volatilities": float(mean[min(20, steps - 1)]), "average_of_the_year_in_volatilities": float(z.mean()),
                                                           "length_scale_days": float(np.ravel(gp.length_scales)[0] * 100.0), "drift_share_of_variance": float(gp.kernel.variance / (gp.kernel.variance + gp.noise))})


class GPPrice(Forecaster):
    name = "gp_price"
    short = "GP price path"
    label = "Gaussian process on the price path (the textbook picture)"
    family = "Gaussian process"
    text = "Fits a smooth curve through the last 120 days of log price (a Matern kernel plus a linear trend) and continues it; posterior draws give price paths whose spread widens with distance. Here as the control: it is what people try first."
    assumes = "The recent path is a smooth trend plus noise, and the trend persists (or fades) at the pace the curve implies."
    fails = "A price is not a smooth curve but the sum of its own shocks, so the curve takes noise for trend and extrapolates it; the spread it gives is set by how wiggly the last 120 days were, not by the volatility. Expect it to come last."
    min_history = 400
    mean_signal = True
    LOOK = 120
    window = 500

    def _inputs(self, close: np.ndarray):
        L = np.log(close[-self.LOOK:])
        return (np.arange(1 - len(L), 1, dtype=float) / self.LOOK)[:, None], L - L[-1]

    def fit(self, h: History, horizons):
        if h.n < self.LOOK + 20:
            raise ForecastError("too few days for a trend")
        x, y = self._inputs(h.close)
        gp = GaussianProcess(make_kernel("matern32+linear", 1), noise=0.05, normalize_y=True, restarts=0, seed=0).fit(x, y)
        return {"kernel": gp.kernel.clone(), "noise": gp.noise}

    def predict(self, state, h: History, horizons, n, rng):
        x, y = self._inputs(h.close)
        gp = GaussianProcess(state["kernel"].clone(), noise=state["noise"], normalize_y=True, optimize=False).fit(x, y)
        steps = max(horizons)
        xs = (np.arange(1, steps + 1, dtype=float) / self.LOOK)[:, None]
        f = gp.sample_y(xs, n=n, seed=int(rng.integers(2 ** 31 - 1)))
        draws = f + np.sqrt(gp.noise) * gp.y_scale * rng.standard_normal(f.shape)
        mean = gp.predict(xs)
        return Forecast(draws, {"trend_per_day_annual": float((mean[-1] - mean[0]) / max(steps - 1, 1) * 252), "noise_share": float(gp.noise / (gp.noise + 1.0)), "length_scale_days": float(np.ravel(gp.length_scales)[0] * self.LOOK)})
