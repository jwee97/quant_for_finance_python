"""Gaussian process regression on price characteristics: a nonparametric forecast with a measure of how sure it is.

    gp_factor_model    the characteristics of ``characteristic_regression`` fed to a Gaussian process (src/models/gaussian_process.py) fitted on the earlier months whose returns are known; the
                       kernel decides what kind of function it can learn (rbf and matern: smooth or rough nonlinear effects and interactions, with a length scale per characteristic that says how much it
                       matters; linear: Bayesian ridge), the marginal likelihood sets its hyperparameters, and the posterior standard deviation says where the forecast rests on little data

A monthly decision on data through the month-end. ``model.uncertainty(bundle)`` returns the posterior standard deviation behind each forecast.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..equity.alpha_model import standardize
from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..models.gaussian_process import GaussianProcess, make_kernel
from .factor_models import CHARACTERISTICS, _characteristic_list, _grid, _to_daily, price_characteristics


@register_model("gp_factor_model", "machine learning", "Gaussian process regression on price characteristics: a nonlinear forecast with a length scale per characteristic and a posterior standard deviation, refitted monthly on returns that have already happened")
class GPFactorModel(ForecastModel):
    """A Gaussian process fitted each month-end to the characteristics (standardised across assets) and the following month's return in excess of the cross-section's average, over the last ``train_months``
    months whose returns are known, subsampled to ``max_points`` rows. ``kernel`` is ``rbf``, ``matern32``, ``matern52``, ``linear`` or a sum (``rbf+linear``); the hyperparameters (a length scale per
    characteristic, the signal variance, the noise) are re-optimised on the marginal likelihood every ``optimize_every`` months and held fixed in between, when only the data are refreshed. A characteristic that
    does not help ends with a long length scale, and a signal that is not there ends with the noise variance near the target's, so the forecast shrinks toward zero by itself."""

    name, family, position_mode = "gp_factor_model", "machine learning", "cross_sectional"
    min_assets = 8

    def __init__(self, kernel: str = "rbf", characteristics: str = ",".join(CHARACTERISTICS), train_months: int = 48, min_months: int = 24, max_points: int = 500, optimize_every: int = 12,
                 restarts: int = 0, seed: int = 0):
        if train_months < 12 or min_months < 6 or max_points < 50 or optimize_every < 1 or restarts < 0:
            raise ValueError("train_months >= 12, min_months >= 6, max_points >= 50, optimize_every >= 1, restarts >= 0")
        _characteristic_list(characteristics)
        make_kernel(kernel, 1)                                                                     # refuses a kernel name it does not know
        self.kernel_spec, self.characteristics, self.train_months, self.min_months = kernel, characteristics, int(train_months), int(min_months)
        self.max_points, self.optimize_every, self.restarts, self.seed = int(max_points), int(optimize_every), int(restarts), int(seed)

    def _run(self, data):
        chars = price_characteristics(data)
        names = _characteristic_list(self.characteristics)
        grid, forward = _grid(data)
        investable = data.investable.loc[grid]
        z = [standardize(chars[n].loc[grid], investable) for n in names]
        X = np.stack([f.fillna(0.0).to_numpy() for f in z], axis=-1)                               # months x assets x features
        have = np.stack([f.notna().to_numpy() for f in z], axis=-1).all(axis=-1)
        y = forward.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y = y - np.nanmean(y, axis=1, keepdims=True)
        mean, std = np.full(have.shape, np.nan), np.full(have.shape, np.nan)
        gp, last_opt = None, -10 ** 9
        rng = np.random.default_rng(self.seed)
        theta = None
        for t in range(len(grid)):
            lo = max(0, t - self.train_months)
            usable = [(s, np.flatnonzero(have[s] & np.isfinite(y[s]))) for s in range(lo, t)]        # months whose following month has ended by t
            usable = [(s, idx) for s, idx in usable if len(idx) >= 8]
            if len(usable) < self.min_months or not have[t].any():
                continue
            Xtr = np.vstack([X[s][idx] for s, idx in usable])
            ytr = np.concatenate([y[s][idx] for s, idx in usable]) * 100.0
            if len(ytr) > self.max_points:
                keep = np.sort(rng.choice(len(ytr), self.max_points, replace=False))
                Xtr, ytr = Xtr[keep], ytr[keep]
            optimise = gp is None or t - last_opt >= self.optimize_every
            kernel = make_kernel(self.kernel_spec, X.shape[2])
            if theta is not None and len(theta) == len(kernel.theta):
                kernel.theta = theta                                                               # warm start from the last optimum (held fixed between optimisations)
            noise = gp.noise if gp is not None else 0.5
            gp = GaussianProcess(kernel, noise=noise, optimize=optimise, restarts=self.restarts if optimise else 0, seed=self.seed)
            try:
                gp.fit(Xtr, ytr)
            except np.linalg.LinAlgError:
                continue
            theta = gp.kernel.theta.copy()
            if optimise:
                last_opt = t
            now = np.flatnonzero(have[t])
            m, s = gp.predict(X[t][now], return_std=True)
            mean[t, now], std[t, now] = m, s
        cols = data.returns.columns
        return (_to_daily(pd.DataFrame(mean, index=grid, columns=cols), data), _to_daily(pd.DataFrame(std, index=grid, columns=cols), data))

    def score(self, data):
        return self._run(data)[0]

    def uncertainty(self, data) -> pd.DataFrame:
        """The posterior standard deviation of the function value behind each forecast, in the same units as the score (percent of return per month)."""
        return self._run(data)[1]
