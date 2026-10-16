"""Monte Carlo forecasters: simulate many futures from a fitted model of how returns behave, and read the distribution off the simulations.

``garch``  a GJR-GARCH(1,1) with Student-t shocks (Glosten, Jagannathan and Runkle 1993; Bollerslev 1986). Volatility clusters (a big move is followed by more big moves), falls hit harder than rises
           (the leverage effect) and the shocks have fat tails. The simulation starts from the volatility the model has today and lets it fall back toward its long-run level at the pace it was fitted to.
``fhs``    filtered historical simulation (Barone-Adesi, Giannopoulos and Vosper 1999): the same volatility model, but the shocks are the model's own past standardised residuals drawn at random, so the tails
           are the ones the stock actually had instead of the ones a Student-t has.
``jump``   Merton's (1976) jump diffusion: ordinary small days plus rare large ones, with the rate and size of the large ones read from the last five years. Constant volatility between jumps.

None of them forecasts a direction: the average return is set to zero, so what they are tested on is how well they describe the *spread* of what might happen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..econometrics.garch import fit_garch
from .base import Forecast, ForecastError, Forecaster, History, cumulate

TRADING_DAYS = 252


class GarchMC(Forecaster):
    name = "garch"
    short = "GJR-GARCH"
    label = "GJR-GARCH with Student-t shocks (Monte Carlo)"
    family = "Monte Carlo"
    text = "Volatility clusters, falls raise it more than rises, shocks have fat tails: simulated forward from today's volatility."
    assumes = "Volatility is a persistent process that reverts to a long-run level, and the same rules hold in calm and in storm."
    fails = "A regime change (a crisis that is unlike the ones in the fit) is a surprise to it; the long-run level is estimated, and slowly."
    min_history = 600
    model, dist = "gjr", "t"

    def fit(self, h: History, horizons):
        r = pd.Series(h.r[-self.window:])
        if len(r) < 500 or float(r.std()) <= 0:
            raise ForecastError("too few, or constant, returns for a volatility model")
        try:
            g = fit_garch(r, self.model, self.dist, "zero")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            raise ForecastError(f"the volatility model could not be fitted: {error}") from None
        if not np.isfinite(g.loglik) or not np.isfinite(g.persistence):
            raise ForecastError("the volatility model did not converge")
        return g

    def _innovations(self, g):
        return None

    def predict(self, state, h: History, horizons, n, rng):
        g = state
        sims = g.simulate(max(horizons), n, seed=int(rng.integers(2 ** 31 - 1)), returns=h.r[-self.lookback:], innovations=self._innovations(g))
        p = g.params
        long_run = float(np.sqrt(g.unconditional_variance)) if np.isfinite(g.unconditional_variance) else float("nan")
        info = {"alpha": p["alpha"], "beta": p["beta"], "gamma": p.get("gamma", 0.0), "persistence": g.persistence, "half_life_days": g.half_life, "tail_degrees_of_freedom": p.get("nu"),
                "volatility_tomorrow_annual": float(sims[:, 0].std() * np.sqrt(TRADING_DAYS)), "long_run_volatility_annual": long_run * np.sqrt(TRADING_DAYS)}
        return Forecast(cumulate(sims), info)


class FilteredHistorical(GarchMC):
    name = "fhs"
    short = "Filtered hist. sim."
    label = "Filtered historical simulation (GARCH volatility, resampled shocks)"
    family = "Monte Carlo"
    text = "The same volatility model, but the shocks are the stock's own past standardised residuals, drawn at random, so the tails are the ones it really had."
    assumes = "The volatility model is right and the shocks, once scaled by it, are independent draws from one fixed distribution."
    fails = "A short history has few extreme shocks to draw on, so the far tail is thinner than it will be; the volatility model's errors are all inherited."
    dist = "normal"                                    # quasi-maximum-likelihood: the shape of the shocks comes from the residuals

    def _innovations(self, g):
        return g.std_resid.to_numpy()


class JumpDiffusion(Forecaster):
    name = "jump"
    short = "Jump diffusion"
    label = "Jump diffusion (Monte Carlo)"
    family = "Monte Carlo"
    text = "Ordinary small days plus rare large ones: the rate and size of the large ones are read from the last five years."
    assumes = "Large moves are surprises that arrive at random at a steady rate; between them volatility is constant."
    fails = "Large moves cluster in time in a real market, and this model spreads them evenly, so it is too wide in calm times and too narrow in storms."
    min_history = 500
    THRESHOLD = 3.5                                    # a day is a jump when it is this many robust standard deviations from the median

    def fit(self, h: History, horizons):
        r = h.r[-self.window:]
        if len(r) < 250:
            raise ForecastError("too few days to find the jumps")
        med = float(np.median(r))
        robust = 1.4826 * float(np.median(np.abs(r - med)))
        if robust <= 0:
            raise ForecastError("returns are constant")
        jump = np.abs(r - med) > self.THRESHOLD * robust
        calm = r[~jump]
        sigma = float(calm.std(ddof=1))
        k = int(jump.sum())
        lam = k / len(r)
        if k >= 5:
            size = r[jump]
            mu_j, sd_j = float(size.mean()), float(max(size.std(ddof=1), 1e-8))
        else:
            lam, mu_j, sd_j = 0.0, 0.0, 1e-8
        return {"sigma": sigma, "lam": lam, "mu_j": mu_j, "sd_j": sd_j, "n_jumps": k, "n_days": len(r)}

    def predict(self, state, h: History, horizons, n, rng):
        s = state
        steps = max(horizons)
        counts = rng.poisson(s["lam"], (n, steps))
        jumps = counts * s["mu_j"] + np.sqrt(counts) * s["sd_j"] * rng.standard_normal((n, steps))
        daily = -s["lam"] * s["mu_j"] + s["sigma"] * rng.standard_normal((n, steps)) + jumps          # the drift is set so that the average return is zero
        return Forecast(cumulate(daily), {"jumps_per_year": s["lam"] * TRADING_DAYS, "mean_jump": s["mu_j"], "jump_volatility": s["sd_j"], "calm_annual_volatility": s["sigma"] * np.sqrt(TRADING_DAYS),
                                          "days_called_jumps": s["n_jumps"], "days_looked_at": s["n_days"]})
