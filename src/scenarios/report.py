"""Scenario generators side by side: what each one assumes, how long it takes, how closely its scenarios resemble the history they were made from, and what they say about the portfolio's return over a horizon.

The Scenarios tab, ``/api/scenarios/*`` and ``quant scenarios`` are this module. It adds nothing to the generators of :mod:`src.scenarios.generators` or the scores of :mod:`src.scenarios.quality`; it runs
a chosen set of them on one history, keeps a generator's failure from ending the others, and turns the numbers into a plain record. The portfolio is equal-weighted: the point is to compare generators on the same
yardstick, not to say how to invest.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .generators import GENERATORS, generate
from .quality import quality_report

LEVELS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
MIN_DAYS = 500                                    # complete days of returns: the learned generators have nothing to learn from less
MAX_ASSETS = 12
HORIZONS = (5, 10, 21, 63)
N_LIMITS = (100, 2000)

# What the app passes the learned generators: smaller and shorter than the library defaults so that a run takes seconds, not minutes. Each is a legitimate setting; none is tuned to a data set.
PRESETS = {"wgan_gp": {"steps": 1200, "hidden": 64, "layers": 2, "lr": 1e-3, "latent": 8}, "factor_vae": {}, "diffusion": {"epochs": 250, "hidden": 96, "layers": 2}}

INFO = {
    "historical": {"label": "Historical windows", "family": "Resampling", "heavy": False,
                   "text": "Random stretches of consecutive days from the actual history.",
                   "assumes": "The future is made of stretches like those that have happened.", "fails": "Nothing worse than the worst stretch in the data can occur, and a short history has few distinct stretches."},
    "bootstrap": {"label": "Block bootstrap", "family": "Resampling", "heavy": False,
                  "text": "Days drawn at random from the history in linked blocks (stationary bootstrap, 20 days on average), whole days at a time so the assets keep moving together.",
                  "assumes": "Days are exchangeable apart from short-range dependence that a block keeps.", "fails": "Cannot make a day more extreme than one that has happened; volatility regimes longer than a block are lost."},
    "copula": {"label": "Copula", "family": "Statistical model", "heavy": False,
               "text": "Each asset's own return distribution joined by a Student-t copula, drawn independently day by day.",
               "assumes": "Days are independent; the dependence between assets is the same in every state.", "fails": "No volatility clustering: a calm month and a crisis month look alike."},
    "risk_factor": {"label": "Risk factors", "family": "Statistical model", "heavy": False,
                    "text": "The first three principal components bootstrapped in blocks, plus each asset's own leftover noise.",
                    "assumes": "A few common factors carry the dependence, and the leftover is independent between assets.", "fails": "Correlations among the leftovers are dropped, so a pair's correlation can come out wrong."},
    "arima_garch": {"label": "AR(1)-GARCH", "family": "Statistical model", "heavy": False,
                    "text": "Each asset follows an AR(1) mean and a GJR-GARCH volatility, driven by shocks resampled jointly across assets.",
                    "assumes": "Volatility follows the fitted GARCH process and the shocks are as the history had them.", "fails": "A model of the past: a regime the GARCH has not seen is not in the scenarios."},
    "wgan_gp": {"label": "GAN (WGAN-GP)", "family": "Learned", "heavy": True,
                "text": "A Wasserstein generative adversarial network with gradient penalty, trained on the days of the history, and its draws.",
                "assumes": "A small network can learn the joint distribution of one day's returns from a few years of days.", "fails": "Adversarial training is unstable: it can miss the tails and the spread, which the quality table shows. Days are drawn independently, so there is no clustering."},
    "factor_vae": {"label": "Factor VAE", "family": "Learned", "heavy": True,
                   "text": "A variational autoencoder with a factor structure and Student-t noise, trained on the days of the history, and its draws.",
                   "assumes": "The dependence between the assets is carried by a few factors, with fat-tailed noise on top.", "fails": "Days are drawn independently, so there is no clustering; a regime with different factors is not in the scenarios."},
    "diffusion": {"label": "Diffusion model", "family": "Learned", "heavy": True,
                  "text": "A denoising diffusion model trained on the days of the history; a scenario day is noise denoised step by step.",
                  "assumes": "A small network can learn the joint distribution of one day's returns from a few years of days.", "fails": "Slow to fit, and a short history gives it little to learn; days are drawn independently, so there is no clustering."},
}
DEFAULT = ["bootstrap", "arima_garch", "factor_vae"]
FAMILIES = ["Resampling", "Statistical model", "Learned"]

# How each quality number reads. ``target`` is the value of a perfect copy; ``tolerance`` is how far from it the page stops calling it close (a rule of thumb, not a test).
METRICS = {
    "marginal_ks": {"label": "Return distributions", "target": 0.0, "tolerance": 0.10, "text": "Average over the assets of the largest gap between the generated and the real distribution of a day's return (Kolmogorov-Smirnov). 0 is a perfect match."},
    "vol_error": {"label": "Volatility", "target": 0.0, "tolerance": 0.25, "text": "The worst asset's relative error in the size of a day's moves. 0.10 means 10% too much or too little."},
    "corr_error": {"label": "Correlations", "target": 0.0, "tolerance": 0.15, "text": "The largest gap in any pair's correlation. Only with two or more assets."},
    "tail_dep_error": {"label": "Crashing together", "target": 0.0, "tolerance": 0.15, "text": "The gap in how often the assets are all in their worst 5% of days at the same time."},
    "kurtosis_ratio": {"label": "Fat tails", "target": 1.0, "tolerance": 0.35, "text": "Generated kurtosis over real kurtosis, averaged over the assets. 1 is the same tails; below 1, thinner tails than the history."},
    "clustering_error": {"label": "Volatility clustering", "target": 0.0, "tolerance": 0.05, "text": "The gap in the autocorrelation of absolute returns, the footprint of calm and stormy spells. Only when the horizon is more than a day."},
    "es_ratio": {"label": "Worst-5% loss", "target": 1.0, "tolerance": 0.25, "text": "The portfolio's average loss on its worst 5% of days, generated over real. Below 1, the scenarios understate the tail."},
    "horizon_vol_ratio": {"label": "Horizon spread", "target": 1.0, "tolerance": 0.25, "text": "The spread of the portfolio's return over the whole horizon, generated over real (the real one from non-overlapping stretches). Below 1, too calm over the horizon."},
}


class ScenarioError(ValueError):
    """The request cannot be run: say what to change."""


def check(returns: pd.DataFrame, names, n: int, horizon: int) -> list[str]:
    """The generators to run, or a ``ScenarioError`` that says what is wrong with the request."""
    names = list(dict.fromkeys(names))
    if not names:
        raise ScenarioError("choose at least one generator")
    unknown = [x for x in names if x not in GENERATORS]
    if unknown:
        raise ScenarioError(f"unknown generator {unknown[0]!r}; choose from {', '.join(GENERATORS)}")
    if not N_LIMITS[0] <= n <= N_LIMITS[1]:
        raise ScenarioError(f"the number of scenarios must be between {N_LIMITS[0]} and {N_LIMITS[1]}")
    if not 1 <= horizon <= 126:
        raise ScenarioError("the horizon must be between 1 and 126 trading days")
    if returns.shape[1] < 1 or returns.shape[1] > MAX_ASSETS:
        raise ScenarioError(f"use between 1 and {MAX_ASSETS} assets")
    if len(returns.dropna()) < max(MIN_DAYS, 4 * horizon):
        raise ScenarioError(f"scenarios need at least {max(MIN_DAYS, 4 * horizon)} days of returns for every asset at once; this has {len(returns.dropna())}. Use an earlier start date or fewer assets")
    return names


def _terminal(R: np.ndarray, w: np.ndarray, horizon: int) -> np.ndarray:
    """The portfolio's return over every stretch of ``horizon`` days of ``R`` (overlapping)."""
    log = np.log1p(np.clip(R @ w, -0.999, None))
    c = np.concatenate([[0.0], np.cumsum(log)])
    return np.expm1(c[horizon:] - c[:-horizon])


def _summary(x: np.ndarray) -> dict:
    x = np.sort(np.asarray(x, dtype=float))
    k = max(int(0.05 * len(x)), 1)
    return {"quantiles": [float(v) for v in np.quantile(x, LEVELS)], "mean": float(x.mean()), "std": float(x.std()), "var5": float(-np.quantile(x, 0.05)), "es5": float(-x[:k].mean()), "count": int(len(x))}


def _flag(key: str, value: float) -> str:
    m = METRICS[key]
    if not np.isfinite(value):
        return "n/a"
    return "close" if abs(value - m["target"]) <= m["tolerance"] else "far"


def run(returns: pd.DataFrame, names, n: int = 500, horizon: int = 21, seed: int = 0, progress=None, stop=None) -> dict:
    """Generate ``n`` scenarios of ``horizon`` days with each generator in ``names`` from ``returns`` (daily simple returns, one column per asset) and score them against the history."""
    R = returns.dropna()
    names = check(R, names, n, horizon)
    A = R.to_numpy(dtype=float)
    w = np.full(A.shape[1], 1.0 / A.shape[1])
    out, notes = [], []
    for i, name in enumerate(names):
        if stop is not None and stop():
            notes.append(f"Stopped before {INFO[name]['label']}; the generators before it are shown.")
            break
        if progress is not None:
            progress(f"{INFO[name]['label']} ({i + 1} of {len(names)})" + (": training the network" if INFO[name]["heavy"] else ""))
        started = time.time()
        row = {"id": name, "label": INFO[name]["label"], "family": INFO[name]["family"], "error": None, "quality": None, "flags": None, "terminal": None}
        try:
            S = generate(name, R, n, horizon, seed, **PRESETS.get(name, {}))
            if not np.all(np.isfinite(S)):
                raise ValueError("it produced values that are not finite")
            q = quality_report(R, S, w)
            row.update(quality={k: float(v) for k, v in q.items()}, flags={k: _flag(k, v) for k, v in q.items()}, terminal=_summary(np.expm1(np.log1p(np.clip(S @ w, -0.999, None)).sum(axis=1))))
        except Exception as error:                                                                    # a generator that fails is reported, and the others still run
            row["error"] = str(error) or type(error).__name__
            notes.append(f"{INFO[name]['label']} could not make scenarios: {row['error']}")
        row["seconds"] = round(time.time() - started, 2)
        out.append(row)
    if not out:
        raise ScenarioError("stopped before the first generator finished")
    if not any(r["error"] is None for r in out):
        raise ScenarioError("no generator could make scenarios: " + "; ".join(r["error"] for r in out if r["error"]))
    history = _summary(_terminal(A, w, horizon))
    notes.append(f"The history is {len(R)} days of daily returns from {R.index[0].date() if hasattr(R.index[0], 'date') else R.index[0]}; its horizon figures come from overlapping {horizon}-day stretches, so they are less independent than their count suggests.")
    notes.append("The portfolio is equal-weighted across the assets and rebalanced every day. A scenario set that matches the past says nothing about what comes next; it is a model of what has happened.")
    return {"assets": list(R.columns), "days": int(len(R)), "first": str(R.index[0])[:10], "last": str(R.index[-1])[:10], "horizon": int(horizon), "n": int(n), "seed": int(seed), "levels": list(LEVELS),
            "generators": out, "history": history, "metrics": [{"id": k, **v} for k, v in METRICS.items()], "notes": notes}


def catalog() -> list[dict]:
    return [{"id": k, **INFO[k]} for k in GENERATORS]
