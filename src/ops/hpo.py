"""Hyperparameter optimisation that counts its own trials and reports how much of the winner is selection.

Search methods (``Study.optimize``): ``random``, ``grid``, ``tpe`` (a Tree-structured Parzen Estimator after Bergstra et al. 2011: after a random start, model the good and the bad trials with Parzen
densities and propose the candidate maximising ``l(x) / g(x)``; ``multivariate=True`` models the settings jointly, as Falkner et al. 2018 and Optuna's multivariate TPE do, so that settings that only
matter together, such as a noise level and the scale applied to it, are found together; ``quasi_start=True`` begins with a scrambled Sobol sequence, which covers the space more evenly than random draws),
``bayes`` (a Gaussian process with a Matern 5/2 kernel and a noise term, and the expected improvement over the best *fitted* value, so that a lucky noisy trial is not chased) and ``halving`` (successive
halving, Jamieson & Talwalkar 2016: many configurations on a small budget, the best third promoted).

After a search, ``Study.plateau`` names the trial whose *neighbourhood* did best (a kernel-smoothed score, shrunk towards the mean where few trials are near), which is a sturdier choice than the single
best score of a noisy objective, and ``Study.centroid`` names the settings at the middle of the best trials.

The part that matters for finance is what happens AFTER the search. The best of ``N`` trials is biased upward, so ``Study.selection_report`` computes, from the trials' own return streams:

* the **deflated Sharpe ratio** of the winner with the Sharpe variance estimated from ALL trials (``stats.deflated_sharpe_ratio``) and the number of trials actually run;
* the **probability of backtest overfitting** by combinatorially symmetric cross-validation (``validation.pbo_cscv``): how often the in-sample winner falls below the out-of-sample median;
* the **Romano-Wolf** family-wise-error p-values (``stats.romano_wolf``): which trials survive after accounting for the others.

``tune_pipeline`` applies this to a research pipeline: the objective is the net Sharpe ratio of the run (optionally averaged over contiguous blocks, which penalises results that depend on one sub-period),
every trial is a separate ``Pipeline`` run, and the returns are kept for the selection report. A study that hides its trial count is how overfit strategies get published; this one cannot.
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Param:
    kind: str                       # "int" | "float" | "log" | "categorical"
    low: float | None = None
    high: float | None = None
    choices: tuple = ()

    def sample(self, rng: np.random.Generator):
        if self.kind == "categorical":
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.kind == "int":
            return int(rng.integers(int(self.low), int(self.high) + 1))
        if self.kind == "log":
            return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def grid(self, n: int) -> list:
        if self.kind == "categorical":
            return list(self.choices)
        if self.kind == "int":
            return sorted({int(round(v)) for v in np.linspace(self.low, self.high, n)})
        if self.kind == "log":
            return list(np.exp(np.linspace(np.log(self.low), np.log(self.high), n)))
        return list(np.linspace(self.low, self.high, n))

    def to_unit(self, value) -> float:
        if self.kind == "categorical":
            return (list(self.choices).index(value) + 0.5) / len(self.choices)                  # the middle of the choice's slot
        if self.kind == "log":
            return (np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        return (float(value) - self.low) / (self.high - self.low) if self.high != self.low else 0.5

    def from_unit(self, u: float):
        """The value at position ``u`` of ``[0, 1]`` in this range (a whole number for ``int``, the choice whose slot holds ``u`` for ``categorical``)."""
        u = min(max(float(u), 0.0), 1.0)
        if self.kind == "categorical":
            return self.choices[min(int(u * len(self.choices)), len(self.choices) - 1)]
        if self.kind == "int":
            return int(min(max(round(self.low + u * (self.high - self.low)), self.low), self.high))
        if self.kind == "log":
            return float(np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low))))
        return float(self.low + u * (self.high - self.low))


def Int(low, high): return Param("int", low, high)                                        # noqa: E704, N802
def Float(low, high): return Param("float", low, high)                                    # noqa: E704, N802
def LogFloat(low, high): return Param("log", low, high)                                   # noqa: E704, N802
def Categorical(*choices): return Param("categorical", choices=tuple(choices))            # noqa: E704, N802


@dataclass
class Trial:
    number: int
    params: dict
    value: float
    returns: pd.Series | None = None
    seconds: float = 0.0
    budget: float | None = None
    error: str | None = None


@dataclass
class Study:
    space: dict
    direction: str = "maximize"
    seed: int = 0
    trials: list = field(default_factory=list)
    _sobol: object = field(default=None, repr=False, compare=False)                  # the scrambled Sobol sequence that quasi-random starts draw from, and the points not yet used
    _sobol_buffer: list = field(default_factory=list, repr=False, compare=False)

    # ------------------------------------------------------------------------------------------------------------------ accounting
    @property
    def n_trials(self) -> int:
        """Every configuration evaluated counts, failed ones included (a failed run is still a look at the data)."""
        return len(self.trials)

    def _sign(self) -> float:
        return 1.0 if self.direction == "maximize" else -1.0

    @property
    def best(self) -> Trial:
        ok = [t for t in self.trials if t.error is None and np.isfinite(t.value)]
        if not ok:
            raise ValueError("no successful trial")
        return max(ok, key=lambda t: self._sign() * t.value)

    def plateau(self, bandwidth: float = 0.2, prior_weight: float = 1.5) -> Trial:
        """The trial whose neighbourhood did best. Each trial's score is replaced by the average over the trials near it (a Gaussian in the distance between settings, in units of each setting's range, with
        ``bandwidth`` the typical distance), shrunk towards the average of all trials by ``prior_weight`` pseudo-trials: a spike with nothing near it is pulled down, and a broad flat top is not. On a
        noisy objective the best single score is mostly luck; the middle of the best plateau is the sturdier choice."""
        ok = self._done()
        if not ok:
            raise ValueError("no successful trial")
        keys = list(self.space)
        y = np.array([self._sign() * t.value for t in ok])
        sq = np.zeros((len(ok), len(ok)))
        for k in keys:
            u = np.array([self._unit(k, t.params[k]) for t in ok])
            diff = u[:, None] - u[None, :]
            sq += (diff != 0.0).astype(float) if self.space[k].kind == "categorical" else diff ** 2
        w = np.exp(-0.5 * (np.sqrt(sq / len(keys)) / bandwidth) ** 2)
        smooth = (w @ y + prior_weight * y.mean()) / (w.sum(axis=1) + prior_weight)
        return ok[int(np.argmax(smooth))]

    def centroid(self, top_fraction: float = 0.15) -> dict:
        """The settings at the middle of the best trials (the best ``top_fraction`` of them, at least three): the mean of a number (the geometric mean on a log scale, rounded for a whole number) and the most
        common choice of a category. It need not be a configuration that was run."""
        ok = sorted(self._done(), key=lambda t: -self._sign() * t.value)
        if not ok:
            raise ValueError("no successful trial")
        group = ok[: max(3, int(np.ceil(top_fraction * len(ok))))]
        out = {}
        for k, p in self.space.items():
            vals = [t.params[k] for t in group]
            if p.kind == "categorical":
                out[k] = max(p.choices, key=lambda c: vals.count(c))
            elif p.kind == "log":
                out[k] = float(np.exp(np.mean(np.log(vals))))
            elif p.kind == "int":
                out[k] = int(round(float(np.mean(vals))))
            else:
                out[k] = float(np.mean(vals))
        return out

    def to_frame(self) -> pd.DataFrame:
        rows = [{"number": t.number, **t.params, "value": t.value, "seconds": t.seconds, "budget": t.budget, "error": t.error} for t in self.trials]
        return pd.DataFrame(rows).set_index("number") if rows else pd.DataFrame()

    # ----------------------------------------------------------------------------------------------------------------- proposals
    def _random(self, rng):
        return {k: p.sample(rng) for k, p in self.space.items()}

    def _done(self) -> list:
        return [t for t in self.trials if t.error is None and np.isfinite(t.value)]

    def _unit(self, key, value) -> float:
        return float(self.space[key].to_unit(value))

    def _decode(self, u) -> dict:
        return {k: self.space[k].from_unit(u[j]) for j, k in enumerate(self.space)}

    def _sobol_point(self) -> dict:
        """The next point of a scrambled Sobol sequence (it covers the space more evenly than random draws, which leave gaps and clumps in the few trials a search starts with)."""
        if not self._sobol_buffer:
            if self._sobol is None:
                self._sobol = _sobol_engine(len(self.space), self.seed)
            self._sobol_buffer = list(self._sobol.random_base2(m=5))                       # 32 points at a time: Sobol' balance needs a power of two
        return self._decode(self._sobol_buffer.pop(0))

    def _tpe(self, rng, n_startup: int = 10, gamma: float = 0.25, n_candidates: int = 32, multivariate: bool = False, quasi_start: bool = False):
        done = self._done()
        if len(done) < n_startup:
            return self._sobol_point() if quasi_start else self._random(rng)
        done.sort(key=lambda t: -self._sign() * t.value)
        n_good = max(2, int(np.ceil(gamma * len(done))))
        good, bad = done[:n_good], done[n_good:]
        if multivariate:
            return self._tpe_joint(rng, good, bad, max(n_candidates, 48))
        candidates = []
        for _ in range(n_candidates):
            candidates.append({k: self._sample_from(k, [t.params[k] for t in good], rng) for k in self.space})
        scores = []
        for c in candidates:
            lg = sum(self._log_density(k, c[k], [t.params[k] for t in good]) for k in self.space)
            lb = sum(self._log_density(k, c[k], [t.params[k] for t in bad]) for k in self.space)
            scores.append(lg - lb)
        return candidates[int(np.argmax(scores))]

    # ---------------------------------------------------------------------------------------------------- a TPE that models the settings together
    def _joint_fit(self, group: list) -> tuple[np.ndarray, np.ndarray]:
        """The units of each setting in ``group`` (n x d) and the width of the Gaussian around each: the larger gap to the neighbouring values of that setting, as in ``_parzen``, per setting."""
        keys = list(self.space)
        U = np.array([[self._unit(k, t.params[k]) for k in keys] for t in group])
        S = np.zeros_like(U)
        for j, k in enumerate(keys):
            if self.space[k].kind == "categorical":
                continue
            order = np.argsort(U[:, j])
            ext = np.concatenate([[0.0], U[order, j], [1.0]])
            width = np.maximum(ext[1:-1] - ext[:-2], ext[2:] - ext[1:-1])
            S[order, j] = np.clip(width, 1.0 / min(100.0, len(group) + 1.0), 1.0)
        return U, S

    def _joint_logpdf(self, C: np.ndarray, U: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Log density of each candidate (m x d, in units) under an equal mixture of one product kernel per observation and a uniform prior: a truncated normal for a number, ``1 - e`` for the
        observation's own choice and ``e / (k - 1)`` for another (``e`` is 0.25)."""
        from scipy.special import logsumexp, ndtr

        keys = list(self.space)
        logk = np.zeros((len(C), len(U)))
        prior = np.zeros(len(C))
        for j, k in enumerate(keys):
            p = self.space[k]
            if p.kind == "categorical":
                n_choices = len(p.choices)
                same = np.isclose(C[:, j][:, None], U[:, j][None, :])
                logk += np.where(same, np.log(0.75), np.log(0.25 / max(n_choices - 1, 1)))
                prior += np.log(1.0 / n_choices)
            else:
                sd = S[:, j][None, :]
                z = (C[:, j][:, None] - U[:, j][None, :]) / sd
                mass = np.maximum(ndtr((1.0 - U[:, j]) / S[:, j]) - ndtr((0.0 - U[:, j]) / S[:, j]), 1e-12)[None, :]
                logk += -0.5 * z ** 2 - np.log(sd * np.sqrt(2.0 * np.pi)) - np.log(mass)
        return logsumexp(np.concatenate([logk, prior[:, None]], axis=1), axis=1) - np.log(len(U) + 1.0)

    def _tpe_joint(self, rng, good: list, bad: list, n_candidates: int) -> dict:
        keys = list(self.space)
        Ug, Sg = self._joint_fit(good)
        cands = np.empty((n_candidates, len(keys)))
        for c in range(n_candidates):
            i = int(rng.integers(len(good) + 1))                                              # one component, all settings from it: the settings are drawn together
            for j, k in enumerate(keys):
                p = self.space[k]
                if i == len(good):
                    cands[c, j] = rng.random()
                elif p.kind == "categorical":
                    n_choices = len(p.choices)
                    slot = int(Ug[i, j] * n_choices) if rng.random() < 0.75 else int(rng.integers(n_choices))
                    cands[c, j] = (min(slot, n_choices - 1) + 0.5) / n_choices
                else:
                    for _ in range(50):
                        u = rng.normal(Ug[i, j], Sg[i, j])
                        if 0.0 <= u <= 1.0:
                            break
                    else:
                        u = float(np.clip(u, 0.0, 1.0))
                    cands[c, j] = u
        params = [self._decode(c) for c in cands]
        C = np.array([[self._unit(k, q[k]) for k in keys] for q in params])                   # where each candidate really is after whole numbers and choices are applied
        score = self._joint_logpdf(C, Ug, Sg)
        if bad:
            Ub, Sb = self._joint_fit(bad)
            score = score - self._joint_logpdf(C, Ub, Sb)
        return params[int(np.argmax(score))]

    # ------------------------------------------------------------------------------------------------------------ a Gaussian process
    def _encode(self, params: dict) -> np.ndarray:
        """A setting's place for the Gaussian process: its unit for a number, one 0/1 column per choice for a category."""
        out = []
        for k, p in self.space.items():
            if p.kind == "categorical":
                out += [1.0 if params[k] == c else 0.0 for c in p.choices]
            else:
                out.append(self._unit(k, params[k]))
        return np.array(out)

    def _bayes(self, rng, n_startup: int = 8, n_candidates: int = 1200, xi: float = 0.0, quasi_start: bool = True, max_fit: int = 150) -> dict:
        """Fit a Gaussian process to the trials (Matern 5/2, a length scale per setting, a noise term) and try the candidate with the greatest expected improvement over the best *fitted* value: the process
        smooths a noisy objective, so one lucky trial does not set the bar. The cost of a fit grows with the cube of the number of trials, so past ``max_fit`` of them (and when a fit fails on degenerate
        data) the proposal comes from the Parzen estimator instead."""
        done = self._done()
        if len(done) < n_startup:
            return self._sobol_point() if quasi_start else self._random(rng)
        if len(done) > max_fit:
            return self._tpe(rng, n_startup=n_startup)
        try:
            return self._bayes_fit(rng, done, n_candidates, xi)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return self._tpe(rng, n_startup=n_startup)

    def _bayes_fit(self, rng, done: list, n_candidates: int, xi: float) -> dict:
        from scipy.special import ndtr
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        keys = list(self.space)
        X = np.array([self._encode(t.params) for t in done])
        y = np.array([self._sign() * t.value for t in done])
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.full(X.shape[1], 0.3), length_scale_bounds=(0.03, 5.0), nu=2.5) + WhiteKernel(0.05, (1e-4, 1.0))
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=int(rng.integers(2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                                                  # a kernel parameter at its bound is a finding about the data, not a fault
            gp.fit(X, y)
        m = int(np.ceil(np.log2(max(n_candidates * 0.7, 8))))
        points = [self._decode(u) for u in _sobol_engine(len(keys), int(rng.integers(2**31 - 1))).random_base2(m)]
        for t in sorted(done, key=lambda t: -self._sign() * t.value)[:5]:                    # and a cloud around each of the best trials, to refine
            u0 = np.array([self._unit(k, t.params[k]) for k in keys])
            points += [self._decode(np.clip(u0 + rng.normal(0.0, 0.06, len(keys)), 0.0, 1.0)) for _ in range(40)]
        used = {tuple(t.params.get(k) for k in keys) for t in self.trials}
        points = [q for q in points if tuple(q[k] for k in keys) not in used] or points
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sd = gp.predict(np.array([self._encode(q) for q in points]), return_std=True)
            best = float(gp.predict(X).max())
        gain = mu - best - xi
        z = gain / np.maximum(sd, 1e-9)
        ei = gain * ndtr(z) + sd * np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
        return points[int(np.argmax(ei))]

    def _propose(self, rng, method: str, **kwargs) -> dict:
        if method == "tpe":
            return self._tpe(rng, **kwargs)
        if method == "bayes":
            return self._bayes(rng, **kwargs)
        return self._random(rng)

    @staticmethod
    def _parzen(units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The adaptive Parzen estimator of hyperopt: one Gaussian per observation PLUS a broad prior component at the centre of the range. Each bandwidth is the larger distance to the
        neighbouring points (the range ends count as neighbours), clipped to ``[1 / min(100, n + 1), 1]``; the prior keeps the search exploring after the good points have clustered."""
        pts = np.sort(np.append(units, 0.5))
        ext = np.concatenate([[0.0], pts, [1.0]])
        sigma = np.maximum(ext[1:-1] - ext[:-2], ext[2:] - ext[1:-1])
        sigma = np.clip(sigma, 1.0 / min(100.0, len(units) + 1.0), 1.0)
        prior = int(np.flatnonzero(pts == 0.5)[0]) if (pts == 0.5).any() else 0
        sigma[prior] = 1.0
        return pts, sigma

    def _sample_from(self, key, observed, rng):
        p = self.space[key]
        if p.kind == "categorical":
            counts = np.array([observed.count(c) for c in p.choices], dtype=float) + 1.0              # add-one smoothing keeps every choice possible
            return p.choices[int(rng.choice(len(p.choices), p=counts / counts.sum()))]
        pts, sigma = self._parzen(np.array([p.to_unit(v) for v in observed]))
        i = int(rng.integers(len(pts)))
        for _ in range(50):                                                                          # truncated normal on [0, 1] by rejection
            u = rng.normal(pts[i], sigma[i])
            if 0.0 <= u <= 1.0:
                break
        else:
            u = float(np.clip(u, 0.0, 1.0))
        if p.kind == "log":
            return float(np.exp(np.log(p.low) + u * (np.log(p.high) - np.log(p.low))))
        v = p.low + u * (p.high - p.low)
        return int(round(v)) if p.kind == "int" else float(v)

    def _log_density(self, key, value, observed) -> float:
        p = self.space[key]
        if p.kind == "categorical":
            counts = np.array([observed.count(c) for c in p.choices], dtype=float) + 1.0
            return float(np.log(counts[list(p.choices).index(value)] / counts.sum()))
        pts, sigma = self._parzen(np.array([p.to_unit(v) for v in observed]))
        u = p.to_unit(value)
        dens = np.mean(np.exp(-0.5 * ((u - pts) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))) + 1e-12
        return float(np.log(dens))

    # ---------------------------------------------------------------------------------------------------------------- optimisation
    def _run(self, objective, params, budget=None) -> Trial:
        t0 = time.perf_counter()
        number = len(self.trials)
        try:
            out = objective(params) if budget is None else objective(params, budget)
            value, rets = (out if isinstance(out, tuple) else (out, None))
            trial = Trial(number, params, float(value), rets, time.perf_counter() - t0, budget)
        except Exception as exc:                              # a failing configuration is recorded, counted and skipped, not allowed to stop the search
            trial = Trial(number, params, float("nan"), None, time.perf_counter() - t0, budget, f"{type(exc).__name__}: {exc}")
        self.trials.append(trial)
        return trial

    def _fresh(self, rng, method: str, **kwargs) -> dict | None:
        """The next configuration to try that no earlier trial used (a small integer or categorical space repeats itself otherwise, and a repeat is a wasted run that is counted as a trial); ``None`` once
        the space is as good as exhausted."""
        used = {tuple(t.params.get(k) for k in self.space) for t in self.trials}
        for attempt in range(40):
            params = self._propose(rng, method, **kwargs) if method != "random" and attempt < 8 else self._random(rng)
            if tuple(params[k] for k in self.space) not in used:
                return params
        return None

    def optimize(self, objective: Callable, n_trials: int = 30, method: str = "random", grid_points: int = 4, callback: Callable | None = None, unique: bool = False, **kwargs) -> "Study":
        """Run the search. ``objective(params)`` returns a number or ``(number, returns_series)``; for ``halving`` it is ``objective(params, budget)``. ``callback(study, trial)``, when given, is
        called after every trial of a ``random``, ``tpe`` or ``grid`` search; returning ``True`` stops the search there (a time limit, a Stop button): the trials run so far are kept and counted.
        With ``unique`` a ``random`` or ``tpe`` search never repeats a configuration (earlier trials count, so a first trial run by hand is not repeated) and ends early when the space is used up."""
        rng = np.random.default_rng(self.seed + len(self.trials))
        if method == "grid":
            axes = [self.space[k].grid(grid_points) for k in self.space]
            for combo in product(*axes):
                trial = self._run(objective, dict(zip(self.space, combo)))
                if callback is not None and callback(self, trial):
                    break
        elif method in ("random", "tpe", "bayes"):
            for _ in range(n_trials):
                if unique:
                    params = self._fresh(rng, method, **kwargs)
                    if params is None:
                        break
                else:
                    params = self._propose(rng, method, **kwargs)
                trial = self._run(objective, params)
                if callback is not None and callback(self, trial):
                    break
        elif method == "halving":
            self._halving(objective, n_trials, rng, **kwargs)
        else:
            raise ValueError("method must be random, grid, tpe, bayes or halving")
        return self

    def _halving(self, objective, n_configs, rng, min_budget: float = 0.2, max_budget: float = 1.0, eta: int = 3):
        configs = [self._random(rng) for _ in range(n_configs)]
        budget = min_budget
        while configs:
            trials = [self._run(objective, c, budget) for c in configs]
            if budget >= max_budget or len(configs) == 1:
                break
            ranked = sorted(trials, key=lambda t: -self._sign() * (t.value if np.isfinite(t.value) else -np.inf * self._sign()))
            configs = [t.params for t in ranked[: max(1, len(configs) // eta)]]
            budget = min(budget * eta, max_budget)

    # ---------------------------------------------------------------------------------------------------------------- selection
    def returns_matrix(self) -> pd.DataFrame:
        """Trial return streams side by side (trials without returns are skipped)."""
        cols = {t.number: t.returns for t in self.trials if t.returns is not None and t.error is None}
        return pd.DataFrame(cols).dropna(how="all")

    def selection_report(self, periods_per_year: int = 252, n_boot: int = 500, pbo_blocks: int = 8, winner: int | None = None) -> dict:
        """How much of the winner is selection? Needs the return streams of the trials. ``winner`` is the number of the trial that was chosen when it is not the one with the best score (a rule such as
        ``plateau`` chose another); the deflated Sharpe ratio and the Romano-Wolf p-value are then the chosen trial's."""
        from ..stats.inference import deflated_sharpe_ratio, romano_wolf
        from ..validation.multiple_testing import pbo_cscv

        mat = self.returns_matrix().dropna()
        if mat.shape[1] < 2:
            raise ValueError("selection_report needs at least two trials with return streams")
        sharpes = mat.mean() / mat.std(ddof=1)
        best = int(winner) if winner is not None and winner in mat.columns else int(self.best.number) if self.best.number in mat.columns else int(sharpes.idxmax())
        n_eff, rho = effective_trials(mat)
        dsr = deflated_sharpe_ratio(mat[best].to_numpy(), n_trials=n_eff)
        classic = deflated_sharpe_ratio(mat[best].to_numpy(), trial_sharpes=sharpes.to_numpy(), n_trials=self.n_trials)
        pbo = pbo_cscv(mat.to_numpy(), blocks=pbo_blocks, annualise=periods_per_year)
        rw = romano_wolf(mat, n_boot=n_boot, seed=self.seed)
        return {"best_trial": best, "best_annual_sharpe": float(sharpes[best] * np.sqrt(periods_per_year)), "n_trials": self.n_trials, "n_with_returns": int(mat.shape[1]),
                "deflated_sharpe_probability": dsr["probability"], "deflated_benchmark_sharpe_annual": float(dsr["benchmark"] * np.sqrt(periods_per_year)),
                "effective_trials": n_eff, "mean_trial_correlation": rho, "deflated_sharpe_probability_classic": classic["probability"],
                "pbo": pbo["pbo"], "mean_oos_sharpe_of_is_winner": pbo["mean_oos_of_best"], "romano_wolf_p_best": float(rw.loc[best, "p_romano_wolf"]),
                "survivors_at_5pct": int((rw["p_romano_wolf"] < 0.05).sum())}


def _sobol_engine(d: int, seed: int):
    """A scrambled Sobol sequence in ``d`` dimensions. SciPy renamed the ``seed`` argument to ``rng`` (1.15); either spelling works here."""
    from scipy.stats import qmc

    try:
        return qmc.Sobol(d=d, scramble=True, rng=seed)
    except TypeError:
        return qmc.Sobol(d=d, scramble=True, seed=seed)


def effective_trials(mat: pd.DataFrame) -> tuple[float, float]:
    """``(N_eff, rho)``: the number of *independent* tries that ``mat``'s columns (the tries' return streams) amount to, and the average correlation between two of them. Tries of settings that sit
    next to each other trade almost the same days, so forty of them are worth far fewer than forty independent looks: ``N_eff = rho + (1 - rho) N`` (Bailey & Lopez de Prado 2014). It is what the
    deflated Sharpe ratio asks the winner to beat, with the sampling variance of one Sharpe ratio (``1 / T``) as the spread: the spread across the tries themselves would count a real difference
    between settings as noise, which on a search where the settings matter hid every real result. At least two."""
    n = mat.shape[1]
    if n < 3:
        return float(max(n, 2)), 0.0
    c = np.corrcoef(mat.to_numpy().T)
    rho = float(np.clip((np.nansum(c) - n) / (n * (n - 1)), 0.0, 1.0))
    return float(max(2.0, rho + (1.0 - rho) * n)), rho


def _block_sharpe(returns: pd.Series, n_blocks: int, periods_per_year: int) -> np.ndarray:
    r = returns.dropna()
    edges = np.linspace(0, len(r), n_blocks + 1).astype(int)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = r.iloc[lo:hi]
        sd = seg.std(ddof=1)
        out.append(seg.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0)
    return np.asarray(out)


def tune_pipeline(spec: dict, bundle, config, space: dict[str, Param], n_trials: int = 20, method: str = "random", n_blocks: int = 1, block_penalty: float = 0.5, seed: int = 0,
                  validate: bool = False) -> Study:
    """Tune a pipeline specification. ``space`` maps DOTTED paths into the spec dict (``"models.0.params.lookback"``, ``"allocation.params.fraction"``) to distributions. Each trial is a full
    ``Pipeline`` run on ``bundle``; its value is the net Sharpe ratio, or with ``n_blocks > 1`` the mean of the block Sharpe ratios minus ``block_penalty`` times their standard deviation (a result
    that depends on one sub-period scores badly). The trials' net returns are kept so ``study.selection_report()`` can deflate the winner."""
    from ..framework import Pipeline, PipelineSpec

    def build(params):
        d = copy.deepcopy(spec)
        for path, value in params.items():
            node = d
            parts = path.split(".")
            for p in parts[:-1]:
                node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
            last = parts[-1]
            if isinstance(node, list):
                node[int(last)] = value
            else:
                node[last] = value
        return d

    def objective(params):
        result = Pipeline(PipelineSpec.from_dict(build(params)), config, bundle).run(validate=validate)
        r = result.window
        if n_blocks > 1:
            b = _block_sharpe(r, n_blocks, 252)
            value = float(b.mean() - block_penalty * b.std(ddof=1))
        else:
            value = float(result.metrics.get("sharpe", np.nan))
        return value, r

    study = Study(space, "maximize", seed)
    study.optimize(objective, n_trials, method)
    return study
