"""Hyperparameter optimisation that counts its own trials and reports how much of the winner is selection.

Search methods (``Study.optimize``): ``random``, ``grid``, ``tpe`` (a Tree-structured Parzen Estimator after Bergstra et al. 2011: after a random start, model the good and the bad trials with Parzen
densities and propose the candidate maximising ``l(x) / g(x)``) and ``halving`` (successive halving, Jamieson & Talwalkar 2016: many configurations on a small budget, the best third promoted).

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
        if self.kind == "log":
            return (np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        return (float(value) - self.low) / (self.high - self.low) if self.high != self.low else 0.5


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

    def to_frame(self) -> pd.DataFrame:
        rows = [{"number": t.number, **t.params, "value": t.value, "seconds": t.seconds, "budget": t.budget, "error": t.error} for t in self.trials]
        return pd.DataFrame(rows).set_index("number") if rows else pd.DataFrame()

    # ----------------------------------------------------------------------------------------------------------------- proposals
    def _random(self, rng):
        return {k: p.sample(rng) for k, p in self.space.items()}

    def _tpe(self, rng, n_startup: int = 10, gamma: float = 0.25, n_candidates: int = 32):
        done = [t for t in self.trials if t.error is None and np.isfinite(t.value)]
        if len(done) < n_startup:
            return self._random(rng)
        done.sort(key=lambda t: -self._sign() * t.value)
        n_good = max(2, int(np.ceil(gamma * len(done))))
        good, bad = done[:n_good], done[n_good:]
        candidates = []
        for _ in range(n_candidates):
            candidates.append({k: self._sample_from(k, [t.params[k] for t in good], rng) for k in self.space})
        scores = []
        for c in candidates:
            lg = sum(self._log_density(k, c[k], [t.params[k] for t in good]) for k in self.space)
            lb = sum(self._log_density(k, c[k], [t.params[k] for t in bad]) for k in self.space)
            scores.append(lg - lb)
        return candidates[int(np.argmax(scores))]

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

    def optimize(self, objective: Callable, n_trials: int = 30, method: str = "random", grid_points: int = 4, **kwargs) -> "Study":
        """Run the search. ``objective(params)`` returns a number or ``(number, returns_series)``; for ``halving`` it is ``objective(params, budget)``."""
        rng = np.random.default_rng(self.seed + len(self.trials))
        if method == "grid":
            axes = [self.space[k].grid(grid_points) for k in self.space]
            for combo in product(*axes):
                self._run(objective, dict(zip(self.space, combo)))
        elif method in ("random", "tpe"):
            for _ in range(n_trials):
                params = self._random(rng) if method == "random" else self._tpe(rng, **kwargs)
                self._run(objective, params)
        elif method == "halving":
            self._halving(objective, n_trials, rng, **kwargs)
        else:
            raise ValueError("method must be random, grid, tpe or halving")
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

    def selection_report(self, periods_per_year: int = 252, n_boot: int = 500, pbo_blocks: int = 8) -> dict:
        """How much of the winner is selection? Needs the return streams of the trials."""
        from ..stats.inference import deflated_sharpe_ratio, romano_wolf
        from ..validation.multiple_testing import pbo_cscv

        mat = self.returns_matrix().dropna()
        if mat.shape[1] < 2:
            raise ValueError("selection_report needs at least two trials with return streams")
        sharpes = mat.mean() / mat.std(ddof=1)
        best = int(self.best.number) if self.best.number in mat.columns else int(sharpes.idxmax())
        dsr = deflated_sharpe_ratio(mat[best].to_numpy(), trial_sharpes=sharpes.to_numpy(), n_trials=self.n_trials)
        pbo = pbo_cscv(mat.to_numpy(), blocks=pbo_blocks, annualise=periods_per_year)
        rw = romano_wolf(mat, n_boot=n_boot, seed=self.seed)
        return {"best_trial": best, "best_annual_sharpe": float(sharpes[best] * np.sqrt(periods_per_year)), "n_trials": self.n_trials, "n_with_returns": int(mat.shape[1]),
                "deflated_sharpe_probability": dsr["probability"], "deflated_benchmark_sharpe_annual": float(dsr["benchmark"] * np.sqrt(periods_per_year)),
                "pbo": pbo["pbo"], "mean_oos_sharpe_of_is_winner": pbo["mean_oos_of_best"], "romano_wolf_p_best": float(rw.loc[best, "p_romano_wolf"]),
                "survivors_at_5pct": int((rw["p_romano_wolf"] < 0.05).sum())}


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
