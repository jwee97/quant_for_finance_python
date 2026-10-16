"""How well do the search methods of ``ops.hpo`` find the best settings of a noisy function whose best is known?

Five test functions (the Branin and Hartmann-6 functions of the optimisation literature, a four-dimensional Rosenbrock valley, a mixed space with a logarithmic setting, a whole number and a choice, and
a broad hill beside a narrow spike) are searched with Gaussian noise added to every evaluation, as a backtest's score is noisy. Each method runs for 20, 40 and 60 tries; what is reported is the *regret*
at the point the search finally picks, measured on the function WITHOUT the noise: how far below the best the pick really is. Lower is better. Three ways of picking are compared: the best noisy score
(what the Tune tab has always done), the neighbourhood whose smoothed score is best (``Study.plateau``) and the middle of the best tries (``Study.centroid``).

    python -m experiments.hpo_benchmark --seeds 12 --procs 4        # about an hour: the Gaussian-process runs dominate
    python -m experiments.hpo_benchmark --seeds 4 --functions branin,mixed --budgets 20,40

The result quoted in docs/tuning.md was made with 12 seeds. Nothing here touches prices or the network.
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from src.ops.hpo import Categorical, Float, Int, LogFloat, Study


def branin(p):
    x1, x2 = p["x1"], p["x2"]
    b, c, t = 5.1 / (4 * np.pi ** 2), 5 / np.pi, 1 / (8 * np.pi)
    return -((x2 - b * x1 ** 2 + c * x1 - 6.0) ** 2 + 10.0 * (1 - t) * np.cos(x1) + 10.0) / 50.0                  # best -0.008


def hartmann6(p):
    a = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    centre = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
    x = np.array([p[f"x{i}"] for i in range(6)])
    return float(np.sum(np.array([1.0, 1.2, 3.0, 3.2]) * np.exp(-np.sum(a * (x - centre) ** 2, axis=1))))          # best 3.32237


def rosenbrock4(p):
    u = np.array([p[f"u{i}"] for i in range(4)])
    return -float(np.sum(100 * (u[1:] - u[:-1] ** 2) ** 2 + (1 - u[:-1]) ** 2)) / 400.0                           # best 0 at (1, 1, 1, 1)


def mixed(p):
    return float(-(p["x"] - 1) ** 2 / 10 - (np.log10(p["y"]) + 1) ** 2 / 4 - ((p["n"] - 8) / 8) ** 2 + (0.3 if p["c"] == "b" else 0.0))                  # best 0.3


def spike(p):
    hill = 0.6 * np.exp(-((p["x"] - 0.3) ** 2 + (p["y"] - 0.3) ** 2) / (2 * 0.15 ** 2))
    peak = 1.0 * np.exp(-((p["x"] - 0.8) ** 2 + (p["y"] - 0.8) ** 2) / (2 * 0.03 ** 2))
    return float(hill + peak)                                                                                      # best 1.0, on a spike 0.03 wide


FUNCTIONS = {                                                        # name: (function, space, noise standard deviation, best value)
    "branin": (branin, {"x1": Float(-5, 10), "x2": Float(0, 15)}, 0.02, -0.0080),
    "hartmann6": (hartmann6, {f"x{i}": Float(0, 1) for i in range(6)}, 0.15, 3.32237),
    "rosenbrock4": (rosenbrock4, {f"u{i}": Float(-2, 2) for i in range(4)}, 0.03, 0.0),
    "mixed": (mixed, {"x": Float(-5, 5), "y": LogFloat(1e-3, 1e1), "n": Int(1, 20), "c": Categorical("a", "b", "c")}, 0.08, 0.3),
    "spike": (spike, {"x": Float(0, 1), "y": Float(0, 1)}, 0.08, 1.0),
}
METHODS = {"random": ("random", {}), "tpe": ("tpe", {}), "tpe+sobol": ("tpe", {"quasi_start": True}), "tpe-joint+sobol": ("tpe", {"multivariate": True, "quasi_start": True}), "bayes": ("bayes", {})}
BUDGETS = (20, 40, 60)
RULES = ("best", "plateau", "centroid")


def _run(task: tuple[str, str, int, tuple[int, ...]]) -> tuple[tuple, dict]:
    warnings.filterwarnings("ignore")
    fname, mname, seed, budgets = task
    fn, space, noise, top = FUNCTIONS[fname]
    method, kw = METHODS[mname]
    rng = np.random.default_rng(10_000 + seed)
    study = Study(space, "maximize", seed)
    out, done = {}, 0
    for budget in budgets:
        options = dict(kw)
        if method == "tpe":
            options.setdefault("n_startup", max(5, budget // 5))
        elif method == "bayes":
            options.setdefault("n_startup", 8)
        study.optimize(lambda p: fn(p) + rng.normal(0.0, noise), budget - done, method, unique=True, **options)
        done = budget
        picks = {"best": study.best.params, "plateau": study.plateau().params, "centroid": study.centroid()}
        for rule, params in picks.items():
            out[(budget, rule)] = top - fn(params)
    return task, out


def regret_table(rows: list, functions: list[str], budgets: tuple[int, ...], methods: list[str]) -> str:
    agg = defaultdict(list)
    for (fname, mname, _, _), out in rows:
        for (budget, rule), regret in out.items():
            agg[(fname, mname, budget, rule)].append(regret)
    lines = []
    for fname in functions:
        seeds = len(agg[(fname, methods[0], budgets[0], "best")])
        lines.append(f"\n{fname}: mean regret on the noise-free function ({seeds} seeds), by rule for picking (best score / best neighbourhood / middle of the best tries)")
        lines.append(f"{'method':18s}" + "".join(f"   {b} tries{'':18s}" for b in budgets))
        for m in methods:
            cells = ["   " + " / ".join(f"{np.mean(agg[(fname, m, b, r)]):7.4f}" for r in RULES) + "  " for b in budgets]
            lines.append(f"{m:18s}" + "".join(cells))
    return "\n".join(lines)


def run(seeds: int, procs: int, functions: list[str] | None = None, budgets: tuple[int, ...] | None = None, methods: list[str] | None = None) -> list:
    functions, budgets, methods = functions or list(FUNCTIONS), budgets or BUDGETS, methods or list(METHODS)
    tasks = [(f, m, s, budgets) for f in functions for m in methods for s in range(seeds)]
    if procs <= 1:
        return [_run(t) for t in tasks]
    with Pool(procs) as pool:
        return pool.map(_run, tasks, chunksize=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--functions", default=",".join(FUNCTIONS))
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args(argv)
    functions, budgets, methods = args.functions.split(","), tuple(int(b) for b in args.budgets.split(",")), args.methods.split(",")
    t0 = time.time()
    rows = run(args.seeds, args.procs, functions, budgets, methods)
    print(f"{len(rows)} searches in {time.time() - t0:.0f} s")
    print(regret_table(rows, functions, budgets, methods))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
