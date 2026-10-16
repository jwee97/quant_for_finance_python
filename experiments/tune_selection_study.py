"""How should the Tune tab choose a winner, and when should it say "good"? A simulation on strategies whose truth is known.

A *world* is a strategy with settings in the unit cube whose daily returns are ``sum_k w_k(u) E_kt + edge(u)``: ``E`` is a set of independent noise streams spread over the settings, ``w_k(u)`` a Gaussian
bump around each (so that two settings close together trade nearly the same days, as the settings of a real strategy do) and ``edge(u)`` the TRUE daily mean a setting earns: nothing, or a bump around
one place. Because the truth is known, every verdict can be marked right or wrong, and every search can be scored by what its pick really earns, which no study of real prices can do.

    python -m experiments.tune_selection_study verdicts --reps 150     # how often each rule says "good", by kind of world (about 12 minutes with 2 processes)
    python -m experiments.tune_selection_study search --reps 40        # what each search method's pick really earns, by rule for choosing the winner
    python -m experiments.tune_selection_study objective --reps 100    # what each way of scoring a try makes the search pick, with steady and with heavy-tailed returns

``verdicts`` runs the real ``webapp.tuning.tune`` on each world and compares the verdict it gives with the one the hold-out gain alone would have given (what the tab said before the checks were
added). ``search`` runs ``ops.hpo.Study`` with each method on the app's own score. ``objective`` runs the default search with each way of scoring a try. All print a table; the numbers quoted in
docs/tuning.md come from them. Nothing here touches prices or the network.
"""

from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.ops.hpo import Study
from src.webapp import tuning as T

N, SIGMA = 2600, 0.01                                                       # days, and the daily volatility of every setting
IDX = pd.bdate_range("2010-01-04", periods=N)
SPLIT_I = int(round(N * 0.7))
TRIALS = 40

# name: (settings, daily edge at the best place, extra keywords)
SCENARIOS = {
    "null-1d": (1, 0.0, {}), "weak-1d": (1, 0.0002, {}), "moderate-1d": (1, 0.0004, {}), "strong-1d": (1, 0.0008, {}), "tuning-only-1d": (1, 0.0004, {"tuning_only": True}),
    "start-on-peak-1d": (1, 0.0004, {"start_on_peak": True}),
    "null-3d": (3, 0.0, {}), "weak-3d": (3, 0.0002, {}), "moderate-3d": (3, 0.0004, {}), "strong-3d": (3, 0.0008, {}), "tuning-only-3d": (3, 0.0004, {"tuning_only": True}),
    "start-on-peak-3d": (3, 0.0004, {"start_on_peak": True}),
}
GROUPS = {"no edge": ["null-1d", "null-3d"], "weak edge": ["weak-1d", "weak-3d"], "moderate edge": ["moderate-1d", "moderate-3d"], "strong edge": ["strong-1d", "strong-3d"],
          "edge only before the split": ["tuning-only-1d", "tuning-only-3d"], "start already best": ["start-on-peak-1d", "start-on-peak-3d"]}
METHODS = {"random": ("random", {}), "tpe": ("tpe", {}), "tpe+sobol": ("tpe", {"quasi_start": True}), "tpe-joint+sobol": ("tpe", {"multivariate": True, "quasi_start": True}), "bayes": ("bayes", {})}
BUDGETS = (20, 40, 60)
WORLDS = {"1d": {"dims": 1, "edge": 0.0006}, "2d": {"dims": 2, "edge": 0.0006}, "4d": {"dims": 4, "edge": 0.0006}, "4d-weak": {"dims": 4, "edge": 0.0003}}


class World:
    """A strategy with ``dims`` settings (each from 1 to 11) and a known edge; see the module docstring."""

    def __init__(self, seed: int, dims: int = 1, edge: float = 0.0, width: float = 0.2, tuning_only: bool = False, start_on_peak: bool = False, heavy_tails: bool = False):
        rng = np.random.default_rng(seed)
        k = 12 if dims == 1 else 30 if dims == 2 else 60
        self.anchors = np.linspace(0, 1, k)[:, None] if dims == 1 else rng.random((k, dims))
        self.bw = 0.12 if dims == 1 else 0.22
        self.E = rng.standard_t(3, (k, N)) * SIGMA / np.sqrt(3.0) if heavy_tails else rng.normal(0.0, SIGMA, (k, N))      # Student t with three degrees of freedom has variance 3
        self.edge, self.width, self.tuning_only = edge, width, tuning_only
        self.peak = 0.3 + 0.4 * rng.random(dims)                                              # the best place, in the middle part of the cube
        start = self.peak.copy() if start_on_peak else np.full(dims, 0.12)
        self.names = [f"p{j}" for j in range(dims)]
        self.settings = [T.Setting(n, "float", float(1.0 + 10.0 * start[j]), 1.0, 11.0, "linear") for j, n in enumerate(self.names)]

    def units(self, params: dict) -> np.ndarray:
        return np.array([(params[n] - 1.0) / 10.0 for n in self.names])

    def true_mean(self, params: dict) -> float:
        return float(self.edge * np.exp(-0.5 * np.sum(((self.units(params) - self.peak) / self.width) ** 2)))

    def true_sharpe(self, params: dict) -> float:
        return self.true_mean(params) / SIGMA * np.sqrt(252)

    @property
    def best_sharpe(self) -> float:
        return self.edge / SIGMA * np.sqrt(252)

    def returns(self, params: dict) -> pd.Series:
        w = np.exp(-0.5 * ((self.anchors - self.units(params)) ** 2).sum(axis=1) / self.bw ** 2)
        r = (w / np.sqrt((w ** 2).sum())) @ self.E
        edge = np.full(N, self.true_mean(params))
        if self.tuning_only:
            edge = np.where(np.arange(N) < SPLIT_I, edge, 0.0)
        return pd.Series(r + edge, index=IDX)

    def run(self, params: dict) -> T.Run:
        return T.Run(self.returns(params))


# ------------------------------------------------------------------------------------------------------------------ the verdicts
def _verdict_run(task: tuple[str, int, int]) -> dict:
    warnings.filterwarnings("ignore")
    name, seed, trials = task
    dims, edge, extra = SCENARIOS[name]
    world = World(seed, dims=dims, edge=edge, **extra)
    info: dict = {}
    out = T.tune(world.run, world.settings, trials=trials, seconds=900, seed=seed, select="best", search="tpe", inspect=info)
    win = info["win"]
    before = T.judge("sharpe", out["start"], out["best"], *T.spans(win), out["same_as_start"], None, out["search"]["tried"], 0, [], None)["level"]       # the hold-out gain alone
    sel = out["selection"] or {}
    return {"name": name, "now": out["verdict"]["level"], "before": before, "same": out["same_as_start"], "true_gain": world.true_sharpe(out["best"]["params"]) - world.true_sharpe(out["start"]["params"]),
            "own": sel.get("best_annual_sharpe"), "expected": sel.get("mean_oos_sharpe_of_is_winner"), "true": world.true_sharpe(out["best"]["params"])}


def verdict_table(rows: list[dict]) -> str:
    """For each kind of world, the share of searches called good by the hold-out gain alone and by the checks; and how often a good was a real improvement (0.15 Sharpe points or more)."""
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    lines = [f"{'kind of world':28s} {'before: good':>13s} {'now: good':>10s}   {'good that were real, before':>28s} {'now':>5s}"]
    for group, names in GROUPS.items():
        rs = [r for n in names for r in by.get(n, [])]
        if not rs:
            continue
        before, now = [r for r in rs if r["before"] == "good"], [r for r in rs if r["now"] == "good"]
        real = lambda sel: f"{100 * np.mean([r['true_gain'] >= 0.15 for r in sel]):4.0f}%" if sel else "   - "      # noqa: E731
        lines.append(f"{group:28s} {100 * len(before) / len(rs):12.0f}% {100 * len(now) / len(rs):9.0f}%   {real(before):>28s} {real(now):>5s}")
    lines.append(f"\n{'kind of world, by number of settings':38s} {'winner own score':>17s} {'expected on new days':>21s} {'what it really earns':>21s}")
    for name in SCENARIOS:
        rs = [r for r in by.get(name, []) if r["own"] is not None and r["expected"] is not None]
        if rs and not name.startswith(("tuning-only", "start-on")):
            lines.append(f"{name:38s} {np.mean([r['own'] for r in rs]):17.2f} {np.mean([r['expected'] for r in rs]):21.2f} {np.mean([r['true'] for r in rs]):21.2f}")
    return "\n".join(lines)


def run_verdicts(reps: int, procs: int, trials: int = TRIALS, names: list[str] | None = None) -> list[dict]:
    tasks = [(n, s, trials) for n in (names or list(SCENARIOS)) for s in range(reps)]
    if procs <= 1:
        return [_verdict_run(t) for t in tasks]
    with Pool(procs) as pool:
        return pool.map(_verdict_run, tasks, chunksize=2)


# ------------------------------------------------------------------------------------------------------------------ the search
def _search_run(task: tuple[str, str, int, int]) -> tuple[tuple, dict]:
    warnings.filterwarnings("ignore")
    wname, mname, budget, seed = task
    world = World(seed, **WORLDS[wname])
    method, kw = METHODS[mname]
    space = {s.name: s.space() for s in world.settings}

    def objective(params):
        part = world.returns(params).iloc[:SPLIT_I]
        return T.score(part, "sharpe", True), part

    study = Study(space, "maximize", seed)
    start = {s.name: s.default for s in world.settings}
    study._run(objective, dict(start))
    options = {"n_startup": max(5, min(12, budget // 4)), **kw} if method != "random" else {}
    study.optimize(objective, budget - 1, method, unique=True, **options)

    def hold(params):
        return T.metrics(world.returns(params).iloc[SPLIT_I:])["sharpe"]

    base = hold(start)
    out = {rule: (world.true_sharpe(p), hold(p) - base) for rule, p in {"best": study.best.params, "plateau": study.plateau().params, "centroid": study.centroid()}.items()}
    out["start"], out["top"] = (world.true_sharpe(start), 0.0), (world.best_sharpe, 0.0)
    return task, out


def search_table(rows: list, worlds: list[str], budgets: tuple[int, ...] | None = None, methods: list[str] | None = None) -> str:
    budgets, methods = budgets or BUDGETS, methods or list(METHODS)
    agg = defaultdict(list)
    for (w, m, b, _), out in rows:
        for rule in ("best", "plateau", "centroid", "start", "top"):
            agg[(w, m, b, rule)].append(out[rule])
    lines = []
    for w in worlds:
        top = np.mean([o["top"][0] for (ww, _, _, _), o in rows if ww == w])
        start = np.mean([o["start"][0] for (ww, _, _, _), o in rows if ww == w])
        lines.append(f"\nworld {w}: mean TRUE annual Sharpe ratio of the pick, by rule for choosing it (best score / best neighbourhood / middle of the best tries); best possible {top:.2f}, starting settings {start:.2f}")
        lines.append(f"{'method':18s}" + "".join(f"   {b} tries{'':16s}" for b in budgets))
        for m in methods:
            cells = ["  " + " / ".join(f"{np.mean([x[0] for x in agg[(w, m, b, r)]]):5.2f}" for r in ("best", "plateau", "centroid")) + "      " for b in budgets]
            lines.append(f"{m:18s}" + "".join(cells))
    return "\n".join(lines)


def run_search(reps: int, procs: int, worlds: list[str] | None = None, budgets: tuple[int, ...] | None = None, methods: list[str] | None = None) -> list:
    tasks = [(w, m, b, s) for w in (worlds or list(WORLDS)) for m in (methods or METHODS) for b in (budgets or BUDGETS) for s in range(reps)]
    if procs <= 1:
        return [_search_run(t) for t in tasks]
    with Pool(procs) as pool:
        return pool.map(_search_run, tasks, chunksize=2)


# ------------------------------------------------------------------------------------------------------------------ the score of a try
SCORES = {"plain Sharpe": ("sharpe", False), "block score (the default)": ("sharpe", True), "plain cautious Sharpe": ("cautious", False), "cautious block score": ("cautious", True)}
TAILS = {"steady (normal) returns": False, "heavy-tailed returns (t, 3 df)": True}


def _objective_run(task: tuple[str, bool, int, int, str]) -> tuple[tuple, float]:
    warnings.filterwarnings("ignore")
    score_name, heavy, dims, seed, kind = task
    world = World(seed, dims=dims, edge=0.0006, heavy_tails=heavy)
    objective, stable = SCORES[score_name]
    space = {s.name: s.space() for s in world.settings}

    def run(params):
        part = world.returns(params).iloc[:SPLIT_I]
        return T.score(part, objective, stable), part

    study = Study(space, "maximize", seed)
    study._run(run, {s.name: s.default for s in world.settings})
    study.optimize(run, TRIALS - 1, "tpe", unique=True, n_startup=10, quasi_start=True)
    return task, world.true_sharpe(study.best.params)


def objective_table(rows: list) -> str:
    agg = defaultdict(list)
    for (score_name, heavy, dims, _, _), true in rows:
        agg[(heavy, dims, score_name)].append(true)
    lines = []
    for tails, heavy in TAILS.items():
        lines.append(f"\n{tails}: mean TRUE annual Sharpe ratio of the pick (the best place earns 0.95), by number of settings")
        lines.append(f"{'how a try is scored':28s}" + "".join(f"{d} setting{'s' if d > 1 else ' '}" .rjust(14) for d in (1, 3)))
        for name in SCORES:
            lines.append(f"{name:28s}" + "".join(f"{np.mean(agg[(heavy, d, name)]):14.3f}" for d in (1, 3)))
    return "\n".join(lines)


def run_objective(reps: int, procs: int) -> list:
    tasks = [(name, heavy, d, s, "") for name in SCORES for heavy in TAILS.values() for d in (1, 3) for s in range(reps)]
    if procs <= 1:
        return [_objective_run(t) for t in tasks]
    with Pool(procs) as pool:
        return pool.map(_objective_run, tasks, chunksize=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("study", choices=["verdicts", "search", "objective"])
    ap.add_argument("--reps", type=int, default=40, help="searches per kind of world")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--trials", type=int, default=TRIALS, help="tries per search in the verdicts study")
    ap.add_argument("--worlds", default=",".join(WORLDS), help="the search study's worlds, comma separated")
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)), help="the search study's numbers of tries, comma separated")
    ap.add_argument("--methods", default=",".join(METHODS), help="the search study's methods, comma separated")
    args = ap.parse_args(argv)
    if args.study == "verdicts":
        print(verdict_table(run_verdicts(args.reps, args.procs, args.trials)))
    elif args.study == "search":
        worlds, budgets, methods = args.worlds.split(","), tuple(int(b) for b in args.budgets.split(",")), args.methods.split(",")
        print(search_table(run_search(args.reps, args.procs, worlds, budgets, methods), worlds, budgets, methods))
    else:
        print(objective_table(run_objective(args.reps, args.procs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
