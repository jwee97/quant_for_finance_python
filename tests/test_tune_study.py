"""The simulation behind the Tune tab's thresholds (``experiments/tune_selection_study.py``) runs, and its worlds have the truth they claim to have. Small runs only: the study itself takes minutes."""

from __future__ import annotations

import numpy as np
import pytest

from experiments import hpo_benchmark as bench
from experiments import tune_selection_study as study


def test_a_world_earns_its_edge_at_the_best_place_only_and_nothing_where_there_is_none():
    w = study.World(3, dims=1, edge=0.0006)
    peak = {"p0": 1.0 + 10.0 * w.peak[0]}
    far = {"p0": 1.0 + 10.0 * (1.0 if w.peak[0] < 0.5 else 0.0)}
    assert w.true_sharpe(peak) == pytest.approx(w.best_sharpe) and w.true_sharpe(far) < 0.1 * w.best_sharpe
    nothing = study.World(3, dims=3, edge=0.0)
    assert nothing.best_sharpe == 0.0 and all(nothing.true_sharpe({n: v for n in nothing.names}) == 0.0 for v in (1.0, 5.0, 11.0))
    on_peak = study.World(3, dims=3, edge=0.0006, start_on_peak=True)
    assert on_peak.true_sharpe({s.name: s.default for s in on_peak.settings}) == pytest.approx(on_peak.best_sharpe, rel=0.02), "the starting settings sit on the best place"


def test_settings_close_together_trade_nearly_the_same_days_and_far_apart_ones_do_not():
    w = study.World(5, dims=1, edge=0.0)
    a, b, c = (w.returns({"p0": v}) for v in (5.0, 5.2, 10.5))
    assert np.corrcoef(a, b)[0, 1] > 0.9 and abs(np.corrcoef(a, c)[0, 1]) < 0.5
    assert a.std() == pytest.approx(study.SIGMA, rel=0.1) and c.std() == pytest.approx(study.SIGMA, rel=0.1), "every setting has the same volatility"
    only_before = study.World(5, dims=1, edge=0.002, tuning_only=True)
    r = only_before.returns({"p0": 1.0 + 10.0 * only_before.peak[0]})
    assert r.iloc[: study.SPLIT_I].mean() > 5 * abs(r.iloc[study.SPLIT_I:].mean()), "the edge stops where the hold-out begins"


def test_the_verdict_study_compares_the_checks_with_the_hold_out_gain_alone():
    rows = study.run_verdicts(reps=2, procs=1, trials=14, names=["null-1d", "strong-1d"])
    assert len(rows) == 4 and {r["name"] for r in rows} == {"null-1d", "strong-1d"}
    assert all(r["now"] in ("good", "mixed", "bad", "keep") and r["before"] in ("good", "mixed", "bad", "keep") for r in rows)
    assert all(not (r["now"] == "good" and r["before"] != "good") for r in rows), "the checks only ever take a good away"
    table = study.verdict_table(rows)
    assert "no edge" in table and "strong edge" in table and "before: good" in table and "weak edge" not in table


def test_the_search_study_runs_every_method_and_scores_the_picks_by_what_they_really_earn(monkeypatch):
    monkeypatch.setattr(study, "BUDGETS", (8, 12))
    monkeypatch.setattr(study, "METHODS", {k: v for k, v in study.METHODS.items() if k in ("random", "tpe+sobol", "bayes")})
    rows = study.run_search(reps=1, procs=1, worlds=["1d"])
    assert len(rows) == 3 * 2
    for _, out in rows:
        assert set(out) == {"best", "plateau", "centroid", "start", "top"} and out["top"][0] > out["start"][0] - 1e-9
        assert all(np.isfinite(out[rule][0]) for rule in ("best", "plateau", "centroid"))
    text = study.search_table(rows, ["1d"])
    assert "world 1d" in text and "tpe+sobol" in text and "bayes" in text and "random" in text


def test_the_benchmark_functions_have_the_best_value_they_say_and_the_runner_prints_a_regret_table():
    for name, (fn, space, noise, top) in bench.FUNCTIONS.items():
        rng = np.random.default_rng(0)
        draws = [fn({k: p.sample(rng) for k, p in space.items()}) for _ in range(300)]
        assert max(draws) <= top + 1e-9 and noise > 0, name                    # no point of the space beats the stated best
    assert bench.branin({"x1": np.pi, "x2": 2.275}) == pytest.approx(-0.397887 / 50.0, abs=1e-5), "one of Branin's three global minima, scaled"
    assert bench.mixed({"x": 1.0, "y": 0.1, "n": 8, "c": "b"}) == pytest.approx(0.3) and bench.rosenbrock4({f"u{i}": 1.0 for i in range(4)}) == pytest.approx(0.0)
    rows = bench.run(seeds=2, procs=1, functions=["branin", "mixed"], budgets=(8, 12), methods=["random", "tpe+sobol"])
    assert len(rows) == 2 * 2 * 2
    for _, out in rows:
        assert set(out) == {(b, r) for b in (8, 12) for r in bench.RULES} and all(v >= -1e-9 for v in out.values()), "regret is never below zero"
    text = bench.regret_table(rows, ["branin", "mixed"], (8, 12), ["random", "tpe+sobol"])
    assert "branin: mean regret" in text and "mixed: mean regret" in text and "tpe+sobol" in text and "12 tries" in text
