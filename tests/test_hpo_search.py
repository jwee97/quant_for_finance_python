"""The search machinery of ``ops.hpo`` beyond random and plain TPE: a space-filling start, a TPE that models the settings together, a Gaussian-process search, and the two ways of picking the final
settings from a noisy search (the best neighbourhood, the middle of the best tries)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc

from src.ops.hpo import Categorical, Float, Int, LogFloat, Study

MIXED = {"x": Float(-5, 5), "y": LogFloat(1e-3, 1e1), "n": Int(1, 20), "c": Categorical("a", "b", "c")}


def mixed(p):
    return float(-(p["x"] - 1) ** 2 / 10 - (np.log10(p["y"]) + 1) ** 2 / 4 - ((p["n"] - 8) / 8) ** 2 + (0.3 if p["c"] == "b" else 0.0))


def inside(space, params):
    for k, p in space.items():
        v = params[k]
        if p.kind == "categorical":
            assert v in p.choices
        else:
            assert p.low <= v <= p.high
            assert (p.kind != "int") or isinstance(v, int)


# ------------------------------------------------------------------------------------------------------ the units of a range
def test_a_value_and_its_place_in_the_range_convert_both_ways_for_every_kind_of_setting():
    for p, values in [(Float(2, 6), [2, 3.5, 6]), (LogFloat(1e-3, 1e1), [1e-3, 0.1, 10.0]), (Int(5, 25), [5, 12, 25]), (Categorical("a", "b", "c"), ["a", "b", "c"])]:
        for v in values:
            back = p.from_unit(p.to_unit(v))
            assert back == pytest.approx(v) if p.kind in ("float", "log") else back == v, (p.kind, v, back)
    assert Float(2, 6).from_unit(-1) == 2 and Float(2, 6).from_unit(7) == 6, "a place outside the range is clipped to its end"
    assert Int(5, 25).from_unit(0.5) == 15 and isinstance(Int(5, 25).from_unit(0.51), int)
    cat = Categorical("a", "b", "c")
    assert [cat.from_unit(u) for u in (0.0, 0.2, 0.4, 0.7, 1.0)] == ["a", "a", "b", "c", "c"]
    assert cat.to_unit("b") == pytest.approx(0.5), "a choice sits in the middle of its slot"


# ------------------------------------------------------------------------------------------------------ the start
def test_a_quasi_random_start_covers_the_space_more_evenly_than_random_draws_and_is_the_same_each_time():
    space = {"x": Float(0, 1), "y": Float(0, 1)}
    quasi, random = [], []
    for seed in range(8):
        study = Study(space, "maximize", seed).optimize(lambda p: p["x"], 16, "tpe", n_startup=16, quasi_start=True)
        plain = Study(space, "maximize", seed).optimize(lambda p: p["x"], 16, "tpe", n_startup=16)
        quasi.append(qmc.discrepancy(np.array([[t.params["x"], t.params["y"]] for t in study.trials])))
        random.append(qmc.discrepancy(np.array([[t.params["x"], t.params["y"]] for t in plain.trials])))
    assert np.mean(quasi) < 0.7 * np.mean(random)
    a = Study(space, "maximize", 3).optimize(lambda p: p["x"], 10, "tpe", n_startup=10, quasi_start=True)
    b = Study(space, "maximize", 3).optimize(lambda p: p["x"], 10, "tpe", n_startup=10, quasi_start=True)
    c = Study(space, "maximize", 4).optimize(lambda p: p["x"], 10, "tpe", n_startup=10, quasi_start=True)
    assert [t.params for t in a.trials] == [t.params for t in b.trials] and [t.params for t in a.trials] != [t.params for t in c.trials]


def test_a_quasi_start_on_a_space_of_whole_numbers_and_choices_gives_valid_settings():
    space = {"k": Int(1, 6), "c": Categorical("u", "v"), "z": LogFloat(0.01, 10)}
    study = Study(space, "maximize", 1).optimize(lambda p: p["k"], 40, "tpe", n_startup=40, quasi_start=True)
    assert len(study.trials) == 40
    for t in study.trials:
        inside(space, t.params)
    assert {t.params["c"] for t in study.trials} == {"u", "v"} and {t.params["k"] for t in study.trials} == {1, 2, 3, 4, 5, 6}, "the points are spread over every choice"


# ------------------------------------------------------------------------------------------------------ TPE on the settings together
def test_the_joint_tpe_runs_on_a_mixed_space_stays_inside_it_and_finds_the_neighbourhood_of_the_optimum():
    for seed in range(3):
        study = Study(MIXED, "maximize", seed).optimize(mixed, 60, "tpe", n_startup=15, multivariate=True, quasi_start=True, unique=True)
        for t in study.trials:
            inside(MIXED, t.params)
        assert study.best.value > -0.15, (seed, study.best.params, study.best.value)
        assert len({tuple(t.params.values()) for t in study.trials}) == 60, "every configuration is new"


def test_the_joint_tpe_finds_a_ridge_that_only_the_two_settings_together_describe():
    """Reward falls with the distance from the line x = y: a search that treats the settings one at a time sees a flat landscape in each; one that models them together follows the ridge."""
    space = {"x": Float(0, 1), "y": Float(0, 1)}
    ridge = lambda p: -abs(p["x"] - p["y"]) + 0.05 * (p["x"] + p["y"])                  # noqa: E731
    joint, single = [], []
    for seed in range(8):
        joint.append(Study(space, "maximize", seed).optimize(ridge, 50, "tpe", n_startup=15, multivariate=True, quasi_start=True).best.value)
        single.append(Study(space, "maximize", seed).optimize(ridge, 50, "tpe", n_startup=15).best.value)
    assert np.mean(joint) > np.mean(single) - 0.01 and np.mean(joint) > 0.05, (np.mean(joint), np.mean(single))


# ------------------------------------------------------------------------------------------------------ the Gaussian process
def test_the_gaussian_process_search_finds_a_noisy_optimum_never_repeats_a_try_and_can_be_repeated():
    rng = np.random.default_rng(0)
    noisy = lambda p: float(-(p["x"] - 0.7) ** 2 + rng.normal(0, 0.01))                  # noqa: E731
    space = {"x": Float(0, 1)}
    study = Study(space, "maximize", 2).optimize(noisy, 22, "bayes", unique=True)
    xs = [t.params["x"] for t in study.trials]
    assert len(set(xs)) == 22 and abs(study.plateau().params["x"] - 0.7) < 0.15
    assert sum(abs(x - 0.7) < 0.2 for x in xs[8:]) >= 7, "after the start the search spends most of its tries near the optimum"

    def clean(p):
        return mixed(p)

    one = Study(MIXED, "maximize", 5).optimize(clean, 14, "bayes")
    two = Study(MIXED, "maximize", 5).optimize(clean, 14, "bayes")
    assert [t.params for t in one.trials] == [t.params for t in two.trials]
    for t in one.trials:
        inside(MIXED, t.params)


def test_the_gaussian_process_search_handles_a_space_with_choices_and_beats_the_blind_start():
    gaps = []
    for seed in range(3):
        study = Study(MIXED, "maximize", seed).optimize(mixed, 34, "bayes")
        start = max(t.value for t in study.trials[:8])
        gaps.append(study.best.value - start)
    assert min(gaps) >= 0.0 and np.mean(gaps) > 0.05, gaps


def test_the_gaussian_process_search_falls_back_to_the_parzen_estimator_on_degenerate_data_a_failed_fit_and_a_long_search(monkeypatch):
    space = {"x": Float(0, 1), "k": Int(1, 5), "c": Categorical("a", "b")}
    flat = Study(space, "maximize", 0).optimize(lambda p: 1.0, 20, "bayes", unique=True)
    assert flat.n_trials == 20, "an objective that never changes has nothing to fit and the search goes on"
    long = Study({"x": Float(0, 1)}, "maximize", 1).optimize(lambda p: -(p["x"] - 0.4) ** 2, 14, "bayes", unique=True, max_fit=10)
    assert long.n_trials == 14 and abs(long.best.params["x"] - 0.4) < 0.15, "past max_fit tries the cost of a fit is not paid"

    def broken(self, *args, **kwargs):
        raise np.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr(Study, "_bayes_fit", broken)
    failed = Study({"x": Float(0, 1)}, "maximize", 2).optimize(lambda p: -(p["x"] - 0.7) ** 2, 16, "bayes", unique=True)
    assert failed.n_trials == 16 and abs(failed.best.params["x"] - 0.7) < 0.2


# ------------------------------------------------------------------------------------------------------ choosing from a noisy search
def _study_with(points):
    study = Study({"x": Float(0, 1)}, "maximize", 0)
    for x, value in points:
        study._run(lambda p, v=value: v, {"x": x})
    return study


def test_the_plateau_prefers_a_broad_hill_to_a_lone_spike_and_the_best_trial_does_not():
    hill = [(0.20, 0.70), (0.25, 0.74), (0.30, 0.76), (0.35, 0.75), (0.40, 0.71)]
    spike = [(0.90, 1.00)]
    rest = [(0.05, 0.1), (0.55, 0.2), (0.65, 0.15), (0.75, 0.3), (0.97, 0.1)]
    study = _study_with(hill + spike + rest)
    assert study.best.params["x"] == 0.90, "the single best score is the spike"
    assert 0.2 <= study.plateau().params["x"] <= 0.4, "the best neighbourhood is the hill"
    assert 0.2 <= study.centroid(top_fraction=0.5)["x"] <= 0.9


def test_the_plateau_of_a_study_that_minimises_and_with_choices_is_the_best_neighbourhood_too():
    study = Study({"x": Float(0, 1), "c": Categorical("a", "b")}, "minimize", 0)
    for x, c, v in [(0.1, "a", 3.0), (0.12, "a", 2.9), (0.14, "a", 3.1), (0.8, "b", 0.2), (0.5, "b", 5.0), (0.9, "a", 4.0), (0.3, "b", 4.5)]:
        study._run(lambda p, v=v: v, {"x": x, "c": c})
    assert study.best.params["c"] == "b" and study.best.params["x"] == 0.8
    pick = study.plateau(bandwidth=0.15)
    assert pick.value <= 3.2 and pick.params["c"] in ("a", "b")


def test_the_centroid_is_the_middle_of_the_best_tries_in_the_units_each_kind_of_setting_has():
    space = {"i": Int(1, 100), "f": Float(0, 1), "z": LogFloat(1e-3, 1e1), "c": Categorical("u", "v", "w")}
    study = Study(space, "maximize", 0)
    rows = [({"i": 10, "f": 0.2, "z": 0.01, "c": "u"}, 5.0), ({"i": 20, "f": 0.4, "z": 0.1, "c": "u"}, 4.0), ({"i": 31, "f": 0.6, "z": 1.0, "c": "v"}, 3.0), ({"i": 90, "f": 0.9, "z": 5.0, "c": "w"}, -1.0)]
    for params, value in rows:
        study._run(lambda p, v=value: v, params)
    mid = study.centroid(top_fraction=0.75)
    assert mid == {"i": 20, "f": pytest.approx(0.4), "z": pytest.approx(0.1), "c": "u"} and isinstance(mid["i"], int)
    with pytest.raises(ValueError):
        Study(space).centroid()
    with pytest.raises(ValueError):
        Study(space).plateau()


def test_every_search_method_is_named_when_one_is_not_known():
    with pytest.raises(ValueError, match="random, grid, tpe, bayes, optuna or halving"):
        Study({"x": Float(0, 1)}).optimize(lambda p: p["x"], 3, "nope")
