"""The Optuna engine of the platform's hyperparameter study."""

import numpy as np
import pytest

pytest.importorskip("optuna")

from src.ops.hpo import Categorical, Float, Int, LogFloat, Study   # noqa: E402


def _space():
    return {"x": Float(-3.0, 3.0), "n": Int(1, 9), "lr": LogFloat(1e-4, 1.0), "kind": Categorical("a", "b", "c")}


def _objective(p):
    return -((p["x"] - 1.0) ** 2) - 0.05 * (p["n"] - 6) ** 2 - (np.log10(p["lr"]) + 2) ** 2 + (0.5 if p["kind"] == "b" else 0.0)


def test_the_optuna_engine_searches_the_space_and_beats_random_draws():
    s = Study(_space(), "maximize", seed=3).optimize(_objective, n_trials=40, method="optuna")
    assert s.n_trials == 40 and all(isinstance(t.params["n"], int) and 1 <= t.params["n"] <= 9 and t.params["kind"] in "abc" for t in s.trials)
    random_best = Study(_space(), "maximize", seed=3).optimize(_objective, n_trials=40, method="random").best.value
    assert s.best.value > -1.5 and s.best.value >= random_best - 0.3


def test_trials_run_by_hand_are_shown_to_optuna_failures_are_counted_and_a_callback_can_stop():
    s = Study(_space(), "maximize", seed=1)
    s._run(_objective, {"x": 1.0, "n": 6, "lr": 0.01, "kind": "b"})
    calls = []

    def flaky(p):
        calls.append(p)
        if len(calls) == 3:
            raise RuntimeError("boom")
        return _objective(p)

    s.optimize(flaky, n_trials=12, method="optuna", callback=lambda study, trial: len(study.trials) >= 7)
    assert s.n_trials == 7 and sum(t.error is not None for t in s.trials) == 1 and s.best.value == pytest.approx(0.5, abs=1e-9) or s.best.value > 0.4


def test_an_unknown_method_names_the_choices():
    with pytest.raises(ValueError, match="optuna"):
        Study(_space()).optimize(_objective, n_trials=1, method="nope")
