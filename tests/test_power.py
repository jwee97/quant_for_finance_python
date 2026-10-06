import numpy as np
import pandas as pd
import pytest

from experiments.stage39_power import alpha_for, dm_cell, minimum_detectable


def test_alpha_for_gives_the_designed_population_sharpe_difference():
    rng = np.random.default_rng(0)
    b = rng.normal(0.0004, 0.006, 200_000)
    te = 0.05 / np.sqrt(252)
    a = b + alpha_for(0.3, b.mean(), b.std(ddof=1), te) + rng.normal(0, te, b.size)
    diff = np.sqrt(252) * (a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1))
    assert abs(diff - 0.3) < 0.02


def test_minimum_detectable_interpolates_and_reports_nan_when_never_reached():
    f = pd.DataFrame({"x": [0.1, 0.2, 0.3], "power": [0.2, 0.6, 1.0]})
    assert minimum_detectable(f, "x") == pytest.approx(0.25)
    assert np.isnan(minimum_detectable(pd.DataFrame({"x": [0.1, 0.2], "power": [0.1, 0.3]}), "x"))


def test_dm_power_rises_with_skill_and_the_null_cell_is_flagged_undefined():
    low, high = dm_cell(0.002, 187, 60, 1)["power"], dm_cell(0.02, 187, 60, 1)["power"]
    assert high > low + 0.3 and np.isnan(dm_cell(0.0, 187, 10, 1)["power"])
