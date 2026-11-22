"""Walk-forward: the sign is chosen on the past only; a factor that only fits its own sample is shown as such."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import walkforward as W


def panels(n_dates=120, n=60, ic=0.15, flip_at=None, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_dates, freq="ME")
    sig = pd.DataFrame(rng.normal(size=(n_dates, n)), index=idx)
    fwd = pd.DataFrame(0.0, index=idx, columns=sig.columns)
    for t in range(n_dates):
        sign = -1.0 if flip_at is not None and t >= flip_at else 1.0
        fwd.iloc[t] = sign * ic * sig.iloc[t].to_numpy() + np.sqrt(1 - ic ** 2) * rng.normal(size=n)
    return sig, fwd


def test_folds_never_overlap_and_leave_an_embargo_between_training_and_test():
    f = W.folds(100, train=36, test=12, embargo=3)
    assert f[0] == (0, 36, 39, 51) and all(c - b == 3 for a, b, c, d in f) and all(d <= 100 for *_, d in f)
    assert all(f[i + 1][2] >= f[i][3] for i in range(len(f) - 1)), "test windows do not overlap"
    assert W.folds(30, 36, 12, 1) == []


def test_a_real_factor_holds_out_of_sample_with_the_sign_chosen_from_the_past_only():
    sig, fwd = panels(ic=0.2)
    r = W.run_panels(sig, fwd, horizon_dates=1, min_assets=20)
    assert r["oos"]["mean_ic"] > 0.1 and r["oos"]["hit_rate"] > 0.8 and r["reading"] == "holds out of sample" and all(f["chosen_sign"] == 1.0 for f in r["folds"])
    neg = W.run_panels(-sig, fwd, 1)
    assert all(f["chosen_sign"] == -1.0 for f in neg["folds"]) and neg["oos"]["mean_ic"] > 0.1, "a factor that works upside down is turned around by the training data"


def test_a_relationship_that_reverses_does_not_hold_out_of_sample():
    sig, fwd = panels(n_dates=140, ic=0.2, flip_at=70)
    r = W.run_panels(sig, fwd, 1, train=36, test=12)
    assert r["in_sample"]["mean_ic"] > r["oos"]["mean_ic"] and r["oos"]["mean_ic"] < 0.1 and r["reading"] != "holds out of sample"


def test_pure_noise_has_no_out_of_sample_ic_and_too_little_history_is_said():
    sig, fwd = panels(ic=0.0, seed=3)
    r = W.run_panels(sig, fwd, 1)
    assert abs(r["oos"]["mean_ic"]) < 0.05
    short = W.run_panels(sig.iloc[:30], fwd.iloc[:30], 1)
    assert short["folds"] == [] and "not enough dates" in short["reason"]


def test_adjusting_p_values_for_the_number_of_factors_tried():
    adj = W.bh_adjust([0.001, 0.02, 0.04, 0.5])
    assert adj[0] == pytest.approx(0.004) and adj[3] == pytest.approx(0.5) and all(a >= p for a, p in zip(adj, [0.001, 0.02, 0.04, 0.5]))
