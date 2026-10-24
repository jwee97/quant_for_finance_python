"""Reality Check, SPA, PBO and the parallel executor (Generation 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.distributed.executor import available_backends, run_tasks, task_seed
from src.validation.multiple_testing import bootstrap_means, hansen_spa, pbo_cscv, white_reality_check


def _noise(n=1500, k=60, seed=1, scale=0.01):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(n, k))


def test_bootstrap_means_are_centred_on_the_sample_mean_with_the_right_spread():
    x = _noise(k=3)
    boot = bootstrap_means(x, n_samples=1500, mean_block=10, seed=2)
    assert boot.shape == (1500, 3)
    assert np.allclose(boot.mean(axis=0), x.mean(axis=0), atol=5e-4)
    iid_se = x.std(axis=0, ddof=1) / np.sqrt(len(x))
    assert np.all(boot.std(axis=0) > 0.7 * iid_se) and np.all(boot.std(axis=0) < 1.6 * iid_se)


def test_bootstrap_is_deterministic_in_its_seed():
    x = _noise(k=5)
    assert np.array_equal(bootstrap_means(x, 100, 10, 3), bootstrap_means(x, 100, 10, 3))
    assert not np.array_equal(bootstrap_means(x, 100, 10, 3), bootstrap_means(x, 100, 10, 4))


def test_reality_check_and_spa_do_not_reject_pure_noise_at_the_nominal_rate():
    rejected_rc = rejected_spa = 0
    trials = 40
    for seed in range(trials):
        x = _noise(n=1000, k=50, seed=100 + seed)
        boot = bootstrap_means(x, 400, 10, seed)
        rejected_rc += white_reality_check(x, boot=boot)["p_value"] <= 0.10
        rejected_spa += hansen_spa(x, boot=boot)["p_consistent"] <= 0.10
    assert rejected_rc <= 0.25 * trials and rejected_spa <= 0.25 * trials     # nominal 10%, wide tolerance for 40 draws


def test_reality_check_and_spa_find_a_planted_edge():
    x = _noise(n=2500, k=80, seed=5)
    x[:, 17] += 0.0012                                            # about 0.12 Sharpe per sqrt(day)... a clear edge at this length
    rc, spa = white_reality_check(x, 800, 10, 3), hansen_spa(x, 800, 10, 3)
    assert rc["best_index"] == 17 and spa["best_index"] == 17
    assert rc["p_value"] < 0.05 and spa["p_consistent"] < 0.05


def test_spa_is_not_fooled_by_adding_terrible_candidates_but_the_reality_check_is():
    """The known weakness of the Reality Check: poor candidates make the null maximum too large."""
    rng = np.random.default_rng(9)
    good = rng.normal(0.0, 0.01, size=(1500, 1)) + 0.0006
    bad = rng.normal(-0.004, 0.01, size=(1500, 400))              # clearly negative-mean noise
    x = np.hstack([good, bad])
    boot = bootstrap_means(x, 800, 10, 1)
    rc, spa = white_reality_check(x, boot=boot), hansen_spa(x, boot=boot)
    assert spa["p_consistent"] <= rc["p_value"] + 1e-12
    assert spa["p_lower"] <= spa["p_consistent"] + 1e-12 <= spa["p_upper"] + 2e-12


def test_pbo_is_about_one_half_for_noise_and_small_for_a_persistent_winner():
    noise = pbo_cscv(_noise(n=1600, k=40, seed=3), blocks=10)
    assert 0.25 < noise["pbo"] < 0.75
    assert noise["n_splits"] == 252
    x = _noise(n=1600, k=40, seed=4)
    x[:, 5] += 0.0015
    persistent = pbo_cscv(x, blocks=10)
    assert persistent["pbo"] < 0.1 and persistent["mean_oos_of_best"] > 0


def test_pbo_degradation_is_negative_for_noise():
    result = pbo_cscv(_noise(n=1600, k=60, seed=8), blocks=10)
    assert result["degradation_slope"] < 0.2 and result["mean_oos_of_best"] < result["mean_is_of_best"]


# ------------------------------------------------------------------------------------------------------------------------ the algebra, against numbers worked out by hand
def test_the_reality_check_statistic_and_p_value_are_the_root_n_times_the_best_mean_and_the_share_of_null_maxima_at_or_above_it():
    x = np.tile([0.01, 0.0], (4, 1))                                                    # means 1% and 0, n = 4
    boot = np.array([[0.00, 0.00], [0.02, 0.00], [0.00, 0.00], [0.01, 0.00]])
    out = white_reality_check(x, boot=boot)
    # statistic 2 * 0.01; the null maxima are 2 * max(boot - mean) = 0, 0.02, 0 and 0, so one of four reaches the statistic
    assert out["statistic"] == pytest.approx(0.02) and out["p_value"] == pytest.approx(0.25) and out["best_index"] == 0 and out["best_mean"] == pytest.approx(0.01)
    behind = white_reality_check(np.tile([-0.02, -0.01], (4, 1)), boot=np.array([[-0.02, -0.01]] * 4))
    assert behind["best_index"] == 1 and behind["best_mean"] == pytest.approx(-0.01) and behind["statistic"] == pytest.approx(-0.02)
    assert behind["p_value"] == 1.0                                                      # the best of them is below zero: every null maximum is at least that


def test_the_studentised_statistic_ignores_a_candidate_with_no_spread_and_never_goes_below_zero():
    n = 100
    rng = np.random.default_rng(0)
    x = np.column_stack([np.full(n, 0.001), np.full(n, 0.002)])
    boot = np.column_stack([0.001 + 0.001 * rng.standard_normal(2000), np.full(2000, 0.002)])         # the second candidate's bootstrap never varies: no t-statistic
    out = hansen_spa(x, boot=boot)
    assert out["best_index"] == 0 and out["best_t"] == pytest.approx(0.001 / boot[:, 0].std(ddof=1)) and out["statistic"] == pytest.approx(out["best_t"])
    losing = hansen_spa(np.full((n, 1), -0.001), boot=-0.001 + 0.001 * rng.standard_normal((2000, 1)))
    assert losing["best_t"] < 0 and losing["statistic"] == 0.0                         # the statistic is the larger of zero and the best t


@pytest.mark.parametrize("loser_mean, recentred", [(-0.0019, True), (-0.0016, False)])
def test_a_candidate_is_recentred_in_the_consistent_p_value_only_when_it_is_clearly_worse_than_the_benchmark(loser_mean, recentred):
    """With n = 100 the bar is sqrt(2 ln ln 100) = 1.75 studentised units below zero. A candidate at -1.9 is held where it is (like the lower p-value does); one at -1.6 is brought up to zero (like the
    upper one does). Moving the bar, or the sign of the comparison, changes which of the two this is."""
    n, rng = 100, np.random.default_rng(0)
    z = rng.standard_normal((6000, 2))
    x = np.column_stack([np.full(n, 0.001), np.full(n, loser_mean)])
    boot = np.column_stack([0.001 + 0.001 * z[:, 0], loser_mean + 0.001 * z[:, 1]])
    out = hansen_spa(x, boot=boot)
    assert out["best_index"] == 0 and out["best_t"] == pytest.approx(1.0, abs=0.05) and out["p_lower"] < out["p_upper"] - 0.05
    if recentred:
        assert out["p_consistent"] == out["p_lower"]
    else:
        assert out["p_consistent"] == out["p_upper"]
    assert np.sqrt(2.0 * np.log(np.log(n))) == pytest.approx(1.7477, abs=1e-3)


def test_when_nothing_beats_the_benchmark_every_spa_p_value_is_exactly_one():
    """The statistic is the larger of zero and the best t, so a field of losers has statistic 0, and every null maximum (also at least 0) is at or above it: no evidence at all, p = 1 and not about a half."""
    rng = np.random.default_rng(0)
    losing = hansen_spa(np.full((100, 1), -0.001), boot=-0.001 + 0.001 * rng.standard_normal((2000, 1)))
    assert losing["statistic"] == 0.0 and losing["p_lower"] == losing["p_consistent"] == losing["p_upper"] == 1.0


def test_the_documented_defaults_are_2000_resamples_in_blocks_of_21_days_with_seed_7():
    x = _noise(n=300, k=4, seed=3)
    assert np.array_equal(bootstrap_means(x), bootstrap_means(x, 2000, 21.0, 7))
    assert white_reality_check(x) == white_reality_check(x, 2000, 21.0, 7)
    assert hansen_spa(x) == hansen_spa(x, 2000, 21.0, 7)


def test_a_candidate_that_never_moves_cannot_win_the_in_sample_ranking_of_pbo_and_changes_nothing_but_the_denominator_of_the_ranks():
    noise = _noise(n=160, k=4, seed=9)
    with_cash = pbo_cscv(np.column_stack([noise, np.zeros(160)]), blocks=8)
    without = pbo_cscv(noise, blocks=8)
    assert np.isfinite(with_cash["is_best_sharpe"]).all() and np.isfinite(with_cash["oos_of_best_sharpe"]).all()          # a constant has no Sharpe ratio: it ranks last, it is not infinite or undefined
    assert np.allclose(with_cash["is_best_sharpe"], without["is_best_sharpe"]) and np.allclose(with_cash["oos_of_best_sharpe"], without["oos_of_best_sharpe"])


def test_pbo_of_a_persistent_winner_is_zero_and_its_sharpe_is_the_one_worked_out_from_the_blocks():
    blocks, a, d = 4, 0.01, 0.002
    persistent = np.tile([a + d, a - d], blocks)                                         # +1% a day, with a little spread inside each block
    x = np.column_stack([persistent, np.tile([0.02, -0.02], blocks), np.tile([0.03, -0.03], blocks)])       # and two candidates that earn nothing on average
    out = pbo_cscv(x, blocks=blocks)
    assert out["n_splits"] == 6 and out["pbo"] == 0.0 and out["share_oos_negative"] == 0.0
    expected_sharpe = a / (d * math.sqrt(4.0 / 3.0)) * math.sqrt(252.0)                    # the sample variance of two blocks of (a + d, a - d) is 4 d^2 / 3
    assert np.allclose(out["is_best_sharpe"], expected_sharpe) and np.allclose(out["oos_of_best_sharpe"], expected_sharpe)
    assert np.allclose(out["logit"], np.log(3.0))                                          # best of three out of sample: relative rank 3/4, logit ln 3
    assert out["mean_oos_of_best"] == pytest.approx(expected_sharpe) and out["mean_is_of_best"] == pytest.approx(expected_sharpe) and math.isnan(out["degradation_slope"])
    assert pbo_cscv(x, blocks=blocks, annualise=12.0)["mean_is_of_best"] == pytest.approx(a / (d * math.sqrt(4.0 / 3.0)) * math.sqrt(12.0))


def test_pbo_counts_the_splits_in_which_the_in_sample_winner_is_the_worst_out_of_sample_and_the_slope_of_that_fall_is_minus_three():
    """Two candidates earn +2% in alternate blocks (and -2% in the others). Where the in-sample blocks happen to be the good ones, such a candidate wins by twice as much as the steady one and then
    loses by as much out of sample: two splits of six. In the other four, the steady candidate wins both halves."""
    blocks, d = 4, 0.002
    steady = np.tile([0.01 + d, 0.01 - d], blocks)
    swing = np.concatenate([[0.02 + d, 0.02 - d], [-0.02 + d, -0.02 - d]] * 2)
    other = np.roll(swing, 2)
    out = pbo_cscv(np.column_stack([steady, swing, other]), blocks=blocks)
    s = 0.01 / (d * math.sqrt(4.0 / 3.0)) * math.sqrt(252.0)
    assert out["pbo"] == pytest.approx(2.0 / 6.0) and out["share_oos_negative"] == pytest.approx(2.0 / 6.0)
    assert sorted(np.round(out["logit"], 6)) == sorted(np.round([np.log(1 / 3.0)] * 2 + [np.log(3.0)] * 4, 6))      # lowest of three: relative rank 1/4; highest: 3/4
    assert out["mean_is_of_best"] == pytest.approx(s * 8.0 / 6.0) and out["mean_oos_of_best"] == pytest.approx(0.0, abs=1e-9)
    assert out["degradation_slope"] == pytest.approx(-3.0)                                   # points (s, s) four times and (2s, -2s) twice


def test_pbo_of_one_candidate_sits_at_the_median_and_an_odd_number_of_blocks_is_refused():
    x = np.random.default_rng(1).normal(0.0, 0.01, size=(80, 1))
    out = pbo_cscv(x, blocks=4)
    assert out["pbo"] == 1.0 and np.allclose(out["logit"], 0.0) and out["n_splits"] == 6                 # rank 1 of 1: relative rank 1/2, logit 0, which counts as overfitting
    two = pbo_cscv(np.random.default_rng(2).normal(0.0, 0.01, size=(80, 2)), blocks=4)
    assert set(np.round(np.abs(two["logit"]), 6)) <= {round(math.log(2.0), 6)}                           # ranks 1 or 2 of 2: relative ranks 1/3 and 2/3
    with pytest.raises(ValueError, match="blocks must be even"):
        pbo_cscv(x, blocks=5)


def test_pbo_drops_the_oldest_rows_that_do_not_fill_a_block():
    rng = np.random.default_rng(4)
    core = rng.normal(0.0, 0.01, size=(80, 3))
    padded = np.vstack([np.full((3, 3), 5.0), core])                                      # three stray rows at the start of 83
    a, b = pbo_cscv(core, blocks=4), pbo_cscv(padded, blocks=4)
    assert np.allclose(a["logit"], b["logit"]) and a["pbo"] == b["pbo"] and np.allclose(a["is_best_sharpe"], b["is_best_sharpe"])


def test_the_bootstrap_means_are_the_means_of_the_resampled_days_and_a_batch_of_any_size_gives_the_same_draws():
    x = np.arange(60, dtype=float).reshape(30, 2)
    one = bootstrap_means(x, n_samples=50, mean_block=5, seed=3, batch=50)
    many = bootstrap_means(x, n_samples=50, mean_block=5, seed=3, batch=7)
    assert one.shape == (50, 2) and np.allclose(one, many) and np.all(one >= x.min(axis=0)) and np.all(one <= x.max(axis=0))
    assert np.allclose(one[:, 1] - one[:, 0], 1.0)                                          # the same days for both columns: they differ by exactly the offset between them


def _square(task):
    return task["a"] ** 2 + task["b"]


def test_parallel_results_equal_serial_results_in_task_order():
    tasks = [{"a": i, "b": 0.5 * i} for i in range(53)]
    serial = run_tasks(_square, tasks, "serial")
    parallel = run_tasks(_square, tasks, "joblib", n_jobs=2, chunk_size=5)
    assert serial == parallel


def test_unknown_backend_and_seed_determinism():
    with pytest.raises(ValueError):
        run_tasks(_square, [{"a": 1, "b": 1}, {"a": 2, "b": 2}], "mpi")
    assert task_seed({"a": 1, "b": 2}) == task_seed({"b": 2, "a": 1})
    assert task_seed({"a": 1}) != task_seed({"a": 2})
    assert "serial" in available_backends()


def test_dask_backend_matches_serial_when_dask_is_installed():
    pytest.importorskip("dask.distributed")
    tasks = [{"a": i, "b": 0.5 * i} for i in range(23)]
    assert run_tasks(_square, tasks, "dask", n_jobs=2, chunk_size=4) == run_tasks(_square, tasks, "serial")
