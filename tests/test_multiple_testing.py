"""Reality Check, SPA, PBO and the parallel executor (Generation 4)."""

from __future__ import annotations

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
