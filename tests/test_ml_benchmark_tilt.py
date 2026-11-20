"""The benchmark_tilt position rule: a long-only book tilted around the equal-weight benchmark by rank.

Known answers: at tilt 0 the book IS the benchmark; it is long only and fully invested at any tilt in [0, 1]; weight increases with the score; it keeps the benchmark's market exposure (beta about 1
in a world with a common factor); a perfect-foresight score (a diagnostic control, never a strategy) earns a positive active return and random scores earn none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.config import experiment_config, recommended_config
from src.ml.signals import benchmark_tilt, build_weights


def _scores(n_days=300, n_assets=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(rng.normal(size=(n_days, n_assets)), index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_tilt_zero_is_the_equal_weight_benchmark():
    s = _scores()
    w = benchmark_tilt(s, 0.0)
    assert np.allclose(w.to_numpy(), 1.0 / s.shape[1])


@pytest.mark.parametrize("tilt", [0.0, 0.25, 0.5, 1.0])
def test_long_only_fully_invested_and_ordered_by_score(tilt):
    s = _scores(seed=1)
    w = benchmark_tilt(s, tilt)
    assert (w >= -1e-15).all().all()
    assert np.allclose(w.sum(axis=1), 1.0)
    if tilt > 0:                                                    # a higher score never gets less weight
        for d in s.index[:20]:
            order = s.loc[d].sort_values().index
            assert (np.diff(w.loc[d, order].to_numpy()) > 0).all()


def test_tilt_one_gives_the_best_twice_the_benchmark_and_the_worst_nothing():
    s = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], columns=list("abcde"))
    w = benchmark_tilt(s, 1.0).iloc[0]
    assert w["e"] == pytest.approx(2 / 5) and w["a"] == pytest.approx(0.0) and w["c"] == pytest.approx(1 / 5)


def test_missing_scores_are_not_held_and_a_lone_asset_is_the_whole_book():
    s = pd.DataFrame([[1.0, np.nan, 3.0], [np.nan, np.nan, 2.0], [np.nan, np.nan, np.nan]], columns=list("abc"))
    w = benchmark_tilt(s, 0.5)
    assert w.loc[0, "b"] == 0.0 and w.loc[0].sum() == pytest.approx(1.0) and w.loc[0, "c"] > w.loc[0, "a"]
    assert w.loc[1, "c"] == pytest.approx(1.0)
    assert w.loc[2].sum() == 0.0                                    # nothing scored: flat


def test_a_tilt_above_one_would_short_and_is_refused():
    with pytest.raises(ValueError, match="between 0 and 1"):
        benchmark_tilt(_scores(), 1.5)


def test_build_weights_reads_the_rule_and_its_tilt():
    s = _scores(seed=2)
    cfg = {**experiment_config()["signal"], "method": "benchmark_tilt", "tilt": 0.8, "max_weight": None, "holding_period": 1}
    assert np.allclose(build_weights(s, cfg).to_numpy(), benchmark_tilt(s, 0.8).to_numpy())


def test_the_recommended_set_up_is_like_for_like_with_the_equal_weight_benchmark_and_valid():
    cfg = experiment_config(defaults=recommended_config())
    assert cfg["signal"]["method"] == "benchmark_tilt" and cfg["backtest"]["target_vol"] is None and cfg["backtest"]["rebalance"] == "monthly"
    assert cfg["dataset"]["target"] == "excess_ret_21" and cfg["dataset"]["horizon"] == 21
    assert experiment_config()["signal"]["method"] == "top_n_long_short", "the library defaults are unchanged"


def test_the_lab_form_starts_from_the_recommended_set_up(tmp_path):
    from src.utils.config import load_config
    from src.webapp.api import App
    from tests.app_roots import isolated_root

    config = load_config()
    app = App(config, root=isolated_root(tmp_path / "root", config))
    d = app.extra_get["/api/ml/catalog"]({})["defaults"]
    assert d["signal"]["method"] == "benchmark_tilt" and d["dataset"]["target"] == "excess_ret_21" and d["backtest"]["rebalance"] == "monthly"


def _world(seed=3, n_days=1500, n_assets=8):
    """Returns with a strong common factor and an asset-specific part. Weights at the close of t earn the return of t + 1 here (no engine lag: this checks the rule, not the timing)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    market = rng.normal(0.0006, 0.012, n_days)
    idio = rng.normal(0.0, 0.015, (n_days, n_assets))
    r = pd.DataFrame(market[:, None] + idio, index=idx, columns=[f"S{i}" for i in range(n_assets)])
    return r


def _book_returns(w, r):
    return (w.shift(1) * r).sum(axis=1).iloc[1:]


def test_keeps_the_benchmark_exposure_and_only_skill_earns_active_return():
    r = _world()
    bench = r.mean(axis=1).iloc[1:]
    perfect = r.shift(-1)                                           # tomorrow's return, known today: a diagnostic control only
    noise = pd.DataFrame(np.random.default_rng(9).normal(size=r.shape), index=r.index, columns=r.columns)
    for name, score in (("perfect", perfect), ("noise", noise)):
        book = _book_returns(benchmark_tilt(score, 0.5), r).iloc[:-1]
        b = bench.reindex(book.index)
        beta = book.cov(b) / b.var()
        active = book - b
        t = active.mean() / active.std() * np.sqrt(len(active))
        assert 0.9 < beta < 1.1, (name, beta)                     # the market exposure of the benchmark, whatever the score
        if name == "perfect":
            assert t > 10
        else:
            assert abs(t) < 3


def test_a_target_or_a_horizon_asked_alone_brings_the_other_with_it():
    assert experiment_config({"dataset": {"target": "fwd_ret_5"}})["dataset"]["horizon"] == 5
    assert experiment_config({"dataset": {"horizon": 10}})["dataset"]["target"] == "fwd_ret_10"
    assert experiment_config({"dataset": {"horizon": 10}}, defaults=recommended_config())["dataset"]["target"] == "excess_ret_10"
    both = experiment_config({"dataset": {"target": "rank_21", "horizon": 21}})["dataset"]
    assert (both["target"], both["horizon"]) == ("rank_21", 21)
