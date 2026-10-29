"""Combining and comparing forecast distributions: the Bates-Granger weights, the weighted pool, the pairwise skill and Diebold-Mariano matrix, the cumulative gain over the random walk and the terminal
histograms, against numbers worked out by hand; and the weighted blend never uses a score that was not yet known."""

from __future__ import annotations

import numpy as np
import pytest

from src.forecast import evaluate as ev
from src.forecast.base import History
from src.forecast.registry import BASELINE, BLEND_WEIGHTED, METHODS, catalog
from tests import forecast_worlds as W


# ------------------------------------------------------------------------------------------------------------------ the weights and the pool
def test_the_weights_are_proportional_to_the_inverse_of_the_mean_loss():
    got = ev.inverse_loss_weights({"a": np.full(20, 1.0), "b": np.full(20, 2.0), "c": np.full(20, 4.0)})
    assert got == pytest.approx({"a": 4 / 7, "b": 2 / 7, "c": 1 / 7}) and sum(got.values()) == pytest.approx(1.0)


def test_equal_losses_give_equal_weights_and_scores_not_yet_known_are_ignored():
    assert ev.inverse_loss_weights({"a": np.full(12, 3.0), "b": np.full(12, 3.0)}) == pytest.approx({"a": 0.5, "b": 0.5})
    with_gaps = {"a": np.array([1.0] * 10 + [np.nan] * 5), "b": np.array([2.0] * 10 + [np.nan] * 5)}
    assert ev.inverse_loss_weights(with_gaps) == pytest.approx({"a": 2 / 3, "b": 1 / 3})


def test_too_few_scored_forecasts_or_a_zero_loss_means_no_weights():
    assert ev.inverse_loss_weights({"a": np.full(9, 1.0), "b": np.full(9, 2.0)}) is None
    assert ev.inverse_loss_weights({"a": np.full(20, 0.0), "b": np.full(20, 2.0)}) is None
    assert ev.inverse_loss_weights({"a": np.full(20, 1.0), "b": np.full(5, 2.0)}) is None


def test_the_weighted_pool_takes_draws_from_each_member_in_proportion_to_its_weight():
    rng = np.random.default_rng(0)
    members = {"a": rng.uniform(0.0, 1.0, (5000, 3)), "b": rng.uniform(10.0, 11.0, (5000, 3))}
    pool = ev.combine_weighted(members, {"a": 0.7, "b": 0.3}, np.random.default_rng(1), 1000)
    assert pool.shape == (1000, 3) and (pool[:, 0] < 5).sum() == 700 and (pool[:, 0] > 5).sum() == 300         # the rows are whole paths, drawn from one member
    rows_from_a = pool[pool[:, 0] < 5]
    assert (rows_from_a < 5).all()                                                                            # a draw keeps its path together
    tiny = ev.combine_weighted(members, {"a": 0.9999, "b": 0.0001}, np.random.default_rng(2), 100)
    assert (tiny[:, 0] > 5).sum() == 1                                                                         # every member contributes at least one draw


# ------------------------------------------------------------------------------------------------------------------ pairwise skill and cumulative gain
class FakeTape:
    def __init__(self, crps):
        self.crps = np.asarray(crps, dtype=float).reshape(-1, 1)


def test_the_pairwise_matrix_has_skill_against_each_other_method_and_one_sided_p_values():
    rng = np.random.default_rng(3)
    n = 200
    noise = rng.normal(0, 0.01, n)
    tapes = {"random_walk": FakeTape(0.30 + noise), "a": FakeTape(0.20 + noise * 0.5), "b": FakeTape(0.40 + noise)}
    out = ev._pairwise(["random_walk", "a", "b"], tapes, n, 0, 0, 5)
    ids = out["ids"]
    i, a, b = ids.index("random_walk"), ids.index("a"), ids.index("b")
    mean = {k: float(t.crps.mean()) for k, t in tapes.items()}
    assert out["horizon"] == 5 and out["skill"][a][b] == pytest.approx(1 - mean["a"] / mean["b"], rel=1e-12)
    assert out["skill"][b][a] == pytest.approx(1 - mean["b"] / mean["a"], rel=1e-12) and out["skill"][a][a] is None
    assert out["skill"][a][b] > 0 > out["skill"][b][a] and out["skill"][a][i] > 0 > out["skill"][b][i]
    assert out["p"][a][b] < 1e-6 and out["p"][b][a] > 1 - 1e-6                                               # a is always better than b: the test says so, and says b is not better than a


def test_the_pairwise_matrix_leaves_out_pairs_with_too_few_scored_forecasts():
    tapes = {"random_walk": FakeTape(np.r_[np.full(5, 0.3), np.full(10, np.nan)]), "a": FakeTape(np.full(15, 0.2))}
    out = ev._pairwise(["random_walk", "a"], tapes, 15, 0, 0, 5)
    assert out["skill"] == [[None, None], [None, None]] and out["p"] == [[None, None], [None, None]]


def test_the_cumulative_gain_is_the_running_sum_of_the_crps_saved_over_the_random_walk_in_units_of_its_average():
    base = np.array([0.30, 0.30, 0.20, 0.40, np.nan, 0.30])
    method = np.array([0.20, 0.35, 0.10, 0.50, 0.25, np.nan])
    out, scale = ev._cumulative_gain(["random_walk", "m"], {"random_walk": FakeTape(base), "m": FakeTape(method)}, 6, 0)
    assert scale == pytest.approx(0.30)
    expected = np.cumsum([0.10, -0.05, 0.10, -0.10, 0.0, 0.0]) / 0.30                                          # an origin missing for either contributes nothing
    assert out["m"] == pytest.approx(expected.tolist()) and list(out) == ["m"]


# ------------------------------------------------------------------------------------------------------------------ end to end
SMALL = {"years": 4.0, "step": 14, "n_paths": 200}


@pytest.fixture(scope="module")
def result():
    h = W.to_history(W.garch(2100, seed=9), high_low=True)
    return h, ev.evaluate(h, ["riskmetrics", "garch", "blend", "blend_weighted"], 5, "standard", 0, **SMALL)


def test_the_evaluation_reports_pairwise_gain_and_blend_weights_for_the_headline_horizon(result):
    h, out = result
    n = out["origins"]["count"]
    assert out["pairwise"]["ids"] == [BASELINE, "riskmetrics", "garch", "blend", "blend_weighted"] and out["pairwise"]["horizon"] == 5
    assert len(out["origin_dates"]) == n and out["origin_dates"] == sorted(out["origin_dates"])
    assert set(out["gain"]) == {"riskmetrics", "garch", "blend", "blend_weighted"} and all(len(v) == n for v in out["gain"].values())
    weights = out["blend_weights"]
    assert set(weights) == {"riskmetrics", "garch"} and sum(weights.values()) == pytest.approx(1.0)
    by_id = {m["id"]: m for m in out["methods"]}
    crps = {k: by_id[k]["scores"]["5"]["crps"] for k in weights}
    assert weights["garch"] / weights["riskmetrics"] == pytest.approx(crps["riskmetrics"] / crps["garch"], rel=1e-6)
    assert by_id[BLEND_WEIGHTED]["scores"]["5"]["crps"] is not None and by_id[BLEND_WEIGHTED]["label"].startswith("Blend weighted")


def test_the_weighted_blend_never_uses_a_score_that_was_not_known_at_its_origin():
    base = W.to_history(W.garch(1700, seed=4))
    cut = 1600                                                                                               # the last origins are in the 380 days before the end: the early ones finish before the cut
    altered = base.close.copy()
    altered[cut + 1:] = altered[cut + 1:] * np.exp(np.random.default_rng(0).normal(0, 0.3, len(altered) - cut - 1))
    noisy = History(altered, base.index)
    names = ["riskmetrics", "garch", "blend_weighted"]
    a = ev.evaluate(base, names, 5, "standard", 0, years=1.5, step=7, n_paths=150, refit_days=21)
    b = ev.evaluate(noisy, names, 5, "standard", 0, years=1.5, step=7, n_paths=150, refit_days=21)
    origins = ev._origins(base, ev.settings_for("standard", base, names, 5, years=1.5, step=7, n_paths=150, refit_days=21))
    known = [k for k, i in enumerate(origins) if i + 5 <= cut]
    assert len(known) > 20
    for name in names:
        raw_a = np.array(a["gain"][name][:len(known)]) * a["gain_scale"]                                          # back to the raw sum of CRPS saved: the scale uses every origin
        raw_b = np.array(b["gain"][name][:len(known)]) * b["gain_scale"]
        assert raw_a == pytest.approx(raw_b, abs=1e-12), name                                                    # the blend's draws and scores up to the cut are the same


def test_the_terminal_histogram_shares_its_bins_and_counts_every_draw():
    rng = np.random.default_rng(5)
    samples = {"a": np.cumsum(rng.normal(0, 0.01, (4000, 21)), axis=1), "b": np.cumsum(rng.normal(0, 0.02, (4000, 21)), axis=1)}
    out = ev._terminal_histogram(samples, 21)
    edges = np.array(out["edges"])
    assert len(edges) == ev.HIST_BINS + 1 and out["horizon"] == 21 and out["draws"] == {"a": 4000, "b": 4000}
    for name, counts in out["counts"].items():
        assert len(counts) == ev.HIST_BINS and sum(counts) >= 0.99 * 4000                                         # the central 99.8% of all the draws pooled
        assert np.histogram(samples[name][:, 20], bins=edges)[0].tolist() == counts
    assert out["counts"]["b"][0] > out["counts"]["a"][0] and out["counts"]["a"][ev.HIST_BINS // 2] > out["counts"]["b"][ev.HIST_BINS // 2]      # the wider forecast has more in the tails


def test_the_forecast_from_the_last_day_has_the_extra_levels_the_tail_measures_and_the_weighted_pool():
    h = W.to_history(W.garch(1500, seed=2))
    weights = {"riskmetrics": 0.25, "garch": 0.75}
    now = ev.forecast_now(h, [BASELINE, "riskmetrics", "garch", "blend", "blend_weighted"], 21, n_paths=800, seed=1, blend_weights=weights)
    fan = now["methods"]["garch"]["fan"]
    assert list(fan) == ["0.01", "0.05", "0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99"]
    assert all(fan["0.01"][k] <= fan["0.05"][k] <= fan["0.5"][k] <= fan["0.95"][k] <= fan["0.99"][k] for k in range(21))
    s = now["methods"]["garch"]["summary"]["21"]
    assert s["es_05"] < s["q05"] < s["median"] and s["vol_annual"] == pytest.approx(s["sd"] * np.sqrt(252 / 21), rel=1e-9)
    assert now["methods"]["blend_weighted"]["info"]["weights"] == {"riskmetrics": 0.25, "garch": 0.75}
    assert now["methods"]["blend"]["info"] == {"members": ["riskmetrics", "garch"]} and set(now["hist"]["counts"]) == set(now["methods"])
    without = ev.forecast_now(h, [BASELINE, "riskmetrics", "garch", "blend_weighted"], 21, n_paths=300, seed=1)
    assert "equal" in without["methods"]["blend_weighted"]["info"]["weights"]


def test_the_catalogue_lists_both_blends_last_in_the_combination_family():
    ids = [m["id"] for m in catalog()]
    assert ids[-2:] == ["blend", "blend_weighted"] and all(m["family"] == "Combination" for m in catalog()[-2:]) and BLEND_WEIGHTED not in METHODS
