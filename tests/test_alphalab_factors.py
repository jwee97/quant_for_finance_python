"""Factor evaluation, comparison and combination on data whose answer is known.

A synthetic market: ``returns = beta * planted + noise``, where the PLANTED factor predicts the next-period return with a known rank IC, a second factor is its noisy copy (so combining helps a little), and a third is
pure noise. What is checked: the planted factor is found and the noise is not, after correction for having looked at several; the interval contains the IC that was planted; a cross-section too thin to rank is
refused (two assets give a +1/-1 rank correlation and must never be scored); neutralising a control the factor was built from removes its IC; and the weights of every combination use only the past (changing
returns AFTER a date changes nothing at or before it) and are judged on a period they were not fitted on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alphalab import factors as fc
from src.models.regression import cross_sectional_ic

N_DATES, N_ASSETS = 900, 30


def market(seed=1, ic_strength=0.25):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=N_DATES)
    cols = [f"A{i:02d}" for i in range(N_ASSETS)]
    planted = pd.DataFrame(rng.normal(size=(N_DATES, N_ASSETS)), index=idx, columns=cols)
    noise = pd.DataFrame(rng.normal(size=(N_DATES, N_ASSETS)), index=idx, columns=cols)
    fwd = ic_strength * planted + np.sqrt(1 - ic_strength ** 2) * noise                    # the forward return: correlation ic_strength with the planted factor
    signals = {"planted": planted, "noisy_copy": planted + 2.0 * pd.DataFrame(rng.normal(size=planted.shape), index=idx, columns=cols),
               "noise": pd.DataFrame(rng.normal(size=planted.shape), index=idx, columns=cols)}
    return signals, fwd


def test_the_planted_factor_is_found_its_interval_covers_the_truth_and_noise_is_not():
    signals, fwd = market()
    e = fc.evaluate_factor(signals["planted"], fwd, horizon=1)
    ic = e["rank_ic"]
    assert 0.15 < ic["mean"] < 0.30 and ic["low"] < 0.25 * 0.97 < ic["high"] + 0.05 and ic["t"] > 10 and ic["p"] < 1e-6
    assert e["quantiles"]["long_short_mean"] > 0 and e["quantiles"]["monotonicity"] > 0.8 and e["quantiles"]["buckets"] == 5
    noise = fc.evaluate_factor(signals["noise"], fwd, horizon=1)
    assert abs(noise["rank_ic"]["mean"]) < 0.03 and noise["rank_ic"]["low"] < 0 < noise["rank_ic"]["high"]
    assert e["pearson_ic"]["mean"] > 0.15 and e["n_assets_median"] == N_ASSETS and e["stability"]["same_sign_share"] == 1.0
    assert 0.3 < e["turnover"] < 0.4, "independent rankings every date: a third of the percentile range changes per period"
    assert e["coverage"]["by_date_min"] == 1.0 and e["coverage"]["extreme_share"] < 0.001


def test_a_factor_chosen_among_many_needs_the_corrected_p_value():
    signals, fwd = market()
    rng = np.random.default_rng(5)
    many = dict(signals)
    for i in range(12):                                                                       # twelve more noise factors: some will look good by luck
        many[f"junk{i}"] = pd.DataFrame(rng.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
    out = fc.compare_factors(many, fwd, horizon=1)
    by = {r["factor"]: r for r in out["table"]}
    assert by["planted"]["survives_holm"] and by["planted"]["survives_bh"]
    assert not any(by[f"junk{i}"]["survives_holm"] for i in range(12)) and not by["noise"]["survives_holm"]
    assert all(r["p_holm"] >= r["p_raw"] - 1e-12 for r in out["table"] if np.isfinite(r["p_raw"])), "correction never lowers a p-value"
    assert out["n_tested"] == 15 and out["table"][0]["factor"] in ("planted", "noisy_copy")


def test_holm_and_benjamini_hochberg_match_hand_computed_values():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20}
    h, bh = fc.holm(p), fc.benjamini_hochberg(p)
    assert h["a"] == pytest.approx(0.04) and h["c"] == pytest.approx(0.09) and h["b"] == pytest.approx(0.09) and h["d"] == pytest.approx(0.20)      # (4*.01, 3*.03, 2*.04 -> max so far .09, 1*.2)
    assert bh["a"] == pytest.approx(0.04) and bh["c"] == pytest.approx(0.0533333, abs=1e-6) and bh["b"] == pytest.approx(0.0533333, abs=1e-6) and bh["d"] == pytest.approx(0.20)
    assert np.isnan(fc.holm({"x": float("nan")})["x"])


def test_a_cross_section_too_thin_to_rank_is_refused_never_scored():
    signals, fwd = market()
    two = list(fwd.columns[:2])
    with pytest.raises(fc.FactorError, match="too thin to rank"):
        fc.evaluate_factor(signals["planted"][two], fwd[two], horizon=1)
    with pytest.raises(fc.FactorError, match="not allowed"):
        fc.evaluate_factor(signals["planted"], fwd, horizon=1, min_assets=2)
    with pytest.raises(ValueError, match="at least 3 assets"):
        cross_sectional_ic(signals["planted"][two], fwd[two], min_assets=2)                   # the low-level function refuses too
    sparse = signals["planted"].copy()
    sparse.iloc[:, 4:] = np.nan                                                                # four assets with a signal: below the default of five
    with pytest.raises(fc.FactorError, match="median of 4 assets"):
        fc.evaluate_factor(sparse, fwd, horizon=1)


def test_neutralising_the_control_a_factor_is_built_from_removes_its_information():
    signals, fwd = market()
    rng = np.random.default_rng(9)
    size = pd.DataFrame(rng.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
    fwd2 = fwd + 0.25 * size                                                                    # the control itself predicts returns
    built = 3.0 * size + 0.3 * signals["planted"]                                               # a factor that is mostly the control
    raw = fc.evaluate_factor(built, fwd2, horizon=1)
    neut = fc.evaluate_factor(built, fwd2, horizon=1, controls={"size": size})
    assert raw["rank_ic"]["mean"] > 0.15
    assert neut["neutralised"]["controls"] == ["size"] and 0.0 < neut["neutralised"]["rank_ic"]["mean"] < raw["rank_ic"]["mean"], "what is left is the planted part"
    pure = fc.evaluate_factor(size, fwd2, horizon=1, controls={"size": size})
    assert pure["neutralised"]["nothing_left"] is True and np.isnan(pure["neutralised"]["rank_ic"]["mean"]), "a factor that IS the control has nothing left once it is removed (and rounding noise is not scored)"


def test_decay_is_reported_when_returns_are_given():
    signals, fwd = market()
    rets = fwd.shift(1)                                                                         # the one-period return OF each date: the planted factor of t predicts the return of t+1 only
    e = fc.evaluate_factor(signals["planted"], fwd, horizon=1, returns=rets)
    assert set(e["decay"]) == {"1", "2", "3", "5", "10", "21"}
    assert e["decay"]["1"] > 0.15 and all(abs(e["decay"][k]) < 0.06 for k in ("5", "10", "21")), "all of the information is in the first period"


def test_combinations_use_only_the_past_and_are_judged_out_of_sample():
    signals, fwd = market()
    out = fc.combine_factors(signals, fwd, horizon=1)
    assert out["train_dates"][1] < out["test_dates"][0] and set(out["verdicts"]) == {"equal_weight", "ic_weighted", "ridge", "orthogonalised"}
    assert out["best_single_on_training"] == "planted", "chosen on the training period only"
    oos = out["out_of_sample"]
    assert oos["single:planted"]["mean"] > 0.15 and abs(oos["single:noise"]["mean"]) < 0.05
    assert oos["ic_weighted"]["mean"] > oos["equal_weight"]["mean"], "weighting by past IC beats weighting a noise factor equally"
    assert oos["ridge"]["mean"] > 0.15
    assert out["verdicts"]["equal_weight"]["improves_on_baselines"] is False, "equal weight with a noise factor does not beat the best single factor"
    # no look-ahead: change the forward returns AFTER the first test date; scores up to and including it must not move
    cut = pd.Timestamp(out["test_dates"][0])
    changed = fwd.copy()
    changed.loc[changed.index > cut] += 5.0 * np.random.default_rng(3).normal(size=changed.loc[changed.index > cut].shape)
    out2 = fc.combine_factors(signals, changed, horizon=1)
    assert out2["training_ic"] == out["training_ic"], "the training period's ICs are untouched by later outcomes"
    assert out2["best_single_on_training"] == out["best_single_on_training"]


def test_too_little_history_or_a_single_factor_is_refused():
    signals, fwd = market()
    with pytest.raises(fc.FactorError, match="at least two"):
        fc.combine_factors({"planted": signals["planted"]}, fwd)
    short = {k: v.iloc[:150] for k, v in signals.items()}
    with pytest.raises(fc.FactorError, match="not enough history"):
        fc.combine_factors(short, fwd.iloc[:150])


def test_discovery_reports_coverage_redundancy_and_refuses_to_rank_a_thin_cross_section():
    signals, fwd = market()
    gappy = signals["planted"].copy()
    gappy.iloc[:300, :10] = np.nan                                                             # ten assets have no value for the first 300 days
    d = fc.discovery({**signals, "gappy": gappy, "copy": signals["planted"] * 2.0 + 1.0})
    g = d["signals"]["gappy"]
    assert g["coverage"]["by_date_min"] < 0.7 and g["coverage"]["by_asset"][fwd.columns[0]] < 0.7 and g["coverage"]["worst_months"] and g["rankable"] is True
    assert ["planted", "copy"] in [c[:2] for c in d["clusters"]] or any({"planted", "copy"} <= set(c) for c in d["redundant"]), "a rescaled copy is redundant with the original"
    assert d["signals"]["noise"]["drift"]["p"] > 0.01 and abs(d["signals"]["noise"]["rank_persistence"]["lag1"]) < 0.05
    m = d["correlation"]["matrix"]
    assert m[0][0] == 1.0 and abs(m[0][d["correlation"]["names"].index("copy")] - 1.0) < 1e-9
    thin = fc.discovery({"two": signals["planted"][list(fwd.columns[:2])]})
    assert thin["signals"]["two"]["rankable"] is False and "at least 5" in thin["signals"]["two"]["reason"] and thin["signals"]["two"]["rank_persistence"]["lag1"] != thin["signals"]["two"]["rank_persistence"]["lag1"]
    shifted = signals["noise"].copy()
    shifted.iloc[len(shifted) // 2:] = shifted.iloc[len(shifted) // 2:] * 3 + 2
    assert fc.discovery({"s": shifted})["signals"]["s"]["drift"]["p"] < 1e-6, "a signal whose scale and level change half-way is flagged as drifting"
