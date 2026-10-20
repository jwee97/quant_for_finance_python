"""Putting alphas together, against known truth: duplicates are found and grouped, the combining rules and the machine-learning models use only what had been realised by each date (shown by changing the
future and watching nothing earlier move), the planted alphas carry the weight, and a model that learns nothing is not credited with having beaten a plain average."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import combine as C
from src.fundamentals import factors as F
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.evaluate import rank_ic
from src.fundamentals.research import ResearchContext, UniverseSpec, horizon_days
from src.fundamentals.service import FundamentalsService
from src.models.regression import cross_sectional_ic
from src.signals.transform import cross_sectional_zscore

NAMES = ["d_roic", "fcf_yield", "asset_growth", "earnings_yield", "roic", "mom_12_1", "log_market_cap", "gross_profitability"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("combine"))
    svc.load_simulated(n_companies=100, start_year=2009, end_year=2018, seed=13, planted={"delta_roic": 0.011, "fcf_yield": 0.0068, "asset_growth": 0.0045})
    ctx = ResearchContext(svc.store, UniverseSpec())
    signals = {n: F.compute(ctx, n) * (1.0 if F.get(n).direction >= 0 else -1.0) for n in NAMES}
    return ctx, signals


def panel(seed: int, rows: int = 30, cols: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(rows, cols)), index=pd.date_range("2020-01-31", periods=rows, freq="ME"), columns=range(cols))


def future_changed(ctx: ResearchContext, after: int, horizon: str = "1m", seed: int = 0) -> ResearchContext:
    """The same context, but the returns that follow every date from row ``after`` on are replaced by noise: anything computed for an earlier date that moves has looked at them."""
    fwd = ctx.forward(horizon)
    other = copy.copy(ctx)
    other._cache = dict(ctx._cache)
    noisy = fwd.copy()
    noise = np.random.default_rng(seed).normal(0.0, 0.1, size=fwd.iloc[after:].shape)
    noisy.iloc[after:] = np.where(fwd.iloc[after:].notna().to_numpy(), noise, np.nan)
    other._cache[("forward", horizon_days(horizon))] = noisy
    return other


# ------------------------------------------------------------------------------------------------------------------------ how alike alphas are
def test_correlation_finds_copies_opposites_and_strangers():
    a = panel(1)
    noisy = a + panel(3) * 1.2
    corr = C.correlation({"a": a, "twice_a": a * 3.0 + 1.0, "minus_a": -a, "other": panel(2), "noisy": noisy})
    assert corr.loc["a", "twice_a"] == pytest.approx(1.0) and corr.loc["a", "minus_a"] == pytest.approx(-1.0)
    assert abs(corr.loc["a", "other"]) < 0.08 and 0.4 < corr.loc["a", "noisy"] < 0.8
    assert np.allclose(np.diag(corr.to_numpy()), 1.0) and np.allclose(corr.to_numpy(), corr.to_numpy().T, equal_nan=True)
    assert corr.loc["a", "noisy"] == pytest.approx(float(cross_sectional_ic(a, noisy, "spearman", 10).mean()), abs=1e-9)


def test_correlation_with_different_coverage_stays_close_to_spearman_on_the_common_companies():
    a, b = panel(4), panel(5)
    b = 0.6 * a + 0.8 * b
    rng = np.random.default_rng(6)
    holes = pd.DataFrame(rng.uniform(size=a.shape) < 0.15, index=a.index, columns=a.columns)
    a_holey = a.mask(holes)
    ours = C.correlation({"a": a_holey, "b": b}).loc["a", "b"]
    exact = float(cross_sectional_ic(a_holey, b, "spearman", 10).mean())
    assert ours == pytest.approx(exact, abs=0.03) and 0.4 < ours < 0.7


def test_correlation_skips_dates_with_too_few_common_companies_and_constant_signals():
    a, b = panel(7), panel(8)
    sparse = a.copy()
    sparse.iloc[:, 5:] = np.nan                                             # only five companies in common with anything
    assert np.isnan(C.correlation({"a": sparse, "b": b}, min_assets=10).loc["a", "b"])
    assert np.isfinite(C.correlation({"a": sparse, "b": b}, min_assets=5).loc["a", "b"])
    flat = a * 0.0 + 1.0                                                    # no ranking at all
    assert np.isnan(C.correlation({"a": a, "flat": flat}).loc["a", "flat"])


def test_effective_number_counts_independent_ideas():
    k = 6
    assert C.effective_number(pd.DataFrame(np.eye(k))) == pytest.approx(k)
    assert C.effective_number(pd.DataFrame(np.ones((k, k)))) == pytest.approx(1.0)
    two_pairs = np.kron(np.eye(2), np.ones((2, 2)))
    assert C.effective_number(pd.DataFrame(two_pairs)) == pytest.approx(2.0)
    with_gaps = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]])                 # unknown correlation is treated as none
    assert C.effective_number(with_gaps) == pytest.approx(2.0)


def test_clusters_put_copies_and_opposites_together_and_order_keeps_them_side_by_side():
    a, d = panel(1), panel(2)
    sig = {"a": a, "twice_a": a * 3.0 + 1.0, "minus_a": -a, "d": d, "near_d": d + 0.3 * panel(9), "lonely": panel(10)}
    corr = C.correlation(sig)
    tree = C.cluster(corr, cut=0.5)
    lab = tree["labels"]
    assert lab["a"] == lab["twice_a"] == lab["minus_a"] and lab["d"] == lab["near_d"]
    assert len({lab["a"], lab["d"], lab["lonely"]}) == 3
    assert sorted(tree["order"]) == sorted(sig) and len(tree["links"]) == len(sig) - 1 and all(len(row) == 4 for row in tree["links"])
    place = {n: i for i, n in enumerate(tree["order"])}
    assert max(place[n] for n in ("a", "twice_a", "minus_a")) - min(place[n] for n in ("a", "twice_a", "minus_a")) == 2
    assert 2.5 < tree["effective_number"] < 4.0
    one = C.cluster(pd.DataFrame([[1.0]], index=["x"], columns=["x"]))
    assert one["order"] == ["x"] and one["labels"] == {"x": 1} and one["links"] == []


def test_heatmap_is_in_the_given_order_and_holds_unknowns_as_none():
    corr = pd.DataFrame([[1.0, 0.5, np.nan], [0.5, 1.0, -0.25], [np.nan, -0.25, 1.0]], index=list("abc"), columns=list("abc"))
    h = C.heatmap(corr, ["c", "a", "b"])
    assert h["names"] == ["c", "a", "b"] and h["values"][0] == [1.0, None, -0.25] and h["values"][1] == [None, 1.0, 0.5]


def test_the_alpha_report_groups_what_is_alike_and_flags_the_redundant(world):
    ctx, signals = world
    plain = C.analyse(ctx, signals)
    twins = C.analyse(ctx, {**signals, "fcf_yield_twin": signals["fcf_yield"] * 3.0 + 1.0})
    where = {m: c["id"] for c in twins["clusters"] for m in c["members"]}
    assert where["fcf_yield"] == where["fcf_yield_twin"] == where["earnings_yield"]
    assert [(r["a"], r["b"]) for r in twins["redundant"]][0] in (("fcf_yield", "fcf_yield_twin"), ("fcf_yield_twin", "fcf_yield")) and twins["redundant"][0]["correlation"] == pytest.approx(1.0)
    assert not plain["redundant"] and twins["effective_number"] < plain["effective_number"] + 0.5
    assert 1.0 <= plain["effective_number"] <= len(NAMES) and sorted(m for c in plain["clusters"] for m in c["members"]) == sorted(NAMES)
    assert plain["heatmap"]["names"] == plain["order"] and "independent ideas" in plain["reading"]
    with pytest.raises(FundamentalsError, match="at least two"):
        C.analyse(ctx, {"only": signals["fcf_yield"]})


def test_standardise_scores_each_date_among_the_investable_companies(world):
    ctx, signals = world
    std = C.standardise({"s": signals["fcf_yield"]}, ctx)["s"]
    rows = std.dropna(how="all")
    full = rows[rows.notna().sum(axis=1) >= 30]
    assert len(full) > 50 and np.allclose(full.mean(axis=1), 0.0, atol=1e-9) and np.allclose(full.std(axis=1, ddof=1), 1.0, atol=1e-6)
    assert std.where(~ctx.investable()).isna().all().all()


# ------------------------------------------------------------------------------------------------------------------------ when a return is known
def test_the_settled_lag_is_the_least_number_of_periods_after_which_every_forward_window_has_ended(world):
    ctx, _ = world
    n = len(ctx.pos)
    for horizon in ("1m", "3m", "6m", "12m"):
        h, lag = horizon_days(horizon), ctx.settled_lag(horizon)
        ends = ctx.pos + ctx.entry_lag + h
        assert all(ctx.pos[s + lag] >= ends[s] for s in range(n - lag)), horizon             # at that lag every window is over
        if lag > 1:
            assert any(ctx.pos[s + lag - 1] < ends[s] for s in range(n - lag + 1)), horizon   # and one period less is not enough somewhere
    assert ctx.settled_lag("1m") >= 2 > 1 and ctx.settled_lag("12m") > ctx.settled_lag("3m") > ctx.settled_lag("1m")


# ------------------------------------------------------------------------------------------------------------------------ the rules
def test_equal_weights_are_equal_and_every_row_sums_to_one(world):
    ctx, signals = world
    r = C.combine(ctx, signals, "equal")
    assert np.allclose(r.weights, 1.0 / len(NAMES)) and np.allclose(r.weights.sum(axis=1), 1.0)
    std = C.standardise(signals, ctx)
    manual = cross_sectional_zscore(sum(std[n].fillna(0.0) for n in NAMES).where(sum(std[n].notna().astype(int) for n in NAMES) > 0))      # a missing alpha is no view (zero), never a bad value
    seen = r.signal.notna()
    assert seen.any().any() and (seen == manual.notna()).all().all() and np.allclose(r.signal[seen], manual[seen], atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("method", ["ic", "icir", "diversified"])
def test_weights_on_a_date_use_only_returns_that_had_ended_by_then(world, method):
    ctx, signals = world
    lag, k = ctx.settled_lag("1m"), 70
    a = C.combine(ctx, signals, method).weights
    b = C.combine(future_changed(ctx, k), signals, method).weights
    early = a.index[: k + lag]                                           # the rows whose known ICs all lie before row k
    pd.testing.assert_frame_equal(a.loc[early], b.loc[early])
    assert not np.allclose(a.iloc[k + lag + 4:].to_numpy(), b.iloc[k + lag + 4:].to_numpy())      # the check has teeth: later weights do move
    assert (a.iloc[:12] == 1.0 / len(NAMES)).all().all()                                          # until enough ICs are known, equal weights
    assert np.allclose(a.sum(axis=1), 1.0) and (a >= -1e-12).all().all()


def test_learned_weights_go_to_the_planted_alphas_and_beat_a_plain_average_after_the_warm_up(world):
    ctx, signals = world
    fwd = ctx.forward("1m")
    equal, icir = C.combine(ctx, signals, "equal"), C.combine(ctx, signals, "icir")
    later = icir.weights.index[30:]
    mean_w = icir.weights.loc[later].mean()
    planted, noise = mean_w[["d_roic", "fcf_yield", "earnings_yield"]].mean(), mean_w[["asset_growth", "roic", "gross_profitability", "mom_12_1"]].mean()
    assert planted > 2 * noise
    ic_equal, ic_icir = rank_ic(equal.signal.loc[later], fwd.loc[later], 20).mean(), rank_ic(icir.signal.loc[later], fwd.loc[later], 20).mean()
    assert ic_icir > ic_equal + 0.01


def test_the_cluster_rule_gives_each_idea_the_same_total_weight(world):
    ctx, signals = world
    crowded = {**signals, "fcf_copy_1": signals["fcf_yield"] * 2.0, "fcf_copy_2": signals["fcf_yield"] + 5.0}
    r = C.combine(ctx, crowded, "cluster")
    tree = C.cluster(C.correlation(C.standardise(crowded, ctx)), 0.5)["labels"]
    row = r.weights.iloc[-1]
    totals = row.groupby(pd.Series(tree)).sum()
    assert np.allclose(totals, 1.0 / len(totals)) and np.allclose(r.weights.sum(axis=1), 1.0)
    assert row["fcf_copy_1"] == pytest.approx(row["fcf_yield"]) and row["fcf_yield"] < row["d_roic"]          # three copies of one idea share the weight one lone alpha gets
    assert "clusters of alphas" in r.notes[0]
    plain = C.combine(ctx, crowded, "equal").weights.iloc[-1]
    assert plain[["fcf_yield", "fcf_copy_1", "fcf_copy_2"]].sum() > row[["fcf_yield", "fcf_copy_1", "fcf_copy_2"]].sum()


def test_the_rules_say_how_they_weigh_and_refuse_nonsense(world):
    ctx, signals = world
    assert "mean IC" in C.combine(ctx, signals, "ic").notes[0] and "information-ratio" in C.combine(ctx, signals, "diversified").notes[0]
    with pytest.raises(FundamentalsError, match="method must be one of"):
        C.combine(ctx, signals, "magic")
    with pytest.raises(FundamentalsError, match="at least two"):
        C.combine(ctx, {"one": signals["fcf_yield"]}, "equal")


# ------------------------------------------------------------------------------------------------------------------------ machine learning
@pytest.mark.parametrize("model,carriers", [("ridge", {"d_roic"}), ("lightgbm", {"fcf_yield", "earnings_yield"})])
def test_models_learn_the_planted_alphas_out_of_sample(world, model, carriers):
    ctx, signals = world
    r = C.ml_combine(ctx, signals, model)
    fwd = ctx.forward("1m")
    ic = rank_ic(r.signal.dropna(how="all"), fwd.reindex(r.signal.dropna(how="all").index), 20)
    assert len(ic) > 50 and ic.mean() > 0.02 and ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)) > 2.0
    perm = r.importance["permutation"]
    assert max(perm, key=perm.get) in carriers
    assert r.signal.shape == (len(ctx.dates), ctx.n) and r.signal.iloc[:ctx.settled_lag("1m") + 36].isna().all().all()


@pytest.mark.parametrize("model", ["ridge", "lightgbm"])
def test_a_prediction_never_rests_on_a_return_that_was_still_unfolding(world, model):
    ctx, signals = world
    purge = ctx.settled_lag("1m") + 1
    k = 80
    a = C.ml_combine(ctx, signals, model, permutation_repeats=0).signal
    b = C.ml_combine(future_changed(ctx, k), signals, model, permutation_repeats=0).signal
    safe = a.index[: k + purge + 1]
    pd.testing.assert_frame_equal(a.loc[safe], b.loc[safe])
    assert not np.allclose(a.iloc[k + purge + 4:].to_numpy(), b.iloc[k + purge + 4:].to_numpy(), equal_nan=True)


def test_every_fold_trains_on_dates_far_enough_before_it_and_a_rolling_window_is_short(world):
    ctx, signals = world
    where = {str(d.date()): i for i, d in enumerate(ctx.dates)}
    purge = ctx.settled_lag("1m") + 1
    expanding = C.ml_combine(ctx, signals, "ridge", permutation_repeats=0)
    for f in expanding.folds:
        assert where[f["predict_from"]] - where[f["train_to"]] >= purge + 1
        assert where[f["train_from"]] == 0
    assert [f["predict_from"] for f in expanding.folds] == sorted(f["predict_from"] for f in expanding.folds)
    assert where[expanding.folds[0]["predict_from"]] >= 36 + purge
    rolling = C.ml_combine(ctx, signals, "ridge", window=24, permutation_repeats=0)
    assert all(where[f["train_to"]] - where[f["train_from"]] + 1 <= 24 for f in rolling.folds) and where[rolling.folds[-1]["train_from"]] > 24
    assert "purge" in expanding.notes[0]


def test_tuning_chooses_from_the_grid_on_a_validation_slice_inside_each_training_window(world):
    ctx, signals = world
    r = C.ml_combine(ctx, signals, "ridge", tune=True, permutation_repeats=0)
    assert {f["params"]["alpha"] for f in r.folds} <= {g["alpha"] for g in C.GRIDS["ridge"]} and "grid of settings" in r.notes[0]
    plain = C.ml_combine(ctx, signals, "ridge", permutation_repeats=0, params={"alpha": 77})
    assert {f["params"]["alpha"] for f in plain.folds} == {77.0}


@pytest.mark.parametrize("model", C.MODELS)
def test_every_model_runs_and_reports_what_it_relied_on(world, model):
    ctx, signals = world
    small = {} if model in ("ridge", "lasso", "elastic_net") else {"n_estimators": 12}
    r = C.ml_combine(ctx, signals, model, min_train=60, retrain_every=24, permutation_repeats=1, params=small)
    imp = r.importance
    assert imp["alphas"] == NAMES and set(imp["permutation"]) == set(NAMES) and len(r.folds) >= 2
    assert np.isfinite(r.signal.to_numpy(float)).any()
    if model in ("ridge", "lasso", "elastic_net"):
        assert set(imp["coefficients"]) == set(NAMES) and all(v >= 0 for v in imp["coefficients"].values()) and "built_in" not in imp
    else:
        assert sum(imp["built_in"].values()) == pytest.approx(1.0) and "coefficients" not in imp and imp["built_in_kind"]
    if model in ("lightgbm", "xgboost"):
        assert set(imp["shap"]) == set(NAMES) and all(v >= 0 for v in imp["shap"].values())
    else:
        assert imp["shap"]["available"] is False


def test_the_model_is_compared_with_plain_rules_over_the_same_dates(world):
    ctx, signals = world
    r = C.ml_combine(ctx, signals, "lightgbm", permutation_repeats=0)
    cmp = C.compare_with_rules(ctx, r, signals)
    assert set(cmp["rules"]) == {"equal", "icir"} and cmp["dates"] == cmp["model"]["n_dates"] > 40
    assert cmp["model"]["mean_ic"] > cmp["rules"]["equal"]["mean_ic"] and cmp["rules"]["equal"]["model_minus_rule_t"] > 1.96
    assert isinstance(cmp["model_beats_rules"], bool) and cmp["verdict"].startswith("The model " + ("beats" if cmp["model_beats_rules"] else "does not"))
    assert cmp["best_single"]["alpha"] in NAMES and "hindsight" not in cmp["best_single"]["alpha"] and "same dates" in cmp["best_single"]["note"]


def test_a_model_fed_noise_is_not_credited_with_beating_the_rules(world):
    ctx, _ = world
    rng = np.random.default_rng(11)
    noise = {f"noise_{i}": pd.DataFrame(rng.normal(size=(len(ctx.dates), ctx.n)), index=ctx.dates, columns=ctx.ids) for i in range(5)}
    r = C.ml_combine(ctx, noise, "ridge", permutation_repeats=0)
    cmp = C.compare_with_rules(ctx, r, noise)
    assert not cmp["model_beats_rules"] and cmp["verdict"].startswith("The model does not clearly beat")
    assert abs(cmp["model"]["mean_ic"]) < 0.03


def test_the_model_settings_and_requests_are_checked(world, monkeypatch):
    ctx, signals = world
    with pytest.raises(FundamentalsError, match="model must be one of"):
        C.ml_combine(ctx, signals, "neural_net")
    with pytest.raises(FundamentalsError, match="unknown model setting"):
        C.ml_combine(ctx, signals, "ridge", params={"eval": "x"})
    with pytest.raises(FundamentalsError, match="must be between"):
        C.ml_combine(ctx, signals, "random_forest", params={"max_depth": 500})
    with pytest.raises(FundamentalsError, match="must be a number"):
        C.ml_combine(ctx, signals, "ridge", params={"alpha": "lots"})
    with pytest.raises(FundamentalsError, match="at least two"):
        C.ml_combine(ctx, {"one": signals["fcf_yield"]}, "ridge")
    with pytest.raises(FundamentalsError, match="not enough history"):
        C.ml_combine(ctx, signals, "ridge", min_train=500)
    with pytest.raises(FundamentalsError, match="at least 1"):
        C.ml_combine(ctx, signals, "ridge", retrain_every=0)
    real = C.importlib.import_module

    def missing(name, *a, **k):
        if name == "lightgbm":
            raise ImportError("no such module")
        return real(name, *a, **k)
    monkeypatch.setattr(C.importlib, "import_module", missing)
    with pytest.raises(FundamentalsError, match="pip install lightgbm"):
        C.ml_combine(ctx, signals, "lightgbm")
