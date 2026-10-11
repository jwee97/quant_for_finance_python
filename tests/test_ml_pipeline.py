"""The machine-learning pipeline: positions from predictions, walk-forward training without look-ahead, evaluation, explanation, the registry, and a whole experiment from data to report."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as ds, evaluation as ev, experiments as xe, explainability as xp, models as md, registry as rg, signals as sg, training as tr
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]


def _cfg(**over):
    base = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 500, "refit_every": 63}}
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) and k not in ("model",) else v
    return mc.experiment_config(base, defaults=mc.DEFAULTS)


@pytest.fixture(scope="module")
def signal_bundle():
    return make_bundle(12, 1900, seed=11, alpha=1.0)


@pytest.fixture(scope="module")
def noise_bundle():
    return make_bundle(12, 1900, seed=12, alpha=0.0)


@pytest.fixture(scope="module")
def dataset(signal_bundle):
    return ds.build_dataset(signal_bundle, _cfg()["dataset"], "pipeline")


# --------------------------------------------------------------------------------------------------------------------- signals
def _scores(seed=0, n=40, k=10):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, k)), index=pd.bdate_range("2021-01-01", periods=n), columns=[f"S{i}" for i in range(k)])


def test_top_n_long_short_is_dollar_neutral_and_holds_the_extremes():
    s = _scores()
    w = sg.top_n_long_short(s, 3, gross=1.0, net=0.0)
    assert np.allclose(w.sum(axis=1), 0.0) and np.allclose(w.abs().sum(axis=1), 1.0)
    assert ((w > 0).sum(axis=1) == 3).all() and ((w < 0).sum(axis=1) == 3).all()
    row = s.iloc[5]
    assert set(w.iloc[5][w.iloc[5] > 0].index) == set(row.nlargest(3).index) and set(w.iloc[5][w.iloc[5] < 0].index) == set(row.nsmallest(3).index)
    biased = sg.top_n_long_short(s, 3, gross=1.5, net=0.5)
    assert np.allclose(biased.sum(axis=1), 0.5) and np.allclose(biased.abs().sum(axis=1), 1.5)
    s.iloc[3, :8] = np.nan
    assert ((sg.top_n_long_short(s, 3).iloc[3] > 0).sum() == 1) and sg.top_n_long_short(s, 3).iloc[3].abs().sum() == pytest.approx(1.0)     # two valid scores: one long, one short
    s.iloc[4, :9] = np.nan
    assert sg.top_n_long_short(s, 3).iloc[4].abs().sum() == 0.0


def test_the_other_methods_have_their_defining_properties():
    s = _scores(1)
    lo = sg.long_only_top_quantile(s, 0.3)
    assert (lo >= 0).all().all() and np.allclose(lo.sum(axis=1), 1.0) and ((lo > 0).sum(axis=1) == 3).all()
    prop = sg.proportional(s)
    assert np.allclose(prop.sum(axis=1), 0.0) and np.allclose(prop.abs().sum(axis=1), 1.0) and (np.sign(prop) == np.sign(s.sub(s.mean(axis=1), axis=0))).all().all()
    vol = pd.DataFrame(np.tile(np.linspace(0.1, 0.4, 10), (len(s), 1)), index=s.index, columns=s.columns)
    vs = sg.vol_scaled(s, vol)
    assert np.allclose(vs.abs().sum(axis=1), 1.0) and (vs.abs().iloc[:, 0] / prop.abs().iloc[:, 0]).mean() > (vs.abs().iloc[:, -1] / prop.abs().iloc[:, -1]).mean()      # the low-volatility name gets more
    rk = sg.rank_normalised(s)
    assert np.allclose(rk.sum(axis=1), 0.0, atol=1e-12) and (rk.iloc[3].idxmax() == s.iloc[3].idxmax())
    th = sg.threshold(s, 1.0)
    assert ((th > 0) == (s > 1.0)).all().all() and ((th < 0) == (s < -1.0)).all().all()
    on = pd.Series([True, False] * 20, index=s.index)
    rf = sg.regime_filter(sg.top_n_long_short(s, 2), on)
    assert rf[~on].abs().sum().sum() == 0 and rf[on].abs().sum().sum() > 0


def test_controls_cap_positions_hold_groups_down_and_average_the_books():
    s = _scores(2)
    w = sg.top_n_long_short(s, 2)
    capped = sg.cap_weights(sg.top_n_long_short(s, 5), 0.12)
    assert capped.abs().max().max() <= 0.12 + 1e-9 and np.allclose(capped.abs().sum(axis=1), 1.0, atol=1e-9) and np.allclose(capped.sum(axis=1), 0.0, atol=1e-9)
    groups = {c: ("a" if i < 5 else "b") for i, c in enumerate(s.columns)}
    gc = sg.group_caps(sg.proportional(s), groups, 0.3)
    assert all(gc[[c for c in s.columns if groups[c] == g]].abs().sum(axis=1).max() <= 0.3 + 1e-9 for g in ("a", "b"))
    hp = sg.holding_period(w, 5)
    assert np.allclose(hp.iloc[10], w.iloc[6:11].mean()) and sg.holding_period(w, 1) is w
    assert sg.extra_lag(w, 2).iloc[:2].abs().sum().sum() == 0 and np.allclose(sg.extra_lag(w, 2).iloc[5], w.iloc[3])
    built = sg.build_weights(s, {"method": "top_n_long_short", "n": 2, "max_weight": 0.3, "holding_period": 3}, investable=pd.DataFrame(True, index=s.index, columns=s.columns))
    assert built.abs().max().max() <= 0.3 + 1e-9
    with pytest.raises(ValueError, match="unknown signal method"):
        sg.build_weights(s, {"method": "magic"})
    with pytest.raises(ValueError, match="volatilities"):
        sg.build_weights(s, {"method": "vol_scaled"})
    with pytest.raises(ValueError, match="regime"):
        sg.build_weights(s, {"method": "regime_filtered"})
    with pytest.raises(ValueError, match="confidences"):
        sg.build_weights(s, {"method": "confidence_weighted"})


# --------------------------------------------------------------------------------------------------------------------- models
def test_every_model_kind_fits_predicts_and_reports_importance(dataset):
    rows = dataset.rows(slice(300, 700))
    X, y = dataset.X.iloc[np.flatnonzero(rows)], dataset.Y["fwd_ret_5"].iloc[np.flatnonzero(rows)]
    ok = y.notna().to_numpy()
    X, y = X[ok], y[ok]
    dates = dataset.positions()[np.flatnonzero(rows)][ok]
    for info in md.model_catalog():
        if not info["available"] or "regression" not in info["tasks"] or info["kind"] == "regime_ridge":
            continue
        m = md.make_model(info["kind"], None, "regression", 0).fit(X, y, dates)
        p = m.predict(X)
        assert p.shape == (len(X),) and np.isfinite(p).all(), info["kind"]
        assert len(m.importance()) == len(m.features_) and m.n_train_ == len(X)
        assert md.Model.from_bytes(m.to_bytes()).predict(X.iloc[:20]).tolist() == pytest.approx(p[:20].tolist())
    ridge = md.make_model("ridge", {"alpha": 1.0}, "regression").fit(X, y, dates)
    assert np.corrcoef(ridge.predict(X), y)[0, 1] > 0.02                                            # the planted signal is in the features
    with pytest.raises(ValueError, match="no parameter"):
        md.make_model("ridge", {"depth": 3})
    with pytest.raises(ValueError, match="unknown model"):
        md.make_model("oracle")
    with pytest.raises(ValueError, match="does not do"):
        md.make_model("kalman", None, "classification")
    assert md.search_space("ridge") and md.default_refit("hist_gb") == 63 and md.default_refit("ridge") == 21
    with pytest.raises(ValueError):
        md.search_space("oracle")


def test_classification_models_return_probabilities_centred_on_one_half(dataset):
    idx = np.flatnonzero(dataset.rows(slice(300, 800)))
    X, y = dataset.X.iloc[idx], dataset.Y["direction_5"].iloc[idx] if "direction_5" in dataset.Y else None
    cfg = _cfg(dataset={"target": "direction_5", "horizon": 5})
    d2 = ds.build_dataset(make_bundle(10, 1100, seed=2, alpha=1.0), cfg["dataset"], "cls")
    rows = np.flatnonzero(d2.rows(slice(300, 800)))
    X, y = d2.X.iloc[rows], d2.Y["direction_5"].iloc[rows]
    keep = y.notna().to_numpy()
    for kind in ("ridge", "logistic", "random_forest", "hist_gb", "online_logistic"):
        m = md.make_model(kind, {"n_estimators": 30} if kind == "random_forest" else None, "classification").fit(X[keep], y[keep], d2.positions()[rows][keep])
        p = m.predict_proba(X[keep])
        assert ((p >= 0) & (p <= 1)).all() and m.predict(X[keep]).tolist() == pytest.approx((p - 0.5).tolist()), kind
    with pytest.raises(ValueError, match="single class"):
        md.make_model("ridge", None, "classification").fit(X[keep], pd.Series(1, index=X[keep].index), None)


def test_the_online_and_kalman_models_track_a_changing_coefficient():
    rng = np.random.default_rng(0)
    n, k = 600, 8
    dates = np.repeat(np.arange(n), k)
    x = rng.normal(size=(n * k, 2))
    beta = np.where(dates < 300, 1.0, -1.0)                                                  # the relationship flips half way
    y = beta * x[:, 0] + 0.5 * rng.normal(size=n * k)
    X = pd.DataFrame(x, columns=["a", "b"])
    for kind, params in (("online_ridge", {"forgetting": 0.98, "alpha": 1.0}), ("kalman", {"q": 1e-3, "max_features": 2})):
        m = md.make_model(kind, params, "regression").fit(X, pd.Series(y), dates)
        assert m.importance()["a"] < -0.2, kind                                                  # it has moved to the new regime
    slow = md.make_model("online_ridge", {"forgetting": 1.0, "alpha": 1.0}, "regression").fit(X, pd.Series(y), dates)
    assert abs(slow.importance()["a"]) < 0.2                                                  # without forgetting the two halves cancel
    upd = md.make_model("online_ridge", {"forgetting": 0.98}, "regression").fit(X.iloc[: 300 * k], pd.Series(y[: 300 * k]), dates[: 300 * k])
    before = upd.importance()["a"]
    upd.partial_fit(X.iloc[300 * k: 310 * k], pd.Series(y[300 * k: 310 * k]))
    assert upd.importance()["a"] < before


# --------------------------------------------------------------------------------------------------------------------- training
def test_walk_forward_predictions_never_see_a_future_label(signal_bundle):
    cfg = _cfg()
    d = ds.build_dataset(signal_bundle, cfg["dataset"], "wf")
    base = tr.walk_forward_train(d, {"kind": "ridge", "params": {"alpha": 5.0}}, cfg["split"], 0)
    cutoff = base.plan.bounds["test"][0] + 40                                                   # a date inside the test segment
    cut_date = d.dates[tr.labelled_dates(d)[cutoff]]
    # replace every price AFTER the cutoff with noise: labels (and features) after it change, so a prediction made before it must not
    noisy = ds.build_dataset(signal_bundle.perturbed_after(cut_date, 3.0, 1), cfg["dataset"], "wf-noise")
    again = tr.walk_forward_train(noisy, {"kind": "ridge", "params": {"alpha": 5.0}}, cfg["split"], 0)
    a, b = base.predictions, again.predictions
    common = a.index.intersection(b.index)
    before = common[common.get_level_values(0) <= cut_date - pd.Timedelta(days=12)]            # the last training window ends lag + horizon days earlier than its first prediction
    assert len(before) > 1000 and np.allclose(a.loc[before].to_numpy(), b.loc[before].to_numpy(), rtol=1e-5, atol=1e-7)
    after = common[common.get_level_values(0) > cut_date + pd.Timedelta(days=40)]
    assert not np.allclose(a.loc[after].to_numpy(), b.loc[after].to_numpy())


def test_walk_forward_fits_only_complete_labels_and_covers_the_reserved_segments(dataset):
    cfg = _cfg()
    out = tr.walk_forward_train(dataset, {"kind": "ridge", "params": {}}, cfg["split"], 0)
    purge = dataset.lag + dataset.horizon
    for f in out.folds:
        assert pd.Timestamp(f["train_end"]) < pd.Timestamp(f["test_start"]) and not f.get("skipped")
        gap = int(((dataset.dates > pd.Timestamp(f["train_end"])) & (dataset.dates < pd.Timestamp(f["test_start"]))).sum())
        assert gap >= purge - 1
    first_pred = dataset.dates[tr.labelled_dates(dataset)[out.plan.bounds["validation"][0]]]
    assert out.predictions.index.get_level_values(0).min() == first_pred
    assert out.insample.index.get_level_values(0).max() < first_pred and len(out.insample) > 500
    s = out.scores_wide(dataset)
    assert s.shape == (len(dataset.dates), len(dataset.assets)) and s.loc[: first_pred - pd.Timedelta(days=1)].isna().all().all()
    assert set(out.importance.columns) == set(out.features) and len(out.models) == len(out.folds) == len(out.fold_rows)
    final = tr.fit_final(dataset, {"kind": "ridge", "params": {}})
    live = tr.predict_latest(final, dataset)
    assert list(live.index) == dataset.assets and live.notna().all()
    with pytest.raises(ValueError, match="warm-up"):
        tr.walk_forward_train(dataset, {"kind": "ridge", "params": {}}, {**cfg["split"], "min_train_days": 5000}, 0)


# --------------------------------------------------------------------------------------------------------------------- evaluation
def test_rank_ic_matches_scipy_and_the_summary_scales_the_t_statistic():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=120)
    a = pd.DataFrame(rng.normal(size=(120, 9)), index=idx)
    b = a * 0.3 + pd.DataFrame(rng.normal(size=(120, 9)), index=idx)
    ic = ev.rank_ic_wide(a, b)
    assert ic.iloc[7] == pytest.approx(spearmanr(a.iloc[7], b.iloc[7]).statistic)
    b.iloc[5, :7] = np.nan                                                                       # two assets left on that date
    assert np.isnan(ev.rank_ic_wide(a, b, min_assets=3).iloc[5]) and abs(ev.rank_ic_wide(a, b, min_assets=2).iloc[5]) == pytest.approx(1.0)
    s1, s5 = ev.ic_summary(ic, 1), ev.ic_summary(ic, 5)
    assert s1["t"] == pytest.approx(s5["t"] * np.sqrt(5)) and s1["mean"] > 0.1 and s1["hit_rate"] > 0.7 and ev.ic_summary(ic.iloc[:3])["n"] == 3


def test_ic_decay_falls_with_horizon_for_a_short_lived_signal():
    rng = np.random.default_rng(4)
    n, k = 800, 12
    idx = pd.bdate_range("2019-01-01", periods=n)
    shock = pd.DataFrame(rng.normal(size=(n, k)), index=idx)
    ret = 0.004 * shock.shift(2).fillna(0.0) + 0.01 * pd.DataFrame(rng.normal(size=(n, k)), index=idx)           # a decision at t is traded at t+1 and earns from t+2: the shock moves exactly that return, and no other
    prices = 100.0 * (1.0 + ret).cumprod()
    decay = {r["horizon"]: r["mean_ic"] for r in ev.ic_decay(shock, prices, lag=1, horizons=(1, 2, 5, 21))}
    assert decay[1] > 0.2 and decay[1] > decay[2] > decay[5] > decay[21] > 0.0 and decay[21] < decay[1] / 3          # a one-day edge is diluted by about sqrt(h) in an h-day return


def test_prediction_metrics_and_the_quantile_spread():
    rng = np.random.default_rng(5)
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=60), list("abcdefgh")])
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    good = y * 0.5 + pd.Series(rng.normal(size=len(idx)), index=idx) * 0.5
    m = ev.regression_metrics(good, y)
    assert m["corr"] > 0.5 and m["sign_accuracy"] > 0.65 and m["r2_oos_zero"] > 0 and ev.regression_metrics(-good, y)["r2_oos_zero"] < 0
    prob = 1.0 / (1.0 + np.exp(-3.0 * good))
    cm = ev.classification_metrics(prob, (y > 0).astype(float))
    assert cm["auc"] > 0.75 and cm["brier"] < cm["brier_base_rate"] and 0.4 < cm["base_rate"] < 0.6
    wide_s, wide_y = good.unstack(), y.unstack()
    q = ev.quantile_spread(wide_s, wide_y, 4)
    assert q["spread"] > 0 and q["monotonicity"] > 0.8 and len(q["buckets"]) == 4


def test_the_overfitting_report_scores_generalisation_and_names_the_problem():
    seg = lambda s, n=400: {"sharpe": s, "n_days": n}                                          # noqa: E731
    good = ev.overfit_report({"train": seg(1.4), "validation": seg(1.1), "test": seg(1.0), "forward": seg(0.9)}, n_trials=1)
    assert good["generalization_score"] > 0.6 and not any("falls to" in w for w in good["warnings"])
    bad = ev.overfit_report({"train": seg(3.0), "validation": seg(0.6), "test": seg(-0.4), "forward": seg(-0.2)}, n_trials=40, ic_by_fold=[-0.1, 0.1, -0.2, -0.1], gross_sharpe=1.2, net_sharpe=0.4)
    text = " ".join(bad["warnings"])
    assert bad["generalization_score"] == 0.0 and "In-sample Sharpe" in text and "negative in the test" in text and "40 configurations" in text and "unstable" in text and "Costs take" in text
    short = ev.overfit_report({"validation": seg(1.0, 60), "test": seg(1.0, 60)}, n_trials=1)
    assert any("only 60 days" in w for w in short["warnings"])
    returns = pd.Series(np.random.default_rng(0).normal(0.0006, 0.01, 400))
    assert 0.0 < ev.overfit_report({"validation": seg(1.0), "test": seg(1.0)}, 1, test_forward_returns=returns)["psr_test_forward"] <= 1.0
    assert ev.sharpe_of(pd.Series([0.01, 0.0, 0.01, 0.0] * 20)) > 5 and np.isnan(ev.sharpe_of(pd.Series([0.01] * 30)))


# --------------------------------------------------------------------------------------------------------------------- explanation
def test_shapley_values_add_up_exactly_and_recover_a_linear_model():
    rng = np.random.default_rng(6)
    n = 800
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(2.0 * X["a"] - 1.0 * X["b"] + 0.0 * X["c"] + 0.1 * rng.normal(size=n))
    model = md.make_model("ridge", {"alpha": 0.1}, "regression").fit(X, y, None)
    x = X.iloc[[10]]
    res = xp.shapley_sampling(model, x, X.iloc[100:400], n_perm=60, seed=1)
    assert res["values"].sum() == pytest.approx(res["prediction"] - res["base"], abs=1e-9)
    coef = model.importance()
    bg_mean = X.iloc[100:400].mean()
    # an additive model: contributions are the coefficient times the distance from the background (in raw units, scaled by the target's standard deviation)
    scale = model.y_std_ / X.std()
    expected = coef * scale * (x.iloc[0] - bg_mean)
    assert np.allclose(res["values"].to_numpy(), expected.to_numpy(), atol=0.35 * np.abs(expected).max())
    assert abs(res["values"]["c"]) < 0.3 * abs(res["values"]["a"]) + 0.05
    boosted = md.make_model("hist_gb", {"max_iter": 40}, "regression").fit(X, y, None)
    r2 = xp.shapley_sampling(boosted, x, X.iloc[100:300], n_perm=30, seed=2)
    assert r2["values"].sum() == pytest.approx(r2["prediction"] - r2["base"], abs=1e-9)


def test_importance_tables_and_permutation_importance_find_the_planted_features(dataset):
    cfg = _cfg()
    out = tr.walk_forward_train(dataset, {"kind": "ridge", "params": {"alpha": 5.0}}, cfg["split"], 0)
    table = xp.importance_table(out.importance)
    assert table["share"].sum() == pytest.approx(1.0) and table.index[0] in table.index and (table["sign_agreement"].dropna().between(0, 1)).all()
    assert -1.0 <= xp.rank_stability(out.importance) <= 1.0
    perm = xp.permutation_importance(out, dataset, max_folds=4, n_repeats=1, top=12)
    assert len(perm) == 12 and perm["mean_drop"].iloc[0] > 0.0 and perm["mean_drop"].iloc[0] >= perm["mean_drop"].iloc[-1]
    fic = xp.feature_ic(dataset)
    assert "ret_21" in fic.index and fic["mean_ic"].abs().max() > 0.03
    expl = xp.explain_prediction(out.models[-1], dataset, dataset.dates[1500], dataset.assets[2], n_background=60, n_perm=20)
    assert expl["contributions"] and abs(sum(c["contribution"] for c in expl["contributions"]) + expl["other"] - (expl["prediction"] - expl["base"])) < 1e-9
    with pytest.raises(ValueError, match="no features for"):
        xp.explain_prediction(out.models[-1], dataset, "2001-01-01", dataset.assets[0])


# --------------------------------------------------------------------------------------------------------------------- registry
def test_registry_records_experiments_trials_models_and_counts_looks(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    assert {p.name for p in (tmp_path / "ml").iterdir()} >= {"experiments", "models", "features", "predictions", "reports", "registry.sqlite"}
    cfg = mc.experiment_config({"name": "first"}, defaults=mc.DEFAULTS)
    assert reg.trial_count("k1") == 0
    eid = reg.new_id("first")
    reg.log_experiment(eid, cfg, mc.config_hash(cfg), "k1", 0)
    reg.log_metrics(eid, {"validation": {"sharpe": 1.2, "n_days": 400}, "oos": {"sharpe": 0.9}, "ic": {"mean": float("nan")}})
    reg.log_trial("study-a", "k1", 0, {"alpha": 3.0}, 0.4, 1.0)
    reg.log_trial("study-a", "k1", 1, {"alpha": 5.0}, float("nan"), 1.0, "bayes")
    assert reg.trial_count("k1") == 3 and reg.trial_count("other") == 0                          # one experiment and two trials, failed ones included
    rec = reg.get_experiment(eid)
    assert rec["metrics"]["validation"]["sharpe"] == 1.2 and "ic" not in rec["metrics"] and rec["model_kind"] == "ridge"
    assert [t["params"]["alpha"] for t in reg.list_trials("study-a")][::-1] == [3.0, 5.0]
    second = reg.new_id("second")
    reg.log_experiment(second, {**cfg, "name": "second"}, "h2", "k1", 1)
    reg.log_metrics(second, {"validation": {"sharpe": 2.0}})
    board = reg.leaderboard("validation.sharpe")
    assert [r["id"] for r in board] == [second, eid] and board[0]["metrics"]["validation.sharpe"] == 2.0
    mid = reg.register_model(eid, b"model-bytes", "ridge", ["a", "b"], "candidate")
    mid2 = reg.register_model(second, b"other", "ridge", ["a"], "candidate")
    reg.promote(mid, "production")
    reg.promote(mid2, "production")                                                              # only one production model per universe
    assert {m["id"]: m["stage"] for m in reg.list_models()} == {mid: "archived", mid2: "production"} and reg.load_model_bytes(mid2) == b"other"
    with pytest.raises(rg.RegistryError):
        reg.promote(mid, "launched")
    with pytest.raises(rg.RegistryError):
        reg.get_experiment("nope")
    sid = reg.log_forward_signal(eid, "2024-05-01", {"A": 0.5}, {"A": 1.0}, 0.2)
    reg.set_realised(sid, 0.01, "2024-05-08")
    assert reg.forward_signals(eid)[0]["realised"] == 0.01
    reg.delete_experiment(second)
    assert [r["id"] for r in reg.list_experiments()] == [eid]


# --------------------------------------------------------------------------------------------------------------------- whole experiments
def test_an_experiment_on_a_planted_signal_finds_it_and_on_noise_does_not(signal_bundle, noise_bundle, tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    over = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 500, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}}}
    real = xe.run_experiment(over, bundle=signal_bundle, registry=reg, explain=False)
    null = xe.run_experiment({**over, "name": "null"}, bundle=noise_bundle, registry=reg, explain=False)
    assert real.report["ic"]["all"]["t"] > 3.0 and real.report["segments"]["stats"]["test"]["sharpe"] > 0.3
    assert abs(null.report["ic"]["all"]["t"]) < 3.0 and null.report["overfit"]["generalization_score"] <= 0.5
    assert real.report["dataset_key"] == null.report["dataset_key"]                                 # the key is the dataset section of the configuration
    assert real.report["trials"]["dataset_trials"] == 1 and null.report["trials"]["dataset_trials"] == 2
    d = tmp_path / "ml" / "experiments" / real.id
    assert {p.name for p in d.iterdir()} >= {"config.yaml", "metrics.json", "predictions.parquet", "positions.parquet", "returns.parquet", "model.pkl", "report.json", "report.html"}
    saved = json.loads((d / "report.json").read_text())
    assert saved["id"] == real.id and set(saved["segments"]["stats"]) == {"train", "validation", "test", "forward"} and saved["equity"]["bounds"]["test"]
    page = (d / "report.html").read_text()
    assert page.startswith("<!doctype html>") and "<script" not in page and "Equity curve" in page
    preds = pd.read_parquet(d / "predictions.parquet")
    assert {"score", "segment", "target"} <= set(preds.columns) and preds["segment"].isin(["validation", "test", "forward"]).all()
    rec = reg.get_experiment(real.id)
    assert rec["metrics"]["test"]["sharpe"] == pytest.approx(real.report["segments"]["stats"]["test"]["sharpe"]) and {m["kind"] for m in reg.list_models()} == {"ridge"}
    assert reg.list_feature_sets() and reg.trial_count(real.report["dataset_key"]) >= 1
    # the trial count grows with every run on the same data and appears in the next report
    again = xe.run_experiment(over, bundle=signal_bundle, registry=reg, explain=False)
    assert again.report["trials"]["dataset_trials"] == 3 and again.report["config_hash"] == real.report["config_hash"]


def test_experiment_errors_are_messages_not_crashes(signal_bundle):
    over = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 500}}
    with pytest.raises(xe.ExperimentError, match="warm-up"):
        xe.run_experiment({**over, "split": {"min_train_days": 3000}}, bundle=signal_bundle)
    with pytest.raises(xe.ExperimentError, match="regime"):
        xe.run_experiment({**over, "signal": {"method": "regime_filtered"}}, bundle=signal_bundle, final_model=False, explain=False)
    with pytest.raises(xe.ExperimentError, match="unknown universe"):
        xe.run_experiment({**over, "dataset": {"universe": "mars"}})
    with pytest.raises(mc.MLConfigError):
        xe.run_experiment({"dataset": {"horizon": 0}}, bundle=signal_bundle)
