"""The parts of the machine-learning platform that sit on top of single experiments: searches that count their trials, strategy combination, meta-labelling, and forward testing."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as ds, ensembles as en, experiments as xe, monitoring as mon, optimization as opt, registry as rg, strategy_search as ss, training as tr, visualization as viz
from src.ml.datasets import _slice_bundle
from src.ops.hpo import Categorical, Float, Int, LogFloat
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
BASE = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 500, "refit_every": 63}, "model": {"kind": "ridge", "params": {}}}


@pytest.fixture(scope="module")
def bundle():
    return make_bundle(10, 1700, seed=21, alpha=1.0)


# --------------------------------------------------------------------------------------------------------------------- optimisation
def test_dotted_parameters_land_in_the_configuration():
    cfg = mc.experiment_config(BASE, defaults=mc.DEFAULTS)
    out = opt.apply_params(cfg, {"model.params.alpha": 3.0, "signal.n": 2, "dataset.horizon": 20})
    assert out["model"]["params"]["alpha"] == 3.0 and out["signal"]["n"] == 2 and out["dataset"]["horizon"] == 20 and out["dataset"]["target"] == "fwd_ret_20" and cfg["signal"]["n"] == 3
    switched = opt.apply_params({**cfg, "model": {"kind": "ridge", "params": {"alpha": 5.0}}}, {"model.kind": "hist_gb", "model.params.max_depth": 3})
    assert switched["model"] == {"kind": "hist_gb", "params": {"max_depth": 3}}                         # the old kind's alpha is gone


def test_the_validation_score_penalises_turnover_drawdown_instability_and_the_train_gap():
    report = {"segments": {"stats": {"train": {"sharpe": 2.0}, "validation": {"sharpe": 1.0, "max_drawdown": -0.2}}}, "metrics": {"oos": {"ann_turnover": 20.0}}, "overfit": {"instability": 0.5}}
    score, parts = opt.validation_score(report)
    assert score == pytest.approx(1.0 - 0.5 * 1.0 - 0.3 * 0.2 - 0.2 * 0.5 - 0.5 * 1.0) and parts["train_validation_gap"] == 1.0
    assert opt.validation_score(report, {"turnover": 0.0, "max_drawdown": 0.0, "instability": 0.0, "train_validation_gap": 0.0})[0] == pytest.approx(1.0)
    report["segments"]["stats"]["train"]["sharpe"] = 0.5
    assert opt.validation_score(report)[1]["train_validation_gap"] == 0.0                              # no penalty when out of sample is better
    assert np.isnan(opt.validation_score({"segments": {"stats": {}}, "metrics": {"oos": {}}, "overfit": {}})[0])


def test_a_search_never_predicts_the_reserved_segments_and_deflates_its_winner(bundle, tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    space = {"model.params.alpha": LogFloat(0.1, 1000.0), "signal.n": Int(1, 4), "signal.holding_period": Categorical(1, 5, 20), "signal.gross": Float(0.8, 1.2)}
    out = opt.tune(BASE, space, n_trials=8, method="random", seed=3, bundle=bundle, registry=reg, study_name="t1")
    assert out["n_trials"] == 8 and len(out["trials"]) == 8 and out["best"]["score"] == max(t["score"] for t in out["trials"] if np.isfinite(t["score"]))
    assert reg.trial_count(mc.dataset_key(mc.experiment_config(BASE, defaults=mc.DEFAULTS))) == 8
    assert out["selection"]["n_trials"] == 8 and 0.0 <= out["selection"]["deflated_sharpe_probability"] <= 1.0 and 0.0 <= out["selection"]["pbo"] <= 1.0
    # the trial experiments were truncated at the end of the validation segment: no test or forward statistic exists for them
    runner = opt.TrialRunner(BASE, bundle=bundle, study="probe")
    cfg = runner.base
    d = runner._dataset_for(cfg)
    exp = xe.run_experiment(cfg, dataset=d, save=False, explain=False, final_model=False, stop_after="validation")
    assert set(exp.report["segments"]["stats"]) == {"train", "validation"} and exp.train.predictions.index.get_level_values(0).max() <= d.dates[tr.labelled_dates(d)[exp.train.plan.bounds["validation"][1] - 1]]
    final = opt.finalize(BASE, out["best"]["params"], bundle=bundle, registry=reg, trials_extra=out["n_trials"], robustness=False)
    assert set(final.report["segments"]["stats"]) == {"train", "validation", "test", "forward"} and final.report["trials"]["dataset_trials"] >= 9
    tpe = opt.tune(BASE, {"model.params.alpha": LogFloat(0.1, 1000.0)}, n_trials=12, method="tpe", seed=1, bundle=bundle, study_name="t2")
    assert tpe["n_trials"] == 12 and tpe["best"] is not None


def test_the_current_settings_are_read_from_the_configuration_and_the_space_is_stretched_to_hold_them():
    space = ss.kind_space("ridge")
    now = opt.current_settings({**BASE, "model": {"kind": "ridge", "params": {"alpha": 30.0}}, "signal": {"n": 2, "holding_period": 10}}, space)
    assert now["model.params.alpha"] == 30.0 and now["signal.n"] == 2 and now["signal.holding_period"] == 10 and now["backtest.rebalance"] == "weekly" and now["dataset.horizon"] == 5
    assert opt.current_settings({**BASE, "model": {"kind": "ridge", "params": {}}}, space)["model.params.alpha"] == 10.0, "a setting the form leaves alone is the model's own default"
    wide = opt.include(space, {**now, "model.params.alpha": 5000.0, "signal.n": 9})
    assert wide["model.params.alpha"].high == 5000.0 and wide["model.params.alpha"].low == 0.1 and wide["signal.n"].high == 9 and 10 in wide["signal.holding_period"].choices
    assert wide["signal.method"] is space["signal.method"], "a distribution that already holds the value is left as it is"
    with pytest.raises(ValueError, match="above zero"):
        opt.include({"model.params.alpha": LogFloat(0.1, 10.0)}, {"model.params.alpha": 0.0})
    with pytest.raises(ValueError, match="not known"):
        opt.current_settings(BASE, {"model.params.nonsense": Float(0, 1)})


def test_a_tuning_that_starts_from_given_settings_runs_them_first_never_repeats_and_can_be_stopped(bundle):
    space = {"model.params.alpha": LogFloat(0.1, 1000.0), "signal.n": Int(1, 3)}
    begin = {"model.params.alpha": 30.0, "signal.n": 2}
    out = opt.tune(BASE, space, n_trials=6, method="tpe", seed=1, bundle=bundle, study_name="t3", start=begin)
    assert out["n_trials"] == 6 and out["start_number"] == 0 and out["trials"][0]["params"] == begin and out["stopped"] is None
    keys = [(round(t["params"]["model.params.alpha"], 9), t["params"]["signal.n"]) for t in out["trials"]]
    assert len(set(keys)) == 6, "no configuration is run twice"
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] >= 2

    cut = opt.tune(BASE, space, n_trials=10, method="tpe", seed=1, bundle=bundle, study_name="t4", start=begin, stop=stop)
    assert cut["stopped"] == "user" and cut["n_trials"] == 3 and cut["best"] is not None, "the trial in progress finishes, then it ends; what ran is kept and reported"
    plain = opt.tune(BASE, {"model.params.alpha": LogFloat(0.1, 1000.0)}, n_trials=4, method="random", seed=1, bundle=bundle, study_name="t5")
    assert plain["start_number"] is None and plain["n_trials"] == 4, "without a start the search is what it always was"


def test_a_failed_candidate_is_counted_not_hidden(bundle):
    runner = opt.TrialRunner({**BASE, "split": {"min_train_days": 500, "refit_every": 63}}, bundle=bundle, study="bad")
    score, net, detail = runner.evaluate({"split.min_train_days": 100000})
    assert np.isnan(score) and net is None and "warm-up" in detail["error"] and len(runner.records) == 1


def test_the_three_stage_search_counts_every_trial_and_finalises_the_best(bundle, tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    out = ss.search(BASE, bundle=bundle, registry=reg, kinds=("ridge", "lasso", "online_ridge"), n_random=6, n_bayes=3, generations=1, population=4, top_k=1, seed=2, study_name="s1")
    assert out["stages"]["random"] == 6 and out["stages"]["bayes"] == 3 and out["stages"]["genetic"] >= 1 and out["n_trials"] == sum(out["stages"].values())
    assert out["best"]["score"] >= max(r["score"] for r in out["trials"] if np.isfinite(r["score"])) - 1e-12 and set(out["kinds"]) == {"ridge", "lasso", "online_ridge"}
    assert len(out["finalists"]) == 1 and set(out["finalists"][0]["segments"]) == {"train", "validation", "test", "forward"} and out["finalists"][0]["trials"] > out["n_trials"]
    assert "not predicted" in out["note"] and out["selection"]["n_trials"] == out["n_trials"]
    assert reg.trial_count(out["dataset_key"]) == out["n_trials"] + 1                                      # every trial and the finalist: all are looks at the same sample, whatever horizon they used
    spaces = {k: ss.kind_space(k) for k in ("ridge", "lasso")}
    rng = np.random.default_rng(0)
    a = {"model.kind": "ridge", "model.params.alpha": 3.0, "signal.n": 2, "signal.holding_period": 5, "dataset.horizon": 5}
    b = {"model.kind": "lasso", "model.params.alpha": 0.1, "signal.n": 3, "signal.holding_period": 20, "dataset.horizon": 20}
    child = ss.crossover(a, b, spaces, rng)
    assert child["model.kind"] in ("ridge", "lasso") and ("model.params.alpha" in child) and child["signal.n"] in (2, 3)
    mutant = ss.mutate(a, spaces, rng, 2)
    assert mutant != a and mutant["model.kind"] == "ridge"
    with pytest.raises(ValueError, match="none of the requested"):
        ss.search(BASE, bundle=bundle, kinds=("kalman",), n_random=2, n_bayes=0, generations=0, finalize_top=False) if False else ss.search({**BASE, "dataset": {**BASE["dataset"], "target": "direction_5"}}, bundle=bundle, kinds=("kalman",), n_random=2)


# --------------------------------------------------------------------------------------------------------------------- combining strategies
@pytest.fixture(scope="module")
def streams():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2016-01-01", periods=1000)
    market = rng.normal(0.0003, 0.008, 1000)
    return pd.DataFrame({"good": 0.0006 + 0.5 * market + 0.004 * rng.standard_normal(1000), "ok": 0.0002 + 0.003 * rng.standard_normal(1000), "bad": -0.0004 + 0.01 * rng.standard_normal(1000),
                         "dup": 0.0006 + 0.5 * market + 0.004 * rng.standard_normal(1000)}, index=idx)


def test_every_combination_method_returns_valid_weights_and_causal_returns(streams):
    regimes = pd.Series(np.where(np.arange(len(streams)) % 300 < 150, "low", "high"), index=streams.index)
    for m in en.METHODS:
        c = en.combine(streams, m, regimes=regimes, min_history=150)
        w = c.weights.dropna()
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6) and (w >= -1e-9).all().all(), m
        assert c.returns.index[0] > streams.index[149] and np.isfinite(c.summary["sharpe"])
        # causal: weights at a date use returns up to that date, applied from the next day
        manual = (c.weights.shift(1) * streams.reindex(c.weights.index)).sum(axis=1, min_count=1).dropna()
        assert np.allclose(manual.to_numpy(), c.returns.to_numpy())
    eq = en.combine(streams, "equal")
    assert np.allclose(eq.weights.iloc[-1], 0.25)
    iv = en.combine(streams, "inverse_vol")
    assert iv.weights.iloc[-1]["ok"] > iv.weights.iloc[-1]["bad"]                                        # the calmer strategy gets more
    rp = en.combine(streams, "risk_parity")
    assert rp.weights.iloc[-1]["dup"] + rp.weights.iloc[-1]["good"] < 0.7                                  # two near-duplicates are one bet
    sel = en.combine(streams, "best_trailing", min_history=150)
    assert (sel.weights.dropna().max(axis=1) == 1.0).all()
    with pytest.raises(ValueError, match="unknown combination"):
        en.combine(streams, "alchemy")
    with pytest.raises(ValueError, match="regime series"):
        en.combine(streams, "regime_conditioned")
    with pytest.raises(ValueError, match="at least two"):
        en.combine(streams[["good"]], "equal")
    with pytest.raises(ValueError, match="share only"):
        en.combine(streams.iloc[:100], "equal")


def test_the_bandit_and_hedge_weights_follow_performance(streams):
    th = en.thompson_weights(streams.iloc[:800], seed=1)
    assert th.sum() == pytest.approx(1.0) and th.idxmax() in ("good", "dup") and th["bad"] < 0.05
    hw = en.hedge_weights(streams, eta=20.0)
    assert np.allclose(hw.sum(axis=1), 1.0) and hw.iloc[-1]["good"] > hw.iloc[-1]["bad"] and (hw >= 0.0099).all().all()
    bs = en.bayes_stein_means(streams.iloc[:500])
    raw = streams.iloc[:500].mean() * 252
    assert bs.std() < raw.std() and bs.mean() == pytest.approx(raw.mean(), abs=0.1)                         # shrunk toward a common value


def test_method_comparison_lists_every_method_and_maps_the_strategies(streams):
    out = en.compare_methods(streams, methods=("equal", "inverse_vol", "bandit", "regime_selector"), regimes=None, min_history=150)
    assert {r["method"] for r in out["rows"]} == {"equal", "inverse_vol", "bandit"} and "regime_selector" in out["errors"]
    assert set(out["strategy_sharpes"]) == set(streams.columns) and out["similarity"]["names"] == list(streams.columns)
    near = lambda a, b: np.hypot(out["similarity"]["x"][list(streams.columns).index(a)] - out["similarity"]["x"][list(streams.columns).index(b)],       # noqa: E731
                                 out["similarity"]["y"][list(streams.columns).index(a)] - out["similarity"]["y"][list(streams.columns).index(b)])
    assert near("good", "dup") < near("good", "bad")


def test_meta_labelling_sizes_the_primary_models_bets(bundle):
    cfg = mc.experiment_config(BASE, defaults=mc.DEFAULTS)
    d = ds.build_dataset(bundle, cfg["dataset"], "meta")
    train = tr.walk_forward_train(d, cfg["model"], cfg["split"], 0)
    meta = en.meta_label_scores(d, train, "logistic", None, min_rows=300)
    assert meta["n_models"] >= 3 and 0.0 < meta["coverage"] <= 1.0 and 0.0 <= meta["auc"] <= 1.0
    s = meta["scores"]
    prim = train.scores_wide(d)
    both = s.notna() & prim.notna()
    sized = both & (s.abs() > 0)
    assert (np.sign(s[sized]) == np.sign(prim[sized])).to_numpy()[sized.to_numpy()].all() and sized.to_numpy().mean() < both.to_numpy().mean()      # the side is the primary's; some bets are dropped
    assert s.abs().max().max() <= 1.0 + 1e-9
    first_meta = s.dropna(how="all").index[0]
    assert first_meta > prim.dropna(how="all").index[0]                                                            # the secondary model needs history of the primary's out-of-sample bets


# --------------------------------------------------------------------------------------------------------------------- monitoring
def test_psi_flags_a_shifted_distribution_and_a_stable_one():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert mon.psi(a, rng.normal(size=1000)) < 0.1 and mon.psi(a, rng.normal(1.0, 1.0, 1000)) > 0.25 and mon.psi(a, rng.normal(0.0, 2.0, 1000)) > 0.25
    assert np.isnan(mon.psi(a[:10], a[:10])) and mon.psi(np.ones(100), np.ones(50)) == 0.0
    assert mon.is_rebalance_day(pd.Timestamp("2024-05-10"), "weekly") and not mon.is_rebalance_day(pd.Timestamp("2024-05-09"), "weekly")
    assert mon.is_rebalance_day(pd.Timestamp("2024-05-31"), "monthly") and not mon.is_rebalance_day(pd.Timestamp("2024-05-30"), "monthly") and mon.is_rebalance_day(pd.Timestamp("2024-05-08"), "daily")


def test_the_forward_loop_uses_only_the_data_of_the_day_and_settles_matured_signals(bundle, tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    cut = bundle.index[1500]
    train_bundle = _slice_bundle(bundle, slice(None, cut))
    exp = xe.run_experiment({**BASE, "signal": {"holding_period": 3}}, bundle=train_bundle, registry=reg, explain=False)
    tracker = mon.ForwardTracker(reg, exp.id)
    days = bundle.index[1501:1521]
    first = None
    for d in days:
        sig = tracker.step(bundle, as_of=d)
        first = first or sig
        assert sig["date"] == str(d.date())
    # the signal for a day is the same whether or not later data exists in the bundle
    later = tracker.step(bundle, as_of=days[5])
    assert later["signal_id"] is None and later["date"] == str(days[5].date())
    sigs = reg.forward_signals(exp.id)
    assert len(sigs) == 20 and sigs[0]["signal_date"] == str(days[0].date())
    assert all(sum(abs(v) for v in s["weights"].values()) <= 1.5 for s in sigs)
    # a signal exists on a rebalance day with trades; a non-rebalance day holds what it had
    held = [s for s in sigs if not mon.is_rebalance_day(pd.Timestamp(s["signal_date"]), exp.cfg["backtest"]["rebalance"])]
    assert held and all(s["turnover"] == 0.0 for s in held[1:])
    settled = tracker.settle(bundle, as_of=bundle.index[1530])
    assert settled and all(np.isfinite(s["realised"]) for s in settled) and len(settled) == len([s for s in sigs if pd.Timestamp(s["signal_date"]) <= bundle.index[1530 - 6]])
    assert tracker.settle(bundle, as_of=bundle.index[1530]) == []                                         # already settled
    tracker.settle(bundle)
    summary = tracker.summary(bundle)
    assert summary["n_signals"] == 20 and summary["n_settled"] >= len(settled) and summary["live_ic"]["backtest_mean"] > 0 and "drift" in summary and summary["drift"]["n_alert"] >= 0
    with pytest.raises(rg.RegistryError, match="no saved model"):
        quiet = xe.run_experiment({**BASE, "name": "nomodel"}, bundle=train_bundle, registry=reg, explain=False, final_model=False)
        mon.ForwardTracker(reg, quiet.id)


def test_visual_payloads_are_small_json_ready_and_thinned():
    assert viz.thin(100, 900).tolist() == list(range(100)) and viz.thin(5000, 500)[-1] == 4999 and len(viz.thin(5000, 500)) <= 501
    h = viz.histogram(np.r_[np.random.default_rng(0).normal(size=1000), [50.0]], 20, (0.01, 0.99))
    assert len(h["counts"]) == 20 and len(h["edges"]) == 21 and h["edges"][-1] < 10 and h["n"] == 1001
    m = viz.similarity_map(pd.DataFrame(np.random.default_rng(1).normal(size=(300, 5)), columns=list("abcde")))
    assert len(m["x"]) == 5 and 0.0 <= m["explained"] <= 1.0
    out = viz.jsonable({"a": np.float32("nan"), "b": [np.int64(3), pd.Timestamp("2024-01-02")], "c": pd.DataFrame({"x": [1.0, np.inf]}), "d": np.array([1.5, np.nan])})
    assert out == {"a": None, "b": [3, "2024-01-02"], "c": [{"index": 0, "x": 1.0}, {"index": 1, "x": None}], "d": [1.5, None]}
