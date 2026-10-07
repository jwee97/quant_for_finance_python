"""Research operations: the artifact store and registry, cross-validation for dependent data, honest hyperparameter search, and profiling."""

import warnings
from math import comb

import numpy as np
import pandas as pd
import pytest

from src.framework import bundle_from_prices
from src.framework.experiments import ExperimentManager
from src.ops import ArtifactStore, Categorical, Float, Int, LogFloat, ModelRegistry, Study, cv, profiling, tune_pipeline
from src.utils.config import load_config

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------------------------------------------- store
def test_artifact_store_round_trips_every_kind_and_deduplicates(tmp_path):
    store = ArtifactStore(tmp_path / "art")
    j = store.put({"a": np.float64(1.5), "b": np.arange(3), "c": pd.Timestamp("2024-01-02")}, "json")
    assert store.get(j) == {"a": 1.5, "b": [0, 1, 2], "c": "2024-01-02T00:00:00"}
    n = store.put({"x": np.arange(5.0), "y": np.eye(2)}, "npz")
    out = store.get(n)
    assert np.array_equal(out["x"], np.arange(5.0)) and np.array_equal(out["y"], np.eye(2))
    frame = pd.DataFrame({"p": [1.0, 2.0]}, index=pd.date_range("2024", periods=2))
    f = store.put(frame, "frame")
    pd.testing.assert_frame_equal(store.get(f), frame)
    assert store.get(store.put(b"raw", "bytes")) == b"raw"
    again = store.put({"a": np.float64(1.5), "b": np.arange(3), "c": pd.Timestamp("2024-01-02")}, "json")
    assert again.sha256 == j.sha256 and len(store.list()) == 4 and store.exists(j) and not list(tmp_path.rglob("*.tmp"))
    assert store.put({"a": 2}, "json").sha256 != j.sha256 and store.get(j.sha256, "json")["a"] == 1.5
    with pytest.raises(ValueError):
        store.put({}, "pickle")
    with pytest.raises(KeyError):
        store.get("0" * 64, "json")
    with pytest.raises(ValueError):
        store.get(j.sha256)


def test_artifact_store_detects_corruption_and_collects_garbage(tmp_path):
    store = ArtifactStore(tmp_path)
    keep, drop = store.put({"k": 1}), store.put({"k": 2})
    assert store.verify() == []
    path = store._path(keep.sha256)
    path.write_bytes(b'{"k": 99}')
    assert store.verify() == [keep.sha256]
    with pytest.raises(ValueError, match="corrupted"):
        store.get(keep)
    assert store.gc({drop.sha256}) == 1 and store.list() == [drop.sha256]


def test_registry_versions_stages_lineage_and_artifacts(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.sqlite")
    v1 = reg.register("garch", params={"p": 1}, artifact={"alpha": 0.08}, metrics={"sharpe": 0.5}, data_key="d1", tags=["vol"])
    v2 = reg.register("garch", params={"p": 2}, artifact={"alpha": 0.09}, metrics={"sharpe": 0.7}, parent_version=v1)
    v3 = reg.register("garch", params={"p": 3}, artifact={"alpha": 0.1}, metrics={"sharpe": 0.6}, parent_version=v2)
    assert (v1, v2, v3) == (1, 2, 3) and reg.get("garch", 1)["stage"] == "staging" and reg.get("garch", 1)["tags"] == ["vol"]
    assert reg.resolve("garch") == 3                                                       # no production yet: the latest
    reg.promote("garch", 2, "production", "best sharpe")
    assert reg.resolve("garch") == 2 and reg.load("garch") == {"alpha": 0.09} and reg.load("garch", 1) == {"alpha": 0.08}
    reg.promote("garch", 3, "production")
    assert reg.get("garch", 2)["stage"] == "archived" and reg.get("garch", 3)["stage"] == "production"
    assert reg.resolve("garch", "production") == 3 and reg.resolve("garch", "staging") == 1
    assert [r["version"] for r in reg.lineage("garch")] == [3, 2, 1]
    hist = reg.history("garch")
    assert set(hist["stage"]) == {"staging", "production", "archived"} and "superseded by v3" in hist["note"].tolist()
    table = reg.versions("garch")
    assert table["sharpe"].tolist() == [0.5, 0.7, 0.6] and table["stage"].tolist() == ["staging", "archived", "production"] and reg.versions().shape[0] == 3
    assert reg.get("garch", 1)["git_commit"] != ""
    with pytest.raises(KeyError):
        reg.get("nope")
    with pytest.raises(KeyError):
        reg.register("garch", parent_version=99)
    with pytest.raises(KeyError):
        reg.promote("garch", 9, "production")
    with pytest.raises(ValueError):
        reg.promote("garch", 1, "bogus")
    with pytest.raises(ValueError):
        reg.register("x", stage="bogus")
    assert reg.register("direct", artifact={"a": 1}, stage="production") == 1 and reg.get("direct")["stage"] == "production"
    reg.register("no_artifact", params={"p": 0})
    with pytest.raises(ValueError, match="no artifact"):
        reg.load("no_artifact")
    reopened = ModelRegistry(tmp_path / "reg.sqlite")
    assert reopened.get("garch", 3)["stage"] == "production" and reopened.load("garch") == {"alpha": 0.1}


def test_registry_records_an_experiment_run_as_a_model_version(tmp_path):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2012", periods=900)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, (900, 6)), axis=0), index=idx, columns=list("ABCDEF"))
    bundle = bundle_from_prices(prices, name="registry-test")
    cfg = load_config()
    manager = ExperimentManager(cfg, db_path=tmp_path / "exp.db", runs_dir=tmp_path / "runs")
    run_id, _ = manager.run({"name": "tsmom-test", "models": [{"name": "tsmom"}], "allocation": {"allocator": "sleeves"}}, bundle, validate=False)
    reg = ModelRegistry(tmp_path / "reg.sqlite")
    v = reg.register_run(manager, run_id)
    rec = reg.get("tsmom-test", v)
    assert rec["params"]["models"][0]["name"] == "tsmom" and rec["metrics"]["run_id"] == run_id and "sharpe" in rec["metrics"] and rec["description"].endswith(run_id)
    assert rec["data_key"] != ""


# ----------------------------------------------------------------------------------------------------------------------- CV
def test_purged_kfold_never_leaks_labels_and_covers_the_sample():
    n, horizon, embargo = 1000, 21, 10
    seen = []
    for train, test in cv.purged_kfold(n, 5, horizon, embargo):
        assert len(set(train) & set(test)) == 0
        lo, hi = test.min(), test.max() + 1
        assert not np.any((train + horizon >= lo) & (train < lo))                         # no training label window reaches into the test block
        assert not np.any((train >= hi) & (train < hi + embargo))                         # nothing right after it either
        assert len(train) < n - len(test)
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(n))


def test_cpcv_split_counts_paths_and_purging():
    n, N, k = 1200, 6, 2
    splits = cv.cpcv_splits(n, N, k, horizon=10, embargo=5)
    assert len(splits) == comb(N, k) and cv.n_cpcv_paths(N, k) == 5
    for s in splits:
        assert len(set(s["train"]) & set(s["test"])) == 0 and len(s["test"]) == 2 * (n // N)
        for g in s["test_groups"]:
            lo, hi = g * (n // N), (g + 1) * (n // N)
            assert not np.any((s["train"] + 10 >= lo) & (s["train"] < lo)) and not np.any((s["train"] >= hi) & (s["train"] < hi + 5))
    paths = cv.cpcv_paths(splits, N)
    assert len(paths) == 5
    for path in paths:
        assert [g for _, g in path] == list(range(N))                                     # every group appears exactly once per path
        assert all(g in splits[i]["test_groups"] for i, g in path)
    used = {}
    for path in paths:
        for i, g in path:
            used.setdefault(g, []).append(i)
    assert all(len(set(v)) == 5 for v in used.values())                                   # a group's five appearances come from five different splits


def test_cpcv_evaluate_separates_signal_from_noise():
    rng = np.random.default_rng(5)
    n = 2400
    x = pd.DataFrame({"f": rng.normal(size=n)})
    y_signal = pd.Series(0.3 * x["f"] * 0.01 + rng.normal(0, 0.01, n))
    y_noise = pd.Series(rng.normal(0, 0.01, n))

    def fit_predict(Xtr, ytr, Xte):
        beta = float((Xtr["f"] * ytr).sum() / (Xtr["f"] ** 2).sum())
        return np.sign(beta * Xte["f"].to_numpy())

    good, bad = cv.cpcv_evaluate(x, y_signal, fit_predict, 6, 2, 5, 2), cv.cpcv_evaluate(x, y_noise, fit_predict, 6, 2, 5, 2)
    assert good["n_paths"] == 5 and good["n_splits"] == 15 and len(good["sharpes"]) == 5
    assert good["mean"] > 1.0 and good["prob_negative"] == 0.0 and abs(bad["mean"]) < 1.5 and bad["mean"] < good["mean"] - 1.0
    assert all(len(p) == n for p in good["path_returns"])


# ----------------------------------------------------------------------------------------------------------------------- HPO
def test_search_space_sampling_and_grids():
    rng = np.random.default_rng(0)
    p = [Int(2, 5), Float(0.0, 1.0), LogFloat(1e-3, 1.0), Categorical("a", "b")]
    for _ in range(50):
        vals = [x.sample(rng) for x in p]
        assert 2 <= vals[0] <= 5 and isinstance(vals[0], int) and 0 <= vals[1] <= 1 and 1e-3 <= vals[2] <= 1 and vals[3] in ("a", "b")
    assert p[0].grid(4) == [2, 3, 4, 5] and len(p[1].grid(5)) == 5 and p[2].grid(4)[0] == pytest.approx(1e-3) and p[3].grid(9) == ["a", "b"]
    assert p[2].to_unit(1e-3) == pytest.approx(0.0) and p[2].to_unit(1.0) == pytest.approx(1.0)


def _bowl(params):
    return -((params["x"] - 0.3) ** 2 + (params["y"] - 3) ** 2 / 10.0)


def test_search_methods_find_the_optimum_and_count_every_trial():
    space = {"x": Float(0.0, 1.0), "y": Int(0, 10)}
    grid = Study(space).optimize(_bowl, method="grid", grid_points=6)
    assert grid.n_trials == len(Int(0, 10).grid(6)) * 6
    assert abs(grid.best.params["x"] - 0.3) < 0.11 and grid.best.params["y"] in (2, 4)
    rnd = Study(space, seed=1).optimize(_bowl, 60, "random")
    assert rnd.n_trials == 60 and rnd.best.value > -0.05
    wins = 0
    for seed in range(6):
        t = Study(space, seed=seed).optimize(_bowl, 40, "tpe")
        r = Study(space, seed=seed).optimize(_bowl, 40, "random")
        wins += t.best.value >= r.best.value
    assert wins >= 4                                                                         # the model-based search is better than blind sampling at the same budget
    assert list(rnd.to_frame().columns[:2]) == ["x", "y"] and rnd.to_frame().shape[0] == 60
    with pytest.raises(ValueError):
        Study(space).optimize(_bowl, 3, "nope")
    with pytest.raises(ValueError):
        Study(space).best


def test_failed_trials_are_recorded_counted_and_skipped():
    def flaky(p):
        if p["x"] > 0.8:
            raise RuntimeError("blew up")
        return p["x"]

    s = Study({"x": Float(0.0, 1.0)}, seed=2).optimize(flaky, 40, "random")
    errors = s.to_frame()["error"].notna()
    assert errors.any() and s.n_trials == 40 and s.best.error is None and s.best.params["x"] <= 0.8


def test_successive_halving_spends_budget_on_the_survivors():
    seen = []

    def objective(p, budget):
        seen.append(budget)
        return -(p["x"] - 0.5) ** 2 + np.random.default_rng(int(budget * 1000)).normal(0, 0.02 / budget)

    s = Study({"x": Float(0.0, 1.0)}, seed=3).optimize(objective, 27, "halving", min_budget=0.1, max_budget=0.9, eta=3)
    budgets = pd.Series(seen).value_counts().sort_index()
    assert list(budgets.index) == [pytest.approx(0.1), pytest.approx(0.3), pytest.approx(0.9)] and budgets.tolist() == [27, 9, 3] and s.n_trials == 39
    assert abs(s.best.params["x"] - 0.5) < 0.25


def test_selection_report_deflates_a_winner_chosen_from_noise_and_keeps_a_real_edge():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015", periods=1500)

    def noise_objective(p):
        r = pd.Series(np.random.default_rng(int(p["seed"])).normal(0, 0.01, len(idx)), index=idx)
        return float(r.mean() / r.std() * np.sqrt(252)), r

    s = Study({"seed": Int(0, 10 ** 6)}, seed=1).optimize(noise_objective, 60, "random")
    rep = s.selection_report(n_boot=200)
    assert rep["n_trials"] == 60 and rep["best_annual_sharpe"] > 0.4 and rep["deflated_sharpe_probability"] < 0.95
    assert rep["survivors_at_5pct"] == 0 and rep["romano_wolf_p_best"] > 0.1 and 0.0 <= rep["pbo"] <= 1.0 and rep["deflated_benchmark_sharpe_annual"] > 0.5

    def edge_objective(p):
        r = pd.Series(np.random.default_rng(int(p["seed"])).normal(0.0012 if p["seed"] == 5 else 0.0, 0.01, len(idx)), index=idx)
        return float(r.mean() / r.std() * np.sqrt(252)), r

    s2 = Study({"seed": Int(0, 20)}, seed=1).optimize(edge_objective, 21, "grid", grid_points=21)
    rep2 = s2.selection_report(n_boot=200)
    assert rep2["best_trial"] == 5 and rep2["romano_wolf_p_best"] < 0.05 and rep2["survivors_at_5pct"] >= 1 and rep2["deflated_sharpe_probability"] > rep["deflated_sharpe_probability"]
    with pytest.raises(ValueError):
        Study({"x": Float(0, 1)}).optimize(lambda p: p["x"], 3).selection_report()
    del rng


def test_tune_pipeline_runs_real_pipelines_and_reports_selection():
    rng = np.random.default_rng(11)
    n = 1200
    idx = pd.bdate_range("2012", periods=n)
    drift = np.where(np.arange(n) % 600 < 300, 0.0008, -0.0004)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(drift[:, None], 0.01, (n, 6)), axis=0), index=idx, columns=list("ABCDEF"))
    bundle = bundle_from_prices(prices, name="tune-test")
    spec = {"name": "tsmom-tune", "models": [{"name": "tsmom", "params": {"lookback": 126}}], "allocation": {"allocator": "sleeves"}}
    study = tune_pipeline(spec, bundle, load_config(), {"models.0.params.lookback": Int(40, 200)}, n_trials=6, n_blocks=3, seed=1)
    assert study.n_trials == 6 and study.best.params["models.0.params.lookback"] >= 40 and study.returns_matrix().shape[1] == 6
    rep = study.selection_report(n_boot=100, pbo_blocks=4)
    assert rep["n_trials"] == 6 and 0.0 <= rep["pbo"] <= 1.0
    plain = tune_pipeline(spec, bundle, load_config(), {"allocation.params.target_vol": Float(0.05, 0.2)}, n_trials=3, seed=2)
    assert plain.n_trials == 3 and np.isfinite(plain.best.value)


# --------------------------------------------------------------------------------------------------------------- profiling
def test_profile_call_and_timer():
    out = profiling.profile_call(lambda n: sum(i * i for i in range(n)), 200000, top=5)
    assert out["result"] == sum(i * i for i in range(200000)) and out["seconds"] > 0 and out["peak_mb"] >= 0 and 1 <= len(out["top"]) <= 5 and "cumulative_s" in out["top"]
    with profiling.Timer() as t:
        np.linalg.svd(np.random.default_rng(0).normal(size=(200, 200)))
    assert t.seconds > 0


def test_benchmark_models_reports_time_memory_and_errors():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2012", periods=700)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (700, 5)), axis=0), index=idx, columns=list("ABCDE"))
    table = profiling.benchmark_models(bundle_from_prices(prices, name="bench"), ["tsmom", "rsi2", "carry_xs"])
    assert set(table.index) == {"tsmom", "rsi2", "carry_xs"} and table.loc["tsmom", "scored_cells"] > 1000 and table.loc["tsmom", "seconds"] > 0
    assert "CARRY" in table.loc["carry_xs", "error"] and table.loc["tsmom", "error"] == ""


def test_scaling_exponent_distinguishes_linear_from_quadratic_work():
    lin = profiling.scaling_exponent(lambda n: np.cumsum(np.random.default_rng(0).normal(size=n * 400)), [500, 1000, 2000, 4000], repeats=3)
    quad = profiling.scaling_exponent(lambda n: np.outer(np.arange(n, dtype=float), np.arange(n, dtype=float)).sum(), [400, 800, 1600, 2400], repeats=3)
    assert 0.6 < lin["exponent"] < 1.4 and 1.5 < quad["exponent"] < 3.2 and quad["exponent"] > lin["exponent"] + 0.4 and list(lin["table"].columns) == ["n", "seconds"]
