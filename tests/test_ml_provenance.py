"""Release 3, P0-24: provenance identity covers the complete input history, the code, the configuration, the features, the folds and the output contract.

The defect these tests pin: ``bundle_fingerprint`` hashes the last 250 price rows, so a restated price from three years ago left the "data version" unchanged. The new identity must change when ANY old value or
publication detail changes, must not change when only the environment does, and must not depend on the datetime unit pandas happened to choose."""

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import datasets as ds_mod, provenance as pv, registry as rg
from src.ml.config import experiment_config
from src.ml.experiments import run_experiment
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
BASE = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 400, "refit_every": 126}, "model": {"kind": "ridge", "params": {}}}


@pytest.fixture(scope="module")
def bundle():
    return make_bundle(6, 1300, seed=21, alpha=1.0)


def dataset(bundle):
    cfg = experiment_config(BASE)
    return cfg, ds_mod.build_dataset(bundle, cfg["dataset"], "prov")


# --------------------------------------------------------------------------------------------------------------------- content hashing
def test_a_frame_hash_changes_with_any_value_and_ignores_the_datetime_unit():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(300, 3)), index=idx, columns=list("abc"))
    base = pv.frame_hash(df)["sha256"]
    for row, col in ((0, 0), (150, 1), (299, 2)):                                   # the first, a middle and the last cell
        other = df.copy()
        other.iloc[row, col] += 1e-9
        assert pv.frame_hash(other)["sha256"] != base
    us = df.copy()
    us.index = us.index.as_unit("us")
    ns = df.copy()
    ns.index = ns.index.as_unit("ns")
    assert pv.frame_hash(us)["sha256"] == pv.frame_hash(ns)["sha256"] == base        # pandas 3 chooses the unit by what it parsed; the identity must not
    assert pv.frame_hash(df[["b", "a", "c"]])["sha256"] != base                      # column order is part of the data
    assert pv.frame_hash(df.iloc[1:])["sha256"] != base
    with_nan = df.copy()
    with_nan.iloc[5, 1] = np.nan
    assert pv.frame_hash(with_nan)["sha256"] == pv.frame_hash(with_nan.copy())["sha256"] != base
    assert pv.frame_hash(None)["sha256"] is None and pv.frame_hash(df["a"])["rows"] == 300


def test_an_old_price_changes_the_input_manifest_although_the_legacy_fingerprint_does_not(bundle):
    old = bundle.prices.index[7]
    mutated_prices = bundle.prices.copy()
    mutated_prices.loc[old, mutated_prices.columns[0]] *= 1.0001                      # a restated price three years before the end
    other = dataclasses.replace(bundle, prices=mutated_prices)
    assert ds_mod.bundle_fingerprint(other) == ds_mod.bundle_fingerprint(bundle)       # the defect: the cache key sees nothing
    assert pv.input_manifest(other)["sha256"] != pv.input_manifest(bundle)["sha256"]    # the identity sees it
    assert pv.input_manifest(bundle)["sha256"] == pv.input_manifest(dataclasses.replace(bundle))["sha256"]


def test_a_changed_macro_value_or_publication_detail_changes_the_input_manifest(bundle):
    base = pv.input_manifest(bundle)["sha256"]
    macro = bundle.macro.copy()
    assert macro.shape[1] > 0
    macro.iloc[3, 0] += 0.5                                                           # a restated macro value long ago
    assert pv.input_manifest(dataclasses.replace(bundle, macro=macro))["sha256"] != base
    meta = {c: {"release_lag_days": 1, "source": "test"} for c in bundle.macro.columns}
    later = {c: {"release_lag_days": 2, "source": "test"} for c in bundle.macro.columns}   # the same values, published a day later
    a = pv.input_manifest(dataclasses.replace(bundle, macro_meta=meta))["sha256"]
    assert a != pv.input_manifest(dataclasses.replace(bundle, macro_meta=later))["sha256"] != base
    assert pv.input_manifest(dataclasses.replace(bundle, group={**bundle.group, "x": "y"}))["sha256"] != base


# --------------------------------------------------------------------------------------------------------------------- the record
def test_the_identity_is_stable_for_equal_inputs_and_moves_with_each_component(bundle):
    cfg, ds = dataset(bundle)
    a = pv.provenance_record(cfg, ds)
    _, ds2 = dataset(bundle)
    b = pv.provenance_record(cfg, ds2)
    assert a["identity"] == b["identity"] and pv.compare(a, b) == {"same_identity": True, "differs": [], "environment_differs": False}
    mutated_prices = bundle.prices.copy()
    mutated_prices.iloc[11, 2] *= 1.01
    other_bundle = dataclasses.replace(bundle, prices=mutated_prices, returns=mutated_prices.pct_change(fill_method=None))
    _, ds3 = dataset(other_bundle)
    c = pv.provenance_record(cfg, ds3)
    assert c["identity"] != a["identity"] and "data" in pv.compare(a, c)["differs"]
    d = pv.provenance_record(experiment_config({**BASE, "seed": 5}), ds)
    assert d["identity"] != a["identity"] and {"config", "seed"} <= set(pv.compare(a, d)["differs"])
    # every identity-bearing field is named in the record
    for key in ("data", "code", "environment", "hardware", "config", "seeds", "features", "folds", "output_contract"):
        assert key in a
    assert a["config"]["merged"]["model"]["kind"] == "ridge" and a["features"]["names"] == ds.features and a["data"]["frames"]["X"]["rows"] == len(ds.X)


def test_the_environment_is_recorded_but_is_not_part_of_the_identity(bundle, monkeypatch):
    cfg, ds = dataset(bundle)
    a = pv.provenance_record(cfg, ds)
    real = pv.environment_record

    def other_machine():
        env = real()
        env["python"] = "3.99.0"
        env["sha256"] = pv.digest(env, 64)
        return env

    monkeypatch.setattr(pv, "environment_record", other_machine)
    b = pv.provenance_record(cfg, ds)
    assert b["identity"] == a["identity"] and pv.compare(a, b)["environment_differs"] is True and b["environment"]["python"] == "3.99.0"


def test_the_source_tree_hash_follows_the_code_and_never_invents_a_commit(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(pv, "ROOT", tmp_path)
    monkeypatch.setattr(pv, "_TREE_CACHE", {})
    first = pv.source_tree_hash()
    assert first["git_commit"] is None and "commit unavailable" in first["git_note"] and first["files"] == 1       # an uploaded archive without .git
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    monkeypatch.setattr(pv, "_TREE_CACHE", {})
    assert pv.source_tree_hash()["sha256"] != first["sha256"]


# --------------------------------------------------------------------------------------------------------------------- end to end
def test_a_saved_experiment_carries_its_provenance_and_a_repeat_reproduces_it(bundle, tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    one = run_experiment(BASE, bundle=bundle, registry=reg, explain=False, robustness=False)
    two = run_experiment(BASE, bundle=bundle, registry=reg, explain=False, robustness=False)
    rec1, rec2 = pv.load(reg, one.id), pv.load(reg, two.id)
    assert rec1 is not None and (reg.root / "experiments" / one.id / "provenance.json").exists()
    assert one.report["reproducibility"]["identity"] == rec1["identity"] and rec1["code"]["sha256"] == pv.source_tree_hash()["sha256"]
    assert rec1["identity"] == rec2["identity"], "the same data, code, configuration, seed, features, folds and contract is the same experiment"
    out = pv.reproducible(rec1, rec2, one.train.predictions.to_numpy(), two.train.predictions.to_numpy(), "ridge")
    assert out["reproduced"] and out["max_abs_difference"] == 0.0 and out["tolerance"] == 0.0
    assert rec1["output_contract"]["contract"]["output_kind"] == "expected_return" and rec1["folds"]["n"] == len(one.train.folds)
    loaded = reg.report(one.id)
    assert loaded["reproducibility"]["identity"] == rec1["identity"]                    # the report on disk carries it too
    changed = run_experiment({**BASE, "model": {"kind": "ridge", "params": {"alpha": 50.0}}}, bundle=bundle, registry=reg, explain=False, robustness=False)
    assert pv.load(reg, changed.id)["identity"] != rec1["identity"]


def test_a_legacy_experiment_says_not_recorded_and_is_not_reconstructed(bundle, tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    assert pv.load(reg, "never-recorded") is None and pv.summary(None)["status"] == "not recorded"


def test_tolerance_is_a_property_of_the_family_and_the_device():
    assert pv.family_tolerance("ridge") == 0.0 and pv.family_tolerance("random_forest") == 0.0
    assert 0.0 < pv.family_tolerance("lstm") < pv.family_tolerance("lstm", "cuda") and pv.family_tolerance("ppo") > 0.0
    a = {"identity": "x", "hardware": {"backend": "cpu"}, "data": {"data_hash": "d"}, "code": {"sha256": "c"}, "config": {"hash": "h"}, "seeds": {"seed": 0}, "features": {"hash": "f"}, "folds": {"hash": "g"},
         "output_contract": {"hash": "o"}, "environment": {"sha256": "e"}}
    p = np.array([0.1, 0.2, 0.3])
    assert pv.reproducible(a, a, p, p + 1e-6, "lstm")["reproduced"] and not pv.reproducible(a, a, p, p + 1e-6, "ridge")["reproduced"]
    assert not pv.reproducible(a, {**a, "identity": "y"}, p, p, "ridge")["reproduced"]
