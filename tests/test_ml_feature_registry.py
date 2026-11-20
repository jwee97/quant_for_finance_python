"""The feature registry and its evidence: every feature documented, per-feature information coefficients on a world with planted alpha, and ablations against a noise control."""

from __future__ import annotations

import warnings

import pytest

from src.ml import experiments as ex
from src.ml import feature_registry as fr
from src.ml.config import experiment_config
from src.ml.features import FEATURES, GROUPS
from src.utils.config import load_config
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def prepared():
    bundle = make_bundle(n_assets=10, n=1300, seed=3, alpha=1.0)
    cfg = experiment_config({"dataset": {"feature_groups": ["price", "momentum", "mean_reversion", "volatility"]}, "split": {"min_train_days": 504, "refit_every": 126}, "model": {"kind": "ridge"}})
    ds, info = ex.prepare(cfg, bundle, load_config(), None, None, None, None)
    return cfg, ds, info


def test_every_feature_is_documented_with_source_availability_calculation_and_hypothesis():
    reg = fr.registry()
    assert {r["name"] for r in reg} == set(FEATURES)
    for r in reg:
        assert r["source"] and r["availability"] and r["calculation"] and r["hypothesis"] and r["reference"], r["name"]
        assert r["lookback_days"] >= 1 and any(k in r["availability"] for k in ("close", "settlement", "interval", "publication")), r["name"]
    assert {g["group"] for g in fr.groups_summary()} == set(GROUPS)
    assert all("(" in g["reference"] or g["reference"].startswith("No settled") for g in fr.groups_summary())
    only = fr.registry(["momentum"])
    assert only and {r["group"] for r in only} == {"momentum"}


def test_a_feature_with_planted_information_ranks_above_noise_and_keeps_its_sign_across_years(prepared):
    cfg, ds, _ = prepared
    rows = fr.feature_table(ds)
    scored = [r for r in rows if "mean_ic" in r]
    assert len(scored) >= 10
    best = max(scored, key=lambda r: abs(r["mean_ic"]))
    assert abs(best["mean_ic"]) > 0.01 and best["n_years"] >= 3 and 0.0 <= best["years_same_sign"] <= 1.0
    noisy = fr.noise_dataset(ds, 6)
    nrows = [r for r in fr.feature_table(noisy) if "mean_ic" in r]
    assert max(abs(r["mean_ic"]) for r in nrows) < abs(best["mean_ic"]), "noise never beats the planted signal"
    assert max(abs(r["t_stat"]) for r in nrows) < 4.0


def test_a_dataset_can_be_restricted_to_chosen_features_and_unknown_ones_are_refused(prepared):
    _, ds, _ = prepared
    sub = fr.with_features(ds, ds.features[:3])
    assert sub.features == ds.features[:3] and len(sub.X) == len(ds.X) and sub.meta["feature_subset"]
    with pytest.raises(ValueError, match="not features"):
        fr.with_features(ds, ["no_such_feature"])


def test_the_ablation_reports_each_group_against_the_full_model_and_a_noise_control(prepared):
    cfg, ds, info = prepared
    out = fr.ablation(ds, cfg, info, ["momentum", "volatility"])
    names = [r["variant"] for r in out["rows"]]
    assert names[0] == "all features" and "without momentum" in names and "without volatility" in names
    assert out["noise"]["variant"] == "noise features only"
    assert out["base"]["rank_ic"] > out["noise"]["rank_ic"] + 0.005, "the planted alpha is found by the real features and not by noise"
    for r in out["rows"][1:]:
        assert r["ic_change"] == pytest.approx(r["rank_ic"] - out["base"]["rank_ic"]) and r["n_features"] < out["base"]["n_features"]
    assert "not tested for significance" in out["reading"]
    with pytest.raises(ValueError, match="none of the requested groups"):
        fr.ablation(ds, cfg, info, ["futures"])
