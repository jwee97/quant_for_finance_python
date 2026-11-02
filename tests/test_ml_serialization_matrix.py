"""P0-12: every model family survives save and reload with its predictions, extras, feature order and (for agents and networks) its fitted state, and damaged or mismatched artifacts are refused clearly."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as dsm, models as md
from src.utils.optional import unavailable_reason
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
TREE, NET = 1e-9, 1e-5
DEEP = {"epochs": 2, "max_rows": 1500, "batch_size": 128, "lookback": 8, "patience": 2}
RL = {"total_timesteps": 256, "batch_size": 32, "hidden": 8, "episode_days": 60}
FAMILIES = {                                   # kind: (params, absolute tolerance, packages it needs); EVERY kind of the catalogue is here, and a test says so
    "linear": ({}, TREE, ()),
    "ridge": ({"alpha": 10.0}, TREE, ()),
    "lasso": ({"alpha": 0.02}, TREE, ()),
    "elasticnet": ({"alpha": 0.02}, TREE, ()),
    "logistic": ({}, TREE, ()),
    "random_forest": ({"n_estimators": 20, "max_depth": 4}, TREE, ()),
    "hist_gb": ({"max_iter": 20}, TREE, ()),
    "xgboost": ({"n_estimators": 20}, TREE, ("xgboost",)),
    "lightgbm": ({"n_estimators": 20}, TREE, ("lightgbm",)),
    "catboost": ({"iterations": 20}, TREE, ("catboost",)),
    "ar": ({}, TREE, ()),
    "kalman": ({"max_features": 6}, TREE, ()),
    "online_ridge": ({}, TREE, ()),
    "online_logistic": ({}, TREE, ()),
    "regime_ridge": ({"min_rows": 100}, TREE, ()),
    "stable_select": ({"subsamples": 10}, TREE, ()),
    "lgbm_ranker": ({"n_estimators": 20}, TREE, ("lightgbm",)),
    "xgb_ranker": ({"n_estimators": 20}, TREE, ("xgboost",)),
    "lgbm_quantile": ({"n_estimators": 20}, TREE, ("lightgbm",)),
    "ngboost": ({"n_estimators": 20, "max_rows": 3000}, TREE, ("ngboost",)),
    "hmm_regime": ({"n_states": 2}, TREE, ("hmmlearn",)),
    "markov_switching": ({"n_states": 2}, TREE, ("statsmodels",)),
    "lstm": ({**DEEP, "hidden": 8}, NET, ("torch",)),
    "gru": ({**DEEP, "hidden": 8}, NET, ("torch",)),
    "tcn": ({**DEEP, "hidden": 8, "layers": 2, "kernel_size": 3}, NET, ("torch",)),
    "patchtst": ({**DEEP, "hidden": 8, "patch_len": 4, "stride": 2, "d_model": 8, "n_heads": 2, "ff_dim": 16, "channels": 4, "layers": 1}, NET, ("torch",)),
    "nbeats": ({**DEEP, "blocks": 2, "width": 8, "layers": 2}, NET, ("torch",)),
    "nhits": ({**DEEP, "blocks": 2, "width": 8, "layers": 1, "pool_sizes": "2,1"}, NET, ("torch",)),
    "tft": ({**DEEP, "hidden": 8, "n_heads": 2}, NET, ("torch",)),
    "autoencoder": ({**DEEP, "hidden": 8, "latent_dim": 4, "lookback": 6, "layers": 1}, NET, ("torch",)),
    "vae": ({**DEEP, "hidden": 8, "latent_dim": 4, "lookback": 6, "layers": 1}, NET, ("torch",)),
    "gcn": ({"epochs": 2, "hidden": 8, "layers": 1, "batch_size": 64, "patience": 2}, NET, ("torch",)),
    "gat": ({"epochs": 2, "hidden": 8, "layers": 1, "heads": 2, "batch_size": 64, "patience": 2}, NET, ("torch",)),
    "model_ensemble": ({"members": "ridge,lgbm_quantile[n_estimators=20]", "weighting": "equal"}, TREE, ("lightgbm",)),
    "ppo": ({**RL, "n_steps": 64, "n_epochs": 1}, NET, ("gymnasium", "stable_baselines3")),
    "sac": ({**RL, "buffer_size": 500, "learning_starts": 50}, NET, ("gymnasium", "stable_baselines3")),
    "recurrent_ppo": ({**RL, "n_steps": 64, "n_epochs": 1, "lstm_hidden": 8}, NET, ("gymnasium", "stable_baselines3", "sb3_contrib")),
}
CLASSIFICATION = {"logistic", "online_logistic"}      # fitted on the direction of the return
NEEDS_REGIME = {"regime_ridge"}                      # needs the regime_state feature, which the dataset adds when a detector is declared


@pytest.fixture(scope="module")
def ds():
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 300}}, defaults=mc.DEFAULTS)
    return dsm.build_dataset(make_bundle(7, 800, seed=71, alpha=1.0), cfg["dataset"], "serialization")


@pytest.fixture(scope="module")
def ds_regime():
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"], "regime_detector": "vol_state"}, "split": {"min_train_days": 300}}, defaults=mc.DEFAULTS)
    return dsm.build_dataset(make_bundle(7, 800, seed=71, alpha=1.0), cfg["dataset"], "serialization_regime")


def fitted(ds, kind):
    params, _, needs = FAMILIES[kind]
    for pkg in needs:
        if unavailable_reason(pkg):
            if pkg == "ngboost" and int(pd.__version__.split(".")[0]) >= 3:
                pytest.skip("NOT_APPLICABLE: ngboost needs lifelines, which needs pandas below 3, so it cannot be installed beside this pandas (the Python 3.10 job covers it)")
            if pkg == "catboost":
                pytest.skip("NOT_APPLICABLE: catboost is an optional boosting package that the CI lanes do not install, so the catalogue lists the model as unavailable here (the evidence environment installs it)")
            pytest.skip(f"{pkg} is not available here")
    pos = ds.positions()
    rows = np.flatnonzero((pos >= 80) & (pos < 520) & ds.Y[ds.target].notna().to_numpy())
    task = "classification" if kind in CLASSIFICATION else "regression"
    y = ds.Y[ds.target].iloc[rows]
    y = (y > 0).astype(float) if task == "classification" else y
    model = md.make_model(kind, params, task, 0).bind(ds)
    model.fit(ds.X.iloc[rows], y, pos[rows])
    test = ds.X.iloc[np.flatnonzero((pos >= 540) & (pos < 600))]
    return model, test


def test_every_model_kind_of_the_catalogue_is_in_the_matrix():
    """A kind added to the catalogue without a save-and-reload test fails here, so the matrix cannot fall behind the zoo."""
    assert set(FAMILIES) == {m["kind"] for m in md.model_catalog()}


@pytest.mark.parametrize("kind", sorted(FAMILIES))
def test_a_family_reloads_with_the_same_features_predictions_and_extras(ds, ds_regime, kind):
    model, test = fitted(ds_regime if kind in NEEDS_REGIME else ds, kind)
    tol = FAMILIES[kind][1]
    blob = model.to_bytes()
    assert blob[:4] == b"QFM1"
    again = md.Model.from_bytes(blob).bind(ds)
    assert again.features_ == model.features_ and again.kind == model.kind and again.params == model.params
    assert np.allclose(again.predict(test), model.predict(test), atol=tol, rtol=0)
    ex, ex2 = model.predict_extras(test), again.predict_extras(test)
    assert set(ex) == set(ex2)
    for name in ex:
        assert np.allclose(ex[name].to_numpy(dtype=float), ex2[name].to_numpy(dtype=float), atol=max(tol, 1e-6), equal_nan=True), name
    assert np.allclose(again.importance().to_numpy(), model.importance().to_numpy(), atol=max(tol, 1e-6))
    assert md.Model.from_bytes(again.to_bytes()).features_ == model.features_, "a second round trip changes nothing"


@pytest.mark.parametrize("kind", sorted(FAMILIES))
def test_data_without_a_feature_the_model_was_fitted_on_is_refused_by_name(ds, ds_regime, kind):
    model, test = fitted(ds_regime if kind in NEEDS_REGIME else ds, kind)
    dropped = test.drop(columns=[model.features_[2]])
    with pytest.raises(md.ModelSchemaError, match=model.features_[2]):
        model.predict(dropped)
    reordered = test[list(reversed(test.columns))]
    assert np.allclose(model.predict(reordered), model.predict(test), atol=1e-6), "columns are found by name, not by position"


def test_a_damaged_truncated_or_foreign_artifact_is_refused_and_an_old_one_without_an_envelope_is_still_read(ds):
    import io

    import joblib

    model, test = fitted(ds, "ridge")
    blob = bytearray(model.to_bytes())
    flipped = bytes(blob[:60] + bytes([blob[60] ^ 0xFF]) + blob[61:])
    for bad, text in ((flipped, "checksum"), (bytes(blob[:-9]), "checksum"), (bytes(blob[:20]), "checksum"), (b"QFM1" + b"\x00" * 32 + b"junk", "checksum")):
        with pytest.raises(md.ModelArtifactError, match=text):
            md.Model.from_bytes(bad)
    for bad in (b"", b"not a model at all", b"\x80\x04garbage"):
        with pytest.raises(md.ModelArtifactError, match="cannot be read"):
            md.Model.from_bytes(bad)
    foreign = io.BytesIO()
    joblib.dump({"not": "a model"}, foreign)
    with pytest.raises(md.ModelArtifactError, match="not a model"):
        md.Model.from_bytes(foreign.getvalue())
    legacy = io.BytesIO()
    joblib.dump(model, legacy, compress=3)                                              # what was saved before the envelope existed
    assert np.allclose(md.Model.from_bytes(legacy.getvalue()).predict(test), model.predict(test))


def test_a_catboost_fit_writes_nothing_into_the_working_directory(ds, tmp_path, monkeypatch):
    """CatBoost writes its training log to ./catboost_info unless told not to: a fit inside the dashboard must not litter the folder the server was started in."""
    monkeypatch.chdir(tmp_path)
    fitted(ds, "catboost")
    assert list(tmp_path.iterdir()) == []


def test_a_recurrent_agent_keeps_its_replay_state_across_a_reload(ds):
    model, test = fitted(ds, "ppo")
    before = pd.Series(model.predict(test), index=test.index)
    again = md.Model.from_bytes(model.to_bytes()).bind(ds)
    assert np.allclose(before, again.predict(test), atol=NET) and again.diagnostics()["state_schema"] == 2
