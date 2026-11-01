"""P0-22: one artifact holds the model, the schema it was fitted on, its preprocessing description, its cutoffs and its label convention; schema drift is refused; a reloaded bundle reproduces the evaluation.

Every family that can run here is saved, loaded and asked for the predictions the walk-forward evaluation saved (with the history it would have had on each decision date). The artifact's checksum and version are
checked against damage. The schema is attacked column by column: a missing, extra, reordered or retyped feature is refused in strict mode, a feature that arrives in another order is shown not to change a prediction
(every model selects by name), the width of a float is not a change, and a declared, versioned adapter is accepted only for the exact change it was written for, and recorded when it is used.
"""

import warnings

import numpy as np
import pytest

from src.ml import bundle as B, config as mc, datasets as dsm, models as md, training as tr
from src.utils.optional import unavailable_reason
from tests.ml_helpers import FAMILIES, make_bundle

warnings.filterwarnings("ignore")
CFG = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 250, "refit_every": 100}}, defaults=mc.DEFAULTS)
SPLIT = CFG["split"]
_CACHE: dict = {}


@pytest.fixture(scope="module")
def ds():
    return dsm.build_dataset(make_bundle(6, 800, seed=91, alpha=1.0), CFG["dataset"], "bundle")


def family(ds, name):
    """``(walk-forward result, first fold's model, the first 36 days of its block, tolerance)``, once per family."""
    kind, params, tol, needs = FAMILIES[name]
    for pkg in needs:
        if unavailable_reason(pkg):
            pytest.skip(f"{pkg} is not available here")
    if name not in _CACHE:
        res = tr.walk_forward_train(ds, {"kind": kind, "params": params}, SPLIT, 0)
        idx = [i for i, m in enumerate(res.models) if m is not None][0]
        _CACHE[name] = (res, res.models[idx], ds.X.iloc[res.fold_rows[idx][: 36 * len(ds.assets)]], tol)
    return _CACHE[name]


# ------------------------------------------------------------------------------------------------------------------ round trip
@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_reloaded_bundle_reproduces_the_predictions_the_evaluation_saved(ds, name, tmp_path):
    res, model, X, tol = family(ds, name)
    path = B.InferenceBundle.from_model(model, ds).save(tmp_path / f"{name}.qfb")
    loaded = B.InferenceBundle.load(path)
    out = loaded.reproduces(X, res.predictions, ds, tol=max(tol, 1e-6))
    assert out["reproduced"] and out["rows"] == len(X), (name, out)
    assert loaded.header["needs_history"] == model.needs_dataset and loaded.header["kind"] == model.kind
    if model.needs_dataset:
        with pytest.raises(ValueError, match="reads history"):
            loaded.predict(X)


def test_the_header_names_what_a_later_reader_needs_to_know(ds, tmp_path):
    res, model, X, _ = family(ds, "ridge")
    h = B.InferenceBundle.load(B.InferenceBundle.from_model(model, ds).save(tmp_path / "r.qfb")).header
    assert h["schema"]["features"] == list(ds.features) and h["schema"]["hash"] == model.fit_info_["schema"]["hash"] and set(h["schema"]["kinds"]) == {"f"}
    assert h["label"]["lag"] == ds.lag and h["label"]["horizon"] == ds.horizon and h["label"]["target"] == ds.target and h["label"]["output_kind"] == "expected_return"
    assert h["label"]["convention"]["entry"] == "close of day t + lag" and h["asset_order"] == list(ds.assets)
    assert h["preprocessing"]["model_steps"] == ["impute", "scale"] and h["preprocessing"]["dataset"]["normalize"] == ds.meta["normalize"] and h["preprocessing"]["target_scaling"]
    assert h["fit_info"]["fit_cutoff"] == model.fit_info_["fit_cutoff"] == h["state"]["information_cutoff"] and h["fit_info"]["train_rows_hash"] == model.fit_info_["train_rows_hash"]
    assert h["state"]["chronological"] == "none" and h["bundle_version"] == B.VERSION
    hmm = family(ds, "hmm")[1] if not unavailable_reason("hmmlearn") else None
    if hmm is not None:
        hh = B.InferenceBundle.from_model(hmm, ds).header
        assert hh["needs_history"] and "bind" in hh["state"]["chronological"] and hh["state"]["information_cutoff"] == hmm.fit_info_["fit_cutoff"]


def test_the_chronological_state_of_a_regime_model_and_an_ensemble_is_rebuilt_from_the_history_it_is_given(ds, tmp_path):
    for name, key in (("hmm", "regimes"), ("ensemble", "weights")):
        if any(unavailable_reason(p) for p in FAMILIES[name][3]):
            continue
        res, model, X, tol = family(ds, name)
        loaded = B.InferenceBundle.load(B.InferenceBundle.from_model(model, ds).save(tmp_path / f"{name}.qfb"))
        loaded.predict(X, ds)
        again = loaded.model.predict_extras(X)[key]
        first = model.predict_extras(X)[key]
        assert np.allclose(again.to_numpy(dtype=float), first.reindex(again.index).to_numpy(dtype=float), atol=1e-6), f"{name}: the {key} state differs after a reload"


def test_a_bundle_for_a_model_that_did_not_come_through_the_trainer_needs_a_sample_of_its_features(ds):
    plain = md.make_model("ridge", {"alpha": 5.0}, "regression", 0).fit(ds.X.iloc[:600], ds.Y[ds.target].iloc[:600])
    with pytest.raises(ValueError, match="no fit-time schema"):
        B.InferenceBundle.from_model(plain)
    b = B.InferenceBundle.from_model(plain, sample=ds.X.iloc[:10])
    assert b.schema["features"] == list(ds.features) and set(b.schema["kinds"]) == {"f"}


# ------------------------------------------------------------------------------------------------------------------ the artifact
def test_a_damaged_foreign_or_newer_bundle_is_refused_with_a_reason(ds, tmp_path):
    blob = B.InferenceBundle.from_model(family(ds, "ridge")[1], ds).to_bytes()
    flipped = bytearray(blob)
    flipped[len(flipped) // 2] ^= 0xFF
    with pytest.raises(B.BundleError, match="checksum"):
        B.InferenceBundle.from_bytes(bytes(flipped))
    with pytest.raises(B.BundleError, match="checksum"):
        B.InferenceBundle.from_bytes(blob[:-40])
    with pytest.raises(B.BundleError, match="not an inference bundle"):
        B.InferenceBundle.from_bytes(b"XXXX" + blob[4:])
    import hashlib
    import json
    import struct

    head_len = struct.unpack(">I", blob[36:40])[0]
    header = json.loads(blob[40:40 + head_len])
    header["bundle_version"] = B.VERSION + 1
    head = json.dumps(header, sort_keys=True).encode()
    body = struct.pack(">I", len(head)) + head + blob[40 + head_len:]
    with pytest.raises(B.BundleError, match="version"):
        B.InferenceBundle.from_bytes(B.MAGIC + hashlib.sha256(body).digest() + body)
    assert isinstance(B.BundleError("x"), md.ModelArtifactError)


# ------------------------------------------------------------------------------------------------------------------ schema drift
@pytest.fixture(scope="module")
def ridge(ds):
    res, model, X, tol = family(ds, "ridge")
    return B.InferenceBundle.from_model(model, ds), model, X


def test_a_missing_extra_reordered_or_retyped_feature_is_refused_in_strict_mode(ridge):
    bundle, model, X = ridge
    one = X.columns[0]
    cases = {
        "missing": (X.drop(columns=[one]), f"missing \\['{one}'\\]"),
        "extra": (X.assign(surprise=1.0), "unexpected \\['surprise'\\]"),
        "reordered": (X[list(X.columns[::-1])], "different order"),
        "retyped": (X.assign(**{one: X[one].astype("int64")}), f"{one}: f->i"),
    }
    for label, (data, text) in cases.items():
        with pytest.raises(B.SchemaDriftError, match=text):
            bundle.predict(data)
        assert isinstance(B.SchemaDriftError("x"), md.ModelSchemaError), label
    assert np.allclose(bundle.predict(X), model.predict(X)), "the unchanged schema is accepted"


def test_the_width_of_a_float_is_not_a_schema_change_and_does_not_move_a_prediction(ridge):
    bundle, model, X = ridge
    narrow = X.astype("float32")
    assert not any(B.schema_drift(bundle.schema, narrow)[k] for k in ("missing", "extra", "reordered", "dtype_changes"))
    assert np.allclose(bundle.predict(narrow), model.predict(X), atol=1e-5)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_family_selects_its_features_by_name_so_a_different_order_changes_nothing(ds, name):
    res, model, X, tol = family(ds, name)
    shuffled = X[list(np.random.default_rng(0).permutation(X.columns))]
    assert np.allclose(np.asarray(model.predict(shuffled), dtype=float), np.asarray(model.predict(X), dtype=float), atol=max(tol, 1e-6), equal_nan=True), name
    loaded = B.InferenceBundle.from_model(model, ds)
    assert np.allclose(loaded.predict(shuffled, ds, strict=False), loaded.predict(X, ds), atol=max(tol, 1e-6), equal_nan=True), "non-strict mode lets the order through and the answer is the same"


def test_the_model_itself_refuses_a_feature_that_has_become_text(ridge):
    bundle, model, X = ridge
    broken = X.copy()
    broken[X.columns[1]] = broken[X.columns[1]].astype(str)
    with pytest.raises(md.ModelSchemaError, match="its schema has drifted"):
        model.predict(broken)


# ------------------------------------------------------------------------------------------------------------------ versioned adapters
def test_a_declared_adapter_is_applied_for_exactly_the_change_it_was_written_for_and_recorded(ds, ridge):
    _, model, X = ridge
    old, new = X.columns[0], "renamed_" + X.columns[0]
    renamed = X.rename(columns={old: new})
    kinds = [np.dtype(renamed[c].dtype).kind for c in renamed.columns]
    adapter = {"version": "2", "from_hash": B.schema_hash(list(renamed.columns), kinds), "rename": {new: old}, "description": f"{old} was renamed {new} in the feature library"}
    bundle = B.InferenceBundle.from_model(model, ds, adapters=[adapter])
    with pytest.raises(B.SchemaDriftError):
        B.InferenceBundle.from_model(model, ds).predict(renamed)
    conformed, report = bundle.conform(renamed)
    assert list(conformed.columns) == list(X.columns) and report["adapters_applied"] == [{"version": "2", "from_hash": adapter["from_hash"], "description": adapter["description"]}]
    assert np.allclose(bundle.predict(renamed), model.predict(X))
    other = X.rename(columns={X.columns[2]: "something_else"})                                       # a different change: the adapter does not match it
    with pytest.raises(B.SchemaDriftError, match="something_else"):
        bundle.predict(other)
    more = renamed.drop(columns=[X.columns[3]])                                                       # the adapter's hash no longer matches either, so it does not hide a second change
    with pytest.raises(B.SchemaDriftError):
        bundle.predict(more)
    survived = B.InferenceBundle.from_bytes(bundle.to_bytes())
    assert survived.header["adapters"][0]["version"] == "2" and np.allclose(survived.predict(renamed), model.predict(X)), "the adapter is part of the saved artifact"
