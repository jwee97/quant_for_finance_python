"""The inference bundle: everything a fitted model needs to answer, in one checksummed artifact, and a schema that refuses to guess (P0-22).

A model on its own is a fitted state and a list of feature names. To answer later it also needs the exact columns it was fitted on (names, order, kinds of dtype), the preprocessing that was fitted with it (the
imputer's fills, the scaler's moments, the target scaling), the label convention and horizon its outputs mean, the day it was fitted and the last day its labels reached, and, for a model that reads history (a
regime filter, a simulated account, dynamic weights), the knowledge that it must be given that history. ``InferenceBundle`` carries those together; ``predict`` refuses data whose schema has drifted (a missing, extra,
reordered or retyped feature) unless a declared, versioned ``adapter`` is registered for exactly that change, and then records that it was applied.

A reloaded bundle reproduces the evaluation's predictions (``reproduces``) when it is given the same features and the same history up to each decision: its chronological state is that history, bound as a causal
view ending on the decision date, never anything the file kept from a later one.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .causal import CausalView
from .contracts import output_kind
from .intervals import CONVENTION
from .models import Model, ModelArtifactError, ModelSchemaError

MAGIC = b"QFB1"
VERSION = 1
NUMERIC = "fiu"


class SchemaDriftError(ModelSchemaError):
    """Data whose columns are not the ones the model was fitted on, and no registered adapter says how to bring them into line."""


def kind_of(dtype) -> str:
    """The one-letter kind of a dtype: 'f' float, 'i' integer, 'b' boolean, 'O' object or text. Pandas extension dtypes (the text dtype of pandas 3 among them) have one too, where ``np.dtype`` would refuse them."""
    return str(getattr(dtype, "kind", "O"))


def schema_hash(features: list, kinds: list) -> str:
    return hashlib.sha256(json.dumps([list(features), list(kinds)]).encode()).hexdigest()[:16]


def describe_schema(X: pd.DataFrame, features: list | None = None) -> dict:
    """The names, order, dtypes and dtype kinds of the columns a model reads from ``X``, and a hash of names and kinds."""
    cols = list(features) if features is not None else list(X.columns)
    kinds = [kind_of(X[c].dtype) if c in X.columns else "?" for c in cols]
    return {"features": cols, "dtypes": [str(X[c].dtype) if c in X.columns else "?" for c in cols], "kinds": kinds, "hash": schema_hash(cols, kinds)}


def schema_drift(expected: dict, X: pd.DataFrame) -> dict:
    """How ``X`` differs from the schema a model was fitted on: ``missing`` and ``extra`` columns, whether the common ones are ``reordered``, and ``dtype_changes`` ({column: (kind then, kind now)}). The width of a float
    (float32 against float64) is not a schema change."""
    want = list(expected["features"])
    have = set(X.columns)
    missing = [c for c in want if c not in have]
    extra = [c for c in X.columns if c not in set(want)]
    reordered = not missing and [c for c in X.columns if c in set(want)] != want
    kinds = dict(zip(want, expected.get("kinds") or [None] * len(want)))
    changes = {c: (kinds[c], kind_of(X[c].dtype)) for c in want if c in have and kinds.get(c) and kind_of(X[c].dtype) != kinds[c]}
    return {"missing": missing, "extra": extra, "reordered": bool(reordered), "dtype_changes": changes}


def drift_sentence(drift: dict) -> str:
    parts = []
    if drift["missing"]:
        parts.append(f"missing {drift['missing'][:6]}{'...' if len(drift['missing']) > 6 else ''}")
    if drift["extra"]:
        parts.append(f"unexpected {drift['extra'][:6]}{'...' if len(drift['extra']) > 6 else ''}")
    if drift["reordered"]:
        parts.append("columns in a different order")
    if drift["dtype_changes"]:
        parts.append("dtype kinds changed " + ", ".join(f"{c}: {a}->{b}" for c, (a, b) in list(drift["dtype_changes"].items())[:4]))
    return "; ".join(parts)


def adapt(X: pd.DataFrame, adapter: dict) -> pd.DataFrame:
    """Apply one declared transformation: ``rename`` (old -> new), ``drop`` (columns to remove) and ``reorder`` (put the common columns in the order the model was fitted on, by ``order``)."""
    out = X.rename(columns=dict(adapter.get("rename") or {}))
    out = out.drop(columns=[c for c in (adapter.get("drop") or []) if c in out.columns])
    if adapter.get("order"):
        out = out[[c for c in adapter["order"] if c in out.columns] + [c for c in out.columns if c not in set(adapter["order"])]]
    return out


def describe_preprocessing(model: Model) -> dict:
    """What was fitted alongside the model, from the model itself: the steps of a scikit-learn pipeline, a feature scaler, the target scaling. The cross-sectional scaling of the dataset is described by the caller."""
    steps = []
    pipe = getattr(model, "pipe_", None)
    if pipe is not None:
        steps = [name for name, _ in pipe.steps[:-1]]
    if getattr(model, "scaler_", None) is not None:
        steps.append(f"{type(model.scaler_).__name__} (fitted on the training part of the window)")
    return {"model_steps": steps, "target_scaling": bool(hasattr(model, "y_mean_") and model.task == "regression"), "fills": "learned from the model's eligible training rows"}


class InferenceBundle:
    """One model, its schema, its preprocessing description, its cutoffs and its label convention; ``predict`` checks the data against all of them."""

    def __init__(self, header: dict, blob: bytes):
        self.header, self.blob = header, blob
        self._model: Model | None = None

    # -- building
    @classmethod
    def from_model(cls, model: Model, ds=None, sample: pd.DataFrame | None = None, adapters: list | None = None) -> "InferenceBundle":
        info = dict(getattr(model, "fit_info_", None) or {})
        schema = info.get("schema")
        if schema and "kinds" not in schema:
            schema = {**schema, "kinds": [pd.api.types.pandas_dtype(t).kind for t in schema["dtypes"]]}
        if not schema:
            if sample is None:
                raise ValueError("the model carries no fit-time schema (it was not fitted through the trainer): give a sample of the features it was fitted on")
            schema = describe_schema(sample, model.features_)
        meta = getattr(ds, "meta", {}) if ds is not None else {}
        lag, horizon = (int(ds.lag), int(ds.horizon)) if ds is not None else (None, None)
        target = getattr(ds, "target", None)
        header = {
            "bundle_version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": model.kind, "task": model.task, "params": json.loads(json.dumps(model.params, default=str)),
            "schema": schema, "fit_info": {k: v for k, v in info.items() if k != "schema"},
            "label": {"convention": meta.get("label_convention", CONVENTION), "lag": lag, "horizon": horizon, "target": target,
                      "output_kind": output_kind(model.kind, model.task, target or "") if target else None},
            "asset_order": list(getattr(ds, "assets", []) or []), "needs_history": bool(model.needs_dataset),
            "preprocessing": {**describe_preprocessing(model), "dataset": {"normalize": meta.get("normalize"), "winsor": meta.get("winsor"), "scope": "per date, across the assets of that date only"}},
            "state": {"chronological": "none" if not model.needs_dataset else "recomputed from the history it is given: a regime filter, a simulated account or dynamic weights read the data up to each decision, and the bundle binds a view that ends there",
                      "information_cutoff": info.get("fit_cutoff"), "label_end_max": info.get("label_end_max")},
            "versions": _versions(), "adapters": [dict(a) for a in (adapters or [])]}
        return cls(header, model.to_bytes())

    # -- persistence
    def to_bytes(self) -> bytes:
        head = json.dumps(self.header, sort_keys=True, default=str).encode()
        body = struct.pack(">I", len(head)) + head + self.blob
        return MAGIC + hashlib.sha256(body).digest() + body

    @classmethod
    def from_bytes(cls, data: bytes) -> "InferenceBundle":
        if bytes(data[:4]) != MAGIC:
            raise BundleError("this is not an inference bundle (it does not start with the QFB1 header)")
        digest, body = bytes(data[4:36]), bytes(data[36:])
        if len(data) < 41 or hashlib.sha256(body).digest() != digest:
            raise BundleError("the inference bundle is corrupted: its checksum does not match its contents")
        (n,) = struct.unpack(">I", body[:4])
        header = json.loads(body[4:4 + n].decode())
        if int(header.get("bundle_version", 0)) != VERSION:
            raise BundleError(f"this bundle is version {header.get('bundle_version')} and this platform reads version {VERSION}: retrain it or migrate it with the version that wrote it")
        return cls(header, body[4 + n:])

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.to_bytes())
        return p

    @classmethod
    def load(cls, path: str | Path) -> "InferenceBundle":
        return cls.from_bytes(Path(path).read_bytes())

    # -- use
    @property
    def model(self) -> Model:
        if self._model is None:
            self._model = Model.from_bytes(self.blob)
        return self._model

    @property
    def schema(self) -> dict:
        return self.header["schema"]

    def conform(self, X: pd.DataFrame, strict: bool = True) -> tuple[pd.DataFrame, dict]:
        """``(X in the model's schema, report)``. A registered adapter for exactly this drift is applied and named in the report; any drift left over is an error in ``strict`` mode (the default)."""
        report = {"adapters_applied": [], "drift": schema_drift(self.schema, X)}
        if not any(report["drift"][k] for k in ("missing", "extra", "reordered", "dtype_changes")):
            return X, report
        here = schema_hash(list(X.columns), [kind_of(X[c].dtype) for c in X.columns])
        for adapter in self.header.get("adapters", []):
            if adapter.get("from_hash") == here:
                X = adapt(X, {**adapter, "order": adapter.get("order") or self.schema["features"]})
                report["adapters_applied"].append({"version": adapter.get("version"), "from_hash": here, "description": adapter.get("description", "")})
                report["drift"] = schema_drift(self.schema, X)
                break
        left = report["drift"]
        if strict and any(left[k] for k in ("missing", "extra", "reordered", "dtype_changes")):
            raise SchemaDriftError(f"the data does not have the schema this {self.header['kind']} model was fitted on ({self.schema['hash']}): {drift_sentence(left)}; register a versioned adapter for this change or rebuild the features")
        return X, report

    def predict(self, X: pd.DataFrame, ds=None, strict: bool = True) -> np.ndarray:
        """The model's scores for the rows of ``X``. A model that reads history is bound to a view of ``ds`` that ends on the last date of ``X`` (``ds`` is required for it)."""
        X, _ = self.conform(X, strict)
        model = self.model
        if model.needs_dataset:
            if ds is None:
                raise ValueError(f"the {self.header['kind']} model reads history: give the dataset (its features and prices up to the decision date)")
            model.bind(CausalView(ds, X.index.get_level_values(0).max()))
        return np.asarray(model.predict(X[self.schema["features"]]), dtype=float)

    def reproduces(self, X: pd.DataFrame, saved: pd.Series | np.ndarray, ds=None, tol: float = 1e-6) -> dict:
        """Whether the bundle's predictions for ``X`` equal the ones the evaluation saved, to ``tol``."""
        got = self.predict(X, ds)
        want = np.asarray(saved.reindex(X.index) if isinstance(saved, pd.Series) else saved, dtype=float)
        both = np.isfinite(got) & np.isfinite(want)
        worst = float(np.max(np.abs(got[both] - want[both]))) if both.any() else float("nan")
        return {"max_abs_difference": worst, "rows": int(len(X)), "reproduced": bool(np.array_equal(np.isfinite(got), np.isfinite(want)) and (not both.any() or worst <= tol))}


class BundleError(ModelArtifactError):
    """A bundle that is not one, is damaged, or was written by another version."""


def _versions() -> dict:
    try:
        from .model_report import versions

        return versions()
    except Exception:                                                                              # noqa: BLE001 - the versions are a courtesy; a bundle is still valid without them
        return {}
