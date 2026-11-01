"""Complete provenance of an experiment: what data, what code, what environment, what configuration produced it (spec v3, P0-24).

The legacy ``datasets.bundle_fingerprint`` hashes the shape of a bundle and its last 250 price rows. It is a fine cache key and a poor identity: an old price, a restated macro value or a changed release date
leaves it unchanged. This module hashes the whole history of every input, in a form that does not depend on the pandas version that held it (values as bytes, dates as UTC nanoseconds), and puts the hashes of
the data, the source tree, the merged configuration, the feature schema, the fold plan and the output contract into one identity. The environment and the hardware are recorded beside it but are not part of the
identity: the same inputs on another machine are the same experiment, and ``reproducible`` says whether the outputs agree within the tolerance that is legitimate for that model on that device.

What a record cannot do is prove that the inputs were right, only that they are the ones recorded. A missing git checkout is stated as ``commit unavailable`` next to the source-tree hash, not invented.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VERSION = 1
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm", "xgboost", "ngboost", "statsmodels", "hmmlearn", "gymnasium", "stable-baselines3", "optuna", "pyarrow")
# exact for the CPU families that are deterministic given a seed; the others are compared with a tolerance that is a property of the model and the device (spec 3.4), not a way to make a test pass
TOLERANCES = {"default": 0.0, "torch_cpu": 1e-5, "torch_gpu": 1e-3, "rl": 1e-4}


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def digest(obj, n: int = 64) -> str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()[:n]


# ------------------------------------------------------------------------------------------------------------------ content hashing
def _column_bytes(values) -> bytes:
    """The bytes of one column, the same on every pandas version: numbers as float64 (NaN canonical), booleans as bytes, dates as UTC nanoseconds, anything else as text."""
    s = pd.Series(values)
    kind = getattr(s.dtype, "kind", "O")
    if kind == "M":
        idx = pd.DatetimeIndex(s)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return np.ascontiguousarray(idx.as_unit("ns").asi8 if hasattr(idx, "as_unit") else idx.asi8).tobytes()
    if kind == "b":
        return s.to_numpy(dtype=bool).astype(np.uint8).tobytes()
    if kind in "fiu":
        x = s.to_numpy(dtype="float64", na_value=np.nan).copy()
        x[np.isnan(x)] = np.nan                                                                    # one bit pattern for every NaN
        return np.ascontiguousarray(x).tobytes()
    return "\x1f".join("" if v is None or (isinstance(v, float) and v != v) else str(v) for v in s.tolist()).encode("utf-8")


def frame_hash(obj) -> dict:
    """The full-content hash of a frame or series: its index (every level), its columns and every value. ``{"sha256", "rows", "columns", "first", "last"}``; ``None`` hashes as an absent input."""
    if obj is None:
        return {"sha256": None, "rows": 0, "columns": 0, "first": None, "last": None}
    if isinstance(obj, pd.Series):
        obj = obj.to_frame("value")
    h = hashlib.sha256()
    h.update(canon([str(c) for c in obj.columns]).encode())
    h.update(canon([str(getattr(t, "kind", "O")) + ":" + str(t) for t in obj.dtypes]).encode())
    index = obj.index
    levels = [index.get_level_values(i) for i in range(index.nlevels)]
    h.update(f"index:{index.nlevels}".encode())
    for level in levels:
        h.update(_column_bytes(level))
    for c in obj.columns:
        h.update(_column_bytes(obj[c]))
    first = str(index[0]) if len(index) else None
    last = str(index[-1]) if len(index) else None
    return {"sha256": h.hexdigest(), "rows": int(len(obj)), "columns": int(obj.shape[1]), "first": first, "last": last}


def mapping_hash(mapping: dict | None) -> str | None:
    return None if mapping is None else digest(mapping, 64)


# ------------------------------------------------------------------------------------------------------------------ data
BUNDLE_INPUTS = ("prices", "returns", "investable", "high", "low", "volume", "macro", "marks")


def input_manifest(bundle) -> dict:
    """The full-history hash of every input a market bundle carries (prices, returns, the investable mask, high, low, volume, every macro series, the closed-market marks), and of the descriptions that say how the
    values became public (``macro_meta``: release lags, qualities), the asset classes and the groups. A changed old price, a restated macro value or a changed publication date changes ``sha256``."""
    parts = {name: frame_hash(getattr(bundle, name, None)) for name in BUNDLE_INPUTS}
    descriptions = {"macro_meta": mapping_hash(dict(getattr(bundle, "macro_meta", {}) or {})), "asset_class": mapping_hash(dict(getattr(bundle, "asset_class", {}) or {})),
                    "group": mapping_hash(dict(getattr(bundle, "group", {}) or {})), "assets": list(getattr(bundle, "assets", []) or [])}
    return {"version": VERSION, "inputs": parts, "descriptions": descriptions, "sha256": digest({"inputs": {k: v["sha256"] for k, v in parts.items()}, "descriptions": descriptions}, 64)}


def dataset_manifest(ds) -> dict:
    """Everything the model could have read, hashed over its whole history: the dataset's prices, returns, investable mask, features and targets, the build-time manifest of the bundle (when the dataset was built
    from one), and the policies that decide which values were eligible (label convention, session policy, point-in-time record)."""
    parts = {"prices": ds.prices, "returns": ds.returns, "investable": ds.investable, "X": ds.X, "Y": ds.Y}
    marks = getattr(ds, "marks", None)
    if marks is not None:
        parts["marks"] = marks
    frames = {name: frame_hash(frame) for name, frame in parts.items()}
    meta = ds.meta or {}
    audit = meta.get("audit") or {}
    policies = {"label_convention": meta.get("label_convention"), "sessions": audit.get("sessions"), "pit": audit.get("pit"), "normalize": meta.get("normalize"), "winsor": meta.get("winsor"),
                "lag": int(ds.lag), "horizon": int(ds.horizon), "target": ds.target, "assets": list(ds.assets), "features": list(ds.features)}
    source = meta.get("input_manifest")
    return {"version": VERSION, "frames": frames, "policies": policies, "bundle": source, "data_hash": digest({"frames": {k: v["sha256"] for k, v in frames.items()}, "policies": policies,
                                                                                                          "bundle": None if source is None else source.get("sha256")}, 64)}


def outcome_hash(ds, first, last) -> str:
    """The hash of the realised returns of every asset over ``[first, last]``: the identity of the outcomes a holdout would reveal, whatever the dataset was called and however it was keyed."""
    window = ds.returns.loc[pd.Timestamp(first):pd.Timestamp(last)]
    return frame_hash(window)["sha256"]


# ------------------------------------------------------------------------------------------------------------------ code
def _tree_files(roots: tuple) -> list[Path]:
    files: list[Path] = []
    for r in roots:
        base = ROOT / r
        if base.is_dir():
            files += [p for p in base.rglob("*") if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".js", ".html") and "__pycache__" not in p.parts]
    return sorted(files)


_TREE_CACHE: dict = {}


def source_tree_hash(roots: tuple = ("src", "config")) -> dict:
    """SHA-256 over the relative path and bytes of every source and configuration file of the platform, with the git commit when the tree is a checkout. An uploaded archive without ``.git`` says
    ``commit unavailable``: the tree hash is then the whole identity of the code."""
    files = _tree_files(roots)
    stamp = (tuple(roots), len(files), max((f.stat().st_mtime_ns for f in files), default=0))
    if stamp in _TREE_CACHE:
        return dict(_TREE_CACHE[stamp])
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    out = {"sha256": h.hexdigest(), "files": len(files), "roots": list(roots), **_git()}
    _TREE_CACHE.clear()
    _TREE_CACHE[stamp] = out
    return dict(out)


def _git() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5, check=False)
        if commit.returncode != 0:
            return {"git_commit": None, "git_dirty": None, "git_note": "commit unavailable: this tree is not a git checkout"}
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
        return {"git_commit": commit.stdout.strip(), "git_dirty": bool(dirty.stdout.strip()), "git_note": None}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None, "git_note": "commit unavailable: git could not be run"}


# ------------------------------------------------------------------------------------------------------------------ environment and hardware
def environment_record() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    torch_info = None
    if "torch" in sys.modules:                                                                      # reported when it is loaded; importing it here would make a base install heavier
        t = sys.modules["torch"]
        try:
            torch_info = {"version": t.__version__, "cuda": bool(t.cuda.is_available()), "threads": int(t.get_num_threads()), "deterministic_algorithms": bool(t.are_deterministic_algorithms_enabled())}
        except Exception:                                                                           # noqa: BLE001 - a courtesy field
            torch_info = {"version": getattr(t, "__version__", None)}
    env = {"python": platform.python_version(), "implementation": platform.python_implementation(), "platform": platform.platform(), "packages": packages, "torch": torch_info,
           "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")}}
    env["sha256"] = digest({k: v for k, v in env.items()}, 64)
    return env


def hardware_record() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = None
    if "torch" in sys.modules:
        t = sys.modules["torch"]
        try:
            if t.cuda.is_available():
                gpu = t.cuda.get_device_name(0)
        except Exception:                                                                           # noqa: BLE001
            gpu = None
    return {"cpu": cpu, "cpu_count": os.cpu_count(), "machine": platform.machine(), "gpu": gpu, "backend": "cuda" if gpu else "cpu"}


def seed_policy(cfg: dict) -> dict:
    return {"seed": int(cfg.get("seed", 0)), "policy": "one integer seed from the configuration seeds every random source the experiment uses (numpy generators, model random_state, torch manual_seed); "
                                                       "a family that cannot be made deterministic on a device says so in its tolerance", "tolerances": TOLERANCES}


# ------------------------------------------------------------------------------------------------------------------ experiment record
def folds_record(train) -> dict:
    folds = [{k: f.get(k) for k in ("fold", "test_start", "test_end", "n_train", "n_test", "fit_cutoff", "label_end_max", "train_rows_hash", "skipped")} for f in getattr(train, "folds", [])]
    return {"n": len(folds), "hash": digest(folds, 64), "folds": folds}


def features_record(ds) -> dict:
    schema = {"features": list(ds.features), "dtypes": [str(t) for t in ds.X[list(ds.features)].dtypes] if len(ds.features) else []}
    return {"names": list(ds.features), "n": len(ds.features), "hash": digest(schema, 64)}


def provenance_record(cfg: dict, ds, train=None, contract: dict | None = None, chash: str | None = None) -> dict:
    """The record of one experiment: data, code, environment, hardware, seeds, the full merged configuration, the features, the folds and the output contract, and the ``identity`` that covers the ones that decide the result."""
    from .config import config_hash

    chash = chash or config_hash(cfg)
    data, code = dataset_manifest(ds), source_tree_hash()
    features, folds = features_record(ds), folds_record(train) if train is not None else {"n": 0, "hash": None, "folds": []}
    contract_hash = digest(contract, 64) if contract else None
    identity = digest({"v": VERSION, "data": data["data_hash"], "code": code["sha256"], "config": chash, "seed": int(cfg.get("seed", 0)), "features": features["hash"], "folds": folds["hash"],
                       "contract": contract_hash}, 64)
    env = environment_record()
    return {"version": VERSION, "identity": identity, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "data": data, "code": code, "environment": env, "hardware": hardware_record(),
            "config": {"hash": chash, "merged": cfg}, "seeds": seed_policy(cfg), "features": features, "folds": folds, "output_contract": {"hash": contract_hash, "contract": contract}}


COMPONENTS = {"data": lambda r: r["data"]["data_hash"], "code": lambda r: r["code"]["sha256"], "config": lambda r: r["config"]["hash"], "seed": lambda r: r["seeds"]["seed"],
              "features": lambda r: r["features"]["hash"], "folds": lambda r: r["folds"]["hash"], "output_contract": lambda r: r["output_contract"]["hash"],
              "environment": lambda r: r["environment"]["sha256"]}


def compare(a: dict, b: dict) -> dict:
    """Which components of two records differ. ``same_identity`` is the identity equality; ``differs`` names the changed components (the environment is listed but is not part of the identity)."""
    differs = [name for name, get in COMPONENTS.items() if get(a) != get(b)]
    return {"same_identity": a["identity"] == b["identity"], "differs": differs, "environment_differs": "environment" in differs}


def family_tolerance(kind: str, backend: str = "cpu") -> float:
    """The tolerance a repeated run of this model family may differ by on this device: exact for the deterministic CPU families, looser for neural and reinforcement learning ones."""
    if kind in ("ppo", "sac", "recurrent_ppo", "rl_ppo", "rl_sac"):
        return TOLERANCES["rl"]
    if kind in ("lstm", "gru", "tcn", "patchtst", "nbeats", "nhits", "tft", "ae", "vae", "gcn", "gat", "mlp"):
        return TOLERANCES["torch_gpu"] if backend == "cuda" else TOLERANCES["torch_cpu"]
    return TOLERANCES["default"]


def reproducible(a: dict, b: dict, predictions_a, predictions_b, kind: str = "ridge") -> dict:
    """Whether two runs reproduce each other: the identities must be equal, and the predictions must agree within the family's tolerance on the device the first ran on."""
    tol = family_tolerance(kind, a.get("hardware", {}).get("backend", "cpu"))
    pa, pb = np.asarray(predictions_a, dtype=float), np.asarray(predictions_b, dtype=float)
    same_shape = pa.shape == pb.shape
    worst = float(np.nanmax(np.abs(pa - pb))) if same_shape and pa.size else float("nan")
    same = compare(a, b)
    ok = bool(same["same_identity"] and same_shape and (worst <= tol or (pa.size == 0)))
    return {"reproduced": ok, "max_abs_difference": worst, "tolerance": tol, "identity": same}


# ------------------------------------------------------------------------------------------------------------------ persistence
def store(registry, experiment_id: str, record: dict) -> int:
    """Append the record to the registry (it is never rewritten) and write ``provenance.json`` beside the experiment's artifacts."""
    with registry._lock, registry._conn() as c:                                                   # noqa: SLF001 - the registry's own tables
        cur = c.execute("INSERT INTO provenance (experiment_id, identity, data_hash, code_hash, environment_hash, config_hash, features_hash, folds_hash, contract_hash, seed, record, created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (experiment_id, record["identity"], record["data"]["data_hash"], record["code"]["sha256"], record["environment"]["sha256"], record["config"]["hash"], record["features"]["hash"],
                         record["folds"]["hash"], record["output_contract"]["hash"], int(record["seeds"]["seed"]), json.dumps(record, default=str), record["created"]))
        seq = int(cur.lastrowid)
    target = registry.experiment_dir(experiment_id) / "provenance.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str))
    os.replace(tmp, target)
    return seq


def load(registry, experiment_id: str) -> dict | None:
    """The latest provenance record of an experiment, or ``None`` for one saved before this record existed (it says ``not recorded``; it is not reconstructed)."""
    with registry._conn() as c:                                                                  # noqa: SLF001
        row = c.execute("SELECT record FROM provenance WHERE experiment_id=? ORDER BY seq DESC LIMIT 1", (experiment_id,)).fetchone()
    return json.loads(row["record"]) if row else None


def summary(record: dict | None) -> dict:
    """The short form that goes into a report: the identity and the hashes it is made of."""
    if record is None:
        return {"status": "not recorded", "note": "this experiment was saved before provenance records existed"}
    return {"status": "recorded", "version": record["version"], "identity": record["identity"], "data_hash": record["data"]["data_hash"], "code_hash": record["code"]["sha256"],
            "environment_hash": record["environment"]["sha256"], "config_hash": record["config"]["hash"], "git_commit": record["code"].get("git_commit"), "git_note": record["code"].get("git_note"),
            "backend": record["hardware"].get("backend")}
