"""Saved, reproducible research runs for the Options and Futures workspaces.

An experiment record keeps what is needed to repeat a run and to judge two runs against each other: the request that produced it (endpoint and parameters, so it can be sent again), the instrument universe,
the data provenance the server reported, the model and strategy settings, the backtest assumptions, the headline metrics, the software version and a timestamp. Two hashes are stored: ``config_hash`` (the
request alone: same hash, same question) and ``result_hash`` (the metrics: same config and result hash on a rerun means the run reproduced; the same config with another result hash means the data or the code
changed). Records are JSON files under ``data/user/experiments/``; nothing is sent anywhere.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

KINDS = ("options_calibration", "options_surface", "options_strategy", "options_walkforward", "futures_backtest", "futures_spread", "futures_research", "other")
ID = re.compile(r"^[A-Za-z0-9_-]{6,40}$")
MAX_METRICS = 60


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj) -> str:
    return hashlib.sha256(_canon(obj).encode()).hexdigest()[:16]


def software_version(root: Path) -> dict:
    out = {"python": __import__("platform").python_version()}
    for pkg in ("numpy", "pandas", "scipy", "statsmodels"):
        try:
            out[pkg] = __import__(pkg).__version__
        except Exception:                                                                  # a missing package is simply not listed
            pass
    try:
        out["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        out["git"] = None
    return out


def flat_metrics(metrics: dict) -> dict[str, float]:
    """Numeric leaves of a result summary, flattened to ``a.b.c`` keys (capped), so two runs can be set side by side."""
    out: dict[str, float] = {}

    def walk(prefix, v):
        if len(out) >= MAX_METRICS:
            return
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
            out[prefix] = float(v)
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}" if prefix else str(k), x)
    walk("", metrics)
    return out


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.dir = self.root / "data" / "user" / "experiments"

    def _path(self, eid: str) -> Path:
        if not ID.match(eid or ""):
            raise ValueError("invalid experiment id")
        return self.dir / f"{eid}.json"

    def save(self, kind: str, name: str, request: dict, metrics: dict, provenance: dict | None = None, assumptions: list | None = None, universe: list | None = None, notes: str = "") -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {list(KINDS)}")
        name = (name or "").strip()
        if not name or len(name) > 120:
            raise ValueError("give the experiment a name of 1 to 120 characters")
        if not isinstance(request, dict) or not request.get("path") or not str(request["path"]).startswith("/api/"):
            raise ValueError("request must name the endpoint (path) and its parameters, so the run can be repeated")
        if not isinstance(metrics, dict) or not flat_metrics(metrics):
            raise ValueError("metrics must contain at least one number")
        prov = provenance or {}
        rec = {"id": uuid.uuid4().hex[:12], "name": name, "kind": kind, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "request": request, "universe": list(universe or []),
               "provenance": {k: prov.get(k) for k in ("kind", "source", "freshness", "instrument", "as_of", "assumptions", "warnings") if k in prov}, "assumptions": list(assumptions or []),
               "metrics": flat_metrics(metrics), "notes": str(notes)[:2000], "software": software_version(self.root)}
        rec["config_hash"], rec["result_hash"] = digest(request), digest({k: round(v, 8) for k, v in rec["metrics"].items()})
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path(rec["id"]).write_text(json.dumps(rec, indent=1, sort_keys=True))
        return rec

    def get(self, eid: str) -> dict:
        p = self._path(eid)
        if not p.is_file():
            raise KeyError(eid)
        return json.loads(p.read_text())

    def list(self, kind: str | None = None) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.json")) if self.dir.is_dir() else []:
            try:
                r = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if kind and r.get("kind") != kind:
                continue
            out.append({k: r.get(k) for k in ("id", "name", "kind", "created_at", "config_hash", "result_hash", "provenance")})
        return sorted(out, key=lambda r: r["created_at"], reverse=True)

    def delete(self, eid: str) -> None:
        p = self._path(eid)
        if not p.is_file():
            raise KeyError(eid)
        p.unlink()

    def check_repeat(self, eid: str, metrics: dict, rel: float = 1e-6) -> dict:
        """Judge a repeat of a saved run: the repeat's metrics against the stored ones. ``reproduced`` is true if every stored number is matched to ``rel`` (relative; absolute below 1e-9) and no number is new or
        missing. The two hashes of the repeat are returned so a changed result can be told apart from a changed request: the request is the stored one, so only the data or the code can have changed."""
        rec = self.get(eid)
        new = flat_metrics(metrics)
        old = rec["metrics"]
        diffs = []
        for k in sorted(set(old) | set(new)):
            a, b = old.get(k), new.get(k)
            if a is None or b is None:
                diffs.append({"metric": k, "stored": a, "repeat": b})
            elif abs(a - b) > max(1e-9, rel * max(abs(a), abs(b))):
                diffs.append({"metric": k, "stored": a, "repeat": b, "change": b - a})
        return {"id": eid, "reproduced": not diffs, "differences": diffs[:40], "n_metrics": len(old), "stored_result_hash": rec["result_hash"], "repeat_result_hash": digest({k: round(v, 8) for k, v in new.items()}),
                "software_then": rec["software"], "software_now": software_version(self.root),
                "reading": "the repeat matches the saved run" if not diffs else "the repeat differs from the saved run: the request is the same, so the data, the random seed or the code changed (compare the software versions)"}

    def compare(self, ids: list[str]) -> dict:
        if not 2 <= len(ids) <= 8:
            raise ValueError("choose 2 to 8 experiments to compare")
        recs = [self.get(i) for i in ids]
        names = sorted({k for r in recs for k in r["metrics"]})
        rows = [{"metric": k, "values": [r["metrics"].get(k) for r in recs]} for k in names]
        flat = [{**{f"param.{k}": v for k, v in _flat_request(r["request"]).items()}} for r in recs]
        keys = sorted({k for f in flat for k in f})
        diff = [{"setting": k, "values": [f.get(k) for f in flat]} for k in keys if len({_canon(f.get(k)) for f in flat}) > 1]
        warn = []
        if len({r["config_hash"] for r in recs}) == 1:
            warn.append("same request: any difference in the metrics comes from the data or the code, not the settings" if len({r["result_hash"] for r in recs}) > 1 else "same request and same result: the run reproduced")
        if len({r["kind"] for r in recs}) > 1:
            warn.append("the experiments are of different kinds: metrics with the same name may not mean the same thing")
        kinds = [(r["provenance"] or {}).get("kind") for r in recs]
        if len(set(kinds)) > 1:
            warn.append(f"the data differ in kind ({', '.join(str(k) for k in kinds)}): a comparison of simulated with observed data says little")
        return {"experiments": [{k: r[k] for k in ("id", "name", "kind", "created_at", "config_hash", "result_hash", "software", "provenance")} for r in recs], "metrics": rows, "setting_differences": diff, "warnings": warn}


def _flat_request(req: dict) -> dict:
    out = {"path": req.get("path")}
    for part in ("query", "body"):
        v = req.get(part)
        if isinstance(v, dict):
            for k, x in v.items():
                out[f"{part}.{k}"] = x
    return out
