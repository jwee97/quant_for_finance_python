"""HTTP routes of the Training Monitor and the resource-aware start of a deep training (``ml.deep.runs``, ``ml.deep.profiles``).

``GET  /api/training/runs``       every run on disk with its status derived from its files (running, completed, stopped, paused, interrupted, orphaned), epoch, best score, whether it can be resumed
``GET  /api/training/run``        one run's metrics (``since`` returns only newer epochs), events, checkpoints and the request that started it
``POST /api/training/control``    ask a running fit to ``stop`` or ``pause`` after the current epoch's checkpoint, or ``clear`` the request
``POST /api/training/resume``     start the saved request again with ``resume`` set (never automatic: this route is the user's decision); ``restart`` starts it as a NEW run; ``warm`` as a new run from saved weights
``POST /api/training/archive``    move a run out of the list (its files are kept under ``_archive``)
``POST /api/training/preflight``  the memory range, the checks and the profile's effect for a configuration
``GET  /api/training/profiles``   the three resource profiles and this machine's resources
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..ml import resources
from ..ml.deep import context as train_context
from ..ml.deep import profiles, runs
from ..ml.param_schema import DEEP_KINDS
from .api import ApiError, _number


def prepare(root, over: dict, body: dict, path: str, label: str) -> dict | None:
    """For a deep model: choose the run folder, apply a resource profile, check the memory estimate and save the request so that the run can be resumed. Returns ``None`` for any other model. The folder and the resume
    flag are NOT put in the experiment's parameters (they would change its fingerprint)."""
    model = over.get("model") or {}
    if model.get("kind") not in DEEP_KINDS or body.get("checkpointing") is False:
        return None
    params = model["params"]
    run_id = str(body.get("run_id") or uuid.uuid4().hex[:10])
    if not runs.ID.match(run_id):
        raise ApiError("run_id is 3 to 80 letters, digits, '.', '_' or '-'")
    profile = str(body.get("profile") or params.get("profile") or "")
    profile = "" if profile == "none" else profile
    record = None
    if profile:
        try:
            model["params"], record = profiles.apply(params, profile)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        params = model["params"]
    width = int(params.get("width", params.get("hidden", 32)))
    est = profiles.estimate(model["kind"], 30, int(params.get("lookback", 20)), width, int(params.get("layers", 2)), int(params.get("blocks", 3)), int(params.get("batch_size", 512)), int(params.get("max_rows", 60000)),
                            bool(params.get("amp")), int((record or {}).get("changed", {}).get("loader_workers", {}).get("to", 0)))
    if est["level"] == "stop" and not body.get("force"):
        raise ApiError(est["check"]["message"] + " (send force: true to run it anyway)")
    folder = runs.base(root) / run_id
    resume = bool(body.get("resume"))
    if folder.exists() and not resume and any(folder.rglob("last.pt")):
        raise ApiError(f"run {run_id} already has checkpoints: resume it, restart it as a new run, or archive it")
    folder.mkdir(parents=True, exist_ok=True)
    runs.control_clear(folder)                                                      # a new start or an explicit resume ends an earlier stop or pause request
    req = folder / "request.json"
    if not req.exists():
        req.write_text(json.dumps({"path": path, "label": label, "body": {k: v for k, v in body.items() if k not in ("resume", "run_id", "warm_start")}, "run_id": run_id, "created_at": time.time(), "profile": record, "preflight": est}, default=str))
    return {"run_id": run_id, "dir": str(folder), "resume": resume, "warm": bool(body.get("warm_start")), "profile": record, "preflight": est}


class TrainingApi:
    def __init__(self, app, ml):
        self.app = app
        self.ml = ml
        self.root = Path(app.root)

    def get_routes(self) -> dict:
        return {"/api/training/runs": self.runs, "/api/training/run": self.run, "/api/training/profiles": self.profiles}

    def post_routes(self) -> dict:
        return {"/api/training/control": self.control, "/api/training/resume": self.resume, "/api/training/archive": self.archive, "/api/training/preflight": self.preflight}

    def runs(self, q: dict) -> dict:
        return {"runs": runs.list_runs(self.root), "disk": runs.disk_usage(self.root), "machine": resources.machine(), "storage_note": self._storage_note()}

    def _storage_note(self) -> str | None:
        if resources.on_replit():
            return "On Replit the workspace disk persists, but a deployment's storage may be replaced on republish: checkpoints written there can be lost. Copy data/user/training to durable storage for long runs."
        return None

    def run(self, q: dict) -> dict:
        try:
            return runs.detail(self.root, str(q.get("id") or ""), q.get("fit") or None, int(_number(q, "since", 0, 0, 10 ** 6, integer=True)))
        except KeyError:
            raise ApiError("unknown training run") from None
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    def control(self, body: dict) -> dict:
        try:
            return runs.control(self.root, str(body.get("id") or ""), str(body.get("action") or ""), body.get("fit") or None)
        except KeyError:
            raise ApiError("unknown training run") from None
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    def archive(self, body: dict) -> dict:
        try:
            return runs.archive(self.root, str(body.get("id") or ""))
        except KeyError:
            raise ApiError("unknown training run") from None
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    def profiles(self, q: dict) -> dict:
        return {"profiles": profiles.PROFILES, "machine": resources.machine()}

    def preflight(self, body: dict) -> dict:
        try:
            est = profiles.estimate(str(body.get("arch") or "nbeats"), int(_number(body, "n_features", 30, 1, 5000, integer=True)), int(_number(body, "lookback", 20, 2, 2000, integer=True)), int(_number(body, "width", 64, 4, 4096, integer=True)),
                                    int(_number(body, "layers", 2, 1, 64, integer=True)), int(_number(body, "blocks", 3, 1, 64, integer=True)), int(_number(body, "batch_size", 512, 1, 100_000, integer=True)),
                                    int(_number(body, "max_rows", 60000, 100, 10_000_000, integer=True)), bool(body.get("amp")), int(_number(body, "workers", 0, 0, 64, integer=True)))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        out = {"estimate": est, "suggestion": profiles.suggest_smaller(int(_number(body, "batch_size", 512, 1, 100_000, integer=True))) if est["level"] != "ok" else None}
        if body.get("profile") and body.get("profile") != "none":
            try:
                out["profile"] = profiles.apply({"batch_size": int(_number(body, "batch_size", 512, 1, 100_000, integer=True))}, str(body["profile"]))[1]
            except ValueError as exc:
                raise ApiError(str(exc)) from None
        return out

    def resume(self, body: dict) -> dict:
        """Submit the saved request again. ``mode``: ``resume`` (the same run continues from its newest valid checkpoint), ``warm`` (the same run continues from saved WEIGHTS when the data window changed: optimizer and
        counters start fresh, reported as ``warm_start``) or ``restart`` (a new run from scratch). Never called by the server on its own."""
        run_id, mode = str(body.get("id") or ""), str(body.get("mode") or "resume")
        if mode not in ("resume", "restart", "warm"):
            raise ApiError("mode must be resume, restart or warm")
        try:
            d = runs.detail(self.root, run_id)
        except (KeyError, ValueError):
            raise ApiError("unknown training run") from None
        req = d["request"]
        if not req:
            raise ApiError("this run has no saved request, so it cannot be restarted from here: start it again from the Train view")
        if d["summary"]["status"] == "running":
            raise ApiError("this run is still running: stop or pause it first")
        new = dict(req["body"])
        if mode == "restart":
            new.pop("run_id", None)
            new["resume"] = False
        else:
            new.update(run_id=run_id, resume=True, warm_start=(mode == "warm"))
        return self.ml.train(new)
