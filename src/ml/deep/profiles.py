"""Resource profiles and a pre-run memory estimate for deep training.

The three profiles choose HOW a run is executed (batch size ceiling, threads, checkpoint frequency, mixed precision, parallelism), never WHAT the model is: widths, depths, lookback, features and data are untouched.
A profile that lowers the batch size changes the optimisation (more steps per epoch), so the result records which values were applied and whether the run stays comparable with one made under another profile.
"""

from __future__ import annotations

from .. import resources

PROFILES = {
    "conservative": {"label": "Conservative (Replit, small CPUs)", "batch_cap": 128, "threads": "min(default, 2)", "checkpoint_every": 1, "amp": False, "max_concurrent_jobs": 1, "loader_workers": 0, "parallel_search": False},
    "balanced": {"label": "Balanced (a typical PC)", "batch_cap": 512, "threads": "default", "checkpoint_every": 1, "amp": False, "max_concurrent_jobs": 1, "loader_workers": 2, "parallel_search": False},
    "performance": {"label": "Performance (a workstation, GPU optional)", "batch_cap": 2048, "threads": "default", "checkpoint_every": 2, "amp": True, "max_concurrent_jobs": 2, "loader_workers": 4, "parallel_search": True},
}


def apply(params: dict, profile: str) -> tuple[dict, dict]:
    """``(params, record)``: ``params`` with the profile's execution settings applied (a setting the caller already gave explicitly is kept) and a record of every value the profile changed."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    prof, out, changed = PROFILES[profile], dict(params), {}
    cap = prof["batch_cap"]
    if int(out.get("batch_size", 512)) > cap:
        changed["batch_size"] = {"from": int(out.get("batch_size", 512)), "to": cap}
        out["batch_size"] = cap
    threads = resources.default_threads()
    want = min(threads, 2) if profile == "conservative" else threads
    if int(out.get("num_threads", 0) or 0) == 0:
        out["num_threads"] = want
        changed["num_threads"] = {"from": 0, "to": want}
    if "checkpoint_every" not in params:
        out["checkpoint_every"] = prof["checkpoint_every"]
    import torch                                                                       # imported when a profile is applied, so that listing profiles stays light

    amp = bool(prof["amp"] and torch.cuda.is_available())
    if amp and "amp" not in params:
        out["amp"] = True
        changed["amp"] = {"from": False, "to": True}
    out["profile"] = profile
    comparable = "batch_size" not in changed
    return out, {"profile": profile, "label": prof["label"], "changed": changed, "comparable_with_unprofiled_run": comparable,
                 "note": "the model, features, lookback and data are unchanged" + ("" if comparable else "; the batch size was capped, so the optimisation differs from a run at the original batch size")}


def estimate(arch: str, n_features: int, lookback: int, width: int, layers: int, blocks: int, batch_size: int, max_rows: int, amp: bool = False, workers: int = 0, heads: int = 1) -> dict:
    """A memory range for one training run. The point estimate is ``resources.deep_mb`` (rows, activations, parameters with optimizer state, framework baseline); the range is a deliberately wide band around
    it (0.7 to 1.8 times: activations and allocator overhead are not exactly predictable) and each data-loader worker adds a process baseline. NOT a measurement of peak memory."""
    depth = max(1, layers) * (blocks if arch in ("nbeats", "nhits") else 1)
    point = resources.deep_mb(arch, n_features, lookback, width, depth, batch_size, max_rows, heads)
    if amp:
        point *= 0.85
    point += workers * 250.0
    out = {"point_mb": round(point), "low_mb": round(point * 0.7), "high_mb": round(point * 1.8), "basis": "heuristic estimate with a wide band; measure the real peak with the benchmark", "workers": workers}
    out["check"] = resources.preflight(point * 1.0, "this training")
    high = resources.preflight(point * 1.8, "this training at the upper end of its range")
    out["upper_check"] = high
    out["level"] = "stop" if out["check"]["level"] == "stop" else "warn" if (out["check"]["level"] == "warn" or high["level"] in ("stop", "warn")) else "ok"
    return out


def suggest_smaller(batch_size: int) -> dict:
    """What to try after a memory failure: half the batch size, with gradient accumulation to keep the effective batch (the optimiser sees the same number of rows per step)."""
    half = max(8, batch_size // 2)
    return {"batch_size": half, "accumulation_steps": max(1, round(batch_size / half)), "note": "the same effective batch with half the memory per step; the run restarts from its last checkpoint"}
