"""What this machine can actually give a run, and a stop before the host kills the app.

A container (a Replit workspace or deployment, Docker, a small VM) shows the program the *host's* CPUs and memory, while a quota limits what it may use: ``os.cpu_count()`` says 32 and the quota allows one, and a
library that starts 32 threads fights itself for that one CPU (training slows to a crawl, and every thread brings its own buffers). The memory limit is the same: ``free`` shows the host, the kernel kills the
process at the cgroup's limit, and from the page that looks like the app silently vanishing halfway through a training.

This module reads the limits the kernel will enforce (cgroup v2 and v1), picks a sensible number of threads, estimates roughly what a deep model will need, and offers :func:`watchdog`: called while a run goes on,
it raises :class:`ResourceLimit` *before* the kernel does, so the job ends with a message that says what to change and the epochs it had finished stay in the training history.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

WATCH_FRACTION = 0.90               # stop a run when the process holds this share of the memory it may use
WARN_FRACTION = 0.60                # warn before a run whose estimate is this share of what is free
_GAP = 1.0                          # seconds between two reads of the memory counters
_last = {"at": 0.0}


class ResourceLimit(BaseException):
    """The run is using more memory than this machine allows. A ``BaseException`` (like the job's cancellation) so that a model's own ``except ValueError`` or ``except Exception`` cannot treat it as a fold to skip;
    the job runner catches it and ends the job with its message."""


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return None


def _number(text: str | None) -> float | None:
    if text is None or text in ("max", ""):
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return v if v < 2 ** 60 else None                                            # cgroup v1 reports "no limit" as a huge number


# ---------------------------------------------------------------------------------------------------- the machine
def cpu_limit() -> float:
    """CPUs this process may use: the cgroup quota when there is one, else the CPUs it is allowed to run on, else every CPU."""
    quota = _read("/sys/fs/cgroup/cpu.max")
    if quota:
        parts = quota.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                return max(0.1, float(parts[0]) / float(parts[1]))
            except (ValueError, ZeroDivisionError):
                pass
    q1, p1 = _number(_read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")), _number(_read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if q1 and p1 and q1 > 0:
        return max(0.1, q1 / p1)
    try:
        return float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return float(os.cpu_count() or 1)


def memory_limit_mb() -> float | None:
    """The memory the kernel lets this process use (cgroup), or the machine's own when it is smaller or there is no cgroup."""
    limits = []
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        v = _number(_read(path))
        if v:
            limits.append(v / 2 ** 20)
    meminfo = _read("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                limits.append(float(line.split()[1]) / 1024.0)
                break
    return min(limits) if limits else None


def memory_used_mb() -> float | None:
    """What the cgroup counts as in use (page cache included, as the kernel's own accounting does), else this process's resident set."""
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        v = _number(_read(path))
        if v:
            return v / 2 ** 20
    return rss_mb()


def rss_mb() -> float | None:
    status = _read("/proc/self/status")
    if status:
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, ValueError):
        return None


def on_replit() -> bool:
    return any(k in os.environ for k in ("REPL_ID", "REPLIT_DEPLOYMENT", "REPL_SLUG", "REPLIT_DB_URL"))


def default_threads() -> int:
    """Threads worth starting for numerical work: the CPU quota rounded up (a quota of 0.5 still needs one), never more than the CPUs the host shows, at least one."""
    return max(1, min(int(math.ceil(cpu_limit())), os.cpu_count() or 1))


def apply_thread_limits(torch_module=None) -> int:
    """Tell the numerical libraries how many threads to use, unless the person already did (an environment variable is respected). Call before numpy and torch start their pools where possible (the app does, at
    start-up); torch's own setting can still be changed afterwards, which is done here when torch is given."""
    n = default_threads()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, str(n))
    if torch_module is not None:
        try:
            torch_module.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", n)))
        except (RuntimeError, ValueError):
            pass
    return n


def machine() -> dict:
    limit, used = memory_limit_mb(), memory_used_mb()
    cpus = cpu_limit()
    notes = []
    if os.cpu_count() and cpus + 0.01 < os.cpu_count() / 2:
        notes.append(f"The host shows {os.cpu_count()} CPUs but this process may use about {cpus:g}: the app starts {default_threads()} worker thread(s) for numerical work instead of one per host CPU.")
    if limit is not None and used is not None and used > WATCH_FRACTION * limit:
        notes.append("Memory is almost full already: a model of any size is likely to be killed. Restart the app, or free memory, before training.")
    if on_replit():
        notes.append("This looks like Replit. Its workspace and Autoscale deployments have little memory and a fraction of a CPU: train big networks on a bigger machine (see docs/deployment.md) and use this one to read the results.")
    return {"cpus_visible": os.cpu_count(), "cpu_limit": round(cpus, 2), "threads": default_threads(), "memory_limit_mb": None if limit is None else round(limit), "memory_used_mb": None if used is None else round(used),
            "process_rss_mb": None if rss_mb() is None else round(rss_mb()), "memory_free_mb": None if limit is None or used is None else round(max(0.0, limit - used)), "replit": on_replit(), "notes": notes}


# ---------------------------------------------------------------------------------------------------- a stop before the kernel's
def watchdog(fraction: float = WATCH_FRACTION, force: bool = False) -> None:
    """Raise :class:`ResourceLimit` when the process is about to hit its memory limit. Cheap (a file read at most once a second), safe to call often, and silent where the limit cannot be known."""
    now = time.time()
    if not force and now - _last["at"] < _GAP:
        return
    _last["at"] = now
    limit, used = memory_limit_mb(), memory_used_mb()
    if limit and used and used > fraction * limit:
        raise ResourceLimit(f"Stopped before the machine killed the app: it is using {used:,.0f} MB of the {limit:,.0f} MB it may use. The epochs finished so far are kept in the training history. "
                            "Train on less (a smaller max_rows, hidden size, lookback or batch size, fewer assets or features), run fewer things at once, or use a machine with more memory (docs/deployment.md).")


# ---------------------------------------------------------------------------------------------------- a rough price before a run
def dataset_mb(n_assets: int, n_days: int, n_features: int, copies: float = 4.0) -> float:
    """The feature table of the machine-learning lab: assets x days x features of 8 bytes, and the few copies pandas makes while it builds, aligns and splits it."""
    return n_assets * n_days * n_features * 8.0 * copies / 2 ** 20


def deep_mb(arch: str, n_features: int, lookback: int, hidden: int, layers: int, batch_size: int, max_rows: int, heads: int = 1) -> float:
    """A rough estimate of what training one network needs on top of the dataset: the rows kept for training and validation (float32), the windows of one batch and the activations that back-propagation keeps
    (about four tensors per layer, more for attention), the parameters with their optimiser state, and the framework's own overhead. An estimate to compare with the free memory, not a promise."""
    rows = max_rows * 1.34 * (n_features + 8) * 4 / 2 ** 20
    per_layer = batch_size * lookback * hidden * 4 * (4 + (2 * heads if arch in ("patchtst", "tft", "itransformer") else 0)) * max(1, layers) / 2 ** 20
    windows = batch_size * lookback * n_features * 4 * 3 / 2 ** 20
    params = (n_features * hidden * 4 + hidden * hidden * 8 * max(1, layers)) * 4 * 4 / 2 ** 20
    return rows + per_layer + windows + params + 350.0                                    # the framework's baseline


def preflight(estimate_mb: float, what: str = "this run") -> dict:
    """Compare an estimate with the memory that is free. Returns ``{"estimate_mb", "free_mb", "limit_mb", "level": ok|warn|stop, "message"}``; ``stop`` when the run would not fit, ``warn`` when it would be tight."""
    limit, used = memory_limit_mb(), memory_used_mb()
    free = None if limit is None or used is None else max(0.0, limit - used)
    level, message = "ok", ""
    if free is not None:
        if estimate_mb > free * 0.95:
            level = "stop"
            message = (f"{what} needs roughly {estimate_mb:,.0f} MB and about {free:,.0f} MB are free (limit {limit:,.0f} MB). It would be killed partway. Choose fewer assets, a shorter history, fewer feature groups, "
                       "a smaller max_rows, hidden size, lookback or batch size, or use a machine with more memory (docs/deployment.md).")
        elif estimate_mb > free * WARN_FRACTION:
            level = "warn"
            message = f"{what} needs roughly {estimate_mb:,.0f} MB of the {free:,.0f} MB that are free: it may be killed if anything else grows meanwhile. Consider trimming it."
    return {"estimate_mb": round(estimate_mb), "free_mb": None if free is None else round(free), "limit_mb": None if limit is None else round(limit), "level": level, "message": message}
