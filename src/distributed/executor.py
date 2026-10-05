"""Parallel execution of independent experiments (Generation 4, Priority 15).

A research grid is embarrassingly parallel: every candidate is a pure function of its parameters and of
read-only data. This module runs such a grid on one of several backends and guarantees the one property
that matters for reproducibility: **the results come back in task order, identical to a serial run**.

Backends: ``serial`` (the reference), ``joblib`` (process pool; the default), and optional ``dask`` and
``ray`` adapters that import lazily and raise a clear error when the library is not installed.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Iterable, Sequence

BACKENDS = ("serial", "joblib", "dask", "ray")


def task_seed(params: Any, base_seed: int = 0) -> int:
    """A deterministic 32-bit seed derived from a task's parameters, independent of scheduling order."""
    payload = json.dumps(params, sort_keys=True, default=str).encode()
    return (int(hashlib.sha256(payload).hexdigest()[:8], 16) + base_seed) % (2 ** 32)


def available_backends() -> list[str]:
    found = ["serial"]
    for name in ("joblib", "dask", "ray"):
        try:
            __import__("dask.distributed" if name == "dask" else name)
            found.append(name)
        except ImportError:
            continue
    return found


def _chunks(items: Sequence, size: int) -> list[list]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def _run_chunk(fn: Callable, chunk: list) -> list:
    return [fn(task) for task in chunk]


def run_tasks(fn: Callable[[Any], Any], tasks: Iterable[Any], backend: str = "joblib", n_jobs: int = -1,
              chunk_size: int = 8) -> list:
    """Apply ``fn`` to every task and return the results in task order."""
    items = list(tasks)
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend '{backend}'; choose one of {BACKENDS}")
    if backend == "serial" or len(items) <= 1:
        return [fn(task) for task in items]
    chunks = _chunks(items, max(1, chunk_size))
    if backend == "joblib":
        from joblib import Parallel, delayed

        done = Parallel(n_jobs=n_jobs)(delayed(_run_chunk)(fn, chunk) for chunk in chunks)
    elif backend == "dask":
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError as error:                              # pragma: no cover - optional dependency
            raise RuntimeError("the dask backend needs `pip install dask distributed`") from error
        workers = None if n_jobs in (-1, None) else n_jobs
        with LocalCluster(n_workers=workers, threads_per_worker=1, processes=True, dashboard_address=None) as cluster, Client(cluster) as client:
            futures = [client.submit(_run_chunk, fn, chunk, pure=False) for chunk in chunks]
            done = client.gather(futures)
    else:                                                          # ray
        try:
            import ray
        except ImportError as error:                              # pragma: no cover - optional dependency
            raise RuntimeError("the ray backend needs `pip install ray`") from error
        started = not ray.is_initialized()
        if started:
            ray.init(num_cpus=None if n_jobs in (-1, None) else n_jobs, include_dashboard=False, ignore_reinit_error=True)
        try:
            remote = ray.remote(_run_chunk)
            done = ray.get([remote.remote(fn, chunk) for chunk in chunks])
        finally:
            if started:
                ray.shutdown()
    return [result for chunk in done for result in chunk]


def timed(fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
