"""Benchmarking and profiling: how long things take, how much memory they use, how they scale, and where the time goes.

* ``profile_call``: wall time, peak Python memory (``tracemalloc``) and the ten most expensive functions (``cProfile``) of one call.
* ``benchmark_models``: every registered model's ``score`` time and peak memory on a bundle, with the number of forecast cells it produced, as a table.
* ``scaling_exponent``: run a function at several problem sizes and fit ``time ~ n^b`` in log-log space; ``b`` close to 1 is linear, 2 quadratic. A regression in ``b`` after a code change is a performance bug even when absolute times look fine.
* ``Timer`` is a context manager for ad-hoc sections.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
import tracemalloc
from typing import Callable

import numpy as np
import pandas as pd


class Timer:
    def __init__(self):
        self.seconds = float("nan")

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0


def profile_call(fn: Callable, *args, top: int = 10, **kwargs) -> dict:
    """Run ``fn`` once under ``cProfile`` and ``tracemalloc``. Returns ``result``, ``seconds``, ``peak_mb`` and a table of the ``top`` functions by cumulative time."""
    prof = cProfile.Profile()
    tracemalloc.start()
    t0 = time.perf_counter()
    prof.enable()
    try:
        result = fn(*args, **kwargs)
    finally:
        prof.disable()
        seconds = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream).sort_stats("cumulative")
    rows = []
    for (filename, line, name), (cc, nc, tt, ct, _) in sorted(stats.stats.items(), key=lambda kv: -kv[1][3])[:top]:
        rows.append({"function": f"{filename.split('/')[-1]}:{line}({name})", "calls": nc, "total_s": tt, "cumulative_s": ct})
    return {"result": result, "seconds": seconds, "peak_mb": peak / 1e6, "top": pd.DataFrame(rows)}


def benchmark_models(bundle, names: list[str] | None = None, repeats: int = 1) -> pd.DataFrame:
    """Time ``model.score(bundle)`` for each model (best of ``repeats``), record peak memory and how many (date, asset) cells received a score. Models that raise are listed with the error."""
    from ..framework import MODELS, load_library

    load_library()
    names = names or [e.name for e in MODELS.entries() if not e.name.startswith("test_")]
    rows = []
    for name in names:
        best, peak, cells, error = np.inf, 0.0, 0, ""
        try:
            for _ in range(repeats):
                model = MODELS.create(name)
                tracemalloc.start()
                t0 = time.perf_counter()
                score = model.score(bundle)
                best = min(best, time.perf_counter() - t0)
                peak = max(peak, tracemalloc.get_traced_memory()[1] / 1e6)
                tracemalloc.stop()
                cells = int(score.notna().to_numpy().sum())
        except Exception as exc:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            error = f"{type(exc).__name__}: {str(exc)[:80]}"
        rows.append({"model": name, "seconds": best if np.isfinite(best) else np.nan, "peak_mb": peak, "scored_cells": cells, "error": error})
    return pd.DataFrame(rows).set_index("model").sort_values("seconds", ascending=False)


def scaling_exponent(fn: Callable[[int], object], sizes, repeats: int = 3) -> dict:
    """Time ``fn(n)`` at each size (best of ``repeats``) and fit ``log t = a + b log n``. Returns the exponent ``b``, the R-squared of the fit and the timing table."""
    sizes = list(sizes)
    times = []
    for n in sizes:
        best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(n)
            best = min(best, time.perf_counter() - t0)
        times.append(best)
    x, y = np.log(sizes), np.log(times)
    b, a = np.polyfit(x, y, 1)
    fit = a + b * x
    r2 = 1.0 - float(((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if len(sizes) > 2 else float("nan")
    return {"exponent": float(b), "r2": r2, "table": pd.DataFrame({"n": sizes, "seconds": times})}
