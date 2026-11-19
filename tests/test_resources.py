"""Running big models on a small machine (src/ml/resources.py): the limits a container really imposes (read from the cgroup, not from the host), how many threads that allows, a price estimate before a run, a refusal
before an experiment that would not fit is queued, and a stop with a message, with the epochs kept, before the kernel's silent kill."""

from __future__ import annotations

import os
import time

import pytest

from src.ml import live, resources
from src.utils.config import load_config
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch, _wait


def _fake(monkeypatch, files: dict, cpus: int = 32):
    monkeypatch.setattr(resources, "_read", lambda path: files.get(path))
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(cpus)), raising=False)


def test_the_cpu_quota_not_the_hosts_cpu_count_decides_the_threads(monkeypatch):
    _fake(monkeypatch, {"/sys/fs/cgroup/cpu.max": "50000 100000"})
    assert resources.cpu_limit() == pytest.approx(0.5) and resources.default_threads() == 1, "half a CPU still needs one thread, not thirty-two"
    _fake(monkeypatch, {"/sys/fs/cgroup/cpu.max": "400000 100000"})
    assert resources.cpu_limit() == pytest.approx(4.0) and resources.default_threads() == 4
    _fake(monkeypatch, {"/sys/fs/cgroup/cpu.max": "max 100000"}, cpus=8)
    assert resources.cpu_limit() == 8.0 and resources.default_threads() == 8, "no quota: every CPU the process may run on"
    _fake(monkeypatch, {"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "200000", "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"})
    assert resources.cpu_limit() == pytest.approx(2.0), "cgroup v1"
    _fake(monkeypatch, {}, cpus=64)
    assert resources.default_threads() == 64
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    _fake(monkeypatch, {"/sys/fs/cgroup/cpu.max": "100000 100000"})
    assert resources.apply_thread_limits() == 1 and os.environ["OMP_NUM_THREADS"] == "1" and os.environ["OPENBLAS_NUM_THREADS"] == "1"
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    resources.apply_thread_limits()
    assert os.environ["OMP_NUM_THREADS"] == "3", "a value the person set is not overridden"


def test_the_memory_limit_is_the_smaller_of_the_cgroup_and_the_machine(monkeypatch):
    meminfo = "MemTotal:       16000000 kB\nMemFree: 1 kB\n"
    _fake(monkeypatch, {"/sys/fs/cgroup/memory.max": str(1024 * 2 ** 20), "/proc/meminfo": meminfo, "/sys/fs/cgroup/memory.current": str(300 * 2 ** 20)})
    assert resources.memory_limit_mb() == pytest.approx(1024) and resources.memory_used_mb() == pytest.approx(300)
    m = resources.machine()
    assert m["memory_limit_mb"] == 1024 and m["memory_free_mb"] == 724 and m["cpus_visible"] == 32
    _fake(monkeypatch, {"/sys/fs/cgroup/memory.max": "max", "/proc/meminfo": meminfo})
    assert resources.memory_limit_mb() == pytest.approx(16000000 / 1024), "no cgroup limit: the machine's own"
    _fake(monkeypatch, {"/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712", "/proc/meminfo": meminfo})
    assert resources.memory_limit_mb() == pytest.approx(16000000 / 1024), "cgroup v1 reports 'no limit' as a huge number"
    monkeypatch.setenv("REPL_ID", "abc")
    assert resources.machine()["replit"] is True and any("Replit" in n for n in resources.machine()["notes"])


def test_the_estimate_grows_with_the_things_that_make_a_network_big_and_is_compared_with_what_is_free(monkeypatch):
    base = dict(arch="lstm", n_features=60, lookback=20, hidden=64, layers=2, batch_size=256, max_rows=50_000)
    small = resources.deep_mb(**base)
    assert resources.deep_mb(**{**base, "hidden": 256}) > small and resources.deep_mb(**{**base, "batch_size": 1024}) > small and resources.deep_mb(**{**base, "max_rows": 400_000}) > small + 80
    assert resources.deep_mb(**{**base, "arch": "patchtst", "heads": 8}) > small, "attention keeps more"
    assert resources.dataset_mb(500, 2500, 120) > 100 * resources.dataset_mb(5, 2500, 120) * 0.99
    _fake(monkeypatch, {"/sys/fs/cgroup/memory.max": str(1000 * 2 ** 20), "/sys/fs/cgroup/memory.current": str(200 * 2 ** 20)})
    assert resources.preflight(300)["level"] == "ok" and resources.preflight(600)["level"] == "warn" and resources.preflight(900)["level"] == "stop"
    stop = resources.preflight(900, "This experiment")
    assert "This experiment needs roughly 900 MB" in stop["message"] and "800 MB are free" in stop["message"] and "docs/deployment.md" in stop["message"] and stop["free_mb"] == 800


def test_the_watchdog_stops_a_run_before_the_kernel_does_and_cannot_be_swallowed(monkeypatch):
    _fake(monkeypatch, {"/sys/fs/cgroup/memory.max": str(1000 * 2 ** 20), "/sys/fs/cgroup/memory.current": str(500 * 2 ** 20)})
    resources.watchdog(force=True)                                                       # half full: fine
    _fake(monkeypatch, {"/sys/fs/cgroup/memory.max": str(1000 * 2 ** 20), "/sys/fs/cgroup/memory.current": str(950 * 2 ** 20)})
    with pytest.raises(resources.ResourceLimit, match="950 MB of the 1,000 MB"):
        resources.watchdog(force=True)
    try:
        try:
            resources.watchdog(force=True)
        except Exception:                                                                # a model's own handlers do this
            pytest.fail("a resource stop must not be caught by `except Exception`")
    except resources.ResourceLimit:
        pass
    _fake(monkeypatch, {})
    resources.watchdog(force=True)                                                       # limit unknown: silent


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fake_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


_fake_fetch = _fetch


def test_a_job_that_runs_out_of_memory_ends_with_a_message_and_keeps_the_epochs_it_finished(app, monkeypatch):
    state = {"epochs": 0}

    def work(progress):
        live.set_context(model="lstm", fold=1, folds=1)
        live.start(epochs=50, patience=5, device="cpu", n_params=10, n_train=100, n_val=10, batch_size=16)
        for e in range(50):
            state["epochs"] = e + 1
            live.epoch(e, 1.0 / (e + 1), 1.1 / (e + 1), lr=1e-3, grad=0.1, seconds=0.01, best=1.0, best_epoch=e, bad=0, improved=True, epochs=50, patience=5)
            if e == 3:                                                                    # the machine fills up
                _fake(monkeypatch, {"/sys/fs/cgroup/memory.max": str(1000 * 2 ** 20), "/sys/fs/cgroup/memory.current": str(980 * 2 ** 20)})
                resources._last["at"] = 0.0
        return {"never": True}

    view = _wait(app, app.submit_task("Train a big one", work, kind="ml")["job"])
    assert view["status"] == "error" and "Stopped before the machine killed the app" in view["error"] and state["epochs"] == 5, "it stopped at the first event after the memory filled up"
    fit = app.extra_get["/api/training/fits"]({})["fits"][0]
    assert fit["status"] == "error" and fit["epochs_run"] == 4 and fit["job_label"] == "Train a big one"
    assert len(app.extra_get["/api/training/fit"]({"id": str(fit["id"])})["epochs_detail"]) == 4, "what it had learned is not lost"


def test_an_experiment_that_would_not_fit_is_refused_before_a_job_is_queued(app, monkeypatch):
    body = {"dataset": {"feature_groups": ["price", "momentum"]}, "split": {"min_train_days": 400}, "model": {"kind": "lstm", "params": {"epochs": 2, "max_rows": 400000, "hidden": 512, "layers": 3, "batch_size": 4096, "lookback": 60}}}
    _fake(monkeypatch, {"/sys/fs/cgroup/memory.max": str(900 * 2 ** 20), "/sys/fs/cgroup/memory.current": str(400 * 2 ** 20)})
    before = len(app.jobs)
    with pytest.raises(ApiError, match="needs roughly .* MB and about 500 MB are free"):
        app.extra_post["/api/ml/train"](body)
    assert len(app.jobs) == before, "nothing was queued"
    monkeypatch.undo()
    res = app.extra_get["/api/system/resources"]({})
    assert res["threads"] >= 1 and res["cpus_visible"] and any("refused before it starts" in p for p in res["protection"])


def test_the_system_route_describes_this_machine(app):
    res = app.extra_get["/api/system/resources"]({})
    assert {"cpu_limit", "threads", "memory_limit_mb", "memory_free_mb", "notes", "protection", "replit"} <= set(res) and res["threads"] <= (os.cpu_count() or 1)
    time.sleep(0)
