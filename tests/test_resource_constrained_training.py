"""Training on a small machine such as a Replit container: monitoring defaults to the lightweight level there, a run that is stopped for lack of memory leaves a valid checkpoint and resumes exactly, the machine's
memory pressure is reported as a warning, and the diagnostics files stay bounded."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml import resources                          # noqa: E402
from src.ml.deep import gradients as gd, runs, trainer as tn   # noqa: E402
from tests.test_training_diagnostics_integration import N, cfg, problem, rows, weights   # noqa: E402


def test_monitoring_defaults_to_lightweight_on_replit_and_standard_elsewhere(tmp_path, monkeypatch):
    batch, loss, make = problem()
    monkeypatch.setattr(resources, "on_replit", lambda: True)
    tn.fit_network(make(), batch, loss, N, 64, cfg(tmp_path / "replit", grad_level="auto"))
    diag = json.loads((tmp_path / "replit" / "diagnostics.json").read_text())
    first = rows(tmp_path / "replit" / "gradients.jsonl")[0]
    assert diag["level"] == "lightweight" and diag["replit"] is True and first["level"] == "lightweight" and first["layers"] == [] and "sampled" not in first["global"]
    monkeypatch.setattr(resources, "on_replit", lambda: False)
    tn.fit_network(make(), batch, loss, N, 64, cfg(tmp_path / "local", grad_level="auto"))
    assert json.loads((tmp_path / "local" / "diagnostics.json").read_text())["level"] == "standard"
    tn.fit_network(make(), batch, loss, N, 64, cfg(None, grad_level="auto"))                       # no run folder: nowhere to persist, so off (and nothing is written)


def test_a_run_stopped_for_lack_of_memory_keeps_a_valid_checkpoint_and_resumes_exactly(tmp_path, monkeypatch):
    batch, loss, make = problem()
    ref = make()
    tn.fit_network(ref, batch, loss, N, 64, cfg(tmp_path / "ref", epochs=6, grad_level="standard"))
    count = {"n": 0}
    limit = {"on": False}

    def starved(net, b):
        count["n"] += 1
        if count["n"] > 3 * 5 + 1:                                                                # the machine runs short in the fourth epoch
            limit["on"] = True
            monkeypatch.setattr(resources, "memory_limit_mb", lambda: 1000.0)
            monkeypatch.setattr(resources, "memory_used_mb", lambda: 990.0)
            resources.watchdog(force=True)                                                        # what the job's telemetry check does between batches
        return loss(net, b)

    with pytest.raises(resources.ResourceLimit):
        tn.fit_network(make(), batch, starved, N, 64, cfg(tmp_path / "run", epochs=6, grad_level="standard"))
    monkeypatch.undo()
    log = rows(tmp_path / "run" / "metrics.jsonl")
    assert log[-1]["type"] == "interrupted" and "ResourceLimit" in log[-1]["reason"]
    assert runs.status_of(log, (tmp_path / "run" / "metrics.jsonl").stat().st_mtime) == "interrupted"
    from src.ml.deep.checkpoint import CheckpointManager
    blob, which, problems = CheckpointManager(tmp_path / "run").latest_valid()
    assert blob is not None and which == "last.pt" and not problems and blob["epoch"] == 2
    resumed = make()
    h = tn.fit_network(resumed, batch, loss, N, 64, cfg(tmp_path / "run", epochs=6, grad_level="standard", resume=True))
    assert h.recovery["level"] == "exact" and torch.equal(weights(ref), weights(resumed))


def test_memory_pressure_is_a_warning_with_the_numbers(tmp_path, monkeypatch):
    batch, loss, make = problem()
    monkeypatch.setattr(resources, "memory_limit_mb", lambda: 1000.0)
    monkeypatch.setattr(resources, "memory_used_mb", lambda: 900.0)
    tn.fit_network(make(), batch, loss, N, 64, cfg(tmp_path, epochs=3, grad_level="lightweight"))
    warns = [r for r in rows(tmp_path / "metrics.jsonl") if r["type"] == "resource_warning"]
    assert len(warns) == 1 and warns[0]["used_mb"] == 900.0 and warns[0]["limit_mb"] == 1000.0         # once, not every epoch


def test_the_gradient_log_stays_bounded_and_keeps_the_newest_units(tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "GRADIENT_FILE_CAP", 6000)
    for epoch in range(120):
        gd.append_summary(tmp_path, {"type": "gradient", "optimizer": "o", "epoch": epoch, "global": {"l2_median": 1.0}, "layers": [{"name": f"l{i}", "rms": 0.1} for i in range(8)]})
    got = [json.loads(line)["epoch"] for line in (tmp_path / "gradients.jsonl").read_text().splitlines()]
    assert (tmp_path / "gradients.jsonl").stat().st_size < 3 * 6000 and got[-40:] == list(range(80, 120)) and got == sorted(got)
