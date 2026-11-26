"""The Training Monitor's backend: a deep model submitted through ``POST /api/ml/train`` gets a run folder with checkpoints and a metrics file, the run is listed with a status derived from its files, it can be
stopped, resumed (never automatically), restarted as a new run, archived, and a request that does not fit in memory is refused before it is queued."""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("torch")

from src.ml.deep import profiles, runs                                   # noqa: E402
from src.webapp.api import ApiError                                      # noqa: E402
from tests.ml_helpers import make_bundle                                 # noqa: E402
from tests.test_webapp_assets_ml import _app, finished, wait             # noqa: E402

TINY = {"epochs": 4, "max_rows": 1500, "width": 16, "blocks": 2, "layers": 2, "batch_size": 128, "lookback": 8, "patience": 5}
BASE = {"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "nbeats", "params": TINY}, "mode": "development"}


def wait_any(app, resp, timeout=600.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = app.job(resp["job"])
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.1)
    raise AssertionError("the job did not end")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("training")
    bundle = make_bundle(n_assets=8, n=1300, seed=3, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


@pytest.fixture(scope="module")
def first(app):
    resp = app.extra_post["/api/ml/train"]({**BASE, "run_id": "run-one"})
    assert resp["training_run"]["run_id"] == "run-one"
    return {"resp": resp, "result": finished(app, resp)}


def test_a_deep_model_trained_through_the_api_leaves_a_run_on_disk(app, first):
    assert first["result"]["training_run"] == "run-one"
    listing = app.extra_get["/api/training/runs"]({})
    run = next(r for r in listing["runs"] if r["id"] == "run-one")
    assert run["status"] in ("completed", "stopped_early") and run["epoch"] >= 1 and run["request_saved"]
    d = app.extra_get["/api/training/run"]({"id": "run-one"})
    assert len(d["series"]["epoch"]) == d["n_epochs_total"] >= 1
    assert d["series"]["train_loss"][0] is not None and d["series"]["lr"][0] > 0
    assert d["checkpoints"] and any(c["file"] == "last.pt" and c["valid"] for c in d["checkpoints"])
    assert d["request"]["body"]["model"]["kind"] == "nbeats"
    assert "resume" not in d["request"]["body"]


def test_the_monitor_never_shows_test_segment_metrics(app, first):
    d = app.extra_get["/api/training/run"]({"id": "run-one"})
    blob = json.dumps(d)
    assert "test_" not in blob and "forward" not in blob


def test_since_returns_only_newer_epochs(app, first):
    d = app.extra_get["/api/training/run"]({"id": "run-one", "since": 2})
    total = d["n_epochs_total"]
    assert len(d["series"]["epoch"]) == max(total - 2, 0)


def test_a_finished_run_resumes_only_when_asked_and_a_second_start_with_the_same_id_is_refused(app, first):
    with pytest.raises(ApiError, match="already has checkpoints"):
        app.extra_post["/api/ml/train"]({**BASE, "run_id": "run-one"})
    resp = app.extra_post["/api/training/resume"]({"id": "run-one", "mode": "resume"})
    out = finished(app, resp)
    assert out["training_run"] == "run-one"
    d = app.extra_get["/api/training/run"]({"id": "run-one"})
    assert d["summary"]["epoch"] >= 1


def test_restart_makes_a_new_run_and_leaves_the_old_one_alone(app, first):
    before = {r["id"] for r in app.extra_get["/api/training/runs"]({})["runs"]}
    resp = app.extra_post["/api/training/resume"]({"id": "run-one", "mode": "restart"})
    finished(app, resp)
    after = {r["id"] for r in app.extra_get["/api/training/runs"]({})["runs"]}
    assert len(after - before) == 1


def test_stop_ends_the_whole_experiment_after_a_checkpoint_and_the_run_can_be_resumed(app):
    resp = app.extra_post["/api/ml/train"]({**BASE, "run_id": "run-stop", "model": {"kind": "nbeats", "params": {**TINY, "epochs": 5000, "patience": 5000}}})
    app.extra_post["/api/training/control"]({"id": "run-stop", "action": "stop"})                       # written before the first epoch ends: the loop reads it after that epoch's checkpoint
    job = wait_any(app, resp)
    assert job["status"] == "cancelled" and "stopped by request" in job["message"] and "resume" in job["message"], job
    d = app.extra_get["/api/training/run"]({"id": "run-stop"})
    assert d["summary"]["status"] == "stopped" and d["summary"]["resumable"] and d["n_epochs_total"] == 1
    assert len(d["fits"]) == 1, "the stop must end the experiment, not let the next walk-forward fit start"
    assert d["control"] == "stop"
    listed = next(r for r in app.extra_get["/api/training/runs"]({})["runs"] if r["id"] == "run-stop")
    assert listed["resumable"] and listed["status"] == "stopped"
    # the epochs a resumed run may use are an execution setting, not part of what the model is: shorten them and resume
    req = runs.base(app.root) / "run-stop" / "request.json"
    saved = json.loads(req.read_text())
    saved["body"]["model"]["params"]["epochs"] = 4
    req.write_text(json.dumps(saved))
    out = finished(app, app.extra_post["/api/training/resume"]({"id": "run-stop", "mode": "resume"}))
    assert out["training_run"] == "run-stop"
    # changing something that IS part of the model (patience also sets the scheduler's) is refused for the whole job, never a silently skipped fit
    saved["body"]["model"]["params"]["patience"] = 7
    req.write_text(json.dumps(saved))
    bad = wait_any(app, app.extra_post["/api/training/resume"]({"id": "run-stop", "mode": "resume"}))
    assert bad["status"] == "error" and "cannot continue safely" in bad["error"] and "hyperparameters differ" in bad["error"], bad
    d = app.extra_get["/api/training/run"]({"id": "run-stop", "fit": d["fit"]})
    assert d["control"] == "" and d["summary"]["epoch"] >= 2
    resume = [e for e in d["events"] if e["type"] == "resume"]
    assert resume and resume[0]["level"] in ("exact", "epoch_boundary")


def test_archive_removes_a_run_from_the_list_without_deleting_its_files(app, first):
    out = app.extra_post["/api/training/archive"]({"id": "run-stop"})
    assert (runs.base(app.root) / "_archive" / out["archived_to"]).is_dir()
    assert all(r["id"] != "run-stop" for r in app.extra_get["/api/training/runs"]({})["runs"])


def test_unknown_and_malformed_ids_are_refused(app):
    for path, body in (("/api/training/control", {"id": "nope-nope", "action": "stop"}), ("/api/training/archive", {"id": "../etc"}), ("/api/training/resume", {"id": "nope-nope"})):
        with pytest.raises(ApiError):
            app.extra_post[path](body)
    with pytest.raises(ApiError):
        app.extra_get["/api/training/run"]({"id": "../x"})
    with pytest.raises(ApiError, match="action"):
        app.extra_post["/api/training/control"]({"id": "run-one", "action": "format"})


def test_preflight_refuses_what_cannot_fit_and_suggests_a_smaller_batch(app):
    big = app.extra_post["/api/training/preflight"]({"arch": "nhits", "n_features": 400, "lookback": 500, "width": 2048, "layers": 8, "blocks": 12, "batch_size": 20000, "max_rows": 5_000_000})
    assert big["estimate"]["level"] in ("warn", "stop") and big["suggestion"]
    small = app.extra_post["/api/training/preflight"]({"arch": "nbeats", "n_features": 20, "lookback": 8, "width": 16, "batch_size": 64, "max_rows": 2000})
    assert small["estimate"]["level"] == "ok" and small["suggestion"] is None
    with pytest.raises(ApiError, match="profile"):
        app.extra_post["/api/training/preflight"]({"profile": "turbo"})


def test_profiles_change_only_execution_and_say_what_they_changed():
    base = {"batch_size": 4096, "num_threads": 0}
    out, rec = profiles.apply(base, "conservative")
    assert out["batch_size"] <= 128 and rec["changed"]["batch_size"]["from"] == 4096
    assert rec["comparable_with_unprofiled_run"] is False
    out2, rec2 = profiles.apply({"batch_size": 64}, "balanced")
    assert out2["batch_size"] == 64 and "batch_size" not in rec2["changed"]
    with pytest.raises(ValueError):
        profiles.apply(base, "turbo")


def test_the_run_folder_does_not_change_the_experiments_fingerprint(app):
    from src.ml.config import build_key, experiment_config
    a = experiment_config({"dataset": BASE["dataset"], "split": BASE["split"], "model": BASE["model"]})
    assert build_key(a) == build_key(experiment_config({"dataset": BASE["dataset"], "split": BASE["split"], "model": BASE["model"]}))


def test_a_run_that_stopped_for_lack_of_memory_says_so_and_suggests_what_to_change(tmp_path):
    fit = runs.base(tmp_path) / "oom-run" / "nbeats-x"
    fit.mkdir(parents=True)
    rows = [{"type": "start", "at": 1.0, "epochs": 10}, {"type": "epoch", "epoch": 1, "epochs": 10, "train_loss": 1.0, "val_loss": 1.0, "lr": 0.001, "grad_norm": 1.0, "seconds": 1.0, "best_val": 1.0, "best_epoch": 1, "bad": 0,
                                                                "patience": 5, "improved": True, "at": 2.0, "checkpoint": None, "peak_mb": 1, "process_rss_mb": 1},
            {"type": "interrupted", "epoch": 1, "at": 3.0, "reason": "ResourceLimit: the process uses 3900 MB of its 4000 MB"}]
    (fit / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (fit.parent / "request.json").write_text(json.dumps({"body": {"model": {"kind": "nbeats", "params": {"batch_size": 1024}}}}))
    d = runs.detail(tmp_path, "oom-run")
    assert d["summary"]["status"] == "interrupted"
    assert d["memory_advice"]["batch_size"] == 512 and d["memory_advice"]["accumulation_steps"] == 2 and "cannot be continued exactly" in d["memory_advice"]["note"]


def test_a_detailed_run_exposes_global_layer_and_histogram_diagnostics(app):
    params = {**TINY, "grad_monitor": "detailed", "epochs": 5, "patience": 50}
    finished(app, app.extra_post["/api/ml/train"]({**BASE, "run_id": "run-grad", "model": {"kind": "nbeats", "params": params}}))
    g = app.extra_get["/api/training/gradients"]({"id": "run-grad"})
    assert g["level"] == "detailed" and g["optimizers"] == ["optimizer"] and len(g["epochs"]) >= 3
    s = g["series"]
    assert all(len(s[k]) == len(g["epochs"]) for k in ("l2_median", "p50", "p99", "zero_frac", "clip_frequency", "nonfinite_steps"))
    assert all(v > 0 for v in s["l2_median"]) and all(a <= b + 1e-12 for a, b in zip(s["p50"], s["p99"]))
    assert g["flow"]["rows"] and g["layers"] == [r["name"] for r in g["flow"]["rows"]]
    row = g["flow"]["rows"][0]
    assert {"mean_abs", "max_abs", "median_abs", "rms", "g_w", "update_ratio", "zero_frac", "n_params"} <= set(row)
    assert g["layer_series"][g["layers"][0]]["rms"] and g["histogram_epochs"]
    h = next(iter(g["histograms"].values()))
    assert len(h["bins"]) == 40 and h["zeros"] >= 0
    again = app.extra_get["/api/training/gradients"]({"id": "run-grad", "flow_epoch": 1, "hist_epochs": "1,2"})
    assert again["flow"]["epoch"] == 0 and set(again["histograms"]) == {"0", "1"}
    d = app.extra_get["/api/training/run"]({"id": "run-grad"})
    assert d["diagnostics"]["level"] == "detailed" and d["series"]["cpu_pct"][0] is not None
    with pytest.raises(ApiError):
        app.extra_get["/api/training/gradients"]({"id": "nope-nope"})


def test_a_run_with_diagnostics_off_says_so(app):
    finished(app, app.extra_post["/api/ml/train"]({**BASE, "run_id": "run-nograd", "model": {"kind": "nbeats", "params": {**TINY, "grad_monitor": "off"}}}))
    g = app.extra_get["/api/training/gradients"]({"id": "run-nograd"})
    assert g["epochs"] == [] and "off" in g["note"]


def test_finished_runs_are_trimmed_and_old_ones_pruned_but_a_resumable_one_never(app, tmp_path):
    from pathlib import Path
    import os
    fit = next((runs.base(app.root) / "run-grad").glob("nbeats-*"))
    assert not list(fit.glob("last.prev.pt")) and not list(fit.glob("epoch-*.pt")) and (fit / "last.pt").exists() and (fit / "best.pt").exists()   # trimmed when the job ended
    old = 0
    for f in (runs.base(app.root) / "run-grad").rglob("*"):
        os.utime(f, (old, old))
    dry = app.extra_post["/api/training/prune"]({"keep_days": 1, "dry_run": True})
    assert "run-grad" in dry["removed"] and dry["freed_bytes"] > 0 and (runs.base(app.root) / "run-grad").exists()
    # a run that was interrupted (resumable) is never removed, however old
    broken = runs.base(app.root) / "broken-old" / "nbeats-x"
    broken.mkdir(parents=True)
    (broken / "metrics.jsonl").write_text(json.dumps({"type": "start", "at": 1.0, "epochs": 5}) + "\n" + json.dumps({"type": "interrupted", "epoch": 0, "at": 2.0, "reason": "killed"}) + "\n")
    for f in broken.parent.rglob("*"):
        os.utime(f, (old, old))
    out = app.extra_post["/api/training/prune"]({"keep_days": 1})
    assert "run-grad" in out["removed"] and "broken-old" not in out["removed"] and not (runs.base(app.root) / "run-grad").exists() and broken.exists()
