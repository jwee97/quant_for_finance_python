"""What a training tells the page while it runs (src/ml/live.py): each epoch's losses, the best so far, the patience used up and the time left; the same story as the job's progress message; a bounded history; and a
cancellation that stops the network between two batches. The real training loop and a real walk-forward fit are run, not a stand-in."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from src.ml import live
from src.ml.live import Telemetry

torch = pytest.importorskip("torch")

from src.ml.deep import trainer as tn  # noqa: E402
from src.webapp.api import Job, JobCancelled  # noqa: E402


def _tiny_fit(epochs=6, patience=3, seed=0, noisy_val=False, telemetry=None, batch=16):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 4)).astype("float32")
    y = (X @ np.array([1.0, -0.5, 0.0, 0.2], dtype="float32") + rng.normal(0, 0.3, 200)).astype("float32")
    if noisy_val:
        y[160:] = rng.normal(0, 3.0, 40)                                                    # a validation tail the network cannot learn from the training rows: its loss stops improving
    net = torch.nn.Linear(4, 1)
    tx, ty = torch.from_numpy(X), torch.from_numpy(y)

    def batch_fn(split, rows):
        lo = 0 if split == "train" else 160
        r = torch.as_tensor(rows) + lo
        return tx[r], ty[r]

    def loss_fn(model, b):
        return torch.mean((model(b[0]).squeeze(-1) - b[1]) ** 2)

    cfg = tn.TrainConfig(epochs=epochs, batch_size=batch, learning_rate=0.05, patience=patience, seed=seed)
    return tn.fit_network(net, batch_fn, loss_fn, 160, 40, cfg)


def test_the_loop_reports_every_epoch_and_the_summary_agrees_with_its_history():
    t = Telemetry()
    with live.use(t):
        live.set_context(model="linear", fold=1, folds=3)
        hist = _tiny_fit(epochs=6, patience=10)
    snap = t.snapshot()
    run = snap["run"]
    assert run["epochs"] == 6 and run["patience"] == 10 and run["n_params"] == 5 and run["device"] == "cpu" and run["context"] == {"model": "linear", "fold": 1, "folds": 3}
    h = run["history"]
    assert h["epoch"] == list(range(6)) and h["train"] == pytest.approx(hist.train_loss) and h["val"] == pytest.approx(hist.val_loss) and h["lr"] == pytest.approx(hist.lr)
    cur = run["current"]
    assert cur["epoch"] == 6 and cur["epochs"] == 6 and cur["best"] == pytest.approx(min(hist.val_loss)) and cur["best_epoch"] == hist.best_epoch + 1 and cur["eta_seconds"] == 0.0
    done = snap["done"][0]
    assert done["epochs_run"] == 6 and done["best_epoch"] == hist.best_epoch + 1 and done["best_val"] == pytest.approx(min(hist.val_loss)) and done["stopped_early"] is False
    json.dumps(snap)                                                                      # the page gets it as JSON


def test_patience_is_counted_and_the_message_tells_the_epoch_in_words():
    msgs = []
    t = Telemetry(on_message=msgs.append)
    with live.use(t):
        live.set_context(model="gru", fold=2, folds=4)
        hist = _tiny_fit(epochs=40, patience=3, noisy_val=True)
    assert hist.stopped_early and len(hist.val_loss) < 40
    last = t.snapshot()["done"][-1]
    assert last["stopped_early"] is True and last["epochs_run"] == len(hist.val_loss)
    cur = t.snapshot()["run"]["current"]
    assert cur["bad"] == 3 and cur["patience"] == 3
    epoch_msgs = [m for m in msgs if "epoch" in m and "train" in m]
    assert len(epoch_msgs) == len(hist.val_loss) and epoch_msgs[0].startswith("gru · fold 2 of 4 · epoch 1 of 40 · train ") and "val " in epoch_msgs[0] and "patience 0/3" in epoch_msgs[0] or "patience 1/3" in epoch_msgs[0]
    assert epoch_msgs[-1].endswith("patience 3/3") and "best " in epoch_msgs[-1]


def test_nothing_happens_without_a_listener_and_the_history_is_bounded():
    _tiny_fit(epochs=2)                                                                   # no telemetry installed: the loop runs as it always did
    assert live.current() is None
    t = Telemetry()
    with live.use(t):
        live.start(epochs=5000, patience=None)
        for e in range(live.MAX_POINTS + 50):
            live.epoch(e, 1.0 / (e + 1), None, epochs=5000)
    h = t.snapshot()["run"]["history"]
    assert len(h["epoch"]) == live.MAX_POINTS and h["epoch"][-1] == live.MAX_POINTS + 49 and t.snapshot()["run"]["current"]["val"] is None


def test_a_job_shows_the_live_view_and_cancelling_stops_the_network_between_batches():
    job = Job("train", "ml")
    with live.use(job.telemetry):
        live.set_context(model="lstm")
        _tiny_fit(epochs=3, patience=5)
    view = job.view()
    assert view["live"]["run"]["current"]["epoch"] == 3 and "epoch 3 of 3" in view["message"]
    stopper = Job("train", "ml")
    stopper.cancel_event.set()
    start = time.time()
    with pytest.raises(JobCancelled), live.use(stopper.telemetry):
        _tiny_fit(epochs=500, patience=500)
    assert time.time() - start < 5 and stopper.telemetry.snapshot() is None, "it stopped at the first event instead of running 500 epochs"


def test_a_walk_forward_fit_names_the_fold_and_keeps_a_summary_of_each_network():
    from src.ml import config as mc, datasets as dsm, training as tr
    from tests.ml_helpers import make_bundle
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 400, "refit_every": 250}}, defaults=mc.DEFAULTS)
    ds = dsm.build_dataset(make_bundle(8, 1100, seed=31, alpha=1.0), cfg["dataset"], "live")
    t = Telemetry()
    with live.use(t):
        tr.walk_forward_train(ds, {"kind": "lstm", "params": {"epochs": 2, "max_rows": 800, "hidden": 4, "batch_size": 128, "lookback": 5, "patience": 2}}, cfg["split"], 1)
    snap = t.snapshot()
    assert snap["fits"] >= 2 and len(snap["done"]) == snap["fits"]
    assert [d["context"]["fold"] for d in snap["done"]] == list(range(1, snap["fits"] + 1)) and all(d["context"]["model"] == "lstm" and d["context"]["folds"] >= snap["fits"] for d in snap["done"])
    assert all(d["epochs_run"] >= 1 and d["best_val"] is not None for d in snap["done"])
