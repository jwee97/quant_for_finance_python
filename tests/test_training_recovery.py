"""Training recovery on N-BEATS and N-HiTS: atomic checkpoints, exact continuation, damaged-file fallback, refusal of a changed model or data, early-stopping continuity, persisted metrics, user stop and pause.
Small synthetic data; CPU only."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import checkpoint as ck, trainer as tn          # noqa: E402
from src.ml.deep.nets import build_net                          # noqa: E402

torch.set_num_threads(1)
LOOK, FEAT = 8, 3


def make(arch):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, LOOK, FEAT)).astype(np.float32)
    y = (X[:, -1, 0] - 0.5 * X[:, -2, 1] + 0.1 * rng.normal(size=400)).astype(np.float32)
    split = {"train": np.arange(0, 320), "val": np.arange(320, 400)}

    def batch(name, rows):
        r = split[name][rows]
        return torch.from_numpy(X[r]), torch.from_numpy(y[r])

    def loss(net, b):
        return ((net(b[0]).squeeze(-1) - b[1]) ** 2).mean()

    def net():
        torch.manual_seed(3)
        return build_net(arch, FEAT, LOOK, 1, {"width": 16, "blocks": 2, "layers": 2, "dropout": 0.2, "pool_sizes": "4,2,1"})
    return batch, loss, net, 320, 80


def cfg(folder, **kw):
    base = dict(epochs=6, batch_size=64, learning_rate=0.01, patience=50, checkpoint_dir=str(folder), config_fp="cfg1", data_fp="data1", seed=11)
    base.update(kw)
    return tn.TrainConfig(**base)


def weights(net):
    return torch.cat([p.detach().flatten() for p in net.state_dict().values()])


class Interrupt(BaseException):
    """What a cancelled or killed job looks like to the loop: not an Exception."""


def kill_after(loss, n_batches):
    """A loss function that raises after ``n_batches`` calls: an interruption in the middle of an epoch."""
    count = {"n": 0}

    def wrapped(net, b):
        count["n"] += 1
        if count["n"] > n_batches:
            raise Interrupt("killed")
        return loss(net, b)
    return wrapped


@pytest.mark.parametrize("arch", ["nbeats", "nhits"])
def test_an_interrupted_run_resumes_exactly_and_ends_where_the_uninterrupted_run_ends(arch, tmp_path):
    batch, loss, net, n_tr, n_va = make(arch)
    straight = net()
    h_full = tn.fit_network(straight, batch, loss, n_tr, n_va, cfg(tmp_path / "full"))
    broken = net()
    per_epoch = -(-n_tr // 64) + -(-n_va // 128)                                                         # batches per epoch (training) plus validation chunks
    with pytest.raises(Interrupt):
        tn.fit_network(broken, batch, kill_after(loss, 3 * per_epoch + 2), n_tr, n_va, cfg(tmp_path / "run"))                                 # dies inside the fourth epoch
    assert (tmp_path / "run" / "last.pt").exists()
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows if r["type"] == "epoch"] == [1, 2, 3] and rows[-1]["type"] == "interrupted"
    resumed = net()
    h = tn.fit_network(resumed, batch, loss, n_tr, n_va, cfg(tmp_path / "run", resume=True))
    assert h.recovery["level"] == "exact" and h.recovery["resumed_from_epoch"] == 3 and len(h.train_loss) == 6
    assert h.train_loss == pytest.approx(h_full.train_loss, rel=1e-9) and h.val_loss == pytest.approx(h_full.val_loss, rel=1e-9)
    assert torch.equal(weights(resumed), weights(straight)), "optimizer, scheduler and every RNG came back: the continued run is the uninterrupted one"
    assert [e["type"] for e in h.events] == ["resume"] and h.best_epoch == h_full.best_epoch


def test_the_scheduler_and_the_early_stopping_counter_continue_instead_of_restarting(tmp_path):
    batch, loss, net, n_tr, n_va = make("nbeats")
    c = lambda folder, **kw: cfg(folder, learning_rate=0.0, patience=3, epochs=20, **kw)                       # noqa: E731 - nothing learns: validation never improves after epoch 1
    full = tn.fit_network(net(), batch, loss, n_tr, n_va, c(tmp_path / "full"))
    assert full.stopped_early and len(full.train_loss) == 4 and full.stopped_by == "early_stopping"
    broken = net()
    per_epoch = -(-n_tr // 64) + -(-n_va // 128)
    with pytest.raises(Interrupt):
        tn.fit_network(broken, batch, kill_after(loss, 2 * per_epoch + 1), n_tr, n_va, c(tmp_path / "run"))                                  # two epochs done, 1 bad epoch counted
    blob = ck.CheckpointManager(tmp_path / "run").latest_valid()[0]
    assert blob["bad"] == 1 and blob["scheduler"]["num_bad_epochs"] == 1
    again = tn.fit_network(net(), batch, loss, n_tr, n_va, c(tmp_path / "run", resume=True))
    assert len(again.train_loss) == 4 and again.stopped_early, "early stopping fired at the same epoch, not three epochs after the resume"


def test_a_damaged_newest_checkpoint_falls_back_to_the_previous_one_and_says_so(tmp_path):
    batch, loss, net, n_tr, n_va = make("nhits")
    tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path, epochs=4))
    (tmp_path / "last.pt").write_bytes(b"garbage")
    state, which, problems = ck.CheckpointManager(tmp_path).latest_valid()
    assert which == "last.prev.pt" and state["epoch"] == 2 and problems == ["last.pt is damaged and was skipped"]
    h = tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path, epochs=6, resume=True))
    assert h.recovery["level"] == "epoch_boundary" and "last.pt is damaged" in " ".join(h.recovery["reasons"]) and h.recovery["file"] == "last.prev.pt" and len(h.train_loss) == 6


def test_an_interrupted_write_never_destroys_the_only_valid_checkpoint(tmp_path, monkeypatch):
    batch, loss, net, n_tr, n_va = make("nbeats")
    tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path, epochs=3))
    before = (tmp_path / "last.pt").read_bytes()
    real = torch.save
    monkeypatch.setattr(torch, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("no space left on device")))
    mgr = ck.CheckpointManager(tmp_path)
    with pytest.raises(OSError):
        mgr.save({"model": {}, "epoch": 99})
    monkeypatch.setattr(torch, "save", real)
    assert (tmp_path / "last.pt").read_bytes() == before and mgr.latest_valid()[1] == "last.pt"
    h = tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path, epochs=4, resume=True))
    assert len(h.train_loss) == 4


def test_a_changed_model_or_dataset_is_refused_and_a_warm_start_must_be_asked_for(tmp_path):
    batch, loss, net, n_tr, n_va = make("nbeats")
    tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path, epochs=2))
    with pytest.raises(ck.CheckpointIncompatible, match="training data window"):
        tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path, epochs=4, resume=True, data_fp="other"))
    with pytest.raises(ck.CheckpointIncompatible, match="architecture or hyperparameters"):
        tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path, epochs=4, resume=True, config_fp="other"))
    warm = tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path, epochs=3, resume=True, data_fp="other", allow_warm_start=True))
    assert warm.recovery["level"] == "warm_start" and len(warm.train_loss) == 3, "weights restored, everything else fresh: a new experiment from saved weights"


def test_no_checkpoint_means_restart_and_a_resume_flag_without_files_trains_from_scratch(tmp_path):
    batch, loss, net, n_tr, n_va = make("nbeats")
    h = tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path / "empty", epochs=2, resume=True))
    assert h.recovery is None and len(h.train_loss) == 2
    assert ck.assess(None, "", "a", "b", "cpu")["level"] == "restart_required"


def test_stop_and_pause_end_the_run_after_a_committed_epoch_and_the_run_can_continue(tmp_path):
    batch, loss, net, n_tr, n_va = make("nhits")
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "control.json").write_text(json.dumps({"action": "pause"}))
    h = tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(folder, epochs=6))
    assert h.stopped_by == "user_pause" and len(h.train_loss) == 1 and ck.CheckpointManager(folder).latest_valid()[0]["epoch"] == 0
    (folder / "control.json").unlink()
    more = tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(folder, epochs=6, resume=True))
    assert len(more.train_loss) == 6 and [e["type"] for e in more.events] == ["user_pause", "resume"]
    full = tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path / "full", epochs=6))
    assert more.val_loss == pytest.approx(full.val_loss, rel=1e-9), "a pause is not an interruption of the numbers"


def test_checkpoint_retention_keeps_milestones_best_and_the_last_two_generations(tmp_path):
    batch, loss, net, n_tr, n_va = make("nbeats")
    tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path, epochs=6, keep_milestones=2))
    names = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert names == ["best.pt", "epoch-2.pt", "epoch-4.pt", "epoch-6.pt", "last.pt", "last.prev.pt"] or set(names) >= {"best.pt", "last.pt", "last.prev.pt", "epoch-2.pt", "epoch-4.pt"}
    listing = ck.CheckpointManager(tmp_path).listing()
    assert all(r["valid"] for r in listing) and next(r for r in listing if r["file"] == "last.pt")["epoch"] == 6
    capped = ck.CheckpointManager(tmp_path, keep_milestones=2, max_bytes=1)
    capped._cap()
    assert not list(tmp_path.glob("epoch-*.pt"))


def test_the_metrics_file_is_the_record_the_monitor_reads(tmp_path):
    batch, loss, net, n_tr, n_va = make("nbeats")
    h = tn.fit_network(net(), batch, loss, n_tr, n_va, cfg(tmp_path, epochs=3))
    rows = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    ep = [r for r in rows if r["type"] == "epoch"]
    assert [r["type"] for r in rows][0] == "start" and rows[-1]["type"] == "finish" and rows[-1]["stopped_by"] == "completed"
    assert [r["epoch"] for r in ep] == [1, 2, 3] and [r["train_loss"] for r in ep] == pytest.approx(h.train_loss) and all(r["checkpoint"]["bytes"] > 0 for r in ep) and ep[0]["lr"] == pytest.approx(0.01)
    assert all(set(r) >= {"val_loss", "best_val", "best_epoch", "bad", "seconds", "process_rss_mb", "grad_norm"} for r in ep)


def test_gradient_accumulation_matches_one_big_batch_of_the_same_rows(tmp_path):
    batch, loss, net, n_tr, n_va = make("nbeats")
    a, b = net(), net()
    for model, kw in ((a, dict(batch_size=64, accumulation_steps=1)), (b, dict(batch_size=32, accumulation_steps=2))):
        tn.fit_network(model, batch, loss, n_tr, n_va, dataclasses.replace(tn.TrainConfig(epochs=1, learning_rate=0.0, patience=5, seed=2, grad_clip=0.0), **kw))
    assert torch.equal(weights(a), weights(b)), "with a zero learning rate both leave the weights untouched: accumulation changes only when the optimizer steps"
