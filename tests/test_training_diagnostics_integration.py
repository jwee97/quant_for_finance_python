"""The gradient monitor inside the shared trainer: it must not change what the training does, it must measure the gradient the optimizer uses (accumulated, unscaled, before clipping), it must persist compact
summaries, and its detector state must survive an interruption."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import trainer as tn          # noqa: E402

torch.set_num_threads(1)
N, D = 256, 5


def problem(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N + 64, D)).astype(np.float32)
    y = (X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0], dtype=np.float32) + 0.1 * rng.normal(size=N + 64)).astype(np.float32)
    split = {"train": np.arange(N), "val": np.arange(N, N + 64)}

    def batch(name, rows):
        r = split[name][rows]
        return torch.from_numpy(X[r]), torch.from_numpy(y[r])

    def loss(net, b):
        return ((net(b[0]).squeeze(-1) - b[1]) ** 2).mean()

    def net():
        torch.manual_seed(5)
        return torch.nn.Sequential(torch.nn.Linear(D, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    return batch, loss, net


def cfg(folder=None, **kw):
    base = dict(epochs=4, batch_size=64, learning_rate=0.01, patience=50, seed=3, checkpoint_dir=None if folder is None else str(folder), config_fp="c", data_fp="d", grad_clip=0.0)
    base.update(kw)
    return tn.TrainConfig(**base)


def weights(net):
    return torch.cat([p.detach().flatten() for p in net.parameters()])


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("level", ["lightweight", "standard", "detailed"])
def test_enabling_diagnostics_does_not_change_the_training(level, tmp_path):
    batch, loss, make = problem()
    off, on = make(), make()
    h0 = tn.fit_network(off, batch, loss, N, 64, cfg(tmp_path / "off", grad_level="off"))
    h1 = tn.fit_network(on, batch, loss, N, 64, cfg(tmp_path / "on", grad_level=level))
    assert torch.equal(weights(off), weights(on)) and h0.train_loss == h1.train_loss and h0.val_loss == h1.val_loss        # bit for bit: the monitor only reads
    assert not (tmp_path / "off" / "gradients.jsonl").exists() and (tmp_path / "on" / "gradients.jsonl").exists()


def test_the_measured_gradient_is_the_accumulated_one_before_clipping(tmp_path):
    batch, loss, make = problem()
    accum, size = 4, 64                                                                       # four batches of 64 = one optimizer step per epoch
    net = make()
    ref = make()
    tn.set_seed(3)
    order = tn.batches(N, size, np.random.default_rng(3), None)                                # the first epoch's batches, as the trainer draws them
    for rows_ in order:
        (loss(ref, batch("train", rows_)) / accum).backward()                                  # what the loop accumulates
    expected = math.sqrt(sum(float((p.grad.double() ** 2).sum()) for p in ref.parameters()))
    tn.fit_network(net, batch, loss, N, 64, cfg(tmp_path, epochs=1, accumulation_steps=accum, grad_level="standard", grad_clip=1e-3))      # a clip far below the norm: the unclipped one is what is reported
    g = rows(tmp_path / "gradients.jsonl")[0]["global"]
    assert g["optimizer_steps"] == 1 and g["l2_median"] == pytest.approx(expected, rel=1e-5)
    assert g["clip_frequency"] == 1.0 and g["post_clip_l2_mean"] == pytest.approx(1e-3, rel=1e-3)


def test_summaries_are_compact_and_layers_follow_the_model(tmp_path):
    batch, loss, make = problem()
    tn.fit_network(make(), batch, loss, N, 64, cfg(tmp_path, grad_level="detailed"))
    out = rows(tmp_path / "gradients.jsonl")
    assert [r["epoch"] for r in out] == [0, 1, 2, 3]
    assert [l["name"] for l in out[0]["layers"]] == ["0", "2"] and "histogram" in out[0] and len(out[0]["histogram"]["bins"]) == 40
    assert (tmp_path / "gradients.jsonl").stat().st_size < 20_000                                  # kilobytes for a whole run, never the gradients themselves
    assert json.loads((tmp_path / "diagnostics.json").read_text())["level"] == "detailed"
    start = [r for r in rows(tmp_path / "metrics.jsonl") if r["type"] == "start"][0]
    assert start["diagnostics"]["level"] == "detailed" and start["diagnostics"]["clip"] == 0.0     # the profile is recorded with the run


def test_frozen_layers_are_excluded(tmp_path):
    batch, loss, make = problem()
    net = make()
    for p in net[0].parameters():
        p.requires_grad_(False)
    tn.fit_network(net, batch, loss, N, 64, cfg(tmp_path, grad_level="standard"))
    g = rows(tmp_path / "gradients.jsonl")[0]
    assert [l["name"] for l in g["layers"]] == ["2"] and g["global"]["frozen_params"] == 2 and g["global"]["missing_gradients"] == 0


class Interrupt(BaseException):
    pass


def test_logs_and_detector_state_continue_after_an_interruption(tmp_path):
    batch, loss, make = problem()
    count = {"n": 0}
    per_epoch = 4 + 1                                                                         # four training batches and one validation chunk

    def dying(net, b):
        count["n"] += 1
        if count["n"] > 2 * per_epoch + 2:
            raise Interrupt()
        return loss(net, b)

    with pytest.raises(Interrupt):
        tn.fit_network(make(), batch, dying, N, 64, cfg(tmp_path, epochs=6, grad_level="standard"))
    h = tn.fit_network(make(), batch, loss, N, 64, cfg(tmp_path, epochs=6, grad_level="standard", resume=True))
    assert h.recovery["level"] == "exact"
    assert [r["epoch"] for r in rows(tmp_path / "gradients.jsonl")] == [0, 1, 2, 3, 4, 5]          # one summary per epoch, none twice, none missing
    assert [r["epoch"] for r in rows(tmp_path / "metrics.jsonl") if r["type"] == "epoch"] == [1, 2, 3, 4, 5, 6]
    straight = tn.fit_network(make(), batch, loss, N, 64, cfg(tmp_path / "ref", epochs=6, grad_level="standard"))
    a, b = rows(tmp_path / "gradients.jsonl"), rows(tmp_path / "ref" / "gradients.jsonl")
    for x, y in zip(a, b):
        assert x["global"]["l2_median"] == pytest.approx(y["global"]["l2_median"], rel=1e-6)       # the gradients of the resumed run are the uninterrupted run's
    assert straight.train_loss == h.train_loss


def test_a_crash_between_the_checkpoint_and_the_log_is_repaired_on_resume(tmp_path):
    batch, loss, make = problem()
    tn.fit_network(make(), batch, loss, N, 64, cfg(tmp_path, epochs=3, grad_level="standard"))
    path = tmp_path / "metrics.jsonl"
    kept = [ln for ln in path.read_text().splitlines() if not (json.loads(ln).get("type") in ("epoch", "finish") and json.loads(ln).get("epoch", 3) == 3)]
    path.write_text("\n".join(kept) + "\n")                                                      # the last row never reached the disk
    gpath = tmp_path / "gradients.jsonl"
    gpath.write_text("\n".join(gpath.read_text().splitlines()[:2]) + "\n")
    tn.fit_network(make(), batch, loss, N, 64, cfg(tmp_path, epochs=3, grad_level="standard", resume=True))
    got = [json.loads(ln) for ln in path.read_text().splitlines()]
    epochs = [r for r in got if r["type"] == "epoch"]
    assert [r["epoch"] for r in epochs] == [1, 2, 3] and epochs[-1].get("recovered") is True
    assert [json.loads(ln)["epoch"] for ln in gpath.read_text().splitlines()] == [0, 1, 2]
