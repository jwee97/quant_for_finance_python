"""Exact continuation after an interruption for every supervised deep architecture the platform trains through ``fit_network`` (the compatibility matrix in docs/training_reliability.md is built from this file):
the resumed run's final weights and losses equal the uninterrupted run's. Each is a CPU run on small synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import trainer as tn          # noqa: E402
from src.ml.deep.nets import build_net         # noqa: E402

torch.set_num_threads(1)
LOOK, FEAT = 8, 3
PARAMS = {"hidden": 8, "layers": 2, "dropout": 0.2, "width": 8, "blocks": 2, "d_model": 8, "n_heads": 2, "ff_dim": 16, "patch_len": 4, "stride": 2, "channels": 4, "kernel_size": 3, "pool_sizes": "4,2,1"}
SUPERVISED = ["lstm", "gru", "tcn", "patchtst", "itransformer", "dlinear", "tft", "nbeats", "nhits"]


class Interrupt(BaseException):
    pass


def setup(arch):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, LOOK, FEAT)).astype(np.float32)
    y = (X[:, -1, 0] - 0.5 * X[:, -2, 1] + 0.1 * rng.normal(size=300)).astype(np.float32)
    split = {"train": np.arange(0, 240), "val": np.arange(240, 300)}

    def batch(name, rows):
        r = split[name][rows]
        return torch.from_numpy(X[r]), torch.from_numpy(y[r])

    def loss(net, b):
        return ((net(b[0]).reshape(len(b[1]), -1)[:, 0] - b[1]) ** 2).mean()

    def net():
        torch.manual_seed(3)
        return build_net(arch, FEAT, LOOK, 1, PARAMS)
    return batch, loss, net


def weights(net):
    return torch.cat([p.detach().flatten() for p in net.state_dict().values()])


@pytest.mark.parametrize("arch", SUPERVISED)
def test_resume_after_an_interruption_matches_the_uninterrupted_run(arch, tmp_path):
    batch, loss, net = setup(arch)

    def cfg(folder, **kw):
        return tn.TrainConfig(**{**dict(epochs=5, batch_size=64, learning_rate=0.01, patience=50, checkpoint_dir=str(folder), config_fp="c", data_fp="d", seed=7), **kw})

    full_net = net()
    full = tn.fit_network(full_net, batch, loss, 240, 60, cfg(tmp_path / "full"))
    count = {"n": 0}
    per_epoch = -(-240 // 64) + -(-60 // 128)

    def dying(model, b):
        count["n"] += 1
        if count["n"] > 2 * per_epoch + 1:
            raise Interrupt()
        return loss(model, b)

    with pytest.raises(Interrupt):
        tn.fit_network(net(), batch, dying, 240, 60, cfg(tmp_path / "run"))
    resumed_net = net()
    h = tn.fit_network(resumed_net, batch, loss, 240, 60, cfg(tmp_path / "run", resume=True))
    assert h.recovery["level"] == "exact" and h.recovery["resumed_from_epoch"] == 2
    np.testing.assert_allclose(h.train_loss, full.train_loss, rtol=1e-5)
    np.testing.assert_allclose(h.val_loss, full.val_loss, rtol=1e-5)
    assert torch.allclose(weights(resumed_net), weights(full_net), atol=1e-6)


def test_resuming_a_run_that_had_stopped_early_does_not_train_another_epoch(tmp_path):
    batch, loss, net = setup("nbeats")
    cfg = tn.TrainConfig(epochs=40, batch_size=64, learning_rate=0.05, patience=2, checkpoint_dir=str(tmp_path), config_fp="c", data_fp="d", seed=7, resume=True)
    first = tn.fit_network(net(), batch, loss, 240, 60, cfg)
    assert first.stopped_early and len(first.train_loss) < 40
    again_net = net()
    again = tn.fit_network(again_net, batch, loss, 240, 60, cfg)
    assert again.stopped_early and len(again.train_loss) == len(first.train_loss) and again.recovery["level"] == "exact"
    assert again.best_epoch == first.best_epoch
