"""PyTorch infrastructure and the networks built on it: windows without look-ahead, the training loop, checkpoints, serialization, every architecture end to end, multi-horizon heads and latent features."""

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.ml import config as mc, datasets as dsm, models as md, training as tr   # noqa: E402
from src.ml.deep import data as dd, model as dm, trainer as tn                  # noqa: E402
from src.ml.deep.nets import build_net                                         # noqa: E402
from tests.ml_helpers import make_bundle                                       # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
SPLIT = {"min_train_days": 400, "refit_every": 250}
TINY = {"epochs": 3, "max_rows": 1500, "hidden": 8, "batch_size": 128, "lookback": 8, "patience": 2}
SPECIFIC = {"patchtst": {"d_model": 8, "n_heads": 2, "ff_dim": 16, "patch_len": 4, "stride": 2, "channels": 4}, "nbeats": {"width": 16, "blocks": 2, "layers": 2}, "nhits": {"width": 16, "blocks": 2, "layers": 1},
            "tft": {"n_heads": 2}, "tcn": {"layers": 2}, "autoencoder": {"latent_dim": 4, "layers": 1}, "vae": {"latent_dim": 4, "layers": 1}}
ALL = ["lstm", "gru", "tcn", "patchtst", "nbeats", "nhits", "tft", "autoencoder", "vae"]


def params_for(kind, **over):
    allowed = md._INFO[kind].params
    return {k: v for k, v in {**TINY, **SPECIFIC.get(kind, {}), **over}.items() if k in allowed}


@pytest.fixture(scope="module")
def cfg():
    return mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": SPLIT}, defaults=mc.DEFAULTS)


@pytest.fixture(scope="module")
def ds(cfg):
    return dsm.build_dataset(make_bundle(8, 1100, seed=31, alpha=1.0), cfg["dataset"], "deep")


def fit_on(ds, kind, bound=True, **over):
    pos = ds.positions()
    rows = np.flatnonzero((pos >= 80) & (pos < 700) & ds.Y[ds.target].notna().to_numpy())
    model = md.make_model(kind, params_for(kind, **over), "regression", 3)
    if bound:
        model.bind(ds)
    return model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos[rows]), rows


# ------------------------------------------------------------------------------------------------------------ windows
def test_a_window_reaches_back_and_never_forward_and_pads_the_start():
    dates = pd.bdate_range("2021-01-04", periods=6)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"])
    X = pd.DataFrame({"f": np.arange(12, dtype=float), "g": -np.arange(12, dtype=float)}, index=idx)
    acc = dd.SeqAccessor.from_frame(X)
    t, a = acc.locate(X.index)
    w = acc.windows(t, a, 3)
    assert w.shape == (12, 3, 2)
    assert list(w[8, :, 0]) == [4.0, 6.0, 8.0]                                           # asset A on the 5th day: its own last three days
    assert np.isnan(w[0, :2, 0]).all() and w[0, 2, 0] == 0.0 and np.isnan(w[1, :2, 0]).all()
    seq = dd.sequence_dataset(X, X["f"], 3)
    assert seq.X.shape == (12, 3, 2) and np.isfinite(seq.X).all() and seq.y[5] == 5.0


def test_the_scaler_uses_the_rows_it_was_fitted_on_and_clips_and_fills():
    s = dd.FeatureScaler().fit(np.array([[1.0, 10.0], [3.0, 10.0], [np.nan, 10.0]]))
    out = s.transform(np.array([[2.0, 10.0], [100.0, np.nan], [np.nan, 11.0]], dtype=np.float32))
    assert np.allclose(out[0], [0.0, 0.0]) and out[1, 0] == 5.0 and out[1, 1] == 0.0 and out[2, 0] == 0.0


# ------------------------------------------------------------------------------------------------------------ the training loop
def _toy_net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


def _toy_batches(noise=0.0, n=600):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X @ np.array([1.0, -2.0, 0.5], dtype=np.float32) + noise * rng.normal(size=n)).astype(np.float32)
    split = {"train": np.arange(0, 480), "val": np.arange(480, n)}

    def batch(name, rows):
        r = split[name][rows]
        return torch.from_numpy(X[r]), torch.from_numpy(y[r])
    return batch, (lambda net, b: ((net(b[0]).squeeze(-1) - b[1]) ** 2).mean()), 480, n - 480


def test_the_loop_learns_keeps_the_best_weights_and_records_the_run():
    batch, loss, n_train, n_val = _toy_batches()
    net = _toy_net()
    hist = tn.fit_network(net, batch, loss, n_train, n_val, tn.TrainConfig(epochs=40, batch_size=64, learning_rate=0.02, patience=40, grad_clip=1.0))
    assert hist.val_loss[hist.best_epoch] == min(hist.val_loss) and hist.val_loss[-1] < 0.5 * hist.val_loss[0]
    assert len(hist.train_loss) == len(hist.lr) == len(hist.grad_norm) == len(hist.epoch_seconds) and hist.n_params == 41 and hist.device == "cpu" and hist.seconds > 0
    assert hist.peak_memory_mb is None or hist.peak_memory_mb > 0 and hist.to_dict()["epochs_run"] == 40


def test_it_stops_when_the_validation_loss_stops_improving():
    batch, loss, n_train, n_val = _toy_batches(noise=5.0)
    hist = tn.fit_network(_toy_net(), batch, loss, n_train, n_val, tn.TrainConfig(epochs=200, batch_size=64, learning_rate=0.05, patience=3, weight_decay=0.0))
    assert hist.stopped_early and len(hist.train_loss) < 200


def test_gradient_clipping_bounds_the_gradient_the_optimiser_sees_and_a_diverging_run_is_refused(monkeypatch):
    seen = []

    class Spy(torch.optim.AdamW):
        def step(self, closure=None):
            seen.append(float(torch.sqrt(sum((p.grad ** 2).sum() for g in self.param_groups for p in g["params"] if p.grad is not None))))
            return super().step(closure)

    monkeypatch.setattr(torch.optim, "AdamW", Spy)
    batch, loss, n_train, n_val = _toy_batches()
    scaled = lambda net, b: 1e4 * loss(net, b)                                                    # noqa: E731  a loss with huge gradients
    peaks = {}
    for clip in (0.0, 0.1):
        seen.clear()
        hist = tn.fit_network(_toy_net(), batch, scaled, n_train, n_val, tn.TrainConfig(epochs=1, batch_size=480, learning_rate=0.01, grad_clip=clip, patience=5))
        peaks[clip] = (max(seen), hist.grad_norm[0])
    assert peaks[0.0][0] > 100 and peaks[0.1][0] <= 0.1 + 1e-4                                    # the optimiser saw the clipped gradient ...
    assert peaks[0.1][1] > 100                                                                    # ... while the history records the norm before clipping
    with pytest.raises(ValueError, match="diverged"):
        tn.fit_network(_toy_net(), batch, lambda net, b: loss(net, b) * float("nan"), n_train, n_val, tn.TrainConfig(epochs=2))


def test_checkpoints_hold_the_best_and_the_last_epoch_and_a_run_resumes_from_them(tmp_path):
    batch, loss, n_train, n_val = _toy_batches()
    net = _toy_net()
    cfg = tn.TrainConfig(epochs=4, batch_size=64, learning_rate=0.02, patience=10, checkpoint_dir=str(tmp_path))
    hist = tn.fit_network(net, batch, loss, n_train, n_val, cfg)
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    other = _toy_net()
    blob = tn.load_checkpoint(tmp_path / "best.pt", other)
    assert blob["epoch"] == hist.best_epoch and all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), other.state_dict().values()))
    more = tn.fit_network(_toy_net(), batch, loss, n_train, n_val, dataclasses.replace(cfg, epochs=7, resume=True))
    assert len(more.train_loss) == 3                                                              # epochs 4, 5 and 6: it picked up after the last saved epoch


def test_each_refit_checkpoints_into_a_folder_of_its_own_so_resuming_never_skips_a_later_fits_training(ds, tmp_path):
    fit_on(ds, "gru", checkpoint_dir=str(tmp_path), resume=True)
    pos = ds.positions()
    rows = np.flatnonzero((pos >= 80) & (pos < 800) & ds.Y[ds.target].notna().to_numpy())               # the next walk-forward fit: a longer window
    later = md.make_model("gru", params_for("gru", checkpoint_dir=str(tmp_path), resume=True), "regression", 3)
    later.bind(ds)
    later.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos[rows])
    folders = sorted(p.name for p in tmp_path.iterdir())
    assert len(folders) == 2 and all(f.startswith("gru-") for f in folders) and all((tmp_path / f / "last.pt").exists() for f in folders)
    assert len(later.history_.train_loss) >= 1, "resumed from the previous window's finished run: the fit trained nothing"


def test_device_choice_falls_back_to_the_cpu_and_says_so():
    assert tn.resolve_device("cpu")[0].type == "cpu" and tn.resolve_device("auto")[0].type in ("cpu", "cuda")
    dev, note = tn.resolve_device("cuda")
    assert dev.type == "cuda" or "CPU" in note
    with pytest.raises(ValueError, match="device"):
        tn.resolve_device("tpu")


# ------------------------------------------------------------------------------------------------------------ every network
@pytest.mark.parametrize("arch", ["lstm", "gru", "tcn", "patchtst", "nbeats", "nhits", "tft"])
def test_each_forecaster_maps_a_window_to_one_output_per_horizon(arch):
    net = build_net(arch, 6, 12, 2, params_for(arch))
    out = net(torch.randn(5, 12, 6))
    assert out.shape == (5, 6 if arch == "tft" else 2) and torch.isfinite(out).all()


def test_a_tcn_output_does_not_depend_on_later_steps():
    net = build_net("tcn", 4, 10, 1, {"hidden": 8, "layers": 2, "kernel_size": 3, "dropout": 0.0}).eval()
    x = torch.randn(3, 10, 4)
    y = x.clone()
    y[:, 7:] = torch.randn(3, 3, 4)
    h1 = net.blocks(x.transpose(1, 2))
    h2 = net.blocks(y.transpose(1, 2))
    assert torch.allclose(h1[:, :, :7], h2[:, :, :7], atol=1e-6) and not torch.allclose(h1[:, :, 7:], h2[:, :, 7:])


@pytest.mark.parametrize("kind", ALL)
def test_every_network_trains_predicts_and_survives_serialization(ds, kind, tmp_path):
    model, rows = fit_on(ds, kind)
    test = ds.X.iloc[np.flatnonzero((ds.positions() >= 720) & (ds.positions() < 790))]
    pred = model.predict(test)
    assert pred.shape == (len(test),) and np.isfinite(pred).all() and pred.std() > 0
    diag = model.diagnostics()
    assert diag["arch"] == kind and diag["epochs_run"] >= 1 and len(diag["train_loss"]) == len(diag["val_loss"]) and diag["n_params"] > 0 and diag["device"] == "cpu"
    imp = model.importance()
    assert set(imp.index) == set(ds.features) and imp.sum() == pytest.approx(1.0, abs=1e-4)
    again = md.Model.from_bytes(model.to_bytes()).bind(ds)
    assert np.allclose(again.predict(test), pred, atol=1e-5)
    meta = dm.save_model(model, tmp_path / "m.bin")
    assert meta["kind"] == kind and len(meta["sha256"]) == 64 and "torch" in meta["versions"]
    assert np.allclose(dm.load_model(tmp_path / "m.bin").bind(ds).predict(test), pred, atol=1e-5)
    (tmp_path / "m.bin").write_bytes((tmp_path / "m.bin").read_bytes()[:-5] + b"xxxxx")
    with pytest.raises(ValueError, match="checksum"):
        dm.load_model(tmp_path / "m.bin")


@pytest.mark.parametrize("kind", ["lstm", "tcn", "patchtst", "tft", "nbeats", "nhits", "vae"])
def test_a_bound_network_scores_a_day_from_the_features_up_to_that_day_only(ds, kind):
    model, _ = fit_on(ds, kind)
    pos = ds.positions()
    cut = 760
    rows = ds.X.iloc[np.flatnonzero((pos >= 740) & (pos <= cut))]
    before = model.predict(rows)
    X2 = ds.X.copy()
    later = pos > cut
    X2.loc[later] = np.random.default_rng(0).normal(size=(int(later.sum()), X2.shape[1]))              # every feature after the cut is scrambled
    model.bind(dataclasses.replace(ds, X=X2))
    assert np.allclose(model.predict(rows), before, atol=1e-6)
    after = np.flatnonzero((pos >= cut + 1) & (pos <= cut + 5))
    assert not np.allclose(model.predict(X2.iloc[after]), model.predict(ds.X.iloc[after]))                          # the model does read the rows it is given


def test_validation_days_come_after_the_training_days_with_a_purge_between(ds):
    model = md.make_model("lstm", params_for("lstm", val_fraction=0.2), "regression", 0).bind(ds)
    pos = np.repeat(np.arange(500), 8)
    train, val = model._split(pos, np.random.default_rng(0))
    assert pos[train].max() <= pos[val].min() - 1 - (ds.lag + ds.horizon) and len(val) > 0


def test_the_training_window_is_the_only_data_the_scaler_and_the_targets_know(ds):
    model, rows = fit_on(ds, "gru")
    assert model.n_train_ == len(rows)
    train_X = ds.X.iloc[rows][model.features_].to_numpy(dtype=np.float32)
    assert model.scaler_.mu_.shape == (len(ds.features),) and np.all(np.abs(model.scaler_.mu_) < np.nanmax(np.abs(train_X)) + 1e-6)


# ------------------------------------------------------------------------------------------------------------ horizons, latent codes, intervals
def test_extra_horizons_train_extra_heads_and_must_be_shorter_than_the_datasets(ds):
    model, _ = fit_on(ds, "lstm", horizons="1,3")
    heads = model.predict_extras(ds.X.iloc[:200])["heads"]
    assert list(heads.columns) == ["h1", "h3", f"h{ds.horizon}"] and np.allclose(heads[f"h{ds.horizon}"], model.predict(ds.X.iloc[:200]))
    with pytest.raises(ValueError, match="shorter"):
        fit_on(ds, "lstm", horizons=str(ds.horizon))
    with pytest.raises(ValueError, match="dataset"):
        fit_on(ds, "lstm", bound=False, horizons="1")


def test_the_temporal_fusion_transformer_gives_an_ordered_interval_and_its_attention(ds):
    model, _ = fit_on(ds, "tft")
    d = model.predict_extras(ds.X.iloc[:300])["distribution"]
    assert (d["lower"] <= d["mean"]).all() and (d["mean"] <= d["upper"]).all() and (d["std"] >= 0).all()
    diag = model.diagnostics()
    assert len(diag["attention"]) == TINY["lookback"] and sum(diag["attention"]) == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError, match="alpha_low"):
        md.make_model("tft", params_for("tft", alpha_low=0.7))


@pytest.mark.parametrize("kind", ["autoencoder", "vae"])
def test_an_autoencoder_exposes_its_latent_code_and_the_others_do_not(ds, kind):
    model, _ = fit_on(ds, kind, latent_dim=3)
    z = model.latent_features(ds.X.iloc[:120])
    assert z.shape == (120, 3) and list(z.columns) == ["z0", "z1", "z2"] and z.std().min() > 0
    assert model.diagnostics()["reconstruction_loss"] > 0
    other, _ = fit_on(ds, "gru")
    with pytest.raises(ValueError, match="no latent code"):
        other.latent_features(ds.X.iloc[:10])


def test_a_walk_forward_run_carries_the_curves_and_extras_of_the_network(ds, cfg):
    res = tr.walk_forward_train(ds, {"kind": "tft", "params": params_for("tft", horizons="1")}, cfg["split"], 0)
    assert len(res.diagnostics) == len(res.models) - res.models.count(None) and res.diagnostics[0]["train_loss"]
    assert set(res.extras) == {"distribution", "heads"} and res.extras["heads"].columns.tolist() == ["h1", f"h{ds.horizon}"]
    assert res.extras["distribution"].index.isin(res.predictions.index).all()
    assert np.allclose(res.extras["distribution"]["mean"], res.predictions.reindex(res.extras["distribution"].index))


def test_unavailable_or_invalid_settings_are_refused_before_any_training():
    with pytest.raises(ValueError, match="invalid setting"):
        md.make_model("lstm", {"dropout": 1.5})
    with pytest.raises(ValueError, match="no parameter"):
        md.make_model("lstm", {"kernel_size": 3})
    with pytest.raises(ValueError, match="regression"):
        md.make_model("lstm", {}, "classification")
