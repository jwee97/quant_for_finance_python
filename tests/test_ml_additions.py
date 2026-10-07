"""Tree ensembles, the Temporal Fusion Transformer, and unsupervised representation learning: causal, valid, and able to learn what they should."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.models import representation as rep
from src.models.deep_forecast import TrainSettings, build_network, train_and_predict
from src.strategies.ml import LEARNERS, MLRidge, make_tree_learner

warnings.filterwarnings("ignore")
load_library()


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(1)
    n = 2000
    idx = pd.bdate_range("2010", periods=n)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, (n, 6)), axis=0), index=idx, columns=list("ABCDEF"))
    return bundle_from_prices(prices, name="ml-additions")


# ------------------------------------------------------------------------------------------------------------------ trees
@pytest.mark.parametrize("learner", ["hgb", "random_forest", "extra_trees"])
def test_tree_models_are_causal_and_forecast(bundle, learner):
    model = MODELS.create("ml_trees", learner=learner, n_estimators=30, min_train=756, refit_every=252)
    assert check_causality(model, bundle, cutoff=bundle.index[1600])["ok"]
    f = model.forecast(bundle)
    valid = f.mean.stack().dropna()
    assert len(valid) > 4000 and np.isfinite(valid).all() and f.mean.index.equals(bundle.index)
    assert model.score(bundle).stack().dropna().shape == valid.shape


@pytest.mark.parametrize("learner", ["lightgbm", "xgboost"])
def test_optional_boosting_libraries_work_when_installed_and_fail_clearly_otherwise(bundle, learner):
    pytest.importorskip(learner)
    model = MODELS.create("ml_trees", learner=learner, n_estimators=30, min_train=756, refit_every=252)
    assert check_causality(model, bundle, cutoff=bundle.index[1600])["ok"] and model.forecast(bundle).mean.stack().dropna().shape[0] > 4000


def test_missing_optional_learner_raises_an_actionable_error(monkeypatch):
    import builtins

    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "catboost":
            raise ImportError("no catboost")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="boosting"):
        make_tree_learner("catboost", 10, 3, 0.1, 10, 0)


def test_tree_learner_validation_and_the_ridge_is_unchanged(bundle):
    assert set(LEARNERS) == {"hgb", "random_forest", "extra_trees", "lightgbm", "xgboost", "catboost"}
    for kwargs in ({"learner": "nope"}, {"n_estimators": 0}, {"learning_rate": 0.0}, {"max_depth": 0}):
        with pytest.raises(ValueError):
            MODELS.create("ml_trees", **kwargs)
    with pytest.raises(ValueError):
        make_tree_learner("bogus", 1, 1, 0.1, 1, 0)
    ridge = MLRidge(min_train=756)
    assert ridge.forecast(bundle).mean.stack().dropna().shape[0] > 4000
    sub = MODELS.create("ml_trees", min_train=756)
    assert isinstance(sub, MLRidge) and sub._fit.__func__ is not MLRidge._fit


def test_trees_can_learn_an_interaction_a_linear_model_cannot():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 2))
    y = np.sign(X[:, 0] * X[:, 1]) * 0.5 + rng.normal(0, 0.5, 4000)                            # pure interaction: no linear relationship at all
    from sklearn.linear_model import Ridge

    tree = make_tree_learner("hgb", 100, 3, 0.1, 50, 0).fit(X[:3000], y[:3000])
    lin = Ridge(alpha=1.0).fit(X[:3000], y[:3000])
    assert np.corrcoef(tree.predict(X[3000:]), y[3000:])[0, 1] > 0.3 and abs(np.corrcoef(lin.predict(X[3000:]), y[3000:])[0, 1]) < 0.1


# ----------------------------------------------------------------------------------------------------------------------- TFT
def test_tft_shapes_variable_selection_and_determinism():
    import torch

    torch.manual_seed(0)
    net = build_network("tft", 5, 252)
    x, a = torch.randn(8, 252), torch.randint(0, 5, (8,))
    net.eval()
    y1, y2 = net(x, a), net(x, a)
    assert y1.shape == (8,) and torch.allclose(y1, y2) and torch.isfinite(y1).all()
    assert net.last_selection.shape == (3,) and float(net.last_selection.sum()) == pytest.approx(1.0, abs=1e-5)
    assert not torch.allclose(net(x, a), net(x, (a + 1) % 5))                                 # the static asset embedding matters
    assert sum(p.numel() for p in net.parameters()) < 50000
    with pytest.raises(ValueError):
        build_network("tft", 5, 252, family={"stride": 5})
    with pytest.raises(ValueError):
        build_network("tft", 5, 252, family={"d_model": 15, "heads": 2})


def test_tft_learns_a_planted_pattern_and_the_plugin_accepts_it():
    rng = np.random.default_rng(0)
    n = 900
    X = rng.normal(size=(n, 252)).astype("float32")
    y = (X[:, -20:].mean(axis=1) * 4.0).astype("float32")
    a = rng.integers(0, 5, n)
    out = train_and_predict("tft", X[:600], a[:600], y[:600], X[600:750], a[600:750], y[600:750], X[750:], a[750:], 0, 5, {"d_model": 16, "heads": 2}, TrainSettings(max_epochs=10, patience=4))
    assert np.corrcoef(out["pred"], y[750:])[0, 1] > 0.2 and out["val_loss"][-1] < out["val_loss"][0]
    from src.strategies.deep import KINDS

    assert "tft" in KINDS and MODELS.create("deep_window", kind="tft").kind == "tft"
    with pytest.raises(ValueError):
        MODELS.create("deep_window", kind="nope")


# ------------------------------------------------------------------------------------------------------- representation learning
def _two_regimes(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(252) / 252
    X = np.vstack([rng.normal(0, 1, (n // 2, 252)), rng.normal(0, 1, (n // 2, 252)) + np.sin(2 * np.pi * t * 3) * 1.5]).astype("float32")
    return X, np.r_[np.zeros(n // 2), np.ones(n // 2)]


def test_augmentation_and_the_contrastive_loss_behave():
    import torch

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(32, 252, generator=gen)
    aug = rep.augment(x, gen)
    assert aug.shape == x.shape and not torch.allclose(aug, x) and ((aug == 0.0).sum(dim=1) >= 25).all()          # a stretch of about 10% of the days is masked
    z = torch.randn(64, 8, generator=gen)
    aligned = rep.nt_xent(z, z + 0.01 * torch.randn_like(z))
    shuffled = rep.nt_xent(z, z[torch.randperm(64, generator=gen)])
    assert float(aligned) < 0.5 * float(shuffled)


def test_encoders_train_without_labels_and_are_deterministic():
    X, y = _two_regimes()
    ae = rep.train_encoder(X, "autoencoder", 8, 64, 12, seed=0)
    assert ae["loss"][-1] < 0.8 * ae["loss"][0] and ae["encode"](X[:5]).shape == (5, 8)
    again = rep.train_encoder(X, "autoencoder", 8, 64, 12, seed=0)
    assert np.allclose(ae["encode"](X[:20]), again["encode"](X[:20])) and not np.allclose(ae["encode"](X[:20]), rep.train_encoder(X, "autoencoder", 8, 64, 12, seed=1)["encode"](X[:20]))
    ct = rep.train_encoder(X, "contrastive", 8, 64, 10, seed=0)
    assert ct["loss"][-1] < ct["loss"][0]
    from sklearn.linear_model import LogisticRegression

    E = ct["encode"](X)
    idx = np.random.default_rng(1).permutation(len(X))
    probe = LogisticRegression(max_iter=500).fit(E[idx[:900]], y[idx[:900]])
    assert probe.score(E[idx[900:]], y[idx[900:]]) > 0.9                                         # the label-free embedding separates the two regimes
    with pytest.raises(ValueError):
        rep.train_encoder(X, "nope")


@pytest.mark.parametrize("kind", ["autoencoder", "contrastive"])
def test_representation_model_is_causal_and_forecasts(bundle, kind):
    model = MODELS.create("deep_representation", kind=kind, epochs=4, min_train=756, refit_every=504)
    assert check_causality(model, bundle, cutoff=bundle.index[1600])["ok"]
    f = model.forecast(bundle)
    assert f.mean.stack().dropna().shape[0] > 2000 and np.isfinite(f.mean.stack().dropna()).all()
    for bad in ({"kind": "x"}, {"embed_dim": 1}, {"epochs": 0}, {"min_train": 100}):
        with pytest.raises(ValueError):
            MODELS.create("deep_representation", **bad)
