"""Deep forecasters: windows are causal, the networks are deterministic and can learn a planted signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src.models.deep_forecast import (TrainSettings, build_network, daily_sigma, normalised_windows, parameter_count,
                                       ridge_window, train_and_predict)


def _returns(n=600, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (n, k)), index=pd.bdate_range("2015-01-01", periods=n), columns=list("ABC"))


def test_windows_end_at_the_position_and_ignore_the_future():
    r = _returns()
    sig = daily_sigma(r, 40)
    positions, assets = np.array([300, 400]), np.array([0, 2])
    X, valid = normalised_windows(r, sig, positions, assets)
    assert X.shape == (2, 252) and valid.all()
    manual = (r.iloc[300 - 251:301, 0] / sig.iloc[300, 0]).clip(-5, 5).to_numpy()
    assert np.allclose(X[0], manual)
    shocked = r.copy()
    shocked.iloc[301:] *= 50.0
    X2, _ = normalised_windows(shocked, daily_sigma(shocked, 40), positions[:1], assets[:1])
    assert np.allclose(X[0], X2[0])                    # the future cannot reach the window
    early, valid_early = normalised_windows(r, sig, np.array([100]), np.array([0]))
    assert not valid_early[0]


@pytest.mark.parametrize("kind", ["patchtst", "tsmixer"])
def test_networks_have_the_declared_shape_and_a_modest_size(kind):
    import torch

    net = build_network(kind, 3)
    out = net(torch.zeros(4, 252), torch.tensor([0, 1, 2, 0]))
    assert out.shape == (4,)
    assert parameter_count(kind, 3) < 40_000
    with pytest.raises(ValueError):
        build_network("mamba", 3)


def _planted(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 252))
    y = 0.8 * X[:, -21:].mean(axis=1) * np.sqrt(21) + 0.3 * rng.normal(size=n)       # the last patch carries the signal
    return X, rng.integers(0, 3, n), y


@pytest.mark.parametrize("kind", ["patchtst", "tsmixer"])
def test_a_network_learns_a_planted_signal_and_is_deterministic(kind):
    X, a, y = _planted()
    split, cut = 3000, 3500
    settings = TrainSettings(max_epochs=40, patience=8)
    args = (kind, X[:split], a[:split], y[:split], X[split:cut], a[split:cut], y[split:cut], X[cut:], a[cut:], 5, 3)
    first = train_and_predict(*args, settings=settings)
    again = train_and_predict(*args, settings=settings)
    assert np.array_equal(first["pred"], again["pred"])
    corr = np.corrcoef(first["pred"], y[cut:])[0, 1]
    assert corr > 0.8, corr
    assert first["val_loss"][first["best_epoch"]] == min(first["val_loss"])


def test_ridge_control_recovers_the_planted_signal_and_pure_noise_has_no_skill():
    X, a, y = _planted()
    assert np.corrcoef(ridge_window(X[:1000], y[:1000], X[1000:], 1000.0), y[1000:])[0, 1] > 0.6
    rng = np.random.default_rng(3)
    noise = rng.normal(size=1000)
    assert abs(np.corrcoef(ridge_window(X[:800], noise[:800], X[800:1000], 1000.0), noise[800:])[0, 1]) < 0.2
