"""Stage 33 models: shapes, determinism, a planted signal, causal graph construction, MC dropout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src.models.deep_forecast import TrainSettings, build_network, parameter_count, train_and_predict
from src.models.graph_forecast import build_graph_network, correlation_neighbours, train_and_predict_graph

CFG = {"nbeats": {"blocks": 3, "layers": 4, "width": 64, "dropout": 0.2, "asset_embedding": 8},
       "nhits": {"pools": [12, 3, 1], "layers": 3, "width": 64, "dropout": 0.2, "asset_embedding": 8},
       "timemixer": {"scales": [1, 3, 7], "width": 32, "dropout": 0.2, "asset_embedding": 8}}
FAST = TrainSettings(1e-3, 1e-2, 128, 12, 4)


@pytest.mark.parametrize("kind", list(CFG))
def test_family_networks_have_the_declared_shape(kind):
    import torch

    net = build_network(kind, 15, family=CFG[kind])
    out = net(torch.randn(7, 252), torch.arange(7))
    assert out.shape == (7,) and torch.isfinite(out).all()
    assert 5_000 < parameter_count(kind, 15, family=CFG[kind]) < 300_000


def test_nhits_pools_must_divide_the_window():
    with pytest.raises(ValueError):
        build_network("nhits", 15, family={"pools": [5]})


def test_zeroed_nbeats_forecasts_zero():
    """A network with every parameter zeroed forecasts exactly zero for any window: nothing leaks in from the input or the asset."""
    import torch

    net = build_network("nbeats", 3, family={**CFG["nbeats"], "blocks": 1, "dropout": 0.0})
    for p in net.parameters():
        torch.nn.init.zeros_(p)
    assert torch.allclose(net(torch.randn(4, 252), torch.arange(4) % 3), torch.zeros(4))


@pytest.mark.parametrize("kind", ["nbeats", "nhits", "timemixer"])
def test_family_networks_learn_a_planted_signal_and_are_deterministic(kind):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(900, 252))
    a = rng.integers(0, 5, 900)
    y = 0.8 * np.tanh(X[:, -21:].mean(axis=1) * 4.5) + 0.3 * rng.normal(size=900)
    run = lambda: train_and_predict(kind, X[:600], a[:600], y[:600], X[600:750], a[600:750], y[600:750], X[750:], a[750:], 3, 5, {"family": CFG[kind]}, FAST, mc_samples=6)
    r1, r2 = run(), run()
    assert np.allclose(r1["pred"], r2["pred"]) and np.allclose(r1["mc_pred"], r2["mc_pred"])
    assert np.corrcoef(r1["pred"], y[750:])[0, 1] > 0.3
    assert r1["mc_pred"].shape == (6, 150) and r1["mc_pred"].std(axis=0).min() > 0     # dropout is on: passes differ


def test_mc_samples_default_does_not_touch_the_stage27_path():
    rng = np.random.default_rng(2)
    X, a, y = rng.normal(size=(300, 252)), rng.integers(0, 3, 300), rng.normal(size=300)
    base = train_and_predict("tsmixer", X[:200], a[:200], y[:200], X[200:250], a[200:250], y[200:250], X[250:], a[250:], 5, 3, None, FAST)
    assert "mc_pred" not in base
    again = train_and_predict("tsmixer", X[:200], a[:200], y[:200], X[200:250], a[200:250], y[200:250], X[250:], a[250:], 5, 3, None, FAST, mc_samples=4)
    assert np.allclose(base["pred"], again["pred"])           # the MC passes come after the deterministic prediction and use a separate seed


def test_correlation_neighbours_are_causal_and_find_the_planted_pair():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, (400, 1))
    data = rng.normal(0, 0.01, (400, 6))
    data[:, 1] = base[:, 0] + 0.1 * data[:, 1]
    data[:, 0] = base[:, 0] + 0.1 * data[:, 0]
    r = pd.DataFrame(data, index=pd.bdate_range("2018-01-01", periods=400), columns=list("ABCDEF"))
    nb = correlation_neighbours(r, np.array([300]), window=252, k=2)
    assert nb.shape == (1, 6, 2) and nb[0, 0, 0] == 1 and nb[0, 1, 0] == 0
    shocked = r.copy()
    shocked.iloc[301:] = rng.normal(size=(99, 6))
    assert (correlation_neighbours(shocked, np.array([300]), window=252, k=2) == nb).all()    # nothing after the position can change the graph
    assert not any(a in nb[0, a] for a in range(6))                                              # no self-loops when enough neighbours exist


def test_graph_network_shapes_masking_and_learning():
    import torch

    n, f, k = 6, 4, 2
    rng = np.random.default_rng(4)
    days = 260
    F = rng.normal(size=(days, n, f))
    nbr = np.stack([np.stack([[(a + 1) % n, (a + 2) % n] for a in range(n)]) for _ in range(days)])
    y = 0.9 * np.tanh(F[:, :, 0]) + 0.3 * rng.normal(size=(days, n))
    y[:20, 0] = np.nan                                                    # masked targets must not break the loss
    net = build_graph_network(n, f, hidden=8, heads=2)
    assert net(torch.as_tensor(F[:3], dtype=torch.float32), torch.as_tensor(nbr[:3])).shape == (3, n)
    run = lambda: train_and_predict_graph(F[:160], nbr[:160], y[:160], F[160:200], nbr[160:200], y[160:200], F[200:], nbr[200:], 2, n, FAST, {"hidden": 8, "heads": 2}, 5)
    r1, r2 = run(), run()
    assert np.allclose(r1["pred"], r2["pred"]) and r1["mc_pred"].shape == (5, 60, n)
    assert np.corrcoef(r1["pred"].ravel(), y[200:].ravel())[0, 1] > 0.3


def test_stage33_tables_follow_the_declared_rules():
    """The decision columns in the generated table must be exactly what the declared rule implies (skips when the stage has not run)."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "reports" / "tables" / "stage33_forecast_tests.csv"
    if not path.exists():
        pytest.skip("stage 33 not run")
    t = pd.read_csv(path, index_col=0)
    assert list(t.index) == ["nbeats", "nhits", "timemixer", "gat"] and (t["n_origins"] == t["n_origins"].iloc[0]).all()
    assert (t["passes"] == (t["bh_significant"] & (t["mean_crps_difference"] < 0))).all()
    b = pd.read_csv(path.with_name("stage33_bayesian.csv"), index_col=0)
    assert (b["mean_sigma_ratio"] >= 1.0).all()                # the epistemic term can only widen


def test_stage34_is_descriptive_and_pins_its_model():
    from pathlib import Path
    import json

    tables = Path(__file__).resolve().parents[1] / "reports" / "tables"
    if not (tables / "stage34_tests.csv").exists():
        pytest.skip("stage 34 not run")
    meta = json.loads((tables / "stage34_model.json").read_text())
    assert meta["model"] == "amazon/chronos-bolt-small" and meta["hub_revision"]
    t = pd.read_csv(tables / "stage34_tests.csv", index_col=0)
    assert {"historical_mean", "stage19_ridge"} <= set(t.index)
    registry = (tables.parents[1] / "experiments" / "registry.jsonl").read_text().splitlines()
    mine = [json.loads(line) for line in registry if '"stage34_foundation"' in line]
    assert mine and all(e["decision"] == "record" for e in mine)          # descriptive by declaration: never retain/reject
