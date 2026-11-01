"""P0-06 and P0-07: an absent asset takes no part in message passing, at any layer, and the graph is built from returns up to the fit cutoff only."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.ml import config as mc, datasets as dsm, models as md                      # noqa: E402
from src.ml.graph import GraphNet, build_adjacency                                   # noqa: E402
from tests.ml_helpers import make_bundle                                             # noqa: E402

KINDS = ["gcn", "gat"]


def adjacency(n, seed=0):
    rng = np.random.default_rng(seed)
    a = np.triu(rng.uniform(0.1, 1.0, (n, n)) * (rng.random((n, n)) < 0.6), 1)
    return a + a.T


def net(kind, adj, seed=0):
    torch.manual_seed(seed)
    return GraphNet(kind, 3, adj, hidden=8, layers=2, dropout=0.0, heads=2 if kind == "gat" else 1).eval()


def params_of(model):
    return {k: v for k, v in model.state_dict().items() if k not in ("A_hat", "mask")}


@pytest.mark.parametrize("kind", KINDS)
def test_garbage_in_an_absent_nodes_slot_changes_no_present_node_and_not_the_pooled_output(kind):
    adj, rng = adjacency(7), np.random.default_rng(1)
    m = net(kind, adj)
    x = torch.tensor(rng.normal(size=(5, 7, 3)), dtype=torch.float32)
    valid = torch.tensor(rng.random((5, 7)) < 0.6)
    valid[:, :2] = True
    node, pooled = m(x, valid)
    for junk in (1e6, -1e6, float("nan")):
        x2 = x.clone()
        x2[~valid] = junk
        node2, pooled2 = m(x2, valid)
        assert torch.equal(node[valid], node2[valid]) and torch.equal(pooled, pooled2) and torch.isfinite(node2[valid]).all()


@pytest.mark.parametrize("kind", KINDS)
def test_with_nodes_absent_the_network_is_the_network_of_the_graph_induced_on_the_present_ones(kind):
    """The independent reference is a second network built on the sub-adjacency of the present nodes with the same weights: GCN degrees and GAT softmaxes must be those of the induced graph."""
    n, rng = 8, np.random.default_rng(2)
    adj = adjacency(n, 3)
    full = net(kind, adj)
    present = np.array([True, False, True, True, False, True, True, False])
    sub = net(kind, adj[np.ix_(present, present)])
    sub.load_state_dict({**sub.state_dict(), **params_of(full)})
    x = torch.tensor(rng.normal(size=(4, n, 3)), dtype=torch.float32)
    valid = torch.tensor(np.tile(present, (4, 1)))
    node, pooled = full(x, valid)
    node_sub, pooled_sub = sub(x[:, present], torch.ones(4, int(present.sum()), dtype=torch.bool))
    assert torch.allclose(node[:, present], node_sub, atol=1e-6) and torch.allclose(pooled, pooled_sub, atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_adding_absent_nodes_to_the_graph_changes_no_present_prediction(kind):
    n, extra, rng = 6, 3, np.random.default_rng(4)
    base_adj = adjacency(n, 5)
    big = np.ones((n + extra, n + extra)) - np.eye(n + extra)
    big[:n, :n] = base_adj                                                               # the new nodes are tied to everything
    small, large = net(kind, base_adj), net(kind, big)
    large.load_state_dict({**large.state_dict(), **params_of(small)})
    x = torch.tensor(rng.normal(size=(3, n, 3)), dtype=torch.float32)
    x_big = torch.cat([x, 50.0 * torch.ones(3, extra, 3)], dim=1)
    valid = torch.ones(3, n, dtype=torch.bool)
    valid_big = torch.cat([valid, torch.zeros(3, extra, dtype=torch.bool)], dim=1)
    node, pooled = small(x, valid)
    node_big, pooled_big = large(x_big, valid_big)
    assert torch.allclose(node, node_big[:, :n], atol=1e-6) and torch.allclose(pooled, pooled_big, atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_a_day_with_no_present_node_is_finite_in_both_directions(kind):
    m = net(kind, adjacency(5, 6)).train()
    x = torch.randn(2, 5, 3, requires_grad=True)
    valid = torch.zeros(2, 5, dtype=torch.bool)
    node, pooled = m(x, valid)
    (node.sum() + pooled.sum()).backward()
    assert torch.isfinite(node).all() and torch.isfinite(pooled).all() and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    valid[0, :1] = True                                                                  # one lone present node attends to itself and nothing else
    node, pooled = m(x, valid)
    assert torch.isfinite(node[0, 0]) and torch.isfinite(pooled).all()


@pytest.fixture(scope="module")
def ds():
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 300}}, defaults=mc.DEFAULTS)
    return dsm.build_dataset(make_bundle(7, 800, seed=61, alpha=1.0), cfg["dataset"], "graph-cutoff")


def fit_graph(data, end_pos):
    pos = data.positions()
    rows = np.flatnonzero((pos >= 100) & (pos <= end_pos) & data.Y[data.target].notna().to_numpy())
    model = md.make_model("gcn", {"epochs": 2, "hidden": 8, "layers": 1, "batch_size": 64, "patience": 2}, "regression", 0).bind(data)
    return model.fit(data.X.iloc[rows], data.Y[data.target].iloc[rows], pos[rows])


def test_the_graph_is_built_from_returns_up_to_the_fit_cutoff_and_records_what_it_used(ds):
    import dataclasses

    first = fit_graph(ds, 500)
    info = first.diagnostics()["graph"]
    assert info["cutoff"] == str(ds.dates[500].date()) and info["asset_order"] == list(ds.assets) and info["fit_window"][1] == info["cutoff"] and info["relationship_version"].startswith("static_group_map:")
    r2 = ds.returns.copy()
    r2.iloc[501:] = np.random.default_rng(0).normal(0, 0.05, r2.iloc[501:].shape)        # a different future
    other = fit_graph(dataclasses.replace(ds, returns=r2), 500)
    assert np.array_equal(first.adjacency_, other.adjacency_), "returns after the cutoff reached the graph"
    r3 = ds.returns.copy()
    r3.iloc[300:500] = np.random.default_rng(1).normal(0, 0.05, r3.iloc[300:500].shape)  # the history it may use does change it
    assert not np.array_equal(first.adjacency_, fit_graph(dataclasses.replace(ds, returns=r3), 500).adjacency_)


def test_the_adjacency_has_no_self_loops_is_symmetric_and_ignores_the_diagonal_of_the_correlation():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(size=(300, 5)), columns=list("ABCDE"))
    a = build_adjacency(r, {c: ("x" if c in "AB" else "y") for c in r}, "both", 2)
    assert np.allclose(a, a.T) and not np.diag(a).any() and a[0, 1] >= 1.0
