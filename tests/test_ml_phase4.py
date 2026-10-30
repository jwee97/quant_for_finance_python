"""Phase 4: graph networks, member ensembles with dynamic and regime weights, and the uncertainty-aware position rules."""

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.ml import config as mc, datasets as dsm, models as md, signals as sg, training as tr   # noqa: E402
from src.ml.ensemble_models import parse_members                                                # noqa: E402
from src.ml.graph import GraphNet, build_adjacency                                              # noqa: E402
from src.utils.optional import unavailable_reason                                               # noqa: E402
from tests.ml_helpers import make_bundle                                                        # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
SPLIT = {"min_train_days": 400, "refit_every": 300}
GRAPH = {"epochs": 4, "hidden": 8, "layers": 2, "batch_size": 32, "patience": 3}


@pytest.fixture(scope="module")
def cfg():
    return mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility", "relative"]}, "split": SPLIT}, defaults=mc.DEFAULTS)


@pytest.fixture(scope="module")
def ds(cfg):
    return dsm.build_dataset(make_bundle(8, 1200, seed=41, alpha=1.0), cfg["dataset"], "phase4")


def fit_rows(ds, lo=100, hi=800):
    pos = ds.positions()
    rows = np.flatnonzero((pos >= lo) & (pos < hi) & ds.Y[ds.target].notna().to_numpy())
    return rows, pos[rows]


# ------------------------------------------------------------------------------------------------------------ graphs
def test_the_adjacency_joins_correlated_assets_and_assets_of_one_group():
    rng = np.random.default_rng(0)
    common = rng.normal(size=(300, 1))
    R = pd.DataFrame(np.hstack([common + 0.1 * rng.normal(size=(300, 2)), rng.normal(size=(300, 4))]), columns=list("ABCDEF"))
    A = build_adjacency(R, {}, "correlation", neighbors=1)
    assert np.allclose(A, A.T) and np.diag(A).sum() == 0 and A[0, 1] > 0.8 and (A > 0).sum(axis=1).min() >= 1
    S = build_adjacency(R, {"A": "x", "B": "x", "C": "y", "D": "y", "E": "y", "F": "z"}, "sector", sector_weight=0.5)
    assert S[0, 1] == 0.5 and S[2, 3] == 0.5 and S[4, 5] == 0.0 and S[0, 2] == 0.0
    with pytest.raises(ValueError, match="sector"):
        build_adjacency(R, {}, "sector")
    assert (build_adjacency(R, {}, "both", neighbors=1) > 0).any()                                    # without labels the sector part is skipped, not refused


@pytest.mark.parametrize("kind", ["gcn", "gat"])
def test_a_graph_layer_reads_its_neighbours_and_nobody_else(kind):
    A = np.zeros((5, 5))
    for i in range(4):
        A[i, i + 1] = A[i + 1, i] = 1.0                                                              # a path 0-1-2-3-4
    net = GraphNet(kind, 3, A, 8, 1, 0.0, heads=2).eval()
    x = torch.randn(1, 5, 3)
    valid = torch.ones(1, 5)
    base = net(x, valid)[0]
    far, near = x.clone(), x.clone()
    far[0, 3] += 5.0
    near[0, 1] += 5.0
    assert torch.allclose(net(far, valid)[0][0, 0], base[0, 0], atol=1e-6) and not torch.allclose(net(near, valid)[0][0, 0], base[0, 0], atol=1e-6)


@pytest.mark.parametrize("kind", ["gcn", "gat"])
def test_a_graph_model_trains_scores_every_asset_and_reports_its_graph(ds, kind, tmp_path):
    rows, pos = fit_rows(ds)
    model = md.make_model(kind, GRAPH, "regression", 1).bind(ds).fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)
    test = ds.X.iloc[np.flatnonzero((ds.positions() >= 820) & (ds.positions() < 860))]
    pred = model.predict(test)
    assert pred.shape == (len(test),) and np.isfinite(pred).all() and pred.std() > 0
    extras = model.predict_extras(test)
    assert extras["portfolio"].shape == (test.index.get_level_values(0).nunique(), 1)
    diag = model.diagnostics()
    assert diag["graph"]["n_edges"] > 0 and diag["epochs_run"] >= 1 and (("attention" in diag) == (kind == "gat"))
    assert model.importance().sum() == pytest.approx(1.0, abs=1e-4)
    again = md.Model.from_bytes(model.to_bytes()).bind(ds)
    assert np.allclose(again.predict(test), pred, atol=1e-5)
    # a day's graph holds the features of that day only: scrambling every later day changes nothing
    pos_all = ds.positions()
    X2 = ds.X.astype("float64")                                                          # pandas 3 refuses to put float64 noise into float32 columns
    later = pos_all > 860
    X2.loc[later] = np.random.default_rng(0).normal(size=(int(later.sum()), X2.shape[1]))
    model.bind(dataclasses.replace(ds, X=X2))
    assert np.allclose(model.predict(test), pred, atol=1e-6)


def test_a_graph_model_needs_the_dataset_and_enough_dates(ds):
    rows, pos = fit_rows(ds)
    with pytest.raises(ValueError, match="bound to a dataset"):
        md.make_model("gcn", GRAPH).fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)
    with pytest.raises(ValueError, match="too few"):
        md.make_model("gcn", GRAPH).bind(ds).fit(ds.X.iloc[:300], ds.Y[ds.target].iloc[:300], ds.positions()[:300])
    with pytest.raises(ValueError, match="graph must be"):
        md.make_model("gcn", {**GRAPH, "graph": "random"})


def test_a_walk_forward_graph_run_has_the_portfolio_forecast(ds, cfg):
    res = tr.walk_forward_train(ds, {"kind": "gcn", "params": GRAPH}, cfg["split"], 0)
    assert "portfolio" in res.extras and res.diagnostics[0]["graph"]["assets"] == list(ds.assets)


# ------------------------------------------------------------------------------------------------------------ ensembles
def test_member_lists_are_parsed_and_checked():
    assert parse_members("ridge, lightgbm[n_estimators=40;learning_rate=0.1],lstm[epochs=2]") == [("ridge", {}), ("lightgbm", {"n_estimators": 40, "learning_rate": 0.1}), ("lstm", {"epochs": 2})]
    with pytest.raises(ValueError, match="cannot read"):
        parse_members("ridge;;lasso")
    with pytest.raises(ValueError, match="between 2 and"):
        md.make_model("model_ensemble", {"members": "ridge"})
    with pytest.raises(ValueError, match="not a model that can be a member"):
        md.make_model("model_ensemble", {"members": "ridge,model_ensemble"})
    with pytest.raises(ValueError, match="no parameter"):
        md.make_model("model_ensemble", {"members": "ridge,lasso[bogus=1]"})
    with pytest.raises(ValueError, match="weighting"):
        md.make_model("model_ensemble", {"members": "ridge,lasso", "weighting": "magic"})


def _need(*pkgs):
    for pkg in pkgs:
        if unavailable_reason(pkg):
            pytest.skip(f"{pkg} is not available")


MEMBERS = "ridge,lasso[alpha=0.05],lgbm_quantile[n_estimators=30;num_leaves=7]"


@pytest.mark.parametrize("weighting", ["equal", "dynamic", "regime"])
def test_an_ensemble_combines_calibrated_members_and_reports_weights_and_a_distribution(ds, cfg, weighting):
    _need("lightgbm")
    res = tr.walk_forward_train(ds, {"kind": "model_ensemble", "params": {"members": MEMBERS, "weighting": weighting}}, cfg["split"], 0)
    w, dist = res.extras["weights"], res.extras["distribution"]
    assert np.allclose(w.sum(axis=1), 1.0) and list(w.columns) == ["ridge", "lasso", "lgbm_quantile"] and (w >= 0).all().all()
    assert (dist["std"] > 0).all() and (dist["lower"] < dist["mean"]).all() and (dist["mean"] < dist["upper"]).all() and (dist["disagreement"] >= 0).all()
    assert np.allclose(dist["mean"], res.predictions.reindex(dist.index))
    assert (w.std() > 1e-6).any() if weighting != "equal" else np.allclose(w, 1 / 3)
    calib = res.diagnostics[0]["calibration"]
    assert [c["member"] for c in calib] == ["ridge", "lasso", "lgbm_quantile"] and all(c["slope"] >= 0 and c["residual_std"] > 0 for c in calib)
    if weighting == "regime":
        assert np.array(res.diagnostics[0]["regime_weights"]).shape == (2, 3) and np.allclose(np.array(res.diagnostics[0]["regime_weights"]).sum(axis=1), 1.0)


def test_dynamic_weights_on_a_day_use_only_outcomes_already_realised_by_that_day(ds):
    _need("lightgbm")
    rows, pos = fit_rows(ds, 100, 700)
    model = md.make_model("model_ensemble", {"members": MEMBERS, "weighting": "dynamic"}, "regression", 0).bind(ds)
    model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)
    block = ds.X.iloc[np.flatnonzero((ds.positions() >= 720) & (ds.positions() < 900))]
    base = model.predict_extras(block)
    cut = ds.dates[800]
    p2 = ds.prices.copy()
    p2.loc[p2.index > cut] = p2.loc[p2.index > cut].to_numpy()[::-1] * 1.7                           # the prices after the cut are replaced
    other = dataclasses.replace(ds, prices=p2, returns=p2.pct_change())
    model.bind(other)
    again = model.predict_extras(block)
    early = base["weights"].index <= cut
    assert early.any() and np.allclose(base["weights"][early], again["weights"][early], atol=1e-9)
    assert np.allclose(model.predict(block.loc[block.index.get_level_values(0) <= cut]), md.Model.from_bytes(model.to_bytes()).bind(ds).predict(block.loc[block.index.get_level_values(0) <= cut]))
    assert not np.allclose(base["weights"][~early], again["weights"][~early])                       # later weights do move once the later outcomes differ


def test_an_ensemble_survives_serialization_and_refuses_what_it_cannot_do(ds):
    _need("lightgbm")
    rows, pos = fit_rows(ds)
    model = md.make_model("model_ensemble", {"members": "ridge,lgbm_quantile[n_estimators=30]", "weighting": "regime"}, "regression", 0).bind(ds)
    model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)
    test = ds.X.iloc[np.flatnonzero((ds.positions() >= 820) & (ds.positions() < 860))]
    again = md.Model.from_bytes(model.to_bytes()).bind(ds)
    assert np.allclose(again.predict(test), model.predict(test)) and model.importance().sum() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="bound to a dataset"):
        md.make_model("model_ensemble", {"members": "ridge,lasso", "weighting": "dynamic"}).fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)
    with pytest.raises(ValueError, match="too few"):
        md.make_model("model_ensemble", {"members": "ridge,lasso", "weighting": "equal"}).fit(ds.X.iloc[:200], ds.Y[ds.target].iloc[:200], ds.positions()[:200])


# ------------------------------------------------------------------------------------------------------------ uncertainty-aware positions
def _panel(n=30, k=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    cols = list("ABCDEF")[:k]
    return pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=cols), pd.DataFrame(rng.uniform(0.5, 2.0, (n, k)), index=idx, columns=cols)


def test_uncertainty_weighted_positions_favour_confident_views_and_keep_the_exposures():
    score, std = _panel()
    w = sg.uncertainty_weighted(score, std, gross=1.0, net=0.0, kappa=0.0, power=1.0)
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-9) and np.allclose(w.abs().sum(axis=1), 1.0)
    flat = sg.uncertainty_weighted(score, std * 0 + 1.0, 1.0, 0.0, 0.0, 1.0)
    assert np.allclose(flat, sg.proportional(score, 1.0, 0.0), atol=1e-9)                              # equal uncertainty: the plain proportional book
    s2, sd2 = score.copy(), std.copy()
    s2.iloc[:, :2] = 1.0
    s2.iloc[:, 2:] = -1.0
    sd2.iloc[:, 0], sd2.iloc[:, 1] = 0.5, 4.0
    sd2.iloc[:, 2:] = 1.0
    ww = sg.uncertainty_weighted(s2, sd2, 1.0, 0.0, 0.0, 1.0)
    assert (ww.iloc[:, 0] > ww.iloc[:, 1]).all()                                                    # the same view, held with less doubt, gets more
    shrunk = sg.uncertainty_weighted(score, std, 1.0, 0.0, kappa=2.0, power=1.0)
    assert (shrunk == 0).sum().sum() > (w == 0).sum().sum()                                         # views on assets the model is unusually unsure about are dropped
    typical = std * 0 + 1.0
    assert np.allclose(sg.uncertainty_weighted(score, typical, 1.0, 0.0, kappa=5.0, power=1.0), sg.uncertainty_weighted(score, typical, 1.0, 0.0, kappa=0.0, power=1.0))      # no excess, no shrinking


def test_uncertainty_mean_variance_is_neutral_scaled_and_uses_only_returns_up_to_the_day():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2021-01-04", periods=300)
    returns = pd.DataFrame(0.01 * rng.normal(size=(300, 6)), index=idx, columns=list("ABCDEF"))
    score, std = pd.DataFrame(rng.normal(size=(300, 6)), index=idx, columns=returns.columns), pd.DataFrame(rng.uniform(0.5, 2, (300, 6)), index=idx, columns=returns.columns)
    w = sg.uncertainty_mean_variance(score, std, returns, gross=1.5, net=0.0, kappa=0.3, lookback=100, rebalance=5)
    live = w.iloc[60:]
    assert np.allclose(live.sum(axis=1), 0.0, atol=1e-9) and np.allclose(live.abs().sum(axis=1), 1.5, atol=1e-9) and w.iloc[:30].abs().sum().sum() == 0
    r2 = returns.copy()
    r2.iloc[200:] = 0.05 * rng.normal(size=(100, 6))
    w2 = sg.uncertainty_mean_variance(score, std, r2, gross=1.5, net=0.0, kappa=0.3, lookback=100, rebalance=5)
    assert np.allclose(w.iloc[:200], w2.iloc[:200]) and not np.allclose(w.iloc[250:], w2.iloc[250:])


def test_the_new_methods_are_registered_and_say_what_they_need():
    score, _ = _panel()
    for method in sg.UNCERTAINTY_METHODS:
        assert method in sg.METHODS
        with pytest.raises(ValueError, match="reports its own uncertainty"):
            sg.build_weights(score, {"method": method}, uncertainty=None)
    with pytest.raises(ValueError, match="returns"):
        sg.build_weights(score, {"method": "uncertainty_mean_variance"}, uncertainty=score.abs() + 1.0, returns=None)
    book = sg.build_weights(score, {"method": "target_weights", "gross": 0.6}, investable=None)
    assert (book.abs().sum(axis=1) <= 0.6 + 1e-9).all() and (book.abs().sum(axis=1) > 0.59).all()                  # a book above the gross exposure is scaled down to it
    small = score * 0.01
    assert np.allclose(sg.build_weights(small, {"method": "target_weights", "gross": 1.0, "max_weight": 1.0}), small)
