"""Allocation studies in Research Sessions: a finished allocation experiment is saved from the server's own record, carries the construction vocabulary of the portfolio lab (so that it can be
compared with portfolio studies on like terms), is kept once however often it is saved, and a comparison says what differs (allocator, risk window, currency) instead of ranking unlike studies."""

from __future__ import annotations

import warnings

import pytest

from src.research import store as rs
from tests.ml_helpers import make_bundle
from tests.test_ml_portfolio_allocation_api import BASE
from tests.test_webapp_assets_ml import _app, finished

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("alloc_research")
    bundle = make_bundle(n_assets=3, n=1500, seed=5, alpha=1.0, macro=False)
    return _app(tmp, lambda config: bundle, root=tmp)


@pytest.fixture(scope="module")
def parent(app):
    return finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "three stocks", "mode": "development"}))


def _child(app, parent, name, **alloc):
    return finished(app, app.extra_post["/api/ml/rebacktest"]({"id": parent["id"], "name": name, "signal": {"method": "portfolio_allocation", "holding_period": 1, "allocation": alloc}}))


def _save(app, exp, **body):
    return app.extra_post["/api/research/artifact/save"]({"experiment": exp["id"], **body})


def test_a_saved_allocation_carries_its_construction_and_is_kept_once(app, parent):
    erc = _child(app, parent, "erc", allocator="erc")
    first = _save(app, erc, new_session="Allocation studies", title="ERC on three stocks")
    art = first["artifact"]
    assert art["kind"] == "ml_experiment" and art["evidence"] == "walk_forward" and art["title"] == "ERC on three stocks"
    study = art["config"]["study"]
    assert study["allocator"] == "erc" and study["cov_method"] and study["holding_days"] == 5 and art["config"]["allocation"]["config_hash"]
    assert art["benchmark"] == "Equal weight" and art["context"]["costs"]["cost_bps"] >= 0 and art["context"]["assets"], "costs, benchmark and the assets are what a comparison checks"
    full = app.extra_get["/api/research/artifact"]({"id": art["id"]})
    full = full.get("artifact", full)
    assert full["validation"]["point_in_time"] and "forecast" in full["validation"]
    assert any("Portfolio allocation by" in n for n in art["notes"])
    again = _save(app, erc, session=first["session"])
    assert again.get("duplicate") is True and again["artifact"]["id"] == art["id"], "the same run is one study however often the button is pressed"


def test_allocations_with_the_same_construction_compare_without_a_construction_block(app, parent):
    ivol = _child(app, parent, "inverse vol", allocator="inverse_vol")
    erc = _child(app, parent, "erc2", allocator="erc")
    s = _save(app, ivol, new_session="Same construction")["session"]
    a = _save(app, ivol, session=s)["artifact"]
    b = _save(app, erc, session=s)["artifact"]
    out = app.extra_post["/api/research/compare"]({"artifacts": [a["id"], b["id"]]})
    assert "different_construction" not in repr(out), "the two differ in allocator only; the window, risk model and limit are shared"
    assert "kind" in repr(out) and a["id"] in repr(out) and b["id"] in repr(out), "both studies are in the comparison"


def test_a_different_risk_window_is_a_blocked_ranking_and_a_portfolio_study_is_compatible(app, parent, tmp_path):
    base = _child(app, parent, "erc 252", allocator="erc", cov_window=252)
    short = _child(app, parent, "erc 126", allocator="erc", cov_window=126)
    s = _save(app, base, new_session="Different window")["session"]
    a = _save(app, base, session=s)["artifact"]
    b = _save(app, short, session=s)["artifact"]
    out = app.extra_post["/api/research/compare"]({"artifacts": [a["id"], b["id"]]})
    flat = repr(out)
    assert "different_construction" in flat and "lookback" in flat, "a different risk window is a different construction, so the ranking is blocked"
    lab = {"kind": "portfolio_lab", "title": "Max Sharpe lab", "evidence": "historical_simulation", "data_kind": "platform", "context": {"assets": a["context"]["assets"], "start": a["context"]["start"], "end": a["context"]["end"], "costs": {"cost_bps": a["context"]["costs"]["cost_bps"]}},
           "config": {"method": "max_sharpe", "study": {"lookback": 252, "step": 5, "cov_method": a["config"]["study"]["cov_method"], "max_weight": a["config"]["study"]["max_weight"], "risk_aversion": a["config"]["study"]["risk_aversion"], "mu_shrink": None}, "mu_source": "historical mean"},
           "metrics": {"sharpe": 0.5}, "refs": {}, "notes": [], "series": {"dates": ["2020-01-01", "2020-01-02"], "equity": [1.0, 1.01], "benchmarks": {}}}
    lab["config"]["study"]["lookback"] = base_lookback = a["config"]["study"]["lookback"]
    lab["config"]["study"]["step"] = a["config"]["study"]["step"]
    lab["config"]["study"]["risk_aversion"] = a["config"]["study"]["risk_aversion"]
    lab["config"]["study"].pop("mu_shrink")
    added = app.research.store.add_artifact(s, lab)
    out = app.extra_post["/api/research/compare"]({"artifacts": [a["id"], added["id"]]})
    flat = repr(out)
    assert "different_expected_returns" in flat, "an allocation study and a portfolio study are compared as portfolios, and the difference in expected-return source is named"
    assert base_lookback == 252 and "different_construction" not in flat.replace("different_construction_unused", "")
