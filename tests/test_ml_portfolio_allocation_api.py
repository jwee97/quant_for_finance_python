"""The portfolio-allocation routes: the catalogue, the validation of the ``signal.allocation`` block (nothing unknown or out of range gets in), an allocation as a child of a saved experiment, and the comparison route's
selection discipline (the protected segments cannot be selected on and are reported only on request)."""

from __future__ import annotations

import json
import warnings

import pytest

from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "horizon": 5, "target": "fwd_ret_5", "min_history": 60}, "split": {"min_train_days": 300, "refit_every": 63, "validation_fraction": 0.2, "test_fraction": 0.2,
        "forward_fraction": 0.1}, "signal": {"method": "proportional", "holding_period": 1}, "model": {"kind": "ridge"}, "backtest": {"rebalance": "weekly", "target_vol": None}, "explain": False}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("alloc_api")
    bundle = make_bundle(n_assets=2, n=1500, seed=3, alpha=1.0, macro=False)
    return _app(tmp, lambda config: bundle, root=tmp)


@pytest.fixture(scope="module")
def parent(app):
    return finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "two stocks", "mode": "development"}))


def test_the_catalogue_documents_every_allocator_and_says_the_default_is_unchanged(app):
    cat = _safe(app.extra_get["/api/ml/allocation/catalog"]({}))
    keys = [a["key"] for a in cat["allocators"]]
    assert keys == ["equal_weight", "inverse_vol", "erc", "mean_variance", "black_litterman", "hrp", "cvar", "robust_mv", "kelly"]
    assert all(a["objective"] and a["assumptions"] and a["needs"] and a["failure"] for a in cat["allocators"])
    assert "default signal method is unchanged" in cat["note"] and "lw_constant_correlation" in cat["risk_models"]
    json.dumps(cat, allow_nan=False)


@pytest.mark.parametrize("allocation, text", [({"allocatr": "erc"}, "no setting 'allocatr'"), ({"allocator": "magic"}, "allocator must be one of"), ({"risk_aversion": -1}, "risk_aversion"), ({"limits": {"max_weight": 3}}, "max_weight"),
                                              ({"limits": {"leverage_everything": True}}, "no setting 'leverage_everything'"), ({"limits": {"long_only": "yes"}}, "true or false"), ({"forecast_currency": "US DOLLAR"}, "currency code"),
                                              ({"limits": {"risk_budget": {"A": 2.0}}}, "risk_budget"), ("erc", "must be an object")])
def test_the_allocation_block_is_checked_before_anything_runs(app, allocation, text):
    with pytest.raises(ApiError, match=text):
        app.extra_post["/api/ml/rebacktest"]({"id": "x", "signal": {"method": "portfolio_allocation", "allocation": allocation}})


def test_an_allocation_is_a_child_of_the_saved_forecasts_and_leaves_the_parent_alone(app, parent):
    child = finished(app, app.extra_post["/api/ml/rebacktest"]({"id": parent["id"], "name": "inverse vol", "signal": {"method": "portfolio_allocation", "holding_period": 1, "allocation": {"allocator": "inverse_vol", "limits": {"max_weight": 0.9}}}}))
    pa = child["report"]["portfolio_allocation"]
    assert child["parent"] == parent["id"] and pa["allocator"] == "inverse_vol" and pa["counts"]["optimal"] == pa["counts"]["decisions"] > 0
    assert pa["config_hash"] and pa["config"]["cov_method"] and pa["config"]["limits"]["max_weight"] == 0.9, "the method, the risk model and the limits are recorded with the run"
    rec = [x for x in pa["records"] if x.get("weights")][-1]
    assert abs(sum(rec["weights"].values()) - 1.0) < 1e-6 and rec["risk_contributions"] and rec["explain"]["driver"] in ("risk", "constraints", "equal")
    again = _safe(app.extra_get["/api/ml/experiment"]({"id": parent["id"]}))
    assert "portfolio_allocation" not in again["report"] and again["report"]["config"]["signal"]["method"] == "proportional"
    json.dumps(pa, allow_nan=False)


def test_the_comparison_selects_on_validation_and_the_protected_segments_need_a_request(app, parent):
    for body, text in (({"id": parent["id"], "select_on": "test"}, "cannot be used to choose"), ({"id": parent["id"], "evaluate": ["validation"]}, "test and/or forward"), ({"id": parent["id"], "allocators": ["magic"]}, "drawn from"),
                       ({"id": parent["id"], "select_by": "luck"}, "select_by"), ({"id": parent["id"], "bogus": 1}, "bogus")):
        with pytest.raises(ApiError, match=text):
            app.extra_post["/api/ml/allocation/compare"](body)
    plain = finished(app, app.extra_post["/api/ml/allocation/compare"]({"id": parent["id"], "allocation": {"limits": {"max_weight": 0.9}}}))
    assert plain["select_on"] == "validation" and plain["evaluated_segments"] == [] and "equal_weight" in plain["allocators"] and "hrp" in plain["skipped"]
    assert all(set(v["segments"]) == {"validation"} for v in plain["table"].values())
    assert plain["curves"]["segments"] == ["validation"] and plain["prediction_quality"]["validation"]["cross_sectional_rank_ic"] is None
    json.dumps(plain, allow_nan=False)
    asked = finished(app, app.extra_post["/api/ml/allocation/compare"]({"id": parent["id"], "evaluate": ["test"], "allocators": ["equal_weight", "inverse_vol"], "allocation": {"limits": {"max_weight": 0.9}}}))
    assert asked["evaluated_segments"] == ["test"] and all(v["segments"]["test"]["used_for_selection"] is False for v in asked["table"].values())
