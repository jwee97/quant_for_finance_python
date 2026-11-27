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
    assert keys[:9] == ["equal_weight", "inverse_vol", "erc", "mean_variance", "black_litterman", "hrp", "cvar", "robust_mv", "kelly"], "the original nine come first and unchanged; the adapters follow"
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


# ------------------------------------------------------------------------------------------------ holding period, calibration, stabilisers
def test_the_catalogue_lists_the_adapters_presets_and_the_four_time_quantities(app):
    cat = _safe(app.extra_get["/api/ml/allocation/catalog"]({}))
    keys = [a["key"] for a in cat["allocators"]]
    assert {"herc", "nco", "nco_max_sharpe", "near_optimal", "wasserstein_cvar"} <= set(keys)
    herc = next(a for a in cat["allocators"] if a["key"] == "herc")
    assert herc["uses_forecast"] is False and herc["min_assets"] == 3 and "risk-only" in herc["assumptions"].lower()
    assert "conservative" in cat["presets"] and "not an optimum" in cat["presets"]["conservative"]["description"]
    assert cat["defaults"]["risk_aversion"] == 5.0, "the historical default risk aversion is unchanged"
    assert {"single_book", "staggered_tranches"} == set(cat["lifecycles"]) and "partial_pooling" in cat["calibration_methods"] and "hac" in cat["calibration_inference"]
    assert "execution lag" in cat["horizon_note"]


@pytest.mark.parametrize("allocation, text", [({"lifecycle": "forever"}, "lifecycle must be one of"), ({"blend": "fixed"}, "needs a baseline"), ({"calib_method": "magic"}, "calib_method must be one of"), ({"preset": "reckless"}, "preset must be one of"),
                                              ({"max_weight_change": 5}, "max_weight_change"), ({"holding_days": 0}, "holding_days"), ({"allocator": "herc", "baseline": "equal_weight", "blend": "fixed"}, "reads no forecast")])
def test_the_new_allocation_settings_are_checked_before_anything_runs(app, allocation, text):
    with pytest.raises(ApiError, match=text):
        app.extra_post["/api/ml/rebacktest"]({"id": "x", "signal": {"method": "portfolio_allocation", "allocation": allocation}})


def test_a_multi_day_holding_period_runs_as_a_child_with_its_window_recorded(app, parent):
    child = finished(app, app.extra_post["/api/ml/rebacktest"]({"id": parent["id"], "name": "tranches", "signal": {"method": "portfolio_allocation", "holding_period": 10,
                     "allocation": {"allocator": "mean_variance", "lifecycle": "staggered_tranches", "opt_every": 5, "limits": {"max_weight": 0.9}}}}))
    pa = child["report"]["portfolio_allocation"]
    rec = [x for x in pa["records"] if x.get("tranche")][-1]
    assert pa["holding_window_days"] == 10 and rec["horizon"]["tranches"] == 2 and rec["horizon"]["decision_every"] == 5 and rec["tranche"]["of"] == 2
    assert pa["config"]["lifecycle"] == "staggered_tranches" and pa["config"]["holding_days"] == 10
    assert child["report"]["config"]["signal"]["holding_period"] == 10


def test_a_blended_conservative_allocation_records_its_blend_and_a_plain_language_account(app, parent):
    child = finished(app, app.extra_post["/api/ml/rebacktest"]({"id": parent["id"], "name": "conservative", "signal": {"method": "portfolio_allocation", "holding_period": 1,
                     "allocation": {"allocator": "mean_variance", "preset": "conservative", "sensitivity": True}}}))
    pa = child["report"]["portfolio_allocation"]
    assert pa["config"]["risk_aversion"] == 20.0 and pa["config"]["preset"] == "conservative" and pa["config"]["blend"] == "confidence"
    rec = [x for x in pa["records"] if x.get("narrative")][-1]
    assert rec["blend"]["status"] == "blended" and rec["blend"]["baseline"] == "inverse_vol" and 0.0 <= rec["blend"]["alpha"] <= 0.5
    assert set(rec["narrative"]["per_asset"]) == set(rec["weights"]) and "Final weight" in next(iter(rec["narrative"]["per_asset"].values()))
    assert rec["sensitivity"]["risk_aversion"] and rec["concentration"]["effective_n"] is not None and rec["correlation"]
    json.dumps(pa, allow_nan=False)


def test_the_comparison_accepts_named_policies_and_the_calibration_route_compares_methods(app, parent):
    out = finished(app, app.extra_post["/api/ml/allocation/compare"]({"id": parent["id"], "allocators": ["equal_weight", "inverse_vol", "mean_variance"], "allocation": {"limits": {"max_weight": 0.9}},
                   "policies": {"mv_cons": {"label": "MV conservative", "allocator": "mean_variance", "preset": "conservative"}}}))
    assert "mv_cons" in out["allocators"] and out["table"]["mv_cons"]["label"] == "MV conservative" and out["table"]["equal_weight"]["segments"]["validation"]["sortino"] is not None
    assert out["sharpe_ci"] and set(out["table"]["mv_cons"]["segments"]) == {"validation"}
    with pytest.raises(ApiError, match="policy 'x' must be an object"):
        app.extra_post["/api/ml/allocation/compare"]({"id": parent["id"], "policies": {"x": "nope"}})
    cal = finished(app, app.extra_post["/api/ml/allocation/calibration"]({"id": parent["id"]}))
    assert cal["select_on"] == "validation" and "pooled_sqrt_horizon" in cal["variants"] and "protected segments were not read" in cal["note"]
    two = cal["variants"]["partial_pooling_hac"]
    assert two["share_with_pooled_fallback"] == 1.0, "two stocks: partial pooling cannot identify between-asset variance and uses the pooled slope throughout"
    for bad in ({"select_on": "test"}, {"variants": {"a": {"risk_aversion": 3}}}, {"step": 0}):
        with pytest.raises(ApiError):
            app.extra_post["/api/ml/allocation/calibration"]({"id": parent["id"], **bad})
    json.dumps(cal, allow_nan=False)
