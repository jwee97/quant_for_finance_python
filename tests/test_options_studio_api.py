"""The Options Strategy Studio routes: scan, build, scenarios, recommend, validate, and the two backtest modes."""

from __future__ import annotations

import pandas as pd
import pytest

from src.webapp.api import ApiError
from tests.test_webapp_assets_ml import _app, _safe
from tests.test_webapp_labs import _rich


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    return _app(tmp_path_factory.mktemp("studio"), _rich)


def call(app, method, path, **body):
    fn = (app.extra_get if method == "get" else app.extra_post)[path]
    return _safe(fn({"data_mode": "model", "underlying": "AAA", **body}))


def contract(app, i=9, offset=0.0):
    chain = call(app, "get", "/api/options/chain")
    expiry = chain["term"][i]["expiry"] if i < len(chain["term"]) else chain["term"][-1]["expiry"]
    view = call(app, "get", "/api/options/chain", expiry=expiry)["chain"]
    strikes = sorted(r["strike"] for r in view["rows"])
    mid = strikes[len(strikes) // 2]
    return expiry, mid


def test_the_scan_reports_leads_with_quality_and_what_it_cannot_judge(app):
    out = call(app, "get", "/api/options/scan")
    assert out["state"] == "ok" and out["iv_history"]["state"] == "unavailable" and out["realised"]["state"] in ("ok", "unavailable") and out["quality"] and out["reading"]
    with pytest.raises(ApiError, match="observed-only"):
        app.extra_get["/api/options/scan"]({"data_mode": "observed", "underlying": "AAA"})


def test_build_scenarios_validate_and_recommend_on_real_contracts_of_the_chain(app):
    expiry, k = contract(app)
    legs = [{"expiry": expiry, "strike": k, "right": "C", "quantity": 1}, {"expiry": expiry, "strike": k + 5, "right": "C", "quantity": -1}]
    built = call(app, "post", "/api/options/studio/build", legs=legs)
    assert built["totals"]["at_touch"] >= built["totals"]["at_mid"] and built["analysis"]["breakevens"] and built["legs"][0]["bid"] <= built["legs"][0]["ask"]
    lab = call(app, "post", "/api/options/studio/scenarios", legs=legs)
    assert len(lab["scenarios"]) == 5 and set(lab["heatmaps"]) == {"spot_vol", "spot_skew", "spot_time"}
    for s in lab["scenarios"]:
        assert sum(s["attribution"].values()) == pytest.approx(s["pnl"], abs=1e-6)
    ok = call(app, "post", "/api/options/studio/validate", legs=legs, max_loss=1e6)
    assert {c["id"] for c in ok["checks"]} >= {"contracts", "two_sided", "fresh", "spread", "fills", "bounded", "max_loss"} and ok["max_loss"] < 0
    naked = call(app, "post", "/api/options/studio/validate", legs=[{"expiry": expiry, "strike": k, "right": "C", "quantity": -1}], max_loss=500)
    assert not naked["passed"] and any(c["id"] == "bounded" and not c["ok"] for c in naked["checks"])
    rec = call(app, "post", "/api/options/studio/recommend", direction="bearish", vol_view="long", max_loss=3000, horizon_days=30)
    assert rec["ranked"] and all(r["delta_dollars"] < 0 for r in rec["ranked"])
    for path, body, text in (("/api/options/studio/build", {"legs": []}, "legs must be"), ("/api/options/studio/build", {"legs": [{"expiry": expiry, "strike": k + 0.011, "right": "C", "quantity": 1}]}, "not listed"),
                             ("/api/options/studio/build", {"legs": [{"expiry": expiry, "strike": k, "right": "X", "quantity": 1}]}, "right"), ("/api/options/studio/recommend", {"direction": "sideways"}, "direction")):
        with pytest.raises(ApiError, match=text):
            call(app, "post", path, **body)


def test_the_backtest_modes_keep_the_synthetic_validation_apart_from_the_historical_one(app):
    modes = _safe(app.extra_get["/api/options/backtest-modes"]({"data_mode": "observed", "underlying": "SPY"}))
    syn, hist = modes["modes"]
    assert syn["state"] == "needs_model_data" and hist["state"] == "unavailable" and hist["snapshots_needed"] == 60 and "never combined" in modes["ranking"]
    assert _safe(app.extra_get["/api/options/backtest-modes"]({"data_mode": "model", "underlying": "SPY"}))["modes"][0]["state"] == "available"
