"""The Options tab's model endpoints: a calibration of the option models to a chain (ranked, benchmarked against a flat smile, with smiles and implied densities that line up), the session keeping
its fits, every product priced under every model with the identities each must satisfy (put-call parity, digital call plus put, knock-in plus knock-out, early exercise worth at least the
European), the hedging-cost report, and the requests refused before a job starts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import integrate, stats

from src.derivatives import calibration, pde
from src.assets import option_models as om
from src.webapp import api, server
from src.webapp.api import ApiError
from tests.test_webapp import _http
from tests.test_webapp_assets_ml import _app, finished, wait
from tests.test_webapp_labs import _rich

UND = "AAA"
MODELS = ["merton", "nig", "heston"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    return _app(tmp_path_factory.mktemp("models_app"), _rich)


def _run(app, path, **body):
    return finished(app, app.extra_post[path](body))


@pytest.fixture(scope="module")
def fitted(app):
    return _run(app, "/api/models/fit", underlying=UND, models=MODELS)


def _price(app, product, **body):
    return _run(app, "/api/models/price", underlying=UND, models=MODELS, product=product, **body)


def _by_id(report):
    return {r["id"]: r for r in report["benchmarks"] + report["models"]}


# ------------------------------------------------------------------------------------------------------------------------------ the plan
def test_the_plan_lists_the_models_products_and_what_the_forms_choose_among(app):
    plan = app.extra_get["/api/models/plan"]({})
    assert [m["id"] for m in plan["models"]] == ["merton", "kou", "vg", "nig", "heston"] and plan["default_models"] == [m["id"] for m in plan["models"]]
    assert all(m["text"] and m["assumes"] and m["fails"] and m["params"] for m in plan["models"])
    heston = next(m for m in plan["models"] if m["id"] == "heston")
    assert not heston["american"] and next(m for m in plan["models"] if m["id"] == "nig")["american"]
    assert {p["id"] for p in plan["products"]} == {"vanilla", "digital", "barrier", "asian", "american"} and plan["barriers"] == ["down-out", "up-out", "down-in", "up-in"]
    assert plan["defaults"]["paths"] == 50_000 and {h["id"] for h in plan["hedge_rules"]} == {"delta", "leland", "band"}


def test_the_routes_are_in_the_app_and_the_api_version_says_so(app):
    assert {"/api/models/plan"} <= set(app.extra_get) and {"/api/models/fit", "/api/models/price", "/api/models/hedge"} <= set(app.extra_post)
    assert app.catalog()["api_version"] == api.API_VERSION == 12


# ------------------------------------------------------------------------------------------------------------------------------ the fit
def test_the_fit_ranks_the_models_and_the_best_beats_the_flat_smile(fitted):
    json.dumps(fitted, allow_nan=False)
    rows = fitted["models"]
    assert [r["id"] for r in rows] == sorted((r["id"] for r in rows), key=lambda m: next(x["rmse"] for x in rows if x["id"] == m)) and fitted["best"] == rows[0]["id"]
    assert set(r["id"] for r in rows) == set(MODELS) and fitted["flat_rmse"] > 0
    assert rows[0]["rmse"] < 0.5 * fitted["flat_rmse"], "the best model explains most of the smile"
    for r in rows:
        assert set(r["params"]) == set(r["std_errors"]) == set(r["bounds"]) and 0.0 <= r["explained"] <= 1.0 and r["n_quotes"] == fitted["quotes"]["n"] and r["rmse"] <= r["max_error"]
        assert all(lo <= r["params"][k] <= hi for k, (lo, hi) in r["bounds"].items()) and r["n_params"] == len(r["params"])
    assert fitted["quotes"]["n"] >= 30 and len(fitted["quotes"]["expiries"]) >= 3 and fitted["source"] == "simulated"


def test_the_fit_says_it_is_a_model_fitted_to_a_model_and_flags_what_deserves_care(fitted):
    assert any("fitted to another model" in w for w in fitted["warnings"]) and fitted["sources"] == {"quotes": "simulated", "fitted_models": "simulated"}
    assert any("risk-neutral" in n for n in fitted["notes"]) and any("flat-volatility benchmark" in n for n in fitted["notes"])
    merton = next(r for r in fitted["models"] if r["id"] == "merton")
    assert merton["rmse"] > 0.02 > fitted["models"][0]["rmse"], "Merton misses this smile by more than two points, the best model by less"
    assert not any("none of these families" in w for w in fitted["warnings"]), "the warning that no family fits is for when the best model misses too"


def test_the_smiles_line_up_with_the_models_and_the_best_one_follows_the_market(fitted):
    smiles = fitted["smiles"]
    assert 2 <= len(smiles) <= 5 and smiles == sorted(smiles, key=lambda s: s["dte"]), "up to four spread along the curve, and the 30-day expiry the density is drawn for"
    assert any(abs(s["dte"] - fitted["density"]["dte"]) < 1e-6 for s in smiles), "the smile of the expiry the density describes is among them"
    for s in smiles:
        n = len(s["strike"])
        assert n >= 6 and len(s["market"]) == len(s["log_moneyness"]) == n and set(s["models"]) == set(MODELS) and all(len(v) == n for v in s["models"].values())
        assert np.all(np.diff(s["strike"]) > 0) and np.all(np.diff(s["log_moneyness"]) > 0)
    best = fitted["best"]
    gap = [abs(a - b) for s in smiles for a, b in zip(s["market"], s["models"][best]) if b is not None]
    assert np.mean(gap) < 0.02, "the best model is within two volatility points of the quotes on average"


def test_the_implied_densities_are_probability_distributions_with_the_odds_of_a_fall(fitted):
    d = fitted["density"]
    assert 10 <= d["dte"] <= 90 and set(d["curves"]) == set(MODELS) and len(d["bs"]) == len(d["x"]) == 161
    for mid, curve in list(d["curves"].items()) + [("bs", d["bs"])]:
        assert integrate.trapezoid(curve, d["x"]) == pytest.approx(1.0, abs=0.04), mid
    tails = {t["id"]: t for t in d["tails"]}
    assert set(tails) == set(MODELS) | {"bs"} and tails["bs"]["skewness"] == 0.0 and tails["bs"]["excess_kurtosis"] == 0.0
    T, sd = d["dte"] / 365.0, tails["bs"]["stdev"]
    assert sd == pytest.approx(d["atm_iv"] * np.sqrt(T)), "Black-Scholes is the lognormal law at the at-the-money volatility"
    lognormal = stats.norm((fitted["rate"] - fitted["dividend"] - 0.5 * d["atm_iv"] ** 2) * T, sd)
    assert tails["bs"]["p_down_10"] == pytest.approx(lognormal.cdf(np.log(0.9)), abs=1e-12) and tails["bs"]["p_up_10"] == pytest.approx(lognormal.sf(np.log(1.1)), abs=1e-12)
    for t in tails.values():
        assert 0.0 <= t["p_down_20"] <= t["p_down_10"] <= 1.0 and 0.0 <= t["p_up_10"] <= 1.0
    assert tails["heston"]["skewness"] < 0 and tails["heston"]["p_down_10"] > tails["bs"]["p_down_10"], "a market with put skew prices a fall of 10% above what a lognormal says"
    for mid, curve in d["drop_curves"].items():
        assert np.all(np.diff(curve) <= 1e-12) and len(curve) == len(d["drop_pct"]) == 30, mid
    assert np.all(np.diff(d["bs_drop"]) <= 0)


def test_the_session_keeps_the_fits_so_pricing_does_not_refit(app, fitted, monkeypatch):
    calls = []
    real = calibration.calibrate
    monkeypatch.setattr(calibration, "calibrate", lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1])
    report = _price(app, "vanilla")
    assert calls == [] and {r["id"] for r in report["models"]} == set(MODELS), "the same chain and settings are not fitted twice"
    _run(app, "/api/models/fit", underlying=UND, models=["merton"], window=0.2)
    assert calls == ["merton"], "a different window is a different fit"


# ------------------------------------------------------------------------------------------------------------------------------ price
def test_a_vanilla_is_priced_by_every_model_and_the_best_fit_reproduces_the_market(app, fitted):
    call = _price(app, "vanilla", strike_pct=97, call=True)
    json.dumps(call, allow_nan=False)
    c = _by_id(call)
    assert set(c) == {"market", "bs"} | set(MODELS) and call["summary"]["bs_price"] == c["bs"]["price"] and call["summary"]["best"] == fitted["best"]
    assert abs(c[fitted["best"]]["price"] - c["market"]["price"]) < 0.03 * c["market"]["price"], "the model that fits the quotes best reprices the option the smile itself prices"
    assert all(c[m]["price"] > 0 and abs(c[m]["price"] - c["market"]["price"]) < 0.25 * c["market"]["price"] for m in MODELS)
    assert call["summary"]["low"] == min(c[m]["price"] for m in MODELS) and call["summary"]["high"] == max(c[m]["price"] for m in MODELS)
    curve = call["curve"]
    assert len(curve["strike_pct"]) == 21 and set(curve["curves"]) == set(MODELS) | {"bs"} and all(np.all(np.diff(v) < 0) for v in curve["curves"].values()), "a call is worth less the higher the strike"


def test_put_call_parity_holds_under_every_model(app, fitted):
    call, put = _price(app, "vanilla", strike_pct=103, call=True), _price(app, "vanilla", strike_pct=103, call=False)
    sp, c, p = call["spec"], _by_id(call), _by_id(put)
    parity = sp["spot"] * np.exp(-sp["q"] * sp["T"]) - sp["K"] * np.exp(-sp["r"] * sp["T"])
    for mid in MODELS + ["bs"]:
        assert c[mid]["price"] - p[mid]["price"] == pytest.approx(parity, abs=1e-6), mid


def test_a_digital_call_plus_a_digital_put_pays_the_discount_factor(app, fitted):
    call, put = _price(app, "digital", strike_pct=100, call=True), _price(app, "digital", strike_pct=100, call=False)
    sp, c, p = call["spec"], _by_id(call), _by_id(put)
    for mid in MODELS + ["bs"]:
        assert 0.0 < c[mid]["price"] < 1.0 and c[mid]["price"] + p[mid]["price"] == pytest.approx(np.exp(-sp["r"] * sp["T"]), abs=1e-6), mid
    assert c["heston"]["price"] > c["bs"]["price"], "with put skew the market gives an at-the-money digital call more chance than a flat volatility does"


def test_a_knock_in_plus_a_knock_out_is_the_vanilla_and_a_knock_out_is_cheaper(app, fitted):
    out = _price(app, "barrier", barrier_kind="down-out", barrier_pct=90, strike_pct=100, paths=20_000)
    inn = _price(app, "barrier", barrier_kind="down-in", barrier_pct=90, strike_pct=100, paths=20_000)
    van = _by_id(_price(app, "vanilla", strike_pct=100, call=True))
    o, i = _by_id(out), _by_id(inn)
    for mid in MODELS:
        assert o[mid]["price"] < van[mid]["price"] and o[mid]["se"] > 0
        assert o[mid]["price"] + i[mid]["price"] == pytest.approx(van[mid]["price"], abs=4.0 * max(o[mid]["se"], 0.05) + 0.1), mid
    assert o["bs_continuous"]["price"] < o["bs"]["price"] + 4.0 * o["bs"]["se"], "watching the barrier continuously cancels a knock-out more often"
    assert out["summary"]["low"] <= out["summary"]["high"] and any("monitored on" in w for w in out["warnings"])


def test_an_average_price_option_is_cheaper_than_the_vanilla_and_black_scholes_is_exact(app, fitted):
    asian, van = _by_id(_price(app, "asian", fixings=12, strike_pct=100, paths=20_000)), _by_id(_price(app, "vanilla", strike_pct=100, call=True))
    for mid in MODELS + ["bs"]:
        assert 0 < asian[mid]["price"] < van[mid]["price"], mid
    assert asian["bs"]["se"] < 2e-3, "the geometric-average control variate removes nearly all the noise under Black-Scholes"


def test_an_american_put_is_worth_at_least_the_european_and_heston_is_not_offered(app, fitted):
    report = _price(app, "american", strike_pct=105)
    rows, sp = _by_id(report), report["spec"]
    assert rows["heston"]["price"] is None and "second state variable" in rows["heston"]["note"]
    for mid in ("merton", "nig", "bs"):
        r = rows[mid]
        assert r["price"] >= r["european"] - 1e-9 and r["premium"] > 0 and r["price"] >= sp["K"] - sp["spot"], mid
    ref = pde.vanilla(sp["spot"], sp["K"], sp["T"], sp["r"], sp["q"], report["sigma_atm"], False, american=True).price
    assert rows["bs"]["price"] == pytest.approx(ref, abs=1e-6), "the Black-Scholes row is the finite-difference solver"
    best = report["summary"]["best"]
    assert best in ("merton", "nig") and report["summary"]["best_price"] == rows[best]["price"], "the best of the models that can be priced"
    lines = report["boundary"]
    assert set(lines) == {best, "bs"}
    for line in lines.values():
        px = np.asarray(line["price"], dtype=float)
        assert len(px) == len(line["dte"]) and np.all(px < sp["K"]) and px[-1] < px[0], "the more time left, the lower the price at which exercising stops being optimal"


# ------------------------------------------------------------------------------------------------------------------------------ the hedge
def test_the_hedging_report_prices_the_cost_of_rebalancing_and_the_band_around_delta(app):
    rep = _run(app, "/api/models/hedge", underlying=UND, cost_bp=10, paths=3000, steps=63, strike_pct=100)
    json.dumps(rep, allow_nan=False)
    lel = rep["leland"]
    assert [x["steps"] for x in lel] == sorted(x["steps"] for x in lel) and 63 in [x["steps"] for x in lel]
    assert all(a["seller_price"] < b["seller_price"] and a["le"] < b["le"] for a, b in zip(lel, lel[1:])), "the more often the hedge is rebalanced the more it costs the seller"
    assert all(x["seller_price"] > rep["premium"] for x in lel) and all(x["buyer_price"] is None or x["buyer_price"] < rep["premium"] for x in lel)
    band = rep["band"]
    assert np.all(np.asarray(band["low"]) < np.asarray(band["delta"])) and np.all(np.asarray(band["delta"]) < np.asarray(band["high"])) and band["half_width_at_spot"] > 0
    sims = {s["strategy"]: s for s in rep["simulation"]}
    assert set(sims) == {"delta", "leland", "band"} and sims["band"]["trades"] < sims["delta"]["trades"] and sims["band"]["cost"] < sims["delta"]["cost"], "a no-trade band trades less and pays less"
    assert sims["leland"]["premium"] > sims["delta"]["premium"] and sims["leland"]["mean"] > sims["delta"]["mean"], "Leland's higher price pays for the hedging cost"
    dpz = rep["dpz"]
    assert dpz["buyer"] < dpz["crr"] < dpz["writer"] and dpz["band"][0] <= dpz["crr_delta"] <= dpz["band"][1] + 1e-9, "with costs the option has a bid and an ask around the frictionless price"
    h = rep["histogram"]
    assert len(h["edges"]) == 31 and set(h["share"]) == {"delta", "leland", "band"} and all(sum(v) == pytest.approx(1.0) for v in h["share"].values())
    assert set(h["outside"]) == {"delta", "leland", "band"} and all(0.0 <= v < 0.03 for v in h["outside"].values()), "only the extremes are left off the picture, and they are counted"
    assert dpz["crr"] == pytest.approx(rep["premium"], rel=0.02), "the tree hedges with the prepaid forward, so with a dividend its frictionless price is still the Black-Scholes one"
    assert rep["inputs"]["cost_bp"] == pytest.approx(10.0) and rep["inputs"]["gamma"] == pytest.approx(rep["inputs"]["risk_aversion"] / rep["premium"])


def test_without_a_cost_the_hedge_is_free_and_the_prices_collapse_to_black_scholes(app):
    rep = _run(app, "/api/models/hedge", underlying=UND, cost_bp=0, paths=2000, steps=40)
    assert all(x["le"] == 0.0 and x["seller_price"] == pytest.approx(rep["premium"]) and x["buyer_price"] == pytest.approx(rep["premium"]) for x in rep["leland"])
    assert all(s["cost"] == 0.0 for s in rep["simulation"]) and rep["dpz"]["writer"] == pytest.approx(rep["dpz"]["crr"], abs=5e-3) and rep["dpz"]["buyer"] == pytest.approx(rep["dpz"]["crr"], abs=5e-3)


def test_a_buyer_cannot_replicate_when_the_leland_number_reaches_one(app):
    rep = _run(app, "/api/models/hedge", underlying=UND, cost_bp=150, paths=1000, steps=252)
    assert rep["leland"][-1]["le"] >= 1.0 and rep["leland"][-1]["buyer_price"] is None and any("cannot replicate" in w for w in rep["warnings"])


# ------------------------------------------------------------------------------------------------------------------------------ refusals
@pytest.mark.parametrize("path, body, message", [
    ("/api/models/fit", {"models": "merton"}, "models must be a list"),
    ("/api/models/fit", {"models": ["black_magic"]}, "unknown model 'black_magic'"),
    ("/api/models/fit", {"models": []}, "models must be a list"),
    ("/api/models/fit", {"max_expiries": 1}, "max expiries must be between 2 and 8"),
    ("/api/models/fit", {"window": 0.9}, "window must be between"),
    ("/api/models/fit", {"min_dte": 0}, "min dte must be between 1 and 120"),
    ("/api/models/fit", {"underlying": "NOPE"}, "unknown underlying 'NOPE'"),
    ("/api/models/price", {"product": "swaption"}, "product must be one of"),
    ("/api/models/price", {"product": "vanilla", "expiry": "2020-01-01"}, "is not on the chain"),
    ("/api/models/price", {"product": "vanilla", "expiry": "next tuesday"}, "expiry must be a date"),
    ("/api/models/price", {"product": "vanilla", "strike_pct": 5}, "strike pct must be between"),
    ("/api/models/price", {"product": "barrier", "barrier_kind": "down-out", "barrier_pct": 105}, "a down barrier must be below the price"),
    ("/api/models/price", {"product": "barrier", "barrier_kind": "up-in", "barrier_pct": 95}, "an up barrier must be above the price"),
    ("/api/models/price", {"product": "barrier", "barrier_kind": "sideways"}, "barrier kind must be one of"),
    ("/api/models/price", {"product": "barrier", "paths": 10}, "paths must be between"),
    ("/api/models/price", {"product": "asian", "fixings": 1}, "fixings must be between"),
    ("/api/models/hedge", {"cost_bp": 900}, "cost bp must be between"),
    ("/api/models/hedge", {"steps": 1}, "steps must be between"),
    ("/api/models/hedge", {"risk_aversion": 0}, "risk aversion must be between"),
    ("/api/models/hedge", {"vol": 900}, "vol must be between"),
    ("/api/models/hedge", {"paths": 50}, "paths must be between"),
])
def test_requests_that_cannot_be_served_are_refused_before_a_job_starts(app, path, body, message):
    with pytest.raises(ApiError, match=message):
        app.extra_post[path]({"underlying": UND, **body} if "underlying" not in body else body)


def test_a_failed_fit_is_reported_not_raised_when_other_models_succeed(app, monkeypatch):
    real = calibration.calibrate

    def flaky(model, *a, **k):
        if model == "merton":
            raise ValueError("the search ended outside the valid region of the model")
        return real(model, *a, **k)

    monkeypatch.setattr(calibration, "calibrate", flaky)
    report = _run(app, "/api/models/fit", underlying=UND, models=["merton", "nig"], window=0.25)
    assert [r["id"] for r in report["models"]] == ["nig"] and any("Merton" in w and "could not be fitted" in w for w in report["warnings"])
    failed = wait(app, app.extra_post["/api/models/fit"]({"underlying": UND, "models": ["merton"], "window": 0.27}))
    assert failed["status"] == "error" and "no model could be fitted" in failed["error"]


def test_the_endpoints_are_served_over_http_and_refusals_are_400s(app):
    running = server.start(app, port=0)
    try:
        status, body, _ = _http(running, "/api/models/plan")
        assert status == 200 and [m["id"] for m in json.loads(body)["models"]][0] == "merton"
        status, body, _ = _http(running, "/api/models/fit", {"models": ["black_magic"]})
        assert status == 400 and "unknown model" in json.loads(body)["error"]
        assert _http(running, "/api/models/plan", token=False)[0] == 403
    finally:
        running.stop()
    assert om.FIT_MODELS == ("merton", "kou", "vg", "nig", "heston")
