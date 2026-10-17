"""The option-model reports and the ``quant models`` command: the checks on what is asked for, the session cache of fits (reused only for the same quotes, the least recently used dropped), the product
specification against a chain, the resampled early-exercise boundary and the price-by-strike curve, the hedging inputs, the text the terminal prints, and the command end to end on a chain file."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import cli
from src.assets import option_models as om
from src.assets import options_analytics as oa
from src.derivatives import calibration, fourier, levy

SPOT, RATE, DIV = 500.0, 0.04, 0.012


@pytest.fixture(scope="module")
def chain():
    return oa.prepare_chain(oa.build_model_chain(SPOT, pd.Timestamp("2026-10-16"), 0.15, 0.18, RATE, DIV, 0.15, "TST", 600.0))


@pytest.fixture(scope="module")
def quotes(chain):
    return om.select_quotes(chain)


@pytest.fixture(scope="module")
def fits(quotes):
    results, failed = om.FitCache().calibrate(quotes, ["merton", "heston"])
    assert not failed
    return results


# ------------------------------------------------------------------------------------------------------------------------------ checks
def test_a_number_or_a_model_that_is_not_allowed_is_refused_with_the_name_of_the_field():
    assert om.check_range("strike_pct", "105", 20, 300) == "105" and om.check_range("window", 0.3, 0.05, 0.6) == 0.3
    for bad, message in ((5, "strike pct must be between 20 and 300"), ("x", "strike pct must be a number"), (float("nan"), "must be between"), (None, "must be a number")):
        with pytest.raises(ValueError, match=message):
            om.check_range("strike_pct", bad, 20, 300)
    assert om.check_models(None) == om.check_models("") == list(om.FIT_MODELS) and om.check_models(["nig", "merton", "nig"]) == ["nig", "merton"]
    for bad, message in (("merton", "models must be a list"), ([], "models must be a list"), (["merton", "garch"], "unknown model 'garch'")):
        with pytest.raises(ValueError, match=message):
            om.check_models(bad)


def test_the_plan_describes_every_model_with_its_parameters_and_the_ranges_the_fit_searches(chain):
    plan = om.plan()
    assert [m["id"] for m in plan["models"]] == list(om.FIT_MODELS) and {p["id"] for p in plan["products"]} == set(om.PRODUCTS)
    for m in plan["models"]:
        assert m["label"] == levy.MODELS[m["id"]].label and all(p["low"] < p["start"] < p["high"] for p in m["params"]) and m["text"] and m["assumes"] and m["fails"]
    assert {m["id"]: m["american"] for m in plan["models"]} == {"merton": True, "kou": True, "vg": True, "nig": True, "heston": False}
    assert set(om.LIMITS) >= {"strike_pct", "paths", "cost_bp", "steps"} and om.DEFAULTS["paths"] == 50_000


# ------------------------------------------------------------------------------------------------------------------------------ the cache
def test_a_fit_is_kept_for_the_same_quotes_and_settings_and_not_for_others(quotes, chain, monkeypatch):
    calls = []
    real = calibration.calibrate
    monkeypatch.setattr(calibration, "calibrate", lambda m, q, **k: (calls.append(m), real(m, q, **k))[1])
    cache = om.FitCache(size=2)
    key = (om.fingerprint(chain), 6, 7, 0.3)
    first, _ = cache.calibrate(quotes, ["merton"], key)
    again, _ = cache.calibrate(quotes, ["merton"], key)
    assert calls == ["merton"] and again["merton"] is first["merton"], "the same key is a hit"
    cache.calibrate(quotes, ["merton"], (om.fingerprint(chain), 6, 7, 0.25))
    assert calls == ["merton", "merton"] and len(cache) == 2, "another window is another fit"
    cache.calibrate(quotes, ["merton"])                                              # no key: neither read nor kept
    assert calls == ["merton"] * 3 and len(cache) == 2
    cache.calibrate(quotes, ["merton"], (om.fingerprint(chain), 4, 7, 0.3))
    assert len(cache) == 2, "the least recently used fit is dropped at the size limit"
    cache.calibrate(quotes, ["merton"], key)
    assert calls == ["merton"] * 5, "the first key was the one dropped, so it is fitted again"


def test_a_search_that_fails_is_reported_beside_the_models_that_did_not(quotes, monkeypatch):
    real = calibration.calibrate

    def flaky(model, q, **k):
        if model == "kou":
            raise ValueError("the search ended outside the valid region of the model")
        return real(model, q, **k)

    monkeypatch.setattr(calibration, "calibrate", flaky)
    messages = []
    done, failed = om.FitCache().calibrate(quotes, ["kou", "merton"], None, messages.append)
    assert list(done) == ["merton"] and failed == {"kou": "the search ended outside the valid region of the model"}
    assert len(messages) == 2 and "Kou" in messages[0] and "1 of 2" in messages[0]


def test_the_fingerprint_changes_when_the_quotes_do(chain):
    assert om.fingerprint(chain) == om.fingerprint(chain.copy())
    other = chain.copy()
    other.loc[other.index[10], "mid"] += 0.01
    assert om.fingerprint(other) != om.fingerprint(chain) and om.fingerprint(chain.iloc[:-5]) != om.fingerprint(chain)


# ------------------------------------------------------------------------------------------------------------------------------ the product
def test_a_product_specification_is_read_against_the_chain(chain):
    s = om.product_spec(chain)
    assert s["product"] == "vanilla" and s["call"] and s["K"] == pytest.approx(SPOT) and s["S"] == pytest.approx(SPOT) and abs(s["dte"] - 90) < 45 and s["paths"] == 0 and s["steps"] >= 10
    assert om.product_spec(chain, "american")["call"] is False and om.product_spec(chain, "american", call=True)["call"] is True
    listed = oa.expiry_list(chain)
    pick = listed[3]["expiry"]
    p = om.product_spec(chain, "barrier", expiry=pick, strike_pct="95", barrier_kind="up-out", paths="20000", seed=3)
    assert p["expiry"] == pick and p["K"] == pytest.approx(0.95 * SPOT) and p["H"] == pytest.approx(1.10 * SPOT) and p["kind"] == "up-out" and p["paths"] == 20_000 and p["seed"] == 3
    assert om.product_spec(chain, "barrier")["H"] == pytest.approx(0.90 * SPOT)
    near = om.product_spec(chain, dte=14)
    assert abs(near["dte"] - 14) <= min(abs(e["dte"] - 14) for e in listed) + 1e-9
    a = om.product_spec(chain, "asian", expiry=pick, fixings=6)
    assert a["fixings"] == 6 and a["paths"] == 50_000


@pytest.mark.parametrize("kwargs, message", [
    ({"product": "swaption"}, "product must be one of"), ({"expiry": "2020-01-01"}, "is not on the chain"), ({"expiry": "someday"}, "expiry must be a date"), ({"strike_pct": 5}, "strike pct must be between"),
    ({"dte": 1}, "dte must be between"), ({"product": "barrier", "barrier_kind": "sideways"}, "barrier kind must be one of"), ({"product": "barrier", "barrier_pct": 105}, "a down barrier must be below the price"),
    ({"product": "barrier", "barrier_kind": "up-in", "barrier_pct": 95}, "an up barrier must be above the price"), ({"product": "barrier", "barrier_pct": 5}, "barrier pct must be between"),
    ({"product": "barrier", "paths": 10}, "paths must be between"), ({"product": "asian", "fixings": 1}, "fixings must be between"), ({"product": "asian", "fixings": 5000}, "fixings must be between"),
])
def test_a_specification_that_cannot_be_priced_is_refused(chain, kwargs, message):
    with pytest.raises(ValueError, match=message):
        om.product_spec(chain, **kwargs)


# ------------------------------------------------------------------------------------------------------------------------------ the reports
def test_the_fit_report_ranks_the_models_and_every_array_lines_up(chain, quotes, fits):
    data, warnings, notes = om.fit_report("TST", quotes, fits, {"kou": "no luck"}, {"max_expiries": 6, "min_dte": 7, "window": 0.3}, simulated=True)
    assert [m["id"] for m in data["models"]] == sorted(fits, key=lambda m: fits[m].rmse) and data["best"] == data["models"][0]["id"] and data["underlying"] == "TST" and data["spot"] == pytest.approx(SPOT)
    assert any("fitted to another model" in w for w in warnings) and any("Kou" in w and "no luck" in w for w in warnings) and len(notes) >= 3
    d = data["density"]
    assert set(d["curves"]) == set(fits) and len(d["x"]) == len(d["bs"]) == 161 and len(d["drop_pct"]) == 30
    assert {t["id"] for t in d["tails"]} == set(fits) | {"bs"} and all(len(s["market"]) == len(s["strike"]) for s in data["smiles"])
    real, _, _ = om.fit_report("TST", quotes, fits, {}, {}, simulated=False)
    assert real["best"] == data["best"]
    plain = om.fit_report("TST", quotes, fits, {}, {}, simulated=False)[1]
    assert not any("fitted to another model" in w for w in plain), "a chain that is not a model is not called one"


def test_a_digital_call_and_put_and_a_vanilla_obey_their_identities_under_every_fitted_model(chain, quotes, fits):
    legs = {}
    for right in (True, False):
        spec = om.product_spec(chain, "digital", call=right)
        legs[right] = om.price_report("TST", chain, spec, quotes, fits, {})[0]
    t = legs[True]["spec"]["T"]
    for mid in list(fits) + ["bs"]:
        c = next(r for r in legs[True]["models"] + legs[True]["benchmarks"] if r["id"] == mid)["price"]
        p = next(r for r in legs[False]["models"] + legs[False]["benchmarks"] if r["id"] == mid)["price"]
        assert c + p == pytest.approx(np.exp(-quotes.r * t), abs=1e-6), mid
    curve = legs[True]["curve"]
    assert set(curve["curves"]) == set(fits) | {"bs"} and all(np.all(np.diff(v) < 0) for v in curve["curves"].values()), "a digital call is worth less the higher the strike"
    data = om.price_report("TST", chain, om.product_spec(chain, "vanilla"), quotes, fits, {})[0]
    spec = data["spec"]
    for mid, c in fits.items():
        row = next(r for r in data["models"] if r["id"] == mid)
        assert row["price"] == pytest.approx(float(fourier.price(c.model, spec["spot"], spec["K"], spec["T"], quotes.r, quotes.q, True)), rel=1e-12)
    assert data["summary"]["best"] == min(fits, key=lambda m: fits[m].rmse) and data["boundary"] is None and data["curve"] is not None


def test_the_spread_across_models_on_an_exotic_is_set_beside_their_spread_on_the_vanilla_at_the_same_strike(chain, quotes, fits):
    vanilla = om.price_report("TST", chain, om.product_spec(chain, "vanilla"), quotes, fits, {})
    assert vanilla[0]["summary"]["vanilla_range_share"] is None and not any("differ on this product" in w for w in vanilla[1]), "on a vanilla the spread is the fit, so there is nothing to compare it with"
    for kwargs in ({"product": "barrier", "barrier_kind": "up-in", "paths": 20_000}, {"product": "asian"}):                      # calls at the same strike and expiry as the vanilla above
        data, warnings, _ = om.price_report("TST", chain, om.product_spec(chain, **kwargs), quotes, fits, {})
        assert data["summary"]["vanilla_range_share"] == pytest.approx(vanilla[0]["summary"]["range_share"], rel=1e-9), kwargs
        assert any("differ on this product by" in w for w in warnings), kwargs
    data, warnings, _ = om.price_report("TST", chain, om.product_spec(chain, "digital", call=False), quotes, fits, {})
    assert data["summary"]["vanilla_range_share"] is not None and any("on the vanilla at this strike" in w for w in warnings)


def test_the_exercise_boundaries_of_two_solvers_are_put_on_one_grid_of_days():
    tau_a, tau_b = np.linspace(0, 0.25, 201), np.linspace(0, 0.25, 401)
    rows = [{"id": "nig", "label": "NIG", "boundary": {"dte": tau_a * 365.0, "price": np.r_[np.nan, 90.0 - 20.0 * np.sqrt(tau_a[1:])]}}]
    bench = [{"id": "bs", "label": "BS", "boundary": {"dte": tau_b * 365.0, "price": np.r_[np.nan, 91.0 - 20.0 * np.sqrt(tau_b[1:])]}}, {"id": "market", "label": "x"}]
    out = om._boundary(rows, bench, points=25)
    assert set(out) == {"nig", "bs"} and len(out["nig"]["dte"]) == 25 and np.array_equal(out["nig"]["dte"], out["bs"]["dte"])
    assert np.all(np.diff(out["nig"]["price"]) < 0) and np.all(out["bs"]["price"] > out["nig"]["price"]) and om._boundary([{"id": "a", "label": "a"}], []) is None


def test_the_hedging_inputs_are_checked_against_the_chain_and_the_report_follows_them(chain):
    spec = om.product_spec(chain, "vanilla", dte=60)
    inputs = om.hedge_inputs(chain, spec, None, 8, None, 2.0, 2000, 4)
    g = chain[chain["expiry"] == pd.Timestamp(spec["expiry"])]
    assert inputs["sigma"] == pytest.approx(oa.atm_iv(g)) and inputs["cost"] == pytest.approx(8e-4) and inputs["steps"] == spec["steps"] and inputs["ra"] == 2.0 and inputs["paths"] == 2000
    assert inputs["r"] == pytest.approx(RATE) and inputs["q"] == pytest.approx(DIV)
    assert om.hedge_inputs(chain, spec, 25, 0, 40)["sigma"] == pytest.approx(0.25) and om.hedge_inputs(chain, spec, "", 0, "")["steps"] == spec["steps"]
    for kwargs, message in (({"vol_pct": 900}, "vol must be between"), ({"cost_bp": 900}, "cost bp must be between"), ({"steps": 1}, "steps must be between"), ({"risk_aversion": 0}, "risk aversion must be between"), ({"paths": 5}, "paths must be between")):
        with pytest.raises(ValueError, match=message):
            om.hedge_inputs(chain, spec, **kwargs)
    messages = []
    data, warnings, notes = om.hedge_report("TST", spec, inputs, messages.append)
    assert data["inputs"]["steps"] == spec["steps"] and data["premium"] > 0 and len(messages) >= 5 and len(notes) >= 5
    assert data["histogram"]["share"]["delta"].sum() == pytest.approx(1.0) and not any("cannot replicate" in w for w in warnings)


# ------------------------------------------------------------------------------------------------------------------------------ the text
def test_the_text_of_each_report_names_the_models_the_numbers_and_what_to_watch(chain, quotes, fits):
    fit, warnings, _ = om.fit_report("TST", quotes, fits, {}, {"max_expiries": 6, "min_dte": 7, "window": 0.3}, simulated=True)
    text = om.format_fit(fit, warnings)
    assert text.startswith("TST: option models fitted to") and "Merton jump diffusion" in text and "Heston stochastic volatility" in text and "flat volatility (to beat)" in text
    assert "Black-Scholes" in text and "fall 10%" in text and "Watch: This chain is itself a model" in text
    price = om.format_price(*om.price_report("TST", chain, om.product_spec(chain, "american", strike_pct=105), quotes, fits, {})[:2])
    assert price.startswith("American option: put, strike 525.00") and "not priced" in price and "premium" in price and "Range across the fitted models" in price
    barrier = om.format_price(*om.price_report("TST", chain, om.product_spec(chain, "barrier", paths=5000), quotes, fits, {})[:2])
    assert "down and out at 450.00" in barrier and "continuously" in barrier
    spec = om.product_spec(chain, "vanilla", dte=60)
    hedge = om.format_hedge(*om.hedge_report("TST", spec, om.hedge_inputs(chain, spec, None, 10, None, 1.0, 1000, 0))[:2])
    assert "Leland no." in hedge and "<- yours" in hedge and "no-trade band" in hedge and "Utility-based prices" in hedge and "delta" in hedge and "band" in hedge


# ------------------------------------------------------------------------------------------------------------------------------ the command
@pytest.fixture(scope="module")
def chain_file(tmp_path_factory, chain):
    path = tmp_path_factory.mktemp("chain") / "TST.csv"
    chain[["date", "expiry", "strike", "right", "bid", "ask", "underlying", "rate", "dividend"]].to_csv(path, index=False)
    return str(path)


def run(capsys, *argv):
    code = cli.main(["models", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_the_command_lists_the_models_and_products(capsys):
    code, out, _ = run(capsys, "list")
    assert code == 0 and "Heston stochastic volatility" in out and "products: vanilla, digital, barrier, asian, american" in out and "barriers: down-out" in out


def test_the_command_fits_a_chain_file_prints_the_ranking_and_writes_the_result(capsys, chain_file, tmp_path):
    out_file = tmp_path / "fit.json"
    code, out, err = run(capsys, "fit", "--chain", chain_file, "--models", "merton", "--notes", "--json", str(out_file))
    assert code == 0 and "TST: option models fitted to" in out and "Merton jump diffusion" in out and "How this was made:" in out and "written to" in out
    assert "fitting Merton" in err and "This chain is itself a model" not in out, "a chain file is not called a model"
    saved = json.loads(out_file.read_text())
    assert saved["data"]["best"] == "merton" and saved["data"]["models"][0]["id"] == "merton" and isinstance(saved["data"]["density"]["x"], list) and saved["notes"]


def test_the_command_prices_a_product_and_costs_a_hedge_from_the_same_chain_file(capsys, chain_file):
    code, out, _ = run(capsys, "price", "--chain", chain_file, "--models", "merton", "--product", "digital", "--right", "put", "--strike", "98", "--dte", "60")
    assert code == 0 and out.startswith("Digital (cash-or-nothing): put, strike 490.00") and "Black-Scholes at the at-the-money volatility" in out and "Range across the fitted models" in out
    code, out, _ = run(capsys, "hedge", "--chain", chain_file, "--dte", "60", "--cost-bp", "10", "--paths", "1000", "--steps", "30")
    assert code == 0 and out.startswith("Hedging a call: strike") and "10.0 bp a trade, 30 rebalancing dates" in out and "<- yours" in out


@pytest.mark.parametrize("argv, message", [
    (["fit"], "choose an underlying from"), (["fit", "NOPE"], "'NOPE' is not in the data"), (["fit", "--chain", "/no/such/file.csv"], "/no/such/file.csv"),
    (["price", "--chain", "{chain}", "--models", "garch"], "unknown model 'garch'"), (["price", "--chain", "{chain}", "--models", "merton", "--product", "barrier", "--barrier", "105"], "a down barrier must be below the price"),
    (["hedge", "--chain", "{chain}", "--cost-bp", "900"], "cost bp must be between"), (["fit", "--chain", "{chain}", "--window", "0.9"], "window must be between"),
])
def test_a_request_the_command_cannot_serve_is_explained_and_exits_with_two(capsys, chain_file, argv, message):
    code, out, err = run(capsys, *[a.replace("{chain}", chain_file) for a in argv])
    assert code == 2 and message in err and out == ""
