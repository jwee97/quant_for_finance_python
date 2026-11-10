"""The yield-curve tool (``assets.rates``, the Rates tab and ``quant curve``): a maturity is read from the way people write it, a Nelson-Siegel or Svensson curve is fitted to yields and recovers a curve it
was drawn from, the zero, forward and discount curves agree with each other, par yields are bootstrapped first, the warnings say when the model cannot be trusted, and the dynamic analysis follows the
factors of a history of curves and checks its own forecast against the random walk. The endpoints answer plain JSON and refuse what cannot be fitted."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.assets import rates as r
from src.webapp.api import ApiError
from tests.test_fund_api import get, make_app, post

MATS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]


def ns_yields(b0=0.045, b1=-0.02, b2=-0.01, lam=0.6, mats=MATS):
    return r.nelson_siegel(np.array(mats, float), b0, b1, b2, lam) * 100.0


# ------------------------------------------------------------------------------------------------------------------ reading maturities and curves
@pytest.mark.parametrize("text, years", [("3M", 0.25), ("6m", 0.5), ("10Y", 10.0), ("2yr", 2.0), ("30 Y", 30.0), ("0.25", 0.25), (1, 1.0), (2.5, 2.5), ("90D", 90 / 365), ("2W", 14 / 365), ("12", 12.0)])
def test_a_maturity_is_read_as_people_write_it(text, years):
    assert r.parse_tenor(text) == pytest.approx(years)


@pytest.mark.parametrize("text", ["", "abc", "3X", "-1", "0", "150Y", "1e3", float("nan"), "Y3"])
def test_a_maturity_that_is_not_one_is_refused(text):
    with pytest.raises(ValueError):
        r.parse_tenor(text)


def test_a_pasted_curve_is_read_with_or_without_a_header_and_any_separator():
    tau, y = r.parse_curve_text("tenor,yield\n3M 5.3\n2Y, 4.8\n10Y\t4.3\n\n30Y;4.5%\n")
    assert tau.tolist() == pytest.approx([0.25, 2.0, 10.0, 30.0]) and y.tolist() == [5.3, 4.8, 4.3, 4.5]
    tau, y = r.parse_curve_text("0.5 4.8\n2 4.5")
    assert tau.tolist() == [0.5, 2.0]
    for bad, why in [("", "type or paste"), ("   ", "type or paste"), ("2Y", "needs a maturity and a yield"), ("2Y 4.5\n3Y abc", "could not convert|is not a maturity|abc"), ("x" * 25_000, "at most 60")]:
        with pytest.raises(ValueError, match=why):
            r.parse_curve_text(bad)


def test_a_curve_is_checked_before_it_is_fitted():
    ok = ns_yields()
    r.check_curve(MATS, ok, "nelson_siegel")
    for tau, y, model, why in [(MATS[:3], ok[:3], "nelson_siegel", "at least 4"), (MATS[:5], ok[:5], "svensson", "at least 6"), (MATS, ok[:-1], "nelson_siegel", "one yield for every maturity"),
                               ([1, 1, 2, 3], [4, 4, 4, 4], "nelson_siegel", "twice"), (MATS, [v * 100 for v in ok], "nelson_siegel", "check the unit"), ([0.25, 0.5, 1, 100.5], [4, 4, 4, 4], "nelson_siegel", "between one day"),
                               (list(range(1, 62)), [4.0] * 61, "nelson_siegel", "at most 60"), (MATS, [np.nan] + list(ok[1:]), "nelson_siegel", "finite"), (MATS, ok, "vasicek", "model must be")]:
        with pytest.raises(ValueError, match=why):
            r.check_curve(tau, y, model)
    tau, y = r.check_curve([10, 2, 5, 3], [4.3, 4.8, 4.4, 4.6], "nelson_siegel")
    assert tau.tolist() == [2, 3, 5, 10] and y.tolist() == [4.8, 4.6, 4.4, 4.3]                      # sorted by maturity, the yields with them


# ------------------------------------------------------------------------------------------------------------------ the fit
def test_a_nelson_siegel_curve_is_recovered_from_yields_drawn_from_it():
    rep = r.curve_report(MATS, ns_yields(), "nelson_siegel")
    p = rep["parameters"]
    assert (p["beta0"], p["beta1"], p["beta2"]) == pytest.approx((0.045, -0.02, -0.01), abs=1e-6) and p["lambda"] == pytest.approx(0.6, abs=0.02)
    assert rep["fit"]["rmse_bp"] < 0.01 and rep["warnings"] == []
    f = rep["factors"]
    assert f["level"] == pytest.approx(4.5, abs=1e-4) and f["short_rate"] == pytest.approx(2.5, abs=1e-4) and f["slope"] == pytest.approx(2.0, abs=1e-4)      # long rate less short rate
    assert f["curvature"] == pytest.approx(-1.0, abs=1e-4) and f["hump_maturity"] == pytest.approx(r.HUMP / p["lambda"])


def test_a_fixed_decay_is_held_and_the_betas_are_the_least_squares_ones():
    y = ns_yields() + np.array([3, -2, 1, 0, -1, 2, -3, 1, 0, -1]) * 0.02                              # a few basis points of noise
    rep = r.curve_report(MATS, y, "nelson_siegel", lam=0.9)
    X = r.nelson_ns_loadings(np.array(MATS, float), 0.9)
    beta = np.linalg.lstsq(X, y / 100, rcond=None)[0]
    assert rep["parameters"]["lambda"] == 0.9 and [rep["parameters"][k] for k in ("beta0", "beta1", "beta2")] == pytest.approx(beta.tolist(), abs=1e-8)
    free = r.curve_report(MATS, y, "nelson_siegel")
    assert free["fit"]["rmse_bp"] <= rep["fit"]["rmse_bp"] + 1e-9                                       # letting the decay go can only fit better


def test_a_svensson_curve_with_two_humps_is_fitted_where_nelson_siegel_cannot():
    mats = [0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
    y = r.svensson(np.array(mats, float), 0.045, -0.025, -0.02, 0.09, 0.8, 0.1) * 100.0
    sv = r.curve_report(mats, y, "svensson")
    ns = r.curve_report(mats, y, "nelson_siegel")
    assert sv["fit"]["rmse_bp"] < 0.05 and ns["fit"]["rmse_bp"] > 3 * sv["fit"]["rmse_bp"] and sv["fit"]["rmse_bp"] < ns["fit"]["rmse_bp"]
    assert any("more shape than this model can draw" in w for w in ns["warnings"]) and not sv["warnings"]
    p = sv["parameters"]
    assert abs(p["lambda1"] - p["lambda2"]) >= 0.05 and p["beta0"] == pytest.approx(0.045, abs=2e-3)
    assert set(sv["factors"]) >= {"second_curvature", "second_hump_maturity"}


def test_six_or_seven_points_for_the_svensson_fit_are_flagged():
    mats = [0.5, 1, 2, 5, 10, 30]
    rep = r.curve_report(mats, r.svensson(np.array(mats, float), 0.045, -0.02, -0.01, 0.01, 0.7, 0.15) * 100, "svensson")
    assert any("almost nothing to check it against" in w for w in rep["warnings"])


def test_the_zero_forward_and_discount_curves_agree_with_each_other():
    rep = r.curve_report(MATS, ns_yields(), "nelson_siegel")
    c = rep["curve"]
    tau, z, f, d = (np.array(c[k]) for k in ("maturities", "zero", "forward", "discount"))
    assert d == pytest.approx(np.exp(-z / 100 * tau))
    h = 1e-5
    p = rep["parameters"]
    up, dn = r.nelson_siegel(tau + h, p["beta0"], p["beta1"], p["beta2"], p["lambda"]), r.nelson_siegel(tau - h, p["beta0"], p["beta1"], p["beta2"], p["lambda"])
    forward_numeric = ((tau + h) * up - (tau - h) * dn) / (2 * h)                                       # f = d(tau y)/d tau
    assert f == pytest.approx(forward_numeric * 100, abs=1e-4)
    sv = r.curve_report(MATS, r.svensson(np.array(MATS, float), 0.045, -0.02, -0.01, 0.01, 0.7, 0.1) * 100, "svensson")
    tau, f2 = np.array(sv["curve"]["maturities"]), np.array(sv["curve"]["forward"])
    q = sv["parameters"]
    s_up, s_dn = (r.svensson(tau + h, q["beta0"], q["beta1"], q["beta2"], q["beta3"], q["lambda1"], q["lambda2"]), r.svensson(tau - h, q["beta0"], q["beta1"], q["beta2"], q["beta3"], q["lambda1"], q["lambda2"]))
    assert f2 == pytest.approx((((tau + h) * s_up - (tau - h) * s_dn) / (2 * h)) * 100, abs=1e-4)
    assert [k["maturity"] for k in rep["key_rates"]] == [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]


def par_yields_from(zero_fn, maturities):
    out = []
    for T in maturities:
        grid = np.arange(1, int(round(T * 2)) + 1) / 2
        df = np.exp(-zero_fn(grid) * grid)
        out.append(2 * (1 - df[-1]) / df.sum() * 100)
    return out


def test_par_yields_are_bootstrapped_to_zero_rates_before_the_fit():
    zero = lambda t: r.nelson_siegel(np.asarray(t, float), 0.045, -0.02, -0.01, 0.6)                  # noqa: E731
    mats = [0.5, 1, 2, 3, 5, 7, 10]
    par = par_yields_from(zero, mats)
    rep = r.curve_report(mats, par, "nelson_siegel", kind="par")
    assert any("bootstrapping" in n for n in rep["notes"])
    assert rep["fit"]["rmse_bp"] < 1.5                                                                    # the par curve between its maturities is interpolated, so the zero curve is close, not exact
    assert [rep["parameters"][k] for k in ("beta0", "beta1", "beta2")] == pytest.approx([0.045, -0.02, -0.01], abs=5e-3)
    assert len(rep["fit"]["points"]) == 20                                                                # the zero curve at half-year steps up to ten years
    plain = r.curve_report(mats, par, "nelson_siegel", kind="zero")                                      # read as zero rates the same numbers give a different curve
    assert plain["parameters"]["beta0"] != pytest.approx(rep["parameters"]["beta0"], abs=1e-6)


def test_the_report_says_when_the_decay_hit_its_edge_the_curve_misses_or_runs_past_the_data():
    flat = r.curve_report(MATS, [4.0, 4.0001, 4.0, 4.0001, 4.0, 4.0001, 4.0, 4.0001, 4.0, 4.0001], "nelson_siegel")
    assert any("edge of the range" in w for w in flat["warnings"])
    kinked = ns_yields().copy()
    kinked[4] += 0.4
    kinked[6] -= 0.4
    assert any("basis points on average" in w for w in r.curve_report(MATS, kinked, "nelson_siegel")["warnings"])
    short = r.curve_report([0.25, 0.5, 1, 2, 3, 5], ns_yields(mats=[0.25, 0.5, 1, 2, 3, 5]), "nelson_siegel")
    assert any("extrapolation" in n for n in short["notes"]) and short["observed_range"] == [0.25, 5.0]
    assert all(m <= 5.0 for m in [k["maturity"] for k in short["key_rates"]])
    falling = r.curve_report(MATS, [1.0, 0.2, 0.05, 0.1, 0.8, 2.0, 3.2, 4.0, 4.5, 4.6], "nelson_siegel")
    assert any("forward rate is negative" in w for w in falling["warnings"]) or min(falling["curve"]["forward"]) >= 0


def test_a_decay_out_of_range_and_an_unknown_kind_are_refused():
    for kw, why in [({"lam": 0.01}, "between 0.05 and 6"), ({"lam": 9}, "between 0.05 and 6"), ({"kind": "swap"}, "zero rates or par yields")]:
        with pytest.raises(ValueError, match=why):
            r.curve_report(MATS, ns_yields(), "nelson_siegel", **kw)


# ------------------------------------------------------------------------------------------------------------------ a history of curves
def test_a_pasted_history_is_read_sorted_in_decimals_and_a_gap_drops_the_day():
    text = "date,10Y,3M,2Y,30Y\n2024-01-03,4.0,5.3,4.4,4.2\n2024-01-02,4.1,5.4,4.5,4.3\n2024-01-04,,5.2,4.3,4.1\n"
    panel = r.parse_panel_text(text)
    assert panel.columns.tolist() == pytest.approx([0.25, 2.0, 10.0, 30.0]) and panel.index.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert panel.iloc[0].tolist() == pytest.approx([0.054, 0.045, 0.041, 0.043])
    assert r.parse_panel_text(text, percent=False).iloc[0].tolist() == pytest.approx([5.4, 4.5, 4.1, 4.3])
    for bad, why in [("", "paste the history"), ("a,b\nnot-a-date,1", "dates"), ("date,xx\n2024-01-02,1", "not a maturity"), ("x" * 160_000, "larger than the page")]:
        with pytest.raises(ValueError, match=why):
            r.parse_panel_text(bad)


def test_the_factors_of_a_history_are_followed_and_the_forecast_is_checked_against_the_random_walk():
    panel = r.synthetic_curve_panel(n_days=1200, seed=3)
    rep = r.dynamic_report(panel, lam=0.5, horizon=21, test_days=500, step=10)
    assert rep["days"] == 1200 and rep["fit_rmse_bp"] < 2.0                                              # the history was made with this decay and about a basis point of noise
    by = {p["factor"]: p for p in rep["persistence"]}
    assert by["level"]["ar1"] > 0.97 and by["slope"]["ar1"] > 0.95 and set(by) == {"level", "slope", "curvature"}
    wf = rep["walk_forward"]
    assert wf["origins"] > 40 and wf["horizon"] == 21 and np.isfinite(wf["rmse_bp"]["dynamic_ns"]) and np.isfinite(wf["rmse_bp"]["random_walk"]) and 0 <= wf["better_share"] <= 1
    assert len(wf["by_maturity"]) == len(panel.columns) and len(rep["latest"]["forecast"]) == len(panel.columns)
    assert len(rep["dates"]) <= 401 and len(rep["factors"]["level"]) == len(rep["dates"]) and rep["dates"][-1] == rep["latest"]["date"]
    json.dumps(rep, allow_nan=False)


def test_the_variance_of_the_forecast_machinery_matches_the_diebold_li_function_of_the_library():
    panel = r.synthetic_curve_panel(n_days=600, seed=1)
    dns = r.dynamic_nelson_siegel(panel, 0.5)
    betas = dns["factors"].to_numpy()
    mine = r.nelson_ns_loadings(np.array(panel.columns, float), 0.5) @ r._var1_path(betas, 21)
    ref = r.forecast_curve(dns["factors"], list(panel.columns), 0.5, 21).to_numpy()
    assert mine == pytest.approx(ref, abs=1e-9)


def test_a_history_that_cannot_be_analysed_is_refused():
    panel = r.synthetic_curve_panel(n_days=400, seed=0)
    for kw, why in [({"lam": 0.01}, "between 0.05 and 6"), ({"horizon": 0}, "horizon is 1 to 250"), ({"test_days": 50}, "test window"), ({"step": 100}, "step 1 to 60")]:
        with pytest.raises(ValueError, match=why):
            r.dynamic_report(panel, **kw)
    with pytest.raises(ValueError, match="at least four maturities"):
        r.dynamic_report(panel.iloc[:, :3])
    with pytest.raises(ValueError, match="at least 250 days"):
        r.dynamic_report(panel.iloc[:100])


# ------------------------------------------------------------------------------------------------------------------ the endpoints
@pytest.fixture(scope="module")
def app(tmp_path_factory):
    return make_app(tmp_path_factory.mktemp("rates_app"))


def plain(value):
    json.dumps(value, allow_nan=False)
    return value


def test_the_routes_are_in_the_app_and_the_plan_names_the_models_and_a_made_up_example(app):
    assert "/api/rates/plan" in app.extra_get and "/api/rates/curve" in app.extra_post and "/api/rates/dynamic" in app.extra_post
    plan = plain(get(app, "/api/rates/plan"))
    assert [m["id"] for m in plan["models"]] == ["nelson_siegel", "svensson"] and [k["id"] for k in plan["kinds"]] == ["zero", "par"]
    assert "not market data" in plan["example"]["label"] and plan["limits"]["max_points"] == 60 and plan["defaults"]["lambda"] == 0.5
    rep = plain(post(app, "/api/rates/curve", text=plan["example"]["text"]))                           # the example fits as it is
    assert rep["model"] == "nelson_siegel" and len(rep["fit"]["points"]) == 10


def test_a_curve_is_fitted_from_text_or_from_lists_in_percent_or_decimals(app):
    y = [float(v) for v in ns_yields()]
    from_text = plain(post(app, "/api/rates/curve", text="\n".join(f"{m} {v}" for m, v in zip(MATS, y))))
    from_lists = plain(post(app, "/api/rates/curve", maturities=MATS, yields=y))
    decimals = plain(post(app, "/api/rates/curve", maturities=MATS, yields=[v / 100 for v in y], unit="decimal"))
    for other in (from_lists, decimals):
        assert other["parameters"]["beta0"] == pytest.approx(from_text["parameters"]["beta0"], abs=1e-9)
    assert from_text["parameters"]["beta0"] == pytest.approx(0.045, abs=1e-6)
    fixed = plain(post(app, "/api/rates/curve", maturities=MATS, yields=y, **{"lambda": 0.6}))
    assert fixed["parameters"]["lambda"] == 0.6
    sv = plain(post(app, "/api/rates/curve", maturities=MATS, yields=y, model="svensson"))
    assert "beta3" in sv["parameters"]
    par = plain(post(app, "/api/rates/curve", maturities=MATS[1:7], yields=[4.8, 4.5, 4.3, 4.2, 4.1, 4.15], kind="par"))
    assert par["kind"] == "par" and par["notes"]


def test_a_curve_that_cannot_be_fitted_is_a_message(app):
    y = [float(v) for v in ns_yields()]
    for body, why in [({}, "give the curve as text"), ({"text": "2Y 4.5\n3Y 4.4"}, "at least 4"), ({"maturities": MATS, "yields": y, "model": "vasicek"}, "model must be"),
                      ({"maturities": MATS, "yields": [v * 100 for v in y]}, "check the unit"), ({"maturities": MATS, "yields": y, "unit": "bp"}, "unit must be"),
                      ({"maturities": MATS, "yields": y, "lambda": 20}, "lambda must be between"), ({"maturities": MATS, "yields": y, "kind": "swap"}, "zero rates or par yields"),
                      ({"maturities": ["3X"] * 5, "yields": [1] * 5}, "not a maturity"), ({"maturities": MATS, "yields": ["a"] * 10}, "could not convert")]:
        with pytest.raises(ApiError, match=why):
            post(app, "/api/rates/curve", **body)


def test_the_dynamic_analysis_runs_on_the_simulated_history_and_says_it_is_simulated(app):
    out = plain(post(app, "/api/rates/dynamic", source="simulated", horizon=10, test_days=400, step=10))
    assert out["source"] == "simulated" and "simulated" in out["note"] and out["horizon"] == 10 and out["walk_forward"]["origins"] > 20
    assert len(out["latest"]["observed"]) == len(out["maturities"]) == 10
    again = plain(post(app, "/api/rates/dynamic", horizon=10, test_days=400, step=10))                  # no source: the simulated history
    assert again["latest"]["forecast"] == out["latest"]["forecast"]
    other = plain(post(app, "/api/rates/dynamic", seed=5, horizon=10, test_days=400, step=10))
    assert other["latest"]["observed"] != out["latest"]["observed"]


def test_a_pasted_history_is_analysed_and_a_bad_request_is_a_message(app):
    panel = r.synthetic_curve_panel(n_days=500, seed=2)
    panel.columns = [f"{m}Y" if m >= 1 else f"{round(m * 12)}M" for m in panel.columns]
    text = (panel * 100).round(4).to_csv(date_format="%Y-%m-%d", index_label="date")
    out = plain(post(app, "/api/rates/dynamic", text=text, horizon=21, test_days=200, step=10))
    assert out["source"] == "text" and out["note"] == "" and out["days"] == 500
    for body, why in [({"source": "feed"}, "source must be"), ({"text": "date,3M\n2024-01-02,1"}, "at least four maturities"), ({"text": text, "lambda": 0}, "lambda must be between"),
                      ({"text": text, "horizon": 999}, "horizon must be between"), ({"text": text, "unit": "bp"}, "unit must be"), ({"source": "text"}, "paste the history")]:
        with pytest.raises(ApiError, match=why):
            post(app, "/api/rates/dynamic", **body)


# ------------------------------------------------------------------------------------------------------------------ the command line
def test_the_command_line_fits_a_curve_from_lists_and_from_a_file_and_writes_the_result(tmp_path, capsys):
    from src import cli

    y = [float(v) for v in ns_yields()]
    assert cli.main(["curve", "fit", "--maturities", *[str(m) for m in MATS], "--yields", *[str(v) for v in y], "--json", str(tmp_path / "c.json")]) == 0
    out = capsys.readouterr().out
    assert "nelson-siegel curve through 10 points" in out and "Level (β0)" in out and "4.500" in out and "discount" in out
    saved = json.loads((tmp_path / "c.json").read_text())
    assert saved["parameters"]["beta0"] == pytest.approx(0.045, abs=1e-6)
    (tmp_path / "curve.txt").write_text("\n".join(f"{m}Y {v}" for m, v in zip(MATS, y)))
    assert cli.main(["curve", "fit", "--file", str(tmp_path / "curve.txt"), "--model", "svensson", "--kind", "zero"]) == 0
    assert "svensson curve through 10 points" in capsys.readouterr().out


def test_the_command_line_refuses_what_cannot_be_fitted_with_a_message(tmp_path, capsys):
    from src import cli

    assert cli.main(["curve", "fit"]) == 2 and "give --file" in capsys.readouterr().out
    assert cli.main(["curve", "fit", "--maturities", "1", "2", "--yields", "4", "4.1"]) == 2 and "at least 4" in capsys.readouterr().out
    assert cli.main(["curve", "fit", "--file", str(tmp_path / "missing.txt")]) == 2 and "error:" in capsys.readouterr().out


def test_the_command_line_follows_the_factors_of_the_simulated_history(capsys):
    from src import cli

    assert cli.main(["curve", "dynamic", "--test-days", "300", "--step", "10"]) == 0
    out = capsys.readouterr().out
    assert "simulated history" in out and "latest factors" in out and "walk-forward at" in out and "random walk" in out
