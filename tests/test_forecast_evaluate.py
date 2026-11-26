"""The walk-forward evaluation: it never shows a method a day after the date it forecasts from, it finds the skill that was planted and none where there was none, and it reports honestly what it could not do."""

from __future__ import annotations

import numpy as np
import pytest

from src.forecast import evaluate as ev
from src.forecast.base import Forecast, ForecastError, Forecaster, History
from src.forecast.registry import BASELINE, BLEND, METHODS
from tests import forecast_worlds as W

SMALL = {"years": 5.0, "step": 21, "n_paths": 300}


def run(history, names, horizon=5, seed=0, **kw):
    return ev.evaluate(history, names, horizon, "standard", seed, **{**SMALL, **kw})


def row(result, mid):
    return next(m for m in result["methods"] if m["id"] == mid)


# --------------------------------------------------------------------------------------------------------------------------------- causality
class Spy(Forecaster):
    """Records every price it is shown, at every fit and every forecast, and forecasts like a random walk."""

    name = "spy"
    short = "Spy"
    label = "Spy"
    min_history = 300
    window, lookback = 400, 300

    def __init__(self):
        self.seen_fit, self.seen_predict = [], []

    def fit(self, h, horizons):
        self.seen_fit.append(h.close.copy())
        return {}

    def predict(self, state, h, horizons, n, rng):
        self.seen_predict.append(h.close.copy())
        return Forecast(np.cumsum(0.01 * rng.standard_normal((n, max(horizons))), axis=1))


def test_a_method_is_never_shown_a_price_from_after_the_date_it_forecasts_from(monkeypatch):
    base = W.to_history(W.garch(1100, seed=1))
    cut = 800
    altered = base.close.copy()
    altered[cut + 1:] = altered[cut + 1:] * np.exp(np.random.default_rng(0).normal(0, 0.2, len(altered) - cut - 1))      # the future after day 800 is replaced by noise
    other = History(altered, base.index)
    seen = {}
    for label, hist in (("real", base), ("noisy", other)):
        spy = Spy()
        monkeypatch.setitem(METHODS, "spy", spy)
        ev.evaluate(hist, ["spy"], 5, "standard", 0, years=1.5, step=7, n_paths=100, refit_days=21)
        seen[label] = spy
    origins = ev._origins(base, ev.settings_for("standard", base, ["spy"], 5, years=1.5, step=7, n_paths=100, refit_days=21))
    assert len(seen["real"].seen_predict) == len(origins) == len(seen["noisy"].seen_predict)
    for i, a, b in zip(origins, seen["real"].seen_predict, seen["noisy"].seen_predict):
        assert a[-1] == pytest.approx(base.close[i]), "a forecast is made from the close of its origin"
        if i <= cut:
            assert np.array_equal(a, b), f"the data at origin {i} changed when only later days did"
        else:
            assert not np.array_equal(a, b)
    for fit_real, fit_noisy in zip(seen["real"].seen_fit, seen["noisy"].seen_fit):
        if fit_real[-1] in base.close[: cut + 1]:
            assert np.array_equal(fit_real, fit_noisy)


# -------------------------------------------------------------------------------------------------------------------------------- the report
@pytest.fixture(scope="module")
def garch_result():
    return run(W.to_history(W.garch(2200, seed=5), high_low=True), ["riskmetrics", "garch", "fhs", "hull_white", "har", "blend"], horizon=5, step=7)


def test_the_report_has_every_method_every_horizon_and_the_scores_that_the_page_reads(garch_result):
    r = garch_result
    assert r["horizons"] == [1, 5, 21] and r["headline"] == 5 and r["baseline"] == BASELINE
    assert [m["id"] for m in r["methods"]] == [BASELINE, "riskmetrics", "garch", "fhs", "hull_white", "har", BLEND]
    o = r["origins"]
    assert o["count"] > 100 and o["step_days"] == 7 and o["first"] < o["last"] and o["paths"] == 300
    for m in r["methods"]:
        assert set(m["scores"]) == {"1", "5", "21"} and m["failures"] == 0, m["id"]
        s = m["scores"]["5"]
        assert 0 < s["n"] <= o["count"] and s["crps"] > 0 and len(s["pit"]) == 10 and abs(sum(s["pit"]) - 1.0) < 1e-9
        assert m["scores"]["1"]["n"] == o["count"], "every origin has a next day"
        assert set(s["coverage"]) == {"50", "90"} and 0 <= s["var5"] <= 1 and set(s["pinball"]) == {"0.05", "0.25", "0.5", "0.75", "0.95"}
    base = row(r, BASELINE)["scores"]["5"]
    assert base["skill"] == 0.0 and "p" in base and "beats_baseline" not in base
    shortest = row(r, "garch")["scores"]["1"]
    assert shortest["n"] >= row(r, "garch")["scores"]["21"]["n"], "a shorter horizon has more outcomes: the last origins have no 21-day outcome yet"


def test_where_volatility_clusters_the_volatility_models_beat_the_random_walk_by_more_than_luck(garch_result):
    for hz in ("1", "5"):
        for mid in ("riskmetrics", "garch", "fhs"):
            s = row(garch_result, mid)["scores"][hz]
            assert s["skill"] > 0.02 and s["beats_baseline"] and s["p_adjusted"] < 0.05, (mid, hz, s["skill"], s["p_adjusted"])
    har = row(garch_result, "har")["scores"]["5"]
    assert har["skill"] > 0 and not har.get("mean"), "a volatility model reports no view on the average"


def test_a_pool_of_forecasts_scores_no_worse_than_the_average_of_its_members(garch_result):
    members = ["riskmetrics", "garch", "fhs", "hull_white", "har"]
    for hz in ("1", "5", "21"):
        pooled = row(garch_result, BLEND)["scores"][hz]["crps"]
        average = np.mean([row(garch_result, m)["scores"][hz]["crps"] for m in members])
        assert pooled <= average * 1.005, "CRPS is convex in the distribution, so a pool cannot do worse than the average of its members"
        assert row(garch_result, BLEND)["scores"][hz]["skill"] > 0.0


def test_in_independent_returns_nothing_beats_the_random_walk_and_the_calibration_of_the_yardstick_is_right():
    r = run(W.to_history(W.iid(2200, 0.01, 0.0003, seed=1)), ["riskmetrics", "historical", "hull_white", "gbm", "markov_returns"], horizon=21)
    for hz in ("1", "5", "21"):
        for m in r["methods"][1:]:
            s = m["scores"][hz]
            assert not s["beats_baseline"] and abs(s["skill"]) < 0.06, (m["id"], hz, s["skill"])
    base = row(r, BASELINE)["scores"]["1"]
    assert base["coverage_p"]["90"] > 0.01 and base["pit_p"] > 0.001, "the yardstick is calibrated where returns are independent and equally volatile"


def test_a_planted_drift_is_found_by_the_trend_followers_and_not_by_the_average_it_is_compared_with():
    r = run(W.to_history(W.slow_drift(2200, seed=2)), ["kalman", "gp_drift", "gp_returns", "historical"], horizon=21)
    for mid in ("kalman", "gp_drift", "gp_returns"):
        s = row(r, mid)["scores"]["21"]
        assert s["mean"]["r2"] > 0.1 and s["mean"]["p"] < 0.01 and s["direction"]["edge"] > 0.1 and s["beats_baseline"], (mid, s["mean"], s["direction"])
    hist = row(r, "historical")["scores"]["21"]
    assert hist["mean"]["r2"] < 0.05, "the average of the last five years is what each mean is compared with, so it cannot beat itself"
    assert r["controls"] == [], "the control of gp_returns (Hull-White) was not run, so no comparison is claimed"


def test_the_controls_say_what_a_method_adds_to_the_simpler_method_it_builds_on():
    r = run(W.to_history(W.reverting(2200, 0.5, seed=1)), ["gp_returns", "analog", "hull_white"], horizon=21)
    controls = {(c["method"], c["horizon"]): c for c in r["controls"]}
    assert ("gp_returns", 21) in controls and ("analog", 21) in controls and all(c["control"] == "hull_white" for c in r["controls"])
    assert controls[("analog", 21)]["skill"] > 0 and controls[("analog", 21)]["p"] < 0.05, "the planted reversal is something the analogues add to the volatility-scaled history"


# ------------------------------------------------------------------------------------------------------------------- control flow and failure
def test_the_run_is_reproducible_for_a_seed_and_the_seed_matters():
    h = W.to_history(W.garch(1700, seed=2))
    a, b, c = (run(h, ["riskmetrics", "garch"], seed=s, years=2.0) for s in (3, 3, 4))
    sa, sb, sc = (row(x, "garch")["scores"]["5"]["crps"] for x in (a, b, c))
    assert sa == pytest.approx(sb, rel=1e-6) and sa != pytest.approx(sc, rel=1e-9)


class Broken(Forecaster):
    name, short, label, min_history = "broken", "Broken", "Broken", 300

    def fit(self, h, horizons):
        raise ForecastError("this method cannot learn from this data")

    def predict(self, state, h, horizons, n, rng):
        raise AssertionError("never reached")


def test_a_method_that_cannot_forecast_is_counted_with_its_reason_and_the_others_are_scored_as_usual(monkeypatch):
    monkeypatch.setitem(METHODS, "broken", Broken())
    r = run(W.to_history(W.garch(1700, seed=3)), ["broken", "riskmetrics"], years=2.0)
    b = row(r, "broken")
    assert b["failures"] == r["origins"]["count"] and "cannot learn" in b["last_error"] and b["scores"]["5"].get("crps") is None
    assert row(r, "riskmetrics")["failures"] == 0 and row(r, "riskmetrics")["scores"]["5"]["crps"] > 0


def test_a_stop_ends_the_run_with_what_is_done_scored_and_an_early_stop_is_refused():
    h = W.to_history(W.garch(1700, seed=4))
    calls = []
    r = run(h, ["riskmetrics"], years=3.0, stop=lambda: len(calls) >= 35, progress=lambda m: calls.append(m))
    assert r["origins"]["count"] == 35 and len(calls) == 35 and "forecast 35 of" in calls[-1]
    with pytest.raises(ForecastError, match="stopped after"):
        run(h, ["riskmetrics"], years=3.0, stop=lambda: True)               # stopped before the first forecast: nothing to say


def test_the_blend_needs_at_least_two_other_methods_to_pool():
    r = run(W.to_history(W.garch(1700, seed=5)), ["riskmetrics", BLEND], years=2.0)
    assert row(r, BLEND)["failures"] == r["origins"]["count"] and row(r, BLEND)["scores"]["5"].get("crps") is None


def test_the_settings_refuse_a_history_that_is_too_short_and_make_the_steps_closer_when_the_stretch_is_short():
    short = W.to_history(W.iid(700, seed=1))
    with pytest.raises(ForecastError, match="too short"):
        ev.settings_for("standard", short, ["garch", "gp_returns"], 21)
    mid = W.to_history(W.iid(1600, seed=1))
    cfg = ev.settings_for("standard", mid, ["garch"], 21)
    assert cfg["step"] <= cfg["span_days"] // 40 and cfg["span_days"] <= mid.n - 2 - cfg["need"] and cfg["horizons"] == (1, 5, 21)
    assert ev.settings_for("quick", mid, ["garch"], 63)["horizons"] == (1, 5, 21, 63)
    for bad in (0, 64):
        with pytest.raises(ForecastError):
            ev.scored_horizons(bad)
    with pytest.raises(ForecastError):
        ev.settings_for("lavish", mid, ["garch"], 21)
    with pytest.raises(ForecastError, match="unknown forecasting method"):
        run(mid, ["astrology"])


# ------------------------------------------------------------------------------------------------------------------------ the forecast from today
def test_the_forecast_from_the_last_day_has_ordered_quantiles_anchored_at_the_price_and_names_what_could_not_forecast(monkeypatch):
    monkeypatch.setitem(METHODS, "broken", Broken())
    h = W.to_history(W.garch(1700, seed=6))
    now = ev.forecast_now(h, [BASELINE, "garch", "hmm", "broken", BLEND], 21, n_paths=1500, seed=1)
    assert now["last_price"] == pytest.approx(h.close[-1]) and now["horizon"] == 21 and now["horizons"] == [1, 5, 21]
    assert set(now["methods"]) == {BASELINE, "garch", "hmm", BLEND} and "broken" in now["errors"]
    for m in now["methods"].values():
        fan = m["fan"]
        assert all(len(v) == 21 for v in fan.values())
        for k in range(21):
            q = [fan[lv][k] for lv in ("0.05", "0.25", "0.5", "0.75", "0.95")]
            assert q == sorted(q) and q[0] > 0
        s = m["summary"]["21"]
        assert s["q05"] < s["median"] < s["q95"] and 0 < s["p_up"] < 1 and s["sd"] > 0
        assert (fan["0.95"][20] - fan["0.05"][20]) > (fan["0.95"][0] - fan["0.05"][0]), "the fan widens"
    wide = now["methods"][BASELINE]["summary"]["21"]
    assert wide["p_loss_5"] == pytest.approx(np.mean(np.random.default_rng(0).normal(0, 0.01 * np.sqrt(21), 100_000) < np.log(0.95)), abs=0.2)
    lone = ev.forecast_now(h, ["riskmetrics", BLEND], 5, n_paths=300)
    assert BLEND not in lone["methods"] and BLEND in lone["errors"] and BASELINE in lone["methods"]


# --------------------------------------------------------------------------------------------------------------------------------- the same dates for every method, and what the page draws
class Flaky(Forecaster):
    """A random walk with a wider spread that refuses at every third forecast, so its average score is over other dates than the others'."""

    name = "flaky"
    short = "Flaky"
    label = "Flaky"
    min_history = 300
    window, lookback = 400, 300

    def __init__(self):
        self.calls = 0

    def fit(self, h, horizons):
        return {}

    def predict(self, state, h, horizons, n, rng):
        self.calls += 1
        if self.calls % 3 == 0:
            raise ForecastError("refused on purpose")
        return Forecast(np.cumsum(0.012 * rng.standard_normal((n, max(horizons))), axis=1))


@pytest.fixture(scope="module")
def flaky_result():
    METHODS["flaky"] = Flaky()
    try:
        history = W.to_history(W.garch(1700, seed=3))
        return history, run(history, ["flaky", "riskmetrics"], horizon=5, years=3.0)
    finally:
        METHODS.pop("flaky", None)


def test_a_method_that_missed_some_dates_is_compared_on_the_dates_all_methods_forecast(flaky_result):
    history, r = flaky_result
    c = r["common"]["5"]
    flaky, rm, base = c["methods"]["flaky"], c["methods"]["riskmetrics"], c["methods"][BASELINE]
    assert c["same_dates"] is False and c["n_common"] < c["n_scorable"] and c["n_common"] == flaky["n_own"], "the common sample is the dates the flaky method managed"
    assert flaky["lost"] > 0 and rm["lost"] == 0 and base["lost"] == 0 and rm["n_own"] == c["n_scorable"] and rm["n_common"] == c["n_common"] == flaky["n_common"]
    s = r["series"]
    ok = np.array([all(s["methods"][m]["crps"][i] is not None for m in (BASELINE, "flaky", "riskmetrics")) for i in range(len(s["dates"]))])
    assert ok.sum() == c["n_common"], "the count is the number of origins where every method has a score"
    crps = lambda m: np.array([np.nan if v is None else v for v in s["methods"][m]["crps"]])
    for m in ("flaky", "riskmetrics"):
        expected = 1.0 - np.nanmean(crps(m)[ok]) / np.nanmean(crps(BASELINE)[ok])                       # recomputed here from the series the page draws
        assert c["methods"][m]["skill_common"] == pytest.approx(expected, abs=1e-12), m
        assert c["methods"][m]["crps_common"] == pytest.approx(np.nanmean(crps(m)[ok]), abs=1e-12)
    assert rm["skill_own"] is not None and rm["skill_common"] is not None
    assert row(r, "flaky")["failures"] == flaky["lost"], "what the common sample drops is what the method reported failing"


def test_the_series_the_page_draws_are_the_realised_returns_and_ordered_forecast_quantiles(flaky_result):
    history, r = flaky_result
    s = r["series"]
    assert s["horizon"] == 5 and len(s["dates"]) == len(s["realised"]) == r["origins"]["count"]
    dates = [str(d.date()) for d in history.index]
    for k in range(0, len(s["dates"]), 7):
        i = dates.index(s["dates"][k])
        want = None if i + 5 >= history.n else float(np.log(history.close[i + 5] / history.close[i]))
        assert (s["realised"][k] is None) if want is None else s["realised"][k] == pytest.approx(want, abs=1e-12), s["dates"][k]
    for m, v in s["methods"].items():
        for a, b, c in zip(v["q05"], v["median"], v["q95"]):
            assert (a is None) == (b is None) == (c is None)
            assert a is None or a <= b <= c, m
        for y, a, c, hit in zip(s["realised"], v["q05"], v["q95"], v["hit90"]):
            if y is not None and a is not None and hit is not None:
                assert bool(hit) == (a <= y <= c) or abs(y - a) < 1e-4 or abs(y - c) < 1e-4, "the 90% hit is whether the outcome lay inside the 5% and 95% quantiles (up to the quantile's interpolation)"


def test_the_rolling_measures_are_the_means_over_the_last_window_of_forecasts(flaky_result):
    _, r = flaky_result
    ro, s = r["rolling"], r["series"]
    w = ro["window"]
    assert ro["coverage_band"][0] < 0.9 < ro["coverage_band"][1] and ro["independent_per_window"] >= 1
    hit = np.array([np.nan if v is None else v for v in s["methods"]["riskmetrics"]["hit90"]])
    y = np.array([np.nan if v is None else v for v in s["realised"]])
    med = np.array([np.nan if v is None else v for v in s["methods"]["riskmetrics"]["median"]])
    for i in (w, w + 5, len(y) - 1):
        win = slice(i + 1 - w, i + 1)
        ok = np.isfinite(hit[win])
        assert ro["methods"]["riskmetrics"]["coverage90"][i] == pytest.approx(hit[win][ok].mean(), abs=1e-12)
        okm = np.isfinite(y[win]) & np.isfinite(med[win])
        assert ro["methods"]["riskmetrics"]["mae"][i] == pytest.approx(np.abs(y[win][okm] - med[win][okm]).mean(), abs=1e-12)
    assert ro["methods"][BASELINE]["skill"][-1] is None, "the baseline has no skill against itself"


def test_without_failures_every_method_is_scored_on_the_same_dates():
    r = run(W.to_history(W.garch(1700, seed=3)), ["riskmetrics", "garch"], years=2.0)
    c = r["common"]["5"]
    assert c["same_dates"] is True and c["n_common"] == c["n_scorable"] and all(m["lost"] == 0 and m["skill_common"] == pytest.approx(m["skill_own"] or 0.0, abs=1e-12) for k, m in c["methods"].items() if k != BASELINE)


def test_the_conclusions_are_drawn_from_the_common_dates_and_say_when_they_were_not_the_same(flaky_result):
    from src.forecast.verdict import conclusions
    _, r = flaky_result
    c = conclusions(r, 5)
    topics = {i["topic"]: i for i in c["items"]}
    assert topics["Sample"]["status"] == "caution" and "did not all forecast on the same dates" in topics["Sample"]["text"] and "Flaky" in topics["Sample"]["text"] and c["common_origins"] == r["common"]["5"]["n_common"]
    assert topics["Scope"]["status"] == "caution"
    assert all(i["status"] in ("established", "not established", "caution") and i["text"] for i in c["items"])
    thin = conclusions({**r, "common": {"5": {**r["common"]["5"], "n_common": 5}}}, 5)
    assert thin["items"][0]["status"] == "not established" and "no comparison between methods is made" in thin["items"][0]["text"]


def test_a_skill_that_exists_in_one_half_only_is_a_caution_and_not_a_finding(flaky_result):
    from src.forecast.verdict import conclusions
    _, r = flaky_result
    fake = {**r, "common": {"5": {**r["common"]["5"], "same_dates": True, "methods": {**r["common"]["5"]["methods"], "riskmetrics": {**r["common"]["5"]["methods"]["riskmetrics"], "skill_common": 0.08, "p_common_adjusted": 0.001, "skill_first_half": 0.2, "skill_second_half": -0.04}}}}}
    items = {i["topic"]: i for i in conclusions(fake, 5)["items"]}
    rm = next(v for k, v in items.items() if "iskmetrics" in k or "RiskMetrics" in k)
    assert rm["status"] == "caution" and "not in both halves" in rm["text"]
    fake["common"]["5"]["methods"]["riskmetrics"]["skill_second_half"] = 0.05
    items = {i["topic"]: i for i in conclusions(fake, 5)["items"]}
    assert next(v for k, v in items.items() if "iskmetrics" in k or "RiskMetrics" in k)["status"] == "established"
