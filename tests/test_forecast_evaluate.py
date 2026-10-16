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
