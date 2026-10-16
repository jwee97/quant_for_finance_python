"""The verdict of a forecast evaluation, in words: what it claims for each kind of result, and that every number it quotes is the one in the report."""

from __future__ import annotations


from src.forecast import evaluate as ev
from src.forecast.verdict import ALPHA, ICON, judge
from tests import forecast_worlds as W


def score(skill, p_adj=0.5, cov90=0.9, cov90_p=0.8, mean=None, direction=None, crps=0.025, beats=None):
    s = {"n": 100, "crps": crps, "skill": skill, "p": p_adj / 2, "p_adjusted": p_adj, "coverage": {"50": 0.5, "90": cov90}, "coverage_p": {"50": 0.9, "90": cov90_p}, "var5": 0.05, "var5_p": 0.9,
         "pit": [0.1] * 10, "pit_p": 0.5, "beats_baseline": (skill > 0 and p_adj < ALPHA) if beats is None else beats}
    if mean is not None:
        s["mean"] = mean
    if direction is not None:
        s["direction"] = direction
    return s


def method(mid, short, skill, hz="21", failures=0, **kw):
    return {"id": mid, "label": f"{short} (long name)", "short": short, "failures": failures, "scores": {hz: score(skill, **kw)}}


def report(methods, controls=()):
    base = {"id": "random_walk", "label": "Random walk, constant volatility", "short": "Random walk", "failures": 0, "scores": {"21": {"n": 100, "crps": 0.03, "skill": 0.0, "coverage": {"50": 0.5, "90": 0.9}}}}
    return {"headline": 21, "baseline": "random_walk", "horizons": [1, 5, 21], "controls": list(controls), "methods": [base] + methods,
            "origins": {"count": 120, "first": "2020-01-02", "last": "2026-09-18", "step_days": 10, "refit_days": 63, "years": 6.0, "paths": 1000, "history_days": 4000, "effort": "standard"}}


def test_methods_that_beat_the_random_walk_by_more_than_luck_make_a_good_verdict_that_names_them_and_the_best_by_its_number():
    r = report([method("garch", "GJR-GARCH", 0.081, p_adj=0.004), method("fhs", "Filtered", 0.064, p_adj=0.02), method("hmm", "Hidden Markov", 0.03, p_adj=0.3), method("jump", "Jump", -0.05, p_adj=0.9)])
    v = judge(r)
    assert v["level"] == "good" and ICON[v["level"]] == "✓"
    assert v["headline"] == "2 methods described the spread of 21-day returns better than the random walk, the best by 8%"
    text = " ".join(v["reasons"])
    assert "GJR-GARCH (+8.1%)" in text and "Filtered (+6.4%)" in text and "Hidden Markov (+3.0%)" not in text.split("Holm-adjusted p under 5%): ")[1].split(".")[0]
    assert "adjusted p = 0.004" in text and "Worse than the random walk: Jump (-5%)" in text and "Tested on 120 forecasts made between 2020-01-02 and 2026-09-18" in text
    assert "A forecast is not a strategy" in v["reasons"][-1]


def test_a_gain_that_luck_could_explain_is_a_mixed_verdict_and_no_gain_at_all_is_a_bad_one():
    mixed = judge(report([method("a", "Alpha", 0.04, p_adj=0.4), method("b", "Beta", -0.01)]))
    assert mixed["level"] == "mixed" and "not by more than luck could explain" in mixed["headline"] and "The best was Alpha at +4.0%" in " ".join(mixed["reasons"])
    bad = judge(report([method("a", "Alpha", -0.02), method("b", "Beta", -0.2)]))
    assert bad["level"] == "bad" and "No method forecast the spread of 21-day returns better than the random walk" in bad["headline"]
    assert "The best, Alpha, scored -2.0%" in " ".join(bad["reasons"]) and ICON["bad"] == "✗"
    none = judge(report([{"id": "x", "label": "X", "short": "X", "failures": 5, "scores": {"21": {"n": 0}}}]))
    assert none["level"] == "bad" and none["headline"] == "Nothing could be scored"


def test_a_win_that_is_too_small_to_matter_does_not_count_even_when_it_is_significant():
    v = judge(report([method("a", "Alpha", 0.004, p_adj=0.001)]))
    assert v["level"] == "mixed", "under one percent of the baseline's score is not a win"


def test_the_calibration_line_quotes_the_coverage_and_says_when_it_is_not_consistent_with_the_nominal():
    ok = " ".join(judge(report([method("g", "Garch", 0.08, p_adj=0.001, cov90=0.91, cov90_p=0.7)]))["reasons"])
    assert "held the outcome 91% of the time (consistent with 90%)" in ok
    off = " ".join(judge(report([method("g", "Garch", 0.08, p_adj=0.001, cov90=0.7, cov90_p=0.0001)]))["reasons"])
    assert "held the outcome 70% of the time (not consistent with 90%)" in off


def test_the_average_return_is_said_to_be_beaten_only_when_the_test_clears_the_adjusted_threshold():
    none = " ".join(judge(report([method("k", "Kalman", 0.02, p_adj=0.2, mean={"r2": 0.01, "p": 0.04, "average": 0.0}),
                                  method("a", "ARMA", 0.0, mean={"r2": -0.05, "p": 0.9, "average": 0.0})]))["reasons"])
    assert "none of the 2 methods that forecast one beat simply using the historical average" in none and "The best was Kalman with an out-of-sample R-squared of +1.0%" in none
    won = " ".join(judge(report([method("k", "Kalman", 0.02, p_adj=0.2, mean={"r2": 0.2, "p": 0.001, "average": 0.0}), method("a", "ARMA", 0.0, mean={"r2": -0.05, "p": 0.9, "average": 0.0})]))["reasons"])
    assert "Kalman forecast it better than the historical average of the last five years (out-of-sample R-squared +20.0%, Clark-West p = 0.001" in won


def test_direction_is_compared_with_always_saying_up_and_one_that_beats_it_is_flagged_as_needing_more_evidence():
    weak = {"hit": 0.68, "always_up": 0.68, "edge": 0.0, "p": 0.5}
    m1 = method("h", "Historical", 0.01, mean={"r2": 0.0, "p": 0.5, "average": 0.0}, direction=weak)
    text = " ".join(judge(report([m1]))["reasons"])
    assert "always saying up was right 68% of the time" in text and "none beat always saying up by more than luck allows" in text
    strong = {"hit": 0.8, "always_up": 0.68, "edge": 0.12, "p": 0.0001}
    m2 = method("k", "Kalman", 0.01, mean={"r2": 0.1, "p": 0.01, "average": 0.0}, direction=strong)
    assert "at least one beat always saying up, which needs checking on other stocks and periods" in " ".join(judge(report([m2]))["reasons"])


def test_a_control_comparison_says_whether_a_method_adds_anything_to_the_simpler_one_it_builds_on():
    ms = [method("gp_returns", "GP regression", 0.06, p_adj=0.01), method("hull_white", "Hull-White", 0.058, p_adj=0.01)]
    no = " ".join(judge(report(ms, [{"method": "gp_returns", "control": "hull_white", "horizon": 21, "skill": 0.004, "p": 0.6, "n": 100}]))["reasons"])
    assert "Does GP regression add anything to its simpler control, Hull-White? Its forecasts scored +0.4% against it (p = 0.60): no, what it shows is what the control already has." in no
    yes = " ".join(judge(report(ms, [{"method": "gp_returns", "control": "hull_white", "horizon": 21, "skill": 0.09, "p": 0.002, "n": 100}]))["reasons"])
    assert "yes, by more than luck allows" in yes
    other = " ".join(judge(report(ms, [{"method": "gp_returns", "control": "hull_white", "horizon": 5, "skill": 0.09, "p": 0.002, "n": 100}]))["reasons"])
    assert "Does GP regression" not in other, "only the comparison at the horizon being judged is quoted"


def test_methods_that_could_not_forecast_at_some_origins_are_named_with_how_many():
    v = judge(report([method("a", "Alpha", 0.05, p_adj=0.001), method("h", "Hidden Markov", 0.01, failures=7)]))
    assert "Could not forecast at some origins: Hidden Markov (7 of 120)" in " ".join(v["reasons"])


def test_the_verdict_of_a_real_evaluation_quotes_the_numbers_of_its_report():
    r = ev.evaluate(W.to_history(W.garch(1800, seed=5)), ["riskmetrics", "garch"], 5, "standard", 0, years=3.0, step=14, n_paths=300)
    v = judge(r)
    best = max((m["scores"]["5"]["skill"] for m in r["methods"] if m["id"] != "random_walk"))
    assert f"{best * 100:+.1f}%" in " ".join(v["reasons"]), "the verdict quotes the best method's skill as the report has it"
    assert str(r["origins"]["count"]) in v["reasons"][0] and len(v["reasons"]) >= 5
    assert judge(r, horizon=1)["reasons"][0].count("scored on the 1 trading days") == 1
