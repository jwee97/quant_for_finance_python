"""The red-flag rules of the Backtest tab and the tear sheet, checked one threshold at a time.

``framework/tearsheet.red_flags`` is the page's verdict: eight fixed rules, each with a threshold, an observed value and a flag. Nothing tested it directly (the tear sheet's own test only builds the
file), so a threshold moved by a notch, a rule that stopped flagging or a wording that lost its number would have passed every test. Each rule is checked here on both sides of its threshold, with the
wording the page shows.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.framework.tearsheet import red_flags


def result(metrics=None, validation=None, tables=None) -> SimpleNamespace:
    """A run that passes every rule, with the pieces a test wants to change."""
    m = {"sharpe": 0.8, "gross_sharpe": 1.0, "ann_turnover": 3.0, "n_days": 10 * 252}
    v = {"deflated_sharpe_probability": 0.97, "n_trials": 4}
    t = {"by_sample": pd.DataFrame({"sharpe": [1.0, 0.6, 0.4]}, index=["development", "validation", "final_holdout"]),
         "benchmarks": pd.DataFrame({"benchmark": ["equal_weight", "risk_parity"], "difference": [0.30, 0.20], "p_value": [0.04, 0.09]})}
    return SimpleNamespace(metrics={**m, **(metrics or {})}, validation={**v, **(validation or {})}, tables={**t, **(tables or {})})


def by_rule(flags: list[dict]) -> dict[str, dict]:
    return {f["rule"]: f for f in flags}


def test_a_run_that_passes_every_rule_has_no_flag_and_each_rule_says_what_it_saw():
    flags = red_flags(result())
    assert [f["rule"] for f in flags] == [
        "Net Sharpe is positive",
        "Positive Sharpe in the final holdout",
        "Beats equal_weight (paired bootstrap, p <= 0.10, positive difference)",
        "Beats risk_parity (paired bootstrap, p <= 0.10, positive difference)",
        "Deflated Sharpe probability >= 0.95",
        "Costs take less than half of the gross Sharpe",
        "Annual turnover below 10x",
        "Evaluation window of at least 8 years",
    ]
    assert not any(f["flag"] for f in flags) and all(isinstance(f["flag"], bool) for f in flags)
    seen = {f["rule"]: f["observed"] for f in flags}
    assert seen["Net Sharpe is positive"] == "+0.80" and seen["Positive Sharpe in the final holdout"] == "+0.40"
    assert seen["Beats equal_weight (paired bootstrap, p <= 0.10, positive difference)"] == "diff +0.30, p = 0.04" and seen["Beats risk_parity (paired bootstrap, p <= 0.10, positive difference)"] == "diff +0.20, p = 0.09"
    assert seen["Deflated Sharpe probability >= 0.95"] == "0.97 with 4 trial(s)" and seen["Costs take less than half of the gross Sharpe"] == "gross +1.00 -> net +0.80"
    assert seen["Annual turnover below 10x"] == "3.0x" and seen["Evaluation window of at least 8 years"] == "10.0 years"
    assert all(f["why"] for f in flags), "every rule says why it matters"


def test_the_net_sharpe_and_the_final_holdout_must_be_above_zero():
    assert by_rule(red_flags(result({"sharpe": 0.01})))["Net Sharpe is positive"]["flag"] is False
    assert by_rule(red_flags(result({"sharpe": 0.0})))["Net Sharpe is positive"]["flag"] is True and by_rule(red_flags(result({"sharpe": -0.3})))["Net Sharpe is positive"]["observed"] == "-0.30"
    weak = by_rule(red_flags(result(tables={"by_sample": pd.DataFrame({"sharpe": [1.0, 0.5, -0.2]}, index=["development", "validation", "final_holdout"])})))
    assert weak["Positive Sharpe in the final holdout"]["flag"] is True and weak["Positive Sharpe in the final holdout"]["observed"] == "-0.20"
    zero = by_rule(red_flags(result(tables={"by_sample": pd.DataFrame({"sharpe": [1.0, 0.5, 0.0]}, index=["development", "validation", "final_holdout"])})))
    assert zero["Positive Sharpe in the final holdout"]["flag"] is True, "a holdout Sharpe of zero is not positive"
    none = red_flags(result(tables={"by_sample": pd.DataFrame({"sharpe": [1.0, 0.5]}, index=["development", "validation"])}))
    assert "Positive Sharpe in the final holdout" not in by_rule(none), "no holdout, no rule"
    assert "Positive Sharpe in the final holdout" not in by_rule(red_flags(SimpleNamespace(metrics=result().metrics, validation=result().validation, tables={})))


def test_a_strategy_must_beat_each_benchmark_with_a_positive_difference_and_a_p_value_of_a_tenth_or_less():
    def flag(difference, p):
        table = pd.DataFrame({"benchmark": ["equal_weight"], "difference": [difference], "p_value": [p]})
        return by_rule(red_flags(result(tables={"benchmarks": table})))["Beats equal_weight (paired bootstrap, p <= 0.10, positive difference)"]["flag"]

    assert flag(0.30, 0.10) is False, "a p-value of exactly a tenth passes"
    assert flag(0.30, 0.11) is True and flag(0.001, 0.05) is False
    assert flag(0.0, 0.01) is True and flag(-0.30, 0.01) is True, "a difference that is not positive fails however small the p-value"
    one = by_rule(red_flags(result(tables={"benchmarks": pd.DataFrame({"benchmark": ["equal_weight"], "difference": [-0.31], "p_value": [0.2345]})})))["Beats equal_weight (paired bootstrap, p <= 0.10, positive difference)"]
    assert one["observed"] == "diff -0.31, p = 0.23"
    gappy = pd.DataFrame({"benchmark": ["equal_weight", "risk_parity"], "difference": [0.3, np.nan], "p_value": [0.04, np.nan]})
    rules = by_rule(red_flags(result(tables={"benchmarks": gappy})))
    assert "Beats equal_weight (paired bootstrap, p <= 0.10, positive difference)" in rules and "Beats risk_parity (paired bootstrap, p <= 0.10, positive difference)" not in rules, "a benchmark with no p-value is not judged"
    assert not any(rule.startswith("Beats") for rule in by_rule(red_flags(result(tables={"benchmarks": pd.DataFrame()})))) and not any(rule.startswith("Beats") for rule in by_rule(red_flags(result(tables={"benchmarks": pd.DataFrame({"benchmark": ["x"], "difference": [1.0]})}))))


def test_the_deflated_sharpe_probability_must_reach_95_per_cent_and_says_when_it_is_a_single_trial():
    def rule(p, trials=4):
        return by_rule(red_flags(result(validation={"deflated_sharpe_probability": p, "n_trials": trials})))["Deflated Sharpe probability >= 0.95"]

    assert rule(0.95)["flag"] is False and rule(0.9499)["flag"] is True and rule(0.2)["observed"] == "0.20 with 4 trial(s)"
    one = rule(0.99, trials=1)
    assert one["observed"] == "0.99 with 1 trial(s) (one trial: the probability the true Sharpe is above zero)" and one["flag"] is False
    assert "(one trial" not in rule(0.99, trials=2)["observed"]


def test_costs_must_leave_at_least_half_the_gross_sharpe_and_a_book_that_never_made_money_gross_is_not_blamed_on_costs():
    def rule(gross, net):
        return by_rule(red_flags(result({"gross_sharpe": gross, "sharpe": net})))["Costs take less than half of the gross Sharpe"]

    assert rule(1.0, 0.5001)["flag"] is False and rule(1.0, 0.4999)["flag"] is True and rule(1.0, 0.4999)["observed"] == "gross +1.00 -> net +0.50"
    assert rule(1.0, 0.5)["flag"] is True, "costs of exactly half are not less than half"
    assert rule(-0.2, -0.5)["flag"] is False and rule(0.0, -0.1)["flag"] is False, "with no gross Sharpe there is nothing for costs to have taken (the net Sharpe has its own rule)"
    assert rule(0.2, 0.15)["flag"] is False and rule(0.2, 0.05)["flag"] is True


def test_turnover_and_the_length_of_the_window_have_their_limits():
    def turnover(x):
        return by_rule(red_flags(result({"ann_turnover": x})))["Annual turnover below 10x"]

    assert turnover(9.99)["flag"] is False and turnover(10.01)["flag"] is True and turnover(10.01)["observed"] == "10.0x" and turnover(25.0)["observed"] == "25.0x"
    assert turnover(10.0)["flag"] is True, "ten times is not below ten times"

    def window(days):
        return by_rule(red_flags(result({"n_days": days})))["Evaluation window of at least 8 years"]

    assert window(8 * 252)["flag"] is False and window(8 * 252 - 1)["flag"] is True and window(8 * 252 - 1)["observed"] == "8.0 years" and window(504)["observed"] == "2.0 years"
    assert window(1525)["observed"] == "6.1 years", "a year is 252 trading days"


def test_every_model_and_detector_must_pass_the_causality_check_when_one_was_run():
    ok = {"momentum": {"ok": True}, "hmm": {"ok": True}}
    rule = "Every model and regime detector passes the causality check"
    passed = by_rule(red_flags(result(validation={"causality": ok})))[rule]
    assert passed["flag"] is False and passed["observed"] == "all pass"
    failed = by_rule(red_flags(result(validation={"causality": {**ok, "reversal": {"ok": False}, "vol": {"ok": False}}})))[rule]
    assert failed["flag"] is True and failed["observed"] == "FAILED: reversal, vol"
    assert rule not in by_rule(red_flags(result())) and rule not in by_rule(red_flags(result(validation={"causality": {}}))), "not run, no rule"


def test_probability_forecasts_must_beat_the_climatological_brier_score_when_there_are_any():
    rule = "Forecast beats the climatological Brier score"
    reliability = pd.DataFrame({"predicted": [0.4, 0.6], "realised": [0.45, 0.55], "n": [100, 100]})
    better = by_rule(red_flags(result({"forecast_brier": 0.2400, "forecast_brier_climatology": 0.2500}, tables={"reliability": reliability})))[rule]
    assert better["flag"] is False and better["observed"] == "0.2400 vs 0.2500"
    worse = by_rule(red_flags(result({"forecast_brier": 0.2600, "forecast_brier_climatology": 0.2500}, tables={"reliability": reliability})))[rule]
    assert worse["flag"] is True and worse["observed"] == "0.2600 vs 0.2500"
    equal = by_rule(red_flags(result({"forecast_brier": 0.25, "forecast_brier_climatology": 0.25}, tables={"reliability": reliability})))[rule]
    assert equal["flag"] is True, "level with always saying the base rate is not better"
    missing = by_rule(red_flags(result(tables={"reliability": reliability})))[rule]
    assert missing["flag"] is True and missing["observed"] == "nan vs nan", "a forecast with no score is not credited"
    assert rule not in by_rule(red_flags(result({"forecast_brier": 0.1, "forecast_brier_climatology": 0.25}))), "no probability forecast, no rule"


def test_the_rules_come_out_in_a_fixed_order_with_the_conditional_ones_in_place():
    full = result({"forecast_brier": 0.2, "forecast_brier_climatology": 0.25}, {"causality": {"m": {"ok": True}}}, {"reliability": pd.DataFrame({"predicted": [0.5], "realised": [0.5], "n": [10]})})
    order = [f["rule"] for f in red_flags(full)]
    assert order[0] == "Net Sharpe is positive" and order[1] == "Positive Sharpe in the final holdout" and order[2].startswith("Beats equal_weight") and order[4] == "Deflated Sharpe probability >= 0.95"
    assert order[-3:] == ["Evaluation window of at least 8 years", "Every model and regime detector passes the causality check", "Forecast beats the climatological Brier score"] and len(order) == 10
