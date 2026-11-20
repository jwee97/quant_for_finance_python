"""The evidence checklist applies stated rules to recorded numbers; here each rule is exercised on hand-built reports so every status is reachable and none is invented."""

from __future__ import annotations

from src.ml import evidence as evd


def report(**kw):
    base = {"id": "x", "name": "x", "config": {"signal": {"method": "decision_policy", "net": None}, "seed": 0}, "config_hash": "abc123", "dataset_key": "def456", "warnings": [],
            "audit": {"features": {"f": {}}, "stale": {"assets": [], "longest_run": 0, "rule": "r"}}, "folds": {"n": 5, "n_skipped": 0, "first": {"fit_cutoff": "2020-01-02", "label_end_max": "2020-01-02", "purge_rule": "p"},
                                                                                                         "last": {"fit_cutoff": "2021-01-04", "label_end_max": "2020-12-31"}},
            "holdout": {"status": "protected"}, "overfit": {"n_trials": 3, "deflated_sharpe_probability": 0.97, "sharpe": {"train": 2.0, "validation": 1.5, "test": 1.2, "forward": 1.0}},
            "robustness": {"tests": [{"name": "t"}], "n_pass": 5, "n_warn": 0, "n_fail": 0, "n_scored": 5, "score": 1.0}, "execution": {"mode": "realistic_cash_equity"}, "provenance": {"source": "platform"}}
    base.update(kw)
    return base


def by(rep, controls=None):
    return {c["id"]: c for c in evd.build(rep, controls)["checks"]}


def test_a_clean_record_passes_where_it_can_and_is_unknown_where_nothing_was_run():
    c = by(report())
    assert all(c[k]["status"] == "pass" for k in ("features_causal", "oos_lineage", "holdout", "multiple_testing", "out_of_sample", "robustness", "execution", "reproducible", "data_source", "warnings"))
    assert c["random_signals"]["status"] == "unknown"
    out = evd.build(report())
    assert sum(out["counts"].values()) == len(out["checks"]) and "combined score" in out["reading"]


def test_each_failure_rule_fires_on_the_number_it_names():
    assert by(report(folds={"first": {"fit_cutoff": "2020-01-02", "label_end_max": "2020-02-01"}}))["oos_lineage"]["status"] == "fail"
    assert by(report(overfit={"n_trials": 40, "deflated_sharpe_probability": 0.2, "sharpe": {"validation": 1.0, "test": -0.3}}))["multiple_testing"]["status"] == "fail"
    assert by(report(overfit={"n_trials": 40, "deflated_sharpe_probability": 0.2, "sharpe": {"validation": 1.0, "test": -0.3}}))["out_of_sample"]["status"] == "fail"
    assert by(report(overfit={"deflated_sharpe_probability": 0.7, "sharpe": {"validation": 2.0, "test": 0.6}}))["multiple_testing"]["status"] == "warn"
    assert by(report(overfit={"deflated_sharpe_probability": 0.99, "sharpe": {"validation": 2.0, "test": 0.6}}))["out_of_sample"]["status"] == "warn", "a test Sharpe under half the validation Sharpe is a warning"
    assert by(report(robustness={"tests": [1], "n_pass": 3, "n_warn": 0, "n_fail": 1, "n_scored": 4}))["robustness"]["status"] == "fail"
    assert by(report(provenance={"source": "simulated"}))["data_source"]["status"] == "fail"
    assert by(report(holdout={"status": "exposed_exploratory", "note": "looked"}))["holdout"]["status"] == "warn"


def test_missing_records_are_unknown_not_pass():
    c = by(report(audit=None, folds={}, holdout={"status": "unrecorded"}, robustness={}, config_hash=None))
    assert {c[k]["status"] for k in ("features_causal", "oos_lineage", "holdout", "robustness", "reproducible")} == {"unknown"}


def test_the_stale_price_line_states_the_finding_not_the_mere_presence_of_the_block():
    assert "No run of repeated prices" in by(report())["features_causal"]["detail"]
    stale = {"assets": ["A", "B"], "longest_run": 9, "rule": "5 or more"}
    assert "2 asset(s)" in by(report(audit={"features": {"f": {}}, "stale": stale}))["features_causal"]["detail"]


def test_the_random_signal_control_and_the_benchmark_match_rules():
    assert by(report(), {"share_of_random_at_least_as_good": 0.02, "draws": 40, "replay_check": {"reproduces": True}})["random_signals"]["status"] == "pass"
    assert by(report(), {"share_of_random_at_least_as_good": 0.15, "draws": 40})["random_signals"]["status"] == "warn"
    assert by(report(), {"share_of_random_at_least_as_good": 0.6, "draws": 40})["random_signals"]["status"] == "fail"
    assert by(report(), {"share_of_random_at_least_as_good": 0.01, "draws": 40, "replay_check": {"reproduces": False}})["random_signals"]["status"] == "warn", "a control whose replay did not reproduce the run cannot pass"
    assert by(report(trading={"analytics": {"avg_net_exposure": 0.2}}))["benchmark_match"]["status"] == "warn"
    assert by(report(trading={"analytics": {"avg_net_exposure": 0.97}}))["benchmark_match"]["status"] == "pass"
    neutral = by(report(config={"signal": {"method": "top_n_long_short", "net": 0.0}}))
    assert neutral["benchmark_match"]["status"] == "warn"


def test_a_simulation_with_a_violation_fails_its_line():
    sim = {"oos_violations": ["2020-01-03: scored by a model fitted after the date"], "model_versions": [1, 2], "selection": {"n_candidates": 4}, "comparison": {"scheduled": {"sharpe": 1.0}}}
    assert by(report(simulation=sim))["simulation"]["status"] == "fail"
    assert by(report(simulation={**sim, "oos_violations": []}))["simulation"]["status"] == "pass"
