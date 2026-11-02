"""Release 8, P3-11: calibration, the prediction distribution, uncertainty and model disagreement are monitored separately, and a missing estimate is never read as zero.

Each check is worked on data whose answer is known: an interval that is right, too narrow, too wide; an uncertainty that doubles or halves; a forecast distribution that moves; members that diverge or go
missing. The monitor's own state is checked for the case that matters most: when nothing could be computed, it is not 'ok'."""

import json

import numpy as np
import pandas as pd
import pytest

from src.ml import uncertainty_monitor as um

P = um.MonitorPolicy(min_settled=100, min_obs=100)
Z80 = 1.2815515655446004


def intervals(n=400, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = rng.normal(0, 0.02, n), np.full(n, 0.03)
    y = mu + rng.normal(0, 1, n) * sd
    return mu - Z80 * sd * scale, mu + Z80 * sd * scale, y


# ------------------------------------------------------------------------------------------------------------------ calibration
def test_calibration_compares_the_realised_coverage_with_the_stated_level_on_settled_outcomes_only():
    lo, hi, y = intervals(2000)
    ok = um.calibration(lo, hi, y, np.ones(len(y)), P)
    assert ok["state"] == "ok" and ok["value"] == pytest.approx(0.8, abs=0.04) and ok["n"] == 2000 and ok["binomial_se"] == pytest.approx(np.sqrt(0.16 / 2000))
    lo2, hi2, y2 = intervals(2000, scale=0.5)
    narrow = um.calibration(lo2, hi2, y2, np.ones(len(y2)), P)
    assert narrow["state"] == "alert" and narrow["value"] < 0.6 and "too narrow" in narrow["reason"]
    lo3, hi3, y3 = intervals(2000, scale=2.0)
    wide = um.calibration(lo3, hi3, y3, np.ones(len(y3)), P)
    assert wide["state"] in ("watch", "alert") and wide["value"] > 0.9 and "too wide" in wide["reason"]
    settled = np.ones(len(y))
    settled[500:] = 0.0                                                                              # the last 1500 have not settled
    part = um.calibration(lo, hi, y, settled, P)
    assert part["n"] == 500, "only settled outcomes inform calibration"
    y_bad = y.copy()
    y_bad[500:] = 99.0                                                                               # unsettled outcomes cannot move the result
    assert um.calibration(lo, hi, y_bad, settled, P)["value"] == pytest.approx(part["value"])


def test_a_missing_interval_is_neither_a_hit_nor_a_miss_and_too_few_settled_forecasts_say_so():
    lo, hi, y = intervals(400)
    lo[:350] = np.nan
    out = um.calibration(lo, hi, y, np.ones(400), P)
    assert out["state"] == "unavailable" and out["value"] is None and out["missing_interval"] == 350 and "100 are needed" in out["reason"] and "350 forecasts had no interval" in out["reason"]
    enough = intervals(400)[0]
    enough[:300] = np.nan
    assert um.calibration(enough, hi, y, np.ones(400), P)["state"] != "unavailable", "the 100 forecasts that do have an interval are enough, and the 300 without are not counted as misses"
    none = um.calibration([np.nan] * 200, [np.nan] * 200, np.zeros(200), np.ones(200), P)
    assert none["state"] == "unavailable" and none["value"] is None, "no interval at all: unavailable, not a perfect score"


# ------------------------------------------------------------------------------------------------------------------ uncertainty and distribution
def test_the_level_of_uncertainty_is_a_ratio_to_the_reference_and_may_be_too_high_or_too_low():
    ref = np.full(300, 0.03)
    assert um.level_check("uncertainty", np.full(200, 0.03), ref, 1.5, 2.0, P, "predictive standard deviation")["state"] == "ok"
    up = um.level_check("uncertainty", np.full(200, 0.07), ref, 1.5, 2.0, P, "predictive standard deviation")
    assert up["state"] == "alert" and up["value"] == pytest.approx(0.07 / 0.03) and "less sure" in up["reason"]
    down = um.level_check("uncertainty", np.full(200, 0.012), ref, 1.5, 2.0, P, "predictive standard deviation")
    assert down["state"] == "alert" and down["value"] == pytest.approx(0.4) and "more sure than it was" in down["reason"], "a model that has become far more confident is also a finding"
    assert um.level_check("uncertainty", np.full(200, 0.05), ref, 1.5, 2.0, P, "x")["state"] == "watch"
    gap = um.level_check("uncertainty", np.full(200, np.nan), ref, 1.5, 2.0, P, "predictive standard deviation")
    assert gap["state"] == "unavailable" and gap["value"] is None and gap["missing"] == 200, "no estimate is not an estimate of zero"
    assert um.level_check("uncertainty", np.full(200, 0.03), np.zeros(300), 1.5, 2.0, P, "x")["reason"].startswith("the reference mean")


def test_a_moved_forecast_distribution_is_an_alert_and_an_unchanged_one_is_not():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 600)
    same = um.distribution_check(rng.normal(0, 1, 300), ref, P)
    moved = um.distribution_check(rng.normal(1.5, 1, 300), ref, P)
    assert same["state"] == "ok" and moved["state"] == "alert" and moved["value"] > 0.25 and moved["ks_p_value"] < 1e-6
    assert um.distribution_check(ref[:50], ref, P)["state"] == "unavailable"


# ------------------------------------------------------------------------------------------------------------------ disagreement and missing members
def members(n=600, split=300, recent_scale=1.0, drop=None, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.02, n)
    m = pd.DataFrame({f"m{j}": base + rng.normal(0, 0.01, n) * np.where(np.arange(n) >= split, recent_scale, 1.0) for j in range(3)})
    for c in drop or []:
        m[c] = np.nan
    return m


def test_the_spread_between_members_is_monitored_and_a_missing_member_is_handled_as_the_policy_says():
    stable = um.disagreement(members(), P, 300)
    assert stable["state"] == "ok" and stable["members"] == ["m0", "m1", "m2"] and stable["missing_members"] == []
    diverged = um.disagreement(members(recent_scale=3.0), P, 300)
    assert diverged["state"] == "alert" and diverged["value"] > 2.0
    degraded = um.disagreement(members(drop=["m2"]), P, 300)
    assert degraded["state"] == "degraded" and degraded["members"] == ["m0", "m1"] and degraded["missing_members"] == ["m2"] and "did not report" in degraded["reason"]
    strict = um.disagreement(members(drop=["m2"]), um.MonitorPolicy(min_settled=100, min_obs=100, on_missing_member="alert"), 300)
    assert strict["state"] == "alert", "the policy may treat a missing member as an alert"
    one_left = um.disagreement(members(drop=["m1", "m2"]), P, 300)
    assert one_left["state"] == "unavailable" and one_left["value"] is None and "at least 2" in one_left["reason"], "one member cannot disagree with itself: the spread is undefined, not zero"
    single = um.disagreement(None, P, 0)
    assert single["state"] == "not_applicable" and single["value"] is None


# ------------------------------------------------------------------------------------------------------------------ the monitor's own state and escalation
def test_a_monitor_with_nothing_computed_is_unavailable_never_ok_and_gaps_make_a_healthy_one_degraded():
    checks = [um._unavailable(c, "no data") for c in um.CHECKS]                                       # noqa: SLF001
    assert um.overall(checks)["state"] == "unavailable" and "absence of evidence" in um.overall(checks)["reason"]
    mixed = [{"check": "calibration", "state": "ok"}, {"check": "distribution", "state": "ok"}, {"check": "uncertainty", "state": "unavailable"}, {"check": "disagreement", "state": "unavailable"}]
    assert um.overall(mixed)["state"] == "degraded" and "uncertainty, disagreement" in um.overall(mixed)["reason"]
    assert um.overall([{"check": "a", "state": "alert"}, {"check": "b", "state": "unavailable"}])["state"] == "alert"
    assert um.overall([{"check": "a", "state": "ok"}, {"check": "b", "state": "ok"}])["state"] == "ok"


def test_an_alert_is_escalated_by_persistence_and_nothing_else_has_a_level():
    pol = um.MonitorPolicy(escalate_after=2)
    assert [um.escalate("alert", k, pol) for k in (1, 2, 3, 4, 5, 6, 9)] == ["info", "info", "warning", "warning", "critical", "critical", "critical"]
    assert all(um.escalate(s, 5, pol) is None for s in ("ok", "watch", "degraded", "unavailable", "not_applicable"))
    for bad in ({"interval_level": 1.2}, {"uncertainty_ratio_watch": 0.9}, {"uncertainty_ratio_watch": 2.0, "uncertainty_ratio_alert": 1.5}, {"min_members": 1}, {"on_missing_member": "ignore"}, {"escalate_after": 0}):
        with pytest.raises(ValueError):
            um.MonitorPolicy(**bad)


def frame(n_days=500, assets=4, seed=5, std_scale_recent=1.0, with_std=True, std_nan=False):
    rng = np.random.default_rng(seed)
    ix = pd.bdate_range("2020-01-01", periods=n_days)
    mi = pd.MultiIndex.from_product([ix, [f"A{i}" for i in range(assets)]], names=["date", "asset"])
    n = len(mi)
    mu = rng.normal(0, 0.02, n)
    recent = (mi.get_level_values(0) > ix[349]).astype(float)
    sd = 0.03 * (1.0 + (std_scale_recent - 1.0) * recent)
    y = mu + rng.normal(0, 1, n) * 0.03
    f = pd.DataFrame({"mean": mu}, index=mi)
    if with_std:
        f["std"] = np.nan if std_nan else sd
        f["lower"], f["upper"] = mu - Z80 * sd, mu + Z80 * sd
    return f, pd.Series(y, index=mi), pd.Series(mi.get_level_values(0) + pd.tseries.offsets.BDay(5), index=mi), ix


def test_the_monitor_runs_the_four_checks_apart_serialises_nulls_and_counts_consecutive_alerts():
    f, y, le, ix = frame(std_scale_recent=3.0)
    out = um.monitor(f, y, le, as_of=ix[-1], reference_end=ix[349], policy=P)
    by = {c["check"]: c for c in out["checks"]}
    assert set(by) == set(um.CHECKS) and by["uncertainty"]["state"] == "alert" and by["calibration"]["state"] in ("watch", "alert") and "too wide" in by["calibration"]["reason"], (
        "the stated uncertainty tripled while the realised errors did not: the level shows it, and so does the coverage, which is now far above its stated level")
    assert by["disagreement"]["state"] == "not_applicable" and out["overall"]["state"] == "alert"
    assert by["uncertainty"]["level"] == "info" and by["uncertainty"]["consecutive_alerts"] == 1
    json.dumps(out, allow_nan=False)
    nxt = um.monitor(f, y, le, as_of=ix[-1], reference_end=ix[349], policy=um.MonitorPolicy(min_settled=100, min_obs=100, escalate_after=1), previous=out)
    assert {c["check"]: c["level"] for c in nxt["checks"]}["uncertainty"] == "warning" and nxt["streak"]["uncertainty"] == 2
    gone = um.monitor(f, y, le, as_of=ix[-1], reference_end=ix[349], policy=P, members_missing_reason="the members' forecasts were not kept")
    gd = {c["check"]: c for c in gone["checks"]}["disagreement"]
    assert gd["state"] == "unavailable" and gd["value"] is None and "not kept" in gd["reason"], "an ensemble whose members were not kept is missing them: that is a gap, not 'a single model'"
    healthy = um.monitor(*frame(std_scale_recent=1.0)[:3], as_of=ix[-1], reference_end=ix[349], policy=P, members_missing_reason="gone")
    assert healthy["overall"]["state"] in ("degraded", "alert", "watch") and healthy["overall"]["state"] != "ok", "a gap keeps a healthy monitor from being called ok"
    only_recent_settled = um.monitor(f, y, le, as_of=ix[352], reference_end=ix[349], policy=P)
    assert {c["check"]: c for c in only_recent_settled["checks"]}["calibration"]["state"] == "unavailable", "three days after the reference there are too few settled forecasts"


def test_a_model_with_no_uncertainty_or_a_nan_one_is_not_applicable_or_unavailable_and_never_healthy():
    f, y, le, ix = frame(with_std=False)
    out = um.monitor(f, y, le, as_of=ix[-1], reference_end=ix[349], policy=P)
    by = {c["check"]: c for c in out["checks"]}
    assert by["calibration"]["state"] == "not_applicable" and by["uncertainty"]["state"] == "not_applicable" and by["distribution"]["state"] in ("ok", "watch", "alert")
    g, y2, le2, ix2 = frame(std_nan=True)
    out2 = um.monitor(g, y2, le2, as_of=ix2[-1], reference_end=ix2[349], policy=P)
    by2 = {c["check"]: c for c in out2["checks"]}
    assert by2["uncertainty"]["state"] == "unavailable" and by2["uncertainty"]["value"] is None and by2["uncertainty"]["missing"] > 0
    assert out2["overall"]["state"] in ("degraded", "alert"), "a missing uncertainty estimate never makes the monitor look healthy"
    json.dumps(out2, allow_nan=False)
