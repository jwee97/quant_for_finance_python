"""Release 5, section 8.1 and P1-25: the versioned validation policy, the statistical diagnostics read against it, and the governance of a strategy version.

The policy is a declaration, not a default: with none adopted there is no threshold and Gates B and C are blocked. Diagnostics computed under one policy are stale under another. A correctly implemented strategy that
loses fails Gate B and still passes Gate A. A strategy version's identity includes the signal, execution and risk settings; a halted or rolled-back version, or a book that breaches a documented limit, is refused by the
service function every entry point uses. Row-level facts are hand-computed; the diagnostics run on a real saved experiment.
"""

import sqlite3
import warnings

import pandas as pd
import pytest

from src.ml import diagnostics as dx, experiments as xe, gates, governance as gv, policy as pol, prospective as pr, registry as rg, verification as vf
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
BASE = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 400, "refit_every": 126}, "model": {"kind": "ridge", "params": {}}}


@pytest.fixture(scope="module")
def bundle():
    return make_bundle(6, 1300, seed=21, alpha=1.0)


@pytest.fixture(scope="module")
def world(bundle, tmp_path_factory):
    reg = rg.Registry(tmp_path_factory.mktemp("policy") / "ml")
    exp = xe.run_experiment({**BASE, "name": "base"}, bundle=bundle, registry=reg, explain=False, robustness=False)
    return reg, exp


@pytest.fixture
def fresh(tmp_path):
    return rg.Registry(tmp_path / "ml")


def items(res, gate):
    return {i["id"]: i for i in res["gates"][gate]["items"]}


def pass_all_checks(reg, kind="ridge", extra=()):
    for req in list(gates.family_requirements(kind)) + list(extra):
        reg.log_check(req, "fixture", "pass", subject_hash=vf.source_hash(vf.REQUIREMENTS[req][1]), reason="fixture row for the evaluator's logic")


# --------------------------------------------------------------------------------------------------------------------- the policy
def test_a_policy_validates_its_criteria_and_its_hash_follows_every_threshold():
    base = pol.ValidationPolicy("p", "v1", {"min_sharpe_ci_lower": 0.0, "max_pbo": 0.5})
    assert base.content_hash() == pol.ValidationPolicy("p", "v1", {"max_pbo": 0.5, "min_sharpe_ci_lower": 0.0}).content_hash(), "the order the criteria were written in does not matter"
    assert base.content_hash() != pol.ValidationPolicy("p", "v1", {"min_sharpe_ci_lower": 0.0, "max_pbo": 0.51}).content_hash(), "a changed threshold is a different policy"
    assert base.required("max_pbo") and not base.required("max_selection_p") and pol.ValidationPolicy("p", "v1", {"max_pbo": None}).required("max_pbo") is False
    for bad in ({"nonsense": 1}, {"max_pbo": 1.5}, {"max_pbo": True}, {"segment": "holdout"}, {"min_segment_days": 3}, {"delay_sessions": 2.5 + 100}):
        with pytest.raises(pol.PolicyError):
            pol.ValidationPolicy("p", "v1", bad)
    with pytest.raises(pol.PolicyError, match="name and a version"):
        pol.ValidationPolicy("", "v1")


def test_the_example_policy_is_labelled_illustrative_and_nothing_adopts_it_by_default(fresh):
    ex = pol.example_policy()
    assert ex.illustrative and "ILLUSTRATIVE" in ex.description and "not been calibrated" in ex.description
    assert pol.adopted(fresh) is None, "a fresh registry implies no threshold"
    pol.register(fresh, ex)
    assert pol.adopted(fresh) is None, "registering is not adopting"


def test_adoption_is_a_named_decision_append_only_and_the_latest_wins(fresh):
    a, b = pol.ValidationPolicy("a", "v1", {"min_segment_days": 100}), pol.ValidationPolicy("b", "v1", {"min_segment_days": 200})
    with pytest.raises(pol.PolicyError, match="actor and a note"):
        pol.adopt(fresh, a, "", "because")
    with pytest.raises(pol.PolicyError, match="actor and a note"):
        pol.adopt(fresh, a, "jo", "  ")
    r1 = pol.adopt(fresh, a, "jo", "predeclared before the study")
    assert pol.adopted(fresh)[1]["actor"] == "jo" and r1["illustrative"] is False
    pol.adopt(fresh, b, "sam", "second thoughts, before looking")
    policy, rec = pol.adopted(fresh)
    assert policy.name == "b" and rec["actor"] == "sam" and [h["name"] for h in pol.history(fresh)] == ["b", "a"]
    pol.adopt(fresh, a, "jo", "back to the first")
    assert pol.adopted(fresh)[0].name == "a", "re-adopting an earlier policy makes it the one in force again, as a new record"
    with fresh._conn() as c:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            c.execute("DELETE FROM policy_adoptions")
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            c.execute("UPDATE validation_policies SET name='x'")


# --------------------------------------------------------------------------------------------------------------------- Gate B reads the policy
def test_without_an_adopted_policy_gates_b_and_c_are_blocked_and_say_no_threshold_is_implied(world):
    reg, exp = world
    res = gates.evaluate(reg, exp.id)
    b, c = items(res, "B"), items(res, "C")
    assert b["statistical_diagnostics"]["status"] == "blocked" and "no validation policy has been adopted" in b["statistical_diagnostics"]["reason"]
    assert "implies no" in b["statistical_diagnostics"]["reason"] or "no Sharpe" in b["statistical_diagnostics"]["reason"]
    assert c["prospective_evidence"]["status"] == "blocked" and c["operational_policy"]["status"] == "missing"
    assert res["policy_version"] == gates.POLICY_VERSION == "gates-v3"


def test_a_losing_strategy_fails_gate_b_on_the_adopted_policy_and_still_passes_gate_a(world, bundle):
    reg, exp = world
    pass_all_checks(reg, extra=("P1-16", "P1-17", "P1-18", "P1-19"))
    strict = pol.ValidationPolicy("strict", "v1", {"segment": "validation", "min_segment_days": 100, "interval_method": "hac", "min_sharpe_ci_lower": 0.0, "confidence_level": 0.95})
    pol.adopt(reg, strict, "jo", "predeclared: the lower bound of the HAC Sharpe interval must be positive")
    pending = items(gates.evaluate(reg, exp.id), "B")
    assert pending["policy"]["status"] == "ok" and "adopted by jo" in pending["policy"]["reason"]
    assert pending["statistical_diagnostics"]["status"] == "missing" and "POST /api/ml/statistical-diagnostics" in pending["statistical_diagnostics"]["reason"]
    out = dx.run(reg, exp.id, actor="jo", n_boot=300, seed=3)
    assert out["policy_hash"] == strict.content_hash() and out["record_seq"] >= 1 and out["segment"] == "validation"
    res = gates.evaluate(reg, exp.id)
    b = items(res, "B")
    assert b["statistical_machinery"]["status"] == "ok" and "P1-16" in b["statistical_machinery"]["reason"]
    assert b["dx:segment_days"]["status"] == "ok" and b["dx:sharpe_interval"]["status"] == "failed", b["dx:sharpe_interval"]
    assert "does not exceed 0" in b["dx:sharpe_interval"]["reason"] and "naive point estimate" in b["dx:sharpe_interval"]["reason"], "the naive estimate is shown beside the dependence-aware bound"
    assert res["gates"]["B"]["status"] == "failed"
    assert res["gates"]["A"]["status"] == "passed", "an implementation that is correct passes Gate A whatever the strategy earns"
    assert res["gates"]["C"]["status"] != "passed", "Gate C needs Gate B"


def test_diagnostics_computed_under_another_policy_are_stale_until_recomputed(world):
    reg, exp = world
    lenient = pol.ValidationPolicy("lenient", "v1", {"segment": "validation", "min_segment_days": 100, "interval_method": "bootstrap", "min_sharpe_ci_lower": -9.0})
    pol.adopt(reg, lenient, "sam", "wide net, declared before the run")
    stale = items(gates.evaluate(reg, exp.id), "B")["statistical_diagnostics"]
    assert stale["status"] == "stale" and "computed under policy" in stale["reason"]
    dx.run(reg, exp.id, actor="sam", n_boot=300, seed=3)
    b = items(gates.evaluate(reg, exp.id), "B")
    assert b["dx:sharpe_interval"]["status"] == "ok" and b["dx:segment_days"]["status"] == "ok" and "statistical_diagnostics" not in b


def test_every_stored_record_carries_the_policy_it_was_computed_under_and_is_append_only(world):
    reg, exp = world
    rec = dx.latest(reg, exp.id, "diagnostics")
    assert rec["policy_hash"] == pol.adopted(reg)[0].content_hash() and rec["content"]["policy"]["name"] == "lenient"
    sh = rec["content"]["sharpe"]
    assert set(sh) == {"naive", "hac", "bootstrap"} and sh["naive"]["value"] == sh["hac"]["value"] == sh["bootstrap"]["value"], "three uncertainty estimates around one point estimate"
    assert sh["hac"]["dependence"] and sh["hac"]["estimator"] and sh["bootstrap"]["seed"] == 3 and sh["bootstrap"]["replications"] == 300
    with reg._conn() as c:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            c.execute("UPDATE analysis_records SET kind='x'")


def test_numbers_from_unverified_statistical_procedures_are_not_evidence(world):
    reg, exp = world
    reg.log_check("P1-16", "fixture", "fail", subject_hash=vf.source_hash(vf.REQUIREMENTS["P1-16"][1]), reason="HAC reference mismatch")
    b = items(gates.evaluate(reg, exp.id), "B")
    assert b["statistical_machinery"]["status"] == "failed" and "HAC reference mismatch" in b["statistical_machinery"]["reason"]
    reg.log_check("P1-16", "fixture", "pass", subject_hash="an old source hash", reason="stale fixture")
    assert items(gates.evaluate(reg, exp.id), "B")["statistical_machinery"]["status"] == "stale", "a check that passed for other source is stale, not current"
    reg.log_check("P1-16", "fixture", "pass", subject_hash=vf.source_hash(vf.REQUIREMENTS["P1-16"][1]), reason="fixture row for the evaluator's logic")
    assert items(gates.evaluate(reg, exp.id), "B")["statistical_machinery"]["status"] == "ok"


def test_a_reinforcement_learning_result_needs_a_multi_seed_record_and_the_best_seed_is_never_the_evidence(world, bundle):
    reg, exp = world
    with reg._conn() as c:
        c.execute("UPDATE experiments SET model_kind='ppo' WHERE id=?", (exp.id,))
    try:
        seed = items(gates.evaluate(reg, exp.id), "B")["dx:seeds"]
        assert seed["status"] == "missing" and "predeclared seeds" in seed["reason"]
        with pytest.raises(ValueError, match="never selected by its result"):
            from src.ml import multiseed as ms

            ms.run_seeds({}, [1, 2, 3], select="best")
        out = dx.multiseed(reg, exp.id, [1, 2, 3], n_boot=200, bundle=bundle)
        assert out["eligibility"] == "applied" and out["declaration"]["seeds"] == [1, 2, 3] and out["record_seq"] >= 1 and out["subject"] == exp.id
        assert len(out["rows"]) == 3 and out["summary"]["median"] is not None and "no seed is chosen" in out["selection"]
        ok = items(gates.evaluate(reg, exp.id), "B")["dx:seeds"]
        assert ok["status"] == "ok" and "3 seeds" in ok["reason"] and "median" in ok["reason"]
        with pytest.raises(ValueError, match="validation segment only"):
            dx.multiseed(reg, exp.id, [1, 2, 3], segment="test", bundle=bundle)
    finally:
        with reg._conn() as c:
            c.execute("UPDATE experiments SET model_kind='ridge' WHERE id=?", (exp.id,))


def test_the_liquidity_regime_reads_the_traded_volume_when_market_data_is_available_and_says_so_when_it_is_not(world, bundle):
    reg, exp = world
    without = [r for r in dx.run(reg, exp.id, n_boot=200, save=False)["regimes"]["rows"] if r["regime"] == "liquidity_20"]
    assert len(without) == 1 and without[0]["state"] is None and without[0]["eligibility"] == "insufficient" and "no volume" in without[0]["reason"]
    with_volume = [r for r in dx.run(reg, exp.id, n_boot=200, save=False, bundle=bundle)["regimes"]["rows"] if r["regime"] == "liquidity_20"]
    assert with_volume and {r["state"] for r in with_volume} <= {"normal", "stressed"} and all(r["n_days"] > 0 for r in with_volume)


# --------------------------------------------------------------------------------------------------------------------- risk limits
def test_risk_limits_check_books_and_returns_against_hand_computed_numbers():
    lim = gv.RiskLimits({"max_gross": 1.5, "max_net_abs": 0.2, "max_position": 0.4, "max_turnover": 1.0, "max_drawdown": 0.04, "max_daily_loss": 0.04})
    book = {"A": 0.5, "B": -0.3, "C": 0.3}                                                                # gross 1.1, net 0.5, largest 0.5
    names = {b["limit"] for b in gv.check_weights(lim, book)}
    assert names == {"max_net_abs", "max_position", "max_turnover"}, "turnover from an empty book is the gross 1.1"
    assert gv.check_weights(lim, {"A": 0.3, "B": -0.3}, previous={"A": 0.3, "B": -0.3}) == [], "a book inside every limit"
    assert {b["limit"] for b in gv.check_weights(lim, {"A": 1.0, "B": -1.0, "C": 0.1}, previous={"A": 1.0, "B": -1.0, "C": 0.0})} == {"max_gross", "max_position"}
    r = pd.Series([0.01, -0.05, 0.02], index=pd.date_range("2024-01-02", periods=3))                      # equity 1.01, 0.9595, 0.97869: drawdown 0.05 from 1.01
    got = {b["limit"]: b["value"] for b in gv.check_returns(lim, r)}
    assert got["max_daily_loss"] == pytest.approx(0.05) and got["max_drawdown"] == pytest.approx(1 - 0.9595 / 1.01)
    assert gv.check_returns(gv.RiskLimits({"max_drawdown": 0.06, "max_daily_loss": 0.06}), r) == []
    assert gv.RiskLimits({"max_gross": None}).set() == {}, "an unset limit is not a limit"
    with pytest.raises(gv.GovernanceError):
        gv.RiskLimits({"max_gross": -1.0})
    with pytest.raises(gv.GovernanceError, match="unknown risk limit"):
        gv.RiskLimits({"leverage": 2.0})


# --------------------------------------------------------------------------------------------------------------------- strategy identity
def test_a_strategy_version_is_the_model_with_its_signal_execution_and_risk_settings(world, bundle):
    reg, exp = world
    plain = gv.register_version(reg, exp.id, actor="jo")
    assert plain["created"] and plain["version_id"].startswith("sv-") and gv.register_version(reg, exp.id, actor="jo")["created"] is False, "registering the same identity again is a no-op"
    limited = gv.register_version(reg, exp.id, actor="jo", risk_limits=gv.RiskLimits({"max_gross": 1.5}))
    other_limit = gv.register_version(reg, exp.id, actor="jo", risk_limits=gv.RiskLimits({"max_gross": 1.2}))
    assert len({plain["version_id"], limited["version_id"], other_limit["version_id"]}) == 3, "a risk limit is part of what is deployed"
    child_signal = xe.rebacktest(reg, exp.id, {"signal": {"n": 2}}, bundle=bundle)
    child_exec = xe.rebacktest(reg, exp.id, {"execution": {"mode": "realistic_cash_equity", "capital": 1e6}}, bundle=bundle)
    v_signal, v_exec = (gv.register_version(reg, e.id, actor="jo") for e in (child_signal, child_exec))
    assert v_signal["content"]["model_artifact"]["sha256"] == plain["content"]["model_artifact"]["sha256"] == v_exec["content"]["model_artifact"]["sha256"], "the same fitted artifact"
    assert len({plain["version_id"], v_signal["version_id"], v_exec["version_id"]}) == 3, "a different signal or execution setting is a different strategy, however identical the model"
    assert v_signal["content"]["settings"]["signal"] != plain["content"]["settings"]["signal"] and v_exec["content"]["settings"]["execution"] != "vectorized_legacy"
    assert plain["content"]["settings"]["execution"] == "vectorized_legacy" and "code" not in plain["content"], "the platform's code hash is recorded with commitments, not in the identity"
    assert [v["version_id"] for v in gv.versions_of(reg, exp.id)][-1] == other_limit["version_id"] and gv.current_version(reg, exp.id)["version_id"] == other_limit["version_id"]
    with reg._conn() as c:
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            c.execute("UPDATE strategy_versions SET actor='x'")
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            c.execute("UPDATE risk_limits SET actor='x'")
    with pytest.raises(gv.GovernanceError, match="who registered"):
        gv.register_version(reg, exp.id, actor="")


# --------------------------------------------------------------------------------------------------------------------- enforcement
def committed_kwargs(day: int, now_offset_hours: float = 20.0, **over):
    d = pd.Timestamp("2024-03-01") + pd.Timedelta(days=day)
    kw = dict(decision_ts=d, universe_version="u", horizon=5, forecasts={"A": 1.0, "B": 0.5, "C": 0.0}, weights={"A": 0.5, "B": -0.5}, data_cutoff=d, fit_cutoff=d - pd.Timedelta(days=30),
              artifact_hashes={"model": "m", "data": "d"}, model_created=d - pd.Timedelta(days=60), label_start=d + pd.offsets.BDay(1), actor="t", now=(d + pd.Timedelta(hours=now_offset_hours)).timestamp())
    kw.update(over)
    return kw


def test_the_service_function_refuses_a_breach_a_halt_and_a_rollback_and_records_every_refusal(world):
    reg, exp = world
    v = gv.register_version(reg, exp.id, actor="jo", risk_limits=gv.RiskLimits({"max_gross": 1.2, "max_position": 0.6, "max_turnover": 1.5}))["version_id"]
    ok = gv.commit_forecast(reg, v, **committed_kwargs(0))
    assert ok["status"] == "committed" and ok["classification"] == "prospective"
    with pytest.raises(gv.LimitBreach, match="max_position") as exc:
        gv.commit_forecast(reg, v, **committed_kwargs(1, weights={"A": 0.9, "B": -0.1}))
    assert exc.value.breaches[0]["limit"] == "max_position"
    with pytest.raises(gv.LimitBreach, match="max_turnover"):
        gv.commit_forecast(reg, v, **committed_kwargs(2, weights={"A": -0.5, "B": 0.5}), previous_weights={"A": 0.5, "B": -0.5})
    assert gv.check_weights(gv.get_limits(reg, gv.get_version(reg, v)["risk_limits_hash"]), {"A": 0.5, "B": -0.5}) == []
    refused = [e for e in gv.events(reg, v, "limit_breach_refused")]
    assert len(refused) == 2 and refused[0]["detail"]["breaches"], "each refusal is in the audit with the numbers"
    with reg._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM commitments WHERE strategy_version=?", (v,)).fetchone()[0] == 1, "a refused book was never committed"
    gv.halt(reg, v, "drawdown review", "jo")
    with pytest.raises(gv.StrategyHalted, match="halted"):
        gv.commit_forecast(reg, v, **committed_kwargs(4))
    assert gv.events(reg, v, "commitment_refused")[0]["detail"]["reason"].startswith("the version is halted")
    assert gv.resume(reg, v, "review done", "jo")["state"] == "active" and gv.commit_forecast(reg, v, **committed_kwargs(4))["status"] == "committed"
    alt = gv.register_version(reg, exp.id, actor="jo", risk_limits=gv.RiskLimits({"max_gross": 1.0}))["version_id"]
    rb = gv.rollback(reg, v, alt, "limits were too loose", "jo")
    assert rb["state"] == "rolled_back" and gv.state(reg, v)["state"] == "rolled_back"
    with pytest.raises(gv.StrategyHalted, match="rolled_back"):
        gv.commit_forecast(reg, v, **committed_kwargs(5))
    with pytest.raises(gv.GovernanceError, match="cannot be resumed"):
        gv.resume(reg, v, "oops", "jo")
    with pytest.raises(gv.GovernanceError, match="rolled back"):
        gv.halt(reg, v, "again", "jo")
    with pytest.raises(gv.GovernanceError, match="not active"):
        gv.rollback(reg, alt, v, "back again", "jo")


def test_staleness_is_a_limit_too(world):
    reg, exp = world
    v = gv.register_version(reg, exp.id, actor="jo", risk_limits=gv.RiskLimits({"max_staleness_days": 3}))["version_id"]
    with pytest.raises(gv.LimitBreach, match="max_staleness_days"):
        gv.commit_forecast(reg, v, **committed_kwargs(0, data_cutoff=pd.Timestamp("2024-03-01") - pd.Timedelta(days=5)))
    assert gv.commit_forecast(reg, v, **committed_kwargs(1, data_cutoff=pd.Timestamp("2024-03-02") - pd.Timedelta(days=2)))["status"] == "committed"


def test_monitoring_halts_on_a_limit_breach_and_only_alerts_on_drift(world):
    reg, exp = world
    v = gv.register_version(reg, exp.id, actor="jo", risk_limits=gv.RiskLimits({"max_drawdown": 0.04, "max_daily_loss": 0.10}))["version_id"]
    calm = pd.Series([0.001, 0.002, -0.001], index=pd.date_range("2024-03-01", periods=3))
    ok = gv.monitor(reg, v, returns=calm, drift={"n_alert": 0, "prediction_psi": 0.02}, live_ic={"z_vs_backtest": 0.3})
    assert ok["status"] == "ok" and not ok["halted"] and gv.state(reg, v)["state"] == "active"
    drift = gv.monitor(reg, v, returns=calm, drift={"n_alert": 3, "prediction_psi": 0.4}, live_ic={"z_vs_backtest": -2.5})
    assert drift["status"] == "alert" and len(drift["alerts"]) == 3 and not drift["halted"] and gv.state(reg, v)["state"] == "active", "drift is for a person: it alerts and does not halt"
    bad = gv.monitor(reg, v, returns=pd.Series([0.0, -0.03, -0.03], index=calm.index))                    # drawdown 1 - 0.97**2 = 5.91 per cent
    assert bad["status"] == "breach" and bad["halted"] and bad["breaches"][0]["limit"] == "max_drawdown" and gv.state(reg, v)["state"] == "halted"
    assert gv.state(reg, v)["reason"].startswith("max_drawdown") and gv.last_monitoring(reg, v)["detail"]["status"] == "breach"
    kinds = [e["kind"] for e in reversed(gv.events(reg, v))]
    assert kinds == ["version_registered", "monitoring_check", "monitoring_check", "monitoring_check", "halted"]


# --------------------------------------------------------------------------------------------------------------------- Gate C
def test_gate_c_reads_the_operating_controls_and_genuine_prospective_evidence(world):
    reg, exp = world
    pol.adopt(reg, pol.ValidationPolicy("ops", "v1", {"prospective_min_days": 2, "prospective_min_settled": 2, "monitoring_max_age_days": 30}), "jo", "operations bar, declared in advance")
    v = gv.register_version(reg, exp.id, actor="jo", risk_limits=gv.RiskLimits({"max_gross": 1.5, "max_drawdown": 0.5}))["version_id"]
    c = items(gates.evaluate(reg, exp.id), "C")
    assert c["governance_machinery"]["status"] == "missing" and c["ledger_machinery"]["status"] == "missing", "the controls themselves must be verified before their records count"
    pass_all_checks(reg, extra=("P1-24", "P1-25"))
    c = items(gates.evaluate(reg, exp.id), "C")
    assert c["governance_machinery"]["status"] == "ok" and c["ledger_machinery"]["status"] == "ok"
    assert c["strategy_version"]["status"] == "ok" and c["risk_limits"]["status"] == "ok" and c["operating_state"]["status"] == "ok"
    assert c["recovery_plan"]["status"] == "missing" and c["monitoring"]["status"] == "missing"
    assert c["dx:prospective_days"]["status"] == "missing" and "Elapsed time cannot be manufactured" in c["dx:prospective_days"]["reason"]
    with pytest.raises(gv.GovernanceError, match="substance"):
        gv.document_recovery(reg, v, "todo", "jo")
    gv.document_recovery(reg, v, "Halt the version, flatten the paper book at the next close, return to the previous version and reconcile the ledger.", "jo")
    gv.monitor(reg, v, returns=pd.Series([0.001, 0.0, 0.001], index=pd.date_range("2024-03-01", periods=3)), drift={"n_alert": 0, "prediction_psi": 0.01})
    c = items(gates.evaluate(reg, exp.id), "C")
    assert c["recovery_plan"]["status"] == "ok" and c["monitoring"]["status"] == "ok" and "no alert" in c["monitoring"]["reason"]
    # a replay never counts; two prospective days settled do
    replay = gv.commit_forecast(reg, v, **committed_kwargs(0, now=pd.Timestamp("2025-01-01").timestamp()))
    assert replay["classification"] == "replay"
    pr.settle(reg, replay["commitment_id"], {"A": 0.01, "B": 0.0, "C": -0.01}, actor="t")
    c = items(gates.evaluate(reg, exp.id), "C")
    assert c["dx:prospective_days"]["status"] == "missing" and "1 replay commitment(s) are not counted" in c["dx:prospective_days"]["reason"]
    for day in (10, 11):
        res = gv.commit_forecast(reg, v, **committed_kwargs(day))
        assert res["classification"] == "prospective"
        pr.settle(reg, res["commitment_id"], {"A": 0.01, "B": 0.0, "C": -0.01}, actor="t")
    c = items(gates.evaluate(reg, exp.id), "C")
    assert c["dx:prospective_days"]["status"] == "ok" and c["dx:prospective_settled"]["status"] == "ok"
    res = gates.evaluate(reg, exp.id)
    assert res["gates"]["C"]["status"] != "passed" or res["gates"]["B"]["status"] == "passed", "Gate C never passes over a Gate B that has not"
    gv.monitor(reg, v, returns=pd.Series([0.0, -0.4, -0.4], index=pd.date_range("2024-03-01", periods=3)))
    c = items(gates.evaluate(reg, exp.id), "C")
    assert c["operating_state"]["status"] == "failed" and c["monitoring"]["status"] == "failed", "a halted version with a breaching check is not operationally ready"


def test_a_policy_with_no_prospective_requirement_cannot_grant_gate_c_by_silence(fresh):
    counts = {"prospective": {"committed": 50, "settled": 50, "distinct_settled_days": 50}, "replay": {"committed": 0, "settled": 0, "distinct_settled_days": 0}}
    [item] = dx.prospective_items(pol.ValidationPolicy("none", "v1", {"min_segment_days": 100}), counts)
    assert item["status"] == "blocked" and "declares no prospective requirement" in item["reason"]


def test_production_approval_is_a_separate_human_record_tied_to_the_version(world):
    reg, exp = world
    v = gv.register_version(reg, exp.id, actor="jo", risk_limits=gv.RiskLimits({"max_gross": 2.0}))["version_id"]
    with pytest.raises(gv.GovernanceError, match="the person and the reason"):
        gv.approve_deployment(reg, v, actor="jo", note="")
    res = gv.approve_deployment(reg, v, actor="jo", note="reviewed the diagnostics, the limits and the recovery procedure", gate_seqs={"A": 1})
    [rec] = gv.approvals(reg, v)
    assert rec["actor"] == "jo" and rec["detail"]["stage"] == "production" and rec["detail"]["identity"] and res["event_seq"] == rec["seq"]
    other = gv.register_version(reg, exp.id, actor="jo", risk_limits=gv.RiskLimits({"max_gross": 1.9}))["version_id"]
    assert gv.approvals(reg, other) == [], "a change of risk settings is another version, and the approval does not carry over"
    assert "approvals" in gv.summary(reg, v) and gv.summary(reg, v)["state"]["state"] == "active"
