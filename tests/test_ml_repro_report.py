"""Release 8, P3-12: reports built from persisted artifact hashes and policy versions, which do not change when the world does.

A report is a pure function of a snapshot; the snapshot is stored with it. The tests build one on a small experiment, rebuild it from its own snapshot, then change the world (a newer diagnostic, a replaced
artifact, a changed policy) and show that the stored report is byte-for-byte what it was, that the change is reported as staleness, and that a new build is a new report."""

import copy
import json

import pandas as pd
import pytest

from src.ml import diagnostics as dx, experiments as xp, r6_reports as r6, registry as rg, repro_report as rp
from tests.ml_helpers import make_bundle


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    reg = rg.Registry(tmp_path_factory.mktemp("rep") / "ml")
    bundle = make_bundle(8, 1150, seed=77, alpha=1.0)
    exp = xp.run_experiment({"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}},
                             "signal": {"method": "top_n_long_short", "n": 3, "holding_period": 1}}, bundle=bundle, registry=reg, explain=True, robustness=False)
    dx.store(reg, "alpha_diagnostics", exp.id, r6.alpha_diagnostics(reg, exp.id), actor="test")
    return reg, exp


def test_a_report_is_built_from_a_snapshot_and_lists_what_was_and_was_not_computed(world):
    reg, exp = world
    out = rp.build(reg, exp.id, actor="alice")
    assert out["status"] == "built" and out["report_id"].startswith("rp-") and len(out["report_hash"]) == 16
    rec = rp.get(reg, out["report_id"])
    c = rec["content"]
    assert set(rp.SECTIONS) <= set(c) and c["header"]["experiment_id"] == exp.id and c["header"]["config_hash"] == rec["inputs"]["config_hash"] and c["report_version"] == rp.VERSION
    assert c["header"]["artifact_hashes"] and all(len(h) == 32 for h in c["header"]["artifact_hashes"].values() if h), "every artifact the report rests on is named by its content hash"
    up = c["uncertainty_selection"]
    assert "alpha_diagnostics" in up["diagnostics_present"] and "statistical_diagnostics" in up["diagnostics_not_computed"] and "drift_report" in up["diagnostics_not_computed"]
    assert "statistical_diagnostics: not computed" in c["limitations"], "a diagnostic that was never computed is a limitation, not a blank"
    assert set(c["gates"]["gates"]) == {"A", "B", "C"} and c["gates"]["policy_version"] and c["header"]["policy_versions"]["gate_policy"] == c["gates"]["policy_version"]
    for n, g in c["gates"]["gates"].items():
        if g["status"] != "passed":
            assert any(x.startswith(f"Gate {n} is {g['status']}") for x in c["limitations"]), "a gate that has not passed explains itself in the limitations"
    assert c["simulated_data_label"] is None or "SIMULATED" in c["simulated_data_label"]
    json.dumps(c, allow_nan=False)


def test_the_same_inputs_are_the_same_report_and_a_stored_report_rerenders_from_its_own_snapshot(world):
    reg, exp = world
    first = rp.build(reg, exp.id, actor="alice")
    again = rp.build(reg, exp.id, actor="bob", now=1.0)
    assert again["status"] == "unchanged" and again["report_id"] == first["report_id"] and again["report_hash"] == first["report_hash"], "building again from unchanged evidence is the same report, whenever it is asked for"
    r = rp.reproduce(reg, first["report_id"])
    assert r["reproduced"] is True and r["stored_hash"] == r["recomputed_hash"]
    snap = copy.deepcopy(rp.get(reg, first["report_id"])["snapshot"])
    snap["report"]["metrics"]["oos"] = {"sharpe": 99.0}
    assert rp.digest(rp.render(snap)) != first["report_hash"], "a different snapshot is a different report: the content depends on the inputs it records"
    assert rp.digest(rp.render(copy.deepcopy(rp.get(reg, first["report_id"])["snapshot"]))) == first["report_hash"]
    with pytest.raises(rp.ReportError, match="records who asked"):
        rp.build(reg, exp.id, actor="")
    with pytest.raises(rp.ReportError, match="unknown report"):
        rp.get(reg, "rp-nope")


def test_a_historical_report_does_not_change_when_the_world_does_and_the_staleness_is_reported(world, tmp_path):
    reg, exp = world
    first = rp.build(reg, exp.id, actor="alice")
    before = json.dumps(rp.get(reg, first["report_id"]), sort_keys=True, default=str)
    clean = rp.verify_inputs(reg, first["report_id"])
    assert clean["stale"] is False and clean["changed"] == [] and "artifact:config" in clean["unchanged"]
    dx.store(reg, "alpha_diagnostics", exp.id, r6.alpha_diagnostics(reg, exp.id), actor="test")                 # a newer look at the same experiment
    stale = rp.verify_inputs(reg, first["report_id"])
    assert stale["stale"] is True and [c["input"] for c in stale["changed"]] == ["diagnostic:alpha_diagnostics"] and stale["changed"][0]["now"]["seq"] > stale["changed"][0]["was"]["seq"]
    cfg_path = next(a["path"] for a in reg.get_experiment(exp.id)["artifacts"] if a["name"] == "config")
    original = open(cfg_path, "rb").read()
    try:
        with open(cfg_path, "ab") as fh:
            fh.write(b"\n# edited after the report was built\n")                                              # an artifact replaced on disk
        edited = rp.verify_inputs(reg, first["report_id"])
        assert "artifact:config" in [c["input"] for c in edited["changed"]] and edited["changed"][0].get("was") != edited["changed"][0].get("now")
        assert json.dumps(rp.get(reg, first["report_id"]), sort_keys=True, default=str) == before, "the stored report is exactly what it was"
        newer = rp.build(reg, exp.id, actor="alice")
        assert newer["status"] == "built" and newer["report_id"] != first["report_id"], "the changed world makes a new report with a new id"
    finally:
        with open(cfg_path, "wb") as fh:
            fh.write(original)
    ids = [r["report_id"] for r in rp.list_reports(reg, exp.id)]
    assert first["report_id"] in ids and newer["report_id"] in ids and ids.index(newer["report_id"]) < ids.index(first["report_id"]), "both reports stay retrievable, the newest first"
    assert rp.reproduce(reg, first["report_id"])["reproduced"] is True and rp.reproduce(reg, newer["report_id"])["reproduced"] is True


def test_the_markdown_carries_the_evidence_label_the_limitations_and_the_gates(world):
    reg, exp = world
    rid = rp.build(reg, exp.id, actor="alice")["report_id"]
    text = rp.to_markdown(rp.get(reg, rid)["content"])
    assert text.startswith("# Report on") and "## Limitations and unresolved items" in text and "## Gates" in text and "Gate A:" in text and "Gate B:" in text and "Gate C:" in text
    assert "statistical_diagnostics: not computed" in text and "implementation defects" in text
    synthetic = copy.deepcopy(rp.get(reg, rid)["snapshot"])
    synthetic["experiment"]["source"] = "synthetic market"
    assert "SIMULATED DATA" in rp.to_markdown(rp.render(synthetic)).split("\n")[2], "a simulated source is labelled beside the first figures"
    assert pd.Timestamp(rp.get(reg, rid)["snapshot"]["captured_at"]) is not pd.NaT
