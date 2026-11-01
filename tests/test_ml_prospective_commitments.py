"""Release 5, P1-24: the prospective forecast ledger.

A forecast is prospective evidence only if it was committed before its outcome could be known AND the data and the fitted model it used already existed. The ledger is exercised directly: commitments are
immutable and idempotent, the creation time is the server's, replays are labelled and never counted, and settlements are appended in versions with a reason. The numbers asserted are hand-computed (rank
correlations of three-asset forecasts) or counts of rows; nothing is compared with the code's own output.
"""

import sqlite3
import threading

import pandas as pd
import pytest

from src.ml import prospective as pr, registry as rg

SV = "sv-test"
DEC = pd.Timestamp("2024-03-01")
BEFORE = pd.Timestamp("2024-03-01 20:00").timestamp()                  # evening of the decision day: before the first session whose price the label needs
AFTER = pd.Timestamp("2024-03-05 09:00").timestamp()                   # after that session has traded
F = {"A": 0.3, "B": -0.1, "C": 0.2, "D": 0.0}
W = {"A": 0.5, "B": -0.5}
ARTS = {"model": "m" * 8, "data": "d" * 8, "code": "c" * 8}


@pytest.fixture
def reg(tmp_path):
    return rg.Registry(tmp_path / "ml")


def commit(reg, *, now=BEFORE, **over):
    kw = dict(strategy_version=SV, decision_ts=DEC, universe_version="u1", horizon=5, forecasts=F, weights=W, data_cutoff=DEC, fit_cutoff=DEC - pd.Timedelta(days=30), artifact_hashes=ARTS,
              model_created=pd.Timestamp("2024-02-01"), label_start=DEC + pd.offsets.BDay(1), actor="tester", now=now)
    kw.update(over)
    return pr.commit(reg, **kw)


def audit(reg, key=None):
    with reg._conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM commitment_audit" + (" WHERE commit_key=?" if key else "") + " ORDER BY seq", (key,) if key else ())]


def test_a_commitment_is_written_once_and_an_identical_repeat_is_idempotent_and_audited(reg):
    first = commit(reg)
    assert first["status"] == "committed" and first["classification"] == "prospective" and first["commitment_id"].startswith("cm-")
    again = commit(reg)
    assert again["status"] == "unchanged" and again["commitment_id"] == first["commitment_id"] and again["content_hash"] == first["content_hash"]
    with reg._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM commitments").fetchone()[0] == 1
    events = [a["event"] for a in audit(reg)]
    assert events == ["committed", "duplicate_identical"], events


def test_different_content_under_the_same_key_is_rejected_audited_and_changes_nothing(reg):
    first = commit(reg)
    with pytest.raises(pr.CommitmentConflict, match="never changed"):
        commit(reg, forecasts={**F, "A": 0.31})
    with reg._conn() as c:
        row = c.execute("SELECT content_hash, forecasts FROM commitments").fetchone()
    assert row["content_hash"] == first["content_hash"] and '"A": 0.3' in row["forecasts"], "the stored commitment is untouched"
    rejected = [a for a in audit(reg) if a["event"] == "rejected_conflict"]
    assert len(rejected) == 1 and rejected[0]["content_hash"] != rejected[0]["existing_hash"] and rejected[0]["actor"] == "tester"
    # a different cutoff, weight or model hash is different content too; the platform's code hash is not content
    for over in ({"weights": {"A": 0.4, "B": -0.4}}, {"data_cutoff": DEC - pd.Timedelta(days=1)}, {"artifact_hashes": {**ARTS, "model": "other"}}):
        with pytest.raises(pr.CommitmentConflict):
            commit(reg, **over)
    assert commit(reg, artifact_hashes={**ARTS, "code": "a newer build"})["status"] == "unchanged"


def test_the_key_is_strategy_version_decision_time_universe_and_horizon(reg):
    base = commit(reg)
    others = [commit(reg, strategy_version="sv-other"), commit(reg, decision_ts=DEC + pd.Timedelta(days=1), data_cutoff=DEC + pd.Timedelta(days=1)), commit(reg, universe_version="u2"), commit(reg, horizon=10)]
    assert {o["status"] for o in others} == {"committed"} and len({base["commitment_id"], *[o["commitment_id"] for o in others]}) == 5
    assert pr.commit_key(SV, DEC, "u1", 5) == f"{SV}|2024-03-01T00:00:00|u1|5"


def test_a_commitment_cannot_be_updated_or_deleted_and_its_creation_time_is_the_servers(reg):
    cid = commit(reg)["commitment_id"]
    with reg._conn() as c:
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            c.execute("UPDATE commitments SET forecasts='{}' WHERE commitment_id=?", (cid,))
        with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
            c.execute("DELETE FROM commitments WHERE commitment_id=?", (cid,))
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            c.execute("DELETE FROM commitment_audit")
    import inspect

    assert "now" in inspect.signature(pr.commit).parameters and pr.commit.__doc__ and "tests only" in pr.commit.__doc__, "the clock override is documented as a test hook"
    live = pr.commit(reg, strategy_version="sv-live", decision_ts=DEC, universe_version="u1", horizon=5, forecasts=F, weights=W, data_cutoff=DEC, fit_cutoff=DEC, artifact_hashes=ARTS,
                     model_created=pd.Timestamp("2000-01-01"), actor="tester")
    stamp = pd.Timestamp(live["created_at"])
    assert abs((pd.Timestamp.utcnow().tz_localize(None) - stamp).total_seconds()) < 60, "without an override the creation time is the clock of the machine that writes the row"
    assert live["classification"] == "replay", "a decision dated in the past, committed now, was already partly observable"


@pytest.mark.parametrize("over,needle", [
    ({"now": AFTER}, "not before the label's first return day"),
    ({"data_cutoff": DEC + pd.Timedelta(days=2)}, "data-availability cutoff"),
    ({"data_cutoff": None}, "data-availability cutoff"),
    ({"fit_cutoff": DEC + pd.Timedelta(days=1)}, "fit cutoff"),
    ({"fit_cutoff": None}, "fit cutoff"),
    ({"model_created": pd.Timestamp("2024-03-02")}, "model artifact"),
    ({"model_created": None}, "model artifact"),
])
def test_creation_before_settlement_is_necessary_but_not_sufficient(reg, over, needle):
    res = commit(reg, **over)
    assert res["classification"] == "replay" and needle in res["why"], res["why"]


def test_a_commitment_with_every_cutoff_in_order_is_prospective(reg):
    res = commit(reg)
    assert res["classification"] == "prospective" and "already existed" in res["why"]
    # the cutoff equal to the decision is allowed: the close of the day is the information, and a model fitted at that close may decide it
    assert commit(reg, decision_ts=DEC + pd.Timedelta(days=1), data_cutoff=DEC + pd.Timedelta(days=1), fit_cutoff=DEC + pd.Timedelta(days=1), now=BEFORE + 86400)["classification"] == "prospective"


def test_replays_are_listed_and_never_counted_as_prospective_evidence(reg):
    pro = commit(reg)
    rep = commit(reg, decision_ts=DEC - pd.Timedelta(days=10), data_cutoff=DEC - pd.Timedelta(days=10), fit_cutoff=DEC - pd.Timedelta(days=40), label_start=DEC - pd.Timedelta(days=9), now=BEFORE)
    assert rep["classification"] == "replay"
    pr.settle(reg, pro["commitment_id"], {"A": 0.02, "B": -0.01, "C": 0.01, "D": 0.0}, actor="t")
    pr.settle(reg, rep["commitment_id"], {"A": 0.02, "B": -0.01, "C": 0.01, "D": 0.0}, actor="t")
    counts = pr.counts(reg, SV)
    assert counts["prospective"] == {"committed": 1, "settled": 1, "distinct_settled_days": 1}
    assert counts["replay"] == {"committed": 1, "settled": 1, "distinct_settled_days": 1} and "never count" in counts["note"]
    assert pr.evaluate(reg, SV)["n_settled"] == 1 and pr.evaluate(reg, SV, classification="replay")["n_settled"] == 1


def test_distinct_settled_days_count_days_and_not_forecasts(reg):
    a, b = commit(reg, horizon=1), commit(reg, horizon=5)
    for r in (a, b):
        pr.settle(reg, r["commitment_id"], {"A": 0.01, "B": 0.0, "C": -0.01}, actor="t")
    c = pr.counts(reg, SV)["prospective"]
    assert c["settled"] == 2 and c["distinct_settled_days"] == 1, "two forecasts for the same decision day are one day of evidence"


def test_a_settlement_is_appended_never_written_over_and_a_revision_needs_a_reason(reg):
    cid = commit(reg)["commitment_id"]
    out1 = {"A": 0.02, "B": -0.01, "C": 0.01, "D": 0.0}                    # forecasts A>C>D>B ; outcomes A>C>D>B: rank IC = 1
    first = pr.settle(reg, cid, out1, actor="t")
    assert first["status"] == "settled" and first["version"] == 1
    assert pr.settle(reg, cid, out1, actor="t")["status"] == "unchanged", "the same outcomes are not a new version"
    revised = {"A": 0.02, "B": 0.03, "C": 0.01, "D": 0.0}                  # the outcome of B was restated: forecasts A>C>D>B ; outcomes B>A>C>D
    with pytest.raises(pr.LedgerError, match="correction reason"):
        pr.settle(reg, cid, revised, actor="t")
    second = pr.settle(reg, cid, revised, reason="vendor restated the adjusted close of B", actor="t")
    assert second["version"] == 2 and second["previous_version"] == 1
    with reg._conn() as c:
        rows = [dict(r) for r in c.execute("SELECT version, correction_reason, outcomes FROM settlements ORDER BY version")]
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            c.execute("UPDATE settlements SET outcomes='{}' WHERE version=1")
    assert [r["version"] for r in rows] == [1, 2] and rows[0]["correction_reason"] is None and "restated" in rows[1]["correction_reason"] and '"B": -0.01' in rows[0]["outcomes"]
    ev = pr.evaluate(reg, SV)
    # hand ranks. forecasts (A,B,C,D) = (.3,-.1,.2,0) -> ranks (4,1,3,2). outcomes v1 (.02,-.01,.01,0) -> (4,1,3,2): IC = 1.
    # outcomes v2 (.02,.03,.01,0) -> ranks (3,4,2,1): spearman = 1 - 6*sum(d^2)/(n(n^2-1)) with d = (1,-3,1,1): 1 - 6*12/60 = -0.2
    assert ev["ic_first_observed"]["mean"] == pytest.approx(1.0, abs=1e-12) and ev["ic_latest_version"]["mean"] == pytest.approx(-0.2, abs=1e-12) and ev["n_revised"] == 1
    with pytest.raises(pr.LedgerError, match="unknown commitment"):
        pr.settle(reg, "cm-nope", out1, actor="t")


def test_evaluation_needs_enough_assets_and_says_so(reg):
    cid = commit(reg, forecasts={"A": 1.0, "B": 2.0}, weights=None)["commitment_id"]
    pr.settle(reg, cid, {"A": 0.1, "B": 0.2}, actor="t")
    ev = pr.evaluate(reg, SV)
    assert ev["n_settled"] == 1 and ev["ic_first_observed"] == {"mean": None, "n": 0}, "two assets give no rank correlation worth reporting: it is unavailable, not zero"


def test_concurrent_identical_commitments_write_exactly_one_row(reg):
    results, errors = [], []

    def work():
        try:
            results.append(commit(reg)["status"])
        except Exception as exc:                                                                  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert sorted(results).count("committed") == 1 and sorted(results).count("unchanged") == 7
    with reg._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM commitments").fetchone()[0] == 1


def test_a_commitment_needs_forecasts_and_a_person(reg):
    with pytest.raises(pr.LedgerError, match="needs forecasts"):
        commit(reg, forecasts={})
    with pytest.raises(pr.LedgerError, match="who made it"):
        commit(reg, actor="")
