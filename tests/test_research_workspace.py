"""The research workspace's library: the context, the store and its migrations, the comparison and the regime lab.

What is pinned here is what would mislead a researcher if it broke: a context that is refused when it is wrong, a saved result that is never edited to fit a new context, a stored record that survives a
schema change, a ranking that is withheld when the results are not comparable, and a regime label on day *t* that does not change when later days are added."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.research import ContextError, ResearchStore, StoreError, compatibility, validate_context
from src.research import compare as cmp
from src.research import extract
from src.research import regime_lab as rl
from src.research.store import SCHEMA, check_artifact


def _art(title, *, kind="backtest", evidence="historical_simulation", data_kind="platform", seed=0, n=700, start="2016-01-04", assets=("SPY", "TLT"), version="v1", drift=0.0004, **ctx):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    eq = (1.0 + pd.Series(rng.normal(drift, 0.01, n), index=idx)).cumprod()
    return {"kind": kind, "title": title, "evidence": evidence, "data_kind": data_kind, "metrics": {"sharpe": 1.0},
            "context": {"assets": list(assets), "start": str(idx[0].date()), "end": str(idx[-1].date()), "dataset_version": version, "frequency": "daily", **ctx},
            "series": {"dates": [str(d.date()) for d in idx], "equity": [round(float(v), 6) for v in eq]}, "refs": {"job": "abc"}}


@pytest.fixture()
def store(tmp_path):
    return ResearchStore(tmp_path / "research")


# ------------------------------------------------------------------------------------------------------ the context
def test_a_context_is_checked_field_by_field_and_says_which_field_is_wrong():
    ok = validate_context({"assets": "spy, qqq  SPY", "start": "2015-01-01", "end": "2020-01-01", "risk_free": 0.02, "costs": {"cost_bps": 5}, "options": {"rate": 0.03, "vol_source": "model"}})
    assert ok["assets"] == ["SPY", "QQQ"] and ok["costs"] == {"cost_bps": 5.0} and ok["options"]["vol_source"] == "model"
    for bad, word in (({"assets": ["not a ticker!"]}, "ticker"), ({"start": "2020-01-01", "end": "2019-01-01"}, "before"), ({"risk_free": 0.9}, "risk-free"), ({"frequency": "weekly"}, "daily"),
                      ({"colour": "red"}, "unknown context field"), ({"start": "yesterday"}, "YYYY-MM-DD"), ({"costs": {"cost_bps": -1}}, "basis points"), ({"asset_class": "tulips"}, "asset class"),
                      ({"options": {"vol_source": "guess"}}, "vol_source"), ({"selected_run": "../etc"}, "id")):
        with pytest.raises(ContextError, match=word):
            validate_context(bad)
    with pytest.raises(ContextError, match="data source"):
        validate_context({"data_source": "nope"}, sources={"yahoo"})
    with pytest.raises(ContextError, match="at most"):
        validate_context({"assets": [f"A{i}" for i in range(41)]})


def test_a_context_that_is_not_json_safe_or_is_huge_is_refused():
    with pytest.raises(ContextError, match="cannot be saved"):
        validate_context({"strategy": {"x": float("nan")}})
    with pytest.raises(ContextError, match="too large"):
        validate_context({"ml": {"blob": "x" * 5000}})


def test_the_compatibility_check_names_what_changed_and_separates_what_alters_the_numbers_from_what_does_not():
    snap = {"assets": ["SPY", "TLT"], "start": "2016-01-04", "end": "2018-12-31", "data_source": "yahoo", "dataset_version": "v1", "costs": {"cost_bps": 5.0}, "risk_free": 0.02, "benchmark": "SPY"}
    assert compatibility(dict(snap), snap) == []
    now = {**snap, "assets": ["SPY", "QQQ"], "end": "2020-12-31", "dataset_version": "v2", "costs": {"cost_bps": 10.0}, "benchmark": "QQQ", "risk_free": 0.03}
    got = {i["field"]: i for i in compatibility(now, snap)}
    assert set(got) == {"assets", "dates", "dataset_version", "costs", "risk_free", "benchmark"}
    assert got["assets"]["severity"] == "mismatch" and "QQQ" in got["assets"]["message"] and "TLT" in got["assets"]["message"]
    assert got["dates"]["severity"] == "mismatch", "asking for days the result does not cover cannot be satisfied by the result"
    assert got["benchmark"]["severity"] == "note", "a benchmark chosen for the comparison does not change the numbers"
    shorter = compatibility({**snap, "start": "2017-01-02"}, snap)
    assert [i["severity"] for i in shorter] == ["note"], "a shorter window is inside the result: a note, not a mismatch"


# ------------------------------------------------------------------------------------------------------ the store
def test_a_session_round_trips_and_a_saved_result_is_not_changed_by_a_new_context(store):
    s = store.create_session("Alpha", {"assets": ["SPY"], "start": "2016-01-04"})
    art = store.add_artifact(s["id"], _art("A"))
    before = json.dumps(store.get_artifact(art["id"]), sort_keys=True)
    store.update_session(s["id"], context={"assets": ["QQQ"], "start": "2010-01-01", "costs": {"cost_bps": 50}})
    assert json.dumps(store.get_artifact(art["id"]), sort_keys=True) == before, "changing the context never edits a result that was already made"
    ctx = store.get_session(s["id"])["context"]
    assert ctx["assets"] == ["QQQ"] and ctx["costs"] == {"cost_bps": 50.0}
    issues = compatibility(ctx, store.get_artifact(art["id"])["context"])
    assert any(i["field"] == "assets" and i["severity"] == "mismatch" for i in issues)


def test_a_field_sent_as_none_is_removed_from_the_context(store):
    s = store.create_session("x", {"assets": ["SPY"], "benchmark": "SPY", "risk_free": 0.02})
    got = store.update_session(s["id"], context={"benchmark": None, "risk_free": None})
    assert "benchmark" not in got["context"] and "risk_free" not in got["context"] and got["context"]["assets"] == ["SPY"]


def test_every_saved_result_must_say_what_kind_of_evidence_and_data_it_is(store):
    s = store.create_session("x")
    bad = _art("A")
    with pytest.raises(StoreError, match="evidence"):
        store.add_artifact(s["id"], {**bad, "evidence": "great"})
    with pytest.raises(StoreError, match="kind of data"):
        store.add_artifact(s["id"], {**bad, "data_kind": None})
    with pytest.raises(StoreError, match="unknown kind"):
        store.add_artifact(s["id"], {**bad, "kind": "magic"})
    mismatched = {**bad, "series": {"dates": ["2020-01-01", "2020-01-02"], "equity": [1.0]}}
    with pytest.raises(StoreError, match="values for"):
        store.add_artifact(s["id"], mismatched)
    unordered = {**bad, "series": {"dates": ["2020-01-02", "2020-01-01"], "equity": [1.0, 1.0]}}
    with pytest.raises(StoreError, match="increasing"):
        store.add_artifact(s["id"], unordered)


def test_nan_in_a_series_or_metric_is_stored_as_null_and_the_file_is_valid_json(store, tmp_path):
    s = store.create_session("x")
    a = _art("A")
    a["metrics"] = {"sharpe": float("nan"), "cagr": 0.1}
    a["series"]["equity"][3] = float("nan")
    art = store.add_artifact(s["id"], a)
    text = (tmp_path / "research" / "artifacts" / f"{art['id']}.json").read_text()
    back = json.loads(text, parse_constant=lambda c: pytest.fail("NaN in the file"))
    assert back["metrics"]["sharpe"] is None and back["series"]["equity"][3] is None


def test_ids_cannot_reach_outside_the_store(store):
    for bad in ("../../etc/passwd", "x" * 12, "", "..", "0123456789ab/../x"):
        with pytest.raises(StoreError):
            store.get_session(bad)
        with pytest.raises(StoreError):
            store.get_artifact(bad)


def test_deleting_a_result_takes_it_out_of_its_session_groups_and_selection(store):
    s = store.create_session("x")
    a, b = store.add_artifact(s["id"], _art("A")), store.add_artifact(s["id"], _art("B", seed=1))
    store.update_session(s["id"], context={"selected_run": a["id"]})
    g = store.save_group(s["id"], "pair", [a["id"], b["id"]])
    store.delete_artifact(a["id"])
    rec = store.get_session(s["id"])
    assert rec["artifacts"] == [b["id"]] and rec["context"]["selected_run"] == "" and next(x for x in rec["groups"] if x["id"] == g["id"])["artifacts"] == [b["id"]]
    with pytest.raises(StoreError):
        store.get_artifact(a["id"])
    with pytest.raises(StoreError, match="not in this session"):
        store.save_group(s["id"], "bad", ["0123456789ab"])


def test_a_duplicate_keeps_context_and_notes_but_not_results(store):
    s = store.create_session("x", {"assets": ["SPY"]})
    store.add_note(s["id"], "why this universe")
    store.add_artifact(s["id"], _art("A"))
    d = store.duplicate_session(s["id"])
    assert d["context"]["assets"] == ["SPY"] and [n["text"] for n in d["notes"]] == ["why this universe"] and d["artifacts"] == []


def test_an_older_session_is_migrated_when_read_and_the_old_file_is_kept(store, tmp_path):
    folder = tmp_path / "research" / "sessions"
    folder.mkdir(parents=True)
    old = {"id": "aaaaaaaaaaaa", "name": "Old workspace", "tickers": ["spy", "tlt"], "start": "2012-01-02", "source": "yahoo", "runs": [], "notes": ["first idea"]}
    (folder / "aaaaaaaaaaaa.json").write_text(json.dumps(old))                       # schema 0: no "schema" key at all
    rec = store.get_session("aaaaaaaaaaaa")
    assert rec["schema"] == SCHEMA and rec["context"] == {"assets": ["SPY", "TLT"], "start": "2012-01-02", "data_source": "yahoo"} and rec["notes"][0]["text"] == "first idea"
    assert json.loads((folder / "aaaaaaaaaaaa.json").read_text())["schema"] == SCHEMA, "the migrated record is written back"
    assert json.loads((folder / "aaaaaaaaaaaa.v0.bak").read_text()) == old, "and the original is kept beside it"
    assert [s["id"] for s in store.list_sessions()] == ["aaaaaaaaaaaa"]


def test_a_session_from_a_newer_version_is_listed_but_never_opened_or_rewritten(store, tmp_path):
    folder = tmp_path / "research" / "sessions"
    folder.mkdir(parents=True)
    future = {"schema": SCHEMA + 1, "id": "bbbbbbbbbbbb", "name": "From the future", "context": {}}
    path = folder / "bbbbbbbbbbbb.json"
    path.write_text(json.dumps(future))
    listed = store.list_sessions()
    assert listed[0]["readable"] is False and "newer version" in listed[0]["reason"]
    with pytest.raises(StoreError, match="newer version"):
        store.update_session("bbbbbbbbbbbb", name="x")
    assert json.loads(path.read_text()) == future


def test_a_crash_between_write_and_rename_leaves_the_old_record(store, monkeypatch):
    s = store.create_session("keep me")
    import src.research.store as mod

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(mod.os, "replace", boom)
    with pytest.raises(OSError):
        store.update_session(s["id"], name="changed")
    monkeypatch.undo()
    assert store.get_session(s["id"])["name"] == "keep me"
    assert not list((store.root / "sessions").glob(".*.part")), "no half-written file is left behind"


def test_limits_are_enforced(store, monkeypatch):
    import src.research.store as mod

    monkeypatch.setattr(mod, "MAX_SESSIONS", 2)
    store.create_session("1"); store.create_session("2")
    with pytest.raises(StoreError, match="at most 2 sessions"):
        store.create_session("3")
    s = store.list_sessions()[0]["id"]
    monkeypatch.setattr(mod, "MAX_POINTS", 50)
    with pytest.raises(StoreError, match="too long"):
        store.add_artifact(s, _art("long", n=100))


def test_check_artifact_keeps_only_what_it_understands():
    art = check_artifact({**_art("A"), "id": "0123456789ab", "session_id": "ba9876543210", "metrics": {"sharpe": 1.2, "bad": [1, 2], "flag": True}, "refs": {"job": "x", "none": None}})
    assert art["metrics"] == {"sharpe": 1.2} and art["refs"] == {"job": "x"}


# ------------------------------------------------------------------------------------------------------ extraction from real run outputs
def test_the_backtest_artifact_comes_from_the_job_result_and_is_labelled_a_historical_simulation():
    dates = [str(d.date()) for d in pd.bdate_range("2018-01-01", periods=60)]
    result = {"name": "Dual momentum", "universe": {"tickers": ["SPY", "TLT"], "first": "2017-01-03", "last": "2020-01-01", "days": 800, "source": "platform dataset + Yahoo Finance", "provider": "yahoo"},
              "dates": dates, "equity": list(np.linspace(1, 1.1, 60)), "benchmarks": {"equal_weight": list(np.linspace(1, 1.05, 60))}, "metrics": {"sharpe": 0.8, "cagr": 0.06, "start": "x", "max_drawdown": -0.1},
              "validation": {"deflated_sharpe_probability": 0.4, "n_trials": 7}, "spec": {"models": [{"name": "dual_momentum"}]}, "capital": 100000.0, "flags": [{"title": "Few trades"}]}
    art = extract.backtest_artifact(result, job_id="abc123def456", code_version="deadbee")
    assert art["evidence"] == "historical_simulation" and art["data_kind"] == "mixed" and art["metrics"]["n_trials"] == 7.0 and "start" not in art["metrics"]
    assert art["refs"]["code_version"] == "deadbee" and art["refs"]["job"] == "abc123def456" and art["context"]["dataset_version"] == extract.fingerprint(["SPY", "TLT"], "2017-01-03", "2020-01-01", 800, "platform dataset + Yahoo Finance")
    assert any("not out-of-sample" in n for n in art["notes"])
    with pytest.raises(StoreError, match="did not produce a backtest"):
        extract.backtest_artifact({"hello": 1})


def test_the_ml_artifact_is_walk_forward_and_carries_turnover_costs_and_segments():
    idx = pd.bdate_range("2019-01-01", periods=120)
    ret = pd.DataFrame({"net": np.full(120, 0.0005), "gross": 0.0006, "costs": 0.0001, "turnover": 0.2, "benchmark": 0.0003, "segment": ["validation"] * 60 + ["test"] * 60}, index=idx)
    report = {"id": "exp_1", "name": "ridge", "config": {"model": {"kind": "ridge"}, "dataset": {"target": "fwd_5d"}}, "config_hash": "h1", "dataset_key": "dk", "dataset": {"n_assets": 8, "horizon": 5, "target": "fwd_5d"},
              "provenance": {"source": "simulated", "label": "synthetic", "notes": []}, "metrics": {"oos": {"sharpe": 1.1, "cagr": 0.1, "note": "x"}}, "trials": {"dataset_trials": 12}}
    art = extract.ml_artifact(report, ret, code_version="c0ffee")
    assert art["evidence"] == "walk_forward" and art["data_kind"] == "simulated" and art["metrics"] == {"sharpe": 1.1, "cagr": 0.1, "n_trials": 12.0}
    assert art["series"]["extra"]["segment"][0] == "validation" and art["series"]["extra"]["turnover"][0] == 0.2 and art["refs"]["ml_experiment"] == "exp_1"
    with pytest.raises(StoreError, match="no saved returns"):
        extract.ml_artifact(report, pd.DataFrame())


# ------------------------------------------------------------------------------------------------------ the comparison
def test_results_are_measured_over_the_days_they_share_and_the_window_is_stated(store):
    s = store.create_session("x")
    a = store.add_artifact(s["id"], _art("A", seed=1))
    b = store.add_artifact(s["id"], _art("B", seed=2, start="2016-06-01"))
    out = cmp.compare([a, b])
    assert out["window"]["start"] == b["series"]["dates"][0] and out["window"]["end"] == a["series"]["dates"][-1]
    assert out["metrics"][a["id"]]["n_days"] == out["metrics"][b["id"]]["n_days"] and len(out["series"]["dates"]) == out["metrics"][a["id"]]["n_days"]
    assert {"window_cut"} <= {i["code"] for i in out["issues"]}
    assert out["rankable"] and sorted(out["ranking"]) == sorted([a["id"], b["id"]]) and "not a forecast" in out["reason"]
    # the same arithmetic as the Backtest tab's tiles
    r = cmp._returns(a).loc[out["window"]["start"]:out["window"]["end"]]
    eq = (1 + r).cumprod()
    assert out["metrics"][a["id"]]["cagr"] == pytest.approx(eq.iloc[-1] ** (252 / len(r)) - 1)
    assert out["metrics"][a["id"]]["sharpe"] == pytest.approx(np.sqrt(252) * r.mean() / r.std(ddof=1))


def test_different_evidence_data_assets_or_versions_stop_the_ranking_and_say_why(store):
    s = store.create_session("x")
    base = store.add_artifact(s["id"], _art("base", seed=1))
    cases = {"mixed_evidence": _art("wf", kind="ml_experiment", evidence="walk_forward", seed=2), "mixed_data": _art("sim", data_kind="simulated", seed=2), "different_assets": _art("other", assets=("QQQ",), seed=2),
             "different_dataset": _art("v2", version="v2", seed=2), "different_frequency": _art("f", seed=2, frequency="daily") | {}}
    for code, art in cases.items():
        if code == "different_frequency":
            continue                                                                  # only daily is accepted by the context, so there is nothing to differ
        other = store.add_artifact(s["id"], art)
        out = cmp.compare([base, other])
        assert code in {i["code"] for i in out["issues"]}, code
        assert out["rankable"] is False and out["ranking"] is None and out["reason"].startswith("not ranked"), code
        assert out["metrics"], "the numbers are still shown, only the ranking is withheld"


def test_ml_experiments_with_different_targets_are_not_ranked(store):
    s = store.create_session("x")
    a = store.add_artifact(s["id"], {**_art("a", kind="ml_experiment", evidence="walk_forward", seed=1), "config": {"target": "fwd_5d", "horizon": 5}})
    b = store.add_artifact(s["id"], {**_art("b", kind="ml_experiment", evidence="walk_forward", seed=2), "config": {"target": "fwd_21d", "horizon": 21}})
    out = cmp.compare([a, b])
    assert "different_target" in {i["code"] for i in out["issues"]} and out["ranking"] is None


def test_too_short_an_overlap_is_refused_not_computed(store):
    s = store.create_session("x")
    a = store.add_artifact(s["id"], _art("a", n=100, start="2016-01-04"))
    b = store.add_artifact(s["id"], _art("b", n=100, start="2016-05-02"))
    out = cmp.compare([a, b])
    assert out["metrics"] == {} and out["ranking"] is None and "short_overlap" in {i["code"] for i in out["issues"]}


def test_the_user_window_and_the_reference_are_honoured_and_the_series_share_one_axis(store):
    s = store.create_session("x")
    a, b = store.add_artifact(s["id"], _art("a", seed=1)), store.add_artifact(s["id"], _art("b", seed=2))
    out = cmp.compare([a, b], benchmark=b["id"], start="2016-09-01", end="2017-09-01")
    assert out["window"] == {"start": "2016-09-01", "end": "2017-09-01"}
    n = len(out["series"]["dates"])
    assert all(len(v) == n for k in ("equity", "drawdown", "rolling_sharpe", "rolling_vol") for v in out["series"][k].values())
    assert out["series"]["equity"][a["id"]][0] == pytest.approx(1.0, abs=0.05) and set(out["versus_reference"]) == {a["id"]} and out["reference"] == b["id"]
    assert out["correlation"]["ids"] == [a["id"], b["id"]] and out["correlation"]["values"][0][0] == 1.0
    json.dumps(out, allow_nan=False)                                                  # what the page receives is valid JSON


def test_the_report_names_references_window_differences_and_caveats(store):
    s = store.create_session("Study", {"assets": ["SPY"]})
    a = store.add_artifact(s["id"], _art("A", seed=1))
    b = store.add_artifact(s["id"], _art("B", seed=2, data_kind="simulated"))
    out = cmp.compare([a, b])
    md = cmp.to_markdown(out, session=s, generated="2026-01-01 00:00 UTC")
    assert "# Research comparison" in md and "job `abc`" in md and "Window shared by all results" in md and "not ranked" in md.lower() and "does not describe future returns" in md and s["id"] in md
    csv_text = cmp.to_csv(out)
    assert csv_text.splitlines()[0].startswith("window,") and "evidence," in csv_text and "blocks ranking" in csv_text.replace("_", " ")


# ------------------------------------------------------------------------------------------------------ the regime lab
def _market(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    state = np.zeros(n, int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() > (0.02 if state[t - 1] == 0 else 0.05) else 1 - state[t - 1]
    r = np.where(state == 0, rng.normal(0.0006, 0.007, n), rng.normal(-0.0004, 0.022, n))
    return pd.Series(r, index=pd.bdate_range("2012-01-02", periods=n)), state


def test_the_filtered_regime_on_a_day_does_not_change_when_later_days_are_added():
    """The look-ahead test: the causal estimate of every day up to *T* is the same whether the data ends at *T* or later."""
    proxy, _ = _market(1800)
    short = rl.filtered_hmm(proxy.iloc[:1500], refit_every=126)["probabilities"]
    long = rl.filtered_hmm(proxy, refit_every=126)["probabilities"].loc[short.index]
    both = short.dropna().index
    assert len(both) > 300
    # the model is refitted on the past at each refit date; days before the first refit that falls after day 1500 must be identical
    last_refit_in_short = both[-1]
    cutoff = [d for d in both if d <= last_refit_in_short][-1]
    assert np.allclose(short.loc[:cutoff].to_numpy(), long.loc[:cutoff].to_numpy(), atol=1e-6, equal_nan=True)


def test_the_retrospective_estimate_does_change_when_later_days_are_added_which_is_why_it_is_never_traded_on():
    proxy, _ = _market(1800)
    a = rl.smoothed_hmm(proxy.iloc[:1400])
    b = rl.smoothed_hmm(proxy).loc[a.index]
    assert not np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-3), "smoothing uses the future, so appending data moves old labels"


def test_the_lab_recovers_a_planted_two_state_market_and_reports_both_views():
    proxy, state = _market(2200)
    filt = rl.filtered_hmm(proxy)
    out = rl.analyse(proxy, filt, rl.smoothed_hmm(proxy))
    assert out["names"] == ["LowVol", "HighVol"] and out["evidence"] == "walk_forward" and "filtered" in out["label_basis"]
    lab = np.array(out["timeline"]["label"])
    truth = state[len(state) - len(lab):]
    assert (lab == truth).mean() > 0.85, "the causal estimate finds the planted states most of the time"
    dur = {d["state"]: d for d in out["durations"]}
    assert dur["LowVol"]["mean_days"] > dur["HighVol"]["mean_days"] * 0.8 and dur["LowVol"]["share"] > 0.5
    assert out["stability"]["agreement_with_retrospective"] > 0.8 and "retrospective" in out and out["retrospective"]["label"].startswith("retrospective")
    nd = {r["state"]: r for r in out["proxy_by_regime"]["next_day"]["rows"]}
    assert nd["HighVol"]["ann_vol"] > 2 * nd["LowVol"]["ann_vol"], "the high-volatility regime is measured on the days after the label"
    tm = out["transitions"]["probabilities"]
    assert tm[0][0] > 0.9 and tm[1][1] > 0.8
    json.dumps(out, allow_nan=False)


def test_performance_by_regime_pairs_the_label_with_the_next_days_return_and_withholds_thin_regimes():
    idx = pd.bdate_range("2020-01-01", periods=400)
    lab = pd.Series(["A"] * 200 + ["B"] * 190 + ["C"] * 10, index=idx, dtype=object)
    r = pd.Series(0.001, index=idx)
    r.iloc[200] = 0.5                                                                  # the first day of B: the return of the day the label first applies at the close of A's last day
    out = rl.performance_by_regime(r, lab, lag=1, min_days=60)
    rows = {x["state"]: x for x in out["rows"]}
    assert rows["C"]["eligibility"] == "insufficient" and "fewer than" in rows["C"]["reason"] and "sharpe" not in rows["C"]
    assert rows["A"]["eligibility"] == "applied"
    # with lag 1 the huge return on day 200 is attributed to the label of day 199 (state A); with lag 0 to state B
    same = {x["state"]: x for x in rl.performance_by_regime(r, lab, lag=0, min_days=60)["rows"]}
    assert rows["A"]["ann_mean"] > same["A"]["ann_mean"] and rows["B"]["ann_mean"] < same["B"]["ann_mean"]
    assert "next-day" in out["basis"] and "descriptive" in rl.performance_by_regime(r, lab, lag=0)["basis"]
    with pytest.raises(ValueError, match="lag"):
        rl.performance_by_regime(r, lab, lag=2)


def test_the_lab_refuses_too_little_history_and_odd_state_counts():
    proxy, _ = _market(500)
    with pytest.raises(ValueError, match="at least"):
        rl.filtered_hmm(proxy)
    with pytest.raises(ValueError, match="2 or 3"):
        rl.filtered_hmm(_market(1200)[0], n_states=5)
