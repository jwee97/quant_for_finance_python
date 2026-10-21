"""Automated alpha discovery against known truth: the planted alphas are found, a placebo world with shuffled returns yields nothing, every idea tried is counted and remembered, a stopped run resumes
exactly, the library moves an alpha between statuses only along allowed paths, and the monitor notices when an alpha stops working."""

from __future__ import annotations

import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import discovery as D
from src.fundamentals import expr as X
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.evaluate import holding_returns
from src.fundamentals.library import build_signal, settings_hash
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import FundamentalsStore


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("discovery"))
    svc.load_simulated(n_companies=100, start_year=2009, end_year=2018, seed=13, planted={"delta_roic": 0.011, "fcf_yield": 0.0068, "asset_growth": 0.0045})
    with svc.store.db.transaction() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return svc, ResearchContext(svc.store, UniverseSpec())


def fresh(base, tmp_path, name="copy") -> D.Discovery:
    """A discovery engine with an empty library over a copy of the loaded database, so that one test's graveyard is not another's."""
    svc, ctx = base
    target = tmp_path / f"{name}.db"
    shutil.copy(svc.store.db.path, target)
    return D.Discovery(FundamentalsStore(target), ctx)


def shuffled(ctx: ResearchContext, seed: int = 0) -> ResearchContext:
    """The same data with every date's returns dealt out to other companies at random: any relation between a formula and what happened next is destroyed, whatever the formula."""
    rng = np.random.default_rng(seed)
    null = copy.copy(ctx)
    null._cache = dict(ctx._cache)
    fwd, hold = ctx.forward("1m"), holding_returns(ctx)
    new_fwd, new_hold = fwd.copy(), hold.copy()
    for t in range(len(fwd)):
        perm = rng.permutation(fwd.shape[1])
        new_fwd.iloc[t] = fwd.iloc[t].to_numpy()[perm]
        new_hold.iloc[t] = hold.iloc[t].to_numpy()[perm]
    null._cache[("forward", 21)] = new_fwd
    null._cache[("holding",)] = new_hold
    return null


SMALL = dict(n_candidates=60, seed=3)


# ------------------------------------------------------------------------------------------------------------------------ the pieces
def test_the_dates_are_cut_into_three_blocks_with_a_gap_as_wide_as_the_horizon(base):
    ctx = base[1]
    spec = D.DiscoverySpec(horizon="3m")
    b = D.splits(ctx, spec)
    t, v, o = b["train"], b["validation"], b["oos"]
    assert t[1] < v[0] and v[1] < o[0] and b["embargo"] == 3 and sum(b["dates"].values()) + 2 * b["embargo"] <= len(ctx.dates)
    gap_train_val = ((ctx.dates > pd.Timestamp(t[1])) & (ctx.dates < pd.Timestamp(v[0]))).sum()
    gap_val_oos = ((ctx.dates > pd.Timestamp(v[1])) & (ctx.dates < pd.Timestamp(o[0]))).sum()
    assert gap_train_val >= b["embargo"] and gap_val_oos >= b["embargo"] and b["dates"]["train"] > b["dates"]["oos"] > b["dates"]["validation"] > 12
    with pytest.raises(FundamentalsError, match="too little history"):
        D.splits(ctx, D.DiscoverySpec(min_assets=10_000))


def test_benjamini_hochberg_q_values_match_a_reference_and_never_fall_below_the_p_value():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    p = np.random.default_rng(0).uniform(0, 1, 60) ** 3
    ours = D.bh_qvalues(p)
    ref = sm.multipletests(p, method="fdr_bh")[1]
    assert np.allclose(ours, ref) and (ours >= p - 1e-12).all() and ours.max() <= 1.0 and D.bh_qvalues(np.array([])).size == 0


def test_the_cache_of_sub_formulas_forgets_the_oldest_when_it_is_full():
    cache = D.BoundedCache(max_bytes=3 * 800 + 100)
    frame = lambda: (pd.DataFrame(np.zeros((10, 10))), "ratio")                   # noqa: E731  (800 bytes)
    for key in "abcde":
        cache[key] = frame()
    assert len(cache) == 3 and "a" not in cache and "e" in cache and cache.bytes == 3 * 800
    _ = cache["c"]                                                                  # a use keeps it
    cache["f"] = frame()
    assert "c" in cache and "d" not in cache


def test_the_fitness_rewards_evidence_and_unlikeness_and_penalises_turnover_and_complexity():
    good = dict(ic=0.05, val_ic=0.05, positive_share=0.8, sharpe=1.5, val_sharpe=1.5, monotonicity=1.0, turnover=0.1, nodes=3)
    base_score, parts = D.fitness(good, 1.0)
    assert base_score > D.fitness(dict(good, turnover=0.6), 1.0)[0] > D.fitness(dict(good, turnover=0.6, nodes=15), 1.0)[0]
    assert D.fitness(good, 0.1)[0] < base_score and D.fitness(dict(good, ic=0.0, val_ic=0.0), 1.0)[0] < base_score
    assert all(0.0 <= v <= 1.0 for v in parts.values()) and D.fitness(good, 1.0, {"ic": 0.0})[0] < base_score


# ------------------------------------------------------------------------------------------------------------------------ the stages
def test_each_formula_meets_the_stage_it_deserves(base):
    ctx = base[1]
    spec = D.DiscoverySpec()
    screen = D.Screen(ctx, spec, D.splits(ctx, spec), D.BoundedCache())
    yield_ = screen.run(X.parse("div(fcf,market_cap)"))
    flipped = screen.run(X.parse("neg(div(fcf,market_cap))"))
    assert yield_["status"] == "survivor" and yield_["stage"] == 6 and yield_["direction"] == 1.0
    assert flipped["status"] == "survivor" and flipped["direction"] == -1.0 and flipped["metrics"]["t_train"] == pytest.approx(-yield_["metrics"]["t_train"])
    fm, ym = flipped["metrics"], yield_["metrics"]
    assert fm["ic_raw"] < 0 < fm["ic"] and fm["ic"] == pytest.approx(ym["ic"]) and fm["icir"] == pytest.approx(ym["icir"])      # the figures kept are in the direction chosen, like the validation and locked ones
    assert fm["ic"] == pytest.approx(-fm["ic_raw"]) and ym["ic"] == pytest.approx(ym["ic_raw"])
    for noise in ("current_ratio", "debt_to_equity"):
        r = screen.run(X.parse(noise))
        assert r["status"] == "rejected" and r["stage"] >= 1, noise
    thin = screen.run(X.parse("rd_intensity"))
    assert thin["stage"] == 0 and "investable companies have a value" in thin["reason"]
    broken = screen.run(X.parse("rank(vibes)"))
    assert broken["stage"] == 0 and "cannot be computed" in broken["reason"]
    m = yield_["metrics"]
    assert m["positive_share"] > 0.7 and m["monotonicity"] >= 0.5 and m["sharpe"] > 0.15 and m["val_t"] > 1.0 and m["turnover"] < 0.7
    lock = screen.locked_test(X.parse("div(fcf,market_cap)"), 1.0)
    assert lock["passed"] and lock["oos_t"] > 1.0 and lock["oos_dates"] > 12 and "returns_net" in lock


# ------------------------------------------------------------------------------------------------------------------------ a run
def test_a_run_counts_everything_remembers_the_failures_and_keeps_only_candidates(base, tmp_path):
    disc = fresh(base, tmp_path)
    run = disc.create(D.DiscoverySpec(**SMALL))
    out = disc.run(run)
    counts = out["counts"]
    assert out["status"] == "finished" and out["progress"] == {"done": 60, "total": 60}
    passed = [row["passed"] for row in out["funnel"]]
    assert passed[0] >= passed[1] >= passed[2] >= passed[3] >= passed[4] >= passed[5] >= passed[6] and counts["tested"] >= passed[1] and counts["n_tested"] == counts["tested"]
    lib = disc.library
    tally = lib.counts()
    assert tally["library"].get("Candidate", 0) == counts["candidates"] >= 3 and set(tally["library"]) == {"Candidate"}
    rejected = [c for c in disc.candidates(run, "rejected", 10_000)]
    assert tally["graveyard"] >= len(rejected) - counts.get("failed_locked_test", 0) and tally["tried"] == tally["graveyard"] + counts["candidates"]
    top = lib.list("Candidate")[0]
    for key in ("oos_ic", "oos_t", "oos_sharpe", "q_value", "deflated_sharpe", "fitness", "n_tested", "max_correlation", "nodes", "direction"):
        assert key in top["metrics"], key
    assert top["status_reason"].startswith("passed every screen") and top["metrics"]["n_tested"] == counts["n_tested"] and top["oos_ic"] is not None
    texts = {a["text"] for a in lib.list("Candidate")}
    assert texts & {"div(fcf,market_cap)", "div(net_income,market_cap)", "div(revenue,market_cap)"}                    # the planted and the built-in value effects are found
    assert out["leaderboard"] and "Nothing here is validated" in out["note"]


def test_the_same_question_is_never_paid_for_twice(base, tmp_path):
    disc = fresh(base, tmp_path)
    spec = D.DiscoverySpec(**SMALL)
    first = disc.run(disc.create(spec))
    second = disc.run(disc.create(spec))
    # (the formulas bred from survivors depend on the run's own history, so a few new ones appear; everything that was tried before is skipped)
    assert first["counts"]["tested"] > 40 and second["counts"]["tested"] < first["counts"]["tested"] / 3 and second["counts"]["skipped_seen"] + second["counts"]["duplicate_idea"] >= 35
    assert disc.library.counts()["library"]["Candidate"] >= first["counts"]["candidates"] and second["counts"]["candidates"] == disc.library.counts()["library"]["Candidate"] - first["counts"]["candidates"]
    ehash = X.expression_hash(X.parse("div(fcf,market_cap)"))
    blocks = D.splits(disc.ctx, spec)
    digest = settings_hash({**spec.conditions(), "blocks": {k: blocks[k] for k in ("train", "validation", "oos")}})
    assert disc.library.seen(ehash, digest)["where"] == "library"
    assert disc.library.seen("0" * 16, digest) is None


def test_a_stopped_run_continues_exactly_where_it_was(base, tmp_path):
    spec = D.DiscoverySpec(**SMALL)
    whole = fresh(base, tmp_path, "whole")
    full = whole.run(whole.create(spec))
    parts = fresh(base, tmp_path, "parts")
    run = parts.create(spec)
    seen = {"n": 0}

    def stop():
        seen["n"] += 1
        return seen["n"] > 25

    halfway = parts.run(run, stop=stop)
    assert halfway["status"] == "stopped" and halfway["progress"]["done"] == 25
    finished = parts.run(run)
    assert finished["status"] == "finished"
    state = lambda d, r: sorted((c["expression_hash"], c["status"], c["stage_reached"]) for c in d.candidates(r, limit=10_000))      # noqa: E731
    assert state(parts, run) == state(whole, whole.runs()[0]["run_id"])
    assert finished["counts"]["candidates"] == full["counts"]["candidates"] and finished["counts"]["tested"] == full["counts"]["tested"]


def test_in_a_world_where_nothing_predicts_anything_nothing_is_found(base, tmp_path):
    survivors = 0
    for shuffle_seed in (1, 3):                                                                           # seed 3 let a lucky formula through every screen until the q-value gate was added
        disc = fresh(base, tmp_path, f"placebo{shuffle_seed}")
        disc.ctx = shuffled(base[1], seed=shuffle_seed)
        out = disc.run(disc.create(D.DiscoverySpec(n_candidates=200, seed=5 if shuffle_seed == 1 else 6, evolve=0.0)))
        assert out["counts"]["tested"] > 60 and out["counts"]["candidates"] == 0 and disc.library.counts()["library"] == {}
        survivors += out["counts"]["survivors"]
        late = [c for c in disc.candidates(out["run_id"], "survivor")]
        assert all(c["metrics"].get("q_value", 1.0) > 0.25 or c["metrics"].get("q_value") is None for c in late)      # whatever got through the screens did not survive the correction for the number tried
    assert survivors <= 5                                                                                   # the screens stacked end to end let almost no accident through
    real = fresh(base, tmp_path, "real")
    truth = real.run(real.create(D.DiscoverySpec(n_candidates=200, seed=5, evolve=0.0)))
    assert truth["counts"]["candidates"] >= 5 and truth["counts"]["survivors"] > 2 * max(survivors, 1)


def test_the_out_of_sample_dates_are_counted_every_time_they_are_looked_at(base, tmp_path):
    disc = fresh(base, tmp_path)
    for formula in ("div(fcf,market_cap)", "div(net_income,market_cap)"):
        out = disc.run(disc.create(D.DiscoverySpec(n_candidates=1, seeds=[formula], seed=1)))
        assert out["counts"]["candidates"] == 1
    assert out["counts"]["oos_looks"] == 2
    assert [a["metrics"]["oos_looks"] for a in sorted(disc.library.list(), key=lambda a: a["created_at"])] == [1, 2]            # the second candidate says its locked dates had already been looked at once


# ------------------------------------------------------------------------------------------------------------------------ the library
def test_an_alpha_moves_between_statuses_only_along_allowed_paths_and_every_move_is_logged(base, tmp_path):
    disc = fresh(base, tmp_path)
    disc.run(disc.create(D.DiscoverySpec(**SMALL)))
    lib = disc.library
    alpha = lib.list("Candidate")[0]
    aid = alpha["alpha_id"]
    with pytest.raises(FundamentalsError, match="cannot go straight to Production"):
        lib.set_status(aid, "Production", "too eager")
    with pytest.raises(FundamentalsError, match="needs a reason"):
        lib.set_status(aid, "Validated", "  ")
    validated = lib.validate(aid)
    assert validated["status"] == "Validated" and "correction" in validated["status_reason"]
    assert lib.set_status(aid, "Production", "approved for the book")["status"] == "Production"
    assert lib.set_status(aid, "Retired", "no longer needed")["status"] == "Retired"
    with pytest.raises(FundamentalsError, match="cannot go straight to Production"):
        lib.set_status(aid, "Production", "back from the dead")
    log = lib.status_log(aid)
    assert [(r["from_status"], r["to_status"]) for r in log] == [(None, "Candidate"), ("Candidate", "Validated"), ("Validated", "Production"), ("Production", "Retired")]
    with pytest.raises(FundamentalsError, match="no alpha called"):
        lib.get("A-nothing")
    with pytest.raises(FundamentalsError, match="status must be one of"):
        lib.set_status(aid, "Great", "x")


def test_validation_demands_evidence_and_a_person_can_override_it_on_the_record(base, tmp_path):
    disc = fresh(base, tmp_path)
    disc.run(disc.create(D.DiscoverySpec(**SMALL)))
    lib = disc.library
    weak = lib.list("Candidate")[-1]
    with lib.store.db.transaction() as conn:
        conn.execute("UPDATE alpha_library SET q_value = 0.9, deflated_sharpe = 0.2 WHERE alpha_id = ?", (weak["alpha_id"],))
    with pytest.raises(FundamentalsError, match="not validated.*ideas tried"):
        lib.validate(weak["alpha_id"])
    forced = lib.validate(weak["alpha_id"], force_reason="the sector desk wants it tracked")
    assert forced["status"] == "Validated" and "approved by a person despite" in forced["status_reason"] and lib.status_log(weak["alpha_id"])[-1]["forced"] == 1


def test_the_graveyard_is_never_overwritten_and_finds_the_same_idea_in_another_spelling(base, tmp_path):
    disc = fresh(base, tmp_path)
    lib = disc.library
    settings = {"universe": {}, "freq": "M", "horizon": "1m"}
    e = X.parse("div(current_ratio,2)")
    assert lib.add_graveyard(e, settings, "first reason", "fast_ic", {"ic": 0.001}, "R1") is True
    assert lib.add_graveyard(e, settings, "second reason", "fast_ic") is False
    row = lib.graveyard()[0]
    assert row["reason_rejected"] == "first reason" and row["text"] == "div(current_ratio,2)" and lib.counts()["graveyard_by_stage"] == {"fast_ic": 1}
    assert lib.add_graveyard(e, {**settings, "horizon": "3m"}, "other conditions", "fast_ic") is True                    # a different question is a different test
    twins = lib.same_idea(X.parse("neg(rank(div(current_ratio,2)))"))
    assert len(twins) == 2 and {t["where"] for t in twins} == {"graveyard"}
    assert lib.same_idea(X.parse("neg(rank(div(current_ratio,2)))"), settings_hash(settings))[0]["reason"] == "first reason"
    assert lib.same_idea(X.parse("roic")) == []


# ------------------------------------------------------------------------------------------------------------------------ the monitor
def _tail(ctx: ResearchContext, mode: str, rows: int, signal: pd.DataFrame | None = None) -> ResearchContext:
    """The context with the returns of its last dates replaced: dealt out at random (``shuffle``: the alpha has stopped working) or made to follow the signal upside down, with noise (``reverse``: it
    works the other way round, as strongly whatever the draw of the world, which negating the real returns would not give)."""
    out = copy.copy(ctx)
    out._cache = dict(ctx._cache)
    fwd = ctx.forward("1m").copy()
    rng = np.random.default_rng(0)
    for t in range(len(fwd) - rows - 1, len(fwd)):
        row = fwd.iloc[t].to_numpy()
        if mode == "reverse":
            s = signal.iloc[t].to_numpy(float)
            z = np.nan_to_num((s - np.nanmean(s)) / (np.nanstd(s) or 1.0))
            fwd.iloc[t] = np.where(np.isfinite(row), -0.04 * z + 0.04 * rng.standard_normal(len(row)), np.nan)
        else:
            fwd.iloc[t] = row[rng.permutation(len(row))]
    out._cache[("forward", 21)] = fwd
    return out


def test_the_monitor_judges_an_alpha_used_upside_down_by_how_well_it_works_that_way_round(base, tmp_path):
    disc = fresh(base, tmp_path)
    disc.run(disc.create(D.DiscoverySpec(n_candidates=2, seed=1, seeds=["neg(div(fcf,market_cap))"], finalists=3)))
    lib, ctx = disc.library, base[1]
    alpha = next(a for a in lib.list("Candidate") if a["text"] == "neg(div(fcf,market_cap))")
    assert alpha["metrics"]["direction"] == -1.0 and alpha["ic"] > 0 > alpha["metrics"]["ic_raw"] and alpha["oos_ic"] > 0
    lib.validate(alpha["alpha_id"], "the test's own approval")
    healthy = lib.monitor(alpha["alpha_id"], ctx)
    assert healthy["monitor"] in ("Healthy", "Weakening") and healthy["status"] == "Validated"               # the flip was once applied twice: this alpha was called Broken and moved to Decaying (a year of a noisy IC is Healthy or Weakening by the draw)
    assert healthy["detail"]["recent_ic"] > 0 and healthy["detail"]["original_ic"] == pytest.approx(alpha["ic"]) and healthy["detail"]["recent_t"] > 1.0
    longer = lib.monitor(alpha["alpha_id"], ctx, recent=24)
    assert longer["monitor"] == "Healthy" and longer["detail"]["recent_t"] > 1.8 and longer["status"] == "Validated"                    # two years of it leave no doubt
    signal = build_signal(ctx, X.loads(alpha["expression"]), alpha["settings"])
    broken = lib.monitor(alpha["alpha_id"], _tail(ctx, "reverse", 14, signal))
    assert broken["monitor"] == "Broken" and broken["detail"]["recent_ic"] < 0 and broken["status"] == "Decaying"


def test_the_monitor_calls_a_working_alpha_healthy_and_a_failing_one_decaying_or_broken(base, tmp_path):
    disc = fresh(base, tmp_path)
    spec = D.DiscoverySpec(**SMALL)
    disc.run(disc.create(spec))
    lib, ctx = disc.library, base[1]
    row = next(a for a in lib.list("Candidate") if a["text"] in ("div(fcf,market_cap)", "div(net_income,market_cap)", "div(revenue,market_cap)"))
    aid = row["alpha_id"]
    lib.validate(aid)
    healthy = lib.monitor(aid, ctx)
    assert healthy["monitor"] in ("Healthy", "Weakening") and healthy["status"] == "Validated" and healthy["detail"]["recent_ic"] > 0      # a working alpha is not called Decaying or Broken; a noisy year may only call for a watch
    decaying = lib.monitor(aid, _tail(ctx, "shuffle", 24))
    assert decaying["monitor"] in ("Decaying", "Broken") and decaying["status"] == "Decaying" and lib.get(aid)["status"] == "Decaying"
    assert "the monitor judged it" in lib.get(aid)["status_reason"]
    lib.set_status(aid, "Validated", "recovered after review")
    broken = lib.monitor(aid, _tail(ctx, "reverse", 14, build_signal(ctx, X.loads(row["expression"]), row["settings"])))
    assert broken["monitor"] == "Broken" and broken["detail"]["recent_t"] <= -2.0
    history = lib.monitor_history(aid)
    assert len(history) == 3 and history[0]["status"] == "Broken" and history[-1]["status"] in ("Healthy", "Weakening") and history[0]["detail"]["recent_t"] <= -2.0
    assert lib.get(aid)["status"] == "Decaying"                                                              # a Broken verdict moved the Validated alpha to Decaying
