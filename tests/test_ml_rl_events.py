"""P0-04: what the environment overrode in an agent's book is not only returned from a call, it is saved with the experiment and recorded in the registry as an audit row that cannot be edited.

The events are the environment's own sentences (clipped actions, forced liquidations, trimmed limits, shortened trades, a cut-back of exposure); each has a stable kind, the table of them is saved beside the
predictions, a summary of it (counts by kind and segment, the dates, a hash of the table) is an append-only analysis record, and a reloaded agent replays to the same events."""

import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import experiments as xe, models as md, registry as rg
from src.utils.optional import unavailable_reason
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
KINDS = {"clipped_action", "no_eligible_asset", "forced_liquidation", "limit_trim", "mandatory_trades_over_budget", "turnover_shortened", "leverage_cutback"}
AGENT = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 500, "refit_every": 400}, "signal": {"method": "target_weights"}, "name": "events",
         "model": {"kind": "ppo", "params": {"total_timesteps": 192, "n_steps": 64, "n_epochs": 1, "batch_size": 32, "hidden": 8, "episode_days": 60, "max_turnover": 0.05}}}


def _need():
    for pkg in ("gymnasium", "stable_baselines3"):
        if unavailable_reason(pkg):
            pytest.skip(f"{pkg} is not available here: {unavailable_reason(pkg)}")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    _need()
    reg = rg.Registry(tmp_path_factory.mktemp("rl_events") / "ml")
    bundle = make_bundle(8, 1300, seed=9, alpha=1.0)
    exp = xe.run_experiment(AGENT, bundle=bundle, registry=reg, explain=False, robustness=False)
    return reg, exp, bundle


def test_every_event_the_environment_reports_has_a_stable_kind_and_the_kinds_cover_what_it_can_say():
    _need()
    from src.ml.rl.env import EVENT_KINDS, PortfolioEnv, event_kind

    assert {k for _, k in EVENT_KINDS} == KINDS and event_kind("something new") == "other"
    assert event_kind("a position or exposure limit trimmed 0.5000 of equity after the turnover budget") == "limit_trim"
    seen = []
    mask = np.ones((60, 3), bool)
    mask[10:, 1] = False
    env = PortfolioEnv(np.zeros((60, 3, 1), np.float32), np.random.default_rng(1).normal(0, 0.01, (60, 3)), mask, episode_days=40, lag=1, cost_bps=0.0, gross=1.0, max_weight=0.5, max_turnover=0.2, reward="return")
    env.reset_at(0)
    for t in range(14):
        seen += env.step(np.array([7.0, -1.0, 0.5], dtype=np.float32) if t == 0 else np.array([1.0, -1.0, 0.0], dtype=np.float32))[4]["events"]
    none = PortfolioEnv(np.zeros((60, 3, 1), np.float32), np.zeros((60, 3)), np.zeros((60, 3), bool), episode_days=10, lag=1, reward="return")
    none.reset_at(0)
    seen += none.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))[4]["events"]
    rets = np.zeros((12, 2))
    rets[1:, 0], rets[1:, 1] = -0.15, 0.15
    lev = PortfolioEnv(np.zeros((12, 2, 1), np.float32), rets, None, episode_days=10, lag=1, cost_bps=0.0, gross=1.0, max_weight=0.5, max_leverage=1.05, reward="return")
    lev.reset_at(0)
    for _ in range(6):
        seen += lev.step(np.array([1.0, -1.0], dtype=np.float32))[4]["events"]
    kinds = {event_kind(e) for e in seen}
    assert "other" not in kinds, [e for e in seen if event_kind(e) == "other"]
    assert {"clipped_action", "no_eligible_asset", "forced_liquidation", "leverage_cutback", "turnover_shortened"} <= kinds


def test_the_events_of_an_agent_are_saved_with_the_experiment_and_each_has_a_known_kind(world):
    reg, exp, _ = world
    table = pd.read_parquet(reg.root / "experiments" / exp.id / "extra_events.parquet")
    assert list(table.columns) == ["n_events", "kinds", "detail"] and len(table) > 10, "a turnover budget of 5% a day is binding on most days"
    kinds = [k for cell in table["kinds"] for k in cell.split(",") if k]
    assert kinds and set(kinds) <= KINDS and len(kinds) == int(table["n_events"].sum())
    assert "turnover_shortened" in set(kinds) and all(isinstance(d, str) and d for d in table["detail"])
    preds = pd.read_parquet(reg.root / "experiments" / exp.id / "predictions.parquet")
    assert table.index.min() >= preds.index.get_level_values(0).min(), "the days are those the policy was replayed over"


def test_a_registry_row_summarises_them_cannot_be_edited_and_matches_the_saved_table(world):
    reg, exp, _ = world
    from src.ml import diagnostics as dx

    rec = dx.latest(reg, exp.id, "rl_events")
    assert rec is not None
    c = rec["content"] if "content" in rec else rec
    table = pd.read_parquet(reg.root / "experiments" / exp.id / "extra_events.parquet")
    kinds = [k for cell in table["kinds"] for k in cell.split(",") if k]
    assert c["events"] == len(kinds) and c["days_with_events"] == len(table) and c["model"] == "ppo"
    assert c["by_kind"] == {k: kinds.count(k) for k in sorted(set(kinds))} and c["first"] == str(table.index.min().date()) and c["last"] == str(table.index.max().date())
    assert sum(sum(v.values()) for v in c["by_segment"].values()) == len(kinds), "every event belongs to a segment"
    assert c["constraints"]["max_turnover"] == pytest.approx(0.05) and len(c["table_hash"]) == 32
    with reg.transaction() as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE analysis_records SET content='{}' WHERE kind='rl_events'")


def test_a_reloaded_agent_replays_to_the_same_events_and_keeps_none_in_its_file():
    _need()
    from src.ml import config as mc, datasets as dsm

    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 300}}, defaults=mc.DEFAULTS)
    ds = dsm.build_dataset(make_bundle(7, 800, seed=71, alpha=1.0), cfg["dataset"], "events")
    pos = ds.positions()
    rows = np.flatnonzero((pos >= 80) & (pos < 520) & ds.Y[ds.target].notna().to_numpy())
    model = md.make_model("ppo", AGENT["model"]["params"], "regression", 0).bind(ds)
    model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos[rows])
    test = ds.X.iloc[np.flatnonzero((pos >= 540) & (pos < 620))]
    first = model.predict_extras(test)["events"]
    assert len(first) > 10 and set(k for cell in first["kinds"] for k in cell.split(",")) <= KINDS
    blob = model.to_bytes()
    assert "_events" not in md.Model.from_bytes(blob).__dict__, "a replay's events are never stored with the model"
    again = md.Model.from_bytes(blob).bind(ds)
    pd.testing.assert_frame_equal(again.predict_extras(test)["events"], first)
    half = test.iloc[: len(test) // 2]
    pd.testing.assert_frame_equal(again.predict_extras(half)["events"], first[first.index <= half.index.get_level_values(0).max()])      # asking for fewer days gives the same events for those days


def test_a_rebacktest_has_the_parents_events_and_records_none_of_its_own(world):
    reg, exp, bundle = world
    before = len(_records(reg, "rl_events"))
    child = xe.rebacktest(reg, exp.id, {"signal": {"method": "target_weights"}, "backtest": {"cost_bps": 3.0}}, bundle=bundle)
    assert before == 1 and len(_records(reg, "rl_events")) == 1 and _records(reg, "rl_events")[0][0] == exp.id
    assert (reg.root / "experiments" / child.id / "extra_events.parquet").exists(), "the table travels with the predictions"


def _records(reg, kind):
    with reg._conn() as c:                                                                           # noqa: SLF001
        return c.execute("SELECT subject_id FROM analysis_records WHERE kind=?", (kind,)).fetchall()
