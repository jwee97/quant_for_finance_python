"""Release 8, P3-10: champion/challenger evidence with a predeclared schedule of looks.

The spending is checked against SciPy's normal distribution; a pair is registered before the observations it is judged on and cannot be backfilled; both forecasts are committed together; a look is spent once,
needs the settled pairs the schedule names, and tests the paired difference at its own level; a replay never counts."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.ml import champion as ch, prospective as pr, registry as rg

REG_AT = pd.Timestamp("2030-01-01 12:00").timestamp()
ASSETS = [f"A{i}" for i in range(8)]


@pytest.fixture
def reg(tmp_path):
    return rg.Registry(tmp_path / "ml")


def pair(reg, looks=(20, 40, 60), **kw):
    kw = dict(champion="champ-v1", challenger="chall-v1", universe_version="u1", horizon=1, looks=list(looks), alpha=0.05, rule="obrien_fleming", assumptions={"cost_bps": 7, "max_gross": 1.0},
              data_cutoff="2030-01-01", actor="alice", now=REG_AT, **kw)
    return ch.register(reg, **kw)


def day(i):
    return pd.Timestamp("2030-01-02") + pd.tseries.offsets.BDay(i)


def commit_and_settle(reg, p, i, edge, rng, settle=True):
    """Decision ``i``: the challenger's forecast is the outcome plus noise scaled by ``edge`` (0 is pure noise), the champion's is noise. Committed at 09:00 on the decision day, settled afterwards."""
    d = day(i)
    outcome = rng.normal(0, 0.02, len(ASSETS))
    champ = rng.normal(0, 0.02, len(ASSETS))
    chall = edge * outcome + rng.normal(0, 0.02, len(ASSETS)) if edge >= 0 else -edge * rng.normal(0, 1, len(ASSETS)) * 0 + rng.normal(0, 0.02, len(ASSETS))
    if edge < 0:                                                                                     # the champion is the informed one
        champ, chall = -edge * outcome + rng.normal(0, 0.02, len(ASSETS)), chall
    at = d.normalize().timestamp() + 9 * 3600
    out = ch.commit_pair(reg, p["pair_id"], decision_ts=d, forecasts={"champion": dict(zip(ASSETS, champ)), "challenger": dict(zip(ASSETS, chall))}, data_cutoff="2030-01-01",
                         fit_cutoffs={"champion": "2029-12-20", "challenger": "2029-12-28"}, model_created={"champion": "2029-12-21", "challenger": "2029-12-29"}, actor="alice", now=at)
    if settle:
        for role in ("champion", "challenger"):
            pr.settle(reg, out[role]["commitment_id"], dict(zip(ASSETS, outcome)), actor="settler", now=at + 3 * 86400)
    return out


# ------------------------------------------------------------------------------------------------------------------ alpha spending
def test_the_spending_functions_match_their_definitions_and_the_levels_add_up_to_the_total():
    z = stats.norm.ppf(0.975)
    obf = ch.spending("obrien_fleming", [0.25, 0.5, 1.0], 0.05)
    assert obf == pytest.approx([2 * stats.norm.sf(z / 0.5), 2 * stats.norm.sf(z / np.sqrt(0.5)), 0.05]) and obf[0] < 0.001
    assert ch.spending("pocock", [0.5, 1.0], 0.05) == pytest.approx([0.05 * np.log(1 + (np.e - 1) * 0.5), 0.05]) and ch.spending("equal", [0.25, 1.0], 0.05) == pytest.approx([0.0125, 0.05])
    for rule in ch.SPENDING:
        lv = ch.levels(rule, [20, 40, 60], 0.05)
        assert sum(lv) == pytest.approx(0.05) and all(x > 0 for x in lv), rule
    first = {r: ch.levels(r, [20, 40, 60], 0.05)[0] for r in ch.SPENDING}
    assert first["obrien_fleming"] < 0.001 < first["equal"] < first["pocock"] and first["equal"] == pytest.approx(0.05 / 3), "O'Brien-Fleming spends least early, equal shares a third, and Pocock the most"
    assert ch.levels("obrien_fleming", [50], 0.05) == pytest.approx([0.05])
    with pytest.raises(ch.ChampionError, match="spending rule"):
        ch.spending("vibes", [1.0], 0.05)


# ------------------------------------------------------------------------------------------------------------------ registration
def test_a_pair_is_frozen_before_the_observations_and_the_same_declaration_is_the_same_pair(reg):
    p = pair(reg)
    assert p["status"] == "registered" and p["pair_id"].startswith("cc-") and p["levels"] == pytest.approx(ch.levels("obrien_fleming", [20, 40, 60], 0.05)) and p["looks_remaining"] == 3
    assert p["registered_epoch"] == REG_AT and p["registered_at"] == "2030-01-01 12:00:00" and len(p["assumptions_hash"]) == 16 and "fixed before" in p["note"]
    again = pair(reg)
    assert again["status"] == "unchanged" and again["pair_id"] == p["pair_id"]
    assert pair(reg, looks=(20, 40))["pair_id"] != p["pair_id"], "a different schedule is a different pair"
    for bad, msg in (({"challenger": "champ-v1"}, "different strategy versions"), ({"looks": [40, 20]}, "strictly increasing"), ({"looks": [5, 20]}, "at least 10"), ({"looks": [20, 20]}, "strictly increasing"),
                     ({"alpha": 0.9}, "alpha in"), ({"rule": "vibes"}, "alpha in"), ({"actor": ""}, "records who made it")):
        kw = dict(champion="champ-v1", challenger="chall-v1", universe_version="u1", horizon=1, looks=[20, 40], alpha=0.05, rule="obrien_fleming", actor="alice", now=REG_AT)
        kw.update(bad)
        with pytest.raises(ch.ChampionError, match=msg):
            ch.register(reg, **kw)
    with pytest.raises(ch.ChampionError, match="unknown champion/challenger pair"):
        ch.get(reg, "cc-nope")


# ------------------------------------------------------------------------------------------------------------------ committing both forecasts
def test_both_forecasts_are_committed_together_under_one_decision_one_cutoff_and_nothing_is_backfilled(reg):
    p = pair(reg)
    rng = np.random.default_rng(0)
    out = commit_and_settle(reg, p, 0, 0.0, rng, settle=False)
    assert out["champion"]["status"] == "committed" == out["challenger"]["status"] and out["champion"]["classification"] == "prospective" == out["challenger"]["classification"]
    assert out["champion"]["commit_key"].startswith("champ-v1|") and out["challenger"]["commit_key"].startswith("chall-v1|") and out["champion"]["commit_key"].split("|")[1:] == out["challenger"]["commit_key"].split("|")[1:]
    f = {"champion": {"A0": 1.0}, "challenger": {"A0": 1.0}}
    kw = dict(forecasts=f, data_cutoff="2030-01-01", fit_cutoffs={"champion": "2029-12-20", "challenger": "2029-12-28"}, model_created={"champion": "2029-12-21", "challenger": "2029-12-29"}, actor="alice")
    with pytest.raises(ch.ChampionError, match="predates the registration"):
        ch.commit_pair(reg, p["pair_id"], decision_ts="2029-12-31", now=REG_AT + 1, **kw)
    with pytest.raises(ch.ChampionError, match="data cutoff .* differs"):
        ch.commit_pair(reg, p["pair_id"], decision_ts=day(3), now=day(3).timestamp(), **{**kw, "data_cutoff": "2030-01-02"})
    with pytest.raises(ch.ChampionError, match="exactly 'champion' and 'challenger'"):
        ch.commit_pair(reg, p["pair_id"], decision_ts=day(3), now=day(3).timestamp(), **{**kw, "forecasts": {"champion": {"A0": 1.0}}})
    late = ch.commit_pair(reg, p["pair_id"], decision_ts=day(4), now=day(6).timestamp(), **kw)
    assert late["champion"]["classification"] == "replay" and late["challenger"]["classification"] == "replay", "a commitment written after the outcome could be seen is a replay"


# ------------------------------------------------------------------------------------------------------------------ looks
def test_a_look_needs_the_settled_pairs_it_names_is_spent_once_and_a_real_edge_is_found_at_its_own_level(reg):
    p = pair(reg)
    rng = np.random.default_rng(1)
    with pytest.raises(ch.ChampionError, match="needs 20 settled decision pairs; there are 0"):
        ch.look(reg, p["pair_id"], actor="alice")
    for i in range(25):
        commit_and_settle(reg, p, i, 1.5, rng)
    first = ch.look(reg, p["pair_id"], actor="alice")
    assert first["look"] == 1 and first["pairs_used"] == 20 and first["verdict"] == "challenger_better" and first["level"] == pytest.approx(p["levels"][0]) and first["promotes"] is False
    assert first["challenger_ic"] > first["champion_ic"] and first["difference"]["p_value"] < first["level"] and not first["final"] and "nothing here changes a stage" in first["note"]
    with pytest.raises(ch.ChampionError, match="look 2 of 3 needs 40"):
        ch.look(reg, p["pair_id"], actor="alice")
    for i in range(25, 65):
        commit_and_settle(reg, p, i, 1.5, rng)
    second, third = ch.look(reg, p["pair_id"], actor="alice"), ch.look(reg, p["pair_id"], actor="alice")
    assert (second["pairs_used"], third["pairs_used"], third["final"]) == (40, 60, True)
    with pytest.raises(ch.ChampionError, match="no unspent look"):
        ch.look(reg, p["pair_id"], actor="alice")
    s = ch.summary(reg, p["pair_id"])
    assert s["looks_remaining"] == 0 and s["verdict"] == "challenger_better" and s["settled_pairs"] == 65 and s["next_look_needs"] is None and s["counts"]["champion"]["prospective"]["settled"] == 65


def test_with_no_difference_nothing_is_claimed_and_an_informed_champion_is_recognised(reg):
    rng = np.random.default_rng(2)
    p = pair(reg, looks=(20, 40))
    for i in range(40):
        commit_and_settle(reg, p, i, 0.0, rng)
    verdicts = [ch.look(reg, p["pair_id"], actor="a")["verdict"] for _ in range(2)]
    assert verdicts == ["no_difference", "no_difference"] and ch.summary(reg, p["pair_id"])["verdict"] == "no_difference"
    q = ch.register(reg, champion="champ-v2", challenger="chall-v2", universe_version="u1", horizon=1, looks=[20], alpha=0.05, rule="pocock", data_cutoff="2030-01-01", actor="alice", now=REG_AT)
    for i in range(20):
        commit_and_settle(reg, q, i, -1.5, rng)
    assert ch.look(reg, q["pair_id"], actor="a")["verdict"] == "champion_better"


def test_replays_and_unsettled_commitments_do_not_count_toward_a_look(reg):
    rng = np.random.default_rng(3)
    p = pair(reg, looks=(20,))
    for i in range(30):
        commit_and_settle(reg, p, i, 1.5, rng, settle=(i < 15))
    assert len(ch.paired_series(reg, ch.get(reg, p["pair_id"]))) == 15
    with pytest.raises(ch.ChampionError, match="needs 20 settled decision pairs; there are 15"):
        ch.look(reg, p["pair_id"], actor="a")


# ------------------------------------------------------------------------------------------------------------------ shared assumptions against the versions' recorded settings
def test_a_pair_of_registered_versions_must_share_execution_costs_and_risk_limits_and_unregistered_ones_are_said_to_be_declared_only(tmp_path):
    from src.ml import experiments as xp, governance as gv
    from tests.ml_helpers import make_bundle

    bundle = make_bundle(n_assets=6, n=1300, seed=3, alpha=1.0)
    reg = rg.Registry(tmp_path / "ml")
    base = xp.run_experiment({"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 126}, "model": {"kind": "ridge"}, "signal": {"n": 2}},
                             bundle=bundle, registry=reg, explain=False, robustness=False)
    other_signal = xp.rebacktest(reg, base.id, {"signal": {"n": 3}}, bundle=bundle)
    dearer = xp.rebacktest(reg, base.id, {"backtest": {"cost_bps": 40.0}}, bundle=bundle)
    realistic = xp.rebacktest(reg, base.id, {"execution": {"mode": "realistic_cash_equity", "capital": 1e6}}, bundle=bundle)
    v = {k: gv.register_version(reg, e.id, actor="jo")["version_id"] for k, e in (("base", base), ("signal", other_signal), ("dearer", dearer), ("realistic", realistic))}
    limited = gv.register_version(reg, base.id, actor="jo", risk_limits=gv.RiskLimits({"max_gross": 1.5}))["version_id"]
    kw = dict(universe_version="u1", horizon=5, looks=[20, 40], actor="alice", now=REG_AT)
    ok = ch.register(reg, champion=v["base"], challenger=v["signal"], **kw)
    assert ok["status"] == "registered" and ok["assumptions_check"] == {"checked": True, "compared": ["execution", "backtest", "risk_limits"], "differing": [], "under_test": ["signal"]}, "only the signal differs"
    assert ch.get(reg, ok["pair_id"])["assumptions_check"]["checked"] is True, "the check is part of the record"
    for other, word in (("dearer", "backtest"), ("realistic", "execution")):
        with pytest.raises(ch.ChampionError, match=f"do not share their recorded {word} settings"):
            ch.register(reg, champion=v["base"], challenger=v[other], **kw)
    with pytest.raises(ch.ChampionError, match="risk_limits"):
        ch.register(reg, champion=v["base"], challenger=limited, **kw)
    assert ch.assumption_check(reg, v["base"], v["base"])["differing"] == [], "a version shares its own settings"
    loose = ch.register(reg, champion="external-v1", challenger=v["signal"], **kw)
    chk = loose["assumptions_check"]
    assert chk["checked"] is False and "champion is not a registered strategy version" in chk["reason"] and "declared and frozen" in chk["reason"], "nothing was compared, and the record says so"
    both = ch.assumption_check(reg, "external-v1", "external-v2")
    assert both["checked"] is False and "champion and the challenger are not registered strategy versions" in both["reason"]
