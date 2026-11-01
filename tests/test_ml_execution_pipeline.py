"""Release 4 in the experiment pipeline: the realistic mode as a declared evaluation mode of a re-run (P1-04, P1-05, P1-11, P1-14 and the delivery conditions of spec 7.3).

A realistic re-run is a CHILD of the experiment it re-runs, with a new id, its own artifacts (orders, fills, journal) and a manifest of hashes; the parent is untouched and is read as the vectorized legacy mode.
"""

import hashlib
import json

import pandas as pd
import pytest

from src.ml import execution as ex, experiments as xe, registry as rg
from src.ml.config import MLConfigError, config_hash
from tests.ml_helpers import make_bundle

FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
OVER = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 500, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}}}


@pytest.fixture(scope="module")
def bundle():
    return make_bundle(10, 1500, seed=21, alpha=1.0, macro=False)


@pytest.fixture(scope="module")
def parent(bundle, tmp_path_factory):
    reg = rg.Registry(tmp_path_factory.mktemp("ml"))
    exp = xe.run_experiment(OVER, bundle=bundle, registry=reg, explain=False, final_model=False)
    return reg, exp


def test_a_new_vectorized_run_says_what_it_assumes_and_cannot_support(parent):
    reg, exp = parent
    block = exp.report["execution"]
    assert block["mode"] == "vectorized_legacy" and block["recorded"] is True and "capacity" in block["cannot_support"] and "order lifecycle and partial fills" in block["not_modelled"]
    assert block["assumptions"]["signal_lag_sessions"] == 1 and "linear" in block["assumptions"]["cost_model"]
    assert "execution" not in exp.cfg and exp.execution is None, "the legacy mode is the absence of the section: no configuration hash changed"


def test_a_realistic_rerun_is_a_child_with_its_own_id_artifacts_manifest_and_a_reconciled_ledger(parent, bundle):
    reg, base = parent
    child = xe.rebacktest(reg, base.id, {"execution": {"mode": "realistic_cash_equity", "capital": 2_000_000.0}}, bundle=bundle)
    assert child.id != base.id and reg.get_experiment(child.id)["parent_id"] == base.id and config_hash(child.cfg) != config_hash(base.cfg), "a new id, a recorded parent, a different configuration identity"
    block = child.report["execution"]
    assert block["mode"] == "realistic_cash_equity" and block["timeline"]["delay_sessions"] == 1 and "applied once" in block["timeline"]["delay_source"]
    assert block["reconciliation"]["reconciled"] and block["reconciliation"]["ledger_ok"], block["reconciliation"]
    g = block["gross_net"]
    assert g["gross_pnl"] - g["costs_total"] == pytest.approx(g["net_pnl"], abs=1e-3 * 2_000_000.0 * 1e-3), "gross less the itemised costs is net"
    assert set(block["costs"]) == {"fee", "spread", "slippage", "impact", "borrow", "financing"} and block["costs"]["fee"] > 0
    assert block["capacity"]["status"] == "measured_on_last_decision" and [c["capital"] for c in block["capacity"]["levels"]] == [1e5, 1e6, 1e7, 1e8, 1e9]
    shares = [c["fill_share"] for c in block["capacity"]["levels"]]
    assert all(a >= b - 1e-12 for a, b in zip(shares, shares[1:])), "more capital never fills a larger share"
    cmp = block["comparison_with_vectorized"]
    assert cmp["n_days"] > 100 and cmp["note"] and "common_subset" in block and block["common_subset"]["outside"]
    assert any("impact coefficient is an assumption" in x for x in block["limitations"]), "an impact coefficient that was not calibrated is said to be an assumption"
    sim = xe.rebacktest(reg, base.id, {"execution": {"capital": 1e6}}, save=False, market={"volume": bundle.volume, "status": "simulated"})
    assert any("the data is simulated" in x for x in sim.report["execution"]["limitations"]) and sim.report["execution"]["capability"]["data_status"] == "simulated"
    # the artifacts exist, the manifest is the one the report cites, and every hash is the hash of the file on disk
    d = reg.root / "experiments" / child.id
    manifest_text = (d / "execution_manifest.json").read_text()
    assert hashlib.sha256(manifest_text.encode()).hexdigest() == block["manifest"]["sha256"]
    manifest = json.loads(manifest_text)
    assert set(manifest["files"]) == {"orders", "fills", "journal", "equity"}
    for key, meta in manifest["files"].items():
        data = (d / meta["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == meta["sha256"] and len(pd.read_parquet(d / meta["file"])) == meta["rows"], key
    assert manifest["inputs"]["prices"]["sha256"] and manifest["inputs"]["weights"]["sha256"] and manifest["specification"]["capital"] == 2_000_000.0
    with reg._conn() as c:
        logged = {r[0] for r in c.execute("SELECT name FROM artifacts WHERE experiment_id=?", (child.id,))}
    assert logged >= {"execution_orders", "execution_fills", "execution_journal", "execution_manifest"}, "the files are in the registry's artifact list, not only on disk"
    # the parent is untouched
    assert reg.report(base.id)["execution"]["mode"] == "vectorized_legacy"


def test_the_realistic_rerun_charges_what_the_vectorized_engine_cannot_and_stays_a_look_at_the_same_study(parent, bundle):
    reg, base = parent
    cheap = xe.rebacktest(reg, base.id, {"execution": {"commission_bps": 0.0, "slippage_bps": 0.0, "impact": {"coefficient": 0.0}, "max_participation": None}}, bundle=bundle)
    dear = xe.rebacktest(reg, base.id, {"execution": {"commission_bps": 10.0, "slippage_bps": 10.0, "half_spread_bps": 5.0}}, bundle=bundle)
    assert dear.report["execution"]["gross_net"]["net_return"] < cheap.report["execution"]["gross_net"]["net_return"]
    assert dear.report["execution"]["costs"]["spread"] > cheap.report["execution"]["costs"]["spread"] == 0.0
    from src.ml import research_controls as rc

    lin = rc.lineage(reg, rc.study_of_experiment(reg, base.id))
    assert lin["by_kind"].get("portfolio_variation", 0) >= 2 and lin["raw_looks"] >= 3, "each realistic re-run is another counted look at the same outcomes"
    # a realistic run reads the parent's holdout exposure: it is never an untouched evaluation
    assert cheap.report["holdout"]["status"] != "untouched"


def test_settings_that_would_lag_twice_or_assume_what_is_missing_are_refused(parent, bundle):
    reg, base = parent
    with pytest.raises(MLConfigError, match="not a setting"):
        xe.rebacktest(reg, base.id, {"execution": {"delay": 2}}, save=False, bundle=bundle)
    with pytest.raises(MLConfigError, match="unknown execution settings"):
        xe.rebacktest(reg, base.id, {"execution": {"borrow": "free"}}, save=False, bundle=bundle)
    with pytest.raises(MLConfigError, match="capital must be positive"):
        xe.rebacktest(reg, base.id, {"execution": {"capital": 0}}, save=False, bundle=bundle)
    with pytest.raises(MLConfigError, match="next_close"):
        xe.rebacktest(reg, base.id, {"execution": {"convention": "same_close_auction"}}, save=False, bundle=bundle)
    from src.ml.config import experiment_config

    with pytest.raises(MLConfigError, match="lag"):
        experiment_config({"dataset": {"lag": 0}, "execution": {"capital": 1e6}})                          # a decision made from a close cannot trade at that close
    with pytest.raises(ValueError, match="at least one session"):
        ex.decision_weights(base.weights, base.dataset.returns, base.dataset.investable, base.cfg["backtest"], 0)
    with pytest.raises(xe.ExperimentError, match="not \\['model'\\]"):
        xe.rebacktest(reg, base.id, {"model": {"kind": "lasso"}, "execution": {"capital": 1e6}}, save=False, bundle=bundle)
    class NoData:
        def default_bundle(self):
            raise FileNotFoundError("no processed prices")

    with pytest.raises(xe.ExperimentError, match="needs the market data"):
        xe.rebacktest(reg, base.id, {"execution": {"capital": 1e6}}, save=False, bundle=None, config=None, builder=NoData())


def test_an_experiment_saved_before_the_modes_existed_is_read_as_vectorized_legacy(parent):
    reg, base = parent
    report = reg.report(base.id)
    old = {k: v for k, v in report.items() if k != "execution"}
    assert "execution" not in old
    block = ex.legacy_block(base.cfg, recorded=False)
    assert block["mode"] == "vectorized_legacy" and block["recorded"] is False and "saved before execution modes were recorded" in block["note"]


def test_point_in_time_screens_keep_an_illiquid_or_new_asset_out_and_say_how_many_asset_days_they_removed(parent, bundle):
    reg, base = parent
    screened = xe.rebacktest(reg, base.id, {"execution": {"min_adv_dollars": 1e9, "min_listing_days": 30}}, save=False, bundle=bundle)
    s = screened.report["execution"]["screens"]
    assert s["min_adv_dollars"] == 1e9 and s["asset_days"] > 0
    assert s["ineligible_asset_days"].get("listed fewer than 30 sessions", 0) > 0 and any(k.startswith("prior average dollar volume below") for k in s["ineligible_asset_days"])
    assert screened.report["execution"]["n_fills"] < reg_report_fills(reg, base, bundle), "an asset that fails the screen is never added to the book"
    off = xe.rebacktest(reg, base.id, {"execution": {"capital": 1e6}}, save=False, bundle=bundle)
    assert off.report["execution"]["screens"] is None, "no screen was asked for: none is applied and none is claimed"


def reg_report_fills(reg, base, bundle):
    return xe.rebacktest(reg, base.id, {"execution": {"capital": 1e6}}, save=False, bundle=bundle).report["execution"]["n_fills"]


def test_dated_membership_corporate_actions_and_borrow_records_can_be_supplied_and_change_the_claim(parent, bundle):
    from src.ml import realism as rz

    reg, base = parent
    assets = list(bundle.prices.columns)
    ends = bundle.prices.index[900]
    listings = [rz.Listing(f"S{i}", a, bundle.prices.index[0], ends if i == 0 else None, status="failed" if i == 0 else "active") for i, a in enumerate(assets)]
    member = rz.Membership(listings, coverage="complete")
    bk = rz.BorrowBook(pd.DataFrame(1e9, index=bundle.prices.index, columns=assets), pd.DataFrame(0.02, index=bundle.prices.index, columns=assets), strict=True)
    split = [rz.CorporateAction(assets[1], "split", bundle.prices.index[1000], {"ratio": 2.0})]
    market = {"volume": bundle.volume, "status": "vendor", "membership": member, "borrow": bk, "actions": split}
    child = xe.rebacktest(reg, base.id, {"execution": {"capital": 2_000_000.0}}, save=False, market=market)
    block, res = child.report["execution"], child.execution.result
    assert (res.shares[assets[0]].loc[ends + pd.Timedelta(days=10):] == 0.0).all(), "the failed security is out of the book at the first weekly decision after its end date (and its lag) and is never bought again"
    assert block["screens"]["membership"] is True and block["screens"]["ineligible_asset_days"]["not a member or not listed"] > 0
    assert block["capability"]["historical_universe"]["limitation"] is None and block["capability"]["borrow"] == "strict records" and block["capability"]["corporate_actions"]["types"] == ["split"]
    assert block["capability"]["blockers"] == [] and block["capability"]["data_status"] == "vendor" and block["capability"]["realistic_execution_claim"] in ("limited", "supported"), block["capability"]
    assert not any("survivors" in x for x in block["limitations"]) and not any("no borrow records" in x for x in block["limitations"])
    plain = xe.rebacktest(reg, base.id, {"execution": {"capital": 2_000_000.0}}, save=False, market={"volume": bundle.volume, "status": "vendor"})
    assert any("survivors" in x for x in plain.report["execution"]["limitations"]) and plain.report["execution"]["capability"]["blockers"], "with no membership and no borrow records the claim is limited and blocked"


def test_strict_validation_refuses_new_shorts_without_borrow_records_and_the_report_says_so(parent, bundle):
    reg, base = parent
    free = xe.rebacktest(reg, base.id, {"execution": {"capital": 2_000_000.0}}, save=False, bundle=bundle)
    strict = xe.rebacktest(reg, base.id, {"execution": {"capital": 2_000_000.0, "shorts": "refuse"}}, save=False, bundle=bundle)
    assert free.report["execution"]["short_positions"] is True and strict.report["execution"]["short_positions"] is False
    assert (strict.execution.result.shares >= 0).all().all() and any("shorts were refused under strict validation" in x for x in strict.report["execution"]["limitations"])
    assert strict.report["execution"]["capability"]["borrow"] == "none", "an empty book is not a borrow record"
    # equity before and after costs and the leverage carried: the numbers 7.3 asks for beside the gross and net series
    eq, lev = free.report["execution"]["equity"], free.report["execution"]["leverage"]
    assert eq["before_costs"] - eq["after_costs"] == pytest.approx(free.report["execution"]["gross_net"]["costs_total"], abs=1.0) and lev["max_gross"] >= lev["mean_gross"] > 0
