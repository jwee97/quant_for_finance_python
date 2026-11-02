"""Release 6, P2-10: horizons of 1, 5, 10 and 20 sessions, tranche accounting and the comparison that does not rescale overlapping returns."""

import numpy as np
import pandas as pd
import pytest

from src.ml import comparison as cp, datasets as ds, horizons as hz, registry as rg, signals as sg, targets as tg
from tests.ml_helpers import make_bundle

BASE = {"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5", "horizon": 5}, "signal": {"holding_period": 5}, "split": {"min_train_days": 400, "refit_every": 63},
        "model": {"kind": "ridge", "params": {"alpha": 10.0}}}


def books(n=45, k=4, seed=0):
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(rng.normal(size=(n, k)), index=pd.bdate_range("2022-01-03", periods=n), columns=list("ABCD"))
    return raw.div(raw.abs().sum(axis=1), axis=0)                                                       # a daily book of gross one


@pytest.mark.parametrize("h", hz.HORIZONS)
def test_the_tranche_book_is_h_sleeves_of_one_hth_of_capital_and_matches_the_platforms_holding_period(h):
    b = books()
    out = hz.tranche_books(b, h)
    comb = out["combined"]
    ref = sg.holding_period(b, h)
    assert np.allclose(comb.to_numpy(), ref.to_numpy(), atol=1e-12), "the independent sleeve loop equals the platform's average of the last h books"
    for t in (h + 3, len(b) - 1):
        expected = sum(b.iloc[t - j] for j in range(h)) / h                                             # the h latest daily books, one per sleeve
        assert np.allclose(comb.iloc[t].to_numpy(), expected.to_numpy(), atol=1e-12)
    assert out["never_more_than_one_book"] and out["max_gross"] <= out["max_single_book_gross"] + 1e-12 and out["warmup_days"] == h - 1
    warm_gross = comb.iloc[h - 1:].abs().sum(axis=1).max()
    assert h * warm_gross - 1e-9 <= out["naive_sum_gross_max"] <= h * out["max_gross"] + 1e-9, "adding the overlapping books (instead of splitting capital among them) multiplies the exposure by the number of live tranches"
    assert len(out["sleeves"]) == h and all(s.shape == b.shape for s in out["sleeves"].values())


@pytest.mark.parametrize("h", (5, 20))
def test_the_pnl_of_the_combined_book_is_the_average_of_the_sleeves_pnl_so_nothing_is_counted_twice(h):
    rng = np.random.default_rng(3)
    b = books(n=70)
    r = pd.DataFrame(rng.normal(0.0003, 0.01, b.shape), index=b.index, columns=b.columns)
    out = hz.tranche_books(b, h)
    warm = h - 1
    combined_pnl = (out["combined"].shift(1) * r).sum(axis=1)                                          # the book held at the close of t earns the return of t + 1
    sleeve_pnl = sum((s.shift(1) * r).sum(axis=1) for s in out["sleeves"].values()) / h
    assert np.allclose(combined_pnl.iloc[warm + 1:], sleeve_pnl.iloc[warm + 1:], atol=1e-12), "linearity: the average book earns the average of the sleeves' earnings"
    horizon_ret = r.rolling(h).sum().shift(-h)
    shortcut = (b * horizon_ret / h).sum(axis=1).dropna()                                              # the tempting shortcut: each day's h-day return divided by h, as if it were that day's P&L
    assert not np.allclose(combined_pnl.reindex(shortcut.index).iloc[warm + 1:].to_numpy(), shortcut.iloc[warm + 1:].to_numpy(), atol=1e-6), "the shortcut is a different series from the equity of the holdings"
    assert combined_pnl.iloc[warm + 1:].abs().max() <= r.abs().max().max() * out["max_single_book_gross"] + 1e-12, "no leverage beyond one daily book: a day's P&L cannot exceed the largest return times one book's gross"


def test_the_label_interval_of_every_horizon_is_its_own_and_starts_after_the_features_end():
    b = make_bundle(8, 700, seed=5, alpha=0.0)
    for h in hz.HORIZONS:
        cfg = {"feature_groups": ["price"], "target": f"fwd_ret_{h}", "horizon": h, "lag": 1, "min_history": 252, "winsor": 0.01, "normalize": "cross_sectional_z", "benchmark": "equal_weight",
               "start": None, "end": None, "regime_detector": None, "universe": "etf"}
        d = ds.build_dataset(b, cfg, f"h{h}")
        iv = d.label_intervals()
        dates = d.dates
        i = 50
        assert pd.Timestamp(iv.iloc[i]["label_start_at"]) == dates[i + 1] and pd.Timestamp(iv.iloc[i]["label_end_at"]) == dates[i + 1 + h], (h, iv.iloc[i].to_dict())
        assert pd.Timestamp(iv.iloc[i]["label_available_at"]) == dates[i + 1 + h] and iv["label_end_at"].isna().sum() == 1 + h, "the last lag + h decisions have no label yet"
        fwd = tg.forward_return(d.prices, h, 1)
        a = d.Y[f"fwd_ret_{h}"].xs(d.assets[2], level="asset").dropna()
        assert np.allclose(a.to_numpy(), fwd[d.assets[2]].reindex(a.index).to_numpy().astype(np.float32), rtol=1e-5, atol=1e-7)
        assert d.meta["label_convention"] and d.meta["audit"]["label"]["unlabelled_tail_dates"] == 1 + h


def test_conventions_must_match_across_horizons_except_the_three_settings_that_define_one():
    changes = hz.horizon_changes(BASE)
    assert [c["dataset"]["horizon"] for c in changes] == [1, 10, 20] and all(c["signal"]["holding_period"] == c["dataset"]["horizon"] for c in changes)
    ok = hz.comparable_conventions(BASE, cp.merge(BASE, changes[1]))
    assert ok["matched"] is True and ok["horizons"] == [5, 10] and sorted(ok["differences"]) == ["dataset.horizon", "dataset.target", "signal.holding_period"] and ok["problems"] == []
    bad = hz.comparable_conventions(BASE, cp.merge(cp.merge(BASE, changes[1]), {"backtest": {"cost_bps": 1.0}}))
    assert bad["matched"] is False and bad["unmatched"] == ["backtest.cost_bps"]
    held = hz.comparable_conventions(BASE, cp.merge(BASE, {"dataset": {"target": "fwd_ret_10", "horizon": 10}}))
    assert any("holding period is not the horizon" in p for p in held["problems"])


def test_a_horizon_study_runs_as_a_frozen_card_and_is_judged_on_the_daily_net_returns_not_on_rescaled_horizon_returns(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    bundle = make_bundle(10, 1150, seed=61, alpha=1.0)
    card = cp.new_card("a longer horizon with a matching holding period earns more net of costs", "horizon", BASE, hz.horizon_changes(BASE, (1, 10)), study_id="study-hz",
                       one_change_description="the forecast horizon, with the holding period that matches it", n_boot=400)
    assert card["primary_metric"] == "net_sharpe" and all(c["changed"] == ["dataset.horizon", "dataset.target", "signal.holding_period"] for c in card["candidates"])
    cp.freeze(reg, card)
    out = cp.run(reg, card["card_id"], bundle=bundle)
    res = out["results"]
    assert res["label_horizon"] == 5 and [c["status"] for c in res["candidates"]] == ["compared", "compared"]
    for c in res["candidates"]:
        d = c["metrics"]["net_sharpe"]["difference"]
        assert d["eligibility"] == "applied" and c["metrics"]["net_sharpe"]["aligned"] > 100 and c["metrics"]["ic_own_horizon"]["note"].startswith("each arm's IC against its own horizon")
        assert d["settings"]["mean_block"] >= 5, "the bootstrap block is at least the longer holding period of the pair: the overlap is inside the blocks"
        assert c["verdict"] in ("supported_on_development", "worse", "inconclusive") and c["reason"]
        assert c["metrics"]["ic"]["label"].startswith("rank IC against the raw 5-session forward return"), "one common label for the IC difference"
    assert out["gate_decision"]["status"] == "not_evaluated"
