"""Release 7, P2-15: IC, expected edge, net return and capacity at predeclared delays.

The timing is checked against a calendar worked by hand, the IC against a second implementation, and the stale-book rule against a book built so that every position earns only at its undelayed fill: a
session of delay must earn nothing, and three sessions of delay (a full cycle of the rotation) must earn it back. The report is then run on a saved experiment.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.ml import delay_report as dr, experiments as xp, registry as rg
from src.ml.targets import forward_return
from tests.ml_helpers import make_bundle

BT = {"rebalance": "daily", "cost_bps": 0.0, "slippage_bps": 0.0, "target_vol": None, "max_leverage": 1.5, "extra_lag": 0}


# ------------------------------------------------------------------------------------------------------------------ the declaration
def test_the_delays_are_declared_before_the_numbers_and_must_include_no_delay():
    d = dr.declare()
    assert d["delays"] == [0, 1, 2, 5] and d["capitals"] == [1e5, 1e6, 1e7, 1e8] and len(d["hash"]) == 16
    assert dr.declare([5, 0, 1, 1])["delays"] == [0, 1, 5] and dr.declare([0, 1])["hash"] != d["hash"] and dr.declare()["hash"] == d["hash"]
    for bad in ([1, 2], [0, 21], [], [0, 1.5]):
        with pytest.raises(ValueError, match="whole numbers of sessions"):
            dr.declare(bad)
    with pytest.raises(ValueError, match="positive"):
        dr.declare([0], [0.0])


# ------------------------------------------------------------------------------------------------------------------ timing from the actual timestamps
def test_a_delay_in_sessions_is_reported_in_calendar_days_from_the_real_index():
    ix = pd.bdate_range("2024-01-01", periods=30).delete(9)                  # 2024-01-12 (a Friday) is missing: a closure
    friday, monday, thursday = pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-11")
    t = dr.timing(ix, pd.DatetimeIndex([friday, monday, thursday]), 1)
    assert t["decisions"] == 3 and t["with_a_fill_session"] == 3 and t["without"] == 0
    assert t["calendar_days"]["mean"] == pytest.approx((3 + 1 + 4) / 3), "Friday to Monday is three days, Monday to Tuesday one, and Thursday to the next open session (Monday: the 12th is closed) four"
    assert t["calendar_days"]["median"] == 3.0 and t["calendar_days"]["max"] == 4.0 and t["longer_than_sessions"] == pytest.approx(2 / 3)
    five = dr.timing(ix, pd.DatetimeIndex([monday]), 5)
    assert five["calendar_days"]["max"] == 8.0, "five sessions from Monday the 8th across the closed Friday end on Tuesday the 16th: eight days"
    tail = dr.timing(ix, ix[-2:], 3)
    assert tail["with_a_fill_session"] == 0 and tail["without"] == 2 and tail["calendar_days"]["mean"] is None, "a decision with no later session is counted and left out, not extrapolated"
    assert dr.timing(ix, pd.DatetimeIndex([pd.Timestamp("2023-12-25")]), 1)["with_a_fill_session"] == 0, "a date not in the index has no fill"


# ------------------------------------------------------------------------------------------------------------------ IC at the delayed entry
def test_the_ic_at_a_delay_matches_a_second_implementation_and_falls_when_the_signal_is_stale():
    rng = np.random.default_rng(7)
    ix = pd.bdate_range("2020-01-01", periods=420)
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (420, 8)), axis=0)), index=ix, columns=list("ABCDEFGH"))
    fwd = forward_return(prices, 1, 1).reindex(index=ix, columns=prices.columns)
    scores = fwd.copy()                                                      # the score of day t IS the return it will earn at the undelayed fill
    mask = np.ones(len(ix), dtype=bool)
    fresh = dr.ic_at_delay(scores, prices, 1, 1, mask)
    assert fresh["mean_ic"] == pytest.approx(1.0) and fresh["entry_lag"] == 1
    stale = dr.ic_at_delay(scores, prices, 1, 2, mask)
    other = forward_return(prices, 1, 2).reindex(index=ix, columns=prices.columns)
    ref = [spearmanr(scores.loc[d], other.loc[d])[0] for d in ix if scores.loc[d].notna().all() and other.loc[d].notna().all()]
    assert stale["mean_ic"] == pytest.approx(float(np.mean(ref)), abs=1e-9) and abs(stale["mean_ic"]) < 0.1, "one session late, the same score knows nothing about the return that is earned"
    partial = mask.copy()
    partial[:300] = False
    assert dr.ic_at_delay(scores, prices, 1, 1, partial)["n_with_ic"] < fresh["n_with_ic"], "only the masked dates are read"
    assert dr.ic_at_delay(scores, prices, 1, 1, np.zeros(len(ix), dtype=bool))["eligibility"] == "insufficient"


# ------------------------------------------------------------------------------------------------------------------ the stale-book rule
def rotation(T=240, lag=1):
    """A long-only book that holds asset (t mod 3) in full when decided on day t, and returns of 1% that go to asset (u - lag - 1) mod 3 on day u. The engine fills a decision of day t at the close of
    day t + lag and the position earns from the next session, so the book earns only if it is executed at its undelayed fill, and again only after a whole turn of the rotation."""
    ix = pd.bdate_range("2021-01-04", periods=T)
    cols = list("ABC")
    book = pd.DataFrame(0.0, index=ix, columns=cols)
    rets = pd.DataFrame(0.0, index=ix, columns=cols)
    for i in range(T):
        book.iloc[i, i % 3] = 1.0
        rets.iloc[i, (i - lag - 1) % 3] = 0.01
    return ix, book, rets


def test_a_delayed_book_cannot_see_a_newer_signal_it_earns_only_where_the_stale_book_happens_to_fit():
    ix, book, rets = rotation()
    labels = pd.Series("validation", index=ix[:1])
    by = {d: dr.book_at_delay(book, rets, None, BT, 1, d, ix[0], labels, "validation") for d in (0, 1, 2, 3, 6)}
    assert by[0]["eligibility"] == "applied" and by[0]["n_days"] > 200
    assert by[0]["gross_ann_return"] == pytest.approx(0.01 * 252, rel=0.02)
    assert by[1]["gross_ann_return"] == pytest.approx(0.0, abs=1e-12) and by[2]["gross_ann_return"] == pytest.approx(0.0, abs=1e-12), "one or two sessions late the book holds the wrong asset on every day"
    assert by[3]["gross_ann_return"] == pytest.approx(by[0]["gross_ann_return"], rel=0.02) and by[6]["gross_ann_return"] == pytest.approx(by[0]["gross_ann_return"], rel=0.02), "a whole turn later it fits again"
    w = by[2]["weights"]
    u = ix.get_loc(w.index[13])
    assert (w.iloc[13].to_numpy() == book.iloc[u - 3].to_numpy()).all() and (w.iloc[14].to_numpy() == book.iloc[u + 1 - 3].to_numpy()).all(), (
        "the position held at the close of day u is the book decided three sessions earlier: lag 1 plus a delay of 2, as it was decided, not re-read")
    assert by[0]["breakeven_cost_bps"] is not None and by[1]["breakeven_cost_bps"] == pytest.approx(0.0, abs=1e-9)
    short = dr.book_at_delay(book.iloc[:20], rets.iloc[:20], None, BT, 1, 0, ix[0], labels, "validation")
    assert short["eligibility"] == "insufficient" and "days on the validation segment" in short["reason"]


def test_costs_come_off_the_executed_book_and_the_breakeven_cost_is_what_would_zero_the_net_result():
    ix, book, rets = rotation()
    labels = pd.Series("validation", index=ix[:1])
    free = dr.book_at_delay(book, rets, None, BT, 1, 0, ix[0], labels, "validation")
    dear = dr.book_at_delay(book, rets, None, {**BT, "cost_bps": 20.0}, 1, 0, ix[0], labels, "validation")
    assert dear["net_ann_return"] < free["net_ann_return"] and dear["costs_ann"] > 0 and dear["gross_ann_return"] == pytest.approx(free["gross_ann_return"], rel=1e-9)
    assert dear["net_ann_return"] == pytest.approx(dear["gross_ann_return"] - dear["costs_ann"], rel=1e-6)
    gross_per_turnover = free["breakeven_cost_bps"] / 1e4
    assert dear["costs_ann"] / dear["ann_turnover"] == pytest.approx(0.0020, rel=0.05) and gross_per_turnover > 0
    at_breakeven = dr.book_at_delay(book, rets, None, {**BT, "cost_bps": free["breakeven_cost_bps"]}, 1, 0, ix[0], labels, "validation")
    assert at_breakeven["net_ann_return"] == pytest.approx(0.0, abs=2e-3), "at the break-even cost the net return is about nothing"


# ------------------------------------------------------------------------------------------------------------------ capacity
def test_capacity_at_a_delay_uses_the_delayed_sessions_liquidity_and_is_not_assumed_without_volume():
    ix = pd.bdate_range("2022-01-03", periods=120)
    cols = list("ABCDEF")
    prices = pd.DataFrame(100.0, index=ix, columns=cols)
    book = pd.DataFrame(1.0 / 6.0, index=ix, columns=cols)
    vol = pd.DataFrame(1e6, index=ix, columns=cols)                         # $100m a day in each name
    assert dr.capacity_at_delay(book, prices, None, ix[60:80], 2, (1e5, 1e8), 0.1, 20)["status"] == "not_established"
    got = dr.capacity_at_delay(book, prices, vol, ix[60:80], 2, (1e5, 1e8), 0.1, 20)
    lo, hi = got["levels"]
    assert got["status"] == "measured" and lo["median_fill_share"] == 1.0 and hi["median_fill_share"] == pytest.approx(0.6, abs=1e-9), "$1e8 x 1/6 = $16.7m per name against a $10m limit: 60% gets through"
    thin = vol.copy()
    thin.iloc[62:] = 1e5                                                    # liquidity collapses on and after the 63rd session
    cut = dr.capacity_at_delay(book, prices, thin, ix[60:62], 3, (1e8,), 0.1, 5)
    assert cut["levels"][0]["median_fill_share"] < 0.6 + 1e-9
    assert dr.capacity_at_delay(book * 0.0, prices, vol, ix[60:80], 2, (1e5,), 0.1, 20)["status"] == "not_established"


# ------------------------------------------------------------------------------------------------------------------ a saved experiment
def test_the_report_on_a_saved_experiment_reads_only_the_validation_segment(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    bundle = make_bundle(10, 1150, seed=91, alpha=1.0)
    over = {"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}},
            "signal": {"method": "top_n_long_short", "n": 3, "holding_period": 1}, "backtest": {"cost_bps": 5.0, "slippage_bps": 2.0, "rebalance": "weekly"}}
    exp = xp.run_experiment(over, bundle=bundle, registry=reg, explain=False, robustness=False)
    out = dr.report(reg, exp.id)
    assert out["eligibility"] == "applied" and out["segment"] == "validation" and out["declaration"]["delays"] == [0, 1, 2, 5] and out["rule"].startswith("stale book")
    rows = out["rows"]
    assert [r["delay"] for r in rows] == [0, 1, 2, 5] and out["declared_cost_bps"] == 7.0
    for r in rows:
        assert r["timing"]["execution_lag_sessions"] == 1 + r["delay"] and r["timing"]["calendar_days"]["min" if False else "mean"] >= r["timing"]["execution_lag_sessions"]
        assert r["ic"]["eligibility"] if "eligibility" in r["ic"] else True
        assert r["portfolio"]["eligibility"] == "applied" and r["portfolio"]["n_days"] > 100 and r["capacity"]["status"] == "not_established"
    first = rows[0]
    if first["ic"]["mean_ic"] and first["ic"]["mean_ic"] > 0:
        assert first["ic_ratio_to_undelayed"] == pytest.approx(1.0)
    assert rows[0]["ic"]["entry_lag"] == 1 and rows[3]["ic"]["entry_lag"] == 6, "the IC is measured from the same fill the portfolio gets"
    assert rows[0]["portfolio"]["gross_ann_return"] == pytest.approx(rows[0]["portfolio"]["net_ann_return"] + rows[0]["portfolio"]["costs_ann"], rel=1e-6)
    with_cap = dr.report(reg, exp.id, volume=bundle.volume if getattr(bundle, "volume", None) is not None else pd.DataFrame(1e6, index=bundle.prices.index, columns=bundle.prices.columns), max_participation=0.1)
    assert all(r["capacity"]["status"] == "measured" for r in with_cap["rows"])
    with pytest.raises(ValueError, match="protected"):
        dr.report(reg, exp.id, segment="test")
    with pytest.raises(ValueError, match="segment must be one of"):
        dr.report(reg, exp.id, segment="whenever")
