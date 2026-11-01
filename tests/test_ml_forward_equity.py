"""Release 5, P1-24: the actual daily equity of a prospective strategy is built from committed holdings, fills, cash and the daily prices.

Forecast evidence (a rank IC per decision) and strategy evidence (daily equity) are separate streams. A multi-day forecast produced every day makes overlapping outcomes; it does not define independent
daily portfolios, and nothing here divides a horizon return by the horizon and compounds it. The reference below is worked independently from the price table, with fractional shares and no costs.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml import execution as xe, monitoring as mon, prospective as pr, realism as rz, registry as rg

SV = "sv-equity"
DATES = pd.bdate_range("2024-01-02", periods=16)
PA = pd.Series(100.0 * np.cumprod(np.r_[1.0, np.full(15, 1.01)]), index=DATES)                       # A rises 1% a day
PB = pd.Series(50.0 * np.cumprod(np.r_[1.0, np.tile([1.02, 0.99], 8)[:15]]), index=DATES)             # B alternates +2% / -1%
PRICES = pd.DataFrame({"A": PA, "B": PB})
FREE = xe.ExecutionSpec(capital=1e8, delay=1, sizing="fill_time", commission_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0, max_participation=None, min_trade_weight=0.0, lot=1,
                        impact=rz.ImpactSpec(coefficient=0.0))
LABEL_DAYS = 1


@pytest.fixture
def reg(tmp_path):
    return rg.Registry(tmp_path / "ml")


def commit_book(reg, day, weights, classification="prospective", horizon=5):
    d = DATES[day]
    if classification == "prospective":
        kw = dict(now=(d + pd.Timedelta(hours=20)).timestamp(), data_cutoff=d, fit_cutoff=d - pd.Timedelta(days=20), model_created=d - pd.Timedelta(days=40))
    else:                                                                                                # written long after: a replay
        kw = dict(now=(d + pd.Timedelta(days=400)).timestamp(), data_cutoff=d, fit_cutoff=d, model_created=d - pd.Timedelta(days=40))
    return pr.commit(reg, strategy_version=SV, decision_ts=d, universe_version="u", horizon=horizon, forecasts={"A": 1.0, "B": 0.0}, weights=weights, artifact_hashes={"model": "m", "data": "d"},
                     label_start=d + pd.offsets.BDay(LABEL_DAYS), actor="t", **kw)


def reference(books, fills):
    """Daily equity of fully invested books re-sized at the fill close, held (not rebalanced) until the next fill, with no costs and fractional shares."""
    eq = pd.Series(1e8, index=DATES, dtype=float)
    shares = {"A": 0.0, "B": 0.0}
    cash = 1e8
    for i, d in enumerate(DATES):
        if i in fills:
            wanted = books[fills[i]]
            equity_now = cash + sum(shares[a] * PRICES.loc[d, a] for a in shares)
            cash = equity_now - sum(wanted.get(a, 0.0) * equity_now for a in shares)
            shares = {a: wanted.get(a, 0.0) * equity_now / PRICES.loc[d, a] for a in shares}
        eq.iloc[i] = cash + sum(shares[a] * PRICES.loc[d, a] for a in shares)
    return eq


def test_the_equity_is_built_from_committed_holdings_and_matches_a_hand_worked_daily_path(reg):
    books = {0: {"A": 1.0}, 4: {"B": 1.0}, 7: {"A": 0.5, "B": 0.5}}
    for day, w in books.items():
        commit_book(reg, day, w)
    out = pr.equity(reg, SV, PRICES, FREE)
    assert out["eligibility"] == "applied" and out["n_decisions"] == 3 and out["decisions_without_prices"] == 0
    ref = reference(books, {1: 0, 5: 4, 8: 7})                                                          # a decision at one close is filled at the next
    got = out["equity"].reindex(DATES)
    assert np.allclose(got.to_numpy(), ref.to_numpy(), rtol=2e-6), (got / ref - 1.0).abs().max()
    assert np.allclose(out["returns"].reindex(DATES).iloc[1:].to_numpy(), ref.pct_change().iloc[1:].to_numpy(), atol=2e-6), "daily returns follow from the equity path"
    assert out["reconciliation"]["reconciled"] and out["reconciliation"]["ledger_ok"], out["reconciliation"]
    assert "forecast horizon plays no part" in out["note"]
    # the day of the decision earns nothing: the book is not held until it is filled one session later
    assert got.iloc[0] == pytest.approx(1e8) and got.iloc[1] == pytest.approx(1e8, rel=1e-9)


def test_the_forecast_horizon_does_not_change_the_equity(reg, tmp_path):
    books = {0: {"A": 1.0}, 4: {"B": 1.0}}
    other = rg.Registry(tmp_path / "other")
    for day, w in books.items():
        commit_book(reg, day, w, horizon=1)
        commit_book(other, day, w, horizon=20)
    a = pr.equity(reg, SV, PRICES, FREE)["equity"]
    b = pr.equity(other, SV, PRICES, FREE)["equity"]
    assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-9), "a 1-day and a 20-day forecast with the same book have the same actual P&L: the horizon is a property of the forecast, not of the holdings"


def test_staggered_overlapping_forecasts_make_one_book_per_decision_and_not_five_portfolios(reg):
    books = {k: ({"A": 1.0} if k % 2 == 0 else {"B": 1.0}) for k in range(6)}                          # a 5-day forecast produced every day, flipping the book daily
    for k, w in books.items():
        commit_book(reg, k, w, horizon=5)
    out = pr.equity(reg, SV, PRICES, FREE)
    ref = reference(books, {k + 1: k for k in books})
    assert np.allclose(out["equity"].reindex(DATES).to_numpy(), ref.to_numpy(), rtol=2e-6)
    naive = (1.0 + (PRICES.shift(-5) / PRICES - 1.0).mul(pd.DataFrame({"A": [1.0, 0, 1, 0, 1, 0], "B": [0.0, 1, 0, 1, 0, 1]}, index=DATES[:6]).reindex(DATES).fillna(0.0)).sum(axis=1)[:6] / 5.0).prod() - 1.0
    actual = out["equity"].iloc[-1] / out["equity"].iloc[0] - 1.0
    assert abs(naive - actual) > 1e-3, "dividing each 5-day return by five and compounding is a different (wrong) number from the equity of the holdings"


def test_replays_have_a_separate_stream_and_do_not_enter_the_prospective_equity(reg):
    commit_book(reg, 0, {"A": 1.0}, classification="prospective")
    commit_book(reg, 4, {"B": 1.0}, classification="replay")
    pro = pr.equity(reg, SV, PRICES, FREE)
    rep = pr.equity(reg, SV, PRICES, FREE, classification="replay")
    assert pro["n_decisions"] == 1 and rep["n_decisions"] == 1
    assert np.allclose(pro["equity"].reindex(DATES).to_numpy(), reference({0: {"A": 1.0}}, {1: 0}).to_numpy(), rtol=2e-6)
    assert np.allclose(rep["equity"].reindex(DATES).to_numpy(), reference({4: {"B": 1.0}}, {5: 4}).to_numpy(), rtol=2e-6)


def test_no_committed_weights_or_no_overlapping_dates_is_insufficient_not_a_flat_line(reg):
    assert pr.equity(reg, SV, PRICES, FREE)["eligibility"] == "insufficient"
    pr.commit(reg, strategy_version=SV, decision_ts=pd.Timestamp("2030-01-02"), universe_version="u", horizon=5, forecasts={"A": 1.0}, weights={"A": 1.0}, actor="t")
    out = pr.equity(reg, SV, PRICES, FREE, classification="replay")
    assert out["eligibility"] == "insufficient" and "none of the committed decision dates" in out["reason"] and out["equity"] is None


def test_the_forward_record_no_longer_compounds_horizon_returns_into_a_pseudo_equity():
    signals = [{"signal_date": str((DATES[k]).date()), "created": str((DATES[k] + pd.Timedelta(days=1)).date()), "realised": r, "realised_date": None, "weights": {}, "scores": {}, "turnover": 0.0, "id": k}
               for k, r in enumerate([0.02, -0.01, 0.015, 0.0, None])]
    rec = mon.forward_record(signals, "x", "exp", horizon=5)
    assert rec["n_settled"] == 4 and rec["horizon_outcomes"]["n"] == 4 and "not a daily P&L" in rec["horizon_outcomes"]["note"]
    assert rec["mean_realised"] == pytest.approx(0.00625) and rec["hit_rate"] == pytest.approx(0.5)
    assert not {"cumulative", "sharpe", "series"} & set(rec), "the old pseudo-equity (return / horizon, compounded) and its Sharpe ratio are gone"
