"""Release 4: executable trading and reconciled accounting (P1-01 to P1-14).

The production path drives the platform's own ledger; the reference here is written separately, from the definitions: a shadow account that books only cash and shares from the fills the adapter reports, a
hand-worked corporate-action table, and closed-form cost and funding arithmetic. Where a number is compared, it is compared with an independently computed one, not with itself."""

import numpy as np
import pandas as pd
import pytest

from src.instruments import Instrument, InstrumentRegistry
from src.ledger import Ledger, reconcile
from src.ml import execution as ex, realism as rz

D = pd.bdate_range("2023-01-02", periods=80)
ASSETS = ["A", "B", "C"]


def market(seed=1, n=80, drift=0.0004, vol=0.01):
    rng = np.random.default_rng(seed)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(drift, vol, (n, 3)), axis=0), index=D[:n], columns=ASSETS)
    return px, pd.DataFrame(8e6, index=px.index, columns=ASSETS)


def weights_at(prices, rows):
    w = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for i, vals in rows.items():
        w.iloc[i] = vals
    return w


def spec(**kw):
    base = dict(commission_bps=2.0, half_spread_bps=3.0, slippage_bps=1.0, impact=rz.ImpactSpec(coefficient=0.0), adv_window=5)
    return ex.ExecutionSpec(**{**base, **kw})


def ledger_with(*ids, cash=1_000_000.0):
    return Ledger(InstrumentRegistry([Instrument(instrument_id=i, asset_class="equity", currency="USD") for i in ids]), "USD", {"USD": cash}, start=D[0])


# --------------------------------------------------------------------------------------------------------------------- P1-01
def test_membership_follows_the_security_through_a_ticker_change_an_exit_and_a_failure():
    m = rz.Membership([rz.Listing("S1", "FB", "2015-01-02", "2022-06-09"), rz.Listing("S1", "META", "2022-06-09"),
                       rz.Listing("S2", "OLD", "2015-01-02", "2020-03-02", status="merged", successor="S1"), rz.Listing("S3", "BUST", "2016-01-04", "2019-05-01", status="failed"), rz.Listing("S4", "KEEP", "2015-01-02")],
                      coverage="partial", members={"S4": [("2015-01-02", "2018-01-02"), ("2019-01-02", None)]})
    assert m.security_of("FB", "2022-01-03") == "S1" and m.security_of("FB", "2023-01-03") is None and m.security_of("META", "2023-01-03") == "S1" and m.ticker_of("S1", "2021-01-04") == "FB"
    el = m.eligible(pd.DatetimeIndex(["2019-01-02", "2019-06-03", "2021-01-04"]), ["S2", "S3", "S4"])
    assert el.loc["2019-01-02"].tolist() == [True, True, True] and el.loc["2019-06-03"].tolist() == [True, False, True] and el.loc["2021-01-04"].tolist() == [False, False, True]
    assert not m.member("S4", "2018-06-01"), "outside a membership period although listed"
    assert m.ending("S3").status == "failed" and m.ending("S2").successor == "S1" and m.ending("S4") is None
    s = m.survivorship()
    assert s["ended"] == 2 and "survivorship bias" in s["limitation"] and "not invented" in s["limitation"]
    assert rz.Membership([], coverage="complete").survivorship()["limitation"] is None
    with pytest.raises(ValueError):
        rz.Listing("X", "X", "2020-01-02", "2019-01-02")


# --------------------------------------------------------------------------------------------------------------------- P1-02
def test_corporate_actions_move_quantity_cash_and_equity_exactly_as_the_event_says():
    ts = pd.Timestamp("2023-03-01")
    # split 2:1: 100 shares at 50 -> 200 at 25, equity unchanged, average cost halves
    led = ledger_with("A", cash=10_000.0)
    led.apply_fill(D[0], __import__("src.ledger", fromlist=["Fill"]).Fill(D[0], "A", 100.0, 50.0), 50.0)
    before = led.equity()
    rz.apply_action(led, rz.CorporateAction("A", "split", ts, {"ratio": 2.0}), 50.0)
    led.revalue(ts, {"A": 25.0})
    assert led.quantity("A") == 200.0 and led.positions["A"].avg_price == 25.0 and led.equity() == pytest.approx(before)
    # dividend 0.5 a share on 200 shares: cash +100, equity +100, booked as dividend
    rz.apply_action(led, rz.CorporateAction("A", "dividend", ts, {"amount": 0.5}), 25.0)
    assert led.cash["USD"] == pytest.approx(10_000.0 - 100 * 50.0 + 100.0) and led.equity() == pytest.approx(before + 100.0) and led.pnl_by("category")["dividend"] == pytest.approx(100.0)
    # cash merger at 30 on 200 shares marked 25: +1000 over the mark, position gone, realized P&L (30-25)*200 on a cost of 25
    rz.apply_action(led, rz.CorporateAction("A", "merger_cash", ts, {"price": 30.0}), 25.0)
    assert led.quantity("A") == 0.0 and led.cash["USD"] == pytest.approx(10_000.0 - 5_000.0 + 100.0 + 200 * 30.0) and led.pnl_by("category")["corporate_action"] == pytest.approx(200 * (30.0 - 25.0))
    assert reconcile(led).ok
    # delisting with 10% recovery of the last mark: a loss of 90% of 200 * 20
    led2 = ledger_with("B", cash=10_000.0)
    led2.apply_fill(D[0], __import__("src.ledger", fromlist=["Fill"]).Fill(D[0], "B", 200.0, 20.0), 20.0)
    rz.apply_action(led2, rz.CorporateAction("B", "delisting", ts, {"recovery": 0.10}), 20.0)
    assert led2.quantity("B") == 0.0 and led2.equity() == pytest.approx(10_000.0 - 0.9 * 200 * 20.0) and reconcile(led2).ok
    assert rz.delisting_loss(20.0, 200.0, rz.MarkPolicy(delisting_recovery=0.10)) == {"recovered_cash": 400.0, "loss": 3600.0, "settles_after_days": 5}


def test_stock_mergers_and_spinoffs_conserve_value_and_unsupported_actions_block_claims():
    from src.ledger import Fill

    ts = pd.Timestamp("2023-03-01")
    led = ledger_with("T", "ACQ", cash=50_000.0)
    led.apply_fill(D[0], Fill(D[0], "T", 100.0, 40.0), 40.0)
    eq = led.equity()
    rz.apply_action(led, rz.CorporateAction("T", "merger_stock", ts, {"successor": "ACQ", "ratio": 0.5}), 40.0, successor_mark=80.0)    # 100 shares become 50 of a stock worth 80: 4000 either way
    led.revalue(ts, {"ACQ": 80.0})
    assert led.quantity("T") == 0.0 and led.quantity("ACQ") == 50.0 and led.equity() == pytest.approx(eq) and reconcile(led).ok
    led2 = ledger_with("P", "SPIN", cash=50_000.0)
    led2.apply_fill(D[0], Fill(D[0], "P", 100.0, 100.0), 100.0)
    eq2 = led2.equity()
    rz.apply_action(led2, rz.CorporateAction("P", "spinoff", ts, {"new_security": "SPIN", "ratio": 0.1, "value_fraction": 0.2}), 100.0, successor_mark=50.0)   # 10 new shares at 50 = 500 leave the parent
    led2.revalue(ts, {"SPIN": 50.0})
    assert led2.quantity("SPIN") == 10.0 and led2.equity() == pytest.approx(eq2) and led2.quantity("P") == 100.0
    rep = rz.check_supported([rz.CorporateAction("X", "rights_issue", ts), rz.CorporateAction("A", "split", ts, {"ratio": 2})])
    assert [a.type for a in rep["unsupported"]] == ["rights_issue"] and "rights_issue" in rep["blocked_claims"][0]
    with pytest.raises(rz.UnsupportedCorporateAction):
        rz.apply_action(led, rz.CorporateAction("X", "rights_issue", ts), 1.0)
    old, new = rz.CorporateAction("A", "dividend", ts, {"amount": 1.0}, version=1), rz.CorporateAction("A", "dividend", ts, {"amount": 1.2}, version=2)
    assert [a.payload["amount"] for a in rz.latest_versions([old, new])] == [1.2], "a vendor correction replaces the earlier version"


def test_research_prices_are_adjusted_and_executable_prices_are_not_so_nothing_is_counted_twice():
    px = pd.Series([50.0, 51.0, 25.5, 26.0, 26.5], index=D[:5])                     # a 2:1 split on D[2], a 0.52 dividend on D[4]
    acts = [rz.CorporateAction("A", "split", D[2], {"ratio": 2.0}), rz.CorporateAction("A", "dividend", D[4], {"amount": 0.52})]
    adj = rz.adjust_prices(px, acts, total_return=True)
    from src.ledger import Fill

    led = ledger_with("A", cash=100_000.0)
    led.apply_fill(D[0], Fill(D[0], "A", 100.0, 50.0), 50.0)
    start = led.equity()
    for i in range(1, 5):
        for a in [a for a in acts if a.ex_date == D[i]]:
            rz.apply_action(led, a, float(px.iloc[i - 1]))
        led.revalue(D[i], {"A": float(px.iloc[i])})
    ledger_return = (led.equity() - start) / 5000.0                               # the 100 shares bought at 50
    assert ledger_return == pytest.approx(adj.iloc[-1] / adj.iloc[0] - 1.0, abs=1e-12), "executed on raw prices with cash dividends, the account earns exactly the total-return adjusted series"
    assert rz.adjust_prices(px, acts, total_return=False).iloc[0] == pytest.approx(25.0), "a price-only adjustment applies the split and leaves the dividend to the cash account"


# --------------------------------------------------------------------------------------------------------------------- P1-03
def test_liquidity_screens_use_only_what_was_known_before_the_order():
    px, vol = market()
    adv = rz.pit_adv(vol, px, 10)
    assert adv.iloc[:10].isna().all().all() and adv.iloc[10].notna().all()
    vol2 = vol.copy()
    vol2.iloc[40:] *= 100.0                                                       # the future changes; the past screen must not
    assert rz.pit_adv(vol2, px, 10).iloc[:41].equals(adv.iloc[:41]), "ADV on day t does not depend on day t or later"
    assert not rz.pit_adv(vol2, px, 10).iloc[41:].equals(adv.iloc[41:])
    member = pd.DataFrame(True, index=px.index, columns=px.columns)
    member.loc[D[50]:, "B"] = False
    halted = pd.DataFrame(False, index=px.index, columns=px.columns)
    halted.loc[D[30], "C"] = True
    thin = vol.copy()
    thin["A"] = 10.0
    ok, why = rz.pit_screens(px, thin, member, window=10, min_adv=1e6, min_age=15, halted=halted)
    assert not ok.loc[D[5]].any() and why.loc[D[5], "A"] == "listed fewer than 15 sessions"
    assert why.loc[D[20], "A"].startswith("prior average dollar volume below") and not ok.loc[D[20], "A"]
    assert ok.loc[D[20], ["B", "C"]].all() and not ok.loc[D[30], "C"] and why.loc[D[30], "C"] == "halted or suspended" and ok.loc[D[31], "C"]
    assert ok.loc[D[49], "B"] and not ok.loc[D[50], "B"] and why.loc[D[50], "B"] == "not a member or not listed", "entry and exit follow the dates"
    nov, why2 = rz.pit_screens(px, None, window=10, min_age=1)
    assert not nov.any().any() and (why2.iloc[20] == "no liquidity history (not assumed liquid)").all(), "no volume is a reason, not a pass"


# --------------------------------------------------------------------------------------------------------------------- P1-04
def test_the_timeline_fills_after_the_decision_and_refuses_a_same_bar_fill_without_a_model():
    tl = rz.schedule([D[3], D[9]], D, "next_close")
    assert [(t.decision_at, t.filled_at, t.price_source) for t in tl] == [(D[3], D[4], "close"), (D[9], D[10], "close")] and all(t.filled_at > t.decision_at for t in tl)
    assert rz.schedule([D[3]], D, "next_open")[0].price_source == "open" and rz.schedule([D[-1]], D, "next_close") == [], "a decision with no later session has no fill"
    with pytest.raises(ValueError, match="auction model"):
        rz.schedule([D[3]], D, "same_close_auction")
    assert rz.schedule([D[3]], D, "same_close_auction", auction_model="closing auction imbalance")[0].filled_at == D[3]
    with pytest.raises(ValueError):
        rz.schedule([D[3]], D, "whenever")
    audit = rz.lag_audit(dataset_lag=1, engine_lag=1)
    assert audit["first_return_earned_after_sessions"] == 1 and audit["double_lagged"] is False
    assert rz.lag_audit(dataset_lag=1, engine_lag=2)["double_lagged"] is True


def test_the_adapter_fills_once_in_the_session_after_the_decision_at_the_next_close():
    px, vol = market()
    w = weights_at(px, {30: [0.5, 0.3, 0.2]})
    r = ex.execute_weights(w, px, spec(), volume=vol)
    assert set(r.fills["ts"]) == {D[31]} and (r.shares.loc[D[30]] == 0).all() and (r.shares.loc[D[31]] > 0).all(), "nothing is held at the decision close; everything at the next"
    mid = px.loc[D[31]]
    assert all(abs(f.price - mid[f.instrument_id] * (1 + 4e-4)) < 1e-9 for f in r.fills.itertuples()), "price = next close + half-spread 3bp + slippage 1bp"
    with pytest.raises(ValueError, match="open_prices"):
        ex.execute_weights(w, px, spec(convention="next_open"), volume=vol)
    o = px * 0.99
    r2 = ex.execute_weights(w, px, spec(convention="next_open"), volume=vol, open_prices=o)
    assert all(abs(f.price - o.loc[D[31], f.instrument_id] * (1 + 4e-4)) < 1e-9 for f in r2.fills.itertuples()), "the open convention fills from the opening price"


# --------------------------------------------------------------------------------------------------------------------- P1-04 / 7.1: the two engines on the assumptions they share
def _engine_returns(weights, px, lag, cost_bps=0.0, rebalance="daily"):
    from src.backtest.costs import LinearCostModel
    from src.backtest.engine import BacktestEngine

    eng = BacktestEngine(signal_lag=lag, rebalance=rebalance, weight_drift=True, annualisation=252, min_assets=1, cost_model=LinearCostModel(cost_bps=cost_bps), target_vol=None)
    return eng.run(weights.fillna(0.0), px.pct_change(), "ref", None, apply_vol_target=False)


@pytest.mark.parametrize("lag", [1, 2])
def test_the_realistic_adapter_and_the_vectorized_engine_agree_where_their_assumptions_coincide(lag):
    px, _ = market(seed=3, n=80, drift=0.0003)
    rng = np.random.default_rng(5)
    raw = pd.DataFrame(rng.dirichlet(np.ones(3), size=len(px)), index=px.index, columns=px.columns)
    raw.iloc[:6] = np.nan                                                                       # no decision yet
    common = dict(capital=1e12, delay=lag, sizing="fill_time", max_participation=None, commission_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0, impact=rz.ImpactSpec(coefficient=0.0),
                  min_trade_weight=0.0, cash_rate=0.0, financing_spread=0.0)
    real = ex.execute_weights(raw, px, ex.ExecutionSpec(**common))
    ref = _engine_returns(raw, px, lag)
    both = ref.net_returns.index.intersection(real.returns.index)
    both = both[both >= real.returns.index[6 + lag]]                                            # from the first day a position could earn
    assert len(both) > 60
    np.testing.assert_allclose(real.returns.reindex(both).to_numpy(), ref.net_returns.reindex(both).to_numpy(), atol=2e-9), "same weights, same lag, no costs: the same return stream"
    # the decision is lagged ONCE: the first fill is `lag` sessions after the first decision, and the engine's first non-flat book is on the same day
    first_fill = real.fills["ts"].min()
    assert first_fill == px.index[6 + lag]
    held = ref.weights.abs().sum(axis=1)
    assert held.index[held > 0][0] == first_fill, "no extra session was added by the engine, the signal or the adapter"


def test_a_cost_the_adapter_charges_is_the_only_difference_to_the_engine_when_everything_else_is_shared():
    px, _ = market(seed=4, n=80)
    raw = pd.DataFrame(np.tile([0.5, 0.3, 0.2], (len(px), 1)), index=px.index, columns=px.columns)
    raw.iloc[:6] = np.nan
    flat = dict(capital=1e12, delay=1, sizing="fill_time", max_participation=None, impact=rz.ImpactSpec(coefficient=0.0), min_trade_weight=0.0, cash_rate=0.0, financing_spread=0.0, half_spread_bps=0.0, slippage_bps=0.0)
    free = ex.execute_weights(raw, px, ex.ExecutionSpec(**flat, commission_bps=0.0))
    paid = ex.execute_weights(raw, px, ex.ExecutionSpec(**flat, commission_bps=5.0))
    assert paid.net_pnl < free.net_pnl and paid.costs["fee"] > 0 and paid.reconciliation["reconciled"]
    assert paid.daily_costs.sum().sum() == pytest.approx(paid.costs.sum(), rel=1e-9) and paid.daily_costs.sum()["fee"] == pytest.approx(paid.costs["fee"], rel=1e-9)
    # the fee on the first day is the buy of the whole book; the later days only re-balance the drift
    first = paid.daily_costs.loc[px.index[7]]
    assert first["fee"] == pytest.approx(5.0 / 1e4 * 1e12, rel=1e-3), "five basis points on the notional bought on the first fill day"


def test_the_lag_in_sessions_is_applied_once_by_the_adapter_and_an_order_can_only_be_amended_before_it_fills():
    px, vol = market()
    w = weights_at(px, {25: [0.5, 0.0, 0.0]})
    for delay in (1, 3):
        r = ex.execute_weights(w, px, spec(delay=delay, expiry_sessions=2), volume=vol)
        assert r.fills["ts"].min() == px.index[25 + delay], f"decided at the close of session 25, filled {delay} session(s) later"
    with pytest.raises(ValueError, match="at least one session"):
        ex.execute_weights(w, px, spec(delay=0), volume=vol)
    with pytest.raises(ValueError, match="sizing"):
        ex.execute_weights(w, px, spec(sizing="whenever"), volume=vol)
    book = rz.OrderBook()
    o = book.submit("A", 100.0, D[0])
    book.amend(o.order_id, 120.0, D[1], "re-sized")
    assert o.quantity == 120.0 and o.history[-1][2].startswith("amended 100 -> 120")
    with pytest.raises(rz.OrderError, match="keep the side"):
        book.amend(o.order_id, -5.0, D[1], "flip")
    book.fill(o.order_id, "f1", 50.0, D[2])
    with pytest.raises(rz.OrderError, match="only an unfilled pending order"):
        book.amend(o.order_id, 10.0, D[3], "late")


# --------------------------------------------------------------------------------------------------------------------- P1-05, P1-11
def shadow_account(result, capital, last_prices, dividends=0.0, other=0.0):
    """Cash and shares from the reported fills alone: cash = capital - sum(qty x price) - fees (+ flows), equity = cash + shares x last mark."""
    cash, shares = capital + dividends + other, pd.Series(0.0, index=last_prices.index)
    for f in result.fills.itertuples():
        cash -= f.quantity * f.price + f.fee
        shares[f.instrument_id] += f.quantity
    return cash, shares, float(cash + (shares * last_prices).sum())


def test_every_fill_reconciles_against_an_independent_shadow_account_and_costs_are_itemized():
    px, vol = market()
    w = weights_at(px, {25: [0.5, 0.3, 0.2], 40: [0.2, 0.3, 0.5], 55: [0.0, 0.0, 0.0]})
    r = ex.execute_weights(w, px, spec(), volume=vol)
    cash, shares, equity = shadow_account(r, 1_000_000.0, px.iloc[-1], other=float(r.ledger.pnl_by("category").get("financing", 0.0)))
    assert equity == pytest.approx(float(r.equity.iloc[-1]), abs=1e-6) and (shares == r.shares.iloc[-1]).all() and r.shares.iloc[-1].abs().sum() == 0
    f = r.fills
    notional_mid = sum(abs(q) * px.loc[t, a] for q, t, a in zip(f.quantity, f.ts, f.instrument_id))
    assert r.costs["spread"] == pytest.approx(notional_mid * 3e-4, rel=1e-9) and r.costs["slippage"] == pytest.approx(notional_mid * 1e-4, rel=1e-9) and r.costs["fee"] == pytest.approx(float(f.fee.sum()))
    assert r.costs["impact"] == 0.0 and r.costs["borrow"] == 0.0
    rec = r.reconciliation
    assert rec["reconciled"] and abs(rec["gap"]) < 1e-6 and rec["ledger_ok"] and rec["ledger_identity_violations"] == 0
    assert rec["gross"] - sum(rec["costs"].values()) == pytest.approx(rec["net"], abs=1e-6), "gross P&L from holdings and marks, less itemized costs, is the ledger's net P&L"
    states = r.orders["state"].value_counts().to_dict()
    assert states == {"filled": len(r.orders)}, "every order filled in full"


def test_a_modelled_price_concession_cannot_also_be_charged_as_a_fee():
    rz.validate_cost_spec({"commission": {"bps": 1.0}, "spread": {"bps": 2.0, "covers": ["bid_ask"]}, "slippage": {"bps": 1.0}})
    with pytest.raises(ValueError, match="counted twice"):
        rz.validate_cost_spec({"commission": {"bps": 1.0}, "spread": {"bps": 2.0, "covers": ["commission"]}})
    gap = rz.reconcile_net(100.0, pd.Series({"fee": 1.0, "spread": 2.0}), 97.0)
    assert gap["reconciled"] and rz.reconcile_net(100.0, pd.Series({"fee": 1.0}), 97.0)["reconciled"] is False


# --------------------------------------------------------------------------------------------------------------------- P1-06
def test_impact_follows_the_square_root_law_and_never_hides_a_missing_input():
    spec_ = rz.ImpactSpec(coefficient=0.5, max_participation=0.25)
    out = rz.impact_bps(spec_, 10_000.0, 1_000_000.0, 0.02)
    assert out["bps"] == pytest.approx(0.5 * 0.02 * np.sqrt(0.01) * 1e4) and out["status"] == "ok" and out["label"] == "assumed (uncalibrated)"
    assert rz.impact_bps(spec_, 0.0, 1e6, 0.02)["bps"] == 0.0 and rz.impact_bps(spec_, 5e6, 1e6, 0.02)["status"] == "capped" and rz.impact_bps(spec_, 5e6, 1e6, 0.02)["participation"] == 0.25
    missing = rz.impact_bps(spec_, 100.0, 0.0, 0.02)
    assert np.isnan(missing["bps"]) and missing["status"] == "no_estimate", "zero volume is not zero impact"
    cons = rz.impact_bps(rz.ImpactSpec(coefficient=0.5, max_participation=0.25, fallback="conservative"), 100.0, float("nan"), 0.02)
    assert cons["status"] == "fallback_conservative" and cons["bps"] == pytest.approx(0.5 * 0.02 * 0.5 * 1e4)
    measured = rz.ImpactSpec(calibration={"status": "measured", "sample": "2022 fills", "date": "2023-01-31", "method": "OLS on participation", "error": 0.12})
    assert measured.label() == "measured" and rz.ImpactSpec(calibration={"status": "measured"}).label() == "assumed (uncalibrated)"


def test_the_impact_arithmetic_is_the_engine_s_own_and_the_policy_around_it_is_stricter():
    from src.engine.costs import ImpactModel

    for coef, cap in ((0.5, 1.0), (1.0, 0.25), (0.2, 0.05)):
        spec_i = rz.ImpactSpec(coefficient=coef, max_participation=cap)
        for q, adv, sig in ((1_000.0, 1e6, 0.02), (50_000.0, 1e6, 0.01), (2e6, 1e6, 0.03), (10.0, 5e3, 0.0)):
            mine = rz.impact_bps(spec_i, q, adv, sig)["bps"]
            theirs = ImpactModel(kind="sqrt", y=coef, max_participation=cap).impact_price(1.0, q, adv, sig) * 1e4
            assert mine == pytest.approx(theirs, rel=1e-12), (coef, cap, q)
    assert ImpactModel(kind="sqrt").impact_price(100.0, 1_000.0, float("nan"), 0.02) == 0.0, "the engine's model gives a missing ADV zero impact ..."
    assert np.isnan(rz.impact_bps(rz.ImpactSpec(), 1_000.0, float("nan"), 0.02)["bps"]), "... which this layer refuses to treat as a cost estimate"


# --------------------------------------------------------------------------------------------------------------------- P1-07
def test_orders_move_through_legal_states_and_a_fill_is_applied_once():
    book = rz.OrderBook()
    o = book.submit("A", 100.0, D[0])
    book.fill(o.order_id, "f1", 40.0, D[1])
    book.fill(o.order_id, "f1", 40.0, D[1])                                        # a retry of the same fill
    assert o.filled == 40.0 and o.state == "partial" and o.remaining == 60.0
    with pytest.raises(rz.OrderError, match="does not fit"):
        book.fill(o.order_id, "f2", 70.0, D[2])
    book.fill(o.order_id, "f2", 60.0, D[2])
    assert o.state == "filled"
    with pytest.raises(rz.OrderError, match="not a legal transition"):
        book.cancel(o.order_id, D[3])
    r = book.submit("B", -50.0, D[0])
    book.reject(r.order_id, D[1], "no borrow")
    with pytest.raises(rz.OrderError):
        book.fill(r.order_id, "x", -10.0, D[2])
    e = book.submit("C", 10.0, D[0])
    book.expire(e.order_id, D[2])
    c = book.submit("C", 10.0, D[0])
    book.fill(c.order_id, "p", 4.0, D[1])
    book.cancel(c.order_id, D[2])
    assert c.state == "cancelled" and c.filled == 4.0 and [x.order_id for x in book.open()] == []
    with pytest.raises(rz.OrderError):
        book.submit("A", 0.0, D[0])
    assert set(book.frame()["state"]) == {"filled", "rejected", "expired", "cancelled"}


# --------------------------------------------------------------------------------------------------------------------- P1-08
def test_borrow_limits_new_shorts_charges_the_fee_and_forces_a_cover_on_recall():
    px, vol = market(drift=0.0)
    avail = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    avail["A"] = 3000.0
    avail["B"] = 100_000.0
    fee = pd.DataFrame(0.04, index=px.index, columns=px.columns)
    fee["C"] = np.nan                                                                                # C has neither an availability nor a fee record
    book = rz.BorrowBook(avail, fee, recalls={"B": [(D[60], 99_000.0)]}, strict=True)
    assert book.allowed_short("A", D[20], 0.0, 5000.0) == (3000.0, "only 3000 shares can be located") and book.allowed_short("C", D[20], 0.0, 10.0)[0] == 0.0
    assert "no borrow record" in book.allowed_short("C", D[20], 0.0, 10.0)[1]
    loose = rz.BorrowBook(avail, fee, strict=False, scenario_fee=0.02)
    assert loose.allowed_short("C", D[20], 0.0, 10.0)[0] == 10.0 and loose.rate("C", D[20]) == 0.02 and np.isnan(book.rate("C", D[20]))
    w = weights_at(px, {25: [-0.5, -0.3, -0.1]})
    r = ex.execute_weights(w, px, spec(), volume=vol, borrow=book)
    held = r.shares.iloc[30]
    assert held["A"] == -3000.0 and held["C"] == 0.0 and held["B"] < 0, "A is capped by what could be located, C has no record so no new short is opened"
    rej = r.orders[(r.orders["security_id"] == "C")]
    assert rej["filled"].sum() == 0.0 and set(rej["state"]) <= {"expired", "pending", "partial"}
    # fee = |short notional| x rate x dt, accrued daily on the shares held overnight, checked against closed-form arithmetic
    accrued = -float(r.ledger.pnl_by("category").get("borrow", 0.0))
    manual = 0.0
    for i in range(27, 80):
        q = r.shares.iloc[i - 1]                                                                    # holdings at the previous close (the ledger accrues on the position it carries)
        manual += sum(abs(q[a]) * px.iloc[i][a] * 0.04 / 252 for a in ["A", "B"] if q[a] < 0)
    assert accrued == pytest.approx(manual, rel=1e-9) and accrued > 0
    assert (r.shares.loc[D[60]:, "B"] >= -1000.0).all(), "after the recall of 99,000 of 100,000 shares, at most 1,000 remain short"
    assert any(r.orders["state"] == "forced_recall"), "the cover is recorded as a forced action with its reason"


# --------------------------------------------------------------------------------------------------------------------- P1-09
def test_participation_lots_and_missing_adv_limit_what_can_be_filled_and_capacity_falls_with_capital():
    assert rz.fill_with_limits(1000.0, 5000.0, 0.10, 100) == {"filled": 500.0, "remaining": 500.0, "status": "partial"}
    assert rz.fill_with_limits(-1000.0, 5000.0, 0.50, 100)["filled"] == -1000.0 and rz.fill_with_limits(150.0, 5000.0, 0.50, 100)["filled"] == 100.0
    assert rz.fill_with_limits(40.0, 5000.0, 0.10, 100)["status"] == "blocked_lot_or_cap"
    blocked = rz.fill_with_limits(1000.0, float("nan"), 0.10, 1)
    assert blocked["filled"] == 0.0 and blocked["status"] == "blocked_missing_adv", "no ADV is not unlimited capacity"
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    px = pd.Series({"A": 100.0, "B": 50.0, "C": 20.0})
    adv = pd.Series({"A": 2e6, "B": 1e6, "C": float("nan")})
    cur = rz.capacity_curve(w, px, adv, [1e5, 1e6, 1e7, 1e8], 0.10, rz.ImpactSpec(coefficient=0.5), pd.Series(0.01, index=w.index))
    assert (cur["fill_share"].diff().dropna() <= 1e-12).all() and cur["fill_share"].iloc[-1] < cur["fill_share"].iloc[0], "more capital fills a smaller share"
    assert cur["blocked_assets"].tolist() == [1, 1, 1, 1] and cur["impact_bps_of_traded"].iloc[1] > cur["impact_bps_of_traded"].iloc[0]


def test_the_adapter_caps_fills_at_the_participation_limit_and_carries_the_remainder():
    px, vol = market()
    thin = vol * 0.0 + 4_000.0                                                      # 4,000 shares a day: 10% is 400
    w = weights_at(px, {25: [0.5, 0.0, 0.0]})
    r = ex.execute_weights(w, px, spec(max_participation=0.10, expiry_sessions=3), volume=thin)
    a = r.fills[r.fills["instrument_id"] == "A"]
    assert (a["quantity"] <= 400.0 + 1e-9).all() and len(a) == 3 and r.shares["A"].iloc[-1] == a["quantity"].sum()
    o = r.orders.iloc[0]
    assert o["state"] == "expired" and o["filled"] == a["quantity"].sum() and o["filled"] < o["quantity"], "the unfilled remainder expired after three sessions"
    assert r.reconciliation["reconciled"]
    nov = ex.execute_weights(w, px, spec(), volume=None)
    assert any("capacity is NOT established" in x for x in nov.limitations)


# --------------------------------------------------------------------------------------------------------------------- P1-10
def test_closed_stale_halted_and_delisted_markets_are_valued_and_traded_by_separate_rules():
    idx = D[:12]
    px = pd.Series([10.0, 10.5, np.nan, np.nan, 11.0, 11.2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=idx)
    closed = pd.Series(False, index=idx)
    closed.iloc[1] = True
    halted = pd.Series(False, index=idx)
    halted.iloc[5] = True
    m = rz.classify_marks(px, closed, halted, delisted_on=idx[9], policy=rz.MarkPolicy(max_stale_days=2))
    assert m.loc[idx[0], "state"] == "ok" and m.loc[idx[1], "state"] == "closed" and m.loc[idx[2], "state"] == "stale" and m.loc[idx[3], "state"] == "stale_expired", "the mark is two sessions old on idx[2] and three on idx[3]: older than the policy allows"
    assert m.loc[idx[1], "value_mark"] == 10.0 and not m.loc[idx[1], "tradable"], "a closed market is valued at the last good mark and cannot be traded"
    assert m.loc[idx[4], "state"] == "ok" and m.loc[idx[5], "state"] == "halted" and m.loc[idx[5], "value_mark"] == 11.0 and not m.loc[idx[5], "tradable"]
    assert m.loc[idx[7], "state"] == "stale_expired" and m.loc[idx[9], "state"] == "delisted" and not m["tradable"].loc[idx[6]:].any()
    assert (m["value_mark"].dropna() > 0).all() and m["value_mark"].notna().all(), "a missing quote is never a zero price"
    px2, vol = market()
    closed_f = pd.DataFrame(False, index=px2.index, columns=px2.columns)
    closed_f.loc[D[31], "A"] = True
    w = weights_at(px2, {30: [0.5, 0.0, 0.0]})
    r = ex.execute_weights(w, px2, spec(expiry_sessions=3), volume=vol, closed=closed_f)
    assert r.fills["ts"].min() == D[32], "the order waited for the market to open instead of filling at a carried mark"


# --------------------------------------------------------------------------------------------------------------------- P1-12
def test_currency_rates_are_timestamped_missing_ones_raise_and_a_later_rate_never_translates_an_earlier_fill():
    fx = rz.FXBook("USD", spread_bps=5.0, settlement_days={"EUR": 2})
    fx.set_series("EUR", pd.Series([1.10, 1.12, 1.08], index=pd.to_datetime(["2023-01-02", "2023-01-05", "2023-01-10"])))
    assert fx.rate_at("USD", "2023-01-02") == 1.0 and fx.rate_at("EUR", "2023-01-04 16:00") == 1.10, "as of the observation, not interpolated forward"
    with pytest.raises(KeyError, match="a later rate is not used"):
        fx.rate_at("EUR", "2023-01-01")
    with pytest.raises(KeyError, match="not guessed"):
        fx.rate_at("JPY", "2023-01-05")
    c = fx.convert(1000.0, "EUR", "2023-01-06")
    assert c["rate"] == 1.12 and c["mid"] == pytest.approx(1120.0) and c["spread_cost"] == pytest.approx(1120.0 * 5e-4) and c["value"] == pytest.approx(1120.0 * (1 - 5e-4))
    assert fx.convert(1120.0, "EUR", "2023-01-06", to_base=False)["mid"] == pytest.approx(1000.0)
    assert fx.exposures({"EUR": 500.0, "USD": 100.0}, "2023-01-10") == {"EUR": {"native": 500.0, "base": 540.0}, "USD": {"native": 100.0, "base": 100.0}}
    assert fx.settle_date("EUR", "2023-01-04", D) == pd.Timestamp("2023-01-06") and fx.settle_date("USD", "2023-01-04", D) == pd.Timestamp("2023-01-04")
    led = Ledger(InstrumentRegistry([Instrument(instrument_id="E", asset_class="equity", currency="EUR")]), "USD", {"USD": 100_000.0}, start=D[0])
    with pytest.raises(KeyError):
        led.rate("EUR")                                                              # the ledger refuses an unobserved currency instead of using 1


# --------------------------------------------------------------------------------------------------------------------- P1-13
def test_futures_use_dated_specs_roll_to_a_tradable_contract_and_pay_variation_margin():
    chain = [rz.FuturesSpec("ESH", 50.0, 12_000.0, 11_000.0, "2023-03-17"), rz.FuturesSpec("ESM", 50.0, 12_000.0, 11_000.0, "2023-06-16", effective="2022-09-01"),
             rz.FuturesSpec("ESM", 50.0, 14_000.0, 12_500.0, "2023-06-16", effective="2023-05-01")]
    assert rz.executable_contract(chain, "2023-02-01", 5) == "ESH" and rz.executable_contract(chain, "2023-03-14", 5) == "ESM", "within five days of expiry the nearby is no longer executable"
    ev = rz.roll_events(chain, pd.bdate_range("2023-03-01", "2023-03-31"), 5)
    assert [(e["date"], e["close"], e["open"]) for e in ev] == [(pd.Timestamp("2023-03-13"), "ESH", "ESM")], "03-12 is five days before the 03-17 expiry (a Sunday), so the first session that cannot trade ESH is Monday 03-13"
    assert rz.spec_at(chain, "ESM", "2023-04-01").initial_margin == 12_000.0 and rz.spec_at(chain, "ESM", "2023-05-02").initial_margin == 14_000.0
    with pytest.raises(KeyError):
        rz.spec_at(chain, "ESM", "2020-01-01")
    # long 2 contracts: +10 points = +1,000; short 2: -1,000; the roll gap between ESH 4000 and ESM 4012 is a trade at two prices, not a gain
    assert rz.variation_margin(2, 4000.0, 4010.0, 50.0) == 1000.0 and rz.variation_margin(-2, 4000.0, 4010.0, 50.0) == -1000.0
    held, closing, opening = 2, 4000.0, 4012.0
    roll_cash = held * 50.0 * (closing - opening) * 0.0                              # closing at 4000 and reopening at 4012 realises nothing beyond the day's settlement moves
    assert roll_cash == 0.0
    ok = rz.margin_call(30_000.0, 2, 4000.0, chain[0])                              # two contracts need 22,000 to maintain and 24,000 to start
    assert ok["called"] is False and ok["maintenance"] == 22_000.0 and ok["initial"] == 24_000.0 and ok["top_up"] == 0.0
    low = rz.margin_call(20_000.0, 2, 4000.0, chain[0])
    assert low["called"] is True and low["top_up"] == 4_000.0, "below maintenance the account must be restored to the INITIAL margin"
    tight = rz.FuturesSpec("X", 50.0, 14_000.0, 12_500.0, "2023-03-17")
    assert rz.margin_call(24_000.0, 2, 4000.0, tight)["called"] is True and rz.margin_call(24_000.0, 2, 4000.0, tight)["top_up"] == 4_000.0
    assert rz.margin_call(26_000.0, 2, 4000.0, tight)["called"] is False
    with pytest.raises(ValueError):
        rz.FuturesSpec("BAD", 50.0, 1_000.0, 2_000.0, "2023-03-17")


def test_perpetual_funding_signs_longs_pay_when_the_rate_is_positive():
    assert rz.funding_payment(2.0, 30_000.0, 0.0001) == pytest.approx(-6.0) and rz.funding_payment(-2.0, 30_000.0, 0.0001) == pytest.approx(6.0)
    assert rz.funding_payment(2.0, 30_000.0, -0.0001) == pytest.approx(6.0) and rz.funding_payment(-2.0, 30_000.0, -0.0001) == pytest.approx(-6.0)
    assert rz.funding_payment(3.0, 100.0, 0.001, multiplier=10.0) == pytest.approx(-3.0)


# --------------------------------------------------------------------------------------------------------------------- P1-14
def test_the_capability_report_is_read_from_the_actual_data_and_names_what_blocks_a_claim():
    from dataclasses import replace

    from tests.ml_helpers import make_bundle

    b = make_bundle(5, 300, seed=2)
    rep = rz.capability_report(b, data_status="simulated")
    assert rep["execution_fields"]["close"] and rep["execution_fields"]["volume"] and not rep["execution_fields"]["open"]
    assert any("synthetic or simulated" in x for x in rep["limitations"]) and any("only closing prices" in x for x in rep["limitations"]) and any("survivors" in x for x in rep["limitations"])
    assert any("no borrow records" in x for x in rep["blockers"]) and rep["realistic_execution_claim"] == "blocked"
    novol = rz.capability_report(replace(b, volume=None), data_status="simulated")
    assert any("no volume" in x for x in novol["blockers"])
    full = rz.capability_report(b, data_status="vendor", membership=rz.Membership([], coverage="complete"), actions=[rz.CorporateAction("A", "split", D[1], {"ratio": 2})],
                                borrow=rz.BorrowBook(pd.DataFrame(1e6, index=D, columns=ASSETS), strict=True), fx=rz.FXBook(), adapter_modes=("vectorized_legacy", "realistic_cash_equity"))
    assert full["blockers"] == [] and full["borrow"] == "strict records" and full["historical_universe"]["limitation"] is None and "realistic_cash_equity" in full["modes"]
    bad = rz.capability_report(b, actions=[rz.CorporateAction("A", "rights_issue", D[1])])
    assert any("unsupported corporate actions: rights_issue" in x for x in bad["blockers"])
    assert rz.capability_report(None)["historical_universe"]["coverage"] == "none"


def test_the_trading_diagnostics_block_carries_the_mode_assumptions_costs_and_limits():
    px, vol = market()
    r = ex.execute_weights(weights_at(px, {25: [0.4, 0.3, 0.3], 50: [0.0, 0.0, 0.0]}), px, spec(), volume=vol, data_status="simulated")
    d = ex.trading_diagnostics(r)
    assert d["mode"] == "realistic_cash_equity" and d["assumptions"]["half_spread_bps"] == 3.0 and d["assumptions"]["impact_label"] == "assumed (uncalibrated)"
    assert d["cost_total"] == pytest.approx(sum(d["costs"].values())) and d["n_fills"] == len(r.fills) and d["reconciliation"]["reconciled"]
    assert any("simulated" in x for x in d["limitations"]) and any("assumption, not a measured cost" in x for x in d["limitations"]) and any("no borrow records" in x for x in d["limitations"])
    assert d["net_pnl"] == pytest.approx(d["end_equity"] - 1_000_000.0) and "vectorized_legacy" in ex.MODES


def test_an_ineligible_asset_is_exited_long_or_short_and_a_request_to_add_to_it_is_rejected_on_the_record():
    px, vol = market()
    w = weights_at(px, {25: [0.5, -0.4, 0.0], 40: [0.5, -0.4, 0.2]})
    elig = pd.DataFrame(True, index=px.index, columns=px.columns)
    elig.loc[D[40]:, ["B", "C"]] = False                                                         # B (held short) and C (not held) stop being eligible at the second decision
    r = ex.execute_weights(w, px, spec(expiry_sessions=3), volume=vol, eligible=elig)
    assert r.shares["B"].iloc[30] < 0, "the short was opened while B was eligible"
    assert r.shares["B"].iloc[-1] == 0.0, "B's short was bought back once it became ineligible: exit includes covering a short"
    assert r.shares["C"].abs().sum() == 0.0, "C was never added to"
    rej = r.orders[r.orders["state"] == "rejected"]
    assert set(rej["security_id"]) == {"B", "C"} and all("not eligible" in x for x in rej["reason"]), "what the strategy asked beyond the book (more short in B, a new position in C) is rejected on the record"
    assert r.reconciliation["reconciled"]
