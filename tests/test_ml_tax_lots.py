"""Release 8, P3-09: optional tax-lot and after-tax accounting under a supplied, versioned policy.

Every figure is worked by hand on small ledgers: which lot a sale takes under FIFO, LIFO and HIFO, how fees move basis and proceeds, where a gain turns long-term, how losses net and carry, what a payment
lag does to the wealth path, and what selling the open lots would add. A missing, incomplete or unimplementable policy leaves the result unavailable; a ledger the supported events cannot represent blocks
it; and the pre-tax figures are the same before and after."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.ml import tax_lots as tl


def policy(**over):
    base = {"schema": "tax-policy-v1", "jurisdiction": "TESTLAND", "policy_version": "2025.1", "investor": "individual", "rule_family": "lot_gain_loss_v1", "currency": "USD",
            "provenance": {"supplied_by": "J. Tester", "source": "Test Revenue Act s.12 (fictional)", "verified_by": "A. Reviewer", "verified_on": "2025-01-15"},
            "lot_method": "fifo", "fee_treatment": "capitalise", "long_term_min_days": 366, "rates": {"short_term": 0.4, "long_term": 0.2}, "loss_offset": "within_term", "loss_carryforward": False,
            "payment_lag_days": 0}
    base.update(over)
    return base


def pol(**over) -> tl.TaxPolicy:
    out = tl.parse_policy(policy(**over))
    assert out["status"] == "applied", out["reason"]
    return out["policy"]


def led(rows):
    return pd.DataFrame(rows, columns=["date", "asset", "quantity", "price", "fee"])


def daily(start="2024-01-02", end="2025-03-31", jump=("2024-06-03", 0.10)):
    r = pd.Series(0.0, index=pd.bdate_range(start, end))
    if jump:
        r.loc[pd.Timestamp(jump[0])] = jump[1]
    return r


# ------------------------------------------------------------------------------------------------------------------ the policy
def test_a_policy_must_be_supplied_complete_and_implementable_or_the_result_is_unavailable():
    for nothing in (None, {}, "US", []):
        out = tl.parse_policy(nothing)
        assert out["status"] == "unavailable" and out["policy"] is None and "no tax policy was supplied" in out["reason"]
    ok = tl.parse_policy(policy())
    assert ok["status"] == "applied" and ok["policy"].verification == "verified"
    h = ok["policy"].hash()
    assert len(h) == 64 and tl.parse_policy(policy())["policy"].hash() == h
    assert tl.parse_policy(policy(rates={"short_term": 0.41, "long_term": 0.2}))["policy"].hash() != h, "the version of the rules is part of what is hashed"
    assert tl.parse_policy(policy(policy_version="2025.2"))["policy"].hash() != h
    unverified = tl.parse_policy(policy(provenance={"supplied_by": "J. Tester", "source": "Test Revenue Act s.12"}))
    assert unverified["status"] == "applied" and unverified["policy"].verification == "supplied_unverified"

    def bad(expect, **over):
        out = tl.parse_policy(policy(**over))
        assert out["status"] == "unavailable" and out["policy"] is None and expect in out["reason"], (over, out["reason"])
        return out

    bad("provenance.supplied_by is required", provenance={"source": "doc"})
    bad("provenance.source is required", provenance={"supplied_by": "me"})
    bad("go together", provenance={"supplied_by": "me", "source": "doc", "verified_by": "A"})
    bad("unknown policy field", wash_sale_days=30)
    bad("lot_method is one of", lot_method="random")
    bad("fee_treatment is one of", fee_treatment="ignore")
    bad("loss_offset is one of", loss_offset="everything")
    bad("rates.short_term is a rate", rates={"short_term": 1.5, "long_term": 0.2})
    bad("rates.long_term is a rate", rates={"short_term": 0.4, "long_term": True})
    bad("rates has exactly", rates={"short_term": 0.4})
    bad("needs loss_offset to allow", loss_offset="none", loss_carryforward=True)
    bad("long_term_min_days is a whole number", long_term_min_days=0)
    bad("long_term_min_days is a whole number", long_term_min_days=1.5)
    bad("long_term_min_days is a whole number", long_term_min_days=True)
    bad("payment_lag_days is a whole number", payment_lag_days=-1)
    bad("policy schema", schema="tax-policy-v0")
    missing = policy()
    del missing["lot_method"]
    assert "lacks 'lot_method'" in tl.parse_policy(missing)["reason"]
    # rules this module does not implement keep the jurisdiction unavailable instead of being approximated
    w = bad("does not implement", features=["wash_sale"])
    assert w["unsupported"] == ["rule 'wash_sale'"] and "stays unavailable rather than being approximated" in w["reason"]
    assert bad("does not implement", rule_family="progressive_brackets_v2")["unsupported"][0].startswith("rule family")
    assert bad("does not implement", period="fiscal_april")["unsupported"][0].startswith("tax period")
    for name in ("short_sales", "dividend_tax", "currency_conversion", "loss_carryback"):
        assert tl.parse_policy(policy(features=[name]))["status"] == "unavailable"


def test_an_unavailable_policy_produces_no_after_tax_figures_at_all():
    out = tl.run(None, led([("2024-01-02", "A", 10, 100.0, 0.0), ("2024-02-01", "A", -10, 110.0, 0.0)]), daily(), 100_000.0)
    assert out["status"] == "unavailable" and out["after_tax"] is None and out["policy"] is None and out["not_tax_advice"] is True
    assert "wash sales" in out["events_not_supported"] and any("FIFO" in e for e in out["events_supported"])
    out2 = tl.run(policy(features=["dividend_tax"]), led([]), daily(), 100_000.0)
    assert out2["status"] == "unavailable" and out2["unsupported"] == ["rule 'dividend_tax'"]
    json.dumps(out2, allow_nan=False)


# ------------------------------------------------------------------------------------------------------------------ lots
LOTS = [("2024-01-02", "A", 10, 100.0, 0.0), ("2024-03-01", "A", 10, 120.0, 0.0), ("2024-06-03", "A", 10, 80.0, 0.0), ("2024-09-02", "A", -15, 110.0, 0.0)]


@pytest.mark.parametrize("method, taken, left", [
    ("fifo", [(10, 100.0), (5, -50.0)], [("2024-03-01", 5.0), ("2024-06-03", 10.0)]),
    ("lifo", [(10, 300.0), (5, -50.0)], [("2024-01-02", 10.0), ("2024-03-01", 5.0)]),
    ("hifo", [(10, -100.0), (5, 50.0)], [("2024-01-02", 5.0), ("2024-06-03", 10.0)]),
])
def test_fifo_lifo_and_hifo_relieve_different_lots_as_worked_by_hand(method, taken, left):
    out = tl.relieve_lots(led(LOTS), pol(lot_method=method))
    assert out["status"] == "applied"
    s = out["slices"]
    assert [float(q) for q in s["quantity"]] == [float(q) for q, _ in taken]
    assert list(s["gain"]) == [pytest.approx(g) for _, g in taken]
    assert s["proceeds"].sum() == pytest.approx(15 * 110.0)
    assert [(str(d.date()), q) for d, q in zip(out["open_lots"]["acquired"], out["open_lots"]["quantity"])] == left
    assert out["open_lots"]["quantity"].sum() == pytest.approx(30 - 15), "shares are conserved: bought less sold is still held"
    assert (s["term"] == "short").all() and (s["period"] == 2024).all()


def test_hifo_ties_go_to_the_older_lot_and_lots_of_different_assets_never_mix():
    rows = [("2024-01-02", "A", 5, 100.0, 0.0), ("2024-01-03", "A", 5, 100.0, 0.0), ("2024-01-04", "B", 5, 50.0, 0.0), ("2024-02-01", "A", -5, 110.0, 0.0), ("2024-02-02", "B", -2, 40.0, 0.0)]
    out = tl.relieve_lots(led(rows), pol(lot_method="hifo"))
    s = out["slices"]
    assert list(s["asset"]) == ["A", "B"] and str(s["acquired"].iloc[0].date()) == "2024-01-02" and list(s["gain"]) == [pytest.approx(50.0), pytest.approx(-20.0)]
    left = out["open_lots"]
    assert dict(zip(left["asset"], left["quantity"])) == {"A": 5.0, "B": 3.0} and str(left.loc[left["asset"] == "A", "acquired"].iloc[0].date()) == "2024-01-03"


def test_fees_move_basis_and_proceeds_or_are_not_deductible_as_the_policy_says():
    rows = [("2024-01-02", "A", 10, 100.0, 5.0), ("2024-02-01", "A", -10, 110.0, 3.0)]
    cap = tl.relieve_lots(led(rows), pol(fee_treatment="capitalise"))
    assert (1100 - 3) - (1000 + 5) == 92 and cap["slices"]["gain"].iloc[0] == pytest.approx(92.0)
    assert cap["fees"] == {"buy": 5.0, "sell": 3.0}
    nd = tl.relieve_lots(led(rows), pol(fee_treatment="not_deductible"))
    assert nd["slices"]["gain"].iloc[0] == pytest.approx(100.0), "fees that the policy does not allow leave the gain on the gross prices"
    part = [("2024-01-02", "A", 10, 100.0, 5.0), ("2024-02-01", "A", -4, 110.0, 2.0)]
    assert (440 - 2) - 4 * 100.5 == 36.0 and tl.relieve_lots(led(part), pol())["slices"]["gain"].iloc[0] == pytest.approx(36.0)
    assert tl.relieve_lots(led(part), pol(fee_treatment="not_deductible"))["slices"]["gain"].iloc[0] == pytest.approx(40.0)
    left = tl.relieve_lots(led(part), pol())["open_lots"]
    assert left["unit_basis"].iloc[0] == pytest.approx(100.5) and left["quantity"].iloc[0] == pytest.approx(6.0) and left["cost"].iloc[0] == pytest.approx(603.0)
    # one sale across two lots: the sale fee is shared by quantity
    two = [("2024-01-02", "A", 5, 100.0, 0.0), ("2024-01-03", "A", 5, 100.0, 0.0), ("2024-02-01", "A", -8, 110.0, 4.0)]
    s = tl.relieve_lots(led(two), pol())["slices"]
    assert list(s["quantity"]) == [5.0, 3.0] and list(s["proceeds"]) == [pytest.approx(550 - 2.5), pytest.approx(330 - 1.5)] and s["proceeds"].sum() == pytest.approx(880 - 4)
    assert list(s["gain"]) == [pytest.approx(47.5), pytest.approx(28.5)]


def test_a_gain_turns_long_term_exactly_at_the_policys_threshold():
    rows = [("2023-01-10", "A", 1, 100.0, 0.0), ("2023-01-10", "B", 1, 100.0, 0.0), ("2024-01-10", "A", -1, 200.0, 0.0), ("2024-01-11", "B", -1, 200.0, 0.0)]
    out = tl.relieve_lots(led(rows), pol(long_term_min_days=366))
    s = out["slices"].set_index("asset")
    assert s.loc["A", "days_held"] == 365 and s.loc["A", "term"] == "short"
    assert s.loc["B", "days_held"] == 366 and s.loc["B", "term"] == "long"
    p = tl.settle_periods(out["slices"], pol(long_term_min_days=366), 2024, 2024).iloc[0]
    assert (p["gains_short"], p["gains_long"], p["tax"]) == (100.0, 100.0, pytest.approx(100 * 0.4 + 100 * 0.2))
    again = tl.relieve_lots(led(rows), pol(long_term_min_days=365))["slices"].set_index("asset")
    assert again.loc["A", "term"] == "long", "the threshold is the policy's, not the module's"


def test_opening_lots_are_matched_before_the_ledgers_own_purchases_under_fifo():
    opening = pd.DataFrame({"asset": ["A"], "quantity": [10.0], "unit_price": [50.0], "acquired": ["2020-01-01"]})
    rows = [("2024-01-02", "A", 10, 100.0, 0.0), ("2024-05-01", "A", -12, 70.0, 0.0)]
    out = tl.relieve_lots(led(rows), pol(), opening)
    s = out["slices"]
    assert list(s["gain"]) == [pytest.approx(200.0), pytest.approx(-60.0)] and list(s["term"]) == ["long", "short"] and bool(s["opening_lot"].iloc[0]) is True
    assert out["open_lots"]["quantity"].iloc[0] == pytest.approx(8.0)


# ------------------------------------------------------------------------------------------------------------------ tax events
def sl(period, term, gain):
    return {"period": period, "term": term, "gain": gain}


def test_loss_netting_and_carry_forward_follow_only_what_the_policy_says():
    year = pd.DataFrame([sl(2024, "short", 1000.0), sl(2024, "short", -300.0), sl(2024, "long", -500.0)])
    none = tl.settle_periods(year, pol(loss_offset="none"), 2024, 2024).iloc[0]
    assert (none["taxable_short"], none["taxable_long"], none["tax"]) == (1000.0, 0.0, pytest.approx(400.0)), "no loss relief: every gain is taxed gross"
    within = tl.settle_periods(year, pol(loss_offset="within_term"), 2024, 2024).iloc[0]
    assert (within["taxable_short"], within["taxable_long"], within["tax"], within["carry_out_long"]) == (700.0, 0.0, pytest.approx(280.0), 0.0), "the long-term loss is lost without a carry-forward"
    across = tl.settle_periods(year, pol(loss_offset="across_terms"), 2024, 2024).iloc[0]
    assert (across["offset_across_terms"], across["taxable_short"], across["tax"]) == (500.0, 200.0, pytest.approx(80.0))
    # carry-forward: the unused long-term loss is applied against later long-term gains, and passes through a year with no sales
    carry = pd.DataFrame([sl(2024, "long", -500.0), sl(2026, "long", 300.0)])
    p = tl.settle_periods(carry, pol(loss_offset="within_term", loss_carryforward=True), 2024, 2026)
    assert list(p["carry_out_long"]) == [500.0, 500.0, 200.0] and list(p["taxable_long"]) == [0.0, 0.0, 0.0] and p["tax"].sum() == 0.0
    more = pd.DataFrame([sl(2024, "long", -500.0), sl(2026, "long", 800.0)])
    q = tl.settle_periods(more, pol(loss_offset="within_term", loss_carryforward=True), 2024, 2026)
    assert list(q["carry_in_long"]) == [0.0, 500.0, 500.0] and q["tax"].iloc[2] == pytest.approx((800 - 500) * 0.2) and q["carry_out_long"].iloc[2] == 0.0
    off = tl.settle_periods(more, pol(loss_offset="within_term", loss_carryforward=False), 2024, 2026)
    assert off["tax"].iloc[2] == pytest.approx(800 * 0.2), "without the carry-forward the same loss would have been worth nothing"
    # across terms with a carry: a long-term loss larger than the short-term gain offsets it, and the rest is carried
    big = pd.DataFrame([sl(2024, "short", 700.0), sl(2024, "long", -1000.0), sl(2025, "long", 400.0)])
    c = tl.settle_periods(big, pol(loss_offset="across_terms", loss_carryforward=True), 2024, 2025)
    assert (c["offset_across_terms"].iloc[0], c["tax"].iloc[0], c["carry_out_long"].iloc[0]) == (700.0, 0.0, 300.0)
    assert c["tax"].iloc[1] == pytest.approx((400 - 300) * 0.2)
    nocarry = tl.settle_periods(big, pol(loss_offset="across_terms", loss_carryforward=False), 2024, 2025)
    assert nocarry["tax"].iloc[1] == pytest.approx(400 * 0.2)
    # the other direction: a short-term loss offsets a long-term gain
    rev = pd.DataFrame([sl(2024, "short", -300.0), sl(2024, "long", 1000.0)])
    r = tl.settle_periods(rev, pol(loss_offset="across_terms"), 2024, 2024).iloc[0]
    assert (r["taxable_long"], r["tax"]) == (700.0, pytest.approx(140.0))
    w = tl.settle_periods(rev, pol(loss_offset="within_term"), 2024, 2024).iloc[0]
    assert (w["taxable_long"], w["tax"]) == (1000.0, pytest.approx(200.0))


def test_a_ledger_the_supported_events_cannot_represent_is_blocked_with_its_reason():
    short = tl.relieve_lots(led([("2024-01-02", "A", 10, 100.0, 0.0), ("2024-02-01", "A", -12, 110.0, 0.0)]), pol())
    assert short["status"] == "blocked" and "exceeds the 10 held" in short["reason"] and "short sales are not supported" in short["reason"]
    never = tl.relieve_lots(led([("2024-02-01", "B", -1, 110.0, 0.0)]), pol())
    assert never["status"] == "blocked" and "exceeds the 0 held" in never["reason"]
    assert tl.relieve_lots(led([("2024-01-02", "A", 10, 0.0, 0.0)]), pol())["status"] == "blocked"
    assert "not positive" in tl.relieve_lots(led([("2024-01-02", "A", 10, -3.0, 0.0)]), pol())["reason"]
    assert "negative fee" in tl.relieve_lots(led([("2024-01-02", "A", 10, 3.0, -1.0)]), pol())["reason"]
    assert "zero quantity" in tl.relieve_lots(led([("2024-01-02", "A", 0, 3.0, 0.0)]), pol())["reason"]
    assert "no usable date" in tl.relieve_lots(led([("not a date", "A", 1, 3.0, 0.0)]), pol())["reason"]
    assert "not filled in" in tl.relieve_lots(led([("2024-01-02", "A", 1, np.nan, 0.0)]), pol())["reason"]
    assert "lacks the column" in tl.relieve_lots(pd.DataFrame({"date": ["2024-01-02"], "asset": ["A"]}), pol())["reason"]
    assert tl.relieve_lots("not a table", pol())["status"] == "blocked"
    ok = tl.relieve_lots(led([("2024-02-01", "A", -3, 110.0, 0.0), ("2024-01-02", "A", 10, 100.0, 0.0)]), pol())
    assert ok["status"] == "applied" and ok["input_was_date_ordered"] is False and ok["slices"]["gain"].iloc[0] == pytest.approx(30.0), "fills are put in date order, and the report says they were not"


# ------------------------------------------------------------------------------------------------------------------ the whole thing
def test_after_tax_wealth_by_hand_with_payment_lag_and_the_deferred_liability_on_open_lots():
    r = daily()
    rows = [("2024-01-02", "A", 100, 100.0, 0.0), ("2024-12-27", "A", -100, 110.0, 0.0)]
    n = len(r)
    out = tl.run(policy(), led(rows), r, 100_000.0)
    assert out["status"] == "applied" and out["policy"]["verification"] == "verified"
    assert out["pre_tax"]["final_wealth"] == pytest.approx(110_000.0) and out["pre_tax"]["total_return"] == pytest.approx(0.10)
    assert out["pre_tax"]["cagr"] == pytest.approx(1.1 ** (252.0 / n) - 1.0)
    s = out["realised"]["slices"][0]
    assert (s["gain"], s["days_held"], s["term"], s["period"]) == (1000.0, 360, "short", 2024) and out["realised"]["net_gain"] == 1000.0
    p24 = out["periods"][0]
    assert (p24["period"], p24["gains_short"], p24["tax"]) == (2024, 1000.0, pytest.approx(400.0)) and out["periods"][1]["tax"] == 0.0
    assert out["payments"] == [{"period": 2024, "tax": pytest.approx(400.0), "due": "2024-12-31", "paid_on": "2024-12-31"}]
    assert out["taxes"] == {"total": pytest.approx(400.0), "paid": pytest.approx(400.0), "accrued_unpaid": 0.0, "effective_rate_on_net_realised_gains": pytest.approx(0.4),
                            "deferred_if_liquidated": {"status": "applied", "value": 0.0, "counted_in_wealth": 0.0, "reason": "no open lots"}}
    assert out["after_tax"]["realised"]["final_wealth"] == pytest.approx(109_600.0) and out["after_tax"]["realised"]["total_return"] == pytest.approx(0.096)
    assert out["after_tax"]["if_liquidated"]["final_wealth"] == pytest.approx(109_600.0)
    assert out["tax_drag"]["total_return_points"] == pytest.approx(0.004) and out["tax_drag"]["cagr_points"] > 0
    # the tax falls due after the sample ends: nothing is paid, and the liability is still taken off the wealth
    late = tl.run(policy(payment_lag_days=120), led(rows), r, 100_000.0)
    assert late["taxes"]["paid"] == 0.0 and late["taxes"]["accrued_unpaid"] == pytest.approx(400.0) and late["payments"][0]["paid_on"] is None and late["payments"][0]["due"] == "2025-04-30"
    assert late["after_tax"]["realised"]["final_wealth"] == pytest.approx(109_600.0)
    # an open lot: its unrealised gain is a liability the realised figures leave out, and the liquidation view shows it
    held = led(rows + [("2024-02-01", "B", 50, 100.0, 0.0)])
    nolast = tl.run(policy(), held, r, 100_000.0)
    d0 = nolast["taxes"]["deferred_if_liquidated"]
    assert d0["status"] == "unavailable" and "no closing price for ['B']" in d0["reason"] and nolast["after_tax"]["if_liquidated"]["status"] == "unavailable"
    assert nolast["after_tax"]["realised"]["final_wealth"] == pytest.approx(109_600.0), "what can be computed still is"
    full = tl.run(policy(), held, r, 100_000.0, last_prices={"B": 120.0})
    d = full["taxes"]["deferred_if_liquidated"]
    assert 50 * 20 * 0.2 == 200.0 and d["status"] == "applied" and d["value"] == pytest.approx(200.0), "424 days held: long-term at 20%"
    assert full["after_tax"]["if_liquidated"]["final_wealth"] == pytest.approx(109_600.0 - 200.0)
    assert full["open_lots"][0]["asset"] == "B" and full["open_lots"][0]["quantity"] == 50.0
    # a liquidation that realises a loss is a benefit: shown, not counted
    low = tl.run(policy(), held, r, 100_000.0, last_prices={"B": 90.0})
    assert low["taxes"]["deferred_if_liquidated"]["value"] == pytest.approx(0.0) and low["after_tax"]["if_liquidated"]["final_wealth"] == pytest.approx(109_600.0)
    offset = tl.run(policy(loss_offset="across_terms"), led([("2025-01-06", "A", 100, 100.0, 0.0), ("2025-03-03", "A", -100, 110.0, 0.0), ("2024-02-01", "B", 50, 100.0, 0.0)]), r, 100_000.0,
                    last_prices={"B": 90.0})
    benefit = offset["taxes"]["deferred_if_liquidated"]
    assert offset["taxes"]["total"] == pytest.approx(400.0) and benefit["value"] == pytest.approx(-200.0) and benefit["counted_in_wealth"] == 0.0, "selling the loser would cut the year's tax, which is shown but not counted"
    assert offset["taxes"]["accrued_unpaid"] == pytest.approx(400.0) and offset["after_tax"]["if_liquidated"]["final_wealth"] == pytest.approx(110_000.0 - 400.0)
    json.dumps(full, allow_nan=False)


def test_pre_tax_figures_are_untouched_and_a_zero_rate_policy_changes_nothing():
    r = daily(jump=None)
    r.iloc[10] = 0.02
    r.iloc[200] = -0.01
    r.iloc[300] = 0.03
    before = r.copy()
    digest = hashlib.sha256(np.ascontiguousarray(r.to_numpy(), dtype="float64").tobytes()).hexdigest()
    rows = [("2024-01-02", "A", 100, 100.0, 1.0), ("2024-09-02", "A", -60, 120.0, 1.0), ("2025-02-03", "A", -40, 90.0, 1.0)]
    out = tl.run(policy(), led(rows), r, 50_000.0)
    pd.testing.assert_series_equal(r, before)
    assert out["pre_tax_unchanged"] is True and out["pre_tax_returns_hash"] == digest
    assert out["pre_tax"]["final_wealth"] == pytest.approx(50_000.0 * float(np.prod(1.0 + r.to_numpy())), rel=1e-12)
    free = tl.run(policy(rates={"short_term": 0.0, "long_term": 0.0}), led(rows), r, 50_000.0)
    assert free["taxes"]["total"] == 0.0 and free["after_tax"]["realised"]["final_wealth"] == pytest.approx(free["pre_tax"]["final_wealth"], rel=1e-12)
    assert free["tax_drag"]["total_return_points"] == pytest.approx(0.0, abs=1e-12)
    again = tl.run(policy(), led(rows), r, 50_000.0)
    assert json.dumps(again, sort_keys=True) == json.dumps(out, sort_keys=True), "the same inputs give the same report"
    assert out["after_tax"]["realised"]["final_wealth"] < out["pre_tax"]["final_wealth"] or out["taxes"]["total"] == 0.0
    assert out["fees"] == {"buy": 1.0, "sell": 2.0, "treatment": "capitalise"}
    assert any("unverified" in x for x in tl.run(policy(provenance={"supplied_by": "me", "source": "doc"}), led(rows), r, 50_000.0)["limitations"])
    assert not any("unverified" in x for x in out["limitations"])


def test_run_blocks_an_inconsistent_ledger_or_return_series_instead_of_guessing():
    r = daily()
    rows = [("2024-01-02", "A", 1, 1.0, 0.0), ("2024-02-01", "A", -1, 1_000_000.0, 0.0)]
    huge = tl.run(policy(rates={"short_term": 1.0, "long_term": 1.0}), led(rows), r, 100.0)
    assert huge["status"] == "blocked" and "exceeds the portfolio's value" in huge["reason"] and huge["after_tax"] is None
    early = tl.run(policy(), led([("2023-12-01", "A", 1, 1.0, 0.0)]), r, 1000.0)
    assert early["status"] == "blocked" and "outside the return series" in early["reason"]
    assert tl.run(policy(), led([("2024-01-02", "A", 1, 1.0, 0.0), ("2024-02-01", "A", -2, 1.0, 0.0)]), r, 1000.0)["status"] == "blocked"
    assert tl.run(policy(), led([]), r.iloc[:1], 1000.0)["status"] == "blocked"
    assert tl.run(policy(), led([]), r, 0.0)["status"] == "blocked"
    gap = r.copy()
    gap.iloc[5] = np.nan
    assert tl.run(policy(), led([]), gap, 1000.0)["status"] == "blocked"
    wipe = r.copy()
    wipe.iloc[5] = -1.0
    assert tl.run(policy(), led([]), wipe, 1000.0)["status"] == "blocked"
    quiet = tl.run(policy(), led([]), r, 1000.0)
    assert quiet["status"] == "applied" and quiet["taxes"]["total"] == 0.0 and quiet["realised"]["slices_total"] == 0 and quiet["taxes"]["effective_rate_on_net_realised_gains"] is None
    capped = tl.run(policy(), led([("2024-01-02", "A", 100, 1.0, 0.0)] + [(f"2024-02-{d:02d}", "A", -1, 2.0, 0.0) for d in range(1, 21)]), r, 1000.0, max_slices=5)
    assert capped["realised"]["slices_total"] == 20 and len(capped["realised"]["slices"]) == 5


def test_an_illustrative_ledger_from_a_book_of_weights_and_the_refusal_of_short_books():
    idx = pd.bdate_range("2024-01-02", periods=3)
    w = pd.DataFrame({"A": [0.5, 0.5, 0.0], "B": [0.0, 0.25, 0.25]}, index=idx)
    px = pd.DataFrame({"A": [10.0, 10.0, 10.0], "B": [25.0, 25.0, 25.0]}, index=idx)
    eq = pd.Series(1000.0, index=idx)
    ledger = tl.ledger_from_positions(w, px, eq, fee_bps=10.0)
    got = [(str(d.date()), a, round(q, 9), p, round(f, 9)) for d, a, q, p, f in ledger[["date", "asset", "quantity", "price", "fee"]].itertuples(index=False)]
    assert got == [("2024-01-02", "A", 50.0, 10.0, 0.5), ("2024-01-03", "B", 10.0, 25.0, 0.25), ("2024-01-04", "A", -50.0, 10.0, 0.5)]
    out = tl.relieve_lots(ledger, pol())
    assert out["status"] == "applied" and out["slices"]["gain"].iloc[0] == pytest.approx(-1.0) and out["open_lots"]["asset"].tolist() == ["B"]
    short = w.copy()
    short.iloc[2, 0] = -0.2
    sh = tl.relieve_lots(tl.ledger_from_positions(short, px, eq), pol())
    assert sh["status"] == "blocked" and "short sales are not supported" in sh["reason"]
    with pytest.raises(ValueError, match="prices and equity"):
        tl.ledger_from_positions(w, px.iloc[:2], eq)
