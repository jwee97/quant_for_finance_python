"""What a strategy says to buy and sell: its holdings on the last day of the data, the orders of its latest rebalance in dollars and shares, when it traded, and the marks on each ticker's price."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.webapp.payload import actions_section

DAYS = pd.bdate_range("2026-01-05", periods=8)
PRICES = pd.DataFrame({"A": [10.0, 11, 12, 13, 14, 15, 16, 20], "B": [50.0, 50, 49, 48, 47, 46, 45, 40], "C": [5.0] * 8}, index=DAYS)


def frames(rows):
    w = pd.DataFrame(rows, index=DAYS, columns=["A", "B", "C"], dtype=float)
    return w, w.diff().fillna(w.iloc[0])


def test_holdings_and_the_latest_orders_are_in_dollars_and_shares_at_the_right_prices():
    w, traded = frames([[0.5, 0.0, 0.0]] * 3 + [[0.5, 0.001, 0.0]] + [[0.2, -0.3, 0.0]] * 4)      # buy A on day 0; a drift of 0.1% in B on day 3 is not a decision; on day 4 sell some A and short B
    a = actions_section(w, traded, PRICES, 2000.0)
    json.dumps(a, allow_nan=False)
    assert a["as_of"] == str(DAYS[-1].date()) and a["value"] == 2000.0 and a["trade_days"] == 2 and a["typical_gap_days"] == 4.0
    assert [(h["ticker"], h["side"], h["weight"], h["amount"], h["price"], h["shares"]) for h in a["holdings"]] == [("B", "short", -0.3, -600.0, 40.0, 15.0), ("A", "long", 0.2, 400.0, 20.0, 20.0)]
    latest = a["latest"]
    assert latest["date"] == str(DAYS[4].date()) and latest["days_ago"] == 3
    got = {o["ticker"]: o for o in latest["orders"]}
    assert set(got) == {"A", "B"} and got["A"]["side"] == "sell" and np.isclose(got["A"]["size"], 0.3) and np.isclose(got["A"]["amount"], 600.0) and got["A"]["price"] == 14.0 and np.isclose(got["A"]["shares"], 42.86)
    assert got["B"]["side"] == "sell" and np.isclose(got["B"]["size"], 0.301) and np.isclose(got["B"]["to"], -0.3) and got["B"]["price"] == 47.0


def test_the_marks_sit_on_the_days_and_prices_of_the_trades_and_the_ticker_never_traded_has_no_chart():
    w, traded = frames([[0.5, 0.0, 0.0]] * 4 + [[0.2, -0.3, 0.0]] * 4)
    a = actions_section(w, traded, PRICES, 1000.0)
    by = {m["ticker"]: m for m in a["markers"]}
    assert set(by) == {"A", "B"}, "C was never traded"
    m = by["A"]
    assert m["dates"] == [str(d.date()) for d in DAYS] and m["price"] == list(PRICES["A"]) and (m["buys"], m["sells"]) == (1, 1)
    assert m["buy"] == [10.0] + [None] * 7 and m["sell"] == [None] * 4 + [14.0] + [None] * 3 and m["position"][0] == 0.5 and m["position"][-1] == 0.2
    assert by["B"]["sell"][4] == 47.0 and by["B"]["buy"] == [None] * 8


def test_a_long_history_is_thinned_but_never_loses_a_mark_or_the_last_day():
    days = pd.bdate_range("2015-01-01", periods=3000)
    prices = pd.DataFrame({"A": np.linspace(10, 40, 3000)}, index=days)
    w = pd.DataFrame({"A": 0.0}, index=days)
    w.iloc[1237:2011] = 1.0                                                                       # bought on an odd day, sold on another
    a = actions_section(w, w.diff().fillna(0.0), prices, 1000.0, points=300)
    m = a["markers"][0]
    assert len(m["dates"]) < 400 and m["dates"][-1] == str(days[-1].date()) and str(days[1237].date()) in m["dates"] and str(days[2011].date()) in m["dates"]
    assert sum(v is not None for v in m["buy"]) == 1 and sum(v is not None for v in m["sell"]) == 1
    assert a["holdings"] == [] and a["latest"]["orders"][0]["side"] == "sell" and a["latest"]["orders"][0]["to"] == 0.0


def test_a_strategy_that_never_traded_says_so():
    w = pd.DataFrame({"A": 0.0, "B": 0.0, "C": 0.0}, index=DAYS)
    a = actions_section(w, w * 0.0, PRICES, 1000.0)
    assert a["holdings"] == [] and a["latest"] is None and a["markers"] == [] and a["trade_days"] == 0 and a["typical_gap_days"] is None


def test_an_index_has_amounts_but_no_count_of_units():
    w, traded = frames([[0.5, 0.0, 0.0]] * 4 + [[0.2, -0.3, 0.0]] * 4)
    a = actions_section(w, traded, PRICES, 1000.0, units=False)
    assert a["units"] is False and all(h["shares"] is None for h in a["holdings"]) and all(o["shares"] is None for o in a["latest"]["orders"]) and a["holdings"][0]["amount"] == -300.0
    assert actions_section(w, traded, PRICES, 1000.0)["units"] is True


def test_the_futures_and_crypto_backtests_carry_the_card_in_their_own_terms():
    from src.assets import crypto_analytics as ca
    from src.assets import futures_analytics as fa
    from src.utils.config import load_config

    config = load_config()
    fut = fa.run_futures_strategy(config, "carry_ts", ["CL", "NG", "ES"], as_of="2026-09-18")["actions"]
    assert fut["units"] is False and fut["simulated"] is True and fut["capital"] == 100000.0 and fut["latest"]["orders"] and fut["markers"] and {m["ticker"] for m in fut["markers"]} <= {"CL", "NG", "ES"}
    coin = ca.run_crypto_strategy(config, "spot_momentum", ["BTC", "ETH"])["actions"]
    assert coin["units"] is True and "simulated" not in coin and coin["latest"]["orders"][0]["shares"] is not None
    carry = ca.run_crypto_strategy(config, "funding_carry", ["BTC"])["actions"]
    assert carry["units"] is False, "a carry index is not bought by the coin"


def test_an_option_backtest_lists_contracts_the_open_positions_and_marks_the_underlying():
    from src.assets import options_analytics as oa

    out = oa.run_strategy_backtest("short_put", None, n_days=120, seed=1)
    a = out["actions"]
    json.dumps(a, allow_nan=False)
    assert a["contracts"] is True and a["simulated"] is True and a["units"] is True and a["trade_days"] >= 2 and a["typical_gap_days"] > 0
    first = a["latest"]["orders"][0]
    assert first["ticker"].startswith(("Put ", "Call ")) and " exp " in first["ticker"] and first["side"] in ("buy", "sell") and first["shares"] > 0 and np.isclose(first["amount"], first["shares"] * first["price"] * 100.0)
    m = a["markers"][0]
    assert m["ticker"] == "Underlying" and m["contracts"] is True and m["sells"] >= 1 and sum(v is not None for v in m["sell"]) == m["sells"] and len(m["dates"]) == len(m["price"]) == len(m["position"])
    for h in a["holdings"]:
        assert h["side"] in ("long", "short") and h["ticker"].split(" exp ")[1] > a["as_of"], "only contracts that have not expired are held"
