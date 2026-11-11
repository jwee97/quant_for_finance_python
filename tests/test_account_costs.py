"""What a personal account keeps of a backtest: a fee on every order, a fee for borrowing shares held short and interest on borrowed money, checked against hand calculations, and the request's
``account`` settings checked before a run starts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.webapp.api import ApiError, App
from src.webapp.payload import ACCOUNT_DEFAULTS, account_section

DAYS = pd.bdate_range("2026-01-05", periods=4)


def frames(weights: list[list[float]], cols=("A", "B")):
    w = pd.DataFrame(weights, index=DAYS, columns=list(cols), dtype=float)
    return w, w.diff().fillna(w.iloc[0])


def test_with_nothing_charged_the_account_is_the_backtest():
    w, traded = frames([[0.5, 0.5]] * 4)
    net = pd.Series([0.0, 0.01, -0.02, 0.03], index=DAYS)
    a = account_section(net, w, traded, 1000.0, fee_per_order=0.0, borrow_pct=0.0, margin_pct=0.0)
    assert a["end_value"] == pytest.approx(1000 * 1.01 * 0.98 * 1.03) == pytest.approx(a["backtest_end_value"]) and a["kept"] == pytest.approx(0.0) and a["paid"] == {"fees": 0.0, "borrow": 0.0, "margin": 0.0}
    assert a["ruined"] is None and a["orders"] == 2 and len(a["values"]) == 4


def test_an_order_fee_is_paid_per_order_and_weighs_more_on_a_small_account():
    w, traded = frames([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [1.0, 0.0]])                       # two orders to open, two to switch
    net = pd.Series([0.0, 0.0, 0.0, 0.0], index=DAYS)
    small, large = (account_section(net, w, traded, c, fee_per_order=5.0, borrow_pct=0.0, margin_pct=0.0) for c in (1000.0, 100000.0))
    assert small["orders"] == 4 and small["paid"]["fees"] == 20.0 and small["end_value"] == pytest.approx(980.0) and large["end_value"] == pytest.approx(99980.0)
    assert small["annual_return"] < large["annual_return"] < 0


def test_shorts_pay_a_borrow_fee_and_leverage_pays_interest_on_the_position_held_the_day_before():
    w, traded = frames([[1.5, -0.5]] * 4)                                                        # long 150%, short 50%: half the account is borrowed money, half is borrowed shares
    net = pd.Series([0.0, 0.0, 0.0, 0.0], index=DAYS)
    a = account_section(net, w, traded, 1000.0, fee_per_order=0.0, borrow_pct=2.52, margin_pct=5.04)
    b, m = 0.5 * 0.0252 / 252, 0.5 * 0.0504 / 252                                               # a day's charge as a fraction of the account
    by_hand = 1000.0 * (1 - b - m) ** 3                                                         # the first day holds nothing yet (the position is bought at its close)
    assert a["end_value"] == pytest.approx(by_hand) and a["paid"]["borrow"] == pytest.approx(1000 * b * (1 + (1 - b - m) + (1 - b - m) ** 2))
    assert a["average_short"] == pytest.approx(0.375) and a["average_borrowed"] == pytest.approx(0.375)
    long_only = account_section(net, *frames([[0.6, 0.4]] * 4), 1000.0, fee_per_order=0.0, borrow_pct=50.0, margin_pct=50.0)
    assert long_only["end_value"] == 1000.0, "fully invested and long only: nothing is borrowed"


def test_an_account_that_runs_out_of_money_stops_and_says_when():
    w, traded = frames([[0.5, 0.5], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
    a = account_section(pd.Series([0.0] * 4, index=DAYS), w, traded, 1000.0, fee_per_order=300.0, borrow_pct=0.0, margin_pct=0.0)
    assert a["ruined"] == str(DAYS[1].date()) and a["end_value"] == 0.0 and a["sharpe"] is None and a["annual_return"] == -1.0 and a["values"][-1] == 0.0


def test_the_defaults_are_a_commission_free_broker_with_ordinary_borrow_and_margin_rates():
    assert ACCOUNT_DEFAULTS == {"fee_per_order": 0.0, "borrow_pct": 1.0, "margin_pct": 6.0}
    assert App._account({}) == ACCOUNT_DEFAULTS and App._account({"account": {"fee_per_order": "1.5", "borrow_pct": ""}}) == {"fee_per_order": 1.5, "borrow_pct": 1.0, "margin_pct": 6.0}
    for bad, why in [({"account": "none"}, "must be an object"), ({"account": {"fee_per_order": -1}}, "between 0 and 100"), ({"account": {"margin_pct": 80}}, "between 0 and 50"), ({"account": {"borrow_pct": "x"}}, "must be a number"),
                     ({"account": {"borrow_pct": float("nan")}}, "between")]:
        with pytest.raises(ApiError, match=why):
            App._account(bad)
    assert np.isfinite(list(ACCOUNT_DEFAULTS.values())).all()
