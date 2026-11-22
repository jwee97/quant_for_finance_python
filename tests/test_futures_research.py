"""Term-structure research on the simulated curves: spreads, regimes, curve-change decomposition, liquidity, calendars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets import futures_analytics as fa
from src.assets import futures_research as fr


@pytest.fixture(scope="module")
def cl():
    return fa.simulated_curve("CL", as_of="2026-09-18")


@pytest.fixture(scope="module")
def ng():
    return fa.simulated_curve("NG", as_of="2026-09-18")


def test_the_spread_matrix_is_later_less_earlier_in_points_percent_and_money(cl):
    m = fr.spread_matrix(cl, depth=5)
    n = len(m["contracts"])
    assert n == 5 and m["points"][0][0] is None and m["points"][2][1] is None, "only later-less-earlier cells exist"
    px = m["price"]
    assert m["points"][0][3] == pytest.approx(px[3] - px[0]) and m["percent"][1][4] == pytest.approx(px[4] / px[1] - 1) and m["dollars"][0][1] == pytest.approx(m["points"][0][1] * cl.spec.multiplier)
    assert "later less earlier" in m["convention"]


def test_the_spread_history_is_continuous_in_rank_not_contract_and_says_where_the_contract_changed(cl):
    h = fr.spread_history(cl, 1, 2, window=120)
    assert len(h["dates"]) == len(h["spread"]) and h["contract_changes"] and abs(h["zscore"]) < 10 and 0 <= h["percentile"] <= 1
    assert h["spread"][-1] == pytest.approx(h["current"], abs=1e-4) and "rank" in h["note"]
    with pytest.raises(ValueError, match="does not reach"):
        fr.spread_history(cl, 1, 40)


def test_regimes_partition_the_days_and_the_transition_matrix_is_a_probability_matrix(ng):
    r = fr.regime_analysis(ng)
    assert sum(r["share"].values()) == pytest.approx(1.0) and r["current"] in r["share"] and r["current_spell_days"] >= 1
    for row in r["transition"].values():
        assert sum(row.values()) == pytest.approx(1.0)
    assert sum(s["count"] for s in r["spells"].values()) >= 1 and all(abs(sum(v.values()) - 1) < 1e-9 for v in r["by_year"].values()) and "sample" in r["caveat"]


def test_a_parallel_shift_is_all_level_and_a_tilt_is_all_slope():
    root = fa.ROOTS["CL"]
    dates = pd.to_datetime(["2026-01-05", "2026-02-05"])
    rows = []
    exps = pd.to_datetime(["2026-03-20", "2026-04-20", "2026-05-20", "2026-06-22", "2026-07-21"])
    for d, tilt, lvl in ((dates[0], 0.0, 0.0), (dates[1], 0.0, 0.05)):
        for e in exps:
            ttm = (e - d).days / 365.0
            rows.append({"date": d, "contract": f"C{e.month}", "expiry": e, "price": 70.0 * np.exp(lvl + tilt * ttm), "ttm": ttm, "open_interest": 1000.0, "volume": 100.0})
    data = fa.CurveData("CL", root, pd.DataFrame(rows), None, "simulated")
    shift = fr.change_decomposition(data, dates[0], dates[1])
    assert shift["level"] == pytest.approx(0.05) and abs(shift["slope_per_year"]) < 1e-9 and shift["share_of_variation"]["level"] == pytest.approx(1.0)
    rows2 = []
    for d, tilt in ((dates[0], 0.0), (dates[1], 0.2)):
        for e in exps:
            ttm = (e - d).days / 365.0
            rows2.append({"date": d, "contract": f"C{e.month}", "expiry": e, "price": 70.0 * np.exp(tilt * ttm), "ttm": ttm, "open_interest": 1000.0, "volume": 100.0})
    tilt = fr.change_decomposition(fa.CurveData("CL", root, pd.DataFrame(rows2), None, "simulated"), dates[0], dates[1])
    assert tilt["slope_per_year"] == pytest.approx(0.2, abs=1e-6) and tilt["share_of_variation"]["slope"] > 0.05 and sum(tilt["share_of_variation"].values()) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="fewer than three"):
        fr.change_decomposition(data.__class__("CL", root, pd.DataFrame(rows[:2] + rows[5:7]), None, "simulated"), dates[0], dates[1])


def test_liquidity_shares_sum_to_one_and_a_curve_without_open_interest_says_so(cl):
    lq = fr.liquidity(cl, depth=6)
    assert lq["has_open_interest"] and sum(r["oi_share"] for r in lq["rows"]) == pytest.approx(1.0) and lq["most_liquid"] in {r["contract"] for r in lq["rows"]} and lq["oi_weighted_dte"] > 0
    blank = fa.CurveData(cl.root, cl.spec, cl.long.assign(open_interest=np.nan), None, "user_file")
    out = fr.liquidity(blank)
    assert not out["has_open_interest"] and "cannot be assessed" in out["note"] and out["oi_weighted_dte"] is None and all(r["oi_share"] is None for r in out["rows"])


def test_the_expiry_calendar_follows_the_exchange_rule_and_names_what_it_does_not_model(cl):
    c = fr.expiry_calendar(cl, 6, roll_days=5)
    assert c["rule"] == "cme_crude" and len(c["rows"]) >= 5 and all(r["weekday"] not in ("Saturday", "Sunday") for r in c["rows"]) and "holidays are not modelled" in c["caveat"]
    r0 = c["rows"][0]
    assert pd.Timestamp(r0["roll_date"]) == pd.Timestamp(r0["expiry"]) - pd.Timedelta(days=5)


def test_curves_of_different_markets_are_compared_by_shape(cl, ng):
    cmp = fr.compare_curves({"CL": cl, "NG": ng}, depth=6)
    assert [c["root"] for c in cmp["curves"]] == ["CL", "NG"] and all(c["relative"][0] == 1.0 and len(c["relative"]) == 6 for c in cmp["curves"])
