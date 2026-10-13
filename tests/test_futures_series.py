"""The History view of a futures product: a daily price series with no curve. What it reports, and what it refuses to claim."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets import futures_analytics as fa


def _frame(n=1500, seed=3, end=None, jump=None, volume=True):
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now().normalize() - pd.offsets.BDay(1)
    dates = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.011, n))
    if jump is not None:
        close[jump[0]:] *= 1.0 + jump[1]
    out = pd.DataFrame({"date": dates, "close": close})
    if volume:
        out["volume"] = rng.integers(1_000, 5_000, n).astype(float)
    return out


def test_the_statistics_are_what_the_prices_say():
    f = _frame()
    v = fa.series_view(f)
    s, c = v["stats"], f.set_index("date")["close"]
    r = np.log(c).diff()
    assert s["last"] == pytest.approx(c.iloc[-1]) and s["bars"] == len(f) and s["last_date"] == str(c.index[-1].date()) and s["first_date"] == str(c.index[0].date())
    assert s["return_1d"] == pytest.approx(c.iloc[-1] / c.iloc[-2] - 1) and s["return_1m"] == pytest.approx(c.iloc[-1] / c.iloc[-22] - 1) and s["return_1y"] == pytest.approx(c.iloc[-1] / c.iloc[-253] - 1)
    assert s["vol_1m"] == pytest.approx(r.tail(21).std() * np.sqrt(252)) and s["vol_1y"] == pytest.approx(r.tail(252).std() * np.sqrt(252))
    peak = c.cummax()
    assert s["max_drawdown"] == pytest.approx((c / peak - 1).min()) and s["drawdown"] == pytest.approx(c.iloc[-1] / peak.iloc[-1] - 1) and s["max_drawdown"] <= s["drawdown"] <= 0
    assert s["high_52w"] == pytest.approx(c.tail(252).max()) and s["off_high"] == pytest.approx(c.iloc[-1] / c.tail(252).max() - 1)
    simple = c.pct_change().dropna()
    assert s["best_day"] == pytest.approx(simple.max()) and s["worst_day"] == pytest.approx(simple.min()) and s["hit_rate"] == pytest.approx((simple > 0).mean()) and s["volume_21d"] > 0
    assert len(v["dates"]) == len(v["close"]) == len(v["drawdown"]) == len(v["vol_21"]) == len(v["vol_63"]) == len(v["volume"]) == len(f)
    assert v["vol_21"][0] is None and v["vol_21"][-1] == pytest.approx(r.tail(21).std() * np.sqrt(252)) and v["warnings"] == []
    assert s["return_ytd"] is not None and -1 < s["return_ytd"] < 5


def test_a_long_history_is_thinned_but_keeps_its_first_and_last_day():
    f = _frame(n=6000)
    v = fa.series_view(f, max_points=1000)
    assert len(v["dates"]) <= 1001 and v["dates"][0] == str(f["date"].iloc[0].date()) and v["dates"][-1] == str(f["date"].iloc[-1].date())
    assert v["close"][-1] == pytest.approx(f["close"].iloc[-1]) and v["stats"]["bars"] == 6000, "the statistics use every bar, not the thinned ones"


def test_the_calendar_months_need_two_and_a_half_years_and_find_a_month_that_always_rises():
    short = fa.series_view(_frame(n=400))
    assert short["seasonal"] == []
    dates = pd.bdate_range("2019-01-01", periods=1800)
    close = np.full(len(dates), 100.0)
    for i in range(1, len(dates)):
        close[i] = close[i - 1] * (1.002 if dates[i].month == 1 else 1.0)            # up two tenths of a percent a day in January, flat otherwise
    v = fa.series_view(pd.DataFrame({"date": dates, "close": close}))
    jan = next(m for m in v["seasonal"] if m["month"] == 1)
    assert len(v["seasonal"]) == 12 and jan["hit_rate"] == 1.0 and jan["mean"] > 0.03 and jan["n"] >= 4
    assert all(m["mean"] == pytest.approx(0.0, abs=1e-12) for m in v["seasonal"] if m["month"] != 1)


def test_the_largest_moves_show_where_a_series_was_spliced_and_the_page_is_warned():
    f = _frame(jump=(700, 0.22))
    v = fa.series_view(f)
    top = v["moves"][0]
    assert top["date"] == str(f["date"].iloc[700].date()) and top["change"] == pytest.approx(f["close"].iloc[700] / f["close"].iloc[699] - 1) and top["to"] > top["from"]
    assert [abs(m["change"]) for m in v["moves"]] == sorted((abs(m["change"]) for m in v["moves"]), reverse=True) and len(v["moves"]) == 8
    assert any("often the day the series switched" in w for w in v["warnings"])


def test_a_stale_history_and_gaps_are_reported():
    old = _frame(end="2026-01-02")
    assert any("days old" in w and "download the history again" in w for w in fa.series_view(old)["warnings"])
    f = _frame(n=300)
    gap = pd.concat([f.iloc[:100], f.iloc[130:]])
    assert any("1 gap longer than a week" in w for w in fa.series_view(gap)["warnings"])


def test_bad_input_is_refused_with_a_reason_and_bad_rows_are_dropped():
    with pytest.raises(ValueError, match="at least 30"):
        fa.series_view(_frame(n=20))
    with pytest.raises(ValueError, match="columns date and close"):
        fa.series_view(pd.DataFrame({"date": [1], "price": [2]}))
    f = _frame(n=200)
    dirty = pd.concat([f, f.iloc[[10]]]).copy()
    dirty.loc[dirty.index[5], "close"] = np.nan
    dirty.loc[dirty.index[6], "close"] = -3.0
    v = fa.series_view(dirty)
    assert v["stats"]["bars"] == 198 and np.isfinite(v["stats"]["last"]), "a missing, negative or repeated bar is not a bar"
    assert fa.series_view(_frame(volume=False))["volume"] is None and "volume_21d" not in fa.series_view(_frame(volume=False))["stats"]
