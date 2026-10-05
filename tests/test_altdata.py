"""Non-price data: positioning is parsed correctly, nothing is visible before its release, features are causal."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.altdata import alt_feature_levels, cftc_series, parse_cftc
from src.data.macro import MacroSeriesSpec, asof_panel


def _write_zip(path: Path, rows: list[dict]):
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data.txt", buffer.getvalue())


def test_parse_cftc_computes_net_positioning_over_open_interest_and_matches_codes_with_leading_zeros(tmp_path):
    (tmp_path / "cftc").mkdir()
    _write_zip(tmp_path / "cftc" / "fin_fut_txt_2006_2016.zip", [
        {"CFTC_Contract_Market_Code": "43602", "Report_Date_as_YYYY-MM-DD": "1/10/2012 12:00:00 AM", "Open_Interest_All": 1000,
         "Lev_Money_Positions_Long_All": 300, "Lev_Money_Positions_Short_All": 100},
        {"CFTC_Contract_Market_Code": "999999", "Report_Date_as_YYYY-MM-DD": "1/10/2012 12:00:00 AM", "Open_Interest_All": 1000,
         "Lev_Money_Positions_Long_All": 1, "Lev_Money_Positions_Short_All": 900}])
    _write_zip(tmp_path / "cftc" / "com_disagg_txt_2020.zip", [
        {"CFTC_Contract_Market_Code": "088691", "Report_Date_as_YYYY-MM-DD": "2020-01-07", "Open_Interest_All": 500,
         "M_Money_Positions_Long_All": 100, "M_Money_Positions_Short_All": 150}])
    contracts = {"ust10": {"code": "043602", "report": "tff", "group": "lev"},
                 "gold": {"code": "088691", "report": "disagg", "group": "mmoney"}}
    table = parse_cftc(tmp_path, contracts).set_index("contract")
    assert set(table.index) == {"ust10", "gold"}                       # the unrelated code is not picked up
    assert table.loc["ust10", "net_pct_oi"] == 0.2 and table.loc["gold", "net_pct_oi"] == -0.1
    assert table.loc["ust10", "date"] == pd.Timestamp("2012-01-10")


def test_a_tuesday_positioning_report_is_first_visible_on_the_following_monday():
    positioning = pd.DataFrame({"date": pd.to_datetime(["2020-01-07", "2020-01-14"]), "contract": "es", "net_pct_oi": [0.1, 0.5],
                                "open_interest": 1.0})
    series, specs = cftc_series(positioning, lag_days=4)
    calendar = pd.bdate_range("2020-01-06", "2020-01-20")
    panel = asof_panel(series, specs, calendar)["cftc_es"]
    assert np.isnan(panel[pd.Timestamp("2020-01-10")])                  # Friday: the 7th is not yet usable
    assert panel[pd.Timestamp("2020-01-13")] == 0.1                     # Monday: it is
    assert panel[pd.Timestamp("2020-01-17")] == 0.1                     # the 14th report is still not usable on Friday
    assert panel[pd.Timestamp("2020-01-20")] == 0.5


def _synthetic_raw(n=2200, seed=0):
    rng = np.random.default_rng(seed)
    daily = pd.bdate_range("2008-01-01", periods=n)
    weekly = pd.date_range("2008-01-05", periods=n // 5, freq="7D")
    raw = {"BAA10Y": pd.Series(2 + rng.standard_normal(n).cumsum() * 0.01, daily),
           "ICSA": pd.Series(300000 + rng.standard_normal(len(weekly)).cumsum() * 1000, weekly),
           "VIX3M": pd.Series(20 + rng.standard_normal(n) * 2, daily), "VXN": pd.Series(22 + rng.standard_normal(n) * 2, daily),
           "GVZ": pd.Series(16 + rng.standard_normal(n), daily), "OVX": pd.Series(35 + rng.standard_normal(n) * 3, daily)}
    vix = pd.Series(19 + rng.standard_normal(n) * 2, daily)
    specs = [MacroSeriesSpec("BAA10Y", "fred", "daily", 1), MacroSeriesSpec("ICSA", "fred", "daily", 6),
             MacroSeriesSpec("VIX3M", "yahoo", "daily", 0), MacroSeriesSpec("VXN", "yahoo", "daily", 0),
             MacroSeriesSpec("GVZ", "yahoo", "daily", 0), MacroSeriesSpec("OVX", "yahoo", "daily", 0)]
    positioning = pd.DataFrame([{"date": d, "contract": c, "net_pct_oi": rng.normal(), "open_interest": 1.0}
                                for c in ("es", "nq", "ust10", "gold", "silver", "crude") for d in weekly])
    return raw, specs, positioning, vix, daily


def test_alt_features_never_use_a_value_before_it_exists():
    raw, specs, positioning, vix, daily = _synthetic_raw()
    vix_spec = MacroSeriesSpec("VIX", "yahoo", "daily", 0)
    base = alt_feature_levels(raw, specs, positioning, vix, vix_spec, 4, daily)
    cut = daily[1500]
    tampered_raw = {k: v.copy() for k, v in raw.items()}
    for k in tampered_raw:
        tampered_raw[k][tampered_raw[k].index > cut] += 1000.0
    tampered_pos = positioning.copy()
    tampered_pos.loc[tampered_pos["date"] > cut, "net_pct_oi"] += 50.0
    changed = alt_feature_levels(tampered_raw, specs, tampered_pos, vix[vix.index <= cut].reindex(vix.index).ffill(), vix_spec, 4, daily)
    horizon = cut - pd.Timedelta(days=12)                                # past the longest release lag
    pd.testing.assert_frame_equal(base.loc[:horizon], changed.loc[:horizon], check_exact=False, atol=1e-9)
    assert list(base.columns)[:2] == ["baa10y_level", "baa10y_change_63"] and base.shape[1] == 14
