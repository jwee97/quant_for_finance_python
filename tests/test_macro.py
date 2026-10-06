"""Macro data: publication lags, transforms and causal standardisation.

The invariant that matters most: a value is never visible before the date it
was published. Every test here is a way of trying to break that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.macro import (
    MacroDownloader,
    MacroSeriesSpec,
    asof_series,
    availability_index,
    staleness,
)
from src.features.macro import (
    FEATURE_COLUMNS,
    expanding_percentile,
    expanding_zscore,
    macro_feature_panel,
    monthly_transforms,
)

MONTHLY = MacroSeriesSpec("M", "fred", "monthly", 45)
DAILY = MacroSeriesSpec("D", "fred", "daily", 1)
INDEX = MacroSeriesSpec("I", "yahoo", "daily", 0)


def _monthly(n=36, start="2018-01-01"):
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.Series(np.arange(1, n + 1, dtype=float), index=idx)


def test_monthly_availability_is_the_stamp_plus_the_lag():
    avail = availability_index(pd.DatetimeIndex(["2022-06-01"]), MONTHLY)
    assert avail[0] == pd.Timestamp("2022-07-16")


def test_a_monthly_value_is_invisible_one_day_before_release_and_visible_on_it():
    series = _monthly()
    calendar = pd.bdate_range("2018-01-01", "2020-12-31")
    asof = asof_series(series, MONTHLY, calendar)
    ref = pd.Timestamp("2019-03-01")
    release = ref + pd.Timedelta(days=45)
    before = calendar[calendar < release][-1]
    after = calendar[calendar >= release][0]
    assert asof.loc[before] == series[ref - pd.DateOffset(months=1)]     # still February's value
    assert asof.loc[after] == series[ref]                                 # March's, now published


def test_nothing_is_visible_before_the_first_release():
    series = _monthly()
    calendar = pd.bdate_range("2018-01-01", "2018-03-31")
    asof = asof_series(series, MONTHLY, calendar)
    assert asof.loc[:"2018-02-14"].isna().all()                           # Jan stamp + 45d = Feb 15
    assert asof.loc["2018-02-15":].notna().all()


def test_daily_series_are_lagged_by_business_days():
    idx = pd.bdate_range("2020-01-01", periods=30)
    series = pd.Series(np.arange(30, dtype=float), index=idx)
    asof = asof_series(series, DAILY, idx)
    assert np.isnan(asof.iloc[0])
    assert asof.iloc[1] == 0.0 and asof.iloc[10] == 9.0                   # one business day behind


def test_a_zero_lag_series_is_visible_the_same_day():
    idx = pd.bdate_range("2020-01-01", periods=10)
    series = pd.Series(np.arange(10, dtype=float), index=idx)
    pd.testing.assert_series_equal(asof_series(series, INDEX, idx), series, check_names=False)


def test_values_published_after_t_cannot_change_anything_at_or_before_t():
    series = _monthly(48)
    calendar = pd.bdate_range("2018-01-01", "2021-12-31")
    base = asof_series(series, MONTHLY, calendar)
    cut = pd.Timestamp("2020-06-30")
    altered = series.copy()
    altered[altered.index + pd.Timedelta(days=45) > cut] += 1000.0         # anything not yet public at `cut`
    changed = asof_series(altered, MONTHLY, calendar)
    pd.testing.assert_series_equal(base.loc[:cut], changed.loc[:cut])
    assert (changed.loc[cut + pd.Timedelta(days=60):] != base.loc[cut + pd.Timedelta(days=60):]).any()


def test_staleness_never_exceeds_the_release_cycle_plus_the_lag_for_monthly_data():
    series = _monthly(48)
    calendar = pd.bdate_range("2019-06-03", "2021-06-30")
    stale = staleness({"M": series}, [MONTHLY], calendar)["M"]
    assert stale.min() >= 0
    assert stale.max() <= 31                                              # never older than one month


def test_weekend_reference_days_collapse_onto_the_next_business_day_without_error():
    """FRED's fed funds series is reported seven days a week; Friday, Saturday and Sunday
    values all become public on Monday. The last of them is the value in force."""
    idx = pd.date_range("2020-01-03", "2020-01-12", freq="D")             # Fri .. Sun
    series = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    calendar = pd.bdate_range("2020-01-02", "2020-01-17")
    asof = asof_series(series, DAILY, calendar)
    stale = staleness({"D": series}, [DAILY], calendar)["D"]
    assert asof.loc["2020-01-06"] == 2.0                                   # Sunday 5th's value, Monday
    assert asof.loc["2020-01-07"] == 3.0                                   # Monday 6th's value, Tuesday
    assert stale.loc["2020-01-06"] == 0                                    # published that very day
    assert asof.loc["2020-01-13"] == 9.0                                   # Fri/Sat/Sun 10-12th, public Monday
    assert stale.loc["2020-01-14"] == 1                                    # the series ends; one day older


def test_cpi_yoy_is_computed_on_reference_months_then_lagged():
    idx = pd.date_range("2015-01-01", periods=84, freq="MS")
    cpi = pd.Series(100.0 * 1.003 ** np.arange(84), index=idx)
    yoy = monthly_transforms({"CPIAUCNS": cpi})["CPI_YOY"]
    assert yoy.dropna().iloc[0] == pytest.approx(1.003 ** 12 - 1.0)
    assert yoy.iloc[:12].isna().all()                                      # needs twelve months of history


def test_cpi_yoy_as_known_on_a_date_uses_only_published_months():
    idx = pd.date_range("2015-01-01", periods=84, freq="MS")
    rng = np.random.default_rng(0)
    cpi = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.003, 0.002, 84)), index=idx)
    spec = MacroSeriesSpec("CPIAUCNS", "fred", "monthly", 45)
    panel = macro_levels_for(cpi, spec)
    t = pd.Timestamp("2019-09-20")
    latest = max(m for m in idx if m + pd.Timedelta(days=45) <= t)
    expected = cpi[latest] / cpi[latest - pd.DateOffset(months=12)] - 1.0
    assert panel.loc[t, "CPI_YOY"] == pytest.approx(expected)


def macro_levels_for(cpi, spec):
    from src.features.macro import build_macro_levels

    calendar = pd.bdate_range("2015-01-01", "2021-12-31")
    return build_macro_levels({"CPIAUCNS": cpi}, [spec], calendar)


def test_a_missing_month_does_not_misalign_year_over_year_arithmetic():
    """October 2025 CPI was never published. Position-based pct_change(12) would compare
    November 2025 with October 2024 (thirteen months) for a full year afterwards."""
    idx = pd.date_range("2022-01-01", periods=48, freq="MS")
    cpi = pd.Series(100.0 * 1.003 ** np.arange(48), index=idx).drop(pd.Timestamp("2024-10-01"))
    yoy = monthly_transforms({"CPIAUCNS": cpi})["CPI_YOY"]
    twelve_months = 1.003 ** 12 - 1.0
    assert np.isnan(yoy.loc["2024-10-01"])                                 # the gap stays a gap
    assert np.isnan(yoy.loc["2025-10-01"])                                 # its anniversary has no base
    valid = yoy.dropna().loc["2023-01-01":]
    assert np.allclose(valid.to_numpy(), twelve_months, atol=1e-12)        # never the 13-month change


def test_derived_monthly_series_are_only_defined_where_the_source_was_observed():
    idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    unemployment = pd.Series(5.0, index=idx).drop(pd.Timestamp("2021-10-01"))
    sahm = monthly_transforms({"UNRATE": unemployment})["SAHM"]
    assert np.isnan(sahm.loc["2021-10-01"])                                # no print, no derived value
    assert sahm.loc["2021-11-01":].notna().all()                           # the next months recover
    assert np.allclose(sahm.dropna().to_numpy(), 0.0)                      # flat unemployment: zero gap


def test_sahm_gap_is_zero_for_flat_unemployment_and_rises_in_a_downturn():
    idx = pd.date_range("2015-01-01", periods=60, freq="MS")
    flat = pd.Series(4.0, index=idx)
    assert monthly_transforms({"UNRATE": flat})["SAHM"].dropna().abs().max() == pytest.approx(0.0)
    rising = flat.copy()
    rising.iloc[40:] = np.linspace(4.0, 6.5, 20)
    sahm = monthly_transforms({"UNRATE": rising})["SAHM"]
    assert sahm.iloc[-1] > 0.5                                             # above the Sahm trigger
    assert sahm.iloc[:38].dropna().abs().max() == pytest.approx(0.0)


def test_expanding_zscore_is_causal():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": rng.normal(size=2000).cumsum()},
                         index=pd.bdate_range("2010-01-01", periods=2000))
    base = expanding_zscore(frame, min_periods=100)
    altered = frame.copy()
    altered.iloc[1500:] += 500.0
    changed = expanding_zscore(altered, min_periods=100)
    pd.testing.assert_frame_equal(base.iloc[:1500], changed.iloc[:1500])


def test_expanding_zscore_needs_history_and_is_bounded():
    frame = pd.DataFrame({"x": np.r_[np.zeros(300), [1e6]]},
                         index=pd.bdate_range("2010-01-01", periods=301))
    z = expanding_zscore(frame, min_periods=100, winsorize=4.0)
    assert z.iloc[:99].isna().all().iloc[0]
    assert z["x"].abs().max() <= 4.0 + 1e-12


def test_expanding_percentile_is_one_for_a_monotone_series_and_causal():
    s = pd.Series(np.arange(500, dtype=float))
    pct = expanding_percentile(s, min_periods=50)
    assert pct.dropna().iloc[-1] == pytest.approx(1.0)
    altered = s.copy()
    altered.iloc[400:] = -1e9
    pd.testing.assert_series_equal(pct.iloc[:400], expanding_percentile(altered, 50).iloc[:400])


def _synthetic_raw():
    rng = np.random.default_rng(5)
    daily_idx = pd.bdate_range("2000-01-03", "2021-12-31")
    monthly_idx = pd.date_range("2000-01-01", "2021-12-01", freq="MS")
    raw = {
        "T10Y2Y": pd.Series(rng.normal(1.0, 0.5, len(daily_idx)).cumsum() * 0.01, index=daily_idx),
        "T10Y3M": pd.Series(rng.normal(1.0, 0.5, len(daily_idx)).cumsum() * 0.01, index=daily_idx),
        "DFF": pd.Series(np.abs(rng.normal(2.0, 0.3, len(daily_idx)).cumsum() * 0.01) + 0.1, index=daily_idx),
        "T10YIE": pd.Series(2.0 + rng.normal(0, 0.05, len(daily_idx)).cumsum() * 0.1, index=daily_idx),
        "VIX": pd.Series(np.exp(rng.normal(3.0, 0.3, len(daily_idx))), index=daily_idx),
        "MOVE": pd.Series(np.exp(rng.normal(4.5, 0.2, len(daily_idx))), index=daily_idx),
        "CPIAUCNS": pd.Series(100 * np.cumprod(1 + rng.normal(0.002, 0.001, len(monthly_idx))), index=monthly_idx),
        "UNRATE": pd.Series(5 + rng.normal(0, 0.2, len(monthly_idx)).cumsum() * 0.1, index=monthly_idx),
        "GACDFSA066MSFRBPHI": pd.Series(rng.normal(10, 15, len(monthly_idx)), index=monthly_idx),
    }
    specs = [MacroSeriesSpec("T10Y2Y", "fred", "daily", 1), MacroSeriesSpec("T10Y3M", "fred", "daily", 1),
             MacroSeriesSpec("DFF", "fred", "daily", 1), MacroSeriesSpec("T10YIE", "fred", "daily", 1),
             MacroSeriesSpec("VIX", "yahoo", "daily", 0), MacroSeriesSpec("MOVE", "yahoo", "daily", 0),
             MacroSeriesSpec("CPIAUCNS", "fred", "monthly", 45), MacroSeriesSpec("UNRATE", "fred", "monthly", 40),
             MacroSeriesSpec("GACDFSA066MSFRBPHI", "fred", "monthly", 22)]
    return raw, specs


def test_feature_panel_has_every_feature_and_is_causal_end_to_end():
    raw, specs = _synthetic_raw()
    target = pd.bdate_range("2010-01-04", "2021-12-31")
    base = macro_feature_panel(raw, specs, target)
    assert list(base["z"].columns) == FEATURE_COLUMNS
    assert base["z"].loc["2012-01-02":].notna().all().all()

    cut = pd.Timestamp("2018-06-29")
    altered = {k: v.copy() for k, v in raw.items()}
    for key, series in altered.items():
        spec = next(s for s in specs if s.id == key)
        published_after_cut = availability_index(series.index, spec) > cut
        altered[key][published_after_cut] = altered[key][published_after_cut] * 7.0 + 123.0
    changed = macro_feature_panel(altered, specs, target)
    for name in ("levels", "features", "z"):
        pd.testing.assert_frame_equal(base[name].loc[:cut], changed[name].loc[:cut])


def test_downloader_keeps_raw_files_immutable_and_records_provenance(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_fetch(self, spec):
        calls["n"] += 1
        return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=10),
                             "value": np.arange(10, dtype=float) + calls["n"]})

    monkeypatch.setattr(MacroDownloader, "_fetch_fred", fake_fetch)
    downloader = MacroDownloader(tmp_path / "raw", tmp_path / "meta", max_retries=1)
    spec = MacroSeriesSpec("X", "fred", "daily", 1)

    first = downloader.download_all([spec])
    content = (tmp_path / "raw" / "X.csv").read_text()
    second = downloader.download_all([spec])                                # must not refetch
    assert calls["n"] == 1
    assert (tmp_path / "raw" / "X.csv").read_text() == content
    assert second["series"]["X"]["status"] == "cached"
    assert first["series"]["X"]["sha256"] == second["series"]["X"]["sha256"]
    assert first["data_version"] == second["data_version"]
    assert (tmp_path / "meta" / "macro_manifest.json").exists()

    downloader.download_all([spec], force=True)                             # explicit override only
    assert calls["n"] == 2
