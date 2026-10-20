"""The point-in-time engine: a figure is usable only from the day it was public, and the tables built from the database never let a date see further than that."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import pit
from src.fundamentals.pit import PeriodTables, PitSettings, availability, step_panel
from src.fundamentals.providers.simulated import SimulatedProvider, SimulatedWorld
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement


def _periods(**rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for c in ("filing_date", "accepted_date", "publication_date", "period_end"):
        if c in frame:
            frame[c] = pd.to_datetime(frame[c])
    return frame


# ------------------------------------------------------------------------------------------------------------------------ availability
def test_a_figure_is_available_the_day_after_the_earliest_date_a_provider_gave():
    frame = _periods(filing_date=["2021-03-01", None, "2021-03-10"], accepted_date=["2021-02-26", None, None], publication_date=[None, None, None],
                     period_end=["2020-12-31", "2020-12-31", "2020-12-31"], fiscal_period=["FY", "FY", "FY"], pit_quality=["filed", "estimated", "filed"])
    available, estimated = availability(frame, PitSettings())
    assert list(available) == [pd.Timestamp("2021-02-27"), pd.Timestamp("2020-12-31") + pd.Timedelta(days=121), pd.Timestamp("2021-03-11")]        # accepted before filed wins; a missing date is the period end plus a lag
    assert list(estimated) == [False, True, False]


def test_a_restated_vintage_is_public_from_the_day_it_was_first_seen_not_from_the_original_filing():
    frame = _periods(filing_date=["2021-03-01"], accepted_date=[None], publication_date=["2022-06-30"], period_end=["2020-12-31"], fiscal_period=["FY"], pit_quality=["restated"])
    available, estimated = availability(frame, PitSettings(info_lag_days=0))
    assert available.iloc[0] == pd.Timestamp("2022-06-30") and not estimated.iloc[0]


def test_the_assumed_lag_depends_on_the_kind_of_period_and_never_equals_the_period_end():
    frame = _periods(filing_date=[None, None], accepted_date=[None, None], publication_date=[None, None], period_end=["2020-12-31", "2020-09-30"], fiscal_period=["FY", "Q3"],
                     pit_quality=["estimated", "estimated"])
    available, _ = availability(frame, PitSettings(info_lag_days=0))
    assert list(available - frame["period_end"]) == [pd.Timedelta(days=120), pd.Timedelta(days=60)]


# ------------------------------------------------------------------------------------------------------------------------ the step function
def _events(rows):
    return pd.DataFrame(rows, columns=["security_id", "available_date", "period_end", "value"]).astype({"available_date": "datetime64[ns]", "period_end": "datetime64[ns]"})


def test_a_value_appears_on_its_availability_date_and_holds_until_the_next_filing():
    dates = pd.date_range("2021-01-01", "2022-12-31", freq="D")
    ev = _events([(1, "2021-03-01", "2020-12-31", 10.0), (1, "2022-03-01", "2021-12-31", 14.0)])
    panel = step_panel(ev, dates, [1], max_age_days=None)
    assert panel[1].loc["2021-02-28"] != panel[1].loc["2021-02-28"]                       # NaN before it was public
    assert panel[1].loc["2021-03-01"] == 10.0 and panel[1].loc["2022-02-28"] == 10.0 and panel[1].loc["2022-03-01"] == 14.0


def test_a_late_restatement_of_an_old_year_never_replaces_a_newer_year():
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    ev = _events([(1, "2021-03-01", "2020-12-31", 10.0), (1, "2022-03-01", "2021-12-31", 14.0), (1, "2022-09-01", "2020-12-31", 11.0)])
    panel = step_panel(ev, dates, [1], max_age_days=None)
    assert panel[1].loc["2022-10-01"] == 14.0                                                 # the 2020 figure was corrected in September 2022, but 2021 is still the latest year


def test_a_figure_that_is_too_old_expires_instead_of_being_carried_forever():
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    ev = _events([(1, "2021-03-01", "2020-12-31", 10.0)])
    panel = step_panel(ev, dates, [1], max_age_days=600)
    assert panel[1].loc["2022-06-01"] == 10.0 and np.isnan(panel[1].loc["2022-09-01"])        # 600 days after the period end


def test_securities_are_independent_and_unknown_ones_are_ignored():
    dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    ev = _events([(1, "2021-03-01", "2020-12-31", 10.0), (2, "2021-06-01", "2021-03-31", 7.0), (9, "2021-01-05", "2020-12-31", 99.0)])
    panel = step_panel(ev, dates, [1, 2])
    assert list(panel.columns) == [1, 2] and panel[1].loc["2021-05-01"] == 10.0 and np.isnan(panel[2].loc["2021-05-01"]) and panel[2].loc["2021-06-01"] == 7.0


# ------------------------------------------------------------------------------------------------------------------------ the tables
@pytest.fixture(scope="module")
def world():
    return SimulatedWorld(n_companies=24, start_year=2010, end_year=2016, seed=5)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory, world):
    svc = FundamentalsService(tmp_path_factory.mktemp("pit"), providers={"simulated": SimulatedProvider(world=world)})
    ids = svc.add_companies("simulated", svc.provider("simulated").symbols())
    svc.sync("simulated", security_ids=ids, periods=("annual", "quarter"))
    return svc, ids


def test_trailing_twelve_months_at_a_fiscal_year_end_equal_the_annual_figure(loaded):
    svc, ids = loaded
    tables = PeriodTables(svc.store, ids)
    ttm, fy = tables.ttm, tables.fy
    both = ttm.merge(fy[["security_id", "period_end", "revenue", "net_income"]], on=["security_id", "period_end"], suffixes=("", "_fy")).dropna(subset=["revenue", "revenue_fy"])
    assert len(both) > 20
    ratio = (both["revenue"] / both["revenue_fy"]).median()
    assert 0.97 < ratio < 1.03                                                                  # four quarters add up to the year (the simulator's quarters are weights of the annual figure)
    assert (ttm["fiscal_period"] == "TTM").all() and (ttm["available_date"] > ttm["period_end"]).all()


def test_nothing_is_visible_before_its_filing_date_plus_the_lag(loaded):
    svc, ids = loaded
    tables = PeriodTables(svc.store, ids)
    row = tables.fy[tables.fy["revenue"].notna() & ~tables.fy["pit_estimated"]].iloc[10]
    dates = pd.date_range(row["period_end"] - pd.Timedelta(days=30), row["available_date"] + pd.Timedelta(days=30), freq="D")
    panel = tables.panel("revenue", dates, basis="fy", security_ids=[int(row["security_id"])])
    series = panel[int(row["security_id"])]
    before = series[series.index < row["available_date"]]
    after = series[series.index >= row["available_date"]]
    assert (before != row["revenue"]).all() and (after == row["revenue"]).all()               # the figure for this year appears on its availability date, not a day before


def test_the_year_before_is_found_by_period_end_not_by_row_position(loaded):
    svc, ids = loaded
    tables = PeriodTables(svc.store, ids)
    last = tables.fy.groupby("security_id").tail(1).index
    prior = tables.prior_col("fy", "revenue", 1)
    previous = tables.fy.groupby("security_id")["revenue"].shift(1)
    both = pd.concat([prior, previous], axis=1).loc[last].dropna()
    assert len(both) > 10 and np.allclose(both.iloc[:, 0], both.iloc[:, 1])
    tables.add_columns("fy", {"double": tables.fy["revenue"] * 2})
    again = tables.prior_col("fy", "double", 1)
    assert np.allclose(again.dropna(), 2 * prior.reindex(again.dropna().index))                 # a derived column can be looked up a year back as well


def test_a_period_without_a_filing_date_can_be_left_out_altogether(tmp_path):
    store = FundamentalsStore(tmp_path / "f.db")
    sid = store.upsert_security(CompanyProfile("simulated", "AAA", "AAA", "A", exchange="NYSE"))
    store.store_financials(sid, "simulated", [PeriodStatement("FY", "2019-12-31", {"revenue": 5.0}, filing_date="2020-02-20"), PeriodStatement("FY", "2020-12-31", {"revenue": 6.0})])
    both = PeriodTables(store, [sid])
    assert len(both.fy) == 2 and list(both.fy["pit_estimated"]) == [False, True] and both.estimated_share() == 0.5
    only_filed = PeriodTables(store, [sid], PitSettings(use_estimated=False))
    assert len(only_filed.fy) == 1 and only_filed.estimated_share() == 0.0


def test_where_two_providers_report_a_period_the_leader_wins_and_the_other_fills_only_what_was_public_by_then(tmp_path):
    store = FundamentalsStore(tmp_path / "f.db")
    early = store.upsert_security(CompanyProfile("openfilings", "us:AAA", "AAA", "A", exchange="US"))
    late = store.upsert_security(CompanyProfile("openfilings", "us:BBB", "BBB", "B", exchange="US"))
    for sid, other_filed in ((early, "2021-02-10"), (late, "2021-04-01")):
        store.store_financials(sid, "openfilings", [PeriodStatement("FY", "2020-12-31", {"revenue": 100.0, "net_income": 10.0}, filing_date="2021-02-20")])
        store.store_financials(sid, "simulated", [PeriodStatement("FY", "2020-12-31", {"revenue": 101.0, "net_income": 11.0, "total_assets": 500.0}, filing_date=other_filed)])
    fy = PeriodTables(store, [early, late], PitSettings(provider_priority=("openfilings", "simulated"))).fy.set_index("security_id")
    assert fy.loc[early, "revenue"] == 100.0 and fy.loc[early, "net_income"] == 10.0              # the leader's figures win
    assert fy.loc[early, "total_assets"] == 500.0                                                  # a gap the other provider filled, with a figure that was public before the leader's
    assert fy.loc[late, "revenue"] == 100.0 and np.isnan(fy.loc[late, "total_assets"])              # a figure filed after the leader's must not appear in the leader's row
    swapped = PeriodTables(store, [early, late], PitSettings(provider_priority=("simulated", "openfilings"))).fy.set_index("security_id")
    assert swapped.loc[early, "revenue"] == 101.0 and swapped.loc[early, "total_assets"] == 500.0


def test_observations_with_a_serious_quality_flag_are_treated_as_missing(tmp_path):
    store = FundamentalsStore(tmp_path / "f.db")
    sid = store.upsert_security(CompanyProfile("simulated", "AAA", "AAA", "A", exchange="NYSE"))
    store.store_financials(sid, "simulated", [PeriodStatement("FY", "2019-12-31", {"revenue": 5.0, "net_income": 1.0}, filing_date="2020-02-20")])
    with store.db.transaction() as conn:
        conn.execute("UPDATE fundamental_observations SET quality_flag = 'suspicious_jump' WHERE metric_code = 'revenue'")
    assert np.isnan(PeriodTables(store, [sid]).fy.iloc[0]["revenue"]) and PeriodTables(store, [sid]).fy.iloc[0]["net_income"] == 1.0
    assert PeriodTables(store, [sid], PitSettings(exclude_flags=())).fy.iloc[0]["revenue"] == 5.0
    assert pit.PitSettings().describe()["info_lag_days"] == 1


def test_a_database_with_annual_statements_only_still_builds_every_table(tmp_path):
    """OpenFilings gives annual statements only: every table, the year-before lookup and the derived metrics must work with no quarter at all."""
    from src.fundamentals import metrics
    store = FundamentalsStore(tmp_path / "annual.db")
    sid = store.upsert_security(CompanyProfile("openfilings", "us:AAA", "AAA", "A", exchange="US"))
    store.store_financials(sid, "openfilings", [PeriodStatement("FY", f"{y}-12-31", {"revenue": 100.0 + 10 * i, "total_assets": 500.0, "net_income": 5.0 + i}, fiscal_year=y, filing_date=f"{y + 1}-02-25")
                                                for i, y in enumerate((2018, 2019, 2020))])
    tables = PeriodTables(store, [sid])
    assert len(tables.ttm) == 3 and tables.ttm["period_end"].dtype.kind == "M" and set(tables.ttm["fiscal_period"]) == {"TTM"}
    assert np.isnan(tables.prior_col("ttm", "revenue").iloc[0]) and list(tables.prior_col("ttm", "revenue").iloc[1:]) == [100.0, 110.0]
    values = metrics.compute(tables, "ttm")
    assert values["revenue_growth"].round(6).tolist()[1:] == [0.1, round(120 / 110 - 1, 6)] and np.isnan(values["revenue_growth"].iloc[0])
    panel = tables.panel("revenue", pd.date_range("2019-01-31", "2021-12-31", freq="ME"), "ttm", [sid])
    assert panel[sid].loc["2019-03-31"] == 100.0 and panel[sid].loc["2020-03-31"] == 110.0 and panel[sid].loc["2021-03-31"] == 120.0


def test_a_figure_missing_from_the_latest_period_stays_missing_instead_of_borrowing_an_older_one():
    dates = pd.date_range("2021-01-31", "2022-12-31", freq="ME")
    ev = _events([(1, "2021-03-01", "2020-12-31", 10.0), (1, "2022-03-01", "2021-12-31", np.nan), (2, "2021-03-01", "2020-12-31", 5.0), (2, "2022-03-01", "2021-12-31", 6.0)])
    panel = step_panel(ev, dates, [1, 2], max_age_days=None)
    assert panel[1].loc["2021-06-30"] == 10.0 and np.isnan(panel[1].loc["2022-06-30"])               # the 2021 accounts did not give this figure: it is not 2020's
    assert panel[2].loc["2022-06-30"] == 6.0
    aged = step_panel(ev, dates, [1, 2], max_age_days=300)
    assert np.isnan(aged[1].loc["2021-12-31"]) and aged[1].loc["2021-09-30"] == 10.0                 # expiry still applies to the period the event describes
