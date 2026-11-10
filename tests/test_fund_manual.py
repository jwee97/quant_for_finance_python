"""Financial reports entered by hand (``fundamentals.manual``): a table pasted from Excel is read into periods and lines; an entry is checked (dates that cannot be, figures that cannot be negative,
statements that do not add up, units that do not match); saved as the source ``manual`` with the day each report became public; corrected in place or restated as a new vintage; read back, removed; and
the ratios of the point-in-time engine are worked out from it exactly as they are for a provider's figures, on a temporary copy, without storing anything."""

from __future__ import annotations

import threading
from datetime import date

import pandas as pd
import pytest

from src.fundamentals import canonical, manual
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.pit import PeriodTables, PitSettings
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement

TODAY = date(2026, 11, 10)
# a statement as it is pasted from Excel: the periods across, the lines down, a line nobody knows, the dates and the currency as rows, costs in brackets
TABLE = "\n".join([
    "Item\tFY2021\tFY2022\tFY2023",
    "Filing date\t2022-02-10\t2023-02-09\t2024-02-08",
    "Currency\tUSD\tUSD\tUSD",
    "Total revenue\t1,000\t1,200\t1,500",
    "Cost of sales\t(600)\t(700)\t(850)",
    "Gross profit\t400\t500\t650",
    "Operating income\t200\t260\t340",
    "Pretax income\t190\t250\t330",
    "Income tax\t40\t55\t70",
    "Net income\t150\t195\t260",
    "Diluted EPS\t1.50\t1.95\t2.60",
    "Cash and cash equivalents\t300\t350\t420",
    "Total current assets\t500\t580\t690",
    "Total assets\t1,500\t1,700\t2,000",
    "Total current liabilities\t300\t340\t400",
    "Long-term debt\t400\t420\t450",
    "Total liabilities\t800\t900\t1,000",
    "Total shareholders equity\t700\t800\t1,000",
    "Retained earnings\t300\t450\t650",
    "Cash from operations\t220\t260\t340",
    "Capex\t(80)\t(90)\t(110)",
    "Depreciation and amortization\t60\t65\t75",
    "Diluted shares\t100\t100\t100",
    "Share price\t20\t25\t30",
    "A line nobody knows\t1\t2\t3",
])
COMPANY = {"ticker": "TEST1", "name": "Test Co", "country": "US", "currency": "USD", "sector": "Technology"}
YEARS = {  # in millions, as a statement shows them
    2021: dict(revenue=1000, cost_of_revenue=600, gross_profit=400, operating_income=200, income_before_tax=190, income_tax=40, net_income=150, eps_diluted=1.5, cash_and_equivalents=300, current_assets=500,
               total_assets=1500, current_liabilities=300, long_term_debt=400, total_liabilities=800, shareholders_equity=700, retained_earnings=300, operating_cash_flow=220, capital_expenditure=80,
               depreciation_amortization=60, weighted_avg_shares_diluted=100),
    2022: dict(revenue=1200, cost_of_revenue=700, gross_profit=500, operating_income=260, income_before_tax=250, income_tax=55, net_income=195, eps_diluted=1.95, cash_and_equivalents=350, current_assets=580,
               total_assets=1700, current_liabilities=340, long_term_debt=420, total_liabilities=900, shareholders_equity=800, retained_earnings=450, operating_cash_flow=260, capital_expenditure=90,
               depreciation_amortization=65, weighted_avg_shares_diluted=100),
    2023: dict(revenue=1500, cost_of_revenue=850, gross_profit=650, operating_income=340, income_before_tax=330, income_tax=70, net_income=260, eps_diluted=2.6, cash_and_equivalents=420, current_assets=690,
               total_assets=2000, current_liabilities=400, long_term_debt=450, total_liabilities=1000, shareholders_equity=1000, retained_earnings=650, operating_cash_flow=340, capital_expenditure=110,
               depreciation_amortization=75, weighted_avg_shares_diluted=100),
}


def period(year: int, **override) -> dict:
    values = dict(YEARS[year])
    values.update(override.pop("values", {}))
    out = {"fiscal_period": "FY", "period_end": f"{year}-12-31", "filing_date": f"{year + 1}-02-10", "currency": "USD", "values": values}
    out.update(override)
    return out


def entry(years=(2021, 2022, 2023), **extra) -> dict:
    out = {"company": dict(COMPANY), "units": {"money": "millions", "shares": "millions"}, "periods": [period(y) for y in years]}
    out.update(extra)
    return out


@pytest.fixture()
def store(tmp_path):
    return FundamentalsStore(tmp_path / "f.db")


def errors(checked) -> list[str]:
    return [i["message"] for i in checked["issues"] if i["level"] == "error"]


def warnings(checked) -> list[str]:
    return [i["message"] for i in checked["issues"] if i["level"] == "warning"]


# ------------------------------------------------------------------------------------------------------------------ reading cells
@pytest.mark.parametrize("cell, value", [("1,234.5", 1234.5), ("(1,234)", -1234.0), ("-1234", -1234.0), ("−1,000", -1000.0), ("$1,234", 1234.0), ("1 234", 1234.0), (" 12 ", 12.0), ("1.5e3", 1500.0),
                                         (7, 7.0), (2.5, 2.5), ("", None), ("-", None), ("—", None), ("n/a", None), ("NM", None), (None, None), ("0", 0.0)])
def test_a_number_is_read_as_a_spreadsheet_shows_it_and_a_blank_is_missing_not_zero(cell, value):
    assert manual.parse_number(cell) == value


@pytest.mark.parametrize("cell", ["abc", "12%", "1,23", "1.2.3", "inf", "1_000", True, float("nan"), "(abc)"])
def test_a_cell_that_is_not_a_number_is_refused(cell):
    with pytest.raises(ValueError):
        manual.parse_number(cell)


@pytest.mark.parametrize("cell, iso", [("2023-12-31", "2023-12-31"), ("12/31/2023", "2023-12-31"), ("31/12/2023", "2023-12-31"), ("31 Dec 2023", "2023-12-31"), ("Dec 31, 2023", "2023-12-31"),
                                       ("2023-12-31T00:00:00", "2023-12-31"), ("", None), (None, None)])
def test_a_date_is_read_in_the_usual_forms(cell, iso):
    assert manual.parse_date(cell) == iso


@pytest.mark.parametrize("cell", ["2023-13-01", "2023-02-30", "yesterday", "31/31/2023"])
def test_a_date_that_cannot_be_is_refused(cell):
    with pytest.raises(ValueError):
        manual.parse_date(cell)


@pytest.mark.parametrize("label, period, year, end, guessed", [
    ("FY2023", "FY", 2023, "2023-12-31", True), ("2023", "FY", 2023, "2023-12-31", True), ("FY 23", "FY", 2023, "2023-12-31", True), ("2023-09-30", "FY", 2023, "2023-09-30", False),
    ("Dec 2023", "FY", 2023, "2023-12-31", False), ("Sep-23", "FY", 2023, "2023-09-30", False), ("Q1 2024", "Q1", 2024, "2024-03-31", True), ("1Q24", "Q1", 2024, "2024-03-31", True),
    ("2024Q3", "Q3", 2024, "2024-09-30", True), ("H1 2024", "H1", 2024, "2024-06-30", True), ("Q4-2023", "Q4", 2023, "2023-12-31", True)])
def test_a_column_header_is_a_period(label, period, year, end, guessed):
    got = manual.parse_period_label(label)
    assert (got["fiscal_period"], got["fiscal_year"], got["period_end"], got["guessed"]) == (period, year, end, guessed)


def test_a_number_in_a_row_of_figures_is_not_a_year_and_a_table_without_a_header_is_refused():
    for label in ("1500", "1850", "2101", "FY1850", "99999"):
        assert manual.parse_period_label(label) is None, label
    with pytest.raises(FundamentalsError, match="no header row of periods"):
        manual.parse_table("Revenue\t1500\t1600\nNet income\t150\t160")


def test_a_header_that_is_not_a_period_is_none_and_ttm_and_estimates_are_refused_with_the_reason():
    for label in ("Notes", "Change %", "", "abc"):
        assert manual.parse_period_label(label) is None
    with pytest.raises(ValueError, match="built by the application from the quarters"):
        manual.parse_period_label("TTM")
    for label in ("2025E", "FY2026 forecast", "2025 estimate"):
        with pytest.raises(ValueError, match="estimate or a forecast"):
            manual.parse_period_label(label)


# ------------------------------------------------------------------------------------------------------------------ reading a pasted table
def test_a_table_from_excel_is_read_into_periods_lines_dates_and_prices():
    got = manual.parse_table(TABLE)
    assert [c["label"] for c in got["columns"]] == ["FY2021", "FY2022", "FY2023"]
    assert [c["period_end"] for c in got["columns"]] == ["2021-12-31", "2022-12-31", "2023-12-31"]
    assert [c["filing_date"] for c in got["columns"]] == ["2022-02-10", "2023-02-09", "2024-02-08"]
    assert [c["currency"] for c in got["columns"]] == ["USD"] * 3
    assert got["values"]["revenue"] == [1000.0, 1200.0, 1500.0] and got["values"]["cost_of_revenue"] == [-600.0, -700.0, -850.0]
    assert got["values"]["capital_expenditure"] == [-80.0, -90.0, -110.0] and got["values"]["weighted_avg_shares_diluted"] == [100.0] * 3
    assert got["price"] == [20.0, 25.0, 30.0]
    assert got["unrecognised"] == [{"label": "A line nobody knows", "figures": 3}] and got["problems"] == [] and got["transposed"] is False


def test_the_line_names_of_reports_and_data_exports_are_recognised():
    for label, code in [("Net sales", "revenue"), ("Total Revenues", "revenue"), ("COGS", "cost_of_revenue"), ("Income from operations", "operating_income"), ("Pre-tax income", "income_before_tax"),
                        ("Net earnings", "net_income"), ("Diluted EPS", "eps_diluted"), ("Stockholders' equity", "shareholders_equity"), ("Cash from operations", "operating_cash_flow"),
                        ("CapEx", "capital_expenditure"), ("D&A", "depreciation_amortization"), ("Net cash used in financing activities", "financing_cash_flow"), ("revenue", "revenue"),
                        ("Total assets", "total_assets"), ("  total   ASSETS ", "total_assets")]:
        assert manual.line_for(label) == code, label
    assert manual.line_for("Something else entirely") is None


def test_a_csv_file_and_a_table_turned_the_other_way_are_read_too():
    csv_text = 'Item,FY2022,FY2023\nRevenue,"1,200","1,500"\nNet income,195,260\nNotes,see 10-K,\n'
    got = manual.parse_table(csv_text)
    assert got["values"]["revenue"] == [1200.0, 1500.0] and got["values"]["net_income"] == [195.0, 260.0] and got["unrecognised"][0]["label"] == "Notes"
    turned = "Period\tRevenue\tNet income\tTotal assets\nFY2022\t1200\t195\t1700\nFY2023\t1500\t260\t2000"
    got = manual.parse_table(turned)
    assert got["transposed"] is True and got["values"]["revenue"] == [1200.0, 1500.0] and got["values"]["total_assets"] == [1700.0, 2000.0]
    assert any("turned" in n for n in got["notes"])


def test_a_line_given_twice_keeps_the_first_and_a_bad_cell_is_reported_not_guessed():
    text = "Item\tFY2022\tFY2023\nNet income\t195\t260\nDepreciation\t65\t75\nNet income\t999\t999\nTotal assets\t1700\toops\n"
    got = manual.parse_table(text)
    assert got["values"]["net_income"] == [195.0, 260.0] and got["duplicates"] == ["Net income"]
    assert got["values"]["total_assets"] == [1700.0, None]
    assert got["problems"] == [{"label": "Total assets", "column": "FY2023", "cell": "oops", "message": "'oops' is not a number"}]


def test_a_ttm_column_and_a_notes_column_are_left_out_with_the_reason():
    got = manual.parse_table("Item\tFY2022\tTTM\tNotes\tFY2023\nRevenue\t1200\t1400\tx\t1500\n")
    assert [c["label"] for c in got["columns"]] == ["FY2022", "FY2023"] and got["values"]["revenue"] == [1200.0, 1500.0]
    assert any("TTM" in n and "quarters" in n for n in got["notes"]) and any("Notes" in n for n in got["notes"])


@pytest.mark.parametrize("text, why", [("", "paste the table first"), ("   \n", "paste the table first"), ("only one row", "needs a header row"), ("Item\tFY2023\nRevenue", "no row of the table is a line"),
                                       ("Item\tNotes\nRevenue\t5", "no header row of periods"), ("Item\tFY2023\nNothing known\t5", "no row of the table is a line"),
                                       (None, "paste the table first"), (5, "paste the table first")])
def test_a_table_that_cannot_be_read_says_why(text, why):
    with pytest.raises(FundamentalsError, match=why):
        manual.parse_table(text)


def test_a_table_too_large_is_refused():
    with pytest.raises(FundamentalsError, match="too large"):
        manual.parse_table("Item\tFY2023\n" + "Revenue\t1\n" * 30000)


# ------------------------------------------------------------------------------------------------------------------ checking an entry
def test_a_good_entry_is_scaled_to_currency_units_and_costs_are_kept_as_amounts():
    got = manual.check_entry(entry(years=(2023,), **{"periods": [period(2023, values={"cost_of_revenue": -850, "capital_expenditure": "(110)", "weighted_avg_shares_diluted": 100})]}), None, TODAY)
    assert got["ok"] and errors(got) == []
    p = got["periods"][0]
    assert p["values"]["revenue"] == 1500e6 and p["values"]["weighted_avg_shares_diluted"] == 100e6 and p["values"]["eps_diluted"] == 2.6       # money and shares scaled, per share never
    assert p["observations"]["cost_of_revenue"]["value"] == 850e6 and p["observations"]["capital_expenditure"]["value"] == 110e6          # a cost typed as a negative is the amount spent
    assert p["observations"]["gross_profit"]["flag"] == "ok" and p["observations"]["free_cash_flow"]["flag"] == "derived"
    assert {"free_cash_flow", "ebit", "ebitda", "total_debt", "cash", "shares_outstanding"} <= set(p["derived"]) and "gross_profit" not in p["derived"]


def test_units_can_differ_between_amounts_and_share_counts():
    got = manual.check_entry({"units": {"money": "thousands", "shares": "units"}, "currency": "JPY", "periods": [{"fiscal_period": "FY", "period_end": "2023-03-31", "filing_date": "2023-05-12",
                                                                                                                  "values": {"revenue": 5000, "shares_outstanding": 2_000_000, "eps_basic": 12.5}}]}, None, TODAY)
    p = got["periods"][0]
    assert p["values"] == {"revenue": 5_000_000.0, "shares_outstanding": 2_000_000.0, "eps_basic": 12.5} and p["currency"] == "JPY"


@pytest.mark.parametrize("period_, why", [
    ({"period_end": "2030-12-31"}, "in the future"), ({"period_end": "2023-12-31", "filing_date": "2023-06-30"}, "before the period end"),
    ({"period_end": "2023-12-31", "filing_date": "2031-01-01"}, "in the future"), ({"period_end": ""}, "period end date is missing"), ({"period_end": "31/31/2023"}, "not a real date"),
    ({"period_end": "2023-12-31", "fiscal_period": "TTM"}, "period type must be one of"), ({"period_end": "2023-12-31", "fiscal_year": 2019}, "does not fit"),
    ({"period_end": "2023-12-31", "fiscal_year": "abc"}, "whole number"), ({"period_end": "2023-12-31", "currency": "DOLLARS"}, "currency is missing"),
    ({"period_end": "2023-12-31", "price": -5}, "must be positive"), ({"period_end": "2023-12-31", "price": "abc"}, "share price"),
    ({"period_end": "2023-12-31", "values": {"revenue": "abc"}}, "Revenue: 'abc' is not a number"), ({"period_end": "2023-12-31", "values": {"nonsense": 5}}, "not a line this application stores"),
    ({"period_end": "2023-12-31", "values": {"total_assets": -5}}, "Total assets cannot be negative"), ({"period_end": "2023-12-31", "values": {"revenue": ""}}, "no figures"),
    ({"period_end": "2023-12-31", "values": {}}, "no figures")])
def test_a_period_that_cannot_be_is_an_error_that_names_the_period_and_the_reason(period_, why):
    base = {"fiscal_period": "FY", "period_end": "2023-12-31", "filing_date": "2024-02-08", "currency": "USD", "values": {"revenue": 1}}
    got = manual.check_entry({"units": {}, "periods": [{**base, **period_}]}, None, TODAY)
    assert not got["ok"] and any(why in m for m in errors(got)), errors(got)


def test_a_period_entered_twice_and_a_malformed_request_are_refused():
    got = manual.check_entry(entry(years=(2023, 2023)), None, TODAY)
    assert not got["ok"] and any("appears twice" in m for m in errors(got))
    for bad, why in [([], "dictionary"), ({"periods": []}, "at least one period"), ({"periods": "x"}, "at least one period"), ({"periods": [5]}, "each period must be a dictionary"),
                     ({"periods": [{}] * 61}, "at most 60"), ({"units": {"money": "lakhs"}, "periods": [{}]}, "must be one of"), ({"units": 5, "periods": [{}]}, "units must"),
                     ({"periods": [{"period_end": "2023-12-31", "values": [1]}], "currency": "USD"}, "dictionary of line name")]:
        with pytest.raises(FundamentalsError, match=why):
            manual.check_entry(bad, None, TODAY)


def test_the_currency_comes_from_the_column_the_entry_or_the_company():
    one = {"fiscal_period": "FY", "period_end": "2023-12-31", "filing_date": "2024-02-08", "values": {"revenue": 1}}
    assert manual.check_entry({"periods": [one]}, None, TODAY)["ok"] is False
    assert manual.check_entry({"periods": [one]}, {"currency": "eur"}, TODAY)["periods"][0]["currency"] == "EUR"
    assert manual.check_entry({"currency": "gbp", "periods": [one]}, {"currency": "EUR"}, TODAY)["periods"][0]["currency"] == "GBP"
    assert manual.check_entry({"currency": "gbp", "periods": [{**one, "currency": "chf"}]}, None, TODAY)["periods"][0]["currency"] == "CHF"


def test_a_missing_publication_date_is_a_warning_that_says_what_will_be_assumed():
    got = manual.check_entry(entry(**{"periods": [period(2023, filing_date=None)]}), None, TODAY)
    assert got["ok"] and any("120 days after the period end" in m and "estimated" in m for m in warnings(got))
    quarter = {"fiscal_period": "Q1", "period_end": "2023-03-31", "currency": "USD", "values": {"revenue": 1}}
    assert any("60 days" in m for m in warnings(manual.check_entry({"periods": [quarter]}, None, TODAY)))
    late = manual.check_entry(entry(**{"periods": [period(2021, filing_date="2023-06-01")]}), None, TODAY)
    assert any("days after the period it covers" in m for m in warnings(late))


def test_statements_that_do_not_add_up_are_warnings_and_figures_that_do_are_not():
    assert warnings(manual.check_entry(entry(), None, TODAY)) == []
    bad = manual.check_entry(entry(**{"periods": [period(2023, values={"total_assets": 2500, "gross_profit": 900, "free_cash_flow": 10, "current_assets": 5000, "cash_and_equivalents": 9000})]}), None, TODAY)
    text = " ".join(warnings(bad))
    assert bad["ok"]                                                                              # a statement that does not add up is a warning: the person may know better than the check
    assert "differ from total liabilities plus equity" in text and "revenue less cost of revenue" in text and "free cash flow" in text
    assert "current assets exceed total assets" in text and "cash exceeds current assets" in text


def test_a_rounding_difference_in_the_unit_the_figures_are_typed_in_is_not_flagged():
    got = manual.check_entry(entry(**{"periods": [period(2023, values={"total_assets": 2001, "gross_profit": 651})]}), None, TODAY)    # one million out in a company of two billion
    assert warnings(got) == []


def test_a_unit_that_does_not_match_between_amounts_and_share_counts_is_noticed():
    got = manual.check_entry(entry(**{"units": {"money": "millions", "shares": "units"}}), None, TODAY)      # 100 shares, not 100 million
    assert any("units of the amounts, the share count and the per-share figure may not match" in m for m in warnings(got))
    assert not any("may not match" in m for m in warnings(manual.check_entry(entry(), None, TODAY)))
    odd = manual.check_entry(entry(**{"periods": [period(2023, values={"revenue": 5_000_000})]}), None, TODAY)
    assert any("same unit" in m for m in warnings(odd))


def test_the_lines_a_ratio_needs_are_listed_when_missing_and_a_gap_in_the_years_is_noted():
    got = manual.check_entry(entry(years=(2021, 2023)), None, TODAY)
    infos = [i["message"] for i in got["issues"] if i["level"] == "info"]
    assert any("not given:" in m and "Interest expense" in m for m in infos) and any("days apart" in m for m in infos)


def test_a_company_to_create_is_checked():
    assert manual.clean_company({"ticker": " nvda "})["ticker"] == "NVDA" and manual.clean_company({"ticker": "7203.T", "country": "jp", "currency": "jpy"})["country"] == "JP"
    for bad, why in [(5, "dictionary"), ({}, "ticker"), ({"ticker": "bad ticker"}, "ticker"), ({"ticker": "A" * 25}, "ticker"), ({"ticker": "X", "country": "USA"}, "two-letter"),
                     ({"ticker": "X", "currency": "US"}, "three-letter"), ({"ticker": "X", "name": "n" * 121}, "longer than"), ({"ticker": "X", "exchange": "no way"}, "exchange"),
                     ({"ticker": "X", "price_symbol": "a b"}, "price symbol")]:
        with pytest.raises(FundamentalsError, match=why):
            manual.clean_company(bad)


# ------------------------------------------------------------------------------------------------------------------ saving
def stored(store, sid=1):
    periods, obs = store.periods([sid]), store.observations([sid])
    return periods, obs


def test_a_saved_entry_is_the_source_manual_with_its_dates_and_scaled_figures(store):
    out = manual.save(store, entry(), today=TODAY)
    assert out["created_company"] and out["added"] == 3 and out["corrected"] == 0 and out["ticker"] == "TEST1" and out["observations"] > 40 and out["metrics_written"] > 100
    info = store.get_security(out["security_id"])
    assert (info["ticker"], info["source"], info["country"], info["currency"], info["company_name"]) == ("TEST1", "manual", "US", "USD", "Test Co")
    periods, obs = stored(store, out["security_id"])
    assert set(periods["provider"]) == {"manual"} and set(periods["pit_quality"]) == {"filed"} and periods["vintage"].tolist() == [0, 0, 0]
    assert periods["filing_date"].dt.strftime("%Y-%m-%d").tolist() == ["2022-02-10", "2023-02-10", "2024-02-10"] and set(periods["reporting_currency"]) == {"USD"}
    got = obs[obs["period_id"] == periods["period_id"].iloc[2]].set_index("metric_code")
    assert got.loc["revenue", "value"] == 1500e6 and got.loc["capital_expenditure", "value"] == 110e6 and got.loc["eps_diluted", "value"] == 2.6 and got.loc["weighted_avg_shares_diluted", "value"] == 100e6
    assert got.loc["revenue", "quality_flag"] == "ok" and got.loc["free_cash_flow", "quality_flag"] == "derived" and got.loc["free_cash_flow", "source_metric"].startswith("derived:")
    assert (got["provider"] == "manual").all()
    raw = store.raw_answers(out["security_id"], "manual", "manual_entry")
    assert len(raw) == 1 and raw[0]["payload"]["units"] == {"money": "millions", "shares": "millions"} and raw[0]["payload"]["periods"][2]["typed"]["revenue"] == 1500.0     # what was typed is kept


def test_a_period_without_a_date_is_estimated_and_the_engine_never_uses_it_on_its_period_end(store):
    out = manual.save(store, entry(years=(2023,), **{"periods": [period(2023, filing_date=None)]}), today=TODAY)
    periods, _ = stored(store, out["security_id"])
    assert periods["pit_quality"].tolist() == ["estimated"]
    fy = PeriodTables(store, [out["security_id"]], PitSettings()).table("fy")
    assert bool(fy["pit_estimated"].iloc[0]) and fy["available_date"].iloc[0] == pd.Timestamp("2023-12-31") + pd.Timedelta(days=121)          # period end + 120 days + the filing lag of a day


def test_the_engine_sees_the_entered_figures_from_the_day_after_they_were_published(store):
    out = manual.save(store, entry(), today=TODAY)
    fy = PeriodTables(store, [out["security_id"]], PitSettings()).table("fy")
    assert fy["available_date"].dt.strftime("%Y-%m-%d").tolist() == ["2022-02-11", "2023-02-11", "2024-02-11"] and not fy["pit_estimated"].any()
    assert fy["revenue"].tolist() == [1000e6, 1200e6, 1500e6] and fy["capital_expenditure"].tolist() == [80e6, 90e6, 110e6]
    from src.fundamentals import metrics
    values = metrics.compute(PeriodTables(store, [out["security_id"]], PitSettings()), "fy")
    assert values["gross_margin"].round(6).tolist() == [0.4, round(500 / 1200, 6), round(650 / 1500, 6)] and values["revenue_growth"].round(6).tolist()[1:] == [0.2, 0.25]
    assert values["revenue_growth"].isna().iloc[0] and round(values["roe"].iloc[2], 6) == 0.26


def test_the_derived_metrics_are_stored_with_the_company(store):
    out = manual.save(store, entry(), today=TODAY)
    with store.db.transaction() as conn:
        rows = conn.execute("SELECT basis, metric_name, metric_value FROM fundamental_metrics WHERE security_id = ?", (out["security_id"],)).fetchall()
    assert {r["basis"] for r in rows} == {"fy", "ttm"} and len(rows) == out["metrics_written"]
    assert round(next(r["metric_value"] for r in rows if r["metric_name"] == "net_margin" and abs(r["metric_value"] - 260 / 1500) < 1e-9), 6) == round(260 / 1500, 6)


def test_a_ticker_already_in_the_database_is_refused_and_an_unknown_company_id_too(store):
    manual.save(store, entry(), today=TODAY)
    with pytest.raises(FundamentalsError, match="already in your database"):
        manual.save(store, entry(), today=TODAY)
    with pytest.raises(FundamentalsError, match="no company with id 99"):
        manual.save(store, {**entry(), "security_id": 99, "company": None}, today=TODAY)
    with pytest.raises(FundamentalsError, match="ticker"):
        manual.save(store, {"periods": entry()["periods"], "units": {}}, today=TODAY)
    with pytest.raises(FundamentalsError, match="security_id must be a whole number"):
        manual.save(store, {**entry(), "security_id": "abc"}, today=TODAY)


def test_an_entry_with_an_error_stores_nothing_not_even_the_company(store):
    bad = entry(**{"periods": [period(2021), period(2022, values={"total_assets": -5})]})
    with pytest.raises(FundamentalsError, match="nothing was saved: FY to 2022-12-31: Total assets cannot be negative"):
        manual.save(store, bad, today=TODAY)
    assert store.counts()["companies"] == 0 and store.counts()["observations"] == 0
    with pytest.raises(FundamentalsError, match="nothing was saved"):
        manual.save(store, entry(**{"periods": [period(2022, filing_date="2021-01-01"), period(2023, values={"revenue": "x"})]}), today=TODAY)       # and says how many more
    assert store.counts()["companies"] == 0


def test_a_new_company_can_carry_the_symbol_its_prices_come_from(store):
    out = manual.save(store, entry(**{"company": {**COMPANY, "price_symbol": "tst1"}}), today=TODAY)
    ids = store.identifiers(out["security_id"])
    assert ids[(ids["provider"] == "yahoo") & (ids["id_type"] == "price_symbol")]["provider_id"].tolist() == ["TST1"]


def test_saving_again_corrects_in_place_and_a_cleared_cell_removes_the_line(store):
    sid = manual.save(store, entry(), today=TODAY)["security_id"]
    again = {"security_id": sid, "units": {"money": "millions", "shares": "millions"}, "periods": [period(2021, values={"revenue": 1010, "gross_profit": None, "inventory": 90}), period(2022), period(2023)]}
    out = manual.save(store, again, today=TODAY)
    assert (out["added"], out["corrected"], out["unchanged"], out["restated"], out["created_company"]) == (0, 1, 2, 0, False)
    assert [p["outcome"] for p in out["periods"]] == ["corrected", "unchanged", "unchanged"]
    periods, obs = stored(store, sid)
    assert len(periods) == 3 and periods["vintage"].tolist() == [0, 0, 0] and store.counts()["restated_vintages"] == 0          # a typing mistake leaves no history
    first = obs[obs["period_id"] == periods["period_id"].iloc[0]].set_index("metric_code")["value"]
    assert first["revenue"] == 1010e6 and first["inventory"] == 90e6 and first["gross_profit"] == 1010e6 - 600e6              # the cleared gross profit is worked out again from revenue less cost
    with store.db.transaction() as conn:
        n = conn.execute("SELECT COUNT(*) FROM fundamental_metrics WHERE security_id = ?", (sid,)).fetchone()[0]
    assert n == out["metrics_written"]                                                                                           # no ratio of the old figures is left behind


def test_saving_the_same_thing_twice_changes_nothing(store):
    sid = manual.save(store, entry(), today=TODAY)["security_id"]
    before = store.counts()
    out = manual.save(store, {**entry(), "company": None, "security_id": sid}, today=TODAY)
    assert (out["added"], out["corrected"], out["unchanged"]) == (0, 0, 3) and store.counts()["observations"] == before["observations"] and store.counts()["raw_payloads"] == before["raw_payloads"]


def test_a_restatement_is_a_new_vintage_public_only_from_its_date_and_the_engine_keeps_the_first_filing(store):
    sid = manual.save(store, entry(), today=TODAY)["security_id"]
    restated = {"security_id": sid, "units": {"money": "millions", "shares": "millions"}, "periods": [period(2022, values={"revenue": 1250, "gross_profit": 550, "net_income": 200})]}
    out = manual.save(store, restated, mode="restate", restated_on="2024-02-08", today=TODAY)
    assert (out["restated"], out["added"], out["unchanged"]) == (1, 0, 0) and out["periods"] == [{"label": "FY to 2022-12-31", "outcome": "restated"}]
    periods, obs = stored(store, sid)
    two = periods[(periods["fiscal_period"] == "FY") & (periods["period_end"] == "2022-12-31")].sort_values("vintage")
    assert two["vintage"].tolist() == [0, 1] and two["pit_quality"].tolist() == ["filed", "restated"]
    assert two["publication_date"].iloc[1] == pd.Timestamp("2024-02-08") and store.counts()["restated_vintages"] == 1
    new = obs[obs["period_id"] == two["period_id"].iloc[1]].set_index("metric_code")
    assert new.loc["revenue", "value"] == 1250e6 and bool(new.loc["revenue", "is_restated"])
    fy = PeriodTables(store, [sid], PitSettings()).table("fy")
    assert fy.loc[fy["period_end"] == "2022-12-31", "revenue"].iloc[0] == 1200e6                                                  # a restatement changes the raw panels, never what was first public
    again = manual.save(store, restated, mode="restate", restated_on="2024-03-01", today=TODAY)
    assert again["restated"] == 0 and again["unchanged"] == 1 and again["periods"][0]["outcome"] == "unchanged"                      # the same restatement again is not a third version
    new_year = {"security_id": sid, "units": {"money": "millions", "shares": "millions"}, "periods": [{**period(2023), "period_end": "2024-12-31", "filing_date": "2025-02-10"}]}
    assert manual.save(store, new_year, mode="restate", restated_on="2025-03-01", today=TODAY)["periods"] == [{"label": "FY to 2024-12-31", "outcome": "added"}]


def test_a_restatement_needs_a_date_that_can_be_true(store):
    sid = manual.save(store, entry(), today=TODAY)["security_id"]
    body = {"security_id": sid, "units": {}, "periods": [period(2022, values={"revenue": 1250})]}
    for date_, why in [(None, "needs the date"), ("", "needs the date"), ("2030-01-01", "in the future"), ("2022-06-01", "before the period it restates"), ("soon", "not a date")]:
        with pytest.raises(FundamentalsError, match=why):
            manual.save(store, body, mode="restate", restated_on=date_, today=TODAY)
    with pytest.raises(FundamentalsError, match="mode must be"):
        manual.save(store, body, mode="overwrite", today=TODAY)
    assert store.counts()["restated_vintages"] == 0


def test_quarters_are_stored_and_four_of_them_give_a_trailing_twelve_months(store):
    quarters = []
    for i, end in enumerate(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"], 1):
        quarters.append({"fiscal_period": f"Q{i}", "period_end": end, "filing_date": f"{end[:7]}-28" if i < 4 else "2024-02-08", "currency": "USD",
                         "values": {"revenue": 100 * i, "net_income": 10 * i, "total_assets": 1000 + 10 * i}})
    quarters[0]["filing_date"], quarters[1]["filing_date"], quarters[2]["filing_date"] = "2023-05-05", "2023-08-04", "2023-11-03"
    out = manual.save(store, {"company": dict(COMPANY), "units": {"money": "millions", "shares": "millions"}, "periods": quarters}, today=TODAY)
    assert out["added"] == 4
    ttm = PeriodTables(store, [out["security_id"]], PitSettings()).table("ttm")
    assert ttm["revenue"].iloc[-1] == (100 + 200 + 300 + 400) * 1e6 and ttm["net_income"].iloc[-1] == 100e6


# ------------------------------------------------------------------------------------------------------------------ reading back and removing
def test_entries_are_read_back_as_stored_with_a_unit_that_reads_well(store):
    sid = manual.save(store, entry(), today=TODAY)["security_id"]
    got = manual.entries(store, sid)
    assert got["ticker"] == "TEST1" and got["suggested_unit"] == "millions" and [p["period_end"] for p in got["periods"]] == ["2021-12-31", "2022-12-31", "2023-12-31"]
    last = got["periods"][2]
    assert last["values"]["revenue"] == 1500e6 and last["currency"] == "USD" and last["filing_date"] == "2024-02-10" and last["quality"] == "filed" and last["vintage"] == 0
    small = manual.save(store, {"company": {"ticker": "TINY"}, "units": {"money": "thousands"}, "currency": "USD", "periods": [{"fiscal_period": "FY", "period_end": "2023-12-31", "values": {"revenue": 40}}]},
                        today=TODAY)
    assert manual.entries(store, small["security_id"])["suggested_unit"] == "thousands"
    assert manual.entries(store, sid)["periods"][0]["fiscal_year"] == 2021
    with pytest.raises(FundamentalsError, match="no company with id 77"):
        manual.entries(store, 77)


def test_the_latest_vintage_is_what_is_read_back_for_correction(store):
    sid = manual.save(store, entry(), today=TODAY)["security_id"]
    manual.save(store, {"security_id": sid, "units": {"money": "millions"}, "periods": [period(2022, values={"revenue": 1250})]}, mode="restate", restated_on="2024-02-08", today=TODAY)
    got = [p for p in manual.entries(store, sid)["periods"] if p["period_end"] == "2022-12-31"]
    assert len(got) == 1 and got[0]["vintage"] == 1 and got[0]["quality"] == "restated" and got[0]["values"]["revenue"] == 1250e6


def test_a_period_entered_by_hand_can_be_removed_with_its_ratios_and_a_providers_is_left_alone(store):
    sid = manual.save(store, entry(), today=TODAY)["security_id"]
    store.store_financials(sid, "sec_edgar", [PeriodStatement("FY", "2023-12-31", {"us-gaap:Revenues": 1_499_000_000.0}, 2023, filing_date="2024-02-09")])
    out = manual.delete_period(store, sid, "fy", "2023-12-31")
    assert out["deleted"] == 1 and out["fiscal_period"] == "FY"
    assert [p["period_end"] for p in manual.entries(store, sid)["periods"]] == ["2021-12-31", "2022-12-31"]
    periods, _ = stored(store, sid)
    assert sorted(periods["provider"].tolist()) == ["manual", "manual", "sec_edgar"]
    with store.db.transaction() as conn:
        left = conn.execute("SELECT COUNT(*) FROM fundamental_metrics fm JOIN financial_periods p ON p.id = fm.financial_period_id WHERE fm.security_id = ? AND p.period_end = '2023-12-31' "
                            "AND p.provider = 'manual'", (sid,)).fetchone()[0]
    assert left == 0
    for args, why in [((sid, "FY", "2023-12-31"), "no period FY to 2023-12-31"), ((sid, "XX", "2023-12-31"), "period type"), ((sid, "FY", "nope"), "period end"), ((sid, "FY", ""), "needed"), ((55, "FY", "2022-12-31"), "no company")]:
        with pytest.raises(FundamentalsError, match=why):
            manual.delete_period(store, *args)


# ------------------------------------------------------------------------------------------------------------------ with a provider's figures
def with_provider(store):
    sid = store.upsert_security(CompanyProfile(provider="sec_edgar", provider_symbol="TEST9", ticker="TEST9", name="Nine Inc", country="US", currency="USD", sector="Technology"))
    store.store_financials(sid, "sec_edgar", [PeriodStatement("FY", "2022-12-31", {"us-gaap:Revenues": 1_190_000_000.0, "us-gaap:Assets": 1_700_000_000.0, "us-gaap:NetIncomeLoss": 195_000_000.0}, 2022,
                                                              filing_date="2023-02-09")])
    return sid


def test_where_a_provider_holds_the_period_it_leads_and_an_earlier_entry_fills_only_the_lines_it_lacks(store):
    sid = with_provider(store)                                                                           # the provider's report of 2022 was filed on 2023-02-09
    out = manual.save(store, {"security_id": sid, "units": {"money": "millions", "shares": "millions"}, "periods": [period(2022, filing_date="2023-02-01"), period(2023)]}, today=TODAY)
    assert out["added"] == 2
    fy = PeriodTables(store, [sid], PitSettings()).table("fy")
    row = fy[fy["period_end"] == "2022-12-31"].iloc[0]
    assert row["revenue"] == 1_190_000_000.0 and row["gross_profit"] == 500e6 and row["operating_cash_flow"] == 260e6        # the provider's revenue, this entry's gross profit and cash flow
    assert fy[fy["period_end"] == "2023-12-31"].iloc[0]["revenue"] == 1500e6                                                  # a year the provider does not have is wholly this entry's
    [overlap] = out["overlaps"]
    assert overlap["label"] == "FY to 2022-12-31" and overlap["provider"] == "sec_edgar" and overlap["compared"] == 3 and overlap["differ_count"] == 1
    assert overlap["differ"] == [{"line": "revenue", "label": "Revenue", "mine": 1200e6, "theirs": 1190e6}]
    assert overlap["used_in_studies"] is True and overlap["only_mine"] > 15 and overlap["mine_public"] == "2023-02-02" and overlap["theirs_public"] == "2023-02-10"


def test_a_line_published_after_the_providers_report_is_not_used_to_fill_its_gaps_and_the_overlap_says_so(store):
    sid = with_provider(store)
    out = manual.save(store, {"security_id": sid, "units": {"money": "millions", "shares": "millions"}, "periods": [period(2022, filing_date="2023-03-01")]}, today=TODAY)     # a month after the provider
    fy = PeriodTables(store, [sid], PitSettings()).table("fy")
    row = fy[fy["period_end"] == "2022-12-31"].iloc[0]
    assert row["revenue"] == 1_190_000_000.0 and pd.isna(row["gross_profit"])                                      # it was not public when the provider's report was: never used as if it had been
    assert out["overlaps"][0]["used_in_studies"] is False and out["overlaps"][0]["mine_public"] == "2023-03-02"


def test_a_manual_company_is_not_counted_as_needing_a_sync_and_has_no_provider_coverage(store, tmp_path):
    manual.save(store, entry(), today=TODAY)
    service = FundamentalsService(tmp_path, db_path=tmp_path / "f.db")
    assert service.data_status()["companies_stale"] == 0
    with pytest.raises(FundamentalsError, match="typed in by hand"):
        service.coverage(1)


def test_writes_are_one_at_a_time(store):
    sid = manual.save(store, entry(), today=TODAY)["security_id"]
    body = {"security_id": sid, "units": {"money": "millions"}, "periods": [period(2021), period(2022)]}
    done, problems = [], []

    def save():
        try:
            manual.save(store, body, today=TODAY)
            done.append(1)
        except Exception as error:                                                               # pragma: no cover - only if the writes met
            problems.append(error)

    def remove():
        try:
            manual.delete_period(store, sid, "FY", "2023-12-31")
            done.append(1)
        except Exception as error:                                                               # pragma: no cover
            problems.append(error)

    threads = [threading.Thread(target=f) for f in (save, remove, save, save)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert problems == [] and len(done) == 4


# ------------------------------------------------------------------------------------------------------------------ the preview
def preview(store, tmp_path, raw, **kw):
    return manual.preview(store, raw, tmp_path, TODAY, **kw)


def test_the_preview_works_out_the_ratios_on_a_copy_and_stores_nothing(store, tmp_path):
    got = preview(store, tmp_path, entry())
    assert got["ok"] and store.counts()["companies"] == 0 and store.counts()["observations"] == 0 and store.counts()["raw_payloads"] == 0
    r = got["ratios"]
    assert [c["label"] for c in r["columns"]] == ["FY 2021", "FY 2022", "FY 2023"] and r["defined"] > 100 and r["computed"] > 60
    rows = {x["name"]: x for x in r["rows"]}
    assert [round(v, 6) for v in rows["gross_margin"]["values"]] == [0.4, round(500 / 1200, 6), round(650 / 1500, 6)] and rows["revenue_growth"]["values"][0] is None
    assert rows["revenue_growth"]["values"][1:] == [pytest.approx(0.2), pytest.approx(0.25)] and rows["net_debt"]["values"][2] == pytest.approx(30e6)
    assert "piotroski_f" in rows and rows["gross_margin"]["family"] == "profitability" and rows["gross_margin"]["text"]
    families = [x["family"] for x in r["rows"]]
    assert families == sorted(families, key=r["families"].index)                                   # grouped by family
    assert got["company"]["profile"]["ticker"] == "TEST1" and len(got["company"]["annual"]) == 3 and got["company"]["piotroski"] and got["company"]["dupont"]
    assert got["derived"]["FY to 2023-12-31"] and got["multiples"] == [] and got["valuation"] is None and got["overlaps"] == []


def test_the_preview_of_an_entry_with_an_error_says_so_and_calculates_nothing(store, tmp_path):
    got = preview(store, tmp_path, entry(**{"periods": [period(2022, values={"total_assets": -1})]}))
    assert got["ok"] is False and "ratios" not in got and any("cannot be negative" in i["message"] for i in got["issues"]) and store.counts()["companies"] == 0
    with pytest.raises(FundamentalsError, match="already in your database"):
        manual.save(store, entry(), today=TODAY)
        preview(store, tmp_path, entry())


def test_a_typed_price_gives_yields_multiples_and_the_valuation_bars_but_is_not_stored(store, tmp_path):
    raw = entry(**{"periods": [period(2022, price=25), period(2023, price=30)]})
    got = preview(store, tmp_path, raw)
    assert [m["label"] for m in got["multiples"]] == ["FY to 2022-12-31", "FY to 2023-12-31"]
    rows = {r["name"]: r for r in got["multiples"][1]["rows"]}
    assert rows["market_cap"]["value"] == 3000e6 and rows["enterprise_value"]["value"] == pytest.approx(3000e6 + 450e6 - 420e6)           # market cap + debt - cash
    assert rows["earnings_yield"]["value"] == pytest.approx(260 / 3000) and rows["pe"]["value"] == pytest.approx(3000 / 260) and rows["book_to_price"]["value"] == pytest.approx(1000 / 3000)
    assert rows["fcf_yield"]["value"] == pytest.approx(230 / 3000) and rows["ev_ebitda"]["value"] == pytest.approx(3030 / 415)            # EBITDA = operating income + depreciation
    assert got["valuation"] and got["valuation"]["price"]["value"] == 30 and store.counts()["price_rows"] == 0
    assert any(r["name"] == "pe" for r in got["multiples"][0]["rows"])


def test_a_loss_or_a_missing_share_count_gives_missing_values_not_huge_ones():
    loss = {"net_income": -50e6, "shares_outstanding": 100e6, "total_debt": 10e6, "cash_and_equivalents": 5e6, "shareholders_equity": -20e6, "revenue": 500e6}
    rows = {r["name"]: r["value"] for r in manual.multiples(loss, 10.0)["rows"]}
    assert rows["earnings_yield"] == pytest.approx(-0.05) and rows["pe"] is None and rows["book_to_price"] == pytest.approx(-0.02) and rows["pb"] is None and rows["sales_to_price"] == pytest.approx(0.5)
    assert "error" in manual.multiples({"net_income": 1.0}, 10.0)
    no_debt = {r["name"]: r["value"] for r in manual.multiples({"shares_outstanding": 10.0, "net_income": 5.0}, 10.0)["rows"]}
    assert no_debt["enterprise_value"] is None and no_debt["ev_ebitda"] is None


def test_quarterly_columns_get_no_multiples_and_the_preview_says_why(store, tmp_path):
    raw = {"company": dict(COMPANY), "units": {"money": "millions"}, "currency": "USD", "periods": [{"fiscal_period": "Q1", "period_end": "2023-03-31", "filing_date": "2023-05-05", "price": 20,
                                                                                                       "values": {"revenue": 100, "net_income": 10}}]}
    got = preview(store, tmp_path, raw)
    assert got["ok"] and got["multiples"] == [] and "annual periods only" in got["multiples_note"]


def test_the_preview_of_an_existing_company_joins_its_history_so_a_new_year_has_a_growth_rate(store, tmp_path):
    sid = manual.save(store, entry(years=(2021, 2022)), today=TODAY)["security_id"]
    got = preview(store, tmp_path, {"security_id": sid, "units": {"money": "millions", "shares": "millions"}, "periods": [period(2023)]})
    assert [c["label"] for c in got["ratios"]["columns"]] == ["FY 2023"]
    rows = {x["name"]: x for x in got["ratios"]["rows"]}
    assert rows["revenue_growth"]["values"] == [pytest.approx(0.25)]                                 # against 2022, which was already stored
    assert len(got["company"]["annual"]) == 3 and got["existing"]["ticker"] == "TEST1"
    assert store.counts()["annual_periods"] == 2                                                      # nothing was added


def test_a_period_typed_again_replaces_the_stored_one_in_the_preview(store, tmp_path):
    sid = manual.save(store, entry(), today=TODAY)["security_id"]
    got = preview(store, tmp_path, {"security_id": sid, "units": {"money": "millions", "shares": "millions"}, "periods": [period(2023, values={"revenue": 3000, "gross_profit": 1000})]})
    rows = {x["name"]: x for x in got["ratios"]["rows"]}
    assert rows["gross_margin"]["values"] == [pytest.approx(1000 / 3000)] and rows["revenue_growth"]["values"] == [pytest.approx(3000 / 1200 - 1)]
    assert len(got["company"]["annual"]) == 3                                                         # not four: the old 2023 was left out of the copy


def test_the_preview_shows_where_a_provider_disagrees_and_the_last_close(store, tmp_path):
    sid = with_provider(store)
    store.store_prices(pd.DataFrame({"security_id": sid, "date": ["2024-01-02", "2024-01-03"], "adj_close": [9.0, 9.5], "close": [10.0, 10.5]}))
    got = preview(store, tmp_path, {"security_id": sid, "units": {"money": "millions", "shares": "millions"}, "periods": [period(2022)]})
    assert got["overlaps"][0]["provider"] == "sec_edgar" and got["overlaps"][0]["differ_count"] == 1 and got["last_close"] == {"date": "2024-01-03", "close": 10.5}


def test_a_restatement_preview_says_what_studies_will_and_will_not_see(store, tmp_path):
    sid = manual.save(store, entry(), today=TODAY)["security_id"]
    got = preview(store, tmp_path, {"security_id": sid, "units": {"money": "millions"}, "periods": [period(2022, values={"revenue": 1250})]}, mode="restate")
    assert "new vintage" in got["mode_note"] and "first filed" in got["mode_note"]


def test_the_template_lists_every_line_once_by_statement_with_the_ones_that_fill_themselves_in():
    t = manual.template()
    codes = [i["code"] for s in t["statements"] for i in s["items"]]
    assert sorted(codes) == sorted(canonical.METRICS) and len(codes) == len(set(codes)) and [s["key"] for s in t["statements"]] == ["income", "balance", "cashflow", "shares"]
    items = {i["code"]: i for s in t["statements"] for i in s["items"]}
    assert items["gross_profit"]["auto"] and items["free_cash_flow"]["auto"] and items["revenue"]["auto"] is None and items["revenue"]["core"] and not items["goodwill"]["core"]
    assert items["capital_expenditure"]["magnitude"] and not items["revenue"]["magnitude"] and items["eps_diluted"]["unit"] == "per_share" and items["shares_outstanding"]["unit"] == "shares"
    assert t["units"] == ["units", "thousands", "millions", "billions"] and "FY" in t["fiscal_periods"] and len(t["rules"]) >= 4
    assert all(manual.line_for(i["label"]) == i["code"] or manual.line_for(i["code"]) == i["code"] for i in items.values())           # every line can be named in a paste
