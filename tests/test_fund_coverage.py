"""Which fiscal years a company has in the database and which it lacks: each year is held, missing, found empty when asked for, due or not yet due, judged from the calendar and the requests made; a year
that is held is never asked for again, one request covers the years picked (the price is per request, not per year), a year the provider sent nothing for is not offered again unasked, and the last
request is told in words so that a company with few years can say why."""

from __future__ import annotations

import json
from datetime import date

import pytest

from src.fundamentals import coverage as C
from src.fundamentals.errors import FundamentalsError, NotFound, QuotaExhausted
from tests.test_fund_data import Clock, _service

TODAY = date(2026, 10, 21)
LAG = 150


def held(*years, end="12-31", filed="02-15"):
    return [{"fiscal_year": y, "period_end": f"{y}-{end}", "filing_date": f"{y + 1}-{filed}"} for y in years]


def states(cells, lo, hi):
    return {c["year"]: c["state"] for c in cells if lo <= c["year"] <= hi}


# ------------------------------------------------------------------------------------------------------------------------ the grid
def test_a_year_is_held_missing_due_or_not_yet_due_by_the_calendar_and_the_grid_reaches_back_twenty_five_years():
    g = C.grid(held(*range(2019, 2025)), [], TODAY, LAG)
    assert (g["first"], g["last"], g["earliest_held"], g["latest_held"]) == (2001, 2026, 2019, 2024)
    assert states(g["cells"], 2001, 2026) == {**{y: "missing" for y in range(2001, 2019)}, **{y: "held" for y in range(2019, 2025)}, 2025: "due", 2026: "notyet"}
    assert g["counts"] == {"held": 6, "missing": 18, "empty": 0, "due": 1, "notyet": 1}
    cell = {c["year"]: c for c in g["cells"]}
    assert cell[2019]["period_end"] == "2019-12-31" and cell[2019]["filed"] == "2020-02-15" and cell[2025]["expected_filing"] == "2026-05-30" and cell[2026]["expected_filing"] == "2027-05-30"
    assert cell[2018]["period_end"] is None and cell[2018]["expected_filing"] is None and cell[2018]["asked_on"] is None


def test_a_gap_between_two_held_years_is_missing_and_a_year_is_due_on_the_day_its_filing_is_expected_not_the_day_before():
    g = C.grid(held(2019, 2021, 2022), [], TODAY, LAG, depth=10)
    assert states(g["cells"], 2016, 2023) == {2016: "missing", 2017: "missing", 2018: "missing", 2019: "held", 2020: "missing", 2021: "held", 2022: "held", 2023: "due"}
    assert (date(2024, 5, 29) - date(2023, 12, 31)).days == LAG                                       # FY2023 ended on 2023-12-31; its filing is expected 150 days on
    on = C.grid(held(2022), [], date(2024, 5, 29), LAG, depth=3)
    before = C.grid(held(2022), [], date(2024, 5, 28), LAG, depth=3)
    assert {c["year"]: c["state"] for c in on["cells"]}[2023] == "due" and {c["year"]: c["state"] for c in before["cells"]}[2023] == "notyet"


def test_a_company_with_nothing_held_has_every_past_year_missing_and_a_year_ends_on_the_calendars_last_day_by_default():
    g = C.grid([], [], TODAY, LAG, depth=3)
    assert (g["first"], g["last"], g["earliest_held"], g["latest_held"]) == (2023, 2026, None, None)
    assert states(g["cells"], 2023, 2026) == {2023: "missing", 2024: "missing", 2025: "missing", 2026: "notyet"}      # nothing held: a past year is simply missing, not "due"
    assert {c["year"]: c["expected_filing"] for c in g["cells"]}[2026] == "2027-05-30"


def test_a_year_is_judged_against_the_year_end_of_the_latest_statement_moved_on_by_whole_years():
    january = C.grid(held(2023, end="01-31", filed="03-15"), [], date(2025, 6, 1), LAG, depth=2)
    cell = {c["year"]: c for c in january["cells"]}
    assert cell[2024]["state"] == "due" and cell[2024]["expected_filing"] == "2024-06-29"               # a year ending 2023-01-31 moved on one year ends 2024-01-31; 150 days on is 2024-06-29
    assert cell[2025]["state"] == "notyet" and cell[2025]["expected_filing"] == "2025-06-30"
    leap = C.grid(held(2024, end="02-29", filed="04-15"), [], date(2026, 1, 1), LAG, depth=2)
    assert {c["year"]: c for c in leap["cells"]}[2025]["expected_filing"] == "2025-07-28"                # 29 February 2024 moved on a year is 28 February 2025; 150 days on is 28 July


def test_the_first_statement_of_a_year_stands_for_it_and_a_restatement_does_not_move_it():
    rows = held(2022) + [{"fiscal_year": 2022, "period_end": "2022-12-31", "filing_date": "2023-09-01"}, {"fiscal_year": 2021, "period_end": "2021-12-31", "filing_date": None}]
    cell = {c["year"]: c for c in C.grid(rows, [], TODAY, LAG, depth=6)["cells"]}
    assert cell[2022]["state"] == "held" and cell[2022]["filed"] == "2023-02-15" and cell[2021]["filed"] is None


def test_dates_that_carry_a_time_are_read_as_dates():
    rows = [{"fiscal_year": 2022, "period_end": "2022-12-31 00:00:00", "filing_date": "2023-02-15T09:30:00"}, {"fiscal_year": 2023, "period_end": "2023-12-31T00:00:00", "filing_date": None}]
    g = C.grid(rows, [{"at": "2026-10-20T09:30:00+00:00", "asked_from": 2010, "asked_to": 2012}], TODAY, LAG, depth=16)
    cell = {c["year"]: c for c in g["cells"]}
    assert (cell[2022]["period_end"], cell[2022]["filed"], cell[2023]["period_end"], cell[2023]["filed"]) == ("2022-12-31", "2023-02-15", "2023-12-31", None)
    assert cell[2011]["state"] == "empty" and cell[2011]["asked_on"] == "2026-10-20" and cell[2024]["expected_filing"] == "2025-05-30"      # the year end of the latest statement is read as a date, and so is the day asked
    assert "On 2026-10-20 the request" in C.reading({"at": "2026-10-20T09:30:00+00:00", "asked_from": None, "asked_to": None, "entries": 1, "kept": 1, "first_year": 2020, "last_year": 2020, "cost": 10}, "OpenFilings", "credits")


def test_the_grid_reaches_back_to_the_earliest_year_held_or_asked_for_and_never_before_1990():
    assert C.grid(held(1995, 2020), [], TODAY, LAG)["first"] == 1995
    assert C.grid(held(2020), [{"at": "2026-01-02", "asked_from": 1998, "asked_to": None}], TODAY, LAG)["first"] == 1998
    assert C.grid(held(1985), [], TODAY, LAG)["first"] == 1990
    assert C.grid(held(2020), [{"at": "2026-01-02", "asked_from": None, "asked_to": None}], TODAY, LAG)["first"] == 2001
    assert C.grid(held(2040), [], TODAY, LAG, depth=1)["last"] == 2040                                    # a year later than today is shown when it is held


# ------------------------------------------------------------------------------------------------------------------------ what was asked
def asked(at, first, last=None):
    return {"at": at, "asked_from": first, "asked_to": last}


def test_a_year_a_request_covered_when_it_was_due_and_found_nothing_is_empty_and_the_request_decides_nothing_it_did_not_cover():
    rows = held(*range(2019, 2025))
    g = C.grid(rows, [asked("2026-10-20", 2005, 2012)], TODAY, LAG)
    s = states(g["cells"], 2003, 2026)
    assert [y for y, v in s.items() if v == "empty"] == list(range(2005, 2013)) and s[2004] == "missing" and s[2013] == "missing" and s[2025] == "due"
    assert {c["year"]: c["asked_on"] for c in g["cells"]}[2007] == "2026-10-20" and g["counts"]["empty"] == 8
    both = C.grid(rows, [asked("2026-01-05", 2010, 2011), asked("2026-10-20", 2011, 2012)], TODAY, LAG)
    assert {c["year"]: c["asked_on"] for c in both["cells"]}[2011] == "2026-10-20" and {c["year"]: c["asked_on"] for c in both["cells"]}[2010] == "2026-01-05"      # the latest request is the one named
    assert states(C.grid(rows, [asked("2026-10-20", 2012, None)], TODAY, LAG)["cells"], 2012, 2018)[2018] == "empty"                              # an open end covers everything after


def test_a_year_newer_than_anything_held_is_empty_only_if_it_was_already_due_when_asked_and_for_a_month():
    rows = held(*range(2019, 2025))
    early = C.grid(rows, [asked("2026-03-01", 2019, None)], TODAY, LAG)                                  # FY2025 could not be filed before 2026-05-30
    assert states(early["cells"], 2025, 2026) == {2025: "due", 2026: "notyet"}
    late = C.grid(rows, [asked("2026-09-30", 2019, None)], TODAY, LAG)
    assert states(late["cells"], 2025, 2026) == {2025: "empty", 2026: "notyet"}
    for day, state in (("2026-09-21", "due"), ("2026-09-22", "empty")):                                   # 30 days ago the filing may have appeared since; 29 days ago it is too soon to ask again
        assert states(C.grid(rows, [asked(day, 2019, None)], TODAY, LAG)["cells"], 2025, 2025)[2025] == state
    old = C.grid(rows, [asked("2026-09-01", 2005, 2012)], TODAY, LAG)
    assert states(old["cells"], 2005, 2012)[2008] == "empty"                                              # a year before the latest held one stays empty however long ago


def test_a_request_made_before_a_years_filing_was_due_says_nothing_about_it_even_when_later_years_are_held():
    rows = held(2017, *range(2019, 2025))
    early = C.grid(rows, [asked("2019-03-01", 2000, None)], TODAY, LAG)                                  # FY2018 could not be filed before 2019-05-30: a request on 2019-03-01 found nothing because there was nothing yet
    assert states(early["cells"], 2016, 2018) == {2016: "empty", 2017: "held", 2018: "missing"}
    later = C.grid(rows, [asked("2019-05-30", 2000, None)], TODAY, LAG)
    assert states(later["cells"], 2018, 2018) == {2018: "empty"}                                         # from the day its filing was expected, a request that found nothing counts
    assert states(C.grid(rows, [asked("2019-05-29", 2000, None)], TODAY, LAG)["cells"], 2018, 2018) == {2018: "missing"}


def test_a_year_that_is_now_held_is_held_whatever_was_asked_before():
    g = C.grid(held(2015, *range(2019, 2025)), [asked("2026-10-20", 2005, 2018)], TODAY, LAG)
    assert states(g["cells"], 2014, 2016) == {2014: "empty", 2015: "held", 2016: "empty"}


# ------------------------------------------------------------------------------------------------------------------------ the plan
def plan_of(picked, rows=None, log=None, ask_again=False, today=TODAY):
    rows = held(*range(2019, 2025)) if rows is None else rows
    g = C.grid(rows, log or [], today, LAG)
    return C.plan(g["cells"], picked, ask_again, g["latest_held"], 10)


def test_the_picked_years_that_are_not_held_become_one_request_from_the_earliest_to_the_latest():
    p = plan_of([2012, 2005, 2007])
    assert p["to_fetch"] == [2005, 2007, 2012] and p["window"] == {"from": 2005, "to": 2012} and p["requests"] == 1 and p["cost"] == 10 and p["picked"] == [2005, 2007, 2012]
    assert p["already_held"] == [] and p["left_out_empty"] == [] and p["left_out_notyet"] == [] and p["outside"] == [] and p["asking_again"] == []
    assert plan_of([2010])["window"] == {"from": 2010, "to": 2010}
    held_too = plan_of([2005, 2020, 2022])                                                                  # held years inside the window come back in the answer and are ignored
    assert held_too["to_fetch"] == [2005] and held_too["already_held"] == [2020, 2022] and held_too["window"] == {"from": 2005, "to": 2005}


def test_a_window_that_reaches_a_year_newer_than_anything_held_is_left_open_so_that_a_newer_filing_comes_along():
    assert plan_of([2025])["window"] == {"from": 2025, "to": None}
    assert plan_of([2010, 2025])["window"] == {"from": 2010, "to": None}
    assert plan_of([2010, 2018])["window"] == {"from": 2010, "to": 2018}                                  # the latest held year is 2024: 2018 is below it
    nothing = plan_of([2001, 2003], rows=[])
    assert nothing["window"] == {"from": 2001, "to": None} and nothing["cost"] == 10                      # a company with nothing held: ask from the first year on


def test_held_years_are_never_asked_for_and_a_plan_of_nothing_costs_nothing():
    p = plan_of([2020, 2021])
    assert (p["to_fetch"], p["window"], p["requests"], p["cost"], p["already_held"]) == ([], None, 0, 0, [2020, 2021])
    assert C.nothing_to_fetch(p) == "nothing to ask for: you already hold FY2020 to FY2021"
    assert C.nothing_to_fetch(plan_of([])) == "pick at least one year" and plan_of([])["cost"] == 0


def test_a_year_found_empty_and_a_year_that_cannot_be_filed_yet_are_left_out_unless_asked_for_again():
    log = [asked("2026-10-20", 2005, 2012)]
    p = plan_of([2006, 2007, 2013, 2026], log=log)
    assert p["to_fetch"] == [2013] and p["left_out_empty"] == [2006, 2007] and p["left_out_notyet"] == [2026] and p["asking_again"] == []
    again = plan_of([2006, 2007, 2013, 2026], log=log, ask_again=True)
    assert again["to_fetch"] == [2006, 2007, 2013, 2026] and again["asking_again"] == [2006, 2007] and again["left_out_empty"] == [] and again["left_out_notyet"] == []
    only_empty = plan_of([2006], log=log)
    assert only_empty["to_fetch"] == [] and only_empty["cost"] == 0
    assert C.nothing_to_fetch(only_empty) == "nothing to ask for: FY2006 was asked for before and the provider sent nothing (ask again only if you want to pay for another try)"
    assert C.nothing_to_fetch(plan_of([2026])) == "nothing to ask for: FY2026 cannot have been filed yet"
    assert C.nothing_to_fetch(plan_of([2000])) == "nothing to ask for: FY2000 is outside the years shown"
    assert C.nothing_to_fetch(plan_of([2000, 2020, 2026, 2006], log=log)) == ("nothing to ask for: you already hold FY2020; FY2006 was asked for before and the provider sent nothing "
                                                                              "(ask again only if you want to pay for another try); FY2026 cannot have been filed yet; FY2000 is outside the years shown")


def test_the_plan_is_told_in_one_sentence_with_what_is_left_alone_and_what_is_left_out():
    p = plan_of([2005, 2006, 2007, 2020])
    assert C.plan_text(p, "credits", 89) == "One request for FY2005 to FY2007: 10 credits (89 left today). Any year in that window not held yet is stored; years you hold are left alone. FY2020 is held and not asked for."
    assert C.plan_text(plan_of([2025]), "credits", None) == "One request for FY2025 onwards: 10 credits. Any year in that window not held yet is stored; years you hold are left alone."
    assert C.plan_text(plan_of([2008]), "credits", 5) == "One request for FY2008: 10 credits (5 left today). Any year in that window not held yet is stored; years you hold are left alone."
    log = [asked("2026-10-20", 2005, 2012)]
    mixed = plan_of([2006, 2013, 2026], log=log)
    assert C.plan_text(mixed, "credits", 100) == ("One request for FY2013: 10 credits (100 left today). Any year in that window not held yet is stored; years you hold are left alone. "
                                                  "Left out because the provider sent nothing for them last time: FY2006. Left out because they cannot have been filed yet: FY2026.")
    again = plan_of([2006, 2013], log=log, ask_again=True)
    assert C.plan_text(again, "credits", 100).endswith("It includes FY2006, which the provider sent nothing for last time: the same price may buy nothing again.")
    assert C.plan_text(plan_of([2020]), "credits", 100) == C.nothing_to_fetch(plan_of([2020]))
    assert C.plan_text(plan_of([2020, 2021, 2005]), "credits", 100).endswith("FY2020 to FY2021 are held and not asked for.")


# ------------------------------------------------------------------------------------------------------------------------ words
def test_an_amount_is_written_in_the_singular_for_one():
    assert [C.amount(n, u) for n, u in ((10, "credits"), (1, "credits"), (0, "credits"), (1, "requests"), (2, "requests"), (1, "credit"))] == ["10 credits", "1 credit", "0 credits", "1 request", "2 requests", "1 credit"]
    assert C.plan_text({"to_fetch": [2014], "window": {"from": 2014, "to": 2014}, "cost": 1, "already_held": [], "asking_again": [], "left_out_empty": [], "left_out_notyet": []}, "requests", 5).startswith("One request for FY2014: 1 request (5 left today).")


def test_years_are_written_as_runs_and_read_the_way_a_person_types_them():
    assert [C.years_text(x) for x in ([2019], [2019, 2021], [2018, 2017, 2016, 2020, 2022, 2015], [2005, 2006], [2005, 2005])] == [
        "FY2019", "FY2019 and FY2021", "FY2015 to FY2018, FY2020 and FY2022", "FY2005 to FY2006", "FY2005"]
    assert C.parse_years("2005-2007,2010") == [2005, 2006, 2007, 2010] and C.parse_years(" 2012 ") == [2012] and C.parse_years("2012, 2010-2011 2012") == [2010, 2011, 2012]
    assert len(C.parse_years("2000-2059")) == 60                                                         # sixty is the most
    for bad, why in (("", "pick at least one year"), ("abc", "'abc' is not a year"), ("2012-2010", "not a range of years"), ("1980", "not a range of years"), ("2101", "not a range of years"),
                     ("1990-2100", "pick at most 60"), ("2000-2060", "pick at most 60")):
        with pytest.raises(ValueError, match=why):
            C.parse_years(bad)
    assert [C.describe_window(w) for w in ({"from": 2005, "to": 2012}, {"from": 2020, "to": None}, {"from": 2020, "to": 2020}, None, {"from": None, "to": None})] == [
        "FY2005 to FY2012", "FY2020 onwards", "FY2020", "every year", "every year"]


def test_the_labels_of_an_answer_are_counted_and_only_full_years_give_its_first_and_last_year():
    answer = {"periods": [{"fiscal_year": 2019, "fiscal_period": "FY", "period_end": "2019-12-31"}, {"fiscal_year": 2020, "fiscal_period": "annual", "period_end": "2020-12-31"},
                          {"fiscal_year": 2018, "fiscal_period": "Q4", "period_end": "2018-12-31"}, {"period_end": "2021-12-31"}, {"fiscal_period": "FY", "period_end": "2017-12-31"}, "junk"]}
    s = C.summarise_answer(answer)
    assert s == {"entries": 5, "labels": {"FY": 4, "Q4": 1}, "annual": 4, "first_year": 2017, "last_year": 2021, "echoed": {}}
    assert C.summarise_answer([{"fiscal_period": "FY", "period_end": "2012-12-31"}]) == {"entries": 1, "labels": {"FY": 1}, "annual": 1, "first_year": 2012, "last_year": 2012, "echoed": {}}
    assert C.summarise_answer({"periods": []}) == {"entries": 0, "labels": {}, "annual": 0, "first_year": None, "last_year": None, "echoed": {}} and C.summarise_answer(None)["entries"] == 0
    assert C.summarise_answer({"periods": [], "query": {"fiscal_year_from": 2019, "fiscal_year_to": None}})["echoed"] == {"from": 2019}                # the window the provider says it ran
    assert C.summarise_answer({"periods": [], "query": {"fiscal_year_from": "2019", "fiscal_year_to": True}})["echoed"] == {} and C.summarise_answer({"query": "x"})["echoed"] == {}
    odd = C.summarise_answer([{"fiscal_year": 2020, "fiscal_period": "FY", "period_end": "2021-01-31"}, {"fiscal_year": True, "fiscal_period": "FY", "period_end": "2018-12-31"}])
    assert (odd["first_year"], odd["last_year"]) == (2018, 2020)                                         # the provider's own year, and the year of the period end when it gives none (a bool is not a year)
    assert [C.label_of(x) for x in ("fy", " Full Year ", "h-1", "ttm", None, "", "a" * 30)] == ["FY", "FY", "H1", "TTM", "FY", "FY", "A" * 12]
    assert C.asked_years({"fiscal_year_from": 2000, "fiscal_year_to": 2010}) == (2000, 2010) and C.asked_years({"start_year": 2009, "end_year": None}) == (2009, None)
    assert C.asked_years({"fiscal_year_from": True}) == (None, None) and C.asked_years(None) == (None, None) and C.asked_years({"fiscal_year_from": "2000"}) == (None, None)


def test_the_last_request_is_told_in_words_and_says_what_the_missing_years_could_mean():
    row = {"at": "2026-10-20", "asked_from": 2000, "asked_to": None, "entries": 6, "kept": 6, "first_year": 2019, "last_year": 2024, "cost": 10}
    text = C.reading(row, "OpenFilings", "credits")
    assert text.startswith("On 2026-10-20 the request to OpenFilings for FY2000 onwards brought back 6 annual statements (FY2019 to FY2024) and cost 10 credits. Credits pay for the request, not for the years")
    assert text.endswith("It sent nothing for FY2000 to FY2018: either it holds no statements that early for this company or it limits how long a series it returns (its documentation says neither). "
                         "Asking for just those years tells which, for the same price.")
    complete = C.reading({**row, "asked_from": 2019}, "OpenFilings", "credits")
    assert "It sent nothing" not in complete and "entries in all" not in complete and "entries in all" not in text
    one_year_short = C.reading({**row, "asked_from": 2018}, "OpenFilings", "credits")
    assert "It sent nothing for FY2018:" in one_year_short and "FY2018 to" not in one_year_short.split("It sent nothing")[1]
    assert "brought back 1 annual statement (FY2020)" in C.reading({**row, "kept": 1, "first_year": 2020, "last_year": 2020, "asked_from": 2020}, "OpenFilings", "credits")
    assert C.reading({**row, "cost": None}, "OpenFilings", "credits").split(" Credits pay")[0].endswith("(FY2019 to FY2024).")                  # a cost that was not recorded is not guessed
    assert C.reading({**row, "entries": 0, "kept": 0, "first_year": None, "last_year": None}, "OpenFilings", "credits") == "On 2026-10-20 the request to OpenFilings for FY2000 onwards brought back no periods at all and cost 10 credits."
    odd = C.reading({**row, "entries": 3, "kept": 0, "first_year": None, "last_year": None, "detail": {"labels": {"Q1": 2, "TTM": 1}}}, "OpenFilings", "credits")
    assert odd == "On 2026-10-20 the request to OpenFilings for FY2000 onwards brought back 3 entries (2 Q1, 1 TTM), none of them an annual statement this application reads and cost 10 credits."
    mixed = C.reading({**row, "entries": 9, "kept": 6, "detail": {"labels": {"FY": 7, "Q4": 2}}}, "OpenFilings", "credits")
    assert " It sent 9 entries in all; 3 were not used because they were not a full year with figures (2 Q4)." in mixed and mixed.index("entries in all") < mixed.index("It sent nothing for FY2000")
    bare = C.reading({**row, "entries": 7, "kept": 6, "detail": {"labels": {"FY": 7}}}, "OpenFilings", "credits")
    assert " It sent 7 entries in all; 1 was not used because it was not a full year with figures." in bare
    assert "entries in all" not in C.reading({**row, "entries": 9, "derived": True}, "OpenFilings", "credits")                      # an old answer carries no count of what was readable
    changed = C.reading({**row, "detail": {"echoed": {"from": 2019, "to": 2024}}}, "OpenFilings", "credits")
    assert " OpenFilings recorded the request it ran as FY2019 to FY2024, not FY2000 onwards: it changed the window before answering. " in changed
    assert changed.index("recorded the request") < changed.index("It sent nothing for FY2000")
    for echo in ({"from": 2000, "to": 2024}, {"from": 2000}, {"to": 2024}, {}, None):                                                         # the same start, an end it was not asked for, or nothing said: no change to report
        assert "recorded the request" not in C.reading({**row, "detail": {"echoed": echo}}, "OpenFilings", "credits"), echo
    bounded = C.reading({**row, "asked_from": 2005, "asked_to": 2012, "first_year": 2005, "last_year": 2012, "detail": {"echoed": {"from": 2005, "to": 2010}}}, "OpenFilings", "credits")
    assert "recorded the request it ran as FY2005 to FY2010, not FY2005 to FY2012" in bounded
    assert C.reading(None, "OpenFilings", "credits") == ""


def test_the_report_of_a_request_says_what_was_added_what_was_not_sent_and_what_it_cost():
    base = {"cost": 10, "unit": "credits", "window": {"from": 2005, "to": 2012}, "entries": 4, "left_alone": 0, "nothing_for": [], "added": [{"year": y} for y in (2008, 2009, 2010)]}
    assert C.report_text(base, "OpenFilings") == "Added FY2008 to FY2010 (3 annual statements). 10 credits spent."
    more = C.report_text({**base, "nothing_for": [2005, 2006, 2007], "left_alone": 2}, "OpenFilings")
    assert more == ("Added FY2008 to FY2010 (3 annual statements). 10 credits spent. OpenFilings sent nothing for FY2005 to FY2007: it may hold nothing that early for this company, so those "
                    "years will not be offered again unless you ask. 2 years in its answer were already held and left alone.")
    assert C.report_text({**base, "added": [{"year": 2009}], "left_alone": 1}, "OpenFilings") == ("Added FY2009 (1 annual statement). 10 credits spent. 1 year in its answer was already held and left alone.")
    none = C.report_text({**base, "added": [], "entries": 0}, "OpenFilings")
    assert none == "OpenFilings brought back nothing new for FY2005 to FY2012. Its answer held no periods. 10 credits spent."
    assert C.report_text({**base, "added": [], "entries": 1}, "OpenFilings").startswith("OpenFilings brought back nothing new for FY2005 to FY2012. It sent 1 entry, none of them an annual statement not already held.")
    assert C.report_text({**base, "added": [], "entries": 3}, "OpenFilings").count("3 entries") == 1


def test_answers_kept_before_the_log_existed_give_log_rows_with_the_cost_unknown():
    answers = [{"id": 2, "parameters": {"fiscal_year_from": 2000, "fiscal_year_to": None, "fiscal_period": "fy"}, "retrieved_at": "2026-10-20T09:30:00+00:00",
                "payload": {"periods": [{"fiscal_year": y, "fiscal_period": "FY", "period_end": f"{y}-12-31"} for y in (2021, 2022, 2023)]}},
               {"id": 1, "parameters": {"start_year": 2015, "end_year": 2016}, "retrieved_at": "2026-01-02 00:00:00", "payload": [{"fiscal_period": "FY", "period_end": "2015-12-31"}]}]
    rows = C.derive_log(answers)
    assert rows[0] == {"at": "2026-10-20", "asked_from": 2000, "asked_to": None, "entries": 3, "kept": 3, "new": None, "first_year": 2021, "last_year": 2023, "cost": None, "reason": "",
                       "derived": True, "detail": {"labels": {"FY": 3}, "echoed": {}}}
    assert (rows[1]["at"], rows[1]["asked_from"], rows[1]["asked_to"], rows[1]["entries"]) == ("2026-01-02", 2015, 2016, 1)
    assert C.derive_log([{"parameters": None, "payload": None, "retrieved_at": None}])[0]["at"] == ""
    grid = C.grid(held(2021, 2022, 2023), rows[:1], TODAY, LAG)
    assert states(grid["cells"], 2000, 2020)[2010] == "empty"


# ------------------------------------------------------------------------------------------------------------------------ against the stand-in for OpenFilings
def _held(svc, sid):
    return sorted(int(r["fiscal_year"]) for r in svc.store.held_periods(sid, "openfilings") if r["fiscal_period"] == "FY")


def _one_company(tmp_path, start=2018, fake=None, clock=None, **allow):
    svc, fake = _service(tmp_path, fake=fake, clock=clock or Clock(), **allow)                            # 2026-10-20 unless a clock is given
    sid = svc.add_companies("openfilings", ["AAPL"])[0]
    assert svc.sync("openfilings", security_ids=[sid], periods=("annual",), start_year=start)["fetched"] == 1
    return svc, fake, sid


def test_a_company_says_what_it_holds_and_why_the_series_stopped_where_it_did(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2000)                                                    # OpenFilings holds 2012 to 2023: the answer to a request from 2000 starts in 2012
    cov = svc.coverage(sid)
    assert (cov["provider"], cov["symbol"], cov["unit"], cov["request_cost"], cov["can_fetch"], cov["why_not"], cov["derived"], cov["today"]) == ("openfilings", "us:AAPL", "credits", 10, True, "", False, "2026-10-20")
    a = cov["annual"]
    assert (a["earliest_held"], a["latest_held"], a["counts"]["held"]) == (2012, 2023, 12) and states(a["cells"], 2000, 2012)[2011] == "empty" and states(a["cells"], 2024, 2026) == {2024: "empty", 2025: "empty", 2026: "notyet"}
    assert cov["reading"].startswith("On 2026-10-20 the request to OpenFilings for FY2000 onwards brought back 12 annual statements (FY2012 to FY2023) and cost 10 credits.")
    assert "It sent nothing for FY2000 to FY2011" in cov["reading"] and cov["remaining_today"] == 89 and cov["sync"]["history_exhausted"] == 1
    assert "OpenFilings recorded the request it ran as FY2012 to FY2023, not FY2000 onwards: it changed the window before answering." in cov["reading"]      # the stand-in answers with the window it ran
    assert cov["last_request"]["reason"] == "missing" and cov["last_request"]["kept"] == 12 and cov["last_request"]["new"] == 12 and len(cov["log"]) == 1 and cov["log"][0]["asked"] == "FY2000 onwards"
    assert cov["quarterly"] == {"supported": False, "held": 0, "rows": [], "note": "OpenFilings quarterly and half-year figures cannot be requested yet: its documentation does not say how to ask for them, and a guess would cost 10 credits a try."}
    assert svc.usage.used_today("openfilings") == 11 and len(fake.requests) == 2                         # a lookup and a series; looking at the years spent nothing


def test_asking_for_chosen_years_costs_one_request_stores_only_what_is_new_and_leaves_what_is_held_alone(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    before = {(r["fiscal_year"], r["updated_at"]) for r in _rows(svc, sid)}
    used = svc.usage.used_today("openfilings")
    plan = svc.plan_years(sid, [2014, 2015, 2016, 2019, 2021])
    assert plan["to_fetch"] == [2014, 2015, 2016] and plan["already_held"] == [2019, 2021] and plan["cost"] == 10 and plan["fits_today"] is True and plan["can_fetch"] is True
    assert plan["text"].startswith("One request for FY2014 to FY2016: 10 credits (89 left today).") and svc.usage.used_today("openfilings") == used and len(fake.requests) == 2
    report = svc.fetch_years(sid, [2014, 2015, 2016, 2019, 2021], budget=10)
    assert report["cost"] == 10 and [a["year"] for a in report["added"]] == [2014, 2015, 2016] and report["nothing_for"] == [] and report["left_alone"] == 0 and report["window"] == {"from": 2014, "to": 2016}
    assert report["text"] == "Added FY2014 to FY2016 (3 annual statements). 10 credits spent."
    assert svc.usage.used_today("openfilings") == used + 10 and len(fake.requests) == 3
    sent = fake.requests[-1]["query"]
    assert (sent["fiscal_year_from"], sent["fiscal_year_to"], sent["fiscal_period"]) == ("2014", "2016", "fy")      # the years asked for, no more
    assert _held(svc, sid) == [2014, 2015, 2016, 2018, 2019, 2020, 2021, 2022, 2023]
    after = {(r["fiscal_year"], r["updated_at"]) for r in _rows(svc, sid)}
    assert before <= after and len(after) == len(before) + 3                                              # the years held are the very rows they were: nothing was written to them
    log = svc.store.fetch_log(sid, "openfilings")[0]
    assert (log["asked_from"], log["asked_to"], log["entries"], log["kept"], log["new"], log["cost"], log["reason"], log["at"]) == (2014, 2016, 3, 3, 3, 10, "chosen", "2026-10-20")
    assert svc.store.get_sync(sid, "openfilings", "financials:annual")["earliest_period_end"] == "2014-12-31"
    again = svc.coverage(sid)["annual"]
    assert states(again["cells"], 2013, 2017) == {2013: "missing", 2014: "held", 2015: "held", 2016: "held", 2017: "missing"}


def _rows(svc, sid):
    with svc.store.db.transaction() as conn:
        return [dict(r) for r in conn.execute("SELECT fiscal_year, updated_at FROM financial_periods WHERE security_id = ?", (sid,)).fetchall()]


def test_a_window_that_holds_years_already_held_gets_them_back_and_ignores_them_and_stores_any_other_year_it_paid_for(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    report = svc.fetch_years(sid, [2013, 2022], budget=10)                                               # 2022 is held: not asked for; the window is the single year 2013
    assert report["window"] == {"from": 2013, "to": 2013} and [a["year"] for a in report["added"]] == [2013]
    svc.fetch_years(sid, [2016], budget=10)
    report = svc.fetch_years(sid, [2014, 2017], budget=10)                                                # the window 2014 to 2017 holds 2016, which comes back and is ignored, and 2015, which is stored
    assert report["window"] == {"from": 2014, "to": 2017} and [a["year"] for a in report["added"]] == [2014, 2015, 2017] and report["left_alone"] == 1 and report["outside"] == 0
    assert report["text"] == "Added FY2014 to FY2015 and FY2017 (3 annual statements). 10 credits spent. 1 year in its answer was already held and left alone."
    assert _held(svc, sid) == [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]


def test_picking_the_newest_year_leaves_the_end_open_and_brings_a_newer_filing_along(tmp_path):
    clock = Clock()
    svc, fake, sid = _one_company(tmp_path, start=2018, clock=clock)
    assert states(svc.coverage(sid)["annual"]["cells"], 2024, 2026) == {2024: "empty", 2025: "empty", 2026: "notyet"}      # asked for today, the provider has nothing yet
    assert svc.plan_years(sid, [2024])["window"] is None
    clock.advance(days=40)                                                                                 # a month on, the filings may have appeared
    fake.last_year = 2025
    assert states(svc.coverage(sid)["annual"]["cells"], 2024, 2026) == {2024: "due", 2025: "due", 2026: "notyet"}
    assert svc.plan_years(sid, [2024])["window"] == {"from": 2024, "to": None}
    report = svc.fetch_years(sid, [2024], budget=10)
    assert [a["year"] for a in report["added"]] == [2024, 2025] and _held(svc, sid)[-3:] == [2023, 2024, 2025] and "fiscal_year_to" not in fake.requests[-1]["query"]


def test_years_that_are_all_held_ask_nothing_and_spend_nothing(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    used, sent = svc.usage.used_today("openfilings"), len(fake.requests)
    with pytest.raises(FundamentalsError, match="nothing to ask for: you already hold FY2019 to FY2020"):
        svc.fetch_years(sid, [2019, 2020], budget=100)
    with pytest.raises(FundamentalsError, match="pick at least one year"):
        svc.fetch_years(sid, [], budget=100)
    assert svc.usage.used_today("openfilings") == used and len(fake.requests) == sent


def test_the_budget_must_cover_the_request_and_the_days_allowance_must_have_it_and_the_company_must_be_known(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    sent = len(fake.requests)
    with pytest.raises(FundamentalsError, match="this asks for 10 credits but the budget is 9"):
        svc.fetch_years(sid, [2014], budget=9)
    assert len(fake.requests) == sent
    poor, poor_fake, poor_sid = _one_company(tmp_path / "poor", start=2018, daily=15)                    # 11 of 15 credits a day spent: 4 left
    assert poor.coverage(poor_sid)["remaining_today"] == 4 and poor.plan_years(poor_sid, [2014])["fits_today"] is False
    with pytest.raises(QuotaExhausted):
        poor.fetch_years(poor_sid, [2014], budget=10)
    assert len(poor_fake.requests) == 2
    with pytest.raises(FundamentalsError, match="no company with id 999"):
        svc.coverage(999)
    with pytest.raises(FundamentalsError, match="years are picked for the annual statements"):
        svc.plan_years(sid, [2014], period="quarter")
    with pytest.raises(FundamentalsError, match="pick at most 60 years"):
        svc.plan_years(sid, list(range(1990, 2051)))
    with pytest.raises(FundamentalsError, match="years must be whole numbers"):
        svc.plan_years(sid, ["abc"])


def test_years_the_provider_sent_nothing_for_are_remembered_not_offered_again_and_can_be_asked_for_again(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    report = svc.fetch_years(sid, [2005, 2006], budget=10)                                                 # OpenFilings holds nothing before 2012
    assert report["added"] == [] and report["nothing_for"] == [2005, 2006] and report["cost"] == 10 and report["entries"] == 0 and report["window"] == {"from": 2005, "to": 2006}
    assert report["text"] == "OpenFilings brought back nothing new for FY2005 to FY2006. Its answer held no periods. 10 credits spent."
    assert svc.usage.used_today("openfilings") == 21
    cov = svc.coverage(sid)
    assert states(cov["annual"]["cells"], 2004, 2007) == {2004: "missing", 2005: "empty", 2006: "empty", 2007: "missing"} and cov["sync"]["history_exhausted"] == 1
    assert cov["last_request"]["entries"] == 0 and cov["last_request"]["kept"] == 0 and cov["reading"] == "On 2026-10-20 the request to OpenFilings for FY2005 to FY2006 brought back no periods at all and cost 10 credits."
    plan = svc.plan_years(sid, [2005, 2006, 2007])
    assert plan["to_fetch"] == [2007] and plan["left_out_empty"] == [2005, 2006]
    with pytest.raises(FundamentalsError, match="FY2005 was asked for before and the provider sent nothing"):
        svc.fetch_years(sid, [2005], budget=10)
    assert svc.usage.used_today("openfilings") == 21
    again = svc.fetch_years(sid, [2005], budget=10, ask_again=True)
    assert again["nothing_for"] == [2005] and svc.usage.used_today("openfilings") == 31


def test_the_history_counts_as_exhausted_only_when_a_request_was_made_from_before_anything_held_and_found_nothing(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    state = lambda: svc.store.get_sync(sid, "openfilings", "financials:annual")["history_exhausted"]  # noqa: E731
    assert state() == 0                                                                                    # held 2018 to 2023, asked from 2018: nothing older was asked for
    svc.fetch_years(sid, [2012, 2013], budget=10)
    assert state() == 0
    fake.first_year = 2015                                                                                 # the provider starts later than the window: 2015 and 2016 come back for 2014 to 2016
    report = svc.fetch_years(sid, [2014, 2015, 2016], budget=10)
    assert [a["year"] for a in report["added"]] == [2015, 2016] and report["nothing_for"] == [2014]
    assert state() == 0                                                                                    # 2012 is held, so the provider has older years than the one found missing
    fake.first_year = 2012
    svc.fetch_years(sid, [2005], budget=10)                                                                # asked from before anything held: nothing older exists
    assert state() == 1


def test_an_answer_whose_periods_are_not_annual_is_kept_and_explained_not_taken_for_an_empty_company(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    fake.period_label = "Q4"
    report = svc.fetch_years(sid, [2014, 2015], budget=10)
    assert report["added"] == [] and report["entries"] == 2 and report["cost"] == 10
    assert report["text"] == "OpenFilings brought back nothing new for FY2014 to FY2015. It sent 2 entries, none of them an annual statement not already held. 10 credits spent."
    log = svc.store.fetch_log(sid, "openfilings")[0]
    assert (log["entries"], log["kept"], log["detail"]) == (2, 0, {"labels": {"Q4": 2}, "echoed": {"from": 2014, "to": 2015}})
    assert svc.coverage(sid)["reading"] == ("On 2026-10-20 the request to OpenFilings for FY2014 to FY2015 brought back 2 entries (2 Q4), none of them an annual statement this application reads and cost 10 credits.")
    stored = svc.store.raw_answers(sid, "openfilings", "financials:annual", 1)[0]
    assert [e["fiscal_period"] for e in stored["payload"]["periods"]] == ["Q4", "Q4"]                       # what the provider really sent is kept, so the cause can be read


def test_a_company_the_provider_does_not_know_is_not_marked_empty_and_costs_nothing(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    fake.unknown.add("AAPL")
    used = svc.usage.used_today("openfilings")
    with pytest.raises(NotFound):
        svc.fetch_years(sid, [2014], budget=10)
    assert svc.usage.used_today("openfilings") == used and len(svc.store.fetch_log(sid, "openfilings")) == 1
    assert states(svc.coverage(sid)["annual"]["cells"], 2014, 2014)[2014] == "missing"


def test_a_database_from_before_the_log_explains_itself_from_the_answers_it_kept(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2000)
    with svc.store.db.transaction() as conn:
        conn.execute("DELETE FROM fetch_log")
    cov = svc.coverage(sid)
    assert cov["derived"] is True and cov["log"][0]["derived"] is True and cov["log"][0]["cost"] is None
    assert "the request to OpenFilings for FY2000 onwards brought back 12 annual statements (FY2012 to FY2023). Credits pay for the request" in cov["reading"]      # the day is the real one: answers are dated by the real clock
    assert states(cov["annual"]["cells"], 2000, 2012)[2005] == "empty"                                      # the request that was made then still counts


def test_a_company_with_no_statements_yet_lists_every_year_as_missing_and_one_request_covers_the_years_picked(tmp_path):
    svc, fake = _service(tmp_path, clock=Clock())
    sid = svc.add_companies("openfilings", ["AAPL"])[0]
    cov = svc.coverage(sid)
    assert cov["annual"]["counts"]["held"] == 0 and cov["annual"]["counts"]["missing"] == 25 and cov["reading"] == "" and cov["log"] == [] and cov["last_request"] is None
    report = svc.fetch_years(sid, [2020, 2021, 2022], budget=10)
    assert [a["year"] for a in report["added"]] == [2020, 2021, 2022, 2023] and report["window"] == {"from": 2020, "to": None}      # the open end brought the newer year along
    assert _held(svc, sid) == [2020, 2021, 2022, 2023]


def test_a_provider_that_is_not_set_up_cannot_be_asked_and_says_why(tmp_path, monkeypatch):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    svc.provider("openfilings").pool._keys.clear()
    plan = svc.plan_years(sid, [2014])
    assert plan["can_fetch"] is False and "OPENFILINGS_API_KEY" in plan["why_not"]
    with pytest.raises(FundamentalsError, match="OPENFILINGS_API_KEY"):
        svc.fetch_years(sid, [2014], budget=10)
    assert svc.coverage(sid)["remaining_today"] is None


def test_the_log_of_requests_is_returned_newest_first_and_a_request_is_logged_by_the_ordinary_sync_too(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    svc.fetch_years(sid, [2014], budget=10)
    rows = svc.store.fetch_log(sid, "openfilings")
    assert [(r["reason"], r["asked_from"], r["asked_to"]) for r in rows] == [("chosen", 2014, 2014), ("missing", 2018, None)]
    assert json.loads(json.dumps(rows[0]["detail"])) == {"labels": {"FY": 1}, "echoed": {"from": 2014, "to": 2014}} and svc.store.fetch_log(sid, "openfilings", limit=1)[0]["reason"] == "chosen"
    assert svc.store.fetch_log(sid, "simulated") == [] and svc.store.fetch_log(sid, "openfilings", "financials:quarter") == []


def test_the_simulated_world_shows_its_quarterly_statements_read_only(tmp_path):
    svc = _simulated(tmp_path)
    sid = int(svc.store.securities().iloc[0]["security_id"])
    cov = svc.coverage(sid)
    assert cov["provider"] == "simulated" and cov["quarterly"]["supported"] is True and cov["quarterly"]["note"] == "" and cov["quarterly"]["held"] == 4 * cov["annual"]["counts"]["held"]
    first = cov["quarterly"]["rows"][0]
    assert sorted(first["periods"]) == ["Q1", "Q2", "Q3", "Q4"] and all(len(v) == 10 for v in first["periods"].values())
    assert cov["annual"]["counts"]["held"] >= 5 and cov["unit"] == "requests"


def _simulated(tmp_path):
    from src.fundamentals.service import FundamentalsService
    svc = FundamentalsService(tmp_path / "sim")
    svc.load_simulated(n_companies=30, start_year=2012, end_year=2018, seed=3)
    return svc


def test_the_answer_of_a_real_provider_that_is_not_in_the_documented_shape_cannot_break_the_view(tmp_path):
    svc, fake, sid = _one_company(tmp_path, start=2018)
    with svc.store.db.transaction() as conn:
        conn.execute("UPDATE raw_fundamental_payloads SET response_payload = ?", (b"not compressed",))
        conn.execute("DELETE FROM fetch_log")
    cov = svc.coverage(sid)
    assert cov["derived"] is False and cov["reading"] == "" and cov["annual"]["counts"]["held"] == 6
