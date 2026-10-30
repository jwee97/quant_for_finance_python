"""The headless Finzdata updater: what it updates, what it reports, and when it fails a scheduled run. Driven through the fake service; no call leaves the machine."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments import finzdata_update as upd
from src.webapp.finzdata import BUDGET_VARIABLE
from tests.fake_finzdata import KEY, FakeFinzdata

NOW = pd.Timestamp("2026-10-29")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("FINZDATA_API_KEY", KEY)
    monkeypatch.delenv(BUDGET_VARIABLE, raising=False)


def go(tmp_path, *args, fake=None, today=NOW):
    lines: list[str] = []
    fake = fake or FakeFinzdata(today=str(today.date()))
    code = upd.main(list(args), root=tmp_path, transport=fake, today=lambda: today, out=lines.append)
    return code, lines, fake


def test_named_markets_are_downloaded_stored_and_reported_one_line_each(tmp_path):
    code, lines, fake = go(tmp_path, "ES", "cl")
    assert code == 0 and lines[0].startswith("ES     ok      ") and lines[1].startswith("CL     ok      ")
    assert "(Finzdata, 2 calls)" in lines[0] and lines[-1] == "2 of 2 updated; 4 Finzdata calls made today, 896 left of 900"
    assert (tmp_path / "data" / "user" / "downloads" / "futures" / "ES.csv").exists() and (tmp_path / "data" / "user" / "finzdata_stored.json").exists()
    assert KEY not in "\n".join(lines) and set(fake.keys) == {KEY}


def test_with_no_names_it_tops_up_every_market_already_stored_and_the_second_download_starts_a_week_before_the_newest_bar(tmp_path):
    go(tmp_path, "ES")
    later = pd.Timestamp("2026-11-12")
    code, lines, fake = go(tmp_path, today=later, fake=FakeFinzdata(today=str(later.date())))
    assert code == 0 and lines[0].startswith("ES     ok") and "2 of 2" not in lines[-1] and lines[-1].startswith("1 of 1 updated")
    assert [q["start"] for p, q in fake.requests if p == "/v1/futures/daily"] == ["2026-10-22"]


def test_the_status_makes_no_call_and_says_which_markets_need_an_update_before_the_window_passes(tmp_path):
    code, lines, _ = go(tmp_path, "--status")
    assert code == 0 and lines == ["nothing is stored from Finzdata yet"]
    go(tmp_path, "ES")
    code, lines, fake = go(tmp_path, "--status", today=pd.Timestamp("2026-11-24"), fake=FakeFinzdata())
    assert code == 0 and fake.urls == [] and lines[0].split() == ["root", "stored", "days", "contracts", "age", "left", "gaps"]
    assert "ES" in lines[1] and "2026-09-29 to 2026-10-29" in lines[1] and lines[1].endswith("<- update now") and " 26 " in lines[1] and "    4 " in lines[1]


def test_no_key_nothing_to_do_and_a_bad_name_each_end_with_a_message_and_exit_code_2(tmp_path, monkeypatch):
    monkeypatch.delenv("FINZDATA_API_KEY")
    code, lines, fake = go(tmp_path, "ES")
    assert code == 2 and "FINZDATA_API_KEY" in lines[0] and fake.urls == []
    monkeypatch.setenv("FINZDATA_API_KEY", KEY)
    code, lines, _ = go(tmp_path)
    assert code == 2 and "nothing to update" in lines[0]
    code, lines, fake = go(tmp_path, "ES;DROP")
    assert code == 2 and "not a Finzdata futures symbol" in lines[0] and fake.urls == []


def test_one_market_failing_is_reported_and_fails_the_run_but_not_the_others(tmp_path):
    fake = FakeFinzdata(forbidden=("CL",))
    code, lines, _ = go(tmp_path, "CL", "ES", fake=fake)
    assert code == 1 and lines[0].startswith("CL     FAILED  ") and "does not include this data" in lines[0] and lines[1].startswith("ES     ok")
    assert lines[-1].startswith("1 of 2 updated")


def test_a_refused_key_or_an_exhausted_day_stops_the_run_and_names_the_markets_not_tried(tmp_path):
    fake = FakeFinzdata(quota=2)
    code, lines, _ = go(tmp_path, "ES", "CL", "E6", fake=fake)
    assert code == 1 and lines[0].startswith("ES     ok") and lines[1].startswith("CL     FAILED  ") and "limit of calls for the day" in lines[1]
    assert lines[2].startswith("E6     FAILED  not tried:") and lines[-1].startswith("1 of 3 updated")
