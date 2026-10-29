"""The live-feed smoke check: what it counts as a feed that changed and what as a refusal, how it reports, and when it fails a scheduled run. The feeds themselves are not called here (nothing in the test
suite touches the network); that is what the check is for."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from experiments import live_feed_smoke as smoke
from src.assets.free_data.http import FetchError
from src.fundamentals.errors import NotFound, ProviderError, RateLimited
from src.fundamentals.providers.sec_edgar import NameList
from tests.test_fund_sec_edgar import TICKER_ROWS, frame_of


def _ok(ctx):
    return "fine"


def _refused(ctx):
    raise FetchError("host refused the request (HTTP 403)", status=403, blocked=True)


def _limited(ctx):
    raise FetchError("host is rate limiting this address (HTTP 429)", status=429)


def _gone(ctx):
    raise FetchError("host has no such item (HTTP 404)", status=404)


def _changed(ctx):
    smoke.expect(False, "0 contracts")


def _crash(ctx):
    raise KeyError("strike")


def _skip(ctx):
    raise smoke.Skipped("no contact")


def test_a_refusal_to_this_address_is_told_apart_from_a_feed_that_changed():
    cases = [(FetchError("x", status=403, blocked=True), "refused"), (FetchError("x", status=429), "refused"), (FetchError("x", status=451), "refused"), (FetchError("x", status=404), "broken"),
             (FetchError("x", status=500), "broken"), (FetchError("not json"), "broken"), (RateLimited("too many", 600.0, "burst"), "refused"), (NotFound("no facts"), "broken"),
             (ProviderError("SEC EDGAR refused the request (HTTP 403)"), "refused"), (ProviderError("SEC EDGAR answered HTTP 502"), "broken"), (smoke.Broken("0 rows"), "broken"),
             (KeyError("strike"), "broken"), (smoke.Skipped("no contact"), "skipped"), (OSError("down"), "broken")]
    for error, expected in cases:
        assert smoke.classify(error)[0] == expected, error
    assert "does not look right" in smoke.classify(smoke.Broken("0 rows"))[1] and "KeyError" in smoke.classify(KeyError("strike"))[1]


def test_a_run_reports_every_check_whatever_happens_in_it_and_fails_only_for_a_feed_that_changed():
    checks = {"good": _ok, "blocked": _refused, "limited": _limited, "gone": _gone, "changed": _changed, "crash": _crash, "needs_contact": _skip}
    ticks = iter(range(100))
    results = smoke.run_checks(checks, {}, clock=lambda: float(next(ticks)))
    assert [(r.name, r.status) for r in results] == [("good", "ok"), ("blocked", "refused"), ("limited", "refused"), ("gone", "broken"), ("changed", "broken"), ("crash", "broken"),
                                                     ("needs_contact", "skipped")]
    assert all(r.seconds == 1.0 for r in results) and results[0].detail == "fine"
    assert smoke.exit_code(results) == 1
    only_refused = [r for r in results if r.status in ("refused", "skipped", "ok")]
    assert smoke.exit_code(only_refused) == 0 and smoke.exit_code(only_refused, strict=True) == 1               # a refusal fails a run only when asked to
    assert smoke.exit_code([]) == 0 and smoke.exit_code([smoke.Result("a", "ok", "", 0.1)], strict=True) == 0


def test_checks_can_be_picked_by_name_or_by_the_group_before_the_underscore():
    checks = {"cboe": _ok, "sec_facts": _ok, "sec_tickers": _ok, "yahoo": _ok}
    assert [r.name for r in smoke.run_checks(checks, {}, ["sec"])] == ["sec_facts", "sec_tickers"]
    assert [r.name for r in smoke.run_checks(checks, {}, ["yahoo", "cboe"])] == ["cboe", "yahoo"]
    assert smoke.run_checks(checks, {}, ["nothing"]) == []


def test_the_report_has_a_line_a_table_and_the_annotations_a_workflow_shows():
    results = [smoke.Result("cboe", "ok", "SPY: 9992 contracts", 1.2), smoke.Result("yahoo", "refused", "HTTP 429 | slow", 0.3), smoke.Result("deribit", "broken", "0 contracts", 0.1)]
    text = smoke.render(results)
    assert "cboe" in text and "1 ok  1 refused  1 broken  0 skipped" in text
    table = smoke.markdown(results)
    assert table.startswith("### Live feeds") and "| yahoo | refused | 0.3 | HTTP 429 / slow |" in table                  # a bar inside a detail cannot break the table
    assert smoke.annotations(results) == ["::warning title=live feed yahoo refused::HTTP 429 | slow", "::error title=live feed deribit broken::0 contracts"]
    assert smoke.render([]) == "\n0 ok  0 refused  0 broken  0 skipped"


def test_the_main_function_writes_the_results_the_summary_and_the_exit_code(tmp_path, monkeypatch, capsys):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    saved = tmp_path / "results.json"
    seen = {}

    def spy(ctx):
        seen.update(ctx)
        return "seen"
    assert smoke.main(["--json", str(saved)], checks={"good": spy, "blocked": _refused}) == 0
    out = capsys.readouterr().out
    assert "good" in out and "::warning title=live feed blocked refused" in out
    assert [r["status"] for r in json.loads(saved.read_text())] == ["ok", "refused"] and "### Live feeds" in summary.read_text()
    assert seen["sec_contact"] == ""                                                                           # nothing is invented: no contact, the SEC checks skip
    assert smoke.main(["--strict"], checks={"blocked": _refused}) == 1 and smoke.main([], checks={"changed": _changed}) == 1


def test_the_sec_contact_comes_from_the_environment_alone_and_nothing_stands_in_for_it(monkeypatch):
    seen = {}

    def spy(ctx):
        seen["contact"] = ctx["sec_contact"]
        return "x"
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Jane Doe jane@example.com")
    smoke.main([], checks={"c": spy})
    assert seen["contact"] == "Jane Doe jane@example.com"
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/quant_repo")                                          # inside Actions: the repository is not made into a contact
    smoke.main([], checks={"c": spy})
    assert seen["contact"] == ""


def test_without_a_contact_the_sec_checks_skip_and_do_not_fail_the_run():
    results = smoke.run_checks({k: v for k, v in smoke.CHECKS.items() if k.startswith("sec")}, {"sec_contact": ""})
    assert [r.name for r in results] == ["sec_facts", "sec_record", "sec_tickers", "sec_filers"]
    assert all(r.status == "skipped" and "SEC_EDGAR_USER_AGENT" in r.detail for r in results)
    assert smoke.exit_code(results) == 0


def test_the_list_of_reporting_companies_is_checked_for_size_and_for_the_company_it_must_hold(monkeypatch):
    def check(rows, fillers):
        names = NameList.parse(frame_of(rows, fillers), "2026-10-23", "CY2026Q2")

        class Stand:
            def __init__(self, user_agent):
                self.user_agent = user_agent

            def filers(self):
                return names
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(smoke, "SecEdgarProvider", Stand)
            return smoke.run_checks({"sec_filers": smoke.sec_filers}, {"sec_contact": "Jane Doe jane@example.com"})[0]
    good = check(TICKER_ROWS, 4500)
    assert good.status == "ok" and "CY2026Q2" in good.detail and "CIK 320193" in good.detail
    small = check(TICKER_ROWS, 20)
    assert small.status == "broken" and "companies in CY2026Q2" in small.detail
    without_apple = check([r for r in TICKER_ROWS if r[0] != 320193], 4500)
    assert without_apple.status == "broken" and "Apple Inc." in without_apple.detail


def test_a_check_rejects_an_answer_that_is_too_small_or_too_old():
    class Fake:
        def __init__(self, meta, frame=None, payload=None):
            self.meta, self.frame, self.payload = meta, frame, payload

    def run(fn, download, module, name):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module, name, lambda *a, **k: download)
            return smoke.run_checks({"c": fn}, {"http": None})[0]
    old = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    good = Fake({"contracts": 9000, "spot": 700.0, "expiries": 20, "as_of": today})
    assert run(smoke.cboe, good, smoke.options, "fetch_cboe_chain").status == "ok"
    assert run(smoke.cboe, Fake({**good.meta, "contracts": 3}), smoke.options, "fetch_cboe_chain").status == "broken"
    stale = run(smoke.cboe, Fake({**good.meta, "as_of": old}), smoke.options, "fetch_cboe_chain")
    assert stale.status == "broken" and "days ago" in stale.detail
    frame = pd.DataFrame({"close": [1.0] * 260})
    assert run(smoke.yahoo, Fake({"as_of": today}, frame), smoke.options, "fetch_underlying_prices").status == "ok"
    assert run(smoke.yahoo, Fake({"as_of": today}, frame.head(20)), smoke.options, "fetch_underlying_prices").status == "broken"
    books = Fake({}, payload={"venues": [{"label": "A"}], "failed": {"kraken": "down"}})
    assert run(smoke.exchanges, books, smoke.crypto, "fetch_venues").status == "broken"                    # one venue of four is not a working feed


# ------------------------------------------------------------------------------------------------------------------ the futures curves
BASE = "data/us/futures/market_statistics/historical_data/VX/"


class StubCboe:
    """Cboe's list of VX contracts and the file of each, as the two calls of the check see them."""

    def __init__(self, months: int = 130, alive: int = 6, last_day: pd.Timestamp | None = None, open_interest: bool = True):
        today = pd.Timestamp.now().normalize()
        first = today + pd.Timedelta(days=30 * alive - 15) - pd.Timedelta(days=30 * months)
        self.expiries = [first + pd.Timedelta(days=30 * k) for k in range(months)]
        self.today, self.last_day, self.open_interest = today, last_day or today, open_interest
        rows: dict[str, list] = {}
        for day in self.expiries:
            rows.setdefault(str(day.year), []).append({"expire_date": str(day.date()), "futures_root": "VX", "duration_type": "M", "path": f"{BASE}VX_{day.date()}.csv"})
        self.listing = rows
        self.calls: list[str] = []

    def get_json(self, url, **kwargs):
        self.calls.append(url)
        return self.listing

    def get_text(self, url, **kwargs):
        self.calls.append(url)
        days = pd.bdate_range(self.last_day - pd.Timedelta(days=60), self.last_day)
        interest = lambda i: str(1000 + i) if self.open_interest else ""                      # noqa: E731
        lines = [f"{d.date()},F,1,1,1,1,{16.0 + 0.01 * i},0,{100 + i},0,{interest(i)}" for i, d in enumerate(days)]
        return "Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest\n" + "\n".join(lines) + "\n"


def run_vx(stub):
    return smoke.run_checks({"cboe_vx": smoke.cboe_vx}, {"http": stub})[0]


def test_the_vix_futures_check_wants_a_long_list_live_contracts_and_a_recent_file_with_open_interest():
    stub = StubCboe()
    good = run_vx(stub)
    assert good.status == "ok" and "130 monthly contracts" in good.detail and "days to" in good.detail
    assert len(stub.calls) == 2 and stub.calls[0].endswith("/product/list/VX/") and "cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_" in stub.calls[1], "the list and one file, nothing more"
    short = run_vx(StubCboe(months=60))
    assert short.status == "broken" and "monthly VX contracts in Cboe's list" in short.detail
    dead = run_vx(StubCboe(alive=2))
    assert dead.status == "broken" and "VX contracts that have not expired" in dead.detail
    no_interest = run_vx(StubCboe(open_interest=False))
    assert no_interest.status == "broken" and "no open interest" in no_interest.detail
    stale = run_vx(StubCboe(last_day=pd.Timestamp.now().normalize() - pd.Timedelta(days=30)))
    assert stale.status == "broken" and "days ago" in stale.detail


def test_the_yahoo_and_deribit_futures_checks_want_enough_contracts_and_a_recent_last_day(monkeypatch):
    class Fake:
        def __init__(self, contracts: int, rows: int, as_of: str, expired: int = 0):
            today = pd.Timestamp.now().normalize()
            expiry = [today - pd.Timedelta(days=30 * (k + 1)) if k < expired else today + pd.Timedelta(days=30 * (k + 1)) for k in range(contracts)]
            per = max(rows // max(contracts, 1), 1)
            self.frame = pd.DataFrame({"contract": [f"C{k}" for k in range(contracts) for _ in range(per)], "expiry": [e for e in expiry for _ in range(per)]})
            self.meta = {"as_of": as_of}

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    old = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    def run(check, module, name, download):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module, name, lambda *a, **k: download)
            return smoke.run_checks({"c": check}, {"http": None})[0]

    assert run(smoke.yahoo_futures, smoke.futures, "fetch_yahoo_curve", Fake(4, 400, today)).status == "ok"
    assert run(smoke.yahoo_futures, smoke.futures, "fetch_yahoo_curve", Fake(2, 400, today)).status == "broken"
    assert run(smoke.yahoo_futures, smoke.futures, "fetch_yahoo_curve", Fake(4, 100, today)).status == "broken"
    stale = run(smoke.yahoo_futures, smoke.futures, "fetch_yahoo_curve", Fake(4, 400, old))
    assert stale.status == "broken" and "days ago" in stale.detail
    good = run(smoke.deribit_futures, smoke.futures, "fetch_deribit_history", Fake(10, 400, today, expired=6))
    assert good.status == "ok" and "6 of them expired" in good.detail
    assert run(smoke.deribit_futures, smoke.futures, "fetch_deribit_history", Fake(10, 400, today, expired=2)).status == "broken", "a history with almost no expired contract is not the history of expired contracts"
    assert run(smoke.deribit_futures, smoke.futures, "fetch_deribit_history", Fake(4, 400, today, expired=4)).status == "broken"
    assert run(smoke.deribit_futures, smoke.futures, "fetch_deribit_history", Fake(10, 400, old, expired=6)).status == "broken"
    assert {"cboe_vx", "yahoo_futures", "deribit_futures"} <= set(smoke.CHECKS) and {r.name for r in smoke.run_checks({k: v for k, v in smoke.CHECKS.items() if k.startswith("cboe")}, {"http": StubCboe()}, only=["cboe_vx"])} == {"cboe_vx"}
