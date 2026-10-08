"""The iTick data source: where the key lives and that it never leaks, the 5-calls-a-minute limiter, symbol mapping, the checks on what comes back, paging, top-ups and failure handling.

Nothing here touches the network: the provider is driven through a fake transport (``tests/fake_itick.py``) and a fake clock, so a wait of a minute costs no time. What cannot be tested
offline is that the real service answers the way its documentation says; ``ITickProvider.test`` (the page's "Test the iTick connection" button) is how you check that with your own key.
"""

from __future__ import annotations

import http.server
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

from src.webapp import providers
from src.webapp.providers import (ITickError, ITickProvider, RateLimiter, RateLimitTimeout, bars_to_frame, map_symbol, read_secret, redact, session_dates, split_warnings, suspected_splits,
                                  urllib_transport)
from tests.fake_itick import KEY, FakeClock, FakeITick

START = "2018-01-01"


def make(fake=None, clock=None, **kwargs):
    fake = fake or FakeITick()
    clock = clock or FakeClock()
    limiter = RateLimiter(kwargs.pop("calls", 5), clock=clock.now, sleep=clock.sleep)
    return ITickProvider(key=KEY, base_url="https://api-free.itick.org", limiter=limiter, transport=fake, sleep=clock.sleep, **kwargs), fake, clock


# ------------------------------------------------------------------------------------------------------------------------------ the key
def test_the_key_comes_from_the_environment_then_a_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ITICK_API_KEY", raising=False)
    assert read_secret("ITICK_API_KEY", tmp_path) == ""
    (tmp_path / ".env").write_text("# a comment\nOTHER=1\nexport ITICK_API_KEY = \"abc123secretvalue\"\n")
    assert read_secret("ITICK_API_KEY", tmp_path) == "abc123secretvalue"
    monkeypatch.setenv("ITICK_API_KEY", "from-the-environment")
    assert read_secret("ITICK_API_KEY", tmp_path) == "from-the-environment"
    assert read_secret("ITICK_API_KEY") == "from-the-environment"


def test_redact_removes_the_key_and_its_url_encoded_form_and_leaves_short_strings_alone():
    key = "ab/cd+ef=ghij"
    assert key not in redact(f"failed for {key} and ab%2Fcd%2Bef%3Dghij", key)
    assert redact("nothing to hide", key) == "nothing to hide"
    assert redact("abc", "abc") == "abc"                                         # too short to be a secret worth masking; masking it would wreck ordinary words


def test_the_key_goes_only_in_the_token_header_and_never_into_the_url_or_a_message():
    p, fake, _ = make()
    frames = p(["AAPL"], START)
    assert isinstance(frames["AAPL"], pd.DataFrame)
    assert set(fake.tokens) == {KEY} and all(KEY not in url for url in fake.urls)
    assert KEY not in json.dumps(p.describe()) and KEY not in repr(p.available())


def test_an_error_page_that_echoes_the_key_is_redacted_before_it_reaches_the_user():
    p, fake, _ = make()
    fake.script = [(418, {}, f"teapot, you sent {KEY} and token={KEY}".encode())]
    message = p(["AAPL"], START)["AAPL"]
    assert isinstance(message, str) and KEY not in message and "418" in message
    fake.script = [(500, {}, f"boom {KEY}".encode())] * 3
    assert KEY not in p(["MSFT"], START)["MSFT"]
    p.transport = lambda url, headers, timeout: (_ for _ in ()).throw(OSError(f"cannot reach {KEY}"))      # an exception that carries the key in its text
    out = p(["NVDA"], START)["NVDA"]
    assert isinstance(out, str) and KEY not in out


def test_an_unexpected_exception_is_reported_without_the_key():
    p, fake, _ = make()
    p._pages = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"internal {KEY}"))
    out = p(["AAPL"], START)["AAPL"]
    assert "RuntimeError" in out and KEY not in out


def test_without_a_key_nothing_is_called_and_every_ticker_gets_the_same_instruction(monkeypatch, tmp_path):
    monkeypatch.delenv("ITICK_API_KEY", raising=False)
    fake = FakeITick()
    p = ITickProvider(root=tmp_path, transport=fake)
    out = p(["AAPL", "MSFT"], START)
    assert fake.calls == [] and out["AAPL"] == out["MSFT"] and "ITICK_API_KEY" in out["AAPL"]
    assert p.available()[0] is False and p.describe()["configured"] is False and p.preflight("AAPL")


@pytest.mark.parametrize("url", ["http://api.itick.org", "ftp://x.example", "api.itick.org"])
def test_a_base_url_that_would_send_the_key_in_clear_text_is_refused(url):
    fake = FakeITick()
    p = ITickProvider(key=KEY, base_url=url, transport=fake)
    assert p.available()[0] is False and "https" in p.available()[1]
    assert "https" in p(["AAPL"], START)["AAPL"] and fake.calls == []


def test_http_is_allowed_for_a_local_test_server_only():
    assert ITickProvider(key=KEY, base_url="http://127.0.0.1:9").available()[0]
    assert ITickProvider(key=KEY, base_url="http://localhost:9").available()[0]


class _Redirecting(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_GET(self):                                                              # noqa: N802
        if self.server.server_port == self.server.first_port:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.second_port}/stock/kline")
            self.end_headers()
        else:
            _Redirecting.seen.append(self.headers.get("token"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def test_a_redirect_is_never_followed_so_the_key_cannot_be_forwarded():
    first, second = http.server.HTTPServer(("127.0.0.1", 0), _Redirecting), http.server.HTTPServer(("127.0.0.1", 0), _Redirecting)
    for srv in (first, second):
        srv.first_port, srv.second_port = first.server_port, second.server_port
    threads = [threading.Thread(target=srv.serve_forever, daemon=True) for srv in (first, second)]
    _Redirecting.seen = []
    for t in threads:
        t.start()
    try:
        status, _, _ = urllib_transport(f"http://127.0.0.1:{first.server_port}/stock/kline?code=X", {"token": KEY}, 5.0)
    finally:
        for srv in (first, second):
            srv.shutdown()
            srv.server_close()
    assert status == 302 and _Redirecting.seen == []


def test_the_real_transport_sends_the_token_header_to_a_local_server():
    got = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):                                                          # noqa: N802
            got["token"], got["accept"], got["path"] = self.headers.get("token"), self.headers.get("accept"), self.path
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"code": 0, "data": []}')

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = urllib_transport(f"http://127.0.0.1:{srv.server_port}/stock/kline?region=US&code=AAPL", {"accept": "application/json", "token": KEY}, 5.0)
    finally:
        srv.shutdown()
        srv.server_close()
    assert status == 200 and json.loads(body)["code"] == 0 and got == {"token": KEY, "accept": "application/json", "path": "/stock/kline?region=US&code=AAPL"}


# ------------------------------------------------------------------------------------------------------------------------ rate limiting
def test_five_calls_go_straight_through_and_the_sixth_waits_for_the_first_to_leave_the_window():
    clock = FakeClock()
    limiter = RateLimiter(5, clock=clock.now, sleep=clock.sleep)
    waits = [limiter.acquire() for _ in range(5)]
    assert waits == [0.0] * 5 and clock.slept == []
    heard = []
    waited = limiter.acquire(on_wait=lambda delay, why: heard.append((delay, why)))
    assert 60.0 <= waited <= 62.0 and heard and "5 calls a minute" in heard[0][1]


def test_the_limit_is_a_sliding_window_not_a_fixed_minute():
    clock = FakeClock()
    limiter = RateLimiter(5, clock=clock.now, sleep=clock.sleep)
    for _ in range(5):                                                              # one call every 10 seconds
        limiter.acquire()
        clock.sleep(10)
    clock.slept.clear()
    stamps = []
    for _ in range(5):
        limiter.acquire()
        stamps.append(clock.now())
    assert stamps[0] - 1_700_000_000.0 >= 61.0                                      # the first of the old calls was 50 s ago, and leaves the window 61 s after it was made
    allstamps = sorted(limiter._stamps)
    assert all(allstamps[i + 5] - allstamps[i] >= 61.0 - 1e-6 for i in range(len(allstamps) - 5))


def test_the_allowance_survives_a_restart(tmp_path):
    clock = FakeClock()
    path = tmp_path / "calls.json"
    first = RateLimiter(5, clock=clock.now, sleep=clock.sleep, path=path)
    for _ in range(5):
        first.acquire()
    clock.sleep(20)
    second = RateLimiter(5, clock=clock.now, sleep=clock.sleep, path=path)       # a new process reading the same file
    assert second.remaining() == 0
    waited = second.acquire()
    assert 40.0 <= waited <= 42.5
    clock.sleep(30)
    assert RateLimiter(5, clock=clock.now, sleep=clock.sleep, path=path).remaining() == 4      # the sixth call is still inside the window; the first five have left it
    clock.sleep(120)
    assert RateLimiter(5, clock=clock.now, sleep=clock.sleep, path=path).remaining() == 5


def test_a_damaged_or_future_dated_state_file_is_ignored(tmp_path):
    clock = FakeClock()
    path = tmp_path / "calls.json"
    path.write_text("not json")
    assert RateLimiter(5, clock=clock.now, sleep=clock.sleep, path=path).remaining() == 5
    path.write_text(json.dumps({"calls": [clock.now() + 86400.0] * 5, "blocked_until": clock.now() + 10 ** 7}))
    limiter = RateLimiter(5, clock=clock.now, sleep=clock.sleep, path=path)
    assert limiter.remaining() == 5 and limiter.acquire() == 0.0                    # a clock that jumped must not lock the app for a day


def test_a_pause_asked_for_by_the_service_blocks_every_call_until_it_ends():
    clock = FakeClock()
    limiter = RateLimiter(5, clock=clock.now, sleep=clock.sleep)
    limiter.hold(30)
    assert limiter.remaining() == 0
    assert 30.0 <= limiter.acquire() <= 31.5


def test_a_wait_longer_than_allowed_raises_instead_of_hanging():
    clock = FakeClock()
    limiter = RateLimiter(1, clock=clock.now, sleep=clock.sleep)
    limiter.acquire()
    with pytest.raises(RateLimitTimeout):
        limiter.acquire(max_wait=10)


def test_estimates_count_free_slots_first_then_whole_windows():
    clock = FakeClock()
    limiter = RateLimiter(5, clock=clock.now, sleep=clock.sleep)
    assert limiter.estimate_seconds(5) == 0.0 and limiter.estimate_seconds(6) == 61.0 and limiter.estimate_seconds(11) == 122.0


def test_many_threads_never_exceed_the_allowance():
    limiter = RateLimiter(3, window=0.25, margin=0.0)
    stamps, lock = [], threading.Lock()

    def worker():
        for _ in range(3):
            limiter.acquire()
            with lock:
                stamps.append(time.time())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    recorded = sorted(limiter._stamps)
    assert len(stamps) == 12
    assert all(recorded[i + 3] - recorded[i] >= 0.25 - 1e-6 for i in range(len(recorded) - 3))      # never more than three calls in any quarter second


def test_invalid_limiter_settings_are_rejected():
    for kwargs in ({"calls": 0}, {"window": 0}, {"margin": -1}):
        with pytest.raises(ValueError):
            RateLimiter(**kwargs)


# ------------------------------------------------------------------------------------------------------------------------------ symbols
@pytest.mark.parametrize("ticker,expected", [
    ("AAPL", ("stock", "US", "AAPL")), ("BRK.B", ("stock", "US", "BRK.B")), ("0700.HK", ("stock", "HK", "700")), ("600519.SS", ("stock", "SH", "600519")),
    ("000001.SZ", ("stock", "SZ", "000001")), ("7203.T", ("stock", "JP", "7203")), ("BTC-USD", ("crypto", "BA", "BTCUSDT")), ("eth-usdt", ("crypto", "BA", "ETHUSDT")),
    ("EURUSD=X", ("forex", "GB", "EURUSD")), ("^SPX", ("indices", "GB", "SPX")), ("T", ("stock", "US", "T"))])
def test_yahoo_style_symbols_map_to_itick_instruments(ticker, expected):
    inst = map_symbol(ticker)
    assert (inst.kind, inst.region, inst.code) == expected


@pytest.mark.parametrize("bad", ["", "A B", "A$B", ".HK", "-USD", "=X"])
def test_a_symbol_that_cannot_be_placed_is_explained(bad):
    with pytest.raises(ITickError, match="Write symbols|market suffix|cannot place"):
        map_symbol(bad)


# ------------------------------------------------------------------------------------------------------------------------- reading bars
def _rows(days, stamp_hours=0, close=100.0):
    return [{"t": int((d + pd.Timedelta(hours=stamp_hours)).value // 10**6), "o": close, "h": close + 1, "l": close - 1, "c": close + i * 0.1, "v": 10.0, "tu": 1.0} for i, d in enumerate(days)]


def test_bars_stamped_at_midnight_utc_get_the_session_date_not_the_previous_evening():
    days = pd.bdate_range("2024-03-04", periods=10)                                 # Monday to Friday, twice
    us = map_symbol("AAPL")
    frame, oldest = bars_to_frame(_rows(days), us)                                  # 00:00 UTC is 19:00 the day before in New York: a Monday would land on Sunday
    assert list(frame.index) == list(days) and oldest == int(days[0].value // 10**6)


def test_bars_stamped_at_local_midnight_keep_their_local_date():
    days = pd.bdate_range("2024-03-04", periods=10)
    hk = map_symbol("0700.HK")
    frame, _ = bars_to_frame(_rows(days, stamp_hours=-8), hk)                       # 00:00 in Hong Kong is 16:00 UTC the day before
    assert list(frame.index) == list(days)


def test_timestamps_in_seconds_instead_of_milliseconds_are_understood():
    days = pd.bdate_range("2024-03-04", periods=10)
    rows = _rows(days)
    for row in rows:
        row["t"] = row["t"] // 1000
    frame, oldest = bars_to_frame(rows, map_symbol("AAPL"))
    assert list(frame.index) == list(days) and oldest == int(days[0].value // 10**6)


def test_every_request_carries_an_agent_name_a_gateway_will_accept_and_the_token():
    p, fake, _ = make()
    seen = []
    original = fake.__call__

    def spying(url, headers, timeout):
        seen.append(dict(headers))
        return original(url, headers, timeout)

    p.transport = spying
    p(["AAPL"], "2024-01-01")
    assert seen and all(h["token"] == KEY and h["accept"] == "application/json" and h["User-Agent"].startswith("QuantLab/") for h in seen)


def test_bars_stamped_at_the_open_keep_their_date_either_way():
    days = pd.bdate_range("2024-03-04", periods=10)
    frame, _ = bars_to_frame(_rows(days, stamp_hours=13.5), map_symbol("AAPL"))
    assert list(frame.index) == list(days)


def test_weekend_bars_are_dropped_for_stocks_and_kept_for_crypto():
    days = pd.date_range("2024-03-04", periods=14)                                  # every calendar day
    stock, _ = bars_to_frame(_rows(days), map_symbol("AAPL"))
    coin, _ = bars_to_frame(_rows(days), map_symbol("BTC-USD"))
    assert (stock.index.dayofweek < 5).all() and len(coin) == 14


def test_hourly_bars_are_recognised_as_not_daily_and_the_message_names_the_setting():
    hours = pd.date_range("2024-03-04 09:00", periods=40, freq="h")
    with pytest.raises(ITickError, match="ITICK_DAILY_KTYPE") as caught:
        bars_to_frame(_rows(hours), map_symbol("AAPL"))
    assert caught.value.fatal
    weekly = pd.date_range("2024-01-01", periods=20, freq="7D")
    with pytest.raises(ITickError, match="not daily"):
        bars_to_frame(_rows(weekly), map_symbol("AAPL"))


def test_bad_rows_are_dropped_and_duplicates_collapse():
    days = pd.bdate_range("2024-03-04", periods=8)
    rows = _rows(days)
    rows[2]["c"] = None
    rows[3]["c"] = -5
    rows.append(dict(rows[5]))
    rows.append({"t": "garbage", "c": 1})
    frame, _ = bars_to_frame(rows, map_symbol("AAPL"))
    assert len(frame) == 6 and frame.index.is_monotonic_increasing and frame.index.is_unique and (frame["close"] > 0).all()
    assert set(frame.columns) >= {"close", "high", "low", "volume"}


def test_an_answer_without_time_and_close_fields_is_rejected_clearly():
    with pytest.raises(ITickError, match="format"):
        bars_to_frame([{"x": 1}], map_symbol("AAPL"))
    frame, oldest = bars_to_frame([], map_symbol("AAPL"))
    assert frame.empty and oldest is None


def test_session_dates_prefers_the_local_date_when_it_fits():
    stamps = np.array([int(pd.Timestamp("2024-03-05 14:30", tz="UTC").value // 10**6)])
    assert session_dates(stamps, "America/New_York", True)[0] == pd.Timestamp("2024-03-05")


# ------------------------------------------------------------------------------------------------------------------------ split warnings
def test_a_four_for_one_split_that_was_not_adjusted_is_flagged_and_ordinary_moves_are_not():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 500)), index=pd.bdate_range("2020-01-01", periods=500))
    assert suspected_splits(close) == []
    split = close.copy()
    split.iloc[300:] = split.iloc[300:] / 4.0
    found = suspected_splits(split)
    assert len(found) == 1 and found[0][0] == split.index[300] and abs(found[0][1] - 0.25) < 0.03
    note = split_warnings(split)[0]
    assert "4.0-for-1" in note and "did not adjust" in note
    reverse = close.copy()
    reverse.iloc[200:] = reverse.iloc[200:] * 10
    assert "reverse split" in split_warnings(reverse)[0]
    crash = close.copy()
    crash.iloc[100:] = crash.iloc[100:] * 0.8                                         # a 20% one-day fall is a crash, not a split
    assert suspected_splits(crash) == []


def test_only_a_few_split_warnings_are_listed():
    close = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=60))
    for i in range(5, 55, 8):
        close.iloc[i:] = close.iloc[i:] / 2.0
    notes = split_warnings(close)
    assert len(notes) == 4 and notes[-1].startswith("... and")


# --------------------------------------------------------------------------------------------------------------------------- downloading
def test_a_download_pages_back_with_et_until_it_reaches_the_start_date():
    p, fake, clock = make(page_size=500)
    messages = []
    out = p(["AAPL"], START, progress=messages.append)["AAPL"]
    assert isinstance(out, pd.DataFrame) and out.index[0] <= pd.Timestamp("2018-01-05") and out.index.is_monotonic_increasing and out.index.is_unique
    assert [c["kType"] for c in fake.calls] == ["8"] * len(fake.calls) and all(c["region"] == "US" and c["code"] == "AAPL" for c in fake.calls)
    assert "et" not in fake.calls[0] and all("et" in c for c in fake.calls[1:])
    ets = [int(c["et"]) for c in fake.calls[1:]]
    assert ets == sorted(ets, reverse=True)                                         # each page ends just before the oldest bar of the one before
    assert messages and all(m.startswith("iTick AAPL (1 of 1)") for m in messages)


def test_the_bars_of_a_long_history_have_no_gaps_or_repeats_across_pages():
    p, fake, _ = make(FakeITick(bars=2600), page_size=400)
    out = p(["MSFT"], "2012-01-01")["MSFT"]
    expected = pd.bdate_range(end=fake.last, periods=2600)
    assert list(out.index[-len(out):]) == list(expected[-len(out):]) and len(out) >= 2000


def test_today_s_unfinished_bar_is_never_used():
    fake = FakeITick(include_today=True)
    p, _, _ = make(fake)
    out = p(["AAPL"], START)["AAPL"]
    assert out.index[-1] < pd.Timestamp.now().normalize()


def test_a_short_history_costs_one_extra_call_to_learn_it_has_ended():
    fake = FakeITick()
    fake.history_start["NEWCO"] = fake.last - pd.Timedelta(days=400)
    p, _, _ = make(fake, page_size=1000)
    out = p(["NEWCO"], START)["NEWCO"]
    assert 250 < len(out) < 330 and len(fake.calls) == 2


def test_a_server_that_returns_fewer_bars_than_asked_still_delivers_the_history_within_the_page_limit():
    fake = FakeITick(cap=100)
    p, _, _ = make(fake, page_size=1000, max_pages=50)
    out = p(["AAPL"], "2023-06-01")["AAPL"]
    assert out.index[0] <= pd.Timestamp("2023-06-05") and len(fake.calls) >= 5


def test_hitting_the_page_limit_is_reported_as_a_note_on_the_ticker():
    fake = FakeITick(cap=100)
    p, _, _ = make(fake, page_size=1000, max_pages=3)
    out = p(["AAPL"], START)["AAPL"]
    assert len(fake.calls) == 3 and any("page limit" in w and "ITICK_MAX_PAGES" in w for w in out.attrs["warnings"])


def test_a_server_that_ignores_et_does_not_loop_forever():
    fake = FakeITick()
    original = fake.__call__

    def ignoring(url, headers, timeout):
        return original(url.split("&et=")[0], headers, timeout)

    p, _, _ = make(fake, page_size=300)
    p.transport = ignoring
    out = p(["AAPL"], START)["AAPL"]
    assert isinstance(out, pd.DataFrame) and len(fake.calls) <= 3


def test_an_unadjusted_split_in_the_data_comes_back_as_a_warning_not_a_change():
    fake = FakeITick(split=("2022-06-01", 4.0))
    p, _, _ = make(fake)
    frame = p(["AAPL"], START)["AAPL"]
    assert any("looks like a 4.0-for-1 split" in w for w in frame.attrs["warnings"])
    assert frame["close"].loc["2022-05-31"] / frame["close"].loc["2022-06-01"] > 3                     # prices are delivered as they are, never silently rewritten


def test_crypto_and_forex_use_their_own_endpoints_and_are_not_checked_for_splits():
    fake = FakeITick(split=("2022-06-01", 4.0))
    p, _, _ = make(fake)
    out = p(["BTC-USD", "EURUSD=X"], START)
    assert fake.urls[0].startswith("https://api-free.itick.org/crypto/kline?") and "region=BA" in fake.urls[0] and "code=BTCUSDT" in fake.urls[0]
    forex = next(u for u in fake.urls if "/forex/kline" in u)
    assert "region=GB" in forex and "code=EURUSD" in forex
    assert not out["BTC-USD"].attrs["warnings"] and not out["EURUSD=X"].attrs["warnings"]


def test_a_top_up_asks_for_one_call_and_matches_a_full_download():
    full_fake = FakeITick()
    p, fake, clock = make(full_fake, page_size=500)
    full = p(["AAPL"], START)["AAPL"]
    older = full.iloc[:-6]                                                           # what a copy saved six sessions ago would hold
    calls_before = len(fake.calls)
    topped = p(["AAPL"], START, known={"AAPL": older})["AAPL"]
    assert len(fake.calls) - calls_before == 1
    pd.testing.assert_frame_equal(topped[["close"]], full[["close"]], check_freq=False)
    assert topped.index[-1] == full.index[-1]


def test_a_top_up_notices_when_the_history_was_restated_and_downloads_it_again():
    fake = FakeITick()
    p, _, _ = make(fake, page_size=500)
    full = p(["AAPL"], START)["AAPL"]
    older = full.iloc[:-6] * 0.5                                                      # the saved copy is on a different price basis than iTick now returns
    calls_before = len(fake.calls)
    redone = p(["AAPL"], START, known={"AAPL": older})["AAPL"]
    assert len(fake.calls) - calls_before > 1
    pd.testing.assert_frame_equal(redone[["close"]], full[["close"]], check_freq=False)


def test_each_pagination_call_is_rate_limited_and_the_wait_is_reported():
    p, fake, clock = make(page_size=200, calls=5)
    heard = []
    p(["AAPL"], "2016-01-01", progress=heard.append)
    assert len(fake.calls) > 5 and sum(clock.slept) >= 60.0
    assert any("waiting" in m and "5 calls a minute" in m for m in heard)


# ---------------------------------------------------------------------------------------------------------------------------- failures
def test_a_rejected_key_stops_the_whole_download_and_says_so_without_more_calls():
    fake = FakeITick(key="a-different-key-entirely")
    p, _, _ = make(fake)
    out = p(["AAPL", "MSFT", "NVDA"], START)
    assert all("rejected the API key" in out[t] for t in out) and len(fake.calls) == 1


def test_http_401_is_also_a_rejected_key():
    p, fake, _ = make()
    fake.script = [(401, {}, b"")]
    out = p(["AAPL", "MSFT"], START)
    assert "rejected the API key" in out["AAPL"] and out["MSFT"] == out["AAPL"] and len(fake.calls) == 1


def test_an_unknown_symbol_fails_alone_and_the_others_still_download():
    p, fake, _ = make()
    out = p(["NOPE1", "AAPL"], START)
    assert "does not have this symbol" in out["NOPE1"] and isinstance(out["AAPL"], pd.DataFrame)


def test_a_symbol_iTick_cannot_place_is_reported_without_a_call():
    p, fake, _ = make()
    assert "cannot place" in p(["A$B"], START)["A$B"] and fake.calls == []


def test_http_429_pauses_for_the_time_the_service_asks_then_retries():
    p, fake, clock = make()
    fake.script = [(429, {"Retry-After": "45"}, b"slow down")]
    out = p(["AAPL"], START)
    assert isinstance(out["AAPL"], pd.DataFrame) and sum(clock.slept) >= 45.0 and len(fake.calls) >= 2


def test_endless_429s_end_in_a_message_after_three_tries():
    p, fake, _ = make()
    fake.script = [(429, {}, b"")] * 10
    out = p(["AAPL"], START)["AAPL"]
    assert "call limit was exceeded" in out and len(fake.calls) == 3


def test_server_errors_are_retried_then_reported():
    p, fake, clock = make()
    fake.script = [(503, {}, b"")] * 3
    assert "server problem" in p(["AAPL"], START)["AAPL"] and len(fake.calls) == 3
    fake2 = FakeITick()
    p2, _, _ = make(fake2)
    fake2.script = [(502, {}, b"")]
    assert isinstance(p2(["AAPL"], START)["AAPL"], pd.DataFrame)                  # one hiccup is absorbed


def test_a_dropped_connection_is_retried_then_explained():
    calls = []

    def flaky(url, headers, timeout):
        calls.append(url)
        raise TimeoutError("timed out")

    p, _, _ = make()
    p.transport = flaky
    out = p(["AAPL"], START)["AAPL"]
    assert "could not reach iTick" in out and "ITICK_BASE_URL" in out and len(calls) == 3


def test_answers_that_are_not_json_or_not_ok_are_reported():
    p, fake, _ = make()
    fake.script = [(200, {}, b"<html>maintenance</html>")]
    assert "not JSON" in p(["AAPL"], START)["AAPL"]
    fake.script = [(200, {}, json.dumps({"code": "E003", "msg": "exceeding the maximum subscription limit"}).encode())]
    out = p(["AAPL", "MSFT"], START)
    assert "plan does not allow" in out["AAPL"] and out["MSFT"] == out["AAPL"]
    fake.script = [(404, {}, b"nope")]
    assert "HTTP 404" in p(["AAPL"], START)["AAPL"]
    fake.script = [(200, {}, json.dumps({"code": 0, "msg": None, "data": "weird"}).encode())]
    assert "no list of bars" in p(["AAPL"], START)["AAPL"]
    fake.script = [(200, {}, json.dumps({"code": 0, "msg": None, "data": []}).encode())]
    assert "no daily bars" in p(["AAPL"], START)["AAPL"]


def test_an_account_that_returns_hourly_bars_is_caught_before_it_poisons_a_backtest():
    p, fake, _ = make(FakeITick(daily=False))
    out = p(["AAPL", "MSFT"], START)
    assert "not daily bars" in out["AAPL"] and out["MSFT"] == out["AAPL"] and len(fake.calls) == 1
    again, _, _ = make(FakeITick(daily=False), daily_ktype=5)
    assert again.daily_ktype == 5                                                   # the override exists for the account that really does use another code


def test_a_download_that_would_take_too_many_calls_is_refused_up_front_with_the_cost():
    p, fake, _ = make(max_calls_per_run=10, page_size=100)
    out = p(["AAPL", "MSFT"], "2005-01-01")
    assert all("iTick calls" in v and "more than the 10" in v for v in out.values()) and fake.calls == []


def test_the_limiter_waits_are_reported_to_a_caller_that_asked():
    p, fake, clock = make(calls=1)
    heard = []
    p(["AAPL", "MSFT"], "2024-01-01", progress=heard.append)
    assert any("waiting" in m for m in heard) and sum(clock.slept) >= 60.0


# --------------------------------------------------------------------------------------------------------------------------- self-test
def test_the_connection_test_spends_one_call_and_reports_what_came_back():
    p, fake, _ = make()
    out = p.test()
    assert out["ok"] and out["bars"] == 10 and len(fake.calls) == 1 and "iTick works" in out["message"] and KEY not in json.dumps(out)
    assert fake.calls[0]["limit"] == "10" and fake.calls[0]["code"] == "AAPL"


def test_the_connection_test_explains_each_way_it_can_fail_without_the_key():
    bad, _, _ = make(FakeITick(key="other-key-entirely"))
    out = bad.test()
    assert not out["ok"] and "rejected" in out["message"] and KEY not in json.dumps(out)
    hourly, _, _ = make(FakeITick(daily=False))
    assert "not daily bars" in hourly.test()["message"]
    busy, _, clock = make(calls=1)
    busy.limiter.acquire()
    assert "in use right now" in busy.test()["message"]
    empty, fake, _ = make()
    fake.script = [(200, {}, json.dumps({"code": 0, "msg": None, "data": []}).encode())]
    assert "no daily bars" in empty.test()["message"]


def test_the_environment_can_tune_the_client(monkeypatch):
    monkeypatch.setenv("ITICK_CALLS_PER_MINUTE", "120")
    monkeypatch.setenv("ITICK_PAGE_SIZE", "250")
    monkeypatch.setenv("ITICK_MAX_PAGES", "4")
    monkeypatch.setenv("ITICK_DAILY_KTYPE", "2")
    monkeypatch.setenv("ITICK_BASE_URL", "https://api.itick.org/")
    p = ITickProvider(key=KEY)
    assert (p.limiter.calls, p.page_size, p.max_pages, p.daily_ktype, p.base_url) == (120, 250, 4, 2, "https://api.itick.org")
    monkeypatch.setenv("ITICK_CALLS_PER_MINUTE", "lots")
    assert ITickProvider(key=KEY).limiter.calls == 5                                 # a typo falls back to the free plan's allowance
    monkeypatch.setenv("ITICK_CALLS_PER_MINUTE", "99999")
    assert ITickProvider(key=KEY).limiter.calls == 1200


def test_provider_module_has_no_logging_of_requests():
    source = open(providers.__file__, encoding="utf-8").read()
    assert "logging" not in source and "print(" not in source                       # nothing here writes the URL, the headers or the key anywhere


def test_the_command_line_check_reports_without_printing_the_key(monkeypatch, capsys, tmp_path):
    from src import cli

    monkeypatch.delenv("ITICK_API_KEY", raising=False)
    assert cli.main(["itick-test"]) == 1
    assert "FAILED: iTick is not set up" in capsys.readouterr().out
    monkeypatch.setenv("ITICK_API_KEY", KEY)
    monkeypatch.setattr(providers.ITickProvider, "_bars", lambda self, inst, limit, end_ms, say=None: (_ for _ in ()).throw(ITickError(f"rejected {KEY}", fatal=True)))
    assert cli.main(["itick-test", "--symbol", "MSFT"]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and KEY not in out


def test_the_repository_keeps_secrets_out_of_git():
    """``.env`` and the downloaded-data folder are git-ignored, and the example file carries no value. (Skipped where the repository files are not present, as in the Docker image.)"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ignore, example = root / ".gitignore", root / ".env.example"
    if not ignore.exists() or not example.exists():
        pytest.skip("repository files not present")
    lines = {line.strip() for line in ignore.read_text().splitlines()}
    assert ".env" in lines and "data/user/" in lines
    values = [line.split("=", 1)[1].strip() for line in example.read_text().splitlines() if line.startswith("ITICK_API_KEY=")]
    assert values == [""]                                                           # a placeholder, never a real key
    for name in (".replit", "replit.md", "README.md"):
        path = root / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            assert "ITICK_API_KEY=" not in text.replace("ITICK_API_KEY=...", "")      # no assignment of a real value in a committed file
