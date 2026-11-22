"""The helpers the keyed providers share: where a key lives, that it is redacted, the sliding-window rate limiter and the transport that never follows a redirect. Nothing here touches the network:
a fake clock makes a wait of a minute cost no time."""

from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from src.webapp.providers import RateLimiter, RateLimitTimeout, read_secret, redact, urllib_transport
from tests.fakes import FakeClock

KEY = "TESTKEY123456"


def test_the_key_comes_from_the_environment_then_a_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    assert read_secret("MASSIVE_API_KEY", tmp_path) == ""
    (tmp_path / ".env").write_text("# a comment\nOTHER=1\nexport MASSIVE_API_KEY = \"abc123secretvalue\"\n")
    assert read_secret("MASSIVE_API_KEY", tmp_path) == "abc123secretvalue"
    monkeypatch.setenv("MASSIVE_API_KEY", "from-the-environment")
    assert read_secret("MASSIVE_API_KEY", tmp_path) == "from-the-environment"
    assert read_secret("MASSIVE_API_KEY") == "from-the-environment"



def test_redact_removes_the_key_and_its_url_encoded_form_and_leaves_short_strings_alone():
    key = "ab/cd+ef=ghij"
    assert key not in redact(f"failed for {key} and ab%2Fcd%2Bef%3Dghij", key)
    assert redact("nothing to hide", key) == "nothing to hide"
    assert redact("abc", "abc") == "abc"                                         # too short to be a secret worth masking; masking it would wreck ordinary words




class _Redirecting(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_GET(self):                                                              # noqa: N802
        if self.server.server_port == self.server.first_port:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.second_port}/redirected")
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
        status, _, _ = urllib_transport(f"http://127.0.0.1:{first.server_port}/x?code=X", {"token": KEY}, 5.0)
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
