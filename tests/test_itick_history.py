"""What the app remembers about the iTick key, so that a refusal comes with a diagnosis: when iTick last took this key, whether it is still the same key, and the address iTick sees the app at. Nothing
here touches the network: the gateway is ``tests/fake_itick.py`` and the address lookup is a function the test gives."""

from __future__ import annotations

import json

import pytest

from src.webapp import providers
from src.webapp.providers import ITickError, ITickProvider, RateLimiter, future_instrument, lookup_egress_ip
from tests.fake_itick import KEY, FakeClock, FakeITick

START = "2024-01-01"
STOCK = providers.map_symbol("AAPL")
ES = future_instrument("US", "ES")
OTHER = "ANOTHER-KEY-0123456789"


@pytest.fixture(autouse=True)
def _no_chosen_address(monkeypatch):
    monkeypatch.delenv("ITICK_BASE_URL", raising=False)


def provider(fake, root, key=KEY, **kwargs):
    clock = FakeClock()
    return ITickProvider(key=key, root=root, limiter=RateLimiter(5, clock=clock.now, sleep=clock.sleep), transport=fake, sleep=clock.sleep, **kwargs)


def refused(root, key, **kwargs):
    """A provider with ``key`` that neither of iTick's hosts accepts, and the message of the refusal it raises."""
    p = provider(FakeITick(hosts={"api.itick.org": "some-key-0000000000", "api-free.itick.org": "another-key-0000000000"}, gateway=True), root, key=key, **kwargs)
    with pytest.raises(ITickError) as caught:
        p.series(ES, START)
    return p, str(caught.value)


def test_a_call_iTick_accepts_is_remembered_with_when_where_and_what_for_and_never_with_the_key(tmp_path):
    p = provider(FakeITick(), tmp_path)
    assert p.history() == {"accepted": None, "refused": None, "same_key": None}
    p.series(STOCK, START)
    h = p.history()
    assert h["accepted"]["host"] == "api-free.itick.org" and h["accepted"]["kind"] == "stock" and h["accepted"]["at"].endswith("Z") and h["refused"] is None and h["same_key"] is True
    assert "fp" not in h["accepted"], "the mark of the key stays on disk"
    saved = (tmp_path / "data" / "user" / "itick_history.json").read_text(encoding="utf-8")
    assert KEY not in saved and json.loads(saved)["accepted"]["fp"] and p.describe()["history"] == h, "the status shows it, the file does not hold the key"


def test_a_key_iTick_stops_accepting_is_reported_with_the_day_it_last_worked_and_the_address_iTick_sees(tmp_path):
    provider(FakeITick(), tmp_path).series(STOCK, START)                                    # it worked
    p, message = refused(tmp_path, KEY, ip_lookup=lambda: "203.0.113.9")                      # and now neither host knows it: the same key, later
    assert "rejected the API key" in message and "This same key was accepted on" in message and "(stock prices)" in message and "ended free trial" in message
    assert "iTick sees this app at 203.0.113.9" in message and "limited to certain addresses" in message
    assert KEY not in message
    h = p.history()
    assert h["refused"]["http"] == 401 and h["refused"]["said"] == "Invalid API key in request" and h["same_key"] is True
    assert h["accepted"]["at"] <= h["refused"]["at"]


def test_a_key_that_was_changed_since_it_worked_is_said_to_have_been_changed(tmp_path):
    provider(FakeITick(), tmp_path).series(STOCK, START)
    p, message = refused(tmp_path, OTHER, ip_lookup=lambda: "203.0.113.9")
    assert "is not the key in the secret now: the secret was changed since" in message and "ended free trial" not in message
    assert p.history()["same_key"] is False and OTHER not in message and KEY not in message
    saved = (tmp_path / "data" / "user" / "itick_history.json").read_text(encoding="utf-8")
    assert OTHER not in saved and KEY not in saved


def test_a_key_that_never_worked_has_no_history_to_quote_and_the_address_is_only_asked_when_a_lookup_was_given(tmp_path):
    p, message = refused(tmp_path, OTHER)
    assert "was accepted on" not in message and "iTick sees this app" not in message, "no lookup was given, so no outside call is made"
    assert p.history()["accepted"] is None and p.history()["same_key"] is None and p.history()["refused"]["http"] == 401


def test_the_address_lookup_is_asked_once_and_a_failing_lookup_is_not_a_failure(tmp_path):
    asked = []

    def lookup():
        asked.append(1)
        return "198.51.100.4"

    p, first = refused(tmp_path, OTHER, ip_lookup=lookup)
    with pytest.raises(ITickError) as again:
        p.series(ES, START)
    assert "198.51.100.4" in first and "198.51.100.4" in str(again.value) and len(asked) == 1, "one lookup serves the refusals of the next ten minutes"

    def broken():
        raise OSError("no route")

    _, message = refused(tmp_path / "other", OTHER, ip_lookup=broken)
    assert "rejected the API key" in message and "iTick sees this app" not in message


def test_a_403_gets_the_same_diagnosis(tmp_path):
    provider(FakeITick(), tmp_path).series(STOCK, START)
    p = provider(FakeITick(gateway=True, forbidden={"future": "Your IP address is not allowed"}), tmp_path, ip_lookup=lambda: "203.0.113.9")
    with pytest.raises(ITickError) as caught:
        p.series(ES, START)
    message = str(caught.value)
    assert "Your IP address is not allowed" in message and "(HTTP 403)" in message and "iTick sees this app at 203.0.113.9" in message and "This same key was accepted on" in message
    assert p.history()["refused"]["http"] == 403


def test_a_product_that_is_not_found_still_counts_as_the_key_being_accepted(tmp_path):
    p = provider(FakeITick(), tmp_path)
    with pytest.raises(ITickError):
        p.series(providers.map_symbol("NOPE1"), START)
    assert p.history()["accepted"] is not None and p.history()["refused"] is None, "iTick took the key and then said it had no such product"


def test_the_same_event_is_not_written_again_within_a_minute(tmp_path):
    p = provider(FakeITick(), tmp_path)
    p.series(STOCK, START)
    path = tmp_path / "data" / "user" / "itick_history.json"
    first = path.stat().st_mtime_ns
    p.series(STOCK, START)
    assert path.stat().st_mtime_ns == first


def test_the_address_lookup_reads_plain_text_and_refuses_anything_else(monkeypatch):
    def serve(status, body):
        monkeypatch.setattr(providers, "urllib_transport", lambda url, headers, timeout: (status, {}, body))

    serve(200, b"203.0.113.9\n")
    assert lookup_egress_ip() == "203.0.113.9"
    serve(200, b"2001:db8::1")
    assert lookup_egress_ip() == "2001:db8::1"
    serve(200, b"<html>blocked</html>")
    assert lookup_egress_ip() is None
    serve(403, b"203.0.113.9")
    assert lookup_egress_ip() is None

    def down(url, headers, timeout):
        raise OSError("offline")

    monkeypatch.setattr(providers, "urllib_transport", down)
    assert lookup_egress_ip() is None
