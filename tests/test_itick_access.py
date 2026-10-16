"""Why iTick may not accept a key, and what the app does about it.

iTick has two documented hosts ("Free trial api url" and "Production api url"), a gateway that answers a key it does not know with HTTP 401 and ``{"message": "Invalid API key in request"}``, and
plans that may not include every kind of data. The app has to tell those apart for the user: a key pasted with quotes, a key the free host does not know but the production host does, a key that is
fine but not allowed futures, and a key that is simply wrong. Nothing here touches the network: the gateway is ``tests/fake_itick.py``, which answers the way the live one was seen to answer.
"""

from __future__ import annotations

import json

import pytest

from src.webapp.providers import FREE_HOST, PAID_HOST, ITickError, ITickProvider, RateLimiter, future_instrument, tidy_key
from tests.fake_itick import KEY, FakeClock, FakeITick
from tests.test_webapp_data import run
from tests.test_webapp_data_itick import _provider, _use, ik  # noqa: F401  (the app with a fake iTick behind it)

START = "2024-01-01"
ES = future_instrument("US", "ES")


@pytest.fixture(autouse=True)
def _no_chosen_address(monkeypatch):
    monkeypatch.delenv("ITICK_BASE_URL", raising=False)


def auto(fake, key=KEY, **kwargs):
    """A provider with no address chosen, so both of iTick's hosts may be tried, driven by ``fake``: ``(provider, clock)``."""
    clock = FakeClock()
    return ITickProvider(key=key, limiter=RateLimiter(kwargs.pop("calls", 5), clock=clock.now, sleep=clock.sleep), transport=fake, sleep=clock.sleep, **kwargs), clock


def paid_only(**kwargs):
    """A service whose free host does not know the key and whose production host does, answering like the live gateway."""
    return FakeITick(hosts={"api.itick.org": KEY, "api-free.itick.org": "some-other-key"}, gateway=True, **kwargs)


# ------------------------------------------------------------------------------------------------------------------------------ the key as pasted
@pytest.mark.parametrize("raw, expected, said", [
    ("abc123def456ghi789", "abc123def456ghi789", []),
    ('  "abc123def456ghi789"\n', "abc123def456ghi789", ["around it", "quotation marks"]),
    ("'abc123def456ghi789'", "abc123def456ghi789", ["quotation marks"]),
    ("ITICK_API_KEY=abc123def456ghi789", "abc123def456ghi789", ["secret's name"]),
    ("export ITICK_API_KEY='abc123def456ghi789'", "abc123def456ghi789", ["quotation marks", "secret's name"]),
    ("Bearer abc123def456ghi789", "abc123def456ghi789", ["Bearer"]),
    ("token: abc123def456ghi789", "abc123def456ghi789", ["label"]),
    ("abc123\u200bdef456ghi789", "abc123def456ghi789", ["invisible"]),
    ("abc123 def456", "abc123 def456", ["space or a line break inside"]),
    ("abc123édef456", "abc123édef456", ["ASCII"]),
    ("YOUR_API_KEY", "YOUR_API_KEY", ["placeholder"]),
    ("<your key>", "<your key>", ["space or a line break inside", "placeholder"]),
    ("", "", []),
    (None, "", []),
])
def test_a_pasted_key_is_cleaned_and_what_is_still_wrong_is_named(raw, expected, said):
    key, notes = tidy_key(raw)
    assert key == expected
    assert len(notes) == len(said) and all(any(part in note for note in notes) for part in said), notes
    assert not any(key and key in note for note in notes), "a note never contains the key"


def test_a_key_pasted_with_quotes_is_sent_clean_and_neither_form_reaches_a_message():
    fake = FakeITick(gateway=True)
    p, _ = auto(fake, key=f' "{KEY}"\n')
    p.series(ES, START)
    assert set(fake.tokens) == {KEY}, "the header carries the key alone"
    assert any("quotation marks" in note for note in p.key_notes())
    wrong = FakeITick(key="the-key-this-service-knows", gateway=True)
    p, _ = auto(wrong, key=f'"{KEY}"')
    with pytest.raises(ITickError) as caught:
        p.series(ES, START)
    assert KEY not in str(caught.value) and f'"{KEY}"' not in str(caught.value)


# ------------------------------------------------------------------------------------------------------------------------------ the two hosts
def test_the_free_host_is_where_a_key_goes_first_and_the_header_is_token():
    fake = FakeITick(gateway=True)
    p, _ = auto(fake)
    assert p.host == FREE_HOST == p.base_url and p.describe()["host"] == "api-free.itick.org"
    p.series(ES, START)
    assert set(fake.hosts_called) == {"api-free.itick.org"} and set(fake.tokens) == {KEY} and fake.urls[0].startswith("https://api-free.itick.org/future/kline?")


def test_a_key_the_free_host_does_not_know_is_tried_on_the_production_host_once_and_remembered():
    fake = paid_only()
    p, _ = auto(fake)
    said = []
    frame = p.series(ES, START, progress=said.append)
    assert len(frame) > 100
    assert fake.hosts_called[:2] == ["api-free.itick.org", "api.itick.org"] and set(fake.hosts_called[1:]) == {"api.itick.org"}, "one refusal, then the other host for everything"
    assert any("api-free.itick.org did not recognise the key; trying api.itick.org" in text for text in said)
    assert p.host == PAID_HOST and p.describe()["host"] == "api.itick.org" and p.base_url == FREE_HOST
    before = len(fake.calls)
    p.series(ES, "2024-06-01")
    assert len(fake.calls) > before and set(fake.hosts_called[before:]) == {"api.itick.org"}, "later calls go straight to the host that knows the key"
    assert fake.hosts_called.count("api-free.itick.org") == 1


def test_the_application_s_own_auth_failure_is_handled_like_the_gateway_s():
    fake = FakeITick(hosts={"api.itick.org": KEY, "api-free.itick.org": "some-other-key"})              # the free host answers 200 with {"code": "E002", "msg": "auth failed"}
    p, _ = auto(fake)
    assert len(p.series(ES, START)) > 100
    assert fake.hosts_called[:2] == ["api-free.itick.org", "api.itick.org"] and p.host == PAID_HOST


def test_a_changed_key_forgets_the_remembered_host_and_the_remembered_host_can_be_wrong_again():
    fake = paid_only()
    p, _ = auto(fake)
    p.series(ES, START)
    assert p.host == PAID_HOST
    p._key_override = "another-key-0123456789"
    assert p.host == FREE_HOST, "a different key starts again from the free host"
    p._key_override = KEY
    fake.hosts = {"api-free.itick.org": KEY, "api.itick.org": "some-other-key"}                      # the account moved to the free plan: the remembered host no longer knows the key
    fake.hosts_called.clear()
    p.series(ES, "2024-06-01")
    assert fake.hosts_called[:2] == ["api.itick.org", "api-free.itick.org"] and p.host == FREE_HOST


def test_when_neither_host_knows_the_key_the_message_gives_both_answers_and_the_next_steps_and_never_the_key():
    fake = FakeITick(hosts={"api.itick.org": "x-key-0000000000", "api-free.itick.org": "y-key-0000000000"}, gateway=True)
    p, _ = auto(fake)
    with pytest.raises(ITickError) as caught:
        p.series(ES, START)
    message = str(caught.value)
    assert caught.value.fatal and len(fake.calls) == 2, "two hosts, one call each, then stop"
    assert 'api-free.itick.org said "Invalid API key in request" (HTTP 401)' in message and 'api.itick.org said "Invalid API key in request" (HTTP 401)' in message
    assert "rejected the API key" in message and "ITICK_API_KEY" in message and "stop and run the app again" in message and "time-limited free trial" in message
    assert KEY not in message


def test_an_address_chosen_by_the_user_is_the_only_one_asked_whether_by_argument_or_environment(monkeypatch):
    fake = paid_only()
    p, _ = auto(fake, base_url=FREE_HOST)
    with pytest.raises(ITickError, match="ITICK_BASE_URL is set, so iTick's other host was not tried"):
        p.series(ES, START)
    assert fake.hosts_called == ["api-free.itick.org"] and not p.auto_host
    monkeypatch.setenv("ITICK_BASE_URL", PAID_HOST)
    fake = FakeITick(hosts={"api.itick.org": "x-key-0000000000"}, gateway=True)
    p, _ = auto(fake)
    with pytest.raises(ITickError):
        p.series(ES, START)
    assert fake.hosts_called == ["api.itick.org"] and p.host == PAID_HOST


# ------------------------------------------------------------------------------------------------------------------------------ a plan, not a key
def test_a_403_says_the_key_may_be_fine_and_the_plan_may_lack_the_data_and_does_not_try_the_other_host():
    fake = FakeITick(gateway=True, forbidden={"future": "You cannot consume this service"})
    p, _ = auto(fake)
    with pytest.raises(ITickError) as caught:
        p.series(ES, START)
    message = str(caught.value)
    assert caught.value.fatal and len(fake.calls) == 1 and fake.hosts_called == ["api-free.itick.org"]
    assert 'said "You cannot consume this service" (HTTP 403)' in message and "refused this request" in message and "futures prices" in message and "may be fine" in message
    assert "rejected the API key" not in message


def test_the_same_key_still_downloads_what_the_plan_does_include():
    fake = FakeITick(gateway=True, forbidden={"future": "You cannot consume this service"})
    p, _ = auto(fake)
    frames = p(["AAPL"], START)
    assert len(frames["AAPL"]) > 100 and fake.urls[0].startswith("https://api-free.itick.org/stock/kline?")


# ------------------------------------------------------------------------------------------------------------------------------ the connection test
def test_the_connection_test_names_the_host_and_asks_for_the_tab_s_own_data():
    fake = paid_only()
    p, _ = auto(fake)
    out = p.test(kind="futures")
    assert out["ok"] and out["host"] == "api.itick.org" and "iTick works at api.itick.org (futures data)" in out["message"]
    assert "not recognised at api-free.itick.org, so the app now uses api.itick.org" in out["message"]
    assert "/future/kline?region=US&code=ES" in fake.urls[-1] and len(fake.calls) == 2, "one refusal and one answer"
    crypto = p.test(kind="crypto")
    assert crypto["ok"] and "/crypto/kline?region=BA&code=BTCUSDT" in fake.urls[-1]
    plain = p.test()
    assert plain["ok"] and "/stock/kline?region=US&code=AAPL" in fake.urls[-1] and "(futures data)" not in plain["message"]
    assert p.test(kind="nonsense")["ok"] and "/stock/kline?region=US&code=AAPL" in fake.urls[-1], "a kind it does not know is ignored"


def test_the_connection_test_tells_a_plan_that_lacks_futures_from_a_key_that_is_wrong():
    fake = FakeITick(gateway=True, forbidden={"future": "You cannot consume this service"})
    p, _ = auto(fake)
    futures = p.test(kind="futures")
    assert not futures["ok"] and "refused this request" in futures["message"] and "may be fine" in futures["message"]
    stock = p.test()
    assert stock["ok"], "the same key works for stocks"
    wrong, _ = auto(FakeITick(key="the-key-this-service-knows", gateway=True))
    bad = wrong.test(kind="futures")
    assert not bad["ok"] and "rejected the API key" in bad["message"] and "api.itick.org said" in bad["message"] and KEY not in json.dumps(bad)


def test_the_connection_test_mentions_what_was_wrong_with_the_pasted_key():
    p, _ = auto(FakeITick(key="the-key-this-service-knows", gateway=True), key=f"'{KEY}'")
    out = p.test()
    assert not out["ok"] and "Also: quotation marks around it were removed." in out["message"] and out["notes"] == ["quotation marks around it were removed"]
    assert KEY not in json.dumps(out)
    nokey, _ = auto(FakeITick(gateway=True), key="")
    assert nokey.test() == {"ok": False, "message": nokey.available()[1], "notes": []}


def test_the_source_test_endpoint_passes_the_tab_kind_on(ik):  # noqa: F811
    app, fake, _ = ik
    out = app.test_source({"source": "itick", "kind": "futures"})
    assert out["ok"] and "(futures data)" in out["message"] and "/future/kline?region=US&code=ES" in fake.urls[-1] and out["host"] == "api-free.itick.org"
    assert app.test_source({"source": "itick"})["ok"] and "/stock/kline?region=US&code=AAPL" in fake.urls[-1]
    assert app.test_source({"source": "itick", "kind": "../../etc"})["ok"] and "/stock/kline" in fake.urls[-1], "only the two known kinds change what is asked"
    assert KEY not in json.dumps(out)


# ------------------------------------------------------------------------------------------------------------------------------ "not found" is not "wrong key"
def missing(*codes, **kwargs):
    """A service for which the codes asked are unknown (the application's E001 "not found this produce"): the key itself is fine."""
    fake = FakeITick(**kwargs)
    fake.unknown_prefix = tuple(codes) if len(codes) > 1 else codes[0]
    return fake


def test_a_product_iTick_does_not_return_is_explained_as_a_code_or_a_plan_problem_not_a_bad_key():
    fake = missing("ES")
    p, _ = auto(fake)
    with pytest.raises(ITickError) as caught:
        p.series(ES, START)
    message = str(caught.value)
    assert caught.value.kind == "symbol" and not caught.value.fatal, "one missing product does not stop a run: the next may work"
    assert "does not have this symbol for your key (E001 not found this produce)" in message and "added to your plan" in message and "customer service" in message
    assert "pricing page does not list futures" in message and "Test the iTick connection" in message
    assert "rejected the API key" not in message and KEY not in message


def test_the_test_asks_for_the_product_iTick_uses_in_its_documentation_and_says_what_the_two_answers_mean():
    p, _ = auto(missing("ES"))
    out = p.test(kind="futures")                                                       # ES is not found, NQ is: the plan has futures, this product was not added
    assert not out["ok"] and out["control"] == {"item": "US:NQ", "ok": True}
    assert "no US:ES for your key, but it does return US:NQ" in out["message"] and "has probably not been added to your plan" in out["message"]
    none, _ = auto(missing("ES", "NQ"))
    out = none.test(kind="futures")                                                    # neither: the plan has no futures
    assert not out["ok"] and out["control"] == {"item": "US:NQ", "ok": False}
    assert "returns neither US:ES nor US:NQ" in out["message"] and "does not include futures data" in out["message"] and "Yahoo Finance has futures without a key" in out["message"]
    coin, _ = auto(missing("BTCUSDT"))
    out = coin.test(kind="crypto")
    assert not out["ok"] and out["control"] == {"item": "BA:ETHUSDT", "ok": True}
    again, _ = auto(missing("NQ"))
    out = again.test(symbol="NQ=F", kind="nonsense")                                   # a stock test (the kind is ignored) asks no control
    assert not out["ok"] and "control" not in out


def test_the_control_is_not_asked_when_the_tested_product_is_the_control_or_the_failure_is_something_else():
    fake = missing("NQ")
    fake.script = []
    p, _ = auto(fake)
    p.test(symbol="NQ=F")
    assert len(fake.calls) == 1, "a stock-style test of NQ itself asks nothing more"
    wrong, _ = auto(FakeITick(key="the-key-this-service-knows", gateway=True))
    out = wrong.test(kind="futures")
    assert "control" not in out and "rejected the API key" in out["message"]
    forbidden, _ = auto(FakeITick(gateway=True, forbidden={"future": "You cannot consume this service"}))
    out = forbidden.test(kind="futures")
    assert "control" not in out and "refused this request" in out["message"]


def test_what_iTick_returned_for_the_key_is_remembered_and_marked_in_the_search(ik, tmp_path):  # noqa: F811
    app, fake, clock = ik
    _use(app, _provider(clock, fake, root=tmp_path), tmp_path)                         # a provider that keeps its memory under the app's folder
    fake.unknown_prefix = "ES"
    result = run(app, kind="futures", source="itick", items=["US:ES", "US:NQ"], years=1)
    by = {r["item"]: r for r in result["results"]}
    assert result["failed"] == 1 and not by["US:ES"]["ok"] and "does not have this symbol for your key" in by["US:ES"]["message"] and by["US:NQ"]["ok"]
    found = app.extra_get["/api/data/search"]({"kind": "futures", "q": "nasdaq"})["results"]
    nq = next(r for r in found if r["item"] == "US:NQ")
    assert nq["access"]["ok"] is True and nq["access"]["at"]
    sp = app.extra_get["/api/data/search"]({"kind": "futures", "q": "e-mini s&p"})["results"]
    es = next(r for r in sp if r["item"] == "US:ES")
    assert es["access"]["ok"] is False and "E001" in es["access"]["why"] and KEY not in json.dumps(es)
    assert all("access" not in r for r in app.extra_get["/api/data/search"]({"kind": "futures", "q": "crude"})["results"]), "a product never asked for has no mark"
    saved = json.loads((tmp_path / "data" / "user" / "itick_access.json").read_text())
    assert saved["future:US:ES"]["ok"] is False and saved["future:US:NQ"]["ok"] is True and KEY not in json.dumps(saved), "product codes and dates only"
    app.builder.sources["itick"].provider._access.clear()
    app.builder.sources["itick"].provider._load_access()
    assert set(app.builder.sources["itick"].provider.access("future")) == {"US:ES", "US:NQ"}, "the memory survives a restart"


def test_the_memory_of_what_iTick_returned_forgets_after_a_month(tmp_path):
    p = ITickProvider(key=KEY, root=tmp_path, base_url=FREE_HOST)
    p.note_access("future", "US:ES", False, "E001")
    p._access["future:US:OLD"] = {"ok": True, "at": "2020-01-01", "why": ""}
    assert set(p.access("future")) == {"US:ES"} and p.access("crypto") == {}
