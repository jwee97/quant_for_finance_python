"""The Finzdata client: where the key lives and that it never leaks, the day's call ledger, and each refusal in words fit for the page.

Nothing here touches the network: the client is driven through ``tests/fake_finzdata.py`` and a fake clock.
"""

from __future__ import annotations

import json

import pytest

from src.webapp.finzdata import BUDGET_VARIABLE, DEFAULT_BUDGET, NOT_CONFIGURED, URL_VARIABLE, CallLedger, FinzdataClient, FinzdataError
from tests.fake_finzdata import KEY, FakeFinzdata


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("FINZDATA_API_KEY", URL_VARIABLE, BUDGET_VARIABLE):
        monkeypatch.delenv(name, raising=False)


class Clock:
    def __init__(self, t: float = 1_793_000_000.0):          # 2026-10-26 UTC
        self.t = t

    def __call__(self) -> float:
        return self.t


def make(fake=None, tmp_path=None, **kwargs):
    fake = fake or FakeFinzdata()
    kwargs.setdefault("key", KEY)
    return FinzdataClient(root=tmp_path, base_url="https://api.finzdata.com", transport=fake, sleep=lambda s: None, pause=0.0, **kwargs), fake


# ------------------------------------------------------------------------------------------------------------------ the key
def test_the_key_comes_from_the_environment_then_a_dotenv_file_and_is_tidied(tmp_path, monkeypatch):
    assert not FinzdataClient(root=tmp_path).available()[0]
    (tmp_path / ".env").write_text(f"# a comment\nexport FINZDATA_API_KEY = '{KEY}'\n")
    c = FinzdataClient(root=tmp_path)
    assert c.available() == (True, "") and c._key() == KEY
    monkeypatch.setenv("FINZDATA_API_KEY", f'  "FINZDATA_API_KEY={KEY}"\n')                 # pasted with its name and quotes
    c = FinzdataClient(root=tmp_path)
    assert c._key() == KEY and any("pasted in front" in n for n in c.key_notes())


def test_a_missing_key_says_where_to_put_it_and_a_key_that_does_not_start_with_fz_is_noted_and_still_tried():
    assert make(key="")[0].available() == (False, NOT_CONFIGURED) and "FINZDATA_API_KEY" in NOT_CONFIGURED
    odd = make(key="abcdefghijklmnop")[0]
    assert odd.available()[0] and any("begin with fz_" in n for n in odd.key_notes())
    with pytest.raises(FinzdataError, match="not set up") as caught:
        make(key="")[0].get("/v1/futures/roots")
    assert caught.value.kind == "key" and caught.value.fatal


def test_the_key_goes_only_in_the_header_and_never_into_the_url_the_description_or_a_message(tmp_path):
    c, fake = make(tmp_path=tmp_path)
    c.roots()
    c.contracts("es")
    assert set(fake.keys) == {KEY} and all(KEY not in url for url in fake.urls)
    d = c.describe()
    assert KEY not in json.dumps(d) and d["configured"] and d["calls_today"] == 2 and d["host"] == "api.finzdata.com"
    assert not any(KEY in str(v) for v in d.values())


def test_a_refusal_that_echoes_the_key_is_redacted_before_it_reaches_the_user():
    fake = FakeFinzdata()
    fake.script = [(401, {}, json.dumps({"detail": f"invalid key {KEY}"}).encode())]
    c, _ = make(fake)
    with pytest.raises(FinzdataError) as caught:
        c.roots()
    assert KEY not in str(caught.value) and "***" in str(caught.value)


# ------------------------------------------------------------------------------------------------------------------ the day's calls
def test_every_call_is_counted_by_utc_day_and_the_count_survives_a_restart(tmp_path):
    clock = Clock()
    c, _ = make(tmp_path=tmp_path, clock=clock)
    c.roots(), c.contracts("ES")
    assert c.ledger.count() == 2
    again, _ = make(tmp_path=tmp_path, clock=clock)
    assert again.ledger.count() == 2 and again.describe()["calls_left"] == DEFAULT_BUDGET - 2
    clock.t += 86400
    assert again.ledger.count() == 0                                                     # a new UTC day


def test_the_client_refuses_to_send_more_than_the_budget_and_says_so_before_calling(tmp_path, monkeypatch):
    monkeypatch.setenv(BUDGET_VARIABLE, "2")
    c, fake = make(tmp_path=tmp_path)
    c.roots(), c.roots()
    with pytest.raises(FinzdataError, match=r"2 Finzdata calls have been made today.*FINZDATA_DAILY_CALLS") as caught:
        c.roots()
    assert caught.value.kind == "budget" and caught.value.fatal and len(fake.urls) == 2


@pytest.mark.parametrize("value, expected", [(None, DEFAULT_BUDGET), ("350", 350), ("x", DEFAULT_BUDGET), ("0", 1), ("99999999", 1_000_000)])
def test_the_budget_is_read_from_the_environment_and_kept_in_range(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(BUDGET_VARIABLE, value)
    assert make()[0].budget == expected


def test_a_ledger_that_cannot_be_read_or_written_never_stops_a_download(tmp_path):
    (tmp_path / "ledger.json").write_text("{not json")
    ledger = CallLedger(tmp_path / "ledger.json")
    assert ledger.count() == 0 and ledger.add() == 1
    blocked = CallLedger(tmp_path / "file" / "ledger.json")
    (tmp_path / "file").write_text("a file where a folder should be")
    assert blocked.add() == 1 and blocked.count() == 0                                     # the write fails quietly: nothing raised, and nothing remembered


# ------------------------------------------------------------------------------------------------------------------ refusals
@pytest.mark.parametrize("status, body, kind, fatal, words", [
    (401, {"detail": "invalid or missing API key"}, "key", True, "does not accept the key (HTTP 401: invalid or missing API key)"),
    (403, {"detail": "plan does not include this data"}, "plan", False, "plan for this key does not include this data (HTTP 403: plan does not include this data)"),
    (429, {"detail": "daily call limit reached"}, "quota", True, "limit of calls for the day is reached (HTTP 429: daily call limit reached)"),
    (422, {"detail": [{"loc": ["query", "root"], "msg": "Field required"}]}, "request", False, "did not understand the request for /v1/futures/roots (HTTP 422: root: Field required)"),
    (404, {"detail": "Not Found"}, "request", False, "(HTTP 404: Not Found)"),
])
def test_each_refusal_is_told_in_the_services_own_words_with_what_it_means(status, body, kind, fatal, words):
    fake = FakeFinzdata()
    fake.script = [(status, {}, json.dumps(body).encode())]
    c, _ = make(fake)
    with pytest.raises(FinzdataError) as caught:
        c.roots()
    assert (caught.value.status, caught.value.kind, caught.value.fatal) == (status, kind, fatal) and words in str(caught.value)


def test_a_server_error_is_tried_again_twice_and_then_reported():
    fake = FakeFinzdata()
    fake.script = [(503, {}, b"busy")] * 3
    c, _ = make(fake)
    with pytest.raises(FinzdataError, match="HTTP 503: busy") as caught:
        c.roots()
    assert caught.value.kind == "server" and len(fake.urls) == 3
    fake.script = [(502, {}, b"")]
    assert c.roots()["data"]                                                              # the next attempt (the fake's normal answer) works: a retry after one 502


def test_a_network_failure_is_retried_and_then_told_without_the_key():
    calls = []

    def transport(url, headers, timeout):
        calls.append(url)
        raise OSError(f"connection reset while sending {headers['X-API-Key']}")

    c = FinzdataClient(key=KEY, base_url="https://api.finzdata.com", transport=transport, sleep=lambda s: None, pause=0.0)
    with pytest.raises(FinzdataError, match="could not be reached") as caught:
        c.roots()
    assert len(calls) == 3 and caught.value.kind == "network" and KEY not in str(caught.value)
    assert c.ledger.count() == 0                                                          # a call that never got an answer is not one the service counted


def test_an_answer_that_is_not_json_is_a_server_problem_not_a_crash():
    fake = FakeFinzdata()
    fake.script = [(200, {}, b"<html>maintenance</html>")]
    with pytest.raises(FinzdataError, match="not JSON") as caught:
        make(fake)[0].roots()
    assert caught.value.kind == "server"


# ------------------------------------------------------------------------------------------------------------------ the address and the endpoints
def test_only_an_https_address_or_a_local_one_is_accepted_and_the_variable_chooses_it(monkeypatch):
    with pytest.raises(ValueError, match="https"):
        FinzdataClient(key=KEY, base_url="http://api.finzdata.com")
    with pytest.raises(ValueError, match="https"):
        FinzdataClient(key=KEY, base_url="ftp://example.com")
    assert FinzdataClient(key=KEY, base_url="http://127.0.0.1:8080/").base_url == "http://127.0.0.1:8080"
    monkeypatch.setenv(URL_VARIABLE, "https://example.test/")
    assert FinzdataClient(key=KEY).base_url == "https://example.test"


def test_the_endpoints_send_the_parameters_the_openapi_document_names():
    c, fake = make()
    c.roots()
    c.contracts("es")
    c.daily("ES", start="2026-09-01", limit=500)
    c.daily("CL", contract="CLZ26", start="2026-09-01T10:00:00", end="2026-10-01", continuous=True)
    assert fake.requests == [("/v1/futures/roots", {}), ("/v1/futures/contracts", {"root": "ES", "format": "json"}),
                             ("/v1/futures/daily", {"root": "ES", "adjust": "none", "start": "2026-09-01", "format": "json", "limit": "500"}),
                             ("/v1/futures/daily", {"root": "CL", "contract": "CLZ26", "continuous": "true", "adjust": "none", "start": "2026-09-01", "end": "2026-10-01", "format": "json"})]
    with pytest.raises(ValueError, match="not a date"):
        c.daily("ES", start="last tuesday")


def test_my_plan_and_usage_is_one_call_and_returned_as_the_service_sent_it():
    fake = FakeFinzdata()
    fake.script = [(200, {}, json.dumps({"plan": "free", "calls_today": 12}).encode())]
    c, _ = make(fake)
    assert c.me() == {"plan": "free", "calls_today": 12} and fake.urls[0].endswith("/v1/me")


def test_the_repository_names_the_finzdata_key_without_a_value_anywhere():
    """The example file carries a placeholder and no committed file assigns a real key. (Skipped where the repository files are not present, as in the Docker image.)"""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    example = root / ".env.example"
    if not example.exists():
        pytest.skip("repository files not present")
    assert [line.split("=", 1)[1].strip() for line in example.read_text().splitlines() if line.startswith("FINZDATA_API_KEY=")] == [""]
    for name in (".replit", "replit.md", "README.md", ".env.example", ".github/workflows/live-feeds.yml", ".github/workflows/ci.yml"):
        path = root / name
        if path.exists():
            assert not re.search(r"FINZDATA_API_KEY\s*[=:]\s*[\"']?fz_", path.read_text(encoding="utf-8")), name       # no assignment of a real value in a committed file
            assert not re.search(r"fz_[A-Za-z0-9]{16,}", path.read_text(encoding="utf-8")), name
