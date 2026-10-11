"""The plumbing of the free-data downloads: the HTTP client (retries, pacing, readable errors) and the store of downloaded files with the record of where each came from."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.assets.free_data.http import FetchError, Http
from src.assets.free_data.sources import SOURCES, describe
from src.assets.free_data.store import Download, DownloadStore
from tests.free_data_helpers import FakeFeeds


def _http(routes, **kw):
    sleeps: list[float] = []
    clock = {"t": 0.0}
    http = Http(FakeFeeds(routes), sleep=lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)), clock=lambda: clock["t"], **kw)
    return http, sleeps


# ------------------------------------------------------------------------------------------------------------------ the client
def test_get_json_returns_the_parsed_answer_and_adds_the_query():
    http, _ = _http({"https://a.test/x": {"ok": 1}})
    assert http.get_json("https://a.test/x", {"q": "BTC USD", "n": 2}) == {"ok": 1}
    assert http.transport.calls == ["https://a.test/x?q=BTC+USD&n=2"]


def test_a_rate_limited_or_failing_server_is_retried_then_succeeds():
    answers = iter([(429, "slow down"), (503, "busy"), (200, {"ok": True})])
    http, sleeps = _http({"https://a.test/": lambda url: next(answers)}, retries=2, backoff=1.0)
    assert http.get_json("https://a.test/x") == {"ok": True} and http.calls == 3
    assert len([s for s in sleeps if s >= 1.0]) == 2, "it waited before each retry"


def test_an_answer_that_means_not_from_here_or_not_found_is_not_retried_and_says_so():
    http, _ = _http({"https://a.test/blocked": (451, "no")})
    with pytest.raises(FetchError, match="does not serve this location") as caught:
        http.get_json("https://a.test/blocked")
    assert caught.value.blocked and caught.value.status == 451 and http.calls == 1
    http, _ = _http({"https://a.test/missing": (404, "no")})
    with pytest.raises(FetchError, match="no such item") as caught:
        http.get_json("https://a.test/missing")
    assert caught.value.status == 404 and not caught.value.blocked
    http, _ = _http({"https://a.test/denied": (403, "no")})
    with pytest.raises(FetchError, match="refused the request") as caught:
        http.get_json("https://a.test/denied")
    assert caught.value.blocked


def test_after_a_refusal_the_same_host_is_not_asked_again_for_a_while_but_a_missing_item_does_not_count():
    http, sleeps = _http({"https://a.test/limited": (429, "slow"), "https://a.test/ok": {"ok": 1}, "https://b.test/ok": {"ok": 2}}, retries=1, backoff=0.1, cooldown=60.0)
    with pytest.raises(FetchError, match="rate limiting"):
        http.get_json("https://a.test/limited")
    calls = http.calls
    with pytest.raises(FetchError, match=r"rate limiting.*not asked again") as caught:
        http.get_json("https://a.test/ok")
    assert http.calls == calls and caught.value.status == 429, "no request was made"
    assert http.get_json("https://b.test/ok") == {"ok": 2}, "other hosts are unaffected"
    http2, _ = _http({"https://a.test/missing": (404, "no"), "https://a.test/ok": {"ok": 1}})
    with pytest.raises(FetchError):
        http2.get_json("https://a.test/missing")
    assert http2.get_json("https://a.test/ok") == {"ok": 1}, "a 404 is about one item, not the host"
    # after the cooldown the host is tried again
    clock = {"t": 0.0}
    again = Http(FakeFeeds({"https://c.test/": (429, "slow")}), retries=0, cooldown=10.0, pause=0.0, sleep=lambda s: None, clock=lambda: clock["t"])
    with pytest.raises(FetchError):
        again.get_json("https://c.test/x")
    clock["t"] = 11.0
    with pytest.raises(FetchError, match=r"HTTP 429\)$"):
        again.get_json("https://c.test/x")
    assert again.calls == 2


def test_a_persistent_server_error_gives_up_with_the_last_reason():
    http, _ = _http({"https://a.test/": (500, "boom")}, retries=1, backoff=0.1)
    with pytest.raises(FetchError, match="HTTP 500"):
        http.get_json("https://a.test/x")
    assert http.calls == 2


def test_a_network_failure_is_retried_and_reported_with_the_host():
    def broken(url, headers, timeout):
        raise OSError("timed out")

    http = Http(broken, retries=2, backoff=0.0, sleep=lambda s: None)
    with pytest.raises(FetchError, match=r"a\.test could not be reached.*timed out"):
        http.get_json("https://a.test/x")


def test_an_answer_that_is_not_json_is_refused():
    http, _ = _http({"https://a.test/": (200, "<html>maintenance</html>")})
    with pytest.raises(FetchError, match="not JSON"):
        http.get_json("https://a.test/x")


def test_calls_to_one_host_are_spaced_and_other_hosts_are_not_delayed():
    http, sleeps = _http({"https://a.test/": {}, "https://b.test/": {}}, pause=0.5)
    http.get_json("https://a.test/1")
    http.get_json("https://a.test/2")
    http.get_json("https://b.test/1")
    assert sleeps == [0.5], "only the second call to the same host waited"


# ------------------------------------------------------------------------------------------------------------------ the store
def _frame():
    return pd.DataFrame({"date": ["2026-10-09"], "price": [1.5]})


def test_a_saved_frame_is_listed_with_its_record_and_read_back_by_name(tmp_path):
    store = DownloadStore(tmp_path)
    rec = store.save(Download("options", "SPY", "chain", frame=_frame(), meta={"source": "Cboe", "as_of": "2026-10-09", "notes": ["a"]}))
    assert rec["name"] == "SPY" and rec["item"] == "chain" and rec["rows"] == 1 and rec["file"] == "SPY.csv" and rec["fetched_at"].endswith("Z")
    assert [r["name"] for r in store.list("options")] == ["SPY"] and store.meta("options", "SPY")["source"] == "Cboe"
    assert store.files("options") == {"SPY": tmp_path / "data" / "user" / "downloads" / "options" / "SPY.csv"} and store.files("options", item="curve") == {}
    assert pd.read_csv(store.files("options")["SPY"]).equals(_frame())
    assert store.list("futures") == [] and store.list("crypto") == []


def test_a_document_replaces_a_table_of_the_same_name_and_is_read_back(tmp_path):
    store = DownloadStore(tmp_path)
    store.save(Download("crypto", "venues_BTC", "venues", frame=_frame()))
    store.save(Download("crypto", "venues_BTC", "venues", payload={"venues": {"kraken": {"bid": 1.0}}}, meta={"source": "x"}))
    assert not (store.folder("crypto") / "venues_BTC.csv").exists() and store.read_json("crypto", "venues_BTC") == {"venues": {"kraken": {"bid": 1.0}}}
    assert store.files("crypto") == {}, "files() lists tables only"


def test_listing_is_newest_first_and_skips_damaged_records_and_orphans(tmp_path):
    store = DownloadStore(tmp_path)
    store.save(Download("futures", "CL", "curve", frame=_frame(), meta={"fetched_at": "2026-10-01T10:00:00Z"}))
    store.save(Download("futures", "GC", "curve", frame=_frame(), meta={"fetched_at": "2026-10-05T10:00:00Z"}))
    folder = store.folder("futures")
    (folder / "BAD.meta.json").write_text("{not json")
    (folder / "ORPHAN.meta.json").write_text(json.dumps({"name": "ORPHAN", "file": "ORPHAN.csv"}))
    (folder / "bad name.meta.json").write_text("{}")
    assert [r["name"] for r in store.list("futures")] == ["GC", "CL"]


def test_a_failed_write_leaves_the_previous_file_untouched(tmp_path, monkeypatch):
    store = DownloadStore(tmp_path)
    store.save(Download("options", "SPY", "chain", frame=_frame(), meta={"source": "old"}))
    good = (store.folder("options") / "SPY.csv").read_text()

    def explode(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("src.assets.free_data.store.os.replace", explode)
    with pytest.raises(OSError):
        store.save(Download("options", "SPY", "chain", frame=pd.DataFrame({"x": [9]}), meta={"source": "new"}))
    monkeypatch.undo()
    assert (store.folder("options") / "SPY.csv").read_text() == good and store.meta("options", "SPY")["source"] == "old"
    assert not list(store.folder("options").glob("*.part")), "no temporary file is left behind"


def test_names_and_kinds_are_validated_and_delete_removes_everything_of_a_name(tmp_path):
    store = DownloadStore(tmp_path)
    for bad in ("../x", "a/b", "", "x" * 41, ".hidden", "a.meta"):
        with pytest.raises(ValueError):
            store.path("options", bad)
    with pytest.raises(ValueError):
        store.folder("prices")
    store.save(Download("options", "BRK.B", "chain", frame=_frame()))
    assert store.delete("options", "BRK.B") is True and store.list("options") == [] and store.delete("options", "BRK.B") is False
    with pytest.raises(ValueError):
        store.save(Download("options", "SPY", "chain"))


def test_every_source_is_described_with_what_it_provides_and_its_limits():
    assert {"cboe", "yahoo", "deribit", "coinbase", "kraken", "bitstamp", "okx"} <= set(SOURCES)
    assert all(s["label"] and s["url"].startswith("https://") and s["provides"] and s["delay"] and s["kinds"] for s in SOURCES.values())
    assert [s["id"] for s in describe(["okx", "nope", "cboe"])] == ["okx", "cboe"]
