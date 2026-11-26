"""Every request gets a correlation id that is also in the log, the log keeps no values from the request, an unexpected error is logged with its traceback and reaches the page without file paths, and the log rotates."""

from __future__ import annotations

import json
import re

import pytest

from src.webapp import obs
from tests.test_server_hardening import _call, app, live  # noqa: F401


def lines(app):
    p = app.root / "data" / "user" / "logs" / "requests.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_a_response_carries_a_request_id_and_a_client_supplied_safe_one_is_kept(live, app):
    s, _, h = _call(live, "GET", "/api/catalog", {"X-Token": live.token})
    assert s == 200 and re.fullmatch(r"[0-9a-f]{12}", h["x-request-id"])
    s, _, h = _call(live, "GET", "/api/catalog", {"X-Token": live.token, "X-Request-Id": "trace-abc-12345"})
    assert h["x-request-id"] == "trace-abc-12345"
    s, _, h = _call(live, "GET", "/api/catalog", {"X-Token": live.token, "X-Request-Id": "has spaces and <markup> \"quotes\""})
    assert re.fullmatch(r"[0-9a-f]{12}", h["x-request-id"]), "an id with unsafe characters is replaced, never echoed"


def test_the_request_log_has_the_id_route_status_duration_and_only_the_names_of_query_fields(live, app):
    s, _, h = _call(live, "GET", "/api/health/missing?dataset=platform&series=SECRETTICKER", {"X-Token": live.token})
    rows = [r for r in lines(app) if r["request_id"] == h["x-request-id"]]
    assert len(rows) == 1
    r = rows[0]
    assert r["route"] == "/api/health/missing" and r["status"] == s and r["method"] == "GET" and r["ms"] >= 0 and r["query_fields"] == ["dataset", "series"]
    assert "SECRETTICKER" not in json.dumps(r) and "X-Token" not in json.dumps(r) and live.token not in json.dumps(r)


def test_a_refused_request_is_logged_with_its_category(live, app):
    s, _, h = _call(live, "GET", "/api/nope", {"X-Token": live.token})
    r = [x for x in lines(app) if x["request_id"] == h["x-request-id"]][0]
    assert s == 404 and r["category"] == "not_found"
    s, _, h = _call(live, "GET", "/api/catalog")
    r = [x for x in lines(app) if x["request_id"] == h["x-request-id"]][0]
    assert r["status"] in (401, 403) and r["category"] in ("unauthorised", "forbidden")


def test_an_unexpected_error_is_logged_with_a_traceback_and_the_page_gets_a_reference_and_no_file_path(live, app):
    def boom(q=None):
        raise FileNotFoundError(f"[Errno 2] No such file: '{app.root}/data/user/secret_folder/x.parquet'")
    app.extra_get["/api/_boom"] = boom
    s, body, h = _call(live, "GET", "/api/_boom", {"X-Token": live.token})
    msg = json.loads(body)
    assert s == 500 and msg["request_id"] == h["x-request-id"] and f"(ref {msg['request_id']})" in msg["error"] and msg["error"].startswith("FileNotFoundError:")
    assert str(app.root) not in msg["error"] and "secret_folder" not in msg["error"]
    errs = [r for r in lines(app) if r.get("kind") == "error" and r["request_id"] == msg["request_id"]]
    assert errs and "Traceback" in errs[0]["traceback"] and errs[0]["error"] == "FileNotFoundError"
    done = [r for r in lines(app) if r.get("kind") == "request" and r["request_id"] == msg["request_id"]][0]
    assert done["category"] == "FileNotFoundError" and done["status"] == 500


def test_scrub_removes_absolute_paths_home_folders_and_long_text():
    assert obs.scrub("cannot read /home/alice/projects/q/data/x.csv now") == "cannot read ... now"
    assert obs.scrub(r"cannot read C:\Users\bob\q\x.csv now") == "cannot read ... now"
    assert obs.scrub("open /srv/app/data/user/a.db failed", roots=["/srv/app"]).count("/srv") == 0
    assert len(obs.scrub("x" * 1000)) == 303
    assert obs.scrub("a plain message, 3/4 done") == "a plain message, 3/4 done"


def test_the_log_rotates_and_keeps_three_generations(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "ROTATE_BYTES", 400)
    log = obs.RequestLog(tmp_path / "logs" / "requests.jsonl")
    for i in range(40):
        log.write(kind="request", request_id=f"id{i:08d}", route="/api/x", status=200)
    names = sorted(p.name for p in (tmp_path / "logs").iterdir())
    assert names == ["requests.jsonl", "requests.jsonl.1", "requests.jsonl.2", "requests.jsonl.3"][:len(names)] and len(names) >= 3 and "requests.jsonl.4" not in names


def test_a_log_that_cannot_be_written_never_fails_a_request(tmp_path):
    blocker = tmp_path / "logs"
    blocker.write_text("a file where the folder should be")
    obs.RequestLog(blocker / "requests.jsonl").write(kind="request", request_id="abcdefgh12", route="/x", status=200)
