"""Security properties of the web layer, each checked rather than assumed: no secret reaches a response, the static server serves only its allow-list, the verification endpoint can run only the tests it lists, request
bodies are bounded, and untrusted input never reaches a shell or an SQL identifier."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.ml import verification as vf
from src.utils.config import load_config
from src.webapp import server
from src.webapp.api import App, ApiError
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch

ROOT = Path(__file__).resolve().parents[1]
SECRET_ENV = {"MASSIVE_API_KEY": "sk_test_MASSIVE_0123456789", "FINZDATA_API_KEY": "sk_test_FINZ_0123456789", "EDINET_API_KEY": "sk_test_EDINET_0123456789", "OPENDART_API_KEY": "sk_test_DART_0123456789",
              "QUANT_LAB_ACCESS_CODE": "sk_test_ACCESS_0123456789"}
STATUS_ROUTES = ("/api/market/status", "/api/data/status", "/api/health/capabilities", "/api/research/overview", "/api/storage/status", "/api/system/resources", "/api/fund/status", "/api/crypto/overview",
                 "/api/ml/verification/status", "/api/lab/plan", "/api/forecast/plan", "/api/models/plan")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    for k, v in SECRET_ENV.items():
        monkeypatch.setenv(k, v)
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


def test_no_status_or_plan_response_contains_a_provider_key_or_the_access_code(app):
    checked = 0
    for path in STATUS_ROUTES:
        if path not in app.extra_get:
            continue
        try:
            body = json.dumps(app.extra_get[path]({}), default=str)
        except ApiError:
            continue
        checked += 1
        for name, secret in SECRET_ENV.items():
            assert secret not in body, f"{path} returns {name}"
    assert checked >= 6


def test_the_page_scripts_never_embed_a_key_and_the_html_token_is_not_the_access_code():
    for f in (ROOT / "src" / "webapp" / "static").glob("*.js"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        assert not re.search(r"(sk|pk)_[A-Za-z0-9]{16,}", text), f.name
        assert "MASSIVE_API_KEY=" not in text and "process.env" not in text, f.name


def test_the_static_server_serves_only_the_names_it_lists_and_cannot_be_walked_out_of_its_folder():
    for rel in ("../server.py", "../../README.md", "..%2fserver.py", "/etc/passwd", "app.js/../../server.py", "vendor/../../server.py"):
        assert rel not in server.FILES
    for rel in server.FILES:
        assert ".." not in rel and not rel.startswith("/")
    assert all((server.STATIC / rel).exists() or rel.startswith("vendor/") for rel in server.FILES if rel.endswith(".js") and not rel.startswith("vendor/")), "every listed script exists"


def test_a_request_body_has_a_size_limit_and_the_query_has_a_field_limit():
    assert 0 < server.MAX_BODY <= 1_000_000 and 0 < server.MAX_QUERY_FIELDS <= 100


def test_the_verification_endpoint_runs_only_test_files_it_lists_and_refuses_anything_else():
    for rid, (paths, group) in vf.REQUIREMENTS.items():
        for p in paths:
            assert p.startswith("tests/test_") and p.endswith(".py") and ".." not in p and not p.startswith("/"), (rid, p)
            assert (ROOT / p).is_file(), (rid, p)
    for bad in ("../../etc/passwd", "tests/test_x.py; rm -rf /", "P0-01 && echo hi", "$(id)", ""):
        with pytest.raises(vf.VerificationError):
            vf.run_verification(None, [bad], None)


def test_no_module_hands_a_request_value_to_a_shell():
    offenders = []
    for f in (ROOT / "src").rglob("*.py"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"shell\s*=\s*True|os\.system\(|os\.popen\(", text):
            offenders.append(str(f.relative_to(ROOT)))
    assert offenders == [], offenders


def test_the_review_listing_chooses_among_fixed_sql_fragments_and_an_unknown_sort_falls_back_to_the_default():
    from src.fundamentals.importer import review

    assert all(";" not in v and "--" not in v and "'" not in v and "{" not in v for v in review.SORTS.values()), review.SORTS
    assert review.SORTS.get("label; DROP TABLE raw_observations", "page") == "page"


def test_an_unknown_health_dataset_or_series_is_refused_not_looked_up(app):
    with pytest.raises(ApiError):
        app.extra_get["/api/health/report"]({"dataset": "../../etc/passwd"})
    with pytest.raises(ApiError):
        app.extra_get["/api/health/source"]({"dataset": "platform", "series": "'; DROP TABLE x;--"})
