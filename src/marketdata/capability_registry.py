"""One honest table of what each data provider can do, and how we know.

Every (provider, dataset) row carries exactly one status:

``verified_live``         a read-only request to the provider answered with the fields the adapter reads, on the date stored with the row (``verified_at``). Only a live answer sets this.
``verified_recorded``     the adapter parses a real response that was recorded earlier and committed as a fixture (``tests/fixtures``); it shows the parser works on a real shape, not that the provider still answers
``mock_tested``           tests exist, but they feed the adapter responses written by the test itself: they prove the code's logic, not compatibility with the provider
``subscription_blocked``  the provider answered that this account's plan does not include the dataset
``unsupported``           the provider offers no such public endpoint, or the platform has no adapter for it
``not_tested``            nothing in this repository has exercised it against the provider or against any response

The evidence is looked up, not asserted: a Massive endpoint is ``mock_tested`` only if its path appears in a test file; a recorded fixture counts only if the file exists; an exchange row is ``verified_live`` only if the
generated matrix (written by ``scripts/verify_exchanges.py`` from live requests) says so. A provider that needs a key this machine does not hold is reported as such (``credential``) and its rows stay at what the
evidence supports. Nothing here makes a request.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("verified_live", "verified_recorded", "mock_tested", "subscription_blocked", "unsupported", "not_tested")
STALE_AFTER_DAYS = 30

# (provider, dataset, credential env var or "", recorded fixtures (relative to the repository), test-file tokens that show the adapter is exercised by tests, note)
STATIC = [
    ("yahoo", "futures_chart", "", ["tests/fixtures/free_data/yahoo_clz26_chart.json"], ["yahoo_clz26_chart", "yahoo_futures"], "futures daily chart (unofficial public endpoint; may change without notice)"),
    ("deribit", "dated_futures", "", ["tests/fixtures/free_data/deribit_btc_futures.json", "tests/fixtures/exchanges/deribit_dated_futures.json"], ["deribit"], "public book summary"),
    ("sec_edgar", "companyfacts", "", ["tests/fixtures/sec/AAPL.json.gz"], ["fixtures/sec", "sec_edgar", "companyfacts"], "XBRL company facts (needs a User-Agent with contact details)"),
    ("finzdata", "futures_contracts", "FINZDATA_API_KEY", [], ["finzdata"], "contract-level futures with open interest; the free plan shows 30 days"),
    ("edinet", "filings", "EDINET_API_KEY", [], ["edinet"], "Japan filings (taxonomy and connector)"),
    ("opendart", "filings", "OPENDART_API_KEY", [], ["opendart"], "South Korea filings"),
    ("cvm", "filings", "", [], ["cvm"], "Brazil filings"),
]
TEST_GLOBS = ("tests/test_*.py",)


def _root(root) -> Path:
    return Path(root) if root else Path(__file__).resolve().parents[2]


def _test_text(root: Path) -> str:
    chunks = []
    for g in TEST_GLOBS:
        for p in sorted(root.glob(g)):
            try:
                chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
    return "\n".join(chunks)


def _age(stamp: str | None, now: datetime) -> float | None:
    if not stamp:
        return None
    try:
        return (now - datetime.fromisoformat(stamp.replace("Z", "+00:00"))).total_seconds() / 86400.0
    except ValueError:
        return None


def _exchange_rows(root: Path, now: datetime) -> list[dict]:
    path = root / "docs" / "generated" / "exchange_capabilities.json"
    if not path.is_file():
        return []
    try:
        rows = json.loads(path.read_text())["rows"]
    except (OSError, ValueError, KeyError):
        return []
    out = []
    for r in rows:
        st = r.get("status")
        fixture = root / "tests" / "fixtures" / "exchanges" / f"{str(r['venue']).lower()}_{r['feature']}.json"
        age = _age(r.get("tested_at"), now)
        if st == "verified":
            status, why = "verified_live", f"answered a live read-only request on {str(r.get('tested_at'))[:10]}"
        elif st == "unsupported":
            status, why = "unsupported", r.get("detail") or r.get("note") or "no public endpoint"
        elif st == "not_implemented":
            status, why = "unsupported", "the endpoint answers but the platform has no adapter for it"
        elif st == "blocked":
            status, why = "not_tested", "the request was refused from the network it ran on (HTTP " + str(r.get("http_status", "403/451")) + "); nothing is inferred about the endpoint"
        else:
            status, why = "not_tested", r.get("detail") or "the live check failed"
        out.append({"provider": str(r["venue"]).lower(), "dataset": r["feature"], "status": status, "evidence": why, "verified_at": r.get("tested_at") if st == "verified" else None,
                    "age_days": None if age is None else round(age, 1), "stale": bool(st == "verified" and age is not None and age > STALE_AFTER_DAYS), "adapter": r.get("adapter") or "", "recorded_fixture": fixture.is_file(), "credential": "", "credential_present": True,
                    "note": r.get("note") or ""})
    return out


def _massive_rows(root: Path, tests: str) -> list[dict]:
    from .massive import DOCUMENTED

    ent = {}
    p = root / "data" / "user" / "massive" / "entitlements.json"
    try:
        if p.is_file():
            ent = json.loads(p.read_text())
    except (OSError, ValueError):
        ent = {}
    key = bool(os.environ.get("MASSIVE_API_KEY"))
    stubs = [d[2].split("{")[0].rstrip("/") for d in DOCUMENTED]
    out = []
    for ac, ds, path, plan in DOCUMENTED:
        k = f"{ac}.{ds}"
        e = ent.get(k) or {}
        stub = path.split("{")[0].rstrip("/")
        in_tests = stub in tests
        st, why, at = "not_tested", "documented by the provider; no test or request in this repository has used it", None
        if e.get("status") == "available":
            st, why, at = "verified_live", "a probe with this key answered", e.get("checked_at")
        elif e.get("status") == "not_entitled":
            st, why, at = "subscription_blocked", "the provider refused it for this key: " + str(e.get("detail", ""))[:140], e.get("checked_at")
        elif e.get("status") == "no_data":
            st, why, at = "not_tested", "the probe answered with no rows; the dataset is reachable but nothing was read", e.get("checked_at")
        elif e.get("status") == "error":
            st, why, at = "not_tested", "the probe failed: " + str(e.get("detail", ""))[:140], e.get("checked_at")
        elif in_tests:
            shared = stubs.count(stub) > 1
            st, why = "mock_tested", "its path appears in tests that use responses the tests wrote" + ("; the path is shared with another dataset, so this is credit by path, not by dataset" if shared else "") + "; live compatibility is not shown"
        out.append({"provider": "massive", "dataset": k, "status": st, "evidence": why, "verified_at": at if st == "verified_live" else None, "age_days": None, "stale": False, "adapter": "marketdata.massive", "recorded_fixture": False,
                    "credential": "MASSIVE_API_KEY", "credential_present": key, "note": plan, "endpoint": path})
    return out


def _static_rows(root: Path, tests: str) -> list[dict]:
    out = []
    for provider, dataset, env, fixtures, tokens, note in STATIC:
        rec = [f for f in fixtures if (root / f).is_file()]
        in_tests = any(t in tests for t in tokens)
        if rec:
            st, why = "verified_recorded", "parses a real response recorded in " + ", ".join(rec) + " (the date it was recorded is not stored)"
        elif in_tests:
            st, why = "mock_tested", "exercised by tests with responses the tests wrote"
        else:
            st, why = "not_tested", "no test or fixture exercises it"
        out.append({"provider": provider, "dataset": dataset, "status": st, "evidence": why, "verified_at": None, "age_days": None, "stale": False, "adapter": "", "recorded_fixture": bool(rec), "credential": env,
                    "credential_present": (not env) or bool(os.environ.get(env)), "note": note})
    return out


def build(root=None, now: datetime | None = None) -> dict:
    root = _root(root)
    now = now or datetime.now(timezone.utc)
    tests = _test_text(root)
    rows = _exchange_rows(root, now) + _massive_rows(root, tests) + _static_rows(root, tests)
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    return {"generated_at": now.replace(microsecond=0).isoformat(), "rows": rows, "counts": counts, "statuses": list(STATUSES),
            "meaning": {"verified_live": "answered a live read-only request; the date is stored", "verified_recorded": "parses a real response recorded earlier; not a proof the provider still answers", "mock_tested": "tests with invented responses: the code's logic, not provider compatibility",
                        "subscription_blocked": "the provider says this plan does not include it", "unsupported": "no public endpoint, or no adapter on this platform", "not_tested": "nothing has exercised it"},
            "notes": ["Linux test runs do not show Windows compatibility.", "A source file existing does not show a dataset works; only the status above does."]}
