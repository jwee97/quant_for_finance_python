"""Data health: coverage, integrity and readiness with answers worked out by hand, the provider capability registry's evidence rules, and the API that serves them."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.data import health as dh
from src.marketdata import capability_registry as cr
from src.utils.config import load_config
from src.webapp.api import App, ApiError
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch, _wait


def frame(n=300, start="2023-01-02", seed=1, drop=(), freq="B"):
    idx = pd.date_range(start, periods=n, freq=freq)
    c = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.005, n)))
    df = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1000.0}, index=idx)
    return df.drop(df.index[list(drop)]) if drop else df


# ------------------------------------------------------------------------------------------------------ coverage
def test_missing_runs_are_listed_against_the_calendar_and_a_weekend_is_not_a_gap():
    df = frame(60, drop=[10, 11, 12, 30])
    runs = dh.missing_runs(df.index, "B")
    assert [r["periods"] for r in runs] == [3, 1] and runs[0]["start"] == str(pd.bdate_range("2023-01-02", periods=60)[10].date())
    cov = dh.coverage({"X": df})[0]
    assert cov["observations"] == 56 and cov["expected"] == 60 and cov["missing_periods"] == 4 and cov["completeness"] == pytest.approx(56 / 60)


def test_a_crypto_series_is_held_to_every_day_and_the_missing_weekend_is_a_gap_while_an_equity_one_is_not():
    idx = pd.bdate_range("2024-01-01", periods=30)
    df = pd.DataFrame({"close": 1.0 + np.arange(30) * 0.01}, index=idx)
    eq = dh.coverage({"E": df}, "equity")[0]
    cy = dh.coverage({"C": df}, "crypto")[0]
    assert eq["missing_periods"] == 0 and cy["missing_periods"] > 0 and cy["calendar"] == "every day"


def test_a_panels_own_calendar_decides_what_is_expected_so_a_shared_holiday_is_not_a_gap():
    panel = pd.bdate_range("2024-01-01", periods=30).delete([5])                    # the panel was closed that day for everyone
    df = pd.DataFrame({"close": 1.0}, index=panel)
    assert dh.coverage({"X": df}, expected_index=panel)[0]["missing_periods"] == 0
    assert dh.coverage({"X": df})[0]["missing_periods"] == 1


# ------------------------------------------------------------------------------------------------------ integrity
def checks(findings):
    return {f.check: f for f in findings}


def test_duplicates_invalid_ohlc_non_positive_and_stale_prices_are_each_found_and_nothing_is_modified():
    df = frame(80)
    df.iloc[10, df.columns.get_loc("low")] = df.iloc[10]["high"] + 1                  # high below low
    df.iloc[20, df.columns.get_loc("close")] = -1.0                                   # non-positive
    df.iloc[40:47, df.columns.get_loc("close")] = 100.0                               # a stale run
    dup = pd.concat([df, df.iloc[[5]]])
    before = dup.copy(deep=True)
    out = checks(dh.check_series(dup, "X"))
    assert {"duplicate_dates", "unsorted", "high_below_low", "non_positive_price", "stale_prices"} <= set(out)
    assert out["stale_prices"].value >= 7 and out["non_positive_price"].severity == "error"
    pd.testing.assert_frame_equal(dup, before)


def test_a_long_gap_is_an_error_a_short_one_is_information_and_an_old_last_observation_is_stale():
    short = frame(100, drop=[50])
    long = frame(100, drop=list(range(50, 60)))
    assert checks(dh.check_series(short, "S", max_gap_days=5))["missing_periods"].severity == "info"
    assert checks(dh.check_series(long, "L", max_gap_days=5))["long_gap"].severity == "error"
    stale = checks(dh.check_series(short, "S", as_of="2024-06-01", max_age_days=7))
    assert stale["stale_series"].value == (pd.Timestamp("2024-06-01") - short.index[-1]).days


def test_a_quote_is_crossed_locked_non_positive_stale_or_from_the_future():
    now = pd.Timestamp("2024-03-01 12:00:10")
    f = lambda b, a, ts: {x.check for x in dh.check_quote(b, a, ts, now)}
    assert f(100.1, 100.0, "2024-03-01 12:00:09") == {"crossed_market"}
    assert f(100.0, 100.0, "2024-03-01 12:00:09") == {"locked_market"}
    assert f(0.0, 100.0, "2024-03-01 12:00:09") == {"invalid_quote"}
    assert f(99.9, 100.0, "2024-03-01 12:00:00") == {"stale_quote"}
    assert f(99.9, 100.0, "2024-03-01 12:00:30") == {"quote_from_the_future"}
    assert f(99.9, 100.0, "2024-03-01 12:00:09") == set()


def test_observations_after_a_contracts_expiry_are_an_error():
    df = frame(60)
    exp = df.index[40]
    assert checks(dh.check_series(df, "ESZ4", expiry=exp))["observations_after_expiry"].value == 19


def test_unmapped_and_ambiguous_provider_symbols_are_reported():
    mapping = {"AAPL": "AAPL.O", "AAPL.US": "AAPL.O", "MSFT": "MSFT.O"}
    out = checks(dh.check_symbol_mapping(["AAPL", "AAPL.US", "MSFT", "ZZZ", "YYY"], mapping))
    assert out["unmapped_symbol"].value == 2 and out["ambiguous_symbol"].value == 1


def test_figures_available_before_their_period_ended_or_with_no_date_cannot_be_used_point_in_time():
    out = checks(dh.check_point_in_time(["2023-12-31", "2023-12-31", "2023-12-31"], ["2024-02-15", "2023-12-20", None]))
    assert out["available_before_period_end"].value == 1 and out["no_availability_date"].value == 1
    assert dh.check_point_in_time(["2023-12-31"], ["2024-02-15"]) == []


def test_a_recorded_split_must_show_in_the_close_and_a_jump_without_a_record_is_flagged():
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series([100, 101, 102, 51, 51.5, 52, 52.5, 53, 53.5, 54], index=idx, dtype=float)
    ok = dh.check_corporate_actions(close, None, pd.DataFrame({"date": [idx[3]], "ratio": [2.0]}))
    assert ok == []
    unexplained = checks(dh.check_corporate_actions(close, None, None))
    assert "unexplained_discontinuity" in unexplained and unexplained["unexplained_discontinuity"].date == str(idx[3].date())
    wrong = checks(dh.check_corporate_actions(close, None, pd.DataFrame({"date": [idx[5]], "ratio": [2.0]})))
    assert "split_not_in_prices" in wrong


# ------------------------------------------------------------------------------------------------------ readiness and versions
def test_readiness_is_blocked_with_reasons_ready_with_caveats_or_ready():
    good = frame(400)
    short = frame(100)
    gappy = frame(400, drop=list(range(100, 112)))
    warn = frame(400)
    warn.iloc[50, warn.columns.get_loc("close")] = warn.iloc[49]["close"] * 1.8        # a 80% move: a warning
    rep = dh.report("t", {"good": good, "short": short, "gappy": gappy, "warn": warn})
    by = {r["series"]: r for r in rep.readiness}
    assert by["good"]["status"] == "ready"
    assert by["short"]["status"] == "blocked" and "100 observations" in by["short"]["blocking"][0]
    assert by["gappy"]["status"] == "blocked" and any("gap of 12" in b for b in by["gappy"]["blocking"])
    assert by["warn"]["status"] == "ready_with_caveats" and any("large_move" in c for c in by["warn"]["caveats"])
    assert rep.counts() == {"ready": 1, "ready_with_caveats": 1, "blocked": 2}


def test_the_fingerprint_changes_when_one_value_changes_and_not_otherwise():
    a, b = frame(50), frame(50)
    assert dh.fingerprint({"X": a}) == dh.fingerprint({"X": b})
    b.iloc[10, b.columns.get_loc("close")] += 0.01
    assert dh.fingerprint({"X": a}) != dh.fingerprint({"X": b})


def test_a_correction_keeps_the_raw_value_needs_a_reason_and_is_applied_to_a_copy_only(tmp_path):
    log = dh.CorrectionLog(tmp_path / "c.jsonl")
    df = frame(20)
    d = df.index[5]
    raw = float(df.loc[d, "close"])
    with pytest.raises(ValueError, match="reason"):
        log.record("X", d, "close", raw, 99.0, "  ")
    log.record("X", d, "close", raw, 99.0, "bad print confirmed against the exchange file")
    fixed = log.apply(df, "X")
    assert fixed.loc[d, "close"] == 99.0 and df.loc[d, "close"] == raw
    e = log.entries("X")[0]
    assert e["raw"] == raw and e["corrected"] == 99.0 and e["reason"].startswith("bad print")
    assert log.entries("Y") == []


# ------------------------------------------------------------------------------------------------------ provider capability registry
def test_the_registry_credits_only_what_the_evidence_supports(tmp_path):
    root = tmp_path
    (root / "docs" / "generated").mkdir(parents=True)
    (root / "docs" / "generated" / "exchange_capabilities.json").write_text(json.dumps({"rows": [
        {"venue": "OKX", "feature": "funding_history", "status": "verified", "tested_at": "2026-11-25T14:00:00Z", "adapter": "a"},
        {"venue": "BYBIT", "feature": "open_interest", "status": "blocked", "tested_at": "2026-11-25T14:00:00Z", "http_status": 403},
        {"venue": "COINBASE", "feature": "derivatives", "status": "unsupported", "detail": "spot only"},
        {"venue": "KRAKEN", "feature": "futures_tickers", "status": "not_implemented"}]}))
    (root / "tests").mkdir()
    (root / "tests" / "test_x.py").write_text("PATH = '/v2/aggs/ticker'\n")
    ent = root / "data" / "user" / "massive"
    ent.mkdir(parents=True)
    (ent / "entitlements.json").write_text(json.dumps({"stocks.quotes": {"status": "not_entitled", "detail": "plan", "checked_at": "2026-11-25T00:00:00Z"}, "crypto.tickers": {"status": "available", "checked_at": "2026-11-25T00:00:00Z"}}))
    now = datetime(2026, 11, 26, tzinfo=timezone.utc)
    rows = {(r["provider"], r["dataset"]): r for r in cr.build(root, now)["rows"]}
    assert rows[("okx", "funding_history")]["status"] == "verified_live" and rows[("okx", "funding_history")]["age_days"] < 2 and not rows[("okx", "funding_history")]["stale"]
    assert rows[("bybit", "open_interest")]["status"] == "not_tested" and "403" in rows[("bybit", "open_interest")]["evidence"]
    assert rows[("coinbase", "derivatives")]["status"] == "unsupported" and rows[("kraken", "futures_tickers")]["status"] == "unsupported"
    assert rows[("massive", "stocks.bars")]["status"] == "mock_tested", "its path appears in a test file"
    assert rows[("massive", "stocks.quotes")]["status"] == "subscription_blocked"
    assert rows[("massive", "crypto.tickers")]["status"] == "verified_live"
    assert rows[("massive", "options.chain_snapshot")]["status"] == "not_tested", "no test mentions it and nothing was probed"
    assert all(r["status"] in cr.STATUSES for r in rows.values())


def test_a_live_verification_older_than_thirty_days_is_marked_stale():
    root = None
    now = datetime(2027, 3, 1, tzinfo=timezone.utc)
    rows = cr.build(root, now)["rows"]
    live = [r for r in rows if r["status"] == "verified_live" and r["provider"] in ("okx", "deribit")]
    assert live and all(r["stale"] for r in live)


def test_with_no_probe_answers_and_no_key_nothing_of_massive_is_called_verified_and_the_missing_key_is_stated(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    rows = [r for r in cr.build(tmp_path)["rows"] if r["provider"] == "massive"]
    assert rows and {r["status"] for r in rows} == {"not_tested"} and all(r["credential"] == "MASSIVE_API_KEY" and r["credential_present"] is False for r in rows)


# ------------------------------------------------------------------------------------------------------ the API
@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


def test_the_health_routes_report_coverage_findings_readiness_sources_and_export(app):
    for p in ("/api/health/capabilities", "/api/health/datasets", "/api/health/report", "/api/health/missing", "/api/health/source", "/api/health/readiness", "/api/health/corrections"):
        assert p in app.extra_get, p
    rep = app.extra_get["/api/health/report"]({"dataset": "platform"})
    assert rep["coverage"] and rep["readiness"] and rep["fingerprint"] and set(rep["counts"]) == {"ready", "ready_with_caveats", "blocked"}
    series = rep["coverage"][0]["series"]
    miss = app.extra_get["/api/health/missing"]({"dataset": "platform", "series": series})
    assert miss["series"] == series and "runs" in miss
    src = app.extra_get["/api/health/source"]({"dataset": "platform", "series": series})
    assert "snapshot" in src["origin"]
    csvx = app.extra_post["/api/health/export"]({"dataset": "platform"})
    assert csvx["text"].splitlines()[1].startswith("series,status,check") and csvx["filename"].endswith(".csv")
    with pytest.raises(ApiError):
        app.extra_get["/api/health/report"]({"dataset": "nonsense"})
    with pytest.raises(ApiError):
        app.extra_get["/api/health/missing"]({"dataset": "platform", "series": "NOPE"})


def test_validate_runs_as_a_job_and_keeps_a_snapshot_that_shows_whether_the_data_changed(app):
    job = app.extra_post["/api/health/validate"]({"dataset": "platform"})
    view = _wait(app, job["job"])
    assert view["status"] == "done", view
    again = _wait(app, app.extra_post["/api/health/retry"]({"dataset": "platform"})["job"])
    assert again["result"]["snapshot"]["changed_since_last"] is False
    snaps = app.extra_get["/api/health/snapshots"]({})["snapshots"]
    assert len(snaps) == 2 and snaps[0]["fingerprint"] == snaps[1]["fingerprint"]


def test_refresh_of_the_platform_snapshot_is_refused_and_a_market_refresh_needs_symbols(app):
    with pytest.raises(ApiError, match="versioned with the repository"):
        app.extra_post["/api/health/refresh"]({"dataset": "platform"})
    with pytest.raises(ApiError, match="symbols"):
        app.extra_post["/api/health/refresh"]({"dataset": "market"})


def test_a_correction_through_the_api_is_logged_with_its_reason_and_refused_without_one(app):
    with pytest.raises(ApiError, match="reason"):
        app.extra_post["/api/health/correction"]({"series": "AAA", "date": "2024-01-02", "field": "close", "raw": 1.0, "corrected": 2.0})
    out = app.extra_post["/api/health/correction"]({"series": "AAA", "date": "2024-01-02", "field": "close", "raw": 1.0, "corrected": 2.0, "reason": "typo in source"})
    assert out["correction"]["raw"] == 1.0
    assert app.extra_get["/api/health/corrections"]({"series": "AAA"})["corrections"][0]["reason"] == "typo in source"


def test_session_readiness_names_an_instrument_the_platform_does_not_hold(app):
    s = app.extra_post["/api/research/session/create"]({"name": "r", "context": {"assets": ["AAA", "ZZZZ"]}})["session"]
    out = app.extra_get["/api/health/readiness"]({"session": s["id"]})
    by = {r["series"]: r for r in out["rows"]}
    assert by["ZZZZ"]["status"] == "blocked" and "holds no data" in by["ZZZZ"]["blocking"][0] and out["overall"] == "blocked"
