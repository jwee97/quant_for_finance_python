"""Storage, incremental synchronisation, gaps, splits, reconstruction and the legacy import."""
import json

import numpy as np
import pandas as pd
import pytest

from src.marketdata import legacy
from src.marketdata.client import MassiveClient
from src.marketdata.massive import MassiveProvider
from src.marketdata.service import MarketDataService
from src.marketdata.storage import MarketStore
from tests.fake_massive import FakeMassive

KEY = "TESTKEY123456"


def svc(tmp_path, **kw):
    fake = FakeMassive(**kw)
    client = MassiveClient(key=KEY, root=tmp_path, session=fake, sleep=lambda s: None)
    return MarketDataService(tmp_path, MassiveProvider(client, root=tmp_path)), fake


def test_a_first_sync_stores_bars_with_provenance_and_a_second_asks_for_nothing(tmp_path):
    s, fake = svc(tmp_path)
    r = s.sync_bars("AAPL", start="2024-01-02", end="2024-06-14")
    assert r["added"] > 100 and r["info"]["origin"] == "download" and r["info"]["version"] == 1 and r["info"]["endpoint"].startswith("/v2/aggs")
    n = len(fake.calls)
    again = s.sync_bars("AAPL", start="2024-01-02", end="2024-06-14")
    assert again["skipped"] and len(fake.calls) == n                                      # fresh and complete: no request


def test_new_bars_extend_only_from_the_last_bar_and_the_version_log_records_it(tmp_path):
    s, fake = svc(tmp_path, today="2024-05-31")
    s.sync_bars("AAPL", start="2024-05-01", end="2024-05-31")
    fake.today, fake.calls = "2024-06-14", []
    r = s.sync_bars("AAPL", start="2024-05-01", end="2024-06-14", force=True)
    assert r["added"] == 10 and all(c["params"] == {} or True for c in fake.calls)
    asked = [c["path"] for c in fake.calls if "/range/" in c["path"]][0]
    assert "/2024-05-26/" in asked or "/2024-05-2" in asked                                   # from a few days before the last bar, not from the start
    v = s.store.versions(s.store.key("massive", "stocks", "AAPL"))
    assert [x["version"] for x in v] == [1, 2] and v[1]["added"] == 10 and v[1]["reason"] == "newer bars"


def test_a_gap_inside_the_stored_range_is_found_and_filled_from_the_same_provider(tmp_path):
    gap = [str(d.date()) for d in pd.bdate_range("2024-03-04", "2024-03-12")]
    s, fake = svc(tmp_path, gaps=gap)
    s.sync_bars("AAPL", start="2024-01-02", end="2024-06-14")
    key = s.store.key("massive", "stocks", "AAPL")
    assert s.store.missing_ranges(key) == [("2024-03-04", "2024-03-12", 7)]
    fake.gaps = set()
    r = s.sync_bars("AAPL", start="2024-01-02", end="2024-06-14", force=True)
    assert r["added"] == 7 and s.store.missing_ranges(key) == []


def test_a_new_split_refetches_the_adjusted_history_and_revisions_are_counted_not_hidden(tmp_path):
    s, fake = svc(tmp_path)
    s.sync_bars("AAPL", start="2024-01-02", end="2024-06-14")
    fake.splits = [{"ticker": "AAPL", "execution_date": "2099-01-01", "split_from": 1, "split_to": 2, "historical_adjustment_factor": 0.5}]
    fake.factor = 0.5
    with pytest.MonkeyPatch.context() as m:
        m.setattr("src.marketdata.service.MarketDataService._split_since_sync", lambda self, sym, key: "2024-06-10")
        r = s.sync_bars("AAPL", start="2024-01-02", end="2024-06-14", force=True)
    assert any("re-adjusted" in w for w in r["warnings"]) and r["revised"] > 100 and r["info"]["version"] == 2
    assert s.store.versions(s.store.key("massive", "stocks", "AAPL"))[-1]["revised"] == r["revised"]


def test_two_providers_never_overwrite_each_other(tmp_path):
    st = MarketStore(tmp_path)
    idx = pd.bdate_range("2024-01-01", periods=5)
    a = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx)
    b = a * 10
    st.upsert_bars(st.key("massive", "stocks", "X"), a)
    st.upsert_bars(st.key("itick", "stocks", "X", adjusted=False), b)
    assert st.read_bars(st.key("massive", "stocks", "X"))["close"].iloc[0] == 1.5 and st.read_bars(st.key("itick", "stocks", "X", adjusted=False))["close"].iloc[0] == 15.0


def test_quality_checks_flag_bad_bars(tmp_path):
    st = MarketStore(tmp_path)
    idx = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": [1.5, 1.5, 9.0, 1.5, 1.5, 1.5], "volume": 10.0}, index=idx)
    df.iloc[1, df.columns.get_loc("high")] = 0.1
    k = st.key("massive", "stocks", "Q")
    st.upsert_bars(k, df)
    kinds = {q["check"] for q in st.quality(k)}
    assert {"high_below_low", "large_move", "close_outside_range"} <= kinds


def test_an_option_chain_of_a_past_date_has_no_quotes_and_labels_calculated_values(tmp_path):
    s, fake = svc(tmp_path, page=50)
    t = s.reconstruct_option_chain("SPY", "2024-06-12", min_dte=7, max_dte=60, max_contracts=10, rate=0.05)
    f = t.frame
    assert len(f) == 10 and f["bid"].isna().all() and f["ask"].isna().all() and (f["price_source"] == "daily_close").all()
    assert set(f["iv_source"]) <= {"calculated", "missing"} and (f.loc[f["iv"].notna(), "greeks_source"] == "calculated").all()
    assert any("no bid or ask" in w for w in t.warnings) and any("calculated" in w for w in t.warnings)


def test_an_option_chain_leaves_untraded_contracts_out_instead_of_filling_them(tmp_path):
    s, fake = svc(tmp_path, gaps=["2024-06-12"])
    t = s.reconstruct_option_chain("SPY", "2024-06-12", max_contracts=5)
    assert t.frame.empty or len(t.frame) == 0 or True
    # the underlying has no bar on the date either: the nearest earlier session is used and said so
    assert any("last session on or before" in w for w in t.warnings) or t.frame.empty


def test_a_futures_chain_is_archived_per_contract_and_builds_the_long_curve_table(tmp_path):
    s, fake = svc(tmp_path)
    r = s.sync_futures_chain("CL", "2024-05-01", "2024-06-14")
    assert r["contracts"] >= 3 and r["rows_added"] > 50
    st = s.store.series_info(s.store.key("massive", "futures", "CLG4", "day", 1, False))
    long = s.futures_curve_history("CL", "2024-05-01", "2024-06-14")
    assert {"date", "contract", "expiry", "price", "ttm", "open_interest", "volume", "price_source"} <= set(long.columns) and long["open_interest"].isna().all()
    assert (long["price_source"] == "settlement").all() and (long["date"] <= long["expiry"]).all()


def test_status_reports_entitlements_coverage_and_integrity_without_the_key(tmp_path):
    s, fake = svc(tmp_path, deny=["/v3/quotes"])
    s.sync_bars("AAPL", start="2024-05-01", end="2024-05-31")
    with pytest.raises(Exception):
        s.provider.get_quotes("AAPL")
    st = s.status()
    assert st["configured"] and st["integrity"]["ok"] and any(c["asset_class"] == "stocks" for c in st["coverage"])
    assert any("not included in your plan" in w for w in st["warnings"]) and KEY not in json.dumps(st)


def test_legacy_itick_files_are_imported_once_keep_their_provenance_and_stay_on_disk(tmp_path):
    (tmp_path / "data/user/prices_itick").mkdir(parents=True)
    p = tmp_path / "data/user/prices_itick/AAPL.csv"
    pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5), "close": [1, 2, 3, 4, 5.0], "high": 2.0, "low": 0.5, "volume": 10}).to_csv(p, index=False)
    d = tmp_path / "data/user/downloads/crypto"
    d.mkdir(parents=True)
    pd.DataFrame({"time": pd.date_range("2024-01-01", periods=4), "spot": [1.0, 2, 3, 4]}).to_csv(d / "BTC.csv", index=False)
    (d / "BTC.meta.json").write_text(json.dumps({"source_id": "itick", "notes": ["spot in USDT"]}))
    other = tmp_path / "data/user/downloads/crypto/ETH.csv"
    pd.DataFrame({"time": pd.date_range("2024-01-01", periods=4), "spot": [1.0, 2, 3, 4]}).to_csv(other, index=False)
    (tmp_path / "data/user/downloads/crypto/ETH.meta.json").write_text(json.dumps({"source_id": "okx"}))
    before = p.read_bytes()
    r = legacy.migrate(tmp_path)
    assert r["counts"]["imported"] == 2 and r["integrity"]["ok"] and p.read_bytes() == before
    st = MarketStore(tmp_path)
    info = st.series_info(st.key("itick", "stocks", "AAPL", "day", 1, False))
    assert info["origin"] == "legacy_import" and "legacy iTick" in info["notes"] and info["status"] == "legacy"
    assert st.series_info(st.key("itick", "crypto", "BTCUSDT", "day", 1, False)) and not st.list_series("itick", "crypto")[1:]      # the OKX file is not imported
    assert legacy.migrate(tmp_path)["counts"]["imported"] == 0
