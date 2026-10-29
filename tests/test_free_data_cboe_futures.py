"""VIX futures from Cboe Futures Exchange's own files: the list of contracts, one CSV per contract (settlement, volume, open interest), the VIX index for the spot, and an update that fetches only what is
still alive. The network is a table of answers in the shape of the live files."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assets import futures_analytics as fa
from src.assets.free_data import cboe_futures as cb
from src.assets.free_data.http import FetchError, Http
from tests.free_data_helpers import FakeFeeds

TODAY = "2026-03-10"
BASE = "data/us/futures/market_statistics/historical_data/VX/"
HEADER = "Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest\n"
EXPIRIES = {"F26": "2026-01-21", "G26": "2026-02-18", "H26": "2026-03-18", "J26": "2026-04-15"}


def _listing() -> dict:
    """Cboe's ``product/list/VX/``: year -> the contracts of the year, monthly (``M``) and weekly (``W``), each with the path of its file."""
    rows = {"2025": [], "2026": []}
    for code, day in EXPIRIES.items():
        rows["2026"].append({"product_display": f"VX+VXT/{code[0]}6", "expire_date": day, "contract_dt": day, "futures_root": "VX", "duration_type": "M", "path": f"{BASE}VX_{day}.csv"})
    rows["2026"].append({"product_display": "VX05/G6", "expire_date": "2026-02-04", "contract_dt": "2026-02-04", "futures_root": "VX", "duration_type": "W", "path": f"{BASE}VX_2026-02-04.csv"})
    rows["2025"].append({"product_display": "VX+VXT/Z5", "expire_date": "2025-12-17", "contract_dt": "2025-12-17", "futures_root": "VX", "duration_type": "M", "path": f"{BASE}VX_2025-12-17.csv"})
    rows["2025"].append({"product_display": "VXM/Z5", "expire_date": "2025-12-17", "contract_dt": "2025-12-17", "futures_root": "VXM", "duration_type": "M", "path": f"{BASE}VXM_2025-12-17.csv"})
    return rows


def _csv(code_label: str, expiry: str, first: str, level: float, today: str = TODAY, zero_settle: bool = False) -> str:
    """One contract's file: a row per trading day from ``first`` to the expiry (or to ``today`` for a live contract), a settlement that ramps, volume and open interest that ramp too."""
    last = min(pd.Timestamp(expiry), pd.Timestamp(today))
    days = pd.bdate_range(first, last)
    lines = []
    for i, day in enumerate(days):
        settle = level + 0.05 * i
        lines.append(f"{day.date()},{code_label},{settle - 0.2:.4f},{settle + 0.3:.4f},{settle - 0.3:.4f},{settle - 0.1:.4f},{0 if zero_settle else format(settle, '.4f')},0.0100,{100 + i},0,{1000 + 10 * i}")
    return HEADER + "\n".join(lines) + "\n"


def _vix_csv(first: str = "2025-01-01", last: str = TODAY, level: float = 16.0) -> str:
    days = pd.bdate_range(first, last)
    return "DATE,OPEN,HIGH,LOW,CLOSE\n" + "\n".join(f"{d:%m/%d/%Y},{level:.2f},{level + 1:.2f},{level - 1:.2f},{level + 0.01 * i:.2f}" for i, d in enumerate(days)) + "\n"


def _routes(missing: tuple = (), today: str = TODAY) -> dict:
    routes = {cb.LIST_URL: _listing(), cb.VIX_URL: _vix_csv(last=today)}
    for code, day in EXPIRIES.items():
        routes[cb.FILE_URL.format(path=f"{BASE}VX_{day}.csv")] = (404, "gone") if code in missing else _csv(f"{code[0]} (…)", day, "2025-06-02", 15.0 + 0.5 * list(EXPIRIES).index(code), today)
    routes[cb.FILE_URL.format(path=f"{BASE}VX_2025-12-17.csv")] = _csv("Z (Dec 2025)", "2025-12-17", "2025-03-03", 15.0, today)
    return routes


def _http(routes):
    feeds = FakeFeeds(routes)
    return Http(feeds, sleep=lambda s: None, pause=0.0, retries=0), feeds


# ------------------------------------------------------------------------------------------------------------------ the list and the files
def test_only_monthly_vx_contracts_with_a_file_this_app_knows_are_taken_and_named_by_the_month_they_expire_in():
    got = cb.contracts(_listing(), since=2013)
    assert [c["contract"] for c in got] == ["Z25", "F26", "G26", "H26", "J26"], "oldest first, weeklies and the mini contract left out"
    assert got[-1]["expiry"] == pd.Timestamp("2026-04-15") and got[-1]["path"] == f"{BASE}VX_2026-04-15.csv"
    assert [c["contract"] for c in cb.contracts(_listing(), since=2026)] == ["F26", "G26", "H26", "J26"]
    tampered = {"2026": [{"futures_root": "VX", "duration_type": "M", "expire_date": "2026-01-21", "path": "https://elsewhere.example/VX_2026-01-21.csv"},
                         {"futures_root": "VX", "duration_type": "M", "expire_date": "not a date", "path": f"{BASE}VX_2026-02-18.csv"}, "not a row"]}
    assert cb.contracts(tampered) == [], "a path outside Cboe's folder, a date nobody can read and a row of another shape are skipped, never fetched"
    with pytest.raises(FetchError, match="not in the shape"):
        cb.contracts([1, 2, 3])


def test_a_contract_file_gives_the_settlement_or_the_close_before_it_existed_and_nothing_after_the_expiry():
    text = (HEADER + "2013-01-02,F (Jan 2013),16.8,16.8,15.5,15.6,0,0,97535,3632,120663\n2013-01-03,F (Jan 2013),15.75,16.0,15.3,15.9,15.85,0.1,59683,3630,118689\n"
            "2013-01-04,F (Jan 2013),0,0,0,0,0,0,0,0,0\n2013-01-15,F (Jan 2013),14.25,14.4,13.9,14.2,14.18,0.1,43022,74,57219\n2013-01-16,F (Jan 2013),0,14.25,14.4,0,14.3,0,0,0,50051\n"
            "2013-01-17,F (Jan 2013),1,1,1,1,1,1,1,1,1\n")
    got = cb.parse_contract(text, "2013-01-16")
    assert list(got.columns) == ["date", "price", "open_interest", "volume"] and list(got["date"].dt.strftime("%Y-%m-%d")) == ["2013-01-02", "2013-01-03", "2013-01-15", "2013-01-16"]
    assert list(got["price"]) == [15.6, 15.85, 14.18, 14.3], "the close where the settlement is 0, the settlement otherwise; a row with no price at all is dropped, and so is the day after the expiry"
    assert list(got["open_interest"]) == [120663, 118689, 57219, 50051] and list(got["volume"]) == [97535, 59683, 43022, 0]
    with pytest.raises(FetchError, match="lacks the columns"):
        cb.parse_contract("a,b\n1,2\n", "2013-01-16")
    assert cb.parse_contract(HEADER + "2013-01-02,F,1,1,1,1,1,0,1,0,1\n2013-01-02,F,1,1,1,1,2,0,1,0,1\n", "2013-12-31")["price"].tolist() == [2.0], "one row per day: the later one wins"


def test_the_vix_history_gives_the_index_close_by_day():
    vix = cb.parse_vix("DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,17.24,17.24,17.24,17.24\n10/28/2026,17.47,17.77,16.21,16.91\nbad,1,1,1,1\n")
    assert list(vix.index) == [pd.Timestamp("1990-01-02"), pd.Timestamp("2026-10-28")] and list(vix.values) == [17.24, 16.91]
    with pytest.raises(FetchError, match="not in the shape"):
        cb.parse_vix("x,y\n1,2\n")


# ------------------------------------------------------------------------------------------------------------------ the curve
def test_the_curve_is_every_contract_since_the_start_with_settlement_open_interest_volume_and_the_vix_as_spot(tmp_path):
    http, feeds = _http(_routes())
    messages = []
    d = cb.fetch_vx_curve(http, since=2025, progress=messages.append)
    df = d.frame
    assert d.kind == "futures" and d.name == "VX" and d.item == "curve" and d.meta["source_id"] == "cboe_cfe" and d.meta["contracts"] == 5 and d.meta["has_open_interest"] is True
    assert d.meta["multiplier"] == 1000.0 and d.meta["tick_size"] == 0.05 and d.meta["asset_class"] == "volatility" and d.meta["root_name"].startswith("Cboe Volatility Index") and d.meta["exchange"] == "CFE"
    assert list(df.columns) == ["date", "contract", "expiry", "price", "open_interest", "volume", "spot"] and set(df["contract"]) == {"Z25", "F26", "G26", "H26", "J26"}
    assert df["date"].max() == pd.Timestamp(TODAY) and (df["date"] <= df["expiry"]).all() and df["spot"].notna().all() and (df["open_interest"] > 0).all()
    last = df[df["date"] == df["date"].max()].sort_values("expiry")
    assert list(last["contract"]) == ["H26", "J26"], "on the last day the live strip is the contracts that have not expired"
    assert last["expiry"].iloc[0] == pd.Timestamp("2026-03-18"), "the expiry comes from Cboe's list, not from a rule"
    assert sum(1 for u in feeds.calls if u == cb.LIST_URL) == 1 and sum(1 for u in feeds.calls if u.endswith(".csv") and "VX_" in u) == 5 and sum(1 for u in feeds.calls if u == cb.VIX_URL) == 1
    assert messages[0] == "VX: Z25 (1 of 5)" and len(messages) == 5
    assert any("settlement, volume and open interest" in n for n in d.meta["notes"]) and any("Weekly VX contracts are left out" in n for n in d.meta["notes"])
    path = tmp_path / "VX.csv"
    df.to_csv(path, index=False)
    curve = fa.load_curve_file(path, "VX")                                                           # the file the Futures workspace reads
    snap = fa.curve_snapshot(curve)
    assert [r["contract"] for r in snap["rows"]] == ["H26", "J26"] and snap["rows"][0]["open_interest"] > 0 and curve.spot is not None


def test_an_update_fetches_only_the_contracts_that_were_alive_at_the_last_download_and_the_new_ones():
    first, _ = _http(_routes(today="2026-02-10"))
    early = cb.fetch_vx_curve(first, since=2025).frame
    assert set(early["contract"]) >= {"Z25", "F26", "G26"}
    http, feeds = _http(_routes(today=TODAY))
    d = cb.fetch_vx_curve(http, since=2025, earlier=early)
    fetched = [u.rsplit("VX_", 1)[1] for u in feeds.calls if "VX_" in u]
    assert sorted(fetched) == ["2026-02-18.csv", "2026-03-18.csv", "2026-04-15.csv"], "Z25 and F26 had expired before the last download: they are final and are not asked for again"
    assert set(d.frame["contract"]) == {"Z25", "F26", "G26", "H26", "J26"} and d.frame["date"].max() == pd.Timestamp(TODAY)
    again = d.frame.drop(columns=[]).copy()
    pd.testing.assert_frame_equal(d.frame[d.frame["contract"].isin(["Z25", "F26"])].reset_index(drop=True), early[early["contract"].isin(["Z25", "F26"])].sort_values(["date", "expiry"]).reset_index(drop=True), check_dtype=False)
    assert len(again) == len(d.frame) and d.frame.duplicated(["date", "contract"]).sum() == 0


def test_a_file_that_cannot_be_read_is_named_and_the_rest_of_the_curve_is_kept():
    d = cb.fetch_vx_curve(_http(_routes(missing=("G26",)))[0], since=2025)
    assert set(d.frame["contract"]) == {"Z25", "F26", "H26", "J26"} and any("G26" in n and "could not be read" in n for n in d.meta["notes"])
    routes = _routes()
    for code, day in EXPIRIES.items():
        routes[cb.FILE_URL.format(path=f"{BASE}VX_{day}.csv")] = (404, "gone")
    routes[cb.FILE_URL.format(path=f"{BASE}VX_2025-12-17.csv")] = (404, "gone")
    with pytest.raises(FetchError, match="no VIX futures file"):
        cb.fetch_vx_curve(_http(routes)[0], since=2025)


def test_a_refusal_ends_the_asking_and_the_curve_says_it_is_incomplete_and_no_spot_is_no_reason_to_fail():
    routes = _routes()
    routes[cb.FILE_URL.format(path=f"{BASE}VX_2026-03-18.csv")] = (403, "AccessDenied")
    d = cb.fetch_vx_curve(_http(routes)[0], since=2025)
    assert d.meta["incomplete"] is True and any("stopped answering" in n for n in d.meta["notes"]) and "H26" not in set(d.frame["contract"])
    routes = _routes()
    del routes[cb.VIX_URL]
    d2 = cb.fetch_vx_curve(_http(routes)[0], since=2025)
    assert "spot" not in d2.frame.columns and any("VIX index could not be read" in n for n in d2.meta["notes"]) and d2.meta["contracts"] == 5
    only = {cb.LIST_URL: _listing()}
    only[cb.FILE_URL.format(path=f"{BASE}VX_2026-04-15.csv")] = (403, "AccessDenied")
    with pytest.raises(FetchError, match="refused"):
        cb.fetch_vx_curve(_http(only)[0], since=2026)


def test_the_list_changing_shape_stops_the_download_before_anything_is_fetched():
    http, feeds = _http({cb.LIST_URL: ["not", "a", "dict"]})
    with pytest.raises(FetchError, match="not in the shape"):
        cb.fetch_vx_curve(http)
    assert feeds.calls == [cb.LIST_URL]
