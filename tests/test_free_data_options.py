"""Real option chains from free sources: the parsers recover what a known model put into the payloads, and the file they write is read by the Options workspace's own loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets import options_analytics as oa
from src.assets.free_data import options as fo
from src.assets.free_data.http import FetchError, Http
from src.derivatives.synthetic import load_option_csv
from tests.free_data_helpers import FakeFeeds, cboe_payload, deribit_options_payload

DATE = pd.Timestamp("2026-10-09")


def _http(routes):
    return Http(FakeFeeds(routes), sleep=lambda s: None, pause=0.0)


def _chain_frame(**kw):
    rows, spot, qd = fo.cboe_rows(cboe_payload(**kw))
    return fo.build_chain(rows, spot, qd, kw.get("rate", 0.04))


# ------------------------------------------------------------------------------------------------------------------ Cboe
def test_cboe_contract_names_become_expiry_strike_and_right_and_the_quote_date_is_the_session_not_today():
    payload = cboe_payload("SPY", spot=500.0, quote_date="2026-10-09")
    payload["timestamp"] = "2026-10-11 05:10:43"                                       # a Sunday: the quotes are still Friday's
    rows, spot, qd = fo.cboe_rows(payload)
    assert spot == 500.0 and qd == DATE
    first = rows.iloc[0]
    assert first["expiry"] == DATE + pd.Timedelta(days=5) and first["right"] in "CP" and first["strike"] == 425.0
    assert set(rows["right"]) == {"C", "P"} and rows["expiry"].nunique() == 5
    for bad in ({}, {"data": {}}, {"data": {"options": []}}, {"data": {"options": [{"option": "junk"}], "current_price": 1.0, "last_trade_time": "2026-10-09T10:00:00"}}):
        with pytest.raises(FetchError):
            fo.cboe_rows(bad)


def test_unusable_rows_are_dropped_and_counted():
    p = cboe_payload(expiry_days=(5, 40, 400))
    options = p["data"]["options"]
    options[0]["bid"], options[0]["ask"] = 2.0, 1.0                                    # crossed
    options[1]["ask"] = 0.0                                                            # no offer
    options.append(dict(options[2]))                                                   # a duplicate contract
    options.append({**options[3], "option": "SPY261009C00500000"})                     # expires on the quote date
    rows, spot, qd = fo.cboe_rows(p)
    df, counts = fo.build_chain(rows, spot, qd, 0.04, max_dte=365)
    assert counts["no_offer_or_crossed"] == 2 and counts["duplicates"] == 1 and counts["outside_range"] >= 22 + 1 and counts["raw"] == len(rows)
    assert list(df.columns) == fo.COLUMNS and df["expiry"].max() <= DATE + pd.Timedelta(days=365) and (df["expiry"] > DATE).all()
    assert not df.duplicated(["expiry", "strike", "right"]).any() and (df["bid"] <= df["ask"]).all() and (df["ask"] > 0).all()
    with pytest.raises(FetchError, match="no option with a live offer"):
        fo.build_chain(rows.assign(ask=0.0), spot, qd)
    with pytest.raises(fo.NoLiveOffers, match=f"sent {len(rows):,} options and not one had an offer.*try again in the session"):
        fo.build_chain(rows.assign(ask=0.0), spot, qd)                                  # a chain with no offer at all is a closed market: it says so
    with pytest.raises(FetchError, match="no option with a live offer") as other:
        fo.build_chain(rows.assign(bid=5.0, ask=1.0), spot, qd)                         # every market crossed: something else is wrong, and it is not called a closed market
    assert not isinstance(other.value, fo.NoLiveOffers)


def test_the_dividend_yield_implied_by_parity_recovers_the_one_that_priced_the_chain():
    for div, rate in ((0.013, 0.04), (0.0, 0.04), (0.035, 0.02), (-0.01, 0.05)):
        rows, spot, qd = fo.cboe_rows(cboe_payload(expiry_days=(5, 12, 33, 96, 187), div=div, rate=rate, spread=0.02))
        df, counts = fo.build_chain(rows, spot, qd, rate)
        per_expiry = df.groupby("expiry")["dividend"].first()
        far = per_expiry.iloc[-3:]                                                    # expiries with at least a week to run are estimated, the nearer ones borrow the nearest
        assert np.allclose(far.to_numpy(), div, atol=0.004), (div, rate, far.tolist())
        assert counts["carry"]["estimated"] >= 3 and counts["carry"]["clipped"] == 0 and abs(counts["carry"]["median"] - div) < 0.004
    rows, spot, qd = fo.cboe_rows(cboe_payload(expiry_days=(2, 4)))
    df, counts = fo.build_chain(rows, spot, qd, 0.04)
    assert counts["carry"]["estimated"] == 0 and (df["dividend"] == 0.0).all(), "too near to read a forward from: the yield is zero and the notes say it was assumed"
    assert "assumptions" in fo._chain_notes(df, counts, 0.04, True, "d")[1]


def test_the_downloaded_file_is_read_by_the_workspace_loader_and_reproduces_the_volatility_that_priced_it(tmp_path):
    http = _http({"https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json": cboe_payload("SPY", spot=500.0, vol=0.22, div=0.013, rate=0.04, spread=0.03)})
    d = fo.fetch_cboe_chain(http, "spy")
    assert d.kind == "options" and d.name == "SPY" and d.item == "chain" and d.meta["source_id"] == "cboe" and d.meta["as_of"] == "2026-10-09" and d.meta["american_style"] is True
    assert d.meta["contracts"] == len(d.frame) and d.meta["spot"] == 500.0 and any("put-call parity" in n for n in d.meta["notes"]) and any("exercised early" in n for n in d.meta["notes"])
    path = tmp_path / "SPY.csv"
    d.frame.to_csv(path, index=False)
    chain = oa.prepare_chain(load_option_csv(path, underlying="SPY"))                  # exactly what the Options workspace does with a file
    near = chain[(chain["T"] * 365 > 25) & (chain["k"].abs() < 0.05)]
    assert len(near) >= 8 and np.nanmedian(np.abs(near["iv"] - 0.22)) < 0.01, "implied volatilities from the file match the one that priced the chain"
    assert chain["date"].nunique() == 1 and abs(chain["underlying"].iloc[0] - 500.0) < 1e-9


def test_an_index_is_asked_for_under_its_underscore_name_and_is_not_american_style():
    feeds = FakeFeeds({"https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json": cboe_payload("SPX", spot=5000.0, step=25.0)})
    d = fo.fetch_cboe_chain(Http(feeds, sleep=lambda s: None, pause=0.0), "^spx")
    assert feeds.calls == ["https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"] and d.name == "SPX" and d.meta["american_style"] is False
    assert not any("exercised early" in n for n in d.meta["notes"])


def test_bad_symbols_and_unknown_chains_are_refused_with_a_reason():
    for bad in ("", "a b", "../x", "TOOLONGSYMBOL1"):
        with pytest.raises(ValueError):
            fo.check_symbol(bad)
    with pytest.raises(FetchError, match="Cboe has no option chain for NOPE") as caught:
        fo.fetch_cboe_chain(_http({}), "NOPE")
    assert caught.value.status == 404
    asked = FakeFeeds({})
    never = type("Y", (), {"Ticker": staticmethod(lambda s: (_ for _ in ()).throw(AssertionError("Yahoo was asked for a symbol Cboe says has no options")))})
    with pytest.raises(FetchError, match="Cboe has no option chain for NOPE") as caught:
        fo.fetch_chain(Http(asked, sleep=lambda s: None, pause=0.0), "NOPE", yf=never)
    assert "Yahoo" not in str(caught.value), "a symbol with no listed options is not looked for elsewhere"
    with pytest.raises(FetchError, match="refused the request") as caught:
        fo.fetch_cboe_chain(_http({"https://cdn.cboe.com/": (403, "no")}), "SPY")
    assert caught.value.blocked and "has no option chain" not in str(caught.value), "a refusal is not reported as a missing chain"


# ------------------------------------------------------------------------------------------------------------------ Yahoo fallback
class _FakeTable:
    def __init__(self, frame):
        self.frame = frame

    def __getitem__(self, key):
        return self.frame[key]


class _FakeYahoo:
    """The corner of ``yfinance`` the fallback uses."""

    def __init__(self, spot=100.0, fail_expiry=None):
        self.spot, self.fail_expiry = spot, fail_expiry
        self.Ticker = lambda symbol: self._ticker(symbol)

    def _ticker(self, symbol):
        outer = self
        rows, _, qd = fo.cboe_rows(cboe_payload(symbol, spot=outer.spot, expiry_days=(10, 40, 90)))
        expiries = sorted(rows["expiry"].unique())

        class T:
            options = tuple(str(e.date()) for e in expiries)

            def history(self, period):
                return pd.DataFrame({"Close": [outer.spot - 1.0, outer.spot]}, index=pd.DatetimeIndex([qd - pd.Timedelta(days=1), qd], tz="America/New_York"))

            def option_chain(self, expiry):
                if expiry == outer.fail_expiry:
                    raise RuntimeError("Too Many Requests")
                g = rows[rows["expiry"] == pd.Timestamp(expiry)]

                class Chain:
                    pass

                chain = Chain()
                for attr, right in (("calls", "C"), ("puts", "P")):
                    h = g[g["right"] == right]
                    setattr(chain, attr, pd.DataFrame({"strike": h["strike"].to_numpy(), "bid": h["bid"].to_numpy(), "ask": h["ask"].to_numpy(), "volume": h["volume"].to_numpy(),
                                                       "openInterest": h["open_interest"].to_numpy(), "impliedVolatility": h["iv_vendor"].to_numpy()}))
                return chain

        return T()


def test_yahoo_is_used_when_cboe_cannot_be_reached_and_the_chain_says_so():
    d = fo.fetch_chain(_http({"https://cdn.cboe.com/": (451, "no")}), "AAPL", yf=_FakeYahoo(spot=200.0))
    assert d.meta["source_id"] == "yahoo" and d.meta["as_of"] == "2026-10-09" and d.meta["spot"] == 200.0 and d.frame["expiry"].nunique() == 3
    assert d.meta["notes"][0].startswith("Cboe did not give the chain (") and "does not serve this location" in d.meta["notes"][0]
    ok = fo.fetch_chain(_http({"https://cdn.cboe.com/api/global/delayed_quotes/options/AAPL.json": cboe_payload("AAPL", spot=200.0)}), "AAPL", yf=_FakeYahoo())
    assert ok.meta["source_id"] == "cboe", "Yahoo is only the fallback"


def test_one_failing_expiry_is_reported_and_all_failing_is_an_error_naming_both_sources():
    d = fo.fetch_yahoo_chain("AAPL", yf=_FakeYahoo(fail_expiry="2026-11-18"))
    assert d.frame["expiry"].nunique() == 2 and any("1 expiries could not be read" in n for n in d.meta["notes"])
    broken = type("Y", (), {"Ticker": staticmethod(lambda s: (_ for _ in ()).throw(RuntimeError("Too Many Requests")))})
    with pytest.raises(FetchError, match=r"cboe\.com refused the request.*; and Yahoo Finance did not answer"):
        fo.fetch_chain(_http({"https://cdn.cboe.com/": (403, "no")}), "AAPL", yf=broken)


# ------------------------------------------------------------------------------------------------------------------ Deribit
def test_deribit_prices_are_converted_from_coin_units_and_the_carry_is_read_off_the_published_forward(tmp_path):
    payload = deribit_options_payload("BTC", index=80000.0, quote_date="2026-10-11", vol=0.55, basis=0.06)
    d = fo.fetch_deribit_chain(_http({fo.DERIBIT_URL: payload}), "btc")
    assert d.name == "BTC-Deribit" and d.meta["source_id"] == "deribit" and d.meta["american_style"] is False and d.meta["spot"] == 80000.0 and d.meta["rate"] == 0.0
    df = d.frame
    assert (df["rate"] == 0.0).all() and df["dividend"].between(-0.07, -0.05).all(), "dividend is minus the annual basis of the forward"
    path = tmp_path / "BTC.csv"
    df.to_csv(path, index=False)
    chain = oa.prepare_chain(load_option_csv(path, underlying="BTC"))
    near = chain[(chain["T"] * 365 > 15) & (chain["k"].abs() < 0.12)]
    assert len(near) >= 8 and np.nanmedian(np.abs(near["iv"] - 0.55)) < 0.02, "the dollar prices and the forward reproduce Deribit's own implied volatility"
    with pytest.raises(ValueError, match="BTC, ETH"):
        fo.fetch_deribit_chain(_http({}), "SOL")
    with pytest.raises(FetchError, match="no option list"):
        fo.fetch_deribit_chain(_http({fo.DERIBIT_URL: {"error": "x"}}), "ETH")
