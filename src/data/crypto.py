"""Crypto perpetual-futures data for the optional crypto branch (Generation 5).

Sources (public, no credentials): Deribit for hourly funding history and index prices and daily perpetual candles, DefiLlama for the
total stablecoin supply. Raw files are written once, kept immutable and hashed in ``data/metadata/crypto_manifest.json``, like every other
raw data set here. Binance and Bybit refuse connections from this environment, so Deribit is the single venue; cross-exchange spreads
need synchronised quotes from several venues and are not built.

Two simplifications are stated, not hidden: the "carry" asset treats the inverse perpetual as linear (a USD-margined approximation),
and returns are measured between weekday 08:00 UTC marks (Deribit's daily settlement) so that the platform's 252-day annualisation applies.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DERIBIT = "https://www.deribit.com/api/v2/public"
LLAMA = "https://stablecoins.llama.fi/stablecoincharts/all"
INSTRUMENTS = {"BTC": "BTC-PERPETUAL", "ETH": "ETH-PERPETUAL"}
START = "2019-06-01"


def _get(url: str, retries: int = 4):
    last = None
    for k in range(retries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))
        except Exception as error:
            last = error
            time.sleep(2 * (2 ** k))
    raise RuntimeError(f"could not fetch {url}: {last}")


def _ms(ts) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def fetch_funding(instrument: str, start: str = START, end: str | None = None) -> pd.DataFrame:
    end = pd.Timestamp(end or datetime.now(timezone.utc).date())
    rows = []
    cursor = pd.Timestamp(start)
    while cursor < end:
        nxt = min(cursor + pd.DateOffset(months=1), end)
        rows += _get(f"{DERIBIT}/get_funding_rate_history?instrument_name={instrument}&start_timestamp={_ms(cursor)}&end_timestamp={_ms(nxt)}")["result"]
        cursor = nxt
    frame = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    frame["time"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame[["time", "index_price", "interest_1h", "interest_8h"]]


def fetch_perp_daily(instrument: str, start: str = "2018-08-14", end: str | None = None) -> pd.DataFrame:
    end = end or str(datetime.now(timezone.utc).date())
    r = _get(f"{DERIBIT}/get_tradingview_chart_data?instrument_name={instrument}&start_timestamp={_ms(start)}&end_timestamp={_ms(end)}&resolution=1D")["result"]
    return pd.DataFrame({"time": pd.to_datetime(r["ticks"], unit="ms", utc=True), "close": r["close"]})


def fetch_stablecoin_supply() -> pd.DataFrame:
    rows = _get(LLAMA)
    return pd.DataFrame({"time": pd.to_datetime([int(r["date"]) for r in rows], unit="s", utc=True),
                         "supply_usd": [r["totalCirculatingUSD"]["peggedUSD"] for r in rows]})


def ensure_crypto_raw(config, force: bool = False) -> Path:
    directory = config.root / "data" / "raw" / "crypto"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = config.path("metadata") / "crypto_manifest.json"
    needed = [directory / f"{a}_funding.csv.gz" for a in INSTRUMENTS] + [directory / f"{a}_perp_daily.csv" for a in INSTRUMENTS] + [directory / "stablecoin_supply.csv"]
    if force or not all(p.exists() for p in needed):
        files = {}
        for asset, instrument in INSTRUMENTS.items():
            fetch_funding(instrument).to_csv(directory / f"{asset}_funding.csv.gz", index=False, compression="gzip")
            fetch_perp_daily(instrument).to_csv(directory / f"{asset}_perp_daily.csv", index=False)
        fetch_stablecoin_supply().to_csv(directory / "stablecoin_supply.csv", index=False)
        for p in needed:
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps({"fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sources": {"deribit": DERIBIT, "defillama": LLAMA},
                                             "files": files, "data_version": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]}, indent=1))
    return directory


def crypto_version(config) -> str:
    path = config.path("metadata") / "crypto_manifest.json"
    return json.loads(path.read_text())["data_version"] if path.exists() else "unversioned"


def build_crypto_frames(directory: Path) -> dict[str, pd.DataFrame]:
    """Daily (08:00 UTC to 08:00 UTC) frames: spot, perpetual, daily funding earned by a short, and the stablecoin supply."""
    spot, perp, funding = {}, {}, {}
    for asset in INSTRUMENTS:
        f = pd.read_csv(directory / f"{asset}_funding.csv.gz", parse_dates=["time"])
        f = f.set_index("time")
        settle = f[f.index.hour == 8]["index_price"]                       # Deribit's daily candle runs 08:00 to 08:00 UTC
        spot[asset] = settle.groupby(settle.index.normalize()).last()
        funding[asset] = f["interest_1h"].groupby((f.index - pd.Timedelta(hours=8)).floor("D")).sum()
        p = pd.read_csv(directory / f"{asset}_perp_daily.csv", parse_dates=["time"]).set_index("time")["close"]
        perp[asset] = p.groupby(p.index.normalize()).last()
    spot_close = pd.DataFrame(spot)
    spot_close.index = spot_close.index - pd.Timedelta(days=1)          # the 08:00 index of D+1 is the close of the candle that opened on D
    perp_close = pd.DataFrame(perp)
    funding_daily = pd.DataFrame(funding)
    stable = pd.read_csv(directory / "stablecoin_supply.csv", parse_dates=["time"]).set_index("time")["supply_usd"]
    stable.index = stable.index.normalize()
    return {"spot": spot_close.dropna(), "perp": perp_close, "funding": funding_daily, "stable": stable}
