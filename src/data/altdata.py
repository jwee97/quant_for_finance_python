"""Non-price data with point-in-time availability (Generation 3, Priority 6).

Same discipline as the macro panel: every series has a reference date and an
availability date, raw files are immutable and checksummed, and the manifest is
separate from the ETF and macro ones so adding data can never rename an earlier
result.

Two sources are handled here. FRED and Yahoo series reuse ``MacroDownloader``
with their own raw directory and manifest. CFTC Commitments of Traders history
files are downloaded as the zips the CFTC publishes, kept byte-for-byte, and
parsed into one tidy positioning table.

Positioning is ``(speculative long - speculative short) / open interest`` for a
named contract (leveraged funds in financial futures, managed money in
commodities). The report is dated Tuesday and released the following Friday
after the close, so a value is treated as known four calendar days after its
date and first becomes usable on the next Monday's close.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .macro import MacroDownloader, MacroSeriesSpec, asof_panel

LOGGER = get_logger(__name__)
CFTC_BASE = "https://www.cftc.gov/files/dea/history/"

# (report type) -> (bundle for the early years, pattern for single years, long column, short column)
_REPORTS = {
    "tff": {"bundle": "fin_fut_txt_2006_2016.zip", "year": "fut_fin_txt_{year}.zip",
            "lev": ("Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All")},
    "disagg": {"bundle": "com_disagg_txt_hist_2006_2016.zip", "year": "com_disagg_txt_{year}.zip",
               "mmoney": ("M_Money_Positions_Long_All", "M_Money_Positions_Short_All")},
}


def load_alt_specs(config) -> list[MacroSeriesSpec]:
    return [MacroSeriesSpec.from_config(n) for n in (config.get("altdata.series", []) or [])]


class AltDownloader(MacroDownloader):
    """FRED and Yahoo series for this stage: the macro downloader with its own manifest."""

    manifest_name = "altdata_manifest.json"


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _fetch(url: str, attempts: int = 4, backoff: float = 2.0) -> bytes:
    last = None
    for k in range(attempts):
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as response:
                return response.read()
        except Exception as exc:                                           # network / provider errors
            last = exc
            LOGGER.warning("cftc %s attempt %d/%d failed: %s", url, k + 1, attempts, exc)
            time.sleep(backoff * 2 ** k)
    raise RuntimeError(f"could not download {url}: {last}")


def download_cftc(raw_dir: str | Path, metadata_dir: str | Path, last_year: int, force: bool = False) -> dict:
    """Download every Commitments of Traders history file once; record sizes and checksums."""
    raw_dir = Path(raw_dir) / "cftc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind, spec in _REPORTS.items():
        names = [spec["bundle"]] + [spec["year"].format(year=y) for y in range(2017, last_year + 1)]
        for name in names:
            path = raw_dir / name
            if force or not path.exists():
                blob = _fetch(CFTC_BASE + name)
                if not zipfile.is_zipfile(io.BytesIO(blob)):
                    raise RuntimeError(f"{name} is not a zip file")
                path.write_bytes(blob)
            files[name] = {"report": kind, "bytes": path.stat().st_size, "sha256": _sha256_bytes(path.read_bytes())}
    manifest = {"manifest_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": CFTC_BASE, "files": files,
                "data_version": hashlib.sha256("".join(v["sha256"] for _, v in sorted(files.items())).encode()).hexdigest()[:12]}
    Path(metadata_dir).mkdir(parents=True, exist_ok=True)
    (Path(metadata_dir) / "cftc_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _normalise_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lstrip("0").str.upper()


def parse_cftc(raw_dir: str | Path, contracts: dict) -> pd.DataFrame:
    """Long table ``date, contract, net_pct_oi, open_interest`` from the downloaded history files."""
    raw_dir = Path(raw_dir) / "cftc"
    frames = []
    for kind, spec in _REPORTS.items():
        wanted = {name: c for name, c in contracts.items() if c["report"] == kind}
        if not wanted:
            continue
        for path in sorted(raw_dir.glob("*.zip")):
            is_tff = path.name.startswith(("fin_fut", "fut_fin"))
            if (kind == "tff") != is_tff:
                continue
            needed = {"CFTC_Contract_Market_Code", "Report_Date_as_YYYY-MM-DD", "Open_Interest_All",
                      *spec[next(iter(g for g in spec if g in ("lev", "mmoney")))]}
            with zipfile.ZipFile(path) as z:
                table = pd.read_csv(z.open(z.namelist()[0]), dtype={"CFTC_Contract_Market_Code": str}, usecols=lambda c: c in needed)
            table["_code"] = _normalise_code(table["CFTC_Contract_Market_Code"])
            table["date"] = pd.to_datetime(table["Report_Date_as_YYYY-MM-DD"], format="mixed").dt.normalize()
            for name, c in wanted.items():
                long_col, short_col = spec[c["group"]]
                part = table[table["_code"] == _normalise_code(pd.Series([c["code"]])).iloc[0]]
                if part.empty:
                    continue
                oi = pd.to_numeric(part["Open_Interest_All"], errors="coerce")
                net = (pd.to_numeric(part[long_col], errors="coerce") - pd.to_numeric(part[short_col], errors="coerce")) / oi
                frames.append(pd.DataFrame({"date": part["date"], "contract": name, "net_pct_oi": net, "open_interest": oi}))
    if not frames:
        raise RuntimeError("no CFTC rows matched the declared contract codes")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["net_pct_oi"])
    return out.sort_values(["contract", "date"]).drop_duplicates(["contract", "date"], keep="last").reset_index(drop=True)


def cftc_series(positioning: pd.DataFrame, lag_days: int) -> tuple[dict[str, pd.Series], list[MacroSeriesSpec]]:
    """Per-contract series and their availability specs (calendar-day lag from the Tuesday date)."""
    series, specs = {}, []
    for name, group in positioning.groupby("contract"):
        key = f"cftc_{name}"
        series[key] = group.set_index("date")["net_pct_oi"].sort_index()
        specs.append(MacroSeriesSpec(id=key, source="cftc", frequency="weekly", release_lag_days=lag_days))
    return series, specs


def ensure_alt_raw(config, force: bool = False):
    """Everything this stage needs on disk: (specs, raw series by id, cftc positioning table)."""
    raw_dir = config.root / "data" / "raw" / "altdata"
    meta = config.path("metadata")
    specs = load_alt_specs(config)
    if force or not all((raw_dir / f"{s.id}.csv").exists() for s in specs):
        manifest = AltDownloader(raw_dir, meta).download_all(specs, force=force)
        failed = [k for k, v in manifest["series"].items() if v["status"] == "failed"]
        if failed:
            raise RuntimeError(f"alternative-data download failed for: {failed}")
    raw = {}
    for s in specs:
        frame = pd.read_csv(raw_dir / f"{s.id}.csv", parse_dates=["date"])
        raw[s.id] = frame.set_index("date")["value"].sort_index()
    table_path = raw_dir / "cftc_positioning.csv"
    cftc_manifest = Path(meta) / "cftc_manifest.json"
    if force or not (table_path.exists() and cftc_manifest.exists()):
        download_cftc(raw_dir, meta, datetime.now(timezone.utc).year, force=force)
        parse_cftc(raw_dir, config.get("altdata.cftc.contracts")).to_csv(table_path, index=False)
    return specs, raw, pd.read_csv(table_path, parse_dates=["date"])


def altdata_version(metadata_dir: str | Path) -> str:
    parts = []
    for name in ("altdata_manifest.json", "cftc_manifest.json"):
        path = Path(metadata_dir) / name
        parts.append(json.loads(path.read_text())["data_version"] if path.exists() else "unversioned")
    return "+".join(parts)


def alt_feature_levels(raw: dict[str, pd.Series], specs: list[MacroSeriesSpec], positioning: pd.DataFrame,
                       vix: pd.Series, vix_spec: MacroSeriesSpec, cftc_lag: int, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Unstandardised features on the trading calendar, each from what was PUBLISHED by that date.

    Transformations that need the raw frequency (a four-week claims average, a 13-week change) are applied before
    the availability shift, so they cannot see a value before it was released.
    """
    by_id = {s.id: s for s in specs}
    claims = raw["ICSA"].astype(float)
    claims_4w = np.log(claims.rolling(4).mean())
    raw_work = {
        "BAA10Y": raw["BAA10Y"], "claims_4w_log": claims_4w, "claims_change_13w": claims_4w.diff(13),
        "VIX3M": raw["VIX3M"], "VXN": raw["VXN"], "GVZ": raw["GVZ"], "OVX": raw["OVX"], "VIX": vix,
    }
    work_specs = {"BAA10Y": by_id["BAA10Y"], "claims_4w_log": by_id["ICSA"], "claims_change_13w": by_id["ICSA"],
                  "VIX3M": by_id["VIX3M"], "VXN": by_id["VXN"], "GVZ": by_id["GVZ"], "OVX": by_id["OVX"], "VIX": vix_spec}
    series_specs = [MacroSeriesSpec(id=k, source=v.source, frequency=v.frequency, release_lag_days=v.release_lag_days)
                    for k, v in work_specs.items()]
    panel = asof_panel(raw_work, series_specs, calendar)
    cf_series, cf_specs = cftc_series(positioning, cftc_lag)
    cf_panel = asof_panel(cf_series, cf_specs, calendar)
    out = pd.DataFrame(index=pd.DatetimeIndex(calendar))
    out["baa10y_level"] = panel["BAA10Y"]
    out["baa10y_change_63"] = panel["BAA10Y"].diff(63)
    out["claims_4w_log"] = panel["claims_4w_log"]
    out["claims_change_13w"] = panel["claims_change_13w"]
    out["vix_over_vix3m"] = panel["VIX"] / panel["VIX3M"]
    out["vxn_minus_vix"] = panel["VXN"] - panel["VIX"]
    out["gvz"] = panel["GVZ"]
    out["ovx"] = panel["OVX"]
    for name in ("es", "nq", "ust10", "gold", "silver", "crude"):
        key = f"cftc_{name}"
        out[key] = cf_panel[key] if key in cf_panel else np.nan
    return out
