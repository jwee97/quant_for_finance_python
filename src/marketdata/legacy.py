"""One-off, repeatable import of what earlier versions of the app downloaded from iTick, into the normalised store.

* ``data/user/prices_itick/<TICKER>.csv`` (+ ``.json``): the Backtest tab's cache.
* ``data/user/downloads/{futures,crypto,options}/<NAME>.csv`` whose ``<NAME>.meta.json`` says ``source_id: itick``: the Futures price histories, the Crypto coin histories and the option underlyings' prices.

The files are READ, never changed or deleted. Each becomes a dataset under the provider ``itick`` with origin ``legacy_import`` and the original source note kept, so it can always be told from Massive's
data and is never merged into it. Prices are stored as the files hold them (iTick did not document whether they were adjusted, so ``adjusted`` is recorded as false). A file already imported (same path
and checksum) is skipped, so running this again is harmless.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import MarketStore, file_sha256

PROVIDER = "itick"
NOTE = "legacy iTick download (imported unchanged); iTick did not document whether prices are adjusted"


def _decode(stem: str) -> str:
    return re.sub(r"_([0-9a-f]{2})", lambda m: chr(int(m.group(1), 16)), stem)


def _asset_class(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith("=F"):
        return "futures"
    if t.endswith("-USD") or t.endswith("USDT"):
        return "crypto"
    return "stocks"


def _frame(df: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(df[date_col]), name="date"))
    for c in ("open", "high", "low", "volume"):
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy() if c in df else np.nan
    out["close"] = pd.to_numeric(df[close_col], errors="coerce").to_numpy()
    return out[["open", "high", "low", "close", "volume"]].dropna(subset=["close"]).sort_index()


def _import(store: MarketStore, path: Path, asset_class: str, symbol: str, frame: pd.DataFrame, notes: str, report: list) -> None:
    sha = file_sha256(path)
    source = str(path)
    if store.migrated(source, sha):
        report.append({"file": source, "status": "already imported"})
        return
    if not len(frame):
        report.append({"file": source, "status": "empty, skipped"})
        return
    key = store.key(PROVIDER, asset_class, symbol, "day", 1, False)
    r = store.upsert_bars(key, frame, endpoint="legacy file", origin="legacy_import", reason="legacy import", notes=notes, status="legacy")
    store.mark_migrated(source, sha, PROVIDER, symbol, int(len(frame)))
    report.append({"file": source, "status": "imported", "symbol": symbol, "asset_class": asset_class, "rows": r["rows"], "first": r["first"], "last": r["last"]})


def migrate(root, store: MarketStore | None = None) -> dict:
    """Import every legacy iTick file found under ``root``. Returns a report; nothing on disk is modified apart from the store's database."""
    root = Path(root)
    store = store or MarketStore(root)
    report: list[dict] = []
    folder = root / "data" / "user" / "prices_itick"
    for p in sorted(folder.glob("*.csv")) if folder.is_dir() else []:
        try:
            ticker = _decode(p.stem)
            df = pd.read_csv(p)
            note = NOTE
            meta = p.with_suffix(".json")
            if meta.is_file():
                try:
                    m = json.loads(meta.read_text())
                    note += f"; downloaded {m.get('fetched')} UTC epoch seconds" if m.get("fetched") else ""
                    if m.get("warnings"):
                        note += "; downloader warnings: " + " | ".join(str(w) for w in m["warnings"][:3])
                except (OSError, ValueError):
                    pass
            _import(store, p, _asset_class(ticker), ticker, _frame(df, "date", "close"), note, report)
        except (OSError, ValueError, KeyError) as exc:
            report.append({"file": str(p), "status": f"failed: {exc}"})
    base = root / "data" / "user" / "downloads"
    for kind in ("futures", "crypto", "options"):
        for p in sorted((base / kind).glob("*.csv")) if (base / kind).is_dir() else []:
            meta_path = p.with_name(p.stem + ".meta.json")
            try:
                meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
                if meta.get("source_id") != "itick":
                    continue
                df = pd.read_csv(p)
                notes = NOTE + "; " + " ".join(str(n) for n in (meta.get("notes") or [])[:2])
                if kind == "futures":
                    _import(store, p, "futures", p.stem, _frame(df, "date", "close"), notes + " (one series per product, not a contract)", report)
                elif kind == "crypto":
                    _import(store, p, "crypto", p.stem + "USDT", _frame(df.rename(columns={"time": "date"}), "date", "spot"), notes + " (spot against USDT)", report)
                else:
                    sym = p.stem.removesuffix("_prices")
                    _import(store, p, "stocks", sym, _frame(df, "date", "close"), notes, report)
            except (OSError, ValueError, KeyError) as exc:
                report.append({"file": str(p), "status": f"failed: {exc}"})
    counts = {s: sum(1 for r in report if r["status"] == s) for s in ("imported", "already imported", "empty, skipped")}
    counts["failed"] = sum(1 for r in report if r["status"].startswith("failed"))
    return {"counts": counts, "files": report, "integrity": store.integrity()}
