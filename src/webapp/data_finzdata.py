"""Finzdata in the Futures tab: find a market, list every contract of it, download its curve and keep what a thirty-day window would otherwise lose.

**What it keeps, and where.** A download of a root is merged into ``data/user/downloads/futures/<ROOT>.csv``, the file the tab already reads for the Yahoo curves, with the rows of the contracts and
days the new answer no longer has: a free plan shows the last 30 days at the time of the call, so the file is what remembers the rest. ``data/user/`` is git-ignored and must stay so: Finzdata's terms
forbid publishing its data "in raw or substantially raw form", and the key is for one person at a time. Beside the files, ``data/user/finzdata_stored.json`` records which roots were updated and when
(the file's own record is rewritten by a Yahoo download of the same root), ``finzdata_roots.json`` keeps the list of markets the service sent and ``finzdata_contracts/<ROOT>.json`` the contracts of a
root, each with the time it was fetched.

**What it does not do.** It cannot recover days the window has already left: the stored file shows them as gaps (``assets.free_data.finzdata.gaps``) and the status says how many days are left before
the newest stored bar slides out of a 30-day window. A downloaded root that the platform has no contract terms for is read with a multiplier of 1 (the notes of the download say so).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..assets.free_data import finzdata as fz
from ..assets.free_data import finzdata_catalog as catalog
from ..assets.free_data import futures as ff
from ..assets.free_data.store import Download, DownloadStore, utc_now
from .finzdata import FinzdataClient

ROOT = re.compile(r"^[A-Z0-9]{1,6}$")
ROOTS_FILE, STORED_FILE, CONTRACTS_FOLDER = "finzdata_roots.json", "finzdata_stored.json", "finzdata_contracts"
ROOTS_FRESH_DAYS, CONTRACTS_FRESH_HOURS = 7, 20
MAX_CALLS_PER_ROOT = 60


def check_root(text: str) -> str:
    root = fz.clean_root(text)
    if not ROOT.match(root):
        raise ValueError(f"'{str(text)[:20]}' is not a Finzdata futures symbol (letters and digits, such as ES, CL or E6): search the list to find one")
    return root


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, default=str, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _now() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def _hours_since(stamp, now: datetime | None = None) -> float:
    try:
        then = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    return ((now or datetime.now(timezone.utc)) - then).total_seconds() / 3600.0


class FinzdataData:
    """The Futures tab's Finzdata source for one app root. ``client`` is made on first use (the key is read then); a test sets it."""

    def __init__(self, root, store: DownloadStore, client: FinzdataClient | None = None, today=None):
        self.root = Path(root)
        self.store = store
        self._client = client
        self.today = today or (lambda: datetime.now(timezone.utc).date())            # a test sets the date the app believes it is
        self._lock = threading.Lock()

    @property
    def client(self) -> FinzdataClient:
        if self._client is None:
            self._client = FinzdataClient(root=self.root)
        return self._client

    @client.setter
    def client(self, value: FinzdataClient) -> None:
        self._client = value

    @property
    def user(self) -> Path:
        return self.root / "data" / "user"

    # ------------------------------------------------------------------------------------------------------------ what is kept
    def registry(self) -> dict[str, dict]:
        """``{root: {first_update, last_update, updates, calls, window_days}}`` of the roots updated from Finzdata, whatever the file's own record now says. ``window_days`` is 30 when the plan showed only the
        last 30 days, else ``None``."""
        raw = _read_json(self.user / STORED_FILE)
        return {k: v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}

    def note_saved(self, d: Download) -> None:
        """Remember that ``d.name`` was updated from Finzdata (call after the download is saved)."""
        with self._lock:
            reg = self.registry()
            old = reg.get(d.name, {})
            reg[d.name] = {"first_update": old.get("first_update") or _now(), "last_update": _now(), "updates": int(old.get("updates", 0)) + 1, "calls": int(old.get("calls", 0)) + int(d.meta.get("calls", 0)),
                           "window_days": d.meta.get("window_days")}
            _write_json(self.user / STORED_FILE, reg)

    def stored_frame(self, root: str) -> pd.DataFrame | None:
        """The stored curve of a root (``date, contract, expiry, price, ...``), or ``None``."""
        meta = self.store.meta("futures", root)
        path = self.store.path("futures", root)
        if not meta or meta.get("item") != "curve" or not path.exists():
            return None
        try:
            frame = pd.read_csv(path, parse_dates=["date", "expiry"])
        except (OSError, ValueError, KeyError):
            return None
        return frame if len(frame) else None

    def coverage(self, root: str, today=None, window_days: int | None = None) -> dict | None:
        """What is stored of a root: from, to, days, contracts, rows, open interest, the gaps in it and, for a plan that shows only a window of ``window_days``, how many days are left before that window
        slides past the newest bar (``left`` is ``None`` for a plan that showed everything)."""
        frame = self.stored_frame(root)
        if frame is None:
            return None
        today = pd.Timestamp(today if today is not None else self.today())
        dates = frame["date"].drop_duplicates()
        newest = dates.max()
        found = fz.gaps(dates, today=None)
        return {"root": root, "stored_from": str(dates.min().date()), "stored_to": str(newest.date()), "days": int(len(dates)), "contracts": int(frame["contract"].nunique()), "rows": int(len(frame)),
                "has_open_interest": bool("open_interest" in frame and frame["open_interest"].notna().any()), "gaps": found[:10], "gap_count": len(found), "window_days": window_days,
                **(fz.staleness(newest, today, window_days) if window_days else {**fz.staleness(newest, today), "left": None, "urgent": False})}

    def stored(self, today=None) -> list[dict]:
        """The roots updated from Finzdata that are still stored, with their coverage, most urgent first."""
        out, live = [], self.live_roots()
        for root, record in self.registry().items():
            cov = self.coverage(root, today, record.get("window_days"))
            if cov is None:
                continue
            market = catalog.find(root, live)
            out.append({**cov, "name": market.name if market else root, "exchange": market.exchange if market else "", "group": market.group if market else "", "last_update": record.get("last_update"),
                        "updates": record.get("updates")})
        return sorted(out, key=lambda r: (r["left"] if r["left"] is not None else 10 ** 6, r["root"]))

    # ------------------------------------------------------------------------------------------------------------ status and search
    def status(self, today=None) -> dict:
        """Everything the card shows, with no call to Finzdata: the key, the day's calls, what is stored and the list of markets held."""
        roots = _read_json(self.user / ROOTS_FILE)
        return {**self.client.describe(), "stored": self.stored(today), "window_days": fz.FREE_WINDOW_DAYS,
                "markets": {"bundled": len(catalog.bundled()), "fetched_at": roots.get("fetched_at") if isinstance(roots, dict) else None, "from_service": len(self.live_roots())},
                "terms": "Finzdata's terms forbid publishing its data and sharing a key: what is stored stays in data/user/ on this machine (git-ignored), and the key is for one person."}

    def live_roots(self) -> list[dict]:
        raw = _read_json(self.user / ROOTS_FILE)
        rows = raw.get("rows") if isinstance(raw, dict) else None
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    def refresh_roots(self) -> dict:
        """Ask Finzdata for its list of markets (one call), keep it, and say what changed against the copy that ships with the app."""
        rows = fz.normalise_roots(self.client.roots())
        _write_json(self.user / ROOTS_FILE, {"fetched_at": _now(), "rows": rows})
        have = {m.root for m in catalog.bundled()}
        new = [r["root"] for r in rows if r["root"] not in have]
        gone = sorted(have - {r["root"] for r in rows})
        return {"markets": len(rows), "new": new, "missing": gone[:20], "missing_count": len(gone), "fetched_at": _now(),
                "with_contract_counts": sum(1 for r in rows if r.get("contracts") is not None)}

    def search(self, query: str = "", group: str | None = None, limit: int = 40) -> dict:
        live = self.live_roots()
        found = catalog.search(query, group, limit, live)
        held = {r["root"]: r for r in self.stored()}
        for row in found["results"]:
            if row["root"] in held:
                row["stored_to"], row["stored_days"] = held[row["root"]]["stored_to"], held[row["root"]]["days"]
        return {**found, "groups": catalog.groups(live), "source": "finzdata"}

    # ------------------------------------------------------------------------------------------------------------ the contracts of a root
    def contracts(self, root: str, refresh: bool = False, today=None) -> dict:
        """Every contract Finzdata lists for ``root``, soonest expiry first, each with what is stored of it. One call, or none when the list was fetched within the last day (``refresh`` asks anyway)."""
        root = check_root(root)
        today = pd.Timestamp(today if today is not None else self.today())
        path = self.user / CONTRACTS_FOLDER / f"{root}.json"
        cached = _read_json(path)
        if not refresh and isinstance(cached, dict) and isinstance(cached.get("rows"), list) and _hours_since(cached.get("fetched_at")) < CONTRACTS_FRESH_HOURS:
            rows, fetched, called, undecodable = cached["rows"], cached["fetched_at"], False, list(cached.get("undecodable") or [])
        else:
            frame = fz.normalise_contracts(self.client.contracts(root), root, today)
            undecodable = frame.attrs["undecodable"]
            rows = [{"code": r.code, "contract": r.contract, "year": int(r.year), "month": int(r.month), "expiry": str(r.expiry.date()), "expiry_basis": r.expiry_basis,
                     "first_date": None if r.first_date is pd.NaT else str(r.first_date.date()), "last_date": None if r.last_date is pd.NaT else str(r.last_date.date())} for r in frame.itertuples()]
            fetched, called = _now(), True
            _write_json(path, {"fetched_at": fetched, "rows": rows, "undecodable": undecodable})
        held = self.stored_frame(root)
        per = held.groupby("contract").agg(stored_rows=("date", "size"), stored_from=("date", "min"), stored_to=("date", "max")) if held is not None else None
        out = []
        for r in rows:
            row = dict(r, status="live" if pd.Timestamp(r["expiry"]) >= today else "expired")
            if per is not None and r["contract"] in per.index:
                p = per.loc[r["contract"]]
                row.update(stored_rows=int(p["stored_rows"]), stored_from=str(p["stored_from"].date()), stored_to=str(p["stored_to"].date()))
            out.append(row)
        live = sum(1 for r in out if r["status"] == "live")
        stored = sum(1 for r in out if r.get("stored_rows"))
        return {"root": root, "name": (catalog.find(root, self.live_roots()) or catalog.Market(root, root, "?", "", "")).name, "contracts": out, "listed": len(out), "live": live, "expired": len(out) - live,
                "stored": stored, "fetched_at": fetched, "called": called, "undecodable": undecodable,
                "basis": {k: sum(1 for r in out if r["expiry_basis"] == k) for k in dict.fromkeys(r["expiry_basis"] for r in out)}}

    # ------------------------------------------------------------------------------------------------------------ a download
    def update_root(self, root: str, *, progress=None, max_calls: int = MAX_CALLS_PER_ROOT, today=None) -> tuple[Download, str]:
        """Download ``root`` and merge it into the stored file; returns the :class:`Download` (not yet saved) and a one-line summary. The first download asks for everything the plan shows; a later one
        asks from a week before the newest stored bar, so a corrected bar replaces its old copy."""
        root = check_root(root)
        today = pd.Timestamp(today if today is not None else self.today())
        earlier = self.stored_frame(root)
        known = root in self.registry()
        start = (pd.to_datetime(earlier["date"]).max() - pd.Timedelta(days=7)) if earlier is not None and known else None
        market = catalog.find(root, self.live_roots())
        d = fz.fetch_curve(self.client, root, start=start, terms=catalog.terms(root, market), today=today, max_calls=max_calls, progress=progress)
        fetched_first, fetched_last = d.frame["date"].min(), d.frame["date"].max()
        notes = list(d.meta["notes"])
        if fz.platform_spec(root) is None:
            notes.append(f"The contract size and tick of {root} are not in the platform's table: dollar figures for it are in price units (a multiplier of 1), and the margins are an illustrative share of the price.")
        window = self.registry().get(root, {}).get("window_days") or (fz.FREE_WINDOW_DAYS if start is None and (fetched_last - fetched_first).days <= fz.FREE_WINDOW_DAYS + 5 else None)
        if window and start is None:
            notes.append(f"Finzdata sent {(fetched_last - fetched_first).days + 1} calendar days of history for {root} (its free plan shows the last {fz.FREE_WINDOW_DAYS}). Each download is merged into the stored file, "
                         f"so the history grows as long as you download again within {fz.FREE_WINDOW_DAYS} days.")
        if earlier is not None:
            before = pd.to_datetime(earlier["date"]).min()
            d.frame = ff.merge_history(earlier, d.frame)
            d.meta["first_date"] = str(d.frame["date"].min().date())
            notes.append(f"Merged with the earlier download: the file now runs from {d.meta['first_date']} (it began {before.date()}) and holds {d.frame['contract'].nunique()} contracts.")
        d.meta["contracts"] = int(d.frame["contract"].nunique())
        found = fz.gaps(d.frame["date"].drop_duplicates(), today=None)
        if found:
            span = ", ".join(f"{g['from']} to {g['to']} ({g['days']} days)" for g in found[:3])
            notes.append(f"{len(found)} stretch{'es' if len(found) > 1 else ''} of the stored history have no bar: {span}{' ...' if len(found) > 3 else ''}. Days that slid out of Finzdata's window between two downloads cannot be "
                         f"fetched again; download at least every {fz.FREE_WINDOW_DAYS - 10} days to keep the history whole.")
        d.meta.update(notes=notes, gap_count=len(found), gaps=found[:10], window_days=window, updated=_now())
        held = d.frame["contract"].nunique()
        message = (f"{held} contracts, {d.frame['date'].nunique()} days to {d.meta['as_of']} (Finzdata, {d.meta['calls']} call{'s' if d.meta['calls'] != 1 else ''})"
                   + (f" · {len(found)} gap{'s' if len(found) > 1 else ''} in the stored history" if found else ""))
        return d, message

    def update_and_save(self, root: str, refresh: dict | None = None, **kwargs) -> dict:
        """:meth:`update_root`, then save the download and remember it. Returns the saved record's summary. A failure leaves the stored file as it was."""
        d, message = self.update_root(root, **kwargs)
        d.meta = {**d.meta, "refresh": refresh or {"source": "finzdata", "items": [d.name]}}
        record = self.store.save(d)
        self.note_saved(d)
        return {"item": d.name, "what": d.item, "ok": True, "name": d.name, "message": message, "source": record.get("source"), "as_of": record.get("as_of"), "rows": record.get("rows")}

