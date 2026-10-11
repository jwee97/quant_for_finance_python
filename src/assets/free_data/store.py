"""Where downloaded free data is kept, and the record of where each file came from.

Files live under ``data/user/downloads/<kind>/`` (``kind`` is ``options``, ``futures`` or ``crypto``), next to the ticker downloads of the dashboard and as git-ignored as they are. Beside every data
file ``NAME.csv`` (or ``NAME.json``) is ``NAME.meta.json``: which source it came from, the address asked, when, the date the numbers are for, how many rows, and what the reader needs to know. The
workspaces read that record to label the data ``downloaded`` and to print where it came from; a file without one is not a download. Your own files in ``data/user/<kind>/`` are never touched, and
they win when the same name exists in both places.

A save is atomic (the file is written beside its final name and renamed), so a download that fails half way never replaces a good file with a broken one.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

KINDS = ("options", "futures", "crypto")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class Download:
    """What a fetcher returns: a table (``frame``) or a JSON document (``payload``), the name it is saved under, and the record of where it came from (``meta``: source, address, the date the numbers
    are for, notes). ``item`` says what the file is: ``chain``, ``curve``, ``history``, ``venues`` or ``funding``."""

    kind: str
    name: str
    item: str
    frame: pd.DataFrame | None = None
    payload: dict | None = None
    meta: dict = field(default_factory=dict)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class DownloadStore:
    """The downloaded files of one app root."""

    def __init__(self, root):
        self.base = Path(root) / "data" / "user" / "downloads"

    # ------------------------------------------------------------------------------------------------------------ paths
    def _check(self, kind: str, name: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not SAFE_NAME.match(name) or name.endswith(".meta"):
            raise ValueError(f"'{name}' is not a usable file name (letters, digits, dot, dash and underscore, at most 40)")

    def folder(self, kind: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        return self.base / kind

    def path(self, kind: str, name: str, ext: str = "csv") -> Path:
        self._check(kind, name)
        return self.folder(kind) / f"{name}.{ext}"

    def meta_path(self, kind: str, name: str) -> Path:
        self._check(kind, name)
        return self.folder(kind) / f"{name}.meta.json"

    # ------------------------------------------------------------------------------------------------------------ write
    def save(self, d: Download) -> dict:
        """Write a :class:`Download` and its record; returns the record."""
        meta = {**d.meta, "item": d.item}
        if d.frame is not None:
            return self.save_frame(d.kind, d.name, d.frame, meta)
        if d.payload is not None:
            return self.save_json(d.kind, d.name, d.payload, meta)
        raise ValueError("a download needs a table or a document")

    def save_frame(self, kind: str, name: str, frame: pd.DataFrame, meta: dict) -> dict:
        """Write ``frame`` as ``NAME.csv`` and its record; returns the record."""
        data = frame.to_csv(index=False).encode("utf-8")
        return self._save(kind, name, "csv", data, {**meta, "rows": int(len(frame))})

    def save_json(self, kind: str, name: str, payload: dict, meta: dict) -> dict:
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return self._save(kind, name, "json", data, meta)

    def _save(self, kind: str, name: str, ext: str, data: bytes, meta: dict) -> dict:
        path = self.path(kind, name, ext)
        record = {**meta, "kind": kind, "name": name, "file": path.name, "bytes": len(data), "fetched_at": meta.get("fetched_at") or utc_now().isoformat().replace("+00:00", "Z")}
        for other in ("csv", "json"):                                                       # one data file per name: a JSON replaces a CSV of the same name and the reverse
            if other != ext:
                try:
                    self.path(kind, name, other).unlink()
                except OSError:
                    pass
        _atomic_write(path, data)
        _atomic_write(self.meta_path(kind, name), json.dumps(record, indent=1, allow_nan=False).encode("utf-8"))
        return record

    # ------------------------------------------------------------------------------------------------------------ read
    def meta(self, kind: str, name: str) -> dict | None:
        try:
            record = json.loads(self.meta_path(kind, name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) and (self.folder(kind) / str(record.get("file", ""))).is_file() else None

    def list(self, kind: str) -> list[dict]:
        """The records of every download of a kind whose data file is still there, newest first. A damaged or orphaned record is skipped, not raised."""
        folder = self.folder(kind)
        out = []
        for meta_file in sorted(folder.glob("*.meta.json")) if folder.is_dir() else []:
            name = meta_file.name[: -len(".meta.json")]
            try:
                record = self.meta(kind, name)
            except ValueError:
                record = None
            if record:
                out.append(record)
        return sorted(out, key=lambda r: str(r.get("fetched_at", "")), reverse=True)

    def files(self, kind: str, item: str | None = None) -> dict[str, Path]:
        """``{name: path}`` of the downloaded CSVs of a kind (optionally only those whose record says ``item``): what a loader reads."""
        out = {}
        for record in self.list(kind):
            if record.get("file", "").endswith(".csv") and (item is None or record.get("item") == item):
                out[record["name"]] = self.folder(kind) / record["file"]
        return out

    def read_json(self, kind: str, name: str) -> dict | None:
        record = self.meta(kind, name)
        if not record or not str(record.get("file", "")).endswith(".json"):
            return None
        try:
            return json.loads((self.folder(kind) / record["file"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def delete(self, kind: str, name: str) -> bool:
        """Remove a download and its record; ``False`` when there was nothing to remove."""
        self._check(kind, name)
        gone = False
        for ext in ("csv", "json", "meta.json"):
            try:
                (self.folder(kind) / f"{name}.{ext}").unlink()
                gone = True
            except FileNotFoundError:
                continue
        return gone
