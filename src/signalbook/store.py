"""Where the signal book lives: plain files under ``data/user/signals`` (kept out of the repository).

``follows.json``   what is being followed
``ledger.jsonl``   every snapshot ever issued, one a line, only ever appended to: the record is read from it
``latest.json``    the result of the last run, so the page opens on it without running anything
``settings.json``  the daily schedule and the alert address
``alerts.jsonl``   what the scheduled runs reported
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from pathlib import Path

from .engine import MARKETS

MAX_FOLLOWS = 30
NAME = re.compile(r"^[^\x00-\x1f<>]{1,60}$")
TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
DEFAULT_SETTINGS = {"schedule": {"enabled": False, "time": "18:30", "weekdays_only": True, "refresh": True}, "webhook": "", "last_scheduled": "", "paper": {"cost_bps": 5.0, "long_only": False, "capital": 100000.0}}


class BookError(ValueError):
    """The request cannot be done: say what to change."""


class Book:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------------------------------------------- files
    def _read(self, name: str, default):
        path = self.folder / name
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def _write(self, name: str, value) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.folder / (name + ".tmp")
        tmp.write_text(json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")
        tmp.replace(self.folder / name)

    def _append(self, name: str, value) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, allow_nan=False) + "\n")

    def _lines(self, name: str) -> list[dict]:
        path = self.folder / name
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:                                                                       # a line cut short by a crash is skipped, the rest of the ledger stands
                continue
        return out

    # ---------------------------------------------------------------------------------------------------------- follows
    def follows(self) -> list[dict]:
        with self._lock:
            return list(self._read("follows.json", []))

    def follow(self, follow_id: str) -> dict:
        for f in self.follows():
            if f["id"] == follow_id:
                return f
        raise BookError(f"no follow with id {follow_id!r}")

    def add(self, name: str, market: str, config: dict, rules: list[str]) -> dict:
        if not isinstance(name, str) or not NAME.match(name.strip()):
            raise BookError("give the follow a name of 1 to 60 characters")
        if market not in MARKETS:
            raise BookError(f"market must be one of {', '.join(MARKETS)}")
        with self._lock:
            current = self._read("follows.json", [])
            if len(current) >= MAX_FOLLOWS:
                raise BookError(f"the book holds at most {MAX_FOLLOWS} follows; remove one first")
            if any(f["name"].lower() == name.strip().lower() for f in current):
                raise BookError(f"a follow named {name.strip()!r} already exists")
            entry = {"id": secrets.token_hex(4), "name": name.strip(), "market": market, "config": config, "rules": list(rules), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            self._write("follows.json", current + [entry])
        return entry

    def remove(self, follow_id: str) -> dict:
        """Stop following. Its snapshots stay in the ledger: a record is not erased by losing interest in it."""
        with self._lock:
            current = self._read("follows.json", [])
            kept = [f for f in current if f["id"] != follow_id]
            if len(kept) == len(current):
                raise BookError(f"no follow with id {follow_id!r}")
            self._write("follows.json", kept)
        return {"removed": follow_id}

    # ---------------------------------------------------------------------------------------------------------- the ledger
    def record(self, snap: dict) -> None:
        with self._lock:
            self._append("ledger.jsonl", snap)

    def snapshots(self, follow_id: str | None = None) -> list[dict]:
        with self._lock:
            rows = self._lines("ledger.jsonl")
        return [s for s in rows if follow_id is None or s.get("follow") == follow_id]

    def latest(self) -> dict | None:
        with self._lock:
            return self._read("latest.json", None)

    def save_latest(self, result: dict) -> None:
        with self._lock:
            self._write("latest.json", result)

    # ---------------------------------------------------------------------------------------------------------- settings and alerts
    def settings(self) -> dict:
        with self._lock:
            saved = self._read("settings.json", {})
        out = json.loads(json.dumps(DEFAULT_SETTINGS))
        out["schedule"].update({k: v for k, v in (saved.get("schedule") or {}).items() if k in out["schedule"]})
        out["webhook"] = saved.get("webhook") or ""
        out["last_scheduled"] = saved.get("last_scheduled") or ""
        out["paper"].update({k: v for k, v in (saved.get("paper") or {}).items() if k in out["paper"]})
        return out

    def save_settings(self, schedule: dict | None = None, webhook: str | None = None, last_scheduled: str | None = None, paper: dict | None = None) -> dict:
        current = self.settings()
        if paper is not None:
            if not isinstance(paper, dict):
                raise BookError("paper must be an object")
            try:
                cost, capital = float(paper.get("cost_bps", current["paper"]["cost_bps"])), float(paper.get("capital", current["paper"]["capital"]))
            except (TypeError, ValueError):
                raise BookError("the cost and the capital must be numbers") from None
            if not 0 <= cost <= 200:
                raise BookError("the cost per trade must be between 0 and 200 basis points")
            if not 100 <= capital <= 1e10:
                raise BookError("the paper capital must be between 100 and 10 billion")
            current["paper"] = {"cost_bps": cost, "long_only": bool(paper.get("long_only", current["paper"]["long_only"])), "capital": capital}
        if schedule is not None:
            if not isinstance(schedule, dict):
                raise BookError("schedule must be an object")
            when = schedule.get("time", current["schedule"]["time"])
            if not isinstance(when, str) or not TIME.match(when):
                raise BookError("the time of day must be written HH:MM, such as 18:30")
            current["schedule"] = {"enabled": bool(schedule.get("enabled", current["schedule"]["enabled"])), "time": when, "weekdays_only": bool(schedule.get("weekdays_only", current["schedule"]["weekdays_only"])),
                                   "refresh": bool(schedule.get("refresh", current["schedule"]["refresh"]))}
        if webhook is not None:
            current["webhook"] = check_webhook(webhook)
        if last_scheduled is not None:
            current["last_scheduled"] = last_scheduled
        with self._lock:
            self._write("settings.json", current)
        return current

    def alert(self, entry: dict) -> None:
        with self._lock:
            self._append("alerts.jsonl", entry)

    def alerts(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return self._lines("alerts.jsonl")[-limit:][::-1]


def check_webhook(url) -> str:
    """An alert address: empty (none), or an ``https://`` address (``http://`` only for this computer)."""
    if url in (None, ""):
        return ""
    if not isinstance(url, str) or len(url) > 500 or any(ch.isspace() for ch in url):
        raise BookError("the alert address must be one web address without spaces")
    from urllib.parse import urlparse

    parts = urlparse(url)
    local = parts.hostname in ("127.0.0.1", "localhost", "::1")
    if parts.scheme != "https" and not (parts.scheme == "http" and local):
        raise BookError("the alert address must start with https://")
    if not parts.hostname:
        raise BookError("the alert address has no host")
    return url
