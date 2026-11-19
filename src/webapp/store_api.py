"""What the app keeps, and what it learned while it ran (``src/store/appdb.py``).

``GET  /api/training/fits``      the networks that were trained, newest first (``job``, ``model``, ``status``, ``limit``, ``offset``)
``GET  /api/training/fit``       one of them with the loss of every epoch (``id``)
``GET  /api/jobs/history``       every job ever run, not only the newest sixty (``kind``, ``status``, ``q``, ``limit``, ``offset``)
``GET  /api/storage/status``     the databases and folders under ``data/user``: size, rows, what is cached
``POST /api/storage/backup``     a consistent copy of every database (SQLite's online backup, so it is safe while the app runs) in one zip under ``data/user/backups``; ``GET /api/storage/backups`` lists them
``POST /api/storage/cache/clear`` forget the kept answers (all, or one ``kind``)

On a host whose disk does not survive a restart or a redeploy (a Replit deployment, a container) the backup is how the databases are carried off before it happens.
"""

from __future__ import annotations

import base64
import sqlite3
import time
import zipfile
from pathlib import Path

from ..ml import resources
from .api import ApiError

BACKUP_LIMIT = 24_000_000          # a backup larger than this is kept on disk but not sent to the page


def _int(raw, label: str, lo: int, hi: int, default: int) -> int:
    if raw in (None, ""):
        return default
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        raise ApiError(f"{label} must be a whole number") from None
    if not lo <= v <= hi:
        raise ApiError(f"{label} must be between {lo} and {hi}")
    return v


def _du(path: Path) -> tuple[int, int]:
    total, files = 0, 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
                files += 1
        except OSError:
            pass
    return total, files


class StoreApi:
    def __init__(self, app):
        self.app = app

    @property
    def db(self):
        return self.app.appdb

    @property
    def user(self) -> Path:
        return self.app.root / "data" / "user"

    def get_routes(self) -> dict:
        return {"/api/training/fits": self.fits, "/api/training/fit": self.fit, "/api/jobs/history": self.jobs_history, "/api/storage/status": self.status, "/api/storage/backups": self.backups,
                "/api/system/resources": self.resources}

    def post_routes(self) -> dict:
        return {"/api/storage/backup": self.backup, "/api/storage/cache/clear": self.cache_clear}

    # ------------------------------------------------------------------------------------------------ training and jobs
    def fits(self, q: dict | None = None) -> dict:
        q = q or {}
        rows = self.db.training_fits(q.get("job") or None, q.get("model") or None, q.get("status") or None, _int(q.get("limit"), "limit", 1, 500, 50), _int(q.get("offset"), "offset", 0, 10**7, 0))
        return {"fits": rows, "ok": self.db.ok, "error": self.db.error}

    def fit(self, q: dict | None = None) -> dict:
        row = self.db.training_fit(_int((q or {}).get("id"), "id", 1, 10**12, 0))
        if row is None:
            raise ApiError("unknown training fit")
        return row

    def jobs_history(self, q: dict | None = None) -> dict:
        q = q or {}
        out = self.db.jobs(q.get("kind") or None, q.get("status") or None, (q.get("q") or "")[:80] or None, _int(q.get("limit"), "limit", 1, 500, 100), _int(q.get("offset"), "offset", 0, 10**7, 0))
        out["counts"] = self.db.job_counts()
        return out

    def resources(self, q: dict | None = None) -> dict:
        """``GET /api/system/resources``: the CPUs and memory this process may really use (the container's limits, not the host's), what is free, and how a run is protected: a memory estimate before it starts and a stop
        with a message, and the epochs kept, before the machine kills the app."""
        out = resources.machine()
        out["protection"] = ["A machine-learning run that would not fit in the free memory is refused before it starts, with what to change.",
                             f"A run that reaches {int(resources.WATCH_FRACTION * 100)}% of the memory it may use is stopped with a message instead of being killed by the host; the epochs it had finished stay in the training history.",
                             f"Numerical work uses {out['threads']} thread(s), the CPU quota rounded up, not one per CPU of the host."]
        return out

    # ------------------------------------------------------------------------------------------------ storage
    def status(self, q: dict | None = None) -> dict:
        databases, folders = [], []
        if self.user.is_dir():
            for path in sorted(list(self.user.rglob("*.db")) + list(self.user.rglob("*.sqlite"))):
                if "backups" in path.parts:
                    continue
                entry = {"name": str(path.relative_to(self.user)), "bytes": sum(p.stat().st_size for p in (path, Path(str(path) + "-wal")) if p.exists()), "tables": None}
                try:
                    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
                    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
                    entry["tables"] = {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names[:60]}
                    conn.close()
                except sqlite3.Error as error:
                    entry["error"] = str(error)
                databases.append(entry)
            for child in sorted(p for p in self.user.iterdir() if p.is_dir()):
                size, files = _du(child)
                folders.append({"name": child.name, "bytes": size, "files": files})
        return {"root": str(self.user), "app_db": {"ok": self.db.ok, "error": self.db.error, "bytes": self.db.size_bytes(), "tables": self.db.table_counts()}, "databases": databases, "folders": folders,
                "cache": self.db.cache_stats(), "jobs": self.db.job_counts(),
                "advice": ["Everything the app keeps is under data/user. On a host whose disk is wiped on a restart or a redeploy (a Replit deployment, a container without a volume), make a backup before it happens and keep the zip elsewhere.",
                           "A backup is a consistent copy even while the app runs. To restore one, stop the app, unzip it over data/user and start it again."]}

    def backups(self, q: dict | None = None) -> dict:
        folder = self.user / "backups"
        rows = [{"name": p.name, "bytes": p.stat().st_size, "created": p.stat().st_mtime} for p in sorted(folder.glob("*.zip"), reverse=True)] if folder.is_dir() else []
        return {"backups": rows[:30], "folder": str(folder)}

    def backup(self, body: dict) -> dict:
        folder = self.user / "backups"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        scratch = folder / f".{stamp}.tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        made = []
        try:
            for path in sorted(list(self.user.rglob("*.db")) + list(self.user.rglob("*.sqlite"))):
                if "backups" in path.parts:
                    continue
                rel = path.relative_to(self.user)
                dest = scratch / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
                    out = sqlite3.connect(str(dest))
                    src.backup(out)
                    out.close()
                    src.close()
                    made.append(str(rel))
                except sqlite3.Error:
                    continue
            if body.get("include_settings", True):
                for name in ("research",):                                           # the research sessions are small JSON files, not a database
                    base = self.user / name
                    if base.is_dir():
                        for p in base.rglob("*"):
                            if p.is_file() and p.suffix == ".json":
                                dest = scratch / p.relative_to(self.user)
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                dest.write_bytes(p.read_bytes())
                                made.append(str(p.relative_to(self.user)))
            if not made:
                raise ApiError("there is nothing to back up yet")
            zpath = folder / f"quant-lab-backup-{stamp}.zip"
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(scratch.rglob("*")):
                    if p.is_file():
                        z.write(p, p.relative_to(scratch))
        finally:
            for p in sorted(scratch.rglob("*"), reverse=True):
                try:
                    p.unlink() if p.is_file() else p.rmdir()
                except OSError:
                    pass
            try:
                scratch.rmdir()
            except OSError:
                pass
        for old in sorted(folder.glob("quant-lab-backup-*.zip"))[:-8]:                # keep the newest eight
            try:
                old.unlink()
            except OSError:
                pass
        size = zpath.stat().st_size
        out = {"name": zpath.name, "bytes": size, "files": made, "path": str(zpath)}
        if size <= BACKUP_LIMIT:
            out["zip_base64"] = base64.b64encode(zpath.read_bytes()).decode("ascii")
        else:
            out["note"] = f"the zip is {size / 1e6:.0f} MB, too large to send to the page: it is on the server at {zpath}"
        return out

    def cache_clear(self, body: dict) -> dict:
        return {"cleared": self.db.cache_clear(str(body.get("kind") or "") or None)}
