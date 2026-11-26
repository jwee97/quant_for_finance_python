"""The shared Data Health workspace: coverage, integrity and research readiness of the data the app holds, and the controls to act on them.

``GET  /api/health/capabilities``  every provider and dataset with exactly one status (verified live, verified from a recorded response, mock tested, subscription blocked, unsupported, not tested) and the evidence
``GET  /api/health/datasets``      the datasets that can be checked (the platform snapshot, the normalised market store) with their size
``GET  /api/health/report``        ``dataset=platform|market``: coverage, findings and readiness of every series (cached by the data's fingerprint; ``fresh=1`` recomputes)
``GET  /api/health/missing``       ``dataset``, ``series``: the missing runs of one series
``GET  /api/health/source``        ``dataset``, ``series``: where one series came from (provider, endpoint, versions, last sync) or, for the platform snapshot, what is known of it
``GET  /api/health/readiness``     ``session``: whether the instruments of a research session can be used, series by series, with reasons
``GET  /api/health/corrections``   the corrections log (raw values are kept; a correction never rewrites stored data)
``POST /api/health/validate``      (a job) run every check on a dataset and keep a snapshot (fingerprint, counts) so that a later run can show whether the data changed
``POST /api/health/refresh``       (a job) bring market-store series up to date incrementally; the platform snapshot is versioned with the repository and is not refreshed here
``POST /api/health/retry``         run again the validation that failed (the same as validate; it is safe to repeat)
``POST /api/health/export``        the findings of a dataset as CSV text
``POST /api/health/correction``    record a correction ``{series, date, field, raw, corrected, reason}``
``GET  /api/health/diagnostics``   storage health (free disk, every database's integrity), the newest backup's verification, the job registry after the last restart, the newest errors with their request ids, and the provider error categories
``POST /api/health/verify-backup`` restore a backup into a scratch folder and prove it: the archive is whole, no member escapes the folder, every database passes SQLite's integrity check and every JSON file parses (``name``)

Nothing is deleted or repaired; no key is ever sent to the page.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd

from ..data import health as dh
from ..marketdata import capability_registry
from .api import ApiError
from .payload import clean

MAX_SERIES = 400
DATASETS = ("platform", "market")


class HealthApi:
    def __init__(self, app):
        self.app = app
        self._cache: dict = {}

    @property
    def dir(self) -> Path:
        return self.app.root / "data" / "user" / "health"

    def get_routes(self) -> dict:
        return {"/api/health/capabilities": self.capabilities, "/api/health/datasets": self.datasets, "/api/health/report": self.report, "/api/health/missing": self.missing, "/api/health/source": self.source,
                "/api/health/readiness": self.readiness, "/api/health/corrections": self.corrections, "/api/health/snapshots": self.snapshots, "/api/health/diagnostics": self.diagnostics}

    def post_routes(self) -> dict:
        return {"/api/health/validate": self.validate, "/api/health/refresh": self.refresh, "/api/health/retry": self.validate, "/api/health/export": self.export, "/api/health/correction": self.correction, "/api/health/verify-backup": self.verify_backup}

    # ------------------------------------------------------------------------------------------------------ the data
    def _dataset(self, name: str):
        """``(frames, asset_class, expected_index, as_of, note)`` of a dataset. The platform snapshot is an aligned panel (one calendar); the market store holds each series on its own calendar."""
        if name == "platform":
            b = self.app.builder.default_bundle()
            marks = b.marks if b.marks is not None else None
            frames = {}
            for t in list(b.assets)[:MAX_SERIES]:
                px = b.prices[t]
                if marks is not None and t in marks.columns:
                    px = px.where(~marks[t].astype(bool))                        # a mark carried over a closed market is not an observation
                cols = {"close": px}
                for k in ("high", "low", "volume"):
                    frame = getattr(b, k, None)
                    if frame is not None and t in frame.columns:
                        cols[k] = frame[t]
                frames[t] = pd.DataFrame(cols).dropna(how="all", subset=["close"])
            return frames, dict(b.asset_class), pd.DatetimeIndex(b.index), pd.Timestamp.now(), "a versioned snapshot kept with the repository: prices carried over a closed market are not counted as observations"
        if name == "market":
            store = self.app.data.market.store
            frames, classes = {}, {}
            for s in store.list_series()[:MAX_SERIES]:
                if s["timespan"] != "day":
                    continue
                key = (s["provider"], s["asset_class"], s["symbol"], s["timespan"], s["multiplier"], s["adjusted"])
                label = f"{s['provider']}:{s['asset_class']}:{s['symbol']}" + ("" if s["adjusted"] else " (unadjusted)")
                frames[label] = store.read_bars(key)
                classes[label] = s["asset_class"]
            return frames, classes, None, pd.Timestamp.now(), "the normalised store of provider bars: every revision is kept as a version"
        raise ApiError("dataset must be one of: " + ", ".join(DATASETS))

    def datasets(self, q: dict | None = None) -> dict:
        out = []
        for name in DATASETS:
            try:
                frames, _, _, _, note = self._dataset(name)
                out.append({"name": name, "series": len(frames), "note": note, "available": bool(frames)})
            except ApiError:
                raise
            except Exception as error:                                           # a dataset that cannot be opened is reported, not hidden
                out.append({"name": name, "series": 0, "available": False, "note": f"could not be read: {error}"})
        return {"datasets": out}

    def _report(self, name: str, fresh: bool = False) -> dict:
        frames, classes, expected, as_of, note = self._dataset(name)
        fp = dh.fingerprint(frames)
        hit = self._cache.get(name)
        if hit and hit["fingerprint"] == fp and not fresh:
            return hit
        rep = dh.report(name, frames, classes if classes else "equity", as_of=as_of, expected_index=expected).as_dict()
        rep["note"] = note
        self._cache[name] = rep
        return rep

    def report(self, q: dict) -> dict:
        name = str(q.get("dataset") or "platform")
        rep = self._report(name, str(q.get("fresh") or "") in ("1", "true"))
        return clean(rep)

    def missing(self, q: dict) -> dict:
        rep = self._report(str(q.get("dataset") or "platform"))
        series = str(q.get("series") or "")
        row = next((c for c in rep["coverage"] if c["series"] == series), None)
        if row is None:
            raise ApiError("unknown series")
        return clean({"series": series, "calendar": row["calendar"], "runs": row["missing_runs"], "periods": row["missing_periods"], "expected": row["expected"], "observations": row["observations"]})

    def source(self, q: dict) -> dict:
        name, series = str(q.get("dataset") or "platform"), str(q.get("series") or "")
        if name == "platform":
            b = self.app.builder.default_bundle()
            if series not in b.assets:
                raise ApiError("unknown series")
            return clean({"series": series, "dataset": name, "asset_class": b.asset_class.get(series, "unknown"), "first": str(b.index[0].date()), "last": str(b.index[-1].date()),
                          "origin": "the platform snapshot kept with the repository; its individual provider is not recorded per series", "versions": []})
        store = self.app.data.market.store
        for s in store.list_series():
            label = f"{s['provider']}:{s['asset_class']}:{s['symbol']}" + ("" if s["adjusted"] else " (unadjusted)")
            if label == series:
                key = (s["provider"], s["asset_class"], s["symbol"], s["timespan"], s["multiplier"], s["adjusted"])
                return clean({"series": series, "dataset": name, "info": s, "versions": store.versions(key), "origin": f"{s['provider']} (see the version list for each sync)"})
        raise ApiError("unknown series")

    def readiness(self, q: dict) -> dict:
        """Can the instruments of a research session be used? Series the platform snapshot does not hold are reported as missing, not skipped."""
        sid = str(q.get("session") or "")
        try:
            sess = self.app.research.store.get_session(sid)
        except Exception as error:
            raise ApiError(str(error)) from None
        assets = list((sess.get("context") or {}).get("assets") or [])
        rep = self._report("platform")
        status = {r["series"]: r for r in rep["readiness"]}
        rows = []
        for a in assets:
            r = status.get(a)
            rows.append(r or {"series": a, "status": "blocked", "blocking": ["the platform snapshot holds no data for this instrument; download it first"], "caveats": []})
        worst = "ready" if all(r["status"] == "ready" for r in rows) else ("blocked" if any(r["status"] == "blocked" for r in rows) else "ready_with_caveats")
        return clean({"session": sid, "assets": assets, "rows": rows, "overall": worst if rows else "no_instruments", "fingerprint": rep["fingerprint"]})

    # ------------------------------------------------------------------------------------------------------ the log and the snapshots
    def snapshots(self, q: dict | None = None) -> dict:
        p = self.dir / "snapshots.jsonl"
        rows = []
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines()[-200:]:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        return {"snapshots": rows[::-1]}

    def corrections(self, q: dict | None = None) -> dict:
        return clean({"corrections": dh.CorrectionLog(self.dir / "corrections.jsonl").entries((q or {}).get("series") or None), "note": "A correction records the raw value and the corrected one. Stored data are not rewritten."})

    def correction(self, body: dict) -> dict:
        try:
            e = dh.CorrectionLog(self.dir / "corrections.jsonl").record(str(body.get("series") or ""), body.get("date"), str(body.get("field") or "close"), body.get("raw"), body.get("corrected"), str(body.get("reason") or ""), "user")
        except (ValueError, TypeError) as error:
            raise ApiError(str(error)) from None
        return {"correction": clean(e)}

    # ------------------------------------------------------------------------------------------------------ controls
    def capabilities(self, q: dict | None = None) -> dict:
        return clean(capability_registry.build(self.app.root))

    def validate(self, body: dict) -> dict:
        name = str(body.get("dataset") or "platform")
        if name not in DATASETS:
            raise ApiError("dataset must be one of: " + ", ".join(DATASETS))

        def work(progress):
            progress(f"checking {name}")
            rep = self._report(name, fresh=True)
            snap = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dataset": name, "fingerprint": rep["fingerprint"], "series": len(rep["coverage"]), "counts": rep["counts"],
                    "findings": {s: sum(1 for f in rep["findings"] if f["severity"] == s) for s in ("error", "warning", "info")}}
            self.dir.mkdir(parents=True, exist_ok=True)
            with (self.dir / "snapshots.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(snap) + "\n")
            previous = [s for s in self.snapshots()["snapshots"][1:] if s["dataset"] == name]
            snap["changed_since_last"] = None if not previous else previous[0]["fingerprint"] != rep["fingerprint"]
            return clean({"snapshot": snap, "report": rep})
        return self.app.submit_task(f"Data health: validate {name}", work, kind="data", lane="data")

    def refresh(self, body: dict) -> dict:
        name = str(body.get("dataset") or "market")
        if name == "platform":
            raise ApiError("the platform snapshot is versioned with the repository and is not refreshed here; download newer data into the market store or the lab you need")
        symbols = [str(s) for s in (body.get("symbols") or [])]
        if not symbols:
            raise ApiError("choose the symbols to bring up to date")
        return self.app.data.market_sync({"symbols": symbols, "asset_class": body.get("asset_class") or "stocks", "start": body.get("start")})

    def export(self, body: dict) -> dict:
        name = str(body.get("dataset") or "platform")
        rep = self._report(name)
        return {"filename": f"data_health_{name}.csv", "text": dh.findings_csv(rep), "format": "csv"}

    # ------------------------------------------------------------------------------------------------------ reliability
    @property
    def user(self) -> Path:
        return self.app.root / "data" / "user"

    def _databases(self) -> list[dict]:
        out = []
        for path in sorted(list(self.user.rglob("*.db")) + list(self.user.rglob("*.sqlite")))[:60]:
            if "backups" in path.parts:
                continue
            row = {"name": str(path.relative_to(self.user)), "bytes": path.stat().st_size}
            try:
                con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
                try:
                    row["integrity"] = con.execute("PRAGMA integrity_check").fetchone()[0]
                finally:
                    con.close()
            except sqlite3.Error as error:
                row["integrity"] = f"cannot be opened: {error}"
            row["ok"] = row["integrity"] == "ok"
            out.append(row)
        return out

    def diagnostics(self, q: dict | None = None) -> dict:
        from .obs import RequestLog

        user = self.user
        user.mkdir(parents=True, exist_ok=True)
        du = shutil.disk_usage(user)
        free_mb = du.free / 1e6
        disk = {"free_mb": round(free_mb), "total_mb": round(du.total / 1e6), "level": "critical" if free_mb < 200 else "low" if free_mb < 1000 else "ok",
                "note": "Checkpoints, logs and databases are written here; when the disk is full a checkpoint fails (training continues, and the Training monitor says so)."}
        jobs = self.app.jobs_list()
        counts = {st: sum(1 for r in jobs["jobs"] if r["status"] == st) for st in ("queued", "running", "done", "error", "interrupted", "cancelled")}
        log = RequestLog(user / "logs" / "requests.jsonl")
        backups = sorted((user / "backups").glob("quant-lab-backup-*.zip"), reverse=True) if (user / "backups").is_dir() else []
        from ..marketdata.models import MarketDataError  # noqa: F401  (the kinds are documented on the class)

        return clean({"disk": disk, "databases": self._databases(), "latest_backup": backups[0].name if backups else None, "jobs": {"counts": counts, "interrupted": counts["interrupted"],
                      "note": "A job that was running when the app stopped is marked interrupted at the next start; nothing is resumed automatically."}, "recent_errors": log.tail(20, only_errors=True),
                      "provider_error_categories": ["not_configured", "auth", "entitlement", "rate_limit", "not_found", "bad_request", "server", "network", "cancelled", "unsupported"],
                      "request_log": str((user / "logs" / "requests.jsonl").relative_to(self.app.root))})

    def verify_backup(self, body: dict) -> dict:
        name = str(body.get("name") or "")
        folder = self.user / "backups"
        if not re.fullmatch(r"quant-lab-backup-\d{8}-\d{6}\.zip", name):
            raise ApiError("name a backup made by this app (quant-lab-backup-YYYYMMDD-HHMMSS.zip)")
        path = folder / name
        if not path.is_file():
            raise ApiError("no such backup")
        problems, files = [], []
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch).resolve()
            try:
                with zipfile.ZipFile(path) as z:
                    bad = z.testzip()
                    if bad:
                        problems.append(f"the archive is damaged at {bad}")
                    for info in z.infolist():
                        target = (base / info.filename).resolve()
                        if info.filename.startswith(("/", "\\")) or ".." in Path(info.filename).parts or base not in target.parents and target != base:
                            problems.append(f"{info.filename}: path leaves the restore folder (refused)")
                            continue
                        if info.is_dir():
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(z.read(info))
                        row = {"file": info.filename, "bytes": info.file_size}
                        if target.suffix in (".db", ".sqlite"):
                            try:
                                con = sqlite3.connect(str(target))
                                try:
                                    row["integrity"] = con.execute("PRAGMA integrity_check").fetchone()[0]
                                    row["tables"] = {t[0]: con.execute(f'SELECT COUNT(*) FROM "{t[0].replace(chr(34), "")}"').fetchone()[0] for t in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                                finally:
                                    con.close()
                            except sqlite3.Error as error:
                                row["integrity"] = f"cannot be opened: {error}"
                            if row["integrity"] != "ok":
                                problems.append(f"{info.filename}: {row['integrity']}")
                        elif target.suffix == ".json":
                            try:
                                json.loads(target.read_text(encoding="utf-8"))
                                row["parsed"] = True
                            except (ValueError, OSError):
                                row["parsed"] = False
                                problems.append(f"{info.filename}: not valid JSON")
                        files.append(row)
            except zipfile.BadZipFile:
                problems.append("not a zip archive")
        return clean({"name": name, "ok": not problems and bool(files), "files": files, "problems": problems, "note": "The backup was restored into a scratch folder and checked; nothing of the live data was touched. Checkpoint signatures are tied to this machine's key (data/user/.integrity_key): a backup restored on another machine keeps working for plain checkpoints."})
