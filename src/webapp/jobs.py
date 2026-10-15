"""A record on disk of every job the dashboard starts, so that a run does not vanish when the page is reloaded or the app restarts.

A backtest, a machine-learning training, a search or a download runs on the app's own worker thread, not in the page: closing the tab does not stop it. What used to be lost was the *knowledge* of it.
The job table lived in memory only, so a page that was reloaded could not find the run it had started, and an app that was restarted forgot it altogether. :class:`JobLog` keeps one small file per job
under ``data/user/jobs/``:

``<id>.json``         the record: label, kind, status, the last progress message, when it started and finished, and the process that ran it
``<id>.result.json``  the result of a finished job (what ``/api/job/<id>`` returns once it is done), read back only when asked for

A job that was queued or running when its process ended (the app was stopped, restarted or killed; on Replit, the workspace crashed or went idle) can never finish. It is marked ``interrupted``,
with the last thing it said it was doing, as soon as that is known: at the next start, and whenever the list is read. A process id alone cannot say whether the app is still there (the numbers are
reused, and an app in a container is process 1 every time), so a running app also says so every few seconds (:meth:`JobLog.beat`) and another process's job with no sign of life for
``STALE`` seconds counts as gone. Nothing is resumed: a training is not checkpointed, so an interrupted run is started again from the beginning. The log keeps the last 60 jobs, like the table,
and a run can be taken off the list (:meth:`JobLog.remove`).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path

JOB_ID = re.compile(r"^[0-9a-f]{12}$")
ACTIVE = ("queued", "running")
BEAT = 10.0                                                            # seconds between the signs of life of an app that has a job queued or running
STALE = 45.0                                                           # another process's job that has shown none for this long is gone: a process id alone can be reused
PROCESS = uuid.uuid4().hex                                             # names this run of the interpreter: after a restart the process id can be the same (an app in a container is process 1 every time)
RESULT_LIMIT = 40_000_000                                              # a result larger than this is not kept (the record still is)


def _alive(pid: int) -> bool:
    """Whether a process with this id exists (a record written by another running app must not be marked as interrupted). Without a way to ask, say no: Windows has no signal 0."""
    if pid <= 0 or os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class JobLog:
    """The job records of one app root. Every method is safe to call from any thread, and none raises for a damaged or missing file: a log that cannot be written must never stop a run."""

    def __init__(self, directory, keep: int = 60):
        self.directory = Path(directory)
        self.keep = int(keep)
        self._lock = threading.Lock()
        self._seen: set[str] = set()                                    # jobs this log has written: the directory is trimmed when a new one appears, not on every progress message

    # ------------------------------------------------------------------------------------------------------------ write
    def save(self, record: dict, result=None) -> None:
        """Write a job's record (and, for a finished job, its result). ``record`` carries ``id``; the process id is added."""
        job_id = str(record.get("id", ""))
        if not JOB_ID.match(job_id):
            return
        record = {**record, "pid": os.getpid(), "process": PROCESS, "updated": time.time()}
        try:
            with self._lock:
                if result is not None:
                    text = json.dumps(result, separators=(",", ":"), allow_nan=False)
                    if len(text) <= RESULT_LIMIT:
                        _write(self.directory / f"{job_id}.result.json", text)
                        record["has_result"] = True
                _write(self.directory / f"{job_id}.json", json.dumps(record, allow_nan=False))
                if job_id not in self._seen:
                    self._seen.add(job_id)
                    self._trim()
        except (OSError, ValueError, TypeError):
            pass

    def _trim(self) -> None:
        """Keep the newest ``keep`` records (by the time the job was created) and delete the rest with their results."""
        for old in self.records()[self.keep:]:
            for path in (self.directory / f"{old['id']}.json", self.directory / f"{old['id']}.result.json"):
                try:
                    path.unlink()
                except OSError:
                    pass

    def beat(self, job_ids) -> None:
        """A sign of life: refresh ``updated`` on the records of these jobs while they are still queued or running in this process. Only the time is touched, and under the same lock as a save, so
        a job that has just ended is not put back to running by a beat that read it a moment earlier."""
        with self._lock:
            for job_id in job_ids:
                path = self.directory / f"{job_id}.json"
                record = self._read(path)
                if record and record.get("status") in ACTIVE and record.get("process") == PROCESS:
                    record["updated"] = time.time()
                    try:
                        _write(path, json.dumps(record, allow_nan=False))
                    except (OSError, ValueError, TypeError):
                        pass

    def remove(self, job_id: str) -> bool:
        """Take a job off the list: delete its record and its result. ``True`` when there was a record."""
        if not JOB_ID.match(str(job_id)):
            return False
        found = False
        with self._lock:
            for name in (f"{job_id}.json", f"{job_id}.result.json"):
                try:
                    (self.directory / name).unlink()
                    found = True
                except OSError:
                    pass
            self._seen.discard(job_id)
        return found

    # ------------------------------------------------------------------------------------------------------------ read
    def _read(self, path: Path) -> dict | None:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) and JOB_ID.match(str(record.get("id", ""))) else None

    def records(self) -> list[dict]:
        """Every record on disk, newest first (by the time the job was created)."""
        if not self.directory.is_dir():
            return []
        out = [r for r in (self._read(p) for p in self.directory.glob("*.json") if JOB_ID.match(p.stem)) if r]
        return sorted(out, key=lambda r: float(r.get("created") or 0.0), reverse=True)

    def get(self, job_id: str) -> dict | None:
        """One job as the page sees it (``result`` included when it finished and the result was kept), or ``None``. The id is checked before any path is built from it."""
        if not JOB_ID.match(str(job_id)):
            return None
        record = self._read(self.directory / f"{job_id}.json")
        if record is None:
            return None
        view = {k: record.get(k) for k in ("id", "label", "kind", "status", "message", "error", "created", "finished", "ref")}
        view["result"] = None
        if record.get("status") == "done":
            try:
                view["result"] = json.loads((self.directory / f"{job_id}.result.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                view["message"] = "finished, but its result was not kept"
        view["elapsed"] = round(float(record.get("finished") or record.get("updated") or time.time()) - float(record.get("created") or 0.0), 1)
        return view

    # ------------------------------------------------------------------------------------------------------------ lost jobs
    @staticmethod
    def gone(record: dict, now: float | None = None) -> bool:
        """Whether the app that was running this queued or running job is gone."""
        if record.get("process") == PROCESS:
            return False                                                # this very process: it is running it
        pid = int(record.get("pid") or 0)
        if pid == os.getpid() or not _alive(pid):
            return True                                                 # the process is gone, or its id is ours again after a restart (an app in a container is process 1 every time)
        return (time.time() if now is None else now) - float(record.get("updated") or 0.0) > STALE   # some process has that id: it is the app only while it keeps saying so

    def mark_interrupted(self) -> int:
        """Every job recorded as queued or running whose app is gone can never finish. Mark it ``interrupted`` and say so; returns how many. Called at start and whenever the list is read."""
        count = 0
        for record in self.records():
            if record.get("status") not in ACTIVE or not self.gone(record):
                continue
            record.update(status="interrupted", message=f"the app stopped while this was {record.get('status')}: {record.get('message') or 'no progress was recorded'}", finished=time.time())
            self.save(record)
            count += 1
        return count
