"""The completion protocol of a training unit (an epoch, an optimizer block, a chunk of timesteps): what is durable, in which order it becomes visible, and how a restart tells a finished unit from a half-finished one.

A unit passes through four steps, always in this order:

1. **begin**     ``inflight.json`` says unit ``n`` has started and is not committed (so a kill leaves evidence that work was in progress).
2. **commit**    the unit's checkpoint replaces ``last.pt`` atomically (``CheckpointManager.save``: write, ``fsync``, read back, verify, ``os.replace``). The checkpoint also carries the unit's own record (``last``:
                 the metrics row, the gradient summaries and the warnings), so the logs below are a *copy* of what is already durable. The instant of the ``os.replace`` is the commit point: before it the unit does
                 not exist for recovery, after it the unit is complete whatever happens next. Then ``commit.json`` is replaced atomically with ``state: "committed"``.
3. **apply**     the gradient summary, the metrics row (``committed: true``) and the warnings are appended to the logs, and ``commit.json`` is replaced atomically with ``state: "applied"``.
4. **announce**  job progress and callbacks (information only: recovery never reads them).

A unit that is not checkpointed (``checkpoint_every`` > 1, no improvement) still logs its row for the live monitor, marked ``committed: false``: it is provisional and is recomputed after a restart.

On restart ``reconcile`` compares three witnesses: the checkpoint that was read (the truth for what is finished), ``commit.json`` (what the previous process believed it had committed and applied) and the logs. It

* repairs a torn last line (a kill in the middle of an append);
* drops every row of a unit the checkpoint does not contain (those units are recomputed) and says which;
* restores the row, the gradient summaries and the warnings of the checkpoint's last unit when the kill came between the commit and the apply (also when only part of them was applied);
* reports ``recomputed_work``: the units that were logged or in flight but are not in the checkpoint, and the units ``commit.json`` had committed that the readable checkpoint no longer holds (a damaged newest file).

Nothing here changes a loss, a gradient, an update or a random draw.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

COMMIT_FILE = "commit.json"
INFLIGHT_FILE = "inflight.json"
SCHEMA = 1
#: rows that describe a particular unit's work: they are dropped with the unit when it is recomputed. Every other row (start, resume, interruption, finish, recomputed_work, notes) is history of the run and is kept.
UNIT_ROWS = ("epoch", "gradient_event", "user_stop", "user_pause", "resource_warning", "checkpoint_failed", "early_stopping", "nonfinite_loss")


# ------------------------------------------------------------------------------------------------------ files
def _replace(src: Path, dst: Path) -> None:
    from .checkpoint import _replace as replace                                               # the retry for Windows sharing violations lives in one place

    replace(src, dst)


def _fsync_dir(folder: Path) -> None:
    """Make a rename durable (POSIX). Windows cannot open a directory for this and does not need it for ``os.replace`` on NTFS."""
    try:
        fd = os.open(str(folder), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str, durable: bool = True) -> None:
    """Replace ``path`` with ``text`` in one step: write to a sibling, ``fsync``, ``os.replace``. A reader sees the old content or the new, never a half."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            if durable:
                os.fsync(fh.fileno())
        _replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if durable:
        _fsync_dir(path.parent)


def repair_tail(path: Path) -> int:
    """Cut a half-written last line (a kill in the middle of an append) so that the next append starts on a fresh line instead of gluing two records. Returns the bytes removed."""
    path = Path(path)
    try:
        size = path.stat().st_size
        if size == 0:
            return 0
        with open(path, "rb+") as fh:
            fh.seek(size - 1)
            if fh.read(1) == b"\n":
                return 0
            fh.seek(0)
            data = fh.read()
            cut = data.rfind(b"\n") + 1                                                         # 0 when the file has no complete line at all
            fh.truncate(cut)
            return size - cut
    except OSError:
        return 0


def append_line(path: Path, row: dict, durable: bool = False) -> None:
    """Append one JSON row as one ``write`` call, terminated by a newline. ``durable`` also flushes it to disk (used for the rows that complete a committed unit). Raises ``OSError`` (a full disk): the caller decides."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(row, default=float) + "\n").encode("utf-8")
    with open(path, "ab") as fh:
        fh.write(line)
        fh.flush()
        if durable:
            os.fsync(fh.fileno())


# ------------------------------------------------------------------------------------------------------ commit.json
def read_commit(folder) -> dict | None:
    try:
        doc = json.loads((Path(folder) / COMMIT_FILE).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def write_commit(folder, *, committed: int, applied: int, state: str, file: str = "last.pt", sha256: str | None = None, phase: str | None = None) -> bool:
    """Replace ``commit.json``: ``committed`` units are in the checkpoint, ``applied`` units are in the logs, ``state`` is ``committed`` (between steps 2 and 3) or ``applied``. Returns False when it could not be written
    (a full disk): recovery then relies on the checkpoint alone, which is always sufficient."""
    doc = {"schema": SCHEMA, "committed_unit": int(committed), "applied_unit": int(applied), "state": state, "file": file, "sha256": sha256, "phase": phase, "at": time.time(), "pid": os.getpid()}
    try:
        atomic_write_text(Path(folder) / COMMIT_FILE, json.dumps(doc))
        return True
    except OSError:
        return False


# ------------------------------------------------------------------------------------------------------ in-flight marker
def mark_inflight(folder, unit: int, phase: str | None) -> None:
    """Unit ``unit`` (0-based) has started. Not durable on purpose (one small write per unit): a kill leaves it behind, a power loss may not, and then the unit is still dropped as uncommitted, just not named."""
    if folder is None:
        return
    try:
        (Path(folder) / INFLIGHT_FILE).write_text(json.dumps({"unit": int(unit), "phase": phase, "at": time.time()}))
    except OSError:
        pass


def clear_inflight(folder) -> None:
    if folder is not None:
        try:
            (Path(folder) / INFLIGHT_FILE).unlink(missing_ok=True)
        except OSError:
            pass


def read_inflight(folder, start: int) -> dict | None:
    """The unit that was running when the previous process ended, if it is not part of the checkpoint (``unit >= start``). The marker is consumed."""
    if folder is None:
        return None
    path = Path(folder) / INFLIGHT_FILE
    try:
        note = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    path.unlink(missing_ok=True)
    return note if int(note.get("unit", -1)) >= start else None


# ------------------------------------------------------------------------------------------------------ reconcile
def _read_rows(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:                                                                      # a half-written line in the middle (cannot come from a kill, only from a damaged disk)
            continue
    return rows


def _event_key(r: dict) -> tuple:
    return (r.get("type"), r.get("epoch"), r.get("kind"), r.get("optimizer"), r.get("at"))


def reconcile(folder, done: int, last: dict | None, which: str = "last.pt") -> dict:
    """Make the logs agree with the checkpoint being resumed, which holds ``done`` finished units, and report what that cost. ``last`` is the checkpoint's own record of its newest unit.

    Returns ``{"torn_bytes", "dropped_units" (1-based, logged but not in the checkpoint), "committed_units" (what commit.json had committed), "recovered_row", "recovered_events", "recovered_gradients", "lost_committed"
    (units commit.json says were committed that the checkpoint does not hold: a damaged newest checkpoint), "state_was"}``."""
    folder = Path(folder)
    out = {"torn_bytes": 0, "dropped_units": [], "committed_units": None, "recovered_row": False, "recovered_events": 0, "recovered_gradients": 0, "lost_committed": [], "state_was": None}
    cm = read_commit(folder)
    if cm:
        out["committed_units"], out["state_was"] = int(cm.get("committed_unit", 0)), cm.get("state")
        if int(cm.get("committed_unit", 0)) > done:
            out["lost_committed"] = list(range(done + 1, int(cm["committed_unit"]) + 1))
    mpath, gpath = folder / "metrics.jsonl", folder / "gradients.jsonl"
    out["torn_bytes"] = repair_tail(mpath) + repair_tail(gpath)
    rows = _read_rows(mpath)
    changed = False
    if rows is not None:
        keep, dropped = [], []
        for r in rows:
            t = r.get("type")
            unit_1based = int(r.get("epoch", 0) or 0) + (1 if t in ("gradient_event", "nonfinite_loss") else 0)              # gradient warnings and the non-finite-loss marker carry the 0-based unit, every other row the 1-based count
            if t in UNIT_ROWS and unit_1based > done:
                if t == "epoch":
                    dropped.append(int(r["epoch"]))
                changed = True
                continue
            keep.append(r)
        out["dropped_units"] = sorted(set(dropped))
        have_row = any(r.get("type") == "epoch" and r.get("epoch") == done for r in keep)
        if last and last.get("row") and done > 0:
            if not have_row:
                keep.append({**last["row"], "committed": True, "recovered": True, "checkpoint": {"epoch": done, "recovered": True}})
                out["recovered_row"] = changed = True
            seen = {_event_key(r) for r in keep}
            for ev in last.get("events") or []:
                if _event_key(ev) not in seen:
                    keep.append(ev)
                    out["recovered_events"] += 1
                    changed = True
        if changed:
            _rewrite(mpath, keep)
    grows = _read_rows(gpath)
    if grows is not None:
        gkeep = [r for r in grows if int(r.get("epoch", 0) or 0) <= done - 1]
        have = {(r.get("optimizer"), r.get("epoch")) for r in gkeep}
        for g in ((last or {}).get("grad") or []):
            if g and (g.get("optimizer"), g.get("epoch")) not in have and g.get("epoch") == done - 1:
                gkeep.append({**g, "recovered": True})
                out["recovered_gradients"] += 1
        if len(gkeep) != len(grows) or out["recovered_gradients"]:
            _rewrite(gpath, gkeep)
    return out


def _rewrite(path: Path, rows: list[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, default=float) + "\n" for r in rows), durable=True)


def recompute_report(unit_name: str, start: int, report: dict, inflight: dict | None) -> dict | None:
    """The ``recomputed_work`` row for a restart: the units that will be run again although something of them was seen. ``start`` is the number of finished units in the checkpoint."""
    logged = list(report.get("dropped_units") or [])
    lost = list(report.get("lost_committed") or [])
    running = None if not inflight else int(inflight["unit"]) + 1
    units = sorted(set(logged) | set(lost) | ({running} if running else set()))
    units = [u for u in units if u > start]
    if not units:
        return None
    parts = []
    if running and running in units:
        parts.append(f"{unit_name} {running}{' (' + inflight['phase'] + ')' if inflight.get('phase') else ''} had started and was not committed")
    extra = [u for u in logged if u != running]
    if extra:
        parts.append(f"{_units(unit_name, extra)} {'was' if len(extra) == 1 else 'were'} logged but never committed to a checkpoint")
    if lost:
        parts.append(f"{_units(unit_name, lost)} {'had' if len(lost) == 1 else 'had all'} been committed but the checkpoint that holds {'it' if len(lost) == 1 else 'them'} could not be read")
    return {"type": "recomputed_work", "epoch": int(start), "at": time.time(), "units": units, "unit": units[0], "in_flight": running, "logged_not_committed": [u for u in logged if u != running], "lost_committed": lost,
            "phase": (inflight or {}).get("phase"), "message": "; ".join(parts) + f": recomputed from the checkpoint after {unit_name} {start}"}


def _units(name: str, units: list[int]) -> str:
    return f"{name} {_span(units)}" if len(units) == 1 else f"{name}s {_span(units)}"


def _span(units: list[int]) -> str:
    units = sorted(units)
    return str(units[0]) if len(units) == 1 else f"{units[0]} to {units[-1]}" if units == list(range(units[0], units[-1] + 1)) else ", ".join(map(str, units))
