"""Sessions and saved results of the research workspace, as small versioned JSON files.

``<root>/sessions/<id>.json``    one session: its name, its context (see :mod:`context`), its notes and its comparison groups
``<root>/artifacts/<id>.json``   one saved result: what produced it, the context it was made under, its metrics and, for a return stream, the dates and the equity curve

Why files and not a database: the application already keeps its other research records this way (``data/user/signals``, ``data/user/jobs``), a session is a few kilobytes, and a person can read, copy
or delete one by hand. The browser keeps nothing but the id of the session that is open: a result may hold thousands of points and must not sit in ``localStorage``.

Every record carries ``schema``. A record from an older version is migrated when it is read (the old file is kept next to it as ``.v<n>.bak``); a record written by a *newer* version is never
rewritten, and reading it says so. Writes are atomic (a temporary file, then a rename), so a crash leaves the old record or the new one, never half of one.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
from pathlib import Path

from .context import ContextError, validate_context

SCHEMA = 1
ID = re.compile(r"^[0-9a-f]{12}$")
KINDS = ("backtest", "ml_experiment", "regime", "options", "forecast", "tune")
EVIDENCE = ("in_sample", "out_of_sample", "walk_forward", "historical_simulation", "simulated", "retrospective", "paper", "live")
DATA_KINDS = ("platform", "user_file", "downloaded", "simulated", "mixed", "unavailable")          # the vocabulary of the provenance badges the tabs already show
MAX_SESSIONS, MAX_ARTIFACTS, MAX_NOTES, MAX_GROUPS = 200, 200, 200, 20
MAX_POINTS, MAX_ARTIFACT_CHARS, MAX_SESSION_CHARS = 20_000, 8_000_000, 1_000_000


class StoreError(ValueError):
    """A request the store cannot honour (unknown id, a limit, a record from a newer version); the message is meant for the person."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return secrets.token_hex(6)


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


def _finite(x):
    """A JSON-safe number: ``None`` for NaN and infinity (a missing value must not become the text ``NaN``)."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    return x


# ------------------------------------------------------------------------------------------------------ migration of older records
def _session_from_v0(rec: dict) -> dict:
    """Version 0 was the first shape a session was kept in: the tickers, the first date and the data source at the top level, no context object and no groups."""
    ctx = {k: v for k, v in {"assets": rec.get("tickers") or [], "start": rec.get("start") or "", "data_source": rec.get("source") or ""}.items() if v}
    try:
        ctx = validate_context(ctx)
    except ContextError:
        ctx = {}
    return {"schema": 1, "id": rec["id"], "name": rec.get("name") or "Imported session", "created": rec.get("created") or _now(), "updated": _now(), "context": ctx,
            "notes": [{"id": _new_id(), "ts": _now(), "text": str(n)[:4000]} for n in rec.get("notes", []) if isinstance(n, str)], "groups": [], "artifacts": list(rec.get("runs") or [])}


SESSION_MIGRATIONS = {0: _session_from_v0}                       # version -> function that returns the record as the next version


def migrate_session(rec: dict) -> tuple[dict, bool]:
    """``(record at the current version, whether it had to change)``. A record from a newer version raises :class:`StoreError`."""
    version = int(rec.get("schema", 0))
    if version > SCHEMA:
        raise StoreError(f"this session was written by a newer version of the app (schema {version}; this one reads up to {SCHEMA}); update the app to open it")
    changed = False
    while version < SCHEMA:
        step = SESSION_MIGRATIONS.get(version)
        if step is None:
            raise StoreError(f"no way to read a session of schema {version}")
        rec, version, changed = step(rec), version + 1, True
    return rec, changed


# ------------------------------------------------------------------------------------------------------ the store
class ResearchStore:
    def __init__(self, root):
        self.root = Path(root)
        self._lock = threading.RLock()

    # ---------------------------------------------------------------------------------------------- paths and reading
    def _path(self, kind: str, rec_id: str) -> Path:
        if not ID.match(str(rec_id)):
            raise StoreError("unknown id")
        return self.root / kind / f"{rec_id}.json"

    def _read(self, path: Path) -> dict | None:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return rec if isinstance(rec, dict) and ID.match(str(rec.get("id", ""))) else None

    def _session(self, session_id: str) -> dict:
        path = self._path("sessions", session_id)
        rec = self._read(path)
        if rec is None:
            raise StoreError("unknown session")
        rec, changed = migrate_session(rec)
        if changed:
            old = int(json.loads(path.read_text(encoding="utf-8")).get("schema", 0))
            shutil.copy2(path, path.with_suffix(f".v{old}.bak"))
            self._save_session(rec)
        return rec

    def _save_session(self, rec: dict) -> None:
        rec["updated"] = _now()
        text = json.dumps(rec, allow_nan=False, indent=1)
        if len(text) > MAX_SESSION_CHARS:
            raise StoreError("this session has grown too large to keep: remove notes or groups")
        _write(self._path("sessions", rec["id"]), text)

    # ---------------------------------------------------------------------------------------------- sessions
    def create_session(self, name: str = "", context: dict | None = None) -> dict:
        with self._lock:
            if len(self.list_sessions()) >= MAX_SESSIONS:
                raise StoreError(f"at most {MAX_SESSIONS} sessions are kept; delete one first")
            ctx = validate_context(context or {})
            rec = {"schema": SCHEMA, "id": _new_id(), "name": (name or ctx.get("name") or "Untitled session").strip()[:80], "created": _now(), "updated": _now(), "context": ctx, "notes": [], "groups": [], "artifacts": []}
            self._save_session(rec)
            return rec

    def get_session(self, session_id: str) -> dict:
        with self._lock:
            return self._session(session_id)

    def list_sessions(self) -> list[dict]:
        """A short summary of every session, most recently changed first. A session this version cannot read is listed with ``readable: false`` and the reason."""
        out = []
        folder = self.root / "sessions"
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            raw = self._read(path)
            if raw is None:
                continue
            try:
                rec, _ = migrate_session(dict(raw))
            except StoreError as error:
                out.append({"id": raw["id"], "name": raw.get("name", "?"), "updated": raw.get("updated", ""), "readable": False, "reason": str(error)})
                continue
            ctx = rec.get("context", {})
            out.append({"id": rec["id"], "name": rec["name"], "updated": rec.get("updated", ""), "created": rec.get("created", ""), "readable": True, "assets": ctx.get("assets", []), "start": ctx.get("start", ""),
                        "end": ctx.get("end", ""), "n_artifacts": len(rec.get("artifacts", [])), "n_notes": len(rec.get("notes", [])), "n_groups": len(rec.get("groups", []))})
        return sorted(out, key=lambda r: r["updated"], reverse=True)

    def update_session(self, session_id: str, *, name: str | None = None, context: dict | None = None, replace_context: bool = False) -> dict:
        """Change the name and/or the context. ``context`` is merged over the old one (a field set to ``None`` is removed) unless ``replace_context``."""
        with self._lock:
            rec = self._session(session_id)
            if name is not None:
                rec["name"] = str(name).strip()[:80] or rec["name"]
            if context is not None:
                new = validate_context(context)
                new.update({k: None for k, v in context.items() if v is None})                            # a field given as null is taken out of the context
                merged = new if replace_context else {**rec["context"], **new}
                rec["context"] = {k: v for k, v in merged.items() if v is not None}               # a field given as null is taken out of the context
            self._save_session(rec)
            return rec

    def delete_session(self, session_id: str, *, keep_artifacts: bool = False) -> None:
        with self._lock:
            rec = self._session(session_id)
            if not keep_artifacts:
                for art in rec.get("artifacts", []):
                    try:
                        self._path("artifacts", art).unlink()
                    except (OSError, StoreError):
                        pass
            self._path("sessions", session_id).unlink()

    def duplicate_session(self, session_id: str) -> dict:
        """A new session with the same context, notes and groups and *no* artifacts: a result belongs to the session that made it."""
        with self._lock:
            old = self._session(session_id)
            new = self.create_session(f"{old['name']} (copy)", old["context"])
            new["notes"] = [dict(n, id=_new_id()) for n in old.get("notes", [])]
            self._save_session(new)
            return new

    # ---------------------------------------------------------------------------------------------- notes
    def add_note(self, session_id: str, text: str) -> dict:
        text = str(text or "").strip()
        if not text:
            raise StoreError("a note needs some text")
        with self._lock:
            rec = self._session(session_id)
            if len(rec["notes"]) >= MAX_NOTES:
                raise StoreError(f"at most {MAX_NOTES} notes in a session")
            note = {"id": _new_id(), "ts": _now(), "text": text[:4000]}
            rec["notes"].append(note)
            self._save_session(rec)
            return note

    def delete_note(self, session_id: str, note_id: str) -> None:
        with self._lock:
            rec = self._session(session_id)
            rec["notes"] = [n for n in rec["notes"] if n.get("id") != note_id]
            self._save_session(rec)

    # ---------------------------------------------------------------------------------------------- artifacts
    def add_artifact(self, session_id: str, artifact: dict) -> dict:
        """Keep a result in a session. ``artifact`` is checked (see :func:`check_artifact`); it gets an id and the session id."""
        with self._lock:
            rec = self._session(session_id)
            if len(rec["artifacts"]) >= MAX_ARTIFACTS:
                raise StoreError(f"at most {MAX_ARTIFACTS} results in a session; delete one first")
            art = check_artifact({**artifact, "id": _new_id(), "session_id": session_id, "created": _now(), "schema": SCHEMA})
            text = json.dumps(art, allow_nan=False, separators=(",", ":"))
            if len(text) > MAX_ARTIFACT_CHARS:
                raise StoreError("this result is too large to keep")
            _write(self._path("artifacts", art["id"]), text)
            rec["artifacts"].append(art["id"])
            self._save_session(rec)
            return art

    def get_artifact(self, artifact_id: str, *, series: bool = True) -> dict:
        rec = self._read(self._path("artifacts", artifact_id))
        if rec is None:
            raise StoreError("unknown result")
        if int(rec.get("schema", 0)) > SCHEMA:
            raise StoreError("this result was written by a newer version of the app")
        if not series:
            rec = {k: v for k, v in rec.items() if k != "series"}
        return rec

    def list_artifacts(self, session_id: str | None = None) -> list[dict]:
        """Short summaries (no series), newest first, of one session's results or of all of them."""
        with self._lock:
            ids = self._session(session_id)["artifacts"] if session_id else [p.stem for p in (self.root / "artifacts").glob("*.json")] if (self.root / "artifacts").is_dir() else []
            out = []
            for art_id in ids:
                try:
                    a = self.get_artifact(art_id, series=False)
                except StoreError:
                    continue
                out.append({k: a.get(k) for k in ("id", "session_id", "kind", "title", "created", "evidence", "data_kind", "metrics", "context", "refs")})
            return sorted(out, key=lambda r: r["created"] or "", reverse=True)

    def delete_artifact(self, artifact_id: str) -> None:
        with self._lock:
            art = self.get_artifact(artifact_id, series=False)
            try:
                rec = self._session(art["session_id"])
            except StoreError:
                rec = None
            if rec is not None:
                rec["artifacts"] = [a for a in rec["artifacts"] if a != artifact_id]
                for g in rec["groups"]:
                    g["artifacts"] = [a for a in g["artifacts"] if a != artifact_id]
                if rec["context"].get("selected_run") == artifact_id:
                    rec["context"]["selected_run"] = ""
                self._save_session(rec)
            self._path("artifacts", artifact_id).unlink()

    # ---------------------------------------------------------------------------------------------- comparison groups
    def save_group(self, session_id: str, name: str, artifact_ids: list[str], *, group_id: str | None = None, benchmark: str = "", start: str = "", end: str = "") -> dict:
        with self._lock:
            rec = self._session(session_id)
            ids = [a for a in dict.fromkeys(artifact_ids) if a in rec["artifacts"]]
            if len(ids) != len(list(dict.fromkeys(artifact_ids))):
                raise StoreError("a result in the group is not in this session")
            ctx = validate_context({"start": start, "end": end, "benchmark": benchmark})
            group = {"id": group_id or _new_id(), "name": (name or "Comparison").strip()[:80], "artifacts": ids, "benchmark": ctx.get("benchmark", ""), "start": ctx.get("start", ""), "end": ctx.get("end", "")}
            rest = [g for g in rec["groups"] if g["id"] != group["id"]]
            if len(rest) >= MAX_GROUPS:
                raise StoreError(f"at most {MAX_GROUPS} comparison groups in a session")
            rec["groups"] = rest + [group]
            self._save_session(rec)
            return group

    def delete_group(self, session_id: str, group_id: str) -> None:
        with self._lock:
            rec = self._session(session_id)
            rec["groups"] = [g for g in rec["groups"] if g["id"] != group_id]
            self._save_session(rec)


# ------------------------------------------------------------------------------------------------------ artifact validation
def check_artifact(raw: dict) -> dict:
    """An artifact as it is kept. Raises :class:`StoreError` for anything that would make a comparison later misleading: an unknown kind, evidence or data kind, a series whose parts have different lengths."""
    kind, evidence, data_kind = raw.get("kind"), raw.get("evidence"), raw.get("data_kind")
    if kind not in KINDS:
        raise StoreError(f"unknown kind of result {kind!r}; the kinds are {', '.join(KINDS)}")
    if evidence not in EVIDENCE:
        raise StoreError(f"every saved result says what kind of evidence it is: one of {', '.join(EVIDENCE)}")
    if data_kind not in DATA_KINDS:
        raise StoreError(f"every saved result says what kind of data it rests on: one of {', '.join(DATA_KINDS)}")
    out = {"schema": raw.get("schema", SCHEMA), "id": raw["id"], "session_id": raw["session_id"], "created": raw.get("created") or _now(), "kind": kind, "title": str(raw.get("title") or kind)[:120],
           "evidence": evidence, "data_kind": data_kind}
    ctx = validate_context(raw.get("context") or {}, classes=("equity", "rates", "fixed_income", "credit", "commodity", "real_estate", "crypto", "futures", "options", "mixed", "unknown"))
    out["context"] = ctx
    config = raw.get("config") or {}
    if len(json.dumps(config, default=str)) > 200_000:
        raise StoreError("the configuration is too large to keep")
    out["config"] = json.loads(json.dumps(config, default=str, allow_nan=False)) if config else {}
    out["metrics"] = {str(k)[:60]: _finite(v) for k, v in (raw.get("metrics") or {}).items() if isinstance(v, (int, float, str, type(None))) and not isinstance(v, bool)}
    refs = raw.get("refs") or {}
    out["refs"] = {str(k)[:40]: str(v)[:200] for k, v in refs.items() if v not in (None, "")}
    out["notes"] = [str(n)[:500] for n in (raw.get("notes") or [])][:20]
    series = raw.get("series")
    if series:
        dates = [str(d)[:10] for d in series.get("dates", [])]
        if len(dates) > MAX_POINTS:
            raise StoreError(f"a series of {len(dates)} points is too long to keep (the limit is {MAX_POINTS})")
        if any(b <= a for a, b in zip(dates, dates[1:])):
            raise StoreError("the dates of a series must be strictly increasing")
        keep = {"dates": dates}
        for key in ("equity",):
            if key in series:
                values = [_finite(v) for v in series[key]]
                if len(values) != len(dates):
                    raise StoreError(f"the series {key!r} has {len(values)} values for {len(dates)} dates")
                keep[key] = values
        for group in ("benchmarks", "extra"):
            for name, values in (series.get(group) or {}).items():
                values = [_finite(v) for v in values]
                if len(values) != len(dates):
                    raise StoreError(f"the series {name!r} has {len(values)} values for {len(dates)} dates")
                keep.setdefault(group, {})[str(name)[:60]] = values
        out["series"] = keep
    return out
