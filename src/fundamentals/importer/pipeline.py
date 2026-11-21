"""The import pipeline: from a file, a link or an archive to approved figures in the database, with every step recorded.

``queued`` -> ``extracted`` / ``needs_review`` -> ``approved`` -> ``saved``   (or ``rejected``, ``failed``, ``duplicate``)

* A **batch** holds items, one per file (an archive's members are items too). Items and their drafts are stored, so a batch survives a restart and *resumes*: ``process`` extracts only what is still
  ``queued``. The bytes of a file are kept once by content hash; a file already saved from a batch is a ``duplicate`` and is not read again.
* **Extraction prefers structure**: XBRL/iXBRL first, then tables (CSV, Excel), then the text of a PDF (always reviewed). The first extraction is kept untouched (``draft``); a person's corrections are
  a separate copy (``reviewed``), so what was read and what was approved can be compared, and the figures a person changed are marked as edited in their provenance.
* **Nothing is stored without an approval.** Approval runs every check again (:func:`manual.check_entry` per period, :mod:`reconcile` across periods and against the database), refuses an error, asks for a
  decision on a conflict with a figure already held, and stores the periods as the ``manual`` source: a restatement is a new vintage (public from its own filing date), never an overwrite.
* **Provenance**: for every stored figure, the file or link, its content hash, the filing accession, the page, the taxonomy tag, the figure as reported (before unit scaling), the extraction method and
  whether a person edited it (``observation_provenance``).
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import uuid
import zipfile
import zlib

from .. import manual
from ..errors import FundamentalsError
from ..schema import now_iso
from . import pdf as pdf_reader
from . import reconcile, tabular, xbrl
from .fetch import Fetcher

MAX_FILE = 40_000_000
MAX_ZIP_MEMBERS = 200
MAX_ZIP_TOTAL = 250_000_000
MAX_RATIO = 200
STATUSES = ("queued", "extracted", "needs_review", "approved", "saved", "rejected", "failed", "duplicate")
EXT_KIND = {".csv": "csv", ".tsv": "csv", ".txt": "csv", ".xlsx": "xlsx", ".xlsm": "xlsx", ".pdf": "pdf", ".xml": "xbrl", ".xbrl": "xbrl", ".htm": "xbrl", ".html": "xbrl", ".xhtml": "xbrl", ".zip": "zip"}


def sniff(name: str, data: bytes) -> str:
    """What a file is, from its bytes first and its name second: ``zip``, ``pdf``, ``xlsx``, ``xbrl`` (a tagged filing), ``csv`` or ``unknown``."""
    head = data[:8]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK"):
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except zipfile.BadZipFile:
            return "unknown"
        return "xlsx" if any(n.startswith("xl/") for n in names) else "zip"
    if xbrl.looks_like_xbrl(data):
        return "xbrl"
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    kind = EXT_KIND.get(ext)
    if kind in ("csv",) or (kind is None and b"," in data[:2000] or b"\t" in data[:2000]):
        return "csv"
    return "unknown" if kind is None else kind


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ImportService:
    def __init__(self, store, fetcher: Fetcher | None = None):
        self.store = store
        self.fetcher = fetcher or Fetcher()

    # ------------------------------------------------------------------------------------------------------------ batches
    def create_batch(self, label: str = "", security_id: int | None = None, ticker: str | None = None, note: str = "") -> str:
        batch_id = "B" + uuid.uuid4().hex[:10]
        now = now_iso()
        with self.store.db.transaction() as c:
            c.execute("INSERT INTO import_batches (batch_id, created_at, updated_at, label, status, security_id, ticker, note) VALUES (?, ?, ?, ?, 'open', ?, ?, ?)",
                      (batch_id, now, now, (label or "")[:200], security_id, (ticker or "").upper()[:20] or None, (note or "")[:500]))
        return batch_id

    def _batch(self, batch_id: str) -> dict:
        with self.store.db.transaction() as c:
            row = c.execute("SELECT * FROM import_batches WHERE batch_id = ?", (str(batch_id),)).fetchone()
            cols = [d[0] for d in c.execute("SELECT * FROM import_batches LIMIT 0").description]
        if row is None:
            raise FundamentalsError("no such import batch")
        return dict(zip(cols, row))

    def _blob(self, content_hash: str, data: bytes | None = None) -> bytes | None:
        with self.store.db.transaction() as c:
            if data is not None:
                c.execute("INSERT OR IGNORE INTO import_blobs (content_hash, data, size_bytes, created_at) VALUES (?, ?, ?, ?)", (content_hash, zlib.compress(data, 6), len(data), now_iso()))
                return data
            row = c.execute("SELECT data FROM import_blobs WHERE content_hash = ?", (content_hash,)).fetchone()
        return zlib.decompress(row[0]) if row else None

    def _insert(self, batch_id: str, kind: str, name: str, url: str | None, content_hash: str, size: int, parent: int | None = None, status: str = "queued", error: str | None = None) -> int:
        now = now_iso()
        with self.store.db.transaction() as c:
            seq = (c.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM import_items WHERE batch_id = ?", (batch_id,)).fetchone()[0])
            cur = c.execute("INSERT INTO import_items (batch_id, seq, parent_item, source_kind, source_name, source_url, content_hash, size_bytes, status, error, created_at, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (batch_id, seq, parent, kind, name[:300], url, content_hash, size, status, error, now, now))
            c.execute("UPDATE import_batches SET updated_at = ? WHERE batch_id = ?", (now, batch_id))
            return cur.lastrowid

    def _already(self, content_hash: str, batch_id: str) -> str | None:
        """Why a file need not be read again: the same bytes were saved before (from any batch) or are already in this batch."""
        with self.store.db.transaction() as c:
            saved = c.execute("SELECT batch_id FROM import_items WHERE content_hash = ? AND status = 'saved' LIMIT 1", (content_hash,)).fetchone()
            same = c.execute("SELECT item_id FROM import_items WHERE content_hash = ? AND batch_id = ? AND status NOT IN ('duplicate','failed','rejected') LIMIT 1", (content_hash, batch_id)).fetchone()
        if saved:
            return f"the same file was already saved from batch {saved[0]}"
        if same:
            return "the same file is already in this batch"
        return None

    # ------------------------------------------------------------------------------------------------------------ adding
    def add_file(self, batch_id: str, name: str, data: bytes, url: str | None = None, parent: int | None = None, depth: int = 0) -> list[dict]:
        """Add a file (or the members of an archive) to a batch: the bytes are kept once, a duplicate is recorded as one, and nothing is read yet (:meth:`process` does that)."""
        self._batch(batch_id)
        name = (name or "file").replace("\\", "/").split("/")[-1][:200] or "file"
        if not data:
            raise FundamentalsError(f"{name} is empty")
        if len(data) > MAX_FILE:
            raise FundamentalsError(f"{name} is larger than {MAX_FILE // 1_000_000} MB")
        kind = sniff(name, data)
        h = _hash(data)
        out: list[dict] = []
        if kind == "zip":
            if depth >= 1:
                raise FundamentalsError(f"{name}: an archive inside an archive is not opened")
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                raise FundamentalsError(f"{name} is not a readable ZIP archive") from None
            members = [m for m in zf.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX/") and not m.filename.rsplit("/", 1)[-1].startswith(".")]
            if len(members) > MAX_ZIP_MEMBERS:
                raise FundamentalsError(f"{name} holds {len(members)} files; at most {MAX_ZIP_MEMBERS} are read from one archive")
            if sum(m.file_size for m in members) > MAX_ZIP_TOTAL or any(m.compress_size and m.file_size / m.compress_size > MAX_RATIO and m.file_size > 5_000_000 for m in members):
                raise FundamentalsError(f"{name} expands to more than it is safe to read (a very large or highly compressed archive)")
            self._blob(h, data)
            parent = self._insert(batch_id, "zip", name, url, h, len(data), parent, "extracted")
            for m in members:
                if m.file_size > MAX_FILE:
                    self._insert(batch_id, "unknown", m.filename, None, _hash(m.filename.encode()), m.file_size, parent, "failed", "larger than the limit")
                    continue
                out += self.add_file(batch_id, m.filename, zf.read(m), None, parent, depth + 1)
            self._set(parent, note=f"{len(members)} file(s) in the archive")
            return out or [self.item(parent)]
        why = self._already(h, batch_id)
        self._blob(h, data)
        item_id = self._insert(batch_id, kind, name, url, h, len(data), parent, "duplicate" if why else "queued", why)
        return [self.item(item_id)]

    def add_url(self, batch_id: str, url: str, force: bool = False) -> list[dict]:
        """Fetch a report from a link (see :mod:`fetch` for what is and is not fetched) and add it to the batch."""
        got = self.fetcher.fetch(url)
        name = (got.url.rsplit("/", 1)[-1].split("?")[0] or "report")
        return self.add_file(batch_id, name, got.data, url=got.url)

    # ------------------------------------------------------------------------------------------------------------ reading
    def item(self, item_id: int, full: bool = False) -> dict:
        with self.store.db.transaction() as c:
            row = c.execute("SELECT * FROM import_items WHERE item_id = ?", (int(item_id),)).fetchone()
            cols = [d[0] for d in c.execute("SELECT * FROM import_items LIMIT 0").description]
        if row is None:
            raise FundamentalsError("no such import item")
        r = dict(zip(cols, row))
        for k in ("draft", "issues", "reviewed", "saved_periods"):
            r[k] = json.loads(r[k]) if r.get(k) else None
        if not full:
            for k in ("draft", "reviewed"):
                if r[k]:
                    r[k] = {"periods": len(r[k]["periods"]), "method": r[k].get("method")}
        return r

    def _set(self, item_id: int, **fields) -> None:
        cols, args = [], []
        for k, v in fields.items():
            if k == "note":
                continue
            cols.append(f"{k} = ?")
            args.append(json.dumps(v, default=str) if k in ("draft", "issues", "reviewed", "saved_periods") and v is not None else v)
        if not cols:
            return
        cols.append("updated_at = ?")
        args += [now_iso(), int(item_id)]
        with self.store.db.transaction() as c:
            c.execute(f"UPDATE import_items SET {', '.join(cols)} WHERE item_id = ?", args)

    def _extract_bytes(self, kind: str, name: str, data: bytes, hint: dict) -> dict:
        if kind == "xbrl":
            return xbrl.extract(data, name, hint.get("accession"), bool(hint.get("comparatives")))
        if kind == "xlsx":
            return tabular.from_xlsx(data)
        if kind == "csv":
            return tabular.from_text(tabular.decode(data), "csv", {"file": name})
        if kind == "pdf":
            return pdf_reader.extract(data)
        raise FundamentalsError(f"{name}: this kind of file cannot be read (supported: XBRL / inline XBRL filings, CSV, Excel, PDF, and ZIP archives of them)")

    def extract_item(self, item_id: int, hint: dict | None = None) -> dict:
        it = self.item(item_id, full=True)
        data = self._blob(it["content_hash"])
        if data is None:
            self._set(item_id, status="failed", error="the stored file is missing")
            return self.item(item_id)
        hint = hint or {}
        try:
            draft = self._extract_bytes(it["source_kind"], it["source_name"] or "", data, hint)
        except FundamentalsError as error:
            self._set(item_id, status="failed", error=str(error))
            return self.item(item_id)
        draft["source"] = {"file": it["source_name"], "url": it["source_url"], "content_hash": it["content_hash"], "kind": it["source_kind"]}
        for p in draft["periods"]:
            if hint.get("filing_date") and not p.get("filing_date"):
                p["filing_date"] = hint["filing_date"]
        batch = self._batch(it["batch_id"])
        issues = self._issues(draft, batch)
        review = bool(draft.get("review_required")) or any(i["level"] in ("error", "warning") for i in issues)
        self._set(item_id, draft=draft, issues=issues, status="needs_review" if review else "extracted", method=draft.get("method"), error=None)
        return self.item(item_id)

    def process(self, batch_id: str, progress=None, stop=None, hint: dict | None = None) -> dict:
        """Extract every item that is still ``queued`` (so a batch interrupted half way carries on where it stopped) and say what came of each."""
        self._batch(batch_id)
        with self.store.db.transaction() as c:
            todo = [r[0] for r in c.execute("SELECT item_id FROM import_items WHERE batch_id = ? AND status = 'queued' ORDER BY seq", (batch_id,)).fetchall()]
        done = 0
        for n, item_id in enumerate(todo):
            if stop is not None and stop():
                break
            if progress:
                progress(f"reading file {n + 1} of {len(todo)}")
            self.extract_item(item_id, hint)
            done += 1
        return {"batch_id": batch_id, "processed": done, "remaining": len(todo) - done, "summary": self.batch(batch_id)["counts"]}

    # ------------------------------------------------------------------------------------------------------------ checking
    def _entry(self, draft: dict, batch: dict, security: dict | None = None) -> dict:
        entry = {"units": draft.get("units") or {"money": "units", "shares": "units"}, "currency": draft.get("currency"),
                 "periods": [{"fiscal_period": p["fiscal_period"], "period_end": p["period_end"], "fiscal_year": p.get("fiscal_year"), "filing_date": p.get("filing_date"), "currency": p.get("currency"),
                              "values": {k: v for k, v in p["values"].items() if v is not None}} for p in draft["periods"]]}
        sid = (security or {}).get("security_id") or batch.get("security_id")
        if sid:
            entry["security_id"] = sid
        return entry

    def _issues(self, draft: dict, batch: dict, security: dict | None = None) -> list[dict]:
        issues: list[dict] = []
        sid = (security or {}).get("security_id") or batch.get("security_id")
        row = self.store.get_security(int(sid)) if sid else None
        try:
            checked = manual.check_entry(self._entry(draft, batch, security), row)
            issues += [{**i, "kind": "period_check"} for i in checked["issues"]]
        except FundamentalsError as error:
            issues.append({"level": "error", "message": str(error), "period": None, "field": None, "kind": "entry"})
        filing = {f"{p['fiscal_period']} to {p['period_end']}": p.get("filing_date") for p in draft["periods"]}
        both = reconcile.check(self.store, int(sid) if sid else None, draft, filing)
        issues += both["issues"]
        draft["conflicts"] = both["conflicts"]
        if not draft.get("units_stated", True) and draft.get("method") in ("pdf-text", "csv", "xlsx"):
            issues.append({"level": "warning", "message": "the report does not state its unit: confirm whether the amounts are in units, thousands or millions before approving.", "period": None, "field": "units", "kind": "units"})
        for p in draft["periods"]:
            if not p.get("filing_date"):
                issues.append({"level": "info", "message": f"{p['fiscal_period']} to {p['period_end']} has no filing date: it is treated as public a conservative time after the period end and flagged estimated. Enter the date it was filed to use it as of that day.",
                               "period": f"{p['fiscal_period']} to {p['period_end']}", "field": "filing_date", "kind": "no_filing_date"})
        if draft.get("review_required"):
            issues.append({"level": "warning", "message": "read from the text of a PDF: check every figure against the page before approving.", "period": None, "field": None, "kind": "pdf_review"})
        return issues

    # ------------------------------------------------------------------------------------------------------------ review
    def review(self, item_id: int, edits: dict) -> dict:
        """Apply a person's corrections to a copy of the draft (the first extraction stays as it was) and check again. ``edits``: ``units``, ``currency`` and ``periods`` (a list, in the draft's order, of
        ``{fiscal_period, period_end, fiscal_year, filing_date, currency, values: {line: number | null}, drop}``)."""
        it = self.item(item_id, full=True)
        if it["status"] in ("saved", "duplicate", "rejected"):
            raise FundamentalsError(f"this item is {it['status']} and cannot be edited")
        base = it["reviewed"] or it["draft"]
        if base is None:
            raise FundamentalsError("this item has no extracted draft to edit")
        draft = copy.deepcopy(base)
        if edits.get("units"):
            draft["units"] = {"money": str(edits["units"].get("money") or draft["units"]["money"]), "shares": str(edits["units"].get("shares") or draft["units"]["shares"])}
            draft["units_stated"] = True
        if "currency" in edits:
            draft["currency"] = (str(edits["currency"]).upper()[:3] or None)
        incoming = edits.get("periods")
        if incoming is not None:
            if not isinstance(incoming, list) or len(incoming) > len(draft["periods"]):
                raise FundamentalsError("periods must be a list no longer than the draft's")
            keep = []
            for i, p in enumerate(draft["periods"]):
                e = incoming[i] if i < len(incoming) and isinstance(incoming[i], dict) else {}
                if e.get("drop"):
                    continue
                for k in ("fiscal_period", "period_end", "fiscal_year", "filing_date", "currency"):
                    if k in e and e[k] not in (None, ""):
                        p[k] = e[k]
                for code, v in (e.get("values") or {}).items():
                    if code not in manual.canonical.METRICS:
                        raise FundamentalsError(f"{code} is not a line this application knows")
                    src = p.setdefault("sources", {}).setdefault(code, {"tag": None, "method": "manual-edit", "unit": draft["units"]["money"]})
                    if v is None or v == "":
                        p["values"].pop(code, None)
                    else:
                        number = manual.parse_number(v) if isinstance(v, str) else float(v)
                        if p["values"].get(code) != number:
                            p["values"][code] = number
                            src["edited"] = True
                keep.append(p)
            draft["periods"] = keep
            if not keep:
                raise FundamentalsError("every period was dropped: reject the item instead")
        batch = self._batch(it["batch_id"])
        issues = self._issues(draft, batch)
        self._set(item_id, reviewed=draft, issues=issues, status="needs_review" if any(i["level"] in ("error", "warning") for i in issues) else "extracted")
        return self.item(item_id, full=True)

    def reject(self, item_id: int, reason: str = "") -> dict:
        it = self.item(item_id)
        if it["status"] == "saved":
            raise FundamentalsError("a saved item cannot be rejected: delete its periods in Enter reports")
        self._set(item_id, status="rejected", error=(reason or "rejected by a person")[:300])
        return self.item(item_id)

    # ------------------------------------------------------------------------------------------------------------ approval
    def approve(self, item_id: int, security: dict | None = None, decision: str | None = None, restated_on: str | None = None) -> dict:
        """Check the item again and, if nothing blocks it, store its periods. ``security``: ``{"security_id": n}`` or ``{"company": {...}}``. ``decision`` resolves a conflict with figures already held:
        ``restate`` (a later filing: a new vintage), ``correct`` (the held figures were a mistake: overwrite) or ``skip``."""
        it = self.item(item_id, full=True)
        if it["status"] in ("saved", "duplicate", "rejected"):
            raise FundamentalsError(f"this item is already {it['status']}")
        draft = it["reviewed"] or it["draft"]
        if draft is None:
            raise FundamentalsError("this item has no extracted draft: process the batch first")
        batch = self._batch(it["batch_id"])
        security = security or ({"security_id": batch["security_id"]} if batch.get("security_id") else None)
        if not security or not (security.get("security_id") or security.get("company")):
            raise FundamentalsError("choose the company this report belongs to before approving it")
        issues = self._issues(draft, batch, security)
        errors = [i for i in issues if i["level"] == "error"]
        if errors:
            raise FundamentalsError("not approved: " + errors[0]["message"] + (f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""))
        conflicts = draft.get("conflicts") or []
        held = {c["period"] for c in conflicts}
        if conflicts and decision not in ("restate", "correct", "skip"):
            raise FundamentalsError(f"{len(held)} period(s) differ from figures already in the database ({', '.join(sorted(held))[:120]}): choose to store them as a restatement, as a correction of the held figures, or skip them")
        entry = self._entry(draft, batch, security)
        if decision == "skip":
            entry["periods"] = [p for p in entry["periods"] if f"{p['fiscal_period']} to {p['period_end']}" not in held]
            if not entry["periods"]:
                self._set(item_id, status="rejected", error="every period conflicts with held figures: skipped")
                return {"item": self.item(item_id), "saved": []}
        mode = "restate" if decision == "restate" else "correct"
        if mode == "restate":
            dates = [p.get("filing_date") for p in entry["periods"] if p.get("filing_date")]
            restated_on = restated_on or (max(dates) if dates else None)
        self._set(item_id, status="approved")
        result = manual.save(self.store, entry, mode, restated_on)
        sid = int(result.get("security_id") or entry.get("security_id") or 0)
        saved = self._provenance(sid, draft, entry, it)
        self._set(item_id, status="saved", saved_periods=saved)
        with self.store.db.transaction() as c:
            if not batch.get("security_id"):
                c.execute("UPDATE import_batches SET security_id = ? WHERE batch_id = ?", (sid, it["batch_id"]))
        return {"item": self.item(item_id), "saved": saved, "counts": result.get("counts")}

    def _provenance(self, security_id: int, draft: dict, entry: dict, it: dict) -> list[dict]:
        saved = []
        periods = self.store.periods([security_id])
        obs = self.store.observations([security_id])
        by_label = {(p["fiscal_period"], p["period_end"]): p for p in draft["periods"]}
        with self.store.db.transaction() as c:
            for ep in entry["periods"]:
                rows = periods[(periods["provider"] == manual.PROVIDER) & (periods["fiscal_period"] == ep["fiscal_period"]) & (periods["period_end"].dt.strftime("%Y-%m-%d") == ep["period_end"])]
                if rows.empty:
                    continue
                row = rows.sort_values("vintage").iloc[-1]
                pid = int(row["period_id"])
                stored = {r["metric_code"] for _, r in obs[obs["period_id"] == pid].iterrows()}
                src = by_label.get((ep["fiscal_period"], ep["period_end"]), {})
                n = 0
                for code, value in ep["values"].items():
                    if code not in stored:
                        continue
                    s = (src.get("sources") or {}).get(code, {})
                    c.execute("INSERT OR REPLACE INTO observation_provenance (financial_period_id, metric_code, source_kind, source_ref, content_hash, accession, page, taxonomy_tag, reported_value, reported_unit, "
                              "extraction_method, extraction_status, edited, import_item_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (pid, code, it["source_kind"], it["source_url"] or it["source_name"], it["content_hash"], draft.get("accession"), s.get("page"), s.get("tag"), s.get("reported", value), s.get("unit"),
                               s.get("method") or draft.get("method"), "approved", 1 if s.get("edited") else 0, it["item_id"], now_iso()))
                    n += 1
                saved.append({"period": f"{ep['fiscal_period']} to {ep['period_end']}", "period_id": pid, "lines": n, "vintage": int(row["vintage"])})
        return saved

    # ------------------------------------------------------------------------------------------------------------ history
    def batch(self, batch_id: str) -> dict:
        b = self._batch(batch_id)
        with self.store.db.transaction() as c:
            ids = [r[0] for r in c.execute("SELECT item_id FROM import_items WHERE batch_id = ? ORDER BY seq", (batch_id,)).fetchall()]
        items = [self.item(i) for i in ids]
        counts = {s: sum(1 for i in items if i["status"] == s) for s in STATUSES if any(i["status"] == s for i in items)}
        return {**b, "items": items, "counts": counts}

    def history(self, limit: int = 30) -> list[dict]:
        with self.store.db.transaction() as c:
            rows = c.execute("SELECT b.batch_id, b.created_at, b.updated_at, b.label, b.ticker, b.security_id, (SELECT COUNT(*) FROM import_items i WHERE i.batch_id = b.batch_id) "
                             "FROM import_batches b ORDER BY b.created_at DESC LIMIT ?", (int(limit),)).fetchall()
            stat = c.execute("SELECT batch_id, status, COUNT(*) FROM import_items GROUP BY batch_id, status").fetchall()
        counts: dict = {}
        for b, s, n in stat:
            counts.setdefault(b, {})[s] = n
        return [{"batch_id": r[0], "created_at": r[1], "updated_at": r[2], "label": r[3], "ticker": r[4], "security_id": r[5], "items": r[6], "counts": counts.get(r[0], {})} for r in rows]

    def provenance(self, security_id: int, fiscal_period: str | None = None, period_end: str | None = None) -> list[dict]:
        sql = ("SELECT p.fiscal_period, p.period_end, p.vintage, v.metric_code, v.source_kind, v.source_ref, v.accession, v.page, v.taxonomy_tag, v.reported_value, v.reported_unit, v.extraction_method, "
               "v.extraction_status, v.edited, v.content_hash FROM observation_provenance v JOIN financial_periods p ON p.id = v.financial_period_id WHERE p.security_id = ?")
        args: list = [int(security_id)]
        if fiscal_period:
            sql += " AND p.fiscal_period = ?"
            args.append(fiscal_period)
        if period_end:
            sql += " AND p.period_end = ?"
            args.append(period_end)
        with self.store.db.transaction() as c:
            cols = [d[0] for d in c.execute(sql + " LIMIT 0", args).description]
            rows = c.execute(sql + " ORDER BY p.period_end, p.fiscal_period, v.metric_code LIMIT 5000", args).fetchall()
        return [dict(zip(cols, r)) for r in rows]
