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
from pathlib import Path

from .. import manual
from ..errors import FundamentalsError
from ..schema import now_iso
from . import ai_extract, arelle_check, checks as checks_module, docling_adapter, engine, reconcile, tabular, taxonomy, xbrl
from . import html as html_reader
from . import pdf as pdf_reader
from . import currency as currency_module
from .fetch import Fetcher

MAX_FILE = 40_000_000
MAX_ZIP_MEMBERS = 200
MAX_ZIP_TOTAL = 250_000_000
MAX_RATIO = 200
STATUSES = ("queued", "extracted", "needs_review", "approved", "saved", "rejected", "failed", "duplicate")
EXT_KIND = {".csv": "csv", ".tsv": "csv", ".txt": "csv", ".xlsx": "xlsx", ".xlsm": "xlsx", ".pdf": "pdf", ".xml": "xbrl", ".xbrl": "xbrl", ".htm": "html", ".html": "html", ".xhtml": "html", ".zip": "zip"}


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


def _unverified(p: dict) -> dict:
    """The lines whose reading is below the confidence threshold and that no person has confirmed or typed: they are stored flagged ``unverified``, which research excludes."""
    return {code: "unverified" for code, s in (p.get("sources") or {}).items()
            if s.get("confidence") is not None and s["confidence"] < checks_module.LOW_CONFIDENCE and not s.get("verified") and not s.get("edited") and code in p["values"]}


def _diff(old: dict | None, new: dict) -> dict:
    """Which figures differ between two readings of the same document."""
    def flat(d):
        return {(p["fiscal_period"], p["period_end"], c): v for p in (d or {}).get("periods", []) for c, v in p["values"].items()}
    a, b = flat(old), flat(new)
    return {"changed": [{"period": f"{k[0]} to {k[1]}", "code": k[2], "was": a[k], "now": b[k]} for k in sorted(a.keys() & b.keys()) if abs(a[k] - b[k]) > 1e-9 * max(1.0, abs(a[k]))][:100],
            "added": [{"period": f"{k[0]} to {k[1]}", "code": k[2], "now": b[k]} for k in sorted(b.keys() - a.keys())][:100], "removed": [{"period": f"{k[0]} to {k[1]}", "code": k[2], "was": a[k]} for k in sorted(a.keys() - b.keys())][:100]}


class ImportService:
    def __init__(self, store, fetcher: Fetcher | None = None, fx_fetch=None):
        self.store = store
        self.fetcher = fetcher or Fetcher()
        self.fx_fetch = fx_fetch                                                        # None: Yahoo Finance, for the exchange rates a currency conversion needs

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

    def add_structured(self, batch_id: str, name: str, url: str | None, raw: bytes, draft: dict, kind: str = "api") -> dict:
        """Add a draft that a connector built from a source that is already structured (an account API, an open-data file): the answer is kept once by hash like any file, and the draft is checked at once."""
        self._batch(batch_id)
        h = _hash(raw)
        why = self._already(h, batch_id)
        self._blob(h, raw)
        item_id = self._insert(batch_id, kind, name, url, h, len(raw), None, "duplicate" if why else "queued", why)
        if why:
            return self.item(item_id)
        draft["source"] = {"file": name, "url": url, "content_hash": h, "kind": kind}
        issues = self._issues(draft, self._batch(batch_id))
        review = bool(draft.get("review_required")) or any(i["level"] in ("error", "warning") for i in issues)
        self._set(item_id, draft=draft, issues=issues, status="needs_review" if review else "extracted", method=draft.get("method"), error=None)
        return self.item(item_id)

    def add_from_connector(self, batch_id: str, connector, ref) -> list[dict]:
        """Get one filing through a country connector: a structured draft is added as it is, a file goes through the importer's readers with what the filing list knew (accession, filing date)."""
        got = connector.get(ref)
        if got["kind"] == "draft":
            return [self.add_structured(batch_id, got["name"], ref.url, got["raw"], got["draft"])]
        items = self.add_file(batch_id, got["name"], got["data"], url=ref.url)
        package_hash = None
        if got.get("package"):                                                                       # the whole report package, kept once, for an XBRL processor that needs the taxonomy files beside the report
            package_hash = _hash(got["package"])
            self._blob(package_hash, got["package"])
        for it in items:
            if it["status"] == "queued":
                self.set_hint(it["item_id"], {**(got.get("hint") or {}), "fiscal_period": ref.fiscal_period, "package_hash": package_hash})
        return items

    # ------------------------------------------------------------------------------------------------------------ reading
    def item(self, item_id: int, full: bool = False) -> dict:
        with self.store.db.transaction() as c:
            row = c.execute("SELECT * FROM import_items WHERE item_id = ?", (int(item_id),)).fetchone()
            cols = [d[0] for d in c.execute("SELECT * FROM import_items LIMIT 0").description]
        if row is None:
            raise FundamentalsError("no such import item")
        r = dict(zip(cols, row))
        for k in ("draft", "issues", "reviewed", "saved_periods", "hint"):
            r[k] = json.loads(r[k]) if r.get(k) else None
        if not full:
            for k in ("draft", "reviewed"):
                if r[k]:
                    r[k] = {"periods": len(r[k]["periods"]), "method": r[k].get("method")}
        return r

    def set_hint(self, item_id: int, hint: dict) -> None:
        """What is known about an item before it is read (the accession and filing date of a filing found in the SEC's list); used when the item is extracted."""
        with self.store.db.transaction() as c:
            c.execute("UPDATE import_items SET hint = ? WHERE item_id = ? AND status = 'queued'", (json.dumps({k: v for k, v in hint.items() if v}), int(item_id)))

    def _set(self, item_id: int, **fields) -> None:
        cols, args = [], []
        for k, v in fields.items():
            if k == "note":
                continue
            cols.append(f"{k} = ?")
            args.append(json.dumps(v, default=str) if k in ("draft", "issues", "reviewed", "saved_periods", "hint") and v is not None else v)
        if not cols:
            return
        cols.append("updated_at = ?")
        args += [now_iso(), int(item_id)]
        with self.store.db.transaction() as c:
            c.execute(f"UPDATE import_items SET {', '.join(cols)} WHERE item_id = ?", args)

    def _extract_bytes(self, kind: str, name: str, data: bytes, hint: dict) -> dict:
        if kind == "xbrl":
            draft = xbrl.extract(data, name, hint.get("accession"), bool(hint.get("comparatives")))
            if arelle_check.wanted(hint):
                try:
                    cache = Path(self.store.db.path).parent / "arelle_cache" if getattr(self.store.db, "path", None) else None
                    package = self._blob(hint["package_hash"]) if hint.get("package_hash") else None
                    check = arelle_check.crosscheck(data, name, cache, package=package)
                    draft["arelle"] = check
                    draft["notes"].append(f"Arelle cross-check: {check['agree']} of {check['compared']} company-wide numeric facts agree with Arelle ({check['facts_arelle']} facts read by Arelle)"
                                          + (f"; {len(check['different'])} differ" if check["different"] else "") + (f"; Arelle reports {check['validation_errors']} validation message(s) about the filing itself" if check["validation_errors"] else "") + ".")
                except FundamentalsError as error:
                    draft["notes"].append(f"Arelle cross-check was not possible: {error}")
            return draft
        if kind in ("xlsx", "csv"):
            first, draft = None, None
            try:
                draft = tabular.from_xlsx(data) if kind == "xlsx" else tabular.from_text(tabular.decode(data), "csv", {"file": name})
                if engine.lines_found(draft) >= engine.MIN_LINES or hint.get("engine") == "legacy":
                    return draft
                first = FundamentalsError(f"the Enter-reports table reader found only {engine.lines_found(draft)} line(s)")
            except FundamentalsError as error:
                first = error
            try:
                better = self._engine(kind, name, data, hint, first)
            except FundamentalsError:
                if draft is None:
                    raise first
                return draft
            return better if draft is None or engine.lines_found(better) > engine.lines_found(draft) else draft
        if kind in ("html", "pdf"):
            if hint.get("engine") == "legacy":
                return html_reader.extract(data) if kind == "html" else self._pdf(data, name, hint)
            return self._engine(kind, name, data, hint, None)
        raise FundamentalsError(f"{name}: this kind of file cannot be read (supported: XBRL / inline XBRL filings, CSV, Excel, PDF, and ZIP archives of them)")

    def _engine(self, kind: str, name: str, data: bytes, hint: dict, earlier) -> dict:
        """The layout engine first (tables with coordinates, statements, taxonomy, checks); if it finds nothing, the readers that came before it, so nothing that was readable before is lost."""
        try:
            draft, _doc = engine.extract(data, name, kind, hint)
            if earlier is not None:
                draft.setdefault("fallbacks", []).insert(0, {"step": "table-reader", "reason": str(earlier)})
            if kind == "pdf" and draft.get("method") == "pdf-layout" and any(f["step"] == "legacy-reader" for f in draft.get("fallbacks", [])):
                draft["method"] = "pdf-text"
            return draft
        except FundamentalsError as error:
            if kind == "pdf":
                draft = self._pdf(data, name, hint)
            elif kind == "html":
                draft = html_reader.extract(data)
            else:
                raise
            draft.setdefault("fallbacks", []).append({"step": "legacy-reader", "reason": f"the layout engine found nothing ({error})"})
            draft.setdefault("versions", {"extractor": taxonomy.EXTRACTOR_VERSION, "mapping": None, "taxonomy": taxonomy.TAXONOMY_VERSION, "reader": "legacy"})
            return draft

    MIN_PDF_LINES = 5

    def _pdf(self, data: bytes, name: str, hint: dict) -> dict:
        """The chain for a PDF, deterministic first: the words on the page (with OCR for a scanned one, if an engine is installed); if that finds fewer than a handful of known lines, Docling's table
        recovery (if installed); and only then, if switched on, an AI model whose every figure must be found in the document. Each step is marked in the draft's method and all are reviewed."""
        language = hint.get("language") or "en"
        problem = None
        try:
            draft = pdf_reader.extract(data, hint.get("ocr_engine"), language)
            lines = len({c for p in draft["periods"] for c in p["values"]})
            if lines >= self.MIN_PDF_LINES:
                return draft
            problem = FundamentalsError(f"only {lines} statement line(s) were found in the text of this PDF")
        except FundamentalsError as error:
            problem = error
        if docling_adapter.available():
            try:
                tables = docling_adapter.tables(data)
                for text in tables:
                    d = tabular.from_text(text, "pdf-docling", {"file": name})
                    d.update(method="pdf-docling", review_required=True)
                    d["notes"].insert(0, "Tables recovered by Docling's layout analysis because the plain reading found too little: check every figure against the page.")
                    return d
            except FundamentalsError as error:
                problem = problem or error
        if ai_extract.enabled():
            try:
                text = " ".join(" ".join(w[3] for w in p) for p in pdf_reader._words(data))
                return ai_extract.extract(text, hint.get("ai_client"), language)
            except FundamentalsError as error:
                problem = FundamentalsError(f"{problem}; AI-assisted extraction: {error}")
        raise problem

    def extract_item(self, item_id: int, hint: dict | None = None) -> dict:
        it = self.item(item_id, full=True)
        data = self._blob(it["content_hash"])
        if data is None:
            self._set(item_id, status="failed", error="the stored file is missing")
            return self.item(item_id)
        hint = dict(hint or {})
        if it.get("hint"):
            hint = {**it["hint"], **hint}
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
        self._record_extraction(item_id, draft)
        return self.item(item_id)

    def _record_extraction(self, item_id: int, draft: dict) -> None:
        v = draft.get("versions") or {}
        conf = (draft.get("confidence") or {}).get("mean")
        with self.store.db.transaction() as c:
            c.execute("INSERT INTO import_extractions (item_id, extractor_version, mapping_version, taxonomy_version, standard, method, lines, mean_confidence, draft, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (int(item_id), v.get("extractor") or taxonomy.EXTRACTOR_VERSION, v.get("mapping"), v.get("taxonomy"), (draft.get("standard") or {}).get("standard"), draft.get("method"),
                       engine.lines_found(draft) if draft.get("periods") else 0, conf, json.dumps(draft, default=str), now_iso()))

    def extractions(self, item_id: int) -> list[dict]:
        """Every reading of an item, oldest first, with the versions that made it."""
        with self.store.db.transaction() as c:
            rows = c.execute("SELECT id, extractor_version, mapping_version, taxonomy_version, standard, method, lines, mean_confidence, created_at FROM import_extractions WHERE item_id = ? ORDER BY id", (int(item_id),)).fetchall()
        return [dict(r) for r in rows]

    def reprocess(self, item_id: int, hint: dict | None = None) -> dict:
        """Read the stored original again with the current extractor and mappings, keeping every earlier reading. An item that is already saved keeps its stored figures untouched and gets a comparison:
        the lines that changed, appeared or disappeared. An item that is not saved is moved to the new reading (the person's earlier corrections stay in the history)."""
        it = self.item(item_id, full=True)
        data = self._blob(it["content_hash"])
        if data is None:
            raise FundamentalsError("the stored original is missing: it cannot be read again")
        hint = {**(it.get("hint") or {}), **(hint or {})}
        old = it["reviewed"] or it["draft"]
        try:
            draft = self._extract_bytes(it["source_kind"], it["source_name"] or "", data, hint)
        except FundamentalsError as error:
            return {"item_id": item_id, "changed": False, "error": str(error), "versions": self.extractions(item_id)}
        draft["source"] = {"file": it["source_name"], "url": it["source_url"], "content_hash": it["content_hash"], "kind": it["source_kind"]}
        diff = _diff(old, draft)
        self._record_extraction(item_id, draft)
        if it["status"] not in ("saved", "duplicate", "rejected"):
            batch = self._batch(it["batch_id"])
            issues = self._issues(draft, batch)
            self._set(item_id, draft=draft, reviewed=None, issues=issues, status="needs_review", method=draft.get("method"), error=None)
        return {"item_id": item_id, "changed": bool(diff["changed"] or diff["added"] or diff["removed"]), "saved_figures_untouched": it["status"] == "saved", "diff": diff, "versions": self.extractions(item_id)}

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
                              "values": {k: v for k, v in p["values"].items() if v is not None}, "flags": _unverified(p)} for p in draft["periods"]]}
        sid = (security or {}).get("security_id") or batch.get("security_id")
        if sid:
            entry["security_id"] = sid
        return entry

    def _issues(self, draft: dict, batch: dict, security: dict | None = None) -> list[dict]:
        issues: list[dict] = []
        issues += currency_module.apply(draft, draft.get("convert_to"), currency_module.Rates(self.store, self.fx_fetch))
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
        ar = draft.get("arelle")
        if ar and not ar.get("agrees"):
            why = "Arelle read no values (its taxonomy could not be loaded)" if ar.get("no_values") else f"{len(ar.get('different') or [])} fact(s) differ, {len(ar.get('only_arelle') or [])} are only in Arelle and {len(ar.get('only_this_reader') or [])} only in this reader"
            issues.append({"level": "warning", "message": f"the Arelle cross-check does not agree with this reader: {why}. See the draft's arelle section.", "period": None, "field": None, "kind": "arelle"})
        elif ar and not ar.get("valid", True):
            issues.append({"level": "info", "message": f"Arelle agrees with every fact this reader found, and reports {ar.get('validation_errors')} validation error(s) about the filing itself (for example an inconsistent calculation in the filer's tagging): the figures are as filed.", "period": None, "field": None, "kind": "arelle"})
        if draft.get("review_required"):
            issues.append({"level": "warning", "message": "read from the text of a PDF: check every figure against the page before approving.", "period": None, "field": None, "kind": "pdf_review"})
        for c in draft.get("checks") or []:
            if c["status"] in ("fail", "warn"):
                issues.append({"level": "warning", "message": f"{c['label']}: {c['detail']}", "period": c.get("period"), "field": (c.get("lines") or [None])[0], "kind": "recon"})
        low = [(p, code) for p in draft["periods"] for code in _unverified(p)]
        if low:
            issues.append({"level": "warning", "message": f"{len(low)} figure(s) were read with low confidence (below {checks_module.LOW_CONFIDENCE:.0%}); approved, they are stored as unverified and kept out of research until you confirm them against the page (tick Confirmed on the line) or correct them.",
                           "period": None, "field": None, "kind": "low_confidence"})
        for f in draft.get("fallbacks") or []:
            issues.append({"level": "info", "message": f"fallback used: {f['step']} ({f['reason']})", "period": None, "field": None, "kind": "fallback"})
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
            currency_module.restore(draft)                                              # a currency typed by a person is the currency the figures are in: undo any conversion first
            draft["currency"] = (str(edits["currency"]).upper()[:3] or None)
            draft.pop("convert_to", None)
        if "convert_to" in edits:
            draft["convert_to"] = (str(edits["convert_to"]).strip().upper()[:3] or None)
            if draft["convert_to"] and not manual.CURRENCY.match(draft["convert_to"]):
                raise FundamentalsError("convert_to is a three-letter currency code such as USD, or empty to keep the report as reported")
        if "currency_change_ok" in edits:
            draft["currency_change_ok"] = bool(edits["currency_change_ok"])
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
        verify = edits.get("verify")
        if verify:
            for p in draft["periods"]:
                for code in (list(p["values"]) if verify == "all" else verify):
                    if code in p["values"]:
                        p.setdefault("sources", {}).setdefault(code, {"tag": None, "method": "manual-edit", "unit": draft["units"]["money"]})["verified"] = True
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
                    v = draft.get("versions") or {}
                    verified = bool(s.get("verified") or s.get("edited") or s.get("confidence") is None or s["confidence"] >= checks_module.LOW_CONFIDENCE)
                    c.execute("INSERT OR REPLACE INTO observation_provenance (financial_period_id, metric_code, source_kind, source_ref, content_hash, accession, page, taxonomy_tag, reported_value, reported_unit, "
                              "extraction_method, extraction_status, edited, import_item_id, created_at, confidence, statement, scope, extractor_version, mapping_version, original_currency, bbox, verified) "
                              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (pid, code, it["source_kind"], it["source_url"] or it["source_name"], it["content_hash"], draft.get("accession"), s.get("page"), s.get("tag"), s.get("reported", value), s.get("unit"),
                               s.get("method") or draft.get("method"), "approved" if verified else "unverified", 1 if s.get("edited") else 0, it["item_id"], now_iso(), s.get("confidence"), s.get("statement"), s.get("scope"),
                               v.get("extractor"), v.get("mapping"), s.get("original_currency") or (src.get("pre_currency") or ep.get("currency")), json.dumps(s["bbox"]) if s.get("bbox") else None, 1 if verified else 0))
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
