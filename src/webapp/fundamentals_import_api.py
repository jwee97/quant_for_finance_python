"""The Fundamentals tab's Report import and Data coverage routes: batches of reports (uploaded in pieces, fetched from links or from the SEC's filing list), read into drafts, reviewed, approved and stored with
their provenance; and the coverage of the database by company, year and quarter.

Real data only, like entering reports by hand: a report is a fact about a real company, and the demo database is a simulated world. The server accepts JSON bodies of up to 200 KB, so a file is sent in
pieces (``/api/fund/import/chunk``) and put together here; the pieces are held in memory, at most 40 MB of them in all, and dropped after half an hour.
"""

from __future__ import annotations

import base64
import binascii
import re
import threading
import time

from ..fundamentals.errors import FundamentalsError
from ..fundamentals import pitstore
from ..fundamentals.filings import calendar as country_calendar
from ..fundamentals.filings import registry as connector_registry
from ..fundamentals.filings.base import FilingRef
from ..fundamentals.importer import coverage_overview as overview_module
from ..fundamentals.importer.fetch import Fetcher
from ..fundamentals.importer.pipeline import MAX_FILE, ImportService
from .api import ApiError

UPLOAD_KEEP_SECONDS = 1800
MAX_PENDING_BYTES = 40_000_000
MAX_CHUNK = 200_000


class ImportRoutes:
    _imports: dict
    _uploads: dict
    _upload_lock: threading.Lock

    def import_get_routes(self) -> dict:
        return {"/api/fund/import/history": self.import_history, "/api/fund/import/batch": self.import_batch, "/api/fund/import/item": self.import_item, "/api/fund/import/provenance": self.import_provenance,
                "/api/fund/coverage/overview": self.coverage_overview, "/api/fund/connectors": self.connectors, "/api/fund/pit/asof": self.pit_asof, "/api/fund/import/diagnostics": self.import_diagnostics, "/api/fund/import/extractions": self.import_extractions}

    def import_post_routes(self) -> dict:
        return {"/api/fund/import/create": self.import_create, "/api/fund/import/chunk": self.import_chunk, "/api/fund/import/urls": self.import_urls, "/api/fund/import/process": self.import_process,
                "/api/fund/import/review": self.import_review, "/api/fund/import/approve": self.import_approve, "/api/fund/import/reject": self.import_reject, "/api/fund/import/filings": self.import_filings,
                "/api/fund/import/filings/add": self.import_filings_add, "/api/fund/coverage/backfill": self.coverage_backfill, "/api/fund/connectors/search": self.connectors_search, "/api/fund/connectors/fetch": self.connectors_fetch,
                "/api/fund/pit/export": self.pit_export, "/api/fund/import/verify": self.import_verify, "/api/fund/import/reprocess": self.import_reprocess}

    # ------------------------------------------------------------------------------------------------------------ plumbing
    def _importer(self) -> ImportService:
        wb = self.bench("real")
        if not hasattr(self, "_imports"):
            self._imports, self._uploads, self._upload_lock = {}, {}, threading.Lock()
        key = id(wb)
        if key not in self._imports:
            ua = ""
            try:
                ua = wb.service.provider("sec_edgar").user_agent
            except Exception:                                                                   # no SEC provider registered: a link to sec.gov will say what is missing
                ua = ""
            self._imports[key] = ImportService(wb.store, Fetcher(user_agent=ua or None))
        return self._imports[key]

    @staticmethod
    def _int(source: dict, name: str) -> int:
        try:
            return int(source.get(name))
        except (TypeError, ValueError):
            raise ApiError(f"{name} must be a whole number") from None

    # ------------------------------------------------------------------------------------------------------------ reading
    def import_history(self, q: dict | None = None) -> dict:
        """``GET /api/fund/import/history``: the batches, newest first, with how many items each holds by status."""
        return {"batches": self._guard(self._importer().history, 40)}

    def import_batch(self, q: dict) -> dict:
        """``GET /api/fund/import/batch?id=``: one batch and its items (without their drafts)."""
        return self._guard(self._importer().batch, str((q or {}).get("id") or ""))

    def import_item(self, q: dict) -> dict:
        """``GET /api/fund/import/item?id=``: one item in full: the first reading (``draft``), the corrected copy (``reviewed``), every issue and conflict."""
        return self._guard(self._importer().item, self._int(q or {}, "id"), True)

    def import_provenance(self, q: dict) -> dict:
        """``GET /api/fund/import/provenance?security_id=[&fiscal_period=&period_end=]``: where each stored figure of a company came from."""
        q = q or {}
        return {"rows": self._guard(self._importer().provenance, self._int(q, "security_id"), q.get("fiscal_period") or None, q.get("period_end") or None)}

    # ------------------------------------------------------------------------------------------------------------ adding
    def import_create(self, body: dict) -> dict:
        """``POST /api/fund/import/create {label?, security_id? | ticker?}``: a new, empty batch (for one company if given)."""
        svc = self._importer()
        sid = body.get("security_id")
        ticker = str(body.get("ticker") or "").strip().upper() or None
        if sid in (None, "") and ticker:
            found = svc.store.securities()
            hit = found[found["ticker"].str.upper() == ticker]
            sid = int(hit.iloc[0]["security_id"]) if len(hit) else None
        return {"batch_id": self._guard(svc.create_batch, str(body.get("label") or ""), int(sid) if sid not in (None, "") else None, ticker, str(body.get("note") or ""))}

    def import_chunk(self, body: dict) -> dict:
        """``POST /api/fund/import/chunk {batch_id, upload_id, name, index, total, data (base64)}``: one piece of a file. The last piece puts the file together and adds it to the batch (an archive adds its members)."""
        svc = self._importer()
        upload = str(body.get("upload_id") or "")
        if not upload or len(upload) > 64:
            raise ApiError("upload_id is needed (up to 64 characters)")
        index, total = self._int(body, "index"), self._int(body, "total")
        if not (0 <= index < total <= 400):
            raise ApiError("index and total do not describe a file of at most 400 pieces")
        try:
            piece = base64.b64decode(str(body.get("data") or ""), validate=True)
        except (binascii.Error, ValueError):
            raise ApiError("data must be base64") from None
        if len(piece) > MAX_CHUNK:
            raise ApiError("a piece may be at most 200,000 bytes")
        with self._upload_lock:
            now = time.time()
            for k in [k for k, v in self._uploads.items() if now - v["at"] > UPLOAD_KEEP_SECONDS]:
                del self._uploads[k]
            slot = self._uploads.setdefault((str(body.get("batch_id")), upload), {"name": str(body.get("name") or "file"), "pieces": {}, "at": now, "total": total})
            if slot["total"] != total:
                raise ApiError("the number of pieces changed during the upload")
            slot["pieces"][index], slot["at"] = piece, now
            if sum(len(p) for v in self._uploads.values() for p in v["pieces"].values()) > MAX_PENDING_BYTES:
                self._uploads.pop((str(body.get("batch_id")), upload), None)
                raise ApiError(f"too much is waiting to be put together (over {MAX_PENDING_BYTES // 1_000_000} MB): wait for the other uploads to finish")
            if len(slot["pieces"]) < total:
                return {"received": len(slot["pieces"]), "total": total, "done": False}
            data = b"".join(slot["pieces"][i] for i in range(total))
            del self._uploads[(str(body.get("batch_id")), upload)]
        if len(data) > MAX_FILE:
            raise ApiError(f"the file is larger than {MAX_FILE // 1_000_000} MB")
        items = self._guard(svc.add_file, str(body.get("batch_id") or ""), slot["name"], data)
        return {"received": total, "total": total, "done": True, "items": items}

    def import_urls(self, body: dict) -> dict:
        """``POST /api/fund/import/urls {batch_id, urls: [...]} (a job)``: fetch reports from links (public http(s) addresses only; see :mod:`importer.fetch`) and add them to the batch."""
        urls = body.get("urls")
        if not isinstance(urls, list) or not urls or len(urls) > 50 or not all(isinstance(u, str) for u in urls):
            raise ApiError("urls must be a list of 1 to 50 links")
        svc, batch = self._importer(), str(body.get("batch_id") or "")
        self._guard(svc.batch, batch)

        def work(progress, stop):
            added, failed = [], []
            for n, url in enumerate(urls):
                if stop is not None and stop.is_set():
                    break
                progress(f"fetching {n + 1} of {len(urls)}")
                try:
                    added += svc.add_url(batch, url)
                except FundamentalsError as error:
                    failed.append({"url": url, "error": str(error)})
            return {"added": len(added), "failed": failed, "batch_id": batch}
        return self._job("fundamentals: fetch reports", work, stoppable=True, lane="data")

    def import_filings(self, body: dict) -> dict:
        """``POST /api/fund/import/filings {symbol | security_id, forms?, limit?}``: the company's 10-K and 10-Q filings at the SEC, with the address of each main document."""
        forms = body.get("forms") or ["10-K", "10-Q"]
        if not isinstance(forms, list) or not all(isinstance(f, str) for f in forms) or len(forms) > 8:
            raise ApiError("forms must be a short list such as [\"10-K\", \"10-Q\"]")
        wb = self.bench("real")
        symbol = str(body.get("symbol") or "").strip()
        if not symbol and body.get("security_id") not in (None, ""):
            row = wb.store.get_security(self._int(body, "security_id"))
            symbol = (row or {}).get("ticker") or ""
        if not symbol:
            raise ApiError("say which company (a ticker, or security_id)")
        try:
            prov = wb.service.provider("sec_edgar")
        except Exception:
            raise ApiError("SEC EDGAR is not available in this app") from None
        return self._guard(prov.filings, symbol, tuple(f.upper() for f in forms), self._limit(body))

    @staticmethod
    def _limit(body: dict) -> int:
        try:
            return max(1, min(int(body.get("limit", 40)), 200))
        except (TypeError, ValueError):
            raise ApiError("limit must be a whole number") from None

    def import_filings_add(self, body: dict) -> dict:
        """``POST /api/fund/import/filings/add {batch_id, filings: [{url, accession, filing_date, form}]} (a job)``: fetch the chosen filings' documents into the batch, each remembering its accession and filing date."""
        items = body.get("filings")
        if not isinstance(items, list) or not items or len(items) > 60:
            raise ApiError("filings must be a list of 1 to 60 filings from the filing list")
        svc, batch = self._importer(), str(body.get("batch_id") or "")
        self._guard(svc.batch, batch)

        def work(progress, stop):
            added, failed = 0, []
            for n, f in enumerate(items):
                if stop is not None and stop.is_set():
                    break
                progress(f"fetching filing {n + 1} of {len(items)}")
                try:
                    got = svc.add_url(batch, str(f.get("url") or ""))
                    for it in got:
                        svc.set_hint(it["item_id"], {"accession": f.get("accession"), "filing_date": f.get("filing_date"), "form": f.get("form")})
                    added += len(got)
                except FundamentalsError as error:
                    failed.append({"accession": f.get("accession"), "error": str(error)})
            return {"added": added, "failed": failed, "batch_id": batch}
        return self._job("fundamentals: fetch SEC filings", work, stoppable=True, lane="data")

    # ------------------------------------------------------------------------------------------------------------ working
    def import_process(self, body: dict) -> dict:
        """``POST /api/fund/import/process {batch_id} (a job)``: read every item still queued, then say what came of each. Safe to run again after an interruption."""
        svc, batch = self._importer(), str(body.get("batch_id") or "")
        self._guard(svc.batch, batch)
        return self._job("fundamentals: read reports", lambda progress, stop: svc.process(batch, progress, stop.is_set), stoppable=True)

    def import_review(self, body: dict) -> dict:
        """``POST /api/fund/import/review {item_id, edits}``: apply corrections to a copy of the draft and check again."""
        edits = body.get("edits")
        if not isinstance(edits, dict):
            raise ApiError("edits must be a dictionary")
        return self._guard(self._importer().review, self._int(body, "item_id"), edits)

    def import_approve(self, body: dict) -> dict:
        """``POST /api/fund/import/approve {item_id, security_id | company, decision?, restated_on?}``: store the item's periods, with their provenance, once every check passes."""
        sec = body.get("security") if isinstance(body.get("security"), dict) else {k: body[k] for k in ("security_id", "company") if body.get(k) not in (None, "")}
        out = self._guard(self._importer().approve, self._int(body, "item_id"), sec or None, body.get("decision") or None, body.get("restated_on") or None)
        self.bench("real").forget()
        return out

    def import_verify(self, body: dict) -> dict:
        """``POST /api/fund/import/verify {item_id, codes?}``: confirm stored figures that were read with low confidence (all of them unless ``codes`` names some); research then includes them."""
        codes = body.get("codes") or "all"
        if codes != "all" and not (isinstance(codes, list) and all(isinstance(c, str) for c in codes)):
            raise ApiError("codes is a list of line names, or leave it out for all")
        out = self._guard(self._importer().verify_saved, self._int(body, "item_id"), codes, str(body.get("note") or "")[:200])
        self.bench("real").forget()
        return out

    def import_reprocess(self, body: dict) -> dict:
        """``POST /api/fund/import/reprocess {item_id}``: read the stored original again with the current extractor and mappings; earlier readings are kept and stored figures are not touched."""
        return self._guard(self._importer().reprocess, self._int(body, "item_id"))

    def import_diagnostics(self, q: dict | None = None) -> dict:
        """``GET /api/fund/import/diagnostics``: extraction success, statement coverage, reconciliation results, fallback usage, rejection reasons, correction rates, and the last benchmark run."""
        from src.fundamentals.importer import diagnostics
        return self._guard(diagnostics.report, self.bench("real").store)

    def import_extractions(self, q: dict | None = None) -> dict:
        """``GET /api/fund/import/extractions?id=``: every reading of an item with the extractor and mapping versions that made it."""
        item = self._int(q or {}, "id")
        return {"item_id": item, "extractions": self._guard(self._importer().extractions, item)}

    def import_reject(self, body: dict) -> dict:
        return self._guard(self._importer().reject, self._int(body, "item_id"), str(body.get("reason") or ""))

    # ------------------------------------------------------------------------------------------------------------ coverage
    def coverage_overview(self, q: dict | None = None) -> dict:
        """``GET /api/fund/coverage/overview[?years=12]``: for every company, which fiscal years and quarters the database holds (and from which source), what is missing between the first and the latest, and the
        open problems: figures awaiting review, conflicts between sources, imports that failed."""
        q = q or {}
        try:
            years = max(3, min(int(q.get("years", 12)), 30))
        except (TypeError, ValueError):
            raise ApiError("years must be a whole number") from None
        return self._guard(overview_module.overview, self.bench("real").store, self._importer(), years)

    def coverage_backfill(self, body: dict) -> dict:
        """``POST /api/fund/coverage/backfill {security_ids?, source?, quarters?} (a job)``: fetch what is missing from SEC EDGAR (annual statements, and the 10-Q quarters when ``quarters`` is true), within the SEC's
        free allowance, for the companies named (all with gaps when none are)."""
        wb = self.bench("real")
        ids = body.get("security_ids")
        if ids is not None and (not isinstance(ids, list) or len(ids) > 500):
            raise ApiError("security_ids must be a list of up to 500 companies")
        periods = ("annual", "quarter") if body.get("quarters", True) else ("annual",)
        ids = [int(i) for i in ids] if ids else overview_module.companies_with_gaps(wb.store, periods)
        if not ids:
            return {"job": None, "message": "no company has gaps to fill"}

        def work(progress, stop):
            return wb.service.sync("sec_edgar", security_ids=ids, periods=periods, start_year=None, progress=progress, stop=stop)
        return self._job("fundamentals: backfill from SEC EDGAR", work, stoppable=True, lane="data")

    # ------------------------------------------------------------------------------------------------------------ country connectors
    def _connector(self, name):
        import os
        return self._guard(connector_registry.make, str(name or ""), self._importer().fetcher, os.environ)

    def connectors(self, q: dict | None = None) -> dict:
        """``GET /api/fund/connectors``: every country connector with what it permits (an API with a key, open data, or links only), whether it is usable now and why not, and each country's expected reporting periods."""
        import os
        return {"connectors": connector_registry.describe(None, os.environ), "calendar": {c: {"periods": list(v["periods"]), "standard": v["standard"], "currency": v["currency"], "note": v["note"]} for c, v in country_calendar.CALENDAR.items()}}

    def connectors_search(self, body: dict) -> dict:
        """``POST /api/fund/connectors/search {connector, company, since?, until?, kinds?, limit?} (a job)``: the company's filings at that source. A source that restricts automated collection answers with an explanation and no list."""
        conn = self._connector(body.get("connector"))
        company = body.get("company")
        if not isinstance(company, dict):
            raise ApiError("company must be a dictionary of identifiers (ticker, sec_code, corp_code, cd_cvm, lei...)")
        kinds = body.get("kinds")
        if kinds is not None and (not isinstance(kinds, list) or not all(isinstance(k, str) for k in kinds)):
            raise ApiError("kinds must be a list such as [\"FY\", \"Q1\"]")
        limit = self._limit(body)

        def work(progress, stop):
            progress(f"asking {conn.label}")
            out = conn.list_filings(company, body.get("since") or None, body.get("until") or None, kinds, limit)
            return {**out, "filings": [f.as_dict() for f in out["filings"]], "connector": conn.describe()}
        return self._job(f"fundamentals: list filings at {conn.label}", work, lane="data")

    def connectors_fetch(self, body: dict) -> dict:
        """``POST /api/fund/connectors/fetch {batch_id, connector, refs} (a job)``: get the chosen filings into a batch: structured drafts as they are, documents for the readers."""
        conn = self._connector(body.get("connector"))
        refs = body.get("refs")
        if not isinstance(refs, list) or not refs or len(refs) > 40:
            raise ApiError("refs must be a list of 1 to 40 filings from the filing list")
        svc, batch = self._importer(), str(body.get("batch_id") or "")
        self._guard(svc.batch, batch)
        known = set(FilingRef.__dataclass_fields__)
        built = []
        for r in refs:
            if not isinstance(r, dict):
                raise ApiError("each filing must be a dictionary")
            ref = FilingRef(**{k: v for k, v in r.items() if k in known})
            if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", str(ref.doc_id or "")):
                raise ApiError("a filing's doc_id may hold only letters, digits, underscore and hyphen")
            if ref.url and conn.hosts and not conn.on_host(ref.url):
                raise ApiError(f"a link for {conn.label} must be on its own site ({', '.join(conn.hosts)})")
            built.append(ref)

        def work(progress, stop):
            added, failed = 0, []
            for n, ref in enumerate(built):
                if stop is not None and stop.is_set():
                    break
                progress(f"getting filing {n + 1} of {len(built)}")
                try:
                    added += len(svc.add_from_connector(batch, conn, ref))
                except FundamentalsError as error:
                    failed.append({"doc_id": ref.doc_id, "error": str(error)})
            return {"added": added, "failed": failed, "batch_id": batch}
        return self._job(f"fundamentals: get filings from {conn.label}", work, stoppable=True, lane="data")

    # ------------------------------------------------------------------------------------------------------------ the point-in-time store
    def _pit_dir(self):
        return self.app.root / "data" / "user" / "pit"

    def pit_export(self, body: dict) -> dict:
        """``POST /api/fund/pit/export (a job)``: write the database's point-in-time table to Parquet (by country and fiscal year) with a manifest."""
        wb = self.bench("real")
        return self._job("fundamentals: export the point-in-time store", lambda progress, stop: self._guard(pitstore.export, wb.store, self._pit_dir()), lane="compute")

    def pit_asof(self, q: dict) -> dict:
        """``GET /api/fund/pit/asof?day=&security_id=&metrics=&countries=``: the figures that were public on a day (the latest vintage by then), from the exported store."""
        q = q or {}
        day = str(q.get("day") or "")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ApiError("day must be a date such as 2024-03-31")
        ids = [int(x) for x in str(q.get("security_id") or "").split(",") if x.strip().isdigit()] or None
        metrics = [m for m in str(q.get("metrics") or "").split(",") if m.strip()] or None
        countries = [c.strip().upper() for c in str(q.get("countries") or "").split(",") if c.strip()] or None
        store = self._guard(pitstore.PitStore, self._pit_dir())
        df = self._guard(store.as_of, day, ids, metrics, countries)
        return {"day": day, "rows": len(df), "manifest": store.manifest, "figures": df.head(5000).astype(object).where(df.head(5000).notna(), None).to_dict("records")}
