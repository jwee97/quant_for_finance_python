"""A Streamlit interface to the filing ingestion system: retrieval through the country connectors, uploads, review and approval, historical coverage, and the point-in-time store.

It is a thin layer over the same services as the dashboard's Fundamentals tab (``ImportService``, the connectors, ``coverage_overview``, ``pitstore``), so what is stored, checked and shown is the same
whichever interface is used; no rule lives here. Run it with ``streamlit run src/fundamentals/streamlit_app.py`` (``pip install streamlit``). The database is ``FUNDAMENTALS_DB`` (default
``data/user/fundamentals.db``) and the point-in-time files go to ``FUNDAMENTALS_PIT_DIR`` (default ``data/user/pit``). A source that needs a key says so and is not called without one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.fundamentals import canonical, pitstore  # noqa: E402
from src.fundamentals.errors import FundamentalsError  # noqa: E402
from src.fundamentals.filings import calendar as cal  # noqa: E402
from src.fundamentals.filings import registry  # noqa: E402
from src.fundamentals.importer import coverage_overview  # noqa: E402
from src.fundamentals.importer.fetch import Fetcher  # noqa: E402
from src.fundamentals.importer.pipeline import ImportService  # noqa: E402
from src.fundamentals.store import FundamentalsStore  # noqa: E402

ID_FIELD = {"edinet": "sec_code", "dart": "stock_code", "cvm": "cd_cvm", "esef": "lei", "sec_edgar": "ticker"}
STATUS = {"queued": "waiting", "extracted": "ready", "needs_review": "needs review", "saved": "saved", "duplicate": "already imported", "failed": "failed", "rejected": "rejected", "approved": "approved"}


def db_path() -> Path:
    return Path(os.environ.get("FUNDAMENTALS_DB") or ROOT / "data" / "user" / "fundamentals.db")


def pit_dir() -> Path:
    return Path(os.environ.get("FUNDAMENTALS_PIT_DIR") or ROOT / "data" / "user" / "pit")


@st.cache_resource
def services(path: str):
    store = FundamentalsStore(Path(path))
    ua = os.environ.get("SEC_EDGAR_USER_AGENT") or None
    return store, ImportService(store, Fetcher(user_agent=ua))


def failure(error: Exception) -> None:
    st.error(str(error))


def retrieve_tab(svc: ImportService) -> None:
    st.subheader("Retrieve from a country's source")
    infos = {c["name"]: c for c in registry.describe(svc.fetcher, os.environ)}
    name = st.selectbox("Source", sorted(infos), format_func=lambda n: f"{infos[n]['label']}" + ("" if infos[n]["available"] else " (needs a key)"), key="conn")
    c = infos[name]
    kind = {"api": "Official API", "open_data": "Open data", "links_only": "Links and files only"}[c["discovery"]]
    st.caption(f"{kind} · {', '.join(c['countries'])} · {', '.join(c['formats'])}. {c['note']}")
    if not c["available"]:
        st.warning(c["reason"])
    for country in c["countries"][:1]:
        rule = cal.expected(country)
        st.caption(f"{country}: expects {', '.join(rule['periods'])} · {rule['standard']} · {rule['currency']}. {rule['note']}")
    ident = st.text_input("Company identifier", key="ident", help="securities code (EDINET), six-digit stock code (OpenDART), CVM code or CNPJ, LEI (ESEF), ticker (SEC)")
    since, until = st.columns(2)
    d0 = since.text_input("From (YYYY-MM-DD)", key="since")
    d1 = until.text_input("To (YYYY-MM-DD)", key="until")
    if st.button("List its filings", key="list"):
        try:
            conn = registry.make(name, svc.fetcher)
            company = {ID_FIELD.get(name, "ticker"): ident}
            out = conn.list_filings(company, d0 or None, d1 or None, None, 80)
            st.session_state["found"] = {"name": name, "out": out}
        except FundamentalsError as error:
            failure(error)
    found = st.session_state.get("found")
    if found and found["name"] == name:
        out = found["out"]
        if out["note"]:
            st.info(out["note"])
        if out["filings"]:
            frame = pd.DataFrame([{"choose": False, "kind": f.fiscal_period, "period_end": f.period_end, "filed": (f.filed_at or "")[:10], "format": f.format, "report": f.title or f.doc_id, "doc_id": f.doc_id} for f in out["filings"]])
            edited = st.data_editor(frame, hide_index=True, key="filings", disabled=[c for c in frame.columns if c != "choose"])
            if st.button("Get the chosen filings", key="get"):
                batch = st.session_state.setdefault("batch", svc.create_batch("Streamlit " + name))
                conn = registry.make(name, svc.fetcher)
                added = 0
                for doc_id in edited.loc[edited["choose"], "doc_id"]:
                    ref = next(f for f in out["filings"] if f.doc_id == doc_id)
                    try:
                        added += len(svc.add_from_connector(batch, conn, ref))
                    except FundamentalsError as error:
                        failure(error)
                st.success(f"{added} added to batch {batch}: read and review them in the next tab.")


def review_tab(svc: ImportService) -> None:
    st.subheader("Uploads, reading and review")
    batch = st.session_state.setdefault("batch", svc.create_batch("Streamlit upload"))
    st.caption(f"Batch {batch}")
    files = st.file_uploader("Reports (XBRL / iXBRL, HTML, CSV, Excel, PDF, ZIP)", accept_multiple_files=True, key="upload")
    for f in files or []:
        key = f"sent-{f.name}-{f.size}"
        if key not in st.session_state:
            try:
                svc.add_file(batch, f.name, f.getvalue())
                st.session_state[key] = True
            except FundamentalsError as error:
                failure(error)
    urls = st.text_area("Or links to reports, one a line", key="urls")
    if st.button("Fetch the links", key="fetch") and urls.strip():
        for url in [u.strip() for u in urls.splitlines() if u.strip()]:
            try:
                svc.add_url(batch, url)
            except FundamentalsError as error:
                failure(error)
    if st.button("Read the queued reports", key="process"):
        st.json(svc.process(batch)["summary"])
    data = svc.batch(batch)
    items = [i for i in data["items"] if i["source_kind"] != "zip"]
    if not items:
        st.info("Nothing in this batch yet.")
        return
    st.dataframe(pd.DataFrame([{"id": i["item_id"], "file": i["source_name"], "kind": i["method"] or i["source_kind"], "status": STATUS.get(i["status"], i["status"]), "note": (i["error"] or "")[:100]} for i in items]), hide_index=True)
    open_ids = [i["item_id"] for i in items if i["status"] in ("needs_review", "extracted", "approved")]
    if not open_ids:
        return
    chosen = st.selectbox("Review", open_ids, format_func=lambda i: next(x["source_name"] for x in items if x["item_id"] == i), key="item")
    item = svc.item(chosen, full=True)
    draft = item["reviewed"] or item["draft"]
    st.caption(f"{draft.get('method')} · standard {draft.get('accounting_standard') or 'not stated'} · currency {draft.get('currency') or 'not stated'}")
    for n in draft.get("notes", []):
        st.caption(n)
    codes = sorted({c for p in draft["periods"] for c in p["values"]})
    table = pd.DataFrame({f"{p['fiscal_period']} {p['period_end']}": [p["values"].get(c) for c in codes] for p in draft["periods"]}, index=codes)
    edited = st.data_editor(table, key=f"grid-{chosen}")
    units = st.selectbox("Amounts are in", ["units", "thousands", "millions", "billions"], index=["units", "thousands", "millions", "billions"].index((draft.get("units") or {}).get("money", "units")), key=f"units-{chosen}")
    if st.button("Save corrections and check again", key=f"fix-{chosen}"):
        edits = {"units": {"money": units}, "periods": [{"values": {c: (None if pd.isna(edited.iloc[r, j]) else float(edited.iloc[r, j])) for r, c in enumerate(codes)}} for j in range(len(draft["periods"]))]}
        try:
            svc.review(chosen, edits)
            st.rerun()
        except FundamentalsError as error:
            failure(error)
    for i in item["issues"] or []:
        {"error": st.error, "warning": st.warning}.get(i["level"], st.info)(f"{i['period'] + ': ' if i['period'] else ''}{i['message']}")
    conflicts = draft.get("conflicts") or []
    decision = None
    if conflicts:
        st.dataframe(pd.DataFrame(conflicts))
        decision = st.selectbox("These differ from held figures: store them as", ["restate", "correct", "skip"], key=f"decision-{chosen}")
    store = svc.store
    secs = store.securities()
    company = st.selectbox("Company", [None] + list(secs["security_id"]), format_func=lambda i: "choose" if i is None else f"{secs[secs['security_id'] == i].iloc[0]['ticker']}", key=f"company-{chosen}")
    left, right = st.columns(2)
    if left.button("Approve and store", key=f"ok-{chosen}", type="primary", disabled=company is None):
        try:
            out = svc.approve(chosen, {"security_id": int(company)}, decision)
            st.success(f"Stored {len(out['saved'])} period(s) with their sources.")
        except FundamentalsError as error:
            failure(error)
    if right.button("Reject", key=f"no-{chosen}"):
        svc.reject(chosen, "rejected in the Streamlit review")
        st.rerun()


def coverage_tab(store: FundamentalsStore, svc: ImportService) -> None:
    st.subheader("What the database holds")
    data = coverage_overview.overview(store, svc, 12)
    t, p = data["totals"], data["problems"]
    cols = st.columns(5)
    cols[0].metric("Years held", t.get("annual_held", 0), f"-{t.get('annual_missing', 0)} missing")
    cols[1].metric("Quarters and half-years held", t.get("quarter_held", 0), f"-{t.get('quarter_missing', 0)} missing")
    cols[2].metric("Awaiting review", p["awaiting_review"])
    cols[3].metric("Conflicts", p["import_conflicts"])
    cols[4].metric("Dates assumed", p["estimated_availability"])
    if not data["companies"]:
        st.info("No company has statements yet.")
        return
    st.dataframe(pd.DataFrame([{"ticker": c["ticker"], "name": c["name"], "country": c["country"], "expects": ", ".join(c["expected"]), "years held": c["annual_held"], "years missing": c["annual_missing"],
                                "quarters held": c["quarter_held"], "quarters missing": c["quarter_missing"], "complete": round(c["completeness"], 2), "sources": ", ".join(c["sources"])} for c in data["companies"]]), hide_index=True)
    view = st.radio("Grid", ["Annual", "Quarterly and half-year"], horizontal=True, key="grid")
    key = "annual" if view == "Annual" else "quarters"
    cells = {c["ticker"] or str(c["security_id"]): {k: v["state"] for k, v in c[key].items()} for c in data["companies"][:60]}
    frame = pd.DataFrame(cells).T.fillna("")
    colours = {"held": "#2a6fdb", "derived": "#7aa5e8", "missing": "#d9534f", "na": "#555555", "notyet": "#333333"}
    st.dataframe(frame.style.map(lambda v: f"background-color: {colours.get(v, '')}; color: white" if v else ""), use_container_width=True)
    st.caption("blue held · light blue derived (Q4 = year less nine months) · red missing · grey not expected or not yet due")


def pit_tab(store: FundamentalsStore) -> None:
    st.subheader("Point-in-time store (Parquet and DuckDB)")
    if st.button("Export the store", key="export"):
        try:
            m = pitstore.export(store, pit_dir())
            st.success(f"{m['rows']} rows, {m['securities']} companies, written to {pit_dir()}")
        except FundamentalsError as error:
            failure(error)
    try:
        pit = pitstore.PitStore(pit_dir())
    except FundamentalsError as error:
        st.info(str(error))
        return
    st.caption(f"Exported {pit.manifest['exported_at']} · {pit.manifest['rows']} rows")
    day = st.text_input("What was public on (YYYY-MM-DD)", key="day")
    metrics = st.multiselect("Lines", sorted(canonical.METRICS), default=["revenue", "net_income", "total_assets"], key="metrics")
    if day:
        st.dataframe(pit.as_of(day, metrics=metrics or None))
    st.dataframe(pit.summary())


def main() -> None:
    st.set_page_config(page_title="Filings", layout="wide")
    st.title("Financial filings: retrieval, review, coverage")
    with st.sidebar:
        st.caption(f"Database: {db_path()}")
        for c in registry.describe(None, os.environ):
            st.write(("✓ " if c["available"] else "✗ ") + c["label"])
    store, svc = services(str(db_path()))
    tabs = st.tabs(["Retrieve", "Upload and review", "Coverage", "Point in time"])
    with tabs[0]:
        retrieve_tab(svc)
    with tabs[1]:
        review_tab(svc)
    with tabs[2]:
        coverage_tab(store, svc)
    with tabs[3]:
        pit_tab(store)


if __name__ == "__main__" or os.environ.get("STREAMLIT_APPTEST"):
    main()
