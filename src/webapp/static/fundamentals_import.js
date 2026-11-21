/* The Fundamentals tab's "Report import" and "Data coverage" views.

   Report import: files (XBRL / inline-XBRL filings, CSV, Excel, PDF, ZIP archives), links and the SEC's own filing list are added to a batch, read into drafts, and every draft is reviewed before it is
   stored: the figures can be corrected, the filing date and unit set, conflicts with held figures decided. Nothing is stored without an approval, and each stored figure keeps its source. It talks to
   /api/fund/import/*. Data coverage: which fiscal years and quarters the database holds for each company, where they came from, what is missing, and what is waiting for review. /api/fund/coverage/*.
   Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const FD = window.FD, Q = window.QL, W = window.WS, C = Q.C;
  if (!FD || !Q || !W) return;
  const h = Q.h, num = Q.num, pct = Q.pct, tile = Q.tile, table = Q.simpleTable, api = Q.api, css = Q.css;
  const { human, day } = FD.fmt;
  const IM = FD.views.import = {}, CV = FD.views.coverage = {};
  FD.modes.splice(3, 0, ["import", "Report import"], ["coverage", "Data coverage"]);
  FD.runLabel.import = "Read the reports"; FD.runLabel.coverage = "Refresh";
  FD.blurb.import = "Add financial reports (a filing, a spreadsheet, a PDF, a ZIP of them, a link, or the SEC's filing list), check what was read from each, correct it, and store it with its source.";
  FD.blurb.coverage = "Which years and quarters of each company the database holds, where each came from, what is missing and what is waiting for review: and a button to fill the gaps.";
  const CHUNK = 120_000;
  const STATUS_WORD = { queued: "Waiting to be read", extracted: "Read: ready to approve", needs_review: "Needs your review", approved: "Approved", saved: "Saved", rejected: "Rejected", failed: "Could not be read", duplicate: "Already imported" };
  const STATUS_TONE = { saved: "tone-good", extracted: "tone-good", needs_review: "tone-warn", failed: "tone-crit", rejected: "hint", duplicate: "hint", queued: "hint", approved: "tone-good" };
  const METHOD_WORD = { xbrl: "XBRL filing", ixbrl: "Inline XBRL filing", csv: "CSV table", xlsx: "Excel workbook", "pdf-text": "PDF text" };
  const LEVEL = { error: ["✗", "tone-crit"], warning: ["!", "tone-warn"], info: ["i", "hint"] };
  const UNITS = [["units", "single units"], ["thousands", "thousands"], ["millions", "millions"], ["billions", "billions"]];

  IM.s = { connectors: null, conn: null, connId: "", foundConn: null, pickedConn: new Set(), batch: null, ticker: "", sid: null, label: "", urls: "", found: null, filings: null, picked: new Set(), item: null, itemData: null, batchData: null, history: null, notice: "", progress: "", busyUpload: false, decision: "", restated_on: "" };
  FD.views.resetImport = () => { IM.s.batch = null; IM.s.item = null; IM.s.itemData = null; IM.s.batchData = null; };
  const s = IM.s;

  /* ---------- sending ---------- */
  const b64 = (bytes) => { let bin = ""; for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000)); return btoa(bin); };
  async function ensureBatch() {
    if (s.batch) return s.batch;
    const r = await FD.send("/api/fund/import/create", { label: s.label || (s.ticker ? `${s.ticker} reports` : "Reports"), ticker: s.ticker || undefined, security_id: s.sid || undefined });
    s.batch = r.batch_id; return s.batch;
  }
  async function sendFile(file, onProgress) {
    const batch = await ensureBatch(), bytes = new Uint8Array(await file.arrayBuffer()), total = Math.max(1, Math.ceil(bytes.length / CHUNK)), id = Math.random().toString(36).slice(2, 12);
    let last = null;
    for (let i = 0; i < total; i++) {
      last = await FD.send("/api/fund/import/chunk", { batch_id: batch, upload_id: id, name: file.name, index: i, total, data: b64(bytes.subarray(i * CHUNK, (i + 1) * CHUNK)) });
      if (onProgress) onProgress(i + 1, total);
    }
    return last;
  }
  async function addFiles(files, note) {
    if (!files.length) return;
    s.busyUpload = true; s.notice = "";
    try {
      let added = 0, dup = 0;
      for (let n = 0; n < files.length; n++) {
        const f = files[n]; s.progress = `Sending ${f.name} (${n + 1} of ${files.length})…`; paintNotice();
        const r = await sendFile(f);
        for (const it of (r.items || [])) { added++; if (it.status === "duplicate") dup++; }
      }
      s.notice = `${added} file${added === 1 ? "" : "s"} added${dup ? `, ${dup} already imported before` : ""}. Press “Read the reports”.`;
    } catch (e) { s.notice = e.message; }
    s.busyUpload = false; s.progress = ""; await refreshBatch(); Q.renderLab(FD);
  }
  const paintNotice = () => { const el = Q.$('[data-role="import-notice"]'); if (el) el.textContent = s.progress || s.notice; };
  async function refreshBatch() {
    if (s.batch) { try { s.batchData = await FD.read("/api/fund/import/batch", { id: s.batch }); } catch (e) { s.batchData = null; } }
    try { s.history = (await FD.read("/api/fund/import/history")).batches; } catch (e) { s.history = null; }
  }

  /* ---------- the panel ---------- */
  IM.check = () => (FD.db !== "real" ? "Report import is for real data: switch to Real data (the demo database is simulated)." : "");
  IM.sections = (panel) => {
    const a = FD.sec(panel, "1. The company");
    const row = h("div", "row tight", a), inp = h("input", "", row); inp.placeholder = "ticker, e.g. AAPL"; inp.value = s.ticker; inp.dataset.key = "import-ticker"; inp.setAttribute("aria-label", "Ticker");
    inp.addEventListener("input", () => { s.ticker = inp.value.trim().toUpperCase(); s.sid = null; });
    const find = h("button", "btn small", row, "Find in my database"); find.type = "button"; find.dataset.action = "import-find";
    const out = h("p", "hint", a, s.sid ? `Importing for ${s.ticker} (company ${s.sid}).` : "Leave empty if the reports name the company: it is created when you approve."); out.dataset.role = "import-company";
    find.addEventListener("click", async () => {
      try {
        const r = await FD.read("/api/fund/companies", { q: s.ticker, limit: 5 }), hit = (r.companies || []).find((c) => c.ticker === s.ticker) || (r.companies || [])[0];
        if (hit) { s.sid = hit.security_id; s.ticker = hit.ticker; inp.value = hit.ticker; out.textContent = `${hit.company_name || hit.ticker} (${hit.ticker}, company ${hit.security_id}): ${hit.annual} annual and ${hit.quarterly} quarterly periods held.`; }
        else { s.sid = null; out.textContent = `${s.ticker || "That"} is not in the database yet: it will be created when you approve the first report.`; }
      } catch (e) { out.textContent = e.message; }
    });
    const b = FD.sec(panel, "2. Add reports");
    const drop = h("div", "dropzone", b); drop.dataset.role = "import-drop"; drop.tabIndex = 0;
    h("p", "", drop, "Drop files here, or choose them: XBRL / inline-XBRL filings (.htm .xml), CSV, Excel (.xlsx), PDF, or a ZIP of them. Several at once.");
    const file = h("input", "", drop); file.type = "file"; file.multiple = true; file.accept = ".pdf,.csv,.tsv,.txt,.xlsx,.xlsm,.xml,.xbrl,.htm,.html,.xhtml,.zip"; file.dataset.role = "import-file"; file.setAttribute("aria-label", "Choose report files");
    file.addEventListener("change", () => { addFiles([...file.files]); file.value = ""; });
    for (const ev of ["dragover", "dragenter"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
    for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
    drop.addEventListener("drop", (e) => addFiles([...(e.dataTransfer ? e.dataTransfer.files : [])]));
    const url = h("textarea", "", h("label", "f", b, "Or links to reports, one a line")); url.rows = 3; url.value = s.urls; url.spellcheck = false; url.dataset.role = "import-urls"; url.placeholder = "https://www.example.com/investors/annual-report-2023.pdf";
    url.addEventListener("input", () => { s.urls = url.value; });
    const go = h("button", "btn small", b, "Fetch the links"); go.type = "button"; go.dataset.action = "import-fetch";
    go.addEventListener("click", () => FD.act("import", "Fetching…", async (run) => {
      const urls = s.urls.split(/\n+/).map((x) => x.trim()).filter(Boolean); if (!urls.length) throw new Error("Write at least one link");
      const batch = await ensureBatch(), { result } = await run.job("/api/fund/import/urls", { batch_id: batch, urls }, "Fetching the reports");
      s.notice = `${result.added} fetched${result.failed.length ? `; ${result.failed.length} failed: ${result.failed.map((f) => f.error).slice(0, 2).join("; ")}` : ""}.`; s.urls = "";
      await refreshBatch(); return { kind: "batch" };
    }));
    h("p", "hint", b, "Only public web addresses are fetched. The SEC needs a contact in the User-Agent (SEC_EDGAR_USER_AGENT), as for its data.");
    const c = FD.sec(panel, "3. Or the SEC's own filings", false);
    const sec = h("button", "btn small", c, "List 10-K and 10-Q filings"); sec.type = "button"; sec.dataset.action = "import-filings";
    sec.addEventListener("click", async () => {
      try { s.found = await FD.send("/api/fund/import/filings", { symbol: s.ticker || undefined, security_id: s.sid || undefined, limit: 60 }); s.picked = new Set(); Q.renderLab(FD); } catch (e) { Q.toast(e.message); }
    });
    h("p", "hint", c, "Each filing's own XBRL is the most exact source there is. Filings before a company's first XBRL one (2009 to 2011) have no tagged numbers.");
    const d = FD.sec(panel, "4. A country's regulator", false), pick = h("select", "", h("label", "f", d, "Source")), what = h("p", "hint", d, "Loading the sources…"), idRow = h("input", "", h("label", "f", d, "Company identifier"));
    pick.dataset.role = "import-connector"; pick.setAttribute("aria-label", "source"); idRow.dataset.role = "import-connector-id"; idRow.value = s.connId || "";
    idRow.addEventListener("input", () => { s.connId = idRow.value.trim(); });
    const since = h("input", "", h("label", "f", d, "From")), until = h("input", "", h("label", "f", d, "To")); since.type = until.type = "date"; since.dataset.role = "import-since"; since.setAttribute("aria-label", "from"); until.setAttribute("aria-label", "to");
    const ID_HELP = { edinet: "securities code, e.g. 7203", dart: "six-digit stock code, e.g. 005930", cvm: "CVM code or CNPJ", esef: "LEI (20 characters)", sec_edgar: "ticker, e.g. AAPL" };
    const paint = () => {
      const c = (s.connectors || []).find((x) => x.name === pick.value); if (!c) return;
      s.conn = c.name; idRow.placeholder = ID_HELP[c.name] || "(not used: links and files only)";
      what.textContent = `${c.discovery === "api" ? "Official API" : c.discovery === "open_data" ? "Open data" : "Links and files only"} · ${c.countries.join(", ")} · ${c.formats.join(", ")}. ${c.available ? "" : c.reason + ". "}${c.note}`;
    };
    FD.read("/api/fund/connectors").then((r) => {
      s.connectors = r.connectors; pick.replaceChildren();
      for (const c of r.connectors) { const o = h("option", "", pick, `${c.label}${c.available ? "" : " (needs a key)"}`); o.value = c.name; if (c.name === s.conn) o.selected = true; }
      paint();
    }).catch((e) => { what.textContent = e.message; });
    pick.addEventListener("change", paint);
    const list = h("button", "btn small", d, "List its filings"); list.type = "button"; list.dataset.action = "import-connector-list";
    list.addEventListener("click", () => FD.act("import", "Asking the source…", async (run) => {
      const c = (s.connectors || []).find((x) => x.name === pick.value); if (!c) throw new Error("Choose a source");
      const idf = { edinet: "sec_code", dart: "stock_code", cvm: "cd_cvm", esef: "lei", sec_edgar: "ticker" }[c.name];
      const company = idf ? { [idf]: s.connId || s.ticker } : { ticker: s.connId || s.ticker };
      const { result } = await run.job("/api/fund/connectors/search", { connector: c.name, company, since: since.value || undefined, until: until.value || undefined, limit: 80 }, "Listing the filings");
      s.foundConn = result; s.pickedConn = new Set(); return { kind: "batch" };
    }));
    const msg = h("p", "hint", panel, s.progress || s.notice); msg.dataset.role = "import-notice"; msg.setAttribute("role", "status"); msg.style.padding = "0 16px";
  };

  /* ---------- the page ---------- */
  IM.auto = () => { if (!s.history) refreshBatch().then(() => { if (FD.mode === "import" && !FD.busy) Q.renderLab(FD); }); };
  IM.run = () => FD.act("import", "Reading the reports…", async (run) => {
    if (!s.batch) throw new Error("Add at least one report first");
    const { result } = await run.job("/api/fund/import/process", { batch_id: s.batch }, "Reading the reports");
    await refreshBatch(); s.notice = `${result.processed} read.`; return { kind: "batch", processed: result.processed };
  });
  IM.empty = (root) => {
    IM.build(root, { kind: "batch" });
  };
  IM.build = (root) => {
    const d = s.batchData;
    Q.labHead(root, "Report import", d ? `batch ${d.batch_id}${d.label ? " · " + d.label : ""}` : "no batch yet");
    if (s.found) filingsCard(root);
    if (s.foundConn) connectorCard(root);
    if (!d || !d.items.length) {
      const c = h("div", "card empty", root); h("h2", "", c, "Bring a company's reports into the database");
      for (const t of ["Add files or links on the left. Each report is read into a draft: a filing's own XBRL exactly as tagged, a table from the columns and rows, a PDF from the words on its pages (the least reliable: it always asks you to check it).",
        "You look at what was read, correct it if it is wrong, and approve it. Only then is it stored, with the file, the filing number, the page and the tag it came from, so any figure can be traced back. A report with different figures from one already held is never written over it: you choose to store it as a restatement, to correct the old one, or to skip it.",
        "Prefer the company's XBRL (every US-listed company files it) over its PDF: nothing has to be guessed."]) h("p", "", c, t);
      histCard(root); return;
    }
    const k = h("div", "earnrow", root);
    for (const key of ["queued", "needs_review", "extracted", "saved", "duplicate", "failed"]) tile(k, STATUS_WORD[key], String(d.counts[key] || 0), {}, d.counts[key] ? "" : "muted");
    const card = W.card(root, "Reports in this batch", "Open one to see what was read, correct it and approve it."); card.dataset.role = "import-items";
    table(card, [{ label: "File", get: (r) => r.source_name }, { label: "Kind", get: (r) => METHOD_WORD[r.method] || human(r.source_kind) }, { label: "Status", get: (r) => STATUS_WORD[r.status] || r.status },
      { label: "Note", get: (r) => (r.error || "").slice(0, 120) }, { label: "", get: (r) => (["needs_review", "extracted", "approved"].includes(r.status) ? "Review →" : "") }], d.items.filter((r) => r.source_kind !== "zip"));
    for (const [i, tr] of [...card.querySelectorAll("tbody tr")].entries()) {
      const it = d.items.filter((r) => r.source_kind !== "zip")[i]; tr.dataset.item = it.item_id; tr.dataset.status = it.status;
      if (["needs_review", "extracted", "approved", "saved"].includes(it.status)) { tr.style.cursor = "pointer"; tr.tabIndex = 0; const open = () => openItem(it.item_id); tr.addEventListener("click", open); tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); }); }
    }
    if (s.itemData) reviewCard(root);
    histCard(root);
  };
  async function openItem(id) { try { s.item = id; s.itemData = await FD.read("/api/fund/import/item", { id }); } catch (e) { Q.toast(e.message); } Q.renderLab(FD); }

  /* ---------- a country source's filing list ---------- */
  function connectorCard(root) {
    const f = s.foundConn, c = f.connector, card = W.card(root, `${c.label}: filings`, f.note || ""); card.dataset.role = "import-connector-filings";
    if (!f.filings.length) { h("p", "hint", card, c.discovery === "links_only" ? "Nothing is listed: this source has no interface for programs. Download the reports and upload them, or paste their links on the left." : "No filing matched."); return; }
    const rows = f.filings;
    table(card, [{ label: "", get: () => "" }, { label: "Kind", get: (r) => r.fiscal_period || "" }, { label: "Period end", get: (r) => r.period_end || "" }, { label: "Filed", get: (r) => (r.filed_at || "").slice(0, 10) }, { label: "Format", get: (r) => r.format },
      { label: "Report", get: (r) => (r.title || r.doc_id) + (r.correction ? " (correction)" : "") }], rows);
    [...card.querySelectorAll("tbody tr")].forEach((tr, i) => { const r = rows[i], cb = h("input", "", tr.firstElementChild); cb.type = "checkbox"; cb.checked = s.pickedConn.has(r.doc_id); cb.setAttribute("aria-label", `choose ${r.title || r.doc_id}`);
      cb.addEventListener("change", () => { if (cb.checked) s.pickedConn.add(r.doc_id); else s.pickedConn.delete(r.doc_id); }); });
    if (f.next_since) h("p", "hint", card, `The source lists by day: the scan stopped at ${f.next_since}. Set “From” to that date and list again to carry on.`);
    const row = h("div", "row tight", card), go = h("button", "btn primary small", row, "Get the chosen filings"); go.type = "button"; go.dataset.action = "import-connector-fetch";
    go.addEventListener("click", () => FD.act("import", "Getting the filings…", async (run) => {
      const chosen = rows.filter((r) => s.pickedConn.has(r.doc_id)); if (!chosen.length) throw new Error("Choose at least one filing");
      const batch = await ensureBatch(), { result } = await run.job("/api/fund/connectors/fetch", { batch_id: batch, connector: c.name, refs: chosen }, "Getting the filings");
      s.notice = `${result.added} added${result.failed.length ? `; ${result.failed.length} failed: ${result.failed.map((x) => x.error).slice(0, 2).join("; ")}` : ""}.`; s.pickedConn = new Set(); await refreshBatch(); return { kind: "batch" };
    }));
  }

  /* ---------- the SEC's filing list ---------- */
  function filingsCard(root) {
    const f = s.found, card = W.card(root, `${f.name || f.ticker}: SEC filings`, f.note); card.dataset.role = "import-filings";
    const rows = f.filings;
    table(card, [{ label: "", get: () => "" }, { label: "Form", get: (r) => r.form }, { label: "Filed", get: (r) => r.filing_date }, { label: "Period", get: (r) => r.period || "" }, { label: "XBRL", get: (r) => (r.xbrl ? "yes" : "no: no tagged numbers") }, { label: "Accession", get: (r) => r.accession }], rows);
    [...card.querySelectorAll("tbody tr")].forEach((tr, i) => {
      const r = rows[i], cb = h("input", "", tr.firstElementChild); cb.type = "checkbox"; cb.setAttribute("aria-label", `choose ${r.form} filed ${r.filing_date}`); cb.disabled = !r.xbrl || !r.url; cb.checked = s.picked.has(r.accession);
      cb.addEventListener("change", () => { if (cb.checked) s.picked.add(r.accession); else s.picked.delete(r.accession); });
    });
    const row = h("div", "row tight", card);
    const all = h("button", "btn small", row, "Choose every XBRL filing"); all.type = "button"; all.addEventListener("click", () => { for (const r of rows) if (r.xbrl && r.url) s.picked.add(r.accession); Q.renderLab(FD); });
    const go = h("button", "btn primary small", row, "Fetch the chosen filings"); go.type = "button"; go.dataset.action = "import-fetch-filings";
    go.addEventListener("click", () => FD.act("import", "Fetching filings…", async (run) => {
      const chosen = rows.filter((r) => s.picked.has(r.accession)); if (!chosen.length) throw new Error("Choose at least one filing");
      const batch = await ensureBatch(), { result } = await run.job("/api/fund/import/filings/add", { batch_id: batch, filings: chosen.map((r) => ({ url: r.url, accession: r.accession, filing_date: r.filing_date, form: r.form })) }, "Fetching the filings");
      s.notice = `${result.added} fetched${result.failed.length ? `; ${result.failed.length} failed` : ""}. Press “Read the reports”.`; s.picked = new Set(); await refreshBatch(); return { kind: "batch" };
    }));
  }

  /* ---------- reviewing one report ---------- */
  function reviewCard(root) {
    const it = s.itemData, draft = it.reviewed || it.draft; if (!draft) return;
    const card = W.card(root, `Review: ${it.source_name}`, `${METHOD_WORD[draft.method] || draft.method}${draft.accession ? " · filing " + draft.accession : ""}${draft.document_type ? " · " + draft.document_type : ""}. ${it.reviewed ? "Showing your corrections; the first reading is kept." : ""}`);
    card.dataset.role = "import-review"; card.dataset.item = it.item_id;
    for (const n of draft.notes || []) h("p", "hint", card, n);
    const bar = h("div", "row tight", card);
    const un = h("select", "", h("label", "f", bar, "Amounts are in")); un.dataset.role = "import-units"; un.setAttribute("aria-label", "unit of the amounts");
    for (const [v, l] of UNITS) { const o = h("option", "", un, l); o.value = v; if (v === (draft.units || {}).money) o.selected = true; }
    const edits = { periods: draft.periods.map(() => ({ values: {} })) };
    const unitEdit = {};
    un.addEventListener("change", () => { unitEdit.money = un.value; });
    const wrap = h("div", "scroll", card), t = h("table", "data review", wrap), hd = h("tr", "", h("thead", "", t));
    h("th", "", hd, "Line");
    draft.periods.forEach((p, j) => {
      const th = h("th", "num", hd), sel = h("select", "", th); sel.setAttribute("aria-label", `fiscal period of column ${j + 1}`);
      for (const v of ["FY", "Q1", "Q2", "Q3", "Q4", "H1", "H2"]) { const o = h("option", "", sel, v); o.value = v; if (v === p.fiscal_period) o.selected = true; }
      sel.addEventListener("change", () => { edits.periods[j].fiscal_period = sel.value; });
      const pe = h("input", "", th); pe.type = "date"; pe.value = p.period_end || ""; pe.setAttribute("aria-label", `period end of column ${j + 1}`); pe.addEventListener("change", () => { edits.periods[j].period_end = pe.value; });
      const fd = h("input", "", th); fd.type = "date"; fd.value = p.filing_date || ""; fd.title = "The day the report became public: figures are used only from this day"; fd.setAttribute("aria-label", `filing date of column ${j + 1}`);
      fd.dataset.role = "import-filing-date"; fd.addEventListener("change", () => { edits.periods[j].filing_date = fd.value; });
    });
    const tb = h("tbody", "", t), codes = [...new Set(draft.periods.flatMap((p) => Object.keys(p.values)))];
    for (const code of codes) {
      const tr = h("tr", "", tb); h("td", "", tr, human(code));
      draft.periods.forEach((p, j) => {
        const td = h("td", "num", tr), v = p.values[code], src = (p.sources || {})[code] || {};
        const inp = h("input", "", td); inp.type = "text"; inp.inputMode = "decimal"; inp.value = v === undefined ? "" : String(v); inp.dataset.code = code; inp.dataset.col = j; inp.setAttribute("aria-label", `${human(code)}, column ${j + 1}`);
        inp.title = [src.tag && `tag ${src.tag}`, src.page && `page ${src.page}`, src.text && `“${src.text}”`, src.edited && "edited by you"].filter(Boolean).join(" · ") || "typed";
        if (src.edited) inp.classList.add("edited");
        inp.addEventListener("input", () => { edits.periods[j].values[code] = inp.value.trim() === "" ? null : inp.value.trim(); inp.classList.add("edited"); });
      });
    }
    const save = h("button", "btn small", bar, "Save corrections and check again"); save.type = "button"; save.dataset.action = "import-correct";
    save.addEventListener("click", async () => {
      try {
        if (unitEdit.money) edits.units = { money: unitEdit.money, shares: (draft.units || {}).shares || "units" };
        s.itemData = await FD.send("/api/fund/import/review", { item_id: it.item_id, edits }); await refreshBatch(); Q.renderLab(FD);
      } catch (e) { Q.toast(e.message); }
    });
    issuesBlock(card, s.itemData.issues || []);
    conflictsBlock(card, draft.conflicts || []);
    const act = h("div", "row tight", card);
    let decision = null;
    if ((draft.conflicts || []).length) {
      const sel = h("select", "", h("label", "f", act, "These differ from held figures: store them as")); sel.dataset.role = "import-decision"; sel.setAttribute("aria-label", "what to do with figures that differ from held ones");
      for (const [v, l] of [["", "(choose)"], ["restate", "a restatement: a new vintage, public from its filing date"], ["correct", "a correction: replace the held figures (they were a mistake)"], ["skip", "skip the periods that differ"]]) { const o = h("option", "", sel, l); o.value = v; }
      sel.addEventListener("change", () => { decision = sel.value || null; });
    }
    const ok = h("button", "btn primary", act, "Approve and store"); ok.type = "button"; ok.dataset.action = "import-approve"; ok.disabled = it.status === "saved";
    ok.addEventListener("click", async () => {
      try {
        const r = await FD.send("/api/fund/import/approve", { item_id: it.item_id, security_id: s.sid || (s.batchData && s.batchData.security_id) || undefined, decision: decision || undefined });
        s.notice = `Stored ${r.saved.length} period${r.saved.length === 1 ? "" : "s"} (${r.saved.reduce((a, x) => a + x.lines, 0)} figures), each with its source.`; s.itemData = null; await refreshBatch(); Q.toast(s.notice); Q.renderLab(FD);
      } catch (e) { Q.toast(e.message); }
    });
    const rej = h("button", "btn", act, "Reject"); rej.type = "button"; rej.dataset.action = "import-reject"; rej.disabled = it.status === "saved";
    rej.addEventListener("click", async () => { try { await FD.send("/api/fund/import/reject", { item_id: it.item_id, reason: "rejected after review" }); s.itemData = null; await refreshBatch(); Q.renderLab(FD); } catch (e) { Q.toast(e.message); } });
    if (it.status === "saved") provenanceBlock(card);
  }
  function issuesBlock(card, issues) {
    if (!issues.length) { h("p", "hint", card, "No problems found."); return; }
    const box = h("div", "issues", card); box.dataset.role = "import-issues";
    for (const i of issues) { const [mark, tone] = LEVEL[i.level] || LEVEL.info, p = h("p", tone, box); p.textContent = `${mark} ${i.period ? i.period + ": " : ""}${i.message}`; p.dataset.level = i.level; }
  }
  function conflictsBlock(card, conflicts) {
    if (!conflicts.length) return;
    h("h4", "", card, "Different from what the database holds");
    table(card, [{ label: "Period", get: (c) => c.period }, { label: "Line", get: (c) => human(c.metric) }, { label: "This report", num: true, get: (c) => num(c.imported, 0) }, { label: "Held", num: true, get: (c) => num(c.stored, 0) },
      { label: "Difference", num: true, get: (c) => (c.difference === null ? "" : pct(c.difference, 1)) }, { label: "Held from", get: (c) => `${c.stored_provider}${c.stored_filing_date ? ", filed " + c.stored_filing_date : ""}` }, { label: "Kind", get: (c) => (c.kind === "restatement" ? "later filing: a restatement" : "another source") }], conflicts.slice(0, 60));
  }
  async function provenanceBlock(card) {
    const sid = s.sid || (s.batchData && s.batchData.security_id); if (!sid) return;
    try {
      const r = await FD.read("/api/fund/import/provenance", { security_id: sid }), mine = r.rows.filter((x) => x.content_hash === s.itemData.content_hash);
      if (!mine.length) return;
      h("h4", "", card, "Where each stored figure came from");
      table(card, [{ label: "Period", get: (x) => `${x.fiscal_period} ${x.period_end.slice(0, 10)}` }, { label: "Line", get: (x) => human(x.metric_code) }, { label: "As reported", num: true, get: (x) => num(x.reported_value, 0) }, { label: "Unit", get: (x) => x.reported_unit || "" },
        { label: "Tag", get: (x) => x.taxonomy_tag || "" }, { label: "Page", get: (x) => (x.page ? String(x.page) : "") }, { label: "Filing", get: (x) => x.accession || "" }, { label: "How", get: (x) => `${x.extraction_method || ""}${x.edited ? " · edited" : ""}` }], mine.slice(0, 80));
    } catch (e) { /* the figures are stored; the list is a convenience */ }
  }
  function histCard(root) {
    if (!s.history || !s.history.length) return;
    const card = W.card(root, "Import history", "Every batch, newest first, and what became of its reports."); card.dataset.role = "import-history";
    table(card, [{ label: "Batch", get: (b) => b.label || b.batch_id }, { label: "Company", get: (b) => b.ticker || "" }, { label: "Started", get: (b) => day(b.created_at) }, { label: "Reports", num: true, get: (b) => String(b.items) },
      { label: "Saved", num: true, get: (b) => String(b.counts.saved || 0) }, { label: "To review", num: true, get: (b) => String((b.counts.needs_review || 0) + (b.counts.extracted || 0)) }, { label: "Failed", num: true, get: (b) => String(b.counts.failed || 0) },
      { label: "", get: (b) => (b.batch_id === s.batch ? "open" : "Open →") }], s.history);
    [...card.querySelectorAll("tbody tr")].forEach((tr, i) => { const b = s.history[i]; tr.style.cursor = "pointer"; tr.tabIndex = 0; const open = async () => { s.batch = b.batch_id; s.sid = b.security_id || s.sid; s.itemData = null; await refreshBatch(); Q.renderLab(FD); }; tr.addEventListener("click", open); tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); }); });
  }

  /* ---------- data coverage ---------- */
  CV.s = { data: null, view: "annual", sort: "completeness" };
  CV.run = () => FD.act("coverage", "Counting what the database holds…", async () => { CV.s.data = await FD.read("/api/fund/coverage/overview", { years: 12 }); return { kind: "coverage" }; });
  CV.auto = () => { if (!CV.s.data && FD.db === "real") CV.run(); };
  CV.empty = (root) => FD.emptyCard(root, "What does the database hold?", ["Press Refresh to count the fiscal years and quarters held for each company, which source each came from, which are missing, and what is waiting for review."], { label: "Refresh", fn: () => CV.run() });
  CV.sections = (panel) => {
    const b = FD.sec(panel, "View");
    W.seg(b, [["annual", "Annual"], ["quarter", "Quarterly"]], CV.s.view, (v) => { CV.s.view = v; Q.renderLab(FD); }).dataset.role = "coverage-view";
    h("p", "hint", b, "Blue is held, red is missing between the first and the latest period held, dark grey is not yet due. The fourth quarter is worked out from the year and marked as derived.");
    const g = FD.sec(panel, "Fill the gaps");
    const all = h("button", "btn", g, "Backfill every gap from the SEC"); all.type = "button"; all.dataset.action = "coverage-backfill";
    all.title = "Asks SEC EDGAR (free) for the annual statements and the 10-Q quarters of the companies that have gaps. US-listed companies only.";
    all.addEventListener("click", () => backfill(null));
    h("p", "hint", g, "Reports older than the SEC's XBRL (before 2009 to 2011), or from a company's own website, come in through Report import.");
  };
  function backfill(ids) {
    return FD.act("coverage", "Backfilling from SEC EDGAR…", async (run) => {
      const { result } = await run.job("/api/fund/coverage/backfill", ids ? { security_ids: ids, quarters: true } : { quarters: true }, "Backfilling");
      CV.s.data = await FD.read("/api/fund/coverage/overview", { years: 12 }); return { kind: "coverage", backfill: result };
    });
  }
  CV.build = (root, d) => {
    const data = CV.s.data; if (!data) { CV.empty(root); return; }
    Q.labHead(root, "Data coverage", `${data.companies.length} companies · as of ${data.as_of}`);
    if (d && d.backfill) h("div", "banner", root, `Backfill done: ${d.backfill.fetched ?? d.backfill.companies ?? ""} ${d.backfill.fetched !== undefined ? "companies fetched" : ""}`.trim());
    const t = data.totals, p = data.problems, k = h("div", "earnrow", root);
    tile(k, "Annual statements held", String(t.annual_held || 0), { text: `${t.annual_missing || 0} missing in between` }, (t.annual_missing || 0) ? "" : "");
    tile(k, "Quarters held", String(t.quarter_held || 0), { text: `${t.quarter_missing || 0} missing` });
    tile(k, "Awaiting review", String(p.awaiting_review), { text: "imported, not yet approved" }, p.awaiting_review ? "" : "muted");
    tile(k, "Conflicts with held figures", String(p.import_conflicts), { text: "reports that differ from the database" }, p.import_conflicts ? "" : "muted");
    tile(k, "Failed imports", String(p.failed_imports), {}, p.failed_imports ? "" : "muted");
    tile(k, "Dates assumed", String(p.estimated_availability), { text: "periods with no filing date: availability is estimated" }, p.estimated_availability ? "" : "muted");
    if (!data.companies.length) { FD.emptyCard(root, "Nothing is held yet", ["Add companies in the Data manager, or import a report."]); return; }
    const annual = CV.s.view === "annual";
    const cols = annual ? [...new Set(data.companies.flatMap((c) => Object.keys(c.annual)))].sort() : [...new Set(data.companies.flatMap((c) => Object.keys(c.quarters)))].sort().slice(-30);
    const shown = data.companies.slice(0, 60), VAL = { held: 1, derived: 0.5, missing: -1, notyet: null, na: null };
    const card = W.card(root, annual ? "Fiscal years by company" : "Fiscal quarters and half-years by company (the latest six years)", `${shown.length < data.companies.length ? `The ${shown.length} least complete of ${data.companies.length} companies. ` : ""}Click a company to fill its gaps.`); card.dataset.role = "coverage-grid";
    const body = h("div", "", card);
    C.matrix(body, { rows: shown.map((c) => `${c.ticker || c.security_id} · ${pct(c.completeness, 0)}`), cols: cols.map((x) => (annual ? x : x.replace(/^(\d{4})([QH]\d)$/, "$2 $1"))), values: shown.map((c) => cols.map((x) => { const cell = (annual ? c.annual : c.quarters)[x]; return cell ? VAL[cell.state] ?? null : null; })),
      diverging: true, mid: 0, min: -1, max: 1, format: (v) => (v === 1 ? "held" : v === 0.5 ? "derived" : v === -1 ? "missing" : ""), name: "State", cellHeight: 22, label: annual ? "Annual statements held and missing by company" : "Quarterly statements held and missing by company" });
    const src = W.card(root, "Where the figures come from", "Periods held by source (a company can have more than one).");
    C.hbars(h("div", "", src), { rows: data.sources.map((x) => ({ label: `${x.provider} (${x.companies} companies)`, value: x.periods })), format: (v) => String(Math.round(v)), name: "Periods", label: "Periods held by source", labelWidth: 200, color: () => css("--s1") });
    const list = W.card(root, "Companies", "Sorted by how complete they are. Backfill asks the SEC for what is missing (US-listed companies only).");
    table(list, [{ label: "Ticker", get: (c) => c.ticker || "" }, { label: "Name", get: (c) => c.name || "" }, { label: "Years held", num: true, get: (c) => String(c.annual_held) }, { label: "Years missing", num: true, get: (c) => String(c.annual_missing) },
      { label: "Quarters held", num: true, get: (c) => String(c.quarter_held) }, { label: "Quarters missing", num: true, get: (c) => String(c.quarter_missing) }, { label: "Complete", num: true, get: (c) => pct(c.completeness, 0) }, { label: "Sources", get: (c) => Object.keys(c.sources).join(", ") },
      { label: "", get: (c) => (c.annual_missing || c.quarter_missing ? "Backfill" : "") }], data.companies.slice(0, 200));
    [...list.querySelectorAll("tbody tr")].forEach((tr, i) => { const c = data.companies[i]; if (!(c.annual_missing || c.quarter_missing)) return; const last = tr.lastElementChild; last.textContent = ""; const b = h("button", "btn small", last, "Backfill"); b.type = "button"; b.dataset.action = "coverage-backfill-one"; b.setAttribute("aria-label", "backfill " + (c.ticker || c.security_id)); b.addEventListener("click", () => backfill([c.security_id])); });
    if (data.recent_imports.length) { const r = W.card(root, "Recent imports"); table(r, [{ label: "Batch", get: (b) => b.label || b.batch_id }, { label: "Started", get: (b) => day(b.created_at) }, { label: "Reports", num: true, get: (b) => String(b.items) }, { label: "Saved", num: true, get: (b) => String(b.counts.saved || 0) },
      { label: "To review", num: true, get: (b) => String((b.counts.needs_review || 0) + (b.counts.extracted || 0)) }, { label: "Failed", num: true, get: (b) => String(b.counts.failed || 0) }], data.recent_imports); }
    if (Object.keys(p.quality_flags).length) FD.notes(root, "Data quality flags", Object.entries(p.quality_flags).map(([f, n]) => `${human(f)}: ${n} figure${n === 1 ? "" : "s"}`));
  };
})();
