/* The Fundamentals tab's "Enter reports" view: financial reports typed or pasted in by the person, like a spreadsheet, stored in the database beside the providers' figures and read by every study.
   The grid has the periods across and the lines of the statements down. Figures can be typed, pasted into the grid from Excel, or read from a pasted table or a CSV file (the server recognises the
   line names). "Calculate" works out the ratios, scores and multiples on a temporary copy of the company and stores nothing; "Save" writes the periods (source "manual") with the day each report
   became public. It talks to /api/fund/manual/*. Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const FD = window.FD, Q = window.QL, W = window.WS;
  if (!FD || !Q || !W) return;
  const h = Q.h, num = Q.num, pct = Q.pct, tile = Q.tile, table = Q.simpleTable, api = Q.api;
  const { compact, human, day } = FD.fmt;
  const EN = FD.views.enter = {};
  const isOpen = () => !Q.$(FD.view).hidden;
  const FACTOR = { units: 1, thousands: 1e3, millions: 1e6, billions: 1e9 };
  const UNIT_CHOICES = [["units", "single units (as filed)"], ["thousands", "thousands"], ["millions", "millions"], ["billions", "billions"]];
  const PERIOD_CHOICES = ["FY", "Q1", "Q2", "Q3", "Q4", "H1", "H2"].map((p) => [p, p]);
  const FAMILY = { dupont: "DuPont", cashflow: "Cash flow" };
  const family = (f) => FAMILY[f] || human(f);
  const PRICE = { code: "price", label: "Share price (optional)", unit: "price", text: "The share price when the report came out, in the reporting currency: used only to work out multiples and yields, never stored.", core: false, statement: "price" };
  const LEVEL = { error: ["✗", "tone-crit", "Error"], warning: ["!", "tone-warn", "Check"], info: ["i", "hint", "Note"] };

  const blank = () => ({
    target: "existing", q: "", found: null, finding: false, findError: "", pick: "", chosen: null, stored: null,
    company: { ticker: "", name: "", exchange: "", country: "", currency: "", sector: "", industry: "", price_symbol: "" },
    units: { money: "millions", shares: "millions" }, currency: "USD",
    cols: [], values: {}, all: false, text: "", report: null,
    save_mode: "correct", restated_on: "", preview: null, map: [], saved: null, stale: false, tpl: null, tplError: "", priceReport: null,
  });
  FD.en = blank();
  FD.views.reset = () => { FD.en = blank(); };
  FD.modes.splice(2, 0, ["enter", "Enter reports"]);
  FD.runLabel.enter = "Calculate the ratios";
  FD.blurb.enter = "Type or paste a company's financial reports like a spreadsheet, work out its ratios and scores, and save them in the database beside the providers' figures.";

  /* ---------- the lines of the grid ---------- */
  EN.items = () => { const out = []; if (!FD.en.tpl) return out; for (const st of FD.en.tpl.statements) for (const it of st.items) out.push(Object.assign({ statement: st.key, group: st.title }, it)); return out; };
  EN.has = (code) => (FD.en.values[code] || []).some((v) => String(v || "").trim() !== "");
  EN.visible = () => { const s = FD.en; return EN.items().filter((it) => s.all || it.core || EN.has(it.code)).concat([PRICE]); };
  EN.ensure = (code) => { const s = FD.en; if (!s.values[code]) s.values[code] = s.cols.map(() => ""); return s.values[code]; };
  EN.addColumn = (col) => {
    const s = FD.en;
    s.cols.push(Object.assign({ fiscal_period: "FY", fiscal_year: "", period_end: "", filing_date: "", currency: "", guessed: false, label: "" }, col || {}));
    for (const code of Object.keys(s.values)) s.values[code].push("");
    return s.cols.length - 1;
  };
  EN.removeColumn = (i) => { const s = FD.en; s.cols.splice(i, 1); for (const code of Object.keys(s.values)) s.values[code].splice(i, 1); s.stale = true; };
  EN.sortColumns = () => {
    const s = FD.en, order = s.cols.map((c, i) => i).sort((a, b) => (s.cols[a].period_end || "9").localeCompare(s.cols[b].period_end || "9") || s.cols[a].fiscal_period.localeCompare(s.cols[b].fiscal_period));
    s.cols = order.map((i) => s.cols[i]);
    for (const code of Object.keys(s.values)) s.values[code] = order.map((i) => s.values[code][i]);
  };
  EN.colLabel = (c, i) => (c.period_end ? `${c.fiscal_period} to ${c.period_end}` : `column ${i + 1}`);
  EN.touch = () => { const s = FD.en; s.stale = true; const el = Q.$('[data-role="manual-stale"]'); if (el) el.hidden = false; Q.syncLab(FD); };   // the run button follows what is typed: it is on once there is a company and a figure
  /* Read a stored period back into the grid's units: the server keeps the reporting currency's own units, the grid shows the unit chosen. */
  EN.fromStored = (periods, replace) => {
    const s = FD.en, items = {}; for (const it of EN.items()) items[it.code] = it;
    if (replace) { s.cols = []; s.values = {}; }
    const f = FACTOR[s.units.money], g = FACTOR[s.units.shares];
    for (const p of periods) {
      let k = s.cols.findIndex((c) => c.fiscal_period === p.fiscal_period && c.period_end === p.period_end);
      if (k < 0) k = EN.addColumn({ fiscal_period: p.fiscal_period, period_end: p.period_end, fiscal_year: p.fiscal_year || "", filing_date: p.filing_date || "", currency: p.currency || "", label: p.period_end });
      else Object.assign(s.cols[k], { fiscal_year: p.fiscal_year || "", filing_date: p.filing_date || "", currency: p.currency || "", guessed: false });
      for (const [code, v] of Object.entries(p.values)) {
        const unit = (items[code] || {}).unit, scaled = unit === "per_share" ? v : v / (unit === "shares" ? g : f);
        EN.ensure(code)[k] = String(+scaled.toPrecision(12));
      }
    }
    EN.sortColumns();
  };
  /* Put a table read by the server into the grid. A period that is already there is filled in (a statement pasted after another one adds its lines to the same columns); a new one is added. */
  EN.apply = (parsed) => {
    const s = FD.en, where = parsed.columns.map((pc) => {
      let k = s.cols.findIndex((c) => c.fiscal_period === pc.fiscal_period && c.period_end === pc.period_end);
      if (k < 0) k = EN.addColumn({ fiscal_period: pc.fiscal_period, fiscal_year: pc.fiscal_year || "", period_end: pc.period_end || "", filing_date: pc.filing_date || "", currency: pc.currency || "", guessed: !!pc.guessed, label: pc.label });
      else { if (pc.filing_date) s.cols[k].filing_date = pc.filing_date; if (pc.currency) s.cols[k].currency = pc.currency; }
      return k;
    });
    const put = (code, vals) => { const row = EN.ensure(code); vals.forEach((v, j) => { if (v !== null && v !== undefined) row[where[j]] = String(v); }); };
    for (const [code, vals] of Object.entries(parsed.values)) put(code, vals);
    if (parsed.price) put("price", parsed.price);
    // the columns that were added after the first line of the grid was made have an empty cell in every line: ``addColumn`` did that
    EN.sortColumns();
    s.report = { columns: parsed.columns.length, lines: Object.keys(parsed.values).length, unrecognised: parsed.unrecognised, duplicates: parsed.duplicates, problems: parsed.problems, notes: parsed.notes, guessed: parsed.columns.filter((c) => c.guessed).length };
    EN.touch();
  };

  /* ---------- the request ---------- */
  EN.hasCompany = () => { const s = FD.en; return s.target === "existing" ? !!s.chosen : !!s.company.ticker.trim(); };
  EN.hasFigures = () => FD.en.cols.some((c, i) => Object.entries(FD.en.values).some(([code, arr]) => code !== "price" && String(arr[i] || "").trim() !== ""));
  /* The columns that have figures, as the server wants them. ``map`` says which column of the grid each one was, so a problem the server finds can be put on the cell it is about. */
  EN.body = () => {
    const s = FD.en, periods = [], map = [];
    s.cols.forEach((c, i) => {
      const values = {}; let any = false;
      for (const [code, arr] of Object.entries(s.values)) { const v = String(arr[i] === undefined ? "" : arr[i]).trim(); if (v !== "" && code !== "price") { values[code] = v; any = true; } }
      if (!any) return;
      const p = { fiscal_period: c.fiscal_period, period_end: String(c.period_end || "").trim(), values };
      if (String(c.fiscal_year).trim() !== "") p.fiscal_year = c.fiscal_year;
      if (String(c.filing_date || "").trim() !== "") p.filing_date = String(c.filing_date).trim();
      if (String(c.currency || "").trim() !== "") p.currency = String(c.currency).trim();
      const price = String((s.values.price || [])[i] || "").trim(); if (price !== "") p.price = price;
      periods.push(p); map.push(i);
    });
    const body = { units: Object.assign({}, s.units), periods, mode: s.save_mode };
    if (String(s.currency).trim()) body.currency = String(s.currency).trim();
    if (s.target === "existing") body.security_id = s.chosen.security_id; else body.company = Object.assign({}, s.company);
    if (s.save_mode === "restate") body.restated_on = s.restated_on.trim();
    return { body, map };
  };
  EN.check = () => {
    const s = FD.en;
    if (FD.db === "demo") return "Reports are entered into your real database: switch to Real data (the demo is a simulated world).";
    if (!s.tpl) return "The grid is still loading.";
    if (!EN.hasCompany()) return s.target === "existing" ? "Choose the company these reports belong to (search for it first), or switch to a new company." : "Give the new company a ticker.";
    if (!EN.hasFigures()) return "Type or paste some figures first.";
    return "";
  };
  EN.marks = () => {
    const out = {}, s = FD.en;
    if (!s.preview || s.stale) return out;
    for (const i of s.preview.issues || []) {
      if (i.column === null || i.column === undefined) continue;
      const col = s.map[i.column]; if (col === undefined) continue;
      const key = `${col}:${i.field || ""}`, rank = { error: 3, warning: 2, info: 1 };
      if (!out[key] || rank[i.level] > rank[out[key].level]) out[key] = { level: i.level, message: i.message };
    }
    return out;
  };
  EN.run = () => FD.act("enter", "Checking the figures and working out the ratios…", async () => {
    const { body, map } = EN.body(), s = FD.en;
    const p = await FD.send("/api/fund/manual/preview", body);
    s.preview = p; s.map = map; s.stale = false;
    return { preview: p };
  });
  EN.save = () => {
    const s = FD.en;
    if (s.save_mode === "restate" && !s.restated_on.trim()) { Q.toast("A restatement needs the date it was published (YYYY-MM-DD)"); return; }
    return FD.act("enter", "Saving to the database…", async () => {
      const { body, map } = EN.body();
      const r = await FD.send("/api/fund/manual/save", body);
      s.saved = r; s.map = map; s.priceReport = null;
      if (s.target === "new") { s.target = "existing"; s.chosen = { security_id: r.security_id, ticker: r.ticker, company_name: r.name, currency: s.company.currency || null }; s.pick = String(r.security_id); s.found = [s.chosen]; }
      await EN.refreshStored();
      FD.resetCaches();
      Q.toast(`${r.ticker}: ${r.added} added, ${r.corrected + r.restated} changed, ${r.unchanged} unchanged`);
      return { preview: s.preview };
    });
  };
  EN.refreshStored = async () => {
    const s = FD.en; if (!s.chosen) { s.stored = null; return; }
    try { s.stored = await FD.read("/api/fund/manual/entries", { id: s.chosen.security_id }); } catch (e) { s.stored = null; }
  };
  EN.parse = async (text) => {
    const s = FD.en;
    if (!String(text || "").trim()) { Q.toast("Paste a table first"); return; }
    try {
      const r = await FD.send("/api/fund/manual/parse", { text });
      EN.apply(r); s.text = ""; Q.renderLab(FD);
      Q.toast(`Read ${r.columns.length} period${r.columns.length === 1 ? "" : "s"} and ${Object.keys(r.values).length} lines`);
    } catch (e) { s.report = { error: e.message }; Q.renderLab(FD); Q.toast(e.message); }
  };
  EN.find = async () => {
    const s = FD.en; s.finding = true; s.findError = ""; Q.renderLab(FD);
    try { const r = await FD.read("/api/fund/companies", { q: s.q, limit: 25 }); s.found = r.companies; }
    catch (e) { s.findError = e.message; s.found = null; }
    s.finding = false; if (isOpen() && FD.mode === "enter") Q.renderLab(FD);
  };
  EN.choose = async (id) => {
    const s = FD.en; s.pick = id; s.chosen = (s.found || []).find((r) => String(r.security_id) === String(id)) || null; s.stored = null; s.saved = null;
    if (s.chosen && s.chosen.currency) s.currency = s.chosen.currency;                      // the usual reporting currency of a company is the one it trades in; the person may change it
    await EN.refreshStored();
    Q.renderLab(FD);
  };
  EN.deleteStored = async (p) => {
    const s = FD.en;
    try { await FD.send("/api/fund/manual/delete", { security_id: s.chosen.security_id, fiscal_period: p.fiscal_period, period_end: p.period_end }); Q.toast(`${p.fiscal_period} to ${p.period_end} removed from the database`); FD.resetCaches(); s.saved = null; await EN.refreshStored(); }
    catch (e) { Q.toast(e.message); }
    Q.renderLab(FD);
  };
  EN.loadPrices = () => {
    const s = FD.en; if (!s.saved) return;
    return FD.act("enter", "Loading prices…", async (run) => {
      const { result } = await run.job("/api/fund/prices", { security_ids: [s.saved.security_id] }, "Loading its prices");
      s.priceReport = result; FD.resetCaches();
      return { preview: s.preview };
    });
  };

  /* ---------- the panel ---------- */
  EN.sections = (panel) => {
    const s = FD.en;
    if (FD.db === "demo") { h("p", "hint tone-warn", FD.sec(panel, "Real data only"), "Reports are typed into your real database. The demo is a simulated world and stays one: switch to Real data above."); return; }
    if (!s.tpl && !s.tplError) {
      api("/api/fund/manual/template").then((t) => { s.tpl = t; if (isOpen() && FD.mode === "enter") Q.renderLab(FD); }).catch((e) => { s.tplError = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + Q.RESTART : e.message; if (isOpen() && FD.mode === "enter") Q.renderLab(FD); });
    }
    EN.companySection(panel);
    const u = FD.sec(panel, "2. Units and currency");
    Q.labField(u, { key: "money", label: "Amounts are typed in", kind: "choice", options: UNIT_CHOICES, help: "Revenue, profit, assets, debt and cash flow. The server stores them in the reporting currency's own units." }, s.units, () => { EN.touch(); FD.after(); });
    Q.labField(u, { key: "shares", label: "Share counts are typed in", kind: "choice", options: UNIT_CHOICES, help: "Shares outstanding and the weighted-average shares. Earnings per share is always per share." }, s.units, () => { EN.touch(); FD.after(); });
    Q.labField(u, { key: "currency", label: "Reporting currency (three letters)", kind: "text", placeholder: "USD", help: "The currency of the statements; each column may override it." }, s, () => { EN.touch(); FD.after(); });
    h("p", "hint", u, "Change the unit before typing, not after: the numbers already in the grid are not converted.");
    const f = FD.sec(panel, "3. Bring the figures in");
    h("p", "hint", f, "Copy a table in Excel (the periods across the top, the line names down the left) and paste it here. Statements can be pasted one after the other: each adds its lines to the same periods. A CSV file works the same way.");
    const box = h("textarea", "", f); box.rows = 6; box.value = s.text; box.placeholder = "Item\tFY2022\tFY2023\nTotal revenue\t1,200\t1,500\nNet income\t195\t260"; box.spellcheck = false;
    box.setAttribute("aria-label", "paste a table from Excel"); box.dataset.role = "manual-paste"; box.addEventListener("input", () => { s.text = box.value; });
    const row = h("div", "row tight", f);
    const read = h("button", "btn", row, "Read the pasted table"); read.type = "button"; read.dataset.action = "manual-parse"; read.addEventListener("click", () => EN.parse(box.value));
    const label = h("label", "f", f); label.appendChild(document.createTextNode("or open a CSV file"));
    const file = h("input", "", label); file.type = "file"; file.accept = ".csv,.tsv,.txt,text/csv,text/plain"; file.dataset.role = "manual-file"; file.setAttribute("aria-label", "open a CSV file");
    file.addEventListener("change", () => {
      const pick = file.files && file.files[0]; if (!pick) return;
      if (pick.size > 140000) { Q.toast("That file is larger than the page reads (about 140 KB): paste the part you need"); return; }
      const reader = new FileReader(); reader.onload = () => EN.parse(String(reader.result)); reader.onerror = () => Q.toast("That file could not be read"); reader.readAsText(pick);
    });
    const more = h("div", "row tight", f);
    const add = h("button", "btn small", more, "Add a period"); add.type = "button"; add.dataset.action = "manual-add-period"; add.addEventListener("click", () => { EN.addColumn({ period_end: "" }); EN.touch(); Q.renderLab(FD); });
    const clear = h("button", "btn small", more, "Clear the grid"); clear.type = "button"; clear.dataset.action = "manual-clear"; clear.addEventListener("click", () => { s.cols = []; s.values = {}; s.report = null; s.preview = null; s.saved = null; Q.renderLab(FD); });
    EN.saveSection(panel);
  };
  EN.companySection = (panel) => {
    const s = FD.en, b = FD.sec(panel, "1. Which company");
    W.seg(b, [["existing", "In my database"], ["new", "A new company"]], s.target, (v) => { if (v !== s.target) { s.target = v; s.saved = null; Q.renderLab(FD); } }).dataset.role = "manual-target";
    if (s.target === "existing") {
      Q.labField(b, { key: "q", label: "Ticker or name", kind: "text", placeholder: "apple, 7203" }, s, () => {});
      const box = b.querySelector("input[data-key=q]");
      if (box) box.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); EN.find(); } });
      const go = h("button", "btn small", b, s.finding ? "Searching…" : "Search"); go.type = "button"; go.dataset.action = "manual-find"; go.disabled = s.finding; go.addEventListener("click", EN.find);
      if (s.findError) h("p", "hint tone-crit", b, s.findError);
      if (s.found) {
        if (!s.found.length) h("p", "hint", b, "No company in your database matches. Add it as a new company, or search the providers in the Company explorer.");
        else Q.labField(b, { key: "pick", label: "Company", kind: "choice", options: [["", "Choose…"]].concat(s.found.map((r) => [String(r.security_id), `${r.ticker}${r.exchange ? " · " + r.exchange : ""} — ${r.company_name || "no name"}`])) }, s, () => EN.choose(s.pick));
      }
      if (s.chosen) {
        const c = s.chosen;
        h("p", "hint", b, `${c.company_name || c.ticker}: ${c.annual || 0} annual and ${c.quarterly || 0} quarterly periods from ${c.source || "its source"}. Where a provider holds the same period, its figures lead and yours fill only the lines it lacks.`).dataset.role = "manual-chosen";
      }
    } else {
      const c = s.company, after = () => { EN.touch(); };
      Q.labField(b, { key: "ticker", label: "Ticker", kind: "text", placeholder: "NVDA or 7203.T" }, c, after);
      Q.labField(b, { key: "name", label: "Company name", kind: "text", placeholder: "Example Corp" }, c, after);
      Q.labField(b, { key: "country", label: "Country (two letters)", kind: "text", placeholder: "US" }, c, after);
      Q.labField(b, { key: "currency", label: "Trading currency (three letters)", kind: "text", placeholder: "USD" }, c, after);
      Q.labField(b, { key: "sector", label: "Sector", kind: "text", placeholder: "Technology" }, c, after);
      Q.labField(b, { key: "industry", label: "Industry", kind: "text", placeholder: "Semiconductors" }, c, after);
      Q.labField(b, { key: "exchange", label: "Exchange (optional)", kind: "text", placeholder: "XNAS" }, c, after);
      Q.labField(b, { key: "price_symbol", label: "Yahoo price symbol (optional)", kind: "text", placeholder: "NVDA", help: "The symbol Yahoo Finance knows it by, so its prices can be loaded for studies." }, c, after);
      h("p", "hint", b, "The company is created when you save. A ticker already in your database is refused: choose it under “In my database” instead.");
    }
  };
  EN.saveSection = (panel) => {
    const s = FD.en, b = FD.sec(panel, "4. Save to the database");
    W.seg(b, [["correct", "Store as typed"], ["restate", "A restatement"]], s.save_mode, (v) => { s.save_mode = v; Q.renderLab(FD); }).dataset.role = "manual-mode";
    if (s.save_mode === "restate") {
      Q.labField(b, { key: "restated_on", label: "Restated on (date published)", kind: "text", placeholder: "2024-02-15" }, s, () => {});
      h("p", "hint", b, "The company restated a period it reported before. The new figures are kept beside the old ones as a new version that is public only from this date, as for a provider; studies read the figures as first filed.");
    } else h("p", "hint", b, "Stores the periods as typed. A period you entered before is overwritten (use this to correct a typing mistake: nothing wrong was ever public, so no history is kept).");
    const go = h("button", "btn primary", b, "Save to the database"); go.type = "button"; go.dataset.action = "manual-save"; go.addEventListener("click", EN.save);
    h("p", "hint", b, "Saving stores the figures with the date each report was published, works out the company's ratios and stores them too. Nothing is stored by “Calculate the ratios”.");
  };

  /* ---------- the main area ---------- */
  EN.empty = (root) => {
    if (FD.db === "demo") { FD.emptyCard(root, "Enter reports", ["Reports are typed into your real database. The demo is a simulated world, kept apart so that a made-up company never mixes with a real one.", "Switch to Real data above to enter reports."]); return; }
    EN.top(root);
    h("p", "hint", root, "Fill in the grid, then press “Calculate the ratios” to see the ratios, scores and multiples it gives. Nothing is stored until you press “Save to the database”.");
  };
  EN.build = (root, data) => {
    EN.top(root);
    const p = data && data.preview;
    if (!p) { h("p", "hint", root, "Press “Calculate the ratios” to see what these figures give."); return; }
    const stale = h("div", "banner", root); stale.dataset.role = "manual-stale"; stale.hidden = !FD.en.stale; stale.setAttribute("role", "status");
    h("strong", "", stale, "The figures changed. "); stale.appendChild(document.createTextNode("What is shown below is from before: press “Calculate the ratios” again."));
    EN.issues(root, p);
    if (!p.ok) return;
    EN.ratios(root, p);
    EN.multiples(root, p);
    EN.overlaps(root, p.overlaps);
    EN.company(root, p);
  };
  EN.top = (root) => {
    const s = FD.en;
    Q.labHead(root, "Enter reports", s.chosen ? `${s.chosen.company_name || s.chosen.ticker} (${s.chosen.ticker})` : s.target === "new" && s.company.ticker.trim() ? `new company ${s.company.ticker.trim().toUpperCase()}` : "financial reports typed in like a spreadsheet", "fundamentals");
    if (s.tplError) { Q.labBanner(root, "The grid did not load. ", s.tplError, true); return; }
    if (!s.tpl) { h("p", "hint", root, "Loading the grid…"); return; }
    EN.savedCard(root);
    EN.reportCard(root);
    EN.storedCard(root);
    EN.grid(root);
    const how = FD.explain(root, "How the figures are used", "Entered figures are stored as the source “manual”, with the date each report was published; a study sees them from that day plus the lag you choose, never from the period end. "
      + "Where a provider holds the same period, the provider's figures lead and yours fill only the lines it lacks. A line left empty is missing, never zero, and a ratio is computed only when every figure it uses is there.");
    how.dataset.role = "manual-how";
  };
  EN.savedCard = (root) => {
    const s = FD.en, r = s.saved; if (!r) return;
    const card = W.card(root, `Saved: ${r.name || r.ticker} (${r.ticker})`, r.created_company ? "The company was created and its reports stored." : "Stored in the database.");
    card.dataset.role = "manual-saved"; card.classList.add("roomy");
    const k = h("div", "earnrow", card);
    tile(k, "Added", String(r.added), { text: "new periods" });
    tile(k, "Changed", String(r.corrected + r.restated), { text: r.restated ? `${r.restated} stored as a restatement` : "overwritten with new figures" });
    tile(k, "Unchanged", String(r.unchanged), { text: "already as typed" });
    tile(k, "Ratios stored", r.metrics_written.toLocaleString("en-US"), { text: "derived metrics of the company" });
    if ((r.overlaps || []).length) { h("h4", "subhead", card, "Where a provider also reports these periods"); EN.overlapTable(card, r.overlaps); }
    const row = h("div", "row tight", card);
    const open = h("button", "btn", row, "Open it in the Company explorer"); open.type = "button"; open.dataset.action = "manual-open-company";
    open.addEventListener("click", () => { FD.co.id = r.security_id; FD.goto("companies"); window.scrollTo({ top: 0 }); });
    const px = h("button", "btn", row, "Load its prices (free)"); px.type = "button"; px.dataset.action = "manual-load-prices"; px.addEventListener("click", EN.loadPrices);
    if (s.priceReport) {
      const pr = s.priceReport;
      h("p", pr.loaded ? "tone-good" : "tone-warn", card, pr.loaded ? `Loaded ${pr.rows.toLocaleString("en-US")} daily prices.` : `No prices were found${pr.failed && pr.failed[0] ? ` (${pr.failed[0].reason})` : ""}: set the price symbol in the Company explorer.`);
    } else h("p", "hint", card, "A study measures what a factor earned, so a company needs prices before it can be studied. They are free, from Yahoo Finance.");
    h("p", "hint", card, "This company now takes part in the Factor lab and the other studies like any other. A company with only a few periods adds little to a cross-section of hundreds.");
  };
  EN.reportCard = (root) => {
    const r = FD.en.report; if (!r) return;
    const card = h("div", "card", root); card.dataset.role = "manual-report";
    if (r.error) { h("h3", "", card, "That table could not be read"); h("p", "tone-crit", card, r.error); return; }
    h("h3", "", card, `Read ${r.columns} period${r.columns === 1 ? "" : "s"} and ${r.lines} line${r.lines === 1 ? "" : "s"}`);
    if (r.guessed) h("p", "tone-warn", card, `${r.guessed} column${r.guessed === 1 ? " has" : "s have"} only a year or a quarter in the header, so the calendar end of the period was assumed (marked in the grid). Correct the period end if the fiscal year ends on another date.`);
    for (const n of r.notes || []) h("p", "hint", card, n);
    if ((r.unrecognised || []).length) h("p", "hint", card, "Rows not recognised, left out: " + r.unrecognised.map((x) => `${x.label}${x.figures ? ` (${x.figures} figure${x.figures === 1 ? "" : "s"})` : ""}`).join("; ") + ". Type those lines into the grid, or rename them to a line the grid has.");
    if ((r.duplicates || []).length) h("p", "hint", card, "Given twice, the first was kept: " + r.duplicates.join("; ") + ".");
    if ((r.problems || []).length) { const ul = h("ul", "", card); for (const p of r.problems.slice(0, 12)) h("li", "tone-crit", ul, `${p.label}, ${p.column}: ${p.message}`); }
  };
  EN.storedCard = (root) => {
    const s = FD.en, st = s.stored; if (!st || !st.periods.length) return;
    const card = W.card(root, "Already entered by hand for this company", "These are in the database now. Load them into the grid to correct them, or remove a period."); card.dataset.role = "manual-stored";
    const wrap = FD.guardScroll(h("div", "scroll", card)), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t));
    for (const label of ["Period", "Published", "Currency", "Lines", "Dates", "Actions"]) h("th", "", hd, label);
    const tb = h("tbody", "", t);
    for (const p of st.periods) {
      const tr = h("tr", "", tb); tr.dataset.period = `${p.fiscal_period}:${p.period_end}`;
      h("td", "", tr, `${p.fiscal_period} to ${p.period_end}`); h("td", "", tr, day(p.filing_date) || "not given"); h("td", "", tr, p.currency || ""); h("td", "num", tr, String(Object.keys(p.values).length));
      h("td", "", tr, p.quality === "estimated" ? "estimated lag" : p.quality === "restated" ? "restated" : "filed");
      const cell = h("td", "", tr), load = h("button", "btn small", cell, "Edit"); load.type = "button"; load.dataset.action = "manual-load-period"; load.setAttribute("aria-label", `edit ${p.fiscal_period} to ${p.period_end}`);
      load.addEventListener("click", () => { EN.fromStored([p], false); EN.touch(); Q.renderLab(FD); window.scrollTo({ top: 0 }); });
      const del = h("button", "btn small", cell, "Remove"); del.type = "button"; del.dataset.action = "manual-delete-period"; del.setAttribute("aria-label", `remove ${p.fiscal_period} to ${p.period_end}`);
      del.addEventListener("click", () => { if (window.confirm(`Remove ${p.fiscal_period} to ${p.period_end} from the database? A provider's figures for the same period are not touched.`)) EN.deleteStored(p); });
    }
    const all = h("button", "btn small", card, "Load every period into the grid"); all.type = "button"; all.dataset.action = "manual-load-all";
    all.addEventListener("click", () => { s.units.money = st.suggested_unit; s.units.shares = st.suggested_unit; EN.fromStored(st.periods, true); s.preview = null; s.saved = null; Q.renderLab(FD); });
  };

  /* The grid: periods across, lines down, one text cell for each figure. */
  EN.grid = (root) => {
    const s = FD.en, card = W.card(root, "The figures", ""), marks = EN.marks();
    card.dataset.role = "manual-grid"; card.classList.add("roomy");
    h("p", "note", card, `Amounts in ${s.units.money === "units" ? "single units" : s.units.money}${String(s.currency).trim() ? " of " + String(s.currency).trim().toUpperCase() : ""}; share counts in ${s.units.shares === "units" ? "single units" : s.units.shares}. Paste a block from Excel into any cell. Empty means missing, not zero.`);
    const bar = h("div", "row tight", card);
    const all = Q.labField(bar, { key: "all", label: `Show every line (${EN.items().length})`, kind: "bool", help: "The main lines are shown; lines that have figures are always shown." }, s, () => { Q.renderLab(FD); });
    all.dataset.role = "manual-all";
    const add = h("button", "btn small", bar, "Add a period"); add.type = "button"; add.dataset.action = "manual-add-period-grid"; add.addEventListener("click", () => { EN.addColumn({ period_end: "" }); EN.touch(); FD.draw(); });
    if (!s.cols.length) {
      h("p", "hint", card, "The grid is empty. Add a period, paste a table into the box on the left, or open a CSV file.");
      const start = h("button", "btn", card, "Start with three years"); start.type = "button"; start.dataset.action = "manual-start";
      start.addEventListener("click", () => { const y = new Date().getFullYear(); for (const k of [3, 2, 1]) EN.addColumn({ period_end: `${y - k}-12-31`, fiscal_year: y - k }); Q.renderLab(FD); });
      return;
    }
    const wrap = FD.guardScroll(h("div", "scroll enterwrap", card)), t = h("table", "data entergrid", wrap);
    const head = h("thead", "", t);
    const headRows = [["fiscal_period", "Period (FY = a full year)"], ["period_end", "Period end"], ["fiscal_year", "Fiscal year"], ["filing_date", "Published on"], ["currency", "Currency"]];
    for (const [key, label] of headRows) {
      const tr = h("tr", "", head), th = h("th", "rowhead", tr, label); th.scope = "row";
      s.cols.forEach((c, i) => {
        const td = h("th", "colhead", tr), mark = marks[`${i}:`];
        let input;
        if (key === "fiscal_period") { input = h("select", "", td); for (const [v, text] of PERIOD_CHOICES) { const o = h("option", "", input, text); o.value = v; if (v === c.fiscal_period) o.selected = true; } }
        else { input = h("input", "", td); input.type = "text"; input.value = c[key] === undefined || c[key] === null ? "" : String(c[key]); input.spellcheck = false; input.placeholder = key === "currency" ? (String(s.currency).trim().toUpperCase() || "USD") : key === "fiscal_year" ? "year" : "YYYY-MM-DD"; }
        input.setAttribute("aria-label", `${label}, ${EN.colLabel(c, i)}`); input.dataset.col = String(i); input.dataset.head = key;
        if (key === "period_end" && c.guessed) { input.classList.add("guess"); input.title = "Assumed from the column header: correct it if the period ends on another date"; }
        if (mark && key === "period_end") { input.classList.add(mark.level === "error" ? "bad" : "warn"); input.title = mark.message; }
        input.addEventListener("input", () => { c[key] = input.value; if (key === "period_end") c.guessed = false; if (key === "period_end") input.classList.remove("guess"); EN.touch(); });
        if (key === "fiscal_period") input.addEventListener("change", () => { c.fiscal_period = input.value; EN.touch(); });
      });
    }
    const rm = h("tr", "", head), rmh = h("th", "rowhead", rm); rmh.scope = "row"; h("span", "vh", rmh, "Remove a period");
    s.cols.forEach((c, i) => {
      const td = h("th", "colhead", rm), b = h("button", "btn small", td, "Remove"); b.type = "button"; b.dataset.action = "manual-remove-period"; b.setAttribute("aria-label", `remove the column ${EN.colLabel(c, i)}`);
      b.addEventListener("click", () => { EN.removeColumn(i); Q.renderLab(FD); });
    });
    const body = h("tbody", "", t), lines = EN.visible();
    let group = null;
    lines.forEach((it, r) => {
      const heading = it.code === "price" ? "Optional" : it.group;
      if (heading !== group) { group = heading; const g = h("tr", "group", body), gt = h("th", "rowhead", g, heading); gt.scope = "row"; h("td", "", g).colSpan = s.cols.length; }
      const tr = h("tr", "", body); tr.dataset.line = it.code;
      const lab = h("th", "rowhead", tr); lab.scope = "row"; lab.title = it.text || "";
      h("span", "", lab, it.label);
      if (it.auto) h("span", "auto", lab, " · filled in if left empty").title = "Left empty, this is worked out as " + it.auto;
      if (it.magnitude) lab.title += " Type it as a positive amount or in brackets: either is stored as an amount.";
      s.cols.forEach((c, i) => {
        const td = h("td", "num", tr), input = h("input", "", td), mark = marks[`${i}:${it.code}`];
        input.type = "text"; input.inputMode = "decimal"; input.spellcheck = false; input.autocomplete = "off";
        input.value = (s.values[it.code] || [])[i] || ""; input.dataset.col = String(i); input.dataset.line = it.code; input.dataset.row = String(r);
        input.setAttribute("aria-label", `${it.label}, ${EN.colLabel(c, i)}`);
        if (mark) { input.classList.add(mark.level === "error" ? "bad" : "warn"); input.title = mark.message; input.setAttribute("aria-invalid", mark.level === "error" ? "true" : "false"); }
        input.addEventListener("input", () => { EN.ensure(it.code)[i] = input.value; EN.touch(); input.classList.remove("bad", "warn"); });
        input.addEventListener("paste", (e) => EN.pasteAt(e, r, i));
        input.addEventListener("keydown", (e) => EN.arrow(e, input));
      });
    });
  };
  /* Excel's block of cells pasted into a cell lands from that cell down and to the right; a single value pastes as usual. */
  EN.pasteAt = (e, row, col) => {
    const s = FD.en, text = ((e.clipboardData || window.clipboardData).getData("text") || "");
    if (!/[\t\n]/.test(text.trim())) return;
    e.preventDefault();
    const rows = text.replace(/\r/g, "").replace(/\n+$/, "").split("\n").map((l) => l.split("\t")), lines = EN.visible();
    let left = 0, used = 0;
    rows.forEach((cells, dr) => cells.forEach((cell, dc) => {
      const line = lines[row + dr], c = col + dc;
      if (!line || c >= s.cols.length) { left += 1; return; }
      EN.ensure(line.code)[c] = cell.trim(); used += 1;
    }));
    EN.touch(); FD.draw();
    Q.toast(left ? `Pasted ${used} cells; ${left} fell outside the grid and were left out` : `Pasted ${used} cells`);
  };
  EN.arrow = (e, input) => {
    const keys = { ArrowDown: 1, Enter: 1, ArrowUp: -1 };
    if (!(e.key in keys) || e.shiftKey || e.altKey || e.ctrlKey || e.metaKey) return;
    const col = input.dataset.col, row = Number(input.dataset.row);
    if (Number.isNaN(row)) return;
    const next = input.closest("table").querySelector(`input[data-col="${col}"][data-row="${row + keys[e.key]}"]`);
    if (next) { e.preventDefault(); next.focus(); next.select(); }
  };

  /* ---------- what the figures give ---------- */
  EN.issues = (root, p) => {
    const list = p.issues || [];
    const card = h("div", "card roomy", root); card.dataset.role = "manual-issues";
    const errors = list.filter((i) => i.level === "error"), warnings = list.filter((i) => i.level === "warning"), notes = list.filter((i) => i.level === "info");
    h("h3", "", card, p.ok ? (errors.length ? "Checks" : warnings.length ? `Checked: ${warnings.length} thing${warnings.length === 1 ? "" : "s"} to look at` : "Checked: nothing to correct") : `Not calculated: ${errors.length} error${errors.length === 1 ? "" : "s"} to fix`);
    if (!p.ok) h("p", "tone-crit", card, "Nothing was worked out or stored. The cells with a problem are marked in the grid; fix them and press “Calculate the ratios” again.");
    if (!list.length) { h("p", "hint", card, "The figures add up and the dates are in order."); return; }
    const ul = h("ul", "issues", card), all = new Set(list.map((i) => i.period).filter(Boolean)).size;
    for (const level of ["error", "warning", "info"]) {
      const groups = new Map();                                                           // the same message about several periods is said once, with the periods
      for (const i of list.filter((x) => x.level === level)) { const k = i.message; if (!groups.has(k)) groups.set(k, []); if (i.period) groups.get(k).push(i.period); }
      const rows = [...groups.entries()], limit = level === "info" ? 3 : 30;
      for (const [message, periods] of rows.slice(0, limit)) {
        const [icon, cls, word] = LEVEL[level], li = h("li", cls, ul); li.dataset.level = level;
        const who = !periods.length ? "" : periods.length > 1 && periods.length === all ? "Every period: " : periods.join(", ") + ": ";
        h("b", "", li, `${icon} ${word}`); li.appendChild(document.createTextNode(` ${who}${message}`));
      }
      if (rows.length > limit) h("li", "hint", ul, `…and ${rows.length - limit} more ${level === "info" ? "notes" : "of this kind"}.`);
    }
    const derived = Object.entries(p.derived || {}).filter(([, codes]) => codes.length);
    if (p.ok && derived.length) h("p", "hint", card, "Filled in from the figures you gave: " + derived.map(([label, codes]) => `${label}: ${codes.map((c) => human(c).toLowerCase()).join(", ")}`).join("; ") + ".");
  };
  const FORMAT = {
    ratio: (v) => num(v, 3), currency: (v) => compact(v), days: (v) => num(v, 1), per_share: (v) => num(v, 2), score: (v) => (Number.isInteger(v) ? String(v) : num(v, 2)),
  };
  EN.ratios = (root, p) => {
    const r = p.ratios; if (!r || !r.columns.length) return;
    const card = W.card(root, "Ratios and fundamentals", `${r.computed} of the ${r.defined} ratios, scores and growth rates the platform defines could be computed from these figures. A ratio is computed only when every figure it uses is present, so a short list usually means a missing line, not a bad company. Each is the same definition the factors and the Company explorer use.`);
    card.dataset.role = "manual-ratios"; card.classList.add("roomy");
    const st = FD.en; st.family = st.family || "all";
    const pick = Q.labField(card, { key: "family", label: "Show", kind: "choice", options: [["all", "Every family"]].concat(r.families.map((f) => [f, family(f)])) }, st, () => draw());
    pick.dataset.role = "manual-family";
    const host = h("div", "", card);
    const draw = () => {
      host.replaceChildren();
      const rows = r.rows.filter((x) => st.family === "all" || x.family === st.family);
      const cols = [{ label: "Family", get: (x) => family(x.family) }, { label: "Metric", title: "Hover a metric for its definition", get: (x) => x.label }]
        .concat(r.columns.map((c, i) => ({ label: c.label, num: true, title: c.period_end, get: (x) => (x.values[i] === null || x.values[i] === undefined ? "" : (FORMAT[x.unit] || FORMAT.ratio)(x.values[i])) })));
      table(host, cols, rows);
      host.querySelectorAll("tbody tr").forEach((tr, i) => { if (rows[i]) tr.title = rows[i].text; });
    };
    draw();
    FD.notes(card, "How to read it", ["These are the numbers the factor library and the screens are built from. A percentage-like ratio is shown as a fraction (0.433 is 43.3%).",
      "Growth, changes and averages need the year before: the first period you enter has none, unless the company's earlier years are already in the database."]);
  };
  EN.multiples = (root, p) => {
    const m = p.multiples || []; if (!m.length && !p.multiples_note) return;
    const card = W.card(root, "Valuation at the share price you typed", "Yields are what the factor library uses (a loss or a tiny denominator gives a missing value, not a huge one); the familiar multiples are beside them. The price is not stored.");
    card.dataset.role = "manual-multiples"; card.classList.add("roomy");
    if (p.multiples_note) h("p", "hint", card, p.multiples_note);
    const ok = m.filter((x) => !x.error);
    for (const x of m.filter((y) => y.error)) h("p", "tone-warn", card, `${x.label}: ${x.error}.`);
    if (ok.length) {
      const fmt = (row, v) => (v === null || v === undefined ? "" : row.unit === "yield" ? pct(v, 2) : row.unit === "multiple" ? num(v, 1) + "×" : compact(v));
      const names = ok[0].rows.map((x) => x);
      table(card, [{ label: "Measure", get: (x) => x.label }].concat(ok.map((x, i) => ({ label: `${x.label} at ${num(x.price, 2)}`, num: true, get: (row) => fmt(row, ok[i].rows.find((y) => y.name === row.name).value) }))), names);
    }
    if (p.valuation_note) h("p", "hint", card, p.valuation_note);
    if (p.last_close) h("p", "hint", card, `The last close in your database is ${num(p.last_close.close, 2)} (${p.last_close.date}).`);
  };
  EN.overlapTable = (parent, list) => {
    h("p", "hint", parent, "Where both hold a period, the provider's figures lead. Lines only you have fill its gaps in studies only if your report was public no later than the provider's: a figure published later cannot be used as if it had been known then. A large difference is worth a look at the filing.");
    table(parent, [{ label: "Period", get: (x) => x.label }, { label: "Provider", get: (x) => x.provider }, { label: "Lines compared", num: true, get: (x) => String(x.compared) },
      { label: "Differing by more than 0.5%", num: true, get: (x) => String(x.differ_count) }, { label: "Largest differences", get: (x) => x.differ.slice(0, 3).map((d) => `${d.label}: ${compact(d.mine)} (yours) against ${compact(d.theirs)}`).join("; ") },
      { label: "Lines only you have", num: true, get: (x) => String(x.only_mine) },
      { label: "Used in studies", title: "Whether your report was public no later than the provider's", get: (x) => (x.used_in_studies === null || x.used_in_studies === undefined ? "n/a" : x.used_in_studies ? `yes (public ${x.mine_public}, theirs ${x.theirs_public})` : `no (public ${x.mine_public}, theirs ${x.theirs_public})`) }], list);
  };
  EN.overlaps = (root, list) => {
    if (!list || !list.length) return;
    const card = W.card(root, "Where a provider also reports these periods"); card.dataset.role = "manual-overlaps"; card.classList.add("roomy");
    EN.overlapTable(card, list);
  };
  /* The Company explorer's own view of the company, drawn from the preview: what it will show once the figures are saved. */
  EN.company = (root, p) => {
    if (!p.company) return;
    const co = FD.views.companies;
    h("h2", "sectionhead", root, "As the Company explorer will show it").dataset.role = "manual-company";
    h("p", "hint", root, "The same charts, DuPont split, Piotroski and anomaly screens, Altman and Graham tests, drawn from these figures together with whatever the database already holds for the company.");
    const holder = h("div", "stack", root);
    co.view(holder, p.company, { preview: true, valuation: p.valuation });
  };
})();
