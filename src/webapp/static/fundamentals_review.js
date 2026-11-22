/* The Report import review as a five-step wizard: 1 filing details, 2 review statements, 3 correct and classify, 4 warnings, 5 final approval.

   The scanner discovers, the person confirms, the application stores only what was approved: every observation starts UNCONFIRMED, a decision (confirm, correct, exclude, unresolved) is the person's,
   and the scanner's confidence is shown as a hint and never confirms anything. Steps 2 and 3 share one table of every figure the scanner found (mapped or not), grouped by statement, searchable,
   sortable, with the stored PDF page and the figure's box beside it. Step 5 lists exactly what an approval will commit. /api/fund/import/{observations,decide,plan,source}.
   Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const FD = window.FD, Q = window.QL;
  if (!FD || !Q || !FD.views || !FD.views.import) return;
  const h = Q.h, IM = FD.views.import;
  const STEPS = [["1", "Filing details"], ["2", "Review statements"], ["3", "Correct and classify"], ["4", "Warnings"], ["5", "Final approval"]];
  const GROUPS = [["all", "All"], ["income_statement", "Income statement"], ["balance_sheet", "Balance sheet"], ["cash_flow", "Cash flow"], ["equity", "Equity"], ["segment", "Segments"], ["notes", "Notes"],
                  ["kpi", "Operating KPIs"], ["summary", "Summary tables"], ["other", "Other"], ["unmapped", "Unmapped"]];
  const STATUS = [["", "Any decision"], ["UNCONFIRMED", "Unconfirmed"], ["CONFIRMED", "Confirmed"], ["CORRECTED_AND_CONFIRMED", "Corrected and confirmed"], ["EXCLUDED", "Excluded"], ["UNRESOLVED", "Unresolved"]];
  const TONE = { UNCONFIRMED: "hint", CONFIRMED: "tone-good", CORRECTED_AND_CONFIRMED: "tone-good", EXCLUDED: "tone-crit", UNRESOLVED: "tone-warn" };
  const UNITS = ["units", "thousands", "millions", "billions", "percent"];
  const fmt = (v) => (v === null || v === undefined ? "" : Math.abs(v) >= 1e4 ? v.toLocaleString("en-US", { maximumFractionDigits: 6 }) : String(v));
  const state = () => (IM.s.wiz = IM.s.wiz || { step: 1, group: "all", status: "", q: "", sort: "page", desc: false, offset: 0, sel: new Set(), edit: null, source: null });

  /* ---------- the stepper: five containers, one visible ---------- */
  IM.stepper = (card, repaint) => {
    const w = state(), bar = h("div", "row tight", card), panes = {};
    bar.dataset.role = "import-steps"; bar.setAttribute("role", "tablist");
    const btns = {};
    for (const [n, label] of STEPS) {
      const b = h("button", "btn small", bar, `${n}. ${label}`); b.type = "button"; b.dataset.step = n; b.setAttribute("role", "tab");
      b.addEventListener("click", () => { w.step = +n; show(); if (repaint) repaint(+n); });
      btns[n] = b; panes[n] = h("div", "wiz-pane", card); panes[n].dataset.pane = n;
    }
    const show = () => { for (const [n] of STEPS) { const on = +n === w.step; panes[n].hidden = !on; btns[n].classList.toggle("primary", on); btns[n].setAttribute("aria-selected", on ? "true" : "false"); } };
    show();
    panes.go = (n) => { w.step = n; show(); if (repaint) repaint(n); };
    return panes;
  };

  /* ---------- steps 2 and 3: every figure the scanner found ---------- */
  IM.observationsPanel = (box, it, edit) => {
    const w = state(); box.textContent = "";
    const top = h("div", "row tight", box);
    const g = h("select", "", h("label", "f", top, "Show")); g.dataset.role = "obs-group"; g.setAttribute("aria-label", "statement group");
    const st = h("select", "", h("label", "f", top, "Decision")); st.dataset.role = "obs-status"; st.setAttribute("aria-label", "decision filter");
    const q = h("input", "", h("label", "f", top, "Search")); q.type = "search"; q.value = w.q; q.dataset.role = "obs-search"; q.placeholder = "label, value or column heading"; q.setAttribute("aria-label", "search observations");
    const so = h("select", "", h("label", "f", top, "Sort")); so.dataset.role = "obs-sort"; so.setAttribute("aria-label", "sort");
    for (const [v, l] of [["page", "Page"], ["label", "Label"], ["value", "Size of value"], ["period", "Period"], ["status", "Decision"], ["group", "Statement"]]) { const o = h("option", "", so, l); o.value = v; if (v === w.sort) o.selected = true; }
    for (const [v, l] of STATUS) { const o = h("option", "", st, l); o.value = v; if (v === w.status) o.selected = true; }
    const chips = h("div", "row tight", box); chips.dataset.role = "obs-counts";
    const wrap = h("div", "scroll", box), meta = h("p", "hint", box), pager = h("div", "row tight", box), src = h("div", "obs-source", box); src.dataset.role = "obs-source";
    const bulk = edit ? h("div", "row tight", box) : null;
    let data = null;
    const load = async () => {
      const params = { id: it.item_id, item_id: it.item_id, group: w.group === "all" ? undefined : w.group, status: w.status || undefined, q: w.q || undefined, sort: w.sort, desc: w.desc ? "1" : undefined, limit: 100, offset: w.offset };
      data = await FD.read("/api/fund/import/observations", params); paint();
    };
    const paint = () => {
      g.textContent = "";
      for (const [v, l] of GROUPS) { const n = (data.by_group || {})[v]; const o = h("option", "", g, `${l}${v === "all" ? "" : ` (${n || 0})`}`); o.value = v; if (v === w.group) o.selected = true; }
      chips.textContent = "";
      const total = Object.values(data.by_status).reduce((a, b) => a + b, 0);
      h("span", "chip", chips, `${total} found`);
      for (const k of ["UNCONFIRMED", "CONFIRMED", "CORRECTED_AND_CONFIRMED", "EXCLUDED", "UNRESOLVED"]) { const c = h("span", "chip " + TONE[k], chips, `${data.by_status[k] || 0} ${k.toLowerCase().replace(/_/g, " ")}`); c.dataset.status = k; }
      wrap.textContent = "";
      const t = h("table", "data review obs", wrap), hd = h("tr", "", h("thead", "", t)); t.dataset.role = "obs-table";
      const cols = ["", "Label", "Value", "Unit", "Currency", "Period", "Column", "Statement", "Line", "Page", "Decision"];
      cols.forEach((c, i) => { const th = h("th", i === 2 ? "num" : "", hd); if (i === 0 && edit) { const all = h("input", "", th); all.type = "checkbox"; all.setAttribute("aria-label", "select every row shown"); all.addEventListener("change", () => { for (const o of data.rows) all.checked ? w.sel.add(o.obs_id) : w.sel.delete(o.obs_id); paint(); }); } else th.textContent = c; });
      const tb = h("tbody", "", t);
      for (const o of data.rows) {
        const e = o.effective, tr = h("tr", o.status === "EXCLUDED" ? "dim" : "", tb); tr.dataset.obs = o.obs_id; tr.dataset.status = o.status;
        const c0 = h("td", "", tr); if (edit) { const cb = h("input", "", c0); cb.type = "checkbox"; cb.checked = w.sel.has(o.obs_id); cb.setAttribute("aria-label", `select ${o.label}`); cb.addEventListener("change", () => { cb.checked ? w.sel.add(o.obs_id) : w.sel.delete(o.obs_id); }); }
        const lab = h("td", "", tr, e.label); lab.title = o.original_label || o.label; if (o.origin === "manual") h("span", "chip", lab, "added");
        h("td", "num", tr, fmt(e.value) + (o.is_percent ? "%" : "")).title = o.value_text || "";
        h("td", "", tr, e.unit || ""); h("td", "", tr, e.currency || "");
        const pe = h("td", "", tr, e.period_end || "?"); if (o.period_guessed && !o.user_period_end) { pe.classList.add("tone-warn"); pe.title = "this period was inferred, not read"; pe.textContent += " ≈"; }
        const roleCell = h("td", "hint", tr, `${(o.column_role || "").replace(/_/g, " ").toLowerCase()}${o.column_header ? " · " + o.column_header : ""}`);
        roleCell.title = o.column_header || "";
        h("td", "", tr, (o.grp || "").replace(/_/g, " "));
        const mp = h("td", "", tr, e.canonical ? FD.fmt.human(e.canonical) : e.metric ? `custom: ${e.metric}` : "unmapped"); if (!e.canonical) mp.classList.add("hint");
        const pg = h("td", "", tr); if (o.page && IM.s.itemData && IM.s.itemData.source_kind === "pdf") { const b = h("button", "btn small", pg, `p.${o.page}`); b.type = "button"; b.title = "show the page with this figure highlighted"; b.dataset.action = "obs-source"; b.addEventListener("click", () => showSource(o)); } else pg.textContent = o.page || "";
        const dc = h("td", "", tr), badge = h("span", "chip " + (TONE[o.status] || ""), dc, o.status.toLowerCase().replace(/_/g, " ")); badge.dataset.role = "obs-status-badge";
        if (edit) { const b = h("button", "btn small", dc, "Edit"); b.type = "button"; b.dataset.action = "obs-edit"; b.addEventListener("click", () => { w.edit = o.obs_id; paintEditor(o); }); }
      }
      meta.textContent = `${data.total} observation${data.total === 1 ? "" : "s"} match; showing ${Math.min(data.rows.length, 100)} from ${w.offset + 1}. Confidence never confirms a figure: only your decision does.`;
      pager.textContent = "";
      const prev = h("button", "btn small", pager, "Previous"); prev.type = "button"; prev.disabled = w.offset === 0; prev.addEventListener("click", () => { w.offset = Math.max(0, w.offset - 100); load(); });
      const next = h("button", "btn small", pager, "Next"); next.type = "button"; next.disabled = w.offset + 100 >= data.total; next.addEventListener("click", () => { w.offset += 100; load(); });
      const dir = h("button", "btn small", pager, w.desc ? "Descending" : "Ascending"); dir.type = "button"; dir.addEventListener("click", () => { w.desc = !w.desc; load(); });
      if (bulk) paintBulk();
    };
    const decide = async (actions, note) => {
      try { await FD.send("/api/fund/import/decide", { item_id: it.item_id, actions }); w.sel.clear(); await load(); if (note) Q.toast(note); } catch (e) { Q.toast(e.message); }
    };
    const paintBulk = () => {
      bulk.textContent = "";
      const mk = (label, role, fn, title) => { const b = h("button", "btn small", bulk, label); b.type = "button"; b.dataset.action = role; if (title) b.title = title; b.addEventListener("click", fn); return b; };
      mk("Confirm selected", "obs-confirm-selected", () => w.sel.size ? decide([{ action: "confirm", ids: [...w.sel], accept_period: false }]) : Q.toast("Select rows first"));
      mk("Exclude selected", "obs-exclude-selected", () => w.sel.size ? decide([{ action: "exclude", ids: [...w.sel] }]) : Q.toast("Select rows first"));
      mk("Mark selected unresolved", "obs-unresolve-selected", () => w.sel.size ? decide([{ action: "unresolve", ids: [...w.sel] }]) : Q.toast("Select rows first"));
      mk("Confirm everything the scanner chose for a line", "obs-confirm-suggested", () => decide([{ action: "confirm_suggested" }], "Confirmed the scanner's suggestions: your decision, recorded"),
         "An explicit bulk decision: confirm after checking the figures against the page. Inferred periods must be accepted first (Step 1).");
      mk("Add a figure the scanner missed", "obs-add", () => paintAdd());
    };
    const field = (parent, label, type, value, role, opts) => {
      const l = h("label", "f", parent, label); let el;
      if (opts) { el = h("select", "", l); for (const [v, t] of opts) { const o = h("option", "", el, t); o.value = v; if (String(v) === String(value)) o.selected = true; } } else { el = h("input", "", l); el.type = type; el.value = value === null || value === undefined ? "" : value; }
      el.dataset.role = role; el.setAttribute("aria-label", label); return el;
    };
    const paintEditor = (o) => {
      src.textContent = ""; const ed = h("div", "card inner", src); ed.dataset.role = "obs-editor"; const e = o.effective;
      h("h4", "", ed, `${o.label} (page ${o.page || "?"}${o.column_header ? ", column “" + o.column_header + "”" : ""})`);
      h("p", "hint", ed, `As read: ${o.value_text || o.value} ${o.unit || ""} ${o.currency || ""}, ${o.period_end || "no period"}${o.period_guessed ? " (inferred)" : ""}. Scanner suggestion: ${o.metric_guess ? FD.fmt.human(o.metric_guess) : "none"}.${o.guess_reason ? " " + o.guess_reason : ""}`);
      const row = h("div", "row tight", ed);
      const val = field(row, "Value", "text", e.value, "obs-value"), cur = field(row, "Currency", "text", e.currency || "", "obs-currency"), un = field(row, "Unit", "text", e.unit || "units", "obs-unit", UNITS.map((u) => [u, u]).concat(UNITS.includes(e.unit) ? [] : [[e.unit, e.unit]]));
      const per = field(row, "Period end", "date", e.period_end || "", "obs-period"), ps = field(row, "Period start", "date", e.period_start || "", "obs-period-start");
      const pt = field(row, "Period type", "text", e.period_type || "", "obs-period-type", [["", "(as read)"], ["ANNUAL", "Annual"], ["QUARTER", "Standalone quarter"], ["HALF_YEAR", "Half year"], ["YTD", "Year to date"], ["INSTANT", "Balance sheet date"]]);
      const sc = field(row, "Scope", "text", e.scope || "", "obs-scope", [["", "(as read)"], ["consolidated", "Consolidated"], ["parent", "Parent only"], ["segment", "Segment"], ["unspecified", "Unspecified"]]);
      const mt = field(row, "Line", "text", e.canonical || "", "obs-metric", [["", "(custom or unmapped)"]].concat(((data && data.metrics) || []).map((m) => [m.code, m.label])));
      const cl = field(row, "Custom name", "text", o.custom_label || (e.canonical ? "" : e.label), "obs-custom-label"), nt = field(row, "Note", "text", o.note || "", "obs-note");
      const acc = h("label", "f", ed, "Accept the inferred period"); const ac = h("input", "", acc); ac.type = "checkbox"; ac.dataset.role = "obs-accept-period"; ac.disabled = !o.period_guessed;
      const act = h("div", "row tight", ed);
      const body = (confirm) => ({ action: "correct", obs_id: o.obs_id, value: val.value, currency: cur.value, unit: un.value, period_end: per.value, period_start: ps.value, period_type: pt.value, scope: sc.value,
                                   metric: mt.value || undefined, custom_label: cl.value || undefined, note: nt.value || undefined, confirm, accept_period: ac.checked });
      const mk = (label, role, fn) => { const b = h("button", "btn small", act, label); b.type = "button"; b.dataset.action = role; b.addEventListener("click", fn); return b; };
      mk("Save correction (stays unconfirmed)", "obs-save", () => decide([body(false)]));
      mk("Save and confirm", "obs-save-confirm", () => decide([body(true)]));
      mk("Confirm as read", "obs-confirm", () => decide([{ action: "confirm", obs_id: o.obs_id, accept_period: ac.checked }]));
      mk("Exclude", "obs-exclude", () => decide([{ action: "exclude", obs_id: o.obs_id, note: nt.value }]));
      mk("Unresolved", "obs-unresolve", () => decide([{ action: "unresolve", obs_id: o.obs_id, note: nt.value }]));
      mk("Reset", "obs-reset", () => decide([{ action: "reset", obs_id: o.obs_id }]));
      if (o.history && o.history.length) { const hist = h("details", "", ed), sm = h("summary", "", hist, `History (${o.history.length})`); for (const x of o.history) h("p", "hint", hist, `${x.at}: ${x.action}${x.note ? " · " + x.note : ""}`); void sm; }
      if (IM.s.itemData && IM.s.itemData.source_kind === "pdf" && o.page) showSource(o, ed);
    };
    const paintAdd = () => {
      src.textContent = ""; const ed = h("div", "card inner", src); ed.dataset.role = "obs-add-form"; h("h4", "", ed, "Add a figure the scanner missed");
      const row = h("div", "row tight", ed);
      const lb = field(row, "Label", "text", "", "add-label"), va = field(row, "Value", "text", "", "add-value"), pe = field(row, "Period end", "date", "", "add-period");
      const cu = field(row, "Currency", "text", (IM.s.itemData.reviewed || IM.s.itemData.draft || {}).currency || "", "add-currency"), un = field(row, "Unit", "text", "units", "add-unit", UNITS.map((u) => [u, u]));
      const gp = field(row, "Statement", "text", "other", "add-group", GROUPS.filter((x) => !["all", "unmapped"].includes(x[0])).map((x) => [x[0], x[1]]));
      const ok = h("button", "btn small", ed, "Add and confirm"); ok.type = "button"; ok.dataset.action = "obs-add-confirm";
      ok.addEventListener("click", () => decide([{ action: "add", label: lb.value, value: va.value, period_end: pe.value, currency: cu.value, unit: un.value, group: gp.value, confirm: true }], "Added"));
    };
    const showSource = async (o, into) => {
      const host = into || src; let pane = host.querySelector("[data-role=source-page]"); if (!pane) pane = h("div", "source-page", host); pane.dataset.role = "source-page"; pane.textContent = "Loading the page…";
      try {
        const r = await FD.read("/api/fund/import/source", { item_id: it.item_id, page: o.page, bbox: o.bbox ? o.bbox.join(",") : undefined });
        pane.textContent = ""; const im = h("img", "page-image", pane); im.alt = `page ${r.page} of ${r.pages}, the figure highlighted`; im.src = "data:image/png;base64," + r.png; im.style.maxWidth = "100%"; h("p", "hint", pane, `Page ${r.page} of ${r.pages}`);
      } catch (e) { pane.textContent = e.message; }
    };
    g.addEventListener("change", () => { w.group = g.value; w.offset = 0; load(); }); st.addEventListener("change", () => { w.status = st.value; w.offset = 0; load(); });
    so.addEventListener("change", () => { w.sort = so.value; load(); });
    let timer = null; q.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { w.q = q.value.trim(); w.offset = 0; load(); }, 250); });
    load().catch((e) => { box.textContent = e.message; });
  };

  /* ---------- step 5: exactly what an approval will commit ---------- */
  IM.planPanel = async (box, it, approveButton) => {
    box.textContent = "";
    let plan; try { plan = await FD.read("/api/fund/import/plan", { item_id: it.item_id }); } catch (e) { box.textContent = e.message; return; }
    const c = plan.counts, row = h("div", "row tight", box); row.dataset.role = "plan-counts";
    for (const [k, l] of [["extracted", "extracted"], ["confirmed", "confirmed"], ["corrected_and_confirmed", "corrected and confirmed"], ["excluded", "excluded"], ["unresolved", "unresolved"], ["unconfirmed", "unconfirmed (stay in the archive)"]]) h("span", "chip", row, `${c[k] || 0} ${l}`);
    h("p", "hint", box, `${plan.will_commit.length} line${plan.will_commit.length === 1 ? "" : "s"} for the canonical tables and ${plan.will_commit_custom.length} custom figure${plan.will_commit_custom.length === 1 ? "" : "s"} will be committed. Nothing else enters research, factors or Alpha Lab.`);
    for (const x of plan.cannot_commit) h("p", "tone-crit", box, "✗ " + x.reason);
    for (const x of plan.warnings.slice(0, 20)) h("p", x.level === "error" ? "tone-crit" : "tone-warn", box, (x.level === "error" ? "✗ " : "! ") + x.message);
    const wrap = h("div", "scroll", box), t = h("table", "data review", wrap); t.dataset.role = "plan-table";
    const hd = h("tr", "", h("thead", "", t)); for (const x of ["Line", "Period", "Value", "Currency", "Unit", ""]) h("th", "", hd, x);
    const tb = h("tbody", "", t);
    for (const l of plan.will_commit) { const tr = h("tr", "", tb); h("td", "", tr, l.label); h("td", "", tr, l.period); h("td", "num", tr, fmt(l.value)); h("td", "", tr, l.currency || ""); h("td", "", tr, l.unit || ""); h("td", "hint", tr, l.corrected ? "corrected" : ""); }
    for (const l of plan.will_commit_custom) { const tr = h("tr", "", tb); h("td", "", tr, l.label + " (custom)"); h("td", "", tr, l.period_end || ""); h("td", "num", tr, fmt(l.value)); h("td", "", tr, l.currency || ""); h("td", "", tr, l.unit || ""); h("td", "hint", tr, l.why); }
    const ack = h("input", "", h("label", "f", box, "I have checked these figures and approve storing them")); ack.type = "checkbox"; ack.dataset.role = "plan-ack";
    const sync = () => { approveButton.disabled = !(ack.checked && plan.approvable) || it.status === "saved"; }; ack.addEventListener("change", sync); sync();
  };
})();
