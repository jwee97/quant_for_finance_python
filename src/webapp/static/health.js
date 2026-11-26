/* Data health: one place that says what the data cover, whether they can be trusted and whether research may use them, with the controls to act on that.

   The server does every check (src/data/health.py) and keeps the evidence behind each provider capability (src/marketdata/capability_registry.py). This page draws what comes back and never repairs anything:
   a correction is a logged note, the stored data stay as they were. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, toast, download } = Q;
  const root = $("#datahealth");
  const S = { dataset: "platform", report: null, caps: null, series: "", seq: 0, msg: "" };
  const pct = (v) => (v === null || v === undefined ? "n/a" : (v * 100).toFixed(1) + "%");
  const STATUS = { ready: ["Ready", "ok"], ready_with_caveats: ["Ready with caveats", "warn"], blocked: ["Blocked", "err"] };
  const CAP = { verified_live: "Verified live", verified_recorded: "Verified from a recorded response", mock_tested: "Mock tested only", subscription_blocked: "Subscription blocked", unsupported: "Unsupported", not_tested: "Not tested" };

  function chip(parent, text, kind) { const s = h("span", "badge " + (kind || ""), parent, text); return s; }

  async function load() {
    const my = ++S.seq;
    root.replaceChildren();
    const d = h("div", "head", root), l = h("div", "", d); h("h2", "", l, "Data health");
    h("p", "sub", l, "What the data cover, whether they can be trusted and whether research may use them. Nothing here edits stored data: a problem is shown, and a correction is a note that keeps the raw value.");
    controls();
    const out = h("div", "stack", root); out.id = "datahealth-out"; out.setAttribute("aria-live", "polite");
    Q.loading(out, "Checking the data…");
    try { S.report = await api("/api/health/report?dataset=" + S.dataset); } catch (e) { out.replaceChildren(); h("div", "banner err", out, "The check could not be run: " + e.message); retry(out); return; }
    if (my !== S.seq) return;
    out.replaceChildren();
    readiness(out); coverage(out); integrity(out); await capabilities(out); await history(out);
  }

  function controls() {
    const row = h("div", "row", root); row.dataset.role = "health-controls";
    const sel = h("select", "", row); sel.setAttribute("aria-label", "dataset"); for (const [v, t] of [["platform", "Platform snapshot"], ["market", "Market data store"]]) { const o = h("option", "", sel, t); o.value = v; }
    sel.value = S.dataset; sel.addEventListener("change", () => { S.dataset = sel.value; S.series = ""; load(); });
    const btn = (label, role, fn, title) => { const b = h("button", "btn", row, label); b.type = "button"; b.dataset.role = role; if (title) b.title = title; b.addEventListener("click", async () => { b.disabled = true; try { await fn(); } catch (e) { toast(e.message); } b.disabled = false; }); return b; };
    btn("Validate now", "health-validate", validate, "Run every check again and keep a snapshot, so a later run can show whether the data changed");
    btn("Refresh from provider", "health-refresh", refresh, "Bring stored provider series up to date (needs the provider's key)");
    btn("Export findings (CSV)", "health-export", async () => { const x = await api("/api/health/export", { dataset: S.dataset }); download(x.filename, x.text); });
    const status = h("span", "hint", row); status.id = "health-status"; status.setAttribute("role", "status"); status.textContent = S.msg;
  }
  const say = (t) => { S.msg = t; const s = $("#health-status"); if (s) s.textContent = t; };           // kept in S so that it survives the redraw that follows a validation

  async function validate() {
    say("Validating…");
    const { job } = await api("/api/health/validate", { dataset: S.dataset });
    try {
      const v = await Q.waitJob(job, (j) => say(`${j.message || "Validating"}…`), 800);
      const snap = (v.result || v).snapshot || {};
      say(`Done: ${snap.findings ? snap.findings.error + " errors, " + snap.findings.warning + " warnings" : "checked"}${snap.changed_since_last === true ? "; the data changed since the last validation" : snap.changed_since_last === false ? "; the data are unchanged since the last validation" : ""}.`);
      await load();
    } catch (e) { say(""); const r = h("div", "banner err", $("#datahealth-out") || root, "The validation failed: " + e.message); retry(r); }
  }
  function retry(parent) { const b = h("button", "btn", parent, "Retry"); b.type = "button"; b.dataset.role = "health-retry"; b.addEventListener("click", async () => { b.disabled = true; try { await validate(); } catch (e) { toast(e.message); } b.disabled = false; }); }

  async function refresh() {
    if (S.dataset === "platform") { toast("The platform snapshot is versioned with the repository and is not refreshed here."); return; }
    const symbols = prompt("Symbols to bring up to date (comma separated)", ""); if (!symbols) return;
    say("Refreshing…");
    const { job } = await api("/api/health/refresh", { dataset: S.dataset, symbols: symbols.split(/[\s,]+/).filter(Boolean) });
    await Q.waitJob(job, (j) => say(`${j.message}…`), 1000); say("Refreshed."); await load();
  }

  function readiness(out) {
    const r = S.report, card = h("div", "card", out); card.dataset.role = "health-readiness"; h("h3", "", card, "Research readiness");
    h("p", "sub", card, "Can a study use each series? A blocked series says why; a series with caveats can be used if the caveats are acceptable for the question.");
    const k = h("div", "kpis", card); Q.tile(k, "Ready", String(r.counts.ready)); Q.tile(k, "Ready with caveats", String(r.counts.ready_with_caveats)); Q.tile(k, "Blocked", String(r.counts.blocked), { dir: r.counts.blocked ? "down" : "" });
    h("p", "hint", card, `Data version ${r.fingerprint} (a hash of the values; two runs on the same data show the same one). ${r.note || ""}`);
    const rows = r.readiness.filter((x) => x.status !== "ready");
    if (!rows.length) h("p", "empty-note", card, "Every series meets the default requirements (250 observations, 95% of the expected periods, no gap beyond 5, no error).");
    else Q.simpleTable(card, [{ label: "Series", get: (x) => x.series }, { label: "Status", get: (x) => STATUS[x.status][0] }, { label: "Why", get: (x) => [...x.blocking, ...x.caveats].join("; ") }], rows.slice(0, 60));
    sessionReadiness(card);
  }
  async function sessionReadiness(card) {
    const R = Q.research; if (!R || !R.ready || !R.ready() || !R.active || !R.active()) return;
    try {
      const x = await api("/api/health/readiness?session=" + R.active().id); const box = h("div", "", card); box.dataset.role = "health-session-readiness";
      h("h4", "", box, `Instruments of the session "${R.active().name}": ${STATUS[x.overall] ? STATUS[x.overall][0] : "no instruments chosen"}`);
      if (x.rows.length) Q.simpleTable(box, [{ label: "Instrument", get: (r) => r.series }, { label: "Status", get: (r) => STATUS[r.status][0] }, { label: "Why", get: (r) => [...r.blocking, ...r.caveats].join("; ") }], x.rows);
    } catch (e) { h("p", "hint", card, "The session's readiness could not be read: " + e.message); }
  }

  function coverage(out) {
    const r = S.report, card = h("div", "card", out); card.dataset.role = "health-coverage"; h("h3", "", card, "Dataset coverage");
    h("p", "sub", card, "First and last observation, how many there are against how many the calendar expects, and the runs of missing periods. A missing period is never filled.");
    const rows = r.coverage.slice().sort((a, b) => a.completeness - b.completeness);
    const t = Q.simpleTable(card, [{ label: "Series", get: (x) => x.series }, { label: "First", get: (x) => x.first || "" }, { label: "Last", get: (x) => x.last || "" }, { label: "Observations", num: true, get: (x) => String(x.observations) },
      { label: "Expected", num: true, get: (x) => String(x.expected) }, { label: "Complete", num: true, get: (x) => pct(x.completeness) }, { label: "Missing", num: true, get: (x) => String(x.missing_periods) }, { label: "Calendar", get: (x) => x.calendar },
      { label: "", get: (x) => "" }], rows.slice(0, 200));
    const body = card.querySelector("tbody"); if (!body) return;
    [...body.rows].forEach((tr, i) => {
      const x = rows[i], cell = tr.cells[tr.cells.length - 1]; tr.dataset.series = x.series;
      const miss = h("button", "linklike", cell, "Missing periods"); miss.type = "button"; miss.addEventListener("click", () => inspect(x.series, "missing"));
      cell.append(" "); const src = h("button", "linklike", cell, "Source"); src.type = "button"; src.addEventListener("click", () => inspect(x.series, "source"));
    });
    const detail = h("div", "", card); detail.id = "health-detail"; detail.dataset.role = "health-detail";
  }
  async function inspect(series, what) {
    const box = $("#health-detail"); if (!box) return; box.replaceChildren(); S.series = series;
    try {
      const x = await api(`/api/health/${what}?dataset=${S.dataset}&series=${encodeURIComponent(series)}`);
      h("h4", "", box, what === "missing" ? `Missing periods of ${series}: ${x.periods} of ${x.expected} expected (${x.runs.length} runs)` : `Source of ${series}`);
      if (what === "missing") { if (!x.runs.length) h("p", "empty-note", box, "No expected period is missing."); else Q.simpleTable(box, [{ label: "From", get: (r) => r.start }, { label: "To", get: (r) => r.end }, { label: "Periods", num: true, get: (r) => String(r.periods) }], x.runs.slice(0, 200)); }
      else { h("p", "", box, x.origin); const dl = h("dl", "kv", box); for (const [k, v] of Object.entries(x.info || { first: x.first, last: x.last, "asset class": x.asset_class })) { if (v === undefined || v === null || v === "") continue; h("dt", "", dl, k); h("dd", "", dl, String(v)); } if ((x.versions || []).length) { h("h4", "", box, "Versions kept"); Q.simpleTable(box, [{ label: "Version", get: (v) => String(v.version) }, { label: "When", get: (v) => v.at }, { label: "Added", num: true, get: (v) => String(v.added) }, { label: "Revised", num: true, get: (v) => String(v.revised) }, { label: "Reason", get: (v) => v.reason }], x.versions); } }
      correctionForm(box, series);
    } catch (e) { h("div", "banner err", box, e.message); }
  }
  function correctionForm(box, series) {
    const d = h("details", "", box); h("summary", "", d, "Record a correction (the raw value is kept)");
    const f = {}; for (const [k, ph] of [["date", "YYYY-MM-DD"], ["field", "close"], ["raw", "raw value"], ["corrected", "corrected value"], ["reason", "why"]]) { f[k] = h("input", "", d); f[k].placeholder = ph; f[k].setAttribute("aria-label", k); }
    const b = h("button", "btn", d, "Record"); b.type = "button"; b.addEventListener("click", async () => { try { await api("/api/health/correction", { series, date: f.date.value, field: f.field.value || "close", raw: Number(f.raw.value), corrected: Number(f.corrected.value), reason: f.reason.value }); toast("Correction recorded; stored data are unchanged"); history($("#datahealth-out")); } catch (e) { toast(e.message); } });
  }

  function integrity(out) {
    const r = S.report, card = h("div", "card", out); card.dataset.role = "health-integrity"; h("h3", "", card, "Data integrity");
    h("p", "sub", card, "Duplicates, invalid prices, stale runs, long gaps, unexplained jumps and the like. A finding is evidence to look at; nothing is deleted.");
    const counts = { error: 0, warning: 0, info: 0 }; for (const f of r.findings) counts[f.severity]++;
    const k = h("div", "kpis", card); Q.tile(k, "Errors", String(counts.error), { dir: counts.error ? "down" : "" }); Q.tile(k, "Warnings", String(counts.warning)); Q.tile(k, "Notes", String(counts.info));
    if (!r.findings.length) { h("p", "empty-note", card, "No findings."); return; }
    const order = { error: 0, warning: 1, info: 2 };
    Q.simpleTable(card, [{ label: "Severity", get: (f) => f.severity }, { label: "Check", get: (f) => f.check.replace(/_/g, " ") }, { label: "Series", get: (f) => f.series }, { label: "Date", get: (f) => f.date || "" }, { label: "Detail", get: (f) => f.detail }],
      r.findings.slice().sort((a, b) => order[a.severity] - order[b.severity]).slice(0, 300));
  }

  async function capabilities(out) {
    const card = h("div", "card", out); card.dataset.role = "health-capabilities"; h("h3", "", card, "Provider capabilities");
    let c; try { c = await api("/api/health/capabilities"); } catch (e) { h("p", "hint", card, "Could not be read: " + e.message); return; }
    h("p", "sub", card, "Exactly one status per dataset, with the evidence. A mock-tested adapter proves the code's logic, not that the provider still answers; Linux runs say nothing about Windows.");
    const k = h("div", "kpis", card); for (const s of c.statuses) Q.tile(k, CAP[s], String(c.counts[s]));
    Q.simpleTable(card, [{ label: "Provider", get: (x) => x.provider }, { label: "Dataset", get: (x) => x.dataset }, { label: "Status", get: (x) => CAP[x.status] + (x.stale ? " (older than 30 days)" : "") }, { label: "Evidence", get: (x) => x.evidence + (x.credential && !x.credential_present ? ` Needs ${x.credential}, which this machine does not hold.` : "") }], c.rows);
  }

  async function history(out) {
    out = out || $("#datahealth-out"); if (!out) return; const old = out.querySelector("[data-role=health-history]"); if (old) old.remove();
    const card = h("div", "card", out); card.dataset.role = "health-history"; h("h3", "", card, "Validation snapshots and corrections");
    try {
      const [s, c] = await Promise.all([api("/api/health/snapshots"), api("/api/health/corrections")]);
      if (!s.snapshots.length) h("p", "empty-note", card, "No validation has been kept yet. Choose Validate now."); else Q.simpleTable(card, [{ label: "When", get: (x) => x.at }, { label: "Dataset", get: (x) => x.dataset }, { label: "Version", get: (x) => x.fingerprint }, { label: "Series", num: true, get: (x) => String(x.series) }, { label: "Blocked", num: true, get: (x) => String(x.counts.blocked) }, { label: "Errors", num: true, get: (x) => String(x.findings.error) }], s.snapshots.slice(0, 20));
      if (c.corrections.length) { h("h4", "", card, "Corrections"); Q.simpleTable(card, [{ label: "When", get: (x) => x.at }, { label: "Series", get: (x) => x.series }, { label: "Date", get: (x) => x.date }, { label: "Raw", get: (x) => String(x.raw) }, { label: "Corrected", get: (x) => String(x.corrected) }, { label: "Reason", get: (x) => x.reason }], c.corrections.slice(-20).reverse()); }
    } catch (e) { h("p", "hint", card, e.message); }
  }

  Q.registerPage("datahealth", { render: () => load() });
})();
