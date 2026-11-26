/* The research gates of one experiment, explained and made actionable.

   Three cards (Algorithm Verified, Strategy Validated, Operationally Ready). Each says what the gate is for in one sentence, its status, how much of its evidence is met, the exact obstacle and KIND of obstacle
   (missing evidence, could not run, failed validation, stale, policy, contaminated holdout), when it was last verified, and the one next action. The action is a button only when the server offers one: the
   page calls the registered verification endpoint with requirement ids, never a path or a command. Technical detail (every check, its tests, counts, source fingerprint, failures and history) opens on demand.
   Nothing here decides a status: the server computes it from the persisted evidence. Whatever the server sends is written with textContent. */
(() => {
  "use strict";
  const Q = window.QL, h = Q.h, api = Q.api;
  const STATUS = { passed: "Passed", failed: "Failed", pending: "Pending", blocked: "Blocked", stale: "Stale", running: "Running", not_applicable: "Not applicable" };
  const KIND = { missing_evidence: "Missing evidence", execution_failure: "Could not run", validation_failure: "Failed validation", stale_evidence: "Stale evidence", policy_restriction: "Policy restriction",
    contaminated_evidence: "Contaminated holdout", insufficient_evidence: "Insufficient evidence" };
  const KIND_HELP = { missing_evidence: "Nothing has produced this evidence yet.", execution_failure: "The check could not run to a verdict here (timed out, tests skipped, runner or files missing).",
    validation_failure: "The check ran and the result does not meet the requirement.", stale_evidence: "It was met, but what it checked has changed since.", policy_restriction: "A rule, or a decision nobody has made yet, blocks it.",
    contaminated_evidence: "The protected outcomes were read outside a frozen plan. This cannot be undone.", insufficient_evidence: "It ran, but there is too little of it to conclude." };
  const GATES = ["A", "B", "C"];
  let LAST = null;                                                                           // the outcome of the last verification this page started: it survives the re-draw that follows it

  function badge(parent, status) { const b = h("span", "gate-badge", parent, STATUS[status] || status); b.dataset.status = status; return b; }
  function seconds(v) { return v === null || v === undefined ? "n/a" : v < 90 ? Math.round(v) + " s" : (v / 60).toFixed(1) + " min"; }

  function render(root, d, ctx) {
    const wrap = h("section", "gates", root); wrap.dataset.role = "gates";
    const head = h("div", "gates-head", wrap); h("h3", "", head, "Research gates");
    const info = h("button", "btn small", head, "What are the gates?"); info.type = "button"; info.setAttribute("aria-expanded", "false"); info.dataset.role = "gates-info";
    const explain = h("div", "gates-explain", wrap); explain.hidden = true; explain.dataset.role = "gates-explainer";
    for (const n of GATES) {
      const g = d.gates[n], box = h("div", "gates-explain-item", explain); h("b", "", box, `Gate ${n}: ${g.title || n}. `); box.appendChild(document.createTextNode(g.purpose || ""));
      if (g.evaluates) h("div", "hint", box, "Looks at: " + g.evaluates.join("; ") + ".");
      if (g.passing_means) h("div", "hint", box, "Passing means: " + g.passing_means);
      if (g.not_implied) h("div", "hint", box, "It does not mean: " + g.not_implied);
    }
    info.addEventListener("click", () => { explain.hidden = !explain.hidden; info.setAttribute("aria-expanded", String(!explain.hidden)); });
    const grid = h("div", "gates-grid", wrap), panel = h("div", "gates-panel", wrap); panel.dataset.role = "gates-panel";
    const progress = progressBox(wrap, d, ctx);
    for (const n of GATES) card(grid, n, d.gates[n], d, ctx, panel, progress);
    toolbar(wrap, d, ctx, progress);
    if (d.verification_runner && d.verification_runner.live) progress.follow();
  }

  function card(grid, n, g, d, ctx, panel, progress) {
    const c = h("div", "gatecard", grid); c.dataset.gate = n; c.dataset.status = g.display_status || g.status; c.dataset.role = "gate-card";
    const top = h("div", "gatecard-top", c); h("span", "gatecard-name", top, `Gate ${n} · ${g.title || ""}`); badge(top, g.display_status || g.status);
    h("p", "gatecard-purpose", c, g.purpose || "");
    h("p", "gatecard-evidence", c, g.evidence_summary || "");
    if (g.blocker) {
      const b = h("div", "gatecard-blocker", c); b.dataset.kind = g.blocker.kind;
      const k = h("span", "kindtag", b, KIND[g.blocker.kind] || g.blocker.kind); k.title = KIND_HELP[g.blocker.kind] || "";
      h("span", "", b, " " + (g.blocker.text || "").slice(0, 260) + ((g.blocker.text || "").length > 260 ? "…" : ""));
    }
    h("p", "hint gatecard-last", c, g.last_verified ? `Last verified ${g.last_verified.ago} (${g.last_verified.requirement}).` : n === "A" ? "Never verified." : "");
    const act = g.next_action || {}, row = h("div", "gatecard-action", c);
    if (act.endpoint && act.body) {
      const b = h("button", "btn primary", row, act.label); b.type = "button"; b.dataset.action = act.id;
      b.addEventListener("click", () => progress.run(act.body, b));
    } else if (act.opens === "plan") {
      const b = h("button", "btn primary", row, act.label); b.type = "button"; b.dataset.action = act.id; b.addEventListener("click", () => readiness(panel, "/api/ml/plan-readiness", d.experiment_id, "plan"));
    } else if (act.opens === "prospective") {
      const b = h("button", "btn primary", row, act.label); b.type = "button"; b.dataset.action = act.id; b.addEventListener("click", () => readiness(panel, "/api/ml/prospective-readiness", d.experiment_id, "prospective"));
    } else if (act.opens === "policy") {
      const b = h("button", "btn", row, act.label); b.type = "button"; b.dataset.action = act.id; b.addEventListener("click", () => { ctx.openMode && ctx.openMode("policy"); });
    } else { h("span", "gatecard-next", row, act.label || ""); }
    if (act.detail) h("p", "hint", c, act.detail);
    const det = h("details", "how", c); det.dataset.role = "gate-details"; h("summary", "", det, "Technical details");
    details(det, n, g, d, ctx, panel, progress);
  }

  function details(det, n, g, d, ctx, panel, progress) {
    h("p", "hint", det, `Policy ${g.policy_version}; evidence hash ${String(g.evidence_hash || "").slice(0, 12)}${g.stale ? "; some evidence is stale" : ""}${g.dependencies.length ? `; needs Gate ${g.dependencies.join(" and ")}` : ""}.`);
    const t = h("table", "tbl gate-items", det), tb = h("tbody", "", t);
    const hr = h("tr", "", h("thead", "", t));
    for (const x of ["Evidence", "State", "Kind", "Why", ""]) h("th", "", hr, x);
    for (const i of g.items) {
      const r = h("tr", "", tb); r.dataset.item = i.id; r.dataset.state = i.status;
      h("td", "", r, i.label); h("td", "", r, i.status === "ok" ? "met" : i.status === "missing" ? "not yet produced" : i.status.replace(/_/g, " ")); h("td", "", r, i.kind ? KIND[i.kind] : "");
      const why = h("td", "", r, i.reason || "");
      if (i.verification) {
        const v = i.verification, c = v.counts;
        h("div", "hint", why, `${v.tests.join(", ")}${c ? ` · ${c.passed} passed, ${c.failed + c.errors} failed, ${c.skipped} skipped` : ""}${v.seconds ? ` · ${seconds(v.seconds)}` : ""}${v.source_fingerprint ? ` · source ${v.source_fingerprint}` : ""}${v.created ? ` · ${v.created}` : ""}`);
        for (const f of (v.failure_details || []).slice(0, 3)) { const fd = h("details", "how", why); h("summary", "", fd, `${f.test}: ${(f.message || "").slice(0, 120)}`); h("pre", "trace", fd, f.trace || f.message || ""); }
      }
      const cell = h("td", "", r);
      if (i.action) { const b = h("button", "btn small", cell, i.action.label); b.type = "button"; b.dataset.action = "run_requirement"; b.dataset.requirement = i.requirement_id; b.addEventListener("click", () => progress.run(i.action.body, b)); }
      if (i.requirement_id) { const hb = h("button", "btn small", cell, "History"); hb.type = "button"; hb.dataset.action = "history"; hb.dataset.requirement = i.requirement_id; hb.addEventListener("click", () => history(panel, i.requirement_id)); }
    }
  }

  function toolbar(wrap, d, ctx, progress) {
    const bar = h("div", "gates-toolbar", wrap), A = d.gates.A, failed = A.items.filter((i) => i.requirement_id && (i.status === "failed" || i.status === "blocked")).map((i) => i.requirement_id);
    const can = d.verification_runner && d.verification_runner.can_run;
    const all = h("button", "btn", bar, "Run all pending requirements"); all.type = "button"; all.dataset.action = "run_all_pending"; all.disabled = !can;
    all.title = can ? "Every requirement that is not currently passed, for all model families (can take several minutes)." : "Verification cannot run here: see the Gate A card.";
    all.addEventListener("click", () => progress.run({ scope: "all_pending" }, all));
    const retry = h("button", "btn", bar, `Retry failed verification${failed.length ? ` (${failed.length})` : ""}`); retry.type = "button"; retry.dataset.action = "retry_failed"; retry.disabled = !can || !failed.length;
    retry.addEventListener("click", () => progress.run({ requirements: failed }, retry));
    if (!can && d.verification_runner) for (const b of d.verification_runner.blockers) { const p = h("p", "banner err", wrap); p.dataset.role = "runner-blocker"; p.appendChild(document.createTextNode(`${b.message}. ${b.fix}.`)); }
  }

  /* progress of a running verification: the requirement and file, files done, counts so far, elapsed time; from the server, so a reload still shows it */
  function progressBox(wrap, d, ctx) {
    const box = h("div", "gates-progress", wrap); box.dataset.role = "verify-progress"; box.hidden = true;
    const line = h("div", "gates-progress-line", box), bar = h("div", "bar", box), fill = h("i", "", bar), counts = h("div", "hint", box), result = h("div", "", wrap); result.dataset.role = "verify-result";
    if (LAST) { const div = h("div", LAST.error ? "banner err" : "banner", result); div.dataset.role = LAST.error ? "verify-error" : "verify-summary"; div.textContent = LAST.text; }
    let timer = null;
    async function tick() {
      try {
        const s = await api("/api/ml/verification/status");
        if (!s.live) { stop(); return false; }
        const v = s.live; box.hidden = false;
        line.textContent = `Running ${v.requirement} · ${v.file || "starting"} · file ${v.index || 0} of ${v.total_files} · ${seconds(v.elapsed_seconds)} so far`;
        fill.style.width = `${Math.round(100 * Math.max((v.index || 1) - 1, 0) / Math.max(v.total_files, 1))}%`;
        counts.textContent = `${v.counts.passed} passed, ${v.counts.failed + v.counts.errors} failed, ${v.counts.skipped} skipped so far`;
        return true;
      } catch (e) { return false; }
    }
    function stop() { clearInterval(timer); timer = null; box.hidden = true; }
    function follow() { if (!timer) { timer = setInterval(tick, 1500); tick(); } }
    async function run(body, button) {
      button.disabled = true; result.replaceChildren();
      try {
        const { job } = await api("/api/ml/verification/run", body);
        box.hidden = false; line.textContent = "Starting…"; follow();
        const out = await Q.waitJob(job, () => {});
        stop();
        const rows = out.checks || [];
        LAST = { text: "Verification finished: " + (rows.map((r) => `${r.requirement_id}: ${r.status}`).join(" · ") || "nothing ran."), error: false };
        if (ctx.reload) ctx.reload();
      } catch (e) { stop(); LAST = { text: e.message, error: true }; const div = h("div", "banner err", result); div.dataset.role = "verify-error"; div.textContent = e.message; button.disabled = false; }
    }
    return { run, follow };
  }

  async function history(panel, req) {
    panel.replaceChildren(); const card = h("div", "card", panel); card.dataset.role = "verify-history"; h("h4", "", card, `Verification history: ${req}`);
    try {
      const out = await api(`/api/ml/verification/history?id=${encodeURIComponent(req)}`);
      h("p", "hint", card, `Tests: ${out.tests.join(", ")}`);
      if (!out.history.length) { h("p", "note", card, "Never run."); return; }
      Q.simpleTable(card, [{ label: "When", get: (r) => r.created }, { label: "Status", get: (r) => r.status + (r.current_source ? "" : " (source changed since)") }, { label: "Passed", num: true, get: (r) => (r.counts ? String(r.counts.passed) : "") },
        { label: "Failed", num: true, get: (r) => (r.counts ? String(r.counts.failed + r.counts.errors) : "") }, { label: "Skipped", num: true, get: (r) => (r.counts ? String(r.counts.skipped) : "") }, { label: "Duration", num: true, get: (r) => seconds(r.seconds) },
        { label: "Source", get: (r) => r.source_fingerprint || "" }, { label: "Reason", get: (r) => r.reason || "" }], out.history);
    } catch (e) { h("div", "banner err", card, e.message); }
  }

  async function readiness(panel, path, id, kind) {
    panel.replaceChildren(); const card = h("div", "card", panel); card.dataset.role = kind === "plan" ? "plan-readiness" : "prospective-readiness";
    h("h4", "", card, kind === "plan" ? "Plan a new frozen evaluation" : "Set up prospective monitoring");
    try {
      const r = await api(`${path}?id=${encodeURIComponent(id)}`);
      if (kind === "plan") {
        h("p", "", card, r.consumed_through ? `The history up to ${r.consumed_through} has been read by this experiment or another look. The first date a new look could protect is ${r.first_untouched_date}; ${r.calendar_days_since} calendar day(s) have passed since.` : "No consumed range is recorded in the ledger.");
        if (r.minimum_evidence) h("p", "hint", card, "Policy minimum: " + Object.entries(r.minimum_evidence).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join("; ") + ".");
        else h("p", "hint", card, "No validation policy is adopted, so no minimum amount of evidence is defined yet.");
        const ol = h("ol", "", card); for (const s of r.steps) h("li", "", ol, s);
        if (r.consumed.length) { h("h5", "", card, "Already consumed"); Q.simpleTable(card, [{ label: "Kind", get: (x) => x.kind }, { label: "From", get: (x) => x.start }, { label: "To", get: (x) => x.end }, { label: "Plan", get: (x) => x.plan_id || "" }, { label: "Purpose", get: (x) => x.purpose || "" }], r.consumed); }
      } else {
        h("p", "hint", card, r.version ? `Strategy version ${r.version}.` : "No strategy version is registered for this experiment.");
        const ul = h("ul", "stepslist", card);
        for (const s of r.steps) { const li = h("li", "", ul); li.dataset.done = String(s.done); h("b", "", li, (s.done ? "✓ " : "○ ") + s.step); h("span", "hint", li, " " + s.how); }
        if (r.counts) h("p", "hint", card, "Prospective record: " + Object.entries(r.counts).map(([k, v]) => `${k} ${v}`).join(", ") + ".");
      }
      const w = h("p", "banner", card); w.dataset.role = "cannot-complete"; h("strong", "", w, "What cannot be done now. "); w.appendChild(document.createTextNode(r.cannot_complete_now));
    } catch (e) { h("div", "banner err", card, e.message); }
  }

  window.QLGates = { render };
})();
