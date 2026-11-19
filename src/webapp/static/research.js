/* The shared research context in the page: which session is open, what its context says, and the actions that move a result from one module to the next.

   The server keeps the sessions (data/user/research); this file keeps only the id of the open one (ql.session) and a copy of its record. A module never reads another module's state: it reads the
   context from here, and a result it wants to hand on is saved to the session first (the server stores it from its own record of the run) and then opened from there.

   Sharing a context changes what the *next* run uses. It never edits a result that is already on screen or saved: when the context no longer describes a saved result, the page says so and offers to
   run it again under the new context, and nothing is re-run unasked. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, store, toast, api } = Q;
  const bar = $("#ctxbar");
  const listeners = [];
  let sessions = [], meta = { sources: [], asset_classes: [] }, active = null, loaded = false;

  const R = Q.research = {
    newSession: () => editDialog(true),
    active: () => active, sessions: () => sessions, meta: () => meta, ready: () => loaded, onChange: (fn) => listeners.push(fn),
    /* ---- the list and the open session */
    async refresh() {
      const d = await api("/api/research/sessions"); sessions = d.sessions; meta = { sources: d.sources, asset_classes: d.asset_classes, limits: d.limits };
      const id = store("session");
      if (id && sessions.some((s) => s.id === id && s.readable)) { try { active = (await api("/api/research/session?id=" + id)).session; } catch (e) { active = null; } } else active = null;
      if (!active && id) store("session", null);
      loaded = true; draw(); listeners.forEach((fn) => fn(active));
    },
    async open(id) {
      if (!id) { active = null; store("session", null); draw(); listeners.forEach((fn) => fn(null)); return null; }
      active = (await api("/api/research/session?id=" + id)).session; store("session", id); await R.refresh(); return active;
    },
    async create(name, context) {
      const s = (await api("/api/research/session/create", { name, context })).session; store("session", s.id); await R.refresh(); return active;
    },
    async setContext(context) {
      if (!active) throw new Error("open or create a research session first");
      active = (await api("/api/research/session/update", { id: active.id, context })).session; await R.refresh(); return active;
    },
    /* The open session, or a new one made from this page's current setup (so that a person who just clicks Save gets a session without a detour). */
    async ensure(name) {
      if (active) return active;
      const ctx = Q.setupContext ? Q.setupContext() : {};
      return R.create(name || "My research", Object.fromEntries(Object.entries(ctx).filter(([, v]) => v !== "" && !(Array.isArray(v) && !v.length))));
    },
    async saveRun(req, label) {
      const s = await R.ensure(label);
      const art = (await api("/api/research/artifact/save", { session: s.id, ...req })).artifact;
      toast(`Saved to "${s.name}"`); await R.refresh(); return art;
    },
    /* Apply the open session's context to a tab's own form. Only the next run is affected. */
    applyTo(tab) {
      if (!active) { toast("Open a research session first"); return false; }
      const c = active.context || {};
      if (tab === "backtest") return Q.applyContext(c);
      const lab = Q.lab(tab);
      if (!lab) return false;
      if (tab === "options") {
        const o = c.options || {}, cat = lab.catalog;
        if (c.assets && c.assets.length && cat && cat.underlyings.some((u) => u.id === c.assets[0])) { lab.form.underlying = c.assets[0]; lab.form.expiry = ""; lab.spot = null; }
        if (o.rate !== undefined) lab.form.rate = String(+(o.rate * 100).toFixed(4));
        if (o.div_yield !== undefined) lab.form.dividend = String(+(o.div_yield * 100).toFixed(4));
        Q.renderLab(lab); return true;
      }
      if (tab === "ml") {
        if (c.assets && c.assets.length >= 2) { lab.form.universe = "custom"; lab.form.tickers = c.assets.join(", "); }
        if (c.start) lab.form.start = c.start;
        if (c.end) lab.form.end = c.end;
        Q.renderLab(lab); return true;
      }
      return false;
    },
  };

  /* ---------------------------------------------------------------- the bar */
  const NO_BAR = new Set(["guides", "settings", "data", "jobs"]);
  const short = (s) => { const c = s.context || {}; const a = c.assets || s.assets || []; return `${a.length ? (a.length > 4 ? a.slice(0, 4).join(", ") + ` +${a.length - 4}` : a.join(", ")) : "no assets yet"} · ${c.start || s.start || "first day"} to ${c.end || s.end || "today"}${c.data_source ? " · " + c.data_source : ""}`; };
  function draw() {
    bar.replaceChildren();
    if (!loaded) { bar.hidden = true; return; }
    bar.hidden = NO_BAR.has(Q.currentTab());
    const row = h("div", "row", bar), lab = h("label", "ctxlabel", row, "Research session"); lab.setAttribute("for", "ctx-select");
    const sel = h("select", "", row); sel.id = "ctx-select";
    const none = h("option", "", sel, "None (each tab on its own)"); none.value = "";
    for (const s of sessions.filter((x) => x.readable)) { const o = h("option", "", sel, s.name); o.value = s.id; if (active && s.id === active.id) o.selected = true; }
    sel.addEventListener("change", () => R.open(sel.value).catch((e) => toast(e.message)));
    h("span", "ctxsummary", row, active ? short(active) : "Choose or start a session to share tickers, dates and assumptions between tabs.");
    const btns = h("span", "row tight", row);
    const mk = (text, fn, title) => { const b = h("button", "btn", btns, text); b.type = "button"; if (title) b.title = title; b.addEventListener("click", fn); return b; };
    mk(active ? "Edit context" : "New session", () => editDialog(!active));
    if (active) {
      const tab = Q.currentTab();
      if (["backtest", "options", "ml"].includes(tab)) mk("Use in this tab", () => { if (R.applyTo(tab)) toast("The next run in this tab uses the session's context"); }, "Fill this tab's form from the session; nothing is run");
      mk("Saved research", () => Q.showTab("saved"));
    }
  }
  Q.onNavigate(() => draw());

  /* ---------------------------------------------------------------- the editor */
  function editDialog(fresh) {
    const d = $("#ctxdlg"); d.replaceChildren();
    const form = h("form", "pal", d); form.method = "dialog";
    h("p", "navtitle", form, fresh ? "New research session" : "Research context");
    const c = (fresh ? (Q.setupContext ? Q.setupContext() : {}) : active.context) || {};
    const field = (label, key, value, attrs) => { const l = h("label", "f", form, label), i = h("input", "", l); i.type = (attrs && attrs.type) || "text"; i.value = value === undefined || value === null ? "" : value; i.dataset.key = key; if (attrs && attrs.placeholder) i.placeholder = attrs.placeholder; return i; };
    const name = field("Session name", "name", fresh ? "" : active.name, { placeholder: "e.g. Momentum on rates and equities" });
    const assets = field("Assets (tickers, separated by commas)", "assets", (c.assets || []).join(", "));
    const row1 = h("div", "params", form), start = field("First day", "start", c.start, { type: "date" }), end = field("Last day", "end", c.end, { type: "date" });
    row1.append(start.parentElement, end.parentElement);
    const sl = h("label", "f", form, "Data source"), src = h("select", "", sl); src.dataset.key = "data_source";
    for (const s of ["", ...meta.sources]) { const o = h("option", "", src, s || "(each tab's own)"); o.value = s; if ((c.data_source || "") === s) o.selected = true; }
    const row2 = h("div", "params", form);
    const bench = field("Benchmark ticker", "benchmark", c.benchmark), rf = field("Risk-free rate (% a year)", "risk_free", c.risk_free === undefined ? "" : +(c.risk_free * 100).toFixed(4), { type: "number" });
    const cost = field("Trading cost (basis points)", "cost_bps", (c.costs || {}).cost_bps, { type: "number" }), borrow = field("Borrow fee (% a year)", "borrow_pct", (c.costs || {}).borrow_pct, { type: "number" });
    const orate = field("Option rate (% a year)", "orate", (c.options || {}).rate === undefined ? "" : +((c.options || {}).rate * 100).toFixed(4), { type: "number" });
    const odiv = field("Dividend yield (% a year)", "odiv", (c.options || {}).div_yield === undefined ? "" : +((c.options || {}).div_yield * 100).toFixed(4), { type: "number" });
    row2.append(bench.parentElement, rf.parentElement, cost.parentElement, borrow.parentElement, orate.parentElement, odiv.parentElement);
    for (const i of row2.querySelectorAll("input[type=number]")) i.step = "any";
    h("p", "hint", form, "Frequency is daily: that is what the engines use. Leave a field empty to leave it to each tab.");
    const err = h("p", "hint err", form); err.setAttribute("role", "alert");
    const row = h("div", "row", form), ok = h("button", "btn primary", row, fresh ? "Create session" : "Save context"), cancel = h("button", "btn", row, "Cancel"); ok.type = "submit"; cancel.type = "button";
    if (!fresh) { const del = h("button", "btn", row, "Delete session"); del.type = "button"; del.addEventListener("click", async () => { if (!confirm(`Delete "${active.name}" and the ${active.artifacts.length} result(s) saved in it?`)) return; try { await api("/api/research/session/delete", { id: active.id }); d.close(); await R.open(""); await R.refresh(); } catch (e) { err.textContent = e.message; } }); }
    cancel.addEventListener("click", () => d.close());
    const num = (i, scale) => (i.value === "" ? null : Number(i.value) / (scale || 1));
    form.addEventListener("submit", async (e) => {
      e.preventDefault(); err.textContent = "";
      const ctx = { assets: assets.value.split(/[\s,;]+/).filter(Boolean), start: start.value || null, end: end.value || null, data_source: src.value, benchmark: bench.value.trim() || null, risk_free: num(rf, 100),
        costs: Object.fromEntries([["cost_bps", num(cost)], ["borrow_pct", num(borrow)]].filter(([, v]) => v !== null)), options: Object.fromEntries([["rate", num(orate, 100)], ["div_yield", num(odiv, 100)]].filter(([, v]) => v !== null)) };
      try {
        if (fresh) { await R.create(name.value || "My research", Object.fromEntries(Object.entries(ctx).filter(([, v]) => v !== null && v !== "" && !(Array.isArray(v) && !v.length) && !(typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length)))); }
        else { await api("/api/research/session/update", { id: active.id, name: name.value || active.name, context: ctx, replace: true }); await R.refresh(); }
        d.close();
      } catch (x) { err.textContent = x.message; }
    });
    d.showModal(); name.focus();
  }

  /* ---------------------------------------------------------------- what a Strategy Lab result can do next */
  Q.onResult((root, run) => {
    const card = h("div", "card ractions", root); card.dataset.role = "research-actions";
    const head = h("div", "row", card); h("strong", "", head, "Research"); h("span", "hint", head, "Keep this result in a session, then look at it from another side. A backtest of the whole history is a historical simulation, not out-of-sample evidence.");
    const row = h("div", "row", card), btn = (text, fn, title) => { const b = h("button", "btn", row, text); b.type = "button"; if (title) b.title = title; b.addEventListener("click", async () => { b.disabled = true; try { await fn(b); } catch (e) { toast(e.message); } b.disabled = false; }); return b; };
    const saved = async () => { if (run.artifact) return run.artifact; if (!run.job) throw new Error("This run was not started from this page, so its record cannot be saved"); run.artifact = await R.saveRun({ job: run.job }, run.result.name); return run.artifact; };
    btn("Save to session", async (b) => { await saved(); b.textContent = "Saved"; });
    btn("Save configuration", async () => {
      const body = Q.setupBody(); if (!body) { toast("The setup cannot run yet, so it cannot be saved"); return; }
      const name = prompt("Name for this configuration", run.result.name); if (!name) return;
      const s = await R.ensure(run.result.name); await api("/api/research/config/save", { session: s.id, name, body }); await R.refresh(); toast(`Configuration "${name}" saved to "${s.name}"`);
    }, "Keep the strategies, portfolio rule and assumptions of this setup under a name, to load again later");
    btn("Analyze by Market Regime", async () => { const a = await saved(); Q.showTab("regime", { result: a.id }); }, "Estimate regimes on these assets and see how this result behaved in each");
    btn("Evaluate with ML", async () => { await saved(); await R.setContext({ assets: run.result.universe.tickers, start: run.result.universe.first }); R.applyTo("ml"); Q.showTab("ml"); toast("The ML tab's dataset now uses this result's tickers and dates"); }, "Open the Machine Learning Lab on the same tickers and dates");
    btn("Explore Options Hedge", async () => { await saved(); const t = run.result.universe.tickers[0]; await R.setContext({ assets: run.result.universe.tickers, start: run.result.universe.first }); const lab = Q.lab("options"); if (lab) { lab.mode = "hedge"; R.applyTo("options"); } Q.showTab("options"); toast(`The Options Lab opens on ${t}. Hedging costs there are simulated, not historical trades.`); }, "Open the Options Lab's hedging simulation");
    btn("Add to Research Comparison", async () => { const a = await saved(); Q.showTab("rcompare", { add: a.id }); });
    const status = h("div", "", card);
    if (run.artifact && R.active()) {
      api(`/api/research/compat?session=${R.active().id}&artifact=${run.artifact.id}`).then((c) => {
        if (!c.issues.length) return;
        const b = h("div", "banner", status); h("strong", "", b, c.compatible ? "Note. " : "The session's context has changed since this result. ");
        b.appendChild(document.createTextNode(c.issues.map((i) => i.message).join("; ") + ". " + c.advice + " "));
        const again = h("button", "btn", b, "Apply the context and run again"); again.type = "button"; again.addEventListener("click", () => { Q.applyContext(R.active().context); const go = $("#panel button.primary"); if (go) go.click(); });
      }, () => {});
    }
  });

  /* The strategy backtests of the Options, Futures and Crypto labs can be kept in a session too. Their market data is simulated or downloaded, and the card says which. */
  Q.onLabDraw((lab, root, got) => {
    if (!["options", "futures", "crypto"].includes(lab.id) || lab.mode !== "backtest" || !got.job || !got.data || !/_backtest$/.test(got.data.kind || "")) return;
    const card = h("div", "card ractions", null); card.dataset.role = "research-actions";
    const head = h("div", "row", card); h("strong", "", head, "Research");
    Q.badges.evidence(head, lab.id === "options" || got.data.source === "simulated" ? "simulated" : "historical_simulation");
    h("span", "hint", head, lab.id === "options" ? "The option market is simulated: premiums and hedging fills are model outputs, not trades that could have been made." : "A simulation on saved data: costs are modelled, fills are not.");
    const row = h("div", "row", card), b = h("button", "btn", row, "Save to session"); b.type = "button";
    b.addEventListener("click", async () => { b.disabled = true; try { await R.saveRun({ job: got.job }, "My research"); b.textContent = "Saved"; } catch (e) { toast(e.message); b.disabled = false; } });
    const go = h("button", "btn", row, "Add to Research Comparison"); go.type = "button";
    go.addEventListener("click", async () => { try { const a = await R.saveRun({ job: got.job }, "My research"); Q.showTab("rcompare", { add: a.id }); } catch (e) { toast(e.message); } });
    root.prepend(card);
  });

  /* sessions and experiments in the search palette */
  if (Q.shell) Q.shell.addSource({ name: "research", async search(q) {
    const ql = q.toLowerCase(), out = [];
    for (const s of sessions.filter((x) => x.readable && x.name.toLowerCase().includes(ql)).slice(0, 4)) out.push({ label: s.name, hint: "Research session", run: () => R.open(s.id).then(() => Q.shell.go("saved")) });
    return out;
  } });

  const boot = () => R.refresh().catch(() => { loaded = true; draw(); });
  if (Q.S && Q.S.catalog) boot(); else { const t = setInterval(() => { if (Q.S && Q.S.catalog) { clearInterval(t); boot(); } }, 100); setTimeout(() => clearInterval(t), 20000); }
})();
