/* The pages that are not labs: Overview, Saved research, Recent experiments, Background jobs, Data management and Settings.

   Nothing on these pages is made up. Every block says where its numbers come from, with a badge (real data, a historical simulation, simulated, unavailable), and a block that has nothing to show says
   why instead of drawing a placeholder. The Overview reads one endpoint (/api/research/overview); the layout of its widgets is a view preference kept in the browser (ql.dash). */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, store, toast } = Q, R = Q.research;
  const pct = (v, d = 1) => (v === null || v === undefined ? "n/a" : Q.pct(v, d)), num = (v, d = 2) => (v === null || v === undefined ? "n/a" : Q.num(v, d));
  const ago = (iso) => { if (!iso) return ""; const s = (Date.now() - Date.parse(iso)) / 1000; return s < 90 ? "just now" : s < 5400 ? Math.round(s / 60) + " min ago" : s < 129600 ? Math.round(s / 3600) + " h ago" : Math.round(s / 86400) + " days ago"; };
  const secs = (t) => { const s = (Date.now() / 1000) - t; return s < 90 ? "just now" : s < 5400 ? Math.round(s / 60) + " min ago" : s < 129600 ? Math.round(s / 3600) + " h ago" : Math.round(s / 86400) + " days ago"; };

  /* ---------------------------------------------------------------- labels: what kind of evidence, what kind of data */
  const EVIDENCE = { in_sample: ["is", "In-sample", "Fitted and measured on the same days"], out_of_sample: ["oos", "Out-of-sample", "Measured on days the model did not see"], walk_forward: ["wf", "Walk-forward", "Each prediction made by a model trained on earlier days only"],
    historical_simulation: ["is", "Historical simulation", "A backtest of the whole history: parameters were chosen with the whole sample in view unless a hold-out was used"], simulated: ["sim", "Simulated", "Made by a model, not observed"],
    retrospective: ["is", "Retrospective", "Uses information that was not known at the time"], paper: ["oos", "Paper trading", "Forward, with no money at stake"], live: ["real", "Live", "Forward, with money at stake"] };
  const DATA = { platform: ["real", "Platform data", "Real prices held with the repository (a versioned snapshot)"], user_file: ["real", "Your file", "Read from a file you put in data/user"], downloaded: ["real", "Downloaded data", "Real prices from a free public source, delayed and unaudited"],
    simulated: ["sim", "Simulated data", "Made by a model on this computer: not a market"], mixed: ["hist", "Mixed sources", "Some parts platform data, some downloaded or yours"], unavailable: ["na", "Unavailable", "There is nothing to show"] };
  const badge = (parent, spec, extra) => { const b = h("span", "ev " + spec[0], parent, spec[1] + (extra ? " · " + extra : "")); b.title = spec[2] || ""; return b; };
  Q.badges = { EVIDENCE, DATA, evidence: (p, k) => badge(p, EVIDENCE[k] || ["na", k]), data: (p, k) => badge(p, DATA[k] || ["na", k]) };
  const head = (root, title, sub) => { const d = h("div", "head", root), l = h("div", "", d); h("h2", "", l, title); if (sub) h("p", "sub", l, sub); return d; };
  const note = (root, text) => h("p", "empty-note", root, text);
  const table = (parent, cols, rows) => { if (!rows.length) return null; Q.simpleTable(parent, cols, rows); return true; };
  const link = (parent, text, tab, args) => { const b = h("button", "btn go", parent, text); b.type = "button"; b.addEventListener("click", () => Q.showTab(tab, args)); return b; };

  /* ---------------------------------------------------------------- the research path: a guide, never a gate */
  const STEPS = [["setup", "Setup", "Choose assets, dates and assumptions", "saved", (a) => a && a.context && a.context.assets && a.context.assets.length], ["backtest", "Backtest", "Test a strategy", "backtest", null, "backtest"], ["ml", "Machine learning", "Predict with models", "ml", null, "ml_experiment"],
    ["regime", "Regimes", "See how markets behave by state", "regime", null, "regime"], ["options", "Options hedging", "Cost a hedge", "options", null, "options"], ["compare", "Comparison", "Put results side by side", "rcompare", (a) => a && a.groups && a.groups.length], ["report", "Report", "Export what you found", "rcompare", () => false]];
  function pathCard(root, session, artifacts) {
    const card = h("div", "card", root); card.dataset.role = "research-path";
    h("h3", "", card, "Research path"); h("p", "sub", card, "A suggested order, not a requirement: open any step on its own.");
    const kinds = new Set((artifacts || []).map((a) => a.kind)), row = h("ol", "steps", card);
    for (const [, label, hint, tab, test, kind] of STEPS) {
      const done = kind ? kinds.has(kind) : test ? !!test(session) : false, li = h("li", done ? "done" : "", row);
      const b = h("button", "", li); b.type = "button"; b.title = hint; b.append(h("span", "stepname", null, label), h("small", "", null, done ? "done" : hint)); b.addEventListener("click", () => Q.showTab(tab));
    }
    return card;
  }

  /* ---------------------------------------------------------------- Overview */
  const WIDGETS = [
    { id: "market", title: "Market overview", go: ["data", "Data"], draw(b, d) {
      const m = d.market; if (!m.available) { note(b, m.reason); return ["na"]; }
      table(b, [{ label: "Ticker", get: (r) => r.ticker }, { label: "1 day", num: true, get: (r) => pct(r.d1) }, { label: "1 month", num: true, get: (r) => pct(r.m1) }, { label: "1 year", num: true, get: (r) => pct(r.y1) }], m.rows.slice(0, 8));
      h("p", "hint", b, `As of ${m.as_of}. ${m.source[0].toUpperCase() + m.source.slice(1)}.`); return ["hist", `as of ${m.as_of}`]; } },
    { id: "watchlist", title: "Watchlist", go: ["backtest", "Strategy Lab"], draw(b, d) {
      const tickers = (Q.S.tickers || []).filter((x) => x.status !== "bad").map((x) => x.t), rows = d.market.rows || [], by = new Map(rows.map((r) => [r.ticker, r]));
      if (!tickers.length) { note(b, "Your ticker list is empty. Add tickers in the Strategy Lab."); return ["na"]; }
      const have = tickers.filter((t) => by.has(t)), missing = tickers.filter((t) => !by.has(t));
      if (have.length) table(b, [{ label: "Ticker", get: (r) => r.ticker }, { label: "1 month", num: true, get: (r) => pct(r.m1) }, { label: "1 year", num: true, get: (r) => pct(r.y1) }], have.map((t) => by.get(t)).slice(0, 8));
      if (missing.length) h("p", "hint", b, `No price history in the platform snapshot for ${missing.slice(0, 8).join(", ")}${missing.length > 8 ? "…" : ""}: they are downloaded when you run a backtest.`);
      return have.length ? ["hist", `as of ${d.market.as_of}`] : ["na"]; } },
    { id: "projects", title: "Active research", go: ["saved", "Saved research"], draw(b, d) {
      const s = d.sessions; if (!s.n) { note(b, "No research session yet. A session keeps your tickers, dates, results and notes together."); const x = h("button", "btn primary", b, "Start a session"); x.type = "button"; x.addEventListener("click", () => Q.shell.go("saved")); return ["na"]; }
      h("ul", "list", b); const ul = b.lastChild;
      for (const r of s.recent) { const li = h("li", "", ul), a = h("button", "link", li, r.name); a.type = "button"; a.addEventListener("click", () => R.open(r.id).then(() => Q.showTab("saved"))); h("span", "hint", li, `${r.n_artifacts} result${r.n_artifacts === 1 ? "" : "s"} · ${ago(r.updated)}`); }
      return ["real", `${s.n} session${s.n === 1 ? "" : "s"}`]; } },
    { id: "backtests", title: "Recent backtests", go: ["saved", "Saved research"], draw(b, d) {
      const rows = d.artifacts.recent.filter((a) => a.kind === "backtest"); if (!rows.length) { note(b, "No backtest has been saved. Run one in the Strategy Lab and choose Save to session."); return ["na"]; }
      table(b, [{ label: "Strategy", get: (r) => r.title }, { label: "Sharpe", num: true, get: (r) => num(r.metrics.sharpe) }, { label: "Saved", get: (r) => ago(r.created) }], rows.slice(0, 5)); return ["is", "historical simulation"]; } },
    { id: "strategy", title: "Strategy performance", go: ["backtest", "Strategy Lab"], draw(b, d) {
      const p = d.portfolio_risk; if (!p.available) { note(b, p.reason); return ["na"]; }
      const k = h("div", "kpis", b); Q.tile(k, "CAGR", pct(p.metrics.cagr)); Q.tile(k, "Sharpe", num(p.metrics.sharpe)); h("p", "hint", b, `${p.title}. ${p.source}.`); return ["is", "historical simulation"]; } },
    { id: "ml", title: "Machine-learning results", go: ["experiments", "Recent experiments"], draw(b, d) {
      const m = d.ml; if (!m.available) { note(b, m.reason); return ["na"]; }
      table(b, [{ label: "Experiment", get: (r) => r.name }, { label: "Model", get: (r) => r.model_kind || "" }, { label: "Sharpe", num: true, get: (r) => num(r.sharpe) }], m.recent.slice(0, 5)); return ["wf", "walk-forward"]; } },
    { id: "regime", title: "Market regime", go: ["regime", "Market Regime Lab"], draw(b, d) {
      const r = d.regime; if (!r.available) { note(b, r.reason); return ["na"]; }
      h("p", "", b, `${r.state} as of ${r.as_of}`).style.fontWeight = "650";
      const ul = h("ul", "list", b); for (const [k, v] of Object.entries(r.probabilities)) { const li = h("li", "", ul, k); h("span", "", li, pct(v, 0)); }
      h("p", "hint", b, `${r.basis}. ${r.source}.`); return ["wf", "filtered"]; } },
    { id: "options", title: "Options volatility conditions", go: ["options", "Options Lab"], draw(b, d) {
      const o = d.options; if (!o.available) { note(b, o.reason); return ["na"]; } const k = h("div", "kpis", b); Q.tile(k, "Sharpe", num(o.metrics.sharpe)); Q.tile(k, "Max drawdown", pct(o.metrics.max_drawdown)); h("p", "hint", b, `${o.title}. ${o.source}.`); return ["sim", "simulated"]; } },
    { id: "risk", title: "Portfolio risk", go: ["backtest", "Strategy Lab"], draw(b, d) {
      const p = d.portfolio_risk; if (!p.available) { note(b, p.reason); return ["na"]; }
      const k = h("div", "kpis", b); Q.tile(k, "Volatility", pct(p.metrics.ann_vol)); Q.tile(k, "Max drawdown", pct(p.metrics.max_drawdown)); h("p", "hint", b, `${p.title}. ${p.source}.`); return ["is", "historical simulation"]; } },
    { id: "jobs", title: "Job status", go: ["jobs", "Background jobs"], draw(b, d) {
      const j = d.jobs, c = j.counts; const k = h("div", "kpis", b); Q.tile(k, "Running or waiting", String(j.active)); Q.tile(k, "Failed", String(c.error + c.interrupted)); Q.tile(k, "Finished", String(c.done));
      if (j.recent[0]) h("p", "hint", b, `Latest: ${j.recent[0].label} (${j.recent[0].status}).`); return ["real", "this app's job log"]; } },
    { id: "data", title: "Data freshness", go: ["data", "Data management"], draw(b, d) {
      const p = d.data.platform, k = h("div", "kpis", b); Q.tile(k, "Platform data ends", p.last, { text: `${p.age_days} days ago`, dir: p.stale ? "down" : "" }); Q.tile(k, "Assets", String(p.n_assets));
      h("p", "hint", b, p.note + (p.stale ? " It is not a live feed: for current prices use a data source." : "")); return [p.stale ? "stale" : "real", p.stale ? "snapshot is old" : "current"]; } },
  ];
  const layout = () => { const L = store("dash") || {}; const ids = WIDGETS.map((w) => w.id); return { order: [...(L.order || []).filter((i) => ids.includes(i)), ...ids.filter((i) => !(L.order || []).includes(i))], hidden: (L.hidden || []).filter((i) => ids.includes(i)) }; };
  const BADGES = { real: ["real", "Real data"], hist: ["hist", "Historical"], is: ["is", "Historical simulation"], sim: ["sim", "Simulated"], wf: ["wf", "Walk-forward"], na: ["na", "Nothing to show"], stale: ["stale", "Stale"] };
  let overview = null, overviewSeq = 0, savedSeq = 0;
  async function drawOverview(root) {
    const my = ++overviewSeq;
    root.replaceChildren();
    head(root, "Overview", "Where your research stands. Every block names its source; a block with nothing to show says why.");
    const box = h("div", "stack", root); Q.loading(box, "Loading…");
    let data, detail = null;
    try { data = await api("/api/research/overview"); if (R.active()) detail = await api("/api/research/session?id=" + R.active().id).catch(() => null); } catch (e) { if (my === overviewSeq) { box.replaceChildren(); h("div", "banner err", box, "The overview could not be read: " + e.message); } return; }
    if (my !== overviewSeq) return;                                                       // a newer drawing started while this one waited for the server
    overview = data; box.replaceChildren();
    pathCard(box, R.active(), detail && detail.artifacts);
    const L = layout(), grid = h("div", "widgets", box); grid.dataset.role = "widgets";
    const move = (id, by) => { const o = layout().order, i = o.indexOf(id), j = i + by; if (j < 0 || j >= o.length) return; [o[i], o[j]] = [o[j], o[i]]; store("dash", { ...layout(), order: o }); drawOverview(root); };
    const hide = (id) => { const l = layout(); store("dash", { ...l, hidden: [...l.hidden, id] }); drawOverview(root); };
    const visible = L.order.filter((id) => !L.hidden.includes(id));
    for (const id of visible) {
      const w = WIDGETS.find((x) => x.id === id), card = h("section", "card widget", grid); card.dataset.widget = id; card.setAttribute("aria-label", w.title);
      const top = h("header", "", card); h("h3", "", top, w.title);
      const tools = h("span", "tools", top);
      const tool = (text, label, fn, off) => { const b = h("button", "", tools, text); b.type = "button"; b.setAttribute("aria-label", `${label}: ${w.title}`); b.title = label; b.disabled = !!off; b.addEventListener("click", fn); };
      tool("◀", "Move earlier", () => move(id, -1), visible.indexOf(id) === 0); tool("▶", "Move later", () => move(id, 1), visible.indexOf(id) === visible.length - 1); tool("✕", "Hide this widget", () => hide(id));
      const body = h("div", "", card); let tag;
      try { tag = w.draw(body, overview); } catch (e) { note(body, "This block could not be drawn: " + e.message); tag = ["na"]; }
      const spec = BADGES[tag[0]] || BADGES.na, bd = h("span", "ev " + spec[0], top, tag[1] || spec[1]); bd.title = "Where this block's numbers come from";
      const go = h("button", "btn go", card, "Open " + w.go[1]); go.type = "button"; go.addEventListener("click", () => Q.showTab(w.go[0]));
    }
    const l = layout();
    const foot = h("div", "row", box);
    if (l.hidden.length) { h("span", "hint", foot, "Hidden:"); for (const id of l.hidden) { const w = WIDGETS.find((x) => x.id === id), b = h("button", "btn", foot, "Show " + w.title); b.type = "button"; b.addEventListener("click", () => { store("dash", { ...layout(), hidden: layout().hidden.filter((x) => x !== id) }); drawOverview(root); }); } }
    const reset = h("button", "btn", foot, "Reset layout"); reset.type = "button"; reset.addEventListener("click", () => { store("dash", null); drawOverview(root); });
  }
  Q.registerPage("overview", { render: () => drawOverview($("#overview")) });

  /* ---------------------------------------------------------------- Saved research */
  async function drawSaved(root) {
    const my = ++savedSeq;
    root.replaceChildren();
    head(root, "Saved research", "Sessions keep a context, the results you saved from the labs, your notes and your comparisons. A result is stored as it was made: changing the context never edits it.");
    const row = h("div", "row", root), mk = h("button", "btn primary", row, "New session"); mk.type = "button"; mk.addEventListener("click", () => R.newSession());
    const list = R.sessions();
    if (!list.length) { const c = h("div", "card empty", root); h("h3", "", c, "No sessions yet"); note(c, "Create one, then save a backtest, an experiment or a regime estimate into it."); return; }
    const cols = [{ label: "Session", get: (s) => s.name }, { label: "Assets", get: (s) => (s.assets || []).slice(0, 5).join(", ") + ((s.assets || []).length > 5 ? "…" : "") }, { label: "Dates", get: (s) => (s.start || "…") + " to " + (s.end || "today") },
      { label: "Results", num: true, get: (s) => (s.readable ? String(s.n_artifacts) : "") }, { label: "Updated", get: (s) => (s.readable ? ago(s.updated) : s.reason) }];
    const card = h("div", "card", root); h("h3", "", card, "Sessions"); table(card, cols, list);
    const pick = h("div", "row", card); for (const s of list.filter((x) => x.readable)) { const b = h("button", "btn", pick, (R.active() && R.active().id === s.id ? "● " : "") + s.name); b.type = "button"; b.addEventListener("click", () => R.open(s.id).catch((e) => toast(e.message))); }
    for (const s of list.filter((x) => !x.readable)) h("div", "banner err", root, `${s.name}: ${s.reason}`);
    if (R.active()) await drawSession(root, my);
  }
  async function drawSession(root, my) {
    const s = R.active(), d = await api("/api/research/session?id=" + s.id), rec = d.session, arts = d.artifacts;
    if (my !== savedSeq) return;
    const card = h("div", "card", root); card.dataset.role = "session";
    const top = h("div", "head", card), l = h("div", "", top); h("h3", "", l, rec.name); h("p", "sub", l, `${(rec.context.assets || []).join(", ") || "no assets"} · ${rec.context.start || "first day"} to ${rec.context.end || "today"} · created ${ago(rec.created)}`);
    const acts = h("div", "row tight", top), act = (t, fn) => { const b = h("button", "btn", acts, t); b.type = "button"; b.addEventListener("click", fn); return b; };
    act("Duplicate", () => api("/api/research/session/duplicate", { id: rec.id }).then((x) => R.open(x.session.id)).catch((e) => toast(e.message)));
    act("Delete", () => { if (confirm(`Delete "${rec.name}" and its ${arts.length} result(s)?`)) api("/api/research/session/delete", { id: rec.id }).then(() => R.open("")).catch((e) => toast(e.message)); });
    pathCard(root, rec, arts);
    const rc = h("div", "card", root); h("h3", "", rc, "Results"); rc.dataset.role = "session-results";
    if (!arts.length) note(rc, "No results saved in this session. In the Strategy Lab, Machine Learning Lab or Market Regime Lab, choose Save to session.");
    else {
      const picked = new Set(), t = h("div", "scroll", rc), tb = h("table", "data", t), hd = h("tr", "", h("thead", "", tb)); for (const c of ["", "Result", "Kind", "Evidence", "Data", "Sharpe", "Saved", ""]) h("th", "", hd, c);
      const body = h("tbody", "", tb), cmpBtn = h("button", "btn primary", rc, "Compare selected"); cmpBtn.type = "button"; cmpBtn.disabled = true;
      cmpBtn.addEventListener("click", () => Q.showTab("rcompare", { ids: [...picked].join(",") }));
      for (const a of arts) {
        const tr = h("tr", "", body), c0 = h("td", "", tr), cb = h("input", "", c0); cb.type = "checkbox"; cb.setAttribute("aria-label", "select " + a.title); cb.addEventListener("change", () => { cb.checked ? picked.add(a.id) : picked.delete(a.id); cmpBtn.disabled = picked.size < 1; });
        h("td", "", tr, a.title); h("td", "", tr, a.kind.replace("_", " ")); badge(h("td", "", tr), EVIDENCE[a.evidence] || ["na", a.evidence], null); badge(h("td", "", tr), DATA[a.data_kind] || ["na", a.data_kind]);
        h("td", "num", tr, num(a.metrics.sharpe)); h("td", "", tr, ago(a.created));
        const last = h("td", "", tr), chk = h("button", "link", last, "Check context"); chk.type = "button"; chk.addEventListener("click", async () => {
          const c = await api(`/api/research/compat?session=${rec.id}&artifact=${a.id}`); toast(c.issues.length ? c.issues.map((i) => i.message).join("; ") : "The session's context still describes this result"); });
        const del = h("button", "link", last, " Delete"); del.type = "button"; del.addEventListener("click", () => api("/api/research/artifact/delete", { id: a.id }).then(() => drawSaved(root), (e) => toast(e.message)));
      }
    }
    const cc = h("div", "card", root); h("h3", "", cc, "Saved configurations"); cc.dataset.role = "session-configs";
    if (!(rec.configs || []).length) note(cc, "No configuration saved. In the Strategy Lab, set up a strategy and choose Save configuration.");
    for (const c of rec.configs || []) {
      const li = h("div", "row", cc); h("span", "", li, `${c.name} · ${(c.body.models || []).map((m) => m.name).join(", ")} · ${(c.body.tickers || []).length} tickers`); h("span", "hint", li, ago(c.created));
      const ld = h("button", "btn", li, "Load into Strategy Lab"); ld.type = "button"; ld.title = "Fill the Strategy Lab's form; nothing is run";
      ld.addEventListener("click", () => { Q.showTab("backtest"); if (Q.applySetup(c.body)) toast(`"${c.name}" loaded. Press Run backtest to run it.`); });
      const dl = h("button", "link", li, "Delete"); dl.type = "button"; dl.addEventListener("click", () => api("/api/research/config/delete", { session: rec.id, config: c.id }).then(() => drawSaved(root), (e) => toast(e.message)));
    }
    const gc = h("div", "card", root); h("h3", "", gc, "Comparisons"); gc.dataset.role = "session-groups";
    if (!rec.groups.length) note(gc, "No saved comparison. Select results above and compare them, then save the group.");
    for (const g of rec.groups) { const li = h("div", "row", gc); h("span", "", li, `${g.name} (${g.artifacts.length} results)`); const o = h("button", "btn", li, "Open"); o.type = "button"; o.addEventListener("click", () => Q.showTab("rcompare", { session: rec.id, group: g.id })); }
    const nc = h("div", "card", root); h("h3", "", nc, "Notes"); nc.dataset.role = "session-notes";
    for (const n of rec.notes) { const li = h("div", "row", nc); h("span", "hint", li, n.ts.slice(0, 10)); h("span", "", li, n.text); const x = h("button", "link", li, "Remove"); x.type = "button"; x.addEventListener("click", () => api("/api/research/note/delete", { session: rec.id, note: n.id }).then(() => drawSaved(root), (e) => toast(e.message))); }
    const ta = h("textarea", "", nc); ta.setAttribute("aria-label", "new note"); ta.placeholder = "Why this universe? What did you rule out?"; const add = h("button", "btn", nc, "Add note"); add.type = "button";
    add.addEventListener("click", () => ta.value.trim() && api("/api/research/note/add", { session: rec.id, text: ta.value }).then(() => drawSaved(root), (e) => toast(e.message)));
  }
  Q.registerPage("saved", { render: () => drawSaved($("#saved")) });
  R.onChange(() => { const tab = Q.currentTab(); if (tab === "saved") drawSaved($("#saved")); else if (tab === "overview") drawOverview($("#overview")); });

  /* ---------------------------------------------------------------- Recent experiments */
  async function drawExperiments(root) {
    root.replaceChildren();
    head(root, "Recent experiments", "The machine-learning registry and the results you saved. Registry experiments are walk-forward: every prediction came from a model trained on earlier days only.");
    const a = h("div", "card", root); h("h3", "", a, "Machine-learning experiments"); const body = h("div", "", a); Q.loading(body, "Loading…");
    try {
      const d = await api("/api/ml/experiments?limit=30"), rows = (d.data || d).rows || [];
      body.replaceChildren();
      if (!rows.length) note(body, "No experiment has been run. Build a dataset and train a model in the Machine Learning Lab.");
      else {
        const t = h("div", "scroll", body), tb = h("table", "data", t), hd = h("tr", "", h("thead", "", tb)); for (const c of ["Experiment", "Model", "Universe", "Validation Sharpe", "Test Sharpe", "Holdout", ""]) h("th", "", hd, c);
        const tbody = h("tbody", "", tb);
        for (const r of rows) {
          const tr = h("tr", "", tbody); h("td", "", tr, r.name); h("td", "", tr, r.model || ""); h("td", "", tr, String(r.universe || ""));
          h("td", "num", tr, num(r["validation.sharpe"])); h("td", "num", tr, num(r["test.sharpe"])); h("td", "", tr, String(r.holdout || "").replace(/_/g, " "));
          const last = h("td", "", tr), save = h("button", "link", last, "Add to session"); save.type = "button"; save.addEventListener("click", async () => { try { await R.saveRun({ experiment: r.id }, "My research"); } catch (e) { toast(e.message); } });
          const open = h("button", "link", last, " Open in ML Lab"); open.type = "button"; open.addEventListener("click", () => Q.showTab("ml"));
        }
        h("p", "hint", body, "Sharpe ratios are net of costs. The deflated Sharpe in each report corrects for how many times the same data was looked at.");
      }
    } catch (e) { body.replaceChildren(); h("div", "banner err", body, "The registry could not be read: " + e.message); }
    const o = await api("/api/research/overview").catch(() => null), b = h("div", "card", root); h("h3", "", b, "Saved results (all sessions)");
    if (!o || !o.artifacts.recent.length) note(b, "Nothing saved yet."); else table(b, [{ label: "Result", get: (r) => r.title }, { label: "Kind", get: (r) => r.kind.replace("_", " ") }, { label: "Evidence", get: (r) => (EVIDENCE[r.evidence] || [0, r.evidence])[1] }, { label: "Sharpe", num: true, get: (r) => num(r.metrics.sharpe) }, { label: "Saved", get: (r) => ago(r.created) }], o.artifacts.recent);
  }
  Q.registerPage("experiments", { render: () => drawExperiments($("#experiments")) });

  /* ---------------------------------------------------------------- what was trained, and every job ever run (the app's database) */
  const STATUS_CLASS = { done: "real", running: "wf", error: "stale", interrupted: "stale", cancelled: "na" };
  async function trainingCard(root) {
    const card = h("div", "card", root); card.dataset.role = "training-history"; h("h3", "", card, "Training history");
    h("p", "sub", card, "Every network that was trained, epoch by epoch, kept in the app's database. A fit marked interrupted is one the app died in (out of memory, a restart, the host going idle): the epochs it did finish are still here.");
    const body = h("div", "", card);
    async function load() {
      let d; try { d = await api("/api/training/fits?limit=30"); } catch (e) { body.replaceChildren(); note(body, e.message); return; }
      body.replaceChildren();
      if (!d.fits.length) { note(body, "No network has been trained yet. Train a deep model in the Machine Learning Lab and its epochs appear here as they happen."); return; }
      const out = h("div", "", body), dur = (v) => (v === null || v === undefined ? "" : v < 90 ? `${Math.round(v)} s` : `${(v / 60).toFixed(1)} min`);
      table(out, [{ label: "When", get: (f) => secs(f.started) }, { label: "Job", get: (f) => f.job_label || "" }, { label: "Model", get: (f) => `${f.model || "network"}${f.fold ? ` · fold ${f.fold} of ${f.folds}` : ""}` },
        { label: "Epochs", num: true, get: (f) => `${f.epochs_run ?? 0}${f.epochs ? " of " + f.epochs : ""}` }, { label: "Best epoch", num: true, get: (f) => (f.best_epoch ? String(f.best_epoch) : "") },
        { label: "Best validation", num: true, get: (f) => (f.best_val === null || f.best_val === undefined ? "" : String(+f.best_val.toPrecision(4))) },
        { label: "How it ended", get: (f) => (f.status === "done" ? (f.stopped_early ? "stopped early (patience)" : "all epochs") : f.status) }, { label: "Time", num: true, get: (f) => dur(f.seconds) },
        { label: "Memory", num: true, get: (f) => (f.peak_mb ? (f.peak_mb >= 1024 ? (f.peak_mb / 1024).toFixed(1) + " GB" : Math.round(f.peak_mb) + " MB") : "") }, { label: "", get: () => "" }], d.fits);
      const rows = out.querySelectorAll("tbody tr");
      d.fits.forEach((f, i) => {
        const tr = rows[i]; if (!tr) return; tr.dataset.fit = String(f.id);
        const cell = tr.lastElementChild, b = h("button", "link", cell, "Show curve"); b.type = "button";
        b.addEventListener("click", async () => {
          const next = tr.nextElementSibling; if (next && next.dataset.curve === String(f.id)) { next.remove(); b.textContent = "Show curve"; return; }
          try {
            const one = await api("/api/training/fit?id=" + f.id), e = one.epochs_detail; b.textContent = "Hide curve";
            const row = document.createElement("tr"), td = document.createElement("td"); row.dataset.curve = String(f.id); td.colSpan = 10; row.appendChild(td); tr.after(row);
            if (e.length < 2) { td.textContent = "Fewer than two epochs were recorded."; return; }
            const host = h("div", "", td); Q.C.line(host, { xs: e.map((x) => x.epoch + 1), series: [{ name: "Training loss", color: Q.css("--s1"), values: e.map((x) => x.train) }, { name: "Validation loss", color: Q.css("--s2"), dash: "5 3", values: e.map((x) => x.val) }],
              xLabel: "epoch", xFormat: (v) => String(Math.round(v)), xWhole: true, zoom: false, height: 200, yFormat: (v) => String(+v.toPrecision(2)), tipFormat: (v) => String(+v.toPrecision(4)), label: "Loss by epoch",
              vlines: f.best_epoch ? [{ x: f.best_epoch, label: "best" }] : [] });
          } catch (err) { toast(err.message); }
        });
      });
    }
    const row = h("div", "row", card), rf = h("button", "btn", row, "Refresh"); rf.type = "button"; rf.addEventListener("click", load);
    await load();
  }
  async function jobHistoryCard(root) {
    const card = h("div", "card", root); card.dataset.role = "job-history"; h("h3", "", card, "Every job ever run");
    h("p", "sub", card, "The list above keeps the newest sixty. The database keeps them all: search by name, message or error, and filter by how they ended.");
    const row = h("div", "row", card), q = h("input", "", row); q.type = "search"; q.placeholder = "Search jobs"; q.setAttribute("aria-label", "Search every job");
    const st = h("select", "", row); st.setAttribute("aria-label", "Filter by outcome"); for (const [v, t] of [["", "Any outcome"], ["done", "Finished"], ["error", "Failed"], ["interrupted", "Interrupted"], ["cancelled", "Cancelled"]]) { const o = h("option", "", st, t); o.value = v; }
    const body = h("div", "", card);
    async function load() {
      let d; try { d = await api(`/api/jobs/history?limit=40&status=${encodeURIComponent(st.value)}&q=${encodeURIComponent(q.value)}`); } catch (e) { body.replaceChildren(); note(body, e.message); return; }
      body.replaceChildren(); h("p", "hint", body, `${d.total} job(s)${Object.keys(d.counts).length ? ": " + Object.entries(d.counts).map(([k, v]) => `${v} ${k}`).join(", ") : ""}`);
      if (!d.jobs.length) { note(body, "No job matches."); return; }
      table(body, [{ label: "When", get: (j) => (j.created ? secs(j.created) : "") }, { label: "Job", get: (j) => j.label || "" }, { label: "Kind", get: (j) => j.kind || "" }, { label: "Outcome", get: (j) => j.status || "" }, { label: "Message", get: (j) => (j.error || j.message || "").slice(0, 140) },
        { label: "Took", num: true, get: (j) => (j.finished && j.created ? `${Math.round(j.finished - j.created)} s` : "") }], d.jobs);
    }
    let t = 0; q.addEventListener("input", () => { clearTimeout(t); t = setTimeout(load, 300); }); st.addEventListener("change", load);
    await load();
  }

  /* ---------------------------------------------------------------- Background jobs */
  let jobTimer = null;
  async function drawJobs(root) {
    clearInterval(jobTimer);
    root.replaceChildren();
    head(root, "Background jobs", "Backtests, trainings, searches and downloads run on the app's own worker; closing the page does not stop them. A job that was running when the app stopped is marked interrupted: nothing is resumed.");
    const card = h("div", "card", root), body = h("div", "", card);
    const tools = h("div", "row", root);
    const clear = h("button", "btn", tools, "Clear finished"); clear.type = "button"; clear.addEventListener("click", () => api("/api/jobs/clear", { ended: true }).then(load, (e) => toast(e.message)));
    const refresh = h("button", "btn", tools, "Refresh"); refresh.type = "button"; refresh.addEventListener("click", load);
    trainingCard(root); jobHistoryCard(root);
    async function load() {
      let d; try { d = await api("/api/jobs"); } catch (e) { body.replaceChildren(); h("div", "banner err", body, e.message); return; }
      body.replaceChildren();
      if (!d.jobs.length) { note(body, "No jobs yet. Anything you run in a lab appears here."); return; }
      const t = h("div", "scroll", body), tb = h("table", "data", t), hd = h("tr", "", h("thead", "", tb)); for (const c of ["Job", "Kind", "Status", "Progress", "Session", "Started", "Took", ""]) h("th", "", hd, c);
      const tbody = h("tbody", "", tb);
      for (const j of d.jobs) {
        const tr = h("tr", "", tbody); tr.dataset.job = j.id; h("td", "", tr, j.label); h("td", "", tr, j.kind);
        const st = h("td", "", tr), sp = h("span", "ev " + (j.status === "done" ? "real" : j.status === "error" || j.status === "interrupted" ? "stale" : j.status === "cancelled" ? "na" : "wf"), st, j.status); sp.title = j.status;
        h("td", "", tr, (j.error || j.message || "") + (j.retry_of ? ` (repeats ${j.retry_of})` : ""));
        const sess = R.sessions().find((x) => x.id === j.session); h("td", "", tr, sess ? sess.name : j.session ? "(deleted session)" : "");
        h("td", "", tr, j.created ? secs(j.created) : ""); h("td", "num", tr, j.elapsed === undefined ? "" : `${Math.round(j.elapsed)} s`);
        const last = h("td", "", tr);
        if (j.cancellable && !j.cancelling) { const c = h("button", "link", last, "Cancel"); c.type = "button"; c.title = "A waiting run is taken off the queue at once; a running one stops at its next checkpoint and keeps no result"; c.addEventListener("click", () => api("/api/jobs/cancel", { id: j.id }).then(load, (e) => toast(e.message))); }
        if (j.cancelling) h("span", "hint", last, "cancelling…");
        if (j.retryable) { const c = h("button", "link", last, "Run again"); c.type = "button"; c.title = "Start it again from the beginning with the same settings"; c.addEventListener("click", () => api("/api/jobs/retry", { id: j.id }).then(() => { toast("Started again"); return load(); }, (e) => toast(e.message))); }
        if (j.status === "done" && j.kind === "backtest" && R.active()) { const s = h("button", "link", last, "Save to session"); s.type = "button"; s.addEventListener("click", () => R.saveRun({ job: j.id }).catch((e) => toast(e.message))); }
        if (j.status === "done" && j.kind === "regime") { const s = h("button", "link", last, "Open in Regime Lab"); s.type = "button"; s.addEventListener("click", () => Q.showTab("regime", { job: j.id })); }
        if (j.status !== "queued" && j.status !== "running") { const c = h("button", "link", last, " Remove"); c.type = "button"; c.addEventListener("click", () => api("/api/jobs/clear", { id: j.id }).then(load, (e) => toast(e.message))); }
      }
    }
    await load();
    jobTimer = setInterval(() => { if (Q.currentTab() !== "jobs" || !root.isConnected) { clearInterval(jobTimer); return; } load(); }, 3000);
  }
  Q.registerPage("jobs", { render: () => drawJobs($("#jobs")) });

  /* ---------------------------------------------------------------- what the app keeps on disk, and a backup of it */
  const bytesText = (n) => (n >= 1e9 ? (n / 1e9).toFixed(1) + " GB" : n >= 1e6 ? (n / 1e6).toFixed(1) + " MB" : n >= 1e3 ? Math.round(n / 1e3) + " kB" : n + " B");
  async function storageCard(root) {
    const card = h("div", "card", root); card.dataset.role = "storage"; h("h3", "", card, "Storage and backup");
    let d; try { d = await api("/api/storage/status"); } catch (e) { note(card, "The storage report could not be read: " + e.message); return; }
    h("p", "sub", card, `Everything the app keeps is under ${d.root}.`);
    const k = h("div", "kpis", card); const total = d.folders.reduce((a, f) => a + f.bytes, 0), rows = d.databases.reduce((a, x) => a + Object.values(x.tables || {}).reduce((s2, v) => s2 + v, 0), 0);
    Q.tile(k, "Databases", String(d.databases.length), { text: `${rows.toLocaleString("en-US")} rows in all` }); Q.tile(k, "Used on disk", bytesText(total), { text: `${d.folders.length} folders` });
    Q.tile(k, "Jobs recorded", String(Object.values(d.jobs).reduce((a, v) => a + v, 0)), { text: "kept beyond the newest sixty" }); Q.tile(k, "Kept answers", String(d.cache.reduce((a, c) => a + c.entries, 0)), { text: d.cache.length ? d.cache.map((c) => c.kind).join(", ") : "none yet" });
    if (d.app_db && !d.app_db.ok) h("div", "banner err", card, "The app's own database could not be opened: " + (d.app_db.error || "unknown reason") + ". Training history and the job archive are not being kept.");
    table(card, [{ label: "Database", get: (x) => x.name }, { label: "Size", num: true, get: (x) => bytesText(x.bytes) }, { label: "Tables", get: (x) => (x.tables ? Object.entries(x.tables).slice(0, 6).map(([n, v]) => `${n} (${v.toLocaleString("en-US")})`).join(", ") + (Object.keys(x.tables).length > 6 ? ", …" : "") : x.error || "") }], d.databases);
    table(card, [{ label: "Folder", get: (x) => x.name }, { label: "Files", num: true, get: (x) => String(x.files) }, { label: "Size", num: true, get: (x) => bytesText(x.bytes) }], d.folders);
    const row = h("div", "row", card), bk = h("button", "btn primary", row, "Back up everything"); bk.type = "button"; bk.title = "A zip of a consistent copy of every database, made while the app runs";
    const say = h("span", "hint", row); say.setAttribute("role", "status");
    bk.addEventListener("click", async () => {
      bk.disabled = true; say.textContent = "Making the backup…";
      try {
        const r = await api("/api/storage/backup", {});
        if (r.zip_base64) { const bin = atob(r.zip_base64), arr = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i); Q.download(r.name, new Blob([arr], { type: "application/zip" })); say.textContent = `${r.name}: ${bytesText(r.bytes)}, ${r.files.length} files. Downloaded.`; }
        else say.textContent = r.note || `${r.name} is on the server.`;
      } catch (e) { say.textContent = ""; toast(e.message); }
      bk.disabled = false;
    });
    if (d.cache.length) { const cl = h("button", "btn", row, "Forget kept answers"); cl.type = "button"; cl.addEventListener("click", async () => { try { const r = await api("/api/storage/cache/clear", {}); toast(`${r.cleared} kept answer(s) forgotten`); drawData($("#data")); } catch (e) { toast(e.message); } }); }
    for (const t of d.advice) h("p", "hint", card, t);
  }

  /* ---------------------------------------------------------------- Data management */
  async function drawData(root) {
    root.replaceChildren();
    head(root, "Data management", "What data the app holds, where it came from and how old it is. Downloads are made from the Futures, Crypto and Options labs and the Strategy Lab; this page is the inventory.");
    let o, ds; try { [o, ds] = await Promise.all([api("/api/research/overview"), api("/api/data/status").catch(() => null)]); } catch (e) { h("div", "banner err", root, e.message); return; }
    const p = o.data.platform, a = h("div", "card", root); h("h3", "", a, "Platform dataset"); const k = h("div", "kpis", a);
    Q.tile(k, "First day", p.first); Q.tile(k, "Last day", p.last, { text: `${p.age_days} days old`, dir: p.stale ? "down" : "" }); Q.tile(k, "Assets", String(p.n_assets));
    h("p", "hint", a, p.note); badge(a, p.stale ? ["stale", "Snapshot is old", "It ends well before today"] : ["real", "Current"]);
    const b = h("div", "card", root); h("h3", "", b, "Data sources");
    table(b, [{ label: "Source", get: (s) => s.label }, { label: "State", get: (s) => (s.available ? "ready" : "not available") }, { label: "Note", get: (s) => s.problem || "" }], o.data.sources);
    const c = h("div", "card", root); h("h3", "", c, "Downloads");
    if (!ds) note(c, "The download inventory could not be read."); else {
      const rows = Object.entries(ds.kinds).map(([kind, v]) => ({ kind, n: v.downloads.length, newest: v.downloads[0] ? (v.downloads[0].fetched || v.downloads[0].as_of || v.downloads[0].last || "") : "" }));
      table(c, [{ label: "Tab", get: (r) => r.kind }, { label: "Downloads", num: true, get: (r) => String(r.n) }, { label: "Newest", get: (r) => String(r.newest).slice(0, 10) }], rows); h("p", "hint", c, `Kept in ${ds.directory}. ${ds.note}`);
    }
    await storageCard(root);
    const d = h("div", "card", root); h("h3", "", d, "Dataset versions in your results");
    const sess = R.sessions().filter((s) => s.readable);
    if (!sess.length) note(d, "Save a result to a session and its dataset version (a fingerprint of the tickers, dates and source it used) is listed here."); else note(d, "Each saved result carries the fingerprint of the data it used. Two results with different fingerprints are never ranked against each other in the Research comparison.");
  }
  Q.registerPage("data", { render: () => drawData($("#data")) });

  /* ---------------------------------------------------------------- Settings */
  /* what the machine behind the app can really give a run (the container's limits, not the host's) */
  async function machineCard(root) {
    const card = h("div", "card", root); card.dataset.role = "machine"; h("h3", "", card, "This machine");
    let m; try { m = await api("/api/system/resources"); } catch (e) { note(card, "The machine could not be read: " + e.message); return; }
    h("p", "sub", card, "What the app can really use here. A big model that needs more than this is refused before it starts, or stopped with a message before the host kills the app.");
    const k = h("div", "kpis", card), mb = (v) => (v === null || v === undefined ? "unknown" : v >= 1024 ? (v / 1024).toFixed(1) + " GB" : v + " MB");
    Q.tile(k, "CPUs it may use", String(m.cpu_limit), { text: `the host shows ${m.cpus_visible}; ${m.threads} worker thread(s) are used` }); Q.tile(k, "Memory limit", mb(m.memory_limit_mb), { text: `${mb(m.memory_used_mb)} in use` });
    Q.tile(k, "Free memory", mb(m.memory_free_mb), { text: m.memory_free_mb !== null && m.memory_limit_mb && m.memory_free_mb < 0.25 * m.memory_limit_mb ? "tight: a big model will not fit" : "" , dir: m.memory_free_mb !== null && m.memory_limit_mb && m.memory_free_mb < 0.25 * m.memory_limit_mb ? "down" : "" });
    Q.tile(k, "This app", mb(m.process_rss_mb), { text: "resident memory of the process" });
    for (const n of m.notes) h("p", "hint", card, n);
    for (const t of m.protection) h("p", "hint", card, t);
    h("p", "hint", card, "Where to run bigger models, and how to move your data there: docs/deployment.md.");
  }
  function drawSettings(root) {
    root.replaceChildren();
    head(root, "Settings", "How the app looks and opens. These are kept in this browser only; they hold no data and no results.");
    const card = h("div", "card", root); h("h3", "", card, "Appearance");
    const cur = store("theme") || "dark", r = h("div", "row", card); r.setAttribute("role", "radiogroup"); r.setAttribute("aria-label", "Theme");
    for (const [v, t] of [["dark", "Dark"], ["light", "Light"], ["system", "Follow the device"]]) { const l = h("label", "f inline", r), i = h("input", "", l); i.type = "radio"; i.name = "theme"; i.value = v; i.checked = cur === v; l.append(document.createTextNode(t)); i.addEventListener("change", () => Q.setTheme(v)); }
    const nav = h("div", "card", root); h("h3", "", nav, "Navigation");
    const op = h("div", "row", nav), lab = h("label", "f", op, "Open on"), sel = h("select", "", lab);
    for (const [v, t] of [["last", "The last page I used"], ["overview", "Overview"], ["backtest", "Strategy Lab"]]) { const o = h("option", "", sel, t); o.value = v; if ((Q.shell.prefs().landing || "last") === v) o.selected = true; }
    sel.addEventListener("change", () => Q.shell.setPref("landing", sel.value));
    const rail = h("label", "f inline", nav), rc = h("input", "", rail); rc.type = "checkbox"; rc.checked = !!Q.shell.prefs().rail; rail.append(document.createTextNode("Narrow sidebar (icons only)")); rc.addEventListener("change", () => { $("#railtoggle").click(); });
    const rs = h("button", "btn", nav, "Reset navigation and dashboard layout"); rs.type = "button"; rs.addEventListener("click", () => { store("shell", null); store("dash", null); toast("Reset. Reload the page to see the default sidebar."); });
    machineCard(root);
    const ab = h("div", "card", root); h("h3", "", ab, "What this browser keeps");
    h("p", "hint", ab, "Theme, the last page, which sidebar groups are folded, the dashboard layout, your ticker list, and the id of the open research session. Sessions and results are kept by the app in data/user/research, not here.");
    const wipe = h("button", "btn", ab, "Forget these browser preferences"); wipe.type = "button"; wipe.addEventListener("click", () => { for (const k of Object.keys(localStorage).filter((x) => x.startsWith("ql."))) localStorage.removeItem(k); toast("Forgotten. Reload the page."); });
  }
  Q.registerPage("settings", { render: () => drawSettings($("#settings")) });
})();
