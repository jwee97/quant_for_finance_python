/* The Tune tab: find good settings for a strategy on the tickers listed in the Backtest tab, and learn whether to believe them.
   The person picks a strategy (the Kalman trend filter, say), ticks the settings to vary and presses one button. The server tries many combinations on the first part of the history, keeps the last part
   back (the hold-out) and, at the end, compares the starting settings with the winner on those untouched days and says in words whether the gain looks real. It talks to /api/tune/*; a search runs as a job on
   the app's worker, so closing the page does not stop it, and finished searches are listed again under "Recent runs". Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable, api = Q.api;
  const GUIDE = "tuning";
  const ACTIVE = new Set(["queued", "running"]);
  const RUN_WORD = { queued: "Waiting", running: "Running", done: "Finished", error: "Failed", interrupted: "Stopped" };
  const RECENT_MS = 6 * 3600 * 1000;
  const ICON = { good: "✓", mixed: "!", bad: "✗", keep: "=" };                                    // a verdict is never colour alone
  const MEASURES = [["sharpe", "Sharpe ratio"], ["cagr", "Growth a year"], ["vol", "Swings a year"], ["max_drawdown", "Worst fall"]];
  const STOPPED = { user: "You stopped it early, so fewer settings were tried than planned.", time: "It stopped when its time limit ran out, so fewer settings were tried than planned.",
    exhausted: "Every combination in the ranges was tried, so it stopped before the planned number." };

  const sig = (v, d = 3) => (Q.bad(v) ? "n/a" : String(+Number(v).toPrecision(d)));                    // a setting: three significant figures, whatever its size
  const show = (v) => (typeof v === "boolean" ? (v ? "yes" : "no") : typeof v === "number" ? sig(v) : String(v));
  const ratio = (v) => (Q.bad(v) ? "n/a" : num(v, 2));
  const growth = (v) => (Q.bad(v) ? "n/a" : pct(v, 1));
  const RATIOS = new Set(["sharpe", "sortino", "calmar", "cautious"]);
  const measure = (key, v) => (RATIOS.has(key) ? ratio(v) : growth(v));
  const measuresFor = (d) => (MEASURES.some(([key]) => key === d.objective.id) ? MEASURES : [[d.objective.id, d.objective.short.replace(/^./, (c) => c.toUpperCase())], ...MEASURES]);       // the measure that was improved is always a column
  const secs = (s) => (s < 90 ? `${Math.round(s)} seconds` : `${(s / 60).toFixed(1)} minutes`);
  const needNumber = (s, name) => { const t = String(s).trim(), v = Number(t); if (t === "" || !isFinite(v)) throw new Error(`${name} must be a number`); return v; };
  const numeric = (r) => r.kind === "int" || r.kind === "float";
  const isOpen = () => !Q.$(T.view).hidden;
  const models = () => (Q.S.catalog ? Q.S.catalog.models : []);
  const tickersNow = () => Q.S.tickers.filter((x) => x.status !== "bad");

  const T = {
    id: "tune", root: "#tune", view: "#view-tune", title: "Tune a strategy", api: 8, mode: "tune", busy: "", results: {}, errors: {}, guard: null, go: null, progress: null,
    about: "Find good settings for a strategy on the tickers in the Backtest tab, and learn whether to believe them.",
    modes: [["tune", "Tune"]], runLabel: { tune: "Find the best settings" },
    blurb: { tune: "Tries many combinations of a strategy's settings on the first part of the history, then checks the winner on the last part, which the search never saw." },
    strategy: "", plans: new Map(), plan: null, planError: "", loading: false, rows: [], needs: "", job: null, stopping: false, runs: { jobs: [], timer: null },
    form: { effort: "standard", trials: "60", seconds: "600", objective: "sharpe", holdout: "0.3", stable: true, seed: "0", search: "tpe", select: "best" },
  };

  /* ---------- the strategy and its settings ---------- */
  const initialStrategy = () => {
    const wanted = [T.strategy, Q.store("tune.strategy"), Q.S.models[0] && Q.S.models[0].name, "kalman_trend"];
    return wanted.find((n) => n && models().some((m) => m.name === n)) || (models()[0] && models()[0].name) || "";
  };
  const rowOf = (s) => ({ ...s, on: !!s.tune, now: numeric(s) || s.kind === "choice" ? String(s.default) : !!s.default, from: numeric(s) ? String(s.low) : "", to: numeric(s) ? String(s.high) : "" });
  function ensurePlan() {                                                                    // the form is drawn first and the plan fetched after it, so a slow answer never blanks the tab
    if (!T.strategy) T.strategy = initialStrategy();
    if (!T.plan && !T.planError && !T.loading && T.strategy) setTimeout(() => loadPlan(T.strategy), 0);
  }
  async function loadPlan(name) {
    T.strategy = name; Q.store("tune.strategy", name); T.planError = ""; T.needs = "";
    if (T.plans.has(name)) { adopt(name); return; }
    T.plan = null; T.rows = []; T.loading = true; if (isOpen()) Q.renderLab(T);
    try { T.plans.set(name, await api("/api/tune/plan?strategy=" + encodeURIComponent(name))); }
    catch (e) { T.planError = e.status === 404 ? "The app behind this page cannot plan a tuning: it is older than the page. " + Q.RESTART : e.message; }
    T.loading = false;
    if (T.strategy === name) adopt(name);
  }
  function adopt(name) {
    const plan = T.plans.get(name) || null;
    T.plan = plan; T.rows = plan ? plan.settings.map(rowOf) : [];
    if (isOpen()) { Q.renderLab(T); refreshGuard(); }
  }
  let guardSeq = 0;
  async function refreshGuard() {                                                           // a strategy that compares tickers needs several: say so before the run, as the Backtest tab does
    const seq = ++guardSeq;
    if (!T.plan || !isOpen()) return;
    const entry = { name: T.strategy, params: {} };
    try {
      const r = await api("/api/requirements", { models: [entry], allocator: Q.pickAllocator(entry) || null, allocator_params: {}, tickers: tickersNow().length });
      if (seq === guardSeq) T.needs = r.ok ? "" : r.message || "";
    } catch (e) { if (seq === guardSeq) T.needs = e.status === 404 ? "" : e.message; }
    if (seq === guardSeq) Q.syncLab(T);
  }

  function strategyOptions(select, current) {
    const groups = {};
    for (const m of models()) (groups[m.user ? "My strategies" : m.family] = groups[m.user ? "My strategies" : m.family] || []).push(m);
    const order = Object.keys(groups).sort((a, b) => (a === "My strategies" ? -1 : b === "My strategies" ? 1 : a.localeCompare(b)));
    for (const g of order) {
      const og = h("optgroup", "", select); og.label = g === "custom" ? "custom (formula)" : g;
      for (const m of groups[g]) { const o = h("option", "", og, m.name + (m.min_assets > 1 ? `  (needs ${m.min_assets}+ tickers)` : "")); o.value = m.name; if (m.name === current) o.selected = true; }
    }
  }

  /* ---------- the form ---------- */
  T.check = () => {
    if (T.planError) return T.planError;
    if (!T.plan) return "Loading the strategy's settings…";
    if (T.plan.note) return T.plan.note;
    if (T.needs) return T.needs;
    const ticked = T.rows.filter((r) => r.on);
    if (!ticked.length) return "Tick at least one setting to tune.";
    if (ticked.length > T.plan.max_settings) return `Tune at most ${T.plan.max_settings} settings at once.`;
    for (const r of ticked.filter(numeric)) {
      const lo = Number(r.from), hi = Number(r.to);
      if (String(r.from).trim() === "" || String(r.to).trim() === "" || !isFinite(lo) || !isFinite(hi)) return `${r.label}: type the lowest and the highest value to try.`;
      if (!(lo < hi)) return `${r.label}: the lowest value to try must be below the highest.`;
      if (r.scale === "log" && lo <= 0) return `${r.label}: the lowest value to try must be above zero.`;
    }
    if (T.form.effort === "custom" && (!isFinite(Number(T.form.trials)) || !isFinite(Number(T.form.seconds)))) return "Type the most tries and the time limit.";
    return "";
  };
  T.sections = (panel) => {
    ensurePlan();
    const sec = (title, open) => { const d = h("details", "", panel); d.open = open !== false; h("summary", "", d, title); return h("div", "body", d); };
    const after = () => Q.syncLab(T);

    const a = sec("1. What to tune");
    const pick = h("label", "f", a, "Strategy"), sel = h("select", "", pick); sel.dataset.role = "tune-strategy"; sel.setAttribute("aria-label", "strategy to tune");
    strategyOptions(sel, T.strategy);
    sel.addEventListener("change", () => loadPlan(sel.value));
    const info = models().find((m) => m.name === T.strategy);
    if (info) h("p", "hint", a, info.description);
    if (T.plan && T.plan.slow) { const w = h("p", "hint", a, "⏱ Slow: " + T.plan.slow + ". Every try runs the whole strategy, so a search takes many times longer: choose Quick, or tick fewer settings."); w.style.color = css("--warn"); }
    const tk = tickersNow(), src = (Q.S.catalog.sources || []).find((x) => x.name === Q.S.source), list = tk.slice(0, 8).map((x) => x.t).join(", ") + (tk.length > 8 ? `, and ${tk.length - 8} more` : "");
    const u = h("p", "hint", a, `It runs on the Backtest tab's list: ${tk.length} ticker${tk.length === 1 ? "" : "s"} (${list}), ${Q.S.start ? "from " + Q.S.start : "from the earliest day they all have"}, data from ${src ? src.label : Q.S.source}. `);
    u.dataset.role = "tune-universe";
    const change = h("button", "link", u, "Change them in the Backtest tab"); change.type = "button"; change.addEventListener("click", () => Q.showTab("backtest"));

    const b = sec("2. Which settings");
    if (T.planError) { const p = h("p", "hint", b, T.planError); p.style.color = css("--crit"); }
    else if (!T.plan) h("p", "hint", b, "Loading…");
    else if (!T.rows.length) h("p", "hint", b, T.plan.note);
    else {
      h("p", "hint", b, "Tick the settings the search may change, and say how far to look. A setting left unticked stays at the value shown. “Start from” is the value your result is compared with: the strategy's own, unless you change it.");
      const summary = h("p", "hint", b, ""); summary.dataset.role = "tune-summary";
      const note = () => { const n = T.rows.filter((r) => r.on).length; summary.textContent = n ? `${n} of ${T.rows.length} settings ticked. ${n > 6 ? "More than six is hard to search well: choose Thorough, or tick fewer." : n > 3 ? "Standard or Thorough is a better fit for this many." : ""}` : "Nothing is ticked."; };
      const holder = h("div", "tunesets", b);
      for (const r of T.rows) settingRow(holder, r, () => { note(); after(); });
      const reset = h("button", "btn small", b, "Back to the suggested settings"); reset.type = "button";
      reset.addEventListener("click", () => { T.rows = T.plan.settings.map(rowOf); Q.renderLab(T); });
      note();
    }

    const c = sec("3. How long to look");
    const efforts = T.plan ? T.plan.efforts : [{ id: "quick", label: "Quick" }, { id: "standard", label: "Standard" }, { id: "thorough", label: "Thorough" }];
    const explain = h("p", "hint", c, ""), custom = h("div", "params", c); custom.dataset.role = "tune-custom";
    const paintEffort = () => {
      const e = efforts.find((x) => x.id === T.form.effort);
      explain.textContent = e ? e.text : "You choose the most tries and the time limit; it stops at whichever comes first.";
      custom.replaceChildren(); custom.hidden = T.form.effort !== "custom";
      if (T.form.effort === "custom" && T.plan) {
        Q.labField(custom, { key: "trials", label: `Most tries (${T.plan.min_trials} to ${T.plan.max_trials})`, kind: "number", step: 1, min: 2, max: T.plan.max_trials }, T.form, after);
        Q.labField(custom, { key: "seconds", label: "Stop after (seconds)", kind: "number", step: 10, min: 10, max: 3600 }, T.form, after);
      }
    };
    const seg = W.seg(c, [...efforts.map((e) => [e.id, e.label]), ["custom", "Custom"]], T.form.effort, (v) => { T.form.effort = v; paintEffort(); after(); });
    seg.dataset.role = "tune-effort"; c.insertBefore(seg, explain);
    paintEffort();

    const d = sec("4. What counts as better", false);
    const objectives = T.plan ? T.plan.objectives : [["sharpe", "Return for the risk taken (Sharpe ratio)"]].map(([id, label]) => ({ id, label, text: "" }));
    const text = h("p", "hint", d, "");
    const note2 = () => { const o = objectives.find((x) => x.id === T.form.objective); text.textContent = o ? o.text : ""; };
    Q.labField(d, { key: "objective", label: "Measure to improve", kind: "choice", options: objectives.map((o) => [o.id, o.label]) }, T.form, () => { note2(); after(); });
    note2();
    const holds = (T.plan ? T.plan.holdouts : [0.2, 0.3, 0.4]).map((v) => [String(v), `${Math.round(v * 100)}% of the days`]);
    Q.labField(d, { key: "holdout", label: "Kept back for the final check", kind: "choice", options: holds, help: "These last days are never used to choose settings. They are looked at once, at the end." }, T.form, after);
    Q.labField(d, { key: "stable", label: "Prefer settings that work in every part of the period", kind: "bool", help: "Scores each setting on four equal parts of the tuning window and subtracts half of how much they differ." }, T.form, after);
    Q.labField(d, { key: "seed", label: "Random seed", kind: "number", step: 1, min: 0 }, T.form, after);
    h("p", "hint", d, "The same seed and settings give the same search. Every try counts as a trial on these tickers, so the Backtest tab's deflated Sharpe ratio takes it into account.");
    const searches = T.plan && T.plan.searches ? T.plan.searches : [], selects = T.plan && T.plan.selects ? T.plan.selects : [];
    if (searches.length && selects.length) {                                                                 // the defaults are the measured best: these are for people who want to compare
      const adv = h("details", "how", d); adv.dataset.role = "tune-advanced"; h("summary", "", adv, "Advanced: how to search, and which try wins");
      const explain = (list, key) => { const o = list.find((x) => x.id === T.form[key]); return o ? o.text : ""; };
      let searchNote, selectNote;
      Q.labField(adv, { key: "search", label: "How to search", kind: "choice", options: searches.map((o) => [o.id, o.label]) }, T.form, () => { searchNote.textContent = explain(searches, "search"); after(); });
      searchNote = h("p", "hint", adv, explain(searches, "search"));
      Q.labField(adv, { key: "select", label: "Which try wins", kind: "choice", options: selects.map((o) => [o.id, o.label]) }, T.form, () => { selectNote.textContent = explain(selects, "select"); after(); });
      selectNote = h("p", "hint", adv, explain(selects, "select"));
    }
  };
  /* One setting: a tick box, what it does, and (ticked) the range to look in, or (unticked) the value it stays at. The row is repainted in place so the panel keeps its scroll position. */
  function settingRow(parent, r, changed) {
    const box = h("div", "tuneset", parent); box.dataset.name = r.name;
    const top = h("div", "top", box), lab = h("label", "", top), cb = h("input", "", lab); cb.type = "checkbox"; cb.checked = r.on; cb.dataset.role = "tune-tick"; cb.setAttribute("aria-label", "tune " + r.label);
    h("b", "", lab, r.label); if (r.curated) { const s = h("span", "meta", top, "suggested range"); s.title = "A range chosen for this strategy by hand."; }
    const help = h("p", "help", box, r.help), vals = h("div", "vals", box);
    const input = (parentEl, caption, key, kind) => {
      const w = h("label", "", parentEl, caption); let el;
      if (kind === "choice") { el = h("select", "", w); for (const c of r.choices) { const o = h("option", "", el, show(c)); o.value = String(c); if (String(r[key]) === String(c)) o.selected = true; } el.addEventListener("change", () => { r[key] = r.kind === "bool" ? el.value === "true" : el.value; changed(); }); }
      else { el = h("input", "", w); el.type = "number"; el.step = r.kind === "int" ? "1" : "any"; el.value = r[key]; el.addEventListener("input", () => { r[key] = el.value; changed(); }); }
      el.dataset.field = key; el.setAttribute("aria-label", `${r.label}: ${caption.toLowerCase()}`);
    };
    const paint = () => {
      vals.replaceChildren(); help.hidden = !r.on; box.classList.toggle("on", r.on);
      const kind = numeric(r) ? "number" : "choice";
      input(vals, r.on ? "Start from" : "Keep at", "now", kind);
      if (r.on && numeric(r)) { input(vals, "Try from", "from", "number"); input(vals, "to", "to", "number"); }
      else if (r.on) h("span", "hint", vals, "tries every choice");
      vals.classList.toggle("three", r.on && numeric(r));
    };
    cb.addEventListener("change", () => { r.on = cb.checked; paint(); changed(); });
    paint();
  }

  /* ---------- running ---------- */
  function requestBody() {
    const universe = Q.universeBody(), f = T.form;
    if (!T.plan) throw new Error("Wait for the strategy's settings to load");
    const values = {}, tune = [];
    for (const r of T.rows) {
      const now = numeric(r) ? needNumber(r.now, r.label) : r.now;
      if (now !== r.default) values[r.name] = now;
      if (!r.on) continue;
      tune.push(numeric(r) ? { name: r.name, low: needNumber(r.from, `${r.label}: the lowest value`), high: needNumber(r.to, `${r.label}: the highest value`), scale: r.scale } : { name: r.name });
    }
    const body = { ...universe, strategy: T.strategy, tune, values, allocator: Q.pickAllocator({ name: T.strategy, params: {} }) || null, allocator_params: {}, objective: f.objective, effort: f.effort,
      holdout: Number(f.holdout), stable: !!f.stable, seed: needNumber(f.seed, "The seed"), search: f.search, select: f.select };
    if (f.effort === "custom") { body.trials = needNumber(f.trials, "The most tries"); body.seconds = needNumber(f.seconds, "The time limit"); }
    return body;
  }
  T.run = async () => {
    if (T.busy) return;
    const problem = T.check(); if (problem) { Q.toast(problem); return; }
    let body; try { body = requestBody(); } catch (e) { Q.toast(e.message); return; }
    T.busy = "Starting…"; T.job = null; T.stopping = false; T.results.tune = null; T.errors.tune = null; Q.syncLab(T); T.draw();
    try {
      const started = await api("/api/tune/run", body);
      T.job = started.job; if (isOpen()) T.draw();                                              // now the card can offer Stop
      const result = await Q.waitJob(started.job, (v) => { T.busy = `${v.message} (${Math.round(v.elapsed)} s)`; Q.syncLab(T); paintRunning(); }, 700);
      T.results.tune = { data: result, body }; T.errors.tune = null; Q.setTrials(result.trials_in_universe);
    } catch (e) { T.results.tune = null; T.errors.tune = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + Q.RESTART : e.message; }
    T.busy = ""; T.job = null; Q.syncLab(T); if (isOpen()) { T.draw(); window.scrollTo({ top: 0 }); }
    watchRuns();
  };
  function paintRunning() { const el = Q.$('[data-role="tune-progress"]'); if (el) el.textContent = T.busy; }
  async function stopSearch(job, button) {
    if (button) { button.disabled = true; button.textContent = "Stopping after this try…"; }
    try { await api("/api/tune/stop", { job }); } catch (e) { Q.toast(e.message); if (button) { button.disabled = false; button.textContent = "Stop and show what it has so far"; } }
  }
  T.empty = (root) => {
    runsCard(root);
    if (T.busy) {
      const c = h("div", "card empty", root); h("div", "spinner", c).style.margin = "0 auto 12px"; h("h2", "", c, "Looking for good settings…");
      h("p", "", c, T.busy).dataset.role = "tune-progress";
      h("p", "hint", c, "The first try is your starting settings; every later try is a full backtest of the tuning window. Closing this page does not stop it: the app keeps going, and the result is listed under Recent runs.");
      if (T.job) { const stop = h("button", "btn", c, "Stop and show what it has so far"); stop.type = "button"; stop.dataset.action = "stop-tune"; stop.addEventListener("click", () => stopSearch(T.job, stop)); }
      return;
    }
    const c = h("div", "card empty", root); h("h2", "", c, "Find good settings without guessing");
    for (const t of ["Pick a strategy on the left, tick the settings that may change and press the button. The app tries many combinations on your tickers and keeps the best.",
      "It keeps the last part of the history back and never lets the search see it. At the end it compares your starting settings with the winner on those untouched days, and tells you in words whether the gain looks real or looks like luck.",
      "Every try counts as a trial on these tickers, so the deflated Sharpe ratio in the Backtest tab stays honest about how much was searched."]) h("p", "", c, t);
    const go = h("button", "btn primary", c, T.runLabel.tune); go.type = "button"; go.addEventListener("click", () => T.run());
    const g = h("button", "link", c, "Read the guide"); g.type = "button"; g.style.marginLeft = "14px"; g.addEventListener("click", () => Q.openGuide(GUIDE));
  };

  /* ---------- recent runs ---------- */
  async function watchRuns() {
    clearTimeout(T.runs.timer); T.runs.timer = null;
    if (!isOpen()) return;
    let got;
    try { got = await api("/api/jobs"); } catch (e) { T.runs.jobs = []; paintRuns(); return; }
    T.runs.jobs = got.jobs.filter((j) => j.kind === "tune");
    paintRuns();
    if (T.runs.jobs.some((j) => ACTIVE.has(j.status))) T.runs.timer = setTimeout(watchRuns, 2000);
  }
  async function clearRuns(body) {
    try { await api("/api/jobs/clear", body); } catch (e) { Q.toast(e.message); }
    await watchRuns();
  }
  async function openRun(id) {
    try {
      const v = await api("/api/job/" + id);
      if (v.status !== "done" || !v.result) throw new Error(v.error || "This run has no result to show.");
      T.results.tune = { data: v.result, body: {} }; T.errors.tune = null; T.draw(); window.scrollTo({ top: 0 });
    } catch (e) { Q.toast(e.message); }
  }
  function paintRuns() {                                                                       // the card alone, put back at the top of the results
    const root = Q.$("#tune-results"); if (!root) return;
    const holder = document.createElement("div"); runsCard(holder);
    const old = root.querySelector(":scope > .runs"); if (old) old.remove();
    if (holder.firstChild) root.prepend(holder.firstChild);
  }
  function runsCard(root) {
    const old = root.querySelector(":scope > .runs"); if (old) old.remove();
    const now = Date.now(), order = { running: 0, queued: 1 };
    const jobs = T.runs.jobs.filter((j) => ACTIVE.has(j.status) || (j.finished && now - j.finished * 1000 < RECENT_MS))
      .sort((a, b) => (order[a.status] ?? 2) - (order[b.status] ?? 2) || (ACTIVE.has(a.status) ? a.created - b.created : (b.finished || 0) - (a.finished || 0))).slice(0, 6);
    if (!jobs.length) return;
    const card = h("div", "card runs", root);
    const going = jobs.some((j) => ACTIVE.has(j.status));
    h("h3", "", card, going ? "Searches in the background" : "Recent runs");
    if (going) h("p", "note", card, "These go on at the app's computer if you close this page. If the app itself stops (on Replit, when the workspace goes idle), the search is lost: it shows here as stopped and has to be started again from the beginning.");
    if (jobs.filter((j) => !ACTIVE.has(j.status)).length >= 2) {
      const all = h("button", "btn small", card, "Clear all ended runs"); all.type = "button"; all.dataset.action = "clear-ended"; all.addEventListener("click", () => clearRuns({ ended: true, kind: "tune" }));
    }
    const ul = h("ul", "runlist", card);
    for (const j of jobs) {
      const li = h("li", j.status, ul); li.dataset.job = j.id; li.dataset.status = j.status;
      const main = h("div", "main", li), head = h("span", "", main); h("b", "", head, (RUN_WORD[j.status] || j.status) + " · "); head.appendChild(document.createTextNode(String(j.label).replace(/^tune: /, "")));
      const detail = ACTIVE.has(j.status) ? `${j.message} (${Math.round(j.elapsed)} s)` : j.status === "done" ? `ended ${W.ago(new Date(j.finished * 1000).toISOString())} after ${Math.round(j.elapsed)} s` : j.status === "error" ? String(j.error || "it failed") : String(j.message || "");
      h("span", "meta", main, detail);
      if (j.status === "running") { const stop = h("button", "btn small", li, "Stop"); stop.type = "button"; stop.dataset.action = "stop-run"; stop.title = "End the search after the try it is on, and show what it has"; stop.addEventListener("click", () => stopSearch(j.id, stop)); }
      const clear = h("button", "btn small", li, ACTIVE.has(j.status) ? "Not running? Clear" : "Clear"); clear.type = "button"; clear.dataset.action = "clear-run";
      clear.title = ACTIVE.has(j.status) ? "If the app was stopped or crashed while this ran, take it off the list. A search that really is going in this app cannot be cleared." : "Take this run off the list.";
      clear.addEventListener("click", () => clearRuns({ id: j.id }));
      if (j.status === "done") { const open = h("button", "btn small", li, "Open"); open.type = "button"; open.dataset.action = "open-run"; open.addEventListener("click", () => openRun(j.id)); }
    }
  }
  T.draw = () => Q.drawLab(T);

  /* ---------- the result ---------- */
  const settingsText = (d, params) => `${d.strategy}: ` + (Object.keys(params).length ? Object.entries(params).map(([k, v]) => `${k}=${typeof v === "number" ? +v.toPrecision(6) : v}`).join(", ") : "the standard settings");
  const specText = (d) => `models:\n  - name: ${d.use.models[0].name}\n    params: ${JSON.stringify(d.use.models[0].params)}\n`;
  T.build = (root, d) => {
    runsCard(root);
    const ob = d.objective, w = d.windows, s = d.search;
    Q.labHead(root, `Best settings for ${d.strategy}`, `${d.universe.tickers.length} ticker${d.universe.tickers.length === 1 ? "" : "s"} · ${d.universe.source} · ${s.tried} setting${s.tried === 1 ? "" : "s"} tried in ${secs(s.seconds)} · tuned on ${w.tuning[0]} to ${w.tuning[1]}, checked on ${w.holdout[0]} to ${w.holdout[1]}`, GUIDE);
    verdictCard(root, d);
    settingsCard(root, d);
    const cmp = h("div", "charts", root);
    comparison(cmp, d, "On the days the search could see", `${w.tuning[0]} to ${w.tuning[1]} (${w.tuning_days.toLocaleString("en-US")} trading days). The winner was picked on these, so it is expected to look good here.`, "tuning", false);
    comparison(cmp, d, "On the days it never saw: the honest test", `${w.holdout[0]} to ${w.holdout[1]} (${w.holdout_days.toLocaleString("en-US")} trading days), ${Math.round(w.holdout_share * 100)}% of the history. No setting was chosen using these days.`, "holdout", true);
    const ch = h("div", "charts", root);
    growthChart(ch, d);
    for (const r of d.settings.slice().sort((a, b) => (b.share || 0) - (a.share || 0)).slice(0, 6)) settingChart(ch, d, r);
    triesChart(ch, d);
    if (d.tickers.length) tickerChart(ch, d);
    triesTable(root, d);
    trustCard(root, d);
    notesCard(root, d);
  };
  function verdictCard(root, d) {
    const v = d.verdict, card = h("div", "card verdict-card " + v.level, root); card.dataset.level = v.level;
    const head = h("div", "vhead", card); h("span", "vicon", head, ICON[v.level] || "?"); h("h2", "", head, v.headline);
    const ul = h("ul", "", card); for (const r of v.reasons) h("li", "", ul, r);
  }
  function settingsCard(root, d) {
    const card = W.card(root, d.same_as_start ? "The settings: nothing beat the starting ones" : "The settings", d.same_as_start ? "Every setting tried did worse than the starting settings on the tuning window, so they stay."
      : "Your starting settings against the best the search found. Values are rounded here; the buttons use the exact ones.");
    const edge = d.settings.some((r) => r.edge && !d.same_as_start);
    table(card, [{ label: "Setting", get: (r) => r.label }, { label: "Starting", num: true, get: (r) => show(r.start) },
      { label: "Best found", num: true, get: (r) => show(r.best) + (r.edge && !d.same_as_start ? " ⚑" : "") },
      { label: "Range tried", get: (r) => (r.kind === "choice" || r.kind === "bool" ? r.choices.map(show).join(" or ") : `${sig(r.low)} to ${sig(r.high)}${r.scale === "log" ? " (log)" : ""}`) },
      { label: "Mattered", num: true, title: "Share of the differences between tries that goes with this setting, compared with the other ticked ones. Rough: it needs a few dozen tries.", get: (r) => (r.share === null || r.share === undefined ? "n/a" : pct(r.share, 0)) }], d.settings);
    if (edge) h("p", "hint", card, "⚑ The best value is at the end of the range tried: the real best may lie beyond it.");
    const pick = d.choice && d.choice.rule === "plateau" && d.choice.raw_best ? d.choice.raw_best : null;
    if (pick) { const won = (d.trials || []).find((t) => t.is_best);
      h("p", "hint", card, `The winner here is the middle of the best region of settings, not the single highest score. The highest (try ${pick.number + 1}, score ${num(pick.score, 2)}) had weaker neighbours, which with a noisy score usually means luck; the pick scored ${won ? num(won.score, 2) : "less"} and its neighbours did as well.`); }
    const what = h("details", "how", card); h("summary", "", what, "What each setting does"); const dl = h("ul", "", what);
    for (const r of d.settings) { const li = h("li", "", dl); h("b", "", li, r.label + ": "); li.appendChild(document.createTextNode(r.help)); }
    const fixed = Object.entries(d.fixed || {});
    if (fixed.length) h("p", "hint", card, "Kept at: " + fixed.map(([k, v]) => `${k.replace(/_/g, " ")} = ${show(v)}`).join(", ") + ".");
    const row = h("div", "row tight", card);
    const use = h("button", "btn primary", row, "Open the Backtest tab with these settings"); use.type = "button"; use.dataset.action = "use-settings";
    use.addEventListener("click", () => { if (Q.useStrategy(d.use.models[0].name, d.use.models[0].params)) Q.toast("The Backtest tab now has these settings: press Run backtest there"); });
    const cp = h("button", "btn", row, "Copy as text"); cp.type = "button"; cp.dataset.action = "copy-settings"; cp.addEventListener("click", () => Q.copy(settingsText(d, d.use.models[0].params), "The settings"));
    const sp = h("button", "btn", row, "Copy as a spec for the command line"); sp.type = "button"; sp.dataset.action = "copy-spec"; sp.addEventListener("click", () => Q.copy(specText(d), "The spec"));
  }
  function comparison(parent, d, title, note, part, honest) {
    const card = h("div", "card compare wide" + (honest ? " honest" : ""), parent), rows = [["Starting settings", d.start[part]], [d.same_as_start ? "Best found (the same)" : "Tuned settings", d.best[part]]];
    if (d.benchmark) rows.push(["Holding every ticker equally", d.benchmark[part]]);
    h("h3", "", card, title); h("p", "note", card, note);
    table(card, [{ label: "", get: (r) => r[0] }, ...measuresFor(d).map(([key, label]) => ({ label, num: true, get: (r) => measure(key, r[1][key]) }))], rows);
    h("p", "hint", card, "Swings a year is how far the value moves up and down in a year (its volatility); worst fall is the deepest drop from a peak.");
  }
  function growthChart(parent, d) {
    const c = d.curves, series = [{ name: "Tuned settings", color: css("--s1"), values: c.best }, { name: "Starting settings", color: css("--s2"), values: c.start }];
    if (d.same_as_start) series.shift();
    const last = c.dates[c.dates.length - 1], all = series.flatMap((x) => x.values).filter((v) => v > 0), log = Math.max(...all) / Math.min(...all) > 2.5;
    chartCard(parent, "Growth of 1: tuned against starting settings", `Net of costs${log ? ", on a log axis (equal slopes are equal percentage growth)" : ""}. Left of the divide is what the search could see; right of it is the hold-out. Holding the tickers is left out here: it swings far more, so it is in the tables, compared by Sharpe ratio.`, { wide: true,
      draw: (b) => C.line(b, { dates: c.dates, series, log, baseline: 1, height: 300, label: "Growth of 1", endLabels: false, yFormat: (v) => String(+v.toPrecision(2)), tipFormat: (v) => v.toFixed(3),
        regions: [{ from: c.dates[0], to: c.split, name: "Tuned on these days" }, { from: c.split, to: last, name: "Never seen by the search" }], tipExtra: (i) => [["Window", c.dates[i] >= c.split ? "never seen by the search" : "tuned on"]] }),
      table: (b) => table(b, [{ label: "Date", get: (n) => c.dates[n] }, ...series.map((x) => ({ label: x.name, num: true, get: (n) => num(x.values[n], 3) }))], Q.sampleRows(c.dates, [], Math.max(1, Math.round(c.dates.length / 60)))) });
  }
  function settingChart(parent, d, r) {
    const share = r.share === null || r.share === undefined ? "" : ` · ${pct(r.share, 0)} of what mattered`;
    if (r.kind === "choice" || r.kind === "bool") {
      const rows = r.choices_summary.filter((x) => x.n && !Q.bad(x.mean)).map((x) => ({ label: show(x.choice), value: x.mean, text: `${num(x.mean, 2)} (${x.n} tr${x.n === 1 ? "y" : "ies"})` }));
      chartCard(parent, r.label, `Average score of the tries that used each choice. Best: ${show(r.best)}; starting: ${show(r.start)}${share}.`, {
        draw: (b) => C.hbars(b, { rows, format: (v) => num(v, 2), name: "Average score", label: r.label + " by choice", labelWidth: 90, valueWidth: 100 }),
        table: (b) => table(b, [{ label: "Choice", get: (x) => show(x.choice) }, { label: "Tries", num: true, get: (x) => x.n }, { label: "Average score", num: true, get: (x) => (Q.bad(x.mean) ? "n/a" : num(x.mean, 3)) }, { label: "Best score", num: true, get: (x) => (Q.bad(x.best) ? "n/a" : num(x.best, 3)) }], r.choices_summary) });
      return;
    }
    const log = r.scale === "log", fx = (v) => (log ? Math.log10(v) : v), back = (v) => (log ? 10 ** v : v), muted = css("--muted");
    const points = r.points.map(([x, y]) => {
      const isBest = Math.abs(x - r.best) <= 1e-12 * Math.max(1, Math.abs(r.best)), isStart = Math.abs(x - r.start) <= 1e-12 * Math.max(1, Math.abs(r.start));
      return { x: fx(x), y, color: isBest ? css("--s1") : isStart ? css("--s2") : muted, r: isBest || isStart ? 7 : 4, tip: [[r.label, show(x)], ["Score", num(y, 3)], ...(isBest ? [["", "the best found"]] : isStart ? [["", "the starting settings"]] : [])] };
    });
    const sweet = r.sweet ? ` · the best quarter of tries sat between ${sig(r.sweet[0])} and ${sig(r.sweet[1])}` : "", edge = r.edge ? ` · the best is at the ${r.edge === "low" ? "low" : "high"} end of the range: try a wider one` : "";
    chartCard(parent, r.label, `Each dot is one try. Best ${show(r.best)}, starting ${show(r.start)}${share}${sweet}${edge}.`, {
      draw: (b) => C.scatter(b, { points, height: 250, labels: "none", xLabel: r.label + (log ? " (log scale)" : ""), yLabel: "Score (higher is better)", xFormat: (v) => String(+back(v).toPrecision(2)), yFormat: (v) => num(v, 2), label: `Score against ${r.label}`,
        legend: [{ name: "A try", color: muted }, { name: "Best found", color: css("--s1") }, { name: "Starting settings", color: css("--s2") }] }),
      table: (b) => table(b, [{ label: r.label, num: true, get: (x) => show(x[0]) }, { label: "Score", num: true, get: (x) => num(x[1], 3) }], r.points) });
  }
  function triesChart(parent, d) {
    const muted = css("--muted"), scored = d.history.filter((x) => x.score !== null), top = Math.max(...scored.map((x) => x.score));
    const pts = scored.map((x) => ({ x: x.number + 1, y: x.score, color: x.number === 0 ? css("--s2") : x.score === top ? css("--s1") : muted, r: x.number === 0 || x.score === top ? 7 : 4,
      tip: [["Try", String(x.number + 1)], ["Score", num(x.score, 3)], ["Best so far", num(x.best, 3)], ...Object.entries(x.params || {}).slice(0, 6).map(([k, v]) => [k.replace(/_/g, " "), show(v)])] }));
    chartCard(parent, "Every try, in order", "The first few tries are random; later ones follow the better regions. A search that is working drifts upward; a flat cloud means the settings hardly matter, or the range is wrong.", {
      draw: (b) => C.scatter(b, { points: pts, height: 250, labels: "none", xLabel: "Try number", yLabel: "Score (higher is better)", xFormat: (v) => (Number.isInteger(v) ? String(v) : ""), yFormat: (v) => num(v, 2), label: "Score of every try",
        legend: [{ name: "A try", color: muted }, { name: "Best found", color: css("--s1") }, { name: "Starting settings", color: css("--s2") }] }),
      table: (b) => table(b, [{ label: "Try", num: true, get: (x) => x.number + 1 }, { label: "Score", num: true, get: (x) => (x.score === null ? "could not run" : num(x.score, 3)) }, { label: "Best so far", num: true, get: (x) => (x.best === null ? "n/a" : num(x.best, 3)) }], d.history) });
  }
  function tickerChart(parent, d) {
    const rows = d.tickers.map((t) => ({ label: t.ticker, value: t.best_holdout - t.start_holdout, text: (t.best_holdout - t.start_holdout >= 0 ? "+" : "−") + sig(Math.abs(t.best_holdout - t.start_holdout), 2) + " pts" }));
    chartCard(parent, "Where the new settings helped", "On the untouched days: what each ticker added to the strategy's return with the tuned settings, minus with the starting settings, in percentage points of capital. Blue helped, red hurt.", {
      draw: (b) => C.hbars(b, { rows, format: (v) => sig(v, 2), name: "Change", label: "Change by ticker on the hold-out", labelWidth: 64, valueWidth: 76 }),
      table: (b) => table(b, [{ label: "Ticker", get: (t) => t.ticker }, { label: "Starting, hold-out (pts)", num: true, get: (t) => num(t.start_holdout, 2) }, { label: "Tuned, hold-out (pts)", num: true, get: (t) => num(t.best_holdout, 2) },
        { label: "Starting, tuning (pts)", num: true, get: (t) => num(t.start_tuning, 2) }, { label: "Tuned, tuning (pts)", num: true, get: (t) => num(t.best_tuning, 2) }], d.tickers) });
  }
  function triesTable(root, d) {
    const card = W.card(root, "The best tries", "The twelve best scores on the tuning window, and the starting settings. The hold-out is not shown for them: choosing among them by it would use it up.");
    const cols = [{ label: "Try", num: true, get: (t) => String(t.number + 1) + (t.is_best ? " (best)" : t.is_start ? " (start)" : "") }, { label: "Score", num: true, get: (t) => num(t.score, 3) },
      { label: d.objective.short, num: true, get: (t) => measure(d.objective.id, t.tuning[d.objective.id]) }, ...d.settings.map((r) => ({ label: r.label, num: true, get: (t) => show(t.params[r.name]) }))];
    table(card, cols, d.trials);
  }
  function trustCard(root, d) {
    const sel = d.selection || {}, card = W.card(root, "How much to trust it", "The best of many tries always looks better than a typical one. These numbers say how much of the winner is that.");
    const k = h("div", "earnrow", card);
    tile(k, "Settings tried", String(d.search.tried), { text: d.search.failed ? `${d.search.failed} could not run` : "all could run" }, "", "How many different combinations were run, the starting settings included.");
    if (sel.deflated_sharpe_probability !== undefined) tile(k, "More than luck?", pct(sel.deflated_sharpe_probability, 0), { text: "95% or more is convincing", dir: sel.deflated_sharpe_probability >= 0.95 ? "up" : "" }, "",
      "The deflated Sharpe ratio: the chance that the winner's Sharpe ratio on the tuning window beats what the best of this many tries would show by luck alone" + (sel.effective_trials ? ` (the tries count as about ${Math.round(sel.effective_trials)} independent ones, because settings near each other trade alike).` : "."));
    if (sel.romano_wolf_p_best !== undefined && sel.romano_wolf_p_best !== null) tile(k, "Luck alone could do this", pct(sel.romano_wolf_p_best, 0), { text: "under 30% is asked for a ✓", dir: sel.romano_wolf_p_best < 0.3 ? "up" : "down" }, "",
      "The Romano-Wolf test: how often the best of everything tried would look at least this good if no setting had any edge, allowing for how many were tried and how alike they are. A lower number is better.");
    if (sel.pbo !== undefined && sel.pbo !== null) tile(k, "Winner's rank on the other half", pct(sel.pbo, 0), { text: "of cuts, below the middle" }, "",
      "The probability of backtest overfitting: how often the best setting on one half of the tuning window ranks below the middle on the other half. With a few settings that behave alike this is high even when there is a real edge, because neighbours swap places; read it beside 'Expected on new days'.");
    if (!d.same_as_start && !Q.bad(sel.mean_oos_sharpe_of_is_winner) && !Q.bad(sel.best_annual_sharpe)) tile(k, "Expected on new days", ratio(sel.mean_oos_sharpe_of_is_winner), { text: `the winner scored ${ratio(sel.best_annual_sharpe)} over the whole window` }, "",
      "Inside the tuning window, the setting that won on one half of the days scored this Sharpe ratio, on average, on the other half. A fairer expectation than the winner's own score, which was picked for being high.");
    const wf = d.walk_forward;
    if (!d.same_as_start && wf && wf.of) tile(k, "Choosing again and again", `${wf.beat} of ${wf.of}`, { text: Q.bad(wf.tuned_value) ? "stretches ahead of the starting settings" : `${ratio(wf.tuned_value)} against ${ratio(wf.start_value)} for the starting settings`,
      dir: wf.beat * 2 > wf.of && wf.gain > 0 ? "up" : wf.gain <= 0 ? "down" : "" }, "",
      `The settings were chosen again ${wf.of} times, each time from tries made before the search had learned anything and using only the days before a stretch, then used on that stretch. This counts the stretches in which that beat the starting settings.`);
    tile(k, "Counted as trials", String(d.trials_in_universe), { text: "on these tickers, in all" }, "", "Every try is a trial of the same data. The counter in the top bar, and the deflated Sharpe ratio in the Backtest tab, use it.");
    if (sel.error) h("p", "hint", card, "The overfitting checks could not be computed for this search: " + sel.error);
    const how = h("details", "how", card); h("summary", "", how, "What these mean"); const ul = h("ul", "", how);
    for (const t of ["More than luck: if you tried many settings, one of them will look good by chance. This asks whether the winner is better than the best of that many lucky tries would be.",
      "Luck alone could do this: pretend no setting had any edge and shuffle the days in the way real returns would vary. How often would the best of all these tries look at least as good as the winner? A ✓ asks for under 30%.",
      "Winner's rank on the other half: the tuning window is cut into pieces; the winner on some pieces is checked on the others. If it often falls below the middle there, the winner was picked for its luck. It is high whenever the tries are alike, so it is shown but does not decide the verdict.",
      "Expected on new days: the same cutting, read as a number. The winner on one half of the tuning window is scored on the other half; this is the average. If it is far below the winner's own score, the winner's score was mostly the choosing.",
      "Choosing again and again: pretend the search had been run before each of four later stretches of the tuning window, using only the days before it, and see whether the settings it picked beat the starting settings on that stretch. A real improvement keeps showing up; luck does not.",
      "All of these use only the tuning window. The hold-out comparison above is a separate, plainer check."]) h("li", "", ul, t);
  }
  function notesCard(root, d) {
    const notes = [...(d.notes || [])];
    if (STOPPED[d.search.stopped]) notes.push(STOPPED[d.search.stopped]);
    for (const f of d.failures || []) notes.push(`${f.count} setting${f.count === 1 ? "" : "s"} could not be scored: ${f.reason}`);
    notes.push(d.search.stable ? "Settings were ranked by their score on four equal parts of the tuning window (the mean, less half the spread), so one lucky stretch counts for less. The numbers in the tables are plain."
      : "Settings were ranked by the measure on the whole tuning window.");
    notes.push("The hold-out stops being untouched once it changes your mind: if you tune again after reading it, it has been used. Treat a second round as a new experiment.");
    notes.push("A setting that is best only at the edge of its range, or that changes the score sharply between neighbouring values, is fragile. Look for a broad, flat top in the charts above, not a spike.");
    const card = W.card(root, "Read before relying on it");
    const ul = h("ul", "", card); for (const t of notes) h("li", "", ul, t);
  }

  T.watch = () => { watchRuns(); refreshGuard(); };
  Q.registerWorkspace("tune", T);
  document.getElementById("tab-tune").addEventListener("click", () => setTimeout(T.watch, 0));
  if (Q.store("tab") === "tune") setTimeout(T.watch, 400);
})();
