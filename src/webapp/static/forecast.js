/* The Forecast tab: forecast one ticker's price as a distribution with several methods (Monte Carlo, Markov chains, Gaussian processes, a Kalman filter and others) and find out how good each of them has been.
   The person picks a ticker from the Backtest tab's list, a horizon and the methods, and presses one button. The server fits every method on the prices up to many past dates, scores what each said then against
   what happened (walk forward: a method only ever sees the days before the one it forecasts from), and forecasts from the last day. The page shows the fan of the price, the verdict in words, the scoreboard and
   how well calibrated each method was. It talks to /api/forecast/*; the work runs as a job on the app's worker, so closing the page does not stop it, and finished runs are listed again under "Recent runs".
   Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable, api = Q.api;
  const GUIDE = "forecasting";
  const ACTIVE = new Set(["queued", "running"]);
  const RUN_WORD = { queued: "Waiting", running: "Running", done: "Finished", error: "Failed", interrupted: "Stopped" };
  const RECENT_MS = 6 * 3600 * 1000;
  const ICON = { good: "✓", mixed: "!", bad: "✗" };                                              // a verdict is never colour alone
  const HORIZON_TEXT = { 5: "1 week (5 trading days)", 10: "2 weeks (10 trading days)", 21: "1 month (21 trading days)", 63: "3 months (63 trading days)" };
  const spct = (v, d = 1) => { if (Q.bad(v)) return "n/a"; const t = Math.abs(v * 100).toFixed(d); return (+t === 0 ? "" : v >= 0 ? "+" : "−") + t + "%"; };      // a signed percentage; a zero has no sign
  const money = (v) => (Q.bad(v) ? "n/a" : v >= 1000 ? v.toFixed(0) : v.toFixed(2));
  const pval = (v) => (Q.bad(v) ? "n/a" : v < 0.001 ? "<0.001" : v.toFixed(3));
  const isOpen = () => !Q.$(F.view).hidden;
  const tickersNow = () => Q.S.tickers.filter((x) => x.status !== "bad");
  const human = (k) => String(k).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

  const F = {
    id: "forecast", root: "#forecast", view: "#view-forecast", title: "Forecast a stock", api: 9, mode: "forecast", busy: "", results: {}, errors: {}, guard: null, go: null, progress: null,
    about: "Forecast one ticker's price as a range of possible futures with several methods, and find out how good each method has been.",
    modes: [["forecast", "Forecast"]], runLabel: { forecast: "Forecast and test the methods" },
    blurb: { forecast: "Every method is tested the way it would have been used: at many past dates, with only the prices before each date, against what happened next." },
    plan: null, planError: "", loading: false, ticker: "", chosen: null, job: null, stopping: false, runs: { jobs: [], timer: null }, show: { method: "", pit: "" },
    form: { horizon: "21", effort: "standard", seed: "0" },
  };

  /* ---------- the plan ---------- */
  function ensurePlan() {
    if (!F.plan && !F.planError && !F.loading) setTimeout(loadPlan, 0);
    const names = tickersNow().map((x) => x.t);
    if (!names.includes(F.ticker)) F.ticker = (Q.store("forecast.ticker") && names.includes(Q.store("forecast.ticker")) ? Q.store("forecast.ticker") : names[0]) || "";
  }
  async function loadPlan() {
    F.loading = true; F.planError = "";
    try { F.plan = await api("/api/forecast/plan"); if (!F.chosen) F.chosen = new Set(F.plan.default.filter((m) => m !== F.plan.baseline)); }
    catch (e) { F.planError = e.status === 404 ? "The app behind this page cannot forecast: it is older than the page. " + Q.RESTART : e.message; }
    F.loading = false;
    if (isOpen()) Q.renderLab(F);
  }
  const methodOf = (id) => (F.plan ? F.plan.methods.find((m) => m.id === id) : null);
  const shortOf = (id, d) => { const m = (d && d.evaluation.methods.find((x) => x.id === id)) || methodOf(id); return m ? m.short || m.label : id; };

  /* ---------- the form ---------- */
  F.check = () => {
    if (F.planError) return F.planError;
    if (!F.plan) return "Loading the methods…";
    if (!F.ticker) return "Choose a ticker in the Backtest tab first.";
    if (!F.chosen || ![...F.chosen].some((m) => m !== F.plan.baseline)) return "Tick at least one method besides the random walk.";
    if (!isFinite(Number(F.form.seed)) || Number(F.form.seed) < 0) return "The seed must be a number from 0 up.";
    return "";
  };
  F.sections = (panel) => {
    ensurePlan();
    const sec = (title, open) => { const d = h("details", "", panel); d.open = open !== false; h("summary", "", d, title); return h("div", "body", d); };
    const after = () => Q.syncLab(F);

    const a = sec("1. Which stock, how far ahead");
    const names = tickersNow().map((x) => x.t);
    const pick = h("label", "f", a, "Ticker"), sel = h("select", "", pick); sel.dataset.role = "forecast-ticker"; sel.setAttribute("aria-label", "ticker to forecast");
    for (const t of names) { const o = h("option", "", sel, t); o.value = t; if (t === F.ticker) o.selected = true; }
    sel.addEventListener("change", () => { F.ticker = sel.value; Q.store("forecast.ticker", F.ticker); after(); });
    const src = (Q.S.catalog.sources || []).find((x) => x.name === Q.S.source);
    const u = h("p", "hint", a, `Prices come from ${src ? src.label : Q.S.source}${Q.S.start ? ", from " + Q.S.start : ""}: the Backtest tab's choices. `); u.dataset.role = "forecast-universe";
    const change = h("button", "link", u, "Change them in the Backtest tab"); change.type = "button"; change.addEventListener("click", () => Q.showTab("backtest"));
    const horizons = (F.plan ? F.plan.horizons : [5, 10, 21, 63]).map((n) => [String(n), HORIZON_TEXT[n] || `${n} trading days`]);
    Q.labField(a, { key: "horizon", label: "How far ahead", kind: "choice", options: horizons, help: "The forecast is a range for the return over this many trading days. Every method is also scored at 1, 5 and 21 days." }, F.form, after);

    const b = sec("2. Which methods");
    if (F.planError) { const p = h("p", "hint", b, F.planError); p.style.color = css("--crit"); }
    else if (!F.plan) h("p", "hint", b, "Loading…");
    else {
      h("p", "hint", b, "The random walk is always run: it is the yardstick that every other method has to beat. Tick the methods to test against it. Hover a name for what it does.");
      const quick = h("div", "row tight", b);
      for (const [label, ids] of [["Recommended", F.plan.default], ["All of them", F.plan.methods.map((m) => m.id)], ["Fast ones only", F.plan.methods.filter((m) => !m.heavy).map((m) => m.id)]]) {
        const btn = h("button", "btn small", quick, label); btn.type = "button"; btn.addEventListener("click", () => { F.chosen = new Set(ids.filter((m) => m !== F.plan.baseline)); Q.renderLab(F); });
      }
      const fixed = h("label", "f inline", b); const c0 = h("input", "", fixed); c0.type = "checkbox"; c0.checked = true; c0.disabled = true; c0.setAttribute("aria-label", "random walk (always run)");
      fixed.appendChild(document.createTextNode("Random walk, constant volatility (always run: the yardstick)"));
      for (const fam of F.plan.families) {
        const list = F.plan.methods.filter((m) => m.family === fam && m.id !== F.plan.baseline);                  // the yardstick has its fixed box above
        if (list.length) Q.checkList(b, fam, list.map((m) => [m.id, m.label, m.text]), F.chosen, () => { note(); after(); });
      }
      const summary = h("p", "hint", b, ""); summary.dataset.role = "forecast-summary";
      const note = () => { const n = [...F.chosen].length; summary.textContent = `${n} method${n === 1 ? "" : "s"} ticked besides the random walk.`; };
      note();
    }

    const c = sec("3. How thorough");
    const efforts = F.plan ? F.plan.efforts : [{ id: "quick", label: "Quick" }, { id: "standard", label: "Standard" }, { id: "thorough", label: "Thorough" }];
    const explain = h("p", "hint", c, "");
    const paint = () => { const e = efforts.find((x) => x.id === F.form.effort); explain.textContent = e && e.text ? e.text : ""; };
    const seg = W.seg(c, efforts.map((e) => [e.id, e.label]), F.form.effort, (v) => { F.form.effort = v; paint(); after(); });
    seg.dataset.role = "forecast-effort"; c.insertBefore(seg, explain);
    paint();
    Q.labField(c, { key: "seed", label: "Random seed", kind: "number", step: 1, min: 0 }, F.form, after);
    h("p", "hint", c, "The same seed and settings give the same forecast. Every method's forecast is made of simulated futures; a different seed changes them a little.");
  };

  /* ---------- running ---------- */
  function requestBody() {
    const u = Q.universeBody(), f = F.form;
    if (!F.plan) throw new Error("Wait for the methods to load");
    return { classes: u.classes, start: u.start, source: u.source, refresh: u.refresh, ticker: F.ticker, methods: [...F.chosen], horizon: Number(f.horizon), effort: f.effort, seed: Number(f.seed) };
  }
  F.run = async () => {
    if (F.busy) return;
    const problem = F.check(); if (problem) { Q.toast(problem); return; }
    let body; try { body = requestBody(); } catch (e) { Q.toast(e.message); return; }
    F.busy = "Starting…"; F.job = null; F.stopping = false; F.results.forecast = null; F.errors.forecast = null; Q.syncLab(F); F.draw();
    try {
      const started = await api("/api/forecast/run", body);
      F.job = started.job; if (isOpen()) F.draw();
      const result = await Q.waitJob(started.job, (v) => { F.busy = `${v.message} (${Math.round(v.elapsed)} s)`; Q.syncLab(F); paintRunning(); }, 700);
      F.results.forecast = { data: result, body }; F.errors.forecast = null; F.show = { method: "", pit: "" };
    } catch (e) { F.results.forecast = null; F.errors.forecast = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + Q.RESTART : e.message; }
    F.busy = ""; F.job = null; Q.syncLab(F); if (isOpen()) { F.draw(); window.scrollTo({ top: 0 }); }
    watchRuns();
  };
  function paintRunning() { const el = Q.$('[data-role="forecast-progress"]'); if (el) el.textContent = F.busy; }
  async function stopRun(job, button) {
    if (button) { button.disabled = true; button.textContent = "Stopping after this forecast…"; }
    try { await api("/api/forecast/stop", { job }); } catch (e) { Q.toast(e.message); if (button) { button.disabled = false; button.textContent = "Stop and show what it has so far"; } }
  }
  F.empty = (root) => {
    runsCard(root);
    if (F.busy) {
      const c = h("div", "card empty", root); h("div", "spinner", c).style.margin = "0 auto 12px"; h("h2", "", c, "Forecasting and testing the methods…");
      h("p", "", c, F.busy).dataset.role = "forecast-progress";
      h("p", "hint", c, "Each method is fitted and tested at many past dates, then asked for a forecast from the last day. Closing this page does not stop it: the app keeps going, and the result is listed under Recent runs.");
      if (F.job) { const stop = h("button", "btn", c, "Stop and show what it has so far"); stop.type = "button"; stop.dataset.action = "stop-forecast"; stop.addEventListener("click", () => stopRun(F.job, stop)); }
      return;
    }
    const c = h("div", "card empty", root); h("h2", "", c, "See how far each way of forecasting can be trusted");
    for (const t of ["Pick a ticker and press the button. The app forecasts the range its price may be in, with Monte Carlo simulation, Markov chains and hidden Markov models, Gaussian processes, a Kalman filter, volatility models and more.",
      "Then it does the part that matters: it goes back in time, makes each method's forecast again at dozens of past dates using only the prices before each date, and scores them against what happened. A method is only as good as that record.",
      "Most of the answer is usually about the spread of returns, which can be forecast, and not about the direction, which almost never can. The page says which, with the numbers."]) h("p", "", c, t);
    const go = h("button", "btn primary", c, F.runLabel.forecast); go.type = "button"; go.addEventListener("click", () => F.run());
    const g = h("button", "link", c, "Read the guide"); g.type = "button"; g.style.marginLeft = "14px"; g.addEventListener("click", () => Q.openGuide(GUIDE));
  };

  /* ---------- recent runs ---------- */
  async function watchRuns() {
    clearTimeout(F.runs.timer); F.runs.timer = null;
    if (!isOpen()) return;
    let got;
    try { got = await api("/api/jobs"); } catch (e) { F.runs.jobs = []; paintRuns(); return; }
    F.runs.jobs = got.jobs.filter((j) => j.kind === "forecast");
    paintRuns();
    if (F.runs.jobs.some((j) => ACTIVE.has(j.status))) F.runs.timer = setTimeout(watchRuns, 2000);
  }
  async function clearRuns(body) {
    try { await api("/api/jobs/clear", body); } catch (e) { Q.toast(e.message); }
    await watchRuns();
  }
  async function openRun(id) {
    try {
      const v = await api("/api/job/" + id);
      if (v.status !== "done" || !v.result) throw new Error(v.error || "This run has no result to show.");
      F.results.forecast = { data: v.result, body: {} }; F.errors.forecast = null; F.show = { method: "", pit: "" }; F.draw(); window.scrollTo({ top: 0 });
    } catch (e) { Q.toast(e.message); }
  }
  function paintRuns() {
    const root = Q.$("#forecast-results"); if (!root) return;
    const holder = document.createElement("div"); runsCard(holder);
    const old = root.querySelector(":scope > .runs"); if (old) old.remove();
    if (holder.firstChild) root.prepend(holder.firstChild);
  }
  function runsCard(root) {
    const old = root.querySelector(":scope > .runs"); if (old) old.remove();
    const now = Date.now(), order = { running: 0, queued: 1 };
    const jobs = F.runs.jobs.filter((j) => ACTIVE.has(j.status) || (j.finished && now - j.finished * 1000 < RECENT_MS))
      .sort((a, b) => (order[a.status] ?? 2) - (order[b.status] ?? 2) || (ACTIVE.has(a.status) ? a.created - b.created : (b.finished || 0) - (a.finished || 0))).slice(0, 6);
    if (!jobs.length) return;
    const card = h("div", "card runs", root);
    const going = jobs.some((j) => ACTIVE.has(j.status));
    h("h3", "", card, going ? "Forecasts in the background" : "Recent runs");
    if (going) h("p", "note", card, "These go on at the app's computer if you close this page. If the app itself stops (on Replit, when the workspace goes idle), the run is lost: it shows here as stopped and has to be started again from the beginning.");
    if (jobs.filter((j) => !ACTIVE.has(j.status)).length >= 2) {
      const all = h("button", "btn small", card, "Clear all ended runs"); all.type = "button"; all.dataset.action = "clear-ended"; all.addEventListener("click", () => clearRuns({ ended: true, kind: "forecast" }));
    }
    const ul = h("ul", "runlist", card);
    for (const j of jobs) {
      const li = h("li", j.status, ul); li.dataset.job = j.id; li.dataset.status = j.status;
      const main = h("div", "main", li), head = h("span", "", main); h("b", "", head, (RUN_WORD[j.status] || j.status) + " · "); head.appendChild(document.createTextNode(String(j.label).replace(/^forecast: /, "")));
      const detail = ACTIVE.has(j.status) ? `${j.message} (${Math.round(j.elapsed)} s)` : j.status === "done" ? `ended ${W.ago(new Date(j.finished * 1000).toISOString())} after ${Math.round(j.elapsed)} s` : j.status === "error" ? String(j.error || "it failed") : String(j.message || "");
      h("span", "meta", main, detail);
      if (j.status === "running") { const stop = h("button", "btn small", li, "Stop"); stop.type = "button"; stop.dataset.action = "stop-run"; stop.title = "End the evaluation after the forecast it is on, and show what it has"; stop.addEventListener("click", () => stopRun(j.id, stop)); }
      const clear = h("button", "btn small", li, ACTIVE.has(j.status) ? "Not running? Clear" : "Clear"); clear.type = "button"; clear.dataset.action = "clear-run";
      clear.title = ACTIVE.has(j.status) ? "If the app was stopped or crashed while this ran, take it off the list. A run that really is going in this app cannot be cleared." : "Take this run off the list.";
      clear.addEventListener("click", () => clearRuns({ id: j.id }));
      if (j.status === "done") { const open = h("button", "btn small", li, "Open"); open.type = "button"; open.dataset.action = "open-run"; open.addEventListener("click", () => openRun(j.id)); }
    }
  }
  F.draw = () => Q.drawLab(F);

  /* ---------- the result ---------- */
  const scores = (d, id, hz) => { const m = d.evaluation.methods.find((x) => x.id === id); return m ? m.scores[String(hz)] || {} : {}; };
  const evaluated = (d) => d.evaluation.methods.filter((m) => m.id !== d.evaluation.baseline);
  function bestMethod(d) {                                                                              // the fan opens on the method with the best record at the chosen horizon
    const ranked = evaluated(d).filter((m) => d.now.methods[m.id]).sort((a, b) => (scores(d, b.id, d.horizon).skill ?? -9) - (scores(d, a.id, d.horizon).skill ?? -9));
    return ranked.length ? ranked[0].id : d.evaluation.baseline;
  }
  F.build = (root, d) => {
    runsCard(root);
    const ev = d.evaluation, o = ev.origins;
    Q.labHead(root, `Forecast for ${d.ticker}`, `as of ${d.asof} · last price ${money(d.last_price)} · ${d.horizon} trading days ahead · ${d.methods.length} method${d.methods.length === 1 ? "" : "s"} tested on ${o.count} forecasts made from ${o.first} to ${o.last}`, GUIDE);
    const acts = root.querySelector(".head .row.tight");
    const dl = h("button", "btn", acts, "Download the forecast (CSV)"); dl.type = "button"; dl.dataset.action = "download-forecast"; dl.addEventListener("click", () => downloadCsv(d));
    if (d.stopped) Q.labBanner(root, "Stopped early. ", `Only the first ${o.count} forecasts were made and scored; a full run would say more.`);
    verdictCard(root, d);
    fanCard(root, d);
    nowCard(root, d);
    scoreboard(root, d);
    skillMatrix(root, d);
    gainCard(root, d);
    pairCard(root, d);
    densityCard(root, d);
    calibrationCard(root, d);
    meanCard(root, d);
    assumptionsCard(root, d);
    learnedCard(root, d);
    notesCard(root, d);
  };
  function verdictCard(root, d) {
    const v = d.verdict, card = h("div", "card verdict-card " + v.level, root); card.dataset.level = v.level;
    const head = h("div", "vhead", card); h("span", "vicon", head, ICON[v.level] || "?"); h("h2", "", head, v.headline);
    const ul = h("ul", "", card); for (const r of v.reasons) h("li", "", ul, r);
  }

  const RANGES = { "80": ["0.1", "0.9"], "90": ["0.05", "0.95"], "98": ["0.01", "0.99"] };                   // the outer range of the fan: which two quantiles bound it
  /* The fan: the last 120 days of the price, then the range of where it may go, for the chosen method, with the random walk's range beside it. The outer range is 80%, 90% or 98%. */
  function fanCard(root, d) {
    const ids = Object.keys(d.now.methods);
    if (!F.show.method || !ids.includes(F.show.method)) F.show.method = bestMethod(d);
    const dates = d.history.dates.slice(-120).concat(d.future_dates), closes = d.history.close.slice(-120), nh = closes.length, last = closes[nh - 1];
    const wrap = (arr) => Array(nh - 1).fill(null).concat([last], arr);
    const mk = (id) => { const f = d.now.methods[id].fan; return { q: (k) => wrap(f[k].slice(0, d.horizon)) }; };
    const ranges = Object.keys(RANGES).filter((r) => RANGES[r].every((k) => ids.every((id) => k in d.now.methods[id].fan)));        // a run made before the wider levels existed has only 90%
    if (!F.show.range || !ranges.includes(F.show.range)) F.show.range = ranges.includes("90") ? "90" : ranges[0];
    const card = chartCard(root, `Where the price may go in the next ${d.horizon} trading days`, "", {
      wide: true,
      draw: (b) => {
        const id = F.show.method, m = mk(id), bands = [];
        const rw = d.now.methods[d.evaluation.baseline];
        const [lowQ, highQ] = RANGES[F.show.range];
        if (rw && id !== d.evaluation.baseline) bands.push({ name: `Random walk, ${F.show.range}% range`, color: css("--muted"), opacity: 0.1, lo: mk(d.evaluation.baseline).q(lowQ), hi: mk(d.evaluation.baseline).q(highQ) });
        bands.push({ name: `${F.show.range}% range`, color: css("--s1"), opacity: 0.14, lo: m.q(lowQ), hi: m.q(highQ) }, { name: "Middle 50%", color: css("--s1"), opacity: 0.26, lo: m.q("0.25"), hi: m.q("0.75") });
        C.line(b, { dates, series: [{ name: "Price", color: css("--ink"), values: closes.concat(Array(d.horizon).fill(null)) }, { name: "Median forecast", color: css("--s1"), values: m.q("0.5") }], bands,
          vlines: [{ x: d.asof, label: "today" }], yFormat: money, tipFormat: money, height: 330, label: `${d.ticker} price and the forecast range of ${shortOf(id, d)}`, legend: true });
      },
      table: (b) => {
        const f = d.now.methods[F.show.method].fan, steps = [1, 2, 3, 5, 10, 15, 21, 42, 63].filter((n) => n <= d.horizon);
        if (!steps.includes(d.horizon)) steps.push(d.horizon);
        table(b, [{ label: "Day", num: true, get: (n) => String(n) }, { label: "Date", get: (n) => d.future_dates[n - 1] }, { label: "5%", num: true, get: (n) => money(f["0.05"][n - 1]) }, { label: "25%", num: true, get: (n) => money(f["0.25"][n - 1]) },
          { label: "Median", num: true, get: (n) => money(f["0.5"][n - 1]) }, { label: "75%", num: true, get: (n) => money(f["0.75"][n - 1]) }, { label: "95%", num: true, get: (n) => money(f["0.95"][n - 1]) }], steps);
      } });
    card.dataset.role = "forecast-fan";
    if (ranges.length > 1) {
      const rbox = h("div", "seg", card._extra); rbox.setAttribute("role", "group"); rbox.setAttribute("aria-label", "outer range of the fan");
      for (const r of ranges) { const b = h("button", "", rbox, r + "%"); b.type = "button"; b.dataset.range = r; b.title = `Draw the outer band so that ${r}% of the simulated futures fall inside it`; b.setAttribute("aria-pressed", String(r === F.show.range)); b.addEventListener("click", () => { F.show.range = r; for (const x of rbox.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x.dataset.range === r)); card._draw(); }); }
    }
    const box = h("div", "tabsmall", card._extra); box.setAttribute("role", "group"); box.setAttribute("aria-label", "method shown");
    const paint = () => { for (const b of box.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.value === F.show.method)); noteEl.textContent = describe(); };
    for (const id of ids) {
      const b = h("button", "", box, shortOf(id, d)); b.type = "button"; b.dataset.value = id; b.title = (methodOf(id) || {}).label || id;
      b.addEventListener("click", () => { F.show.method = id; paint(); card._draw(); });
    }
    const describe = () => {
      const id = F.show.method, f = d.now.methods[id].fan, i = d.horizon - 1, sm = d.now.methods[id].summary[String(d.horizon)], rec = scores(d, id, d.horizon);
      return `${(methodOf(id) || {}).label || id}: from ${money(d.last_price)} today, the middle half of its forecasts for the price in ${d.horizon} trading days runs from ${money(f["0.25"][i])} to ${money(f["0.75"][i])}, and 9 in 10 fall between ${money(f["0.05"][i])} and ${money(f["0.95"][i])}.` +
        ` It puts the chance of a rise at ${pct(sm.p_up, 0)}. Its record: ${id === d.evaluation.baseline ? "it is the yardstick" : Q.bad(rec.skill) ? "not scored" : spct(rec.skill) + " against the random walk"}.`;
    };
    const noteEl = card.querySelector("p.note") || h("p", "note", card, "");
    paint();
  }

  /* What each method says from the last day, at the chosen horizon, beside how it has done. */
  function nowCard(root, d) {
    const hz = String(d.horizon), card = W.card(root, `What each method says now, for ${d.horizon} trading days`, "A range, not a point: the 5% and 95% columns are where the price would be if the return were worse, or better, than 19 forecasts in 20. The last column is how much better than the random walk the method's past forecasts scored.");
    card.dataset.role = "forecast-now";
    const rows = Object.values(d.now.methods);
    table(card, [{ label: "Method", get: (m) => shortOf(m.id, d), title: "The method" }, { label: "Average return", num: true, get: (m) => spct(m.summary[hz].mean, 1) }, { label: "Middle", num: true, get: (m) => spct(m.summary[hz].median, 1), title: "The median return" },
      { label: "5% worst case", num: true, get: (m) => spct(m.summary[hz].q05, 1) }, { label: "95% best case", num: true, get: (m) => spct(m.summary[hz].q95, 1) }, { label: "Chance of a rise", num: true, get: (m) => pct(m.summary[hz].p_up, 0) },
      { label: "Chance of losing 5%+", num: true, get: (m) => pct(m.summary[hz].p_loss_5, 0) },
      { label: "Worst-5% average", num: true, get: (m) => (m.summary[hz].es_05 === undefined ? "n/a" : spct(m.summary[hz].es_05, 1)), title: "Expected shortfall: the average return in the worst 5% of the method's simulated futures" },
      { label: "Volatility (year)", num: true, get: (m) => (m.summary[hz].vol_annual === undefined ? "n/a" : pct(m.summary[hz].vol_annual, 0)), title: "The spread of the method's simulated returns over this horizon, scaled to a year" },
      { label: "Past record", num: true, title: "How far below the random walk's average CRPS this method's was over the tested forecasts; a tick means it held up after allowing for the number of methods compared", get: (m) => { const s = scores(d, m.id, d.horizon); return m.id === d.evaluation.baseline ? "yardstick" : Q.bad(s.skill) ? "n/a" : spct(s.skill, 1) + (s.beats_baseline ? " ✓" : ""); } }], rows);
  }

  /* The scoreboard: how much better than the random walk each method's forecasts were, at the three horizons. */
  function scoreboard(root, d) {
    const hz = String(d.horizon), ms = evaluated(d).filter((m) => !Q.bad(scores(d, m.id, hz).skill)).sort((a, b) => scores(d, b.id, hz).skill - scores(d, a.id, hz).skill);
    const rows = ms.map((m) => { const s = scores(d, m.id, hz); return { label: shortOf(m.id, d), value: s.skill, text: spct(s.skill, 1) + (s.beats_baseline ? " ✓" : ""),
      tip: [["Average CRPS", Q.bad(s.crps) ? "n/a" : (s.crps * 1000).toFixed(2) + " per 1,000"], ["p-value, adjusted", pval(s.p_adjusted)], ["90% range held", pct((s.coverage || {})["90"], 0)]] }; });
    const ch = h("div", "charts", root);
    const card = chartCard(ch, `How much better than the random walk, ${d.horizon}-day forecasts`, "Skill is how far below the random walk's average CRPS (a score for a whole forecast distribution: lower is better) a method's own was. A tick means the gain held up after allowing for the number of methods compared. Blue is better than the random walk, red worse.", {
      wide: true,
      draw: (b) => C.hbars(b, { rows, format: (v) => (v * 100).toFixed(0) + "%", name: "Skill", label: `Forecast skill against the random walk at ${d.horizon} days`, labelWidth: 150, valueWidth: 70 }),
      table: (b) => {
        const cols = [{ label: "Method", get: (m) => shortOf(m.id, d) }];
        for (const n of d.evaluation.horizons) cols.push({ label: `Skill at ${n} day${n === 1 ? "" : "s"}`, num: true, get: (m) => { const s = scores(d, m.id, n); return Q.bad(s.skill) ? "n/a" : spct(s.skill, 1) + (s.beats_baseline ? " ✓" : ""); } });
        cols.push({ label: "90% range held", num: true, get: (m) => pct((scores(d, m.id, d.horizon).coverage || {})["90"], 0) }, { label: "Below 5% quantile", num: true, get: (m) => pct(scores(d, m.id, d.horizon).var5, 0), title: "How often the outcome fell below the method's 5% quantile: 5% is right" },
          { label: "Calibration p", num: true, get: (m) => pval(scores(d, m.id, d.horizon).pit_p), title: "Kolmogorov-Smirnov p-value that the PIT values are uniform; small means not calibrated" },
          { label: "Seconds", num: true, get: (m) => String(m.seconds), title: "Time spent on this method in the whole evaluation" }, { label: "Could not forecast", num: true, get: (m) => String(m.failures) });
        table(b, cols, ms);
      } });
    card.dataset.role = "forecast-scoreboard";
  }

  /* Skill by horizon: every method and every horizon scored, at a glance. */
  function skillMatrix(root, d) {
    const hs = d.evaluation.horizons, ms = evaluated(d).filter((m) => hs.some((n) => !Q.bad(scores(d, m.id, n).skill))).sort((a, b) => (scores(d, b.id, d.horizon).skill ?? -9) - (scores(d, a.id, d.horizon).skill ?? -9));
    if (ms.length < 2) return;
    const ch = h("div", "charts", root), head = (n) => `${n} day${n === 1 ? "" : "s"}`;
    const card = chartCard(ch, "Skill against the random walk, by method and horizon", "Each cell is how far below the random walk's average CRPS the method's was: blue is better, red is worse. An outlined cell held up after allowing for the number of methods compared. A method that only helps at one horizon is probably luck.", { wide: true,
      draw: (b) => C.matrix(b, { rows: ms.map((m) => shortOf(m.id, d)), cols: hs.map(head), values: ms.map((m) => hs.map((n) => { const s = scores(d, m.id, n); return Q.bad(s.skill) ? null : s.skill; })), diverging: true, mid: 0, format: (v) => spct(v, 0), tipFormat: (v) => spct(v, 1), name: "Skill",
        rowTitle: "Method", colTitle: "Horizon", cellHeight: 24, mark: (i, j) => !!scores(d, ms[i].id, hs[j]).beats_baseline, label: "Forecast skill against the random walk, by method and horizon",
        tipRows: (i, j) => { const s = scores(d, ms[i].id, hs[j]); return [["p-value, adjusted", pval(s.p_adjusted)], ["Forecasts scored", String(s.n || 0)]]; } }),
      table: (b) => table(b, [{ label: "Method", get: (m) => shortOf(m.id, d) }, ...hs.map((n) => ({ label: head(n), num: true, get: (m) => { const s = scores(d, m.id, n); return Q.bad(s.skill) ? "n/a" : spct(s.skill, 1) + (s.beats_baseline ? " ✓" : ""); } }))], ms) });
    card.dataset.role = "forecast-skill-matrix";
  }

  /* The cumulative loss differential over the forecast origins (Goyal and Welch 2008): a line that climbs is a method that kept beating the random walk. */
  function gainCard(root, d) {
    const gain = d.evaluation.gain || {}, dates = d.evaluation.origin_dates || [], ids = Object.keys(gain).filter((id) => gain[id].some((v) => v !== null));
    if (!ids.length || dates.length < 20) return;
    const last = (id) => { const v = gain[id].filter((x) => x !== null); return v.length ? v[v.length - 1] : 0; };
    const order = ids.slice().sort((a, b) => last(b) - last(a)), shown = order.length > 6 ? order.slice(0, 5).concat(order.slice(-1)) : order;
    const ch = h("div", "charts", root);
    const card = chartCard(ch, "Has each method kept beating the random walk?", `The running total, over the forecasts made from ${dates[0]} to ${dates[dates.length - 1]}, of how much lower the method's CRPS was than the random walk's, in units of the random walk's average. A line that climbs is a method that kept helping; one that climbs and goes flat stopped; one below zero has cost accuracy. ${order.length > 6 ? "The five best and the worst are drawn; the Table view lists all." : ""}`, { wide: true,
      draw: (b) => C.line(b, { dates, series: shown.map((id, k) => ({ name: shortOf(id, d), color: Q.colorOf(k), values: gain[id] })), baseline: 0, include: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), height: 280, label: "Cumulative CRPS gain over the random walk", legend: true }),
      table: (b) => table(b, [{ label: "Method", get: (id) => shortOf(id, d) }, { label: "Total gain over the random walk", num: true, get: (id) => last(id).toFixed(2), title: "In units of the random walk's average CRPS: 10 means ten forecasts' worth of its typical loss was saved" }], order) });
    card.dataset.role = "forecast-gain";
  }

  /* Method against method: the skill of the row over the column and the Diebold-Mariano test of it. */
  function pairCard(root, d) {
    const pw = d.evaluation.pairwise;
    if (!pw || pw.ids.length < 3) return;
    const hz = pw.horizon, base = d.evaluation.baseline, rank = (k) => (pw.ids[k] === base ? 0 : scores(d, pw.ids[k], hz).skill ?? -9);
    const order = pw.ids.map((_, k) => k).sort((a, b) => rank(b) - rank(a)), names = order.map((k) => shortOf(pw.ids[k], d)), ch = h("div", "charts", root), bi = pw.ids.indexOf(base);
    const card = chartCard(ch, `Method against method, ${hz}-day forecasts`, "Each cell is how far below the column method's average CRPS the row method's was: blue means the row method was better. An outlined cell is one where the Diebold-Mariano test says the row method is better, at 5% with no allowance for the number of pairs, so with this many pairs some outlined cells are chance.", { wide: true,
      draw: (b) => C.matrix(b, { rows: names.map((n, k) => `${k + 1}. ${n}`), cols: names.map((n, k) => String(k + 1)), values: order.map((a) => order.map((c) => pw.skill[a][c])), diverging: true, mid: 0, format: (v) => spct(v, 0), tipFormat: (v) => spct(v, 1), name: "Row against column",
        rowTitle: "Row beats column by", colTitle: "Column (the number of the method)", cellHeight: 22, mark: (i, j) => pw.p[order[i]][order[j]] !== null && pw.p[order[i]][order[j]] < 0.05, label: "Pairwise forecast skill",
        tipRows: (i, j) => [["One-sided p-value", pval(pw.p[order[i]][order[j]])]] }),
      table: (b) => table(b, [{ label: "Method", get: (k) => shortOf(pw.ids[k], d) }, { label: "Better than the random walk by", num: true, get: (k) => (k === bi ? "" : spct(pw.skill[k][bi], 1)) },
        { label: "Methods it beats at 5%", num: true, get: (k) => String(pw.p[k].filter((p) => p !== null && p < 0.05).length) }, { label: "Methods that beat it at 5%", num: true, get: (k) => String(pw.p.filter((row) => row[k] !== null && row[k] < 0.05).length) }], order) });
    card.dataset.role = "forecast-pairs";
  }

  /* Where each method puts the return at the horizon: the same bins for all, so the shapes can be compared. */
  function densityCard(root, d) {
    const hist = d.now.hist;
    if (!hist || !hist.counts) return;
    const ids = Object.keys(hist.counts);
    if (!F.show.dens || !ids.includes(F.show.dens)) F.show.dens = F.show.method && ids.includes(F.show.method) ? F.show.method : ids.find((id) => id !== d.evaluation.baseline) || ids[0];
    const ch = h("div", "charts", root);
    const card = chartCard(ch, `How the methods see the return in ${hist.horizon} trading days`, "The share of each method's simulated futures in each band of return. The random walk is drawn for comparison: a method that moves the middle left or right has a view on direction; one with a fatter or thinner tail has a view on risk. The markers are the chosen method's 5% worst case and its median.", { wide: true,
      draw: (b) => {
        const id = F.show.dens, sm = d.now.methods[id].summary[String(hist.horizon)], series = [{ name: shortOf(id, d), edges: hist.edges, counts: hist.counts[id], color: css("--s1") }];
        if (id !== d.evaluation.baseline && hist.counts[d.evaluation.baseline]) series.push({ name: shortOf(d.evaluation.baseline, d), edges: hist.edges, counts: hist.counts[d.evaluation.baseline], color: css("--muted") });
        C.histogram(b, { series, markers: [{ x: sm.q05, label: "5% worst" }, { x: sm.median, label: "median" }], xFormat: (v) => (v * 100).toFixed(0) + "%", unit: hist.edges[1] - hist.edges[0], yFormat: (v) => (v * 100).toFixed(0) + "%", height: 260, label: `Distribution of the ${hist.horizon}-day return` });
      },
      table: (b) => table(b, [{ label: "Method", get: (id) => shortOf(id, d) }, { label: "Chance of a rise", num: true, get: (id) => pct(d.now.methods[id].summary[String(hist.horizon)].p_up, 0) }, { label: "Chance of losing 5%+", num: true, get: (id) => pct(d.now.methods[id].summary[String(hist.horizon)].p_loss_5, 0) },
        { label: "Chance of gaining 5%+", num: true, get: (id) => pct(d.now.methods[id].summary[String(hist.horizon)].p_gain_5, 0) }], ids) });
    card.dataset.role = "forecast-density";
    const box = h("div", "tabsmall", card._extra); box.setAttribute("role", "group"); box.setAttribute("aria-label", "method whose distribution is shown");
    const paint = () => { for (const x of box.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x.dataset.value === F.show.dens)); };
    for (const id of ids) { const x = h("button", "", box, shortOf(id, d)); x.type = "button"; x.dataset.value = id; x.addEventListener("click", () => { F.show.dens = id; paint(); card._draw(); }); }
    paint();
  }

  /* Calibration: where the outcome fell among the forecast's own range. */
  function calibrationCard(root, d) {
    const ms = evaluated(d).filter((m) => scores(d, m.id, d.horizon).pit), hz = d.horizon;
    if (!ms.length) return;
    if (!F.show.pit || !ms.some((m) => m.id === F.show.pit)) F.show.pit = ms.some((m) => m.id === F.show.method) ? F.show.method : ms[0].id;
    const card = chartCard(root, "Were the ranges the right size?", "Where the outcome fell within each forecast's own range (0 to 10% is the lowest tenth of the simulated futures). A flat chart means the ranges were the right size. A hump in the middle means they were too wide, a U means too narrow, a slope means the forecasts were biased.", {
      draw: (b) => { const s = scores(d, F.show.pit, hz), labels = s.pit.map((_, i) => `${i * 10}–${i * 10 + 10}%`);
        if (F.show.pitView === "curve") {                                                                  // the cumulative version: share of outcomes at or below each nominal probability, against the diagonal
          const xs = [0], ys = [0]; let acc = 0; s.pit.forEach((v, i) => { acc += v; xs.push((i + 1) / s.pit.length); ys.push(acc); });
          const n = s.n_independent || 0, band = n > 0 ? 1.36 / Math.sqrt(n) : null;                           // the 5% critical distance of the Kolmogorov-Smirnov test for that many independent forecasts
          C.line(b, { xs, xLabel: "Forecast probability (the share of the simulated futures at or below the outcome)", series: [{ name: shortOf(F.show.pit, d), color: css("--s1"), values: ys, dots: true, dotSize: 3 }, { name: "Calibrated", color: css("--muted"), dash: "5 4", width: 1.25, values: xs }],
            bands: band === null ? [] : [{ name: "95% band if calibrated", color: css("--muted"), opacity: 0.14, lo: xs.map((x) => Math.max(0, x - band)), hi: xs.map((x) => Math.min(1, x + band)) }], xFormat: (v) => (v * 100).toFixed(0) + "%", yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => (v * 100).toFixed(1) + "%",
            height: 260, label: `Calibration curve of ${shortOf(F.show.pit, d)}` });
          return;
        }
        C.columns(b, { labels, values: s.pit, yFormat: (v) => (v * 100).toFixed(0) + "%", name: "Share of outcomes", label: `Calibration of ${shortOf(F.show.pit, d)}`, height: 230, colors: () => css("--s1"), tipLabel: (i) => `Outcome in the ${labels[i]} slice of the forecast` }); },
      table: (b) => { const s = scores(d, F.show.pit, hz); table(b, [{ label: "Slice of the forecast", get: (i) => `${i * 10}–${i * 10 + 10}%` }, { label: "Share of outcomes", num: true, get: (i) => pct(s.pit[i], 1) }, { label: "Calibrated would be", num: true, get: () => "10.0%" }], s.pit.map((_, i) => i)); } });
    card.dataset.role = "forecast-calibration";
    const vbox = h("div", "seg", card._extra); vbox.setAttribute("role", "group"); vbox.setAttribute("aria-label", "calibration view");
    for (const [v, label, tip] of [["bars", "Bars", "The share of outcomes in each tenth of the forecast's range"], ["curve", "Curve", "The cumulative share against the diagonal of a perfectly calibrated forecast, with the band the Kolmogorov-Smirnov test allows"]]) {
      const b = h("button", "", vbox, label); b.type = "button"; b.dataset.view = v; b.title = tip; b.setAttribute("aria-pressed", String((F.show.pitView || "bars") === v));
      b.addEventListener("click", () => { F.show.pitView = v; for (const x of vbox.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x.dataset.view === v)); card._draw(); });
    }
    const box = h("div", "tabsmall", card._extra); box.setAttribute("role", "group"); box.setAttribute("aria-label", "method whose calibration is shown");
    const paint = () => { for (const b of box.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.value === F.show.pit)); facts(); };
    const k = h("div", "earnrow", card);
    const facts = () => {
      k.replaceChildren(); const s = scores(d, F.show.pit, hz), cov = s.coverage || {}, cp = s.coverage_p || {};
      tile(k, "90% range held the outcome", pct(cov["90"], 0), { text: `90% is right (p = ${pval(cp["90"])})`, dir: cp["90"] !== null && cp["90"] < 0.05 ? "down" : "up" }, "", "How often the realised return fell inside the central 90% of the method's simulated futures, tested with the forecasts one horizon apart so that they are independent.");
      tile(k, "50% range held the outcome", pct(cov["50"], 0), { text: `50% is right (p = ${pval(cp["50"])})`, dir: cp["50"] !== null && cp["50"] < 0.05 ? "down" : "up" }, "", "The same for the central half.");
      tile(k, "Fell below the 5% quantile", pct(s.var5, 1), { text: `5% is right (p = ${pval(s.var5_p)})`, dir: s.var5_p !== null && s.var5_p < 0.05 ? "down" : "up" }, "", "The value-at-risk check: how often the outcome was worse than the method's 5% worst case. Too often means the method understates the risk of a fall.");
      tile(k, "Calibration test", pval(s.pit_p), { text: "small means not calibrated" }, "", "Kolmogorov-Smirnov test that the outcomes are spread evenly across the forecast ranges.");
    };
    for (const m of ms) { const b = h("button", "", box, shortOf(m.id, d)); b.type = "button"; b.dataset.value = m.id; b.title = m.label; b.addEventListener("click", () => { F.show.pit = m.id; paint(); card._draw(); }); }
    paint();
  }

  /* The average and the direction: a different question from the spread, and a harder one. */
  function meanCard(root, d) {
    const hz = String(d.horizon), ms = evaluated(d).filter((m) => scores(d, m.id, hz).mean);
    if (!ms.length) return;
    const card = W.card(root, "Did any method forecast the average return or the direction?", "The average return is compared with simply using the average of the last five years (the standard benchmark: out-of-sample R-squared above zero means better; the Clark-West test says whether that is more than luck). The direction is compared with always saying 'up', which is right as often as the market rose.");
    card.dataset.role = "forecast-mean";
    table(card, [{ label: "Method", get: (m) => shortOf(m.id, d) }, { label: "R-squared against the 5-year average", num: true, get: (m) => spct((scores(d, m.id, hz).mean || {}).r2, 1) }, { label: "Clark-West p", num: true, get: (m) => pval((scores(d, m.id, hz).mean || {}).p) },
      { label: "Called the direction right", num: true, get: (m) => pct((scores(d, m.id, hz).direction || {}).hit, 0) }, { label: "Always saying up", num: true, get: (m) => pct((scores(d, m.id, hz).direction || {}).always_up, 0) },
      { label: "Edge", num: true, get: (m) => { const e = (scores(d, m.id, hz).direction || {}).edge; return Q.bad(e) ? "n/a" : (e >= 0 ? "+" : "−") + Math.abs(e * 100).toFixed(0) + " points"; } }, { label: "p", num: true, get: (m) => pval((scores(d, m.id, hz).direction || {}).p) }], ms);
    h("p", "hint", card, "With this many methods and one stock, one of them looks good by chance now and then: the page asks for a p-value below 5% divided by the number of methods before it calls one useful.");
  }
  function assumptionsCard(root, d) {
    const card = W.card(root, "What each method assumes, and where it breaks");
    for (const m of [{ id: d.evaluation.baseline }, ...evaluated(d)]) {
      const info = methodOf(m.id) || d.evaluation.methods.find((x) => x.id === m.id); if (!info) continue;
      const det = h("details", "how", card); h("summary", "", det, info.label); h("p", "", det, info.text); const ul = h("ul", "", det); h("li", "", ul, "Assumes: " + info.assumes); h("li", "", ul, "Breaks when: " + info.fails);
      Q.refsLine(det, info.references);
    }
  }
  function learnedCard(root, d) {
    const card = W.card(root, "What the methods learned from the last years", "The numbers each method fitted, and the state it is in on the last day. They are what its forecast is made of.");
    const fmt = (v) => (typeof v === "number" ? (Math.abs(v) >= 100 || Number.isInteger(v) ? String(Math.round(v * 100) / 100) : +v.toPrecision(3)) : typeof v === "boolean" ? (v ? "yes" : "no") : String(v));
    const flat = (v) => (Array.isArray(v) ? v.map((x) => (Array.isArray(x) ? "[" + x.map(fmt).join(", ") + "]" : fmt(x))).join("; ") : v && typeof v === "object" ? Object.entries(v).map(([k, x]) => `${k}: ${fmt(x)}`).join("; ") : fmt(v));
    for (const m of Object.values(d.now.methods)) {
      const keys = Object.keys(m.info || {}).filter((k) => !/matrix/.test(k)); if (!keys.length) continue;
      const det = h("details", "how", card); h("summary", "", det, shortOf(m.id, d)); const ul = h("ul", "", det);
      for (const k of keys) h("li", "", ul, `${human(k)}: ${flat(m.info[k])}`);
    }
  }
  function notesCard(root, d) {
    const card = W.card(root, "Read before relying on it"); const ul = h("ul", "", card);
    for (const t of d.notes) h("li", "", ul, t);
    h("li", "", ul, `Tested on ${d.history_days.toLocaleString("en-US")} days of prices from ${d.first_date}; the ranking could be different in another period or for another stock.`);
    h("li", "", ul, "How the forecasts are scored and compared (CRPS, Diebold-Mariano, Clark-West, the probability integral transform, the coverage test of Kupiec, the cumulative gain plot): the sources are listed below.");
    h("li", "", ul, "A forecast is a statement about a range, not an instruction to trade. How good a forecast is and how much money a rule built on it makes are different questions; the Backtest tab answers the second.");
    if (F.plan && F.plan.evaluation_references) Q.refsLine(card, F.plan.evaluation_references);
  }
  function downloadCsv(d) {
    const rows = [["method", "step", "date", "q05", "q25", "median", "q75", "q95"]];
    for (const m of Object.values(d.now.methods)) for (let i = 0; i < d.horizon; i++) rows.push([m.id, i + 1, d.future_dates[i], m.fan["0.05"][i], m.fan["0.25"][i], m.fan["0.5"][i], m.fan["0.75"][i], m.fan["0.95"][i]]);
    W.csvDownload(`forecast_${d.ticker}_${d.asof}.csv`, rows);
  }

  F.watch = () => { watchRuns(); };
  Q.registerWorkspace("forecast", F);
  document.getElementById("tab-forecast").addEventListener("click", () => setTimeout(F.watch, 0));
  if (Q.store("tab") === "forecast") setTimeout(F.watch, 400);
})();
