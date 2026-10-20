/* The Fundamentals tab: company fundamentals kept in a local database, factor studies, automated alpha discovery, the alpha library and its monitor, and the backtest of an alpha.
   This file is the frame (the lab object, the requests, the shared pieces of a form) and three of the seven views: Overview, Company explorer and Data manager. fundamentals_lab.js adds the research views
   (Factor lab, Alpha discovery, Alpha library, Alpha monitor) to the same object. It talks to /api/fund/*; anything slow runs as a job on the app's worker, so closing the page does not stop it.
   Two databases are kept apart: Real data and a Demo made of a simulated world. A provider key is never typed into this page (it goes into a Replit secret). Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable, api = Q.api;
  const GUIDE = "fundamentals";
  const ACTIVE = new Set(["queued", "running"]);
  const ICON = { good: "✓", mixed: "!", bad: "✗" };
  const STATE_WORD = { Healthy: "Healthy", Weakening: "Weakening", Decaying: "Decaying", Broken: "Broken" };
  const isOpen = () => !Q.$(FD.view).hidden;
  const spct = (v, d = 1) => { if (Q.bad(v)) return "n/a"; const t = Math.abs(v * 100).toFixed(d); return (+t === 0 ? "" : v >= 0 ? "+" : "−") + t + "%"; };
  const sgn = (v, d = 3) => (Q.bad(v) ? "n/a" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d));
  const compact = (v) => { if (Q.bad(v)) return "n/a"; const a = Math.abs(v), s = v < 0 ? "−" : ""; return a >= 1e12 ? s + (a / 1e12).toFixed(2) + "T" : a >= 1e9 ? s + (a / 1e9).toFixed(2) + "B" : a >= 1e6 ? s + (a / 1e6).toFixed(1) + "M" : a >= 1e3 ? s + (a / 1e3).toFixed(1) + "k" : s + a.toFixed(a >= 10 ? 0 : 2); };
  const human = (k) => String(k).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
  const day = (s) => (s ? String(s).slice(0, 10) : "");
  const qs = W.qs;
  const KEY_SLOTS = ["OPENFILINGS_API_KEY_1", "OPENFILINGS_API_KEY_2", "OPENFILINGS_API_KEY_3", "OPENFILINGS_API_KEY_4", "OPENFILINGS_API_KEY_5"];

  const FD = {
    id: "fundamentals", root: "#fundamentals", view: "#view-fundamentals", title: "Fundamentals", api: 11, mode: "overview", busyBy: {}, jobs: {}, epoch: 0, results: {}, errors: {}, cache: new Map(), guard: null, go: null, progress: null,
    about: "Company fundamentals from around the world in a local database, tested as factors and searched for alphas, with the evidence counted honestly.",
    modes: [["overview", "Overview"], ["companies", "Company explorer"], ["factors", "Factor lab"], ["discovery", "Alpha discovery"], ["library", "Alpha library"], ["monitor", "Alpha monitor"], ["data", "Data manager"]],
    db: Q.store("fund.db") === "demo" ? "demo" : "real", facets: {}, views: {},
    co: { q: "", country: "", sector: "", offset: 0, limit: 25, id: null },
    dm: { symbols: "", market: "us", start_year: "2015", budget: "", plan: null, report: null, prices: null, price_start: "2010-01-01", price_refresh: false, price_note: "", companies: "120", from: "2009", to: "2022", seed: "7",
          strong: false, tested: null },
    uni: { countries: new Set(), sectors: new Set(), min_cap: "", min_vol: "", developed: false, delisted: true, freq: "M", lag: "1", start: "", end: "", info_lag: "1" },
  };
  /* The lab frame asks ``busy`` of the view on screen. Each view has its own: a long search in Alpha discovery does not stop the Company explorer, the library or the Data manager from being read. */
  Object.defineProperty(FD, "busy", { get: () => FD.busyBy[FD.mode] || "", enumerable: true });
  window.FD = FD;
  FD.runLabel = { overview: "Refresh", companies: "Search", factors: "Study this factor", discovery: "Start the search", get library() { return FD.views.libraryLabel ? FD.views.libraryLabel() : "Refresh the list"; }, monitor: "Run the monitor now", data: "Refresh the status" };
  FD.blurb = {
    overview: "What the database holds, what is missing, and what has been found.",
    companies: "Look up a company, its statements as they were filed, and the screens built on them.",
    factors: "Test one factor or formula against what happened next, or scan every factor in the library at once.",
    discovery: "Search for formulas that predict returns, with every attempt counted and the best tested once on dates they never saw.",
    get library() { return FD.views.libraryBlurb ? FD.views.libraryBlurb() : "The alphas found so far, the ones that failed, and ways of putting them together."; },
    monitor: "Check whether the alphas in the library still work on the data now in the database.",
    data: "Companies, statements, prices, the allowance of each provider and every key slot.",
  };

  /* ---------- requests ---------- */
  const dbq = (extra) => Object.assign({ db: FD.db }, extra || {});
  FD.read = (path, query) => api(path + qs(dbq(query)));
  FD.send = (path, body) => api(path, Object.assign({ db: FD.db }, body || {}));
  /* Start a job for ``mode``, wait for it, keep the page's progress line current. ``say`` names the work in the progress line. */
  FD.runJob = async (mode, path, body, say) => {
    const started = await FD.send(path, body), epoch = FD.epoch;
    FD.jobs[mode] = started.job;
    try {
      return { result: await Q.waitJob(started.job, (v) => {
        if (epoch !== FD.epoch) return;
        FD.busyBy[mode] = `${say}: ${v.message} (${Math.round(v.elapsed)} s)`;
        if (mode === FD.mode) Q.syncLab(FD);
        paintBusy(); paintBackground();
      }, 600), started };
    } finally { if (FD.jobs[mode] === started.job) delete FD.jobs[mode]; }
  };
  /* Run an action for ``mode`` with its busy line and error handling; ``fn(run)`` returns what to file under ``mode`` and starts a job with ``run.job(path, body, say)``.
     An answer that arrives after the data was switched is about the other database and is dropped. */
  FD.act = async (mode, message, fn) => {
    if (FD.busyBy[mode]) { if (mode === FD.mode) Q.toast("Still working: " + FD.busyBy[mode]); return; }
    const blocked = mode === FD.mode ? FD.check() : "";                                      // the buttons on an empty page are not the run button: they meet the same guard
    if (blocked) { Q.toast(blocked); return; }
    const epoch = FD.epoch, run = { job: (path, body, say) => FD.runJob(mode, path, body, say) };
    FD.busyBy[mode] = message; FD.errors[mode] = null; paintBackground();
    if (mode === FD.mode) { Q.syncLab(FD); if (isOpen()) FD.draw(); }
    let got = null, failed = null;
    try { got = { data: await fn(run), body: {} }; }
    catch (e) { failed = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + Q.RESTART : e.message; }
    if (epoch !== FD.epoch) return;
    delete FD.busyBy[mode];
    if (failed) FD.errors[mode] = failed; else FD.results[mode] = got;
    paintBackground();
    if (mode === FD.mode) { Q.syncLab(FD); if (isOpen()) FD.draw(); }
  };
  function paintBusy() { const el = Q.$('[data-role="fund-progress"]'); if (el) el.textContent = FD.busy; }
  /* What the other views are doing, in the panel of this one. */
  function paintBackground() {
    const box = Q.$('[data-role="fund-bg"]'); if (!box || !box.isConnected) return;
    const others = Object.entries(FD.busyBy).filter(([m]) => m !== FD.mode);
    box.hidden = !others.length; box.replaceChildren();
    for (const [m, text] of others) {
      const row = h("div", "row tight", box), name = (FD.modes.find((x) => x[0] === m) || [m, m])[1];
      h("span", "hint", row, `${name}: ${text}`);
      const go = h("button", "btn small", row, "Open"); go.type = "button"; go.setAttribute("aria-label", "open " + name); go.addEventListener("click", () => FD.goto(m));
    }
  }
  FD.paintBackground = paintBackground;
  FD.stop = async (button) => {
    const job = FD.jobs[FD.mode]; if (!job) return;
    if (button) { button.disabled = true; button.textContent = "Stopping after this step…"; }
    try { await api("/api/fund/stop", { job }); } catch (e) { Q.toast(e.message); if (button) { button.disabled = false; button.textContent = "Stop and keep what is done"; } }
  };
  FD.catalogue = () => { if (!FD.cache.has("catalogue")) FD.cache.set("catalogue", api("/api/fund/catalogue").catch((e) => { FD.cache.delete("catalogue"); throw e; })); return FD.cache.get("catalogue"); };
  FD.loadFacets = async () => {
    if (FD.facets[FD.db]) return FD.facets[FD.db];
    try { const r = await FD.read("/api/fund/companies", { limit: 1 }); FD.facets[FD.db] = { countries: r.countries, sectors: r.sectors, total: r.total }; }
    catch (e) { FD.facets[FD.db] = { countries: [], sectors: [], total: 0, error: e.message }; }
    return FD.facets[FD.db];
  };
  FD.resetData = () => { FD.epoch++; FD.busyBy = {}; FD.jobs = {}; FD.results = {}; FD.errors = {}; FD.cache = new Map(); FD.facets = {}; FD.co.id = null; FD.dm.plan = null; FD.dm.report = null; if (FD.views.reset) FD.views.reset(); };

  /* ---------- the parts of a form every view uses ---------- */
  FD.sec = (panel, title, open) => { const d = h("details", "", panel); d.open = open !== false; h("summary", "", d, title); return h("div", "body", d); };
  FD.after = () => Q.syncLab(FD);
  FD.demoBanner = (root) => {
    if (FD.db !== "demo") return;
    const b = Q.labBanner(root, "Demo data. ", "These companies are simulated, with a few alphas planted in them so that the machinery has something to find. Nothing found here says anything about real markets.");
    b.dataset.role = "fund-demo";
  };
  FD.dbSwitch = (panel) => {
    const box = h("div", "dbswitch", panel);
    h("span", "grouplabel", box, "Data");
    const seg = W.seg(box, [["real", "Real data"], ["demo", "Demo (simulated)"]], FD.db, (v) => {
      if (v === FD.db) return;
      FD.db = v; Q.store("fund.db", v); FD.resetData(); Q.renderLab(FD);
    });
    seg.dataset.role = "fund-db";
    h("p", "hint", box, FD.db === "demo" ? "A simulated world, kept in its own file. Fetching from a provider is switched off here, so a simulated company never mixes with a real one." : "Companies and statements from the provider you set up, kept in a local database that every study reads. A study never calls a provider.");
  };
  /* The universe and timing of a study: shared by the Factor lab, Alpha discovery and the combination views. */
  FD.universeBody = () => {
    const u = FD.uni, body = { freq: u.freq, entry_lag: Number(u.lag) };
    const universe = {};
    if (u.countries.size) universe.countries = [...u.countries];
    if (u.sectors.size) universe.sectors = [...u.sectors];
    if (u.min_cap !== "") universe.min_market_cap_usd = Number(u.min_cap);
    if (u.min_vol !== "") universe.min_dollar_volume_usd = Number(u.min_vol);
    if (u.developed) universe.developed_only = true;
    universe.include_inactive = !!u.delisted;
    body.universe = universe;
    if (u.start) body.start = u.start;
    if (u.end) body.end = u.end;
    if (u.info_lag !== "") body.pit = { info_lag_days: Number(u.info_lag) };
    for (const k of ["min_cap", "min_vol", "lag", "info_lag"]) if (u[k] !== "" && !isFinite(Number(u[k]))) throw new Error("The size, volume and lag fields must be numbers");
    return body;
  };
  FD.universeSection = (panel, title, search) => {
    const b = FD.sec(panel, title || "1. Which companies and when"), u = FD.uni, f = FD.facets[FD.db];
    if (!f) { h("p", "hint", b, "Loading the countries and sectors…"); FD.loadFacets().then(() => { if (isOpen()) Q.renderLab(FD); }); return b; }
    if (f.total === 0) h("p", "hint tone-warn", b, FD.db === "demo" ? "The demo has not been made yet: open the Data manager." : "There are no companies yet: open the Data manager.");
    const pick = (label, list, chosen) => {
      if (!list.length) return;
      const d = h("details", "how", b); h("summary", "", d, `${label}${chosen.size ? ` (${chosen.size} chosen)` : " (all)"}`);
      Q.checkList(d, "Tick to restrict; none ticked means all:", list.map((x) => [x, x]), chosen, () => { d.querySelector("summary").textContent = `${label}${chosen.size ? ` (${chosen.size} chosen)` : " (all)"}`; FD.after(); });
    };
    pick("Countries", f.countries, u.countries); pick("Sectors", f.sectors, u.sectors);
    Q.labField(b, { key: "min_cap", label: "Smallest market value (US dollars)", kind: "number", min: 0, placeholder: "no minimum" }, u, FD.after);
    Q.labField(b, { key: "min_vol", label: "Smallest daily trading value (US dollars)", kind: "number", min: 0, placeholder: "no minimum" }, u, FD.after);
    Q.labField(b, { key: "developed", label: "Developed markets only", kind: "bool" }, u, FD.after);
    Q.labField(b, { key: "delisted", label: "Keep companies that have since delisted", kind: "bool", help: "Leaving them out is how survivorship bias gets in: the sample would contain only the companies that did well enough to stay listed." }, u, FD.after);
    Q.labField(b, { key: "freq", label: "How often the portfolio is rebalanced", kind: "choice", options: [["W", "Weekly"], ["M", "Monthly"], ["Q", "Quarterly"]] }, u, FD.after);
    Q.labField(b, { key: "lag", label: "Days between a signal and the trade", kind: "number", step: 1, min: 0, max: 10, help: "The signal is computed at the close; the trade is struck this many days later, so it is never traded at the price it was computed from." }, u, FD.after);
    if (search) return b;                                                                  // a search runs on the whole history with the usual lag; the dates and the filing lag belong to a study
    Q.labField(b, { key: "info_lag", label: "Days after a filing before it is used", kind: "number", step: 1, min: 0, max: 60, help: "A filing may have come after the close; 1 means it is first used the next day." }, u, FD.after);
    Q.labField(b, { key: "start", label: "From (blank = all history)", kind: "text", placeholder: "2012-01-31" }, u, FD.after);
    Q.labField(b, { key: "end", label: "To (blank = latest)", kind: "text", placeholder: "2022-12-30" }, u, FD.after);
    return b;
  };

  /* ---------- the lab interface ---------- */
  FD.needsData = () => {
    const f = FD.facets[FD.db];
    if (!f || f.total !== 0) return "";
    return FD.db === "demo" ? "Make the simulated world in the Data manager first." : "Add companies and fetch their statements in the Data manager first.";
  };
  FD.check = () => {
    const v = FD.views[FD.mode]; if (v && v.check) return v.check();
    return "";
  };
  FD.sections = (panel) => {
    FD.dbSwitch(panel);
    const bg = h("div", "dbswitch bgnote", panel); bg.dataset.role = "fund-bg"; bg.setAttribute("role", "status"); bg.hidden = true;
    paintBackground();
    const v = FD.views[FD.mode]; if (v && v.sections) v.sections(panel);
  };
  FD.run = () => { const v = FD.views[FD.mode]; if (v && v.run) return v.run(); };
  FD.auto = () => { const v = FD.views[FD.mode]; if (v && v.auto) v.auto(); };
  FD.empty = (root) => {
    FD.demoBanner(root);
    if (FD.busy) { busyCard(root); return; }
    const v = FD.views[FD.mode]; if (v && v.empty) v.empty(root);
  };
  FD.build = (root, data, body) => {
    FD.demoBanner(root);
    const v = FD.views[FD.mode]; if (v && v.build) v.build(root, data, body);
  };
  FD.draw = () => Q.drawLab(FD);
  function busyCard(root) {
    const c = h("div", "card empty", root); h("div", "spinner", c).style.margin = "0 auto 12px"; h("h2", "", c, "Working…");
    h("p", "", c, FD.busy).dataset.role = "fund-progress";
    h("p", "hint", c, "Closing this page does not stop it: the app keeps going.");
    if (FD.jobs[FD.mode]) { const stop = h("button", "btn", c, "Stop and keep what is done"); stop.type = "button"; stop.dataset.action = "fund-stop"; stop.addEventListener("click", () => FD.stop(stop)); }
  }
  FD.busyCard = busyCard;

  /* ---------- small pieces ---------- */
  const badge = (parent, text, cls) => { const b = h("span", "pill " + (cls || ""), parent, text); return b; };
  FD.badge = badge;
  FD.notes = (root, title, list) => {
    if (!list || !list.length) return;
    const card = W.card(root, title); const ul = h("ul", "", card); for (const t of list) h("li", "", ul, t);
    return card;
  };
  FD.explain = (root, title, text) => { const d = h("details", "how", root); h("summary", "", d, title); h("p", "", d, text); return d; };
  FD.emptyCard = (root, title, paragraphs, button) => {
    const c = h("div", "card empty", root); h("h2", "", c, title); for (const t of [].concat(paragraphs)) h("p", "", c, t);
    if (button) { const b = h("button", "btn primary", c, button.label); b.type = "button"; b.addEventListener("click", button.fn); }
    return c;
  };
  FD.goto = (mode) => { FD.mode = mode; Q.renderLab(FD); };
  FD.fmt = { spct, sgn, compact, human, day };

  /* =========================================================== Overview */
  const OV = FD.views.overview = {};
  OV.run = () => FD.act("overview", "Reading the database…", () => FD.read("/api/fund/overview"));
  OV.auto = () => OV.run();
  OV.empty = (root) => FD.emptyCard(root, "Reading the database…", "");
  OV.sections = (panel) => {
    const b = FD.sec(panel, "What this shows");
    h("p", "hint", b, "The state of the local database: how many companies, how much history, how clean, what the library holds. Nothing here calls a provider.");
    const go = h("button", "btn", b, "Open the Data manager"); go.type = "button"; go.addEventListener("click", () => FD.goto("data"));
  };
  OV.build = (root, d) => {
    Q.labHead(root, "Overview", `${d.counts.companies.toLocaleString("en-US")} companies · ${d.counts.with_financials.toLocaleString("en-US")} with statements · ${d.counts.price_rows.toLocaleString("en-US")} daily prices`, GUIDE);
    const ready = d.readiness;
    if (!ready.ready) {
      const c = h("div", "card empty", root); c.dataset.role = "fund-not-ready"; h("h2", "", c, "Not ready to study yet");
      for (const p of ready.problems) h("p", "", c, p);
      const go = h("button", "btn primary", c, "Open the Data manager"); go.type = "button"; go.addEventListener("click", () => FD.goto("data"));
    }
    const k = h("div", "earnrow", root);
    tile(k, "Companies", d.counts.companies.toLocaleString("en-US"), { text: `${d.counts.countries} countries · ${d.counts.markets} markets` });
    tile(k, "With statements", d.counts.with_financials.toLocaleString("en-US"), { text: `${d.counts.annual_periods.toLocaleString("en-US")} annual · ${d.counts.quarterly_periods.toLocaleString("en-US")} quarterly periods` });
    tile(k, "Daily prices", d.counts.price_rows.toLocaleString("en-US"), { text: `${d.counts.observations.toLocaleString("en-US")} figures stored` });
    tile(k, "Restated filings", d.counts.restated_vintages.toLocaleString("en-US"), { text: "kept beside the original, never overwriting it" });
    tile(k, "Alphas in the library", String(Object.values(d.library.library || {}).reduce((a, b) => a + b, 0)), { text: `${d.library.graveyard} failed ideas kept in the graveyard` });
    tile(k, "Ideas tried in all", String(d.library.tried || 0), { text: "every one counts against a result" });
    const ch = h("div", "charts", root);
    if (d.by_country.length) chartCard(ch, "Where the companies are", "Companies by country, and the share of them that have annual statements.", {
      draw: (b) => C.hbars(b, { rows: d.by_country.map((r) => ({ label: r.country, value: r.companies, text: `${r.companies}`, tip: [["With annual statements", pct(r.annual_coverage, 0)], ["Median history", num(r.median_history_years, 1) + " years"]] })), format: (v) => String(Math.round(v)), name: "Companies", label: "Companies by country", labelWidth: 54, valueWidth: 50, color: () => css("--s1") }),
      table: (b) => table(b, [{ label: "Country", get: (r) => r.country }, { label: "Companies", num: true, get: (r) => String(r.companies) }, { label: "With annual statements", num: true, get: (r) => pct(r.annual_coverage, 0) }, { label: "Median years of history", num: true, get: (r) => num(r.median_history_years, 1) }], d.by_country) });
    if (d.by_sector.length) chartCard(ch, "Which sectors", "Companies by sector. A study that is mostly one sector is a study of that sector.", {
      draw: (b) => C.hbars(b, { rows: d.by_sector.map((r) => ({ label: r.sector, value: r.companies })), format: (v) => String(Math.round(v)), name: "Companies", label: "Companies by sector", labelWidth: 150, valueWidth: 40, color: () => css("--s3") }),
      table: (b) => table(b, [{ label: "Sector", get: (r) => r.sector }, { label: "Companies", num: true, get: (r) => String(r.companies) }], d.by_sector) });
    if (d.metric_coverage.length) chartCard(ch, "How complete the statements are", "The share of companies that report each figure at least once. A figure few companies report cannot support a factor.", {
      wide: true,
      draw: (b) => C.hbars(b, { rows: d.metric_coverage.slice(0, 24).map((r) => ({ label: r.metric_code, value: r.company_share })), format: (v) => (v * 100).toFixed(0) + "%", name: "Companies reporting", label: "Coverage of each figure", labelWidth: 180, valueWidth: 46, color: () => css("--s2") }),
      table: (b) => table(b, [{ label: "Figure", get: (r) => r.metric_code }, { label: "Companies", num: true, get: (r) => String(r.companies) }, { label: "Share", num: true, get: (r) => pct(r.company_share, 0) }, { label: "Observations", num: true, get: (r) => String(r.observations) }], d.metric_coverage) });
    const q = d.quality, qc = W.card(root, "Data quality", "Every figure carries a flag. Flagged figures are treated as missing in research, never repaired silently.");
    qc.dataset.role = "fund-quality";
    const obs = Object.entries(q.observations || {}).map(([flag, n]) => [flag, n]), per = Object.entries(q.periods || {}).map(([flag, n]) => [flag, n]);
    if (obs.length) table(qc, [{ label: "Figures by flag", get: (r) => r[0].replace(/_/g, " ") }, { label: "Count", num: true, get: (r) => r[1].toLocaleString("en-US") }], obs);
    if (per.length) table(qc, [{ label: "Filing dates", get: (r) => ({ filed: "a real filing date", restated: "a restated figure, dated when the restatement was published", estimated: "an estimated lag after the period end" })[r[0]] || r[0].replace(/_/g, " ") }, { label: "Periods", num: true, get: (r) => r[1].toLocaleString("en-US") }], per);
    if (!obs.length && !per.length) h("p", "hint", qc, "No statements have been loaded yet, so there is nothing to check.");
    const lib = W.card(root, "The alpha library", "Candidates have passed tests on dates they were not chosen on and still need time. Nothing is called validated without the evidence.");
    lib.dataset.role = "fund-library-summary";
    const counts = Object.entries(d.library.library || {});
    if (counts.length) table(lib, [{ label: "Status", get: (r) => r[0] }, { label: "Alphas", num: true, get: (r) => String(r[1]) }], counts); else h("p", "hint", lib, "Nothing has been found yet. Start a search in Alpha discovery.");
    if (d.library.top.length) table(lib, [{ label: "Formula", cls: "formula", get: (r) => r.text }, { label: "Status", get: (r) => r.status }, { label: "IC", num: true, get: (r) => sgn(r.ic) }, { label: "Out-of-sample IC", num: true, get: (r) => sgn(r.oos_ic) }, { label: "Sharpe", num: true, get: (r) => num(r.sharpe) }], d.library.top);
    const recent = W.card(root, "Recent work");
    if (d.runs.length) { h("h4", "subhead", recent, "Discovery runs"); table(recent, [{ label: "Run", get: (r) => r.run_id }, { label: "Started", get: (r) => day(r.created_at) }, { label: "Status", get: (r) => r.status }, { label: "Formulas", num: true, get: (r) => String((r.stage_counts || {}).next_index || 0) }, { label: "Tested", num: true, get: (r) => String((r.stage_counts || {}).tested || 0) }], d.runs); }
    if (d.backtests.length) { h("h4", "subhead", recent, "Backtests"); table(recent, [{ label: "Signal", get: (r) => (r.spec.signal || {}).label || r.experiment_id }, { label: "Run", get: (r) => day(r.created_at) }, { label: "Sharpe", num: true, get: (r) => num((r.summary || {}).sharpe) }, { label: "Return a year", num: true, get: (r) => pct((r.summary || {}).cagr) }, { label: "Deflated Sharpe", num: true, get: (r) => num(r.summary.deflated_sharpe) }], d.backtests); }
    if (!d.runs.length && !d.backtests.length) h("p", "hint", recent, "Nothing has been run yet.");
    FD.explain(root, "How this is built", "Each company is a security with its own id, never a bare ticker. A statement figure is used only from the day it was public (plus the lag you choose), restated figures are kept beside the originals, and a missing figure is never turned into a zero. The number of ideas tried is counted, because the best of many tries looks good by luck.");
    if (d.simulated) {
      const [n, from, to, seed] = String(d.simulated).split(":");
      FD.notes(root, "About this database", [`This is a simulated world: ${n} companies, ${from} to ${to}, seed ${seed}. A few alphas are hidden in it on purpose, so that a search has something to find.`]);
    }
  };

  /* =========================================================== Company explorer */
  const CO = FD.views.companies = {};
  CO.run = () => { FD.co.offset = 0; FD.co.id = null; return CO.load(); };
  CO.load = () => FD.act("companies", "Searching…", () => FD.read("/api/fund/companies", { q: FD.co.q, country: FD.co.country, sector: FD.co.sector, limit: FD.co.limit, offset: FD.co.offset }));
  CO.auto = () => { if (!FD.co.id) CO.load(); };
  CO.empty = (root) => { if (FD.co.id) CO.detail(root, FD.co.id); else FD.emptyCard(root, "Company explorer", "Search for a company by name or ticker."); };
  CO.sections = (panel) => {
    const b = FD.sec(panel, "Find a company"), s = FD.co;
    Q.labField(b, { key: "q", label: "Name, ticker or ISIN", kind: "text", placeholder: "apple, 7203, US0378331005" }, s, FD.after);
    const f = FD.facets[FD.db];
    if (f) {
      Q.labField(b, { key: "country", label: "Country", kind: "choice", options: [["", "All"]].concat(f.countries.map((c) => [c, c])) }, s, FD.after);
      Q.labField(b, { key: "sector", label: "Sector", kind: "choice", options: [["", "All"]].concat(f.sectors.map((c) => [c, c])) }, s, FD.after);
    } else FD.loadFacets().then(() => { if (isOpen() && FD.mode === "companies") Q.renderLab(FD); });
    h("p", "hint", b, "Open a company to see every figure with the date it became public, the ratios built from them, the DuPont split of the change in return on equity, the Piotroski F-score and the accounting-anomaly screen.");
  };
  CO.build = (root, d) => {
    if (FD.co.id) { CO.detail(root, FD.co.id); return; }
    Q.labHead(root, "Companies", `${d.total.toLocaleString("en-US")} match${d.total === 1 ? "es" : ""}${FD.co.q ? ` “${FD.co.q}”` : ""}`, GUIDE);
    const card = h("div", "card roomy", root); card.dataset.role = "fund-companies";
    if (!d.companies.length) { h("p", "hint", card, d.total === 0 && !FD.co.q && !FD.co.country && !FD.co.sector ? "There are no companies yet. Add some in the Data manager." : "No company matches."); return; }
    const wrap = h("div", "scroll", card), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t));
    for (const [label, numeric] of [["Ticker", 0], ["Company", 0], ["Country", 0], ["Sector", 0], ["Annual", 1], ["Last period", 0], ["Price days", 1], ["Open", 2]]) { const th = h("th", numeric === 1 ? "num" : "", hd); if (numeric === 2) h("span", "vh", th, label); else th.textContent = label; }
    const tb = h("tbody", "", t);
    for (const r of d.companies) {
      const tr = h("tr", "", tb); tr.dataset.id = r.security_id;
      h("td", "", tr, r.ticker + (r.exchange ? " · " + r.exchange : "")); h("td", "", tr, r.company_name || ""); h("td", "", tr, r.country || ""); h("td", "", tr, r.sector || "");
      h("td", "num", tr, String(r.annual)); h("td", "", tr, day(r.last_period_end)); h("td", "num", tr, String(r.price_days));
      const open = h("button", "btn small", h("td", "", tr), "Open"); open.type = "button"; open.dataset.action = "open-company"; open.setAttribute("aria-label", "open " + r.ticker);
      open.addEventListener("click", () => { FD.co.id = r.security_id; FD.draw(); window.scrollTo({ top: 0 }); });
    }
    const pager = h("div", "row tight", card), from = d.offset + 1, to = d.offset + d.companies.length;
    h("span", "hint", pager, `${from.toLocaleString("en-US")} to ${to.toLocaleString("en-US")} of ${d.total.toLocaleString("en-US")}`);
    const prev = h("button", "btn small", pager, "Previous"); prev.type = "button"; prev.disabled = d.offset === 0; prev.addEventListener("click", () => { FD.co.offset = Math.max(0, d.offset - d.limit); CO.load(); });
    const next = h("button", "btn small", pager, "Next"); next.type = "button"; next.disabled = to >= d.total; next.addEventListener("click", () => { FD.co.offset = d.offset + d.limit; CO.load(); });
  };
  CO.detail = (root, id) => {
    const back = h("button", "btn small", root, "← Back to the list"); back.type = "button"; back.dataset.action = "back-to-list"; back.addEventListener("click", () => { FD.co.id = null; FD.draw(); });
    const key = "company:" + FD.db + ":" + id, holder = h("div", "stack", root);
    const show = (d) => { holder.replaceChildren(); CO.view(holder, d); };
    if (FD.cache.has(key) && FD.cache.get(key).data) { show(FD.cache.get(key).data); return; }
    h("p", "hint", holder, "Reading the company…");
    FD.read("/api/fund/company", { id }).then((d) => { FD.cache.set(key, { data: d }); if (isOpen() && FD.co.id === id) show(d); })
      .catch((e) => { holder.replaceChildren(); Q.labBanner(holder, "That did not load. ", e.message, true); });
  };
  CO.view = (root, d) => {
    const p = d.profile;
    Q.labHead(root, `${p.company_name || p.ticker} (${p.ticker}${p.exchange ? ", " + p.exchange : ""})`, [p.country, p.sector, p.industry, p.currency].filter(Boolean).join(" · "), GUIDE);
    if (p.active === 0 || p.active === false) Q.labBanner(root, "Delisted. ", `This company is no longer listed${p.delisted_date ? " (since " + day(p.delisted_date) + ")" : ""}; its history is kept so that studies do not leave out the ones that failed.`);
    const k = h("div", "earnrow", root);
    tile(k, "Annual statements", String(d.summary.annual || 0), { text: `${day(d.summary.first_end)} to ${day(d.summary.last_end)}` });
    tile(k, "Last filed", day(d.summary.last_filed) || "n/a", { text: p.source ? "from " + p.source : "" });
    tile(k, "Prices", d.prices ? d.prices.days.toLocaleString("en-US") + " days" : "none", { text: d.prices ? `${d.prices.first} to ${d.prices.last}` : "load them in the Data manager" });
    tile(k, "Last close", d.prices && !Q.bad(d.prices.last_close) ? num(d.prices.last_close) : "n/a", { text: p.currency || "" });
    if (d.anomaly_reading) tile(k, "Accounting-anomaly screen", d.annual.length && !Q.bad(d.annual[d.annual.length - 1].accounting_anomaly_score) ? num(d.annual[d.annual.length - 1].accounting_anomaly_score, 2) : "n/a", { text: d.anomaly_reading }, "", "A screen to look closer, never evidence of fraud");
    const rows = d.annual;
    const series = (names) => names.filter((n) => d.series[n]).map((n, i) => ({ name: human(n), color: Q.colorOf(i), values: d.series[n] }));
    const ch = h("div", "charts", root);
    const money = (v) => compact(v);
    const lineCard = (title, note, names, fmtY) => {
      const s = series(names); if (!s.length || d.dates.length < 2) return;
      chartCard(ch, title, note, { draw: (b) => C.line(b, { dates: d.dates, series: s, yFormat: fmtY, tipFormat: fmtY, height: 240, label: title, legend: true }),
        table: (b) => table(b, [{ label: "Period end", get: (i) => d.dates[i] }].concat(s.map((x) => ({ label: x.name, num: true, get: (i) => (x.values[i] === null ? "" : fmtY(x.values[i])) }))), d.dates.map((_, i) => i)) });
    };
    lineCard("Revenue and profit", `In the company's reporting currency (${rows.length ? rows[rows.length - 1].currency || "" : ""}), as filed.`, ["revenue", "gross_profit", "operating_income", "net_income"], money);
    lineCard("Cash", "Cash from operations, capital spending and free cash flow.", ["operating_cash_flow", "capital_expenditure", "free_cash_flow"], money);
    lineCard("Margins", "Profit as a share of revenue.", ["gross_margin", "operating_margin", "net_margin", "fcf_margin"], (v) => pct(v, 0));
    lineCard("Returns", "Profit as a share of equity, assets and invested capital.", ["roe", "roa", "roic"], (v) => pct(v, 0));
    lineCard("Balance sheet", "Assets, debt and equity.", ["total_assets", "total_debt", "shareholders_equity", "cash"], money);
    const fig = W.card(root, "The statements, with the day each became public", "A figure enters research on its 'public from' date plus the lag you choose, never on its period end. 'Estimated' means the filing date was not known and a lag was assumed.");
    fig.dataset.role = "fund-statements"; fig.classList.add("roomy");
    const cols = [{ label: "Year", get: (r) => String(r.fiscal_year) }, { label: "Period end", get: (r) => day(r.period_end) }, { label: "Public from", get: (r) => day(r.available) + (r.estimated ? " (estimated)" : "") }];
    for (const name of d.figures) cols.push({ label: human(name), num: true, get: (r) => compact(r[name]) });
    table(fig, cols, rows.slice().reverse());
    const ratios = W.card(root, "Ratios"); table(ratios, [{ label: "Year", get: (r) => String(r.fiscal_year) }].concat(d.ratios.map((name) => ({ label: human(name), num: true, get: (r) => (name.startsWith("piotroski") ? (Q.bad(r[name]) ? "n/a" : String(Math.round(r[name]))) : Math.abs(r[name]) > 100 ? compact(r[name]) : num(r[name], 3)) }))), rows.slice().reverse());
    const dup = W.card(root, "What moved the return on equity (DuPont)", "Return on equity is the product of five steps. The columns on the right split the change from the year before exactly into what each step contributed (the log-mean Divisia method); they need every step positive in both years.");
    dup.dataset.role = "fund-dupont";
    table(dup, [{ label: "Year", get: (r) => String(r.fiscal_year) }, { label: "Return on equity", num: true, get: (r) => pct(r.roe, 1) }, { label: "Change", num: true, get: (r) => spct(r.d_roe, 1) }]
      .concat(d.dupont_steps.map((s) => ({ label: s.label + " effect", num: true, title: "Contribution to the change in return on equity", get: (r) => spct(r["c_" + s.key], 2) }))).concat([{ label: "Method", get: (r) => r.method }]), d.dupont.slice().reverse());
    const pio = W.card(root, "Piotroski F-score", "Nine yes-or-no signs of improving financial health (Piotroski 2000); the score is how many are yes. It is a screen, not a verdict.");
    pio.dataset.role = "fund-piotroski";
    const sig = d.piotroski_signals, yrs = d.piotroski.map((r) => String(r.fiscal_year));
    C.matrix(h("div", "", pio), { rows: sig.map((s) => s.label), cols: yrs, values: sig.map((s) => d.piotroski.map((r) => (Q.bad(r[s.key]) ? null : r[s.key]))), format: (v) => (v >= 0.5 ? "yes" : "no"), min: 0, max: 1, name: "Signal", label: "Piotroski signals by year", labelWidth: 210, cellHeight: 20, tipFormat: (v) => (v >= 0.5 ? "yes" : "no") });
    h("p", "hint", pio, "Score by year: " + d.piotroski.map((r) => `${r.fiscal_year}: ${Q.bad(r.piotroski_f) ? "n/a" : Math.round(r.piotroski_f)}`).join(" · "));
    const an = W.card(root, "Accounting-anomaly screen (Beneish-type)", d.anomaly_reading || "Eight ratios that compare this year with last, combined as in Beneish (1999). A high value means the accounts changed in ways that were common among companies later found to have manipulated earnings. It is fitted on old United States data and is never evidence of fraud.");
    an.dataset.role = "fund-anomaly";
    const parts = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"];
    table(an, [{ label: "Year", get: (r) => String(r.fiscal_year) }].concat(parts.map((k) => ({ label: k.toUpperCase(), num: true, get: (r) => num(r[k], 2) }))), d.anomaly.slice().reverse());
    if (d.flagged.length) {
      const fl = W.card(root, "Figures the quality checks flagged", "A flagged figure is treated as missing in research. Check it against the filing.");
      table(fl, [{ label: "Year", get: (r) => String(r.fiscal_year) }, { label: "Figure", get: (r) => r.metric_code }, { label: "Value", num: true, get: (r) => compact(r.value) }, { label: "Flag", get: (r) => r.quality_flag.replace(/_/g, " ") }], d.flagged);
    }
    const sym = W.card(root, "Where its prices come from", "The symbol used to ask a market-data source for this company's prices. The market's usual suffix is tried first; if it is wrong, set the right one here.");
    sym.dataset.role = "fund-price-symbol";
    const ids = d.identifiers.filter((x) => x.provider === "yahoo" && x.id_type === "price_symbol");
    const form = h("div", "row tight", sym), input = h("input", "", form); input.type = "text"; input.placeholder = ids.length ? ids[0].provider_id : "e.g. 7203.T"; input.setAttribute("aria-label", "price symbol"); input.spellcheck = false;
    const save = h("button", "btn small", form, "Save"); save.type = "button"; save.addEventListener("click", async () => {
      try { const r = await FD.send("/api/fund/price-symbol", { security_id: p.security_id, symbol: input.value }); Q.toast("Saved: " + r.symbol); FD.cache.delete("company:" + FD.db + ":" + p.security_id); FD.draw(); } catch (e) { Q.toast(e.message); }
    });
    if (FD.db === "demo") { input.disabled = true; save.disabled = true; h("p", "hint", sym, "Simulated companies have no real market-data symbol."); }
    const idl = h("details", "how", root); h("summary", "", idl, "Identifiers and sync state"); const ul = h("ul", "", idl);
    for (const x of d.identifiers) h("li", "", ul, `${x.provider} ${x.id_type.replace(/_/g, " ")}: ${x.provider_id}`);
    for (const [k2, v] of Object.entries(d.sync || {})) if (v) h("li", "", ul, `${k2} statements: last checked ${day(v.last_checked)}, status ${v.status || "n/a"}${v.error ? " (" + v.error + ")" : ""}`);
    FD.notes(root, "Read before relying on it", d.notes);
  };

  /* =========================================================== Data manager */
  const DM = FD.views.data = {};
  DM.run = () => FD.act("data", "Reading the status…", async () => { const s = await FD.read("/api/fund/status"); FD.dm.tested = null; return s; });
  DM.auto = () => DM.run();
  DM.empty = (root) => FD.emptyCard(root, "Reading the status…", "");
  DM.refresh = async () => { try { FD.results.data = { data: await FD.read("/api/fund/status"), body: {} }; } catch (e) { FD.errors.data = e.message; } if (isOpen() && FD.mode === "data") FD.draw(); };
  DM.sections = (panel) => {
    const s = FD.dm, after = FD.after, real = FD.db === "real";
    if (real) {
      const a = FD.sec(panel, "1. Add companies"), area = h("textarea", "", h("label", "f", a, "Tickers, one or several (us:AAPL, jp:7203, uk:HSBA)")); area.rows = 3; area.value = s.symbols; area.spellcheck = false; area.dataset.role = "fund-symbols";
      area.addEventListener("input", () => { s.symbols = area.value; after(); });
      Q.labField(a, { key: "market", label: "Market for tickers without one", kind: "choice", options: [["us", "United States"], ["jp", "Japan"], ["uk", "United Kingdom"], ["eu", "Euro area (ESEF)"], ["fr", "France"], ["ca", "Canada"], ["au", "Australia"], ["hk", "Hong Kong"], ["sg", "Singapore"], ["kr", "Korea"], ["tw", "Taiwan"], ["in", "India"], ["cn", "China"], ["br", "Brazil"], ["mx", "Mexico"], ["ch", "Switzerland"], ["il", "Israel"]] }, s, after);
      const add = h("button", "btn", a, "Add the companies (free)"); add.type = "button"; add.dataset.action = "add-companies";
      add.addEventListener("click", () => DM.add());
      h("p", "hint", a, "Adding costs nothing: only the ticker and market are stored. The first fetch fills in the name, sector and identifiers.");
      const b = FD.sec(panel, "2. Fetch statements (spends credits)");
      Q.labField(b, { key: "start_year", label: "First fiscal year", kind: "number", step: 1, min: 1990, max: 2100 }, s, after);
      Q.labField(b, { key: "budget", label: "Most credits this run may spend", kind: "number", step: 1, min: 1, placeholder: "see the plan" }, s, after);
      const row = h("div", "row tight", b);
      const plan = h("button", "btn", row, "Show the plan"); plan.type = "button"; plan.dataset.action = "plan-sync"; plan.addEventListener("click", () => DM.plan());
      const go = h("button", "btn primary", row, "Fetch within the budget"); go.type = "button"; go.dataset.action = "start-sync"; go.addEventListener("click", () => DM.sync());
      h("p", "hint", b, "A new company costs about 11 credits (a lookup and its annual statements). The plan says what is missing, what is already fresh and how many fit in today's allowance. The run stops by itself when the budget or the allowance is reached.");
      const c = FD.sec(panel, "3. Prices and exchange rates");
      Q.labField(c, { key: "price_start", label: "Prices from", kind: "text", placeholder: "2010-01-01" }, s, after);
      Q.labField(c, { key: "price_refresh", label: "Fetch the whole history again", kind: "bool" }, s, after);
      const lp = h("button", "btn", c, "Load prices and rates"); lp.type = "button"; lp.dataset.action = "load-prices"; lp.addEventListener("click", () => DM.prices());
      h("p", "hint", c, "From Yahoo Finance, unadjusted and adjusted. London prices are quoted in pence and divided by 100. Companies that have delisted usually have no history there, which leaves them out of a study: the research context says so.");
    } else {
      const a = FD.sec(panel, "Make the simulated world");
      Q.labField(a, { key: "companies", label: "Companies", kind: "number", step: 1, min: 30, max: 400 }, s, after);
      Q.labField(a, { key: "from", label: "First year", kind: "number", step: 1, min: 1995, max: 2020 }, s, after);
      Q.labField(a, { key: "to", label: "Last year", kind: "number", step: 1, min: 2001, max: 2024 }, s, after);
      Q.labField(a, { key: "seed", label: "Seed", kind: "number", step: 1, min: 0 }, s, after);
      Q.labField(a, { key: "strong", label: "Plant stronger alphas (easier to find in a small world)", kind: "bool" }, s, after);
      const make = h("button", "btn primary", a, "Make the simulated world"); make.type = "button"; make.dataset.action = "make-demo"; make.addEventListener("click", () => DM.simulate(false));
      const again = h("button", "btn", a, "Start again with these settings"); again.type = "button"; again.addEventListener("click", () => DM.simulate(true));
      h("p", "hint", a, "A world of companies with statements, prices and a few alphas hidden in them, made on this computer in about a minute. It is for trying the tab: what is found in it says nothing about real markets.");
    }
  };
  DM.add = () => FD.act("data", "Adding the companies…", async () => {
    const symbols = FD.dm.symbols.split(/[\s,;]+/).filter(Boolean); if (!symbols.length) throw new Error("Type at least one ticker");
    const r = await FD.send("/api/fund/companies/add", { symbols, market: FD.dm.market });
    Q.toast(`${r.added} companies registered`); FD.dm.plan = null; FD.dm.symbols = ""; return FD.read("/api/fund/status");
  });
  DM.planBody = () => { const s = FD.dm, b = { periods: ["annual"] }; if (s.start_year !== "") b.start_year = Number(s.start_year); return b; };
  DM.plan = () => FD.act("data", "Planning…", async () => { FD.dm.plan = await FD.send("/api/fund/sync/plan", DM.planBody()); if (FD.dm.budget === "") FD.dm.budget = String(Math.min(FD.dm.plan.plan.cost, FD.dm.plan.affordability.remaining_today === null ? FD.dm.plan.plan.cost : FD.dm.plan.affordability.remaining_today)); return FD.read("/api/fund/status"); });
  DM.sync = () => FD.act("data", "Fetching statements…", async (run) => {
    const budget = Number(FD.dm.budget); if (!isFinite(budget) || budget < 1) throw new Error("Say how many credits this run may spend (the plan suggests a number)");
    const { result } = await run.job("/api/fund/sync", Object.assign(DM.planBody(), { budget }), "Fetching statements");
    FD.dm.report = result; FD.dm.plan = null; FD.cache = new Map(); return FD.read("/api/fund/status");
  });
  DM.prices = () => FD.act("data", "Loading prices…", async (run) => {
    const { result } = await run.job("/api/fund/prices", { start: FD.dm.price_start || undefined, refresh: !!FD.dm.price_refresh }, "Loading prices");
    FD.dm.prices = result; FD.cache = new Map(); return FD.read("/api/fund/status");
  });
  DM.simulate = (reset) => FD.act("data", "Making the simulated world…", async (run) => {
    const s = FD.dm, body = { companies: Number(s.companies), start_year: Number(s.from), end_year: Number(s.to), seed: Number(s.seed), reset: !!reset };
    if (s.strong) body.planted = { delta_roic: 0.011, fcf_yield: 0.0068, asset_growth: 0.0045 };
    const { result } = await run.job("/api/fund/simulate", body, "Making the simulated world");
    FD.dm.report = result; FD.resetCaches(); return FD.read("/api/fund/status");
  });
  FD.resetCaches = () => { FD.cache = new Map(); FD.facets = {}; for (const m of ["overview", "companies", "library", "monitor"]) { delete FD.results[m]; delete FD.errors[m]; } };
  DM.test = (button) => {
    button.disabled = true;
    FD.act("data", "Testing the connection (one credit)…", async () => { const r = await FD.send("/api/fund/provider/test", {}); FD.dm.tested = r; return FD.read("/api/fund/status"); });
  };
  DM.build = (root, d) => {
    Q.labHead(root, "Data manager", `${d.counts.companies.toLocaleString("en-US")} companies · database ${(d.database_bytes / 1e6).toFixed(1)} MB · last update ${d.last_update ? day(d.last_update) : "never"}`, GUIDE);
    if (!d.readiness.ready) { const c = h("div", "card", root); c.dataset.role = "fund-not-ready"; h("h3", "", c, "Next steps"); const ul = h("ul", "", c); for (const p of d.readiness.problems) h("li", "", ul, p.replace("in the Data Manager", "with the form on the left")); }
    const k = h("div", "earnrow", root);
    tile(k, "Companies", d.counts.companies.toLocaleString("en-US"), { text: `${d.counts.countries} countries` });
    tile(k, "With statements", d.counts.with_financials.toLocaleString("en-US"), { text: `${d.companies_stale} due a fetch (missing or stale)` });
    tile(k, "Prices", d.counts.price_rows.toLocaleString("en-US"), { text: "daily rows" });
    tile(k, "Raw answers kept", d.counts.raw_payloads.toLocaleString("en-US"), { text: "so a rule can be changed without asking again" });
    for (const p of d.providers) { if (FD.db === "demo" ? p.name === "simulated" : p.name === "openfilings") providerCard(root, d, p); }
    if (FD.dm.plan) planCard(root, FD.dm.plan);
    if (FD.dm.report) reportCard(root, FD.dm.report);
    if (FD.dm.prices) pricesCard(root, FD.dm.prices);
    const q = h("div", "card", root); h("h3", "", q, "Quality of what is stored");
    const obs = Object.entries(d.quality.observations || {});
    if (obs.length) table(q, [{ label: "Figures by flag", get: (r) => r[0].replace(/_/g, " ") }, { label: "Count", num: true, get: (r) => r[1].toLocaleString("en-US") }], obs); else h("p", "hint", q, "Nothing is stored yet, so there is nothing to check.");
    FD.explain(root, "What 'credits' are and why the first working key is the only one used", d.policy);
  };
  function providerCard(root, d, p) {
    const card = h("div", "card", root); card.dataset.role = "fund-provider"; card.dataset.provider = p.name;
    const head = h("div", "bar", card); h("h3", "", head, p.name === "openfilings" ? "OpenFilings" : "Simulated world");
    badge(head, p.available ? (p.keyed ? "key set" : "ready") : "not set up", p.available ? "" : "warn");
    if (p.plan) badge(head, "plan: " + p.plan);
    h("p", "note", card, p.note || (p.name === "openfilings" ? "As-filed financial statements for listed companies in about twenty-five markets, from each regulator's own filings. Annual statements; each is stamped with the day it was filed." : ""));
    if (!p.available) {
      if (p.name === "openfilings") setupGuide(card, p);
      else h("p", "hint tone-warn", card, p.reason);
    }
    if (p.keyed || p.name === "openfilings") {
      const unit = p.unit;
      h("h4", "subhead", card, `Allowance (${unit})`);
      meterRow(card, `Today (${p.resets.daily})`, p.used_today, p.daily_limit, unit);
      if (p.monthly_limit) meterRow(card, `This month (${p.resets.monthly})`, p.used_month, p.monthly_limit, unit);
      h("p", "hint", card, `${p.allowance_source}. A new company costs about ${p.cost_per_company} ${unit}. ${p.requests_today} requests today (${p.successful} ok, ${p.failed} failed).`);
      h("h4", "subhead", card, "Key slots");
      const named = new Set((p.keys || []).map((r) => r.variable)), rows = (p.keys || []).concat(p.name === "openfilings" ? KEY_SLOTS.filter((v) => !named.has(v)).map((v) => ({ slot: "", variable: v, role: "", state: "not set" })) : []);
      table(card, [{ label: "Slot", get: (r) => String(r.slot) }, { label: "Variable", get: (r) => r.variable || "—" }, { label: "Role", get: (r) => r.role, title: "The first working key is active; later ones are standbys that take over only if the active key is rejected as invalid" },
        { label: "State", get: (r) => human(r.state) }, { label: "Used today", num: true, get: (r) => (r.used_today === undefined ? "" : `${r.used_today} / ${r.daily_limit}`) }, { label: "Used this month", num: true, get: (r) => (r.used_month === undefined ? "" : r.monthly_limit ? `${r.used_month} / ${r.monthly_limit}` : String(r.used_month)) }, { label: "Resets", get: (r) => r.resets || "" }], rows);
      for (const n of p.key_notes || []) h("p", "hint", card, n);
      if (p.name === "openfilings") h("p", "hint", card, "Several keys do not add up: keys of one account share one allowance, and the provider's usage policy does not allow rotating keys or accounts to get past its quotas. Extra keys are standbys. More credits come from a larger plan or prepaid top-ups.").dataset.role = "fund-policy";
      if (p.available && p.name === "openfilings") {
        const row = h("div", "row tight", card), test = h("button", "btn", row, "Test the connection (one credit)"); test.type = "button"; test.dataset.action = "test-provider"; test.addEventListener("click", () => DM.test(test));
        if (FD.dm.tested) { const t = FD.dm.tested, out = h("span", "hint " + (t.ok ? "tone-good" : "tone-crit"), row, t.ok ? `Works. Plan: ${t.plan || "unknown"}; markets: ${(t.allowed_markets || []).join(", ") || "n/a"}.` : (t.error || t.reason || "It did not work.")); out.dataset.role = "fund-test-result"; }
      }
      if (p.recent && p.recent.length) {
        const det = h("details", "how", card); h("summary", "", det, "The last requests (no key is ever recorded)");
        table(det, [{ label: "When", get: (r) => String(r.at || r.ts || "").replace("T", " ").slice(0, 19) }, { label: "Endpoint", get: (r) => r.endpoint || "" }, { label: "Company", get: (r) => r.symbol || "" }, { label: "Result", get: (r) => (r.ok === 0 || r.success === 0 ? "failed" : "ok") }, { label: "Cost", num: true, get: (r) => String(r.credits ?? r.cost ?? "") }], p.recent);
      }
    }
  }
  function meterRow(parent, label, used, limit, unit) {
    const row = h("div", "", parent);
    W.meter(row, limit ? used / limit : 0, `${label}: ${used.toLocaleString("en-US")} of ${limit.toLocaleString("en-US")} ${unit} used`, 0.7, 0.95);
  }
  /* The key guide: the key never goes through this page. */
  function setupGuide(parent, p) {
    const guide = h("div", "setup", parent); guide.dataset.role = "fund-key-guide";
    h("p", "tone-warn", guide, "OpenFilings is not set up yet. It takes about two minutes, and the key never goes through this page.");
    const steps = h("ol", "steps", guide);
    h("li", "", steps, "Sign in at openfilings.org and copy your API key from the dashboard (it starts with of_).");
    const two = h("li", "", steps, "In Replit open Tools, then Secrets, then New Secret. Name it "); h("code", "", two, "OPENFILINGS_API_KEY_1"); two.appendChild(document.createTextNode(", paste the key as its value and click Add Secret. A second key goes in OPENFILINGS_API_KEY_2, and so on up to 5."));
    h("li", "", steps, "Press Stop and then Run: an app only sees the secrets that existed when it started. Then reload this page.");
    h("li", "", steps, "Come back here and press Test the connection (it spends one credit and tells the app which plan the account is on).");
    const row = h("div", "row tight", guide);
    const name = h("button", "btn", row, "Copy the first secret name"); name.type = "button"; name.addEventListener("click", () => Q.copy(KEY_SLOTS[0], "The secret name"));
    const again = h("button", "btn", row, "I added it and restarted: reload"); again.type = "button"; again.addEventListener("click", () => location.reload());
    h("p", "hint", guide, "The key is read by the server only and is never sent to this page. Do not paste it into a chat or a file; people you add to the Repl can see its secrets. " + (p.reason || ""));
  }
  function planCard(root, pl) {
    const card = h("div", "card", root); card.dataset.role = "fund-plan"; h("h3", "", card, "The plan");
    const a = pl.affordability, p = pl.plan;
    const k = h("div", "earnrow", card);
    tile(k, "To fetch", String(p.to_fetch), { text: `${p.fresh} already fresh` }); tile(k, "Cost", `${p.cost} ${p.unit}`, { text: Object.entries(p.reasons).map(([r, n]) => `${n} ${r}`).join(", ") || "nothing to do" });
    tile(k, "Left today", a.remaining_today === null ? "n/a" : `${a.remaining_today} ${p.unit}`, { text: `${a.fit_today} of ${a.tasks} fit today` }); tile(k, "Days needed", a.days_needed === null ? "n/a" : String(a.days_needed), { text: a.daily_limit ? `at ${a.daily_limit} a day` : "" });
    for (const n of p.notes || []) h("p", "hint", card, n);
    if (pl.tasks.length) table(card, [{ label: "Company", get: (r) => r.symbol }, { label: "Why", get: (r) => r.reason }, { label: "Cost", num: true, get: (r) => String(r.cost) }], pl.tasks.slice(0, 25));
  }
  function reportCard(root, r) {
    const card = h("div", "card", root); card.dataset.role = "fund-report"; h("h3", "", card, "What the last run did");
    if (r.periods === undefined) { h("p", "", card, `Made ${r.companies} simulated companies with ${r.price_rows ? r.price_rows.toLocaleString("en-US") + " daily prices" : "their prices"}.`); return; }
    const k = h("div", "earnrow", card);
    tile(k, "Fetched", String(r.fetched), { text: `${r.periods} periods stored` }); tile(k, "Spent", `${r.cost} ${r.unit}`); tile(k, "Failed", String(r.failed.length), { text: `${r.not_found.length} not found · ${r.pending.length} still being collected` });
    if (r.stopped) h("p", "banner", card, r.stopped);
    if (r.failed.length) table(card, [{ label: "Company", get: (x) => x.symbol }, { label: "Why", get: (x) => x.error }], r.failed.slice(0, 20));
    if (r.unmapped.length) h("p", "hint", card, "Figures the provider sent that this app does not map yet (kept in the raw answers): " + r.unmapped.slice(0, 20).join(", "));
  }
  function pricesCard(root, r) {
    const card = h("div", "card", root); card.dataset.role = "fund-prices"; h("h3", "", card, "What the last price load did");
    const k = h("div", "earnrow", card);
    tile(k, "Companies with prices", `${r.loaded} of ${r.companies}`, { text: `${r.rows.toLocaleString("en-US")} daily rows` }); tile(k, "No prices", String(r.failed.length));
    if (r.fx) tile(k, "Exchange rates", `${(r.fx.loaded || []).length} of ${(r.fx.currencies || []).length}`, { text: (r.fx.loaded || []).join(", ") });
    for (const n of r.notes || []) h("p", "hint", card, n);
    if (r.failed.length) table(card, [{ label: "Company", get: (x) => x.ticker }, { label: "Symbol tried", get: (x) => x.symbol || "" }, { label: "Why", get: (x) => x.reason }], r.failed.slice(0, 30));
    if (r.stopped) h("p", "banner", card, r.stopped);
  }

  Q.registerWorkspace("fundamentals", FD);
  document.getElementById("tab-fundamentals").addEventListener("click", () => setTimeout(() => { if (FD.views.watch) FD.views.watch(); }, 0));
})();
