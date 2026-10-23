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
  const PROVIDERS = [["openfilings", "OpenFilings (about 25 markets, paid in credits)"], ["sec_edgar", "SEC EDGAR (US-listed companies, free, the deepest history)"]];
  const WORD = { openfilings: "OpenFilings", sec_edgar: "SEC EDGAR" };
  const MARKETS = [["us", "United States"], ["jp", "Japan"], ["uk", "United Kingdom"], ["eu", "Euro area (ESEF)"], ["fr", "France"], ["ca", "Canada"], ["au", "Australia"], ["hk", "Hong Kong"], ["sg", "Singapore"],
    ["kr", "Korea"], ["tw", "Taiwan"], ["in", "India"], ["cn", "China"], ["br", "Brazil"], ["mx", "Mexico"], ["ch", "Switzerland"], ["il", "Israel"]];

  const FD = {
    id: "fundamentals", root: "#fundamentals", view: "#view-fundamentals", title: "Fundamentals", api: 11, mode: "overview", busyBy: {}, jobs: {}, epoch: 0, results: {}, errors: {}, cache: new Map(), guard: null, go: null, progress: null,
    about: "Company fundamentals from around the world in a local database, tested as factors and searched for alphas, with the evidence counted honestly.",
    modes: [["overview", "Overview"], ["companies", "Company explorer"], ["factors", "Factor lab"], ["discovery", "Alpha discovery"], ["library", "Alpha library"], ["monitor", "Alpha monitor"], ["data", "Data manager"]],
    db: Q.store("fund.db") === "demo" ? "demo" : "real", facets: {}, views: {},
    co: { q: "", country: "", sector: "", market: "", provider: "openfilings", offset: 0, limit: 25, id: null, found: null, report: null },
    dm: { symbols: "", market: "us", provider: "openfilings", start_year: "2000", budget: "", plan: null, report: null, prices: null, price_start: "2010-01-01", price_refresh: false, price_note: "", companies: "120", from: "2009", to: "2022", seed: "7",
          strong: false, tested: null },
    uni: { countries: new Set(), sectors: new Set(), min_cap: "", min_vol: "", developed: false, delisted: true, freq: "M", lag: "1", start: "", end: "", info_lag: "1" },
    cov: null,                                                                               // the years of the company on show: what is held, which are picked, the plan for them
  };
  const freshCov = (id, provider) => ({ id: id === undefined ? null : id, provider: provider || null, data: null, picked: new Set(), plan: null, report: null, seq: 0, timer: null, last: null });
  FD.cov = freshCov();
  /* The lab frame asks ``busy`` of the view on screen. Each view has its own: a long search in Alpha discovery does not stop the Company explorer, the library or the Data manager from being read. */
  Object.defineProperty(FD, "busy", { get: () => FD.busyBy[FD.mode] || "", enumerable: true });
  window.FD = FD;
  FD.runLabel = { overview: "Refresh", companies: "Search my database", factors: "Study this factor", discovery: "Start the search", get library() { return FD.views.libraryLabel ? FD.views.libraryLabel() : "Refresh the list"; }, monitor: "Run the monitor now", data: "Refresh the status" };
  FD.blurb = {
    overview: "What the database holds, what is missing, and what has been found.",
    companies: "Find a company in your database or in a provider's directory (OpenFilings, or the SEC's list of US-listed companies), its statements as they were filed, and the screens built on them.",
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
        if (mode === FD.mode) { Q.syncLab(FD); FD.lockActions(); }
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
    if (mode === FD.mode) { Q.syncLab(FD); FD.lockActions(); if (isOpen()) FD.draw(); }
    let got = null, failed = null;
    try { got = { data: await fn(run), body: {} }; }
    catch (e) { failed = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + Q.RESTART : e.message; }
    if (epoch !== FD.epoch) return;
    delete FD.busyBy[mode];
    if (failed) FD.errors[mode] = failed; else FD.results[mode] = got;
    paintBackground();
    if (mode === FD.mode) { Q.syncLab(FD); FD.lockActions(); if (isOpen()) FD.draw(); }
  };
  /* A button in the panel that starts work is disabled while its view is working: pressed, it would only be refused ("Still working"), and a press that does nothing looks like a broken button.
     A browser takes the focus away from a button that becomes disabled, so the one that had it gets it back when the work is over (a keyboard user is not sent to the top of the page). */
  FD.lockActions = () => {
    const busy = !!FD.busyBy[FD.mode], panel = Q.$("#fundamentals .panel"), at = document.activeElement;
    if (busy && panel && at && at.dataset && at.dataset.action && panel.contains(at)) FD.focusAfterBusy = at.dataset.action;
    for (const b of document.querySelectorAll("#fundamentals .panel button[data-action]")) {
      if (busy && !b.disabled) { b.disabled = true; b.dataset.busyLock = "1"; }
      else if (!busy && b.dataset.busyLock) { b.disabled = false; delete b.dataset.busyLock; }
    }
    if (!busy && FD.focusAfterBusy) {
      const back = panel && panel.querySelector(`button[data-action="${FD.focusAfterBusy}"]`);
      if (back && (!document.activeElement || document.activeElement === document.body)) back.focus();
      FD.focusAfterBusy = null;
    }
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
  FD.loadFacets = async (again) => {
    if (FD.facets[FD.db] && !again) return FD.facets[FD.db];
    const db = FD.db, epoch = FD.epoch;
    try { const r = await FD.read("/api/fund/companies", { limit: 1 }); if (epoch === FD.epoch) FD.facets[db] = { countries: r.countries, sectors: r.sectors, total: r.total, lookup: r.lookup || null, lookups: r.lookups || null }; }
    catch (e) { if (epoch === FD.epoch && !FD.facets[db]) FD.facets[db] = { countries: [], sectors: [], total: 0, error: e.message }; }
    return FD.facets[db] || { countries: [], sectors: [], total: 0 };
  };
  FD.resetData = () => { FD.epoch++; FD.busyBy = {}; FD.jobs = {}; FD.results = {}; FD.errors = {}; FD.cache = new Map(); FD.facets = {}; FD.co.id = null; FD.co.found = null; FD.co.report = null; FD.dm.plan = null; FD.dm.report = null; clearTimeout(FD.cov.timer); FD.cov = freshCov(); if (FD.views.reset) FD.views.reset(); };

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
    return FD.db === "demo" ? "Make the simulated world in the Data manager first." : "Add companies and fetch their statements in the Data manager first (or find one by name in the Company explorer).";
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
    FD.lockActions();
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
  /* A table built by hand in a scrolling wrapper: when it is wider or taller than its card it must be reachable from the keyboard, and its region needs a name (what ``simpleTable`` does for its own). */
  FD.guardScroll = (wrap) => {
    const guard = () => {
      if (!wrap.isConnected) return;
      if (wrap.scrollWidth > wrap.clientWidth + 1 || wrap.scrollHeight > wrap.clientHeight + 1) {
        const head = wrap.closest(".card") && wrap.closest(".card").querySelector("h3");
        wrap.tabIndex = 0; wrap.setAttribute("role", "region"); wrap.setAttribute("aria-label", `${head ? head.textContent : "Table"}: table, scrolls`);
      } else { wrap.removeAttribute("tabindex"); wrap.removeAttribute("role"); wrap.removeAttribute("aria-label"); }
    };
    if (window.ResizeObserver) { const watch = new ResizeObserver(guard); watch.observe(wrap); const t = wrap.querySelector("table"); if (t) watch.observe(t); } else requestAnimationFrame(guard);
    return wrap;
  };
  FD.explain = (root, title, text) => { const d = h("details", "how", root); h("summary", "", d, title); h("p", "", d, text); return d; };
  FD.emptyCard = (root, title, paragraphs, button) => {
    const c = h("div", "card empty", root); h("h2", "", c, title); for (const t of [].concat(paragraphs)) h("p", "", c, t);
    if (button) { const b = h("button", "btn primary", c, button.label); b.type = "button"; b.addEventListener("click", button.fn); }
    return c;
  };
  FD.goto = (mode) => { FD.mode = mode; Q.renderLab(FD); };
  FD.fmt = { spct, sgn, compact, human, day };

  /* What to do next, in order: each step says whether it is done, the first one that is not is marked, and every step has the button that does it. ``o`` is what the database and the account hold. */
  FD.setupSteps = (o) => {
    const n = (x) => Number(x || 0).toLocaleString("en-US"), c = o.counts, steps = [];
    if (!o.real) {
      steps.push({ done: c.companies > 0, title: "Make the simulated world", detail: c.companies > 0 ? `${n(c.companies)} simulated companies.` : "A world of companies with statements, prices and a few alphas hidden in them, made on this computer in about a minute.", action: { key: "make-demo", label: "Make it", mode: "data" } });
    } else {
      const src = o.sources || {}, onlySec = !!src.sec_edgar && !src.openfilings, unit = o.unit || "credits";
      steps.push({ done: !!o.keyOk, title: onlySec ? "Connect a data source" : "Set up your OpenFilings key", detail: onlySec ? "SEC EDGAR is ready: free, for US-listed companies. A key from OpenFilings adds other markets." : o.keyOk ? "A key is set." : "Statements come from OpenFilings (a key, paid in credits, about 25 markets) or, for US-listed companies, free from SEC EDGAR (a contact address). Neither goes through this page: both go in Replit's Secrets.", action: { key: "set-up-key", label: "Show me how", mode: "data" } });
      steps.push({ done: c.companies > 0, title: "Add companies", detail: c.companies > 0 ? `${n(c.companies)} in the database.` : "Find a company by name or type tickers such as us:AAPL (free). A search in OpenFilings costs 1 credit; in SEC EDGAR it is free.", action: { key: "find-company", label: "Find a company", mode: "companies" } });
      steps.push({ done: c.with_financials > 0, title: "Fetch their statements", detail: c.with_financials > 0 ? `${n(c.with_financials)} of ${n(c.companies)} have annual statements.` : `About ${o.fetchCost || 11} ${unit} a company${o.remaining === null || o.remaining === undefined ? "" : `, ${n(o.remaining)} ${unit} left today`}.`, action: { key: "fetch-statements", label: "Fetch statements", mode: "data" } });
      steps.push({ done: c.price_rows > 0, title: "Load their prices", detail: c.price_rows > 0 ? `${n(c.price_rows)} daily prices.` : "Free, from Yahoo Finance. A study measures what a factor earned, so it needs prices.", action: { key: "load-prices", label: "Load prices", mode: "data" } });
      if (o.studySize && c.companies > 0 && c.with_financials > 0 && c.price_rows > 0) {
        const z = o.studySize;
        steps.push({ done: !!(z.available && z.adequate), title: "Enough companies to see a real signal", detail: z.available ? `${n(Math.round(z.companies))} companies on ${n(z.dates)} dates can see a mean IC of ${z.mde_ic.toFixed(3)} or more; useful signals are 0.02 to 0.05.` : z.reading, action: { key: "more-companies", label: "Add more companies", mode: "companies" } });
      }
    }
    if (o.runs !== null && o.runs !== undefined && c.companies > 0 && (c.with_financials > 0 || !o.real)) steps.push({ done: o.runs > 0, title: "Study a factor, then search for alphas", detail: o.runs > 0 ? `${o.runs} search${o.runs === 1 ? "" : "es"} run.` : "Start with the Factor lab: does one idea predict what happened next? Then let Alpha discovery try many.", action: { key: "open-factor-lab", label: "Open the Factor lab", mode: "factors" } });
    return steps;
  };
  FD.checklist = (root, o) => {
    const steps = FD.setupSteps(o); if (!steps.length || steps.every((x) => x.done)) return null;
    const ready = o.ready, card = h("div", "card checklist", root); card.dataset.role = ready ? "fund-checklist" : "fund-not-ready";
    h("h2", "", card, ready ? "What to do next" : "Not ready to study yet");
    const list = h("ol", "", card); let next = true, i = 0;
    for (const step of steps) {
      i += 1;
      const li = h("li", step.done ? "done" : next ? "next" : "todo", list); li.dataset.step = step.action.key;
      h("span", "mark", li, step.done ? "✓" : String(i)).setAttribute("aria-hidden", "true");
      const body = h("div", "body", li); h("b", "", body, step.title); h("span", "vh", body, step.done ? " (done)" : next ? " (next)" : " (to do)"); h("span", "meta", body, step.detail);
      if (!step.done) { const b = h("button", "btn small" + (next ? " primary" : ""), li, step.action.label); b.type = "button"; b.dataset.action = step.action.key; b.addEventListener("click", () => FD.goto(step.action.mode)); }
      if (!step.done) next = false;
    }
    return card;
  };

  /* =========================================================== Overview */
  const OV = FD.views.overview = {};
  OV.run = () => FD.act("overview", "Reading the database…", () => FD.read("/api/fund/overview"));
  OV.auto = () => OV.run();
  OV.empty = (root) => FD.emptyCard(root, "Reading the database…", "");
  OV.sections = () => {};                                                                    // the page says what it shows: nothing to set here, only the data to choose above
  OV.build = (root, d) => {
    Q.labHead(root, "Overview", `${d.counts.companies.toLocaleString("en-US")} companies · ${d.counts.with_financials.toLocaleString("en-US")} with statements · ${d.counts.price_rows.toLocaleString("en-US")} daily prices`, GUIDE);
    const setup = d.setup || {};
    const lookups = d.lookups || {}, sources = { openfilings: !!(lookups.openfilings && lookups.openfilings.available), sec_edgar: !!(lookups.sec_edgar && lookups.sec_edgar.available) };
    FD.checklist(root, { real: FD.db === "real", ready: d.readiness.ready, counts: d.counts, keyOk: !!setup.available, sources, unit: setup.unit, fetchCost: setup.fetch_cost, remaining: setup.remaining_today, studySize: d.study_size, runs: d.runs.length });
    const k = h("div", "earnrow", root);
    tile(k, "Companies", d.counts.companies.toLocaleString("en-US"), { text: `${d.counts.countries} countries · ${d.counts.markets} markets` });
    tile(k, "With statements", d.counts.with_financials.toLocaleString("en-US"), { text: `${d.counts.annual_periods.toLocaleString("en-US")} annual · ${d.counts.quarterly_periods.toLocaleString("en-US")} quarterly periods` });
    tile(k, "Daily prices", d.counts.price_rows.toLocaleString("en-US"), { text: `${d.counts.observations.toLocaleString("en-US")} figures stored` });
    tile(k, "Restated filings", d.counts.restated_vintages.toLocaleString("en-US"), { text: "kept beside the original, never overwriting it" });
    tile(k, "Alphas in the library", String(Object.values(d.library.library || {}).reduce((a, b) => a + b, 0)), { text: `${d.library.graveyard} failed ideas kept in the graveyard` });
    tile(k, "Ideas tried in all", String(d.library.tried || 0), { text: "every one counts against a result" });
    const sz = d.study_size;
    if (sz && d.counts.companies > 0) {
      const card = W.card(root, "What this much data can tell you", sz.reading); card.dataset.role = "fund-study-size";
      if (sz.available) table(card, [{ label: "A true mean IC of", num: true, get: (r) => num(r.ic, 2) }, { label: "would show a t of about", num: true, get: (r) => num(r.expected_t, 1) }, { label: "chance of |t| above 2", num: true, get: (r) => pct(r.power_t2, 0) }, { label: "chance of |t| above 3", num: true, get: (r) => pct(r.power_t3, 0) }], sz.table);
    }
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
  const ALT_WORD = { symbol: "a ticker", name: "a company name" };
  CO.local = () => ({ q: FD.co.q, country: FD.co.country, sector: FD.co.sector, limit: FD.co.limit, offset: FD.co.offset });
  CO.run = () => { FD.co.offset = 0; FD.co.id = null; return CO.load(); };
  CO.load = () => FD.act("companies", "Searching…", () => FD.read("/api/fund/companies", CO.local()));
  CO.auto = () => { if (!FD.co.id) CO.load(); };
  CO.empty = (root) => { if (FD.co.id) CO.detail(root, FD.co.id); else FD.emptyCard(root, "Company explorer", "Search the companies in your database by name, ticker or ISIN."); };
  /* The Country and Sector lists are made from the companies the database holds, so they have nothing to offer until some are added; they are made again after companies are added. */
  CO.refreshFacets = () => FD.loadFacets(true).then(() => { if (isOpen() && FD.mode === "companies") Q.renderLab(FD); });
  CO.sections = (panel) => {
    const b = FD.sec(panel, "Find a company"), s = FD.co, f = FD.facets[FD.db];
    Q.labField(b, { key: "q", label: "Name, ticker or ISIN", kind: "text", placeholder: "apple, 7203, US0378331005" }, s, FD.after);
    const box = b.querySelector("input[data-key=q]");
    if (box) box.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); CO.run(); } });
    if (!f) FD.loadFacets().then(() => { if (isOpen() && FD.mode === "companies") Q.renderLab(FD); });
    else {
      if (f.countries.length > 1) Q.labField(b, { key: "country", label: "Country", kind: "choice", options: [["", "All"]].concat(f.countries.map((c) => [c, c])) }, s, FD.after);
      if (f.sectors.length > 1) Q.labField(b, { key: "sector", label: "Sector", kind: "choice", options: [["", "All"]].concat(f.sectors.map((c) => [c, c])) }, s, FD.after);
      if (f.total === 0) h("p", "hint", b, "The Country and Sector lists fill in as companies are added to the database.");
    }
    h("p", "hint", b, "This searches the companies already in your database. Open one to see every figure with the date it became public, the ratios built from them, the DuPont split of the change in return on equity, the Piotroski F-score and the accounting-anomaly screen.");
    if (FD.db === "real") CO.lookupSection(panel, f);
  };
  /* The state of the provider the search is aimed at: the lookups the server sent for each, else the one it sent at the top (the first provider that can be used). */
  CO.info = (d) => (d && ((d.lookups || {})[FD.co.provider] || d.lookup)) || null;
  CO.lookupSection = (panel, f) => {
    const b = FD.sec(panel, "Not in your database yet?"), info = CO.info(f), sec = FD.co.provider === "sec_edgar", word = WORD[FD.co.provider];
    Q.labField(b, { key: "provider", label: "Look it up in", kind: "choice", options: PROVIDERS }, FD.co, () => { FD.after(); Q.renderLab(FD); });
    if (sec) h("p", "hint", b, "Search the SEC's list of US-listed companies by name or ticker (us:NVDA), or name one by its CIK number (cik:1045810). A search is free. Adding the company costs nothing; fetching its statements takes two free requests to the SEC and brings back its whole history, about fifteen years of annual statements for most US companies.");
    else h("p", "hint", b, "Ask OpenFilings for a company by name, ticker (us:NVDA) or ISIN. A search uses 1 credit of today's allowance and the same search again is free. Adding the company costs nothing; fetching its statements costs about 11 credits.");
    if (!sec) Q.labField(b, { key: "market", label: "Only this market", kind: "choice", options: [["", "Any market"]].concat(MARKETS) }, FD.co, FD.after);
    const go = h("button", "btn", b, sec ? "Search SEC EDGAR (free)" : "Search OpenFilings (1 credit)"); go.type = "button"; go.dataset.action = "search-provider";
    go.addEventListener("click", () => CO.remote("auto"));
    if (info && !info.available) { go.disabled = true; h("p", "hint tone-warn", b, sec ? `SEC EDGAR is not set up yet, so it cannot be searched: ${info.reason}` : `${word} is not set up yet, so it cannot be searched: the Data manager shows how to add a key.`); }
    else if (!sec && info && info.remaining_today !== null && info.remaining_today !== undefined) {
      if (info.daily_limit) meterRow(b, "Today", info.daily_limit - info.remaining_today, info.daily_limit, "credits");
      if (info.monthly_limit && info.remaining_month !== null && info.remaining_month !== undefined) meterRow(b, "This month", info.monthly_limit - info.remaining_month, info.monthly_limit, "credits");
      h("p", "hint", b, `${info.remaining_today.toLocaleString("en-US")} credits left today.`);
    }
  };
  /* Ask a provider's directory (OpenFilings: one credit; SEC EDGAR: free). Nothing is added: that is the next, free step. */
  CO.remote = (kind) => {
    const text = FD.co.q.trim(), provider = FD.co.provider;
    if (text.length < 2) { Q.toast("Type at least two characters of a name, a ticker or an ISIN first"); return; }
    FD.act("companies", `Searching ${WORD[provider]}…`, async () => {
      FD.co.found = await FD.send("/api/fund/companies/search", { q: text, kind: kind || "auto", provider, market: provider === "openfilings" ? FD.co.market || undefined : undefined }); FD.co.report = null;
      return FD.read("/api/fund/companies", CO.local());
    });
  };
  /* Add the company (free) and, if asked, fetch its statements within the cost stated on the button. The saved answer of the search is read again, for nothing, so each row says what the database now holds. */
  CO.add = (row, fetch) => FD.act("companies", fetch ? "Adding the company and fetching its statements…" : "Adding the company…", async (run) => {
    const found = FD.co.found, provider = found.provider || "openfilings", again = () => FD.send("/api/fund/companies/search", { q: found.query, kind: found.kind, provider, market: found.market || undefined }); FD.co.report = null;
    if (fetch) {
      const { result } = await run.job("/api/fund/sync", { symbols: [row.symbol], provider, budget: found.cost_to_fetch || 11 }, "Fetching statements");
      FD.co.report = Object.assign({ symbol: row.symbol }, result);
      if (result.fetched) {                                                                  // a study needs prices as well: they are free, so they come with the statements
        const added = ((await again()).results || []).find((x) => x.symbol === row.symbol);
        if (added && added.security_id !== undefined && added.security_id !== null) {
          try { FD.co.report.prices = (await run.job("/api/fund/prices", { security_ids: [added.security_id] }, "Loading its prices")).result; } catch (e) { FD.co.report.prices_error = e.message; }
        }
      }
    } else {
      await FD.send("/api/fund/companies/add", { symbols: [row.symbol], provider }); Q.toast(`${row.name || row.symbol} added`);
    }
    FD.co.found = await again();
    FD.resetCaches(); CO.refreshFacets();                                                    // what the other views showed (the overview, its list of what to do next) is out of date now
    return FD.read("/api/fund/companies", CO.local());
  });
  CO.reportLine = (r) => {
    const word = WORD[r.provider] || "OpenFilings";
    if (r.not_found && r.not_found.length) return [r.provider === "sec_edgar" ? `SEC EDGAR has no XBRL statements for ${r.symbol}: it may not file with the SEC, or be too small or too new to have filed with XBRL.` : `OpenFilings has no annual statements for ${r.symbol} (a request that fails is refunded).`, "tone-warn"];
    if (r.pending && r.pending.length) return [`${word} is still collecting the filings of ${r.symbol} (some filings are gathered on demand). Ask again in a few minutes.`, "tone-warn"];
    if (r.failed && r.failed.length) return [`${r.symbol}: ${r.failed[0].error}`, "tone-crit"];
    if (r.fetched) {
      const px = r.prices, spent = `Fetched ${r.periods} annual periods for ${r.symbol} and spent ${amount(r.cost, r.unit || "credits")}.` + (r.stopped ? " " + r.stopped : "");
      if (px && px.loaded) return [`${spent} Loaded ${px.rows.toLocaleString("en-US")} daily prices (free), so it can be studied now.`, "tone-good"];
      if (px) return [`${spent} No prices were found for it at Yahoo Finance${px.failed && px.failed[0] ? ` (${px.failed[0].reason})` : ""}: open the company and set its price symbol, or it will be left out of studies.`, "tone-warn"];
      return [spent + (r.prices_error ? ` Its prices could not be loaded (${r.prices_error}): load them in the Data manager.` : " Load its prices in the Data manager to study it."), r.prices_error ? "tone-warn" : "tone-good"];
    }
    return [r.stopped || `Nothing was fetched for ${r.symbol}: it is already up to date.`, "hint"];
  };
  CO.none = (card, dbTotal) => {
    if (FD.db === "demo") { h("p", "hint", card, dbTotal === 0 ? "The demo has not been made yet: open the Data manager." : "No company in the demo matches."); return; }
    if (dbTotal === 0) {
      h("h3", "", card, "There are no companies in your database yet");
      h("p", "", card, "This list shows only the companies you have added; nothing is added by itself. Find a company by name with the search below (OpenFilings, 1 credit; SEC EDGAR, free), or type tickers such as us:AAPL in the Data manager (free).");
    } else h("p", "", card, `None of the ${dbTotal.toLocaleString("en-US")} companies in your database matches ${FD.co.q ? "“" + FD.co.q + "”" : "these filters"}. Ask a provider below, or clear the filters.`);
  };
  /* What the provider said, or the offer to ask. */
  CO.remoteCard = (root, d) => {
    const f = FD.facets[FD.db] || {}, info = CO.info(d.lookup ? d : f), q = FD.co.q.trim(), found = FD.co.found && (FD.co.found.provider || "openfilings") === FD.co.provider ? FD.co.found : null;
    const sec = FD.co.provider === "sec_edgar", word = WORD[FD.co.provider];
    const card = h("div", "card roomy", root); card.dataset.role = "fund-remote";
    if (!found || found.query.toLowerCase() !== q.toLowerCase()) {
      h("h3", "", card, "Not in your database?");
      if (info && !info.available) {
        h("p", "tone-warn", card, sec ? `SEC EDGAR is not set up yet, so it cannot be searched. ${info.reason}` : "OpenFilings is not set up yet, so it cannot be searched. The Data manager shows how to add a key.");
        const go = h("button", "btn", card, "Open the Data manager"); go.type = "button"; go.addEventListener("click", () => FD.goto("data"));
        return;
      }
      const left = info && info.remaining_today !== null && info.remaining_today !== undefined ? ` (${info.remaining_today.toLocaleString("en-US")} left today)` : "";
      h("p", "", card, sec ? (q.length < 2 ? "Type a name, a ticker (us:NVDA) or a CIK (cik:1045810) on the left, then search the SEC's list. A search is free; adding the company is free."
          : `Search the SEC's list of US-listed companies for “${q}”. It is free, and nothing is added until you choose.`)
        : q.length < 2 ? "Type a name, a ticker (us:NVDA) or an ISIN on the left, then ask OpenFilings. A search uses 1 credit; adding the company is free."
        : `Ask OpenFilings for “${q}”. A search uses 1 credit of today's allowance${left}, and the same search again is free.`);
      const label = sec ? (q.length < 2 ? "Search SEC EDGAR (free)" : `Search SEC EDGAR for “${q}” (free)`) : q.length < 2 ? "Search OpenFilings (1 credit)" : `Search OpenFilings for “${q}” (1 credit)`;
      const go = h("button", "btn primary", card, label); go.type = "button"; go.dataset.action = "search-provider";
      go.addEventListener("click", () => CO.remote("auto"));
      return;
    }
    const head = h("div", "bar", card); h("h3", "", head, `${word}: “${found.query}”`);
    badge(head, `read as ${found.kind_word}${found.market && !sec ? " on the " + found.market.toUpperCase() + " market" : ""}`);
    badge(head, found.cached ? "saved answer, free" : found.credits ? `${found.credits} credit${found.credits === 1 ? "" : "s"} spent` : "nothing spent");
    if (found.remaining_today !== null && found.remaining_today !== undefined && !sec) h("p", "hint", card, `${found.remaining_today.toLocaleString("en-US")} credits left today. Adding a company is free; fetching its statements costs about ${found.cost_to_fetch} credits.`);
    if (sec) h("p", "hint", card, `Adding a company is free; fetching its statements takes about ${found.cost_to_fetch} free requests to the SEC and brings back its whole history.`);
    if (FD.co.report) { const [text, cls] = CO.reportLine(FD.co.report); h("p", "note " + cls, card, text).dataset.role = "fund-fetch-report"; }
    if (found.message) h("p", "hint", card, found.message);
    if (found.markets_skipped && found.markets_skipped.length) h("p", "hint tone-warn", card, "OpenFilings did not search these markets this time: " + found.markets_skipped.map((m) => m.toUpperCase()).join(", ") + ". A company listed only there will not appear.").dataset.role = "fund-skipped";
    if (!found.results.length) {
      h("p", "", card, `${word} has no company that matches “${found.query}” as ${found.kind_word}.`).dataset.role = "fund-remote-none";
      for (const alt of found.alternatives) {
        const b = h("button", "btn", card, `Try it as ${ALT_WORD[alt] || alt}${sec ? "" : " (1 credit)"}`); b.type = "button"; b.dataset.action = "search-as-" + alt; b.addEventListener("click", () => CO.remote(alt));
      }
      return;
    }
    if (found.ambiguous) h("p", "hint", card, "More than one company matches. The best match is first.");
    const wrap = FD.guardScroll(h("div", "scroll", card)), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t));
    for (const [label, hidden] of [["Company", 0], ["Symbol", 0], ["Country", 0], ["ISIN", 0], ["In your database", 0], ["Actions", 1]]) { const th = h("th", "", hd); if (hidden) h("span", "vh", th, label); else th.textContent = label; }
    const tb = h("tbody", "", t);
    for (const r of found.results) {
      const tr = h("tr", "", tb); tr.dataset.symbol = r.symbol;
      const name = h("td", "", tr, r.name || "(no name)"); if (r.also_listed.length) h("div", "hint", name, "also listed as " + r.also_listed.join(", "));
      h("td", "", tr, r.symbol); h("td", "", tr, r.country || ""); h("td", "", tr, r.isin || "");
      h("td", "", tr, r.in_database ? (r.periods ? `Yes, ${r.periods} annual periods` : "Yes, no statements yet") : "No");
      const act = h("td", "", tr), button = (label, aria, action, fn) => { const b = h("button", "btn small", act, label); b.type = "button"; b.dataset.action = action; b.setAttribute("aria-label", aria); b.addEventListener("click", fn); };
      if (r.in_database) {
        button("Open", "open " + r.symbol, "open-company", () => { FD.co.id = r.security_id; FD.draw(); window.scrollTo({ top: 0 }); });
        if (!r.periods && r.has_financials && r.plan_allows !== false) button(`Fetch statements (about ${amount(found.cost_to_fetch, found.unit)})`, "fetch the statements of " + r.symbol, "fetch-statements", () => CO.add(r, true));
      } else {
        const caution = !r.has_financials ? `${word} lists no statements for this market` : r.plan_allows === false ? "Not among the markets of your plan" : "";
        button("Add (free)", "add " + r.symbol, "add-found", () => CO.add(r, false));
        if (caution) h("div", "hint tone-warn", act, caution + ", so fetching is not offered.");           // adding costs nothing; a fetch that is bound to find nothing would still spend the lookup
        else button(`Add and fetch (about ${amount(found.cost_to_fetch, found.unit)})`, "add and fetch the statements of " + r.symbol, "add-fetch-found", () => CO.add(r, true));
      }
    }
  };
  CO.build = (root, d) => {
    if (FD.co.id) { CO.detail(root, FD.co.id); return; }
    const f = FD.facets[FD.db]; if (f && d.lookup) { f.lookup = d.lookup; f.lookups = d.lookups || null; }
    const dbTotal = Math.max(f ? f.total : 0, d.total);                                      // what the database holds is never fewer than what matched
    Q.labHead(root, "Companies", dbTotal === 0 ? "none in the database yet" : `${d.total.toLocaleString("en-US")} match${d.total === 1 ? "" : "es"}${FD.co.q ? ` “${FD.co.q}”` : ""}`, GUIDE);
    const card = h("div", "card roomy", root); card.dataset.role = "fund-companies";
    if (!d.companies.length) CO.none(card, dbTotal);
    else {
      const wrap = FD.guardScroll(h("div", "scroll", card)), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t));
      for (const [label, numeric] of [["Ticker", 0], ["Company", 0], ["Country", 0], ["Sector", 0], ["Annual", 1], ["Gaps", 1], ["First period", 0], ["Last period", 0], ["Price days", 1], ["Open", 2]]) { const th = h("th", numeric === 1 ? "num" : "", hd); if (numeric === 2) h("span", "vh", th, label); else th.textContent = label; }
      const tb = h("tbody", "", t);
      for (const r of d.companies) {
        const tr = h("tr", "", tb); tr.dataset.id = r.security_id;
        h("td", "", tr, r.ticker + (r.exchange ? " · " + r.exchange : "")); h("td", "", tr, r.company_name || ""); h("td", "", tr, r.country || ""); h("td", "", tr, r.sector || "");
        h("td", "num", tr, String(r.annual)); h("td", "num", tr, r.annual_gaps ? String(r.annual_gaps) : "").title = "Fiscal years missing between the first and the last that are held"; h("td", "", tr, day(r.first_period_end)); h("td", "", tr, day(r.last_period_end)); h("td", "num", tr, String(r.price_days));
        const open = h("button", "btn small", h("td", "", tr), "Open"); open.type = "button"; open.dataset.action = "open-company"; open.setAttribute("aria-label", "open " + r.ticker);
        open.addEventListener("click", () => { FD.co.id = r.security_id; FD.draw(); window.scrollTo({ top: 0 }); });
      }
      const pager = h("div", "row tight", card), from = d.offset + 1, to = d.offset + d.companies.length;
      h("span", "hint", pager, `${from.toLocaleString("en-US")} to ${to.toLocaleString("en-US")} of ${d.total.toLocaleString("en-US")}`);
      const prev = h("button", "btn small", pager, "Previous"); prev.type = "button"; prev.disabled = d.offset === 0; prev.addEventListener("click", () => { FD.co.offset = Math.max(0, d.offset - d.limit); CO.load(); });
      const next = h("button", "btn small", pager, "Next"); next.type = "button"; next.disabled = to >= d.total; next.addEventListener("click", () => { FD.co.offset = d.offset + d.limit; CO.load(); });
    }
    if (FD.db === "real") CO.remoteCard(root, d);
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
  /* ---------- the years a company holds ---------- */
  const YEAR_MARK = { held: "✓", missing: "+", empty: "∅", due: "!", notyet: "…" };
  const PICKED_MARK = "●";
  const YEAR_WORD = { held: "held", missing: "missing", empty: "asked for, nothing sent", due: "due: a filing should exist", notyet: "not due yet" };
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many || one + "s"}`;
  const amount = (n, unit) => `${n} ${n === 1 && unit.endsWith("s") ? unit.slice(0, -1) : unit}`;
  /* The card is made at once and filled when the years arrive. It is made again only when the company or the data changes, so that picking a year keeps the focus where it is. */
  CO.coverage = (root, p) => {
    const card = h("div", "card roomy", root); card.dataset.role = "fund-coverage";
    h("h3", "", card, "Which fiscal years do you hold?");
    const body = h("div", "stack", card);
    if (FD.cov.id !== p.security_id) { clearTimeout(FD.cov.timer); FD.cov = freshCov(p.security_id); }
    if (FD.cov.data) { CO.coverageBody(body); return card; }
    h("p", "hint", body, "Reading which years are held…");
    const id = p.security_id, db = FD.db;
    FD.read("/api/fund/coverage", { id, provider: FD.cov.provider || undefined }).then((d) => { if (FD.db !== db || FD.cov.id !== id || !body.isConnected) return; FD.cov.data = d; body.replaceChildren(); CO.coverageBody(body); })
      .catch((e) => { if (!body.isConnected) return; body.replaceChildren(); Q.labBanner(body, "That did not load. ", e.message, true); });
    return card;
  };
  CO.coverageBody = (body) => {
    const c = FD.cov.data, a = c.annual, cells = new Map(a.cells.map((x) => [x.year, x])), live = !!c.can_fetch, k = a.counts;
    if (FD.cov.report) h("p", "note " + (FD.cov.report.added.length ? "tone-good" : "tone-warn"), body, FD.cov.report.text).dataset.role = "fund-years-report";
    const held = a.earliest_held === null ? "none yet" : `${plural(k.held, "year")}, FY${a.earliest_held} to FY${a.latest_held}`;
    if ((c.providers || []).length > 1) {                                                   // a company that more than one provider knows: which one's years are shown
      const pick = h("div", "row tight", body); pick.setAttribute("role", "group"); pick.setAttribute("aria-label", "Whose statements"); pick.dataset.role = "fund-years-provider";
      for (const x of c.providers) {
        const b = h("button", "btn small" + (x.name === c.provider ? " primary" : ""), pick, `${x.word} (${plural(x.periods, "year")})`); b.type = "button"; b.dataset.provider = x.name; b.setAttribute("aria-pressed", x.name === c.provider ? "true" : "false");
        b.addEventListener("click", () => { if (x.name === c.provider) return; clearTimeout(FD.cov.timer); FD.cov = freshCov(c.security_id, x.name); FD.draw(); });
      }
    }
    h("p", "", body, `Annual statements held: ${held}. Missing: ${k.missing}. Asked for, nothing sent: ${k.empty}. Due now: ${k.due}. Quarterly and half-year statements held: ${c.quarterly.held}.`).dataset.role = "fund-years-summary";
    if (c.reading) { h("h4", "subhead", body, "What the last request brought back"); h("p", "note", body, c.reading).dataset.role = "fund-years-reading"; }
    const legend = h("div", "legend years-legend", body);
    for (const state of ["held", "missing", "empty", "due", "notyet"]) { const item = h("span", "", legend); h("b", "mk", item, YEAR_MARK[state]).setAttribute("aria-hidden", "true"); item.appendChild(document.createTextNode(` ${YEAR_WORD[state]} (${k[state]})`)); }
    if (live) { const item = h("span", "", legend); h("b", "mk", item, PICKED_MARK).setAttribute("aria-hidden", "true"); item.appendChild(document.createTextNode(" picked")); }
    const grid = h("div", "years", body); grid.setAttribute("role", "group"); grid.setAttribute("aria-label", `Fiscal years of ${c.name || c.ticker}`);
    const toggle = (year, range) => {
      const picked = FD.cov.picked, was = picked.has(year);
      if (range && FD.cov.last !== null) {                                                  // shift-click: every pickable year between this one and the last one clicked
        const [lo, hi] = [Math.min(year, FD.cov.last), Math.max(year, FD.cov.last)];
        for (let y = lo; y <= hi; y++) if (pickable(cells.get(y))) { if (was) picked.delete(y); else picked.add(y); }
      } else if (was) picked.delete(year); else picked.add(year);
      FD.cov.last = year; syncPicks();
    };
    const pickable = (cell) => !!cell && live && cell.state !== "held" && cell.state !== "notyet";
    const mark = (cell) => cell.state === "held" ? `held, filed ${cell.filed || "(date unknown)"}` : cell.state === "empty" ? `asked for on ${cell.asked_on}, nothing sent` : cell.state === "due" ? `due: filing expected from ${cell.expected_filing}` : cell.state === "notyet" ? `not due until ${cell.expected_filing}` : "missing";
    for (let decade = Math.floor(a.first / 10) * 10; decade <= a.last; decade += 10) {
      const group = h("div", "decade", grid); h("h4", "subhead", group, `${decade}s`);
      const row = h("div", "row", group);
      for (let year = decade; year < decade + 10; year++) {
        const cell = cells.get(year);
        if (!cell) { h("span", "gap", row).setAttribute("aria-hidden", "true"); continue; }
        const b = h("button", `y ${cell.state}`, row); b.type = "button"; b.dataset.year = String(year); b.dataset.state = cell.state;
        h("span", "n", b, String(year)); h("span", "mk", b, YEAR_MARK[cell.state]).setAttribute("aria-hidden", "true");
        b.setAttribute("aria-label", `${year}, ${mark(cell)}`); b.title = mark(cell);
        if (pickable(cell)) { b.setAttribute("aria-pressed", "false"); b.addEventListener("click", (e) => toggle(year, e.shiftKey)); } else b.disabled = true;
      }
    }
    grid.addEventListener("keydown", (e) => {                                               // the arrow keys move between the years that can be picked
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key]; if (!step || !e.target.matches || !e.target.matches("button.y")) return;
      const all = [...grid.querySelectorAll("button.y:not(:disabled)")], at = all.indexOf(e.target), next = all[at + step];
      if (next) { e.preventDefault(); next.focus(); }
    });
    const tools = h("div", "row tight", body);
    const tool = (label, action, years, off) => { const b = h("button", "btn small", tools, label); b.type = "button"; b.dataset.action = action; b.disabled = !live || off; b.addEventListener("click", years); return b; };
    tool(`Pick every missing year (${k.missing})`, "cov-pick-missing", () => { for (const x of a.cells) if (x.state === "missing") FD.cov.picked.add(x.year); syncPicks(); }, k.missing === 0);
    tool(`Pick what is due (${k.due})`, "cov-pick-due", () => { for (const x of a.cells) if (x.state === "due") FD.cov.picked.add(x.year); syncPicks(); }, k.due === 0);
    const older = (x) => x.state === "empty" && a.earliest_held !== null && x.year < a.earliest_held;
    tool(`Try the older years found empty (${a.cells.filter(older).length})`, "cov-pick-empty", () => { for (const x of a.cells) if (older(x)) FD.cov.picked.add(x.year); syncPicks(); }, !a.cells.some(older));
    tool("Clear", "cov-clear", () => { FD.cov.picked.clear(); FD.cov.last = null; syncPicks(); }, false);
    const planBox = h("div", "stack", body); planBox.dataset.role = "fund-years-plan"; planBox.setAttribute("role", "status"); planBox.setAttribute("aria-live", "polite");
    if (c.provider === "sec_edgar") h("p", "hint", body, `SEC EDGAR sends a company's whole history in one free request (${amount(c.request_cost, c.unit)}), whichever years you pick; picking only chooses which of them are stored, and a year you already hold is never stored again.`);
    else h("p", "hint", body, `Credits pay for a request, not for years: one request costs ${amount(c.request_cost, c.unit)} whether it brings back three years or thirty, and a year you already hold is never asked for again. So pick all the years you want at once; they are asked for in one request.`);
    const quarters = c.quarterly, qd = h("details", "how", body); h("summary", "", qd, `Quarterly and half-year statements (${quarters.held} held)`);
    const labels = ["Q1", "Q2", "Q3", "Q4", "H1", "H2"].filter((l) => quarters.rows.some((r) => r.periods[l]));
    if (quarters.rows.length) table(qd, [{ label: "Year", get: (r) => String(r.year) }].concat(labels.map((l) => ({ label: l, get: (r) => (r.periods[l] ? "✓ " + r.periods[l] : "") }))), quarters.rows.slice().reverse());
    if (quarters.note) h("p", "hint", qd, quarters.note);
    if (!quarters.rows.length && !quarters.note) h("p", "hint", qd, "None are held for this company.");
    if (c.log.length) {
      const ld = h("details", "how", body); h("summary", "", ld, `Requests made for this company (${c.log.length})`);
      table(ld, [{ label: "Day", get: (r) => day(r.at) }, { label: "Asked for", get: (r) => r.asked }, { label: "Entries sent", num: true, get: (r) => String(r.entries) }, { label: "Annual read", num: true, get: (r) => String(r.kept) },
        { label: "New", num: true, get: (r) => (r.new === null || r.new === undefined ? "" : String(r.new)) }, { label: "Cost", num: true, get: (r) => (r.cost === null || r.cost === undefined ? "not recorded" : `${r.cost} ${c.unit}`) },
        { label: "Why", get: (r) => ({ missing: "first fetch", history: "older years", stale: "a newer filing was due", chosen: "years you chose" })[r.reason] || r.reason || "" }], c.log);
    }
    const paintPlan = () => {
      planBox.replaceChildren();
      if (!live) { h("p", "hint tone-warn", planBox, c.why_not || "Years cannot be asked for now."); return; }
      const years = [...FD.cov.picked].sort((x, y) => x - y), plan = FD.cov.plan;
      if (!years.length) { h("p", "hint", planBox, "Click the years you want (Shift-click to pick a run of them), then ask for them. The cost is shown before anything is spent."); return; }
      if (!plan) { h("p", "hint", planBox, `Working out what ${plural(years.length, "year")} would cost…`); return; }
      if (plan.error) { h("p", "hint tone-crit", planBox, plan.error); return; }
      h("p", "", planBox, plan.text).dataset.role = "fund-years-plan-text";
      if (plan.to_fetch.length && !plan.fits_today) h("p", "hint tone-warn", planBox, `That does not fit in what is left of today's allowance (${plan.remaining_today} ${plan.unit}). Pick fewer years, or wait for it to reset.`);
      const go = h("button", "btn primary", planBox, plan.to_fetch.length ? `Ask for ${plural(plan.to_fetch.length, "year")} (${amount(plan.cost, plan.unit)})` : "Nothing to ask for"); go.type = "button"; go.dataset.action = "cov-fetch";
      go.disabled = !plan.to_fetch.length || !plan.fits_today; go.addEventListener("click", () => fetchYears(plan, years));
    };
    function showPicks() {
      for (const b of grid.querySelectorAll("button.y[aria-pressed]")) {
        const on = FD.cov.picked.has(Number(b.dataset.year));
        b.setAttribute("aria-pressed", on ? "true" : "false"); b.querySelector(".mk").textContent = on ? PICKED_MARK : YEAR_MARK[b.dataset.state];
      }
    }
    const asksAgain = (years) => years.some((y) => (cells.get(y) || {}).state === "empty");
    function syncPicks() {
      showPicks();
      const years = [...FD.cov.picked].sort((x, y) => x - y), seq = ++FD.cov.seq; FD.cov.plan = null; paintPlan();
      clearTimeout(FD.cov.timer);
      if (!years.length) return;
      FD.cov.timer = setTimeout(() => FD.send("/api/fund/coverage/plan", { security_id: c.security_id, years, provider: c.provider, ask_again: asksAgain(years) })
        .then((r) => { if (seq === FD.cov.seq && planBox.isConnected) { FD.cov.plan = r; paintPlan(); } })
        .catch((e) => { if (seq === FD.cov.seq && planBox.isConnected) { FD.cov.plan = { error: e.message }; paintPlan(); } }), 120);
    }
    const fetchYears = (plan, years) => FD.act("companies", `Asking ${c.provider_word} for the chosen years…`, async (run) => {
      const { result } = await run.job("/api/fund/coverage/fetch", { security_id: c.security_id, years, provider: c.provider, budget: plan.cost, ask_again: asksAgain(years) }, "Asking for the chosen years");
      FD.resetCaches(); CO.refreshFacets();                                                                          // the figures, the years, the list and the overview are out of date now
      FD.cov = Object.assign(freshCov(c.security_id, c.provider), { report: result });
      return FD.read("/api/fund/companies", CO.local());
    });
    showPicks();
    paintPlan();
    if (FD.cov.picked.size && !FD.cov.plan) syncPicks();
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
    CO.coverage(root, p);
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
  DM.run = () => FD.act("data", "Reading the status…", async () => { const s = await FD.read("/api/fund/status"); FD.dm.tested = null; FD.dm.chose = null; return s; });
  DM.auto = () => DM.run();
  DM.empty = (root) => FD.emptyCard(root, "Reading the status…", "");
  DM.refresh = async () => { try { FD.results.data = { data: await FD.read("/api/fund/status"), body: {} }; } catch (e) { FD.errors.data = e.message; } if (isOpen() && FD.mode === "data") FD.draw(); };
  DM.sections = (panel) => {
    const s = FD.dm, after = FD.after, real = FD.db === "real";
    if (real) {
      const sec = s.provider === "sec_edgar", a = FD.sec(panel, "1. Add companies");
      Q.labField(a, { key: "provider", label: "Statements from", kind: "choice", options: PROVIDERS }, s, () => { s.plan = null; after(); Q.renderLab(FD); });
      const area = h("textarea", "", h("label", "f", a, sec ? "Tickers, one or several (AAPL, MSFT, BRK-B), or CIK numbers (cik:320193)" : "Tickers, one or several (us:AAPL, jp:7203, uk:HSBA)")); area.rows = 3; area.value = s.symbols; area.spellcheck = false; area.dataset.role = "fund-symbols";
      area.addEventListener("input", () => { s.symbols = area.value; after(); });
      if (!sec) Q.labField(a, { key: "market", label: "Market for tickers without one", kind: "choice", options: MARKETS }, s, after);
      const add = h("button", "btn", a, "Add the companies (free)"); add.type = "button"; add.dataset.action = "add-companies";
      add.addEventListener("click", () => DM.add());
      h("p", "hint", a, sec ? "Adding costs nothing: only the ticker is stored. The first fetch fills in the name, sector and industry. Do not know the ticker? Find the company by name in the Company explorer (SEC EDGAR, free)."
        : "Adding costs nothing: only the ticker and market are stored. The first fetch fills in the name, sector and identifiers. Do not know the ticker? Find the company by name in the Company explorer (OpenFilings, 1 credit).");
      const b = FD.sec(panel, sec ? "2. Fetch statements (free)" : "2. Fetch statements (spends credits)");
      Q.labField(b, { key: "start_year", label: "First fiscal year", kind: "number", step: 1, min: 1990, max: 2100 }, s, after);
      Q.labField(b, { key: "budget", label: sec ? "Most requests this run may make" : "Most credits this run may spend", kind: "number", step: 1, min: 1, placeholder: "see the plan" }, s, after);
      const row = h("div", "row tight", b);
      const plan = h("button", "btn", row, "Show the plan"); plan.type = "button"; plan.dataset.action = "plan-sync"; plan.addEventListener("click", () => DM.plan());
      const go = h("button", "btn primary", row, "Fetch within the budget"); go.type = "button"; go.dataset.action = "start-sync"; go.addEventListener("click", () => DM.sync());
      if (sec) h("p", "hint", b, "A new company takes two requests to the SEC (its record, then its facts), both free, and brings back its whole history: about fifteen years of annual statements for most US companies, starting from its first XBRL filing (2009 for the largest). The program pauses between requests to stay inside the SEC's limit, and stops at the first refusal. The plan says what is missing and what is already fresh.");
      else h("p", "hint", b, "A new company costs about 11 credits (a lookup and its annual statements), however many years are asked for, so the default is a long history. The years that come back are the ones OpenFilings holds for that company, which can be fewer than asked: open the company to see which years you hold and to ask for the ones that are missing. The plan says what is missing, what is already fresh and how many fit in today's allowance. The run stops by itself when the budget or the allowance is reached.");
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
    const r = await FD.send("/api/fund/companies/add", { symbols, provider: FD.dm.provider, market: FD.dm.provider === "openfilings" ? FD.dm.market : undefined });
    Q.toast(`${r.added} companies registered`); FD.dm.plan = null; FD.dm.symbols = ""; FD.resetCaches(); return FD.read("/api/fund/status");
  });
  DM.planBody = () => { const s = FD.dm, b = { periods: ["annual"], provider: s.provider }; if (s.start_year !== "") b.start_year = Number(s.start_year); return b; };
  DM.plan = () => FD.act("data", "Planning…", async () => { FD.dm.plan = await FD.send("/api/fund/sync/plan", DM.planBody()); if (FD.dm.budget === "") FD.dm.budget = String(Math.min(FD.dm.plan.plan.cost, FD.dm.plan.affordability.remaining_today === null ? FD.dm.plan.plan.cost : FD.dm.plan.affordability.remaining_today)); return FD.read("/api/fund/status"); });
  DM.sync = () => FD.act("data", "Fetching statements…", async (run) => {
    const budget = Number(FD.dm.budget); if (!isFinite(budget) || budget < 1) throw new Error(`Say how many ${FD.dm.provider === "sec_edgar" ? "requests" : "credits"} this run may ${FD.dm.provider === "sec_edgar" ? "make" : "spend"} (the plan suggests a number)`);
    const { result } = await run.job("/api/fund/sync", Object.assign(DM.planBody(), { budget }), "Fetching statements");
    FD.dm.report = result; FD.dm.plan = null; FD.resetCaches(); return FD.read("/api/fund/status");
  });
  DM.prices = () => FD.act("data", "Loading prices…", async (run) => {
    const { result } = await run.job("/api/fund/prices", { start: FD.dm.price_start || undefined, refresh: !!FD.dm.price_refresh }, "Loading prices");
    FD.dm.prices = result; FD.resetCaches(); return FD.read("/api/fund/status");
  });
  DM.simulate = (reset) => FD.act("data", "Making the simulated world…", async (run) => {
    const s = FD.dm, body = { companies: Number(s.companies), start_year: Number(s.from), end_year: Number(s.to), seed: Number(s.seed), reset: !!reset };
    if (s.strong) body.planted = { delta_roic: 0.011, fcf_yield: 0.0068, asset_growth: 0.0045 };
    const { result } = await run.job("/api/fund/simulate", body, "Making the simulated world");
    FD.dm.report = result; FD.resetCaches(); return FD.read("/api/fund/status");
  });
  FD.resetCaches = () => { FD.cache = new Map(); FD.facets = {}; clearTimeout(FD.cov.timer); FD.cov = freshCov(); for (const m of ["overview", "companies", "library", "monitor"]) { delete FD.results[m]; delete FD.errors[m]; } };
  DM.test = (button, provider) => {
    button.disabled = true;
    FD.act("data", provider === "sec_edgar" ? "Testing the connection (one free request)…" : "Testing the connection (one credit)…", async () => { const r = await FD.send("/api/fund/provider/test", { provider: provider || "openfilings" }); FD.dm.tested = Object.assign({ provider: provider || "openfilings" }, r); return FD.read("/api/fund/status"); });
  };
  DM.useKey = (row) => FD.act("data", `Switching to ${row.slot}…`, async () => {
    const r = await FD.send("/api/fund/keys/use", { slot: row.slot });
    FD.dm.tested = null; FD.dm.chose = r.active;                                              // what was tested was the other key
    return FD.read("/api/fund/status");
  });
  DM.build = (root, d) => {
    Q.labHead(root, "Data manager", `${d.counts.companies.toLocaleString("en-US")} companies · database ${(d.database_bytes / 1e6).toFixed(1)} MB · last update ${d.last_update ? day(d.last_update) : "never"}`, GUIDE);
    const prov = (d.providers || []).find((x) => x.name === "openfilings"), sec = (d.providers || []).find((x) => x.name === "sec_edgar"), main = prov && prov.available ? prov : sec && sec.available ? sec : prov;
    FD.checklist(root, { real: FD.db === "real", ready: d.readiness.ready, counts: d.counts, keyOk: !!((prov && prov.available) || (sec && sec.available)), sources: { openfilings: !!(prov && prov.available), sec_edgar: !!(sec && sec.available) }, unit: main && main.unit,
      fetchCost: main && main.cost_per_company, remaining: main ? (main.daily_limit === null || main.daily_limit === undefined ? null : Math.max(main.daily_limit - main.used_today, 0)) : null, studySize: null, runs: null });
    const k = h("div", "earnrow", root);
    tile(k, "Companies", d.counts.companies.toLocaleString("en-US"), { text: `${d.counts.countries} countries` });
    tile(k, "With statements", d.counts.with_financials.toLocaleString("en-US"), { text: `${d.companies_stale} due a fetch (missing or stale)` });
    tile(k, "Prices", d.counts.price_rows.toLocaleString("en-US"), { text: "daily rows" });
    tile(k, "Raw answers kept", d.counts.raw_payloads.toLocaleString("en-US"), { text: "so a rule can be changed without asking again" });
    for (const p of d.providers) { if (FD.db === "demo" ? p.name === "simulated" : p.name === "openfilings" || p.name === "sec_edgar") providerCard(root, d, p); }
    if (FD.dm.plan) planCard(root, FD.dm.plan);
    if (FD.dm.report) reportCard(root, FD.dm.report);
    if (FD.dm.prices) pricesCard(root, FD.dm.prices);
    const q = h("div", "card", root); h("h3", "", q, "Quality of what is stored");
    const obs = Object.entries(d.quality.observations || {});
    if (obs.length) table(q, [{ label: "Figures by flag", get: (r) => r[0].replace(/_/g, " ") }, { label: "Count", num: true, get: (r) => r[1].toLocaleString("en-US") }], obs); else h("p", "hint", q, "Nothing is stored yet, so there is nothing to check.");
    FD.explain(root, "What 'credits' are and which key is used", d.policy);
  };
  function providerCard(root, d, p) {
    const card = h("div", "card", root); card.dataset.role = "fund-provider"; card.dataset.provider = p.name;
    const head = h("div", "bar", card); h("h3", "", head, p.name === "openfilings" ? "OpenFilings" : p.name === "sec_edgar" ? "SEC EDGAR" : "Simulated world");
    badge(head, p.available ? (p.keyed ? "key set" : "ready") : "not set up", p.available ? "" : "warn");
    if (p.plan) badge(head, "plan: " + p.plan);
    if (p.name === "sec_edgar") badge(head, "free");
    h("p", "note", card, p.note || (p.name === "openfilings" ? "As-filed financial statements for listed companies in about twenty-five markets, from each regulator's own filings. Annual statements; each is stamped with the day it was filed."
      : p.name === "sec_edgar" ? "The statements US-listed companies file with the Securities and Exchange Commission, as filed and stamped with the day each was filed. Free, no key, and the whole history in one request: about fifteen years of annual statements for most US companies." : ""));
    if (!p.available) {
      if (p.name === "openfilings") setupGuide(card, p);
      else if (p.name === "sec_edgar") secGuide(card, p);
      else h("p", "hint tone-warn", card, p.reason);
    }
    if (p.name === "sec_edgar") {
      h("p", "hint", card, `${p.requests_today} requests today (${p.successful} ok, ${p.failed} failed). A new company takes ${p.cost_per_company} requests. ${p.contact ? (p.contact_has_address ? "Your contact is set." : "Your contact is set, but has no email address in it, which the SEC asks for.") : ""}`);
      if (p.available) {
        const row = h("div", "row tight", card), test = h("button", "btn", row, "Test the connection (one free request)"); test.type = "button"; test.dataset.action = "test-provider"; test.dataset.provider = "sec_edgar"; test.addEventListener("click", () => DM.test(test, "sec_edgar"));
        if (FD.dm.tested && FD.dm.tested.provider === "sec_edgar") { const t = FD.dm.tested, out = h("span", "hint " + (t.ok ? "tone-good" : "tone-crit"), row, t.ok ? "Works: the SEC accepted the contact and answered." : (t.error || t.reason || "It did not work.")); out.dataset.role = "fund-test-result"; }
      }
      if (p.recent && p.recent.length) {
        const det = h("details", "how", card); h("summary", "", det, "The last requests (the contact is never recorded)");
        table(det, [{ label: "When", get: (r) => String(r.at || r.ts || r.timestamp || "").replace("T", " ").slice(0, 19) }, { label: "Endpoint", get: (r) => r.endpoint || "" }, { label: "Company", get: (r) => r.symbol || "" }, { label: "Result", get: (r) => (r.ok === 0 || r.success === 0 ? "failed" : "ok") }], p.recent);
      }
      return;
    }
    if (p.keyed || p.name === "openfilings") {
      const unit = p.unit;
      h("h4", "subhead", card, `Allowance (${unit})`);
      meterRow(card, `Today (${p.resets.daily})`, p.used_today, p.daily_limit, unit);
      if (p.monthly_limit) meterRow(card, `This month (${p.resets.monthly})`, p.used_month, p.monthly_limit, unit);
      h("p", "hint", card, `${p.allowance_source}. A new company costs about ${p.cost_per_company} ${unit}. ${p.requests_today} requests today (${p.successful} ok, ${p.failed} failed).`);
      h("h4", "subhead", card, "Key slots");
      const named = new Set((p.keys || []).map((r) => r.variable)), rows = (p.keys || []).concat(p.name === "openfilings" ? KEY_SLOTS.filter((v) => !named.has(v)).map((v) => ({ slot: "", variable: v, role: "", state: "not set" })) : []);
      keyTable(card, p, rows);
      for (const n of p.key_notes || []) h("p", "hint", card, n);
      if (p.name === "openfilings") h("p", "hint", card, "Several keys do not add up: keys of one account share one allowance, and the provider's usage policy does not allow rotating keys or accounts to get past its quotas. The first working key is used unless you choose another with Use this key; nothing switches by itself, and a choice is refused while the key in use has spent its allowance. A key rejected as invalid is replaced by the next one. More credits come from a larger plan or prepaid top-ups.").dataset.role = "fund-policy";
      if (p.available && p.name === "openfilings") {
        const row = h("div", "row tight", card), test = h("button", "btn", row, "Test the connection (one credit)"); test.type = "button"; test.dataset.action = "test-provider"; test.addEventListener("click", () => DM.test(test, "openfilings"));
        if (FD.dm.tested && (FD.dm.tested.provider || "openfilings") === "openfilings") { const t = FD.dm.tested, out = h("span", "hint " + (t.ok ? "tone-good" : "tone-crit"), row, t.ok ? `Works. Plan: ${t.plan || "unknown"}; markets: ${(t.allowed_markets || []).join(", ") || "n/a"}.` : (t.error || t.reason || "It did not work.")); out.dataset.role = "fund-test-result"; }
      }
      if (p.recent && p.recent.length) {
        const det = h("details", "how", card); h("summary", "", det, "The last requests (no key is ever recorded)");
        table(det, [{ label: "When", get: (r) => String(r.at || r.ts || "").replace("T", " ").slice(0, 19) }, { label: "Endpoint", get: (r) => r.endpoint || "" }, { label: "Company", get: (r) => r.symbol || "" }, { label: "Result", get: (r) => (r.ok === 0 || r.success === 0 ? "failed" : "ok") }, { label: "Cost", num: true, get: (r) => String(r.credits ?? r.cost ?? "") }], p.recent);
      }
    }
  }
  /* The key slots. Where there are several keys, a key that is not in use has a button to make it the one in use: a choice, never a rotation, and refused (with its reason shown) while the key in use has spent its allowance. */
  function keyTable(card, p, rows) {
    const several = p.name === "openfilings" && (p.keys || []).length > 1, wrap = FD.guardScroll(h("div", "scroll", card)), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t));
    const cols = [["Slot", 0], ["Variable", 0], ["Role", 0], ["State", 0], ["Used today", 1], ["Used this month", 1], ["Resets", 0]].concat(several ? [["Choose", 2]] : []);
    for (const [label, numeric] of cols) {
      const th = h("th", numeric === 1 ? "num" : "", hd); if (numeric === 2) h("span", "vh", th, label); else th.textContent = label;
      if (label === "Role") th.title = "The key in use is the one you chose, or else the first working key. Standbys take over by themselves only if the key in use is rejected as invalid";
    }
    const tb = h("tbody", "", t), blocked = new Set();
    for (const r of rows) {
      const tr = h("tr", "", tb);
      h("td", "", tr, String(r.slot)); h("td", "", tr, r.variable || "—"); h("td", "", tr, r.role); h("td", "", tr, human(r.state));
      h("td", "num", tr, r.used_today === undefined ? "" : `${r.used_today} / ${r.daily_limit}`);
      h("td", "num", tr, r.used_month === undefined ? "" : r.monthly_limit ? `${r.used_month} / ${r.monthly_limit}` : String(r.used_month));
      h("td", "", tr, r.resets || "");
      if (!several) continue;
      const cell = h("td", "", tr);
      if (r.slot && r.role === "standby") {
        const b = h("button", "btn small", cell, "Use this key"); b.type = "button"; b.dataset.action = "use-key"; b.dataset.slot = r.slot; b.setAttribute("aria-label", "use " + r.slot);
        if (r.can_use) b.addEventListener("click", () => DM.useKey(r)); else { b.disabled = true; if (r.why_not) blocked.add(r.why_not); }
      }
    }
    for (const why of blocked) h("p", "hint tone-warn", card, why).dataset.role = "fund-key-refused";
    const mine = rows.find((r) => r.chosen);
    const cap = (text) => text.charAt(0).toUpperCase() + text.slice(1);
    if (mine) h("p", "hint", card, `${cap(mine.slot)} is in use because you chose it; the order of the slots no longer decides.`).dataset.role = "fund-key-chosen";
    if (FD.dm.chose) h("p", "note tone-good", card, `${cap(FD.dm.chose)} is the key in use now. Press Test the connection to check it (one credit).`).dataset.role = "fund-key-choice";
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
    h("li", "", steps, "Sign in at openfilings.org and create an API key in the dashboard (their examples show keys that begin with of_).");
    const two = h("li", "", steps, "In Replit open Tools, then Secrets, then New Secret. Name it "); h("code", "", two, "OPENFILINGS_API_KEY_1"); two.appendChild(document.createTextNode(", paste the key as its value and click Add Secret. A second key goes in OPENFILINGS_API_KEY_2, and so on up to 5."));
    h("li", "", steps, "Press Stop and then Run: an app only sees the secrets that existed when it started. Then reload this page.");
    h("li", "", steps, "Come back here and press Test the connection (it spends one credit and tells the app which plan the account is on).");
    const row = h("div", "row tight", guide);
    const name = h("button", "btn", row, "Copy the first secret name"); name.type = "button"; name.addEventListener("click", () => Q.copy(KEY_SLOTS[0], "The secret name"));
    const again = h("button", "btn", row, "I added it and restarted: reload"); again.type = "button"; again.addEventListener("click", () => location.reload());
    h("p", "hint", guide, "The key is read by the server only and is never sent to this page. Do not paste it into a chat or a file; people you add to the Repl can see its secrets. " + (p.reason || ""));
  }
  /* The SEC asks every program to say who runs it; that contact goes in a secret, never through this page. */
  function secGuide(parent, p) {
    const guide = h("div", "setup", parent); guide.dataset.role = "sec-contact-guide";
    h("p", "tone-warn", guide, "SEC EDGAR is not set up yet. It needs no key and takes a minute: the SEC asks every program that reads its data to say who runs it, and that goes in a secret, not through this page.");
    const steps = h("ol", "steps", guide);
    const one = h("li", "", steps, "In Replit open Tools, then Secrets, then New Secret. Name it "); h("code", "", one, "SEC_EDGAR_USER_AGENT"); one.appendChild(document.createTextNode(", give it your name and an email address as its value (for example Jane Doe jane@example.com) and click Add Secret."));
    h("li", "", steps, "Press Stop and then Run: an app only sees the secrets that existed when it started. Then reload this page.");
    h("li", "", steps, "Come back here and press Test the connection (one free request).");
    const row = h("div", "row tight", guide);
    const name = h("button", "btn", row, "Copy the secret name"); name.type = "button"; name.addEventListener("click", () => Q.copy("SEC_EDGAR_USER_AGENT", "The secret name"));
    const again = h("button", "btn", row, "I added it and restarted: reload"); again.type = "button"; again.addEventListener("click", () => location.reload());
    h("p", "hint", guide, "The SEC sees this contact with every request, so use an address you are happy to give it. This application never sends it anywhere else and never shows it. " + (p.reason || ""));
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
