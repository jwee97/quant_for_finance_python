/* Shared parts of the Options, Futures, Crypto and Machine learning workspaces.
   A workspace is a lab (see app.js): a form on the left, the results on the right, one view per mode. This file holds what the four have in common: how a request is made and a long one is polled, how the
   provenance of a payload is shown (data, your file or a model), a stable colour for each entity, and a few small controls. Every string from the server goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, h = Q.h, api = Q.api;

  const SOURCE_TEXT = { platform: "Platform data", user_file: "Your file", downloaded: "Downloaded: free data", simulated: "Simulated: a model", mixed: "Mixed sources" };
  const SOURCE_HELP = {
    platform: "Real data held in this repository (prices, funding, interest rates).",
    user_file: "Read from a file you put in data/user/.",
    downloaded: "Real data this app downloaded for you from a free public source (named in the notes), saved under data/user/downloads/. It is delayed or unaudited: good for research, not a feed to trade on.",
    simulated: "Made by a model on this computer. It is not a market: use it to check that the analytics and the machinery behave, not to look for an edge.",
    mixed: "Some parts are real data and some are simulated or yours; the parts are listed beside this badge." };
  const sourceText = (s) => SOURCE_TEXT[s] || String(s).replace(/_/g, " ");

  /* ---------- requests ---------- */
  const qs = (obj) => {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(obj || {})) if (v !== undefined && v !== null && v !== "") p.set(k, Array.isArray(v) ? v.join(",") : String(v));
    const s = p.toString(); return s ? "?" + s : "";
  };
  function poll(lab, job, message) {                                                        // a long computation runs as a job on the server; ask after it until it ends (or is lost with the app)
    return Q.waitJob(job, (v) => { lab.busy = `${message} ${v.message} (${Math.round(v.elapsed)} s)`; Q.syncLab(lab); }, 500);
  }
  /* Run the requests of a plan one after the other and file each answer under the modes that draw from it. A step is { modes, path, query | body, job, cacheKey }: a GET with a query string, or a POST
     (a job when ``job`` is set); an answer kept under ``cacheKey`` in ``lab.cache`` is not asked for twice. Whatever goes wrong is shown on the mode that was open when the request was made. */
  async function request(lab, build, message) {
    if (lab.busy) return;
    let steps; try { steps = build(); } catch (e) { Q.toast(e.message); return; }
    const mode = lab.mode; lab.busy = message; Q.syncLab(lab);
    try {
      for (const st of steps) {
        let data = st.cacheKey && lab.cache ? lab.cache.get(st.cacheKey) : undefined;
        if (data === undefined) {
          if (st.job) data = await poll(lab, (await api(st.path, st.body)).job, message);
          else data = st.body ? await api(st.path, st.body) : await api(st.path + qs(st.query));
          if (st.cacheKey && lab.cache) lab.cache.set(st.cacheKey, data);
        }
        for (const m of st.modes) { lab.results[m] = { data, body: st.body || st.query || {} }; lab.errors[m] = null; }
      }
    } catch (e) {
      lab.results[mode] = null;
      lab.errors[mode] = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + Q.RESTART : e.message;
    }
    lab.busy = ""; Q.syncLab(lab);
    if (!Q.$(lab.view).hidden) {                                                            // the reader may have moved to another mode while this ran: draw what is open now, and load it if it is waiting
      lab.draw();
      if (lab.auto && !lab.results[lab.mode] && !lab.errors[lab.mode]) lab.auto();
    }
  }
  let catalogPromise = null;
  const catalog = () => (catalogPromise = catalogPromise || api("/api/assets/catalog").catch((e) => { catalogPromise = null; throw e; }));

  /* ---------- provenance: where the numbers come from ---------- */
  function provenance(root, p) {
    const box = h("div", "prov", root), badge = h("span", "src " + p.source, box, sourceText(p.source));
    badge.title = SOURCE_HELP[p.source] || "";
    if (p.as_of) h("span", "", box, "as of " + p.as_of);
    const parts = Object.entries(p.sources || {});
    if (parts.length > 1) {
      const list = h("span", "srcs", box);
      for (const [k, v] of parts) { const item = h("span", "", list); item.appendChild(document.createTextNode(k.replace(/_/g, " ").replace("@", " on ") + ": ")); h("b", "", item, sourceText(v)); }
    }
    flags(root, p.warnings || []);
    if ((p.notes || []).length) { const d = h("details", "how", root); h("summary", "", d, "How this was made"); const ul = h("ul", "", d); for (const n of p.notes) h("li", "", ul, n); }
  }
  function flags(root, list, title) {
    const shown = list.slice(0, 4), rest = list.slice(4);
    for (const w of shown) { const b = h("div", "banner", root); h("strong", "", b, title || "Watch. "); b.appendChild(document.createTextNode(w)); }
    if (rest.length) { const d = h("details", "how", root); h("summary", "", d, `${rest.length} more`); const ul = h("ul", "", d); for (const w of rest) h("li", "", ul, w); }
  }

  /* ---------- small controls ---------- */
  function seg(parent, items, current, pick) {                                              // a row of buttons of which one is pressed
    const box = h("div", "seg", parent), buttons = [];
    for (const [value, text] of items) {
      const b = h("button", "", box, text); b.type = "button"; b.dataset.value = value; b.setAttribute("aria-pressed", String(value === current));
      b.addEventListener("click", () => { buttons.forEach((x) => x.setAttribute("aria-pressed", String(x === b))); pick(value); });
      buttons.push(b);
    }
    return box;
  }
  function pills(parent, items, current, pick) {                                            // rounded chips, for choosing among many (expiries, experiments)
    const box = h("div", "tabsmall", parent);
    for (const [value, text, title] of items) {
      const b = h("button", "", box, text); b.type = "button"; b.dataset.value = value; b.setAttribute("aria-selected", String(value === current)); if (title) b.title = title;
      b.addEventListener("click", () => pick(value));
    }
    return box;
  }
  function meter(parent, fraction, text, warnAt, critAt) {                                  // a share of a limit: the fill carries severity, the track is a lighter step of the same ramp
    const wrap = h("div", "meterwrap", parent), m = h("div", "meter " + (fraction >= (critAt || 1) ? "crit" : fraction >= (warnAt || 0.7) ? "warn" : ""), wrap), i = h("i", "", m);
    i.style.width = Math.max(0, Math.min(1, fraction)) * 100 + "%"; m.setAttribute("role", "img"); m.setAttribute("aria-label", text);
    if (text) h("span", "hint", wrap, text);
    return wrap;
  }
  function palette() {                                                                      // a colour per entity that does not change when others come and go
    const slots = new Map();
    return (key) => { if (!slots.has(key)) slots.set(key, slots.size); return Q.colorOf(slots.get(key)); };
  }
  function kv(parent, title, rows) {
    const box = h("div", "kv", parent); h("h4", "", box, title);
    Q.simpleTable(box, [{ label: "Measure", get: (x) => x[0] }, { label: "Value", num: true, get: (x) => x[1] }], rows.filter((x) => x[1] !== null && x[1] !== undefined));
    return box;
  }
  const sub = (root, text) => h("h4", "subhead", root, text);
  const hint = (root, text) => h("p", "hint", root, text);
  const card = (root, title, note) => { const c = h("div", "card", root); if (title) h("h3", "", c, title); if (note) h("p", "note", c, note); return c; };
  const signed = (v, d = 1) => (Q.bad(v) ? "n/a" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d));
  const points = (v, d = 1) => (Q.bad(v) ? "n/a" : signed(v * 100, d) + " pts");                // a volatility difference as percentage points
  const shortDate = (s) => String(s).slice(0, 10);
  const dayMonth = (s) => new Date(String(s).slice(0, 10) + "T00:00:00Z").toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });         // "Nov 15": a date short enough for a tile
  /* A list of numbers typed as "-10, -5, 0, 5" (percent) in a form, as fractions for a request. */
  function percentList(text, name) {
    const parts = String(text).split(/[\s,;]+/).filter(Boolean);
    if (!parts.length) return undefined;
    return parts.map((x) => { const v = Number(x); if (!isFinite(v)) throw new Error(`${name} must be numbers separated by commas`); return v / 100; });
  }
  function csvDownload(name, rows) {                                                        // the table behind a view, for a spreadsheet
    const cell = (v) => (v === null || v === undefined ? "" : /[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g, '""') + '"' : String(v));
    Q.download(name, rows.map((r) => r.map(cell).join(",")).join("\n"));
  }

  /* The result of a strategy that ran through the platform's pipeline (futures and crypto): the headline numbers, the equity curve against the equal-weight benchmark, the drawdown, the return of each calendar
     year and the latest positions. ``d`` is ``assets.strategy_runs.compact_result``; ``opts.benchmarkName`` names the benchmark ("Equal weight"). */
  function pipelineResult(root, d, opts) {
    const o = opts || {}, m = d.metrics, bn = o.benchmarkName || "Equal weight", tile = Q.tile, num = Q.num, pct = Q.pct, C = Q.C, css = Q.css, table = Q.simpleTable;
    const k = h("div", "earnrow", root), dsr = d.deflated_sharpe_probability;
    tile(k, "Net Sharpe ratio", num(m.sharpe), { text: `gross ${num(m.gross_sharpe)} · Sortino ${num(m.sortino)}` }, "", "Annualised excess return per unit of volatility, after costs");
    tile(k, "Return a year", pct(m.cagr), { text: `volatility ${pct(m.ann_vol)}` });
    tile(k, "Worst fall", pct(m.max_drawdown), { text: `Calmar ${num(m.calmar)}` });
    tile(k, "Turnover a year", num(m.ann_turnover, 1) + "×", { text: `cost drag ${num(m.ann_cost_bps, 0)} bps a year` });
    tile(k, "Days up", pct(m.hit_rate, 0), { text: `average gross exposure ${num(m.gross_exposure_mean, 2)}×` });
    tile(k, "Deflated Sharpe", dsr === null || dsr === undefined ? "n/a" : num(dsr), { text: `${d.n_trials || 1} trial${(d.n_trials || 1) === 1 ? "" : "s"}; 0.95 is the bar`, dir: dsr >= 0.95 ? "up" : "" }, "", "Probability the Sharpe ratio is above what the best of your tries would show by luck alone");
    const ch = h("div", "charts", root), series = [{ name: "Strategy", color: css("--s1"), values: d.equity }], bench = (d.benchmarks || {}).equal_weight;
    if (bench) series.push({ name: bn, color: css("--s2"), values: bench });
    Q.chartCard(ch, "Growth of 1", `Net of costs, from ${d.start} to ${d.end}.`, { wide: true,
      draw: (b) => C.line(b, { dates: d.dates, series, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 290, label: "Growth of 1", baseline: 1 }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, ...series.map((x) => ({ label: x.name, num: true, get: (n) => (x.values[n] === null ? "" : num(x.values[n], 4)) }))], d.dates.map((_, n) => n).filter((n) => n % 4 === 0)) });
    Q.chartCard(ch, "Drawdown from the previous peak", "How far below its high the strategy has been.", {
      draw: (b) => C.line(b, { dates: d.dates, series: [{ name: "Strategy", color: css("--s1"), values: d.drawdown }], area: true, areaBase: 0, baseline: 0, ceil: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => Q.fmt.pct(v), height: 230, label: "Drawdown" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, { label: "Drawdown", num: true, get: (n) => pct(d.drawdown[n]) }], d.dates.map((_, n) => n).filter((n) => n % 4 === 0)) });
    Q.chartCard(ch, "Return by calendar year", "Faded bars are partial years.", {
      draw: (b) => C.columns(b, { labels: d.years.map((y) => y.year), values: d.years.map((y) => y.value), yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: Q.fmt.spct, faded: d.years.map((y) => y.days < 240), height: 230, name: "Return", label: "Annual returns",
        tipLabel: (i) => d.years[i].year + (d.years[i].days < 240 ? ` (${d.years[i].days} trading days)` : "") }),
      table: (b) => table(b, [{ label: "Year", get: (y) => y.year }, { label: "Return", num: true, get: (y) => Q.fmt.spct(y.value) }, { label: "Days", num: true, get: (y) => y.days }], d.years) });
    const w = Object.entries(d.weights || {}).map(([label, value]) => ({ label, value }));
    if (w.length) Q.chartCard(ch, "Latest positions", "Weights on the last day: blue is long, red is short.", {
      draw: (b) => C.hbars(b, { rows: w, format: (v) => (v * 100).toFixed(0) + "%", tipFormat: Q.fmt.spct, name: "Weight", label: "Latest weights" }),
      table: (b) => table(b, [{ label: "Asset", get: (x) => x.label }, { label: "Weight", num: true, get: (x) => Q.fmt.spct(x.value) }], w) });
    const spec = d.spec || {}, card = h("div", "card", root); h("h3", "", card, "What ran");
    kv(card, "Settings", [["Strategy", d.label || d.strategy], ["Universe", (d.universe || []).join(", ")], ["Rebalance", (spec.execution || {}).rebalance || "the strategy's own"], ["Cost, one way", (spec.execution || {}).cost_bps === undefined ? null : num(spec.execution.cost_bps, 1) + " bps"],
      ["Volatility target", spec.risk ? (spec.risk.mode === "regime" ? "by regime (" + ((spec.regime || {}).detector || "") + ")" : pct(spec.risk.target_vol, 0)) : "the platform's default"], ["Evaluated days", String(d.n_days)]]);
  }


  /* ---------- free data: real data downloaded from free public sources ---------- */
  const WHAT_LABEL = { history: "daily history", venues: "order books", funding: "OKX funding", series: "price history", prices: "underlying prices" };           // the parts of a download
  let statusPromise = null, statusCache = null;
  const dataStatus = (fresh) => {                                                           // what is downloaded and what can be asked for (GET /api/data/status)
    if (fresh) statusPromise = null;
    if (!statusPromise) statusPromise = api("/api/data/status").then((s) => (statusCache = s), (e) => { statusPromise = null; throw e; });
    return statusPromise;
  };
  const refreshCatalog = () => { catalogPromise = null; return catalog(); };
  const ago = (iso) => {
    const t = Date.parse(iso); if (!isFinite(t)) return "";
    const s = Math.max(0, (Date.now() - t) / 1000);
    return s < 90 ? "just now" : s < 5400 ? Math.round(s / 60) + " min ago" : s < 129600 ? Math.round(s / 3600) + " h ago" : Math.round(s / 86400) + " days ago";
  };
  function picks(parent, items, chosen, after) {                                            // chips that switch on and off, for choosing several things
    const box = h("div", "picks", parent);
    for (const [value, text, title] of items) {
      const b = h("button", "", box, text); b.type = "button"; b.dataset.value = value; b.setAttribute("aria-pressed", String(chosen.has(value))); if (title) b.title = title;
      b.addEventListener("click", () => { if (chosen.has(value)) chosen.delete(value); else chosen.add(value); b.setAttribute("aria-pressed", String(chosen.has(value))); if (after) after(); });
    }
    return box;
  }
  /* iTick in a card: either the steps that turn it on (the key is never typed into this page) or the plan's allowance and a button that spends one call (two when the first host does not recognise the key)
     to show what the account returns. ``info`` is the server's iTick status (GET /api/data/status); ``kind`` ("futures" or "crypto") makes the test ask for that tab's own data, because a plan can include
     stocks and not futures. */
  function itickBox(parent, info, kind) {
    const box = h("div", "itick", parent);
    if (!info || !info.configured) {
      if (info && info.setup === "key") Q.setupGuide(box, "Come back to this card, choose iTick and press Download real data.");
      else { const w = h("p", "hint", box, (info && info.problem) || "iTick is not available."); w.style.color = Q.css("--warn"); }
      return box;
    }
    h("p", "hint", box, `Your iTick key is set. Your plan allows ${info.calls_per_minute} call${info.calls_per_minute === 1 ? "" : "s"} a minute (${info.calls_available_now} free right now), shared with the Backtest tab: a download waits between calls and says so while it does. Requests go to ${info.host || "iTick"}.`);
    const row = h("div", "row tight", box), test = h("button", "btn small", row, "Test the iTick connection (one or two calls)"), out = h("span", "hint", row); test.type = "button";
    out.dataset.role = "itick-test-result";
    test.addEventListener("click", async () => {
      test.disabled = true; out.textContent = "Testing…"; out.style.color = "";
      try { const r = await api("/api/source/test", kind ? { source: "itick", kind } : { source: "itick" }); out.textContent = r.message; out.style.color = Q.css(r.ok ? "--good-text" : "--crit"); }
      catch (e) { out.textContent = e.message; out.style.color = Q.css("--crit"); }
      test.disabled = false;
    });
    return box;
  }
  /* A search over the products iTick lists (the list is bundled with the app: searching costs no call). ``chosen`` is a Set of items such as "US:NQ" and ``names`` an object that remembers the name of each,
     so a chosen product keeps its label when the search moves on. ``opts``: kind ("futures" or "crypto"), regions [{id, name}] and groups for the futures filters, ``after`` (called when the choice changes). */
  function finder(parent, chosen, names, opts) {
    const box = h("div", "finder", parent), chips = h("div", "picks chosen", box), bar = h("div", "finderbar", box), q = h("input", "", bar), hits = h("div", "hits", box), note = h("p", "hint", box);
    q.type = "search"; q.placeholder = opts.kind === "crypto" ? "Search a coin: sol, doge, avalanche" : "Search a product: nasdaq, crude, gold, hang seng, rebar"; q.spellcheck = false; q.setAttribute("aria-label", "search iTick products");
    q.dataset.role = "finder-search";
    const state = { region: "", group: "" };
    const select = (label, key, options) => {
      const sel = h("select", "", bar); sel.setAttribute("aria-label", label); sel.dataset.role = "finder-" + key;
      for (const [value, text] of options) { const o = h("option", "", sel, text); o.value = value; }
      sel.addEventListener("change", () => { state[key] = sel.value; look(); });
    };
    if (opts.kind === "futures") {
      if (opts.regions) select("market", "region", [["", "All markets"], ...opts.regions.map((r) => [r.id, r.id === "US" ? "United States" : r.id === "HK" ? "Hong Kong" : "Mainland China"])]);
      select("group", "group", [["", "All groups"], ...(opts.groups || []).map((g) => [g, g])]);
    }
    const paintChosen = () => {
      chips.replaceChildren();
      if (!chosen.size) { h("span", "hint", chips, "Nothing chosen yet: click a product below."); return; }
      for (const item of chosen) {
        const b = h("button", "", chips, `${item}${names[item] ? " · " + names[item] : ""} ✕`); b.type = "button"; b.dataset.value = item; b.setAttribute("aria-pressed", "true"); b.title = "Remove";
        b.addEventListener("click", () => { chosen.delete(item); paintChosen(); paintHits(last); if (opts.after) opts.after(); });
      }
    };
    let last = null, seq = 0, timer = null;
    const paintHits = (res) => {
      last = res; hits.replaceChildren(); note.textContent = "";
      if (!res) return;
      for (const r of res.results) {
        const b = h("button", "hit", hits); b.type = "button"; b.dataset.value = r.item; b.setAttribute("aria-pressed", String(chosen.has(r.item)));
        h("b", "", b, r.item); h("span", "", b, r.name + (r.exchange ? " · " + r.exchange : "") + (r.kind === "contract" ? " · one contract" : ""));
        if (r.access) {                                                                      // what iTick gave this key when it was asked: the list is iTick's, the plan is yours
          const tag = h("span", "access " + (r.access.ok ? "ok" : "no"), b, r.access.ok ? "✓ worked" : "✗ not for your key");
          tag.title = (r.access.ok ? "iTick returned this product for your key" : "iTick did not return this product for your key") + " on " + r.access.at + (r.access.why ? ": " + r.access.why : "") + ". It may have changed since.";
        }
        b.addEventListener("click", () => {
          if (chosen.has(r.item)) chosen.delete(r.item); else { chosen.add(r.item); names[r.item] = r.name; }
          b.setAttribute("aria-pressed", String(chosen.has(r.item))); paintChosen(); if (opts.after) opts.after();
        });
      }
      const typed = String(q.value).trim();
      if (!res.results.length) {
        note.textContent = opts.kind === "crypto" ? "iTick's list has no such coin." : opts.source === "yahoo" ? "No future in the bundled list matches: ask Yahoo, below." : "No product in the list matches.";
        if (opts.source === "yahoo") {
          if (/^[A-Za-z0-9]{1,9}=F$/.test(typed)) {
            const item = typed.toUpperCase(), use = h("button", "link", note, ` Try ${item} anyway`); use.type = "button";
            use.addEventListener("click", () => { chosen.add(item); names[item] = ""; paintChosen(); if (opts.after) opts.after(); });
          }
        } else if (opts.kind === "futures" && /^([A-Za-z]{2}:)?[A-Za-z0-9.]{1,12}$/.test(typed)) {
          const item = /:/.test(typed) ? typed.toUpperCase() : "US:" + typed.toUpperCase(), use = h("button", "link", note, `Try ${item} anyway`); use.type = "button";
          use.addEventListener("click", () => { chosen.add(item); names[item] = ""; paintChosen(); if (opts.after) opts.after(); });
        }
      } else if (!typed && !state.region && !state.group) note.textContent = opts.source === "yahoo" ? `${res.shown} of the ${res.total} futures in the list, the best known first. Type a name to search them all.` : `The ${res.shown} products people look for most${res.total > res.shown ? " (of " + res.total + ")" : ""}. Type to search all of iTick's list.`;
      else if (res.total > res.shown) note.textContent = `${res.shown} of ${res.total} shown: type more to narrow the list.`;
    };
    let ask = null;
    if (opts.source === "yahoo") {                                                          // Yahoo has no list to read, only a search: ask it for the words typed (one to three calls) and keep the answer
      ask = h("button", "btn small", box, "Ask Yahoo for what you typed"); ask.type = "button"; ask.dataset.role = "finder-ask"; ask.hidden = true;
      ask.title = "Yahoo's lookup finds futures by name (lumber, nikkei, 6E). It makes one to three calls and the answer is kept, so the next search finds it without a call.";
      ask.addEventListener("click", async () => {
        const typed = String(q.value).trim(); if (!typed) return;
        ask.disabled = true; note.textContent = "Asking Yahoo…";
        try {
          const res = await api("/api/data/lookup", { kind: "futures", source: "yahoo", q: typed }); paintHits(res);
          note.textContent = res.found ? `Yahoo found ${res.found} future${res.found === 1 ? "" : "s"} for “${typed}”${res.new ? ` (${res.new} new to the list, kept)` : ""}.${res.incomplete ? " It stopped answering part-way." : ""}` : `Yahoo has no future for “${typed}”.`;
        } catch (e) { note.textContent = e.status === 404 ? "The app behind this page cannot ask Yahoo: it is older than the page. " + Q.RESTART : e.message; }
        ask.disabled = false;
      });
    }
    const look = async () => {
      const mine = ++seq;
      if (ask) ask.hidden = !String(q.value).trim();
      try { const res = await api("/api/data/search" + qs({ kind: opts.kind, source: opts.source, q: q.value, region: state.region, group: state.group, limit: 24 })); if (mine === seq) paintHits(res); }
      catch (e) { if (mine === seq) { hits.replaceChildren(); note.textContent = e.status === 404 ? "The app behind this page cannot search iTick's list: it is older than the page. " + Q.RESTART : e.message; } }
    };
    q.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(look, 200); });
    paintChosen(); look();
    return box;
  }

  /* The "Real data (free)" section of an asset tab. ``spec``: kind (options | futures | crypto), title, intro, defaults() (the form's initial state), controls(box, form, kindStatus, changed, redraw) (the inputs, filled
     from the server's presets; call ``changed`` when one is edited and ``redraw`` when the set of inputs itself changes), request(form) (the body for /api/data/fetch; throws a message), estimate(body) (seconds, for the hint beside the button), describe(record) ([headline, detail]), setCatalog(catalog), pick(result) (select what
     was downloaded). The card keeps its state in ``lab.free``, so it survives a redraw of the form while a download runs. */
  function dataCard(panel, lab, spec) {
    const st = lab.free = lab.free || { form: spec.defaults(), busy: "", error: "", last: null, open: null, armed: "" };
    const d = h("details", "freedata", panel); d.dataset.free = spec.kind; h("summary", "", d, spec.title);
    const body = h("div", "body", d); st.el = { details: d, body, live: null, list: null };
    d.addEventListener("toggle", () => { st.open = d.open; });
    const build = (status) => {
      if (!body.isConnected) return;
      const ks = status.kinds[spec.kind]; body.replaceChildren();
      if (st.open === null) st.open = ks.downloads.length === 0;
      d.open = st.open;
      h("p", "hint", body, spec.intro);
      const estimate = () => {                                                                // how long what is ticked will take, in words
        if (!st.el.estimate) return;
        let text = ""; try { const s = spec.estimate(spec.request(st.form)); text = s ? "takes about " + (s < 90 ? Math.round(s) + " seconds" : Math.round(s / 60) + " minutes") : ""; } catch (e) { /* not a valid request yet */ }
        st.el.estimate.textContent = text;
      };
      const box = h("div", "fdcontrols", body), redraw = () => { box.replaceChildren(); spec.controls(box, st.form, ks, estimate, redraw); estimate(); };         // ``redraw``: the controls themselves change (another source)
      spec.controls(box, st.form, ks, estimate, redraw);
      const row = h("div", "row tight fdgo", body), go = h("button", "btn primary", row, "Download real data"); go.type = "button"; go.dataset.action = "download";
      go.addEventListener("click", () => { let req; try { req = spec.request(st.form); } catch (e) { Q.toast(e.message); return; } submit(lab, spec, req); });
      const est = h("span", "hint", row, ""); est.dataset.role = "estimate"; st.el.estimate = est; st.el.go = go; estimate();
      st.el.live = h("div", "fdlive", body); st.el.live.setAttribute("aria-live", "polite"); st.el.list = h("div", "fdlist", body);
      const how = h("details", "how", body); h("summary", "", how, "Where this comes from"); const ul = h("ul", "", how);
      for (const s of ks.sources) { const li = h("li", "", ul); h("b", "", li, s.label); li.appendChild(document.createTextNode(`: ${s.provides}; ${s.delay}. `)); }
      h("li", "", ul, status.note);
      const more = h("button", "link", body, "Read more about the free data"); more.type = "button"; more.addEventListener("click", () => Q.openGuide("free_data"));
      paintLive(lab); paintList(lab, spec, status);
    };
    if (statusCache) build(statusCache);
    else {
      h("p", "hint", body, "Loading…");
      dataStatus().then(build, (e) => {
        if (!body.isConnected) return;
        body.replaceChildren(); d.open = true;                                                // the reason is shown, not left inside a closed card
        h("p", "hint", body, e.status === 404 ? "The app behind this page does not offer free data downloads: it is older than the page. " + Q.RESTART : e.message);
      });
    }
    st.spec = spec;
  }
  function paintLive(lab) {                                                                  // progress, error and the result of the last download
    const st = lab.free, el = st && st.el && st.el.live; if (!el || !el.isConnected) return;
    el.replaceChildren();
    if (st.el.go) st.el.go.disabled = !!st.busy;
    if (st.busy) { const p = h("div", "progress", el); h("span", "spinner", p); h("span", "", p, st.busy); }
    if (st.error) { const b = h("div", "banner err", el); h("strong", "", b, "Not downloaded. "); b.appendChild(document.createTextNode(st.error)); }
    if (st.last && !st.busy) {
      const ok = st.last.results.filter((r) => r.ok), bad = st.last.results.filter((r) => !r.ok), box = h("div", "fdres", el);
      h("p", "hint", box, `${ok.length} downloaded${bad.length ? `, ${bad.length} could not be` : ""}.`);
      const ul = h("ul", "", box);
      for (const r of st.last.results) { const li = h("li", r.ok ? "" : "fail", ul); h("b", "", li, (r.ok ? "✓ " : "✗ ") + r.item + (r.what && r.what !== "chain" && r.what !== "curve" ? " · " + (WHAT_LABEL[r.what] || r.what) : "")); li.appendChild(document.createTextNode(" " + r.message)); }
    }
  }
  function paintList(lab, spec, status) {                                                    // the downloads, newest first, each with its source, its date and what to do with it
    const st = lab.free, el = st.el && st.el.list; if (!el || !el.isConnected) return;
    const items = status.kinds[spec.kind].downloads; el.replaceChildren();
    if (!items.length) { h("p", "hint", el, "Nothing is downloaded yet: until it is, this tab shows a model."); return; }
    h("h4", "subhead", el, "Downloaded");
    const ul = h("ul", "dl", el);
    for (const rec of items) {
      const li = h("li", "", ul); li.dataset.name = rec.name; const [headline, detail] = spec.describe(rec);
      const main = h("div", "main", li); h("b", "", main, headline); h("span", "meta", main, `${detail} · downloaded ${ago(rec.fetched_at)}`);
      const acts = h("div", "acts", li), up = h("button", "btn small", acts, "Update"), del = h("button", "btn small", acts, "Delete"); up.type = del.type = "button";
      up.disabled = !rec.refresh; up.title = "Download it again"; up.dataset.action = "update"; del.dataset.action = "delete";
      up.addEventListener("click", () => submit(lab, spec, { kind: spec.kind, ...rec.refresh }));
      del.addEventListener("click", async () => {
        if (st.armed !== rec.name) { st.armed = rec.name; del.textContent = "Really delete?"; setTimeout(() => { if (st.armed === rec.name) { st.armed = ""; del.textContent = "Delete"; } }, 4000); return; }
        st.armed = ""; del.disabled = true;
        try { await api("/api/data/delete", { kind: spec.kind, name: rec.name, confirm: true }); st.last = null; st.error = ""; await changed(lab, spec); } catch (e) { st.error = e.message; paintLive(lab); }
      });
    }
  }
  async function changed(lab, spec) {                                                        // the downloads changed: ask again what exists, so the menus and the views follow
    const st = lab.free; await dataStatus(true);
    spec.setCatalog(await refreshCatalog());
    lab.cache.clear(); lab.results = {}; lab.errors = {};
    if (!Q.$(lab.view).hidden) Q.renderLab(lab); else if (st.el) paintList(lab, spec, statusCache);
  }
  async function submit(lab, spec, body) {
    const st = lab.free; if (st.busy) return;
    st.busy = "Starting…"; st.error = ""; st.last = null; paintLive(lab);
    try {
      const { job } = await api("/api/data/fetch", body);
      st.last = await Q.waitJob(job, (v) => { st.busy = `${v.message} (${Math.round(v.elapsed)} s)`; paintLive(lab); });
      st.busy = "";
      if (spec.pick) spec.pick(st.last);
      await changed(lab, spec);
    } catch (e) { st.error = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + Q.RESTART : e.message; }
    st.busy = ""; paintLive(lab);
  }

  window.WS = { pipelineResult, request, catalog, refreshCatalog, dataCard, picks, finder, itickBox, ago, provenance, flags, seg, pills, meter, palette, kv, sub, hint, card, signed, points, shortDate, dayMonth, percentList, csvDownload, qs, sourceText };
})();
