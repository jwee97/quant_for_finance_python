/* Shared parts of the Options, Futures, Crypto and Machine learning workspaces.
   A workspace is a lab (see app.js): a form on the left, the results on the right, one view per mode. This file holds what the four have in common: how a request is made and a long one is polled, how the
   provenance of a payload is shown (data, your file or a model), a stable colour for each entity, and a few small controls. Every string from the server goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, h = Q.h, api = Q.api;

  const SOURCE_TEXT = { platform: "Platform data", user_file: "Your file", simulated: "Simulated: a model", mixed: "Mixed sources" };
  const SOURCE_HELP = {
    platform: "Real data held in this repository (prices, funding, interest rates).",
    user_file: "Read from a file you put in data/user/.",
    simulated: "Made by a model on this computer. It is not a market: use it to check that the analytics and the machinery behave, not to look for an edge.",
    mixed: "Some parts are real data and some are simulated or yours; the parts are listed beside this badge." };
  const sourceText = (s) => SOURCE_TEXT[s] || String(s).replace(/_/g, " ");

  /* ---------- requests ---------- */
  const qs = (obj) => {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(obj || {})) if (v !== undefined && v !== null && v !== "") p.set(k, Array.isArray(v) ? v.join(",") : String(v));
    const s = p.toString(); return s ? "?" + s : "";
  };
  async function poll(lab, job, message) {                                                  // a long computation runs as a job on the server; ask after it until it is done
    for (;;) {
      await new Promise((r) => setTimeout(r, 500));
      const v = await api("/api/job/" + job);
      lab.busy = `${message} ${v.message} (${Math.round(v.elapsed)} s)`; Q.syncLab(lab);
      if (v.status === "done") return v.result;
      if (v.status === "error") throw new Error(v.error);
    }
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

  window.WS = { pipelineResult, request, catalog, provenance, flags, seg, pills, meter, palette, kv, sub, hint, card, signed, points, shortDate, dayMonth, percentList, csvDownload, qs, sourceText };
})();
