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
  /* Forget the answers kept for a workspace (the data changed, or Refresh was pressed). An answer still on its way was asked for before this, so it is shown when it arrives but not kept. */
  function dropCache(lab) { if (lab.cache) lab.cache.clear(); lab.gen = (lab.gen || 0) + 1; }
  async function request(lab, build, message) {
    if (lab.busy) { lab.queued = () => request(lab, build, message); return; }             // a control changed while a request was running: ask again, with what the form says now, when it ends
    let steps; try { steps = build(); } catch (e) { Q.toast(e.message); return; }
    const mode = lab.mode, gen = lab.gen || 0; lab.busy = message; Q.syncLab(lab);
    try {
      for (const st of steps) {
        let data = st.cacheKey && lab.cache ? lab.cache.get(st.cacheKey) : undefined;
        if (data === undefined) {
          if (st.job) data = await poll(lab, (await api(st.path, st.body)).job, message);
          else data = st.body ? await api(st.path, st.body) : await api(st.path + qs(st.query));
          if (st.cacheKey && lab.cache && (lab.gen || 0) === gen) lab.cache.set(st.cacheKey, data);        // not if the cache was dropped while this was on its way: it was asked for before the data changed
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
    if (lab.queued) { const next = lab.queued; lab.queued = null; next(); }
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
  function pills(parent, items, current, pick, label) {                                     // rounded chips, for choosing among many (expiries, experiments); toggle buttons in a group
    const box = h("div", "tabsmall", parent); box.setAttribute("role", "group"); if (label) box.setAttribute("aria-label", label);
    for (const [value, text, title] of items) {
      const b = h("button", "", box, text); b.type = "button"; b.dataset.value = value; b.setAttribute("aria-pressed", String(value === current)); if (title) b.title = title;
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

  /* The result of a strategy that ran through the platform's pipeline (futures and crypto): the headline numbers and a second row of risk measures, the equity curve against the equal-weight benchmark, the
     drawdown with its deepest falls shaded, the volatility through time, the returns by year, calendar month and month, the distribution of days, how the book was spread across the assets through time,
     what each asset earned, and every risk measure with what it means. ``d`` is ``assets.strategy_runs.compact_result``; ``opts.benchmarkName`` names the benchmark ("Equal weight"). */
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  /* The charts that describe a daily return series, shared by every page that has one (the pipeline result, the options backtest): the drawdown with its deepest falls shaded, the volatility through time, the
     returns by year, by calendar month and by month, and the distribution of days with its tail measures. ``d`` carries dates, drawdown, drawdown_episodes, rolling_vol, years, calendar_months, monthly,
     distribution and risk (``risk.performance``); each chart is left out when its data is. */
  /* A percentage axis with as many decimals as its range needs, so that neighbouring ticks never read alike ("-0%, -0%, 0%" for a strategy that moves a tenth of a percent); ``floor`` is the fewest decimals. */
  function pctAxis(values, floor) {
    const v = values.filter((x) => x !== null && x !== undefined && isFinite(x)), span = v.length ? Math.max(...v) - Math.min(...v) : 1, digits = span >= 0.05 ? 0 : span >= 0.005 ? 1 : span >= 0.0005 ? 2 : 3;
    return (x) => (x * 100).toFixed(Math.max(digits, floor || 0)) + "%";
  }
  function riskCharts(ch, d) {
    const num = Q.num, pct = Q.pct, C = Q.C, css = Q.css, table = Q.simpleTable, fmt = Q.fmt, rk = d.risk || {};
    const na = (v, f) => (Q.bad(v) ? "n/a" : f(v)), rows = (n) => d.dates.map((_, i) => i).filter((i) => i % n === 0);
    const episodes = d.drawdown_episodes || [], last = d.dates[d.dates.length - 1];
    Q.chartCard(ch, "Drawdown from the previous peak", episodes.length ? "How far below its high the strategy has been; the three deepest falls are shaded, from the peak to the day it was regained. The Table view lists the five deepest." : "How far below its high the strategy has been.", {
      draw: (b) => C.line(b, { dates: d.dates, series: [{ name: "Strategy", color: css("--s1"), values: d.drawdown }], area: true, areaBase: 0, baseline: 0, ceil: 0, yFormat: pctAxis(d.drawdown), tipFormat: (v) => fmt.pct(v), height: 230, label: "Drawdown",
        regions: episodes.slice(0, 3).map((e, i) => ({ from: e.peak, to: e.recovered || last, name: `#${i + 1}  ${pctAxis(d.drawdown)(e.depth)}`, color: css("--neg"), opacity: 0.08 })) }),
      table: (b) => (episodes.length
        ? table(b, [{ label: "Peak", get: (e) => e.peak }, { label: "Deepest", get: (e) => e.trough }, { label: "Regained", get: (e) => e.recovered || "not yet" }, { label: "Fall", num: true, get: (e) => fmt.pct(e.depth) },
          { label: "Days down to the low", num: true, get: (e) => e.days_to_trough }, { label: "Days in all", num: true, get: (e) => e.days }], episodes)
        : table(b, [{ label: "Date", get: (n) => d.dates[n] }, { label: "Drawdown", num: true, get: (n) => pct(d.drawdown[n]) }], rows(4))) });
    const rv = d.rolling_vol || [], live = rv.filter((v) => v !== null);
    if (live.length > 20) {
      const mean = live.reduce((a, b) => a + b, 0) / live.length;
      Q.chartCard(ch, "Rolling three-month volatility", "Annualised standard deviation of the last 63 daily returns. The dashed line is its average: a strategy whose risk swings far from it is not the one the volatility tile describes.", {
        draw: (b) => C.line(b, { dates: d.dates, series: [{ name: "63-day volatility", color: css("--s1"), values: rv }, { name: "Average", color: css("--muted"), dash: "5 4", width: 1.25, values: rv.map((v) => (v === null ? null : mean)) }],
          include: 0, yFormat: pctAxis(live), tipFormat: (v) => fmt.pct(v), height: 230, label: "Rolling volatility" }),
        table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, { label: "63-day volatility", num: true, get: (n) => (rv[n] === null ? "" : fmt.pct(rv[n])) }], rows(4)) });
    }
    Q.chartCard(ch, "Return by calendar year", "Faded bars are partial years.", {
      draw: (b) => C.columns(b, { labels: d.years.map((y) => y.year), values: d.years.map((y) => y.value), yFormat: pctAxis([0, ...d.years.map((y) => y.value)]), tipFormat: Q.fmt.spct, faded: d.years.map((y) => y.days < 240), height: 230, name: "Return", label: "Annual returns",
        tipLabel: (i) => d.years[i].year + (d.years[i].days < 240 ? ` (${d.years[i].days} trading days)` : "") }),
      table: (b) => table(b, [{ label: "Year", get: (y) => y.year }, { label: "Return", num: true, get: (y) => Q.fmt.spct(y.value) }, { label: "Days", num: true, get: (y) => y.days }], d.years) });
    const cm = d.calendar_months || [];
    if (cm.length >= 6) Q.chartCard(ch, "Average return by calendar month", "The mean of every January, every February and so on over the backtest. With a few years of each it describes the sample; it is not a calendar to trade.", {
      draw: (b) => C.columns(b, { labels: cm.map((c) => MONTHS[c.month - 1]), values: cm.map((c) => c.mean), yFormat: pctAxis([0, ...cm.map((c) => c.mean)], 1), tipFormat: fmt.spct, colors: (v) => (v >= 0 ? css("--pos") : css("--neg")), height: 230, name: "Average return",
        tipLabel: (i) => `${MONTHS[cm[i].month - 1]} · ${cm[i].n} years · up in ${(cm[i].hit_rate * 100).toFixed(0)}% of them`, label: "Average return by calendar month" }),
      table: (b) => table(b, [{ label: "Month", get: (c) => MONTHS[c.month - 1] }, { label: "Years", num: true, get: (c) => c.n }, { label: "Average", num: true, get: (c) => fmt.spct(c.mean) }, { label: "Median", num: true, get: (c) => fmt.spct(c.median) },
        { label: "Best", num: true, get: (c) => fmt.spct(c.best) }, { label: "Worst", num: true, get: (c) => fmt.spct(c.worst) }, { label: "Up in", num: true, get: (c) => pct(c.hit_rate, 0) }], cm) });
    if ((d.monthly || []).length >= 6) Q.chartCard(ch, "Monthly returns", "Blue is a gain and red a loss; shade shows size (capped at ±12%).", { wide: true,
      draw: (b) => C.heatmap(b, { cells: d.monthly, label: "Monthly returns" }),
      table: (b) => table(b, [{ label: "Month", get: (c) => `${c.year}-${String(c.month).padStart(2, "0")}` }, { label: "Return", num: true, get: (c) => fmt.spct(c.value) }], d.monthly.slice().reverse()) });
    const dist = d.distribution;
    if (dist && dist.counts) Q.chartCard(ch, "Distribution of daily returns", "The central 99% of days: the height is the share of days in each band of return. The second line is the normal distribution with the same mean and spread: where the strategy's line is higher at the centre and in the tails, its days are more extreme than a bell curve says. Markers: the loss exceeded on 5% of days (VaR) and the average loss beyond it (ES).", {
      draw: (b) => C.histogram(b, { series: [{ name: "Strategy", edges: dist.edges, counts: dist.counts, color: css("--s1") }, { name: "Normal, same mean and spread", edges: dist.edges, counts: dist.normal, color: css("--s2") }],
        markers: [{ x: 0, label: "0" }, ...(rk.var_95 === undefined ? [] : [{ x: -rk.var_95, label: "VaR" }, { x: -rk.es_95, label: "ES" }])], xFormat: pctAxis(dist.edges, 1), unit: dist.edges[1] - dist.edges[0], yFormat: (v) => (v * 100).toFixed(0) + "%", height: 230, label: "Distribution of daily returns" }),
      table: (b) => table(b, [{ label: "Measure", get: (x) => x[0] }, { label: "Value", num: true, get: (x) => x[1] }], [["Days", String(dist.n)], ["Mean daily return", fmt.spct(dist.mean, 3)], ["Daily standard deviation", pct(dist.sd, 2)],
        ["Loss exceeded on 5% of days (VaR 95%)", na(rk.var_95, (v) => pct(v, 2))], ["Average loss on those days (ES 95%)", na(rk.es_95, (v) => pct(v, 2))]]) });
  }
  /* "Every risk measure": the ratios, the losses and the shape of the returns in three columns of labelled figures, and the beta and alpha against each benchmark in ``d.benchmark_relative``. */
  function riskGrid(root, d, benchmarkName) {
    const num = Q.num, pct = Q.pct, fmt = Q.fmt, rk = d.risk || {}, bn = benchmarkName || "Equal weight";
    const na = (v, f) => (Q.bad(v) ? "n/a" : f(v)), n2 = (v) => num(v);
    if (rk.var_95 !== undefined) {
      const rc = card(root, "Every risk measure", "Over the whole backtest, from daily net returns, 252 days to a year, with a risk-free rate of zero. A ratio whose denominator is zero (no losing day, no fall) is n/a."), grid = h("div", "kvgrid", rc);
      kv(grid, "Return for the risk taken", [["Sortino ratio (downside risk only)", na(rk.sortino, n2)], ["Calmar ratio (CAGR ÷ deepest fall)", na(rk.calmar, n2)], ["Martin ratio (CAGR ÷ Ulcer index)", na(rk.martin_ratio, n2)],
        ["Omega ratio (gains ÷ losses)", na(rk.omega, n2)], ["Gain to pain (sum of months ÷ sum of the losing months)", na(rk.gain_to_pain, n2)], ["Recovery factor (total return ÷ deepest fall)", na(rk.recovery_factor, n2)], ["Kelly leverage (mean ÷ variance of daily returns)", na(rk.kelly_leverage, n2)]]);
      kv(grid, "Losses", [["Loss exceeded on 1 day in 20 (VaR 95%)", na(rk.var_95, (v) => pct(v, 2))], ["Average loss on those days (ES 95%)", na(rk.es_95, (v) => pct(v, 2))], ["Loss exceeded on 1 day in 100 (VaR 99%)", na(rk.var_99, (v) => pct(v, 2))],
        ["Cornish-Fisher VaR 95% (adjusts for skew and fat tails)", na(rk.cornish_fisher_var_95, (v) => pct(v, 2))], ["Average drawdown", na(rk.average_drawdown, (v) => pct(v, 1))], ["Pain index (mean depth of the drawdown)", na(rk.pain_index, (v) => pct(v, 1))]]);
      kv(grid, "Shape of the returns", [["Days with a gain", na(rk.positive_days, (v) => pct(v, 1))], ["Skewness", na(rk.skew, n2)], ["Excess kurtosis", na(rk.excess_kurtosis, n2)], ["Tail ratio (95th ÷ 5th percentile of daily returns)", na(rk.tail_ratio, n2)],
        ["Years of returns needed for 95% confidence that the Sharpe ratio is above 0", na(rk.min_track_record_years, (v) => num(v, 1))]]);
      for (const [name, v] of Object.entries(d.benchmark_relative || {})) {
        if (!v || v.beta === undefined) continue;
        kv(grid, "Against " + (name === "equal_weight" ? bn.toLowerCase() : name.replace(/_/g, " ")), [["Beta", na(v.beta, n2)], ["Alpha a year (after beta)", na(v.alpha_annual, (x) => fmt.spct(x))], ["Correlation", na(v.correlation, n2)], ["Tracking error", na(v.tracking_error, (x) => pct(x, 1))],
          ["Information ratio", na(v.information_ratio, n2)], ["Up-capture (gain on its up days)", na(v.up_capture, (x) => pct(x, 0))], ["Down-capture (loss on its down days)", na(v.down_capture, (x) => pct(x, 0))]]);
      }
    }
  }

  function pipelineResult(root, d, opts) {
    const o = opts || {}, m = d.metrics, bn = o.benchmarkName || "Equal weight", tile = Q.tile, num = Q.num, pct = Q.pct, C = Q.C, css = Q.css, table = Q.simpleTable, fmt = Q.fmt, rk = d.risk || {};
    const na = (v, f) => (Q.bad(v) ? "n/a" : f(v)), n2 = (v) => num(v), rows = (n) => d.dates.map((_, i) => i).filter((i) => i % n === 0);
    const k = h("div", "earnrow", root), dsr = d.deflated_sharpe_probability;
    tile(k, "Net Sharpe ratio", num(m.sharpe), { text: `gross ${num(m.gross_sharpe)} · Sortino ${num(m.sortino)}` }, "", "Annualised excess return per unit of volatility, after costs");
    tile(k, "Return a year", pct(m.cagr), { text: `volatility ${pct(m.ann_vol)}` });
    tile(k, "Worst fall", pct(m.max_drawdown), { text: `Calmar ${num(m.calmar)}` });
    tile(k, "Turnover a year", num(m.ann_turnover, 1) + "×", { text: `cost drag ${num(m.ann_cost_bps, 0)} bps a year` });
    tile(k, "Days up", pct(m.hit_rate, 0), { text: `average gross exposure ${num(m.gross_exposure_mean, 2)}×` });
    tile(k, "Deflated Sharpe", dsr === null || dsr === undefined ? "n/a" : num(dsr), { text: `${d.n_trials || 1} trial${(d.n_trials || 1) === 1 ? "" : "s"}; 0.95 is the bar`, dir: dsr >= 0.95 ? "up" : "" }, "", "Probability the Sharpe ratio is above what the best of your tries would show by luck alone");
    if (rk.var_95 !== undefined) {
      const k2 = h("div", "earnrow", root);
      tile(k2, "Sortino ratio", na(rk.sortino, n2), { text: "return per unit of downside risk" }, "", "Annualised mean return over the root mean square of the losing days (Sortino and Price 1994). Unlike the Sharpe ratio it does not count a big gain as risk");
      tile(k2, "Calmar ratio", na(rk.calmar, n2), { text: "CAGR ÷ deepest fall" }, "", "Compound annual growth over the size of the deepest fall from a peak (Young 1991)");
      tile(k2, "Ulcer index", na(rk.ulcer_index, (v) => num(v, 1)), { text: rk.martin_ratio === null ? "" : `Martin ratio ${num(rk.martin_ratio)}` }, "", "Root mean square of the drawdowns, in percentage points: it counts how deep and how long the falls were, not just the worst one (Martin and McCann 1989). The Martin ratio is CAGR over it");
      tile(k2, "Omega ratio", na(rk.omega, n2), { text: "gains ÷ losses" }, "", "Total of the days with a gain over the total of the days with a loss (Keating and Shadwick 2002); above 1 means the gains outweigh the losses");
      tile(k2, "A bad day", pct(rk.var_95, 2), { text: `lost on 1 day in 20 · average beyond it ${na(rk.es_95, (v) => pct(v, 2))}` }, "", "Historical value at risk (VaR) and expected shortfall (ES) at 95% over one day");
      tile(k2, "Track record needed", na(rk.min_track_record_years, (v) => `${num(v, 1)} years`), { text: "for 95% confidence" }, "", "Bailey and Lopez de Prado (2012): how many years of returns with this Sharpe ratio, skewness and kurtosis it takes before luck can be ruled out at 95%. Compare it with the length of this backtest");
    }
    const ch = h("div", "charts", root), series = [{ name: "Strategy", color: css("--s1"), values: d.equity }], bench = (d.benchmarks || {}).equal_weight;
    let log = false;
    if (bench) series.push({ name: bn, color: css("--s2"), values: bench });
    const growth = Q.chartCard(ch, "Growth of 1", `Net of costs, from ${d.start} to ${d.end}.`, { wide: true,
      draw: (b) => C.line(b, { dates: d.dates, series, log, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 290, label: "Growth of 1", baseline: 1 }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, ...series.map((x) => ({ label: x.name, num: true, get: (n) => (x.values[n] === null ? "" : num(x.values[n], 4)) }))], rows(4)) });
    const lg = h("div", "seg", growth._extra), l1 = h("button", "", lg, "Linear"), l2 = h("button", "", lg, "Log");
    const mark = () => { l1.setAttribute("aria-pressed", String(!log)); l2.setAttribute("aria-pressed", String(log)); };
    mark();
    l1.addEventListener("click", () => { log = false; mark(); growth._draw(); });
    l2.addEventListener("click", () => { log = true; mark(); growth._draw(); });
    riskCharts(ch, d);
    const pos = d.positions, ex = d.exposure;
    if (pos && pos.assets.length > 1 && pos.months.length > 2) Q.chartCard(ch, "Positions through time", "The average weight of each asset in each month: blue is long, red short, grey flat. A book that sits in one colour for years is a bet on one direction.", { wide: true,
      draw: (b) => C.matrix(b, { rows: pos.assets, cols: pos.months.map((x) => x.slice(2)), values: pos.values, diverging: true, mid: 0, format: (v) => (v * 100).toFixed(0) + "%", name: "Average weight", cellText: false, cellHeight: 22, label: "Positions through time" }),
      table: (b) => table(b, [{ label: "Month", get: (n) => pos.months[n] }, ...pos.assets.map((a, j) => ({ label: a, num: true, get: (n) => fmt.spct(pos.values[j][n]) }))], pos.months.map((_, n) => n).reverse()) });
    if (ex && ex.dates.length > 8) Q.chartCard(ch, "Gross and net exposure", "Weekly. Net 0 with gross 1 is a market-neutral book; gross above 1 uses leverage.", {
      draw: (b) => C.line(b, { dates: ex.dates, series: [{ name: "Gross", color: css("--s1"), values: ex.gross }, { name: "Net", color: css("--s2"), values: ex.net }], baseline: 0, include: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), height: 230, label: "Exposure" }),
      table: (b) => table(b, [{ label: "Week", get: (n) => ex.dates[n] }, { label: "Gross", num: true, get: (n) => num(ex.gross[n]) }, { label: "Net", num: true, get: (n) => num(ex.net[n]) }], ex.dates.map((_, n) => n).filter((n) => n % 4 === 0)) });
    const w = Object.entries(d.weights || {}).map(([label, value]) => ({ label, value }));
    if (w.length) Q.chartCard(ch, "Latest positions", "Weights on the last day: blue is long, red is short.", {
      draw: (b) => C.hbars(b, { rows: w, format: (v) => (v * 100).toFixed(0) + "%", tipFormat: Q.fmt.spct, name: "Weight", label: "Latest weights" }),
      table: (b) => table(b, [{ label: "Asset", get: (x) => x.label }, { label: "Weight", num: true, get: (x) => Q.fmt.spct(x.value) }], w) });
    const ct = (d.contribution || []).slice(0, 15).map((x) => ({ label: x.asset, value: x.contribution_pct / 100, tip: [["Average weight", fmt.spct(x.average_weight)], ["Class", x.asset_class]] }));
    if (ct.length > 1) Q.chartCard(ch, "Return contributed by asset", "Total return points over the backtest, before compounding and costs: what each asset's held weight earned.", {
      draw: (b) => C.hbars(b, { rows: ct, format: (v) => (v * 100).toFixed(0) + " pts", tipFormat: (v) => (v * 100).toFixed(1) + " points", name: "Contribution", label: "Contribution by asset" }),
      table: (b) => table(b, [{ label: "Asset", get: (x) => x.asset }, { label: "Contribution (points)", num: true, get: (x) => num(x.contribution_pct, 1) }, { label: "Average weight", num: true, get: (x) => fmt.spct(x.average_weight) }, { label: "Class", get: (x) => x.asset_class }], d.contribution) });
    riskGrid(root, d, bn);
    const spec = d.spec || {}, own = ((spec.models || [])[0] || {}).params, what = h("div", "card", root); h("h3", "", what, "What ran");
    kv(what, "Settings", [["Strategy", d.label || d.strategy], ["Universe", (d.universe || []).join(", ")], ["Rebalance", (spec.execution || {}).rebalance || "the strategy's own"], ["Cost, one way", (spec.execution || {}).cost_bps === undefined ? null : num(spec.execution.cost_bps, 1) + " bps"],
      ["Volatility target", spec.risk ? (spec.risk.mode === "regime" ? "by regime (" + ((spec.regime || {}).detector || "") + ")" : pct(spec.risk.target_vol, 0)) : "the platform's default"], ["Evaluated days", String(d.n_days)],
      ["Strategy settings", own && Object.keys(own).length ? Object.entries(own).map(([k, v]) => `${k.replace(/_/g, " ")} ${JSON.stringify(v)}`).join(", ") : "the strategy's own"]]);
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
    const past = Q.keyHistory(info.history); if (past) h("p", "hint", box, past);
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
  /* Finzdata in a card: the steps that turn it on (the key is never typed into this page), or the day's calls and the markets already stored with how long each has before a window of 30 days slides past
     its newest bar. ``info`` is the server's Finzdata status (GET /api/data/status); ``hooks.choose(roots)`` adds the stored markets to the choice. */
  const FINZ_GUIDE = { product: "Finzdata", name: "FINZDATA_API_KEY", get: "Get a free API key at finzdata.com (it asks only for an email) and copy it.", last: "Come back to this card, choose Finzdata and press Download real data." };
  function finzdataBox(parent, info, hooks) {
    const box = h("div", "finzdata", parent); box.dataset.role = "finzdata-box";
    if (!info || !info.configured) {
      if (info && info.setup === "key") Q.setupGuide(box, undefined, FINZ_GUIDE);
      else { const w = h("p", "hint", box, (info && info.problem) || "Finzdata is not available."); w.style.color = Q.css("--warn"); }
      return box;
    }
    h("p", "hint", box, `Your Finzdata key is set. ${info.calls_today.toLocaleString("en-US")} of the ${info.budget.toLocaleString("en-US")} calls this app sends in a day have been used today (the free plan allows ${info.daily_allowance.toLocaleString("en-US")}); a market costs two.`);
    for (const note of info.key_notes || []) { const w = h("p", "hint", box, "The key: " + note + "."); w.style.color = Q.css("--warn"); }
    if (!(info.stored || []).length) { h("p", "hint", box, `Nothing is stored from Finzdata yet. The free plan shows only the last ${info.window_days} days, so download a market now and again later: each download is merged into the stored file and the history grows.`); return box; }
    h("h3", "subhead", box, "Stored from Finzdata");
    Q.simpleTable(box, [{ label: "Market", get: (r) => `${r.root} · ${r.name}` }, { label: "Stored", get: (r) => `${r.stored_from} to ${r.stored_to}` }, { label: "Days", num: true, get: (r) => String(r.days) },
      { label: "Contracts", num: true, get: (r) => String(r.contracts) }, { label: "Open interest", get: (r) => (r.has_open_interest ? "yes" : "no") },
      { label: "Last bar", num: true, get: (r) => `${r.age} days ago` }, { label: "Window left", num: true, title: "Days before the plan's window slides past the newest stored bar", get: (r) => (r.left === null ? "full history" : r.urgent ? `${Math.max(r.left, 0)} days: update now` : `${r.left} days`) },
      { label: "Gaps", num: true, get: (r) => (r.gap_count ? `${r.gap_count}: ${r.gaps.slice(0, 2).map((g) => g.from + " to " + g.to).join(", ")}${r.gap_count > 2 ? " …" : ""}` : "none") }], info.stored);
    if (hooks && hooks.choose) {
      const row = h("div", "row tight", box), all = h("button", "btn small", row, `Choose the ${info.stored.length} stored market${info.stored.length === 1 ? "" : "s"} to update`); all.type = "button"; all.dataset.action = "choose-stored";
      all.addEventListener("click", () => hooks.choose(info.stored));
      h("span", "hint", row, "then press Download real data");
    }
    h("p", "hint", box, info.terms);
    return box;
  }
  /* Every contract Finzdata lists for one market (GET /api/data/contracts): a table with each contract's expiry, whether it still trades, its first and last bar and what is stored of it. */
  function finzdataContracts(parent, getRoot) {
    const box = h("div", "finzcontracts", parent), row = h("div", "row tight", box), go = h("button", "btn small", row, "List every contract of the market chosen above"), out = h("div", "", box);
    go.type = "button"; go.dataset.action = "list-contracts"; out.setAttribute("aria-live", "polite");
    const show = async (refresh) => {
      const root = getRoot(); if (!root) { out.textContent = "Choose a market above first."; return; }
      go.disabled = true; out.textContent = `Asking Finzdata for the contracts of ${root}…`;
      try {
        const r = await api("/api/data/contracts" + qs({ root, refresh: refresh ? 1 : "" }));
        out.replaceChildren();
        h("p", "hint", out, `${r.root} · ${r.name}: ${r.listed} contracts (${r.live} still trading, ${r.expired} expired); ${r.stored} stored here. ${r.called ? "Fetched just now" : "From the list kept " + ago(r.fetched_at)}.` +
          (Object.keys(r.basis).some((k) => k !== "service") ? ` Expiry dates: ${Object.entries(r.basis).map(([k, n]) => n + " from " + (k === "service" ? "Finzdata" : k === "last bar" ? "the last bar" : k === "rule" ? "the exchange's rule" : "an estimate (the 15th)")).join(", ")}.` : ""));
        Q.simpleTable(out, [{ label: "Contract", get: (c) => c.contract }, { label: "Code", get: (c) => c.code }, { label: "Expiry", get: (c) => c.expiry + (c.expiry_basis === "service" ? "" : " ~") }, { label: "Status", get: (c) => c.status },
          { label: "First bar", get: (c) => c.first_date || "" }, { label: "Last bar", get: (c) => c.last_date || "" }, { label: "Stored", num: true, get: (c) => (c.stored_rows ? `${c.stored_rows} rows` : "") }], r.contracts);
        const again = h("button", "link", out, "Fetch the list again"); again.type = "button"; again.addEventListener("click", () => show(true));
      } catch (e) { out.textContent = e.status === 404 ? "The app behind this page cannot list contracts: it is older than the page. " + Q.RESTART : e.message; }
      go.disabled = false;
    };
    go.addEventListener("click", () => show(false));
    return box;
  }
  /* A search over the products iTick lists (the list is bundled with the app: searching costs no call). ``chosen`` is a Set of items such as "US:NQ" and ``names`` an object that remembers the name of each,
     so a chosen product keeps its label when the search moves on. ``opts``: kind ("futures" or "crypto"), regions [{id, name}] and groups for the futures filters, ``after`` (called when the choice changes). */
  function finder(parent, chosen, names, opts) {
    const box = h("div", "finder", parent), chips = h("div", "picks chosen", box), bar = h("div", "finderbar", box), q = h("input", "", bar), hits = h("div", "hits", box), note = h("p", "hint", box);
    const finz = opts.source === "finzdata";
    q.type = "search"; q.placeholder = opts.kind === "crypto" ? "Search a coin: sol, doge, avalanche" : finz ? "Search a market: nasdaq, crude, gold, euro, dax" : "Search a product: nasdaq, crude, gold, hang seng, rebar"; q.spellcheck = false;
    q.setAttribute("aria-label", finz ? "search Finzdata markets" : "search iTick products");
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
        h("b", "", b, r.item); h("span", "", b, r.name + (r.exchange ? " · " + r.exchange : "") + (r.kind === "contract" ? " · one contract" : "") + (finz ? ` · ${r.group}${r.history_from ? " · since " + r.history_from : ""}${r.contracts ? " · " + r.contracts + " contracts" : ""}` : ""));
        if (finz && r.stored_to) { const tag = h("span", "access ok", b, "✓ stored to " + r.stored_to); tag.title = `${r.stored_days} days are stored here`; }
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
        note.textContent = opts.kind === "crypto" ? "iTick's list has no such coin." : opts.source === "yahoo" ? "No future in the bundled list matches: ask Yahoo, below." : finz ? "No market in Finzdata's list matches: try the symbol, or refresh the list from Finzdata below." : "No product in the list matches.";
        if (opts.source === "yahoo") {
          if (/^[A-Za-z0-9]{1,9}=F$/.test(typed)) {
            const item = typed.toUpperCase(), use = h("button", "link", note, ` Try ${item} anyway`); use.type = "button";
            use.addEventListener("click", () => { chosen.add(item); names[item] = ""; paintChosen(); if (opts.after) opts.after(); });
          }
        } else if (finz) {
          if (/^[A-Za-z0-9]{1,6}$/.test(typed)) {
            const item = typed.toUpperCase(), use = h("button", "link", note, ` Try ${item} anyway`); use.type = "button";
            use.addEventListener("click", () => { chosen.add(item); names[item] = ""; paintChosen(); if (opts.after) opts.after(); });
          }
        } else if (opts.kind === "futures" && /^([A-Za-z]{2}:)?[A-Za-z0-9.]{1,12}$/.test(typed)) {
          const item = /:/.test(typed) ? typed.toUpperCase() : "US:" + typed.toUpperCase(), use = h("button", "link", note, `Try ${item} anyway`); use.type = "button";
          use.addEventListener("click", () => { chosen.add(item); names[item] = ""; paintChosen(); if (opts.after) opts.after(); });
        }
      } else if (!typed && !state.region && !state.group) note.textContent = opts.source === "yahoo" ? `${res.shown} of the ${res.total} futures in the list, the best known first. Type a name to search them all.` : finz ? `The ${res.total} markets people look for most, of the ${(opts.markets || 139)} Finzdata lists. Type a name to search them all, or choose an asset class.` : `The ${res.shown} products people look for most${res.total > res.shown ? " (of " + res.total + ")" : ""}. Type to search all of iTick's list.`;
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
    if (finz) {                                                                              // Finzdata's own list of markets replaces the copy that ships with the app (one call)
      const fresh = h("button", "btn small", box, "Refresh the list from Finzdata (1 call)"); fresh.type = "button"; fresh.dataset.role = "finder-refresh";
      fresh.title = "Asks Finzdata for the markets it lists now and keeps the answer, so the search shows its names and contract counts.";
      fresh.addEventListener("click", async () => {
        fresh.disabled = true; note.textContent = "Asking Finzdata…";
        try { const res = await api("/api/data/lookup", { kind: "futures", source: "finzdata", q: q.value }); paintHits(res); note.textContent = `Finzdata lists ${res.markets} markets${res.new.length ? `; new to this app: ${res.new.slice(0, 8).join(", ")}` : ""}${res.missing_count ? `; ${res.missing_count} of the bundled list are no longer sent` : ""}.`; }
        catch (e) { note.textContent = e.message; }
        fresh.disabled = false;
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
    h("h3", "subhead", el, "Downloaded");
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
    dropCache(lab); lab.results = {}; lab.errors = {};
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

  window.WS = { pipelineResult, riskCharts, riskGrid, pctAxis, request, dropCache, catalog, refreshCatalog, dataCard, picks, finder, itickBox, finzdataBox, finzdataContracts, ago, provenance, flags, seg, pills, meter, palette, kv, sub, hint, card, signed, points, shortDate, dayMonth, percentList, csvDownload, qs, sourceText };
})();
