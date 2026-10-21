/* The research views of the Fundamentals tab: the Factor lab (one factor or formula studied against what happened next, or every factor scanned), Alpha discovery (a search whose every attempt is counted and whose
   best formulas are tested once on dates they never saw), the Alpha library (the alphas, the graveyard of failures, their correlation, combinations, a model fitted walk-forward, and the backtest of an alpha
   through the platform's own engine) and the Alpha monitor. They register themselves on the object fundamentals.js made. Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const FD = window.FD, Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable, api = Q.api;
  const { spct, sgn, compact, human, day } = FD.fmt;
  const GUIDE = "fundamentals";
  const isOpen = () => !Q.$(FD.view).hidden;
  const levelOf = (study) => {                                                                // a verdict is never colour alone: the icon and the words carry it too
    if (study.evidence_level) return study.evidence_level;                                    // the app says it, with the sentence under it, so the two cannot disagree
    const t = Math.abs(study.ic.t_stat), net = (study.portfolio.net || {}).sharpe;           // (a result from an older app: the same rule)
    if (Q.bad(t)) return "bad";
    return t >= 3 && !Q.bad(net) && net > 0.5 ? "good" : t >= 2 ? "mixed" : "bad";
  };
  const ICON = { good: "✓", mixed: "!", bad: "✗" };
  const HEAD = { good: "Strong in this sample", mixed: "Some evidence", bad: "No clear evidence" };
  const STAGE = { sanity: "Has enough data", fast_ic: "Predicts returns (training dates)", stability: "Holds up across years", quantiles: "Buckets rise in order", long_short: "Long-short pays", costs: "Still pays after costs", validation: "Holds on validation dates", locked_test: "Passes the locked test" };
  const stageWord = (s) => STAGE[s] || human(s);
  const WAY = (d) => (Q.bad(d) ? "" : Number(d) < 0 ? "lower is better" : "higher is better");           // the sign a formula is used with: the search learned it on the training dates

  /* ---------- shared drawing ---------- */
  /* What to bear in mind about a result: one notice, a list inside it when there are several, never the same line twice. */
  function watch(root, list) {
    const items = [...new Set(list || [])];
    if (!items.length) return;
    if (items.length === 1) { Q.labBanner(root, "Watch. ", items[0]); return; }
    const b = Q.labBanner(root, "Watch. ", `${items.length} things to bear in mind:`);
    const ul = h("ul", "plain", b); for (const t of items) h("li", "", ul, t);
  }
  function lineCard(parent, title, note, dates, series, fmtY, extra) {
    const o = extra || {};
    return chartCard(parent, title, note, { wide: !!o.wide,
      draw: (b) => C.line(b, { dates, series, yFormat: fmtY, tipFormat: o.tip || fmtY, height: o.height || 250, label: title, legend: series.length > 1, baseline: o.baseline, include: o.include }),
      table: (b) => table(b, [{ label: "Date", get: (i) => dates[i] }].concat(series.map((x) => ({ label: x.name, num: true, get: (i) => (x.values[i] === null || x.values[i] === undefined ? "" : fmtY(x.values[i])) }))), W_sample(dates, o.every || 4)) });
  }
  const W_sample = (dates, every) => Q.sampleRows(dates, [], every);
  const pointsToSeries = (pts) => ({ dates: pts.map((p) => p[0]), values: pts.map((p) => p[1]) });
  const align = (dates, pts) => { const m = new Map(pts.map((p) => [p[0], p[1]])); return dates.map((d) => (m.has(d) ? m.get(d) : null)); };
  /* The table behind a view, for a spreadsheet: numbers go out as numbers (not as the rounded text on the page). */
  const csvButton = (parent, label, filename, header, rows) => {
    const row = h("div", "row tight", parent), b = h("button", "btn small", row, label); b.type = "button"; b.dataset.action = "download-csv";
    b.addEventListener("click", () => W.csvDownload(filename, [header].concat(rows)));
    return b;
  };

  /* The tear sheet of a study: the verdict in words, the numbers, the charts and the breakdowns. Used by the Factor lab, by combinations and by models. */
  FD.tearSheet = (root, d, opts) => {
    const o = opts || {}, level = levelOf(d), ic = d.ic, pf = d.portfolio, net = pf.net || {}, gross = pf.gross || {}, q = d.quantiles;
    const v = h("div", "card verdict-card " + level, root); v.dataset.level = level; v.dataset.role = "fund-verdict";
    const head = h("div", "vhead", v); h("span", "vicon", head, ICON[level]); h("h2", "", head, `${HEAD[level]}: ${d.label}`);
    h("p", "", v, d.verdict);
    for (const w of d.warnings) h("p", "hint", v, w);
    const k = h("div", "earnrow", root);
    const span = (ci, f) => (ci && ci.length === 2 && !Q.bad(ci[0]) && !Q.bad(ci[1]) ? `${f(ci[0])} to ${f(ci[1])}` : "");
    const icSpan = span(ic.ci95, sgn), shSpan = span(net.sharpe_ci95, (x) => num(x, 2));
    tile(k, "Mean rank IC", sgn(ic.mean_ic), { text: `t = ${num(ic.t_stat, 1)} (allowing for overlap)${icSpan ? ` · 95% interval ${icSpan}` : ""}`, dir: Math.abs(ic.t_stat) >= 2 ? "up" : "" }, "", "The average over dates of the rank correlation between the factor and the next period's return across companies. The interval is the range of true averages the data cannot rule out.");
    tile(k, "IC information ratio", num(ic.icir_annualised, 2), { text: `${pct(ic.positive_share, 0)} of dates positive` }, "", "Mean IC over its standard deviation, scaled to a year");
    tile(k, "Top minus bottom bucket", spct(q.spread_top_minus_bottom, 2), { text: `monotonic ${num((q.monotonicity || {}).spearman, 2)} · ${q.buckets} buckets` }, "", "Average forward return of the top bucket less the bottom bucket per period");
    tile(k, "Long-short Sharpe, net", num(net.sharpe, 2), { text: `gross ${num(gross.sharpe, 2)} · ${pct(net.cagr, 1)} a year${shSpan ? ` · 95% interval ${shSpan}` : ""}`, dir: net.sharpe > 0.5 ? "up" : "" }, "", "Top bucket long, bottom short, after trading costs and borrowing. The interval allows for skew and fat tails (Mertens).");
    tile(k, "Turnover a rebalance", pct(pf.avg_turnover, 0), { text: `cost drag ${num(pf.cost_drag_sharpe, 2)} Sharpe` }, "", "Share of the long book replaced at each rebalance");
    tile(k, "Dates and coverage", String(d.n_dates), { text: `${num(d.average_companies, 0)} companies a date` });
    const pw = d.power;
    if (pw && pw.available) {
      const card = W.card(root, "Could this study have seen a real effect?", pw.reading); card.dataset.role = "fund-power";
      table(card, [{ label: "A true mean IC of", num: true, get: (r) => num(r.ic, 2) }, { label: "would show a t of about", num: true, get: (r) => num(r.expected_t, 1) }, { label: "chance of |t| above 2", num: true, get: (r) => pct(r.power_t2, 0) }, { label: "chance of |t| above 3", num: true, get: (r) => pct(r.power_t3, 0) }], pw.table);
    }
    const fm = d.premium;
    if (fm) {
      const card = W.card(root, "The premium, holding other things fixed", fm.reading); card.dataset.role = "fund-premium";
      const a = fm.with_controls || {}, b = fm.without_controls || {}, g = fm.explained_by_controls || {}, rows = [];
      if (!Q.bad(a.premium) && !Q.bad(b.premium)) {
        rows.push([fm.controls.length ? `Holding ${fm.controls.join(", ")} fixed` : "On its own", a]);
        if (fm.controls.length) { rows.push(["On its own", b]); if (!Q.bad(g.premium)) rows.push(["What holding them fixed took away", g]); }
        table(card, [{ label: "Version", get: (r) => r[0] }, { label: "Premium over the window", num: true, get: (r) => spct(r[1].premium, 2) }, { label: "95% interval", num: true, get: (r) => span(r[1].ci95, (x) => spct(x, 2)) }, { label: "t", num: true, get: (r) => num(r[1].t_stat, 1) }, { label: "Dates", num: true, get: (r) => String(r[1].n_dates ?? "") }], rows);
      }
    }
    const ch = h("div", "charts", root), cs = d.charts, dates = cs.ic.map((p) => p[0]);
    if (fm && fm.series && fm.series.length) lineCard(ch, "The premium on each date", "The return, over the window, of one standard deviation more of the factor with the other things held fixed, from a regression across the companies on each date. A real premium is positive on most dates, not on a few.", fm.series.map((p) => p[0]), [{ name: "Premium", color: css("--s1"), values: fm.series.map((p) => p[1]) }], (x) => (x * 100).toFixed(1) + "%", { include: 0, baseline: 0 });
    lineCard(ch, "Information coefficient over time", "Each month's rank correlation between the factor and the next period's return. The line is the 12-period average: a factor that works earns a positive IC most of the time, not on a few dates.", dates,
      [{ name: "IC", color: css("--s1"), values: cs.ic.map((p) => p[1]) }, { name: "12-period average", color: css("--s2"), values: align(dates, cs.ic_rolling) }], (x) => x.toFixed(2), { wide: true, include: 0 });
    chartCard(ch, "How the IC is spread", "The distribution of the per-date IC. A factor with real information has the whole distribution shifted to the right of zero, not a few big positive dates.", {
      draw: (b) => C.histogram(b, { series: [{ name: "Rank IC", edges: cs.ic_hist.edges, counts: cs.ic_hist.counts, color: css("--s1") }], markers: [{ x: 0, label: "0" }], xFormat: (x) => x.toFixed(2), label: "Distribution of the rank IC", height: 220 }),
      table: (b) => table(b, [{ label: "From", num: true, get: (i) => num(cs.ic_hist.edges[i], 3) }, { label: "To", num: true, get: (i) => num(cs.ic_hist.edges[i + 1], 3) }, { label: "Dates", num: true, get: (i) => String(cs.ic_hist.counts[i]) }], cs.ic_hist.counts.map((_, i) => i)) });
    chartCard(ch, "Return by bucket", "Companies are sorted into buckets on each date; the bars are the average return of each bucket over the next period. Bars that rise from left to right are what a real factor looks like.", {
      draw: (b) => C.columns(b, { labels: cs.quantile_mean.map((r) => (r.bucket === 1 ? "1 low" : r.bucket === cs.quantile_mean.length ? `${r.bucket} high` : String(r.bucket))), values: cs.quantile_mean.map((r) => r.return), yFormat: (x) => (x * 100).toFixed(1) + "%", name: "Average return", label: "Return by bucket", height: 220, colors: () => css("--s1") }),
      table: (b) => table(b, [{ label: "Bucket", get: (r) => String(r.bucket) }, { label: "Average return", num: true, get: (r) => spct(r.return, 2) }], cs.quantile_mean) });
    const qd = Object.keys(cs.quantile_cumulative);
    if (qd.length) { const dts = cs.quantile_cumulative[qd[0]].map((p) => p[0]); lineCard(ch, "Growth of 1 by bucket", "Buy-and-rebalance each bucket. The lines should fan out in order: the top bucket highest, the bottom lowest.", dts, qd.map((key, i) => ({ name: key === "1" ? "1 (lowest)" : key === String(qd.length) ? `${key} (highest)` : key, color: Q.colorOf(i), values: cs.quantile_cumulative[key].map((p) => p[1]) })), (x) => x.toFixed(2), { baseline: 1 }); }
    lineCard(ch, "Long-short: growth of 1", "The top bucket bought and the bottom sold, rebalanced each period. Net is after trading costs and the cost of borrowing the short side.", cs.ls_cumulative_net.map((p) => p[0]),
      [{ name: "Net of costs", color: css("--s1"), values: cs.ls_cumulative_net.map((p) => p[1]) }, { name: "Gross", color: css("--s2"), values: cs.ls_cumulative_gross.map((p) => p[1]) }], (x) => x.toFixed(2), { baseline: 1 });
    lineCard(ch, "Long-short: fall from the peak", "How far below its best the net long-short portfolio has been.", cs.ls_drawdown.map((p) => p[0]), [{ name: "Drawdown", color: css("--neg"), values: cs.ls_drawdown.map((p) => p[1]) }], (x) => (x * 100).toFixed(0) + "%", {});
    chartCard(ch, "How fast the signal fades", "The mean IC against returns over longer and longer horizons. A signal that is still there at 3 or 6 months can be traded slowly and cheaply; one that fades in a week must be traded fast.", {
      draw: (b) => C.columns(b, { labels: cs.decay.map((r) => r.horizon), values: cs.decay.map((r) => r.mean_ic), yFormat: (x) => x.toFixed(3), name: "Mean IC", label: "Mean IC by horizon", height: 220, colors: () => css("--s3") }),
      table: (b) => table(b, [{ label: "Horizon", get: (r) => r.horizon }, { label: "Mean IC", num: true, get: (r) => sgn(r.mean_ic) }, { label: "t", num: true, get: (r) => num(r.t_stat, 1) }, { label: "Dates", num: true, get: (r) => String(r.n_dates) }], cs.decay) });
    lineCard(ch, "Turnover", "The share of the long book that changes at each rebalance.", cs.turnover.map((p) => p[0]), [{ name: "Turnover", color: css("--s4"), values: cs.turnover.map((p) => p[1]) }], (x) => (x * 100).toFixed(0) + "%", {});
    lineCard(ch, "How many companies have a value", "The share of investable companies for which the factor can be computed on each date. A factor that covers few companies is a statement about those companies only.", cs.coverage.map((p) => p[0]), [{ name: "Coverage", color: css("--s5"), values: cs.coverage.map((p) => p[1]) }], (x) => (x * 100).toFixed(0) + "%", { include: 0 });
    const st = d.stability, yearly = Object.values(st.by_year);
    chartCard(ch, "IC by year", `A factor that works in only one year of ten is a story about that year. Positive in ${yearly.filter((v) => v > 0).length} of ${yearly.length} years; the two halves of the sample ${st.halves_agree ? "agree" : "disagree"} in sign (${sgn(st.first_half_ic)} then ${sgn(st.second_half_ic)}).`, {
      draw: (b) => C.columns(b, { labels: Object.keys(st.by_year), values: Object.values(st.by_year), yFormat: (x) => x.toFixed(3), name: "Mean IC", label: "Mean IC by year", height: 220, colors: (val) => (val >= 0 ? css("--s1") : css("--neg")) }),
      table: (b) => table(b, [{ label: "Year", get: (y) => y }, { label: "Mean IC", num: true, get: (y) => sgn(st.by_year[y]) }], Object.keys(st.by_year)) });
    const bd = d.breakdown || {};
    for (const [key, title] of [["country", "By country"], ["sector", "By sector"], ["region", "By region"]]) {
      const rows = Object.entries(bd[key] || {}); if (!rows.length) continue;
      const card = W.card(root, title, "The same study run inside each group. Where the factor works in one group and not another, the average hides it; a group with few companies says little.");
      card.dataset.role = "fund-breakdown-" + key;
      table(card, [{ label: human(key), get: (r) => r[0] }, { label: "Companies", num: true, get: (r) => String(r[1].companies ?? "") }, { label: "Mean IC", num: true, get: (r) => sgn(r[1].mean_ic) }, { label: "t", num: true, get: (r) => num(r[1].t_stat, 1) }, { label: "Dates positive", num: true, get: (r) => pct(r[1].positive_share, 0) }], rows);
    }
    if (bd.regime) {
      const card = W.card(root, "By market regime", "How the factor did when the market was rising or falling, and when volatility was high or low (each known at the start of the period).");
      card.dataset.role = "fund-breakdown-regime";
      const rows = []; for (const [kind, parts] of Object.entries(bd.regime)) for (const [name, r] of Object.entries(parts || {})) rows.push([`${human(kind)}: ${name}`, r]);
      table(card, [{ label: "Regime", get: (r) => r[0] }, { label: "Dates", num: true, get: (r) => String(r[1].n_dates ?? "") }, { label: "Mean IC", num: true, get: (r) => sgn(r[1].mean_ic) }, { label: "t", num: true, get: (r) => num(r[1].t_stat, 1) }], rows);
    }
    const det = h("details", "how", root); h("summary", "", det, "How this study was made");
    const ul = h("ul", "", det);
    h("li", "", ul, `Universe: ${d.universe.companies} companies, ${d.universe.dates} ${d.universe.freq === "M" ? "monthly" : d.universe.freq === "W" ? "weekly" : "quarterly"} dates from ${d.universe.first} to ${d.universe.last}; signals are traded ${d.universe.entry_lag_days} day(s) after the date they are computed.`);
    h("li", "", ul, `Survivorship: ${d.universe.survivorship.verdict}.`);
    h("li", "", ul, `Point in time: a figure is used from ${d.universe.pit.info_lag_days} day(s) after the day it was public; ${d.universe.pit_estimated_share === null ? "all" : pct(d.universe.pit_estimated_share, 1)} of the periods needed an estimated filing date.`);
    h("li", "", ul, `Steps: ${(d.spec.steps || []).map((s) => s.op + (s.pct ? ` ${s.pct}%` : "") + (s.by ? ` (${[].concat(s.by, s.continuous || []).join(", ")})` : "")).join(" → ") || "raw values"}; direction ${d.spec.direction}; horizon ${d.spec.horizon}; ${d.spec.quantiles} buckets; costs ${d.spec.cost_bps} bps one way, borrow ${d.spec.borrow_bps} bps a year.`);
    if (fm) h("li", "", ul, `Premium: on each date the returns (cut at the 1st and 99th percentile of the date) are regressed on the normal score of the rank of the factor${fm.controls.length ? ` and of ${fm.controls.join(", ")}` : ""}; the mean of the coefficients has a Newey-West error that allows for overlapping windows (Fama and MacBeth, 1973).`);
    h("li", "", ul, "Interesting is not validated: a result in a sample that was looked at is a lead. Test it on dates it was not chosen on, count how many ideas were tried, and check it after costs.");
    if (o.actions) o.actions(root, d);
  };

  /* =========================================================== Factor lab */
  const FL = FD.views.factors = {};
  FL.s = { source: "factor", factor: "fcf_yield", formula: "div(fcf,market_cap)", scale: "zscore", winsor: "1", neutral: new Set(), controls: new Set(), horizon: "1m", direction: "theory", quantiles: "5", top: "", weighting: "equal",
           cost: "10", borrow: "50", min_assets: "20", fixed: new Set(["size", "beta", "volatility", "momentum"]), extra_fixed: "" };
  FL.steps = () => {
    const s = FL.s, steps = [];
    if (s.scale === "raw") return steps;
    if (s.winsor !== "none") steps.push({ op: "winsorize", pct: Number(s.winsor) });
    if (s.neutral.size || s.controls.size) steps.push({ op: "neutralize", by: [...s.neutral], continuous: [...s.controls] });
    steps.push(s.scale === "rank" ? { op: "rank" } : s.scale === "percentile" ? { op: "percentile" } : { op: "zscore" });
    return steps;
  };
  FL.spec = () => {
    const s = FL.s, spec = { steps: FL.steps(), horizon: s.horizon, direction: s.direction === "theory" || s.direction === "auto" ? s.direction : Number(s.direction), quantiles: Number(s.quantiles), weighting: s.weighting, cost_bps: Number(s.cost), borrow_bps: Number(s.borrow), min_assets: Number(s.min_assets) };
    if (s.source === "factor") spec.factor = s.factor;
    if (s.top !== "") spec.top_fraction = Number(s.top) / 100;
    for (const k of ["quantiles", "cost_bps", "borrow_bps", "min_assets"]) if (!isFinite(spec[k])) throw new Error("The test settings must be numbers");
    spec.premium = true;
    spec.controls = [...s.fixed].concat(s.extra_fixed.split(/[\s,;]+/).map((x) => x.trim()).filter(Boolean));
    return spec;
  };
  FL.check = () => {
    const s = FL.s, none = FD.needsData();
    if (none) return none;
    if (s.source === "factor" && !s.factor) return "Choose a factor.";
    if (s.source === "formula" && !s.formula.trim()) return "Write a formula.";
    return "";
  };
  FL.sections = (panel) => {
    const s = FL.s, after = FD.after;
    FD.universeSection(panel, "1. Which companies and when");
    const b = FD.sec(panel, "2. The factor");
    W.seg(b, [["factor", "A library factor"], ["formula", "My own formula"]], s.source, (v) => { s.source = v; Q.renderLab(FD); }).dataset.role = "fund-source";
    if (s.source === "factor") {
      const holder = h("div", "", b);
      FD.catalogue().then((cat) => {
        if (!holder.isConnected) return;
        const label = h("label", "f", holder, "Factor"), sel = h("select", "", label); sel.dataset.role = "fund-factor"; sel.setAttribute("aria-label", "factor");
        for (const fam of cat.families) {
          const list = cat.factors.filter((f) => f.family === fam); if (!list.length) continue;
          const g = h("optgroup", "", sel); g.label = human(fam);
          for (const f of list) { const o = h("option", "", g, f.label); o.value = f.name; if (f.name === s.factor) o.selected = true; }
        }
        const note = h("p", "hint", holder, ""), paint = () => { const f = cat.factors.find((x) => x.name === s.factor); note.textContent = f ? `${f.text} Expected direction: ${f.direction > 0 ? "higher is better" : "lower is better"}. Reads: ${f.needs.join(", ") || "prices only"}.` : ""; };
        sel.addEventListener("change", () => { s.factor = sel.value; paint(); after(); }); paint();
      }).catch((e) => h("p", "hint", holder, e.message));
    } else {
      Q.labField(b, { key: "formula", label: "Formula", kind: "text", placeholder: "div(fcf,market_cap)" }, s, after);
      const out = h("p", "hint", b, ""); out.dataset.role = "fund-formula-check";
      const row = h("div", "row tight", b), chk = h("button", "btn small", row, "Check the formula"); chk.type = "button"; chk.dataset.action = "check-formula";
      chk.addEventListener("click", async () => {
        try { const r = await FD.send("/api/fund/formula/check", Object.assign({ formula: s.formula }, FD.universeBody())); out.textContent = r.ok ? `Reads ${r.fields.join(", ")}${r.coverage !== undefined ? `; covers ${pct(r.coverage, 0)} of the companies that could be bought` : ""}. Written canonically: ${r.canonical}.` : r.error; out.className = "hint " + (r.ok ? "tone-good" : "tone-crit"); }
        catch (e) { out.textContent = e.message; out.className = "hint tone-crit"; }
      });
      for (const ex of ["div(fcf,market_cap)", "delta(roic,12)", "neg(growth(total_assets,12))", "sector_rank(div(operating_income,total_assets))"]) { const x = h("button", "link", row, ex); x.type = "button"; x.addEventListener("click", () => { s.formula = ex; Q.renderLab(FD); }); }
      h("p", "hint", b, "Operators: add sub mul div, neg abs log, delta growth accel rolling_mean rolling_std over a number of periods, rank sector_rank industry_rank zscore. Fields are statement figures, derived ratios, market_cap, enterprise_value and the price factors. A rank, z-score or sign on the outside does not change which companies it prefers.");
    }
    const t = FD.sec(panel, "3. Transformations");
    Q.labField(t, { key: "winsor", label: "Cut the extremes of each date at", kind: "choice", options: [["none", "nothing"], ["0.5", "0.5% each tail"], ["1", "1% each tail"], ["2.5", "2.5% each tail"], ["5", "5% each tail"]] }, s, after);
    Q.labField(t, { key: "scale", label: "Put the companies on a common scale by", kind: "choice", options: [["zscore", "z-score"], ["rank", "rank"], ["percentile", "percentile"], ["raw", "nothing (raw values)"]] }, s, after);
    Q.checkList(t, "Remove what the factor shares with (neutralise):", [["sector", "sector"], ["industry", "industry"], ["country", "country"], ["market", "market"]], s.neutral, after);
    Q.checkList(t, "...and with these measures:", [["mcap", "size (market value)"], ["beta", "market beta"], ["volatility", "volatility"]], s.controls, after);
    h("p", "hint", t, "Each step uses only the companies of its own date, so none can look ahead. Neutralising asks whether the factor still works once a sector or size bet is removed.");
    const e = FD.sec(panel, "4. How it is tested");
    Q.labField(e, { key: "horizon", label: "Return measured over the next", kind: "choice", options: [["1w", "week"], ["2w", "2 weeks"], ["1m", "month"], ["2m", "2 months"], ["3m", "3 months"], ["6m", "6 months"], ["12m", "year"]] }, s, after);
    Q.labField(e, { key: "direction", label: "Which way is good", kind: "choice", options: [["theory", "the factor's own (from theory)"], ["1", "higher is better"], ["-1", "lower is better"], ["auto", "whichever worked (counts as a second trial)"]] }, s, after);
    Q.labField(e, { key: "quantiles", label: "Buckets", kind: "number", step: 1, min: 2, max: 10 }, s, after);
    Q.labField(e, { key: "top", label: "Long and short legs: top and bottom % (blank = one bucket)", kind: "number", min: 2, max: 50 }, s, after);
    Q.labField(e, { key: "weighting", label: "Weights inside a leg", kind: "choice", options: [["equal", "equal"], ["signal", "by signal strength"], ["inv_vol", "by inverse volatility"]] }, s, after);
    Q.labField(e, { key: "cost", label: "Trading cost, one way (bps)", kind: "number", min: 0, max: 500 }, s, after);
    Q.labField(e, { key: "borrow", label: "Cost of borrowing the short side (bps a year)", kind: "number", min: 0, max: 2000 }, s, after);
    Q.labField(e, { key: "min_assets", label: "Fewest companies on a date", kind: "number", step: 1, min: 5, max: 2000 }, s, after);
    Q.checkList(e, "The premium regression holds these fixed:", [["size", "size (market value)"], ["beta", "market beta"], ["volatility", "volatility"], ["momentum", "momentum (twelve months less the last)"]], s.fixed, after);
    Q.labField(e, { key: "extra_fixed", label: "...and these library factors (names, separated by commas)", kind: "text", placeholder: "roic, gross_margin", help: "A premium that vanishes once something it is known to resemble is held fixed was that thing in disguise." }, s, after);
    const more = h("div", "row tight", panel); more.style.padding = "0 16px";
    const scan = h("button", "btn", more, "Scan every factor"); scan.type = "button"; scan.dataset.action = "scan-factors"; scan.addEventListener("click", () => FL.scan());
    const audit = h("button", "btn", more, "Look-ahead audit"); audit.type = "button"; audit.dataset.action = "audit-factor"; audit.addEventListener("click", () => FL.audit());
    audit.title = "Run the study twice: once as it should be (a figure is used from the day it was public) and once as if every figure were known at the end of its period. A big difference means the result leans on knowing things early.";
  };
  FL.body = () => Object.assign({ spec: FL.spec() }, FD.universeBody());
  FL.run = () => FD.act("factors", "Studying…", async (run) => {
    const s = FL.s, body = FL.body();
    if (s.source === "formula") { body.formula = s.formula; const { result } = await run.job("/api/fund/evaluate/formula", body, "Studying the formula"); return { kind: "study", study: result, source: "formula", formula: s.formula }; }
    const { result } = await run.job("/api/fund/evaluate", body, "Studying " + s.factor);
    return { kind: "study", study: result, source: "factor", factor: s.factor };
  });
  FL.scan = () => FD.act("factors", "Scanning the factors…", async (run) => {
    const body = FL.body(); delete body.spec.factor;
    const { result } = await run.job("/api/fund/scan", body, "Scanning every factor");
    return { kind: "scan", scan: result };
  });
  FL.audit = () => FD.act("factors", "Auditing for look-ahead…", async (run) => {
    const s = FL.s; if (s.source !== "factor") throw new Error("The look-ahead audit is for library factors: choose one in section 2");
    const body = FL.body(); const { result } = await run.job("/api/fund/lookahead", body, "Running the study twice");
    const was = FD.results.factors && FD.results.factors.data && FD.results.factors.data.kind === "study" ? FD.results.factors.data : null;
    return Object.assign({}, was || { kind: "audit" }, { audit: result, kind: was ? "study" : "audit" });
  });
  FL.empty = (root) => {
    const c = h("div", "card empty", root); h("h2", "", c, "Does a factor predict what happens next?");
    for (const t of ["Choose a factor (or write a formula), a universe and how it is to be tested, and press the button. The study sorts the companies by the factor on every date and looks at what each group earned afterwards.",
      "Then it asks the questions that matter: is the relation steady or the work of a few dates, does it survive costs and borrowing, does it hold in every country and sector, and how fast does it fade?",
      "A result here is a lead, not a finding: it was found in the sample it is measured on. Alpha discovery counts every idea tried and tests the best on dates they never saw."]) h("p", "", c, t);
    const go = h("button", "btn primary", c, "Study this factor"); go.type = "button"; go.addEventListener("click", () => FL.run());
    const sc = h("button", "btn", c, "Scan every factor"); sc.type = "button"; sc.style.marginLeft = "10px"; sc.addEventListener("click", () => FL.scan());
  };
  FL.build = (root, data) => {
    if (data.kind === "scan") { FL.scanView(root, data.scan); return; }
    if (data.kind === "audit") { auditView(root, data.audit); return; }
    const d = data.study;
    Q.labHead(root, `Factor study: ${d.label}`, `${d.universe.companies} companies · ${d.n_dates} dates · ${d.universe.first} to ${d.universe.last}`, GUIDE);
    if (data.audit) auditView(root, data.audit);
    FD.tearSheet(root, d, { actions: (r, study) => {
      const row = h("div", "row tight", r);
      const bt = h("button", "btn primary", row, "Backtest this alpha"); bt.type = "button"; bt.dataset.action = "backtest-this"; bt.addEventListener("click", () => FD.views.libraryUse(data.source === "formula" ? [{ kind: "formula", formula: data.formula }] : [{ kind: "factor", name: data.factor }], "backtest"));
      const dl = h("button", "btn", row, "Download the IC series (CSV)"); dl.type = "button"; dl.addEventListener("click", () => W.csvDownload(`${study.label}_ic.csv`.replace(/[^\w.\-]+/g, "_"), [["date", "rank_ic"]].concat(study.charts.ic)));
    } });
  };
  function auditView(root, a) {
    const card = h("div", "card verdict-card " + (a.flattered ? "bad" : "good"), root); card.dataset.role = "fund-audit";
    const head = h("div", "vhead", card); h("span", "vicon", head, a.flattered ? "✗" : "✓"); h("h2", "", head, a.flattered ? "Knowing figures early flatters this factor" : "The result does not lean on early knowledge");
    h("p", "", card, a.verdict);
    table(card, [{ label: "Version", get: (r) => r[0] }, { label: "Mean IC", num: true, get: (r) => sgn(r[1].mean_ic) }, { label: "t", num: true, get: (r) => num(r[1].t_stat, 1) }, { label: "Dates", num: true, get: (r) => String(r[1].n_dates) }],
      [["Honest: each figure from the day it was public", a.honest], ["Naive: each figure from its period end", a.naive]]);
    if (a.by_lag && a.by_lag.length) {
      h("h3", "subhead", card, "What is left if a figure is used only some days after its filing");
      h("p", "", card, a.lag_verdict);
      table(card, [{ label: "Days after the filing", num: true, get: (r) => String(r.lag_days) }, { label: "Mean IC", num: true, get: (r) => sgn(r.mean_ic) }, { label: "t", num: true, get: (r) => num(r.t_stat, 1) }, { label: "Dates", num: true, get: (r) => String(r.n_dates) }], a.by_lag);
    }
  }
  FL.scanView = (root, sc) => {
    Q.labHead(root, "Every factor, side by side", `${sc.tested} factors under one set of conditions · ${sc.universe.companies} companies · ${sc.universe.dates} dates`, GUIDE);
    const warn = h("div", "banner", root); h("strong", "", warn, "Many factors looked at. "); warn.appendChild(document.createTextNode(sc.note));
    const rows = sc.rows.filter((r) => !r.error);
    chartCard(root, "Which factors stand out", "The t-statistic of each factor's mean IC, allowing for the overlap of the return windows. A bar beyond ±2 is about the most luck alone gives, once in twenty tries; beyond ±3 is worth a closer look.", {
      wide: true,
      draw: (b) => C.hbars(b, { rows: rows.slice(0, 24).map((r) => ({ label: r.label, value: r.t_stat, text: `t ${num(r.t_stat, 1)}`, tip: [["Mean IC", sgn(r.mean_ic)], ["Net Sharpe", num(r.net_sharpe, 2)], ["Coverage", pct(r.coverage, 0)]] })), format: (v) => v.toFixed(0), name: "t-statistic", label: "t-statistics of the factors", labelWidth: 200, valueWidth: 60 }),
      table: (b) => table(b, [{ label: "Factor", get: (r) => r.label }, { label: "t", num: true, get: (r) => num(r.t_stat, 1) }], rows) });
    const card = W.card(root, "The table", "Click a factor to study it in full."); card.dataset.role = "fund-scan";
    csvButton(card, "Download the table (CSV)", "factor_scan.csv", ["factor", "family", "coverage", "mean_ic", "t_stat", "icir", "positive_share", "top_minus_bottom", "monotonicity", "net_sharpe", "turnover"],
      sc.rows.filter((r) => !r.error).map((r) => [r.name, r.family, r.coverage, r.mean_ic, r.t_stat, r.icir, r.positive_share, r.spread, r.monotonicity, r.net_sharpe, r.turnover]));
    const wrap = FD.guardScroll(h("div", "scroll tall", card)), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t));
    for (const [label, numeric] of [["Factor", 0], ["Family", 0], ["Coverage", 1], ["Mean IC", 1], ["t", 1], ["ICIR", 1], ["Dates positive", 1], ["Top minus bottom", 1], ["Monotonic", 1], ["Net Sharpe", 1], ["Turnover", 1], ["Study", 2]]) { const th = h("th", numeric === 1 ? "num" : "", hd); if (numeric === 2) h("span", "vh", th, label); else th.textContent = label; }
    const tb = h("tbody", "", t);
    for (const r of sc.rows) {
      const tr = h("tr", "", tb); tr.dataset.factor = r.name;
      h("td", "", tr, r.label); h("td", "", tr, human(r.family));
      if (r.error) { const td = h("td", "", tr, r.error); td.colSpan = 9; } else {
        h("td", "num", tr, pct(r.coverage, 0)); h("td", "num", tr, sgn(r.mean_ic)); h("td", "num", tr, num(r.t_stat, 1)); h("td", "num", tr, num(r.icir, 2)); h("td", "num", tr, pct(r.positive_share, 0));
        h("td", "num", tr, spct(r.spread, 2)); h("td", "num", tr, num(r.monotonicity, 2)); h("td", "num", tr, num(r.net_sharpe, 2)); h("td", "num", tr, pct(r.turnover, 0));
        const open = h("button", "btn small", h("td", "", tr), "Study"); open.type = "button"; open.dataset.action = "study-factor"; open.addEventListener("click", () => { FL.s.source = "factor"; FL.s.factor = r.name; Q.renderLab(FD); FL.run(); });
      }
    }
  };

  /* =========================================================== Alpha discovery */
  const DS = FD.views.discovery = {};
  DS.s = { n: "300", seed: "7", horizon: "1m", max_nodes: "15", finalists: "15", train: "0.55", val: "0.20", cost: "10", borrow: "50", min_assets: "20", quantiles: "5", seed_mode: "classics", seeds: "", thr: {}, strictness: "standard", repeats: "3" };
  DS.runs = null; DS.runId = null;
  DS.spec = () => {
    const s = DS.s, u = FD.universeBody(), spec = { universe: u.universe, freq: u.freq, entry_lag: u.entry_lag, n_candidates: Number(s.n), seed: Number(s.seed), horizon: s.horizon, max_nodes: Number(s.max_nodes), finalists: Number(s.finalists),
      train_fraction: Number(s.train), validation_fraction: Number(s.val), cost_bps: Number(s.cost), borrow_bps: Number(s.borrow), min_assets: Number(s.min_assets), quantiles: Number(s.quantiles), strictness: s.strictness };
    for (const k of ["n_candidates", "seed", "max_nodes", "finalists", "train_fraction", "validation_fraction", "cost_bps", "borrow_bps", "min_assets", "quantiles"]) if (!isFinite(spec[k])) throw new Error("The search settings must be numbers");
    const list = s.seeds.split(/\n+/).map((x) => x.trim()).filter(Boolean);
    if (s.seed_mode === "mine") { if (!list.length) throw new Error("Write at least one formula to start from, or choose another starting point"); spec.seeds = list; }
    else if (s.seed_mode === "none") spec.seeds = [];
    const thr = {}; for (const [k, v] of Object.entries(s.thr)) if (v !== "") thr[k] = Number(v);
    if (Object.keys(thr).length) spec.thresholds = thr;
    return spec;
  };
  DS.sections = (panel) => {
    const s = DS.s, after = FD.after;
    FD.universeSection(panel, "1. Which companies", true);
    const a = FD.sec(panel, "2. The search");
    Q.labField(a, { key: "n", label: "Formulas to try", kind: "number", step: 1, min: 1, max: 20000, help: "Every formula tried counts against the result, whether or not it looked good." }, s, after);
    Q.labField(a, { key: "horizon", label: "Return measured over the next", kind: "choice", options: [["1w", "week"], ["1m", "month"], ["2m", "2 months"], ["3m", "3 months"], ["6m", "6 months"], ["12m", "year"]] }, s, after);
    Q.labField(a, { key: "max_nodes", label: "Largest formula (parts)", kind: "number", step: 1, min: 3, max: 31 }, s, after);
    Q.labField(a, { key: "finalists", label: "Finalists given the one look at the locked dates", kind: "number", step: 1, min: 1, max: 200 }, s, after);
    Q.labField(a, { key: "seed", label: "Random seed", kind: "number", step: 1, min: 0 }, s, after);
    Q.labField(a, { key: "seed_mode", label: "Start from", kind: "choice", options: [["classics", "the classic formulas of the literature, then random ones"], ["mine", "only my formulas, then random ones"], ["none", "random formulas only"]] }, s, () => Q.renderLab(FD));
    if (s.seed_mode === "mine") { const area = h("textarea", "", h("label", "f", a, "My formulas, one a line")); area.rows = 4; area.value = s.seeds; area.spellcheck = false; area.dataset.role = "fund-seeds"; area.addEventListener("input", () => { s.seeds = area.value; after(); }); }
    const b = FD.sec(panel, "3. The dates it is judged on");
    Q.labField(b, { key: "train", label: "Training share of the dates", kind: "number", step: 0.05, min: 0.3, max: 0.8 }, s, after);
    Q.labField(b, { key: "val", label: "Validation share", kind: "number", step: 0.05, min: 0.1, max: 0.4 }, s, after);
    h("p", "hint", b, "The rest is locked: the best formulas are tested on it once, and the number of looks is recorded. A gap as wide as the forecast horizon separates the blocks so no return window sits in two of them. The sign of a formula is learned on the training dates only.");
    const c = FD.sec(panel, "4. How strict, and costs", true);
    Q.labField(c, { key: "strictness", label: "How strict", kind: "choice", options: [["exploratory", "Exploratory: lets more through to be looked at"], ["standard", "Standard: |t| of 2, false-discovery rate 10%"], ["strict", "Strict: |t| of 3, false-discovery rate 5%"]],
      help: "The bars a formula must clear at every stage. A stricter search finds less of what is real and lets less luck through; the preflight says how much, and the calibration on shuffled returns measures what luck gets through on this data." }, s, () => Q.renderLab(FD));
    Q.labField(c, { key: "repeats", label: "Shuffles for the calibration (1 to 5)", kind: "number", step: 1, min: 1, max: 5 }, s, after);
    Q.labField(c, { key: "cost", label: "Trading cost, one way (bps)", kind: "number", min: 0, max: 500 }, s, after);
    Q.labField(c, { key: "borrow", label: "Borrowing cost (bps a year)", kind: "number", min: 0, max: 2000 }, s, after);
    Q.labField(c, { key: "quantiles", label: "Buckets", kind: "number", step: 1, min: 2, max: 10 }, s, after);
    Q.labField(c, { key: "min_assets", label: "Fewest companies on a date", kind: "number", step: 1, min: 5, max: 2000 }, s, after);
    FD.catalogue().then((cat) => {
      if (!c.isConnected) return;
      const d = h("details", "how", c); h("summary", "", d, "Thresholds each stage asks for");
      const preset = (cat.discovery.presets || {})[s.strictness] || cat.discovery.thresholds;
      h("p", "hint", d, `Blank means the ${s.strictness} setting's own bar (shown in grey). A bar typed here replaces it.`);
      for (const [k, v] of Object.entries(cat.discovery.thresholds)) { const spec = { key: k, label: human(k), kind: "number", step: "any", placeholder: String(preset[k] ?? v) }; Q.labField(d, spec, s.thr, after); if (s.thr[k] === undefined) s.thr[k] = ""; }
    }).catch(() => {});
    const more = h("div", "row tight", panel); more.style.padding = "0 16px";
    const pre = h("button", "btn", more, "What could this search find?"); pre.type = "button"; pre.dataset.action = "preflight"; pre.addEventListener("click", () => DS.preflight());
    pre.title = "Before spending the afternoon: the smallest effect this many companies and dates can tell from luck, and the chance that a real alpha of each strength would get through. Nothing is run or written.";
    const cal = h("button", "btn", more, "Calibrate on shuffled returns"); cal.type = "button"; cal.dataset.action = "placebo"; cal.addEventListener("click", () => DS.placebo());
    cal.title = "Run this search again where the companies' returns are dealt out at random, so nothing can be found, and see what it finds anyway. Takes a few times as long as the search itself. Nothing is written to the library.";
  };
  DS.check = () => FD.needsData();
  DS.run = () => FD.act("discovery", "Searching…", async (run) => {
    const { result, started } = await run.job("/api/fund/discover", { spec: DS.spec() }, "Searching");
    DS.runId = started.run_id; await DS.refreshRuns(); LB.stale(); return { summary: result };
  });
  DS.preflight = () => FD.act("discovery", "Working out what the search could find…", async (run) => {
    const { result } = await run.job("/api/fund/discover/preflight", { spec: DS.spec() }, "Working out what the search could find");
    return { pre: result };
  });
  DS.placebo = (runId) => FD.act("discovery", "Searching shuffled returns…", async (run) => {
    const repeats = Number(DS.s.repeats); if (!isFinite(repeats)) throw new Error("The number of shuffles must be a number");
    const was = FD.results.discovery && FD.results.discovery.data ? FD.results.discovery.data : null;
    const { result } = await run.job("/api/fund/placebo", runId ? { run_id: runId, repeats } : { spec: DS.spec(), repeats }, "Searching shuffled returns");
    return runId && was && was.summary ? { summary: was.summary, placebo: result } : { placebo: result };
  });
  DS.resume = (id) => FD.act("discovery", "Resuming…", async (run) => { const { result } = await run.job("/api/fund/discover", { run_id: id }, "Resuming"); DS.runId = id; await DS.refreshRuns(); LB.stale(); return { summary: result }; });
  DS.open = (id) => FD.act("discovery", "Opening the run…", async () => { DS.runId = id; return { summary: await FD.read("/api/fund/run", { id, limit: 200 }) }; });
  DS.refreshRuns = async () => { try { DS.runs = (await FD.read("/api/fund/runs")).runs; } catch (e) { DS.runs = []; } };
  DS.loadRuns = async () => { await DS.refreshRuns(); if (isOpen() && FD.mode === "discovery" && !FD.busy) paintRuns(); };
  DS.auto = () => { DS.loadRuns(); };
  FD.views.watch = () => { if (FD.mode === "discovery") DS.loadRuns(); };
  function paintRuns() {
    const root = Q.$("#fundamentals-results"); if (!root) return;
    const old = root.querySelector(":scope > .runs"); if (old) old.remove();
    const card = runsCard(); if (card) root.appendChild(card);
  }
  function runsCard() {
    if (!DS.runs || !DS.runs.length) return null;
    const card = document.createElement("div"); card.className = "card runs"; card.dataset.role = "fund-runs";
    h("h3", "", card, "Earlier searches");
    const ul = h("ul", "runlist", card);
    for (const r of DS.runs.slice(0, 10)) {
      const li = h("li", r.status, ul); li.dataset.run = r.run_id;
      const main = h("div", "main", li), head = h("span", "", main); h("b", "", head, (r.status === "finished" ? "Finished" : r.status === "stopped" ? "Stopped" : r.status === "running" ? "Interrupted" : human(r.status)) + " · "); head.appendChild(document.createTextNode(`${(r.stage_counts || {}).next_index || 0} formulas, ${(r.stage_counts || {}).tested || 0} tested`));
      h("span", "meta", main, `${r.run_id} · started ${day(r.created_at)}${r.error ? " · " + r.error : ""}`);
      const open = h("button", "btn small", li, "Open"); open.type = "button"; open.dataset.action = "open-run"; open.addEventListener("click", () => DS.open(r.run_id));
      if (r.status !== "finished") { const res = h("button", "btn small", li, "Resume"); res.type = "button"; res.dataset.action = "resume-run"; res.title = "Carry on exactly where it stopped; nothing already tried is tried again"; res.addEventListener("click", () => DS.resume(r.run_id)); }
    }
    return card;
  }
  DS.empty = (root) => {
    const c = h("div", "card empty", root); h("h2", "", c, "Let the data suggest the formulas");
    for (const t of ["The search builds formulas from the statements (ratios, changes, growth, ranks within a sector) and tests each one in stages: does it have enough data, does it predict returns on the training dates, is it steady across years, do the buckets rise in order, does a long-short portfolio pay, does it still pay after costs.",
      "Every formula tried is counted and remembered, including the ones that fail: a result is only as good as the search that found it, and the best of a thousand tries looks good by luck. The survivors are corrected for the number tried, then tested once on dates they never saw.",
      "What comes out is called a Candidate, never validated. Validation needs the evidence to hold and time to pass."]) h("p", "", c, t);
    const go = h("button", "btn primary", c, "Start the search"); go.type = "button"; go.addEventListener("click", () => DS.run());
    const card = runsCard(); if (card) root.appendChild(card);
  };
  DS.build = (root, data) => {
    if (data.pre) DS.preflightView(root, data.pre);
    if (data.summary) { DS.summaryView(root, data.summary, data.placebo ? (r) => DS.placeboView(r, data.placebo, false) : null); return; }
    if (data.placebo) DS.placeboView(root, data.placebo, true);
    if (data.pre || data.placebo) { const runs = runsCard(); if (runs) root.appendChild(runs); }
  };
  DS.summaryView = (root, d, after) => {
    const c = d.counts || {};
    Q.labHead(root, `Discovery run ${d.run_id}`, `${d.status} · ${d.progress.done} of ${d.progress.total} formulas · ${d.strictness || "exploratory"} bars`, GUIDE);
    const k = h("div", "earnrow", root);
    tile(k, "Formulas tried", String(c.generated ?? d.progress.done), { text: `${c.skipped_seen || 0} already known · ${c.duplicate_idea || 0} the same idea twice` });
    tile(k, "Tested in full", String(c.tested || 0), { text: c.n_tested_family ? `${c.n_tested_family} ideas counted against the result, from ${c.runs_in_family} run${c.runs_in_family === 1 ? "" : "s"} on this question` : "every one counts against the result" }, "", "The correction for the number tried covers every idea tried on the same question, in this run and in earlier ones: the more the same data is searched, the better the best result looks by luck alone.");
    tile(k, "Survived the stages", String(c.survivors ?? 0), { text: `${c.finalists ?? 0} finalists` });
    tile(k, "Candidates", String(c.candidates ?? 0), { text: `${c.failed_locked_test ?? 0} failed the locked test` }, "", "Passed every stage, including the locked dates. Candidates are not validated.");
    tile(k, "Chance of overfitting", c.pbo === null || c.pbo === undefined ? "n/a" : pct(c.pbo, 0), { text: "probability of backtest overfitting among the survivors" }, "", "The share of splits of the history on which the best formula in half the data was below average in the other half");
    tile(k, "Looks at the locked dates", String(c.oos_looks ?? 0), { text: "each look spends some of its value" });
    chartCard(root, "Where the formulas fell", "How many formulas passed each stage. Most should fall early: a funnel that stays wide means the bars are too low.", {
      draw: (b) => C.hbars(b, { rows: d.funnel.map((r) => ({ label: stageWord(r.stage), value: r.passed })), format: (v) => String(Math.round(v)), name: "Passed", label: "Formulas passing each stage", labelWidth: 230, valueWidth: 50, color: () => css("--s1") }),
      table: (b) => table(b, [{ label: "Stage", get: (r) => stageWord(r.stage) }, { label: "Passed", num: true, get: (r) => String(r.passed) }], d.funnel) });
    const board = W.card(root, "The best so far", "Ranked by fitness (information coefficient, stability, Sharpe, monotonicity and decorrelation, less turnover and complexity). q is the false-discovery rate over everything tried; DSR is the deflated Sharpe probability. Nothing here is validated.");
    board.dataset.role = "fund-leaderboard";
    if (!d.leaderboard.length) h("p", "hint", board, "No formula has survived the stages yet."); else {
      const wrap = FD.guardScroll(h("div", "scroll tall", board)), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t));
      for (const [label, numeric, title] of [["Formula", 0, ""], ["Used as", 0, "Which way the formula is used: the sign was learned on the training dates only. The IC figures are in this direction."], ["Status", 0, ""], ["IC", 1, "Mean rank IC on the training dates, in the direction the formula is used"], ["Validation IC", 1, ""], ["Locked IC", 1, "Mean IC on the locked dates (only for finalists)"], ["t (locked)", 1, ""], ["Sharpe", 1, "Net long-short Sharpe on training and validation"],
        ["Turnover", 1, ""], ["q", 1, "False-discovery rate over every formula tried"], ["DSR", 1, "Deflated Sharpe: probability the Sharpe exceeds what the best of this many tries would show by luck"], ["Parts", 1, ""]]) { const th = h("th", numeric ? "num" : "", hd, label); if (title) th.title = title; }
      const tb = h("tbody", "", t);
      for (const r of d.leaderboard) {
        const tr = h("tr", "", tb); h("td", "formula", tr, r.expression); h("td", "", tr, WAY(r.direction)); h("td", "", tr, r.status === "candidate" ? "Candidate" : "survivor" === r.status ? "Survivor" : human(r.status)); h("td", "num", tr, sgn(r.ic)); h("td", "num", tr, sgn(r.val_ic)); h("td", "num", tr, sgn(r.oos_ic)); h("td", "num", tr, num(r.oos_t, 1));
        h("td", "num", tr, num(r.sharpe, 2)); h("td", "num", tr, pct(r.turnover, 0)); h("td", "num", tr, num(r.q_value, 2)); h("td", "num", tr, num(r.deflated_sharpe, 2)); h("td", "num", tr, String(r.nodes ?? ""));
      }
    }
    if (d.leaderboard.length) csvButton(board, "Download the list (CSV)", `discovery_${d.run_id}.csv`, ["formula", "used_as", "status", "ic", "val_ic", "oos_ic", "oos_t", "sharpe", "turnover", "q_value", "deflated_sharpe", "nodes"],
      d.leaderboard.map((r) => [r.expression, WAY(r.direction), r.status, r.ic, r.val_ic, r.oos_ic, r.oos_t, r.sharpe, r.turnover, r.q_value, r.deflated_sharpe, r.nodes]));
    const act = h("div", "row tight", root);
    const lib = h("button", "btn primary", act, "Open the library"); lib.type = "button"; lib.addEventListener("click", () => { FD.views.libraryView = "alphas"; FD.goto("library"); });
    if (d.status !== "finished") { const res = h("button", "btn", act, "Resume this run"); res.type = "button"; res.dataset.action = "resume-run"; res.addEventListener("click", () => DS.resume(d.run_id)); }
    else { const chk = h("button", "btn", act, "How much of this is luck?"); chk.type = "button"; chk.dataset.action = "placebo-run"; chk.addEventListener("click", () => DS.placebo(d.run_id));
      chk.title = "Run the same search on returns dealt out at random among the companies, where nothing can be found, and set what it finds beside what this run found."; }
    const done = (d.candidates || []).length ? W.card(root, "Every formula in this run", "Those that were rejected are in the graveyard with the reason: nothing is deleted.") : null;
    if (done) table(done, [{ label: "Formula", cls: "formula", get: (r) => r.text }, { label: "Status", get: (r) => human(r.status) }, { label: "Stage reached", get: (r) => stageWord(Q.bad(r.stage_reached) ? "" : (FD.stageNames || [])[r.stage_reached] || String(r.stage_reached)) }, { label: "Why", get: (r) => r.reason || "" }], d.candidates.slice(0, 120));
    if (after) after(root);
    FD.notes(root, "Read before relying on it", [d.note]);
    const card = runsCard(); if (card) root.appendChild(card);
  };
  /* What a search could find, before it is run. */
  DS.preflightView = (root, pf) => {
    Q.labHead(root, "What this search could find", `${pf.strictness} bars · ${pf.formulas} formulas · ${pf.blocks.dates.train} training, ${pf.blocks.dates.validation} validation, ${pf.blocks.dates.oos} locked dates`, GUIDE);
    const level = pf.adequate ? "good" : "mixed", v = h("div", "card verdict-card " + level, root); v.dataset.level = level; v.dataset.role = "fund-preflight";
    const head = h("div", "vhead", v); h("span", "vicon", head, ICON[level]); h("h2", "", head, pf.adequate ? "Able to find a real alpha of the usual size" : "Underpowered for the alphas worth having");
    h("p", "", v, pf.reading);
    const k = h("div", "earnrow", root), p = pf.power;
    tile(k, "Companies on a date", num(p.train.companies, 0), { text: "with a return to learn from, typical date" });
    tile(k, "First screen sees", sgn((pf.limits.t_ic + 0.8416) * p.train.se_of_mean_ic), { text: `a real mean IC, four times in five (|t| of ${num(pf.limits.t_ic, 2)})` }, "", "The smallest average rank IC that clears the first screen on the training dates with 80% probability");
    tile(k, "Locked test sees", sgn((pf.limits.oos_t + 0.8416) * p.oos.se_of_mean_ic), { text: `on ${pf.blocks.dates.oos} dates (t of ${num(pf.limits.oos_t, 2)})` }, "", "The same for the locked dates, which are the fewest");
    tile(k, "Luck at the first screen", num(pf.luck_first_screen, 0), { text: `of ${pf.formulas} formulas pass by chance alone` }, "", "The number of formulas expected to reach the bar with no real effect at all. The later stages and the correction for the number tried remove most of them.");
    chartCard(root, "The chance that a real alpha gets through", "For each strength of true mean IC (a useful signal is usually 0.02 to 0.05): the chance of passing the first screen, the validation block and the locked test. The screens in between only lower it.", {
      draw: (b) => C.columns(b, { labels: pf.table.map((r) => num(r.ic, 2)), values: pf.table.map((r) => r.found), yFormat: (x) => (x * 100).toFixed(0) + "%", name: "Chance of becoming a Candidate", label: "Chance that a real alpha gets through, by its true IC", height: 220, colors: () => css("--s1") }),
      table: (b) => table(b, [{ label: "True mean IC", num: true, get: (r) => num(r.ic, 2) }, { label: "First screen", num: true, get: (r) => pct(r.first_screen, 0) }, { label: "Validation", num: true, get: (r) => pct(r.validation, 0) }, { label: "Locked test", num: true, get: (r) => pct(r.locked_test, 0) }, { label: "All three", num: true, get: (r) => pct(r.found, 0) }], pf.table) });
  };

  /* The search on shuffled returns: what luck alone gets through. */
  DS.placeboView = (root, r, first) => {
    const s = r.summary;
    if (first) Q.labHead(root, "The search on shuffled returns", `${r.strictness} bars · ${s.runs} shuffle${s.runs === 1 ? "" : "s"} · ${s.tested} formulas tested`, GUIDE);
    const real = r.real, share = s.luck_share;
    const upper = real && real.survivors && s.luck_share_interval ? s.luck_share_interval[1] : 0;
    const level = !s.runs ? "bad" : s.t_calibration === "generous" || (real && real.survivors && share >= 0.5) ? "bad" : (real && real.survivors && (share > 0.2 || upper >= 0.5)) || s.candidates_max > 0 ? "mixed" : "good";
    const v = h("div", "card verdict-card " + level, root); v.dataset.level = level; v.dataset.role = "fund-placebo";
    const head = h("div", "vhead", v); h("span", "vicon", head, ICON[level]);
    h("h2", "", head, !s.runs ? "No shuffled search finished" : real && real.survivors ? (share >= 0.5 ? "Most of what this run found is what luck gives" : `Luck is expected to account for about ${pct(share, 0)} of what this run let through, and could be as much as ${pct(upper, 0)}`) : "What luck alone gets through");
    h("p", "", v, r.reading);
    if (!s.runs) return;
    const k = h("div", "earnrow", root);
    tile(k, "Luck lets through", num(s.survivors_mean, 1), { text: `of ${num(s.tested_per_run, 0)} formulas a search · ${num(s.survivors_interval[0], 1)} to ${num(s.survivors_interval[1], 1)}` }, "", "Formulas that clear every screen before the locked test on shuffled returns, where none can be real");
    if (real && real.survivors) tile(k, "Share of the real run that is luck", pct(share, 0), { text: `${pct(s.luck_share_interval[0], 0)} to ${pct(s.luck_share_interval[1], 0)} · the real run let ${real.survivors} through` }, "", "What luck lets through, divided by what the real search let through");
    tile(k, "Spread of the t-statistics", num(s.t_sd, 2), { text: Q.bad(s.t_sd_error) ? "1.00 is correct" : `1.00 is correct · give or take ${num(s.t_sd_error, 2)}`, dir: s.t_calibration === "generous" ? "down" : "" }, "", "On shuffled returns the t-statistics of the formulas should have a spread of 1. More than that and the standard errors are too small for this data.");
    tile(k, "5% bar on this data", Q.bad(s.t_bar_5pct) ? "n/a" : num(s.t_bar_5pct, 1), { text: "the |t| that luck exceeds one time in twenty" }, "", "1.96 if the t-statistic is correct");
    tile(k, "Candidates from luck", `${s.runs_with_a_candidate} of ${s.runs}`, { text: `${num(s.finalists_mean, 1)} reached the locked test on average` });
    const ch = h("div", "charts", root), hist = r.t_histogram || { edges: [], counts: [] };
    if (hist.counts.length) chartCard(ch, "The t-statistics of the shuffled formulas", "One value per formula tried on shuffled returns. A correct t-statistic puts about one formula in ten beyond ±1.65, one in twenty beyond ±1.96, and almost none beyond ±3.", {
      draw: (b) => C.histogram(b, { series: [{ name: "Shuffled formulas", edges: hist.edges, counts: hist.counts, color: css("--s1") }], markers: [{ x: -1.96, label: "-1.96" }, { x: 1.96, label: "1.96" }], xFormat: (x) => x.toFixed(1), label: "t-statistics of the formulas on shuffled returns", height: 220 }),
      table: (b) => table(b, [{ label: "From", num: true, get: (i) => num(hist.edges[i], 2) }, { label: "To", num: true, get: (i) => num(hist.edges[i + 1], 2) }, { label: "Formulas", num: true, get: (i) => String(hist.counts[i]) }], hist.counts.map((_, i) => i)) });
    const stages = Object.entries(s.funnel_mean);
    chartCard(ch, "Where the shuffled formulas fell", "Formulas passing each stage, on average per search. On real data most of the later stages should hold on to more than this.", {
      draw: (b) => C.hbars(b, { rows: stages.map(([stage, v]) => ({ label: stageWord(stage), value: v })), format: (x) => x.toFixed(1), name: "Passed", label: "Shuffled formulas passing each stage", labelWidth: 230, valueWidth: 50, color: () => css("--s3") }),
      table: (b) => table(b, [{ label: "Stage", get: (x) => stageWord(x[0]) }, { label: "Passed (average)", num: true, get: (x) => num(x[1], 1) }], stages) });
    const tails = W.card(root, "Does the t-statistic mean what it says here?", "The share of shuffled formulas whose |t| reached each level, beside the share a correct t-statistic gives.");
    tails.dataset.role = "fund-placebo-tails";
    table(tails, [{ label: "|t| of at least", num: true, get: (x) => x[0] }, { label: "Found", num: true, get: (x) => pct(x[1].share, 1) }, { label: "Expected", num: true, get: (x) => pct(x[1].expected, 1) }], Object.entries(s.tail));
    FD.explain(root, "What this does and does not show", "The companies' returns are dealt out at random on every date, the same way for every window that starts on that date, so no formula can predict anything; everything known on a date is untouched. What the search finds anyway is luck, and how much of a real run that makes is the share above. A shuffle also breaks up the tendency of companies in one country or sector to move together, so it cannot show the extra overconfidence this causes for a formula that is the same across a whole sector or country: the premium regression and the neutralised studies in the Factor lab are the checks for that. Nothing is written to the library or the graveyard, and the real locked dates are not looked at.");
  };
  FD.catalogue().then((cat) => { FD.stageNames = cat.discovery.stages; }).catch(() => {});

  /* =========================================================== Alpha library and what is built from it */
  const LB = FD.views.library = {};
  FD.views.libraryView = "alphas";
  const VIEWS = [["alphas", "Alphas"], ["graveyard", "Graveyard"], ["correlation", "Correlation"], ["combine", "Combine"], ["model", "Model"], ["backtest", "Backtest"]];
  const LABEL = { alphas: "Refresh the list", graveyard: "Show the graveyard", correlation: "Analyse the correlation", combine: "Combine the alphas", model: "Fit the model and compare", backtest: "Backtest this alpha" };
  const BLURB = { alphas: "Everything the searches have kept, best first, with the evidence and the status. A Candidate has passed tests on dates it was not chosen on; none is validated until the evidence holds and time has passed.",
    graveyard: "Every idea that failed, with the stage it failed at and why. Nothing is ever deleted: the record of what was tried is what makes the survivors honest.",
    correlation: "Two alphas that rank the companies the same way are one idea. See which are alike, and how many independent ideas a set really holds.",
    combine: "Put alphas together with a rule that uses only what was known on each date, and see whether the whole beats the best part.",
    model: "Fit a model walk-forward (no training row whose return was still unfolding) and check it against the simple rules. A model that cannot beat a plain average is not earning its complexity.",
    backtest: "Turn an alpha into a book under limits and run it through the platform's backtest engine, net of costs, set against owning every eligible company." };
  FD.views.libraryLabel = () => LABEL[FD.views.libraryView];
  FD.views.libraryBlurb = () => BLURB[FD.views.libraryView];
  LB.s = { status: "", order: "fitness", limit: "100", stage: "" };
  LB.parts = [{ kind: "factor", name: "fcf_yield", weight: "1" }, { kind: "factor", name: "d_roic", weight: "1" }];
  LB.opts = { method: "icir", model: "ridge", horizon: "1m", window: "36", min_train: "36", retrain_every: "12", embargo: "1", tune: false, cut: "0.5", how: "weights" };
  LB.pf = { side: "long_only", selection: "top_n", n: "30", percentile: "20", weighting: "equal", leg_exposure: "1", max_position: "0.10", max_sector: "", max_industry: "", max_country: "", max_beta: "", max_turnover: "", min_market_cap_usd: "",
            min_dollar_volume_usd: "", max_adv_share: "", capital_usd: "1000000", min_names: "5", cost_bps: "10", borrow_bps: "50", start: "", end: "", n_trials: "" };
  LB.sel = new Set(); LB.alphas = null; LB.detail = null; LB.keep = {};
  /* The lab keeps one result for the whole library tab; a view that is left is filed in ``keep`` so that coming back shows it again, and anything that changes the library (a search, the monitor, a new status) empties it. */
  LB.stale = () => { LB.alphas = null; LB.detail = null; LB.keep = {}; delete FD.results.library; delete FD.errors.library; };
  LB.switch = (view) => {
    if (view === FD.views.libraryView) return;
    if (FD.results.library) LB.keep[FD.views.libraryView] = FD.results.library;
    FD.views.libraryView = view; FD.errors.library = null;
    if (LB.keep[view]) FD.results.library = LB.keep[view]; else delete FD.results.library;
    Q.renderLab(FD);                                                                           // the lists are read when the view opens; the heavier views wait for their button
  };
  FD.views.reset = () => { LB.stale(); LB.sel = new Set(); DS.runs = null; };
  FD.views.libraryUse = (parts, view) => {
    LB.parts = parts.map((p) => Object.assign({ weight: "1" }, p));
    if (FD.results.library && FD.views.libraryView !== view) LB.keep[FD.views.libraryView] = FD.results.library;
    delete LB.keep[view]; delete FD.results.library; delete FD.errors.library;                 // new signals: the old result of this view is not theirs
    FD.views.libraryView = view; FD.mode = "library"; Q.renderLab(FD);
  };
  LB.check = () => {
    const v = FD.views.libraryView;
    if (["correlation", "combine", "model", "backtest"].includes(v) && FD.needsData()) return FD.needsData();
    if (["correlation", "combine", "model"].includes(v) && LB.parts.length < 2) return "Choose at least two signals.";
    if (v === "backtest" && !LB.parts.length) return "Choose a signal to backtest.";
    return "";
  };
  LB.sections = (panel) => {
    const v = FD.views.libraryView, after = FD.after;
    const sw = FD.sec(panel, "View");
    const seg = W.seg(sw, VIEWS, v, (x) => LB.switch(x)); seg.dataset.role = "fund-lib-view"; seg.classList.add("wrap");
    if (v === "alphas") {
      const b = FD.sec(panel, "Show");
      Q.labField(b, { key: "status", label: "Status", kind: "choice", options: [["", "all"], ["Candidate", "Candidate"], ["Validated", "Validated"], ["Production", "Production"], ["Decaying", "Decaying"], ["Retired", "Retired"], ["Rejected", "Rejected"]] }, LB.s, after);
      Q.labField(b, { key: "order", label: "Best first by", kind: "choice", options: [["fitness", "fitness"], ["ic", "information coefficient"], ["icir", "IC information ratio"], ["sharpe", "Sharpe"], ["oos_ic", "locked-dates IC"], ["oos_sharpe", "locked-dates Sharpe"], ["created_at", "newest"], ["turnover", "turnover"]] }, LB.s, after);
      const u = FD.sec(panel, "Use the ticked alphas");
      for (const [view, label] of [["correlation", "Correlation"], ["combine", "Combine"], ["model", "Model"], ["backtest", "Backtest"]]) {
        const b2 = h("button", "btn small", u, label); b2.type = "button"; b2.dataset.action = "use-" + view; b2.addEventListener("click", () => {
          if (!LB.sel.size) { Q.toast("Tick one or more alphas in the list first"); return; }
          FD.views.libraryUse([...LB.sel].map((id) => ({ kind: "alpha", alpha_id: id })), view);
        });
      }
      return;
    }
    if (v === "graveyard") {
      const b = FD.sec(panel, "Show");
      FD.catalogue().then((cat) => { if (!b.isConnected) return; Q.labField(b, { key: "stage", label: "Stage failed at", kind: "choice", options: [["", "any"]].concat(cat.discovery.stages.map((x) => [x, stageWord(x)])) }, LB.s, after); }).catch(() => {});
      return;
    }
    partsEditor(panel);
    FD.universeSection(panel, "Which companies and when");
    const o = LB.opts;
    if (v === "correlation") { const b = FD.sec(panel, "Clusters"); Q.labField(b, { key: "cut", label: "Group alphas closer than (distance = 1 − |correlation|)", kind: "number", step: 0.05, min: 0.05, max: 0.95 }, o, after); }
    if (v === "combine") {
      const b = FD.sec(panel, "The rule");
      Q.labField(b, { key: "method", label: "Weigh the alphas by", kind: "choice", options: [["equal", "nothing (equal weights)"], ["ic", "their recent information coefficient"], ["icir", "their recent IC information ratio"], ["diversified", "maximum information ratio (allows for how the ICs move together)"], ["cluster", "equal weight to each cluster of alike alphas"]] }, o, after);
      Q.labField(b, { key: "horizon", label: "Return measured over the next", kind: "choice", options: [["1m", "month"], ["3m", "3 months"], ["6m", "6 months"], ["12m", "year"]] }, o, after);
      Q.labField(b, { key: "window", label: "Periods of IC history used", kind: "number", step: 1, min: 6, max: 240 }, o, after);
      h("p", "hint", b, "Each date's weights use only information coefficients whose return windows had ended by that date; until enough are known the weights are equal.");
    }
    if (v === "model") {
      const b = FD.sec(panel, "The model");
      FD.catalogue().then((cat) => { if (!b.isConnected) return; Q.labField(b, { key: "model", label: "Model", kind: "choice", options: cat.combination.models.map((m) => [m, human(m)]) }, o, after);
        Q.labField(b, { key: "horizon", label: "Return measured over the next", kind: "choice", options: [["1m", "month"], ["3m", "3 months"], ["6m", "6 months"]] }, o, after);
        Q.labField(b, { key: "min_train", label: "Periods before the first prediction", kind: "number", step: 1, min: 12, max: 240 }, o, after);
        Q.labField(b, { key: "retrain_every", label: "Refit every (periods)", kind: "number", step: 1, min: 1, max: 60 }, o, after);
        Q.labField(b, { key: "embargo", label: "Extra gap after each training window (periods)", kind: "number", step: 1, min: 0, max: 12 }, o, after);
        Q.labField(b, { key: "tune", label: "Try a small grid of settings inside each training window", kind: "bool" }, o, after);
        h("p", "hint", b, "Each prediction comes from a model that saw only dates at least as far back as the forecast horizon plus the gap. A missing alpha counts as no view (zero), never as a bad value."); }).catch((e) => h("p", "hint", b, e.message));
    }
    if (v === "backtest") backtestForm(panel);
  };
  /* The signals a combination or a backtest is made of: library alphas, library factors, price-based signals and formulas. */
  function partsEditor(panel) {
    const b = FD.sec(panel, "Signals"), list = h("div", "", b), after = FD.after;
    const draw = () => {
      list.replaceChildren();
      LB.parts.forEach((p, i) => {
        const row = h("div", "row tight", list); row.dataset.part = String(i); row.style.alignItems = "end";
        const kind = h("select", "", h("label", "f", row, "Kind")); kind.setAttribute("aria-label", `signal ${i + 1} kind`);
        for (const [v, t] of [["alpha", "Library alpha"], ["factor", "Library factor"], ["technical", "Price-based"], ["formula", "Formula"]]) { const o = h("option", "", kind, t); o.value = v; if (v === p.kind) o.selected = true; }
        kind.addEventListener("change", () => { p.kind = kind.value; delete p.name; delete p.alpha_id; delete p.formula; draw(); after(); });
        const which = h("label", "f", row, "Which");
        if (p.kind === "formula") { const inp = h("input", "", which); inp.type = "text"; inp.value = p.formula || ""; inp.placeholder = "div(fcf,market_cap)"; inp.spellcheck = false; inp.setAttribute("aria-label", `signal ${i + 1} formula`); inp.addEventListener("input", () => { p.formula = inp.value; after(); }); }
        else {
          const sel = h("select", "", which); sel.setAttribute("aria-label", `signal ${i + 1} name`);
          const fill = (items, current, key) => { for (const [v, t] of items) { const o = h("option", "", sel, t); o.value = v; if (v === current) o.selected = true; } if (!current && items.length) p[key] = items[0][0]; sel.addEventListener("change", () => { p[key] = sel.value; after(); }); };
          if (p.kind === "alpha") { ensureAlphas().then((a) => { if (!sel.isConnected) return; fill(a.map((x) => [x.alpha_id, `${x.text} (${x.status})`]), p.alpha_id, "alpha_id"); if (!a.length) h("option", "", sel, "no alphas in the library yet"); }); }
          else FD.catalogue().then((cat) => { if (!sel.isConnected) return; const names = p.kind === "technical" ? cat.portfolio.technical : cat.factors.map((f) => f.name); fill(names.map((n) => [n, (cat.factors.find((f) => f.name === n) || { label: n }).label]), p.name, "name"); });
        }
        const w = h("label", "f", row, "Weight"), wi = h("input", "", w); wi.type = "number"; wi.step = "any"; wi.value = p.weight; wi.style.width = "64px"; wi.setAttribute("aria-label", `signal ${i + 1} weight`); wi.addEventListener("input", () => { p.weight = wi.value; after(); });
        const rm = h("button", "btn small", row, "Remove"); rm.type = "button"; rm.setAttribute("aria-label", `remove signal ${i + 1}`); rm.addEventListener("click", () => { LB.parts.splice(i, 1); draw(); after(); });
      });
    };
    draw();
    const add = h("button", "btn small", b, "Add a signal"); add.type = "button"; add.dataset.action = "add-part"; add.addEventListener("click", () => { LB.parts.push({ kind: "factor", weight: "1" }); draw(); after(); });
    h("p", "hint", b, "Weights apply when signals are blended with fixed weights (a negative weight turns a signal upside down); the combination rules and the model choose their own.");
  }
  async function ensureAlphas() {
    if (LB.alphas) return LB.alphas;
    try { LB.alphas = (await FD.read("/api/fund/library", { limit: 500 })).alphas; } catch (e) { LB.alphas = []; }
    return LB.alphas;
  }
  /* Parts as the server wants them; a formula is parsed on the server first (the request carries the tree). */
  async function partsBody() {
    const out = [];
    for (const p of LB.parts) {
      const part = { kind: p.kind };
      if (p.kind === "alpha") { if (!p.alpha_id) throw new Error("Choose an alpha for every alpha signal"); part.alpha_id = p.alpha_id; }
      else if (p.kind === "formula") {
        if (!(p.formula || "").trim()) throw new Error("Write the formula for every formula signal");
        const r = await FD.send("/api/fund/formula/check", { formula: p.formula }); if (!r.ok) throw new Error(`${p.formula}: ${r.error}`);
        part.kind = "expression"; part.expression = r.tree;
      } else { if (!p.name) throw new Error("Choose a factor for every factor signal"); part.name = p.name; }
      const w = Number(p.weight); if (!isFinite(w)) throw new Error("Weights must be numbers");
      part.weight = w; out.push(part);
    }
    return out;
  }
  function backtestForm(panel) {
    const o = LB.opts, p = LB.pf, after = FD.after;
    const how = FD.sec(panel, "How the signals become one");
    Q.labField(how, { key: "how", label: "Blend", kind: "choice", options: [["weights", "with the weights above"], ["rule", "by a rule (equal, IC, ICIR, ...)"], ["model", "with a model fitted walk-forward"]] }, o, () => { Q.renderLab(FD); });
    if (o.how === "rule") Q.labField(how, { key: "method", label: "Rule", kind: "choice", options: [["equal", "equal"], ["ic", "IC"], ["icir", "ICIR"], ["diversified", "maximum information ratio"], ["cluster", "cluster-balanced"]] }, o, after);
    if (o.how === "model") FD.catalogue().then((cat) => { if (how.isConnected) Q.labField(how, { key: "model", label: "Model", kind: "choice", options: cat.combination.models.map((m) => [m, human(m)]) }, o, after); });
    FD.universeSection(panel, "Which companies and when");
    const a = FD.sec(panel, "The book");
    Q.labField(a, { key: "side", label: "Long only or long and short", kind: "choice", options: [["long_only", "long only"], ["long_short", "long and short"]] }, p, after);
    Q.labField(a, { key: "selection", label: "Choose by", kind: "choice", options: [["top_n", "the top N companies"], ["percentile", "the top percentile"]] }, p, () => { Q.renderLab(FD); });
    if (p.selection === "top_n") Q.labField(a, { key: "n", label: "Companies in each leg", kind: "number", step: 1, min: 1, max: 5000 }, p, after);
    else Q.labField(a, { key: "percentile", label: "Percent of the eligible companies in each leg", kind: "number", min: 1, max: 100 }, p, after);
    Q.labField(a, { key: "weighting", label: "Weights inside a leg", kind: "choice", options: [["equal", "equal"], ["signal", "by how far the signal clears the cutoff"], ["inverse_vol", "by inverse volatility"]] }, p, after);
    Q.labField(a, { key: "leg_exposure", label: "What each leg adds up to (1 = the whole account)", kind: "number", min: 0.1, max: 3 }, p, after);
    Q.labField(a, { key: "min_names", label: "Fewest companies in a leg (fewer: cash)", kind: "number", step: 1, min: 1 }, p, after);
    const l = FD.sec(panel, "Limits (blank = none)");
    for (const [k, label, help] of [["max_position", "Largest position (share of the account)", ""], ["max_sector", "Most in one sector, net", ""], ["max_industry", "Most in one industry, net", ""], ["max_country", "Most in one country, net", ""],
      ["max_beta", "Largest net beta to the universe, either sign", ""], ["max_turnover", "Most bought or sold at one rebalance (share of the book)", "A whole new book is 1.0; the first book is built over several dates"],
      ["min_market_cap_usd", "Smallest market value (US dollars)", ""], ["min_dollar_volume_usd", "Smallest daily trading value (US dollars)", ""], ["max_adv_share", "Most of a company's daily trading value one position may be", "Needs the account size below"]])
      Q.labField(l, { key: k, label, kind: "number", min: 0, step: "any", help }, p, after);
    Q.labField(l, { key: "capital_usd", label: "Account size (US dollars)", kind: "number", min: 1 }, p, after);
    h("p", "hint", l, "Limits are met by the nearest book that respects them. When they cannot all be met, both legs are scaled down together and the rest stays in cash; the report says how often.");
    const c = FD.sec(panel, "Costs and dates");
    Q.labField(c, { key: "cost_bps", label: "Trading cost, one way (bps)", kind: "number", min: 0, max: 1000 }, p, after);
    Q.labField(c, { key: "borrow_bps", label: "Borrowing cost of shorts (bps a year)", kind: "number", min: 0, max: 5000 }, p, after);
    Q.labField(c, { key: "start", label: "First rebalance (blank = all)", kind: "text", placeholder: "2014-01-31" }, p, after);
    Q.labField(c, { key: "end", label: "Last rebalance", kind: "text" }, p, after);
    Q.labField(c, { key: "n_trials", label: "Ideas tried to find this alpha (blank = from the library)", kind: "number", step: 1, min: 1, help: "The deflated Sharpe ratio is penalised for this number." }, p, after);
  }
  const portfolioBody = () => {
    const p = LB.pf, out = { side: p.side, selection: p.selection, weighting: p.weighting };
    for (const k of ["n", "percentile", "leg_exposure", "max_position", "max_sector", "max_industry", "max_country", "max_beta", "max_turnover", "min_market_cap_usd", "min_dollar_volume_usd", "max_adv_share", "capital_usd", "min_names", "cost_bps", "borrow_bps"]) {
      if (p[k] === "") continue; const v = Number(p[k]); if (!isFinite(v)) throw new Error(`${human(k)} must be a number`); out[k] = v;
    }
    if (p.start) out.start = p.start; if (p.end) out.end = p.end;
    return out;
  };

  LB.run = () => {
    const v = FD.views.libraryView, o = LB.opts;
    if (v === "alphas") return FD.act("library", "Reading the library…", async () => { LB.alphas = null; return { view: "alphas", data: await FD.read("/api/fund/library", { status: LB.s.status, order: LB.s.order, limit: LB.s.limit }) }; });
    if (v === "graveyard") return FD.act("library", "Reading the graveyard…", async () => ({ view: "graveyard", data: await FD.read("/api/fund/graveyard", { stage: LB.s.stage, limit: 300 }) }));
    return FD.act("library", { correlation: "Correlating…", combine: "Combining…", model: "Fitting the model…", backtest: "Backtesting…" }[v], async (run) => {
      const parts = await partsBody(), ctxBody = FD.universeBody();
      if (v === "correlation") { const { result } = await run.job("/api/fund/correlation", Object.assign({ parts, cut: Number(o.cut) }, ctxBody), "Correlating"); return { view: v, data: result }; }
      if (v === "combine") {
        const { result } = await run.job("/api/fund/combine", Object.assign({ parts, method: o.method, horizon: o.horizon, options: { window: Number(o.window) } }, ctxBody), "Combining");
        return { view: v, data: result };
      }
      if (v === "model") {
        const options = { min_train: Number(o.min_train), retrain_every: Number(o.retrain_every), embargo: Number(o.embargo), tune: !!o.tune, permutation_repeats: 2 };
        const { result } = await run.job("/api/fund/ml", Object.assign({ parts, model: o.model, horizon: o.horizon, options }, ctxBody), "Fitting the model");
        return { view: v, data: result };
      }
      const portfolio = portfolioBody(), body = Object.assign({ portfolio }, ctxBody);
      const only = parts.length === 1 ? parts[0] : null;
      if (o.how === "rule" && parts.length > 1) body.signal = { kind: "combination", method: o.method, of: parts.map(stripWeight), horizon: "1m" };
      else if (o.how === "model" && parts.length > 1) body.signal = { kind: "ml", model: o.model, of: parts.map(stripWeight), horizon: "1m" };
      else body.signal = only ? stripWeight(only) : { parts };
      if (LB.pf.n_trials !== "") body.n_trials = Number(LB.pf.n_trials);
      const { result } = await run.job("/api/fund/backtest", body, "Backtesting");
      return { view: v, data: result };
    });
  };
  const stripWeight = (p) => { const q = Object.assign({}, p); delete q.weight; return q; };
  LB.auto = () => { if (["alphas", "graveyard"].includes(FD.views.libraryView)) LB.run(); };
  LB.empty = (root) => {
    const v = FD.views.libraryView;
    if (v === "alphas" || v === "graveyard") { FD.emptyCard(root, "Reading…", ""); return; }
    FD.emptyCard(root, { correlation: "How alike are the alphas?", combine: "Does the whole beat its best part?", model: "Does a model add anything to a plain average?", backtest: "What would trading this alpha have earned?" }[v], BLURB[v] + " Choose the signals on the left and press the button.",
      { label: LABEL[v], fn: () => LB.run() });
  };
  LB.build = (root, got) => {
    // a result belongs to the view that made it; a different view open now shows its own empty state
    if (got.view !== FD.views.libraryView) { LB.empty(root); return; }
    ({ alphas: alphasView, graveyard: graveyardView, correlation: correlationView, combine: combineView, model: modelView, backtest: backtestView })[got.view](root, got.data);
  };
  function alphasView(root, d) {
    Q.labHead(root, "Alpha library", `${d.alphas.length} shown · ${d.counts.graveyard} failed ideas in the graveyard · ${d.counts.tried} ideas tried in all`, GUIDE);
    const k = h("div", "earnrow", root);
    for (const st of d.statuses) tile(k, st, String((d.counts.library || {})[st] || 0), {}, (d.counts.library || {})[st] ? "" : "muted");
    const card = W.card(root, "Alphas", "Tick alphas to correlate, combine, model or backtest them. Click a formula for its evidence, history and status.");
    card.dataset.role = "fund-alphas";
    if (!d.alphas.length) { h("p", "hint", card, "The library is empty. Start a search in Alpha discovery: what survives the stages is added here as a Candidate."); return; }
    const wrap = FD.guardScroll(h("div", "scroll tall", card)), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t));
    h("span", "vh", h("th", "", hd), "Tick");
    for (const [label, numeric] of [["Formula", 0], ["Used as", 0], ["Status", 0], ["IC", 1], ["Locked IC", 1], ["Sharpe", 1], ["Turnover", 1], ["Fitness", 1], ["q", 1], ["DSR", 1], ["Found", 0]]) h("th", numeric ? "num" : "", hd, label);
    const tb = h("tbody", "", t);
    for (const a of d.alphas) {
      const tr = h("tr", "", tb); tr.dataset.alpha = a.alpha_id;
      const cb = h("input", "", h("td", "", tr)); cb.type = "checkbox"; cb.checked = LB.sel.has(a.alpha_id); cb.setAttribute("aria-label", "tick " + a.text); cb.addEventListener("change", () => { if (cb.checked) LB.sel.add(a.alpha_id); else LB.sel.delete(a.alpha_id); });
      const f = h("td", "formula", tr), link = h("button", "link", f, a.text); link.type = "button"; link.dataset.action = "open-alpha"; link.addEventListener("click", () => openAlpha(a.alpha_id));
      h("td", "", tr, WAY(a.direction));
      const st = h("td", "", tr); badge(st, a.status);
      h("td", "num", tr, sgn(a.ic)); h("td", "num", tr, sgn(a.oos_ic)); h("td", "num", tr, num(a.sharpe, 2)); h("td", "num", tr, pct(a.turnover, 0)); h("td", "num", tr, num(a.fitness, 2)); h("td", "num", tr, num(a.q_value, 2)); h("td", "num", tr, num(a.deflated_sharpe, 2)); h("td", "", tr, day(a.created_at));
    }
    csvButton(card, "Download the list (CSV)", "alpha_library.csv", ["alpha_id", "formula", "used_as", "status", "ic", "oos_ic", "sharpe", "turnover", "fitness", "q_value", "deflated_sharpe"],
      d.alphas.map((a) => [a.alpha_id, a.text, WAY(a.direction), a.status, a.ic, a.oos_ic, a.sharpe, a.turnover, a.fitness, a.q_value, a.deflated_sharpe]));
    const holder = h("div", "stack", root); holder.id = "fund-alpha-detail";
    if (LB.detail) paintDetail(holder, LB.detail);
  }
  const badge = (p, t) => FD.badge(p, t, t === "Rejected" || t === "Retired" || t === "Decaying" ? "warn" : "");
  async function openAlpha(id) {
    try { LB.detail = await FD.read("/api/fund/alpha", { id }); } catch (e) { Q.toast(e.message); return; }
    const holder = Q.$("#fund-alpha-detail"); if (holder) { holder.replaceChildren(); paintDetail(holder, LB.detail); holder.scrollIntoView({ block: "start" }); }
  }
  function paintDetail(root, a) {
    const card = h("div", "card", root); card.dataset.role = "fund-alpha-detail";
    const head = h("div", "bar", card); h("h3", "", head, a.text); badge(head, a.status);
    h("p", "note", card, a.status_reason || "");
    const m = a.metrics || {};
    const k = h("div", "earnrow", card);
    tile(k, "IC on the training dates", sgn(a.ic), { text: `ICIR ${num(a.icir, 2)}` }); tile(k, "IC on the locked dates", sgn(a.oos_ic), { text: `t ${num(m.oos_t, 1)} · looked at ${m.oos_looks ?? "?"} time(s)` });
    tile(k, "Net Sharpe", num(a.sharpe, 2), { text: `max drawdown ${pct(a.max_drawdown, 0)}` }); tile(k, "After the number tried", `q ${num(a.q_value, 2)}`, { text: `deflated Sharpe ${num(a.deflated_sharpe, 2)} of ${m.n_tested ?? "?"} tried` });
    const s = a.settings || {};
    h("p", "hint", card, `Used with ${WAY(m.direction) || "its own sign"}; the IC figures above are in that direction.`);
    h("p", "hint", card, `Found on ${(s.universe && Object.keys(s.universe).length) ? JSON.stringify(s.universe) : "the whole universe"}, ${s.freq === "M" ? "monthly" : s.freq === "W" ? "weekly" : "quarterly"}, ${s.horizon || ""} ahead. Blocks: ${Object.entries(s.blocks || {}).map(([n, b]) => `${n} ${b[0]} to ${b[1]}`).join("; ")}.`);
    const row = h("div", "row tight", card);
    const bt = h("button", "btn primary", row, "Backtest this alpha"); bt.type = "button"; bt.dataset.action = "backtest-alpha"; bt.addEventListener("click", () => FD.views.libraryUse([{ kind: "alpha", alpha_id: a.alpha_id }], "backtest"));
    const study = h("button", "btn", row, "Study it in the Factor lab"); study.type = "button"; study.addEventListener("click", () => { FL.s.source = "formula"; FL.s.formula = a.text; FL.s.direction = String((a.metrics || {}).direction || 1) === "-1" ? "-1" : "1"; FD.goto("factors"); });
    if (a.status === "Candidate") { const val = h("button", "btn", row, "Validate…"); val.type = "button"; val.dataset.action = "validate-alpha"; val.title = "Move to Validated if the evidence holds after the correction for the number of ideas tried"; val.addEventListener("click", () => validate(a.alpha_id)); }
    if (a.transitions.length) {
      const sel = h("select", "", h("label", "f", card, "Move to")); sel.setAttribute("aria-label", "new status"); for (const t of a.transitions) { const o = h("option", "", sel, t); o.value = t; }
      const reason = h("input", "", h("label", "f", card, "Reason (required)")); reason.type = "text"; reason.setAttribute("aria-label", "reason for the new status");
      const go = h("button", "btn small", card, "Change the status"); go.type = "button"; go.dataset.action = "set-status";
      go.addEventListener("click", async () => {
        if (!reason.value.trim()) { Q.toast("A change of status needs a reason"); reason.focus(); return; }
        try { await FD.send("/api/fund/library/status", { alpha_id: a.alpha_id, status: sel.value, reason: reason.value }); Q.toast("Status changed"); LB.alphas = null; await openAlpha(a.alpha_id); LB.run(); } catch (e) { Q.toast(e.message); } });
    }
    h("h4", "subhead", card, "Status history");
    table(card, [{ label: "When", get: (r) => String(r.at).replace("T", " ").slice(0, 16) }, { label: "From", get: (r) => r.from_status || "" }, { label: "To", get: (r) => r.to_status }, { label: "Why", get: (r) => r.reason || "" }, { label: "Forced", get: (r) => (r.forced ? "yes" : "") }], a.status_log);
    if (a.monitor.length) { h("h4", "subhead", card, "Monitor history"); table(card, [{ label: "Checked", get: (r) => String(r.checked_at).replace("T", " ").slice(0, 16) }, { label: "State", get: (r) => r.status }, { label: "Recent IC", num: true, get: (r) => sgn(r.recent_ic) }, { label: "Original IC", num: true, get: (r) => sgn(r.historical_ic) }, { label: "Coverage", num: true, get: (r) => num(r.coverage, 0) }], a.monitor); }
  }
  async function validate(id) {
    try { await FD.send("/api/fund/library/validate", { alpha_id: id }); Q.toast("Validated"); LB.run(); await openAlpha(id); return; }
    catch (e) {
      const why = window.prompt(`${e.message}\n\nTo approve it anyway, write why (this is recorded as a forced approval). Leave empty to cancel.`);
      if (!why || !why.trim()) return;
      try { await FD.send("/api/fund/library/validate", { alpha_id: id, reason: why }); Q.toast("Approved by hand"); LB.run(); await openAlpha(id); } catch (e2) { Q.toast(e2.message); }
    }
  }
  function graveyardView(root, d) {
    Q.labHead(root, "Graveyard", `${d.counts.graveyard} failed ideas · ${d.counts.library ? Object.values(d.counts.library).reduce((a, b) => a + b, 0) : 0} in the library · ${d.counts.tried} tried in all`, GUIDE);
    const stages = Object.entries(d.counts.graveyard_by_stage || {});
    if (stages.length) chartCard(root, "Where the ideas died", "Most ideas should fail early and cheaply. Those that fail at the locked test are the most informative: they looked good until they met data they had not been chosen on.", {
      draw: (b) => C.hbars(b, { rows: stages.map(([k, v]) => ({ label: stageWord(k), value: v })), format: (v) => String(Math.round(v)), name: "Ideas", label: "Failed ideas by stage", labelWidth: 230, valueWidth: 46, color: () => css("--s2") }),
      table: (b) => table(b, [{ label: "Stage", get: (r) => stageWord(r[0]) }, { label: "Ideas", num: true, get: (r) => String(r[1]) }], stages) });
    const card = W.card(root, "The failures", "Newest first. The reason says which test the idea failed and by how much. Nothing here is deleted.");
    card.dataset.role = "fund-graveyard";
    if (!d.rows.length) { h("p", "hint", card, "Nothing has failed yet, because nothing has been tried."); return; }
    csvButton(card, "Download the list (CSV)", "graveyard.csv", ["formula", "failed_at", "reason", "ic", "oos_ic", "sharpe", "date_tested", "run_id"], d.rows.map((r) => [r.text, r.stage, r.reason_rejected, r.ic, r.oos_ic, r.sharpe, r.date_tested, r.run_id]));
    table(card, [{ label: "Formula", cls: "formula", get: (r) => r.text }, { label: "Failed at", get: (r) => stageWord(r.stage) }, { label: "Why", get: (r) => r.reason_rejected }, { label: "IC", num: true, get: (r) => sgn(r.ic) }, { label: "Locked IC", num: true, get: (r) => sgn(r.oos_ic) }, { label: "Tested", get: (r) => day(r.date_tested) }], d.rows);
  }
  const numberOf = (name) => String(name).split(":")[0];                                          // each signal is named "<its place in the list>:<what it is>": the columns carry the same number as their rows
  function correlationView(root, d) {
    Q.labHead(root, "How alike the alphas are", d.reading, GUIDE);
    watch(root, d.warnings);
    const k = h("div", "earnrow", root);
    tile(k, "Alphas", String(d.alphas)); tile(k, "Independent ideas", num(d.effective_number, 1), { text: "from the eigenvalues of the correlation matrix" }, "", "(sum of eigenvalues)² / sum of squared eigenvalues: 1 for identical alphas, the number of alphas for unrelated ones");
    tile(k, "Clusters", String(d.clusters.length), { text: `alphas closer than ${d.cut} are grouped` }); tile(k, "Near-duplicates", String(d.redundant.length), { text: "pairs 80% or more alike" });
    chartCard(root, "Correlation of the alphas", "The average over dates of the rank correlation between each pair, in the order that puts alike alphas side by side. Dark blue is alike, dark red is opposite, grey is unrelated.", {
      wide: true,
      draw: (b) => C.matrix(b, { rows: d.heatmap.names, cols: d.heatmap.names.map(numberOf), values: d.heatmap.values, diverging: true, mid: 0, min: -1, max: 1, format: (v) => v.toFixed(2), name: "Correlation", label: "Correlation matrix of the alphas", labelWidth: Math.min(260, 7 * Math.max(...d.heatmap.names.map((n) => n.length)) + 14), cellHeight: 24 }),
      table: (b) => table(b, [{ label: "Alpha", get: (i) => d.heatmap.names[i] }].concat(d.heatmap.names.map((n, j) => ({ label: numberOf(n), num: true, title: n, get: (i) => (d.heatmap.values[i][j] === null ? "" : d.heatmap.values[i][j].toFixed(2)) }))), d.heatmap.names.map((_, i) => i)) });
    const card = W.card(root, "Clusters", "Average-linkage on the distance 1 − |correlation|. Alphas in one cluster say much the same thing; five copies of one idea should not outvote one different idea.");
    card.dataset.role = "fund-clusters";
    table(card, [{ label: "Cluster", get: (c) => String(c.id) }, { label: "Alphas", cls: "formula", get: (c) => c.members.join("  ·  ") }, { label: "Average |correlation|", num: true, get: (c) => (c.mean_abs_correlation === null ? "" : num(c.mean_abs_correlation, 2)) }], d.clusters);
    if (d.redundant.length) { const r = W.card(root, "Near-duplicates", "One of each pair adds almost nothing to the other."); table(r, [{ label: "A", cls: "formula", get: (x) => x.a }, { label: "B", cls: "formula", get: (x) => x.b }, { label: "Correlation", num: true, get: (x) => num(x.correlation, 2) }], d.redundant); }
  }
  function combineView(root, d) {
    Q.labHead(root, `Combined by ${d.method}`, `${d.evaluation.universe.companies} companies · ${d.evaluation.n_dates} dates`, GUIDE);
    watch(root, d.warnings);
    const best = d.parts.slice().sort((a, b) => Math.abs(b.mean_ic || 0) - Math.abs(a.mean_ic || 0))[0];
    const k = h("div", "earnrow", root);
    tile(k, "Combined mean IC", sgn(d.combined.mean_ic), { text: `t ${num(d.combined.t_stat, 1)}`, dir: d.combined.mean_ic > (d.equal_weight.mean_ic || 0) ? "up" : "down" });
    tile(k, "Equal weights", sgn(d.equal_weight.mean_ic), { text: `t ${num(d.equal_weight.t_stat, 1)}` }, "", "The plain average of the same alphas");
    tile(k, "Best single alpha", best ? sgn(best.mean_ic) : "n/a", { text: best ? best.label : "" }, "", "Chosen after seeing these same dates, so a generous yardstick");
    const card = W.card(root, "The parts", "Each alpha's own rank IC over the same dates, standardised the same way.");
    table(card, [{ label: "Alpha", cls: "formula", get: (r) => r.label }, { label: "Mean IC", num: true, get: (r) => sgn(r.mean_ic) }, { label: "t", num: true, get: (r) => num(r.t_stat, 1) }, { label: "IC information ratio", num: true, get: (r) => num(r.icir_annualised, 2) }, { label: "Dates positive", num: true, get: (r) => pct(r.positive_share, 0) }], d.parts);
    const ch = h("div", "charts", root), dates = d.weights.dates;
    lineCard(ch, "Weights over time", d.notes.join(" "), dates, d.weights.names.map((n, j) => ({ name: n, color: Q.colorOf(j), values: d.weights.values.map((row) => row[j]) })), (v) => (v * 100).toFixed(0) + "%", { wide: true, include: 0 });
    FD.tearSheet(root, d.evaluation, {});
  }
  function modelView(root, d) {
    Q.labHead(root, `${human(d.model)} model on ${d.parts.length} alphas`, `${d.folds.length} walk-forward fits · ${d.evaluation.n_dates} dates predicted`, GUIDE);
    watch(root, d.warnings);
    const cmp = d.comparison, level = cmp.model_beats_rules ? "good" : "mixed";
    const v = h("div", "card verdict-card " + level, root); v.dataset.role = "fund-model-verdict"; v.dataset.level = level;
    const head = h("div", "vhead", v); h("span", "vicon", head, cmp.model_beats_rules ? "✓" : "!"); h("h2", "", head, cmp.model_beats_rules ? "The model beats the simple rules" : "The model has not beaten the simple rules");
    h("p", "", v, cmp.verdict); for (const n of d.notes) h("p", "hint", v, n);
    const rows = [["The model", cmp.model, null, null]].concat(Object.entries(cmp.rules).map(([name, r]) => [name === "equal" ? "Equal weights" : "IC information ratio weights", r, r.model_minus_rule, r.model_minus_rule_t]));
    const card = W.card(root, "Against the simple rules, on the same dates", "The last two columns test the model's IC minus the rule's, allowing for the overlap of the return windows. The model needs a positive gap and t above about 2 to count as better.");
    card.dataset.role = "fund-model-compare";
    table(card, [{ label: "Measure", get: (r) => r[0] }, { label: "Mean IC", num: true, get: (r) => sgn(r[1].mean_ic) }, { label: "t", num: true, get: (r) => num(r[1].t_stat, 1) }, { label: "Model minus this", num: true, get: (r) => (r[2] === null ? "" : sgn(r[2])) }, { label: "t of the gap", num: true, get: (r) => (r[3] === null ? "" : num(r[3], 1)) }], rows);
    h("p", "hint", card, `Best single alpha: ${cmp.best_single.alpha} (${sgn(cmp.best_single.mean_ic)}); ${cmp.best_single.note}.`);
    const imp = d.importance, ch = h("div", "charts", root);
    for (const [key, title, note, fmt2] of [["coefficients", "What the model weighs (coefficients)", "Average absolute size of each alpha's coefficient across the fits: the linear models' own view of what matters.", (x) => x.toFixed(3)],
      ["built_in", "What the model splits on", `${imp.built_in_kind || "The model's own importance"}, averaged over the fits.`, (x) => (x * 100).toFixed(0) + "%"],
      ["permutation", "What breaks it if scrambled (permutation)", "How much the IC falls when one alpha is shuffled across companies within each date. Positive means the model relied on it; near zero or negative means it did not.", (x) => x.toFixed(3)],
      ["shap", "Contribution to predictions (SHAP)", "Average absolute contribution of each alpha to the prediction, from the boosting library's own tree explanations.", (x) => x.toFixed(3)]]) {
      const values = imp[key]; if (!values || values.available === false) continue;
      const names = imp.alphas; chartCard(ch, title, note, { draw: (b) => C.hbars(b, { rows: names.map((n) => ({ label: n, value: values[n] })).sort((a, b2) => b2.value - a.value), format: fmt2, name: "Importance", label: title, labelWidth: 190, valueWidth: 54, color: () => css("--s3") }),
        table: (b) => table(b, [{ label: "Alpha", get: (n) => n }, { label: "Value", num: true, get: (n) => fmt2(values[n]) }], names) });
    }
    const f = W.card(root, "The fits", "Each fit uses only dates far enough back that no training return was still unfolding; its predictions are used until the next fit.");
    table(f, [{ label: "Trained", get: (r) => `${r.train_from} to ${r.train_to}` }, { label: "Predicts", get: (r) => `${r.predict_from} to ${r.predict_to}` }, { label: "Rows", num: true, get: (r) => r.train_rows.toLocaleString("en-US") }, { label: "Settings", get: (r) => Object.entries(r.params || {}).map(([k2, x]) => `${k2}=${x}`).join(", ") || "defaults" }, { label: "IC in its period", num: true, get: (r) => sgn(r.oos_ic) }], d.folds);
    FD.tearSheet(root, d.evaluation, {});
  }
  function backtestView(root, r) {
    Q.labHead(root, `Backtest: ${r.signal.label}`, `${r.summary.start} to ${r.summary.end} · ${r.portfolio.side === "long_short" ? "long and short" : "long only"} · ${r.portfolio.selection === "top_n" ? "top " + r.portfolio.n : "top " + r.portfolio.percentile + "%"} · ${r.portfolio.weighting.replace("_", " ")} weights`, GUIDE);
    watch(root, r.warnings);
    const s = r.summary, b = r.benchmark, dsr = r.deflated_sharpe;
    const v = h("div", "card verdict-card " + (s.sharpe > b.sharpe && s.alpha_ann > 0 ? "mixed" : "bad"), root); v.dataset.role = "fund-backtest-verdict";
    const head = h("div", "vhead", v); h("span", "vicon", head, s.alpha_ann > 0 ? "!" : "✗"); h("h2", "", head, s.alpha_ann > 0 ? "Earned more than owning the universe in this sample" : "Did not earn more than owning the universe in this sample");
    h("p", "", v, r.reading);
    const k = h("div", "earnrow", root);
    tile(k, "Net Sharpe", num(s.sharpe, 2), { text: `gross ${num(r.gross_summary.sharpe, 2)} · benchmark ${num(b.sharpe, 2)}` }); tile(k, "Return a year", pct(s.cagr, 1), { text: `volatility ${pct(s.ann_vol, 1)} · benchmark ${pct(b.cagr, 1)}` });
    tile(k, "Alpha against the universe", pct(s.alpha_ann, 1), { text: `beta ${num(s.beta_to_benchmark, 2)} · information ratio ${num(s.information_ratio, 2)}`, dir: s.alpha_ann > 0 ? "up" : "down" }, "", "Return a year not explained by owning every eligible company equally");
    tile(k, "Worst fall", pct(s.max_drawdown, 1), { text: `Calmar ${num(s.calmar, 2)}` }); tile(k, "Turnover a year", num(r.costs.annual_turnover, 1) + "×", { text: `trading ${num(r.costs.trading_bps_a_year, 0)} bps · borrow ${num(r.costs.borrow_bps_a_year, 0)} bps a year` });
    tile(k, "Deflated Sharpe", dsr ? num(dsr.probability, 2) : "n/a", { text: dsr ? `${dsr.n_trials} idea${dsr.n_trials === 1 ? "" : "s"} tried; 0.95 is the bar` : "", dir: dsr && dsr.probability >= 0.95 ? "up" : "" }, "", "The probability the Sharpe ratio is above what the best of this many tries would show by luck alone");
    const ch = h("div", "charts", root), eq = r.series.equity, dates = eq.map((p) => p[0]);
    lineCard(ch, "Growth of 1", "Net of costs and borrowing, from the first day the book was held, against owning every eligible company equally and rebalancing daily.", dates,
      [{ name: "This alpha, net", color: css("--s1"), values: eq.map((p) => p[1]) }, { name: "Gross", color: css("--s3"), values: align(dates, r.series.gross_equity) }, { name: "All eligible companies", color: css("--s2"), values: align(dates, r.series.benchmark) }], (x) => x.toFixed(2), { wide: true, baseline: 1, every: 21 });
    lineCard(ch, "Fall from the peak", "How far below its best the net book has been.", r.series.drawdown.map((p) => p[0]), [{ name: "Drawdown", color: css("--neg"), values: r.series.drawdown.map((p) => p[1]) }], (x) => (x * 100).toFixed(0) + "%", { every: 21 });
    lineCard(ch, "Exposure", "Gross exposure of the book (long plus short, as a share of the account).", r.series.exposure.map((p) => p[0]), [{ name: "Gross exposure", color: css("--s4"), values: r.series.exposure.map((p) => p[1]) }], (x) => x.toFixed(2), { every: 21, include: 0 });
    const cost = W.card(root, "What costs do to it", `Breakeven: the one-way cost at which the net return would be zero is ${num(r.costs.breakeven_bps, 0)} bps. A strategy whose breakeven is below a plausible real-world cost is dead however good its gross numbers look.`);
    cost.dataset.role = "fund-costs";
    table(cost, [{ label: "One-way cost (bps)", num: true, get: (x) => String(x.cost_bps) }, { label: "Net return a year", num: true, get: (x) => pct(x.net_ann_return, 1) }, { label: "Net Sharpe", num: true, get: (x) => num(x.net_sharpe, 2) }, { label: "Cost a year", num: true, get: (x) => pct(x.annual_cost_drag, 2) }], r.costs.sensitivity);
    const blocks = Object.entries(r.blocks || {});
    if (blocks.length) {
      const bc = W.card(root, "On the dates the alpha was found on", "Training and validation are the dates the search could see; the locked block is the one it was tested on once. The honest number is the locked one.");
      bc.dataset.role = "fund-blocks";
      table(bc, [{ label: "Block", get: (x) => human(x[0]) }, { label: "From", get: (x) => x[1].start }, { label: "To", get: (x) => x[1].end }, { label: "Days", num: true, get: (x) => String(x[1].n_obs || 0) }, { label: "Net Sharpe", num: true, get: (x) => num(x[1].sharpe, 2) }, { label: "Return a year", num: true, get: (x) => pct(x[1].cagr, 1) }], blocks);
    }
    const lim = r.limits, lc = W.card(root, "How the limits bit", "On how many rebalance dates the limits changed the book, and how often it could not be fully invested.");
    lc.dataset.role = "fund-limits";
    table(lc, [{ label: "Measure", get: (x) => x[0] }, { label: "Value", num: true, get: (x) => x[1] }], [["Rebalance dates", String(lim.dates)], ["Dates the limits changed the book", String(lim.projected_dates)], ["Dates not fully invested", String(lim.dates_not_fully_invested)], ["Dates held in cash", String(lim.dates_in_cash)],
      ["Dates a turnover limit applied", String(lim.turnover_limited_dates)], ["Average companies, long leg", num(lim.average_names_long, 1)], ["Average companies, short leg", num(lim.average_names_short, 1)], ["Largest position held", pct(lim.max_position_held, 1)],
      ["Largest net beta at a rebalance", lim.max_net_beta === null || lim.max_net_beta === undefined ? "not limited" : num(lim.max_net_beta, 2)]]);
    const hold = W.card(root, `Latest book (${r.as_of})`, "The positions decided at the last rebalance. The sign is the side.");
    hold.dataset.role = "fund-holdings";
    table(hold, [{ label: "Ticker", get: (x) => x.ticker }, { label: "Company", get: (x) => x.name }, { label: "Weight", num: true, get: (x) => spct(x.weight, 2) }, { label: "Sector", get: (x) => x.sector || "" }, { label: "Country", get: (x) => x.country || "" }], r.holdings);
    h("p", "hint", root, `Recorded as experiment ${r.experiment_id}: the signal, the portfolio rules, the universe and the data it ran on, so it can be shown and run again.`);
  }

  /* =========================================================== Alpha monitor */
  const MN = FD.views.monitor = {};
  MN.run = () => FD.act("monitor", "Monitoring…", async (run) => { const { result } = await run.job("/api/fund/monitor", {}, "Checking the alphas"); LB.stale(); return result; });
  MN.empty = (root) => {
    const c = h("div", "card empty", root); h("h2", "", c, "Is it still working?");
    for (const t of ["An alpha that worked can stop working: it is crowded out, the market changes, or the data behind it changes. The monitor recomputes every alpha that is not retired on the data now in the database and compares its recent information coefficient with the one it earned when it was found.",
      "Healthy: recent IC at least half the original. Weakening: below half. Decaying: zero or negative over a year, or under a quarter of the original over two. Broken: reversed with evidence, or no longer computable for most companies. A validated or production alpha judged Decaying or Broken is moved to Decaying; nothing is deleted."]) h("p", "", c, t);
    const go = h("button", "btn primary", c, "Run the monitor now"); go.type = "button"; go.addEventListener("click", () => MN.run());
  };
  MN.sections = (panel) => { const b = FD.sec(panel, "What it checks"); h("p", "hint", b, "Every alpha in the library that is not retired or rejected, on the whole universe it was found on, up to the latest date in the database. Fetch new statements and prices in the Data manager first if you want it to see recent months."); };
  MN.build = (root, d) => {
    Q.labHead(root, "Alpha monitor", `${d.checked} alpha${d.checked === 1 ? "" : "s"} checked`, GUIDE);
    const counts = { Healthy: 0, Weakening: 0, Decaying: 0, Broken: 0 }; for (const r of d.results) counts[r.monitor] = (counts[r.monitor] || 0) + 1;
    const k = h("div", "earnrow", root);
    for (const [name, n] of Object.entries(counts)) tile(k, STATE_WORD[name] || name, String(n), {}, n ? "" : "muted");
    const card = W.card(root, "State of each alpha", d.note); card.dataset.role = "fund-monitor";
    if (!d.results.length) { h("p", "hint", card, "The library has no alpha to check."); return; }
    table(card, [{ label: "Formula", cls: "formula", get: (r) => r.text }, { label: "State", get: (r) => `${r.monitor === "Healthy" ? "✓" : r.monitor === "Weakening" ? "!" : "✗"} ${r.monitor}` }, { label: "Status", get: (r) => r.status }, { label: "Recent IC", num: true, get: (r) => sgn(r.detail.recent_ic) },
      { label: "Original IC", num: true, get: (r) => sgn(r.detail.original_ic) }, { label: "Recent as share of original", num: true, get: (r) => (Q.bad(r.detail.ratio) ? "" : pct(r.detail.ratio, 0)) }, { label: "Recent t", num: true, get: (r) => num(r.detail.recent_t, 1) },
      { label: "Note", get: (r) => r.detail.error || r.detail.note || "" }], d.results);
  };
  const STATE_WORD = { Healthy: "Healthy", Weakening: "Weakening", Decaying: "Decaying", Broken: "Broken" };

})();
