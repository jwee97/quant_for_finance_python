/* The Research comparison: saved results side by side, measured over the days they all share.

   The server does the arithmetic (src/research/compare.py): it measures every result again over the common window, names the differences in assumptions that would make a ranking misleading and
   withholds the ranking when one of them does. This page only draws what comes back: the same x-axis for every chart (one shared window, set here), the issues first, then the numbers. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, toast, css, download } = Q, C = Q.C, R = Q.research, B = Q.badges;
  const pct = (v, d = 1) => (v === null || v === undefined ? "n/a" : Q.pct(v, d)), num = (v, d = 2) => (v === null || v === undefined ? "n/a" : Q.num(v, d));
  const root = $("#rcompare");
  const S = { picked: new Set(), bench: "", start: "", end: "", rf: "", result: null, list: [], group: null, seq: 0 };
  const colorOf = (i) => css(`--s${(i % 8) + 1}`);

  async function load(args) {
    const my = ++S.seq;
    root.replaceChildren(); const d = h("div", "head", root), l = h("div", "", d); h("h2", "", l, "Research comparison");
    h("p", "sub", l, "Strategies, models and regime estimates side by side. Everything is measured over the days all of them share; when their assumptions differ in a way that makes a ranking misleading, no ranking is shown.");
    if (!R.ready()) { Q.loading(root, "Loading…"); return; }
    if (!R.active()) { h("div", "card empty", root).append(h("h3", "", null, "Open a research session"), h("p", "", null, "Results are compared from a session. Save a backtest, an experiment or a regime estimate into one first.")); return; }
    const sid = args.session || R.active().id;
    if (args.session && (!R.active() || R.active().id !== args.session)) { await R.open(args.session).catch((e) => toast(e.message)); }
    let det; try { det = await api("/api/research/session?id=" + sid); } catch (e) { h("div", "banner err", root, e.message); return; }
    if (my !== S.seq) return;
    S.list = det.artifacts; S.group = null;
    if (args.group) { S.group = det.session.groups.find((g) => g.id === args.group) || null; if (S.group) { S.picked = new Set(S.group.artifacts); S.bench = S.group.benchmark || ""; S.start = S.group.start || ""; S.end = S.group.end || ""; } }
    if (args.ids) S.picked = new Set(args.ids.split(",").filter((i) => S.list.some((a) => a.id === i)));
    if (args.add && S.list.some((a) => a.id === args.add)) S.picked.add(args.add);
    S.picked = new Set([...S.picked].filter((i) => S.list.some((a) => a.id === i)));
    picker(det.session); const out = h("div", "stack", root); out.id = "rcompare-results"; out.setAttribute("aria-live", "polite");
    if (S.picked.size) await run(out, det.session); else h("p", "empty-note", out, "Select at least one result above.");
  }

  function picker(session) {
    const card = h("div", "card", root); card.dataset.role = "compare-picker"; h("h3", "", card, `Results in "${session.name}"`);
    if (!S.list.length) { h("p", "empty-note", card, "No results saved in this session."); return; }
    const t = h("div", "scroll", card), tb = h("table", "data", t), hd = h("tr", "", h("thead", "", tb)); for (const c of ["", "Result", "Kind", "Version", "Evidence", "Data", "Period", ""]) h("th", "", hd, c);
    const body = h("tbody", "", tb);
    for (const a of S.list) {
      const tr = h("tr", "", body), c0 = h("td", "", tr), cb = h("input", "", c0); cb.type = "checkbox"; cb.checked = S.picked.has(a.id); cb.dataset.id = a.id; cb.setAttribute("aria-label", "compare " + a.title);
      cb.addEventListener("change", () => { cb.checked ? S.picked.add(a.id) : S.picked.delete(a.id); });
      tr.dataset.id = a.id; h("td", "", tr, a.title); h("td", "", tr, a.kind.replace("_", " ")); h("td", "", tr, a.version && a.version > 1 ? `v${a.version}` : "v1"); B.evidence(h("td", "", tr), a.evidence); B.data(h("td", "", tr), a.data_kind); h("td", "", tr, `${a.context.start || "?"} to ${a.context.end || "?"}`);
      const act = h("td", "", tr), op = h("button", "btn small", act, "Open"); op.type = "button"; op.dataset.action = "open-study"; op.setAttribute("aria-label", "open " + a.title); op.addEventListener("click", () => openStudy(a.id));
      const rr = h("button", "btn small", act, "Run again"); rr.type = "button"; rr.dataset.action = "rerun-study"; rr.title = "Runs the request that produced this study again with the current data, and saves the result as a new version beside it"; rr.setAttribute("aria-label", "run again " + a.title); rr.addEventListener("click", () => rerun(a, session));
    }
    const note = h("div", "stack", root); note.id = "rcompare-status"; note.setAttribute("aria-live", "polite");
    const viewer = h("div", "stack", root); viewer.id = "rcompare-open";
    const row = h("div", "row", card);
    const sel = (label, key, opts) => { const l = h("label", "f", row, label), s = h("select", "", l); s.dataset.key = key; for (const [v, t2] of opts) { const o = h("option", "", s, t2); o.value = v; if (S[key] === v) o.selected = true; } s.addEventListener("change", () => { S[key] = s.value; }); return s; };
    sel("Reference", "bench", [["", "None"], ...S.list.filter((a) => a.kind !== "regime").map((a) => [a.id, a.title])]);
    const inp = (label, key, type) => { const l = h("label", "f", row, label), i = h("input", "", l); i.type = type; i.value = S[key]; i.dataset.key = key; i.addEventListener("input", () => { S[key] = i.value; }); return i; };
    inp("From", "start", "date"); inp("To", "end", "date"); const rf = inp("Risk-free (% a year)", "rf", "number"); rf.step = "any";
    const go = h("button", "btn primary", row, "Compare"); go.type = "button"; go.addEventListener("click", () => { S.seq++; const out = $("#rcompare-results"); out.replaceChildren(); if (S.picked.size) run(out, session); else h("p", "empty-note", out, "Select at least one result above."); });
  }

  const body = (session) => ({ artifacts: [...S.picked], benchmark: S.bench || undefined, start: S.start || undefined, end: S.end || undefined, risk_free: S.rf === "" ? undefined : Number(S.rf) / 100 });

  async function run(out, session) {
    h("p", "hint", out, "Comparing…");
    let r; try { r = await api("/api/research/compare", body(session)); } catch (e) { out.replaceChildren(); h("div", "banner err", out, e.message); return; }
    out.replaceChildren(); S.result = r;
    const tools = h("div", "row", out); tools.dataset.role = "compare-tools";
    const mk = (t, fn) => { const b = h("button", "btn", tools, t); b.type = "button"; b.addEventListener("click", async () => { b.disabled = true; try { await fn(); } catch (e) { toast(e.message); } b.disabled = false; }); return b; };
    mk("Save this comparison", async () => { const name = prompt("Name for this comparison", S.group ? S.group.name : "Comparison"); if (!name) return; const g = (await api("/api/research/group/save", { session: session.id, id: S.group ? S.group.id : undefined, name, artifacts: [...S.picked], benchmark: S.bench, start: S.start, end: S.end })).group; S.group = g; toast("Comparison saved to the session"); });
    mk("Download metrics (CSV)", async () => { const x = await api("/api/research/export", { ...body(session), format: "csv" }); download(x.filename, x.text); });
    mk("Download report (Markdown)", async () => { const x = await api("/api/research/export", { ...body(session), format: "markdown", session: S.group ? session.id : undefined, group: S.group ? S.group.id : undefined }); download(x.filename, x.text); });
    for (const m of r.missing_series) h("div", "banner", out, `${(r.artifacts.find((a) => a.id === m) || {}).title || m} has no return series and is listed but not measured.`);
    issues(out, r);
    studiesBlock(out, r);
    if (!r.window || !Object.keys(r.metrics).length) return;
    const title = (id) => (r.artifacts.find((a) => a.id === id) || {}).title || id, ids = Object.keys(r.metrics);
    const wnd = h("p", "hint", out, `Shared window: ${r.window.start} to ${r.window.end} (${r.series.dates.length} dates every result observed; the calendar is ${r.conventions ? r.conventions.common_dates.calendar.replace(/_/g, " ") : "unknown"}). Ratios use a risk-free rate of ${pct(r.risk_free)} a year, accrued by elapsed time.`); wnd.dataset.role = "compare-window";
    metricsTable(out, r, ids, title);
    conventions(out, r, title);
    const dates = r.series.dates, colorFor = Object.fromEntries(ids.map((id, i) => [id, colorOf(i)]));
    const lines = (k) => ids.map((id) => ({ name: title(id), color: colorFor[id], values: r.series[k][id] }));
    const rowsFor = (k, f) => (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => dates[i] }, ...ids.map((id) => ({ label: title(id), num: true, get: (i) => f(r.series[k][id][i]) }))], Q.sampleRows(dates, null, Math.max(1, Math.ceil(dates.length / 60))));
    const chart = (name, role, k, o, f) => { const c = Q.chartCard(out, name, "One window for every chart: change From and To above to move all of them together.", { wide: true, draw: (b) => C.line(b, { dates, series: lines(k), height: 260, sync: "compare", ...o }), table: rowsFor(k, f) }); c.dataset.role = role;
      const ex = h("button", "btn", c._extra, "CSV"); ex.type = "button"; ex.addEventListener("click", () => { const head = ["date", ...ids.map(title)].join(","), rows = dates.map((d, i) => [d, ...ids.map((id) => r.series[k][id][i] ?? "")].join(",")); download(`${role}.csv`, [head, ...rows].join("\n")); }); };
    chart("Growth of 1 (rebased to the start of the window)", "compare-equity", "equity", { yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), label: "Equity curves over the shared window" }, (v) => num(v, 3));
    chart("Drawdown", "compare-drawdown", "drawdown", { area: false, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v), baseline: 0, ceil: 0, label: "Drawdowns" }, (v) => pct(v));
    chart(`Rolling Sharpe (${r.series.roll_window} observations)`, "compare-rolling-sharpe", "rolling_sharpe", { baseline: 0, include: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), label: "Rolling Sharpe ratios" }, (v) => num(v));
    chart(`Rolling volatility (${r.series.roll_window} observations)`, "compare-rolling-vol", "rolling_vol", { yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v), label: "Rolling volatilities" }, (v) => pct(v));
    if (r.distribution) { const dc = Q.chartCard(out, "Distribution of daily returns", "The same bins for every result.", { wide: true, draw: (b) => C.histogram(b, { series: ids.map((id) => ({ name: title(id), edges: r.distribution.edges, counts: r.distribution.counts[id], color: colorFor[id] })), xFormat: (v) => (v * 100).toFixed(1) + "%", unit: r.distribution.edges[1] - r.distribution.edges[0], yFormat: (v) => (v * 100).toFixed(0) + "%", height: 220, label: "Distribution of daily returns" }), table: (b) => Q.simpleTable(b, [{ label: "Bin from", num: true, get: (i) => pct(r.distribution.edges[i], 2) }, ...ids.map((id) => ({ label: title(id), num: true, get: (i) => String(r.distribution.counts[id][i]) }))], r.distribution.counts[ids[0]].map((_, i) => i)) }); dc.dataset.role = "compare-distribution"; }
    corr(out, r, title);
    if (r.versus_reference) versus(out, r, title);
    provenance(out, r);
  }

  /* Saved studies that have no return series are compared on their assumptions and on what they report about themselves; predictive quality is never mixed with trading performance. */
  function studiesBlock(out, r) {
    const st = r.studies; if (!st) return;
    const title = (id) => (r.artifacts.find((a) => a.id === id) || {}).title || id, ids = r.artifacts.map((a) => a.id);
    if (ids.length > 1 || st.attribution.length) {
      const card = h("div", "card", out); card.dataset.role = "compare-assumptions"; h("h3", "", card, "Assumptions side by side");
      h("p", "hint", card, st.assumptions.differing.length ? `They differ in: ${st.assumptions.differing.map((k) => st.assumptions.rows.find((x) => x.key === k).label.toLowerCase()).join(", ")}.` : "No saved assumption differs.");
      Q.simpleTable(card, [{ label: "", get: (x) => x.label }, ...ids.map((id) => ({ label: `${title(id)}${(r.artifacts.find((a) => a.id === id) || {}).version > 1 ? " (v" + r.artifacts.find((a) => a.id === id).version + ")" : ""}`, get: (x) => { const v = st.assumptions.values[id][x.key]; return v === "" || v === undefined ? "not recorded" : String(v); } })), { label: "", get: (x) => (st.assumptions.differing.includes(x.key) ? "differs" : "") }], st.assumptions.rows);
    }
    const preds = Object.entries(st.predictive.studies);
    if (preds.length) {
      const card = h("div", "card", out); card.dataset.role = "compare-predictive"; h("h3", "", card, "Predictive quality (not trading performance)");
      h("p", "hint", card, st.predictive.note);
      Q.simpleTable(card, [{ label: "Study", get: (x) => title(x[0]) }, { label: "Kind", get: (x) => x[1].kind.replace("_", " ") }, { label: "Headline", num: true, get: (x) => (x[1].headline === null || x[1].headline === undefined ? "n/a" : num(x[1].headline, 4)) }, { label: "What the headline is", get: (x) => x[1].headline_means },
        { label: "Rank", num: true, get: (x) => (st.predictive.ranking ? String(st.predictive.ranking.indexOf(x[0]) + 1) : "not ranked") }], preds);
      if (st.predictive.blocked_because.length) { const b = h("div", "issue blocks_ranking", card); b.dataset.role = "predictive-blocked"; h("b", "", b, "Stops a ranking"); b.appendChild(document.createTextNode("These studies are not ranked against each other: " + st.predictive.blocked_because.join("; ") + ".")); }
    }
    if (st.attribution.length) {
      const card = h("div", "card", out); card.dataset.role = "compare-attribution"; h("h3", "", card, "Attribution differences");
      const lines = [...new Set(st.attribution.flatMap((a) => Object.keys(a.factor_totals || {})))];
      Q.simpleTable(card, [{ label: "Study", get: (a) => title(a.id) }, { label: "Factor model", get: (a) => a.kind || "n/a" }, ...lines.map((l) => ({ label: l, num: true, get: (a) => (a.factor_totals && a.factor_totals[l] !== undefined ? pct(a.factor_totals[l], 2) : "") })),
        { label: "Brinson (allocation / selection)", num: true, get: (a) => (a.brinson_totals ? `${pct(a.brinson_totals.allocation, 2)} / ${pct(a.brinson_totals.selection, 2)}` : "not made") }, { label: "Reconciliation", get: (a) => a.factor_status || "n/a" }], st.attribution);
      h("p", "hint", card, "Lines are compared only between studies that used the same factor model (a warning above says when they did not).");
    }
    if (st.unmeasured.length) {
      const card = h("div", "card", out); card.dataset.role = "compare-unmeasured"; h("h3", "", card, "Not measured over the shared window");
      const ul = h("ul", "", card); for (const u of st.unmeasured) h("li", "", ul, `${title(u.id)}: ${u.reason}.`);
    }
  }

  async function openStudy(id) {
    const box = $("#rcompare-open"); if (!box) return; box.replaceChildren();
    let a, vs; try { a = await api("/api/research/artifact?id=" + id); vs = (await api("/api/research/artifact/versions?id=" + id)).versions; } catch (e) { h("div", "banner err", box, e.message); return; }
    const card = h("div", "card", box); card.dataset.role = "study-viewer"; h("h3", "", card, `${a.title}${a.version > 1 ? " (version " + a.version + ")" : ""}`);
    if (a.migrated_from) h("p", "hint", card, `Saved by an older version of the app (schema ${a.migrated_from}); shown as the current schema. Its file has not been changed.`);
    const dl = h("dl", "kv", card), add = (k, v) => { if (v === undefined || v === null || v === "") return; h("dt", "", dl, k); h("dd", "", dl, String(v)); };
    add("Kind", a.kind.replace("_", " ")); add("Evidence", a.evidence); add("Data", a.data_kind); add("Instruments", (a.context.assets || []).join(", ")); add("Dates", `${a.context.start || "?"} to ${a.context.end || "?"}`); add("Benchmark", a.benchmark);
    add("Costs", Object.entries(a.costs || {}).map(([k, v]) => `${k} ${v}`).join("; ")); add("Hypothesis", a.hypothesis); add("Created", (a.created || "").slice(0, 19).replace("T", " ")); add("Code version", (a.refs || {}).code_version); add("Data version", (a.refs || {}).dataset);
    for (const [k, v] of Object.entries(a.metrics || {})) add("Metric: " + k.replace(/_/g, " "), typeof v === "number" ? num(v, 4) : v);
    if (a.notes && a.notes.length) { h("h4", "", card, "Notes"); const ul = h("ul", "", card); for (const n of a.notes) h("li", "", ul, n); }
    if (a.validation && Object.keys(a.validation).length) { h("h4", "", card, "Validation"); const ul = h("ul", "", card); for (const [k, v] of Object.entries(a.validation)) if (v !== null && typeof v !== "object") h("li", "", ul, `${k.replace(/_/g, " ")}: ${v}`); }
    if (vs.length > 1) { h("h4", "", card, "Versions"); const ul = h("ul", "", card); ul.dataset.role = "study-versions"; for (const v of vs) { const li = h("li", "", ul); li.dataset.id = v.id; const btn = h("button", "linklike", li, `v${v.version}: ${v.title}`); btn.type = "button"; btn.addEventListener("click", () => openStudy(v.id)); if (v.id === a.id) li.append(" (open)"); } }
    if (a.attribution && Q.alphaResearch) {
      const at = a.attribution, wrap = { kind: at.kind, benchmark: at.benchmark, methodology: at.methodology, brinson_reason: at.brinson_reason, methods: { [at.method || "study"]: { factor: at.factor, brinson: at.brinson } } };
      Q.alphaResearch.attributionView(box, wrap, { [at.method || "study"]: at.method || a.title }, { state: { key: at.method || "study" } });
    }
  }

  async function rerun(a, session) {
    const say = (txt, cls) => { const box = $("#rcompare-status"); if (!box) return; box.replaceChildren(); const d = h("div", cls || "banner", box, txt); d.dataset.role = "rerun-status"; d.setAttribute("role", "status"); };
    say("");
    let spec; try { spec = await api("/api/research/artifact/rerun?id=" + a.id); } catch (e) { say("Cannot run again: " + e.message, "banner err"); return; }
    say(`Running "${a.title}" again with the current data…`);
    try {
      const { job } = await api(spec.route, spec.body);
      await Q.waitJob(job, (v) => say(`${v.message} (${Math.round(v.elapsed)} s)`), 1000);
      const full = await api("/api/research/artifact?id=" + a.id);
      const saved = await R.saveStudy({ job, ...((full.config || {}).method ? { method: full.config.method } : {}) }, { parent: a.id, session: a.session_id || session.id, title: `${a.title.replace(/ \(re-run.*$/, "")} (re-run)` });
      await load({ session: a.session_id || session.id });                   // the list now has the new version; the page is drawn once, after the save
      say(saved ? `Saved as version ${saved.version}. The earlier study is unchanged.` : "Not saved: the new run is not kept unless you save it.");
    } catch (e) { say("The new run failed: " + e.message, "banner err"); }
  }

  function issues(out, r) {
    const card = h("div", "card", out); card.dataset.role = "compare-issues"; h("h3", "", card, "Differences in assumptions");
    if (!r.issues.length) { h("p", "empty-note", card, "None found among the saved assumptions. That is not proof the results are alike: only what was saved can be compared."); return; }
    const label = { blocks_ranking: "Stops a ranking", warning: "Warning", note: "Note" };
    for (const i of r.issues) { const d = h("div", "issue " + i.severity, card); h("b", "", d, label[i.severity] || i.severity); d.appendChild(document.createTextNode(i.message)); d.dataset.code = i.code; }
  }

  function metricsTable(out, r, ids, title) {
    const card = h("div", "card", out); card.dataset.role = "compare-metrics"; h("h3", "", card, "Metrics over the dates all results share");
    const verdict = h("p", r.ranking ? "sub" : "hint", card, r.reason); verdict.dataset.role = "compare-verdict";
    const rows = [["cagr", "CAGR", pct], ["ann_vol", "Volatility (annual)", pct], ["sharpe", "Sharpe", num], ["sortino", "Sortino", num], ["max_drawdown", "Max drawdown", pct], ["calmar", "Calmar", num], ["total_return", "Total return", pct], ["turnover_annual", "Turnover (annual)", (v) => num(v, 1)], ["cost_drag_annual", "Costs (annual)", (v) => pct(v, 2)], ["n_days", "Days", (v) => (v === undefined ? "n/a" : String(v))]];
    const t = h("div", "scroll", card), tb = h("table", "data", t), hd = h("tr", "", h("thead", "", tb)); h("th", "", hd, "Metric");
    ids.forEach((id) => { const th = h("th", "num", hd, title(id)); const rank = r.ranking ? r.ranking.indexOf(id) + 1 : 0; if (rank) th.title = `Ranked ${rank} by ${(r.rank_by || {}).label || "Sharpe"}`; });
    const body = h("tbody", "", tb);
    const ev = h("tr", "", body); h("td", "", ev, "Evidence"); ids.forEach((id) => B.evidence(h("td", "num", ev), (r.artifacts.find((a) => a.id === id) || {}).evidence));
    const dk = h("tr", "", body); h("td", "", dk, "Data"); ids.forEach((id) => B.data(h("td", "num", dk), (r.artifacts.find((a) => a.id === id) || {}).data_kind));
    if (r.ranking) { const rk = h("tr", "", body); h("td", "", rk, "Rank by " + ((r.rank_by || {}).label || "Sharpe") + ((r.rank_by || {}).direction === "lower" ? " (lowest first)" : "")); ids.forEach((id) => h("td", "num", rk, String(r.ranking.indexOf(id) + 1))); }
    for (const [key, label, f] of rows) { const tr = h("tr", "", body); h("td", "", tr, label); ids.forEach((id) => h("td", "num", tr, f(r.metrics[id][key]))); }
  }

  function conventions(out, r, title) {
    if (!r.conventions) return;
    const c = r.conventions, g = c.common_dates, card = h("details", "card", out); card.dataset.role = "compare-conventions"; h("summary", "", card, "How these numbers were measured: calendar, annualisation and coverage");
    const dl = h("dl", "kv", card), add = (k, v) => { h("dt", "", dl, k); h("dd", "", dl, v); };
    add("Common dates", `${g.n_dates} dates, ${g.calendar.replace(/_/g, " ")}, ${g.frequency}`);
    add("Annualisation", g.basis === "periods" ? `${g.periods_per_year} periods a year (${g.source})` : `by elapsed calendar time (${g.source})`);
    add("CAGR", c.cagr); add("Volatility", c.volatility); add("Risk-free rate", c.risk_free);
    const ids = Object.keys(r.coverage || {});
    if (ids.length) {
      h("h4", "", card, "Coverage of each result");
      Q.simpleTable(card, [{ label: "Result", get: (i) => title(i) }, { label: "Own dates", get: (i) => `${r.coverage[i].first} to ${r.coverage[i].last}` }, { label: "Calendar", get: (i) => `${r.coverage[i].calendar.replace(/_/g, " ")}, ${r.coverage[i].frequency}` },
        { label: "Observations in window", num: true, get: (i) => String(r.coverage[i].in_window) }, { label: "Left out (not on every result)", num: true, get: (i) => String(r.coverage[i].dropped_in_window) }, { label: "Longest gap (days)", num: true, get: (i) => String(r.coverage[i].longest_gap_days) }], ids);
    }
    const own = Object.entries(r.individual || {});
    if (own.length) {
      h("h4", "", card, "Each result on its own whole history");
      h("p", "hint", card, "Measured under the result's own calendar over every date it has. These numbers describe one result; they are not comparable with another's because the years differ.");
      Q.simpleTable(card, [{ label: "Result", get: (q) => title(q[0]) }, { label: "Dates", get: (q) => `${q[1].first} to ${q[1].last}` }, { label: "CAGR", num: true, get: (q) => pct(q[1].cagr) }, { label: "Volatility", num: true, get: (q) => pct(q[1].ann_vol) }, { label: "Sharpe", num: true, get: (q) => num(q[1].sharpe) }, { label: "Max drawdown", num: true, get: (q) => pct(q[1].max_drawdown) }], own);
    }
  }

  function corr(out, r, title) {
    if (r.correlation.ids.length < 2) return;
    const card = h("div", "card", out); card.dataset.role = "compare-correlation"; h("h3", "", card, "Correlation of daily returns"); h("p", "sub", card, "Results that move together add little when combined.");
    const v = r.correlation.values, ids = r.correlation.ids;
    Q.simpleTable(card, [{ label: "", get: (i) => title(ids[i]) }, ...ids.map((id, j) => ({ label: title(id), num: true, get: (i) => num(v[i][j]) }))], ids.map((_, i) => i));
  }

  function versus(out, r, title) {
    const card = h("div", "card", out); card.dataset.role = "compare-reference"; h("h3", "", card, `Against the reference: ${title(r.reference)}`);
    const rows = Object.entries(r.versus_reference).map(([id, x]) => ({ id, x }));
    Q.simpleTable(card, [{ label: "Result", get: (q) => title(q.id) }, { label: "Beta", num: true, get: (q) => num(q.x.beta) }, { label: "Alpha (annual)", num: true, get: (q) => pct(q.x.alpha_annual) }, { label: "Tracking error", num: true, get: (q) => pct(q.x.tracking_error) }, { label: "Information ratio", num: true, get: (q) => num(q.x.information_ratio) },
      { label: "Up capture", num: true, get: (q) => num(q.x.up_capture) }, { label: "Down capture", num: true, get: (q) => num(q.x.down_capture) }, { label: "Correlation", num: true, get: (q) => num(q.x.correlation) }], rows);
  }

  function provenance(out, r) {
    const card = h("details", "card", out); card.dataset.role = "compare-provenance"; h("summary", "", card, "References: what each result rests on");
    for (const a of r.artifacts) {
      h("h4", "", card, a.title); const dl = h("dl", "kv", card);
      const ctx = a.context, add = (k, v) => { h("dt", "", dl, k); h("dd", "", dl, v); };
      add("Kind", a.kind.replace("_", " ")); add("Evidence", a.evidence_text); add("Data", a.data_kind); add("Assets", (ctx.assets || []).join(", ") || "not recorded"); add("Dates", `${ctx.start || "?"} to ${ctx.end || "?"}`); add("Source", ctx.data_source || "not recorded"); add("Dataset version", ctx.dataset_version || "not recorded");
      for (const [k, v] of Object.entries(a.refs)) add(k.replace(/_/g, " "), v);
    }
  }

  Q.registerPage("rcompare", { render: (args) => load(args || {}) });
  R.onChange(() => { if (Q.currentTab() === "rcompare") load(Q.route().args); });
})();
