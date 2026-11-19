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
    const t = h("div", "scroll", card), tb = h("table", "data", t), hd = h("tr", "", h("thead", "", tb)); for (const c of ["", "Result", "Kind", "Evidence", "Data", "Period"]) h("th", "", hd, c);
    const body = h("tbody", "", tb);
    for (const a of S.list) {
      const tr = h("tr", "", body), c0 = h("td", "", tr), cb = h("input", "", c0); cb.type = "checkbox"; cb.checked = S.picked.has(a.id); cb.dataset.id = a.id; cb.setAttribute("aria-label", "compare " + a.title);
      cb.addEventListener("change", () => { cb.checked ? S.picked.add(a.id) : S.picked.delete(a.id); });
      h("td", "", tr, a.title); h("td", "", tr, a.kind.replace("_", " ")); B.evidence(h("td", "", tr), a.evidence); B.data(h("td", "", tr), a.data_kind); h("td", "", tr, `${a.context.start || "?"} to ${a.context.end || "?"}`);
    }
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
    if (!r.window || !Object.keys(r.metrics).length) return;
    const title = (id) => (r.artifacts.find((a) => a.id === id) || {}).title || id, ids = Object.keys(r.metrics);
    const wnd = h("p", "hint", out, `Shared window: ${r.window.start} to ${r.window.end} (${r.series.dates.length} days). Ratios use a risk-free rate of ${pct(r.risk_free)} a year.`); wnd.dataset.role = "compare-window";
    metricsTable(out, r, ids, title);
    const dates = r.series.dates, colorFor = Object.fromEntries(ids.map((id, i) => [id, colorOf(i)]));
    const lines = (k) => ids.map((id) => ({ name: title(id), color: colorFor[id], values: r.series[k][id] }));
    const rowsFor = (k, f) => (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => dates[i] }, ...ids.map((id) => ({ label: title(id), num: true, get: (i) => f(r.series[k][id][i]) }))], Q.sampleRows(dates, null, Math.max(1, Math.ceil(dates.length / 60))));
    const chart = (name, role, k, o, f) => { const c = Q.chartCard(out, name, "One window for every chart: change From and To above to move all of them together.", { wide: true, draw: (b) => C.line(b, { dates, series: lines(k), height: 260, sync: "compare", ...o }), table: rowsFor(k, f) }); c.dataset.role = role;
      const ex = h("button", "btn", c._extra, "CSV"); ex.type = "button"; ex.addEventListener("click", () => { const head = ["date", ...ids.map(title)].join(","), rows = dates.map((d, i) => [d, ...ids.map((id) => r.series[k][id][i] ?? "")].join(",")); download(`${role}.csv`, [head, ...rows].join("\n")); }); };
    chart("Growth of 1 (rebased to the start of the window)", "compare-equity", "equity", { yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), label: "Equity curves over the shared window" }, (v) => num(v, 3));
    chart("Drawdown", "compare-drawdown", "drawdown", { area: false, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v), baseline: 0, ceil: 0, label: "Drawdowns" }, (v) => pct(v));
    chart(`Rolling Sharpe (${r.series.roll_window} days)`, "compare-rolling-sharpe", "rolling_sharpe", { baseline: 0, include: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), label: "Rolling Sharpe ratios" }, (v) => num(v));
    chart(`Rolling volatility (${r.series.roll_window} days)`, "compare-rolling-vol", "rolling_vol", { yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v), label: "Rolling volatilities" }, (v) => pct(v));
    if (r.distribution) { const dc = Q.chartCard(out, "Distribution of daily returns", "The same bins for every result.", { wide: true, draw: (b) => C.histogram(b, { series: ids.map((id) => ({ name: title(id), edges: r.distribution.edges, counts: r.distribution.counts[id], color: colorFor[id] })), xFormat: (v) => (v * 100).toFixed(1) + "%", unit: r.distribution.edges[1] - r.distribution.edges[0], yFormat: (v) => (v * 100).toFixed(0) + "%", height: 220, label: "Distribution of daily returns" }), table: (b) => Q.simpleTable(b, [{ label: "Bin from", num: true, get: (i) => pct(r.distribution.edges[i], 2) }, ...ids.map((id) => ({ label: title(id), num: true, get: (i) => String(r.distribution.counts[id][i]) }))], r.distribution.counts[ids[0]].map((_, i) => i)) }); dc.dataset.role = "compare-distribution"; }
    corr(out, r, title);
    if (r.versus_reference) versus(out, r, title);
    provenance(out, r);
  }

  function issues(out, r) {
    const card = h("div", "card", out); card.dataset.role = "compare-issues"; h("h3", "", card, "Differences in assumptions");
    if (!r.issues.length) { h("p", "empty-note", card, "None found among the saved assumptions. That is not proof the results are alike: only what was saved can be compared."); return; }
    const label = { blocks_ranking: "Stops a ranking", warning: "Warning", note: "Note" };
    for (const i of r.issues) { const d = h("div", "issue " + i.severity, card); h("b", "", d, label[i.severity] || i.severity); d.appendChild(document.createTextNode(i.message)); d.dataset.code = i.code; }
  }

  function metricsTable(out, r, ids, title) {
    const card = h("div", "card", out); card.dataset.role = "compare-metrics"; h("h3", "", card, "Metrics over the shared window");
    const verdict = h("p", r.ranking ? "sub" : "hint", card, r.reason); verdict.dataset.role = "compare-verdict";
    const rows = [["cagr", "CAGR", pct], ["ann_vol", "Volatility (annual)", pct], ["sharpe", "Sharpe", num], ["sortino", "Sortino", num], ["max_drawdown", "Max drawdown", pct], ["calmar", "Calmar", num], ["total_return", "Total return", pct], ["turnover_annual", "Turnover (annual)", (v) => num(v, 1)], ["cost_drag_annual", "Costs (annual)", (v) => pct(v, 2)], ["n_days", "Days", (v) => (v === undefined ? "n/a" : String(v))]];
    const t = h("div", "scroll", card), tb = h("table", "data", t), hd = h("tr", "", h("thead", "", tb)); h("th", "", hd, "Metric");
    ids.forEach((id) => { const th = h("th", "num", hd, title(id)); const rank = r.ranking ? r.ranking.indexOf(id) + 1 : 0; if (rank) th.title = `Ranked ${rank} by Sharpe`; });
    const body = h("tbody", "", tb);
    const ev = h("tr", "", body); h("td", "", ev, "Evidence"); ids.forEach((id) => B.evidence(h("td", "num", ev), (r.artifacts.find((a) => a.id === id) || {}).evidence));
    const dk = h("tr", "", body); h("td", "", dk, "Data"); ids.forEach((id) => B.data(h("td", "num", dk), (r.artifacts.find((a) => a.id === id) || {}).data_kind));
    if (r.ranking) { const rk = h("tr", "", body); h("td", "", rk, "Rank by Sharpe"); ids.forEach((id) => h("td", "num", rk, String(r.ranking.indexOf(id) + 1))); }
    for (const [key, label, f] of rows) { const tr = h("tr", "", body); h("td", "", tr, label); ids.forEach((id) => h("td", "num", tr, f(r.metrics[id][key]))); }
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
