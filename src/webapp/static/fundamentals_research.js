/* The Fundamentals tab's research-reliability views: Data health (per company: what is held, missing, conflicting, unconfirmed, and whether factors can run; corporate events and sector families),
   Screen (rules, ranking, why every other company is out, and the same rules on past dates) and Scenarios (the valuation yardsticks under named assumptions, with a rate-by-growth grid).
   /api/fund/{health,events,family,screen,screen/history,valuation/scenarios}. Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const FD = window.FD, Q = window.QL, W = window.WS;
  if (!FD || !Q || !W || !FD.views) return;
  const h = Q.h, tile = Q.tile, table = Q.simpleTable, pct = Q.pct;
  const HV = FD.views.health = {}, SV = FD.views.screen = {}, VV = FD.views.scenarios = {};
  const at = FD.modes.findIndex((m) => m[0] === "coverage");
  FD.modes.splice(at >= 0 ? at + 1 : FD.modes.length, 0, ["health", "Data health"], ["screen", "Screen"], ["scenarios", "Scenarios"]);
  FD.runLabel.health = "Refresh"; FD.runLabel.screen = "Run the screen"; FD.runLabel.scenarios = "Run the scenarios";
  FD.blurb.health = "For each company: the latest filing and quarter held, missing years, source conflicts, figures nobody confirmed, restatements, and whether the inputs factors need are there.";
  FD.blurb.screen = "Filter and rank companies by factor rules on a date, from what was public then. Every company left out says why: failed, missing, not applicable to its industry, or not investable.";
  FD.blurb.scenarios = "The valuation yardsticks of one company under named assumptions, and a grid of discount rate against growth. Nothing here is a forecast.";
  HV.s = { data: null, events: null }; SV.s = { rules: [{ factor: "earnings_yield", op: ">", value: "0" }, { factor: "roe", op: "top_pct", value: "50" }], asOf: "", rankBy: "", top: 20, within: false, out: null, hist: null, factors: null };
  VV.s = { sid: "", cases: [["base", "0.09", "0.02"], ["bear", "0.11", "0.00"], ["bull", "0.08", "0.03"]], out: null };
  const OPS = [[">", ">"], [">=", "≥"], ["<", "<"], ["<=", "≤"], ["between", "between"], ["top_pct", "top % of"], ["bottom_pct", "bottom % of"]];
  const num = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : Math.abs(v) >= 1e6 ? v.toExponential(2) : Number(v).toFixed(d));
  const real = () => "";
  HV.check = SV.check = VV.check = real;

  /* ---------- data health ---------- */
  HV.run = () => FD.act("health", "Counting what each company holds…", async () => { HV.s.data = await FD.read("/api/fund/health"); HV.s.events = await FD.read("/api/fund/events").catch(() => null); return { kind: "health" }; });
  HV.auto = () => { if (!HV.s.data) HV.run(); };
  HV.empty = (root) => FD.emptyCard(root, "How healthy is the data?", ["Press Refresh to count, for each company, what is held, missing, conflicting and unconfirmed."], { label: "Refresh", fn: HV.run });
  HV.sections = (panel) => { const b = FD.sec(panel, "Reading this"); h("p", "hint", b, "“Attention” names why: missing years, source conflicts (the sources disagree by more than half a per cent), observations the scanner found and nobody confirmed (they are in no ratio, factor or backtest), or fewer than 70% of the inputs the common factors need."); };
  HV.build = (root) => {
    const d = HV.s.data; if (!d) { HV.empty(root); return; }
    Q.labHead(root, "Data health", `${d.n} companies`);
    const k = h("div", "earnrow", root);
    tile(k, "Companies", String(d.n), {}); tile(k, "Needing attention", String(d.needing_attention), { text: "see the status column" }, d.needing_attention ? "" : "muted");
    tile(k, "Unconfirmed observations", String(d.unconfirmed_total), { text: "kept out of research until confirmed" }, d.unconfirmed_total ? "" : "muted");
    const card = W.card(root, "Companies", "Sorted with the ones that need attention first."); card.dataset.role = "health-table";
    const rows = [...d.companies].sort((a, b) => (a.status === "ok") - (b.status === "ok") || String(a.ticker).localeCompare(String(b.ticker)));
    table(card, [
      { label: "Company", get: (x) => `${x.ticker || x.security_id}${x.name ? " · " + x.name : ""}` },
      { label: "Latest annual", get: (x) => (x.latest_annual ? `${x.latest_annual.period_end}${x.latest_annual.filing_date ? " (filed " + x.latest_annual.filing_date + ")" : ""}` : "none") },
      { label: "Latest quarter", get: (x) => (x.latest_quarter ? `${x.latest_quarter.period} ${x.latest_quarter.period_end}` : "none") },
      { label: "Missing years", get: (x) => (x.missing_periods || []).join(", ") || "" },
      { label: "Conflicts", num: true, get: (x) => String(x.source_conflicts || 0) },
      { label: "Unconfirmed", num: true, get: (x) => String(x.unconfirmed_observations || 0) },
      { label: "Restated", num: true, get: (x) => String(x.restatements || 0) },
      { label: "Factor inputs", num: true, get: (x) => pct(x.factor_readiness, 0) },
      { label: "Last refresh", get: (x) => (x.last_provider_refresh || "").slice(0, 10) },
      { label: "Status", get: (x) => x.status },
      { label: "Kind of company", get: (x) => { const s = h("select", ""); s.setAttribute("aria-label", `kind of company ${x.ticker}`); for (const [v, l] of [["", "from sector text"], ["bank", "bank"], ["insurer", "insurer"], ["reit", "REIT"], ["general", "general"]]) { const o = h("option", "", s, l); o.value = v; } s.addEventListener("change", async () => { try { await FD.send("/api/fund/family", { security_id: x.security_id, family: s.value || null }); Q.toast("Saved: factors that do not apply to this kind of company are now missing for it"); } catch (e) { Q.toast(e.message); } }); return s; } },
    ], rows);
    const ev = HV.s.events;
    if (ev) {
      const c = W.card(root, "Corporate events", "Splits are found in the prices. A spin-off or merger cannot be told from a data error: record it, and market-based figures are left out until the next filing reflects it.");
      c.dataset.role = "health-events";
      h("p", "hint", c, `${(ev.splits || []).length} split(s) found · ${(ev.events || []).length} event(s) recorded · ${(ev.candidates || []).length} large price move(s) to look at`);
      for (const x of ev.candidates || []) {
        const row = h("div", "row tight", c); h("span", "", row, `Security ${x.security_id}, ${x.date}: price moved by a factor of ${num(x.ratio, 2)}`);
        const kind = h("select", "", row); for (const k2 of ["spin_off", "merger", "acquisition", "reorganisation", "other"]) { const o = h("option", "", kind, k2.replace("_", " ")); o.value = k2; }
        const frac = h("input", "", row); frac.placeholder = "share of value that left (0-1)"; frac.setAttribute("aria-label", "share of value that left");
        const b = h("button", "btn small", row, "Record"); b.type = "button";
        b.addEventListener("click", async () => { try { await FD.send("/api/fund/events/record", { security_id: x.security_id, event_date: x.date, kind: kind.value, value_fraction: frac.value ? Number(frac.value) : undefined }); Q.toast("Recorded"); HV.run(); } catch (e) { Q.toast(e.message); } });
      }
      for (const x of ev.events || []) h("p", "hint", c, `${x.event_date} · security ${x.security_id} · ${x.kind.replace("_", " ")}${x.counterparty ? " · " + x.counterparty : ""}${x.value_fraction ? " · " + Math.round(x.value_fraction * 100) + "% of value" : ""}`);
    }
  };

  /* ---------- screen ---------- */
  const factorList = async () => { if (!SV.s.factors) { try { SV.s.factors = (await FD.read("/api/fund/catalogue")).factors || []; } catch (e) { SV.s.factors = []; } } return SV.s.factors; };
  SV.sections = (panel) => {
    const b = FD.sec(panel, "Rules"); b.dataset.role = "screen-rules";
    factorList().then((fs) => { const dl = document.getElementById("screen-factors") || h("datalist", "", document.body); dl.id = "screen-factors"; dl.replaceChildren(); for (const f of fs) { const o = h("option", "", dl); o.value = f.name; o.textContent = f.label || f.name; } });
    SV.s.rules.forEach((r, i) => {
      const row = h("div", "row tight", b);
      const f = h("input", "", row); f.setAttribute("list", "screen-factors"); f.value = r.factor; f.dataset.key = `screen-factor-${i}`; f.setAttribute("aria-label", `factor of rule ${i + 1}`); f.addEventListener("input", () => { r.factor = f.value.trim(); });
      const op = h("select", "", row); op.setAttribute("aria-label", `comparison of rule ${i + 1}`); for (const [v, l] of OPS) { const o = h("option", "", op, l); o.value = v; if (v === r.op) o.selected = true; } op.addEventListener("change", () => { r.op = op.value; });
      const val = h("input", "", row); val.value = r.value; val.dataset.key = `screen-value-${i}`; val.placeholder = r.op === "between" ? "low, high" : "value"; val.setAttribute("aria-label", `value of rule ${i + 1}`); val.addEventListener("input", () => { r.value = val.value; });
      const del = h("button", "btn small", row, "Remove"); del.type = "button"; del.addEventListener("click", () => { SV.s.rules.splice(i, 1); Q.renderLab(FD); });
    });
    const add = h("button", "btn small", b, "Add a rule"); add.type = "button"; add.dataset.action = "screen-add"; add.addEventListener("click", () => { SV.s.rules.push({ factor: "", op: ">", value: "0" }); Q.renderLab(FD); });
    const o = FD.sec(panel, "Date and ranking");
    const d = h("input", "", h("label", "f", o, "As of (blank: the latest)")); d.type = "date"; d.value = SV.s.asOf; d.addEventListener("change", () => { SV.s.asOf = d.value; });
    const rb = h("input", "", h("label", "f", o, "Rank by (blank: the first rule's factor)")); rb.setAttribute("list", "screen-factors"); rb.value = SV.s.rankBy; rb.addEventListener("input", () => { SV.s.rankBy = rb.value.trim(); });
    const tp = h("input", "", h("label", "f", o, "Show the top")); tp.type = "number"; tp.value = SV.s.top; tp.min = 1; tp.addEventListener("change", () => { SV.s.top = Number(tp.value) || 20; });
    const wi = h("input", "", h("label", "f", o, "Percent rules within each sector")); wi.type = "checkbox"; wi.checked = SV.s.within; wi.addEventListener("change", () => { SV.s.within = wi.checked; });
    const hist = h("button", "btn small", o, "Run the same rules over past dates"); hist.type = "button"; hist.dataset.action = "screen-history";
    hist.addEventListener("click", () => FD.act("screen", "Screening past dates…", async () => { SV.s.hist = await FD.send("/api/fund/screen/history", { rules: rulesOut(), step: 6 }); return { kind: "screen" }; }));
  };
  const rulesOut = () => SV.s.rules.map((r) => ({ factor: r.factor, op: r.op, value: r.op === "between" ? String(r.value).split(",").map(Number) : Number(r.value) }));
  SV.run = () => FD.act("screen", "Screening…", async () => { SV.s.out = await FD.send("/api/fund/screen", { rules: rulesOut(), as_of: SV.s.asOf || undefined, rank_by: SV.s.rankBy || undefined, top: SV.s.top, within_sector: SV.s.within }); return { kind: "screen" }; });
  SV.empty = (root) => FD.emptyCard(root, "Screen the universe", ["Set the rules on the left and press Run the screen. A company is never passed on a missing figure and never failed for a line that does not apply to its industry."], { label: "Run the screen", fn: SV.run });
  SV.build = (root) => {
    const o = SV.s.out; if (!o) { SV.empty(root); return; }
    Q.labHead(root, "Screen", `as of ${o.as_of} · ${o.counts.pass} of ${o.universe} pass`);
    const k = h("div", "earnrow", root);
    for (const [key, label] of [["pass", "Pass"], ["failed", "Failed a rule"], ["missing", "Figure missing"], ["not applicable", "Not applicable"], ["not investable", "Not investable"]]) tile(k, label, String(o.counts[key] || 0), {}, key === "pass" ? "" : "muted");
    const card = W.card(root, `Ranked by ${o.ranked_by}`, o.point_in_time); card.dataset.role = "screen-results";
    const cols = [{ label: "#", num: true, get: (x) => String(x.rank) }, { label: "Company", get: (x) => `${x.ticker || x.security_id}${x.name ? " · " + x.name : ""}` }, { label: "Sector", get: (x) => x.sector || "" }];
    for (const key of Object.keys((o.rows[0] || {}).values || {})) cols.push({ label: FD.fmt.human(key), num: true, get: (x) => num(x.values[key], 4) });
    table(card, cols, o.rows);
    const ex = W.card(root, "Why the others are out", `${o.excluded.length} shown`); ex.dataset.role = "screen-excluded";
    table(ex, [{ label: "Security", get: (x) => String(x.security_id) }, { label: "Status", get: (x) => x.status }, { label: "Why", get: (x) => x.why }], o.excluded.slice(0, 200));
    if (SV.s.hist) {
      const hc = W.card(root, "The same rules on past dates", "Each date uses only what was public then."); hc.dataset.role = "screen-history";
      table(hc, [{ label: "Date", get: (x) => x.date }, { label: "Passing", num: true, get: (x) => String(x.count) }, { label: "Turnover vs previous", num: true, get: (x) => (x.turnover_vs_previous === null ? "" : pct(x.turnover_vs_previous, 0)) }], SV.s.hist.dates);
    }
  };

  /* ---------- scenarios ---------- */
  VV.sections = (panel) => {
    const b = FD.sec(panel, "Company and cases"); b.dataset.role = "scenario-inputs";
    const id = h("input", "", h("label", "f", b, "Security id")); id.type = "number"; id.value = VV.s.sid; id.dataset.key = "scenario-sid"; id.addEventListener("input", () => { VV.s.sid = id.value; });
    VV.s.cases.forEach((c, i) => {
      const row = h("div", "row tight", b);
      const n = h("input", "", row); n.value = c[0]; n.setAttribute("aria-label", `name of case ${i + 1}`); n.addEventListener("input", () => { c[0] = n.value; });
      const r = h("input", "", row); r.value = c[1]; r.setAttribute("aria-label", `discount rate of ${c[0]}`); r.addEventListener("input", () => { c[1] = r.value; });
      const g = h("input", "", row); g.value = c[2]; g.setAttribute("aria-label", `growth of ${c[0]}`); g.addEventListener("input", () => { c[2] = g.value; });
    });
    h("p", "hint", b, "Name, discount rate and perpetual growth (as decimals: 0.09 is 9%). The rate must be at least two points above the growth.");
  };
  VV.run = () => FD.act("scenarios", "Valuing…", async () => {
    if (!VV.s.sid) throw new Error("Give the security id of the company");
    const cases = {}; for (const [n, r, g] of VV.s.cases) if (n) cases[n] = [Number(r), Number(g)];
    VV.s.out = await FD.send("/api/fund/valuation/scenarios", { security_id: Number(VV.s.sid), cases }); return { kind: "scenarios" };
  });
  VV.empty = (root) => FD.emptyCard(root, "Compare assumptions", ["Enter a security id and press Run the scenarios."], { label: "Run the scenarios", fn: VV.run });
  VV.build = (root) => {
    const o = VV.s.out; if (!o) { VV.empty(root); return; }
    Q.labHead(root, "Scenarios", `${o.currency || ""} · last close ${num(o.price, 2)}`);
    const card = W.card(root, "Value per share under each case", "The low and high bars move the rate a point either way; “n/a” says why in the tooltip."); card.dataset.role = "scenario-table";
    const names = Object.keys(o.cases);
    table(card, [{ label: "Yardstick", get: (k) => FD.fmt.human(k) }, ...names.map((n) => ({ label: `${n} (${o.cases[n].inputs.rate * 100}% / ${o.cases[n].inputs.growth * 100}%)`, num: true, get: (k) => { const r = o.cases[n].rows[k]; return r.point === null ? "n/a" : num(r.point, 2); } }))], o.yardsticks);
    const g = W.card(root, "Free cash flow capitalised: discount rate against growth", o.sensitivity.flow); g.dataset.role = "scenario-grid";
    const t = h("table", "data", h("div", "scroll", g)), hd = h("tr", "", h("thead", "", t)); h("th", "", hd, "rate \\ growth");
    for (const x of o.sensitivity.growths) h("th", "num", hd, `${(x * 100).toFixed(1)}%`);
    const tb = h("tbody", "", t); o.sensitivity.rates.forEach((r, i) => { const tr = h("tr", "", tb); h("td", "", tr, `${(r * 100).toFixed(1)}%`); o.sensitivity.values[i].forEach((v) => h("td", "num", tr, v === null ? "n/a" : num(v, 2))); });
  };
})();
