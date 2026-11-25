/* The Options Strategy Studio: an opportunity scanner, ideas ranked from a stated view, a builder that prices real listed contracts from their bid and ask, a scenario laboratory (heatmaps and an attribution
   with an explicit residual) and a trade check. It talks to /api/options/scan and /api/options/studio/*. Server text goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS;
  if (!Q || !W) return;
  const C = Q.C, h = Q.h, num = Q.num, pct = Q.pct, money = Q.money, smoney = Q.smoney, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable;
  const MODES = new Set(["scan", "ideas", "builder", "scenarios", "tradecheck"]);
  const STUDIO = { modes: MODES };
  const KIND = { iv_history: "Implied vs its history", iv_vs_realised: "Implied vs realised", skew_dislocation: "Skew dislocation", negative_forward_variance: "Calendar arbitrage in the quotes", event_vol: "Possible event volatility", term_kink: "Term-structure kink", calendar: "Calendar candidate" };
  const qualityText = (q) => { if (!q) return "n/a"; if (q.front) return `${qualityText(q.front)} / ${qualityText(q.back)}`; return q.tradable ? `ok (spread ${q.median_spread_pct === null ? "?" : pct(q.median_spread_pct, 1)}, OI ${num(q.median_open_interest, 0)})` : "not tradable: " + q.reason; };

  /* ---------- the contracts ticket, shared by the builder, the scenarios and the trade check ---------- */
  function ticketEditor(O, parent, onChange) {
    const list = O.ticket = O.ticket || [], box = h("div", "legs", parent);
    const draw = () => {
      box.replaceChildren();
      list.forEach((x, i) => {
        const row = h("div", "row tight", box), field = (el, label) => { el.setAttribute("aria-label", `${label}, leg ${i + 1}`); return el; };
        const type = field(h("select", "", row), "type"); for (const [v, t] of [["option", "Option"], ["stock", "Shares"]]) { const o = h("option", "", type, t); o.value = v; if (x.type === v) o.selected = true; }
        let expiry;
        if (O.expiries && O.expiries.length) { expiry = field(h("select", "", row), "expiry"); for (const e of O.expiries) { const o = h("option", "", expiry, `${e.expiry} · ${e.dte}d`); o.value = e.expiry; if (e.expiry === x.expiry) o.selected = true; } if (!x.expiry) x.expiry = O.expiries[Math.min(4, O.expiries.length - 1)].expiry, expiry.value = x.expiry; }
        else { expiry = field(h("input", "", row), "expiry"); expiry.type = "text"; expiry.placeholder = "YYYY-MM-DD"; expiry.value = x.expiry || ""; }
        const right = field(h("select", "", row), "call or put"); for (const [v, t] of [["C", "Call"], ["P", "Put"]]) { const o = h("option", "", right, t); o.value = v; if (x.right === v) o.selected = true; }
        const strike = field(h("input", "", row), "strike"); strike.type = "number"; strike.step = "any"; strike.value = x.strike; strike.placeholder = "strike";
        const qty = field(h("input", "", row), "quantity"); qty.type = "number"; qty.step = "1"; qty.value = x.quantity; qty.placeholder = "+buy / -sell";
        const limit = field(h("input", "", row), "limit price"); limit.type = "number"; limit.step = "any"; limit.value = x.limit === undefined ? "" : x.limit; limit.placeholder = "limit (blank = mid)";
        const rm = field(h("button", "btn", row, "Remove"), "remove");
        const sync = () => { const stock = x.type === "stock"; expiry.disabled = right.disabled = strike.disabled = limit.disabled = stock; };
        type.addEventListener("change", () => { x.type = type.value; sync(); onChange(); });
        expiry.addEventListener("change", () => { x.expiry = expiry.value; onChange(); });
        right.addEventListener("change", () => { x.right = right.value; onChange(); });
        strike.addEventListener("change", () => { x.strike = strike.value; onChange(); });
        qty.addEventListener("change", () => { x.quantity = qty.value; onChange(); });
        limit.addEventListener("change", () => { x.limit = limit.value; onChange(); });
        rm.addEventListener("click", () => { list.splice(i, 1); draw(); onChange(); });
        sync();
      });
      if (!list.length) h("p", "hint", box, "No contracts yet: add one, or load legs from the scanner or the ideas.");
    };
    draw();
    const row = h("div", "row tight", parent);
    const add = h("button", "btn", row, "+ Option"), shares = h("button", "btn", row, "+ Shares"), clear = h("button", "btn", row, "Clear");
    add.addEventListener("click", () => { list.push({ type: "option", expiry: O.expiries && O.expiries.length ? O.expiries[Math.min(4, O.expiries.length - 1)].expiry : "", right: "C", strike: O.spot ? String(Math.round(O.spot)) : "", quantity: "1", limit: "" }); draw(); onChange(); });
    shares.addEventListener("click", () => { list.push({ type: "stock", quantity: "100" }); draw(); onChange(); });
    clear.addEventListener("click", () => { list.length = 0; draw(); onChange(); });
    return { draw };
  }
  STUDIO.load = (O, legs) => { O.ticket = legs.map((l) => (l.type === "stock" ? { type: "stock", quantity: String(l.quantity) } : { type: "option", expiry: l.expiry, right: l.right, strike: String(l.strike), quantity: String(l.quantity), limit: l.limit === undefined || l.limit === null ? "" : String(l.limit) })); O.mode = "builder"; Q.renderLab(O); };
  const legsBody = (O) => {
    const list = O.ticket || [];
    if (!list.length) throw new Error("Add at least one contract to the ticket");
    return list.map((x, i) => {
      const q = Number(x.quantity); if (!isFinite(q) || q === 0) throw new Error(`Leg ${i + 1} needs a quantity (positive to buy, negative to sell)`);
      if (x.type === "stock") return { type: "stock", quantity: q };
      const k = Number(x.strike); if (!(k > 0)) throw new Error(`Leg ${i + 1} needs a strike`); if (!/^\d{4}-\d{2}-\d{2}$/.test(String(x.expiry))) throw new Error(`Leg ${i + 1} needs an expiry`);
      return { type: "option", expiry: x.expiry, strike: k, right: x.right, quantity: q, ...(x.limit !== "" && x.limit !== undefined ? { limit: Number(x.limit) } : {}) };
    });
  };

  /* The book of the Greeks and risk view (hypothetical unless you typed your own), sent so that a trade is judged against what is already held. */
  const bookBody = (O) => { const list = (O.form.book || []).filter((x) => Number(x.quantity)); if (!list.length) return undefined;
    return list.map((x) => (x.type === "stock" ? { type: "stock", quantity: Number(x.quantity) } : { type: "option", right: x.right, strike: Number(x.strike), dte: Number(x.dte), quantity: Number(x.quantity), iv: Number(x.iv || O.atmIv || 0.25) })); };

  /* ---------- the form ---------- */
  const STAGES = [["scan", "1. Opportunity"], ["ideas", "2. Hypothesis"], ["builder", "3. Construction"], ["scenarios", "4. Scenarios"], ["tradecheck", "5. Validation"]];
  function stepper(O, box) {
    const i = STAGES.findIndex(([id]) => id === O.mode), bar = h("div", "row tight", box); bar.dataset.role = "studio-steps";
    for (const [id, label] of STAGES) { const b = h("button", "btn" + (id === O.mode ? " primary" : ""), bar, label); b.setAttribute("aria-current", String(id === O.mode)); b.addEventListener("click", () => { O.mode = id; Q.renderLab(O); }); }
    h("p", "hint", box, "One workflow: the underlying, the contracts of the ticket and your limits carry from stage to stage." + (i < STAGES.length - 1 ? ` Next: ${STAGES[i + 1][1].slice(3)}.` : " Last stage: nothing is sent anywhere."));
  }
  STUDIO.sections = (O, sec, after) => {
    const f = O.form, m = O.mode;
    if (MODES.has(m)) stepper(O, sec("Workflow"));
    if (m === "ideas") {
      const b = sec("2. Your view");
      Q.labField(b, { key: "direction", label: "Direction", kind: "choice", options: [["bullish", "Bullish"], ["bearish", "Bearish"], ["neutral", "Neutral (stays near here)"]] }, f, after);
      Q.labField(b, { key: "move", label: "Size of the move (%)", kind: "number", step: 0.5, min: 0.5, max: 50, help: "Up for bullish, down for bearish; for neutral, the range it stays within" }, f, after);
      Q.labField(b, { key: "volView", label: "Volatility", kind: "choice", options: [["neutral", "No view"], ["long", "Implied volatility rises"], ["short", "Implied volatility falls"]] }, f, after);
      Q.labField(b, { key: "volMove", label: "Volatility change (points)", kind: "number", step: 0.5, min: 0, max: 50 }, f, after);
      Q.labField(b, { key: "ideaHorizon", label: "Holding period (days)", kind: "number", step: 1, min: 3, max: 400 }, f, after);
      const c = sec("3. Limits");
      Q.labField(c, { key: "maxLoss", label: "Largest loss you accept ($)", kind: "number", min: 0, help: "Blank: no limit. A structure with an unlimited loss is excluded when a limit is set" }, f, after);
      Q.labField(c, { key: "marginBudget", label: "Margin you can tie up ($, indicative)", kind: "number", min: 0 }, f, after);
      h("p", "hint", c, "This ranks structures by what they pay if your view is right, and shows what they lose if it is wrong. It does not say which view is right.");
    } else if (m === "builder" || m === "scenarios" || m === "tradecheck") {
      const b = sec("2. Contracts");
      h("p", "hint", b, "Real listed contracts of the chain: positive quantity buys, negative sells. A limit price (blank: the mid) is judged against the bid and the ask.");
      ticketEditor(O, b, after);
      if (m === "tradecheck") {
        const c = sec("3. Your limits");
        Q.labField(c, { key: "maxLoss", label: "Largest loss you accept ($)", kind: "number", min: 0 }, f, after);
        Q.labField(c, { key: "marginBudget", label: "Margin you can tie up ($, indicative)", kind: "number", min: 0 }, f, after);
      }
      if (m === "builder") { const c = sec("3. Costs", false); Q.labField(c, { key: "commission", label: "Commission per contract ($)", kind: "number", step: 0.05, min: 0 }, f, after); }
    } else if (m === "scan") {
      h("p", "hint", sec("2. What it looks at"), "Every expiry of the chain: implied volatility against its own history and against realised volatility, skew across expiries, kinks and inversions in the term structure, possible event volatility and calendar candidates, each with the freshness and liquidity of the quotes behind it.");
    }
  };
  STUDIO.check = (O) => {
    const m = O.mode;
    if (m === "ideas" && !(Number(O.form.move) > 0)) return "Enter the size of the move.";
    if ((m === "builder" || m === "scenarios" || m === "tradecheck") && !(O.ticket || []).length) return "Add at least one contract to the ticket.";
    return "";
  };

  /* ---------- running ---------- */
  STUDIO.run = (O, settings) => {
    const m = O.mode, f = O.form;
    if (m === "scan") return W.request(O, () => [{ modes: ["scan"], path: "/api/options/scan", query: settings(), cacheKey: "scan:" + JSON.stringify(settings()) }], "Scanning the chain…");
    const base = settings();
    if (m === "ideas") return W.request(O, () => [{ modes: ["ideas"], path: "/api/options/studio/recommend", body: { ...base, direction: f.direction, move: Number(f.move) / 100, vol_view: f.volView, vol_move: Number(f.volMove) / 100, horizon_days: Number(f.ideaHorizon),
      max_loss: f.maxLoss === "" ? undefined : Number(f.maxLoss), margin_budget: f.marginBudget === "" ? undefined : Number(f.marginBudget) } }], "Ranking the structures…");
    const path = { builder: "/api/options/studio/build", scenarios: "/api/options/studio/scenarios", tradecheck: "/api/options/studio/validate" }[m];
    return W.request(O, () => [{ modes: [m], path, body: { ...base, legs: legsBody(O), commission: Number(f.commission), ...(m === "tradecheck" ? { max_loss: f.maxLoss === "" ? undefined : Number(f.maxLoss), margin_budget: f.marginBudget === "" ? undefined : Number(f.marginBudget), book: bookBody(O) } : {}) } }],
      m === "builder" ? "Pricing the contracts…" : m === "scenarios" ? "Revaluing under the scenarios…" : "Checking the trade…");
  };
  STUDIO.empty = (O, root) => {
    const t = { scan: ["Scan the chain for leads", "Press the button to list where implied volatility, skew and the term structure look unusual, with the quote quality behind each."], ideas: ["Rank structures from a view", "State a direction, a volatility view, a holding period and the loss you accept; the page ranks the feasible structures by what they pay if you are right."],
      builder: ["Price real contracts", "Add listed contracts with your limits; the page prices the package at the bid and ask, the mid and your limits and draws its payoff at that entry."], scenarios: ["Revalue under scenarios", "Four heatmaps and an attribution of each scenario to the underlying, volatility, skew, time and costs, with the residual shown."],
      tradecheck: ["Check a trade before it is placed", "Contracts listed, two-sided and fresh quotes, limits that can fill, a bounded loss within your maximum, margin."] }[O.mode];
    Q.labEmpty(root, O, t[0], [t[1]], O.runLabel[O.mode]);
  };

  /* ---------- results ---------- */
  const head = (O, root, title, sub, d) => { Q.labHead(root, title, sub, "technique-options-strategies"); W.provenance(root, d); };
  function unavailable(root, d, title) { const b = h("div", "unavail", root); h("strong", "", b, title + ": "); b.appendChild(document.createTextNode(d.reason || "not available")); if (d.needs && d.needs.length) h("div", "hint", b, "Needed: " + d.needs.join("; ")); }

  function buildScan(O, root, d) {
    head(O, root, `${d.underlying} opportunity scanner`, `chain of ${d.chain_date || "n/a"} · ${d.items.length} lead${d.items.length === 1 ? "" : "s"}`, d);
    const k = h("div", "earnrow", root);
    if (d.iv_history.state === "ok") tile(k, "30-day implied vs its history", pct(d.iv_history.percentile, 0), { text: `${pct(d.iv_history.iv_30d, 1)}, z ${num(d.iv_history.zscore, 2)} over ${d.iv_history.n} chains` }); else unavailable(root, d.iv_history, "Implied volatility against its own history");
    if (d.realised.state === "ok") { tile(k, "Implied / realised (30d / 21d)", num(d.realised.ratio, 2), { text: `implied ${pct(d.realised.iv_30d, 1)} · realised ${pct(d.realised.rv_21d, 1)}` }); tile(k, "Realised vol percentile", pct(d.realised.rv_percentile, 0), { text: "of the last two years" }); } else unavailable(root, d.realised, "Implied against realised");
    const card = h("div", "card", root); h("h3", "", card, "Leads");
    if (!d.items.length) h("p", "hint", card, "Nothing in this chain stands out against its neighbours or its history.");
    else table(card, [{ label: "Kind", get: (r) => KIND[r.kind] || r.kind }, { label: "What", get: (r) => r.title }, { label: "Quotes", get: (r) => (r.tradable === null ? "n/a" : r.tradable ? "tradable" : "not tradable") }, { label: "Quality", get: (r) => qualityText(r.quality) },
      { label: "Detail", get: (r) => r.detail }, { label: "", get: (r) => r.legs || "" }], d.items);
    d.items.forEach((r, i) => { if (!r.legs) return; const row = card.querySelectorAll("tbody tr")[i]; if (!row) return; const cell = row.lastElementChild, b = h("button", "btn", cell, "Load into the builder"); b.addEventListener("click", () => STUDIO.load(O, r.legs)); });
    const ch = h("div", "charts", root);
    chartCard(ch, "Term structure and forward volatility", "At-the-money volatility by expiry, and the forward volatility between neighbouring expiries. A bump in the forward line can be an event inside that window.", { wide: true,
      draw: (b) => C.line(b, { xs: d.term.map((t) => t.dte), xWhole: true, xLabel: "Days to expiry", series: [{ name: "ATM implied", color: Q.css("--s1"), values: d.term.map((t) => t.atm_iv) }, { name: "Forward between expiries", color: Q.css("--s2"), values: d.term.map((_, i) => (i === 0 || !d.forward_vols[i - 1] ? null : d.forward_vols[i - 1].forward_vol)), width: 1.5 }], yFormat: (v) => (v * 100).toFixed(0) + "%", label: "Term structure", endLabels: false }),
      table: (b) => table(b, [{ label: "Expiry", get: (t) => t.expiry }, { label: "Days", num: true, get: (t) => t.dte }, { label: "ATM implied", num: true, get: (t) => pct(t.atm_iv, 1) }, { label: "Quotes", get: (t) => qualityText(d.quality[t.expiry]) }], d.term) });
    if (d.skew.length) chartCard(ch, "25-delta risk reversal by expiry", "Call volatility minus put volatility at 25 delta. An expiry far from its neighbours is a dislocation candidate.", {
      draw: (b) => C.line(b, { xs: d.skew.map((s) => s.dte), xWhole: true, xLabel: "Days to expiry", series: [{ name: "Risk reversal", color: Q.css("--s1"), values: d.skew.map((s) => s.rr_25d) }], baseline: 0, yFormat: (v) => (v * 100).toFixed(1) + "%", label: "Risk reversal", endLabels: false }),
      table: (b) => table(b, [{ label: "Expiry", get: (s) => s.expiry }, { label: "Risk reversal 25d", num: true, get: (s) => pct(s.rr_25d, 2) }, { label: "Butterfly 25d", num: true, get: (s) => pct(s.bf_25d, 2) }], d.skew) });
    W.hint(root, d.reading);
  }

  function buildIdeas(O, root, d) {
    const hy = d.hypothesis;
    head(O, root, `${d.underlying}: structures for your view`, `${hy.direction}, ${pct(hy.move, 1)} · volatility ${hy.vol_view} · ${hy.horizon_days} days · ${d.n_feasible} feasible`, d);
    const card = h("div", "card", root); h("h3", "", card, "Ranked by profit if you are right");
    if (!d.ranked.length) h("p", "hint", card, "No structure fits these limits and this view.");
    else {
      table(card, [{ label: "Structure", get: (r) => r.strategy.replace(/_/g, " ") }, { label: "Family", get: (r) => r.group }, { label: "If right", num: true, get: (r) => smoney(r.pnl_if_right) }, { label: "If wrong", num: true, get: (r) => smoney(r.pnl_if_wrong) },
        { label: "Max loss", num: true, get: (r) => (r.unlimited_loss ? "unlimited" : money(-r.max_loss, 0)) }, { label: "Margin", num: true, get: (r) => (r.margin === null ? "unbounded" : money(r.margin, 0)) }, { label: "Delta $", num: true, get: (r) => smoney(r.delta_dollars) },
        { label: "Vega / pt", num: true, get: (r) => smoney(r.vega_per_point) }, { label: "Spread cost", num: true, get: (r) => money(r.friction, 0) }, { label: "", get: () => "" }], d.ranked);
      d.ranked.forEach((r, i) => { const row = card.querySelectorAll("tbody tr")[i]; if (!row) return; const b = h("button", "btn", row.lastElementChild, "Use these legs"); b.addEventListener("click", () => STUDIO.load(O, r.legs)); });
    }
    if (d.stability) { h("p", d.stability.stable ? "hint" : "verdict-bad", root, `Stability: ${d.stability.reading} (top structure unchanged in ${d.stability.top_unchanged_in} of ${d.stability.of} perturbed runs).`); }
    if (d.baselines) h("p", "hint", root, `For comparison: ${smoney(d.baselines.hold_100_shares)} on 100 shares if right, and ${smoney(d.baselines.cash)} in cash. ${d.baselines.note}.`);
    if (d.excluded.length) { const dd = h("details", "how", root); h("summary", "", dd, `Excluded (${d.excluded.length})`); table(dd, [{ label: "Structure", get: (r) => r.strategy.replace(/_/g, " ") }, { label: "Why", get: (r) => r.reason }], d.excluded); }
    const dd = h("details", "how", root); h("summary", "", dd, "Assumptions"); for (const a of d.assumptions) h("p", "hint", dd, a);
    W.hint(root, d.reading);
  }

  function buildBuilder(O, root, d, body) {
    head(O, root, `${d.underlying}: the position`, `spot ${money(d.spot, 2)} · ${d.legs.length} leg${d.legs.length === 1 ? "" : "s"}`, d);
    const t = d.totals, a = d.analysis, k = h("div", "earnrow", root);
    tile(k, "At your limits", smoney(t.at_limits), { text: "debit (+) or credit (-), with commission" }); tile(k, "At the mid", smoney(t.at_mid)); tile(k, "Crossing every spread", smoney(t.at_touch), { text: `spread cost ${money(t.spread_cost, 0)}` });
    tile(k, "Maximum profit", a.unlimited_profit ? "unlimited" : smoney(a.max_profit)); tile(k, "Maximum loss", a.unlimited_loss ? "unlimited" : smoney(a.max_loss)); tile(k, "Chance of profit", pct(a.prob_profit, 0), { text: "lognormal, market volatility" });
    const card = h("div", "card", root); h("h3", "", card, "Legs");
    table(card, [{ label: "Contract", get: (r) => r.label }, { label: "Qty", num: true, get: (r) => r.quantity }, { label: "Bid", num: true, get: (r) => (r.bid === null ? "" : num(r.bid, 2)) }, { label: "Ask", num: true, get: (r) => (r.ask === null ? "" : num(r.ask, 2)) },
      { label: "Spread", num: true, get: (r) => (r.spread_pct === undefined || r.spread_pct === null ? "" : pct(r.spread_pct, 1)) }, { label: "Volume", num: true, get: (r) => (r.volume === undefined ? "" : num(r.volume, 0)) }, { label: "Open int.", num: true, get: (r) => (r.open_interest === undefined ? "" : num(r.open_interest, 0)) },
      { label: "Limit", num: true, get: (r) => (r.limit === null || r.limit === undefined ? "mid" : num(r.limit, 2)) }, { label: "Fill", get: (r) => r.status + (r.stale ? " · stale quote" : "") }], d.legs);
    W.hint(root, d.fill.reading);
    const ch = h("div", "charts", root);
    chartCard(ch, "Profit and loss at the horizon", `Expiry, today and half way, from the actual entry cost. Break-evens: ${a.breakevens.map((b) => num(b, 2)).join(", ") || "none"}.`, { wide: true,
      draw: (b) => C.line(b, { xs: a.curve.spot, series: [{ name: "At expiry", color: Q.css("--s1"), values: a.curve.expiry }, { name: "Half way", color: Q.css("--s2"), values: a.curve.half, width: 1.5 }, { name: "Today", color: Q.css("--s3"), values: a.curve.now, width: 1.5 }], baseline: 0, yFormat: (v) => smoney(v), xFormat: (v) => num(v, 0), label: "Payoff", endLabels: false }),
      table: (b) => table(b, [{ label: "Price", num: true, get: (n) => num(a.curve.spot[n], 2) }, { label: "At expiry", num: true, get: (n) => smoney(a.curve.expiry[n]) }, { label: "Today", num: true, get: (n) => smoney(a.curve.now[n]) }], a.curve.spot.map((_, n) => n).filter((n) => n % 12 === 0)) });
    const g = a.greeks; W.kv(h("div", "card", root), "Risk today", [["Delta ($)", smoney(g.delta_dollars)], ["Gamma P&L of a 1% move", smoney(g.gamma_pnl_1pct)], ["Vega per point", smoney(g.vega_per_point)], ["Theta per day", smoney(g.theta_per_day)], ["Indicative margin", money(a.margin.estimate, 0)], ["Expected value (market view)", smoney(a.expected_value)]]);
    if (window.Experiments) window.Experiments.saveButton(root, { kind: "options_studio", name: `${d.underlying} ${d.legs.length}-leg position`, request: { path: "/api/options/studio/build", body }, metrics: window.Experiments.metricsOf.options_studio(d),
      provenance: d.provenance, universe: [d.underlying], assumptions: d.provenance && d.provenance.assumptions });
  }

  function buildScenarios(O, root, d, body) {
    head(O, root, `${d.underlying}: scenario laboratory`, `value today ${smoney(d.value_today)} · entry ${smoney(d.entry.at_limits)}`, d);
    const hm = d.heatmaps, ch = h("div", "charts", root), pctText = (x) => W.signed(x * 100, 0) + "%";
    const heat = (title, text, rows, cols, values, rowTitle, colTitle, fmt) => chartCard(ch, title, text, { draw: (b) => C.matrix(b, { rows: rows.map(fmt), cols: cols.map(pctText), values, diverging: true, mid: 0, format: (v) => smoney(v), name: "Profit and loss", rowTitle, colTitle, label: title }),
      table: (b) => table(b, [{ label: rowTitle, get: (r) => r.r }, ...cols.map((c, j) => ({ label: pctText(c), num: true, get: (r) => smoney(r.v[j]) }))], rows.map((x, i) => ({ r: fmt(x), v: values[i] }))) });
    heat("Spot and volatility", "P&L against today's value. Rows: parallel change of every implied volatility in points; columns: the underlying's move.", hm.spot_vol.vol, hm.spot_vol.spot, hm.spot_vol.pnl, "Volatility (pts)", "Underlying", (v) => W.signed(v * 100, 0));
    heat("Spot and skew", "Rows: a shift of the skew slope (each strike's volatility moves by shift × ln K/F). Negative steepens the equity skew.", hm.spot_skew.skew, hm.spot_skew.spot, hm.spot_skew.pnl, "Skew shift", "Underlying", (v) => W.signed(v, 2));
    heat("Spot and time", "Rows: days passing with volatility unchanged; the time decay (or gain) of the position against price.", hm.spot_time.days, hm.spot_time.spot, hm.spot_time.pnl, "Days passed", "Underlying", (v) => v + "d");
    const card = h("div", "card", root); h("h3", "", card, "Scenarios and where their P&L comes from");
    table(card, [{ label: "Scenario", get: (r) => r.name }, { label: "P&L after costs", num: true, get: (r) => smoney(r.pnl) }, { label: "Delta", num: true, get: (r) => smoney(r.attribution.delta) }, { label: "Gamma", num: true, get: (r) => smoney(r.attribution.gamma) }, { label: "Vega", num: true, get: (r) => smoney(r.attribution.vega) }, { label: "Vanna+volga", num: true, get: (r) => smoney(r.attribution.cross_greeks) },
      { label: "Skew", num: true, get: (r) => smoney(r.attribution.skew) }, { label: "Theta", num: true, get: (r) => smoney(r.attribution.theta) }, { label: "Costs", num: true, get: (r) => smoney(r.attribution.costs) }, { label: "Residual", num: true, get: (r) => smoney(r.attribution.residual) }], d.scenarios);
    W.hint(root, d.attribution_method);
    const g = d.greeks_today; W.kv(h("div", "card", root), "Greeks today", [["Delta ($)", smoney(g.delta_dollars)], ["Gamma P&L of a 1% move", smoney(g.gamma_pnl_1pct)], ["Vega per point", smoney(g.vega_per_point)], ["Theta per day", smoney(g.theta_per_day)]]);
    const dd = h("details", "how", root); h("summary", "", dd, "Assumptions"); for (const a of d.assumptions) h("p", "hint", dd, a);
    if (window.Experiments) window.Experiments.saveButton(root, { kind: "options_scenarios", name: `${d.underlying} scenarios`, request: { path: "/api/options/studio/scenarios", body }, metrics: window.Experiments.metricsOf.options_scenarios(d), provenance: d.provenance, universe: [d.underlying], assumptions: d.provenance && d.provenance.assumptions });
  }

  function buildTradeCheck(O, root, d) {
    head(O, root, `${d.underlying}: trade check`, d.passed ? "every check passes" : "at least one check fails", d);
    const card = h("div", "card", root); h("h3", "", card, "Checks");
    table(card, [{ label: "Check", get: (r) => r.label }, { label: "Result", get: (r) => (r.ok ? "pass" : "FAIL") }, { label: "Detail", get: (r) => r.detail }], d.checks);
    h("p", d.passed ? "verdict-ok" : "verdict-bad", root, d.reading);
    if (d.portfolio_impact) { const pi = d.portfolio_impact, c = h("div", "card", root); h("h3", "", c, "What it does to the book" + (pi.hypothetical_book ? " (hypothetical)" : ""));
      table(c, [{ label: "", get: (r) => r[0] }, { label: "Before", num: true, get: (r) => smoney(pi.before[r[1]]) }, { label: "After", num: true, get: (r) => smoney(pi.after[r[1]]) }, { label: "Change", num: true, get: (r) => smoney(pi.change[r[1]]) }],
        [["Delta ($)", "delta_dollars"], ["Gamma P&L, 1% move", "gamma_pnl_1pct"], ["Vega per point", "vega_per_point"], ["Theta per day", "theta_per_day"], ["Vanna", "vanna_delta_per_vol_point"], ["Volga", "volga_vega_per_vol_point"]]);
      h("p", "hint", c, `Worst stress loss: ${smoney(pi.worst_stress.before)} before, ${smoney(pi.worst_stress.after)} after.`); }
    const k = h("div", "earnrow", root); tile(k, "Cost at the limits", smoney(d.totals.at_limits)); tile(k, "Crossing every spread", smoney(d.totals.at_touch)); tile(k, "Indicative margin", money(d.margin.estimate, 0), { text: d.margin.basis });
  }
  STUDIO.build = (O, root, d, body) => ({ scan: buildScan, ideas: buildIdeas, builder: buildBuilder, scenarios: buildScenarios, tradecheck: buildTradeCheck })[O.mode](O, root, d, body);
  if (window.Experiments) {
    window.Experiments.metricsOf.options_scenarios = (d) => Object.fromEntries(d.scenarios.map((s, i) => [`scenario_${i + 1}_pnl`, s.pnl]));
    window.Experiments.metricsOf.options_studio = (d) => ({ debit_at_limits: d.totals.at_limits, debit_at_mid: d.totals.at_mid, debit_at_touch: d.totals.at_touch, max_profit: d.analysis.max_profit, max_loss: d.analysis.max_loss, prob_profit: d.analysis.prob_profit,
      delta_dollars: d.analysis.greeks.delta_dollars, vega_per_point: d.analysis.greeks.vega_per_point, theta_per_day: d.analysis.greeks.theta_per_day });
  }
  window.OptionsStudio = STUDIO;
})();
