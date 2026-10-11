/* The Futures workspace: the forward curve and how it moved, the roll (which contract to hold and when to leave it), carry and basis, a margin dashboard for a book of futures, and strategies run on the
   simulated roots through the platform's pipeline. It talks to /api/futures/*. No futures prices are held in this repository: the curves are a model (Schwartz-Smith) unless a file of yours replaces them, and
   every payload says which, and this page prints it. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, money = Q.money, smoney = Q.smoney, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable;
  const GUIDE = "technique-futures-curves-carry-and-rolls";
  const CURVE_MODES = new Set(["curve", "roll", "carry"]);
  const STATE_TEXT = { backwardation: "Backwardation", contango: "Contango", mixed: "Mixed (humped or kinked)" };
  const STATE_HELP = { backwardation: "Later contracts are cheaper: a long position earns the roll yield as the contract rolls up the curve.", contango: "Later contracts are dearer: a long position pays the roll yield as the contract rolls down the curve.",
    mixed: "The curve is neither clearly rising nor clearly falling over its front contracts." };

  const F = {
    id: "futures", root: "#futures", view: "#view-futures", title: "Futures", api: 4, mode: "curve", busy: "", results: {}, errors: {}, cache: new Map(), guard: null, go: null, progress: null,
    about: "Read a futures curve, decide when to roll, see carry and basis, check the margin of a book and test a strategy on several roots. The prices are a model unless you bring a file.",
    modes: [["curve", "Curve"], ["roll", "Roll"], ["carry", "Carry and basis"], ["margin", "Margin"], ["backtest", "Backtest"]],
    runLabel: { curve: "Refresh", roll: "Refresh", carry: "Refresh", margin: "Check the margin", backtest: "Run the backtest" },
    blurb: { curve: "Price by contract month, how the curve has moved over the last two months, the roll yield of each contract and the contract specification.",
      roll: "Which contract you hold, the next one, the suggested roll date (by calendar and by open interest), what the roll costs, and the rolls of the past.",
      carry: "Carry (the annualised roll yield) through time against its own history, and the basis: how far a futures price sits from the spot price.",
      margin: "Notional, initial and maintenance margin, leverage and the distance to a margin call for a book of futures, with warnings for heavy use, concentration and contracts close to expiry.",
      backtest: "Trade carry, trend, breakout or momentum across the roots through the platform's pipeline: costed, lagged and volatility-targeted like any other backtest." },
    form: { root: "CL", rollDays: "5", book: [], equity: "100000", scale: "1", near: "7", strategy: "carry_ts", roots: new Set(["CL", "NG", "GC", "HG", "ES"]), lookback: "", frequency: "weekly", targetVol: "", cost: "2" },
    catalog: null,
  };
  const contractColor = W.palette();
  W.catalog().then((c) => { F.catalog = c; if (!Q.$(F.view).hidden) Q.renderLab(F); }, () => {});
  const rootsList = () => (F.catalog ? F.catalog.futures_roots : []);
  const rootInfo = (id) => rootsList().find((r) => r.id === id) || {};
  const key = (name, q) => name + ":" + JSON.stringify(q);

  /* ---------- the form ---------- */
  F.check = () => {
    if (!F.catalog) return "Loading the list of roots…";
    if (F.mode === "backtest") { if (!F.form.roots.size) return "Tick at least one root."; if (["carry_xs", "xs_momentum"].includes(F.form.strategy) && F.form.roots.size < 2) return "A cross-sectional strategy needs at least two roots."; }
    return "";
  };
  F.sections = (panel) => {
    const f = F.form, after = () => Q.syncLab(F), sec = (title, open = true) => { const d = h("details", "", panel); d.open = open; h("summary", "", d, title); return h("div", "body", d); };
    const m = F.mode, cat = F.catalog;
    if (!cat) { h("p", "hint", sec("1. Root"), "Loading the list of roots…"); return; }
    if (CURVE_MODES.has(m)) {
      const b = sec("1. Root");
      Q.labField(b, { key: "root", label: "Futures root", kind: "choice", options: cat.futures_roots.map((r) => [r.id, `${r.id}: ${r.name}${r.source === "user_file" ? " (your file)" : ""}`]) }, f, () => F.run(true));
      const info = rootInfo(f.root);
      h("p", "hint", b, info.source === "user_file" ? "Prices are read from your file in data/user/futures/." : "No futures prices are held here: this curve is a model (Schwartz-Smith: a short-term factor that decays and a long-term level), with simulated open interest. Put " + f.root + ".csv in data/user/futures/ to use your own.");
      if (m === "roll") Q.labField(b, { key: "rollDays", label: "Leave the contract this many days before expiry", kind: "number", step: 1, min: 0, max: 60 }, f, () => F.run(true));
    }
    if (m === "margin") {
      const b = sec("1. The book");
      h("p", "hint", b, "Quantities: positive long, negative short. Leave the contract blank to use the front one. Margins are illustrative figures per contract, not your broker's.");
      marginRows(b, f.book, after);
      const ex = h("button", "btn", b, "Load an example"); ex.addEventListener("click", () => { f.book = [{ root: "CL", contract: "", quantity: "2" }, { root: "GC", contract: "", quantity: "1" }, { root: "ES", contract: "", quantity: "-1" }]; Q.renderLab(F); });
      const c = sec("2. Account");
      for (const s of [{ key: "equity", label: "Account equity ($)", kind: "number", min: 1000 }, { key: "scale", label: "Margin multiplier (1 = the table's figures)", kind: "number", step: 0.1, min: 0.1, max: 10, help: "Brokers raise margins in stress; try 1.5 or 2" },
        { key: "near", label: "Warn when a contract expires within (days)", kind: "number", step: 1, min: 0, max: 60 }]) Q.labField(c, s, f, after);
    }
    if (m === "backtest") {
      const b = sec("1. The strategy");
      Q.labField(b, { key: "strategy", label: "Strategy", kind: "choice", options: cat.futures_strategies.map((s) => [s.id, s.label]) }, f, () => Q.renderLab(F));
      const info = cat.futures_strategies.find((s) => s.id === f.strategy); if (info) h("p", "hint", b, info.text);
      const roots = sec("2. Roots");
      const box = h("div", "checks", roots); h("span", "hint", box, "Tick the roots to trade:");
      for (const r of cat.futures_roots.filter((x) => x.source !== "user_file" || x.id in { CL: 1, NG: 1, GC: 1, HG: 1, ES: 1 })) {
        const label = h("label", "f inline", box), c = h("input", "", label); c.type = "checkbox"; c.checked = f.roots.has(r.id); label.appendChild(document.createTextNode(`${r.id}: ${r.name} (${r.asset_class})`));
        c.addEventListener("change", () => { if (c.checked) f.roots.add(r.id); else f.roots.delete(r.id); after(); });
      }
      h("p", "hint", roots, "Only the simulated roots can be backtested: they share one calendar and one model.");
      const c = sec("3. Trading");
      Q.labField(c, { key: "frequency", label: "Rebalance", kind: "choice", options: [["daily", "Daily"], ["weekly", "Weekly"], ["monthly", "Monthly"]] }, f, after);
      for (const s of [{ key: "lookback", label: "Lookback in days (blank = the strategy's own)", kind: "number", step: 1, min: 5, max: 756 }, { key: "targetVol", label: "Volatility target (% a year, blank = the platform's)", kind: "number", step: 1, min: 1, max: 100 },
        { key: "cost", label: "Cost of trading, one way (bps)", kind: "number", step: 0.5, min: 0, max: 200 }]) Q.labField(c, s, f, after);
      h("p", "hint", c, "The first run builds the simulated futures and takes up to half a minute; later runs reuse them.");
    }
  };
  function marginRows(parent, list, onChange) {
    const box = h("div", "legs", parent), draw = () => {
      box.replaceChildren();
      list.forEach((x, i) => {
        const row = h("div", "leg", box), root = h("select", "", row), contract = h("input", "", row), rm = h("button", "iconbtn", row, "×"), nums = h("div", "l2", row), qty = h("input", "", nums);
        for (const r of rootsList()) { const o = h("option", "", root, r.id); o.value = r.id; if (r.id === x.root) o.selected = true; }
        contract.type = "text"; contract.placeholder = "contract"; contract.value = x.contract; contract.spellcheck = false; contract.setAttribute("aria-label", "contract of row " + (i + 1));
        qty.type = "number"; qty.step = "1"; qty.value = x.quantity; qty.placeholder = "contracts (− short)"; qty.setAttribute("aria-label", "quantity of row " + (i + 1)); root.setAttribute("aria-label", "root of row " + (i + 1)); rm.setAttribute("aria-label", "remove row " + (i + 1));
        nums.style.gridTemplateColumns = "1fr";
        root.addEventListener("change", () => { x.root = root.value; onChange(); }); contract.addEventListener("input", () => { x.contract = contract.value; onChange(); }); qty.addEventListener("input", () => { x.quantity = qty.value; onChange(); });
        rm.addEventListener("click", () => { list.splice(i, 1); draw(); onChange(); });
      });
      if (!list.length) h("p", "hint", box, "No rows yet.");
    };
    draw();
    const row = h("div", "row tight", parent), add = h("button", "btn", row, "+ Position"), clear = h("button", "btn", row, "Clear");
    add.addEventListener("click", () => { list.push({ root: rootsList()[0] ? rootsList()[0].id : "CL", contract: "", quantity: "1" }); draw(); onChange(); });
    clear.addEventListener("click", () => { list.length = 0; draw(); onChange(); });
  }

  /* ---------- running ---------- */
  F.run = (fromForm) => {
    const m = F.mode, f = F.form;
    if (CURVE_MODES.has(m)) {
      if (!fromForm) F.cache.clear();
      return W.request(F, () => {
        const q = { root: f.root }, steps = [{ modes: ["curve", "carry"], path: "/api/futures/curve", query: q, cacheKey: key("curve", q) }];
        if (m === "roll") { const r = { root: f.root, roll_days: Number(f.rollDays) }; steps.push({ modes: ["roll"], path: "/api/futures/roll", query: r, cacheKey: key("roll", r) }); }
        return steps;
      }, "Reading the curve…");
    }
    if (m === "margin") return W.request(F, () => {
      if (!f.book.length) throw new Error("Add at least one position");
      return [{ modes: ["margin"], path: "/api/futures/margin", body: { positions: f.book.map((x, i) => { const q = Number(x.quantity); if (!isFinite(q) || q === 0) throw new Error(`Row ${i + 1} needs a quantity`); return { root: x.root, contract: x.contract || undefined, quantity: q }; }),
        equity: Number(f.equity), margin_scale: Number(f.scale), near_expiry_days: Number(f.near) } }];
    }, "Computing the margin…");
    return W.request(F, () => [{ modes: ["backtest"], job: true, path: "/api/futures/strategy/backtest", body: { strategy: f.strategy, roots: [...f.roots], frequency: f.frequency, lookback: f.lookback === "" ? undefined : Number(f.lookback),
      target_vol: f.targetVol === "" ? undefined : Number(f.targetVol) / 100, cost_bps: Number(f.cost) } }], "Running the backtest…");
  };
  F.auto = () => { if (CURVE_MODES.has(F.mode) && F.catalog) F.run(true); };
  F.empty = (root) => {
    const text = { margin: ["Check the margin of a book", "Add the contracts you hold (or load the example) to see the notional, the margin they need, the leverage and how far you are from a margin call."],
      backtest: ["Backtest a futures strategy", "Carry, trend, breakout or momentum across the roots, costed and lagged like every other backtest here. The prices are a model, so this checks the machinery, not the strategy."] }[F.mode] || ["Read the curve", "Loading."];
    Q.labEmpty(root, F, text[0], [text[1]], F.runLabel[F.mode]);
  };
  F.draw = () => Q.drawLab(F);
  F.build = (root, d, body) => ({ curve: buildCurve, roll: buildRoll, carry: buildCarry, margin: buildMargin, backtest: buildBacktest })[F.mode](root, d, body);
  const header = (root, title, sub) => Q.labHead(root, title, sub, GUIDE);

  /* ---------- curve ---------- */
  function buildCurve(root, d) {
    const c = d.curve, sp = d.spec, rows = c.rows;
    header(root, `${c.name} futures (${d.root})`, `${sp.exchange} · front ${c.front} at ${num(c.front_price, 2)} · one contract is ${money(c.notional)} of exposure`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Shape of the curve", STATE_TEXT[c.state] || c.state, { text: `slope ${W.points(c.log_slope)} a year` }, "", STATE_HELP[c.state] || "");
    tile(k, "Roll yield, annualised", pct(c.annualised_carry, 1), { text: c.annualised_carry >= 0 ? "a long position earns it" : "a long position pays it", dir: c.annualised_carry >= 0 ? "up" : "down" }, "", "The front contract against the next, annualised: positive in backwardation");
    tile(k, "Front contract", num(c.front_price, 2), { text: `${c.front} · ${rows[0].dte} days to expiry` });
    tile(k, "Contract value", money(c.notional), { text: `tick ${c.tick_size} = ${money(c.tick_value, 2)}` }, "", "Price times the contract multiplier of " + sp.multiplier);
    tile(k, "Margin per contract", money(sp.initial_margin), { text: `maintenance ${money(sp.maintenance_margin)}` }, "", "Illustrative figures per contract, not an exchange's current margin");
    tile(k, "Curvature", num(c.curvature * 100, 2) + "%", { text: "second difference of the first three" }, "", "(Front + third − 2 × second) over the second: positive when the curve is humped upward at the front");
    const ch = h("div", "charts", root), hist = d.history, last = (hist[0] || { points: [] }).points;
    chartCard(ch, "Forward curve", "Price of each contract month today and as it was one and two months ago (matching contract months by their place in the curve: M1 is the front). The shape shows what the market charges to hold the commodity.", { wide: true,
      draw: (b) => C.line(b, { xs: last.map((_, i) => i + 1), xWhole: true, xLabel: "Contract month (1 = front)", series: hist.map((x, n) => ({ name: n === 0 ? `Today (${x.date})` : x.date, color: contractColor(x.date), values: x.points.map((p) => p.price), dots: n === 0, dotSize: 3, width: n === 0 ? 2 : 1.5 })),
        yFormat: (v) => num(v, 2), tipFormat: (v) => num(v, 2), xFormat: (v) => "M" + Math.round(v), tipTitle: (i) => `Month ${i + 1}: ${last[i].contract} (${last[i].dte} days)`, height: 300, label: "Forward curve", endLabels: false }),
      table: (b) => table(b, [{ label: "Month", get: (r) => "M" + r.rank }, { label: "Contract", get: (r) => r.contract }, { label: "Expiry", get: (r) => r.expiry }, { label: "Days", num: true, get: (r) => r.dte }, { label: "Price", num: true, get: (r) => num(r.price, 3) },
        { label: "vs front", num: true, get: (r) => Q.fmt.spct(r.change_vs_front) }, { label: "Roll yield (a year)", num: true, get: (r) => (r.roll_yield === null ? "" : pct(r.roll_yield, 1)) }, { label: "Open interest", num: true, get: (r) => (r.open_interest === null ? "" : Math.round(r.open_interest).toLocaleString("en-US")) },
        { label: "Volume", num: true, get: (r) => (r.volume === null ? "" : Math.round(r.volume).toLocaleString("en-US")) }], rows) });
    const ry = rows.filter((r) => r.roll_yield !== null);
    chartCard(ch, "Roll yield by contract", "Each contract against the next, annualised: above zero (backwardation) a long position gains as the contract ages, below zero (contango) it pays.", {
      draw: (b) => C.columns(b, { labels: ry.map((r) => r.contract), values: ry.map((r) => r.roll_yield), yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 230, name: "Roll yield", label: "Roll yield by contract", tipLabel: (i) => `${ry[i].contract} against the next (${ry[i].dte} days)` }),
      table: (b) => table(b, [{ label: "Contract", get: (r) => r.contract }, { label: "Roll yield (a year)", num: true, get: (r) => pct(r.roll_yield, 2) }], ry) });
    const oi = rows.filter((r) => r.open_interest !== null);
    chartCard(ch, "Open interest by contract", `Contracts outstanding. The front month is where most positions sit until the roll${d.sources.open_interest === "simulated" ? " (simulated here)" : ""}.`, {
      draw: (b) => C.columns(b, { labels: oi.map((r) => r.contract), values: oi.map((r) => r.open_interest), yFormat: (v) => (v >= 1000 ? (v / 1000).toFixed(0) + "k" : String(v)), tipFormat: (v) => Math.round(v).toLocaleString("en-US"), height: 230, name: "Open interest", label: "Open interest" }),
      table: (b) => table(b, [{ label: "Contract", get: (r) => r.contract }, { label: "Open interest", num: true, get: (r) => Math.round(r.open_interest).toLocaleString("en-US") }, { label: "Volume", num: true, get: (r) => (r.volume === null ? "" : Math.round(r.volume).toLocaleString("en-US")) }], oi) });
  }

  /* ---------- roll ---------- */
  function buildRoll(root, d) {
    const r = d.roll;
    header(root, `Rolling ${d.root}: ${r.current_contract} into ${r.next_contract}`, `${r.front_expiry} expiry · ${r.days_to_expiry} days left · roll ${r.roll_days} days before expiry`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Suggested roll date", W.dayMonth(r.suggested_roll_date), { text: `${r.suggested_roll_date.slice(0, 4)} · in ${r.days_to_roll} day${r.days_to_roll === 1 ? "" : "s"}` }, "", "The earlier of the calendar date and the open-interest crossover");
    tile(k, "Calendar roll", W.dayMonth(r.calendar_roll_date), { text: `${r.calendar_roll_date.slice(0, 4)} · ${r.roll_days} days before expiry` });
    tile(k, "Open-interest crossover", r.oi_crossover ? W.dayMonth(r.oi_crossover) : "not yet", { text: r.oi_crossover ? `${r.oi_crossover.slice(0, 4)} · the next contract holds more` : `${r.next_contract} is still the smaller one` }, "", "The date from which the next contract's open interest stays above the front's");
    tile(k, "Cost of the roll", smoney(-r.roll_cost_dollars), { text: `${pct(r.roll_cost_pct, 2)} of the position's value` }, "", "For a LONG position: the calendar spread you pay when selling the old contract and buying the next, plus a tick of execution cost. Negative means you pay; a short position earns it.");
    tile(k, "Calendar spread", num(r.calendar_spread_points, 3), { text: `${money(r.calendar_spread_dollars)} per contract` }, "", "Next contract's price less the front's");
    tile(k, "Annualised roll", pct(r.annualised_roll, 1), { text: r.annualised_roll >= 0 ? "a long earns" : "a long pays" });
    const hs = r.history, ch = h("div", "charts", root);
    chartCard(ch, "Front and next contract", "The two contracts' prices over the last 120 days; the gap is the calendar spread you cross when you roll.", {
      draw: (b) => C.line(b, { dates: hs.dates, series: [{ name: r.current_contract, color: css("--s1"), values: hs.front }, { name: r.next_contract, color: css("--s2"), values: hs.next }], yFormat: (v) => num(v, 1), tipFormat: (v) => num(v, 3), height: 250, label: "Front and next contract" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => hs.dates[n] }, { label: r.current_contract, num: true, get: (n) => num(hs.front[n], 3) }, { label: r.next_contract, num: true, get: (n) => num(hs.next[n], 3) }, { label: "Spread", num: true, get: (n) => num(hs.spread[n], 3) }], hs.dates.map((_, n) => n).filter((n) => n % 5 === 0)) });
    chartCard(ch, "Calendar spread", "Next contract less the front: above zero is contango, below zero backwardation.", {
      draw: (b) => C.line(b, { dates: hs.dates, series: [{ name: "Spread", color: css("--s1"), values: hs.spread }], area: true, areaBase: 0, baseline: 0, include: 0, yFormat: (v) => num(v, 2), tipFormat: (v) => num(v, 3), height: 250, label: "Calendar spread" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => hs.dates[n] }, { label: "Spread", num: true, get: (n) => num(hs.spread[n], 3) }], hs.dates.map((_, n) => n).filter((n) => n % 5 === 0)) });
    chartCard(ch, "Open interest: where the positions are", "Positions outstanding in the two contracts you roll between, over the last 120 days. A contract's open interest builds as it nears the front; the crossover date is when the next contract has become the liquid one.", { wide: true,
      draw: (b) => C.line(b, { dates: hs.dates, series: [{ name: r.current_contract, color: css("--s1"), values: hs.oi_front }, { name: r.next_contract, color: css("--s2"), values: hs.oi_next }], include: 0, yFormat: (v) => (v >= 1000 ? (v / 1000).toFixed(0) + "k" : String(Math.round(v))), tipFormat: (v) => Math.round(v).toLocaleString("en-US"), height: 250, label: "Open interest" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => hs.dates[n] }, { label: r.current_contract, num: true, get: (n) => Math.round(hs.oi_front[n]).toLocaleString("en-US") }, { label: r.next_contract, num: true, get: (n) => Math.round(hs.oi_next[n]).toLocaleString("en-US") }], hs.dates.map((_, n) => n).filter((n) => n % 5 === 0)) });
    const past = h("div", "card", root); h("h3", "", past, "The rolls of the past year");
    table(past, [{ label: "Contract", get: (x) => x.contract }, { label: "Rolled into", get: (x) => x.next }, { label: "Expiry", get: (x) => x.expiry }, { label: "Calendar roll", get: (x) => x.roll_date }, { label: "Spread at the roll", num: true, get: (x) => num(x.spread, 3) }, { label: "Open-interest crossover", get: (x) => x.oi_crossover || "never" }], r.past_rolls);
    W.hint(root, "The cost shown is for a long position; a short position's is the opposite. Execution cost is one tick per leg. Prices and open interest are simulated unless your file has them.");
  }

  /* ---------- carry and basis ---------- */
  function buildCarry(root, d) {
    const cy = d.carry, b = d.basis;
    header(root, `Carry and basis: ${d.root}`, `${d.curve.name} · ${cy.state}`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Carry, annualised", pct(cy.carry, 1), { text: `21-day average ${pct(cy.carry_ma_21, 1)}` }, "", "The front contract's roll yield: what holding the position earns from the shape of the curve alone");
    tile(k, "Carry against its year", pct(cy.carry_percentile_1y, 0), { text: `z-score ${num(cy.carry_zscore, 1)}` }, "", "Percentile of today's carry among the last year's, and its distance from the average in standard deviations");
    tile(k, "Carry per unit of risk", num(cy.carry_to_vol, 2), { text: `volatility ${pct(cy.excess_return_vol, 0)}` }, "", "Carry divided by the volatility of the contract's excess return");
    tile(k, "Carry earned, last year", pct(cy.carry_earned_1y, 1), { text: `total excess return ${pct(cy.excess_return_1y, 1)}` }, "", "What the roll yield added to a long position over the last year, against what it earned in total");
    if (b.has_spot) {
      tile(k, "Basis to spot", pct(b.basis, 2), { text: `${num(b.basis_points, 2)} points · ${b.dte} days to expiry` }, "", "Front futures price over the spot price less one");
      tile(k, "Basis, annualised", pct(b.basis_annualised, 1), { text: `z-score ${num(b.basis_zscore, 1)}` });
    }
    tile(k, "Calendar basis", pct(b.calendar_basis_annualised, 1), { text: "front against next, annualised" });
    tile(k, "Basis momentum", num(b.basis_momentum * 100, 2) + "%", { text: `percentile ${pct(b.basis_momentum_percentile, 0)}` }, "", "Front-contract return over the last year less the second contract's: commodities whose front has beaten the second tend to keep doing so (Boons and Prado)");
    const ch = h("div", "charts", root), s = cy.series;
    chartCard(ch, "Carry through time", "Annualised roll yield and its 21-day average. Above zero a long position earns carry.", {
      draw: (c) => C.line(c, { dates: s.dates, series: [{ name: "Carry", color: css("--s1"), values: s.carry, width: 1.5 }, { name: "21-day average", color: css("--s2"), values: s.carry_ma }], baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 260, label: "Carry through time", endLabels: false }),
      table: (c) => table(c, [{ label: "Date", get: (n) => s.dates[n] }, { label: "Carry", num: true, get: (n) => pct(s.carry[n], 2) }, { label: "21-day average", num: true, get: (n) => pct(s.carry_ma[n], 2) }], s.dates.map((_, n) => n).filter((n) => n % 10 === 0)) });
    if (b.has_spot) chartCard(ch, "Basis to spot", "Front futures price over spot, less one. It converges to zero at expiry: the sawtooth is the roll to the next contract.", {
      draw: (c) => C.line(c, { dates: b.series.dates, series: [{ name: "Basis", color: css("--s1"), values: b.series.basis }], baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(1) + "%", tipFormat: (v) => pct(v, 2), height: 260, label: "Basis to spot" }),
      table: (c) => table(c, [{ label: "Date", get: (n) => b.series.dates[n] }, { label: "Basis", num: true, get: (n) => pct(b.series.basis[n], 3) }, { label: "Days to expiry", num: true, get: (n) => b.series.dte[n] }], b.series.dates.map((_, n) => n).filter((n) => n % 10 === 0)) });
    chartCard(ch, "Calendar basis", "Next contract against the front, as a share of the front: the term structure's own signal.", {
      draw: (c) => C.line(c, { dates: b.calendar_series.dates, series: [{ name: "Calendar basis", color: css("--s1"), values: b.calendar_series.values }], baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(1) + "%", tipFormat: (v) => pct(v, 2), height: 240, label: "Calendar basis" }),
      table: (c) => table(c, [{ label: "Date", get: (n) => b.calendar_series.dates[n] }, { label: "Calendar basis", num: true, get: (n) => pct(b.calendar_series.values[n], 3) }], b.calendar_series.dates.map((_, n) => n).filter((n) => n % 10 === 0)) });
    chartCard(ch, "Basis momentum", "The front contract's trailing return less the second's. Positive means the front has been stronger: the shape of the curve has been tightening.", {
      draw: (c) => C.line(c, { dates: b.basis_momentum_series.dates, series: [{ name: "Basis momentum", color: css("--s1"), values: b.basis_momentum_series.values }], baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 240, label: "Basis momentum" }),
      table: (c) => table(c, [{ label: "Date", get: (n) => b.basis_momentum_series.dates[n] }, { label: "Basis momentum", num: true, get: (n) => pct(b.basis_momentum_series.values[n], 2) }], b.basis_momentum_series.dates.map((_, n) => n).filter((n) => n % 10 === 0)) });
  }

  /* ---------- margin ---------- */
  function buildMargin(root, d) {
    header(root, "Margin of the book", `${d.positions.length} position${d.positions.length === 1 ? "" : "s"} · equity ${money(d.equity)} · margins × ${num(d.margin_scale, 1)}`);
    W.provenance(root, { ...d, warnings: d.risk_flags });
    const k = h("div", "earnrow", root), usage = d.margin_usage;
    const t = tile(k, "Margin used", pct(usage, 0), { text: `initial ${money(d.initial_margin)} of ${money(d.equity)}`, dir: usage >= 0.7 ? "down" : "" }, "", "Initial margin over account equity; above 70% there is little room for a loss before a margin call");
    W.meter(t, usage, "", 0.7, 1.0);
    tile(k, "Leverage", num(d.leverage, 1) + "×", { text: `gross notional ${money(d.gross_notional)}` }, "", "Gross notional over account equity");
    tile(k, "Net notional", smoney(d.net_notional), { text: "long less short" });
    tile(k, "Excess liquidity", money(d.excess_liquidity), { text: `maintenance margin ${money(d.maintenance_margin)}` }, "", "Equity less the initial margin");
    tile(k, "Distance to a margin call", money(d.buffer_to_call), { text: "equity above maintenance margin" }, "", "How much the account can lose before equity falls to the maintenance margin");
    const ch = h("div", "charts", root), rows = Object.entries(d.by_root).map(([label, value]) => ({ label, value }));
    if (rows.length > 1) chartCard(ch, "Margin by root", "Where the margin goes: a book dominated by one root is a bet on that root.", {
      draw: (b) => C.hbars(b, { rows, format: (v) => money(v), name: "Initial margin", label: "Margin by root", color: () => css("--s1") }),
      table: (b) => table(b, [{ label: "Root", get: (x) => x.label }, { label: "Initial margin", num: true, get: (x) => money(x.value) }, { label: "Share", num: true, get: (x) => pct(x.value / d.initial_margin, 0) }], rows) });
    const card = h("div", "card wide", ch); h("h3", "", card, "Positions");
    table(card, [{ label: "Contract", get: (p) => `${p.root} ${p.contract}` }, { label: "Qty", num: true, get: (p) => num(p.quantity, 0) }, { label: "Price", num: true, get: (p) => num(p.price, 2) }, { label: "Days", num: true, get: (p) => p.dte },
      { label: "Notional", num: true, get: (p) => smoney(p.notional) }, { label: "Initial margin", num: true, get: (p) => money(p.initial_margin) }, { label: "Maintenance", num: true, get: (p) => money(p.maintenance_margin) },
      { label: "Volatility", num: true, get: (p) => pct(p.volatility, 0) }, { label: "A typical day (1 s.d.)", num: true, get: (p) => money(p.daily_1sd) }], d.positions);
    W.hint(root, "Warnings: margin use above 70% of equity, one root above half the margin, a contract within the chosen number of days of expiry, and a book with no cushion for a one-standard-deviation day. The margin figures are illustrative; scale them with the multiplier to see a stress.");
  }

  function buildBacktest(root, d) {
    header(root, `${d.label} on simulated futures`, `${d.text} · roots ${d.roots.join(", ")} · ${d.start} to ${d.end}`);
    W.provenance(root, d);
    W.pipelineResult(root, d);
    W.hint(root, "The futures are a model: the result checks that carry, roll and costs are booked correctly, not whether the strategy earns money in the world. Returns are the excess return of holding the front contract and rolling it; margin and funding are not modelled.");
  }

  Q.registerWorkspace("futures", F);
})();
