/* The Options workspace: what a chain says (overview, chain, smile and skew, surface and term structure), which pricing models reproduce it and what they imply for exotic and American options and for the cost of
   hedging (model fit, model prices, hedging cost), what a book of options is exposed to (Greeks and stress), what a structure pays (payoff, break-evens, probability of profit) and what a rolling strategy would
   have paid (a backtest on a simulated option market). It talks to /api/options/* and /api/models/*; every payload says whether its numbers are a model, a file of yours or platform data, and this page prints that
   beside them. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, money = Q.money, smoney = Q.smoney, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable;
  const GUIDE = "technique-option-chains-and-greeks";
  const STRUCTURES_GUIDE = "technique-option-structures-distribution-and-positioning";
  const GUIDES = { fit: "technique-option-pricing-models-and-calibration", price: "technique-exotic-and-american-options", hedge: "technique-hedging-with-transaction-costs", distribution: STRUCTURES_GUIDE, positioning: STRUCTURES_GUIDE,
    strategy: STRUCTURES_GUIDE, backtest: "technique-option-strategies-and-vol-premium" };
  const ANALYTIC = new Set(["overview", "chain", "smile", "surface", "distribution", "positioning"]);
  const MODEL_MODES = new Set(["fit", "price", "hedge"]);
  const MODEL_SLOT = { market: 0, heston: 1, nig: 2, vg: 3, merton: 4, kou: 6 };                  // a colour for each model that does not change with what else is shown (the palette's own order, without its second green)
  const modelColor = (id) => (id === "bs" || id === "bs_continuous" ? css("--muted") : Q.colorOf(MODEL_SLOT[id] === undefined ? 6 : MODEL_SLOT[id]));
  const ALL_MODELS = ["merton", "kou", "vg", "nig", "heston"];
  const WINDOWS = [["5", "±5%"], ["10", "±10%"], ["15", "±15%"], ["25", "±25%"], ["40", "±40%"]];

  const O = {
    id: "options", root: "#options", view: "#view-options", title: "Options", api: 13, mode: "overview", busy: "", results: {}, errors: {}, cache: new Map(), guard: null, go: null, progress: null,
    about: "Read an option chain, see the distribution it implies and where the open interest sits, see what a book of options is exposed to, draw what a structure pays and test a rolling strategy. The chain is a model quoted at the underlying's real price unless you download a real one (free, below) or bring your own file.",
    modes: [["overview", "Overview"], ["chain", "Chain"], ["smile", "Smile and skew"], ["surface", "Surface and term"], ["distribution", "Implied distribution"], ["positioning", "Open interest and gamma"], ["fit", "Model fit"],
      ["price", "Model prices"], ["hedge", "Hedging cost"], ["risk", "Greeks and risk"], ["strategy", "Strategy builder"], ["backtest", "Backtest"]],
    runLabel: { overview: "Refresh", chain: "Refresh", smile: "Refresh", surface: "Refresh", distribution: "Refresh", positioning: "Refresh", fit: "Fit the models", price: "Price the product", hedge: "Cost the hedge", risk: "Revalue the book",
      strategy: "Draw the payoff", backtest: "Run the backtest" },
    blurb: { overview: "Implied against realised volatility, where it ranks, the put-call balance, the term structure and the skew: the page to open first.",
      chain: "Calls and puts side by side for one expiry: bid, ask, implied volatility, delta, volume and open interest, with the at-the-money row marked and stale or wide quotes flagged.",
      smile: "How implied volatility changes with strike (the smile), across expiries, and what that says about the price of crash protection (risk reversal, butterfly).",
      surface: "The whole implied-volatility surface at once, the term structure and the checks that a surface admits no free lunch.",
      distribution: "What the options say about where the price will be: the distribution the quotes imply against the lognormal, the move the straddle prices in, the model-free volatility behind the VIX, and the cone of the stock's own history.",
      positioning: "Where the open interest sits by strike: the walls, the max-pain strike of an expiry, and the gamma exposure that follows, with the price at which it changes sign.",
      fit: "Fit five pricing models (jumps, infinite-activity jumps, stochastic volatility) to the chain, see which reproduces the smile, and what each says about the odds of a fall.",
      price: "Price a digital, barrier, average-price or American option under every fitted model. Models fitted to the same quotes differ on a vanilla by how well each fits; on an exotic by more. Set the spread beside the vanilla's: what is left is model risk.",
      hedge: "What hedging an option costs when every trade costs money: Leland's price, the band in which not to trade, and a simulation of the rules.",
      risk: "Delta, gamma, vega and theta of a book of options and shares, by expiry, and what it would make or lose when the price and implied volatility move.",
      strategy: "Choose one of twenty-five structures and see its payoff at expiry, today and half way, where the price is likely to end, its profit by price and day, its Greeks by price, its break-evens, maximum profit and loss, margin and odds; set up to three others beside it.",
      backtest: "Sell or buy premium on a schedule on a simulated option market: one strategy, one setting varied to see how fragile the choice is, or every strategy side by side; with the equity curve, the risk measures, where the profit came from by Greek, the costs and the trades." },
    form: { underlying: "", expiry: "", window: "15", rate: "", dividend: "", premium: "15", hideStale: false, minOi: "",
      book: [], spotShocks: "-10, -5, -2, 0, 2, 5, 10", volShocks: "-5, 0, 5, 10", days: "0",
      strategy: "long_straddle", sDte: "30", sFar: "", offset: "5", width: "5", quantity: "1", fill: "mid", spread: "5", commission: "0.65", viewVol: "", viewDrift: "", horizon: "", legs: [], compare: new Set(),
      bt: "short_put", btMode: "single", btDays: "252", btSeed: "0", btCapital: "100000", btPremium: "25", btCommission: "0.65", btFill: "100", btParams: {}, btTiming: false, btWindow: "21", btThreshold: "0",
      sweepParam: "", sweepFrom: "", sweepTo: "", sweepN: "5",
      fitModels: new Set(ALL_MODELS), fitExpiries: "6", fitMinDte: "7", fitWindow: "30",
      priceModels: new Set(ALL_MODELS), pProduct: "vanilla", pRight: "call", pStrike: "100", pBarrierKind: "down-out", pBarrier: "90", pFixings: "12", pPaths: "50000", pSeed: "0",
      hRight: "call", hStrike: "100", hVol: "", hCost: "5", hSteps: "", hRisk: "1", hPaths: "10000", hSeed: "0" },
    catalog: null, plan: null, fitSmile: null, spot: null, expiries: [], used: null,
  };
  const expiryColor = W.palette();
  const setCatalog = (c) => {                                                               // a download changed what exists: the menu follows, and a chain that was deleted is not left selected
    O.catalog = c;
    if (!c.underlyings.some((u) => u.id === O.form.underlying)) { O.form.underlying = (c.underlyings.find((u) => u.id === "SPY") || c.underlyings[0] || {}).id || ""; O.form.expiry = ""; O.spot = null; }
  };
  let itick = null;                                                                          // the server's iTick status, kept by the card for its time estimate
  /* The Free data card: real chains from Cboe (and Yahoo when Cboe cannot be reached), and Bitcoin and Ether from Deribit; and, optionally, the underlying's daily prices (Yahoo or iTick) so that the chain
     has a realised volatility to be compared with. iTick has no option data. */
  const FREE = {
    kind: "options", title: "Real option chains (free)",
    intro: "Download real option quotes, about 15 minutes late, for US-listed tickers from Cboe, or Bitcoin and Ether options from Deribit. No account and no key. A downloaded chain appears in the Underlying menu as “real chain” and is analysed exactly like a model chain. iTick has no option data; it can only supply the underlying's daily prices.",
    defaults: () => ({ chosen: new Set(["SPY"]), extra: "", maxDte: "365", prices: false, pricesSource: "yahoo", pricesYears: "3" }),
    controls: (box, f, ks, changed, redraw) => {
      itick = ks.itick_status;
      h("span", "hint", box, "Pick tickers:");
      W.picks(box, [...ks.symbols.map((s) => [s, s]), ...ks.crypto.map((c) => ["deribit:" + c, c + " · Deribit", "Bitcoin and Ether options from Deribit, live"])], f.chosen, changed);
      Q.labField(box, { key: "extra", label: "Other tickers (separated by spaces)", kind: "text", placeholder: "NVDA AMD", help: "Any ticker that has listed options in the US" }, f, changed);
      Q.labField(box, { key: "maxDte", label: "Expiries up to", kind: "choice", options: ks.max_dte.map((d) => [String(d), d < 365 ? d + " days" : d === 365 ? "1 year" : d / 365 + " years"]), help: "Longer chains are bigger files and slower pages" }, f, changed);
      Q.labField(box, { key: "prices", label: "Also download the underlying's daily prices", kind: "bool", help: "Gives a downloaded chain a realised volatility to set beside the implied one, for a ticker the platform's own price data does not hold" }, f, redraw);
      if (!f.prices) return;
      Q.labField(box, { key: "pricesSource", label: "Prices from", kind: "choice", options: ks.prices.sources.map((x) => [x.id, x.text]) }, f, redraw);
      Q.labField(box, { key: "pricesYears", label: "Years of prices", kind: "choice", options: ks.prices.years.map((y) => [String(y), y === 1 ? "1 year" : y + " years"]) }, f, changed);
      if (f.pricesSource === "itick") W.itickBox(box, ks.itick_status);
      h("p", "hint", box, "The prices are for the realised volatility only. A ticker the platform already holds (the ETFs of the Backtest tab) keeps its own prices.");
    },
    request: (f) => {
      const items = [...f.chosen, ...String(f.extra).split(/[\s,;]+/).filter(Boolean)];
      if (!items.length) throw new Error("Pick a ticker, or type one");
      const body = { kind: "options", items, max_dte: Number(f.maxDte) };
      if (f.prices) Object.assign(body, { what: ["chain", "prices"], prices_source: f.pricesSource, years: Number(f.pricesYears) });
      return body;
    },
    estimate: (b) => {
      let seconds = b.items.length * 4;
      if (!(b.what || []).includes("prices")) return seconds;
      const n = b.items.filter((x) => !String(x).startsWith("deribit:")).length;
      if (b.prices_source !== "itick") return seconds + n * 3;
      const info = itick || {}, cpm = info.calls_per_minute || 5, page = info.page_size || 1000, free = info.calls_available_now === undefined ? cpm : info.calls_available_now;
      const calls = n * Math.max(1, Math.ceil((b.years * ((info.bars_per_year || {}).stock || 252)) / page));
      return seconds + 2 * calls + Math.max(0, calls - free) * (60 / cpm);
    },
    describe: (rec) => rec.item === "prices" ? [`${rec.underlying} · daily prices`, `${rec.source} · ${Number(rec.bars || rec.rows).toLocaleString("en-US")} days to ${rec.as_of}`]
      : [rec.name, `${rec.source} · quotes of ${rec.as_of} · ${Number(rec.contracts || rec.rows).toLocaleString("en-US")} contracts`],
    setCatalog,
    pick: (res) => { const first = res.results.find((r) => r.ok && r.what === "chain"); if (first) { O.form.underlying = "dl:" + first.name; O.form.expiry = ""; O.spot = null; } },
  };
  const nameOf = (d) => (String(d.underlying).startsWith("dl:") ? d.underlying.slice(3) + " (real chain)" : d.underlying);          // "dl:SPY" is an id, not a name
  const fileName = (d) => String(d.underlying).replace(/[^A-Za-z0-9_.-]/g, "_");
  const volFormat = (values) => {                                                           // whole points unless the volatilities are so close that ticks would repeat
    const v = values.filter((x) => x !== null && isFinite(x)), spread = v.length ? Math.max(...v) - Math.min(...v) : 1, d = spread < 0.03 ? 2 : spread < 0.12 ? 1 : 0;
    return (x) => (x * 100).toFixed(d) + "%";
  };

  /* ---------- the data this page needs before it can draw ---------- */
  W.catalog().then((c) => { O.catalog = c; if (!O.form.underlying && c.underlyings.length) O.form.underlying = (c.underlyings.find((u) => u.id === "SPY") || c.underlyings[0]).id; if (!Q.$(O.view).hidden) Q.renderLab(O); }, () => {});
  Q.api("/api/models/plan").then((p) => { O.plan = p; if (!Q.$(O.view).hidden && MODEL_MODES.has(O.mode)) Q.renderLab(O); }, () => {});            // an older app has no plan: the three model modes say so when they run
  const settings = () => ({ underlying: O.form.underlying, rate: O.form.rate === "" ? undefined : Number(O.form.rate) / 100, dividend: O.form.dividend === "" ? undefined : Number(O.form.dividend) / 100, premium: O.form.premium === "" ? undefined : Number(O.form.premium) / 100 });
  const chainQuery = () => ({ ...settings(), expiry: O.form.expiry || undefined, window: Number(O.form.window) / 100 });
  const key = (name, q) => name + ":" + JSON.stringify(q);
  const remember = (d) => { if (d && d.chain) { O.spot = d.chain.spot; O.expiries = d.chain.expiries; O.used = d.chain.expiry; syncExpiry(); } };
  const rememberExpiries = (d) => { if (d && d.expiries) { O.expiries = d.expiries; O.used = d.expiry; if (d.distribution) O.spot = d.distribution.spot; else if (d.positioning) O.spot = d.positioning.spot; syncExpiry(); } };
  function syncExpiry() {                                                                   // the expiry menu follows the chain that was last loaded
    const sel = Q.$(`${O.root} select[data-key=expiry]`); if (!sel) return;
    sel.replaceChildren(); const first = h("option", "", sel, O.mode === "price" || O.mode === "hedge" ? "About three months" : O.mode === "distribution" ? "About 30 days" : O.mode === "positioning" ? "All expiries" : "Nearest"); first.value = "";
    for (const e of O.expiries) { const o = h("option", "", sel, `${e.expiry} · ${e.dte} days`); o.value = e.expiry; if (e.expiry === O.form.expiry) o.selected = true; }
  }
  /* The chain for the current settings, from the cache if it was fetched already. Other modes use its spot and expiries (to suggest strikes). */
  async function context() {
    if (!O.form.underlying) return null;
    const q = chainQuery(), k = key("chain", q); let d = O.cache.get(k);
    if (!d) { try { d = await Q.api("/api/options/chain" + W.qs(q)); O.cache.set(k, d); } catch (e) { return null; } }
    remember(d); return d;
  }

  /* ---------- the form ---------- */
  function strikeNear(x) { const step = O.spot > 500 ? 5 : O.spot > 100 ? 1 : 0.5; return String(Math.round(x / step) * step); }
  function exampleBook() {
    if (!O.spot) { Q.toast("Open the Overview first so that there is a price to put strikes around"); return; }
    O.form.book = [{ type: "option", right: "P", strike: strikeNear(O.spot * 0.95), dte: "30", quantity: "-5" }, { type: "option", right: "C", strike: strikeNear(O.spot * 1.05), dte: "30", quantity: "-5" }, { type: "stock", right: "C", strike: "", dte: "", quantity: "100" }];
  }
  /* The rows of a book or of the legs of a structure: type, call or put, strike, days to expiry, quantity (positive buys, negative sells). Two lines to a row, so that the form fits its narrow column. */
  function legEditor(parent, list, onChange) {
    const box = h("div", "legs", parent);
    const draw = () => {
      box.replaceChildren();
      list.forEach((x, i) => {
        const row = h("div", "leg", box), type = h("select", "", row), right = h("select", "", row), rm = h("button", "iconbtn", row, "×"), nums = h("div", "l2", row);
        const strike = h("input", "", nums), dte = h("input", "", nums), qty = h("input", "", nums);
        for (const [v, t] of [["option", "Option"], ["stock", "Shares"]]) { const o = h("option", "", type, t); o.value = v; if (v === x.type) o.selected = true; }
        for (const [v, t] of [["C", "Call"], ["P", "Put"]]) { const o = h("option", "", right, t); o.value = v; if (v === x.right) o.selected = true; }
        for (const [el, k, ph, title] of [[strike, "strike", "strike", "Strike price"], [dte, "dte", "days", "Days to expiry"], [qty, "quantity", "quantity", "Contracts or shares: positive buys, negative sells"]]) {
          el.type = "number"; el.step = "any"; el.value = x[k]; el.placeholder = ph; el.title = title; el.setAttribute("aria-label", `${title} of row ${i + 1}`); el.addEventListener("input", () => { x[k] = el.value; onChange(); });
        }
        type.setAttribute("aria-label", "type of row " + (i + 1)); right.setAttribute("aria-label", "call or put, row " + (i + 1)); rm.setAttribute("aria-label", "remove row " + (i + 1));
        const sync = () => { const stock = x.type === "stock"; right.disabled = strike.disabled = dte.disabled = stock; };
        type.addEventListener("change", () => { x.type = type.value; sync(); onChange(); }); right.addEventListener("change", () => { x.right = right.value; onChange(); });
        rm.addEventListener("click", () => { list.splice(i, 1); draw(); onChange(); });
        sync();
      });
      if (!list.length) h("p", "hint", box, "No rows yet.");
    };
    draw();
    const row = h("div", "row tight", parent);
    const addOption = h("button", "btn", row, "+ Option"), addStock = h("button", "btn", row, "+ Shares"), clear = h("button", "btn", row, "Clear");
    addOption.addEventListener("click", () => { list.push({ type: "option", right: "C", strike: O.spot ? strikeNear(O.spot) : "", dte: "30", quantity: "1" }); draw(); onChange(); });
    addStock.addEventListener("click", () => { list.push({ type: "stock", right: "C", strike: "", dte: "", quantity: "100" }); draw(); onChange(); });
    clear.addEventListener("click", () => { list.length = 0; draw(); onChange(); });
    return { draw, row };
  }
  function legsBody(list, what) {
    if (!list.length) throw new Error(`Add at least one row to the ${what}`);
    return list.map((x, i) => {
      const q = Number(x.quantity); if (!isFinite(q) || q === 0) throw new Error(`Row ${i + 1} needs a quantity (positive to buy, negative to sell)`);
      if (x.type === "stock") return { type: "stock", quantity: q };
      const k = Number(x.strike), d = Number(x.dte); if (!(k > 0)) throw new Error(`Row ${i + 1} needs a strike`); if (!(d >= 1)) throw new Error(`Row ${i + 1} needs days to expiry of at least 1`);
      return { type: "option", right: x.right, strike: k, dte: d, quantity: q };
    });
  }
  const underlyingField = (box, after) => {
    const f = O.form, cat = O.catalog;
    if (!cat) { h("p", "hint", box, "Loading the list of underlyings…"); return; }
    Q.labField(box, { key: "underlying", label: "Underlying", kind: "choice", options: cat.underlyings.map((u) => [u.id, u.label]), help: "The platform's ETFs, quoted from a model at their real last close, real chains you have downloaded, and any chain file you put in data/user/options/" }, f,
      () => { f.expiry = ""; O.spot = null; after(); });
    const u = cat.underlyings.find((x) => x.id === f.underlying);
    if (u) h("p", "hint", box, u.source === "user_file" ? "Quotes are read from your file." : u.source === "downloaded" ? `Real quotes, delayed: downloaded ${W.ago(u.fetched_at)} for the session of ${u.as_of}. Download again below for fresher ones.`
      : "No option quotes are held for this ETF: the chain is a model (an arbitrage-free surface) anchored to its real price and realised volatility. Download a real chain below to replace it.");
  };
  const SWEEPABLE = (p) => (p.kind === "float" || p.kind === "int") && !["size", "min_open_dte"].includes(p.name);          // the settings worth varying: not the position size or the least days to open
  const backtestInfo = () => O.catalog && O.catalog.option_backtests.find((s) => s.id === O.form.bt);
  function resetSweep() {                                                                   // a range around the setting's current value, within what the strategy accepts
    const info = backtestInfo(), p = info && info.params.find((x) => x.name === O.form.sweepParam);
    if (!p) return;
    const cur = (O.form.btParams[O.form.bt] || {})[p.name], base = cur === undefined || cur === "" ? p.default : Number(cur), fix = (v) => String(p.kind === "int" ? Math.round(v) : Number(v.toPrecision(3)));
    O.form.sweepFrom = fix(Math.max(p.lo, base * 0.4)); O.form.sweepTo = fix(Math.min(p.hi, base * 1.6));
  }
  function sweepValues() {                                                                  // the values to try, evenly spaced from the first to the last (whole numbers for a setting that is one)
    const f = O.form, info = backtestInfo(), p = info && info.params.find((x) => x.name === f.sweepParam);
    if (!p) throw new Error("Choose a setting to vary");
    const lo = Number(f.sweepFrom), hi = Number(f.sweepTo), n = Math.round(Number(f.sweepN));
    if (f.sweepFrom === "" || f.sweepTo === "" || !isFinite(lo) || !isFinite(hi)) throw new Error("Enter the first and the last value to try");
    if (!(hi > lo)) throw new Error("The last value must be larger than the first");
    if (!(n >= 2 && n <= 9)) throw new Error("Try between 2 and 9 values");
    if (lo < p.lo || hi > p.hi) throw new Error(`${p.name.replace(/_/g, " ")} must stay between ${p.lo} and ${p.hi}`);
    const values = [...new Set(Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1)).map((v) => (p.kind === "int" ? Math.round(v) : Number(v.toFixed(6)))))];
    if (values.length < 2) throw new Error("Those values round to the same whole number: widen the range");
    return values;
  }
  O.check = () => {
    if (!O.catalog) return "Loading the list of underlyings…";
    if (!O.form.underlying) return "There is no underlying to analyse.";
    if (O.mode === "strategy" && O.form.compare.size > 3) return "Compare at most three other structures.";
    if (O.mode === "backtest" && O.form.btMode === "sweep") { try { sweepValues(); } catch (e) { return e.message; } }
    if (O.mode === "fit" && !O.form.fitModels.size) return "Choose at least one model to fit.";
    if (O.mode === "price" && !O.form.priceModels.size) return "Choose at least one model to price under.";
    if (O.mode === "price" && O.form.pProduct === "barrier") {
      const level = Number(O.form.pBarrier), down = O.form.pBarrierKind.startsWith("down");
      if (down && !(level < 100)) return "A down barrier must be below the price: under 100% of it.";
      if (!down && !(level > 100)) return "An up barrier must be above the price: over 100% of it.";
    }
    return "";
  };
  O.sections = (panel) => {
    const f = O.form, after = () => Q.syncLab(O), rerun = () => { O.run(true); };
    const sec = (title, open = true) => { const d = h("details", "", panel); d.open = open; h("summary", "", d, title); return h("div", "body", d); };
    const m = O.mode;
    underlyingField(sec("1. Underlying"), ANALYTIC.has(m) ? rerun : () => { context(); after(); });
    if (m !== "backtest") W.dataCard(panel, O, FREE);
    if (m === "chain" || m === "smile" || m === "distribution" || m === "positioning") {
      const b = sec(m === "chain" || m === "smile" ? "2. Expiry and strikes" : "2. Expiry");
      Q.labField(b, { key: "expiry", label: "Expiry", kind: "choice", options: [["", m === "distribution" ? "About 30 days" : m === "positioning" ? "All expiries" : "Nearest"], ...O.expiries.map((e) => [e.expiry, `${e.expiry} · ${e.dte} days`])] }, f, rerun);
      if (m === "chain") Q.labField(b, { key: "window", label: "Strikes within", kind: "choice", options: WINDOWS }, f, rerun);
      if (m === "chain") { Q.labField(b, { key: "hideStale", label: "Hide strikes where both quotes are stale", kind: "bool" }, f, () => O.draw()); Q.labField(b, { key: "minOi", label: "Least open interest (calls + puts)", kind: "number", step: 1, min: 0 }, f, () => O.draw()); }
    }
    if (ANALYTIC.has(m) || MODEL_MODES.has(m) || m === "risk" || m === "strategy") {
      const b = sec("Assumptions", false);
      for (const s of [{ key: "rate", label: "Interest rate (% a year, blank = 4)", kind: "number", step: 0.25 }, { key: "dividend", label: "Dividend yield (% a year, blank = the ETF's)", kind: "number", step: 0.1, min: 0 },
        { key: "premium", label: "Implied over realised variance (%)", kind: "number", step: 1, help: "How much richer the model's implied volatility is than the recent realised volatility. The simulated chain only." }]) Q.labField(b, s, f, after);
      h("p", "hint", b, "Used to build and price the simulated chain. They change the numbers, so they are part of what you are looking at; a chain from your own file ignores them except for the rate and dividend it lacks, and a downloaded chain brings its own (the dividend of each expiry is read from the market's forward).");
    }
    if (m === "fit") modelFitForm(sec);
    if (m === "price") modelPriceForm(sec);
    if (m === "hedge") modelHedgeForm(sec);
    if (m === "risk") {
      const b = sec("2. The book");
      h("p", "hint", b, "Contracts are 100 shares. Quantities: positive buys, negative sells. Strikes and expiries need not be listed: each is priced from the surface at its strike and nearest expiry.");
      legEditor(b, f.book, after);
      const ex = h("button", "btn", b, "Load an example (short strangle with shares)"); ex.addEventListener("click", () => { exampleBook(); Q.renderLab(O); });
      const s = sec("3. Stress");
      Q.labField(s, { key: "spotShocks", label: "Price moves (%)", kind: "text", placeholder: "-10, -5, 0, 5, 10", help: "Up to nine, separated by commas" }, f, after);
      Q.labField(s, { key: "volShocks", label: "Implied volatility moves (points)", kind: "text", placeholder: "-5, 0, 5, 10" }, f, after);
      Q.labField(s, { key: "days", label: "Days that pass", kind: "number", step: 1, min: 0, max: 365, help: "Zero values the book today; more shows the time decay as well" }, f, after);
    }
    if (m === "strategy") {
      const b = sec("2. The structure"), cat = O.catalog;
      if (cat) {
        const opts = [...cat.option_strategies.map((s) => [s.id, s.id.replace(/_/g, " "), s.group]), ["custom", "your own legs", "Your own"]];
        Q.labField(b, { key: "strategy", label: "Structure", kind: "choice", options: opts }, f, () => Q.renderLab(O));
        const info = cat.option_strategies.find((s) => s.id === f.strategy); h("p", "hint", b, f.strategy === "custom" ? "Enter the legs yourself below." : info ? info.text : "");
        if (info) Q.refsLine(b, info.references);
      }
      if (f.strategy === "custom") { legEditor(b, f.legs, after); }
      else {
        for (const s of [{ key: "sDte", label: "Days to expiry", kind: "number", step: 1, min: 1, max: 700 }, { key: "offset", label: "Distance of the short strikes (% of the price)", kind: "number", step: 0.5, min: 0, max: 50 },
          { key: "width", label: "Width of the wings (% of the price)", kind: "number", step: 0.5, min: 0.5, max: 50 }, { key: "quantity", label: "Contracts per leg", kind: "number", step: 1, min: 1, max: 1000 }]) Q.labField(b, s, f, after);
        if (/calendar|diagonal/.test(f.strategy)) Q.labField(b, { key: "sFar", label: "Days to the far expiry (blank = about twice)", kind: "number", step: 1, min: 2 }, f, after);
      }
      const c = sec("3. Costs and view", false);
      Q.labField(c, { key: "fill", label: "Entry price", kind: "choice", options: [["mid", "The model's mid"], ["worst", "Cross the spread"]] }, f, after);
      for (const s of [{ key: "spread", label: "Bid-ask spread as a share of the price (%)", kind: "number", step: 1, min: 0, max: 100 }, { key: "commission", label: "Commission per contract ($)", kind: "number", step: 0.05, min: 0 },
        { key: "viewVol", label: "Volatility for the odds (%, blank = the at-the-money)", kind: "number", step: 0.5, min: 1 }, { key: "viewDrift", label: "Expected return of the underlying (% a year, blank = rate less dividend)", kind: "number", step: 1 },
        { key: "horizon", label: "Days to the horizon (blank = the first expiry)", kind: "number", step: 1, min: 1 }]) Q.labField(c, s, f, after);
      h("p", "hint", c, "The probability of profit and the expected value assume a lognormal price with one volatility, not the smile: read them as a first estimate.");
      if (cat) {
        const v = sec("4. Draw others beside it", false);
        h("p", "hint", v, "Up to three other structures, built on the same expiry, distance, width and size, drawn on the same chart and listed beside this one.");
        W.picks(v, cat.option_strategies.filter((x) => x.id !== f.strategy).map((x) => [x.id, x.id.replace(/_/g, " "), x.text]), f.compare, after);
      }
    }
    if (m === "backtest") {
      const b = sec("2. What to run"), cat = O.catalog;
      if (cat) {
        Q.labField(b, { key: "btMode", label: "Run", kind: "choice", options: [["single", "One strategy"], ["sweep", "One strategy with one setting varied"], ["compare", "Every strategy side by side"]],
          help: "Varying a setting shows how fragile the choice is; the comparison runs every strategy at its defaults on the same market" }, f, () => Q.renderLab(O));
        if (f.btMode === "compare") h("p", "hint", b, "All eleven strategies at their default settings, on the market and with the costs below.");
        else {
          const groups = { short_put: "Sell premium", put_spread: "Sell premium", short_strangle: "Sell premium", iron_condor: "Sell premium", iron_butterfly: "Sell premium", covered_call: "Own the shares", collar: "Own the shares", protective_put: "Own the shares" };
          Q.labField(b, { key: "bt", label: "Strategy", kind: "choice", options: cat.option_backtests.map((x) => [x.id, x.id.replace(/_/g, " "), groups[x.id] || "Trade volatility and skew"]) }, f, () => { resetSweep(); Q.renderLab(O); });
          const info = backtestInfo();
          if (info) {
            h("p", "hint", b, info.text);
            Q.refsLine(b, info.references);
            const grid = h("div", "params", b);
            for (const p of info.params) {
              const cur = f.btParams[f.bt] && p.name in f.btParams[f.bt] ? f.btParams[f.bt][p.name] : p.default, wrap = h("label", "f" + (p.kind === "bool" ? " inline" : ""), grid);
              let input;
              if (p.kind === "bool") { input = h("input", "", wrap); input.type = "checkbox"; input.checked = !!cur; wrap.appendChild(document.createTextNode(p.name.replace(/_/g, " "))); }
              else {
                wrap.appendChild(document.createTextNode(p.name.replace(/_/g, " ")));
                if (p.kind === "choice") { input = h("select", "", wrap); for (const c of p.choices) { const o = h("option", "", input, String(c)); o.value = String(c); if (c === cur) o.selected = true; } }
                else { input = h("input", "", wrap); input.type = "number"; input.step = p.kind === "int" ? "1" : "any"; input.min = p.lo; input.max = p.hi; input.value = cur; }
              }
              wrap.title = p.text;
              input.addEventListener("input", () => { (f.btParams[f.bt] = f.btParams[f.bt] || {})[p.name] = p.kind === "bool" ? input.checked : p.kind === "choice" ? Number(input.value) : input.value; });
              input.addEventListener("change", () => input.dispatchEvent(new Event("input")));
            }
            if (f.btMode === "sweep") {
              const numeric = info.params.filter(SWEEPABLE), sw = sec("3. The setting to vary");
              if (!numeric.some((p) => p.name === f.sweepParam)) { f.sweepParam = (numeric.find((p) => /delta|moneyness|wing/.test(p.name)) || numeric[0]).name; resetSweep(); }
              h("p", "hint", sw, "The backtest is run at each value (evenly spaced from the first to the last) with everything else as set above, on the same market. The settings above are used for the main run.");
              Q.labField(sw, { key: "sweepParam", label: "Setting to vary", kind: "choice", options: numeric.map((p) => [p.name, p.name.replace(/_/g, " ")]) }, f, () => { resetSweep(); Q.renderLab(O); });
              for (const x of [{ key: "sweepFrom", label: "First value", kind: "number", step: "any" }, { key: "sweepTo", label: "Last value", kind: "number", step: "any" }, { key: "sweepN", label: "Number of values (2 to 9)", kind: "number", step: 1, min: 2, max: 9 }]) Q.labField(sw, x, f, after);
              try { h("p", "hint", sw, "Values: " + sweepValues().join(", ")); } catch (e) { h("p", "hint", sw, e.message); }
            }
          }
        }
      }
      const c = sec(f.btMode === "sweep" ? "4. The market and costs" : "3. The market and costs");
      for (const s of [{ key: "btDays", label: "Trading days to simulate (60 to 756)", kind: "number", step: 1, min: 60, max: 756 }, { key: "btSeed", label: "Random seed", kind: "number", step: 1, min: 0 },
        { key: "btCapital", label: "Capital ($)", kind: "number", min: 1000 }, { key: "btPremium", label: "Variance risk premium built into the market (%)", kind: "number", step: 1, help: "How much more the simulated market charges for volatility than it delivers. A seller of premium earns it by construction." },
        { key: "btCommission", label: "Commission per contract ($)", kind: "number", step: 0.05, min: 0 }, { key: "btFill", label: "Share of each order that fills (%)", kind: "number", step: 5, min: 0, max: 100 }]) Q.labField(c, s, f, after);
      h("p", "hint", c, "The first run of a given length builds the simulated market and takes several seconds; the next ones reuse it.");
      const t = sec(f.btMode === "sweep" ? "5. Only when volatility is rich" : "4. Only when volatility is rich", false);
      Q.labField(t, { key: "btTiming", label: "Trade only while implied volatility exceeds recent realised volatility", kind: "bool", help: "The variance-premium filter (Bollerslev, Tauchen and Zhou 2009; Carr and Wu 2009): positions are closed while the premium is below the threshold and opened again when it is above" }, f, () => Q.renderLab(O));
      if (f.btTiming) for (const x of [{ key: "btWindow", label: "Days of realised volatility (5 to 126)", kind: "number", step: 1, min: 5, max: 126 }, { key: "btThreshold", label: "Premium needed (volatility points)", kind: "number", step: 0.5 }]) Q.labField(t, x, f, after);
      if (O.catalog && O.catalog.option_timing) Q.refsLine(t, O.catalog.option_timing.references);
    }
  };

  /* ---------- running ---------- */
  O.run = (fromForm) => {
    const m = O.mode, f = O.form;
    if (ANALYTIC.has(m)) {
      if (!fromForm) O.cache.clear();                                                                       // the Refresh button asks again; a change in the form asks only for what it changed
      return W.request(O, () => {
        if (m === "distribution" || m === "positioning") { const a = { ...settings(), expiry: f.expiry || undefined }; return [{ modes: [m], path: "/api/options/" + m, query: a, cacheKey: key(m, a) }]; }
        const q = chainQuery(), steps = [];
        if (m !== "surface") steps.push({ modes: m === "smile" ? ["overview", "chain", "smile"] : ["overview", "chain"], path: "/api/options/chain", query: q, cacheKey: key("chain", q) });
        if (m === "smile" || m === "surface") { const s = settings(); steps.push({ modes: m === "surface" ? ["surface"] : [], path: "/api/options/surface", query: s, cacheKey: key("surface", s) }); }
        return steps;
      }, "Reading the chain…").then(() => {
        const d = O.results[m] && O.results[m].data;
        if (d && (m === "distribution" || m === "positioning")) rememberExpiries(d);
        else if (d && m !== "surface") remember(d);
        if (m === "chain" || m === "smile") syncExpiry();
      });
    }
    if (m === "fit") return W.request(O, () => {
      if (!f.fitModels.size) throw new Error("Choose at least one model to fit");
      return [{ modes: ["fit"], job: true, path: "/api/models/fit", body: { ...settings(), models: [...f.fitModels], max_expiries: Number(f.fitExpiries), min_dte: Number(f.fitMinDte), window: Number(f.fitWindow) / 100 } }];
    }, "Fitting the models…");
    if (m === "price") return W.request(O, () => {
      if (!f.priceModels.size) throw new Error("Choose at least one model to price under");
      const body = { ...settings(), product: f.pProduct, call: f.pRight === "call", expiry: f.expiry || undefined, strike_pct: Number(f.pStrike), models: [...f.priceModels] };
      if (f.pProduct === "barrier") Object.assign(body, { barrier_kind: f.pBarrierKind, barrier_pct: Number(f.pBarrier) });
      if (f.pProduct === "asian") body.fixings = Number(f.pFixings);
      if (f.pProduct === "barrier" || f.pProduct === "asian") Object.assign(body, { paths: Number(f.pPaths), seed: Number(f.pSeed) });
      return [{ modes: ["price"], job: true, path: "/api/models/price", body }];
    }, "Pricing under the models…");
    if (m === "hedge") return W.request(O, () => [{ modes: ["hedge"], job: true, path: "/api/models/hedge", body: { ...settings(), call: f.hRight === "call", expiry: f.expiry || undefined, strike_pct: Number(f.hStrike),
      vol: f.hVol === "" ? undefined : Number(f.hVol), cost_bp: Number(f.hCost), steps: f.hSteps === "" ? undefined : Number(f.hSteps), risk_aversion: Number(f.hRisk), paths: Number(f.hPaths), seed: Number(f.hSeed) } }], "Costing the hedge…");
    if (m === "risk") return W.request(O, () => {
      const spots = W.percentList(f.spotShocks, "Price moves"), vols = W.percentList(f.volShocks, "Implied volatility moves");
      return [{ modes: ["risk"], path: "/api/options/portfolio-risk", body: { ...settings(), positions: legsBody(f.book, "book"), spot_shocks: spots, vol_shocks: vols, days: Number(f.days) || 0 } }];
    }, "Revaluing the book…");
    if (m === "strategy") return W.request(O, () => {
      const body = { ...settings(), strategy: f.strategy === "custom" ? undefined : f.strategy, fill: f.fill };
      if (f.strategy === "custom") body.legs = legsBody(f.legs, "structure");
      else Object.assign(body, { dte: Number(f.sDte), far_dte: f.sFar === "" ? undefined : Number(f.sFar), offset: Number(f.offset) / 100, width: Number(f.width) / 100, quantity: Number(f.quantity) });
      Object.assign(body, { spread_pct: Number(f.spread) / 100, commission: Number(f.commission), view_vol: f.viewVol === "" ? undefined : Number(f.viewVol) / 100, view_drift: f.viewDrift === "" ? undefined : Number(f.viewDrift) / 100, horizon_days: f.horizon === "" ? undefined : Number(f.horizon),
        compare: [...f.compare].filter((x) => x !== f.strategy) });
      return [{ modes: ["strategy"], path: "/api/options/strategy/payoff", body }];
    }, "Pricing the legs…");
    const params = {};
    for (const [k, v] of Object.entries(f.btParams[f.bt] || {})) if (v !== "" && v !== undefined) params[k] = v;
    return W.request(O, () => {
      const market = { days: Number(f.btDays), seed: Number(f.btSeed), capital: Number(f.btCapital), premium: Number(f.btPremium) / 100, commission: Number(f.btCommission), fill_fraction: Number(f.btFill) / 100,
        timing: f.btTiming ? { window: Number(f.btWindow), threshold: Number(f.btThreshold) / 100 } : undefined };
      O.results.backtest_sweep = null;                                                      // a sweep from an earlier run does not belong beside this one
      if (f.btMode === "compare") return [{ modes: ["backtest"], job: true, path: "/api/options/strategy/compare", body: market }];
      const steps = [{ modes: ["backtest"], job: true, path: "/api/options/strategy/backtest", body: { strategy: f.bt, params, ...market } }];
      if (f.btMode === "sweep") steps.push({ modes: ["backtest_sweep"], job: true, path: "/api/options/strategy/sweep", body: { strategy: f.bt, params, sweep_param: f.sweepParam, sweep_values: sweepValues(), ...market } });
      return steps;
    }, f.btMode === "compare" ? "Running every strategy…" : f.btMode === "sweep" ? "Running the backtest and the sweep…" : "Running the backtest…");
  };
  O.auto = () => { if (ANALYTIC.has(O.mode) && O.form.underlying) O.run(true); };
  O.empty = (root) => {
    const text = { fit: ["Fit the pricing models to this chain", "Fit Merton, Kou, variance gamma, NIG and Heston to the out-of-the-money quotes and see which reproduces the smile, how much better it does than a flat volatility, and what each implies for the odds of a fall. About half a minute for all five."],
      price: ["Price a product under every model", "Choose a digital, barrier, average-price or American option. Each model is fitted to the chain (once per session) and prices it. The spread between them, set beside their spread on the plain option at the same strike (which is only how well each fits), is the model risk of the product, which no vanilla quote can tell you."],
      hedge: ["Cost the hedge of an option", "Hedging a sold option means trading the underlying, and every trade costs. See what that does to the price, where the no-trade band lies and which rule loses least on simulated paths."],
      risk: ["Value a book of options", "Add the options and shares you hold (or load the example), choose the moves to test and revalue: Greeks by expiry and a stress grid of price against implied volatility."],
      strategy: ["Draw what a structure pays", "Pick a structure and press the button: payoff at expiry, today and half way, break-evens, the most it can make and lose, the margin it needs and the odds."],
      backtest: ["Backtest a rolling option strategy", "Sell or buy premium on a schedule on a simulated option market, with the costs and the hedging booked; vary one setting to see how fragile the choice is, or run every strategy side by side. The market pays the premium it was built with, so this checks the machinery."] }[O.mode] || ["Read the chain", "Choose an underlying."];
    Q.labEmpty(root, O, text[0], [text[1]], O.runLabel[O.mode]);
  };
  O.draw = () => Q.drawLab(O);
  O.build = (root, d, body) => ({ overview: buildOverview, chain: buildChain, smile: buildSmile, surface: buildSurface, distribution: buildDistribution, positioning: buildPositioning, fit: buildFit, price: buildPrice, hedge: buildHedge, risk: buildRisk, strategy: buildStrategy, backtest: buildBacktest })[O.mode](root, d, body);

  /* ---------- overview ---------- */
  const header = (root, title, sub, guide) => Q.labHead(root, title, sub, guide || GUIDE);
  function buildOverview(root, d) {
    const ov = d.overview, ref = ov.reference_index, sk = ov.skew || {};
    remember(d);
    header(root, `${nameOf(d)} options`, `${money(ov.spot, 2)} · nearest expiry ${ov.nearest_expiry} (${ov.nearest_dte} days) · ${d.chain.quotes} quotes in that expiry`);
    W.provenance(root, { ...d, warnings: ov.risk_flags });
    const k = h("div", "earnrow", root);
    tile(k, "Implied volatility, 30 days", pct(ov.atm_iv_30d), { text: `${W.points(ov.iv_premium)} against realised` }, "", "At-the-money implied volatility interpolated to 30 days");
    tile(k, "Realised volatility, 21 days", pct(ov.realized_vol_21d), { text: `63 days ${pct(ov.realized_vol_63d)} · long run ${pct(ov.long_run_vol)}` }, "", "Annualised standard deviation of the underlying's daily returns");
    tile(k, "IV rank, one year", pct(ov.iv_rank, 0), { text: `percentile ${pct(ov.iv_percentile, 0)}${d.source === "user_file" ? "" : " · model-based"}` }, "", ov.iv_rank_basis);
    tile(k, "Put-call ratio", num(ov.put_call_volume, 2), { text: `open interest ${num(ov.put_call_oi, 2)}` }, "", "Put volume over call volume across all expiries (above 1: more puts traded)");
    tile(k, "Term structure", W.points(ov.term_slope), { text: `${ov.term_inverted ? "inverted: the front is dearer" : "normal"} · 90 days ${pct(ov.atm_iv_90d)}` }, "", "Implied volatility at 90 days less at 30 days, in volatility points");
    tile(k, "25Δ risk reversal", sk.rr_25d === undefined ? "n/a" : W.points(sk.rr_25d), { text: sk.bf_25d === undefined ? "" : `butterfly ${W.points(sk.bf_25d)}` }, "", "Call volatility less put volatility at 25 delta: negative when puts are dearer (crash protection costs more)");
    tile(k, "At-the-money spread", pct(ov.atm_spread_pct, 1), { text: `liquidity score ${num(ov.snapshot.liquidity_score, 0)} of 100` }, "", "Bid-ask spread of the at-the-money options as a share of their mid");
    tile(k, "Stale quotes", ov.quote_ages === false ? "n/a" : pct(ov.stale_share, 0), { text: ov.quote_ages === false ? "the data has no quote times" : "older than 15 minutes" }, "", "Share of the chain's quotes older than 15 minutes");
    if (ref) tile(k, `${ref.name} (real index)`, num(ref.level * 100, 1), { text: `rank ${pct(ref.rank_1y, 0)} · chain ${pct(ref.chain_atm_iv, 1)}` }, "", `The published ${ref.tenor_days}-day implied-volatility index for this underlying, as of ${ref.as_of}, against the chain's own at-the-money volatility`);
    const ch = h("div", "charts", root), term = d.term;
    chartCard(ch, "Term structure of at-the-money volatility", "Implied volatility of the at-the-money option by days to expiry. Upward is the normal shape; downward means the market expects trouble soon.", {
      draw: (b) => C.line(b, { xs: term.map((t) => t.dte), xWhole: true, xLabel: "Days to expiry", series: [{ name: "At-the-money implied volatility", color: css("--s1"), values: term.map((t) => t.atm_iv), dots: true, dotSize: 3 }],
        yFormat: volFormat(term.map((t) => t.atm_iv)), tipFormat: (v) => pct(v, 1), xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `${term[i].dte} days (${term[i].expiry})`, height: 250, label: "Term structure", vlines: [{ x: 30, label: "30 days" }, { x: 90, label: "90 days" }] }),
      table: (b) => table(b, [{ label: "Expiry", get: (t) => t.expiry }, { label: "Days", num: true, get: (t) => t.dte }, { label: "At-the-money volatility", num: true, get: (t) => pct(t.atm_iv, 2) }, { label: "Forward", num: true, get: (t) => num(t.forward, 2) }], term) });
    smileCard(ch, d.smile, d.chain.spot, "Smile of the first expiry");
    const flagsNow = ov.risk_flags || [];
    if (!flagsNow.length) W.hint(root, "Nothing in the chain trips a warning: spreads are tight, the term structure is normal and implied volatility is neither at the top nor at the bottom of its range.");
  }
  function smileCard(parent, sm, spot, title) {
    const p = sm.points, ks = p.map((x) => x.strike);
    return chartCard(parent, title, `Implied volatility of the out-of-the-money options at each strike for ${sm.expiry} (${sm.dte} days). Higher on the left than the right is put skew.`, {
      draw: (b) => C.line(b, { xs: ks, xLabel: "Strike", series: [{ name: "Implied volatility", color: css("--s1"), values: p.map((x) => x.iv), dots: true, dotSize: 3 }], yFormat: volFormat(p.map((x) => x.iv)), tipFormat: (v) => pct(v, 1),
        xFormat: (v) => num(v, 0), tipTitle: (i) => `Strike ${num(ks[i], 2)} (${p[i].right === "P" ? "put" : "call"}, ${(p[i].moneyness * 100 - 100).toFixed(1)}% from the price)`, height: 250, label: "Volatility smile", vlines: [{ x: spot, label: "Price" }, { x: sm.forward, label: "Forward" }] }),
      table: (b) => table(b, [{ label: "Strike", num: true, get: (x) => num(x.strike, 2) }, { label: "Strike ÷ price", num: true, get: (x) => num(x.moneyness, 3) }, { label: "Option", get: (x) => (x.right === "P" ? "put" : "call") }, { label: "Implied volatility", num: true, get: (x) => pct(x.iv, 2) }], p) });
  }

  /* ---------- chain ---------- */
  function buildChain(root, d) {
    const ch = d.chain, f = O.form;
    remember(d);
    header(root, `${nameOf(d)} chain: ${ch.expiry}`, `${ch.dte} days · forward ${num(ch.forward, 2)} · at-the-money volatility ${pct(ch.atm_iv, 1)} · ${ch.quotes} quotes`);
    W.provenance(root, { ...d, warnings: d.warnings.filter((w) => !d.overview.risk_flags.includes(w)) });
    const row = h("div", "filters", root);
    W.pills(row, ch.expiries.slice(0, 20).map((e) => [e.expiry, `${e.expiry.slice(5)} · ${e.dte}d`, `${e.expiry}: ${e.dte} days, ${e.strikes} strikes`]), ch.expiry, (v) => { f.expiry = v; O.run(true); });
    const k = h("div", "earnrow", root), t = ch.totals;
    tile(k, "Call volume", grouped(t.call_volume), { text: `open interest ${grouped(t.call_oi)}` });
    tile(k, "Put volume", grouped(t.put_volume), { text: `open interest ${grouped(t.put_oi)}` });
    tile(k, "Put-call ratio", num(t.put_call_volume, 2), { text: `open interest ${num(t.put_call_oi, 2)}` }, "", "Put volume over call volume in this expiry");
    tile(k, "Stale quotes", ch.quote_ages === false ? "n/a" : `${ch.stale_quotes} of ${ch.quotes}`, { text: `${ch.wide_quotes} with a spread over 10%` }, "", "A quote older than 15 minutes, or a spread wider than a tenth of the mid, is not a price you can trade at");
    const card = h("div", "card", root), scroll = h("div", "scroll", card); scroll.style.maxHeight = "560px";
    const minOi = Number(f.minOi) || 0;
    const rows = ch.rows.filter((r) => !(f.hideStale && r.call.stale && r.put.stale) && r.call.oi + r.put.oi >= minOi);
    chainTable(scroll, rows);
    const chainSource = (d.sources || {}).chain, ages = ch.quote_ages !== false;
    const provenance = chainSource === "simulated" ? "Volume, open interest and quote ages are simulated." : `Volume and open interest are ${chainSource === "downloaded" ? "the provider's, as downloaded" : "from your file"}${ages ? "" : "; there are no quote times, so a stale quote is not flagged"}.`;
    W.hint(card, `${rows.length} of ${ch.rows.length} strikes shown, within ${Math.round(Number(f.window))}% of the price. The at-the-money row is outlined; shaded cells are in the money; ${ages ? "⚠ marks a quote older than 15 minutes and " : ""}orange spreads are wider than 10% of the mid. ${provenance}`);
    const dl = h("button", "btn", card, "Download this expiry (CSV)");
    dl.addEventListener("click", () => W.csvDownload(`${fileName(d)}_${ch.expiry}_chain.csv`, [["strike", "call_bid", "call_ask", "call_iv", "call_delta", "call_volume", "call_oi", "put_bid", "put_ask", "put_iv", "put_delta", "put_volume", "put_oi"],
      ...ch.rows.map((r) => [r.strike, r.call.bid, r.call.ask, r.call.iv, r.call.delta, r.call.volume, r.call.oi, r.put.bid, r.put.ask, r.put.iv, r.put.delta, r.put.volume, r.put.oi])]));
  }
  const grouped = (v) => (Q.bad(v) ? "n/a" : Math.round(v).toLocaleString("en-US"));
  function chainTable(parent, rows) {
    const t = h("table", "chain", parent), head = h("thead", "", t), top = h("tr", "", head);
    const calls = h("th", "side", top, "Calls"); calls.colSpan = 7; h("th", "side", top, ""); const puts = h("th", "side", top, "Puts"); puts.colSpan = 7;
    const cols = ["Bid", "Ask", "IV", "Δ", "Volume", "Open int.", "Spread"], second = h("tr", "", head);
    for (const c of cols) h("th", "", second, c); h("th", "", second, "Strike"); for (const c of cols) h("th", "", second, c);
    const body = h("tbody", "", t);
    const side = (tr, s) => {
      const cls = (extra) => (s.itm ? "itm " : "") + (extra || "");
      h("td", cls(), tr, num(s.bid, 2)); h("td", cls(), tr, num(s.ask, 2));
      const iv = h("td", cls(s.stale ? "stale" : ""), tr, pct(s.iv, 1)); if (s.stale) { const m = h("span", "warnmark", iv, "⚠"); m.title = "stale: the quote is older than 15 minutes"; }
      h("td", cls(), tr, num(s.delta, 2)); h("td", cls(), tr, grouped(s.volume)); h("td", cls(), tr, grouped(s.oi));
      const sp = h("td", cls(s.spread_pct > 0.1 ? "wide" : ""), tr, pct(s.spread_pct, 1)); if (s.spread_pct > 0.1) sp.title = "wider than 10% of the mid";
    };
    for (const r of rows) {
      const tr = h("tr", r.atm ? "atm" : "", body); side(tr, r.call); h("td", "strike", tr, num(r.strike, 2)); side(tr, r.put);
    }
  }

  /* ---------- smile and skew ---------- */
  function buildSmile(root, d, body) {
    const ch = d.chain, sk = d.skew || {}, surf = O.cache.get(key("surface", settings()));
    remember(d);
    header(root, `${nameOf(d)} smile and skew`, `${ch.expiry} · ${ch.dte} days · at-the-money volatility ${pct(ch.atm_iv, 1)}`);
    W.provenance(root, { ...d, warnings: [] });
    const row = h("div", "filters", root);
    W.pills(row, ch.expiries.slice(0, 20).map((e) => [e.expiry, `${e.expiry.slice(5)} · ${e.dte}d`, `${e.expiry}: ${e.dte} days`]), ch.expiry, (v) => { O.form.expiry = v; O.run(true); });
    const k = h("div", "earnrow", root);
    tile(k, "At-the-money", pct(sk.atm_iv, 1), { text: `${ch.dte} days` });
    tile(k, "25Δ put volatility", pct(sk.put_25d_iv, 1), { text: `call ${pct(sk.call_25d_iv, 1)}` });
    tile(k, "25Δ risk reversal", W.points(sk.rr_25d), { text: "call less put volatility" }, "", "Negative: the market pays more for downside protection than for upside");
    tile(k, "25Δ butterfly", W.points(sk.bf_25d), { text: "wings over the middle" }, "", "Average of the 25-delta put and call volatilities less the at-the-money: the price of both tails (curvature)");
    tile(k, "10Δ risk reversal", W.points(sk.rr_10d), { text: `butterfly ${W.points(sk.bf_10d)}` }, "", "The same measures further out in the tails");
    tile(k, "Skew slope", sk.skew_slope === undefined ? "n/a" : num(sk.skew_slope, 2), { text: "volatility change per unit of log-moneyness" });
    const grid = h("div", "charts", root);
    smileCard(grid, d.smile, ch.spot, `Smile: ${ch.expiry}`);
    if (surf) {
      const s = surf.surface, picks = [], want = [7, 30, 60, 120, 240]; for (const w of want) { const i = s.expiries.reduce((best, e, n) => (Math.abs(e.dte - w) < Math.abs(s.expiries[best].dte - w) ? n : best), 0); if (!picks.includes(i)) picks.push(i); }
      chartCard(grid, "Smiles by expiry", "Implied volatility by strike ÷ price for expiries near a week, a month, two, four and eight months. Short expiries are steeper; the smile flattens with time.", {
        draw: (b) => C.line(b, { xs: s.moneyness, xLabel: "Strike ÷ price", series: picks.map((i) => ({ name: `${s.expiries[i].dte} days`, color: expiryColor(s.expiries[i].expiry), values: s.iv[i] })), yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1),
          xFormat: (v) => num(v, 2), height: 250, label: "Smiles by expiry", vlines: [{ x: 1, label: "Price" }], endLabels: false }),
        table: (b) => table(b, [{ label: "Strike ÷ price", num: true, get: (n) => num(s.moneyness[n], 3) }, ...picks.map((i) => ({ label: `${s.expiries[i].dte} days`, num: true, get: (n) => (s.iv[i][n] === null ? "" : pct(s.iv[i][n], 2)) }))], s.moneyness.map((_, n) => n)) });
      const sks = surf.skews;
      chartCard(grid, "Skew by expiry", "Risk reversal (call less put volatility) and butterfly at 25 delta, in volatility points. A risk reversal below zero is put skew.", {
        draw: (b) => C.line(b, { xs: sks.map((x) => x.dte), xWhole: true, xLabel: "Days to expiry", series: [{ name: "25Δ risk reversal", color: css("--s1"), values: sks.map((x) => x.rr_25d * 100), dots: true }, { name: "25Δ butterfly", color: css("--s2"), values: sks.map((x) => x.bf_25d * 100), dots: true }],
          baseline: 0, include: 0, yFormat: (v) => v.toFixed(0), tipFormat: (v) => v.toFixed(1) + " pts", xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `${sks[i].dte} days (${sks[i].expiry})`, height: 250, label: "Skew by expiry", endLabels: false }),
        table: (b) => table(b, [{ label: "Expiry", get: (x) => x.expiry }, { label: "Days", num: true, get: (x) => x.dte }, { label: "At the money", num: true, get: (x) => pct(x.atm_iv, 1) }, { label: "25Δ put", num: true, get: (x) => pct(x.put_25d_iv, 1) }, { label: "25Δ call", num: true, get: (x) => pct(x.call_25d_iv, 1) },
          { label: "25Δ RR (pts)", num: true, get: (x) => num(x.rr_25d * 100, 2) }, { label: "25Δ BF (pts)", num: true, get: (x) => num(x.bf_25d * 100, 2) }, { label: "10Δ RR (pts)", num: true, get: (x) => num(x.rr_10d * 100, 2) }, { label: "10Δ BF (pts)", num: true, get: (x) => num(x.bf_10d * 100, 2) }], sks) });
    } else W.hint(root, "The smiles of other expiries load with the surface (press Refresh).");
  }

  /* ---------- surface and term structure ---------- */
  function buildSurface(root, d) {
    const s = d.surface, tm = d.term_metrics;
    header(root, `${nameOf(d)} implied-volatility surface`, `${s.expiries.length} expiries · strikes from ${num(s.moneyness[0], 2)} to ${num(s.moneyness[s.moneyness.length - 1], 2)} times the price`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "30-day volatility", pct(tm.iv_30d, 1)); tile(k, "90-day volatility", pct(tm.iv_90d, 1));
    tile(k, "Term slope", W.points(tm.term_slope), { text: tm.inverted ? "inverted" : "normal" }, "", "90-day less 30-day at-the-money volatility");
    const ch = h("div", "charts", root), atm = s.moneyness.reduce((best, m, i) => (Math.abs(m - 1) < Math.abs(s.moneyness[best] - 1) ? i : best), 0);
    chartCard(ch, "Implied volatility by expiry and strike", "Rows are expiries (days to expiry), columns are strike ÷ price; darker is higher volatility. The outlined column is at the money.", { wide: true,
      draw: (b) => C.matrix(b, { rows: s.expiries.map((e) => `${e.dte}d`), cols: s.moneyness.map((m) => num(m, 2)), values: s.iv, format: (v) => (v * 100).toFixed(0), tipFormat: (v) => pct(v, 1), name: "Implied volatility", mark: (r, c) => c === atm,
        rowTitle: "Days to expiry", colTitle: "Strike ÷ price", label: "Implied volatility surface", cellText: true }),
      table: (b) => table(b, [{ label: "Expiry", get: (n) => `${s.expiries[n].expiry} (${s.expiries[n].dte}d)` }, ...[0, Math.floor(s.moneyness.length / 4), Math.floor(s.moneyness.length / 2), Math.floor((3 * s.moneyness.length) / 4), s.moneyness.length - 1].map((c) => ({ label: `K/S ${num(s.moneyness[c], 2)}`, num: true, get: (n) => (s.iv[n][c] === null ? "" : pct(s.iv[n][c], 1)) }))], s.expiries.map((_, n) => n)) });
    const term = d.term;
    chartCard(ch, "Term structure", "At-the-money implied volatility by days to expiry.", {
      draw: (b) => C.line(b, { xs: term.map((t) => t.dte), xWhole: true, xLabel: "Days to expiry", series: [{ name: "At-the-money implied volatility", color: css("--s1"), values: term.map((t) => t.atm_iv), dots: true, dotSize: 3 }], yFormat: volFormat(term.map((t) => t.atm_iv)), tipFormat: (v) => pct(v, 1),
        xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `${term[i].dte} days (${term[i].expiry})`, height: 250, label: "Term structure" }),
      table: (b) => table(b, [{ label: "Expiry", get: (t) => t.expiry }, { label: "Days", num: true, get: (t) => t.dte }, { label: "At-the-money volatility", num: true, get: (t) => pct(t.atm_iv, 2) }], term) });
    const arb = h("div", "card", ch); h("h3", "", arb, "Does the surface admit a free lunch?");
    h("p", "note", arb, "Static no-arbitrage checks on the quotes: a vertical spread cannot be worth more than its width or less than nothing, a butterfly cannot cost less than zero, and a longer option cannot be worth less than a shorter one at the same strike. Violations on prices rounded to a cent are mostly tick-size noise; many large ones mean the surface is not usable.");
    table(arb, [{ label: "Check", get: (x) => x.check }, { label: "Tested", num: true, get: (x) => grouped(x.tested) }, { label: "Violations", num: true, get: (x) => grouped(x.violations) }, { label: "Share", num: true, get: (x) => pct(x.tested ? x.violations / x.tested : null, 1) }], d.arbitrage);
  }

  /* ---------- implied distribution ---------- */
  const sources = (root, refs, topics) => {                                                 // the papers and books behind a view, folded away under a heading
    const shown = topics.filter(([, k]) => refs && refs[k] && refs[k].length);
    if (!shown.length) return;
    const dd = h("details", "how", root); h("summary", "", dd, "Where these ideas come from");
    for (const [title, k] of shown) { h("p", "hint", dd, title); Q.refsLine(dd, refs[k]); }
  };
  const bigMoney = (v) => (Q.bad(v) ? "n/a" : (v < 0 ? "−" : "") + "$" + (Math.abs(v) >= 1e9 ? (Math.abs(v) / 1e9).toFixed(1) + "B" : Math.abs(v) >= 1e6 ? (Math.abs(v) / 1e6).toFixed(1) + "M" : Math.abs(v) >= 1e3 ? (Math.abs(v) / 1e3).toFixed(0) + "k" : Math.abs(v).toFixed(0)));
  function buildDistribution(root, d) {
    const dist = d.distribution, em = dist.expected_move, mf = d.model_free, mom = dist.moments, fit = dist.fit, q = dist.quantiles;
    rememberExpiries(d);
    header(root, `${nameOf(d)}: what the options imply`, `${dist.expiry} · ${dist.dte} days · forward ${num(dist.forward, 2)} · at-the-money volatility ${pct(dist.atm_iv, 1)}`, GUIDES.distribution);
    W.provenance(root, d);
    W.pills(h("div", "filters", root), d.expiries.slice(0, 20).map((e) => [e.expiry, `${e.expiry.slice(5)} · ${e.dte}d`, `${e.expiry}: ${e.dte} days`]), dist.expiry, (v) => { O.form.expiry = v; O.run(true); }, "expiry");
    const fall = dist.tails.find((t) => t.side === "fall" && t.move === 0.1) || dist.tails.find((t) => t.side === "fall"), k = h("div", "earnrow", root);
    tile(k, "Move the straddle prices in", "±" + pct(em.straddle_pct, 1), { text: `±${num(em.straddle, 2)} · one standard deviation ±${pct(em.one_sd_pct, 1)}` }, "", "The price of the at-the-money call and put together as a share of the price. For a normal move it is about 0.8 of a standard deviation, so the two differ when the smile does");
    tile(k, "68% of outcomes between", `${num(em.band_low, 0)} to ${num(em.band_high, 0)}`, { text: `median ${num(q["0.5"], 0)} · forward ${num(dist.forward, 0)}` }, "", "The 16th to the 84th percentile of the implied distribution: the range that holds 68% of the probability the options put on the price at this expiry");
    if (fall) tile(k, `Chance of a ${Math.round(fall.move * 100)}% fall`, pct(fall.implied, fall.implied < 0.1 ? 2 : 1), { text: `lognormal ${pct(fall.lognormal, 2)}${fall.lognormal > 1e-6 ? " · " + (fall.implied / fall.lognormal > 100 ? "over 100" : num(fall.implied / fall.lognormal, 1)) + "×" : ""}` }, "", "The probability the options put on the price ending this far below today's, against a lognormal distribution with the at-the-money volatility. The ratio is what the smile adds: the market's price of crash risk");
    tile(k, "Skewness of the return", num(mom.skewness, 2), { text: `excess kurtosis ${num(mom.excess_kurtosis, 2)}` }, "", "Of the log return implied by the options: below zero is a longer tail to the downside, above zero the upside; the excess kurtosis is how much fatter the tails are than a normal's");
    tile(k, "Model-free volatility, 30 days", pct(mf.iv_30d, 1), { text: `at the money ${pct(mf.atm_30d, 1)}${Q.bad(mf.realized_21d) ? "" : " · realised " + pct(mf.realized_21d, 1)}` }, "", "Every out-of-the-money option weighted by 1/K², the rule the VIX uses (Cboe; Britten-Jones and Neuberger 2000): the volatility a variance swap would pay, interpolated to 30 days");
    if (mf.reference) tile(k, `${mf.reference.name} (published index)`, num(mf.reference.level * 100, 1), { text: `this chain, model-free: ${num(mf.reference.model_free * 100, 1)}` }, "", `The ${mf.reference.tenor_days}-day implied-volatility index as of ${mf.reference.as_of}, set beside the model-free 30-day figure of this chain`);
    tile(k, "How well the curve fits", `${num(fit.max_error_vol * 100, 2)} pts`, { text: `${fit.quotes} quotes · ${fit.arbitrage_free ? "no butterfly arbitrage" : "butterfly arbitrage in the fit"}` }, "", "The largest gap, in volatility points, between the fitted SVI curve and a quote it was fitted to. A curve that misses the quotes by points gives a density you should not trust");
    const ch = h("div", "charts", root);
    chartCard(ch, `What the options imply for the price in ${dist.dte} days`, `The chance, per dollar, that ${nameOf(d)} ends at each price on ${dist.expiry}, read from the quotes (solid) against a lognormal with the at-the-money volatility (dashed). More on the left than the dashed line is the price of crash protection; the shaded band holds 68% of the probability. It is the distribution the options are priced on, with the market's price of risk inside it, not a forecast.`, { wide: true,
      draw: (b) => C.line(b, { xs: dist.price, xLabel: "Price at expiry", series: [{ name: "Implied by the options", color: css("--s1"), values: dist.density }, { name: "Lognormal, at-the-money volatility", color: css("--muted"), dash: "5 4", width: 1.5, values: dist.lognormal }], area: true, areaBase: 0, include: 0, baseline: 0,
        yFormat: (v) => (v * 100).toFixed(1) + "%", tipFormat: (v) => (v * 100).toFixed(3) + "% per $1", xFormat: (v) => num(v, 0), height: 300, label: "Implied price distribution", regions: [{ from: em.band_low, to: em.band_high, color: css("--s1"), opacity: 0.07 }],
        vlines: [{ x: dist.spot, label: "Price" }, { x: q["0.05"], label: "5%" }, { x: q["0.95"], label: "95%" }] }),
      table: (b) => table(b, [{ label: "Chance of ending below", get: (x) => x[0] }, { label: "Price", num: true, get: (x) => x[1] }, { label: "From today's price", num: true, get: (x) => x[2] }],
        Object.entries(q).map(([p, v]) => [pct(Number(p), 0), num(v, 2), W.signed((v / dist.spot - 1) * 100, 1) + "%"])) });
    const tails = dist.tails.slice().sort((a, b) => (a.side === b.side ? a.move - b.move : a.side === "fall" ? -1 : 1));
    chartCard(ch, "Chance of a large move", `The chance that the price ends at least this far below (a fall) or above (a rise) today's on ${dist.expiry}, as the options imply it. Beside each is the ratio to what a lognormal with the same at-the-money volatility says: how much the market adds to that tail.`, {
      draw: (b) => C.hbars(b, { rows: tails.map((t) => ({ label: `${t.side === "fall" ? "−" : "+"}${Math.round(t.move * 100)}%`, value: t.implied, side: t.side, text: `${pct(t.implied, 2)}${t.lognormal > 1e-6 ? " · " + (t.implied / t.lognormal > 100 ? "over 100" : num(t.implied / t.lognormal, 1)) + "× lognormal" : ""}`,
        tip: [["Lognormal", pct(t.lognormal, 3)], ["Price level", num(t.level, 2)]] })), format: (v) => pct(v, 1), name: "Chance", label: "Chance of a large move", color: (v, r) => (r.side === "fall" ? css("--neg") : css("--pos")), valueWidth: 150 }),
      table: (b) => table(b, [{ label: "Move", get: (t) => `${t.side === "fall" ? "fall" : "rise"} of ${Math.round(t.move * 100)}%` }, { label: "Price level", num: true, get: (t) => num(t.level, 2) }, { label: "Options imply", num: true, get: (t) => pct(t.implied, 2) },
        { label: "Lognormal", num: true, get: (t) => pct(t.lognormal, 2) }], tails) });
    const mv = d.expected_moves;
    if (mv.length > 1) chartCard(ch, "The move each expiry prices in", "The at-the-money straddle's price over the price of the underlying (solid), and one standard deviation of the move, the at-the-money volatility times the square root of the time (dashed), by days to expiry. The straddle is worth about 0.8 of a standard deviation when the move is normal.", {
      draw: (b) => C.line(b, { xs: mv.map((r) => r.dte), xWhole: true, xLabel: "Days to expiry", series: [{ name: "At-the-money straddle ÷ price", color: css("--s1"), values: mv.map((r) => r.straddle_pct), dots: true, dotSize: 3 }, { name: "One standard deviation", color: css("--s2"), dash: "5 4", values: mv.map((r) => r.one_sd_pct), dots: true, dotSize: 3 }],
        include: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 2), xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `${mv[i].dte} days (${mv[i].expiry})`, height: 250, label: "Expected move by expiry" }),
      table: (b) => table(b, [{ label: "Expiry", get: (r) => r.expiry }, { label: "Days", num: true, get: (r) => r.dte }, { label: "At the money", num: true, get: (r) => pct(r.atm_iv, 1) }, { label: "Straddle", num: true, get: (r) => num(r.straddle, 2) }, { label: "Straddle ÷ price", num: true, get: (r) => pct(r.straddle_pct, 2) },
        { label: "One standard deviation", num: true, get: (r) => pct(r.one_sd_pct, 2) }], mv) });
    const sm = dist.smile;
    chartCard(ch, "The smile the density comes from", `The out-of-the-money implied volatilities of the ${dist.expiry} expiry (dots) and the SVI curve fitted to them (line), by strike. The distribution above is the second derivative of the option prices that this curve implies: a steeper left side is the heavier left tail.`, {
      draw: (b) => C.line(b, { xs: sm.strike, xLabel: "Strike", series: [{ name: "Fitted curve", color: css("--s1"), values: sm.fit_iv }, { name: "Quotes (out of the money)", color: css("--s2"), values: sm.quote_iv, dots: true, dotSize: 2.5, width: 0.01 }],
        yFormat: volFormat(sm.quote_iv), tipFormat: (v) => pct(v, 2), xFormat: (v) => num(v, 0), height: 250, label: "Fitted volatility smile", vlines: [{ x: dist.spot, label: "Price" }, { x: dist.forward, label: "Forward" }], endLabels: false }),
      table: (b) => table(b, [{ label: "Strike", num: true, get: (n) => num(sm.strike[n], 2) }, { label: "Quote", num: true, get: (n) => pct(sm.quote_iv[n], 2) }, { label: "Fitted", num: true, get: (n) => pct(sm.fit_iv[n], 2) }, { label: "Difference (points)", num: true, get: (n) => num((sm.fit_iv[n] - sm.quote_iv[n]) * 100, 2) }], sm.strike.map((_, n) => n)) });
    const rows = mf.rows;
    if (rows.length > 1) chartCard(ch, "Model-free against at-the-money volatility", "Each expiry's volatility read two ways: from the one at-the-money quote, and model-free from every out-of-the-money option weighted by 1/K² (the VIX rule). The model-free figure is the higher when the smile slopes, because it prices the wings too.", {
      draw: (b) => C.line(b, { xs: rows.map((r) => r.dte), xWhole: true, xLabel: "Days to expiry", series: [{ name: "Model-free", color: css("--s1"), values: rows.map((r) => r.vol), dots: true, dotSize: 3 }, { name: "At the money", color: css("--s2"), values: rows.map((r) => r.atm_iv), dots: true, dotSize: 3 }],
        yFormat: volFormat(rows.flatMap((r) => [r.vol, r.atm_iv])), tipFormat: (v) => pct(v, 2), xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `${rows[i].dte} days (${rows[i].expiry})`, height: 250, label: "Model-free and at-the-money volatility", vlines: [{ x: 30, label: "30 days" }] }),
      table: (b) => table(b, [{ label: "Expiry", get: (r) => r.expiry }, { label: "Days", num: true, get: (r) => r.dte }, { label: "Model-free", num: true, get: (r) => pct(r.vol, 2) }, { label: "At the money", num: true, get: (r) => pct(r.atm_iv, 2) }, { label: "Difference", num: true, get: (r) => W.points(r.difference, 2) },
        { label: "Strikes used", num: true, get: (r) => r.strikes }, { label: "Lowest to highest", num: true, get: (r) => `${num(r.lowest, 0)} to ${num(r.highest, 0)}` }], rows) });
    const cone = d.cone;
    if (cone && cone.rows.length > 1) {
      const cr = cone.rows, col = (key) => cr.map((r) => r[key]);
      chartCard(ch, "Volatility cone: implied against the stock's own history", `Realised volatility over windows of a week to a year across ${grouped(cone.history_days)} days of ${nameOf(d)}'s history: the lightest band is everything it has done, the next the 10th to 90th percentile, the darkest the quartiles; the grey line is the median and the blue line what it is now. The orange line is the at-the-money implied volatility of an expiry as long as each window (Burghardt and Lane 1990): above the cone the options are dear against history, below it cheap.`, { wide: true,
        draw: (b) => C.line(b, { xs: col("calendar_days"), xWhole: true, xLabel: "Calendar days (realised window, or days to expiry)", series: [{ name: "Median realised", color: css("--muted"), width: 1.5, values: col("median") }, { name: "Realised now", color: css("--s1"), values: col("now"), dots: true }, { name: "Implied, at the money", color: css("--s2"), values: col("implied"), dots: true, connect: true }],
          bands: [{ lo: col("min"), hi: col("max"), color: css("--s1"), opacity: 0.06, name: "Range" }, { lo: col("p10"), hi: col("p90"), color: css("--s1"), opacity: 0.1, name: "10th to 90th" }, { lo: col("p25"), hi: col("p75"), color: css("--s1"), opacity: 0.16, name: "Quartiles" }],
          yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `${cr[i].days} trading days (${cr[i].calendar_days} calendar)`, height: 300, label: "Volatility cone" }),
        table: (b) => table(b, [{ label: "Window (trading days)", num: true, get: (r) => r.days }, { label: "Now", num: true, get: (r) => pct(r.now, 1) }, { label: "Percentile", num: true, get: (r) => pct(r.percentile, 0) }, { label: "Lowest", num: true, get: (r) => pct(r.min, 1) }, { label: "10th", num: true, get: (r) => pct(r.p10, 1) },
          { label: "Median", num: true, get: (r) => pct(r.median, 1) }, { label: "90th", num: true, get: (r) => pct(r.p90, 1) }, { label: "Highest", num: true, get: (r) => pct(r.max, 1) }, { label: "Implied (at the money)", num: true, get: (r) => (r.implied === null ? "" : pct(r.implied, 1)) }], cr) });
    } else W.hint(root, "The volatility cone needs the underlying's price history, which this chain does not come with.");
    W.hint(root, "The density is read from a smooth curve (SVI) fitted to this expiry's out-of-the-money volatilities, so it shows the shape the market prices rather than the noise in each quote. It is the risk-neutral distribution: it weights bad states more than their frequency, so a fall looks likelier here than it has been in practice. The model-free volatility needs no curve at all.");
    sources(root, d.references, [["The distribution the quotes imply", "distribution"], ["Model-free volatility, the VIX and the variance premium", "model_free"], ["The volatility cone", "cone"]]);
  }

  /* ---------- open interest and gamma ---------- */
  function buildPositioning(root, d) {
    const p = d.positioning, g = p.gex;
    rememberExpiries(d);
    header(root, `${nameOf(d)}: open interest and gamma`, `${p.scope === "all expiries" ? "all expiries together" : p.scope + " expiry"} · price ${money(p.spot, 2)}`, GUIDES.positioning);
    W.provenance(root, d);
    W.pills(h("div", "filters", root), [["", "All expiries", "Every expiry together"], ...d.expiries.slice(0, 19).map((e) => [e.expiry, `${e.expiry.slice(5)} · ${e.dte}d`, `${e.expiry}: ${e.dte} days`])], d.expiry || "", (v) => { O.form.expiry = v; O.run(true); }, "expiry");
    const k = h("div", "earnrow", root), away = (x) => (x === null || x === undefined ? "" : `${W.signed((x / p.spot - 1) * 100, 1)}% from the price`);
    tile(k, "Put-call ratio, open interest", Q.bad(p.put_call_oi) ? "n/a" : num(p.put_call_oi, 2), { text: `calls ${grouped(p.call_oi)} · puts ${grouped(p.put_oi)}` }, "", "Open interest in puts over open interest in calls (above 1: more puts outstanding)");
    tile(k, "Call wall", p.call_wall === null ? "none" : num(p.call_wall, 2), { text: away(p.call_wall) }, "", "The strike at or above the price with the most call open interest: a level at which a rally meets a lot of short calls (if customers have written them)");
    tile(k, "Put wall", p.put_wall === null ? "none" : num(p.put_wall, 2), { text: away(p.put_wall) }, "", "The strike at or below the price with the most put open interest: a level where a fall meets a lot of puts");
    if (p.max_pain !== null) tile(k, "Max pain", num(p.max_pain, 2), { text: away(p.max_pain) }, "", "The settlement price at which all the open interest would pay its holders the least. A folk rule: prices have tended to cluster at strikes with large open interest near expiry (Ni, Pearson and Poteshman 2005)");
    tile(k, "Gamma exposure", bigMoney(g.net), { text: `per 1% move · ${g.net >= 0 ? "positive: hedging would damp moves" : "negative: hedging would add to moves"}` }, "", "Gamma × open interest × 100 × price² ÷ 100, calls counted positive and puts negative: the change in dollar delta for a 1% move. Open interest has no sign, so this assumes dealers hold the other side of customers who write calls and buy puts");
    tile(k, "Gamma flip", g.flip === null ? "none within ±15%" : num(g.flip, 2), { text: g.flip === null ? "the sign does not change" : away(g.flip) }, "", "The price at which the net gamma exposure crosses zero, with each option's volatility held fixed: above it hedging would damp moves, below it add to them (on the convention)");
    const ch = h("div", "charts", root), near = p.rows.filter((r) => Math.abs(r.strike / p.spot - 1) <= 0.2), shown = near.length >= 5 ? near : p.rows, step = Math.max(1, Math.ceil(shown.length / 14));
    const labels = shown.map((r, i) => (i % step === 0 ? num(r.strike, 0) : "")), mark = shown.reduce((best, r, i) => (Math.abs(r.strike - p.spot) < Math.abs(shown[best].strike - p.spot) ? i : best), 0), nearTable = (cols) => (b) => table(b, cols, shown);
    chartCard(ch, "Open interest by strike", `Contracts outstanding at each strike within 20% of the price, ${p.scope === "all expiries" ? "all expiries together" : "for the " + p.scope + " expiry"}: calls above the line, puts below. The dashed line is the price. Open interest is how many contracts are open, not who is long and who is short.`, { wide: true,
      draw: (b) => C.mirror(b, { labels, up: shown.map((r) => r.call_oi), down: shown.map((r) => r.put_oi), upName: "Calls", downName: "Puts", yFormat: grouped, mark, markLabel: "Price", tipTitle: (i) => `Strike ${num(shown[i].strike, 2)}`, height: 280, label: "Open interest by strike" }),
      table: nearTable([{ label: "Strike", num: true, get: (r) => num(r.strike, 2) }, { label: "Call open interest", num: true, get: (r) => grouped(r.call_oi) }, { label: "Put open interest", num: true, get: (r) => grouped(r.put_oi) }, { label: "Call volume", num: true, get: (r) => grouped(r.call_volume) }, { label: "Put volume", num: true, get: (r) => grouped(r.put_volume) }]) });
    chartCard(ch, "Gamma exposure by strike", "Dollars of delta that change for a 1% move in the underlying, by strike: calls (above the line) are counted as positive and puts (below) as negative, the convention that holds if dealers hold the other side of customers who write calls and buy puts. Large bars are the strikes whose options the price is most sensitive to.", { wide: true,
      draw: (b) => C.mirror(b, { labels, up: shown.map((r) => r.call_gex), down: shown.map((r) => -r.put_gex), upName: "Calls (positive)", downName: "Puts (negative)", yFormat: bigMoney, mark, markLabel: "Price", tipTitle: (i) => `Strike ${num(shown[i].strike, 2)}`, height: 280, label: "Gamma exposure by strike" }),
      table: nearTable([{ label: "Strike", num: true, get: (r) => num(r.strike, 2) }, { label: "Call gamma exposure", num: true, get: (r) => bigMoney(r.call_gex) }, { label: "Put gamma exposure", num: true, get: (r) => bigMoney(r.put_gex) }, { label: "Net", num: true, get: (r) => bigMoney(r.call_gex + r.put_gex) }]) });
    const gc = g.curve;
    chartCard(ch, "Net gamma exposure as the price moves", "The total gamma exposure recomputed with the underlying at each price from 15% below to 15% above, every option's volatility held where it is. Blue is positive (hedging would damp a move), red negative (it would add to one); where the line crosses zero is the flip. It describes the open interest, not what dealers will do.", {
      draw: (b) => C.line(b, { xs: gc.spot, xLabel: "Price of the underlying", series: [{ name: "Net gamma exposure", color: css("--s1"), values: gc.net }], baseline: 0, include: 0, signFill: true, yFormat: bigMoney, tipFormat: bigMoney, xFormat: (v) => num(v, 0), height: 250, label: "Net gamma exposure by price",
        vlines: [{ x: p.spot, label: "Price" }, ...(g.flip === null ? [] : [{ x: g.flip, label: "Flip " + num(g.flip, 1) }])] }),
      table: (b) => table(b, [{ label: "Price", num: true, get: (n) => num(gc.spot[n], 2) }, { label: "From today's", num: true, get: (n) => W.signed((gc.spot[n] / p.spot - 1) * 100, 1) + "%" }, { label: "Net gamma exposure", num: true, get: (n) => bigMoney(gc.net[n]) }], gc.spot.map((_, n) => n).filter((n) => n % 2 === 0)) });
    const top = p.rows.slice().sort((a, b) => b.call_oi + b.put_oi - (a.call_oi + a.put_oi)).slice(0, 10);
    const big = h("div", "card", ch); h("h3", "", big, "The ten strikes with the most open interest");
    table(big, [{ label: "Strike", num: true, get: (r) => num(r.strike, 2) }, { label: "From the price", num: true, get: (r) => W.signed((r.strike / p.spot - 1) * 100, 1) + "%" }, { label: "Calls", num: true, get: (r) => grouped(r.call_oi) }, { label: "Puts", num: true, get: (r) => grouped(r.put_oi) }, { label: "Net gamma exposure", num: true, get: (r) => bigMoney(r.call_gex + r.put_gex) }], top);
    W.hint(root, "Open interest has no sign: the data say how many contracts are outstanding, not who holds them. The gamma exposure counts calls positive and puts negative, the convention behind most 'gamma exposure' charts; the academic work uses signed trades (Ni, Pearson, Poteshman and White 2021; Barbon and Buraschi 2020; Garleanu, Pedersen and Poteshman 2009 show that the demand for options moves their prices). Max pain is a rule of thumb for the settlement of one expiry, not a forecast.");
    sources(root, d.references, [["Option demand, dealer hedging and where prices settle", "positioning"]]);
  }

  /* ---------- greeks and risk ---------- */
  function buildRisk(root, d) {
    const t = d.totals;
    header(root, `Book on ${nameOf(d)}`, `${d.positions.length} position${d.positions.length === 1 ? "" : "s"} · price ${money(d.spot, 2)} · valued today`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Value", smoney(t.value), { text: "at the model's mids" });
    tile(k, "Delta", smoney(t.delta_dollars), { text: `${num(t.delta_shares, 0)} shares equivalent` }, "", "Dollar value of the shares the book is equivalent to");
    tile(k, "Gamma: P&L of a 1% move", smoney(t.gamma_pnl_1pct), { text: "from convexity alone" }, "", "Half the gamma times the squared move: what a 1% move adds or costs beyond delta");
    tile(k, "Vega", smoney(t.vega_per_point), { text: "per volatility point" }, "", "What one point of implied volatility is worth to the book");
    tile(k, "Theta", smoney(t.theta_per_day), { text: "per calendar day" }, "", "What a day passing is worth, all else equal");
    tile(k, "Rho", smoney(t.rho_per_1pct), { text: "per 1% in the rate" });
    const card = h("div", "card", root); h("h3", "", card, "Positions");
    table(card, [{ label: "Position", get: (p) => (p.kind === "stock" ? "Shares" : `${p.right === "C" ? "Call" : "Put"} ${num(p.strike, 2)} ${p.expiry}`) }, { label: "Qty", num: true, get: (p) => num(p.quantity, 0) }, { label: "Days", num: true, get: (p) => (p.dte === null ? "" : p.dte) },
      { label: "IV", num: true, get: (p) => (p.iv === null ? "" : pct(p.iv, 1)) }, { label: "Value", num: true, get: (p) => smoney(p.value) }, { label: "Delta (sh)", num: true, get: (p) => num(p.delta, 1) }, { label: "Gamma (per $)", num: true, get: (p) => num(p.gamma, 3) },
      { label: "Vega ($/pt)", num: true, get: (p) => num(p.vega, 1) }, { label: "Theta ($/day)", num: true, get: (p) => num(p.theta, 1) }], d.positions);
    const ch = h("div", "charts", root);
    const byExpiry = (b) => table(b, [{ label: "Expiry", get: (e) => e.expiry }, { label: "Days", num: true, get: (e) => e.dte }, { label: "Delta ($)", num: true, get: (e) => smoney(e.delta_dollars) }, { label: "Gamma, 1% move ($)", num: true, get: (e) => smoney(e.gamma_pnl_1pct) },
      { label: "Vega ($/pt)", num: true, get: (e) => smoney(e.vega_per_point) }, { label: "Theta ($/day)", num: true, get: (e) => smoney(e.theta_per_day) }], d.by_expiry);
    if (d.by_expiry.length > 1) chartCard(ch, "Exposure by expiry", "Vega per volatility point by expiry: where the book is sensitive to implied volatility.", {
      draw: (b) => C.hbars(b, { rows: d.by_expiry.map((e) => ({ label: `${e.dte}d`, value: e.vega_per_point })), format: (v) => smoney(v), name: "Vega per point", label: "Vega by expiry", color: (v) => (v >= 0 ? css("--pos") : css("--neg")) }), table: byExpiry });
    else if (d.by_expiry.length) { const c = h("div", "card", ch); h("h3", "", c, "Exposure by expiry"); byExpiry(c); }          // one expiry is a row of numbers, not a chart
    const spots = d.spot_shocks, vols = d.vol_shocks, cell = (i, j) => d.stress.find((s) => s.spot_shock === spots[i] && s.vol_shock === vols[j]), worst = d.worst;
    chartCard(ch, "Stress: profit and loss", "Full revaluation. Rows are price moves, columns are moves in implied volatility (points). Blue is a gain, red a loss; the outlined cell is the worst.", {
      draw: (b) => C.matrix(b, { rows: spots.map((s) => W.signed(s * 100, 0) + "%"), cols: vols.map((v) => W.signed(v * 100, 0)), values: spots.map((_, i) => vols.map((__, j) => (cell(i, j) || {}).pnl)), diverging: true, mid: 0, format: (v) => smoney(v), name: "Profit and loss",
        rowTitle: "Price", colTitle: "Implied volatility (points)", mark: (i, j) => cell(i, j) && cell(i, j).spot_shock === worst.spot_shock && cell(i, j).vol_shock === worst.vol_shock, label: "Stress grid", tipRows: (i, j) => [["From the Greeks", smoney(cell(i, j).taylor)], ["Left out by the Greeks", smoney(cell(i, j).residual)]] }),
      table: (b) => table(b, [{ label: "Price", num: true, get: (s) => W.signed(s.spot_shock * 100, 1) + "%" }, { label: "Volatility (pts)", num: true, get: (s) => W.signed(s.vol_shock * 100, 1) }, { label: "Profit and loss", num: true, get: (s) => smoney(s.pnl) },
        { label: "Greeks estimate", num: true, get: (s) => smoney(s.taylor) }, { label: "Left out", num: true, get: (s) => smoney(s.residual) }], d.stress) });
    W.hint(root, "Greeks are slopes at today's price; a large move is revalued with Black-Scholes, which is why the 'left out' column grows with the size of the move. Each option is priced at the volatility the surface gives at its strike, moved by the volatility shock in parallel.");
  }

  /* ---------- strategy builder ---------- */
  const ROLLING = { covered_call: ["covered_call"], collar: ["collar"], protective_put: ["protective_put"], iron_condor: ["iron_condor"], iron_butterfly: ["iron_butterfly"], short_strangle: ["short_strangle"], short_straddle: ["delta_hedged_straddle", { side: -1 }],
    long_straddle: ["delta_hedged_straddle", { side: 1 }], calendar_spread: ["calendar_spread"], cash_secured_put: ["short_put"], risk_reversal: ["risk_reversal"] };
  const lerp = (xs, ys, x) => {                                                              // the value of a piecewise-linear curve at x
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
    return ys[lo] + ((ys[hi] - ys[lo]) * (x - xs[lo])) / (xs[hi] - xs[lo] || 1);
  };
  const zonesOf = (c, breakevens, lo, hi) => {                                              // the stretches of price between break-evens, each a profit or a loss at the horizon
    const cuts = [lo, ...breakevens.filter((x) => x > lo && x < hi), hi], out = [];
    for (let i = 0; i + 1 < cuts.length; i++) out.push({ from: cuts[i], to: cuts[i + 1], profit: lerp(c.spot, c.expiry, (cuts[i] + cuts[i + 1]) / 2) > 0 });
    return out;
  };
  const kmoney = (v) => (Q.bad(v) ? "" : Math.abs(v) >= 1000 ? (v / 1000).toFixed(Math.abs(v) >= 10000 ? 0 : 1) + "k" : v.toFixed(0));
  const GREEKS = { delta: ["Delta", "Delta (shares)", (v) => num(v, 0)], gamma: ["Gamma", "Gamma (shares per $1)", (v) => num(v, 2)], vega: ["Vega", "Vega ($ per volatility point)", (v) => smoney(v)], theta: ["Theta", "Theta ($ per day)", (v) => smoney(v)] };
  function buildStrategy(root, d) {
    const a = d.analysis, c = a.curve, f = O.form, label = (n) => (n ? n.replace(/_/g, " ") : "your own legs"), name = label(d.strategy), others = d.compare || [];
    header(root, `${name[0].toUpperCase()}${name.slice(1)} on ${nameOf(d)}`, `${d.text} · expiry ${d.expiry} · horizon ${a.horizon_days} days`, GUIDES.strategy);
    W.provenance(root, d);
    const k = h("div", "earnrow", root), cash = a.net_debit > 0 ? `pay ${money(a.net_debit)}` : `receive ${money(a.net_credit)}`;
    tile(k, "Most it can make", a.unlimited_profit ? "Unlimited" : smoney(a.max_profit), { text: a.reward_risk === null ? cash : `reward to risk ${num(a.reward_risk, 2)} · ${cash}` });
    tile(k, "Most it can lose", a.unlimited_loss ? "Unlimited" : smoney(a.max_loss), { text: a.unlimited_loss ? "no floor: size it with care" : "if held to the horizon" });
    tile(k, "Break-even" + (a.breakevens.length === 1 ? "" : "s"), a.breakevens.length ? a.breakevens.map((x) => num(x, 2)).join(" · ") : "none", { text: `price ${num(a.spot, 2)}` }, "", "Underlying prices at the horizon where the profit is zero");
    tile(k, "Probability of profit", pct(a.prob_profit, 0), { text: `expected value ${smoney(a.expected_value)}` }, "", `Lognormal price with volatility ${pct(a.view_vol, 1)} and drift ${pct(a.view_drift, 1)}; the smile, jumps and your view are not in it`);
    tile(k, "Margin (indicative)", money(a.margin.estimate), { text: a.margin.cash_secured ? "cash held against the put" : a.margin.defined_risk ? "defined risk" : "naked short options" }, "", a.margin.basis);
    const ch = h("div", "charts", root), keep = c.spot.map((_, n) => n).filter((n) => c.spot[n] > 0), at = (arr) => keep.map((n) => arr[n]), xs = at(c.spot);        // the grid starts at a price of zero to find the worst case; the picture starts above it
    const series = [{ name: "At the horizon", color: css("--s1"), values: at(c.expiry) }, { name: "Today", color: css("--s2"), values: at(c.now), width: 1.5 }, { name: "Half way", color: css("--s3"), values: at(c.half), width: 1.5 },
      ...others.map((o, i) => ({ name: label(o.strategy), color: Q.colorOf(3 + i), dash: "5 4", width: 1.5, values: at(o.curve) }))];
    const own = [...at(c.expiry), ...at(c.now), ...at(c.half)], span = Math.max(...own, 0) - Math.min(...own, 0), ylim = others.length ? [Math.min(...own, 0) - 0.35 * span, Math.max(...own, 0) + 0.35 * span] : undefined;       // beside other structures the axis fits this one and theirs are cut off
    chartCard(ch, "Profit and loss by price of the underlying", `At the horizon (the line, shaded blue where it makes money and red where it loses), today, and half way to it. The dashed lines mark today's price and the break-evens (labelled with their prices).${others.length ? " The dashed coloured lines are the structures you chose to compare, each at its horizon; the vertical range fits this structure, so a line that runs off the picture is a structure with far more to win or lose." : ""}`, { wide: true,
      draw: (b) => C.line(b, { xs, xLabel: "Price of the underlying", series, baseline: 0, include: 0, signFill: true, ylim, yFormat: (v) => smoney(v).replace("+", ""), tipFormat: (v) => smoney(v), xFormat: (v) => num(v, 0), height: 330, label: "Payoff diagram", endLabels: false,
        vlines: [{ x: a.spot, label: "Price" }, ...a.breakevens.map((x) => ({ x, label: num(x, 1) }))] }),
      table: (b) => table(b, [{ label: "Price", num: true, get: (n) => num(c.spot[n], 2) }, { label: "At the horizon", num: true, get: (n) => smoney(c.expiry[n]) }, { label: "Today", num: true, get: (n) => smoney(c.now[n]) }, { label: "Half way", num: true, get: (n) => smoney(c.half[n]) },
        ...others.map((o) => ({ label: label(o.strategy), num: true, get: (n) => smoney(o.curve[n]) }))], keep.filter((n, i) => i % 6 === 0)) });
    if (others.length) {
      const line = (lab, x) => ({ lab, cost: x.net_debit > 0 ? `pay ${money(x.net_debit)}` : `receive ${money(x.net_credit)}`, up: x.unlimited_profit ? "unlimited" : smoney(x.max_profit), down: x.unlimited_loss ? "unlimited" : smoney(x.max_loss),
        be: x.breakevens.map((v) => num(v, 2)).join(" · ") || "none", pop: pct(x.prob_profit, 0), ev: smoney(x.expected_value), margin: money(typeof x.margin === "object" ? x.margin.estimate : x.margin) });
      const cmp = h("div", "card wide", ch); h("h3", "", cmp, "The structures side by side");
      table(cmp, [{ label: "Structure", get: (x) => x.lab }, { label: "Entry", num: true, get: (x) => x.cost }, { label: "Most it can make", num: true, get: (x) => x.up }, { label: "Most it can lose", num: true, get: (x) => x.down }, { label: "Break-evens", num: true, get: (x) => x.be },
        { label: "Probability of profit", num: true, get: (x) => x.pop }, { label: "Expected value", num: true, get: (x) => x.ev }, { label: "Margin", num: true, get: (x) => x.margin }], [line(name + " (this one)", a), ...others.map((o) => line(label(o.strategy), o))]);
      h("p", "note", cmp, "All are priced at the chain's own volatilities with the same expiry, distance, width and number of contracts, and their odds use the same volatility and drift. The expected value is zero at the market's own view except for what the spread, the commission and the smile take or give.");
    }
    const dn = a.density, centers = dn.edges.slice(0, -1).map((e, i) => (e + dn.edges[i + 1]) / 2), dens = dn.mass.map((m, i) => m / (dn.edges[i + 1] - dn.edges[i])), zs = zonesOf(c, a.breakevens, dn.edges[0], dn.edges[dn.edges.length - 1]);
    chartCard(ch, "Where the price may end up", `The chance of each price at the horizon (${a.horizon_days} days) with the volatility and drift behind the odds (${pct(a.view_vol, 1)} and ${pct(a.view_drift, 1)} a year). Blue is where the structure makes money at the horizon and red where it loses; the area under the curve inside the blue is the probability of profit, ${pct(a.prob_profit, 0)}.`, {
      draw: (b) => C.line(b, { xs: centers, xLabel: "Price at the horizon", series: [{ name: "Chance per $1", color: css("--s1"), values: dens }], area: true, areaBase: 0, include: 0, baseline: 0, yFormat: (v) => (v * 100).toFixed(1) + "%", tipFormat: (v) => (v * 100).toFixed(3) + "% per $1",
        xFormat: (v) => num(v, 0), height: 250, label: "Distribution of the price at the horizon", regions: zs.map((z) => ({ from: z.from, to: z.to, name: z.profit ? "profit" : "loss", color: z.profit ? css("--pos") : css("--neg"), opacity: 0.1 })), vlines: [{ x: a.spot, label: "Price" }, ...a.breakevens.map((x) => ({ x, label: num(x, 1) }))] }),
      table: (b) => table(b, [{ label: "Price from", num: true, get: (z) => num(z.from, 2) }, { label: "to", num: true, get: (z) => num(z.to, 2) }, { label: "At the horizon", get: (z) => (z.profit ? "profit" : "loss") }, { label: "Chance", num: true, get: (z) => pct(centers.reduce((t, x, i) => (x >= z.from && x < z.to ? t + dn.mass[i] : t), 0), 1) }], zs) });
    const g = a.grid, order = g.spot.map((_, i) => i).reverse(), nearest = g.spot.reduce((best, x, i) => (Math.abs(x - a.spot) < Math.abs(g.spot[best] - a.spot) ? i : best), 0), dayName = (x, j) => (j === 0 ? "Today" : j === g.days.length - 1 ? "Expiry" : `+${x}d`);
    chartCard(ch, "Profit and loss by price and day", "What the structure would be worth, less what it cost, at each price (rows) and after each number of days (columns), the options repriced with Black-Scholes at today's volatilities. Blue is a gain, red a loss; the outlined row is nearest today's price. Time decay shows as the colours changing from left to right.", { wide: true,
      draw: (b) => C.matrix(b, { rows: order.map((i) => num(g.spot[i], 0)), cols: g.days.map(dayName), values: order.map((i) => g.pnl[i]), diverging: true, mid: 0, format: kmoney, tipFormat: (v) => smoney(v), name: "Profit and loss", cellText: true, rowTitle: "Price", colTitle: "Days from now", cellHeight: 22,
        mark: (r) => order[r] === nearest, label: "Profit and loss by price and day" }),
      table: (b) => table(b, [{ label: "Price", num: true, get: (i) => num(g.spot[i], 2) }, ...g.days.map((x, j) => ({ label: dayName(x, j), num: true, get: (i) => smoney(g.pnl[i][j]) }))], order) });
    let gsel = O.gsel || "delta";
    const gc = a.greeks_curve, gcard = chartCard(ch, "Greeks of the structure by price", "How the structure's sensitivities change with the price of the underlying today, each leg at its own volatility. Delta is the shares it is equivalent to, gamma how fast that changes, vega the dollars one volatility point is worth, theta the dollars a day of time decay costs or earns.", {
      draw: (b) => C.line(b, { xs: gc.spot, xLabel: "Price of the underlying", series: [{ name: GREEKS[gsel][1], color: css("--s1"), values: gc[gsel] }], baseline: 0, include: 0, signFill: true, yFormat: GREEKS[gsel][2], tipFormat: GREEKS[gsel][2], xFormat: (v) => num(v, 0), height: 250, label: GREEKS[gsel][1] + " by price", vlines: [{ x: a.spot, label: "Price" }] }),
      table: (b) => table(b, [{ label: "Price", num: true, get: (n) => num(gc.spot[n], 2) }, ...Object.entries(GREEKS).map(([key, v]) => ({ label: v[1], num: true, get: (n) => v[2](gc[key][n]) }))], gc.spot.map((_, n) => n)) });
    W.seg(gcard._extra, Object.entries(GREEKS).map(([key, v]) => [key, v[0]]), gsel, (v) => { gsel = v; O.gsel = v; gcard._draw(); });
    const gr = a.greeks, cardG = h("div", "card", ch); h("h3", "", cardG, "Greeks of the structure today");
    table(cardG, [{ label: "Measure", get: (x) => x[0] }, { label: "Value", num: true, get: (x) => x[1] }], [["Delta (dollars)", smoney(gr.delta_dollars)], ["Delta (share equivalent)", num(gr.delta_shares, 1)], ["Gamma: P&L of a 1% move", smoney(gr.gamma_pnl_1pct)], ["Vega per point", smoney(gr.vega_per_point)],
      ["Theta per day", smoney(gr.theta_per_day)], ["Rho per 1%", smoney(gr.rho_per_1pct)], ["Margin basis", a.margin.basis], ["Premium received", money(a.margin.premium_received)]]);
    const legs = h("div", "card", ch); h("h3", "", legs, "Legs");
    table(legs, [{ label: "Leg", get: (l) => (l.type === "stock" ? "Shares" : `${l.right === "C" ? "Call" : "Put"} ${num(l.strike, 2)}`) }, { label: "Qty", num: true, get: (l) => num(l.quantity, 0) }, { label: "Days", num: true, get: (l) => (l.dte === undefined ? "" : l.dte) },
      { label: "IV", num: true, get: (l) => (l.iv === undefined ? "" : pct(l.iv, 1)) }, { label: "Price", num: true, get: (l) => (l.price === undefined ? "" : num(l.price, 2)) }], a.legs);
    if ((d.references || []).length) { const src = h("div", "card", root); h("h3", "", src, "Where the structure comes from"); Q.refsLine(src, d.references); }
    const rolling = ROLLING[f.strategy];
    if (rolling) {                                                                           // the structures a rolling strategy exists for
      const go = h("button", "btn", root, "Backtest the rolling version of this idea");
      go.addEventListener("click", () => { O.form.bt = rolling[0]; O.form.btMode = "single"; if (rolling[1]) O.form.btParams[rolling[0]] = { ...rolling[1] }; O.mode = "backtest"; Q.renderLab(O); });
    }
  }

  /* ---------- backtest ---------- */
  const nameOfStrategy = (id) => String(id).replace(/_/g, " ");
  function buildBacktest(root, d) {
    if (d.kind === "options_compare") return buildCompare(root, d);
    const s = d.summary, os = d.option_summary, tm = d.timing, sweep = O.results.backtest_sweep && O.results.backtest_sweep.data;
    header(root, `${nameOfStrategy(d.strategy)}: simulated option market`, `${d.n_days} trading days · ${d.start} to ${d.end} · capital ${money(d.capital)} · seed ${d.seed}${tm ? ` · traded only while implied volatility is ${pct(tm.threshold, 1)} above realised` : ""}`, GUIDES.backtest);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Sharpe ratio", num(s.sharpe), { text: `Sortino ${num(s.sortino)} · Calmar ${num(s.calmar)}` });
    tile(k, "Return a year", Q.pct(s.cagr), { text: `volatility ${pct(s.ann_vol)}` });
    tile(k, "Total return", pct(s.total_return), { text: `on ${money(d.capital)} of capital` });
    tile(k, "Worst fall", pct(s.max_drawdown), { text: `worst day ${pct(s.worst_day)}` });
    tile(k, "Tail risk, 95%", pct(d.cvar_95, 2), { text: `a day: value at risk ${pct(d.var_95, 2)}` }, "", "Expected loss on the worst 5% of days (CVaR) and the loss exceeded on 5% of days (VaR)");
    tile(k, "Trades", grouped(d.trades), { text: `${money(os.costs_total, 0)} in costs` + (d.skipped_orders ? ` · ${d.skipped_orders} skipped` : "") });
    if (tm) tile(k, "Time in the market", pct(d.share_in_market, 0), { text: "days with a position open" }, "", "The variance-premium filter closes the position while implied volatility is not far enough above recent realised volatility");
    const ch = h("div", "charts", root);
    chartCard(ch, "Equity curve", "Growth of 1 in the strategy: the capital plus each day's profit over the capital. Margin and funding are not modelled.", { wide: true,
      draw: (b) => C.line(b, { dates: d.dates, series: [{ name: "Strategy", color: css("--s1"), values: d.equity }], area: true, areaBase: 1, baseline: 1, yFormat: (v) => v.toFixed(3), tipFormat: (v) => v.toFixed(4), height: 260, label: "Equity curve" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, { label: "Strategy", num: true, get: (n) => num(d.equity[n], 4) }, { label: "Underlying", num: true, get: (n) => num(d.benchmark[n], 4) }], d.dates.map((_, n) => n).filter((n) => n % 4 === 0)) });
    if (sweep) sweepCards(ch, sweep);
    chartCard(ch, "The underlying over the same days", "Growth of 1 in the underlying the options were written on. Its own chart, because a strategy that earns a fraction of a percent would be a flat line on the same axis.", {
      draw: (b) => C.line(b, { dates: d.dates, series: [{ name: "Underlying (buy and hold)", color: css("--s2"), values: d.benchmark }], baseline: 1, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 230, label: "Underlying" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, { label: "Underlying", num: true, get: (n) => num(d.benchmark[n], 4) }], d.dates.map((_, n) => n).filter((n) => n % 4 === 0)) });
    const att = Object.entries(d.attribution).map(([label, value]) => ({ label, value }));
    chartCard(ch, "Where the profit came from", "Dollars by source over the backtest. Delta and gamma are the price moves, vega the volatility moves, theta the time decay, hedge the stock hedge; slippage and costs are what trading took.", {
      draw: (b) => C.hbars(b, { rows: att, format: (v) => smoney(v), name: "Dollars", label: "Profit by source", color: (v) => (v >= 0 ? css("--pos") : css("--neg")) }),
      table: (b) => table(b, [{ label: "Source", get: (x) => x.label }, { label: "Dollars", num: true, get: (x) => smoney(x.value) }], att) });
    if (d.net_delta.some((v) => v !== null)) chartCard(ch, "Net delta through time", "Shares equivalent held (the options' delta plus any stock hedge): how far the book leans on the price.", {
      draw: (b) => C.line(b, { dates: d.dates, series: [{ name: "Net delta (shares)", color: css("--s1"), values: d.net_delta }], baseline: 0, include: 0, yFormat: (v) => num(v, 0), tipFormat: (v) => num(v, 1), height: 230, label: "Net delta" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, { label: "Net delta", num: true, get: (n) => (d.net_delta[n] === null ? "" : num(d.net_delta[n], 1)) }], d.dates.map((_, n) => n).filter((n) => n % 4 === 0)) });
    W.riskCharts(ch, d);
    const trades = h("div", "card", ch); h("h3", "", trades, "Latest trades");
    table(trades, [{ label: "Date", get: (x) => x.date }, { label: "Instrument", get: (x) => (x.kind === "option" ? `${x.right === "C" ? "Call" : "Put"} ${num(x.strike, 2)} ${x.expiry}` : x.kind) }, { label: "Qty", num: true, get: (x) => num(x.quantity, 0) }, { label: "Price", num: true, get: (x) => num(x.price, 2) }, { label: "Cost ($)", num: true, get: (x) => num(x.cost, 2) }], d.recent_trades);
    W.riskGrid(root, d, "Underlying");
    W.hint(root, "The market pays the variance premium it was built with and has jumps, so a premium seller does well here by construction: the result checks that costs, hedges and settlement are booked correctly, not that the premium exists in your market.");
    if ((d.references || []).length) { const src = h("div", "card", root); h("h3", "", src, "Where the rule comes from"); Q.refsLine(src, d.references); }
  }
  /* One setting varied: how the result moves, which is how a setting is fine-tuned and how fragile the choice of it is. */
  function sweepCards(ch, sw) {
    const rows = sw.rows, xs = rows.map((r) => r.value), lab = nameOfStrategy(sw.param), fx = (v) => String(Number(v.toPrecision(4))), best = rows.reduce((bi, r, i) => ((r.sharpe ?? -99) > (rows[bi].sharpe ?? -99) ? i : bi), 0);
    chartCard(ch, `Sharpe ratio as ${lab} changes`, `The same backtest at each value of ${lab} (${nameOfStrategy(sw.strategy)}; everything else, the market and the seed fixed); the vertical line is the value used above, ${fx(sw.base)}, and the best of those tried is ${fx(rows[best].value)}. A ratio that peaks at one value and falls away either side is a fit to this market; a plateau means the choice matters little.`, {
      draw: (b) => C.line(b, { xs, xLabel: lab, series: [{ name: "Sharpe ratio", color: css("--s1"), values: rows.map((r) => r.sharpe), dots: true }], baseline: 0, include: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), xFormat: fx, vlines: [{ x: sw.base, label: "used" }], height: 240, label: `Sharpe ratio by ${lab}` }),
      table: (b) => table(b, [{ label: lab, num: true, get: (r) => fx(r.value) }, { label: "Sharpe", num: true, get: (r) => num(r.sharpe, 2) }, { label: "Return a year", num: true, get: (r) => pct(r.cagr, 2) }, { label: "Worst fall", num: true, get: (r) => pct(r.max_drawdown, 2) }], rows) });
    chartCard(ch, `Return and worst fall as ${lab} changes`, "The yearly return and the deepest fall of the same runs: a setting that raises the return by taking a deeper fall is buying return with risk, not finding it.", {
      draw: (b) => C.line(b, { xs, xLabel: lab, series: [{ name: "Return a year", color: css("--s1"), values: rows.map((r) => r.cagr), dots: true }, { name: "Worst fall", color: css("--s2"), values: rows.map((r) => r.max_drawdown), dots: true }], baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(1) + "%", tipFormat: (v) => pct(v, 2), xFormat: fx, vlines: [{ x: sw.base, label: "used" }], height: 240, label: `Return and worst fall by ${lab}` }),
      table: (b) => table(b, [{ label: lab, num: true, get: (r) => fx(r.value) }, { label: "Total return", num: true, get: (r) => pct(r.total_return, 2) }, { label: "Volatility", num: true, get: (r) => pct(r.ann_vol, 2) }, { label: "Tail risk 95%", num: true, get: (r) => pct(r.cvar_95, 2) }, { label: "Trades", num: true, get: (r) => grouped(r.trades) },
        { label: "Costs ($)", num: true, get: (r) => num(r.costs, 0) }, { label: "Time in the market", num: true, get: (r) => pct(r.share_in_market, 0) }], rows) });
  }
  /* Every strategy at its defaults on one market. */
  function buildCompare(root, d) {
    const p = d.params, rows = d.rows.slice().sort((a, b) => (b.sharpe ?? -99) - (a.sharpe ?? -99)), order = d.rows.map((r) => r.strategy);
    header(root, "Every rolling strategy on one simulated market", `${d.rows.length} strategies at their default settings · ${p.n_days} days · seed ${p.seed} · capital ${money(p.capital)}${p.timing ? " · traded only while volatility is rich" : ""}`, GUIDES.backtest);
    W.provenance(root, d);
    const ch = h("div", "charts", root), color = (id) => Q.colorOf(order.indexOf(id));
    chartCard(ch, "Sharpe ratio by strategy", "The annualised return over the volatility of each strategy's daily profit on the same simulated market, after costs. The market pays the variance premium it was built with, so the sellers of premium lead it by construction; what the comparison shows is how much their structures give back in the bad days.", {
      draw: (b) => C.hbars(b, { rows: rows.map((r) => ({ label: nameOfStrategy(r.strategy), value: r.sharpe ?? 0, text: `${num(r.sharpe, 2)} · ${pct(r.cagr, 1)} a year`, tip: [["Worst fall", pct(r.max_drawdown, 2)], ["Tail risk 95%", pct(r.cvar_95, 2)], ["Trades", grouped(r.trades)], ["Costs", money(r.costs, 0)]] })), format: (v) => num(v, 1), name: "Sharpe ratio", label: "Sharpe ratio by strategy", labelWidth: 150, valueWidth: 130 }),
      table: (b) => table(b, [{ label: "Strategy", get: (r) => nameOfStrategy(r.strategy) }, { label: "Sharpe", num: true, get: (r) => num(r.sharpe, 2) }], rows) });
    chartCard(ch, "Return against the worst fall", "Each strategy's yearly return against the deepest fall from a peak it suffered on this market: up and to the left is better. Strategies that hold shares are on the same fixed capital as those that do not.", {
      draw: (b) => C.scatter(b, { points: d.rows.map((r) => ({ x: -r.max_drawdown, y: r.cagr, label: nameOfStrategy(r.strategy), color: color(r.strategy), tip: [["Return a year", pct(r.cagr, 2)], ["Worst fall", pct(r.max_drawdown, 2)], ["Sharpe ratio", num(r.sharpe, 2)]] })),
        xLabel: "Worst fall (as a loss)", yLabel: "Return a year", xFormat: W.pctAxis(d.rows.map((r) => -r.max_drawdown)), yFormat: W.pctAxis(d.rows.map((r) => r.cagr)), yZero: true, labels: "all", height: 300, label: "Return against the worst fall" }),
      table: (b) => table(b, [{ label: "Strategy", get: (r) => nameOfStrategy(r.strategy) }, { label: "Return a year", num: true, get: (r) => pct(r.cagr, 2) }, { label: "Worst fall", num: true, get: (r) => pct(r.max_drawdown, 2) }], rows) });
    const top = rows.slice(0, 5);
    chartCard(ch, "Growth of 1", "The five strategies with the highest Sharpe ratio (the underlying is left out: it moves a hundred times as much and would flatten them; the single-strategy view draws it). The table lists them all.", { wide: true,
      draw: (b) => C.line(b, { dates: d.dates, series: top.map((r) => ({ name: nameOfStrategy(r.strategy), color: color(r.strategy), values: d.equity[r.strategy] })), baseline: 1, yFormat: (v) => v.toFixed(3), tipFormat: (v) => v.toFixed(4), height: 300, label: "Growth of 1", endLabels: false }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, ...top.map((r) => ({ label: nameOfStrategy(r.strategy), num: true, get: (n) => num(d.equity[r.strategy][n], 4) }))], d.dates.map((_, n) => n).filter((n) => n % 4 === 0)) });
    const all = h("div", "card wide", ch); h("h3", "", all, "Every strategy");
    table(all, [{ label: "Strategy", get: (r) => nameOfStrategy(r.strategy) }, { label: "Sharpe", num: true, get: (r) => num(r.sharpe, 2) }, { label: "Return a year", num: true, get: (r) => pct(r.cagr, 2) }, { label: "Volatility", num: true, get: (r) => pct(r.ann_vol, 2) }, { label: "Worst fall", num: true, get: (r) => pct(r.max_drawdown, 2) },
      { label: "Worst day", num: true, get: (r) => pct(r.worst_day, 2) }, { label: "Tail risk 95%", num: true, get: (r) => pct(r.cvar_95, 2) }, { label: "Trades", num: true, get: (r) => grouped(r.trades) }, { label: "Costs ($)", num: true, get: (r) => num(r.costs, 0) },
      { label: "Time in the market", num: true, get: (r) => pct(r.share_in_market, 0) }], rows);
    W.hint(root, "Each strategy is run at its default settings; pick one in the form and choose 'one strategy with one setting varied' to see how sensitive it is to its own. Margin and funding are not modelled, and strategies that hold shares are compared on the same fixed capital as those that do not.");
  }

  /* ---------- the model forms ---------- */
  const modelItems = () => (O.plan ? O.plan.models.map((x) => [x.id, x.label, x.text]) : ALL_MODELS.map((id) => [id, id, ""]));
  function modelFitForm(sec) {
    const f = O.form, after = () => Q.syncLab(O), b = sec("2. Models to fit");
    Q.checkList(b, "Fit these models", modelItems(), f.fitModels, after);
    h("p", "hint", b, "Each is fitted by least squares to the out-of-the-money quotes (puts below the forward, calls above), every strike weighed by its vega so that the wings count. All five take about half a minute.");
    const c = sec("3. Which quotes", false);
    for (const x of [{ key: "fitExpiries", label: "Expiries to use (2 to 8)", kind: "number", step: 1, min: 2, max: 8 }, { key: "fitMinDte", label: "Shortest expiry (days)", kind: "number", step: 1, min: 1, max: 120, help: "Very short expiries are dominated by rounding and single jumps" },
      { key: "fitWindow", label: "Strikes within (% of the forward)", kind: "number", step: 1, min: 5, max: 60 }]) Q.labField(c, x, f, after);
  }
  const productNote = (id) => { const p = O.plan && O.plan.products.find((x) => x.id === id); return p ? p.text : ""; };
  const expiryField = (box, after) => Q.labField(box, { key: "expiry", label: "Expiry", kind: "choice", options: [["", "About three months"], ...O.expiries.map((e) => [e.expiry, `${e.expiry} · ${e.dte} days`])] }, O.form, after);
  function modelPriceForm(sec) {
    const f = O.form, after = () => Q.syncLab(O), b = sec("2. The product");
    Q.labField(b, { key: "pProduct", label: "Product", kind: "choice", options: O.plan ? O.plan.products.map((p) => [p.id, p.label]) : [["vanilla", "European option"]] }, f, () => { if (f.pProduct === "american") f.pRight = "put"; Q.renderLab(O); });
    h("p", "hint", b, productNote(f.pProduct));
    Q.labField(b, { key: "pRight", label: "Call or put", kind: "choice", options: [["call", "Call"], ["put", "Put"]] }, f, after);
    expiryField(b, after);
    Q.labField(b, { key: "pStrike", label: "Strike (% of the price)", kind: "number", step: 0.5, min: 20, max: 300 }, f, after);
    if (f.pProduct === "barrier") {
      Q.labField(b, { key: "pBarrierKind", label: "Barrier", kind: "choice", options: (O.plan ? O.plan.barriers : ["down-out", "up-out", "down-in", "up-in"]).map((k) => [k, k.replace("-", " and ")]) }, f, () => { f.pBarrier = f.pBarrierKind.startsWith("down") ? "90" : "110"; Q.renderLab(O); });
      Q.labField(b, { key: "pBarrier", label: "Barrier level (% of the price)", kind: "number", step: 0.5, min: 10, max: 400, help: "Below 100 for a down barrier, above 100 for an up barrier" }, f, after);
    }
    if (f.pProduct === "asian") Q.labField(b, { key: "pFixings", label: "Averaging dates", kind: "number", step: 1, min: 2, max: 252 }, f, after);
    if (f.pProduct === "barrier" || f.pProduct === "asian") for (const x of [{ key: "pPaths", label: "Simulated paths", kind: "number", step: 5000, min: 5000, max: 200000 }, { key: "pSeed", label: "Random seed", kind: "number", step: 1, min: 0 }]) Q.labField(b, x, f, after);
    const c = sec("3. Models");
    Q.checkList(c, "Price under these models", modelItems(), f.priceModels, after);
    h("p", "hint", c, "Each model is fitted to the chain first (about half a minute for all five, once in a session). Black-Scholes at the at-the-money volatility is always shown beside them.");
    if (!O.expiries.length && O.form.underlying) context();
  }
  function modelHedgeForm(sec) {
    const f = O.form, after = () => Q.syncLab(O), b = sec("2. The option");
    Q.labField(b, { key: "hRight", label: "Call or put", kind: "choice", options: [["call", "Call"], ["put", "Put"]] }, f, after);
    expiryField(b, after);
    Q.labField(b, { key: "hStrike", label: "Strike (% of the price)", kind: "number", step: 0.5, min: 20, max: 300 }, f, after);
    const c = sec("3. Costs and hedging");
    for (const x of [{ key: "hVol", label: "Volatility (% a year, blank = the at-the-money volatility)", kind: "number", step: 0.5, min: 1, max: 300 },
      { key: "hCost", label: "Cost of one trade (basis points of the value traded)", kind: "number", step: 0.5, min: 0, max: 500, help: "Half the relative bid-ask spread plus fees: a few basis points for a liquid ETF, tens for a thin stock" },
      { key: "hSteps", label: "Rebalancing dates (blank = every trading day)", kind: "number", step: 1, min: 2, max: 504 },
      { key: "hRisk", label: "Risk aversion (in units of the option's price)", kind: "number", step: 0.1, min: 0.01, max: 100, help: "A loss of the option's whole price is worth this many units of utility: 1 is moderate, 0.1 very tolerant, 10 timid" },
      { key: "hPaths", label: "Simulated paths", kind: "number", step: 1000, min: 1000, max: 40000 }, { key: "hSeed", label: "Random seed", kind: "number", step: 1, min: 0 }]) Q.labField(c, x, f, after);
    h("p", "hint", c, "The simulation sells the option at its Black-Scholes price and hedges it on paths of that model with the volatility above: nothing else is simulated.");
    if (!O.expiries.length && O.form.underlying) context();
  }

  /* ---------- model fit ---------- */
  const GREEK = { sigma: "σ", theta: "θ", nu: "ν", kappa: "κ", xi: "ξ", rho: "ρ", lam: "λ", mu_j: "μ_J", sigma_j: "σ_J", v0: "v₀", p: "p", eta1: "η₁", eta2: "η₂" };
  const pnum = (v) => (Q.bad(v) ? "n/a" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3));
  const paramText = (m) => Object.entries(m.params).map(([k, v]) => {
    const se = m.std_errors[k], edge = m.at_bounds.includes(k), loose = !Q.bad(se) && Math.abs(se) > 5 * Math.abs(v);                 // a parameter on the edge of its range, or known to within many times its size, has no usable standard error
    return `${GREEK[k] || k} ${pnum(v)}${edge ? " (at the edge of its range: not a measurement)" : loose ? " (not pinned down by the quotes)" : Q.bad(se) ? "" : ` ±${pnum(se)}`}`;
  }).join(" · ");
  function buildFit(root, d) {
    const best = d.models[0], dens = d.density, tails = dens.tails, bs = tails.find((t) => t.id === "bs"), bt = tails.find((t) => t.id === d.best), dte = Math.round(dens.dte), days = d.quotes.expiries.map((e) => e.dte);
    header(root, `${nameOf(d)}: option models fitted to the chain`, `${d.quotes.n} out-of-the-money quotes · ${d.quotes.expiries.length} expiries from ${Math.round(Math.min(...days))} to ${Math.round(Math.max(...days))} days · price ${money(d.spot, 2)}`, GUIDES.fit);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Best model", best.label, { text: `${num(best.rmse * 100, 2)} pts error · ${pct(best.explained, 0)} of the smile explained` }, "", "Root-mean-square gap between the model's implied volatilities and the quotes, in volatility points. 'Explained' is one minus the squared ratio of that error to the flat-volatility error.");
    tile(k, "Flat volatility", `${num(d.flat_rmse * 100, 2)} pts`, { text: "the error to beat" }, "", "The error of the best constant volatility for each expiry: what a model must beat to claim that it captured the smile.");
    tile(k, `A 10% fall in ${dte} days`, bt ? pct(bt.p_down_10, 1) : "n/a", { text: `${best.label}; Black-Scholes ${pct(bs.p_down_10, 1)}` }, "", "The risk-neutral probability that the price ends at least a tenth lower at the expiry nearest 30 days, from the density the best model implies. It includes the market's price of crash risk: it is not a forecast.");
    tile(k, "Skew of the implied returns", bt ? num(bt.skewness, 2) : "n/a", { text: bt ? `excess kurtosis ${num(bt.excess_kurtosis, 2)}` : "" }, "", "Skewness and excess kurtosis of the log return at that expiry, under the best model. A lognormal (Black-Scholes) has none.");
    tile(k, "Time to fit", `${num(d.models.reduce((a, m) => a + m.seconds, 0), 0)} s`, { text: `${d.models.length} model${d.models.length === 1 ? "" : "s"}, ${d.models.reduce((a, m) => a + m.n_evals, 0)} evaluations` });
    const ch = h("div", "charts", root);
    const err = [...d.models.map((m) => ({ label: m.label, value: m.rmse * 100, color: modelColor(m.id), tip: [["Largest error", num(m.max_error * 100, 2) + " pts"], ["Parameters", String(m.n_params)]] })), { label: "Flat volatility", value: d.flat_rmse * 100, color: css("--muted"), tip: [["Note", "the best flat volatility for each expiry"]] }];
    chartCard(ch, "How far each model is from the quotes", "Root-mean-square error in implied-volatility points (1 point is a volatility of 20% against 21%). The grey bar is the best flat volatility for each expiry: a model that does not beat it has learned nothing about the smile.", { wide: true,
      draw: (b) => C.hbars(b, { rows: err, format: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(2) + " pts", name: "Error", label: "Fit error by model", labelWidth: 200, color: (v, row) => row.color }),
      table: (b) => table(b, [{ label: "Model", get: (x) => x.label }, { label: "Error (pts)", num: true, get: (x) => num(x.value, 2) }], err) });
    const nearest = d.smiles.reduce((best, x, i) => (Math.abs(x.dte - 30) < Math.abs(d.smiles[best].dte - 30) ? i : best), 0);       // the first view is the expiry nearest 30 days, where the density below is drawn
    const si = O.fitSmile === null || O.fitSmile >= d.smiles.length ? nearest : O.fitSmile, sm = d.smiles[si];
    const pickRow = h("div", "filters", root);
    W.pills(pickRow, d.smiles.map((x, i) => [String(i), `${Math.round(x.dte)} days`, `${x.expiry}: ${Math.round(x.dte)} days`]), String(si), (v) => { O.fitSmile = Number(v); O.draw(); }, "Expiry of the smile");
    const grid = h("div", "charts", root), ivs = [...sm.market, ...Object.values(sm.models).flat()];
    chartCard(grid, `Smile at ${Math.round(sm.dte)} days: the quotes and each model`, "Implied volatility by strike. The dots are the market's out-of-the-money quotes; a line that follows them reproduces the smile, one that goes straight does not.", { wide: true,
      draw: (b) => C.line(b, { xs: sm.strike, xLabel: "Strike", series: [{ name: "Quotes", color: modelColor("market"), values: sm.market, dots: true, dotSize: 3, width: 1 }, ...d.models.map((m) => ({ name: m.label, color: modelColor(m.id), values: sm.models[m.id], width: 2 }))],
        yFormat: volFormat(ivs), tipFormat: (v) => pct(v, 1), xFormat: (v) => num(v, 0), tipTitle: (i) => `Strike ${num(sm.strike[i], 2)} (${((Math.exp(sm.log_moneyness[i]) - 1) * 100).toFixed(1)}% from the forward)`, height: 290, label: "Fitted smiles", vlines: [{ x: sm.forward, label: "Forward" }], endLabels: false }),
      table: (b) => table(b, [{ label: "Strike", num: true, get: (n) => num(sm.strike[n], 2) }, { label: "Quotes", num: true, get: (n) => pct(sm.market[n], 2) }, ...d.models.map((m) => ({ label: m.label, num: true, get: (n) => (sm.models[m.id][n] === null ? "" : pct(sm.models[m.id][n], 2)) }))], sm.strike.map((_, n) => n)) });
    const curves = [{ name: "Black-Scholes", color: modelColor("bs"), values: dens.bs, dash: "5 4", width: 1.5 }, ...d.models.filter((m) => dens.curves[m.id]).map((m) => ({ name: m.label, color: modelColor(m.id), values: dens.curves[m.id], width: 2 }))];
    chartCard(grid, `What the quotes imply for the price in ${dte} days`, "The risk-neutral density of the log return: the chance, in percent, of ending in each one-percent slice. Compared with the dashed lognormal of Black-Scholes, a skewed market puts more weight on a big fall and less on a modest rise.", {
      draw: (b) => C.line(b, { xs: dens.x, xLabel: "Return to expiry (log, %)", series: curves, yFormat: (v) => pct(v, 0), tipFormat: (v) => pct(v, 2), xFormat: (v) => v.toFixed(0) + "%", height: 260, label: "Implied density", endLabels: false, vlines: [{ x: 0, label: "Today" }] }),
      table: (b) => table(b, [{ label: "Return (%)", num: true, get: (n) => num(dens.x[n], 1) }, ...curves.map((c) => ({ label: c.name, num: true, get: (n) => pct(c.values[n], 3) }))], dens.x.map((_, n) => n).filter((n) => n % 8 === 0)) });
    const tiny = (arr) => arr.map((v) => (v < 1e-5 ? null : v));                                                          // a probability under 0.001% has no place on the log axis
    const drops = [{ name: "Black-Scholes", color: modelColor("bs"), values: tiny(dens.bs_drop), dash: "5 4", width: 1.5 }, ...d.models.filter((m) => dens.drop_curves[m.id]).map((m) => ({ name: m.label, color: modelColor(m.id), values: tiny(dens.drop_curves[m.id]), width: 2 }))];
    chartCard(grid, "The chance of a fall at least this big", `Risk-neutral probability that the price is down by at least this much in ${dte} days, on a log scale so that the tail can be seen. The gap to the grey line is what the market charges for crash protection beyond a lognormal's.`, {
      draw: (b) => C.line(b, { xs: dens.drop_pct, xLabel: "Fall (% of the price)", xWhole: true, log: true, series: drops, yFormat: (v) => pct(v, v < 0.001 ? 3 : v < 0.01 ? 2 : v < 0.1 ? 1 : 0), tipFormat: (v) => pct(v, 3), xFormat: (v) => v.toFixed(0) + "%", height: 260, label: "Chance of a fall", endLabels: false }),
      table: (b) => table(b, [{ label: "Fall", num: true, get: (n) => `${dens.drop_pct[n]}%` }, ...drops.map((c) => ({ label: c.name, num: true, get: (n) => (c.values[n] === null ? "under 0.001%" : pct(c.values[n], 3)) }))], dens.drop_pct.map((_, n) => n).filter((n) => [1, 2, 4, 5, 9, 14, 19, 29].includes(n + 1))) });
    const fits = h("div", "card wide", grid); h("h3", "", fits, "The fitted parameters");
    table(fits, [{ label: "Model", get: (m) => m.label }, { label: "Error (pts)", num: true, get: (m) => num(m.rmse * 100, 2) }, { label: "Largest (pts)", num: true, get: (m) => num(m.max_error * 100, 2) }, { label: "Explained", num: true, get: (m) => pct(m.explained, 0) },
      { label: "Conditioning", num: true, get: (m) => (Q.bad(m.condition) ? "n/a" : m.condition.toExponential(0)) }, { label: "Parameters (± one standard error)", get: paramText }], d.models);
    W.hint(fits, "Standard errors are the least-squares ones (they say how well the quotes pin each parameter down, not whether the model is right). A parameter at the edge of its range is not a measurement; a large conditioning number means some combination of parameters is not determined by the quotes.");
    const tl = h("div", "card wide", grid); h("h3", "", tl, `What each model implies at ${dte} days`);
    table(tl, [{ label: "Model", get: (t) => t.label }, { label: "Fall of 10%", num: true, get: (t) => pct(t.p_down_10, 1) }, { label: "Fall of 20%", num: true, get: (t) => pct(t.p_down_20, 2) }, { label: "Rise of 10%", num: true, get: (t) => pct(t.p_up_10, 1) },
      { label: "Skewness", num: true, get: (t) => num(t.skewness, 2) }, { label: "Excess kurtosis", num: true, get: (t) => num(t.excess_kurtosis, 2) }], tails);
    W.hint(root, "A calibration describes the risk-neutral measure: what the quotes imply, with the market's price of risk built in. It is the right input for pricing other products on the same underlying and the wrong one for forecasting.");
  }

  /* ---------- model prices ---------- */
  const PRODUCT_NAME = { vanilla: "European option", digital: "Digital option", barrier: "Barrier option", asian: "Average-price option", american: "American option" };
  function buildPrice(root, d) {
    const sp = d.spec, sum = d.summary, names = { ...Object.fromEntries(d.models.map((m) => [m.id, m.label])) };
    const bs = d.benchmarks.find((b) => b.id === "bs"), market = d.benchmarks.find((b) => b.id === "market"), best = sum && d.models.find((m) => m.id === sum.best);
    const barrier = sp.product === "barrier" ? ` · ${sp.kind.replace("-", " and ")} at ${num(sp.H, 2)} (${pct(sp.H / sp.spot, 0)} of the price)` : sp.product === "asian" ? ` · averaged on ${sp.fixings} dates` : "";
    header(root, `${PRODUCT_NAME[sp.product]} on ${nameOf(d)}`, `${sp.call ? "call" : "put"} · strike ${num(sp.K, 2)} (${pct(sp.K / sp.spot, 0)} of the price) · expiry ${sp.expiry} (${Math.round(sp.dte)} days)${barrier}`, GUIDES.price);
    W.provenance(root, d);
    const fmt = (v) => (Q.bad(v) ? "n/a" : sp.product === "digital" ? num(v, 4) : money(v, 3));
    const k = h("div", "earnrow", root);
    if (best) tile(k, "Best-fitting model", fmt(sum.best_price), { text: `${best.label} · fit error ${num(best.fit_error * 100, 2)} pts` }, "", "The price under the model that reproduces the chain's quotes most closely among those that can price this product.");
    tile(k, "Black-Scholes", fmt(sum && sum.bs_price), { text: `at the at-the-money volatility ${pct(d.sigma_atm, 1)}` }, "", "One volatility for every strike: the price when the smile is ignored.");
    if (market) tile(k, "The market's smile", fmt(market.price), { text: "Black-Scholes at this strike's own volatility" }, "", "A vanilla is worth what the quoted smile says. The models were fitted to reproduce it.");
    if (sum) tile(k, "Range across models", `${fmt(sum.low)} to ${fmt(sum.high)}`, { text: sum.range_share === null ? "" : `${pct(sum.range_share, 1)} of the Black-Scholes price${sum.vanilla_range_share === null || sum.vanilla_range_share === undefined ? "" : `; the same models differ by ${pct(sum.vanilla_range_share, 1)} on the vanilla at this strike`}` }, "", "Lowest and highest price among the fitted models. On a vanilla the spread is only how well each model fits the quotes; on another product it is that plus what the quotes do not determine (model risk). Compare it with the same models' spread on the vanilla at this strike.");
    if (sp.product === "american" && best && best.premium !== undefined) tile(k, "Early-exercise premium", fmt(best.premium), { text: `${best.label}: American ${fmt(best.price)} against European ${fmt(best.european)}` }, "", "What the right to exercise early adds to the European price under the same model.");
    const ch = h("div", "charts", root), bsPrice = bs && bs.price, rel = Number.isFinite(bsPrice) && Math.abs(bsPrice) > 1e-9;
    const dev = (r) => (rel ? ((r.price - bsPrice) / Math.abs(bsPrice)) * 100 : r.price - bsPrice), devText = (v) => `${W.signed(v, rel ? 1 : 3)}${rel ? "%" : ""}`;
    const rows = [...d.benchmarks, ...d.models].filter((r) => r.price !== null && r.id !== "bs").map((r) => ({ label: r.id === "bs_continuous" ? "BS, continuous barrier" : r.id === "market" ? "Market smile" : r.label, value: dev(r), color: modelColor(r.id), id: r.id,
      text: `${fmt(r.price)}${r.se ? ` ±${num(r.se, 3)}` : ""} · ${devText(dev(r))}`, tip: [["Price", fmt(r.price)]] }));
    chartCard(ch, "How far each price is from Black-Scholes", `Each bar is a price against Black-Scholes at the at-the-money volatility (${fmt(bsPrice)}): to the right it is dearer, to the left cheaper (${rel ? "per cent" : "in the product's own units"}). Models fitted to the same quotes differ on a vanilla by how well each fits (the fit error is in the table below) and on another product also by what the quotes do not determine: compare the spread with the vanilla's.`, { wide: !(d.curve || d.boundary),
      draw: (b) => C.hbars(b, { rows, format: (v) => (rel ? `${W.signed(v, 0)}%` : W.signed(v, 2)), tipFormat: devText, name: "Against Black-Scholes", label: "Price against Black-Scholes by model", labelWidth: 200, valueWidth: 190, color: (v, row) => row.color }),
      table: (b) => table(b, [{ label: "Model", get: (x) => x.label }, { label: "Price", num: true, get: (x) => x.text }], rows) });
    if (d.curve) {
      const c = d.curve, base = c.curves.bs, series = Object.keys(c.curves).filter((m) => m !== "bs").map((m) => ({ name: names[m] || m, color: modelColor(m), values: c.curves[m].map((v, i) => v - base[i]), width: 2 }));
      chartCard(ch, "How each model differs from Black-Scholes by strike", `Price under the model less the Black-Scholes price at the at-the-money volatility, for ${sp.product === "digital" ? "a digital" : "a vanilla"} ${sp.call ? "call" : "put"} at strikes from 80% to 120% of the price. The gap away from the money is the smile's effect.`, {
        draw: (b) => C.line(b, { xs: c.strike_pct, xLabel: "Strike (% of the price)", series, baseline: 0, include: 0, yFormat: (v) => (sp.product === "digital" ? v.toFixed(2) : v.toFixed(1)), tipFormat: (v) => (v >= 0 ? "+" : "") + (sp.product === "digital" ? v.toFixed(4) : v.toFixed(3)), xFormat: (v) => v.toFixed(0) + "%",
          height: 260, label: "Difference from Black-Scholes by strike", endLabels: false, vlines: [{ x: 100, label: "Price" }] }),
        table: (b) => table(b, [{ label: "Strike", num: true, get: (n) => `${c.strike_pct[n].toFixed(0)}%` }, { label: "Black-Scholes", num: true, get: (n) => fmt(base[n]) }, ...Object.keys(c.curves).filter((m) => m !== "bs").map((m) => ({ label: names[m] || m, num: true, get: (n) => fmt(c.curves[m][n]) }))], c.strike_pct.map((_, n) => n).filter((n) => n % 2 === 0)) });
    }
    if (d.boundary) {
      const lines = Object.entries(d.boundary), top = Math.max(...lines.flatMap(([, v]) => v.price));
      chartCard(ch, "Where early exercise starts paying", `The price of the underlying at which exercising stops being optimal, by days left. Exercise below the line (a put). The line sits under the strike (${num(sp.K, 2)}) and falls as more time remains.`, {
        draw: (b) => C.line(b, { xs: lines[0][1].dte, xLabel: "Days to expiry", xWhole: true, series: lines.map(([id, v]) => ({ name: id === "bs" ? "Black-Scholes" : v.label, color: modelColor(id), values: v.price, width: 2 })), include: sp.call ? top : sp.K, yFormat: (v) => num(v, 0), tipFormat: (v) => num(v, 2), xFormat: (v) => String(Math.round(v)),
          height: 260, label: "Early-exercise boundary", endLabels: false }),
        table: (b) => table(b, [{ label: "Days left", num: true, get: (n) => num(lines[0][1].dte[n], 0) }, ...lines.map(([id, v]) => ({ label: id === "bs" ? "Black-Scholes" : v.label, num: true, get: (n) => num(v.price[n], 2) }))], lines[0][1].dte.map((_, n) => n).filter((n) => n % 4 === 0)) });
    }
    const card = h("div", "card", root); h("h3", "", card, "Every price");
    const diff = (r) => (r.price === null || !bs || !bs.price ? "" : `${W.signed(((r.price - bs.price) / Math.abs(bs.price)) * 100, 1)}%`);
    const cols = [{ label: "Source", get: (r) => r.label }, { label: "Price", num: true, get: (r) => (r.price === null ? "not priced" : fmt(r.price)) }, { label: "± one s.e.", num: true, get: (r) => (r.se ? num(r.se, 3) : "") }, { label: "Against Black-Scholes", num: true, get: diff }];
    if (sp.product === "american") cols.push({ label: "European", num: true, get: (r) => (r.european === undefined ? "" : fmt(r.european)) }, { label: "Early-exercise premium", num: true, get: (r) => (r.premium === undefined ? "" : fmt(r.premium)) });
    cols.push({ label: "Fit error (pts)", num: true, get: (r) => (r.fit_error === null || r.fit_error === undefined ? "" : num(r.fit_error * 100, 2)) });
    const hows = new Set([...d.benchmarks, ...d.models].map((r) => r.note || r.method).filter(Boolean));
    if (hows.size > 1) cols.push({ label: "How", get: (r) => r.note || r.method });                          // the column appears only when the rows differ in method
    table(card, cols, [...d.benchmarks, ...d.models]);
    if (hows.size === 1) W.hint(card, [...hows][0] + ".");
    W.hint(card, "A standard error is the Monte Carlo noise of that price (one sigma); the models are priced on the same random numbers, so their differences are less noisy than their levels. A model with a large fit error reproduces the vanillas poorly: weigh its exotic price less.");
  }

  /* ---------- hedging cost ---------- */
  function buildHedge(root, d) {
    const inp = d.inputs, last = d.leland.find((x) => x.steps === inp.steps) || d.leland[d.leland.length - 1], dpz = d.dpz, band = d.band, sims = Object.fromEntries(d.simulation.map((x) => [x.strategy, x]));
    header(root, `Hedging a ${inp.call ? "call" : "put"} on ${nameOf(d)}`, `strike ${num(inp.strike, 2)} (${pct(inp.strike / inp.spot, 0)} of the price) · expiry ${inp.expiry} (${Math.round(inp.dte)} days) · volatility ${pct(inp.vol, 1)} · ${num(inp.cost_bp, 1)} bp a trade · ${inp.steps} rebalancing dates`, GUIDES.hedge);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Black-Scholes price", money(d.premium, 3), { text: "with free trading" }, "", "The cost of replicating the option with a hedge that is rebalanced continuously and costs nothing.");
    tile(k, "Seller's price with costs", money(last.seller_price, 3), { text: `Leland: rebalancing ${inp.steps} times, ${W.signed(((last.seller_price / d.premium) - 1) * 100, 1)}%` }, "", "Black-Scholes at Leland's higher volatility: what the seller of the option must charge to pay for hedging at this frequency and cost.");
    tile(k, "Cost in volatility", W.points(last.seller_vol - inp.vol), { text: `Leland number ${num(last.le, 2)}` }, "", "The extra volatility Leland's price adds. Above a Leland number of one a buyer cannot replicate the option at a profit.");
    tile(k, "Utility-based ask and bid", `${money(dpz.writer, 3)} / ${money(dpz.buyer, 3)}`, { text: `frictionless ${money(dpz.crr, 3)} · a binomial market` }, "", "The prices at which an investor with exponential utility is indifferent to writing (ask) or buying (bid) the option, hedging optimally with costs, in a 40-step binomial tree.");
    tile(k, "No-trade band", `± ${num(band.half_width_at_spot, 3)} shares`, { text: "around the Black-Scholes delta, at today's price" }, "", "Whalley and Wilmott: trade only when the holding leaves this band, and then only to its edge.");
    const ch = h("div", "charts", root), steps = d.leland.map((x) => x.steps), yours = steps.indexOf(inp.steps);
    chartCard(ch, "What rebalancing more often costs", "Leland's price of the option against the number of rebalancing dates: the seller's price rises and the buyer's falls as trades multiply, because every trade pays the spread. The grey line is Black-Scholes with free trading; the buyer's line stops where the Leland number reaches one.", {
      draw: (b) => C.line(b, { xs: steps.map((_, i) => i), xLabel: "Rebalancing dates", series: [{ name: "Seller", color: css("--s2"), values: d.leland.map((x) => x.seller_price), dots: true }, { name: "Black-Scholes", color: css("--muted"), values: d.leland.map(() => d.premium), dash: "5 4", width: 1.5 }, { name: "Buyer", color: css("--s1"), values: d.leland.map((x) => x.buyer_price), dots: true }],
        yFormat: (v) => num(v, 2), tipFormat: (v) => money(v, 3), xFormat: (v) => (steps[Math.round(v)] === undefined ? "" : String(steps[Math.round(v)])), tipTitle: (i) => `${steps[i]} rebalancing dates`, xWhole: true, height: 260, label: "Leland price by rebalancing frequency", endLabels: false, vlines: yours >= 0 ? [{ x: yours, label: "Yours" }] : [] }),
      table: (b) => table(b, [{ label: "Dates", num: true, get: (x) => x.steps }, { label: "Leland number", num: true, get: (x) => num(x.le, 3) }, { label: "Seller", num: true, get: (x) => money(x.seller_price, 3) }, { label: "Buyer", num: true, get: (x) => (x.buyer_price === null ? "cannot replicate" : money(x.buyer_price, 3)) }], d.leland) });
    chartCard(ch, "The no-trade band around the delta", "The holding to keep, against the price of the underlying: the Black-Scholes delta (centre) and the edges of the Whalley-Wilmott band. Between the edges, do nothing; at an edge, trade back to it.", {
      draw: (b) => C.line(b, { xs: band.spot, xLabel: "Price of the underlying", series: [{ name: "Upper edge", color: css("--s2"), values: band.high, width: 1.5 }, { name: "Delta", color: css("--s1"), values: band.delta, width: 2 }, { name: "Lower edge", color: css("--s2"), values: band.low, width: 1.5, dash: "5 4" }],
        yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3) + " shares", xFormat: (v) => num(v, 0), height: 260, label: "No-trade band", endLabels: false, vlines: [{ x: inp.spot, label: "Price" }] }),
      table: (b) => table(b, [{ label: "Price", num: true, get: (n) => num(band.spot[n], 2) }, { label: "Lower edge", num: true, get: (n) => num(band.low[n], 3) }, { label: "Delta", num: true, get: (n) => num(band.delta[n], 3) }, { label: "Upper edge", num: true, get: (n) => num(band.high[n], 3) }], band.spot.map((_, n) => n).filter((n) => n % 6 === 0)) });
    const hist = d.histogram, mids = hist.edges.slice(0, -1).map((e, i) => (e + hist.edges[i + 1]) / 2), labels = { delta: "Delta every date", leland: "Leland (sold dearer)", band: "Whalley-Wilmott band" }, colors = { delta: css("--s2"), leland: css("--s3"), band: css("--s1") };
    const left = Math.max(...Object.values(hist.outside || {}), 0);
    chartCard(ch, "Profit and loss of each hedging rule", `The result of selling the option and hedging it, on ${inp.paths.toLocaleString("en-US")} simulated paths, after every cost. A narrower hump nearer zero is a better hedge; the Leland rule is sold at a higher price, which moves its whole hump to the right.${left > 0.0005 ? ` The most extreme outcomes (at most ${pct(left, 1)} of the paths of a rule) are left off the picture.` : ""}`, { wide: true,
      draw: (b) => C.histogram(b, { series: ["delta", "leland", "band"].map((id) => ({ name: labels[id], color: colors[id], edges: hist.edges, counts: hist.share[id] })), xFormat: (v) => W.signed(v, Number.isInteger(v) ? 0 : 1), markers: [{ x: 0, label: "Break-even" }], height: 240, label: "Profit and loss of the hedging rules" }),
      table: (b) => table(b, [{ label: "P&L ($, bin centre)", num: true, get: (n) => W.signed(mids[n], 2) }, ...["delta", "leland", "band"].map((id) => ({ label: labels[id], num: true, get: (n) => pct(hist.share[id][n], 1) }))], mids.map((_, n) => n).filter((n) => n % 3 === 0)) });
    const card = h("div", "card", root); h("h3", "", card, "The three rules on simulated paths");
    table(card, [{ label: "Rule", get: (x) => labels[x.strategy] }, { label: "Premium received", num: true, get: (x) => money(x.premium, 3) }, { label: "Mean P&L ($)", num: true, get: (x) => W.signed(x.mean, 2) }, { label: "± s.e.", num: true, get: (x) => num(x.se, 3) }, { label: "Spread (std, $)", num: true, get: (x) => num(x.std, 2) },
      { label: "5th percentile ($)", num: true, get: (x) => W.signed(x.q05, 2) }, { label: "95th percentile ($)", num: true, get: (x) => W.signed(x.q95, 2) }, { label: "Cost paid", num: true, get: (x) => money(x.cost, 3) }, { label: "Trades", num: true, get: (x) => num(x.trades, 1) }], d.simulation);
    W.hint(card, `Selling at the Black-Scholes price and hedging at ${inp.steps} dates ${sims.delta.mean < 0 ? "loses" : "makes"} ${money(Math.abs(sims.delta.mean), 2)} on average, with ${money(sims.delta.cost, 2)} paid in costs. The band trades about ${num((sims.band.trades / Math.max(sims.delta.trades, 1)) * 100, 0)}% as often and pays ${num((sims.band.cost / Math.max(sims.delta.cost, 1e-9)) * 100, 0)}% of the cost, ${sims.band.std > sims.delta.std ? `at the price of a wider spread of outcomes (${num(sims.band.std, 2)} against ${num(sims.delta.std, 2)}): what is saved in costs is paid in risk` : `and here its spread of outcomes is no wider (${num(sims.band.std, 2)} against ${num(sims.delta.std, 2)})`}.`);
    const dc = h("div", "card", root); h("h3", "", dc, "Utility-based prices in a binomial market");
    table(dc, [{ label: "Measure", get: (x) => x[0] }, { label: "Value", num: true, get: (x) => x[1] }], [["Writer's (ask) price", money(dpz.writer, 3)], ["Buyer's (bid) price", money(dpz.buyer, 3)], ["Frictionless (Cox-Ross-Rubinstein) price", money(dpz.crr, 3)], ["Bid-ask spread of the option", money(dpz.spread, 3)],
      ["Writer's no-trade band at the start (shares)", `${num(dpz.band[0], 3)} to ${num(dpz.band[1], 3)}`], ["Frictionless delta", num(dpz.crr_delta, 3)]]);
    W.hint(dc, "Hodges-Neuberger / Davis-Panas-Zariphopoulou: between the buyer's and the writer's price the option cannot be hedged at a profit. The band shows the holdings at which the writer's best move is not to trade.");
  }

  Q.registerWorkspace("options", O);
})();
