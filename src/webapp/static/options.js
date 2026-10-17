/* The Options workspace: what a chain says (overview, chain, smile and skew, surface and term structure), which pricing models reproduce it and what they imply for exotic and American options and for the cost of
   hedging (model fit, model prices, hedging cost), what a book of options is exposed to (Greeks and stress), what a structure pays (payoff, break-evens, probability of profit) and what a rolling strategy would
   have paid (a backtest on a simulated option market). It talks to /api/options/* and /api/models/*; every payload says whether its numbers are a model, a file of yours or platform data, and this page prints that
   beside them. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, money = Q.money, smoney = Q.smoney, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable;
  const GUIDE = "technique-option-chains-and-greeks";
  const GUIDES = { fit: "technique-option-pricing-models-and-calibration", price: "technique-exotic-and-american-options", hedge: "technique-hedging-with-transaction-costs" };
  const ANALYTIC = new Set(["overview", "chain", "smile", "surface"]);
  const MODEL_MODES = new Set(["fit", "price", "hedge"]);
  const MODEL_SLOT = { market: 0, heston: 1, nig: 2, vg: 3, merton: 4, kou: 6 };                  // a colour for each model that does not change with what else is shown (the palette's own order, without its second green)
  const modelColor = (id) => (id === "bs" || id === "bs_continuous" ? css("--muted") : Q.colorOf(MODEL_SLOT[id] === undefined ? 6 : MODEL_SLOT[id]));
  const ALL_MODELS = ["merton", "kou", "vg", "nig", "heston"];
  const WINDOWS = [["5", "±5%"], ["10", "±10%"], ["15", "±15%"], ["25", "±25%"], ["40", "±40%"]];

  const O = {
    id: "options", root: "#options", view: "#view-options", title: "Options", api: 6, mode: "overview", busy: "", results: {}, errors: {}, cache: new Map(), guard: null, go: null, progress: null,
    about: "Read an option chain, see what a book of options is exposed to, draw what a structure pays and test a rolling strategy. The chain is a model quoted at the underlying's real price unless you download a real one (free, below) or bring your own file.",
    modes: [["overview", "Overview"], ["chain", "Chain"], ["smile", "Smile and skew"], ["surface", "Surface and term"], ["fit", "Model fit"], ["price", "Model prices"], ["hedge", "Hedging cost"], ["risk", "Greeks and risk"],
      ["strategy", "Strategy builder"], ["backtest", "Backtest"]],
    runLabel: { overview: "Refresh", chain: "Refresh", smile: "Refresh", surface: "Refresh", fit: "Fit the models", price: "Price the product", hedge: "Cost the hedge", risk: "Revalue the book", strategy: "Draw the payoff", backtest: "Run the backtest" },
    blurb: { overview: "Implied against realised volatility, where it ranks, the put-call balance, the term structure and the skew: the page to open first.",
      chain: "Calls and puts side by side for one expiry: bid, ask, implied volatility, delta, volume and open interest, with the at-the-money row marked and stale or wide quotes flagged.",
      smile: "How implied volatility changes with strike (the smile), across expiries, and what that says about the price of crash protection (risk reversal, butterfly).",
      surface: "The whole implied-volatility surface at once, the term structure and the checks that a surface admits no free lunch.",
      fit: "Fit five pricing models (jumps, infinite-activity jumps, stochastic volatility) to the chain, see which reproduces the smile, and what each says about the odds of a fall.",
      price: "Price a digital, barrier, average-price or American option under every fitted model. Models fitted to the same quotes differ on a vanilla by how well each fits; on an exotic by more. Set the spread beside the vanilla's: what is left is model risk.",
      hedge: "What hedging an option costs when every trade costs money: Leland's price, the band in which not to trade, and a simulation of the rules.",
      risk: "Delta, gamma, vega and theta of a book of options and shares, by expiry, and what it would make or lose when the price and implied volatility move.",
      strategy: "Choose a structure and see its payoff at expiry, today and half way, its break-evens, maximum profit and loss, margin, Greeks, probability of profit and expected value.",
      backtest: "Sell or buy premium on a schedule on a simulated option market and see the equity curve, where the profit came from by Greek, the costs and the trades." },
    form: { underlying: "", expiry: "", window: "15", rate: "", dividend: "", premium: "15", hideStale: false, minOi: "",
      book: [], spotShocks: "-10, -5, -2, 0, 2, 5, 10", volShocks: "-5, 0, 5, 10", days: "0",
      strategy: "long_straddle", sDte: "30", sFar: "", offset: "5", width: "5", quantity: "1", fill: "mid", spread: "5", commission: "0.65", viewVol: "", viewDrift: "", horizon: "", legs: [],
      bt: "short_put", btDays: "252", btSeed: "0", btCapital: "100000", btPremium: "25", btCommission: "0.65", btFill: "100", btParams: {},
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
  function syncExpiry() {                                                                   // the expiry menu follows the chain that was last loaded
    const sel = Q.$(`${O.root} select[data-key=expiry]`); if (!sel) return;
    sel.replaceChildren(); const first = h("option", "", sel, O.mode === "price" || O.mode === "hedge" ? "About three months" : "Nearest"); first.value = "";
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
  O.check = () => {
    if (!O.catalog) return "Loading the list of underlyings…";
    if (!O.form.underlying) return "There is no underlying to analyse.";
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
    if (m === "chain" || m === "smile") {
      const b = sec("2. Expiry and strikes");
      Q.labField(b, { key: "expiry", label: "Expiry", kind: "choice", options: [["", "Nearest"], ...O.expiries.map((e) => [e.expiry, `${e.expiry} · ${e.dte} days`])] }, f, rerun);
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
        const opts = [...cat.option_strategies.map((s) => [s.id, s.id.replace(/_/g, " ")]), ["custom", "your own legs"]];
        Q.labField(b, { key: "strategy", label: "Structure", kind: "choice", options: opts }, f, () => Q.renderLab(O));
        const info = cat.option_strategies.find((s) => s.id === f.strategy); h("p", "hint", b, f.strategy === "custom" ? "Enter the legs yourself below." : info ? info.text : "");
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
    }
    if (m === "backtest") {
      const b = sec("2. The strategy"), cat = O.catalog;
      if (cat) {
        Q.labField(b, { key: "bt", label: "Strategy", kind: "choice", options: cat.option_backtests.map((s) => [s.id, s.id.replace(/_/g, " ")]) }, f, () => Q.renderLab(O));
        const info = cat.option_backtests.find((s) => s.id === f.bt);
        if (info) {
          h("p", "hint", b, info.text);
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
        }
      }
      const c = sec("3. The market and costs");
      for (const s of [{ key: "btDays", label: "Trading days to simulate (60 to 756)", kind: "number", step: 1, min: 60, max: 756 }, { key: "btSeed", label: "Random seed", kind: "number", step: 1, min: 0 },
        { key: "btCapital", label: "Capital ($)", kind: "number", min: 1000 }, { key: "btPremium", label: "Variance risk premium built into the market (%)", kind: "number", step: 1, help: "How much more the simulated market charges for volatility than it delivers. A seller of premium earns it by construction." },
        { key: "btCommission", label: "Commission per contract ($)", kind: "number", step: 0.05, min: 0 }, { key: "btFill", label: "Share of each order that fills (%)", kind: "number", step: 5, min: 0, max: 100 }]) Q.labField(c, s, f, after);
      h("p", "hint", c, "The first run of a given length builds the simulated market and takes several seconds; the next ones reuse it.");
    }
  };

  /* ---------- running ---------- */
  O.run = (fromForm) => {
    const m = O.mode, f = O.form;
    if (ANALYTIC.has(m)) {
      if (!fromForm) O.cache.clear();                                                                       // the Refresh button asks again; a change in the form asks only for what it changed
      return W.request(O, () => {
        const q = chainQuery(), steps = [];
        if (m !== "surface") steps.push({ modes: m === "smile" ? ["overview", "chain", "smile"] : ["overview", "chain"], path: "/api/options/chain", query: q, cacheKey: key("chain", q) });
        if (m === "smile" || m === "surface") { const s = settings(); steps.push({ modes: m === "surface" ? ["surface"] : [], path: "/api/options/surface", query: s, cacheKey: key("surface", s) }); }
        return steps;
      }, "Reading the chain…").then(() => { const d = O.results[m] && O.results[m].data; if (d && m !== "surface") remember(d); if (m === "chain" || m === "smile") syncExpiry(); });
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
      Object.assign(body, { spread_pct: Number(f.spread) / 100, commission: Number(f.commission), view_vol: f.viewVol === "" ? undefined : Number(f.viewVol) / 100, view_drift: f.viewDrift === "" ? undefined : Number(f.viewDrift) / 100, horizon_days: f.horizon === "" ? undefined : Number(f.horizon) });
      return [{ modes: ["strategy"], path: "/api/options/strategy/payoff", body }];
    }, "Pricing the legs…");
    const params = {};
    for (const [k, v] of Object.entries(f.btParams[f.bt] || {})) if (v !== "" && v !== undefined) params[k] = v;
    return W.request(O, () => [{ modes: ["backtest"], job: true, path: "/api/options/strategy/backtest", body: { strategy: f.bt, params, days: Number(f.btDays), seed: Number(f.btSeed), capital: Number(f.btCapital), premium: Number(f.btPremium) / 100,
      commission: Number(f.btCommission), fill_fraction: Number(f.btFill) / 100 } }], "Running the backtest…");
  };
  O.auto = () => { if (ANALYTIC.has(O.mode) && O.form.underlying) O.run(true); };
  O.empty = (root) => {
    const text = { fit: ["Fit the pricing models to this chain", "Fit Merton, Kou, variance gamma, NIG and Heston to the out-of-the-money quotes and see which reproduces the smile, how much better it does than a flat volatility, and what each implies for the odds of a fall. About half a minute for all five."],
      price: ["Price a product under every model", "Choose a digital, barrier, average-price or American option. Each model is fitted to the chain (once per session) and prices it. The spread between them, set beside their spread on the plain option at the same strike (which is only how well each fits), is the model risk of the product, which no vanilla quote can tell you."],
      hedge: ["Cost the hedge of an option", "Hedging a sold option means trading the underlying, and every trade costs. See what that does to the price, where the no-trade band lies and which rule loses least on simulated paths."],
      risk: ["Value a book of options", "Add the options and shares you hold (or load the example), choose the moves to test and revalue: Greeks by expiry and a stress grid of price against implied volatility."],
      strategy: ["Draw what a structure pays", "Pick a structure and press the button: payoff at expiry, today and half way, break-evens, the most it can make and lose, the margin it needs and the odds."],
      backtest: ["Backtest a rolling option strategy", "Sell or buy premium on a schedule on a simulated option market, with the costs and the hedging booked. The market pays the premium it was built with, so this checks the machinery."] }[O.mode] || ["Read the chain", "Choose an underlying."];
    Q.labEmpty(root, O, text[0], [text[1]], O.runLabel[O.mode]);
  };
  O.draw = () => Q.drawLab(O);
  O.build = (root, d, body) => ({ overview: buildOverview, chain: buildChain, smile: buildSmile, surface: buildSurface, fit: buildFit, price: buildPrice, hedge: buildHedge, risk: buildRisk, strategy: buildStrategy, backtest: buildBacktest })[O.mode](root, d, body);

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
  const ROLLING = { covered_call: ["covered_call"], iron_condor: ["iron_condor"], short_strangle: ["short_strangle"], short_straddle: ["delta_hedged_straddle", { side: -1 }], long_straddle: ["delta_hedged_straddle", { side: 1 }], calendar_spread: ["calendar_spread"] };
  function buildStrategy(root, d) {
    const a = d.analysis, c = a.curve, f = O.form, name = d.strategy ? d.strategy.replace(/_/g, " ") : "your own legs";
    header(root, `${name[0].toUpperCase()}${name.slice(1)} on ${nameOf(d)}`, `${d.text} · expiry ${d.expiry} · horizon ${a.horizon_days} days`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root), cash = a.net_debit > 0 ? `pay ${money(a.net_debit)}` : `receive ${money(a.net_credit)}`;
    tile(k, "Most it can make", a.unlimited_profit ? "Unlimited" : smoney(a.max_profit), { text: a.reward_risk === null ? cash : `reward to risk ${num(a.reward_risk, 2)} · ${cash}` });
    tile(k, "Most it can lose", a.unlimited_loss ? "Unlimited" : smoney(a.max_loss), { text: a.unlimited_loss ? "no floor: size it with care" : "if held to the horizon" });
    tile(k, "Break-even" + (a.breakevens.length === 1 ? "" : "s"), a.breakevens.length ? a.breakevens.map((x) => num(x, 2)).join(" · ") : "none", { text: `price ${num(a.spot, 2)}` }, "", "Underlying prices at the horizon where the profit is zero");
    tile(k, "Probability of profit", pct(a.prob_profit, 0), { text: `expected value ${smoney(a.expected_value)}` }, "", `Lognormal price with volatility ${pct(a.view_vol, 1)} and drift ${pct(a.view_drift, 1)}; the smile, jumps and your view are not in it`);
    tile(k, "Margin (indicative)", money(a.margin.estimate), { text: a.margin.defined_risk ? "defined risk" : "naked short options" }, "", a.margin.basis);
    const ch = h("div", "charts", root), keep = c.spot.map((_, n) => n).filter((n) => c.spot[n] > 0), at = (arr) => keep.map((n) => arr[n]), xs = at(c.spot);        // the grid starts at a price of zero to find the worst case; the picture starts above it
    chartCard(ch, "Profit and loss by price of the underlying", "At the horizon (the line, shaded blue where it makes money and red where it loses), today, and half way to it. The dashed lines mark today's price and the break-evens (labelled with their prices).", { wide: true,
      draw: (b) => C.line(b, { xs, xLabel: "Price of the underlying", series: [{ name: "At the horizon", color: css("--s1"), values: at(c.expiry) }, { name: "Today", color: css("--s2"), values: at(c.now), width: 1.5 }, { name: "Half way", color: css("--s3"), values: at(c.half), width: 1.5 }],
        baseline: 0, include: 0, signFill: true, yFormat: (v) => smoney(v).replace("+", ""), tipFormat: (v) => smoney(v), xFormat: (v) => num(v, 0), height: 330, label: "Payoff diagram", endLabels: false,
        vlines: [{ x: a.spot, label: "Price" }, ...a.breakevens.map((x) => ({ x, label: num(x, 1) }))] }),
      table: (b) => table(b, [{ label: "Price", num: true, get: (n) => num(c.spot[n], 2) }, { label: "At the horizon", num: true, get: (n) => smoney(c.expiry[n]) }, { label: "Today", num: true, get: (n) => smoney(c.now[n]) }, { label: "Half way", num: true, get: (n) => smoney(c.half[n]) }], keep.filter((n, i) => i % 6 === 0)) });
    const g = a.greeks, cardG = h("div", "card", ch); h("h3", "", cardG, "Greeks of the structure today");
    table(cardG, [{ label: "Measure", get: (x) => x[0] }, { label: "Value", num: true, get: (x) => x[1] }], [["Delta (dollars)", smoney(g.delta_dollars)], ["Delta (share equivalent)", num(g.delta_shares, 1)], ["Gamma: P&L of a 1% move", smoney(g.gamma_pnl_1pct)], ["Vega per point", smoney(g.vega_per_point)],
      ["Theta per day", smoney(g.theta_per_day)], ["Rho per 1%", smoney(g.rho_per_1pct)], ["Margin basis", a.margin.basis], ["Premium received", money(a.margin.premium_received)]]);
    const legs = h("div", "card", ch); h("h3", "", legs, "Legs");
    table(legs, [{ label: "Leg", get: (l) => (l.type === "stock" ? "Shares" : `${l.right === "C" ? "Call" : "Put"} ${num(l.strike, 2)}`) }, { label: "Qty", num: true, get: (l) => num(l.quantity, 0) }, { label: "Days", num: true, get: (l) => (l.dte === undefined ? "" : l.dte) },
      { label: "IV", num: true, get: (l) => (l.iv === undefined ? "" : pct(l.iv, 1)) }, { label: "Price", num: true, get: (l) => (l.price === undefined ? "" : num(l.price, 2)) }], a.legs);
    const rolling = ROLLING[f.strategy];
    if (rolling) {                                                                           // the structures a rolling strategy exists for
      const go = h("button", "btn", root, "Backtest the rolling version of this idea");
      go.addEventListener("click", () => { O.form.bt = rolling[0]; if (rolling[1]) O.form.btParams[rolling[0]] = { ...rolling[1] }; O.mode = "backtest"; Q.renderLab(O); });
    }
  }

  /* ---------- backtest ---------- */
  function buildBacktest(root, d) {
    const s = d.summary, os = d.option_summary;
    header(root, `${d.strategy.replace(/_/g, " ")}: simulated option market`, `${d.n_days} trading days · ${d.start} to ${d.end} · capital ${money(d.capital)} · seed ${d.seed}`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Sharpe ratio", num(s.sharpe), { text: `Sortino ${num(s.sortino)} · Calmar ${num(s.calmar)}` });
    tile(k, "Return a year", Q.pct(s.cagr), { text: `volatility ${pct(s.ann_vol)}` });
    tile(k, "Total return", pct(s.total_return), { text: `on ${money(d.capital)} of capital` });
    tile(k, "Worst fall", pct(s.max_drawdown), { text: `worst day ${pct(s.worst_day)}` });
    tile(k, "Tail risk, 95%", pct(d.cvar_95, 2), { text: `a day: value at risk ${pct(d.var_95, 2)}` }, "", "Expected loss on the worst 5% of days (CVaR) and the loss exceeded on 5% of days (VaR)");
    tile(k, "Trades", grouped(d.trades), { text: `${money(os.costs_total, 0)} in costs` + (d.skipped_orders ? ` · ${d.skipped_orders} skipped` : "") });
    const ch = h("div", "charts", root);
    chartCard(ch, "Equity curve", "Growth of 1 in the strategy: the capital plus each day's profit over the capital. Margin and funding are not modelled.", { wide: true,
      draw: (b) => C.line(b, { dates: d.dates, series: [{ name: "Strategy", color: css("--s1"), values: d.equity }], area: true, areaBase: 1, baseline: 1, yFormat: (v) => v.toFixed(3), tipFormat: (v) => v.toFixed(4), height: 260, label: "Equity curve" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, { label: "Strategy", num: true, get: (n) => num(d.equity[n], 4) }, { label: "Underlying", num: true, get: (n) => num(d.benchmark[n], 4) }], d.dates.map((_, n) => n).filter((n) => n % 4 === 0)) });
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
    const trades = h("div", "card", ch); h("h3", "", trades, "Latest trades");
    table(trades, [{ label: "Date", get: (x) => x.date }, { label: "Instrument", get: (x) => (x.kind === "option" ? `${x.right === "C" ? "Call" : "Put"} ${num(x.strike, 2)} ${x.expiry}` : x.kind) }, { label: "Qty", num: true, get: (x) => num(x.quantity, 0) }, { label: "Price", num: true, get: (x) => num(x.price, 2) }, { label: "Cost ($)", num: true, get: (x) => num(x.cost, 2) }], d.recent_trades);
    W.hint(root, "The market pays the variance premium it was built with and has jumps, so a premium seller does well here by construction: the result checks that costs, hedges and settlement are booked correctly, not that the premium exists in your market.");
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
