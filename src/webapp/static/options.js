/* The Options workspace: what a chain says (overview, chain, smile and skew, surface and term structure), what a book of options is exposed to (Greeks and stress), what a structure pays (payoff, break-evens,
   probability of profit) and what a rolling strategy would have paid (a backtest on a simulated option market). It talks to /api/options/*; every payload says whether its numbers are a model, a file of yours or
   platform data, and this page prints that beside them. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, money = Q.money, smoney = Q.smoney, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable;
  const GUIDE = "technique-option-chains-and-greeks";
  const ANALYTIC = new Set(["overview", "chain", "smile", "surface"]);
  const WINDOWS = [["5", "±5%"], ["10", "±10%"], ["15", "±15%"], ["25", "±25%"], ["40", "±40%"]];

  const O = {
    id: "options", root: "#options", view: "#view-options", title: "Options", api: 5, mode: "overview", busy: "", results: {}, errors: {}, cache: new Map(), guard: null, go: null, progress: null,
    about: "Read an option chain, see what a book of options is exposed to, draw what a structure pays and test a rolling strategy. The chain is a model quoted at the underlying's real price unless you download a real one (free, below) or bring your own file.",
    modes: [["overview", "Overview"], ["chain", "Chain"], ["smile", "Smile and skew"], ["surface", "Surface and term"], ["risk", "Greeks and risk"], ["strategy", "Strategy builder"], ["backtest", "Backtest"]],
    runLabel: { overview: "Refresh", chain: "Refresh", smile: "Refresh", surface: "Refresh", risk: "Revalue the book", strategy: "Draw the payoff", backtest: "Run the backtest" },
    blurb: { overview: "Implied against realised volatility, where it ranks, the put-call balance, the term structure and the skew: the page to open first.",
      chain: "Calls and puts side by side for one expiry: bid, ask, implied volatility, delta, volume and open interest, with the at-the-money row marked and stale or wide quotes flagged.",
      smile: "How implied volatility changes with strike (the smile), across expiries, and what that says about the price of crash protection (risk reversal, butterfly).",
      surface: "The whole implied-volatility surface at once, the term structure and the checks that a surface admits no free lunch.",
      risk: "Delta, gamma, vega and theta of a book of options and shares, by expiry, and what it would make or lose when the price and implied volatility move.",
      strategy: "Choose a structure and see its payoff at expiry, today and half way, its break-evens, maximum profit and loss, margin, Greeks, probability of profit and expected value.",
      backtest: "Sell or buy premium on a schedule on a simulated option market and see the equity curve, where the profit came from by Greek, the costs and the trades." },
    form: { underlying: "", expiry: "", window: "15", rate: "", dividend: "", premium: "15", hideStale: false, minOi: "",
      book: [], spotShocks: "-10, -5, -2, 0, 2, 5, 10", volShocks: "-5, 0, 5, 10", days: "0",
      strategy: "long_straddle", sDte: "30", sFar: "", offset: "5", width: "5", quantity: "1", fill: "mid", spread: "5", commission: "0.65", viewVol: "", viewDrift: "", horizon: "", legs: [],
      bt: "short_put", btDays: "252", btSeed: "0", btCapital: "100000", btPremium: "25", btCommission: "0.65", btFill: "100", btParams: {} },
    catalog: null, spot: null, expiries: [], used: null,
  };
  const expiryColor = W.palette();
  const setCatalog = (c) => {                                                               // a download changed what exists: the menu follows, and a chain that was deleted is not left selected
    O.catalog = c;
    if (!c.underlyings.some((u) => u.id === O.form.underlying)) { O.form.underlying = (c.underlyings.find((u) => u.id === "SPY") || c.underlyings[0] || {}).id || ""; O.form.expiry = ""; O.spot = null; }
  };
  /* The Free data card: real chains from Cboe (and Yahoo when Cboe cannot be reached), and Bitcoin and Ether from Deribit. */
  const FREE = {
    kind: "options", title: "Real option chains (free)",
    intro: "Download real option quotes, about 15 minutes late, for US-listed tickers from Cboe, or Bitcoin and Ether options from Deribit. No account and no key. A downloaded chain appears in the Underlying menu as “real chain” and is analysed exactly like a model chain.",
    defaults: () => ({ chosen: new Set(["SPY"]), extra: "", maxDte: "365" }),
    controls: (box, f, ks, changed) => {
      h("span", "hint", box, "Pick tickers:");
      W.picks(box, [...ks.symbols.map((s) => [s, s]), ...ks.crypto.map((c) => ["deribit:" + c, c + " · Deribit", "Bitcoin and Ether options from Deribit, live"])], f.chosen, changed);
      Q.labField(box, { key: "extra", label: "Other tickers (separated by spaces)", kind: "text", placeholder: "NVDA AMD", help: "Any ticker that has listed options in the US" }, f, changed);
      Q.labField(box, { key: "maxDte", label: "Expiries up to", kind: "choice", options: ks.max_dte.map((d) => [String(d), d < 365 ? d + " days" : d === 365 ? "1 year" : d / 365 + " years"]), help: "Longer chains are bigger files and slower pages" }, f, changed);
    },
    request: (f) => {
      const items = [...f.chosen, ...String(f.extra).split(/[\s,;]+/).filter(Boolean)];
      if (!items.length) throw new Error("Pick a ticker, or type one");
      return { kind: "options", items, max_dte: Number(f.maxDte) };
    },
    estimate: (b) => b.items.length * 4,
    describe: (rec) => [rec.name, `${rec.source} · quotes of ${rec.as_of} · ${Number(rec.contracts || rec.rows).toLocaleString("en-US")} contracts`],
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
  const settings = () => ({ underlying: O.form.underlying, rate: O.form.rate === "" ? undefined : Number(O.form.rate) / 100, dividend: O.form.dividend === "" ? undefined : Number(O.form.dividend) / 100, premium: O.form.premium === "" ? undefined : Number(O.form.premium) / 100 });
  const chainQuery = () => ({ ...settings(), expiry: O.form.expiry || undefined, window: Number(O.form.window) / 100 });
  const key = (name, q) => name + ":" + JSON.stringify(q);
  const remember = (d) => { if (d && d.chain) { O.spot = d.chain.spot; O.expiries = d.chain.expiries; O.used = d.chain.expiry; syncExpiry(); } };
  function syncExpiry() {                                                                   // the expiry menu follows the chain that was last loaded
    const sel = Q.$(`${O.root} select[data-key=expiry]`); if (!sel) return;
    sel.replaceChildren(); const first = h("option", "", sel, "Nearest"); first.value = "";
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
    if (ANALYTIC.has(m) || m === "risk" || m === "strategy") {
      const b = sec("Assumptions", false);
      for (const s of [{ key: "rate", label: "Interest rate (% a year, blank = 4)", kind: "number", step: 0.25 }, { key: "dividend", label: "Dividend yield (% a year, blank = the ETF's)", kind: "number", step: 0.1, min: 0 },
        { key: "premium", label: "Implied over realised variance (%)", kind: "number", step: 1, help: "How much richer the model's implied volatility is than the recent realised volatility. The simulated chain only." }]) Q.labField(b, s, f, after);
      h("p", "hint", b, "Used to build and price the simulated chain. They change the numbers, so they are part of what you are looking at; a chain from your own file ignores them except for the rate and dividend it lacks, and a downloaded chain brings its own (the dividend of each expiry is read from the market's forward).");
    }
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
    const text = { risk: ["Value a book of options", "Add the options and shares you hold (or load the example), choose the moves to test and revalue: Greeks by expiry and a stress grid of price against implied volatility."],
      strategy: ["Draw what a structure pays", "Pick a structure and press the button: payoff at expiry, today and half way, break-evens, the most it can make and lose, the margin it needs and the odds."],
      backtest: ["Backtest a rolling option strategy", "Sell or buy premium on a schedule on a simulated option market, with the costs and the hedging booked. The market pays the premium it was built with, so this checks the machinery."] }[O.mode] || ["Read the chain", "Choose an underlying."];
    Q.labEmpty(root, O, text[0], [text[1]], O.runLabel[O.mode]);
  };
  O.draw = () => Q.drawLab(O);
  O.build = (root, d, body) => ({ overview: buildOverview, chain: buildChain, smile: buildSmile, surface: buildSurface, risk: buildRisk, strategy: buildStrategy, backtest: buildBacktest })[O.mode](root, d, body);

  /* ---------- overview ---------- */
  const header = (root, title, sub) => Q.labHead(root, title, sub, GUIDE);
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

  Q.registerWorkspace("options", O);
})();
