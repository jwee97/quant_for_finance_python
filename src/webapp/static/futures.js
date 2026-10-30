/* The Futures workspace: the forward curve and how it moved, the roll (which contract to hold and when to leave it), carry and basis, the price history of any futures product, a margin dashboard for a book of
   futures, and strategies run through the platform's pipeline on the simulated roots or on the real curves you downloaded. It talks to /api/futures/*. No futures prices are held in this repository: the curves are a model (Schwartz-Smith)
   unless a file of yours or a download replaces them, and every payload says which, and this page prints it. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, money = Q.money, smoney = Q.smoney, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable;
  const GUIDE = "technique-futures-curves-carry-and-rolls";
  const CURVE_MODES = new Set(["curve", "roll", "carry"]);
  const STATE_TEXT = { backwardation: "Backwardation", contango: "Contango", mixed: "Mixed (humped or kinked)" };
  const STATE_HELP = { backwardation: "Later contracts are cheaper: a long position earns the roll yield as the contract rolls up the curve.", contango: "Later contracts are dearer: a long position pays the roll yield as the contract rolls down the curve.",
    mixed: "The curve is neither clearly rising nor clearly falling over its front contracts." };

  const F = {
    id: "futures", root: "#futures", view: "#view-futures", title: "Futures", api: 12, mode: "curve", busy: "", results: {}, errors: {}, cache: new Map(), guard: null, go: null, progress: null,
    about: "Read a futures curve, decide when to roll, see carry and basis, read the price history of any futures product, check the margin of a book and test a strategy on several roots. The prices are a model unless you download real data (below) or bring a file.",
    modes: [["curve", "Curve"], ["roll", "Roll"], ["carry", "Carry and basis"], ["history", "History"], ["margin", "Margin"], ["backtest", "Backtest"], ["signals", "Signals"]],
    runLabel: { curve: "Refresh", roll: "Refresh", carry: "Refresh", history: "Refresh", margin: "Check the margin", backtest: "Run the backtest", signals: "Read the signals" },
    blurb: { curve: "Price by contract month, how the curve has moved, how its shape changed through time, what moves it (its principal components), the roll yield of each contract and the contract specification.",
      roll: "Which contract you hold, the next one, the suggested roll date (by calendar and by open interest), what the roll costs, and the rolls of the past.",
      carry: "Carry (the annualised roll yield) through time against the range of its own last year, its average in each calendar month, and the basis: how far a futures price sits from the spot price.",
      history: "The daily price history of a futures product downloaded from iTick or Yahoo Finance: returns, volatility, drawdown, the calendar months and the biggest one-day moves. It is one price series, so there is no curve, roll or carry here.",
      margin: "Notional, initial and maintenance margin, leverage and the distance to a margin call for a book of futures, with warnings for heavy use, concentration and contracts close to expiry.",
      backtest: "Trade carry, trend, breakout, skewness or a mix across the roots through the platform's pipeline, with the strategy's own settings: costed, lagged and volatility-targeted like any other backtest, with the risk measures and how the book was spread.",
      signals: "What every rule says about every root today (trend of several kinds, carry, basis momentum, skewness, the variance-ratio switch), each as a strength against its own usual size, and where the rules agree." },
    form: { root: "CL", series: "", rollDays: "5", spacing: "21", dates: "3", contracts: "12", book: [], equity: "100000", scale: "1", near: "7", strategy: "carry_ts", params: {}, roots: new Set(["CL", "NG", "GC", "HG", "ES"]), data: "simulated", realRoots: null, frequency: "weekly", targetVol: "", cost: "2" },
    catalog: null,
  };
  const contractColor = W.palette();
  const setCatalog = (c) => {
    F.catalog = c; if (!c.futures_roots.some((r) => r.id === F.form.root)) F.form.root = (c.futures_roots[0] || {}).id || "CL";
    const list = c.futures_series || []; if (!list.some((x) => x.id === F.form.series)) F.form.series = (list[0] || {}).id || "";
  };
  const CONTRACTS = [["4", "4 contracts"], ["6", "6 contracts"], ["8", "8 contracts"], ["10", "10 contracts"], ["12", "12 contracts"], ["16", "16 contracts"], ["24", "24 contracts"], ["36", "36 contracts"], ["0", "Every contract Yahoo lists (up to 14 years out)"]];
  const HISTORY = { "6mo": "6 months", "1y": "1 year", "2y": "2 years", "5y": "5 years", "10y": "10 years", "max": "As long as Yahoo has it" };
  const DERIBIT = [["curve", "Today's curve: mark prices and open interest (adds a day each time)"], ["history", "Every monthly and quarterly future since 2019, expired ones included (last trades, no open interest)"]];
  const YEARS = { 1: "1 year", 2: "2 years", 3: "3 years", 5: "5 years", 10: "10 years", 20: "20 years" };
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  let itick = null;                                                                          // the server's iTick status, kept by the card for its time estimate
  /* The Free data card, from two sources. Yahoo Finance: each contract's daily prices, stacked into a curve, for 44 roots (and Bitcoin and Ether futures from Deribit). iTick: the daily price history of any of
     the ~1,600 futures products it lists, which has no contract months and so no curve; the History view reads it. */
  const FREE = {
    kind: "futures", title: "Real futures data (free)",
    intro: "Download real futures prices. Yahoo Finance (no account, no key) gives the delayed daily price of each contract month of 44 roots, stacked into a forward curve (ask for every contract it lists: crude oil has about 50); the VIX futures (VX) come from Cboe's own files, every contract since 2013 with its open interest; Bitcoin and Ether futures come from Deribit, today's curve or all their past. Finzdata (needs your key) gives every contract month of 139 markets with open interest, expired ones included, but its free plan shows only the last 30 days. iTick (needs your key) gives the daily price history of any futures product it lists, about 1,600 in the United States, Hong Kong and China, but no curve. Expiry dates are computed from each exchange's rule unless the source sends them, Yahoo has no open interest, and a later download is merged with the earlier one, so the contracts that have expired keep their history.",
    defaults: () => ({ source: "yahoo", chosen: new Set(["CL", "ES", "GC"]), contracts: "10", history: "1y", deribit: "curve", products: new Set(["US:NQ"]), names: { "US:NQ": "E-mini Nasdaq-100 Futures" }, years: "5",
      yproducts: new Set(["ES=F"]), ynames: { "ES=F": "E-Mini S&P 500" }, yyears: "5", fproducts: new Set(["ES"]), fnames: { ES: "E-mini S&P 500" } }),
    controls: (box, f, ks, changed, redraw) => {
      itick = ks.itick_status;
      const seg = W.seg(box, [["yahoo", "Yahoo: curves"], ["finzdata", "Finzdata: every contract"], ["yahoo_series", "Yahoo: search any future"], ["itick", "iTick: any product"]], f.source, (v) => { f.source = v; redraw(); });
      seg.setAttribute("role", "group"); seg.setAttribute("aria-label", "data source"); seg.dataset.role = "source";
      if (f.source === "finzdata") {
        const z = ks.finzdata_status;
        W.finzdataBox(box, z, { choose: (rows) => { for (const r of rows) { f.fproducts.add(r.root); f.fnames[r.root] = r.name; } redraw(); } });
        if (!z || !z.configured) return;
        h("p", "hint", box, `Finzdata keeps the daily price, volume and open interest of every contract month of 139 markets, expired ones included, so you get a curve with open interest and its roll. The free plan shows only the last ${z.window_days} days: ` +
          "each download is merged into the file of the last one, so download again within three weeks to keep the history whole (or run experiments/finzdata_update.py every day). What is stored stays on this machine: Finzdata's terms forbid publishing its data.");
        W.finder(box, f.fproducts, f.fnames, { kind: "futures", source: "finzdata", groups: ks.finzdata.groups, markets: ks.finzdata.markets, after: changed });
        W.finzdataContracts(box, () => [...f.fproducts].slice(-1)[0] || "");
        return;
      }
      if (f.source === "yahoo_series") {
        h("p", "hint", box, "Search every future Yahoo Finance lists (CME Group and ICE US: about 340 continuous series are in the list, and “Ask Yahoo” finds the rest by name). Each gives one daily price history, the front month rolled by Yahoo, to read in the History view: no curve, roll or carry, and Yahoo has no open interest. No key is needed.");
        W.finder(box, f.yproducts, f.ynames, { kind: "futures", source: "yahoo", groups: ks.yahoo.groups, after: changed });
        Q.labField(box, { key: "yyears", label: "Years of history", kind: "choice", options: ks.yahoo.years.map((y) => [String(y), YEARS[y] || y + " years"]), help: "One call whatever you choose" }, f, changed);
        return;
      }
      if (f.source === "itick") {
        W.itickBox(box, ks.itick_status, "futures");
        h("p", "hint", box, "iTick keeps one daily price series for each futures product (NQ, CL, HSI), not one for each contract month: you get a price history to read in the History view, not a curve, a roll or carry, and no open interest.");
        W.finder(box, f.products, f.names, { kind: "futures", regions: ks.itick.regions, groups: ks.itick.groups, after: changed });
        Q.labField(box, { key: "years", label: "Years of history", kind: "choice", options: ks.itick.years.map((y) => [String(y), YEARS[y] || y + " years"]), help: "Each 1,000 daily bars (about four years) cost one call of your plan" }, f, changed);
        return;
      }
      h("span", "hint", box, "Pick roots:");
      for (const g of ks.groups) {
        h("span", "grouplabel", box, g);
        W.picks(box, ks.roots.filter((r) => r.group === g).map((r) => [r.id, r.id, `${r.name} · ${r.exchange}`]), f.chosen, changed);
      }
      h("span", "grouplabel", box, "Crypto");
      W.picks(box, ks.crypto.map((c) => [c.id, c.id + " · Deribit", `${c.name} futures from Deribit, live`]), f.chosen, changed);
      Q.labField(box, { key: "contracts", label: "Contracts per root", kind: "choice", options: CONTRACTS, help: "How many of the next delivery months to ask Yahoo for. Yahoo lists far more than 12 of some (crude oil: about 50, ten years out) and few of others (the E-mini S&P 500: 5); the far ones trade rarely" }, f, changed);
      Q.labField(box, { key: "history", label: "History to download", kind: "choice", options: ks.history.map((x) => [x, HISTORY[x] || x]), help: "Yahoo drops a contract when it expires, so this is the history of the contracts still alive. VX comes from Cboe (every contract since 2013, expired ones included) and is not limited by it" }, f, changed);
      Q.labField(box, { key: "deribit", label: "Bitcoin and Ether futures", kind: "choice", options: DERIBIT, help: "Only for BTC and ETH above" }, f, changed);
      h("p", "hint", box, "Need a product that is not here (lumber, the Nikkei, a single stock, iron ore, Hang Seng …)? Choose Yahoo's search or iTick above and search their lists.");
    },
    request: (f) => {
      if (f.source === "yahoo_series") { if (!f.yproducts.size) throw new Error("Choose at least one future"); return { kind: "futures", source: "yahoo_series", items: [...f.yproducts], years: Number(f.yyears) }; }
      if (f.source === "itick") { if (!f.products.size) throw new Error("Choose at least one product"); return { kind: "futures", source: "itick", items: [...f.products], years: Number(f.years) }; }
      if (f.source === "finzdata") { if (!f.fproducts.size) throw new Error("Choose at least one market"); return { kind: "futures", source: "finzdata", items: [...f.fproducts] }; }
      if (!f.chosen.size) throw new Error("Pick at least one root");
      return { kind: "futures", items: [...f.chosen], contracts: Number(f.contracts), history: f.history, deribit: f.deribit };
    },
    estimate: (b) => {
      if (b.source === "yahoo_series") return b.items.length * 2;
      if (b.source === "finzdata") return b.items.length * 3;
      if (b.source !== "itick") return b.items.reduce((sum, r) => sum + (r === "VX" ? 50 : (r === "BTC" || r === "ETH") ? (b.deribit === "history" ? 40 : 3) : 4 + (b.contracts || 60) * 0.4), 0);
      const info = itick || {}, cpm = info.calls_per_minute || 5, page = info.page_size || 1000, free = info.calls_available_now === undefined ? cpm : info.calls_available_now;
      const calls = b.items.length * Math.max(1, Math.ceil((b.years * ((info.bars_per_year || {}).future || 260)) / page));
      return 2 * calls + Math.max(0, calls - free) * (60 / cpm);
    },
    describe: (rec) => rec.item === "series" ? [`${rec.code} · ${rec.root_name || rec.name} (${rec.region === "YF" ? "Yahoo" : rec.region})`, `${rec.region === "YF" ? "Yahoo Finance" : "iTick"} · price history · ${Number(rec.bars || rec.rows).toLocaleString("en-US")} daily bars · ${rec.first_date} to ${rec.as_of}`]
      : [`${rec.name} · ${rec.root_name || ""}`.replace(/ · $/, ""), `${rec.source} · ${rec.contracts} contracts · ${rec.first_date && rec.first_date !== rec.as_of ? rec.first_date + " to " : ""}${rec.as_of}${rec.has_open_interest ? " · with open interest" : ""}${rec.gap_count ? ` · ${rec.gap_count} gap${rec.gap_count > 1 ? "s" : ""}` : ""}`],
    setCatalog,
    pick: (res) => {
      const first = res.results.find((r) => r.ok); if (!first) return;
      if (first.what === "series") { F.mode = "history"; F.form.series = first.name; return; }
      F.form.root = first.name; if (F.mode === "history") F.mode = "curve";
    },
  };
  W.catalog().then((c) => { F.catalog = c; if (!Q.$(F.view).hidden) Q.renderLab(F); }, () => {});
  const rootsList = () => (F.catalog ? F.catalog.futures_roots : []);
  const rootInfo = (id) => rootsList().find((r) => r.id === id) || {};
  const realRoots = () => rootsList().filter((r) => r.source !== "simulated");                // the roots with a downloaded or uploaded curve
  const realChosen = () => { const ids = new Set(realRoots().map((r) => r.id)), f = F.form; if (!f.realRoots) f.realRoots = new Set(ids); for (const id of [...f.realRoots]) if (!ids.has(id)) f.realRoots.delete(id); return f.realRoots; };
  const key = (name, q) => name + ":" + JSON.stringify(q);
  const strategyInfo = (id) => (F.catalog ? (F.catalog.futures_strategies || []).find((x) => x.id === id) : null) || null;

  /* ---------- the form ---------- */
  F.check = () => {
    if (!F.catalog) return "Loading the list of roots…";
    if (F.mode === "history" && !F.form.series) return "Download a product first (the card below: choose Yahoo's search or iTick).";
    if (F.mode === "backtest" || F.mode === "signals") {
      const chosen = F.form.data === "real" ? realChosen() : F.form.roots, info = strategyInfo(F.form.strategy);
      if (!chosen.size) return F.form.data === "real" ? "Download a curve first (the card below), then tick it here." : "Tick at least one root.";
      if (F.mode === "backtest" && info && ["cross", "class"].includes(info.needs) && chosen.size < 2) return "A cross-sectional strategy needs at least two roots.";
    }
    return "";
  };
  F.sections = (panel) => {
    const f = F.form, after = () => Q.syncLab(F), sec = (title, open = true) => { const d = h("details", "", panel); d.open = open; h("summary", "", d, title); return h("div", "body", d); };
    const m = F.mode, cat = F.catalog;
    if (!cat) { h("p", "hint", sec("1. Root"), "Loading the list of roots…"); return; }
    if (CURVE_MODES.has(m)) {
      const b = sec("1. Root");
      Q.labField(b, { key: "root", label: "Futures root", kind: "choice", options: cat.futures_roots.map((r) => [r.id, `${r.id}: ${r.name}${r.source === "user_file" ? " (your file)" : r.source === "downloaded" ? " (downloaded)" : ""}`]) }, f, () => F.run(true));
      const info = rootInfo(f.root);
      h("p", "hint", b, info.source === "user_file" ? "Prices are read from your file in data/user/futures/." : info.source === "downloaded" ? "Real prices, downloaded from a free source and delayed: see the notes under the badge for what they are and are not." : "This curve is a model (Schwartz-Smith: a short-term factor that decays and a long-term level), with simulated open interest. Download the real curve below, or put " + f.root + ".csv in data/user/futures/ to use your own.");
      if (m === "roll") Q.labField(b, { key: "rollDays", label: "Leave the contract this many days before expiry", kind: "number", step: 1, min: 0, max: 60 }, f, () => F.run(true));
      if (m === "curve") {
        Q.labField(b, { key: "spacing", label: "Draw the earlier curves", kind: "choice", options: [["5", "a week apart"], ["21", "a month apart"], ["63", "three months apart"], ["126", "six months apart"], ["252", "a year apart"]], help: "The forward curve chart shows today's curve and the ones this far back, one after another" }, f, () => F.run(true));
        Q.labField(b, { key: "dates", label: "How many curves", kind: "choice", options: [["2", "2"], ["3", "3"], ["4", "4"], ["5", "5"], ["6", "6"]] }, f, () => F.run(true));
        Q.labField(b, { key: "contracts", label: "Contracts shown", kind: "choice", options: [["6", "the nearest 6"], ["12", "the nearest 12"], ["24", "the nearest 24"], ["60", "the nearest 60"], ["0", "every contract the file has"]], help: "How far out the forward curve and its table go. The shape through time follows up to 24 contract months; what moves the curve stays on the nearest 8" }, f, () => F.run(true));
      }
    }
    if (m === "history") {
      const b = sec("1. Product"), list = cat.futures_series || [];
      if (list.length) {
        Q.labField(b, { key: "series", label: "Downloaded price history", kind: "choice", options: list.map((x) => [x.id, `${x.code}: ${x.name} (${x.region})`]) }, f, () => F.run(true));
        h("p", "hint", b, "Daily prices as the provider keeps them for the product. They are not the prices of one contract: see the notes under the badge for how the series is joined, and the list of the biggest moves for where it may jump.");
      } else h("p", "hint", b, "No price history is downloaded yet. Use the card below: choose Yahoo's search (no key) or iTick, search for a product (nasdaq, crude, gold, lumber …) and press Download real data.");
    }
    if ((m !== "backtest" && m !== "signals") || f.data === "real") W.dataCard(panel, F, FREE);
    if (m === "margin") {
      const b = sec("1. The book");
      h("p", "hint", b, "Quantities: positive long, negative short. Leave the contract blank to use the front one. Margins are illustrative figures per contract, not your broker's.");
      marginRows(b, f.book, after);
      const ex = h("button", "btn", b, "Load an example"); ex.addEventListener("click", () => { f.book = [{ root: "CL", contract: "", quantity: "2" }, { root: "GC", contract: "", quantity: "1" }, { root: "ES", contract: "", quantity: "-1" }]; Q.renderLab(F); });
      const c = sec("2. Account");
      for (const s of [{ key: "equity", label: "Account equity ($)", kind: "number", min: 1000 }, { key: "scale", label: "Margin multiplier (1 = the table's figures)", kind: "number", step: 0.1, min: 0.1, max: 10, help: "Brokers raise margins in stress; try 1.5 or 2" },
        { key: "near", label: "Warn when a contract expires within (days)", kind: "number", step: 1, min: 0, max: 60 }]) Q.labField(c, s, f, after);
    }
    if (m === "backtest" || m === "signals") {
      if (m === "backtest") {
        const box = sec("1. The strategy");
        Q.labField(box, { key: "strategy", label: "Strategy", kind: "choice", options: cat.futures_strategies.map((x) => [x.id, x.label, x.group]) }, f, () => Q.renderLab(F));
        const info = strategyInfo(f.strategy);
        if (info) {
          h("p", "hint", box, info.text);
          Q.refsLine(box, info.references);
          if (info.needs === "cross") h("p", "hint", box, "Needs at least two roots: it works by comparing them with each other.");
          else if (info.needs === "class") h("p", "hint", box, "Compares roots within an asset class: needs at least two roots of the same class (CL, NG, HG are commodities).");
          else if (info.needs === "commodities") h("p", "hint", box, "Ranks commodities: needs at least two commodity roots (CL, NG, HG).");
          if (info.params.length) {
            h("h4", "subhead", box, "Settings of this strategy");
            const grid = h("div", "params", box), mine = f.params[info.id] = f.params[info.id] || {};
            for (const q of info.params) {
              const cur = q.name in mine ? mine[q.name] : q.default;
              grid.appendChild(Q.fieldFor(q, cur, (v) => { if (JSON.stringify(v) === JSON.stringify(q.default)) delete mine[q.name]; else mine[q.name] = v; after(); }, { wide: q.kind === "json" }));
            }
            h("p", "hint", box, "Only what you change is sent; the rest is the strategy's own default.");
          }
        }
      }
      const roots = sec(m === "backtest" ? "2. Prices and roots" : "1. Prices and roots");
      Q.labField(roots, { key: "data", label: "Prices", kind: "choice", options: [["simulated", "Simulated roots (a model)"], ["real", "Real curves (a backtest)"]] }, f, () => Q.renderLab(F));
      const box = h("div", "checks", roots), real = f.data === "real", pool = real ? realRoots() : cat.futures_roots.filter((x) => x.source === "simulated" || x.id in { CL: 1, NG: 1, GC: 1, HG: 1, ES: 1 }), chosen = real ? realChosen() : f.roots;
      h("span", "hint", box, m === "backtest" ? "Tick the roots to trade:" : "Tick the roots to read:");
      for (const r of pool) {
        const label = h("label", "f inline", box), c = h("input", "", label); c.type = "checkbox"; c.checked = chosen.has(r.id);
        label.appendChild(document.createTextNode(`${r.id}: ${r.name} (${r.asset_class})` + (real ? (r.first_date ? ` · ${r.first_date} to ${r.as_of} · ${r.contracts} contracts` : " · your file") : "")));
        c.addEventListener("change", () => { if (c.checked) chosen.add(r.id); else chosen.delete(r.id); after(); });
      }
      if (real && !pool.length) h("p", "hint", box, "No real curves yet. Download Cboe's VX futures or Deribit's BTC history with the card below.");
      h("p", "hint", roots, real
        ? "A backtest needs the front month on every day, so it needs the contracts that have expired. Cboe's VX (since 2013) and Deribit's history (since 2019) keep them; Yahoo lists only unexpired contracts, so a Yahoo download can be backtested only after it has been repeated over a year (each download keeps what expired since). Roots with less than a year of front-month history are refused."
        : "The simulated roots share one calendar and one model. Choose Real curves to trade what you downloaded.");
      if (m === "backtest") {
        const c = sec("3. Trading");
        Q.labField(c, { key: "frequency", label: "Rebalance", kind: "choice", options: [["daily", "Daily"], ["weekly", "Weekly"], ["monthly", "Monthly"]] }, f, after);
        for (const x of [{ key: "targetVol", label: "Volatility target (% a year, blank = the platform's)", kind: "number", step: 1, min: 1, max: 100 }, { key: "cost", label: "Cost of trading, one way (bps)", kind: "number", step: 0.5, min: 0, max: 200 }]) Q.labField(c, x, f, after);
        h("p", "hint", c, real ? "A run builds the front-month series of each root from its contract table: several seconds a root the first time, then it is reused." : "The first run builds the simulated futures and takes up to half a minute; later runs reuse them.");
      }
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
      if (!fromForm) W.dropCache(F);
      return W.request(F, () => {
        const q = { root: f.root, spacing: Number(f.spacing), dates: Number(f.dates), contracts: Number(f.contracts) }, steps = [{ modes: ["curve", "carry"], path: "/api/futures/curve", query: q, cacheKey: key("curve", q) }];
        if (m === "roll") { const r = { root: f.root, roll_days: Number(f.rollDays) }; steps.push({ modes: ["roll"], path: "/api/futures/roll", query: r, cacheKey: key("roll", r) }); }
        return steps;
      }, "Reading the curve…");
    }
    if (m === "history") {
      return W.request(F, () => {
        if (!f.series) throw new Error("Download a product first");
        const q = { name: f.series };
        return [{ modes: ["history"], path: "/api/futures/series", query: q, cacheKey: key("series", q) }];
      }, "Reading the price history…");
    }
    if (m === "margin") return W.request(F, () => {
      if (!f.book.length) throw new Error("Add at least one position");
      return [{ modes: ["margin"], path: "/api/futures/margin", body: { positions: f.book.map((x, i) => { const q = Number(x.quantity); if (!isFinite(q) || q === 0) throw new Error(`Row ${i + 1} needs a quantity`); return { root: x.root, contract: x.contract || undefined, quantity: q }; }),
        equity: Number(f.equity), margin_scale: Number(f.scale), near_expiry_days: Number(f.near) } }];
    }, "Computing the margin…");
    if (m === "signals") return W.request(F, () => [{ modes: ["signals"], job: true, path: "/api/futures/signals", body: { data: f.data, roots: f.data === "real" ? [...realChosen()] : [...f.roots] } }], "Reading the signals…");
    return W.request(F, () => [{ modes: ["backtest"], job: true, path: "/api/futures/strategy/backtest", body: { strategy: f.strategy, data: f.data, roots: f.data === "real" ? [...realChosen()] : [...f.roots], frequency: f.frequency, params: f.params[f.strategy] || {},
      target_vol: f.targetVol === "" ? undefined : Number(f.targetVol) / 100, cost_bps: Number(f.cost) } }], "Running the backtest…");
  };
  F.auto = () => { if ((CURVE_MODES.has(F.mode) || (F.mode === "history" && F.form.series)) && F.catalog) F.run(true); };
  F.empty = (root) => {
    const text = { history: ["Read a price history", "Download a futures product with iTick (the card below: choose iTick, search, press Download real data) and its returns, volatility, drawdown and biggest moves appear here."],
      margin: ["Check the margin of a book", "Add the contracts you hold (or load the example) to see the notional, the margin they need, the leverage and how far you are from a margin call."],
      backtest: ["Backtest a futures strategy", "Carry, trend, breakout, skewness or a mix across the roots, with the strategy's own settings, costed and lagged like every other backtest here. On the simulated roots the prices are a model, so it checks the machinery, not the strategy; on a real curve (VX, Deribit) it is a backtest."],
      signals: ["Read what the rules say today", "Every trend rule, carry, basis momentum, skewness and the variance-ratio switch, on every root you tick, each as a strength against its own usual size, and where they agree."] }[F.mode] || ["Read the curve", "Loading."];
    Q.labEmpty(root, F, text[0], [text[1]], F.runLabel[F.mode]);
  };
  F.draw = () => Q.drawLab(F);
  F.build = (root, d, body) => ({ curve: buildCurve, roll: buildRoll, carry: buildCarry, history: buildHistory, margin: buildMargin, backtest: buildBacktest, signals: buildSignals })[F.mode](root, d, body);
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
    if (c.listed > rows.length) W.hint(root, `The curve lists the nearest ${rows.length} of the ${c.listed} contracts the file has for ${c.date}. Choose "every contract the file has" under Contracts shown to see them all.`);
    chartCard(ch, "Forward curve", `Price of each contract month today and on the earlier dates in the legend (matching contract months by their place in the curve: M1 is the front). The shape shows what the market charges to hold the commodity: the choice of how far apart the curves are drawn is on the left.`, { wide: true,
      draw: (b) => C.line(b, { xs: last.map((_, i) => i + 1), xWhole: true, xLabel: "Contract month (1 = front)", series: hist.map((x, n) => ({ name: n === 0 ? `Today (${x.date})` : x.date, color: contractColor(x.date), values: x.points.map((p) => p.price), dots: n === 0, dotSize: 3, width: n === 0 ? 2 : 1.5 })),
        yFormat: (v) => num(v, 2), tipFormat: (v) => num(v, 2), xFormat: (v) => "M" + Math.round(v), tipTitle: (i) => `Month ${i + 1}: ${last[i].contract} (${last[i].dte} days)`, height: 300, label: "Forward curve", endLabels: false }),
      table: (b) => table(b, [{ label: "Month", get: (r) => "M" + r.rank }, { label: "Contract", get: (r) => r.contract }, { label: "Expiry", get: (r) => r.expiry }, { label: "Days", num: true, get: (r) => r.dte }, { label: "Price", num: true, get: (r) => num(r.price, 3) },
        { label: "vs front", num: true, get: (r) => Q.fmt.spct(r.change_vs_front) }, { label: "Roll yield (a year)", num: true, get: (r) => (r.roll_yield === null ? "" : pct(r.roll_yield, 1)) }, { label: "Open interest", num: true, get: (r) => (r.open_interest === null ? "" : Math.round(r.open_interest).toLocaleString("en-US")) },
        { label: "Volume", num: true, get: (r) => (r.volume === null ? "" : Math.round(r.volume).toLocaleString("en-US")) }], rows) });
    const ry = rows.filter((r) => r.roll_yield !== null);
    chartCard(ch, "Roll yield by contract", "Each contract against the next, annualised: above zero (backwardation) a long position gains as the contract ages, below zero (contango) it pays.", {
      draw: (b) => C.columns(b, { labels: ry.map((r) => r.contract), values: ry.map((r) => r.roll_yield), yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 230, name: "Roll yield", label: "Roll yield by contract", tipLabel: (i) => `${ry[i].contract} against the next (${ry[i].dte} days)` }),
      table: (b) => table(b, [{ label: "Contract", get: (r) => r.contract }, { label: "Roll yield (a year)", num: true, get: (r) => pct(r.roll_yield, 2) }], ry) });
    const sh = d.shape;
    if (sh && sh.dates.length > 2) chartCard(ch, "How the shape of the curve changed", "Each cell is how much cheaper (blue, backwardation: a long earns the roll) or dearer (red, contango: a long pays it) that contract month is than the front month, on the date at the top, one column a month. Rows that change colour together are the curve moving from one regime to the other.", { wide: true,
      draw: (b) => C.matrix(b, { rows: sh.ranks.map((r) => "M" + r), cols: sh.dates.map((x) => x.slice(2, 7)), values: sh.values.map((row) => row.map((v) => (v === null ? null : -v))), diverging: true, mid: 0, format: (v) => (v * 100).toFixed(1) + "%", name: "Cheaper than the front by", cellText: false, cellHeight: 22,
        tipRows: (i, j) => [["Price against the front", sh.values[i][j] === null ? "n/a" : Q.fmt.spct(Math.expm1(sh.values[i][j]), 1)]], label: "Shape of the curve through time" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => sh.dates[n] }, ...sh.ranks.map((r, j) => ({ label: "M" + r, num: true, get: (n) => (sh.values[j][n] === null ? "" : Q.fmt.spct(Math.expm1(sh.values[j][n]), 1)) }))], sh.dates.map((_, n) => n).reverse()) });
    const pc = d.pca;
    if (pc && pc.components.length) chartCard(ch, "What moves the curve", `The principal components of ${pc.n_days.toLocaleString("en-US")} days of price changes across the contract months: each line is how one factor moves each contract (its loadings), and the legend says how much of the variance it holds. One sign across the curve is a level move, a tilt a slope move, a hump a curvature move.`, {
      draw: (b) => C.line(b, { xs: pc.ranks, xWhole: true, xLabel: "Contract month (1 = front)", series: pc.components.map((c, k) => ({ name: `${c.label[0].toUpperCase()}${c.label.slice(1)} · ${(c.explained * 100).toFixed(1)}%`, color: Q.colorOf(k), values: c.loadings, dots: true, dotSize: 3 })),
        baseline: 0, include: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), xFormat: (v) => "M" + Math.round(v), height: 260, label: "Loadings of the principal components", endLabels: false }),
      table: (b) => table(b, [{ label: "Factor", get: (c) => c.label }, { label: "Share of the variance", num: true, get: (c) => pct(c.explained, 1) }, ...pc.ranks.map((r, j) => ({ label: "M" + r, num: true, get: (c) => num(c.loadings[j], 2) }))], pc.components) });
    const oi = rows.filter((r) => r.open_interest !== null);
    if (oi.length) chartCard(ch, "Open interest by contract", `Contracts outstanding. The front month is where most positions sit until the roll${d.sources.open_interest === "simulated" ? " (simulated here)" : ""}.`, {
      draw: (b) => C.columns(b, { labels: oi.map((r) => r.contract), values: oi.map((r) => r.open_interest), yFormat: (v) => (v >= 1000 ? (v / 1000).toFixed(0) + "k" : String(v)), tipFormat: (v) => Math.round(v).toLocaleString("en-US"), height: 230, name: "Open interest", label: "Open interest" }),
      table: (b) => table(b, [{ label: "Contract", get: (r) => r.contract }, { label: "Open interest", num: true, get: (r) => Math.round(r.open_interest).toLocaleString("en-US") }, { label: "Volume", num: true, get: (r) => (r.volume === null ? "" : Math.round(r.volume).toLocaleString("en-US")) }], oi) });
    else W.hint(root, "This curve has no open interest, so there is no chart of it: the feed it came from does not carry it.");
    sources(root, d.references, [["Why a curve slopes: storage and the convenience yield", "curve"], ["Convenience yield and the two-factor model of commodity prices", "convenience"], ["The factors of a curve", "pca"]]);
  }
  /* The papers and books behind a view, folded away under a heading: ``topics`` is [[heading, key]] into the ``refs`` of the payload. */
  function sources(root, refs, topics) {
    const shown = topics.filter(([, k]) => refs && refs[k] && refs[k].length);
    if (!shown.length) return;
    const dd = h("details", "how", root); h("summary", "", dd, "Where these ideas come from");
    for (const [title, k] of shown) { h("p", "hint", dd, title); Q.refsLine(dd, refs[k]); }
  }

  /* ---------- roll ---------- */
  function buildRoll(root, d) {
    const r = d.roll;
    header(root, `Rolling ${d.root}: ${r.current_contract} into ${r.next_contract}`, `${r.front_expiry} expiry · ${r.days_to_expiry} days left · roll ${r.roll_days} days before expiry`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Suggested roll date", W.dayMonth(r.suggested_roll_date), { text: `${r.suggested_roll_date.slice(0, 4)} · in ${r.days_to_roll} day${r.days_to_roll === 1 ? "" : "s"}` }, "", "The earlier of the calendar date and the open-interest crossover");
    tile(k, "Calendar roll", W.dayMonth(r.calendar_roll_date), { text: `${r.calendar_roll_date.slice(0, 4)} · ${r.roll_days} days before expiry` });
    const hs = r.history, hasOi = [...hs.oi_front, ...hs.oi_next].some((v) => v !== null && v !== undefined);              // a downloaded curve may come from a feed without open interest
    tile(k, "Open-interest crossover", !hasOi ? "No data" : r.oi_crossover ? W.dayMonth(r.oi_crossover) : "not yet", { text: !hasOi ? "this curve has no open interest" : r.oi_crossover ? `${r.oi_crossover.slice(0, 4)} · the next contract holds more` : `${r.next_contract} is still the smaller one` }, "", "The date from which the next contract's open interest stays above the front's");
    tile(k, "Cost of the roll", smoney(-r.roll_cost_dollars), { text: `${pct(r.roll_cost_pct, 2)} of the position's value` }, "", "For a LONG position: the calendar spread you pay when selling the old contract and buying the next, plus a tick of execution cost. Negative means you pay; a short position earns it.");
    tile(k, "Calendar spread", num(r.calendar_spread_points, 3), { text: `${money(r.calendar_spread_dollars)} per contract` }, "", "Next contract's price less the front's");
    tile(k, "Annualised roll", pct(r.annualised_roll, 1), { text: r.annualised_roll >= 0 ? "a long earns" : "a long pays" });
    const ch = h("div", "charts", root);
    chartCard(ch, "Front and next contract", "The two contracts' prices over the last 120 days; the gap is the calendar spread you cross when you roll.", {
      draw: (b) => C.line(b, { dates: hs.dates, series: [{ name: r.current_contract, color: css("--s1"), values: hs.front }, { name: r.next_contract, color: css("--s2"), values: hs.next }], yFormat: (v) => num(v, 1), tipFormat: (v) => num(v, 3), height: 250, label: "Front and next contract" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => hs.dates[n] }, { label: r.current_contract, num: true, get: (n) => num(hs.front[n], 3) }, { label: r.next_contract, num: true, get: (n) => num(hs.next[n], 3) }, { label: "Spread", num: true, get: (n) => num(hs.spread[n], 3) }], hs.dates.map((_, n) => n).filter((n) => n % 5 === 0)) });
    chartCard(ch, "Calendar spread", "Next contract less the front: above zero is contango, below zero backwardation.", {
      draw: (b) => C.line(b, { dates: hs.dates, series: [{ name: "Spread", color: css("--s1"), values: hs.spread }], area: true, areaBase: 0, baseline: 0, include: 0, yFormat: (v) => num(v, 2), tipFormat: (v) => num(v, 3), height: 250, label: "Calendar spread" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => hs.dates[n] }, { label: "Spread", num: true, get: (n) => num(hs.spread[n], 3) }], hs.dates.map((_, n) => n).filter((n) => n % 5 === 0)) });
    if (hasOi) chartCard(ch, "Open interest: where the positions are", "Positions outstanding in the two contracts you roll between, over the last 120 days. A contract's open interest builds as it nears the front; the crossover date is when the next contract has become the liquid one.", { wide: true,
      draw: (b) => C.line(b, { dates: hs.dates, series: [{ name: r.current_contract, color: css("--s1"), values: hs.oi_front }, { name: r.next_contract, color: css("--s2"), values: hs.oi_next }], include: 0, yFormat: (v) => (v >= 1000 ? (v / 1000).toFixed(0) + "k" : String(Math.round(v))), tipFormat: (v) => Math.round(v).toLocaleString("en-US"), height: 250, label: "Open interest" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => hs.dates[n] }, { label: r.current_contract, num: true, get: (n) => Math.round(hs.oi_front[n]).toLocaleString("en-US") }, { label: r.next_contract, num: true, get: (n) => Math.round(hs.oi_next[n]).toLocaleString("en-US") }], hs.dates.map((_, n) => n).filter((n) => n % 5 === 0)) });
    const past = h("div", "card", root); h("h3", "", past, "The rolls of the past year");
    if (r.past_rolls.length) table(past, [{ label: "Contract", get: (x) => x.contract }, { label: "Rolled into", get: (x) => x.next }, { label: "Expiry", get: (x) => x.expiry }, { label: "Calendar roll", get: (x) => x.roll_date }, { label: "Spread at the roll", num: true, get: (x) => num(x.spread, 3) }, { label: "Open-interest crossover", get: (x) => x.oi_crossover || (hasOi ? "never" : "no data") }], r.past_rolls);
    else h("p", "hint", past, "No contract in this file has expired yet, so there is no past roll to show. A downloaded curve gains them as later downloads are merged into it.");
    W.hint(root, "The cost shown is for a long position; a short position's is the opposite. Execution cost is one tick per leg. Prices and open interest are simulated unless your file or a download has them.");
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
    const hasMomentum = b.basis_momentum !== null && b.basis_momentum !== undefined && b.basis_momentum_series.dates.length > 0;
    tile(k, "Basis momentum", hasMomentum ? num(b.basis_momentum * 100, 2) + "%" : "n/a", { text: hasMomentum ? `percentile ${pct(b.basis_momentum_percentile, 0)}` : "needs about a year of prices" }, "", "Front-contract return over the last year less the second contract's: commodities whose front has beaten the second tend to keep doing so (Boons and Prado)");
    const ch = h("div", "charts", root), s = cy.series;
    const bands = s.bands && s.bands.p50 ? [{ lo: s.bands.p10.map((v) => (v === null ? null : v)), hi: s.bands.p90.map((v) => (v === null ? null : v)), color: css("--s1"), opacity: 0.1, name: "Middle 80% of the last year" }, { lo: s.bands.p25, hi: s.bands.p75, color: css("--s1"), opacity: 0.18, name: "Middle 50%" }] : [];
    chartCard(ch, "Carry through time", "Annualised roll yield and its 21-day average. Above zero a long position earns carry. The bands are the range of the carry of the year before each day: a carry outside the dark band is unusual for this market.", { wide: bands.length > 0,
      draw: (c) => C.line(c, { dates: s.dates, series: [{ name: "Carry", color: css("--s1"), values: s.carry, width: 1.5 }, { name: "21-day average", color: css("--s2"), values: s.carry_ma }, ...(bands.length ? [{ name: "Median of the last year", color: css("--muted"), dash: "5 4", width: 1.25, values: s.bands.p50 }] : [])], bands, baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 280, label: "Carry through time", endLabels: false }),
      table: (c) => table(c, [{ label: "Date", get: (n) => s.dates[n] }, { label: "Carry", num: true, get: (n) => pct(s.carry[n], 2) }, { label: "21-day average", num: true, get: (n) => pct(s.carry_ma[n], 2) }, ...(bands.length ? [{ label: "10th percentile of the year", num: true, get: (n) => (s.bands.p10[n] === null ? "" : pct(s.bands.p10[n], 1)) }, { label: "Median", num: true, get: (n) => (s.bands.p50[n] === null ? "" : pct(s.bands.p50[n], 1)) }, { label: "90th percentile", num: true, get: (n) => (s.bands.p90[n] === null ? "" : pct(s.bands.p90[n], 1)) }] : [])], s.dates.map((_, n) => n).filter((n) => n % 10 === 0)) });
    const bm = cy.by_month || [];
    if (bm.length >= 6) chartCard(ch, "Carry by calendar month", `The average carry on the days of each month of the year, over the ${Math.max(...bm.map((x) => x.years))} year${Math.max(...bm.map((x) => x.years)) === 1 ? "" : "s"} of this curve. A seasonal market (natural gas has a winter premium) has a shape; a market with none is flat. With a few years of history this describes the sample: it is not a calendar to trade.`, {
      draw: (c) => C.columns(c, { labels: bm.map((x) => MONTHS[x.month - 1]), values: bm.map((x) => x.mean), yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), colors: (v) => (v >= 0 ? css("--pos") : css("--neg")), height: 230, name: "Average carry", label: "Carry by calendar month",
        tipLabel: (i) => `${MONTHS[bm[i].month - 1]} · ${bm[i].years} years · positive on ${pct(bm[i].positive, 0)} of days` }),
      table: (c) => table(c, [{ label: "Month", get: (x) => MONTHS[x.month - 1] }, { label: "Average", num: true, get: (x) => pct(x.mean, 2) }, { label: "Median", num: true, get: (x) => pct(x.median, 2) }, { label: "Days in backwardation", num: true, get: (x) => pct(x.positive, 0) }, { label: "Years", num: true, get: (x) => x.years }], bm) });
    if (b.has_spot) chartCard(ch, "Basis to spot", "Front futures price over spot, less one. It converges to zero at expiry: the sawtooth is the roll to the next contract.", {
      draw: (c) => C.line(c, { dates: b.series.dates, series: [{ name: "Basis", color: css("--s1"), values: b.series.basis }], baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(1) + "%", tipFormat: (v) => pct(v, 2), height: 260, label: "Basis to spot" }),
      table: (c) => table(c, [{ label: "Date", get: (n) => b.series.dates[n] }, { label: "Basis", num: true, get: (n) => pct(b.series.basis[n], 3) }, { label: "Days to expiry", num: true, get: (n) => b.series.dte[n] }], b.series.dates.map((_, n) => n).filter((n) => n % 10 === 0)) });
    chartCard(ch, "Calendar basis", "Next contract against the front, as a share of the front: the term structure's own signal.", {
      draw: (c) => C.line(c, { dates: b.calendar_series.dates, series: [{ name: "Calendar basis", color: css("--s1"), values: b.calendar_series.values }], baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(1) + "%", tipFormat: (v) => pct(v, 2), height: 240, label: "Calendar basis" }),
      table: (c) => table(c, [{ label: "Date", get: (n) => b.calendar_series.dates[n] }, { label: "Calendar basis", num: true, get: (n) => pct(b.calendar_series.values[n], 3) }], b.calendar_series.dates.map((_, n) => n).filter((n) => n % 10 === 0)) });
    if (hasMomentum) chartCard(ch, "Basis momentum", "The front contract's trailing return less the second's. Positive means the front has been stronger: the shape of the curve has been tightening.", {
      draw: (c) => C.line(c, { dates: b.basis_momentum_series.dates, series: [{ name: "Basis momentum", color: css("--s1"), values: b.basis_momentum_series.values }], baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 240, label: "Basis momentum" }),
      table: (c) => table(c, [{ label: "Date", get: (n) => b.basis_momentum_series.dates[n] }, { label: "Basis momentum", num: true, get: (n) => pct(b.basis_momentum_series.values[n], 2) }], b.basis_momentum_series.dates.map((_, n) => n).filter((n) => n % 10 === 0)) });
    sources(root, d.references, [["Carry and the roll yield of futures", "carry"], ["Seasonal commodities and the shape of the curve", "seasonality"]]);
  }

  /* ---------- the price history of a product ---------- */
  function buildHistory(root, d) {
    const s = d.stats, mv = (v) => (v === null || v === undefined ? "n/a" : Q.fmt.spct(v)), pv = (v, n = 0) => (v === null || v === undefined ? "n/a" : pct(v, n));
    header(root, `${d.label} (${d.code}): price history`, `${d.exchange ? d.exchange + " · " : ""}${d.region} market · ${s.bars.toLocaleString("en-US")} daily bars · ${s.first_date} to ${s.last_date}`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Last price", num(s.last, 2), { text: `${s.last_date} · ${mv(s.return_1d)} on the day`, dir: s.return_1d >= 0 ? "up" : "down" }, "", "The newest daily close in the file");
    tile(k, "Last month", mv(s.return_1m), { text: `3 months ${mv(s.return_3m)}`, dir: s.return_1m >= 0 ? "up" : "down" }, "", "Change over the last 21 trading days; the text gives the last 63");
    tile(k, "Last year", mv(s.return_1y), { text: `this calendar year ${mv(s.return_ytd)}` }, "", "Change over the last 252 trading days");
    tile(k, "Volatility, a year", pv(s.vol_1y), { text: `1 month ${pv(s.vol_1m)} · 3 months ${pv(s.vol_3m)}` }, "", "Annualised standard deviation of daily log returns");
    tile(k, "Worst fall", pv(s.max_drawdown), { text: `trough ${s.max_drawdown_date} · now ${pv(s.drawdown, 1)} from the peak` }, "", "The largest peak-to-trough fall in the file");
    tile(k, "From the year's high", pv(s.off_high, 1), { text: `high ${num(s.high_52w, 2)} · low ${num(s.low_52w, 2)}` }, "", "The last close against the highest close of the last 252 trading days; the text gives that high and low");
    tile(k, "Best day", mv(s.best_day), { text: s.best_day_date, dir: "up" }, "", "The largest one-day gain in the file");
    tile(k, "Worst day", mv(s.worst_day), { text: s.worst_day_date, dir: "down" }, "", "The largest one-day loss in the file");
    const ch = h("div", "charts", root), idx = d.dates.map((_, n) => n), every = Math.max(1, Math.round(d.dates.length / 60)), rows = idx.filter((n) => n % every === 0 || n === idx.length - 1);
    const finite = d.close.filter((v) => v !== null && v > 0), logScale = finite.length > 0 && Math.max(...finite) / Math.min(...finite) > 2;
    chartCard(ch, "Price", `Daily close${logScale ? ", on a log scale (equal distances are equal percentage moves)" : ""}. A futures product's series is joined from successive contracts, so a sudden step can be a contract change rather than the market.`, { wide: true,
      draw: (b) => C.line(b, { dates: d.dates, series: [{ name: d.code, color: css("--s1"), values: d.close }], log: logScale, yFormat: (v) => num(v, v >= 100 ? 0 : 2), tipFormat: (v) => num(v, 2), height: 300, label: `${d.code} price`, endLabels: false }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, { label: "Close", num: true, get: (n) => num(d.close[n], 3) }], rows) });
    chartCard(ch, "Drawdown from the previous peak", "How far below its high the price has been.", {
      draw: (b) => C.line(b, { dates: d.dates, series: [{ name: "Drawdown", color: css("--s1"), values: d.drawdown }], area: true, areaBase: 0, baseline: 0, ceil: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 230, label: "Drawdown" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, { label: "Drawdown", num: true, get: (n) => pv(d.drawdown[n], 1) }], rows) });
    chartCard(ch, "Realised volatility", "Annualised standard deviation of daily returns over the last 21 and 63 trading days.", {
      draw: (b) => C.line(b, { dates: d.dates, series: [{ name: "21 days", color: css("--s1"), values: d.vol_21, width: 1.2 }, { name: "63 days", color: css("--s2"), values: d.vol_63 }], include: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 230, label: "Realised volatility", endLabels: false }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.dates[n] }, { label: "21 days", num: true, get: (n) => pv(d.vol_21[n], 1) }, { label: "63 days", num: true, get: (n) => pv(d.vol_63[n], 1) }], rows) });
    if (d.seasonal.length) chartCard(ch, "Average return by calendar month", `Each month's average return over the ${Math.max(...d.seasonal.map((m) => m.n))} years or so in the file. With so few observations a month tells you little: treat it as a description, not a signal.`, {
      draw: (b) => C.columns(b, { labels: d.seasonal.map((m) => MONTHS[m.month - 1]), values: d.seasonal.map((m) => m.mean), yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 230, name: "Average return", label: "Return by calendar month",
        tipLabel: (i) => `${MONTHS[d.seasonal[i].month - 1]}: up in ${pct(d.seasonal[i].hit_rate, 0)} of ${d.seasonal[i].n} years` }),
      table: (b) => table(b, [{ label: "Month", get: (m) => MONTHS[m.month - 1] }, { label: "Average", num: true, get: (m) => pct(m.mean, 2) }, { label: "Median", num: true, get: (m) => pct(m.median, 2) }, { label: "Up in", num: true, get: (m) => pct(m.hit_rate, 0) }, { label: "Years", num: true, get: (m) => m.n }], d.seasonal) });
    const moves = h("div", "card wide", ch); h("h3", "", moves, "The eight largest one-day moves");
    h("p", "note", moves, "A step this size on a futures product is often the day the series switched from one contract to the next, not a market event: compare the dates with the contract expiries before reading anything into them.");
    table(moves, [{ label: "Date", get: (m) => m.date }, { label: "Move", num: true, get: (m) => Q.fmt.spct(m.change) }, { label: "From", num: true, get: (m) => num(m.from, 3) }, { label: "To", num: true, get: (m) => num(m.to, 3) }], d.moves);
    W.hint(root, "This is a price history, not a curve: there is no roll yield, carry or basis in it, and no open interest. For those, download the curve of a root from Yahoo Finance instead.");
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
    const real = !!d.coverage;
    header(root, `${d.label} on ${real ? "real" : "simulated"} futures`, `${d.text} · roots ${d.roots.join(", ")} · ${d.start} to ${d.end}`);
    W.provenance(root, d);
    W.pipelineResult(root, d);
    const cr = d.correlation, info = strategyInfo(d.strategy);
    if (cr) chartCard(h("div", "charts", root), "How alike the roots are", "The correlation of the roots' daily returns over the backtest, with each root's own annualised volatility beside its name. A book of roots that move together has fewer independent bets than it has positions.", { wide: cr.assets.length > 4,
      draw: (b) => C.matrix(b, { rows: cr.assets.map((a, i) => `${a} · ${cr.volatility[i] === null ? "n/a" : (cr.volatility[i] * 100).toFixed(0) + "%"}`), cols: cr.assets, values: cr.values, diverging: true, mid: 0, min: -1, max: 1, format: (v) => v.toFixed(2), name: "Correlation", cellHeight: 26, label: "Correlation of the roots" }),
      table: (b) => table(b, [{ label: "Root", get: (n) => cr.assets[n] }, { label: "Volatility", num: true, get: (n) => (cr.volatility[n] === null ? "" : pct(cr.volatility[n], 0)) }, ...cr.assets.map((a, j) => ({ label: a, num: true, get: (n) => (cr.values[n][j] === null ? "" : num(cr.values[n][j], 2)) }))], cr.assets.map((_, n) => n)) });
    if (info && info.references && info.references.length) { const c = h("div", "card", root); h("h3", "", c, "Where the rule comes from"); Q.refsLine(c, info.references); }
    if (real) {
      const card = h("div", "card wide", root); h("h3", "", card, "What each root contributes");
      table(card, [{ label: "Root", get: (x) => x.root }, { label: "Front-month days", num: true, get: (x) => x.front_days }, { label: "First", get: (x) => x.first }, { label: "Last", get: (x) => x.last },
        { label: "Days in the file", num: true, get: (x) => x.file_days }, { label: "Contracts", num: true, get: (x) => x.contracts }], Object.entries(d.coverage).map(([k, v]) => ({ root: k, ...v })));
      W.hint(root, "A backtest on real prices, not a forecast: the excess return of holding the nearest contract of each root and rolling it, from the settlements in the files. Margin, funding, the spread paid at the roll and the capacity of each market are not modelled.");
    } else W.hint(root, "The futures are a model: the result checks that carry, roll and costs are booked correctly, not whether the strategy earns money in the world. Returns are the excess return of holding the front contract and rolling it; margin and funding are not modelled.");
  }

  /* ---------- the signals board ---------- */
  function buildSignals(root, d) {
    const rules = d.signals.filter((r) => r.strength), failed = d.signals.filter((r) => r.error), roots = d.assets;
    header(root, "What the rules say today", `${roots.length} root${roots.length === 1 ? "" : "s"} · ${rules.length} of ${d.signals.length} rules · as of ${d.as_of}`);
    W.provenance(root, d);
    const ag = d.agreement, rows = roots.map((a) => ({ a, ...ag[a] })), lean = (r) => (r.long >= r.short ? "long" : "short"), most = rows.slice().sort((x, y) => Math.max(y.long, y.short) - Math.max(x.long, x.short))[0];
    const k = h("div", "earnrow", root);
    if (most) tile(k, "Most agreement", `${most.a} ${lean(most)}`, { text: `${Math.max(most.long, most.short)} of ${rules.length} rules`, dir: lean(most) === "long" ? "up" : "down" }, "", "The root on which most rules point the same way: a strength beyond a quarter of the rule's usual size counts as a view");
    const longs = rows.reduce((t, r) => t + r.long, 0), shorts = rows.reduce((t, r) => t + r.short, 0);
    tile(k, "Views taken", `${longs + shorts} of ${rules.length * roots.length}`, { text: `${longs} long · ${shorts} short` }, "", "Rule-and-root pairs with a strength beyond a quarter of the rule's usual size");
    tile(k, "Rules computed", `${rules.length} of ${d.signals.length}`, { text: failed.length ? `${failed.length} could not be computed` : "all of them" });
    const ch = h("div", "charts", root);
    if (rules.length && roots.length) chartCard(ch, "Every rule on every root", `Each cell is the rule's score on the root today over the average size of that rule's own score in the last ${d.window} days: blue is long, red is short, 1 is a typical position and the scale stops at 3. An outlined cell is a full-sized view. A row that is all one colour is a rule that sees one trade in every market; a column that is all one colour is a root on which the rules agree.`, { wide: true,
      draw: (b) => C.matrix(b, { rows: rules.map((r) => r.label), cols: roots, values: rules.map((r) => roots.map((a) => r.strength[a])), diverging: true, mid: 0, min: -3, max: 3, format: (v) => v.toFixed(1), name: "Strength", cellHeight: 26, labelWidth: 190,
        mark: (i, j) => Math.abs(rules[i].strength[roots[j]]) >= 1, tipRows: (i, j) => [["Score", rules[i].score[roots[j]] === null ? "n/a" : num(rules[i].score[roots[j]], 3)], ["As of", rules[i].date]], label: "Strength of every rule on every root" }),
      table: (b) => table(b, [{ label: "Rule", get: (r) => r.label }, ...roots.map((a) => ({ label: a, num: true, get: (r) => (r.score[a] === null ? "" : num(r.score[a], 3) + " (" + num(r.strength[a], 1) + ")") }))], rules) });
    const card = h("div", "card wide", ch); h("h3", "", card, "Where the rules agree");
    table(card, [{ label: "Root", get: (r) => r.a }, { label: "Long", num: true, get: (r) => r.long }, { label: "Short", num: true, get: (r) => r.short }, { label: "No view", num: true, get: (r) => r.flat }, { label: "Lean", get: (r) => (r.long === r.short ? "split" : lean(r)) }], rows);
    h("p", "note", card, "A rule has a view on a root when its strength is beyond a quarter of its usual size. Agreement is not accuracy: rules of one family (the trend rules) are close to one rule counted six times, so look at the rows as well as the counts.");
    if (failed.length) { const f = h("div", "card wide", root); h("h3", "", f, "Rules that could not be computed"); table(f, [{ label: "Rule", get: (r) => r.label }, { label: "Why", get: (r) => r.error }], failed); }
    if ((d.references || []).length) { const c = h("div", "card", root); h("h3", "", c, "Where the rules come from"); Q.refsLine(c, d.references); }
    W.hint(root, "These are the platform's own strategies read on the last day, on the excess return of holding the nearest contract of each root; the backtest mode runs any of them over history, with costs.");
  }

  Q.registerWorkspace("futures", F);
})();
