/* The Crypto workspace: an overview of a coin (trend, funding, momentum, volatility, liquidity, on-chain), the spot and perpetual prices and the basis between them, funding by venue and coin, the spread
   between exchanges net of what it costs to trade it, the depth of the books and strategies run on the Deribit data through the platform's pipeline. It talks to /api/crypto/*.
   Deribit's BTC and ETH data is real. The other venues' funding and quotes, the order books and every coin but BTC and ETH are SIMULATED unless real data has been downloaded for them (the Free data card:
   OKX, Coinbase, Kraken, Bitstamp and Deribit); every payload says which, and this page prints it. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, money = Q.money, smoney = Q.smoney, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable;
  const GUIDE = "technique-crypto-funding-basis-and-venues";
  const OVERVIEW_MODES = new Set(["overview", "basis", "liquidity"]);
  const venueColor = W.palette();
  const cap = (s) => String(s).charAt(0).toUpperCase() + String(s).slice(1);
  const compact = (v) => (Math.abs(v) >= 1e9 ? (v / 1e9).toFixed(1) + "B" : Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(1) + "M" : Math.abs(v) >= 1e3 ? (v / 1e3).toFixed(0) + "k" : String(Math.round(v)));

  const X = {
    id: "crypto", root: "#crypto", view: "#view-crypto", title: "Crypto", api: 12, mode: "overview", busy: "", results: {}, errors: {}, cache: new Map(), guard: null, go: null, progress: null,
    about: "A coin at a glance, funding and the basis between spot and the perpetual, spreads between exchanges after fees, the depth of the order books and strategies on the Deribit data. Deribit's BTC and ETH are data; download real prices, books and funding for the rest (free, below) or they are simulated and labelled.",
    modes: [["overview", "Overview"], ["funding", "Funding"], ["basis", "Basis and carry"], ["cross", "Across exchanges"], ["liquidity", "Liquidity"], ["backtest", "Backtest"], ["signals", "Signals"]],
    runLabel: { overview: "Refresh", funding: "Refresh", basis: "Refresh", cross: "Refresh", liquidity: "Refresh", backtest: "Run the backtest", signals: "Read the signals" },
    blurb: { overview: "Trend, funding, momentum, volatility, liquidity and on-chain supply on six cards; the price against its 200-day average (the Mayer multiple), the cone of its volatility, its drawdowns and how it moves against BTC or ETH; spot and the perpetual side by side.",
      funding: "What each venue charges the longs on the perpetual, by coin and venue, with its z-score and percentile against its own history: where a short is paid most.",
      basis: "How far the perpetual trades from spot, and what holding spot long against the perpetual short (the cash-and-carry trade) would have earned.",
      cross: "Every pair of exchanges: the gap between a buy price and a sell price, less both fees and the slippage. Only a gap that survives that is an opportunity.",
      liquidity: "Spread, depth and the cost of trading $100 thousand, $1 million and $10 million at each venue: from the real order books once you have downloaded them, else from a parametric book.",
      backtest: "Trend of many kinds, the Mayer multiple, volatility management, funding carry, basis and stablecoin flow on Deribit BTC and ETH, with the strategy's own settings, fees and slippage, through the platform's pipeline.",
      signals: "What every rule says about BTC and ETH today (trend of several kinds, the Mayer multiple, volatility management, funding carry, basis, stablecoin flow), as a strength against its own usual size, and where the rules agree." },
    form: { symbol: "BTC", days: "365", fee: "5", slippage: "1", notional: "1000000", symbols: new Set(["BTC", "ETH", "SOL"]), fundDays: "120", coin: "", strategy: "spot_momentum", params: {}, assets: new Set(["BTC", "ETH"]), frequency: "weekly", targetVol: "", fees: "5", slip: "2", regime: "" },
    catalog: null,
  };
  W.catalog().then((c) => { X.catalog = c; if (!Q.$(X.view).hidden) Q.renderLab(X); }, () => {});
  const key = (name, q) => name + ":" + JSON.stringify(q);
  const strategyInfo = (id) => (X.catalog ? (X.catalog.crypto_strategies || []).find((x) => x.id === id) : null) || { available: true, params: [], references: [] };
  const symbols = () => (X.catalog ? X.catalog.crypto_symbols : []);
  const SOURCE_NAME = { platform: "Deribit data", user_file: "your file", downloaded: "downloaded", simulated: "simulated" };
  const setCatalog = (c) => { X.catalog = c; if (!c.crypto_symbols.some((s) => s.id === X.form.symbol)) X.form.symbol = "BTC"; };
  const WHAT = { history: "Daily history (spot, perpetual, funding)", venues: "Real quotes and order books, now", funding: "OKX funding history" };
  const DAYS = [["260", "260 days (the least the tab needs)"], ["400", "400 days"], ["730", "2 years"], ["1000", "1000 days"]];
  let itick = null;                                                                          // the server's iTick status, kept by the card for its time estimate
  /* The Free data card: a coin's history and funding from OKX (or spot alone from Kraken, Bitstamp and Coinbase), and its quotes and order books on up to six venues at this moment; or, from iTick, a coin's
     daily history against USDT (spot only). */
  const FREE = {
    kind: "crypto", title: "Real crypto data (free)",
    intro: "Download real prices for coins other than BTC and ETH, the real quotes and order books of a coin on Coinbase, Kraken, Bitstamp, OKX and Deribit, and OKX's funding history. No account and no key. iTick (needs your key) can supply a coin's daily history instead, spot only. What is downloaded replaces the simulation for that coin, venue or book, and is labelled.",
    defaults: () => ({ source: "exchanges", chosen: new Set(["SOL"]), extra: "", what: new Set(["history", "venues", "funding"]), days: "400" }),
    controls: (box, f, ks, changed, redraw) => {
      itick = ks.itick_status;
      const seg = W.seg(box, [["exchanges", "Exchanges: OKX, Coinbase, Kraken …"], ["itick", "iTick: daily history"]], f.source, (v) => { f.source = v; redraw(); });
      seg.setAttribute("role", "group"); seg.setAttribute("aria-label", "data source"); seg.dataset.role = "source";
      h("span", "hint", box, "Pick coins:");
      W.picks(box, ks.coins.map((c) => [c, c]), f.chosen, changed);
      if (f.source === "itick") {
        Q.labField(box, { key: "extra", label: "Other coins (separated by spaces)", kind: "text", placeholder: "DOT ATOM", help: "Any coin iTick quotes against USDT (about " + ks.itick.coins + " of them)" }, f, changed);
        W.itickBox(box, ks.itick_status, "crypto");
        h("p", "hint", box, "iTick has the daily closes of a coin against USDT: spot only, no perpetual, funding or order books (the exchanges have those). Which exchange its prices come from is not documented. BTC and ETH keep the platform's longer Deribit history.");
        Q.labField(box, { key: "days", label: "Days of history", kind: "choice", options: DAYS, help: "Each 1,000 daily bars cost one call of your plan" }, f, changed);
        return;
      }
      Q.labField(box, { key: "extra", label: "Other coins (separated by spaces)", kind: "text", placeholder: "DOT ATOM", help: "A coin symbol listed on OKX, Kraken, Bitstamp or Coinbase" }, f, changed);
      Q.checkList(box, "Download:", ks.what.map((w) => [w.id, WHAT[w.id] || w.text]), f.what, changed);
      Q.labField(box, { key: "days", label: "Days of history", kind: "choice", options: DAYS, help: "OKX keeps about three months of funding, whatever you ask" }, f, changed);
      h("p", "hint", box, "BTC and ETH keep the platform's longer Deribit history: for them the card adds the venues and the funding.");
    },
    request: (f) => {
      const items = [...f.chosen, ...String(f.extra).toUpperCase().split(/[\s,;]+/).filter(Boolean)];
      if (!items.length) throw new Error("Pick a coin, or type one");
      if (f.source === "itick") return { kind: "crypto", source: "itick", items, what: ["history"], days: Number(f.days) };
      if (!f.what.size) throw new Error("Tick what to download");
      return { kind: "crypto", items, what: ["history", "venues", "funding"].filter((w) => f.what.has(w)), days: Number(f.days) };
    },
    estimate: (b) => {
      if (b.source !== "itick") return b.items.length * (b.what.length * 5);
      const info = itick || {}, cpm = info.calls_per_minute || 5, page = info.page_size || 1000, free = info.calls_available_now === undefined ? cpm : info.calls_available_now;
      const calls = b.items.length * Math.max(1, Math.ceil(b.days / page));
      return 2 * calls + Math.max(0, calls - free) * (60 / cpm);
    },
    describe: (rec) => rec.item === "history" ? [`${rec.name} · daily history`, `${rec.source} · ${rec.rows} days to ${rec.as_of}`] : rec.item === "venues" ? [`${rec.coin} · order books`, `${(rec.venues || []).length} venues: ${(rec.venues || []).join(", ")}`]
      : [`${rec.coin} · OKX funding`, `${rec.rows} days to ${rec.as_of}`],
    setCatalog,
    pick: (res) => { const first = res.results.find((r) => r.ok && !r.skipped && r.what !== "funding") || res.results.find((r) => r.ok); if (first && X.catalog) X.form.symbol = first.item; },
  };


  /* ---------- the form ---------- */
  X.check = () => {
    if (!X.catalog) return "Loading the list of coins…";
    if (X.mode === "funding" && (!X.form.symbols.size || X.form.symbols.size > 6)) return "Choose between one and six coins.";
    if (X.mode === "backtest" && !X.form.assets.size) return "Choose at least one asset.";
    if (X.mode === "backtest" && !strategyInfo(X.form.strategy).available) return "The platform has no data for this strategy.";
    return "";
  };
  X.sections = (panel) => {
    const f = X.form, after = () => Q.syncLab(X), rerun = () => X.run(true), sec = (title, open = true) => { const d = h("details", "", panel); d.open = open; h("summary", "", d, title); return h("div", "body", d); };
    const m = X.mode, cat = X.catalog;
    if (!cat) { h("p", "hint", sec("1. Coin"), "Loading the list of coins…"); return; }
    if (m === "funding") {
      const b = sec("1. Coins"), box = h("div", "checks", b); h("span", "hint", box, "Up to six:");
      for (const s of symbols()) {
        const label = h("label", "f inline", box), c = h("input", "", label); c.type = "checkbox"; c.checked = f.symbols.has(s.id); label.appendChild(document.createTextNode(`${s.id} (${SOURCE_NAME[s.source] || s.source})`));
        c.addEventListener("change", () => { if (c.checked) f.symbols.add(s.id); else f.symbols.delete(s.id); after(); });
      }
      Q.labField(b, { key: "fundDays", label: "Days of history to chart", kind: "number", step: 10, min: 30, max: 720 }, f, after);
      h("p", "hint", b, "Deribit's funding (BTC and ETH) is data, and so is OKX's once downloaded; every other venue is simulated from the coin's own series with offsets and noise, and so is every coin not downloaded. The rows say which.");
    } else if (m !== "backtest" && m !== "signals") {
      const b = sec("1. Coin");
      Q.labField(b, { key: "symbol", label: "Coin", kind: "choice", options: symbols().map((s) => [s.id, `${s.id} (${SOURCE_NAME[s.source] || s.source})`]) }, f, rerun);
      if (OVERVIEW_MODES.has(m)) { Q.labField(b, { key: "days", label: "Days of history to chart", kind: "number", step: 30, min: 60, max: 2000 }, f, after); }
      if (m === "basis") Q.labField(b, { key: "fee", label: "Fee for each leg, one way (bps)", kind: "number", step: 0.5, min: 0, max: 100, help: "Used for the break-even: how many days of carry pay for opening and closing both legs" }, f, after);
      if (m === "cross") { Q.labField(b, { key: "slippage", label: "Slippage per leg (bps)", kind: "number", step: 0.5, min: 0, max: 50 }, f, after); Q.labField(b, { key: "notional", label: "Trade size ($)", kind: "number", min: 1000 }, f, after); }
    }
    if (m !== "backtest" && m !== "signals") W.dataCard(panel, X, FREE);
    if (m === "signals") { const b = sec("1. What it reads"); h("p", "hint", b, "Every rule is read on the platform's Deribit data for BTC and ETH (the funding and basis rules on each coin's long-spot, short-perpetual index): there is nothing to choose, press Read the signals."); }
    if (m === "backtest") {
      const b = sec("1. The strategy");
      Q.labField(b, { key: "strategy", label: "Strategy", kind: "choice", options: cat.crypto_strategies.map((x) => [x.id, x.label + (x.available ? "" : " (needs data we lack)"), x.group]) }, f, () => Q.renderLab(X));
      const info = strategyInfo(f.strategy); if (info.text) h("p", "hint", b, info.text);
      Q.refsLine(b, info.references);
      if (!info.available) { const w = h("p", "hint", b, "The platform holds no exchange net-flow data: use the stablecoin-flow strategy (the on-chain signal that is available) or load your own data."); w.style.color = css("--warn"); }
      if (info.params.length) {
        h("h4", "subhead", b, "Settings of this strategy");
        const grid = h("div", "params", b), mine = f.params[info.id] = f.params[info.id] || {};
        for (const q of info.params) {
          const cur = q.name in mine ? mine[q.name] : q.default;
          grid.appendChild(Q.fieldFor(q, cur, (v) => { if (JSON.stringify(v) === JSON.stringify(q.default)) delete mine[q.name]; else mine[q.name] = v; after(); }, { wide: q.kind === "json" }));
        }
        h("p", "hint", b, "Only what you change is sent; the rest is the strategy's own default.");
      }
      const a = sec("2. Assets"), box = h("div", "checks", a); h("span", "hint", box, "The platform's data covers:");
      for (const s of ["BTC", "ETH"]) { const label = h("label", "f inline", box), c = h("input", "", label); c.type = "checkbox"; c.checked = f.assets.has(s); label.appendChild(document.createTextNode(s)); c.addEventListener("change", () => { if (c.checked) f.assets.add(s); else f.assets.delete(s); after(); }); }
      h("p", "hint", a, "The carry and basis strategies trade the long-spot, short-perpetual index of each coin.");
      const c = sec("3. Trading");
      Q.labField(c, { key: "frequency", label: "Rebalance", kind: "choice", options: [["daily", "Daily"], ["weekly", "Weekly"], ["monthly", "Monthly"]] }, f, after);
      Q.labField(c, { key: "regime", label: "Regime filter", kind: "choice", options: [["", "The strategy's own"], ["vol_state", "Volatility state"], ["hmm", "Hidden Markov model"], ["gmm", "Gaussian mixture"]], help: "Sets the volatility target by market regime (lower in the stressed one)" }, f, after);
      for (const s of [{ key: "targetVol", label: "Volatility target (% a year, blank = the platform's)", kind: "number", step: 1, min: 1, max: 200 },
        { key: "fees", label: "Exchange fees, one way (bps)", kind: "number", step: 0.5, min: 0, max: 200 }, { key: "slip", label: "Slippage, one way (bps)", kind: "number", step: 0.5, min: 0, max: 200 }]) Q.labField(c, s, f, after);
      h("p", "hint", c, "Weekday marks and 252-day annualisation, like every other backtest here; funding is earned inside the carry index.");
    }
  };

  /* ---------- running ---------- */
  X.run = (fromForm) => {
    const m = X.mode, f = X.form;
    if (!fromForm) X.cache.clear();
    if (OVERVIEW_MODES.has(m)) return W.request(X, () => { const q = { symbol: f.symbol, days: Number(f.days), fee_bps: Number(f.fee) }; return [{ modes: ["overview", "basis", "liquidity"], path: "/api/crypto/overview", query: q, cacheKey: key("overview", q) }]; }, "Reading the coin…");
    if (m === "funding") return W.request(X, () => { const q = { symbols: [...f.symbols], days: Number(f.fundDays) }; return [{ modes: ["funding"], path: "/api/crypto/funding", query: q, cacheKey: key("funding", q) }]; }, "Reading funding…");
    if (m === "cross") return W.request(X, () => { const q = { symbol: f.symbol, slippage_bps: Number(f.slippage), notional: Number(f.notional) }; return [{ modes: ["cross"], path: "/api/crypto/cross-exchange", query: q, cacheKey: key("cross", q) }]; }, "Comparing venues…");
    if (m === "signals") return W.request(X, () => [{ modes: ["signals"], job: true, path: "/api/crypto/signals", body: {} }], "Reading the signals…");
    return W.request(X, () => [{ modes: ["backtest"], job: true, path: "/api/crypto/strategy/backtest", body: { strategy: f.strategy, assets: [...f.assets], frequency: f.frequency, params: f.params[f.strategy] || {},
      target_vol: f.targetVol === "" ? undefined : Number(f.targetVol) / 100, fees_bps: Number(f.fees), slippage_bps: Number(f.slip), regime: f.regime || undefined } }], "Running the backtest…");
  };
  X.auto = () => { if (X.mode !== "backtest" && X.mode !== "signals" && X.catalog) X.run(true); };
  X.empty = (root) => (X.mode === "signals"
    ? Q.labEmpty(root, X, "Read what the rules say today", ["Trend of several kinds, the Mayer multiple, volatility management, funding carry, basis and stablecoin flow on BTC and ETH, each as a strength against its own usual size, and where they agree."], X.runLabel.signals)
    : Q.labEmpty(root, X, "Backtest a crypto strategy", ["Trend of many kinds, the Mayer multiple, volatility management, funding carry, basis or stablecoin flow on Deribit BTC and ETH, with the strategy's own settings, fees and slippage, through the platform's pipeline."], X.runLabel.backtest));
  X.draw = () => Q.drawLab(X);
  X.build = (root, d, body) => ({ overview: buildOverview, funding: buildFunding, basis: buildBasis, cross: buildCross, liquidity: buildLiquidity, backtest: buildBacktest, signals: buildSignals })[X.mode](root, d, body);
  const header = (root, title, sub) => Q.labHead(root, title, sub, GUIDE);
  const STATE_LABEL = { uptrend: "Uptrend", downtrend: "Downtrend", sideways: "Sideways", balanced: "Balanced", crowded_long: "Crowded long", crowded_short: "Crowded short", positive: "Positive", negative: "Negative", flat: "Flat",
    quiet: "Quiet", normal: "Normal", elevated: "Elevated", stressed: "Stressed", deep: "Deep", thin: "Thin", adequate: "Adequate", inflow: "Inflow", outflow: "Outflow", unavailable: "Not available" };

  /* ---------- overview ---------- */
  function buildOverview(root, d) {
    const ov = d.overview;
    header(root, `${d.symbol} overview`, `${money(ov.price, 2)} · as of ${ov.as_of}`);
    W.provenance(root, { ...d, warnings: ov.risk_flags });
    const k = h("div", "earnrow", root);
    for (const c of ov.cards) tile(k, c.title, STATE_LABEL[c.state] || cap(String(c.state).replace(/_/g, " ")), { text: c.detail + (c.source === "simulated" ? " · simulated" : "") }, "", c.source === "simulated" ? "A model output, not data" : c.source === "downloaded" ? "From data downloaded for you" : "From the platform's data");
    const r = h("div", "earnrow", root);
    tile(r, "Return, 24 hours", Q.fmt.spct(ov.return_24h || 0), null); tile(r, "Return, 7 days", Q.fmt.spct(ov.return_7d || 0), null); tile(r, "Return, 30 days", Q.fmt.spct(ov.return_30d || 0), null); tile(r, "Return, 90 days", Q.fmt.spct(ov.return_90d || 0), null);
    tile(r, "Volatility, 21 days", pct(ov.realized_vol_21d, 0), { text: `63 days ${pct(ov.realized_vol_63d, 0)}` });
    const sp = d.spot_perp;
    if (!sp) { W.hint(root, "No perpetual prices for this coin: the spot and basis charts need them."); technical(root, d); return; }
    const ch = h("div", "charts", root);
    chartCard(ch, "Spot and perpetual", "Spot (the index price) and the perpetual swap's price. They are within a few basis points of each other, so the next chart shows the gap.", { wide: true,
      draw: (b) => C.line(b, { dates: sp.dates, series: [{ name: "Spot", color: css("--s1"), values: sp.spot }, { name: "Perpetual", color: css("--s2"), values: sp.perp, width: 1.5 }], log: true, yFormat: (v) => compact(v), tipFormat: (v) => money(v, 2), height: 280, label: "Spot and perpetual", endLabels: false }),
      table: (b) => table(b, [{ label: "Date", get: (n) => sp.dates[n] }, { label: "Spot", num: true, get: (n) => money(sp.spot[n], 2) }, { label: "Perpetual", num: true, get: (n) => money(sp.perp[n], 2) }, { label: "Basis", num: true, get: (n) => pct(sp.basis[n], 3) }], sp.dates.map((_, n) => n).filter((n) => n % 7 === 0)) });
    chartCard(ch, "Basis: perpetual over spot", "Positive when the perpetual trades above spot (longs are paying to hold it).", {
      draw: (b) => C.line(b, { dates: sp.dates, series: [{ name: "Basis", color: css("--s1"), values: sp.basis }], area: true, areaBase: 0, baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(2) + "%", tipFormat: (v) => pct(v, 3), height: 240, label: "Basis" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => sp.dates[n] }, { label: "Basis", num: true, get: (n) => pct(sp.basis[n], 3) }], sp.dates.map((_, n) => n).filter((n) => n % 7 === 0)) });
    chartCard(ch, "Funding, annualised", "What the longs paid the shorts (or received, below zero), as a yearly rate.", {
      draw: (b) => C.line(b, { dates: sp.dates, series: [{ name: "Funding", color: css("--s1"), values: sp.funding_annualised }], baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 240, label: "Funding" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => sp.dates[n] }, { label: "Funding", num: true, get: (n) => (sp.funding_annualised[n] === null ? "" : pct(sp.funding_annualised[n], 2)) }], sp.dates.map((_, n) => n).filter((n) => n % 7 === 0)) });
    const st = sp.stats, c = h("div", "card", root); h("h3", "", c, "How close the perpetual stays to spot");
    W.kv(c, "Basis", [["Average", pct(st.basis_mean, 3)], ["Standard deviation", pct(st.basis_std, 3)], ["Latest", pct(st.basis_last, 3)], ["z-score of the latest", num(st.basis_zscore, 2)], ["Correlation of daily returns", num(st.return_correlation, 4)], ["Tracking error, annualised", pct(st.tracking_error_annual, 2)]]);
    technical(root, d);
  }

  /* The price against its own history: the multiple of its 200-day average, the cone of its volatility, its drawdowns, and how it moves against another coin. */
  function technical(root, d) {
    const t = d.technical; if (!t) return;
    const ch = h("div", "charts", root), la = t.long_average, vc = t.volatility_cone, dd = t.drawdown, rel = t.relative, sym = d.symbol;
    if (la) chartCard(ch, `${sym} against its ${la.window}-day average`, `The multiple of the price over its ${la.window}-day average (the Mayer multiple, Trace Mayer's measure for Bitcoin). Above ${la.cap} has been called overheated and below 1 cheap: those thresholds are a practitioner's, not tested in a paper. Today ${num(la.now, 2)}, higher than ${pct(la.percentile, 0)} of the days in the history; above ${la.cap} on ${pct(la.share_above_cap, 1)} of them.`, { wide: true,
      draw: (b) => C.line(b, { dates: la.dates, series: [{ name: "Multiple", color: css("--s1"), values: la.multiple }, { name: `Overheated (${la.cap})`, color: css("--neg"), dash: "5 4", width: 1.25, values: la.multiple.map(() => la.cap) }, { name: "Average (1)", color: css("--muted"), dash: "5 4", width: 1.25, values: la.multiple.map(() => 1) }],
        yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), height: 270, label: `${sym} multiple of its long average`, endLabels: false }),
      table: (b) => table(b, [{ label: "Date", get: (n) => la.dates[n] }, { label: "Price", num: true, get: (n) => money(la.price[n], 2) }, { label: `${la.window}-day average`, num: true, get: (n) => money(la.average[n], 2) }, { label: "Multiple", num: true, get: (n) => num(la.multiple[n], 2) }], la.dates.map((_, n) => n).filter((n) => n % 7 === 0)) });
    if (vc && vc.rows.length) chartCard(ch, "The cone of volatility", `Realised volatility over the last 7 to 90 days (the dot) against every window of that length in the ${vc.history_days.toLocaleString("en-US")} days of history (the bands: 10th to 90th and 25th to 75th percentiles, and the median; the Table view adds the lowest and the highest, which the very short windows push far out). A short window above a long one is a spike; the reverse is a calm after a storm.`, {
      draw: (b) => C.line(b, { xs: vc.rows.map((r) => r.days), xWhole: true, xLabel: "Window (days)", series: [{ name: "Median", color: css("--muted"), dash: "5 4", width: 1.25, values: vc.rows.map((r) => r.median) }, { name: "Now", color: css("--s1"), values: vc.rows.map((r) => r.now), dots: true, dotSize: 4, width: 2 }],
        bands: [{ lo: vc.rows.map((r) => r.p10), hi: vc.rows.map((r) => r.p90), color: css("--s1"), opacity: 0.12, name: "10th to 90th percentile" }, { lo: vc.rows.map((r) => r.p25), hi: vc.rows.map((r) => r.p75), color: css("--s1"), opacity: 0.2, name: "25th to 75th" }],
        include: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 0), xFormat: (v) => v + "d", height: 270, label: "Volatility cone", endLabels: false }),
      table: (b) => table(b, [{ label: "Window", get: (r) => r.days + " days" }, { label: "Now", num: true, get: (r) => pct(r.now, 0) }, { label: "Percentile", num: true, get: (r) => pct(r.percentile, 0) }, { label: "Lowest", num: true, get: (r) => pct(r.min, 0) }, { label: "Median", num: true, get: (r) => pct(r.median, 0) }, { label: "Highest", num: true, get: (r) => pct(r.max, 0) }], vc.rows) });
    if (dd) chartCard(ch, "Drawdown from the previous high", `How far below its previous high ${sym} has been. The Table view lists the five deepest falls, from the peak to the day it was regained.`, {
      draw: (b) => C.line(b, { dates: dd.dates, series: [{ name: "Drawdown", color: css("--s1"), values: dd.depth }], area: true, areaBase: 0, baseline: 0, ceil: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 250, label: "Drawdown" }),
      table: (b) => table(b, [{ label: "Peak", get: (e) => e.peak }, { label: "Low", get: (e) => e.trough }, { label: "Regained", get: (e) => e.recovered || "not yet" }, { label: "Fall", num: true, get: (e) => pct(e.depth, 0) }, { label: "Days down to the low", num: true, get: (e) => e.days_to_trough }, { label: "Days in all", num: true, get: (e) => e.days }], dd.episodes) });
    if (rel) {
      chartCard(ch, `${rel.pair} against its own ${rel.window} days`, `How many standard deviations the log of ${rel.pair} is from its ${rel.window}-day average: beyond 2 in either direction the pair is stretched (${sym} is rich against the other beyond +2, cheap beyond -2). Today ${rel.z_now === null ? "n/a" : num(rel.z_now, 1)}. A stretched pair can stay stretched: this is how far, not a call.`, {
        draw: (b) => C.line(b, { dates: rel.dates, series: [{ name: "z-score", color: css("--s1"), values: rel.z }], baseline: 0, include: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), height: 240, label: `z-score of ${rel.pair}`, endLabels: false }),
        table: (b) => table(b, [{ label: "Date", get: (n) => rel.dates[n] }, { label: rel.pair, num: true, get: (n) => num(rel.ratio[n], 5) }, { label: "z-score", num: true, get: (n) => (rel.z[n] === null ? "" : num(rel.z[n], 2)) }], rel.dates.map((_, n) => n).filter((n) => n % 7 === 0)) });
      chartCard(ch, `${rel.pair}: rolling correlation`, `The correlation of the two coins' daily returns over the last ${rel.corr_window} days. Today ${rel.correlation_now === null ? "n/a" : num(rel.correlation_now, 2)}; the average over the history ${rel.correlation_mean === null ? "n/a" : num(rel.correlation_mean, 2)}. Two coins that move together give little diversification, and the correlation tends to rise in a sell-off.`, {
        draw: (b) => C.line(b, { dates: rel.dates, series: [{ name: "Correlation", color: css("--s1"), values: rel.correlation }, { name: "Average", color: css("--muted"), dash: "5 4", width: 1.25, values: rel.correlation.map((v) => (v === null || rel.correlation_mean === null ? null : rel.correlation_mean)) }], include: 0, ceil: 1, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), height: 240, label: `Correlation of ${rel.pair}`, endLabels: false }),
        table: (b) => table(b, [{ label: "Date", get: (n) => rel.dates[n] }, { label: "Correlation", num: true, get: (n) => (rel.correlation[n] === null ? "" : num(rel.correlation[n], 2)) }], rel.dates.map((_, n) => n).filter((n) => n % 7 === 0)) });
    }
    sources(root, d.references, [["Trend and the long moving average", "trend"], ["Volatility", "volatility"], ["How coins relate to each other and to the price elsewhere", "relative"], ["Funding, basis and carry", "carry"]]);
  }
  /* The papers and books behind a view, folded away under a heading: ``topics`` is [[heading, key]] into the ``refs`` of the payload. */
  function sources(root, refs, topics) {
    const shown = topics.filter(([, k]) => refs && refs[k] && refs[k].length);
    if (!shown.length) return;
    const dd = h("details", "how", root); h("summary", "", dd, "Where these ideas come from");
    for (const [title, k] of shown) { h("p", "hint", dd, title); Q.refsLine(dd, refs[k]); }
  }

  /* ---------- funding ---------- */
  function buildFunding(root, d) {
    const rows = d.rows, coins = [...new Set(rows.map((r) => r.symbol))], venues = d.venues;
    header(root, "Funding by venue and coin", `${coins.join(", ")} · ${venues.length} venues · the rate longs pay shorts on the perpetual (positive) or receive (negative)`);
    W.provenance(root, d);
    const by = {}; for (const r of rows) by[r.symbol + "|" + r.venue] = r;
    const top = rows.reduce((a, b) => (b.annualised > a.annualised ? b : a)), low = rows.reduce((a, b) => (b.annualised < a.annualised ? b : a));
    const k = h("div", "earnrow", root);
    tile(k, "Highest funding now", pct(top.annualised, 1), { text: `${top.symbol} on ${top.venue}${top.source === "simulated" ? " · simulated" : ""}` }, "", "A short position here is paid this rate, annualised");
    tile(k, "Lowest funding now", pct(low.annualised, 1), { text: `${low.symbol} on ${low.venue}${low.source === "simulated" ? " · simulated" : ""}` });
    const spread = Math.max(...coins.map((c) => Math.max(...venues.map((v) => (by[c + "|" + v] || {}).annualised ?? -Infinity)) - Math.min(...venues.map((v) => (by[c + "|" + v] || {}).annualised ?? Infinity))));
    tile(k, "Widest gap across venues", pct(spread, 1), { text: "for one coin, annualised" }, "", "The same coin's funding differs by venue: that gap is what a funding arbitrage tries to harvest");
    const ch = h("div", "charts", root);
    chartCard(ch, "Funding now, by coin and venue", "Annualised rate. Blue: longs pay shorts; red: shorts pay longs. Cells marked simulated in the table are modelled; the others are data.", { wide: true,
      draw: (b) => C.matrix(b, { rows: coins, cols: venues, values: coins.map((c) => venues.map((v) => (by[c + "|" + v] ? by[c + "|" + v].annualised : null))), diverging: true, mid: 0, format: (v) => (v * 100).toFixed(1) + "%", name: "Funding, annualised",
        label: "Funding by coin and venue", rowTitle: "Coin", colTitle: "Venue", cellHeight: 30, tipRows: (i, j) => { const r = by[coins[i] + "|" + venues[j]]; return r ? [["7-day average", pct(r.avg_7d, 1)], ["30-day average", pct(r.avg_30d, 1)], ["z-score, 90 days", num(r.zscore_90d, 2)], ["Percentile, 1 year", pct(r.percentile_1y, 0)], ["Source", r.source]] : []; } }),
      table: (b) => table(b, fundingColumns(), rows) });
    const card = h("div", "card", root); h("h3", "", card, "Every venue and coin");
    table(card, fundingColumns(), rows);
    if (!X.form.coin || !coins.includes(X.form.coin)) X.form.coin = coins[0];
    const pick = h("div", "row", root); h("span", "hint", pick, "Funding through time for"); W.seg(pick, coins.map((c) => [c, c]), X.form.coin, (v) => { X.form.coin = v; draw(); });
    const slot = h("div", "charts", root);
    const draw = () => {
      slot.replaceChildren();
      const s = d.series[X.form.coin], names = Object.keys(s);
      chartCard(slot, `${X.form.coin}: funding by venue, annualised`, "Each venue's funding rate through time. A gap between venues is an arbitrage in principle; fees, transfer time and margin usually eat it.", { wide: true,
        draw: (b) => C.line(b, { dates: s[names[0]].dates, series: names.map((v) => ({ name: v + (s[v].source === "simulated" ? " (simulated)" : ""), color: venueColor(v), values: s[v].annualised, width: v === "DERIBIT" ? 2 : 1.5 })), baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 290, label: "Funding by venue", endLabels: false }),
        table: (b) => table(b, [{ label: "Date", get: (n) => s[names[0]].dates[n] }, ...names.map((v) => ({ label: v, num: true, get: (n) => pct(s[v].annualised[n], 2) }))], s[names[0]].dates.map((_, n) => n).filter((n) => n % 5 === 0)) });
    };
    draw();
  }
  const fundingColumns = () => [{ label: "Coin", get: (r) => r.symbol }, { label: "Venue", get: (r) => r.venue }, { label: "Now (a year)", num: true, get: (r) => pct(r.annualised, 1) }, { label: "7-day average", num: true, get: (r) => pct(r.avg_7d, 1) }, { label: "30-day average", num: true, get: (r) => pct(r.avg_30d, 1) },
    { label: "z-score, 90 days", title: "Distance of today's funding from its 90-day average, in standard deviations", num: true, get: (r) => num(r.zscore_90d, 2) }, { label: "Percentile, 1 year", num: true, get: (r) => pct(r.percentile_1y, 0) }, { label: "Interval", num: true, get: (r) => r.interval_hours + " h" }, { label: "Source", get: (r) => r.source }];

  /* ---------- basis and carry ---------- */
  function buildBasis(root, d) {
    const b = d.basis;
    header(root, `${d.symbol} basis and carry`, "Long spot against a short perpetual: delta-neutral, paid the funding rate");
    W.provenance(root, { ...d, warnings: (b && b.risk_flags) || [] });
    if (!b) { W.hint(root, "This coin has no funding data, so the basis trade cannot be measured."); return; }
    const k = h("div", "earnrow", root);
    tile(k, "Basis", pct(b.basis, 3), { text: `z-score ${num(b.basis_zscore, 2)} · percentile ${pct(b.basis_percentile_1y, 0)}` }, "", "Perpetual price over spot, less one");
    tile(k, "Funding, 30-day average", pct(b.funding_annualised_30d, 1), { text: "annualised, paid to the short" });
    tile(k, "Carry earned", pct(b.carry_annualised, 1), { text: `volatility ${pct(b.carry_vol, 2)}` }, "", "Return a year of holding spot long against the perpetual short, over the chart's window");
    tile(k, "Carry Sharpe", num(b.carry_sharpe, 1), { text: `worst fall ${pct(b.carry_max_drawdown, 2)}` }, "", "A carry trade has a high Sharpe ratio until the day it does not: read the worst fall and the tail, not the ratio");
    tile(k, "Break-even", num(b.breakeven_days, 0) + " days", { text: `of carry pays the ${num(b.fee_bps, 1)} bps fee on each of four trades` }, "", "How long the position must be held for its funding to cover opening and closing both legs");
    const ch = h("div", "charts", root);
    chartCard(ch, "Basis", "Perpetual over spot.", {
      draw: (c) => C.line(c, { dates: b.dates, series: [{ name: "Basis", color: css("--s1"), values: b.basis_series }], area: true, areaBase: 0, baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(2) + "%", tipFormat: (v) => pct(v, 3), height: 250, label: "Basis" }),
      table: (c) => table(c, [{ label: "Date", get: (n) => b.dates[n] }, { label: "Basis", num: true, get: (n) => pct(b.basis_series[n], 3) }], b.dates.map((_, n) => n).filter((n) => n % 7 === 0)) });
    chartCard(ch, "Cumulative carry", "Growth of 1 from holding the delta-neutral pair, funding and the change in the basis included, before fees.", {
      draw: (c) => C.line(c, { dates: b.dates, series: [{ name: "Carry", color: css("--s1"), values: b.carry_cumulative }], baseline: 1, yFormat: (v) => v.toFixed(3), tipFormat: (v) => v.toFixed(4), height: 250, label: "Cumulative carry" }),
      table: (c) => table(c, [{ label: "Date", get: (n) => b.dates[n] }, { label: "Growth of 1", num: true, get: (n) => num(b.carry_cumulative[n], 4) }], b.dates.map((_, n) => n).filter((n) => n % 7 === 0)) });
    W.hint(root, b.note + " The risks of the real trade are not in this curve: margin calls on the short leg in a rally, exchange failure, and funding turning negative.");
  }

  /* ---------- across exchanges ---------- */
  function buildCross(root, d) {
    const pairs = d.pairs, venues = d.quotes.map((q) => q.venue);
    header(root, `${d.symbol} across exchanges`, `${venues.length} venues · trade size ${money(d.notional)} · buy at the ask, sell at the bid, pay both taker fees and slippage`);
    W.provenance(root, d);
    if (!pairs.length) { W.hint(root, "There is no pair to compare: a venue is compared only with another venue of the same kind (spot with spot, perpetual with perpetual)."); return; }
    const best = pairs.reduce((a, b) => (b.net_bps > a.net_bps ? b : a));
    const k = h("div", "earnrow", root);
    tile(k, "Best net spread", Q.fmt.signed(best.net_bps, 1) + " bps", { text: `buy ${best.buy}, sell ${best.sell} · ${smoney(best.net_dollars)}`, dir: best.profitable ? "up" : "down" }, "", "Gross price gap less both fees and the slippage on each leg");
    tile(k, "Pairs that pay", `${pairs.filter((p) => p.profitable).length} of ${pairs.length}`, { text: "after fees and slippage" });
    tile(k, "Widest gross gap", num(Math.max(...pairs.map((p) => p.gross_bps)), 2) + " bps", { text: "before any cost" });
    tile(k, "Cheapest round trip", num(Math.min(...pairs.map((p) => p.fees_bps + p.slippage_bps)), 1) + " bps", { text: "fees plus slippage, two legs" });
    const ch = h("div", "charts", root), idx = Object.fromEntries(venues.map((v, i) => [v, i])), grid = venues.map(() => venues.map(() => null));
    for (const p of pairs) grid[idx[p.buy]][idx[p.sell]] = p.net_bps;
    chartCard(ch, "Net spread, buy at the row and sell at the column", "Basis points after both fees and the slippage. Red is a loss: a spread has to beat the cost of crossing it twice before it is an opportunity.", {
      draw: (b) => C.matrix(b, { rows: venues, cols: venues, values: grid, diverging: true, mid: 0, format: (v) => num(v, 1), name: "Net spread (bps)", rowTitle: "Buy at", colTitle: "Sell at", label: "Net spread by venue pair", cellHeight: 34,
        tipRows: (i, j) => { const p = pairs.find((x) => x.buy === venues[i] && x.sell === venues[j]); return p ? [["Gross", num(p.gross_bps, 2) + " bps"], ["Fees", num(p.fees_bps, 1) + " bps"], ["Slippage", num(p.slippage_bps, 1) + " bps"], ["Dollars", smoney(p.net_dollars)]] : []; } }),
      table: (b) => pairTable(b, pairs) });
    const q = h("div", "card", ch); h("h3", "", q, "Quotes");
    const real = d.quotes.some((x) => x.quote_asset);
    table(q, [{ label: "Venue", get: (x) => x.label || x.venue }, ...(real ? [{ label: "Market", get: (x) => `${x.kind} · ${x.quote_asset}` }] : []), { label: "Bid", num: true, get: (x) => money(x.bid, x.bid < 10 ? 4 : 2) }, { label: "Ask", num: true, get: (x) => money(x.ask, x.ask < 10 ? 4 : 2) },
      { label: "Spread (bps)", num: true, get: (x) => num(x.spread_bps, 3) }, { label: "Taker fee (bps)", num: true, get: (x) => num(x.taker_bps, 1) }], d.quotes);
    if (real) W.hint(q, "Real quotes at one moment. A venue is compared only with others of the same kind: spot with spot, perpetual with perpetual. Taker fees are entry-level assumptions, not your tier.");
    const all = h("div", "card", root); h("h3", "", all, "Every pair, best first"); pairTable(all, pairs.slice().sort((a, b) => b.net_bps - a.net_bps));
  }
  const pairTable = (b, rows) => table(b, [{ label: "Buy", get: (p) => p.buy }, { label: "Sell", get: (p) => p.sell }, { label: "Gross (bps)", num: true, get: (p) => num(p.gross_bps, 2) }, { label: "Fees (bps)", num: true, get: (p) => num(p.fees_bps, 1) },
    { label: "Slippage (bps)", num: true, get: (p) => num(p.slippage_bps, 1) }, { label: "Net (bps)", num: true, get: (p) => num(p.net_bps, 2) }, { label: "Net ($)", num: true, get: (p) => smoney(p.net_dollars) }, { label: "Pays?", get: (p) => (p.profitable ? "yes" : "no") }], rows);

  /* ---------- liquidity ---------- */
  function buildLiquidity(root, d) {
    const books = Object.values(d.books), venues = books.map((b) => b.venue);
    header(root, `${d.symbol} liquidity`, `${venues.length} venues · spread, depth and the cost of trading size`);
    W.provenance(root, { ...d, warnings: [] });
    const tight = books.reduce((a, b) => (b.spread_bps < a.spread_bps ? b : a)), deep = books.reduce((a, b) => (b.depth["25bp"].bid + b.depth["25bp"].ask > a.depth["25bp"].bid + a.depth["25bp"].ask ? b : a));
    const k = h("div", "earnrow", root);
    tile(k, "Tightest spread", num(tight.spread_bps, 2) + " bps", { text: tight.venue });
    tile(k, "Deepest book", "$" + compact(deep.depth["25bp"].bid + deep.depth["25bp"].ask), { text: `${deep.venue}, within 25 bps of the mid` });
    const buys = books.map((b) => b.impact_bps["1,000,000"].buy).filter((v) => v !== null && v !== undefined);
    tile(k, "Cost of $1M, buying", buys.length ? num(Math.min(...buys), 2) + " bps" : "n/a", { text: buys.length ? "at the cheapest venue" : "no book is deep enough" }, "", "Price impact beyond the spread of a market buy of $1 million");
    const ch = h("div", "charts", root), rows = (f) => books.map((b) => ({ label: b.venue, value: f(b) }));
    chartCard(ch, "Spread by venue", "Basis points between the best bid and ask.", {
      draw: (b) => C.hbars(b, { rows: rows((x) => x.spread_bps), format: (v) => num(v, 2), name: "Spread (bps)", label: "Spread by venue", color: () => css("--s1") }),
      table: (b) => table(b, [{ label: "Venue", get: (x) => x.venue }, { label: "Spread (bps)", num: true, get: (x) => num(x.spread_bps, 2) }, { label: "Imbalance", title: "Bid depth less ask depth over their sum: positive means more buyers resting than sellers", num: true, get: (x) => num(x.imbalance, 2) }, { label: "Source", get: (x) => x.source }], books) });
    chartCard(ch, "Depth within 25 bps of the mid", "Dollars resting on both sides within a quarter of a percent: what a market order can take before the price moves that far.", {
      draw: (b) => C.hbars(b, { rows: rows((x) => x.depth["25bp"].bid + x.depth["25bp"].ask), format: (v) => "$" + compact(v), name: "Depth (both sides)", label: "Depth within 25 bps", color: () => css("--s1") }),
      table: (b) => table(b, [{ label: "Venue", get: (x) => x.venue }, ...["10bp", "25bp", "50bp"].flatMap((w) => [{ label: `${w} bid`, num: true, get: (x) => "$" + compact(x.depth[w].bid) }, { label: `${w} ask`, num: true, get: (x) => "$" + compact(x.depth[w].ask) }])], books) });
    const sizes = Object.keys(books[0].impact_bps);
    const filled = books.some((x) => sizes.some((s) => x.impact_bps[s].buy !== null && x.impact_bps[s].buy !== undefined));      // real books can be too thin for every size shown
    if (filled) chartCard(ch, "Cost of a market buy, in basis points, by size", "Price impact beyond the spread, for orders of the sizes shown. It grows faster than the size: the last dollars are the dear ones.", { wide: true,
      draw: (b) => C.line(b, { xs: sizes.map((s, i) => i), xWhole: true, xLabel: "Order size", series: books.map((x) => ({ name: x.venue, color: venueColor(x.venue), values: sizes.map((s) => x.impact_bps[s].buy), dots: true })), include: 0, yFormat: (v) => num(v, 1), tipFormat: (v) => num(v, 2) + " bps",
        xFormat: (v) => "$" + (sizes[Math.round(v)] || ""), tipTitle: (i) => "$" + sizes[i], height: 260, label: "Impact by order size", endLabels: false }),
      table: (b) => table(b, [{ label: "Venue", get: (x) => x.venue }, ...sizes.flatMap((s) => [{ label: `$${s} buy`, num: true, get: (x) => num(x.impact_bps[s].buy, 2) }, { label: `$${s} sell`, num: true, get: (x) => num(x.impact_bps[s].sell, 2) }])], books) });
    const stress = " Real books are thinner in stress, which is when the cost matters.";
    W.hint(root, d.snapshot ? `Real order books taken ${d.snapshot.fetched_at ? String(d.snapshot.fetched_at).replace("T", " ").replace("Z", " UTC") : "when they were downloaded"}, on ${d.snapshot.venues.join(", ")}. A cost shown as n/a means the saved levels (up to 400 a side, within 5%) cannot fill an order that size. OKX is quoted in USDT.` + stress
      : books[0].note + stress);
  }

  function buildBacktest(root, d) {
    header(root, `${d.label} on ${d.assets.join(" and ")}`, `${d.text} · ${d.start} to ${d.end} · fees ${num(d.fees_bps, 1)} bps + slippage ${num(d.slippage_bps, 1)} bps one way`);
    W.provenance(root, d);
    W.pipelineResult(root, d);
    const cr = d.correlation, info = strategyInfo(d.strategy);
    if (cr) chartCard(h("div", "charts", root), "How alike the assets are", "The correlation of the assets' daily returns over the backtest, with each asset's own annualised volatility beside its name.", {
      draw: (b) => C.matrix(b, { rows: cr.assets.map((a, i) => `${a} · ${cr.volatility[i] === null ? "n/a" : (cr.volatility[i] * 100).toFixed(0) + "%"}`), cols: cr.assets, values: cr.values, diverging: true, mid: 0, min: -1, max: 1, format: (v) => v.toFixed(2), name: "Correlation", cellHeight: 30, label: "Correlation of the assets" }),
      table: (b) => table(b, [{ label: "Asset", get: (n) => cr.assets[n] }, ...cr.assets.map((a, j) => ({ label: a, num: true, get: (n) => (cr.values[n][j] === null ? "" : num(cr.values[n][j], 2)) }))], cr.assets.map((_, n) => n)) });
    if (info.references && info.references.length) { const c = h("div", "card", root); h("h3", "", c, "Where the rule comes from"); Q.refsLine(c, info.references); }
  }

  /* ---------- the signals board ---------- */
  function buildSignals(root, d) {
    const rules = d.signals.filter((r) => r.strength), failed = d.signals.filter((r) => r.error), coins = d.assets;
    header(root, "What the rules say today", `${coins.join(" and ")} · ${rules.length} of ${d.signals.length} rules · as of ${d.last_day}`);
    W.provenance(root, d);
    const ag = d.agreement, rows = coins.map((a) => ({ a, ...ag[a] })), lean = (r) => (r.long >= r.short ? "long" : "short");
    const k = h("div", "earnrow", root);
    for (const r of rows) tile(k, r.a, `${r.long} long · ${r.short} short`, { text: `${r.flat} without a view of ${rules.length} rules`, dir: r.long > r.short ? "up" : r.long < r.short ? "down" : "" }, "", "Rules with a strength beyond a quarter of their usual size count as a view");
    tile(k, "Rules computed", `${rules.length} of ${d.signals.length}`, { text: failed.length ? `${failed.length} could not be computed` : "all of them" });
    const ch = h("div", "charts", root);
    if (rules.length) chartCard(ch, "Every rule on each coin", `Each cell is the rule's score on the coin today over the average size of that rule's own score in the last ${d.window} days: blue is long, red is short, 1 is a typical position and the scale stops at 3. An outlined cell is a full-sized view. The funding and basis rules are read on the coin's long-spot, short-perpetual index; the Mayer multiple and the volatility-managed hold are long or out, never short.`, { wide: true,
      draw: (b) => C.matrix(b, { rows: rules.map((r) => r.label), cols: coins, values: rules.map((r) => coins.map((a) => r.strength[a])), diverging: true, mid: 0, min: -3, max: 3, format: (v) => v.toFixed(1), name: "Strength", cellHeight: 28, labelWidth: 190,
        mark: (i, j) => Math.abs(rules[i].strength[coins[j]]) >= 1, tipRows: (i, j) => [["Score", rules[i].score[coins[j]] === null ? "n/a" : num(rules[i].score[coins[j]], 3)], ["As of", rules[i].date]], label: "Strength of every rule on each coin" }),
      table: (b) => table(b, [{ label: "Rule", get: (r) => r.label }, ...coins.map((a) => ({ label: a, num: true, get: (r) => (r.score[a] === null ? "" : num(r.score[a], 3) + " (" + num(r.strength[a], 1) + ")") }))], rules) });
    const card = h("div", "card wide", ch); h("h3", "", card, "Where the rules agree");
    table(card, [{ label: "Coin", get: (r) => r.a }, { label: "Long", num: true, get: (r) => r.long }, { label: "Short", num: true, get: (r) => r.short }, { label: "No view", num: true, get: (r) => r.flat }, { label: "Lean", get: (r) => (r.long === r.short ? "split" : lean(r)) }], rows);
    h("p", "note", card, "Agreement is not accuracy: seven of the rules are trend rules, close to one rule counted several times, and with two coins that move together (their correlation is on the Overview) the two columns are nearly one bet.");
    if (failed.length) { const f = h("div", "card wide", root); h("h3", "", f, "Rules that could not be computed"); table(f, [{ label: "Rule", get: (r) => r.label }, { label: "Why", get: (r) => r.error }], failed); }
    if ((d.references || []).length) { const c = h("div", "card", root); h("h3", "", c, "Where the rules come from"); Q.refsLine(c, d.references); }
    W.hint(root, "These are the platform's own strategies read on the last day of the Deribit data; the backtest mode runs any of them over history, with fees and slippage.");
  }

  Q.registerWorkspace("crypto", X);
})();
