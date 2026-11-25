/* The Crypto research views: the market overview with percentiles, derivatives intelligence (funding per interval and annualised, regimes, dispersion, what is not stored), the cash-and-carry P&L decomposition,
   the depth-aware execution simulator and cross-venue executable edge, strategy validation on a 24/7 calendar, and portfolio liquidation risk. It talks to /api/crypto/{market,derivatives,carry,executable,
   execution,validate,liquidation}. Every view says what is not observed instead of drawing it. Server text goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS;
  if (!Q || !W) return;
  const C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, money = Q.money, smoney = Q.smoney, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable;
  const R = { modes: new Set(["market", "derivs", "carry", "exec", "validate", "liq"]) };
  const GUIDE = "technique-crypto-funding-basis-and-venues";
  const head = (root, title, sub, d) => { Q.labHead(root, title, sub, GUIDE); if (d) W.provenance(root, d); };
  function unavailable(root, d, title) { const b = h("div", "unavail", root); h("strong", "", b, title + ": "); b.appendChild(document.createTextNode(d.reason || d.detail || "not available")); if (d.needs && d.needs.length) h("div", "hint", b, "Needed: " + d.needs.join("; ")); return b; }
  const fmtTile = (t) => (t.unit === "pct" ? pct(t.value, 2) : t.unit === "usd" ? money(t.value, 2) : num(t.value, 2));
  R.defaults = () => ({ useTiers: true, haircut: "0",  mkDays: "365", derivSymbols: new Set(["BTC", "ETH"]), notional: "1000000", leverage: "3", cfee: "5", cslip: "2", financing: "0", cdays: "365", eVenue: "", eSide: "buy", eNotional: "100000", eType: "twap", eLimit: "5", eSlices: "6", eReplenish: "50", eProfile: "flat",
    eVol: "", eHours: "0", maxAge: "5", feeMode: "taker_taker", vStrategy: "tsmom", vParams: {}, vAssets: new Set(["BTC", "ETH"]), vFee: "5", vSlip: "2", vTarget: "", vLev: "1", vShort: true, vWalk: true, collateral: "100000", betaNote: "",
    positions: [{ symbol: "BTC", venue: "BINANCE", side: "long", notional: "200000" }, { symbol: "ETH", venue: "OKX", side: "short", notional: "50000" }] });

  /* ---------- the forms ---------- */
  R.sections = (X, sec, symbols, after, rerun) => {
    const f = X.form, m = X.mode, cat = X.catalog;
    const coin = (b) => Q.labField(b, { key: "symbol", label: "Coin", kind: "choice", options: symbols().map((s) => [s.id, `${s.id} (${s.source})`]) }, f, rerun);
    if (m === "market") { const b = sec("1. Coin and window"); coin(b); Q.labField(b, { key: "mkDays", label: "Window", kind: "choice", options: [["90", "90 days"], ["180", "180 days"], ["365", "1 year"], ["730", "2 years"], ["1460", "4 years"]], help: "Every percentile is taken inside this window" }, f, rerun); }
    if (m === "derivs") {
      const b = sec("1. Coins"), box = h("div", "checks", b); h("span", "hint", box, "Up to six:");
      for (const s of symbols()) { const label = h("label", "f inline", box), c = h("input", "", label); c.type = "checkbox"; c.checked = f.derivSymbols.has(s.id); label.appendChild(document.createTextNode(s.id)); c.addEventListener("change", () => { if (c.checked) f.derivSymbols.add(s.id); else f.derivSymbols.delete(s.id); after(); }); }
      h("p", "hint", b, "In observed-only mode a venue whose funding is simulated is listed as unavailable, not shown.");
    }
    if (m === "carry") {
      const b = sec("1. Coin and position"); coin(b);
      for (const s of [{ key: "notional", label: "Notional of each leg ($)", kind: "number", min: 1000 }, { key: "leverage", label: "Perpetual leverage", kind: "number", step: 0.5, min: 1, max: 125, help: "Its initial margin is the notional divided by this; the spot leg is paid in full" },
        { key: "cfee", label: "Fee per fill (bps)", kind: "number", step: 0.5, min: 0 }, { key: "cslip", label: "Slippage per fill (bps)", kind: "number", step: 0.5, min: 0 }, { key: "financing", label: "Financing of the capital (% a year)", kind: "number", step: 0.5, min: 0, help: "The opportunity cost of the spot leg and the margin" },
        { key: "cdays", label: "Days held", kind: "number", step: 30, min: 30, max: 3000 }]) Q.labField(b, s, f, rerun);
    }
    if (m === "exec") {
      const b = sec("1. Order"); coin(b);
      Q.labField(b, { key: "eVenue", label: "Venue", kind: "choice", options: [["", "The first venue of the snapshot"], ...((R.venues || []).map((v) => [v, v]))] }, f, after);
      Q.labField(b, { key: "eSide", label: "Side", kind: "choice", options: [["buy", "Buy"], ["sell", "Sell"]] }, f, after);
      Q.labField(b, { key: "eNotional", label: "Notional ($)", kind: "number", min: 100 }, f, after);
      Q.labField(b, { key: "eType", label: "Order", kind: "choice", options: [["market", "Market"], ["limit", "Limit (takes what is inside the limit)"], ["twap", "TWAP (hypothetical)"], ["vwap", "VWAP (hypothetical)"]] }, f, () => Q.renderLab(X));
      if (f.eType === "limit") Q.labField(b, { key: "eLimit", label: "Limit, bps from the mid", kind: "number", step: 0.5 }, f, after);
      if (f.eType === "twap" || f.eType === "vwap") {
        Q.labField(b, { key: "eSlices", label: "Slices", kind: "number", step: 1, min: 1, max: 200 }, f, after);
        Q.labField(b, { key: "eReplenish", label: "Depth restored between slices (%)", kind: "number", step: 10, min: 0, max: 100, help: "An assumption: the share of the depth a slice consumed that is back before the next one. 0 = never, 100 = fully" }, f, after);
        if (f.eType === "vwap") Q.labField(b, { key: "eProfile", label: "Volume profile", kind: "choice", options: [["flat", "Flat"], ["u_shape", "U shape (assumed, not measured)"]] }, f, after);
        Q.labField(b, { key: "eVol", label: "Volatility for timing risk (% a year, blank = none)", kind: "number", step: 5, min: 0 }, f, after);
        Q.labField(b, { key: "eHours", label: "Hours the schedule runs", kind: "number", step: 1, min: 0, max: 720 }, f, after);
      }
      h("p", "hint", b, "The order walks the real levels of the stored order book. A scheduled order is a hypothesis about how the book would behave, with the assumptions printed beside the result.");
    }
    if (m === "validate" && cat) {
      const b = sec("1. Strategy"), fams = cat.crypto_validation || [];
      Q.labField(b, { key: "vStrategy", label: "Strategy", kind: "choice", options: fams.map((x) => [x.id, x.label, x.family]) }, f, () => Q.renderLab(X));
      const info = fams.find((x) => x.id === f.vStrategy);
      if (info) {
        h("p", "hint", b, `${info.family}. Needs: ${info.needs}.`);
        const grid = h("div", "params", b), mine = f.vParams[info.id] = f.vParams[info.id] || {};
        for (const q of info.params) grid.appendChild(Q.fieldFor({ name: q.name, kind: Number.isInteger(q.default) ? "int" : "float", default: q.default, lo: 0, hi: 1000, text: "" }, q.name in mine ? mine[q.name] : q.default, (v) => { if (v === q.default) delete mine[q.name]; else mine[q.name] = v; after(); }, {}));
      }
      const a = sec("2. Coins"), box = h("div", "checks", a);
      for (const s of symbols()) { const label = h("label", "f inline", box), c = h("input", "", label); c.type = "checkbox"; c.checked = f.vAssets.has(s.id); label.appendChild(document.createTextNode(`${s.id} (${s.source})`)); c.addEventListener("change", () => { if (c.checked) f.vAssets.add(s.id); else f.vAssets.delete(s.id); after(); }); }
      const c = sec("3. Costs and constraints");
      for (const s of [{ key: "vFee", label: "Fee per one-way trade (bps)", kind: "number", step: 0.5, min: 0 }, { key: "vSlip", label: "Slippage per one-way trade (bps)", kind: "number", step: 0.5, min: 0 }, { key: "vTarget", label: "Volatility target (% a year, blank = none)", kind: "number", step: 1, min: 2 },
        { key: "vLev", label: "Largest exposure (x capital)", kind: "number", step: 0.5, min: 0.1, max: 10 }, { key: "vShort", label: "Allow shorts (through the perpetual, with its funding)", kind: "bool" }, { key: "vWalk", label: "Walk-forward over the strategy's own settings", kind: "bool" }]) Q.labField(c, s, f, after);
      h("p", "hint", c, "Daily marks on every calendar day, annualised by 365. The signal at one close trades the next day's return.");
    }
    if (m === "liq") {
      const b = sec("1. Account"); Q.labField(b, { key: "collateral", label: "Collateral (equity, $)", kind: "number", min: 1 }, f, after);
      Q.labField(b, { key: "useTiers", label: "Use illustrative exchange risk tiers (by position size)", kind: "bool", help: "Not any exchange's current schedule: replace it with your venue's tiers through the API" }, f, after);
      Q.labField(b, { key: "haircut", label: "Collateral haircut (%)", kind: "number", step: 1, min: 0, max: 90 }, f, after);
      const p = sec("2. Perpetual positions"), box = h("div", "legs", p);
      const draw = () => {
        box.replaceChildren();
        f.positions.forEach((x, i) => {
          const row = h("div", "row tight", box), mk = (el, label) => { el.setAttribute("aria-label", `${label}, position ${i + 1}`); return el; };
          const sym = mk(h("input", "", row), "coin"); sym.value = x.symbol; sym.addEventListener("change", () => { x.symbol = sym.value.toUpperCase(); after(); });
          const venue = mk(h("select", "", row), "venue"); for (const v of ["BINANCE", "BYBIT", "OKX", "DERIBIT"]) { const o = h("option", "", venue, v); o.value = v; if (x.venue === v) o.selected = true; } venue.addEventListener("change", () => { x.venue = venue.value; after(); });
          const side = mk(h("select", "", row), "side"); for (const v of ["long", "short"]) { const o = h("option", "", side, v); o.value = v; if (x.side === v) o.selected = true; } side.addEventListener("change", () => { x.side = side.value; after(); });
          const n = mk(h("input", "", row), "notional"); n.type = "number"; n.value = x.notional; n.addEventListener("change", () => { x.notional = n.value; after(); });
          const rm = mk(h("button", "btn", row, "Remove"), "remove"); rm.addEventListener("click", () => { f.positions.splice(i, 1); draw(); after(); });
        });
      };
      draw();
      const add = h("button", "btn", h("div", "row tight", p), "+ Position"); add.addEventListener("click", () => { f.positions.push({ symbol: "BTC", venue: "BINANCE", side: "long", notional: "100000" }); draw(); after(); });
      const rf = h("button", "btn", h("div", "row tight", p), "Refresh OKX risk tiers (read-only public request)"), msg = h("span", "hint", p); rf.dataset.role = "refresh-schedule";
      rf.addEventListener("click", async () => { rf.disabled = true; try { const r = await Q.api("/api/crypto/margin-schedules/refresh", { symbol: (f.positions[0] || {}).symbol || "BTC" }); msg.textContent = ` Stored ${r.n_tiers} tiers retrieved ${r.schedule.retrieved_at}. Positions on OKX now use them.`; } catch (e) { msg.textContent = " " + e.message; } finally { rf.disabled = false; } });
      h("p", "hint", p, "Betas of each coin to BTC are measured on observed data when it exists; otherwise 1 is assumed and said.");
    }
  };
  R.check = (X) => {
    const f = X.form;
    if (X.mode === "derivs" && (!f.derivSymbols.size || f.derivSymbols.size > 6)) return "Choose between one and six coins.";
    if (X.mode === "validate" && !f.vAssets.size) return "Choose at least one coin.";
    if (X.mode === "liq" && (!f.positions.length || !(Number(f.collateral) > 0))) return "Add a position and the collateral.";
    return "";
  };

  /* ---------- running ---------- */
  R.run = (X, key) => {
    const m = X.mode, f = X.form, num_ = (v) => Number(v), opt = (v) => (v === "" ? undefined : Number(v));
    if (m === "market") return W.request(X, () => { const q = { symbol: f.symbol, days: num_(f.mkDays) }; return [{ modes: ["market"], path: "/api/crypto/market", query: q, cacheKey: key("market", q) }]; }, "Reading the market…");
    if (m === "derivs") return W.request(X, () => { const q = { symbols: [...f.derivSymbols] }; return [{ modes: ["derivs"], path: "/api/crypto/derivatives", query: q, cacheKey: key("derivs", q) }]; }, "Reading funding on every venue…");
    if (m === "carry") return W.request(X, () => { const q = { symbol: f.symbol, notional: num_(f.notional), leverage: num_(f.leverage), fee_bps: num_(f.cfee), slippage_bps: num_(f.cslip), financing_apr: num_(f.financing) / 100, days: num_(f.cdays) }; return [{ modes: ["carry"], path: "/api/crypto/carry", query: q, cacheKey: key("carry", q) }]; }, "Decomposing the carry…");
    if (m === "exec") return W.request(X, () => { const q = { symbol: f.symbol, venue: f.eVenue || undefined, side: f.eSide, notional: num_(f.eNotional), order_type: f.eType, limit_bps: num_(f.eLimit), slices: num_(f.eSlices), replenish: num_(f.eReplenish) / 100, profile: f.eProfile, vol_annual: opt(f.eVol) === undefined ? undefined : num_(f.eVol) / 100, horizon_hours: num_(f.eHours) };
      return [{ modes: ["exec"], path: "/api/crypto/execution", query: q, cacheKey: key("exec", q) }]; }, "Walking the order book…");
    if (m === "validate") return W.request(X, () => { const params = f.vParams[f.vStrategy] || {}; return [{ modes: ["validate"], job: true, path: "/api/crypto/validate", body: { strategy: f.vStrategy, assets: [...f.vAssets], params, fees_bps: num_(f.vFee), slippage_bps: num_(f.vSlip), target_vol: opt(f.vTarget) === undefined ? undefined : num_(f.vTarget) / 100, max_leverage: num_(f.vLev), allow_short: f.vShort, walk_forward: f.vWalk } }]; }, "Validating on the 24/7 calendar…");
    return W.request(X, () => [{ modes: ["liq"], path: "/api/crypto/liquidation", body: { collateral: num_(f.collateral), use_tiers: f.useTiers, collateral_haircut: num_(f.haircut) / 100, positions: f.positions.map((p) => ({ symbol: p.symbol, venue: p.venue, side: p.side, notional: num_(p.notional) })) } }], "Revaluing under shocks…");
  };
  R.auto = (X) => X.mode !== "validate" && X.mode !== "liq" && X.catalog;
  R.empty = (X, root) => { const t = { validate: ["Validate a strategy on observed history", "Directional, relative-value, derivatives and risk-managed strategies on a 24/7 calendar, with costs, constraints, benchmarks and a walk-forward."], liq: ["Stress a perpetual book", "Positions and collateral: leverage, the shock at which the account is liquidated, and the table of shocks."] }[X.mode] || ["Nothing yet", "Press the button."]; Q.labEmpty(root, X, t[0], [t[1]], X.runLabel[X.mode]); };

  /* Keep a run: its request (so it can be repeated), its headline numbers and where its data came from. */
  const METRICS = {
    crypto_carry: (d) => ({ total: d.pnl.total, funding: d.pnl.funding_received, basis: d.pnl.basis_change, costs: d.pnl.trading_costs, financing: d.pnl.financing, residual: d.pnl.residual, margin_calls: d.margin.margin_calls, min_margin_ratio: d.margin.min_ratio }),
    crypto_execution: (d) => ({ filled_usd: d.result.filled_usd, slippage_bps: d.result.slippage_bps, all_in_bps: d.result.all_in_bps, unfilled_usd: d.result.unfilled_usd }),
    crypto_liquidation: (d) => ({ gross_leverage: d.gross_leverage, net_leverage: d.net_leverage, liquidation_down: d.liquidation_shock_down === null ? -1 : d.liquidation_shock_down, liquidation_up: d.liquidation_shock_up === null ? 1 : d.liquidation_shock_up }),
    crypto_derivatives: (d) => ({ rows: d.rows.length, mean_annualised: d.rows.reduce((a, r) => a + r.annualised, 0) / Math.max(d.rows.length, 1) }),
  };
  if (window.Experiments) Object.assign(window.Experiments.metricsOf, METRICS);
  function save(root, kind, name, d, path, body, post, universe) {
    if (!window.Experiments || !METRICS[kind]) return;
    window.Experiments.saveButton(root, { kind, name, request: post ? { path, body } : { path, query: body }, metrics: METRICS[kind](d), provenance: d.provenance, universe: universe || [], assumptions: d.provenance && d.provenance.assumptions });
  }

  /* ---------- the views ---------- */
  function buildMarket(X, root, d) {
    head(root, `${d.symbol}: market overview`, `${d.window_days}-day window · last mark ${d.last_mark}`, d);
    const k = h("div", "earnrow", root), gone = [];
    for (const t of d.tiles) { if (t.state === "ok") tile(k, t.label, fmtTile(t), { text: (t.percentile === null || t.percentile === undefined ? "" : `${pct(t.percentile, 0)} percentile · `) + t.detail }); else gone.push(t); }
    for (const t of gone) unavailable(root, t, t.label);
    const rg = d.regime; W.kv(h("div", "card", root), "Regime (a description of this window, not a forecast)", [["Trend", rg.trend || "n/a"], ["Volatility", rg.volatility || "n/a"]]); W.hint(root, rg.rule);
    const s = d.series, ch = h("div", "charts", root);
    chartCard(ch, "Price and drawdown", "Daily marks over the window; the second line is the fall from the window's high.", { wide: true,
      draw: (b) => C.line(b, { dates: s.dates, series: [{ name: "Spot", color: css("--s1"), values: s.spot }], log: true, label: "Spot", endLabels: false }), table: (b) => table(b, [{ label: "Date", get: (n) => s.dates[n] }, { label: "Spot", num: true, get: (n) => money(s.spot[n], 2) }, { label: "Drawdown", num: true, get: (n) => pct(s.drawdown[n], 1) }], s.dates.map((_, n) => n).filter((n) => n % 7 === 0)) });
  }
  function buildDerivs(X, root, d) {
    head(root, "Derivatives intelligence", `${d.rows.length} funding series · ${d.observed_only ? "observed only" : "simulated venues included and labelled"}`, d);
    W.hint(root, d.normalisation);
    const card = h("div", "card", root); h("h3", "", card, "Funding by coin and venue");
    table(card, [{ label: "Coin", get: (r) => r.symbol }, { label: "Venue", get: (r) => r.venue + (r.source === "simulated" ? " (simulated)" : "") }, { label: "Per interval", num: true, get: (r) => pct(r.rate_per_interval, 4) + ` / ${r.interval_hours}h` }, { label: "Annualised", num: true, get: (r) => pct(r.annualised, 1) },
      { label: "7d avg", num: true, get: (r) => pct(r.avg_7d, 1) }, { label: "30d avg", num: true, get: (r) => pct(r.avg_30d, 1) }, { label: "z, 90d", num: true, get: (r) => num(r.zscore_90d, 2) }, { label: "Pctile 1y", num: true, get: (r) => pct(r.percentile_1y, 0) },
      { label: "Pctile all", num: true, get: (r) => pct(r.percentile_full, 0) }, { label: "Regime", get: (r) => `${r.regime.current || "n/a"} since ${r.regime.since || "n/a"} (${r.regime.n_changes} changes)` }, { label: "Next payment (UTC)", get: (r) => (r.calendar.next[0] || "n/a") }], d.rows);
    if (d.unavailable_rows.length) { const dd = h("details", "how", root); h("summary", "", dd, `Not shown (${d.unavailable_rows.length})`); table(dd, [{ label: "Coin", get: (r) => r.symbol }, { label: "Venue", get: (r) => r.venue || "" }, { label: "Why", get: (r) => r.reason }], d.unavailable_rows); }
    const disp = Object.entries(d.dispersion); if (disp.length) { const c = h("div", "card", root); h("h3", "", c, "Dispersion across venues"); table(c, [{ label: "Coin", get: (r) => r[0] }, { label: "Venues", get: (r) => r[1].venues.join(", ") }, { label: "Gap (a year)", num: true, get: (r) => (r[1].dispersion_annualised === null || r[1].dispersion_annualised === undefined ? r[1].reason : pct(r[1].dispersion_annualised, 1)) }, { label: "Highest", get: (r) => r[1].highest || "" }, { label: "Lowest", get: (r) => r[1].lowest || "" }], disp); }
    const bs = Object.entries(d.basis); if (bs.length) { const c = h("div", "card", root); h("h3", "", c, "Perpetual against spot"); table(c, [{ label: "Coin", get: (r) => r[0] }, { label: "Basis", num: true, get: (r) => pct(r[1].basis, 3) }, { label: "Pctile 1y", num: true, get: (r) => pct(r[1].percentile_1y, 0) }, { label: "z, 90d", num: true, get: (r) => num(r[1].zscore_90d, 2) }], bs); }
    const oi = d.open_interest;
    if (oi.state === "ok") { const c = h("div", "card", root); h("h3", "", c, "Open interest by exchange"); table(c, [{ label: "Venue", get: (r) => r.venue }, { label: "Unit", get: (r) => r.unit }, { label: "Latest", num: true, get: (r) => num(r.latest, 0) }, { label: "1d", num: true, get: (r) => (r.change_1d === null ? "" : pct(r.change_1d, 1)) }, { label: "7d", num: true, get: (r) => (r.change_7d === null ? "" : pct(r.change_7d, 1)) },
      { label: "30d", num: true, get: (r) => (r.change_30d === null ? "" : pct(r.change_30d, 1)) }, { label: "Percentile", num: true, get: (r) => pct(r.percentile, 0) }, { label: "As of", get: (r) => r.as_of }], oi.venues); h("p", "hint", c, oi.units_note + "."); } else unavailable(root, oi, "Open interest by exchange");
    const lq = d.liquidations;
    if (lq.state === "ok") W.kv(h("div", "card", root), "Recent liquidations (" + lq.venue + ", " + num(lq.window_hours, 1) + " hours)", [["Events", String(lq.events)], ["Longs liquidated", String(lq.longs_liquidated) + (lq.longs_notional === undefined ? "" : " · " + money(lq.longs_notional, 0))], ["Shorts liquidated", String(lq.shorts_liquidated) + (lq.shorts_notional === undefined ? "" : " · " + money(lq.shorts_notional, 0))], ["Note", lq.note]]); else unavailable(root, lq, "Liquidations");
    const ts = d.dated_futures;
    if (ts.state === "ok") { const c = h("div", "card", root); h("h3", "", c, "Dated futures term structure (" + ts.venue + ")"); table(c, [{ label: "Contract", get: (r) => r.instrument }, { label: "Days", num: true, get: (r) => num(r.days, 0) }, { label: "Basis", num: true, get: (r) => pct(r.basis, 2) }, { label: "Annualised", num: true, get: (r) => pct(r.annualised_basis, 1) }, { label: "Open interest", num: true, get: (r) => (r.open_interest === null ? "" : num(r.open_interest, 0)) }], ts.contracts); h("p", "hint", c, ts.note + ", fetched " + ts.fetched_at + "."); } else unavailable(root, ts, "Dated futures term structure");
    save(root, "crypto_derivatives", `funding, ${[...new Set(d.rows.map((r) => r.symbol))].join("+")}`, d, "/api/crypto/derivatives", X.lastBody, false, [...new Set(d.rows.map((r) => r.symbol))]);
  }
  function buildCarry(X, root, d) {
    if (d.state === "unavailable") { head(root, "Cash-and-carry", "", d); unavailable(root, d, "Cash-and-carry"); return; }
    head(root, `${d.symbol} cash-and-carry`, `${d.days} days · ${money(d.notional, 0)} on each leg · ${d.leverage}x perpetual leverage · ${d.perp_venue.toLowerCase()} ${d.perp_type} perpetual`, d);
    const p = d.pnl, m = d.margin, k = h("div", "earnrow", root);
    tile(k, "P&L", smoney(p.total), { text: `${pct(p.return_on_capital, 2)} on ${money(d.capital, 0)} · ${pct(p.annualised_on_capital, 1)} a year`, dir: p.total >= 0 ? "up" : "down" });
    tile(k, "Funding received", smoney(p.funding_received)); tile(k, "Costs and financing", smoney(p.trading_costs + p.financing)); tile(k, "Margin calls", String(m.margin_calls), { text: m.first_breach_day ? `first on ${m.first_breach_day}` : `lowest margin ratio ${num(m.min_ratio, 2)}` });
    tile(k, "Liquidation at", pct(m.liquidation_move_from_entry, 1) + " up", { text: `perpetual price ${money(m.liquidation_price_at_entry, 2)}` });
    const parts = [["Funding received", p.funding_received], ["Basis change", p.basis_change], ["Unhedged delta", p.unhedged_delta], ["Collateral revaluation", p.collateral_revaluation], ["Trading costs", p.trading_costs], ["Financing", p.financing], ["Residual", p.residual]];
    let acc = 0; const rows = parts.map(([label, v]) => { const from = acc; acc += v; return { label, value: acc, from, tip: [["Amount", smoney(v)]] }; });
    const ch = h("div", "charts", root);
    chartCard(ch, "Where the P&L came from", "Each bar starts where the one before ended; the last is the explicit residual against an independent total. " + d.residual_note + ".", { draw: (b) => C.hbars(b, { rows, format: (v) => smoney(v), name: "P&L", label: "P&L decomposition", labelWidth: 150, valueWidth: 90, color: (v) => (v >= 0 ? css("--pos") : css("--neg")) }), table: (b) => table(b, [{ label: "Component", get: (r) => r[0] }, { label: "$", num: true, get: (r) => smoney(r[1]) }], parts) });
    chartCard(ch, "Cumulative funding and basis", "Dollars through the holding period.", { draw: (b) => C.line(b, { dates: d.series.dates, series: [{ name: "Funding", color: css("--s1"), values: d.series.funding }, { name: "Basis", color: css("--s2"), values: d.series.basis, width: 1.5 }], baseline: 0, yFormat: (v) => smoney(v), label: "Funding and basis", endLabels: false }),
      table: (b) => table(b, [{ label: "Date", get: (n) => d.series.dates[n] }, { label: "Funding", num: true, get: (n) => smoney(d.series.funding[n]) }, { label: "Basis", num: true, get: (n) => smoney(d.series.basis[n]) }], d.series.dates.map((_, n) => n).filter((n) => n % 14 === 0)) });
    const fr = d.funding, c = h("div", "card", root); h("h3", "", c, "Funding reversal and margin risk");
    W.kv(c, "Funding", [["Average per day", smoney(fr.mean_per_day_usd)], ["Days the short paid", String(fr.negative_days)], ["Longest run of paid days", String(fr.longest_negative_run)], ["Worst fall in cumulative funding", smoney(fr.worst_cumulative_drawdown_usd)], ["Days of funding that pay the four fills", fr.breakeven_days === null ? "never (funding is not positive)" : num(fr.breakeven_days, 1)]]);
    table(c, [{ label: "Perpetual price up", get: (r) => r[0] }, { label: "Margin ratio", num: true, get: (r) => num(r[1].margin_ratio, 2) }, { label: "Top-up needed", num: true, get: (r) => money(r[1].topup_needed, 0) }], Object.entries(m.shock_table));
    const dd = h("details", "how", root); h("summary", "", dd, "Assumptions"); for (const a of d.assumptions) h("p", "hint", dd, a);
    save(root, "crypto_carry", `${d.symbol} carry, ${d.leverage}x`, d, "/api/crypto/carry", X.lastBody, false, [d.symbol]);
  }
  function buildExec(X, root, d) {
    head(root, `${d.symbol}: order execution`, "", d);
    if (d.state === "unavailable") { unavailable(root, d, "Execution simulator"); return; }
    const r = d.result, m = d.market_reference, k = h("div", "earnrow", root);
    R.venues = d.venues;
    tile(k, "Filled", pct(r.fill_fraction, 1), { text: `${money(r.filled_usd, 0)} of ${money(d.notional, 0)}` }); tile(k, "Average price", r.vwap === null ? "n/a" : money(r.vwap, 2), { text: `mid ${money(d.mid, 2)}` });
    tile(k, "Slippage", r.slippage_bps === null ? "n/a" : num(r.slippage_bps, 2) + " bps", { text: `all in ${r.all_in_bps === null ? "n/a" : num(r.all_in_bps, 2)} bps with the ${num(d.fee_bps, 1)} bps fee` });
    tile(k, "Against a market order", r.saving_vs_market_bps === null ? "n/a" : Q.fmt.signed(r.saving_vs_market_bps, 2) + " bps", { text: `the market order slips ${m.slippage_bps === null ? "n/a" : num(m.slippage_bps, 2)} bps` });
    if (r.timing_risk_bps_1sd !== undefined) tile(k, "Timing risk, 1 sd", num(r.timing_risk_bps_1sd, 1) + " bps", { text: "price drift while the schedule runs" });
    if (r.unfilled_usd > 0) W.hint(root, `${money(r.unfilled_usd, 0)} stays unfilled.`);
    if (d.children) { const c = h("div", "card", root); h("h3", "", c, "Child orders"); table(c, [{ label: "Slice", num: true, get: (x) => x.slice }, { label: "Notional", num: true, get: (x) => money(x.notional, 0) }, { label: "Price", num: true, get: (x) => (x.vwap === null ? "" : money(x.vwap, 2)) }, { label: "Slippage", num: true, get: (x) => (x.slippage_bps === null ? "" : num(x.slippage_bps, 2) + " bps") }], d.children); }
    if (d.assumptions) { const dd = h("details", "how", root); dd.open = true; h("summary", "", dd, "Assumptions behind this schedule"); for (const a of d.assumptions) h("p", "hint", dd, a); }
    W.hint(root, d.reading);
    save(root, "crypto_execution", `${d.symbol} ${d.order_type} ${d.side}`, d, "/api/crypto/execution", X.lastBody, false, [d.symbol]);
  }
  function buildValidate(X, root, d) {
    head(root, `${d.label}: validation`, `${d.assets.join(", ")} · ${d.start} to ${d.end} · 24/7 calendar, annualised by ${d.annualisation}`, d);
    const n = d.net, g = d.gross, k = h("div", "earnrow", root);
    tile(k, "Sharpe (net)", num(n.sharpe, 2), { text: `gross ${num(g.sharpe, 2)}` }); tile(k, "Return a year", pct(n.cagr, 1), { text: `volatility ${pct(n.ann_vol, 1)}` }); tile(k, "Worst fall", pct(n.max_drawdown, 1), { text: `Calmar ${num(n.calmar, 2)}` });
    tile(k, "Costs a year", pct(d.costs.cost_drag_per_year, 2), { text: `turnover ${num(d.costs.turnover_per_year, 1)}x a year` });
    if (d.walk_forward && d.walk_forward.oos_sharpe !== undefined) tile(k, "Out-of-sample Sharpe", num(d.walk_forward.oos_sharpe, 2), { text: `in-sample best ${num(d.walk_forward.in_sample_sharpe, 2)} · ${d.walk_forward.n_settings} settings` });
    for (const note of d.verdict.notes) h("p", "verdict-bad", root, note);
    W.hint(root, d.verdict.reading);
    const ch = h("div", "charts", root), s = d.series;
    chartCard(ch, "Equity and drawdown", "Growth of 1 on the net returns.", { wide: true, draw: (b) => C.line(b, { dates: s.dates, series: [{ name: "Strategy (net)", color: css("--s1"), values: s.equity }], label: "Equity", endLabels: false }), table: (b) => table(b, [{ label: "Date", get: (i) => s.dates[i] }, { label: "Equity", num: true, get: (i) => num(s.equity[i], 3) }, { label: "Drawdown", num: true, get: (i) => pct(s.drawdown[i], 1) }], s.dates.map((_, i) => i).filter((i) => i % 14 === 0)) });
    const bm = Object.entries(d.benchmarks), c = h("div", "card", root); h("h3", "", c, "Against benchmarks");
    table(c, [{ label: "", get: (r) => r[0] }, { label: "Return a year", num: true, get: (r) => (r[1].cagr === undefined ? "" : pct(r[1].cagr, 1)) }, { label: "Volatility", num: true, get: (r) => (r[1].ann_vol === undefined ? "" : pct(r[1].ann_vol, 1)) }, { label: "Sharpe", num: true, get: (r) => (r[1].sharpe === null || r[1].sharpe === undefined ? "" : num(r[1].sharpe, 2)) }, { label: "Worst fall", num: true, get: (r) => (r[1].max_drawdown === undefined ? "" : pct(r[1].max_drawdown, 1)) }], [["This strategy (net)", n], ...bm]);
    const bs = d.bootstrap; if (bs && bs.available) W.hint(root, `Bootstrap 95% interval of the Sharpe ratio: ${num(bs.ci.sharpe[0], 2)} to ${num(bs.ci.sharpe[1], 2)}. ${bs.reading}.`);
    if (d.info && d.info.cointegration && d.info.cointegration.available) W.hint(root, `Engle-Granger cointegration of ETH and BTC over the sample: p = ${num(d.info.cointegration.p_value, 3)}.`);
    const cal = h("div", "card", root); h("h3", "", cal, "Calendar"); W.kv(cal, "Marks", [["Calendar days", String(d.calendar.calendar_days)], ["Missing days", String(d.calendar.missing_days)], ["Weekend marks", String(d.calendar.weekend_marks)], ["Annualisation", `${d.annualisation} (the weekday pipeline would use 252 and drop ${d.calendar.weekday_pipeline.marks_it_drops} marks)`]]);
    if (d.walk_forward && d.walk_forward.folds) { const w = h("div", "card", root); h("h3", "", w, "Walk-forward folds"); table(w, [{ label: "Train", get: (f) => f.train.join(" to ") }, { label: "Test", get: (f) => f.test.join(" to ") }, { label: "Chosen", get: (f) => f.chosen }, { label: "Train Sharpe", num: true, get: (f) => num(f.train_sharpe, 2) }, { label: "Test Sharpe", num: true, get: (f) => num(f.test_sharpe, 2) }], d.walk_forward.folds); W.hint(root, d.walk_forward.reading); }
    const dd = h("details", "how", root); h("summary", "", dd, "Assumptions"); for (const a of (d.notes || [])) h("p", "hint", dd, a);
    if (window.Experiments) window.Experiments.saveButton(root, { kind: "crypto_validation", name: `${d.label} on ${d.assets.join("+")}`, request: { path: "/api/crypto/validate", body: X.lastBody || {} }, metrics: window.Experiments.metricsOf.crypto_validation(d), provenance: d.provenance, universe: d.assets, assumptions: d.notes });
  }
  function buildLiq(X, root, d) {
    head(root, "Portfolio liquidation risk", `collateral ${money(d.collateral, 0)} · gross ${money(d.gross_notional, 0)} · ${num(d.gross_leverage, 2)}x gross, ${num(d.net_leverage, 2)}x net`, d);
    const k = h("div", "earnrow", root);
    tile(k, "Liquidated on a fall of", d.liquidation_shock_down === null ? "none up to -95%" : pct(d.liquidation_shock_down, 1), { text: "market factor, cross margin" }); tile(k, "Liquidated on a rise of", d.liquidation_shock_up === null ? "none up to +95%" : pct(d.liquidation_shock_up, 1));
    tile(k, "Liquidation fee at the nearest", d.liquidation_fee_loss_at_nearest === null ? "n/a" : money(d.liquidation_fee_loss_at_nearest, 0));
    const c = h("div", "card", root); h("h3", "", c, "Shocks");
    table(c, [{ label: "Market move", get: (r) => r[0] }, { label: "Equity", num: true, get: (r) => money(r[1].equity, 0) }, { label: "Required", num: true, get: (r) => money(r[1].required, 0) }, { label: "Margin ratio", num: true, get: (r) => (r[1].margin_ratio === null ? "" : num(r[1].margin_ratio, 2)) }, { label: "", get: (r) => (r[1].liquidated ? "LIQUIDATED" : "") }], Object.entries(d.shock_table));
    const p = h("div", "card", root); h("h3", "", p, "Each position on its own share of the collateral");
    table(p, [{ label: "Coin", get: (x) => x.symbol }, { label: "Venue", get: (x) => x.venue }, { label: "Notional", num: true, get: (x) => money(x.notional, 0) }, { label: "Leverage", num: true, get: (x) => num(x.isolated.leverage, 2) + "x" }, { label: "Liquidated at", num: true, get: (x) => pct(x.isolated.liquidation_move, 1) }, { label: "Over the venue limit", get: (x) => (x.isolated.over_venue_limit ? "yes" : "") }, { label: "Margin basis", get: (x) => x.margin_basis }], d.positions);
    const acc = h("p", d.tiers_verified ? "hint" : "verdict-bad", root, d.accuracy_note); acc.dataset.role = "margin-accuracy";
    h("p", "hint", p, "Betas to BTC: " + (Object.keys(d.betas).length ? Object.entries(d.betas).map(([a, b]) => `${a} ${num(b, 2)}`).join(", ") : "none measured (1 assumed)") + ".");
    h("p", "hint", root, "Margin model: " + d.margin_model + ". " + (d.approximate ? "An approximate stress scenario, not an exchange's liquidation engine." : ""));
    const dd = h("details", "how", root); h("summary", "", dd, "Assumptions"); for (const a of d.assumptions) h("p", "hint", dd, a);
    save(root, "crypto_liquidation", `${d.positions.length}-position book`, d, "/api/crypto/liquidation", X.lastBody, true, [...new Set(d.positions.map((p) => p.symbol))]);
  }
  R.build = (X, root, d, body) => { X.lastBody = body; return ({ market: buildMarket, derivs: buildDerivs, carry: buildCarry, exec: buildExec, validate: buildValidate, liq: buildLiq })[X.mode](X, root, d, body); };

  /* The depth-aware executable edge of the cross-venue view: fetched after the quoted spreads and drawn under them. */
  R.executable = (X, root, symbol, notional) => {
    const card = h("div", "card", root); h("h3", "", card, "Executable on the real order books"); const body = h("div", "", card); Q.loading(body, "Walking both order books…");
    Q.api("/api/crypto/executable?symbol=" + encodeURIComponent(symbol) + "&notional=" + encodeURIComponent(notional)).then((d) => {
      body.replaceChildren();
      if (d.state === "unavailable") { unavailable(body, d, "Executable edge"); return; }
      h("p", "hint", body, `Snapshot of ${d.fetched_at}${d.age_seconds === null ? "" : ` (${num(d.age_seconds, 0)} seconds old)`}${d.stale ? ": STALE, older than " + d.max_age_s + " seconds" : ""}.`);
      table(body, [{ label: "Buy", get: (r) => r.buy }, { label: "Sell", get: (r) => r.sell }, { label: "Quoted (bps)", num: true, get: (r) => num(r.quoted_bps, 2) }, { label: "Executable (bps)", num: true, get: (r) => num(r.executable_bps, 2) }, { label: "Impact (bps)", num: true, get: (r) => num(r.price_impact_bps, 2) },
        { label: "Fees (bps)", num: true, get: (r) => num(r.fees_bps, 1) }, { label: "Net ($)", num: true, get: (r) => smoney(r.net_usd) }, { label: "Filled", num: true, get: (r) => pct(r.fill_fraction, 0) }, { label: "Verdict", get: (r) => r.verdict.replace(/_/g, " ") }, { label: "Needs", get: (r) => r.requirements.join("; ") }, { label: "Notes", get: (r) => r.notes.join("; ") }], d.pairs);
      h("p", "hint", body, "Not modelled: " + d.not_modelled.join("; ") + "."); h("p", "verdict-bad", body, d.reading);
    }, (e) => { body.replaceChildren(); h("p", "hint", body, e.message); });
  };
  if (window.Experiments) window.Experiments.metricsOf.crypto_validation = (d) => ({ sharpe: d.net.sharpe, cagr: d.net.cagr, ann_vol: d.net.ann_vol, max_drawdown: d.net.max_drawdown, cost_drag: d.costs.cost_drag_per_year, ...(d.walk_forward && d.walk_forward.oos_sharpe !== undefined ? { oos_sharpe: d.walk_forward.oos_sharpe } : {}) });
  window.CryptoResearch = R;
})();
