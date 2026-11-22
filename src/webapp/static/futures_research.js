/* Two Futures views on the research endpoints: Term-structure research (calendar-spread matrix, the history and z-score of one spread, contango/backwardation regimes, how the curve changed, liquidity, the expiry
   and roll calendar, curve shapes side by side) and the Spread lab (hedge ratio, stationarity and cointegration tests, half-life, rolling correlation, seasonality and a walk-forward mean-reversion backtest).
   Loaded before futures.js; futures.js calls window.FuturesResearch.{research,spread}(root, data, helpers). Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q && Q.C;
  if (!Q || !W) return;
  const h = Q.h, num = Q.num, pct = Q.pct, tile = Q.tile, table = Q.simpleTable, chartCard = Q.chartCard;
  const nz = (v, d) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : num(v, d));

  function research(root, d, ctx) {
    ctx.header(root, `${d.root}: the term structure as a research object`, `${d.spec.name} · ${d.spec.exchange} · multiplier ${d.spec.multiplier} · expiry rule ${d.spec.expiry_rule}`);
    W.provenance(root, d);
    const sh = d.spread_history, m = d.spread_matrix, k = h("div", "earnrow", root);
    tile(k, `Spread ${sh.pair}`, nz(sh.current, 3), { text: `${sh.contracts_now} · a price difference, not a cost` }, "", "The later contract's price less the earlier's. Positive is contango.");
    tile(k, "Z-score against its last year", nz(sh.zscore, 2), { text: `percentile ${nz(sh.percentile, 0)} · window ${sh.window} days` }, "", "Where this spread sits against its own recent distribution. A description, not a signal.");
    if (d.regimes) tile(k, "Regime", d.regimes.current, { text: `for ${d.regimes.current_spell_days} days` }, "", d.regimes.caveat);
    if (d.liquidity.most_liquid) tile(k, "Most open interest", d.liquidity.most_liquid, { text: `OI-weighted ${nz(d.liquidity.oi_weighted_dte, 0)} days to expiry` });
    const ch = h("div", "charts", root);
    chartCard(ch, `Calendar spread ${sh.pair} through time`, sh.note, {
      draw: (b) => C.line(b, { dates: sh.dates, series: [{ name: sh.pair, color: Q.colorOf(0), values: sh.spread, width: 1.8 }], yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 260, label: "Calendar spread history" }),
      table: (b) => table(b, [{ label: "Date", get: (x) => x }, { label: "Spread", num: true, get: (x, i) => nz(sh.spread[i], 3) }], sh.dates.slice(-40)) });
    const n = m.contracts.length;
    const vals = m.percent.map((r) => r.map((v) => (v === null ? null : v)));
    chartCard(ch, `Every calendar spread on ${m.date} (per cent of the earlier contract)`, m.convention, {
      draw: (b) => C.matrix(b, { rows: m.contracts, cols: m.contracts, values: vals, format: (v) => pct(v, 2), name: "Spread", cellHeight: 28, label: "Calendar spread matrix" }),
      table: (b) => table(b, [{ label: "Contract", get: (x) => m.contracts[x] }, ...[1, 2].filter((j) => j < n).map((j) => ({ label: `vs ${m.contracts[j]}`, num: true, get: (x) => (m.points[x][j] === null ? "" : nz(m.points[x][j], 3)) }))], m.contracts.map((_, i) => i)) });
    if (d.regimes) {
      const r = d.regimes, c = h("div", "card", root); c.dataset.role = "regimes"; h("h3", "", c, "Contango and backwardation regimes");
      table(c, [{ label: "Regime", get: (x) => x[0] }, { label: "Share of days", num: true, get: (x) => pct(r.share[x[0]] || 0, 0) }, { label: "Spells", num: true, get: (x) => String((r.spells[x[0]] || {}).count || 0) }, { label: "Mean spell (days)", num: true, get: (x) => nz((r.spells[x[0]] || {}).mean_days, 0) }], Object.keys(r.share).map((x) => [x]));
      h("p", "hint", c, r.caveat);
    }
    const lq = h("div", "card", root); lq.dataset.role = "liquidity"; h("h3", "", lq, `Liquidity by contract, ${d.liquidity.date}`);
    if (d.liquidity.note) h("p", "hint", lq, d.liquidity.note);
    table(lq, [{ label: "Contract", get: (x) => x.contract }, { label: "Expiry", get: (x) => x.expiry }, { label: "Days", num: true, get: (x) => String(x.dte) }, { label: "Open interest", num: true, get: (x) => (x.open_interest === null ? "n/a" : num(x.open_interest, 0)) },
      { label: "Share", num: true, get: (x) => (x.oi_share === null ? "n/a" : pct(x.oi_share, 0)) }, { label: "Volume", num: true, get: (x) => (x.volume === null ? "n/a" : num(x.volume, 0)) }], d.liquidity.rows);
    const cal = h("div", "card", root); cal.dataset.role = "expiry-calendar"; h("h3", "", cal, "Expiry and roll calendar");
    table(cal, [{ label: "Contract", get: (x) => x.contract }, { label: "Last trade", get: (x) => `${x.expiry} (${x.weekday})` }, { label: "Roll date", get: (x) => x.roll_date }, { label: "Days", num: true, get: (x) => String(x.dte) }], d.calendar.rows);
    h("p", "hint", cal, d.calendar.caveat);
    if (d.change) {
      const c = h("div", "card", root); h("h3", "", c, `How the curve changed, ${d.change.from} to ${d.change.to}`);
      table(c, [{ label: "Contract", get: (x) => x }, { label: "Log change", num: true, get: (x, i) => pct(d.change.log_change[i], 2) }], d.change.contracts);
    }
    if (d.compare) chartCard(ch, "Curve shapes compared (each over its own front contract)", d.compare.note, {
      draw: (b) => C.line(b, { xs: d.compare.curves[0].dte, xLabel: "days to expiry", series: d.compare.curves.map((c, i) => ({ name: c.root, color: Q.colorOf(i), values: c.relative, width: 1.8 })), yFormat: (v) => v.toFixed(3), tipFormat: (v) => v.toFixed(4), height: 260, label: "Curve shapes" }),
      table: (b) => table(b, [{ label: "Root", get: (x) => x.root }, { label: "Back / front", num: true, get: (x) => nz(x.relative[x.relative.length - 1], 3) }], d.compare.curves) });
  }

  function spread(root, d, ctx) {
    const hr = d.hedge_ratio, dg = d.diagnostics, bt = d.backtest, s = bt.summary, v = bt.validation;
    ctx.header(root, `${d.a} against ${d.b}: a spread hypothesis`, `ln I_${d.a} − β·ln I_${d.b} on the excess-return indices · ${dg.n} common days`);
    W.provenance(root, d);
    h("p", "hint", root, d.hypothesis);
    const k = h("div", "earnrow", root);
    tile(k, "Stationarity (ADF)", `p = ${nz(dg.adf.p_value, 3)}`, { text: dg.stationary_at_5pct ? "rejects a unit root at 5%" : "cannot reject a unit root" }, dg.stationary_at_5pct ? "" : "muted", dg.adf.null);
    if (dg.engle_granger) tile(k, "Cointegration (Engle-Granger)", `p = ${nz(dg.engle_granger.p_value, 3)}`, { text: dg.cointegrated_at_5pct ? "rejects no-cointegration at 5%" : "no evidence of cointegration" }, dg.cointegrated_at_5pct ? "" : "muted", dg.engle_granger.null);
    tile(k, "Hedge ratio β", nz(hr.ols, 3), { text: `TLS ${nz(hr.tls, 3)} · ${hr.stable ? "steady" : "unstable"} across windows` }, hr.stable ? "" : "muted", hr.reading);
    tile(k, "Half-life of reversion", dg.half_life_days === null ? "none" : `${nz(dg.half_life_days, 1)} days`, { text: "from an AR(1) fit" }, "", "Days for a deviation to halve. None when the spread does not mean-revert.");
    const b = h("div", "earnrow", root);
    tile(b, "Backtest Sharpe", nz(s.sharpe, 2), { text: v.bootstrap && v.bootstrap.available ? `95% interval ${nz(v.bootstrap.interval.sharpe[0], 2)} to ${nz(v.bootstrap.interval.sharpe[1], 2)}` : "no interval" }, "", "Walk-forward: hedge ratio and z-score from the days before each date. Bootstrap: stationary block.");
    tile(b, "Annual return / vol", `${pct(s.annual_return, 1)} / ${pct(s.annual_vol, 1)}`, { text: `max drawdown ${pct(s.max_drawdown, 1)}` });
    tile(b, "Trades", String(s.trades), { text: `${pct(s.share_in_market, 0)} of days in the market` });
    tile(b, "Cost drag", `${pct(s.cost_drag_annual, 2)} a year`, { text: `turnover ${nz(s.turnover_per_year, 1)}× a year` });
    const rl = h("div", "card", root); rl.dataset.role = "realism"; h("h3", "", rl, `Realism: ${v.realism.label}`);
    h("p", "hint", rl, `Not modelled: ${v.realism.not_modelled.join("; ") || "nothing"}.`);
    const ch = h("div", "charts", root);
    chartCard(ch, "The spread and its z-score", `β = ${nz(hr.ols, 3)} (ordinary least squares on the whole sample; the backtest re-estimates it each day from the days before).`, {
      draw: (c) => C.line(c, { dates: d.spread.dates, series: [{ name: "spread", color: Q.colorOf(0), values: d.spread.values, width: 1.6 }], yFormat: (x) => x.toFixed(2), tipFormat: (x) => x.toFixed(4), height: 240, label: "Spread" }),
      table: (c) => table(c, [{ label: "Date", get: (x) => x }, { label: "Spread", num: true, get: (x, i) => nz(d.spread.values[i], 4) }], d.spread.dates.slice(-30)) });
    chartCard(ch, "Backtest equity (walk-forward, net of costs)", bt.caveat, {
      draw: (c) => C.line(c, { dates: bt.dates, series: [{ name: "equity", color: Q.colorOf(1), values: bt.equity, width: 1.8 }], yFormat: (x) => x.toFixed(2), tipFormat: (x) => x.toFixed(3), height: 240, label: "Spread backtest equity" }),
      table: (c) => table(c, [{ label: "Date", get: (x) => x }, { label: "Equity", num: true, get: (x, i) => nz(bt.equity[i], 4) }], bt.dates.slice(-30)) });
    if (dg.rolling_correlation) {
      const rc = dg.rolling_correlation;
      chartCard(ch, "Rolling 63-day correlation of daily changes", `mean ${nz(rc.mean, 2)}, from ${nz(rc.min, 2)} to ${nz(rc.max, 2)}`, {
        draw: (c) => C.line(c, { dates: rc.dates, series: [{ name: "correlation", color: Q.colorOf(2), values: rc.values, width: 1.6 }], yFormat: (x) => x.toFixed(2), tipFormat: (x) => x.toFixed(3), height: 220, label: "Rolling correlation" }),
        table: (c) => table(c, [{ label: "Date", get: (x) => x }, { label: "Correlation", num: true, get: (x, i) => nz(rc.values[i], 3) }], rc.dates.slice(-30)) });
    }
    const sc = h("div", "card", root); sc.dataset.role = "seasonality"; h("h3", "", sc, "Seasonality of the spread (deviation from its mean, by month)");
    table(sc, [{ label: "Month", get: (x) => String(x.month) }, { label: "Mean deviation", num: true, get: (x) => nz(x.mean_deviation, 4) }, { label: "Std", num: true, get: (x) => nz(x.std, 4) }, { label: "Days", num: true, get: (x) => String(x.days) }], dg.seasonality.by_month);
    h("p", "hint", sc, dg.seasonality.caveat);
    const rg = h("div", "card", root); rg.dataset.role = "spread-regimes"; h("h3", "", rg, "Volatility regimes");
    h("p", "hint", rg, `${dg.regimes.split}: now ${dg.regimes.current}; high-volatility share ${pct(dg.regimes.high_vol_share, 0)}; half-life in high ${dg.regimes.mean_reversion_in_high_vol === null ? "n/a" : nz(dg.regimes.mean_reversion_in_high_vol, 1)}, in low ${dg.regimes.mean_reversion_in_low_vol === null ? "n/a" : nz(dg.regimes.mean_reversion_in_low_vol, 1)} days.`);
    h("p", "hint", root, dg.verdict);
    save(root, d, ctx);
  }

  window.Experiments.metricsOf.futures_spread = (d) => ({ summary: d.backtest.summary, adf_p: d.diagnostics.adf.p_value, hedge_ratio: d.hedge_ratio.ols });
  function save(root, d, ctx) {
    const st = d.backtest.settings;
    window.Experiments.saveButton(root, { kind: "futures_spread", name: `${d.a}/${d.b} spread`, request: { path: "/api/futures/spread", query: { a: d.a, b: d.b, lookback: st.lookback, entry: st.entry, exit: st.exit, stop: st.stop, cost_bps: st.cost_bps, simulated: ctx.simulated ? 1 : 0 } },
      metrics: window.Experiments.metricsOf.futures_spread(d), provenance: d.provenance, universe: [d.a, d.b], assumptions: d.provenance && d.provenance.assumptions });
  }

  window.FuturesResearch = { research, spread };
})();
