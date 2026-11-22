/* Two Options views built on the research endpoints: Surface research (the volatility surface in log-forward-moneyness and total variance, per-expiry SVI fits with residuals and bid-ask bands, calendar and
   butterfly diagnostics, a joint SSVI fit) and Model validation (calibration judged on held-out quotes, complexity and numerical health, not only on the fit). Loaded before options.js; options.js calls
   window.OptionsResearch.{research,validate}(root, data, helpers). Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q && Q.C;
  if (!Q || !W) return;
  const h = Q.h, num = Q.num, pct = Q.pct, tile = Q.tile, table = Q.simpleTable, chartCard = Q.chartCard, css = Q.css;
  const pts = (v) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : (v * 100).toFixed(2) + " pts");

  function research(root, d, ctx) {
    const s = d.surface, arb = s.arbitrage_summary;
    ctx.header(root, `${ctx.nameOf(d)}: the surface as a research object`, `${s.slices.length} expiries · coordinates: log-forward-moneyness k = ln(K/F), total variance w = σ²T, delta`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Butterfly (per slice)", `${arb.butterfly_violating_slices} of ${arb.slices} fail`, { text: "Gatheral's g(k) ≥ 0 on a grid" }, arb.butterfly_violating_slices ? "" : "muted", "A negative g(k) means the fitted slice implies a negative density somewhere on the grid.");
    tile(k, "Calendar (between slices)", `${arb.calendar_violating_pairs} of ${Math.max(arb.slices - 1, 0)} pairs fail`, { text: "total variance must not fall with expiry" }, arb.calendar_violating_pairs ? "" : "muted", "Independently fitted slices can cross each other.");
    const ss = s.ssvi && !s.ssvi.error ? s.ssvi : null;
    tile(k, "SSVI conditions", ss ? (ss.conditions.free ? "satisfied" : "violated") : "n/a", { text: ss ? `ρ ${num(ss.rho, 2)} · η ${num(ss.eta, 2)} · γ ${num(ss.gamma, 2)} · error ${num(ss.rmse_vol_points, 2)} pts` : (s.ssvi && s.ssvi.error) || "" }, "", "Gatheral-Jacquier sufficient conditions: free of static arbitrage by construction where they hold.");
    const mean = s.slices.reduce((a, x) => a + x.rmse_vol_points, 0) / s.slices.length;
    tile(k, "SVI fit error", `${num(mean, 2)} pts`, { text: "mean over the expiries" }, "", "Root-mean-square gap, in implied-volatility points, between each fitted slice and its quotes.");
    const ch = h("div", "charts", root);
    const labels = s.grid.expiries.map((e) => `${e.dte}d`), cols = s.grid.k.map((x) => x.toFixed(2));
    const ivVals = s.grid.iv.map((row, i) => row.map((v, j) => (s.grid.observed[i][j] ? v : null)));
    chartCard(ch, "Implied volatility by log-moneyness and expiry", "Cells outside the strikes actually quoted are empty: nothing is extrapolated. Fitted SVI values inside the quoted range.", {
      draw: (b) => C.matrix(b, { rows: labels, cols, values: ivVals, min: Math.min(...ivVals.flat().filter((v) => v !== null)), max: Math.max(...ivVals.flat().filter((v) => v !== null)), format: (v) => pct(v, 1), name: "Implied volatility", cellHeight: 26, label: "Implied volatility heatmap" }),
      table: (b) => table(b, [{ label: "Expiry", get: (n) => labels[n] }, ...[0, Math.floor(cols.length / 2), cols.length - 1].map((j) => ({ label: `k ${cols[j]}`, num: true, get: (n) => (ivVals[n][j] === null ? "n/a" : pct(ivVals[n][j], 1)) }))], labels.map((_, n) => n)) });
    chartCard(ch, "Smiles compared across expiries (total variance)", "w = σ²T against k: total variance should rise with expiry at every k; lines that cross are calendar arbitrage in the fit.", {
      draw: (b) => C.line(b, { xs: s.grid.k, xLabel: "log-forward-moneyness k", series: s.grid.w.map((row, i) => ({ name: labels[i], color: Q.colorOf(i), values: row.map((v, j) => (s.grid.observed[i][j] ? v : null)), width: 1.5 })), yFormat: (v) => v.toFixed(3), tipFormat: (v) => v.toFixed(4), xFormat: (v) => v.toFixed(2), height: 280, label: "Total variance by expiry" }),
      table: (b) => table(b, [{ label: "Expiry", get: (x) => `${x.dte} days` }, { label: "ATM total variance", num: true, get: (x) => num(x.atm_total_variance, 4) }], s.slices) });
    const sl = h("div", "card", root); h("h3", "", sl, "Per-expiry SVI fits"); sl.dataset.role = "svi-slices";
    table(sl, [{ label: "Expiry", get: (x) => `${x.expiry} (${x.dte}d)` }, { label: "Quotes", num: true, get: (x) => String(x.quotes) }, { label: "Error", num: true, get: (x) => `${num(x.rmse_vol_points, 2)} pts` },
      { label: "Inside bid-ask", num: true, get: (x) => (x.share_inside_band === null ? "n/a" : pct(x.share_inside_band, 0)) }, { label: "Min g(k)", num: true, get: (x) => num(x.min_butterfly_g, 3) }, { label: "Butterfly", get: (x) => (x.butterfly_free_on_grid ? "ok on the grid" : "violated") },
      { label: "a, b, ρ, m, s", get: (x) => [x.params.a, x.params.b, x.params.rho, x.params.m, x.params.s].map((v) => num(v, 3)).join(", ") }], s.slices);
    const worst = s.quotes.reduce((best, x) => (Math.abs(x.iv_residual) > Math.abs(best.iv_residual) ? x : best), s.quotes[0]);
    const rc = h("div", "card", root); h("h3", "", rc, "Residuals and bid-ask uncertainty"); rc.dataset.role = "svi-residuals";
    h("p", "hint", rc, `The largest gap between a fitted slice and a quote is ${pts(worst.iv_residual)} (${worst.expiry}, strike ${num(worst.strike, 2)}). A fit inside a quote's bid-ask band is as good as the market itself can say.`);
    const worstRows = s.quotes.slice().sort((a, b) => Math.abs(b.iv_residual) - Math.abs(a.iv_residual)).slice(0, 12);
    table(rc, [{ label: "Expiry", get: (x) => x.expiry }, { label: "Strike", num: true, get: (x) => num(x.strike, 2) }, { label: "k", num: true, get: (x) => num(x.k, 3) }, { label: "Delta", num: true, get: (x) => num(x.delta, 2) },
      { label: "Market σ", num: true, get: (x) => pct(x.iv, 2) }, { label: "Bid σ – ask σ", num: true, get: (x) => (x.bid_iv === null ? "n/a" : `${pct(x.bid_iv, 1)} – ${pct(x.ask_iv, 1)}`) }, { label: "Fit − market", num: true, get: (x) => pts(x.iv_residual) },
      { label: "In band", get: (x) => (x.inside_band === null ? "n/a" : x.inside_band ? "yes" : "no") }], worstRows);
    const lim = h("details", "how", root); h("summary", "", lim, "What is not guaranteed"); const ul = h("ul", "", lim); for (const x of s.limits) h("li", "", ul, x);
  }

  function validate(root, d, ctx) {
    const sc = d.scorecard;
    ctx.header(root, `${ctx.nameOf(d)}: calibration judged on quotes it was not fitted to`, `${d.n_quotes} quotes · loss ${d.loss} · held out: ${sc.validation.mode}`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Preferred by combined rank", sc.preferred_by_combined_rank || "n/a", { text: "fit, out-of-sample error and complexity together" }, "", "The average of the ranks on in-sample error, out-of-sample error and BIC: not the in-sample winner alone.");
    for (const n of sc.notes.slice(0, 2)) tile(k, "Watch", n.split(":")[0], { text: n }, "");
    const ok = sc.models.filter((m) => !m.failed);
    const card = h("div", "card", root); h("h3", "", card, "Models side by side"); card.dataset.role = "model-scorecard";
    table(card, [{ label: "Model", get: (m) => m.label || m.model }, { label: "In-sample", num: true, get: (m) => pts(m.in_sample_rmse) }, { label: "Held out", num: true, get: (m) => (m.out_of_sample_rmse === null || m.out_of_sample_rmse === undefined ? "n/a" : pts(m.out_of_sample_rmse)) },
      { label: "Held-out ÷ in-sample", num: true, get: (m) => (m.oos_ratio ? num(m.oos_ratio, 2) : "n/a") }, { label: "Parameters", num: true, get: (m) => String(m.n_params) }, { label: "BIC", num: true, get: (m) => num(m.bic, 1) },
      { label: "At bounds", get: (m) => (m.at_bounds.length ? m.at_bounds.join(", ") : "none") }, { label: "Time", num: true, get: (m) => `${num(m.seconds, 1)} s` }, { label: "Combined rank", num: true, get: (m) => num(sc.combined_rank_score[m.model], 2) }], ok);
    for (const m of sc.models.filter((x) => x.failed)) h("p", "hint", card, `${m.model}: ${m.failed}`);
    if (d.residual_grid) {
      const g = d.residual_grid, ch = h("div", "charts", root);
      const vals = g.error.map((row) => row.map((v) => (v === null ? null : v * 100)));
      chartCard(ch, `Residuals of ${g.model}: model less market, in volatility points`, "Rows are expiries, columns log-moneyness bins. Structure here (all the wings one colour, one expiry biased) is what an RMSE hides.", {
        draw: (b) => C.matrix(b, { rows: g.expiries_days.map((x) => `${Math.round(x)}d`), cols: g.log_moneyness.map((x) => x.toFixed(2)), values: vals, diverging: true, mid: 0, min: -Math.max(0.1, ...vals.flat().filter((v) => v !== null).map(Math.abs)), max: Math.max(0.1, ...vals.flat().filter((v) => v !== null).map(Math.abs)), format: (v) => `${num(v, 2)} pts`, name: "Residual", cellHeight: 26, label: "Calibration residuals" }),
        table: (b) => table(b, [{ label: "Expiry", get: (n) => `${Math.round(g.expiries_days[n])}d` }, ...[0, Math.floor(g.log_moneyness.length / 2), g.log_moneyness.length - 1].map((j) => ({ label: `k ${g.log_moneyness[j].toFixed(2)}`, num: true, get: (n) => (vals[n][j] === null ? "n/a" : num(vals[n][j], 2)) }))], g.expiries_days.map((_, n) => n)) });
    }
    W.hint(root, sc.caveat);
  }

  window.OptionsResearch = { research, validate };
})();
