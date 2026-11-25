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
    surface3d(root, s);
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
    if (window.Experiments) window.Experiments.saveButton(root, { kind: "options_surface", name: `${ctx.nameOf(d)} surface research`, request: { path: "/api/options/surface/research", query: ctx.body || {} },
      metrics: window.Experiments.metricsOf.options_surface(d), provenance: d.provenance, universe: [d.underlying || ctx.nameOf(d)], assumptions: d.provenance && d.provenance.assumptions });
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
    const best = sc.models.find((m) => m.model === sc.preferred_by_combined_rank) || sc.models[0];
    if (window.Experiments && best) window.Experiments.saveButton(root, { kind: "options_calibration", name: `${ctx.nameOf(d)} model validation`, request: { path: "/api/models/validate", body: ctx.body || {} },
      metrics: window.Experiments.metricsOf.options_calibration(d), provenance: d.provenance, universe: [d.underlying || ctx.nameOf(d)], assumptions: d.provenance && d.provenance.assumptions });
    history(root, { underlying: d.underlying || ctx.underlying, model: sc.preferred_by_combined_rank });
  }

  if (window.Experiments) {
    const M = window.Experiments.metricsOf;
    M.options_surface = (d) => { const sl = d.surface.slices, ar = d.surface.arbitrage_summary, ss = d.surface.ssvi && !d.surface.ssvi.error ? d.surface.ssvi : null;
      return { svi_mean_error_pts: sl.reduce((a, x) => a + x.rmse_vol_points, 0) / sl.length, butterfly_failing_slices: ar.butterfly_violating_slices, calendar_failing_pairs: ar.calendar_violating_pairs, ssvi_error_pts: ss ? ss.rmse_vol_points : null }; };
    M.options_strategy = (d) => { const a = d.analysis; return { prob_profit: a.prob_profit, expected_value: a.expected_value, max_profit: a.max_profit, max_loss: a.max_loss, cost: a.cost, margin: a.margin.estimate, greeks: a.greeks }; };
    M.options_calibration = (d) => { const sc = d.scorecard, best = sc.models.find((m) => m.model === sc.preferred_by_combined_rank) || sc.models[0];
      return { n_quotes: d.n_quotes, combined_rank_score: sc.combined_rank_score, preferred: { in_sample: best.in_sample_rmse, held_out: best.out_of_sample_rmse, bic: best.bic, params: best.n_params } }; };
  }
  const smoney = (v) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : (v < 0 ? "-" : "+") + "$" + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 }));

  /* The book's cross-Greeks, its exposure by underlying and by strategy tag, the stress P&L split into the Greeks' contributions and the same stress under the other volatility rule. */
  function riskExtras(root, d) {
    const t = d.totals, k = h("div", "earnrow", root);
    tile(k, "Vanna", num(t.vanna_delta_per_vol_point, 1), { text: "shares of delta per volatility point" }, "", "How much the book's delta changes when implied volatility rises one point: the cross-term between price and volatility.");
    tile(k, "Volga", smoney(t.volga_vega_per_vol_point), { text: "vega change per volatility point" }, "", "How much the book's vega changes when implied volatility rises one point: the convexity in volatility.");
    tile(k, "Rule dependence", smoney(d.rule_gap), { text: `largest gap in the grid, ${d.vol_rule.replace("_", "-")} against ${d.other_rule.replace("_", "-")}` }, d.rule_gap > 1 ? "" : "muted", "Zero with a flat skew slope: the two rules only differ where the smile has a slope.");
    const groups = h("div", "card", root); groups.dataset.role = "risk-groups"; h("h3", "", groups, "Exposure by underlying and by strategy");
    const cols = (key) => [{ label: key === "underlying" ? "Underlying" : "Strategy", get: (g) => g[key] }, { label: "Positions", num: true, get: (g) => String(g.positions) }, { label: "Value", num: true, get: (g) => smoney(g.value) }, { label: "Delta ($)", num: true, get: (g) => smoney(g.delta_dollars) },
      { label: "Gamma, 1% ($)", num: true, get: (g) => smoney(g.gamma_pnl_1pct) }, { label: "Vega ($/pt)", num: true, get: (g) => smoney(g.vega_per_point) }, { label: "Theta ($/day)", num: true, get: (g) => smoney(g.theta_per_day) },
      { label: "Vanna", num: true, get: (g) => num(g.vanna_delta_per_vol_point, 1) }, { label: "Volga", num: true, get: (g) => smoney(g.volga_vega_per_vol_point) }];
    table(groups, cols("underlying"), d.by_underlying); h("div", "", groups);
    table(groups, cols("strategy"), d.by_strategy);
    h("p", "hint", groups, "Tag a position with an underlying or a strategy in the book to group it. A stress moves every underlying by the same percentage (a correlation of one).");
    const att = h("div", "card", root); att.dataset.role = "risk-attribution"; h("h3", "", att, "Where the stress profit and loss comes from");
    const parts = ["delta", "gamma", "vega", "vanna", "volga", "theta", "smile_move", "unexplained"];
    table(att, [{ label: "Price", num: true, get: (r) => (r.spot_shock * 100).toFixed(1) + "%" }, { label: "Volatility (pts)", num: true, get: (r) => (r.vol_shock * 100).toFixed(1) }, { label: "P&L", num: true, get: (r) => smoney(r.pnl) },
      ...parts.map((x) => ({ label: x.replace("_", " "), num: true, get: (r) => smoney(r.attribution[x]) })), { label: `P&L, ${d.other_rule.replace("_", "-")}`, num: true, get: (r) => smoney(d.stress_other_rule.find((o) => o.spot_shock === r.spot_shock && o.vol_shock === r.vol_shock).pnl) }], d.stress);
    h("p", "hint", att, `Volatility rule: ${d.vol_rules[d.vol_rule]}. Unexplained is what the Taylor terms leave: higher orders and the kink at expiry. ` + d.assumptions.join(" · "));
  }

  /* What the probability of profit rests on, the volatility rules and early-exercise risk of a structure. */
  function strategyExtras(parent, a, d, ctx) {
    const p = a.probability, c = h("div", "card wide", parent); c.dataset.role = "probability-audit"; h("h3", "", c, `Probability of profit: ${pct(a.prob_profit, 0)} under ${p.measure}`);
    h("p", "hint", c, `${p.distribution}. ${p.reading}. Range over the volatilities and drifts below: ${pct(p.range[0], 0)} to ${pct(p.range[1], 0)}.`);
    table(c, [{ label: "Volatility", num: true, get: (r) => pct(r.vol, 1) + ` (×${r.vol_multiple})` }, { label: "Drift", get: (r) => r.drift }, { label: "Probability of profit", num: true, get: (r) => pct(r.prob_profit, 0) }, { label: "Expected value", num: true, get: (r) => smoney(r.expected_value) }], p.table);
    for (const x of p.caveats) h("p", "note", c, x);
    const v = a.volatility_rules, r = h("div", "card", parent); r.dataset.role = "vol-rules"; h("h3", "", r, "Volatility rules for the lines drawn today");
    h("p", "hint", r, v.note + `. Largest gap between the two T+0 lines: ${smoney(v.max_gap_now)}.`);
    if (a.early_exercise.length) {
      const e = h("div", "card", parent); e.dataset.role = "early-exercise"; h("h3", "", e, "Early exercise (the legs are priced as European)");
      table(e, [{ label: "Leg", get: (x) => x.leg }, { label: "European value", num: true, get: (x) => num(x.european_value, 2) }, { label: "Intrinsic", num: true, get: (x) => num(x.intrinsic, 2) }, { label: "Why it matters", get: (x) => x.why }], a.early_exercise);
    }
    if (window.Experiments) window.Experiments.saveButton(parent, { kind: "options_strategy", name: `${d.strategy || "own legs"} on ${d.underlying}`, request: { path: "/api/options/strategy/payoff", body: ctx.body || {} },
      metrics: window.Experiments.metricsOf.options_strategy(d), provenance: d.provenance, universe: [d.underlying], assumptions: d.provenance && d.provenance.assumptions });
  }

  /* Dated chains kept on this machine and how a model's parameters move between them. */
  function history(root, ctx) {
    const box = h("div", "card", root); box.dataset.role = "chain-history"; h("h3", "", box, "Calibration stability through time");
    const body = h("div", "", box), snap = h("button", "btn", box, "Keep today's chain"), out = h("span", "hint", box);
    const underlying = ctx.underlying;
    h("p", "hint", box, "Parameter stability needs one chain per date. Free sources give only today's, so a history exists only if you keep one: press the button each trading day. A simulated chain is refused.");
    const load = async () => {
      body.replaceChildren();
      try {
        const d = await Q.api(`/api/options/history?underlying=${encodeURIComponent(underlying)}&model=${encodeURIComponent(ctx.model || "heston")}`);
        h("p", "hint", body, `${d.dates.length} dated chain${d.dates.length === 1 ? "" : "s"} kept for ${underlying}.` + (d.needed ? ` ${d.needed} more needed.` : ""));
        for (const w of d.warnings || []) h("p", "note", body, w);
        if (d.stability) {
          const st = d.stability, names = Object.keys(st.rows[0].params);
          h("p", "hint", body, `${st.model}: ${st.reading} (mean instability ${num(st.mean_instability, 2)}, 0 stable, 1 as large as the whole range).`);
          table(body, [{ label: "Date", get: (r) => r.date }, { label: "Error", num: true, get: (r) => pts(r.rmse) }, ...names.map((n) => ({ label: n, num: true, get: (r) => num(r.params[n], 4) }))], st.rows);
        }
      } catch (e) { h("p", "note", body, e.message); }
    };
    snap.addEventListener("click", async () => { snap.disabled = true; try { const r = await Q.api("/api/options/snapshot", { underlying }); out.textContent = ` Kept ${r.date} (${r.dates_kept} in all).`; } catch (e) { out.textContent = " " + e.message; } finally { snap.disabled = false; } load(); });
    load();
  }

  /* The implied-volatility surface as a rotatable 3D shape (drawn on a canvas; the heatmap beside it carries the same numbers). */
  function surface3d(parent, s) {
    const c = h("div", "card", parent); c.dataset.role = "surface-3d"; h("h3", "", c, "Implied volatility surface in 3D");
    h("p", "hint", c, "Drag to rotate, wheel to zoom. Cells outside the quoted strikes stay open: nothing is extrapolated.");
    const z = s.grid.iv.map((row, i) => row.map((v, j) => (s.grid.observed[i][j] ? v : null))), days = s.grid.expiries.map((e) => e.dte);
    if (window.VolSurface && window.VolSurface.plain3d) { window.VolSurface.plain3d(c, { x: s.grid.k, y: days, z, xLabel: "log-moneyness" }, 380); return; }          // the one shared Plotly component
    window.Surface3D.draw(c, { x: s.grid.k, y: days, z, xLabel: "log-moneyness", yLabel: "days", zLabel: "implied vol", format: (v) => pct(v, 1), label: "Implied volatility surface" });
  }

  window.OptionsResearch = { research, validate, riskExtras, strategyExtras, history, surface3d };
})();
