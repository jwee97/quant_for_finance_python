/* Model Comparison: several models and the baselines, trained and traded on identical data, splits, costs and days.

   Predictive quality (does the score order the assets, is it calibrated, does it beat a forecast of nothing) and trading quality (what the book earned after costs, its drawdown, turnover, beta) are separate
   tables, because a model can rank well and still not trade profitably, and the reverse. The baselines go through the same engine; the random-signal books are what no skill earns at this pace. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, css } = Q, C = Q.C;
  const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.num(v, d)), pct = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.pct(v, d));
  const root = $("#modelcompare");
  const MODELS = [["ridge", "Ridge"], ["lasso", "Lasso"], ["elasticnet", "Elastic net"], ["random_forest", "Random forest"], ["hist_gb", "Gradient boosting"]];
  const S = { models: new Set(["ridge", "lasso"]), rule: "benchmark_tilt", tilt: 0.5, quantile: 0.3, rebalance: "monthly", cost: 5, slip: 2, min_train: 756, draws: 10, result: null, busy: false };

  async function draw() {
    root.replaceChildren();
    const d = h("div", "head", root), l = h("div", "", d); h("h2", "", l, "Model Comparison");
    h("p", "sub", l, "Several models and simple baselines, trained and traded on identical data, splits, costs and days, so that a difference between rows is a difference between strategies and not between set-ups.");
    form(); const out = h("div", "stack", root); out.id = "compare-results"; out.setAttribute("aria-live", "polite"); if (S.result) results(out, S.result);
    else { const e = h("div", "card empty", out); h("h3", "", e, "No comparison yet"); h("p", "", e, "Choose the models and the position rule, then run. Training several models walk-forward takes a minute or more each on the default universe."); }
  }

  function form() {
    const card = h("div", "card", root); card.dataset.role = "compare-form"; const grid = h("div", "params", card);
    const box = h("div", "wide", grid); h("div", "hint", box, "Models (up to 6)"); const row = h("div", "chips", box);
    for (const [k, label] of MODELS) { const l = h("label", "f inline", row), cb = h("input", "", l); cb.type = "checkbox"; cb.checked = S.models.has(k); cb.dataset.model = k; l.appendChild(document.createTextNode(" " + label)); cb.addEventListener("change", () => { cb.checked ? S.models.add(k) : S.models.delete(k); }); }
    const sl = h("label", "f", grid, "Position rule"), ss = h("select", "", sl); ss.dataset.key = "rule";
    for (const [v, t] of [["benchmark_tilt", "Long only, tilted around equal weight by rank (like for like with it)"], ["long_only_top_quantile", "Long the top share of scores (long only, more concentrated)"], ["top_n_long_short", "Long the best, short the worst (dollar neutral)"], ["decision_policy", "Explicit buy / sell / hold decisions"]]) { const o = h("option", "", ss, t); o.value = v; if (v === S.rule) o.selected = true; }
    ss.addEventListener("change", () => { S.rule = ss.value; });
    const num_ = (label, key, help) => { const l = h("label", "f", grid, label); if (help) l.title = help; const i = h("input", "", l); i.type = "number"; i.step = "any"; i.value = S[key]; i.dataset.key = key; i.addEventListener("input", () => { S[key] = i.value; }); };
    num_("Tilt (tilted around equal weight)", "tilt", "0 is the benchmark, 1 the strongest tilt"); num_("Top share (long-only rule)", "quantile", "0.3 holds the best 30% of names"); num_("Cost (bp, one way)", "cost"); num_("Slippage (bp)", "slip"); num_("Training history before the first prediction (days)", "min_train"); num_("Random-signal books to average", "draws");
    const rl = h("label", "f", grid, "Rebalance"), rs = h("select", "", rl); rs.dataset.key = "rebalance"; for (const v of ["daily", "weekly", "monthly", "on_change"]) { const o = h("option", "", rs, v); o.value = v; if (v === S.rebalance) o.selected = true; } rs.addEventListener("change", () => { S.rebalance = rs.value; });
    const act = h("div", "row", card), go = h("button", "btn primary", act, "Run the comparison"), status = h("span", "hint", act, ""); go.type = "button"; go.disabled = S.busy;
    go.addEventListener("click", async () => {
      const signal = S.rule === "long_only_top_quantile" ? { method: S.rule, quantile: Number(S.quantile) } : S.rule === "benchmark_tilt" ? { method: S.rule, tilt: Number(S.tilt) } : { method: S.rule };
      const body = { models: [...S.models].map((kind) => ({ kind })), signal, backtest: { rebalance: S.rebalance, cost_bps: Number(S.cost), slippage_bps: Number(S.slip) }, split: { min_train_days: Number(S.min_train) }, draws: Number(S.draws), explain: false };
      S.busy = true; go.disabled = true; status.textContent = "Starting…";
      try { const { job } = await api("/api/ml/compare-strategies", body); S.result = await Q.waitJob(job, (v) => { status.textContent = `${v.message} (${Math.round(v.elapsed)} s)`; }, 1500); S.busy = false; draw(); }
      catch (e) { S.busy = false; go.disabled = false; status.textContent = ""; h("div", "banner err", card, e.message); }
    });
  }

  function results(out, r) {
    const kindLabel = { model: "Model", baseline: "Baseline", control: "Random-signal control" };
    h("p", "hint", out, `${r.period.days} common days, ${r.period.from} to ${r.period.to}. Position rule: ${r.design.position_rule}; cost ${r.design.cost_bps} bp one way; lag ${r.design.lag} day(s).`);
    const tc = h("div", "card", out); tc.dataset.role = "compare-trading"; h("h3", "", tc, "Trading quality (net of costs)");
    Q.simpleTable(tc, [{ label: "Strategy", get: (x) => x.name }, { label: "Kind", get: (x) => kindLabel[x.kind] }, { label: "CAGR", num: true, get: (x) => pct(x.cagr, 1) }, { label: "Sharpe", num: true, get: (x) => num(x.sharpe, 2) }, { label: "Sortino", num: true, get: (x) => num(x.sortino, 2) },
      { label: "Max drawdown", num: true, get: (x) => pct(x.max_drawdown, 0) }, { label: "Info ratio vs equal weight", num: true, get: (x) => num(x.information_ratio, 2) }, { label: "Beta", num: true, get: (x) => num(x.beta, 2) }, { label: "Active over beta-matched", num: true, get: (x) => pct(x.active_over_beta_matched, 1) },
      { label: "Turnover / yr", num: true, get: (x) => (x.ann_turnover === undefined ? "" : num(x.ann_turnover, 1) + "x") }, { label: "Cost drag / yr", num: true, get: (x) => (x.cost_drag === undefined ? "" : pct(x.cost_drag, 2)) },
      { label: "Net exposure", num: true, get: (x) => (x.net_exposure === undefined ? "" : pct(x.net_exposure, 0)) }], r.rows);
    const models = r.rows.filter((x) => x.kind === "model"), pc = h("div", "card", out); pc.dataset.role = "compare-predictive"; h("h3", "", pc, "Predictive quality (does the score order the assets?)");
    h("p", "note", pc, "Rank IC is the daily rank correlation of the score with what followed. Calibration slope of 1 means the score is, on average, the return that followed it (meaningful for a return forecast). The error of the score is shown beside the error of a forecast of nothing: a model that does not beat it has no skill as a forecaster even if it ranks.");
    Q.simpleTable(pc, [{ label: "Model", get: (x) => x.name }, { label: "Rank IC", num: true, get: (x) => num(x.predictive.rank_ic, 4) }, { label: "IC information ratio", num: true, get: (x) => num(x.predictive.ic_ir, 2) }, { label: "Days with positive IC", num: true, get: (x) => pct(x.predictive.ic_hit_rate, 0) },
      { label: "Calibration slope", num: true, get: (x) => num(x.predictive.calibration_slope, 2) }, { label: "MAE of score", num: true, get: (x) => num(x.predictive.mae_score, 4) }, { label: "MAE of a forecast of 0", num: true, get: (x) => num(x.predictive.mae_zero, 4) },
      { label: "MSE of score", num: true, get: (x) => num(x.predictive.mse_score, 5) }, { label: "MSE of a forecast of 0", num: true, get: (x) => num(x.predictive.mse_zero, 5) }], models);
    const eq = r.equity, names = Object.keys(eq).filter((k) => k !== "dates"), col = ["--s1", "--s6", "--s3", "--s4", "--s2", "--s5", "--s7", "--s8"].map(css);
    Q.chartCard(out, "Equity, all strategies on the same days", "Growth of 1 after costs. Click a name in the legend to hide it. The random-signal line is what no skill earns at this pace of trading.", {
      wide: true, draw: (b) => C.line(b, { dates: eq.dates, series: names.map((k, i) => ({ name: k, color: col[i % col.length], values: eq[k], dash: r.rows.find((x) => x.name === k && x.kind !== "model") ? "5 4" : undefined })), yFormat: (v) => num(v, 2), height: 300, label: "Equity", baseline: 1 }),
      table: (b) => Q.simpleTable(b, [{ label: "Date", get: (x) => x.d }, ...names.map((k) => ({ label: k, num: true, get: (x) => num(x[k], 3) }))], eq.dates.map((d, i) => ({ d, ...Object.fromEntries(names.map((k) => [k, eq[k][i]])) })).filter((_, i) => i % 4 === 0)) });
    for (const n of r.notes) h("p", "hint", out, n);
  }

  Q.registerPage("modelcompare", { render: () => draw() });
})();
