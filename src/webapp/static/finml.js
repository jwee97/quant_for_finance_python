/* The Financial ML Lab: after Lopez de Prado's "Advances in Financial Machine Learning".

   One pipeline on one ticker's daily prices: CUSUM events, triple-barrier labels with a primary side, uniqueness-aware weights, purged and combinatorial cross-validation, a meta-model that says whether to take each
   signal, probability-based sizing, a cost-aware backtest and the evidence for it. The page is built to say what the evidence does and does not show: the last part of the history was never fitted on, the filtered
   strategy is set against the strategy it filters, a random filter and buy-and-hold, and a result that does not beat them is reported as a negative result. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, toast, css } = Q, C = Q.C;
  const num = (v, d = 2) => (v === null || v === undefined ? "n/a" : Q.num(v, d)), pct = (v, d = 1) => (v === null || v === undefined ? "n/a" : Q.pct(v, d));
  const root = $("#finml");
  const MAIN = ["primary", "lookback", "fast", "cusum_k", "pt", "sl", "vertical_bars", "entry_lag", "cost_bps", "holdout", "threshold", "n_trials"];
  const S = { form: { ticker: "", start: "", source: "", use_range: false, config: {} }, plan: null, result: null, structure: null, busy: false };
  const LABEL = { primary: "Primary strategy (every signal)", meta_filter: "Meta-labelled: take or skip", meta_sized: "Meta-labelled: sized by probability", random_filter: "Random filter (same number skipped)", buy_and_hold: "Buy and hold" };

  function defaults() {
    const c = (Q.research && Q.research.active() && Q.research.active().context) || {}, setup = Q.setupContext ? Q.setupContext() : {};
    S.form.ticker = S.form.ticker || ((c.assets && c.assets[0]) || (setup.assets && setup.assets[0]) || "SPY");
    S.form.start = S.form.start || c.start || setup.start || ""; S.form.source = S.form.source || c.data_source || setup.data_source || "";
  }

  async function draw() {
    root.replaceChildren();
    const d = h("div", "head", root), l = h("div", "", d); h("h2", "", l, "Financial ML Lab");
    h("p", "sub", l, "Event sampling, triple-barrier labels, meta-labelling, purged validation and the evidence for a result, on one ticker's daily prices.");
    if (!S.plan) { try { S.plan = await api("/api/finml/plan"); } catch (e) { h("div", "banner err", root, "The Financial ML service is not available: " + e.message + " (restart the app if it was just updated)"); return; } for (const [k, v] of Object.entries(S.plan.defaults)) if (!(k in S.form.config)) S.form.config[k] = v; }
    form();
    const out = h("div", "stack", root); out.id = "finml-results"; out.setAttribute("aria-live", "polite");
    if (S.result) results(out, S.result); else h("div", "card empty", out).append(h("h3", "", null, "No run yet"), h("p", "", null, "Choose a ticker and run the pipeline. A few seconds to half a minute for ten years of daily data."));
    if (S.structure) structure(out, S.structure);
  }

  function form() {
    const card = h("div", "card", root); card.dataset.role = "finml-form"; h("h3", "", card, "Run the pipeline");
    const f = S.form, grid = h("div", "params", card);
    const field = (parent, spec) => {
      const key = spec.key, choices = S.plan.choices[key], l = h("label", "f", parent, spec.label); l.title = spec.help;
      if (choices) { const s = h("select", "", l); s.dataset.key = key; for (const c of choices) { const o = h("option", "", s, String(c)); o.value = c; if (f.config[key] === c) o.selected = true; } s.addEventListener("change", () => { f.config[key] = s.value; }); }
      else if (key === "importance") { l.className = "f inline"; l.textContent = ""; const cb = h("input", "", l); cb.type = "checkbox"; cb.dataset.key = key; cb.checked = !!f.config[key]; l.append(document.createTextNode(spec.label)); cb.addEventListener("change", () => { f.config[key] = cb.checked; }); }
      else { const i = h("input", "", l); i.type = "number"; i.step = "any"; i.value = f.config[key]; i.dataset.key = key; i.addEventListener("input", () => { f.config[key] = i.value === "" ? "" : +i.value; }); }
      h("span", "hint", l, spec.help);
    };
    const text = (label, key, type) => { const l = h("label", "f", grid, label), i = h("input", "", l); i.type = type || "text"; i.value = f[key]; i.dataset.key = key; i.addEventListener("input", () => { f[key] = i.value; }); };
    text("Ticker", "ticker"); text("First day", "start", "date");
    const sl = h("label", "f", grid, "Data source"), src = h("select", "", sl); src.dataset.key = "source";
    for (const s of ["", ...(Q.S.catalog ? Q.S.catalog.sources.map((x) => x.name) : [])]) { const o = h("option", "", src, s || "(Backtest tab's source)"); o.value = s; if (s === f.source) o.selected = true; } src.addEventListener("change", () => { f.source = src.value; });
    const by = Object.fromEntries(S.plan.settings.map((s) => [s.key, s]));
    for (const k of MAIN) field(grid, by[k]);
    const more = h("details", "wide", card); h("summary", "", more, "More settings"); const mg = h("div", "params", more);
    for (const s of S.plan.settings) if (!MAIN.includes(s.key) && s.key !== "same_bar") field(mg, s);
    const rg = h("label", "f inline wide", mg), rc = h("input", "", rg); rc.type = "checkbox"; rc.checked = f.use_range; rg.append(document.createTextNode("Use each day's high and low for the barriers (a day that reaches both is decided by the assumption below)")); rc.addEventListener("change", () => { f.use_range = rc.checked; });
    field(mg, by.same_bar);
    const row = h("div", "row", card), run = h("button", "btn primary", row, "Run the pipeline"); run.type = "button"; run.disabled = S.busy;
    const st = h("button", "btn", row, "Structure and entropy"); st.type = "button"; st.disabled = S.busy; st.title = "Supremum ADF, a CUSUM test on levels and entropy of the encoded returns, for the same ticker";
    const status = h("span", "hint", row); status.id = "finml-status"; status.setAttribute("role", "status");
    for (const n of S.plan.notes) h("p", "hint", card, n);
    run.addEventListener("click", () => start("/api/finml/run", run, status, true)); st.addEventListener("click", () => start("/api/finml/structure", st, status, false));
  }

  async function start(path, btn, status, pipeline) {
    const f = S.form, body = { ticker: (f.ticker || "").trim(), start: f.start || undefined, source: f.source || undefined };
    if (pipeline) { body.config = { ...f.config }; body.use_range = f.use_range; for (const k of Object.keys(body.config)) if (body.config[k] === "" || body.config[k] === null) delete body.config[k]; }
    S.busy = true; btn.disabled = true; status.textContent = "Starting…";
    try {
      const { job } = await api(path, body);
      const r = await Q.waitJob(job, (v) => { status.textContent = `${v.message} (${Math.round(v.elapsed)} s)`; }, 1000);
      if (pipeline) S.result = r; else S.structure = r;
    } catch (e) { S.busy = false; status.textContent = ""; toast(e.message); h("div", "banner err", $("#finml-results") || root, e.message); btn.disabled = false; return; }
    S.busy = false; draw();
  }

  /* ---------------------------------------------------------------- results */
  const sh = (v) => num(v, 2);
  function results(out, r) {
    const dev = r.development, hold = r.holdout, shown = hold || dev, pf = (x) => x.meta_filter, vd = r.validation, cp = r.cpcv;
    const top = h("div", "card", out); top.dataset.role = "finml-verdict";
    const hd = h("div", "row", top); h("h3", "", hd, `${r.ticker}: ${r.meta_labelling_demonstrated ? "the filter helped" : "the filter was not shown to help"}`); h("span", "ev wf", hd, "Cross-validated and held out"); h("span", "ev is", hd, "Research evidence");
    h("div", r.meta_labelling_demonstrated ? "banner" : "banner warn", top, r.verdict);
    for (const w of r.warnings) h("div", "banner", top, w);
    const k = h("div", "kpis", top);
    Q.tile(k, "Sharpe, primary (development)", sh(dev.primary.sharpe), { text: "every signal taken" }); Q.tile(k, "Sharpe, filtered (development)", sh(dev.meta_filter.sharpe), { text: `${dev.meta_filter.taken} of ${dev.meta_filter.signals} signals`, dir: dev.meta_filter.sharpe > dev.primary.sharpe ? "up" : "down" });
    if (hold) { Q.tile(k, "Sharpe, primary (holdout)", sh(hold.primary.sharpe), { text: `${hold.window.start} to ${hold.window.end}` }); Q.tile(k, "Sharpe, filtered (holdout)", sh(hold.meta_filter.sharpe), { text: `${hold.meta_filter.taken} of ${hold.meta_filter.signals} signals`, dir: hold.meta_filter.sharpe > hold.primary.sharpe ? "up" : "down" }); }
    Q.tile(k, "Meta-model AUC (out of sample)", num(vd.auc, 3), { text: "0.5 is no skill" }); Q.tile(k, "Log loss against base rate", `${num(vd.log_loss, 3)} vs ${num(vd.base_rate_log_loss, 3)}`, { text: vd.log_loss < vd.base_rate_log_loss ? "better than guessing the base rate" : "no better than the base rate", dir: vd.log_loss < vd.base_rate_log_loss ? "up" : "down" });
    Q.tile(k, "Independent labels", num(r.quality.effective_sample_size, 0), { text: `of ${r.quality.events} events (mean uniqueness ${num(r.quality.mean_uniqueness, 2)})` });
    Q.tile(k, "CPCV paths", `${cp.n_paths}`, { text: `Sharpe ${num(cp.mean)} ± ${num(cp.std)}; negative in ${pct(cp.prob_negative, 0)}` });
    const ev = (hold || dev).evidence.meta_filter; Q.tile(k, "Probabilistic Sharpe (filtered)", pct(ev.psr, 0), { text: "chance the true Sharpe exceeds 0" }); Q.tile(k, "Deflated Sharpe (filtered)", pct(ev.dsr, 0), { text: `for ${r.config.n_trials} configuration(s) tried` });
    for (const n of r.notes) h("p", "hint", top, n);
    evidence(out, r);
    const colors = [css("--s1"), css("--s2"), css("--s3"), css("--ink-2")];
    const eq = (title, note, block, role) => {
      const s = r.series[block], names = ["primary", "meta_filter", "meta_sized", "buy_and_hold"].filter((n) => s[n]); if (!names.length) return;
      const dates = s[names[0]].dates, card = Q.chartCard(out, title, note, { wide: true, draw: (b) => C.line(b, { dates, series: names.map((n, i) => ({ name: LABEL[n], color: colors[i], values: s[n].values })), baseline: 1, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), label: title, height: 280 }),
        table: (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => dates[i] }, ...names.map((n) => ({ label: LABEL[n], num: true, get: (i) => num(s[n].values[i], 3) }))], Q.sampleRows(dates, null, Math.max(1, Math.ceil(dates.length / 60)))) });
      card.dataset.role = role;
    };
    eq("Growth of 1 on the development window", "Out-of-sample probabilities from purged folds: each signal was scored by a model that had not seen it or its overlapping neighbours.", "development", "finml-equity-dev");
    if (hold) eq("Growth of 1 on the untouched holdout", "Nothing was fitted, tuned or chosen on these days. They were scored once, after everything else.", "holdout", "finml-equity-hold");
    comparison(out, "Strategies on the development window", dev, "finml-table-dev"); if (hold) comparison(out, "Strategies on the untouched holdout", hold, "finml-table-hold");
    const two = h("div", "charts", out);
    const ps = cp.sharpes.map((v, i) => ({ v, i })).filter((x) => x.v !== null);
    const cc = Q.chartCard(two, "Out-of-sample Sharpe across CPCV paths", `Each bar is one recombined path of the filtered strategy; the primary strategy's development Sharpe is ${num(cp.primary_sharpe)}. A distribution says more than the best path does.`, { draw: (b) => C.columns(b, { labels: ps.map((x) => `P${x.i + 1}`), values: ps.map((x) => x.v), yFormat: (v) => v.toFixed(1), label: "Sharpe by CPCV path", height: 220 }), table: (b) => Q.simpleTable(b, [{ label: "Path", get: (x) => `P${x.i + 1}` }, { label: "Sharpe", num: true, get: (x) => num(x.v) }], ps) });
    cc.dataset.role = "finml-cpcv";
    const rel = vd.reliability;
    const rc = Q.chartCard(two, "Calibration of the meta-model", "Predicted probability that the signal pays against how often it did, in eight equal-count groups. On the diagonal is calibrated.", { draw: (b) => C.line(b, { xs: rel.map((x) => x.predicted), series: [{ name: "Observed", color: colors[0], values: rel.map((x) => x.observed), dots: true }, { name: "Perfectly calibrated", color: colors[3], dash: "4 4", values: rel.map((x) => x.predicted) }], xFormat: (v) => v.toFixed(2), yFormat: (v) => v.toFixed(2), xLabel: "Predicted probability", label: "Calibration", height: 220, zoom: false }),
      table: (b) => Q.simpleTable(b, [{ label: "Predicted", num: true, get: (x) => num(x.predicted, 3) }, { label: "Observed", num: true, get: (x) => num(x.observed, 3) }, { label: "Signals", num: true, get: (x) => String(x.n) }], rel) });
    rc.dataset.role = "finml-calibration";
    quality(out, r);
    if (r.importance) importance(out, r.importance);
    const refs = h("details", "card", out); h("summary", "", refs, "How this was made"); const dl = h("dl", "kv", refs);
    for (const [a, b2] of [["Settings", JSON.stringify(r.config)], ["Data", `${r.ticker}, ${r.universe.first} to ${r.universe.last}, ${r.universe.days} days (${r.universe.source}); ${r.universe.range_used ? "daily high and low used" : "closes only"}`], ["Split", `${r.split.development_events} development events, ${r.split.holdout_events} holdout events, ${r.split.dropped_for_overlap} dropped because their labels straddled the boundary; holdout starts ${r.split.holdout_starts || "(none)"}`],
      ["Labels", `${r.labels.events} labelled events (${Object.entries(r.labels.barrier).map(([a2, b3]) => `${b3} ${a2}`).join(", ")}); mean holding ${num(r.labels.mean_holding_bars, 1)} days`]]) { h("dt", "", dl, a); h("dd", "", dl, b2); }
  }

  function evidence(out, r) {
    const card = h("div", "card", out); card.dataset.role = "finml-evidence"; h("h3", "", card, "Research evidence"); h("p", "sub", card, "Each line says what was checked and what was not. \"Review\" and \"Not completed\" are not failures: they are what is still owed.");
    const hold = r.holdout, rows = [
      ["Purging", r.validation.purging_leak_free ? "Passed" : "FAILED", r.validation.purging_leak_free ? "no training label overlaps a test label in any fold" : "a training label overlapped a test label"],
      ["Holdout untouched", hold ? "Passed" : "Not applicable", hold ? `nothing fitted or chosen on ${r.split.holdout_starts} onward; labels straddling the boundary were dropped` : "no holdout was set aside"],
      ["Overlapping labels", r.quality.effective_sample_size < 0.5 * r.quality.events ? "Review" : "Passed", `${r.quality.events} events are worth about ${num(r.quality.effective_sample_size, 0)} independent ones`],
      ["Execution assumptions", r.config.entry_lag === 0 ? "Review" : "Passed", r.config.entry_lag === 0 ? "entry at the close that produced the signal is optimistic" : `entry ${r.config.entry_lag} day(s) after the signal, cost ${r.config.cost_bps} bp per unit traded`],
      ["Barrier ordering", r.universe.range_used ? (r.labels.ambiguous_share > 0.02 ? "Review" : "Passed") : "Review", r.universe.range_used ? `${pct(r.labels.ambiguous_share, 1)} of labels ended on a bar that touched both barriers (${r.config.same_bar} assumed)` : "closes only: a barrier crossed and recrossed inside a day is not seen"],
      ["Multiple-testing adjustment", r.config.n_trials > 1 ? "Available" : "Review", r.config.n_trials > 1 ? `deflated for ${r.config.n_trials} configurations tried` : "the deflated Sharpe assumes this was the only configuration tried"],
      ["Distribution of paths (CPCV)", "Available", `${r.cpcv.n_paths} paths; ${r.cpcv.note}`],
      ["Prospective test", "Not completed", "only forward, chronological testing on data that did not exist when the settings were chosen can supply it"],
      ["Independent replication", "Not completed", "another ticker, period or data source has not been run"]];
    Q.simpleTable(card, [{ label: "Check", get: (x) => x[0] }, { label: "Status", get: (x) => x[1] }, { label: "Detail", get: (x) => x[2] }], rows);
  }

  function comparison(out, title, block, role) {
    const card = h("div", "card", out); card.dataset.role = role; h("h3", "", card, title);
    const names = ["primary", "meta_filter", "meta_sized"], rows = names.map((n) => ({ n, ...block[n], ev: block.evidence[n] }));
    rows.push({ n: "random_filter", sharpe: block.random_filter.sharpe, taken: block.random_filter.taken, signals: block.random_filter.signals, note: `5th to 95th percentile of 40 random filters: ${num(block.random_filter.sharpe_p05)} to ${num(block.random_filter.sharpe_p95)}` });
    rows.push({ n: "buy_and_hold", ...block.buy_and_hold });
    Q.simpleTable(card, [{ label: "Strategy", get: (x) => LABEL[x.n] }, { label: "Sharpe", num: true, get: (x) => sh(x.sharpe) }, { label: "Return (annual)", num: true, get: (x) => pct(x.annual_return) }, { label: "Volatility", num: true, get: (x) => pct(x.annual_vol) }, { label: "Worst drawdown", num: true, get: (x) => pct(x.max_drawdown) },
      { label: "Signals taken", num: true, get: (x) => (x.taken === undefined ? "" : `${x.taken} of ${x.signals}`) }, { label: "Turnover / day", num: true, get: (x) => (x.turnover_per_day === undefined ? "" : num(x.turnover_per_day, 3)) }, { label: "PSR", num: true, get: (x) => (x.ev ? pct(x.ev.psr, 0) : "") }, { label: "DSR", num: true, get: (x) => (x.ev ? pct(x.ev.dsr, 0) : "") }], rows);
    const c = block.comparison;
    h("p", "hint", card, `Filtered minus primary Sharpe: ${num(c.filter_minus_primary_sharpe)}. Block bootstrap of the daily difference: positive in ${pct(c.bootstrap.p_positive, 0)} of resamples, annualised difference ${pct(c.bootstrap.low)} to ${pct(c.bootstrap.high)}. ${c.beats_random_filters_p95 ? "It beat 95% of the random filters." : "It did not clear the best of the random filters, so selection skill is not shown."}`);
  }

  function quality(out, r) {
    const q = r.quality, card = Q.chartCard(out, "Label overlap: how unique each label is", `Average uniqueness is 1 for a label that shares its days with no other and about 1 divided by the overlap for one that shares them all. Mean concurrency ${num(q.mean_concurrency, 1)}; the most overlapped day had ${q.max_concurrency} labels alive.`,
      { draw: (b) => C.columns(b, { labels: q.uniqueness_deciles.map((_, i) => `${i * 10}%`), values: q.uniqueness_deciles, yFormat: (v) => v.toFixed(2), label: "Uniqueness by decile", height: 200 }), table: (b) => Q.simpleTable(b, [{ label: "Percentile", get: (i) => `${i * 10}%` }, { label: "Uniqueness", num: true, get: (i) => num(q.uniqueness_deciles[i], 3) }], q.uniqueness_deciles.map((_, i) => i)) });
    card.dataset.role = "finml-quality"; h("p", "hint", card, q.note);
  }

  function importance(out, imp) {
    const card = h("div", "card", out); card.dataset.role = "finml-importance"; h("h3", "", card, "Feature importance (development events)"); h("p", "sub", card, "MDI explains the training sample; MDA shuffles a feature, or a cluster of correlated ones, out of sample in purged folds; SFI scores each feature alone. A feature that helps in some folds and hurts in others is not reliable.");
    const mda = imp.mda.filter((x) => x.mean !== null);
    const mc = Q.chartCard(card, "MDA by cluster: score lost when shuffled", "Higher means the model leans on it out of sample. The bar's note says the share of folds in which shuffling hurt.", { draw: (b) => C.hbars(b, { rows: mda.map((x) => ({ label: x.members.length > 28 ? x.members.slice(0, 26) + "…" : x.members, value: x.mean, tip: [["Folds in which it helped", pct(x.positive, 0)], ["Spread across folds", num(x.std, 4)], ["Members", x.members]] })), format: (v) => v.toFixed(4), name: "Score lost", label: "MDA by cluster", labelWidth: 170, valueWidth: 70, color: (v) => (v >= 0 ? css("--s1") : css("--neg")) }),
      table: (b) => Q.simpleTable(b, [{ label: "Features", get: (x) => x.members }, { label: "Mean", num: true, get: (x) => num(x.mean, 4) }, { label: "Spread", num: true, get: (x) => num(x.std, 4) }, { label: "Folds helped", num: true, get: (x) => pct(x.positive, 0) }], mda) });
    mc.dataset.role = "finml-mda";
    Q.simpleTable(card, [{ label: "Feature", get: (x) => x.feature }, { label: "MDI", num: true, get: (x) => pct(x.mean, 1) }, { label: "SFI gain over base rate", num: true, get: (x) => num((imp.sfi.find((s) => s.feature === x.feature) || {}).gain, 4) }, { label: "SFI folds better", num: true, get: (x) => pct((imp.sfi.find((s) => s.feature === x.feature) || {}).positive, 0) }], imp.mdi);
  }

  function structure(out, r) {
    const card = h("div", "card", out); card.dataset.role = "finml-structure"; h("h3", "", card, `Structure and entropy: ${r.ticker}, ${r.first} to ${r.last}`);
    const k = h("div", "kpis", card);
    Q.tile(k, "SADF now", num(r.sadf.last), { text: `95% value ${num(r.sadf.critical_95)}`, dir: r.sadf.last !== null && r.sadf.last > r.sadf.critical_95 ? "down" : "" }); Q.tile(k, "Days above the 95% value", pct(r.sadf.exceeds_share, 0), { text: "of days with a reading" });
    Q.tile(k, "Level-shift flags", pct(r.cusum.break_share, 1), { text: r.cusum.last_break ? `latest ${r.cusum.last_break}` : "none" });
    const e = r.entropy.whole_sample; Q.tile(k, "Entropy of signs (bits)", num(e.sign.plug_in_bits_per_symbol, 3), { text: "1.0 is a fair coin" }); Q.tile(k, "Lempel-Ziv, quartiles", num(e.quartiles.lempel_ziv, 2), { text: "about 1 for a random sequence" });
    const sd = r.sadf.dates, sc = Q.chartCard(card, "Supremum ADF statistic", r.sadf.note, { wide: true, draw: (b) => C.line(b, { dates: sd, series: [{ name: "SADF", color: css("--s1"), values: r.sadf.values }, { name: "95% value (simulated)", color: css("--s2"), dash: "4 4", values: sd.map(() => r.sadf.critical_95) }], yFormat: (v) => v.toFixed(1), label: "SADF", height: 220 }),
      table: (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => sd[i] }, { label: "SADF", num: true, get: (i) => num(r.sadf.values[i], 2) }], sd.map((_, i) => i)) });
    sc.dataset.role = "finml-sadf";
    h("p", "hint", card, r.entropy.note);
  }

  Q.registerPage("finml", { render: () => { if (!S.form.ticker) defaults(); draw(); } });
})();
