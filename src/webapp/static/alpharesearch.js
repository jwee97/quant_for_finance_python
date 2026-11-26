/* Alpha research, in five stages that hand one sample to the next: Discovery (is there enough data to rank, and are the signals distinct?), Validation (does each factor survive being one of many tried, and what is left
   once the market and momentum are taken out?), Combination (weighted from the past only, judged on a period nothing was fitted on), Portfolio construction (optimisers compared on the same windows, costs and caps,
   with concentration, stability and sensitivity) and Performance and attribution (what the paper alpha keeps once it is traded).

   Stages 1 to 3 are one server run because they read the same factor panels. Every result remembers the sample and settings it was run on (tickers, first day, data source, horizon, signals, costs). When the form no
   longer matches, the result is called stale and is not drawn next to results of the current sample, so two stages are never read as one study when they are not. This file adds no estimator: it draws what
   src/alphalab/factors.py, src/alphalab/studies.py and src/portfolio/lab.py return. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, toast, css } = Q, C = Q.C;
  const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.num(v, d)), pct = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.pct(v, d));
  const STAGES = [["discovery", "Discovery"], ["validation", "Validation"], ["combination", "Combination"], ["portfolio", "Portfolio construction"], ["performance", "Performance and attribution"]];
  const MIN_TICKERS = 6, MIN_RANK = 3;
  const DEFAULT_METHODS = ["equal_weight", "inverse_vol", "min_variance", "max_sharpe", "risk_parity", "hrp"];
  const R = { stage: "discovery", signals: null, methods: new Set(DEFAULT_METHODS), perf: { signal: "", cost_bps: 5, long_only: true }, results: {}, keys: {}, busy: {}, jobs: {} };
  const colors = () => ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"].map(css);

  /* What identifies a sample, and what identifies a run: the second contains the first. A result is current only when its run key is the form's. */
  const tickersOf = (S) => S.form.tickers.split(/[\s,;]+/).filter(Boolean);
  const sampleOf = (S) => ({ tickers: tickersOf(S), start: S.form.start || "", source: S.form.source || "" });
  const sig = (S) => (R.signals || S.plan.signals.map((s) => s.key)).slice().sort();
  const runKey = {
    factors: (S) => JSON.stringify({ ...sampleOf(S), horizon: +S.form.horizon, signals: sig(S) }),
    portfolio: (S) => JSON.stringify({ ...sampleOf(S), methods: [...R.methods].sort(), config: S.form.config, cov: S.form.cov_method || "" }),
    performance: (S) => JSON.stringify({ ...sampleOf(S), horizon: +S.form.horizon, ...R.perf }),
  };
  const sampleKey = (S) => JSON.stringify(sampleOf(S));
  const current = (S, name) => !!R.results[name] && R.keys[name] === runKey[name](S);
  const sameSample = (S, name) => !!R.results[name] && R.samples[name] === sampleKey(S);
  R.samples = {};
  const state = (S, name) => (!R.results[name] ? "not run" : current(S, name) ? "current" : sameSample(S, name) ? "settings changed" : "different sample");

  /* The guard the server enforces, said before the run: a factor study needs six assets, and a rank correlation needs three with a value on the date (a date with fewer is skipped, not scored). */
  const guard = (S) => {
    const n = tickersOf(S).length;
    if (n < MIN_TICKERS) return `A factor study needs at least ${MIN_TICKERS} tickers; ${n} given. With fewer, the cross-section is too thin to rank (a rank correlation needs at least ${MIN_RANK} assets with a value on a date, and dates with fewer are skipped, never scored as zero).`;
    return "";
  };

  function render(out, S) {
    const box = h("div", "stack", out); box.dataset.role = "alpha-research";
    context(box, S);
    const g = guard(S);
    if (g) { const b = h("div", "banner warn", box, g); b.dataset.role = "breadth-guard"; b.setAttribute("role", "alert"); }
    const nav = h("div", "seg stages", box); nav.setAttribute("role", "group"); nav.setAttribute("aria-label", "Research stage");
    const need = { discovery: "factors", validation: "factors", combination: "factors", portfolio: "portfolio", performance: "performance" };
    for (const [k, label] of STAGES) {
      const st = state(S, need[k]), b = h("button", "", nav, `${STAGES.findIndex((x) => x[0] === k) + 1}. ${label}`); b.type = "button"; b.dataset.stage = k; b.dataset.state = st; b.setAttribute("aria-pressed", String(R.stage === k));
      b.title = st === "current" ? "Run on the current sample and settings" : st === "not run" ? "Not run yet" : st === "settings changed" ? "Run on this sample with other settings" : "Run on a different sample: not comparable with the form";
      if (st !== "current" && st !== "not run") b.classList.add("stale");
      b.addEventListener("click", () => { R.stage = k; S.redraw(); });
    }
    const pane = h("div", "stack", box); pane.dataset.role = "stage-" + R.stage; pane.setAttribute("aria-live", "polite");
    ({ discovery, validation, combination, portfolio, performance })[R.stage](pane, S);
  }

  /* The sample and assumptions every stage is read under, in one place. */
  function context(box, S) {
    const c = h("div", "card ctxstrip", box); c.dataset.role = "research-context";
    const s = sampleOf(S), f = R.results.factors, p = R.results.portfolio;
    const bits = [["Assets", s.tickers.length ? `${s.tickers.length}` : "none"], ["From", s.start || "first available"], ["Source", s.source || "backtest tab's source"], ["Horizon", `${S.form.horizon} days`],
      ["Signals", `${sig(S).length}`], ["Cost", `${R.perf.cost_bps} bp (trading) / ${(p && p.config && p.config.cost_bps) ?? S.form.config.cost_bps ?? "default"} bp (optimisers)`], ["Data", f && f.universe && f.universe.source ? f.universe.source : "not loaded"]];
    const row = h("div", "kv", c); for (const [k, v] of bits) { const d = h("span", "", row); h("b", "", d, k + ": "); d.append(document.createTextNode(String(v))); }
    h("p", "hint", c, "Every stage below is run on this sample. Change the tickers, first day, source or horizon and the earlier results are marked stale instead of being drawn next to the new ones.");
  }

  function banner(pane, S, name, what) {
    const st = state(S, name);
    if (st === "not run") return false;
    if (st === "current") return true;
    const b = h("div", "banner warn", pane, st === "different sample" ? `This ${what} was run on a different sample (${sampleLabel(R.samples[name])}) from the form's. It is not shown, so that it is not read with results of the current sample. Run it again to continue.`
      : `This ${what} was run with other settings than the form's now has. It is shown as it was run, and is not comparable with anything run under the new settings until it is run again.`);
    b.dataset.role = "stale-" + name; b.setAttribute("role", "alert");
    return st !== "different sample";
  }
  const sampleLabel = (k) => { try { const s = JSON.parse(k); return `${s.tickers.length} assets from ${s.start || "the first day"}${s.source ? ", " + s.source : ""}`; } catch { return "unknown"; } };

  async function run(S, name, path, body, btn, status) {
    if (R.busy[name]) return;
    R.busy[name] = true; btn.disabled = true; status.textContent = "Starting…";
    try {
      const { job } = await api(path, body);
      R.jobs[name] = job;
      const res = await Q.waitJob(job, (v) => { status.textContent = `${v.message} (${Math.round(v.elapsed)} s)`; }, 1000);
      R.results[name] = res; R.keys[name] = runKey[name](S); R.samples[name] = sampleKey(S);
    } catch (e) { toast(e.message); R.busy[name] = false; status.textContent = ""; btn.disabled = false; h("div", "banner err", $("#lab-results") || document.body, e.message); return; }
    R.busy[name] = false; S.redraw();
  }

  function runRow(pane, S, name, label, path, bodyFn, extraGuard) {
    const row = h("div", "row", pane), btn = h("button", "btn primary", row, label); btn.type = "button"; btn.dataset.role = "run-" + name;
    const status = h("span", "hint", row); status.setAttribute("role", "status"); status.id = "research-status";
    const g = extraGuard || guard(S); if (g) { btn.disabled = true; btn.title = g; }
    btn.addEventListener("click", () => run(S, name, path, bodyFn(), btn, status));
    return row;
  }
  const baseBody = (S) => ({ ...sampleOf(S), start: S.form.start || undefined, source: S.form.source || undefined });
  const factorBody = (S) => ({ ...baseBody(S), signals: sig(S).sort((a, b) => S.plan.signals.findIndex((x) => x.key === a) - S.plan.signals.findIndex((x) => x.key === b)), horizon: +S.form.horizon });

  /* ------------------------------------------------------------------ 1 Discovery */
  function signalChips(pane, S) {
    const box = h("div", "chips", pane); h("span", "hint", box, "Signals to compare:");
    for (const s of S.plan.signals) { const l = h("label", "f inline", box); l.title = s.help || ""; const cb = h("input", "", l); cb.type = "checkbox"; cb.dataset.signal = s.key; cb.checked = sig(S).includes(s.key);
      cb.addEventListener("change", () => { const cur = new Set(sig(S)); cb.checked ? cur.add(s.key) : cur.delete(s.key); R.signals = [...cur]; S.redraw(); }); l.append(document.createTextNode(s.label)); }
  }

  function discovery(pane, S) {
    const head = h("div", "card", pane); h("h3", "", head, "1. Discovery: is there enough to rank, and are the signals different from each other?");
    h("p", "hint", head, "Coverage, extreme values, drift between the first and second half of the history, how persistent each ranking is (so how much it will trade) and how redundant the signals are. Nothing here looks at returns, so nothing here can be overfitted to them.");
    signalChips(head, S);
    if (R.signals && R.signals.length < 2) h("p", "hint warn", head, "Choose at least two signals: multiple-testing corrections and combinations need more than one.");
    runRow(head, S, "factors", "Run the factor study (stages 1 to 3)", "/api/lab/factors", () => factorBody(S), guard(S) || (sig(S).length < 2 ? "Choose at least two signals." : ""));
    if (!banner(pane, S, "factors", "factor study")) { const e = h("div", "card empty", pane); h("p", "", e, "No factor study on this sample yet."); return; }
    const r = R.results.factors, d = r.discovery, rows = Object.entries(d.signals).map(([k, v]) => ({ key: k, ...v })), label = Object.fromEntries(S.plan.signals.map((s) => [s.key, s.label]));
    const thin = rows.filter((x) => !x.rankable);
    if (thin.length) { const b = h("div", "banner warn", pane, `${thin.length} signal${thin.length > 1 ? "s have" : " has"} too thin a cross-section to rank (${thin.map((x) => `${label[x.key] || x.key}: median ${x.n_assets_median} assets`).join("; ")}). They are listed but not scored.`); b.dataset.role = "thin-cross-section"; }
    const tc = h("div", "card", pane); tc.dataset.role = "disc-table"; h("h3", "", tc, `${r.assets.length} assets, ${r.first} to ${r.last}, ${r.days} days`);
    Q.simpleTable(tc, [{ label: "Signal", get: (x) => label[x.key] || x.key }, { label: "Median assets per date", num: true, get: (x) => String(x.n_assets_median) }, { label: "Rankable", get: (x) => (x.rankable ? "yes" : "no: " + (x.reason || "")) },
      { label: "Coverage (mean)", num: true, get: (x) => pct(x.coverage.by_date_mean, 0) }, { label: "Coverage (worst date)", num: true, get: (x) => pct(x.coverage.by_date_min, 0) }, { label: "Worst months (missing)", get: (x) => x.coverage.worst_months.map((m) => `${m.month} ${pct(m.missing, 0)}`).join(", ") || "none" },
      { label: "Beyond 4 sd", num: true, get: (x) => pct(x.extreme_share, 2) }, { label: "Drift p (1st vs 2nd half)", num: true, get: (x) => (x.drift ? (x.drift.p < 0.001 ? "<0.001" : num(x.drift.p, 3)) : "n/a") },
      { label: "Rank persistence 1 d / 21 d", num: true, get: (x) => `${num(x.rank_persistence.lag1, 2)} / ${num(x.rank_persistence.lag21, 2)}` }], rows);
    for (const n of r.notes.slice(0, 1)) h("p", "hint", tc, n);
    const names = d.correlation.names;
    if (names.length > 1) {
      const mc = Q.chartCard(pane, "How alike the signals are", `Average daily cross-sectional rank correlation between each pair. ${d.redundant.length ? "Redundant (|r| above " + num(0.7, 1) + "): " + d.redundant.map((p) => p.join(" & ")).join("; ") + "." : "No pair is redundant."}`,
        { draw: (b) => C.matrix(b, { rows: names.map((n) => label[n] || n), cols: names.map((n) => label[n] || n), values: d.correlation.matrix, diverging: true, min: -1, max: 1, format: (v) => v.toFixed(2), label: "Signal correlation" }),
          table: (b) => Q.simpleTable(b, [{ label: "", get: (i) => label[names[i]] || names[i] }, ...names.map((n, j) => ({ label: label[n] || n, num: true, get: (i) => num(d.correlation.matrix[i][j], 2) }))], names.map((_, i) => i)) });
      mc.dataset.role = "disc-corr";
    }
  }

  /* ------------------------------------------------------------------ 2 Validation */
  function validation(pane, S) {
    const head = h("div", "card", pane); h("h3", "", head, "2. Validation: does each factor survive being one of many?");
    h("p", "hint", head, "The same factors, each judged by its rank IC with a heteroskedasticity- and autocorrelation-robust interval, p-values corrected for the number compared (Holm for any false positive, Benjamini-Hochberg for the share of false discoveries), the IC left once beta and momentum are taken out, the IC by thirds of the history and under three clippings of outliers.");
    if (!banner(pane, S, "factors", "factor study")) { const e = h("div", "card empty", pane); h("p", "", e, "Run the factor study in stage 1 first."); const b = h("button", "btn", e, "Go to stage 1"); b.type = "button"; b.addEventListener("click", () => { R.stage = "discovery"; S.redraw(); }); return; }
    const r = R.results.factors, cmp = r.comparison, rows = cmp.table, c = colors(), label = Object.fromEntries(S.plan.signals.map((s) => [s.key, s.label])), nm = (k) => label[k] || k;
    const k = h("div", "kpis", head);
    Q.tile(k, "Factors tested", String(cmp.n_tested), { text: `family of ${cmp.n_tested}, level ${cmp.alpha}` });
    Q.tile(k, "Survive Holm", String(rows.filter((x) => x.survives_holm).length), { text: "family-wise error controlled", dir: rows.some((x) => x.survives_holm) ? "up" : "" });
    Q.tile(k, "Survive Benjamini-Hochberg", String(rows.filter((x) => x.survives_bh).length), { text: "false-discovery rate controlled", dir: rows.some((x) => x.survives_bh) ? "up" : "" });
    Q.tile(k, "Horizon", `${r.horizon} days`, { text: `${r.days} days, ${r.assets.length} assets` });
    h("p", "hint", head, cmp.note);
    const tc = h("div", "card", pane); tc.dataset.role = "val-table"; h("h3", "", tc, "Factors, corrected for the number tried");
    Q.simpleTable(tc, [{ label: "Factor", get: (x) => nm(x.factor) }, { label: "Rank IC", num: true, get: (x) => num(x.rank_ic, 4) }, { label: "95% interval (HAC)", num: true, get: (x) => `${num(x.ci_low, 4)} to ${num(x.ci_high, 4)}` }, { label: "t", num: true, get: (x) => num(x.t, 1) },
      { label: "p raw", num: true, get: (x) => num(x.p_raw, 3) }, { label: "p Holm", num: true, get: (x) => num(x.p_holm, 3) }, { label: "p BH", num: true, get: (x) => num(x.p_bh, 3) }, { label: "Survives", get: (x) => (x.survives_holm ? "Holm and BH" : x.survives_bh ? "BH only" : "neither") },
      { label: "Pearson IC", num: true, get: (x) => num(x.pearson_ic, 4) }, { label: "IC after beta and momentum", num: true, get: (x) => (x.nothing_left ? "nothing left" : num(x.ic_neutralised, 4)) }, { label: "Share kept", num: true, get: (x) => (x.share_kept === null ? "n/a" : pct(x.share_kept, 0)) },
      { label: "IC by thirds", num: true, get: (x) => (x.thirds || []).map((v) => num(v, 3)).join(" / ") }, { label: "Same sign in all thirds", get: (x) => (x.stable === 1 || x.stable === true ? "yes" : "no") }, { label: "Long-short / day", num: true, get: (x) => pct(x.long_short, 3) }, { label: "Turnover", num: true, get: (x) => num(x.turnover, 2) },
      { label: "Median assets", num: true, get: (x) => String(x.n_assets_median) }], rows);
    const ic = Q.chartCard(pane, "IC with its interval, and what neutralising leaves", "Dots: the rank IC with its 95% interval. The second series is the IC after the market beta and 12-1 momentum are regressed out of the signal each day. A factor that only repackages those is not new information.",
      { wide: true, draw: (b) => C.hbars(b, { rows: rows.map((x) => ({ label: nm(x.factor), value: x.rank_ic, from: 0, lo: x.ci_low, hi: x.ci_high, tip: [["After neutralising", x.nothing_left ? "nothing left" : num(x.ic_neutralised, 4)], ["p Holm", num(x.p_holm, 3)]] })), format: (v) => v.toFixed(3), name: "Rank IC", label: "Rank IC by factor", labelWidth: 120, valueWidth: 60, zero: true }),
        table: (b) => Q.simpleTable(b, [{ label: "Factor", get: (x) => nm(x.factor) }, { label: "Rank IC", num: true, get: (x) => num(x.rank_ic, 4) }, { label: "Neutralised IC", num: true, get: (x) => (x.nothing_left ? "nothing left" : num(x.ic_neutralised, 4)) }], rows) });
    ic.dataset.role = "val-ic";
    const sens = r.sensitivity.pearson_ic_by_winsorisation, names = Object.keys(sens), levels = names.length ? Object.keys(sens[names[0]]) : [];
    const sc = h("div", "card", pane); sc.dataset.role = "val-sensitivity"; h("h3", "", sc, "Does clipping the outliers change the answer?");
    h("p", "hint", sc, `The Pearson IC (the one outliers can move) with the signal clipped at 0, 2 and 5 percent in each tail each day. A factor whose IC changes sign or collapses here lives in a few extreme values. Neutralised against: ${r.sensitivity.controls.join(", ")}.`);
    Q.simpleTable(sc, [{ label: "Factor", get: (x) => nm(x) }, ...levels.map((l) => ({ label: l.replace("winsor_", "clip ").replace("pct", "%"), num: true, get: (x) => num(sens[x][l], 4) })), { label: "Moves with clipping", get: (x) => { const v = levels.map((l) => sens[x][l]).filter((y) => y !== null); return v.length && Math.max(...v) - Math.min(...v) > 0.5 * Math.max(Math.abs(v[0]), 0.002) ? "yes" : "no"; } }], names);
  }

  /* ------------------------------------------------------------------ 3 Combination */
  function combination(pane, S) {
    const head = h("div", "card", pane); h("h3", "", head, "3. Combination: is a blend better than its best part?");
    h("p", "hint", head, "Equal weight, IC-weighted, ridge and orthogonalised blends, with weights from the past only. The first part of the history chooses the best single factor; the later part, which nothing was fitted on, judges everything. A blend counts as an improvement only if it beats both the best single factor and equal weight by more than 1.96 standard errors of the difference.");
    if (!banner(pane, S, "factors", "factor study")) { const e = h("div", "card empty", pane); h("p", "", e, "Run the factor study in stage 1 first."); return; }
    const r = R.results.factors, cb = r.combination, label = Object.fromEntries(S.plan.signals.map((s) => [s.key, s.label])), nm = (k) => label[k] || k;
    const k = h("div", "kpis", head);
    Q.tile(k, "Training period", `${cb.train_dates[0]} to ${cb.train_dates[1]}`, { text: `best single factor on it: ${nm(cb.best_single_on_training)}` });
    Q.tile(k, "Test period", `${cb.test_dates[0]} to ${cb.test_dates[1]}`, { text: "nothing was fitted here" });
    const better = Object.entries(cb.verdicts).filter(([, v]) => v.improves_on_baselines).map(([m]) => m);
    Q.tile(k, "Blends that improve on the baselines", String(better.length), { text: better.length ? better.join(", ") : "none does, at 1.96 standard errors", dir: better.length ? "up" : "" });
    const rows = Object.entries(cb.out_of_sample).map(([key, v]) => ({ key, name: key.startsWith("single:") ? "Single: " + nm(key.slice(7)) : key.replace("_", " "), ...v, verdict: cb.verdicts[key] }));
    const tc = h("div", "card", pane); tc.dataset.role = "comb-table"; h("h3", "", tc, "Out-of-sample rank IC");
    Q.simpleTable(tc, [{ label: "Candidate", get: (x) => x.name }, { label: "Mean IC", num: true, get: (x) => num(x.mean, 4) }, { label: "95% interval", num: true, get: (x) => `${num(x.low, 4)} to ${num(x.high, 4)}` }, { label: "t", num: true, get: (x) => num(x.t, 1) }, { label: "Test dates", num: true, get: (x) => String(x.n_test_dates) },
      { label: "Beats best single by", num: true, get: (x) => (x.verdict ? `${num(x.verdict.best_single.difference, 4)} ± ${num(x.verdict.best_single.se, 4)}` : "") }, { label: "Beats equal weight by", num: true, get: (x) => (x.verdict && x.verdict.equal_weight ? `${num(x.verdict.equal_weight.difference, 4)} ± ${num(x.verdict.equal_weight.se, 4)}` : "") },
      { label: "Verdict", get: (x) => (x.verdict ? (x.verdict.improves_on_baselines ? "improves" : "does not improve") : "baseline") }], rows);
    h("p", "hint", tc, cb.note);
    const tr = cb.training_ic, cc = Q.chartCard(pane, "What the training period saw", "The mean rank IC of each factor on the training period only: how the best single factor was chosen, and what the IC-weighted blend leans on.",
      { draw: (b) => C.hbars(b, { rows: Object.entries(tr).map(([f, v]) => ({ label: nm(f), value: v, from: 0 })), format: (v) => v.toFixed(3), name: "Training IC", label: "Training IC", labelWidth: 120, valueWidth: 60 }), table: (b) => Q.simpleTable(b, [{ label: "Factor", get: (x) => nm(x[0]) }, { label: "Training IC", num: true, get: (x) => num(x[1], 4) }], Object.entries(tr)) });
    cc.dataset.role = "comb-train";
  }

  /* ------------------------------------------------------------------ 4 Portfolio construction */
  function portfolio(pane, S) {
    const head = h("div", "card", pane); h("h3", "", head, "4. Portfolio construction: optimisers on identical windows, costs and constraints");
    h("p", "hint", head, "Every method gets the same lookback, rebalance dates, covariance estimate, expected-return forecast, per-asset cap and trading cost. Beyond the Sharpe ratio, each is shown with how concentrated it is, how much its weights move between rebalances and when the inputs are nudged, what it pays in costs, and where it sits against the efficient frontier of the same cap.");
    const chips = h("div", "chips", head); h("span", "hint", chips, "Methods:");
    for (const m of S.plan.methods) { const l = h("label", "f inline", chips); l.title = m.note || ""; const cb = h("input", "", l); cb.type = "checkbox"; cb.dataset.method = m.key; cb.checked = R.methods.has(m.key); cb.addEventListener("change", () => { cb.checked ? R.methods.add(m.key) : R.methods.delete(m.key); S.redraw(); }); l.append(document.createTextNode(m.label)); }
    const g = tickersOf(S).length < 4 ? "A portfolio comparison needs at least four tickers." : R.methods.size < 2 ? "Choose at least two methods." : "";
    runRow(head, S, "portfolio", "Run the optimiser comparison", "/api/lab/portfolio", () => ({ ...baseBody(S), methods: [...R.methods], config: { ...S.form.config, cov_method: S.form.cov_method || S.form.config.cov_method || "shrinkage" } }), g);
    const f = R.results.factors;
    if (f && !current(S, "factors")) h("p", "hint warn", head, "The factor study on screen was run on other settings than the form's: the optimisers below do not use it, but should not be read as its sequel until it is re-run.");
    if (!banner(pane, S, "portfolio", "optimiser comparison")) { const e = h("div", "card empty", pane); h("p", "", e, "No optimiser comparison on this sample yet."); return; }
    const r = R.results.portfolio, d = r.diagnostics, rows = r.table, cfg = r.config, c = colors();
    const top = h("div", "card", pane); top.dataset.role = "pf-head"; h("h3", "", top, `${rows.length} methods, ${cfg.assets} assets, ${cfg.first} to ${cfg.last}`);
    h("p", "sub", top, `Window ${cfg.lookback} days, rebalanced every ${cfg.step}, cost ${cfg.cost_bps} bp, ${cfg.cov_method} covariance, cap ${pct(cfg.max_weight, 0)} per asset.`);
    if (!d || d.error) { const b = h("div", "banner warn", pane, "The concentration and sensitivity view is not available: " + ((d && d.error) || "the server did not return it") + ". The comparison above stands without it."); b.dataset.role = "pf-diag-missing"; return; }
    const cond = d.conditions; h("p", "hint", top, `Diagnostics are at the last rebalance (${d.as_of_rebalance}), window ${d.window[0]} to ${d.window[1]}; same lookback ${cond.lookback}, ${cond.cov_method} covariance, cap ${pct(cond.max_weight, 0)}, ${cond.cost_bps} bp.`);
    const m = Object.entries(d.methods).map(([key, v]) => ({ key, ...v })), byKey = Object.fromEntries(rows.map((x) => [x.key, x]));
    const tc = h("div", "card", pane); tc.dataset.role = "pf-concentration"; h("h3", "", tc, "Concentration, stability, sensitivity and cost, side by side");
    Q.simpleTable(tc, [{ label: "Method", get: (x) => x.label }, { label: "Net Sharpe", num: true, get: (x) => num(byKey[x.key] && byKey[x.key].sharpe, 2) }, { label: "Effective N", num: true, get: (x) => num(x.effective_n, 1) }, { label: "Largest weight", num: true, get: (x) => pct(x.max_weight, 0) },
      { label: "Top 3 share", num: true, get: (x) => pct(x.top3_share, 0) }, { label: "Zero positions", num: true, get: (x) => num(x.zero_weight_positions, 1) }, { label: "Cap binding", num: true, get: (x) => pct(x.cap_binding_share, 0) },
      { label: "Weight turnover / rebalance", num: true, get: (x) => pct(x.weight_turnover, 1) }, { label: "Weight rank correlation", num: true, get: (x) => num(x.weight_rank_correlation, 2) },
      { label: "Moves with noisy returns", num: true, get: (x) => (x.mu_sensitivity === null ? "ignores them" : pct(x.mu_sensitivity, 0)) }, { label: "Moves with another covariance", num: true, get: (x) => (x.cov_sensitivity === null ? "n/a" : pct(x.cov_sensitivity, 0)) },
      { label: "Turnover / year", num: true, get: (x) => num(x.turnover_per_year, 1) }, { label: "Cost drag / year", num: true, get: (x) => pct(x.cost_drag_per_year, 2) }, { label: "Sharpe gross to net", num: true, get: (x) => `${num(x.gross_net_sharpe[0], 2)} to ${num(x.gross_net_sharpe[1], 2)}` }], m);
    for (const n of d.notes) h("p", "hint", tc, n);
    const two = h("div", "charts", pane);
    const cn = Q.chartCard(two, "Effective number of positions", "One over the sum of squared weights, averaged over rebalances: how many equal positions the portfolio is worth. Equal weight is the number of assets.",
      { draw: (b) => C.hbars(b, { rows: m.slice().sort((a, b2) => b2.effective_n - a.effective_n).map((x) => ({ label: x.label, value: x.effective_n, from: 0, tip: [["Largest weight", pct(x.max_weight, 0)], ["Top 3", pct(x.top3_share, 0)]] })), format: (v) => v.toFixed(1), name: "Effective N", label: "Effective number of positions", labelWidth: 150, valueWidth: 50, color: () => c[0] }),
        table: (b) => Q.simpleTable(b, [{ label: "Method", get: (x) => x.label }, { label: "Effective N", num: true, get: (x) => num(x.effective_n, 2) }, { label: "HHI", num: true, get: (x) => num(x.hhi, 3) }], m) });
    cn.dataset.role = "pf-effn";
    const sens = m.filter((x) => x.cov_sensitivity !== null), sc = Q.chartCard(two, "How far the weights move when the covariance is estimated another way", "Half the L1 distance between the weights at the last rebalance under the chosen covariance and under the alternative (0: the same portfolio; 1: a different one). A method that moves a lot is leaning on its covariance estimate.",
      { draw: (b) => C.hbars(b, { rows: sens.slice().sort((a, b2) => a.cov_sensitivity - b2.cov_sensitivity).map((x) => ({ label: x.label, value: x.cov_sensitivity, from: 0, tip: [["With noisy returns", x.mu_sensitivity === null ? "ignores them" : pct(x.mu_sensitivity, 0)]] })), format: (v) => (v * 100).toFixed(0) + "%", name: "Weight change", label: "Covariance sensitivity", labelWidth: 150, valueWidth: 50, color: () => c[1] }),
        table: (b) => Q.simpleTable(b, [{ label: "Method", get: (x) => x.label }, { label: "Covariance", num: true, get: (x) => pct(x.cov_sensitivity, 1) }, { label: "Noisy returns", num: true, get: (x) => (x.mu_sensitivity === null ? "ignores them" : pct(x.mu_sensitivity, 1)) }], m) });
    sc.dataset.role = "pf-sensitivity";
    const fr = d.frontier, pl = d.placed;
    const fc = Q.chartCard(pane, "Where each method sits against the efficient frontier", `The lowest-volatility long-only portfolios under the same ${pct(cond.max_weight, 0)} cap for a ladder of expected returns, from the last window's covariance and the shrunk mean. A method well inside the curve takes more risk than needed for what it expects to earn, or optimises something else on purpose (drawdown, tail risk).`,
      { wide: true, draw: (b) => C.scatter(b, { points: [...fr.map((x) => ({ x: x.vol, y: x.return, r: 2.5, color: c[0], tip: [["Frontier volatility", pct(x.vol, 2)], ["Expected return", pct(x.return, 2)]] })), ...Object.entries(pl).map(([k, v], i) => ({ x: v.vol, y: v.return, label: d.methods[k].label, r: 5, color: c[(i + 1) % 8], tip: [["Volatility", pct(v.vol, 2)], ["Expected return", pct(v.return, 2)]] }))],
          lines: fr.slice(1).map((x, i) => ({ x1: fr[i].vol, y1: fr[i].return, x2: x.vol, y2: x.return, color: c[0], name: i === 0 ? "Efficient frontier" : undefined })), xFormat: (v) => (v * 100).toFixed(1) + "%", yFormat: (v) => (v * 100).toFixed(1) + "%", xLabel: "Volatility (annual)", yLabel: "Expected return", height: 280, label: "Optimisers against the efficient frontier" }),
        table: (b) => Q.simpleTable(b, [{ label: "Portfolio", get: (x) => x.name }, { label: "Volatility", num: true, get: (x) => pct(x.vol, 2) }, { label: "Expected return", num: true, get: (x) => pct(x.ret, 2) }], [...fr.map((x) => ({ name: "frontier", vol: x.vol, ret: x.return })), ...Object.entries(pl).map(([k, v]) => ({ name: d.methods[k].label, vol: v.vol, ret: v.return }))]) });
    fc.dataset.role = "pf-frontier";
  }

  /* ------------------------------------------------------------------ 5 Performance and attribution */
  function performance(pane, S) {
    const head = h("div", "card", pane); h("h3", "", head, "5. Performance and attribution: what the alpha keeps once it is traded");
    h("p", "hint", head, "One chosen signal is traded with costs and impact, delayed a day, and constrained; the table shows the Sharpe ratio step by step from the paper forecast to the realised book and what each real-world step took. Next to it, the optimisers' own cost drag from stage 4 on the same sample.");
    const g = guard(S), f = R.results.factors, ok = f && sameSample(S, "factors");
    const row = h("div", "params", head); const sl = h("label", "f", row, "Signal to trade"), sel = h("select", "", sl); sel.dataset.key = "perf_signal";
    const first = ok ? (f.comparison.table.find((x) => x.survives_bh) || f.comparison.table[0]).factor : S.plan.signals[0].key; if (!R.perf.signal) R.perf.signal = first;
    for (const s of S.plan.signals) { const o = h("option", "", sel, s.label); o.value = s.key; if (s.key === R.perf.signal) o.selected = true; } sel.addEventListener("change", () => { R.perf.signal = sel.value; S.redraw(); });
    for (const [k, label] of [["cost_bps", "Cost (bp per unit traded)"]]) { const l = h("label", "f", row, label), i = h("input", "", l); i.type = "number"; i.step = "any"; i.value = R.perf[k]; i.dataset.key = k; i.addEventListener("input", () => { R.perf[k] = +i.value; }); i.addEventListener("change", () => S.redraw()); }
    const lo = h("label", "f inline", row), cb = h("input", "", lo); cb.type = "checkbox"; cb.checked = R.perf.long_only; cb.dataset.key = "perf_long_only"; cb.addEventListener("change", () => { R.perf.long_only = cb.checked; S.redraw(); }); lo.append(document.createTextNode("Long only"));
    if (ok) { const r0 = f.comparison.table.find((x) => x.factor === R.perf.signal); if (r0 && !r0.survives_bh) h("p", "hint warn", head, `${R.perf.signal} did not survive the multiple-testing correction in stage 2 (p Holm ${num(r0.p_holm, 3)}). Trading it shows what a signal that may be noise would keep; it is not evidence that it works.`); }
    else if (f) h("p", "hint warn", head, "The factor study on screen is for another sample or settings, so nothing here is read against its validation.");
    runRow(head, S, "performance", "Trade the signal", "/api/lab/trading", () => ({ ...baseBody(S), signal: R.perf.signal, horizon: +S.form.horizon, long_only: R.perf.long_only, config: { cost_bps: R.perf.cost_bps, target_gross: 1, lag: 1 } }), g);
    if (!banner(pane, S, "performance", "trading study")) { const e = h("div", "card empty", pane); h("p", "", e, "No trading study on this sample yet."); return; }
    const r = R.results.performance, wf = r.waterfall, c = colors();
    const top = h("div", "card", pane); top.dataset.role = "perf-head"; h("h3", "", top, `${r.label}: ${r.days} days, ${r.first} to ${r.last}`); h("p", "sub", top, `Costs ${r.config.cost_bps} bp, ${r.config.long_only ? "long only" : "long-short"}, delayed ${r.config.lag} day.`);
    for (const n of r.notes) h("p", "hint", top, n);
    const wc = Q.chartCard(pane, "Where the paper alpha goes", "The Sharpe ratio of the same forecasts as each real-world step is added.",
      { wide: true, draw: (b) => C.columns(b, { labels: wf.map((x) => x.index), values: wf.map((x) => x.sharpe), yFormat: (v) => v.toFixed(1), label: "Sharpe by implementation step", height: 220 }),
        table: (b) => Q.simpleTable(b, [{ label: "Step", get: (x) => x.index }, { label: "Sharpe", num: true, get: (x) => num(x.sharpe, 2) }, { label: "Change", num: true, get: (x) => num(x.lost_sharpe, 2) }, { label: "Return (annual)", num: true, get: (x) => pct(x.ann_return) }], wf) });
    wc.dataset.role = "perf-waterfall";
    const pc = h("div", "card", pane); pc.dataset.role = "perf-policies"; h("h3", "", pc, "Four ways to trade the same alpha");
    Q.simpleTable(pc, [{ label: "Policy", get: (x) => x.index }, { label: "Sharpe, net", num: true, get: (x) => (x.turnover_per_year === 0 ? "never traded" : num(x.sharpe, 2)) }, { label: "Sharpe, gross", num: true, get: (x) => (x.turnover_per_year === 0 ? "never traded" : num(x.gross_sharpe, 2)) },
      { label: "Costs / year", num: true, get: (x) => pct(x.cost_per_year, 2) }, { label: "Turnover / year", num: true, get: (x) => num(x.turnover_per_year, 1) }], r.policies);
    const p = R.results.portfolio;
    const at = h("div", "card", pane); at.dataset.role = "perf-attribution"; h("h3", "", at, "Cost attribution of the optimisers (stage 4)");
    if (p && p.diagnostics && !p.diagnostics.error && sameSample(S, "portfolio")) {
      const same = current(S, "portfolio");
      if (!same) h("p", "hint warn", at, "The optimiser comparison was run with other settings than the form's now has; its costs are shown as run.");
      h("p", "hint", at, `Optimiser costs are ${p.config.cost_bps} bp; the trading study above uses ${r.config.cost_bps} bp. ${p.config.cost_bps === r.config.cost_bps ? "The same cost assumption." : "They are different assumptions: read the two cost columns separately, not against each other."}`);
      Q.simpleTable(at, [{ label: "Method", get: (x) => x.label }, { label: "Turnover / year", num: true, get: (x) => num(x.turnover_per_year, 1) }, { label: "Cost drag / year", num: true, get: (x) => pct(x.cost_drag_per_year, 2) }, { label: "Sharpe gross to net", num: true, get: (x) => `${num(x.gross_net_sharpe[0], 2)} to ${num(x.gross_net_sharpe[1], 2)}` }], Object.values(p.diagnostics.methods));
    } else h("p", "hint", at, "Run stage 4 on this sample to put the optimisers' cost drag next to the signal's.");
  }

  Q.alphaResearch = { render, state: R, runKey, guard: (S) => guard(S), STAGES };
})();
