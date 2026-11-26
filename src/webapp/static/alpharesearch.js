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
  const R = { mu: { source: "historical", factor: "momentum", forecast_method: "historical", calibrate: "past_slope", combine: "equal", compare: true }, attr: { kind: "market", sector_source: "auto" }, stage: "discovery", signals: null, fundamental: new Set(), fundCatalogue: null, methods: new Set(DEFAULT_METHODS), perf: { signal: "", cost_bps: 5, long_only: true }, results: {}, keys: {}, busy: {}, jobs: {} };
  const colors = () => ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"].map(css);

  /* What identifies a sample, and what identifies a run: the second contains the first. A result is current only when its run key is the form's. */
  const tickersOf = (S) => S.form.tickers.split(/[\s,;]+/).filter(Boolean);
  const sampleOf = (S) => ({ tickers: tickersOf(S), start: S.form.start || "", source: S.form.source || "" });
  const sig = (S) => (R.signals || S.plan.signals.map((s) => s.key)).slice().sort();
  const runKey = {
    factors: (S) => JSON.stringify({ ...sampleOf(S), horizon: +S.form.horizon, signals: sig(S), fundamental: [...R.fundamental].sort() }),
    portfolio: (S) => JSON.stringify({ ...sampleOf(S), methods: [...R.methods].sort(), config: S.form.config, cov: S.form.cov_method || "", mu: R.mu, attr: R.attr }),
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
      ["Factors", `${sig(S).length} price${R.fundamental.size ? " + " + R.fundamental.size + " fundamental" : ""}`], ["Cost", `${R.perf.cost_bps} bp (trading) / ${(p && p.config && p.config.cost_bps) ?? S.form.config.cost_bps ?? "default"} bp (optimisers)`], ["Data", f && f.universe && f.universe.source ? f.universe.source : "not loaded"], ...(f && f.fundamental ? [["Point in time", f.fundamental.pit.verdict.status.replace(/_/g, " ")]] : [])];
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
  const factorBody = (S) => ({ ...baseBody(S), signals: sig(S).sort((a, b) => S.plan.signals.findIndex((x) => x.key === a) - S.plan.signals.findIndex((x) => x.key === b)), horizon: +S.form.horizon, ...(R.fundamental.size ? { fundamental_factors: [...R.fundamental] } : {}) });
  const total = (S) => sig(S).length + R.fundamental.size;
  const labelOf = (S) => { const m = Object.fromEntries(S.plan.signals.map((s) => [s.key, s.label])); for (const x of (R.fundCatalogue || [])) m["fund:" + x.name] = "Fundamental: " + x.label; return (k) => m[k] || (String(k).startsWith("fund:") ? "Fundamental: " + String(k).slice(5).replace(/_/g, " ") : k); };

  /* ------------------------------------------------------------------ 1 Discovery */
  function signalChips(pane, S) {
    const box = h("div", "chips", pane); h("span", "hint", box, "Signals to compare:");
    for (const s of S.plan.signals) { const l = h("label", "f inline", box); l.title = s.help || ""; const cb = h("input", "", l); cb.type = "checkbox"; cb.dataset.signal = s.key; cb.checked = sig(S).includes(s.key);
      cb.addEventListener("change", () => { const cur = new Set(sig(S)); cb.checked ? cur.add(s.key) : cur.delete(s.key); R.signals = [...cur]; S.redraw(); }); l.append(document.createTextNode(s.label)); }
  }

  const FUND_DEFAULT = ["roe", "net_margin", "gross_margin", "roa", "operating_margin", "debt_to_assets", "sloan_accruals", "asset_turnover"];
  function fundamentalChips(pane, S) {
    const box = h("div", "chips", pane); box.dataset.role = "fundamental-factors"; h("span", "hint", box, "Fundamental factors (point in time):");
    if (R.fundCatalogue === null) { R.fundCatalogue = []; api("/api/fund/catalogue").then((c) => { R.fundCatalogue = (c.metrics || []).filter((m) => FUND_DEFAULT.includes(m.name)); S.redraw(); }).catch(() => { R.fundCatalogue = []; }); }
    for (const m of R.fundCatalogue) {
      const l = h("label", "f inline", box); l.title = m.text || ""; const cb = h("input", "", l); cb.type = "checkbox"; cb.dataset.fundamental = m.name; cb.checked = R.fundamental.has(m.name);
      cb.addEventListener("change", () => { cb.checked ? R.fundamental.add(m.name) : R.fundamental.delete(m.name); S.redraw(); }); l.append(document.createTextNode(m.label));
    }
    if (!R.fundCatalogue.length) h("span", "hint", box, "none available");
    h("p", "hint", pane, "A fundamental factor is the latest figure public on each date, from the Fundamentals tab's point-in-time tables. It is admitted only after the point-in-time check of those companies is not blocked, needs statements for at least 6 of the tickers, and is counted in the correction for the number of factors tried.");
  }

  function discovery(pane, S) {
    const head = h("div", "card", pane); h("h3", "", head, "1. Discovery: is there enough to rank, and are the signals different from each other?");
    h("p", "hint", head, "Coverage, extreme values, drift between the first and second half of the history, how persistent each ranking is (so how much it will trade) and how redundant the signals are. Nothing here looks at returns, so nothing here can be overfitted to them.");
    signalChips(head, S);
    fundamentalChips(head, S);
    if (total(S) < 2) h("p", "hint warn", head, "Choose at least two factors: multiple-testing corrections and combinations need more than one.");
    runRow(head, S, "factors", "Run the factor study (stages 1 to 3)", "/api/lab/factors", () => factorBody(S), guard(S) || (total(S) < 2 ? "Choose at least two factors." : ""));
    if (!banner(pane, S, "factors", "factor study")) { const e = h("div", "card empty", pane); h("p", "", e, "No factor study on this sample yet."); return; }
    const r = R.results.factors, d = r.discovery, rows = Object.entries(d.signals).map(([k, v]) => ({ key: k, ...v })), nmf = labelOf(S), label = new Proxy({}, { get: (_, k) => nmf(k) });
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
    const r = R.results.factors, cmp = r.comparison, rows = cmp.table, c = colors(), nm = labelOf(S);
    const k = h("div", "kpis", head);
    Q.tile(k, "Factors tested", String(cmp.n_tested), { text: `family of ${cmp.n_tested}, level ${cmp.alpha}` });
    Q.tile(k, "Survive Holm", String(rows.filter((x) => x.survives_holm).length), { text: "family-wise error controlled", dir: rows.some((x) => x.survives_holm) ? "up" : "" });
    Q.tile(k, "Survive Benjamini-Hochberg", String(rows.filter((x) => x.survives_bh).length), { text: "false-discovery rate controlled", dir: rows.some((x) => x.survives_bh) ? "up" : "" });
    Q.tile(k, "Horizon", `${r.horizon} days`, { text: `${r.days} days, ${r.assets.length} assets` });
    h("p", "hint", head, cmp.note);
    if (r.fundamental) pitPanel(pane, r.fundamental);
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

  const PIT_WORD = { ready: "Ready: every availability date is a real filing date", ready_with_assumptions: "Ready with assumptions", blocked: "Blocked" };
  function pitPanel(pane, f) {
    const pit = f.pit, card = h("div", "card", pane); card.dataset.role = "val-pit"; h("h3", "", card, "Fundamental factors: the point-in-time check that admitted them");
    const b = h("div", "banner " + (pit.verdict.status === "ready" ? "ok" : "warn"), card, `${PIT_WORD[pit.verdict.status] || pit.verdict.status}. ${pit.periods} periods of ${pit.companies} companies; ${pit.estimated} with an assumed date; ${pit.impossible} dated before they end; typical real lag ${pit.lag_days ? Math.round(pit.lag_days.median) + " days" : "n/a"}.`); b.dataset.status = pit.verdict.status;
    for (const a of pit.assumptions) h("p", "hint", card, a);
    Q.simpleTable(card, [{ label: "Factor", get: (x) => "Fundamental: " + x[0] }, { label: "Companies with a value", num: true, get: (x) => `${x[1].tickers_with_a_value} of ${x[1].tickers_requested}` }, { label: "Share of company-days with a value", num: true, get: (x) => pct(x[1].share_of_cells, 0) },
      { label: "Median companies per day", num: true, get: (x) => num(x[1].median_assets_per_date, 1) }, { label: "First value", get: (x) => x[1].first_value || "none" }], Object.entries(f.coverage));
    if (f.missing_tickers.length) h("p", "hint warn", card, `No statements for ${f.missing_tickers.join(", ")}: those companies have no fundamental factor values.`);
    h("p", "hint", card, `Basis: ${f.basis === "fy" ? "annual statements" : f.basis}; each figure is used from the day after its filing (${f.settings.info_lag_days} day lag), the figure is dropped when the latest period is older than ${f.settings.max_age_days[f.basis] || "the allowed"} days.`);
  }

  async function saveDialog(S, status, req, opts) {
    const saved = await Q.research.saveStudy(req, { title: opts && opts.title, hypothesis: opts && opts.hypothesis, sessionName: "Alpha research" });
    if (saved) { status.textContent = saved.duplicate ? "Already saved: the same run is kept once." : `Saved as "${saved.title}".`; status.dataset.saved = saved.id; }
    return saved;
  }
  function keepRow(pane, S, what, body, title) {
    const row = h("div", "row", pane), b = h("button", "btn", row, what); b.type = "button"; b.dataset.role = "keep-study"; const st = h("span", "hint", row); st.setAttribute("role", "status");
    b.title = "Saves the study in a research session with its sample, horizon, data version and assumptions. A study has no return series, so a comparison lists it and checks that it is comparable; it is not ranked by performance.";
    b.addEventListener("click", () => saveDialog(S, st, body(), { title })); return row;
  }

  /* ------------------------------------------------------------------ 3 Combination */
  function combination(pane, S) {
    const head = h("div", "card", pane); h("h3", "", head, "3. Combination: is a blend better than its best part?");
    h("p", "hint", head, "Equal weight, IC-weighted, ridge and orthogonalised blends, with weights from the past only. The first part of the history chooses the best single factor; the later part, which nothing was fitted on, judges everything. A blend counts as an improvement only if it beats both the best single factor and equal weight by more than 1.96 standard errors of the difference.");
    if (!banner(pane, S, "factors", "factor study")) { const e = h("div", "card empty", pane); h("p", "", e, "Run the factor study in stage 1 first."); return; }
    const r = R.results.factors, cb = r.combination, nm = labelOf(S);
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
    keepRow(pane, S, "Save this study to a research session…", () => ({ job: R.jobs.factors }), "Factor study");
  }

  /* ------------------------------------------------------------------ 4 Portfolio construction */
  const stepOf = (S) => +(S.form.config && S.form.config.step) || 21;
  function muSpecs(S) {
    const m = R.mu, step = stepOf(S), one = (src) => ({ source: src, factor: m.factor, forecast_method: m.forecast_method, calibrate: m.calibrate, combine: m.combine, horizon: step });
    const chosen = one(m.source);
    return m.source !== "historical" && m.compare ? [chosen, { source: "historical" }] : [chosen];
  }
  const portfolioBody = (S) => ({ ...baseBody(S), methods: [...R.methods], config: { ...S.form.config, cov_method: S.form.cov_method || S.form.config.cov_method || "shrinkage" }, expected_returns: muSpecs(S),
    attribution: { kind: R.attr.kind }, sector_source: R.attr.sector_source });

  /* The settings every method and every source of expected returns shares: window, rebalance step, cost, covariance estimator, caps. */
  function settingsControls(pane, S) {
    const box = h("div", "params", pane); box.dataset.role = "pf-settings";
    for (const s of S.plan.settings) {
      if (s.key === "cov_method") {
        const l = h("label", "f", box, s.label); l.title = s.help; const sel = h("select", "", l); sel.dataset.key = "cov_method";
        for (const c of S.plan.cov_methods) { const o = h("option", "", sel, c); o.value = c; if ((S.form.cov_method || S.form.config.cov_method || "shrinkage") === c) o.selected = true; }
        sel.addEventListener("change", () => { S.form.cov_method = sel.value; S.form.config.cov_method = sel.value; S.redraw(); });
        continue;
      }
      const l = h("label", "f", box, s.label); l.title = s.help; const i = h("input", "", l); i.type = "number"; i.step = "any"; i.value = S.form.config[s.key]; i.dataset.key = s.key;
      i.addEventListener("input", () => { S.form.config[s.key] = i.value === "" ? "" : +i.value; });
      i.addEventListener("change", () => S.redraw());
    }
  }

  /* Where the optimisers' expected returns come from. Everything else (risk estimate, caps, costs, rebalance dates) is the same for every source. */
  function muControls(pane, S) {
    const box = h("div", "params", pane); box.dataset.role = "mu-controls";
    const select = (label, key, options, title) => { const l = h("label", "f", box, label); if (title) l.title = title; const s = h("select", "", l); s.dataset.key = "mu_" + key; for (const [v, tx] of options) { const o = h("option", "", s, tx); o.value = v; if (R.mu[key] === v) o.selected = true; } s.addEventListener("change", () => { R.mu[key] = s.value; S.redraw(); }); return s; };
    select("Expected returns from", "source", [["historical", "The window's own mean (the Lab's default)"], ["factor", "A factor (information coefficient x volatility x z-score)"], ["forecast", "A forecasting method"], ["combined", "Factor and forecast combined"]],
      "Methods that use expected returns (maximum Sharpe, mean-variance, Black-Litterman and others) are given these; methods that do not (minimum variance, risk parity, hierarchical) are unaffected, so they are the control.");
    if (["factor", "combined"].includes(R.mu.source)) select("Factor", "factor", S.plan.signals.map((s) => [s.key, s.label]));
    if (["forecast", "combined"].includes(R.mu.source)) {
      select("Forecasting method", "forecast_method", [["historical", "Historical mean"], ["gbm", "Geometric Brownian motion"], ["theta", "Theta"], ["kalman", "Kalman trend"], ["arma", "ARMA"], ["hull_white", "Mean-reverting (Hull-White)"], ["markov_returns", "Markov returns"]],
        "Each forecast is made from the prices up to the day before a rebalance and used only from then on.");
      select("Trust in forecasts", "calibrate", [["past_slope", "By their past record (slope of realised on forecast)"], ["none", "Fully (no calibration)"]], "With too few forecasts already resolved the past-record option gives them no weight.");
    }
    if (R.mu.source === "combined") select("Blend", "combine", [["equal", "Equal weights"], ["inverse_error", "By past squared error"]]);
    if (R.mu.source !== "historical") {
      const l = h("label", "f inline", box), cb = h("input", "", l); cb.type = "checkbox"; cb.checked = R.mu.compare; cb.dataset.key = "mu_compare"; cb.addEventListener("change", () => { R.mu.compare = cb.checked; S.redraw(); });
      l.append(document.createTextNode("Run the historical mean beside it, on identical settings"));
      h("p", "hint wide", box, `The forecast horizon is the rebalance step (${stepOf(S)} days): a forecast is used only at its own horizon. It is put in the units of the annualised covariance by the same scaling for every source, which is a change of units and not an annual return forecast.`);
    }
  }
  function attrControls(pane, S) {
    const box = h("div", "params", pane); box.dataset.role = "attr-controls";
    const one = (label, key, options, title) => { const l = h("label", "f", box, label); if (title) l.title = title; const s = h("select", "", l); s.dataset.key = "attr_" + key; for (const [v, tx] of options) { const o = h("option", "", s, tx); o.value = v; if (R.attr[key] === v) o.selected = true; } s.addEventListener("change", () => { R.attr[key] = s.value; S.redraw(); }); };
    one("Factor model for attribution", "kind", [["market", "Market (equal-weighted universe)"], ["statistical", "Statistical (principal components)"]], "Exposures are estimated on the window before each rebalance; contributions are what happened afterwards.");
    one("Groups for Brinson attribution", "sector_source", [["auto", "From the data (asset class), if it has them"], ["fundamentals", "Sectors from the Fundamentals tab"], ["none", "No Brinson attribution"]], "Brinson attribution needs every asset in a group and a benchmark: equal weight here.");
  }

  function portfolio(pane, S) {
    const head = h("div", "card", pane); h("h3", "", head, "4. Portfolio construction: optimisers on identical windows, costs and constraints");
    h("p", "hint", head, "Every method gets the same lookback, rebalance dates, covariance estimate, expected-return forecast, per-asset cap and trading cost. Beyond the Sharpe ratio, each is shown with how concentrated it is, how much its weights move between rebalances and when the inputs are nudged, what it pays in costs, and where it sits against the efficient frontier of the same cap.");
    const chips = h("div", "chips", head); h("span", "hint", chips, "Methods:");
    for (const m of S.plan.methods) { const l = h("label", "f inline", chips); l.title = m.note || ""; const cb = h("input", "", l); cb.type = "checkbox"; cb.dataset.method = m.key; cb.checked = R.methods.has(m.key); cb.addEventListener("change", () => { cb.checked ? R.methods.add(m.key) : R.methods.delete(m.key); S.redraw(); }); l.append(document.createTextNode(m.label)); }
    const g = tickersOf(S).length < 4 ? "A portfolio comparison needs at least four tickers." : R.methods.size < 2 ? "Choose at least two methods." : "";
    settingsControls(head, S);
    muControls(head, S);
    attrControls(head, S);
    runRow(head, S, "portfolio", "Run the optimiser comparison", "/api/lab/portfolio", () => portfolioBody(S), g);
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
    expectedCards(pane, S, r, d, c);
    const kc = h("div", "card", pane); kc.dataset.role = "pf-keep"; h("h3", "", kc, "Save a method to a research session");
    h("p", "hint", kc, `Saves one method's net equity curve, with equal weight as its benchmark, its attribution, the source of its expected returns and its costs, so it can be reopened, re-run as a new version and compared. Its Sharpe ratio is not corrected for having been picked from ${rows.length} methods.`);
    const kr = h("div", "row", kc), ks = h("select", "", kr); ks.dataset.key = "keep_method"; ks.setAttribute("aria-label", "Method to save"); for (const x of rows) { const o = h("option", "", ks, x.method); o.value = x.key; }
    const kb = h("button", "btn", kr, "Save to research session…"); kb.type = "button"; kb.dataset.role = "keep-method"; const kt = h("span", "hint", kr); kt.setAttribute("role", "status");
    kb.addEventListener("click", () => saveDialog(S, kt, { job: R.jobs.portfolio, method: ks.value }, { title: `${(rows.find((x) => x.key === ks.value) || {}).method || ks.value}, expected returns: ${r.expected_returns ? r.expected_returns.source : "historical mean"}` }));
  }

  function expectedCards(pane, S, r, d, c) {
    const er = r.expected_returns, cmp = r.source_comparison;
    if (!er) return;
    const card = h("div", "card", pane); card.dataset.role = "pf-expected"; h("h3", "", card, `Expected returns used by the optimisers: ${er.source}`);
    const rep = Object.entries(er.report), filled = rep.filter(([, v]) => v.filled_with_historical > 0).length, info = er.last_info || {}, meta = er.meta || {};
    const k = h("div", "kpis", card);
    Q.tile(k, "Rebalances with the source's own estimate", `${rep.length - filled} of ${rep.length}`, { text: filled ? `${filled} used the window's mean for some assets: no usable estimate yet, or it was stale` : "every asset had an estimate" });
    if (er.source !== "historical mean") {
      Q.tile(k, "Forecast horizon", `${er.horizon_days} days`, { text: meta.allow_horizon_scaling ? "scaled to the holding period (constant-drift assumption)" : "the holding period: used only at its own horizon" });
      const unc = rep.map(([, v]) => v.mean_uncertainty).filter((x) => x !== null && x !== undefined); if (unc.length) Q.tile(k, "Uncertainty of an estimate (mean)", pct(unc.reduce((a, b) => a + b, 0) / unc.length, 1), { text: "standard deviation over the horizon" });
    }
    const rel = (info.reliability) || (info.sources && info.sources.forecast && info.sources.forecast.reliability);
    if (rel) {
      Q.tile(k, "Model reliability (past forecasts only)", rel.ic === null || rel.ic === undefined ? "too few" : num(rel.ic, 3), { text: `IC over ${rel.n} resolved forecasts; hit rate ${pct(rel.hit_rate, 0)}; slope of realised on forecast ${num(rel.slope, 2)}` });
      const w = info.weight !== undefined ? info.weight : (info.sources && info.sources.forecast && info.sources.forecast.weight);
      Q.tile(k, "Weight given to the forecasts", w === undefined ? "n/a" : pct(w, 0), { text: (info.calibration) || (info.sources && info.sources.forecast && info.sources.forecast.calibration) || "" });
    }
    if (info.ic !== undefined) Q.tile(k, "Factor information coefficient (completed dates only)", num(info.ic, 3), { text: `${info.ic_dates} dates whose forward return was complete` });
    if (info.how) h("p", "hint", card, "Blend: " + info.how);
    if (meta.forecast_contract) {
      const fc = meta.forecast_contract, dl = h("dl", "kv", card); dl.dataset.role = "forecast-contract";
      for (const [lab, v] of [["Kind", fc.kind], ["Units", fc.units], ["Horizon", `${fc.horizon_days} days`], ["Model", `${fc.model_id} (version ${fc.model_version})`], ["Training data fingerprint", fc.training_fingerprint], ["Evaluation status", fc.evaluation.status], ["Available from", "the day after its origin, never earlier"]]) { h("dt", "", dl, lab); h("dd", "", dl, String(v)); }
    }
    h("p", "hint", card, "Every source is run with the same risk estimate, caps, costs and rebalance dates; only the expected returns differ. A better forecast does not by itself make a better portfolio.");
    if (cmp && Object.keys(cmp).length > 1) {
      const labels = Object.keys(cmp), keys = Object.keys(cmp[labels[0]].equity), pickKeys = keys.filter((k2) => { const a = cmp[labels[0]].table.find((x) => x.key === k2).sharpe, b = cmp[labels[1]].table.find((x) => x.key === k2).sharpe; return a !== b; });
      const cc = h("div", "card", pane); cc.dataset.role = "pf-source-compare"; h("h3", "", cc, "The same optimisers under each source of expected returns");
      Q.simpleTable(cc, [{ label: "Method", get: (x) => (S.plan.methods.find((m) => m.key === x) || {}).label || x }, ...labels.flatMap((l) => [{ label: `Sharpe: ${l}`, num: true, get: (x) => num(cmp[l].table.find((y) => y.key === x).sharpe, 2) }, { label: `Turnover / year: ${l}`, num: true, get: (x) => num(cmp[l].table.find((y) => y.key === x).turnover_per_year, 1) }])], keys);
      h("p", "hint", cc, pickKeys.length ? `Methods that do not use expected returns are identical under every source (${keys.length - pickKeys.length} of ${keys.length}); the difference is in the other ${pickKeys.length}. A Sharpe difference smaller than about two of its standard errors is not evidence.` : "No method in this run uses expected returns, so the sources give identical results.");
      if (pickKeys.length) {
        const dates = r.equity.dates, colors2 = colors();
        const ec = Q.chartCard(pane, "Growth of 1 under each source, for the methods that use expected returns", "Net of costs, the same dates, the same costs.", { wide: true,
          draw: (b) => C.line(b, { dates, series: labels.flatMap((l, i) => pickKeys.slice(0, 3).map((k2, j) => ({ name: `${(S.plan.methods.find((m) => m.key === k2) || {}).label || k2}: ${l}`, color: colors2[(i * 3 + j) % 8], values: cmp[l].equity[k2], dash: i ? true : undefined }))), baseline: 1, yFormat: (v) => v.toFixed(2), height: 260, legend: true }),
          table: (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => dates[i] }, ...labels.map((l) => ({ label: l, num: true, get: (i) => num(cmp[l].equity[pickKeys[0]][i], 3) }))], Q.sampleRows(dates, null, Math.max(1, Math.ceil(dates.length / 60)))) });
        ec.dataset.role = "pf-source-equity";
      }
    }
    const used = d && d.methods ? Object.values(d.methods).filter((x) => x.uses_expected_returns && x.mu_error_curve && Object.keys(x.mu_error_curve).length) : [];
    if (used.length) {
      const sc = Q.chartCard(pane, "How far the weights move if the expected returns are wrong", "At the last rebalance: half the L1 change in weights when noise of a quarter, a half, one and two cross-sectional standard deviations is added to the expected returns. A method that moves a lot is betting on its forecasts being right.", { wide: true,
        draw: (b) => C.line(b, { xs: ["0.25", "0.5", "1.0", "2.0"].map(Number), series: used.map((x, i) => ({ name: x.label, color: c[i % 8], values: ["0.25", "0.5", "1.0", "2.0"].map((m) => x.mu_error_curve[m]) })), yFormat: (v) => (v * 100).toFixed(0) + "%", xFormat: (v) => v + " sd", height: 220, legend: true, label: "Weight change by size of forecast error" }),
        table: (b) => Q.simpleTable(b, [{ label: "Method", get: (x) => x.label }, ...["0.25", "0.5", "1.0", "2.0"].map((m) => ({ label: `noise ${m} sd`, num: true, get: (x) => pct(x.mu_error_curve[m], 0) }))], used) });
      sc.dataset.role = "pf-error-sensitivity";
    }
  }

  /* The attribution of a portfolio's return, from a Portfolio Lab result or a saved study (the same shape): what the exposures earned, what was asset-specific, what costs took, and what does not reconcile. */
  function attributionView(pane, attr, names, opts) {
    opts = opts || {};
    const card = h("div", "card", pane); card.dataset.role = "attribution"; h("h3", "", card, "Where the return came from");
    const keys = Object.keys(attr.methods).filter((k) => attr.methods[k].factor);
    if (!keys.length) { h("p", "hint", card, "No attribution was made for this study."); return card; }
    const pick = (opts.state = opts.state || { key: keys[0] });
    if (!keys.includes(pick.key)) pick.key = keys[0];
    const chips = h("div", "chips", card); chips.setAttribute("role", "group"); chips.setAttribute("aria-label", "Method attributed");
    const body = h("div", "stack", pane); body.dataset.role = "attribution-body";
    const nm = (k) => (names && names[k]) || k;
    const paint = () => {
      body.replaceChildren(); for (const b of chips.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.key === pick.key));
      const m = attr.methods[pick.key], f = m.factor, rec = f.reconciliation, c = colors();
      const kp = h("div", "kpis", body); kp.dataset.role = "attr-reconciliation";
      Q.tile(kp, "Reconciliation with the portfolio's own return", rec.status === "reconciled" ? "Reconciles" : "Does not reconcile", { text: `largest daily residual ${rec.max_daily_residual.toExponential(1)}, cumulative gap ${rec.cumulative_gap.toExponential(1)} (tolerance ${rec.tolerance.toExponential(0)})`, dir: rec.status === "reconciled" ? "up" : "down" });
      Q.tile(kp, "Cumulative net return", pct(rec.authoritative_cumulative_return, 2), { text: `the lines add up to ${pct(rec.sum_of_lines, 2)}` });
      Q.tile(kp, "Replayed accounting against the Lab's", rec.replay_vs_authoritative_max_daily.toExponential(1), { text: "largest daily difference" });
      const order = [...f.factors, "specific", "costs", "financing", "residual"], label = (x) => (f.factors.includes(x) ? `Factor: ${x}` : { specific: "Asset-specific", costs: "Trading costs", financing: "Financing (not modelled)", residual: "Unexplained residual" }[x]);
      let run = 0; const rows = order.map((x) => { const v = f.totals[x] || 0, from = run; run += v; return { label: label(x), value: from + v, from, tip: [["Contribution", pct(v, 2)]] }; });
      rows.push({ label: "Net return", value: run, from: 0, tip: [["Cumulative", pct(run, 2)]] });
      const wf = Q.chartCard(body, `Factor contribution waterfall, ${nm(pick.key)}`, `Each bar starts where the one before ended. Ex-ante exposures times realised factor returns, then what was specific to each asset, then costs, ending at the cumulative net return (${f.kind} factor model; each day's contribution scaled by the value at the start of the day, so the lines add up at every date). Financing is shown as zero: the Lab does not model it.`, { wide: true,
        draw: (b) => C.hbars(b, { rows, format: (v) => (v * 100).toFixed(2) + "%", name: "Cumulative", label: "Factor contribution waterfall", labelWidth: 150, valueWidth: 60, zero: true }),
        table: (b) => Q.simpleTable(b, [{ label: "Line", get: (x) => x.label }, { label: "Contribution", num: true, get: (x) => pct(x.value - x.from, 3) }, { label: "Cumulative", num: true, get: (x) => pct(x.value, 3) }], rows) });
      wf.dataset.role = "attr-waterfall";
      const lines = [...f.factors, "specific", "costs"].concat(Math.abs(f.totals.residual || 0) > 0 ? ["residual"] : []);
      const hc = Q.chartCard(body, "Cumulative contribution through time", "Running total of each line (value-weighted). They add up to the portfolio's cumulative return at every date.", { wide: true,
        draw: (b) => C.line(b, { dates: f.days, series: lines.map((x, i) => ({ name: label(x), color: c[i % 8], values: f.cumulative[x] })), baseline: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 2), height: 240, legend: true, label: "Cumulative contributions" }),
        table: (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => f.days[i] }, ...lines.map((x) => ({ label: label(x), num: true, get: (i) => pct(f.cumulative[x][i], 3) }))], f.days.map((_, i) => i).filter((i) => i % Math.max(1, Math.ceil(f.days.length / 40)) === 0)) });
      hc.dataset.role = "attr-history";
      const ea = h("div", "card", body); ea.dataset.role = "attr-exante"; h("h3", "", ea, "Ex-ante exposures and risk (what the portfolio was set up to do)");
      h("p", "hint", ea, "At each rebalance, from the window before it: the exposure of the target weights to each factor and the share of forecast variance the factors account for. These are intentions; the waterfall above is what happened.");
      Q.simpleTable(ea, [{ label: "Rebalance", get: (x) => x.date }, ...f.factors.map((x) => ({ label: `Exposure to ${x}`, num: true, get: (r) => num(r.exposures[x], 2) })), { label: "Share of risk from factors", num: true, get: (x) => pct(x.factor_risk_share, 0) }, { label: "Forecast volatility (annual)", num: true, get: (x) => pct(x.ex_ante_volatility, 1) }], f.ex_ante);
      if (m.brinson && !m.brinson.error) brinsonView(body, m.brinson, attr, nm(pick.key), c);
      else { const b = h("div", "card", body); b.dataset.role = "brinson-off"; h("h3", "", b, "Brinson attribution"); h("p", "hint", b, m.brinson && m.brinson.error ? "Not made: " + m.brinson.error : (attr.brinson_reason || (pick.key === attr.benchmark ? "This is the benchmark: there is no active return to attribute." : "Not available: it needs a benchmark and a group for every asset.")) + " It is switched on only when the holdings, the benchmark and the classification exist."); }
    };
    for (const k of keys) { const b = h("button", "btn", chips, nm(k)); b.type = "button"; b.dataset.key = k; b.addEventListener("click", () => { pick.key = k; paint(); }); }
    paint();
    const meth = h("details", "card", pane); meth.dataset.role = "attr-methodology"; h("summary", "", meth, "Methodology and assumptions");
    const ul = h("ul", "", meth); for (const x of attr.methodology || []) h("li", "", ul, x);
    return card;
  }
  function brinsonView(body, b, attr, name, c) {
    const card = h("div", "card", body); card.dataset.role = "brinson"; h("h3", "", card, `Brinson-Fachler attribution against ${String(attr.benchmark || "the benchmark").replace(/_/g, " ")}: ${name}`);
    h("p", "sub", card, b.convention + ".");
    const rec = b.reconciliation, k = h("div", "kpis", card);
    Q.tile(k, "Cumulative net active return", pct(b.cumulative_net_active_return, 2), { text: `portfolio ${pct(b.cumulative_returns.portfolio_net, 2)} against benchmark ${pct(b.cumulative_returns.benchmark_net, 2)}` });
    Q.tile(k, "Reconciliation", rec.status === "reconciled" ? "Reconciles" : "Does not reconcile", { text: `daily identity ${rec.daily_identity_error.toExponential(1)}, cumulative gap ${rec.cumulative_gap.toExponential(1)}`, dir: rec.status === "reconciled" ? "up" : "down" });
    for (const w of b.warnings) h("p", "hint warn", card, w);
    const T = b.totals, rows = []; let run = 0;
    for (const [lab, key] of [["Allocation", "allocation"], ["Selection", "selection"], ["Interaction", "interaction"], ["Trading costs (portfolio minus benchmark)", "costs"], ["Unexplained residual", "residual"]]) { rows.push({ label: lab, from: run, value: run + T[key], tip: [["Linked effect", pct(T[key], 3)]] }); run += T[key]; }
    rows.push({ label: "Net active return", from: 0, value: run, tip: [["Cumulative", pct(run, 3)]] });
    const wf = Q.chartCard(body, "Active return: allocation, selection, interaction, costs", "Carino-linked on the net returns, so the bars add up to the cumulative net active return. Allocation is being in the right groups, selection is picking the right assets inside them, interaction is the two together.", { wide: true,
      draw: (x) => C.hbars(x, { rows, format: (v) => (v * 100).toFixed(2) + "%", name: "Cumulative", label: "Brinson waterfall", labelWidth: 220, valueWidth: 60, zero: true }),
      table: (x) => Q.simpleTable(x, [{ label: "Line", get: (r) => r.label }, { label: "Effect", num: true, get: (r) => pct(r.value - r.from, 3) }, { label: "Cumulative", num: true, get: (r) => pct(r.value, 3) }], rows) });
    wf.dataset.role = "brinson-waterfall";
    const gr = h("div", "card", body); gr.dataset.role = "brinson-groups"; h("h3", "", gr, "By group");
    Q.simpleTable(gr, [{ label: "Group", get: (g) => g }, { label: "Average weight (portfolio)", num: true, get: (g) => pct(b.by_group[g].average_portfolio_weight, 1) }, { label: "Average weight (benchmark)", num: true, get: (g) => pct(b.by_group[g].average_benchmark_weight, 1) },
      { label: "Allocation", num: true, get: (g) => pct(b.by_group[g].allocation, 3) }, { label: "Selection", num: true, get: (g) => pct(b.by_group[g].selection, 3) }, { label: "Interaction", num: true, get: (g) => pct(b.by_group[g].interaction, 3) }], b.groups);
    const pc = Q.chartCard(body, "Cumulative effects through time", "Each effect linked over the days up to each date (sampled).", { wide: true,
      draw: (x) => C.line(x, { dates: b.dates, series: ["allocation", "selection", "interaction", "costs"].map((n, i) => ({ name: n, color: c[i % 8], values: b.path[n] })), baseline: 0, yFormat: (v) => (v * 100).toFixed(1) + "%", tipFormat: (v) => pct(v, 3), height: 220, legend: true, label: "Brinson effects through time" }),
      table: (x) => Q.simpleTable(x, [{ label: "Date", get: (i) => b.dates[i] }, ...["allocation", "selection", "interaction", "costs", "residual"].map((n) => ({ label: n, num: true, get: (i) => pct(b.path[n][i], 3) }))], b.dates.map((_, i) => i)) });
    pc.dataset.role = "brinson-history";
  }

  /* ------------------------------------------------------------------ 5 Performance and attribution */
  function attributionSection(pane, S) {
    const pr = R.results.portfolio, box = h("div", "stack", pane); box.dataset.role = "attribution-section";
    const head = h("div", "card", box); h("h3", "", head, "Attribution of the optimisers' returns (from stage 4)");
    if (!pr || !sameSample(S, "portfolio")) { h("p", "hint", head, "Run the optimiser comparison in stage 4 on this sample to explain what its portfolios earned: by factor and, where the data have groups, by Brinson-Fachler against the equal-weight benchmark."); return; }
    if (!current(S, "portfolio")) h("p", "hint warn", head, "The optimiser comparison was run with other settings than the form's now has; its attribution is shown as it was run.");
    if (!pr.attribution) { h("p", "hint", head, "Attribution was switched off for that run."); return; }
    h("p", "hint", head, `${pr.attribution.kind === "statistical" ? "Statistical (principal-component)" : "Market"} factor model. Exposures use only the window before each rebalance. ${pr.attribution.brinson_available ? "Brinson attribution is on: the data carry a group for every asset." : "Brinson attribution is off: " + (pr.attribution.brinson_reason || "no classification") + "."}`);
    const names = Object.fromEntries(pr.table.map((x) => [x.key, x.method]));
    attributionView(box, pr.attribution, names, R.attrView = R.attrView || {});
  }

  function performance(pane, S) {
    attributionSection(pane, S);
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
    keepRow(pane, S, "Save this trading study to a research session…", () => ({ job: R.jobs.performance }), `Trading study: ${R.perf.signal}`);
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

  Q.alphaResearch = { render, state: R, runKey, guard: (S) => guard(S), STAGES, attributionView };
})();
