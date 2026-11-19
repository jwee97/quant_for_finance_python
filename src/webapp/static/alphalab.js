/* The Alpha, Risk and Portfolio Lab: after Grinold and Kahn (Active Portfolio Management and its Advances), Paleologo (Advanced Portfolio Management) and Cajas (Advanced Portfolio Optimization).

   Four studies on a basket of tickers' daily prices: how good a signal is and how fast it fades (alpha), where a portfolio's risk comes from and whether the forecast held (risk), optimisers compared on identical
   windows, costs and caps (portfolio), and where the paper alpha goes between the forecast and the trade (trading). Every table says what it does not show: differences between optimisers are usually inside the
   sampling error, a method that fails on a date is counted, and a signal made from prices is an example of the interface and not a recommendation. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, toast, css } = Q, C = Q.C;
  const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.num(v, d)), pct = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.pct(v, d));
  const root = $("#alphalab");
  const MODES = [["alpha", "Alpha"], ["risk", "Risk"], ["portfolio", "Portfolio"], ["trading", "Trading"]];
  const S = { mode: "alpha", plan: null, busy: false, results: {}, form: { tickers: "", start: "", source: "", signal: "momentum", horizon: 5, kind: "statistical", n_factors: "", methods: null, config: {}, trading: { cost_bps: 5, target_gross: 1, lag: 1 }, long_only: true }, weightsOf: null };
  const colors = () => ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"].map(css);

  function defaults() {
    const c = (Q.research && Q.research.active() && Q.research.active().context) || {}, setup = Q.setupContext ? Q.setupContext() : {};
    const assets = (c.assets && c.assets.length ? c.assets : setup.assets) || [];
    S.form.tickers = S.form.tickers || assets.slice(0, 12).join(", ");
    S.form.start = S.form.start || c.start || setup.start || ""; S.form.source = S.form.source || c.data_source || setup.data_source || "";
  }

  async function draw() {
    root.replaceChildren();
    const d = h("div", "head", root), l = h("div", "", d); h("h2", "", l, "Alpha, Risk and Portfolio Lab");
    h("p", "sub", l, "From a signal to a calibrated forecast, a factor model of the risk, optimisers compared like for like, and what the trading costs of the alpha.");
    if (!S.plan) { try { S.plan = await api("/api/lab/plan"); } catch (e) { h("div", "banner err", root, "The lab service is not available: " + e.message + " (restart the app if it was just updated)"); return; } if (!S.form.methods) S.form.methods = new Set(S.plan.defaults_methods); Object.assign(S.form.config, S.plan.defaults); }
    const seg = h("div", "seg", root); seg.setAttribute("role", "group"); seg.setAttribute("aria-label", "Study");
    for (const [k, label] of MODES) { const b = h("button", "", seg, label); b.type = "button"; b.dataset.mode = k; b.setAttribute("aria-pressed", String(S.mode === k)); b.addEventListener("click", () => { S.mode = k; draw(); }); }
    form();
    const out = h("div", "stack", root); out.id = "lab-results"; out.setAttribute("aria-live", "polite");
    const r = S.results[S.mode];
    if (r) ({ alpha, risk, portfolio, trading })[S.mode](out, r);
    else { const e = h("div", "card empty", out); h("h3", "", e, "No run yet"); h("p", "", e, (S.plan.studies.find((s) => s.key === S.mode) || {}).what || ""); }
  }

  function form() {
    const card = h("div", "card", root); card.dataset.role = "lab-form"; const f = S.form, grid = h("div", "params", card);
    const text = (label, key, type, wide, parent) => { const l = h("label", "f" + (wide ? " wide" : ""), parent || grid, label), i = h("input", "", l); i.type = type || "text"; i.value = f[key]; i.dataset.key = key; i.addEventListener("input", () => { f[key] = i.value; }); return i; };
    text("Tickers (at least " + ({ alpha: 6, risk: 4, portfolio: 4, trading: 6 })[S.mode] + ")", "tickers", "text", true); text("First day", "start", "date");
    const sl = h("label", "f", grid, "Data source"), src = h("select", "", sl); src.dataset.key = "source";
    for (const s of ["", ...(Q.S.catalog ? Q.S.catalog.sources.map((x) => x.name) : [])]) { const o = h("option", "", src, s || "(Backtest tab's source)"); o.value = s; if (s === f.source) o.selected = true; } src.addEventListener("change", () => { f.source = src.value; });
    const select = (label, key, options, parent, title) => { const l = h("label", "f", parent || grid, label); if (title) l.title = title; const s = h("select", "", l); s.dataset.key = key; for (const [v, t] of options) { const o = h("option", "", s, t); o.value = v; if (String(f[key]) === String(v)) o.selected = true; } s.addEventListener("change", () => { f[key] = s.value; if (key === "signal") draw(); }); };
    const sigs = S.plan.signals.map((s) => [s.key, s.label]);
    if (S.mode === "alpha" || S.mode === "trading") { select("Signal", "signal", sigs); text("Forecast horizon (days)", "horizon", "number"); h("p", "hint wide", grid, (S.plan.signals.find((s) => s.key === f.signal) || {}).help || ""); }
    if (S.mode === "risk") { select("Factor model", "kind", [["statistical", "Statistical (principal components)"], ["time_series", "Time series (equal-weighted market)"]]); text("Number of factors (blank: by random-matrix bound)", "n_factors", "number"); }
    if (S.mode === "trading") {
      for (const [k, label, help] of [["cost_bps", "Cost (bp per unit traded)", "Proportional cost on every trade."], ["target_gross", "Target gross exposure", "Risk aversion is set so the full-trading book averages about this gross exposure."], ["lag", "Delay (days)", "Extra days between the decision and the fill, in the last step of the waterfall."]]) {
        const l = h("label", "f", grid, label); l.title = help; const i = h("input", "", l); i.type = "number"; i.step = "any"; i.value = f.trading[k]; i.dataset.key = k; i.addEventListener("input", () => { f.trading[k] = +i.value; }); h("span", "hint", l, help); }
      const lo = h("label", "f inline wide", grid), cb = h("input", "", lo); cb.type = "checkbox"; cb.checked = f.long_only; cb.dataset.key = "long_only"; cb.addEventListener("change", () => { f.long_only = cb.checked; }); lo.append(document.createTextNode("Long only (the position-limit step of the waterfall)"));
    }
    if (S.mode === "portfolio") {
      const fam = {}; for (const m of S.plan.methods) (fam[m.family] = fam[m.family] || []).push(m);
      const box = h("div", "wide", grid); h("div", "hint", box, "Methods");
      for (const [family, ms] of Object.entries(fam)) { const row = h("div", "chips", box); h("span", "hint", row, family + ":"); for (const m of ms) { const l = h("label", "f inline", row); l.title = m.note || ""; const cb = h("input", "", l); cb.type = "checkbox"; cb.dataset.method = m.key; cb.checked = f.methods.has(m.key); cb.addEventListener("change", () => { if (cb.checked) f.methods.add(m.key); else f.methods.delete(m.key); }); l.append(document.createTextNode(" " + m.label)); } }
      for (const s of S.plan.settings) {
        if (s.key === "cov_method") { select(s.label, "cov_method", S.plan.cov_methods.map((c) => [c, c]), grid, s.help); continue; }
        const l = h("label", "f", grid, s.label); l.title = s.help; const i = h("input", "", l); i.type = "number"; i.step = "any"; i.value = f.config[s.key]; i.dataset.key = s.key; i.addEventListener("input", () => { f.config[s.key] = i.value === "" ? "" : +i.value; }); h("span", "hint", l, s.help);
      }
    }
    const row = h("div", "row", card), run = h("button", "btn primary", row, "Run the " + MODES.find((m) => m[0] === S.mode)[1].toLowerCase() + " study"); run.type = "button"; run.disabled = S.busy;
    const status = h("span", "hint", row); status.id = "lab-status"; status.setAttribute("role", "status");
    if (S.mode === "alpha") { const reg = h("button", "btn", row, "Keep this signal"); reg.type = "button"; reg.title = "Store the signal panel as a new version in the registry (unchanged panels keep their version)."; reg.addEventListener("click", () => register(reg, status)); }
    for (const n of S.plan.notes) h("p", "hint", card, n);
    run.addEventListener("click", () => start(run, status));
  }

  const body = () => {
    const f = S.form, b = { tickers: f.tickers.split(/[\s,;]+/).filter(Boolean), start: f.start || undefined, source: f.source || undefined };
    if (S.mode === "alpha") Object.assign(b, { signal: f.signal, horizon: +f.horizon });
    if (S.mode === "trading") Object.assign(b, { signal: f.signal, horizon: +f.horizon, long_only: f.long_only, config: { ...f.trading } });
    if (S.mode === "risk") { b.kind = f.kind; if (f.n_factors !== "" && f.n_factors !== null) b.n_factors = +f.n_factors; }
    if (S.mode === "portfolio") { b.methods = [...f.methods]; b.config = { ...f.config, cov_method: f.cov_method || f.config.cov_method || "shrinkage" }; for (const k of Object.keys(b.config)) if (b.config[k] === "" || b.config[k] === null) delete b.config[k]; }
    return b;
  };

  async function start(btn, status) {
    const mode = S.mode; S.busy = true; btn.disabled = true; status.textContent = "Starting…";
    try {
      const { job } = await api("/api/lab/" + mode, body());
      S.results[mode] = await Q.waitJob(job, (v) => { status.textContent = `${v.message} (${Math.round(v.elapsed)} s)`; }, 1000);
    } catch (e) { S.busy = false; status.textContent = ""; toast(e.message); h("div", "banner err", $("#lab-results") || root, e.message); btn.disabled = false; return; }
    S.busy = false; draw();
  }

  async function register(btn, status) {
    const f = S.form; btn.disabled = true; status.textContent = "Keeping the signal…";
    try {
      const { job } = await api("/api/lab/signals/register", { ...body(), name: f.signal });
      const m = await Q.waitJob(job, () => {}, 800);
      status.textContent = m.new ? `Kept as ${m.name} version ${m.version}.` : `${m.name} version ${m.version} is unchanged.`;
    } catch (e) { status.textContent = ""; toast(e.message); }
    btn.disabled = false;
  }

  /* ---------------------------------------------------------------- alpha */
  function alpha(out, r) {
    const top = h("div", "card", out); top.dataset.role = "lab-alpha-head"; h("h3", "", top, `${r.label}: ${r.assets.length} assets, ${r.first} to ${r.last}, ${r.horizon}-day horizon`);
    const k = h("div", "kpis", top), ic = r.ic, fit = r.decay.fit, cal = r.calibration;
    Q.tile(k, "Mean IC", num(ic.mean, 4), { text: `t = ${num(ic.t, 1)} (Newey-West, ${Math.max(r.horizon - 1, 0)} lags)`, dir: ic.t > 2 ? "up" : ic.t < -2 ? "down" : "" });
    Q.tile(k, "IC information ratio (annual)", num(ic.ir_annual), { text: `hit rate ${pct(ic.hit_rate, 0)}, sign steady in ${pct(ic.stability, 0)} of blocks` });
    Q.tile(k, "Information half-life", fit.half_life === null || fit.half_life === undefined ? "n/a" : (isFinite(fit.half_life) ? `${num(fit.half_life, 1)} days` : "no decay"), { text: fit.significant ? `IC at lag 1 ${num(fit.ic1, 4)}; fit R² ${num(fit.r2, 2)}` : "the first-lag IC is not distinguishable from zero" });
    Q.tile(k, "Calibration slope", num(cal.slope, 2), { text: cal.verdict, dir: cal.verdict === "calibrated" ? "up" : "" });
    Q.tile(k, "Independent bets a year", num(r.breadth.breadth, 0), { text: `${num(r.breadth.n_effective, 1)} of ${r.breadth.n_assets} assets x ${num(r.breadth.forecasts_per_year, 0)} horizons x ${pct(1 - r.breadth.rho, 0)} new` });
    if (r.transfer) Q.tile(k, "Transfer coefficient (long only)", num(r.transfer.long_only_tc, 2), { text: `IR ${num(r.transfer.ir_unconstrained, 2)} unconstrained, ${num(r.transfer.ir_long_only, 2)} long only` });
    if (r.law) Q.tile(k, "Fundamental law: IC x sqrt(breadth) x TC", num(r.law.ir_at_tc, 2), { text: `${num(r.law.ir_at_tc_1, 2)} at TC = 1 (per year)` });
    for (const n of r.notes) h("p", "hint", top, n);
    const ser = r.ic_series, c = colors();
    const cc = Q.chartCard(out, "Cumulative IC and rolling IC", "The cumulative sum of the daily IC climbs when the signal works. The rolling mean (126 days) shows whether it still does.", { wide: true,
      draw: (b) => C.line(b, { dates: ser.dates, series: [{ name: "Cumulative IC", color: c[0], values: ser.cumulative }], yFormat: (v) => v.toFixed(1), height: 200 }),
      table: (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => ser.dates[i] }, { label: "Cumulative IC", num: true, get: (i) => num(ser.cumulative[i], 2) }, { label: "Rolling mean IC", num: true, get: (i) => num(ser.rolling[i], 4) }], Q.sampleRows(ser.dates, null, Math.max(1, Math.ceil(ser.dates.length / 60)))) });
    cc.dataset.role = "lab-ic";
    const two = h("div", "charts", out), dt = r.decay.table;
    const dc = Q.chartCard(two, "How fast the information fades", `The signal of day t against the return of the single day t+k. ${fit.significant && fit.tau ? `Exponential fit: time constant ${num(fit.tau, 1)} days, half-life ${num(fit.half_life, 1)}, 90% gone after ${num(fit.periods_to_90pct, 0)} days.` : "No significant decay curve to fit."}`,
      { draw: (b) => C.columns(b, { labels: dt.map((x) => String(x.index)), values: dt.map((x) => x.ic), yFormat: (v) => v.toFixed(3), label: "IC by lag", height: 200 }),
        table: (b) => Q.simpleTable(b, [{ label: "Lag (days)", get: (x) => x.index }, { label: "IC", num: true, get: (x) => num(x.ic, 4) }, { label: "Std error", num: true, get: (x) => num(x.se, 4) }, { label: "t", num: true, get: (x) => num(x.t, 1) }], dt) });
    dc.dataset.role = "lab-decay";
    const bins = cal.bins;
    if (bins.length) { const bc = Q.chartCard(two, "Is the alpha the right size?", `Mean realised residual return against mean forecast in five groups. On the diagonal is calibrated; the regression slope is ${num(cal.slope, 2)} ± ${num(cal.se, 2)} (${cal.verdict}).`,
      { draw: (b) => C.line(b, { xs: bins.map((x) => x.alpha), series: [{ name: "Realised", color: c[0], values: bins.map((x) => x.realised) }, { name: "Perfect calibration", color: c[2], values: bins.map((x) => x.alpha), dash: true }], yFormat: (v) => (v * 100).toFixed(2) + "%", height: 200 }),
        table: (b) => Q.simpleTable(b, [{ label: "Mean forecast", num: true, get: (x) => pct(x.alpha, 3) }, { label: "Mean realised", num: true, get: (x) => pct(x.realised, 3) }, { label: "Observations", num: true, get: (x) => String(x.n) }], bins) }); bc.dataset.role = "lab-calibration"; }
    if (r.transfer && r.transfer.ok) {
      const rows = r.transfer.assets.slice().sort((a, b2) => b2.alpha - a.alpha), tc = Q.chartCard(out, "Today's forecast and what a long-only book keeps of it", "Bars: the expected residual return of each asset over the horizon. The table adds the long-only weight that maximises the information ratio.",
        { draw: (b) => C.hbars(b, { rows: rows.map((x) => ({ label: x.asset, value: x.alpha, tip: [["Long-only weight", pct(x.long_only_weight, 1)]] })), format: (v) => (v * 100).toFixed(2) + "%", name: "Alpha", label: "Alpha by asset", labelWidth: 70, valueWidth: 60, color: (v) => (v >= 0 ? css("--s1") : css("--neg")) }),
          table: (b) => Q.simpleTable(b, [{ label: "Asset", get: (x) => x.asset }, { label: "Alpha", num: true, get: (x) => pct(x.alpha, 3) }, { label: "Long-only weight", num: true, get: (x) => pct(x.long_only_weight, 1) }], rows) }); tc.dataset.role = "lab-transfer";
    }
  }

  /* ---------------------------------------------------------------- risk */
  function risk(out, r) {
    const top = h("div", "card", out); top.dataset.role = "lab-risk-head"; h("h3", "", top, `${r.kind === "statistical" ? "Statistical" : "Time-series"} factor model: ${r.assets.length} assets, ${r.window.first} to ${r.window.last}`);
    const k = h("div", "kpis", top), m = r.model;
    Q.tile(k, "Portfolio volatility (equal weight, annual)", pct(r.risk.volatility), { text: `${pct(r.risk.factor_share, 0)} from factors, ${pct(r.risk.specific_share, 0)} specific` });
    Q.tile(k, "Factors", String(m.factors), { text: `${r.spectrum.above_noise} correlation eigenvalues above the random-matrix bound ${num(r.spectrum.noise_bound, 2)}` });
    Q.tile(k, "Median R² of the assets", pct(m.median_r2, 0), { text: `${pct(m.systematic_share, 0)} of variance is systematic` });
    Q.tile(k, "Bias statistic (latest)", num(r.bias.latest, 2), { text: r.bias.verdict, dir: r.bias.verdict === "ok" ? "up" : "down" });
    if (r.neutral) Q.tile(k, `Neutralise ${r.neutral.factor}`, `${pct(r.neutral.volatility_after)} vs ${pct(r.neutral.volatility_before)}`, { text: `exposure ${num(r.neutral.exposure_before, 2)} to 0; trades ${pct(r.neutral.turnover, 0)} of capital` });
    for (const n of r.notes) h("p", "hint", top, n);
    const cs = r.contributions.filter((x) => x.contribution !== null);
    const cc = Q.chartCard(out, "Where the risk comes from", "Each factor's contribution to the portfolio's volatility (annual). With the specific part, they add exactly to the total.",
      { wide: true, draw: (b) => C.hbars(b, { rows: cs.map((x) => ({ label: x.factor, value: x.contribution, tip: [["Group", x.group], ["Exposure", x.exposure === null ? "n/a" : num(x.exposure, 3)]] })), format: (v) => (v * 100).toFixed(2) + "%", name: "Contribution", label: "Risk by factor", labelWidth: 90, valueWidth: 60, color: (v) => (v >= 0 ? css("--s1") : css("--neg")) }),
        table: (b) => Q.simpleTable(b, [{ label: "Factor", get: (x) => x.factor }, { label: "Group", get: (x) => x.group }, { label: "Exposure", num: true, get: (x) => (x.exposure === null ? "" : num(x.exposure, 3)) }, { label: "Contribution to volatility", num: true, get: (x) => pct(x.contribution, 2) }], cs) });
    cc.dataset.role = "lab-risk-factors";
    const two = h("div", "charts", out), bi = r.bias, c = colors();
    const bc = Q.chartCard(two, "Did the risk forecast hold?", `Rolling 126-day standard deviation of the portfolio's daily return over the volatility forecast made the day before, with a two-standard-error band. ${bi.verdict}.`,
      { draw: (b) => C.line(b, { dates: bi.dates, series: [{ name: "Bias statistic", color: c[0], values: bi.values }, { name: "Lower band", color: c[2], values: bi.low, dash: true }, { name: "Upper band", color: c[2], values: bi.high, dash: true }], baseline: 1, yFormat: (v) => v.toFixed(2), height: 200 }),
        table: (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => bi.dates[i] }, { label: "Bias", num: true, get: (i) => num(bi.values[i], 3) }], Q.sampleRows(bi.dates, null, Math.max(1, Math.ceil(bi.dates.length / 40)))) });
    bc.dataset.role = "lab-bias";
    const ev = r.spectrum.eigenvalues, sc = Q.chartCard(two, "Eigenvalues of the correlation matrix", `The first ${r.spectrum.above_noise} stand above the random-matrix bound of ${num(r.spectrum.noise_bound, 2)}: the rest are what noise alone would give.`,
      { draw: (b) => C.columns(b, { labels: ev.map((_, i) => String(i + 1)), values: ev, yFormat: (v) => v.toFixed(1), label: "Eigenvalues", height: 200 }), table: (b) => Q.simpleTable(b, [{ label: "#", get: (x) => x.i }, { label: "Eigenvalue", num: true, get: (x) => num(x.v, 3) }], ev.map((v, i) => ({ i: i + 1, v }))) });
    sc.dataset.role = "lab-spectrum";
    const ar = h("div", "card", out); ar.dataset.role = "lab-asset-risk"; h("h3", "", ar, "Risk by asset");
    Q.simpleTable(ar, [{ label: "Asset", get: (x) => x.asset }, { label: "Weight", num: true, get: (x) => pct(x.weight, 1) }, { label: "Contribution to volatility", num: true, get: (x) => pct(x.contribution, 2) }, { label: "Marginal contribution", num: true, get: (x) => num(x.marginal, 3) }], r.assets_risk);
  }

  /* ---------------------------------------------------------------- portfolio */
  function portfolio(out, r) {
    const cfg = r.config, rows = r.table.slice(), top = h("div", "card", out); top.dataset.role = "lab-pf-head";
    h("h3", "", top, `${rows.length} methods, ${cfg.assets} assets, ${cfg.first} to ${cfg.last}`);
    h("p", "sub", top, `Window ${cfg.lookback} days, rebalanced every ${cfg.step}, cost ${cfg.cost_bps} bp, ${cfg.cov_method} covariance, cap ${pct(cfg.max_weight, 0)} per asset.`);
    for (const n of r.notes) h("p", "hint", top, n);
    const tc = h("div", "card", out); tc.dataset.role = "lab-pf-table"; h("h3", "", tc, "Results, net of costs");
    Q.simpleTable(tc, [{ label: "Method", get: (x) => x.method }, { label: "Sharpe", num: true, get: (x) => num(x.sharpe, 2) }, { label: "± s.e.", num: true, get: (x) => num(x.sharpe_se, 2) }, { label: "Return", num: true, get: (x) => pct(x.ann_return) }, { label: "Volatility", num: true, get: (x) => pct(x.ann_vol) },
      { label: "Max drawdown", num: true, get: (x) => pct(x.max_drawdown) }, { label: "CDaR 95%", num: true, get: (x) => pct(x.cdar_95, 2) }, { label: "Turnover / year", num: true, get: (x) => num(x.turnover_per_year, 1) }, { label: "Effective N", num: true, get: (x) => num(x.effective_n, 1) },
      { label: "Diversification", num: true, get: (x) => num(x.diversification_ratio, 2) }, { label: "Largest weight", num: true, get: (x) => pct(x.max_weight, 0) }, { label: "Weight moved by cap", num: true, get: (x) => pct(x.capped, 1) }, { label: "Instability", num: true, get: (x) => (x.instability === null ? "not run" : num(x.instability, 2)) },
      { label: "Fallbacks", num: true, get: (x) => String(x.fallbacks) }, { label: "Sharpe 1st / 2nd half", num: true, get: (x) => `${num(x.sharpe_first_half, 2)} / ${num(x.sharpe_second_half, 2)}` }], rows);
    const eq = r.equity, keys = Object.keys(eq.series), ranked = rows.slice().sort((a, b) => (b.sharpe || -9) - (a.sharpe || -9)).map((x) => x.key), pick = [...new Set([...(keys.includes("equal_weight") ? ["equal_weight"] : []), ...ranked])].slice(0, 8), c = colors(), label = Object.fromEntries(rows.map((x) => [x.key, x.method]));
    const ec = Q.chartCard(out, "Growth of 1, net of costs", pick.length < keys.length ? `The equal-weight baseline and the highest-Sharpe methods (${pick.length} of ${keys.length}); the table has them all.` : "Every method over the same dates.",
      { wide: true, draw: (b) => C.line(b, { dates: eq.dates, series: pick.map((k, i) => ({ name: label[k], color: c[i % 8], values: eq.series[k] })), baseline: 1, yFormat: (v) => v.toFixed(2) }),
        table: (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => eq.dates[i] }, ...pick.map((k) => ({ label: label[k], num: true, get: (i) => num(eq.series[k][i], 3) }))], Q.sampleRows(eq.dates, null, Math.max(1, Math.ceil(eq.dates.length / 60)))) });
    ec.dataset.role = "lab-pf-equity";
    const two = h("div", "charts", out), sel = S.weightsOf && r.mean_weights[S.weightsOf] ? S.weightsOf : ranked[0];
    const wc = Q.chartCard(two, `Average weights: ${label[sel]}`, "The mean of the target weights over every rebalance. Choose another method below.", { draw: (b) => C.hbars(b, { rows: Object.entries(r.mean_weights[sel]).sort((a, b2) => b2[1] - a[1]).map(([a, v]) => ({ label: a, value: v })), format: (v) => (v * 100).toFixed(1) + "%", name: "Weight", label: "Average weight", labelWidth: 70, valueWidth: 50, color: () => css("--s1") }),
      table: (b) => Q.simpleTable(b, [{ label: "Asset", get: (x) => x[0] }, { label: "Average", num: true, get: (x) => pct(x[1], 1) }, { label: "Latest", num: true, get: (x) => pct(r.last_weights[sel][x[0]], 1) }], Object.entries(r.mean_weights[sel])) });
    wc.dataset.role = "lab-pf-weights";
    const pickr = h("div", "chips", wc); for (const x of rows) { const bt = h("button", "btn" + (x.key === sel ? " primary" : ""), pickr, x.method); bt.type = "button"; bt.addEventListener("click", () => { S.weightsOf = x.key; draw(); }); }
    const inst = rows.filter((x) => x.instability !== null);
    if (inst.length) { const ic = Q.chartCard(two, "How much the weights jump when the data are resampled", "Mean one-way turnover between the weights on the real window and on a bootstrap resample of it. A stable method asks for similar portfolios from similar data.",
      { draw: (b) => C.hbars(b, { rows: inst.slice().sort((a, b2) => a.instability - b2.instability).map((x) => ({ label: x.method, value: x.instability })), format: (v) => v.toFixed(2), name: "Instability", label: "Weight instability", labelWidth: 170, valueWidth: 50, color: () => css("--s2") }),
        table: (b) => Q.simpleTable(b, [{ label: "Method", get: (x) => x.method }, { label: "Instability", num: true, get: (x) => num(x.instability, 3) }], inst) }); ic.dataset.role = "lab-pf-instability"; }
  }

  /* ---------------------------------------------------------------- trading */
  function trading(out, r) {
    const top = h("div", "card", out); top.dataset.role = "lab-trading-head"; h("h3", "", top, `${r.label}: ${r.days} days, ${r.first} to ${r.last}`);
    h("p", "sub", top, `Costs ${r.config.cost_bps} bp per unit traded plus quadratic impact; ${r.config.long_only ? "long only" : "long-short"}; target gross ${num(r.config.target_gross, 1)}; risk aversion ${num(r.risk_aversion, 0)}.`);
    for (const n of r.notes) h("p", "hint", top, n);
    const wf = r.waterfall, c = colors();
    const wc = Q.chartCard(out, "Where the paper alpha goes", "The Sharpe ratio of the same forecasts, realised, as each real-world step is added. The table has the annual return and the change from the step before.",
      { wide: true, draw: (b) => C.columns(b, { labels: wf.map((x) => x.index), values: wf.map((x) => x.sharpe), yFormat: (v) => v.toFixed(1), label: "Sharpe by implementation step", height: 220 }),
        table: (b) => Q.simpleTable(b, [{ label: "Step", get: (x) => x.index }, { label: "Sharpe", num: true, get: (x) => num(x.sharpe, 2) }, { label: "Change", num: true, get: (x) => num(x.lost_sharpe, 2) }, { label: "Return (annual)", num: true, get: (x) => pct(x.ann_return, 2) }, { label: "Change", num: true, get: (x) => pct(x.lost_return, 2) }, { label: "Turnover / year", num: true, get: (x) => num(x.turnover_per_year, 1) }], wf) });
    wc.dataset.role = "lab-waterfall";
    const pc = h("div", "card", out); pc.dataset.role = "lab-policies"; h("h3", "", pc, "Four ways to trade the same alpha");
    Q.simpleTable(pc, [{ label: "Policy", get: (x) => x.index }, { label: "Sharpe, net", num: true, get: (x) => (x.turnover_per_year === 0 ? "never traded" : num(x.sharpe, 2)) }, { label: "Sharpe, gross", num: true, get: (x) => (x.turnover_per_year === 0 ? "never traded" : num(x.gross_sharpe, 2)) }, { label: "Return", num: true, get: (x) => pct(x.ann_return, 2) }, { label: "Volatility", num: true, get: (x) => pct(x.ann_vol, 2) },
      { label: "Costs / year", num: true, get: (x) => pct(x.cost_per_year, 2) }, { label: "Turnover / year", num: true, get: (x) => num(x.turnover_per_year, 1) }], r.policies);
  }

  Q.registerPage("alphalab", { render: () => { if (!S.form.tickers) defaults(); draw(); } });
})();
