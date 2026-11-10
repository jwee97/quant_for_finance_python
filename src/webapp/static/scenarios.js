/* The Scenarios tab: make thousands of possible futures for the Backtest tab's tickers with several generators (resampling the history, statistical models, and learned ones: a GAN, a variational autoencoder and a
   diffusion model) and check how closely each one's scenarios resemble the history they were made from. The person ticks generators, a horizon and a number of scenarios and presses one button; the server
   fits and samples each generator, scores it (the shape of a day's returns, the correlations, the tails, the clustering of volatility, the portfolio's worst days) and the page shows the table and the
   distribution of the equal-weighted portfolio's return over the horizon, beside the history's own. It talks to /api/scenarios/*; the work runs as a job on the app's worker.
   Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, table = Q.simpleTable, chartCard = Q.chartCard, api = Q.api;
  const GUIDE = "scenarios";
  const isOpen = () => !Q.$(S.view).hidden;
  const tickersNow = () => Q.S.tickers.filter((x) => x.status !== "bad");
  const HORIZON_TEXT = { 5: "1 week (5 trading days)", 10: "2 weeks (10 trading days)", 21: "1 month (21 trading days)", 63: "3 months (63 trading days)" };
  const FLAG = { close: "✓", far: "!", "n/a": "–" };                                              // a flag is never colour alone
  const pct = (v, d = 1) => (Q.bad(v) ? "n/a" : (v * 100).toFixed(d) + "%");
  const secs = (v) => (v < 1 ? "<1 s" : v < 90 ? Math.round(v) + " s" : (v / 60).toFixed(1) + " min");
  const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];

  const S = {
    id: "scenarios", root: "#scenarios", view: "#view-scenarios", title: "Scenarios", api: 16, mode: "scenarios", busy: "", results: {}, errors: {}, guard: null, go: null, progress: null,
    about: "Make thousands of possible futures for your tickers with several generators, from resampling the past to a GAN, and check how closely each resembles the history it was made from.",
    modes: [["scenarios", "Scenario generators"]], runLabel: { scenarios: "Generate the scenarios" },
    blurb: { scenarios: "Each generator makes the same number of scenarios from the same history. The page scores every one against that history, and none of them knows the future." },
    plan: null, planError: "", loading: false, chosen: null, job: null, form: { horizon: "21", n: "500", seed: "0" },
  };

  /* ---------- the plan ---------- */
  function ensurePlan() { if (!S.plan && !S.planError && !S.loading) setTimeout(loadPlan, 0); }
  async function loadPlan() {
    S.loading = true; S.planError = "";
    try { S.plan = await api("/api/scenarios/plan"); if (!S.chosen) S.chosen = new Set(S.plan.default); S.form.n = String(S.plan.default_n); }
    catch (e) { S.planError = e.status === 404 ? "The app behind this page cannot make scenarios: it is older than the page. " + Q.RESTART : e.message; }
    S.loading = false;
    if (isOpen()) Q.renderLab(S);
  }
  const info = (id) => (S.plan ? S.plan.generators.find((g) => g.id === id) : null);

  /* ---------- the form ---------- */
  S.check = () => {
    if (S.planError) return S.planError;
    if (!S.plan) return "Loading the generators…";
    const names = tickersNow();
    if (!names.length) return "Choose tickers in the Backtest tab first.";
    if (names.length > S.plan.max_assets) return `Scenarios use at most ${S.plan.max_assets} tickers; take some out in the Backtest tab.`;
    if (!S.chosen || !S.chosen.size) return "Tick at least one generator.";
    const n = Number(S.form.n), [lo, hi] = S.plan.n_limits;
    if (!Number.isInteger(n) || n < lo || n > hi) return `The number of scenarios must be a whole number from ${lo} to ${hi}.`;
    if (!isFinite(Number(S.form.seed)) || Number(S.form.seed) < 0) return "The seed must be a number from 0 up.";
    return "";
  };
  S.sections = (panel) => {
    ensurePlan();
    const sec = (title) => { const d = h("details", "", panel); d.open = true; h("summary", "", d, title); return h("div", "body", d); };
    const after = () => Q.syncLab(S);
    const a = sec("1. Which tickers");
    const names = tickersNow().map((x) => x.t);
    const src = (Q.S.catalog.sources || []).find((x) => x.name === Q.S.source);
    const p = h("p", "", a, names.length ? names.join(", ") : "No tickers are chosen."); p.dataset.role = "scenarios-tickers";
    const u = h("p", "hint", a, `The prices come from ${src ? src.label : Q.S.source}${Q.S.start ? ", from " + Q.S.start : ""}: the Backtest tab's choices. The scenarios are for an equal-weighted portfolio of these tickers, and keep how they move together. `);
    const change = h("button", "link", u, "Change them in the Backtest tab"); change.type = "button"; change.addEventListener("click", () => Q.showTab("backtest"));

    const b = sec("2. Which generators");
    if (S.planError) { const e = h("p", "hint", b, S.planError); e.style.color = css("--crit"); }
    else if (!S.plan) h("p", "hint", b, "Loading…");
    else {
      h("p", "hint", b, "Tick the generators to compare. The learned ones train a small neural network on your history and take a few seconds each; the resampling and statistical ones take under two.");
      const quick = h("div", "row tight", b);
      for (const [label, ids] of [["Recommended", S.plan.default], ["All of them", S.plan.generators.map((g) => g.id)], ["Fast ones only", S.plan.generators.filter((g) => !g.heavy).map((g) => g.id)]]) {
        const btn = h("button", "btn small", quick, label); btn.type = "button"; btn.addEventListener("click", () => { S.chosen = new Set(ids); Q.renderLab(S); });
      }
      for (const fam of S.plan.families) {
        const list = S.plan.generators.filter((g) => g.family === fam);
        if (list.length) Q.checkList(b, fam, list.map((g) => [g.id, g.label, g.text]), S.chosen, () => { note(); after(); });
      }
      const summary = h("p", "hint", b, ""); summary.dataset.role = "scenarios-summary";
      const note = () => { const k = S.chosen.size; summary.textContent = `${k} generator${k === 1 ? "" : "s"} ticked.`; };
      note();
    }

    const c = sec("3. How many, how far");
    const horizons = (S.plan ? S.plan.horizons : [5, 10, 21, 63]).map((n) => [String(n), HORIZON_TEXT[n] || `${n} trading days`]);
    Q.labField(c, { key: "horizon", label: "How far ahead", kind: "choice", options: horizons, help: "The length of each scenario, in trading days." }, S.form, after);
    Q.labField(c, { key: "n", label: "Scenarios per generator", kind: "number", step: 100, min: S.plan ? S.plan.n_limits[0] : 100, max: S.plan ? S.plan.n_limits[1] : 2000, help: "More scenarios make the distribution smoother, not truer." }, S.form, after);
    Q.labField(c, { key: "seed", label: "Random seed", kind: "number", step: 1, min: 0 }, S.form, after);
    h("p", "hint", c, "The same seed and settings give the same scenarios.");
  };

  /* ---------- running ---------- */
  function requestBody() {
    const u = Q.universeBody(), f = S.form;
    return { classes: u.classes, start: u.start, source: u.source, refresh: u.refresh, tickers: tickersNow().map((x) => x.t), generators: [...S.chosen], horizon: Number(f.horizon), n: Number(f.n), seed: Number(f.seed) };
  }
  S.run = async () => {
    if (S.busy) return;
    const problem = S.check(); if (problem) { Q.toast(problem); return; }
    const body = requestBody();
    S.busy = "Starting…"; S.job = null; S.results.scenarios = null; S.errors.scenarios = null; Q.syncLab(S); S.draw();
    try {
      const started = await api("/api/scenarios/run", body);
      S.job = started.job; if (isOpen()) S.draw();
      const result = await Q.waitJob(started.job, (v) => { S.busy = `${v.message} (${Math.round(v.elapsed)} s)`; Q.syncLab(S); const el = Q.$('[data-role="scenarios-progress"]'); if (el) el.textContent = S.busy; }, 700);
      S.results.scenarios = { data: result, body }; S.errors.scenarios = null;
    } catch (e) { S.results.scenarios = null; S.errors.scenarios = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + Q.RESTART : e.message; }
    S.busy = ""; S.job = null; Q.syncLab(S); if (isOpen()) { S.draw(); window.scrollTo({ top: 0 }); }
  };
  async function stopRun(button) {
    button.disabled = true; button.textContent = "Stopping after this generator…";
    try { await api("/api/scenarios/stop", { job: S.job }); } catch (e) { Q.toast(e.message); button.disabled = false; button.textContent = "Stop and show what it has so far"; }
  }
  S.draw = () => Q.drawLab(S);
  S.empty = (root) => {
    const c = h("div", "card empty", root);
    if (S.busy) {
      h("div", "spinner", c).style.margin = "0 auto 12px"; h("h2", "", c, "Making the scenarios…");
      h("p", "", c, S.busy).dataset.role = "scenarios-progress";
      h("p", "hint", c, "Each generator is fitted to the history and then sampled; the learned ones train a network first. Closing this page does not stop it.");
      if (S.job) { const stop = h("button", "btn", c, "Stop and show what it has so far"); stop.type = "button"; stop.dataset.action = "stop-scenarios"; stop.addEventListener("click", () => stopRun(stop)); }
      return;
    }
    h("h2", "", c, "Compare ways of making possible futures");
    for (const t of ["Choose the generators and press the button. The app makes the same number of scenarios with each, from the same history of your tickers: resampled stretches of the past, statistical models (a copula, risk factors, AR-GARCH) and learned ones (a GAN, a variational autoencoder, a diffusion model).",
      "Then it scores each set against that history: does a day's return look the same, do the assets move together as they did, are the tails as fat, does volatility cluster, and how bad are the worst days of the portfolio? A learned generator is not better for being a neural network, and the table shows where it falls short."]) h("p", "", c, t);
    const g = h("button", "link", c, "Read the guide"); g.type = "button"; g.addEventListener("click", () => Q.openGuide(GUIDE));
  };

  /* ---------- the result ---------- */
  S.build = (root, d) => {
    const ok = d.generators.filter((g) => !g.error), tickers = d.assets;
    Q.labHead(root, "Scenarios", `${tickers.length} ticker${tickers.length === 1 ? "" : "s"} (${tickers.join(", ")}) · ${d.days.toLocaleString("en-US")} days from ${d.first} to ${d.last} · ${d.n} scenarios of ${d.horizon} days from each of ${ok.length} generator${ok.length === 1 ? "" : "s"}`, GUIDE);
    if (d.stopped) Q.labBanner(root, "Stopped early. ", "Only the generators before the stop are shown.");
    for (const g of d.generators.filter((x) => x.error)) Q.labBanner(root, `${g.label} did not run. `, g.error).dataset.role = "scenarios-failed";
    for (const n of d.notes.filter((x) => !/could not make scenarios/.test(x))) h("p", "hint", root, n);

    const q = W.card(root, "How closely the scenarios resemble the history", "A ✓ means within a rule-of-thumb distance of a perfect copy and a ! means further than that; hover a column name for what it measures. A set can be close on every column and still say nothing about the future.");
    q.dataset.role = "scenarios-quality";
    const metrics = d.metrics.filter((m) => ok.some((g) => g.quality && g.quality[m.id] !== undefined && g.quality[m.id] !== null));
    const cols = [{ label: "Generator", get: (g) => g.label }, { label: "Time", num: true, get: (g) => secs(g.seconds) }];
    for (const m of metrics) cols.push({ label: m.label, num: true, title: `${m.text} A perfect copy scores ${m.target}.`, get: (g) => `${FLAG[g.flags[m.id]] || "–"} ${Q.bad(g.quality[m.id]) ? "n/a" : num(g.quality[m.id], 2)}` });
    table(q, cols, ok);
    const tip = h("p", "hint", q, "Target for a perfect copy: " + metrics.map((m) => `${m.label} ${m.target}`).join(" · ") + ".");
    tip.dataset.role = "scenarios-targets";

    const levels = d.levels, hist = d.history, xs = levels.map((l) => l * 100);
    const series = ok.map((g, i) => ({ name: g.label, color: css(SERIES[i % SERIES.length]), values: g.terminal.quantiles.map((v) => v * 100) }));
    series.push({ name: "The history", color: css("--ink"), values: hist.quantiles.map((v) => v * 100), dash: "5 4" });
    const ch = h("div", "charts", root);
    chartCard(ch, `The portfolio's return over ${d.horizon} days`, "The return of an equal-weighted portfolio over the horizon at each percentile of the scenarios, beside the same percentiles of overlapping stretches of the history (dashed). A generator that tracks the dashed line has the right spread and tails; one that does not is too calm or too wild.", {
      wide: true,
      draw: (b) => C.line(b, { xs, xLabel: "Percentile", series, yFormat: (v) => v.toFixed(0) + "%", tipFormat: (v) => v.toFixed(1) + "%", xFormat: (v) => v + "%", tipTitle: (i) => `${xs[i]}th percentile`, baseline: 0, height: 300, label: "Distribution of the portfolio's return over the horizon" }),
      table: (b) => table(b, [{ label: "Percentile", num: true, get: (i) => `${xs[i]}%` }, ...series.map((s) => ({ label: s.name, num: true, get: (i) => `${s.values[i].toFixed(2)}%` }))], xs.map((_, i) => i)),
    });
    const risk = W.card(root, `Risk over ${d.horizon} days`, "Value at risk is the loss the portfolio stays inside on 95% of scenarios; expected shortfall is the average loss on the worst 5%. The history's figures come from overlapping stretches.");
    risk.dataset.role = "scenarios-risk";
    const rows = [...ok.map((g) => ({ name: g.label, t: g.terminal })), { name: `The history (${hist.count.toLocaleString("en-US")} stretches)`, t: hist }];
    table(risk, [{ label: "", get: (r) => r.name }, { label: "Average return", num: true, get: (r) => pct(r.t.mean, 2) }, { label: "Spread (standard deviation)", num: true, get: (r) => pct(r.t.std, 2) }, { label: "Value at risk 95%", num: true, get: (r) => pct(r.t.var5, 2) }, { label: "Expected shortfall 95%", num: true, get: (r) => pct(r.t.es5, 2) }], rows);

    const gl = W.card(root, "What each generator assumes", "And where it breaks.");
    table(gl, [{ label: "Generator", get: (g) => g.label }, { label: "Assumes", get: (g) => (info(g.id) || {}).assumes || "" }, { label: "Breaks when", get: (g) => (info(g.id) || {}).fails || "" }], ok);
    const how = h("details", "how", root); h("summary", "", how, "How it is scored");
    for (const m of d.metrics) h("p", "", how, `${m.label}: ${m.text} Perfect: ${m.target}. The page calls it close within ${m.tolerance}.`);
    h("p", "", how, "The limits are rules of thumb for reading the table, not a test: a generator can be close on every column and still be wrong about the future, and a far column may not matter for what you want the scenarios for.");
  };

  Q.registerWorkspace("scenarios", S);
})();
