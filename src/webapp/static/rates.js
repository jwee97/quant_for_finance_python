/* The Rates tab: fit a Nelson-Siegel or Svensson curve to a yield curve typed or pasted in, and follow the level, slope and curvature of a history of curves (the Diebold-Li dynamic Nelson-Siegel model),
   with a walk-forward check of its forecast against the random walk. Nothing is fetched from a data provider: the yields are what the person gives, or a simulated history that says it is simulated.
   It talks to /api/rates/*. Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable, api = Q.api;
  const GUIDE = "technique-yield-curves-and-fixed-income";
  const isOpen = () => !Q.$(R.view).hidden;
  const rate = (v, d = 2) => (Q.bad(v) ? "n/a" : v.toFixed(d) + "%");
  const bp = (v, d = 1) => (Q.bad(v) ? "n/a" : v.toFixed(d) + " bp");
  const MODEL = { nelson_siegel: "Nelson-Siegel", svensson: "Svensson" };

  const R = {
    id: "rates", root: "#rates", view: "#view-rates", title: "Rates", api: 15, mode: "curve", busy: "", results: {}, errors: {}, guard: null, go: null, progress: null,
    about: "Fit a smooth curve to a set of yields, read its level, slope and curvature, and follow how they move through time.",
    modes: [["curve", "Yield curve"], ["dynamic", "Factors over time"]], runLabel: { curve: "Fit the curve", dynamic: "Analyse the history" },
    blurb: { curve: "Type or paste a curve (a maturity and a yield on each line) and fit a Nelson-Siegel or Svensson curve: its level, slope and curvature, how closely it passes through each yield, and the zero, forward and discount curves it implies.",
      dynamic: "Fit the three factors every day of a history of curves, see how persistent each is, forecast the curve ahead and find out whether that forecast has beaten the random walk." },
    plan: null, planError: "", loading: false,
    form: { model: "nelson_siegel", kind: "zero", unit: "percent", text: "", lam: "", source: "simulated", panel: "", dlam: "0.5", horizon: "21", test_days: "750", step: "5", seed: "0", dunit: "percent" },
  };

  /* ---------- the plan ---------- */
  function ensurePlan() { if (!R.plan && !R.planError && !R.loading) setTimeout(loadPlan, 0); }
  async function loadPlan() {
    R.loading = true; R.planError = "";
    try { R.plan = await api("/api/rates/plan"); if (!R.form.text) R.form.text = R.plan.example.text; }
    catch (e) { R.planError = e.status === 404 ? "The app behind this page cannot fit curves: it is older than the page. " + Q.RESTART : e.message; }
    R.loading = false;
    if (isOpen()) Q.renderLab(R);
  }

  /* ---------- the form ---------- */
  R.check = () => {
    const f = R.form;
    if (R.planError) return R.planError;
    if (!R.plan) return "Loading…";
    if (R.mode === "curve") {
      if (!f.text.trim()) return "Type the curve: a maturity and a yield on each line, such as 2Y 4.25.";
      if (f.lam.trim() !== "" && !(Number(f.lam) >= 0.05 && Number(f.lam) <= 6)) return "The decay is a number between 0.05 and 6 (leave it empty to find it).";
      return "";
    }
    if (f.source === "text" && !f.panel.trim()) return "Paste the history, or use the simulated one.";
    for (const [k, lo, hi, name] of [["dlam", 0.05, 6, "The decay"], ["horizon", 1, 250, "The horizon"], ["test_days", 100, 5000, "The test window"], ["step", 1, 60, "The step"]]) {
      const v = Number(f[k]); if (!isFinite(v) || v < lo || v > hi) return `${name} must be a number from ${lo} to ${hi}.`;
    }
    return "";
  };
  R.sections = (panel) => {
    ensurePlan();
    const f = R.form, after = () => Q.syncLab(R), sec = (title) => { const d = h("details", "", panel); d.open = true; h("summary", "", d, title); return h("div", "body", d); };
    if (R.planError) { h("p", "hint tone-crit", sec("Not available"), R.planError); return; }
    if (R.mode === "curve") {
      const a = sec("1. The yields");
      const box = h("textarea", "", a); box.rows = 11; box.value = f.text; box.spellcheck = false; box.dataset.role = "rates-text"; box.setAttribute("aria-label", "maturities and yields, one pair on each line");
      box.addEventListener("input", () => { f.text = box.value; after(); });
      h("p", "hint", a, "One maturity and one yield on each line: 3M 5.10, 2Y 4.35, 10 4.3. A maturity is in years (0.25) or written 3M, 2Y, 30Y.");
      if (R.plan) { const ex = h("button", "btn small", a, "Use the made-up example"); ex.type = "button"; ex.dataset.action = "rates-example"; ex.addEventListener("click", () => { f.text = R.plan.example.text; Q.renderLab(R); }); }
      Q.labField(a, { key: "unit", label: "Yields are written in", kind: "choice", options: [["percent", "Percent (4.25)"], ["decimal", "Decimals (0.0425)"]] }, f, after);
      Q.labField(a, { key: "kind", label: "The yields are", kind: "choice", options: (R.plan ? R.plan.kinds : []).map((k) => [k.id, k.label]), help: "Par yields of coupon bonds are turned into zero (spot) rates first; they are close to zero rates only when the curve is flat." }, f, after);
      const b = sec("2. The model");
      Q.labField(b, { key: "model", label: "Curve", kind: "choice", options: (R.plan ? R.plan.models : []).map((m) => [m.id, m.label]) }, f, () => Q.renderLab(R));
      const info = R.plan && R.plan.models.find((m) => m.id === f.model);
      if (info) { h("p", "hint", b, info.text); h("p", "hint", b, `Assumes: ${info.assumes} Fails: ${info.fails} Needs at least ${info.min_points} points.`); }
      if (f.model === "nelson_siegel") Q.labField(b, { key: "lam", label: "Decay λ per year (empty: find it)", kind: "text", placeholder: "0.6", help: "The curvature factor peaks at 1.79/λ years. Fix it to compare curves on the same footing; leave it empty to let the data choose." }, f, after);
    } else {
      const a = sec("1. The history of curves");
      Q.labField(a, { key: "source", label: "Use", kind: "choice", options: [["simulated", "A simulated history"], ["text", "A history I paste"]] }, f, () => Q.renderLab(R));
      if (f.source === "text") {
        const box = h("textarea", "", a); box.rows = 9; box.value = f.panel; box.spellcheck = false; box.dataset.role = "rates-panel"; box.placeholder = "date,3M,2Y,10Y,30Y\n2024-01-02,5.3,4.4,4.0,4.2";
        box.setAttribute("aria-label", "a history of yield curves as CSV"); box.addEventListener("input", () => { f.panel = box.value; after(); });
        const lab = h("label", "f", a); lab.appendChild(document.createTextNode("or open a CSV file"));
        const file = h("input", "", lab); file.type = "file"; file.accept = ".csv,.txt,text/csv,text/plain"; file.setAttribute("aria-label", "open a CSV file");
        file.addEventListener("change", () => { const pick = file.files && file.files[0]; if (!pick) return; if (pick.size > 140000) { Q.toast("That file is larger than the page reads (about 140 KB)"); return; } const rd = new FileReader(); rd.onload = () => { f.panel = String(rd.result); Q.renderLab(R); }; rd.readAsText(pick); });
        h("p", "hint", a, "A header of maturities (3M, 2Y, 10Y or years) after a date column, then one row of yields a day, oldest or newest first. A day with a missing yield is left out. At least 250 days.");
        Q.labField(a, { key: "dunit", label: "Yields are written in", kind: "choice", options: [["percent", "Percent (4.25)"], ["decimal", "Decimals (0.0425)"]] }, f, after);
      } else {
        h("p", "hint", a, "Simulated curves: Nelson-Siegel factors that move as a VAR(1), with a basis point of noise. It shows how the analysis reads and says nothing about real rates.");
        Q.labField(a, { key: "seed", label: "Seed", kind: "number", min: 0, step: 1 }, f, after);
      }
      const b = sec("2. The analysis");
      Q.labField(b, { key: "dlam", label: "Decay λ per year", kind: "number", min: 0.05, max: 6, step: 0.05, help: "Fixed for the whole history, as in Diebold and Li (2006): the three factors are then a linear regression of each day's curve." }, f, after);
      Q.labField(b, { key: "horizon", label: "Forecast how many days ahead", kind: "number", min: 1, max: 250, step: 1 }, f, after);
      Q.labField(b, { key: "test_days", label: "Walk-forward window (days)", kind: "number", min: 100, max: 5000, step: 50, help: "The forecast is made again at many past dates with only the data up to each, and scored against the curve that followed." }, f, after);
      Q.labField(b, { key: "step", label: "Days between test dates", kind: "number", min: 1, max: 60, step: 1 }, f, after);
    }
  };

  /* ---------- the request ---------- */
  R.run = async () => {
    if (R.busy) return;
    const problem = R.check(); if (problem) { Q.toast(problem); return; }
    const f = R.form, mode = R.mode;
    let path, body;
    if (mode === "curve") { path = "/api/rates/curve"; body = { text: f.text, model: f.model, kind: f.kind, unit: f.unit }; if (f.model === "nelson_siegel" && f.lam.trim() !== "") body.lambda = Number(f.lam); }
    else { path = "/api/rates/dynamic"; body = { source: f.source, lambda: Number(f.dlam), horizon: Number(f.horizon), test_days: Number(f.test_days), step: Number(f.step), unit: f.dunit }; if (f.source === "text") body.text = f.panel; else body.seed = Number(f.seed); }
    R.busy = mode === "curve" ? "Fitting the curve…" : "Following the factors and testing the forecast…"; Q.syncLab(R);
    try { R.results[mode] = { data: await api(path, body), body }; R.errors[mode] = null; }
    catch (e) { R.results[mode] = null; R.errors[mode] = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + Q.RESTART : e.message; }
    R.busy = ""; Q.syncLab(R); if (isOpen()) R.draw();
  };
  R.draw = () => Q.drawLab(R);
  R.empty = (root) => {
    const c = h("div", "card empty", root);
    if (R.mode === "curve") {
      h("h2", "", c, "Read the level, slope and curvature of a yield curve");
      h("p", "", c, "Type the yields you have (a maturity and a yield on each line) and press the button. The page fits a Nelson-Siegel or Svensson curve through them and says how well it fits, what its three or four factors are, and what the curve implies between and beyond the maturities you gave.");
      h("p", "hint", c, "The made-up example in the box is not market data: replace it with yields you have.");
    } else {
      h("h2", "", c, "Follow the factors of the curve through time");
      h("p", "", c, "Press the button to analyse the simulated history, or paste your own: the page fits the level, slope and curvature every day, forecasts the curve ahead from a VAR(1) on them, and tests that forecast against the random walk on dates it had not seen.");
    }
  };
  R.build = (root, d) => (R.mode === "curve" ? buildCurve(root, d) : buildDynamic(root, d));

  /* ---------- the curve ---------- */
  function buildCurve(root, d) {
    const f = d.factors, fit = d.fit, c = d.curve;
    Q.labHead(root, `${MODEL[d.model]} curve`, `${fit.points.length} points · it misses the yields by ${bp(fit.rmse_bp)} on average · observed maturities ${d.observed_range[0]} to ${d.observed_range[1]} years`, GUIDE);
    for (const w of d.warnings) Q.labBanner(root, "Look at this. ", w).dataset.role = "rates-warning";
    for (const n of d.notes) h("p", "hint", root, n);
    const k = h("div", "earnrow", root);
    tile(k, "Level", rate(f.level), { text: "the yield at very long maturities (β0)" });
    tile(k, "Short rate", rate(f.short_rate), { text: "the yield as maturity goes to zero (β0 + β1)" });
    tile(k, "Slope", rate(f.slope), { text: "long rate less short rate (−β1)" }, "", "Negative: the curve is inverted");
    tile(k, "Curvature", rate(f.curvature), { text: d.model === "svensson" ? `second ${rate(f.second_curvature)}` : "a hump when positive, a trough when negative (β2)" });
    tile(k, "Hump at", `${num(f.hump_maturity, 1)} years`, { text: d.model === "svensson" ? `second ${num(f.second_hump_maturity, 1)} years` : "where the curvature factor peaks (1.79/λ)" });
    tile(k, "Fit", bp(fit.rmse_bp), { text: `largest miss ${bp(fit.max_abs_bp)}` });
    const ch = h("div", "charts", root), mats = c.maturities, seen = new Map(fit.points.map((p) => [p.maturity, p.observed]));
    chartCard(ch, "The curve through the yields", "The fitted zero (spot) curve and the forward curve it implies, with the yields that were fitted as dots. Where the forward curve lies above the zero curve the zero curve is rising.", {
      wide: true,
      draw: (b) => C.line(b, { xs: mats, xLabel: "Maturity (years)", series: [{ name: "Zero curve", color: css("--s1"), values: c.zero }, { name: "Forward curve", color: css("--s2"), values: c.forward, dash: "5 4" }, { name: "Observed", color: css("--s3"), values: mats.map((m) => (seen.has(m) ? seen.get(m) : null)), dots: true, dotSize: 4 }],
        yFormat: (v) => v.toFixed(1) + "%", tipFormat: (v) => v.toFixed(2) + "%", xFormat: (v) => num(v, 0), tipTitle: (i) => `${num(mats[i], 2)} years`, height: 280, label: "Fitted yield curve", vlines: [{ x: d.observed_range[1], label: "Last observed" }] }),
      table: (b) => table(b, [{ label: "Maturity (years)", num: true, get: (i) => num(mats[i], 2) }, { label: "Zero", num: true, get: (i) => rate(c.zero[i]) }, { label: "Forward", num: true, get: (i) => rate(c.forward[i]) }, { label: "Discount factor", num: true, get: (i) => num(c.discount[i], 4) }], mats.map((_, i) => i).filter((i) => i % 3 === 0)) });
    const pts = fit.points;
    chartCard(ch, "Where the curve misses", "The fitted curve less each observed yield, in basis points. A curve that is right on average but misses in a pattern (always high at the short end, low in the middle) is missing a factor.", {
      draw: (b) => C.line(b, { xs: pts.map((p) => p.maturity), xLabel: "Maturity (years)", series: [{ name: "Observed less fitted", color: css("--s1"), values: pts.map((p) => p.residual_bp), dots: true }], baseline: 0, include: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2) + " bp", xFormat: (v) => num(v, 0), height: 240, label: "Residuals of the fit" }),
      table: (b) => table(b, [{ label: "Maturity", num: true, get: (p) => num(p.maturity, 2) }, { label: "Observed", num: true, get: (p) => rate(p.observed) }, { label: "Fitted", num: true, get: (p) => rate(p.fitted) }, { label: "Observed less fitted", num: true, get: (p) => bp(p.residual_bp, 2) }], pts) });
    const sh = W.card(root, "The parameters", "What each number means; rates are in percent."); sh.dataset.role = "rates-parameters";
    table(sh, [{ label: "Parameter", get: (x) => x.name }, { label: "Value", num: true, get: (x) => (x.value === null || x.value === undefined ? "" : num(x.value, 3)) }, { label: "Meaning", get: (x) => x.text }], d.shape);
    if (d.key_rates.length) {
      const kr = W.card(root, "Rates the curve implies", "The zero (spot) rate to each maturity, the forward rate at that maturity, and what one unit paid then is worth today.");
      table(kr, [{ label: "Maturity (years)", num: true, get: (x) => num(x.maturity, 2) }, { label: "Zero rate", num: true, get: (x) => rate(x.zero) }, { label: "Forward rate", num: true, get: (x) => rate(x.forward) }, { label: "Discount factor", num: true, get: (x) => num(x.discount, 4) }], d.key_rates);
    }
    const how = h("details", "how", root); h("summary", "", how, "How it is fitted");
    h("p", "", how, "Nelson and Siegel (1987) write the zero rate at maturity t as β0 + β1 (1−e^(−λt))/(λt) + β2 [(1−e^(−λt))/(λt) − e^(−λt)]: three loadings that are a constant (the level), a factor that falls from 1 to 0 (the slope) and a hump (the curvature). For a given decay λ the betas are an ordinary regression; the page searches λ on a grid and keeps the one with the smallest squared miss. Svensson (1994) adds a second hump with its own decay. The yields are treated as continuously compounded zero rates; par yields are bootstrapped to zero rates first. The curve describes the yields it was given: it is not a forecast, and it says nothing between and beyond the maturities except what the model's shape implies.");
  }

  /* ---------- the history ---------- */
  function buildDynamic(root, d) {
    const L = d.latest, wf = d.walk_forward;
    Q.labHead(root, "The factors of the curve through time", `${d.days.toLocaleString("en-US")} days to ${L.date} · decay ${d.lambda}/year · the curve is fitted every day with an average miss of ${bp(d.fit_rmse_bp)}`, GUIDE);
    if (d.note) Q.labBanner(root, "Simulated. ", d.note).dataset.role = "rates-simulated";
    const k = h("div", "earnrow", root);
    tile(k, "Level today", rate(L.factors.level), { text: `forecast in ${d.horizon} days ${rate(L.forecast_factors.level)}` });
    tile(k, "Slope today", rate(L.factors.slope), { text: `forecast ${rate(L.forecast_factors.slope)}` }, "", "Long rate less short rate");
    tile(k, "Curvature today", rate(L.factors.curvature), { text: `forecast ${rate(L.forecast_factors.curvature)}` });
    if (wf) {
      const better = wf.rmse_bp.dynamic_ns < wf.rmse_bp.random_walk * 0.95;
      tile(k, `Forecast miss at ${wf.horizon} days`, bp(wf.rmse_bp.dynamic_ns), { text: `random walk ${bp(wf.rmse_bp.random_walk)}`, dir: better ? "up" : "flat" }, "", "Root-mean-square error of the forecast curve against the curve that followed, over all maturities and test dates");
    }
    const ch = h("div", "charts", root), dates = d.dates;
    chartCard(ch, "Level, slope and curvature", "The three factors fitted to each day's curve, in percent. The level is the long-run yield, the slope is the long rate less the short rate, the curvature is the hump.", {
      wide: true, draw: (b) => C.line(b, { dates, series: [{ name: "Level", color: css("--s1"), values: d.factors.level }, { name: "Slope", color: css("--s2"), values: d.factors.slope }, { name: "Curvature", color: css("--s3"), values: d.factors.curvature }], baseline: 0, yFormat: (v) => v.toFixed(1) + "%", tipFormat: (v) => v.toFixed(2) + "%", height: 280, label: "Factors through time" }),
      table: (b) => table(b, [{ label: "Date", get: (i) => dates[i] }, { label: "Level", num: true, get: (i) => rate(d.factors.level[i]) }, { label: "Slope", num: true, get: (i) => rate(d.factors.slope[i]) }, { label: "Curvature", num: true, get: (i) => rate(d.factors.curvature[i]) }], dates.map((_, i) => i).filter((i) => i % 5 === 0)) });
    const mats = d.maturities;
    chartCard(ch, `The latest curve and its forecast`, `The curve on ${L.date}, the Nelson-Siegel curve fitted to it, and the curve ${d.horizon} days ahead that a VAR(1) on the three factors forecasts.`, {
      draw: (b) => C.line(b, { xs: mats, xLabel: "Maturity (years)", series: [{ name: "Observed", color: css("--s3"), values: L.observed, dots: true }, { name: "Fitted", color: css("--s1"), values: L.fitted }, { name: `Forecast in ${d.horizon} days`, color: css("--s2"), values: L.forecast, dash: "5 4" }], yFormat: (v) => v.toFixed(1) + "%", tipFormat: (v) => v.toFixed(2) + "%", xFormat: (v) => num(v, 0), height: 260, label: "Latest curve and forecast" }),
      table: (b) => table(b, [{ label: "Maturity", num: true, get: (i) => num(mats[i], 2) }, { label: "Observed", num: true, get: (i) => rate(L.observed[i]) }, { label: "Fitted", num: true, get: (i) => rate(L.fitted[i]) }, { label: "Forecast", num: true, get: (i) => rate(L.forecast[i]) }], mats.map((_, i) => i)) });
    if (wf) {
      const c = W.card(root, "Has the forecast beaten the random walk?", `At ${wf.origins} dates from ${wf.first} to ${wf.last} the curve ${wf.horizon} days ahead was forecast using only the data up to the date, and scored against the curve that followed. The random walk says the curve will be what it is today.`); c.dataset.role = "rates-walk";
      const better = wf.rmse_bp.dynamic_ns < wf.rmse_bp.random_walk * 0.95, worse = wf.rmse_bp.dynamic_ns > wf.rmse_bp.random_walk * 1.05;
      h("p", better ? "tone-good" : worse ? "tone-warn" : "", c, better ? `Yes: the forecast missed by ${bp(wf.rmse_bp.dynamic_ns)} against ${bp(wf.rmse_bp.random_walk)}, and was closer on ${Math.round(wf.better_share * 100)}% of the dates.`
        : worse ? `No: the random walk was closer (${bp(wf.rmse_bp.random_walk)} against ${bp(wf.rmse_bp.dynamic_ns)}).` : `Not clearly: the two missed by about the same (${bp(wf.rmse_bp.dynamic_ns)} against ${bp(wf.rmse_bp.random_walk)}), and the forecast was closer on ${Math.round(wf.better_share * 100)}% of the dates.`);
      table(c, [{ label: "Maturity (years)", num: true, get: (x) => num(x.maturity, 2) }, { label: "Factor forecast miss", num: true, get: (x) => bp(x.dynamic_ns) }, { label: "Random walk miss", num: true, get: (x) => bp(x.random_walk) }], wf.by_maturity);
    }
    const pc = W.card(root, "How persistent the factors are", "The first-order autocorrelation of each factor's daily values: close to 1 means it drifts slowly (a level that is hard to forecast beyond itself).");
    table(pc, [{ label: "Factor", get: (x) => x.factor }, { label: "Autocorrelation", num: true, get: (x) => num(x.ar1, 4) }, { label: "Mean", num: true, get: (x) => rate(x.mean) }, { label: "Standard deviation", num: true, get: (x) => rate(x.std) }], d.persistence);
    const how = h("details", "how", root); h("summary", "", how, "How it is done");
    h("p", "", how, "Diebold and Li (2006) fix the decay λ and treat each day's three factors as the regression of that day's yields on the Nelson-Siegel loadings. The factors then form a time series; a first-order vector autoregression on them is iterated ahead and mapped back to a curve. The walk-forward check repeats the forecast at past dates with only the data before each, so the comparison with the random walk is on curves the model had not seen.");
  }

  Q.registerWorkspace("rates", R);
})();
