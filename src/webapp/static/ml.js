/* The Machine learning workspace. A model is trained only inside the whole pipeline: features built without look-ahead, walk-forward fits whose training rows are purged around every test day, a backtest
   with costs through the platform's engine, and a result cut into train (in-sample), validation (where choices are made), test and forward (reserved) segments, with every look at the same data counted.
   Nine views share one form and one idea of the current experiment: Dataset, Train, Experiments, Compare, Explain, Overfitting, Search, Combine and Forward test. It talks to /api/ml/*.
   Whatever the server sends goes in with textContent. The ETF and crypto universes are platform data; the futures universe is simulated and every payload says so, and this page prints it. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable, api = Q.api;
  const GUIDE = "technique-machine-learning-research-pipeline";
  const SEGMENTS = ["train", "validation", "test", "forward"], OOS = ["validation", "test", "forward"];
  const SEG_NAME = { train: "Train (in-sample)", validation: "Validation", test: "Test", forward: "Forward" };
  const SORTS = [["validation.sharpe", "Validation Sharpe"], ["test.sharpe", "Test Sharpe"], ["forward.sharpe", "Forward Sharpe"], ["oos.sharpe", "Sharpe over all out-of-sample days"], ["ic.mean", "Rank IC"], ["ic.ir", "Rank IC information ratio"], ["overfit.generalization_score", "Generalization score"]];
  const SIGNAL_NAME = { top_n_long_short: "Long the top n, short the bottom n", long_only_top_quantile: "Long only: the top quantile", proportional: "Proportional to the score", vol_scaled: "Score divided by volatility", rank_normalised: "Weighted by rank",
    threshold: "Long above, short below a threshold", regime_filtered: "Only in the calm regime", confidence_weighted: "Score times the model's confidence" };
  const SIGNAL_FIELDS = { top_n_long_short: ["n"], long_only_top_quantile: ["quantile"], threshold: ["threshold"] };
  const METHOD_NAME = { equal: "Equal weight", inverse_vol: "Inverse volatility", risk_parity: "Risk parity", mean_variance: "Mean-variance", bayesian: "Bayes-Stein", regime_conditioned: "Regime-conditioned", online_learning: "Online learning (Hedge)",
    bandit: "Bandit (Thompson)", best_trailing: "Best trailing Sharpe", regime_selector: "Best in this regime" };
  const STAGE_COLOR = { random: "--s1", bayes: "--s2", genetic: "--s3" };
  const expColor = W.palette();
  const cap = (s) => String(s).charAt(0).toUpperCase() + String(s).slice(1);
  const sig = (v, d = 2) => (Q.bad(v) ? "n/a" : num(v, d));
  const spct = (v, d = 1) => (Q.bad(v) ? "n/a" : Q.fmt.spct(v, d));
  const sct = (v) => (Q.bad(v) ? "n/a" : String(+Number(v).toPrecision(2)));                 // a score or a contribution: a few significant figures, whatever its size
  const ssct = (v) => (Q.bad(v) ? "n/a" : (v >= 0 ? "+" : "−") + String(+Math.abs(v).toPrecision(2)));
  const pc = (v) => String(+(v * 100).toFixed(6));                                          // 0.15 -> "15", for a field that is typed in percent
  const pctTick = (v) => { const p = +(v * 100).toFixed(2); return (p % 1 === 0 ? String(p) : p.toFixed(1)) + "%"; };       // an axis label: -2.5%, not -3%
  const shortId = (id) => String(id).slice(-6);
  const expLabel = (r) => `${r.name} · ${r.model} (${shortId(r.id)})`;
  const tickerList = (text) => String(text).split(/[\s,;]+/).filter(Boolean).map((s) => s.toUpperCase());
  const needNumber = (s, name) => { const t = String(s).trim(), v = Number(t); if (t === "" || !isFinite(v)) throw new Error(`${name} must be a number`); return v; };
  const optNumber = (s, name) => (String(s).trim() === "" ? undefined : needNumber(s, name));

  const M = {
    id: "ml", root: "#ml", view: "#view-ml", title: "Machine learning", api: 4, mode: "experiments", busy: "", results: {}, errors: {}, cache: new Map(), guard: null, go: null, progress: null,
    about: "Train models only inside the whole pipeline: features without look-ahead, walk-forward fits, a backtest with costs, and results cut into train, validation, test and forward segments, with every look at the data counted.",
    modes: [["dataset", "Dataset"], ["train", "Train"], ["experiments", "Experiments"], ["compare", "Compare"], ["explain", "Explain"], ["overfit", "Overfitting"], ["search", "Search"], ["combine", "Combine"], ["forward", "Forward test"]],
    runLabel: { dataset: "Build the dataset", train: "Train and backtest", experiments: "Refresh", compare: "Compare", explain: "Explain", overfit: "Check again", search: "Run the search", combine: "Combine", forward: "Record today's signal" },
    blurb: { dataset: "Causal features and forward-looking targets, with each feature's coverage and its information coefficient on the warm-up segment only.",
      train: "Walk-forward training, position rules, a backtest with costs, and the result split into train, validation, test and forward segments.",
      experiments: "Every saved experiment, ranked, with the number of looks at the same data when it ran. Open one, delete one, pick several to compare or combine.",
      compare: "Settings and results side by side, equity curves over their shared days, correlations, and a map of how alike the strategies are.",
      explain: "Why the saved model scored an asset as it did on a date: Shapley values split the score exactly into what each feature added.",
      overfit: "How much of the best result survives where nothing was fitted: Sharpe by segment, the deflated Sharpe ratio, stress tests and meta-labelling.",
      search: "Random trials, then Bayesian, then genetic, on validation only; the best few are run on every segment once, and every trial is counted.",
      combine: "Ten ways to weight several strategies, each estimated only from the past, side by side.",
      forward: "Paper signals recorded as the days arrive, settled when their window ends, with the live rank IC and the drift of each feature against the backtest." },
    catalog: null, catalogError: "", board: null, boardPromise: null, picked: new Set(), cur: Q.store("ml.cur") || "",
    form: { name: "", notes: "", universe: "etf", tickers: "", groups: new Set(), kind: "fwd_ret", horizon: "5", normalize: "cross_sectional_z", winsor: "1", minHistory: "252", start: "", end: "", regime: "",
      window: "expanding", rollingDays: "1260", minTrain: "756", refit: "", embargo: "0", valFrac: "20", testFrac: "15", fwdFrac: "15", model: "ridge", params: {},
      signal: "top_n_long_short", n: "3", quantile: "30", threshold: "0", gross: "1", net: "0", maxWeight: "35", holding: "5", rebalance: "weekly", cost: "5", slippage: "2", targetVol: "10", maxLev: "1.5",
      explain: true, robust: false, seed: "0", sort: "validation.sharpe", fModel: "", fStage: "", fUniverse: "", exp: "", asset: "", date: "", perms: "40",
      searchKind: "search", kinds: new Set(), nRandom: "8", nBayes: "6", generations: "2", population: "6", topK: "2", tuneMethod: "random", tuneTrials: "12",
      methods: new Set(), lookback: "252", every: "21", riskAversion: "5", capital: "100000", asOf: "" },
  };

  /* ---------- what the server offers, and the list of experiments ---------- */
  function init(cat) {
    const d = cat.defaults, t = /^(.*)_(\d+)$/.exec(d.dataset.target);
    Object.assign(M.form, { universe: d.dataset.universe, groups: new Set(d.dataset.feature_groups), kind: t ? t[1] : "fwd_ret", horizon: String(d.dataset.horizon), normalize: d.dataset.normalize, winsor: pc(d.dataset.winsor), minHistory: String(d.dataset.min_history),
      window: d.split.window, rollingDays: String(d.split.rolling_days), minTrain: String(d.split.min_train_days), embargo: String(d.split.embargo), valFrac: pc(d.split.validation_fraction), testFrac: pc(d.split.test_fraction), fwdFrac: pc(d.split.forward_fraction),
      model: d.model.kind, signal: d.signal.method, n: String(d.signal.n), quantile: pc(d.signal.quantile), threshold: String(d.signal.threshold), gross: String(d.signal.gross), net: String(d.signal.net), maxWeight: pc(d.signal.max_weight), holding: String(d.signal.holding_period),
      rebalance: d.backtest.rebalance, cost: String(d.backtest.cost_bps), slippage: String(d.backtest.slippage_bps), targetVol: d.backtest.target_vol === null ? "" : pc(d.backtest.target_vol), maxLev: String(d.backtest.max_leverage),
      kinds: new Set(cat.search_kinds), methods: new Set(cat.ensembles.map((e) => e.id)) });
  }
  const isOpen = () => !Q.$(M.view).hidden;
  function loadCatalog() {
    return api("/api/ml/catalog").then((c) => { M.catalog = c; M.catalogError = ""; init(c); if (isOpen()) Q.renderLab(M); }, (e) => { M.catalogError = e.status === 404 ? "The app behind this page is older than the page. " + Q.RESTART : e.message; if (isOpen()) Q.syncLab(M); });
  }
  loadCatalog();
  /* The saved experiments, for the pickers of the views that work on one or several of them. Cleared whenever something is saved or deleted. */
  function ensureBoard() {
    if (M.boardPromise) return M.boardPromise;
    M.boardPromise = api("/api/ml/experiments?limit=500").then((d) => {
      M.board = d;
      const newest = d.rows.slice().sort((a, b) => String(b.created).localeCompare(String(a.created)))[0];
      if (!d.rows.some((r) => r.id === M.cur)) M.cur = newest ? newest.id : "";
      M.form.exp = M.cur; for (const id of [...M.picked]) if (!d.rows.some((r) => r.id === id)) M.picked.delete(id);
      if (isOpen()) Q.renderLab(M);
    }, (e) => { M.boardPromise = null; Q.toast(e.message); });
    return M.boardPromise;
  }
  const invalidate = () => { M.board = null; M.boardPromise = null; M.cache.clear(); M.results.experiments = M.results.models = null; M.errors.experiments = null; };           // something was saved or deleted: the lists are out of date
  const setCurrent = (id) => { M.cur = M.form.exp = id; Q.store("ml.cur", id); };
  const go = (mode) => { M.mode = mode; Q.renderLab(M); };
  const modelKinds = () => { const cls = isClassification(); return M.catalog.models.filter((m) => m.tasks.includes(cls ? "classification" : "regression")); };
  const isClassification = () => !!(M.catalog.targets.find((x) => x.kind === M.form.kind) || {}).classification;
  function fixModel() {
    const ok = modelKinds().filter((m) => m.available).map((m) => m.kind);
    if (!ok.includes(M.form.model)) M.form.model = ok.includes("ridge") ? "ridge" : ok.includes("logistic") ? "logistic" : ok[0];
  }

  /* ---------- the form ---------- */
  M.check = () => {
    const m = M.mode, f = M.form;
    if (!M.catalog) return M.catalogError || "Loading what the page offers…";
    if (["dataset", "train", "search"].includes(m)) {
      if (!f.groups.size) return "Choose at least one group of features.";
      if (f.universe === "custom" && tickerList(f.tickers).length < 2) return "Enter at least two tickers.";
      if (Number(f.valFrac) + Number(f.testFrac) + Number(f.fwdFrac) > 70) return "Validation, test and forward together must leave at least 30% of the days for training.";
    }
    if (m === "train" && !modelKinds().some((k) => k.kind === f.model && k.available)) return "This model cannot predict the chosen target.";
    if (m === "search" && f.searchKind === "search" && !f.kinds.size) return "Choose at least one model to search over.";
    if (m === "compare" && (M.picked.size < 2 || M.picked.size > 8)) return "Choose between two and eight experiments.";
    if (m === "combine" && (M.picked.size < 2 || M.picked.size > 12)) return "Choose between two and twelve experiments.";
    if (m === "combine" && !f.methods.size) return "Choose at least one method.";
    if (["explain", "overfit", "forward"].includes(m) && !M.cur) return M.board ? "Train an experiment first." : "Loading the experiments…";
    return "";
  };
  function universeBlock(b) {
    const f = M.form, cat = M.catalog;
    Q.labField(b, { key: "universe", label: "Universe", kind: "choice", options: [...cat.universes.map((u) => [u.id, u.id === "etf" ? "ETFs (platform data)" : u.id === "crypto" ? "Crypto: BTC and ETH (platform data)" : "Futures (simulated)"]), ["custom", "My tickers"]] }, f, () => Q.renderLab(M));
    const u = cat.universes.find((x) => x.id === f.universe);
    if (u) h("p", "hint", b, u.text);
    if (f.universe === "custom") {
      Q.labField(b, { key: "tickers", label: "Tickers, two or more", kind: "text", placeholder: "SPY, QQQ, IWM, TLT, GLD" }, f, () => Q.syncLab(M));
      h("p", "hint", b, "Prices come from the Backtest tab's source; run a backtest once with new tickers so they are downloaded.");
    }
  }
  function featureBlock(b) {
    Q.checkList(b, "Groups of features, each built without look-ahead:", M.catalog.feature_groups.map((g) => [g.id, `${cap(g.id.replace(/_/g, " "))} (${g.n})`, g.text]), M.form.groups, () => Q.syncLab(M));
    h("p", "hint", b, "Volume, macro, futures, options and crypto features are built only where the data has them; anything skipped is listed with its reason.");
  }
  function targetBlock(b) {
    const f = M.form, cat = M.catalog;
    Q.labField(b, { key: "kind", label: "What to predict", kind: "choice", options: cat.targets.map((t) => [t.kind, cap(t.text)]) }, f, () => { fixModel(); Q.renderLab(M); });
    Q.labField(b, { key: "horizon", label: "Horizon, in trading days", kind: "choice", options: cat.horizons.map((x) => [String(x), String(x)]) }, f, () => Q.syncLab(M));
    h("p", "hint", b, "Decided at the close of day t, traded one day later; the target runs from that trade to the end of the horizon.");
  }
  function prepBlock(b) {
    const f = M.form, after = () => Q.syncLab(M);
    Q.labField(b, { key: "normalize", label: "Scale features each day", kind: "choice", options: [["cross_sectional_z", "Z-score across assets"], ["rank", "Rank across assets"], ["none", "Leave as they are"]] }, f, after);
    Q.labField(b, { key: "winsor", label: "Clip each tail (% of values)", kind: "number", step: 0.5, min: 0, max: 20 }, f, after);
    Q.labField(b, { key: "minHistory", label: "History an asset needs before it enters (days)", kind: "number", step: 21, min: 60, max: 3000 }, f, after);
    Q.labField(b, { key: "start", label: "First date (blank = all data)", kind: "text", placeholder: "2012-01-31" }, f, after);
    Q.labField(b, { key: "end", label: "Last date (blank = all data)", kind: "text", placeholder: "2024-12-31" }, f, after);
    Q.labField(b, { key: "regime", label: "Add a market-regime feature", kind: "choice", options: [["", "None"], ["vol_state", "Volatility state"], ["hmm", "Hidden Markov model"], ["gmm", "Gaussian mixture"]] }, f, after);
  }
  function segmentBlock(b) {
    const f = M.form, after = () => Q.syncLab(M);
    Q.labField(b, { key: "valFrac", label: "Validation, % of the days (choices are made here)", kind: "number", step: 1, min: 5, max: 40 }, f, after);
    Q.labField(b, { key: "testFrac", label: "Test, % of the days (reserved)", kind: "number", step: 1, min: 5, max: 40 }, f, after);
    Q.labField(b, { key: "fwdFrac", label: "Forward, % of the days (reserved)", kind: "number", step: 1, min: 5, max: 40 }, f, after);
    Q.labField(b, { key: "minTrain", label: "Days before the first prediction", kind: "number", step: 21, min: 126, max: 4000 }, f, after);
    Q.labField(b, { key: "refit", label: "Refit every (days; blank = the model's own)", kind: "number", step: 1, min: 1, max: 252 }, f, after);
    Q.labField(b, { key: "embargo", label: "Embargo after each test window (days)", kind: "number", step: 1, min: 0, max: 60, help: "Extra days dropped from training beyond the purge that the label's own length already requires" }, f, after);
    Q.labField(b, { key: "window", label: "Training window", kind: "choice", options: [["expanding", "Expanding: everything before"], ["rolling", "Rolling: the last days only"]] }, f, () => Q.renderLab(M));
    if (f.window === "rolling") Q.labField(b, { key: "rollingDays", label: "Rolling window (days)", kind: "number", step: 21, min: 252, max: 5000 }, f, after);
  }
  function modelBlock(b) {
    const f = M.form, cat = M.catalog, kinds = modelKinds(), usable = kinds.filter((k) => k.available);
    fixModel();
    Q.labField(b, { key: "model", label: "Model", kind: "choice", options: usable.map((k) => [k.kind, `${k.label} (${k.family})`]) }, f, () => Q.renderLab(M));
    const info = cat.models.find((k) => k.kind === f.model);
    if (info) {
      h("p", "hint", b, info.description);
      const own = (f.params[f.model] = f.params[f.model] || Object.fromEntries(Object.entries(info.params).filter(([, v]) => typeof v === "number").map(([k, v]) => [k, String(v)])));
      for (const k of Object.keys(own)) Q.labField(b, { key: k, label: k.replace(/_/g, " "), kind: "number", step: "any" }, own, () => Q.syncLab(M));
    }
    const missing = kinds.filter((k) => !k.available);
    if (missing.length) h("p", "hint", b, "Not available here: " + missing.map((k) => `${k.label} (${k.reason})`).join("; ") + ".");
  }
  function signalBlock(b) {
    const f = M.form, cat = M.catalog, after = () => Q.syncLab(M);
    Q.labField(b, { key: "signal", label: "Turn scores into positions", kind: "choice", options: cat.signals.map((s) => [s.id, SIGNAL_NAME[s.id] || s.id]) }, f, () => Q.renderLab(M));
    const info = cat.signals.find((s) => s.id === f.signal); if (info) h("p", "hint", b, info.text + ".");
    const spec = { n: { label: "n (assets on each side)", step: 1, min: 1, max: 20 }, quantile: { label: "Top quantile (%)", step: 5, min: 5, max: 50 }, threshold: { label: "Threshold (in score units)", step: "any" } };
    for (const k of SIGNAL_FIELDS[f.signal] || []) Q.labField(b, { key: k, kind: "number", ...spec[k] }, f, after);
    Q.labField(b, { key: "gross", label: "Gross exposure", kind: "number", step: 0.1, min: 0.1, max: 3 }, f, after);
    Q.labField(b, { key: "maxWeight", label: "Largest position (% of the book)", kind: "number", step: 1, min: 2, max: 100 }, f, after);
    Q.labField(b, { key: "holding", label: "Holding period (days)", kind: "number", step: 1, min: 1, max: 60 }, f, after);
  }
  function costBlock(b) {
    const f = M.form, after = () => Q.syncLab(M);
    Q.labField(b, { key: "rebalance", label: "Rebalance", kind: "choice", options: [["daily", "Daily"], ["weekly", "Weekly"], ["monthly", "Monthly"]] }, f, after);
    Q.labField(b, { key: "cost", label: "Commission, one way (bps)", kind: "number", step: 0.5, min: 0, max: 250 }, f, after);
    Q.labField(b, { key: "slippage", label: "Slippage, one way (bps)", kind: "number", step: 0.5, min: 0, max: 250 }, f, after);
    Q.labField(b, { key: "targetVol", label: "Volatility target (% a year; blank = none)", kind: "number", step: 1, min: 1, max: 100 }, f, after);
    Q.labField(b, { key: "maxLev", label: "Largest leverage", kind: "number", step: 0.25, min: 0.5, max: 5 }, f, after);
  }
  const newest = (x, y) => String(y.created).localeCompare(String(x.created));
  const NO_MODEL = "This run re-used its parent's predictions and has no model of its own: use the experiment it came from.";
  function expSelect(b, needModel) {                                                        // one saved experiment, shared by Explain, Overfitting and Forward test
    if (!M.board) { h("p", "hint", b, "Loading the experiments…"); ensureBoard(); return false; }
    const rows = M.board.rows.slice().sort(newest), usable = rows.filter((r) => !needModel || r.has_model !== false), f = M.form;
    if (!rows.length) { h("p", "hint", b, "No experiment is saved yet: train one in the Train view first."); return false; }
    if (!usable.length) { h("p", "hint", b, "No saved experiment has a model of its own: train one in the Train view first."); return false; }
    if (!usable.some((r) => r.id === M.cur)) { const now = rows.find((r) => r.id === M.cur), parent = now && usable.find((r) => r.id === now.parent); setCurrent((parent || usable[0]).id); }
    f.exp = M.cur;
    Q.labField(b, { key: "exp", label: "Experiment", kind: "choice", options: usable.map((r) => [r.id, expLabel(r)]) }, f, () => {
      setCurrent(f.exp); f.asset = ""; for (const k of ["explain", "overfit", "forward", "forward_step", "meta", "child"]) { M.results[k] = null; M.errors[k] = null; }
      Q.renderLab(M);
    });
    if (usable.length < rows.length) h("p", "hint", b, "Re-backtests and meta-labelled runs are not listed: they have no model of their own.");
    return true;
  }
  function pickList(b, title, max) {                                                       // several saved experiments, shared by Compare and Combine
    if (!M.board) { h("p", "hint", b, "Loading the experiments…"); ensureBoard(); return; }
    if (!M.board.rows.length) { h("p", "hint", b, "No experiment is saved yet: train two or more in the Train view first."); return; }
    const box = h("div", "checks picklist", b); h("span", "hint", box, title);
    for (const r of M.board.rows.slice().sort(newest)) {
      const label = h("label", "f inline", box), c = h("input", "", label); c.type = "checkbox"; c.checked = M.picked.has(r.id); c.setAttribute("data-value", r.id); label.appendChild(document.createTextNode(expLabel(r)));
      c.addEventListener("change", () => {
        if (c.checked) { if (M.picked.size >= max) { c.checked = false; Q.toast(`At most ${max} at a time`); return; } M.picked.add(r.id); } else M.picked.delete(r.id);
        Q.syncLab(M);
      });
    }
  }
  M.sections = (panel) => {
    const m = M.mode, f = M.form, cat = M.catalog, after = () => Q.syncLab(M);
    const sec = (title, open = true) => { const d = h("details", "", panel); d.open = open; h("summary", "", d, title); return h("div", "body", d); };
    if (!cat) { h("p", "hint", sec("1. Settings"), M.catalogError || "Loading what the page offers…"); return; }
    if (m === "experiments") {
      const b = sec("1. Show");
      Q.labField(b, { key: "sort", label: "Rank by", kind: "choice", options: SORTS }, f, () => M.run(true));
      Q.labField(b, { key: "fModel", label: "Model", kind: "choice", options: [["", "All models"], ...cat.models.map((k) => [k.kind, k.label])] }, f, () => M.run(true));
      Q.labField(b, { key: "fStage", label: "Stage", kind: "choice", options: [["", "Any stage"], ...cat.stages.map((s) => [s, cap(s)])] }, f, () => M.run(true));
      Q.labField(b, { key: "fUniverse", label: "Universe", kind: "choice", options: [["", "All universes"], ...cat.universes.map((u) => [u.id, u.id])] }, f, () => M.run(true));
      h("p", "hint", b, `Experiments, models and datasets are kept under ${cat.registry.root}/ (SQLite for the records, files for the predictions and the final model).`);
    } else if (m === "dataset") {
      Q.labField(sec("1. Name"), { key: "name", label: "Name", kind: "text", placeholder: "dataset" }, f, after);
      universeBlock(sec("2. Universe")); featureBlock(sec("3. Features")); targetBlock(sec("4. Target"));
      prepBlock(sec("5. Preparation and window", false)); segmentBlock(sec("6. Segments", false));
    } else if (m === "train") {
      const a = sec("1. Name");
      Q.labField(a, { key: "name", label: "Name", kind: "text", placeholder: "ridge on ETFs" }, f, after);
      Q.labField(a, { key: "notes", label: "Notes (kept with the experiment)", kind: "text" }, f, after);
      modelBlock(sec("2. Model"));
      const d = sec("3. Data", false); universeBlock(d); featureBlock(d); targetBlock(d);
      prepBlock(sec("4. Preparation and window", false)); segmentBlock(sec("5. Segments and refits", false)); signalBlock(sec("6. Position rule")); costBlock(sec("7. Costs and risk", false));
      const k = sec("8. Checks");
      Q.labField(k, { key: "explain", label: "Measure feature importance (permutation)", kind: "bool" }, f, after);
      Q.labField(k, { key: "robust", label: "Run the stress tests (slower)", kind: "bool", help: "Random start dates, universe subsets, cost multiples, delays, noise and more" }, f, after);
      Q.labField(k, { key: "seed", label: "Random seed", kind: "number", step: 1, min: 0 }, f, after);
    } else if (m === "compare") {
      pickList(sec("1. Experiments"), "Two to eight, saved on the same kind of data:");
    } else if (m === "explain") {
      const b = sec("1. The prediction");
      if (expSelect(b, true)) {
        const last = (M.results.explain || {}).data;
        Q.labField(b, { key: "asset", label: "Asset", kind: last ? "choice" : "text", options: last ? [["", "The one with the largest score"], ...last.assets.map((a) => [a, a])] : undefined, placeholder: "blank = the largest score" }, f, after);
        Q.labField(b, { key: "date", label: "Date (blank = the newest)", kind: "text", placeholder: "2024-06-28" }, f, after);
        Q.labField(b, { key: "perms", label: "Shuffles per feature", kind: "number", step: 10, min: 10, max: 200, help: "More shuffles give steadier Shapley values and take longer" }, f, after);
      }
    } else if (m === "overfit" || m === "forward") {
      const b = sec("1. The experiment");
      if (expSelect(b, m === "forward") && m === "forward") {
        Q.labField(b, { key: "capital", label: "Paper capital ($)", kind: "number", step: 1000, min: 1000 }, f, after);
        Q.labField(b, { key: "asOf", label: "Record as of (blank = the newest date)", kind: "text", placeholder: "2024-06-28" }, f, after);
        h("p", "hint", b, "A signal is a list of paper orders: nothing is sent anywhere. The newest date of the data may be old; the signal is dated by it.");
      }
    } else if (m === "search") {
      const b = sec("1. What to run");
      W.seg(b, [["search", "Search for strategies"], ["tune", "Tune one model"]], f.searchKind, (v) => { f.searchKind = v; Q.renderLab(M); });
      if (f.searchKind === "search") {
        const usable = cat.models.filter((k) => k.available && k.tasks.includes("regression"));
        Q.checkList(b, "Models to search over:", usable.map((k) => [k.kind, k.label]), f.kinds, after);
        for (const s of [{ key: "nRandom", label: "Random trials", min: 0, max: cat.limits.search_random }, { key: "nBayes", label: "Bayesian (TPE) trials", min: 0, max: cat.limits.search_bayes }, { key: "generations", label: "Genetic generations", min: 0, max: cat.limits.search_generations },
          { key: "population", label: "Genetic population", min: 2, max: cat.limits.search_population }, { key: "topK", label: "Finalists run on every segment", min: 1, max: cat.limits.search_top_k }]) Q.labField(b, { kind: "number", step: 1, ...s }, f, after);
        h("p", "hint", b, "The search varies the model, its settings, the horizon, the position rule and the rebalancing. The data and cost settings below stay as they are.");
      } else {
        const tunable = cat.models.filter((k) => k.available && Object.values(k.params).some((v) => typeof v === "number"));
        if (!tunable.some((k) => k.kind === f.model)) f.model = tunable[0].kind;                 // the form shows what it sends
        Q.labField(b, { key: "model", label: "Model to tune", kind: "choice", options: tunable.map((k) => [k.kind, k.label]) }, f, after);
        Q.labField(b, { key: "tuneMethod", label: "Method", kind: "choice", options: [["random", "Random"], ["tpe", "Bayesian (TPE)"]] }, f, after);
        Q.labField(b, { key: "tuneTrials", label: "Trials", kind: "number", step: 1, min: 4, max: cat.limits.tune_trials }, f, after);
      }
      const d = sec("2. Data", false); universeBlock(d); featureBlock(d); targetBlock(d);
      prepBlock(sec("3. Preparation and window", false)); segmentBlock(sec("4. Segments and refits", false)); signalBlock(sec("5. Position rule", false)); costBlock(sec("6. Costs and risk", false));
    } else if (m === "combine") {
      pickList(sec("1. Strategies"), "Two to twelve saved experiments:", 12);
      const b = sec("2. Methods");
      Q.checkList(b, "Ways to weight them:", cat.ensembles.map((e) => [e.id, METHOD_NAME[e.id] || e.id, e.text]), f.methods, after);
      Q.labField(b, { key: "lookback", label: "Estimation window (days)", kind: "number", step: 21, min: 60, max: 1000 }, f, after);
      Q.labField(b, { key: "every", label: "Re-estimate every (days)", kind: "number", step: 1, min: 5, max: 126 }, f, after);
      Q.labField(b, { key: "riskAversion", label: "Risk aversion (mean-variance methods)", kind: "number", step: 0.5, min: 0.5, max: 50 }, f, after);
    }
  };

  /* ---------- running ---------- */
  function datasetBody(f) {
    return { universe: f.universe === "custom" ? tickerList(f.tickers) : f.universe, feature_groups: [...f.groups], target: `${f.kind}_${f.horizon}`, horizon: Number(f.horizon), normalize: f.normalize, winsor: needNumber(f.winsor, "Clipping") / 100,
      min_history: needNumber(f.minHistory, "History needed"), start: f.start.trim(), end: f.end.trim(), regime_detector: f.regime };
  }
  function splitBody(f) {
    return { window: f.window, rolling_days: needNumber(f.rollingDays, "Rolling window"), min_train_days: needNumber(f.minTrain, "Days before the first prediction"), refit_every: optNumber(f.refit, "Refit every"), embargo: needNumber(f.embargo, "Embargo"),
      validation_fraction: needNumber(f.valFrac, "Validation") / 100, test_fraction: needNumber(f.testFrac, "Test") / 100, forward_fraction: needNumber(f.fwdFrac, "Forward") / 100 };
  }
  function signalBody(f) {
    const out = { method: f.signal, gross: needNumber(f.gross, "Gross exposure"), max_weight: needNumber(f.maxWeight, "Largest position") / 100, holding_period: needNumber(f.holding, "Holding period"), net: needNumber(f.net, "Net exposure") };
    for (const k of SIGNAL_FIELDS[f.signal] || []) out[k] = k === "quantile" ? needNumber(f[k], "Quantile") / 100 : needNumber(f[k], k);
    return out;
  }
  function backtestBody(f) {
    const tv = optNumber(f.targetVol, "Volatility target");
    return { rebalance: f.rebalance, cost_bps: needNumber(f.cost, "Commission"), slippage_bps: needNumber(f.slippage, "Slippage"), target_vol: tv === undefined ? "" : tv / 100, max_leverage: needNumber(f.maxLev, "Largest leverage") };
  }
  function modelBody(f) {
    const own = f.params[f.model] || {}, params = {};
    for (const [k, v] of Object.entries(own)) if (String(v).trim() !== "") params[k] = needNumber(v, k.replace(/_/g, " "));
    return { kind: f.model, params };
  }
  const baseBody = (f) => ({ dataset: datasetBody(f), split: splitBody(f), signal: signalBody(f), backtest: backtestBody(f) });
  const experiment = (id) => ({ modes: [], path: "/api/ml/experiment", query: { id }, cacheKey: "exp:" + id });
  const fresh = (mode) => { M.results[mode] = null; M.errors[mode] = null; };

  M.run = async (fromForm) => {
    const m = M.mode, f = M.form;
    if (!M.catalog) { loadCatalog(); return; }
    if (m === "experiments") return W.request(M, () => [{ modes: ["experiments"], path: "/api/ml/experiments", query: { sort: f.sort, limit: 100, model: f.fModel, stage: f.fStage, universe: f.fUniverse } }, { modes: ["models"], path: "/api/ml/models", query: {} }], "Reading the registry…");
    if (m === "dataset") return W.request(M, () => [{ modes: ["dataset"], job: true, path: "/api/ml/dataset/build", body: { name: f.name.trim() || "dataset", dataset: datasetBody(f), split: splitBody(f) } }], "Building the dataset…");
    if (m === "train") {
      const before = M.results.train;
      await W.request(M, () => [{ modes: ["train"], job: true, path: "/api/ml/train", body: { ...baseBody(f), model: modelBody(f), name: f.name.trim() || undefined, notes: f.notes.trim() || undefined, seed: needNumber(f.seed, "Seed"), explain: f.explain, robustness: f.robust } }], "Training and backtesting…");
      if (M.results.train && M.results.train !== before) { setCurrent(M.results.train.data.id); invalidate(); }
      return;
    }
    if (m === "compare") return W.request(M, () => [{ modes: ["compare"], path: "/api/ml/compare", query: { ids: [...M.picked] } }], "Comparing…");
    if (m === "explain") return W.request(M, () => [{ ...experiment(M.cur), modes: ["explain_report"] }, { modes: ["explain"], path: "/api/ml/explain", query: { id: M.cur, asset: f.asset, date: f.date.trim(), permutations: f.perms } }], "Explaining the prediction…");
    if (m === "overfit") return W.request(M, () => [{ ...experiment(M.cur), modes: ["overfit"] }], "Reading the experiment…");
    if (m === "search") {
      const tune = f.searchKind === "tune", before = M.results.search;
      await W.request(M, () => (tune
        ? [{ modes: ["search"], job: true, path: "/api/ml/tune", body: { ...baseBody(f), model: modelBody(f), n_trials: needNumber(f.tuneTrials, "Trials"), method: f.tuneMethod, finalize: true } }]
        : [{ modes: ["search"], job: true, path: "/api/ml/search", body: { ...baseBody(f), kinds: [...f.kinds], n_random: needNumber(f.nRandom, "Random trials"), n_bayes: needNumber(f.nBayes, "Bayesian trials"), generations: needNumber(f.generations, "Generations"),
          population: needNumber(f.population, "Population"), top_k: needNumber(f.topK, "Finalists") } }]), tune ? "Tuning…" : "Searching…");
      if (M.results.search !== before) invalidate();
      return;
    }
    if (m === "combine") return W.request(M, () => [{ modes: ["combine"], path: "/api/ml/combine", body: { ids: [...M.picked], methods: [...f.methods], lookback: needNumber(f.lookback, "Window"), rebalance: needNumber(f.every, "Re-estimate every"), risk_aversion: needNumber(f.riskAversion, "Risk aversion") } }], "Combining…");
    if (m === "forward") {
      const read = { modes: ["forward"], path: "/api/ml/forward", query: { id: M.cur } };
      if (fromForm === true) return W.request(M, () => [read], "Reading the signals…");
      return W.request(M, () => [{ modes: ["forward_step"], job: true, path: "/api/ml/forward/step", body: { id: M.cur, capital: needNumber(f.capital, "Capital"), as_of: f.asOf.trim() || undefined } }, read], "Recording the signal…");
    }
  };
  M.auto = () => {
    const m = M.mode;
    if (!M.catalog) { loadCatalog(); return; }
    if (m === "experiments") { M.run(true); return; }
    if (["compare", "combine", "explain", "overfit", "forward"].includes(m)) {
      if (!M.board) { ensureBoard(); return; }
      if ((m === "overfit" || m === "forward") && M.cur) M.run(true);
      else if (m === "compare" && M.picked.size >= 2) M.run(true);
    }
  };
  const EMPTY = {
    dataset: ["Build a dataset", ["Features at the close of day t, built from data up to t only, and targets that look forward from the next trading day. The result lists every feature with its coverage and its rank IC on the warm-up segment, the only part of the data it may look at."]],
    train: ["Train a model the honest way", ["Walk-forward fits that never see the future, a backtest with costs, and the result split into train, validation, test and forward segments. Every look at the same data is counted, and the deflated Sharpe ratio pays for it."]],
    experiments: ["Experiments", ["Every saved experiment, ranked."]],
    compare: ["Compare experiments", ["Choose two to eight experiments on the left: their settings, their results by segment, their equity curves over the days they share, how correlated they are and a map of how alike they are."]],
    explain: ["Explain a prediction", ["Why did the saved model score an asset as it did on a date? Shapley values split the score exactly into what each feature added. They describe the model, not the market."]],
    overfit: ["Check for overfitting", ["How much of the best result survives where nothing was fitted, and whether the stress tests hold."]],
    search: ["Search for strategies", ["Random trials, then Bayesian, then genetic. Trials see only the warm-up and validation segments; the best few are run on every segment once at the end, and every trial is counted against them."]],
    combine: ["Combine strategies", ["Choose two to twelve experiments on the left: ten ways to weight them, each estimated only from the returns up to the day it is used."]],
    forward: ["Forward test", ["Record the saved model's signal for the newest date, settle the signals whose window has passed, and watch the live rank IC and the drift of each feature against the backtest."]] };
  M.empty = (root) => Q.labEmpty(root, M, EMPTY[M.mode][0], EMPTY[M.mode][1], M.runLabel[M.mode]);
  M.draw = () => Q.drawLab(M);
  M.build = (root, d, body) => ({ dataset: buildDataset, train: buildTrain, experiments: buildExperiments, compare: buildCompare, explain: buildExplain, overfit: buildOverfit, search: buildSearch, combine: buildCombine, forward: buildForward })[M.mode](root, d, body);
  const header = (root, title, sub) => Q.labHead(root, title, sub, GUIDE);

  /* ---------- opening, deleting and re-running saved experiments ---------- */
  async function openExperiment(id) {
    setCurrent(id); M.mode = "train"; Q.renderLab(M);
    await W.request(M, () => [{ ...experiment(id), modes: ["train"] }], "Opening the experiment…");
  }
  const actions = (root, items) => {
    const row = h("div", "row tight", root);
    for (const [text, fn, cls, title, off] of items) { const b = h("button", "btn" + (cls ? " " + cls : ""), row, text); b.type = "button"; if (title) b.title = title; if (off) b.disabled = true; b.addEventListener("click", fn); }
    return row;
  };
  function useFor(id, mode) {
    setCurrent(id); M.form.asset = ""; for (const k of ["explain", "overfit", "forward", "forward_step", "meta", "child"]) { M.results[k] = null; M.errors[k] = null; }
    go(mode); if (mode === "explain") M.run(true);                                          // asked for by name, so run it: the other views load by themselves
  }
  /* A job that makes a new experiment from the one open (a re-backtest, a meta-labelled run). If it fails, say so and leave the open view as it was. */
  async function sideJob(key, path, body, label) {
    const mode = M.mode, keep = M.results[mode]; fresh(key);
    await W.request(M, () => [{ modes: [key], job: true, path, body }], label);
    if (M.errors[mode]) { const message = M.errors[mode]; M.results[mode] = keep; M.errors[mode] = null; Q.toast(message); M.draw(); return null; }
    return M.results[key] ? M.results[key].data : null;
  }

  /* ---------- the leaderboard ---------- */
  function buildExperiments(root, d) {
    const rows = d.rows, models = (M.results.models || {}).data;
    header(root, "Experiments", `${d.n} saved · ranked by ${(SORTS.find((s) => s[0] === d.sort) || [0, d.sort])[1].toLowerCase()}`);
    if (!rows.length) {
      const c = h("div", "card empty", root); h("h2", "", c, d.n || M.form.fModel || M.form.fStage || M.form.fUniverse ? "Nothing matches these filters" : "No experiments yet");
      h("p", "", c, "Train a model, or search for strategies: each run is saved here with its settings, its results by segment and the number of looks at the same data.");
      actions(c, [["Train a model", () => go("train"), "primary"], ["Build a dataset first", () => go("dataset")]]);
      return;
    }
    const best = rows.reduce((a, b) => ((b["validation.sharpe"] ?? -Infinity) > (a["validation.sharpe"] ?? -Infinity) ? b : a)), trials = Object.values(d.dataset_trials || {}).reduce((a, b) => a + b, 0);
    const k = h("div", "earnrow", root);
    tile(k, "Experiments", String(d.n), { text: `${(models || { models: [] }).models.length} registered model${models && models.models.length === 1 ? "" : "s"}` });
    tile(k, "Best validation Sharpe", sig(best["validation.sharpe"]), { text: best.name }, "", "Validation is where choices are made: a high value here is partly selection");
    tile(k, "Test Sharpe of that one", sig(best["test.sharpe"]), { text: `forward ${sig(best["forward.sharpe"])}` }, "", "Test and forward were not used to choose anything");
    tile(k, "Looks at the data", String(trials), { text: "experiments and search trials" }, "", "Every look at the same sample makes the best result look better than it is; the deflated Sharpe ratio corrects for it");
    if (rows.some((r) => r.universe === "futures")) W.flags(root, ["Experiments on the futures universe use simulated prices: they check the machinery, not a market."], "Simulated. ");
    const card = W.card(root, "Leaderboard", d.note);
    const cols = [["", ""], ["Experiment", "Name, then the model, universe, target and position rule"], ["Validation", "Sharpe, net of costs"], ["Test", "Sharpe"], ["Forward", "Sharpe"], ["Rank IC", "Mean rank IC out of sample"],
      ["Generalization", "How much of the best Sharpe survives out of sample, 0 to 1"], ["Looks", "How many looks at the same data had been recorded when it ran"], ["Stage", ""], ["", ""]];
    const wrap = h("div", "scroll tall", card), t = h("table", "data lb", wrap), head = h("tr", "", h("thead", "", t)), body = h("tbody", "", t);
    cols.forEach(([label, title], i) => { const th = h("th", i >= 2 && i <= 7 ? "num" : "", head, label); if (title) th.title = title; });
    const bar = h("div", "row tight", card), sync = () => { cmp.disabled = M.picked.size < 2 || M.picked.size > 8; comb.disabled = M.picked.size < 2 || M.picked.size > 12; note.textContent = M.picked.size ? `${M.picked.size} selected` : "Tick two or more to compare or combine them."; };
    const cmp = h("button", "btn", bar, "Compare the selected"), comb = h("button", "btn", bar, "Combine the selected"), note = h("span", "hint", bar);
    cmp.addEventListener("click", () => { fresh("compare"); go("compare"); }); comb.addEventListener("click", () => { fresh("combine"); go("combine"); });
    for (const r of rows) {
      const tr = h("tr", r.id === M.cur ? "cur" : "", body), c0 = h("td", "", tr), pick = h("input", "", c0); pick.type = "checkbox"; pick.checked = M.picked.has(r.id); pick.setAttribute("aria-label", "Select " + r.name); pick.setAttribute("data-id", r.id);
      pick.addEventListener("change", () => { if (pick.checked) { if (M.picked.size >= 12) { pick.checked = false; Q.toast("At most 12 at a time"); return; } M.picked.add(r.id); } else M.picked.delete(r.id); sync(); });
      const name = h("td", "name", tr); name.title = `${r.id}\n${r.created}`; h("div", "", name, r.name); h("div", "sub", name, [r.model, r.universe, r.target, String(r.signal).replace(/_/g, " "), r.parent ? "child of " + shortId(r.parent) : ""].filter(Boolean).join(" · "));
      for (const key of ["validation.sharpe", "test.sharpe", "forward.sharpe"]) h("td", "num", tr, sig(r[key]));
      h("td", "num", tr, sig(r["ic.mean"], 3)); h("td", "num", tr, sig(r["overfit.generalization_score"])); h("td", "num", tr, r.trials === null || r.trials === undefined ? "n/a" : String(r.trials)); h("td", "", tr, r.stage === "none" ? "" : r.stage);
      const ac = h("td", "", tr), open = h("button", "btn small", ac, "Open"), del = h("button", "btn small", ac, "Delete");
      open.addEventListener("click", () => openExperiment(r.id));
      let armed = null;
      del.addEventListener("click", async () => {
        if (!armed) { del.textContent = "Click again to delete"; armed = setTimeout(() => { armed = null; del.textContent = "Delete"; }, 4000); return; }
        clearTimeout(armed); armed = null;
        try { await api("/api/ml/delete", { id: r.id, confirm: true }); M.picked.delete(r.id); if (M.cur === r.id) M.cur = ""; invalidate(); Q.toast("Experiment deleted"); M.run(true); } catch (e) { Q.toast(e.message); }
      });
    }
    sync();
    const pts = rows.filter((r) => !Q.bad(r["validation.sharpe"]) && !Q.bad(r["test.sharpe"]));
    if (pts.length >= 3) {
      const ch = h("div", "charts", root);
      chartCard(ch, "Does validation predict test?", "One point per experiment. If choosing on validation worked, the points rise from left to right; a cloud with no slope means the validation winners were chosen by luck.", { wide: true,
        draw: (b) => C.scatter(b, { points: pts.map((r) => ({ x: r["validation.sharpe"], y: r["test.sharpe"], label: r.name, color: css("--s1"), tip: [["Validation Sharpe", sig(r["validation.sharpe"])], ["Test Sharpe", sig(r["test.sharpe"])], ["Forward Sharpe", sig(r["forward.sharpe"])], ["Model", r.model]] })), xLabel: "Validation Sharpe", yLabel: "Test Sharpe", xZero: true, yZero: true, height: 300, label: "Validation against test Sharpe" }),
        table: (b) => table(b, [{ label: "Experiment", get: (r) => r.name }, { label: "Validation", num: true, get: (r) => sig(r["validation.sharpe"]) }, { label: "Test", num: true, get: (r) => sig(r["test.sharpe"]) }, { label: "Forward", num: true, get: (r) => sig(r["forward.sharpe"]) }], pts) });
    }
    if (models && models.models.length) {
      const rc = W.card(root, "Model registry", "A saved model moves through stages; at most one production model is kept per universe, and promoting another archives the previous one.");
      const t2 = h("table", "data", h("div", "scroll", rc)), hd = h("tr", "", h("thead", "", t2)), tb = h("tbody", "", t2);
      for (const label of ["Model", "Experiment", "Created", "Features", "Stage"]) h("th", "", hd, label);
      for (const mo of models.models) {
        const tr = h("tr", "", tb); h("td", "", tr, mo.kind); const ex = h("td", "", tr), link = h("button", "link", ex, shortId(mo.experiment_id)); link.addEventListener("click", () => openExperiment(mo.experiment_id));
        h("td", "", tr, mo.created); h("td", "num", tr, String(mo.n_features));
        const sel = h("select", "", h("td", "", tr)); for (const s of models.stages) { const o = h("option", "", sel, s); o.value = s; o.selected = s === mo.stage; }
        sel.setAttribute("aria-label", "Stage of " + mo.id);
        sel.addEventListener("change", async () => { try { await api("/api/ml/promote", { model_id: mo.id, stage: sel.value }); Q.toast(`${mo.kind} is now ${sel.value}`); invalidate(); M.run(true); } catch (e) { Q.toast(e.message); M.run(true); } });
      }
    }
  }

  /* ---------- one experiment: Train, and the pieces Overfitting and Explain reuse ---------- */
  function regions(e, withTrain) {
    const out = [], t = e.train;
    if (withTrain && t && t.dates.length) out.push({ from: t.dates[0], to: t.dates[t.dates.length - 1], name: SEG_NAME.train });
    for (const k of OOS) if (e.bounds && e.bounds[k]) out.push({ from: e.bounds[k][0], to: e.bounds[k][1], name: SEG_NAME[k] });
    return out;
  }
  const benchName = (r) => (r.config.dataset.benchmark === "equal_weight" ? "Equal weight" : r.config.dataset.benchmark);
  function equityChart(b, r) {
    const e = r.equity, t = e.train || { dates: [], strategy: [] }, scale = t.strategy.length ? t.strategy[t.strategy.length - 1] : 1, blank = t.dates.map(() => null), series = [];
    if (t.dates.length) series.push({ name: SEG_NAME.train, color: css("--muted"), values: t.strategy.concat(e.dates.map(() => null)), dash: "4 3", width: 1.5 });
    series.push({ name: "Strategy", color: css("--s1"), values: blank.concat(e.strategy.map((v) => v * scale)) });
    series.push({ name: benchName(r), color: css("--s2"), values: blank.concat(e.benchmark.map((v) => v * scale)), width: 1.5 });
    C.line(b, { dates: t.dates.concat(e.dates), series, regions: regions(e, true), baseline: 1, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 300, label: "Equity curve by segment", endLabels: false });
  }
  function equityTable(b, r) {
    const e = r.equity, every = e.dates.map((_, n) => n).filter((n) => n % 4 === 0);
    table(b, [{ label: "Date", get: (n) => e.dates[n] }, { label: "Segment", get: (n) => e.segment[n] }, { label: "Strategy", num: true, get: (n) => num(e.strategy[n], 4) }, { label: benchName(r), num: true, get: (n) => num(e.benchmark[n], 4) }, { label: "Drawdown", num: true, get: (n) => pct(e.drawdown[n]) }], every);
  }
  const segmentColumns = (b, r) => C.columns(b, { labels: SEGMENTS.map((k) => SEG_NAME[k].split(" ")[0]), values: SEGMENTS.map((k) => (r.overfit.sharpe[k] === undefined ? null : r.overfit.sharpe[k])), yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2),
    colors: (v, i) => (i === 0 ? css("--muted") : css("--s1")), name: "Sharpe", height: 230, label: "Sharpe by segment", tipLabel: (i) => SEG_NAME[SEGMENTS[i]] });
  const segmentTable = (b, r) => table(b, [{ label: "Segment", get: (k) => SEG_NAME[k] }, { label: "Days", num: true, get: (k) => String(r.segments.stats[k] ? r.segments.stats[k].n_days : "") }, { label: "Sharpe", num: true, get: (k) => sig(r.overfit.sharpe[k]) },
    { label: "Return a year", num: true, get: (k) => (r.segments.stats[k] ? spct(r.segments.stats[k].ann_return) : "") }, { label: "Worst fall", num: true, get: (k) => (r.segments.stats[k] ? pct(r.segments.stats[k].max_drawdown) : "") }], SEGMENTS.filter((k) => r.segments.stats[k]));
  function importanceCards(ch, r) {
    const imp = r.importance, bars = imp.bars || [], perm = imp.permutation || [];
    if (bars.length) chartCard(ch, "What the model leans on", `Each feature's share of the model's total absolute contribution, averaged over its refits. Stability of the ranking across refits: ${sig(imp.rank_stability)} (1 is identical). It describes the model, not the market.`, {
      draw: (b) => C.hbars(b, { rows: bars.map((x) => ({ label: x.feature, value: x.share, tip: [["Mean contribution", sct(x.mean)], ["Refits agreeing in sign", pct(x.sign_agreement, 0)], ["Average rank", sig(x.mean_rank, 1)]] })), format: (v) => pct(v, 0), name: "Share", label: "Feature importance", labelWidth: 112, color: () => css("--s1") }),
      table: (b) => table(b, [{ label: "Feature", get: (x) => x.feature }, { label: "Share", num: true, get: (x) => pct(x.share, 1) }, { label: "Mean contribution", num: true, get: (x) => sct(x.mean) }, { label: "Sign agreement", num: true, get: (x) => pct(x.sign_agreement, 0) }, { label: "Average rank", num: true, get: (x) => sig(x.mean_rank, 1) }], imp.table || bars) });
    if (perm.length) chartCard(ch, "What breaks when a feature is shuffled", "The drop in out-of-sample rank IC when one feature's values are shuffled, averaged over refits: how much the predictions need it. A negative bar means shuffling it helped, so the model is leaning on noise.", {
      draw: (b) => C.hbars(b, { rows: perm.map((x) => ({ label: x.feature, value: x.mean_drop, tip: [["Spread across refits", sct(x.std_drop)], ["Refits tested", String(x.n)]] })), format: (v) => v.toFixed(3), name: "Drop in rank IC", label: "Permutation importance", labelWidth: 112 }),
      table: (b) => table(b, [{ label: "Feature", get: (x) => x.feature }, { label: "Drop in rank IC", num: true, get: (x) => num(x.mean_drop, 4) }, { label: "Spread", num: true, get: (x) => num(x.std_drop, 4) }, { label: "Refits", num: true, get: (x) => String(x.n) }], perm) });
  }
  function reportView(root, d, opts) {
    const r = d.report, cfg = r.config, ds = r.dataset, st = r.segments.stats, ov = r.overfit, oos = r.metrics.oos, ic = r.ic, prov = r.provenance, parent = (d.record || {}).parent_id;
    header(root, r.name, `${cfg.model.kind} · ${ds.universe} · ${ds.target} · ${ds.start} to ${ds.end} · ${ds.n_assets} assets, ${ds.n_features} features · ran in ${r.seconds} s${parent ? " · child of " + shortId(parent) : ""}`);
    const hasModel = !d.artifacts || d.artifacts.some((a) => a.name === "model");
    actions(root, [["Overfitting checks", () => useFor(d.id, "overfit")], ["Explain a prediction", () => useFor(d.id, "explain"), "", hasModel ? "" : NO_MODEL, !hasModel], ["Forward test it", () => useFor(d.id, "forward"), "", hasModel ? "" : NO_MODEL, !hasModel],
      ["Download the report", async () => { try { const x = await api("/api/ml/report-html?id=" + encodeURIComponent(d.id)); Q.download(`${d.id}.html`, x.html); } catch (e) { Q.toast(e.message); } }, "", "A self-contained HTML page"]]);
    W.provenance(root, { source: prov.source, sources: {}, as_of: ds.end, warnings: opts && opts.noWarnings ? [] : r.warnings, notes: prov.notes });
    const k = h("div", "earnrow", root);
    for (const key of SEGMENTS) {
      const s = st[key]; if (!s) continue;
      tile(k, SEG_NAME[key], sig(s.sharpe), { text: `${spct(s.ann_return)} a year · worst fall ${pct(s.max_drawdown, 0)} · ${s.n_days} days` }, key === "train" ? "muted" : "",
        key === "train" ? "In-sample: the model was fitted on these days, so this is not evidence" : key === "validation" ? "Where settings were chosen: partly selection" : "Reserved: no choice has looked at it");
    }
    const k2 = h("div", "earnrow", root);
    tile(k2, "Rank IC", sig(ic.all.mean, 3), { text: `IR ${sig(ic.all.ir)} · t ${sig(ic.all.t)} · positive ${pct(ic.all.hit_rate, 0)} of days` }, "", "Mean cross-sectional rank correlation of the score with the realised forward return, out of sample");
    tile(k2, "Generalization", sig(ov.generalization_score), { text: "of the best Sharpe survives", dir: ov.generalization_score >= 0.5 ? "up" : "down" }, "", "The worst out-of-sample Sharpe over the larger of the train and validation Sharpe, clipped to 0 to 1");
    tile(k2, "Deflated Sharpe", sig(ov.deflated_sharpe_probability), { text: `${ov.n_trials} look${ov.n_trials === 1 ? "" : "s"} at this data; 0.95 is the bar`, dir: ov.deflated_sharpe_probability >= 0.95 ? "up" : "" }, "", "Probability the validation Sharpe beats what the best of this many tries would show by luck");
    tile(k2, "Test and forward", sig(ov.psr_test_forward), { text: "probability the true Sharpe is above 0" }, "", "Probabilistic Sharpe ratio over the two segments no choice has seen");
    tile(k2, "Turnover a year", sig(oos.ann_turnover, 1) + "×", { text: `cost drag ${pct(oos.ann_cost_drag, 1)} a year` });
    tile(k2, "Worst fall", pct(oos.max_drawdown), { text: `out of sample · Calmar ${sig(oos.calmar)}` });
    const ch = h("div", "charts", root);
    chartCard(ch, "Equity curve by segment", "Growth of 1 from the first training day. The dashed line is in-sample, so it flatters; the three segments after it are out of sample. The benchmark is rebased to the strategy where validation starts.", { wide: true, draw: (b) => equityChart(b, r), table: (b) => equityTable(b, r) });
    chartCard(ch, "Sharpe by segment", "Net of costs. The grey bar is in-sample; a model that is real keeps a similar height from validation to forward.", { draw: (b) => segmentColumns(b, r), table: (b) => segmentTable(b, r) });
    chartCard(ch, "Drawdown from the previous peak", "Out-of-sample days only.", {
      draw: (b) => C.line(b, { dates: r.equity.dates, series: [{ name: "Drawdown", color: css("--s1"), values: r.equity.drawdown }], area: true, areaBase: 0, baseline: 0, ceil: 0, regions: regions(r.equity, false), yFormat: pctTick, tipFormat: (v) => pct(v), height: 230, label: "Drawdown" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => r.equity.dates[n] }, { label: "Drawdown", num: true, get: (n) => pct(r.equity.drawdown[n]) }], r.equity.dates.map((_, n) => n).filter((n) => n % 4 === 0)) });
    if (ic.series && ic.series.dates.length) chartCard(ch, "Rank IC through time", `The ${ic.series.window}-day average of each day's rank correlation between the score and what happened next. A real signal sits above zero in every section, not only in one.`, {
      draw: (b) => C.line(b, { dates: ic.series.dates, series: [{ name: "Rank IC", color: css("--s1"), values: ic.series.rolling_ic }], baseline: 0, include: 0, regions: regions(r.equity, false), yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 230, label: "Rank IC through time" }),
      table: (b) => table(b, [{ label: "Date", get: (n) => ic.series.dates[n] }, { label: `Rank IC, ${ic.series.window}-day`, num: true, get: (n) => (Q.bad(ic.series.rolling_ic[n]) ? "" : num(ic.series.rolling_ic[n], 3)) }, { label: "Cumulative", num: true, get: (n) => num(ic.series.cumulative[n], 2) }], ic.series.dates.map((_, n) => n).filter((n) => n % 5 === 0)) });
    if (ic.decay && ic.decay.length) chartCard(ch, "Rank IC by horizon", "How fast the signal fades: the same scores against returns 1, 2, 3 … days after the trade. A signal that lasts is easier to trade than one that is gone in a day.", {
      draw: (b) => C.columns(b, { labels: ic.decay.map((x) => x.horizon + "d"), values: ic.decay.map((x) => x.mean_ic), yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), name: "Mean rank IC", height: 230, label: "Rank IC by horizon", tipLabel: (i) => `${ic.decay[i].horizon} days after the trade (t = ${sig(ic.decay[i].t)})` }),
      table: (b) => table(b, [{ label: "Horizon", get: (x) => x.horizon + " days" }, { label: "Mean rank IC", num: true, get: (x) => num(x.mean_ic, 4) }, { label: "t-statistic", num: true, get: (x) => num(x.t, 2) }, { label: "Days", num: true, get: (x) => String(x.n) }], ic.decay) });
    const q = ic.quantiles;
    if (q && q.buckets && q.buckets.length) chartCard(ch, "Return by score bucket", `The average forward return of the assets in each fifth of the day's scores, lowest to highest. The top minus the bottom is ${spct(q.spread, 2)}; how steadily the bars rise (monotonicity ${sig(q.monotonicity)}, 1 is perfect) matters as much as the gap.`, {
      draw: (b) => C.columns(b, { labels: q.buckets.map((x) => "Q" + x.bucket), values: q.buckets.map((x) => x.mean_return), yFormat: pctTick, tipFormat: (v) => spct(v, 2), name: "Average forward return", height: 230, label: "Return by score bucket", tipLabel: (i) => `Score bucket ${q.buckets[i].bucket} of ${q.buckets.length}` }),
      table: (b) => table(b, [{ label: "Bucket", get: (x) => String(x.bucket) }, { label: "Average forward return", num: true, get: (x) => spct(x.mean_return, 3) }, { label: "Observations", num: true, get: (x) => String(x.n) }], q.buckets) });
    const dist = r.distribution, parts = OOS.filter((s) => dist[s] && dist[s].counts && dist[s].counts.length);
    if (parts.length) chartCard(ch, "What the model predicts", "The shape of the scores in each out-of-sample segment, as the share of predictions per bin. If the shapes differ a lot, the model sees a different world than it trained on.", {
      draw: (b) => C.histogram(b, { series: parts.map((s, i) => ({ name: SEG_NAME[s], edges: dist[s].edges, counts: dist[s].counts, color: css(["--s1", "--s2", "--s3"][i]) })), markers: [{ x: 0, label: "0" }], xFormat: sct, height: 240, label: "Prediction distribution" }),
      table: (b) => table(b, [{ label: "Segment", get: (s) => SEG_NAME[s] }, { label: "Predictions", num: true, get: (s) => String(dist[s].n) }, { label: "Mean", num: true, get: (s) => sct(dist[s].mean) }, { label: "Standard deviation", num: true, get: (s) => sct(dist[s].std) }], parts) });
    importanceCards(ch, r);
    const card = h("div", "card", root); h("h3", "", card, "What ran");
    const plan = r.segments.plan, own = cfg.model.params || {};
    W.kv(card, "Settings", [["Model", cfg.model.kind + (Object.keys(own).length ? " " + Object.entries(own).map(([a, v]) => `${a}=${v}`).join(", ") : "")], ["Target", ds.target], ["Features", `${ds.n_features} (${(r.features.names || []).length ? r.features.names.slice(0, 6).join(", ") + (r.features.names.length > 6 ? ", …" : "") : ""})`],
      ["Position rule", SIGNAL_NAME[cfg.signal.method] || cfg.signal.method], ["Rebalance", cfg.backtest.rebalance], ["Costs, one way", `${num(cfg.backtest.cost_bps, 1)} bps commission + ${num(cfg.backtest.slippage_bps, 1)} bps slippage`],
      ["Volatility target", cfg.backtest.target_vol ? pct(cfg.backtest.target_vol, 0) : "none"], ["Refits", `${r.folds.n} walk-forward fits, every ${cfg.split.refit_every} days${r.folds.n_skipped ? `, ${r.folds.n_skipped} skipped` : ""}`], ["Training window", cfg.split.window + (cfg.split.window === "rolling" ? ` (${cfg.split.rolling_days} days)` : "")],
      ["Seed", String(cfg.seed)], ["Experiment", d.id]]);
    const segRows = SEGMENTS.filter((k) => plan[k]);
    W.kv(card, "Segments", segRows.map((k) => [SEG_NAME[k], `${plan[k].start} to ${plan[k].end} (${plan[k].n_days} days)`]));
    const skipped = Object.entries(r.features.skipped || {});
    if (skipped.length) { const dt = h("details", "how", card); h("summary", "", dt, `${skipped.length} feature${skipped.length === 1 ? "" : "s"} could not be built from this data`); table(dt, [{ label: "Feature", get: (x) => x[0] }, { label: "Why", get: (x) => x[1] }], skipped); }
    if (r.trials) W.hint(card, r.trials.note);
  }
  function buildTrain(root, d) {
    reportView(root, d);
    const r = d.report, box = h("div", "card", root); h("h3", "", box, "Try other trading rules on the same predictions");
    h("p", "hint", box, "A re-backtest re-uses the saved predictions: nothing is refitted, so it is fast, and the result is saved as a child experiment. It counts as another look at the data.");
    const st = { signal: r.config.signal.method, n: String(r.config.signal.n), rebalance: r.config.backtest.rebalance, cost: String(r.config.backtest.cost_bps), slippage: String(r.config.backtest.slippage_bps) }, grid = h("div", "params", box);
    Q.labField(grid, { key: "signal", label: "Position rule", kind: "choice", options: M.catalog.signals.map((s) => [s.id, SIGNAL_NAME[s.id] || s.id]) }, st);
    Q.labField(grid, { key: "n", label: "n (top and bottom)", kind: "number", step: 1, min: 1, max: 20 }, st);
    Q.labField(grid, { key: "rebalance", label: "Rebalance", kind: "choice", options: [["daily", "Daily"], ["weekly", "Weekly"], ["monthly", "Monthly"]] }, st);
    Q.labField(grid, { key: "cost", label: "Commission (bps)", kind: "number", step: 0.5, min: 0, max: 250 }, st);
    Q.labField(grid, { key: "slippage", label: "Slippage (bps)", kind: "number", step: 0.5, min: 0, max: 250 }, st);
    const go2 = h("button", "btn primary", box, "Re-backtest");
    go2.addEventListener("click", async () => {
      let body; try { body = { id: d.id, signal: { method: st.signal, n: needNumber(st.n, "n") }, backtest: { rebalance: st.rebalance, cost_bps: needNumber(st.cost, "Commission"), slippage_bps: needNumber(st.slippage, "Slippage") } }; } catch (e) { Q.toast(e.message); return; }
      const got = await sideJob("child", "/api/ml/rebacktest", body, "Re-backtesting…");
      if (got) { invalidate(); setCurrent(got.id); M.results.train = { data: got, body: {} }; M.draw(); }
    });
  }

  /* ---------- dataset ---------- */
  function buildDataset(root, d) {
    const ds = d.dataset, tm = d.timing, plan = d.plan || {};
    header(root, `Dataset: ${ds.name}`, `${ds.n_rows.toLocaleString("en-US")} rows · ${ds.n_dates.toLocaleString("en-US")} dates · ${ds.n_assets} assets · ${ds.n_features} features · ${ds.start} to ${ds.end}`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Features", String(ds.n_features), { text: Object.entries(d.groups).map(([g, n]) => `${g} ${n}`).join(" · ") });
    tile(k, "Target", ds.target, { text: `${d.target.n.toLocaleString("en-US")} labelled rows` }, "", "What the model is asked to predict");
    tile(k, "Purge", `${tm.purge} days`, { text: `lag ${tm.lag} + horizon ${tm.horizon}` }, "", "Training rows whose label window overlaps a test day are dropped, so nothing the label saw is in the training set");
    tile(k, "Missing", pct(ds.missing_share, 2), { text: "of feature values" });
    tile(k, "Positive targets", pct(d.target.share_positive, 0), { text: `mean ${spct(d.target.mean, 2)} · std ${pct(d.target.std, 2)}` });
    const ch = h("div", "charts", root), feats = d.features.filter((x) => !Q.bad(x.ic)), top = feats.slice().sort((a, b) => Math.abs(b.ic) - Math.abs(a.ic)).slice(0, 20);
    if (top.length) chartCard(ch, "Which features carry information", `Mean daily rank IC with the target on the ${d.target.scope} only: it is the only part of the data a feature may be judged on without leaking later segments into the choice. Blue is a positive relationship, red a negative one.`, { wide: true,
      draw: (b) => C.hbars(b, { rows: top.map((x) => ({ label: x.name, value: x.ic, tip: [["Information ratio", sig(x.ic_ir)], ["t-statistic", sig(x.ic_t)], ["Group", x.group], ["Coverage", pct(x.coverage, 0)]] })), format: (v) => v.toFixed(3), name: "Rank IC", label: "Feature rank IC", labelWidth: 120 }),
      table: (b) => featureTable(b, d.features) });
    const card = h("div", "card", root); h("h3", "", card, "Every feature");
    let group = "";
    const bar = W.pills(card, [["", "All"], ...Object.keys(d.groups).map((g) => [g, `${g} (${d.groups[g]})`])], group, (v) => { group = v; for (const x of bar.querySelectorAll("button")) x.setAttribute("aria-selected", String(x.dataset.value === v)); paint(); });
    const slot = h("div", "", card), paint = () => { slot.replaceChildren(); featureTable(slot, group ? d.features.filter((x) => x.group === group) : d.features); };
    paint();
    if (plan.error) W.flags(root, [plan.error], "Segments. "); else {
      const c2 = W.card(root, "How the days are split", "The first prediction comes after the warm-up. Each later refit trains on what came before, minus the purge, and predicts the next block.");
      table(c2, [{ label: "Segment", get: (s) => SEG_NAME[s] }, { label: "From", get: (s) => plan[s].start }, { label: "To", get: (s) => plan[s].end }, { label: "Days", num: true, get: (s) => String(plan[s].n_days) }], SEGMENTS.filter((s) => plan[s]));
    }
    const skipped = Object.entries(d.skipped || {});
    if (skipped.length) { const c3 = W.card(root, "Features that could not be built", "The data does not have what they need; nothing was filled in."); table(c3, [{ label: "Feature", get: (x) => x[0] }, { label: "Why", get: (x) => x[1] }], skipped); }
    const c4 = W.card(root, "Assets"); table(c4, [{ label: "Asset", get: (a) => a.asset }, { label: "Rows", num: true, get: (a) => a.rows.toLocaleString("en-US") }, { label: "First row", get: (a) => a.first || "" }], d.assets);
    actions(root, [["Train a model on it", () => go("train"), "primary"]]);
  }
  const featureTable = (b, rows) => table(b, [{ label: "Feature", get: (x) => x.name }, { label: "Group", get: (x) => x.group }, { label: "Coverage", num: true, get: (x) => pct(x.coverage, 0) }, { label: "Rank IC", num: true, get: (x) => (Q.bad(x.ic) ? "n/a" : num(x.ic, 3)) },
    { label: "IC information ratio", num: true, get: (x) => sig(x.ic_ir) }, { label: "t", num: true, get: (x) => sig(x.ic_t) }, { label: "What it is", get: (x) => x.text + (x.broadcast ? " (the same for every asset)" : "") }], rows);

  /* ---------- compare ---------- */
  function uniqueNames(rows) {
    const count = {}; for (const r of rows) count[r.name] = (count[r.name] || 0) + 1;
    return Object.fromEntries(rows.map((r) => [r.id, (count[r.name] > 1 ? `${r.name} (${shortId(r.id)})` : r.name)]));
  }
  function runs(segment) {                                                                 // contiguous stretches of one segment label, as regions of the date axis
    const out = []; let start = 0;
    segment.forEach((s, i) => { if (i === segment.length - 1 || segment[i + 1] !== s) { out.push({ name: SEG_NAME[s] || s, first: start, last: i }); start = i + 1; } });
    return out;
  }
  function buildCompare(root, d) {
    const names = uniqueNames(d.rows), n = d.rows.length;
    header(root, `Comparing ${n} experiments`, `${d.common_days} shared days · ${d.start} to ${d.end} · the out-of-sample days only`);
    W.provenance(root, d);
    const best = d.rows.reduce((a, b) => (b.segments.validation.sharpe > a.segments.validation.sharpe ? b : a)), k = h("div", "earnrow", root);
    tile(k, "Experiments", String(n), { text: `${d.common_days} days in common` });
    tile(k, "Best on validation", sig(best.segments.validation.sharpe), { text: names[best.id] });
    tile(k, "Its test Sharpe", sig(best.segments.test.sharpe), { text: `forward ${sig(best.segments.forward.sharpe)}` });
    const pairs = []; d.correlation.matrix.forEach((row, i) => row.forEach((v, j) => { if (j > i) pairs.push(v); }));
    if (pairs.length) tile(k, "Average correlation", sig(pairs.reduce((a, b) => a + b, 0) / pairs.length), { text: "of daily returns, between the strategies" }, "", "Strategies that move together add little to one another");
    const ch = h("div", "charts", root), ids = d.rows.map((r) => r.id), reg = runs(d.segment).filter((x) => x.name !== "Train (in-sample)").map((x) => ({ from: d.dates[x.first], to: d.dates[x.last], name: x.name }));
    chartCard(ch, "Growth of 1, out of sample", "Each strategy's equity curve over the days they share, from the start of validation. The shaded sections are the segments.", { wide: true,
      draw: (b) => C.line(b, { dates: d.dates, series: ids.map((id) => ({ name: names[id], color: expColor(id), values: d.curves[id] })), regions: reg, baseline: 1, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 320, label: "Equity curves of the compared experiments", endLabels: false }),
      table: (b) => table(b, [{ label: "Date", get: (i) => d.dates[i] }, { label: "Segment", get: (i) => d.segment[i] }, ...ids.map((id) => ({ label: names[id], num: true, get: (i) => num(d.curves[id][i], 4) }))], d.dates.map((_, i) => i).filter((i) => i % 4 === 0)) });
    const card = W.card(root, "Settings and results", "Sharpe ratios are net of costs. The in-sample column is the model grading its own homework.");
    const col = (label, get, title) => ({ label, num: true, title, get });
    table(card, [{ label: "Experiment", get: (r) => names[r.id] }, { label: "Settings", get: (r) => [r.model, r.target, String(r.signal).replace(/_/g, " "), r.rebalance].join(" · ") },
      col("Train", (r) => sig(r.segments.train && r.segments.train.sharpe), "Sharpe, in-sample"), col("Validation", (r) => sig(r.segments.validation.sharpe)), col("Test", (r) => sig(r.segments.test.sharpe)), col("Forward", (r) => sig(r.segments.forward.sharpe)),
      col("Rank IC", (r) => sig(r.ic, 3)), col("Generalization", (r) => sig(r.generalization)), col("Turnover", (r) => sig(r.turnover, 1) + "×"), col("Looks", (r) => String(r.trials))], d.rows);
    const warned = d.rows.filter((r) => (r.warnings || []).length);
    if (warned.length) { const w = W.card(root, "What each experiment's checks said"); for (const r of warned) { W.sub(w, names[r.id]); const ul = h("ul", "plain", w); for (const t of r.warnings) h("li", "", ul, t); } }
    const ch2 = h("div", "charts", root);
    if (n >= 3) chartCard(ch2, "Correlation of daily returns", "Blue is moving together, red against. Three strategies that all correlate near 1 are one strategy.", {
      draw: (b) => C.matrix(b, { rows: ids.map((id) => names[id]), cols: ids.map((id) => names[id]), values: d.correlation.matrix, diverging: true, mid: 0, min: -1, max: 1, format: (v) => v.toFixed(2), name: "Correlation", label: "Correlation matrix", cellHeight: 32 }),
      table: (b) => table(b, [{ label: "", get: (id) => names[id] }, ...ids.map((id2, j) => ({ label: names[id2], num: true, get: (id) => num(d.correlation.matrix[ids.indexOf(id)][j], 2) }))], ids) });
    if (n >= 3 && d.similarity && d.similarity.explained > 0) chartCard(ch2, "Strategy similarity map", `Strategies placed so that the ones whose returns move together sit close (classical multidimensional scaling of the correlations; it keeps ${pct(d.similarity.explained, 0)} of the structure in two dimensions).`, {
      draw: (b) => C.scatter(b, { points: d.similarity.names.map((id, i) => ({ x: d.similarity.x[i], y: d.similarity.y[i], label: names[id] || id, color: expColor(id) })), labels: "all", height: 300, label: "Strategy similarity map", xFormat: (v) => v.toFixed(1), yFormat: (v) => v.toFixed(1) }),
      table: (b) => table(b, [{ label: "Strategy", get: (i) => names[d.similarity.names[i]] || d.similarity.names[i] }, { label: "x", num: true, get: (i) => num(d.similarity.x[i], 3) }, { label: "y", num: true, get: (i) => num(d.similarity.y[i], 3) }], d.similarity.names.map((_, i) => i)) });
    actions(root, [["Combine these", () => { fresh("combine"); go("combine"); }]]);
  }

  /* ---------- explain ---------- */
  function buildExplain(root, d) {
    const ex = d.explanation, steps = ex.steps, rep = (M.results.explain_report || {}).data;
    header(root, `Why the model scored ${d.asset} as it did`, `${d.model} on ${d.target} · ${d.date} · base ${sct(ex.base)} → score ${sct(ex.prediction)}`);
    W.provenance(root, d);
    const rank = d.scores.findIndex((s) => s.asset === d.asset) + 1, k = h("div", "earnrow", root);
    tile(k, "Score", sct(ex.prediction), { text: `rank ${rank} of ${d.scores.length} on ${d.date}` }, "", "The model's prediction for this asset on this date");
    tile(k, "Average score", sct(ex.base), { text: "what it predicts knowing nothing" }, "", "The Shapley base value: the expected prediction over a background sample");
    const top = steps.filter((s) => s.value !== null).reduce((a, b) => (Math.abs(b.contribution) > Math.abs(a.contribution) ? b : a), steps[0]);
    tile(k, "Biggest driver", top.feature, { text: `${ssct(top.contribution)} · value ${top.value === null ? "n/a" : num(top.value, 2)}`, dir: top.contribution >= 0 ? "up" : "down" });
    const ch = h("div", "charts", root), rows = [{ label: "average score", value: ex.base, text: sct(ex.base), kind: "total" }, ...steps.map((s) => ({ label: s.value === null ? s.feature : `${s.feature} = ${num(s.value, 2)}`, from: s.start, value: s.end, text: ssct(s.contribution), tip: [["Contribution", ssct(s.contribution)]] })), { label: "this score", value: ex.prediction, text: sct(ex.prediction), kind: "total" }];
    chartCard(ch, "From the average score to this score", "Each bar is what one feature added (blue) or took away (red), largest first; they sum exactly to the difference between the two grey bars. A feature's value is its z-score across the assets that day.", { wide: true,
      draw: (b) => C.hbars(b, { rows, format: sct, name: "Running score", label: "Shapley waterfall", labelWidth: 170, valueWidth: 70, color: (v, r) => (r.kind === "total" ? css("--ink-2") : v >= (r.from || 0) ? css("--pos") : css("--neg")) }),
      table: (b) => table(b, [{ label: "Feature", get: (s) => s.feature }, { label: "Value", num: true, get: (s) => (s.value === null ? "" : num(s.value, 3)) }, { label: "Contribution", num: true, get: (s) => ssct(s.contribution) }, { label: "Running score", num: true, get: (s) => sct(s.end) }], steps) });
    chartCard(ch, `Every asset's score on ${d.date}`, "The same model on every asset that day: positive scores are the buys, negative the sells. The asset explained above is marked.", { wide: true,
      draw: (b) => C.hbars(b, { rows: d.scores.map((s) => ({ label: s.asset === d.asset ? s.asset + " ◂" : s.asset, value: s.score })), format: sct, name: "Score", label: "Scores by asset", labelWidth: 80 }),
      table: (b) => table(b, [{ label: "Asset", get: (s) => s.asset }, { label: "Score", num: true, get: (s) => sct(s.score) }], d.scores) });
    if (rep) importanceCards(ch, rep.report);
    W.hint(root, "Shapley values describe the model, not the market: a feature that drives a score has not been shown to drive returns. Evidence that it does is the out-of-sample record on the Train and Overfitting views.");
  }

  /* ---------- overfitting ---------- */
  const VERDICT = { pass: "✓ pass", warn: "! warn", fail: "✕ fail" };
  function robustnessCard(root, r, id) {
    const rob = r.robustness, card = W.card(root, "Stress tests", "The same predictions under pressure: other start dates, fewer assets, higher costs, later trades and noisier scores. A strategy that is real survives most of them.");
    if (!rob) {
      h("p", "hint", card, "The stress tests have not been run for this experiment. They re-use its saved predictions and take under a minute; the result is saved as a child experiment.");
      const b = h("button", "btn primary", card, "Run the stress tests");
      b.addEventListener("click", async () => { const got = await sideJob("child", "/api/ml/rebacktest", { id, robustness: true }, "Running the stress tests…"); if (got) { invalidate(); setCurrent(got.id); fresh("overfit"); Q.renderLab(M); } });
      return;
    }
    const k = h("div", "earnrow", card);
    tile(k, "Passed", String(rob.n_pass), { text: `of ${rob.n_scored} scored` }); tile(k, "Warned", String(rob.n_warn), {}); tile(k, "Failed", String(rob.n_fail), { dir: rob.n_fail ? "down" : "" }); tile(k, "Score", sig(rob.score), { text: "share of tests passed, warnings count half" });
    const wrap = h("div", "scroll tall", card), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t)), tb = h("tbody", "", t);
    for (const label of ["Test", "Verdict", "What it found"]) h("th", "", hd, label);
    for (const x of rob.tests) {
      const tr = h("tr", "", tb), name = h("td", "", tr, x.title); name.title = x.description;
      h("td", "verdict " + x.verdict, tr, VERDICT[x.verdict] || x.verdict); const td = h("td", "", tr, x.detail);
      if (x.rows && x.rows.length) {
        const dt = h("details", "how", td); h("summary", "", dt, `${x.rows.length} scenarios`);
        table(dt, [{ label: "Scenario", get: (s) => s.scenario }, { label: "Sharpe", num: true, get: (s) => sig(s.sharpe) }, { label: "Return a year", num: true, get: (s) => spct(s.ann_return) }, { label: "Worst fall", num: true, get: (s) => pct(s.max_drawdown) }], x.rows);
      }
    }
  }
  function metaCard(root, d, id) {
    const card = W.card(root, "Meta-labelling", "A second, simpler model learns which of the first model's bets tend to pay and sizes each position by its confidence; it never flips a side. It is trained only on the first model's out-of-sample bets, so it can only help where the first model has some edge.");
    const got = M.results.meta && M.results.meta.data && M.results.meta.data.parent === id ? M.results.meta.data : null;
    if (!got) {
      const b = h("button", "btn", card, "Meta-label this experiment");
      b.addEventListener("click", async () => { if (await sideJob("meta", "/api/ml/meta-label", { id }, "Meta-labelling…")) { invalidate(); M.draw(); } });
      return;
    }
    const m = got.meta, k = h("div", "earnrow", card);
    tile(k, "Classifier AUC", sig(m.auc, 2), { text: "0.5 is a coin toss" }); tile(k, "Bets sized", pct(m.coverage, 0), { text: `${m.n_models} models` });
    tile(k, "First model hit rate", pct(m.primary_hit_rate, 1), { text: "bets that paid" }); tile(k, "After meta-labelling", pct(m.meta_hit_rate, 1), { text: "bets that paid" });
    table(card, [{ label: "Segment", get: (s) => SEG_NAME[s] }, { label: "Sharpe before", num: true, get: (s) => sig((got.parent_segments[s] || {}).sharpe) }, { label: "Sharpe after", num: true, get: (s) => sig((got.report.segments.stats[s] || {}).sharpe) }], OOS);
    actions(card, [["Open the meta-labelled experiment", () => openExperiment(got.id)]]);
  }
  function buildOverfit(root, d) {
    const r = d.report, ov = r.overfit, id = d.id, parent = (d.record || {}).parent_id;
    header(root, `${r.name}: how much to trust it`, `${r.config.model.kind} on ${r.dataset.universe} · ${ov.n_trials} look${ov.n_trials === 1 ? "" : "s"} at this data${parent ? " · child of " + shortId(parent) : ""}`);
    W.provenance(root, { source: r.provenance.source, sources: {}, as_of: r.dataset.end, warnings: [], notes: r.provenance.notes });
    W.flags(root, ov.warnings, "Check. ");
    const k = h("div", "earnrow", root), g = ov.gaps || {};
    tile(k, "Generalization", sig(ov.generalization_score), { text: "of the best Sharpe survives", dir: ov.generalization_score >= 0.5 ? "up" : "down" }, "", "The worst out-of-sample Sharpe over the larger of the train and validation Sharpe, clipped to 0 to 1; 0 if any out-of-sample Sharpe is not positive");
    tile(k, "Deflated Sharpe", sig(ov.deflated_sharpe_probability), { text: "0.95 is the bar", dir: ov.deflated_sharpe_probability >= 0.95 ? "up" : "" }, "", "The validation Sharpe, deflated by the number of looks at this data");
    tile(k, "Test and forward", sig(ov.psr_test_forward), { text: "probability the true Sharpe is above 0" });
    tile(k, "Looks at the data", String(ov.n_trials), { text: "experiments and search trials" });
    tile(k, "Train to validation", signedGap(g.train_to_validation), { text: "Sharpe lost out of sample" }, "", "A large positive gap is the signature of a model that has memorised its training data");
    tile(k, "Instability", sig(ov.instability), { text: "spread of Sharpe across validation blocks" });
    const ch = h("div", "charts", root);
    chartCard(ch, "Sharpe by segment", "The grey bar is in-sample. Out-of-sample bars far below it mean the training result was a mirage.", { draw: (b) => segmentColumns(b, r), table: (b) => segmentTable(b, r) });
    if (ov.validation_blocks && ov.validation_blocks.length) chartCard(ch, "Sharpe through the validation segment", "The validation days cut into consecutive blocks. A result that comes from one block only is a regime, not an edge.", {
      draw: (b) => C.columns(b, { labels: ov.validation_blocks.map((_, i) => "Block " + (i + 1)), values: ov.validation_blocks, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), name: "Sharpe", height: 230, label: "Sharpe by validation block" }),
      table: (b) => table(b, [{ label: "Block", get: (i) => "Block " + (i + 1) }, { label: "Sharpe", num: true, get: (i) => sig(ov.validation_blocks[i]) }], ov.validation_blocks.map((_, i) => i)) });
    if (r.ic.by_fold && r.ic.by_fold.length) chartCard(ch, "Rank IC by refit", "The mean rank IC in the window each walk-forward fit predicted. A signal that is real shows up in most of them, not in two.", { wide: true,
      draw: (b) => C.columns(b, { labels: r.ic.by_fold.map((_, i) => String(i + 1)), values: r.ic.by_fold.map((v) => (Q.bad(v) ? null : v)), yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), name: "Mean rank IC", height: 230, label: "Rank IC by refit",
        tipLabel: (i) => { const f = (r.folds.list.filter((x) => !x.skipped) || [])[i]; return f ? `Refit ${i + 1}: ${f.test_start} to ${f.test_end}` : `Refit ${i + 1}`; } }),
      table: (b) => table(b, [{ label: "Refit", get: (i) => String(i + 1) }, { label: "Window", get: (i) => { const f = r.folds.list.filter((x) => !x.skipped)[i]; return f ? `${f.test_start} to ${f.test_end}` : ""; } }, { label: "Mean rank IC", num: true, get: (i) => sig(r.ic.by_fold[i], 3) }], r.ic.by_fold.map((_, i) => i)) });
    robustnessCard(root, r, id); metaCard(root, d, id);
    const t = W.card(root, "Counting the looks"); W.hint(t, r.trials.note);
    W.kv(t, "Looks at this sample", [["Experiments and search trials on this dataset", String(r.trials.dataset_trials)], ["Counted against this result", String(ov.n_trials)]]);
    const kids = d.children || [];
    if (kids.length) { W.sub(t, "Children of this experiment"); const row = h("div", "chips", t); for (const c2 of kids) { const b = h("button", "btn small", row, `${c2.name} (${shortId(c2.id)})`); b.addEventListener("click", () => openExperiment(c2.id)); } }
  }
  const signedGap = (v) => (Q.bad(v) ? "n/a" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2));

  /* ---------- search and tune ---------- */
  const family = (stage) => String(stage).replace(/-\d+$/, "");
  function paramText(p) { return Object.entries(p).map(([k, v]) => `${k.split(".").pop()}=${typeof v === "number" ? +v.toPrecision(3) : v}`).join(", "); }
  function buildSearch(root, d) {
    const tune = !d.finalists, sel = d.selection || {}, trials = d.trials || [], best = d.best;
    header(root, tune ? "Tuning one model" : "Strategy search", `${d.n_trials} trials${d.n_failed ? ` (${d.n_failed} failed)` : ""}${d.seconds ? ` · ${num(d.seconds, 1)} s` : ""}${d.stages ? " · " + Object.entries(d.stages).map(([s, n]) => `${n} ${s}`).join(", ") : ""}`);
    W.flags(root, [d.note || "Scores are validation-only: the test and forward segments were not predicted during the search."], "Read first. ");
    const k = h("div", "earnrow", root);
    tile(k, "Trials", String(d.n_trials), { text: tune ? "validation only" : `${(d.kinds || []).length} model kinds` });
    tile(k, "Best validation Sharpe", sig(best && best.validation_sharpe), { text: best ? `trial ${best.number}` : "" }, "", "The best of all the trials: it is inflated by having been picked from many");
    tile(k, "Deflated Sharpe", sig(sel.deflated_sharpe_probability), { text: `benchmark ${sig(sel.deflated_benchmark_sharpe_annual)} · 0.95 is the bar`, dir: sel.deflated_sharpe_probability >= 0.95 ? "up" : "" }, "", "Probability the best Sharpe beats what the best of this many random strategies would show");
    tile(k, "Probability of overfitting", sig(sel.pbo), { text: "chance the in-sample winner ranks below the median out of sample" }, "", "Combinatorially symmetric cross-validation (Bailey et al.)");
    tile(k, "Romano-Wolf p", sig(sel.romano_wolf_p_best, 3), { text: `${sel.survivors_at_5pct} survive at 5%` }, "", "Family-wise p-value of the best trial after correcting for having tried them all");
    const ch = h("div", "charts", root), pts = trials.filter((t) => !Q.bad(t.validation_sharpe)), fams = [...new Set(pts.map((t) => family(t.stage)))];
    chartCard(ch, "Every trial", "Validation Sharpe of each trial in the order it ran. Random trials explore, Bayesian ones follow the better regions, genetic ones recombine the best; a search that works drifts upward, and it is still only validation.", { wide: true,
      draw: (b) => C.scatter(b, { points: pts.map((t) => ({ x: t.number, y: t.validation_sharpe, color: css(STAGE_COLOR[family(t.stage)] || "--s4"), label: best && t.number === best.number ? "best" : "", tip: [["Stage", t.stage], ["Validation Sharpe", sig(t.validation_sharpe)], ["Score", sig(t.score)], ["Turnover", sig(t.turnover, 1) + "×"], ["Settings", paramText(t.params)]] })), legend: fams.map((s) => ({ name: s, color: css(STAGE_COLOR[s] || "--s4") })),
        xLabel: "Trial", yLabel: "Validation Sharpe", yZero: true, height: 300, label: "Search trials", labels: "all", xFormat: (v) => (Number.isInteger(v) ? String(v) : "") }),
      table: (b) => trialTable(b, trials) });
    const c2 = W.card(root, "Best trials by score", "The score is the validation Sharpe less penalties for turnover, drawdown, instability across validation blocks and the train-to-validation gap.");
    trialTable(c2, trials.slice().sort((a, b) => b.score - a.score).slice(0, 12));
    const fin = d.finalists || (d.final ? [{ trial: best.number, experiment_id: d.final.id, name: d.final.report.name, params: best.params, validation_score: best.score, segments: d.final.report.segments.stats }] : []);
    if (fin.length) {
      const c3 = W.card(root, tune ? "The best settings, on every segment" : "Finalists, run on every segment", "Each is fitted once more on all the segments and saved as an experiment, with every trial counted against its deflated Sharpe ratio.");
      const t = h("table", "data", h("div", "scroll", c3)), hd = h("tr", "", h("thead", "", t)), tb = h("tbody", "", t);
      for (const label of ["Trial", "Settings", "Train", "Validation", "Test", "Forward", ""]) h("th", label === "Settings" || label === "Trial" || !label ? "" : "num", hd, label);
      for (const f of fin) {
        const tr = h("tr", "", tb); h("td", "", tr, String(f.trial)); h("td", "", tr, paramText(f.params));
        for (const s of SEGMENTS) h("td", "num", tr, sig((f.segments[s] || {}).sharpe));
        const open = h("button", "btn small", h("td", "", tr), "Open"); open.addEventListener("click", () => { invalidate(); openExperiment(f.experiment_id); });
      }
    }
  }
  const trialTable = (b, rows) => table(b, [{ label: "#", num: true, get: (t) => String(t.number) }, { label: "Stage", get: (t) => String(t.stage).replace("-", "\u2011") }, { label: "Validation Sharpe", num: true, get: (t) => sig(t.validation_sharpe) }, { label: "Score", num: true, get: (t) => sig(t.score) },
    { label: "Turnover", num: true, get: (t) => sig(t.turnover, 1) + "×" }, { label: "Settings", get: (t) => paramText(t.params) }], rows);

  /* ---------- combine ---------- */
  function buildCombine(root, d) {
    const rows = d.rows.slice().sort((a, b) => b.sharpe - a.sharpe), best = rows[0];
    header(root, `Combining ${d.experiments.length} strategies`, `${rows.length} methods · from ${d.start} · each re-estimates its weights every ${M.form.every} days from the past only`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Best method", METHOD_NAME[best.method] || best.method, { text: `Sharpe ${sig(best.sharpe)} · return ${spct(best.ann_return)}` });
    const eq = rows.find((r) => r.method === "equal"); if (eq) tile(k, "Equal weight", sig(eq.sharpe), { text: `return ${spct(eq.ann_return)} · worst fall ${pct(eq.max_drawdown, 0)}` }, "", "The simple benchmark every other method has to beat after its own estimation error");
    const strat = Object.values(d.strategy_sharpes); tile(k, "The strategies alone", `${sig(Math.min(...strat))} to ${sig(Math.max(...strat))}`, { text: "Sharpe, full period" });
    for (const [id, msg] of Object.entries(d.errors || {})) W.flags(root, [`${METHOD_NAME[id] || id}: ${msg}`], "Not run. ");
    const ch = h("div", "charts", root), top = rows.slice(0, 4), ensure = eq && !top.includes(eq) ? [...top, eq] : top;
    chartCard(ch, "Growth of 1", `The best ${top.length} methods by Sharpe${ensure.length > top.length ? ", and equal weight" : ""}; the table has all of them. Estimated from the past only, so nothing here looks ahead.`, { wide: true,
      draw: (b) => C.line(b, { dates: d.curves.dates, series: ensure.map((r) => ({ name: METHOD_NAME[r.method] || r.method, color: expColor("m:" + r.method), values: d.curves[r.method] })), baseline: 1, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 300, label: "Equity curves of the combination methods", endLabels: false }),
      table: (b) => table(b, [{ label: "Date", get: (i) => d.curves.dates[i] }, ...rows.map((r) => ({ label: METHOD_NAME[r.method] || r.method, num: true, get: (i) => num(d.curves[r.method][i], 4) }))], d.curves.dates.map((_, i) => i).filter((i) => i % 4 === 0)) });
    const card = W.card(root, "Every method", "Sharpe ratios are net of the strategies' own costs.");
    const ids = d.experiments, nm = d.names;
    table(card, [{ label: "Method", get: (r) => METHOD_NAME[r.method] || r.method }, { label: "Sharpe", num: true, get: (r) => sig(r.sharpe) }, { label: "Return a year", num: true, get: (r) => spct(r.ann_return) }, { label: "Volatility", num: true, get: (r) => pct(r.ann_vol) },
      { label: "Worst fall", num: true, get: (r) => pct(r.max_drawdown) }, { label: "Weight turnover", num: true, get: (r) => sig(r.turnover, 2) }, ...OOS.filter((s) => rows[0].segments[s] !== undefined).map((s) => ({ label: SEG_NAME[s], num: true, get: (r) => sig((r.segments || {})[s]) }))], rows);
    const cut = (id) => { const t = nm[id] || shortId(id); return t.length > 13 ? t.slice(0, 12) + "…" : t; }, ch3 = h("div", "charts", root);
    chartCard(ch3, "Where each method ended", "The weight each method gave each strategy on the last day: the darker the cell, the larger the weight. A method that puts everything in one strategy is choosing, not diversifying.", { wide: true,
      draw: (b) => C.matrix(b, { rows: rows.map((r) => METHOD_NAME[r.method] || r.method), cols: ids.map(cut), values: rows.map((r) => ids.map((id) => (r.final_weights || {})[id] || 0)), min: 0, max: 1, format: (v) => pct(v, 0), name: "Weight", label: "Final weights by method and strategy", cellHeight: 26 }),
      table: (b) => table(b, [{ label: "Method", get: (r) => METHOD_NAME[r.method] || r.method }, ...ids.map((id) => ({ label: nm[id] || shortId(id), num: true, get: (r) => pct((r.final_weights || {})[id] || 0, 1) }))], rows) });
    const c2 = W.card(root, "The strategies on their own", "Sharpe by segment, from each experiment's saved returns.");
    table(c2, [{ label: "Strategy", get: (id) => nm[id] || shortId(id) }, { label: "Full period", num: true, get: (id) => sig(d.strategy_sharpes[id]) }, ...OOS.map((s) => ({ label: SEG_NAME[s], num: true, get: (id) => sig(((d.strategy_segments || {})[id] || {})[s]) }))], ids);
    if (ids.length >= 3 && d.similarity && d.similarity.explained > 0) {
      const ch2 = h("div", "charts", root);
      chartCard(ch2, "Strategy similarity map", `Strategies whose returns move together sit close (it keeps ${pct(d.similarity.explained, 0)} of the structure in two dimensions). Combining helps most when the points are far apart.`, {
        draw: (b) => C.scatter(b, { points: d.similarity.names.map((id, i) => ({ x: d.similarity.x[i], y: d.similarity.y[i], label: nm[id] || id, color: expColor(id) })), labels: "all", height: 300, label: "Strategy similarity map", xFormat: (v) => v.toFixed(1), yFormat: (v) => v.toFixed(1) }),
        table: (b) => table(b, [{ label: "Strategy", get: (i) => nm[d.similarity.names[i]] || d.similarity.names[i] }, { label: "x", num: true, get: (i) => num(d.similarity.x[i], 3) }, { label: "y", num: true, get: (i) => num(d.similarity.y[i], 3) }], d.similarity.names.map((_, i) => i)) });
    }
  }

  /* ---------- forward test ---------- */
  function buildForward(root, d) {
    const s = d.summary, step = (M.results.forward_step || {}).data, live = step && step.summary ? step.summary : s, drift = (live.drift || {}), lic = live.live_ic || {};
    header(root, `Forward test: ${s.name}`, `${s.n_signals} signal${s.n_signals === 1 ? "" : "s"} recorded · ${s.n_settled} settled · horizon ${s.horizon} day${s.horizon === 1 ? "" : "s"} · rebalanced ${d.rebalance}${s.first_signal ? ` · ${s.first_signal} to ${s.last_signal}` : ""}`);
    W.provenance(root, d);
    const k = h("div", "earnrow", root);
    tile(k, "Signals recorded", String(s.n_signals), { text: `${s.n_backfilled} replayed from history` }, "", "A replayed signal was recorded after its date: it is history, not a live forecast");
    tile(k, "Settled", String(s.n_settled), { text: "window completed" });
    tile(k, "Live rank IC", Q.bad(lic.mean) ? "n/a" : sig(lic.mean, 3), { text: Q.bad(lic.mean) ? "needs settled signals" : `${lic.n} signals · backtest ${sig(lic.backtest_mean === undefined ? d.backtest_ic : lic.backtest_mean, 3)}` }, "", "The rank IC of signals recorded as the days arrived, against what the backtest promised");
    if (drift.n_alert !== undefined) { tile(k, "Features drifting", String(drift.n_alert), { text: `${drift.n_watch} to watch`, dir: drift.n_alert ? "down" : "" }, "", "Population stability index of each feature against its training distribution: above 0.25 is a shift"); tile(k, "Prediction drift", sig(drift.prediction_psi, 2), { text: "PSI of the scores" }); }
    const alerts = (live.alerts || []).filter((a) => !/recorded after their date/.test(a)); if (alerts.length) W.flags(root, alerts, "Alert. ");
    const sig1 = step && step.signal;
    if (sig1) {
      const c = W.card(root, `Signal for ${sig1.date}`, `${sig1.rebalance_day ? "A rebalance day: these orders move the book to the target." : "Not a rebalance day: the book stays as it was."} Paper only: nothing is sent anywhere. Turnover ${sig(sig1.turnover, 2)}.`);
      if (sig1.orders.length) table(c, [{ label: "Asset", get: (o) => o.asset }, { label: "Side", get: (o) => o.side }, { label: "Weight change", num: true, get: (o) => spct(o.weight_change, 1) }, { label: "Notional", num: true, get: (o) => Q.money(o.notional) }, { label: "Target weight", num: true, get: (o) => spct(sig1.weights[o.asset] || 0, 1) }], sig1.orders);
      else W.hint(c, "No orders: the book already matches the target.");
    }
    const top = drift.top || [];
    if (top.length) {
      const ch = h("div", "charts", root);
      chartCard(ch, "Which features have drifted", "Population stability index of each feature's live distribution against its training distribution. Below 0.1 is steady, above 0.25 is a shift: the model is seeing inputs it did not train on.", { wide: true,
        draw: (b) => C.hbars(b, { rows: top.map((x) => ({ label: x.feature, value: x.psi, tip: [["Status", x.status]] })), format: (v) => v.toFixed(2), name: "PSI", label: "Feature drift", labelWidth: 120, color: (v) => (v >= 0.25 ? css("--neg") : v >= 0.1 ? css("--warn") : css("--s1")) }),
        table: (b) => table(b, [{ label: "Feature", get: (x) => x.feature }, { label: "PSI", num: true, get: (x) => num(x.psi, 3) }, { label: "Status", get: (x) => x.status }], top) });
    }
    const c2 = W.card(root, "Recorded signals", "Each signal is the list of positions the model wanted on its date. 'Realised' is what those positions earned over the horizon once it was over.");
    if (!d.signals.length) { W.hint(c2, "No signal is recorded yet: press the button on the left to record the newest."); return; }
    table(c2, [{ label: "Date", get: (x) => x.date }, { label: "Recorded", get: (x) => x.recorded }, { label: "Positions", num: true, get: (x) => String(x.n_positions) }, { label: "Turnover", num: true, get: (x) => sig(x.turnover, 2) },
      { label: "Realised", num: true, get: (x) => (Q.bad(x.realised) ? "open" : spct(x.realised, 2)) }, { label: "Settled through", get: (x) => x.settled_through || "" }], d.signals.slice().reverse());
  }

  Q.registerWorkspace("ml", M);
})();
