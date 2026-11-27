/* The Machine learning workspace. A model is trained only inside the whole pipeline: features built without look-ahead, walk-forward fits whose training rows are purged around every test day, a backtest
   with costs through the platform's engine, and a result cut into train (in-sample), validation (where choices are made), test and forward (reserved) segments, with every look at the same data counted.
   Ten views share one form and one idea of the current experiment: Dataset, Train, Experiments, Compare, Explain, Overfitting, Search, Combine, Forward test and Research controls. It talks to /api/ml/*.
   Whatever the server sends goes in with textContent. The ETF and crypto universes are platform data; the futures universe is simulated and every payload says so, and this page prints it. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q.C, h = Q.h, css = Q.css, num = Q.num, pct = Q.pct, tile = Q.tile, chartCard = Q.chartCard, table = Q.simpleTable, api = Q.api;
  const GUIDE = "technique-machine-learning-research-pipeline";
  const SEGMENTS = ["train", "validation", "test", "forward"], OOS = ["validation", "test", "forward"];
  const SEG_NAME = { train: "Train (in-sample)", validation: "Validation", test: "Test", forward: "Forward" };
  const SORTS = [["validation.sharpe", "Validation Sharpe"], ["test.sharpe", "Test Sharpe"], ["forward.sharpe", "Forward Sharpe"], ["oos.sharpe", "Sharpe over all out-of-sample days"], ["ic.mean", "Rank IC"], ["ic.ir", "Rank IC information ratio"], ["overfit.generalization_score", "Generalization score"]];
  const SIGNAL_NAME = { top_n_long_short: "Long the top n, short the bottom n", long_only_top_quantile: "Long only: the top quantile", benchmark_tilt: "Long only: tilted around equal weight", proportional: "Proportional to the score", vol_scaled: "Score divided by volatility", rank_normalised: "Weighted by rank",
    threshold: "Long above, short below a threshold", regime_filtered: "Only in the calm regime", confidence_weighted: "Score times the model's confidence",
    uncertainty_weighted: "Score over the model's own uncertainty", uncertainty_mean_variance: "Uncertainty-adjusted views through the covariance", target_weights: "The scores are the weights (the agents)" };
  const SIGNAL_FIELDS = { benchmark_tilt: ["tilt"], top_n_long_short: ["n"], long_only_top_quantile: ["quantile"], threshold: ["threshold"], uncertainty_weighted: ["kappa", "uncertainty_power"], uncertainty_mean_variance: ["kappa", "uncertainty_power", "mv_lookback"], cost_aware_mean_variance: ["risk_aversion", "cov_window", "opt_every", "turnover_penalty"], calibrated_sizing: ["risk_aversion", "calib_window", "calib_min_t"] };
  const RL_KINDS = new Set(["ppo", "sac", "recurrent_ppo"]), SLOW = new Set(["deep sequence", "forecasting", "representation", "graph", "reinforcement"]);
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
    id: "ml", root: "#ml", view: "#view-ml", title: "Machine learning", api: 8, mode: "experiments", busy: "", results: {}, errors: {}, cache: new Map(), guard: null, go: null, progress: null,
    about: "Train models only inside the whole pipeline: features without look-ahead, walk-forward fits, a backtest with costs, and results cut into train, validation, test and forward segments, with every look at the data counted.",
    modes: [["dataset", "Dataset"], ["train", "Train"], ["experiments", "Experiments"], ["compare", "Compare"], ["explain", "Explain"], ["overfit", "Overfitting"], ["search", "Search"], ["select", "Select"], ["combine", "Combine"], ["forward", "Forward test"], ["allocation", "Portfolio allocation"], ["controls", "Research controls"], ["training", "Training monitor"]],
    runLabel: { dataset: "Build the dataset", train: "Train and backtest", experiments: "Refresh", compare: "Compare", explain: "Explain", overfit: "Check again", search: "Run the search", select: "Choose", combine: "Combine", allocation: "Allocate", forward: "Record today's signal", controls: "Refresh", training: "Refresh" },
    blurb: { dataset: "Causal features and forward-looking targets, with each feature's coverage and its information coefficient on the warm-up segment only.",
      train: "Walk-forward training, position rules, a backtest with costs, and the result split into train, validation, test and forward segments.",
      experiments: "Every saved experiment, ranked, with the number of looks at the same data when it ran. Open one, delete one, pick several to compare or combine.",
      compare: "Settings and results side by side, equity curves over their shared days, correlations, and a map of how alike the strategies are.",
      explain: "Why the saved model scored an asset as it did on a date: Shapley values split the score exactly into what each feature added.",
      overfit: "How much of the best result survives where nothing was fitted: Sharpe by segment, the deflated Sharpe ratio, stress tests and meta-labelling.",
      search: "Random trials, then Bayesian, then genetic, on validation only; the best few are run on every segment once, and every trial is counted.",
      select: "Choose among saved experiments on their validation rows only: eligibility gates, the pooled validation net Sharpe, a Pareto comparison and the simplest candidate that is not meaningfully worse.",
      combine: "Ten ways to weight several strategies, each estimated only from the past, side by side.",
      training: "Every deep-network training on this machine, read from its files on disk: loss and learning rate by epoch, memory, checkpoints, and what to do with a run that was interrupted. Reloading the page loses nothing.",
      allocation: "Turn the saved forecasts into per-stock weights from expected returns, risk, correlation and limits, backtest them with costs, and compare every allocator with equal weight on identical forecasts.",
      forward: "Paper signals recorded as the days arrive, settled when their window ends, with the live rank IC and the drift of each feature against the backtest.",
      controls: "The research record of one experiment: the three independent gates and why each is or is not passed, every try in its study, what was read of the protected test and forward outcomes, and which selection tests applied." },
    catalog: null, catalogError: "", board: null, boardPromise: null, picked: new Set(), cur: Q.store("ml.cur") || "",
    form: { name: "", notes: "", universe: "etf", tickers: "", groups: new Set(), kind: "fwd_ret", horizon: "5", normalize: "cross_sectional_z", winsor: "1", minHistory: "252", start: "", end: "", regime: "",
      window: "expanding", rollingDays: "1260", minTrain: "756", refit: "", embargo: "0", valFrac: "20", testFrac: "15", fwdFrac: "15", model: "ridge", params: {},
      signal: "top_n_long_short", n: "3", quantile: "30", threshold: "0", gross: "1", net: "0", maxWeight: "35", holding: "5", rebalance: "weekly", cost: "5", slippage: "2", targetVol: "10", maxLev: "1.5",
      explain: true, robust: false, finalize: false, seed: "0", sort: "validation.sharpe", fModel: "", fStage: "", fUniverse: "", exp: "", asset: "", date: "", perms: "40",
      selection: false, selPrimary: "net_sharpe", selMaxDd: "40", selMinDays: "126", selMinCov: "80", selMinSharpe: "0", selSimplicity: true, selRule: "one_se", selK: "1", selMargin: "0.25",
      searchKind: "search", engine: "builtin", kinds: new Set(), nRandom: "8", nBayes: "6", generations: "2", population: "6", topK: "2", tuneEffort: "standard",
      methods: new Set(), lookback: "252", every: "21", riskAversion: "5", capital: "100000", asOf: "", kappa: "0", uncertainty_power: "1", mv_lookback: "126", risk_aversion: "5", cov_window: "252", opt_every: "5", turnover_penalty: "0", calib_window: "504", calib_min_t: "2" },
  };

  Object.assign(M.form, { alWhat: "allocate", alAllocator: "mean_variance", alKind: "return", alCov: "lw_constant_correlation", alCovWindow: "252", alRiskAversion: "5", alCalibrate: "slope", alMinT: "1", alMinW: "0", alMaxW: "60", alLongOnly: true, alCash: false,
    alGross: "1", alTurnover: "", alVolTarget: "", alSector: "", alOptEvery: "", alOnFailure: "risk_only", alTurnPen: "0", alKelly: "0.5", alRebalance: "weekly", alCommission: "5", alSlippage: "2", alProtected: false, alStab: false,
    alHolding: "1", alLifecycle: "single_book", alSource: "scaled", alEdgeModel: "linear", alBeyond: "zero", alFxBase: "none", alFxCcy: "", alFxHedge: "0", alFxCost: "3", alFxHedgeCost: "2", alFxMissing: "raise", alCovHorizon: "linear", alCalMethod: "pooled", alInference: "sqrt_horizon", alShrink: "0", alMinEff: "30", alMinNeff: "0", alMinConf: "0", alPreset: "none", alBaseline: "none", alBlend: "none",
    alAlpha: "0.5", alBlendMax: "1", alMaxChange: "", alSens: false });
  M.alloc = { cat: null, run: null, compare: null, calibration: null, error: "" };

  /* ---------- what the server offers, and the list of experiments ---------- */
  function init(cat) {
    const d = cat.defaults, t = /^(.*)_(\d+)$/.exec(d.dataset.target);
    Object.assign(M.form, { universe: d.dataset.universe, groups: new Set(d.dataset.feature_groups), kind: t ? t[1] : "fwd_ret", horizon: String(d.dataset.horizon), normalize: d.dataset.normalize, winsor: pc(d.dataset.winsor), minHistory: String(d.dataset.min_history),
      window: d.split.window, rollingDays: String(d.split.rolling_days), minTrain: String(d.split.min_train_days), embargo: String(d.split.embargo), valFrac: pc(d.split.validation_fraction), testFrac: pc(d.split.test_fraction), fwdFrac: pc(d.split.forward_fraction),
      model: d.model.kind, signal: d.signal.method, tilt: String(d.signal.tilt ?? 0.5), n: String(d.signal.n), quantile: pc(d.signal.quantile), threshold: String(d.signal.threshold), gross: String(d.signal.gross), net: String(d.signal.net), maxWeight: pc(d.signal.max_weight), holding: String(d.signal.holding_period),
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
    if (m === "select" && (M.picked.size < 1 || M.picked.size > 20)) return "Choose between one and twenty experiments.";
    if (m === "combine" && (M.picked.size < 2 || M.picked.size > 12)) return "Choose between two and twelve experiments.";
    if (m === "combine" && !f.methods.size) return "Choose at least one method.";
    if (m === "allocation") { if (!M.cur) return M.board ? "Train an experiment first." : "Loading the experiments…"; try { allocationBody(f); } catch (e) { return e.message; } }
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
    Q.labField(b, { key: "normalize", label: "Scale features each day", kind: "choice", options: [["cross_sectional_z", "Z-score across assets"], ["rank", "Rank across assets"], ["time_series_z", "Z-score against each asset's own past (works with 1 or 2 tickers)"], ["none", "Leave as they are"]] }, f, after);
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
  /* A model's settings come with the catalogue's schema (type, range, choices); the hints and the default's own type fill in for a catalogue that has none. */
  const paramSpec = (info, key) => {
    const d = info.params[key], sp = (info.specs || {})[key] || {}, sc = (info.schema || {})[key] || {};
    return { type: sc.type || sp.type || (typeof d === "number" ? "number" : "text"), choices: (sc.choices && sc.choices.length ? sc.choices : sp.choices) || [], help: sp.help || "", range: sc.range || "", format: sc.format || "" };
  };
  const formValue = (info, key, v) => { const t = paramSpec(info, key).type; return t === "bool" ? !!Number(v) : Array.isArray(v) ? v.join(",") : String(v); };      // what a setting's field shows: the default, as the person would type it
  /* Every setting of the chosen model is on the form, each starting at its default; the form keeps what the person typed, and a setting that has no value yet (the tuned ones come first) gets its default. */
  function ownParams(info) {
    const own = (M.form.params[info.kind] = M.form.params[info.kind] || {});
    for (const [k, v] of Object.entries(info.params)) if (!(k in own)) own[k] = formValue(info, k, v);
    return own;
  }
  const sameAsDefault = (sp, value, d) => (sp.type === "bool" ? value === (Number(d) ? 1 : 0) : sp.type === "list" ? value === (Array.isArray(d) ? d.join(",") : String(d).trim()) : sp.type === "choice" || sp.type === "text" ? value === String(d).trim() : value === Number(d));
  function modelBlock(b) {
    const f = M.form, cat = M.catalog, kinds = modelKinds(), usable = kinds.filter((k) => k.available);
    fixModel();
    Q.labField(b, { key: "model", label: "Model", kind: "choice", options: usable.map((k) => [k.kind, `${k.label} (${k.family})`]) }, f, () => {
      if (RL_KINDS.has(f.model) && f.signal !== "target_weights") f.signal = "target_weights";               // an agent sizes its own book
      else if (!RL_KINDS.has(f.model) && f.signal === "target_weights") f.signal = "top_n_long_short";
      Q.renderLab(M);
    });
    const info = cat.models.find((k) => k.kind === f.model);
    if (info) {
      h("p", "hint", b, info.description);
      const arch = (cat.architectures || {})[f.model];                                  // the architecture manifest: what the network is, and, for a reduced one, what it leaves out
      if (arch) {
        h("p", "hint", b, `Architecture: ${arch.variant} (${arch.reference}).` + (arch.reduced && (arch.differences || []).length ? ` It leaves out or changes: ${arch.differences.join("; ")}.` : ""));
        const dt = h("details", "how", b); h("summary", "", dt, "What this network reads, returns and does with missing values");
        table(dt, [{ label: "Aspect", get: (x) => x[0] }, { label: "This network", get: (x) => x[1] }], [["Input", arch.input_axes], ["Output", arch.output_axes], ["Missing values", arch.masking], ["Normalisation", arch.normalization],
          ["Residual connections", arch.residuals], ["Attention", arch.attention], ["Forecast horizons", arch.horizon_mapping]]);
      }
      if (SLOW.has(info.family)) h("p", "hint", b, "This model trains a network or an agent at every refit, which takes minutes rather than seconds on a CPU. Raise Refit every (days) or lower the epochs or timesteps while exploring.");
      const own = ownParams(info);
      for (const k of Object.keys(info.params)) {
        const sp = paramSpec(info, k), label = k.replace(/_/g, " "), after = () => Q.syncLab(M);
        const help = [sp.help, sp.format ? `Written as ${sp.format}.` : "", sp.range ? `Allowed: ${sp.type === "int" ? "a whole number, " : ""}${sp.range}.` : ""].filter(Boolean).join(" ");
        if (sp.type === "choice") Q.labField(b, { key: k, label, kind: "choice", options: sp.choices.map((c) => [c, c]), help }, own, after);
        else if (sp.type === "bool") Q.labField(b, { key: k, label, kind: "bool", help }, own, after);
        else if (sp.type === "text" || sp.type === "list") Q.labField(b, { key: k, label, kind: "text", help }, own, after);
        else Q.labField(b, { key: k, label, kind: "number", step: "any", help }, own, after);
      }
      if (Object.keys(info.params).length) h("p", "hint", b, "Only the settings you change are sent; the server fills in the rest with the model's defaults, and the saved experiment records them all.");
      const says = { distribution: "prediction intervals", regimes: "regime probabilities", heads: "forecasts by horizon", training: "training curves", weights: "member weights", ranking: "ranking metrics", attention: "attention", rewards: "reward curves" };
      if ((info.outputs || []).length) h("p", "hint", b, "Besides its scores this model reports: " + info.outputs.map((o) => says[o] || o).join(", ") + ".");
    }
    const missing = kinds.filter((k) => !k.available);
    if (missing.length) h("p", "hint", b, "Not available here: " + missing.map((k) => `${k.label} (${k.reason})`).join("; ") + ".");
  }
  function signalBlock(b) {
    const f = M.form, cat = M.catalog, after = () => Q.syncLab(M);
    Q.labField(b, { key: "signal", label: "Turn scores into positions", kind: "choice", options: cat.signals.map((s) => [s.id, SIGNAL_NAME[s.id] || s.id]) }, f, () => Q.renderLab(M));
    const info = cat.signals.find((s) => s.id === f.signal); if (info) h("p", "hint", b, info.text + ".");
    const spec = { tilt: { label: "Tilt (0 = the benchmark, 1 = the best asset at twice its weight, the worst at none)", step: 0.05, min: 0, max: 1 }, n: { label: "n (assets on each side)", step: 1, min: 1, max: 20 }, quantile: { label: "Top quantile (%)", step: 5, min: 5, max: 50 }, threshold: { label: "Threshold (in score units)", step: "any" },
      kappa: { label: "Shrink views on unusually uncertain assets (kappa)", step: 0.1, min: 0, max: 10 }, uncertainty_power: { label: "Divide the view by uncertainty to the power", step: 0.25, min: 0, max: 3 },
      mv_lookback: { label: "Covariance window (days)", step: 21, min: 40, max: 504 }, risk_aversion: { label: "Risk aversion (the weight on variance)", step: 0.5, min: 0, max: 100 },
      cov_window: { label: "Covariance window (days)", step: 21, min: 60, max: 1000 }, opt_every: { label: "Re-solve every (days; set to the horizon)", step: 1, min: 1, max: 60 },
      turnover_penalty: { label: "Quadratic turnover penalty", step: 0.1, min: 0, max: 10 }, calib_window: { label: "Calibration window (settled decisions)", step: 21, min: 60, max: 2000 },
      calib_min_t: { label: "Smallest t-statistic of the calibration that sizes a bet", step: 0.5, min: 0, max: 10 } };
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
    if (!M.board) { Q.loading(b, "Loading the experiments…"); ensureBoard(); return false; }
    const rows = M.board.rows.slice().sort(newest), usable = rows.filter((r) => !needModel || r.has_model !== false), f = M.form;
    if (!rows.length) { h("p", "hint", b, "No experiment is saved yet: train one in the Train view first."); return false; }
    if (!usable.length) { h("p", "hint", b, "No saved experiment has a model of its own: train one in the Train view first."); return false; }
    if (!usable.some((r) => r.id === M.cur)) { const now = rows.find((r) => r.id === M.cur), parent = now && usable.find((r) => r.id === now.parent); setCurrent((parent || usable[0]).id); }
    f.exp = M.cur;
    Q.labField(b, { key: "exp", label: "Experiment", kind: "choice", options: usable.map((r) => [r.id, expLabel(r)]) }, f, () => {
      setCurrent(f.exp); f.asset = ""; for (const k of ["explain", "overfit", "forward", "forward_step", "meta", "child", "allocation", "alloc_compare", "alloc_cal"]) { M.results[k] = null; M.errors[k] = null; }
      M.alloc.run = M.alloc.compare = M.alloc.calibration = null;                                                      // an allocation belongs to the forecasts it was made from
      Q.renderLab(M);
    });
    if (usable.length < rows.length) h("p", "hint", b, "Re-backtests and meta-labelled runs are not listed: they have no model of their own.");
    return true;
  }
  function pickList(b, title, max) {                                                       // several saved experiments, shared by Compare and Combine
    if (!M.board) { Q.loading(b, "Loading the experiments…"); ensureBoard(); return; }
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
  /* ---------- the constrained selection (P1-15, P2-17): thresholds are fields, each with the default the server documents ---------- */
  function selectionFields(b, f, withSwitch) {
    const after = () => Q.syncLab(M), again = () => Q.renderLab(M);
    if (withSwitch) Q.labField(b, { key: "selection", label: "Choose the winner with the constrained selection", kind: "bool", help: "Off by default: the result is what it always was. On: every trial is judged on its pooled validation returns against the requirements below, and the simplest trial that is not meaningfully worse than the best is selected." }, f, again);
    if (withSwitch && !f.selection) return;
    Q.labField(b, { key: "selPrimary", label: "Rank by", kind: "choice", options: [["net_sharpe", "Validation net Sharpe (default)"], ["net_sortino", "Net Sortino"], ["net_calmar", "Net Calmar"], ["net_ann_return", "Net return a year"], ["ic", "Rank IC"]] }, f, after);
    Q.labField(b, { key: "selMaxDd", label: "Largest drawdown allowed (%)", kind: "number", step: 1, min: 1, max: 100 }, f, after);
    Q.labField(b, { key: "selMinDays", label: "Fewest validation days", kind: "number", step: 1, min: 20, max: 5000 }, f, after);
    Q.labField(b, { key: "selMinCov", label: "Fewest days with two assets scored (%)", kind: "number", step: 1, min: 0, max: 100 }, f, after);
    Q.labField(b, { key: "selMinSharpe", label: "Net Sharpe must be above (blank = no floor)", kind: "number", step: 0.1 }, f, after);
    Q.labField(b, { key: "selSimplicity", label: "Prefer the simplest candidate that is not meaningfully worse", kind: "bool", help: "Uses the paired difference of the candidates on the days they share, with a dependence-aware bootstrap. No fixed penalty on the Sharpe ratio." }, f, again);
    if (f.selSimplicity) {
      Q.labField(b, { key: "selRule", label: "Not meaningfully worse means", kind: "choice", options: [["one_se", "Heuristic: within k standard errors of the paired gap"], ["non_inferiority", "Formal: non-inferior within a margin I declare"]] }, f, again);
      if (f.selRule === "non_inferiority") Q.labField(b, { key: "selMargin", label: "Margin (in units of the ranking metric, declared before the results)", kind: "number", step: 0.05, min: 0.001 }, f, after);
      else Q.labField(b, { key: "selK", label: "k (standard errors of the paired gap)", kind: "number", step: 0.25, min: 0, max: 10 }, f, after);
    }
    h("p", "hint", b, "Every other threshold has a documented default: GET /api/ml/selection-policy. The test and forward segments are never read.");
  }
  function selectionBody(f) {
    const floor = optNumber(f.selMinSharpe, "Net Sharpe floor");
    const eligibility = { max_drawdown: needNumber(f.selMaxDd, "Largest drawdown") / 100, min_validation_days: needNumber(f.selMinDays, "Validation days"), min_coverage: needNumber(f.selMinCov, "Coverage") / 100, min_net_sharpe: floor === undefined ? null : floor };
    const simplicity = !f.selSimplicity ? { enabled: false } : f.selRule === "non_inferiority" ? { rule: "non_inferiority", margin: needNumber(f.selMargin, "Margin") } : { rule: "one_se", k: needNumber(f.selK, "k") };
    return { primary_metric: f.selPrimary, eligibility, simplicity };
  }
  M.sections = (panel) => {
    watchRuns();                                                                             // find the runs again when the tab opens or the mode changes
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
    } else if (m === "training") {
      const b = sec("1. Training runs");
      h("p", "hint", b, "Runs are read from data/user/training. Metrics are the training and validation loss only: the test and forward segments are never read while a network trains.");
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
    } else if (m === "overfit" || m === "forward" || m === "controls") {
      const b = sec("1. The experiment");
      if (expSelect(b, m === "forward") && m === "forward") {
        Q.labField(b, { key: "capital", label: "Paper capital ($)", kind: "number", step: 1000, min: 1000 }, f, after);
        Q.labField(b, { key: "asOf", label: "Record as of (blank = the newest date)", kind: "text", placeholder: "2024-06-28" }, f, after);
        h("p", "hint", b, "A signal is a list of paper orders: nothing is sent anywhere. The newest date of the data may be old; the signal is dated by it.");
      }
    } else if (m === "allocation") {
      allocationSections(sec, f, after);
    } else if (m === "search") {
      const b = sec("1. What to run");
      W.seg(b, [["search", "Search for strategies"], ["tune", "Tune one model"]], f.searchKind, (v) => { f.searchKind = v; Q.renderLab(M); });
      Q.labField(b, { key: "finalize", label: "Also evaluate the best on the test and forward segments, once", kind: "bool", help: "Off by default: the search stops at the end of the validation segment. On: a plan is frozen from the best and evaluated once, which uses up that holdout (it is recorded, and a second look would not be untouched)." }, f, after);
      if (f.searchKind === "search") {
        const usable = cat.models.filter((k) => k.available && k.tasks.includes("regression"));
        Q.checkList(b, "Models to search over:", usable.map((k) => [k.kind, k.label]), f.kinds, after);
        for (const s of [{ key: "nRandom", label: "Random trials", min: 0, max: cat.limits.search_random }, { key: "nBayes", label: "Bayesian (TPE) trials", min: 0, max: cat.limits.search_bayes }, { key: "generations", label: "Genetic generations", min: 0, max: cat.limits.search_generations },
          { key: "population", label: "Genetic population", min: 2, max: cat.limits.search_population }, { key: "topK", label: "Finalists run on every segment", min: 1, max: cat.limits.search_top_k }]) Q.labField(b, { kind: "number", step: 1, ...s }, f, after);
        h("p", "hint", b, "The search varies the model, its settings, the horizon, the position rule and the rebalancing. The data and cost settings below stay as they are.");
      } else {
        const spaces = cat.tune_spaces || {}, tunable = cat.models.filter((k) => k.available && spaces[k.kind]);
        if (!tunable.some((k) => k.kind === f.model)) f.model = tunable[0].kind;                 // the form shows what it sends
        function paintVary() {                                                                   // what tuning this model varies, so nothing happens that the page did not say
          vary.replaceChildren();
          const rows = spaces[f.model] || [], label = (tunable.find((k) => k.kind === f.model) || {}).label || f.model;
          h("p", "hint", vary, `Tuning ${label} tries ${rows.length} settings: its own, and the ones that decide how its predictions become trades. The first try is the settings in the Train form (the model's own where you left them blank), so the answer comes with a comparison.`);
          const ul = h("ul", "tunelist", vary);
          for (const r of rows) {
            const li = h("li", "", ul); h("b", "", li, r.label + ": ");
            li.appendChild(document.createTextNode((r.kind === "categorical" ? r.choices.join(" or ") : `${setting(r.low)} to ${setting(r.high)}`) + ". " + r.text + "."));
          }
        }
        const box = h("div", "", b);
        Q.labField(box, { key: "model", label: "Model to tune", kind: "choice", options: tunable.map((k) => [k.kind, k.label]) }, f, () => { paintVary(); after(); });
        const vary = h("div", "tunevary", box), efforts = cat.tune_efforts || [];
        h("span", "hint", box, "How long to look:");
        const how = W.seg(box, efforts.map((e) => [e.id, cap(e.id)]), f.tuneEffort, (v) => { f.tuneEffort = v; paintEffort(); });
        how.dataset.role = "tune-effort";
        const engines = (cat.engines || []).filter((e) => e.available);
        if (engines.length > 1) Q.labField(box, { key: "engine", label: "Search engine", kind: "choice", options: engines.map((e) => [e.id, e.text]) }, f, after);
        const effort = h("p", "hint", box, "");
        function paintEffort() { effort.textContent = ((efforts.find((e) => e.id === f.tuneEffort) || {}).text || "") + ". Each try is a walk-forward experiment on the validation segment, which takes seconds to a minute."; }
        paintVary(); paintEffort();
      }
      const d = sec("2. Data", false); universeBlock(d); featureBlock(d); targetBlock(d);
      prepBlock(sec("3. Preparation and window", false)); segmentBlock(sec("4. Segments and refits", false)); signalBlock(sec("5. Position rule", false)); costBlock(sec("6. Costs and risk", false));
      selectionFields(sec("7. Selection (optional)", false), f, true);
    } else if (m === "select") {
      pickList(sec("1. Experiments"), "One to twenty saved experiments:", 20);
      selectionFields(sec("2. Requirements and rules"), f, false);
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
  /* The model of a request: its kind and only the settings the person changed. A setting left at the catalogue's default is not sent (a network has twenty or more, and the server lays what is sent over the
     defaults, so the experiment still records them all); an agent's cost left at its default is therefore the Backtest section's costs, which the server fills in. */
  function modelBody(f) {
    const info = (M.catalog.models.find((k) => k.kind === f.model)) || { kind: f.model, params: {}, specs: {} }, own = ownParams(info), params = {};
    for (const k of Object.keys(info.params)) {
      const sp = paramSpec(info, k), v = own[k], name = k.replace(/_/g, " ");
      let value;
      if (sp.type === "bool") value = v ? 1 : 0;
      else if (sp.type === "choice" || sp.type === "text" || sp.type === "list") value = String(v).trim();
      else if (String(v).trim() === "") continue;                                                               // a blank number is the default
      else { value = needNumber(v, name); if (sp.type === "int" && !Number.isInteger(value)) throw new Error(`${name} must be a whole number`); }
      if (!sameAsDefault(sp, value, info.params[k])) params[k] = value;
    }
    return { kind: f.model, params };
  }
  const baseBody = (f) => ({ dataset: datasetBody(f), split: splitBody(f), signal: signalBody(f), backtest: backtestBody(f) });
  const experiment = (id) => ({ modes: [], path: "/api/ml/experiment", query: { id }, cacheKey: "exp:" + id });
  const fresh = (mode) => { M.results[mode] = null; M.errors[mode] = null; };
  const JOB_MODES = new Set(["dataset", "train", "search", "forward"]);
  const CONTROLS = (id) => [{ modes: ["controls"], path: "/api/ml/promotion-status", query: { id } }, { modes: ["controls_lineage"], path: "/api/ml/lineage", query: { id } }, { modes: ["controls_holdout"], path: "/api/ml/holdout", query: {} },
    { modes: ["controls_stats"], path: "/api/ml/statistical-diagnostics", query: { id } }, { modes: ["controls_exec"], path: "/api/ml/trading-diagnostics", query: { id } },
    { modes: ["controls_policy"], path: "/api/ml/validation-policy", query: {} }, { modes: ["controls_gov"], path: "/api/ml/governance", query: { id } },
    { modes: ["controls_alpha"], path: "/api/ml/alpha-diagnostics", query: { id, stored: 1 } }, { modes: ["controls_port"], path: "/api/ml/portfolio-diagnostics", query: { id, stored: 1 } },
    { modes: ["controls_improve"], path: "/api/ml/improvement-experiment", query: {} }, { modes: ["controls_delay"], path: "/api/ml/delay-report", query: { id, stored: 1 } },
    { modes: ["controls_drift"], path: "/api/ml/drift", query: { id, stored: 1 } }, { modes: ["controls_monitor"], path: "/api/ml/uncertainty-monitor", query: { id, stored: 1 } },
    { modes: ["controls_retrain"], path: "/api/ml/retraining", query: { id } }, { modes: ["controls_reports"], path: "/api/ml/reports", query: { id } }];

  M.run = async (fromForm) => {
    const m = M.mode, f = M.form;
    if (!M.catalog) { loadCatalog(); return; }
    if (JOB_MODES.has(m)) setTimeout(watchRuns, 1200);                                       // a run that was just started appears in the list of runs
    if (m === "training") { stopTrainingPoll(); return W.request(M, () => [{ modes: ["training"], path: "/api/training/runs", query: {} }], "Reading the training runs…"); }
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
    if (m === "controls") return W.request(M, () => CONTROLS(M.cur), "Reading the research record…");
    if (m === "search") {
      const tune = f.searchKind === "tune", before = M.results.search;
      await W.request(M, () => (tune
        ? [{ modes: ["search"], job: true, path: "/api/ml/tune", body: { ...baseBody(f), model: modelBody(f), effort: f.tuneEffort, engine: f.engine, finalize: !!f.finalize, selection: f.selection ? selectionBody(f) : undefined } }]
        : [{ modes: ["search"], job: true, path: "/api/ml/search", body: { ...baseBody(f), kinds: [...f.kinds], finalize: !!f.finalize, n_random: needNumber(f.nRandom, "Random trials"), n_bayes: needNumber(f.nBayes, "Bayesian trials"), generations: needNumber(f.generations, "Generations"),
          population: needNumber(f.population, "Population"), top_k: needNumber(f.topK, "Finalists"), selection: f.selection ? selectionBody(f) : undefined } }]), tune ? "Tuning…" : "Searching…");
      if (M.results.search !== before) invalidate();
      return;
    }
    if (m === "select") return W.request(M, () => [{ modes: ["select"], path: "/api/ml/select", body: { experiments: [...M.picked], policy: selectionBody(f) } }], "Choosing among the experiments…");
    if (m === "combine") return W.request(M, () => [{ modes: ["combine"], path: "/api/ml/combine", body: { ids: [...M.picked], methods: [...f.methods], lookback: needNumber(f.lookback, "Window"), rebalance: needNumber(f.every, "Re-estimate every"), risk_aversion: needNumber(f.riskAversion, "Risk aversion") } }], "Combining…");
    if (m === "allocation") {
      if (f.alWhat === "compare") {
        const policies = f.alStab ? { mv_conservative: { label: "Mean-variance, conservative preset", allocator: "mean_variance", preset: "conservative" },
          mv_blend: { label: "Mean-variance blended 50/50 with inverse volatility", allocator: "mean_variance", baseline: "inverse_vol", blend: "fixed", blend_alpha: 0.5 } } : undefined;
        const got = await sideJob("alloc_compare", "/api/ml/allocation/compare", { id: M.cur, allocation: allocationBody(f, true), evaluate: f.alProtected ? ["test", "forward"] : [], policies }, "Comparing the allocators…");
        if (got) { M.alloc.compare = got; if (!M.results.allocation) M.results.allocation = { data: { compareOnly: true }, body: {} }; M.draw(); }
        return;
      }
      if (f.alWhat === "calibration") {
        const got = await sideJob("alloc_cal", "/api/ml/allocation/calibration", { id: M.cur, allocation: allocationBody(f, true) }, "Comparing the calibration methods…");
        if (got) { M.alloc.calibration = got; if (!M.results.allocation) M.results.allocation = { data: { compareOnly: true }, body: {} }; M.draw(); }
        return;
      }
      const before = M.results.allocation;
      await W.request(M, () => [{ modes: ["allocation"], job: true, path: "/api/ml/rebacktest", body: { id: M.cur, name: "allocation-" + f.alAllocator, signal: { method: "portfolio_allocation", holding_period: needNumber(f.alHolding, "Holding period"), allocation: allocationBody(f) },
        backtest: { rebalance: f.alRebalance, cost_bps: needNumber(f.alCommission, "Commission"), slippage_bps: needNumber(f.alSlippage, "Slippage"), ...fxBody(f) } } }], "Allocating and backtesting…");
      if (M.results.allocation && M.results.allocation !== before) { M.alloc.run = M.results.allocation.data; invalidate(); }
      return;
    }
    if (m === "forward") {
      const read = { modes: ["forward"], path: "/api/ml/forward", query: { id: M.cur } };
      if (fromForm === true) return W.request(M, () => [read], "Reading the signals…");
      return W.request(M, () => [{ modes: ["forward_step"], job: true, path: "/api/ml/forward/step", body: { id: M.cur, capital: needNumber(f.capital, "Capital"), as_of: f.asOf.trim() || undefined } }, read], "Recording the signal…");
    }
  };
  M.auto = () => {
    const m = M.mode;
    if (!M.catalog) { loadCatalog(); return; }
    if (m === "experiments" || m === "training") { M.run(true); return; }
    if (["compare", "select", "combine", "explain", "overfit", "forward", "controls", "allocation"].includes(m)) {
      if (m === "allocation" && !M.alloc.cat) loadAllocationCatalog();
      if (!M.board) { ensureBoard(); return; }
      if ((m === "overfit" || m === "forward" || m === "controls") && M.cur) M.run(true);
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
    select: ["Choose among experiments", ["Choose one to twenty experiments on the left. Only their validation rows are read. A candidate must meet the data, risk, exposure, liquidity and statistical requirements before it is ranked; the ranking is by the validation net Sharpe of the pooled chronological returns (never an average of fold Sharpe ratios); a Pareto comparison shows the trade-offs, and the simplest candidate that is not meaningfully worse than the highest ranked is selected. Nothing is promoted."]],
    combine: ["Combine strategies", ["Choose two to twelve experiments on the left: ten ways to weight them, each estimated only from the returns up to the day it is used."]],
    allocation: ["Portfolio allocation", ["Choose a saved experiment and an allocation policy on the left. Equal weight stays the benchmark; the other policies weight each stock by its calibrated forecast, its risk and its correlation with the others, inside the limits you set. The saved predictions are re-used (nothing is refitted) and the result is a child experiment."]],
    forward: ["Forward test", ["Record the saved model's signal for the newest date, settle the signals whose window has passed, and watch the live rank IC and the drift of each feature against the backtest."]],
    training: ["Training monitor", ["No deep-network training has been run here yet. Train an N-BEATS, N-HiTS, LSTM, GRU, TCN or similar model in the Train view: each run is written to disk epoch by epoch, with a checkpoint, so it can be watched here and resumed after an interruption."]],
    controls: ["Research controls", ["Choose an experiment on the left: its three independent gates (Algorithm Verified, Strategy Validated, Operationally Ready) with the reason each is not passed, every try recorded in its study, what was read of the protected test and forward outcomes, and which selection tests applied."]] };
  M.empty = (root) => Q.labEmpty(root, M, EMPTY[M.mode][0], EMPTY[M.mode][1], M.runLabel[M.mode]);
  M.draw = () => Q.drawLab(M);
  M.paint = () => paintRuns();

  /* ---------- runs in the background ----------
     A training, a search or a re-backtest runs on the app's own worker, not in this page: closing the tab does not stop it, and the experiment is saved when it ends. What a reloaded page lost was the knowledge
     of it, so the server keeps a record of every job (GET /api/jobs). This card lists the runs that are going, waiting, or ended in the last hours (finished, failed, or lost when the app stopped), and
     follows them until they end. An app that predates the list answers 404 and the card stays away. */
  const ACTIVE = new Set(["queued", "running"]);
  const RUN_WORD = { queued: "Waiting", running: "Running", done: "Finished", error: "Failed", interrupted: "Stopped", cancelled: "Cancelled" };
  const RECENT_MS = 6 * 3600 * 1000;
  M.runs = { jobs: [], seen: new Map(), timer: null };
  async function watchRuns() {
    clearTimeout(M.runs.timer); M.runs.timer = null;
    if (!isOpen()) return;
    let got;
    try { got = await api("/api/jobs"); } catch (e) { M.runs.jobs = []; paintRuns(); return; }
    const mine = got.jobs.filter((j) => j.kind === "ml");
    let completed = false;
    for (const j of mine) {
      const before = M.runs.seen.get(j.id);
      if (before !== undefined && ACTIVE.has(before) && j.status === "done") completed = true;                // it ended while this page was open: the lists are out of date
      M.runs.seen.set(j.id, j.status);
    }
    M.runs.jobs = mine;
    if (completed) { invalidate(); if (M.mode === "experiments" && !M.busy) M.run(true); }
    paintRuns();
    if (mine.some((j) => ACTIVE.has(j.status))) M.runs.timer = setTimeout(watchRuns, 2000);
  }
  async function clearRuns(body) {
    try { await api("/api/jobs/clear", body); } catch (e) { Q.toast(e.message); }
    await watchRuns();
  }
  function paintRuns() {
    const root = Q.$("#ml-results"); if (!root) return;
    const old = root.querySelector(".runs"); if (old) old.remove();
    const now = Date.now(), order = { running: 0, queued: 1 };
    const jobs = M.runs.jobs.filter((j) => ACTIVE.has(j.status) || (j.finished && now - j.finished * 1000 < RECENT_MS))
      .sort((a, b) => (order[a.status] ?? 2) - (order[b.status] ?? 2) || (ACTIVE.has(a.status) ? a.created - b.created : (b.finished || 0) - (a.finished || 0))).slice(0, 6);       // what is running, then what waits in the order it was queued, then the latest endings
    if (!jobs.length) return;
    const card = h("div", "card runs", root); root.prepend(card);
    const going = jobs.some((j) => ACTIVE.has(j.status));
    h("h3", "", card, going ? "Runs in the background" : "Recent runs");
    if (going) h("p", "note", card, "These go on at the app's computer if you close this page, and the experiment is saved when a run ends. If the app itself stops (on Replit, when the workspace goes idle after you leave), the run is lost: it shows here as stopped and has to be started again from the beginning.");
    const ended = jobs.filter((j) => !ACTIVE.has(j.status)).length;
    if (ended >= 2) {
      const all = h("button", "btn small", card, "Clear all ended runs"); all.type = "button"; all.dataset.action = "clear-ended";
      all.addEventListener("click", () => clearRuns({ ended: true, kind: "ml" }));
    }
    const ul = h("ul", "runlist", card);
    for (const j of jobs) {
      const li = h("li", j.status, ul); li.dataset.job = j.id; li.dataset.status = j.status;
      const main = h("div", "main", li), head = h("span", "", main); h("b", "", head, (RUN_WORD[j.status] || j.status) + " · "); head.appendChild(document.createTextNode(String(j.label).replace(/^ml: /, "")));
      const detail = ACTIVE.has(j.status) ? `${j.message} (${Math.round(j.elapsed)} s)` : j.status === "done" ? `ended ${W.ago(new Date(j.finished * 1000).toISOString())} after ${Math.round(j.elapsed)} s` : j.status === "error" ? String(j.error || "it failed") : String(j.message || "");
      h("span", "meta", main, detail);
      if (j.status === "running" && /^ml: tune /.test(j.label)) {
        const stop = h("button", "btn small", li, "Stop"); stop.type = "button"; stop.dataset.action = "stop-run"; stop.title = "End the tuning after the trial it is on; the best so far is still run on every segment";
        stop.addEventListener("click", async () => { stop.disabled = true; try { await api("/api/ml/tune/stop", { job: j.id }); } catch (e) { Q.toast(e.message); stop.disabled = false; } });
      }
      if (j.training_run) {                                                                    // a deep network's run: its checkpoints can be read, resumed or archived from the monitor
        const mon = h("button", "btn small", li, ACTIVE.has(j.status) ? "Watch training" : "Training run"); mon.type = "button"; mon.dataset.action = "open-training"; mon.title = "Epoch-by-epoch metrics, checkpoints and, if it was interrupted, resume or restart";
        mon.addEventListener("click", () => { M.tr.id = ""; M.tr.data = null; go("training"); loadTrainingRun(j.training_run, true); });
      }
      const clear = h("button", "btn small", li, ACTIVE.has(j.status) ? "Not running? Clear" : "Clear"); clear.type = "button"; clear.dataset.action = "clear-run";
      clear.title = ACTIVE.has(j.status) ? "If the app was stopped or crashed while this ran, take it off the list. A run that really is going in this app cannot be cleared." : "Take this run off the list (a finished experiment stays in the registry).";
      clear.addEventListener("click", () => clearRuns({ id: j.id }));
      if (j.status === "done" && (j.ref || /^ml: /.test(j.label))) {                                  // a dataset build makes no experiment: nothing to open
        const open = h("button", "btn small", li, j.ref ? "Open" : "Show the experiments"); open.type = "button"; open.dataset.action = "open-run";
        open.addEventListener("click", () => { if (j.ref) openExperiment(j.ref); else { invalidate(); go("experiments"); } });
      }
    }
  }
  M.build = (root, d, body) => ({ dataset: buildDataset, train: buildTrain, experiments: buildExperiments, compare: buildCompare, explain: buildExplain, overfit: buildOverfit, search: buildSearch, select: buildSelect, combine: buildCombine, forward: buildForward, allocation: buildAllocation, controls: buildControls, training: buildTraining })[M.mode](root, d, body);
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

  /* ---------- Portfolio allocation: forecasts turned into per-stock weights from risk, correlation and limits ---------- */
  const ALLOC_FALLBACK = [["equal_weight", "Equal weight (benchmark)"], ["inverse_vol", "Inverse volatility"], ["erc", "Equal risk contribution"], ["mean_variance", "Mean-variance"], ["black_litterman", "Black-Litterman"], ["hrp", "Hierarchical risk parity (3+ assets)"],
    ["cvar", "CVaR"], ["robust_mv", "Robust mean-variance"], ["kelly", "Fractional Kelly"]];
  const DRIVER_WORD = { forecast: "the forecast", risk: "risk and correlation", constraints: "the limits", equal: "nothing: it is the equal-weight book" };
  const PRESET_FIELDS = ["alRiskAversion", "alInference", "alMinT", "alShrink", "alBaseline", "alBlend", "alMaxChange", "alTurnPen", "alMaxW"];          // what a preset sets: hidden while one is chosen, so that it cannot be overridden by a form default
  function loadAllocationCatalog() {
    if (M.alloc.loading) return;
    M.alloc.loading = true;
    api("/api/ml/allocation/catalog").then((c) => { M.alloc.cat = c; M.alloc.loading = false; if (isOpen()) Q.renderLab(M); }, (e) => { M.alloc.loading = false; M.alloc.error = e.message; });
  }
  function fxBody(f) {                                                                       // the currency settings of the backtest, or nothing: a currency is never assumed
    if (!f.alFxBase || f.alFxBase === "none") return {};
    const map = {};
    for (const part of String(f.alFxCcy).split(/[\s,;]+/).filter(Boolean)) { const [t, c] = part.split(":"); if (!t || !c) throw new Error(`Currencies are written TICKER:CCY, for example NVDA:USD, SAP.DE:EUR (not '${part}')`); map[t.toUpperCase()] = c.toUpperCase(); }
    if (!Object.keys(map).length) throw new Error("Say which currency each ticker trades in (TICKER:CCY): a currency is never assumed.");
    return { fx: { base: f.alFxBase, currencies: map, hedge: needNumber(f.alFxHedge, "Hedge ratio") / 100, cost_bps: needNumber(f.alFxCost, "Conversion cost"), hedge_cost_bps: needNumber(f.alFxHedgeCost, "Hedge cost"), missing: f.alFxMissing } };
  }
  function allocationBody(f, forCompare) {
    const preset = f.alPreset && f.alPreset !== "none" ? f.alPreset : null;
    const lim = { long_only: !!f.alLongOnly, cash_allowed: !!f.alCash, min_weight: needNumber(f.alMinW, "Minimum weight") / 100, gross_max: needNumber(f.alGross, "Gross exposure") };
    if (!preset) lim.max_weight = needNumber(f.alMaxW, "Maximum weight") / 100;
    for (const [key, name, field] of [["turnover_max", "Turnover limit", "alTurnover"], ["vol_target", "Volatility target", "alVolTarget"], ["max_sector", "Sector cap", "alSector"]]) { const v = optNumber(f[field], name); if (v !== undefined) lim[key] = v / 100; }
    const a = { forecast_kind: f.alKind, cov_method: f.alCov, cov_window: needNumber(f.alCovWindow, "Risk window"), calibrate: f.alCalibrate, calib_method: f.alCalMethod, calib_asset_min_eff: needNumber(f.alMinEff, "Independent windows per asset"),
      on_failure: f.alOnFailure, kelly_fraction: needNumber(f.alKelly, "Kelly fraction"), cost_bps: needNumber(f.alCommission, "Commission") + needNumber(f.alSlippage, "Slippage"), cov_horizon: f.alCovHorizon, lifecycle: f.alLifecycle, forecast_source: f.alSource, edge_model: f.alEdgeModel, edge_beyond_horizon: f.alBeyond,
      min_n_eff: needNumber(f.alMinNeff, "Minimum independent observations"), min_confidence: needNumber(f.alMinConf, "Minimum confidence"), sensitivity: !!f.alSens, limits: lim };
    if (preset) a.preset = preset;
    else {
      Object.assign(a, { risk_aversion: needNumber(f.alRiskAversion, "Risk aversion"), calib_inference: f.alInference, calib_min_t: needNumber(f.alMinT, "Calibration t-statistic"), uncertainty_shrink: needNumber(f.alShrink, "Uncertainty shrinkage"),
        turnover_penalty: needNumber(f.alTurnPen, "Turnover penalty") });
      if (f.alBaseline !== "none" && f.alBlend !== "none") { a.baseline = f.alBaseline; a.blend = f.alBlend; a.blend_alpha = needNumber(f.alAlpha, "Blend weight"); a.blend_max = needNumber(f.alBlendMax, "Maximum blend weight"); }
      const mc = optNumber(f.alMaxChange, "Maximum weight change"); if (mc !== undefined) a.max_weight_change = mc / 100;
    }
    const every = optNumber(f.alOptEvery, "Decide every"); if (every !== undefined) a.opt_every = every;
    const hold = needNumber(f.alHolding, "Holding period"); if (hold !== 1 || f.alLifecycle === "staggered_tranches") a.holding_days = hold;
    if (!forCompare) a.allocator = f.alAllocator;
    return a;
  }
  function allocationSections(sec, f, after) {
    const cat = M.alloc.cat, again = () => { M.results.allocation = null; M.errors.allocation = null; Q.renderLab(M); };
    if (!cat) { loadAllocationCatalog(); h("p", "hint", sec("1. The experiment"), M.alloc.error || "Loading the allocators…"); return; }
    const preset = f.alPreset && f.alPreset !== "none";
    const b1 = sec("1. The experiment");
    expSelect(b1, true);
    h("p", "hint", b1, "The saved predictions are re-used: nothing is refitted. The model must forecast an asset's own return (fwd_ret_h) or the direction of its move; a rank or relative-return forecast says nothing about how much to hold, and is refused with what to change.");
    const b2 = sec("2. What to run");
    W.seg(b2, [["allocate", "Allocate and backtest one policy"], ["compare", "Compare all allocators"], ["calibration", "Compare calibration methods"]], f.alWhat, (v) => { f.alWhat = v; Q.renderLab(M); });
    if (f.alWhat === "compare") {
      Q.labField(b2, { key: "alStab", label: "Also compare the stabilisers (conservative preset; 50% confidence-weighted blend with inverse volatility)", kind: "bool" }, f, after);
      Q.labField(b2, { key: "alProtected", label: "Also report the test and forward segments, flagged as not used to select", kind: "bool", help: "Allocators are ranked and one is selected on the validation segment only. Reporting the protected segments is a look at them and is counted as one." }, f, after);
      h("p", "hint", b2, "Every allocator gets the same forecasts, dates, limits, costs, risk windows and execution. The ranking and the multiple-testing control use the validation segment only.");
    }
    if (f.alWhat === "calibration") h("p", "hint", b2, "Pooled, asset-specific, partially pooled and regularised calibration, each from outcomes resolved before every decision, scored on the validation segment. A method is called better than the pooled baseline only when the interval of the paired difference excludes zero.");
    const b3 = sec("3. Allocation policy");
    const opts = (cat.allocators || []).map((a) => [a.key, a.label + (a.uses_forecast ? "" : " (risk only)")]);
    if (f.alWhat === "allocate") Q.labField(b3, { key: "alAllocator", label: "Allocator", kind: "choice", options: opts.length ? opts : ALLOC_FALLBACK }, f, again);
    const doc = (cat.allocators || []).find((a) => a.key === f.alAllocator);
    if (doc && f.alWhat === "allocate") {
      const p = h("p", "hint", b3); p.setAttribute("data-allocator-doc", doc.key);
      p.textContent = `${doc.objective}. Assumes: ${doc.assumptions}. Needs: ${doc.needs}. Fails when: ${doc.failure}.`;
      const u = h("p", doc.uses_forecast ? "hint" : "banner", b3); u.setAttribute("data-uses-forecast", String(!!doc.uses_forecast));
      u.textContent = doc.uses_forecast ? "This allocator reads the calibrated forecast." : "This allocator is risk-only: it does not respond to the ML forecast. It uses only the covariance of past returns.";
      if (doc.min_assets > 2) h("p", "hint", b3, `Needs at least ${doc.min_assets} assets.`);
    }
    Q.labField(b3, { key: "alKind", label: "The forecast is a", kind: "choice", options: [["return", "Return over the horizon"], ["price", "Future price"], ["probability", "Probability of a rise (needs a direction classifier)"], ["distribution", "Return distribution"]] }, f, after);
    Q.labField(b3, { key: "alCov", label: "Risk model", kind: "choice", options: (cat.risk_models || ["lw_constant_correlation"]).map((k) => [k, k.replace(/_/g, " ")]) }, f, after);
    Q.labField(b3, { key: "alCovWindow", label: "Risk window (days)", kind: "number", step: 21, min: 60, max: 1000 }, f, after);
    if (!preset) Q.labField(b3, { key: "alRiskAversion", label: "Risk aversion", kind: "number", step: 1, min: 0.01, max: 1000, help: "Higher means a smaller bet on the forecast and a book nearer the minimum-risk one. The historical default is 5." }, f, after);
    if (f.alAllocator === "kelly" || f.alWhat === "compare") Q.labField(b3, { key: "alKelly", label: "Kelly fraction", kind: "number", step: 0.05, min: 0.05, max: 1 }, f, after);
    Q.labField(b3, { key: "alOnFailure", label: "When a decision has no solution", kind: "choice", options: [["risk_only", "Risk-based book (inverse volatility)"], ["hold", "Hold the previous book"], ["equal_weight", "Equal weight"], ["cash", "Go to cash"]], help: "Never silent: every such decision is counted and recorded" }, f, after);
    const bt = sec("4. Time: horizon, holding, decisions", false);
    h("p", "hint", bt, "Four quantities are kept apart: the prediction horizon (fixed by the model's target), the holding period, how often a new decision is made, and the execution lag (set in the dataset and backtest settings).");
    Q.labField(bt, { key: "alHolding", label: "Holding period (days; 1 = use the forecast horizon, the original behaviour)", kind: "number", step: 1, min: 1, max: 60, help: "The window the optimisation is posed over and, for staggered tranches, how long each tranche is held. The forecast's edge accrues within its own horizon and is not extrapolated beyond it." }, f, after);
    Q.labField(bt, { key: "alLifecycle", label: "Overlapping decisions", kind: "choice", options: [["single_book", "One book, re-optimised at each decision (a new forecast replaces the last)"], ["staggered_tranches", "Staggered tranches, each held for the holding period"]] }, f, after);
    Q.labField(bt, { key: "alSource", label: "The forecast of the holding window is", kind: "choice", options: [["scaled", "The model's horizon forecast, scaled to the holding window (states its assumption)"], ["direct", "A direct forecast: the model was trained on the holding window itself (no scaling)"]] }, f, after);
    if (f.alSource === "scaled") {
      Q.labField(bt, { key: "alEdgeModel", label: "Scaling of a forecast to another window", kind: "choice", options: [["linear", "Linear: the return scales with the window"], ["compound", "Compound: constant daily growth"]] }, f, after);
      Q.labField(bt, { key: "alBeyond", label: "Days beyond the forecast horizon", kind: "choice", options: [["zero", "No edge is credited (conservative)"], ["persist", "The same daily rate continues (an assumption)"]] }, f, after);
    }
    Q.labField(bt, { key: "alOptEvery", label: "Decide every (days; blank = the holding period)", kind: "number", step: 1, min: 1, max: 60 }, f, after);
    Q.labField(bt, { key: "alCovHorizon", label: "Risk over the holding window", kind: "choice", options: [["linear", "Daily covariance times the window (the original rule)"], ["variance_ratio", "Adjusted for serial correlation (variance ratio)"], ["multiperiod", "Estimated directly from overlapping windows, shrunk toward the linear one"]] }, f, after);
    const bc = sec("5. Forecast calibration and evidence", false);
    Q.labField(bc, { key: "alCalibrate", label: "Calibrate the forecast on past outcomes", kind: "choice", options: [["slope", "Yes: only outcomes resolved before the decision"], ["none", "No: use it as it is (labelled uncalibrated)"]] }, f, after);
    Q.labField(bc, { key: "alCalMethod", label: "Calibration", kind: "choice", options: (cat.calibration_methods || ["pooled"]).map((k) => [k, { pooled: "Pooled slope (baseline)", asset: "A slope per asset (enough history only)", partial_pooling: "Partial pooling across assets (3+ assets)", sector_pooling: "Partial pooling by sector (needs a sector map)", regularized: "Pooled slope shrunk toward no edge" }[k] || k]) }, f, after);
    if (!preset) {
      Q.labField(bc, { key: "alInference", label: "Standard errors", kind: "choice", options: [["sqrt_horizon", "Independent errors times the square root of the horizon (original)"], ["hac", "HAC: overlapping outcomes, serial and cross-asset correlation"], ["block_bootstrap", "Block bootstrap of dates"]] }, f, after);
      Q.labField(bc, { key: "alMinT", label: "Forecast confidence: minimum calibration t-statistic", kind: "number", step: 0.5, min: 0, max: 10, help: "Below this the forecast is treated as no edge and the book is set by risk and the limits alone" }, f, after);
      Q.labField(bc, { key: "alShrink", label: "Uncertainty shrinkage (standard errors subtracted from the slope)", kind: "number", step: 0.25, min: 0, max: 5 }, f, after);
    }
    Q.labField(bc, { key: "alMinEff", label: "Independent outcome windows an asset needs for a slope of its own", kind: "number", step: 5, min: 5, max: 1000 }, f, after);
    Q.labField(bc, { key: "alMinNeff", label: "Minimum independent observations behind the calibration (0 = none)", kind: "number", step: 10, min: 0, max: 10000 }, f, after);
    Q.labField(bc, { key: "alMinConf", label: "Minimum confidence per asset (probability its slope is positive; 0 = none)", kind: "number", step: 0.05, min: 0, max: 1 }, f, after);
    const bs = sec("6. Stability: baseline, blending, weight changes", false);
    Q.labField(bs, { key: "alPreset", label: "Preset", kind: "choice", options: [["none", "None: the settings below"], ...Object.entries(cat.presets || {}).map(([k]) => [k, "Conservative (documented, not an optimum)"])] }, f, again);
    if (preset) { const d = (cat.presets || {})[f.alPreset]; const p = h("p", "hint", bs); p.setAttribute("data-preset-doc", f.alPreset); p.textContent = d ? d.description : ""; }
    else {
      Q.labField(bs, { key: "alBaseline", label: "Baseline book (the risk-based book the ML book is blended with)", kind: "choice", options: [["none", "None: the allocator's own book"], ["equal_weight", "Equal weight"], ["inverse_vol", "Inverse volatility"], ["erc", "Equal risk contribution"]] }, f, again);
      if (f.alBaseline !== "none") {
        Q.labField(bs, { key: "alBlend", label: "Blend weight on the ML book (alpha)", kind: "choice", options: [["none", "No blend"], ["fixed", "Fixed alpha"], ["confidence", "From forecast confidence"], ["validation", "From past resolved decisions only"]] }, f, again);
        if (f.alBlend === "fixed") Q.labField(bs, { key: "alAlpha", label: "alpha (0 = baseline, 1 = the ML book)", kind: "number", step: 0.05, min: 0, max: 1 }, f, after);
        if (f.alBlend !== "none" && f.alBlend !== "fixed") Q.labField(bs, { key: "alBlendMax", label: "Largest alpha allowed", kind: "number", step: 0.05, min: 0, max: 1 }, f, after);
        h("p", "hint", bs, "w_final = (1 - alpha) w_baseline + alpha w_ML. alpha uses only what was known at the decision; it is never fitted on the test or forward segments.");
      }
      Q.labField(bs, { key: "alMaxChange", label: "Largest weight change per decision, % (blank = none)", kind: "number", step: 1, min: 0.1, max: 100 }, f, after);
    }
    Q.labField(bs, { key: "alSens", label: "Compute sensitivities of the latest decision (forecast, risk aversion, costs)", kind: "bool" }, f, after);
    const b4 = sec("7. Weight limits", false);
    Q.labField(b4, { key: "alLongOnly", label: "Long only", kind: "bool" }, f, after);
    Q.labField(b4, { key: "alCash", label: "Cash allowed (weights may sum to less than the gross limit)", kind: "bool" }, f, after);
    Q.labField(b4, { key: "alMinW", label: "Minimum weight per stock, %", kind: "number", step: 1, min: 0, max: 100 }, f, after);
    if (!preset) Q.labField(b4, { key: "alMaxW", label: "Maximum weight per stock, %", kind: "number", step: 5, min: 0, max: 100 }, f, after);
    Q.labField(b4, { key: "alGross", label: "Gross exposure limit", kind: "number", step: 0.1, min: 0.1, max: 3 }, f, after);
    Q.labField(b4, { key: "alSector", label: "Sector cap, % (blank = none)", kind: "number", step: 5, min: 5, max: 100 }, f, after);
    Q.labField(b4, { key: "alTurnover", label: "Turnover limit per rebalance, % one way (blank = none)", kind: "number", step: 5, min: 0, max: 100 }, f, after);
    Q.labField(b4, { key: "alVolTarget", label: "Volatility target, % a year (blank = none)", kind: "number", step: 1, min: 1, max: 100 }, f, after);
    h("p", "hint", b4, "Limits are checked before the optimiser runs. If no book can satisfy them the decision is reported infeasible and the failure policy applies: a limit is never relaxed silently.");
    const bf = sec("9. Currency", false);
    Q.labField(bf, { key: "alFxBase", label: "Base currency (the portfolio is valued in it)", kind: "choice", options: [["none", "Not used: every price is in one currency"], ...["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "HKD", "SGD"].map((c) => [c, c])] }, f, again);
    if (f.alFxBase && f.alFxBase !== "none") {
      Q.labField(bf, { key: "alFxCcy", label: "Currency of each ticker (TICKER:CCY, separated by commas)", kind: "text", placeholder: "NVDA:USD, SAP.DE:EUR, 7203.T:JPY", help: "A currency is never assumed: every ticker of the experiment must be listed. Exchange rates come from the same market-data source as the prices (pair <currency><base>=X)." }, f, after);
      Q.labField(bf, { key: "alFxHedge", label: "Hedge ratio of foreign exposure, %", kind: "number", step: 10, min: 0, max: 100, help: "An explicit forward overlay struck when the book trades, settled daily. No forward points or financing rates are available, so it is not exchange-accurate." }, f, after);
      Q.labField(bf, { key: "alFxCost", label: "Currency conversion cost on foreign trades (bps)", kind: "number", step: 0.5, min: 0, max: 500 }, f, after);
      Q.labField(bf, { key: "alFxHedgeCost", label: "Hedge cost, bps of the notional changed", kind: "number", step: 0.5, min: 0, max: 500 }, f, after);
      Q.labField(bf, { key: "alFxMissing", label: "When an exchange rate is missing", kind: "choice", options: [["raise", "Stop and say which (recommended)"], ["exclude", "Leave that stock out on those days"]] }, f, after);
      h("p", "hint", bf, "A forecast is converted to the base currency with a random-walk exchange rate (expected change 0): an assumption, shown on the page. Only the portfolio-allocation method supports currency conversion.");
    }
    const b5 = sec("8. Trading", false);
    Q.labField(b5, { key: "alRebalance", label: "Rebalance frequency (when the engine may trade)", kind: "choice", options: [["daily", "Daily"], ["weekly", "Weekly"], ["monthly", "Monthly"]] }, f, after);
    Q.labField(b5, { key: "alCommission", label: "Commission (bps)", kind: "number", step: 0.5, min: 0, max: 250 }, f, after);
    Q.labField(b5, { key: "alSlippage", label: "Slippage (bps)", kind: "number", step: 0.5, min: 0, max: 250 }, f, after);
    if (!preset) Q.labField(b5, { key: "alTurnPen", label: "Turnover penalty in the optimiser", kind: "number", step: 0.5, min: 0, max: 100 }, f, after);
  }
  const lastRecord = (pa) => { const r = (pa.records || []).filter((x) => x.weights && Object.keys(x.weights).length && x.explain); return r.length ? r[r.length - 1] : null; };
  function buildAllocation(root, d) {
    const cmp = M.alloc.compare, cal = M.alloc.calibration, run = d && d.report ? d : (M.alloc.run && M.alloc.run.report ? M.alloc.run : null);
    if (run) allocationRun(root, run);
    else header(root, "Portfolio allocation", "Compare allocators and calibration methods on the saved forecasts");
    if (cmp) allocationCompare(root, cmp);
    if (cal) allocationCalibration(root, cal);
    if (run) reportView(root, run);
  }
  function allocationRun(root, d) {
    const r = d.report, pa = r.portfolio_allocation;
    if (!pa) { header(root, "Portfolio allocation", ""); W.flags(root, ["This run did not use the portfolio-allocation method, so it has no allocation record."], ""); return; }
    const c = pa.counts, rec = lastRecord(pa), doc = pa.documentation || {}, bt = (r.config || {}).backtest || {}, costBps = Number(bt.cost_bps || 0) + Number(bt.slippage_bps || 0);
    header(root, `Portfolio allocation: ${doc.label || pa.allocator}`, `${c.decisions} decisions · ${c.optimal} solved · settings ${pa.config_hash} · child of ${shortId(d.parent || "")}`);
    if ((r.warnings || []).length) W.flags(root, r.warnings, "Watch. ");
    if (Q.research && d.id) {
      const sv = h("div", "row", root), sb = h("button", "btn", sv, "Save to a research session");
      sb.type = "button"; sb.dataset.role = "save-allocation-study";
      sb.addEventListener("click", async () => {
        try { await Q.research.saveStudy({ experiment: d.id }, { title: `Allocation: ${doc.label || pa.allocator}`, sessionName: "Portfolio allocation" }); } catch (e) { Q.toast("Not saved: " + e.message); }
      });
    }
    const k = h("div", "earnrow", root);
    tile(k, "Decisions solved", `${c.optimal} of ${c.decisions}`, { text: c.failure_policy_used ? `${c.failure_policy_used} used the failure policy` : "no fallback used", dir: c.failure_policy_used ? "down" : "" }, "", "A decision with no solution is never hidden: it is counted here and the configured failure policy applies");
    tile(k, "Forecast had no edge", String(c.no_edge + c.insufficient_history), { text: "book set by risk and limits alone", dir: "" }, "", "The calibration slope was not distinguishable from zero (or there was too little past outcome history), so the forecast was not used as a return");
    if (rec) {
      const conc = rec.concentration || {};
      tile(k, "Expected return (window)", spct(rec.expected_return, 2), { text: "from the calibrated forecast" }, "", "The weighted calibrated expected return over the holding window, at the latest decision");
      tile(k, "Estimated risk (window)", pct(rec.forecast_vol_window, 2), { text: "from the covariance as of that date" });
      tile(k, "Turnover of the latest rebalance", pct(rec.turnover_from_previous, 1), { text: `one way · about ${sig((rec.turnover_from_previous || 0) * 2 * costBps, 1)} bps of cost` }, "", "One-way turnover times twice the commission and slippage in basis points: an estimate; the backtest charges the actual trades");
      tile(k, "Effective number of positions", Q.bad(conc.effective_n) ? "n/a" : sig(conc.effective_n, 2), { text: `largest weight ${pct(conc.max_weight, 0)}` }, "", "The inverse of the Herfindahl index of the weights: 2.00 for two equal weights, 1.00 for one stock");
    }
    if (rec) {
      const ex = rec.explain, names = rec.assets || Object.keys(rec.weights), fc = rec.forecast || {}, bl = rec.blend || {}, hz = rec.horizon || {};
      const hist0 = (pa.records || []).filter((x) => x.weights && Object.keys(x.weights).length), prev = hist0.length > 1 ? hist0[hist0.length - 2] : null;
      const held = rec.held_weights || rec.weights;
      const cw = W.card(root, `Why each stock got its weight (decision of ${rec.date})`, `The weights are split exactly into the equal-weight book, what risk and correlation changed, and what the forecast changed. ${DRIVER_WORD[ex.driver] ? "At this decision the weights are mainly set by " + DRIVER_WORD[ex.driver] + "." : ""}`);
      cw.setAttribute("data-allocation-explain", ex.driver);
      const cols = [{ label: "Stock", get: (a) => a }, { label: "Raw forecast", num: true, get: (a) => spct((rec.mu_raw || {})[a], 2) }, { label: "Calibration slope", num: true, get: (a) => { const s = (fc.slopes || {})[a]; return Q.bad(s === undefined ? fc.slope : s) ? "n/a" : sig(s === undefined ? fc.slope : s, 2); } },
        { label: "Forecast confidence", num: true, get: (a) => { const x = (fc.confidence || {})[a]; return x === undefined || x === null ? (fc.status === "calibrated" ? "n/a" : "no edge") : pct(x, 0); } },
        { label: "Calibrated forecast", num: true, get: (a) => spct((rec.mu || {})[a], 2) }, { label: "Forecast uncertainty (sd)", num: true, get: (a) => (Q.bad((rec.sd || {})[a]) ? "n/a" : pct(rec.sd[a], 2)) },
        { label: "Volatility (a year)", num: true, get: (a) => pct(((rec.volatility_daily || {})[a] || 0) * Math.sqrt(252), 1) }, { label: "Equal weight", num: true, get: (a) => pct(ex.equal[a], 1) }];
      if (bl.status === "blended") cols.push({ label: "Baseline book", num: true, get: (a) => pct((bl.baseline_weights || {})[a], 1) }, { label: "ML-only book", num: true, get: (a) => pct((bl.ml_weights || {})[a], 1) });
      cols.push({ label: "From risk", num: true, get: (a) => spct(ex.risk_effect[a], 1) }, { label: "From forecast", num: true, get: (a) => spct(ex.forecast_effect[a], 1) }, { label: bl.status === "blended" ? "Final (blended)" : "Weight", num: true, get: (a) => pct(rec.weights[a], 1) },
        { label: "Change from the previous decision", num: true, get: (a) => (prev && prev.weights ? spct(rec.weights[a] - ((prev.weights || {})[a] || 0), 1) : "first book") },
        { label: "Share of risk", num: true, get: (a) => (rec.risk_contributions && rec.risk_contributions[a] && !Q.bad(rec.risk_contributions[a].share) ? pct(rec.risk_contributions[a].share, 1) : "n/a") });
      table(cw, cols, names);
      const why = [];
      if (fc.status && fc.status !== "calibrated") why.push(`The forecast is ${String(fc.status).replace(/_/g, " ")}${fc.note ? ": " + fc.note : ""}, so it did not move the weights.`);
      else if (fc.status === "calibrated") why.push(`The forecast is calibrated on ${fc.n_obs} past outcomes worth ${sig(fc.n_eff, 0)} independent observations (slope ${sig(fc.slope, 2)}, t ${sig(fc.t_stat, 1)}${fc.method && fc.method !== "pooled" ? ", " + String(fc.method).replace(/_/g, " ") + " calibration" : ""}): it is shrunk by the slope before it is used, so it moves the book by ${pct(ex.forecast_share, 0)} of the total change from equal weight.`);
      if (fc.fallback) why.push("Calibration fell back to the pooled slope: " + fc.fallback + ".");
      if ((ex.binding || []).length) why.push("Binding limits: " + ex.binding.join("; ") + ".");
      if (rec.weight_change_limited) why.push(`The limit of ${pct(rec.weight_change_limited.limit, 0)} on weight changes restricted the move.`);
      if (rec.turnover_limited) why.push("The turnover limit restricted the move.");
      if (!why.length) why.push("No limit is binding.");
      W.hint(cw, why.join(" "));
      if (rec.narrative) {
        const cn = W.card(root, "In words", rec.narrative.book || "");
        cn.setAttribute("data-allocation-narrative", "1");
        const ul = h("ul", "plain", cn);
        for (const a of names) h("li", "", ul, rec.narrative.per_asset[a] || "");
      }
      if (bl.status === "blended") W.hint(cw, `Blended: w_final = (1 − α) w_${String(bl.baseline).replace(/_/g, " ")} + α w_ML with α = ${sig(bl.alpha, 2)} (${bl.mode}${bl.note ? ": " + bl.note : ""}).${bl.projection ? " The blend broke a limit and was projected onto the feasible set (moved " + pct(bl.projection.l1_deviation, 1) + " in total)." : ""}`);
      const hh = W.card(root, "Horizon, holding and decisions", "Four separate quantities: the horizon the model forecasts, how long a position is meant to be held, how often a decision is made, and the execution lag.");
      table(hh, [{ label: "Quantity", get: (x) => x[0] }, { label: "Value", get: (x) => x[1] }], [["Prediction horizon", `${hz.forecast_days || pa.holding_window_days} days`], ["Holding window", `${hz.holding_days || pa.holding_window_days} days${hz.edge_scale !== undefined ? ` (the edge is scaled by ${sig(hz.edge_scale, 2)}: it accrues within the horizon, not beyond it)` : ""}`],
        ["Decision every", `${hz.decision_every || pa.opt_every} days`], ["Execution lag", `${hz.lag ?? pa.lag} day(s)`], ["Overlapping decisions", hz.lifecycle === "staggered_tranches" ? `staggered tranches: ${hz.tranches} alive at once, each held ${hz.holding_days} days` : "one book; each new decision replaces the last"],
        ["Risk over the window", ((hz.covariance || {}).method || "linear") + ((hz.covariance || {}).fallback ? ": " + hz.covariance.fallback : "")],
        ["Modelling assumptions", (hz.assumptions || []).join(" · ") || "none beyond the daily covariance times the window"]]);
      const covs = rec.covariance_window, cor = rec.correlation;
      if (cor && names.length > 1) {
        const cr = W.card(root, "Covariance and correlation used", "Estimated only from returns before the decision date, over the holding window. Correlation is what makes two stocks one bet; volatility is what each adds on its own.");
        table(cr, [{ label: "", get: (a) => a }, ...names.map((b) => ({ label: b, num: true, get: (a) => sig(cor[a][b], 2) }))], names);
        table(cr, [{ label: "Window volatility", get: (a) => a }, { label: "Std dev over the window", num: true, get: (a) => pct(Math.sqrt(covs[a][a]), 2) }, { label: "Annualised", num: true, get: (a) => pct(Math.sqrt(covs[a][a] * 252 / (hz.holding_days || pa.holding_window_days)), 1) }], names);
      }
      const hist = hist0, assets = names, step = Math.max(1, Math.floor(hist.length / 60));
      if (hist.length > 1) {
        const ch = h("div", "charts", root);
        chartCard(ch, "Weights through time", `The target weight of each stock at every decision${pa.records_sampled ? " (thinned to 400 of " + pa.records_total + ")" : ""}. Equal weight would be a flat line at ${pct(1 / assets.length, 0)}.`, { wide: true,
          draw: (b) => C.line(b, { dates: hist.map((x) => x.date), series: assets.map((a, i) => ({ name: a, color: expColor(i), values: hist.map((x) => ((x.held_weights || x.weights)[a] === undefined ? null : (x.held_weights || x.weights)[a])) })), baseline: 0, yFormat: pctTick, tipFormat: (v) => pct(v, 1), height: 240, label: "Target weights through time" }),
          table: (b) => table(b, [{ label: "Date", get: (x) => x.date }, ...assets.map((a) => ({ label: a, num: true, get: (x) => (x.weights[a] === undefined ? "n/a" : pct(x.weights[a], 1)) }))], hist.filter((_, i) => i % step === 0)) });
        const turn = hist.map((x) => x.turnover_from_previous), mean = turn.reduce((s2, v) => s2 + (v || 0), 0) / turn.length, big = turn.filter((v) => v > 0.01).length / turn.length;
        chartCard(ch, "Turnover and cost of each decision", `One-way turnover of the target book from the previous decision, and the cost it implies at ${sig(costBps, 1)} bps each way. Mean ${pct(mean, 1)} per decision; ${pct(big, 0)} of decisions moved the book by more than one point (allocation stability).`, {
          draw: (b) => C.line(b, { dates: hist.map((x) => x.date), series: [{ name: "Turnover (one way)", color: css("--s2"), values: turn }], baseline: 0, yFormat: pctTick, tipFormat: (v) => pct(v, 1), height: 200, label: "Turnover by decision" }),
          table: (b) => table(b, [{ label: "Date", get: (x) => x.date }, { label: "Turnover", num: true, get: (x) => pct(x.turnover_from_previous, 1) }, { label: "Estimated cost, bps", num: true, get: (x) => sig((x.turnover_from_previous || 0) * 2 * costBps, 1) }], hist.filter((_, i) => i % step === 0)) });
        chartCard(ch, "Risk contribution at the latest decision", "Each stock's share of the portfolio variance. A book that is 50/50 in weight is rarely 50/50 in risk.", {
          draw: (b) => C.hbars(b, { rows: assets.map((a) => ({ label: a, value: ((rec.risk_contributions || {})[a] || {}).share || 0, tip: [["Weight", pct(rec.weights[a], 1)]] })), format: (v) => pct(v, 0), name: "Share of risk", label: "Risk contributions", labelWidth: 70 }),
          table: (b) => table(b, [{ label: "Stock", get: (a) => a }, { label: "Weight", num: true, get: (a) => pct(rec.weights[a], 1) }, { label: "Share of risk", num: true, get: (a) => pct(((rec.risk_contributions || {})[a] || {}).share, 1) }], assets) });
      }
      if (rec.sensitivity && rec.sensitivity.base) {
        const sn = rec.sensitivity, cs = W.card(root, "How sensitive is this book?", "The latest decision recomputed under perturbed forecasts, risk aversion and trading costs. A book that moves a lot under small changes is amplifying noise.");
        cs.setAttribute("data-allocation-sensitivity", "1");
        if (sn.forecast && sn.forecast.available) table(cs, [{ label: "Forecast perturbed within one standard error of the slope", get: (a) => a }, { label: "Base", num: true, get: (a) => pct(sn.base[a], 1) }, { label: "Mean", num: true, get: (a) => pct(sn.forecast.weight_mean[a], 1) }, { label: "Std dev", num: true, get: (a) => pct(sn.forecast.weight_sd[a], 1) },
          { label: "Min", num: true, get: (a) => pct(sn.forecast.weight_min[a], 1) }, { label: "Max", num: true, get: (a) => pct(sn.forecast.weight_max[a], 1) }], names);
        else W.hint(cs, (sn.forecast || {}).note || "The forecast has no calibrated slope to perturb.");
        table(cs, [{ label: "Risk aversion", num: true, get: (x) => sig(x.risk_aversion, 2) }, ...names.map((a) => ({ label: a, num: true, get: (x) => (x.weights ? pct(x.weights[a], 1) : "n/a") })), { label: "Effective positions", num: true, get: (x) => (x.concentration ? sig(x.concentration.effective_n, 2) : "n/a") }], sn.risk_aversion);
        table(cs, [{ label: "Commission + slippage (bps)", num: true, get: (x) => sig(x.cost_bps, 1) }, { label: "Turnover penalty", num: true, get: (x) => sig(x.turnover_penalty, 1) }, ...names.map((a) => ({ label: a, num: true, get: (x) => (x.weights ? pct(x.weights[a], 1) : "n/a") })), { label: "Turnover from the previous book", num: true, get: (x) => (Q.bad(x.turnover) ? "n/a" : pct(x.turnover, 1)) }], sn.costs);
      }
    }
    if (r.fx) allocationFx(root, r.fx);
    const cd = W.card(root, "The allocator", `${doc.label || ""}`);
    table(cd, [{ label: "Objective", get: (x) => x[0] }, { label: "Detail", get: (x) => x[1] }], [["Objective", doc.objective], ["Assumes", doc.assumptions], ["Needs", doc.needs], ["Fails when", doc.failure], ["Reads the forecast", doc.uses_forecast ? "yes" : "no: risk-only"],
      ["Risk model", `${(pa.config || {}).cov_method} over ${(pa.config || {}).cov_window} days (only observations before each decision)`], ["Failure policy", (pa.config || {}).on_failure], ["Limits", JSON.stringify((pa.config || {}).limits)]]);
  }
  function allocationFx(root, x) {
    const c = W.card(root, `Currency and FX (base ${x.base})`, `Quotation: ${x.quote}. ${x.identity}.`);
    c.setAttribute("data-allocation-fx", x.base);
    if (!x.exchange_accurate) W.flags(root, [`Not exchange-accurate: ${(x.not_exchange_accurate_because || []).join("; ")}.`], "Assumption. ");
    const ccys = Object.keys(x.exposure.latest);
    table(c, [{ label: "Currency", get: (k) => k }, { label: "Exposure now", num: true, get: (k) => pct(x.exposure.latest[k], 1) }, { label: "Mean exposure", num: true, get: (k) => pct(x.exposure.mean[k], 1) },
      { label: "Hedge notional now", num: true, get: (k) => (x.hedge.notional_latest[k] === undefined ? "—" : pct(x.hedge.notional_latest[k], 1)) }, { label: "Unhedged now", num: true, get: (k) => (x.hedge.residual_exposure_latest[k] === undefined ? "—" : pct(x.hedge.residual_exposure_latest[k], 1)) },
      { label: "Local-currency return (sum)", num: true, get: (k) => spct(x.attribution.by_currency_local[k], 2) }, { label: "Currency return (sum)", num: true, get: (k) => spct(x.attribution.by_currency_fx[k], 2) }], ccys);
    table(c, [{ label: "Item", get: (r2) => r2[0] }, { label: "Value", get: (r2) => r2[1] }], [["Forecast conversion", `${x.forecast_conversion.assumption}: ${x.forecast_conversion.note}`], ["Hedge", x.hedge.active ? `${x.hedge.instrument}; carry ${x.hedge.carry}` : "none"],
      ["Hedge P&L / cost", `${spct(x.hedge.pnl_total, 2)} / ${pct(x.hedge.cost_total, 3)}`], ["Currency conversion cost", `${pct(x.costs.conversion_cost_total, 3)} at ${sig(x.costs.conversion_cost_bps, 1)} bps (inside the backtest's costs)`],
      ["Hedge ratio", Object.entries(x.hedge_ratio).map(([a, v]) => `${a} ${pct(v, 0)}`).join(", ") || "none"], ["Sources", Object.values((x.data || {}).series || {}).map((s2) => `${s2.symbol} (${s2.provider}, ${s2.first} to ${s2.last}, ${s2.observations} days)`).join("; ") || "none"]]);
    table(c, [{ label: "Currency", get: (k) => k }, { label: "Usable days", num: true, get: (k) => `${x.data.coverage[k].usable} of ${x.data.coverage[k].days}` }, { label: "Coverage", num: true, get: (k) => pct(x.data.coverage[k].coverage, 1) },
      { label: "Longest gap (days)", num: true, get: (k) => String(x.data.coverage[k].longest_gap_days) }, { label: "First usable", get: (k) => x.data.coverage[k].first || "none" }, { label: "Last usable", get: (k) => x.data.coverage[k].last || "none" }], Object.keys(x.data.coverage || {}));
    for (const m2 of Object.entries(((x.data || {}).missing || {}).assets || {})) W.hint(c, `${m2[0]} (${m2[1].currency}): ${m2[1].days_missing_after_first_rate} day(s) with no usable rate after the first one, ${m2[1].days_before_first_rate} before it (policy: ${x.data.missing.policy}).`);
    for (const a of (x.data || {}).assumptions || []) W.hint(c, a);
    const ex = x.exposure.series;
    if (ex.dates.length > 1) { const ch = h("div", "charts", root); chartCard(ch, "Currency exposure through time", "Share of the book in each currency (before the hedge).", { draw: (b) => C.line(b, { dates: ex.dates, series: ccys.map((k, i) => ({ name: k, color: expColor(i), values: ex[k] })), baseline: 0, yFormat: pctTick, tipFormat: (v) => pct(v, 1), height: 200, label: "Currency exposure" }), table: (b) => table(b, [{ label: "Date", get: (i) => ex.dates[i] }, ...ccys.map((k) => ({ label: k, num: true, get: (i) => pct(ex[k][i], 1) }))], ex.dates.map((_, i) => i).filter((i) => i % 10 === 0)) }); }
  }
  function allocationCompare(root, o) {
    const sel = o.select_on, rows = o.allocators.concat(o.table.existing_rule ? ["existing_rule"] : []).filter((k, i, arr) => arr.indexOf(k) === i && o.table[k]);
    const c = W.card(root, `Allocators on identical forecasts (${SEG_NAME[sel] || sel} segment)`, `${o.n_allocators} policies, same forecasts, dates, limits, costs, risk windows and execution. Selected on the ${sel} segment only: ${o.selected ? (o.table[o.selected] || {}).label : "none eligible"}. ${o.deflated_sharpe_of_selected && o.deflated_sharpe_of_selected.probability !== undefined ? "Probability its Sharpe beats the best of " + o.deflated_sharpe_of_selected.n_trials + " trials by chance: " + pct(o.deflated_sharpe_of_selected.probability, 0) + "." : ""}`);
    c.setAttribute("data-allocator-comparison", "1");
    const seg = (k, s) => (o.table[k].segments[s] || {});
    const ci = (k) => { const x = ((o.sharpe_ci || {})[k] || {}).hac; return x && x.interval ? `[${sig(x.interval[0], 2)}, ${sig(x.interval[1], 2)}]` : "—"; };
    const cols = [{ label: "Allocator", get: (k) => o.table[k].label + (k === "equal_weight" ? " (benchmark)" : "") + (k === o.selected ? " ★" : "") }, { label: "Net return", num: true, get: (k) => spct(seg(k, sel).net_return, 1) }, { label: "Sharpe", num: true, get: (k) => sig(seg(k, sel).sharpe) },
      { label: "Sharpe 95% interval (HAC)", num: true, get: ci }, { label: "Sortino", num: true, get: (k) => sig(seg(k, sel).sortino) },
      { label: "Max drawdown", num: true, get: (k) => pct(seg(k, sel).max_drawdown, 1) }, { label: "Turnover / yr", num: true, get: (k) => sig(seg(k, sel).turnover_per_year, 1) }, { label: "Cost bps / yr", num: true, get: (k) => sig(seg(k, sel).cost_bps_per_year, 0) },
      { label: "Effective N", num: true, get: (k) => sig(seg(k, sel).effective_n, 2) }, { label: "Max weight", num: true, get: (k) => pct(seg(k, sel).max_weight, 0) }, { label: "Mean weight change", num: true, get: (k) => pct(seg(k, sel).mean_change_per_decision, 1) },
      { label: "vs equal weight, daily mean (95%)", num: true, get: (k) => { const p = (o.paired_vs_equal_weight || {})[k]; return p && p.interval ? `${spct(p.mean_daily_difference, 3)} [${spct(p.interval[0], 3)}, ${spct(p.interval[1], 3)}]` : "—"; } },
      { label: "Sharpe vs equal weight (95%)", num: true, get: (k) => { const p = (o.paired_sharpe_vs_equal_weight || {})[k]; return p && p.interval ? `${sig(p.difference, 2)} [${sig(p.interval[0], 2)}, ${sig(p.interval[1], 2)}]` : "—"; } },
      { label: "Holm p", num: true, get: (k) => { const p = (o.paired_vs_equal_weight || {})[k]; return p && !Q.bad(p.p_value_holm) ? sig(p.p_value_holm, 3) : "—"; } }];
    table(c, cols, rows);
    for (const [k, why] of Object.entries(o.skipped || {})) W.hint(c, `${o.table[k] ? o.table[k].label : k} was not run: ${why}`);
    for (const n of o.notes || []) W.hint(c, n);
    W.hint(c, "Equal weight is the benchmark and is not expected to lose: when an interval includes zero the allocators cannot be told apart on this sample.");
    for (const s of o.evaluated_segments || []) {
      const cp = W.card(root, `${SEG_NAME[s] || s} segment: reported for every allocator, not used to select`, "These outcomes were not used to rank or choose. Reading them is a look at a protected segment and counts as one.");
      table(cp, [{ label: "Allocator", get: (k) => o.table[k].label }, { label: "Net return", num: true, get: (k) => spct(seg(k, s).net_return, 1) }, { label: "Sharpe", num: true, get: (k) => sig(seg(k, s).sharpe) }, { label: "Sortino", num: true, get: (k) => sig(seg(k, s).sortino) },
        { label: "Max drawdown", num: true, get: (k) => pct(seg(k, s).max_drawdown, 1) }, { label: "Turnover / yr", num: true, get: (k) => sig(seg(k, s).turnover_per_year, 1) }], rows);
    }
    const eq = (o.curves.equity || {})[sel] || {}, keys = Object.keys(eq);
    if (keys.length) {
      const ch = h("div", "charts", root), first = eq[keys[0]];
      chartCard(ch, `Net equity by allocator (${SEG_NAME[sel] || sel} segment, rebased to 1)`, "Each line is the same forecasts through a different allocation rule and the same backtest. The grey line is the daily-rebalanced equal-weight benchmark.", { wide: true,
        draw: (b) => C.line(b, { dates: first.dates, series: keys.map((k, i) => ({ name: k === "_equal_weight_daily_rebalanced" ? "Equal weight (daily)" : (o.table[k] || {}).label || k, color: k === "_equal_weight_daily_rebalanced" ? css("--muted") : expColor(i), values: (eq[k].values.length === first.values.length ? eq[k].values : first.values.map(() => null)) })), baseline: 1, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 280, label: "Net equity by allocator" }),
        table: (b) => table(b, [{ label: "Allocator", get: (k) => (o.table[k] || {}).label || k }, { label: "Final value", num: true, get: (k) => sig(eq[k].values[eq[k].values.length - 1], 3) }], keys) });
    }
    const pq = (o.prediction_quality || {})[sel];
    if (pq) {
      const cq = W.card(root, "Forecast quality, separate from portfolio quality", "How well the forecasts predicted what followed, on the same segment. A forecast can be good and the portfolio built on it poor (or the reverse): the table above is the portfolio, this is the forecast.");
      table(cq, [{ label: "Measure", get: (x) => x[0] }, { label: "Value", num: true, get: (x) => x[1] }], [["Forecasts with a known outcome", String(pq.n)], ["Pooled correlation of forecast and outcome", sig(pq.pooled_correlation, 3)], ["Share with the right sign", Q.bad(pq.hit_rate) ? "n/a" : pct(pq.hit_rate, 1)],
        ["RMSE of the forecast", pct(pq.rmse, 2)], ["RMSE of a zero forecast", pct(pq.rmse_zero_forecast, 2)], ["Skill against a zero forecast", spct(pq.skill_vs_zero, 1)], ["Cross-sectional rank IC", pq.cross_sectional_rank_ic === null || pq.cross_sectional_rank_ic === undefined ? "not estimated" : sig(pq.cross_sectional_rank_ic, 3)]]);
      if (pq.cross_sectional_note) W.hint(cq, pq.cross_sectional_note);
    }
  }
  function allocationCalibration(root, o) {
    const c = W.card(root, `Calibration methods, out of sample (${SEG_NAME[o.select_on] || o.select_on} segment)`, `${o.decision_dates} decision dates, every ${o.step}th; each method used only outcomes resolved before the decision. ${o.note}`);
    c.setAttribute("data-calibration-comparison", "1");
    const keys = Object.keys(o.variants);
    table(c, [{ label: "Method", get: (k) => k + (k === o.baseline ? " (baseline)" : "") }, { label: "Squared error", num: true, get: (k) => (o.variants[k].mse === undefined ? "n/a" : String(+o.variants[k].mse.toPrecision(3))) }, { label: "Skill vs a zero forecast", num: true, get: (k) => spct(o.variants[k].skill_vs_zero, 1) },
      { label: "Correlation", num: true, get: (k) => sig(o.variants[k].correlation, 3) }, { label: "Used the pooled fallback", num: true, get: (k) => pct(o.variants[k].share_with_pooled_fallback, 0) }, { label: "Independent windows", num: true, get: (k) => sig(o.variants[k].mean_independent_windows, 0) },
      { label: "Against the baseline", get: (k) => (o.variants[k].versus_baseline ? o.variants[k].versus_baseline.verdict : k === o.baseline ? "the reference" : "—") }], keys);
    W.hint(c, "No method is called better unless the interval of the paired difference in squared error excludes zero; with two stocks asset-specific and hierarchical calibration fall back to the pooled slope.");
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
    const vals = series.flatMap((s) => s.values).filter((v) => v !== null && isFinite(v) && v > 0), log = vals.length > 0 && Math.max(...vals) / Math.min(...vals) > 5;      // decades of compounding read on a log scale: equal distances are equal percentage changes
    C.line(b, { dates: t.dates.concat(e.dates), series, regions: regions(e, true), baseline: 1, log, yFormat: (v) => v.toFixed(!log ? 2 : v >= 10 ? 0 : 1), tipFormat: (v) => v.toFixed(3), height: 300, label: "Equity curve by segment" + (log ? " (log scale)" : ""), endLabels: false });
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
    if (perm.length || imp.permutation_status) window.QLImportance.card(ch, perm, imp.permutation_status);                    // src/webapp/static/importance.js: diverging bars, top N, intervals, and an explanation instead of an empty axis
  }
  /* ---------- what a model reports beside its scores (r.model_extras) ---------- */
  const COLORS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];
  const padTo = (a, n) => a.concat(Array(Math.max(0, n - a.length)).fill(null));
  function modelCards(ch, r) {
    const x = r.model_extras || {};
    if (x.training) {
      const t = x.training, longest = Math.max(...t.folds.map((f) => f.train_loss.length)), xs = Array.from({ length: longest }, (_, i) => i + 1), series = [];
      t.folds.forEach((f, i) => {
        const tag = t.folds.length > 1 ? ` (fit ${f.fold + 1})` : "";
        series.push({ name: "Training loss" + tag, color: css(COLORS[i % 6]), values: padTo(f.train_loss, longest), width: 1.5 });
        series.push({ name: "Validation loss" + tag, color: css(COLORS[i % 6]), values: padTo(f.val_loss, longest), dash: "4 3" });
      });
      chartCard(ch, "Training curves", `The loss on the rows the network trains on (solid) and on the later validation tail it stops on (dashed), epoch by epoch, for the first, middle and last refit. Validation that turns up while training keeps falling is overfitting, and the best epoch is the one kept. ${t.n_fits} fits, a median of ${sig(t.epochs_median, 0)} epochs, ${pct(t.stopped_early_share, 0)} stopped early.`, {
        draw: (b) => C.line(b, { xs, series, xLabel: "epoch", xFormat: (v) => String(Math.round(v)), yFormat: (v) => String(+v.toPrecision(2)), tipFormat: (v) => String(+v.toPrecision(3)), height: 260, label: "Training and validation loss by epoch" }),
        table: (b) => table(b, [{ label: "Epoch", get: (n) => String(xs[n]) }, ...series.map((s) => ({ label: s.name, num: true, get: (n) => (Q.bad(s.values[n]) ? "" : num(s.values[n], 5)) }))], xs.map((_, n) => n)) });
    }
    if (x.ranking) {
      const k = x.ranking, sr = k.series;
      chartCard(ch, "Ranking quality", `NDCG@${k.k}: how close the daily ordering of the assets comes to the best possible, gains from the realised return fifths, out of sample. ${sig(k.ndcg, 3)} against ${sig(k.ndcg_random, 3)} for a random ordering; ${pct(k.precision_top, 0)} of the ${k.k} top-scored assets landed in the best two fifths (${pct(k.precision_random, 0)} by chance). The line is the 63-day average.`, {
        draw: (b) => C.line(b, { dates: sr.dates, series: [{ name: `NDCG@${k.k}, 63-day average`, color: css("--s1"), values: sr.ndcg }], baseline: sr.random, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 230, label: "Rolling NDCG against a random ordering" }),
        table: (b) => table(b, [{ label: "Segment", get: (s) => SEG_NAME[s] || s }, { label: "NDCG", num: true, get: (s) => sig(k.by_segment[s].ndcg, 3) }, { label: "Top-scored in the best two fifths", num: true, get: (s) => pct(k.by_segment[s].precision, 0) },
          { label: "Days", num: true, get: (s) => String(k.by_segment[s].n_days) }], Object.keys(k.by_segment || {})) });
    }
    if (x.intervals) {
      const iv = x.intervals, names = Object.keys(iv.series), pick = { asset: names[0] };
      const card = chartCard(ch, "Prediction intervals", `The model's ${pct(iv.nominal, 0)} interval around its forecast of ${r.dataset.target}, with what happened, out of sample. The realised value fell inside the interval ${pct(iv.coverage, 0)} of the time (${pct(iv.nominal, 0)} if the intervals were calibrated); the average width is ${spct(iv.mean_width, 2)}.`, {
        draw: (b) => { const s = iv.series[pick.asset]; C.line(b, { dates: s.dates, series: [{ name: "Forecast", color: css("--s1"), values: s.mean }, { name: "Realised", color: css("--muted"), values: s.actual, width: 1 }],
          bands: [{ lo: s.lower, hi: s.upper, color: css("--s1"), opacity: 0.18, name: `${pct(iv.nominal, 0)} interval` }], baseline: 0, yFormat: pctTick, tipFormat: (v) => spct(v, 2), height: 260, label: `Forecast interval and outcome for ${pick.asset}` }); },
        table: (b) => { const s = iv.series[pick.asset]; table(b, [{ label: "Date", get: (n) => s.dates[n] }, { label: "Lower", num: true, get: (n) => spct(s.lower[n], 2) }, { label: "Forecast", num: true, get: (n) => spct(s.mean[n], 2) },
          { label: "Upper", num: true, get: (n) => spct(s.upper[n], 2) }, { label: "Realised", num: true, get: (n) => spct(s.actual[n], 2) }], s.dates.map((_, n) => n)); } });
      const sel = h("select", "", card._extra); sel.setAttribute("aria-label", "asset"); names.forEach((a) => { const o = h("option", "", sel, a); o.value = a; });
      sel.addEventListener("change", () => { pick.asset = sel.value; card._draw(); });
      const u = iv.by_uncertainty;
      chartCard(ch, "Does the uncertainty mean anything?", `The predictions cut into fifths by the model's own predicted spread, lowest to highest: the realised error should grow with it. Coverage by fifth: ${u.map((v) => pct(v.coverage, 0)).join(", ")}.`, {
        draw: (b) => C.columns(b, { labels: u.map((v) => "Q" + v.bucket), values: u.map((v) => v.rmse), yFormat: pctTick, tipFormat: (v) => spct(v, 2), name: "Root mean squared error", height: 220, label: "Error by predicted uncertainty", tipLabel: (i) => `Fifth ${u[i].bucket}: predicted spread ${spct(u[i].mean_std, 2)}` }),
        table: (b) => table(b, [{ label: "Fifth", get: (v) => String(v.bucket) }, { label: "Predicted spread", num: true, get: (v) => spct(v.mean_std, 2) }, { label: "Realised error", num: true, get: (v) => spct(v.rmse, 2) }, { label: "Coverage", num: true, get: (v) => pct(v.coverage, 0) }], u) });
    }
    if (x.regimes) {
      const g = x.regimes, cols = g.columns.map((_, i) => i);
      chartCard(ch, "Regime probabilities", `The probability of each regime on each day, FILTERED: from the market's data up to that day only, never smoothed with the days after it. State 0 is the calmest. Time in each regime: ${g.share.map((v, i) => `${i}: ${pct(v, 0)}`).join(", ")}${g.mean_run_days ? `; a stretch lasts ${sig(g.mean_run_days, 0)} days on average` : ""}.`, {
        draw: (b) => C.line(b, { dates: g.dates, series: cols.map((i) => ({ name: `Regime ${i}`, color: css(COLORS[i % 6]), values: g.probabilities.map((row) => row[i]) })), yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), height: 240, include: 1, label: "Regime probabilities through time" }),
        table: (b) => table(b, [{ label: "Regime", get: (i) => String(i) }, { label: "Time in it", num: true, get: (i) => pct(g.share[i], 0) }, { label: "Market return a year", num: true, get: (i) => (g.regimes && g.regimes[i] ? spct(g.regimes[i].ann_return, 1) : "") },
          { label: "Volatility", num: true, get: (i) => (g.regimes && g.regimes[i] && !Q.bad(g.regimes[i].ann_vol) ? pct(g.regimes[i].ann_vol, 1) : "") }], cols) });
    }
    if (x.heads) {
      const hz = x.heads.horizons;
      chartCard(ch, "Rank IC by forecast horizon", "The network has one output for each horizon; each is scored against the return over its own horizon. The one for the dataset's horizon is the score that is traded.", {
        draw: (b) => C.columns(b, { labels: hz.map((v) => v.horizon + "d"), values: hz.map((v) => v.mean_ic), yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), name: "Mean rank IC", height: 220, label: "Rank IC by forecast horizon", colors: (v, i) => (hz[i].primary ? css("--s1") : css("--muted")) }),
        table: (b) => table(b, [{ label: "Horizon", get: (v) => v.horizon + " days" + (v.primary ? " (traded)" : "") }, { label: "Mean rank IC", num: true, get: (v) => num(v.mean_ic, 4) }, { label: "t-statistic", num: true, get: (v) => sig(v.t, 2) }, { label: "Days", num: true, get: (v) => String(v.n_days) }], hz) });
    }
    if (x.weights) {
      const w = x.weights, cols = w.members.map((_, i) => i);
      chartCard(ch, "Ensemble weights", `The weight of each member on each day (${w.weighting} weighting). A weight on a day uses only outcomes already known that day. Average weights: ${w.members.map((m) => `${m} ${pct(w.mean[m], 0)}`).join(", ")}.`, {
        draw: (b) => C.line(b, { dates: w.dates, series: cols.map((i) => ({ name: w.members[i], color: css(COLORS[i % 6]), values: w.weights.map((row) => row[i]) })), yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), height: 240, include: 0, label: "Ensemble member weights through time" }),
        table: (b) => table(b, [{ label: "Member", get: (i) => w.members[i] }, { label: "Average weight", num: true, get: (i) => pct(w.mean[w.members[i]], 1) }, { label: "Slope", num: true, get: (i) => (w.calibration ? sig(w.calibration[i].slope, 3) : "") },
          { label: "Residual spread", num: true, get: (i) => (w.calibration ? spct(w.calibration[i].residual_std, 2) : "") }, { label: "Held-out rank IC", num: true, get: (i) => (w.calibration ? sig(w.calibration[i].heldout_ic, 3) : "") }], cols) });
    }
    if (x.latent) {
      const f = x.latent.factors;
      chartCard(ch, "What each latent factor predicts", "The autoencoder learned these codes without seeing returns. Each bar is a code's own rank IC with the forward return, out of sample: the factors that carry signal, and their sign.", {
        draw: (b) => C.hbars(b, { rows: f.map((v) => ({ label: v.factor, value: v.mean_ic || 0, tip: [["Spread of the code", sct(v.std)], ["Days", String(v.n_days)]] })), format: (v) => v.toFixed(3), name: "Rank IC", label: "Rank IC of each latent factor" }),
        table: (b) => table(b, [{ label: "Factor", get: (v) => v.factor }, { label: "Rank IC", num: true, get: (v) => num(v.mean_ic, 4) }, { label: "Spread", num: true, get: (v) => sct(v.std) }], f) });
    }
    if (x.portfolio) {
      const p = x.portfolio, s = p.series;
      chartCard(ch, "Portfolio-level forecast", `The graph network's forecast of the day's average target against what it was, out of sample (correlation ${sig(p.correlation, 2)} over ${p.n_days} days): the market-wide part of its view, beside the asset-by-asset one.`, {
        draw: (b) => C.line(b, { dates: s.dates, series: [{ name: "Forecast", color: css("--s1"), values: s.predicted }, { name: "Realised", color: css("--muted"), values: s.actual, width: 1 }], baseline: 0, yFormat: pctTick, tipFormat: (v) => spct(v, 2), height: 230, label: "Portfolio-level forecast and outcome" }),
        table: (b) => table(b, [{ label: "Date", get: (n) => s.dates[n] }, { label: "Forecast", num: true, get: (n) => spct(s.predicted[n], 2) }, { label: "Realised", num: true, get: (n) => spct(s.actual[n], 2) }], s.dates.map((_, n) => n)) });
    }
    if (x.graph) {
      const g = x.graph.graph, att = x.graph.attention;
      chartCard(ch, "The asset graph", `Edges the network passes information along (${g.n_edges} in all, ${sig(g.mean_degree, 1)} per asset, built from ${g.mode === "both" ? "correlation and shared sector or class" : g.mode}); the strongest are listed${att ? ", then the strongest attention weights of the last fit" : ""}.`, {
        draw: (b) => table(b, [{ label: "Asset", get: (e) => e[0] }, { label: "Neighbour", get: (e) => e[1] }, { label: "Edge weight", num: true, get: (e) => num(e[2], 3) }], g.edges),
        table: (b) => table(b, [{ label: "Asset", get: (e) => e[0] }, { label: "Neighbour", get: (e) => e[1] }, { label: "Weight", num: true, get: (e) => num(e[2], 3) }], g.edges.concat(att || [])) });
    }
    if (x.attention) {
      const a = x.attention.steps;
      chartCard(ch, "Attention over the window", "How much weight the transformer's last step gives each day of its lookback window (the right-hand bar is the day being forecast).", {
        draw: (b) => C.columns(b, { labels: a.map((_, i) => (i - a.length + 1 === 0 ? "t" : String(i - a.length + 1))), values: a, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), name: "Attention", height: 200, label: "Attention weight by day in the window" }),
        table: (b) => table(b, [{ label: "Day", get: (i) => (i - a.length + 1 === 0 ? "t" : String(i - a.length + 1)) }, { label: "Attention", num: true, get: (i) => num(a[i], 4) }], a.map((_, i) => i)) });
    }
    if (x.rewards) {
      const w = x.rewards, c = w.curve, e = w.in_sample;
      chartCard(ch, "Agent reward while training", `The ${w.algo === "recurrent_ppo" ? "recurrent PPO" : w.algo.toUpperCase()} agent's average reward per episode (${w.reward} reward, in percent per day) over its ${w.timesteps.toLocaleString("en-US")} training steps, last refit. A curve that rises and flattens is learning; a flat one has found nothing in these returns. The positions are limited to gross ${num(w.constraints.gross, 2)}, ${pct(w.constraints.max_weight, 0)} per asset${w.constraints.long_only ? ", long only" : ", dollar neutral"}.`, {
        draw: (b) => (c.timesteps.length ? C.line(b, { xs: c.timesteps, series: [{ name: "Mean episode reward", color: css("--s1"), values: c.mean_reward }], xLabel: "training steps", baseline: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), height: 230, label: "Mean episode reward during training" }) : h("p", "hint", b, "The run was too short to complete an episode.")),
        table: (b) => table(b, [{ label: "Steps", num: true, get: (n) => String(c.timesteps[n]) }, { label: "Mean episode reward", num: true, get: (n) => num(c.mean_reward[n], 4) }], c.timesteps.map((_, n) => n)) });
      chartCard(ch, "Agent on its own training days", `What the final policy earns on the days it trained on, with costs (Sharpe ${sig(e.sharpe, 2)}). This is in-sample and flatters; the equity curve above is the evidence.`, {
        draw: (b) => C.line(b, { dates: e.dates, series: [{ name: "Training window (in-sample)", color: css("--muted"), values: e.equity, dash: "4 3" }], baseline: 1, yFormat: (v) => v.toFixed(2), tipFormat: (v) => v.toFixed(3), height: 200, label: "Equity on the training window" }),
        table: (b) => table(b, [{ label: "Date", get: (n) => e.dates[n] }, { label: "Equity", num: true, get: (n) => num(e.equity[n], 4) }], e.dates.map((_, n) => n)) });
    }
  }
  function provenanceRows(r) {
    const e = r.model_extras || {}, res = e.resources || {}, rep = e.reproducibility || {};
    if (!e.resources) return [];
    return [["Training time", `${num(res.train_seconds, 1)} s of ${num(res.total_seconds, 1)} s for the run`], ["Device", (res.devices || []).join(", ") + ((res.notes || []).length ? ` (${res.notes.join("; ")})` : "") + (res.cuda === false ? ", no GPU visible" : "")],
      ["Parameters", res.n_params ? res.n_params.toLocaleString("en-US") : "not a parametric network"], ["Peak memory", res.peak_memory_mb ? `${num(res.peak_memory_mb, 0)} MB (the process's peak)` : "not measured"],
      ["Configuration hash", String(rep.config_hash || "").slice(0, 16)], ["Data fingerprint", String(rep.data_fingerprint || "n/a").slice(0, 16)], ["Code", rep.git_commit || "not a git checkout"],
      ["Libraries", Object.entries(rep.versions || {}).map(([k, v]) => `${k} ${v}`).join(", ")], ["Model file", rep.model_artifact ? `${rep.model_artifact.bytes.toLocaleString("en-US")} bytes, SHA-256 ${rep.model_artifact.sha256.slice(0, 16)}…` : "none kept"]];
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
    if (oos.information_ratio !== undefined && oos.information_ratio !== null) {                     // what a book is judged on against its benchmark: the return it adds per unit of the risk it takes away from it
      const k3 = h("div", "earnrow", root); k3.dataset.role = "benchmark-relative"; const bm = r.metrics.benchmark || {};
      tile(k3, "Information ratio", sig(oos.information_ratio), { text: `active return per unit of tracking error, out of sample`, dir: oos.information_ratio > 0 ? "up" : "down" }, "", "Mean return over the benchmark divided by its standard deviation, annualised. 0.5 is good for a long-only book, above 1 is rare");
      tile(k3, "Active return", spct((oos.cagr ?? 0) - (bm.cagr ?? 0)), { text: `CAGR ${pct(oos.cagr, 1)} against ${pct(bm.cagr, 1)} for ${benchName(r).toLowerCase()}` }, "", "Compound annual return of the strategy minus the benchmark's, out of sample");
      tile(k3, "Tracking error", pct(oos.tracking_error, 1), { text: "a year: how far it strays from the benchmark" });
      tile(k3, "Beta to the benchmark", sig(oos.beta_to_benchmark), { text: `alpha ${spct(oos.alpha_ann)} a year after beta`, dir: (oos.alpha_ann ?? 0) > 0 ? "up" : "down" }, "", "A beta far from 1 means the book is a different bet from the benchmark (below 1: less market; above: more). Alpha is the return beta does not explain");
    }
    const ch = h("div", "charts", root);
    chartCard(ch, "Equity curve by segment", "Growth of 1 from the first training day, on a log scale when it spans more than five times (equal heights are then equal percentage moves). The dashed line is in-sample, so it flatters; the three segments after it are out of sample. The benchmark is rebased to the strategy where validation starts.", { wide: true, draw: (b) => equityChart(b, r), table: (b) => equityTable(b, r) });
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
    modelCards(ch, r);
    if (Q.howCard && d.id && d.id !== "unsaved") Q.howCard(root, d.id);
    if (Q.tradesCard && d.id && d.id !== "unsaved") Q.tradesCard(root, d.id, (id) => { history.pushState(null, "", "#/strategy?id=" + encodeURIComponent(id)); window.dispatchEvent(new PopStateEvent("popstate")); });      // every rule's buys and sells, drawn on the asset's price
    const card = h("div", "card", root); h("h3", "", card, "What ran");
    const plan = r.segments.plan, own = cfg.model.params || {};
    W.kv(card, "Settings", [["Model", cfg.model.kind + (Object.keys(own).length ? " " + Object.entries(own).map(([a, v]) => `${a}=${v}`).join(", ") : "")], ["Target", ds.target], ["Features", `${ds.n_features} (${(r.features.names || []).length ? r.features.names.slice(0, 6).join(", ") + (r.features.names.length > 6 ? ", …" : "") : ""})`],
      ["Position rule", SIGNAL_NAME[cfg.signal.method] || cfg.signal.method], ["Rebalance", cfg.backtest.rebalance], ["Costs, one way", `${num(cfg.backtest.cost_bps, 1)} bps commission + ${num(cfg.backtest.slippage_bps, 1)} bps slippage`],
      ["Volatility target", cfg.backtest.target_vol ? pct(cfg.backtest.target_vol, 0) : "none"], ["Refits", `${r.folds.n} walk-forward fits, every ${cfg.split.refit_every} days${r.folds.n_skipped ? `, ${r.folds.n_skipped} skipped` : ""}`], ["Training window", cfg.split.window + (cfg.split.window === "rolling" ? ` (${cfg.split.rolling_days} days)` : "")],
      ["Seed", String(cfg.seed)], ["Experiment", d.id]]);
    const how = provenanceRows(r);
    if (how.length) W.kv(card, "Compute and provenance", how);
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
    const sc = r.config.signal, bc = r.config.backtest;
    const st = { signal: sc.method, n: String(sc.n), tilt: String(sc.tilt ?? 0.5), quantile: pc(sc.quantile ?? 0.3), holding: String(sc.holding_period ?? 1), rebalance: bc.rebalance, cost: String(bc.cost_bps), slippage: String(bc.slippage_bps),
      targetVol: bc.target_vol === null || bc.target_vol === undefined ? "" : pc(bc.target_vol) }, grid = h("div", "params", box);
    if (sc.method === "top_n_long_short" && Number(sc.net || 0) === 0) h("p", "hint", box, "This run's rule is dollar neutral: it holds no market exposure, so it cannot be compared with the long-only equal-weight benchmark. 'Long only: tilted around equal weight' keeps the benchmark's exposure and lets the model move weight between assets.");
    Q.labField(grid, { key: "signal", label: "Position rule", kind: "choice", options: M.catalog.signals.map((s) => [s.id, SIGNAL_NAME[s.id] || s.id]) }, st);
    Q.labField(grid, { key: "n", label: "n, top and bottom (long the top n, short the bottom n)", kind: "number", step: 1, min: 1, max: 20 }, st);
    Q.labField(grid, { key: "tilt", label: "Tilt (tilted around equal weight: 0 = the benchmark, 1 = the strongest)", kind: "number", step: 0.05, min: 0, max: 1 }, st);
    Q.labField(grid, { key: "quantile", label: "Top quantile, % (long only: the top quantile)", kind: "number", step: 5, min: 5, max: 50 }, st);
    Q.labField(grid, { key: "holding", label: "Holding period (days)", kind: "number", step: 1, min: 1, max: 60 }, st);
    Q.labField(grid, { key: "rebalance", label: "Rebalance", kind: "choice", options: [["daily", "Daily"], ["weekly", "Weekly"], ["monthly", "Monthly"]] }, st);
    Q.labField(grid, { key: "targetVol", label: "Volatility target (% a year; blank = none)", kind: "number", step: 1, min: 1, max: 100 }, st);
    Q.labField(grid, { key: "cost", label: "Commission (bps)", kind: "number", step: 0.5, min: 0, max: 250 }, st);
    Q.labField(grid, { key: "slippage", label: "Slippage (bps)", kind: "number", step: 0.5, min: 0, max: 250 }, st);
    const go2 = h("button", "btn primary", box, "Re-backtest");
    go2.addEventListener("click", async () => {
      let body; try { body = { id: d.id, signal: { method: st.signal, n: needNumber(st.n, "n"), tilt: needNumber(st.tilt, "Tilt"), quantile: needNumber(st.quantile, "Quantile") / 100, holding_period: needNumber(st.holding, "Holding period") },
        backtest: { rebalance: st.rebalance, cost_bps: needNumber(st.cost, "Commission"), slippage_bps: needNumber(st.slippage, "Slippage"), target_vol: st.targetVol === "" ? null : needNumber(st.targetVol, "Volatility target") / 100 } }; } catch (e) { Q.toast(e.message); return; }
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
    const bar = W.pills(card, [["", "All"], ...Object.keys(d.groups).map((g) => [g, `${g} (${d.groups[g]})`])], group, (v) => { group = v; for (const x of bar.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x.dataset.value === v)); paint(); }, "Filter the features by group");
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
  const ICON = { good: "✓", mixed: "!", bad: "✗", keep: "=" };                              // a verdict is never colour alone
  const setting = (v) => (typeof v === "number" ? String(+Number(v).toPrecision(3)) : String(v));
  /* A tuning that started from the form's own settings: the verdict in words, the settings it chose against the ones it started from, and a button that puts the chosen ones in the Train form. */
  function plainTune(root, d) {
    const v = d.summary, card = h("div", "card verdict-card " + v.level, root); card.dataset.level = v.level;
    const head = h("div", "vhead", card); h("span", "vicon", head, ICON[v.level] || "?"); h("h2", "", head, v.headline);
    const ul = h("ul", "", card); for (const r of v.reasons) h("li", "", ul, r);
    const c = W.card(root, "The settings", "The settings you started from, and the best the search found. Values are rounded; the button uses the exact ones.");
    table(c, [{ label: "Setting", get: (p) => p.split(".").pop().replace(/_/g, " ") }, { label: "Starting", num: true, get: (p) => setting(d.start[p]) }, { label: "Best found", num: true, get: (p) => setting(d.best.params[p]) },
      { label: "What it does", get: (p) => (d.texts || {})[p] || "" }], d.order || d.space);
    const row = h("div", "row tight", c), use = h("button", "btn primary", row, "Use these settings in the Train form"); use.type = "button"; use.dataset.action = "use-tuned";
    use.addEventListener("click", () => useTuned(d));
    const other = h("button", "link", row, "Tuning a strategy from the Backtest tab (the Kalman trend filter, momentum…)? The Tune tab does that."); other.type = "button"; other.addEventListener("click", () => Q.showTab("tune"));
  }
  function useTuned(d) {
    const f = M.form; f.model = d.kind; f.params[d.kind] = f.params[d.kind] || {};
    for (const [path, value] of Object.entries(d.best.params)) {
      const leaf = path.split(".").pop();
      if (path.startsWith("model.params.")) f.params[d.kind][leaf] = String(value);
      else if (path === "dataset.horizon") f.horizon = String(value);
      else if (path === "signal.method") f.signal = value;
      else if (path === "signal.n") f.n = String(value);
      else if (path === "signal.holding_period") f.holding = String(value);
      else if (path === "backtest.rebalance") f.rebalance = value;
    }
    Q.toast("The Train form has these settings: press Train and backtest"); go("train");
  }
  function buildSearch(root, d) {
    const tune = !d.finalists, sel = d.selection || {}, trials = d.trials || [], best = d.best;
    header(root, tune ? "Tuning one model" : "Strategy search", `${d.n_trials} trials${d.n_failed ? ` (${d.n_failed} failed)` : ""}${d.seconds ? ` · ${num(d.seconds, 1)} s` : ""}${d.stages ? " · " + Object.entries(d.stages).map(([s, n]) => `${n} ${s}`).join(", ") : ""}`);
    if (tune && d.summary) plainTune(root, d);
    W.flags(root, [d.note || "Scores are validation-only: the test and forward segments were not predicted during the search."], "Read first. ");
    if (d.finalize && d.finalize.note) W.flags(root, [d.finalize.note], "Holdout. ");
    if (d.evaluation && d.evaluation.deprecated) { const ev = W.card(root, "The holdout was used", d.evaluation.deprecated); W.kv(ev, "Evaluation", [["Plan", ((d.evaluation.plan || {}).plan_id) || "n/a"], ["Idempotency key", d.evaluation.idempotency_key || "n/a"], ["Untouched", d.evaluation.untouched ? "yes" : "no"]]); }
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
    if (d.model_selection) selectionCard(root, d.model_selection);
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

  /* ---------- the constrained selection: eligibility, ranking, Pareto, simplicity ---------- */
  const SEL_STATE = { eligible: "eligible", ineligible: "ineligible", insufficient: "too little data", failed: "failed to run" };
  const snake = (t) => String(t).replace(/_/g, " ");
  function selectionCard(root, s) {
    const nm = (x) => { const id = typeof x === "string" ? x : x.id, c = typeof x === "string" ? (s.candidates || []).find((k) => k.id === x) : x; return c && c.label && c.label !== id ? c.label : (String(id).startsWith("trial-") ? id : shortId(id)); };
    const c = W.card(root, "Constrained selection", `${s.n_candidates} candidates · ranked by ${snake(s.primary_metric)} on the pooled chronological validation returns · policy ${s.policy_hash}`);
    W.flags(c, [s.scope.note, s.claims.out_of_sample], "Read first. ");
    const sel = (s.candidates || []).find((x) => x.id === s.selected), top = (s.candidates || []).find((x) => x.id === s.highest_ranked);
    W.kv(c, "Result", [["Eligible", `${s.counts.eligible} of ${s.n_candidates} (${s.counts.ineligible} ineligible, ${s.counts.insufficient} with too little data, ${s.counts.failed} failed to run)`], ["Highest ranked", top ? nm(top) : "none"],
      ["Selected", sel ? nm(sel) + (s.changed_by_simplicity ? ": simpler than the highest ranked, and not meaningfully worse by the rule below" : "") : "nothing: " + (s.reason || "")], ["What the choice means", s.claims.selection], ["Equivalence", s.claims.statistical_equivalence]]);
    table(c, [{ label: "Candidate", get: (r) => nm(r) }, { label: "State", get: (r) => SEL_STATE[r.state] || r.state }, { label: cap(snake(s.primary_metric)), num: true, get: (r) => sig(r.primary) },
      { label: "Max drawdown", num: true, get: (r) => (r.metrics && !Q.bad(r.metrics.max_drawdown) ? pct(r.metrics.max_drawdown, 1) : "n/a") }, { label: "Turnover a year", num: true, get: (r) => (r.metrics && !Q.bad(r.metrics.turnover) ? sig(r.metrics.turnover, 1) + "×" : "n/a") },
      { label: "Instability", num: true, get: (r) => (r.metrics && !Q.bad(r.metrics.instability) ? sig(r.metrics.instability, 2) : "n/a") },
      { label: "Complexity", get: (r) => (r.complexity ? `family ${r.complexity.model_complexity}, ${Q.bad(r.complexity.n_features) ? "?" : sig(r.complexity.n_features, 0)} features` : "n/a") }, { label: "Why it is not eligible", get: (r) => r.reason || "" }], s.candidates || []);
    if (s.pareto && s.pareto.eligibility === "applied") {
      W.sub(c, "Pareto comparison");
      W.hint(c, `On ${s.pareto.objectives.map(snake).join(", ")} the Pareto front is ${s.pareto.front.map(nm).join(", ") || "empty"}: no other candidate is at least as good on every criterion and better on one.${(s.pareto.dropped || []).length ? " Left out because some candidate lacks them: " + s.pareto.dropped.map((x) => snake(x.objective)).join(", ") + "." : ""}`);
      if ((s.against_selected || []).length) table(c, [{ label: "Compared with the selected", get: (r) => nm(r.b) }, { label: "Verdict", get: (r) => r.verdict }, { label: "It is better on", get: (r) => r.better_b.map(snake).join(", ") || "nothing" },
        { label: "It is worse on", get: (r) => r.better_a.map(snake).join(", ") || "nothing" }], s.against_selected);
    }
    const ne = s.nested;
    if (ne && ne.eligibility === "applied") {
      W.sub(c, "Nested estimate of the ranking rule");
      W.hint(c, ne.reading + ".");
      W.kv(c, "Chained result", [["Days chained", `${ne.chain_days} (${ne.outer_first} to ${ne.outer_last})`], ["The rule, choosing as it went", sig(ne.chain_sharpe)], ["The selected candidate on the same days", sig(ne.selected_on_outer_days)],
        ["The median candidate on the same days", sig(ne.median_candidate)], ["The best candidate in hindsight", sig(ne.best_in_hindsight)], ["Audit of the windows", ne.audit.ok ? `passed (${ne.audit.steps_checked} steps)` : "FAILED: " + ne.audit.violations.join("; ")]]);
      table(c, [{ label: "Cutoff", get: (r) => r.cutoff }, { label: "Chosen", get: (r) => (r.chosen ? nm(r.chosen) : "") }, { label: "Inner days", num: true, get: (r) => String(r.n_inner || 0) },
        { label: "Outer days", num: true, get: (r) => String(r.n_outer || 0) }, { label: "State", get: (r) => r.state }], ne.steps);
    } else if (ne && ne.reason) W.hint(c, "Nested estimate: " + ne.reason + ".");
    const sp = s.simplicity;
    if (sp && sp.table) {
      W.sub(c, sp.claim.kind === "formal_non_inferiority" ? "Simplicity: a formal non-inferiority test" : "Simplicity: a selection heuristic, not a test of equivalence");
      W.hint(c, sp.claim.statement);
      table(c, [{ label: "Candidate", get: (r) => nm(r.id) }, { label: "Gap to the highest ranked", num: true, get: (r) => (Q.bad(r.gap) ? "n/a" : sig(r.gap, 3)) }, { label: "Standard error of the paired gap", num: true, get: (r) => (Q.bad(r.se_of_gap) ? "n/a" : sig(r.se_of_gap, 3)) },
        { label: "How", get: (r) => snake(r.how || "") }, { label: "Status", get: (r) => snake(r.statistical_status) }, { label: "Why", get: (r) => r.reason }], sp.table);
      W.flags(c, sp.claim.caveats || [], "Limits. ");
    }
  }
  function buildSelect(root, d) {
    header(root, "Choosing among saved experiments", `${d.n_candidates} candidates${d.selection_id ? " · " + d.selection_id : ""}`);
    const ig = (d.origin || {}).ignored_rows || {}, n = Object.values(ig).reduce((a, r) => a + Object.values(r).reduce((x, y) => x + y, 0), 0);
    if (n) W.flags(root, [`${n} rows outside the validation segment (test, forward or after the cut) were counted and never read.`], "Holdout. ");
    selectionCard(root, d);
  }

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
    if (d.mixture_study) {
      const ms = d.mixture_study, bs = ms.best_single;
      const mc = W.card(root, "Mixtures against the best single member", `The best member (${nm[bs.member] || bs.member}) was chosen on ${ms.development[0]} to ${ms.development[1]} only; every mixture is judged on ${ms.evaluation[0]} to ${ms.evaluation[1]}, with weights learned from earlier days.`);
      W.hint(mc, bs.selection_note);
      const diff = (x) => (x && x.value !== null && x.value !== undefined ? `${sig(x.value)} (${ci(x)})` : x ? String(x.eligibility).replace(/_/g, " ") : "n/a");
      table(mc, [{ label: "Method", get: (r) => METHOD_NAME[r.method] || r.method }, { label: "Sharpe on the evaluation days", num: true, get: (r) => sig(r.sharpe) }, { label: "Minus the best single member", num: true, get: (r) => diff(r.vs_best_single) },
        { label: "Minus the equal-weight mixture", num: true, get: (r) => (r.vs_equal_weight ? diff(r.vs_equal_weight) : "the benchmark") }], ms.rows);
      W.flags(mc, ms.caveats || [], "Limits. ");
    }
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
    const led = (live.ledger && live.ledger.strategy_version) ? live.ledger : d.ledger;
    if (led && led.strategy_version) {
      const cn = led.counts;
      const cl = W.card(root, "Prospective ledger", `Strategy version ${led.strategy_version} · ${led.state.state}. Each decision is committed before its outcome and classified by the server; a replay of history is listed and never counted as prospective evidence.`);
      table(cl, [{ label: "Class", get: (r) => r[0] }, { label: "Committed", num: true, get: (r) => String(r[1].committed) }, { label: "Settled", num: true, get: (r) => String(r[1].settled) }, { label: "Distinct settled days", num: true, get: (r) => String(r[1].distinct_settled_days) }],
        [["prospective", cn.prospective], ["replay (never counted)", cn.replay]]);
      table(cl, [{ label: "Decision", get: (r) => String(r.decision_ts).slice(0, 10) }, { label: "Committed at", get: (r) => r.created_at }, { label: "Class", get: (r) => r.classification }, { label: "Why", get: (r) => r.why }], led.commitments.slice(-10).reverse());
      if (!led.equity) W.hint(cl, "The daily equity of the committed holdings is built from the prices when a step runs: press the button on the left.");
      for (const [cls, eq] of Object.entries(led.equity || {})) {
        if (eq.eligibility !== "applied") { W.hint(cl, `${cls}: ${eq.reason}`); continue; }
        const e2 = h("div", "charts", root);
        chartCard(e2, `Daily equity of the committed holdings (${cls})`, `Built from the holdings, fills and cash of the ledger on the daily prices: ${eq.n_days} days, total return ${spct(eq.total_return, 2)}${eq.sharpe === null || eq.sharpe === undefined ? ` (${eq.reason})` : `, Sharpe ${sig(eq.sharpe)}`}. A multi-day forecast is not turned into a daily return.`, {
          draw: (b) => C.line(b, { dates: eq.dates, series: [{ name: "Equity", color: css(cls === "prospective" ? "--s1" : "--muted"), values: eq.values }], baseline: eq.values[0], yFormat: (v) => Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 }), tipFormat: (v) => Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 }), height: 240, label: `Daily equity, ${cls} commitments` }),
          table: (b) => table(b, [{ label: "Date", get: (i) => eq.dates[i] }, { label: "Equity", num: true, get: (i) => Number(eq.values[i]).toLocaleString("en-US", { maximumFractionDigits: 0 }) }], eq.dates.map((_, i) => i).filter((i) => i % 5 === 0)) });
      }
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


  /* ---------- Release 5: the validation policy, the statistical diagnostics, governance and the prospective ledger ---------- */
  const ci = (r) => (r && r.interval && r.interval[0] !== null ? `${sig(r.interval[0])} to ${sig(r.interval[1])}` : "n/a");
  function policyCard(root, pol) {
    const a = pol.adopted;
    const c = W.card(root, "Validation policy", a ? `${a.policy.name} ${a.policy.version} · adopted by ${a.adoption.actor} on ${a.adoption.at}: ${a.adoption.note}`
      : "No policy is adopted: the platform implies no Sharpe, deflated-Sharpe, PBO, p-value or paper-trading threshold, so Gates B and C are blocked.");
    if (!a) { W.hint(c, "A named person adopts a policy before the evaluation: quant ml-policy --file policy.json --actor NAME --note REASON (or POST /api/ml/validation-policy). An illustrative example exists (--adopt-example); it is not a recommendation."); return; }
    if (a.policy.illustrative) W.flags(c, ["This is the ILLUSTRATIVE example policy: its numbers are not calibrated to any market or strategy and are not a recommendation."], "Read first. ");
    table(c, [{ label: "Criterion", get: (r) => r[0] }, { label: "Declared value", get: (r) => String(r[1]) }, { label: "Meaning", get: (r) => (pol.criteria[r[0]] || {}).text || "" }], Object.entries(a.policy.criteria).filter((r) => r[1] !== null));
  }
  function diagnosticsCard(root, dg) {
    const rec = dg.record, trim = (t) => String(t || "").slice(0, 10);
    const c = W.card(root, "Statistical diagnostics", rec ? `${rec.sample.n} days on the ${rec.segment} segment (${trim(rec.sample.first)} to ${trim(rec.sample.last)}) · computed ${dg.created} under policy ${dg.policy_hash || "none"}${dg.stale ? " · STALE: another policy is adopted now" : ""}`
      : "No diagnostics have been computed for this experiment.");
    actions(c, [["Compute the diagnostics", () => W.request(M, () => [{ modes: ["controls_dx_run"], job: true, path: "/api/ml/statistical-diagnostics/run", body: { id: M.cur } }, ...CONTROLS(M.cur)], "Computing the diagnostics…")]]);
    if (dg.note) W.hint(c, dg.note);
    if (rec) {
      W.sub(c, "Sharpe ratio: one estimate, three statements of its uncertainty");
      table(c, [{ label: "Estimator", get: (r) => r.label }, { label: "Annualised Sharpe", num: true, get: (r) => sig(r.value) }, { label: `${pct(rec.level, 0)} interval`, num: true, get: (r) => ci(r) },
        { label: "Dependence assumed", get: (r) => r.dependence }, { label: "State", get: (r) => (r.eligibility === "applied" ? "applied" : `${r.eligibility}: ${r.reason}`) }],
      [["naive", "Naive (treats days as independent)"], ["hac", "HAC (Newey-West, delta method)"], ["bootstrap", "Stationary bootstrap"]].map(([k, label]) => ({ ...rec.sharpe[k], label })));
      W.sub(c, "Against matched benchmarks (estimated from information known before each day)");
      table(c, [{ label: "Comparison", get: (b) => String(b.kind).replace(/_/g, " ") }, { label: "Matched", get: (b) => (b.matched ? "yes" : "no: descriptive only") }, { label: "Days", num: true, get: (b) => String(b.n_aligned) },
        { label: "Active mean a day", num: true, get: (b) => (Q.bad(b.active_mean.value) ? "n/a" : num(b.active_mean.value, 5)) }, { label: "Interval", num: true, get: (b) => ci(b.active_mean) },
        { label: "Sharpe difference", num: true, get: (b) => (Q.bad(b.sharpe_difference.value) ? "n/a" : sig(b.sharpe_difference.value)) }], Object.values(rec.benchmarks));
      W.sub(c, "Regimes (declared before the evaluation)");
      table(c, [{ label: "Regime", get: (r) => `${r.regime}: ${r.state}` }, { label: "Kind", get: (r) => (r.causal ? "causal rule" : "retrospective slice") }, { label: "Days", num: true, get: (r) => String(r.n_days) },
        { label: "Sharpe (HAC interval)", num: true, get: (r) => (r.eligibility === "applied" ? ci({ interval: r.sharpe_hac_interval }) : "insufficient evidence") }], rec.regimes.rows);
      if (rec.neighborhood) {
        W.sub(c, `Neighbourhood of ${rec.neighborhood.declared_points} predeclared settings (the whole grid, not the best cell)`);
        if (rec.neighborhood.summary) table(c, [{ label: "Minimum", num: true, get: (x) => sig(x.min) }, { label: "Median", num: true, get: (x) => sig(x.median) }, { label: "Maximum", num: true, get: (x) => sig(x.max) },
          { label: "Share positive", num: true, get: (x) => spct(x.share_positive, 0) }, { label: "Baseline", num: true, get: (x) => sig(rec.neighborhood.baseline) }], [rec.neighborhood.summary]);
        else W.hint(c, rec.neighborhood.reason || "insufficient");
        if (rec.neighborhood.refit) W.hint(c, (rec.neighborhood.notes || []).join(" "));
      }
      if (rec.delay) { W.sub(c, "Delay between decision and fill"); table(c, [{ label: "Extra sessions", num: true, get: (r) => String(r.extra_sessions) }, { label: "Sharpe", num: true, get: (r) => sig(r.value) }, { label: "Share of the undelayed result", num: true, get: (r) => (Q.bad(r.ratio_to_undelayed) ? "undefined" : spct(r.ratio_to_undelayed, 0)) }], rec.delay.rows); }
      if (rec.capacity) { W.sub(c, "Capacity (realistic execution)"); table(c, [{ label: "Capital", num: true, get: (r) => Number(r.capital).toLocaleString("en-US") }, { label: "State", get: (r) => r.state }, { label: "Sharpe", num: true, get: (r) => sig(r.value) }], rec.capacity.rows); }
      const pcal = rec.calibration, ical = rec.interval_calibration;
      if (pcal && pcal.eligibility === "not_applicable") W.hint(c, `Probability calibration: ${pcal.reason}`);
      else if (pcal && pcal.segments) {
        W.sub(c, "Class probabilities: Brier score, log loss and calibration error, each beside the forecast that always says the base rate");
        table(c, [{ label: "Segment", get: (r) => r.segment }, { label: "Forecasts", num: true, get: (r) => String(r.n) }, { label: "Brier (base rate)", num: true, get: (r) => (r.eligibility === "applied" ? `${sig(r.brier, 4)} (${sig(r.brier_base_rate, 4)})` : r.reason) },
          { label: "Log loss (base rate)", num: true, get: (r) => (r.eligibility === "applied" ? `${sig(r.log_loss, 4)} (${sig(r.log_loss_base_rate, 4)})` : "") }, { label: "Calibration error", num: true, get: (r) => (r.eligibility === "applied" ? sig(r.reliability.ece, 3) : "") }],
        Object.entries(pcal.segments).map(([segment, r]) => ({ ...r, segment })));
      }
      if (ical && ical.eligibility === "applied") {
        const q = ical.quantiles, span = (r) => `${spct(r.coverage_interval[0], 1)} to ${spct(r.coverage_interval[1], 1)}`;
        W.sub(c, `Prediction interval of the ${ical.model} model: nominal ${spct(q.nominal, 0)}, from the ${num(q.alpha_low, 2)} and ${num(q.alpha_high, 2)} quantiles (the ${ical.segment} segment)`);
        table(c, [{ label: "Segment", get: (r) => r.segment }, { label: "Forecasts", num: true, get: (r) => String(r.n) }, { label: "Coverage (interval)", num: true, get: (r) => (r.eligibility === "applied" ? `${spct(r.coverage, 1)} (${span(r)})` : r.reason) },
          { label: "Mean width", num: true, get: (r) => (r.eligibility === "applied" ? spct(r.width, 2) : "") }, { label: "Interval score", num: true, get: (r) => (r.eligibility === "applied" ? sig(r.interval_score, 4) : "") },
          { label: "Pinball: lower / centre / upper", num: true, get: (r) => (r.eligibility === "applied" ? [r.pinball.lower, r.pinball.centre, r.pinball.upper].map((v) => sig(v, 4)).join(" / ") : "") }],
        Object.entries(q.segments).map(([segment, r]) => ({ ...r, segment })));
        W.sub(c, `The same level from time-series conformal prediction, calibrated only on outcomes that had ended (${ical.conformal.delay_sessions} sessions earlier or more)`);
        table(c, [{ label: "Method", get: (r) => r.method }, { label: "Forecasts with an interval", num: true, get: (r) => String(r.n) }, { label: "Coverage (interval)", num: true, get: (r) => (r.eligibility === "applied" ? `${spct(r.coverage, 1)} (${span(r)})` : r.reason) },
          { label: "Mean width", num: true, get: (r) => (r.eligibility === "applied" ? spct(r.width, 2) : "") }, { label: "What is guaranteed", get: (r) => r.guarantee }],
        Object.entries(ical.conformal.methods).flatMap(([name, m]) => Object.values(m.segments).map((r) => ({ ...r, method: name === "split" ? "Rolling split conformal" : "Adaptive conformal inference", guarantee: m.guarantee }))));
        W.hint(c, ical.note);
      } else if (ical && ical.reason) W.hint(c, `Interval calibration: ${ical.reason}`);
      W.flags(c, rec.limitations || [], "Limits. ");
    }
    if (dg.items.length) { W.sub(c, "Read against the adopted policy"); table(c, [{ label: "Criterion", get: (i) => i.label }, { label: "State", get: (i) => ITEM_WORD[i.status] || i.status }, { label: "Why", get: (i) => i.reason }], dg.items); }
    if (dg.multiseed && dg.multiseed.rows) {
      const m = dg.multiseed;
      W.sub(c, "Seeds declared in advance (the median is the estimate; no seed is chosen)");
      table(c, [{ label: "Seed", num: true, get: (r) => String(r.seed) }, { label: m.metric, num: true, get: (r) => sig(r.value) }, { label: "Its own HAC interval", num: true, get: (r) => ci({ interval: r.sharpe_hac_interval }) }], m.rows);
      if (m.summary) W.hint(c, `Median ${sig(m.summary.median)}, spread across seeds ${sig(m.summary.std_across_seeds)}. ${m.flag || ""}`);
    }
  }
  function governanceCard(root, gv) {
    const cur = gv.current, led = gv.ledger || {};
    const c = W.card(root, "Strategy version, risk limits and prospective evidence", cur ? `Version ${cur.version.version_id} (${gv.versions.length} registered) · ${cur.state.state}${cur.state.state !== "active" ? ` since ${cur.state.since}: ${cur.state.reason}` : ""}`
      : "No strategy version is registered. A version is the model together with its signal, execution and risk settings; changing any of them is a new version and its evidence starts again.");
    if (!cur) { W.hint(c, "Register one: quant ml-governance EXPERIMENT --register --actor NAME [--limits JSON] (or POST /api/ml/governance/version)."); return; }
    const lim = cur.risk_limits && Object.keys(cur.risk_limits.limits).length ? Object.entries(cur.risk_limits.limits).map(([k, v]) => `${k} ${v}`).join(", ") : "none documented";
    W.kv(c, "Operating controls", [["Risk limits", lim], ["Recovery procedure", cur.recovery_plan ? `written by ${cur.recovery_plan.actor}` : "not recorded"], ["Latest monitoring", cur.last_monitoring ? `${cur.last_monitoring.at}: ${cur.last_monitoring.detail.status}` : "never"],
      ["Deployment approvals", String(cur.approvals.length)], ["Model artifact", String((cur.version.content.model_artifact || {}).sha256 || "none").slice(0, 12)], ["Execution", String(cur.version.content.settings.execution === "vectorized_legacy" ? "vectorized legacy" : "declared")]]);
    if (led.strategy_version && led.commitments.length) {
      const cn = led.counts;
      W.sub(c, "Prospective ledger (immutable commitments; replays are listed and never counted)");
      table(c, [{ label: "Class", get: (r) => r[0] }, { label: "Committed", num: true, get: (r) => String(r[1].committed) }, { label: "Settled", num: true, get: (r) => String(r[1].settled) }, { label: "Distinct settled days", num: true, get: (r) => String(r[1].distinct_settled_days) }],
        [["prospective", cn.prospective], ["replay (never counted)", cn.replay]]);
      if (led.commitments.length) table(c, [{ label: "Decision", get: (r) => String(r.decision_ts).slice(0, 10) }, { label: "Committed at", get: (r) => r.created_at }, { label: "Class", get: (r) => r.classification }, { label: "Why", get: (r) => r.why }, { label: "Settled", num: true, get: (r) => String(r.settled_versions) }], led.commitments.slice(-15).reverse());
    } else W.hint(c, "Nothing has been committed yet: the Forward test records the first commitment.");
  }

  /* ---------- Release 6: alpha and portfolio diagnostics, and controlled comparisons against a frozen baseline ---------- */
  function alphaCard(root, al) {
    const ok = al.eligibility === "applied";
    const c = W.card(root, "Alpha diagnostics", ok ? `${al.model} on ${al.target} · ${al.segment} segment · ${al.label} · recorded ${al.computed_at || "just now"} (each request is kept as a look)`
      : "Rank IC, decay, coverage, stability and redundancy of the saved scores, on development data. The test and forward segments stay sealed.");
    actions(c, [["Compute the alpha diagnostics", () => W.request(M, () => [{ modes: ["controls_alpha_run"], path: "/api/ml/alpha-diagnostics", query: { id: M.cur } }, ...CONTROLS(M.cur)], "Computing the alpha diagnostics…")]]);
    if (!ok) { W.hint(c, al.reason || "Not computed yet."); return; }
    const m = al.ic.mean;
    W.kv(c, "Rank IC", [["Mean", m.eligibility === "applied" ? `${num(m.value, 4)} (${pct(m.level, 0)} interval ${ci(m)})` : `${m.eligibility}: ${m.reason}`], ["Dates with an IC", String(al.ic.n_with_ic)],
      ["Dependence", `Newey-West, lag ${m.settings ? m.settings.lags : "n/a"} (labels of ${al.horizon} sessions overlap)`], ["Hit rate", Q.bad(al.ic.hit_rate) ? "n/a" : spct(al.ic.hit_rate, 0)],
      ["Dates without an IC", Object.keys(al.ic.unavailable_by_reason).length ? Object.entries(al.ic.unavailable_by_reason).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(", ") : "none"]]);
    const lags = [...new Set(al.decay.cells.map((x) => x.extra_lag))], hs = [...new Set(al.decay.cells.map((x) => x.horizon))], cell = (k, hz) => al.decay.cells.find((x) => x.extra_lag === k && x.horizon === hz);
    W.sub(c, "Decay: mean rank IC by extra delay (rows) and return horizon (columns)");
    table(c, [{ label: "Extra sessions", get: (k) => `+${k}` }, ...hs.map((hz) => ({ label: `${hz}-session return`, num: true, get: (k) => { const x = cell(k, hz); return Q.bad(x.mean_ic) ? "n/a" : `${num(x.mean_ic, 4)}${x.share_of_undelayed !== null && k !== lags[0] ? ` (${spct(x.share_of_undelayed, 0)})` : ""}`; } }))], lags);
    W.kv(c, "Coverage and stability", [["Coverage of investable cells", Q.bad(al.coverage.coverage) ? "n/a" : spct(al.coverage.coverage, 1)], ["Investable cells with no score", String(al.coverage.missing_cells || 0)],
      ["Blocks with the same sign", Q.bad(al.stability.same_sign_share) ? "n/a" : spct(al.stability.same_sign_share, 0)], ["Features / independent features", al.redundancy.eligibility === "applied" ? `${al.redundancy.n_features} / ${num(al.redundancy.effective_independent_features, 1)}` : "n/a"],
      ["Pairs above the redundancy threshold", al.redundancy.eligibility === "applied" ? String((al.redundancy.pairs_above_threshold || []).length) : "n/a"]]);
    if (al.incremental) {
      const i = al.incremental;
      W.sub(c, `What it adds to ${shortId(i.baseline_id)} (predictive, on this sample; not a causal claim and not a feature importance)`);
      if (i.eligibility !== "applied") W.hint(c, i.reason);
      else W.kv(c, "Given the existing signal", [["Conditional IC", `${num(i.conditional_ic.mean.value, 4)} (${ci(i.conditional_ic.mean)})`], ["Plain IC", num(i.plain_ic.mean.value, 4)],
        ["Blend minus baseline, Sharpe", i.blend.difference && i.blend.difference.value !== null ? `${sig(i.blend.difference.value)} (${ci(i.blend.difference)})` : `${i.blend.eligibility}: ${i.blend.reason || ""}`]]);
    }
  }
  function portfolioCard(root, pf) {
    const ok = pf.eligibility === "applied";
    const c = W.card(root, "Portfolio diagnostics", ok ? `${pf.model} on ${pf.target} · ${pf.segment} segment · recorded ${pf.computed_at || "just now"}` : "The book the experiment held, the covariance estimators compared on the risk it realised, risk contributions and the sensitivity of a cost-aware book.");
    actions(c, [["Compute the portfolio diagnostics", () => W.request(M, () => [{ modes: ["controls_port_run"], path: "/api/ml/portfolio-diagnostics", query: { id: M.cur } }, ...CONTROLS(M.cur)], "Computing the portfolio diagnostics…")]]);
    if (!ok) { W.hint(c, pf.reason || "Not computed yet."); return; }
    const b = pf.book;
    W.kv(c, "The book it held", [["Days with a book", String(b.days_with_a_book)], ["Gross (mean / max)", `${num(b.gross.mean, 2)} / ${num(b.gross.max, 2)}`], ["Net (min to max)", `${num(b.net.min, 2)} to ${num(b.net.max, 2)}`],
      ["Effective positions", num(b.concentration.effective_positions, 1)], ["Top three share", spct(b.concentration.top3_share, 0)], ["Mean daily turnover", Q.bad(b.mean_daily_turnover) ? "n/a" : num(b.mean_daily_turnover, 3)]]);
    W.sub(c, `Covariance estimators on the risk the book realised over ${pf.risk_models.horizon} days (${pf.risk_models.decision_dates} dates; forecast over realised of 1 is unbiased)`);
    table(c, [{ label: "Estimator", get: (r) => r[0] }, { label: "Windows", num: true, get: (r) => String(r[1].windows) }, { label: "Forecast / realised", num: true, get: (r) => (Q.bad(r[1].risk_ratio_forecast_to_realised) ? "n/a" : num(r[1].risk_ratio_forecast_to_realised, 2)) },
      { label: "QLIKE", num: true, get: (r) => (Q.bad(r[1].qlike) ? "n/a" : num(r[1].qlike, 3)) }, { label: "Worst condition", num: true, get: (r) => (Q.bad(r[1].worst_condition) ? "n/a" : num(r[1].worst_condition, 0)) },
      { label: "Repair rate", num: true, get: (r) => (Q.bad(r[1].repair_rate) ? "n/a" : spct(r[1].repair_rate, 0)) }, { label: "Minimum-variance book risk", num: true, get: (r) => (Q.bad(r[1].gmv_realised_risk) ? "n/a" : num(r[1].gmv_realised_risk, 4)) }], Object.entries(pf.risk_models.estimators));
    const rc = pf.risk_contributions;
    if (rc && rc.table) {
      W.sub(c, `Risk contributions of the last book (${rc.estimator}, as of ${rc.asof})`);
      const rows = rc.table.filter((r) => Math.abs(r.share) > 0.001).sort((x, y) => Math.abs(y.share) - Math.abs(x.share)).slice(0, 12);
      const ch = h("div", "charts", c);
      chartCard(ch, "Share of the book's variance", rc.chosen_because, { wide: true, draw: (bx) => C.hbars(bx, { rows: rows.map((r) => ({ label: r.asset, value: r.share, tip: [["Weight", num(r.weight, 3)]] })), format: (v) => spct(v, 0), name: "Share", label: "Risk contribution", labelWidth: 90 }),
        table: (bx) => table(bx, [{ label: "Asset", get: (r) => r.asset }, { label: "Weight", num: true, get: (r) => num(r.weight, 3) }, { label: "Share of variance", num: true, get: (r) => spct(r.share, 1) }], rows) });
    }
    const al = pf.allocation;
    W.sub(c, "Cost-aware allocation: how the optimal book moves under a predeclared grid");
    if (al.eligibility !== "applied") W.hint(c, `${al.eligibility === "not_applicable" ? "Not applicable" : "Not run"}: ${al.reason}`);
    else {
      const sm = al.summary || {};
      W.kv(c, `On ${al.asof}`, [["Grid points / without a solution", `${al.points.length} / ${al.failed}`], ["Largest change in the book", Q.bad(sm.max_weight_change) ? "n/a" : num(sm.max_weight_change, 3)], ["Points that do not trade", Q.bad(sm.share_no_trade) ? "n/a" : spct(sm.share_no_trade, 0)],
        ["Risk range", sm.risk_range ? `${num(sm.risk_range[0], 4)} to ${num(sm.risk_range[1], 4)}` : "n/a"], ["Base solve", al.base_status], ["Units", al.units]]);
      W.hint(c, al.interpretation);
    }
    W.flags(c, pf.caveats || [], "Limits. ");
  }
  const FAMILY_PRESETS = { target: [["excess_ret", "Return minus the benchmark's"], ["sector_ret", "Return minus the sector's"], ["resid_ret", "Residual of the benchmark (beta from trailing data)"]],
    horizon: [[1, "1 session"], [5, "5 sessions"], [10, "10 sessions"], [20, "20 sessions"]], feature: [["stable_select", "Stability-selected features inside each fold"]],
    portfolio: [["cost_aware", "Cost-aware allocation in place of the position rule"], ["calibrated_sizing", "Size from a calibrated forecast and its uncertainty"]],
    rebalancing: [["no_trade:1.0", "No-trade region, margin 1 x the incremental cost"], ["no_trade:3.0", "No-trade region, margin 3 x the incremental cost"], ["trade_filter:1.0", "Trade filter: net edge of at least one forecast-error deviation"]],
    risk_control: [["regime_exposure:vol_quantile", "Exposure by volatility band (causal)"], ["regime_exposure:hmm", "Exposure by hidden-Markov state (filtered, never smoothed)"], ["regime_exposure:markov_switching", "Exposure by Markov-switching state (filtered)"]],
    ensemble: [["weighting:equal", "Equal weights"], ["weighting:dynamic", "Weights follow trailing rank IC"], ["weighting:regime", "The best members of each volatility regime"], ["weighting:stacked", "Regularised stacked weights"], ["weighting:stacked_regime", "Stacked weights per regime"]],
    execution: [["realistic", "Orders, fills and participation limits in place of a flat cost"]], combination: [] };
  const FAMILY_NAME = { target: "Target engineering", horizon: "Forecast horizon", feature: "Feature selection", portfolio: "Portfolio construction", rebalancing: "Rebalancing", risk_control: "Risk control", ensemble: "Ensembles",
    execution: "Execution", combination: "Combination or ablation of compared components" };
  function improvementCard(root, im) {
    const cards = im.cards || [];
    const c = W.card(root, "Controlled comparisons against this experiment", "One change at a time, on the same data, folds and costs, on the validation segment. The card is frozen before it runs, runs once, and keeps the negative results. Combinations come only after their components were compared on their own. Nothing here promotes a strategy.");
    const f = M.improve = M.improve || { family: "target", hypothesis: "", change: "", presets: new Set(["excess_ret"]) };
    f.parts = f.parts || new Set(); f.members = f.members || "ridge,lasso,elasticnet"; if (f.select === undefined) f.select = true;
    const row = h("div", "row", c);
    const lab = (text, el) => { const l = h("label", "f", row); h("span", "", l, text); l.appendChild(el); return el; };
    const fam = lab("Family", h("select", "", null)); for (const k of Object.keys(FAMILY_NAME)) { const o = h("option", "", fam, FAMILY_NAME[k]); o.value = k; o.selected = k === f.family; }
    fam.addEventListener("change", () => { f.family = fam.value; f.presets = new Set(); f.parts = new Set(); M.draw(); });
    const hyp = lab("Hypothesis (before the numbers)", h("input", "", null)); hyp.type = "text"; hyp.value = f.hypothesis; hyp.addEventListener("input", () => { f.hypothesis = hyp.value; });
    const chg = lab("The one change", h("input", "", null)); chg.type = "text"; chg.value = f.change; chg.addEventListener("input", () => { f.change = chg.value; });
    if (f.family === "ensemble") {
      const mem = lab("Members of the ensemble", h("input", "", null)); mem.type = "text"; mem.value = f.members; mem.addEventListener("input", () => { f.members = mem.value; });
      const l = h("label", "f inline", row), i = h("input", "", l); i.type = "checkbox"; i.checked = f.select; i.addEventListener("change", () => { f.select = i.checked; }); l.appendChild(document.createTextNode("Compare with the best single member on the development segment"));
    }
    const pick = h("div", "row tight", c);
    if (f.family === "combination") {
      const runCards = cards.filter((x) => x.state === "run");
      for (const x of runCards) x.candidates.forEach((name, j) => {
        const value = JSON.stringify({ card_id: x.card_id, candidate: name }), l = h("label", "f inline", pick), i = h("input", "", l);
        i.type = "checkbox"; i.checked = f.parts.has(value); i.addEventListener("change", () => { if (i.checked) f.parts.add(value); else f.parts.delete(value); });
        l.appendChild(document.createTextNode(`${FAMILY_NAME[x.family] || x.family}: ${name} (${String((x.verdicts || [])[j] || "").replace(/_/g, " ") || "not run"})`));
      });
      if (!runCards.length) W.hint(c, "Run at least two single-change cards first: a combination is built only from components that were compared on their own.");
      const l = h("label", "f inline", pick), i = h("input", "", l); i.type = "checkbox"; i.checked = !!f.ablation; i.addEventListener("change", () => { f.ablation = i.checked; });
      l.appendChild(document.createTextNode("Ablate: remove one component at a time from the full combination"));
    } else for (const [v, text] of FAMILY_PRESETS[f.family]) { const l = h("label", "f inline", pick), i = h("input", "", l); i.type = "checkbox"; i.checked = f.presets.has(v); i.addEventListener("change", () => { if (i.checked) f.presets.add(v); else f.presets.delete(v); }); l.appendChild(document.createTextNode(text)); }
    actions(c, [["Freeze the card and run it", () => {
      const combo = f.family === "combination";
      if (!(combo && f.ablation) && (!f.hypothesis.trim() || !f.change.trim())) { Q.toast("State the hypothesis and the one change before the numbers exist."); return; }
      if (combo ? f.parts.size < 2 : !f.presets.size) { Q.toast(combo ? "Choose at least two compared components." : "Choose at least one candidate."); return; }
      const body = combo ? { family: f.family, hypothesis: f.hypothesis, one_change: f.change, combination_of: [...f.parts].map((x) => JSON.parse(x)), ablation: !!f.ablation }
        : { family: f.family, hypothesis: f.hypothesis, one_change: f.change, from_experiment: M.cur, presets: [...f.presets], ...(f.family === "ensemble" ? { members: f.members, select_baseline: f.select } : {}) };
      W.request(M, () => [{ modes: ["controls_improve_run"], job: true, path: "/api/ml/improvement-experiment", body }, ...CONTROLS(M.cur)], "Running the comparison…");
    }]]);
    if (!cards.length) { W.hint(c, "No card is stored yet."); return; }
    table(c, [{ label: "Card", get: (x) => x.card_id }, { label: "Family", get: (x) => FAMILY_NAME[x.family] || x.family }, { label: "Hypothesis", get: (x) => x.hypothesis }, { label: "State", get: (x) => x.state },
      { label: "Verdicts", get: (x) => (x.verdicts.length ? x.verdicts.map((v) => String(v).replace(/_/g, " ")).join(", ") : "not run") }, { label: "Frozen", get: (x) => x.frozen_at }], cards.slice(0, 12));
  }
  function delayCard(root, dl) {
    const ok = dl.eligibility === "applied";
    const c = W.card(root, "Delay report", ok ? `${dl.model} on ${dl.target} · ${dl.segment} segment · delays ${dl.declaration.delays.join(", ")} (declaration ${dl.declaration.hash}) · recorded ${dl.computed_at || "just now"}`
      : "The rank IC, expected edge, net result and capacity when the book is executed later than planned, at delays declared up front. The test and forward segments stay sealed.");
    actions(c, [["Compute the delay report", () => W.request(M, () => [{ modes: ["controls_delay_run"], path: "/api/ml/delay-report", query: { id: M.cur } }, ...CONTROLS(M.cur)], "Computing the delay report…")]]);
    if (!ok) { W.hint(c, dl.reason || "Not computed yet."); return; }
    W.hint(c, dl.rule + ". The wait is measured on the real calendar, and the IC starts from the same fill the portfolio gets.");
    const cap = (r) => (r.capacity.status === "measured" ? r.capacity.levels.map((x) => `${Number(x.capital).toExponential(0)}: ${spct(x.median_fill_share, 0)}`).join(" · ") : "not established");
    table(c, [{ label: "Delay (sessions)", get: (r) => String(r.delay) }, { label: "Calendar days (mean)", num: true, get: (r) => (Q.bad(r.timing.calendar_days.mean) ? "n/a" : num(r.timing.calendar_days.mean, 1)) },
      { label: "Rank IC", num: true, get: (r) => (Q.bad(r.ic.mean_ic) ? "n/a" : num(r.ic.mean_ic, 4)) }, { label: "IC against undelayed", num: true, get: (r) => (Q.bad(r.ic_ratio_to_undelayed) ? "undefined" : spct(r.ic_ratio_to_undelayed, 0)) },
      { label: "Gross a year", num: true, get: (r) => (r.portfolio.eligibility === "applied" ? spct(r.portfolio.gross_ann_return, 1) : "n/a") }, { label: "Net a year", num: true, get: (r) => (r.portfolio.eligibility === "applied" ? spct(r.portfolio.net_ann_return, 1) : "n/a") },
      { label: "Sharpe", num: true, get: (r) => (r.portfolio.eligibility === "applied" ? sig(r.portfolio.sharpe) : "n/a") }, { label: "Break-even cost (bps)", num: true, get: (r) => (Q.bad(r.portfolio.breakeven_cost_bps) ? "n/a" : sig(r.portfolio.breakeven_cost_bps, 1)) },
      { label: "Share of the book filled", get: cap }], dl.rows);
    W.hint(c, `The experiment declared ${sig(dl.declared_cost_bps, 1)} bps one way. A ratio is undefined when the undelayed result is not positive.`);
    W.flags(c, dl.caveats || [], "Limits. ");
  }

  /* ---------- Release 8: the optional advanced modules. Off by default: each action is a request that names it, is recorded, and changes no stage ---------- */
  const ADV_STATE = { ok: "ok", watch: "watch", alert: "ALERT", insufficient: "not enough data", unavailable: "unavailable", not_applicable: "not applicable", degraded: "degraded" };
  function advancedCard(root, a) {
    const c = W.card(root, "Advanced (optional)", "Off by default: nothing here runs until you ask. Every request is recorded as a look, and none of it changes a stage or promotes a model.");
    const done = (text) => { Q.toast(text); invalidate(); M.run(true); };
    actions(c, [
      ["Check for drift", () => W.request(M, () => [{ modes: ["controls_drift_run"], path: "/api/ml/drift", query: { id: M.cur } }, ...CONTROLS(M.cur)], "Checking the features and the settled IC for drift…")],
      ["Monitor uncertainty", () => W.request(M, () => [{ modes: ["controls_monitor_run"], path: "/api/ml/uncertainty-monitor", query: { id: M.cur } }, ...CONTROLS(M.cur)], "Monitoring calibration, uncertainty and disagreement…")],
      ["Record a manual retraining trigger", async () => { try { const r = await api("/api/ml/retraining/action", { id: M.cur, action: "trigger", kind: "manual" }); const d = r.data || r; done(d.outcome === "proposed" ? `Proposal ${d.proposal} recorded; nothing is trained until a person approves it` : `Not a proposal: ${d.reason}`); } catch (e) { Q.toast(e.message); } }],
      ["List the advanced modules", () => W.request(M, () => [{ modes: ["controls_advanced_run"], path: "/api/ml/advanced", query: {} }, ...CONTROLS(M.cur)], "Listing the advanced modules…")],
      ["Build a report", async () => { try { const r = await api("/api/ml/report/build", { id: M.cur }); const d = r.data || r; done(d.status === "built" ? `Report ${d.report_id} built` : `Report ${d.report_id} is unchanged: the inputs are the same`); } catch (e) { Q.toast(e.message); } }],
    ]);
    const fresh_ = (mode, stored) => { const r = (M.results[mode] || {}).data; return r && r.experiment_id === M.cur ? r : stored; };                 // what was just asked for, else what is on record
    const dr = fresh_("controls_drift_run", a.drift);
    W.sub(c, "Drift");
    if (dr && dr.eligibility === "applied") {
      const inp = dr.inputs, pf = dr.performance, lin = dr.lineage;
      W.hint(c, `Reference ${inp.reference.first} to ${inp.reference.last} (${inp.reference.dates} dates, fixed at the fit cutoff ${inp.reference.fixed_at}) against ${inp.recent.first} to ${inp.recent.last} · as of ${dr.as_of} · recorded ${dr.computed_at || "just now"} · look ${lin.looks_so_far}`);
      table(c, [{ label: "Feature", get: (r) => r.feature }, { label: "State", get: (r) => ADV_STATE[r.state] || r.state }, { label: "PSI (a heuristic)", num: true, get: (r) => sig(r.psi, 3) },
        { label: "KS p, adjusted", num: true, get: (r) => sig(r.ks_p_adjusted, 4) }, { label: "Reference rows", num: true, get: (r) => String(r.n_reference) }, { label: "Recent rows", num: true, get: (r) => String(r.n_recent) }], inp.features.slice(0, 10));
      W.hint(c, `Settled-IC drift: ${ADV_STATE[pf.state] || pf.state}. ${pf.reason || ""} ${lin.new.length ? "New alerts: " + lin.new.join(", ") + ". " : ""}${lin.persisting.length ? "Still alerting: " + lin.persisting.join(", ") + ". " : ""}${dr.note}`);
    } else W.hint(c, (dr && dr.reason && dr.eligibility !== "not_computed") ? dr.reason : "Drift has not been checked for this experiment.");
    const cat = (M.results.controls_advanced_run || {}).data;
    if (cat) {
      W.sub(c, "The twelve optional modules (all off by default)");
      table(c, [{ label: "Requirement", get: (r) => r.id }, { label: "Module", get: (r) => r.name }, { label: "Reached by", get: (r) => r.surface.join(" and ") }, { label: "Runs here", get: (r) => (r.available ? "yes" : "no: " + r.reason) },
        { label: "Verification", get: (r) => (r.verification ? r.verification.status + (r.verification.stale ? " (stale)" : "") : "not run") }], cat.modules);
      W.hint(c, cat.note);
    }
    const mo = fresh_("controls_monitor_run", a.monitor);
    W.sub(c, "Uncertainty and disagreement");
    if (mo && mo.eligibility === "applied") {
      W.hint(c, `Overall: ${ADV_STATE[mo.overall.state] || mo.overall.state}. ${mo.overall.reason} · reference ends ${mo.reference_end} · as of ${mo.as_of}`);
      table(c, [{ label: "Check", get: (r) => cap(r.check) }, { label: "State", get: (r) => ADV_STATE[r.state] || r.state }, { label: "Level", get: (r) => r.level || "" }, { label: "Value", num: true, get: (r) => (Q.bad(r.value) ? "n/a" : sig(r.value, 3)) },
        { label: "Why", get: (r) => r.reason }], mo.checks);
      W.hint(c, mo.note);
    } else W.hint(c, (mo && mo.reason && mo.eligibility !== "not_computed") ? `${mo.reason}` : "The uncertainty monitor has not been run for this experiment. A model that states no uncertainty has nothing to monitor, and that is reported as such, never as healthy.");
    const rt = a.retrain;
    W.sub(c, "Retraining: schedule, diagnostics and a constrained challenger search");
    if (rt) {
      const st = rt.state, ap = rt.adaptive_policy, ev = rt.evidence, approved = rt.events.some((e) => e.event === "approved" && e.proposal === st.open_proposal);
      W.hint(c, `Open proposal: ${st.open_proposal || "none"} · consecutive failures: ${st.consecutive_failures} · ${st.suspended ? "SUSPENDED until a person documents a recovery" : "not suspended"} · ${rt.events_total} event${rt.events_total === 1 ? "" : "s"}`);
      if (rt.events.length) table(c, [{ label: "Event", get: (r) => r.event }, { label: "By", get: (r) => r.actor }, { label: "When", get: (r) => r.created }, { label: "Why", get: (r) => r.reason || r.note || r.trigger || "" }], rt.events.slice(-8));
      W.hint(c, `Default policy: a ${ap.schedule.frequency} schedule (a daily strategy refits every ${ap.refit_every.join(" or ")} sessions), training windows tried ${ap.windows.map((w) => (w[0] === "rolling" ? `rolling ${w[1]} days` : "expanding")).join(" and ")}, at most ${ap.max_challengers} challengers, a challenger must be better at a Holm-adjusted level of ${ap.level} on the same out-of-sample observations. Autonomous retraining is off: ${rt.note}`);
      const te = (M.results.controls_retrain_eval || {}).data;
      if (te && te.subject === M.cur) {
        W.hint(c, `Triggers as of ${te.as_of}: the last fit was ${te.last_fit}, ${te.rows_since_fit} rows since, ${te.train_days} days of history.`);
        table(c, [{ label: "Trigger", get: (r) => snake(r.kind) }, { label: "State", get: (r) => (r.fired ? "FIRED" : snake(r.state)) }, { label: "Enabled", get: (r) => (r.enabled ? "yes" : "no") }, { label: "Why", get: (r) => r.reason }], te.triggers);
        const fired = te.triggers.filter((x) => x.fired && x.enabled && x.kind !== "manual");
        if (fired.length) actions(c, fired.map((x) => [`Propose a retraining: ${snake(x.kind)}`, async () => {
          try { const r = await api("/api/ml/retraining/action", { id: M.cur, action: "trigger", kind: x.kind }); const d = r.data || r; done(d.outcome === "proposed" ? `Proposal ${d.proposal} recorded; nothing is trained until a person approves it` : `Not a proposal: ${d.reason}`); } catch (e) { Q.toast(e.message); }
        }, "", "A trigger is a proposal: a person approves it before anything is trained"]));
      }
      const row = h("div", "row tight", c), who = h("input", "", row), why = h("input", "", row);
      who.type = "text"; who.placeholder = "your name (an approval names a person)"; who.value = M.retrainActor || ""; who.setAttribute("aria-label", "Who decides"); why.type = "text"; why.placeholder = "why"; why.value = M.retrainNote || ""; why.setAttribute("aria-label", "Why");
      who.addEventListener("input", () => { M.retrainActor = who.value; }); why.addEventListener("input", () => { M.retrainNote = why.value; });
      const decide = (action) => async () => {
        try {
          const r = await api("/api/ml/retraining/action", { id: M.cur, action, proposal: st.open_proposal, actor: who.value.trim(), note: why.value.trim() }); const d = r.data || r;
          done(`${cap(d.outcome || action)}`);
        } catch (e) { Q.toast(e.message); }
      };
      actions(c, [
        ["Check the triggers now", () => W.request(M, () => [{ modes: ["controls_retrain_eval"], path: "/api/ml/retraining/action", body: { id: M.cur, action: "evaluate", refresh: true } }, ...CONTROLS(M.cur)], "Reading the schedule and the diagnostics…")],
        ["Propose a scheduled retraining", async () => { try { const r = await api("/api/ml/retraining/action", { id: M.cur, action: "trigger", kind: "schedule" }); const d = r.data || r; done(d.outcome === "proposed" ? `Proposal ${d.proposal} recorded` : `Not a proposal: ${d.reason}`); } catch (e) { Q.toast(e.message); } }],
        ["Approve the open proposal", decide("approve"), "", "Needs your name and a reason", !st.open_proposal],
        ["Reject it", decide("reject"), "", "Needs your name and a reason", !st.open_proposal],
        ["Run the challenger search", () => W.request(M, () => [{ modes: ["controls_retrain_start"], job: true, path: "/api/ml/retraining/action", body: { id: M.cur, action: "start", proposal: st.open_proposal, actor: who.value.trim() } }, ...CONTROLS(M.cur)],
          "Retraining the challengers on the same data and comparing them with the incumbent…"), "primary", "Development data only; nothing is promoted", !st.open_proposal || !approved],
      ]);
      if (ev) {
        W.sub(c, "Latest challenger search");
        W.kv(c, "Outcome", [["Decision", cap(snake(ev.decision.decision))], ["Why", ev.decision.reason], ["Data read", `${ev.scope.mode} mode, the validation segment; protected segments read: ${ev.scope.protected_segments_read ? "yes" : "no"}`],
          ["The re-run incumbent matches the saved one", ev.incumbent.reproduced.checked ? (ev.incumbent.reproduced.reproduced ? "yes" : "no: " + (ev.incumbent.reproduced.reason || "the data has changed since")) : "not checked"], ["Multiplicity", ev.search_space.multiplicity], ["Promoted", "no: " + ev.promotion]]);
        table(c, [{ label: "Challenger", get: (r) => r.name.replace(/^retrain: /, "") }, { label: "Verdict", get: (r) => snake(r.verdict || r.status) }, { label: "Sharpe, incumbent to challenger", get: (r) => `${sig(r.net_sharpe.incumbent)} → ${sig(r.net_sharpe.challenger)}` },
          { label: "Difference (interval)", get: (r) => { const d = r.net_sharpe.difference || {}; return Q.bad(d.value) || !d.interval ? "n/a" : `${sig(d.value)} (${sig(d.interval[0])} to ${sig(d.interval[1])})`; } },
          { label: "Same observations", get: (r) => (r.identical_oos.identical ? "yes" : "no") }, { label: "Adjusted p", num: true, get: (r) => sig(r.adjusted_p_value, 3) }], ev.challengers);
      }
      if ((rt.versions || []).length) {
        W.sub(c, "Versions, kept for rollback");
        table(c, [{ label: "Version", get: (r) => r.version_id }, { label: "Role", get: (r) => r.role }, { label: "Name", get: (r) => r.name }, { label: "State", get: (r) => snake(r.state) }], rt.versions);
        const rb = h("div", "row tight", c), from = h("select", "", rb), to = h("select", "", rb);
        for (const sel of [from, to]) for (const v of rt.versions) { const o = h("option", "", sel, `${v.role}: ${v.version_id}`); o.value = v.version_id; }
        from.setAttribute("aria-label", "Version to take out of service"); to.setAttribute("aria-label", "Version to return to"); if (rt.versions.length > 1) to.selectedIndex = 0;
        actions(c, [["Roll back", async () => {
          try { const r = await api("/api/ml/retraining/action", { id: M.cur, action: "rollback", from_version: from.value, to_version: to.value, actor: who.value.trim(), note: why.value.trim() }); const d = r.data || r; done(`${cap(snake(d.outcome || "rolled_back"))}`); } catch (e) { Q.toast(e.message); }
        }, "", "Takes the first version out of service and returns to the second; both are kept. Needs your name and a reason"]]);
      }
      W.hint(c, "Approval, rejection, recovery and rollback name a person: quant ml-advanced retrain-approve|reject|recover|rollback EXPERIMENT --proposal N --actor NAME --note TEXT, or POST /api/ml/retraining/action.");
    }
    W.sub(c, "Champion and challenger");
    const prow = h("div", "row tight", c), pin = h("input", "", prow), pbtn = h("button", "btn", prow, "Show the pair");
    pin.type = "text"; pin.placeholder = "pair id, for example cc-0123456789abcd"; pin.value = M.pairId || ""; pin.setAttribute("aria-label", "Champion and challenger pair id"); pbtn.type = "button";
    pbtn.addEventListener("click", () => { M.pairId = pin.value.trim(); W.request(M, () => [{ modes: ["controls_champion_run"], path: "/api/ml/champion", query: { id: M.pairId } }, ...CONTROLS(M.cur)], "Reading the pair…"); });
    const cp = (M.results.controls_champion_run || {}).data;
    if (cp && cp.pair_id === M.pairId) {
      const cn = (who) => { const x = ((cp.counts || {})[who] || {}).prospective || {}; return `${x.committed || 0} committed, ${x.settled || 0} settled`; };
      W.kv(c, "Pair", [["Champion", cp.champion], ["Challenger", cp.challenger], ["Horizon", String(cp.horizon)], ["Universe version", cp.universe_version], ["Spending", `${String(cp.rule).replace(/_/g, " ")} · alpha ${cp.alpha}`],
        ["Registered", `${cp.registered_at} by ${cp.actor}`], ["Data cutoff", cp.data_cutoff || "not declared"], ["Settled pairs", String(cp.settled_pairs)],
        ["Next look needs", cp.next_look_needs === null ? "none left" : `${cp.next_look_needs} settled pairs (${cp.next_look_available ? "available" : "not yet"})`], ["Verdict", String(cp.verdict).replace(/_/g, " ")],
        ["Prospective commitments", `champion ${cn("champion")}; challenger ${cn("challenger")}`]]);
      if (cp.looks_taken.length) table(c, [{ label: "Look", get: (r) => `${r.look} of ${r.of}` }, { label: "Pairs", num: true, get: (r) => String(r.pairs_used) }, { label: "Level", num: true, get: (r) => sig(r.level, 4) },
        { label: "Champion IC", num: true, get: (r) => sig(r.champion_ic, 4) }, { label: "Challenger IC", num: true, get: (r) => sig(r.challenger_ic, 4) }, { label: "Verdict", get: (r) => String(r.verdict).replace(/_/g, " ") }, { label: "Why", get: (r) => r.reason }], cp.looks_taken);
      W.hint(c, "A look is a statement about the evidence at that look and promotes nothing. Replays are listed and never counted.");
    } else W.hint(c, "A pair is registered before the observations it is judged on (quant ml-advanced champion-register, or POST /api/ml/champion/register). Enter its id to read its evidence.");
    const rp = a.reports;
    W.sub(c, "Reproducible reports");
    if (rp && rp.reports.length) {
      table(c, [{ label: "Report", get: (r) => r.report_id }, { label: "Built", get: (r) => r.built }, { label: "Snapshot taken", get: (r) => r.captured_at }, { label: "Hash", get: (r) => r.report_hash }], rp.reports.slice(0, 8));
      W.hint(c, "A stored report is rendered from its own snapshot and does not change when models or data are updated. Open one: GET /api/ml/report?report=ID&verify=1, or quant ml-advanced report-verify ID.");
    } else W.hint(c, "No report has been built for this experiment.");
  }

  /* ---------- research controls: the gates, the lineage, the holdout and the selection evidence of one experiment ---------- */
  const GATE_NAME = { A: "Algorithm Verified", B: "Strategy Validated", C: "Operationally Ready" };
  const GATE_DIR = { passed: "up", failed: "down", blocked: "down", pending: "" };
  const ITEM_WORD = { ok: "met", failed: "failed", missing: "not yet produced", stale: "stale", blocked: "blocked", not_applicable: "not applicable" };
  function buildControls(root, d) {
    const lin = (M.results.controls_lineage || {}).data, ho = (M.results.controls_holdout || {}).data, st = (M.results.controls_stats || {}).data, xd = (M.results.controls_exec || {}).data;
    const pol = (M.results.controls_policy || {}).data, gv = (M.results.controls_gov || {}).data;
    const al = (M.results.controls_alpha || {}).data, pf = (M.results.controls_port || {}).data, im = (M.results.controls_improve || {}).data, dl = (M.results.controls_delay || {}).data;
    const adv = { drift: (M.results.controls_drift || {}).data, monitor: (M.results.controls_monitor || {}).data, retrain: (M.results.controls_retrain || {}).data, reports: (M.results.controls_reports || {}).data };
    header(root, `Research record for ${shortId(d.experiment_id)}`, `${d.model_kind} · gate policy ${d.policy_version}`);
    W.provenance(root, d);
    window.QLGates.render(root, d, { reload: () => M.run(true), openMode: () => { const el = [...document.querySelectorAll("#ml-results .card")].find((c) => /Validation policy/.test(c.textContent)); if (el) el.scrollIntoView({ block: "start" }); } });          // the three gates: purpose, status, the exact obstacle, the next action (src/webapp/static/gates.js)
    if (pol) policyCard(root, pol);
    if (st && st.diagnostics) diagnosticsCard(root, st.diagnostics);
    if (gv) governanceCard(root, gv);
    if (al) alphaCard(root, al);
    if (pf) portfolioCard(root, pf);
    if (dl) delayCard(root, dl);
    if (adv.drift || adv.monitor || adv.retrain || adv.reports) advancedCard(root, adv);
    if (im) improvementCard(root, im);
    if (xd) {                                                                                  // the evaluation mode of the result, and what it can and cannot support
      const real = xd.mode === "realistic_cash_equity", fmtc = (v) => (Q.bad(v) ? "n/a" : Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 }));
      const c = W.card(root, `Execution mode: ${String(xd.mode).replace(/_/g, " ")}`, real ? `Orders, fills and the ledger produced this result${xd.parent ? ` (a re-run of ${shortId(xd.parent)})` : ""}; gross less the itemised costs reconciles to net.`
        : `A vectorized result: a cost per unit of turnover, with no orders, capacity or cash. ${xd.recorded === false ? xd.note || "" : ""}`);
      if (real) {
        W.kv(c, "Result", [["Gross P&L", fmtc(xd.gross_net.gross_pnl)], ["Itemised costs", fmtc(xd.gross_net.costs_total)], ["Net P&L", fmtc(xd.gross_net.net_pnl)], ["Reconciled", xd.reconciliation.reconciled && xd.reconciliation.ledger_ok ? "yes" : "NO"],
          ["Fills", String(xd.n_fills)], ["Turnover a year", sig(xd.ann_turnover, 1) + "x"], ["Lag", `${xd.timeline.delay_sessions} session(s), applied once`], ["Prices", xd.price_convention]]);
        table(c, [{ label: "Cost", get: (r) => r[0] }, { label: "Amount", num: true, get: (r) => fmtc(r[1]) }], Object.entries(xd.costs));
        if ((xd.capacity.levels || []).length) table(c, [{ label: "Capital", num: true, get: (r) => fmtc(r.capital) }, { label: "Share of the target filled", num: true, get: (r) => spct(r.fill_share, 0) },
          { label: "Impact (bps of traded)", num: true, get: (r) => sig(r.impact_bps_of_traded, 1) }], xd.capacity.levels);
        else W.hint(c, `Capacity is not established: ${xd.capacity.reason || ""}`);
      } else table(c, [{ label: "Not modelled by this engine", get: (r) => r }], xd.not_modelled || []);
      W.flags(c, xd.limitations || xd.cannot_support || [], "Limits. ");
    }
    const nl = W.card(root, "Models and stages", "A stage change is decided from the gates above; production also needs a recorded human approval.");
    if (!(d.models || []).length) W.hint(nl, "This experiment has no registered model.");
    else table(nl, [{ label: "Model", get: (m) => m.id }, { label: "Stage", get: (m) => m.stage }, { label: "Changes", num: true, get: (m) => String(m.history.length) }], d.models);
    if (lin && lin.operations) {
      const c = W.card(root, "Experiment lineage", `Study ${(lin.study || {}).study_id || "none"} · ${lin.raw_looks} tries recorded (${lin.failed} failed, ${lin.cancelled} cancelled) · ${lin.feature_sets.length} feature sets · ${lin.plans.length} evaluation plans. ${lin.note}`);
      if (!lin.operations.length) W.hint(c, "No try of this study is recorded (the experiment ran before lineage was kept).");
      else table(c, [{ label: "When", get: (o) => o.registered }, { label: "Kind", get: (o) => String(o.kind).replace(/_/g, " ") }, { label: "State", get: (o) => o.state }, { label: "Counted", get: (o) => (o.counts ? "yes" : "no") },
        { label: "Experiment", get: (o) => (o.experiment_id ? shortId(o.experiment_id) : "") }], lin.operations.slice(-40).reverse());
    }
    if (ho) {
      const c = W.card(root, "Holdout status", "A plan reserves the test and forward outcomes; its one evaluation consumes them. A run outside a plan is an exposure, and the history it read is no longer untouched.");
      if (!ho.plans.length && !ho.exposures.length) W.hint(c, "No plan and no exposure is recorded.");
      if (ho.plans.length) table(c, [{ label: "Plan", get: (p) => p.plan_id }, { label: "State", get: (p) => p.state }, { label: "Evaluations", num: true, get: (p) => String(p.evaluations) },
        { label: "Untouched", get: (p) => (p.reservation ? (p.reservation.untouched ? "yes" : "no: " + p.reservation.overlaps.length + " earlier look(s)") : "no reservation") },
        { label: "Dates", get: (p) => (p.reservation ? p.reservation.range.join(" to ") : "") }], ho.plans);
      if (ho.exposures.length) { W.sub(c, "Exposures (outcomes read outside a plan)"); table(c, [{ label: "When", get: (x) => x.at }, { label: "Experiment", get: (x) => (x.result_ref ? shortId(x.result_ref) : "") }, { label: "From", get: (x) => x.range_start || "" }, { label: "To", get: (x) => x.range_end || "" }], ho.exposures.slice(0, 15)); }
    }
    if (st) {
      const c = W.card(root, "Selection tests", st.statement);
      if (!st.latest.length) W.hint(c, "No selection test has been run for this study: POST /api/ml/selection-tests/run.");
      else table(c, [{ label: "Procedure", get: (e) => e.method }, { label: "Status", get: (e) => e.status }, { label: "Result", get: (e) => (e.status !== "applied" ? e.reason || "" : (e.statistic || {}).passes ? "criterion met: " + ((e.statistic || {}).criterion || "") : "criterion not met: " + ((e.statistic || {}).criterion || "")) }], st.latest);
      if ((st.null_studies || []).length) { W.sub(c, "Null controls (false-promotion rate on placebo data)"); table(c, [{ label: "Study", get: (n) => shortId(n.null_id) }, { label: "Replications", num: true, get: (n) => `${n.done}/${n.planned}` }, { label: "Verdict", get: (n) => (n.result || {}).verdict || n.state }], st.null_studies.slice(0, 6)); }
    }
  }

  /* ---------- the Training monitor ----------
     Everything shown is read from the run's files on the server (metrics.jsonl, the checkpoint folder, control.json), never from this page's memory: reloading, reopening in another browser or restarting the app shows the
     same history. While a run is going the page asks for the epochs it has not seen yet every two seconds; a run that stops reporting is shown as interrupted or orphaned, never as running. */
  M.tr = M.tr || { id: "", fit: "", data: null, series: null, timer: null, error: "", busy: false, view: "overview", grad: null, gradError: "", optimizer: "", flowEpoch: 0, flowMetric: "mean_abs", layerPick: new Set(), histPick: [] };
  const TR_WORD = { running: "Running", completed: "Completed", stopped_early: "Early stopping", stopped: "Stopped by you", paused: "Paused", interrupted: "Interrupted", orphaned: "Orphaned", unknown: "Unknown" };
  const TR_HINT = { running: "A new epoch was written a moment ago.", completed: "All epochs ran.", stopped_early: "Validation loss stopped improving; the best epoch's weights were kept.", stopped: "Stopped on request after a checkpoint.",
    paused: "Paused after a checkpoint: resume continues from there.", interrupted: "The process ended without finishing (the app was closed, or the training raised an error). Nothing was resumed automatically.",
    orphaned: "No end marker and no recent activity: the process may have been killed (for example by the system for lack of memory). The last checkpoint is intact if it validates below." };
  const RECOVERY_WORD = { exact: "Exact: same weights, optimizer, scheduler, random streams and early-stopping counters; the continued run follows the uninterrupted one on the same software and device.",
    epoch_boundary: "Epoch boundary: continues from the last finished epoch, but something needed for bit-for-bit continuation differs or is missing.",
    warm_start: "Warm start: weights only. The optimizer and counters start fresh, so the curve will not match an uninterrupted run.", restart_required: "Restart required: nothing safe can be restored." };
  const mb = (v) => (Q.bad(v) ? "n/a" : v >= 1024 ? (v / 1024).toFixed(2) + " GB" : Math.round(v) + " MB");
  const bytesText = (v) => (Q.bad(v) ? "n/a" : v >= 1e9 ? (v / 1e9).toFixed(2) + " GB" : v >= 1e6 ? (v / 1e6).toFixed(1) + " MB" : Math.round(v / 1e3) + " kB");
  const ago = (t) => { if (!t) return "n/a"; const s = Math.max(0, Date.now() / 1000 - t); return s < 90 ? Math.round(s) + " s ago" : s < 5400 ? Math.round(s / 60) + " min ago" : s < 172800 ? (s / 3600).toFixed(1) + " h ago" : Math.round(s / 86400) + " days ago"; };
  function stopTrainingPoll() { clearTimeout(M.tr.timer); M.tr.timer = null; }
  async function trainingPost(path, body, done) {
    try { const r = await api(path, body); if (done) done(r); return r; } catch (e) { Q.toast(e.message); return null; }
  }
  async function loadTrainingRun(id, fresh) {
    const s = M.tr;
    if (fresh) { s.id = id; s.series = null; s.data = null; s.fit = ""; }
    if (s.busy || !isOpen() || M.mode !== "training") return;
    s.busy = true;
    try {
      const d = await api(`/api/training/run?id=${encodeURIComponent(id)}${s.fit ? "&fit=" + encodeURIComponent(s.fit) : ""}`);
      if (s.id !== id) return;
      s.series = d.series;
      s.data = d; s.error = "";
    } catch (e) { s.error = e.message; }
    finally { s.busy = false; }
    paintTrainingDetail();
    if (s.data && !s.error) loadTrainingGradients();
    if (s.data && s.data.summary.status === "running" && !s.error) { stopTrainingPoll(); s.timer = setTimeout(() => loadTrainingRun(id, false), 2000); }
  }
  function paintTrainingDetail() { const box = Q.$("#ml-training-detail"); if (box && M.mode === "training") { box.replaceChildren(); trainingDetail(box); } }
  const TR_VIEWS = [["overview", "Overview"], ["gradient", "Gradient health"], ["layers", "Layer diagnostics"], ["dist", "Distributions"], ["events", "Recovery & events"]];
  const trUnit = () => { const u = (M.tr.data && M.tr.data.summary && M.tr.data.summary.unit) || "epoch"; return u; };
  const trUnitCap = () => cap(trUnit());
  const SEV_WORD = { info: "Info", warning: "Warning", critical: "Critical" };
  const LEVEL_NOTE = { off: "Gradient diagnostics are off for this run.", lightweight: "Lightweight: global norms, non-finite checks and clipping frequency only.", standard: "Standard: global statistics and per-layer summaries on sampled steps.",
    detailed: "Detailed: standard plus per-layer medians and histograms." };
  const fnum = (v, d = 3) => (Q.bad(v) ? "n/a" : (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) ? Number(v).toExponential(2) : String(+Number(v).toPrecision(d))));
  async function loadTrainingGradients() {
    const s = M.tr;
    if (!s.id || !s.data || !["gradient", "layers", "dist"].includes(s.view)) return;
    const q = [`id=${encodeURIComponent(s.id)}`, `fit=${encodeURIComponent(s.data.fit)}`];
    if (s.optimizer) q.push(`optimizer=${encodeURIComponent(s.optimizer)}`);
    if (s.flowEpoch) q.push(`flow_epoch=${s.flowEpoch}`);
    if (s.histPick.length) q.push(`hist_epochs=${s.histPick.join(",")}`);
    try { s.grad = await api("/api/training/gradients?" + q.join("&")); s.gradError = ""; } catch (e) { s.gradError = e.message; }
    paintTrainingDetail();
  }
  function trainingDetail(box) {
    const s = M.tr, d = s.data;
    if (s.error) { h("p", "hint", box, "Could not read the run: " + s.error); return; }
    if (!d) { Q.loading(box, "Reading the run…"); return; }
    const sm = d.summary, ser = s.series, n = ser.epoch.length, running = sm.status === "running";
    const head = h("div", "card", box);
    h("h3", "", head, `Run ${d.id}${d.fits.length > 1 ? ` · fit ${d.fit}` : ""}`);
    const st = h("p", "", head); const tag = h("strong", "", st, TR_WORD[sm.status] || sm.status); tag.setAttribute("data-status", sm.status); st.appendChild(document.createTextNode(" · " + (TR_HINT[sm.status] || "")));
    const cf = sm.checkpoint_failures;                                                           // a failed checkpoint write is the one thing here a person must not miss: it changes what a restart loses
    if (cf) {
      const b = h("div", "banner err", head); b.dataset.role = "checkpoint-warning"; b.setAttribute("role", "alert");
      h("b", "", b, cf.durable ? "Some checkpoints could not be written, but the run is now saved. " : "Checkpoints could not be written. ");
      b.appendChild(document.createTextNode(`${cf.count} write${cf.count === 1 ? "" : "s"} failed (${trUnit()}${cf.units.length === 1 ? "" : "s"} ${cf.units.join(", ")}): ${cf.last_error}. The latest completed ${trUnit()} is ${cf.latest_completed}; the latest saved checkpoint, the one a restart continues from, is ${cf.latest_durable || "none"}. ${cf.unsafe_units ? `If the process stops now, ${cf.unsafe_units} ${trUnit()}${cf.unsafe_units === 1 ? "" : "s"} of work will be redone. ` : ""}Training itself is not affected.`));
    }
    if (d.memory_advice) { const a = d.memory_advice; const p2 = h("p", "hint", head); p2.setAttribute("data-memory-advice", "1"); p2.appendChild(document.createTextNode(`Stopped for lack of memory (${a.reason}). ${a.batch_size ? `Try a batch size of ${a.batch_size}${a.accumulation_steps > 1 ? ` with ${a.accumulation_steps} accumulation steps` : ""}. ` : ""}${a.note || ""}`)); }
    const k = h("div", "earnrow", head);
    const last = (a) => { for (let i = (a || []).length - 1; i >= 0; i--) if (!Q.bad(a[i])) return a[i]; return null; };
    const pend = sm.uncommitted_epochs;
    tile(k, trUnitCap(), `${sm.epoch}${sm.epochs ? " / " + sm.epochs : ""}`, { text: `last written ${ago(sm.last_at)}` });
    if (sm.committed_epoch != null && (sm.status === "interrupted" || sm.status === "orphaned" || sm.status === "stopped" || sm.status === "paused")) tile(k, "Committed", String(sm.committed_epoch), { text: pend ? `${pend} ${trUnit()}${pend === 1 ? "" : "s"} logged after it will be recomputed on resume` : "a resume continues from here" });
    const tsv = ser.metrics && ser.metrics.timesteps ? last(ser.metrics.timesteps) : null, rw = ser.metrics && ser.metrics.mean_reward ? last(ser.metrics.mean_reward) : null;
    if (!Q.bad(tsv)) tile(k, "Timesteps", Number(tsv).toLocaleString(), { text: Q.bad(rw) ? "" : `mean episode reward ${fnum(rw, 3)}` });
    if (sm.phase) tile(k, "Phase", sm.phase === "adam" ? "Adam" : sm.phase === "lbfgs" ? "L-BFGS" : sm.phase, { text: ser.metrics && ser.metrics.lbfgs_iterations && !Q.bad(last(ser.metrics.lbfgs_iterations)) ? `${last(ser.metrics.lbfgs_iterations)} L-BFGS iterations` : ser.metrics && ser.metrics.adam_steps ? `${last(ser.metrics.adam_steps)} Adam steps` : "" });
    tile(k, "Best validation loss", Q.bad(sm.best_val) ? "n/a" : String(+Number(sm.best_val).toPrecision(4)), { text: Q.bad(sm.best_epoch) ? "" : `at epoch ${sm.best_epoch}` });
    tile(k, "Learning rate", n ? String(+Number(last(ser.lr)).toPrecision(3)) : "n/a", { text: "as the scheduler left it" });
    tile(k, "Process memory", n ? mb(last(ser.rss_mb)) : "n/a", { text: Q.bad(last(ser.peak_mb)) ? "" : `peak ${mb(last(ser.peak_mb))}` });
    tile(k, trUnitCap() + " time", n ? (+last(ser.seconds)).toFixed(1) + " s" : "n/a", { text: n ? `${(ser.seconds.reduce((a, b) => a + (b || 0), 0) / 60).toFixed(1)} min so far` : "" });
    const rec = (d.events || []).filter((e) => e.type === "resume").pop();
    if (rec) { const r = h("p", "hint", head); r.setAttribute("data-recovery", rec.level || ""); r.appendChild(document.createTextNode(`Resumed after epoch ${rec.epoch}. ${RECOVERY_WORD[rec.level] || ""}${(rec.reasons || []).length ? " " + rec.reasons.join("; ") + "." : ""}`)); }
    const row = [];
    if (running) row.push(["Stop after this epoch", () => trainingPost("/api/training/control", { id: d.id, fit: d.fit, action: "stop" }, () => { Q.toast("Stop requested: it takes effect after the current epoch's checkpoint"); loadTrainingRun(d.id, false); }), "", "Writes a checkpoint, then ends the run; it can be resumed"],
      ["Pause after this epoch", () => trainingPost("/api/training/control", { id: d.id, fit: d.fit, action: "pause" }, () => { Q.toast("Pause requested"); loadTrainingRun(d.id, false); }), "", "Writes a checkpoint and waits"]);
    else {
      const can = sm.resumable && d.request;
      row.push(["Resume this run", () => resumeRun(d.id, "resume"), "primary", can ? "Continue from the newest valid checkpoint" : "No valid checkpoint or no saved request", !can]);
      row.push(["Warm start from weights", () => resumeRun(d.id, "warm"), "", "Use only if the data window changed: optimizer and counters start fresh", !can]);
      row.push(["Restart as a new run", () => resumeRun(d.id, "restart"), "", "A new run from scratch with the same request", !d.request]);
      row.push(["Archive", () => trainingPost("/api/training/archive", { id: d.id }, () => { M.tr.id = ""; M.tr.data = null; M.run(true); }), "", "Moves it out of the list; its files are kept"]);
    }
    actions(head, row);
    if (d.control) h("p", "hint", head, `A "${d.control}" request is waiting for the end of the current epoch.`);
    const crit = (d.gradient_events || []).filter((e) => e.severity === "critical").length, warn = (d.gradient_events || []).filter((e) => e.severity === "warning").length;
    if (crit || warn) { const p3 = h("p", "hint", head); p3.setAttribute("data-gradient-alerts", "1"); p3.appendChild(document.createTextNode(`Gradient diagnostics raised ${crit ? crit + " critical" : ""}${crit && warn ? " and " : ""}${warn ? warn + " warning" : ""}${crit + warn === 1 ? "" : "s"}: see Gradient health and Recovery & events. Nothing was changed in the training because of them.`)); }
    const tabs = h("div", "tabsmall trtabs", box);
    for (const [v, text] of TR_VIEWS) {
      const b = h("button", "", tabs, text); b.type = "button"; b.dataset.trview = v; b.setAttribute("aria-selected", String(s.view === v));
      b.addEventListener("click", () => { s.view = v; s.grad = null; paintTrainingDetail(); loadTrainingGradients(); });
    }
    const body = h("div", "", box); body.id = "ml-training-view";
    if (!n && s.view !== "events") { h("p", "hint", body, "No epoch has finished yet."); return; }
    ({ overview: trOverview, gradient: trGradient, layers: trLayers, dist: trDist, events: trEvents })[s.view](body, d, ser, sm);
  }
  function trOverview(box, d, ser, sm) {
    const stamps = ser.metrics && ser.metrics.timesteps && ser.metrics.timesteps.every((v) => !Q.bad(v)) ? ser.metrics.timesteps : null;       // reinforcement learning: progress is in timesteps, not epochs
    const xs = stamps || ser.epoch, ch = h("div", "grid2", box), XL = stamps ? "timesteps" : trUnit();
    const phases = (ser.phase || []).some((p) => p === "lbfgs") && (ser.phase || []).some((p) => p === "adam");
    const switchAt = phases ? ser.epoch[ser.phase.indexOf("lbfgs")] : null;
    const lossSeries = phases ? [{ name: "Adam loss", color: css("--s1"), values: ser.train_loss.map((v, i) => (ser.phase[i] === "adam" ? v : null)), width: 1.5 }, { name: "L-BFGS loss", color: css("--s3"), values: ser.train_loss.map((v, i) => (ser.phase[i] === "lbfgs" ? v : null)), width: 1.5 }] : [{ name: "Training loss", color: css("--s1"), values: ser.train_loss, width: 1.5 }, ...(ser.val_loss.some((v) => !Q.bad(v)) ? [{ name: "Validation loss", color: css("--s2"), values: ser.val_loss, dash: "4 3" }] : [])];
    chartCard(ch, phases ? "Loss by optimisation phase" : `Loss by ${XL}`, "Training loss (solid) on the rows the network learns from and validation loss (dashed) on the later validation tail that early stopping watches. Test and forward results are never shown here.",
      { wide: true, draw: (b) => C.line(b, { xs, series: lossSeries, xLabel: XL, xFormat: (v) => String(Math.round(v)), yFormat: (v) => String(+v.toPrecision(3)), tipFormat: (v) => String(+v.toPrecision(4)), height: 260, label: "Training loss",
        vlines: phases ? [{ x: stamps ? stamps[ser.phase.indexOf("lbfgs")] : switchAt, label: "L-BFGS starts" }] : (sm.best_epoch && !stamps && ser.val_loss.some((v) => !Q.bad(v)) ? [{ x: sm.best_epoch, label: "best" }] : []) }),
        table: (b) => table(b, [{ label: stamps ? "Timesteps" : trUnitCap(), get: (i) => String(xs[i]) }, { label: "Phase", get: (i) => ser.phase[i] || "" }, { label: "Training loss", num: true, get: (i) => (Q.bad(ser.train_loss[i]) ? "" : num(ser.train_loss[i], 5)) }, { label: "Validation loss", num: true, get: (i) => (Q.bad(ser.val_loss[i]) ? "" : num(ser.val_loss[i], 5)) }], xs.map((_, i) => i)) });
    chartCard(ch, "Learning rate", "Falls by a factor each time validation loss fails to improve for the scheduler's patience.", { draw: (b) => C.line(b, { xs, series: [{ name: "Learning rate", color: css("--s3"), values: ser.lr }], xLabel: XL, xFormat: (v) => String(Math.round(v)), yFormat: (v) => v.toExponential(0), tipFormat: (v) => v.toExponential(2), height: 200, label: "Learning rate by epoch" }) });
    chartCard(ch, "Memory", "The training process's resident memory after each epoch, as the operating system reports it. A steady climb is a leak; a flat line is a network that fits.", { draw: (b) => C.line(b, { xs, series: [{ name: "Process memory (MB)", color: css("--s4"), values: ser.rss_mb }], xLabel: XL, xFormat: (v) => String(Math.round(v)), yFormat: (v) => String(Math.round(v)), tipFormat: (v) => Math.round(v) + " MB", height: 200, label: "Process memory by epoch" }) });
    const cpu = (ser.cpu_pct || []).some((v) => !Q.bad(v)), gpu = (ser.gpu_mb || []).some((v) => !Q.bad(v));
    if (cpu || gpu) chartCard(ch, "CPU and GPU", "Process CPU use over each epoch (100 is one core) and, when a GPU is present, the memory PyTorch holds on it.", { draw: (b) => C.line(b, { xs, series: [...(cpu ? [{ name: "CPU (% of one core)", color: css("--s5"), values: ser.cpu_pct }] : []), ...(gpu ? [{ name: "GPU memory (MB)", color: css("--s6"), values: ser.gpu_mb }] : [])], xLabel: trUnit(), xFormat: (v) => String(Math.round(v)), yFormat: (v) => String(Math.round(v)), tipFormat: (v) => String(Math.round(v)), height: 200, label: "CPU and GPU use by epoch" }) });
    const metricNames = Object.keys(ser.metrics || {}).filter((m) => ser.metrics[m].some((v) => !Q.bad(v)));
    for (const name of metricNames.filter((m) => m !== "timesteps")) chartCard(ch, name.replace(/_/g, " "), "A model-specific quantity the loop reports each unit.", { draw: (b) => C.line(b, { xs, series: [{ name, color: css("--s2"), values: ser.metrics[name] }], xLabel: XL, xFormat: (v) => String(Math.round(v)), yFormat: (v) => fnum(v, 2), tipFormat: (v) => fnum(v, 4), height: 180, label: name }) });
    const ck = h("div", "card", box); h("h3", "", ck, "Checkpoints");
    if (!d.checkpoints.length) h("p", "hint", ck, "No checkpoint has been written yet.");
    else table(ck, [{ label: "File", get: (c) => c.file }, { label: "Epoch", num: true, get: (c) => (Q.bad(c.epoch) ? "n/a" : String(c.epoch)) }, { label: "Validation loss", num: true, get: (c) => (Q.bad(c.val_loss) ? "n/a" : num(c.val_loss, 5)) },
      { label: "Size", num: true, get: (c) => bytesText(c.bytes) }, { label: "Written", get: (c) => ago(c.saved_at) }, { label: "Verified", get: (c) => (c.valid ? "yes" : "damaged: skipped on resume") }], d.checkpoints);
  }
  function gradMissing(box, g) {
    const s = M.tr;
    if (s.gradError) { h("p", "hint", box, "Could not read the gradient diagnostics: " + s.gradError); return true; }
    if (!g) { Q.loading(box, "Reading the gradient diagnostics…"); return true; }
    if (!g.epochs.length) { h("p", "hint", box, g.note || "This run recorded no gradient diagnostics (the monitoring level was off, or no epoch has finished since it was on)."); return true; }
    return false;
  }
  function optimizerPicker(box, g) {
    if (g.optimizers.length < 2) return;
    const p = h("p", "hint", box, "Optimizer: ");
    W.seg(p, g.optimizers.map((o) => [o, o]), g.optimizer, (v) => { M.tr.optimizer = v; M.tr.grad = null; paintTrainingDetail(); loadTrainingGradients(); });
  }
  function trGradient(box, d) {
    const g = M.tr.grad;
    if (gradMissing(box, g)) return;
    optimizerPicker(box, g);
    h("p", "hint", box, `${LEVEL_NOTE[g.level] || ""} Interval: ${g.interval}. Family thresholds: ${g.family}. Measured after the backward pass, with mixed precision unscaled, after accumulation and before clipping.`);
    const S = g.series, xs = S.epoch, tot = (S.nonfinite_steps || []).reduce((a, b) => a + (b || 0), 0), lastv = (a) => { for (let i = a.length - 1; i >= 0; i--) if (!Q.bad(a[i])) return a[i]; return null; };
    const k = h("div", "earnrow", box);
    tile(k, "Global L2 norm", fnum(lastv(S.l2_median)), { text: `largest step ${fnum(lastv(S.l2_max))}` });
    tile(k, "Largest entry", fnum(lastv(S.linf)), { text: "L-infinity norm (sampled steps)" });
    tile(k, "Clipped steps", Q.bad(lastv(S.clip_frequency)) ? "n/a" : pct(lastv(S.clip_frequency), 0), { text: Q.bad(lastv(S.clip_value)) ? "clipping off" : `clip value ${fnum(lastv(S.clip_value))}` });
    tile(k, "Non-finite steps", String(tot), { text: tot ? "see events" : "none seen", dir: tot ? "down" : "" });
    tile(k, "Zero gradients", Q.bad(lastv(S.zero_frac)) ? "n/a" : pct(lastv(S.zero_frac), 1), { text: Q.bad(lastv(S.near_zero_frac)) ? "" : `${pct(lastv(S.near_zero_frac), 1)} near zero` });
    const ch = h("div", "grid2", box);
    const vl = (d.gradient_events || []).filter((e) => e.severity !== "info" && e.optimizer === g.optimizer).map((e) => ({ x: e.epoch + 1, label: e.kind.replace(/_/g, " ") })).slice(0, 12);
    chartCard(ch, "Global gradient norm", "The L2 norm of the whole gradient at each optimizer step, summarised per epoch: the median, the 90th percentile and the largest step. Log scale. Dashed lines mark warnings.",
      { wide: true, draw: (b) => C.line(b, { xs, series: [{ name: "Median", color: css("--s1"), values: S.l2_median }, { name: "90th percentile", color: css("--s2"), values: S.l2_p90, width: 1 }, { name: "Largest", color: css("--s3"), values: S.l2_max, width: 1 }], log: true, xLabel: trUnit(), xFormat: (v) => String(Math.round(v)), yFormat: (v) => fnum(v, 2), tipFormat: (v) => fnum(v, 3), height: 260, vlines: vl, label: "Global gradient norm by epoch" }),
        table: (b) => table(b, [{ label: "Epoch", get: (i) => String(xs[i]) }, { label: "Median", num: true, get: (i) => fnum(S.l2_median[i]) }, { label: "P90", num: true, get: (i) => fnum(S.l2_p90[i]) }, { label: "Largest", num: true, get: (i) => fnum(S.l2_max[i]) }], xs.map((_, i) => i)) });
    if ((S.p50 || []).some((v) => !Q.bad(v))) chartCard(ch, "Gradient percentiles", "Percentiles of the absolute value of single gradient entries on the sampled steps (a fixed-stride subsample when the model is very large). Zero entries are included, so a percentile of zero means that share of entries is exactly zero. Log scale hides zeros: the table shows them.",
      { draw: (b) => C.line(b, { xs, series: [["p50", "P50", "--s1"], ["p90", "P90", "--s2"], ["p95", "P95", "--s3"], ["p99", "P99", "--s4"]].map(([key, name, c]) => ({ name, color: css(c), values: S[key] })), log: true, xLabel: trUnit(), xFormat: (v) => String(Math.round(v)), yFormat: (v) => fnum(v, 2), tipFormat: (v) => fnum(v, 3), height: 220, label: "Absolute-gradient percentiles by epoch" }),
        table: (b) => table(b, [{ label: "Epoch", get: (i) => String(xs[i]) }, ...["p50", "p90", "p95", "p99"].map((key) => ({ label: key.toUpperCase(), num: true, get: (i) => fnum(S[key][i]) }))], xs.map((_, i) => i)) });
    if ((S.zero_frac || []).some((v) => !Q.bad(v))) chartCard(ch, "Zero and near-zero gradients", "The share of gradient entries that are exactly zero and that are below the near-zero threshold. Structural zeros (padding, masks, ReLU units that are off for a batch) are normal; a layer that stays at one hundred percent is not.",
      { draw: (b) => C.line(b, { xs, series: [{ name: "Exactly zero", color: css("--s1"), values: S.zero_frac }, { name: "Near zero", color: css("--s2"), values: S.near_zero_frac, dash: "4 3" }], xLabel: trUnit(), xFormat: (v) => String(Math.round(v)), yFormat: (v) => pct(v, 0), tipFormat: (v) => pct(v, 2), height: 200, label: "Zero gradient share by epoch" }) });
    chartCard(ch, "Clipping and non-finite steps", "The share of optimizer steps whose gradient was clipped, and the count of steps with a non-finite gradient. The post-clip norm is measured separately on the sampled steps.",
      { draw: (b) => C.line(b, { xs, series: [{ name: "Clipped share", color: css("--s5"), values: S.clip_frequency }, { name: "Skipped by the mixed-precision scaler (share of steps)", color: css("--s6"), values: S.skipped_share }], xLabel: trUnit(), xFormat: (v) => String(Math.round(v)), yFormat: (v) => pct(v, 0), tipFormat: (v) => pct(v, 1), height: 200, label: "Clipping by epoch" }),
        table: (b) => table(b, [{ label: "Epoch", get: (i) => String(xs[i]) }, { label: "Clipped", num: true, get: (i) => (Q.bad(S.clip_frequency[i]) ? "off" : pct(S.clip_frequency[i], 1)) }, { label: "Post-clip norm", num: true, get: (i) => fnum(S.post_clip[i]) }, { label: "Non-finite steps", num: true, get: (i) => String(S.nonfinite_steps[i] || 0) }, { label: "Skipped steps", num: true, get: (i) => String(S.skipped_steps[i] || 0) }], xs.map((_, i) => i)) });
  }
  function trLayers(box, d) {
    const s = M.tr, g = s.grad;
    if (gradMissing(box, g)) return;
    optimizerPicker(box, g);
    if (!g.flow || !g.flow.rows.length) { h("p", "hint", box, "Layer-level diagnostics need the standard or detailed level; this run recorded " + g.level + "."); return; }
    const rows = g.flow.rows, flowEpoch = g.flow.epoch;
    const bar = h("p", "hint", box, "Epoch: ");
    const sel = h("select", "", bar); sel.dataset.role = "flow-epoch";
    for (const e of g.epochs) { const o = h("option", "", sel, String(e + 1)); o.value = String(e + 1); if (e + 1 === flowEpoch + 1) o.selected = true; }
    sel.addEventListener("change", () => { s.flowEpoch = Number(sel.value); loadTrainingGradients(); });
    h("span", "", bar, "  Show: ");
    W.seg(bar, [["mean_abs", "Mean |g|"], ["median_abs", "Median |g|"], ["max_abs", "Max |g|"], ["g_w", "Gradient / weight"], ["update_ratio", "Update / weight"]], s.flowMetric, (v) => { s.flowMetric = v; paintTrainingDetail(); });
    const names = rows.map((r) => r.name), xs = names.map((_, i) => i + 1);
    const zeros = rows.filter((r) => (r.zero_frac || 0) >= 0.999 || r.l2 === 0).map((r) => r.name), nonfinite = rows.filter((r) => r.nonfinite > 0).map((r) => r.name);
    const metric = s.flowMetric, vals = (key) => rows.map((r) => (Q.bad(r[key]) ? null : r[key]));
    const ch = h("div", "grid2", box);
    const lines = metric === "mean_abs" || metric === "median_abs" || metric === "max_abs" ? [["mean_abs", "Mean |g|", "--s1"], ["median_abs", "Median |g|", "--s2"], ["max_abs", "Max |g|", "--s3"]].filter(([key]) => rows.some((r) => !Q.bad(r[key]))) : [[metric, metric === "g_w" ? "Gradient / weight" : "Update / weight", "--s1"]];
    chartCard(ch, `Gradient flow through the model (epoch ${flowEpoch + 1})`, "Layers in depth order, input to output. Log scale. A layer whose line drops out of the chart has zero, missing or non-finite values at that point (they are not replaced by small numbers); they are listed below the chart.",
      { wide: true, draw: (b) => C.line(b, { xs, series: lines.map(([key, name, c]) => ({ name, color: css(c), values: vals(key), width: key === "max_abs" ? 1 : 2 })), log: true, xLabel: "layer (input to output)", xFormat: (v) => names[Math.round(v) - 1] || "", tipTitle: (i) => names[i] || "", yFormat: (v) => fnum(v, 2), tipFormat: (v) => fnum(v, 3), height: 280, label: "Gradient flow by layer" }),
        table: (b) => table(b, [{ label: "Layer", get: (i) => names[i] }, { label: "Parameters", num: true, get: (i) => String(rows[i].n_params || "") }, { label: "Mean |g|", num: true, get: (i) => fnum(rows[i].mean_abs) }, { label: "Max |g|", num: true, get: (i) => fnum(rows[i].max_abs) }, { label: "RMS", num: true, get: (i) => fnum(rows[i].rms) },
          { label: "Grad/weight", num: true, get: (i) => fnum(rows[i].g_w) }, { label: "Update/weight", num: true, get: (i) => fnum(rows[i].update_ratio) }, { label: "Zero share", num: true, get: (i) => (Q.bad(rows[i].zero_frac) ? "n/a" : pct(rows[i].zero_frac, 1)) }, { label: "Non-finite", num: true, get: (i) => String(rows[i].nonfinite || 0) }], names.map((_, i) => i)) });
    const flags = h("div", "card", ch); h("h3", "", flags, "What stands out");
    const rms = rows.map((r) => r.rms).filter((v) => !Q.bad(v) && v > 0), top = rms.length ? Math.max(...rms) : 0;
    const weak = rows.filter((r) => !Q.bad(r.rms) && r.rms > 0 && top > 0 && r.rms < 1e-3 * top).map((r) => r.name), strong = rows.filter((r) => !Q.bad(r.rms) && r.rms > 0 && top > 0 && r.rms === top).map((r) => r.name);
    const list = [[zeros, "exactly zero gradient"], [nonfinite, "non-finite gradient values"], [weak, "RMS gradient below a thousandth of the strongest layer's"]].filter(([l]) => l.length);
    if (!list.length) h("p", "hint", flags, "No layer has zero, non-finite or negligible gradients at this epoch.");
    for (const [l, text] of list) { const p = h("p", "", flags); p.setAttribute("data-flag", text); p.appendChild(document.createTextNode(`${l.slice(0, 8).join(", ")}${l.length > 8 ? ` and ${l.length - 8} more` : ""}: ${text}.`)); }
    if (strong.length) h("p", "hint", flags, `Strongest layer: ${strong[0]}. Frozen parameters are excluded from every figure.`);
    // comparison of chosen layers over the epochs
    const cmp = h("div", "card", box); h("h3", "", cmp, "Compare layers over training");
    const picks = s.layerPick.size ? [...s.layerPick] : names.slice(0, 1).concat(names.slice(-1)).filter((v, i, a) => a.indexOf(v) === i);
    const checks = h("div", "checks", cmp);
    for (const nme of names) { const lab = h("label", "f inline", checks), c = h("input", "", lab); c.type = "checkbox"; c.checked = picks.includes(nme); c.dataset.layer = nme; lab.appendChild(document.createTextNode(" " + nme)); c.addEventListener("change", () => { const next = new Set(picks); c.checked ? next.add(nme) : next.delete(nme); s.layerPick = next; paintTrainingDetail(); }); }
    const LS = g.layer_series, colors = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];
    const pickedNow = picks.filter((p) => LS[p]).slice(0, 6);
    if (pickedNow.length) {
      const mk = (key, title, note) => chartCard(cmp, title, note, { draw: (b) => C.line(b, { xs: g.series.epoch, series: pickedNow.map((nme, i) => ({ name: nme, color: css(colors[i % 6]), values: LS[nme][key] })), log: true, xLabel: trUnit(), xFormat: (v) => String(Math.round(v)), yFormat: (v) => fnum(v, 2), tipFormat: (v) => fnum(v, 3), height: 200, label: title }) });
      mk("rms", "RMS gradient by epoch", "Root-mean-square gradient entry of each chosen layer: comparable between layers of different sizes.");
      mk("g_w", "Gradient-to-weight ratio by epoch", "The layer's gradient norm over its weight norm.");
      mk("update_ratio", "Update-to-weight ratio by epoch", "How far the optimizer really moved the layer relative to its norm, measured on sampled steps.");
    }
  }
  function trDist(box) {
    const s = M.tr, g = s.grad;
    if (gradMissing(box, g)) return;
    optimizerPicker(box, g);
    const have = Object.keys(g.histograms).map(Number).sort((a, b) => a - b);
    if (!have.length) { h("p", "hint", box, g.level === "detailed" ? "No histogram has been recorded yet." : `Histograms are collected at the detailed level; this run used ${g.level}. Start the training with grad_monitor set to detailed to record them.`); return; }
    h("p", "hint", box, "Choose up to three epochs to compare. Each histogram counts gradient entries by the base-10 logarithm of their absolute value; exact zeros and non-finite values are counted apart, never folded into the smallest bin.");
    const picks = s.histPick.length ? s.histPick.filter((e) => have.includes(e - 1)) : [have[0] + 1, have[have.length - 1] + 1].filter((v, i, a) => a.indexOf(v) === i);
    const checks = h("div", "checks", box);
    for (const e of have) { const lab = h("label", "f inline", checks), c = h("input", "", lab); c.type = "checkbox"; c.checked = picks.includes(e + 1); c.dataset.epoch = String(e + 1); lab.appendChild(document.createTextNode(" epoch " + (e + 1)));
      c.addEventListener("change", () => { const next = new Set(picks); if (c.checked) { if (next.size >= 3) { c.checked = false; Q.toast("At most three epochs at a time"); return; } next.add(e + 1); } else next.delete(e + 1); s.histPick = [...next].sort((a, b) => a - b); loadTrainingGradients(); }); }
    const series = [], colors = ["--s1", "--s2", "--s3"];
    picks.slice(0, 3).forEach((e, i) => { const hst = g.histograms[String(e - 1)]; if (!hst) return; const w = (hst.hi - hst.lo) / hst.bins.length; series.push({ name: `epoch ${e}`, color: css(colors[i]), counts: hst.bins, edges: hst.bins.map((_, j) => hst.lo + j * w).concat([hst.hi]), zeros: hst.zeros, nonfinite: hst.nonfinite }); });
    const ch = h("div", "grid2", box);
    if (series.length) chartCard(ch, "Distribution of gradient magnitudes", "Density of log10 |gradient| over the sampled steps of each chosen epoch. A distribution that slides to the left over training is gradients shrinking; a long right tail is a few very large entries.",
      { wide: true, draw: (b) => C.histogram(b, { series, height: 260, xFormat: (v) => "1e" + Math.round(v), yFormat: (v) => fnum(v, 2), label: "Gradient magnitude histogram" }),
        table: (b) => table(b, [{ label: "Epoch", get: (x) => x.name }, { label: "Exact zeros", num: true, get: (x) => String(x.zeros) }, { label: "Non-finite", num: true, get: (x) => String(x.nonfinite) }, { label: "Entries counted", num: true, get: (x) => String(x.counts.reduce((a, b) => a + b, 0)) }], series) });
  }
  function trEvents(box, d, ser) {
    const ev = d.events || [], gev = d.gradient_events || [];
    const c1 = h("div", "card", box); h("h3", "", c1, "Recovery status");
    const rec = ev.filter((e) => e.type === "resume");
    if (!rec.length) h("p", "hint", c1, d.summary.status === "interrupted" || d.summary.status === "orphaned" ? "Interrupted: nothing has resumed it. Use Resume above to continue from the newest valid checkpoint." : "This run has not been resumed.");
    else for (const e of rec) { const p = h("p", "", c1); p.setAttribute("data-recovery", e.level || ""); p.appendChild(document.createTextNode(`${e.at ? new Date(e.at * 1000).toLocaleString() + ": " : ""}resumed after epoch ${e.epoch} (${e.level}). ${RECOVERY_WORD[e.level] || ""}${(e.reasons || []).length ? " " + e.reasons.join("; ") + "." : ""}`)); }
    const c2 = h("div", "card", box); h("h3", "", c2, "Timeline");
    const marks = [];
    for (const c of d.checkpoints) if (c.file === "best.pt" && c.epoch) marks.push({ at: c.saved_at, epoch: c.epoch, what: "best checkpoint", detail: `${c.file}, validation loss ${fnum(c.val_loss, 4)}` });
    for (const e of ev) { if (e.type === "start") continue; marks.push({ at: e.at, epoch: e.epoch, what: e.type.replace(/_/g, " "), detail: e.level ? `level ${e.level}` : (e.message || e.reason || "") }); }
    for (const e of gev) marks.push({ at: e.at, epoch: e.epoch + 1, what: `${e.kind.replace(/_/g, " ")} (${e.severity})`, detail: e.message });
    marks.sort((a, b) => (a.at || 0) - (b.at || 0));
    if (!marks.length) h("p", "hint", c2, "Nothing notable has happened: no interruption, resume, stop, resource warning or gradient warning.");
    else table(c2, [{ label: "When", get: (m) => (m.at ? new Date(m.at * 1000).toLocaleString() : "") }, { label: trUnitCap(), num: true, get: (m) => (Q.bad(m.epoch) ? "" : String(m.epoch)) }, { label: "Event", get: (m) => m.what }, { label: "Detail", get: (m) => m.detail || "" }], marks);
    const c3 = h("div", "card", box); h("h3", "", c3, "Gradient warnings");
    h("p", "hint", c3, "Warnings describe what was measured and the thresholds used. They never change the training: no learning rate, clip value or step was altered because of them.");
    if (!gev.length) h("p", "hint", c3, "No gradient warnings.");
    else table(c3, [{ label: "Severity", get: (e) => SEV_WORD[e.severity] || e.severity }, { label: "Kind", get: (e) => e.kind.replace(/_/g, " ") }, { label: "Epoch", num: true, get: (e) => String(e.epoch + 1) }, { label: "Step", num: true, get: (e) => (Q.bad(e.step) ? "" : String(e.step)) },
      { label: "Layers", get: (e) => (e.layers || []).join(", ") }, { label: "Learning rate", num: true, get: (e) => fnum(e.lr) }, { label: "Training", get: (e) => (e.action || "").replace(/_/g, " ") }, { label: "Measured", get: (e) => JSON.stringify(e.metrics).slice(0, 90) }, { label: "Thresholds", get: (e) => JSON.stringify(e.thresholds).slice(0, 90) }], gev);
    const warns = ev.filter((e) => e.type === "resource_warning");
    if (warns.length) { const c4 = h("div", "card", box); h("h3", "", c4, "Resource warnings"); table(c4, [{ label: "When", get: (e) => new Date(e.at * 1000).toLocaleString() }, { label: "Epoch", num: true, get: (e) => String(e.epoch) }, { label: "Message", get: (e) => e.message || "" }], warns); }
  }
  async function resumeRun(id, mode) {
    const r = await trainingPost("/api/training/resume", { id, mode });
    if (!r) return;
    Q.toast(mode === "restart" ? "A new run was queued" : "The run was queued and will continue from its checkpoint");
    M.tr.id = ""; M.tr.data = null; setTimeout(() => M.run(true), 1500);
  }
  function buildTraining(root, d) {
    header(root, "Training monitor", `${d.runs.length} run${d.runs.length === 1 ? "" : "s"} on disk · ${bytesText(d.disk.training_bytes)} of checkpoints and logs`);
    const k = h("div", "earnrow", root);
    const running = d.runs.filter((r) => r.status === "running").length, broken = d.runs.filter((r) => r.status === "interrupted" || r.status === "orphaned").length;
    tile(k, "Running now", String(running), { text: "training on this machine" });
    tile(k, "Interrupted or orphaned", String(broken), { text: broken ? "resume or archive from the table" : "none", dir: broken ? "down" : "" }, "", "Nothing is resumed automatically: you decide");
    tile(k, "Free memory", d.machine && d.machine.free_mb ? mb(d.machine.free_mb) : "n/a", { text: d.machine && d.machine.limit_mb ? `limit ${mb(d.machine.limit_mb)}` : "" });
    tile(k, "Free disk", mb(Q.bad(d.disk.disk_free_bytes) ? null : d.disk.disk_free_bytes / 1048576), { text: "for checkpoints" });
    if (d.storage_note) W.flags(root, [d.storage_note], "");
    if (!d.runs.length) { Q.labEmpty(root, M, EMPTY.training[0], EMPTY.training[1], null); return; }
    const c = h("div", "card", root); h("h3", "", c, "Runs");
    const wrap = h("div", "scroll", c), tb = h("table", "data", wrap), hr = h("tr", "", h("thead", "", tb));
    for (const [label, numeric] of [["Run"], ["Model"], ["Status"], ["Progress", 1], ["Best validation loss", 1], ["Last activity"], ["Size", 1], [""]]) h("th", numeric ? "num" : "", hr, label);
    const body = h("tbody", "", tb);
    for (const r of d.runs) {
      const tr = h("tr", "", body); tr.setAttribute("data-run", r.id);
      h("td", "", tr, r.id); h("td", "", tr, r.kind || ""); const st = h("td", "", tr, TR_WORD[r.status] || r.status); st.title = TR_HINT[r.status] || ""; st.setAttribute("data-status", r.status);
      h("td", "num", tr, `${r.epoch}${r.epochs ? " / " + r.epochs : ""} ${r.unit && r.unit !== "epoch" ? r.unit : ""}${r.phase ? " · " + r.phase : ""}`); h("td", "num", tr, Q.bad(r.best_val) ? "n/a" : String(+Number(r.best_val).toPrecision(4)));
      h("td", "", tr, ago(r.last_at)); h("td", "num", tr, bytesText(r.disk_bytes));
      const b = h("button", "btn", h("td", "", tr), M.tr.id === r.id ? "Close" : "Open"); b.type = "button";
      b.addEventListener("click", () => { if (M.tr.id === r.id) { stopTrainingPoll(); M.tr.id = ""; M.tr.data = null; M.draw(); return; } loadTrainingRun(r.id, true).then(() => 0); M.draw(); });
    }
    const box = h("div", "", root); box.id = "ml-training-detail";
    if (!M.tr.id && d.runs.length) { const first = d.runs.find((r) => r.status === "running") || d.runs[0]; loadTrainingRun(first.id, true); }
    else if (M.tr.id) loadTrainingRun(M.tr.id, false);
    paintTrainingDetail();
    if (running) M.tr.listTimer = setTimeout(() => { if (isOpen() && M.mode === "training") M.run(true); }, 5000);
  }

  Q.registerWorkspace("ml", M);
})();
