/* ML Strategy Explorer: what a model predicted, when the policy bought and sold, how it sized, why, and what that earned after costs.

   Everything on the page is read from a saved experiment that used the decision policy (src/ml/decisions.py). The three layers are kept apart in every view: the model's PREDICTION (a score and its rank on a date),
   the policy's DECISION (BUY / SELL / SHORT / COVER / HOLD ... with its reason, from that day's information only) and the EXECUTION (the trade the engine made later, at that day's close, at the configured cost).
   Nothing here is recomputed: a number on the page is the number the run recorded. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, toast, css } = Q, C = Q.C;
  const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.num(v, d)), pct = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.pct(v, d));
  const root = $("#strategy");
  const S = { runs: null, id: null, data: null, asset: null, decisions: null, trades: null, replay: null, busy: false, log: { action: "", asset: "", q: "", hold: false, offset: 0, limit: 50 }, tradeKind: "round_trips", tradeOffset: 0,
              form: { name: "", universe: "etf", tickers: "", model: "ridge", entry_quantile: 0.2, exit_quantile: 0.4, min_confidence: 0, max_positions: 5, max_position: 0.25, sizing: "equal", allow_short: false, long_exposure: 1, short_exposure: 0,
                      cash_floor: 0, stop_loss_vol: "", take_profit_vol: "", max_holding: "", cooldown: 5, rebalance_days: 5, resize_band: 0.25, cost_bps: 5, slippage_bps: 2, min_train_days: 756, simulate: false } };
  // markers: the shape and the colour say the same thing twice, so the chart does not rely on colour alone
  const KINDS = { OPEN_LONG: ["up", "--pos", "Buy (open long)"], ADD_LONG: ["up", "--pos", "Add to long"], CLOSE_SHORT: ["up", "--s2", "Cover (close short)"], TRIM_SHORT: ["up", "--s2", "Reduce short"], FLIP_TO_LONG: ["up", "--pos", "Cover and buy"],
                  CLOSE_LONG: ["down", "--neg", "Sell (close long)"], TRIM_LONG: ["down", "--neg", "Reduce long"], OPEN_SHORT: ["down", "--s5", "Short (open short)"], ADD_SHORT: ["down", "--s5", "Add to short"], FLIP_TO_SHORT: ["down", "--s5", "Sell and short"] };
  const STOPS = new Set(["STOP_LOSS", "TAKE_PROFIT", "TIME_EXIT"]);

  async function draw() {
    root.replaceChildren();
    const head = h("div", "head", root), l = h("div", "", head); h("h2", "", l, "ML Strategy Explorer");
    h("p", "sub", l, "What the model predicted, when the policy bought and sold, how big it went, why, and whether the decisions earned anything after costs. The prediction, the decision and the execution are three different things and are shown apart.");
    if (!S.runs) { try { S.runs = (await api("/api/ml/strategy/runs")).rows; } catch (e) { h("div", "banner err", root, "The strategy service is not available: " + e.message + " (restart the app if it was just updated)"); return; } if (!S.id && S.runs.length) S.id = S.runs[0].id; }
    builder();
    picker();
    if (!S.id) { const e = h("div", "card empty", root); h("h3", "", e, "No strategy yet"); h("p", "", e, "Run a strategy above. It trains the model walk-forward, turns its scores into explicit buy, sell and hold decisions, executes them with costs and saves the decision log."); return; }
    if (!S.data || S.data.id !== S.id || S.data._asset !== S.asset) {
      try { S.data = await api(`/api/ml/strategy?id=${encodeURIComponent(S.id)}${S.asset ? "&asset=" + encodeURIComponent(S.asset) : ""}`); S.data._asset = S.asset; S.asset = S.data.chart.asset; } catch (e) { h("div", "banner err", root, e.message); return; }
    }
    await view();
  }

  function builder() {
    const card = h("details", "card"); card.dataset.role = "strategy-builder"; root.appendChild(card);
    const sm = h("summary", "", card, "Build and run a strategy"); sm.style.cursor = "pointer";
    h("p", "note", card, "A model produces a score for every asset each day. The policy below turns scores into actions: it opens positions on the best-ranked names, keeps them until they fall out of a wider band (so it does not churn on a one-place change of rank), closes them on a stop, a target or a time limit, and sizes what it holds. Thresholds are ranks, so they mean the same for every model. Training is walk-forward and nothing about the policy is chosen on the evaluation period; the settings here are yours.");
    const grid = h("div", "params", card), f = S.form;
    const field = (label, key, type, help, opts) => { const l = h("label", "f", grid, label); if (help) l.title = help; let i; if (type === "select") { i = h("select", "", l); for (const [v, t] of opts) { const o = h("option", "", i, t); o.value = v; if (String(f[key]) === String(v)) o.selected = true; } } else { i = h("input", "", l); i.type = type; if (type === "number") i.step = "any"; if (type === "checkbox") i.checked = !!f[key]; else i.value = f[key]; }
      i.dataset.key = key; i.addEventListener(type === "select" || type === "checkbox" ? "change" : "input", () => { f[key] = type === "checkbox" ? i.checked : i.value; }); return i; };
    field("Name", "name", "text", "Optional label for the run");
    field("Universe", "universe", "select", "What the model scores and the policy trades. The platform's ETFs, Deribit crypto, the simulated futures roots, or tickers of your own", [["etf", "ETFs (platform data)"], ["crypto", "Crypto: BTC and ETH (platform data)"], ["futures", "Futures (simulated)"], ["custom", "My tickers"]]);
    field("My tickers (two or more, used when the universe is 'My tickers')", "tickers", "text", "For example GOOGL, AMZN, AAPL, META, NVDA. Prices come from the selected data source and are downloaded when they are not saved yet (a rate-limited source asks you to run a backtest with them once first)");
    field("Model", "model", "select", "The model whose scores are traded", [["ridge", "Ridge regression"], ["lasso", "Lasso"], ["elasticnet", "Elastic net"], ["random_forest", "Random forest"], ["hist_gb", "Gradient boosting"], ["xgboost", "XGBoost"], ["lightgbm", "LightGBM"], ["xgb_ranker", "XGBoost ranker"]]);
    field("Open on the best (share)", "entry_quantile", "number", "A name is bought when its score is in this top share of the day's ranks (0.2 = top 20%)");
    field("Keep until it falls below (share)", "exit_quantile", "number", "A held name is sold only when it drops out of this wider top share: the gap to the entry share is the hysteresis");
    field("Minimum confidence (0-1)", "min_confidence", "number", "The model's confidence needed to open a position");
    field("Positions at most", "max_positions", "number", "Per side");
    field("Largest position (weight)", "max_position", "number", "No single holding above this share of the book");
    field("Sizing", "sizing", "select", "Equal weights, weights inversely proportional to recent volatility, or weights growing with how extreme the rank is", [["equal", "Equal"], ["inverse_vol", "Inverse volatility"], ["score", "By rank"]]);
    field("Invested when full (weight)", "long_exposure", "number", "What is left is cash");
    field("Minimum cash (weight)", "cash_floor", "number", "A share of the book never invested");
    field("Allow short positions", "allow_short", "checkbox", "Selling a long and opening a short are different actions; shorts are off unless you turn them on");
    field("Short exposure (weight)", "short_exposure", "number", "Gross short weight when shorting is on");
    field("Stop loss (x volatility)", "stop_loss_vol", "number", "Close a position when its loss reaches this many times the entry-day volatility scaled to the holding horizon; blank for no stop");
    field("Take profit (x volatility)", "take_profit_vol", "number", "Close when the gain reaches this; blank for none");
    field("Longest holding (days)", "max_holding", "number", "Close after this many days; blank for no limit");
    field("Cooldown after a forced exit (days)", "cooldown", "number", "A name that was stopped out is not bought again for this long");
    field("Decide every (days)", "rebalance_days", "number", "Entries and rule exits happen on this schedule; stops and time exits are checked daily");
    field("Resize only beyond (share)", "resize_band", "number", "A held position is resized only when its target moves by more than this share of itself");
    field("Cost (bp, one way)", "cost_bps", "number", "Fee per unit traded");
    field("Slippage (bp)", "slippage_bps", "number", "Added to every trade");
    field("Training history before the first prediction (days)", "min_train_days", "number", "The model is first fitted on this many days; shorter histories give an earlier but noisier start");
    field("Choose the policy at each retrain from a grid, on earlier days only (scheduled simulation)", "simulate", "checkbox", "Retrains the model on a schedule and, at each refit, picks the best of six entry/stop policies using only the days before it. The model versions, the choices and their scores are saved and shown below.");
    const row = h("div", "row", card), go = h("button", "btn primary", row, "Run the strategy"); go.type = "button"; go.disabled = S.busy;
    const status = h("span", "hint", row, "Walk-forward training takes about a minute on the ETF universe; more assets or a heavier model take longer."); status.setAttribute("aria-live", "polite");
    go.addEventListener("click", async () => {
      const n = (v) => (v === "" || v === null || v === undefined ? undefined : Number(v));
      const sig = { method: "decision_policy", decision_entry_quantile: n(f.entry_quantile), decision_exit_quantile: n(f.exit_quantile), decision_min_confidence: n(f.min_confidence), decision_max_positions: n(f.max_positions), decision_max_position: n(f.max_position),
                    decision_sizing: f.sizing, decision_allow_short: !!f.allow_short, decision_long_exposure: n(f.long_exposure), decision_short_exposure: f.allow_short ? n(f.short_exposure) : 0, decision_cash_floor: n(f.cash_floor), decision_stop_loss_vol: n(f.stop_loss_vol),
                    decision_take_profit_vol: n(f.take_profit_vol), decision_max_holding: n(f.max_holding), decision_cooldown: n(f.cooldown), decision_rebalance_days: n(f.rebalance_days), decision_resize_band: n(f.resize_band) };
      for (const k of Object.keys(sig)) if (sig[k] === undefined) delete sig[k];
      const tickers = String(f.tickers || "").toUpperCase().split(/[\s,;]+/).filter(Boolean);
      if (f.universe === "custom" && tickers.length < 2) { toast("Enter at least two tickers."); return; }
      S.busy = true; go.disabled = true; status.textContent = "Starting…";
      try {
        const { job } = await api(f.simulate ? "/api/ml/simulate" : "/api/ml/train", { name: f.name.trim() || `${f.model} decision policy`, dataset: { universe: f.universe === "custom" ? tickers : f.universe }, model: { kind: f.model }, signal: sig, backtest: { rebalance: "on_change", target_vol: null, cost_bps: n(f.cost_bps), slippage_bps: n(f.slippage_bps) }, split: { min_train_days: n(f.min_train_days) }, explain: false });
        const res = await Q.waitJob(job, (v) => { status.textContent = `${v.message} (${Math.round(v.elapsed)} s)`; }, 1500);
        S.runs = null; S.id = res.id; S.data = null; S.asset = null; toast("Strategy finished"); S.busy = false; draw();
      } catch (e) { S.busy = false; go.disabled = false; status.textContent = ""; h("div", "banner err", card, e.message); }
    });
  }

  function picker() {
    if (!S.runs.length) return;
    const bar = h("div", "card"), l = h("label", "f", bar, "Strategy run"); root.appendChild(bar); bar.dataset.role = "strategy-picker";
    const sel = h("select", "", l); sel.dataset.key = "run";
    for (const r of S.runs) { const o = h("option", "", sel, `${r.name || r.id}  ·  ${r.model || ""}  ·  Sharpe ${num(r.sharpe, 2)}  ·  ${r.n_round_trips ?? "?"} trades`); o.value = r.id; if (r.id === S.id) o.selected = true; }
    sel.addEventListener("change", () => { S.id = sel.value; S.data = null; S.asset = null; S.decisions = null; S.trades = null; S.replay = null; S.log.offset = 0; S.tradeOffset = 0; draw(); });
  }

  async function view() {
    const d = S.data, s = d.summary, a = s.analytics || {}, out = h("div", "stack", root);
    const banner = h("div", "banner", out); banner.textContent = `Layers. Prediction: ${d.layers.prediction}. Decision: ${d.layers.decision}. Execution: ${d.layers.execution}.`;
    for (const w of d.warnings || []) h("div", "banner warn", out, w);
    const tiles = h("div", "tiles2", out); tiles.dataset.role = "strategy-tiles";
    const t = (label, value, title) => Q.tile(tiles, label, value, null, "", title);
    if (s.rule_based) t("Trades", `${s.n_executions ?? 0}`, "Every change of the book the position rule made: buys and sells, with the cost of each");      // a rule that holds every asset all the time has no round trips to win or lose: those tiles would read 100%
    else {
      t("Round trips", `${a.n_round_trips ?? 0}`, "Positions opened and closed, long and short"); t("Win rate", pct(a.win_rate, 0), "Share of closed positions with a positive return"); t("Profit factor", num(a.profit_factor, 2), "Gross profit over gross loss of the positions, net of their costs");
      t("Average win / loss", `${pct(a.avg_win, 1)} / ${pct(a.avg_loss, 1)}`, "Mean return of winning and of losing positions"); t("Average holding", `${num(a.avg_holding_days, 1)} d`, "Median " + num(a.median_holding_days, 1) + " days");
    }
    t("Invested", pct(a.avg_gross_exposure, 0), "Average gross exposure; the rest is cash: " + pct(a.avg_cash, 0)); t("Net exposure", pct(a.avg_net_exposure, 0)); t("Positions held", num(a.avg_positions, 1));
    t("Turnover / year", `${num(a.ann_turnover, 1)}x`); t("Cost drag / year", pct(a.cost_drag_ann, 2), "Gross minus net annual return"); t("Gross return / year", pct(a.gross_ann_return, 1)); t("Net return / year", pct(a.net_ann_return, 1));
    if (s.note) h("p", "note", out, s.note);
    if (d.simulation) simulationCard(out, d.simulation);
    priceCard(out); bookCards(out); statCards(out);
    Q.howCard(out, S.id); controlCard(out); if (!s.rule_based) await logCard(out); await tradesCard(out); await windowCard(out); await replayCard(out);                   // a rule without a decision log has no log to show
  }

  function simulationCard(out, sim) {
    const card = h("div", "card", out); card.dataset.role = "simulation-card"; h("h3", "", card, "Scheduled retraining: model versions and policy choices");
    h("p", "note", card, `The model was refitted ${sim.model_versions.length} times. At each refit the decision policy was chosen from ${sim.selection.n_candidates} candidates using only the ${sim.selection.lookback_days} days that ended ${sim.selection.embargo_days} days before it. ` +
      `${sim.oos_violations.length ? "The framework found predictions made by a model that had seen their date." : "The framework found no prediction made by a model that had seen its date."}`);
    const comp = sim.comparison; if (comp) h("p", "", card, `Scheduled Sharpe ${num(comp.scheduled.sharpe, 2)} over ${comp.scheduled.days} days. Each fixed policy held for the whole run: ` + comp.fixed.map((x) => `${x.id} ${num(x.sharpe, 2)}`).join("; ") + ". " + comp.reading);
    h("h4", "", card, "Model versions"); Q.simpleTable(card, [{ label: "Version", get: (x) => x.version }, { label: "Trained through", get: (x) => x.fit_cutoff }, { label: "Labels reached", get: (x) => x.label_end_max }, { label: "Predicts", get: (x) => `${x.predicts_from} to ${x.predicts_to}` }, { label: "Days", num: true, get: (x) => String(x.n_days) },
      { label: "Rows hash", get: (x) => x.train_rows_hash }, { label: "Model / params", get: (x) => `${x.model} ${x.params_hash}` }], sim.model_versions);
    h("h4", "", card, "Policy chosen at each refit"); Q.simpleTable(card, [{ label: "From", get: (x) => x.from }, { label: "Chosen", get: (x) => x.chosen }, { label: "Sharpe it was chosen on", num: true, get: (x) => num(x.score, 2) }, { label: "Days used", num: true, get: (x) => String(x.days_used) }, { label: "Basis", get: (x) => x.basis }], sim.choices);
    h("p", "hint", card, `Reproducibility: configuration ${sim.reproducibility.config_hash}, dataset ${sim.reproducibility.dataset_key}, seed ${sim.reproducibility.seed}.`);
  }

  function controlCard(out) {
    const card = h("div", "card", out); card.dataset.role = "control-card"; h("h3", "", card, "Is this better than random signals?");
    h("p", "note", card, "The same decision policy, costs and pace of trading, fed random scores that have the model's own rank persistence and no information about returns. If the model's Sharpe ratio is not clearly above these, nothing about its predictions has been shown. A matched benchmark (the market scaled by the book's own beta) is shown beside it.");
    const row = h("div", "row", card), go = h("button", "btn", row, "Run the random-signal control"), status = h("span", "hint", row, ""), body = h("div", "", card); go.type = "button"; body.setAttribute("aria-live", "polite");
    const show = (r) => {
      body.replaceChildren(); const t = h("div", "tiles2", body);
      Q.tile(t, "Model Sharpe", num(r.model_sharpe, 2)); Q.tile(t, "Random signals: median", num(r.random.p50, 2), null, "", `5th to 95th percentile: ${num(r.random.p05, 2)} to ${num(r.random.p95, 2)}`);
      Q.tile(t, "Random draws as good", pct(r.share_of_random_at_least_as_good, 0), null, "", `${r.draws} draws`); Q.tile(t, "Rank persistence", num(r.rank_persistence, 2), null, "", "The model's day-to-day rank autocorrelation, copied into the random scores");
      Q.tile(t, "Beta-matched benchmark Sharpe", num(r.matched_benchmark.matched_sharpe, 2), null, "", `Mean beta ${num(r.matched_benchmark.beta_mean, 2)}; active Sharpe over it ${num(r.matched_benchmark.active_sharpe, 2)}`);
      if (r.replay_check && !r.replay_check.reproduces) h("div", "banner warn", body, `The replay did not reproduce the saved run (saved Sharpe ${num(r.replay_check.saved_sharpe, 2)}, replayed ${num(r.replay_check.replayed_sharpe, 2)}), so read this control with care.`);
      h("p", "hint", body, r.reading);
    };
    go.addEventListener("click", async () => {
      go.disabled = true; status.textContent = "Starting…";
      try { const { job } = await api("/api/ml/strategy/controls", { id: S.id, draws: 40 }); show(await Q.waitJob(job, (v) => { status.textContent = `${v.message} (${Math.round(v.elapsed)} s)`; }, 1000)); status.textContent = ""; }
      catch (e) { status.textContent = ""; h("div", "banner err", body, e.message); }
      go.disabled = false;
    });
  }

  function priceCard(out) {
    const d = d_(), ch = d.chart;
    const pick = h("div", "card", out); pick.dataset.role = "asset-picker";
    const l = h("label", "f", pick, "Asset"), sel = h("select", "", l); sel.dataset.key = "asset";
    for (const r of d.assets) { const o = h("option", "", sel, `${r.asset}  ·  ${r.executions} trades  ·  ${r.round_trips} round trips  ·  P&L ${pct(r.net_contribution, 2)}`); o.value = r.asset; if (r.asset === ch.asset) o.selected = true; }
    sel.addEventListener("change", () => { S.asset = sel.value; S.log.asset = sel.value; S.log.offset = 0; S.decisions = null; S.trades = null; draw(); });
    tradeChart(out, ch, d.book);
  }

  /* The chart, the position and the table of one asset's trades. Shared by the Explorer and by the Machine Learning Lab's report (Q.tradesCard), so a buy or a sell looks the same on both. */
  function tradeChart(out, ch, book) {
    Q.chartCard(out, `${ch.asset}: price, position and trades`, "One date axis for the price, the trades and the position: drag on the chart (mouse) or on the strip below it to zoom and pan, and the three move together. A decision is made from the close of one day and executed later, so a marker sits on the day the trade was made.", {
      wide: true, draw: (b) => { const host = h("div", "", b); host.dataset.role = "trade-panel"; C.tradePanel(host, { dates: ch.dates, close: ch.close, weight: ch.weight, markers: ch.markers, asset: ch.asset, summary: ch.summary, book }); },
      table: (b) => Q.simpleTable(b, [{ label: "Date", get: (x) => x.date }, { label: "Trade", get: (x) => (KINDS[x.kind] || [0, 0, x.kind])[2] }, { label: "Price", num: true, get: (x) => num(x.price, 2) }, { label: "Change", num: true, get: (x) => pct(x.weight_change, 2) }, { label: "Book after", num: true, get: (x) => pct(x.weight_after, 2) },
        { label: "Cost", num: true, get: (x) => pct(x.cost, 3) }, { label: "P&L", num: true, get: (x) => (x.pnl_return === null || x.pnl_return === undefined ? "" : pct(x.pnl_return, 1)) }, { label: "Decided", get: (x) => x.decision_date || "" }, { label: "Why", get: (x) => x.reason || "" },
        { label: "Score rank", num: true, get: (x) => (x.rank_pct === null || x.rank_pct === undefined ? "" : pct(x.rank_pct, 0)) }], ch.markers) });
  }

  /* "How this model decides": what it predicts, which indicators it reads and leans on, how its score becomes a buy or a sell, and what keeps the test honest. Shared by the Explorer and the Lab's report. */
  Q.howCard = (parent, expId) => {
    const card = h("div", "card", parent); card.dataset.role = "how-it-decides"; h("h3", "", card, "How this model decides to buy or sell");
    const body = h("div", "", card); Q.loading(body, "Reading the run…");
    api(`/api/ml/strategy/how?id=${encodeURIComponent(expId)}`).then((d) => {
      body.replaceChildren();
      h("p", "note", body, `It is not fitting the price. It predicts ${d.predicts}, from ${d.n_features} indicators, and trades on that.`);
      h("h4", "", body, "From data to a trade"); const ol = h("ol", "steps", body); for (const t of d.steps) h("li", "", ol, t);
      h("h4", "", body, "The indicators it reads"); const g = h("div", "howgroups", body);
      for (const x of d.groups) { const c = h("div", "howgroup", g); h("b", "", c, `${x.group.replace(/_/g, " ")} (${x.n})`); h("p", "hint", c, x.what); const ul = h("ul", "", c); for (const e of x.examples) { const li = h("li", "", ul); h("code", "", li, e.name); li.append(e.meaning ? ` — ${e.meaning}` : ""); } }
      if (d.leaning_on.length) {
        h("h4", "", body, "What the model leaned on most"); h("p", "hint", body, "The indicators that moved its scores the most, as a share of the total (permutation and fold importance from the run). A model leaning on one indicator is a model to distrust.");
        Q.simpleTable(body, [{ label: "Indicator", get: (r) => r.name }, { label: "Group", get: (r) => (r.group || "").replace(/_/g, " ") }, { label: "Share", num: true, get: (r) => pct(r.share, 1) }, { label: "What it measures", get: (r) => r.meaning || "" }], d.leaning_on);
      } else h("p", "hint", body, "This run did not compute feature importance (train with explanations on to see which indicators it leaned on).");
      h("h4", "", body, "The model and the rule"); h("p", "", body, `${d.model.label}${d.model.text ? ": " + d.model.text : ""} Re-fitted every ${d.model.refit_every ?? "?"} days on ${d.model.window || "past"} data.`);
      h("p", "", body, `Position rule '${d.rule.method}': ${d.rule.text}.`);
      h("h4", "", body, "Why this is not a curve fit"); const ul = h("ul", "next", body); for (const t of d.guards) h("li", "", ul, t);
    }).catch((e) => { body.replaceChildren(); h("p", "hint", body, "No explanation is available for this run: " + e.message); });
    return card;
  };

  /* "Buys and sells" for a saved experiment of any position rule, drawn inside another page (the Machine Learning Lab's report). Loads the run's trades from the Strategy API, one asset at a time. */
  Q.tradesCard = (parent, expId, openExplorer) => {
    const card = h("div", "card", parent); card.dataset.role = "lab-trades"; h("h3", "", card, "Buys and sells");
    const body = h("div", "", card); Q.loading(body, "Loading the trades…");
    const load = async (asset) => {
      let d;
      try { d = await api(`/api/ml/strategy?id=${encodeURIComponent(expId)}${asset ? "&asset=" + encodeURIComponent(asset) : ""}`); }
      catch (e) { body.replaceChildren(); h("p", "hint", body, "No trades are recorded for this run: " + e.message); return; }
      body.replaceChildren();
      const t = d.summary || {}, a = t.analytics || {};
      h("p", "note", body, t.note || "The trades the engine made, each at the close of the day after the decision, at the configured cost.");
      const tiles = h("div", "tiles2", body);
      Q.tile(tiles, "Trades", String(t.n_executions ?? 0)); Q.tile(tiles, "Turnover / year", num(a.ann_turnover, 1) + "×"); Q.tile(tiles, "Cost drag / year", pct(a.cost_drag_ann, 2)); Q.tile(tiles, "Invested", pct(a.avg_gross_exposure, 0));
      if (!t.rule_based) { Q.tile(tiles, "Round trips", String(a.n_round_trips ?? 0)); Q.tile(tiles, "Win rate", pct(a.win_rate, 0)); }
      const l = h("label", "f", body, "Asset"), sel = h("select", "", l); sel.dataset.key = "asset";
      for (const r of d.assets) { const o = h("option", "", sel, `${r.asset}  ·  ${r.executions} trades  ·  P&L ${pct(r.net_contribution, 2)}`); o.value = r.asset; if (r.asset === d.chart.asset) o.selected = true; }
      sel.addEventListener("change", () => load(sel.value));
      tradeChart(body, d.chart, d.book);
      if (openExplorer) { const row = h("div", "row", body), go = h("button", "btn", row, "Open in the ML Strategy Explorer"); go.type = "button"; go.addEventListener("click", () => openExplorer(expId)); }
    };
    load("");
    return card;
  };
  const d_ = () => S.data;

  function bookCards(out) {
    const b = d_().book, c = [css("--s1"), css("--s6")];
    Q.chartCard(out, "Equity against the benchmark", "Growth of 1 after costs, from the first trade. The benchmark is the daily-rebalanced equal-weight universe: a long-only, fully invested comparison. A strategy that holds cash or shorts is not like-for-like with it, which the Model Comparison view corrects with beta- and exposure-matched benchmarks.", {
      wide: true, draw: (x) => C.line(x, { dates: b.dates, series: [{ name: "Strategy (net)", color: c[0], values: b.net }, { name: "Equal-weight benchmark", color: c[1], values: b.benchmark }], yFormat: (v) => num(v, 2), height: 260, label: "Equity", baseline: 1 }),
      table: (x) => Q.simpleTable(x, [{ label: "Date", get: (r) => r.d }, { label: "Strategy", num: true, get: (r) => num(r.n, 3) }, { label: "Benchmark", num: true, get: (r) => num(r.b, 3) }], b.dates.map((dt, i) => ({ d: dt, n: b.net[i], b: b.benchmark[i] })).filter((_, i) => i % 5 === 0)) });
    Q.chartCard(out, "Drawdown", "Distance below the previous peak, after costs.", { draw: (x) => C.line(x, { dates: b.dates, series: [{ name: "Drawdown", color: css("--neg"), values: b.drawdown }], area: true, baseline: 0, yFormat: (v) => pct(v, 0), height: 180, label: "Drawdown" }),
      table: (x) => Q.simpleTable(x, [{ label: "Date", get: (r) => r.d }, { label: "Drawdown", num: true, get: (r) => pct(r.v, 2) }], b.dates.map((dt, i) => ({ d: dt, v: b.drawdown[i] })).filter((_, i) => i % 5 === 0)) });
    Q.chartCard(out, "Exposure and cash", "How much of the book was invested and how much sat in cash, and the number of positions held.", { draw: (x) => C.line(x, { dates: b.dates, series: [{ name: "Invested (gross)", color: css("--s1"), values: b.gross }, { name: "Cash", color: css("--s4"), values: b.cash }], yFormat: (v) => pct(v, 0), height: 180, label: "Exposure" }),
      table: (x) => Q.simpleTable(x, [{ label: "Date", get: (r) => r.d }, { label: "Invested", num: true, get: (r) => pct(r.g, 1) }, { label: "Cash", num: true, get: (r) => pct(r.c, 1) }, { label: "Positions", num: true, get: (r) => String(r.p) }], b.dates.map((dt, i) => ({ d: dt, g: b.gross[i], c: b.cash[i], p: b.positions[i] })).filter((_, i) => i % 5 === 0)) });
  }

  function statCards(out) {
    const a = d_().summary.analytics || {}, hist = a.holding_histogram || [], by = a.by_asset || [];
    if (hist.length) Q.chartCard(out, "How long positions were held", "Closed positions by days held.", { draw: (x) => C.columns(x, { labels: hist.map((r) => r.label), values: hist.map((r) => r.count), yFormat: (v) => String(Math.round(v)), label: "Holding periods", height: 200 }),
      table: (x) => Q.simpleTable(x, [{ label: "Days", get: (r) => r.label }, { label: "Positions", num: true, get: (r) => String(r.count) }], hist) });
    if (by.length) Q.chartCard(out, "Where the profit and loss came from", "Net contribution to the book by asset (sum of the positions' weight x return less their costs), and each asset's win rate.", {
      draw: (x) => C.columns(x, { labels: by.map((r) => r.asset), values: by.map((r) => r.contribution), yFormat: (v) => pct(v, 1), label: "Contribution by asset", height: 220 }),
      table: (x) => Q.simpleTable(x, [{ label: "Asset", get: (r) => r.asset }, { label: "Round trips", num: true, get: (r) => String(r.trips) }, { label: "Win rate", num: true, get: (r) => pct(r.win_rate, 0) }, { label: "Avg days held", num: true, get: (r) => num(r.avg_holding, 1) }, { label: "Net contribution", num: true, get: (r) => pct(r.contribution, 2) }], by) });
  }

  async function logCard(out) {
    const card = h("div", "card", out); card.dataset.role = "decision-log"; h("h3", "", card, "Decision log");
    h("p", "note", card, "Every decision with the information it was made from: the model's score and rank that day, the rule that fired, and the book before and after. A decision is made from the close of its date and executed later. Holds are listed on request.");
    const f = S.log, bar = h("div", "params", card);
    const input = (label, key, type, opts) => { const l = h("label", "f", bar, label); let i; if (opts) { i = h("select", "", l); for (const [v, t] of opts) { const o = h("option", "", i, t); o.value = v; if (f[key] === v) o.selected = true; } } else { i = h("input", "", l); i.type = type; if (type === "checkbox") i.checked = f[key]; else i.value = f[key]; } i.dataset.key = key;
      i.addEventListener(type === "checkbox" ? "change" : (opts ? "change" : "input"), () => { f[key] = type === "checkbox" ? i.checked : i.value; f.offset = 0; clearTimeout(S._t); S._t = setTimeout(reload, 250); }); return i; };
    input("Action", "action", "text", [["", "All decisions"], ["BUY", "Buy"], ["SELL", "Sell"], ["SHORT", "Short"], ["COVER", "Cover"], ["INCREASE,REDUCE", "Resize"], ["STOP_LOSS,TAKE_PROFIT,TIME_EXIT", "Forced exits"]]);
    input("Asset", "asset", "text", [["", "All assets"], ...d_().assets.map((r) => [r.asset, r.asset])]); input("Search the reasons", "q", "search"); input("Include holds", "hold", "checkbox");
    const body = h("div", "", card); body.setAttribute("aria-live", "polite");
    async function reload() {
      const p = new URLSearchParams({ id: S.id, limit: f.limit, offset: f.offset }); if (f.action) p.set("action", f.action); if (f.asset) p.set("asset", f.asset); if (f.q) p.set("q", f.q); if (f.hold) p.set("hold", "1");
      let r; try { r = await api("/api/ml/strategy/decisions?" + p); } catch (e) { body.replaceChildren(); h("div", "banner err", body, e.message); return; }
      S.decisions = r; body.replaceChildren();
      h("p", "hint", body, `${r.n} decisions` + (Object.keys(r.counts).length ? "  ·  " + Object.entries(r.counts).map(([k, v]) => `${k} ${v}`).join(", ") : ""));
      Q.simpleTable(body, [{ label: "Date", get: (x) => x.date }, { label: "Asset", get: (x) => x.asset }, { label: "Decision", get: (x) => x.action }, { label: "Prediction (score)", num: true, get: (x) => num(x.score, 4) }, { label: "Rank", num: true, get: (x) => pct(x.rank_pct, 0) },
        { label: "Confidence", num: true, get: (x) => pct(x.confidence, 0) }, { label: "Book before", num: true, get: (x) => pct(x.prev_weight, 1) }, { label: "Target", num: true, get: (x) => pct(x.target_weight, 1) }, { label: "Close", num: true, get: (x) => num(x.price, 2) },
        { label: "Entry price", num: true, get: (x) => num(x.entry_price, 2) }, { label: "Held (d)", num: true, get: (x) => String(x.holding_days) }, { label: "Unrealised", num: true, get: (x) => pct(x.unrealised_return, 1) }, { label: "Why", get: (x) => x.reason }], r.rows);
      pager(body, r.n, f.offset, f.limit, (o) => { f.offset = o; reload(); });
    }
    await reload();
  }

  function pager(parent, n, offset, limit, go) {
    if (n <= limit) return;
    const row = h("div", "row", parent), prev = h("button", "btn small", row, "Earlier"), nxt = h("button", "btn small", row, "Later"); prev.type = nxt.type = "button";
    h("span", "hint", row, `${offset + 1}-${Math.min(n, offset + limit)} of ${n}`);
    prev.disabled = offset <= 0; nxt.disabled = offset + limit >= n; prev.addEventListener("click", () => go(Math.max(0, offset - limit))); nxt.addEventListener("click", () => go(offset + limit));
  }

  async function tradesCard(out) {
    const card = h("div", "card", out); card.dataset.role = "trades"; h("h3", "", card, "Trades and round trips");
    const seg = h("div", "seg", card);
    for (const [k, label] of [["round_trips", "Round trips"], ["executions", "Executed trades"]]) { const b = h("button", "", seg, label); b.type = "button"; b.setAttribute("aria-pressed", String(S.tradeKind === k)); b.addEventListener("click", () => { S.tradeKind = k; S.tradeOffset = 0; draw(); }); }
    h("p", "note", card, S.tradeKind === "round_trips" ? "Each position from its opening to its closing: its own return, its contribution to the book, what it cost, and why it was closed." : "Each trade the engine made, linked to the decision behind it. A line marked DRIFT is a position put back to its target on a day the book was being traded anyway.");
    let r; try { r = await api(`/api/ml/strategy/trades?id=${encodeURIComponent(S.id)}&kind=${S.tradeKind}&limit=50&offset=${S.tradeOffset}`); } catch (e) { h("div", "banner err", card, e.message); return; }
    if (S.tradeKind === "round_trips") Q.simpleTable(card, [{ label: "Asset", get: (x) => x.asset }, { label: "Side", get: (x) => x.side }, { label: "Opened", get: (x) => x.open_date }, { label: "Closed", get: (x) => x.close_date || "open" }, { label: "Days", num: true, get: (x) => String(x.holding_days) },
      { label: "Peak weight", num: true, get: (x) => pct(x.peak_weight, 1) }, { label: "Position return", num: true, get: (x) => pct(x.position_return, 2) }, { label: "Contribution", num: true, get: (x) => pct(x.contribution, 2) }, { label: "Cost", num: true, get: (x) => pct(x.cost, 3) }, { label: "Net", num: true, get: (x) => pct(x.net_contribution, 2) },
      { label: "Opened because", get: (x) => x.open_reason || "" }, { label: "Closed because", get: (x) => (x.close_action ? x.close_action + ": " : "") + (x.close_reason || "") }], r.rows);
    else Q.simpleTable(card, [{ label: "Executed", get: (x) => x.date }, { label: "Decided", get: (x) => x.decision_date || "" }, { label: "Asset", get: (x) => x.asset }, { label: "Trade", get: (x) => (KINDS[x.kind] || [0, 0, x.kind])[2] }, { label: "Weight traded", num: true, get: (x) => pct(x.trade_weight, 2) },
      { label: "Book after", num: true, get: (x) => pct(x.weight_after, 2) }, { label: "Cost", num: true, get: (x) => pct(x.cost, 3) }, { label: "Decision", get: (x) => x.action || "" }, { label: "Why", get: (x) => x.reason || "" }], r.rows);
    pager(card, r.n, S.tradeOffset, 50, (o) => { S.tradeOffset = o; draw(); });
  }

  /* A forward test of a period: the run's own book over the last few trading days (or any dates), against the equal-weight benchmark. Nothing is re-fitted; the page says which of the days were out of sample. */
  async function windowCard(out) {
    const card = h("div", "card", out); card.dataset.role = "window-test"; h("h3", "", card, "Test a period");
    h("p", "note", card, "How the run's book did over a stretch of days rather than one date: the last week, the last month, or dates of your own. The book is the one the run recorded, so a day the model trained on is flagged and does not count as forward.");
    const row = h("div", "row", card), body = h("div", "", card); body.setAttribute("aria-live", "polite");
    const from = h("input", "", row), to = h("input", "", row); from.type = to.type = "date"; from.setAttribute("aria-label", "From"); to.setAttribute("aria-label", "To"); from.dataset.key = "window-from"; to.dataset.key = "window-to";
    const seg = h("div", "tp-seg", row); seg.setAttribute("role", "group"); seg.setAttribute("aria-label", "Period length");
    async function run(query) {
      body.replaceChildren(); h("p", "hint", body, "Testing…");
      let r; try { r = await api(`/api/ml/strategy/window?id=${encodeURIComponent(S.id)}&${query}`); } catch (e) { body.replaceChildren(); h("div", "banner err", body, e.message); return; }
      body.replaceChildren(); from.value = r.window.from; to.value = r.window.to;
      for (const w of r.warnings || []) h("div", "banner warn", body, w);
      const m = r.metrics, tiles = h("div", "tiles2", body); tiles.dataset.role = "window-tiles";
      Q.tile(tiles, "Window", `${r.window.from} to ${r.window.to}`, null, "", `${m.days} trading days`); Q.tile(tiles, "Strategy", pct(m.net_return, 2), null, "", `After costs of ${pct(m.cost, 2)}`);
      Q.tile(tiles, "Equal-weight benchmark", pct(m.benchmark_return, 2)); Q.tile(tiles, "Active return", pct(m.active_return, 2), null, "", "Strategy minus benchmark over the window");
      Q.tile(tiles, "Up days", `${m.up_days} of ${m.days}`); Q.tile(tiles, "Beat benchmark", `${m.beat_benchmark_days} of ${m.days} days`); Q.tile(tiles, "Worst drawdown", pct(m.max_drawdown, 2));
      Q.tile(tiles, "Out of sample", `${r.coverage.out_of_sample_days} of ${r.coverage.days} days`, null, "", Object.entries(r.coverage.segments).map(([k, v]) => `${k}: ${v}`).join(", "));
      C.line(h("div", "", body), { dates: r.curve.dates, series: [{ name: "Strategy", color: css("--s1"), values: r.curve.net }, { name: "Equal-weight benchmark", color: css("--muted"), values: r.curve.benchmark }], baseline: 1, yFormat: (v) => v.toFixed(3),
                                   tipFormat: (v) => v.toFixed(4), height: 220, label: "Growth of 1 over the window" });
      if (r.read) h("p", "hint", body, `At the start of the window the model's top half of assets returned ${pct(r.read.top_half_return, 2)} against ${pct(r.read.bottom_half_return, 2)} for its bottom half (rank correlation ${num(r.read.rank_correlation, 2)} over ${r.read.n_assets} assets). ${r.read.note}`);
      if (r.assets.length) { h("h4", "", body, "What made and lost the money"); Q.simpleTable(body, [{ label: "Asset", get: (x) => x.asset }, { label: "Average weight", num: true, get: (x) => pct(x.avg_weight, 1) }, { label: "Asset return", num: true, get: (x) => pct(x.asset_return, 2) },
        { label: "Contribution to the book", num: true, get: (x) => pct(x.contribution, 2) }, { label: "Model rank at the start", num: true, get: (x) => pct(x.rank_at_start, 0) }], r.assets.slice(0, 12)); }
      if (r.trades.length) { h("h4", "", body, "Trades inside the window"); Q.simpleTable(body, [{ label: "Executed", get: (x) => x.date }, { label: "Asset", get: (x) => x.asset }, { label: "Action", get: (x) => x.action || "" }, { label: "Weight change", num: true, get: (x) => pct(x.weight_change, 2) },
        { label: "Book after", num: true, get: (x) => pct(x.weight_after, 1) }, { label: "Why", get: (x) => x.reason || "" }], r.trades.slice(0, 30)); }
    }
    for (const [label, d] of [["1 week", 5], ["2 weeks", 10], ["1 month", 21], ["3 months", 63]]) { const b = h("button", "btn small", seg, label); b.type = "button"; b.dataset.days = String(d); b.addEventListener("click", () => run("days=" + d)); }
    const go = h("button", "btn", row, "Test these dates"); go.type = "button"; go.addEventListener("click", () => { if (from.value || to.value) run(`${from.value ? "from=" + from.value + "&" : ""}${to.value ? "to=" + to.value : ""}`); });
    await run("days=5");
  }

  async function replayCard(out) {
    const card = h("div", "card", out); card.dataset.role = "replay"; h("h3", "", card, "Replay a decision date");
    h("p", "note", card, "Pick a date to see only what was known then: every asset's score and rank, the book going in, and what the policy decided. The trades that followed happen later and are listed apart.");
    const row = h("div", "row", card), inp = h("input", "", row); inp.type = "date"; inp.setAttribute("aria-label", "Decision date"); inp.dataset.key = "replay-date";
    const go = h("button", "btn", row, "Replay"), body = h("div", "", card); body.setAttribute("aria-live", "polite"); go.type = "button";
    const dates = S.data && S.data.decision_dates && S.data.decision_dates.length ? S.data.decision_dates : (S.decisions && S.decisions.rows.length ? S.decisions.rows : (await api(`/api/ml/strategy/decisions?id=${encodeURIComponent(S.id)}&limit=500`)).rows).map((x) => x.date);
    inp.value = S.replay ? S.replay.date : dates[Math.floor(dates.length / 2)] || "";
    const step = (dir) => { const u = [...new Set(dates)].sort(), i = u.findIndex((x) => x >= inp.value); const j = dir < 0 ? Math.max(0, (i < 0 ? u.length : i) - 1) : Math.min(u.length - 1, i < 0 ? u.length - 1 : (u[i] === inp.value ? i + 1 : i)); inp.value = u[j]; run(); };
    const prev = h("button", "btn small", row, "Previous decision day"), nxt = h("button", "btn small", row, "Next decision day"); prev.type = nxt.type = "button"; prev.addEventListener("click", () => step(-1)); nxt.addEventListener("click", () => step(1));
    async function run() {
      if (!inp.value) return;
      let r; try { r = await api(`/api/ml/strategy/replay?id=${encodeURIComponent(S.id)}&date=${inp.value}`); } catch (e) { body.replaceChildren(); h("div", "banner err", body, e.message); return; }
      S.replay = r; body.replaceChildren();
      h("p", "hint", body, `Known on ${r.date}: ${r.known}. Trades are executed ${r.lag_days} day(s) after the decision.`);
      Q.simpleTable(body, [{ label: "Asset", get: (x) => x.asset }, { label: "Score", num: true, get: (x) => num(x.score, 4) }, { label: "Rank", num: true, get: (x) => pct(x.rank_pct, 0) }, { label: "Close", num: true, get: (x) => num(x.close, 2) }, { label: "Book going in", num: true, get: (x) => pct(x.weight_held_before, 1) },
        { label: "Decision", get: (x) => x.decision || "no action" }, { label: "Book after", num: true, get: (x) => pct(x.book_after, 1) }, { label: "Why", get: (x) => x.reason || "" }], r.assets);
      if (r.executed_later.length) { h("h4", "", body, "Executed afterwards (not known at the decision)"); Q.simpleTable(body, [{ label: "Executed", get: (x) => x.date }, { label: "Asset", get: (x) => x.asset }, { label: "Trade", get: (x) => (KINDS[x.kind] || [0, 0, x.kind])[2] }, { label: "Weight traded", num: true, get: (x) => pct(x.weight_change, 2) }, { label: "Cost", num: true, get: (x) => pct(x.cost, 3) }], r.executed_later); }
    }
    go.addEventListener("click", run); if (S.replay) run();
  }

  Q.registerPage("strategy", { render: (args) => {
    if (args && args.id && args.id !== S.id) { S.id = args.id; S.data = null; S.asset = null; S.decisions = null; S.trades = null; S.replay = null; S.log.offset = 0; S.tradeOffset = 0; S.runs = null; }   // opened from a report: show that run
    draw();
  } });
})();
