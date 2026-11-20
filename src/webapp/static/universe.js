/* Stock Universe Explorer: which stocks to hold, and whether a portfolio built from them survives honest out-of-sample testing.

   The screen, the construction method and the lookback are chosen together on past data only (a nested chronological search) and judged on the data that followed. The page says what the universe is (today's
   names with hindsight, or a point-in-time table), how many candidates were tried, and shows the in-sample table apart from the out-of-sample result so the one is never read as the other. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, toast, css } = Q, C = Q.C;
  const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.num(v, d)), pct = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.pct(v, d));
  const root = $("#universe");
  const S = { plan: null, busy: false, result: null, form: { tickers: "", start: "", source: "", screens: { all: true, low_vol: true, momentum: false, size: false }, top: 0.5, methods: null, lookbacks: "252", folds: 4, min_train_days: 504, cost_bps: 10, membership: "", complete: false } };

  function defaults() {
    const c = (Q.research && Q.research.active() && Q.research.active().context) || {}, setup = Q.setupContext ? Q.setupContext() : {};
    const assets = (c.assets && c.assets.length ? c.assets : setup.assets) || [];
    S.form.tickers = S.form.tickers || assets.slice(0, 15).join(", ");
    S.form.start = S.form.start || c.start || setup.start || ""; S.form.source = S.form.source || c.data_source || setup.data_source || "";
    if (!S.form.methods && S.plan) S.form.methods = new Set(S.plan.default_methods);
  }

  async function draw() {
    root.replaceChildren();
    const d = h("div", "head", root), l = h("div", "", d); h("h2", "", l, "Stock Universe Explorer");
    h("p", "sub", l, "Choose a set of stocks, the screens and optimisers to consider, and see whether the procedure of picking among them would have earned anything on data it had not seen.");
    if (!S.plan) { try { S.plan = await api("/api/universe/plan"); } catch (e) { h("div", "banner err", root, "The universe service is not available: " + e.message + " (restart the app if it was just updated)"); return; } defaults(); }
    form();
    const out = h("div", "stack", root); out.id = "universe-results"; out.setAttribute("aria-live", "polite");
    if (S.result) results(out, S.result); else { const e = h("div", "card empty", out); h("h3", "", e, "No study yet"); h("p", "", e, S.plan.notes.join(" ")); }
  }

  function form() {
    const card = h("div", "card", root); card.dataset.role = "universe-form"; const f = S.form, grid = h("div", "params", card);
    const text = (label, key, type, wide, parent, help) => { const l = h("label", "f" + (wide ? " wide" : ""), parent || grid, label); if (help) l.title = help; const i = h("input", "", l); i.type = type || "text"; i.value = f[key]; i.dataset.key = key; i.addEventListener("input", () => { f[key] = i.value; }); return i; };
    text("Tickers (at least 6)", "tickers", "text", true); text("First day", "start", "date");
    const sl = h("label", "f", grid, "Data source"), src = h("select", "", sl); src.dataset.key = "source";
    for (const s of ["", ...(Q.S.catalog ? Q.S.catalog.sources.map((x) => x.name) : [])]) { const o = h("option", "", src, s || "(Backtest tab's source)"); o.value = s; if (s === f.source) o.selected = true; } src.addEventListener("change", () => { f.source = src.value; });
    const sc = h("div", "wide", grid); h("div", "hint", sc, "Screens to consider (each narrows the names before a portfolio is built, using only data known at the time)");
    const chips = h("div", "chips", sc);
    for (const [k, label] of [["all", "All names"], ["low_vol", "Lowest volatility"], ["momentum", "Highest 12-1 momentum"], ["size", "Largest by dollar volume (a size proxy)"]]) { const l = h("label", "f inline", chips), cb = h("input", "", l); cb.type = "checkbox"; cb.checked = !!f.screens[k]; cb.dataset.screen = k; l.appendChild(document.createTextNode(" " + label)); cb.addEventListener("change", () => { f.screens[k] = cb.checked; }); }
    text("Share kept by the volatility, momentum and size screens", "top", "number", false, grid, "0.5 keeps the best half");
    const mb = h("div", "wide", grid); h("div", "hint", mb, "Construction methods to consider"); const mrow = h("div", "chips", mb);
    for (const m of S.plan.methods) { const l = h("label", "f inline", mrow), cb = h("input", "", l); cb.type = "checkbox"; cb.checked = f.methods.has(m.key); cb.dataset.method = m.key; l.appendChild(document.createTextNode(" " + m.label)); cb.addEventListener("change", () => { cb.checked ? f.methods.add(m.key) : f.methods.delete(m.key); }); }
    text("Lookbacks (days, comma separated)", "lookbacks", "text"); text("Out-of-sample blocks", "folds", "number"); text("Selection history before the first block (days)", "min_train_days", "number"); text("Cost (bp, one way)", "cost_bps", "number");
    const mem = h("label", "f wide", grid, "Point-in-time membership (optional CSV: ticker,start,end,status)"), ta = h("textarea", "", mem); ta.rows = 3; ta.value = f.membership; ta.dataset.key = "membership"; ta.placeholder = "ticker,start,end,status\nAAA,2010-01-04,,active\nBBB,2010-01-04,2018-03-02,delisted"; ta.addEventListener("input", () => { f.membership = ta.value; });
    const cl = h("label", "f inline wide", grid), cc = h("input", "", cl); cc.type = "checkbox"; cc.checked = f.complete; cc.dataset.key = "complete"; cl.appendChild(document.createTextNode(" The table includes the securities that left (complete history). Leave unticked if it only lists survivors.")); cc.addEventListener("change", () => { f.complete = cc.checked; });
    const row = h("div", "row", card), run = h("button", "btn primary", row, "Run the universe study"); run.type = "button"; run.disabled = S.busy;
    const status = h("span", "hint", row, ""); status.setAttribute("aria-live", "polite");
    run.addEventListener("click", async () => {
      const screens = []; if (f.screens.all) screens.push({ kind: "all" }); for (const k of ["low_vol", "momentum", "size"]) if (f.screens[k]) screens.push({ kind: k, top: Number(f.top) });
      const body = { tickers: f.tickers, start: f.start || undefined, source: f.source || undefined, screens, methods: [...f.methods], lookbacks: String(f.lookbacks).split(/[ ,]+/).filter(Boolean).map(Number),
        design: { folds: Number(f.folds), min_train_days: Number(f.min_train_days), cost_bps: Number(f.cost_bps) } };
      if (f.membership.trim()) { body.membership = f.membership; body.membership_complete = !!f.complete; }
      S.busy = true; run.disabled = true; status.textContent = "Starting…";
      try { const { job } = await api("/api/universe/explore", body); S.result = await Q.waitJob(job, (v) => { status.textContent = `${v.message} (${Math.round(v.elapsed)} s)`; }, 1000); S.busy = false; draw(); }
      catch (e) { S.busy = false; run.disabled = false; status.textContent = ""; h("div", "banner err", card, e.message); }
    });
  }

  function results(out, r) {
    for (const w of r.warnings || []) h("div", "banner warn", out, w);
    const e = r.eligibility, b = r.benchmarks.equal_weight || {}, tiles = h("div", "tiles2", out); tiles.dataset.role = "universe-tiles";
    const t = (label, value, title) => Q.tile(tiles, label, value, null, "", title);
    t("Out-of-sample Sharpe", num(r.oos.sharpe, 2), `Equal weight over the same days: ${num(b.sharpe, 2)}`); t("Out-of-sample CAGR", pct(r.oos.cagr, 1), `Equal weight: ${pct(b.cagr, 1)}`); t("Max drawdown", pct(r.oos.max_drawdown, 0), `Equal weight: ${pct(b.max_drawdown, 0)}`);
    t("Expected shortfall (95%)", pct(r.oos.es95, 2), "Average of the worst 5% of days"); t("Candidates tried", String(r.n_candidates), "Screen x method x lookback: the number the result is corrected for");
    t("Deflated Sharpe probability", r.deflated_sharpe ? pct(r.deflated_sharpe.probability, 0) : "n/a", "The chance the Sharpe beats what the best of this many no-skill candidates would show");
    t("Beta to equal weight", num(r.beta, 2)); t("Turnover / year", `${num(r.turnover_per_year, 1)}x`); t("Cost drag / year", pct(r.annual_cost_drag, 2)); t("Universe basis", e.basis === "point_in_time" ? "Point in time" : "Today's names (hindsight)", e.limitation || "A complete point-in-time history");
    const colors = [css("--s1"), css("--s6"), css("--s3"), css("--s4")], eq = r.equity, names = Object.keys(eq).filter((k) => k !== "dates");
    Q.chartCard(out, "Out-of-sample equity", "Growth of 1 over the stitched out-of-sample blocks, after costs. Each block used the configuration chosen on the days before it. 'Fixed equal-weight walk' is the same engine with no screen and no search.", {
      wide: true, draw: (b2) => C.line(b2, { dates: eq.dates, series: names.map((k, i) => ({ name: k === "oos" ? "Selected by the procedure (net)" : k.replace(/_/g, " "), color: colors[i % 4], values: eq[k] })), yFormat: (v) => num(v, 2), height: 280, label: "Out-of-sample equity", baseline: 1 }),
      table: (b2) => Q.simpleTable(b2, [{ label: "Date", get: (x) => x.d }, ...names.map((k) => ({ label: k.replace(/_/g, " "), num: true, get: (x) => num(x[k], 3) }))], eq.dates.map((d, i) => ({ d, ...Object.fromEntries(names.map((k) => [k, eq[k][i]])) })).filter((_, i) => i % 5 === 0)) });
    const fc = h("div", "card", out); fc.dataset.role = "universe-folds"; h("h3", "", fc, "What was chosen, block by block");
    h("p", "note", fc, "Selection Sharpe is what the chosen candidate showed on the days before the block (the days it was chosen on). Out-of-sample Sharpe is what it earned on the block. A selection number that is much higher than the out-of-sample one is selection optimism.");
    Q.simpleTable(fc, [{ label: "Block", get: (x) => `${x.from} to ${x.to}` }, { label: "Chosen", get: (x) => x.chosen || "none" }, { label: "Selection Sharpe", num: true, get: (x) => num(x.selection_sharpe, 2) }, { label: "Out-of-sample Sharpe", num: true, get: (x) => num(x.oos_sharpe, 2) },
      { label: "Out-of-sample return", num: true, get: (x) => pct(x.oos_return, 1) }], r.folds);
    h("p", "hint", fc, `The same choice was made in ${pct(r.pick_stability, 0)} of the blocks (${r.modal_choice}).`);
    const rc = h("div", "card", out); rc.dataset.role = "universe-robustness"; h("h3", "", rc, "How far to believe it");
    const bs = r.bootstrap || {}, lines = [];
    if (bs.sharpe_lo !== undefined) lines.push(`Block bootstrap of the out-of-sample days: the Sharpe ratio is between ${num(bs.sharpe_lo, 2)} and ${num(bs.sharpe_hi, 2)} (90%), positive in ${pct(bs.p_sharpe_positive, 0)} of resamples` + (bs.p_beats_benchmark !== undefined ? `, above equal weight in ${pct(bs.p_beats_benchmark, 0)}.` : "."));
    lines.push(`Holdings: about ${num(r.concentration.effective_names, 1)} effective names; the largest group holds ${pct(r.concentration.largest_group_share, 0)} on average. Between rebalances ${pct(r.stability.jaccard, 0)} of the names stay the same.`);
    for (const l2 of lines) h("p", "", rc, l2);
    h("h4", "", rc, "If one of its biggest holdings were not there"); Q.simpleTable(rc, [{ label: "Removed", get: (x) => x.removed }, { label: "Average weight", num: true, get: (x) => pct(x.average_weight, 1) }, { label: "Sharpe without it", num: true, get: (x) => num(x.sharpe, 2) }, { label: "Change", num: true, get: (x) => num(x.change, 2) }], r.removal);
    if (r.regimes.length) { h("h4", "", rc, "In different markets"); Q.simpleTable(rc, [{ label: "Market", get: (x) => x.regime }, { label: "Days", num: true, get: (x) => String(x.days) }, { label: "Strategy Sharpe", num: true, get: (x) => num(x.strategy_sharpe, 2) }, { label: "Equal-weight Sharpe", num: true, get: (x) => num(x.benchmark_sharpe, 2) }], r.regimes); }
    const ic = h("div", "card", out); ic.dataset.role = "universe-insample"; h("h3", "", ic, "Every candidate over the whole history (in-sample)");
    h("p", "note", ic, "For context only. These numbers use the whole history, including the days the procedure above never saw when it chose. The best row here is a search result, not a finding: with this many candidates one is bound to look good.");
    Q.simpleTable(ic, [{ label: "Candidate", get: (x) => x.id }, { label: "Sharpe (whole history)", num: true, get: (x) => num(x.sharpe, 2) }, { label: "Fallbacks", num: true, get: (x) => String(x.fallbacks) }], r.in_sample);
    for (const n of r.notes) h("p", "hint", out, n);
  }

  Q.registerPage("universe", { render: () => { if (!S.form.tickers || !S.form.methods) defaults(); draw(); } });
})();
