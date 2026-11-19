/* The Market Regime Lab.

   A regime estimate is only as honest as its timing, so the page keeps two views apart and says which is which. The FILTERED view is causal: each day's probabilities use only the data up to that day and a
   model fitted before it, so a rule could have traded on it. The RETROSPECTIVE view uses the whole sample, including the days after each label; it describes the past better and cannot be traded on, and it
   is drawn only beside the filtered one so the difference can be seen. Performance by regime pairs the label known at the close with the NEXT day's return, because that is what a position could earn. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, toast, css } = Q, C = Q.C, R = Q.research, B = Q.badges;
  const pct = (v, d = 1) => (v === null || v === undefined ? "n/a" : Q.pct(v, d)), num = (v, d = 2) => (v === null || v === undefined ? "n/a" : Q.num(v, d));
  const root = $("#regime");
  const S = { form: { assets: "", start: "", source: "", model: "hmm", n_states: "2", min_train: "750", refit_every: "126", min_days: "60", save: true }, result: null, busy: false, args: {}, plan: null };
  const COLORS = () => [css("--s1"), css("--s2"), css("--s4")];

  function defaults() {
    const c = (R.active() && R.active().context) || {}, setup = Q.setupContext ? Q.setupContext() : {};
    S.form.assets = (c.assets && c.assets.length ? c.assets : setup.assets || []).join(", ");
    S.form.start = c.start || setup.start || ""; S.form.source = c.data_source || setup.data_source || "";
  }

  async function draw() {
    root.replaceChildren();
    const d = h("div", "head", root), l = h("div", "", d); h("h2", "", l, "Market Regime Lab");
    h("p", "sub", l, "What state the market is in, how long states last and how a result behaved in each. Filtered estimates are causal; retrospective ones are shown only for comparison.");
    if (!S.plan) { try { S.plan = await api("/api/regime/plan"); } catch (e) { h("div", "banner err", root, "The regime service is not available: " + e.message + " (restart the app if it was just updated)"); return; } }
    form();
    const out = h("div", "stack", root); out.id = "regime-results"; out.setAttribute("aria-live", "polite");
    if (S.result) results(out, S.result); else h("div", "card empty", out).append(h("h3", "", null, "No estimate yet"), h("p", "", null, "Choose assets and a model, then run the estimate. About a minute for ten years of daily data."));
    await byResult(out);
  }

  function form() {
    const card = h("div", "card", root); card.dataset.role = "regime-form"; h("h3", "", card, "Estimate regimes");
    const f = S.form, grid = h("div", "params", card);
    const text = (label, key, type, wide, help) => { const l = h("label", "f" + (wide ? " wide" : ""), grid, label), i = h("input", "", l); i.type = type || "text"; i.value = f[key]; i.dataset.key = key; if (help) i.title = help; i.addEventListener("input", () => { f[key] = i.value; }); return i; };
    text("Assets (tickers)", "assets", "text", true, "The market proxy is the equal-weight return of these assets");
    text("First day", "start", "date");
    const sl = h("label", "f", grid, "Data source"), src = h("select", "", sl); src.dataset.key = "source";
    for (const s of ["", ...(Q.S.catalog ? Q.S.catalog.sources.map((x) => x.name) : [])]) { const o = h("option", "", src, s || "(Backtest tab's source)"); o.value = s; if (s === f.source) o.selected = true; } src.addEventListener("change", () => { f.source = src.value; });
    const ml = h("label", "f", grid, "Model"), mod = h("select", "", ml); mod.dataset.key = "model";
    for (const m of S.plan.models) { const o = h("option", "", mod, m.name); o.value = m.name; o.title = m.description; if (m.name === f.model) o.selected = true; }
    const desc = h("p", "hint wide", grid); const showDesc = () => { desc.textContent = (S.plan.models.find((m) => m.name === f.model) || {}).description || ""; };
    mod.addEventListener("change", () => { f.model = mod.value; showDesc(); extra.hidden = f.model !== "hmm"; }); showDesc();
    const extra = h("div", "params wide", grid); extra.hidden = f.model !== "hmm";
    const sub = (label, key, type, opts) => { const l = h("label", "f", extra, label); if (opts) { const s = h("select", "", l); s.dataset.key = key; for (const [v, t] of opts) { const o = h("option", "", s, t); o.value = v; if (f[key] === v) o.selected = true; } s.addEventListener("change", () => { f[key] = s.value; }); } else { const i = h("input", "", l); i.type = "number"; i.value = f[key]; i.dataset.key = key; i.addEventListener("input", () => { f[key] = i.value; }); } };
    sub("States", "n_states", null, [["2", "2 (calm, stressed)"], ["3", "3 (calm, middle, stressed)"]]); sub("Days to fit the first model", "min_train"); sub("Refit every (days)", "refit_every");
    text("Fewest days for a reading", "min_days", "number", false, "A regime with fewer days than this is reported as insufficient evidence, never as zero");
    const row = h("div", "row", card), run = h("button", "btn primary", row, "Estimate regimes"); run.type = "button"; run.disabled = S.busy;
    const keep = h("label", "f inline", row), cb = h("input", "", keep); cb.type = "checkbox"; cb.checked = f.save && !!R.active(); cb.disabled = !R.active(); keep.append(document.createTextNode(R.active() ? `Save to "${R.active().name}"` : "Open a session to save the estimate"));
    cb.addEventListener("change", () => { f.save = cb.checked; });
    const status = h("span", "hint", row); status.id = "regime-status"; status.setAttribute("role", "status");
    for (const n of S.plan.notes) h("p", "hint", card, n);
    run.addEventListener("click", () => start(run, status));
  }

  async function start(btn, status) {
    const f = S.form, body = { assets: f.assets.split(/[\s,;]+/).filter(Boolean), start: f.start || undefined, source: f.source || undefined, model: f.model, n_states: +f.n_states, min_train: +f.min_train, refit_every: +f.refit_every, min_days: +f.min_days };
    if (f.save && R.active()) body.session = R.active().id;
    S.busy = true; btn.disabled = true; status.textContent = "Starting…";
    try {
      const { job } = await api("/api/regime/run", body);
      S.result = await Q.waitJob(job, (v) => { status.textContent = `${v.message} (${Math.round(v.elapsed)} s)`; }, 1000);
      S.job = job; if (S.result.artifact) { await R.refresh(); toast("Regime estimate saved to the session"); }
    } catch (e) { S.busy = false; status.textContent = ""; toast(e.message); h("div", "banner err", $("#regime-results") || root, e.message); btn.disabled = false; return; }
    S.busy = false; draw();
  }

  /* ---------------------------------------------------------------- results */
  function results(out, r) {
    const cur = r.current;
    const top = h("div", "card", out); top.dataset.role = "regime-now";
    const hd = h("div", "row", top); h("h3", "", hd, cur ? `${cur.state} as of ${cur.as_of}` : "No estimate"); B.evidence(hd, "walk_forward"); h("span", "ev wf", hd, "Filtered (causal)");
    const k = h("div", "kpis", top);
    if (cur) for (const n of r.names) Q.tile(k, "P(" + n + ")", pct(cur.probabilities[n], 0));
    Q.tile(k, "Days with an estimate", String(r.timeline.dates.length), { text: `from ${r.first_estimate}` });
    h("p", "hint", top, `${r.model_text}. Market proxy: ${r.proxy.definition}, ${r.proxy.first} to ${r.proxy.last}. ${r.label_basis}.`);
    if (R.active() && !r.artifact) { const s = h("button", "btn", top, "Save to session"); s.type = "button"; s.addEventListener("click", async () => { if (!S.job) return; try { await R.saveRun({ job: S.job }); s.textContent = "Saved"; s.disabled = true; } catch (e) { toast(e.message); } }); }
    const colors = COLORS(), dates = r.timeline.dates;
    const pc = Q.chartCard(out, "Probability of each state (filtered)", "Each day uses only data up to that day.", { wide: true, draw: (b) => C.line(b, { dates, series: r.names.map((n, i) => ({ name: n, color: colors[i % 3], values: r.timeline.probabilities[n] })), yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 0), ylim: [0, 1], height: 240, label: "Filtered regime probabilities" }),
      table: (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => dates[i] }, ...r.names.map((n) => ({ label: n, num: true, get: (i) => num(r.timeline.probabilities[n][i], 3) }))], Q.sampleRows(dates, null, Math.max(1, Math.ceil(dates.length / 60)))) });
    if (r.retrospective) {
      const hi = r.names.length - 1, rr = r.retrospective;
      const card = Q.chartCard(out, `Filtered against retrospective: probability of ${r.names[hi]}`, "The retrospective line uses the whole sample and cannot be traded on; where the two disagree, the filtered line is what was knowable.", { wide: true,
        draw: (b) => C.line(b, { dates, series: [{ name: "Filtered (causal)", color: colors[0], values: r.timeline.probabilities[r.names[hi]] }, { name: "Retrospective (not tradeable)", color: colors[1], values: rr.probabilities[r.names[hi]], dash: true }], yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 0), ylim: [0, 1], height: 240, label: "Filtered and retrospective probability" }),
        table: (b) => Q.simpleTable(b, [{ label: "Date", get: (i) => dates[i] }, { label: "Filtered", num: true, get: (i) => num(r.timeline.probabilities[r.names[hi]][i], 3) }, { label: "Retrospective", num: true, get: (i) => num(rr.probabilities[r.names[hi]][i], 3) }], Q.sampleRows(dates, null, Math.max(1, Math.ceil(dates.length / 60)))) });
      card.dataset.role = "regime-retro"; h("span", "ev is", card._extra, "Retrospective");
    } else h("p", "hint", out, r.retrospective_note);
    const two = h("div", "charts", out);
    const tm = h("div", "card", two); tm.dataset.role = "regime-transitions"; h("h3", "", tm, "Transitions between regimes"); h("p", "sub", tm, "Share of each state's days that were followed by each state, from the filtered labels. Rows sum to 100%.");
    const tr = r.transitions; Q.simpleTable(tm, [{ label: "From \\ to", get: (x) => x.s }, ...tr.states.map((s, j) => ({ label: s, num: true, get: (x) => { return pct(x.p[j], 1); } }))], tr.states.map((s, i) => ({ s, p: tr.probabilities[i] })));
    if (r.model_transition) { h("p", "hint", tm, "The model's own one-day transition probabilities (its last refit):"); Q.simpleTable(tm, [{ label: "From \\ to", get: (x) => x.s }, ...tr.states.map((s, j) => ({ label: s, num: true, get: (x) => pct(x.p[j], 1) }))], tr.states.map((s, i) => ({ s, p: r.model_transition[i] }))); }
    const du = h("div", "card", two); du.dataset.role = "regime-durations"; h("h3", "", du, "How long regimes last"); h("p", "sub", du, "Spells are unbroken runs of the same filtered label, in trading days.");
    Q.simpleTable(du, [{ label: "State", get: (x) => x.state }, { label: "Spells", num: true, get: (x) => String(x.spells) }, { label: "Mean days", num: true, get: (x) => num(x.mean_days, 1) }, { label: "Median", num: true, get: (x) => num(x.median_days, 0) }, { label: "Longest", num: true, get: (x) => (x.longest_days === null ? "n/a" : String(x.longest_days)) }, { label: "Share of days", num: true, get: (x) => pct(x.share) }], r.durations);
    perf(out, "Market behaviour by regime: the next day", r.proxy_by_regime.next_day, "regime-perf-next"); perf(out, "Market behaviour by regime: the same day (descriptive)", r.proxy_by_regime.same_day, "regime-perf-same");
    stab(out, r);
    const refs = h("details", "card", out); h("summary", "", refs, "How this estimate was made"); const dl = h("dl", "kv", refs);
    for (const [a, b2] of [["Model", r.model], ["Settings", JSON.stringify(r.settings)], ["Universe", `${r.universe.tickers.join(", ")} (${r.universe.source})`], ["Code version", r.code_version || "unknown"], ["Evidence", "walk-forward: each day's model was fitted on earlier days only"]]) { h("dt", "", dl, a); h("dd", "", dl, b2); }
  }

  function perf(out, title, p, role) {
    const card = h("div", "card", out); card.dataset.role = role; h("h3", "", card, title); h("p", "sub", card, p.basis + `. Regimes with fewer than ${p.min_days} days are marked insufficient evidence.`);
    Q.simpleTable(card, [{ label: "State", get: (x) => x.state }, { label: "Days", num: true, get: (x) => String(x.n_days) }, { label: "Share", num: true, get: (x) => pct(x.share) }, { label: "Return (annual)", num: true, get: (x) => (x.eligibility === "applied" ? pct(x.ann_mean) : "insufficient evidence") },
      { label: "Volatility", num: true, get: (x) => (x.eligibility === "applied" ? pct(x.ann_vol) : "") }, { label: "Sharpe", num: true, get: (x) => (x.eligibility === "applied" ? num(x.sharpe) : "") }, { label: "Sharpe interval (95%)", num: true, get: (x) => (x.sharpe_interval ? `${num(x.sharpe_interval[0])} to ${num(x.sharpe_interval[1])}` : "") },
      { label: "Worst day", num: true, get: (x) => (x.eligibility === "applied" ? pct(x.worst_day, 1) : "") }], p.rows);
  }

  function stab(out, r) {
    const s = r.stability, card = h("div", "card", out); card.dataset.role = "regime-stability"; h("h3", "", card, "How far to trust the labels"); h("p", "sub", card, "A regime model that is unsure, flips often or changes meaning between refits is describing noise.");
    const k = h("div", "kpis", card);
    Q.tile(k, "Mean confidence", pct(s.mean_confidence, 0), { text: "average probability of the most likely state" }); Q.tile(k, "Uncertain days", pct(s.uncertain_share, 1), { text: `most likely state below ${pct(s.uncertain_below, 0)}`, dir: s.uncertain_share > 0.15 ? "down" : "" });
    Q.tile(k, "Switches per year", num(s.switches_per_year, 1), { text: `${s.switches} in all` }); Q.tile(k, "Short spells", pct(s.short_spells_share, 0), { text: "switches undone within 5 days" });
    if (s.agreement_with_retrospective !== undefined) Q.tile(k, "Agreement with retrospective", pct(s.agreement_with_retrospective, 0), { text: "days where the causal and the hindsight label agree" });
    if (s.parameter_drift) { const p = s.parameter_drift; h("p", "hint", card, `Across ${p.refits} refits the volatility a state stands for ranged ${r.names.map((n, i) => `${n} ${pct(p.vol_min[i], 0)} to ${pct(p.vol_max[i], 0)}`).join("; ")}. ${p.degenerate_fits ? p.degenerate_fits + " refit(s) were degenerate. " : ""}${p.note}.`); }
  }

  /* ---------------------------------------------------------------- a saved result against a saved regime estimate */
  async function byResult(out) {
    const card = h("div", "card", out); card.dataset.role = "regime-by-result"; h("h3", "", card, "How a saved result behaved in each regime");
    if (!R.active()) { h("p", "empty-note", card, "Open a research session, save a backtest or experiment and a regime estimate into it, then score one against the other here."); return; }
    let d; try { d = await api("/api/research/session?id=" + R.active().id); } catch (e) { h("p", "empty-note", card, e.message); return; }
    const regs = d.artifacts.filter((a) => a.kind === "regime"), ress = d.artifacts.filter((a) => a.kind === "backtest" || a.kind === "ml_experiment");
    if (!regs.length || !ress.length) { h("p", "empty-note", card, `This session has ${regs.length} regime estimate(s) and ${ress.length} backtest/experiment result(s). Both are needed.`); return; }
    h("p", "sub", card, "The label known at the close of one day is paired with the result's return on the next day.");
    const row = h("div", "row", card), pick = (label, list, want) => { const l = h("label", "f", row, label), s = h("select", "", l); for (const a of list) { const o = h("option", "", s, `${a.title} (${a.created.slice(0, 10)})`); o.value = a.id; if (a.id === want) o.selected = true; } return s; };
    const rs = pick("Regime estimate", regs, null), xs = pick("Result", ress, S.args.result), go = h("button", "btn primary", row, "Score by regime"); go.type = "button";
    const box = h("div", "", card);
    const run = async () => {
      box.replaceChildren();
      try {
        const p = await api("/api/regime/by-result", { regime: rs.value, result: xs.value, min_days: +S.form.min_days });
        const ev = h("div", "row", box); B.evidence(ev, p.result_evidence); h("span", "hint", ev, p.window ? `Days in common: ${p.window.start} to ${p.window.end}` : "");
        if (p.caution) h("div", "banner", box, p.caution);
        perf(box, p.result_title + " by " + p.regime_title, p, "regime-by-result-table");
      } catch (e) { h("div", "banner err", box, e.message); }
    };
    go.addEventListener("click", run); if (S.args.result) run();
  }

  /* ---------------------------------------------------------------- entry */
  Q.registerPage("regime", { render: async (args) => {
    S.args = args || {};
    if (S.args.job && (!S.result || S.job !== S.args.job)) { try { const v = await api("/api/job/" + S.args.job); if (v.status === "done") { S.result = v.result; S.job = S.args.job; } } catch (e) { toast(e.message); } }
    if (S.args.result && R.ready()) { /* prefill from the result's context */
      try { const a = await api("/api/research/artifact?id=" + S.args.result); if (a.context && a.context.assets) { S.form.assets = a.context.assets.join(", "); S.form.start = (R.active() && R.active().context.start) || ""; } /* the result's own first day is when its trading began, not when its data did: the lab asks for the session's start, or all the history */ } catch (e) { /* the form keeps its defaults */ }
    } else if (!S.result) defaults();
    draw();
  } });
  R.onChange(() => { if (Q.currentTab() === "regime" && !S.busy) { if (!S.form.assets) defaults(); draw(); } });
})();
