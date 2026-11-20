/* Research Evidence: the checklist behind one experiment, and the feature registry with its evidence.

   Every line of the checklist applies a stated rule to a number the run recorded, and says where the number came from. An unknown is something nobody has checked, not something that is fine; there is no
   combined score. The feature section lists each feature's source, when it is known, how it is calculated and the hypothesis and reference that make it worth testing, and can measure each one on a dataset
   and ablate whole groups against a noise control. */
(() => {
  "use strict";
  const Q = window.QL, { h, $, api, toast, css } = Q, C = Q.C;
  const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.num(v, d)), pct = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Q.pct(v, d));
  const root = $("#evidence");
  const S = { runs: null, id: null, ev: null, reg: null, study: null, busy: false, group: "", ablate: true, model: "ridge", min_train: 756 };
  const MARK = { pass: ["✓", "Pass"], warn: ["!", "Warning"], fail: ["✕", "Fail"], unknown: ["?", "Unknown"] };

  async function draw() {
    root.replaceChildren();
    const d = h("div", "head", root), l = h("div", "", d); h("h2", "", l, "Research Evidence");
    h("p", "sub", l, "What a saved experiment's own record supports, line by line, with the rule behind each line. Plus the feature registry: where each input comes from, when it is known and why it was worth testing.");
    try { if (!S.runs) { S.runs = (await api("/api/ml/experiments")).rows; if (!S.id && S.runs.length) S.id = S.runs[0].id; } if (!S.reg) S.reg = await api("/api/ml/features"); } catch (e) { h("div", "banner err", root, e.message); return; }
    await checklist(); featureSection();
  }

  async function checklist() {
    const card = h("div", "card", root); card.dataset.role = "evidence-card"; h("h3", "", card, "Evidence checklist");
    if (!S.runs.length) { h("p", "note", card, "No experiment yet. Train a model in the Machine Learning Lab or run a strategy in the ML Strategy Explorer, then come back."); return; }
    const l = h("label", "f", card, "Experiment"), sel = h("select", "", l); sel.dataset.key = "run";
    for (const r of S.runs.slice(0, 60)) { const o = h("option", "", sel, `${r.name || r.id}  ·  ${r.model || ""}  ·  ${r.signal || ""}  ·  Sharpe ${num(r["oos.sharpe"], 2)}`); o.value = r.id; if (r.id === S.id) o.selected = true; }
    sel.addEventListener("change", () => { S.id = sel.value; S.ev = null; draw(); });
    if (!S.ev || S.ev.id !== S.id) { try { S.ev = await api(`/api/ml/evidence?id=${encodeURIComponent(S.id)}`); } catch (e) { h("div", "banner err", card, e.message); return; } }
    const ev = S.ev, row = h("div", "chips", card); row.dataset.role = "evidence-counts";
    for (const k of ["pass", "warn", "fail", "unknown"]) { const c = h("span", "chip", row), i = h("i", "", c, MARK[k][0]); i.style.color = css(k === "pass" ? "--pos" : k === "fail" ? "--neg" : "--ink"); h("span", "", c, ` ${ev.counts[k]} ${MARK[k][1].toLowerCase()}`); }
    h("p", "note", card, ev.reading);
    const list = h("div", "stack", card); list.dataset.role = "evidence-list";
    for (const c of ev.checks) {
      const d = h("details", "card"); d.dataset.status = c.status; d.dataset.check = c.id; list.appendChild(d);
      const sm = h("summary", "", d), mk = h("b", "", sm, `${MARK[c.status][0]} ${MARK[c.status][1]}`); mk.style.color = css(c.status === "pass" ? "--pos" : c.status === "fail" ? "--neg" : "--ink"); h("span", "", sm, "  " + c.title);
      h("p", "", d, c.detail); h("p", "hint", d, "Rule: " + c.rule); h("p", "hint", d, "Source: " + c.source);
      if (c.status === "unknown" && c.id === "random_signals") { const b = h("button", "btn small", d, "Run the random-signal control"); b.type = "button"; b.addEventListener("click", async () => { b.disabled = true; try { const { job } = await api("/api/ml/strategy/controls", { id: S.id, draws: 40 }); await Q.waitJob(job, () => {}, 1000); S.ev = null; draw(); } catch (e) { toast(e.message); b.disabled = false; } }); }
    }
  }

  function featureSection() {
    const card = h("div", "card", root); card.dataset.role = "feature-registry"; h("h3", "", card, "Feature registry");
    h("p", "note", card, "Every feature the platform can build. The hypothesis is a reason to test the feature, not a result: the study below measures it.");
    Q.simpleTable(card, [{ label: "Group", get: (g) => g.group }, { label: "Features", num: true, get: (g) => String(g.n_features) }, { label: "Known when", get: (g) => g.availability }, { label: "Hypothesis", get: (g) => g.hypothesis }, { label: "Reference", get: (g) => g.reference }], S.reg.groups);
    const f = h("div", "params", card);
    const gl = h("label", "f", f, "Show features of"), gs = h("select", "", gl); gs.dataset.key = "group"; for (const g of ["", ...S.reg.groups.map((x) => x.group)]) { const o = h("option", "", gs, g || "all groups"); o.value = g; if (g === S.group) o.selected = true; }
    const body = h("div", "", card);
    const showFeatures = () => { body.replaceChildren(); const rows = S.reg.features.filter((r) => !S.group || r.group === S.group), st = S.study ? Object.fromEntries(S.study.features.map((x) => [x.feature, x])) : {};
      Q.simpleTable(body, [{ label: "Feature", get: (r) => r.name }, { label: "Group", get: (r) => r.group }, { label: "Source", get: (r) => r.source }, { label: "Calculation", get: (r) => r.calculation }, { label: "Days of history", num: true, get: (r) => String(r.lookback_days) },
        ...(S.study ? [{ label: "Mean rank IC", num: true, get: (r) => num((st[r.name] || {}).mean_ic, 4) }, { label: "t", num: true, get: (r) => num((st[r.name] || {}).t_stat, 1) }, { label: "Years with the same sign", num: true, get: (r) => pct((st[r.name] || {}).years_same_sign, 0) }] : [])], rows); };
    gs.addEventListener("change", () => { S.group = gs.value; showFeatures(); }); showFeatures();
    const sc = h("div", "card", root); sc.dataset.role = "feature-study"; h("h3", "", sc, "Measure the features on a dataset");
    h("p", "note", sc, "The rank information coefficient of each feature against the target, and how often it keeps its sign across years. With ablation, the same model is retrained without each group and on noise alone: a change smaller than the gap between the full model and the noise run cannot be told from nothing.");
    const g2 = h("div", "params", sc);
    const mrow = h("label", "f", g2, "Model for the ablation"), ms = h("select", "", mrow); ms.dataset.key = "model"; for (const m of ["ridge", "lasso", "elasticnet"]) { const o = h("option", "", ms, m); o.value = m; if (m === S.model) o.selected = true; } ms.addEventListener("change", () => { S.model = ms.value; });
    const tl = h("label", "f", g2, "Training history before the first prediction (days)"), ti = h("input", "", tl); ti.type = "number"; ti.value = S.min_train; ti.dataset.key = "min_train"; ti.addEventListener("input", () => { S.min_train = ti.value; });
    const al = h("label", "f inline", g2), ac = h("input", "", al); ac.type = "checkbox"; ac.checked = S.ablate; ac.dataset.key = "ablate"; al.appendChild(document.createTextNode(" Also ablate each group (retrains the model several times)")); ac.addEventListener("change", () => { S.ablate = ac.checked; });
    const row = h("div", "row", sc), go = h("button", "btn primary", row, "Run the feature study"), status = h("span", "hint", row, ""), out = h("div", "", sc); go.type = "button"; out.setAttribute("aria-live", "polite");
    const show = () => { out.replaceChildren(); const st = S.study; if (!st) return; h("p", "hint", out, `${st.dataset.n_features} features over ${st.dataset.n_dates} days against ${st.dataset.target}.`);
      if (st.ablation) { h("h4", "", out, "Ablation"); Q.simpleTable(out, [{ label: "Variant", get: (r) => r.variant }, { label: "Features", num: true, get: (r) => String(r.n_features) }, { label: "Rank IC", num: true, get: (r) => num(r.rank_ic, 4) }, { label: "Change in rank IC", num: true, get: (r) => (r.ic_change === undefined ? "" : num(r.ic_change, 4)) },
        { label: "Net Sharpe", num: true, get: (r) => num(r.sharpe, 2) }, { label: "Change in Sharpe", num: true, get: (r) => (r.sharpe_change === undefined || r.sharpe_change === null ? "" : num(r.sharpe_change, 2)) }], [...st.ablation.rows, st.ablation.noise]); h("p", "hint", out, st.ablation.reading); }
      showFeatures(); };
    go.addEventListener("click", async () => {
      go.disabled = true; status.textContent = "Starting…";
      try { const { job } = await api("/api/ml/features/study", { model: { kind: S.model }, split: { min_train_days: Number(S.min_train) }, ablation: S.ablate, explain: false }); S.study = await Q.waitJob(job, (v) => { status.textContent = `${v.message} (${Math.round(v.elapsed)} s)`; }, 1500); status.textContent = ""; show(); }
      catch (e) { status.textContent = ""; h("div", "banner err", out, e.message); }
      go.disabled = false;
    });
    show();
  }

  Q.registerPage("evidence", { render: () => draw() });
})();
