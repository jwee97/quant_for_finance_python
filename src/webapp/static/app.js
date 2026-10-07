/* Quant Lab front end. Every string that comes from the server or the user is inserted with textContent, never innerHTML. */
(function () {
  "use strict";
  const C = window.Charts, h = C.html, fmt = C.fmt;
  const $ = (s, r = document) => r.querySelector(s);
  const token = $("meta[name=token]").content;
  const SLOTS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  async function api(path, body) {
    const res = await fetch(path, { method: body ? "POST" : "GET", headers: Object.assign({ "X-Token": token }, body ? { "Content-Type": "application/json" } : {}), body: body ? JSON.stringify(body) : undefined });
    let data = null; try { data = await res.json(); } catch (e) { data = { error: "the server sent an unreadable response" }; }
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }
  function store(key, value) { try { if (value === undefined) return JSON.parse(localStorage.getItem("ql." + key)); localStorage.setItem("ql." + key, JSON.stringify(value)); } catch (e) { return null; } }
  function toast(msg) { const t = h("div", "toast", document.body, msg); setTimeout(() => t.remove(), 3500); }
  function copy(text, what) { (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(() => toast(`${what} copied`), () => toast("Copy is blocked in this browser: select the text and copy it by hand")); }
  function download(name, text) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
  const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "n/a" : v.toFixed(d));
  const pct = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "n/a" : (v * 100).toFixed(d) + "%");

  /* ================================================================= state */
  const S = {
    catalog: null, tickers: [], start: "", models: [], combination: "equal", allocator: "", allocatorTouched: false, allocParams: {}, regime: "", regimeRisk: false,
    aum: "", validate: false, refresh: false, label: "", runs: [], selected: new Set(), current: null, range: "all", busy: false, nextSlot: 0, trials: {},
  };

  /* ================================================================= theme, tabs */
  function setTheme(t) { if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme"); store("theme", t); redraw(); }
  $("#theme").addEventListener("click", () => { const dark = document.documentElement.getAttribute("data-theme") ? document.documentElement.getAttribute("data-theme") === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; setTheme(dark ? "light" : "dark"); });
  const saved = store("theme"); if (saved) document.documentElement.setAttribute("data-theme", saved);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => redraw());

  const TABS = ["backtest", "compare", "builder", "guides"];
  function showTab(name) {
    for (const t of TABS) { $("#tab-" + t).setAttribute("aria-selected", String(t === name)); $("#view-" + t).hidden = t !== name; }
    if (name === "compare") renderCompare();
    if (name === "builder") renderBuilder();
    if (name === "guides") renderGuides();
    if (name === "backtest") redraw();
    store("tab", name);
  }
  for (const t of TABS) $("#tab-" + t).addEventListener("click", () => showTab(t));

  /* ================================================================= universe */
  function defaultTickers() { return S.catalog.default_tickers.map((t) => ({ t, cls: S.catalog.default_classes[t] || "unknown", status: "default" })); }
  function saveUniverse() { store("tickers", S.tickers.filter((x) => x.status !== "bad").map((x) => ({ t: x.t, cls: x.cls }))); }
  function renderUniverse() {
    const box = $("#u-chips"); box.replaceChildren();
    const isDefault = new Set(S.catalog.default_tickers);
    for (const x of S.tickers) {
      const chip = h("span", "chip " + (x.status === "bad" ? "bad" : x.status === "pending" ? "pending" : ""), box);
      h("b", "", chip, x.t);
      if (isDefault.has(x.t)) h("span", "meta", chip, x.cls.replace("_", " "));
      else {
        const sel = h("select", "", chip); sel.setAttribute("aria-label", "asset class of " + x.t);
        for (const c of S.catalog.classes) { const o = h("option", "", sel, c.replace("_", " ")); o.value = c; if (c === x.cls) o.selected = true; }
        sel.addEventListener("change", () => { x.cls = sel.value; saveUniverse(); });
      }
      if (x.status === "ok") { const ok = h("span", "meta", chip, "✓"); ok.style.color = css("--good-text"); chip.title = x.info || ""; }
      if (x.status === "bad") { h("span", "meta", chip, "✗"); chip.title = x.error || ""; }
      const rm = h("button", "", chip, "×"); rm.setAttribute("aria-label", "remove " + x.t); rm.title = "Remove";
      rm.addEventListener("click", () => { S.tickers = S.tickers.filter((y) => y !== x); saveUniverse(); renderUniverse(); });
    }
    const bad = S.tickers.filter((x) => x.status === "bad");
    if (bad.length) { const p = h("p", "hint", box, bad.map((x) => `${x.t}: ${x.error}`).join(" · ")); p.style.color = css("--crit"); p.style.flexBasis = "100%"; }
    const n = S.tickers.filter((x) => x.status !== "bad").length;
    $("#u-count").textContent = `${n} ticker${n === 1 ? "" : "s"}`;
  }
  async function addTickers(text) {
    const raw = text.split(/[\s,;]+/).map((s) => s.trim().toUpperCase()).filter(Boolean);
    const fresh = raw.filter((t) => !S.tickers.some((x) => x.t === t));
    if (!fresh.length) { toast(raw.length ? "Those tickers are already in the list" : "Type one or more tickers first"); return; }
    for (const t of fresh) S.tickers.push({ t, cls: "unknown", status: "pending" });
    renderUniverse();
    try {
      const res = await api("/api/tickers", { tickers: fresh, start: S.start || undefined, refresh: S.refresh });
      for (const t of fresh) {
        const x = S.tickers.find((y) => y.t === t), r = res.tickers[t]; if (!x || !r) continue;
        if (r.ok) { x.status = "ok"; x.cls = r.class && r.class !== "unknown" ? r.class : x.cls; x.info = `${r.days.toLocaleString()} days from ${r.first.slice(0, 4)}`; x.error = null; }
        else { x.status = "bad"; x.error = r.error; x.info = null; }
      }
      const bad = fresh.filter((t) => (res.tickers[t] || {}).ok === false);
      if (bad.length) toast(`${bad.join(", ")}: ${res.tickers[bad[0]].error}`);
    } catch (e) {
      for (const t of fresh) { const x = S.tickers.find((y) => y.t === t); if (x) { x.status = "bad"; x.error = "could not check"; } }
      toast(e.message);
    }
    saveUniverse(); renderUniverse();
  }

  /* ================================================================= strategy form */
  const modelInfo = (name) => S.catalog.models.find((m) => m.name === name);
  function fieldFor(p, value, onchange, extra) {
    const wrap = h("label", "f" + (extra && extra.wide ? " wide" : ""));
    wrap.appendChild(document.createTextNode(p.name.replace(/_/g, " ")));
    let input;
    if (p.kind === "choice") { input = h("select", "", wrap); for (const c of p.choices) { const o = h("option", "", input, c); o.value = c; if (c === value) o.selected = true; } }
    else if (p.kind === "bool") { wrap.className += " inline"; input = h("input", "", wrap); input.type = "checkbox"; input.checked = !!value; }
    else if (p.kind === "int" || p.kind === "float") { input = h("input", "", wrap); input.type = "number"; input.step = p.kind === "int" ? "1" : "any"; input.value = value; }
    else if (p.name === "expr") { input = h("textarea", "", wrap); input.rows = 3; input.value = value; input.spellcheck = false; }
    else { input = h("input", "", wrap); input.type = "text"; input.value = p.kind === "json" ? JSON.stringify(value) : (value === null || value === undefined ? "" : value); }
    input.addEventListener("input", () => {
      let v;
      if (p.kind === "bool") v = input.checked; else if (p.kind === "int") v = input.value === "" ? p.default : parseInt(input.value, 10); else if (p.kind === "float") v = input.value === "" ? p.default : parseFloat(input.value);
      else if (p.kind === "json") { try { v = JSON.parse(input.value); input.style.borderColor = ""; } catch (e) { input.style.borderColor = css("--crit"); return; } } else v = input.value;
      onchange(v);
    });
    input.addEventListener("change", () => input.dispatchEvent(new Event("input")));
    return wrap;
  }
  function modelOptions(select, current) {
    const groups = {};
    for (const m of S.catalog.models) (groups[m.user ? "My strategies" : m.family] = groups[m.user ? "My strategies" : m.family] || []).push(m);
    const order = Object.keys(groups).sort((a, b) => (a === "My strategies" ? -1 : b === "My strategies" ? 1 : a === "custom" ? -1 : b === "custom" ? 1 : a.localeCompare(b)));
    for (const g of order) { const og = h("optgroup", "", select); og.label = g === "custom" ? "custom (formula)" : g; for (const m of groups[g]) { const o = h("option", "", og, m.name); o.value = m.name; if (m.name === current) o.selected = true; } }
  }
  function renderStrategies() {
    const box = $("#s-body"); box.replaceChildren();
    S.models.forEach((entry, idx) => {
      const info = modelInfo(entry.name), block = h("div", "body", box);
      block.style.cssText = idx ? "border-top:1px dashed var(--axis);padding-top:10px" : "";
      const row = h("div", "row", block), sel = h("select", "", row); sel.setAttribute("aria-label", "strategy " + (idx + 1)); sel.style.flex = "1";
      modelOptions(sel, entry.name);
      sel.addEventListener("change", () => { entry.name = sel.value; entry.params = {}; autoAllocator(); renderStrategies(); renderPortfolio(); });
      if (S.models.length > 1) { const rm = h("button", "btn", row, "Remove"); rm.addEventListener("click", () => { S.models.splice(idx, 1); autoAllocator(); renderStrategies(); renderPortfolio(); }); }
      if (!info) return;
      h("p", "hint", block, info.description);
      if (info.rebalance) h("p", "hint", block, "This rule rebalances " + info.rebalance + " unless you change it in the spec.");
      if (info.requires.length) h("p", "hint", block, "Needs macro series: " + info.requires.join(", ") + " (included for the platform ETFs, and attached to any universe).");
      if (info.slow) { const w = h("p", "hint", block, "⏱ Slow: " + info.slow + "."); w.style.color = css("--warn"); }
      if (info.params.length) {
        const grid = h("div", "params", block);
        for (const p of info.params) {
          const cur = p.name in entry.params ? entry.params[p.name] : p.default;
          grid.appendChild(fieldFor(p, cur, (v) => { if (JSON.stringify(v) === JSON.stringify(p.default)) delete entry.params[p.name]; else entry.params[p.name] = v; if (p.name === "mode") { autoAllocator(); renderPortfolio(); } }, { wide: p.name === "expr" || p.kind === "json" }));
        }
      }
      if (entry.name === "expression") {
        const ex = h("select", "", block); ex.setAttribute("aria-label", "formula examples");
        const first = h("option", "", ex, "Load an example formula…"); first.value = "";
        S.catalog.formula.examples.forEach((e, i) => { const o = h("option", "", ex, e.title + "  ·  " + e.expr); o.value = String(i); });
        ex.addEventListener("change", () => { if (ex.value === "") return; const e = S.catalog.formula.examples[+ex.value]; entry.params.expr = e.expr; entry.params.mode = e.mode; renderStrategies(); });
        const link = h("button", "link", block, "Open the formula editor with the function reference"); link.addEventListener("click", () => showTab("builder"));
      }
    });
    if (S.models.length < 4) { const add = h("button", "btn", box, "+ Add another strategy and combine them"); add.addEventListener("click", () => { S.models.push({ name: "momentum", params: {} }); renderStrategies(); renderPortfolio(); }); }
    if (S.models.length > 1) {
      const lab = h("label", "f", box, "How to combine their forecasts"), sel = h("select", "", lab);
      for (const c of S.catalog.combinations) { const o = h("option", "", sel, c.name + ": " + c.description); o.value = c.name; if (c.name === S.combination) o.selected = true; }
      sel.addEventListener("change", () => { S.combination = sel.value; });
    }
  }
  function autoAllocator() {
    if (S.allocatorTouched) return;
    const entry = S.models.length === 1 ? S.models[0] : null, only = entry ? modelInfo(entry.name) : null;
    let pick = "";
    if (only && only.book === "sleeves") pick = "sleeves";
    else if (only && only.family === "custom") pick = (entry.params.mode || (only.params.find((p) => p.name === "mode") || {}).default) === "time_series" ? "sleeves" : "score_stack";
    else if (only && only.user) pick = only.position_mode === "time_series" ? "sleeves" : "score_stack";
    S.allocator = pick; S.allocParams = {};
  }
  function renderPortfolio() {
    const box = $("#p-body"); box.replaceChildren();
    const lab = h("label", "f", box, "Allocation"), sel = h("select", "", lab);
    const auto = h("option", "", sel, "Automatic (calibrated forecast, sized by risk)"); auto.value = "";
    for (const a of S.catalog.allocators) { const o = h("option", "", sel, a.name); o.value = a.name; if (a.name === S.allocator) o.selected = true; }
    if (!S.allocator) auto.selected = true;
    sel.addEventListener("change", () => { S.allocator = sel.value; S.allocatorTouched = true; S.allocParams = {}; renderPortfolio(); });
    const info = S.catalog.allocators.find((a) => a.name === S.allocator);
    h("p", "hint", box, info ? info.description : "The strategy's forecast is calibrated against past returns and turned into risk-scaled positions. A signal that has not paid in the past gets no position.");
    if (info && info.slow) h("p", "hint", box, "⏱ " + info.slow + ".");
    if (info && info.params.length) {
      const grid = h("div", "params", box);
      for (const p of info.params) { const cur = p.name in S.allocParams ? S.allocParams[p.name] : p.default; grid.appendChild(fieldFor(p, cur, (v) => { if (JSON.stringify(v) === JSON.stringify(p.default)) delete S.allocParams[p.name]; else S.allocParams[p.name] = v; }, { wide: p.kind === "json" })); }
    }
    const rl = h("label", "f", box, "Market regime detector"), rs = h("select", "", rl); const none = h("option", "", rs, "None"); none.value = "";
    for (const d of S.catalog.detectors) { const o = h("option", "", rs, d.name); o.value = d.name; if (d.name === S.regime) o.selected = true; }
    rs.addEventListener("change", () => { S.regime = rs.value; if (!S.regime) S.regimeRisk = false; renderPortfolio(); });
    const det = S.catalog.detectors.find((d) => d.name === S.regime);
    if (det) h("p", "hint", box, det.description);
    const rr = h("label", "f inline", box), cb = h("input", "", rr); cb.type = "checkbox"; cb.checked = S.regimeRisk; cb.disabled = !S.regime;
    rr.appendChild(document.createTextNode("Set the volatility target by regime (lower in crises)")); cb.addEventListener("change", () => { S.regimeRisk = cb.checked; });
    const al = h("label", "f", box, "Assets under management in dollars (adds market-impact costs)"), ai = h("input", "", al); ai.type = "number"; ai.min = "10000"; ai.placeholder = "blank = trading costs only"; ai.value = S.aum;
    ai.addEventListener("input", () => { S.aum = ai.value; });
  }

  /* ================================================================= panel skeleton */
  function buildPanel() {
    const p = $("#panel"); p.replaceChildren();
    const sec = (id, title, open) => { const d = h("details", "", p); if (open) d.open = true; const s = h("summary", "", d, title); const b = h("div", "body", d); b.id = id; return { d, s, b }; };
    const u = sec("u-body", "1. Tickers", true);
    h("span", "pill", u.s).id = "u-count";
    h("p", "hint", u.b, "The platform's 15 ETFs, plus anything on Yahoo Finance: NVDA, BTC-USD, ^GSPC, EURUSD=X. New tickers are downloaded as split- and dividend-adjusted prices and cached for a day.");
    h("div", "chips", u.b).id = "u-chips";
    const add = h("div", "row", u.b), inp = h("input", "", add); inp.type = "text"; inp.placeholder = "Add tickers, e.g. NVDA, BTC-USD"; inp.style.flex = "1"; inp.setAttribute("aria-label", "tickers to add");
    const go = h("button", "btn primary", add, "Add");
    const doAdd = () => { addTickers(inp.value); inp.value = ""; inp.focus(); };
    go.addEventListener("click", doAdd); inp.addEventListener("keydown", (e) => { if (e.key === "Enter") doAdd(); });
    const r2 = h("div", "row tight", u.b);
    const reset = h("button", "btn", r2, "Reset to the 15 ETFs"); reset.addEventListener("click", () => { S.tickers = defaultTickers(); saveUniverse(); renderUniverse(); });
    const clear = h("button", "btn", r2, "Clear"); clear.addEventListener("click", () => { S.tickers = []; saveUniverse(); renderUniverse(); });
    const dl = h("label", "f inline", u.b), dc = h("input", "", dl); dc.type = "checkbox"; dl.appendChild(document.createTextNode("Re-download new tickers (ignore the one-day cache)")); dc.addEventListener("change", () => { S.refresh = dc.checked; });
    const sl = h("label", "f", u.b, "Use data from"), si = h("input", "", sl); si.type = "date"; si.min = "1990-01-01"; si.addEventListener("input", () => { S.start = si.value; });
    h("p", "hint", u.b, "Platform ETFs end on " + S.catalog.data_range[1] + " (a versioned snapshot); other tickers run to today. A mixed universe stops at the shorter end and never fills missing prices.");
    sec("s-body", "2. Strategy", true);
    sec("p-body", "3. Portfolio and costs", false);
    const o = sec("o-body", "4. Run", true);
    const nm = h("label", "f", o.b, "Name this run (optional)"), ni = h("input", "", nm); ni.type = "text"; ni.maxLength = 80; ni.addEventListener("input", () => { S.label = ni.value; });
    const vl = h("label", "f inline", o.b), vc = h("input", "", vl); vc.type = "checkbox"; vl.appendChild(document.createTextNode("Run the look-ahead check (replaces the future with noise; slower)")); vc.addEventListener("change", () => { S.validate = vc.checked; });
    const foot = h("div", "runbar", p), run = h("button", "btn primary", foot, "Run backtest"); run.id = "run"; run.addEventListener("click", runBacktest);
    h("div", "progress", foot).id = "progress";
    h("p", "hint", o.b, "Costs: 10 bps per unit traded, monthly rebalance, signals act the next day. Every idea you try on the same tickers counts as a trial against the deflated Sharpe ratio.");
  }

  /* ================================================================= run */
  function buildBody() {
    const tickers = S.tickers.filter((x) => x.status !== "bad");
    if (S.tickers.some((x) => x.status === "pending")) throw new Error("Wait for the ticker checks to finish");
    if (tickers.length < 2) throw new Error("Choose at least two tickers");
    const classes = {}; const isDefault = new Set(S.catalog.default_tickers); for (const x of tickers) if (!isDefault.has(x.t) && x.cls !== "unknown") classes[x.t] = x.cls;
    return { tickers: tickers.map((x) => x.t), classes, start: S.start || null, models: S.models.map((m) => ({ name: m.name, params: m.params })), combination: S.combination,
      allocator: S.allocator || null, allocator_params: S.allocParams, regime: S.regime || null, regime_risk: S.regimeRisk, aum: S.aum ? parseFloat(S.aum) : null,
      validate: S.validate, refresh: S.refresh, label: S.label || null };
  }
  function setBusy(busy, message) {
    S.busy = busy; $("#run").disabled = busy; const p = $("#progress"); p.replaceChildren();
    if (busy) { h("span", "spinner", p); h("span", "", p, message || "Working…").id = "progress-text"; }
    $("#results").classList.toggle("fade", busy && !!S.current);
  }
  async function runBacktest() {
    if (S.busy) return;
    let body; try { body = buildBody(); } catch (e) { toast(e.message); return; }
    setBusy(true, "Submitting…");
    if (!S.current) { const r = $("#results"); r.replaceChildren(); const c = h("div", "card empty", r); h("div", "spinner", c).style.margin = "0 auto 12px"; h("h2", "", c, "Running…"); h("p", "", c, "New tickers are downloaded first, then the strategy runs through the same pipeline as the command line. Neural-network strategies take longer.").id = "wait-note"; }
    try {
      const { job } = await api("/api/backtest", body);
      for (;;) {
        await new Promise((r) => setTimeout(r, 600));
        const v = await api("/api/job/" + job);
        const t = $("#progress-text"); if (t) t.textContent = `${v.message} (${Math.round(v.elapsed)} s)`;
        if (v.status === "done") { addRun(v.result); break; }
        if (v.status === "error") throw new Error(v.error);
      }
    } catch (e) { showError(e.message); }
    setBusy(false);
  }
  function showError(msg) {
    toast("The run failed");
    if (!S.current) { const r = $("#results"); r.replaceChildren(); }
    const old = $("#errbanner"); if (old) old.remove();
    const b = h("div", "banner err"); b.id = "errbanner"; h("strong", "", b, "The run failed."); b.appendChild(document.createTextNode(msg));
    $("#results").prepend(b);
  }
  function addRun(result) {
    const slot = S.nextSlot++ % SLOTS.length, run = { id: S.runs.length + 1, slot, result, at: new Date() };
    S.runs.push(run); S.current = run; S.selected.add(run.id); S.range = "all";
    $("#runcount").textContent = String(S.runs.length);
    S.trials = { n: result.trials_in_universe }; updateTrials();
    renderResult();
  }
  function updateTrials() { const n = S.trials.n || 0; $("#trials").textContent = n ? `${n} trial${n === 1 ? "" : "s"} on this universe` : "no trials yet"; }
  $("#trials").addEventListener("dblclick", async () => { await api("/api/trials/reset", {}); S.trials = {}; updateTrials(); toast("Trial counter reset"); });

  /* ================================================================= window maths */
  function windowOf(run, range) {
    const r = run.result, n = r.dates.length; let i0 = 0;
    if (range !== "all") {
      const yrs = { "10y": 10, "5y": 5, "3y": 3, "1y": 1 }[range], end = Date.parse(r.dates[n - 1]), cut = end - yrs * 365.25 * 864e5;
      i0 = r.dates.findIndex((d) => Date.parse(d) >= cut); if (i0 < 0 || n - i0 < 40) i0 = 0;
    }
    const rebase = (arr) => { const base = i0 > 0 ? arr[i0 - 1] : 1; return arr.slice(i0).map((v) => (v === null || !base ? null : v / base)); };
    const eq = rebase(r.equity), bench = {}; for (const k in r.benchmarks) bench[k] = rebase(r.benchmarks[k]);
    const dd = []; let peak = 0; for (const v of eq) { peak = Math.max(peak, v); dd.push(v / peak - 1); }
    const first = r.dates[i0], year0 = +first.slice(0, 4);
    return { i0, dates: r.dates.slice(i0), equity: eq, bench, drawdown: dd, rolling: r.rolling_sharpe.slice(i0), first,
      monthly: r.monthly.filter((c) => `${c.year}-${String(c.month).padStart(2, "0")}` >= first.slice(0, 7)), annual: r.annual.filter((a) => a.year >= year0),
      exposure: { dates: r.exposure.dates.filter((d) => d >= first), gross: r.exposure.gross.slice(r.exposure.dates.length - r.exposure.dates.filter((d) => d >= first).length), net: r.exposure.net.slice(r.exposure.dates.length - r.exposure.dates.filter((d) => d >= first).length) } };
  }
  function statsOf(eq) {
    const rets = []; for (let i = 1; i < eq.length; i++) rets.push(eq[i] / eq[i - 1] - 1);
    const n = rets.length, mean = rets.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)), end = eq[eq.length - 1] / eq[0];
    let peak = 0, mdd = 0; for (const v of eq) { peak = Math.max(peak, v); mdd = Math.min(mdd, v / peak - 1); }
    return { sharpe: sd > 0 ? (Math.sqrt(252) * mean) / sd : NaN, cagr: Math.pow(end, 252 / n) - 1, vol: sd * Math.sqrt(252), mdd, years: n / 252 };
  }

  /* ================================================================= results */
  function tile(parent, label, value, delta, cls, title) {
    const t = h("div", "tile" + (cls ? " " + cls : ""), parent); if (title) t.title = title;
    h("div", "label", t, label); h("div", "value", t, value); if (delta) { const d = h("div", "delta " + (delta.dir || ""), t, delta.text); }
    return t;
  }
  function chartCard(parent, title, note, opts) {
    const card = h("div", "card chartcard" + (opts && opts.wide ? " wide" : ""), parent), bar = h("div", "bar", card);
    h("h3", "", bar, title);
    const body = h("div", "", card);
    if (note) h("p", "note", card, note);
    const seg = h("div", "seg", bar), a = h("button", "", seg, "Chart"), b = h("button", "", seg, "Table"); a.setAttribute("aria-pressed", "true"); b.setAttribute("aria-pressed", "false");
    let mode = "chart", extra = h("span", "", bar);
    const draw = () => { body.replaceChildren(); if (mode === "chart") opts.draw(body); else opts.table(body); };
    a.addEventListener("click", () => { mode = "chart"; a.setAttribute("aria-pressed", "true"); b.setAttribute("aria-pressed", "false"); draw(); });
    b.addEventListener("click", () => { mode = "table"; b.setAttribute("aria-pressed", "true"); a.setAttribute("aria-pressed", "false"); draw(); });
    card._draw = draw; card._extra = extra; card._body = body; draw();
    return card;
  }
  function simpleTable(parent, cols, rows) {
    const wrap = h("div", "scroll", parent), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t));
    for (const c of cols) h("th", c.num ? "num" : "", hd, c.label);
    const tb = h("tbody", "", t);
    for (const r of rows) { const tr = h("tr", "", tb); for (const c of cols) h("td", c.num ? "num" : "", tr, c.get ? c.get(r) : String(r[c.key])); }
  }
  function sampleRows(dates, cols, every) { const out = []; for (let i = 0; i < dates.length; i += every) out.push(i); if (out[out.length - 1] !== dates.length - 1) out.push(dates.length - 1); return out; }

  function renderResult() {
    const run = S.current; if (!run) return;
    const root = $("#results"); root.replaceChildren();
    const r = run.result, w = windowOf(run, S.range), st = statsOf(w.equity), eqw = w.bench.equal_weight ? statsOf(w.bench.equal_weight) : null;
    const head = h("div", "head", root), left = h("div", "", head);
    h("h1", "", left, r.name);
    h("p", "sub", left, `${r.universe.tickers.length} tickers · ${r.universe.source} · ${r.metrics.start} to ${r.metrics.end} (${(r.stats.years).toFixed(1)} years) · ran in ${r.seconds} s · run #${run.id}`);
    const acts = h("div", "row tight", head);
    const ex = h("button", "btn", acts, "Copy spec (YAML)"); ex.addEventListener("click", () => copy(r.yaml, "Spec"));
    const dn = h("button", "btn", acts, "Download spec"); dn.addEventListener("click", () => download(r.name.replace(/[^\w-]+/g, "_") + ".yaml", r.yaml));
    const cmp = h("button", "btn", acts, "Compare runs"); cmp.addEventListener("click", () => showTab("compare"));
    if (r.warning) { const b = h("div", "banner", root); h("strong", "", b, "Heads up."); b.appendChild(document.createTextNode(r.warning)); }
    const nBad = r.flags.filter((f) => f.flag).length;
    const range = h("div", "row", root); h("span", "hint", range, "Chart window");
    const seg = h("div", "seg", range);
    for (const [k, lab] of [["all", "All"], ["10y", "10y"], ["5y", "5y"], ["3y", "3y"], ["1y", "1y"]]) { const b = h("button", "", seg, lab); b.setAttribute("aria-pressed", String(S.range === k)); b.addEventListener("click", () => { S.range = k; renderResult(); }); }
    h("span", "hint", range, S.range === "all" ? "Return, volatility and drawdown tiles follow the window; costs, turnover and the deflated Sharpe are full-sample." : `Window starts ${w.first}. Tables below always show the full backtest.`);
    const k = h("div", "kpis", root), dl = eqw ? st.sharpe - eqw.sharpe : null;
    tile(k, "Net Sharpe ratio" + (S.range === "all" ? "" : " (window)"), num(st.sharpe), dl === null ? null : { text: `${fmt.signed(dl)} vs equal weight (${num(eqw.sharpe)})`, dir: dl >= 0 ? "up" : "down" }, "hero", "Annualised excess return per unit of volatility, after costs");
    tile(k, "CAGR", pct(st.cagr), eqw ? { text: `equal weight ${pct(eqw.cagr)}` } : null);
    tile(k, "Volatility", pct(st.vol), eqw ? { text: `equal weight ${pct(eqw.vol)}` } : null);
    tile(k, "Max drawdown", pct(st.mdd), eqw ? { text: `equal weight ${pct(eqw.mdd)}` } : null);
    tile(k, "Turnover per year", num(r.metrics.ann_turnover, 1) + "×", { text: `cost drag ${num(r.metrics.ann_cost_bps, 0)} bps/yr` });
    const dsr = r.validation.deflated_sharpe_probability;
    tile(k, "Deflated Sharpe", dsr === null ? "n/a" : num(dsr), { text: `${r.validation.n_trials} trial${r.validation.n_trials === 1 ? "" : "s"}; ≥ 0.95 is the bar`, dir: dsr !== null && dsr >= 0.95 ? "up" : "" }, "", "Probability the Sharpe ratio is above what the best of your tries would show by luck alone");
    const ch = h("div", "charts", root);
    let growthLog = false;
    const series = [{ name: "Strategy", color: css("--s1"), values: w.equity }];
    const bn = { equal_weight: ["Equal weight", "--s2"], risk_parity: ["Risk parity", "--s3"] };
    for (const key in w.bench) series.push({ name: bn[key] ? bn[key][0] : key, color: css(bn[key] ? bn[key][1] : "--s4"), values: w.bench[key] });
    const growth = chartCard(ch, "Growth of $1", "Net of costs. Benchmarks need five or more assets.", { wide: true,
      draw: (b) => C.line(b, { dates: w.dates, series, log: growthLog, yFormat: (v) => "$" + (v >= 10 ? v.toFixed(0) : v.toFixed(2)), tipFormat: (v) => "$" + v.toFixed(2), height: 300, label: "Growth of one dollar" }),
      table: (b) => { const idx = sampleRows(w.dates, series, 21); simpleTable(b, [{ label: "Date", get: (i) => w.dates[i] }, ...series.map((s) => ({ label: s.name, num: true, get: (i) => (s.values[i] === null ? "" : s.values[i].toFixed(3)) }))], idx); } });
    const lg = h("div", "seg", growth._extra), l1 = h("button", "", lg, "Linear"), l2 = h("button", "", lg, "Log"); l1.setAttribute("aria-pressed", "true"); l2.setAttribute("aria-pressed", "false");
    l1.addEventListener("click", () => { growthLog = false; l1.setAttribute("aria-pressed", "true"); l2.setAttribute("aria-pressed", "false"); growth._draw(); });
    l2.addEventListener("click", () => { growthLog = true; l2.setAttribute("aria-pressed", "true"); l1.setAttribute("aria-pressed", "false"); growth._draw(); });
    chartCard(ch, "Drawdown from the previous peak", "How far below its high the strategy has been.", {
      draw: (b) => C.line(b, { dates: w.dates, series: [{ name: "Strategy", color: css("--s1"), values: w.drawdown }], area: true, areaBase: 0, baseline: 0, ceil: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => fmt.pct(v), height: 220, label: "Drawdown" }),
      table: (b) => simpleTable(b, [{ label: "Date", get: (i) => w.dates[i] }, { label: "Drawdown", num: true, get: (i) => fmt.pct(w.drawdown[i]) }], sampleRows(w.dates, null, 21)) });
    chartCard(ch, "Rolling one-year Sharpe ratio", "A strategy that only works in some years shows up here.", {
      draw: (b) => C.line(b, { dates: w.dates, series: [{ name: "1-year Sharpe", color: css("--s1"), values: w.rolling }], baseline: 0, include: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), height: 220, label: "Rolling Sharpe" }),
      table: (b) => simpleTable(b, [{ label: "Date", get: (i) => w.dates[i] }, { label: "Rolling Sharpe", num: true, get: (i) => (w.rolling[i] === null ? "" : w.rolling[i].toFixed(2)) }], sampleRows(w.dates, null, 21)) });
    const years = w.annual;
    chartCard(ch, "Return by calendar year", "Faded bars are partial years.", {
      draw: (b) => C.columns(b, { labels: years.map((a) => a.year), values: years.map((a) => a.value), yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: fmt.spct, faded: years.map((a) => a.partial), height: 240, name: "Return", tipLabel: (i) => years[i].year + (years[i].partial ? ` (${years[i].days} trading days)` : ""), label: "Annual returns" }),
      table: (b) => simpleTable(b, [{ label: "Year", get: (a) => a.year }, { label: "Return", num: true, get: (a) => fmt.spct(a.value) }, { label: "Days", num: true, get: (a) => a.days }], years) });
    chartCard(ch, "Gross and net exposure", "Weekly. Net 0 with gross 1 is a market-neutral book; gross above 1 uses leverage.", {
      draw: (b) => C.line(b, { dates: w.exposure.dates, series: [{ name: "Gross", color: css("--s1"), values: w.exposure.gross }, { name: "Net", color: css("--s2"), values: w.exposure.net }], baseline: 0, include: 0, yFormat: (v) => v.toFixed(1), tipFormat: (v) => v.toFixed(2), height: 240, label: "Exposure" }),
      table: (b) => simpleTable(b, [{ label: "Week", get: (i) => w.exposure.dates[i] }, { label: "Gross", num: true, get: (i) => num(w.exposure.gross[i]) }, { label: "Net", num: true, get: (i) => num(w.exposure.net[i]) }], sampleRows(w.exposure.dates, null, 4)) });
    const wt = r.weights.slice(0, 15).map((x) => ({ label: x.index, value: x.average_weight }));
    chartCard(ch, "Average weight by asset", "Blue is long, red is short. Biggest 15 by absolute weight.", {
      draw: (b) => C.hbars(b, { rows: wt, format: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => fmt.spct(v), name: "Average weight", label: "Average weights" }),
      table: (b) => simpleTable(b, [{ label: "Asset", get: (x) => x.index }, { label: "Average weight", num: true, get: (x) => fmt.spct(x.average_weight) }, { label: "Average |weight|", num: true, get: (x) => pct(x.average_abs_weight) }, { label: "Latest", num: true, get: (x) => fmt.spct(x.latest_weight) }], r.weights) });
    const ct = r.contribution.slice(0, 15).map((x) => ({ label: x.index, value: x.contribution_pct / 100 }));
    if (ct.length) chartCard(ch, "Return contributed by asset", "Total return points over the backtest, before compounding effects.", {
      draw: (b) => C.hbars(b, { rows: ct, format: (v) => (v * 100).toFixed(0) + " pts", tipFormat: (v) => (v * 100).toFixed(1) + " points", name: "Contribution", label: "Contribution by asset" }),
      table: (b) => simpleTable(b, [{ label: "Asset", get: (x) => x.index }, { label: "Contribution (points)", num: true, get: (x) => num(x.contribution_pct, 1) }, { label: "Class", get: (x) => x.asset_class || "" }], r.contribution) });
    chartCard(ch, "Monthly returns", "Blue is a gain and red a loss; shade shows size (capped at ±12%).", { wide: true,
      draw: (b) => C.heatmap(b, { cells: w.monthly, label: "Monthly returns" }),
      table: (b) => simpleTable(b, [{ label: "Month", get: (c) => `${c.year}-${String(c.month).padStart(2, "0")}` }, { label: "Return", num: true, get: (c) => fmt.spct(c.value) }], w.monthly.slice().reverse()) });
    renderDetails(root, r, nBad);
  }
  function renderDetails(root, r, nBad) {
    const flags = h("div", "card", root); h("h2", "", flags, "Red flags: fixed rules, each with its threshold"); h("p", "sub", flags, `${nBad} of ${r.flags.length} rules flagged. A flag is a question, not a verdict.`);
    for (const f of r.flags) { const row = h("div", "flag " + (f.flag ? "bad" : "ok"), flags); h("span", "ic", row, f.flag ? "⚑" : "✓"); const d = h("div", "", row); d.appendChild(document.createTextNode(f.rule + " "));
      h("b", "", d, f.flag ? "FLAG" : "ok"); h("small", "", d, `Observed ${f.observed}. ${f.why}`); }
    const T = r.tables, tabs = [];
    const add = (id, label, draw) => tabs.push({ id, label, draw });
    const tab = (key, cols) => (b) => simpleTable(b, cols || Object.keys(T[key][0]).map((c, i) => ({ label: c.replace(/_/g, " "), num: typeof T[key][0][c] === "number", get: (x) => cell(x[c]) })), T[key]);
    if (T.benchmarks) add("bench", "Against benchmarks", tab("benchmarks"));
    if (T.by_sample) add("sample", "By period", tab("by_sample"));
    if (T.by_regime) add("regime", "By regime", tab("by_regime"));
    if (T.attribution) add("attr", "Where the return came from", tab("attribution"));
    if (T.attribution_vs_benchmark) add("brinson", "Brinson vs equal weight", tab("attribution_vs_benchmark"));
    if (T.attribution_by_model) add("bymodel", "By model", tab("attribution_by_model"));
    if (T.cost_breakdown) add("cost", "Costs", tab("cost_breakdown"));
    if (T.capacity) add("cap", "Capacity", tab("capacity"));
    if (T.risk_limits) add("limits", "Risk limits", tab("risk_limits"));
    if (T.alpha_decay) add("decay", "Alpha decay", tab("alpha_decay"));
    for (const key of Object.keys(T)) if (key.startsWith("explain_")) add(key, "What drives " + key.slice(8), tab(key));
    const fq = Object.entries(r.metrics).filter(([k]) => k.startsWith("forecast_") && typeof r.metrics[k] === "number");
    if (fq.length) add("fq", "Forecast quality", (b) => simpleTable(b, [{ label: "Measure", get: (x) => x[0].replace("forecast_", "").replace(/_/g, " ") }, { label: "Value", num: true, get: (x) => num(x[1], 4) }], fq));
    if (r.validation.causality) add("caus", "Look-ahead check", (b) => simpleTable(b, [{ label: "Component", get: (x) => x[0] }, { label: "Result", get: (x) => (x[1].ok ? "✓ passes" : "✗ FAILS") }, { label: "Largest change when the future was replaced by noise", num: true, get: (x) => String(x[1].max_abs_difference) }], Object.entries(r.validation.causality)));
    add("spec", "Specification (YAML)", (b) => { const pre = h("pre", "", b); h("code", "", pre, r.yaml); });
    const card = h("div", "card", root); h("h2", "", card, "Details"); const bar = h("div", "tabsmall", card), body = h("div", "", card);
    const open = (t) => { [...bar.children].forEach((c) => c.setAttribute("aria-selected", String(c.dataset.id === t.id))); body.replaceChildren(); t.draw(body); };
    tabs.forEach((t) => { const b = h("button", "", bar, t.label); b.dataset.id = t.id; b.setAttribute("aria-selected", "false"); b.addEventListener("click", () => open(t)); });
    if (tabs.length) open(tabs[0]);
  }
  function cell(v) { if (v === null || v === undefined) return ""; if (typeof v === "number") return Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3); return String(v); }

  function renderEmpty() {
    const root = $("#results"); root.replaceChildren();
    const c = h("div", "card empty", root); h("h2", "", c, "Pick tickers and a strategy, then run it");
    h("p", "", c, "You get net-of-cost performance against equal weight and risk parity, drawdowns, rolling Sharpe, monthly and annual returns, exposure, red flags and a deflated Sharpe ratio that counts how many ideas you tried.");
    h("p", "", c, "Nothing is tuned for you. A strategy that does not beat equal weight after costs will say so.");
    const row = h("div", "starters", c);
    const starters = [["Dual momentum on the 15 ETFs", () => { S.tickers = defaultTickers(); S.models = [{ name: "dual_momentum", params: {} }]; S.allocatorTouched = false; autoAllocator(); }],
      ["12-1 momentum formula on tech stocks", () => { S.tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO"].map((t) => ({ t, cls: "equity", status: "ok" })); S.start = ""; S.models = [{ name: "expression", params: { expr: "mom(252, 21)" } }]; S.allocatorTouched = false; autoAllocator(); }],
      ["Trend filter on stocks, bonds, gold and bitcoin", () => { S.tickers = ["SPY", "TLT", "GLD", "BTC-USD", "VNQ", "DBC"].map((t) => ({ t, cls: t === "BTC-USD" ? "crypto" : S.catalog.default_classes[t] || "unknown", status: S.catalog.default_tickers.includes(t) ? "default" : "ok" })); S.models = [{ name: "expression", params: { expr: "where(close > sma(200), 1, -1)", mode: "time_series" } }]; S.allocatorTouched = false; autoAllocator(); }]];
    for (const [label, apply] of starters) { const b = h("button", "btn", row, label); b.addEventListener("click", () => { apply(); renderUniverse(); renderStrategies(); renderPortfolio(); runBacktest(); }); }
  }

  /* ================================================================= compare */
  function renderCompare() {
    const root = $("#compare"); root.replaceChildren();
    const head = h("div", "head", root), l = h("div", "", head); h("h1", "", l, "Compare runs"); h("p", "sub", l, "Every run you make this session. Tick up to six to overlay them. Colour belongs to the run, so it never changes when you tick or untick others.");
    if (!S.runs.length) { const c = h("div", "card empty", root); h("h2", "", c, "No runs yet"); h("p", "", c, "Run a backtest and it appears here."); const b = h("button", "btn primary", c, "Go to Backtest"); b.addEventListener("click", () => showTab("backtest")); return; }
    const clr = h("button", "btn", head, "Clear all runs"); clr.addEventListener("click", () => { S.runs = []; S.selected.clear(); S.current = null; $("#runcount").textContent = "0"; renderCompare(); renderEmpty(); });
    const card = h("div", "card", root), list = h("div", "runs", card);
    const hd = h("div", "run header", list); for (const t of ["", "", "Run", "Sharpe", "CAGR", "Vol", "Max DD", "Deflated"]) h("span", t === "Run" || !t ? "" : "n hide-s", hd, t); h("span", "", hd);
    for (const run of S.runs) {
      const r = run.result, row = h("div", "run", list), cb = h("input", "", row); cb.type = "checkbox"; cb.checked = S.selected.has(run.id); cb.setAttribute("aria-label", "show " + r.name);
      cb.addEventListener("change", () => { if (cb.checked) { if (S.selected.size >= 6) { cb.checked = false; toast("Six runs at most on one chart"); return; } S.selected.add(run.id); } else S.selected.delete(run.id); drawCompare(chart); });
      const sw = h("span", "sw", row); sw.style.setProperty("--c", css(SLOTS[run.slot]));
      const nm = h("div", "nm", row); nm.appendChild(document.createTextNode(`#${run.id} ${r.name}`)); h("small", "", nm, `${r.universe.tickers.length} tickers: ${r.universe.tickers.slice(0, 6).join(", ")}${r.universe.tickers.length > 6 ? "…" : ""} · ${r.metrics.start.slice(0, 4)}–${r.metrics.end.slice(0, 4)}`);
      h("span", "n", row, num(r.metrics.sharpe)); h("span", "n", row, pct(r.metrics.cagr)); h("span", "n hide-s", row, pct(r.metrics.ann_vol)); h("span", "n hide-s", row, pct(r.metrics.max_drawdown));
      h("span", "n hide-s", row, r.validation.deflated_sharpe_probability === null ? "n/a" : num(r.validation.deflated_sharpe_probability));
      const rm = h("button", "iconbtn", row, "×"); rm.setAttribute("aria-label", "remove run"); rm.addEventListener("click", () => { S.runs = S.runs.filter((x) => x !== run); S.selected.delete(run.id); if (S.current === run) S.current = S.runs[S.runs.length - 1] || null; $("#runcount").textContent = String(S.runs.length); renderCompare(); if (S.current) renderResult(); else renderEmpty(); });
    }
    const chart = h("div", "card chartcard", root); drawCompare(chart);
    h("p", "hint", root, "Sharpe ratios of runs on different tickers or periods are not directly comparable. The deflated Sharpe already accounts for how many ideas you tried on the same tickers; it does not account for ideas you tried elsewhere.");
  }
  function drawCompare(card) {
    card.replaceChildren(); const runs = S.runs.filter((x) => S.selected.has(x.id));
    h("h3", "", card, "Growth of 100 from the latest common start");
    if (!runs.length) { h("p", "note", card, "Tick at least one run."); return; }
    const start = runs.map((x) => x.result.dates[0]).sort().pop(), ref = runs.find((x) => x.result.dates[0] === start) || runs[0], dates = ref.result.dates.filter((d) => d >= start);
    const body = h("div", "", card);
    const series = runs.map((x) => { const m = new Map(x.result.dates.map((d, i) => [d, x.result.equity[i]])), base = m.get(start) || x.result.equity[x.result.dates.findIndex((d) => d >= start)]; return { name: `#${x.id} ${x.result.name}`.slice(0, 40), color: css(SLOTS[x.slot]), values: dates.map((d) => (m.has(d) ? (100 * m.get(d)) / base : null)) }; });
    const lgSeg = h("div", "seg", card), a = h("button", "", lgSeg, "Linear"), b = h("button", "", lgSeg, "Log"); let log = false; a.setAttribute("aria-pressed", "true"); b.setAttribute("aria-pressed", "false"); lgSeg.style.marginBottom = "6px";
    const draw = () => C.line(body, { dates, series, log, yFormat: (v) => v.toFixed(0), tipFormat: (v) => v.toFixed(1), height: 320, label: "Growth of 100", endLabels: false });
    a.addEventListener("click", () => { log = false; a.setAttribute("aria-pressed", "true"); b.setAttribute("aria-pressed", "false"); draw(); });
    b.addEventListener("click", () => { log = true; b.setAttribute("aria-pressed", "true"); a.setAttribute("aria-pressed", "false"); draw(); });
    draw();
  }

  /* ================================================================= strategy builder */
  const B = { expr: "rank(mom(126, 21)) - rank(vol(63))", mode: "cross_sectional" };
  function renderBuilder() {
    const root = $("#builder"); root.replaceChildren();
    const head = h("div", "head", root), l = h("div", "", head); h("h1", "", l, "Strategy builder"); h("p", "sub", l, "Two ways to test your own idea: a one-line formula (no code), or a Python file for anything bigger.");
    const grid = h("div", "builder", root), left = h("div", "stack", grid), right = h("div", "stack", grid);
    const f = h("div", "card", left); h("h2", "", f, "1. Write a formula"); h("p", "sub", f, "The formula gives every asset a score each day. Higher means you expect a higher return. It may only look backwards, so it cannot cheat.");
    const ta = h("textarea", "", f); ta.rows = 3; ta.value = B.expr; ta.spellcheck = false; ta.setAttribute("aria-label", "formula"); ta.addEventListener("input", () => { B.expr = ta.value; });
    const mrow = h("div", "row", f);
    const ms = h("select", "", mrow); ms.style.width = "auto"; for (const [v, t] of [["cross_sectional", "Rank assets against each other"], ["time_series", "Each asset on its own"]]) { const o = h("option", "", ms, t); o.value = v; if (v === B.mode) o.selected = true; } ms.addEventListener("change", () => { B.mode = ms.value; });
    const chk = h("button", "btn", mrow, "Check on my tickers"), go = h("button", "btn primary", mrow, "Backtest this formula");
    const out = h("div", "", f);
    chk.addEventListener("click", async () => {
      out.replaceChildren(); h("span", "spinner", out);
      try {
        const body = { expr: B.expr, mode: B.mode, tickers: S.tickers.filter((x) => x.status !== "bad").map((x) => x.t), start: S.start || null };
        const res = await api("/api/formula", body); out.replaceChildren();
        if (!res.ok) { const b = h("div", "banner err", out); h("strong", "", b, "Not valid."); b.appendChild(document.createTextNode(res.error)); return; }
        h("p", "hint", out, `Latest scores on ${res.as_of}; the formula has values for ${(res.coverage * 100).toFixed(0)}% of asset-days from ${res.first}. Top of the list is what the formula would buy first.`);
        const host = h("div", "", out); C.hbars(host, { rows: res.latest.slice(0, 20).map((x) => ({ label: x.ticker, value: x.score })), format: (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)), name: "Score", label: "Latest scores" });
      } catch (e) { out.replaceChildren(); const b = h("div", "banner err", out); h("strong", "", b, "Could not check."); b.appendChild(document.createTextNode(e.message)); }
    });
    go.addEventListener("click", () => { S.models = [{ name: "expression", params: { expr: B.expr, mode: B.mode } }]; S.allocatorTouched = false; autoAllocator(); renderStrategies(); renderPortfolio(); showTab("backtest"); runBacktest(); });
    h("p", "hint", f, "Backtests of formulas trade the score as written (rank and scale it, no calibration), so a fade like -ret(5) really fades. Change Allocation in the Backtest tab to calibrate it against history instead.");
    const ex = h("div", "card", left); h("h3", "", ex, "Examples: click to load"); const eb = h("div", "examples", ex);
    for (const e of S.catalog.formula.examples) { const b = h("button", "btn", eb, e.title); b.title = e.expr; b.addEventListener("click", () => { B.expr = e.expr; B.mode = e.mode; renderBuilder(); }); }
    h("p", "hint", ex, "Combine anything: rank(mom(126, 21)) - rank(vol(63)) is momentum among the calm. Conditions use & and |: where((close > sma(50)) & (close > sma(200)), 1, 0).");
    const ref = h("div", "card", right); h("h2", "", ref, "Function reference"); h("p", "sub", ref, "Click one to insert it. x is any table-valued expression; n is a whole number of days.");
    const search = h("input", "", ref); search.type = "text"; search.placeholder = "Filter…"; search.setAttribute("aria-label", "filter functions");
    const list = h("div", "fnlist", ref);
    const draw = () => { list.replaceChildren(); const q = search.value.toLowerCase(); for (const fn of S.catalog.formula.functions) { if (q && !(fn.name + fn.help).toLowerCase().includes(q)) continue; const b = h("button", "fn", list); b.type = "button"; h("code", "", b, fn.signature); h("span", "", b, fn.help); b.addEventListener("click", () => { const pos = ta.selectionStart; ta.setRangeText(fn.signature.replace(/\(.*\)/, (m) => m), pos, ta.selectionEnd, "end"); B.expr = ta.value; ta.focus(); }); } };
    search.addEventListener("input", draw); draw();
    const py = h("div", "card", right); h("h2", "", py, "2. Write it in Python"); h("p", "sub", py, "For ideas that need more than a formula: your own parameters, several steps, extra data.");
    const steps = h("ol", "", py); steps.style.paddingLeft = "20px";
    for (const t of ["Create a template: run the command below in a terminal.", "Open user_strategies/my_idea.py and replace the body of score(): return a table of dates by assets, higher = expect a higher return, using only past data.", "Press Reload below. Your strategy appears in the Strategy list under “My strategies”, with its parameters as form fields."]) h("li", "", steps, t);
    const cmd = h("pre", "", py); h("code", "", cmd, "quant new-strategy my_idea"); const cb = h("button", "btn", py, "Copy command"); cb.addEventListener("click", () => copy("quant new-strategy my_idea", "Command"));
    const tpl = h("details", "", py); tpl.style.marginTop = "10px"; h("summary", "", tpl, "See what the template looks like"); const pre = h("pre", "", tpl); h("code", "", pre, S.catalog.template);
    const row = h("div", "row", py); row.style.marginTop = "10px";
    const reload = h("button", "btn primary", row, "Reload my strategies"), guide = h("button", "btn", row, "Read the full guide"); guide.addEventListener("click", () => { showTab("guides"); loadDoc("how_to_add_a_strategy"); });
    const status = h("div", "", py);
    reload.addEventListener("click", async () => {
      status.replaceChildren();
      try { const res = await api("/api/reload", {}); S.catalog = await api("/api/catalog"); renderStrategies(); renderPortfolio();
        const names = Object.keys(res.files); if (!names.length) h("p", "hint", status, "No files in user_strategies/ yet. Create one with the command above.");
        for (const n of names) { const p = h("p", "hint", status, (res.files[n] ? "✗ " : "✓ ") + n + (res.files[n] ? ": " + res.files[n] : " loaded")); if (res.files[n]) p.style.color = css("--crit"); }
      } catch (e) { h("p", "hint", status, e.message); }
    });
    h("p", "hint", py, "Files in user_strategies/ run as ordinary Python on this computer, so only put code there that you wrote or trust. The browser never sends code to the server.");
  }

  /* ================================================================= guides + markdown */
  const G = { index: null, slug: null, filter: "" };
  const FILE_SLUGS = { "dashboard.md": "dashboard", "how_to_add_a_strategy.md": "how_to_add_a_strategy", "START_HERE.md": "start_here", "glossary.md": "glossary", "tour_of_a_backtest_day.md": "tour_of_a_backtest_day", "roadmap_coverage.md": "roadmap_coverage", "platform_integration.md": "platform_integration", "feature_audit.md": "feature_audit" };
  function slugFromHref(href) {
    const clean = href.split("#")[0], base = clean.split("/").pop();
    if (/techniques\//.test(clean) || (/^[a-z0-9-]+\.md$/.test(clean) && G.index && G.index.some((d) => d.slug === "technique-" + base.replace(".md", "")))) return "technique-" + base.replace(".md", "");
    if (/strategies\//.test(clean)) return "strategy-" + base.replace(".md", "");
    return FILE_SLUGS[base] || null;
  }
  function inline(text, parent) {
    const re = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\[[^\]]+\]\([^)\s]+\))/g; let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      const t = m[0];
      if (t[0] === "`") h("code", "", parent, t.slice(1, -1));
      else if (t.startsWith("**")) inline(t.slice(2, -2), h("strong", "", parent));
      else if (t[0] === "*") inline(t.slice(1, -1), h("em", "", parent));
      else {
        const [, label, href] = /\[([^\]]+)\]\(([^)\s]+)\)/.exec(t);
        if (/^https?:\/\//.test(href)) { const a = h("a", "", parent, label); a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer"; }
        else { const slug = slugFromHref(href); if (slug) { const a = h("a", "", parent, label); a.href = "#" + slug; a.addEventListener("click", (e) => { e.preventDefault(); loadDoc(slug); }); } else parent.appendChild(document.createTextNode(label)); }
      }
      last = m.index + t.length;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
  }
  function markdown(md, root) {
    const lines = md.replace(/\r/g, "").split("\n"); let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^```/.test(line)) { const buf = []; i++; while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]); i++; const pre = h("pre", "", root); h("code", "", pre, buf.join("\n")); continue; }
      let m;
      if ((m = /^(#{1,4})\s+(.*)$/.exec(line))) { const el = h("h" + Math.min(m[1].length, 4), "", root); inline(m[2], el); i++; continue; }
      if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { h("hr", "", root); i++; continue; }
      if (/^\|/.test(line) && i + 1 < lines.length && /^\|?\s*:?-{2,}/.test(lines[i + 1])) {
        const cells = (s) => s.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()); const t = h("table", "", root), head = cells(line); i += 2;
        const tr = h("tr", "", h("thead", "", t)); for (const c of head) inline(c, h("th", "", tr));
        const tb = h("tbody", "", t); while (i < lines.length && /^\|/.test(lines[i])) { const r = h("tr", "", tb); for (const c of cells(lines[i])) inline(c, h("td", "", r)); i++; } continue;
      }
      if (/^>\s?/.test(line)) { const bq = h("blockquote", "", root), buf = []; while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, "")); inline(buf.join(" "), h("p", "", bq)); continue; }
      if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
        const ordered = /^\s*\d+\./.test(line), list = h(ordered ? "ol" : "ul", "", root); let lastLi = null;
        while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
          const mm = /^(\s*)([-*]|\d+\.)\s+(.*)$/.exec(lines[i]); const indent = mm[1].length;
          if (indent >= 2 && lastLi) { let sub = lastLi.querySelector(":scope > ul"); if (!sub) sub = h("ul", "", lastLi); inline(mm[3], h("li", "", sub)); } else { lastLi = h("li", "", list); inline(mm[3], lastLi); }
          i++;
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s+/.test(lines[i])) { lastLi.appendChild(document.createTextNode(" " + lines[i].trim())); i++; }
        }
        continue;
      }
      if (!line.trim()) { i++; continue; }
      const buf = []; while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|>|\||\s*([-*]|\d+\.)\s+|\s*(-{3,}|\*{3,})\s*$)/.test(lines[i])) buf.push(lines[i++]);
      if (!buf.length) { buf.push(lines[i++]); } inline(buf.join(" "), h("p", "", root));
    }
  }
  async function renderGuides() {
    const root = $("#guides");
    if (!G.index) {
      root.replaceChildren(); h("div", "card", root, "Loading guides…");
      try { G.index = (await api("/api/docs")).docs; } catch (e) { root.replaceChildren(); h("div", "card", root, e.message); return; }
    }
    root.replaceChildren();
    const toc = h("nav", "card toc", root); toc.setAttribute("aria-label", "Guides");
    const search = h("input", "", toc); search.type = "text"; search.placeholder = "Search guides…"; search.value = G.filter; search.setAttribute("aria-label", "search guides");
    const list = h("div", "", toc);
    const draw = () => {
      list.replaceChildren(); let group = null; const q = search.value.toLowerCase();
      for (const d of G.index) {
        if (q && !(d.title + " " + d.slug).toLowerCase().includes(q)) continue;
        if (d.group !== group) { group = d.group; h("h4", "", list, group); }
        const b = h("button", "", list, d.title); if (d.slug === G.slug) b.setAttribute("aria-current", "true"); b.addEventListener("click", () => loadDoc(d.slug));
      }
    };
    search.addEventListener("input", () => { G.filter = search.value; draw(); }); draw();
    const doc = h("article", "card md", root); doc.id = "doc";
    if (!G.slug) loadDoc(store("doc") || "dashboard"); else loadDoc(G.slug);
  }
  async function loadDoc(slug) {
    G.slug = slug; store("doc", slug);
    if ($("#view-guides").hidden) showTab("guides");
    const doc = $("#doc"); if (!doc) return; doc.replaceChildren(); h("p", "", doc, "Loading…");
    try {
      const d = await api("/api/docs/" + encodeURIComponent(slug)); doc.replaceChildren(); markdown(d.markdown, doc); doc.scrollIntoView({ block: "start" }); window.scrollTo({ top: 0 });
      document.querySelectorAll(".toc button").forEach((b) => b.removeAttribute("aria-current"));
      const idx = G.index.findIndex((x) => x.slug === slug); if (idx >= 0) [...document.querySelectorAll(".toc button")].find((b) => b.textContent === G.index[idx].title)?.setAttribute("aria-current", "true");
    } catch (e) { doc.replaceChildren(); const b = h("div", "banner err", doc); b.textContent = e.message; }
  }

  /* ================================================================= redraw and boot */
  function redraw() { if (!$("#view-backtest").hidden && S.current) renderResult(); if (!$("#view-compare").hidden) renderCompare(); if (!$("#view-builder").hidden) renderBuilder(); }
  let rz; new ResizeObserver(() => { clearTimeout(rz); rz = setTimeout(() => { if (!$("#view-backtest").hidden && S.current && !S.busy) renderResult(); }, 180); }).observe($("main"));

  async function boot() {
    try { S.catalog = await api("/api/catalog"); } catch (e) { $("#results").textContent = "Could not reach the server: " + e.message; return; }
    const savedT = store("tickers"); S.tickers = savedT && savedT.length ? savedT.map((x) => ({ t: x.t, cls: x.cls, status: S.catalog.default_tickers.includes(x.t) ? "default" : "ok" })) : defaultTickers();
    S.models = [{ name: S.catalog.models.some((m) => m.name === "dual_momentum") ? "dual_momentum" : S.catalog.models[0].name, params: {} }];
    buildPanel(); renderUniverse(); renderStrategies(); renderPortfolio(); renderEmpty(); updateTrials();
    const t = store("tab"); if (t && TABS.includes(t) && t !== "backtest") showTab(t);
  }
  boot();
})();
