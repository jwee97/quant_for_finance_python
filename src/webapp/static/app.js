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
    if (!res.ok) { const error = new Error(data.error || res.statusText); error.status = res.status; throw error; }
    return data;
  }
  /* The server API this page was written for (API_VERSION in api.py). A server that reports less is older than the page: the files were updated while the app was running, so it still runs the
     old code. Say so, instead of drawing half a page. */
  const PAGE_API = 2;                                                                       // what the whole page needs; the Execution and Cash flows tabs ask for more (LAB_API below) and say so on their own
  const RESTART = "Stop the app and press Run again (on Replit: the Stop button, then Run), then reload this page. Your saved tickers and runs are kept.";
  /* A banner above the page for what the person has to act on. One banner per ``id``, so repeating it does not stack. */
  function notice(id, title, text, serious) {
    const box = $("#notice"); let el = box.querySelector(`[data-notice="${id}"]`);
    if (!el) { el = h("div", "banner" + (serious ? " err" : ""), box); el.setAttribute("data-notice", id); }
    el.replaceChildren(); h("strong", "", el, title); el.appendChild(document.createTextNode(text)); box.hidden = false;
  }
  function store(key, value) { try { if (value === undefined) return JSON.parse(localStorage.getItem("ql." + key)); localStorage.setItem("ql." + key, JSON.stringify(value)); } catch (e) { return null; } }
  function toast(msg) { const t = h("div", "toast", document.body, msg); setTimeout(() => t.remove(), 3500); }
  function copy(text, what) { (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(() => toast(`${what} copied`), () => toast("Copy is blocked in this browser: select the text and copy it by hand")); }
  function download(name, text) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
  const unsign = (s) => (/^-0(\.0+)?$/.test(s) ? s.slice(1) : s);                                   // a value that rounds to zero is not "-0.0"
  const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "n/a" : unsign(v.toFixed(d)));
  const pct = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "n/a" : unsign((v * 100).toFixed(d)) + "%");
  const bad = (v) => v === null || v === undefined || !isFinite(v);
  /* "\u2060" (word joiner) keeps the minus sign on the same line as the amount. */
  const money = (v, d = 0) => (bad(v) ? "n/a" : (v < 0 ? "−\u2060$" : "$") + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
  const smoney = (v) => (bad(v) ? "n/a" : (v >= 0 ? "+\u2060$" : "−\u2060$") + Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 }));
  const axisMoney = (v) => (v < 0 ? "−" : "") + "$" + (Math.abs(v) >= 1e6 ? (Math.abs(v) / 1e6).toFixed(Math.abs(v) >= 1e7 ? 0 : 1) + "M" : Math.abs(v) >= 1e4 ? (Math.abs(v) / 1e3).toFixed(0) + "k" : Math.round(Math.abs(v)).toLocaleString("en-US"));

  /* ================================================================= state */
  const S = {
    catalog: null, tickers: [], start: "", models: [], combination: "equal", allocator: "", allocatorTouched: false, allocParams: {}, regime: "", regimeRisk: false,
    aum: "", capital: "", source: "yahoo", guard: { ok: true, message: "" }, validate: false, refresh: false, label: "", runs: [], selected: new Set(), current: null, range: "all", busy: false, nextSlot: 0, trials: {},
    view: freshView(),
  };
  /* What the user has chosen inside a result: which cards show their table, linear or log growth, which details tab is open. A redraw (new window size, theme change) rebuilds the page
     from the data, so these choices live here and not in the DOM; a new run starts with a fresh view. */
  function freshView() { return { modes: {}, log: false, detail: null }; }

  /* ================================================================= theme, tabs */
  function setTheme(t) { if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme"); store("theme", t); redraw(); }
  $("#theme").addEventListener("click", () => { const dark = document.documentElement.getAttribute("data-theme") ? document.documentElement.getAttribute("data-theme") === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; setTheme(dark ? "light" : "dark"); });
  const saved = store("theme"); if (saved) document.documentElement.setAttribute("data-theme", saved);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => redraw());

  const TABS = ["backtest", "compare", "builder", "exec", "cash", "guides"];
  function showTab(name) {
    for (const t of TABS) { $("#tab-" + t).setAttribute("aria-selected", String(t === name)); $("#view-" + t).hidden = t !== name; }
    if (name === "compare") renderCompare();
    if (name === "builder") renderBuilder();
    if (name === "exec") renderLab(X);
    if (name === "cash") renderLab(K);
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
      if (x.status === "ok") { const ok = h("span", "meta", chip, x.later ? "↓" : "✓"); ok.style.color = css("--good-text"); chip.title = x.info || ""; }
      if (x.warn && x.warn.length) { const wn = h("span", "meta", chip, "⚠"); wn.style.color = css("--warn"); wn.title = x.warn.join("\n"); }
      if (x.status === "bad") { h("span", "meta", chip, "✗"); chip.title = x.error || ""; }
      const rm = h("button", "", chip, "×"); rm.setAttribute("aria-label", "remove " + x.t); rm.title = "Remove";
      rm.addEventListener("click", () => { S.tickers = S.tickers.filter((y) => y !== x); saveUniverse(); renderUniverse(); });
    }
    const failed = S.tickers.filter((x) => x.status === "bad"), byError = {};
    for (const x of failed) (byError[x.error] = byError[x.error] || []).push(x.t);
    if (failed.length) { const p = h("p", "hint", box, Object.entries(byError).map(([msg, list]) => `${list.join(", ")}: ${msg}`).join(" · ")); p.style.color = css("--crit"); p.style.flexBasis = "100%"; }
    const warned = S.tickers.filter((x) => x.warn && x.warn.length);
    if (warned.length) { const p = h("p", "hint", box, "⚠ " + warned.map((x) => `${x.t}: ${x.warn[0]}`).join(" · ")); p.style.color = css("--warn"); p.style.flexBasis = "100%"; }
    const later = S.tickers.filter((x) => x.later);
    if (later.length) {
      const total = later.reduce((a, x) => a + (x.calls || 0), 0), src = sources().find((x) => x.name === S.source), perMin = src && src.calls_per_minute;
      const cost = total ? `: about ${total} call${total === 1 ? "" : "s"}` + (perMin ? `, around ${Math.max(1, Math.ceil(total / perMin))} minute${Math.ceil(total / perMin) > 1 ? "s" : ""} at ${perMin} a minute` : "") + " (once; a later start date means fewer)" : "";
      const p = h("p", "hint", box, "↓ " + later.map((x) => x.t).join(", ") + (later.length === 1 ? " is" : " are") + " downloaded when you run" + cost + "."); p.style.flexBasis = "100%";
    }
    const n = S.tickers.filter((x) => x.status !== "bad").length;
    $("#u-count").textContent = `${n} ticker${n === 1 ? "" : "s"}`;
    refreshGuard();
  }
  function applyChecks(list, res) {
    for (const t of list) {
      const x = S.tickers.find((y) => y.t === t), r = res.tickers[t]; if (!x || !r) continue;
      x.later = false; x.warn = r.warnings || [];
      if (r.ok) { x.status = "ok"; x.cls = r.class && r.class !== "unknown" ? r.class : x.cls; x.error = null; x.later = !!r.pending; x.calls = r.calls || 0; x.info = r.pending ? r.note : `${r.days.toLocaleString()} days from ${r.first.slice(0, 4)}`; }
      else { x.status = "bad"; x.error = r.error; x.info = null; }
    }
  }
  const TICKER_SYNTAX = /^[A-Z0-9^][A-Z0-9.^=-]{0,11}$/;
  async function addTickers(text) {
    const raw = text.split(/[\s,;]+/).map((s) => s.trim().toUpperCase()).filter(Boolean);
    const fresh = raw.filter((t) => !S.tickers.some((x) => x.t === t));
    if (!fresh.length) { toast(raw.length ? "Those tickers are already in the list" : "Type one or more tickers first"); return; }
    const valid = [];
    for (const t of fresh) {                                                            // a misspelt symbol is refused here, so it cannot spoil the check of the others
      if (TICKER_SYNTAX.test(t)) { S.tickers.push({ t, cls: "unknown", status: "pending" }); valid.push(t); }
      else S.tickers.push({ t, cls: "unknown", status: "bad", error: "not a valid ticker (letters, digits and . ^ = - only, up to 12 characters)" });
    }
    renderUniverse();
    if (!valid.length) { toast(`${fresh[0]} is not a valid ticker`); saveUniverse(); return; }
    try {
      const res = await api("/api/tickers", { tickers: valid, start: S.start || undefined, refresh: S.refresh, source: S.source });
      applyChecks(valid, res);
      const failed = valid.filter((t) => (res.tickers[t] || {}).ok === false);
      if (failed.length) toast(`${failed.join(", ")}: ${res.tickers[failed[0]].error}`);
    } catch (e) {
      for (const t of valid) { const x = S.tickers.find((y) => y.t === t); if (x) { x.status = "bad"; x.error = "could not check"; } }
      toast(e.message);
    }
    saveUniverse(); renderUniverse();
  }

  /* Check the tickers that came from a download source again (the source changed, or a run has downloaded the ones that were waiting). Only a source that downloads while checking is slow. */
  async function recheckTickers(onlyLater) {
    const list = S.tickers.filter((x) => x.status !== "default" && (!onlyLater || x.later));
    if (!list.length) return;
    if (!onlyLater) { list.forEach((x) => { x.status = "pending"; x.later = false; }); renderUniverse(); }
    try { applyChecks(list.map((x) => x.t), await api("/api/tickers", { tickers: list.map((x) => x.t), start: S.start || undefined, refresh: false, source: S.source })); }
    catch (e) { list.forEach((x) => { if (x.status === "pending") { x.status = "bad"; x.error = "could not check"; } }); toast(e.message); }
    saveUniverse(); renderUniverse();
  }
  const sources = () => (S.catalog && Array.isArray(S.catalog.sources) ? S.catalog.sources : []);
  /* What turns iTick on. The key is never typed into this page: it goes into a Replit secret, and an app only sees the secrets that existed when it started. */
  const KEY_NAME = "ITICK_API_KEY";
  function setupGuide(box) {
    const guide = h("div", "setup", box);
    const lead = h("p", "", guide, "iTick is not set up yet. It takes about two minutes, and the key never goes through this page."); lead.style.color = css("--warn");
    const steps = h("ol", "steps", guide);
    h("li", "", steps, "Copy your API key from your iTick account.");
    const two = h("li", "", steps, "In Replit open Tools, then Secrets, then New Secret. Name it "); h("code", "", two, KEY_NAME); two.appendChild(document.createTextNode(", paste the key as its value and click Add Secret."));
    h("li", "", steps, "Press Stop and then Run: an app only sees the secrets that existed when it started. Then reload this page.");
    h("li", "", steps, "Choose iTick here and press Test the iTick connection (it spends one call).");
    const row = h("div", "row tight", guide);
    const name = h("button", "btn", row, "Copy the secret name"); name.addEventListener("click", () => copy(KEY_NAME, "The secret name"));
    const again = h("button", "btn", row, "I added it and restarted: reload"); again.addEventListener("click", () => location.reload());
    h("p", "hint", guide, "The key is read by the server only and is never sent to this page. Do not paste it into a chat or a file; people you add to the Repl can see its secrets.");
  }
  function renderSource() {
    const box = $("#u-source"); if (!box) return; box.replaceChildren();
    const lab = h("label", "f", box, "Download other tickers from"), sel = h("select", "", lab); sel.id = "u-source-select";
    for (const src of sources()) { const o = h("option", "", sel, src.label + (src.available ? "" : " (not set up)")); o.value = src.name; if (src.name === S.source) o.selected = true; }
    sel.addEventListener("change", () => { S.source = sel.value; store("source", S.source); renderSource(); recheckTickers(false); });
    const cur = sources().find((x) => x.name === S.source); if (!cur) return;
    h("p", "hint", box, cur.note);
    if (!cur.available) {
      if (cur.setup === "key") setupGuide(box);
      else { const w = h("p", "hint", box, cur.problem); w.style.color = css("--warn"); }
    } else if (cur.name === "itick") {
      h("p", "hint", box, `Your plan allows ${cur.calls_per_minute} call${cur.calls_per_minute === 1 ? "" : "s"} a minute; the app waits between calls and says so while it does. A ticker costs about one call per four years of daily history, once; after that a top-up is one call.`);
    }
    if (cur.symbols) { const more = h("button", "link", box, "How to write symbols"), para = h("p", "hint", box, cur.symbols + "."); para.hidden = true; more.addEventListener("click", () => { para.hidden = !para.hidden; }); }
    if (cur.name === "itick" && cur.available) {
      const row = h("div", "row tight", box), test = h("button", "btn", row, "Test the iTick connection (one call)"), out = h("span", "hint", row); out.id = "u-source-result";
      test.addEventListener("click", async () => {
        test.disabled = true; out.textContent = "Testing…"; out.style.color = "";
        try { const r = await api("/api/source/test", { source: S.source }); out.textContent = r.message; out.style.color = css(r.ok ? "--good-text" : "--crit"); }
        catch (e) { out.textContent = e.message; out.style.color = css("--crit"); }
        test.disabled = false;
      });
    }
  }

  /* ================================================================= how many tickers the chosen strategy needs */
  let guardSeq = 0, guardTimer = null;
  function refreshGuard() {
    clearTimeout(guardTimer);
    guardTimer = setTimeout(async () => {
      if (!S.catalog || !S.models.length) return;
      const seq = ++guardSeq, n = S.tickers.filter((x) => x.status !== "bad").length;
      try {
        const r = await api("/api/requirements", { models: S.models.map((m) => ({ name: m.name, params: m.params })), allocator: S.allocator || null, allocator_params: S.allocParams, tickers: n });
        if (seq === guardSeq) S.guard = { ok: r.ok, message: r.message || "" };
      } catch (e) { if (seq === guardSeq) S.guard = { ok: false, message: e.status === 404 ? "The app running behind this page is older than the page and cannot check the ticker count. " + RESTART : e.message }; }
      syncRun();
    }, 120);
  }
  function syncRun() {
    const run = $("#run"); if (!run) return;
    run.disabled = S.busy || !S.guard.ok;
    const g = $("#guard"); if (g) { g.textContent = S.guard.ok ? "" : S.guard.message; g.hidden = S.guard.ok; }
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
    for (const g of order) { const og = h("optgroup", "", select); og.label = g === "custom" ? "custom (formula)" : g; for (const m of groups[g]) { const o = h("option", "", og, m.name + (m.min_assets > 1 ? `  (needs ${m.min_assets}+ tickers)` : "")); o.value = m.name; if (m.name === current) o.selected = true; } }
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
      if (info.min_assets > 1) h("p", "hint", block, `Needs at least ${info.min_assets} tickers: it works by comparing them with each other.`);
      else if (info.book === "sleeves") h("p", "hint", block, "Works on each ticker alone, sized by its own signal: long when the signal is positive, short when it is negative (where the rule shorts) and in cash when there is none. The Trades tab of the result lists when it bought and sold.");
      if (info.rebalance) h("p", "hint", block, "This rule rebalances " + info.rebalance + " unless you change it in the spec.");
      if (info.requires.length) h("p", "hint", block, "Needs macro series: " + info.requires.join(", ") + " (included for the platform ETFs, and attached to any universe).");
      if (info.slow) { const w = h("p", "hint", block, "⏱ Slow: " + info.slow + "."); w.style.color = css("--warn"); }
      if (info.params.length) {
        const grid = h("div", "params", block);
        for (const p of info.params) {
          const cur = p.name in entry.params ? entry.params[p.name] : p.default;
          grid.appendChild(fieldFor(p, cur, (v) => { if (JSON.stringify(v) === JSON.stringify(p.default)) delete entry.params[p.name]; else entry.params[p.name] = v; if (p.name === "mode") { autoAllocator(); renderPortfolio(); } refreshGuard(); }, { wide: p.name === "expr" || p.kind === "json" }));
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
    refreshGuard();
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
      for (const p of info.params) { const cur = p.name in S.allocParams ? S.allocParams[p.name] : p.default; grid.appendChild(fieldFor(p, cur, (v) => { if (JSON.stringify(v) === JSON.stringify(p.default)) delete S.allocParams[p.name]; else S.allocParams[p.name] = v; refreshGuard(); }, { wide: p.kind === "json" })); }
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
    const cl = h("label", "f", box, "Starting capital in dollars (the earnings numbers are shown on this)"), ci = h("input", "", cl); ci.type = "number"; ci.min = "1000"; ci.placeholder = "blank = the AUM above, else $100,000"; ci.value = S.capital;
    ci.addEventListener("input", () => { S.capital = ci.value; });
    refreshGuard();
  }

  /* ================================================================= panel skeleton */
  function buildPanel() {
    const p = $("#panel"); p.replaceChildren();
    const sec = (id, title, open) => { const d = h("details", "", p); if (open) d.open = true; const s = h("summary", "", d, title); const b = h("div", "body", d); b.id = id; return { d, s, b }; };
    const u = sec("u-body", "1. Tickers", true);
    h("span", "pill", u.s).id = "u-count";
    h("p", "hint", u.b, "The platform's 15 ETFs, plus any ticker the data source below has: NVDA, BTC-USD, ^GSPC, EURUSD=X on Yahoo Finance. Downloads are cached for a day. One ticker is enough for a strategy that trades each ticker on its own signal.");
    h("div", "chips", u.b).id = "u-chips";
    const add = h("div", "row", u.b), inp = h("input", "", add); inp.type = "text"; inp.placeholder = "Add tickers, e.g. NVDA, BTC-USD"; inp.style.flex = "1"; inp.setAttribute("aria-label", "tickers to add");
    const go = h("button", "btn primary", add, "Add");
    const doAdd = () => { addTickers(inp.value); inp.value = ""; inp.focus(); };
    go.addEventListener("click", doAdd); inp.addEventListener("keydown", (e) => { if (e.key === "Enter") doAdd(); });
    h("div", "body", u.b).id = "u-source";
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
    const foot = h("div", "runbar", p), guard = h("p", "guard", foot); guard.id = "guard"; guard.hidden = true; guard.setAttribute("role", "status");
    const run = h("button", "btn primary", foot, "Run backtest"); run.id = "run"; run.addEventListener("click", runBacktest);
    h("div", "progress", foot).id = "progress";
    h("p", "hint", o.b, "Costs: 10 bps per unit traded, monthly rebalance, signals act the next day. Every idea you try on the same tickers counts as a trial against the deflated Sharpe ratio.");
  }

  /* ================================================================= run */
  function buildBody() {
    const tickers = S.tickers.filter((x) => x.status !== "bad");
    if (S.tickers.some((x) => x.status === "pending")) throw new Error("Wait for the ticker checks to finish");
    if (tickers.length < 1) throw new Error("Choose at least one ticker");
    if (!S.guard.ok) throw new Error(S.guard.message);
    const classes = {}; const isDefault = new Set(S.catalog.default_tickers); for (const x of tickers) if (!isDefault.has(x.t) && x.cls !== "unknown") classes[x.t] = x.cls;
    return { tickers: tickers.map((x) => x.t), classes, start: S.start || null, models: S.models.map((m) => ({ name: m.name, params: m.params })), combination: S.combination,
      allocator: S.allocator || null, allocator_params: S.allocParams, regime: S.regime || null, regime_risk: S.regimeRisk, aum: S.aum ? parseFloat(S.aum) : null,
      validate: S.validate, refresh: S.refresh, label: S.label || null, source: S.source, capital: S.capital ? parseFloat(S.capital) : null };
  }
  function setBusy(busy, message) {
    S.busy = busy; syncRun(); const p = $("#progress"); p.replaceChildren();
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
        if (v.status === "done") { addRun(v.result); if (S.tickers.some((x) => x.later)) recheckTickers(true); break; }
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
    renderResult({ fresh: true });
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
    const seg = h("div", "seg", bar), a = h("button", "", seg, "Chart"), b = h("button", "", seg, "Table");
    let mode = S.view.modes[title] === "table" ? "table" : "chart";
    const extra = h("span", "", bar);
    const mark = () => { a.setAttribute("aria-pressed", String(mode === "chart")); b.setAttribute("aria-pressed", String(mode === "table")); };
    const draw = () => { if (mode === "chart") opts.draw(body); else { body.replaceChildren(); opts.table(body); } };       // the chart kit measures its host and then clears it itself
    const pick = (m) => { mode = m; S.view.modes[title] = m; mark(); keepScroll(draw, body); };
    a.addEventListener("click", () => pick("chart"));
    b.addEventListener("click", () => pick("table"));
    mark(); card._draw = () => keepScroll(draw, body); card._extra = extra; card._body = body; draw();
    return card;
  }
  /* Run a redraw of part of the page without letting the page move. Emptying a card and measuring the chart that replaces it (a chart reads its container's width) shrinks the page for a
     moment, and the browser answers by clamping or re-anchoring the scroll position. Holding the height steady while the new content is built avoids that; the scroll position is also put back. */
  function keepScroll(fn, host) {
    const y = window.scrollY, x = window.scrollX, box = host || null, held = box ? box.offsetHeight : 0;
    if (box) box.style.minHeight = held + "px";
    try { fn(); } finally { if (box) box.style.minHeight = ""; if (window.scrollY !== y || window.scrollX !== x) window.scrollTo(x, y); }
  }
  function simpleTable(parent, cols, rows) {
    const wrap = h("div", "scroll", parent), t = h("table", "data", wrap), hd = h("tr", "", h("thead", "", t));
    for (const c of cols) { const th = h("th", c.num ? "num" : "", hd, c.label); if (c.title) th.title = c.title; }
    const tb = h("tbody", "", t);
    for (const r of rows) { const tr = h("tr", "", tb); for (const c of cols) h("td", c.num ? "num" : "", tr, c.get ? c.get(r) : String(r[c.key])); }
  }
  function sampleRows(dates, cols, every) { const out = []; for (let i = 0; i < dates.length; i += every) out.push(i); if (out[out.length - 1] !== dates.length - 1) out.push(dates.length - 1); return out; }

  /* Rebuild the whole result from the data. ``fresh`` is a new run (start from the top with the default view); otherwise this is a redraw of what the user was looking at (theme, window
     width, chart window) and the page must stay where it is, with the same tables and tabs open. */
  function renderResult(opts) {
    const run = S.current; if (!run) return;
    const root = $("#results");
    if (opts && opts.fresh) { S.view = freshView(); root.replaceChildren(); buildResult(root, run); if (!$("#view-backtest").hidden) window.scrollTo(0, 0); return; }
    keepScroll(() => { root.replaceChildren(); buildResult(root, run); }, root);
  }
  function buildResult(root, run) {
    const r = run.result, w = windowOf(run, S.range), st = statsOf(w.equity), single = r.universe.tickers.length === 1, refName = single ? "buy and hold" : "equal weight", eqw = w.bench.equal_weight ? statsOf(w.bench.equal_weight) : null;
    const head = h("div", "head", root), left = h("div", "", head);
    h("h1", "", left, r.name);
    h("p", "sub", left, `${r.universe.tickers.length} ticker${single ? "" : "s"} · ${r.universe.source} · ${r.metrics.start} to ${r.metrics.end} (${(r.stats.years).toFixed(1)} years) · ran in ${r.seconds} s · run #${run.id}`);
    const acts = h("div", "row tight", head);
    const ex = h("button", "btn", acts, "Copy spec (YAML)"); ex.addEventListener("click", () => copy(r.yaml, "Spec"));
    const dn = h("button", "btn", acts, "Download spec"); dn.addEventListener("click", () => download(r.name.replace(/[^\w-]+/g, "_") + ".yaml", r.yaml));
    const cmp = h("button", "btn", acts, "Compare runs"); cmp.addEventListener("click", () => showTab("compare"));
    if (r.warning) { const b = h("div", "banner", root); h("strong", "", b, "Heads up."); b.appendChild(document.createTextNode(r.warning)); }
    const notes = Object.entries(r.universe.data_notes || {});
    if (notes.length) { const b = h("div", "banner", root); h("strong", "", b, "Data notes."); b.appendChild(document.createTextNode(notes.map(([t, list]) => `${t}: ${list.join(" ")}`).join(" · "))); }
    const nBad = r.flags.filter((f) => f.flag).length;
    const range = h("div", "row", root); h("span", "hint", range, "Chart window");
    const seg = h("div", "seg", range);
    for (const [k, lab] of [["all", "All"], ["10y", "10y"], ["5y", "5y"], ["3y", "3y"], ["1y", "1y"]]) { const b = h("button", "", seg, lab); b.setAttribute("aria-pressed", String(S.range === k)); b.addEventListener("click", () => { S.range = k; renderResult(); }); }
    h("span", "hint", range, S.range === "all" ? "Return, volatility and drawdown tiles follow the window; costs, turnover and the deflated Sharpe are full-sample." : `Window starts ${w.first}. Tables below always show the full backtest.`);
    const k = h("div", "kpis", root), dl = eqw ? st.sharpe - eqw.sharpe : null;
    tile(k, "Net Sharpe ratio" + (S.range === "all" ? "" : " (window)"), num(st.sharpe), dl === null ? null : { text: `${fmt.signed(dl)} vs ${refName} (${num(eqw.sharpe)})`, dir: dl >= 0 ? "up" : "down" }, "hero", "Annualised excess return per unit of volatility, after costs");
    tile(k, "CAGR", pct(st.cagr), eqw ? { text: `${refName} ${pct(eqw.cagr)}` } : null);
    tile(k, "Volatility", pct(st.vol), eqw ? { text: `${refName} ${pct(eqw.vol)}` } : null);
    tile(k, "Max drawdown", pct(st.mdd), eqw ? { text: `${refName} ${pct(eqw.mdd)}` } : null);
    tile(k, "Turnover per year", num(r.metrics.ann_turnover, 1) + "×", { text: `cost drag ${num(r.metrics.ann_cost_bps, 0)} bps/yr` });
    const dsr = r.validation.deflated_sharpe_probability;
    tile(k, "Deflated Sharpe", dsr === null ? "n/a" : num(dsr), { text: `${r.validation.n_trials} trial${r.validation.n_trials === 1 ? "" : "s"}; ≥ 0.95 is the bar`, dir: dsr !== null && dsr >= 0.95 ? "up" : "" }, "", "Probability the Sharpe ratio is above what the best of your tries would show by luck alone");
    if (r.earnings) earningsTiles(root, r);
    const ch = h("div", "charts", root);
    const cap = r.capital || 100000, dollars = (arr) => arr.map((v) => (v === null ? null : v * cap));
    const series = [{ name: "Strategy", color: css("--s1"), values: dollars(w.equity) }];
    const bn = { equal_weight: [single ? "Buy and hold" : "Equal weight", "--s2"], risk_parity: ["Risk parity", "--s3"] };
    for (const key in w.bench) series.push({ name: bn[key] ? bn[key][0] : key, color: css(bn[key] ? bn[key][1] : "--s4"), values: dollars(w.bench[key]) });
    const growth = chartCard(ch, `Account value from ${money(cap)}`, `Net of costs, starting from ${money(cap)} at the start of the chart window. ${single ? "Buy and hold is the benchmark." : "Benchmarks need five or more assets."}`, { wide: true,
      draw: (b) => C.line(b, { dates: w.dates, series, log: S.view.log, yFormat: axisMoney, tipFormat: (v) => money(v), height: 300, label: "Account value in dollars" }),
      table: (b) => { const idx = sampleRows(w.dates, series, 21); simpleTable(b, [{ label: "Date", get: (i) => w.dates[i] }, ...series.map((s) => ({ label: s.name, num: true, get: (i) => (s.values[i] === null ? "" : money(s.values[i])) }))], idx); } });
    const lg = h("div", "seg", growth._extra), l1 = h("button", "", lg, "Linear"), l2 = h("button", "", lg, "Log");
    const markLog = () => { l1.setAttribute("aria-pressed", String(!S.view.log)); l2.setAttribute("aria-pressed", String(S.view.log)); };
    markLog();
    l1.addEventListener("click", () => { S.view.log = false; markLog(); growth._draw(); });
    l2.addEventListener("click", () => { S.view.log = true; markLog(); growth._draw(); });
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
  /* ================================================================= earnings and trades */
  function earningsTiles(root, r) {
    const e = r.earnings, single = r.universe.tickers.length === 1, ref = e.benchmarks.equal_weight || null, refName = single ? "buy and hold" : "equal weight";
    h("p", "hint", root, `Earnings on ${money(e.capital)} of starting capital, over the full backtest. The Earnings and Trades tabs below have the detail.`);
    const k = h("div", "earnrow", root);
    tile(k, "Net profit", smoney(e.net_profit), { text: `${fmt.spct(e.total_return)} on ${money(e.capital)}` + (ref ? ` · ${refName} ${smoney(ref.pnl)}` : ""), dir: e.net_profit >= 0 ? "up" : "down" }, "", "What the strategy would have earned after trading costs");
    tile(k, "Ending value", money(e.end_value), { text: `from ${money(e.capital)}` });
    tile(k, "Average per month", smoney(e.profit_per_month), { text: `${smoney(e.profit_per_year)} a year` });
    tile(k, "Profitable months", `${e.winning_months} of ${e.months}`, { text: `${pct(e.months ? e.winning_months / e.months : null, 0)} of months made money` });
    tile(k, "Profit factor", e.profit_factor_monthly === null ? "n/a" : num(e.profit_factor_monthly), { text: e.profit_factor_monthly === null ? "no losing month" : "months won ÷ months lost" }, "", "Total made in winning months divided by total lost in losing months; above 1 means a net gain");
    tile(k, "Deepest fall from a peak", money(e.drawdown.max_dollars), { text: `${pct(e.drawdown.max_pct)}` + (e.drawdown.longest_calendar_days ? ` · up to ${e.drawdown.longest_calendar_days} days below a peak` : "") });
  }
  function kvTable(parent, title, rows) {
    const box = h("div", "kv", parent); h("h4", "", box, title);
    simpleTable(box, [{ label: "Measure", get: (x) => x[0] }, { label: "Value", num: true, get: (x) => x[1] }], rows.filter((x) => x[1] !== null && x[1] !== undefined));
  }
  function earningsTab(b, r) {
    const e = r.earnings, single = r.universe.tickers.length === 1, d = e.drawdown, sp = (x) => (x ? `${x.period}: ${smoney(x.pnl)} (${fmt.spct(x.ret)})` : "n/a");
    if (e.exposure && e.exposure.peak > 1.05) {
      const n = h("div", "banner", b); h("strong", "", n, "Leverage.");
      n.appendChild(document.createTextNode(`The strategy held up to ${num(e.exposure.peak, 1)}× your capital (${num(e.exposure.average, 1)}× on average; above 1× on ${pct(e.exposure.share_of_days_above_one, 0)} of days), so gains and losses are scaled the same way. Only trading costs are charged: no borrowing or short-selling fees.`));
    }
    h("p", "hint", b, `Dollars on ${money(e.capital)} of starting capital, compounded day by day after trading costs. They show what the historical rules would have earned, not what to expect.`);
    const grid = h("div", "kvgrid", b);
    kvTable(grid, "Profit", [["Starting capital", money(e.capital)], ["Ending value", money(e.end_value)], ["Net profit", `${smoney(e.net_profit)} (${fmt.spct(e.total_return)})`],
      ["Average per year", smoney(e.profit_per_year)], ["Average per month", smoney(e.profit_per_month)], ["Average per trading day", smoney(e.profit_per_day)],
      ...Object.entries(e.benchmarks).map(([k, v]) => [(k === "equal_weight" ? (single ? "Buy and hold" : "Equal weight") : k.replace(/_/g, " ")) + " would have made", `${smoney(v.pnl)} (${fmt.spct(v.ret)})`])]);
    kvTable(grid, "Costs and leverage", [["Profit before trading costs", smoney(e.gross_profit)], ["Trading costs paid", money(e.costs_paid)],
      ["Costs as a share of that profit", e.costs_share_of_gross === null ? null : pct(e.costs_share_of_gross, 0)],
      ["Average exposure", e.exposure ? num(e.exposure.average, 2) + "× capital" : null], ["Peak exposure", e.exposure ? num(e.exposure.peak, 2) + "× capital" : null]]);
    kvTable(grid, "Best and worst", [["Best day", sp(e.best_day)], ["Worst day", sp(e.worst_day)], ["Best month", sp(e.best_month)], ["Worst month", sp(e.worst_month)], ["Best year", sp(e.best_year)], ["Worst year", sp(e.worst_year)]]);
    kvTable(grid, "How often it made money", [["Profitable months", `${e.winning_months} of ${e.months} (${pct(e.months ? e.winning_months / e.months : null, 0)})`],
      ["Profitable years", `${e.winning_years} of ${e.annual.length}`], ["Winning days", `${e.winning_days} of ${e.winning_days + e.losing_days + e.flat_days}` + (e.flat_days ? ` (${e.flat_days} flat)` : "")],
      ["Average winning month", smoney(e.average_winning_month)], ["Average losing month", smoney(e.average_losing_month)],
      ["Profit factor, months", e.profit_factor_monthly === null ? "n/a (no losing month)" : num(e.profit_factor_monthly)], ["Profit factor, days", e.profit_factor_daily === null ? "n/a" : num(e.profit_factor_daily)],
      ["Longest winning / losing run, months", `${e.longest_winning_streak_months} / ${e.longest_losing_streak_months}`], ["Longest winning / losing run, days", `${e.longest_winning_streak_days} / ${e.longest_losing_streak_days}`]]);
    kvTable(grid, "Deepest fall from a peak", [["Fall", `${money(d.max_dollars)} (${pct(d.max_pct)})`], ["Peak to trough", d.peak_date ? `${d.peak_date} → ${d.trough_date}` : null],
      ["Peak regained", d.peak_date ? (d.recovered_on || "not yet") : null], ["Longest time below a peak", `${d.longest_calendar_days} days`]]);
    h("h4", "subhead", b, "Profit by calendar year");
    const chartHost = h("div", "", b);
    C.columns(chartHost, { labels: e.annual.map((a) => a.year), values: e.annual.map((a) => a.pnl), yFormat: axisMoney, tipFormat: smoney, faded: e.annual.map((a) => a.partial), height: 220, name: "Profit",
      tipLabel: (i) => e.annual[i].year + (e.annual[i].partial ? ` (${e.annual[i].days} trading days)` : ""), label: "Profit by calendar year" });
    simpleTable(b, [{ label: "Year", get: (a) => a.year + (a.partial ? "*" : "") }, { label: "Start", num: true, get: (a) => money(a.start_value) }, { label: "End", num: true, get: (a) => money(a.end_value) },
      { label: "Profit", num: true, get: (a) => smoney(a.pnl) }, { label: "Return", num: true, get: (a) => fmt.spct(a.ret) }, { label: "Worst fall", num: true, get: (a) => pct(a.max_drawdown) }], e.annual);
    h("p", "hint", b, "* part of a year.");
    h("h4", "subhead", b, "Profit by month");
    simpleTable(b, [{ label: "Month", get: (m) => `${m.year}-${String(m.month).padStart(2, "0")}` }, { label: "Profit", num: true, get: (m) => smoney(m.pnl) }, { label: "Return", num: true, get: (m) => fmt.spct(m.ret) },
      { label: "Account value", num: true, get: (m) => money(m.end_value) }], e.monthly.slice().reverse());
  }
  function tradesTab(b, r) {
    const t = r.trades, s = t.stats, tp = (x) => (x ? `${x.ticker} ${x.side}, ${x.entered} → ${x.exited}: ${smoney(x.pnl)}` : null);
    h("p", "hint", b, "A round trip is one stretch in which a ticker is held on the same side: bought (long) or sold first (short). It opens when the position appears and closes when the position goes to zero or flips. Profit is what the position earned while it was held, before trading costs, in dollars on the account value of each day; the move is the price change between the two dates (reversed for a short). Prices are the adjusted closes the backtest used.");
    if (!t.total) h("p", "hint", b, "This strategy never opened a position, so there is nothing to list.");
    const grid = h("div", "kvgrid", b);
    kvTable(grid, "Round trips", [["Closed", String(s.round_trips)], ["Still open", String(s.open_positions)], ["Winners / losers", `${s.winners} / ${s.losers}`], ["Win rate", s.win_rate === null ? null : pct(s.win_rate, 0)],
      ["Average winner", smoney(s.average_win)], ["Average loser", smoney(s.average_loss)], ["Payoff ratio (winner ÷ loser)", s.payoff_ratio === null ? null : num(s.payoff_ratio)],
      ["Profit factor", s.profit_factor === null ? (s.round_trips ? "n/a (no losers)" : null) : num(s.profit_factor)], ["Expected profit per trip", smoney(s.expectancy)], ["Best", tp(s.best)], ["Worst", tp(s.worst)]]);
    kvTable(grid, "Holding", [["Average days held", s.average_days_held === null ? null : num(s.average_days_held, 0)], ["Longest / shortest", s.longest_days_held === null ? null : `${s.longest_days_held} / ${s.shortest_days_held} days`],
      ["Days with a position", pct(s.days_in_market, 0)], ["Average exposure", num(s.average_exposure, 2) + "× capital"]]);
    kvTable(grid, "Buys and sells", [["Changes of position (≥ 0.25% of the account)", String(s.orders)], ["Buys / sells", `${s.buys} / ${s.sells}`],
      ["Long trips", `${s.long_trips} · ${smoney(s.long_pnl)}`], ["Short trips", `${s.short_trips} · ${smoney(s.short_pnl)}`]]);
    if (t.round_trips.length) {
      h("h4", "subhead", b, `Round trips (${t.listed} of ${t.total}: open positions first, then the latest closed)`);
      simpleTable(b, [{ label: "Ticker", get: (x) => x.ticker }, { label: "Side", get: (x) => x.side }, { label: "Opened", get: (x) => x.entered }, { label: "Closed", get: (x) => (x.open ? "open" : x.exited) },
        { label: "Days", num: true, get: (x) => x.days }, { label: "Entry price", num: true, get: (x) => (x.entry_price === null ? "" : num(x.entry_price)) }, { label: "Exit price", num: true, get: (x) => (x.exit_price === null ? "" : num(x.exit_price)) },
        { label: "Move", num: true, get: (x) => (x.move === null ? "" : fmt.spct(x.move)) }, { label: "Profit", num: true, get: (x) => smoney(x.pnl) }], t.round_trips);
    }
    if (t.orders.length) {
      h("h4", "subhead", b, "Latest buys and sells");
      simpleTable(b, [{ label: "Date", get: (x) => x.date }, { label: "Ticker", get: (x) => x.ticker }, { label: "", get: (x) => x.side }, { label: "Size", num: true, get: (x) => pct(x.size, 1) },
        { label: "Amount", num: true, get: (x) => money(x.amount) }, { label: "Price", num: true, get: (x) => (x.price === null ? "" : num(x.price)) }], t.orders);
    }
  }
  function renderDetails(root, r, nBad) {
    const flags = h("div", "card", root); h("h2", "", flags, "Red flags: fixed rules, each with its threshold"); h("p", "sub", flags, `${nBad} of ${r.flags.length} rules flagged. A flag is a question, not a verdict.`);
    for (const f of r.flags) { const row = h("div", "flag " + (f.flag ? "bad" : "ok"), flags); h("span", "ic", row, f.flag ? "⚑" : "✓"); const d = h("div", "", row); d.appendChild(document.createTextNode(f.rule + " "));
      h("b", "", d, f.flag ? "FLAG" : "ok"); h("small", "", d, `Observed ${f.observed}. ${f.why}`); }
    const T = r.tables, tabs = [], single = r.universe.tickers.length === 1;
    const add = (id, label, draw) => tabs.push({ id, label, draw });
    const tab = (key, cols) => (b) => {
      const rows = key === "benchmarks" && single ? T[key].map((x) => Object.assign({}, x, { benchmark: x.benchmark === "equal_weight" ? "buy and hold" : x.benchmark })) : T[key];
      simpleTable(b, cols || Object.keys(rows[0]).map((c) => ({ label: c.replace(/_/g, " "), num: typeof rows[0][c] === "number", get: (x) => cell(x[c]) })), rows);
    };
    if (r.earnings) add("earn", "Earnings", (b) => earningsTab(b, r));
    if (r.trades) add("trades", "Trades", (b) => tradesTab(b, r));
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
    const card = h("div", "card", root); h("h2", "", card, "Details"); const bar = h("div", "tabsmall", card), body = h("div", "tabbody", card);
    const open = (t) => { S.view.detail = t.id; [...bar.children].forEach((c) => c.setAttribute("aria-selected", String(c.dataset.id === t.id))); keepScroll(() => { body.replaceChildren(); t.draw(body); }, body); };
    tabs.forEach((t) => { const b = h("button", "", bar, t.label); b.dataset.id = t.id; b.setAttribute("aria-selected", "false"); b.addEventListener("click", () => open(t)); });
    if (tabs.length) open(tabs.find((t) => t.id === S.view.detail) || tabs[0]);
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
      ["50/200 moving-average crossover on SPY alone", () => { S.tickers = [{ t: "SPY", cls: S.catalog.default_classes.SPY || "equity", status: "default" }]; S.start = ""; S.models = [{ name: "ma_crossover", params: {} }]; S.allocatorTouched = false; autoAllocator(); }],
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
        const body = { expr: B.expr, mode: B.mode, tickers: S.tickers.filter((x) => x.status !== "bad").map((x) => x.t), start: S.start || null, source: S.source };
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

  /* ================================================================= execution and cash-flow labs */
  /* Two tabs that run the research simulators behind /api/exec and /api/cash. The Execution tab works on a stylised market made from the numbers in its form (nothing is downloaded); the Cash
     flows tab follows a portfolio of real tickers from the data source. Each tab keeps its form while another tab is open, and a result is redrawn from the data it came from. */
  const LAB_API = 3;
  const colorOf = (i) => css(SLOTS[i % SLOTS.length]);
  const labelPx = (labels) => Math.min(260, Math.max(74, Math.round(Math.max(0, ...labels.map((x) => String(x).length)) * 6.4) + 14));
  const running = (a) => { let total = 0; return a.map((v) => (total += v)); };
  const bps = (v, d = 1) => (bad(v) ? "n/a" : num(v, d) + " bps");
  const grouped = (v) => (bad(v) ? "n/a" : Math.round(v).toLocaleString("en-US"));
  const plain = (v) => String(+v.toFixed(2));                                                // an axis tick without trailing zeros: 12.5, 15, 17.5
  const openGuide = (slug) => { G.slug = slug; showTab("guides"); };

  /* One form field, a number, a choice, a line of text or a tick box, bound to ``state[spec.key]``. Numbers are kept as typed, so a half-written value is not lost when the tab is redrawn. */
  function labField(parent, spec, state, after) {
    const wrap = h("label", "f" + (spec.kind === "bool" ? " inline" : ""), parent);
    let input;
    if (spec.kind === "bool") { input = h("input", "", wrap); input.type = "checkbox"; input.checked = !!state[spec.key]; wrap.appendChild(document.createTextNode(spec.label)); }
    else {
      wrap.appendChild(document.createTextNode(spec.label));
      if (spec.kind === "choice") { input = h("select", "", wrap); for (const [value, text] of spec.options) { const o = h("option", "", input, text); o.value = value; if (value === state[spec.key]) o.selected = true; } }
      else {
        input = h("input", "", wrap); input.type = spec.kind === "number" ? "number" : "text"; input.value = state[spec.key]; if (spec.placeholder) input.placeholder = spec.placeholder;
        if (spec.kind === "number") { input.step = spec.step || "any"; if (spec.min !== undefined) input.min = spec.min; if (spec.max !== undefined) input.max = spec.max; }
        else input.spellcheck = false;
      }
    }
    if (spec.help) wrap.title = spec.help;
    input.setAttribute("data-key", spec.key);
    input.addEventListener("input", () => { state[spec.key] = spec.kind === "bool" ? input.checked : input.value; if (after) after(); });
    input.addEventListener("change", () => input.dispatchEvent(new Event("input")));
    return wrap;
  }
  function checkList(parent, title, items, chosen, after) {
    const box = h("div", "checks", parent); h("span", "hint", box, title);
    for (const [value, text, help] of items) {
      const label = h("label", "f inline", box), c = h("input", "", label); c.type = "checkbox"; c.checked = chosen.has(value); c.setAttribute("data-value", value); label.appendChild(document.createTextNode(text)); if (help) label.title = help;
      c.addEventListener("change", () => { if (c.checked) chosen.add(value); else chosen.delete(value); after(); });
    }
  }
  /* The numbers of a form as a request: blank fields are left out (the server has a default for each), ``scale`` turns a percentage into a fraction. */
  function numbers(state, keys, scale) {
    const out = {};
    for (const key of keys) {
      const raw = String(state[key] === undefined ? "" : state[key]).trim(); if (raw === "") continue;
      const v = Number(raw); if (!isFinite(v)) throw new Error(`${key.replace(/_/g, " ")} must be a number`);
      out[key] = v * ((scale && scale[key]) || 1);
    }
    return out;
  }
  function labOld(root) {
    const c = h("div", "card empty", root); c.style.gridColumn = "1 / -1"; h("h2", "", c, "This tab needs a newer app");
    h("p", "", c, "The page was updated, but the app behind it is still running the old code. " + RESTART + ` (The app reports version ${serverApi()}; this tab needs ${LAB_API}.)`);
    const again = h("button", "btn primary", c, "Reload this page"); again.addEventListener("click", () => location.reload());
  }
  function labHead(root, title, sub, guide) {
    const head = h("div", "head", root), left = h("div", "", head); h("h1", "", left, title); if (sub) h("p", "sub", left, sub);
    if (guide) { const acts = h("div", "row tight", head), b = h("button", "btn", acts, "Read the guide"); b.addEventListener("click", () => openGuide(guide)); }
  }
  function labBanner(root, title, text, serious) { const b = h("div", "banner" + (serious ? " err" : ""), root); h("strong", "", b, title); b.appendChild(document.createTextNode(text)); return b; }
  function labEmpty(root, lab, title, text, button) {
    const c = h("div", "card empty", root); h("h2", "", c, title); for (const t of [].concat(text)) h("p", "", c, t);
    const b = h("button", "btn primary", c, button || "Run it with these settings"); b.addEventListener("click", () => lab.run());
  }
  function syncLab(lab) {
    if (!lab.go || !lab.go.isConnected) return;
    const message = lab.check ? lab.check() : "";
    lab.go.disabled = !!lab.busy || !!message; lab.guard.textContent = message; lab.guard.hidden = !message;
    lab.progress.replaceChildren(); if (lab.busy) { h("span", "spinner", lab.progress); h("span", "", lab.progress, lab.busy); }
  }
  async function labRun(lab, path, bodyFor, message) {
    if (lab.busy) return;
    let body; try { body = bodyFor(); } catch (e) { toast(e.message); return; }
    const mode = lab.mode; lab.busy = message; syncLab(lab);
    try { lab.results[mode] = { data: await api(path, body), body }; lab.errors[mode] = null; }
    catch (e) { lab.results[mode] = null; lab.errors[mode] = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + RESTART : e.message; }
    lab.busy = ""; syncLab(lab);
    if (lab.mode === mode && !$(lab.view).hidden) lab.draw();
  }
  function renderLab(lab) {
    const root = $(lab.root); root.replaceChildren();
    if (serverApi() < LAB_API) { labOld(root); return; }
    const panel = h("aside", "card panel", root); panel.setAttribute("aria-label", lab.title + " setup");
    const top = h("div", "labtop", panel); h("h2", "", top, lab.title); h("p", "hint", top, lab.about);
    const pills = h("div", "tabsmall", top);
    for (const [key, text] of lab.modes) { const b = h("button", "", pills, text); b.dataset.mode = key; b.setAttribute("aria-selected", String(key === lab.mode)); b.addEventListener("click", () => { lab.mode = key; renderLab(lab); }); }
    h("p", "hint", top, lab.blurb[lab.mode]);
    lab.sections(panel);
    const foot = h("div", "runbar", panel); lab.guard = h("p", "guard", foot); lab.guard.setAttribute("role", "status"); lab.guard.hidden = true;
    lab.go = h("button", "btn primary", foot, lab.runLabel[lab.mode]); lab.go.addEventListener("click", () => lab.run()); lab.progress = h("div", "progress", foot);
    const results = h("div", "stack", root); results.id = lab.id + "-results"; results.setAttribute("aria-live", "polite");
    syncLab(lab); lab.draw();
  }
  /* Rebuild the results from the data they came from. The page must stay where it is, so this is the same redraw the Backtest tab uses for a theme or width change. */
  function drawLab(lab) {
    const root = $("#" + lab.id + "-results"); if (!root || serverApi() < LAB_API) return;
    keepScroll(() => {
      root.replaceChildren();
      const error = lab.errors[lab.mode], got = lab.results[lab.mode];
      if (error) labBanner(root, "That did not run. ", error, true);
      if (got) lab.build(root, got.data, got.body); else if (!error) lab.empty(root);
    }, root);
  }

  /* ---------- the Execution tab ---------- */
  const ALGO_PRESETS = [["twap", "TWAP: equal slices"], ["vwap", "VWAP: follow the day's volume"], ["pov", "POV: take a fixed share of the volume"], ["arrival_price", "Arrival price: trade early"],
    ["is:risk_aversion=0.003", "Implementation shortfall (risk aversion 0.003)"], ["aim+vwap", "VWAP, aggressive in the money (AIM)"], ["pim+vwap", "VWAP, passive in the money (PIM)"],
    ["target_cost+is:risk_aversion=0.003", "Target cost, on implementation shortfall"], ["liquidity_seeking", "Liquidity seeking: slow down when the market is thin"]];
  const HFT_TEXT = { pairs: "Pair trading: a z-score rule on the spread of two prices, orders arriving after a delay, a cost on both legs.", etf: "ETF against its basket: how the profit of trading the premium falls with the delay of the order.",
    rebate: "Rebate and liquidity trading: a maker living on rebates and spread, with and without a view of the order flow that is about to hit its quotes.", amm: "Auto market making: quotes shaded by inventory (Avellaneda-Stoikov) against symmetric quotes." };
  const HFT_COLUMNS = { days: "Days", mean_daily_bps: "Profit per day (bps)", std_daily_bps: "Std dev per day (bps)", sharpe: "Sharpe", round_trips: "Round trips", win_rate: "Win rate", average_hold_bars: "Hold (bars)",
    average_trip_bps: "Per trip (bps)", gross_bps: "Gross (bps)", cost_bps_total: "Costs (bps)", mean_wealth: "Mean profit", se_wealth: "Std error", rebate: "Rebates", spread: "Spread earned", price_pnl: "Price moves",
    adverse_selection: "Adverse selection", inventory_pnl: "Inventory", fills: "Fills", mean_abs_inventory: "Mean |inventory|", std_wealth: "Std dev of profit", mean_trades: "Trades", std_inventory: "Std dev of inventory",
    mean_squared_inventory: "Mean squared inventory" };
  const X = {
    id: "exec", root: "#exec", view: "#view-exec", title: "Execution lab", mode: "run", busy: "", results: {}, errors: {}, cumulative: false, guard: null, go: null, progress: null,
    about: "Trade an order through a simulated day and compare algorithms. Nothing is downloaded: the market is made from the numbers below.",
    modes: [["run", "Compare algorithms"], ["frontier", "Cost-risk frontier"], ["basket", "Basket"], ["hft", "High-frequency"]],
    runLabel: { run: "Run the algorithms", frontier: "Trace the frontier", basket: "Work the basket", hft: "Run the simulation" },
    blurb: { run: "Run several algorithms on the same simulated days and see what they cost, and how much that cost varies.",
      frontier: "For every price of risk, the cheapest way to trade the order: the curve between paying market impact and carrying price risk.",
      basket: "Trade a list of stocks together: keep the unexecuted list as hedged as possible, see what can go to dark pools and what a partial execution should be.",
      hft: "Research simulators of strategies that live on spreads, rebates and short-lived mispricings, to size how much edge a delay or a cost leaves." },
    form: { side: "buy", shares: "200000", adv: "2000000", price: "50", sigma: "2", spread_bps: "4", intervals: "26", style: "aggressive", scenario: "normal", max_participation: "35", target_bps: "", paths: "300", seed: "0",
      algos: new Set(["twap", "vwap", "pov", "is:risk_aversion=0.003", "arrival_price"]), custom: "", size: "8", risk_aversion: "0.001", block_threshold: "0.5", share: "50", sim: "etf" },
  };
  const EXEC_ORDER = [{ key: "side", label: "Side", kind: "choice", options: [["buy", "Buy"], ["sell", "Sell"]] }, { key: "shares", label: "Shares to trade", kind: "number", min: 1 }];
  const EXEC_MARKET = [{ key: "adv", label: "Average daily volume (shares)", kind: "number", min: 1000 }, { key: "price", label: "Price per share ($)", kind: "number", min: 0.01 },
    { key: "sigma", label: "Daily volatility (%)", kind: "number", min: 0.1, max: 20 }, { key: "spread_bps", label: "Bid-ask spread (bps)", kind: "number", min: 0, max: 200 },
    { key: "intervals", label: "Slices in the day", kind: "number", step: 1, min: 4, max: 100 }];
  const EXEC_WORK = [
    { key: "style", label: "Trading style", kind: "choice", options: [["aggressive", "Aggressive: take liquidity"], ["working", "Working order: a mix"], ["passive", "Passive: limit and dark orders"]] },
    { key: "scenario", label: "Market scenario", kind: "choice", options: [["normal", "Normal"], ["trend_up", "Trending up"], ["trend_down", "Trending down"], ["mean_reverting", "Mean-reverting"], ["crisis", "Crisis"]] },
    { key: "max_participation", label: "Most of the volume to take (%)", kind: "number", min: 1, max: 100 },
    { key: "target_bps", label: "Target cost in bps (blank = none)", kind: "number", help: "The budget the target-cost tactic tries to keep; the other algorithms ignore it." },
    { key: "paths", label: "Simulated days per algorithm", kind: "number", step: 1, min: 20, max: 2000 }, { key: "seed", label: "Random seed", kind: "number", step: 1, min: 0 }];
  const EXEC_BASKET = [{ key: "size", label: "Names in the list (2 to 16)", kind: "number", step: 1, min: 2, max: 16 }, { key: "seed", label: "Random seed", kind: "number", step: 1, min: 0 },
    { key: "risk_aversion", label: "Risk aversion", kind: "number", min: 0.000001, help: "Zero would trade every name on its own volume; higher hedges first." },
    { key: "block_threshold", label: "A block is at least this share of a day's volume (%)", kind: "number", min: 0.01, max: 100 }, { key: "share", label: "Share of the list's value to execute (%)", kind: "number", min: 0, max: 100 }];
  const EXEC_HFT = [{ key: "sim", label: "Simulation", kind: "choice", options: [["etf", "ETF against its basket"], ["pairs", "Pair trading"], ["rebate", "Rebate and liquidity trading"], ["amm", "Auto market making"]] },
    { key: "seed", label: "Random seed", kind: "number", step: 1, min: 0 }];
  const PERCENT = { sigma: 0.01, max_participation: 0.01, share: 0.01, block_threshold: 0.01 };
  const execAlgos = () => [...ALGO_PRESETS.map((a) => a[0]).filter((a) => X.form.algos.has(a)), ...X.form.custom.split(/\s+/).filter(Boolean)];
  X.check = () => (X.mode === "run" && (execAlgos().length < 1 || execAlgos().length > 8) ? "Choose between one and eight algorithms." : "");
  X.sections = (panel) => {
    const f = X.form, after = () => syncLab(X), sec = (title) => { const d = h("details", "", panel); d.open = true; h("summary", "", d, title); return h("div", "body", d); };
    const fields = (box, specs, then) => { for (const s of specs) labField(box, s, f, then || after); };
    if (X.mode === "run" || X.mode === "frontier") { fields(sec("1. The order"), EXEC_ORDER); fields(sec("2. The market"), EXEC_MARKET); }
    if (X.mode === "run") {
      fields(sec("3. How it is worked"), EXEC_WORK);
      const a = sec("4. Algorithms"); checkList(a, "Tick up to eight:", ALGO_PRESETS, f.algos, after);
      labField(a, { key: "custom", label: "Others, separated by spaces", kind: "text", placeholder: "pov:rate=0.15 exp_trade:kappa=3", help: "name:key=value,key=value, with an optional tactic first: aim, pim or target_cost, then +" }, f, after);
      h("p", "hint", a, "Any name from the guide works: twap, vwap, pov, arrival_price, is, min_cost, min_cost_risk, min_risk_cost, balanced, price_improvement, exp_trade, exp_residual, trade_rate, liquidity_seeking.");
    }
    if (X.mode === "basket") fields(sec("The basket"), EXEC_BASKET);
    if (X.mode === "hft") { const b = sec("The simulation"), note = h("p", "hint", b, ""); fields(b, EXEC_HFT, () => { note.textContent = HFT_TEXT[f.sim]; after(); }); note.textContent = HFT_TEXT[f.sim]; }
  };
  X.run = () => {
    const f = X.form, m = X.mode, market = ["shares", "adv", "price", "sigma", "spread_bps", "intervals"];
    if (m === "run") return labRun(X, "/api/exec", () => ({ action: "run", side: f.side, style: f.style, scenario: f.scenario, algos: execAlgos(), ...numbers(f, [...market, "max_participation", "target_bps", "paths", "seed"], PERCENT) }), "Simulating…");
    if (m === "frontier") return labRun(X, "/api/exec", () => ({ action: "frontier", side: f.side, ...numbers(f, market, PERCENT) }), "Tracing the frontier…");
    if (m === "basket") return labRun(X, "/api/exec", () => ({ action: "basket", ...numbers(f, ["size", "seed", "risk_aversion", "block_threshold", "share"], PERCENT) }), "Working the basket…");
    return labRun(X, "/api/exec", () => ({ action: "hft", sim: f.sim, ...numbers(f, ["seed"]) }), "Simulating (the rebate simulation takes a few seconds)…");
  };
  X.empty = (root) => {
    const text = { run: ["Compare execution algorithms on a simulated day", "Each algorithm trades the same order on the same random days. You see the average cost against the arrival price, how much it varies from day to day, and how each one spreads the order through the day."],
      frontier: ["Trace the cost-risk frontier", "Trading faster costs market impact and trading slower leaves the order exposed to the price. The frontier is the best trade-off for every price of risk."],
      basket: ["Work a basket of stocks", "A demonstration list of buys and sells traded together: the joint schedule against stock by stock, the minimum-risk partial execution, what can go dark, and block against program names."],
      hft: ["Run a high-frequency research simulation", "Pair trading, ETF arbitrage, rebate trading and market making on stylised markets, to see how a delay or a cost eats an edge."] }[X.mode];
    labEmpty(root, X, text[0], [text[1], "The defaults run in a second or two."], "Run it with the defaults");
  };
  X.draw = () => drawLab(X);
  X.build = (root, r, body) => ({ run: execCompare, frontier: execFrontier, basket: execBasket, hft: execHft })[X.mode](root, r, body);
  const sideWord = (side) => (side === "buy" ? "Buying" : "Selling");

  function execCompare(root, r) {
    const rows = r.rows, o = r.order, paths = rows.length ? rows[0].paths : 0, colors = rows.map((_, i) => colorOf(i));
    labHead(root, `${sideWord(o.side)} ${grouped(o.shares)} shares`, `${money(o.value)} · ${pct(o.participation, 1)} of a day's volume · ${r.scenario.name.replace(/_/g, " ")} market · ${r.style.name} · ${grouped(paths)} simulated days per algorithm · ${r.intervals} slices`, "technique-execution-algorithms");
    h("p", "hint", root, `Market: ${r.scenario.description}. Style: ${r.style.description}.`);
    const cheapest = rows.reduce((a, b) => (b.shortfall_bps < a.shortfall_bps ? b : a)), calmest = rows.reduce((a, b) => (b.std_bps < a.std_bps ? b : a));
    const k = h("div", "earnrow", root);
    tile(k, "Lowest average shortfall", bps(cheapest.shortfall_bps), { text: `${cheapest.algo} · varies by ${bps(cheapest.std_bps, 0)} between days` }, "", "Average cost of the whole order against the price when it arrived; positive is a cost");
    tile(k, "Least variation between days", bps(calmest.std_bps, 1), { text: `${calmest.algo} · average shortfall ${bps(calmest.shortfall_bps)}` }, "", "Standard deviation of the shortfall across the simulated days");
    tile(k, "Order size", pct(o.participation, 1) + " of a day", { text: `${grouped(o.shares)} shares, ${money(o.value)}` });
    const card = h("div", "card", root); h("h3", "", card, "Results"); h("p", "note", card, "Shortfall is the cost of the whole order against the price when it arrived, in basis points of its value: positive is a cost, negative a gain. The variation between days is its standard deviation across the simulated days; the ± is the standard error of the average.");
    simpleTable(card, [{ label: "Algorithm", get: (x) => x.algo }, { label: "Shortfall (bps)", num: true, get: (x) => `${num(x.shortfall_bps, 1)} ± ${num(x.se_bps, 1)}` }, { label: "Std dev (bps)", title: "Standard deviation of the shortfall across the simulated days", num: true, get: (x) => num(x.std_bps, 1) },
      { label: "5th to 95th pct (bps)", title: "The 5th and 95th percentiles of the shortfall across the simulated days", num: true, get: (x) => `${num(x.p05_bps, 0)} to ${num(x.p95_bps, 0)}` },
      { label: "Against VWAP (bps)", title: "How much worse than the market's volume-weighted average price over the order's window", num: true, get: (x) => num(x.vwap_slippage_bps, 1) },
      { label: "Share of volume", title: "Average share of the market's volume that the algorithm took", num: true, get: (x) => pct(x.participation, 1) }], rows);
    const ch = h("div", "charts", root), width = labelPx(rows.map((x) => x.algo));
    const bars = (key) => rows.map((x, i) => ({ label: x.algo, value: x[key], color: colors[i] }));
    chartCard(ch, "Average shortfall by algorithm", "Basis points of the order's value; lower is cheaper. The Table view splits it into spread, impact and timing.", {
      draw: (b) => C.hbars(b, { rows: bars("shortfall_bps"), format: (v) => num(v, 1), tipFormat: (v) => bps(v), name: "Average shortfall", label: "Average shortfall by algorithm", labelWidth: width, color: (v, row) => row.color }),
      table: (b) => simpleTable(b, [{ label: "Algorithm", get: (x) => x.algo }, { label: "Shortfall", num: true, get: (x) => num(x.shortfall_bps, 1) }, { label: "Spread", num: true, get: (x) => num(x.spread_bps, 1) },
        { label: "Temporary impact", num: true, get: (x) => num(x.temporary_bps, 1) }, { label: "Permanent impact", num: true, get: (x) => num(x.permanent_bps, 1) },
        { label: "Timing (price moves while it works)", num: true, get: (x) => num(x.timing_bps, 1) }, { label: "Fees", num: true, get: (x) => num(x.fees_bps, 1) }], rows) });
    chartCard(ch, "Variation between days by algorithm", "Standard deviation of the shortfall across the simulated days, in basis points. A lower average cost with more variation is not better; say which you want.", {
      draw: (b) => C.hbars(b, { rows: bars("std_bps"), format: (v) => num(v, 0), tipFormat: (v) => bps(v), name: "Variation between days", label: "Variation between days by algorithm", labelWidth: width, color: (v, row) => row.color }),
      table: (b) => simpleTable(b, [{ label: "Algorithm", get: (x) => x.algo }, { label: "Std dev", num: true, get: (x) => num(x.std_bps, 1) }, { label: "5th percentile", num: true, get: (x) => num(x.p05_bps, 1) },
        { label: "Median", num: true, get: (x) => num(x.median_bps, 1) }, { label: "95th percentile", num: true, get: (x) => num(x.p95_bps, 1) }], rows) });
    const slices = Array.from({ length: r.intervals }, (_, i) => i + 1), volume = r.volume_profile;
    const lines = () => {
      const scaled = (a) => (X.cumulative ? running(a) : a).map((v) => v * 100);
      return [...rows.map((x, i) => ({ name: x.algo, color: colors[i], values: scaled(r.schedules[x.algo]) })), { name: "Market volume", color: css("--muted"), values: scaled(volume), dash: "5 4", width: 1.5 }];
    };
    const sched = chartCard(ch, "How each algorithm spreads the order through the day", "Share of the order traded in each slice (dashed: the market's own volume, as a share of the day). VWAP follows the dashed line; front-loaded algorithms start above it.", { wide: true,
      draw: (b) => C.line(b, { xs: slices, xWhole: true, xLabel: "Slice of the day (1 is the open)", series: lines(), yFormat: (v) => v.toFixed(0) + "%", tipFormat: (v) => v.toFixed(1) + "%", xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `Slice ${slices[i]}`, height: 300, label: "Share of the order by slice", endLabels: false, include: 0 }),
      table: (b) => { const L = lines(); simpleTable(b, [{ label: "Slice", num: true, get: (i) => slices[i] }, ...L.map((s) => ({ label: s.name, num: true, get: (i) => num(s.values[i], 2) + "%" }))], slices.map((_, i) => i)); } });
    const seg = h("div", "seg", sched._extra), once = h("button", "", seg, "Per slice"), total = h("button", "", seg, "Cumulative");
    const mark = () => { once.setAttribute("aria-pressed", String(!X.cumulative)); total.setAttribute("aria-pressed", String(X.cumulative)); };
    mark(); once.addEventListener("click", () => { X.cumulative = false; mark(); sched._draw(); }); total.addEventListener("click", () => { X.cumulative = true; mark(); sched._draw(); });
    const sd = rows.map((x) => x.std_bps), xs = [...new Set(sd)].sort((a, b) => a - b);
    chartCard(ch, "Cost against risk", "Each point is one algorithm: average shortfall (up) against the variation between days (right). Lower and further left is better; a point that is higher and further right than another is beaten by it.", { wide: true,
      draw: (b) => C.line(b, { xs, xLabel: "Variation between days (standard deviation of the shortfall, bps)", series: rows.map((x, i) => ({ name: x.algo, color: colors[i], dots: true, dotSize: 5, pointLabel: x.algo, values: xs.map((v) => (v === x.std_bps ? x.shortfall_bps : null)) })),
        yFormat: plain, tipFormat: (v) => bps(v), xFormat: (v) => v.toFixed(0), tipTitle: (i) => `Varies by ${bps(xs[i])}`, height: 280, label: "Cost against risk", endLabels: false }),
      table: (b) => simpleTable(b, [{ label: "Algorithm", get: (x) => x.algo }, { label: "Shortfall (bps)", num: true, get: (x) => num(x.shortfall_bps, 1) }, { label: "Std dev (bps)", num: true, get: (x) => num(x.std_bps, 1) }], rows) });
    h("p", "hint", root, "A stylised market: impact follows an Almgren-Chriss model with illustrative parameters, there is no order book or queue, and the passive and dark fill rates are round numbers. Use it to compare methods and see which way a setting pushes, not to quote a cost.");
  }

  function execFrontier(root, r) {
    const f = r.frontier.slice().sort((a, b) => a.risk_bps - b.risk_bps), inp = r.inputs, first = r.frontier[0], last = r.frontier[r.frontier.length - 1];
    labHead(root, "The cost-risk frontier", `${sideWord(inp.side)} ${grouped(inp.shares)} shares · ${pct(inp.shares / inp.adv, 1)} of a day's volume · ${(inp.sigma * 100).toFixed(1)}% daily volatility · ${inp.spread_bps} bps spread · ${inp.intervals} slices`, "technique-execution-algorithms");
    const k = h("div", "earnrow", root);
    tile(k, "VWAP", bps(r.vwap.cost_bps), { text: `risk ${bps(r.vwap.risk_bps, 0)}` }, "", "Follow the day's volume: cheapest, and it carries the most price risk");
    tile(k, "TWAP", bps(r.twap.cost_bps), { text: `risk ${bps(r.twap.risk_bps, 0)}` });
    tile(k, "Most urgent schedule", bps(last.cost_bps), { text: `risk ${bps(last.risk_bps, 0)} · ${pct(last.first_slice, 0)} in the first slice` }, "", "The highest risk aversion on the curve");
    tile(k, "Least urgent schedule", bps(first.cost_bps), { text: `risk ${bps(first.risk_bps, 0)} · ${pct(first.first_slice, 0)} in the first slice` });
    const others = [["VWAP", r.vwap, "--s2"], ["TWAP", r.twap, "--s3"]], xs = [...new Set([...f.map((p) => p.risk_bps), ...others.map(([, v]) => v.risk_bps)])].sort((a, b) => a - b);
    const curve = xs.map(() => null); f.forEach((p) => { curve[xs.indexOf(p.risk_bps)] = p.cost_bps; });
    const series = [{ name: "Best schedule for each risk aversion", color: css("--s1"), values: curve, connect: true, dots: true },
      ...others.map(([name, v, c]) => { const values = xs.map(() => null); values[xs.indexOf(v.risk_bps)] = v.cost_bps; return { name, color: css(c), values, dots: true }; })];
    const ch = h("div", "charts", root);
    chartCard(ch, "Expected cost against risk", "Each point on the curve is the cheapest schedule for one price of risk (risk aversion). Trading faster costs more impact and leaves less exposure to the price; VWAP and TWAP are shown for reference.", { wide: true,
      draw: (b) => C.line(b, { xs, xLabel: "Risk (standard deviation of the shortfall, bps)", series, yFormat: plain, tipFormat: (v) => bps(v), xFormat: (v) => v.toFixed(0), tipTitle: (i) => `Risk ${bps(xs[i])}`, height: 320, label: "Expected cost against risk", endLabels: false,
        tipExtra: (i) => { const p = f.find((q) => q.risk_bps === xs[i]); return p ? [["Risk aversion", p.risk_aversion.toExponential(1)], ["First slice", pct(p.first_slice, 1) + " of the order"]] : []; } }),
      table: (b) => simpleTable(b, [{ label: "Risk aversion", num: true, get: (p) => p.risk_aversion.toExponential(1) }, { label: "Expected cost (bps)", num: true, get: (p) => num(p.cost_bps, 1) }, { label: "Risk (bps)", num: true, get: (p) => num(p.risk_bps, 1) },
        { label: "First slice (share of the order)", num: true, get: (p) => pct(p.first_slice, 1) }], r.frontier) });
    h("p", "hint", root, "The curve is the optimiser's answer for each risk aversion on a stylised market with illustrative impact parameters. Its shape (how much risk a few basis points buy) is the lesson; the levels are not a quote.");
  }

  function execBasket(root, r) {
    const j = r.joint, s = r.independent, m = r.mtrq, t = r.mto, p = r.program_block, share = r.inputs.share, n = r.intervals, slices = Array.from({ length: n }, (_, i) => i + 1);
    labHead(root, `A basket of ${r.size} names`, `${money(r.gross)} gross · ${money(r.net_exposure)} net · ${n} slices · risk aversion ${r.risk_aversion}`, "technique-basket-and-liquidity-algorithms");
    const k = h("div", "earnrow", root);
    tile(k, "Joint schedule", num(j.objective, 2), { text: `cost ${bps(j.cost_bps)}, risk ${bps(j.risk_bps)}`, dir: j.objective <= s.objective ? "up" : "" }, "", "Cost plus risk aversion times risk squared; lower is better");
    tile(k, "Stock by stock", num(s.objective, 2), { text: `cost ${bps(s.cost_bps)}, risk ${bps(s.risk_bps)}` }, "", "Each name scheduled alone at the same price of risk");
    tile(k, `Risk left after executing ${pct(share, 0)}`, money(m.residual_risk), { text: `${money(m.naive_risk)} in proportion, ${money(m.original_risk)} untouched`, dir: m.residual_risk <= m.naive_risk ? "up" : "" }, "", "Standard deviation of the remaining list's value change, executing the best names first");
    tile(k, "If only the buys are on offer", pct(t.share_of_list, 0), { text: t.feasible ? (t.binding ? "of the list's value can go before the rest is riskier" : "of the list's value: all that is on offer can go") : "cannot be done without raising risk" }, "", "Most value that can be executed without the remaining list being riskier than the original");
    tile(k, "Dark-safe names", `${p.dark.length} of ${p.block.length}`, { text: p.block.length ? (p.dark.length ? p.dark.join(", ") : "none can go dark safely") : "no block names" }, "", "Block names that may be entered in dark pools without raising risk, whatever fills");
    const ch = h("div", "charts", root), C1 = css("--s1"), C2 = css("--s2");
    chartCard(ch, "Risk of the unexecuted list through the day", "Standard deviation of the remaining list's value, in basis points of the list's gross value, after each slice. The joint schedule minimises cost plus a price on this risk for the whole list, so it may carry more or less of it than trading each name alone, wherever that saves cost.", {
      draw: (b) => C.line(b, { xs: slices, xWhole: true, xLabel: "Slice of the day", series: [{ name: "Joint schedule", color: C1, values: r.risk_left_bps.joint }, { name: "Stock by stock", color: C2, values: r.risk_left_bps.independent }], yFormat: (v) => v.toFixed(0), tipFormat: (v) => bps(v), xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `After slice ${slices[i]}`, height: 260, label: "Risk of the unexecuted list", endLabels: false, include: 0 }),
      table: (b) => simpleTable(b, [{ label: "After slice", num: true, get: (i) => slices[i] }, { label: "Joint (bps)", num: true, get: (i) => num(r.risk_left_bps.joint[i], 1) }, { label: "Stock by stock (bps)", num: true, get: (i) => num(r.risk_left_bps.independent[i], 1) }], slices.map((_, i) => i)) });
    chartCard(ch, "Value traded in each slice", "Share of the list's gross value traded in each slice.", {
      draw: (b) => C.line(b, { xs: slices, xWhole: true, xLabel: "Slice of the day", series: [{ name: "Joint schedule", color: C1, values: r.executed.joint.map((v) => v * 100) }, { name: "Stock by stock", color: C2, values: r.executed.independent.map((v) => v * 100) }], yFormat: (v) => v.toFixed(0) + "%", tipFormat: (v) => v.toFixed(1) + "%", xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `Slice ${slices[i]}`, height: 260, label: "Value traded by slice", endLabels: false, include: 0 }),
      table: (b) => simpleTable(b, [{ label: "Slice", num: true, get: (i) => slices[i] }, { label: "Joint", num: true, get: (i) => num(r.executed.joint[i] * 100, 2) + "%" }, { label: "Stock by stock", num: true, get: (i) => num(r.executed.independent[i] * 100, 2) + "%" }], slices.map((_, i) => i)) });
    const names = r.names.map((nm, i) => ({ name: nm, side: r.side[i] > 0 ? "buy" : "sell", value: r.value[i], mtrq: m.executed_fraction[i], mto: t.executed_fraction[i], block: p.block.includes(nm), dark: p.dark.includes(nm) }));
    chartCard(ch, `Share of each order to execute when ${pct(share, 0)} of the list's value can go`, "The fractions that leave the remaining list least risky (the minimum trading risk quantity). Hedged legs are executed together; a name at 0% is left for later.", { wide: true,
      draw: (b) => C.hbars(b, { rows: names.map((x) => ({ label: `${x.name} (${x.side})`, value: x.mtrq * 100, color: x.side === "buy" ? css("--s1") : css("--s2") })), format: (v) => v.toFixed(0) + "%", tipFormat: (v) => v.toFixed(0) + "% of the order", name: "Executed", label: "Share of each order to execute", labelWidth: 96, color: (v, row) => row.color }),
      table: (b) => simpleTable(b, [{ label: "Name", get: (x) => x.name }, { label: "Side", get: (x) => x.side }, { label: "Value", num: true, get: (x) => money(x.value) }, { label: "Class", get: (x) => (x.block ? "block" : "program") },
        { label: `Minimum-risk execution of ${pct(share, 0)}`, num: true, get: (x) => pct(x.mtrq, 0) }, { label: "If only the buys are on offer", num: true, get: (x) => pct(x.mto, 0) }, { label: "May go dark", get: (x) => (x.dark ? "yes" : "no") }], names) });
    h("p", "hint", root, p.exact ? "The dark-safe set is checked against every subset of its names that might fill." : "The dark-safe set is checked on a sample of the subsets that might fill, not all of them.");
    h("p", "hint", root, "A random demonstration list on a stylised market: the correlations come from one common factor and the impact parameters are illustrative. A list that is more hedged than a random one gains more from trading jointly.");
  }

  function execHft(root, r) {
    labHead(root, r.title, r.note, "technique-black-box-and-high-frequency-strategies");
    const fmtFor = (c) => (v) => (c === "win_rate" ? pct(v, 0) : ["days", "round_trips", "fills"].includes(c) ? grouped(v) : cell(v));
    const cols = [{ label: "", get: (x) => x.label }, ...r.columns.map((c) => ({ label: HFT_COLUMNS[c] || c.replace(/_/g, " "), num: true, get: (x) => fmtFor(c)(x[c]) }))];
    const card = h("div", "card", root); h("h3", "", card, "Results"); simpleTable(card, cols, r.rows);
    const ch = h("div", "charts", root), name = HFT_COLUMNS[r.chart] || r.chart.replace(/_/g, " ");
    if (r.rows.length > 1) chartCard(ch, name, "The figure the simulation is built to move. Blue is a gain and red a loss.", { wide: true,
      draw: (b) => C.hbars(b, { rows: r.rows.map((x) => ({ label: x.label, value: x[r.chart] })), format: (v) => cell(v), tipFormat: (v) => cell(v), name, label: name, labelWidth: labelPx(r.rows.map((x) => x.label)) }),
      table: (b) => simpleTable(b, [{ label: "", get: (x) => x.label }, { label: name, num: true, get: (x) => cell(x[r.chart]) }], r.rows) });
    if (r.series) chartCard(ch, r.series.name, "Running total of the simulated days.", { wide: true,
      draw: (b) => C.line(b, { xs: r.series.x, xWhole: true, xLabel: r.series.x_label.charAt(0).toUpperCase() + r.series.x_label.slice(1), series: [{ name: r.series.name, color: css("--s1"), values: r.series.y }], yFormat: (v) => v.toFixed(0), tipFormat: (v) => bps(v), xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `${r.series.x_label} ${r.series.x[i]}`, height: 260, label: r.series.name, baseline: 0, include: 0 }),
      table: (b) => simpleTable(b, [{ label: r.series.x_label, num: true, get: (i) => r.series.x[i] }, { label: r.series.name, num: true, get: (i) => num(r.series.y[i], 1) }], r.series.x.map((_, i) => i)) });
  }

  /* ---------- the Cash flows tab ---------- */
  const POLICY_NAMES = { pro_rata: "Pro rata", correct_drift: "Fix drift", rebalance: "Full rebalance", cash: "Hold cash", liquid: "Most liquid first" };
  const POLICY_ITEMS = [["pro_rata", "Pro rata: in proportion to the target"], ["correct_drift", "Fix drift: buy what is below target, sell what is above"], ["rebalance", "Full rebalance with every flow"], ["cash", "Hold in cash until the next rebalance"], ["liquid", "Most liquid first"]];
  const RULE_NAMES = { fixed_real: "Fixed real amount", percent_of_nav: "Percent of value", endowment: "Endowment (smoothed)", guardrails: "Guardrails" };
  const RULE_ITEMS = [["fixed_real", "Fixed real amount: the 4% rule"], ["percent_of_nav", "A share of the current value"], ["endowment", "Endowment: smoothed between the two"], ["guardrails", "Guardrails: cut or raise by 10% at the limits"]];
  const K = {
    id: "cash", root: "#cash", view: "#view-cash", title: "Cash flows", mode: "simulate", busy: "", results: {}, errors: {}, rule: null, guard: null, go: null, progress: null,
    about: "Follow a portfolio of real tickers through money coming in and going out. Tickers come from the Backtest tab's data source and start date.",
    modes: [["simulate", "Deposits and withdrawals"], ["spending", "Spending rules"], ["redeem", "Redemption"], ["ldi", "Liabilities (LDI)"]],
    runLabel: { simulate: "Follow the portfolio", spending: "Simulate spending", redeem: "Price the redemption", ldi: "Run the plan" },
    blurb: { simulate: "Add money, take money out and receive dividends under each policy, and see what each policy does to the money you end with, the drift from your target and the cost.",
      spending: "Spend from a portfolio each year under different rules over many futures made by resampling its own history: the chance of running out, and what spending looks like.",
      redeem: "Raise cash for a redemption from a fund: what selling pro rata, most liquid first, or from a cash buffer costs, and how long it takes.",
      ldi: "A plan that owes dated payments: how the funding ratio moves with interest rates if the plan holds long bonds against its liabilities, and a glide path that hedges more as it gets funded." },
    form: { holdings: "", initial: "100000", deposit: "1000", withdraw: "0", growth: "0", rebalance: "none", dividend_yield: "0", dividend_policy: "flow", cost_bps: "5", dca_months: "1", policies: new Set(["pro_rata", "correct_drift", "rebalance", "cash"]),
      spend_initial: "1000000", rate: "4", years: "30", inflation: "2", paths: "1000", target_ruin: "5", rules: new Set(["fixed_real", "percent_of_nav", "endowment", "guardrails"]),
      aum: "1000000000", redemption: "10", cash: "5", participation: "10", hedge: "", seeking: "", funding_ratio: "0.85", liability_years: "25" },
  };
  const CASH_HOLDINGS = { key: "holdings", label: "Holdings and weights", kind: "text", placeholder: "SPY=60, IEF=40", help: "Tickers with weights (any scale; they are normalised). Leave the weights out for equal weight." };
  const CASH_SIMULATE = [{ key: "initial", label: "Starting value ($)", kind: "number", min: 1000 }, { key: "deposit", label: "Deposit each month ($)", kind: "number", min: 0 }, { key: "withdraw", label: "Withdraw each month ($)", kind: "number", min: 0 },
    { key: "growth", label: "Yearly growth of the flows (%)", kind: "number", min: -50, max: 100, help: "A contribution that rises with pay, or a withdrawal that keeps up with inflation." },
    { key: "rebalance", label: "Scheduled rebalance", kind: "choice", options: [["none", "None"], ["monthly", "Monthly"], ["quarterly", "Quarterly"], ["annual", "Annual"], ["weekly", "Weekly"], ["daily", "Daily"]] },
    { key: "dividend_yield", label: "Dividend yield (%)", kind: "number", min: 0, max: 20 },
    { key: "dividend_policy", label: "Dividends are", kind: "choice", options: [["flow", "Handled like a deposit (by the policy)"], ["reinvest", "Reinvested in the asset that paid them"], ["cash", "Held as cash"]] },
    { key: "cost_bps", label: "Trading cost (bps)", kind: "number", min: 0, max: 200 }, { key: "dca_months", label: "Spread the starting value and each deposit over this many months", kind: "number", step: 1, min: 1, max: 36, help: "Dollar-cost averaging: invest in equal parts at the start of each month; the part not yet invested waits in cash." }];
  const CASH_SPENDING = [{ key: "spend_initial", label: "Starting portfolio ($)", kind: "number", min: 1000 }, { key: "rate", label: "First-year spending (% of the starting value)", kind: "number", min: 0.1, max: 50 },
    { key: "years", label: "Years", kind: "number", step: 1, min: 1, max: 50 }, { key: "inflation", label: "Inflation (% a year)", kind: "number", min: -5, max: 20 }, { key: "paths", label: "Simulated futures", kind: "number", step: 1, min: 100, max: 2000 },
    { key: "target_ruin", label: "Acceptable chance of running out (%)", kind: "number", min: 0.1, max: 50 }];
  const CASH_REDEEM = [{ key: "aum", label: "Fund size ($)", kind: "number", min: 100000 }, { key: "redemption", label: "Redemption (% of the fund)", kind: "number", min: 0.1, max: 90 }, { key: "cash", label: "Cash buffer (% of the fund)", kind: "number", min: 0, max: 90 },
    { key: "participation", label: "Most of the daily volume to take (%)", kind: "number", min: 1, max: 100 }];
  const CASH_LDI = [{ key: "hedge", label: "Bond funds to hedge with", kind: "text", placeholder: "TLT" }, { key: "seeking", label: "Return-seeking funds", kind: "text", placeholder: "SPY, EFA" },
    { key: "funding_ratio", label: "Funding ratio at the start (assets ÷ liabilities)", kind: "number", min: 0.3, max: 3 }, { key: "liability_years", label: "Years of payments owed (100 a year)", kind: "number", step: 1, min: 5, max: 60 }];
  const CASH_SCALE = { growth: 0.01, dividend_yield: 0.01, rate: 0.01, inflation: 0.01, target_ruin: 0.01, redemption: 0.01, cash: 0.01, participation: 0.01 };
  /* A ticker, which may end in =X or =F (Yahoo's currency pairs and futures), so a weight can only follow an = when it is a number: "SPY=60" is a weight, "SPY=abc" is a mistake. */
  const HOLDING_TICKER = /^[A-Za-z0-9^][A-Za-z0-9.^-]{0,11}(?:=[XxFf])?$/;
  function parseHoldings(text) {
    const weights = {}, order = [];
    for (const part of text.split(/[,;\n]+/).map((x) => x.trim()).filter(Boolean)) {
      const m = /^(.+?)\s*[=:]\s*(\d*\.?\d+)\s*%?$/.exec(part), weighted = m && HOLDING_TICKER.test(m[1]), ticker = weighted ? m[1] : HOLDING_TICKER.test(part) ? part : null;
      if (!ticker) throw new Error(`Cannot read "${part}": write holdings like SPY=60, IEF=40`);
      const t = ticker.toUpperCase(); if (t in weights) throw new Error(`${t} is listed twice`);
      weights[t] = weighted ? parseFloat(m[2]) : null; order.push(t);
    }
    if (!order.length) throw new Error("Name at least one holding");
    const given = order.filter((t) => weights[t] !== null).length;
    if (given && given < order.length) throw new Error("Give a weight to every holding, or to none");
    if (!given) for (const t of order) weights[t] = 1;
    if (order.every((t) => weights[t] === 0)) throw new Error("The weights add up to zero");
    return { tickers: order, weights };
  }
  const tickerList = (text) => text.split(/[\s,;]+/).map((x) => x.trim().toUpperCase()).filter(Boolean);
  const usable = () => S.tickers.filter((x) => x.status !== "bad");
  const classOf = (t) => ((usable().find((x) => x.t === t) || {}).cls) || (S.catalog.default_classes || {})[t] || "unknown";
  function seedCash() {
    const f = K.form, have = usable().map((x) => x.t);
    if (!f.holdings) f.holdings = have.includes("SPY") && have.includes("IEF") ? "SPY=60, IEF=40" : have.slice(0, 3).join(", ");
    if (!f.hedge) f.hedge = have.find((t) => ["rates", "fixed_income"].includes(classOf(t))) || "";
    if (!f.seeking) f.seeking = have.filter((t) => classOf(t) === "equity").slice(0, 2).join(", ");
  }
  function cashCommon() {
    const classes = {}, isDefault = new Set(S.catalog.default_tickers); for (const x of usable()) if (!isDefault.has(x.t) && x.cls !== "unknown") classes[x.t] = x.cls;
    return { classes, start: S.start || null, source: S.source };
  }
  K.check = () => {
    try { if (K.mode === "ldi") { if (!tickerList(K.form.hedge).length || !tickerList(K.form.seeking).length) return "Name a bond fund to hedge with and a return-seeking fund."; } else parseHoldings(K.form.holdings); }
    catch (e) { return e.message; }
    if (K.mode === "simulate" && !K.form.policies.size) return "Choose at least one policy.";
    if (K.mode === "spending" && !K.form.rules.size) return "Choose at least one spending rule.";
    return "";
  };
  K.sections = (panel) => {
    seedCash();
    const f = K.form, after = () => syncLab(K), sec = (title) => { const d = h("details", "", panel); d.open = true; h("summary", "", d, title); return h("div", "body", d); };
    const fields = (box, specs) => { for (const s of specs) labField(box, s, f, after); };
    if (K.mode === "ldi") fields(sec("The plan"), CASH_LDI);
    else { const p = sec("The portfolio"); labField(p, CASH_HOLDINGS, f, after); h("p", "hint", p, "Daily prices from the data source chosen in the Backtest tab, from its start date to the end of the shared history. Tickers that are not downloaded yet are fetched, within that source's rate limit."); }
    if (K.mode === "simulate") { fields(sec("The flows"), CASH_SIMULATE); checkList(sec("Policies to compare"), "How each flow is traded (up to five):", POLICY_ITEMS, f.policies, after); }
    if (K.mode === "spending") { fields(sec("The spending"), CASH_SPENDING); checkList(sec("Rules to compare"), "How the amount is set each year:", RULE_ITEMS, f.rules, after); }
    if (K.mode === "redeem") fields(sec("The redemption"), CASH_REDEEM);
  };
  K.run = () => {
    const f = K.form, m = K.mode, base = () => cashCommon();
    const held = () => { const p = parseHoldings(f.holdings); return { tickers: p.tickers, weights: p.weights }; };
    if (m === "simulate") return labRun(K, "/api/cash", () => ({ action: "simulate", ...base(), ...held(), policies: POLICY_ITEMS.map((p) => p[0]).filter((p) => f.policies.has(p)), ...numbers(f, ["initial", "deposit", "withdraw", "growth", "dividend_yield", "cost_bps", "dca_months"], CASH_SCALE), rebalance: f.rebalance, dividend_policy: f.dividend_policy }), "Following the portfolio day by day…");
    if (m === "spending") return labRun(K, "/api/cash", () => ({ action: "spending", ...base(), ...held(), rules: RULE_ITEMS.map((r) => r[0]).filter((r) => f.rules.has(r)), ...numbers({ ...f, initial: f.spend_initial }, ["initial", "rate", "years", "inflation", "paths", "target_ruin"], CASH_SCALE) }), "Simulating futures…");
    if (m === "redeem") return labRun(K, "/api/cash", () => ({ action: "redeem", ...base(), ...held(), ...numbers(f, ["aum", "redemption", "cash", "participation"], CASH_SCALE) }), "Pricing the sales…");
    return labRun(K, "/api/cash", () => { const hedge = tickerList(f.hedge), seeking = tickerList(f.seeking); return { action: "ldi", ...base(), tickers: [...new Set([...hedge, ...seeking])], hedge, seeking, ...numbers(f, ["funding_ratio", "liability_years"]) }; }, "Running the plan…");
  };
  K.empty = (root) => {
    const text = { simulate: ["Follow a portfolio through deposits and withdrawals", "Each policy decides which assets to buy with a deposit and sell for a withdrawal. You see the money you end with (what you earned, with the timing of the flows) beside the drift from your target mix and the cost of keeping it."],
      spending: ["See whether a spending rule lasts", "Resample the portfolio's own history into thousands of futures and spend from each under the rules you choose: how often the money runs out, how much spending has to be cut, and the highest safe starting rate."],
      redeem: ["Price a redemption", "Selling in proportion keeps the mix, selling the most liquid assets first is cheapest and leaves the book less liquid, and a cash buffer avoids selling at all. See what each costs and how long it takes."],
      ldi: ["Fund a plan that owes payments", "A plan owes 100 a year for a number of years. Holding long bonds against that liability keeps its funding ratio steady when interest rates move; compare no hedge, a glide path and a full hedge."] }[K.mode];
    labEmpty(root, K, text[0], [text[1], "Uses the tickers in the field on the left. The defaults need SPY and IEF (the platform's 15 ETFs include both)."], "Run it with these settings");
  };
  K.draw = () => drawLab(K);
  K.build = (root, r, body) => ({ simulate: cashSimulate, spending: cashSpending, redeem: cashRedeem, ldi: cashLdi })[K.mode](root, r, body);
  const holdingsText = (mix) => Object.entries(mix).map(([t, w]) => `${t} ${(w * 100).toFixed(0)}%`).join(" · ");
  function cashHead(root, r, title, guide, withMix) {
    labHead(root, title, [withMix === false ? null : holdingsText(r.mix), `${r.period[0]} to ${r.period[1]}`, r.universe && r.universe.source ? r.universe.source : null].filter(Boolean).join(" · "), guide);
  }

  function cashSimulate(root, r) {
    const rows = r.rows, i = r.inputs, names = rows.map((x) => POLICY_NAMES[x.policy] || x.policy), most = rows.reduce((a, b) => (b.final_value > a.final_value ? b : a)), closest = rows.reduce((a, b) => (b.mean_deviation < a.mean_deviation ? b : a));
    cashHead(root, r, "Money in and money out", "technique-cash-flow-strategies");
    h("p", "hint", root, `Starting with ${money(i.initial)}` + (i.deposit ? `, ${money(i.deposit)} added each month` : "") + (i.withdraw ? `, ${money(i.withdraw)} taken out each month` : "") + (i.growth ? `, the flows growing ${pct(i.growth, 1)} a year` : "") + `. Scheduled rebalance: ${i.rebalance}.` + (i.dividend_yield ? ` Dividend yield ${pct(i.dividend_yield, 1)}.` : "") + ` Trading cost ${i.cost_bps} bps.`);
    const k = h("div", "earnrow", root);
    tile(k, "Ended with the most", money(most.final_value), { text: `${POLICY_NAMES[most.policy] || most.policy} · money-weighted return ${pct(most.irr, 1)} a year` }, "", "Final value of the account");
    tile(k, "Stayed closest to the target", pct(closest.mean_deviation, 1), { text: `${POLICY_NAMES[closest.policy] || closest.policy} · average distance from the target weights` }, "", "Half the sum of the absolute differences between the weights and the target, averaged over the days");
    tile(k, "Put in and taken out", money(rows[0].deposits), { text: rows[0].withdrawals ? `taken out ${money(Math.abs(rows[0].withdrawals))}` : "nothing taken out" });
    const card = h("div", "card", root); h("h3", "", card, "By policy"); h("p", "note", card, "The money-weighted return is what the investor earned given when the money arrived; the time-weighted return is what the portfolio did, whatever the flows. They differ by the timing of the flows.");
    simpleTable(card, [{ label: "Policy", get: (x) => POLICY_NAMES[x.policy] || x.policy }, { label: "Ends with", num: true, get: (x) => money(x.final_value) }, { label: "Profit", num: true, get: (x) => smoney(x.profit) },
      { label: "Money-weighted", title: "Yearly return of the investor, given when the money arrived (the internal rate of return)", num: true, get: (x) => pct(x.irr, 1) },
      { label: "Time-weighted", title: "Yearly return of the portfolio, whatever the flows", num: true, get: (x) => pct(x.twr_annual, 1) }, { label: "Traded a year", title: "Dollars traded a year as a share of the average account value", num: true, get: (x) => pct(x.turnover, 1) },
      { label: "Costs", num: true, get: (x) => money(x.costs) }, { label: "Drift, average", title: "Average distance from the target weights (half the sum of the absolute differences)", num: true, get: (x) => pct(x.mean_deviation, 1) },
      { label: "Drift, largest", title: "Largest distance from the target weights", num: true, get: (x) => pct(x.max_deviation, 1) }, { label: "Cash, average", title: "Average share of the account held as cash", num: true, get: (x) => pct(x.mean_cash, 1) }], rows);
    const ch = h("div", "charts", root), series = (key, scale) => rows.map((x, n) => ({ name: names[n], color: colorOf(n), values: r[key][x.policy].map((v) => (v === null ? null : v * (scale || 1))) }));
    chartCard(ch, "Account value", "Weekly. Net of trading costs.", { wide: true,
      draw: (b) => C.line(b, { dates: r.dates, series: series("nav"), yFormat: axisMoney, tipFormat: (v) => money(v), height: 300, label: "Account value by policy", endLabels: false }),
      table: (b) => simpleTable(b, [{ label: "Week", get: (n) => r.dates[n] }, ...names.map((nm, n) => ({ label: nm, num: true, get: (j) => money(r.nav[rows[n].policy][j]) }))], sampleRows(r.dates, null, 13)) });
    chartCard(ch, "Distance from the target weights", "Half the sum of the absolute differences between the weights held and the target, weekly. Zero is exactly on target.", { wide: true,
      draw: (b) => C.line(b, { dates: r.dates, series: series("deviation"), baseline: 0, include: 0, yFormat: (v) => (v * 100).toFixed(0) + "%", tipFormat: (v) => pct(v, 1), height: 240, label: "Distance from the target weights", endLabels: false }),
      table: (b) => simpleTable(b, [{ label: "Week", get: (n) => r.dates[n] }, ...names.map((nm, n) => ({ label: nm, num: true, get: (j) => pct(r.deviation[rows[n].policy][j], 1) }))], sampleRows(r.dates, null, 13)) });
    h("p", "hint", root, "One historical path. The policy that ended highest did so partly because drift toward the asset that did well was rewarded in this sample; drift is a risk as often as a reward. The policies buy control of the mix, not return.");
  }

  function cashSpending(root, r) {
    const rows = r.rows, rules = rows.map((x) => x.rule); if (!rules.includes(K.rule)) K.rule = rules[0];
    cashHead(root, r, `Spending ${pct(r.rate, 1)} of ${money(r.initial)} a year for ${r.years} years`, "technique-cash-flow-strategies");
    h("p", "hint", root, `${grouped(r.paths)} futures made by resampling ${grouped(r.days)} days of this portfolio's own history in blocks of about a month (so calm and stormy spells stay together). Inflation ${pct(r.inflation, 1)}; all amounts are in today's dollars.`);
    const k = h("div", "earnrow", root);
    tile(k, "Highest safe starting rate", pct(r.sustainable_rate, 1), { text: `fixed real withdrawals, at most ${pct(r.target_ruin, 1)} chance of running out` }, "", "Found by bisection over the same futures");
    const first = rows[0]; tile(k, "Chance of running out", pct(first.ruin_probability, 1), { text: `${RULE_NAMES[first.rule] || first.rule} at ${pct(r.rate, 1)}` });
    tile(k, "Typical outcome", money(first.median_final_real_wealth), { text: `median wealth left after ${r.years} years (${RULE_NAMES[first.rule] || first.rule})` });
    const card = h("div", "card", root); h("h3", "", card, "By spending rule");
    simpleTable(card, [{ label: "Rule", get: (x) => RULE_NAMES[x.rule] || x.rule }, { label: "Runs out", title: "Share of futures in which the portfolio could not pay a year's spending in full", num: true, get: (x) => pct(x.ruin_probability, 1) },
      { label: "Spending cut", title: "Share of futures in which spending fell below 80% of the first year's at some point", num: true, get: (x) => pct(x.spending_cut_probability, 1) },
      { label: "Spent a year", title: "Average spending a year, in today's dollars", num: true, get: (x) => money(x.mean_real_spending) }, { label: "Years paid", title: "Average number of years paid in full", num: true, get: (x) => num(x.years_funded, 1) },
      { label: "Median left", title: "Median wealth at the end, in today's dollars", num: true, get: (x) => money(x.median_final_real_wealth) },
      { label: "5th pct left", title: "5th percentile of wealth at the end, in today's dollars: one future in twenty ends with less", num: true, get: (x) => money(x.p05_final_real_wealth) }], rows);
    const pick = h("div", "row", root); h("span", "hint", pick, "Show the range of outcomes for"); const seg = h("div", "seg", pick), buttons = [];
    const ch = h("div", "charts", root);
    const fan = (a, offset) => {
      const q = r.bands[K.rule][a], years = r.bands[K.rule].years, xs = offset ? years.slice(1) : years;
      return { xs, q };
    };
    const fanChart = (title, note, key, label) => chartCard(ch, title, note, { wide: true,
      draw: (b) => { const { xs, q } = fan(key, key === "spending"); C.line(b, { xs, xWhole: true, xLabel: "Year", series: [{ name: "Median", color: css("--s1"), values: q[2], width: 2.5 }],
        bands: [{ lo: q[0], hi: q[4], color: css("--s1"), name: "5th to 95th percentile", opacity: 0.12 }, { lo: q[1], hi: q[3], color: css("--s1"), name: "25th to 75th percentile", opacity: 0.22 }],
        yFormat: axisMoney, tipFormat: (v) => money(v), xFormat: (v) => String(Math.round(v)), tipTitle: (i) => `Year ${xs[i]}`, height: 300, label, include: 0, endLabels: false }); },
      table: (b) => { const { xs, q } = fan(key, key === "spending"); simpleTable(b, [{ label: "Year", num: true, get: (i) => xs[i] }, ...["5th", "25th", "Median", "75th", "95th"].map((nm, n) => ({ label: nm, num: true, get: (i) => money(q[n][i]) }))], xs.map((_, i) => i)); } });
    const wealth = fanChart(`Portfolio value left, in today's dollars`, `The middle half of futures lie in the darker band and nine in ten in the lighter one.`, "wealth", "Portfolio value by year");
    const spent = fanChart("Spending each year, in today's dollars", "What was actually paid: it falls below the plan when the money runs short.", "spending", "Spending by year");
    for (const rule of rules) { const b = h("button", "", seg, RULE_NAMES[rule] || rule); b.dataset.rule = rule; buttons.push(b); b.addEventListener("click", () => { K.rule = rule; mark(); wealth._draw(); spent._draw(); }); }
    const mark = () => buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.rule === K.rule))); mark();
    h("p", "hint", root, "A bootstrap of the past cannot produce a future worse than the history it samples. Twenty years of a 60/40 contains one financial crisis and one inflation shock; widen the sample or the stress before reading a small chance as a promise.");
  }

  function cashRedeem(root, r) {
    const i = r.inputs, rows = r.rows, cheapest = rows.reduce((a, b) => (b.cost_bps_of_fund < a.cost_bps_of_fund ? b : a));
    cashHead(root, r, `Redeeming ${pct(i.redemption, 1)} of a ${money(i.aum)} fund`, "technique-cash-flow-strategies");
    h("p", "hint", root, `${pct(1 - r.cash, 0)} of the fund is invested in the mix above` + (r.cash ? ` and ${pct(r.cash, 0)} is held as cash` : "") + `. Sales take at most ${pct(i.participation, 0)} of each asset's daily dollar volume. ${r.note}`);
    const k = h("div", "earnrow", root);
    tile(k, "Cheapest overall", bps(cheapest.cost_bps_of_fund, 2), { text: `of the fund · ${POLICY_NAMES[cheapest.policy] || cheapest.policy} (${bps(cheapest.cost_bps)} of what it sold)` }, "", "Spread, temporary and permanent impact of the sales, as a share of the whole fund so that policies that sell different amounts can be compared");
    const slow = rows.reduce((a, b) => (b.days > a.days ? b : a)); tile(k, "Slowest to finish", num(slow.days, 1) + " days", { text: POLICY_NAMES[slow.policy] || slow.policy });
    const drift = rows.reduce((a, b) => (b.drift_after > a.drift_after ? b : a)); tile(k, "Furthest from the mix afterwards", pct(drift.drift_after, 1), { text: POLICY_NAMES[drift.policy] || drift.policy });
    const card = h("div", "card", root); h("h3", "", card, "By policy");
    simpleTable(card, [{ label: "Policy", get: (x) => POLICY_NAMES[x.policy] || x.policy }, { label: "Cost, bps of the fund", title: "Cost of the sales as a share of the whole fund: the measure that compares policies selling different amounts", num: true, get: (x) => num(x.cost_bps_of_fund, 2) },
      { label: "Cost, bps of sales", title: "Cost of the sales as a share of the amount sold", num: true, get: (x) => num(x.cost_bps, 1) }, { label: "Days", title: "Days until the last sale is done, at the daily volume limit", num: true, get: (x) => num(x.days, 1) },
      { label: "Assets sold", num: true, get: (x) => String(Math.round(x.assets_sold)) }, { label: "Raised by selling", title: "Cash raised from the holdings; the rest of the redemption came from the cash buffer", num: true, get: (x) => money(x.raised) },
      { label: "Drift left", title: "Distance from the target mix after the sales (half the sum of the absolute differences)", num: true, get: (x) => pct(x.drift_after, 1) },
      { label: "Timing risk", title: "Average price risk of the sales while they are worked, in basis points", num: true, get: (x) => bps(x.risk_bps) }], rows);
    const ch = h("div", "charts", root), labels = rows.map((x) => POLICY_NAMES[x.policy] || x.policy), width = labelPx(labels);
    chartCard(ch, "Cost of the sales", "Basis points of the whole fund, which compares policies that sell different amounts; lower is cheaper. The table also has basis points of the amount sold.", {
      draw: (b) => C.hbars(b, { rows: rows.map((x, n) => ({ label: labels[n], value: x.cost_bps_of_fund, color: colorOf(n) })), format: (v) => num(v, 2), tipFormat: (v) => bps(v, 2) + " of the fund", name: "Cost", label: "Cost of the sales", labelWidth: width, color: (v, row) => row.color }),
      table: (b) => simpleTable(b, [{ label: "Policy", get: (x) => POLICY_NAMES[x.policy] || x.policy }, { label: "bps of the fund", num: true, get: (x) => num(x.cost_bps_of_fund, 3) }, { label: "bps of the amount sold", num: true, get: (x) => num(x.cost_bps, 2) }], rows) });
    chartCard(ch, "Days to finish", "At the daily volume limit above.", {
      draw: (b) => C.hbars(b, { rows: rows.map((x, n) => ({ label: labels[n], value: x.days, color: colorOf(n) })), format: (v) => num(v, 1), tipFormat: (v) => num(v, 2) + " days", name: "Days", label: "Days to finish", labelWidth: width, color: (v, row) => row.color }),
      table: (b) => simpleTable(b, [{ label: "Policy", get: (x) => POLICY_NAMES[x.policy] || x.policy }, { label: "Days", num: true, get: (x) => num(x.days, 2) }], rows) });
    h("p", "hint", root, "The fund holds exactly its target mix here, so 'sell what has drifted' would be the same as selling pro rata and is not listed. The cost uses the same impact model as the execution algorithms with illustrative parameters.");
  }

  function cashLdi(root, r) {
    const i = r.inputs, rows = r.rows;
    cashHead(root, r, "A plan that owes payments", "technique-cash-flow-strategies", false);
    h("p", "hint", root, `Owes 100 a year for ${i.liability_years} years, discounted at the 10-year Treasury yield; starts ${pct(i.funding_ratio, 0)} funded. Hedge with ${r.hedge.join(", ")}; seek return with ${r.seeking.join(", ")}.`);
    const k = h("div", "earnrow", root);
    for (const x of rows) tile(k, x.plan.charAt(0).toUpperCase() + x.plan.slice(1), pct(x.funding_ratio_vol, 1), { text: `volatility of the funding ratio · worst fall ${pct(x.max_funding_drawdown, 0)}` }, "", "Annualised volatility of the funding ratio");
    const card = h("div", "card", root); h("h3", "", card, "By plan");
    simpleTable(card, [{ label: "Plan", get: (x) => x.plan }, { label: "Funding ratio at the start", num: true, get: (x) => pct(x.funding_ratio_start, 0) }, { label: "At the end", num: true, get: (x) => pct(x.funding_ratio_end, 0) }, { label: "Lowest", num: true, get: (x) => pct(x.funding_ratio_min, 0) },
      { label: "Volatility", num: true, get: (x) => pct(x.funding_ratio_vol, 1) }, { label: "Worst fall", num: true, get: (x) => pct(x.max_funding_drawdown, 0) }, { label: "Share of the time below 100%", num: true, get: (x) => pct(x.share_below_one, 0) }], rows);
    const ch = h("div", "charts", root), plans = rows.map((x) => x.plan);
    const series = (key, scale) => plans.map((p, n) => ({ name: p, color: colorOf(n), values: r[key][p].map((v) => (v === null ? null : v * (scale || 1))) }));
    chartCard(ch, "Funding ratio", "Assets divided by the present value of the payments owed, weekly. Above 100% the plan can meet its promises.", { wide: true,
      draw: (b) => C.line(b, { dates: r.dates, series: series("funding_ratio", 100), baseline: 100, yFormat: (v) => v.toFixed(0) + "%", tipFormat: (v) => v.toFixed(0) + "%", height: 300, label: "Funding ratio by plan", endLabels: false }),
      table: (b) => simpleTable(b, [{ label: "Week", get: (n) => r.dates[n] }, ...plans.map((p) => ({ label: p, num: true, get: (n) => pct(r.funding_ratio[p][n], 0) }))], sampleRows(r.dates, null, 13)) });
    chartCard(ch, "Share of the assets in the hedge", "How much of the plan is held in the bond funds. The glide path raises it as the plan gets better funded.", { wide: true,
      draw: (b) => C.line(b, { dates: r.dates, series: series("hedge_weight", 100), include: 0, yFormat: (v) => v.toFixed(0) + "%", tipFormat: (v) => v.toFixed(0) + "%", height: 220, label: "Share of assets in the hedge", endLabels: false }),
      table: (b) => simpleTable(b, [{ label: "Week", get: (n) => r.dates[n] }, ...plans.map((p) => ({ label: p, num: true, get: (n) => pct(r.hedge_weight[p][n], 0) }))], sampleRows(r.dates, null, 13)) });
    h("p", "hint", root, "A flat discount yield and equal-weight buckets: a real plan discounts every payment on its own curve and hedges key-rate durations, not one number. The plan is open: no payments are made out of the assets in this view, so the funding ratio shows the hedge and not the cash drain.");
  }

  /* ================================================================= guides + markdown */
  const G = { index: null, slug: null, filter: "" };
  const FILE_SLUGS = { "dashboard.md": "dashboard", "how_to_add_a_strategy.md": "how_to_add_a_strategy", "START_HERE.md": "start_here", "glossary.md": "glossary", "tour_of_a_backtest_day.md": "tour_of_a_backtest_day", "roadmap_coverage.md": "roadmap_coverage", "algorithmic_trading.md": "algorithmic_trading", "quant_books_coverage.md": "quant_books_coverage", "platform_integration.md": "platform_integration", "feature_audit.md": "feature_audit" };
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
  function redraw() {
    if (!$("#view-backtest").hidden && S.current) renderResult(); if (!$("#view-compare").hidden) renderCompare(); if (!$("#view-builder").hidden) renderBuilder();
    if (!$("#view-exec").hidden) drawLab(X); if (!$("#view-cash").hidden) drawLab(K);
  }
  /* Charts are drawn at the pixel width of their container, so they are redrawn when the page gets wider or narrower (a window resize, a phone turned sideways, a scroll bar appearing).
     A change of HEIGHT is not a reason: it happens every time the user opens a table or a details tab, and redrawing the whole result then undid what they had just chosen and sent the
     page back to the top. */
  let rz, lastWidth = null;
  new ResizeObserver((entries) => {
    const w = Math.round(entries[entries.length - 1].contentRect.width);
    if (lastWidth === null) { lastWidth = w; return; }
    if (Math.abs(w - lastWidth) < 2) return;
    lastWidth = w;
    clearTimeout(rz); rz = setTimeout(() => {
      if (!$("#view-backtest").hidden && S.current && !S.busy) renderResult();
      if (!$("#view-exec").hidden) drawLab(X); if (!$("#view-cash").hidden) drawLab(K);
    }, 180);
  }).observe($("main"));

  /* The server is older than the page when it reports a lower API version; the first release with data sources did not report one, but it did send ``sources``. */
  const serverApi = () => S.catalog.api_version || (Array.isArray(S.catalog.sources) ? 2 : 1);
  function serverTooOld() {
    $(".layout").hidden = true;
    notice("old-server", "The app is older than this page. ", "The page was updated, but the app behind it is still running the old code: the files changed while it was running. " + RESTART
      + ` (The app reports version ${serverApi()}; this page needs ${PAGE_API}.)`, true);
    const again = h("button", "btn primary", $('#notice [data-notice="old-server"]'), "Reload this page"); again.style.marginLeft = "12px"; again.addEventListener("click", () => location.reload());
  }

  async function boot() {
    try { S.catalog = await api("/api/catalog"); } catch (e) { $("#results").textContent = "Could not reach the server: " + e.message; return; }
    if (serverApi() < PAGE_API) { serverTooOld(); return; }
    if (S.catalog.restart_needed) notice("restart", "The app was updated after it started. ", "It may still be running the old code. " + RESTART);
    const failed = [];
    const draw = (what, fn) => { try { fn(); } catch (e) { console.error(what, e); failed.push(`${what} (${e.message})`); } };
    draw("the ticker list", () => {
      const savedT = store("tickers"); S.tickers = savedT && savedT.length ? savedT.map((x) => ({ t: x.t, cls: x.cls, status: S.catalog.default_tickers.includes(x.t) ? "default" : "ok" })) : defaultTickers();
    });
    S.models = [{ name: S.catalog.models.some((m) => m.name === "dual_momentum") ? "dual_momentum" : S.catalog.models[0].name, params: {} }];
    const savedSource = store("source"); if (savedSource && sources().some((x) => x.name === savedSource)) S.source = savedSource;
    draw("the setup panel", buildPanel);
    draw("the data source menu", renderSource); draw("the tickers", renderUniverse); draw("the strategy list", renderStrategies); draw("the portfolio list", renderPortfolio);
    draw("the results area", renderEmpty); draw("the trial counter", updateTrials);
    if (failed.length) notice("draw", "Part of the page could not be drawn: ", failed.join("; ") + ". Reload the page; if it happens again, restart the app.", true);
    const t = store("tab"); if (t && TABS.includes(t) && t !== "backtest") showTab(t);
  }
  boot();
})();
