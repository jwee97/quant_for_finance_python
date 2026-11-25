/* The volatility surface view of the Options tab: four synchronised views (3D surface, 2D heatmap, volatility smiles, term structure) of ONE data set, drawn with Plotly.js when it is vendored
   (scripts/vendor_libs.py) and with the app's own ECharts and canvas charts when it is not (the page says which). The data are built on the server (assets/vol_surface.py): cleaned quotes, one grid for
   every expiry, nodes only where an expiry's own quotes support them (masked elsewhere, never extrapolated), the raw surface apart from the fitted one. This file only draws; it never smooths,
   interpolates or extrapolates. Server-side choices (side, coordinate, expiry range, dates) live in the lab's left panel; the display choices (view, units, raw/fitted, colour range, camera) live in the
   toolbar above the chart and are remembered while you change views. Whatever the server sends goes in with textContent or as chart data. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, C = Q && Q.C;
  if (!Q || !W) return;
  const h = Q.h, num = Q.num, tile = Q.tile, table = Q.simpleTable;
  const STATE = { tab: "3d", value: "vol", kind: "raw", cmin: "", cmax: "", camera: null };            // display choices, kept across runs and across views
  const VIRIDIS = ["#440154", "#482878", "#3e4a89", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];
  const DEFAULT_CAMERA = { eye: { x: 1.55, y: -1.75, z: 0.85 }, center: { x: 0, y: 0, z: -0.12 }, up: { x: 0, y: 0, z: 1 } };
  const PRESETS = { default: DEFAULT_CAMERA, top: { eye: { x: 0.01, y: -0.01, z: 2.4 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } }, smile: { eye: { x: 0.01, y: -2.4, z: 0.35 }, center: { x: 0, y: 0, z: -0.1 }, up: { x: 0, y: 0, z: 1 } },
    term: { eye: { x: 2.4, y: -0.01, z: 0.35 }, center: { x: 0, y: 0, z: -0.1 }, up: { x: 0, y: 0, z: 1 } } };
  let plotlyReady = null;
  const getPlotly = () => (plotlyReady || (plotlyReady = window.Libs ? window.Libs.load("plotly") : Promise.resolve(null)));
  const cssv = (n, d) => (getComputedStyle(document.documentElement).getPropertyValue(n).trim() || d);
  const colourAt = (t) => VIRIDIS[Math.max(0, Math.min(VIRIDIS.length - 1, Math.round(t * (VIRIDIS.length - 1))))];
  const unitLabel = () => (STATE.value === "vol" ? "Implied volatility (%)" : "Total variance σ²T");
  const fmtZ = (v) => (STATE.value === "vol" ? num(v, 2) + "%" : num(v, 4));

  /* ---- the numbers every view draws from: one place, so that units and masks never differ between views ---- */
  function model(d) {
    const S = d.surface, usable = S.meta.status === "surface" ? S.slices.filter((s) => s.usable) : [], wantFit = STATE.kind === "fitted" && S.z.fit, grid = (wantFit ? S.z.fit : S.z.raw)[STATE.value];
    const obs = S.observations.map((o) => ({ ...o, z: STATE.value === "vol" ? o.iv * 100 : o.w }));
    const byExpiry = new Map(); for (const o of obs) { if (!byExpiry.has(o.expiry)) byExpiry.set(o.expiry, []); byExpiry.get(o.expiry).push(o); }
    const custom = grid.map((row, i) => row.map((_, j) => [usable[i].expiry, usable[i].forward * Math.exp(S.meta.coordinate === "k" ? S.x[j] : 0)]));
    const flat = grid.flat().filter((v) => v !== null), auto = flat.length ? [Math.min(...flat), Math.max(...flat)] : [0, 1];
    const lo = STATE.cmin !== "" && isFinite(Number(STATE.cmin)) ? Number(STATE.cmin) : auto[0], hi = STATE.cmax !== "" && isFinite(Number(STATE.cmax)) ? Number(STATE.cmax) : auto[1];
    return { S, usable, grid, obs, byExpiry, custom, range: [lo, hi < lo ? lo + 1e-9 : hi], usedFit: !!wantFit, fitMissing: STATE.kind === "fitted" && !S.z.fit };
  }
  const themed = (extra) => ({ paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: cssv("--ink", "#e6e6e6"), size: 12 }, margin: { l: 56, r: 16, t: 28, b: 52 }, ...extra });
  const axis = (title, extra) => ({ title: { text: title }, gridcolor: cssv("--border", "rgba(128,128,128,.3)"), zerolinecolor: cssv("--border", "rgba(128,128,128,.3)"), color: cssv("--muted", "#aaa"), automargin: true, ...extra });
  const CONFIG = { responsive: true, displaylogo: false, scrollZoom: true, modeBarButtonsToRemove: ["toImage", "sendDataToCloud"] };

  /* ---- Plotly figures ---- */
  function fig3d(m) {
    const S = m.S, xt = S.x_label;
    const surf = { type: "surface", x: S.x, y: m.usable.map((s) => s.dte), z: m.grid, customdata: m.custom, connectgaps: false, colorscale: "Viridis", cmin: m.range[0], cmax: m.range[1], showscale: true, name: m.usedFit ? "Fitted (SVI)" : "Interpolated from quotes",
      colorbar: { title: { text: unitLabel(), side: "right" }, thickness: 14, len: 0.72, tickfont: { color: cssv("--muted", "#aaa") }, titlefont: { color: cssv("--ink", "#ddd") } },
      hovertemplate: `Expiry %{customdata[0]}<br>${"days to expiry"} %{y:.0f}<br>${xt} %{x:.3f}` + (S.meta.coordinate === "k" ? "<br>strike ≈ %{customdata[1]:.2f}" : "") + `<br>${STATE.value === "vol" ? "IV %{z:.2f}%" : "w %{z:.5f}"}<extra></extra>`,
      lighting: { ambient: 0.8, diffuse: 0.6, specular: 0.15, roughness: 0.7 }, contours: { z: { show: false } } };
    const pts = { type: "scatter3d", mode: "markers", name: "Quotes", x: m.obs.map((o) => o.x), y: m.obs.map((o) => o.dte), z: m.obs.map((o) => o.z), marker: { size: 2.6, color: "#ffffff", line: { color: "#111", width: 0.5 }, opacity: 0.9 },
      customdata: m.obs.map((o) => [o.expiry, o.strike, o.right]), hovertemplate: `Quote ${"%{customdata[2]}"}<br>Expiry %{customdata[0]}<br>days %{y:.0f}<br>strike %{customdata[1]:.2f}<br>${xt} %{x:.3f}<br>${STATE.value === "vol" ? "IV %{z:.2f}%" : "w %{z:.5f}"}<extra></extra>` };
    const layout = themed({ margin: { l: 0, r: 0, t: 0, b: 0 }, uirevision: "vs3d", showlegend: false, scene: { camera: STATE.camera || DEFAULT_CAMERA, aspectmode: "manual", aspectratio: { x: 1.5, y: 1.25, z: 0.7 },
      xaxis: axis(S.meta.coordinate === "k" ? "ln(K/F)" : "call delta"), yaxis: axis("days to expiry"), zaxis: axis(unitLabel()), bgcolor: "rgba(0,0,0,0)" } });
    return { data: [surf, pts], layout };
  }
  function figHeat(m) {
    const S = m.S, xt = S.x_label;
    const heat = { type: "heatmap", x: S.x, y: m.usable.map((s) => s.dte), z: m.grid, zmin: m.range[0], zmax: m.range[1], colorscale: "Viridis", hoverongaps: false, customdata: m.custom, xgap: 0, ygap: 0,
      colorbar: { title: { text: unitLabel(), side: "right" }, thickness: 14, tickfont: { color: cssv("--muted", "#aaa") }, titlefont: { color: cssv("--ink", "#ddd") } },
      hovertemplate: `Expiry %{customdata[0]}<br>days %{y:.0f}<br>${xt} %{x:.3f}` + (S.meta.coordinate === "k" ? "<br>strike ≈ %{customdata[1]:.2f}" : "") + `<br>${STATE.value === "vol" ? "IV %{z:.2f}%" : "w %{z:.5f}"}<extra></extra>` };
    const pts = { type: "scatter", mode: "markers", name: "Quotes", x: m.obs.map((o) => o.x), y: m.obs.map((o) => o.dte), marker: { size: 3, color: "rgba(255,255,255,.85)", line: { color: "#111", width: 0.4 } },
      customdata: m.obs.map((o) => [o.expiry, o.strike, o.z]), hovertemplate: `Quote<br>Expiry %{customdata[0]}<br>strike %{customdata[1]:.2f}<br>${xt} %{x:.3f}<br>${STATE.value === "vol" ? "IV %{customdata[2]:.2f}%" : "w %{customdata[2]:.5f}"}<extra></extra>` };
    return { data: [heat, pts], layout: themed({ uirevision: "vsheat", showlegend: false, plot_bgcolor: "rgba(128,128,128,.22)", xaxis: axis(xt), yaxis: axis("days to expiry") }) };
  }
  function figSmiles(m) {
    const S = m.S, xt = S.x_label, take = m.usable.length > 10 ? m.usable.filter((_, i) => i % Math.ceil(m.usable.length / 10) === 0) : m.usable, all = S.slices.filter((s) => (!s.usable || !m.usable.includes(s)) && m.byExpiry.has(s.expiry));
    const data = [];
    take.forEach((s) => {
      const i = m.usable.indexOf(s), col = colourAt(m.usable.length > 1 ? i / (m.usable.length - 1) : 0), o = m.byExpiry.get(s.expiry) || [];
      data.push({ type: "scatter", mode: "markers", name: `${s.dte}d quotes`, legendgroup: s.expiry, x: o.map((q) => q.x), y: o.map((q) => q.z), marker: { size: 5, color: col, line: { color: "#111", width: 0.5 } }, customdata: o.map((q) => [q.expiry, q.strike, q.right]),
        hovertemplate: `Quote ${"%{customdata[2]}"}<br>Expiry %{customdata[0]}<br>strike %{customdata[1]:.2f}<br>${xt} %{x:.3f}<br>${STATE.value === "vol" ? "IV %{y:.2f}%" : "w %{y:.5f}"}<extra></extra>` });
      data.push({ type: "scatter", mode: "lines", name: `${s.dte}d ${m.usedFit ? "SVI fit" : "interpolated"}`, legendgroup: s.expiry, showlegend: false, x: S.x, y: m.grid[i], connectgaps: false, line: { color: col, width: 1.6, dash: m.usedFit ? "solid" : "dot" },
        hovertemplate: `${s.dte}d<br>${xt} %{x:.3f}<br>${STATE.value === "vol" ? "%{y:.2f}%" : "%{y:.5f}"}<extra>${m.usedFit ? "fitted" : "interpolated"}</extra>` });
    });
    for (const s of all) { const o = m.byExpiry.get(s.expiry); data.push({ type: "scatter", mode: "markers", name: `${s.dte}d (too few quotes)`, x: o.map((q) => q.x), y: o.map((q) => q.z), marker: { size: 5, symbol: S.meta.status === "surface" ? "x" : "circle", color: cssv("--muted", "#999") }, visible: S.meta.status === "surface" ? "legendonly" : true }); }
    return { data, layout: themed({ uirevision: "vssmile", legend: { orientation: "h", y: -0.22, font: { color: cssv("--muted", "#aaa") } }, hovermode: "closest", xaxis: axis(xt), yaxis: axis(unitLabel()) }) };
  }
  function figTerm(d, m) {
    const S = m.S, rows = S.term.rows, conv = (iv, dte) => (STATE.value === "vol" ? iv : (iv / 100) ** 2 * dte / 365);
    const x = rows.map((r) => r.dte), data = [{ type: "scatter", mode: "lines+markers", name: "At the money (quotes)", x, y: rows.map((r) => (r.atm_iv === null ? null : conv(r.atm_iv, r.dte))), connectgaps: false, line: { color: colourAt(0.55), width: 2 }, marker: { size: 6 },
      customdata: rows.map((r) => r.expiry), hovertemplate: `Expiry %{customdata}<br>days %{x:.0f}<br>${STATE.value === "vol" ? "ATM IV %{y:.2f}%" : "ATM w %{y:.5f}"}<extra>quotes</extra>` }];
    if (rows.some((r) => r.atm_iv_fit !== null)) data.push({ type: "scatter", mode: "lines", name: "At the money (SVI fit)", x, y: rows.map((r) => (r.atm_iv_fit === null ? null : conv(r.atm_iv_fit, r.dte))), connectgaps: false, line: { color: colourAt(0.95), width: 1.6, dash: "dash" } });
    const hist = S.history && S.history.term && S.history.term.length ? S.history : null;
    if (hist) data.push({ type: "scatter", mode: "lines+markers", name: `${hist.date}`, x: hist.term.map((r) => r.dte), y: hist.term.map((r) => conv(r.atm_iv, r.dte)), line: { color: cssv("--muted", "#999"), width: 1.4, dash: "dot" }, marker: { size: 4 } });
    return { data, layout: themed({ uirevision: "vsterm", legend: { orientation: "h", y: -0.22, font: { color: cssv("--muted", "#aaa") } }, xaxis: axis("days to expiry"), yaxis: axis(STATE.value === "vol" ? "At-the-money volatility (%)" : "At-the-money total variance") }) };
  }

  /* ---- fallback drawings (no Plotly): the app's own charts, same data and masks ---- */
  function fallback(tab, host, m, d) {
    host.replaceChildren();
    const S = m.S;
    if (tab === "3d") { window.Surface3D.draw(host, { x: S.x, y: m.usable.map((s) => s.dte), z: m.grid.map((r) => r.map((v) => (v === null ? null : v))), xLabel: S.x_label, yLabel: "days", zLabel: unitLabel(), format: fmtZ, label: "Implied volatility surface" }); return; }
    if (tab === "heat") { C.matrix(host, { rows: m.usable.map((s) => `${s.dte}d`), cols: S.x.map((x) => x.toFixed(2)), values: m.grid, min: m.range[0], max: m.range[1], format: fmtZ, name: unitLabel(), cellHeight: 24, label: "Implied volatility heatmap" }); return; }
    if (tab === "smile" && !m.usable.length) {                                                          // discrete smiles only: the quotes themselves, no lines
      const card = h("div", "", host); table(card, [{ label: "Expiry", get: (o) => o.expiry }, { label: "Strike", num: true, get: (o) => num(o.strike, 2) }, { label: S.x_label, num: true, get: (o) => num(o.x, 3) }, { label: unitLabel(), num: true, get: (o) => fmtZ(o.z) }], m.obs); return;
    }
    if (tab === "smile") {
      const series = m.usable.slice(0, 10).map((s, i) => ({ name: `${s.dte}d`, color: colourAt(m.usable.length > 1 ? i / (m.usable.length - 1) : 0), values: m.grid[m.usable.indexOf(s)], width: 1.6 }));
      C.line(host, { xs: S.x, xLabel: S.x_label, series, yFormat: (v) => num(v, STATE.value === "vol" ? 1 : 3), tipFormat: fmtZ, xFormat: (v) => v.toFixed(2), height: 320, label: "Volatility smiles" }); return;
    }
    const rows = S.term.rows;
    C.line(host, { xs: rows.map((r) => r.dte), xLabel: "days to expiry", series: [{ name: "At the money", color: colourAt(0.55), values: rows.map((r) => (r.atm_iv === null ? null : STATE.value === "vol" ? r.atm_iv : (r.atm_iv / 100) ** 2 * r.dte / 365)) }], yFormat: (v) => num(v, 2), tipFormat: (v) => num(v, 3), height: 300, label: "Term structure" });
  }

  /* ---- the view ---- */
  function view(root, d, ctx) {
    const S = d.surface, m0 = S.meta;
    ctx.header(root, `${ctx.nameOf(d)}: implied-volatility surface`, `${d.as_of} · ${m0.selection === "otm" ? "out-of-the-money quotes" : m0.selection + " quotes"} · coordinate ${S.x_label}`);
    W.provenance(root, d);
    const strip = h("div", "earnrow", root); strip.dataset.role = "vs-status";
    const q = m0.quotes, cov = S.coverage || { share: 0, supported: 0, nodes: 0 };
    tile(strip, "Source", d.source_kind === "snapshot" ? "Kept chain" : d.source_kind === "simulated" ? "Simulated model" : d.source_kind === "user_file" ? "Your file" : "Downloaded", { text: `observed ${d.as_of}` }, "", "Where the quotes come from and the date they are for.");
    tile(strip, "Quotes used", `${q.kept} of ${q.received}`, { text: `${S.slices.length} expiries listed, ${m0.expiries_usable} usable` }, "", "Quotes that survived the cleaning; the rejections by reason are below.");
    tile(strip, "Surface coverage", m0.status === "surface" ? `${Math.round(cov.share * 100)}%` : "none", { text: m0.status === "surface" ? `${cov.supported} of ${cov.nodes} grid nodes have a quote around them` : "no surface drawn" }, m0.status === "surface" ? "" : "muted", "The rest is masked: no quote supports it, and nothing is extrapolated.");
    tile(strip, "Calibration", S.fit ? `${num(S.fit.mean_rmse_vol_points, 2)} pts` : "n/a", { text: S.fit ? `SVI per expiry · ${S.fit.calendar.violations} calendar, ${S.fit.butterfly_failing_slices} butterfly violations` : "no fit for this coordinate" }, S.fit && (S.fit.calendar.violations || S.fit.butterfly_failing_slices) ? "" : "muted", S.fit ? S.fit.reading : "The SVI fit is in log-moneyness.");
    if (S.fit && S.fit.status_text) { const a = h("p", "hint", root, S.fit.status_text); a.dataset.role = "vs-arbitrage"; }
    if (m0.recommended_view === "smile" && m0.recommend_reason) { const a = h("div", "banner", root, m0.recommend_reason); a.dataset.role = "vs-sparse"; if (STATE.tab === "3d" && !STATE.touched) STATE.tab = "smile"; }
    const note = h("div", "vs-notice", root); note.dataset.role = "vs-notice";
    if (m0.status !== "surface") { note.textContent = m0.reason; note.className = "banner err vs-notice"; }
    const tabs = [["3d", "3D surface"], ["heat", "2D heatmap"], ["smile", "Volatility smiles"], ["term", "Term structure"]].filter(([id]) => m0.status === "surface" || (id === "smile" && S.observations.length) || (id === "term" && S.term && S.term.rows.length));
    if (!tabs.length) { renderRejections(root, S); notesBlock(root, S); return; }
    if (!tabs.some(([id]) => id === STATE.tab)) STATE.tab = tabs[0][0];
    const bar = h("div", "vs-bar", root);
    const tabsEl = h("div", "tabsmall vs-tabs", bar); tabsEl.setAttribute("role", "tablist");
    const ctl = h("div", "vs-controls", bar);
    const field = (label, build) => { const l = h("label", "vs-field", ctl); h("span", "", l, label); return build(l); };
    field("Show", (l) => { const s = h("select", "", l); s.dataset.role = "vs-value"; for (const [v, t] of [["vol", "Implied volatility"], ["w", "Total variance"]]) { const o = h("option", "", s, t); o.value = v; if (STATE.value === v) o.selected = true; } s.addEventListener("change", () => { STATE.value = s.value; STATE.cmin = STATE.cmax = ""; sync(); }); return s; });
    field("Surface", (l) => { const s = h("select", "", l); s.dataset.role = "vs-kind"; for (const [v, t] of [["raw", "Raw (interpolated from quotes)"], ["fitted", "Fitted (SVI)"]]) { const o = h("option", "", s, t); o.value = v; if (STATE.kind === v) o.selected = true; } s.addEventListener("change", () => { STATE.kind = s.value; sync(); }); return s; });
    const mkRange = (key, ph) => field(ph, (l) => { const i = h("input", "", l); i.type = "number"; i.step = "any"; i.placeholder = "auto"; i.value = STATE[key]; i.dataset.role = "vs-" + key; i.setAttribute("aria-label", ph); i.addEventListener("change", () => { STATE[key] = i.value; sync(); }); return i; });
    mkRange("cmin", "Colour min"); mkRange("cmax", "Colour max");
    const camBox = h("span", "vs-cam", ctl);
    const preset = h("select", "", camBox); preset.dataset.role = "vs-camera"; preset.setAttribute("aria-label", "camera preset");
    for (const [v, t] of [["default", "Camera: default"], ["top", "Top view"], ["smile", "Along the smile"], ["term", "Along the term structure"]]) { const o = h("option", "", preset, t); o.value = v; }
    const reset = h("button", "btn small", camBox, "Reset camera"); reset.type = "button"; reset.dataset.role = "vs-reset";
    const stage = h("div", "vs-stage", root); stage.dataset.role = "vs-stage";
    const banner = h("div", "hint vs-engine", stage);
    const panels = {};
    for (const [id] of tabs) { const p = h("div", "vs-panel vs-" + id, stage); p.dataset.panel = id; p.dataset.tab = id; p.hidden = id !== STATE.tab; panels[id] = h("div", "vs-plot", p); panels[id].dataset.role = "vs-plot-" + id; }
    let P = null, drawn = {};
    const m = () => model(d);
    const draw = (id) => {
      const mm = m(), host = panels[id];
      const msg = [];
      if (mm.fitMissing) msg.push("The fitted surface exists only in log-moneyness: showing the raw one.");
      if (mm.usedFit) msg.push("Fitted values are drawn only where quotes exist; points are the quotes.");
      banner.textContent = (P ? "" : "Plotly.js is not installed, so the app's own charts are used (rotate the 3D view by dragging; run scripts/vendor_libs.py for the full interactive version). ") + msg.join(" ");
      if (!P) { fallback(id, host, mm, d); drawn[id] = true; return; }
      const f = id === "3d" ? fig3d(mm) : id === "heat" ? figHeat(mm) : id === "smile" ? figSmiles(mm) : figTerm(d, mm);
      P.react(host, f.data, f.layout, CONFIG);                                                           // react, not newPlot: the figure is updated in place and keeps its camera and zoom
      if (!drawn[id] && host.on) {
        host.on("plotly_relayout", (e) => { const c = e && (e["scene.camera"] || (host.layout && host.layout.scene && host.layout.scene.camera)); if (id === "3d" && c) STATE.camera = JSON.parse(JSON.stringify(c)); });
      }
      drawn[id] = true;
    };
    const sync = () => { drawn = {}; for (const [id] of tabs) if (id === STATE.tab) draw(id); };
    const show = (id) => {
      STATE.tab = id; STATE.touched = true;
      for (const b of tabsEl.children) b.setAttribute("aria-selected", String(b.dataset.tab === id));
      for (const [k, p] of Object.entries(panels)) p.parentElement.hidden = k !== id;
      if (!drawn[id]) draw(id); else if (P && P.Plots) P.Plots.resize(panels[id]);                       // a panel drawn while hidden has no size: resize when it shows
    };
    for (const [id, label] of tabs) { const b = h("button", "", tabsEl, label); b.type = "button"; b.dataset.tab = id; b.setAttribute("role", "tab"); b.setAttribute("aria-selected", String(id === STATE.tab)); b.addEventListener("click", () => show(id)); }
    preset.addEventListener("change", () => { STATE.camera = PRESETS[preset.value]; if (P && panels["3d"]) P.relayout(panels["3d"], { "scene.camera": STATE.camera }); });
    reset.addEventListener("click", () => { STATE.camera = null; preset.value = "default"; if (P && panels["3d"] && drawn["3d"]) P.relayout(panels["3d"], { "scene.camera": DEFAULT_CAMERA }); });
    let ro = null;
    getPlotly().then((lib) => {
      P = lib && lib.react ? lib : null;
      show(STATE.tab);
      if (P && window.ResizeObserver) { ro = new ResizeObserver(() => { const p = panels[STATE.tab]; if (p && !p.parentElement.hidden) P.Plots.resize(p); }); ro.observe(stage); }
    });
    renderRejections(root, S);
    notesBlock(root, S);
  }
  function renderRejections(root, S) {
    const q = S.meta.quotes, det = h("details", "vs-rej", root); det.dataset.role = "vs-rejections"; h("summary", "", det, `Quote cleaning: ${q.kept} kept of ${q.received}`);
    const names = { expired: "expired", crossed_or_no_bid: "no bid or a crossed market", no_implied_vol: "no implied volatility", extreme_vol: "volatility above 300%", negligible_price: "price too small to carry information", other_side: "other side of the chain (not selected)", duplicate: "duplicate contract" };
    table(det, [{ label: "Reason", get: (r) => names[r[0]] || r[0] }, { label: "Quotes", num: true, get: (r) => String(r[1]) }], Object.entries(q.rejected));
    const sl = h("div", "", det); table(sl, [{ label: "Expiry", get: (s) => s.expiry }, { label: "Days", num: true, get: (s) => String(s.dte) }, { label: "Quotes", num: true, get: (s) => String(s.n) }, { label: "In the surface", get: (s) => (s.usable ? `yes · ${s.supported_nodes || 0} nodes` : "no: " + s.reason) },
      { label: "Fit", get: (s) => (s.fit && s.fit.rmse_vol_points !== undefined ? `${num(s.fit.rmse_vol_points, 2)} pts · butterfly ${s.fit.butterfly_ok_in_quotes ? "ok" : "violated"} (${s.fit.quality})` : "") }, { label: "Gaps in the strikes", get: (s) => (s.gaps && s.gaps.length ? s.gaps.map((g) => `${num(g[0], 2)} to ${num(g[1], 2)}`).join(", ") : "none") }], S.slices);
  }
  function notesBlock(root, S) {
    const box = h("div", "vs-notes", root); box.dataset.role = "vs-notes";
    for (const t of S.limits) h("p", "hint", box, t);
    if (S.fit) h("p", "hint", box, S.fit.reading);
  }
  /* The one 3D surface component: other views (the surface research card) call this instead of drawing their own. ``spec``: x, y (days), z (volatility in decimals, null where unsupported), xLabel. */
  function plain3d(host, spec, height) {
    const z = spec.z.map((r) => r.map((v) => (v === null ? null : v * 100)));
    const div = h("div", "", host); div.dataset.engine = "plotly"; div.style.cssText = `width:100%;height:${height || 380}px`;
    getPlotly().then((P) => {
      if (!P || !P.newPlot) { div.remove(); window.Surface3D.draw(host, { x: spec.x, y: spec.y, z: spec.z, xLabel: spec.xLabel, yLabel: "days", zLabel: "implied vol", format: (v) => num(v * 100, 1) + "%", label: "Implied volatility surface" }); return; }
      const ax = (title) => ({ title, gridcolor: cssv("--grid", "#8884"), color: cssv("--ink", "#222") });
      P.newPlot(div, [{ type: "surface", x: spec.x, y: spec.y, z, connectgaps: false, colorscale: VIRIDIS.map((c, i) => [i / (VIRIDIS.length - 1), c]), colorbar: { title: "IV %", len: 0.6 }, hovertemplate: spec.xLabel + " %{x:.3f}<br>%{y:.0f} days<br>IV %{z:.2f}%<extra></extra>" }],
        { margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)", font: { color: cssv("--ink", "#222") }, uirevision: "plain3d", scene: { camera: DEFAULT_CAMERA, xaxis: ax(spec.xLabel), yaxis: ax("days to expiry"), zaxis: ax("implied vol (%)") } }, { responsive: true, displaylogo: false });
    });
    return div;
  }
  window.VolSurface = { view, STATE, model, fig3d, figHeat, figSmiles, figTerm, plain3d };
})();
