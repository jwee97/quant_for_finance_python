/* Trade panel: one asset's price with its trades, and the position it held, on ONE shared date axis.

   Price (top), position as a step area (below: long up in blue, short down in red; or the whole book's long and short exposure), and an overview strip (a draggable window over the whole history) are
   drawn in a single SVG, so zooming, panning and the crosshair move all of them together. Above the chart sit the numbers that matter now; above the plot a compact bar: range buttons, reset, a
   view switch and a collapsible legend. Nothing is recomputed: every number is one the run recorded.

   Large histories stay light: the price line is reduced to the minimum and maximum of each pixel column (the shape of every spike survives), the position is drawn from its change points (a step line loses
   nothing), and trades are never dropped: openings, closings, flips and stops are always drawn one by one, and only the routine adds and trims of a crowded stretch are merged into one numbered mark
   (zoom in and they separate). A mouse drags to zoom and moves a crosshair; a finger drags the crosshair and uses the overview strip and the range buttons to zoom and pan, so the page still scrolls. */
(() => {
  "use strict";
  const C = window.Charts, { el, html } = C;
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const pct = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "n/a" : (v * 100).toFixed(d) + "%");
  const spct = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "n/a" : (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%");
  const px = (v) => (v === null || v === undefined || !isFinite(v) ? "n/a" : Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2));
  const day = (s) => Date.parse(s + "T00:00:00Z");
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  /* What each kind of trade looks like. Shape and fill say it as well as colour: a filled triangle opens or closes a position, an outlined one adds to it or trims it, a cross is a stop, target or time exit. */
  const KIND = {
    OPEN_LONG: { shape: "up", color: "--pos", label: "Buy (open long)", group: "buy" }, ADD_LONG: { shape: "up", color: "--pos", label: "Add to long", group: "buy", hollow: true },
    CLOSE_LONG: { shape: "down", color: "--neg", label: "Sell (close long)", group: "sell" }, TRIM_LONG: { shape: "down", color: "--neg", label: "Trim long", group: "sell", hollow: true },
    OPEN_SHORT: { shape: "down", color: "--s5", label: "Short (open short)", group: "short" }, ADD_SHORT: { shape: "down", color: "--s5", label: "Add to short", group: "short", hollow: true },
    CLOSE_SHORT: { shape: "up", color: "--s2", label: "Cover (close short)", group: "cover" }, TRIM_SHORT: { shape: "up", color: "--s2", label: "Reduce short", group: "cover", hollow: true },
    FLIP_TO_LONG: { shape: "up", color: "--pos", label: "Flip to long", group: "buy" }, FLIP_TO_SHORT: { shape: "down", color: "--s5", label: "Flip to short", group: "short" },
  };
  const EXITS = /^(STOP_LOSS|TAKE_PROFIT|TIME_EXIT)/;
  const isExit = (m) => EXITS.test(String(m.action || ""));
  const look = (m) => (isExit(m) ? { shape: "cross", color: "--ink", label: "Exit: " + m.action, group: "exit" } : KIND[m.kind] || { shape: m.side === "BUY" ? "up" : "down", color: m.side === "BUY" ? "--pos" : "--neg", label: m.kind, group: "other" });
  const ROUTINE = (m) => !isExit(m) && /^(ADD|TRIM)_/.test(m.kind || "");                                       // may be merged when crowded; everything else is always drawn alone

  function shapePath(shape, x, y, r) {
    if (shape === "up") return `M${x},${y - r}L${x + r},${y + r * 0.8}L${x - r},${y + r * 0.8}Z`;
    if (shape === "down") return `M${x},${y + r}L${x + r},${y - r * 0.8}L${x - r},${y - r * 0.8}Z`;
    return `M${x - r * 0.8},${y - r * 0.8}L${x + r * 0.8},${y + r * 0.8}M${x + r * 0.8},${y - r * 0.8}L${x - r * 0.8},${y + r * 0.8}`;
  }

  /* Ticks along the date axis: the finest of 1, 2, 3, 6, 12, 24 or 60 months that leaves room for the labels. */
  function dateTicks(t0, t1, room) {
    const a = new Date(t0), b = new Date(t1);
    for (const step of [1, 2, 3, 6, 12, 24, 60]) {
      const out = []; let y = a.getUTCFullYear(), m = step >= 12 ? 0 : Math.floor(a.getUTCMonth() / step) * step; if (step >= 12) y = Math.ceil(y / (step / 12)) * (step / 12);
      for (let t = Date.UTC(y, m, 1); t <= t1; m += step, t = Date.UTC(y + Math.floor(m / 12), m % 12, 1)) if (t >= t0) out.push(t);
      if (out.length <= room) return { ticks: out, step };
    }
    return { ticks: [t0], step: 60 };
  }
  const tickText = (t, step) => { const d = new Date(t); return step >= 12 ? String(d.getUTCFullYear()) : `${MON[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}`; };

  function niceRange(lo, hi, n = 4) {
    if (!(hi > lo)) { hi = lo + 1; }
    const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p, step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p, ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }

  /* o: { dates, close, weight, markers, asset, summary, book: { dates, long, short } | null, label }  (the shape of GET /api/ml/strategy's chart and book) */
  function tradePanel(host, o) {
    const st = host._tp || (host._tp = { i0: 0, i1: o.dates.length - 1, view: "asset", legend: false });
    if (st.n !== o.dates.length) { st.n = o.dates.length; st.i0 = 0; st.i1 = o.dates.length - 1; }
    host.replaceChildren(); host.classList.add("tp");
    const n = o.dates.length, T = o.dates.map(day);
    const draw = () => render(host, o, st, T);
    if (!n) { html("p", "hint", host, "No price history is recorded for this asset."); return; }
    metrics(host, o);
    draw();
    if (!host._tpObserver && window.ResizeObserver) {                                                   // a new width (a phone turned round, a panel opened) redraws at that width, keeping the zoom
      let last = host.clientWidth; host._tpObserver = new ResizeObserver(() => { if (host.isConnected && Math.abs(host.clientWidth - last) > 8) { last = host.clientWidth; draw(); } }); host._tpObserver.observe(host);
    }
  }

  /* The numbers above the chart. */
  function metrics(host, o) {
    const s = o.summary || {}, row = html("div", "tp-metrics", host); row.dataset.role = "trade-metrics";
    const side = s.position === null || s.position === undefined || Math.abs(s.position) < 1e-9 ? "Flat" : s.position > 0 ? "Long" : "Short";
    const chg = s.first_price ? s.last_price / s.first_price - 1 : null;
    const tile = (label, value, sub, tone) => { const t = html("div", "tp-m" + (tone ? " " + tone : ""), row); html("span", "k", t, label); html("b", "v", t, value); if (sub) html("span", "s", t, sub); };
    const tone = (v) => (v === null || v === undefined || Math.abs(v) < 1e-12 ? "" : v > 0 ? "up" : "down");
    const t = (label, value, sub, title, tn) => { tile(label, value, sub, tn); row.lastChild.title = title; };
    t("Price", px(s.last_price), s.last_date || "", `Last close${chg !== null ? `; ${spct(chg, 0)} since the first day shown` : ""}`);
    t("Position", side + (side === "Flat" ? "" : " " + pct(Math.abs(s.position))), "of the book", "The share of the book held in this asset on the last day");
    t("Book exposure", pct(s.book_gross, 0), "gross, all assets", "Sum of the absolute weights of every asset on the last day; the rest is cash");
    t("Realized P&L", spct(s.realized, 2), "of the book, closed", "Net contribution to the book of this asset's closed positions, after costs", tone(s.realized));
    t("Unrealized P&L", spct(s.unrealized, 2), "of the book, open", "Contribution to the book of the position still open", tone(s.unrealized));
    t("Strategy return", spct(s.book_return, 0), s.benchmark_return !== null && s.benchmark_return !== undefined ? `benchmark ${spct(s.benchmark_return, 0)}` : "", "Whole-book return after costs over the run, against the equal-weight benchmark", tone(s.book_return));
  }

  function render(host, o, st, T) {
    for (const n of host.querySelectorAll(":scope > :not(.tp-metrics)")) n.remove();
    const n = o.dates.length, W = Math.max(260, Math.floor(host.clientWidth || 600)), small = W < 520;
    const L = small ? 40 : 52, R = small ? 8 : 12, pw = W - L - R;
    const Hp = small ? 200 : 270, He = small ? 78 : 96, Hm = small ? 34 : 40, gap = 20, top = 8, axisH = 22;
    const yE = top + Hp + gap, yM = yE + He + axisH + 6, H = yM + Hm + 4;
    const clamp = () => { const min = Math.min(20, n - 1); st.i0 = Math.max(0, Math.min(st.i0, n - 1 - min)); st.i1 = Math.min(n - 1, Math.max(st.i1, st.i0 + min)); };
    clamp();
    const redraw = () => render(host, o, st, T);

    // ---------- the compact bar: ranges, reset, view switch, legend
    const bar = html("div", "tp-bar", host), rng = html("div", "tp-seg", bar); rng.setAttribute("role", "group"); rng.setAttribute("aria-label", "Date range");
    const win = (days) => { if (days === null) { st.i0 = 0; st.i1 = n - 1; } else { const t1 = T[n - 1], t0 = t1 - days * 86400000; st.i1 = n - 1; st.i0 = Math.max(0, T.findIndex((t) => t >= t0)); } redraw(); };
    const active = (days) => (days === null ? st.i0 === 0 && st.i1 === n - 1 : st.i1 === n - 1 && st.i0 === Math.max(0, T.findIndex((t) => t >= T[n - 1] - days * 86400000)));
    for (const [text, days] of [["1M", 30], ["3M", 91], ["6M", 182], ["1Y", 365], ["3Y", 1095], ["All", null]]) {
      const b = html("button", "", rng, text); b.type = "button"; b.setAttribute("aria-pressed", String(active(days))); b.addEventListener("click", () => win(days));
    }
    if (o.book && o.book.dates && o.book.dates.length) {
      const sw = html("div", "tp-seg", bar); sw.setAttribute("role", "group"); sw.setAttribute("aria-label", "Exposure view");
      for (const [v, text] of [["asset", o.asset], ["book", "Whole book"]]) { const b = html("button", "", sw, text); b.type = "button"; b.dataset.view = v; b.setAttribute("aria-pressed", String(st.view === v)); b.addEventListener("click", () => { st.view = v; redraw(); }); }
    }
    const reset = html("button", "tp-link", bar, "Reset zoom"); reset.type = "button"; reset.disabled = st.i0 === 0 && st.i1 === n - 1; reset.addEventListener("click", () => win(null));
    const lg = html("details", "tp-legend", bar); lg.dataset.role = "marker-legend"; lg.open = st.legend; lg.addEventListener("toggle", () => { st.legend = lg.open; });
    html("summary", "", lg, "Legend");
    const chips = html("div", "chips", lg);
    for (const [shape, color, text, hollow] of [["up", "--pos", "Buy", false], ["up", "--pos", "Add", true], ["down", "--neg", "Sell", false], ["down", "--neg", "Trim", true], ["down", "--s5", "Short", false], ["up", "--s2", "Cover", false], ["cross", "--ink", "Exit (stop, target, time)", false]]) {
      const c = html("span", "chip", chips), i = html("i", "", c, shape === "cross" ? "✕" : hollow ? (shape === "up" ? "△" : "▽") : shape === "up" ? "▲" : "▼"); i.style.color = css(color); c.appendChild(document.createTextNode(" " + text));
    }
    html("span", "chip", chips, "Filled = open or close · outline = add or trim · a numbered mark merges several adds or trims");

    // ---------- the window and what is visible
    const i0 = st.i0, i1 = st.i1, span = Math.max(1, i1 - i0), X = (i) => L + ((i - i0) / span) * pw, vis = i1 - i0 + 1;
    const close = o.close, weight = o.weight;
    let lo = Infinity, hi = -Infinity;
    for (let i = i0; i <= i1; i++) { const v = close[i]; if (v !== null && v !== undefined && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad;
    const Yp = (v) => top + Hp - ((v - lo) / (hi - lo)) * Hp;

    // the exposure being shown: this asset's position, or the whole book's long and short sides, aligned to the price dates
    let longS, shortS;
    if (st.view === "book" && o.book && o.book.dates.length) {
      const at = new Map(o.book.dates.map((d, k) => [d, k]));
      longS = o.dates.map((d) => (at.has(d) ? o.book.long[at.get(d)] : null)); shortS = o.dates.map((d) => (at.has(d) ? o.book.short[at.get(d)] : null));
    } else { longS = weight.map((w) => (w === null || w === undefined ? null : Math.max(0, w))); shortS = weight.map((w) => (w === null || w === undefined ? null : Math.min(0, w))); }
    let eHi = 0.05, eLo = 0;
    for (let i = i0; i <= i1; i++) { if (longS[i] > eHi) eHi = longS[i]; if (shortS[i] < eLo) eLo = shortS[i]; }
    eHi *= 1.12; eLo *= 1.12; if (eLo > -0.02) eLo = Math.min(eLo, 0);
    const Ye = (v) => yE + He - ((v - eLo) / (eHi - eLo)) * He;

    const svg = el("svg", { class: "chart tp-svg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `${o.asset}: price with trades, and position, from ${o.dates[i0]} to ${o.dates[i1]}` }, null);
    svg.style.touchAction = "pan-y"; svg.style.maxWidth = "100%"; host.appendChild(svg);
    const clipP = "tpc" + Math.random().toString(36).slice(2, 7), clipE = clipP + "e";
    const defs = el("defs", {}, svg);
    el("clipPath", { id: clipP }, defs).appendChild(el("rect", { x: L, y: top - 2, width: pw, height: Hp + 4 }));
    el("clipPath", { id: clipE }, defs).appendChild(el("rect", { x: L, y: yE, width: pw, height: He }));

    // grid and axes
    const grid = css("--grid"), muted = css("--muted"), fs = small ? 10 : 11;
    for (const v of niceRange(lo, hi, small ? 3 : 4)) { const y = Yp(v); if (y < top || y > top + Hp) continue; el("line", { x1: L, x2: L + pw, y1: y, y2: y, stroke: grid, "stroke-width": 1 }, svg); el("text", { x: L - 6, y: y + 3.5, "text-anchor": "end", fill: muted, "font-size": fs }, svg, px(v)); }
    el("text", { x: L, y: top + 9, fill: muted, "font-size": fs, "font-weight": 600 }, svg).textContent = "";                    // (the readout is written here by the crosshair)
    const ticks = dateTicks(T[i0], T[i1], Math.max(2, Math.floor(pw / 64)));
    for (const t of ticks.ticks) {
      const k = T.findIndex((u) => u >= t); if (k < 0) continue; const x = X(k); if (x < L || x > L + pw) continue;
      el("line", { x1: x, x2: x, y1: top, y2: yE + He, stroke: grid, "stroke-width": 1, "stroke-dasharray": "2 4" }, svg); el("text", { x, y: yE + He + 15, "text-anchor": "middle", fill: muted, "font-size": fs }, svg, tickText(t, ticks.step));
    }

    // price: min and max of each pixel column when there are more points than pixels
    const pts = [];
    if (vis > pw * 1.5) {
      const per = vis / pw;
      for (let c = 0; c < pw; c++) {
        const a = Math.floor(i0 + c * per), b = Math.min(i1, Math.floor(i0 + (c + 1) * per)); let mn = Infinity, mx = -Infinity, imn = a, imx = a;
        for (let i = a; i <= Math.max(a, b); i++) { const v = close[i]; if (v === null || v === undefined || !isFinite(v)) continue; if (v < mn) { mn = v; imn = i; } if (v > mx) { mx = v; imx = i; } }
        if (mn === Infinity) continue; (imn <= imx ? [imn, imx] : [imx, imn]).forEach((i) => pts.push(`${X(i).toFixed(1)},${Yp(close[i]).toFixed(1)}`));
      }
    } else for (let i = i0; i <= i1; i++) { const v = close[i]; if (v !== null && v !== undefined && isFinite(v)) pts.push(`${X(i).toFixed(1)},${Yp(v).toFixed(1)}`); }
    const gP = el("g", { "clip-path": `url(#${clipP})` }, svg);
    el("polyline", { points: pts.join(" "), fill: "none", stroke: css("--ink"), "stroke-opacity": 0.78, "stroke-width": 1.5, "stroke-linejoin": "round" }, gP);

    // trades: openings, closings, flips and exits one by one; routine adds and trims merged only where they would overlap
    const idx = new Map(o.dates.map((d, k) => [d, k])), drawn = [];
    const inWin = o.markers.map((m) => ({ m, k: idx.get(m.date) })).filter((e) => e.k !== undefined && e.k >= i0 && e.k <= i1 && e.m.price !== null);
    const solo = inWin.filter((e) => !ROUTINE(e.m)), routine = inWin.filter((e) => ROUTINE(e.m)), bins = new Map(), cell = small ? 12 : 10;
    for (const e of routine) { const key = Math.round(X(e.k) / cell) + (e.m.side === "BUY" ? "b" : "s"); (bins.get(key) || bins.set(key, []).get(key)).push(e); }
    for (const e of solo) drawn.push({ x: X(e.k), y: Yp(e.m.price), items: [e.m], look: look(e.m) });
    for (const g of bins.values()) { const mid = g[Math.floor(g.length / 2)]; drawn.push({ x: g.length === 1 ? X(g[0].k) : g.reduce((a, e) => a + X(e.k), 0) / g.length, y: Yp(g.length === 1 ? g[0].m.price : mid.m.price), items: g.map((e) => e.m), look: look(g[0].m) }); }
    const gM = el("g", { "clip-path": `url(#${clipP})`, "pointer-events": "none" }, svg), r = small ? 5 : 5.5;
    for (const d of drawn) {
      const g = el("g", { class: "event" }, gM), c = css(d.look.color), many = d.items.length > 1;
      const first = d.items[0], text = many ? `${d.items.length} ${first.side === "BUY" ? "adds" : "trims"} (${first.date} to ${d.items[d.items.length - 1].date})` : `${first.date}  ${d.look.label}  ${spct(first.weight_change, 1)} of the book at ${px(first.price)}${first.reason ? "\n" + first.reason : ""}`;
      el("title", {}, g, text);
      const p = el("path", { d: shapePath(d.look.shape, d.x, d.y, many ? r + 1.5 : r), fill: d.look.hollow && !many ? css("--surface") : d.look.shape === "cross" ? "none" : c, stroke: c, "stroke-width": d.look.shape === "cross" ? 2 : 1.4, "stroke-linejoin": "round" }, g);
      if (many) el("text", { x: d.x, y: d.y + (d.look.shape === "down" ? -r - 4 : r + 11), "text-anchor": "middle", fill: c, "font-size": fs - 1, "font-weight": 700 }, g, String(d.items.length));
      d.node = p;
    }

    // position: a step area, long up and short down, drawn from its change points
    const gE = el("g", { "clip-path": `url(#${clipE})` }, svg), zero = Ye(0);
    el("line", { x1: L, x2: L + pw, y1: zero, y2: zero, stroke: css("--axis"), "stroke-width": 1 }, svg);
    for (const [series, color, sign] of [[longS, css("--pos"), 1], [shortS, css("--neg"), -1]]) {
      let path = "", prev = null, started = false;
      const point = (i, v) => { const x = X(i).toFixed(1), y = Ye(v).toFixed(1); if (!started) { path += `M${x},${zero.toFixed(1)}L${x},${y}`; started = true; } else path += `L${x},${Ye(prev).toFixed(1)}L${x},${y}`; prev = v; };
      for (let i = i0; i <= i1; i++) { const v = series[i]; if (v === null || v === undefined) continue; if (!started || Math.abs(v - prev) > 1e-9 || i === i1) point(i, v); }
      if (started) { path += `L${X(i1).toFixed(1)},${Ye(prev).toFixed(1)}L${X(i1).toFixed(1)},${zero.toFixed(1)}Z`; el("path", { d: path, fill: color, "fill-opacity": 0.28, stroke: color, "stroke-width": 1.2, "stroke-linejoin": "round" }, gE); }
    }
    for (const v of eLo < -0.001 ? [eHi / 1.12, 0, eLo / 1.12] : [eHi / 1.12, 0]) { const y = Ye(v); if (y < yE || y > yE + He) continue; el("text", { x: L - 6, y: y + 3.5, "text-anchor": "end", fill: muted, "font-size": fs }, svg, v === 0 ? "0" : pct(v, 0)); }
    el("text", { x: L, y: yE - 6, fill: muted, "font-size": fs }, svg, (st.view === "book" ? "Book exposure" : `${o.asset} position`) + (small ? ", % of book" : ", % of the book: long up, short down"));

    // overview strip: the whole history with the window on it; drag the window to pan, its edges to resize, anywhere else to recentre
    const gO = el("g", {}, svg), oy = (v, a, b) => yM + Hm - ((v - a) / (b - a || 1)) * (Hm - 4) - 2;
    let omin = Infinity, omax = -Infinity; for (const v of close) if (v !== null && isFinite(v)) { if (v < omin) omin = v; if (v > omax) omax = v; }
    const op = []; const perO = n / pw;
    for (let c = 0; c < pw; c++) { const a = Math.floor(c * perO), b = Math.min(n - 1, Math.floor((c + 1) * perO)); let m = -Infinity; for (let i = a; i <= Math.max(a, b); i++) if (close[i] > m) m = close[i]; if (isFinite(m)) op.push(`${(L + c).toFixed(1)},${oy(m, omin, omax).toFixed(1)}`); }
    el("rect", { x: L, y: yM, width: pw, height: Hm, fill: css("--ghost"), rx: 4 }, gO);
    el("polyline", { points: op.join(" "), fill: "none", stroke: muted, "stroke-width": 1, "stroke-opacity": 0.8 }, gO);
    const ox = (i) => L + (i / Math.max(1, n - 1)) * pw, wx0 = ox(i0), wx1 = ox(i1);
    el("rect", { x: wx0, y: yM, width: Math.max(4, wx1 - wx0), height: Hm, fill: css("--accent"), "fill-opacity": 0.16, stroke: css("--accent"), "stroke-width": 1.2, rx: 4, class: "tp-window" }, gO);
    for (const x of [wx0, wx1]) el("rect", { x: x - 3, y: yM + Hm / 2 - 8, width: 6, height: 16, rx: 3, fill: css("--accent") }, gO);
    const ov = el("rect", { x: L, y: yM - 4, width: pw, height: Hm + 8, fill: "transparent", style: "cursor:grab;touch-action:none" }, svg);
    let drag = null;
    const idxAt = (e, a, w) => { const b = svg.getBoundingClientRect(); return Math.round(((e.clientX - b.left) * (W / b.width) - L) / pw * (n - 1)); };
    ov.addEventListener("pointerdown", (e) => {
      const i = idxAt(e), edge = Math.max(3, Math.round((n - 1) * 9 / pw));
      drag = Math.abs(i - i0) <= edge ? { mode: "l" } : Math.abs(i - i1) <= edge ? { mode: "r" } : i > i0 && i < i1 ? { mode: "pan", off: i - i0, len: i1 - i0 } : { mode: "pan", off: Math.round((i1 - i0) / 2), len: i1 - i0 };
      ov.setPointerCapture(e.pointerId); moveOv(e);
    });
    const moveOv = (e) => {
      if (!drag) return; const i = Math.max(0, Math.min(n - 1, idxAt(e)));
      if (drag.mode === "l") st.i0 = Math.min(i, st.i1 - 5); else if (drag.mode === "r") st.i1 = Math.max(i, st.i0 + 5); else { st.i0 = Math.max(0, Math.min(n - 1 - drag.len, i - drag.off)); st.i1 = st.i0 + drag.len; }
      clearTimeout(host._tpT); host._tpT = setTimeout(redraw, 0);
    };
    ov.addEventListener("pointermove", moveOv);
    const end = () => { drag = null; }; ov.addEventListener("pointerup", end); ov.addEventListener("pointercancel", end);

    // crosshair across both panels, the readout, the trade tooltip, and a drag to zoom (mouse)
    const line = el("line", { y1: top, y2: yE + He, stroke: css("--ink-2"), "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden", "pointer-events": "none" }, svg);
    const dot = el("circle", { r: 3.5, fill: css("--ink"), visibility: "hidden", "pointer-events": "none" }, svg);
    const readout = el("text", { x: L + 4, y: top + 11, fill: css("--ink"), "font-size": fs + 1, "font-weight": 600, "pointer-events": "none" }, svg);
    const brush = el("rect", { y: top, height: Hp + gap + He, fill: css("--accent"), "fill-opacity": 0.14, stroke: css("--accent"), visibility: "hidden", "pointer-events": "none" }, svg);
    const tip = host.querySelector(":scope > .tip") || html("div", "tip tp-tip", host); tip.hidden = true;
    const plot = el("rect", { x: L, y: top, width: pw, height: Hp + gap + He, fill: "transparent", style: "cursor:crosshair" }, svg);
    const svgX = (e) => { const b = svg.getBoundingClientRect(); return (e.clientX - b.left) * (W / b.width); };
    const idxOf = (x) => Math.max(i0, Math.min(i1, Math.round(i0 + ((x - L) / pw) * span)));
    let brushing = null;
    const hover = (e) => {
      const x = svgX(e), i = idxOf(x), cx = X(i), v = close[i], tone = weight[i] > 1e-9 ? "long" : weight[i] < -1e-9 ? "short" : "flat";
      line.setAttribute("x1", cx); line.setAttribute("x2", cx); line.setAttribute("visibility", "visible");
      if (v !== null && v !== undefined) { dot.setAttribute("cx", cx); dot.setAttribute("cy", Yp(v)); dot.setAttribute("visibility", "visible"); }
      readout.textContent = `${o.dates[i]}   ${px(v)}   ${tone === "flat" ? "flat" : tone + " " + pct(Math.abs(weight[i]))}`;
      let near = null, best = (small ? 16 : 12);
      for (const d of drawn) { const dist = Math.hypot(d.x - cx, 0) + Math.abs(d.y - (e.clientY - svg.getBoundingClientRect().top) * (H / svg.getBoundingClientRect().height)) * 0.5; if (dist < best) { best = dist; near = d; } }
      if (!near) { tip.hidden = true; return; }
      tip.replaceChildren(); const lk = near.look;
      if (near.items.length > 1) {
        const buys = near.items.filter((m) => m.side === "BUY").length; html("b", "", tip, `${near.items.length} ${buys === near.items.length ? "adds" : buys === 0 ? "trims" : "trades"}`); html("div", "m", tip, `${near.items[0].date} to ${near.items[near.items.length - 1].date}`);
        for (const m of near.items.slice(0, 6)) html("div", "m", tip, `${m.date}  ${look(m).label}  ${spct(m.weight_change, 1)} at ${px(m.price)}`); if (near.items.length > 6) html("div", "m", tip, `… and ${near.items.length - 6} more: zoom in to see each`);
      } else {
        const m = near.items[0]; html("b", "", tip, lk.label); const add = (k, v) => { if (v === null || v === undefined || v === "") return; const r2 = html("div", "m", tip); html("span", "k", r2, k + " "); r2.appendChild(document.createTextNode(v)); };
        add("Executed", `${m.date} at ${px(m.price)}`); add("Decided", m.decision_date); add("Quantity", `${spct(m.weight_change, 2)} of the book, ${pct(m.weight_after, 1)} after`); add("Cost", m.cost !== null ? pct(m.cost, 3) : "");
        if (m.pnl_return !== null && m.pnl_return !== undefined) add("P&L", `${spct(m.pnl_return, 1)} on the position, ${spct(m.pnl_contribution, 2)} of the book`);
        add("Score rank", m.rank_pct !== null && m.rank_pct !== undefined ? pct(m.rank_pct, 0) : ""); add("Why", m.reason);
      }
      const hb = host.getBoundingClientRect(), sb = svg.getBoundingClientRect(); tip.hidden = false;
      const left = sb.left - hb.left + near.x * (sb.width / W), tw = tip.offsetWidth, th = tip.offsetHeight;
      tip.style.left = Math.max(4, Math.min(host.clientWidth - tw - 4, left + 12 > host.clientWidth - tw ? left - tw - 12 : left + 12)) + "px"; tip.style.top = Math.max(4, sb.top - hb.top + near.y * (sb.height / H) - th - 10) + "px";
    };
    const hide = () => { line.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); readout.textContent = ""; tip.hidden = true; };
    plot.addEventListener("pointermove", (e) => {
      if (brushing) { const x = Math.max(L, Math.min(L + pw, svgX(e))); brush.setAttribute("x", Math.min(brushing.x, x)); brush.setAttribute("width", Math.abs(x - brushing.x)); brush.setAttribute("visibility", "visible"); return; }
      hover(e);
    });
    plot.addEventListener("pointerdown", (e) => { if (e.pointerType === "mouse") { brushing = { x: svgX(e) }; plot.setPointerCapture(e.pointerId); } else hover(e); });
    plot.addEventListener("pointerup", (e) => {
      if (!brushing) return; const x = svgX(e), a = Math.min(brushing.x, x), b = Math.max(brushing.x, x); brushing = null; brush.setAttribute("visibility", "hidden");
      if (b - a > 8) { const j0 = idxOf(a), j1 = idxOf(b); if (j1 - j0 >= 5) { st.i0 = j0; st.i1 = j1; redraw(); } }
    });
    plot.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse" && !brushing) hide(); });
    plot.addEventListener("dblclick", () => win(null));
    plot.addEventListener("wheel", (e) => {                                                                    // Ctrl (or a trackpad pinch) with the wheel zooms around the pointer; a plain wheel scrolls the page
      if (!e.ctrlKey) return; e.preventDefault(); const c = idxOf(svgX(e)), f = e.deltaY > 0 ? 1.25 : 0.8, a = Math.round(c - (c - i0) * f), b = Math.round(c + (i1 - c) * f);
      st.i0 = Math.max(0, a); st.i1 = Math.min(n - 1, Math.max(b, st.i0 + 5)); redraw();
    }, { passive: false });
  }

  C.tradePanel = tradePanel;
})();
