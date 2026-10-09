/* Small SVG chart kit: line, area, columns, horizontal bars, heat map. No dependencies; labels are inserted with textContent only. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function html(tag, cls, parent, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const fmt = {
    pct: (v, d = 1) => (v * 100).toFixed(d) + "%",
    pct0: (v) => (v * 100).toFixed(0) + "%",
    x: (v) => v.toFixed(2),
    num: (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)),
    signed: (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d),
    spct: (v, d = 1) => (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%",
  };

  function niceTicks(lo, hi, n = 5) {
    if (!(hi > lo)) { hi = lo + 1; }
    const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
    const start = Math.ceil(lo / step - 1e-9) * step, out = [];
    for (let v = start; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function wholeTicks(lo, hi, n) {                                                   // ticks on whole numbers (slices of a day, years)
    const step = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000].find((v) => (hi - lo) / v <= n) || 1000, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
    return out;
  }
  function logTicks(lo, hi) {
    const base = [1, 1.5, 2, 3, 4, 5, 7.5], out = [];
    for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++) for (const b of base) { const v = b * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
    return out.length > 7 ? out.filter((_, i) => i % 2 === 0) : out;
  }
  function yearTicks(t0, t1, maxTicks) {
    const y0 = new Date(t0).getUTCFullYear(), y1 = new Date(t1).getUTCFullYear(), spanY = y1 - y0;
    const step = [1, 2, 5, 10].find((s) => spanY / s <= maxTicks) || 10, out = [];
    for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) { const t = Date.UTC(y, 0, 1); if (t >= t0 && t <= t1) out.push({ t, label: String(y) }); }
    if (out.length < 2) out.push({ t: t0, label: new Date(t0).toISOString().slice(0, 7) });
    return out;
  }
  /* The drawing width is the host's width. Read it BEFORE the host is emptied: measuring an emptied host forces a layout while the page is shorter, and the browser answers by clamping or
     re-anchoring the scroll position, so the page would jump every time a chart is redrawn (a Chart/Table or Linear/Log toggle). */
  function width(host) { return Math.max(280, Math.floor(host.getBoundingClientRect().width || 600)); }
  function tooltip(host) {
    let tip = host.querySelector(":scope > .tip");
    if (!tip) { tip = html("div", "tip", host); tip.hidden = true; }
    return tip;
  }
  function place(tip, host, x, y) {
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight, hw = host.clientWidth;
    tip.style.left = Math.max(4, Math.min(hw - w - 4, x + 14 > hw - w ? x - w - 14 : x + 14)) + "px";
    tip.style.top = Math.max(4, y - h - 8) + "px";
  }
  function legend(host, items, boxes) {
    const lg = html("div", "legend" + (boxes ? " boxes" : ""), host);
    for (const s of items) { const span = html("span", "", lg); const i = html("i", s.box ? "box" : "", span); i.style.setProperty("--c", s.color); if (s.box) i.style.opacity = String(Math.min(1, (s.opacity || 0.14) * 2.5)); span.appendChild(document.createTextNode(s.name)); }
    return lg;
  }

  /* ---------- line / area over dates, or over numbers ----------
     Dates by default (o.dates). With o.xs (numbers, ascending) the horizontal axis is numeric: o.xFormat writes a tick or a hover title, o.xLabel names the axis, o.xWhole puts the ticks on whole numbers. A series may be drawn dashed
     (dash), with a dot at every point (dots, dotSize; a series of one point shows as a dot, and pointLabel names it) and may skip over missing values instead of breaking (connect). o.bands shades the space between two
     arrays ({ lo, hi, color, name }), which is how a range of outcomes (a fan of wealth paths) is drawn behind its median. o.tipExtra(i) adds [label, text] rows to the hover. */
  function line(host, o) {
    const W = width(host);
    host.replaceChildren();
    const numeric = Array.isArray(o.xs), dates = numeric ? o.xs.slice() : o.dates.map((d) => Date.parse(d)), H = o.height || 280;
    const bands = o.bands || [], m = { l: 46, r: 14, t: 8, b: o.xLabel ? 40 : 24 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
    const legendItems = o.series.concat(bands.filter((b) => b.name).map((b) => ({ name: b.name, color: b.color, box: true, opacity: b.opacity })));
    if (o.legend !== false && legendItems.length > 1) legend(host, legendItems);
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "line chart" }, host);
    const log = !!o.log;
    const f = (v) => (log ? Math.log(v) : v);
    let lo = Infinity, hi = -Infinity;
    for (const s of o.series) for (const v of s.values) if (v !== null && isFinite(v) && (!log || v > 0)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (const b of bands) for (const arr of [b.lo, b.hi]) for (const v of arr) if (v !== null && isFinite(v) && (!log || v > 0)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (o.include !== undefined) { lo = Math.min(lo, o.include); hi = Math.max(hi, o.include); }
    if (!isFinite(lo)) { el("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, svg, "No data"); return; }
    const pad = (hi - lo) * 0.05 || 0.5;
    if (!log) { lo -= pad; hi += pad; if (o.floor !== undefined) lo = Math.min(lo, o.floor); if (o.ceil !== undefined) hi = Math.max(hi, o.ceil); }
    const ticks = log ? logTicks(lo, hi) : niceTicks(lo, hi, 5);
    const flo = log ? f(lo * 0.98) : lo, fhi = log ? f(hi * 1.02) : hi;
    const t0 = dates[0], t1 = dates[dates.length - 1];
    const X = (t) => m.l + ((t - t0) / (t1 - t0 || 1)) * iw, Y = (v) => m.t + ih - ((f(v) - flo) / (fhi - flo || 1)) * ih;
    for (const v of ticks) { const y = Y(v); if (y < m.t - 1 || y > m.t + ih + 1) continue; el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" }, svg); el("text", { x: m.l - 6, y: y + 4, "text-anchor": "end" }, svg, (o.yFormat || fmt.num)(v)); }
    const tickY = o.xLabel ? H - 22 : H - 6;
    if (numeric) for (const v of (o.xWhole ? wholeTicks : niceTicks)(t0, t1, Math.max(2, Math.floor(iw / 80)))) el("text", { x: X(v), y: tickY, "text-anchor": "middle" }, svg, (o.xFormat || fmt.num)(v));
    else for (const t of yearTicks(t0, t1, Math.floor(iw / 70))) el("text", { x: X(t.t), y: tickY, "text-anchor": "middle" }, svg, t.label);
    if (o.xLabel) el("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle", class: "ink" }, svg, o.xLabel);
    if (o.baseline !== undefined) el("line", { x1: m.l, x2: W - m.r, y1: Y(o.baseline), y2: Y(o.baseline), class: "axis" }, svg);
    el("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: "axis" }, svg);
    for (const b of bands) {
      const keep = []; b.hi.forEach((v, i) => { const l = b.lo[i]; if (v !== null && l !== null && isFinite(v) && isFinite(l) && (!log || (v > 0 && l > 0))) keep.push(i); });
      if (keep.length < 2) continue;
      const top = keep.map((i, k) => (k ? "L" : "M") + X(dates[i]).toFixed(1) + " " + Y(b.hi[i]).toFixed(1)).join(""), bottom = keep.slice().reverse().map((i) => "L" + X(dates[i]).toFixed(1) + " " + Y(b.lo[i]).toFixed(1)).join("");
      el("path", { d: top + bottom + "Z", fill: b.color, "fill-opacity": b.opacity || 0.14, stroke: "none" }, svg);
    }
    const paths = [];
    o.series.forEach((s, k) => {
      let d = "", pen = false, lastY = null, lastX = null, first = null;
      const marks = [];
      s.values.forEach((v, i) => {
        if (v === null || !isFinite(v) || (log && v <= 0)) { if (!s.connect) pen = false; return; }
        const x = X(dates[i]), y = Y(v);
        d += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1); pen = true; lastX = x; lastY = y; if (first === null) first = [x, y];
        if (s.dots) marks.push([x, y]);
      });
      if (o.area && k === 0 && first) el("path", { d: d + `L${lastX.toFixed(1)} ${(o.areaBase !== undefined ? Y(o.areaBase) : m.t + ih).toFixed(1)}L${first[0].toFixed(1)} ${(o.areaBase !== undefined ? Y(o.areaBase) : m.t + ih).toFixed(1)}Z`, fill: s.color, "fill-opacity": 0.1, stroke: "none" }, svg);
      const stroke = { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round" };
      if (s.dash) stroke["stroke-dasharray"] = s.dash;
      if (d) el("path", stroke, svg);
      for (const [x, y] of marks) el("circle", { cx: x, cy: y, r: s.dotSize || 3.5, fill: s.color, stroke: css("--surface"), "stroke-width": 1.5 }, svg);
      if (s.pointLabel && marks.length) {
        const [x, y] = marks[marks.length - 1], flip = x > W - m.r - 120;
        el("text", { x: flip ? x - 9 : x + 9, y: y + 4, "text-anchor": flip ? "end" : "start", class: "ink" }, svg, s.pointLabel);
      }
      paths.push({ lastX, lastY });
    });
    // direct end labels only when they do not collide; the legend carries identity otherwise
    if (o.series.length > 1 && o.endLabels !== false) {
      const ys = paths.map((p) => p.lastY).filter((y) => y !== null).sort((a, b) => a - b);
      if (ys.every((y, i) => i === 0 || y - ys[i - 1] >= 14) && W > 520) o.series.forEach((s, k) => { if (paths[k].lastY !== null) el("text", { x: W - m.r - 4, y: paths[k].lastY - 6, "text-anchor": "end", class: "ink" }, svg, s.name); });
    }
    // hover: crosshair snaps to the nearest date; one tooltip lists every series
    const cross = el("line", { y1: m.t, y2: m.t + ih, class: "axis", visibility: "hidden" }, svg);
    const dots = o.series.map((s) => { const g = el("g", { visibility: "hidden" }, svg); el("circle", { r: 5, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, g); return g; });
    const tip = tooltip(host);
    const overlay = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", tabindex: 0 }, svg);
    const show = (clientX) => {
      const rect = svg.getBoundingClientRect(), px = ((clientX - rect.left) / rect.width) * W, t = t0 + ((px - m.l) / iw) * (t1 - t0);
      let lo2 = 0, hi2 = dates.length - 1;
      while (hi2 - lo2 > 1) { const mid = (lo2 + hi2) >> 1; if (dates[mid] < t) lo2 = mid; else hi2 = mid; }
      const i = Math.abs(dates[lo2] - t) < Math.abs(dates[hi2] - t) ? lo2 : hi2, x = X(dates[i]);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("visibility", "visible");
      tip.replaceChildren(); html("div", "t", tip, numeric ? (o.tipTitle ? o.tipTitle(i) : (o.xFormat || fmt.num)(dates[i])) : o.dates[i]);
      o.series.forEach((s, k) => {
        const v = s.values[i], g = dots[k];
        if (v === null || !isFinite(v)) { g.setAttribute("visibility", "hidden"); return; }
        g.setAttribute("visibility", "visible"); g.setAttribute("transform", `translate(${x},${Y(v)})`);
        const r = html("div", "r", tip), kk = html("span", "k", r), sw = html("i", "", kk); sw.style.setProperty("--c", s.color); kk.appendChild(document.createTextNode(s.name));
        html("b", "", r, (o.tipFormat || o.yFormat || fmt.num)(v));
      });
      for (const b of bands) {
        if (!b.name || b.lo[i] === null || b.hi[i] === null || !isFinite(b.lo[i]) || !isFinite(b.hi[i])) continue;
        const r = html("div", "r", tip), kk = html("span", "k", r), sw = html("i", "", kk); sw.style.setProperty("--c", b.color); kk.appendChild(document.createTextNode(b.name));
        html("b", "", r, `${(o.tipFormat || o.yFormat || fmt.num)(b.lo[i])} to ${(o.tipFormat || o.yFormat || fmt.num)(b.hi[i])}`);
      }
      if (o.tipExtra) for (const [label, text] of o.tipExtra(i)) { const r = html("div", "r", tip); html("span", "k", r, label); html("b", "", r, text); }
      place(tip, host, (x / W) * rect.width, m.t + 20);
    };
    const hide = () => { cross.setAttribute("visibility", "hidden"); dots.forEach((g) => g.setAttribute("visibility", "hidden")); tip.hidden = true; };
    overlay.addEventListener("pointermove", (e) => show(e.clientX));
    overlay.addEventListener("pointerleave", hide);
    overlay.addEventListener("focus", () => show(svg.getBoundingClientRect().left + svg.getBoundingClientRect().width * 0.5));
    overlay.addEventListener("blur", hide);
  }

  /* ---------- columns by label (annual returns) ---------- */
  function columns(host, o) {
    const W = width(host);
    host.replaceChildren();
    const H = o.height || 240, m = { l: 46, r: 10, t: 14, b: 26 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "column chart" }, host);
    const vals = o.values.map((v) => (v === null ? 0 : v)), lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), pad = (hi - lo) * 0.1 || 0.1;
    const ticks = niceTicks(lo - (lo < 0 ? pad : 0), hi + pad, 5), min = Math.min(ticks[0], lo), max = Math.max(ticks[ticks.length - 1], hi);
    const Y = (v) => m.t + ih - ((v - min) / (max - min || 1)) * ih;
    for (const v of ticks) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg); el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, svg, o.yFormat(v)); }
    el("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "axis" }, svg);
    const slot = iw / o.labels.length, bw = Math.min(24, slot - 2), tip = tooltip(host);
    const best = vals.indexOf(Math.max(...vals)), worst = vals.indexOf(Math.min(...vals));
    o.labels.forEach((lab, i) => {
      const v = vals[i], cx = m.l + slot * (i + 0.5), x = cx - bw / 2, y0 = Y(0), y1 = Y(v), top = Math.min(y0, y1), h = Math.max(1, Math.abs(y1 - y0)), r = Math.min(4, bw / 2, h);
      const up = v >= 0, color = o.colors ? o.colors(v, i) : css("--s1");
      const d = up ? `M${x},${top + h}V${top + r}Q${x},${top} ${x + r},${top}H${x + bw - r}Q${x + bw},${top} ${x + bw},${top + r}V${top + h}Z`
                   : `M${x},${top}V${top + h - r}Q${x},${top + h} ${x + r},${top + h}H${x + bw - r}Q${x + bw},${top + h} ${x + bw},${top + h - r}V${top}Z`;
      const bar = el("path", { d, fill: color, opacity: o.faded && o.faded[i] ? 0.55 : 1 }, svg);
      if (iw / o.labels.length > 28 || i % 2 === 0) el("text", { x: cx, y: H - 8, "text-anchor": "middle" }, svg, String(lab));
      if (i === best || i === worst) el("text", { x: cx, y: up ? top - 4 : Y(0) - 4, "text-anchor": "middle", class: "ink" }, svg, o.yFormat(v));
      const hit = el("rect", { x: cx - slot / 2, y: m.t, width: slot, height: ih, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", () => { bar.setAttribute("opacity", 0.8); tip.replaceChildren(); html("div", "t", tip, o.tipLabel ? o.tipLabel(i) : String(lab)); const r2 = html("div", "r", tip); html("span", "k", r2, o.name || "Value"); html("b", "", r2, (o.tipFormat || o.yFormat)(v)); place(tip, host, cx, Y(Math.max(v, 0))); });
      hit.addEventListener("pointerleave", () => { bar.setAttribute("opacity", 1); tip.hidden = true; });
    });
  }

  /* ---------- horizontal bars (weights, contribution, scores) ---------- */
  function hbars(host, o) {
    const W = width(host);
    host.replaceChildren();
    const rows = o.rows, rowH = 22, H = rows.length * rowH + 22, m = { l: o.labelWidth || 74, r: o.valueWidth || 54, t: 4, b: 18 }, iw = W - m.l - m.r;
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "bar chart" }, host);
    const vals = rows.map((r) => r.value), lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), ticks = niceTicks(lo, hi, 4);
    const min = Math.min(lo, ticks[0]), max = Math.max(hi, ticks[ticks.length - 1]), X = (v) => m.l + ((v - min) / (max - min || 1)) * iw, tip = tooltip(host);
    for (const v of ticks) { el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, class: "grid" }, svg); el("text", { x: X(v), y: H - 4, "text-anchor": "middle" }, svg, o.format(v)); }
    el("line", { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, class: "axis" }, svg);
    rows.forEach((r, i) => {
      const y = m.t + i * rowH + 3, h = 14, x0 = X(0), x1 = X(r.value), left = Math.min(x0, x1), w = Math.max(1, Math.abs(x1 - x0)), rad = Math.min(4, w);
      const pos = r.value >= 0, color = o.color ? o.color(r.value, r) : pos ? css("--pos") : css("--neg");
      const d = pos ? `M${left},${y}H${left + w - rad}Q${left + w},${y} ${left + w},${y + rad}V${y + h - rad}Q${left + w},${y + h} ${left + w - rad},${y + h}H${left}Z`
                    : `M${left + w},${y}H${left + rad}Q${left},${y} ${left},${y + rad}V${y + h - rad}Q${left},${y + h} ${left + rad},${y + h}H${left + w}Z`;
      const bar = el("path", { d, fill: color }, svg);
      el("text", { x: m.l - 8, y: y + 11, "text-anchor": "end", class: "ink" }, svg, r.label);
      el("text", { x: pos ? left + w + 5 : x0 + 5, y: y + 11, "text-anchor": "start", class: "ink" }, svg, o.format(r.value));
      const hit = el("rect", { x: 0, y: y - 3, width: W, height: rowH, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (e) => { bar.setAttribute("opacity", 0.75); tip.replaceChildren(); html("div", "t", tip, r.label); const r2 = html("div", "r", tip); html("span", "k", r2, o.name || "Value"); html("b", "", r2, (o.tipFormat || o.format)(r.value)); const rect = svg.getBoundingClientRect(); place(tip, host, e.clientX - rect.left, y + 6); });
      hit.addEventListener("pointerleave", () => { bar.setAttribute("opacity", 1); tip.hidden = true; });
    });
  }

  /* ---------- month x year heat map (diverging, gray midpoint) ---------- */
  function heatmap(host, o) {
    const W = width(host);
    host.replaceChildren();
    const years = [...new Set(o.cells.map((c) => c.year))].sort(), m = { l: 38, r: 8, t: 18, b: 4 };
    const cw = Math.max(18, Math.min(56, (W - m.l - m.r) / 12)), ch = 20, H = m.t + years.length * ch + m.b, svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "heat map" }, host);
    const mx = Math.max(0.02, ...o.cells.map((c) => Math.abs(c.value))) , cap = Math.min(mx, 0.12), pos = css("--pos"), neg = css("--neg"), mid = css("--mid"), tip = tooltip(host);
    const mix = (a, b, t) => { const pa = rgb(a), pb = rgb(b); return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",")})`; };
    const names = "JFMAMJJASOND".split(""), by = {};
    o.cells.forEach((c) => { by[c.year + "-" + c.month] = c; });
    names.forEach((n, i) => el("text", { x: m.l + cw * (i + 0.5), y: 11, "text-anchor": "middle" }, svg, n));
    years.forEach((y, r) => {
      el("text", { x: m.l - 6, y: m.t + r * ch + 14, "text-anchor": "end" }, svg, String(y));
      for (let k = 1; k <= 12; k++) {
        const c = by[y + "-" + k]; if (!c) continue;
        const t = Math.min(1, Math.abs(c.value) / cap), fill = c.value >= 0 ? mix(mid, pos, t) : mix(mid, neg, t);
        const rect = el("rect", { x: m.l + cw * (k - 1) + 1, y: m.t + r * ch + 1, width: cw - 2, height: ch - 2, rx: 3, fill }, svg);
        if (cw >= 34) { const dark = t > 0.55; el("text", { x: m.l + cw * (k - 0.5), y: m.t + r * ch + 14, "text-anchor": "middle", style: `fill:${dark ? "#fff" : css("--ink-2")};font-size:10px` }, svg, (c.value * 100).toFixed(1)); }
        rect.addEventListener("pointermove", (e) => { rect.setAttribute("stroke", css("--ink")); tip.replaceChildren(); html("div", "t", tip, `${y}-${String(k).padStart(2, "0")}`); const r2 = html("div", "r", tip); html("span", "k", r2, "Return"); html("b", "", r2, fmt.spct(c.value)); const b = svg.getBoundingClientRect(); place(tip, host, e.clientX - b.left, m.t + r * ch); });
        rect.addEventListener("pointerleave", () => { rect.removeAttribute("stroke"); tip.hidden = true; });
      }
    });
  }
  function rgb(c) {
    if (c.startsWith("#")) { const n = parseInt(c.length === 4 ? c.replace(/./g, (x, i) => (i ? x + x : x)).slice(1) : c.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
    return (c.match(/\d+/g) || [128, 128, 128]).slice(0, 3).map(Number);
  }

  window.Charts = { line, columns, hbars, heatmap, fmt, html, el, niceTicks };
})();
