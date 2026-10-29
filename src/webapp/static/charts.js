/* Small SVG chart kit: line, area, columns, mirrored columns, horizontal bars (and waterfalls), month heat map, matrix, scatter, histogram. No dependencies; labels are inserted with textContent only. */
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
  function logTicks(lo, hi) {                                                        // the finest set of 1-1.5-2-3-4-5-7.5 steps that does not crowd the axis, then coarser ones, down to the decades
    let out = [];
    for (const base of [[1, 1.5, 2, 3, 4, 5, 7.5], [1, 2, 5], [1, 3], [1]]) {
      out = [];
      for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++) for (const b of base) { const v = b * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
      if (out.length <= 8) break;
    }
    return out;
  }
  function monthTicks(t0, t1, maxTicks) {                                            // a window of under two years: the first of a month, every 1, 2, 3 or 6 months
    const a = new Date(t0), b = new Date(t1), months = (b.getUTCFullYear() - a.getUTCFullYear()) * 12 + b.getUTCMonth() - a.getUTCMonth() + 1;
    const step = [1, 2, 3, 6].find((s) => months / s <= maxTicks) || 6, out = [];
    for (let k = 0; k <= months; k++) {
      const t = Date.UTC(a.getUTCFullYear(), a.getUTCMonth() + k, 1), d = new Date(t), m = d.getUTCMonth();
      if (t >= t0 && t <= t1 && m % step === 0) out.push({ t, label: m === 0 ? String(d.getUTCFullYear()) : d.toISOString().slice(0, 7) });
    }
    return out;
  }
  function yearTicks(t0, t1, maxTicks) {
    if (t1 - t0 < 2 * 365.25 * 864e5) { const out = monthTicks(t0, t1, maxTicks); if (out.length >= 2) return out; }
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
     arrays ({ lo, hi, color, name }), which is how a range of outcomes (a fan of wealth paths) is drawn behind its median. o.tipExtra(i) adds [label, text] rows to the hover.
   o.regions shades stretches of the horizontal axis behind the lines ({ from, to, name }, in the axis' own units: date strings, or numbers) and names them above the plot: the training, validation, test and
   forward sections of a machine-learning equity curve. o.vlines marks single positions ({ x, label }): today's price and the break-evens on a payoff chart. o.signFill shades the first series against
   the baseline, blue where it is above and red where it is below (profit and loss). o.ylim = [low, high] fixes the vertical range and cuts every line and band off at the edges of the plot, so that one series that runs
   far beyond the others (a structure with unlimited profit drawn beside one without) does not flatten them. */
  let uid = 0;
  function line(host, o) {
    const W = width(host);
    host.replaceChildren();
    const numeric = Array.isArray(o.xs), dates = numeric ? o.xs.slice() : o.dates.map((d) => Date.parse(d)), H = o.height || 280;
    const bands = o.bands || [], regions = o.regions || [], vlines = o.vlines || [], m = { l: 46, r: 14, t: regions.length || vlines.length ? 22 : 8, b: o.xLabel ? 40 : 24 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
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
    if (o.ylim) { lo = o.ylim[0]; hi = o.ylim[1]; }
    const pad = (hi - lo) * 0.05 || 0.5;
    if (!log) { lo -= pad; hi += pad; if (o.floor !== undefined) lo = Math.min(lo, o.floor); if (o.ceil !== undefined) hi = Math.max(hi, o.ceil); }
    const ticks = log ? logTicks(lo, hi) : niceTicks(lo, hi, 5);
    const flo = log ? f(lo * 0.98) : lo, fhi = log ? f(hi * 1.02) : hi;
    const t0 = dates[0], t1 = dates[dates.length - 1];
    const X = (t) => m.l + ((t - t0) / (t1 - t0 || 1)) * iw, Y = (v) => m.t + ih - ((f(v) - flo) / (fhi - flo || 1)) * ih;
    const at = (v) => (numeric ? v : Date.parse(v));
    regions.forEach((r, k) => {                                                                // stretches of the axis, drawn first so that the grid and the lines sit on top
      const x0 = Math.max(m.l, X(at(r.from))), x1 = Math.min(m.l + iw, X(at(r.to)));
      if (!(x1 > x0)) return;
      el("rect", { x: x0, y: m.t, width: x1 - x0, height: ih, fill: r.color || css("--ink"), "fill-opacity": r.opacity || (k % 2 ? 0.07 : 0.025) }, svg);
      if (r.name && x1 - x0 > 8 * r.name.length * 0.8) el("text", { x: (x0 + x1) / 2, y: m.t - 8, "text-anchor": "middle", class: "ink" }, svg, r.name);
    });
    for (const v of ticks) { const y = Y(v); if (y < m.t - 1 || y > m.t + ih + 1) continue; el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" }, svg); el("text", { x: m.l - 6, y: y + 4, "text-anchor": "end" }, svg, (o.yFormat || fmt.num)(v)); }
    const tickY = o.xLabel ? H - 22 : H - 6;
    if (numeric) for (const v of (o.xWhole ? wholeTicks : niceTicks)(t0, t1, Math.max(2, Math.floor(iw / 80)))) el("text", { x: X(v), y: tickY, "text-anchor": "middle" }, svg, (o.xFormat || fmt.num)(v));
    else for (const t of yearTicks(t0, t1, Math.floor(iw / 70))) el("text", { x: X(t.t), y: tickY, "text-anchor": "middle" }, svg, t.label);
    if (o.xLabel) el("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle", class: "ink" }, svg, o.xLabel);
    if (o.baseline !== undefined) el("line", { x1: m.l, x2: W - m.r, y1: Y(o.baseline), y2: Y(o.baseline), class: "axis" }, svg);
    el("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: "axis" }, svg);
    let lastLabelX = -1e9;
    for (const v of vlines.slice().sort((a, b) => at(a.x) - at(b.x))) {                        // reference positions: today's price, a break-even; labelled left to right, only where there is room
      const xv = at(v.x);
      if (!(xv >= t0 && xv <= t1)) continue;
      const px = X(xv);
      el("line", { x1: px, x2: px, y1: m.t, y2: m.t + ih, class: "ref" }, svg);
      if (v.label && px - lastLabelX > 8 * v.label.length * 0.8 + 6) { el("text", { x: px, y: m.t - 8, "text-anchor": "middle", class: "ink" }, svg, v.label); lastLabelX = px; }
    }
    const clipId = o.ylim ? "yc" + ++uid : null, clip = clipId ? { "clip-path": `url(#${clipId})` } : {};
    if (clipId) el("rect", { x: m.l, y: m.t, width: iw, height: ih }, el("clipPath", { id: clipId }, el("defs", {}, svg)));
    for (const b of bands) {
      const keep = []; b.hi.forEach((v, i) => { const l = b.lo[i]; if (v !== null && l !== null && isFinite(v) && isFinite(l) && (!log || (v > 0 && l > 0))) keep.push(i); });
      if (keep.length < 2) continue;
      const top = keep.map((i, k) => (k ? "L" : "M") + X(dates[i]).toFixed(1) + " " + Y(b.hi[i]).toFixed(1)).join(""), bottom = keep.slice().reverse().map((i) => "L" + X(dates[i]).toFixed(1) + " " + Y(b.lo[i]).toFixed(1)).join("");
      el("path", { d: top + bottom + "Z", fill: b.color, "fill-opacity": b.opacity || 0.14, stroke: "none", ...clip }, svg);
    }
    const paths = [];
    o.series.forEach((s, k) => {
      let d = "", pen = false, lastY = null, lastX = null, first = null;
      const marks = [];
      s.values.forEach((v, i) => {
        if (v === null || !isFinite(v) || (log && v <= 0)) { if (!s.connect) pen = false; return; }
        const x = X(dates[i]), y = Y(v);
        d += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1); pen = true; lastX = x; lastY = y; if (first === null) first = [x, y];
        if (s.dots && (!o.ylim || (v >= o.ylim[0] && v <= o.ylim[1]))) marks.push([x, y]);
      });
      if (o.signFill && k === 0 && first) {
        const yz = Y(o.baseline !== undefined ? o.baseline : 0), id = "sf" + ++uid, defs = el("defs", {}, svg), area = d + `L${lastX.toFixed(1)} ${yz.toFixed(1)}L${first[0].toFixed(1)} ${yz.toFixed(1)}Z`;
        el("rect", { x: m.l, y: m.t, width: iw, height: Math.max(0, yz - m.t) }, el("clipPath", { id: id + "u" }, defs));
        el("rect", { x: m.l, y: yz, width: iw, height: Math.max(0, m.t + ih - yz) }, el("clipPath", { id: id + "d" }, defs));
        el("path", { d: area, fill: css("--pos"), "fill-opacity": 0.14, stroke: "none", "clip-path": `url(#${id}u)` }, svg);
        el("path", { d: area, fill: css("--neg"), "fill-opacity": 0.14, stroke: "none", "clip-path": `url(#${id}d)` }, svg);
      }
      if (o.area && k === 0 && first) el("path", { d: d + `L${lastX.toFixed(1)} ${(o.areaBase !== undefined ? Y(o.areaBase) : m.t + ih).toFixed(1)}L${first[0].toFixed(1)} ${(o.areaBase !== undefined ? Y(o.areaBase) : m.t + ih).toFixed(1)}Z`, fill: s.color, "fill-opacity": 0.1, stroke: "none", ...clip }, svg);
      const stroke = { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", ...clip };
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

  /* ---------- mirrored columns: two measures of the same strikes or dates, one drawn up and one down from the baseline ----------
     o.labels names each column (a blank label draws nothing); o.up and o.down are arrays of non-negative numbers, the second drawn below the baseline (calls above and puts below); o.upName and o.downName name them
     in the legend and the hover and o.upColor and o.downColor colour them; o.yFormat writes an axis value (without a sign) and o.tipFormat a hover value; o.tipTitle(i) names a column in the hover;
     o.mark is the index of a column to point out (the one nearest the price) and o.markLabel names it. */
  function mirror(host, o) {
    const W = width(host);
    host.replaceChildren();
    const clean = (a) => a.map((v) => (v === null || !isFinite(v) ? 0 : Math.max(v, 0))), up = clean(o.up), down = clean(o.down), upColor = o.upColor || css("--s1"), downColor = o.downColor || css("--s2");
    legend(host, [{ name: o.upName, color: upColor, box: true, opacity: 0.4 }, { name: o.downName, color: downColor, box: true, opacity: 0.4 }]);
    const H = o.height || 260, m = { l: 56, r: 10, t: 16, b: 26 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "mirrored column chart" }, host);
    const top = Math.max(0, ...up), bottom = Math.max(0, ...down);
    if (!(top || bottom)) { el("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, svg, "No data"); return; }
    const ticks = niceTicks(-bottom * 1.08, top * 1.08, 5), min = Math.min(ticks[0], -bottom), max = Math.max(ticks[ticks.length - 1], top), fy = o.yFormat || fmt.num;
    const Y = (v) => m.t + ih - ((v - min) / (max - min || 1)) * ih;
    for (const v of ticks) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg); el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, svg, fy(Math.abs(v))); }
    el("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "axis" }, svg);
    const n = o.labels.length, slot = iw / n, bw = Math.max(1, Math.min(24, slot - 1.5)), tip = tooltip(host), y0 = Y(0);
    if (o.mark !== undefined && o.mark >= 0 && o.mark < n) {                                  // the column to point out: a reference line behind the bars
      const px = m.l + slot * (o.mark + 0.5);
      el("line", { x1: px, x2: px, y1: m.t, y2: m.t + ih, class: "ref" }, svg);
      if (o.markLabel) el("text", { x: px, y: m.t - 5, "text-anchor": "middle", class: "ink" }, svg, o.markLabel);
    }
    o.labels.forEach((lab, i) => {
      const cx = m.l + slot * (i + 0.5), x = cx - bw / 2, bars = [];
      if (up[i] > 0) { const hgt = Math.max(1, y0 - Y(up[i])); bars.push(el("rect", { x, y: y0 - hgt, width: bw, height: hgt, fill: upColor, rx: Math.min(2, bw / 2) }, svg)); }
      if (down[i] > 0) { const hgt = Math.max(1, Y(-down[i]) - y0); bars.push(el("rect", { x, y: y0, width: bw, height: hgt, fill: downColor, rx: Math.min(2, bw / 2) }, svg)); }
      if (lab !== "" && lab !== null && lab !== undefined) el("text", { x: cx, y: H - 8, "text-anchor": "middle" }, svg, String(lab));
      const hit = el("rect", { x: cx - slot / 2, y: m.t, width: slot, height: ih, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", () => {
        bars.forEach((b) => b.setAttribute("opacity", 0.75)); tip.replaceChildren(); html("div", "t", tip, o.tipTitle ? o.tipTitle(i) : String(lab));
        for (const [name, v, c] of [[o.upName, up[i], upColor], [o.downName, down[i], downColor]]) { const r = html("div", "r", tip), kk = html("span", "k", r), sw = html("i", "", kk); sw.style.setProperty("--c", c); kk.appendChild(document.createTextNode(name)); html("b", "", r, (o.tipFormat || fy)(v)); }
        place(tip, host, cx, m.t + 20);
      });
      hit.addEventListener("pointerleave", () => { bars.forEach((b) => b.setAttribute("opacity", 1)); tip.hidden = true; });
    });
  }

  /* ---------- horizontal bars (weights, contribution, scores) ---------- */
  function hbars(host, o) {
    const W = width(host);
    host.replaceChildren();
    const rows = o.rows, rowH = 22, H = rows.length * rowH + 22, m = { l: o.labelWidth || 74, r: o.valueWidth || 54, t: 4, b: 18 }, iw = W - m.l - m.r;
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "bar chart" }, host);
    const vals = rows.map((r) => r.value), starts = rows.map((r) => r.from || 0), lo = Math.min(0, ...vals, ...starts), hi = Math.max(0, ...vals, ...starts), ticks = niceTicks(lo, hi, 4);
    const min = Math.min(lo, ticks[0]), max = Math.max(hi, ticks[ticks.length - 1]), X = (v) => m.l + ((v - min) / (max - min || 1)) * iw, tip = tooltip(host);
    for (const v of ticks) { el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, class: "grid" }, svg); el("text", { x: X(v), y: H - 4, "text-anchor": "middle" }, svg, o.format(v)); }
    el("line", { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, class: "axis" }, svg);
    rows.forEach((r, i) => {
      const y = m.t + i * rowH + 3, h = 14, x0 = X(r.from || 0), x1 = X(r.value), left = Math.min(x0, x1), w = Math.max(1, Math.abs(x1 - x0)), rad = Math.min(4, w);   // a row may start away from zero (r.from): a waterfall
      const pos = r.value >= (r.from || 0), color = o.color ? o.color(r.value, r) : pos ? css("--pos") : css("--neg");
      const d = pos ? `M${left},${y}H${left + w - rad}Q${left + w},${y} ${left + w},${y + rad}V${y + h - rad}Q${left + w},${y + h} ${left + w - rad},${y + h}H${left}Z`
                    : `M${left + w},${y}H${left + rad}Q${left},${y} ${left},${y + rad}V${y + h - rad}Q${left},${y + h} ${left + rad},${y + h}H${left + w}Z`;
      const bar = el("path", { d, fill: color }, svg);
      el("text", { x: m.l - 8, y: y + 11, "text-anchor": "end", class: "ink" }, svg, r.label);
      el("text", { x: Math.max(x0, x1) + 5, y: y + 11, "text-anchor": "start", class: "ink" }, svg, r.text !== undefined ? r.text : o.format(r.value));
      const hit = el("rect", { x: 0, y: y - 3, width: W, height: rowH, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (e) => { bar.setAttribute("opacity", 0.75); tip.replaceChildren(); html("div", "t", tip, r.label); const r2 = html("div", "r", tip); html("span", "k", r2, o.name || "Value"); html("b", "", r2, r.text !== undefined ? r.text : (o.tipFormat || o.format)(r.value));
        for (const [k, v] of r.tip || []) { const r3 = html("div", "r", tip); html("span", "k", r3, k); html("b", "", r3, v); } const rect = svg.getBoundingClientRect(); place(tip, host, e.clientX - rect.left, y + 6); });
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

  /* ---------- matrix: rows by columns, each cell shaded by its value ----------
     o.rows and o.cols name the axes, o.values[r][c] is a number or null. Sequential (one hue, light to dark) by default; o.diverging shades around o.mid (0 unless given) from red below to blue above, with the
     neutral gray at the middle. o.min and o.max fix the scale (default: the data's; a diverging scale is symmetric). o.format writes a value, o.cellText prints it inside a cell wide enough to hold it,
     o.rowTitle and o.colTitle name the axes, o.mark(r, c) outlines the cell to point out (the worst stress, the at-the-money strike), o.tipRows(r, c) adds [label, text] rows to the hover. */
  function matrix(host, o) {
    const W = width(host);
    host.replaceChildren();
    const rows = o.rows, cols = o.cols, vals = o.values, fm = o.format || fmt.num, flat = vals.flat().filter((v) => v !== null && v !== undefined && isFinite(v));
    const labW = o.labelWidth || Math.min(150, Math.max(46, Math.round(Math.max(...rows.map((r) => String(r).length), 2) * 6.4) + 14)), m = { l: labW, r: 8, t: o.colTitle ? 34 : 20, b: 36 };
    const cw = (W - m.l - m.r) / cols.length, ch = o.cellHeight || 22, H = m.t + rows.length * ch + m.b;
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "matrix" }, host);
    if (!flat.length) { el("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, svg, "No data"); return; }
    const div = !!o.diverging, mid = o.mid !== undefined ? o.mid : 0;
    let lo = o.min !== undefined ? o.min : Math.min(...flat), hi = o.max !== undefined ? o.max : Math.max(...flat);
    if (div) { const span = Math.max(Math.abs(lo - mid), Math.abs(hi - mid), 1e-12); lo = mid - span; hi = mid + span; } else if (!(hi > lo)) hi = lo + 1;
    const mix = (a, b, t) => { const pa = rgb(a), pb = rgb(b); return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",")})`; };
    const seq = [css("--scale-lo"), css("--scale-mid"), css("--scale-hi")], neg = css("--neg"), pos = css("--pos"), neutral = css("--mid");
    const colour = (v) => {
      if (div) { const t = Math.min(1, Math.abs(v - mid) / (hi - mid)); return v >= mid ? mix(neutral, pos, t) : mix(neutral, neg, t); }
      const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo))); return t < 0.5 ? mix(seq[0], seq[1], t * 2) : mix(seq[1], seq[2], (t - 0.5) * 2);
    };
    const dark = (c) => { const [r, g, b] = rgb(c); return 0.299 * r + 0.587 * g + 0.114 * b < 140; };
    const every = Math.max(1, Math.ceil(46 / Math.max(cw, 1))), tip = tooltip(host);
    cols.forEach((c, j) => { if (j % every === 0) el("text", { x: m.l + cw * (j + 0.5), y: m.t - 6, "text-anchor": "middle" }, svg, String(c)); });
    if (o.colTitle) el("text", { x: m.l + (cw * cols.length) / 2, y: 11, "text-anchor": "middle", class: "ink" }, svg, o.colTitle);
    if (o.rowTitle) el("text", { x: 4, y: 11, "text-anchor": "start", class: "ink" }, svg, o.rowTitle);
    rows.forEach((rl, i) => {
      el("text", { x: m.l - 6, y: m.t + i * ch + ch / 2 + 4, "text-anchor": "end" }, svg, String(rl));
      cols.forEach((cl, j) => {
        const v = vals[i][j], x = m.l + cw * j, y = m.t + i * ch;
        if (v === null || v === undefined || !isFinite(v)) { el("rect", { x: x + 1, y: y + 1, width: Math.max(1, cw - 2), height: ch - 2, rx: 2, fill: css("--ghost") }, svg); return; }
        const fill = colour(v), cell = el("rect", { x: x + 1, y: y + 1, width: Math.max(1, cw - 2), height: ch - 2, rx: 2, fill }, svg);
        if (o.mark && o.mark(i, j)) el("rect", { x: x + 0.5, y: y + 0.5, width: Math.max(1, cw - 1), height: ch - 1, rx: 2, fill: "none", stroke: css("--ink"), "stroke-width": 1.5, "pointer-events": "none" }, svg);
        const text = o.cellText === false ? null : fm(v);
        if (text && cw >= 7 * text.length + 6) el("text", { x: x + cw / 2, y: y + ch / 2 + 4, "text-anchor": "middle", style: `fill:${dark(fill) ? "#fff" : css("--ink-2")};font-size:10px` }, svg, text);
        cell.addEventListener("pointermove", (e) => {
          cell.setAttribute("stroke", css("--ink")); tip.replaceChildren(); html("div", "t", tip, `${rl} · ${cl}`);
          const r2 = html("div", "r", tip); html("span", "k", r2, o.name || "Value"); html("b", "", r2, (o.tipFormat || fm)(v));
          for (const [k, t] of (o.tipRows ? o.tipRows(i, j) : [])) { const r3 = html("div", "r", tip); html("span", "k", r3, k); html("b", "", r3, t); }
          const b = svg.getBoundingClientRect(); place(tip, host, e.clientX - b.left, y);
        });
        cell.addEventListener("pointerleave", () => { cell.removeAttribute("stroke"); tip.hidden = true; });
      });
    });
    // the scale: a gradient bar under the grid with its two ends and, for a diverging scale, the middle
    const id = "mx" + ++uid, gy = H - 22, gx = m.l, gw = Math.min(160, Math.max(80, W - m.l - m.r)), grad = el("linearGradient", { id }, el("defs", {}, svg));
    for (const t of [0, 0.5, 1]) el("stop", { offset: `${t * 100}%`, "stop-color": colour(lo + t * (hi - lo)) }, grad);
    el("rect", { x: gx, y: gy, width: gw, height: 8, rx: 2, fill: `url(#${id})` }, svg);
    el("text", { x: gx, y: gy + 20, "text-anchor": "start" }, svg, fm(lo)); el("text", { x: gx + gw, y: gy + 20, "text-anchor": "end" }, svg, fm(hi));
    if (div) el("text", { x: gx + gw / 2, y: gy + 20, "text-anchor": "middle" }, svg, fm(mid));
  }

  /* ---------- scatter: points with their own labels ----------
     o.points is [{ x, y, label, color, r, tip }] (tip: [[label, text]]). The pointer finds the NEAREST point, so no point has to be hit dead centre. Labels are written beside the points where they do not collide
     (o.labels: "auto", "all" or "none"); o.legend is [{ name, color }] for points coloured by group; o.xLabel and o.yLabel name the axes; o.xZero and o.yZero draw the zero lines. o.lines is
     [{ x1, y1, x2, y2, color, dash, name }]: straight lines in data units (a regression fit, the diagonal of a Q-Q plot), clipped to the plot and named in the legend. */
  function scatter(host, o) {
    const W = width(host);
    host.replaceChildren();
    const H = o.height || 320, pts = o.points.filter((p) => isFinite(p.x) && isFinite(p.y)), m = { l: 50, r: 16, t: o.yLabel ? 26 : 10, b: o.xLabel ? 42 : 26 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
    if (o.legend && o.legend.length > 1) legend(host, o.legend.map((g) => ({ name: g.name, color: g.color, box: true, opacity: 0.4 })));
    const named = (o.lines || []).filter((l) => l.name);
    if (named.length) legend(host, named.map((l) => ({ name: l.name, color: l.color || css("--s2") })));
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "scatter plot" }, host);
    if (!pts.length) { el("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, svg, "No data"); return; }
    const span = (arr, pad) => { let lo = Math.min(...arr), hi = Math.max(...arr); const p = (hi - lo) * pad || 0.5; return [lo - p, hi + p]; };
    const [x0, x1] = o.xRange || span(pts.map((p) => p.x), 0.08), [y0, y1] = o.yRange || span(pts.map((p) => p.y), 0.1);
    const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw, Y = (v) => m.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih;
    for (const v of niceTicks(y0, y1, 5)) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg); el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, svg, (o.yFormat || fmt.num)(v)); }
    for (const v of niceTicks(x0, x1, Math.max(2, Math.floor(iw / 90)))) el("text", { x: X(v), y: H - (o.xLabel ? 24 : 8), "text-anchor": "middle" }, svg, (o.xFormat || fmt.num)(v));
    el("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: "axis" }, svg);
    if (o.xZero && x0 < 0 && x1 > 0) el("line", { x1: X(0), x2: X(0), y1: m.t, y2: m.t + ih, class: "axis" }, svg);
    if (o.yZero && y0 < 0 && y1 > 0) el("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "axis" }, svg);
    if (o.xLabel) el("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle", class: "ink" }, svg, o.xLabel);
    if (o.yLabel) el("text", { x: 4, y: 12, "text-anchor": "start", class: "ink" }, svg, o.yLabel);
    if ((o.lines || []).length) {
      const clip = "sc" + ++uid; el("rect", { x: m.l, y: m.t, width: iw, height: ih }, el("clipPath", { id: clip }, el("defs", {}, svg)));
      for (const l of o.lines) el("line", { x1: X(l.x1), y1: Y(l.y1), x2: X(l.x2), y2: Y(l.y2), stroke: l.color || css("--s2"), "stroke-width": 2, "stroke-dasharray": l.dash || "", "clip-path": `url(#${clip})`, "pointer-events": "none" }, svg);
    }
    const dots = pts.map((p) => el("circle", { cx: X(p.x), cy: Y(p.y), r: Math.max(4, p.r || 4.5), fill: p.color || css("--s1"), stroke: css("--surface"), "stroke-width": 2 }, svg));
    const placed = [];
    if (o.labels !== "none") pts.forEach((p, i) => {
      if (!p.label) return;
      const w = 6.2 * String(p.label).length, px = X(p.x), py = Y(p.y), right = px + 9 + w < W - m.r, lx = right ? px + 9 : px - 9 - w, box = [lx, py - 7, lx + w, py + 5];
      if (o.labels !== "all" && placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) return;
      placed.push(box); el("text", { x: right ? px + 9 : px - 9, y: py + 4, "text-anchor": right ? "start" : "end", class: "ink" }, svg, String(p.label));
    });
    const tip = tooltip(host), ring = el("circle", { r: 8, fill: "none", stroke: css("--ink"), "stroke-width": 1.5, visibility: "hidden", "pointer-events": "none" }, svg);
    const overlay = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", tabindex: 0 }, svg);
    const show = (i) => {
      const p = pts[i]; ring.setAttribute("cx", X(p.x)); ring.setAttribute("cy", Y(p.y)); ring.setAttribute("r", Math.max(4, p.r || 4.5) + 4); ring.setAttribute("visibility", "visible");
      tip.replaceChildren(); html("div", "t", tip, p.label ? String(p.label) : "");
      for (const [k, v] of p.tip || [[o.xLabel || "x", (o.xFormat || fmt.num)(p.x)], [o.yLabel || "y", (o.yFormat || fmt.num)(p.y)]]) { const r = html("div", "r", tip); html("span", "k", r, k); html("b", "", r, v); }
      const b = svg.getBoundingClientRect(); place(tip, host, (X(p.x) / W) * b.width, (Y(p.y) / H) * b.height);
    };
    const hide = () => { ring.setAttribute("visibility", "hidden"); tip.hidden = true; };
    let focus = 0;
    overlay.addEventListener("pointermove", (e) => {
      const b = svg.getBoundingClientRect(), px = ((e.clientX - b.left) / b.width) * W, py = ((e.clientY - b.top) / b.height) * H;
      let best = -1, bd = 28 * 28; pts.forEach((p, i) => { const d = (X(p.x) - px) ** 2 + (Y(p.y) - py) ** 2; if (d < bd) { bd = d; best = i; } });
      if (best < 0) hide(); else show(best);
    });
    overlay.addEventListener("pointerleave", hide);
    overlay.addEventListener("focus", () => show(focus));
    overlay.addEventListener("blur", hide);
    overlay.addEventListener("keydown", (e) => { if (e.key === "ArrowRight" || e.key === "ArrowDown") focus = (focus + 1) % pts.length; else if (e.key === "ArrowLeft" || e.key === "ArrowUp") focus = (focus + pts.length - 1) % pts.length; else return; e.preventDefault(); show(focus); });
    void dots;
  }

  /* ---------- histogram: the shape of a distribution, one or several samples on the same axis ----------
     o.series is [{ name, edges, counts, color }]: bin edges (one more than the counts) and how many observations fell in each. Samples of different sizes are compared as the SHARE of their observations per
     bin width, drawn as stepped lines (the first with a faint fill). o.markers is [{ x, label }] for a reference position (zero). With o.unit the vertical scale is the share of observations per
     band of that width (a unit of 0.001 on a return axis: the share of days per 0.1% band) and o.yFormat writes it; without, it is the density (share per unit of x). */
  function histogram(host, o) {
    const W = width(host);
    host.replaceChildren();
    const H = o.height || 220, m = { l: 46, r: 14, t: 8, b: 26 }, iw = W - m.l - m.r, ih = H - m.t - m.b, series = o.series.filter((s) => s.counts && s.counts.length);
    if (series.length > 1) legend(host, series.map((s) => ({ name: s.name, color: s.color })));
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "histogram" }, host);
    if (!series.length) { el("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, svg, "No data"); return; }
    const unit = o.unit || 1, dens = series.map((s) => { const n = s.counts.reduce((a, b) => a + b, 0) || 1; return s.counts.map((c, i) => (c / n / (s.edges[i + 1] - s.edges[i] || 1)) * unit); });
    const x0 = Math.min(...series.map((s) => s.edges[0])), x1 = Math.max(...series.map((s) => s.edges[s.edges.length - 1])), top = Math.max(...dens.flat()) * 1.08 || 1;
    const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw, Y = (v) => m.t + ih - (v / top) * ih;
    for (const v of niceTicks(0, top, 4)) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg); el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, svg, o.yFormat ? o.yFormat(v) : v >= 10 ? v.toFixed(0) : v.toPrecision(2)); }
    for (const v of niceTicks(x0, x1, Math.max(2, Math.floor(iw / 80)))) el("text", { x: X(v), y: H - 8, "text-anchor": "middle" }, svg, (o.xFormat || fmt.num)(v));
    el("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: "axis" }, svg);
    for (const mk of o.markers || []) if (mk.x >= x0 && mk.x <= x1) { el("line", { x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: m.t + ih, class: "ref" }, svg); if (mk.label) el("text", { x: X(mk.x) + 4, y: m.t + 10, class: "ink" }, svg, mk.label); }
    series.forEach((s, k) => {
      let d = ""; s.counts.forEach((_, i) => { const a = X(s.edges[i]), b = X(s.edges[i + 1]), y = Y(dens[k][i]); d += (i ? `L${a.toFixed(1)} ${y.toFixed(1)}` : `M${a.toFixed(1)} ${y.toFixed(1)}`) + `L${b.toFixed(1)} ${y.toFixed(1)}`; });
      if (k === 0) el("path", { d: d + `L${X(s.edges[s.edges.length - 1]).toFixed(1)} ${Y(0)}L${X(s.edges[0]).toFixed(1)} ${Y(0)}Z`, fill: s.color, "fill-opacity": 0.1, stroke: "none" }, svg);
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }, svg);
    });
    const cross = el("line", { y1: m.t, y2: m.t + ih, class: "axis", visibility: "hidden" }, svg), tip = tooltip(host), overlay = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", tabindex: 0 }, svg);
    const show = (clientX) => {
      const b = svg.getBoundingClientRect(), px = ((clientX - b.left) / b.width) * W, v = x0 + ((px - m.l) / iw) * (x1 - x0);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible"); tip.replaceChildren(); html("div", "t", tip, (o.xFormat || fmt.num)(v));
      series.forEach((s, k) => {
        const i = s.edges.findIndex((e, j) => j < s.counts.length && v >= e && v < s.edges[j + 1]); if (i < 0) return;
        const n = s.counts.reduce((a, c) => a + c, 0) || 1, r = html("div", "r", tip), kk = html("span", "k", r), sw = html("i", "", kk); sw.style.setProperty("--c", s.color); kk.appendChild(document.createTextNode(s.name));
        html("b", "", r, `${((s.counts[i] / n) * 100).toFixed(1)}% of ${n.toLocaleString("en-US")}`);
      });
      place(tip, host, px, m.t + 20);
    };
    const hide = () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; };
    overlay.addEventListener("pointermove", (e) => show(e.clientX)); overlay.addEventListener("pointerleave", hide);
    overlay.addEventListener("focus", () => show(svg.getBoundingClientRect().left + svg.getBoundingClientRect().width * 0.5)); overlay.addEventListener("blur", hide);
  }

  window.Charts = { line, columns, mirror, hbars, heatmap, matrix, scatter, histogram, fmt, html, el, niceTicks };
})();
