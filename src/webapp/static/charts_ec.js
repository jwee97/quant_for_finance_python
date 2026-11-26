/* The chart kit on Apache ECharts (vendor/echarts.min.js, with its LICENSE and NOTICE beside it).

   ``window.Charts`` keeps its API: every page still calls ``C.line(host, options)`` and so on, with the same options and the same numbers. This file replaces a chart type with an ECharts version when ECharts is
   loaded AND the options asked for are ones the port covers; anything else falls through to the SVG code in charts.js, unchanged, so no chart can lose a feature by being ported. ECharts draws with its SVG
   renderer (the root <svg> carries the class "chart"), so "Save as SVG/PNG", the Table view and the CSV export keep working; the SVG kit is also the fallback if the library cannot load.
   ``localStorage.setItem("ql.charts", "svg")`` (and a reload) switches back to the SVG kit for every chart.

   What ECharts brings: a time axis with sensible ticks, a zoom slider and two-finger pinch on touch screens, a crosshair that snaps to the data, tooltips kept inside the screen, line sampling for long series,
   and cursors and zoom that follow each other across charts of one group (``o.sync``). */
(() => {
  "use strict";
  const C = window.Charts, ec = window.echarts;
  if (!C || !ec) return;                                                                            // no library: the SVG kit stays in charge
  const legacy = { ...C };
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const on = () => { try { return localStorage.getItem("ql.charts") !== "svg"; } catch (e) { return true; } };
  const live = new Set();                                                                           // every chart on the page, so that a theme change can repaint them and a removed one can be freed

  function sweep() { for (const h of live) if (!h.isConnected) { try { h._ec && h._ec.dispose(); } catch (e) { /* already gone */ } h._ro && h._ro.disconnect(); live.delete(h); } }
  function mount(host, height, label) {
    sweep();
    if (host._ec) { try { host._ec.dispose(); } catch (e) { /* already gone */ } host._ro && host._ro.disconnect(); live.delete(host); host._ec = null; }
    host.replaceChildren(); host._ecWide = false;
    return host;
  }
  /* The area between a line and a constant base (the area under a growth curve, a profit and loss fill), as stacked series so that it follows zoom and the axis like any other series: a floor at the base,
     and on it the part of the line above the base (and, in a second stack, the part below it). ``up`` and ``down`` are the fill colours; null leaves that side unfilled. */
  function areaSeries(xs, vals, base, up, down) {
    const pts = (f) => xs.map((x, i) => [x, vals[i] === null || vals[i] === undefined || !isFinite(vals[i]) ? null : f(vals[i])]);
    const common = { type: "line", silent: true, animation: false, showSymbol: false, tooltip: { show: false }, legendHoverLink: false, emphasis: { disabled: true }, stackStrategy: "all", z: 1 }, floor = { ...common, lineStyle: { opacity: 0 }, data: xs.map((x) => [x, base]) };
    const out = [];
    if (up) out.push({ ...floor, stack: "areaU" }, { ...common, stack: "areaU", lineStyle: { opacity: 0 }, areaStyle: { color: up }, data: pts((v) => Math.max(v - base, 0)) });
    if (down) out.push({ ...floor, stack: "areaD" }, { ...common, stack: "areaD", lineStyle: { opacity: 0 }, areaStyle: { color: down }, data: pts((v) => Math.min(v - base, 0)) });
    return out;
  }
  function boxHeight(host, o) { return host.closest(".expanded") ? Math.max(o.height || 280, Math.round(window.innerHeight - 250)) : o.height || 280; }
  const theme = () => ({ ink: css("--ink"), ink2: css("--ink-2"), muted: css("--muted"), grid: css("--grid"), axis: css("--axis"), surface: css("--surface"), font: getComputedStyle(document.body).fontFamily });

  /* ---------------------------------------------------------------------------------------------- line */
  const LINE_OK = (o) => Array.isArray(o.series) && !o.events && !(o.xs && o.sync);
  function line(host, o) {
    if (!on() || !LINE_OK(o)) return legacy.line(host, o);
    mount(host);
    const numeric = Array.isArray(o.xs), n = (numeric ? o.xs : o.dates).length, H = boxHeight(host, o), fmt = C.fmt;
    const xs = numeric ? o.xs : o.dates.map((d) => Date.parse(d));
    const yf = o.yFormat || fmt.num, tf = o.tipFormat || o.yFormat || fmt.num, log = !!o.log;
    const finite = (v) => v !== null && v !== undefined && isFinite(v) && (!log || v > 0);
    const bands = o.bands || [], regions = o.regions || [], vlines = o.vlines || [];
    const zoomable = n >= 20 && o.zoom !== false;
    // a one-series chart needs no legend; the title names it
    const shown = o.series.length + bands.filter((b) => b.name).length > 1 && o.legend !== false;
    const key = o.series.map((s) => s.name).join("\u0001"); if (host._ecKey !== key) { host._ecKey = key; host._ecOff = new Set(); }
    const off = host._ecOff, toggleable = o.series.length > 1 && o.legend !== false && o.toggle !== false && !o.area;
    const bar = document.createElement("div"); bar.className = "chartbar ec-bar";
    const box = document.createElement("div"); box.className = "ec-chart"; box.style.cssText = `width:100%;height:${H}px;touch-action:pan-y`; box.setAttribute("role", "img"); box.setAttribute("aria-label", o.label || "line chart");
    let legendBox = null;
    if (shown) {
      legendBox = document.createElement("div"); legendBox.className = "legend";
      const items = o.series.map((s, k) => ({ name: s.name, color: s.color, k })).concat(bands.filter((b) => b.name).map((b) => ({ name: b.name, color: b.color, box: true, opacity: b.opacity })));
      for (const it of items) {
        const span = document.createElement("span"), i = document.createElement("i"); if (it.box) { i.className = "box"; i.style.opacity = String(Math.min(1, (it.opacity || 0.14) * 2.5)); } i.style.setProperty("--c", it.color); span.append(i, document.createTextNode(it.name));
        if (toggleable && !it.box) {
          span.setAttribute("role", "button"); span.tabIndex = 0; span.dataset.series = String(it.k); span.setAttribute("aria-pressed", String(!off.has(it.k))); span.title = off.has(it.k) ? `Show ${it.name}` : `Hide ${it.name}`; if (off.has(it.k)) span.classList.add("off");
          const flip = () => {
            if (off.has(it.k)) off.delete(it.k); else if (off.size < o.series.length - 1) off.add(it.k); else return;                         // the last visible series stays
            build(); for (const sp of legendBox.querySelectorAll("[data-series]")) { const kk = Number(sp.dataset.series), hidden = off.has(kk); sp.classList.toggle("off", hidden); sp.setAttribute("aria-pressed", String(!hidden)); sp.title = (hidden ? "Show " : "Hide ") + o.series[kk].name; }
            span.focus({ preventScroll: true });
          };
          span.addEventListener("click", flip); span.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flip(); } });
        }
        legendBox.appendChild(span);
      }
    }
    host.append(...(legendBox ? [legendBox] : []), bar, box);
    let lo = Infinity, hi = -Infinity;
    for (const s of o.series) for (const v of s.values) if (finite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (const b of bands) for (const arr of [b.lo, b.hi]) for (const v of arr) if (finite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (!isFinite(lo)) { const p = document.createElement("p"); p.className = "hint"; p.textContent = "No data"; box.replaceChildren(p); return; }
    const chart = ec.init(box, null, { renderer: "svg", width: undefined, height: H });
    host._ec = chart; live.add(host);
    if (o.sync && !numeric) chart.group = "sync:" + o.sync;

    const build = () => {
      const t = theme(), seriesData = (vals) => vals.map((v, i) => [xs[i], finite(v) ? v : null]);
      const series = [];
      o.series.forEach((s, k) => {
        const area = o.area && k === 0 && o.areaBase === undefined;
        if (k === 0 && o.area && o.areaBase !== undefined) series.push(...areaSeries(xs, s.values.map((v) => (finite(v) ? v : null)), o.areaBase, s.color + "30", s.color + "30"));
        if (k === 0 && o.signFill) { const base = o.baseline !== undefined ? o.baseline : 0, v0 = s.values.map((v) => (finite(v) ? v : null)); series.push(...areaSeries(xs, v0, base, css("--pos") + "2e", css("--neg") + "2e")); }
        const lastAt = s.pointLabel ? (() => { for (let i = s.values.length - 1; i >= 0; i--) if (finite(s.values[i])) return i; return -1; })() : -1;
        series.push({
          id: "s" + k, name: s.name, type: "line", data: lastAt < 0 ? seriesData(s.values) : seriesData(s.values).map((d, i) => (i === lastAt ? { value: d, symbol: "circle", symbolSize: 8, label: { show: true, formatter: s.pointLabel, position: "right", color: t.ink, fontSize: 11 } } : d)),
          connectNulls: !!s.connect, showSymbol: !!s.dots || lastAt >= 0, symbol: lastAt >= 0 && !s.dots ? "none" : "circle", symbolSize: (s.dotSize || 3.5) * 2, sampling: "lttb", animation: false, emphasis: { disabled: true },
          lineStyle: { color: s.color, width: s.width || 2, type: s.dash ? "dashed" : "solid" }, itemStyle: { color: s.color, borderColor: t.surface, borderWidth: 1.5 },
          areaStyle: area ? { color: new ec.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: s.color + "4d" }, { offset: 1, color: s.color + "05" }]) } : undefined,
          markLine: k === 0 && (o.baseline !== undefined || vlines.length) ? {
            silent: true, symbol: "none", animation: false, label: { color: t.ink2, fontSize: 11 },
            data: [].concat(o.baseline !== undefined ? [{ yAxis: o.baseline, lineStyle: { color: t.axis, type: "solid", width: 1 }, label: { show: false } }] : [],
              vlines.filter((v) => v.x !== undefined).map((v) => ({ xAxis: numeric ? v.x : Date.parse(v.x), lineStyle: { color: t.muted, type: "dashed", width: 1 }, label: { show: !!v.label, formatter: v.label || "", position: "end", color: t.ink2, fontSize: 11 } }))) } : undefined,
          markArea: k === 0 && regions.length ? { silent: true, animation: false, data: regions.map((r, j) => [{ xAxis: numeric ? r.from : Date.parse(r.from), name: r.name || "", itemStyle: { color: r.color || t.ink, opacity: r.opacity || (j % 2 ? 0.07 : 0.025) }, label: { show: !!r.name, color: t.ink2, fontSize: 11, position: "insideTop" } }, { xAxis: numeric ? r.to : Date.parse(r.to) }]) } : undefined,
        });
      });
      bands.forEach((b, j) => {                                                                     // a band is a transparent floor with the difference stacked on it
        const stack = "band" + j, okAt = (i) => finite(b.lo[i]) && finite(b.hi[i]);
        series.push({ name: b.name || "", type: "line", stack, data: xs.map((x, i) => [x, okAt(i) ? b.lo[i] : null]), lineStyle: { opacity: 0 }, showSymbol: false, silent: true, tooltip: { show: false }, animation: false, emphasis: { disabled: true }, legendHoverLink: false });
        series.push({ name: b.name || "", type: "line", stack, data: xs.map((x, i) => [x, okAt(i) ? b.hi[i] - b.lo[i] : null]), lineStyle: { opacity: 0 }, showSymbol: false, silent: true, tooltip: { show: false }, animation: false, emphasis: { disabled: true }, legendHoverLink: false,
          areaStyle: { color: b.color, opacity: b.opacity || 0.14 } });
      });
      const winA = Array.isArray(o.window) ? Math.max(0, Math.min(n - 2, Math.round(o.window[0]))) : 0, winB = Array.isArray(o.window) ? Math.max(winA + 1, Math.min(n - 1, Math.round(o.window[1]))) : n - 1;
      const padded = (e) => { const span = (e.max - e.min) || 1, p = span * 0.05 || 0.5; let a = e.min - p, b = e.max + p; if (o.include !== undefined) { a = Math.min(a, o.include); b = Math.max(b, o.include); } if (o.floor !== undefined) a = Math.min(a, o.floor); if (o.ceil !== undefined) b = Math.max(b, o.ceil); return [a, b]; };
      chart.setOption({
        animation: false, backgroundColor: "transparent", textStyle: { fontFamily: t.font, color: t.muted },
        grid: { left: 8, right: o.series.some((x) => x.pointLabel) ? 78 : 14, top: regions.length || vlines.length ? 24 : 10, bottom: (zoomable ? 30 : 6) + (o.xLabel ? 18 : 0), containLabel: true },
        legend: { show: false, selectedMode: "multiple", data: [...new Set(series.map((s) => s.name).filter(Boolean))], selected: Object.fromEntries(o.series.map((s, k) => [s.name, !off.has(k)])) },
        xAxis: { type: numeric ? "value" : "time", name: o.xLabel || "", nameLocation: "middle", nameGap: 26, nameTextStyle: { color: t.ink2 }, min: numeric ? "dataMin" : undefined, max: numeric ? "dataMax" : undefined, scale: numeric,
          axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, hideOverlap: true, formatter: numeric ? (v) => (o.xFormat || fmt.num)(v) : undefined }, splitLine: { show: false },
          axisPointer: { snap: true, lineStyle: { color: t.ink2, width: 1, type: "dashed" }, label: { show: false } } },
        yAxis: { type: log ? "log" : "value", scale: true, min: o.ylim ? o.ylim[0] : (e) => padded(e)[0], max: o.ylim ? o.ylim[1] : (e) => padded(e)[1], splitNumber: 5,
          axisLabel: { color: t.muted, formatter: (v) => yf(v), showMaxLabel: !log, showMinLabel: true }, axisLine: { show: false }, splitLine: { lineStyle: { color: t.grid } } },
        dataZoom: zoomable ? [
          { type: "inside", xAxisIndex: 0, filterMode: "filter", zoomOnMouseWheel: "ctrl", moveOnMouseWheel: false, moveOnMouseMove: "shift", preventDefaultMouseMove: false, startValue: xs[winA], endValue: xs[winB] },
          { type: "slider", xAxisIndex: 0, filterMode: "filter", height: 16, bottom: o.xLabel ? 20 : 4, startValue: xs[winA], endValue: xs[winB], showDetail: false, borderColor: "transparent", backgroundColor: t.grid,
            fillerColor: "rgba(57,135,229,0.18)", handleSize: "120%", handleStyle: { color: css("--accent"), borderColor: css("--accent") }, moveHandleStyle: { color: css("--accent") }, dataBackground: { lineStyle: { color: t.axis }, areaStyle: { color: "transparent" } }, textStyle: { color: t.muted } },
        ] : [],
        tooltip: { trigger: "axis", triggerOn: matchMedia("(pointer: coarse)").matches ? "click" : "mousemove|click", hideDelay: 80, confine: true, className: "tip ec-tip", appendToBody: false, backgroundColor: t.surface, borderColor: t.axis, borderWidth: 1, padding: [6, 9], textStyle: { color: t.ink, fontSize: 12 }, extraCssText: "box-shadow:0 4px 14px rgba(0,0,0,.12);border-radius:8px;position:absolute",
          axisPointer: { type: "line", snap: true, lineStyle: { color: t.ink2, width: 1, type: "dashed" } },
          formatter: (params) => {
            const mine = params.filter((p) => /^s\d+$/.test(p.seriesId || "")), have = new Set(mine.map((p) => p.seriesId)), first = mine[0]; if (!first) return "";      // (a hidden series sends no params; the helper series are not read)
            const i = xs.indexOf(first.value[0]); if (i < 0) return "";
            const div = document.createElement("div"), add = (cls, parent, text) => { const e = document.createElement(cls[0]); e.className = cls[1] || ""; if (text !== undefined) e.textContent = text; (parent || div).appendChild(e); return e; };
            add(["div", "t"], null, numeric ? (o.tipTitle ? o.tipTitle(i) : (o.xFormat || fmt.num)(xs[i])) : o.dates[i]);
            o.series.forEach((s, k) => {
              if (!have.has("s" + k)) return;                                                          // a hidden series is not read
              const v = s.values[i]; if (!finite(v)) return;
              const r = add(["div", "r"]), kk = add(["span", "k"], r), sw = add(["i"], kk); sw.style.setProperty("--c", s.color); kk.appendChild(document.createTextNode(s.name)); add(["b"], r, tf(v));
            });
            for (const b of bands) { if (!b.name || b.noTip || !finite(b.lo[i]) || !finite(b.hi[i])) continue; const r = add(["div", "r"]), kk = add(["span", "k"], r), sw = add(["i"], kk); sw.style.setProperty("--c", b.color); kk.appendChild(document.createTextNode(b.name)); add(["b"], r, `${tf(b.lo[i])} to ${tf(b.hi[i])}`); }
            if (o.tipExtra) for (const [label, text] of o.tipExtra(i)) { const r = add(["div", "r"]); add(["span", "k"], r, label); add(["b"], r, text); }
            return div;
          } },
        series,
      }, true);
      const svg = box.querySelector("svg"); if (svg) { svg.classList.add("chart"); svg.setAttribute("viewBox", `0 0 ${box.clientWidth || svg.getAttribute("width")} ${H}`); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", o.label || "line chart"); }
      const tipEl = box.querySelector(".ec-tip");
      if (tipEl && !tipEl._watched) {                                                                // the SVG kit's contract: a tooltip that is not shown carries `hidden`
        tipEl._watched = true; const sync = () => { const gone = tipEl.style.display === "none" || tipEl.style.visibility === "hidden" || getComputedStyle(tipEl).opacity === "0" || !tipEl.textContent; if (tipEl.hidden !== gone) tipEl.hidden = gone; };
        new MutationObserver(sync).observe(tipEl, { attributes: true, attributeFilter: ["style"], childList: true, subtree: true }); tipEl.hidden = true;
      }
    };
    build();
    host._ecRebuild = build;

    // the compact bar: ranges named the way a trader asks for them, and one reset; the slider and two-finger pinch do the rest
    if (zoomable) {
      const ranges = document.createElement("div"); ranges.className = "rangebar"; bar.appendChild(ranges);
      const go = (a, b) => chart.dispatchAction({ type: "dataZoom", startValue: a, endValue: b });
      const buttons = [];
      if (!numeric && xs[n - 1] - xs[0] > 70 * 864e5) {
        const last = xs[n - 1], span = last - xs[0], first = (ms) => xs.find((x) => x >= last - ms) ?? xs[0];
        const y = new Date(last).getUTCFullYear(), ytd = xs.find((x) => new Date(x).getUTCFullYear() >= y);
        const list = [["1M", 30], ["3M", 91], ["6M", 182], ["1Y", 365], ["3Y", 1096], ["5Y", 1826]].filter(([, d]) => d * 864e5 < span * 0.9).map(([lab, d]) => [lab, first(d * 864e5), `Last ${lab}`]);
        if (ytd !== undefined && ytd > xs[0] && ytd < last - 5 * 864e5) list.push(["YTD", ytd, "Year to date"]);
        list.push(["All", xs[0], "Whole history"]);
        for (const [lab, from, name] of list) {
          const b = document.createElement("button"); b.type = "button"; b.textContent = lab; b.dataset.range = lab; b.title = name; b.setAttribute("aria-label", name); b.addEventListener("click", () => go(from, last)); ranges.appendChild(b); buttons.push([b, from]);
        }
      }
      const zb = document.createElement("div"); zb.className = "zoombar"; bar.appendChild(zb);
      const reset = document.createElement("button"); reset.type = "button"; reset.className = "ec-reset"; reset.textContent = "Reset"; reset.title = "Show everything"; reset.setAttribute("aria-label", "Show everything"); reset.addEventListener("click", () => go(xs[0], xs[n - 1])); zb.appendChild(reset);
      const hint = document.createElement("span"); hint.className = "zhint"; hint.textContent = "Drag across the chart to zoom, or use the strip below"; ranges.appendChild(hint);
      let note = null;
      const mark = () => {
        const z = chart.getOption().dataZoom[0], a = z.startValue, b = z.endValue, full = a <= xs[0] && b >= xs[n - 1]; reset.disabled = full;
        for (const [btn, from] of buttons) btn.setAttribute("aria-pressed", String(Math.abs(a - from) < 1 && b >= xs[n - 1] - 1));
        const shown = xs.filter((x) => x >= a && x <= b).length;
        if (full) { if (note) { note.remove(); note = null; } } else { if (!note) { note = document.createElement("span"); zb.appendChild(note); } note.textContent = `${shown} of ${n} points`; }
      };
      chart.on("datazoom", mark); mark();
      // a mouse drag across the plot zooms to the stretch it covers (shaded while the mouse is down); a double-click shows everything again
      let drag = null;
      const bounds = () => box.getBoundingClientRect();
      box.addEventListener("pointerdown", (e) => { if (e.pointerType !== "mouse" || e.button !== 0 || e.shiftKey) return; drag = { x: e.clientX, id: e.pointerId, el: null }; });
      box.addEventListener("pointermove", (e) => {
        if (!drag || e.pointerId !== drag.id) return; if (!drag.el && Math.abs(e.clientX - drag.x) < 6) return;
        const hb = host.getBoundingClientRect(), bb = bounds(), x0 = Math.max(bb.left, Math.min(drag.x, e.clientX)), x1 = Math.min(bb.right, Math.max(drag.x, e.clientX));
        if (!drag.el) { drag.el = document.createElement("div"); drag.el.className = "brush"; host.appendChild(drag.el); }
        Object.assign(drag.el.style, { left: x0 - hb.left + "px", width: Math.max(0, x1 - x0) + "px", top: bb.top - hb.top + 6 + "px", height: Math.max(0, bb.height - 40) + "px" });
      });
      const drop = (e, apply) => {
        if (!drag || e.pointerId !== drag.id) return; const d = drag; drag = null; if (!d.el) return;
        const hb = host.getBoundingClientRect(), bb = bounds(), l = parseFloat(d.el.style.left) + hb.left - bb.left, w = parseFloat(d.el.style.width); d.el.remove();
        if (apply && w > 8) { const a = chart.convertFromPixel({ xAxisIndex: 0 }, l), b = chart.convertFromPixel({ xAxisIndex: 0 }, l + w); if (isFinite(a) && isFinite(b) && b > a) go(a, b); }
      };
      box.addEventListener("pointerup", (e) => drop(e, true)); box.addEventListener("pointercancel", (e) => drop(e, false));
      box.addEventListener("dblclick", () => go(xs[0], xs[n - 1]));
    }
    if (o.sync && !numeric) ec.connect("sync:" + o.sync);
    if (window.ResizeObserver) { host._ro = new ResizeObserver(() => { if (host.isConnected && box.clientWidth) { chart.resize({ width: box.clientWidth, height: H }); const svg = box.querySelector("svg"); if (svg) svg.setAttribute("viewBox", `0 0 ${box.clientWidth} ${H}`); } }); host._ro.observe(box); }
    if (zoomable) {                                                                                  // the keyboard, as in the SVG kit: + zooms in, - out, 0 shows everything
      box.tabIndex = 0;
      box.addEventListener("keydown", (e) => {
        const z = chart.getOption().dataZoom[0], a = z.startValue, b = z.endValue, c = (a + b) / 2, w = b - a;
        if (e.key === "+" || e.key === "=") chart.dispatchAction({ type: "dataZoom", startValue: c - w / 4, endValue: c + w / 4 });
        else if (e.key === "-") chart.dispatchAction({ type: "dataZoom", startValue: Math.max(xs[0], c - w), endValue: Math.min(xs[n - 1], c + w) });
        else if (e.key === "0") chart.dispatchAction({ type: "dataZoom", startValue: xs[0], endValue: xs[n - 1] });
        else return; e.preventDefault();
      });
    }
  }

  /* ---------------------------------------------------------------------------------------------- shared by the bar-like charts */
  function frame(host, o, H, label) {                                                                // an empty host with a chart box in it, the legend (if asked for) above it, and the instance
    mount(host);
    const box = document.createElement("div"); box.className = "ec-chart"; box.style.cssText = `width:100%;height:${H}px;touch-action:pan-y`; box.setAttribute("role", "img"); box.setAttribute("aria-label", o.label || label);
    host.appendChild(box);
    const chart = ec.init(box, null, { renderer: "svg", height: H }); host._ec = chart; live.add(host);
    if (window.ResizeObserver) { host._ro = new ResizeObserver(() => { if (host.isConnected && box.clientWidth) { chart.resize({ width: box.clientWidth, height: H }); if (host._ecWide && host._ecRebuild) host._ecRebuild(); const svg = box.querySelector("svg"); if (svg) svg.setAttribute("viewBox", `0 0 ${box.clientWidth} ${H}`); } }); host._ro.observe(box); }
    return { box, chart };
  }
  function simpleLegend(host, items, before) {
    const lg = document.createElement("div"); lg.className = "legend";
    for (const it of items) { const span = document.createElement("span"), i = document.createElement("i"); if (it.box) { i.className = "box"; i.style.opacity = String(Math.min(1, (it.opacity || 0.14) * 2.5)); } i.style.setProperty("--c", it.color); span.append(i, document.createTextNode(it.name)); lg.appendChild(span); }
    host.insertBefore(lg, before); return lg;
  }
  function finish(box, chart, label) {                                                                // what every chart gets after its option is set: the class and viewBox the export and the tests read, and a tooltip that carries `hidden`
    const svg = box.querySelector("svg"); if (svg) { svg.classList.add("chart"); svg.setAttribute("viewBox", `0 0 ${box.clientWidth || svg.getAttribute("width")} ${svg.getAttribute("height") || 0}`); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", label); }
    const tipEl = box.querySelector(".ec-tip");
    if (tipEl && !tipEl._watched) { tipEl._watched = true; const sync = () => { const gone = tipEl.style.display === "none" || tipEl.style.visibility === "hidden" || getComputedStyle(tipEl).opacity === "0" || !tipEl.textContent; if (tipEl.hidden !== gone) tipEl.hidden = gone; }; new MutationObserver(sync).observe(tipEl, { attributes: true, attributeFilter: ["style"], childList: true, subtree: true }); tipEl.hidden = true; }
  }
  const tipBase = (t, extra) => ({ confine: true, className: "tip ec-tip", appendToBody: false, backgroundColor: t.surface, borderColor: t.axis, borderWidth: 1, padding: [6, 9], textStyle: { color: t.ink, fontSize: 12 }, triggerOn: matchMedia("(pointer: coarse)").matches ? "click" : "mousemove|click",
    extraCssText: "box-shadow:0 4px 14px rgba(0,0,0,.12);border-radius:8px;position:absolute", ...extra });
  const row = (cls, parent, text, tag) => { const e = document.createElement(tag || "div"); e.className = cls || ""; if (text !== undefined) e.textContent = text; parent.appendChild(e); return e; };
  const sw = (parent, color, label) => { const k = row("k", parent, undefined, "span"), i = document.createElement("i"); i.style.setProperty("--c", color); k.append(i, document.createTextNode(label)); return k; };

  /* ---------------------------------------------------------------------------------------------- columns */
  const columnsLegacy = C.columns;
  function columns(host, o) {
    if (!on() || !o.labels || !o.values || !o.yFormat) return columnsLegacy(host, o);
    const H = o.height || 240, { box, chart } = frame(host, o, H, "column chart"), t = theme(), vals = o.values.map((v) => (v === null ? 0 : v)), best = vals.indexOf(Math.max(...vals)), worst = vals.indexOf(Math.min(...vals));
    const build = () => {
      const t2 = theme();
      chart.setOption({
        animation: false, backgroundColor: "transparent", textStyle: { fontFamily: t2.font, color: t2.muted },
        grid: { left: 8, right: 10, top: 18, bottom: 4, containLabel: true },
        xAxis: { type: "category", data: o.labels.map(String), axisLine: { lineStyle: { color: t2.axis } }, axisTick: { show: false }, axisLabel: { color: t2.muted, hideOverlap: o.labels.length > 12, interval: o.labels.length > 12 ? "auto" : 0 } },                 // a few well-spaced labels are all shown; a long list thins itself
        yAxis: { type: "value", axisLabel: { color: t2.muted, formatter: (v) => o.yFormat(v) }, splitLine: { lineStyle: { color: t2.grid } }, axisLine: { show: false } },
        tooltip: { trigger: "axis", ...tipBase(t2), axisPointer: { type: "shadow", shadowStyle: { color: "rgba(127,127,127,0.12)" } },
          formatter: (ps) => { const i = ps[0].dataIndex, div = document.createElement("div"); row("t", div, o.tipLabel ? o.tipLabel(i) : String(o.labels[i])); const r = row("r", div); const k = row("k", r, o.name || "Value", "span"); row("", r, (o.tipFormat || o.yFormat)(vals[i]), "b"); return div; } },
        series: [{ type: "bar", barMaxWidth: 24, animation: false, data: vals.map((v, i) => ({ value: v, itemStyle: { color: o.colors ? o.colors(v, i) : css("--s1"), opacity: o.faded && o.faded[i] ? 0.55 : 1, borderRadius: v >= 0 ? [4, 4, 0, 0] : [0, 0, 4, 4] },
          label: i === best || i === worst ? { show: true, position: v >= 0 ? "top" : "bottom", formatter: () => o.yFormat(v), color: t2.ink, fontSize: 11 } : { show: false } })), emphasis: { itemStyle: { opacity: 0.8 } } }],
      }, true);
      finish(box, chart, o.label || "column chart");
    };
    build(); host._ecRebuild = build;
  }

  /* ---------------------------------------------------------------------------------------------- horizontal bars (and waterfalls: a row may start away from zero) */
  const hbarsLegacy = C.hbars;
  function hbars(host, o) {
    if (!on() || !o.rows || !o.format) return hbarsLegacy(host, o);
    const rows = o.rows, H = rows.length * 22 + 36, { box, chart } = frame(host, o, H, "bar chart");
    const colour = (r) => (o.color ? o.color(r.value, r) : r.value >= (r.from || 0) ? css("--pos") : css("--neg")), textOf = (r) => (r.text !== undefined ? r.text : o.format(r.value));
    const build = () => {
      const t = theme();
      chart.setOption({
        animation: false, backgroundColor: "transparent", textStyle: { fontFamily: t.font, color: t.muted },
        grid: { left: 8, right: (o.valueWidth || 54) + 8, top: 4, bottom: 4, containLabel: true },
        yAxis: { type: "category", inverse: true, data: rows.map((r) => r.label), axisTick: { show: false }, axisLine: o.zero ? { show: true, onZero: true, lineStyle: { color: t.ink2, width: 1.5 } } : { show: false }, axisLabel: { color: t.ink, width: o.labelWidth || 74, overflow: "truncate", interval: 0 } },
        xAxis: { type: "value", splitNumber: box.clientWidth && box.clientWidth < 420 ? 3 : 5, axisLabel: { color: t.muted, formatter: (v) => o.format(v), hideOverlap: true }, splitLine: { lineStyle: { color: t.grid } }, axisLine: { show: false }, scale: false },
        tooltip: { trigger: "item", ...tipBase(t), formatter: (p) => { const r = rows[p.dataIndex], div = document.createElement("div"); row("t", div, r.label); const a = row("r", div); row("k", a, o.name || "Value", "span"); row("", a, r.text !== undefined ? r.text : (o.tipFormat || o.format)(r.value), "b");
          for (const [k, v] of r.tip || []) { const b = row("r", div); row("k", b, k, "span"); row("", b, v, "b"); } return div; } },
        series: [{ type: "custom", animation: false, data: rows.map((r, i) => [i, r.from || 0, r.value]), encode: { x: [1, 2], y: 0 },
          renderItem: (params, api) => {
            const i = api.value(0), p0 = api.coord([api.value(1), i]), p1 = api.coord([api.value(2), i]), r = rows[i], left = Math.min(p0[0], p1[0]), w = Math.max(1, Math.abs(p1[0] - p0[0])), h = 14, y = p0[1] - h / 2, pos = r.value >= (r.from || 0), rad = Math.min(4, w);
            if (!isFinite(r.value)) return null;                                                      // an undefined value draws nothing (the caller says why); it must not break the other rows
            const whisk = isFinite(r.lo) && isFinite(r.hi) ? [api.coord([r.lo, i])[0], api.coord([r.hi, i])[0]] : null;
            return { type: "group", children: [
              { type: "rect", shape: { x: left, y, width: w, height: h, r: pos ? [0, rad, rad, 0] : [rad, 0, 0, rad] }, style: { fill: colour(r) } },
              ...(whisk ? [{ type: "line", shape: { x1: whisk[0], x2: whisk[1], y1: p0[1], y2: p0[1] }, style: { stroke: css("--ink"), lineWidth: 1.5 } }, { type: "line", shape: { x1: whisk[0], x2: whisk[0], y1: p0[1] - 4, y2: p0[1] + 4 }, style: { stroke: css("--ink"), lineWidth: 1.5 } },
                { type: "line", shape: { x1: whisk[1], x2: whisk[1], y1: p0[1] - 4, y2: p0[1] + 4 }, style: { stroke: css("--ink"), lineWidth: 1.5 } }] : []),
              { type: "text", style: { x: Math.max(p0[0], p1[0], whisk ? Math.max(...whisk) : 0) + 5, y: p0[1], text: textOf(r), fill: css("--ink"), fontSize: 12, textVerticalAlign: "middle", textAlign: "left" } },
            ] };
          } }],
      }, true);
      finish(box, chart, o.label || "bar chart");
    };
    build(); host._ecRebuild = build;
  }

  /* ---------------------------------------------------------------------------------------------- histogram: each series as a step line over its bins, the first one filled; markers are vertical references */
  const histLegacy = C.histogram;
  function histogram(host, o) {
    const series = (o.series || []).filter((s) => s.counts && s.counts.length);
    if (!on() || !series.length) return histLegacy(host, o);
    const H = o.height || 220, { box, chart } = frame(host, o, H, "histogram"), unit = o.unit || 1;
    if (series.length > 1) simpleLegend(host, series.map((s) => ({ name: s.name, color: s.color })), box);
    const dens = series.map((s) => { const n = s.counts.reduce((a, b) => a + b, 0) || 1; return s.counts.map((c, i) => (c / n / (s.edges[i + 1] - s.edges[i] || 1)) * unit); });
    const x0 = Math.min(...series.map((s) => s.edges[0])), x1 = Math.max(...series.map((s) => s.edges[s.edges.length - 1])), top = Math.max(...dens.flat()) * 1.08 || 1;
    const yf = o.yFormat || ((v) => (v >= 10 ? v.toFixed(0) : v.toPrecision(2))), xf = o.xFormat || C.fmt.num;
    const build = () => {
      const t = theme();
      chart.setOption({
        animation: false, backgroundColor: "transparent", textStyle: { fontFamily: t.font, color: t.muted },
        grid: { left: 8, right: 14, top: 10, bottom: 4, containLabel: true },
        xAxis: { type: "value", min: x0, max: x1, axisLabel: { color: t.muted, formatter: (v) => xf(v), hideOverlap: true }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, splitLine: { show: false } },
        yAxis: { type: "value", min: 0, max: top, axisLabel: { color: t.muted, formatter: (v) => yf(v) }, splitLine: { lineStyle: { color: t.grid } }, axisLine: { show: false } },
        tooltip: { trigger: "axis", ...tipBase(t), axisPointer: { type: "line", lineStyle: { color: t.ink2, type: "dashed" } },
          formatter: (ps) => { const x = ps[0].axisValue, div = document.createElement("div"); row("t", div, `${xf(x)}`);
            series.forEach((s, k) => { let i = s.counts.length - 1; for (let j = 0; j < s.counts.length; j++) if (x < s.edges[j + 1]) { i = j; break; } if (x < s.edges[0] || x > s.edges[s.edges.length - 1]) return; const r = row("r", div); sw(r, s.color, s.name); row("", r, `${xf(s.edges[i])} to ${xf(s.edges[i + 1])}: ${yf(dens[k][i])}`, "b"); }); return div; } },
        series: series.map((s, k) => {
          const pts = []; s.counts.forEach((_, i) => { pts.push([s.edges[i], dens[k][i]], [s.edges[i + 1], dens[k][i]]); });
          return { name: s.name, type: "line", data: pts, showSymbol: false, animation: false, lineStyle: { color: s.color, width: 2 }, itemStyle: { color: s.color }, emphasis: { disabled: true }, areaStyle: k === 0 ? { color: s.color, opacity: 0.1 } : undefined,
            markLine: k === 0 && (o.markers || []).length ? { silent: true, symbol: "none", animation: false, data: (o.markers || []).filter((m) => m.x >= x0 && m.x <= x1).map((m) => ({ xAxis: m.x, lineStyle: { color: t.muted, type: "dashed", width: 1 }, label: { show: !!m.label, formatter: m.label || "", position: "insideEndTop", color: t.ink2, fontSize: 11 } })) } : undefined };
        }),
      }, true);
      finish(box, chart, o.label || "histogram");
    };
    build(); host._ecRebuild = build;
  }

  document.addEventListener("pointerdown", (e) => { for (const h of live) if (h._ec && !h.contains(e.target)) { try { h._ec.dispatchAction({ type: "hideTip" }); } catch (err) { /* being removed */ } } }, true);

  /* ---------------------------------------------------------------------------------------------- colours shared by the heat map and the matrix */
  const rgb = (c) => { if (c.startsWith("#")) { const n = parseInt(c.length === 4 ? c.replace(/./g, (x, i) => (i ? x + x : x)).slice(1) : c.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; } return (c.match(/\d+/g) || [128, 128, 128]).slice(0, 3).map(Number); };
  const mix = (a, b, t) => { const pa = rgb(a), pb = rgb(b); return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",")})`; };
  const isDark = (c) => { const [r, g, b] = rgb(c); return 0.299 * r + 0.587 * g + 0.114 * b < 140; };
  const num = (v) => v !== null && v !== undefined && isFinite(v);
  const widthSensitive = (host, build) => { host._ecRebuild = build; host._ecWide = true; };

  /* ---------------------------------------------------------------------------------------------- matrix: rows by columns, each cell shaded by its value */
  const matrixLegacy = C.matrix;
  function matrix(host, o) {
    const flat = (o.values || []).flat().filter(num);
    if (!on() || !o.rows || !o.cols || !flat.length) return matrixLegacy(host, o);
    const rows = o.rows, cols = o.cols, vals = o.values, fm = o.format || C.fmt.num, ch = o.cellHeight || 22, t0 = theme();
    const labW = o.labelWidth || Math.min(150, Math.max(46, Math.round(Math.max(...rows.map((r) => String(r).length), 2) * 6.4) + 14)), top = o.colTitle ? 48 : 22, bottom = 36, H = top + rows.length * ch + bottom;
    const div = !!o.diverging, mid = o.mid !== undefined ? o.mid : 0;
    let lo = o.min !== undefined ? o.min : Math.min(...flat), hi = o.max !== undefined ? o.max : Math.max(...flat);
    if (div) { const span = Math.max(Math.abs(lo - mid), Math.abs(hi - mid), 1e-12); lo = mid - span; hi = mid + span; } else if (!(hi > lo)) hi = lo + 1;
    const { box, chart } = frame(host, o, H, "matrix");
    const build = () => {
      const t = theme(), seq = [css("--scale-lo"), css("--scale-mid"), css("--scale-hi")], neg = css("--neg"), pos = css("--pos"), neutral = css("--mid");
      const colour = (v) => { if (div) { const k = Math.min(1, Math.abs(v - mid) / (hi - mid)); return v >= mid ? mix(neutral, pos, k) : mix(neutral, neg, k); } const k = Math.min(1, Math.max(0, (v - lo) / (hi - lo))); return k < 0.5 ? mix(seq[0], seq[1], k * 2) : mix(seq[1], seq[2], (k - 0.5) * 2); };
      const W = box.clientWidth || 600, cw = (W - labW - 8) / cols.length, every = Math.max(1, Math.ceil(46 / Math.max(cw, 1)));
      const data = []; rows.forEach((_, i) => cols.forEach((__, j) => data.push([j, i, num(vals[i][j]) ? vals[i][j] : null])));
      chart.setOption({
        animation: false, backgroundColor: "transparent", textStyle: { fontFamily: t.font, color: t.muted },
        title: [o.colTitle ? { text: o.colTitle, left: labW + (W - labW - 8) / 2, top: 0, textAlign: "center", textStyle: { color: t.ink, fontSize: 12, fontWeight: "normal" } } : {}, o.rowTitle ? { text: o.rowTitle, left: 4, top: 0, textStyle: { color: t.ink, fontSize: 12, fontWeight: "normal" } } : {}].filter((x) => x.text),
        grid: { left: labW, right: 8, top, bottom, containLabel: false },
        xAxis: { type: "category", data: cols.map(String), position: "top", splitArea: { show: false }, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: t.muted, interval: every - 1 } },
        yAxis: { type: "category", data: rows.map(String), inverse: true, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: t.ink, width: labW - 10, overflow: "truncate", interval: 0 } },
        tooltip: { trigger: "item", ...tipBase(t), formatter: (p) => { const [j, i, v] = p.value; if (!num(v)) return ""; const d = document.createElement("div"); row("t", d, `${rows[i]} · ${cols[j]}`); const r = row("r", d); row("k", r, o.name || "Value", "span"); row("", r, (o.tipFormat || fm)(v), "b");
          for (const [k, tx] of (o.tipRows ? o.tipRows(i, j) : [])) { const r3 = row("r", d); row("k", r3, k, "span"); row("", r3, tx, "b"); } return d; } },
        series: [{ type: "custom", animation: false, data, clip: true,
          renderItem: (params, api) => {
            const j = api.value(0), i = api.value(1), v = api.value(2), p = api.coord([j, i]), sz = api.size([1, 1]), w = Math.max(1, sz[0] - 2), h = Math.max(1, sz[1] - 2), x = p[0] - sz[0] / 2 + 1, y = p[1] - sz[1] / 2 + 1;
            if (!num(v)) return { type: "rect", shape: { x, y, width: w, height: h, r: 2 }, style: { fill: css("--ghost") }, silent: true };
            const fill = colour(v), text = o.cellText === false ? null : fm(v), marked = o.mark && o.mark(i, j);
            const kids = [{ type: "rect", shape: { x, y, width: w, height: h, r: 2 }, style: { fill, stroke: marked ? css("--ink") : "none", lineWidth: marked ? 1.5 : 0 } }];
            if (text && sz[0] >= 7 * text.length + 6) kids.push({ type: "text", style: { x: p[0], y: p[1], text, fill: isDark(fill) ? "#fff" : css("--ink-2"), fontSize: 10, textAlign: "center", textVerticalAlign: "middle" }, silent: true });
            return { type: "group", children: kids };
          } }],
      }, true);
      finish(box, chart, o.label || "matrix");
      let scale = host.querySelector(":scope > .mx-scale"); if (scale) scale.remove();               // the scale: a gradient bar under the grid with its two ends and, for a diverging scale, the middle
      scale = document.createElement("div"); scale.className = "mx-scale"; scale.style.cssText = `margin:2px 0 0 ${labW}px;width:${Math.min(160, Math.max(80, W - labW - 8))}px;font-size:11px;color:${t.muted}`;
      const bar = document.createElement("div"); bar.style.cssText = `height:8px;border-radius:2px;background:linear-gradient(90deg,${[0, 0.5, 1].map((k) => colour(lo + k * (hi - lo))).join(",")})`; scale.appendChild(bar);
      const lab = document.createElement("div"); lab.style.cssText = "display:flex;justify-content:space-between;margin-top:2px"; for (const tx of div ? [fm(lo), fm(mid), fm(hi)] : [fm(lo), fm(hi)]) { const sp = document.createElement("span"); sp.textContent = tx; lab.appendChild(sp); } scale.appendChild(lab);
      host.appendChild(scale);
    };
    build(); widthSensitive(host, build);
  }

  /* ---------------------------------------------------------------------------------------------- month x year heat map (diverging, gray midpoint) */
  const heatLegacy = C.heatmap;
  function heatmap(host, o) {
    if (!on() || !o.cells || !o.cells.length) return heatLegacy(host, o);
    const years = [...new Set(o.cells.map((c) => c.year))].sort(), ch = 20, H = 22 + years.length * ch, { box, chart } = frame(host, o, H, "heat map");
    const by = {}; o.cells.forEach((c) => { by[c.year + "-" + c.month] = c; });
    const mx = Math.max(0.02, ...o.cells.map((c) => Math.abs(c.value))), cap = Math.min(mx, 0.12);
    const build = () => {
      const t = theme(), pos = css("--pos"), neg = css("--neg"), mid = css("--mid"), W = box.clientWidth || 600, cw = Math.max(18, Math.min(56, (W - 46) / 12));
      const data = []; years.forEach((y, r) => { for (let k = 1; k <= 12; k++) { const c = by[y + "-" + k]; if (c) data.push([k - 1, r, c.value]); } });
      chart.setOption({
        animation: false, backgroundColor: "transparent", textStyle: { fontFamily: t.font, color: t.muted },
        grid: { left: 38, right: 8, top: 18, bottom: 4 },
        xAxis: { type: "category", data: "JFMAMJJASOND".split(""), position: "top", axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: t.muted } },
        yAxis: { type: "category", data: years.map(String), inverse: true, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: t.muted } },
        tooltip: { trigger: "item", ...tipBase(t), formatter: (p) => { const [m, r, v] = p.value, d = document.createElement("div"); row("t", d, `${years[r]}-${String(m + 1).padStart(2, "0")}`); const a = row("r", d); row("k", a, "Return", "span"); row("", a, C.fmt.spct(v), "b"); return d; } },
        series: [{ type: "custom", animation: false, data, renderItem: (params, api) => {
          const m = api.value(0), r = api.value(1), v = api.value(2), p = api.coord([m, r]), sz = api.size([1, 1]), k = Math.min(1, Math.abs(v) / cap), fill = v >= 0 ? mix(mid, pos, k) : mix(mid, neg, k);
          const kids = [{ type: "rect", shape: { x: p[0] - sz[0] / 2 + 1, y: p[1] - sz[1] / 2 + 1, width: Math.max(1, sz[0] - 2), height: Math.max(1, sz[1] - 2), r: 3 }, style: { fill } }];
          if (cw >= 34) kids.push({ type: "text", style: { x: p[0], y: p[1], text: (v * 100).toFixed(1), fill: k > 0.55 ? "#fff" : css("--ink-2"), fontSize: 10, textAlign: "center", textVerticalAlign: "middle" }, silent: true });
          return { type: "group", children: kids };
        } }],
      }, true);
      finish(box, chart, o.label || "heat map");
    };
    build(); widthSensitive(host, build);
  }

  /* ---------------------------------------------------------------------------------------------- scatter: points with their own labels, named lines, zero lines */
  const scatterLegacy = C.scatter;
  function scatter(host, o) {
    const pts = (o.points || []).filter((p) => isFinite(p.x) && isFinite(p.y));
    if (!on() || !pts.length) return scatterLegacy(host, o);
    const H = o.height || 320, { box, chart } = frame(host, o, H, "scatter plot"), xf = o.xFormat || C.fmt.num, yf = o.yFormat || C.fmt.num, named = (o.lines || []).filter((l) => l.name);
    const legends = (o.legend && o.legend.length > 1 ? o.legend.map((g) => ({ name: g.name, color: g.color, box: true, opacity: 0.4 })) : []).concat(named.map((l) => ({ name: l.name, color: l.color || css("--s2") })));
    if (legends.length) simpleLegend(host, legends, box);
    const span = (arr, pad) => { const lo = Math.min(...arr), hi = Math.max(...arr), p = (hi - lo) * pad || 0.5; return [lo - p, hi + p]; };
    const [x0, x1] = o.xRange || span(pts.map((p) => p.x), 0.08), [y0, y1] = o.yRange || span(pts.map((p) => p.y), 0.1);
    const build = () => {
      const t = theme(), lines = (o.lines || []).map((l) => [{ coord: [l.x1, l.y1], lineStyle: { color: l.color || css("--s2"), width: 2, type: l.dash ? "dashed" : "solid" } }, { coord: [l.x2, l.y2] }]);
      const zeros = [].concat(o.xZero && x0 < 0 && x1 > 0 ? [{ xAxis: 0, lineStyle: { color: t.axis, type: "solid", width: 1 } }] : [], o.yZero && y0 < 0 && y1 > 0 ? [{ yAxis: 0, lineStyle: { color: t.axis, type: "solid", width: 1 } }] : []);
      chart.setOption({
        animation: false, backgroundColor: "transparent", textStyle: { fontFamily: t.font, color: t.muted },
        grid: { left: 8, right: 22, top: o.yLabel ? 26 : 12, bottom: o.xLabel ? 30 : 6, containLabel: true },
        xAxis: { type: "value", min: x0, max: x1, name: o.xLabel || "", nameLocation: "middle", nameGap: 28, nameTextStyle: { color: t.ink2 }, axisLabel: { color: t.muted, formatter: (v) => xf(v), hideOverlap: true, showMinLabel: false, showMaxLabel: false }, axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, splitLine: { show: false } },
        yAxis: { type: "value", min: y0, max: y1, name: o.yLabel || "", nameLocation: "end", nameTextStyle: { color: t.ink2, align: "left" }, axisLabel: { color: t.muted, formatter: (v) => yf(v), showMinLabel: false, showMaxLabel: false }, axisLine: { show: false }, splitLine: { lineStyle: { color: t.grid } } },
        tooltip: { trigger: "item", ...tipBase(t), formatter: (p) => { const pt = pts[p.dataIndex], d = document.createElement("div"); row("t", d, pt.label ? String(pt.label) : "");
          for (const [k, v] of pt.tip || [[o.xLabel || "x", xf(pt.x)], [o.yLabel || "y", yf(pt.y)]]) { const r = row("r", d); row("k", r, k, "span"); row("", r, v, "b"); } return d; } },
        series: [{ type: "scatter", animation: false, clip: true, data: pts.map((p) => ({ value: [p.x, p.y], symbolSize: Math.max(8, (p.r || 4.5) * 2), itemStyle: { color: p.color || css("--s1"), borderColor: t.surface, borderWidth: 2 },
            label: { show: o.labels !== "none" && !!p.label, formatter: String(p.label || ""), position: "right", color: t.ink, fontSize: 11 } })),
          labelLayout: { hideOverlap: o.labels !== "all", moveOverlap: false }, emphasis: { scale: 1.3 },
          markLine: lines.length || zeros.length ? { silent: true, symbol: "none", animation: false, label: { show: false }, data: lines.concat(zeros) } : undefined }],
      }, true);
      finish(box, chart, o.label || "scatter plot");
    };
    build(); host._ecRebuild = build;
  }

  /* ---------------------------------------------------------------------------------------------- mirrored columns: two measures of the same strikes or dates, one up and one down */
  const mirrorLegacy = C.mirror;
  function mirror(host, o) {
    if (!on() || !o.labels || !o.up || !o.down) return mirrorLegacy(host, o);
    const clean = (a) => a.map((v) => (!num(v) ? 0 : Math.max(v, 0))), up = clean(o.up), down = clean(o.down), upColor = o.upColor || css("--s1"), downColor = o.downColor || css("--s2");
    if (!(Math.max(0, ...up) || Math.max(0, ...down))) return mirrorLegacy(host, o);
    const H = o.height || 260, { box, chart } = frame(host, o, H, "mirrored column chart"), fy = o.yFormat || C.fmt.num;
    simpleLegend(host, [{ name: o.upName, color: upColor, box: true, opacity: 0.4 }, { name: o.downName, color: downColor, box: true, opacity: 0.4 }], box);
    const build = () => {
      const t = theme();
      chart.setOption({
        animation: false, backgroundColor: "transparent", textStyle: { fontFamily: t.font, color: t.muted },
        grid: { left: 8, right: 10, top: 18, bottom: 4, containLabel: true },
        xAxis: { type: "category", data: o.labels.map((l) => (l === null || l === undefined ? "" : String(l))), axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, hideOverlap: true } },
        yAxis: { type: "value", axisLabel: { color: t.muted, formatter: (v) => fy(Math.abs(v)) }, splitLine: { lineStyle: { color: t.grid } }, axisLine: { show: false } },
        tooltip: { trigger: "axis", ...tipBase(t), axisPointer: { type: "shadow", shadowStyle: { color: "rgba(127,127,127,0.12)" } },
          formatter: (ps) => { const i = ps[0].dataIndex, d = document.createElement("div"); row("t", d, o.tipTitle ? o.tipTitle(i) : String(o.labels[i])); for (const [name, v, c] of [[o.upName, up[i], upColor], [o.downName, down[i], downColor]]) { const r = row("r", d); sw(r, c, name); row("", r, (o.tipFormat || fy)(v), "b"); } return d; } },
        series: [
          { name: o.upName, type: "bar", stack: "m", barMaxWidth: 24, data: up, itemStyle: { color: upColor, borderRadius: [2, 2, 0, 0] }, animation: false,
            markLine: o.mark !== undefined && o.mark >= 0 && o.mark < o.labels.length ? { silent: true, symbol: "none", animation: false, data: [{ xAxis: o.mark, lineStyle: { color: t.muted, type: "dashed" }, label: { show: !!o.markLabel, formatter: o.markLabel || "", color: t.ink2, fontSize: 11, position: "end" } }] } : undefined },
          { name: o.downName, type: "bar", stack: "m", barMaxWidth: 24, data: down.map((v) => -v), itemStyle: { color: downColor, borderRadius: [0, 0, 2, 2] }, animation: false },
        ],
      }, true);
      finish(box, chart, o.label || "mirrored column chart");
    };
    build(); host._ecRebuild = build;
  }

  /* ---------------------------------------------------------------------------------------------- stacked columns with a sign: what each part added to a total */
  const stackLegacy = C.stack;
  function stack(host, o) {
    if (!on() || !o.labels || !o.series) return stackLegacy(host, o);
    const H = o.height || 260, { box, chart } = frame(host, o, H, "stacked column chart"), fy = o.yFormat || C.fmt.num;
    simpleLegend(host, o.series.map((s) => ({ name: s.name, color: s.color, box: true, opacity: 0.4 })).concat(o.total ? [{ name: o.total.name, color: css("--ink") }] : []), box);
    const build = () => {
      const t = theme();
      chart.setOption({
        animation: false, backgroundColor: "transparent", textStyle: { fontFamily: t.font, color: t.muted },
        grid: { left: 8, right: 10, top: 14, bottom: 4, containLabel: true },
        xAxis: { type: "category", data: o.labels.map(String), axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false }, axisLabel: { color: t.muted, hideOverlap: o.labels.length > 12, interval: o.labels.length > 12 ? "auto" : 0 } },
        yAxis: { type: "value", axisLabel: { color: t.muted, formatter: (v) => fy(v) }, splitLine: { lineStyle: { color: t.grid } }, axisLine: { show: false } },
        tooltip: { trigger: "axis", ...tipBase(t), axisPointer: { type: "shadow", shadowStyle: { color: "rgba(127,127,127,0.12)" } },
          formatter: (ps) => { const i = ps[0].dataIndex, d = document.createElement("div"); row("t", d, String(o.labels[i])); for (const sr of o.series) { if (!num(sr.values[i])) continue; const r = row("r", d); sw(r, sr.color, sr.name); row("", r, (o.tipFormat || fy)(sr.values[i]), "b"); }
            if (o.total && num(o.total.values[i])) { const r = row("r", d); row("k", r, o.total.name, "span"); row("", r, (o.tipFormat || fy)(o.total.values[i]), "b"); } return d; } },
        series: o.series.map((sr) => ({ name: sr.name, type: "bar", stack: "parts", barMaxWidth: 34, animation: false, data: sr.values.map((v) => (num(v) ? v : null)), itemStyle: { color: sr.color, borderColor: t.surface, borderWidth: 1 } }))
          .concat(o.total ? [{ name: o.total.name, type: "scatter", animation: false, z: 5, symbolSize: 8, data: o.total.values.map((v) => (num(v) ? v : null)), itemStyle: { color: t.ink, borderColor: t.surface, borderWidth: 1.5 }, tooltip: { show: false } }] : []),
      }, true);
      finish(box, chart, o.label || "stacked column chart");
    };
    build(); host._ecRebuild = build;
  }

  /* ---------------------------------------------------------------------------------------------- ranges: one floating bar per row, a mark inside it and a reference line (a football field) */
  const rangesLegacy = C.ranges;
  function ranges(host, o) {
    const rows = o.rows || [], have = rows.filter((r) => num(r.low) && num(r.high));
    if (!on() || !have.length) return rangesLegacy(host, o);
    const fm = o.format || C.fmt.num, H = (o.ref ? 22 : 8) + rows.length * 30 + 28, { box, chart } = frame(host, o, H, "range chart"), labW = o.labelWidth || 150;
    const values = have.flatMap((r) => [r.low, r.high]).concat(o.ref ? [o.ref.value] : []), lo = Math.min(0, ...values), hi = Math.max(0, ...values);
    const build = () => {
      const t = theme();
      chart.setOption({
        animation: false, backgroundColor: "transparent", textStyle: { fontFamily: t.font, color: t.muted },
        grid: { left: 8, right: 16, top: o.ref ? 26 : 8, bottom: 4, containLabel: true },
        yAxis: { type: "category", inverse: true, data: rows.map((r) => r.label), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: t.ink, width: labW - 12, overflow: "truncate", interval: 0 } },
        xAxis: { type: "value", min: lo, max: hi * 1.04 || 1, axisLabel: { color: t.muted, formatter: (v) => fm(v) }, splitLine: { lineStyle: { color: t.grid } }, axisLine: { show: false } },
        tooltip: { trigger: "item", ...tipBase(t), formatter: (p) => { const r = rows[p.dataIndex]; if (!num(r.low) || !num(r.high)) return r.missing || "n/a"; const d = document.createElement("div"), point = num(r.point) ? r.point : (r.low + r.high) / 2; row("t", d, r.label);
          const lines = r.low === r.high ? [["Value", fm(r.low)]] : [["Low", fm(r.low)], ["Middle", fm(point)], ["High", fm(r.high)]]; if (o.ref) lines.push(["Price against the middle", (o.ref.value / point - 1 >= 0 ? "+" : "−") + Math.abs((o.ref.value / point - 1) * 100).toFixed(0) + "%"]);
          for (const [k, v] of lines) { const a = row("r", d); row("k", a, k, "span"); row("", a, v, "b"); } if (r.text) row("n", d, r.text); return d; } },
        series: [{ type: "custom", animation: false, data: rows.map((r, i) => [i, r.low, r.high, num(r.point) ? r.point : null]), encode: { x: [1, 2], y: 0 },
          markLine: o.ref ? { silent: true, symbol: "none", animation: false, data: [{ xAxis: o.ref.value, lineStyle: { color: t.muted, type: "dashed" }, label: { show: true, formatter: o.ref.label || fm(o.ref.value), color: t.ink2, fontSize: 11, position: "start", distance: 4 } }] } : undefined,
          renderItem: (params, api) => {
            const i = api.value(0), r = rows[i], cs = params.coordSys;
            if (!num(r.low) || !num(r.high)) return { type: "text", style: { x: cs.x + 6, y: api.coord([0, i])[1], text: r.missing || "n/a", fill: css("--muted"), fontSize: 12, textVerticalAlign: "middle" }, silent: false };
            const p0 = api.coord([r.low, i]), p1 = api.coord([r.high, i]), color = o.color ? o.color(r, i) : css("--s1"), mid = p0[1], pointV = num(r.point) ? r.point : (r.low + r.high) / 2, px = api.coord([pointV, i])[0];
            if (p1[0] - p0[0] < 3) return { type: "polygon", shape: { points: [[p0[0], mid - 8], [p0[0] + 7, mid], [p0[0], mid + 8], [p0[0] - 7, mid]] }, style: { fill: color, stroke: css("--surface"), lineWidth: 1.5 } };
            return { type: "group", children: [{ type: "rect", shape: { x: p0[0], y: mid - 9, width: p1[0] - p0[0], height: 18, r: 4 }, style: { fill: color, opacity: 0.28, stroke: color, lineWidth: 1.5 } },
              { type: "line", shape: { x1: px, y1: mid - 11, x2: px, y2: mid + 11 }, style: { stroke: color, lineWidth: 3, lineCap: "round" } }] };
          } }],
      }, true);
      finish(box, chart, o.label || "range chart");
    };
    build(); host._ecRebuild = build;
  }

  /* A theme change repaints every live chart in the new colours. */
  new MutationObserver(() => { sweep(); for (const h of live) if (h._ecRebuild) { try { h._ecRebuild(); } catch (e) { /* a chart being removed */ } } }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });

  C.line = line;
  C.columns = columns;
  C.hbars = hbars;
  C.histogram = histogram;
  C.matrix = matrix;
  C.heatmap = heatmap;
  C.scatter = scatter;
  C.mirror = mirror;
  C.stack = stack;
  C.ranges = ranges;
  C.legacy = legacy;
  C.engine = "echarts";
})();
