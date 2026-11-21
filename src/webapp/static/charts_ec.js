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
    host.replaceChildren();
    return host;
  }
  function boxHeight(host, o) { return host.closest(".expanded") ? Math.max(o.height || 280, Math.round(window.innerHeight - 250)) : o.height || 280; }
  const theme = () => ({ ink: css("--ink"), ink2: css("--ink-2"), muted: css("--muted"), grid: css("--grid"), axis: css("--axis"), surface: css("--surface"), font: getComputedStyle(document.body).fontFamily });

  /* ---------------------------------------------------------------------------------------------- line */
  const LINE_OK = (o) => Array.isArray(o.series) && !o.signFill && !o.events && !(o.area && o.areaBase !== undefined) && !o.series.some((s) => s.pointLabel) && !(o.xs && o.sync);
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
        const area = o.area && k === 0;
        series.push({
          name: s.name, type: "line", data: seriesData(s.values), connectNulls: !!s.connect, showSymbol: !!s.dots, symbolSize: (s.dotSize || 3.5) * 2, sampling: "lttb", animation: false, emphasis: { disabled: true },
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
        grid: { left: 8, right: 14, top: regions.length || vlines.length ? 24 : 10, bottom: (zoomable ? 30 : 6) + (o.xLabel ? 18 : 0), containLabel: true },
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
            const first = params.find((p) => p.seriesIndex < o.series.length); if (!first) return "";                // (a hidden series sends no params)
            const i = xs.indexOf(first.value[0]); if (i < 0) return "";
            const div = document.createElement("div"), add = (cls, parent, text) => { const e = document.createElement(cls[0]); e.className = cls[1] || ""; if (text !== undefined) e.textContent = text; (parent || div).appendChild(e); return e; };
            add(["div", "t"], null, numeric ? (o.tipTitle ? o.tipTitle(i) : (o.xFormat || fmt.num)(xs[i])) : o.dates[i]);
            o.series.forEach((s, k) => {
              if (params.every((p) => p.seriesIndex !== k)) return;                                  // a hidden series is not read
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
    if (window.ResizeObserver) { host._ro = new ResizeObserver(() => { if (host.isConnected && box.clientWidth) { chart.resize({ width: box.clientWidth, height: H }); const svg = box.querySelector("svg"); if (svg) svg.setAttribute("viewBox", `0 0 ${box.clientWidth} ${H}`); } }); host._ro.observe(box); }
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
        yAxis: { type: "category", inverse: true, data: rows.map((r) => r.label), axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: t.ink, width: o.labelWidth || 74, overflow: "truncate", interval: 0 } },
        xAxis: { type: "value", axisLabel: { color: t.muted, formatter: (v) => o.format(v) }, splitLine: { lineStyle: { color: t.grid } }, axisLine: { show: false }, scale: false },
        tooltip: { trigger: "item", ...tipBase(t), formatter: (p) => { const r = rows[p.dataIndex], div = document.createElement("div"); row("t", div, r.label); const a = row("r", div); row("k", a, o.name || "Value", "span"); row("", a, r.text !== undefined ? r.text : (o.tipFormat || o.format)(r.value), "b");
          for (const [k, v] of r.tip || []) { const b = row("r", div); row("k", b, k, "span"); row("", b, v, "b"); } return div; } },
        series: [{ type: "custom", animation: false, data: rows.map((r, i) => [i, r.from || 0, r.value]), encode: { x: [1, 2], y: 0 },
          renderItem: (params, api) => {
            const i = api.value(0), p0 = api.coord([api.value(1), i]), p1 = api.coord([api.value(2), i]), r = rows[i], left = Math.min(p0[0], p1[0]), w = Math.max(1, Math.abs(p1[0] - p0[0])), h = 14, y = p0[1] - h / 2, pos = r.value >= (r.from || 0), rad = Math.min(4, w);
            return { type: "group", children: [
              { type: "rect", shape: { x: left, y, width: w, height: h, r: pos ? [0, rad, rad, 0] : [rad, 0, 0, rad] }, style: { fill: colour(r) } },
              { type: "text", style: { x: Math.max(p0[0], p1[0]) + 5, y: p0[1], text: textOf(r), fill: css("--ink"), fontSize: 12, textVerticalAlign: "middle", textAlign: "left" } },
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

  /* A theme change repaints every live chart in the new colours. */
  new MutationObserver(() => { sweep(); for (const h of live) if (h._ecRebuild) { try { h._ecRebuild(); } catch (e) { /* a chart being removed */ } } }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });

  C.line = line;
  C.columns = columns;
  C.hbars = hbars;
  C.histogram = histogram;
  C.legacy = legacy;
  C.engine = "echarts";
})();
