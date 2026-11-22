/* A 3D surface drawn on a canvas, without a graphics library: window.Surface3D.draw(host, {x, y, z, xLabel, yLabel, zLabel, format}) with z[i][j] the value at y[i], x[j] (null where there is none: left open, not
   interpolated). Drag to rotate, wheel to zoom, the keyboard arrows rotate too; a table view beside it carries the same numbers for anyone who cannot use the picture. Painter's algorithm over the quads, coloured by
   height. This is a small viewer, not a plotting library: no lighting, no picking. */
(function () {
  "use strict";
  const finite = (v) => v !== null && v !== undefined && isFinite(v);
  function draw(host, o) {
    const canvas = document.createElement("canvas"); canvas.style.cssText = "width:100%;height:340px;touch-action:none;cursor:grab"; canvas.tabIndex = 0;
    canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", o.label || "3D surface");
    host.appendChild(canvas);
    const state = { yaw: -0.7, pitch: 0.55, zoom: 1 };
    const vals = o.z.flat().filter(finite), lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
    const nx = o.x.length, ny = o.y.length, norm = (a, v) => (v - a[0]) / ((a[a.length - 1] - a[0]) || 1) - 0.5;
    const css = (n, d) => (getComputedStyle(document.documentElement).getPropertyValue(n).trim() || d);
    const mix = (c0, c1, t) => { const p = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)); const a = p(c0), b = p(c1); return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`; };
    function paint() {
      const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth || 600, H = canvas.clientHeight || 340;
      canvas.width = W * dpr; canvas.height = H * dpr;
      const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch), scale = Math.min(W, H) * 0.62 * state.zoom;
      const proj = (xi, yi, v) => {
        const X = norm(o.x, o.x[xi]), Y = norm(o.y, o.y[yi]), Z = (v - lo) / span - 0.5;
        const x1 = X * cy - Y * sy, y1 = X * sy + Y * cy, y2 = y1 * cp - Z * sp, z2 = y1 * sp + Z * cp;
        return [W / 2 + x1 * scale, H / 2 + y2 * scale, z2];
      };
      const c0 = css("--scale-lo", "#2a6fb0"), c1 = css("--scale-hi", "#c8452f"), quads = [];
      for (let i = 0; i + 1 < ny; i++) for (let j = 0; j + 1 < nx; j++) {
        const vs = [o.z[i][j], o.z[i][j + 1], o.z[i + 1][j + 1], o.z[i + 1][j]];
        if (!vs.every(finite)) continue;                                              // a hole stays a hole
        const pts = [proj(j, i, vs[0]), proj(j + 1, i, vs[1]), proj(j + 1, i + 1, vs[2]), proj(j, i + 1, vs[3])];
        quads.push({ pts, depth: pts.reduce((s, p) => s + p[2], 0) / 4, t: (vs.reduce((s, v) => s + v, 0) / 4 - lo) / span });
      }
      quads.sort((a, b) => a.depth - b.depth);
      g.lineWidth = 0.5; g.strokeStyle = css("--border", "#00000022");
      for (const q of quads) { g.beginPath(); q.pts.forEach((p, k) => (k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.closePath(); g.fillStyle = mix(c0, c1, q.t); g.fill(); g.stroke(); }
      g.fillStyle = css("--muted", "#666"); g.font = "11px sans-serif";
      const fmt = o.format || ((v) => v.toFixed(3));
      g.fillText(`${o.zLabel || "z"}: ${fmt(lo)} to ${fmt(hi)}   ·   ${o.xLabel || "x"} → ${o.yLabel || "y"} ↑   ·   drag to rotate`, 8, H - 8);
    }
    let drag = null;
    canvas.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = "grabbing"; });
    canvas.addEventListener("pointerup", () => { drag = null; canvas.style.cursor = "grab"; });
    canvas.addEventListener("pointermove", (e) => { if (!drag) return; state.yaw += (e.clientX - drag[0]) * 0.01; state.pitch = Math.max(0.05, Math.min(1.5, state.pitch + (e.clientY - drag[1]) * 0.01)); drag = [e.clientX, e.clientY]; paint(); });
    canvas.addEventListener("wheel", (e) => { e.preventDefault(); state.zoom = Math.max(0.5, Math.min(2.5, state.zoom * (e.deltaY < 0 ? 1.08 : 0.92))); paint(); }, { passive: false });
    canvas.addEventListener("keydown", (e) => { const k = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.08], ArrowDown: [0, 0.08] }[e.key]; if (k) { e.preventDefault(); state.yaw += k[0]; state.pitch = Math.max(0.05, Math.min(1.5, state.pitch + k[1])); paint(); } });
    if (window.ResizeObserver) new ResizeObserver(paint).observe(canvas);
    paint();
    return { redraw: paint, state };
  }
  window.Surface3D = { draw };
})();
