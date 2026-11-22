/* Optional libraries, loaded only if their files are in static/vendor/ (scripts/vendor_libs.py copies them from pip packages): Plotly (3D surface) and AG Grid Community (the Saved runs table).
   window.Libs.load("plotly" | "ag-grid") resolves to the global (window.Plotly, window.agGrid) or null; the callers keep their own canvas viewer and plain tables for the null case. The server tells the page which files exist (window.__LIBS__), so a page without them
   logs nothing. */
(function () {
  "use strict";
  const FILES = { plotly: ["/static/vendor/plotly.min.js", () => window.Plotly], "ag-grid": ["/static/vendor/ag-grid-community.min.js", () => window.agGrid] };
  const cache = {};
  try { window.__LIBS__ = JSON.parse((document.querySelector('meta[name=libs]') || {}).content || "{}"); } catch (e) { window.__LIBS__ = {}; }
  function load(name) {
    const spec = FILES[name];
    if (!spec) return Promise.resolve(null);
    if (spec[1]()) return Promise.resolve(spec[1]());
    if (cache[name]) return cache[name];
    if (!(window.__LIBS__ && window.__LIBS__[name])) return Promise.resolve(null);                      // the server says which files exist: no probing, no 404 in the console
    if (name === "ag-grid") for (const f of window.__LIBS__["ag-grid-css"] || []) { const l = document.createElement("link"); l.rel = "stylesheet"; l.href = "/static/vendor/" + f; document.head.appendChild(l); }
    cache[name] = new Promise((resolve) => { const s = document.createElement("script"); s.src = spec[0]; s.onload = () => resolve(spec[1]() || null); s.onerror = () => resolve(null); document.head.appendChild(s); });
    return cache[name];
  }
  window.Libs = { load };
})();
