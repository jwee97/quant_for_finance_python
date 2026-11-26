/* Permutation importance: "what breaks when a feature is shuffled".

   The value of a feature is the DROP in out-of-sample rank IC when its values are shuffled across the rows of a refit's test block, averaged over refits. Positive: the predictions need the feature. Negative:
   shuffling it improved the metric (the model leaned on noise). The chart is diverging (zero line, both signs), sorted, limited to the top N, with approximate intervals where the server computed them, and a
   Table view with the same rows. When the drop is undefined for every feature (for example a universe of two assets, where a cross-sectional rank IC does not exist) the card says so and why instead of
   drawing an empty axis. Rows whose drop is undefined are left out of the chart AND the table (so the two always show the same rows), and the card says how many. Whatever the server sends is written with textContent. */
(() => {
  "use strict";
  const Q = window.QL, h = Q.h, C = Q.C;
  const finite = (v) => typeof v === "number" && isFinite(v);
  const fmt3 = (v) => (finite(v) ? v.toFixed(3) : "n/a");
  const css = Q.css;
  const EXPLAIN = [
    ["What it measures", "How much the model's out-of-sample rank IC falls when one feature's values are shuffled across assets, with the model left as it is. The bigger the fall, the more the predictions need that feature."],
    ["Positive bar", "Shuffling the feature made the predictions worse: the model relies on it to rank assets."],
    ["Negative bar", "Shuffling the feature made the metric better: the model was leaning on noise in that feature. Small negative bars near zero are what chance looks like."],
    ["Correlated features", "If two features carry the same information, shuffling one leaves the other to compensate, so each looks less important than the pair is. Read groups of related features together."],
    ["Not causality", "It describes this fitted model on this sample. It does not say the feature moves returns, and it is not a reason to keep a feature."],
  ];

  function rowsOf(perm) {
    return (perm || []).map((x) => ({ feature: String(x.feature), mean: finite(x.mean_drop) ? x.mean_drop : null, std: finite(x.std_drop) ? x.std_drop : null, n: x.n, lo: finite(x.ci_low) ? x.ci_low : null, hi: finite(x.ci_high) ? x.ci_high : null }));
  }

  /* ch: the container; perm: the server's rows; status: the server's permutation_status (may be absent for an older experiment); opts.top: how many to show */
  function card(ch, perm, status, opts) {
    opts = opts || {};
    const rows = rowsOf(perm), valid = rows.filter((r) => r.mean !== null);
    const wrap = h("div", "permcard", ch); wrap.dataset.role = "permutation";
    if (!valid.length) return empty(wrap, rows, status);
    let top = Math.min(opts.top || 15, valid.length);
    const sorted = valid.slice().sort((a, b) => b.mean - a.mean);
    const note = "How far out-of-sample rank IC falls when one feature is shuffled, averaged over refits. Positive: the predictions need it. Negative: shuffling it helped, so the model is leaning on noise.";
    const c = Q.chartCard(wrap, "What breaks when a feature is shuffled", note, {
      label: "Permutation importance",
      draw: (b) => {
        const shown = sorted.slice(0, top).concat(sorted.slice(Math.max(top, sorted.length - Math.min(3, Math.max(sorted.length - top, 0))))), seen = [];                       // the top N, and the three most negative that did not make it, so noise reliance is visible
        for (const r of shown) if (!seen.includes(r)) seen.push(r);
        const longest = Math.max(...seen.map((r) => r.feature.length));
        C.hbars(b, { rows: seen.map((r) => ({ label: r.feature, value: r.mean, lo: r.lo === null ? undefined : r.lo, hi: r.hi === null ? undefined : r.hi,
          tip: [["Spread across refits", r.std === null ? "n/a" : fmt3(r.std)], ["Refits × shuffles used", String(r.n)], ...(r.lo !== null ? [["Approx. 95% interval", `${fmt3(r.lo)} to ${fmt3(r.hi)}`]] : [])] })),
          format: (v) => v.toFixed(3), name: "Drop in rank IC", label: "Permutation importance", labelWidth: Math.min(150, Math.max(74, longest * 7.2)), zero: true });
      },
      table: (b) => Q.simpleTable(b, [{ label: "Feature", get: (x) => x.feature }, { label: "Drop in rank IC", num: true, get: (x) => (x.mean === null ? "n/a" : x.mean.toFixed(4)) }, { label: "Spread", num: true, get: (x) => (x.std === null ? "n/a" : x.std.toFixed(4)) },
        { label: "Approx. 95% low", num: true, get: (x) => (x.lo === null ? "" : x.lo.toFixed(4)) }, { label: "Approx. 95% high", num: true, get: (x) => (x.hi === null ? "" : x.hi.toFixed(4)) }, { label: "Used", num: true, get: (x) => String(x.n) }], sorted),
    });
    if (valid.length > 5) {                                                                      // how many features to chart
      const sel = h("select", "", c._extra); sel.dataset.role = "perm-top"; sel.setAttribute("aria-label", "Features shown");
      for (const n of [...new Set([5, 10, 15, 20, valid.length].filter((v) => v <= valid.length))]) { const o = h("option", "", sel, n === valid.length ? `All ${n}` : `Top ${n}`); o.value = String(n); if (n === top) o.selected = true; }
      sel.addEventListener("change", () => { top = Number(sel.value); c._draw(); });
    }
    const dropped = rows.length - valid.length;
    if (dropped) h("p", "hint", wrap, `${dropped} feature${dropped === 1 ? " has" : "s have"} no defined drop and ${dropped === 1 ? "is" : "are"} not charted.${status && status.reason ? " " + status.reason + "." : ""}`);
    if (status && status.interval) h("p", "hint", wrap, "Interval: " + status.interval + ".");
    const d = h("details", "how", wrap); h("summary", "", d, "How to read this chart");
    const ul = h("ul", "", d); for (const [k, v] of EXPLAIN) { const li = h("li", "", ul); h("b", "", li, k + ". "); li.appendChild(document.createTextNode(v)); }
    if (status && status.sample) h("p", "hint", d, "Sample: " + status.sample + ".");
    return c;
  }

  function empty(wrap, rows, status) {
    const c = h("div", "card chartcard permempty", wrap); c.dataset.role = "permutation-empty";
    h("h3", "", c, "What breaks when a feature is shuffled");
    const reason = (status && status.reason) || "The drop in rank IC is undefined for every feature, so there is nothing to chart.";
    const b = h("div", "banner", c); h("strong", "", b, "No permutation importance. "); b.appendChild(document.createTextNode(reason + "."));
    h("p", "hint", c, "A rank IC compares how the model ranked the assets with how they actually ranked on the same day. With fewer than three assets on a date that comparison is meaningless, so the platform does not draw one.");
    return c;
  }

  window.QLImportance = { card, rowsOf };
})();
