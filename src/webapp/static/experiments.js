/* Saved research runs, shared by the Options and Futures tabs. window.Experiments.saveButton(root, spec) keeps the run on screen (request, metrics, provenance); window.Experiments.view(root, d, ctx)
   lists the saved runs, compares two to eight of them (metrics side by side, the settings that differ, what the hashes say about reproducibility) and deletes them. Server text goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS;
  if (!Q || !W) return;
  const h = Q.h, table = Q.simpleTable;

  function saveButton(root, spec) {
    const box = h("div", "card", root); box.dataset.role = "save-experiment"; h("h3", "", box, "Save this run");
    h("p", "hint", box, "Keeps the settings, the headline numbers, where the data came from and the software version, so the run can be repeated and compared with another (the Saved runs view).");
    const name = h("input", "", box); name.type = "text"; name.value = spec.name; name.setAttribute("aria-label", "experiment name");
    const btn = h("button", "primary", box, "Save"), out = h("span", "hint", box);
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        const rec = await Q.api("/api/experiments/save", { kind: spec.kind, name: name.value, request: spec.request, metrics: spec.metrics, provenance: spec.provenance, universe: spec.universe || [], assumptions: spec.assumptions || [] });
        out.textContent = ` Saved as ${rec.id} (config ${rec.config_hash}, result ${rec.result_hash}).`;
      } catch (e) { out.textContent = " " + (e && e.message ? e.message : String(e)); } finally { btn.disabled = false; }
    });
    return box;
  }

  function view(root, d, ctx) {
    ctx.header(root, "Saved runs", `${d.experiments.length} saved · stored on this machine in data/user/experiments/`);
    const chosen = new Set(), card = h("div", "card", root); card.dataset.role = "experiments";
    if (!d.experiments.length) { h("p", "hint", card, "Nothing saved yet. Run a Spread lab, a model validation or a strategy and press Save this run under the result."); return; }
    const out = h("div", "", root);
    table(card, [{ label: "", get: (r) => "" }, { label: "Name", get: (r) => r.name }, { label: "Kind", get: (r) => r.kind.replace(/_/g, " ") }, { label: "Saved", get: (r) => r.created_at.replace("T", " ").replace("Z", " UTC") },
      { label: "Data", get: (r) => (r.provenance && r.provenance.kind) || "n/a" }, { label: "Config", get: (r) => r.config_hash }, { label: "Result", get: (r) => r.result_hash }], d.experiments);
    const rows = card.querySelectorAll("tbody tr");
    d.experiments.forEach((r, i) => {
      const tr = rows[i]; if (!tr) return;
      const cb = h("input", "", tr.firstChild); cb.type = "checkbox"; cb.setAttribute("aria-label", "compare " + r.name); cb.dataset.id = r.id;
      cb.addEventListener("change", () => { if (cb.checked) chosen.add(r.id); else chosen.delete(r.id); });
      const del = h("button", "btn", tr.lastChild, "Delete"); del.type = "button";
      del.addEventListener("click", async () => { try { await Q.api("/api/experiments/delete", { id: r.id }); Q.renderLab(ctx.lab); } catch (e) { out.textContent = e.message; } });
    });
    const cmp = h("button", "primary", card, "Compare the ticked runs"); cmp.dataset.role = "compare";
    cmp.addEventListener("click", async () => {
      out.replaceChildren();
      try {
        const c = await Q.api("/api/experiments/compare", { ids: [...chosen] });
        const box = h("div", "card", out); box.dataset.role = "comparison"; h("h3", "", box, "Comparison");
        for (const w of c.warnings) h("p", "note", box, w);
        table(box, [{ label: "Metric", get: (r) => r.metric }, ...c.experiments.map((e, j) => ({ label: e.name, num: true, get: (r) => (r.values[j] === null || r.values[j] === undefined ? "n/a" : Q.num(r.values[j], 4)) }))], c.metrics);
        if (c.setting_differences.length) { h("h3", "", box, "Settings that differ"); table(box, [{ label: "Setting", get: (r) => r.setting }, ...c.experiments.map((e, j) => ({ label: e.name, get: (r) => String(r.values[j]) }))], c.setting_differences); }
      } catch (e) { h("p", "note", out, e.message); }
    });
  }

  window.Experiments = { saveButton, view };
})();
