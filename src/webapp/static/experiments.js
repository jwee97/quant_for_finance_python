/* Saved research runs, shared by the Options and Futures tabs. window.Experiments.saveButton(root, spec) keeps the run on screen (request, metrics, provenance); window.Experiments.view(root, d, ctx)
   lists the saved runs, compares two to eight of them (metrics side by side, the settings that differ, what the hashes say about reproducibility) and deletes them. Server text goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS;
  if (!Q || !W) return;
  const h = Q.h, table = Q.simpleTable;
  const metricsOf = {};                                              // kind -> (the answer of the stored request) -> the numbers a saved run keeps; each view registers its own, so saving and repeating use one definition

  /* Send the stored request again and judge the answer against the saved numbers. */
  async function repeat(id) {
    const rec = await Q.api("/api/experiments/get?id=" + encodeURIComponent(id)), req = rec.request, fn = metricsOf[rec.kind];
    if (!fn) throw new Error(`no way to repeat a "${rec.kind}" run in this page`);
    let res = req.body ? await Q.api(req.path, req.body) : await Q.api(req.path + (req.query && Object.keys(req.query).length ? "?" + new URLSearchParams(Object.entries(req.query).filter(([, v]) => v !== undefined && v !== null)).toString() : ""));
    if (res && res.job) res = await Q.waitJob(res.job, () => {}, 1000);
    return Q.api("/api/experiments/check", { id, metrics: fn(res) });
  }

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
    const out = h("div", "", root), byId = new Map(d.experiments.map((r) => [r.id, r]));
    const doRepeat = async (r, btn) => {
      if (btn) btn.disabled = true; out.replaceChildren();
      try {
        const c = await repeat(r.id), box = h("div", "card", out); box.dataset.role = "repeat-result"; h("h3", "", box, `Repeat of "${r.name}": ${c.reproduced ? "reproduced" : "differs"}`);
        h("p", "hint", box, c.reading);
        if (c.differences.length) table(box, [{ label: "Metric", get: (x) => x.metric }, { label: "Saved", num: true, get: (x) => (x.stored === null ? "missing" : Q.num(x.stored, 6)) }, { label: "Repeat", num: true, get: (x) => (x.repeat === null ? "missing" : Q.num(x.repeat, 6)) }], c.differences);
        h("p", "note", box, `Software then: ${JSON.stringify(c.software_then)}; now: ${JSON.stringify(c.software_now)}`);
      } catch (e) { h("p", "note", out, e.message); } finally { if (btn) btn.disabled = false; }
    };
    const doDelete = async (r) => { try { await Q.api("/api/experiments/delete", { id: r.id }); Q.renderLab(ctx.lab); } catch (e) { out.textContent = e.message; } };
    const doCompare = async () => {
      out.replaceChildren();
      try {
        const c = await Q.api("/api/experiments/compare", { ids: [...chosen] });
        const box = h("div", "card", out); box.dataset.role = "comparison"; h("h3", "", box, "Comparison");
        for (const w of c.warnings) h("p", "note", box, w);
        table(box, [{ label: "Metric", get: (r) => r.metric }, ...c.experiments.map((e, j) => ({ label: e.name, num: true, get: (r) => (r.values[j] === null || r.values[j] === undefined ? "n/a" : Q.num(r.values[j], 4)) }))], c.metrics);
        if (c.setting_differences.length) { h("h3", "", box, "Settings that differ"); table(box, [{ label: "Setting", get: (r) => r.setting }, ...c.experiments.map((e, j) => ({ label: e.name, get: (r) => String(r.values[j]) }))], c.setting_differences); }
      } catch (e) { h("p", "note", out, e.message); }
    };
    const plain = () => {                                                                       // the built-in table: a tick box, Repeat and Delete on each row
      table(card, [{ label: "", get: () => "" }, { label: "Name", get: (r) => r.name }, { label: "Kind", get: (r) => r.kind.replace(/_/g, " ") }, { label: "Saved", get: (r) => r.created_at.replace("T", " ").replace("Z", " UTC") },
        { label: "Data", get: (r) => (r.provenance && r.provenance.kind) || "n/a" }, { label: "Config", get: (r) => r.config_hash }, { label: "Result", get: (r) => r.result_hash }], d.experiments);
      const rows = card.querySelectorAll("tbody tr");
      d.experiments.forEach((r, i) => {
        const tr = rows[i]; if (!tr) return;
        const cb = h("input", "", tr.firstChild); cb.type = "checkbox"; cb.setAttribute("aria-label", "compare " + r.name); cb.dataset.id = r.id;
        cb.addEventListener("change", () => { if (cb.checked) chosen.add(r.id); else chosen.delete(r.id); });
        const rep = h("button", "btn", tr.lastChild, "Repeat"); rep.type = "button"; rep.dataset.role = "repeat"; rep.addEventListener("click", () => doRepeat(r, rep));
        const del = h("button", "btn", tr.lastChild, "Delete"); del.type = "button"; del.addEventListener("click", () => doDelete(r));
      });
      const cmp = h("button", "primary", card, "Compare the ticked runs"); cmp.dataset.role = "compare"; cmp.addEventListener("click", doCompare);
    };
    const holder = h("div", "", card);
    const tools = () => {                                                                       // AG Grid: sort, filter and select in the grid; the buttons act on the selection
      const bar = h("div", "row tight", card);
      for (const [label, role, fn] of [["Compare the selected runs", "compare", doCompare], ["Repeat the first selected", "repeat", () => { const r = byId.get([...chosen][0]); if (r) doRepeat(r); }],
        ["Delete the selected", "delete", async () => { for (const id of [...chosen]) await doDelete(byId.get(id)); }]]) { const b = h("button", role === "compare" ? "primary" : "btn", bar, label); b.dataset.role = role; b.addEventListener("click", fn); }
    };
    if (window.Libs) window.Libs.load("ag-grid").then((ag) => {
      if (!ag || !ag.createGrid) { plain(); return; }
      holder.style.cssText = "width:100%"; holder.dataset.grid = "ag-grid"; holder.classList.add("ag-theme-quartz");
      ag.createGrid(holder, { rowData: d.experiments, domLayout: "autoHeight", rowSelection: "multiple", suppressRowClickSelection: false,
        columnDefs: [{ headerName: "Name", field: "name", checkboxSelection: true, headerCheckboxSelection: true, filter: true, flex: 2 }, { headerName: "Kind", valueGetter: (p) => p.data.kind.replace(/_/g, " "), filter: true },
          { headerName: "Saved", valueGetter: (p) => p.data.created_at.replace("T", " ").replace("Z", " UTC"), sort: "desc", flex: 1.5 }, { headerName: "Data", valueGetter: (p) => (p.data.provenance && p.data.provenance.kind) || "n/a", filter: true },
          { headerName: "Config", field: "config_hash" }, { headerName: "Result", field: "result_hash" }],
        onSelectionChanged: (e) => { chosen.clear(); for (const n of e.api.getSelectedNodes()) chosen.add(n.data.id); } });
      tools();
    }); else plain();
  }

  window.Experiments = { saveButton, view, repeat, metricsOf };
})();
