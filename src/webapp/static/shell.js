/* The application shell: the grouped sidebar, the breadcrumb, the command palette, the keyboard shortcuts and the drawer a phone gets instead of the sidebar.

   The sidebar is the same set of ``role="tab"`` buttons the page always had (``#tab-<name>`` opens ``#view-<name>``); this file only groups, folds and searches them, so a link, a test or a habit that
   addresses a tab by its id still works. The list of pages, with their groups and search words, is read from index.html, so there is one place to add a page. What a person chose (folded groups, the
   narrow rail, the pages visited last) is kept in the browser under ``ql.shell``. It holds view preferences only: no result, no data. */
(() => {
  "use strict";
  const Q = window.QL, { store } = Q;
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const el = (tag, cls, parent, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; if (parent) parent.appendChild(e); return e; };

  /* ---------------------------------------------------------------- preferences */
  const DEFAULTS = { rail: false, folded: [], recent: [], landing: "last" };
  const prefs = () => ({ ...DEFAULTS, ...(store("shell") || {}) });
  const setPref = (key, value) => { store("shell", { ...prefs(), [key]: value }); };

  /* ---------------------------------------------------------------- the pages, read from the sidebar */
  const text = (node) => [...node.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join("").trim();
  const pages = () => $$('nav.tabs [role="tab"]').map((b) => ({ id: b.id.slice(4), label: text(b.querySelector(".lbl")), group: b.closest(".navgroup").dataset.label, groupId: b.closest(".navgroup").dataset.group, keywords: b.dataset.keywords || "" }));
  const pageOf = (id) => pages().find((p) => p.id === id);
  const go = (id, args) => { closeDrawer(); Q.showTab(id, args); const t = $("#tab-" + id); if (t && document.activeElement !== t && $("#side").contains(document.activeElement)) t.focus(); };

  /* ---------------------------------------------------------------- folding, the rail, the drawer */
  function fold(group, folded) {
    const g = $(`.navgroup[data-group="${group}"]`); if (!g) return;
    g.classList.toggle("folded", folded); $(".navhead", g).setAttribute("aria-expanded", String(!folded));
  }
  function applyPrefs() {
    const p = prefs();
    $$(".navgroup").forEach((g) => fold(g.dataset.group, p.folded.includes(g.dataset.group)));
    document.body.classList.toggle("rail", !!p.rail); $("#railtoggle").setAttribute("aria-pressed", String(!!p.rail));
  }
  $$(".navgroup .navhead").forEach((b) => b.addEventListener("click", () => {
    const g = b.closest(".navgroup"), now = !g.classList.contains("folded"), set = new Set(prefs().folded);
    fold(g.dataset.group, now); now ? set.add(g.dataset.group) : set.delete(g.dataset.group); setPref("folded", [...set]);
  }));
  const toggleRail = () => { const on = !document.body.classList.contains("rail"); setPref("rail", on); applyPrefs(); };
  $("#railtoggle").addEventListener("click", toggleRail);

  const drawerOpen = () => document.body.classList.contains("nav-open");
  function openDrawer() {
    document.body.classList.add("nav-open"); $("#navtoggle").setAttribute("aria-expanded", "true");
    const sel = $('nav.tabs [role="tab"][aria-selected="true"]'); if (sel) sel.focus();
  }
  function closeDrawer(restore) {
    if (!drawerOpen()) return;
    document.body.classList.remove("nav-open"); $("#navtoggle").setAttribute("aria-expanded", "false"); if (restore) $("#navtoggle").focus();
  }
  $("#navtoggle").addEventListener("click", () => (drawerOpen() ? closeDrawer(true) : openDrawer()));
  document.addEventListener("click", (e) => { if (drawerOpen() && !e.target.closest("#side, #navtoggle")) closeDrawer(); });
  $("#side").addEventListener("click", (e) => { if (e.target.closest('[role="tab"], .navrecent a')) closeDrawer(); });

  /* ---------------------------------------------------------------- breadcrumb and recently visited */
  let extra = "";                                                                          // a page may add a last crumb (the session it shows)
  function crumbs(tab) {
    const p = pageOf(tab), box = $("#crumbs"); box.replaceChildren(); if (!p) return;
    const ol = el("ol", "", box);
    el("li", "", ol, p.group); const last = el("li", "", ol, p.label); last.setAttribute("aria-current", "page");
    if (extra) el("li", "", ol, extra);
  }
  function recents(tab) {
    const keep = [tab, ...prefs().recent.filter((x) => x !== tab)].filter((x) => pageOf(x)).slice(0, 7);
    setPref("recent", keep);
    const list = $("#recent-list"); list.replaceChildren();
    for (const id of keep.slice(1, 6)) {
      const a = el("a", "", el("li", "", list), pageOf(id).label); a.href = "#/" + id;
      a.addEventListener("click", (e) => { e.preventDefault(); go(id); });
    }
    $("#recents").hidden = list.children.length === 0;
  }
  function onNavigate(tab) {
    const g = $("#tab-" + tab).closest(".navgroup");
    if (g.classList.contains("folded")) { fold(g.dataset.group, false); setPref("folded", prefs().folded.filter((x) => x !== g.dataset.group)); }
    extra = ""; crumbs(tab); recents(tab);
    const p = pageOf(tab); document.title = (p ? p.label + " · " : "") + "Quant Lab";
  }
  Q.onNavigate(onNavigate);

  /* ---------------------------------------------------------------- command palette */
  const sources = [];                                                                       // {name, search(q) -> Promise<[{label, hint, run}]>}: sessions and experiments add themselves here
  const actions = () => [
    { label: "Switch between dark and light", group: "Action", run: () => $("#theme").click() },
    { label: "Collapse or expand the sidebar", group: "Action", run: toggleRail },
    { label: "Show the keyboard shortcuts", group: "Action", run: () => keysDialog() },
  ];
  const score = (q, item) => {
    const label = item.label.toLowerCase(), hay = (item.label + " " + (item.group || "") + " " + (item.keywords || "")).toLowerCase();
    let s = 0;
    for (const w of q.toLowerCase().split(/\s+/).filter(Boolean)) {
      if (label.startsWith(w)) s += 4; else if (label.includes(w)) s += 3; else if (hay.includes(w)) s += 1; else return 0;
    }
    return s || (q.trim() ? 0 : 1);
  };
  let seq = 0;
  function palette() {
    const d = $("#palette"); if (d.open) return;
    d.replaceChildren();
    const box = el("div", "pal", d), input = el("input", "", box); input.type = "text"; input.placeholder = "Search pages, actions, sessions, experiments"; input.setAttribute("role", "combobox");
    input.setAttribute("aria-label", "Search pages and actions"); input.setAttribute("aria-controls", "pal-list"); input.setAttribute("aria-expanded", "true"); input.setAttribute("autocomplete", "off");
    const list = el("ul", "", box); list.id = "pal-list"; list.setAttribute("role", "listbox"); list.setAttribute("aria-label", "Results");
    el("p", "hint", box, "↑ ↓ to choose, Enter to open, Esc to close.");
    let items = [], at = 0;
    const draw = () => {
      list.replaceChildren();
      items.forEach((it, i) => {
        const li = el("li", "", list); li.id = "pal-" + i; li.setAttribute("role", "option"); li.setAttribute("aria-selected", String(i === at));
        el("span", "", li, it.label); el("small", "", li, it.hint || it.group || "");
        li.addEventListener("mousedown", (e) => { e.preventDefault(); pick(i); });
      });
      if (!items.length) el("li", "none", list, "Nothing matches.");
      input.setAttribute("aria-activedescendant", items.length ? "pal-" + at : "");
      const cur = $("#pal-" + at); if (cur) cur.scrollIntoView({ block: "nearest" });
    };
    const base = (q) => {
      const own = [...pages().map((p) => ({ ...p, hint: p.group, run: () => go(p.id) })), ...actions()];
      return own.map((it) => [score(q, it), it]).filter((x) => x[0] > 0).sort((a, b) => b[0] - a[0]).map((x) => x[1]).slice(0, 12);
    };
    const search = (q) => {
      items = base(q); at = 0; draw();
      const mine = ++seq;
      if (!q.trim()) return;
      for (const s of sources) {
        Promise.resolve().then(() => s.search(q)).then((found) => { if (mine === seq && d.open) { items = items.concat((found || []).slice(0, 6)); draw(); } }, () => {});
      }
    };
    const pick = (i) => { const it = items[i]; if (!it) return; d.close(); it.run(); };
    input.addEventListener("input", () => search(input.value));
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); at = Math.min(items.length - 1, at + 1); draw(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); at = Math.max(0, at - 1); draw(); }
      else if (e.key === "Enter") { e.preventDefault(); pick(at); }
    });
    d.addEventListener("click", (e) => { if (e.target === d) d.close(); }, { once: false });
    search(""); d.showModal(); input.focus();
  }
  $("#palette-open").addEventListener("click", palette);

  /* ---------------------------------------------------------------- shortcuts */
  const SHORTCUTS = [["Ctrl K  or  /", "Search pages and actions"], ["g then o", "Overview"], ["g then b", "Strategy Lab"], ["g then p", "Options Lab"], ["g then m", "Machine Learning Lab"], ["g then r", "Market Regime Lab"],
    ["g then c", "Research comparison"], ["g then j", "Background jobs"], ["g then d", "Data management"], ["g then s", "Settings"], ["[", "Collapse or expand the sidebar"], ["?", "This list"], ["Esc", "Close a window or the menu"]];
  const CHORDS = { o: "overview", b: "backtest", p: "options", m: "ml", r: "regime", c: "rcompare", j: "jobs", d: "data", s: "settings" };
  function keysDialog() {
    const d = $("#keys"); if (d.open) return; d.replaceChildren();
    const box = el("div", "pal", d); el("p", "navtitle", box, "Keyboard shortcuts");
    const dl = el("dl", "keys", box); for (const [k, v] of SHORTCUTS) { el("dt", "", dl).appendChild(Object.assign(document.createElement("kbd"), { textContent: k })); el("dd", "", dl, v); }
    const b = el("button", "btn", box, "Close"); b.type = "button"; b.addEventListener("click", () => d.close());
    d.addEventListener("click", (e) => { if (e.target === d) d.close(); }); d.showModal(); b.focus();
  }
  let chord = 0;
  document.addEventListener("keydown", (e) => {
    const typing = e.target.closest && e.target.closest("input, textarea, select, [contenteditable]");
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); palette(); return; }
    if (e.key === "Escape") { if (drawerOpen()) { closeDrawer(true); } return; }
    if (typing || e.ctrlKey || e.metaKey || e.altKey || $("dialog[open]")) return;
    if (e.key === "/") { e.preventDefault(); palette(); }
    else if (e.key === "?") { e.preventDefault(); keysDialog(); }
    else if (e.key === "[") { toggleRail(); }
    else if (chord && Date.now() - chord < 2500 && CHORDS[e.key]) { chord = 0; if ($("#tab-" + CHORDS[e.key])) go(CHORDS[e.key]); }
    else if (e.key === "g") chord = Date.now();
    else chord = 0;
  });

  applyPrefs();
  Q.shell = { prefs, setPref, pages, pageOf, go, palette, addSource: (s) => sources.push(s), setCrumb: (t) => { extra = t || ""; crumbs(Q.currentTab()); } };
  crumbs(Q.currentTab());
})();
