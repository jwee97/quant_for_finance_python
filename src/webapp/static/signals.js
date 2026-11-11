/* The Signals tab: follow rules on stocks and ETFs, futures, crypto and options, read them all with one button (or once a day on a schedule), see what each rule says about each asset, what changed
   since the last data day, and how the signals issued before have done since. Every run is appended to a ledger on the app's computer; the record is read from that ledger and from nothing else,
   so it only ever holds signals that were written down before their outcome existed. It talks to /api/signals/*; a run is a job on the app's worker.
   Whatever the server sends goes in with textContent. */
(function () {
  "use strict";
  const Q = window.QL, W = window.WS, h = Q.h, css = Q.css, num = Q.num, table = Q.simpleTable, api = Q.api;
  const GUIDE = "signals";
  const isOpen = () => !Q.$(S.view).hidden;
  const tickersNow = () => Q.S.tickers.filter((x) => x.status !== "bad").map((x) => x.t);
  const ARROW = { long: "▲", short: "▼", flat: "–" };                                             // a side is never colour alone
  const WORD = { long: "long", short: "short", flat: "no view" };
  const pct = (v, d = 2) => (Q.bad(v) ? "n/a" : (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%");
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

  const S = {
    id: "signals", root: "#signals", view: "#view-signals", title: "Signals", api: 17, mode: "signals", busy: "", results: {}, errors: {}, guard: null, go: null, progress: null,
    about: "Follow rules on your markets, read them every day, see what changed and keep an honest record of every signal issued.",
    modes: [["signals", "Signal book"]], runLabel: { signals: "Read the signals now" },
    blurb: { signals: "Each run writes what every rule said into a ledger. The record below it is scored only on prices that came after each signal was written down." },
    plan: null, book: null, planError: "", loading: false, note: "",
    form: { market: "equity", name: "", roots: "", data: "simulated", underlyings: "SPY", refresh: false, time: "18:30", enabled: false, weekdays_only: true, sched_refresh: true, webhook: "" },
  };

  /* ---------- loading ---------- */
  function ensure() { if (!S.plan && !S.planError && !S.loading) setTimeout(load, 0); }
  async function load() {
    S.loading = true; S.planError = "";
    try {
      [S.plan, S.book] = await Promise.all([api("/api/signals/plan"), api("/api/signals/book")]);
      const sc = S.book.settings.schedule; Object.assign(S.form, { time: sc.time, enabled: sc.enabled, weekdays_only: sc.weekdays_only, sched_refresh: sc.refresh });
      if (S.book.latest && !S.results.signals) S.results.signals = { data: S.book.latest, body: {} };
    } catch (e) { S.planError = e.status === 404 ? "The app behind this page has no signal book: it is older than the page. " + Q.RESTART : e.message; }
    S.loading = false;
    if (isOpen()) { Q.renderLab(S); S.draw(); }
  }
  async function reloadBook() { try { S.book = await api("/api/signals/book"); } catch (e) { Q.toast(e.message); } if (isOpen()) Q.renderLab(S); }

  /* ---------- the form ---------- */
  S.check = () => {
    if (S.planError) return S.planError;
    if (!S.plan || !S.book) return "Loading the signal book…";
    if (!S.book.follows.length) return "Follow a market first (on the left).";
    return "";
  };
  S.sections = (panel) => {
    ensure();
    const f = S.form, sec = (title) => { const d = h("details", "", panel); d.open = true; h("summary", "", d, title); return h("div", "body", d); };
    if (S.planError) { h("p", "hint tone-crit", sec("Not available"), S.planError); return; }
    if (!S.plan || !S.book) { h("p", "hint", sec("The signal book"), "Loading…"); return; }
    const marketName = (id) => (S.plan.markets.find((m) => m.id === id) || {}).label || id;

    const a = sec("1. What you follow"); a.dataset.role = "signals-follows";
    if (!S.book.follows.length) h("p", "hint", a, "Nothing yet. Follow a market below: its rules are then read every time you press the button, and written into the ledger.");
    const ul = h("ul", "runlist", a);
    for (const fo of S.book.follows) {
      const li = h("li", "", ul); li.dataset.follow = fo.id;
      const main = h("div", "main", li); h("b", "", main, fo.name);
      h("span", "meta", main, `${marketName(fo.market)} · ${plural(fo.rules.length, "rule")} · ${plural(fo.snapshots, "snapshot")} in the ledger`);
      const rm = h("button", "btn small", li, "Stop following"); rm.type = "button"; rm.dataset.action = "signals-unfollow"; rm.title = "The snapshots already issued stay in the ledger";
      rm.addEventListener("click", async () => { rm.disabled = true; try { await api("/api/signals/unfollow", { id: fo.id }); } catch (e) { Q.toast(e.message); } await reloadBook(); Q.syncLab(S); });
    }

    const b = sec("2. Follow something new");
    Q.labField(b, { key: "market", label: "Market", kind: "choice", options: S.plan.markets.map((m) => [m.id, m.label]) }, f, () => Q.renderLab(S));
    Q.labField(b, { key: "name", label: "Name", kind: "text", placeholder: f.market === "equity" ? "My ETFs" : marketName(f.market) }, f, () => {});
    if (f.market === "equity") {
      const names = tickersNow().slice(0, S.plan.max_tickers);
      h("p", "hint", b, `The Backtest tab's tickers, data source and start date: ${names.length ? names.join(", ") : "none chosen"}. Change them there before following; a follow keeps the list it was made with.`).dataset.role = "signals-tickers";
    } else if (f.market === "futures") {
      Q.labField(b, { key: "data", label: "Data", kind: "choice", options: [["simulated", "Simulated roots"], ["real", "Curves I downloaded"]], help: "Simulated roots show how the board reads; they are not a market. Real curves come from the Futures tab's Free data card or your files." }, f, () => Q.renderLab(S));
      Q.labField(b, { key: "roots", label: "Roots (empty: all the simulated ones)", kind: "text", placeholder: f.data === "real" ? "VX, BTC" : S.plan.futures_roots.slice(0, 4).join(", ") }, f, () => {});
    } else if (f.market === "crypto") {
      h("p", "hint", b, `${S.plan.crypto_assets.join(" and ")}, on the platform's Deribit data (spot, perpetual and funding).`);
    } else {
      Q.labField(b, { key: "underlyings", label: "Underlyings", kind: "text", placeholder: "SPY, QQQ", help: "As the Options tab names them: an ETF of the platform (a model chain on its real price), dl:NAME for a chain you downloaded, file:NAME for your own file." }, f, () => {});
      h("p", "hint", b, "Option signals are about volatility (own options, or sell them), and are not scored later: that needs a history of option prices.");
    }
    h("p", "hint", b, `Rules: ${S.plan.boards[f.market].map((r) => r.label).join("; ")}.`);
    const add = h("button", "btn", b, "Follow"); add.type = "button"; add.dataset.action = "signals-follow";
    add.addEventListener("click", async () => {
      const list = (t) => t.split(",").map((x) => x.trim()).filter(Boolean);
      const u = Q.universeBody();
      const config = f.market === "equity" ? { tickers: tickersNow().slice(0, S.plan.max_tickers), source: u.source, start: u.start, classes: u.classes }
        : f.market === "futures" ? { data: f.data, roots: list(f.roots) } : f.market === "options" ? { underlyings: list(f.underlyings) } : {};
      add.disabled = true;
      try { await api("/api/signals/follow", { name: f.name.trim() || (f.market === "equity" ? "My tickers" : marketName(f.market)), market: f.market, config }); f.name = ""; await reloadBook(); Q.syncLab(S); }
      catch (e) { Q.toast(e.message); add.disabled = false; }
    });

    const c = sec("3. Every day");
    const st = S.book.settings;
    Q.labField(c, { key: "enabled", label: "Read the signals once a day", kind: "bool" }, f, () => {});
    Q.labField(c, { key: "time", label: "At (this computer's clock, HH:MM)", kind: "text", placeholder: "18:30" }, f, () => {});
    Q.labField(c, { key: "weekdays_only", label: "Weekdays only", kind: "bool" }, f, () => {});
    Q.labField(c, { key: "sched_refresh", label: "Download new prices first", kind: "bool" }, f, () => {});
    Q.labField(c, { key: "webhook", label: "Send what changed to (optional)", kind: "text", placeholder: st.webhook_set ? `set: ${st.webhook_host} (type a new one to replace it)` : "https://… a webhook address of yours",
      help: "A web address that accepts a JSON message (Slack, Discord and most chat tools give you one). The changes are sent there after each daily run. It is stored on this computer only." }, f, () => {});
    const row = h("div", "row tight", c);
    const save = h("button", "btn", row, "Save"); save.type = "button"; save.dataset.action = "signals-save-settings";
    save.addEventListener("click", async () => {
      const body = { schedule: { enabled: !!f.enabled, time: f.time.trim(), weekdays_only: !!f.weekdays_only, refresh: !!f.sched_refresh } };
      if (f.webhook.trim()) body.webhook = f.webhook.trim();
      try { S.book.settings = await api("/api/signals/settings", body); f.webhook = ""; S.note = "Saved."; Q.renderLab(S); } catch (e) { Q.toast(e.message); }
    });
    if (st.webhook_set) {
      const test = h("button", "btn small", row, "Send a test alert"); test.type = "button"; test.dataset.action = "signals-test-alert";
      test.addEventListener("click", async () => { try { await api("/api/signals/alert/test", {}); Q.toast("Sent."); } catch (e) { Q.toast(e.message); } });
      const clear = h("button", "btn small", row, "Remove the address"); clear.type = "button";
      clear.addEventListener("click", async () => { try { S.book.settings = await api("/api/signals/settings", { webhook: "" }); Q.renderLab(S); } catch (e) { Q.toast(e.message); } });
    }
    const state = h("p", "hint", c, st.schedule.enabled ? `On: every ${st.schedule.weekdays_only ? "weekday" : "day"} at ${st.schedule.time}${st.last_scheduled ? `, last started ${st.last_scheduled}` : ""}. It runs only while this app is running; for a computer that is not always on, put "quant signals run --refresh" in its scheduler instead.` : "Off. " + (S.note || ""));
    state.dataset.role = "signals-schedule";
    S.note = "";

    const d = sec("4. This run");
    Q.labField(d, { key: "refresh", label: "Download new prices first", kind: "bool", help: "Without it the run reads the prices the app already has, which may be days old; the result says how old." }, f, () => {});
  };

  /* ---------- running ---------- */
  S.run = async () => {
    if (S.busy) return;
    const problem = S.check(); if (problem) { Q.toast(problem); return; }
    S.busy = "Starting…"; S.results.signals = null; S.errors.signals = null; Q.syncLab(S); S.draw();
    try {
      const started = await api("/api/signals/run", { refresh: !!S.form.refresh });
      const result = await Q.waitJob(started.job, (v) => { S.busy = `${v.message} (${Math.round(v.elapsed)} s)`; Q.syncLab(S); const el = Q.$('[data-role="signals-progress"]'); if (el) el.textContent = S.busy; }, 700);
      S.results.signals = { data: result, body: {} };
    } catch (e) { S.errors.signals = e.status === 404 ? "The app behind this page does not know this request: it is older than the page. " + Q.RESTART : e.message; }
    S.busy = ""; Q.syncLab(S); await reloadBook(); if (isOpen()) { S.draw(); window.scrollTo({ top: 0 }); }
  };
  S.draw = () => Q.drawLab(S);
  S.empty = (root) => {
    const c = h("div", "card empty", root);
    if (S.busy) { h("div", "spinner", c).style.margin = "0 auto 12px"; h("h2", "", c, "Reading the signals…"); h("p", "", c, S.busy).dataset.role = "signals-progress"; return; }
    h("h2", "", c, "One page for what your rules say today");
    for (const t of ["Follow a market on the left: your tickers, futures roots, the two coins or option underlyings, with the rules the platform has for each (trend, carry, reversal, volatility and others).",
      "Press the button, or switch on the daily run. Each run shows what every rule says about every asset, flags data that is old or simulated, lists what changed since the last data day, and writes it all into a ledger.",
      "The record is scored from that ledger only: a signal read at one close is entered at the next and compared with always holding the asset. It starts empty, because a record you can trust cannot be made from the past."]) h("p", "", c, t);
    const g = h("button", "link", c, "Read the guide"); g.type = "button"; g.addEventListener("click", () => Q.openGuide(GUIDE));
  };

  /* ---------- the result ---------- */
  function followCard(root, e) {
    const fo = e.follow, card = W.card(root, fo.name, ""); card.dataset.role = "signals-card"; card.dataset.follow = fo.id;
    if (e.error) { const p = h("p", "tone-crit", card, `Could not be read: ${e.error}`); p.dataset.role = "signals-error"; return; }
    const snap = e.snapshot, rows = snap.rows;
    h("p", "hint", card, `Data to ${snap.as_of} (${snap.source}) · ${plural(rows.length, "signal")} · snapshot ${e.snapshots} of this follow`);
    for (const p of snap.data.problems) { const w = h("p", "tone-warn", card, "! " + p); w.dataset.role = "signals-problem"; }
    const assets = [...new Set(rows.map((r) => r.asset))], rules = [...new Map(rows.map((r) => [r.rule, r.label])).entries()];
    const cell = new Map(rows.map((r) => [r.rule + "|" + r.asset, r]));
    h("h4", "", card, "What the rules agree on");
    table(card, [{ label: "Asset", get: (c) => c.asset }, { label: "View", get: (c) => c.view }, { label: "Long", num: true, get: (c) => c.long }, { label: "Short", num: true, get: (c) => c.short }, { label: "No view", num: true, get: (c) => c.flat },
      { label: "Average strength", num: true, title: "1 is a rule's typical position; the sign is the side", get: (c) => (Q.bad(c.mean_strength) ? "n/a" : num(c.mean_strength, 2)) }], e.consensus);
    h("h4", "", card, "Every rule");
    table(card, [{ label: "Rule", get: (r) => r[1] }, ...assets.map((a) => ({ label: a, num: true, title: "Side and strength: 1 is the rule's typical position",
      get: (r) => { const x = cell.get(r[0] + "|" + a); return !x ? "" : `${ARROW[x.side]} ${Q.bad(x.strength) ? "n/a" : num(x.strength, 2)}`; } }))], rules);
    if (snap.skipped.length) { const sk = h("details", "how", card); h("summary", "", sk, `${plural(snap.skipped.length, "rule")} could not be read`); for (const x of snap.skipped) h("p", "", sk, `${x.label}: ${x.why}`); }
    h("h4", "", card, "What changed");
    const ch = h("div", "", card); ch.dataset.role = "signals-changes";
    if (!e.compared_with) h("p", "hint", ch, "This is the first snapshot of this follow: there is nothing to compare it with yet.");
    else if (!e.changes.length) h("p", "hint", ch, `Nothing changed side since the snapshot for ${e.compared_with.as_of}.`);
    else table(ch, [{ label: "Asset", get: (c) => c.asset }, { label: "Rule", get: (c) => c.label }, { label: "Was", get: (c) => `${ARROW[c.from]} ${WORD[c.from]}` }, { label: "Now", get: (c) => `${ARROW[c.to]} ${WORD[c.to]}` },
      { label: "Strength", num: true, get: (c) => (Q.bad(c.strength) ? "n/a" : num(c.strength, 2)) }], e.changes);
    h("h4", "", card, "How past signals did");
    const rc = h("div", "", card); rc.dataset.role = "signals-record";
    const rec = e.record;
    if (!rec) h("p", "hint", rc, "Not scored: a volatility signal's result needs a history of option prices, and a chain is one day's snapshot.");
    else if (!Object.keys(rec.overall).length) h("p", "hint", rc, `No signal is old enough to score yet (${plural(rec.snapshots, "data day")} in the ledger). A signal read at one close is entered at the next and needs at least one more day of prices.` + (rec.late ? ` ${plural(rec.late, "snapshot")} read on data that was already old ${rec.late === 1 ? "is" : "are"} left out: the prices they would be scored on existed before they were written down.` : ""));
    else {
      const line = (name, hz) => rec.horizons.map((d) => { const v = hz[String(d)]; return v ? `${v.n} · ${Math.round(v.hit_rate * 100)}% · ${pct(v.mean)} (${pct(v.edge)})` : "–"; });
      const data = [["Every rule", rec.overall], ...rec.rules.map((r) => [r.label, r.horizons])].map(([name, hz]) => [name, ...line(name, hz)]);
      h("p", "hint", rc, `From ${plural(rec.snapshots, "data day")} in the ledger (${rec.first} to ${rec.last}). Each cell: signals scored · share that were right · average result (and how far that is above always holding the asset). ${rec.note}${rec.late ? ` ${plural(rec.late, "snapshot")} read on old data left out.` : ""}`);
      table(rc, [{ label: "Rule", get: (r) => r[0] }, ...rec.horizons.map((d, i) => ({ label: `After ${plural(d, "day")}`, get: (r) => r[i + 1] }))], data);
    }
  }
  S.build = (root, d) => {
    const ok = d.follows.filter((e) => !e.error);
    Q.labHead(root, "Signals", `read ${d.ran_at.replace("T", " ").replace("Z", " UTC")}${d.scheduled ? " by the daily run" : ""} · ${plural(d.follows.length, "follow")} · ${plural(d.changed, "change")} since the last data day`, GUIDE);
    if (d.problems.length) Q.labBanner(root, "Look at this first. ", `${plural(d.problems.length, "thing")} about the data below: old or simulated data gives old or simulated signals.`).dataset.role = "signals-banner";
    for (const e of d.follows) followCard(root, e);
    if (S.book && S.book.alerts.length) {
      const al = W.card(root, "Alerts from the daily runs", "The last ten. Sent means the alert address accepted it.");
      table(al, [{ label: "When (UTC)", get: (x) => x.at.replace("T", " ").replace("Z", "") }, { label: "Sent", get: (x) => (x.delivered === null ? "no address" : x.delivered ? "✓ sent" : `✗ ${x.error || "failed"}`) }, { label: "What", get: (x) => x.text.split("\n")[0] }], S.book.alerts);
    }
    const how = h("details", "how", root); h("summary", "", how, "How to read it");
    h("p", "", how, `A rule's strength is its score over the average size of its own score in the last two years: 1 is a typical position, the sign is the side, and within ±${S.plan ? S.plan.strong : 0.25} it counts as no view. The view of an asset is long or short when at least half the rules net out that way.`);
    h("p", "", how, "Nothing here is advice or an order. The rules are the platform's strategies read on the last day of the data; whether any of them is worth trading is what the Backtest tab and the record on this page are for.");
    if (!ok.length) h("p", "hint", root, "No follow could be read: see the messages above.");
  };

  Q.registerWorkspace("signals", S);
})();
