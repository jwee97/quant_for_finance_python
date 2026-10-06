"""``quant dashboard``: one self-contained HTML page that lets you explore everything the platform has recorded.

It reads only files the pipeline already wrote (the decision registry, the strategy-library table, the experiment database if you have one,
and the figure captions), embeds them as JSON and draws with plain JavaScript, so the page works offline with no server. Figures are linked by
relative path, so keep ``dashboard.html`` next to the ``figures`` folder (the default ``reports/dashboard.html`` does).

Tabs: the decision ledger (every hypothesis, retained or rejected), the strategy library leaderboard with parameter sensitivity, your own runs,
the information-coefficient decay, walk-forward folds, risk, a figure gallery, and what the platform's tests could detect.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.replace([np.inf, -np.inf], np.nan).to_json(orient="records"))


def _parse_name(name: str) -> tuple[str, dict]:
    m = re.match(r"^(?P<base>[^\[]+)(\[(?P<params>.*)\])?$", str(name))
    params = {}
    if m and m.group("params"):
        for part in m.group("params").split(","):
            k, _, v = part.partition("=")
            try:
                params[k.strip()] = float(v)
            except ValueError:
                params[k.strip()] = v.strip()
    return (m.group("base") if m else str(name)), params


def _parsed_entry(name, family, sharpe, cagr, vol, max_drawdown, turnover) -> dict:
    base, params = _parse_name(name)
    return {"name": name, "base": base, "params": params, "family": family, "sharpe": sharpe, "cagr": cagr, "vol": vol, "max_drawdown": max_drawdown, "turnover": turnover}


def _all_runs(config) -> list[dict]:
    db = config.root / str(config.get("framework.experiment_db.path", "data/experiments.db"))
    if not db.exists():
        return []
    import sqlite3

    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT name, sharpe, cagr, ann_vol, max_drawdown, ann_turnover FROM runs", con)
    return _records(frame)


def collect(config) -> dict:
    root, tables, figures = config.root, config.reports_dir("tables"), config.reports_dir("figures")
    ledger = []
    registry = root / "experiments" / "registry.jsonl"
    if registry.exists():
        for line in registry.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            results = {k: v for k, v in (d.get("results") or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            ledger.append({"id": d["experiment_id"], "stage": d.get("stage", ""), "hypothesis": d["hypothesis"], "decision": d["decision"], "notes": d.get("notes", ""),
                           "results": dict(list(results.items())[:6]), "date": d.get("date", "")[:10]})
    specs = _read(tables / "stage30_specs.csv", index_col=0)
    library = [_parsed_entry(name, row.get("family", ""), row.get("sharpe"), row.get("cagr"), row.get("ann_vol"), row.get("max_drawdown"), row.get("ann_turnover"))
               for name, row in specs.iterrows()]
    runs, curves = [], {}
    db = config.root / str(config.get("framework.experiment_db.path", "data/experiments.db"))
    if db.exists():
        from .experiments import ExperimentManager
        manager = ExperimentManager(config)
        board = manager.leaderboard(None, 50, "sharpe")
        runs = _records(board)
        for run_id in board["run_id"].head(8):
            try:
                net = manager.returns_of(run_id)["net"].dropna()
                curve = (1 + net).cumprod().resample("ME").last()
                curves[run_id] = {"x": [d.strftime("%Y-%m") for d in curve.index], "y": [round(float(v), 4) for v in curve]}
            except Exception:
                pass
    for r in _all_runs(config):                                      # your own sweeps appear in the parameter-sensitivity chart next to the library
        library.append(_parsed_entry(r["name"], "your runs", r["sharpe"], r["cagr"], r["ann_vol"], r["max_drawdown"], r["ann_turnover"]))
    ic = _read(tables / "stage05_ic_decay_all.csv")
    walk = _read(tables / "stage11_walk_forward_summary.csv", index_col=0)
    risk = _read(tables / "stage09_risk_summary.csv", index_col=0)
    gallery = []
    for png in sorted(figures.glob("fig*.png")):
        caption = png.with_suffix(".txt")
        gallery.append({"file": png.name, "caption": caption.read_text(encoding="utf-8").strip().splitlines()[0] if caption.exists() else png.stem})
    power = {"sharpe_mde": _records(_read(tables / "stage39_sharpe_mde.csv")), "forecast_mde": _records(_read(tables / "stage39_forecast_mde.csv")),
             "context": _records(_read(tables / "stage39_earlier_results_in_context.csv"))}
    return {"ledger": ledger, "library": library, "runs": runs, "curves": curves, "ic": _records(ic[["signal", "horizon", "mean_ic", "t_stat_overlap_adjusted"]]) if len(ic) else [],
            "walk_forward": _records(walk.reset_index().rename(columns={"index": "book"})[[c for c in ("book", "n_folds", "folds_positive", "fold_sharpe_mean", "sharpe") if c in walk.reset_index().columns]]) if len(walk) else [],
            "risk": _records(risk.reset_index().rename(columns={"index": "book"})) if len(risk) else [], "gallery": gallery, "power": power}


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quant Research Dashboard</title>
<style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--muted:#6b6b66;--card:#fff;--line:#dcdcd6;--accent:#0072B2;--retain:#009E73;--reject:#D55E00;--record:#888}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--fg:#ecece8;--muted:#9a9a94;--card:#1f1f1d;--line:#34342f;--accent:#56B4E9;--retain:#2cc9a0;--reject:#f0884a;--record:#aaa}}
:root[data-theme=dark]{--bg:#161615;--fg:#ecece8;--muted:#9a9a94;--card:#1f1f1d;--line:#34342f;--accent:#56B4E9;--retain:#2cc9a0;--reject:#f0884a;--record:#aaa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{padding:20px 16px 8px;max-width:1200px;margin:auto}h1{margin:0;font-size:22px}p.sub{margin:4px 0 0;color:var(--muted)}
nav{display:flex;flex-wrap:wrap;gap:6px;max-width:1200px;margin:12px auto;padding:0 16px}
nav button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:6px 12px;border-radius:6px;cursor:pointer}nav button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
main{max-width:1200px;margin:auto;padding:0 16px 40px}section{display:none}section.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{cursor:pointer;color:var(--muted);white-space:nowrap}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff}.retain{background:var(--retain)}.reject{background:var(--reject)}.record,.investigate{background:var(--record)}
input,select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px;margin:0 6px 6px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}.grid figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.grid img{width:100%;height:auto;border-radius:4px}.grid figcaption{font-size:12px;color:var(--muted)}svg text{fill:var(--fg);font-size:11px}.note{color:var(--muted);font-size:13px}
</style></head><body>
<header><h1>Quant research dashboard</h1><p class="sub">Everything the platform has recorded. Retained and rejected hypotheses are shown side by side on purpose.</p></header>
<nav id="nav"></nav><main>
<section id="ledger"><div class="card"><input id="q" placeholder="filter text"><select id="dec"><option value="">any decision</option><option>retain</option><option>reject</option><option>record</option><option>investigate</option></select><span id="count" class="note"></span><div id="ledgerTable"></div></div></section>
<section id="library"><div class="card"><p class="note">Stage 30 strategy library, one backtest each, net of costs. These were found by a search; read the search-aware tests in the report before trusting any single row.</p><div id="libTable"></div></div>
<div class="card"><b>Parameter sensitivity</b> <select id="sensBase"></select><div id="sens"></div><p class="note">A strategy whose Sharpe collapses next to its best parameter is a fitted strategy, not a robust one.</p></div></section>
<section id="runs"><div class="card" id="runsBox"></div><div class="card"><div id="curves"></div></div></section>
<section id="ic"><div class="card"><p class="note">Information coefficient by horizon: the rank correlation between a signal today and the return over the next h days.</p><div id="icChart"></div></div></section>
<section id="walk"><div class="card"><div id="walkTable"></div></div></section>
<section id="risk"><div class="card"><div id="riskTable"></div></div></section>
<section id="figs"><div class="card"><input id="fq" placeholder="filter figures"></div><div class="grid" id="gallery"></div></section>
<section id="power"><div class="card"><p class="note">The smallest true effect the platform's tests would notice 80% of the time (Stage 39). A "not significant" result is only evidence of no effect up to this size.</p><div id="powerBox"></div></div></section>
</main>
<script>
const D=__DATA__;
const $=(s)=>document.querySelector(s);const fmt=(v,d=3)=>v==null||Number.isNaN(v)?'':(typeof v==='number'?v.toFixed(d):v);
const tabs=[['ledger','Decision ledger'],['library','Strategy library'],['runs','Your runs'],['ic','IC decay'],['walk','Walk-forward'],['risk','Risk'],['figs','Figures'],['power','What can we detect?']];
$('#nav').innerHTML=tabs.map(([id,t])=>`<button data-t="${id}">${t}</button>`).join('');
function show(id){document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===id));document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>show(b.dataset.t));show('ledger');
function table(rows,cols,opt={}){if(!rows.length)return '<p class="note">Nothing recorded yet.</p>';let h='<table><thead><tr>'+cols.map(c=>`<th data-c="${c.k}">${c.h||c.k}</th>`).join('')+'</tr></thead><tbody>';
 h+=rows.map(r=>'<tr>'+cols.map(c=>`<td>${c.f?c.f(r[c.k],r):fmt(r[c.k])}</td>`).join('')+'</tr>').join('')+'</tbody></table>';return h;}
function sortable(el,rows,cols,draw){el.onclick=(e)=>{const th=e.target.closest('th');if(!th)return;const k=th.dataset.c;el._d=el._k===k?-(el._d||1):1;el._k=k;rows.sort((a,b)=>((a[k]>b[k])-(a[k]<b[k]))*el._d);draw();};}
// ledger
function drawLedger(){const q=$('#q').value.toLowerCase(),d=$('#dec').value;const rows=D.ledger.filter(r=>(!d||r.decision===d)&&(!q||(r.hypothesis+r.stage+r.notes).toLowerCase().includes(q)));
 $('#count').textContent=`${rows.length} of ${D.ledger.length} experiments; ${D.ledger.filter(r=>r.decision==='retain').length} retained, ${D.ledger.filter(r=>r.decision==='reject').length} rejected`;
 $('#ledgerTable').innerHTML=table(rows,[{k:'id',h:'ID'},{k:'stage'},{k:'hypothesis',f:v=>v},{k:'decision',f:v=>`<span class="tag ${v}">${v}</span>`},{k:'results',h:'key results',f:v=>Object.entries(v).map(([a,b])=>`${a}: ${fmt(b)}`).join('<br>')},{k:'notes',f:v=>v}]);}
$('#q').oninput=drawLedger;$('#dec').onchange=drawLedger;drawLedger();
// library
const lib=D.library.slice().sort((a,b)=>b.sharpe-a.sharpe);const libCols=[{k:'name'},{k:'family',f:v=>v},{k:'sharpe',h:'net Sharpe'},{k:'cagr'},{k:'vol'},{k:'max_drawdown',h:'max drawdown'},{k:'turnover',h:'turnover/yr',f:v=>fmt(v,1)}];
function drawLib(){$('#libTable').innerHTML=table(lib,libCols);}drawLib();sortable($('#libTable'),lib,libCols,drawLib);
const bases=[...new Set(D.library.filter(r=>Object.keys(r.params).length).map(r=>r.base))];$('#sensBase').innerHTML=bases.map(b=>`<option>${b}</option>`).join('');
function line(el,series,xlab,ylab,w=640,h=260){const all=series.flatMap(s=>s.pts);if(!all.length){el.innerHTML='<p class="note">No data.</p>';return;}
 const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);const m=40;const X=v=>m+(w-m-10)*(x1===x0?.5:(v-x0)/(x1-x0)),Y=v=>h-24-(h-34)*(y1===y0?.5:(v-y0)/(y1-y0));
 const cols=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#999','#444'];
 el.innerHTML=`<svg viewBox="0 0 ${w} ${h}" width="100%"><line x1="${m}" y1="${h-24}" x2="${w-10}" y2="${h-24}" stroke="#888"/><line x1="${m}" y1="10" x2="${m}" y2="${h-24}" stroke="#888"/>`+
 series.map((s,i)=>`<polyline fill="none" stroke="${cols[i%8]}" stroke-width="1.6" points="${s.pts.map(p=>X(p[0])+','+Y(p[1])).join(' ')}"/>`+s.pts.map(p=>`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="${s.pts.length<30?3:0}" fill="${cols[i%8]}"/>`).join('')).join('')+
 `<text x="${m}" y="${h-8}">${xlab} ${x0.toFixed?x0.toFixed(2):x0} to ${x1.toFixed?x1.toFixed(2):x1}</text><text x="4" y="14">${ylab} ${y1.toFixed(2)}</text><text x="4" y="${h-26}">${y0.toFixed(2)}</text></svg>`+
 '<div class="note">'+series.map((s,i)=>`<span style="color:${cols[i%8]}">&#9632;</span> ${s.name}`).join(' &nbsp; ')+'</div>';}
function drawSens(){const b=$('#sensBase').value;const rows=D.library.filter(r=>r.base===b&&Object.keys(r.params).length);const keys=[...new Set(rows.flatMap(r=>Object.keys(r.params)))].filter(k=>rows.every(r=>typeof r.params[k]==='number'));
 $('#sens').innerHTML='';keys.forEach(k=>{const d=document.createElement('div');$('#sens').appendChild(d);const base=D.library.find(r=>r.name===b);const pts=rows.map(r=>[r.params[k],r.sharpe]).sort((a,c)=>a[0]-c[0]);line(d,[{name:`${b}: net Sharpe against ${k}`,pts}],k,'Sharpe');});
 if(!keys.length)$('#sens').innerHTML='<p class="note">No numeric parameter variants for this strategy.</p>';}
$('#sensBase').onchange=drawSens;if(bases.length)drawSens();
// runs
$('#runsBox').innerHTML=D.runs.length?table(D.runs,[{k:'run_id'},{k:'name'},{k:'group_name',h:'group'},{k:'sharpe'},{k:'gross_sharpe',h:'gross Sharpe'},{k:'deflated_sharpe_probability',h:'deflated Sharpe prob.'},{k:'n_trials',h:'trials',f:v=>v},{k:'max_drawdown'},{k:'ann_turnover',h:'turnover',f:v=>fmt(v,1)}]):'<p>No runs of your own yet. Try <code>quant backtest --model momentum</code>, then rebuild this page with <code>quant dashboard</code>.</p>';
line($('#curves'),Object.entries(D.curves).map(([n,c])=>({name:n,pts:c.x.map((d,i)=>[i,c.y[i]])})),'month index','growth of 1');
// IC
const hs=[...new Set(D.ic.map(r=>r.horizon))].sort((a,b)=>a-b);line($('#icChart'),[...new Set(D.ic.map(r=>r.signal))].map(s=>({name:s,pts:D.ic.filter(r=>r.signal===s).map(r=>[Math.log10(r.horizon),r.mean_ic]).sort((a,b)=>a[0]-b[0])})),'log10 horizon (days)','mean IC');
$('#walkTable').innerHTML=table(D.walk_forward,[{k:'book'},{k:'n_folds'},{k:'folds_positive'},{k:'fold_sharpe_mean',h:'mean fold Sharpe'},{k:'sharpe',h:'full-sample Sharpe'}]);
$('#riskTable').innerHTML=D.risk.length?table(D.risk,Object.keys(D.risk[0]).slice(0,10).map(k=>({k}))):'<p class="note">Run stage 9.</p>';
// gallery
function drawG(){const q=$('#fq').value.toLowerCase();$('#gallery').innerHTML=D.gallery.filter(g=>!q||g.caption.toLowerCase().includes(q)).map(g=>`<figure><a href="figures/${g.file}"><img loading="lazy" src="figures/${g.file}" alt=""></a><figcaption>${g.caption}</figcaption></figure>`).join('');}$('#fq').oninput=drawG;drawG();
// power
$('#powerBox').innerHTML='<h4>Paired Sharpe test: minimum detectable Sharpe difference</h4>'+table(D.power.sharpe_mde,[{k:'tracking_error',h:'tracking error',f:v=>(v*100).toFixed(0)+'%'},{k:'years',f:v=>v},{k:'min_detectable_sharpe_difference',h:'MDE',f:v=>v==null?'above grid (0.6)':fmt(v,2)}])+
'<h4>CRPS forecast test: minimum detectable out-of-sample R-squared</h4>'+table(D.power.forecast_mde,[{k:'origins',f:v=>v},{k:'min_detectable_r2',h:'MDE R-squared',f:v=>(v*100).toFixed(2)+'%'}])+
'<h4>Earlier comparisons read against it</h4>'+table(D.power.context,[{k:'a',f:v=>v},{k:'b',f:v=>v},{k:'observed_difference',h:'observed diff.'},{k:'p_value',h:'p'},{k:'min_detectable_difference_at_grid',h:'MDE for that sample'}]);
</script></body></html>
"""


def build_dashboard(config, out: str | Path = "reports/dashboard.html") -> Path:
    data = collect(config)
    out = Path(out)
    if not out.is_absolute():
        out = config.root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    out.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    return out
