"""Shaping results for the dashboard (JSON payloads for the vanilla-JS charts) and for the self-contained HTML report.

Everything here is a pure function of numbers already computed: no model is fitted and no data is read. Series are thinned (never smoothed) so a payload stays small: when a series is longer than
``max_points`` every k-th point is kept and the last point is always kept.
"""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd


def _num(v, digits: int = 5):
    return None if v is None or not np.isfinite(v) else round(float(v), digits)


def thin(n: int, max_points: int = 900) -> np.ndarray:
    """Positions to keep from a series of length ``n``: evenly spaced, always including the last."""
    if n <= max_points:
        return np.arange(n)
    step = int(math.ceil(n / max_points))
    keep = np.arange(0, n, step)
    return keep if keep[-1] == n - 1 else np.append(keep, n - 1)


def equity_payload(net: pd.Series, benchmark: pd.Series | None, labels: pd.Series, insample_net: pd.Series | None = None, max_points: int = 900) -> dict:
    """The equity curve (starting at 1.0) of the out-of-sample strategy with the segment of every point, the benchmark's curve over the same days, the drawdown, and the in-sample (train) curve if there
    is one. The train curve is a different strategy run (the first model on its own training data), so it is returned separately and not chained to the others."""
    eq = (1.0 + net.fillna(0.0)).cumprod()
    keep = thin(len(eq), max_points)
    out = {"dates": [str(d.date()) for d in eq.index[keep]], "strategy": [_num(v) for v in eq.to_numpy()[keep]],
           "segment": [str(s) for s in labels.reindex(eq.index).ffill().to_numpy()[keep]], "drawdown": [_num(v) for v in (eq / eq.cummax() - 1.0).to_numpy()[keep]]}
    if benchmark is not None:
        b = (1.0 + benchmark.reindex(eq.index).fillna(0.0)).cumprod()
        out["benchmark"] = [_num(v) for v in b.to_numpy()[keep]]
    if insample_net is not None and len(insample_net):
        ie = (1.0 + insample_net.fillna(0.0)).cumprod()
        k2 = thin(len(ie), max_points // 2)
        out["train"] = {"dates": [str(d.date()) for d in ie.index[k2]], "strategy": [_num(v) for v in ie.to_numpy()[k2]]}
    bounds = {}
    for seg in ("validation", "test", "forward"):
        dates = labels.index[labels == seg]
        if len(dates):
            bounds[seg] = [str(dates[0].date()), str(dates[-1].date())]
    out["bounds"] = bounds
    return out


def histogram(values, bins: int = 40, clip: tuple | None = None) -> dict:
    """Counts of ``values`` in equal-width bins (edges returned); ``clip`` limits the range to given quantiles so one outlier does not flatten the chart."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return {"edges": [], "counts": [], "n": int(len(v))}
    lo, hi = (np.quantile(v, clip[0]), np.quantile(v, clip[1])) if clip else (v.min(), v.max())
    if hi <= lo:
        hi = lo + 1e-9
    counts, edges = np.histogram(np.clip(v, lo, hi), bins=bins, range=(lo, hi))
    return {"edges": [_num(e, 6) for e in edges], "counts": [int(c) for c in counts], "n": int(len(v)), "mean": _num(v.mean(), 6), "std": _num(v.std(ddof=1), 6)}


def prediction_distribution(scores: pd.Series, labels: pd.Series | None = None, bins: int = 40) -> dict:
    """The distribution of out-of-sample scores, overall and by segment (the segment of each score's date)."""
    out = {"all": histogram(scores.to_numpy(), bins, (0.005, 0.995))}
    if labels is not None:
        dates = scores.index.get_level_values(0)
        seg = labels.reindex(dates.unique()).reindex(dates).to_numpy()
        for name in ("validation", "test", "forward"):
            sel = seg == name
            if sel.sum() >= 30:
                out[name] = histogram(scores.to_numpy()[sel], bins, (0.005, 0.995))
    return out


def importance_bars(table: pd.DataFrame, n: int = 20) -> list[dict]:
    """The ``n`` most important features: share of total importance, mean value (signed for linear models), the share of refits agreeing on its sign and its average rank."""
    if table is None or table.empty:
        return []
    return [{"feature": f, "share": _num(r.get("share")), "mean": _num(r.get("mean")), "sign_agreement": _num(r.get("sign_agreement")), "mean_rank": _num(r.get("mean_rank"), 2)} for f, r in table.head(n).iterrows()]


def ic_series_payload(ic: pd.Series, window: int = 63, max_points: int = 600) -> dict:
    """The IC through time: a trailing mean of ``window`` dates (an unsmoothed daily IC is mostly noise)."""
    x = ic.dropna()
    roll = x.rolling(window, min_periods=max(10, window // 3)).mean()
    keep = thin(len(roll), max_points)
    return {"dates": [str(d.date()) for d in roll.index[keep]], "rolling_ic": [_num(v) for v in roll.to_numpy()[keep]], "window": window, "cumulative": [_num(v) for v in x.cumsum().to_numpy()[keep]]}


def similarity_map(returns: pd.DataFrame, min_overlap: int = 60) -> dict:
    """A two-dimensional map of strategies by how their returns co-move: classical multidimensional scaling of the distance ``sqrt(0.5 (1 - correlation))``. Strategies close together are near-duplicates, which
    matters when combining them. Returns the coordinates and the share of the variance the two axes keep."""
    r = returns.dropna(axis=1, thresh=min_overlap)
    if r.shape[1] < 3:
        return {"names": list(r.columns), "x": [0.0] * r.shape[1], "y": [0.0] * r.shape[1], "explained": 0.0}
    corr = r.corr(min_periods=min_overlap).fillna(0.0).to_numpy()
    d2 = 0.5 * (1.0 - np.clip(corr, -1, 1))
    n = len(d2)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ d2 @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    pos = np.clip(w, 0, None)
    xy = v[:, :2] * np.sqrt(pos[:2])
    return {"names": [str(c) for c in r.columns], "x": [_num(a, 4) for a in xy[:, 0]], "y": [_num(a, 4) for a in xy[:, 1]], "explained": _num(pos[:2].sum() / max(pos.sum(), 1e-12), 3)}


def leaderboard_payload(rows: list[dict], columns: tuple = ("validation.sharpe", "test.sharpe", "forward.sharpe", "oos.sharpe", "ic.mean", "ic.ir", "overfit.generalization_score", "oos.max_drawdown", "oos.ann_turnover")) -> list[dict]:
    """Registry leaderboard rows flattened for the table: identity first, then the requested metrics."""
    out = []
    for r in rows:
        m = r.get("metrics", {})
        out.append({"id": r["id"], "name": r["name"], "created": r["created"], "model": r["model_kind"], "universe": r["universe"], "target": r["target"], "signal": r["signal_method"], "stage": r.get("stage", "none"),
                    "trials": r.get("trials_at_run"), **{c: _num(m.get(c)) for c in columns}})
    return out


def shap_payload(expl: dict) -> dict:
    """An explanation of one prediction as a waterfall: the base, each contribution, an 'all other features' bar and the final score."""
    steps, running = [], expl["base"]
    for c in expl["contributions"]:
        steps.append({"feature": c["feature"], "value": c["value"], "contribution": _num(c["contribution"], 6), "start": _num(running, 6), "end": _num(running + c["contribution"], 6)})
        running += c["contribution"]
    steps.append({"feature": "all other features", "value": None, "contribution": _num(expl["other"], 6), "start": _num(running, 6), "end": _num(running + expl["other"], 6)})
    return {"date": expl["date"], "asset": expl["asset"], "base": _num(expl["base"], 6), "prediction": _num(expl["prediction"], 6), "steps": steps}


# ------------------------------------------------------------------------------------------------------------------ the HTML report
def _svg_lines(series: dict, dates: list, width: int = 760, height: int = 220) -> str:
    vals = [v for s in series.values() for v in s if v is not None]
    if not vals or len(dates) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    colors = ["#2a6ebb", "#9aa4b2", "#c4562b"]
    paths = []
    for (name, s), color in zip(series.items(), colors):
        pts = [f"{40 + (width - 60) * i / (len(s) - 1):.1f},{height - 24 - (height - 40) * (v - lo) / span:.1f}" for i, v in enumerate(s) if v is not None]
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{" ".join(pts)}"><title>{html.escape(name)}</title></polyline>')
    legend = "".join(f'<text x="{48 + 150 * i}" y="14" font-size="11" fill="{c}">{html.escape(n)}</text>' for i, ((n, _), c) in enumerate(zip(series.items(), colors)))
    axis = (f'<text x="4" y="{height - 24}" font-size="10" fill="#666">{lo:.2f}</text><text x="4" y="30" font-size="10" fill="#666">{hi:.2f}</text>'
            f'<text x="40" y="{height - 6}" font-size="10" fill="#666">{html.escape(dates[0])}</text><text x="{width - 90}" y="{height - 6}" font-size="10" fill="#666">{html.escape(dates[-1])}</text>')
    return f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="equity curves">{legend}{axis}{"".join(paths)}</svg>'


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fmt(v, kind: str = "num") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.1%}" if kind == "pct" else f"{v:,.2f}"


def render_html(report: dict) -> str:
    """The experiment as one self-contained HTML page (no scripts, no external files): headline, equity curve, segments, information coefficient, overfitting warnings, features and robustness."""
    cfg, seg, ov = report["config"], report["segments"]["stats"], report["overfit"]
    eq = report["equity"]
    chart = _svg_lines({"strategy": eq["strategy"], **({"benchmark": eq["benchmark"]} if "benchmark" in eq else {})}, eq["dates"])
    seg_rows = [[k, v["n_days"], _fmt(v["ann_return"], "pct"), _fmt(v["ann_vol"], "pct"), _fmt(v["sharpe"]), _fmt(v["max_drawdown"], "pct")] for k, v in seg.items()]
    ic = report["ic"]["all"]
    feats = [[f["feature"], _fmt(f["share"], "pct"), _fmt(f["mean"]), _fmt(f["sign_agreement"], "pct")] for f in report["importance"]["bars"][:15]]
    rob = report.get("robustness")
    rob_html = ""
    if rob:
        rob_html = "<h2>Robustness</h2>" + _table(["test", "verdict", "what happened"], [[t["title"], t["verdict"], t["detail"]] for t in rob["tests"]])
    warn = "".join(f"<li>{html.escape(w)}</li>" for w in ov["warnings"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(report['name'])}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1b1f24}}table{{border-collapse:collapse;width:100%;margin:.5rem 0}}th,td{{text-align:right;padding:4px 8px;border-bottom:1px solid #e3e6ea}}
th:first-child,td:first-child{{text-align:left}}h1{{font-size:1.4rem}}h2{{font-size:1.1rem;margin-top:1.6rem}}.note{{color:#5a6472;font-size:.9rem}}.warn li{{margin:.3rem 0}}</style></head><body>
<h1>{html.escape(report['name'])}</h1>
<p class="note">{html.escape(cfg['model']['kind'])} on {html.escape(str(cfg['dataset']['universe']))}, target {html.escape(cfg['dataset']['target'])}, signal {html.escape(cfg['signal']['method'])}.
Data source: {html.escape(report['provenance']['source'])} ({html.escape(report['provenance']['label'])}). Configuration {html.escape(report['config_hash'])}; {report['trials']['dataset_trials']} looks at this data have been recorded.</p>
<h2>Equity curve (out of sample)</h2>{chart}
<h2>By segment</h2>{_table(['segment', 'days', 'return', 'volatility', 'Sharpe', 'max drawdown'], seg_rows)}
<h2>Information coefficient</h2><p>Mean rank IC {_fmt(ic['mean'])} (information ratio {_fmt(ic['ir'])}, t {_fmt(ic['t'])} on {ic['n']} dates; hit rate {_fmt(ic['hit_rate'], 'pct')}).</p>
<h2>Overfitting</h2><p>Generalisation score <b>{_fmt(ov['generalization_score'])}</b> from 0 (nothing survives out of sample) to 1.</p><ul class="warn">{warn}</ul>
<h2>Features</h2>{_table(['feature', 'share', 'mean', 'sign agreement'], feats)}{rob_html}
<p class="note">Generated by the platform's machine-learning research pipeline. Test and forward segments are reserved: judge a strategy on them once.</p></body></html>"""


def jsonable(value):
    """Recursively turn numpy / pandas values into plain JSON types; NaN and infinity become ``None``, timestamps become ISO dates, frames become records."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return jsonable(value.reset_index().to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    if value is pd.NaT or value is pd.NA:
        return None
    return value
