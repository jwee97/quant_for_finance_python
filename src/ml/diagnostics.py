"""The statistical diagnostics of a saved experiment, and their reading against the adopted validation policy (spec v3, Release 5, P1-16 to P1-22 and section 8.4).

``run`` computes, for one experiment and one segment, everything Release 5 defines on its stored returns: the three Sharpe estimates (naive beside HAC and bootstrap), the HAC mean, the incremental return
against each matched benchmark (and the unmatched equal-weight one, labelled), the predeclared regimes, calibration where the model produced class probabilities, the selection-test evidence the study already
holds, and optionally the neighbourhood, delay and capacity analyses (which re-run the saved predictions). The result is stored whole in the append-only ``analysis_records`` table with the hash of the
policy it was computed under.

``evaluate_policy`` reads that record against a ``ValidationPolicy`` and returns one item per criterion the policy requires: ``ok``, ``failed`` (the evidence says no), ``missing`` (the evidence has not been
produced or is too short *yet*), ``blocked`` (it cannot be evaluated: a procedure was not applicable, or no policy is adopted) or ``not_applicable``. A correctly implemented strategy that loses fails here and
passes Gate A; nothing in this module can make a gate pass by itself.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from ..backtest.execution import build_held_weights
from . import benchmarks as bm
from . import calibration as cal
from . import experiments as xp
from . import inference as inf
from . import multiseed as ms
from . import policy as pol
from . import regime_eval as rg
from . import research_controls as rc
from . import sensitivity as sn

VERSION = "diagnostics-v1"
PROCEDURES = ("dsr", "pbo", "reality_check", "spa", "fdr")


def store(registry, kind: str, subject_id: str, content: dict, *, policy_hash: str | None = None, study_id: str | None = None, actor: str = "system") -> int:
    text = json.dumps(content, sort_keys=True, default=str, allow_nan=False)
    with registry.transaction() as c:
        cur = c.execute("INSERT INTO analysis_records (kind, subject_id, study_id, content, content_hash, policy_hash, actor, created) VALUES (?,?,?,?,?,?,?,?)",
                        (kind, subject_id, study_id, text, hashlib.sha256(text.encode()).hexdigest()[:32], policy_hash, actor, time.strftime("%Y-%m-%d %H:%M:%S")))
    return int(cur.lastrowid)


def latest(registry, subject_id: str, kind: str) -> dict | None:
    with registry._conn() as c:                                                                  # noqa: SLF001
        row = c.execute("SELECT * FROM analysis_records WHERE subject_id=? AND kind=? ORDER BY seq DESC LIMIT 1", (subject_id, kind)).fetchone()
    if row is None:
        return None
    return {"seq": row["seq"], "content": json.loads(row["content"]), "content_hash": row["content_hash"], "policy_hash": row["policy_hash"], "created": row["created"], "actor": row["actor"]}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return str(x)
    if isinstance(x, (pd.Series, pd.DataFrame, pd.Index)):
        return None
    return x


def _selection(registry, experiment_id: str) -> dict:
    study = rc.study_of_experiment(registry, experiment_id)
    if not study:
        return {"study": None, "procedures": {}, "note": "this experiment is in no study, so no selection test can be attributed to it"}
    family = rc.study_family(registry, study)
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = [dict(r) for r in c.execute(f"SELECT * FROM selection_evidence WHERE study_id IN ({','.join('?' * len(family))}) ORDER BY id DESC LIMIT 400", family)]
    out: dict = {}
    for r in rows:
        if r["method"] in out:
            continue
        out[r["method"]] = {"status": r["status"], "reason": r.get("reason") or "", "statistic": json.loads(r["statistic"]) if r["statistic"] else {}}
    return {"study": study, "procedures": out, "raw_looks": rc.raw_looks(registry, study),
            "note": "a raw trial count is recorded beside these results; it does not by itself prove that the selection bias has been corrected completely"}


def _volume(cfg: dict, market: dict):
    """The traded volume of the experiment's universe for the liquidity regime, from the bundle given or the one the configuration resolves; ``None`` (and the regime says so) when there is none."""
    from .datasets import resolve_bundle

    bundle = market.get("bundle")
    if bundle is None and market.get("config") is not None:
        try:
            bundle, _ = resolve_bundle(market["config"], cfg["dataset"]["universe"], market.get("builder"))
        except Exception:                                                                         # noqa: BLE001 - no market data to resolve: the liquidity regime reports insufficient evidence
            bundle = None
    return getattr(bundle, "volume", None)


def _interval_calibration(cfg: dict, ds, train, preds: pd.DataFrame, segment: str, lag_total: int) -> dict:
    """Quantile and interval scores (pinball, coverage with its interval, width, interval score) and the two conformal procedures, for the diagnosed segment only, when the model stored a predictive
    distribution. The conformal calibration residuals are those of earlier forecasts whose outcomes had ended (horizon plus lag sessions before), so nothing in the segment calibrates itself."""
    dist = train.extras.get("distribution")
    if dist is None or not len(dist):
        return {"eligibility": "not_applicable", "reason": "the model stored no predictive distribution (its output is a point forecast, a rank or a probability)"}
    levels = cal.interval_levels(cfg["model"])
    if levels is None:
        return {"eligibility": "unavailable", "reason": f"the quantile levels of the {cfg['model']['kind']} model's interval are not recorded, so coverage cannot be compared with a nominal level"}
    lo, hi = levels
    seg = preds["segment"].reindex(dist.index).where(lambda v: v == segment)
    y = preds["target"].reindex(dist.index)
    horizon = int(ds.horizon)
    return {"eligibility": "applied", "segment": segment, "model": cfg["model"]["kind"], "quantiles": cal.quantile_report(dist, y, seg, alpha_low=lo, alpha_high=hi),
            "conformal": cal.conformal_report(dist["mean"], y, seg, ds.dates, alpha=1.0 - (hi - lo), delay=lag_total + horizon),
            "note": "coverage is read against the nominal level with its interval; the conformal intervals are centred on the stored centre forecast, use only outcomes that had ended, and carry the guarantee stated with each method (none for the split, a long-run average for the adaptive)"}


def run(registry, experiment_id: str, *, policy: pol.ValidationPolicy | None = None, actor: str = "system", frequency: str = "daily", volume: pd.DataFrame | None = None, n_boot: int = 500,
        seed: int = 0, neighborhood_grid: dict | None = None, delays=None, capitals=None, crisis_windows: dict | None = None, save: bool = True, **market) -> dict:
    """Compute and (by default) store the diagnostics. ``policy`` defaults to the adopted one; without any policy the diagnostics are still computed with declared neutral settings, are stored with no policy hash,
    and the gate will not read them as evidence (it needs an adopted policy). ``volume`` (date x asset shares) enables the liquidity regime; ``neighborhood_grid``, ``delays`` and ``capitals`` switch on the
    analyses that re-run the saved predictions."""
    adopted = pol.adopted(registry)
    policy = policy or (adopted[0] if adopted else None)
    crit = dict(policy.criteria) if policy else {}
    segment = crit.get("segment") or "validation"
    level = float(crit.get("confidence_level") or 0.95)
    min_regime_days = int(crit.get("min_regime_days") or 60)
    holdout = (registry.report(experiment_id).get("holdout") or {}).get("status")
    if segment in ("test", "forward") and holdout != "evaluated_under_plan":
        raise ValueError(f"the {segment} segment of this experiment is protected (holdout status: {holdout or 'not recorded'}): statistics are computed on it only for an experiment that a frozen plan evaluated once "
                         "(POST /api/ml/evaluation-plan, then /api/ml/evaluate). Judge the validation segment, or evaluate under a plan")
    cfg, ds, train, info = xp.load_run(registry, experiment_id)
    folder = registry.root / "experiments" / experiment_id
    ret = pd.read_parquet(folder / "returns.parquet")
    net = ret["net"][ret["segment"] == segment].dropna()
    label = f"{experiment_id[-8:]} {segment}"
    lag_total = int(cfg["dataset"]["lag"]) + int(cfg["backtest"].get("extra_lag", 0))
    out: dict = {"version": VERSION, "experiment_id": experiment_id, "segment": segment, "frequency": frequency, "level": level, "policy_hash": policy.content_hash() if policy else None,
                 "policy": policy.content() if policy else None, "limitations": [], "execution_mode": registry.report(experiment_id).get("execution", {}).get("mode", "vectorized_legacy")}
    out["sample"] = inf.sample_identity(net, segment)
    out["sharpe"] = {k: v.as_dict() for k, v in inf.sharpe_inference(net, frequency=frequency, label=label, level=level, n_boot=n_boot, seed=seed).items()}
    out["mean"] = inf.hac_mean(net, frequency=frequency, label=label, level=level).as_dict()
    # --- matched benchmarks
    positions = pd.read_parquet(folder / "positions.parquet").reindex(ds.dates).fillna(0.0)
    held = build_held_weights(positions.reindex(columns=ds.returns.columns).fillna(0.0), ds.returns, str(cfg["backtest"]["rebalance"]), lag_total, True)[0].fillna(0.0)
    market_ret = bm.equal_weight(ds.returns, ds.investable)
    strategy_full = ret["net"].dropna()
    bench = {"beta": bm.beta_matched(strategy_full, market_ret)[0], "net_exposure": bm.net_exposure_matched(held, market_ret), "sector": bm.sector_matched(held, ds.returns, dict(ds.group or ds.asset_class or {})),
             "risk": bm.risk_matched(strategy_full, market_ret)}
    inside = net.index
    out["benchmarks"] = {k: bm.incremental(net, b.reindex(inside), kind=f"{k}_matched", frequency=frequency, label=label, n_boot=min(n_boot, 400), seed=seed) for k, b in bench.items()}
    out["benchmarks"]["descriptive_equal_weight"] = bm.incremental(net, market_ret.reindex(inside), kind="descriptive_equal_weight", frequency=frequency, label=label, n_boot=min(n_boot, 400), seed=seed)
    if out["execution_mode"] == "realistic_cash_equity":
        out["limitations"].append("exposures are those of the target weights through the vectorized engine's lag and schedule; the realistic mode's executed holdings differ a little")
    # --- regimes (declared before the evaluation, from the policy's minimum)
    decl = rg.declare(rg.default_specs(crisis_windows))
    volume = volume if volume is not None else _volume(cfg, market)
    dollar = None if volume is None else (volume.reindex(index=ds.dates, columns=ds.prices.columns) * ds.prices).sum(axis=1)
    out["regimes"] = rg.evaluate(net, market_ret, decl, volume=dollar, frequency=frequency, min_days=min_regime_days, label_prefix=f"{label} ")
    out["regime_declaration"] = decl
    # --- calibration, when the model produced class probabilities
    preds = pd.read_parquet(folder / "predictions.parquet")
    if "prob" in preds.columns and preds["prob"].notna().any():
        p, y, seg = preds["prob"], preds["target"], preds["segment"]
        out["calibration"] = _jsonable(cal.probability_report(p, y, seg))
    else:
        out["calibration"] = {"eligibility": "not_applicable", "reason": "the model produced no class probabilities (its output is not a probability)"}
    out["interval_calibration"] = _interval_calibration(cfg, ds, train, preds, segment, lag_total)
    # --- selection bias evidence already held by the study
    out["selection"] = _selection(registry, experiment_id)
    # --- analyses that re-run the saved predictions
    if neighborhood_grid:
        out["neighborhood"] = sn.neighborhood(registry, experiment_id, neighborhood_grid, segment=segment, **market)
    if delays:
        out["delay"] = sn.delay_decay(registry, experiment_id, delays, segment=segment, **market)
    if capitals:
        out["capacity"] = sn.capacity_decay(registry, experiment_id, capitals, segment=segment, **market)
    out["limitations"] += ["no number here is a recommendation: thresholds belong to an adopted validation policy", "the Monte Carlo error of the bootstrap is in the replications recorded with each result"]
    out = _jsonable(out)
    if save:
        out["record_seq"] = store(registry, "diagnostics", experiment_id, out, policy_hash=out["policy_hash"], study_id=out["selection"].get("study"), actor=actor)
    return out


def multiseed(registry, experiment_id: str, seeds, *, segment: str = "validation", metric: str = "sharpe", n_boot: int = 300, actor: str = "system", save: bool = True, **market) -> dict:
    """Train the experiment's own configuration once per predeclared seed, in development mode (validation segment only), as looks of the SAME study, and store the per-seed results beside the spread across
    seeds and each seed's time-series uncertainty. The record is attached to ``experiment_id`` (kind ``multiseed``)."""
    if segment in ("test", "forward"):
        raise ValueError("multi-seed evaluation trains in development mode and reads the validation segment only; the protected segments are read once, by a frozen plan")
    cfg = registry.report(experiment_id)["config"]
    over = {k: cfg[k] for k in ("dataset", "split", "model", "signal", "backtest") if k in cfg}
    over["name"] = f"{(registry.get_experiment(experiment_id).get('name') or 'seeded')}"
    out = ms.run_seeds(over, seeds, segment=segment, metric=metric, n_boot=n_boot, registry=registry, study=rc.study_of_experiment(registry, experiment_id), stop_after="validation", **market)
    out["subject"] = experiment_id
    out = _jsonable(out)
    if save:
        out["record_seq"] = store(registry, "multiseed", experiment_id, out, actor=actor, study_id=rc.study_of_experiment(registry, experiment_id))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------ reading a record against a policy
def _item(item_id: str, label: str, status: str, reason: str = "", value=None) -> dict:
    return {"id": item_id, "label": label, "status": status, "reason": reason, "value": value}


def evaluate_policy(policy: pol.ValidationPolicy, diag: dict) -> list[dict]:
    """One item per retrospective criterion that ``policy`` requires, read from the stored ``diag`` (Gate B). A criterion the policy leaves at ``None`` produces no item. The prospective criteria are read from the
    ledger by ``prospective_items`` (Gate C)."""
    items: list[dict] = []
    c = policy.criteria
    sh = diag["sharpe"]
    if policy.required("min_segment_days"):
        n = diag["sample"]["n"]
        items.append(_item("dx:segment_days", f"evidence length on the {diag['segment']} segment", "ok" if n >= c["min_segment_days"] else "missing",
                           "" if n >= c["min_segment_days"] else f"{n} days on the segment; the policy asks for {c['min_segment_days']}", n))
    if policy.required("min_sharpe_ci_lower"):
        method = policy.get("interval_method", "hac")
        r = sh["hac" if method == "hac" else "bootstrap"]
        if r["eligibility"] != "applied" or not r.get("interval") or r["interval"][0] is None:
            items.append(_item("dx:sharpe_interval", "dependence-aware Sharpe interval", "missing", f"the {method} interval was not computed: {r.get('reason') or 'no finite interval'}"))
        else:
            lo = r["interval"][0]
            ok = lo > c["min_sharpe_ci_lower"]
            items.append(_item("dx:sharpe_interval", f"{method} Sharpe interval lower bound", "ok" if ok else "failed",
                               "" if ok else f"the lower bound {lo:.2f} of the {r['level']:.0%} {method} interval does not exceed {c['min_sharpe_ci_lower']:g} (naive point estimate {sh['naive']['value']:.2f})", lo))
    sel = diag.get("selection", {}).get("procedures", {})

    def selection_item(item_id, label, key, criterion, stat_key, better):
        e = sel.get(key)
        if e is None:
            return _item(item_id, label, "missing", f"no {key} evidence for this study: run the selection tests")
        if e["status"] != "applied":
            return _item(item_id, label, "blocked", f"{key} was {e['status']}: {e.get('reason') or 'not applied'}")
        v = e["statistic"].get(stat_key)
        if v is None:
            return _item(item_id, label, "missing", f"{key} recorded no {stat_key}")
        ok = (v >= c[criterion]) if better == "min" else (v <= c[criterion])
        return _item(item_id, label, "ok" if ok else "failed", "" if ok else f"{stat_key} is {v:.3f}; the policy asks for {'at least' if better == 'min' else 'at most'} {c[criterion]:g}", v)

    if policy.required("min_deflated_probability"):
        item = selection_item("dx:dsr", "deflated Sharpe probability", "dsr", "min_deflated_probability", "probability", "min")
        e = sel.get("dsr")
        classic = None if e is None else (e.get("statistic") or {}).get("probability_classic")
        if item["status"] in ("ok", "failed") and classic is not None and classic < item["value"]:           # the conservative of the two trial conventions must reach the threshold
            ok = classic >= c["min_deflated_probability"]
            item = _item("dx:dsr", "deflated Sharpe probability", "ok" if ok else "failed", "" if ok else f"probability is {classic:.3f} counting every recorded try (the effective-trials form gives {item['value']:.3f}); "
                         f"the policy asks for at least {c['min_deflated_probability']:g}", classic)
        items.append(item)
    if policy.required("max_pbo"):
        items.append(selection_item("dx:pbo", "probability of backtest overfitting", "pbo", "max_pbo", "pbo", "max"))
    if policy.required("max_selection_p"):
        spa = sel.get("spa")
        key, stat = ("spa", "p_consistent") if spa and spa["status"] == "applied" else ("reality_check", "p_value")
        items.append(selection_item("dx:selection_p", "family-wise selection p-value", key, "max_selection_p", stat, "max"))
    if policy.required("min_active_ci_lower"):
        kind = policy.get("benchmark_kind", "beta")
        b = diag.get("benchmarks", {}).get(kind)
        am = (b or {}).get("active_mean", {})
        if not b or am.get("eligibility") != "applied":
            items.append(_item("dx:active_return", f"active return against the {kind}-matched benchmark", "missing", f"no {kind}-matched comparison could be computed ({(am or {}).get('reason', 'not run')})"))
        else:
            lo = am["interval"][0]
            ok = lo > c["min_active_ci_lower"]
            items.append(_item("dx:active_return", f"active return against the {kind}-matched benchmark", "ok" if ok else "failed",
                               "" if ok else f"the lower bound {lo:.5f} of the active mean per period does not exceed {c['min_active_ci_lower']:g}", lo))
    if policy.required("min_regime_sharpe_upper"):
        rows = [r for r in diag.get("regimes", {}).get("rows", []) if r.get("causal") and r.get("eligibility") == "applied"]
        if not rows:
            items.append(_item("dx:regimes", "causal regimes", "missing", "no causal regime had enough days for a reading"))
        else:
            bad = [r for r in rows if (r.get("sharpe_hac_interval") or [None, None])[1] is not None and r["sharpe_hac_interval"][1] < c["min_regime_sharpe_upper"]]
            items.append(_item("dx:regimes", "no applied causal regime is clearly losing", "failed" if bad else "ok",
                               "" if not bad else "clearly losing in " + ", ".join(f"{r['regime']}={r['state']}" for r in bad), len(rows)))
    if policy.required("neighborhood_min_share_positive"):
        nb = diag.get("neighborhood")
        if not nb or nb.get("eligibility") != "applied":
            items.append(_item("dx:neighborhood", "neighbourhood robustness", "missing", "the predeclared neighbourhood has not been evaluated" if not nb else nb.get("reason", "insufficient points")))
        else:
            share = nb["summary"]["share_positive"]
            ok = share >= c["neighborhood_min_share_positive"]
            items.append(_item("dx:neighborhood", "neighbourhood robustness", "ok" if ok else "failed", "" if ok else f"{share:.0%} of the neighbourhood is positive; the policy asks for {c['neighborhood_min_share_positive']:.0%}", share))
    if policy.required("delay_min_ratio"):
        d = diag.get("delay")
        row = next((r for r in (d or {}).get("rows", []) if r["extra_sessions"] == int(policy.get("delay_sessions", 1))), None)
        if not d or d.get("eligibility") != "applied" or row is None or row.get("ratio_to_undelayed") is None:
            items.append(_item("dx:delay", "delay decay", "missing", "the delayed result was not computed, or the undelayed result is not positive so the ratio is undefined"))
        else:
            ok = row["ratio_to_undelayed"] >= c["delay_min_ratio"]
            items.append(_item("dx:delay", f"{row['extra_sessions']} extra session(s) of delay", "ok" if ok else "failed", "" if ok else f"keeps {row['ratio_to_undelayed']:.0%} of the result; the policy asks for {c['delay_min_ratio']:.0%}", row["ratio_to_undelayed"]))
    if policy.required("calibration_max_ece"):
        cb = diag.get("calibration", {})
        seg = (cb.get("segments") or {}).get(diag["segment"])
        if not seg:
            items.append(_item("dx:calibration", "calibration", "not_applicable", cb.get("reason", "no class probabilities on this segment")))
        elif seg["eligibility"] != "applied":
            items.append(_item("dx:calibration", "calibration", "missing", seg.get("reason", "insufficient")))
        else:
            ece = seg["reliability"]["ece"]
            ok = ece <= c["calibration_max_ece"]
            items.append(_item("dx:calibration", "expected calibration error", "ok" if ok else "failed", "" if ok else f"ECE {ece:.3f} exceeds {c['calibration_max_ece']:g}", ece))
    if policy.required("capacity_min_net_ratio"):
        cp = diag.get("capacity")
        rows = [r for r in (cp or {}).get("rows", []) if r.get("state") == "ok" and r.get("net_return") is not None]
        target = next((r for r in rows if policy.get("capacity_capital") and abs(r["capital"] - float(c["capacity_capital"])) < 1e-6), None)
        if not cp or cp.get("eligibility") != "applied" or target is None or rows[0]["net_return"] <= 0:
            items.append(_item("dx:capacity", "capacity", "missing", "the realistic result at the policy's capital was not computed, or the smallest level is not positive"))
        else:
            ratio = target["net_return"] / rows[0]["net_return"]
            ok = ratio >= c["capacity_min_net_ratio"]
            items.append(_item("dx:capacity", f"net result at capital {target['capital']:g}", "ok" if ok else "failed", "" if ok else f"keeps {ratio:.0%} of the smallest level's net result; the policy asks for {c['capacity_min_net_ratio']:.0%}", ratio))
    return items


def prospective_items(policy: pol.ValidationPolicy, counts: dict) -> list[dict]:
    """The policy's prospective criteria against ``prospective.counts(...)``. Only commitments classified ``prospective`` count: replays are listed beside them and never added. A policy that names neither
    criterion yields a ``blocked`` item - Gate C cannot be judged against requirements nobody declared."""
    c = policy.criteria
    pc = counts["prospective"]
    rp = counts["replay"]
    tail = f" ({rp['committed']} replay commitment(s) are not counted)" if rp["committed"] else ""
    items: list[dict] = []
    if policy.required("prospective_min_days"):
        ok = pc["distinct_settled_days"] >= c["prospective_min_days"]
        items.append(_item("dx:prospective_days", "settled prospective days", "ok" if ok else "missing",
                           "" if ok else f"{pc['distinct_settled_days']} distinct settled prospective days; the policy asks for {c['prospective_min_days']}. Elapsed time cannot be manufactured{tail}", pc["distinct_settled_days"]))
    if policy.required("prospective_min_settled"):
        ok = pc["settled"] >= c["prospective_min_settled"]
        items.append(_item("dx:prospective_settled", "settled prospective forecasts", "ok" if ok else "missing",
                           "" if ok else f"{pc['settled']} settled prospective forecasts; the policy asks for {c['prospective_min_settled']}{tail}", pc["settled"]))
    if not items:
        items.append(_item("dx:prospective_policy", "prospective evidence", "blocked", "the adopted policy declares no prospective requirement (prospective_min_days / prospective_min_settled): Gate C cannot be judged against requirements nobody declared"))
    return items
