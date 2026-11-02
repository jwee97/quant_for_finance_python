"""Command-line front door to the machine-learning platform (registered into ``python -m src.cli``):

    ml-build-dataset   build and save a leak-free panel of features and targets for a universe
    ml-train           train a model walk-forward and run the whole experiment (positions, backtest, segments, diagnostics)
    ml-backtest        re-run a saved experiment's predictions under different position rules, rebalancing or costs, with no refit
    ml-report          the leaderboard, or one experiment in full (text, JSON or the HTML page)

Examples::

    python -m src.cli ml-build-dataset --universe etf --groups price momentum volatility --target fwd_ret_5
    python -m src.cli ml-train --model ridge --param alpha=10 --universe etf --signal top_n_long_short --n 3 --rebalance weekly
    python -m src.cli ml-backtest 20260101-120000-ab12cd --cost-bps 10 --holding-period 20
    python -m src.cli ml-report 20260101-120000-ab12cd
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from .audit import audit_lines
from .config import build_key, experiment_config, merge
from .datasets import build_dataset, resolve_bundle
from .experiments import ExperimentError, rebacktest, run_experiment
from .features import GROUPS
from .registry import Registry, RegistryError
from .targets import parse_target
from .visualization import jsonable, leaderboard_payload


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("none", "null"):
        return None
    return value


def _fmt(v, kind: str = "num") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.1%}" if kind == "pct" else f"{v:,.2f}"


def _bundle_and_info(args, config):
    if getattr(args, "prices", None):
        from ..framework import bundle_from_prices, load_prices_csv

        return bundle_from_prices(load_prices_csv(args.prices), name=Path(args.prices).stem), {"source": "user", "label": args.prices}
    return resolve_bundle(config, args.universe)


def _dataset_overrides(args) -> dict:
    d: dict = {"universe": args.universe}
    if getattr(args, "prices", None):
        d["universe"] = Path(args.prices).stem
    if args.groups:
        d["feature_groups"] = args.groups
    if args.target:
        d["target"] = args.target
        d["horizon"] = parse_target(args.target)[1]
    if args.horizon:
        d["horizon"] = args.horizon
        if not args.target:
            d["target"] = f"fwd_ret_{args.horizon}"
    for key, attr in (("lag", "lag"), ("normalize", "normalize"), ("start", "start"), ("end", "end"), ("regime_detector", "regime_detector"), ("winsor", "winsor"), ("min_history", "min_history")):
        v = getattr(args, attr, None)
        if v is not None:
            d[key] = v
    return d


def cmd_build_dataset(args, config) -> int:
    registry = Registry(args.registry)
    cfg = experiment_config({"dataset": _dataset_overrides(args)})
    bundle, info = _bundle_and_info(args, config)
    ds = build_dataset(bundle, cfg["dataset"], args.name or f"{cfg['dataset']['universe']}-{build_key(cfg)}", args.extra_targets or (), info)
    fid = registry.register_feature_set(ds, build_key(cfg))
    s = ds.summary()
    print(f"dataset '{ds.name}' saved as feature set {fid}")
    print(f"  {s['n_rows']:,} rows = {s['n_dates']:,} dates x up to {s['n_assets']} assets; {s['n_features']} features; {s['start']} to {s['end']}; {s['missing_share']:.1%} of feature cells missing")
    print(f"  target {s['target']} (horizon {s['horizon']} days, starts {s['lag']} day(s) after the decision); source: {info.get('source')} ({info.get('label')})")
    print("\n".join("  " + line for line in audit_lines(ds.meta.get("audit"))))
    try:                                                                                       # the look at the features is adaptive feature discovery: adjusted for the number tested, and recorded in the study
        from . import explainability as xp
        from . import research_controls as rc
        from . import selection as sel
        from .training import plan_for

        plan, _ = plan_for(ds, cfg["split"], ds.target)
        fic = xp.feature_ic(ds, until=plan.dates[plan.bounds["train"][1] - 1])
        screen = sel.register_screen(registry, rc.study_of(cfg), fid, fic)
        print(f"  feature screen on the warm-up segment: {screen['status']}; " + (f"{screen['n_selected']} of {screen['n_tested']} features survive the {screen['method']} adjustment" if screen["status"] == "applied" else screen.get("reason", "")))
    except ValueError as exc:                                                                  # a timeline too short for the split has no warm-up segment to screen on
        print(f"  feature screen: not made ({exc})")
    for name, why in sorted(ds.meta.get("skipped", {}).items())[:12]:
        print(f"  skipped {name}: {why}")
    extra = len(ds.meta.get("skipped", {})) - 12
    if extra > 0:
        print(f"  ... and {extra} more skipped features")
    return 0


def _experiment_overrides(args) -> dict:
    over: dict = {}
    if args.config:
        over = yaml.safe_load(Path(args.config).read_text()) or {}
    over = merge(over, {"dataset": _dataset_overrides(args)})
    model = {"kind": args.model} if args.model else {}
    if args.param:
        model["params"] = {k: _coerce(v) for k, v in (p.split("=", 1) for p in args.param)}
    if model:
        over = merge(over, {"model": model})
    split = {k: v for k, v in (("refit_every", args.refit_every), ("min_train_days", args.min_train_days), ("window", args.window), ("embargo", args.embargo)) if v is not None}
    signal = {k: v for k, v in (("method", args.signal), ("n", args.n), ("holding_period", args.holding_period), ("max_weight", args.max_weight), ("quantile", args.quantile)) if v is not None}
    backtest = {k: v for k, v in (("rebalance", args.rebalance), ("cost_bps", args.cost_bps), ("slippage_bps", args.slippage_bps), ("target_vol", args.target_vol), ("extra_lag", args.extra_lag)) if v is not None}
    over = merge(over, {"split": split, "signal": signal, "backtest": backtest})
    if args.name:
        over["name"] = args.name
    if args.seed is not None:
        over["seed"] = args.seed
    if args.robustness:
        over = merge(over, {"robustness": {"enabled": True}})
    return over


def _print_run(exp, registry: Registry | None) -> None:
    r = exp.report
    print(f"\nexperiment {r['id']}: {r['name']}  ({r['seconds']}s)")
    print(f"  {r['config']['model']['kind']} on {r['config']['dataset']['universe']}, target {r['config']['dataset']['target']}, {r['config']['signal']['method']}, rebalance {r['config']['backtest']['rebalance']}; "
          f"data: {r['provenance']['source']}; {r['trials']['dataset_trials']} look(s) at this sample so far")
    rows = [{"segment": k, "days": v["n_days"], "return": _fmt(v["ann_return"], "pct"), "vol": _fmt(v["ann_vol"], "pct"), "sharpe": _fmt(v["sharpe"]), "max dd": _fmt(v["max_drawdown"], "pct")} for k, v in r["segments"]["stats"].items()]
    print(pd.DataFrame(rows).to_string(index=False))
    h = r.get("holdout") or {}
    if h.get("status"):
        print(f"holdout: {h['status']}" + (f" - {h['note']}" if h.get("note") else ""))
    if (r.get("execution") or {}).get("mode"):
        print(f"execution mode: {r['execution']['mode']}")
    ic = r["ic"]["all"]
    print(f"\nrank IC {ic['mean']:.4f} (IR {_fmt(ic['ir'])}, t {_fmt(ic['t'])}, hit rate {_fmt(ic['hit_rate'], 'pct')}); turnover {_fmt(r['metrics']['oos'].get('ann_turnover'))}x a year; "
          f"generalisation score {_fmt(r['overfit']['generalization_score'])}")
    for w in r["overfit"]["warnings"]:
        print(f"  ! {w}")
    if r.get("robustness"):
        rb = r["robustness"]
        print(f"\nrobustness: {rb['n_pass']} pass, {rb['n_warn']} warn, {rb['n_fail']} fail (score {_fmt(rb['score'])})")
        for t in rb["tests"]:
            print(f"  [{t['verdict']:>4}] {t['title']}: {t['detail']}")
    if registry is not None and r["id"] != "unsaved":
        print(f"\nsaved to {registry.root / 'experiments' / r['id']} (report.html, report.json, predictions, positions, returns, model)")


def cmd_train(args, config) -> int:
    registry = None if args.no_save else Registry(args.registry)
    over = _experiment_overrides(args)
    bundle = info = None
    if args.prices:
        bundle, info = _bundle_and_info(args, config)
    if getattr(args, "study", None):
        over["study"] = args.study
    try:
        exp = run_experiment(over, bundle=bundle, config=config, registry=registry, save=not args.no_save, info=info, progress=lambda m: print(f"  {m}", file=sys.stderr, flush=True) if args.verbose else None,
                             explain=not args.no_explain, stop_after="validation" if getattr(args, "development", False) else None, entry_point="cli")
    except (ExperimentError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_run(exp, registry)
    return 0


def cmd_backtest(args, config) -> int:
    registry = Registry(args.registry)
    over = {"signal": {k: v for k, v in (("method", args.signal), ("n", args.n), ("holding_period", args.holding_period), ("max_weight", args.max_weight), ("quantile", args.quantile)) if v is not None},
            "backtest": {k: v for k, v in (("rebalance", args.rebalance), ("cost_bps", args.cost_bps), ("slippage_bps", args.slippage_bps), ("target_vol", args.target_vol), ("extra_lag", args.extra_lag)) if v is not None}}
    try:
        parent = registry.report(args.experiment)
        exp = rebacktest(registry, args.experiment, over, save=not args.no_save)
    except (ExperimentError, RegistryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_run(exp, registry if not args.no_save else None)
    print("\nagainst the original run (same predictions, no refit):")
    rows = []
    for seg in ("train", "validation", "test", "forward"):
        a, b = parent["segments"]["stats"].get(seg), exp.report["segments"]["stats"].get(seg)
        if a and b:
            rows.append({"segment": seg, "original sharpe": _fmt(a["sharpe"]), "new sharpe": _fmt(b["sharpe"])})
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def text_execution(block: dict, parent: str | None = None) -> str:
    """The evaluation mode of a result as plain text: what it assumes, what it earned before and after costs, how it reconciles, and what qualifies the claim."""
    lines = [f"execution mode: {block.get('mode')}" + (f" (a re-run of {parent})" if parent else "")]
    if block.get("mode") != "realistic_cash_equity":
        lines.append("  assumptions: " + "; ".join(f"{k} {v}" for k, v in (block.get("assumptions") or {}).items()))
        lines.append("  not modelled: " + ", ".join(block.get("not_modelled", [])))
        lines.append("  cannot support: " + ", ".join(block.get("cannot_support", [])) + ("" if block.get("recorded", True) else f"  [{block.get('note')}]"))
        return "\n".join(lines)
    t, g, rec = block["timeline"], block["gross_net"], block["reconciliation"]
    lines += [f"  timeline: decided at the close, filled at the close {t['delay_sessions']} session(s) later ({t['convention']}, sizing {t['sizing']}); the lag is applied once, by the adapter",
              f"  price convention: {block['price_convention']}",
              f"  gross P&L {g['gross_pnl']:,.0f} - costs {g['costs_total']:,.0f} = net {g['net_pnl']:,.0f}  (net {g['net_return']:+.2%} of capital {block['assumptions']['capital']:,.0f}; costs {g['costs_pct_of_capital']:.2%})",
              "  costs: " + ", ".join(f"{k} {v:,.0f}" for k, v in block["costs"].items() if v),
              f"  reconciliation: gross less costs vs net gap {rec['gap']:.2e} ({'reconciled' if rec['reconciled'] else 'DOES NOT RECONCILE'}); ledger identities {'hold' if rec.get('ledger_ok') else 'FAIL'}",
              f"  turnover {block['ann_turnover']:.1f}x a year; {block['n_fills']} fills; orders {block['order_states']}"]
    cap = block.get("capacity", {})
    if cap.get("levels"):
        lines.append(f"  capacity (last target, {cap['date']}): " + "; ".join(f"{c['capital']:,.0f}: {c['fill_share']:.0%} filled, {c['impact_bps_of_traded']:.1f} bps impact" for c in cap["levels"]))
    else:
        lines.append(f"  capacity: not established ({cap.get('reason', '')})")
    cmp = block.get("comparison_with_vectorized")
    if cmp:
        lines.append(f"  against the vectorized engine on the same weights: total return {cmp['vectorized_total_return']:+.2%} -> {cmp['realistic_total_return']:+.2%}, Sharpe {cmp['vectorized_sharpe']:.2f} -> {cmp['realistic_sharpe']:.2f}")
    lines.append(f"  claim: {block['capability']['realistic_execution_claim']}" + (" - blocked by: " + "; ".join(block["capability"]["blockers"]) if block["capability"]["blockers"] else ""))
    lines += [f"  limit: {x}" for x in block.get("limitations", [])]
    return "\n".join(lines)


def cmd_realistic(args, config) -> int:
    registry = Registry(args.registry)
    ex: dict = {"mode": "realistic_cash_equity"}
    for key, value in (("capital", args.capital), ("commission_bps", args.commission_bps), ("half_spread_bps", args.half_spread_bps), ("slippage_bps", args.slippage_bps), ("adv_window", args.adv_window),
                       ("sizing", args.sizing), ("expiry_sessions", args.expiry_sessions), ("cash_rate", args.cash_rate), ("financing_spread", args.financing_spread), ("shorts", args.shorts),
                       ("min_adv_dollars", args.min_adv_dollars), ("min_listing_days", args.min_listing_days)):
        if value is not None:
            ex[key] = value
    if args.participation is not None:
        ex["max_participation"] = None if args.participation <= 0 else args.participation                      # 0 means no limit, which the report then states
    if args.impact_coefficient is not None:
        ex["impact"] = {"coefficient": args.impact_coefficient}
    over: dict = {"execution": ex}
    if args.name:
        over["name"] = args.name
    try:
        bundle = None
        if args.prices:
            bundle, _ = _bundle_and_info(args, config)
        exp = rebacktest(registry, args.experiment, over, save=not args.no_save, bundle=bundle, config=config)
    except (ExperimentError, RegistryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    block = exp.report["execution"]
    if args.json:
        print(json.dumps(jsonable(block), indent=1))
        return 0
    print(f"experiment {exp.id}: {exp.report['name']}")
    print(text_execution(block, args.experiment))
    return 0


def text_report(report: dict) -> str:
    """One experiment as plain text."""
    out = [f"{report['name']}  [{report['id']}]", f"  created {report['created']}; config {report['config_hash']}; data {report['provenance']['source']} ({report['provenance']['label']}); "
                                                  f"{report['trials']['dataset_trials']} looks at this sample", ""]
    rows = [{"segment": k, "days": v["n_days"], "return": _fmt(v["ann_return"], "pct"), "vol": _fmt(v["ann_vol"], "pct"), "sharpe": _fmt(v["sharpe"]), "max dd": _fmt(v["max_drawdown"], "pct")}
            for k, v in report["segments"]["stats"].items()]
    out.append(pd.DataFrame(rows).to_string(index=False))
    ic = report["ic"]["all"]
    out += ["", f"rank IC {_fmt(ic['mean'])} (IR {_fmt(ic['ir'])}, t {_fmt(ic['t'])}); decay: " + ", ".join(f"{d['horizon']}d {d['mean_ic']:.3f}" for d in report["ic"]["decay"] if d["mean_ic"] == d["mean_ic"]),
            f"generalisation score {_fmt(report['overfit']['generalization_score'])}; deflated Sharpe probability {_fmt(report['overfit']['deflated_sharpe_probability'], 'pct')}"]
    out += [f"  ! {w}" for w in report["overfit"]["warnings"]]
    if report["importance"]["bars"]:
        out += ["", "top features: " + ", ".join(f"{b['feature']} ({_fmt(b['share'], 'pct')})" for b in report["importance"]["bars"][:8])]
    if report.get("robustness"):
        out += ["", f"robustness score {_fmt(report['robustness']['score'])}"] + [f"  [{t['verdict']:>4}] {t['title']}: {t['detail']}" for t in report["robustness"]["tests"]]
    folds = [f for f in report["folds"].get("list", []) if not f.get("skipped")]
    if folds and folds[0].get("fit_cutoff"):
        out += ["", "fold lineage (fitted on rows whose labels had ended by the fit cutoff):"]
        out += [f"  fold {f['fold']}: fitted {f['fit_cutoff']} on {f['n_train']:,} rows (labels end by {f['label_end_max']}, rows {f['train_rows_hash']}); predicts {f['test_start']} to {f['test_end']}" for f in folds[:12]]
        if len(folds) > 12:
            out.append(f"  ... {len(folds) - 12} more folds")
    elif folds:
        out += ["", "fold lineage: not recorded (the experiment was saved before it was kept)"]
    if report.get("holdout"):
        h = report["holdout"]
        out += ["", f"holdout: {h.get('status')}" + (f" (plan {h['plan_id']})" if h.get("plan_id") else "") + (f" - {h['note']}" if h.get("note") else "")]
    if report.get("reproducibility", {}).get("identity"):
        rp = report["reproducibility"]
        out.append(f"provenance: identity {rp['identity'][:16]}; data {rp['data_hash'][:12]}, code {rp['code_hash'][:12]} ({rp.get('git_commit') and rp['git_commit'][:8] or rp.get('git_note')}), environment {rp['environment_hash'][:12]}")
    out += ["", text_execution(report["execution"])] if report.get("execution") else ["", "execution mode: vectorized_legacy (this experiment was saved before execution modes were recorded)"]
    dep = report.get("dependence")
    if dep:
        out.append(f"overlapping labels: mean uniqueness {_fmt(dep.get('mean_uniqueness'))}; treatment {dep['treatment']}" + (f" ({'; '.join(dep['notes'])})" if dep.get("notes") else ""))
    if report.get("audit"):
        out += ["", "dataset audit:"] + [f"  {line}" for line in audit_lines(report["audit"])]
    return "\n".join(out)


def cmd_report(args, config) -> int:
    registry = Registry(args.registry)
    try:
        if not args.experiment or args.list:
            rows = leaderboard_payload(registry.leaderboard(args.sort, args.limit))
            if not rows:
                print("no experiments recorded yet; run ml-train first")
                return 0
            cols = ["id", "name", "model", "validation.sharpe", "test.sharpe", "forward.sharpe", "ic.mean", "overfit.generalization_score", "trials"]
            frame = pd.DataFrame(rows)[cols]
            with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 38, "display.float_format", "{:.2f}".format):
                print(frame.to_string(index=False))
            return 0
        report = registry.report(args.experiment)
    except (RegistryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(jsonable(report), indent=1))
    elif args.html:
        path = registry.root / "experiments" / args.experiment / "report.html"
        if isinstance(args.html, str) and args.html != "-":
            Path(args.html).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"wrote {args.html}")
        else:
            print(path)
    else:
        print(text_report(report))
    return 0


def cmd_verify(args, config) -> int:
    """Show the persisted verification evidence of the model-correctness requirements, or run the tests behind them and record the result."""
    from . import verification as vf

    registry = Registry(args.registry)
    try:
        if args.run:
            rows = vf.run_verification(registry, args.requirement or None, progress=lambda msg: print(msg, file=sys.stderr))
            rows = [{"requirement_id": r["requirement_id"], "status": r["status"], "reason": r["reason"]} for r in rows]
        else:
            rows = vf.current(registry, args.requirement[0] if args.requirement else None)
    except vf.VerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(jsonable(rows), indent=1))
    else:
        for r in rows:
            flag = " (stale)" if r.get("stale") else ""
            print(f"{r['requirement_id']:7s} {r['status']:9s}{flag}  {str(r.get('reason') or '')[:110]}")
    if args.require_pass:
        return 0 if all(r["status"] == "pass" and not r.get("stale") for r in rows) else 1
    return 1 if any(r["status"] == "fail" for r in rows) else 0


# ------------------------------------------------------------------------------------------------------------------ research controls (Release 3)
def _print_json_or(args, payload, lines) -> None:
    if getattr(args, "json", False):
        print(json.dumps(jsonable(payload), indent=1, default=str))
    else:
        print("\n".join(lines))


def cmd_lineage(args, config) -> int:
    """A study's record: every try, its state and parent, the raw look counts, the plans and the holdout access history."""
    from . import research_controls as rc

    registry = Registry(args.registry)
    try:
        data = rc.lineage(registry, args.study, args.experiment)
    except (rc.ControlError, RegistryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lines = [f"study {data['study'].get('study_id')} (continues {', '.join(data['studies'][1:]) or 'nothing'}): {data['raw_looks']} tries recorded; by kind {data['by_kind']}; by state {data['by_state']}",
             f"  {len(data['feature_sets'])} feature sets and {len(data['configurations'])} configurations tried; {len(data['plans'])} evaluation plans; {len(data['holdout_access'])} holdout accesses; "
             f"{len(data['tombstones'])} deleted experiments", f"  {data['note']}"]
    lines += [f"  {o['registered']}  {o['kind']:<18} {o['state']:<9} {'counted' if o['counts'] else 'not counted':<12} {o['op_key']}" + (f" -> {o['experiment_id']}" if o['experiment_id'] else "") for o in data["operations"][-args.limit:]]
    _print_json_or(args, data, lines)
    return 0


def _read_overrides(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def cmd_plan(args, config) -> int:
    """Create (and reserve) an evaluation plan from candidate configurations, freeze it, or show it."""
    from . import frozen_eval as fe
    from . import research_controls as rc

    registry = Registry(args.registry)
    try:
        if args.action == "create":
            if not args.candidate:
                print("error: give at least one --candidate FILE.yaml (experiment overrides)", file=sys.stderr)
                return 2
            bundle, info = (None, None)
            if args.prices:
                bundle, info = _bundle_and_info(args, config)
            state = fe.plan_and_reserve(registry, args.study, [_read_overrides(c) for c in args.candidate], _read_overrides(args.baseline) if args.baseline else None, freeze=args.freeze, actor="cli", bundle=bundle,
                                        config=config, info=info, selection_adjustment=args.selection_adjustment, uncertainty_method=args.uncertainty_method)
        elif args.action == "freeze":
            state = rc.freeze_plan(registry, args.plan, "cli")
        else:
            state = rc.plan_state(registry, args.plan)
    except (rc.ControlError, ExperimentError, RegistryError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    res = state.get("reservation") or {}
    lines = [f"plan {state['plan_id']}: {state['state']} (content {state['content_hash'][:12]}, study {state['study_id']})"]
    if res:
        lines.append(f"  holdout {res['reservation_id']}: {len(res['assets'])} assets, {res['range'][0]} to {res['range'][1]}; " + ("untouched" if res["untouched"] else f"NOT untouched: {len(res['overlaps'])} earlier look(s) overlap it"))
    _print_json_or(args, state, lines)
    return 0


def cmd_evaluate(args, config) -> int:
    """The one evaluation event of a frozen plan. The same --key returns the same result; a second key on a consumed plan needs --allow-reuse."""
    from . import frozen_eval as fe
    from . import research_controls as rc

    registry = Registry(args.registry)
    try:
        bundle = _bundle_and_info(args, config)[0] if args.prices else None
        out = fe.evaluate_plan(registry, args.plan, idempotency_key=args.key, actor="cli", entry_point="cli", allow_reuse=args.allow_reuse, resume=args.resume, bundle=bundle, config=config,
                               progress=lambda m: print(f"  {m}", file=sys.stderr, flush=True))
    except (rc.ControlError, ExperimentError, RegistryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out.pop("_experiments", None)
    results = out.get("experiments") or (out.get("result") or {}).get("experiments") or []
    lines = [f"evaluation {out['status']} (plan {args.plan}, key {args.key})"] + [f"  {r['role']:<9} {r['name']}  sharpe {_fmt(r.get('sharpe'))}  -> {r['experiment_id']}" for r in results]
    _print_json_or(args, out, lines)
    return 0


def cmd_holdout(args, config) -> int:
    from . import research_controls as rc

    registry = Registry(args.registry)
    data = rc.holdout_status(registry, args.study, args.plan)
    lines = [f"{len(data['plans'])} plan(s), {len(data['exposures'])} exposure(s) of protected outcomes, {len(data['accesses'])} ledger entries"]
    for p in data["plans"]:
        r = p.get("reservation") or {}
        lines.append(f"  {p['plan_id']} {p['state']:<9} evaluations {p['evaluations']}" + (f"  {r['range'][0]}..{r['range'][1]} " + ("untouched" if r["untouched"] else "not untouched") if r else "  no reservation"))
    _print_json_or(args, data, lines)
    return 0


def cmd_null_controls(args, config) -> int:
    """Freeze a null-study design and run the whole search pipeline on placebo data (or list the studies)."""
    from . import null_controls as nc
    from . import research_controls as rc

    registry = Registry(args.registry)
    if args.list:
        with registry._conn() as c:
            rows = [dict(r) for r in c.execute("SELECT null_id, study_id, state, planned, done, result FROM null_studies ORDER BY rowid DESC LIMIT 50")]
        for r in rows:
            res = json.loads(r["result"]) if r["result"] else {}
            print(f"{r['null_id']}  {r['state']:<8} {r['done']}/{r['planned']}  rate {res.get('rate')}  sufficient {res.get('sufficient')}  {res.get('verdict', '')[:90]}")
        return 0
    over = _read_overrides(args.config) if args.config else {}
    spec = nc.NullSpec(study_id=args.study, base=over, kinds=tuple(args.kinds), n_random=args.n_random, n_bayes=args.n_bayes, replications=args.replications, seed=args.seed, tolerance=args.tolerance,
                       precision=args.precision, min_replications=args.min_replications, min_shift=args.min_shift)
    try:
        made = nc.register(registry, spec)
        print(f"design frozen as {made['null_id']} ({made['spec_hash'][:12]})", file=sys.stderr)
        bundle, info = (None, None)
        if args.prices:
            bundle, info = _bundle_and_info(args, config)
        out = nc.run(registry, made["null_id"], nc.search_pipeline(spec, bundle=bundle, config=config, info=info), progress=lambda m: print(f"  {m}", file=sys.stderr, flush=True))
    except (rc.ControlError, ExperimentError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_json_or(args, out, [f"{out['null_id']}: {out['false_promotions']} false promotion(s) in {out['replications']} replication(s) (rate {_fmt(out['rate'], 'pct')}, exact interval "
                               f"{_fmt(out['exact_interval'][0], 'pct')} to {_fmt(out['exact_interval'][1], 'pct')})", f"  {out['verdict']}"])
    return 0 if out["controlled"] is not False else 1


def cmd_selection_tests(args, config) -> int:
    """Apply the selection-bias procedures to a study's recorded trials where they apply, and persist each result or the reason it was not applied."""
    from . import policy as pol
    from . import selection as sel

    registry = Registry(args.registry)
    adopted = pol.adopted(registry)
    out = sel.evaluate_study(registry, args.study, n_boot=args.n_boot, seed=args.seed, criteria=sel.criteria_from_policy(adopted[0] if adopted else None))
    lines = [f"study {args.study}: {out['n_tries']} tries recorded, {out['n_candidates']} with return streams", f"  {out['statement']}"] + [f"  {''.join(e['method']):<14} {e['status']:<15} {e['reason'][:100] or ('met' if e['statistic'].get('passes') else 'not met')}" for e in out["evidence"]]
    _print_json_or(args, out, lines)
    return 0


def cmd_policy(args, config) -> int:
    """Show the validation policy in force, or adopt one (a named person and a reason are required)."""
    from . import policy as pol

    registry = Registry(args.registry)
    try:
        if args.adopt_example or args.file:
            if args.adopt_example and args.file:
                raise ValueError("give --file or --adopt-example, not both")
            if args.file:
                content = json.loads(Path(args.file).read_text())
                policy = pol.ValidationPolicy(str(content["name"]), str(content["version"]), dict(content.get("criteria") or {}), False, str(content.get("description", "")))
            else:
                policy = pol.example_policy()
            made = pol.adopt(registry, policy, args.actor or "", args.note or "")
            print(f"adopted {made['name']} {made['version']} ({made['policy_hash']}) by {made['actor']}" + (" - an ILLUSTRATIVE example, not a recommendation" if made["illustrative"] else ""))
        adopted = pol.adopted(registry)
        history = pol.history(registry, 10)
    except (pol.PolicyError, ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = {"adopted": None if adopted is None else {"policy": adopted[0].content(), "policy_hash": adopted[0].content_hash(), "adoption": adopted[1]}, "history": history, "example": pol.example_policy().content()}
    lines = ["no validation policy is adopted: the platform implies no Sharpe, deflated-Sharpe, PBO, p-value or paper-trading threshold, and Gates B and C are blocked"] if adopted is None else \
        [f"policy in force: {adopted[0].name} {adopted[0].version} ({adopted[0].content_hash()}), adopted by {adopted[1]['actor']} on {adopted[1]['at']}: {adopted[1]['note']}" + (" [ILLUSTRATIVE]" if adopted[0].illustrative else "")] + \
        [f"  {k} = {v}" for k, v in sorted(adopted[0].criteria.items()) if v is not None]
    _print_json_or(args, payload, lines)
    return 0


def cmd_diagnostics(args, config) -> int:
    """Compute and store the statistical diagnostics of one experiment under the adopted policy: three Sharpe estimates, matched benchmarks, regimes, calibration and the study's selection evidence."""
    from . import diagnostics as dx
    from . import policy as pol

    registry = Registry(args.registry)
    market: dict = {}
    try:
        grid = json.loads(args.neighborhood) if args.neighborhood else None
        delays = [int(x) for x in args.delays.split(",")] if args.delays else None
        capitals = [float(x) for x in args.capitals.split(",")] if args.capitals else None
        if grid or delays or capitals:
            bundle = _bundle_and_info(args, config)[0] if args.prices else None
            market = {"bundle": bundle, "config": config}
        out = dx.run(registry, args.experiment, actor="cli", n_boot=args.n_boot, seed=args.seed, neighborhood_grid=grid, delays=delays, capitals=capitals, **market)
        adopted = pol.adopted(registry)
        items = dx.evaluate_policy(adopted[0], out) if adopted and out["policy_hash"] == adopted[0].content_hash() else []
    except (ExperimentError, RegistryError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sh = out["sharpe"]
    ci = lambda r: "n/a" if not r.get("interval") else f"[{r['interval'][0]:.2f}, {r['interval'][1]:.2f}]"          # noqa: E731
    lines = [f"{args.experiment}: {out['sample']['n']} days on the {out['segment']} segment ({out['sample']['first']} to {out['sample']['last']}), policy {out['policy_hash'] or 'none adopted'}",
             f"  annualised Sharpe {_fmt(sh['naive']['value'])}: naive interval {ci(sh['naive'])} (assumes independent days) | HAC {ci(sh['hac'])} | stationary bootstrap {ci(sh['bootstrap'])}"]
    for b in out["benchmarks"].values():
        am, kind = b["active_mean"], b["kind"]
        lines.append(f"  vs {kind:<26} {'matched' if b['matched'] else 'UNMATCHED (descriptive)':<22} active mean/day " + ("n/a" if am.get("value") is None else f"{am['value']:.5f} {ci(am)}") + f" over {b['n_aligned']} days")
    applied = [r for r in out["regimes"]["rows"] if r.get("eligibility") == "applied"]
    lines.append(f"  regimes: {len(applied)} readable of {len(out['regimes']['rows'])} (a regime with fewer than {out['regimes']['min_days']} days is insufficient evidence)")
    for key, label in (("neighborhood", "neighbourhood"), ("delay", "delay"), ("capacity", "capacity")):
        if key in out:
            lines.append(f"  {label}: {out[key]['eligibility']}" + (f" - {out[key]['reason']}" if out[key].get("reason") else ""))
    lines += [f"  [{i['status']:^9}] {i['label']}" + (f": {i['reason']}" if i["reason"] else "") for i in items] if items else ["  no policy item is read: " + ("no policy is adopted" if out["policy_hash"] is None else "the diagnostics were computed under another policy")]
    _print_json_or(args, {k: v for k, v in out.items() if k != "regime_declaration"}, lines)
    return 0


def cmd_governance(args, config) -> int:
    """Strategy versions, risk limits, operating state, recovery procedure and approvals of an experiment; with a flag, record one action by a named person."""
    from . import governance as gv

    registry = Registry(args.registry)
    try:
        registry.get_experiment(args.experiment)
        actor = args.actor or ""
        if args.register:
            limits = gv.RiskLimits(json.loads(args.limits), "cli") if args.limits else None
            made = gv.register_version(registry, args.experiment, actor=actor, risk_limits=limits)
            print(f"version {made['version_id']} {'registered' if made['created'] else 'already registered'}")
        versions = gv.versions_of(registry, args.experiment)
        vid = args.version or (versions[-1]["version_id"] if versions else None)
        if vid and args.halt:
            gv.halt(registry, vid, args.halt, actor)
        if vid and args.resume:
            gv.resume(registry, vid, args.resume, actor)
        if vid and args.rollback_to:
            gv.rollback(registry, vid, args.rollback_to, args.reason or "", actor)
        if vid and args.recovery:
            gv.document_recovery(registry, vid, args.recovery, actor)
        if vid and args.approve:
            gv.approve_deployment(registry, vid, actor=actor, note=args.approve, stage=args.stage)
        summary = gv.summary(registry, vid) if vid else None
    except (gv.GovernanceError, RegistryError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if summary is None:
        _print_json_or(args, {"experiment_id": args.experiment, "versions": []}, ["no strategy version is registered for this experiment (use --register [--limits JSON] --actor NAME)"])
        return 0
    st, lim = summary["state"], summary["risk_limits"]
    lines = [f"{args.experiment}: version {summary['version']['version_id']} ({len(versions)} registered), {st['state']}" + (f" since {st['since']}: {st['reason']}" if st["state"] != "active" else ""),
             "  risk limits: " + (", ".join(f"{k}={v}" for k, v in lim["limits"].items()) if lim and lim["limits"] else "none documented"),
             "  recovery procedure: " + ("recorded" if summary["recovery_plan"] else "not recorded"),
             "  last monitoring: " + (f"{summary['last_monitoring']['at']} {summary['last_monitoring']['detail']['status']}" if summary["last_monitoring"] else "never"),
             f"  deployment approvals recorded: {len(summary['approvals'])}"]
    _print_json_or(args, summary, lines)
    return 0


def cmd_forward(args, config) -> int:
    """One forward-test step: record today's signal and commit it to the immutable ledger, settle the commitments whose window has completed, check monitoring and print the ledger's view."""
    from . import monitoring as mon

    registry = Registry(args.registry)
    try:
        bundle = _bundle_and_info(args, config)[0] if args.prices else None
        tracker = mon.ForwardTracker(registry, args.experiment, config)
        signal = None if args.no_record else tracker.step(bundle, args.as_of, args.capital)
        settled = tracker.settle(bundle, args.as_of)
        summary = tracker.summary(bundle, args.as_of, monitor=True)
    except (ExperimentError, RegistryError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    led = summary.get("ledger") or {}
    lines = []
    if signal:
        c = signal["commitment"]
        lines.append(f"{signal['date']}: {c['status']}" + (f" as {c['classification']}: {c['why']}" if c.get("classification") else f": {c.get('reason', '')}"))
    lines.append(f"settled {len(settled)} signal(s) this run")
    if led.get("strategy_version"):
        cnt = led["counts"]
        lines.append(f"ledger {led['strategy_version']} ({led['state']['state']}): prospective {cnt['prospective']['settled']}/{cnt['prospective']['committed']} settled over {cnt['prospective']['distinct_settled_days']} day(s); "
                     f"replays {cnt['replay']['committed']} (never counted)")
        for cls, eq in (led.get("equity") or {}).items():
            lines.append(f"  {cls} equity: " + (f"{eq['n_days']} days, total return {_fmt(eq['total_return'], 'pct')}" + ("" if eq.get("sharpe") is not None else f" ({eq['reason']})") if eq.get("eligibility") == "applied" else eq["reason"]))
    lines += [f"alert: {a}" for a in summary["alerts"]]
    _print_json_or(args, {"signal": signal, "settled": settled, "summary": summary}, lines)
    return 0


def cmd_multiseed(args, config) -> int:
    """Train an experiment's own configuration once per predeclared seed (development mode) and report the spread across seeds beside the time-series uncertainty of each. No seed is chosen."""
    from . import diagnostics as dx

    registry = Registry(args.registry)
    try:
        seeds = [int(x) for x in args.seeds.split(",")]
        bundle = _bundle_and_info(args, config)[0] if args.prices else None
        out = dx.multiseed(registry, args.experiment, seeds, n_boot=args.n_boot, actor="cli", bundle=bundle, config=config)
    except (ExperimentError, RegistryError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lines = [f"{args.experiment}: {len(out['rows'])} seeds {out['declaration']['seeds']} on the {out['segment']} segment, {out['metric']} per seed (the point estimate is the median; no seed is chosen)"]
    lines += [f"  seed {r['seed']:>6}: {r['value'] if r['value'] is None else round(r['value'], 3)}  HAC interval {r['sharpe_hac_interval'] and [round(x, 2) for x in r['sharpe_hac_interval']]}" for r in out["rows"]]
    if out.get("summary"):
        s = out["summary"]
        lines.append(f"  median {s['median']:.3f}, spread across seeds (std) {s['std_across_seeds']:.3f}, a seed's own sampling error {s['mean_time_series_se'] if s['mean_time_series_se'] is None else round(s['mean_time_series_se'], 3)}: {out['flag']}")
    else:
        lines.append(f"  {out['eligibility']}: {out.get('reason', '')}")
    _print_json_or(args, out, lines)
    return 0


def cmd_promotion(args, config) -> int:
    """The three independent gates of an experiment, computed from the persisted evidence."""
    from . import gates

    registry = Registry(args.registry)
    try:
        res = gates.decide(registry, args.experiment, "cli")
    except (RegistryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lines = [f"{args.experiment} ({res['model_kind']}), policy {res['policy_version']}"]
    for name, g in res["gates"].items():
        lines.append(f"  Gate {name}: {g['status']}{' (stale)' if g['stale'] else ''}" + (f" - {g['reasons'][0][:140]}" if g["reasons"] else ""))
    lines.append(f"  {res['note']}")
    _print_json_or(args, res, lines)
    return 0


def cmd_promote(args, config) -> int:
    """Move a registered model to a stage. The registry decides from the persisted evidence; production also needs a recorded human approval."""
    from . import gates

    registry = Registry(args.registry)
    approval = {"actor": args.approval_actor, "note": args.approval_note} if args.approval_actor else None
    try:
        out = registry.promote(args.model, args.stage, actor="cli", note=args.note or "", approval=approval)
    except gates.PromotionBlocked as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    except (RegistryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{args.model} is now {args.stage} (policy {out.get('policy_version')})")
    return 0


def _ci(r: dict | None) -> str:
    return "n/a" if not r or not r.get("interval") else f"[{r['interval'][0]:+.4f}, {r['interval'][1]:+.4f}]"


def cmd_alpha(args, config) -> int:
    """Alpha diagnostics of a saved experiment: rank IC with a dependence-aware interval, decay, coverage, stability and redundancy, and (given a baseline) the conditional IC and the blend."""
    from . import diagnostics as dx
    from . import r6_reports as r6

    registry = Registry(args.registry)
    try:
        out = r6.alpha_diagnostics(registry, args.experiment, segment=args.segment, baseline_id=args.baseline)
        dx.store(registry, "alpha_diagnostics", args.experiment, out, actor="cli")
    except (ExperimentError, RegistryError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(out, indent=1, allow_nan=False))
        return 0
    if out["eligibility"] != "applied":
        print(f"{args.experiment}: {out['reason']}")
        return 0
    ic = out["ic"]
    print(f"{args.experiment} ({out['model']} on {out['target']}), {out['segment']} segment: mean rank IC {_fmt(ic['mean']['value'])} {_ci(ic['mean'])} over {ic['n_with_ic']} dates "
          f"(Newey-West lag {ic['mean']['settings']['lags']}, horizon {out['horizon']}); hit rate {_fmt(ic['hit_rate'])}")
    if ic["unavailable_by_reason"]:
        print("  dates without an IC: " + ", ".join(f"{k} {v}" for k, v in ic["unavailable_by_reason"].items()))
    cells = {(c["extra_lag"], c["horizon"]): c for c in out["decay"]["cells"]}
    print("  decay (mean IC by extra delay in rows, horizon in columns): horizons " + " ".join(f"{h:>7}" for h in sorted({h for _, h in cells})))
    for k in sorted({k for k, _ in cells}):
        print(f"    +{k} sessions  " + " ".join(f"{_fmt(cells[(k, h)]['mean_ic']):>7}" for h in sorted({h for _, h in cells})))
    cov, st, red = out["coverage"], out["stability"], out["redundancy"]
    print(f"  coverage {_fmt(cov.get('coverage'))} of investable cells; stability: same sign in {_fmt(st.get('same_sign_share'))} of blocks; features: {red.get('n_features')} with about {_fmt(red.get('effective_independent_features'))} independent")
    inc = out.get("incremental")
    if inc:
        if inc["eligibility"] != "applied":
            print(f"  incremental value: {inc['eligibility']} ({inc['reason']})")
        else:
            print(f"  given {args.baseline}: conditional IC {_fmt(inc['conditional_ic']['mean']['value'])} {_ci(inc['conditional_ic']['mean'])} (plain {_fmt(inc['plain_ic']['mean']['value'])}); blend vs baseline Sharpe difference "
                  f"{_fmt((inc['blend'].get('difference') or {}).get('value'))} {_ci(inc['blend'].get('difference'))}  [predictive, not causal]")
    return 0


def cmd_delay(args, config) -> int:
    """The delay report of a saved experiment: rank IC, expected edge, net result and capacity when its book is executed later, at delays declared on the command line (0 must be among them)."""
    from . import delay_report as dly
    from . import diagnostics as dx

    registry = Registry(args.registry)
    try:
        delays = [int(x) for x in str(args.delays).split(",") if x.strip() != ""]
        out = dly.report(registry, args.experiment, segment=args.segment, declaration=dly.declare(delays))
        dx.store(registry, "delay_report", args.experiment, out, actor="cli")
    except (ExperimentError, RegistryError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(out, indent=1, allow_nan=False))
        return 0
    if out["eligibility"] != "applied":
        print(f"{args.experiment}: {out['reason']}")
        return 0
    print(f"{args.experiment} ({out['model']} on {out['target']}), {out['segment']} segment, delays {out['declaration']['delays']} (declaration {out['declaration']['hash']}): {out['rule']}")
    print(f"  {'delay':>5} {'calendar days':>14} {'rank IC':>8} {'ratio':>6} {'gross/yr':>9} {'net/yr':>8} {'Sharpe':>7} {'break-even bps':>15}  capacity")
    for r in out["rows"]:
        p, cap = r["portfolio"], r["capacity"]
        cal = r["timing"]["calendar_days"]["mean"]
        print(f"  {r['delay']:>5} {_fmt(cal):>14} {_fmt(r['ic']['mean_ic']):>8} {_fmt(r['ic_ratio_to_undelayed']):>6} {_fmt(p.get('gross_ann_return')):>9} {_fmt(p.get('net_ann_return')):>8} "
              f"{_fmt(p.get('sharpe')):>7} {_fmt(p.get('breakeven_cost_bps')):>15}  {cap['status'] if cap['status'] != 'measured' else 'measured'}")
    print(f"  declared one-way cost {_fmt(out['declared_cost_bps'])} bps; a ratio is undefined when the undelayed result is not positive")
    return 0


def cmd_portfolio(args, config) -> int:
    """Portfolio diagnostics of a saved experiment: the book it held, covariance estimators compared on realised risk, risk contributions, and (for a return forecast) the cost-aware allocation's sensitivity."""
    from . import diagnostics as dx
    from . import r6_reports as r6

    registry = Registry(args.registry)
    try:
        out = r6.portfolio_diagnostics(registry, args.experiment, segment=args.segment, horizon=args.horizon, max_dates=args.dates)
        dx.store(registry, "portfolio_diagnostics", args.experiment, out, actor="cli")
    except (ExperimentError, RegistryError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(out, indent=1, allow_nan=False))
        return 0
    if out["eligibility"] != "applied":
        print(f"{args.experiment}: {out['reason']}")
        return 0
    b = out["book"]
    print(f"{args.experiment}, {out['segment']} segment: {b['days_with_a_book']} days with a book; gross {_fmt(b['gross']['mean'])}, net {_fmt(b['net']['mean'])}, effective positions {_fmt(b['concentration']['effective_positions'])}, "
          f"mean daily turnover {_fmt(b['mean_daily_turnover'])}")
    print(f"  covariance estimators on the risk the book realised over {out['risk_models']['horizon']} days ({out['risk_models']['decision_dates']} dates; forecast/realised 1.0 is unbiased):")
    for name, e in out["risk_models"]["estimators"].items():
        print(f"    {name:<34} windows {e['windows']:>3}  ratio {_fmt(e['risk_ratio_forecast_to_realised'])}  QLIKE {_fmt(e['qlike'])}  worst condition {_fmt(e['worst_condition'])}  repairs {_fmt(e['repair_rate'])}")
    al = out["allocation"]
    if al["eligibility"] != "applied":
        print(f"  allocation sensitivity: {al['eligibility']} ({al['reason']})")
    else:
        sm = al["summary"] or {}
        print(f"  allocation sensitivity on {al['asof']} ({len(al['points'])} grid points, {al['failed']} without a solution): largest change in the book {_fmt(sm.get('max_weight_change'))}, share of points that do not trade {_fmt(sm.get('share_no_trade'))}")
    return 0


def cmd_improve(args, config) -> int:
    """Freeze and run a comparison card from a JSON spec (family, hypothesis, one_change, baseline, candidates, ...), or show a stored card. The arms are development runs; nothing is promoted."""
    from . import comparison as cmpn

    registry = Registry(args.registry)
    try:
        if args.card:
            card = cmpn.get(registry, args.card)
        else:
            spec = json.loads(Path(args.spec).read_text())
            bundle = _bundle_and_info(args, config)[0] if args.prices else None
            card = cmpn.new_card(spec["hypothesis"], spec["family"], spec["baseline"], spec["candidates"], study_id=spec.get("study") or f"improve-{spec['family']}", one_change_description=spec["one_change"],
                                 primary_metric=spec.get("primary_metric"), parsimony=spec.get("parsimony"), min_effect=float(spec.get("min_effect", 0.0)), n_boot=int(spec.get("n_boot", 1000)))
            cmpn.freeze(registry, card, actor="cli")
            card = cmpn.run(registry, card["card_id"], bundle=bundle, config=config, progress=lambda m: print(f"  {m}", file=sys.stderr, flush=True) if args.verbose else None)
    except (ExperimentError, RegistryError, ValueError, KeyError, OSError, cmpn.ComparisonError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(card, indent=1, default=str, allow_nan=False))
        return 0
    print(f"{card['card_id']} ({card['family_label']}, study {card['study_id']}, frozen {card['frozen_at']}): {card['hypothesis']}")
    print(f"  one change: {card['one_change_description']}; primary metric {card['primary_metric']}; {card['state']}")
    for a in (card["results"] or {}).get("candidates", []):
        d = (a.get("metrics") or {}).get(card["primary_metric"], {}).get("difference")
        print(f"  {a['name']}: {a.get('verdict', a['status'])} - {a.get('reason', '')}" + ("" if not d or d.get("value") is None else f"  [difference {_fmt(d['value'])} {_ci(d)}]"))
    for f in card["negative_findings"]:
        print(f"  negative finding: {f.get('candidate', '-')}: {f['what']} ({f.get('detail')})")
    print(f"  gate: {card['gate_decision']['status']} - {card['gate_decision']['reason']}")
    return 0


def register(sub) -> None:
    """Add the four commands to a ``quant`` sub-parser collection."""
    def common(q, data: bool = True):
        q.add_argument("--registry", default="artifacts/ml", help="where experiments, features and the registry live (default artifacts/ml)")
        if data:
            q.add_argument("--universe", default="etf", help="etf, crypto or futures (the futures are simulated)")
            q.add_argument("--prices", help="your own wide CSV of prices instead of a universe")
            q.add_argument("--groups", nargs="*", choices=list(GROUPS), help="feature groups (default price momentum mean_reversion volatility volume relative macro)")
            q.add_argument("--target", help="e.g. fwd_ret_5, excess_ret_20, direction_5, rank_5, fwd_vol_20")
            q.add_argument("--horizon", type=int); q.add_argument("--lag", type=int, help="days between decision and the start of the target (default 1, the engine's)")
            q.add_argument("--normalize", choices=["none", "cross_sectional_z", "rank"]); q.add_argument("--start"); q.add_argument("--end")
            q.add_argument("--regime-detector", dest="regime_detector", choices=["vol_state", "hmm", "gmm"]); q.add_argument("--winsor", type=float); q.add_argument("--min-history", dest="min_history", type=int)

    def signal_options(q):
        q.add_argument("--signal", help="top_n_long_short, long_only_top_quantile, proportional, vol_scaled, rank_normalised, threshold, regime_filtered, confidence_weighted")
        q.add_argument("--n", type=int); q.add_argument("--quantile", type=float); q.add_argument("--holding-period", dest="holding_period", type=int); q.add_argument("--max-weight", dest="max_weight", type=float)
        q.add_argument("--rebalance", choices=["daily", "weekly", "monthly"]); q.add_argument("--cost-bps", dest="cost_bps", type=float); q.add_argument("--slippage-bps", dest="slippage_bps", type=float)
        q.add_argument("--target-vol", dest="target_vol", type=float); q.add_argument("--extra-lag", dest="extra_lag", type=int)

    p = sub.add_parser("ml-build-dataset", help="build and save a leak-free panel of features and targets")
    common(p); p.add_argument("--name"); p.add_argument("--extra-targets", nargs="*", dest="extra_targets", help="more targets to build alongside the main one")
    p.set_defaults(func=cmd_build_dataset)
    p = sub.add_parser("ml-train", help="train walk-forward and run the whole experiment: positions, backtest, segments, diagnostics")
    common(p); signal_options(p)
    p.add_argument("--config", help="a YAML file of overrides (sections dataset, split, model, signal, backtest)"); p.add_argument("--model", help="ridge, lasso, elasticnet, random_forest, hist_gb, xgboost, lightgbm, kalman, ...")
    p.add_argument("--param", action="append", help="model parameter, e.g. --param alpha=10"); p.add_argument("--refit-every", dest="refit_every", type=int); p.add_argument("--min-train-days", dest="min_train_days", type=int)
    p.add_argument("--window", choices=["expanding", "rolling"]); p.add_argument("--embargo", type=int); p.add_argument("--name"); p.add_argument("--seed", type=int)
    p.add_argument("--robustness", action="store_true", help="also run the stress tests"); p.add_argument("--no-explain", action="store_true", help="skip permutation importance"); p.add_argument("--no-save", action="store_true")
    p.add_argument("--verbose", action="store_true"); p.add_argument("--development", action="store_true", help="stop at the end of the validation segment: the test and forward outcomes are not read")
    p.add_argument("--study", help="the study this try belongs to (default: the universe)"); p.set_defaults(func=cmd_train)
    p = sub.add_parser("ml-backtest", help="re-run a saved experiment's predictions under different position rules, rebalancing or costs (no refit)")
    p.add_argument("experiment"); common(p, data=False); signal_options(p); p.add_argument("--no-save", action="store_true"); p.set_defaults(func=cmd_backtest)
    p = sub.add_parser("ml-realistic", help="re-run a saved experiment's predictions through orders, fills and the ledger (the realistic execution mode), as a child run")
    p.add_argument("experiment"); common(p, data=False); p.add_argument("--prices", help="a price CSV to take the volume from (default: the universe the experiment used)"); p.add_argument("--universe", default="etf")
    p.add_argument("--capital", type=float); p.add_argument("--participation", type=float, help="the share of the prior average volume one session may trade (0 = no limit, which is then stated)")
    p.add_argument("--commission-bps", type=float, dest="commission_bps"); p.add_argument("--half-spread-bps", type=float, dest="half_spread_bps"); p.add_argument("--slippage-bps", type=float, dest="slippage_bps")
    p.add_argument("--impact-coefficient", type=float, dest="impact_coefficient"); p.add_argument("--adv-window", type=int, dest="adv_window"); p.add_argument("--sizing", choices=("decision_close", "fill_time"))
    p.add_argument("--expiry-sessions", type=int, dest="expiry_sessions"); p.add_argument("--cash-rate", type=float, dest="cash_rate"); p.add_argument("--financing-spread", type=float, dest="financing_spread")
    p.add_argument("--shorts", choices=("unconstrained", "refuse"), help="refuse: strict validation, a new short without a borrow record is refused"); p.add_argument("--min-adv-dollars", type=float, dest="min_adv_dollars")
    p.add_argument("--min-listing-days", type=int, dest="min_listing_days")
    p.add_argument("--name"); p.add_argument("--no-save", action="store_true", dest="no_save"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_realistic)
    p = sub.add_parser("ml-report", help="the leaderboard, or one experiment in full")
    p.add_argument("experiment", nargs="?"); common(p, data=False); p.add_argument("--list", action="store_true"); p.add_argument("--sort", default="validation.sharpe"); p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true"); p.add_argument("--html", nargs="?", const="-", help="print the path of the HTML report, or copy it to a file"); p.set_defaults(func=cmd_report)
    p = sub.add_parser("ml-verify", help="the persisted verification evidence of the model-correctness requirements; --run runs the tests behind them and records the result")
    common(p, data=False); p.add_argument("--requirement", action="append", help="a requirement id such as P0-01 (repeatable)"); p.add_argument("--run", action="store_true"); p.add_argument("--json", action="store_true")
    p.add_argument("--require-pass", action="store_true", help="exit 1 unless every requirement shown is `pass` and current (what CI asks)")
    p.set_defaults(func=cmd_verify)

    # Release 3: research controls
    p = sub.add_parser("ml-lineage", help="a study's record: every try, its state and parent, the raw look counts, plans and holdout accesses")
    common(p, data=False); p.add_argument("experiment", nargs="?"); p.add_argument("--study"); p.add_argument("--limit", type=int, default=30); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_lineage)
    p = sub.add_parser("ml-plan", help="create (and reserve), freeze or show an evaluation plan for the test and forward segments")
    common(p, data=False); p.add_argument("action", choices=["create", "freeze", "show"]); p.add_argument("plan", nargs="?", help="the plan id (freeze, show)")
    p.add_argument("--study", default="u-etf"); p.add_argument("--candidate", action="append", help="a YAML file of experiment overrides (repeatable)"); p.add_argument("--baseline", help="a YAML file for the baseline")
    p.add_argument("--selection-adjustment", dest="selection_adjustment", default="none", help="none, bonferroni, holm, reality_check, spa or dsr_all_trials (required for several candidates)")
    p.add_argument("--uncertainty-method", dest="uncertainty_method", default="not_specified"); p.add_argument("--freeze", action="store_true"); p.add_argument("--prices", help="your own wide CSV of prices")
    p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_plan)
    p = sub.add_parser("ml-evaluate", help="the one evaluation event of a frozen plan (the same --key returns the same result)")
    common(p, data=False); p.add_argument("plan"); p.add_argument("--key", required=True, help="idempotency key"); p.add_argument("--allow-reuse", dest="allow_reuse", action="store_true", help="a deliberate re-evaluation, recorded as a re-use")
    p.add_argument("--resume", action="store_true"); p.add_argument("--prices", help="your own wide CSV of prices (the plan's data)"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("ml-holdout", help="evaluation plans, their reservations (untouched or not) and the holdout access ledger")
    common(p, data=False); p.add_argument("--study"); p.add_argument("--plan"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_holdout)
    p = sub.add_parser("ml-null-controls", help="freeze a design and run the whole search pipeline on placebo data; report the false-promotion rate with its interval")
    common(p, data=False); p.add_argument("--list", action="store_true"); p.add_argument("--study", default="u-etf"); p.add_argument("--config", help="a YAML file of experiment overrides the pipeline starts from")
    p.add_argument("--kinds", nargs="+", default=["ridge", "lasso"]); p.add_argument("--n-random", dest="n_random", type=int, default=4); p.add_argument("--n-bayes", dest="n_bayes", type=int, default=0)
    p.add_argument("--replications", type=int, default=20); p.add_argument("--min-replications", dest="min_replications", type=int, default=100); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-shift", dest="min_shift", type=int, default=200, help="the smallest placebo shift, in days (beyond every feature window)")
    p.add_argument("--tolerance", type=float, default=0.10); p.add_argument("--precision", type=float, default=0.10); p.add_argument("--prices", help="your own wide CSV of prices"); p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_null_controls)
    p = sub.add_parser("ml-selection-tests", help="apply the selection-bias procedures (DSR, PBO, Reality Check, SPA, FDR) to a study's trials where they apply")
    common(p, data=False); p.add_argument("study"); p.add_argument("--n-boot", dest="n_boot", type=int, default=500); p.add_argument("--seed", type=int, default=0); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_selection_tests)
    p = sub.add_parser("ml-policy", help="show the validation policy in force, or adopt one (--adopt-example for the illustrative example, --file for your own); a named person and a reason are required")
    common(p, data=False); p.add_argument("--adopt-example", dest="adopt_example", action="store_true"); p.add_argument("--file", help="a JSON file with name, version, criteria and an optional description")
    p.add_argument("--actor"); p.add_argument("--note"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_policy)
    p = sub.add_parser("ml-diagnostics", help="compute and store the statistical diagnostics of an experiment under the adopted policy (HAC and bootstrap Sharpe intervals, matched benchmarks, regimes, calibration)")
    common(p, data=False); p.add_argument("experiment"); p.add_argument("--n-boot", dest="n_boot", type=int, default=500); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--neighborhood", help="JSON grid of settings to vary, e.g. '{\"signal.n\": [2, 3, 4]}' (re-runs the saved predictions on the validation segment); a grid over the model's own settings, e.g. '{\"model.params.alpha\": [1, 10, 100]}', refits the model at each point (up to 12 points)"); p.add_argument("--delays", help="extra sessions of delay, e.g. 0,1,2,5")
    p.add_argument("--capitals", help="capital levels for the capacity analysis, e.g. 1e5,1e6,1e7"); p.add_argument("--prices", help="your own wide CSV of prices (for the analyses that re-run)"); p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_diagnostics)
    p = sub.add_parser("ml-multiseed", help="train an experiment's configuration over predeclared seeds (development mode); report the spread across seeds beside each seed's time-series uncertainty")
    common(p, data=False); p.add_argument("experiment"); p.add_argument("--seeds", required=True, help="three or more distinct seeds declared in advance, e.g. 1,2,3,4,5"); p.add_argument("--n-boot", dest="n_boot", type=int, default=300)
    p.add_argument("--prices", help="your own wide CSV of prices"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_multiseed)
    p = sub.add_parser("ml-governance", help="strategy versions (signal, execution and risk settings), risk limits, halt/resume/rollback, recovery procedure and the separate deployment approval")
    common(p, data=False); p.add_argument("experiment"); p.add_argument("--actor"); p.add_argument("--register", action="store_true"); p.add_argument("--limits", help='JSON risk limits, e.g. \'{"max_gross": 1.5, "max_drawdown": 0.2}\'')
    p.add_argument("--version"); p.add_argument("--halt", metavar="REASON"); p.add_argument("--resume", metavar="NOTE"); p.add_argument("--rollback-to", dest="rollback_to", metavar="VERSION"); p.add_argument("--reason")
    p.add_argument("--recovery", metavar="PROCEDURE"); p.add_argument("--approve", metavar="NOTE"); p.add_argument("--stage", choices=["staging", "production"], default="production"); p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_governance)
    p = sub.add_parser("ml-forward", help="one forward-test step: commit today's signal to the immutable ledger, settle matured commitments, check monitoring")
    common(p, data=False); p.add_argument("experiment"); p.add_argument("--capital", type=float, default=100_000.0); p.add_argument("--as-of", dest="as_of"); p.add_argument("--no-record", dest="no_record", action="store_true")
    p.add_argument("--prices", help="your own wide CSV of prices"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_forward)
    p = sub.add_parser("ml-alpha", help="alpha diagnostics of a saved experiment: rank IC with a dependence-aware interval, decay, coverage, stability, redundancy, and the value it adds to a baseline experiment")
    common(p, data=False); p.add_argument("experiment"); p.add_argument("--segment", default="validation", help="validation (default); test and forward only for an experiment a frozen plan evaluated")
    p.add_argument("--baseline", help="another saved experiment treated as the existing signal"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_alpha)
    p = sub.add_parser("ml-portfolio", help="portfolio diagnostics of a saved experiment: the book it held, covariance estimators on realised risk, risk contributions, allocation sensitivity")
    common(p, data=False); p.add_argument("experiment"); p.add_argument("--segment", default="validation"); p.add_argument("--horizon", type=int, default=21, help="days over which realised risk is measured")
    p.add_argument("--dates", type=int, default=24, help="decision dates sampled"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_portfolio)
    p = sub.add_parser("ml-delay", help="delay report of a saved experiment: rank IC, expected edge, net result and capacity when its book is executed later, at delays you declare")
    common(p, data=False); p.add_argument("experiment"); p.add_argument("--segment", default="validation"); p.add_argument("--delays", default="0,1,2,5", help="sessions of delay, comma separated; 0 must be among them")
    p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_delay)
    p = sub.add_parser("ml-improve", help="freeze and run a controlled comparison card (target, feature, horizon, portfolio, rebalancing, risk_control, ensemble or execution) from a JSON spec, or show a stored card with --card")
    common(p, data=False); p.add_argument("--spec", help="JSON file: family, hypothesis, one_change, baseline, candidates (changes laid over the baseline), and optional study, primary_metric, parsimony, min_effect, n_boot")
    p.add_argument("--card", help="show this stored card instead"); p.add_argument("--prices", help="your own wide CSV of prices instead of the universe in the baseline"); p.add_argument("--universe", default="etf")
    p.add_argument("--verbose", action="store_true"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_improve)
    p = sub.add_parser("ml-promotion", help="the three independent gates of an experiment, from the persisted evidence")
    common(p, data=False); p.add_argument("experiment"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_promotion)
    p = sub.add_parser("ml-promote", help="move a registered model to a stage; the registry checks the gates, production also needs a recorded human approval")
    common(p, data=False); p.add_argument("model"); p.add_argument("stage", choices=["none", "candidate", "staging", "production", "archived"]); p.add_argument("--note")
    p.add_argument("--approval-actor", dest="approval_actor"); p.add_argument("--approval-note", dest="approval_note"); p.set_defaults(func=cmd_promote)
    from . import advanced_cli                                                  # the optional advanced modules of Release 8: one command, nothing imported until it runs

    advanced_cli.register(sub)
