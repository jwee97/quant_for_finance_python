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
    from . import selection as sel

    registry = Registry(args.registry)
    out = sel.evaluate_study(registry, args.study, n_boot=args.n_boot, seed=args.seed)
    lines = [f"study {args.study}: {out['n_tries']} tries recorded, {out['n_candidates']} with return streams", f"  {out['statement']}"] + [f"  {''.join(e['method']):<14} {e['status']:<15} {e['reason'][:100] or ('met' if e['statistic'].get('passes') else 'not met')}" for e in out["evidence"]]
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
    p = sub.add_parser("ml-promotion", help="the three independent gates of an experiment, from the persisted evidence")
    common(p, data=False); p.add_argument("experiment"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_promotion)
    p = sub.add_parser("ml-promote", help="move a registered model to a stage; the registry checks the gates, production also needs a recorded human approval")
    common(p, data=False); p.add_argument("model"); p.add_argument("stage", choices=["none", "candidate", "staging", "production", "archived"]); p.add_argument("--note")
    p.add_argument("--approval-actor", dest="approval_actor"); p.add_argument("--approval-note", dest="approval_note"); p.set_defaults(func=cmd_promote)
