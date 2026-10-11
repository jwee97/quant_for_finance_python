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
    try:
        exp = run_experiment(over, bundle=bundle, config=config, registry=registry, save=not args.no_save, info=info, progress=lambda m: print(f"  {m}", file=sys.stderr, flush=True) if args.verbose else None,
                             explain=not args.no_explain)
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
    p.add_argument("--verbose", action="store_true"); p.set_defaults(func=cmd_train)
    p = sub.add_parser("ml-backtest", help="re-run a saved experiment's predictions under different position rules, rebalancing or costs (no refit)")
    p.add_argument("experiment"); common(p, data=False); signal_options(p); p.add_argument("--no-save", action="store_true"); p.set_defaults(func=cmd_backtest)
    p = sub.add_parser("ml-report", help="the leaderboard, or one experiment in full")
    p.add_argument("experiment", nargs="?"); common(p, data=False); p.add_argument("--list", action="store_true"); p.add_argument("--sort", default="validation.sharpe"); p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true"); p.add_argument("--html", nargs="?", const="-", help="print the path of the HTML report, or copy it to a file"); p.set_defaults(func=cmd_report)
