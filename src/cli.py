"""``quant``: the command-line front door to the research framework.

    quant list models                     what can be run
    quant backtest --model momentum       a one-line strategy through the whole pipeline
    quant run spec.yaml --tearsheet       a full specification, with a tear sheet
    quant leaderboard / compare / tearsheet RUN_ID
    quant data check --prices my.csv      is your data fit to use?
    quant dashboard                       a self-contained HTML explorer of everything recorded
    quant explain "deflated sharpe"       a plain-language explanation of a term, technique or experiment
    quant demo                            a five-minute tour on cached data
    quant docs build                      regenerate the strategy cards, chapter map and findings digest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .framework import ALLOCATORS, DETECTORS, MODELS, PipelineSpec, bundle_from_prices, load_default_bundle, load_library, load_prices_csv
from .framework.allocation import BOOKS
from .framework.experiments import ExperimentManager
from .utils.config import load_config


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _bundle(config, prices: str | None):
    if prices:
        return bundle_from_prices(load_prices_csv(prices), name=Path(prices).stem)
    return load_default_bundle(config)


def _print_frame(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> None:
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60, "display.float_format", floatfmt.format):
        print(frame.to_string(index=False) if frame.index.name is None and not isinstance(frame.index, pd.MultiIndex) and list(frame.index) == list(range(len(frame))) else frame.to_string())


def cmd_list(args, config) -> int:
    registries = {"models": MODELS, "detectors": DETECTORS, "allocators": ALLOCATORS}
    if args.what == "books":
        print("\n".join(BOOKS))
        return 0
    registry = registries[args.what]
    rows = [{"name": e.name, "family": e.family, "description": e.description[:110]} for e in registry.entries()]
    _print_frame(pd.DataFrame(rows))
    print(f"\n{len(rows)} {args.what}")
    return 0


def _spec_from_args(args) -> dict:
    params = {k: _coerce(v) for k, v in (p.split("=", 1) for p in (args.param or []))}
    spec = {"name": args.name or f"{args.model}", "models": [{"name": args.model, "params": params}]}
    if args.regime:
        spec["regime"] = {"detector": args.regime}
    if args.combine:
        spec["combination"] = {"rule": args.combine}
    if args.allocator:
        spec["allocation"] = {"allocator": args.allocator}
    if args.group:
        spec["evaluation"] = {"group": args.group}
    return spec


def _run(spec, args, config) -> int:
    manager = ExperimentManager(config)
    bundle = _bundle(config, getattr(args, "prices", None))
    run_id, result = manager.run(spec, bundle, tearsheet=args.tearsheet, force=args.force, validate=not args.no_validate)
    if result is None:
        print(f"run {run_id} already recorded for this spec, data and configuration (use --force to rerun)")
        _print_frame(manager.compare([run_id]).reset_index())
        return 0
    m = result.metrics
    print(f"\nrun {run_id}: {m['start']} to {m['end']}")
    print(f"  net Sharpe {m['sharpe']:+.2f}  (gross {m['gross_sharpe']:+.2f})  CAGR {m['cagr']:.1%}  vol {m['ann_vol']:.1%}  max drawdown {m['max_drawdown']:.1%}  turnover {m['ann_turnover']:.1f}x")
    print(f"  deflated Sharpe probability {result.validation['deflated_sharpe_probability']:.2f} ({result.validation['n_trials']} trial(s) in group)")
    from .framework.tearsheet import red_flags
    flags = [f for f in red_flags(result) if f["flag"]]
    print(f"  {len(flags)} red flag(s)" + ("" if not flags else ":"))
    for f in flags:
        print(f"    - {f['rule']}: {f['observed']}")
    if args.tearsheet:
        print(f"  tear sheet: reports/tearsheets/{run_id}/tearsheet.md")
    return 0


def cmd_backtest(args, config) -> int:
    return _run(_spec_from_args(args), args, config)


def cmd_run(args, config) -> int:
    spec = PipelineSpec.from_yaml(args.spec)
    if args.name:
        spec.name = args.name
    if args.group:
        spec.evaluation = {**spec.evaluation, "group": args.group}
    return _run(spec, args, config)


def cmd_leaderboard(args, config) -> int:
    _print_frame(ExperimentManager(config).leaderboard(args.group, args.top, args.by))
    return 0


def cmd_compare(args, config) -> int:
    _print_frame(ExperimentManager(config).compare(args.run_ids).reset_index())
    return 0


def cmd_tearsheet(args, config) -> int:
    manager = ExperimentManager(config)
    spec = manager.spec_of(args.run_id)
    from .framework import Pipeline
    from .framework.tearsheet import make_tearsheet
    result = Pipeline(spec, config, _bundle(config, args.prices)).run()
    path = make_tearsheet(result, config.root / "reports" / "tearsheets" / args.run_id, args.run_id)
    print(path)
    return 0


def cmd_data(args, config) -> int:
    from .framework.dataquality import check_prices, format_report
    prices = load_prices_csv(args.prices) if args.prices else load_default_bundle(config, with_macro=False).prices
    report = check_prices(prices)
    print(format_report(report))
    return 0 if not report["errors"] else 1


def cmd_dashboard(args, config) -> int:
    from .framework.dashboard import build_dashboard
    path = build_dashboard(config, Path(args.out))
    print(path)
    return 0


def cmd_explain(args, config) -> int:
    from .assistant.explain import explain
    print(explain(" ".join(args.term), config, full=args.full))
    return 0


def cmd_demo(args, config) -> int:
    from .framework.demo import run_demo
    return run_demo(config)


def cmd_docs(args, config) -> int:
    from .framework.docs import build_docs
    for path in build_docs(config):
        print(path.relative_to(config.root))
    return 0


def cmd_sql(args, config) -> int:
    from .research_db.__main__ import main as sql_main
    return sql_main([args.sql])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list", help="list models, detectors, allocators or books")
    p.add_argument("what", choices=["models", "detectors", "allocators", "books"])
    p.set_defaults(func=cmd_list)

    def run_options(q):
        q.add_argument("--prices", help="your own wide CSV of prices (date column plus one column per asset)")
        q.add_argument("--name"); q.add_argument("--group", help="experiments in a group count as trials in the deflated Sharpe ratio")
        q.add_argument("--tearsheet", action="store_true"); q.add_argument("--force", action="store_true"); q.add_argument("--no-validate", action="store_true")

    p = sub.add_parser("backtest", help="one model through the whole pipeline")
    p.add_argument("--model", required=True); p.add_argument("--param", action="append", help="key=value, repeatable")
    p.add_argument("--regime"); p.add_argument("--combine"); p.add_argument("--allocator")
    run_options(p); p.set_defaults(func=cmd_backtest)
    p = sub.add_parser("run", help="run a YAML specification")
    p.add_argument("spec"); run_options(p); p.set_defaults(func=cmd_run)
    p = sub.add_parser("leaderboard", help="the recorded runs")
    p.add_argument("--group"); p.add_argument("--top", type=int, default=20); p.add_argument("--by", default="sharpe"); p.set_defaults(func=cmd_leaderboard)
    p = sub.add_parser("compare", help="compare recorded runs"); p.add_argument("run_ids", nargs="+"); p.set_defaults(func=cmd_compare)
    p = sub.add_parser("tearsheet", help="regenerate the tear sheet of a recorded run")
    p.add_argument("run_id"); p.add_argument("--prices"); p.set_defaults(func=cmd_tearsheet)
    p = sub.add_parser("data", help="check a price file"); p.add_argument("action", choices=["check"]); p.add_argument("--prices"); p.set_defaults(func=cmd_data)
    p = sub.add_parser("dashboard", help="write a self-contained HTML dashboard"); p.add_argument("--out", default="reports/dashboard.html"); p.set_defaults(func=cmd_dashboard)
    p = sub.add_parser("explain", help="explain a term, a technique or an experiment"); p.add_argument("term", nargs="+"); p.add_argument("--full", action="store_true", help="print the whole guide"); p.set_defaults(func=cmd_explain)
    p = sub.add_parser("demo", help="a five-minute tour"); p.set_defaults(func=cmd_demo)
    p = sub.add_parser("docs", help="regenerate strategy cards, the chapter map and the findings digest"); p.add_argument("action", choices=["build"]); p.set_defaults(func=cmd_docs)
    p = sub.add_parser("sql", help="a read-only query on the research database"); p.add_argument("sql"); p.set_defaults(func=cmd_sql)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    load_library()
    return int(args.func(args, config) or 0)


if __name__ == "__main__":
    sys.exit(main())
