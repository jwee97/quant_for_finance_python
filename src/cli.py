"""``quant``: the command-line front door to the research framework.

    quant list models                     what can be run
    quant backtest --model momentum       a one-line strategy through the whole pipeline
    quant run spec.yaml --tearsheet       a full specification, with a tear sheet
    quant leaderboard / compare / tearsheet RUN_ID
    quant data check --prices my.csv      is your data fit to use?
    quant dashboard                       a self-contained HTML explorer of everything recorded
    quant explain "deflated sharpe"       a plain-language explanation of a term, technique or experiment
    quant demo                            a five-minute tour on cached data
    quant sweep --model dual_momentum --grid models.0.params.lookback=63,126,252   a parallel grid, each variant a counted trial
    quant sql --db runs "SELECT ..."      query your own runs
    quant capacity --model momentum       net Sharpe by assets under management
    quant docs build                      regenerate the strategy cards, chapter map and findings digest
    quant tune --model tsmom --space models.0.params.lookback=int:21:504 --trials 24   a hyperparameter search that counts its trials and deflates the winner
    quant benchmark --models tsmom momentum   score time and peak memory of models on the default bundle
    quant registry list                   versioned models: stages, lineage, promotion
"""

from __future__ import annotations

import argparse
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
        spec["allocation"] = {"allocator": args.allocator, "params": {k: _coerce(v) for k, v in (p.split("=", 1) for p in (args.alloc_param or []))}}
    elif getattr(args, "alloc_param", None):
        raise SystemExit("--alloc-param needs --allocator")
    if getattr(args, "aum", None):
        spec["execution"] = {"aum": float(args.aum)}
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


def cmd_capacity(args, config) -> int:
    """Net Sharpe against assets under management for a spec (a YAML file) or a one-model strategy, and the AUM at which it halves."""
    from .framework import Pipeline
    spec = PipelineSpec.from_yaml(args.spec) if args.spec else PipelineSpec.from_dict(_spec_from_args(args))
    spec.execution = {**spec.execution, "capacity": [float(g) for g in args.grid] if args.grid else True}
    result = Pipeline(spec, config, _bundle(config, args.prices)).run(validate=False)
    table = result.tables["capacity"].copy()
    table.index = [f"${a:,.0f}" for a in table.index]
    _print_frame(table.reset_index().rename(columns={"index": "AUM"}))
    cap = result.metrics.get("capacity_usd")
    print("\ncapacity (AUM at half the linear-cost Sharpe): " + (f"${cap:,.0f}" if cap == cap else result.metrics.get("capacity_note") or "n/a"))
    return 0


def _param_from_text(text: str):
    """``int:21:504``, ``float:0:1``, ``log:1e-4:1`` or ``cat:a,b,c`` -> a search distribution."""
    from .ops.hpo import Categorical, Float, Int, LogFloat

    kind, _, rest = text.partition(":")
    if kind == "cat":
        return Categorical(*[_coerce(v) for v in rest.split(",")])
    low, _, high = rest.partition(":")
    factory = {"int": Int, "float": Float, "log": LogFloat}.get(kind)
    if factory is None or not low or not high:
        raise SystemExit(f"search space '{text}': use int:LOW:HIGH, float:LOW:HIGH, log:LOW:HIGH or cat:A,B,C")
    cast = int if kind == "int" else float
    return factory(cast(low), cast(high))


def cmd_tune(args, config) -> int:
    """A hyperparameter search over a one-model strategy or a YAML spec. Every trial is a full pipeline run and a counted look at the data; the report deflates the winner."""
    from .ops.hpo import tune_pipeline

    spec = PipelineSpec.from_yaml(args.spec).to_dict() if args.spec else _spec_from_args(args)
    space = {}
    for item in args.space:
        path, _, text = item.partition("=")
        space[path] = _param_from_text(text)
    study = tune_pipeline(spec, _bundle(config, args.prices), config, space, n_trials=args.trials, method=args.method, n_blocks=args.blocks, seed=args.seed)
    frame = study.to_frame().sort_values("value", ascending=False)
    _print_frame(frame.head(args.top).reset_index(drop=True))
    best = study.best
    print(f"\nbest of {study.n_trials} trials: {best.params} (value {best.value:.3f})")
    report = study.selection_report()
    for key in ("best_annual_sharpe", "deflated_sharpe_probability", "pbo", "mean_oos_sharpe_of_is_winner", "romano_wolf_p_best", "survivors_at_5pct"):
        if key in report:
            print(f"  {key}: {report[key]:.3f}")
    return 0


def cmd_benchmark(args, config) -> int:
    from .ops.profiling import benchmark_models

    table = benchmark_models(_bundle(config, args.prices), names=args.models, repeats=args.repeats)
    _print_frame(table.reset_index() if table.index.name else table)
    if args.csv:
        table.to_csv(args.csv)
        print(f"written to {args.csv}")
    return 0


def cmd_registry(args, config) -> int:
    from .ops.store import ModelRegistry

    registry = ModelRegistry(config.root / "data" / "registry" / "models.db")
    if args.action == "list":
        table = registry.versions(args.name)
        print("no registered models" if table.empty else table.to_string(index=False))
    elif args.action == "register-run":
        if not args.run_id:
            raise SystemExit("register-run needs --run-id")
        version = registry.register_run(ExperimentManager(config), args.run_id, name=args.name)
        print(f"registered version {version}")
    elif args.action == "promote":
        if not (args.name and args.version and args.stage):
            raise SystemExit("promote needs --name, --version and --stage")
        registry.promote(args.name, args.version, args.stage)
        print(f"{args.name} v{args.version} -> {args.stage}")
    elif args.action == "lineage":
        if not args.name:
            raise SystemExit("lineage needs --name")
        print(pd.DataFrame(registry.lineage(args.name, args.version)).to_string(index=False))
    return 0


def cmd_docs(args, config) -> int:
    from .framework.docs import build_docs
    for path in build_docs(config):
        print(path.relative_to(config.root))
    return 0


def cmd_sql(args, config) -> int:
    from .research_db.__main__ import main as sql_main
    return sql_main([args.sql, "--db", args.db])


def cmd_serve(args, config) -> int:
    from .webapp.server import serve
    return serve(config, args.host, args.port, not args.no_browser)


def cmd_new_strategy(args, config) -> int:
    from .strategies.user import new_strategy
    try:
        path = new_strategy(config.root, args.name)
    except (ValueError, FileExistsError) as error:
        print(error)
        return 1
    print(f"wrote {path}\nedit `score`, then:  quant backtest --model {args.name} --tearsheet   (or pick it in `quant serve`)")
    return 0


def cmd_ask(args, config) -> int:
    """Ask the research database a recognised question (templates) or, with --llm, let a model write the SQL (read-only, validated like any other statement)."""
    from .assistant.sqlguard import Refused, ResearchAssistant, SQLBackend
    db = config.root / "data" / "processed" / "research.db"
    if not db.exists():
        print(f"{db} does not exist yet: build it with `python -m experiments.stage29_research_db`")
        return 1
    backend = SQLBackend(args.llm) if args.llm else None
    try:
        answer = ResearchAssistant(db, backend).ask(" ".join(args.question))
    except Refused as refusal:
        print(f"I will not guess. {refusal}")
        return 1
    except ImportError as error:
        print(f"--llm needs the optional `anthropic` package and an API key ({error})")
        return 1
    print(f"-- {answer.backend}\n-- {answer.sql}  {answer.params if answer.params else ''}")
    _print_frame(answer.rows)
    return 0


def cmd_causal(args, config) -> int:
    """The causal estimators against simulated worlds with a known effect."""
    from .causal.check import recovery_table
    _print_frame(recovery_table(args.n, args.seed), "{:.3f}")
    return 0


def cmd_sweep(args, config) -> int:
    """Run a grid of variations of one specification in parallel; every variant counts as a trial for the deflated Sharpe ratio."""
    from .framework.experiments import expand_grid
    base = PipelineSpec.from_yaml(args.spec).to_dict() if args.spec else _spec_from_args(args)
    grid = {}
    for item in args.grid:
        key, _, values = item.partition("=")
        grid[key] = [_coerce(v) for v in values.split(",")]
    variants = expand_grid(base, grid)
    group = args.group or f"{base.get('name', 'sweep')}_sweep"
    for v in variants:
        v["evaluation"] = {**(v.get("evaluation") or {}), "group": group}
        if args.no_validate:
            v["evaluation"]["causality"] = False
    manager = ExperimentManager(config)
    ids = manager.sweep(variants, _bundle(config, args.prices), validate=not args.no_validate, backend=args.backend, n_jobs=args.jobs, force=args.force)
    board = manager.compare(ids).reset_index()
    print(f"{len(ids)} variants in group '{group}'")
    _print_frame(board[["run_id", "name", "sharpe", "gross_sharpe", "max_drawdown", "ann_turnover", "deflated_sharpe_probability", "n_trials"]])
    return 0


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
    p.add_argument("--alloc-param", action="append", help="allocator parameter key=value, repeatable (e.g. --allocator static --alloc-param book=hrp)")
    p.add_argument("--aum", type=float, help="assets under management in dollars: charges square-root market impact")
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
    p = sub.add_parser("capacity", help="net Sharpe by assets under management (square-root market impact)")
    p.add_argument("--spec"); p.add_argument("--model"); p.add_argument("--param", action="append"); p.add_argument("--regime"); p.add_argument("--combine"); p.add_argument("--allocator")
    p.add_argument("--alloc-param", action="append"); p.add_argument("--name"); p.add_argument("--group"); p.add_argument("--prices"); p.add_argument("--aum", type=float)
    p.add_argument("--grid", type=float, nargs="*", help="AUM values in dollars")
    p.set_defaults(func=cmd_capacity)
    p = sub.add_parser("docs", help="regenerate strategy cards, the chapter map and the findings digest"); p.add_argument("action", choices=["build"]); p.set_defaults(func=cmd_docs)
    p = sub.add_parser("sql", help="a read-only query on the research database (--db runs: your own experiment runs)"); p.add_argument("sql")
    p.add_argument("--db", choices=["research", "runs"], default="research"); p.set_defaults(func=cmd_sql)
    p = sub.add_parser("serve", help="open the dashboard: backtest on your own tickers, compare runs, build formula strategies, read the guides")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8765); p.add_argument("--no-browser", action="store_true"); p.set_defaults(func=cmd_serve)
    p = sub.add_parser("new-strategy", help="write a template for your own strategy into user_strategies/"); p.add_argument("name"); p.set_defaults(func=cmd_new_strategy)
    p = sub.add_parser("ask", help="ask the research database a question in words"); p.add_argument("question", nargs="+"); p.add_argument("--llm", help="model that writes the SQL (needs the anthropic package and an API key)"); p.set_defaults(func=cmd_ask)
    p = sub.add_parser("causal", help="causal estimators against simulated worlds with a known effect"); p.add_argument("--n", type=int, default=3000); p.add_argument("--seed", type=int, default=7); p.set_defaults(func=cmd_causal)
    p = sub.add_parser("sweep", help="a parallel grid of variations of one specification, counted as trials")
    p.add_argument("--spec"); p.add_argument("--model"); p.add_argument("--param", action="append"); p.add_argument("--regime"); p.add_argument("--combine"); p.add_argument("--allocator")
    p.add_argument("--alloc-param", action="append"); p.add_argument("--name"); p.add_argument("--aum", type=float); p.add_argument("--prices"); p.add_argument("--group")
    p.add_argument("--grid", action="append", required=True, help="dotted.path=v1,v2,... repeatable, e.g. models.0.params.lookback=63,126,252")
    p.add_argument("--backend", default="joblib", choices=["serial", "joblib", "dask", "ray"]); p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--force", action="store_true"); p.add_argument("--no-validate", action="store_true"); p.set_defaults(func=cmd_sweep)
    p = sub.add_parser("tune", help="hyperparameter search that counts its trials and deflates the winner")
    p.add_argument("--spec"); p.add_argument("--model"); p.add_argument("--param", action="append"); p.add_argument("--regime"); p.add_argument("--combine"); p.add_argument("--allocator")
    p.add_argument("--alloc-param", action="append"); p.add_argument("--name"); p.add_argument("--aum", type=float); p.add_argument("--prices"); p.add_argument("--group")
    p.add_argument("--space", action="append", required=True, help="dotted.path=int:LOW:HIGH | float:LOW:HIGH | log:LOW:HIGH | cat:A,B,C, repeatable")
    p.add_argument("--trials", type=int, default=20); p.add_argument("--method", default="tpe", choices=["random", "grid", "tpe", "halving"])
    p.add_argument("--blocks", type=int, default=4, help="average the Sharpe ratio over this many contiguous blocks and penalise their dispersion (1: the plain Sharpe)")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--top", type=int, default=8); p.set_defaults(func=cmd_tune)
    p = sub.add_parser("benchmark", help="score time and peak memory of registered models on the default bundle")
    p.add_argument("--models", nargs="*"); p.add_argument("--repeats", type=int, default=1); p.add_argument("--prices"); p.add_argument("--csv"); p.set_defaults(func=cmd_benchmark)
    p = sub.add_parser("registry", help="the model registry: versions, stages, lineage")
    p.add_argument("action", choices=["list", "register-run", "promote", "lineage"]); p.add_argument("--name"); p.add_argument("--version", type=int)
    p.add_argument("--stage", choices=["staging", "production", "archived"]); p.add_argument("--run-id"); p.set_defaults(func=cmd_registry)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    load_library()
    return int(args.func(args, config) or 0)


if __name__ == "__main__":
    sys.exit(main())
