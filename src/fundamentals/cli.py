"""``quant fundamentals``: the Fundamentals tab in the terminal, over the same workbench the page uses (so a number here is the number there).

    quant fundamentals status                  what the database holds, the allowance of each provider and every key slot (never a key)
    quant fundamentals simulate                make the demo database: a simulated world with a few alphas planted in it (add --demo to the commands below to work in it)
    quant fundamentals search nvidia           look a company up in OpenFilings' directory by name, ticker or ISIN (1 credit; the same search again is free)
    quant fundamentals add us:AAPL jp:7203     register companies (free); `plan` says what fetching them would cost, `sync --budget 100` fetches within that many credits
    quant fundamentals prices                  daily prices and exchange rates for the companies, from Yahoo Finance
    quant fundamentals factors [--scan]        the factor library; with --scan every factor studied under one set of conditions
    quant fundamentals evaluate fcf_yield      one factor (or --formula "div(fcf,market_cap)") against what happened next: IC, buckets, long-short, decay, years, countries
    quant fundamentals preflight               before a search: what it could find with this many companies and dates, and the chance a real alpha of each strength gets through
    quant fundamentals discover --n 300        search for alphas: every formula tried is counted, the best tested once on dates they never saw; --resume RUN_ID carries on
    quant fundamentals placebo [--run ID]      the same search on returns shuffled among the companies: what it finds when there is nothing to find, and how much of a real run that makes luck
    quant fundamentals library [--graveyard]   the alphas found so far, or the ideas that failed (nothing is ever deleted)
    quant fundamentals monitor                 is each alpha still working on the data now in the database
    quant fundamentals backtest --alpha ID     an alpha as a book under limits, through the platform's backtest engine, net of costs

The real database is ``data/user/fundamentals.db`` and the demo is ``data/user/fundamentals_demo.db``: separate files, so a simulated company never mixes with a real one. A provider key is never an
argument of this command: it is read from the environment (on Replit, a secret named OPENFILINGS_API_KEY_1 ... _5)."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

from . import report as R
from .discovery import STRICTNESS, Discovery, DiscoverySpec
from .errors import FundamentalsError
from .jsonutil import clean
from .library import STATUSES
from .service import FundamentalsService
from .workbench import Workbench, eval_spec

ACTIONS = ("status", "simulate", "search", "add", "plan", "sync", "prices", "factors", "evaluate", "preflight", "discover", "placebo", "library", "monitor", "backtest")
REAL_ONLY = {"search": "searching OpenFilings", "add": "adding companies", "plan": "planning a sync", "sync": "fetching statements", "prices": "loading prices"}
DEMO_FILE, REAL_FILE = "fundamentals_demo.db", "fundamentals.db"


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _progress():
    """A progress line on stderr at most every few seconds, so a long run is not silent and not noisy."""
    last = [0.0]

    def progress(*args) -> None:
        text = args[-1] if args else ""
        if time.time() - last[0] > 5:
            _say(str(text))
            last[0] = time.time()
    return progress


class _Stop:
    """The first Ctrl-C asks a running job to stop after its current step (what is done is kept); a second one ends the command at once."""

    def __init__(self):
        self.flag = threading.Event()
        self._old = None

    def __enter__(self):
        def handler(signum, frame):
            if self.flag.is_set():
                raise KeyboardInterrupt
            self.flag.set()
            _say("stopping after the step it is on (what is done is kept); press Ctrl-C again to end at once")
        try:
            self._old = signal.signal(signal.SIGINT, handler)
        except ValueError:                                                                   # not the main thread: nothing to catch
            self._old = None
        return self.flag.is_set

    def __exit__(self, *exc):
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)
        return False


def _path(config, demo: bool) -> Path:
    return Path(config.root) / "data" / "user" / (DEMO_FILE if demo else REAL_FILE)


def _bench(config, demo: bool, action: str) -> Workbench:
    path = _path(config, demo)
    if demo and action != "simulate" and not path.exists():
        raise FundamentalsError("the demo has not been made yet: run `quant fundamentals simulate`")
    return Workbench(service=FundamentalsService(config.root, db_path=path))


def _universe(args) -> dict:
    u: dict = {}
    if args.countries:
        u["countries"] = [c.upper() for c in args.countries]
    if args.sectors:
        u["sectors"] = list(args.sectors)
    if args.min_cap is not None:
        u["min_market_cap_usd"] = args.min_cap
    if args.min_volume is not None:
        u["min_dollar_volume_usd"] = args.min_volume
    if args.developed:
        u["developed_only"] = True
    u["include_inactive"] = not args.no_delisted
    return u


def _context(wb: Workbench, args):
    pit = {"info_lag_days": args.info_lag} if args.info_lag is not None else None
    return wb.context(_universe(args), args.freq, args.lag, args.start, args.end, pit)


def _ready(wb: Workbench) -> None:
    r = wb.readiness()
    if not r["ready"]:
        raise FundamentalsError(r["problems"][0])


def _steps(args) -> list[dict]:
    if args.scale == "raw":
        return []
    steps = []
    if args.winsor > 0:
        steps.append({"op": "winsorize", "pct": args.winsor})
    if args.neutralize or args.control:
        steps.append({"op": "neutralize", "by": list(args.neutralize or []), "continuous": list(args.control or [])})
    steps.append({"op": {"rank": "rank", "percentile": "percentile"}.get(args.scale, "zscore")})
    return steps


def _eval_raw(args, factor: str | None) -> dict:
    if args.direction not in ("theory", "auto", "1", "-1"):
        raise FundamentalsError("direction is theory (the factor's own), auto (a choice made on the data: it counts as a second trial), 1 or -1")
    raw = {"horizon": args.horizon, "direction": args.direction if args.direction in ("theory", "auto") else int(args.direction), "quantiles": args.quantiles, "weighting": args.weighting,
           "cost_bps": args.cost_bps, "borrow_bps": args.borrow_bps, "steps": _steps(args)}
    if factor:
        raw["factor"] = factor
    if args.premium or args.premium_controls:
        raw["premium"] = True
        if args.premium_controls:
            raw["controls"] = list(args.premium_controls)
    return raw


def _out(args, text: str, data=None) -> int:
    print(text)
    if args.json and data is not None:
        Path(args.json).write_text(json.dumps(clean(data), indent=1, allow_nan=False), encoding="utf-8")
        print(f"\nwritten to {args.json}")
    return 0


# ------------------------------------------------------------------------------------------------------------ the actions
def _status(wb, args) -> int:
    s = wb.status()
    return _out(args, R.format_status(s, "demo" if args.demo else "real"), s)


def _simulate(wb, args) -> int:
    if not args.demo:
        raise FundamentalsError("the simulated world is made in the demo database: add --demo (the real database never holds simulated companies)")
    planted = {"delta_roic": 0.011, "fcf_yield": 0.0068, "asset_growth": 0.0045} if args.strong else None
    report = wb.service.load_simulated(args.companies, args.from_year, args.to_year, args.seed, _progress(), planted)
    return _out(args, f"Made {report['companies']} simulated companies with {report.get('price_rows', 0):,} daily prices, {args.from_year} to {args.to_year}. They are for trying the tool: what is found in them says nothing about real markets.", report)


def _search(wb, args) -> int:
    text = " ".join(args.items)
    if len(text.strip()) < 2:
        raise FundamentalsError("say what to look for: quant fundamentals search nvidia (a name, a ticker such as NVDA or us:NVDA, or an ISIN)")
    found = wb.service.search_companies("openfilings", text, args.kind, args.market, 10, args.refresh)
    return _out(args, R.format_search(found), found)


def _add(wb, args) -> int:
    if not args.items:
        raise FundamentalsError("name at least one company, such as us:AAPL or jp:7203")
    ids = wb.service.add_companies("openfilings", args.items, None, args.market)
    return _out(args, f"{len(ids)} companies registered. Adding costs nothing; `quant fundamentals plan` says what fetching their statements would cost.", {"added": len(ids), "security_ids": ids})


def _plan_of(wb, args):
    svc = wb.service
    periods = tuple(args.periods)
    plan = svc.plan("openfilings", None, periods, args.start_year)
    return {"plan": plan.summary(), "affordability": svc.affordability("openfilings", plan), "tasks": [{"security_id": t.security_id, "symbol": t.symbol, "reason": t.reason, "cost": t.cost} for t in plan.tasks[:200]]}


def _plan(wb, args) -> int:
    p = _plan_of(wb, args)
    return _out(args, R.format_plan(p), p)


def _sync(wb, args) -> int:
    if args.budget is None:
        raise FundamentalsError("say how many credits this run may spend (--budget N): fetching statements uses the account's allowance. `quant fundamentals plan` shows what is needed")
    symbols = list(args.items) or None
    with _Stop() as stop:
        report = wb.service.sync("openfilings", symbols, None, tuple(args.periods), args.start_year, args.budget, _progress(), stop, args.market)
    wb.forget()
    return _out(args, R.format_sync(report), report)


def _prices(wb, args) -> int:
    with _Stop() as stop:
        report = wb.load_prices(None, args.price_start, None, _progress(), stop, args.refresh)
    return _out(args, R.format_prices(report), report)


def _factors(wb, args) -> int:
    if not args.scan:
        cat = wb.catalogue()
        return _out(args, R.format_factors(cat), cat["factors"])
    _ready(wb)
    ctx = _context(wb, args)
    with _Stop() as stop:
        scan = wb.scan(ctx, args.items or None, eval_spec(_eval_raw(args, None)), _progress(), stop)
    return _out(args, R.format_scan(scan, args.top), scan)


def _evaluate(wb, args) -> int:
    _ready(wb)
    if bool(args.formula) == bool(args.items):
        raise FundamentalsError("name one factor (`quant fundamentals factors` lists them) or give a formula with --formula, not both")
    ctx = _context(wb, args)
    if args.formula:
        study = wb.evaluate_formula(ctx, args.formula, eval_spec({**_eval_raw(args, None), "factor": None}))
    else:
        spec = eval_spec(_eval_raw(args, args.items[0]))
        study = wb.evaluate(ctx, spec)
    text = R.format_study(study)
    data = {"study": study}
    if args.lookahead and not args.formula:
        audit = wb.lookahead(eval_spec(_eval_raw(args, args.items[0])), _universe(args), args.freq, {"info_lag_days": args.info_lag} if args.info_lag is not None else None)
        text += "\n\nLook-ahead audit: " + R.format_lookahead(audit)
        data["lookahead"] = audit
    return _out(args, text, data)


def _search_spec(wb, args) -> DiscoverySpec:
    """The settings of a search from the command line (the same checks as the page's form)."""
    raw = {"universe": _universe(args), "freq": args.freq, "entry_lag": args.lag, "n_candidates": args.n_formulas, "seed": args.seed, "horizon": args.horizon, "max_nodes": args.max_nodes, "finalists": args.finalists,
           "cost_bps": args.cost_bps, "borrow_bps": args.borrow_bps}
    if args.strictness:
        raw["strictness"] = args.strictness
    if args.formulas is not None:
        raw["seeds"] = list(args.formulas)
    elif args.no_classics:
        raw["seeds"] = []
    return wb.discovery_spec(raw)


def _preflight(wb, args) -> int:
    _ready(wb)
    spec = _search_spec(wb, args)
    pf = wb.preflight(wb.context(spec.universe, spec.freq, spec.entry_lag), spec)
    return _out(args, R.format_preflight(pf), pf)


def _discover(wb, args) -> int:
    _ready(wb)
    d = Discovery(wb.store)
    if args.resume:
        record = d.get_run(args.resume)
        if record["status"] == "finished":
            raise FundamentalsError("that run has finished: its locked test has been looked at, and looking again would spend it. Start a new run instead")
        spec, run_id = DiscoverySpec.from_dict(record["spec"]), args.resume
    else:
        spec = _search_spec(wb, args)
        run_id = d.create(spec)
        _say(f"run {run_id}")
    ctx = wb.context(spec.universe, spec.freq, spec.entry_lag)
    progress = _progress()
    with _Stop() as stop:
        summary = Discovery(wb.store, ctx).run(run_id, lambda done, total, text: progress(f"{done} of {total} formulas: {text}"), stop)
    text = R.format_discovery(summary, args.top)
    if summary["status"] != "finished":
        text += f"\n\nStopped. Carry on exactly where it left off with: quant fundamentals discover {'--demo ' if args.demo else ''}--resume {run_id}"
    return _out(args, text, summary)


def _placebo(wb, args) -> int:
    _ready(wb)
    real = None
    if args.run:
        record = Discovery(wb.store).get_run(args.run)
        counts = record["stage_counts"]
        if record["status"] != "finished":
            raise FundamentalsError("that run has not finished: check a finished run, or leave --run out to check a search that has not been run")
        spec, real = DiscoverySpec.from_dict(record["spec"]), {"tested": counts.get("tested"), "survivors": counts.get("survivors"), "candidates": counts.get("candidates")}
    else:
        spec = _search_spec(wb, args)
    with _Stop() as stop:
        result = wb.placebo(spec, args.repeats, real, _progress(), stop)
    text = R.format_placebo(result)
    if not result["complete"]:
        text += "\n\nStopped before every shuffle finished: nothing was kept."
    return _out(args, text, result)


def _library(wb, args) -> int:
    lib = wb.library
    if args.validate:
        row = lib.validate(args.validate, args.reason)
        return _out(args, f"{row['alpha_id']} is now {row['status']}: {row['status_reason']}", row)
    if args.set_status:
        if not args.to:
            raise FundamentalsError(f"say the status to move it to (--to {'|'.join(STATUSES)}) and why (--reason)")
        row = lib.set_status(args.set_status, args.to, args.reason or "", args.force)
        return _out(args, f"{row['alpha_id']} is now {row['status']}", row)
    if args.graveyard:
        rows = lib.graveyard(args.top, args.stage)
        return _out(args, R.format_graveyard(rows, lib.counts()), rows)
    rows = lib.list(args.status, args.order, args.top)
    return _out(args, R.format_library(rows, lib.counts()), rows)


def _monitor(wb, args) -> int:
    _ready(wb)
    with _Stop() as stop:
        result = wb.monitor(None, _progress(), stop)
    return _out(args, R.format_monitor(result), result)


def _backtest(wb, args) -> int:
    _ready(wb)
    chosen = [x for x in (args.alpha, args.factor, args.formula) if x]
    if len(chosen) != 1:
        raise FundamentalsError("say what to backtest: one of --alpha ID, --factor NAME or --formula TEXT")
    if args.alpha:
        signal_spec = {"kind": "alpha", "alpha_id": args.alpha}
    elif args.factor:
        signal_spec = {"kind": "factor", "name": args.factor}
    else:
        checked = wb.check_formula(args.formula)
        if not checked["ok"]:
            raise FundamentalsError(f"{args.formula}: {checked['error']}")
        signal_spec = {"kind": "expression", "expression": checked["tree"]}
    portfolio = {"side": args.side, "selection": "percentile" if args.percentile is not None else "top_n", "weighting": args.weighting_book, "cost_bps": args.cost_bps, "borrow_bps": args.borrow_bps,
                 "capital_usd": args.capital}
    if args.percentile is not None:
        portfolio["percentile"] = args.percentile
    else:
        portfolio["n"] = args.leg_size
    for key in ("max_position", "max_sector", "max_industry", "max_country", "max_beta", "max_turnover"):
        value = getattr(args, key)
        if value is not None:
            portfolio[key] = value
    ctx = _context(wb, args)
    result = wb.backtest(ctx, signal_spec, portfolio, None, args.trials)
    return _out(args, R.format_backtest(result), result)


HANDLERS = {"status": _status, "simulate": _simulate, "search": _search, "add": _add, "plan": _plan, "sync": _sync, "prices": _prices, "factors": _factors, "evaluate": _evaluate, "preflight": _preflight,
            "discover": _discover, "placebo": _placebo, "library": _library, "monitor": _monitor, "backtest": _backtest}


def cmd_fundamentals(args, config) -> int:
    try:
        if args.demo and args.action in REAL_ONLY:
            raise FundamentalsError(f"{REAL_ONLY[args.action]} is for real data: the demo database is a simulated world and never calls a provider")
        if args.action == "simulate" and not args.demo:
            raise FundamentalsError("the simulated world is made in the demo database: add --demo (the real database never holds simulated companies)")
        if args.action == "simulate" and args.reset:
            for suffix in ("", "-wal", "-shm"):                                              # start the demo file again: only the demo, never the real database
                Path(str(_path(config, True)) + suffix).unlink(missing_ok=True)
        wb = _bench(config, args.demo, args.action)
        return HANDLERS[args.action](wb, args)
    except FundamentalsError as error:
        print(error, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ended", file=sys.stderr)
        return 130


class _Intermixed(argparse.ArgumentParser):
    """A parser that accepts the names of companies or factors after an option too (``factors --scan fcf_yield roic --demo``, ``sync --budget 20 us:AAPL``): argparse alone takes positionals only before the first option."""

    _mixing = False

    def parse_known_args(self, args=None, namespace=None):
        if self._mixing:
            return super().parse_known_args(args, namespace)
        self._mixing = True
        try:
            return self.parse_known_intermixed_args(args, namespace)
        finally:
            self._mixing = False


def register(sub) -> None:
    ordinary = sub._parser_class
    sub._parser_class = _Intermixed                                                          # the sub-command's parser is made by the action that holds the class
    try:
        _add_parser(sub)
    finally:
        sub._parser_class = ordinary


def _add_parser(sub) -> None:
    p = sub.add_parser("fundamentals", help="company fundamentals in a local database: factor studies, automated alpha discovery, the alpha library, its monitor and the backtest of an alpha (the Fundamentals tab, in the terminal)",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=ACTIONS, help="what to do (see above)")
    p.add_argument("items", nargs="*", help="search: the name, ticker or ISIN to look for; add: tickers (us:AAPL jp:7203); sync: limit the fetch to these; evaluate: the factor; factors --scan: limit the scan to these")
    p.add_argument("--demo", action="store_true", help="work in the demo database (a simulated world) instead of the real one")
    p.add_argument("--json", help="also write the whole result to this file")
    p.add_argument("--top", type=int, default=25, help="rows to show of a long table, and how many alphas `library` lists")
    u = p.add_argument_group("the companies and dates of a study")
    u.add_argument("--countries", nargs="*", help="only these countries (US JP GB ...)"); u.add_argument("--sectors", nargs="*", help="only these sectors")
    u.add_argument("--min-cap", type=float, help="smallest market value, US dollars"); u.add_argument("--min-volume", type=float, help="smallest daily trading value, US dollars")
    u.add_argument("--developed", action="store_true", help="developed markets only"); u.add_argument("--no-delisted", action="store_true", help="leave out companies that have since delisted (survivorship bias gets in this way)")
    u.add_argument("--freq", choices=["W", "M", "Q"], default="M", help="how often the portfolio is rebalanced"); u.add_argument("--lag", type=int, default=1, help="days between a signal and the trade")
    u.add_argument("--info-lag", type=int, help="days after a filing before its figures are used (default 1)"); u.add_argument("--start", help="first date, 2012-01-31"); u.add_argument("--end", help="last date")
    d = p.add_argument_group("simulate")
    d.add_argument("--companies", type=int, default=120); d.add_argument("--from-year", type=int, default=2009); d.add_argument("--to-year", type=int, default=2022); d.add_argument("--seed", type=int, default=7)
    d.add_argument("--reset", action="store_true", help="start the demo file again"); d.add_argument("--strong", action="store_true", help="plant stronger alphas (easier to find in a small world)")
    s = p.add_argument_group("search, add, plan, sync, prices")
    s.add_argument("--market", help="search: only this market; add, sync: the market of tickers typed without one (us, jp, uk, ...)"); s.add_argument("--start-year", type=int, help="first fiscal year to fetch")
    s.add_argument("--kind", choices=["auto", "name", "symbol", "isin", "lei"], default="auto", help="search: what the text is (auto reads it from its shape: us:NVDA and NVDA are tickers, Nvidia is a name)")
    s.add_argument("--periods", nargs="*", choices=["annual", "quarter"], default=["annual"]); s.add_argument("--budget", type=int, help="sync: the most credits this run may spend (required: statements use the account's allowance)")
    s.add_argument("--price-start", default="2010-01-01", help="prices: the first date"); s.add_argument("--refresh", action="store_true", help="prices: fetch the whole history again; search: ask OpenFilings again instead of using the saved answer")
    e = p.add_argument_group("evaluate, factors --scan")
    e.add_argument("--formula", help="evaluate or backtest a formula such as div(fcf,market_cap)"); e.add_argument("--scan", action="store_true", help="factors: study every factor and set them side by side")
    e.add_argument("--horizon", default="1m", help="the return measured over the next 1w, 2w, 1m, 2m, 3m, 6m or 12m"); e.add_argument("--direction", default="theory", help="theory (the factor's own), auto (counts as a second trial), 1 or -1")
    e.add_argument("--quantiles", type=int, default=5); e.add_argument("--weighting", choices=["equal", "signal", "inv_vol"], default="equal"); e.add_argument("--cost-bps", type=float, default=10.0)
    e.add_argument("--borrow-bps", type=float, default=50.0); e.add_argument("--winsor", type=float, default=1.0, help="cut this per cent of each tail of each date (0: none)")
    e.add_argument("--scale", choices=["zscore", "rank", "percentile", "raw"], default="zscore"); e.add_argument("--neutralize", nargs="*", choices=["sector", "industry", "country", "market"], help="remove what the factor shares with these groups")
    e.add_argument("--control", nargs="*", choices=["mcap", "beta", "volatility"], help="and with these measures"); e.add_argument("--lookahead", action="store_true", help="evaluate: also run the study as if every figure were known at its period end, and after waiting 30, 60 and 90 days")
    e.add_argument("--premium", action="store_true", help="evaluate: also the premium from a regression on every date holding size, beta, volatility and momentum fixed (Fama-MacBeth)")
    e.add_argument("--premium-controls", nargs="*", metavar="NAME", help="evaluate: hold these fixed instead (size beta volatility momentum, or the name of a library factor such as roic)")
    f = p.add_argument_group("preflight, discover, placebo")
    f.add_argument("--strictness", choices=list(STRICTNESS), help="the bars a formula must clear: exploratory (a first look), standard (the default: |t| of 2, q-value 0.10) or strict (|t| of 3, q-value 0.05)")
    f.add_argument("--repeats", type=int, default=5, help="placebo: how many times to shuffle the returns and search again (1 to 5; the spread of the t-statistics needs at least three)"); f.add_argument("--run", help="placebo: check this finished run (its search, set beside what it found)")
    f.add_argument("--n", type=int, default=300, dest="n_formulas", help="formulas to try"); f.add_argument("--finalists", type=int, default=15, help="formulas given the one look at the locked dates"); f.add_argument("--max-nodes", type=int, default=15)
    f.add_argument("--formulas", nargs="*", help="start from only these formulas, then random ones"); f.add_argument("--no-classics", action="store_true", help="random formulas only"); f.add_argument("--resume", help="carry on a stopped run")
    g = p.add_argument_group("library")
    g.add_argument("--status", choices=list(STATUSES)); g.add_argument("--order", default="fitness", choices=["fitness", "ic", "icir", "sharpe", "oos_ic", "oos_sharpe", "created_at", "turnover"])
    g.add_argument("--graveyard", action="store_true", help="list the ideas that failed"); g.add_argument("--stage", help="graveyard: only those that failed at this stage"); g.add_argument("--validate", metavar="ALPHA_ID", help="Candidate to Validated, if the evidence holds")
    g.add_argument("--set-status", metavar="ALPHA_ID", help="move an alpha along the allowed paths (--to and --reason are required)"); g.add_argument("--to", choices=list(STATUSES)); g.add_argument("--reason", help="why (a move is never made without one)")
    g.add_argument("--force", action="store_true", help="a move the table does not allow, recorded as forced")
    b = p.add_argument_group("backtest")
    b.add_argument("--alpha", help="a library alpha"); b.add_argument("--factor", help="a library factor"); b.add_argument("--side", choices=["long_only", "long_short"], default="long_only")
    b.add_argument("-n", "--legs", type=int, default=30, dest="leg_size", help="companies in each leg"); b.add_argument("--percentile", type=float, help="instead of -n: the top per cent of the eligible companies in each leg")
    b.add_argument("--weighting-book", choices=["equal", "signal", "inverse_vol"], default="equal"); b.add_argument("--capital", type=float, default=1_000_000.0, help="account size, US dollars")
    for name in ("max-position", "max-sector", "max-industry", "max-country", "max-beta", "max-turnover"):
        b.add_argument(f"--{name}", type=float, help="a limit, as a share of the account (blank: none)")
    b.add_argument("--trials", type=int, help="ideas tried to find this alpha, for the deflated Sharpe ratio (default: from the library)")
    p.set_defaults(func=cmd_fundamentals)


__all__ = ["register", "cmd_fundamentals", "ACTIONS"]
