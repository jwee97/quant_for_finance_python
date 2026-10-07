"""``quant demo``: a five-minute guided tour on the cached data.

It runs one real strategy through the whole pipeline, reads the red-flag list aloud, then shows on live numbers the most important lesson of the
platform: the same strategy tried under several parameter settings looks better and better as the best one is picked, and the deflated Sharpe
ratio is what pulls it back. Nothing is canned; the numbers come from the run. A scratch database is used so the tour never touches your own runs.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .experiments import ExperimentManager
from .tearsheet import red_flags


def _say(text: str = "") -> None:
    print(text, flush=True)


def run_demo(config) -> int:
    from . import DETECTORS, load_default_bundle, load_library

    import logging

    logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)
    started = time.time()
    load_library()
    _say("QUANT PLATFORM DEMO (about five minutes)")
    _say("=" * 60)
    bundle = load_default_bundle(config)
    _say(f"\n1. The data: {len(bundle.assets)} ETFs ({', '.join(bundle.assets[:6])}, ...), {bundle.index[0].date()} to {bundle.index[-1].date()}.")
    _say("   Equities, bonds, gold, silver, commodities, real estate. Prices are adjusted for splits and dividends; nothing was deleted by hand.")

    scratch = Path(tempfile.mkdtemp(prefix="quant_demo_"))
    manager = ExperimentManager(config, db_path=scratch / "demo.db", runs_dir=scratch / "runs")

    _say("\n2. One strategy through the whole pipeline: dual momentum (hold the top assets by 12-month return, but only those that beat cash).")
    spec = {"name": "dual_momentum", "models": [{"name": "dual_momentum"}], "regime": {"detector": "vol_state"}, "evaluation": {"group": "demo"}}
    run_id, result = manager.run(spec, bundle, tearsheet=True, force=True)
    m = result.metrics
    _say(f"   {m['start']} to {m['end']}:  net Sharpe {m['sharpe']:+.2f}  (before costs {m['gross_sharpe']:+.2f}),  CAGR {m['cagr']:.1%},  volatility {m['ann_vol']:.1%},  worst drawdown {m['max_drawdown']:.1%}")
    _say("   Costs are charged on every trade and the weights are applied one day after the decision, so the number is net and cannot see the future.")

    _say("\n3. The red-flag list (fixed rules, each with its threshold; this is what a sceptical reviewer asks first):")
    flags = red_flags(result)
    for f in flags:
        _say(f"   [{'FLAG' if f['flag'] else ' ok '}] {f['rule']}: {f['observed']}")
    _say(f"   Full tear sheet: reports/tearsheets/{run_id}/tearsheet.md")

    _say("\n4. Regimes: the market state the detector believes it is in (filtered probabilities, so only information available on each day).")
    regimes = DETECTORS.create("vol_state").detect(bundle)
    for name, share in regimes.share().round(3).items():
        _say(f"   {name:<10s} {share:6.1%} of days")

    _say("\n5. The most important lesson: how many things did you try, and compared with what?")
    _say("   The same idea under five lookback settings. A researcher who reports only the best has not told you the whole story.")
    import numpy as np

    from ..backtest.metrics import deflated_sharpe_ratio

    rows = []
    for lookback in (63, 126, 189, 252, 315):
        sp = {"name": f"dual_momentum[lookback={lookback}]", "models": [{"name": "dual_momentum", "params": {"lookback": lookback}}], "evaluation": {"group": "demo_search", "causality": False}}
        _, r = manager.run(sp, bundle, force=True, validate=True)
        rows.append((lookback, r))
        _say(f"   lookback {lookback:>3d}:  net Sharpe {r.metrics['sharpe']:+.2f}")
    sharpes = np.array([r.metrics["sharpe"] for _, r in rows])
    best_lookback, best = rows[int(sharpes.argmax())]
    from ..backtest.metrics import performance_summary

    perf = performance_summary(best.window)
    dsr = deflated_sharpe_ratio(perf["sharpe"], len(rows), len(best.window), perf["skew"], perf["excess_kurtosis"] + 3.0)
    bench = best.tables.get("benchmarks")
    _say(f"   Best {sharpes.max():+.2f} (lookback {best_lookback}), median {np.median(sharpes):+.2f}, worst {sharpes.min():+.2f}: the spread is the price of choosing a parameter.")
    _say(f"   Deflated Sharpe probability of the best, counting all {len(rows)} tries: {dsr:.2f}. It asks whether the best beats what the best of {len(rows)} strategies with no skill would show.")
    if bench is not None and len(bench):
        for _, b in bench.dropna(subset=["p_value"]).iterrows():
            _say(f"   Against {b['benchmark']}: Sharpe difference {b['difference']:+.2f} (paired bootstrap p = {b['p_value']:.2f}).")
    _say("   A strategy can clear 'better than luck' and still not beat simply holding the assets in equal weight. Both questions have to be asked; the tear sheet asks both.")

    _say("\n6. Where to go next")
    _say("   quant dashboard                 an explorer of every hypothesis the project ever declared (most were rejected, on purpose)")
    _say("   quant explain deflated sharpe   plain-language explanations of any term, technique, strategy or experiment")
    _say("   quant list models               the 90 models you can run with `quant backtest --model NAME`")
    _say("   docs/START_HERE.md              a learning path for your background")
    _say(f"\nDone in {time.time() - started:.0f} seconds.")
    return 0
