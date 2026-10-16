"""How well do the forecasting methods really do on real prices? The tables quoted in docs/forecasting.md.

Each method of ``src.forecast`` is tested the way the Forecast tab tests it: walk forward over the last eight years of an ETF (a forecast every ten trading days, refitted every 63, 1,000 simulated futures
each), every forecast made from the prices up to its day and scored against what happened, with the random walk of constant volatility as the yardstick.

    python -m experiments.forecast_study                                   # SPY, QQQ, EEM, TLT and GLD at 21 days (about two minutes each)
    python -m experiments.forecast_study --tickers SPY --horizons 5 21 63  # one ticker at three horizons
    python -m experiments.forecast_study --tickers SPY --methods garch hmm gp_drift --effort quick

It prints, for every horizon, the skill of each method on each ticker (how far below the random walk's CRPS its own was; a star means the gain held up after allowing for the number of methods compared),
then for each ticker the calibration of the yardstick and the verdict the tab gives, and the tests of the average return and of the direction against the average of the last five years and against always
saying up. Nothing here touches the network: the prices are the platform's cached ETFs.
"""

from __future__ import annotations

import argparse
import time
import warnings

from src.forecast import BASELINE, BLEND, METHODS, History
from src.forecast import evaluate as ev
from src.forecast.verdict import judge
from src.framework import load_default_bundle
from src.utils.config import load_config

TICKERS = ("SPY", "QQQ", "EEM", "TLT", "GLD")


def run(history: History, names: list[str], horizon: int, effort: str, seed: int) -> dict:
    return ev.evaluate(history, names, horizon, effort, seed)


def _row(result: dict, mid: str) -> dict:
    return next(m for m in result["methods"] if m["id"] == mid)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tickers", nargs="*", default=list(TICKERS))
    ap.add_argument("--horizons", nargs="*", type=int, default=[21])
    ap.add_argument("--methods", nargs="*", help="method ids (default: every method and the blend)")
    ap.add_argument("--effort", choices=list(ev.EFFORTS), default="standard")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    bundle = load_default_bundle(load_config(), with_macro=False)
    names = args.methods or [n for n in METHODS if n != BASELINE] + [BLEND]
    results: dict[tuple[str, int], dict] = {}
    for hz in args.horizons:
        for ticker in args.tickers:
            history = History.from_frame(bundle.prices[ticker], None if bundle.high is None else bundle.high.get(ticker), None if bundle.low is None else bundle.low.get(ticker))
            t0 = time.time()
            results[(ticker, hz)] = run(history, names, hz, args.effort, args.seed)
            print(f"{ticker} at {hz} days: {time.time() - t0:.0f} s", flush=True)
    for hz in args.horizons:
        done = [t for t in args.tickers if (t, hz) in results]
        first = results[(done[0], hz)]["origins"]
        print(f"\n== skill at {hz} days: how far below the random walk's CRPS each method's was ({first['count']} forecasts, {first['first']} to {first['last']}; * = the gain holds after the Holm adjustment) ==")
        print(f"{'method':18s}" + "".join(f"{t:>10s}" for t in done) + f"{'average':>10s}")
        for mid in names:
            cells, skills = "", []
            for t in done:
                s = _row(results[(t, hz)], mid)["scores"][str(hz)]
                skills.append(s["skill"])
                cells += f"{s['skill'] * 100:+8.1f}%{'*' if s.get('beats_baseline') else ' '}"
            print(f"{mid:18s}{cells}{sum(skills) / len(skills) * 100:+9.1f}%")
        print(f"\n== calibration of the yardstick and the verdict, at {hz} days ==")
        for t in done:
            r = results[(t, hz)]
            base = _row(r, BASELINE)["scores"][str(hz)]
            v = judge(r, hz)
            print(f"{t:5s} random walk: 90% range held {base['coverage']['90'] * 100:.0f}%, fell below its 5% quantile {base['var5'] * 100:.1f}%, calibration p {base['pit_p']:.3f}   verdict {v['level']}: {v['headline']}")
        print(f"\n== the average return (out-of-sample R-squared against the five-year average; Clark-West p) and the direction (hit rate against always saying up), at {hz} days ==")
        for t in done:
            r = results[(t, hz)]
            rows = [(m["id"], m["scores"][str(hz)]) for m in r["methods"] if "mean" in m["scores"][str(hz)]]
            best = max(rows, key=lambda x: x[1]["mean"]["r2"])
            top = max(rows, key=lambda x: x[1]["direction"]["edge"])
            print(f"{t:5s} best R-squared {best[1]['mean']['r2'] * 100:+.1f}% ({best[0]}, p {best[1]['mean']['p']:.3f}); always up {top[1]['direction']['always_up'] * 100:.0f}%, best caller {top[0]} at {top[1]['direction']['hit'] * 100:.0f}% (p {top[1]['direction']['p']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
