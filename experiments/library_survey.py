"""Run every rule-based library strategy once on the platform's 15 ETFs and write ``docs/strategy_survey.md``.

This is a survey, not a study: nothing was tuned, every strategy runs with its default parameters, rules are traded as written (``score_stack``, no calibration against
history) with the rebalance frequency each rule declares, costs are the platform defaults, and the deflated Sharpe ratio counts every strategy in the table as a trial.
It answers "what does each popular rule do on this universe and sample", and the honest reading of a list this long is that the top of it is partly luck.

    python -m experiments.library_survey          # about five minutes on a laptop
    python -m experiments.library_survey --new    # only the strategies added since the saved survey, counting every strategy as a trial (writes docs/strategy_survey_2.md)
    python -m experiments.library_survey --part3  # only the strategies added since parts one and two (writes docs/strategy_survey_3.md)
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from src.framework import MODELS, Pipeline, PipelineSpec, load_default_bundle, load_library
from src.utils.config import load_config

SKIP_FAMILIES = ("machine learning", "crypto", "custom")


def survey_names() -> list[str]:
    load_library()
    return [e.name for e in MODELS.entries() if e.family not in SKIP_FAMILIES and not e.name.startswith("test_") and getattr(e.factory, "__module__", "").startswith("src.strategies")]


def run_survey(config, bundle, names: list[str] | None = None, n_trials: int | None = None) -> pd.DataFrame:
    """One row per strategy. ``n_trials`` is the number of strategies the deflated Sharpe ratio counts (default: the strategies run now)."""
    names = names or survey_names()
    rows = []
    for name in names:
        entry = next(e for e in MODELS.entries() if e.name == name)
        structured = bool(getattr(entry.factory, "structured", False))
        spec = {"name": name, "models": [{"name": name}], "evaluation": {"causality": False, "benchmarks": ["equal_weight"], "n_trials": n_trials or len(names)}}
        if not structured:
            spec["allocation"] = {"allocator": "sleeves" if getattr(entry.factory, "book", None) == "sleeves" else "score_stack"}
        started = time.perf_counter()
        try:
            result = Pipeline(PipelineSpec.from_dict(spec), config, bundle).run(validate=False)
            m = result.metrics
            bench = result.tables["benchmarks"]
            rows.append({"strategy": name, "family": entry.family, "net_sharpe": m["sharpe"], "cagr": m["cagr"], "volatility": m["ann_vol"], "max_drawdown": m["max_drawdown"],
                         "turnover": m["ann_turnover"], "deflated_sharpe_probability": result.validation["deflated_sharpe_probability"], "start": m["start"],
                         "equal_weight_sharpe": float(bench["sharpe_benchmark"].iloc[0]) if "sharpe_benchmark" in bench and len(bench) else float("nan"),
                         "seconds": round(time.perf_counter() - started, 1), "error": ""})
        except Exception as error:                                          # a strategy that cannot run on this bundle is reported, not hidden
            rows.append({"strategy": name, "family": entry.family, "error": f"{type(error).__name__}: {error}"[:120]})
    return pd.DataFrame(rows)


def render(table: pd.DataFrame, first: str, last: str, years: float) -> str:
    ok = table[table["error"] == ""].sort_values("net_sharpe", ascending=False)
    n = len(table)
    eq = float(ok["equal_weight_sharpe"].dropna().median()) if len(ok) else float("nan")
    lines = [
        "# Strategy survey: every rule-based strategy, once, on the same 15 ETFs", "",
        f"{n} strategies, the platform's 15 ETFs, {first} to {last} ({years:.1f} years), net of costs. **This is a survey, not a study.** Nothing was tuned: every strategy uses its default parameters,",
        "rules are traded as written (no calibration against history) at the rebalance frequency each rule declares (daily, weekly or monthly): per-asset rules such as a pullback entry or a trend filter run as independent sleeves with an equal slice of capital each and cash when flat, selection and cross-sectional rules as a ranked book; costs are the platform's 10 bps per unit traded,",
        f"and the deflated Sharpe probability counts all {n} strategies as trials. Equal weight over the same ETFs earns a net Sharpe of about {eq:.2f} (the median over the windows below). **Each strategy starts on a different day** (it needs history before its first signal), so compare it with the *Equal weight, same dates* column, not with a single number.", "",
        "How to read it: with this many strategies, some at the top are there by luck. Look for a rule that beats equal weight **and** has a deflated probability near or above 0.95, then ask whether its mechanism is plausible and "
        "whether it survives your own tickers, a different period and a higher cost. Several rules (calendar effects, single-stock factors) are weaker on a 15-ETF universe than in the papers that made them famous.", "",
        "| Strategy | Family | Net Sharpe | Equal weight, same dates | CAGR | Volatility | Max drawdown | Turnover (x/yr) | Deflated Sharpe | First day |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in ok.iterrows():
        dsr = "n/a" if pd.isna(r["deflated_sharpe_probability"]) else f"{r['deflated_sharpe_probability']:.2f}"
        sharpe = "n/a" if pd.isna(r["net_sharpe"]) else f"{r['net_sharpe']:+.2f}"
        ew = "n/a" if pd.isna(r["equal_weight_sharpe"]) else f"{r['equal_weight_sharpe']:+.2f}"
        lines.append(f"| [{r['strategy']}](strategies/{r['strategy']}.md) | {r['family']} | {sharpe} | {ew} | {r['cagr']:.1%} | {r['volatility']:.1%} | {r['max_drawdown']:.1%} | {r['turnover']:.1f} | {dsr} | {r['start']} |")
    failed = table[table["error"] != ""]
    if len(failed):
        lines += ["", "Could not run on this bundle:", ""] + [f"- `{r['strategy']}`: {r['error']}" for _, r in failed.iterrows()]
    lines += ["", "Regenerate with `python -m experiments.library_survey`. Single runs of this kind are exploratory: see [the research reports](../README.md) for the pre-registered studies.", ""]
    return "\n".join(lines)


PART_THREE_NOTE = ("How to read it: these are the factor-timing, characteristic-regression, APT, macro-factor and equity-factor rules of the two books. Rules that read a file the repository does not have "
                   "(`fundamental_*`, `earnings_season_premium`) are listed under *could not run*, not scored. Rules that learn from earlier data start late and stay flat while they have no evidence. "
                   "A negative or flat row on 15 ETFs says the rule does not pay on liquid funds at these costs, not that it fails on the single stocks it was written for.")


def render_supplement(added: pd.DataFrame, earlier: int, total: int, first: str, last: str, years: float, part: int = 2) -> str:
    """The part of the survey written after the first one: same universe, sample, costs and defaults, with the deflated Sharpe probability counting every strategy (``total``) as a trial."""
    ok = added[added["error"] == ""].sort_values("net_sharpe", ascending=False)
    title = "# Strategy survey, part two: investment, portfolio and economic-outlook strategies" if part == 2 else "# Strategy survey, part three: factor timing, factor models and equity factors"
    before = "the first survey ([strategy_survey.md](strategy_survey.md)" if part == 2 else "parts one and two ([strategy_survey.md](strategy_survey.md), [strategy_survey_2.md](strategy_survey_2.md)"
    lines = [
        title, "",
        f"{len(added)} strategies added after {before}: {earlier} strategies), run the same way: the platform's 15 ETFs, {first} to {last} ({years:.1f} years), net of costs, default parameters,",
        f"nothing tuned. **The deflated Sharpe probability here counts all {total} strategies as trials** ({earlier} before, {len(added)} now). The first survey's probabilities counted {earlier}, so they are slightly flattering next to these. This is a survey, not a study.", "",
        ("How to read it: several of these rules are made for single stocks, company news or data you supply, and ETFs are not their natural test bed; a negative row on 15 ETFs says the rule does not pay on liquid funds at these costs, not that it fails on the assets it was written for. "
         "Rules that learn from earlier data (`curve_quadrant`, `credit_cycle_rotation`, `event_study_drift`) start late and stay flat while they have no evidence.") if part == 2 else PART_THREE_NOTE, "",
        "| Strategy | Family | Net Sharpe | Equal weight, same dates | CAGR | Volatility | Max drawdown | Turnover (x/yr) | Deflated Sharpe | First day |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in ok.iterrows():
        dsr = "n/a" if pd.isna(r["deflated_sharpe_probability"]) else f"{r['deflated_sharpe_probability']:.2f}"
        sharpe = "n/a" if pd.isna(r["net_sharpe"]) else f"{r['net_sharpe']:+.2f}"
        ew = "n/a" if pd.isna(r["equal_weight_sharpe"]) else f"{r['equal_weight_sharpe']:+.2f}"
        lines.append(f"| [{r['strategy']}](strategies/{r['strategy']}.md) | {r['family']} | {sharpe} | {ew} | {r['cagr']:.1%} | {r['volatility']:.1%} | {r['max_drawdown']:.1%} | {r['turnover']:.1f} | {dsr} | {r['start']} |")
    failed = added[added["error"] != ""]
    if len(failed):
        lines += ["", "Could not run on this bundle:", ""] + [f"- `{r['strategy']}`: {r['error']}" for _, r in failed.iterrows()]
    flag = "--new" if part == 2 else "--part3"
    lines += ["", f"Regenerate with `python -m experiments.library_survey {flag}`; the next full run (`python -m experiments.library_survey`) folds these rows into the first survey.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--new", action="store_true", help="run only the strategies missing from the saved survey; the deflated Sharpe counts all of them as trials")
    parser.add_argument("--part3", action="store_true", help="run only the strategies missing from the saved survey and from parts one and two, and write docs/strategy_survey_3.md")
    parser.add_argument("--redo", nargs="*", default=[], help="with --new: also run these again, replacing their rows")
    args = parser.parse_args(argv)
    config = load_config()
    bundle = load_default_bundle(config)
    first, last = str(bundle.index[0].date()), str(bundle.index[-1].date())
    tables = Path(config.root) / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    saved = tables / "library_survey.csv"
    if args.new or args.part3:
        lines = saved.read_text(encoding="utf-8").splitlines(keepends=True)                                 # old rows are kept as written, so the file only grows
        kept_lines = lines[:1] + [row for row in lines[1:] if row.split(",", 1)[0] not in set(args.redo)]
        kept_names = {row.split(",", 1)[0] for row in kept_lines[1:]}
        names = [n for n in survey_names() if n not in kept_names]
        total = len(kept_names) + len(names)
        added = run_survey(config, bundle, names, n_trials=total) if names else None
        saved.write_text("".join(kept_lines) + (added.to_csv(index=False, header=False) if added is not None else ""), encoding="utf-8")
        combined = pd.read_csv(saved).fillna({"error": ""})
        first_page = (Path(config.root) / "docs" / "strategy_survey.md").read_text(encoding="utf-8")
        first_survey = set(re.findall(r"\]\(strategies/([a-z0-9_]+)\.md\)", first_page)) | set(re.findall(r"^- `([a-z0-9_]+)`:", first_page, flags=re.M))      # the ones that ran and the ones that could not
        second_page = (Path(config.root) / "docs" / "strategy_survey_2.md").read_text(encoding="utf-8")
        second_survey = set(re.findall(r"\]\(strategies/([a-z0-9_]+)\.md\)", second_page)) | set(re.findall(r"^- `([a-z0-9_]+)`:", second_page, flags=re.M))
        if args.part3:                                                                                      # part two stays as written; everything newer is part three
            earlier = first_survey | second_survey
            part_three = combined[~combined["strategy"].isin(earlier)]
            out = Path(config.root) / "docs" / "strategy_survey_3.md"
            out.write_text(render_supplement(part_three, len(earlier), len(earlier) + len(part_three), first, last, len(bundle.index) / 252.0, part=3), encoding="utf-8")
        else:
            part_two = combined[~combined["strategy"].isin(first_survey)]                                   # everything run since the first survey, not only this run's rows
            out = Path(config.root) / "docs" / "strategy_survey_2.md"
            out.write_text(render_supplement(part_two, len(first_survey), len(first_survey) + len(part_two), first, last, len(bundle.index) / 252.0), encoding="utf-8")
    else:
        table = run_survey(config, bundle)
        out = Path(config.root) / "docs" / "strategy_survey.md"
        out.write_text(render(table, first, last, len(bundle.index) / 252.0), encoding="utf-8")
        table.to_csv(saved, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
