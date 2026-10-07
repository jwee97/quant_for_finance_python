"""Run every rule-based library strategy once on the platform's 15 ETFs and write ``docs/strategy_survey.md``.

This is a survey, not a study: nothing was tuned, every strategy runs with its default parameters, rules are traded as written (``score_stack``, no calibration against
history) with the rebalance frequency each rule declares, costs are the platform defaults, and the deflated Sharpe ratio counts every strategy in the table as a trial.
It answers "what does each popular rule do on this universe and sample", and the honest reading of a list this long is that the top of it is partly luck.

    python -m experiments.library_survey          # about five minutes on a laptop
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.framework import MODELS, Pipeline, PipelineSpec, load_default_bundle, load_library
from src.utils.config import load_config

SKIP_FAMILIES = ("machine learning", "crypto", "custom")


def survey_names() -> list[str]:
    load_library()
    return [e.name for e in MODELS.entries() if e.family not in SKIP_FAMILIES and not e.name.startswith("test_") and getattr(e.factory, "__module__", "").startswith("src.strategies")]


def run_survey(config, bundle, names: list[str] | None = None) -> pd.DataFrame:
    names = names or survey_names()
    rows = []
    for name in names:
        entry = next(e for e in MODELS.entries() if e.name == name)
        structured = bool(getattr(entry.factory, "structured", False))
        spec = {"name": name, "models": [{"name": name}], "evaluation": {"causality": False, "benchmarks": ["equal_weight"], "n_trials": len(names)}}
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
        f"and the deflated Sharpe probability counts all {n} strategies as trials. Equal weight over the same ETFs earns a net Sharpe of about {eq:.2f} on these dates.", "",
        "How to read it: with this many strategies, some at the top are there by luck. Look for a rule that beats equal weight **and** has a deflated probability near or above 0.95, then ask whether its mechanism is plausible and "
        "whether it survives your own tickers, a different period and a higher cost. Several rules (calendar effects, single-stock factors) are weaker on a 15-ETF universe than in the papers that made them famous.", "",
        "| Strategy | Family | Net Sharpe | CAGR | Volatility | Max drawdown | Turnover (x/yr) | Deflated Sharpe | First day |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in ok.iterrows():
        dsr = "n/a" if pd.isna(r["deflated_sharpe_probability"]) else f"{r['deflated_sharpe_probability']:.2f}"
        sharpe = "n/a" if pd.isna(r["net_sharpe"]) else f"{r['net_sharpe']:+.2f}"
        lines.append(f"| [{r['strategy']}](strategies/{r['strategy']}.md) | {r['family']} | {sharpe} | {r['cagr']:.1%} | {r['volatility']:.1%} | {r['max_drawdown']:.1%} | {r['turnover']:.1f} | {dsr} | {r['start']} |")
    failed = table[table["error"] != ""]
    if len(failed):
        lines += ["", "Could not run on this bundle:", ""] + [f"- `{r['strategy']}`: {r['error']}" for _, r in failed.iterrows()]
    lines += ["", "Regenerate with `python -m experiments.library_survey`. Single runs of this kind are exploratory: see [the research reports](../README.md) for the pre-registered studies.", ""]
    return "\n".join(lines)


def main() -> int:
    config = load_config()
    bundle = load_default_bundle(config)
    table = run_survey(config, bundle)
    first, last = str(bundle.index[0].date()), str(bundle.index[-1].date())
    out = Path(config.root) / "docs" / "strategy_survey.md"
    out.write_text(render(table, first, last, len(bundle.index) / 252.0), encoding="utf-8")
    (Path(config.root) / "reports" / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(config.root) / "reports" / "tables" / "library_survey.csv", index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
