"""Run the whole research pipeline, in the order the specification fixes.

    python -m experiments.run_all                    # everything
    python -m experiments.run_all --from 6 --to 9    # a range of stages
    python -m experiments.run_all --only 1 --download --fresh

Each stage is independent and writes its own tables, figures and registry
entries, so a stage can be re-run on its own after a change. ``--fresh``
clears the derived artefacts first, which is what makes the "reproducible
from a commit hash plus a config fingerprint" claim testable rather than
aspirational.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import time

from src.utils.config import load_config
from src.utils.logging import stage_logger

# (number, module, description, generation)
STAGES = [
    (1, "stage01_data", "Financial data infrastructure (Ch. 7)", 1),
    (2, "stage02_eda", "Exploratory analysis and PCA (Ch. 8)", 1),
    (3, "stage03_momentum", "Momentum alpha research (Ch. 22)", 1),
    (4, "stage04_mean_reversion", "Mean-reversion alpha research (Ch. 22)", 1),
    (5, "stage05_expected_returns", "Expected returns and IC (Ch. 20)", 1),
    (6, "stage06_backtest", "Backtesting the alpha models (Ch. 22)", 1),
    (7, "stage07_portfolio", "Portfolio construction (Ch. 19)", 1),
    (8, "stage08_covariance", "Volatility and covariance (Ch. 20 §20.2)", 1),
    (9, "stage09_risk", "Risk management (Ch. 21)", 1),
    (10, "stage10_combination", "Combining strategies (Ch. 22 §22.5)", 1),
    (11, "stage11_validation", "Walk-forward, robustness and leakage", 1),
    (12, "stage12_results", "Final comparison and results", 1),
    (13, "stage13_ml", "Machine learning extension (Ch. 23)", 1),
    # A separate research branch (pairs / PCA stat-arb). It is part of the full
    # run so that every table the report cites can be regenerated.
    (14, "stage14_extensions", "Pairs trading and PCA stat-arb (Ch. 22 §22.3.3-7)", 1),
    # Generation 2. Numbered by build order; the dependencies are: macro (15)
    # feeds regimes (16) and probabilistic forecasting (19).
    (15, "stage15_macro", "Macro features and predictability", 2),
    (16, "stage16_regimes", "Regime detection", 2),
    (17, "stage17_dynamic_covariance", "Dynamic covariance (DCC, O-GARCH)", 2),
    (18, "stage18_hierarchical", "Hierarchical risk parity (HRP, HERC)", 2),
    (19, "stage19_probabilistic", "Probabilistic forecasting and sizing", 2),
    (20, "stage20_attribution", "Portfolio attribution", 2),
    # Generation 3. Build order again; 24 combines the alphas of Gen 1-2 with 22's online
    # weights and 23's non-price features, and 25 re-costs every book.
    (21, "stage21_bayesian", "Bayesian portfolio construction", 3),
    (22, "stage22_online", "Online learning", 3),
    (23, "stage23_altdata", "Non-price data", 3),
    (24, "stage24_combination", "Alpha-combination engine", 3),
    (25, "stage25_execution", "Execution model", 3),
    # Generation 4. The research database runs last because it ingests every other stage's output.
    (26, "stage26_distributed", "Distributed experimentation", 4),
    (27, "stage27_deep", "Deep learning", 4),
    (28, "stage28_text", "Text features and research assistant", 4),
    # Generation 5. The research database (stage 29) is last in execution order because it ingests every other stage's output.
    (30, "stage30_library", "The strategy library as a search", 5),
    (31, "stage31_adaptive", "The adaptive pipeline", 5),
    (32, "stage32_crypto", "The optional crypto branch", 5),
    (33, "stage33_frontier", "Deep-forecasting family, graph and Bayesian deep learning", 5),
    (34, "stage34_foundation", "Zero-shot foundation model (descriptive)", 5),
    (35, "stage35_explain", "Explainability and calibration", 5),
    (36, "stage36_diffusion", "Diffusion scenarios for tail risk", 5),
    (37, "stage37_causal", "Causal inference", 5),
    (38, "stage38_rl", "Reinforcement-learning allocation", 5),
    (39, "stage39_power", "Statistical power of the platform tests", 5),
    (29, "stage29_research_db", "Research database", 4),

]
LAST_STAGE = max(number for number, *_ in STAGES)


def clear_derived(config, logger, full: bool) -> None:
    """Delete rebuilt artefacts, never the immutable raw data.

    ``full=True`` (the whole pipeline is about to run) clears processed data,
    caches, every figure and table, and the registry. With a *partial* run
    that would silently delete the outputs of every stage that is not about to
    be re-run, so only the processed data and caches are cleared and the
    reports, tables and registry are left for the selected stages to
    overwrite.
    """
    targets = [config.path("processed"), config.path("features")]
    if full:
        targets += [config.reports_dir("figures"), config.reports_dir("tables"),
                    config.root / "experiments" / "registry.jsonl"]
    else:
        logger.warning("partial run: clearing processed data and caches only; reports, "
                       "tables and the experiment registry are kept")
    for target in targets:
        if target.is_dir():
            for path in target.iterdir():
                if path.name == ".gitkeep":
                    continue
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            logger.info("cleared %s", target)
        elif target.exists():
            target.unlink()
            logger.info("removed %s", target)
    logger.info("raw data left untouched: %s", config.path("raw"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the research pipeline")
    parser.add_argument("--from", dest="start", type=int, default=1, help="first stage")
    parser.add_argument("--to", dest="end", type=int, default=LAST_STAGE, help="last stage")
    parser.add_argument("--only", type=int, nargs="*", help="run only these stages")
    parser.add_argument("--generation", type=int, choices=(1, 2, 3, 4, 5),
                        help="run only the stages of one generation")
    parser.add_argument("--download", action="store_true", help="fetch raw data in stage 1")
    parser.add_argument("--fresh", action="store_true",
                        help="clear derived artefacts first (the whole pipeline: everything; "
                             "a partial run: processed data and caches only)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep going if a stage fails")
    args = parser.parse_args(argv)

    config = load_config()
    logger = stage_logger("run_all", config.root)
    logger.info("#" * 78)
    logger.info("SYSTEMATIC MULTI-ASSET RESEARCH PIPELINE")
    logger.info("config fingerprint: %s", config.fingerprint())
    logger.info("#" * 78)

    if args.only:
        selected = [s for s in STAGES if s[0] in set(args.only)]
    else:
        selected = [s for s in STAGES if args.start <= s[0] <= args.end]
        if args.generation:
            selected = [s for s in selected if s[3] == args.generation]
    if not selected:
        logger.error("no stages selected")
        return 1

    if args.fresh:
        clear_derived(config, logger, full=len(selected) == len(STAGES))

    results, started = [], time.time()
    for number, module_name, description, _generation in selected:
        logger.info("")
        logger.info("=" * 78)
        logger.info("STAGE %d | %s", number, description)
        logger.info("=" * 78)
        stage_started = time.time()
        try:
            module = importlib.import_module(f"experiments.{module_name}")
            argv_stage = ["--download"] if (number == 1 and args.download) else []
            code = module.main(argv_stage)
            elapsed = time.time() - stage_started
            status = "ok" if code == 0 else f"exit {code}"
            results.append((number, module_name, status, elapsed))
        except Exception as exc:
            elapsed = time.time() - stage_started
            logger.exception("stage %d failed: %s", number, exc)
            results.append((number, module_name, f"FAILED: {exc}", elapsed))
            if not args.continue_on_error:
                break

    # The research database ingests every stage's output, so after any stage numbered above 29 it is rebuilt even when it was not selected.
    ran = {number for number, _, status, _ in results if not status.startswith("FAILED")}
    if ran and max(ran) > 29 and 29 not in {s[0] for s in selected} and not any(r[2].startswith("FAILED") for r in results):
        started_db = time.time()
        module = importlib.import_module("experiments.stage29_research_db")
        code = module.main([])
        results.append((29, "stage29_research_db", "ok" if code == 0 else f"exit {code}", time.time() - started_db))

    logger.info("")
    logger.info("=" * 78)
    logger.info("PIPELINE SUMMARY (%.1f minutes total)", (time.time() - started) / 60.0)
    logger.info("=" * 78)
    for number, name, status, elapsed in results:
        logger.info("  stage %2d  %-26s %-12s %6.1fs", number, name, status, elapsed)

    # The index is built by scanning caption files, so it is complete after any combination of stages.
    from src.utils.plotting import write_figure_index
    write_figure_index(config.reports_dir() / "figure_index.md")
    from src.utils.experiments import ExperimentRegistry
    ExperimentRegistry(config.root / "experiments" / "registry.jsonl").to_markdown(config.root / "experiments" / "registry.md")

    failed = [r for r in results if r[2].startswith("FAILED")]
    if failed:
        logger.error("%d stage(s) failed", len(failed))
        return 1
    logger.info("all stages completed. Reports in %s", config.reports_dir())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
