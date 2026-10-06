"""Stage 28 - Text features and the research assistant (Generation 4, Priority 16).

FOMC statements (2006-2026) are turned into three structured, point-in-time features (tone, change from the previous
statement, announced rate action) and tested the way Stages 15 and 23 tested macro and non-price data: a nested
Clark-West test of price + macro + text against price + macro for each of five sleeves. Rules: ``config/assistant.yaml``.

NO LANGUAGE MODEL IS CALLED HERE. The features come from the offline lexicon backend, the baseline any model-based
extraction has to beat. The Anthropic backend and the text-to-SQL assistant are implemented and tested against fake
clients (``tests/test_assistant.py``) but were not run: this environment has no model credentials.

Figures 56-57.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.assistant.documents import corpus_version, ensure_fomc_corpus
from src.assistant.extraction import ExtractionCache, LexiconBackend, extract_corpus, schema_hash
from src.data.macro import ensure_macro_raw
from src.features.macro import expanding_zscore, macro_feature_panel
from src.features.sleeves import month_end_dates, monthly_compound, sleeve_returns
from src.models.macro_forecast import build_monthly_panel, evaluate_nested, walk_forward_forecasts
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg
from experiments.context import build_context

STAGE = "stage28_text"


def text_features(features: pd.DataFrame, burn_in: int) -> pd.DataFrame:
    """Expanding z-scores over statements (burn-in ``burn_in``), indexed by the date each statement became available."""
    f = features.set_index("available_at").sort_index()
    action_total = f["action"].rolling(4, min_periods=1).sum()
    raw = pd.DataFrame({"tone": f["tone"], "change": f["change"], "action": action_total})
    mean, std = raw.expanding(min_periods=burn_in).mean(), raw.expanding(min_periods=burn_in).std(ddof=1)
    return ((raw - mean) / std.replace(0.0, np.nan)).clip(-4.0, 4.0)


def asof_month_ends(z: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """The latest value known at each month-end: ``available_at <= month_end``."""
    return z.reindex(z.index.union(month_ends)).ffill().reindex(month_ends)


def figure_corpus(features: pd.DataFrame, z: pd.DataFrame, per_year: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ax.bar(per_year.index, per_year["statements"], color=PALETTE[0])
    ax.set_ylabel("FOMC statements in the corpus")
    ax.set_title(f"Corpus: {int(per_year['statements'].sum())} statements")
    ax = axes[1]
    ax.plot(features["date"], features["tone"], color=PALETTE[1], linewidth=1.2)
    up, down = features[features["action"] > 0], features[features["action"] < 0]
    ax.scatter(up["date"], up["tone"], color="#CC0000", s=22, zorder=3, label="read as a rate increase")
    ax.scatter(down["date"], down["tone"], color="#0072B2", s=22, zorder=3, label="read as a rate cut")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Tone: (hawkish - dovish terms) per 1,000 words")
    ax.set_title("What the lexicon reads in each statement")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(features["date"], features["change"], color=PALETTE[2], linewidth=1.2)
    ax.set_ylabel("Change from the previous statement (1 - tf-idf cosine)")
    ax.set_title("How much each statement rewrites the last")
    fig.suptitle("Figure 56. The FOMC statement corpus and its offline features", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "What does the FOMC statement corpus look like through the offline lexicon: how many statements, what tone, and how much "
                           "does each statement rewrite the last?", 56)


def figure_results(declared: pd.DataFrame, matched: pd.DataFrame, corr: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    x = np.arange(len(declared))
    ax.bar(x - 0.2, 100 * declared["oos_r2"], width=0.4, color=PALETTE[0], label="declared (restricted model: own history)")
    ax.bar(x + 0.2, 100 * matched["oos_r2"], width=0.4, color=PALETTE[3], label="matched-sample control (post-hoc)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, declared["sleeve"])
    ax.set_ylabel("Out-of-sample R-squared of adding text, %")
    ax.set_title("Does text add to price + macro?")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(x, declared["p_value"], color=PALETTE[0])
    ax.axhline(0.10, color="#CC0000", linestyle="--", linewidth=1.0)
    for i, p in enumerate(declared["p_value"]):
        ax.text(i, p, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, declared["sleeve"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Clark-West p-value (one-sided)")
    ax.set_title("Per sleeve (dashed = 10%, before BH)")
    ax = axes[2]
    im = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(corr.shape[1]), corr.columns)
    ax.set_yticks(range(corr.shape[0]), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Rank correlation with next month's excess return")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Figure 57. Does the text carry information beyond price and macro?", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does tone, change or the announced rate action in FOMC statements add out-of-sample forecasting information to price and "
                           "macro features for the five sleeves, and does the answer depend on how the restricted model is trained?", 57)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 28: text features").parse_args(argv)
    context, logger = build_context(STAGE, generation=4)
    cfg = context.config
    node = cfg.get("assistant", {}) or {}
    logger.info("=" * 72)
    logger.info("STAGE 28 | text features and the research assistant (Generation 4, Priority 16)")
    logger.info("=" * 72)
    fdr = float((node.get("decisions", {}) or {}).get("fdr", 0.10))
    market = context.market_data()
    returns = market.returns()
    index = pd.DatetimeIndex(market.prices.index)

    text_dir = cfg.root / "data" / "raw" / "text" / "fomc"
    corpus = ensure_fomc_corpus(text_dir, 2006, index.max().year)
    version = corpus_version(text_dir)
    backend = LexiconBackend.from_config(node["lexicon_backend"])
    cache = ExtractionCache(context.processed / "extraction_cache.jsonl")
    features = extract_corpus(corpus, backend, cache)
    logger.info("corpus %s: %d statements from %s to %s; backend %s; schema %s", version, len(corpus), corpus["date"].min().date(), corpus["date"].max().date(),
                backend.identity(), schema_hash())
    per_year = features.groupby(features["date"].dt.year).agg(statements=("tone", "size"), mean_tone=("tone", "mean"), mean_change=("change", "mean"),
                                                             raises=("action", lambda s: int((s > 0).sum())), cuts=("action", lambda s: int((s < 0).sum())))
    context.save_table(features, "stage28_features_by_statement.csv", index=False)
    context.save_table(per_year, "stage28_corpus_by_year.csv")
    logger.info("by year:\n%s", per_year.round(2).to_string())

    # agreement of the lexicon's action reading with the effective federal funds rate (reported, not judged)
    macro_specs, macro_raw = ensure_macro_raw(cfg)
    dff = macro_raw["DFF"].sort_index()
    after = pd.Series({d: dff.loc[d + pd.Timedelta(days=1):d + pd.Timedelta(days=10)].mean() for d in features["date"]})
    before = pd.Series({d: dff.loc[d - pd.Timedelta(days=10):d - pd.Timedelta(days=1)].mean() for d in features["date"]})
    realised = np.sign((after - before).where((after - before).abs() > 0.10, 0.0))
    agreement = pd.DataFrame({"lexicon": features.set_index("date")["action"], "realised": realised}).dropna()
    confusion = pd.crosstab(agreement["realised"], agreement["lexicon"])
    rate = float((agreement["lexicon"] == agreement["realised"]).mean())
    context.save_table(confusion, "stage28_action_agreement.csv")
    logger.info("lexicon action vs the effective funds rate over the 10 days either side (rows realised, columns lexicon):\n%s\nagreement %.1f%% of %d statements",
                confusion.to_string(), 100 * rate, len(agreement))

    # ----------------------------------------------------------- nested test
    pred = cfg.get("macro.predictability", {}) or {}
    sleeves = {k: list(v) for k, v in (pred.get("sleeves", {}) or {}).items()}
    cash = str(pred.get("cash", "SHY"))
    macro_z = macro_feature_panel(macro_raw, macro_specs, index, cfg)["z"]
    monthly = build_monthly_panel(sleeve_returns(returns, sleeves, market.investable), returns[cash], macro_z)
    dates = monthly.dates
    z = text_features(features, int((node.get("features", {}) or {}).get("burn_in_statements", 16)))
    block = asof_month_ends(z, dates)
    mask = block.where(block.isna(), 0.0)
    logger.info("first month with a full text block: %s", block.dropna().index[0].date())
    kwargs = dict(min_train_months=int(pred.get("min_train_months", 60)), alphas=list(pred.get("ridge_alphas", [1, 10, 100, 1000])),
                  cv_splits=int(pred.get("cv_splits", 5)), first_test_year=int(pred.get("first_test_year", 2010)))
    result = walk_forward_forecasts(monthly, models=("both", "all", "both_matched"), extra_blocks={"text": block, "mask": mask},
                                    extra_models={"all": ("price", "macro", "text"), "both_matched": ("price", "macro", "mask")}, **kwargs)
    declared = evaluate_nested(result, [("both", "all")], fdr)
    matched = evaluate_nested(result, [("both_matched", "all")], fdr)
    context.save_table(declared, "stage28_clark_west_declared.csv", index=False)
    context.save_table(matched, "stage28_clark_west_matched_posthoc.csv", index=False)
    cols = ["sleeve", "oos_r2", "statistic", "p_value", "bh_significant", "n_obs"]
    logger.info("Clark-West, price+macro+text vs price+macro (declared):\n%s", declared[cols].round(4).to_string(index=False))
    logger.info("matched-sample control (post-hoc):\n%s", matched[cols].round(4).to_string(index=False))

    # reported, not judged: rank correlations with next month's excess return
    excess = monthly_compound(returns).sub(monthly_compound(returns[cash]), axis=0)
    month_ends = month_end_dates(index)
    forward = excess.reindex(month_ends).shift(-1)
    zm = asof_month_ends(z, month_ends)
    targets = {"SPY": "SPY", "IEF": "IEF", "TLT": "TLT", "HYG": "HYG", "GLD": "GLD"}
    corr = pd.DataFrame(index=z.columns, columns=list(targets), dtype=float)
    pvals = corr.copy()
    for feature in z.columns:
        for label, etf in targets.items():
            pair = pd.concat([zm[feature], forward[etf]], axis=1).dropna()
            if len(pair) >= 36:
                corr.loc[feature, label], pvals.loc[feature, label] = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    context.save_table(corr, "stage28_feature_rank_correlations.csv")
    context.save_table(pvals, "stage28_feature_rank_correlation_pvalues_posthoc.csv")
    logger.info("rank correlations with next month's excess return:\n%s", corr.round(3).to_string())

    figure_corpus(features, z, per_year, context.figure("fig56_text_corpus.png"))
    figure_results(declared, matched, corr, context.figure("fig57_text_results.png"))

    adds = bool(declared["bh_significant"].any() and declared["oos_r2"].mean() > 0)
    context.registry.log(
        "Tone, change and the announced rate action of FOMC statements (offline lexicon features) add out-of-sample forecasting information to price "
        "and macro features for monthly sleeve returns.",
        stage=STAGE, parameters={"backend": backend.identity(), "schema": schema_hash(), "corpus_version": version, "statements": int(len(corpus)),
                                 "fdr": fdr, "min_train_months": kwargs["min_train_months"], "no_language_model_called": True},
        results={"sleeves_bh_significant": int(declared["bh_significant"].sum()), "sleeves_raw_significant": int((declared["p_value"] < 0.05).sum()),
                 "mean_oos_r2": float(declared["oos_r2"].mean()), "min_p_value": float(declared["p_value"].min()),
                 "oos_months_min": int(declared["n_obs"].min()), "posthoc_matched_sleeves_bh_significant": int(matched["bh_significant"].sum()),
                 "posthoc_matched_mean_oos_r2": float(matched["oos_r2"].mean()), "lexicon_action_agreement": rate, "statements": float(len(features))},
        decision="retain" if adds else "reject", test_period="walk-forward monthly, first test year 2010",
        notes="Retained if at least one sleeve passes Clark-West at FDR 10% AND the mean out-of-sample R-squared is positive. Features are from the offline "
              "lexicon backend; no language model was called. The matched-sample control is post-hoc. Statements are available the day after their date.",
    )
    logger.info("STAGE 28 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
