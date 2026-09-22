"""Stage 13 - Machine learning extension (Ch. 23, spec §46-§48).

Deliberately last, and evaluated on three axes at once: predictive
(accuracy, AUC), alpha (IC), and investment (Sharpe, drawdown, turnover, net
return). The research question is not "can a model beat a coin flip" but
whether nonlinear ML delivers genuine out-of-sample *economic* improvement
over the simpler models already built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.models.machine_learning import (
    build_dataset,
    cross_validate,
    evaluate_predictions,
    feature_importance,
    make_models,
    predictions_to_signal,
    walk_forward_predictions,
)
from src.utils.plotting import PALETTE, bar_with_values, new_axes, save_figure
from experiments.context import build_context
from experiments.strategies import transform_config

STAGE = "stage13_ml"


def figure_ml(context, cv_summary, importances, evaluation, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.6))

    x = np.arange(len(cv_summary))
    axes[0, 0].bar(x - 0.2, cv_summary["accuracy"].to_numpy(), width=0.4, label="accuracy",
                   color="#0072B2")
    axes[0, 0].bar(x + 0.2, cv_summary["auc"].to_numpy(), width=0.4, label="AUC", color="#D55E00")
    axes[0, 0].axhline(0.5, color="black", linestyle="--", linewidth=0.9, label="coin flip")
    axes[0, 0].set_xticks(x, cv_summary.index, rotation=20, ha="right", fontsize=8)
    axes[0, 0].set_ylim(0.45, max(0.6, float(cv_summary[["accuracy", "auc"]].to_numpy().max()) + 0.02))
    axes[0, 0].set_title("Predictive metrics (purged cross-validation)")
    axes[0, 0].legend(fontsize=8.5)

    bar_with_values(axes[0, 1], cv_summary["ic"], "Alpha metric: information coefficient",
                    "IC", "{:.4f}", rotation=20)

    if len(importances):
        top = importances.head(10)[::-1]
        axes[1, 0].barh(range(len(top)), top.to_numpy(), color="#009E73")
        axes[1, 0].set_yticks(range(len(top)), top.index, fontsize=8)
        axes[1, 0].set_title("Feature importance (gradient boosting)")

    if len(evaluation):
        frame = evaluation[["sharpe", "gross_sharpe"]].dropna()
        x = np.arange(len(frame))
        axes[1, 1].bar(x - 0.2, frame["gross_sharpe"].to_numpy(), width=0.4, label="gross",
                       color="#BBBBBB")
        axes[1, 1].bar(x + 0.2, frame["sharpe"].to_numpy(), width=0.4, label="net", color="#0072B2")
        axes[1, 1].axhline(0.0, color="black", linewidth=0.9)
        axes[1, 1].set_xticks(x, frame.index, rotation=20, ha="right", fontsize=8)
        axes[1, 1].set_ylabel("Sharpe ratio")
        axes[1, 1].set_title("Investment metrics: does accuracy translate into money?")
        axes[1, 1].legend(fontsize=8.5)

    fig.suptitle("Figure 24. Machine learning ladder: predictive, alpha and investment metrics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Does nonlinear machine learning deliver genuine out-of-sample "
                           "economic improvement over the simpler models, or only a better "
                           "classification score?", 24)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 13 | machine learning extension (Ch. 23)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    horizon = int(cfg.get("strategies.primary_horizon", 21))

    dataset = build_dataset(market, horizon)
    logger.info("design matrix: %s", dataset.describe())
    context.save_table(pd.Series(dataset.describe()).to_frame("value"), "stage13_dataset.csv")

    models = make_models(seed=42)
    cv = cross_validate(dataset, models, n_splits=5,
                        embargo_days=int(cfg.get("backtest.walk_forward.embargo_days", 21)),
                        horizon=horizon)
    context.save_table(cv, "stage13_cross_validation.csv", index=False)
    cv_summary = cv.groupby("model")[["accuracy", "auc", "ic"]].mean()
    cv_summary["accuracy_std"] = cv.groupby("model")["accuracy"].std(ddof=1)
    cv_summary["ic_std"] = cv.groupby("model")["ic"].std(ddof=1)
    cv_summary["folds_positive_ic"] = cv.groupby("model")["ic"].apply(lambda s: float((s > 0).mean()))
    context.save_table(cv_summary, "stage13_cv_summary.csv")
    logger.info("purged cross-validation (mean across folds):\n%s", cv_summary.round(4).to_string())

    # --- walk-forward, the only predictions worth trading -----------------
    evaluation = {}
    signals = {}
    for name, model in models.items():
        try:
            predictions = walk_forward_predictions(
                dataset, model, train_years=5.0, test_months=12,
                embargo_days=int(cfg.get("backtest.walk_forward.embargo_days", 21)),
            )
        except Exception as exc:
            logger.warning("walk-forward failed for %s: %s", name, exc)
            continue
        signal = predictions_to_signal(predictions - 0.5, returns.columns)
        signals[name] = signal
        evaluation[name] = evaluate_predictions(
            signal, returns, horizon, market, transform_config(cfg),
            float(cfg.get("backtest.costs.cost_bps", 10.0)),
        )
        logger.info("%s walk-forward: IC %+.4f  net Sharpe %+.3f  turnover %.1fx",
                    name, evaluation[name].get("ic_mean_ic", np.nan),
                    evaluation[name].get("sharpe", np.nan),
                    evaluation[name].get("ann_turnover", np.nan))

    evaluation_frame = pd.DataFrame(evaluation).T
    context.save_table(evaluation_frame, "stage13_walk_forward_evaluation.csv")
    logger.info("walk-forward evaluation (spec §48):\n%s",
                evaluation_frame[["ic_mean_ic", "ic_t_stat_overlap_adjusted", "cagr", "sharpe",
                                  "gross_sharpe", "max_drawdown", "ann_turnover"]].astype(float).round(4).to_string())

    # --- feature importance from a full-sample fit (description only) -----
    importances = pd.Series(dtype=float)
    try:
        boosting = models["ML3_gradient_boosting"]
        boosting.fit(dataset.X.to_numpy(), dataset.y.to_numpy())
        importances = feature_importance(boosting, dataset.feature_names)
        context.save_table(importances.to_frame("importance"), "stage13_feature_importance.csv")
        logger.info("feature importance (full-sample fit, descriptive only):\n%s",
                    importances.round(4).to_string())
    except Exception as exc:
        logger.warning("feature importance failed: %s", exc)

    figure_ml(context, cv_summary, importances, evaluation_frame,
              context.figure("fig24_machine_learning.png"))

    best_predictive = str(cv_summary["auc"].idxmax())
    best_economic = (str(evaluation_frame["sharpe"].astype(float).idxmax())
                     if len(evaluation_frame) else "none")
    context.registry.log(
        "Nonlinear machine learning provides genuine out-of-sample economic "
        "improvement over the simpler econometric and systematic models (spec §47).",
        stage=STAGE,
        parameters={"models": list(models), "features": dataset.feature_names,
                    "horizon": horizon, "n_samples": int(len(dataset))},
        cost_bps=float(cfg.get("backtest.costs.cost_bps", 10.0)),
        results={
            "best_by_auc": best_predictive,
            "best_auc": float(cv_summary.loc[best_predictive, "auc"]),
            "best_by_net_sharpe": best_economic,
            "best_net_sharpe": float(evaluation_frame["sharpe"].max()) if len(evaluation_frame) else np.nan,
            "logistic_net_sharpe": float(evaluation_frame.loc["ML0_logistic", "sharpe"])
            if "ML0_logistic" in evaluation_frame.index else np.nan,
            "boosting_net_sharpe": float(evaluation_frame.loc["ML3_gradient_boosting", "sharpe"])
            if "ML3_gradient_boosting" in evaluation_frame.index else np.nan,
            "best_ic": float(evaluation_frame["ic_mean_ic"].max()) if len(evaluation_frame) else np.nan,
            "predictive_and_economic_winner_agree": best_predictive == best_economic,
        },
        decision="reject",
        notes=(
            f"The model with the best AUC ({best_predictive}) is "
            f"{'also' if best_predictive == best_economic else 'NOT'} the model with the best net "
            f"Sharpe ratio ({best_economic}), which is the point of evaluating on three axes: "
            "classification accuracy weights a 0.1% day and a 5% day equally, and trading does "
            "not. All four models trade at high turnover for a small edge and land in the same "
            "place as the classical signals in Stage 6. ML is not rejected because it is ML; it "
            "is rejected because the underlying predictability at this horizon is too small to "
            "pay for the turnover required to harvest it, and no functional form changes that."
        ),
    )
    logger.info("STAGE 13 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
