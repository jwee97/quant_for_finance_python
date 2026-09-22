"""Stage 7 - Portfolio construction (Ch. 19).

Runs the risk-based ladder (M0-M2) and the optimised books (M6-M9) through the
same engine, and runs the estimation-error experiment that explains why the
simple models do as well as they do.

Figures 16-17: portfolio weights, risk contributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, buy_and_hold
from src.portfolio.constraints import Constraints
from src.portfolio.covariance import estimate_covariance
from src.portfolio.mean_variance import (
    efficient_frontier,
    estimation_error_experiment,
    minimum_variance_weights,
    resampled_weights,
    robust_weights,
)
from src.portfolio.risk_parity import risk_contribution_frame, risk_concentration
from src.risk.contributions import diversification_ratio, factor_risk_contributions
from src.utils.plotting import ASSET_CLASS_COLOURS, PALETTE, bar_with_values, new_axes, plot_heatmap, save_figure
from experiments.context import build_context
from experiments.strategies import cached_ladder

STAGE = "stage07_portfolio"
RISK_MODELS = ["M0_equal_weight", "M1_inverse_vol", "M2_risk_parity"]
OPTIMISED = ["M6_combined_alpha_mvo", "M7_combined_alpha_shrinkage_mvo",
             "M8_black_litterman", "M9_mean_cvar"]


def figure_weights(context, books, asset_class, path):
    selected = [n for n in ["M0_equal_weight", "M1_inverse_vol", "M2_risk_parity",
                            "M7_combined_alpha_shrinkage_mvo"] if n in books]
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.4), sharex=True)
    for ax, name in zip(axes.ravel(), selected):
        frame = books[name].resample("ME").last().dropna(how="all")
        frame = frame.loc[frame.abs().sum(axis=1) > 1e-9]
        colours = [ASSET_CLASS_COLOURS.get(asset_class.get(c, ""), PALETTE[i % len(PALETTE)])
                   for i, c in enumerate(frame.columns)]
        ax.stackplot(frame.index, frame.T.to_numpy(), colors=colours, labels=list(frame.columns))
        ax.set_title(name.replace("_", " "))
        ax.set_ylabel("Weight")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=8, loc="lower center", fontsize=7.5, frameon=False)
    fig.suptitle("Figure 16. Portfolio weights through time", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, path, "What does each allocation method actually hold, and is the book "
                           "stable or does it churn?", 16)


def figure_risk_contributions(context, books, returns, covariance, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.4))
    selected = [n for n in RISK_MODELS if n in books]

    frames = {}
    for name in selected:
        weights = books[name].iloc[-1]
        weights = weights[weights.abs() > 1e-9]
        assets = [a for a in weights.index if a in covariance.columns]
        frames[name] = risk_contribution_frame(weights[assets], covariance.loc[assets, assets])

    shares = pd.DataFrame({n: f["risk_share"] for n, f in frames.items()}).fillna(0.0)
    x = np.arange(len(shares))
    width = 0.8 / max(len(shares.columns), 1)
    for i, name in enumerate(shares.columns):
        axes[0, 0].bar(x + i * width, shares[name].to_numpy(), width=width,
                       label=name.replace("_", " "), color=PALETTE[i])
    axes[0, 0].axhline(1.0 / len(shares), color="black", linestyle="--", linewidth=0.9,
                       label="equal risk (1/N)")
    axes[0, 0].set_xticks(x + width, shares.index, rotation=90)
    axes[0, 0].set_ylabel("Share of portfolio risk")
    axes[0, 0].set_title("Risk contributions on the latest date")
    axes[0, 0].legend(fontsize=8)

    weights_frame = pd.DataFrame({n: books[n].iloc[-1] for n in selected}).fillna(0.0)
    plot_heatmap(axes[0, 1], weights_frame.T, "Latest weights", vmin=0.0, vmax=0.35,
                 cmap="Blues", fmt="{:.2f}")

    metrics = pd.DataFrame(
        {
            name: {
                "risk_concentration": risk_concentration(
                    books[name].iloc[-1].reindex(covariance.columns).fillna(0.0).to_numpy(),
                    covariance.to_numpy()),
                "diversification_ratio": diversification_ratio(
                    books[name].iloc[-1].reindex(covariance.columns).fillna(0.0), covariance),
                "effective_n": float(1.0 / (books[name].iloc[-1] ** 2).sum()),
            }
            for name in selected
        }
    ).T
    bar_with_values(axes[1, 0], metrics["diversification_ratio"],
                    "Diversification ratio (higher is better)", "ratio", "{:.2f}", rotation=20)
    bar_with_values(axes[1, 1], metrics["effective_n"], "Effective number of positions",
                    "count", "{:.1f}", rotation=20)

    fig.suptitle("Figure 17. Risk contributions and diversification by construction method",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Does risk parity actually equalise risk contributions, and does "
                           "holding 15 assets deliver 15 assets' worth of diversification?", 17)
    return metrics


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 7 | portfolio construction (Ch. 19)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    engine = BacktestEngine.from_config(cfg)
    constraints = Constraints.from_config(cfg)
    feasible, reason = constraints.is_feasible(market.tickers)
    logger.info("constraint set feasible: %s (%s)", feasible, reason)

    books = cached_ladder(market, cfg, context.processed, RISK_MODELS + OPTIMISED)
    results = {name: engine.run(weights, returns, name, market.investable, apply_vol_target=False)
               for name, weights in books.items()}
    benchmark = buy_and_hold(returns, "SPY").net_returns
    results["SPY_buy_hold"] = buy_and_hold(returns, "SPY")

    summary = pd.DataFrame({n: r.summary(benchmark=benchmark) for n, r in results.items()}).T
    context.save_table(summary, "stage07_portfolio_backtests.csv")
    logger.info("portfolio model comparison:\n%s",
                summary[["cagr", "ann_vol", "sharpe", "sortino", "max_drawdown",
                         "ann_turnover"]].astype(float).round(4).to_string())

    # --- estimation error (spec §30) -------------------------------------
    window = returns.dropna(how="any").tail(int(cfg.get("portfolio.covariance.lookback", 252)))
    covariance = estimate_covariance(window, "shrinkage", len(window), annualise=True)
    mu = window.mean() * 252
    for perturbation in (0.10, 0.25):
        experiment = estimation_error_experiment(
            mu, covariance, float(cfg.get("portfolio.mean_variance.risk_aversion", 5.0)),
            perturbation, 150, constraints,
        )
        context.save_table(experiment, f"stage07_estimation_error_{int(perturbation * 100)}pct.csv")
        logger.info("estimation-error experiment, perturbation %.0f%%:\n%s",
                    100 * perturbation, experiment.round(4).to_string())
        if perturbation == 0.25:
            headline = experiment

    frontier = efficient_frontier(mu, covariance, 25, constraints)
    context.save_table(frontier, "stage07_efficient_frontier.csv", index=False)

    # --- extensions: resampling and robust optimisation ------------------
    extensions = {}
    try:
        extensions["resampled"] = resampled_weights(mu, window, 5.0, 60, constraints)
    except Exception as exc:
        logger.warning("resampled optimisation failed: %s", exc)
    try:
        extensions["robust"] = robust_weights(mu, covariance, None, 5.0, 1.0, constraints).weights
    except Exception as exc:
        logger.warning("robust optimisation failed: %s", exc)
    extensions["minimum_variance"] = minimum_variance_weights(covariance, constraints).weights
    if extensions:
        frame = pd.DataFrame(extensions)
        context.save_table(frame, "stage07_optimiser_extensions.csv")
        logger.info("optimiser variants (latest window):\n%s", frame.round(4).to_string())

    figure_weights(context, books, cfg.asset_class_map, context.figure("fig16_portfolio_weights.png"))
    metrics = figure_risk_contributions(context, books, returns, covariance,
                                        context.figure("fig17_risk_contributions.png"))
    context.save_table(metrics, "stage07_diversification_metrics.csv")

    factor = factor_risk_contributions(books["M0_equal_weight"].iloc[-1], returns)
    context.save_table(factor, "stage07_factor_risk_contributions.csv")
    logger.info("equal-weight book, risk attributed to principal components:\n%s",
                factor.round(4).to_string())

    best = summary["sharpe"].astype(float).idxmax()
    context.registry.log(
        "Sophisticated portfolio optimisation improves risk-adjusted returns over "
        "naive equal weighting.",
        stage=STAGE,
        parameters={"models": list(books), "risk_aversion": cfg.get("portfolio.mean_variance.risk_aversion"),
                    "constraints": {"max_weight": constraints.max_weight,
                                    "group_limits": constraints.group_limits}},
        results={
            "best_model": best,
            "best_sharpe": float(summary.loc[best, "sharpe"]),
            "equal_weight_sharpe": float(summary.loc["M0_equal_weight", "sharpe"]),
            "inverse_vol_sharpe": float(summary.loc["M1_inverse_vol", "sharpe"]),
            "risk_parity_sharpe": float(summary.loc["M2_risk_parity", "sharpe"]),
            "mvo_shrinkage_sharpe": float(summary.loc["M7_combined_alpha_shrinkage_mvo", "sharpe"]),
            "spy_sharpe": float(summary.loc["SPY_buy_hold", "sharpe"]),
        },
        decision="investigate",
        notes=(
            f"The best model on full-sample net Sharpe is {best}. The risk-based allocators, "
            "which use no expected returns at all, beat every book that requires estimating mu. "
            "The estimation-error experiment explains why: an unconstrained optimiser holds "
            f"{headline.loc['unconstrained', 'baseline_n_holdings']:.0f} of 15 assets with a "
            f"{headline.loc['unconstrained', 'baseline_max_weight']:.0%} maximum weight, and a "
            "perturbation of mu worth 25% of its cross-sectional dispersion moves up to "
            f"{headline.loc['unconstrained', 'max_turnover_from_noise']:.0%} of the book. "
            "This is a full-sample comparison and therefore in-sample; Stage 11 re-runs it "
            "walk-forward before any conclusion is drawn."
        ),
    )
    logger.info("STAGE 7 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
