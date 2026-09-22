"""Stage 8 - Volatility and covariance modelling (Ch. 20 §20.2).

Answers spec §33: which covariance method produces the most stable and useful
out-of-sample risk forecasts? Judged by forecasting realised portfolio risk,
not by mathematical elegance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.volatility import (
    ewma_volatility,
    evaluate_forecasts,
    forward_realised_volatility,
    garch_parameters,
    garch_rolling_forecast,
    parkinson_volatility,
    rolling_volatility,
)
from src.portfolio.covariance import (
    condition_number,
    estimate_covariance,
    evaluate_covariance_forecasts,
    ledoit_wolf_shrinkage,
)
from src.utils.plotting import PALETTE, bar_with_values, new_axes, save_figure
from experiments.context import build_context

STAGE = "stage08_covariance"


def figure_covariance(context, scores, intensity, conditions, forecasts, realised, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.4))

    bar_with_values(axes[0, 0], scores["qlike"], "Out-of-sample QLIKE loss (lower is better)",
                    "QLIKE", "{:.3f}", rotation=20)

    bar_with_values(axes[0, 1], conditions, "Covariance condition number (log scale)",
                    "condition number", "{:.0f}", rotation=20)
    axes[0, 1].set_yscale("log")

    axes[1, 0].plot(intensity.index, intensity.to_numpy(), color="#0072B2")
    axes[1, 0].axhline(float(intensity.mean()), color="#D55E00", linestyle="--",
                       label=f"mean {intensity.mean():.3f}")
    axes[1, 0].set_ylabel("Ledoit-Wolf shrinkage intensity")
    axes[1, 0].set_title("How much shrinkage does the data ask for?")
    axes[1, 0].legend(fontsize=8.5)

    for i, (name, series) in enumerate(forecasts.items()):
        aligned = series.reindex(realised.index)
        axes[1, 1].plot(aligned.index, aligned.to_numpy(), color=PALETTE[i], linewidth=0.9,
                        label=name)
    axes[1, 1].plot(realised.index, realised.to_numpy(), color="black", linewidth=1.1,
                    alpha=0.7, label="subsequently realised")
    axes[1, 1].set_ylabel("Annualised volatility")
    axes[1, 1].set_title("Forecast vs realised portfolio volatility")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Figure 18. Which covariance estimator produces useful risk forecasts?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Which covariance method produces the most stable and useful "
                           "out-of-sample risk forecasts, and what does shrinkage actually buy?", 18)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 8 | volatility and covariance modelling (Ch. 20 §20.2)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    complete = returns.dropna(how="any")
    lookback = int(cfg.get("portfolio.covariance.lookback", 252))
    halflife = float(cfg.get("portfolio.covariance.ewma_halflife", 60.0))

    # --- covariance horse race -------------------------------------------
    scores = evaluate_covariance_forecasts(
        complete, ("sample", "ewma", "shrinkage", "pca_denoised"), lookback, 21, 21,
        halflife=halflife,
    )
    context.save_table(scores, "stage08_covariance_scores.csv")
    logger.info("covariance forecast evaluation:\n%s", scores.round(4).to_string())

    conditions = pd.Series(
        {method: condition_number(estimate_covariance(complete.tail(lookback), method, lookback, halflife))
         for method in ("sample", "ewma", "shrinkage", "pca_denoised")},
        name="condition_number",
    )
    context.save_table(conditions.to_frame(), "stage08_condition_numbers.csv")
    logger.info("condition numbers: %s", conditions.round(1).to_dict())

    # --- shrinkage intensity through time --------------------------------
    intensity = {}
    for end in range(lookback, len(complete), 21):
        window = complete.iloc[end - lookback:end]
        try:
            _, delta = ledoit_wolf_shrinkage(window, "constant_correlation", None, True)
            intensity[complete.index[end - 1]] = delta
        except (ValueError, np.linalg.LinAlgError):
            continue
    intensity = pd.Series(intensity, name="shrinkage_intensity")
    context.save_table(intensity.to_frame(), "stage08_shrinkage_intensity.csv")
    logger.info("Ledoit-Wolf intensity: mean %.3f min %.3f max %.3f",
                intensity.mean(), intensity.min(), intensity.max())

    # --- portfolio-level forecast vs realised -----------------------------
    equal = pd.DataFrame(1.0 / complete.shape[1], index=complete.index, columns=complete.columns)
    portfolio = (equal.shift(1) * complete).sum(axis=1)
    realised_forward = forward_realised_volatility(portfolio.to_frame("p"), 21)["p"]
    forecasts = {}
    for method in ("sample", "ewma", "shrinkage"):
        series = {}
        for end in range(lookback, len(complete), 5):
            window = complete.iloc[end - lookback:end]
            try:
                cov = estimate_covariance(window, method, lookback, halflife)
            except (ValueError, np.linalg.LinAlgError):
                continue
            w = np.full(complete.shape[1], 1.0 / complete.shape[1])
            series[complete.index[end - 1]] = float(np.sqrt(w @ cov.to_numpy() @ w))
        forecasts[method] = pd.Series(series)
    forecast_scores = evaluate_forecasts(forecasts, realised_forward)
    context.save_table(forecast_scores, "stage08_portfolio_vol_forecasts.csv")
    logger.info("portfolio volatility forecast scores:\n%s", forecast_scores.round(4).to_string())

    # --- single-asset volatility estimators, including GARCH --------------
    vol_rows = []
    for ticker in returns.columns:
        candidates = {
            "rolling_63": rolling_volatility(returns, 63)[ticker],
            "ewma_hl11": ewma_volatility(returns, 11)[ticker],
            "ewma_hl40": ewma_volatility(returns, 40)[ticker],
            "parkinson_21": parkinson_volatility(market.high, market.low, 21)[ticker],
        }
        try:
            candidates["garch_11"] = garch_rolling_forecast(returns[ticker], 21, 126, 750)
        except Exception as exc:
            logger.warning("GARCH failed for %s: %s", ticker, exc)
        scored = evaluate_forecasts(candidates, forward_realised_volatility(returns, 21)[ticker])
        scored.insert(0, "ticker", ticker)
        vol_rows.append(scored)
    vol_scores = pd.concat(vol_rows).rename_axis("estimator").reset_index()
    ranking = vol_scores.groupby("estimator")[["corr", "rmse", "qlike", "r2"]].mean().sort_values("qlike")
    context.save_table(ranking, "stage08_volatility_ranking.csv")
    logger.info("volatility estimator ranking:\n%s", ranking.round(4).to_string())

    garch = pd.DataFrame({t: garch_parameters(returns[t]) for t in returns.columns}).T
    context.save_table(garch, "stage08_garch_parameters.csv")

    figure_covariance(context, scores, intensity, conditions, forecasts, realised_forward,
                      context.figure("fig18_covariance_evaluation.png"))

    best_cov = str(scores.index[0])
    best_vol = str(ranking.index[0])
    context.registry.log(
        "Which covariance and volatility estimators produce the most useful "
        "out-of-sample risk forecasts? (spec §33)",
        stage=STAGE,
        parameters={"lookback": lookback, "halflife": halflife, "horizon_days": 21},
        results={
            "best_covariance_by_qlike": best_cov,
            "best_covariance_qlike": float(scores.loc[best_cov, "qlike"]),
            "sample_qlike": float(scores.loc["sample", "qlike"]) if "sample" in scores.index else np.nan,
            "shrinkage_qlike": float(scores.loc["shrinkage", "qlike"]) if "shrinkage" in scores.index else np.nan,
            "condition_number_sample": float(conditions.get("sample", np.nan)),
            "condition_number_shrinkage": float(conditions.get("shrinkage", np.nan)),
            "mean_shrinkage_intensity": float(intensity.mean()),
            "best_volatility_estimator": best_vol,
            "garch_mean_persistence": float(garch["persistence"].mean()),
            "garch_integrated_count": int(garch["integrated"].sum()),
        },
        decision="retain",
        notes=(
            f"EWMA forecasts risk most accurately; shrinkage's contribution is conditioning, "
            f"cutting the condition number from {conditions.get('sample', np.nan):.0f} to "
            f"{conditions.get('shrinkage', np.nan):.0f}. Those are different goods and the "
            "distinction matters: an optimiser inverts the covariance matrix, so it is hurt far "
            "more by an ill-conditioned matrix than by a slightly less accurate one. The "
            "production configuration therefore uses EWMA for position sizing and shrinkage "
            "for anything that gets inverted. GARCH(1,1) is retained as a documented extension "
            f"only: its mean persistence of {garch['persistence'].mean():.3f} implies volatility "
            "shocks that decay over months, and it is integrated for "
            f"{int(garch['integrated'].sum())} of {len(garch)} assets."
        ),
    )
    logger.info("STAGE 8 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
