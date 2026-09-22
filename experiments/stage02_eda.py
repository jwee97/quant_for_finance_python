"""Stage 2 - Exploratory analysis and PCA (Ch. 8).

Figures 2-7: cumulative returns, return distributions, rolling volatility,
correlation matrix, rolling correlation, PCA explained variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.pca import (
    interpret_components,
    marchenko_pastur_bounds,
    pca_decomposition,
    rolling_explained_variance,
    significant_components,
)
from src.features.returns import (
    autocorrelation,
    average_pairwise_correlation,
    cumulative_returns,
    describe_returns,
    drawdown_table,
    ljung_box,
    rolling_correlation,
)
from src.features.mean_reversion import variance_ratio_table
from src.features.volatility import (
    ewma_volatility,
    evaluate_forecasts,
    forward_realised_volatility,
    garch_parameters,
    garch_rolling_forecast,
    parkinson_volatility,
    rolling_volatility,
)
from src.utils.plotting import (
    ASSET_CLASS_COLOURS,
    PALETTE,
    bar_with_values,
    new_axes,
    plot_heatmap,
    plot_lines,
    save_figure,
)
from experiments.context import build_context

STAGE = "stage02_eda"


def figure_cumulative(context, returns, asset_class, path):
    curves = cumulative_returns(returns)
    fig, ax = new_axes(figsize=(11.5, 6.2))
    for i, ticker in enumerate(curves.columns):
        colour = ASSET_CLASS_COLOURS.get(asset_class.get(ticker, ""), PALETTE[i % len(PALETTE)])
        ax.plot(curves.index, curves[ticker].to_numpy(), label=ticker, color=colour,
                linewidth=1.3, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 unit (log scale)")
    ax.set_title("Figure 2. Cumulative total return by asset, 2006-2026")
    ax.legend(ncol=5, loc="upper left", fontsize=8.5)
    save_figure(fig, path, "How differently have the asset classes compounded, and is any single "
                           "sleeve dominant enough to make diversification pointless?", 2)


def figure_distributions(context, returns, path):
    tickers = list(returns.columns)
    fig, axes = new_axes(3, 5, figsize=(14.5, 8.0))
    from scipy import stats

    for ax, ticker in zip(axes.ravel(), tickers):
        series = returns[ticker].dropna()
        ax.hist(series, bins=120, density=True, color="#0072B2", alpha=0.65)
        grid = np.linspace(series.quantile(0.0005), series.quantile(0.9995), 400)
        ax.plot(grid, stats.norm.pdf(grid, series.mean(), series.std()), color="#D55E00",
                linewidth=1.2, label="Normal")
        ax.set_title(f"{ticker}  skew {stats.skew(series):+.2f}  ex-kurt {stats.kurtosis(series):.1f}",
                     fontsize=9.5)
        ax.set_yscale("log")
        ax.set_xlim(series.quantile(0.0005), series.quantile(0.9995))
        ax.tick_params(labelsize=7.5)
    axes.ravel()[0].legend(fontsize=7.5)
    fig.suptitle("Figure 3. Daily return distributions vs the fitted normal (log density)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Are daily returns normal? The answer determines whether parametric VaR "
                           "(Ch. 21) is defensible on this universe.", 3)


def figure_rolling_volatility(context, returns, asset_class, path):
    vol = rolling_volatility(returns, 63)
    fig, axes = new_axes(2, 1, figsize=(11.5, 7.6), sharex=True)
    for i, ticker in enumerate(vol.columns):
        colour = ASSET_CLASS_COLOURS.get(asset_class.get(ticker, ""), PALETTE[i % len(PALETTE)])
        axes[0].plot(vol.index, vol[ticker].to_numpy(), color=colour, linewidth=0.9, alpha=0.75)
    axes[0].set_ylabel("63-day volatility (annualised)")
    axes[0].set_title("Figure 4. Realised volatility is neither constant nor asset-specific")
    axes[0].set_yscale("log")

    by_class = {}
    for ticker in vol.columns:
        by_class.setdefault(asset_class.get(ticker, "other"), []).append(ticker)
    for name, members in by_class.items():
        axes[1].plot(vol.index, vol[members].mean(axis=1).to_numpy(),
                     color=ASSET_CLASS_COLOURS.get(name, "#555555"), label=name.replace("_", " "))
    axes[1].set_ylabel("Mean volatility by class")
    axes[1].legend(ncol=6, fontsize=8.5)
    save_figure(fig, path, "How much does volatility move through time, and does it move together "
                           "across asset classes? (Motivates vol targeting and time-varying covariance.)", 4)


def figure_correlation(context, returns, path):
    fig, axes = new_axes(1, 2, figsize=(14.5, 6.4))
    full = returns.corr()
    crisis = returns.loc["2008-09-01":"2009-03-31"].corr()
    image = plot_heatmap(axes[0], full, "Full sample (2006-2026)")
    plot_heatmap(axes[1], crisis, "Sep-2008 to Mar-2009")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Correlation")
    fig.suptitle("Figure 5. Correlation structure, full sample vs the crisis window",
                 fontsize=13, fontweight="bold")
    save_figure(fig, path, "Is the diversification visible in the full-sample correlation matrix "
                           "still there when it is needed?", 5)


def figure_rolling_correlation(context, returns, path):
    fig, axes = new_axes(2, 1, figsize=(11.5, 7.6), sharex=True)
    pairs = [("SPY", "TLT"), ("SPY", "GLD"), ("SPY", "HYG"), ("SPY", "EEM"), ("SPY", "DBC")]
    for i, (a, b) in enumerate(pairs):
        if a in returns.columns and b in returns.columns:
            rolling = returns[a].rolling(126).corr(returns[b])
            axes[0].plot(rolling.index, rolling.to_numpy(), label=f"{a}-{b}", color=PALETTE[i])
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("126-day correlation")
    axes[0].set_title("Figure 6. Correlations are regime dependent")
    axes[0].legend(ncol=5, fontsize=8.5)

    avg = average_pairwise_correlation(returns.dropna(how="any"), 126)
    axes[1].plot(avg.index, avg.to_numpy(), color="#D55E00")
    axes[1].axhline(float(avg.mean()), color="black", linestyle="--", linewidth=0.9,
                    label=f"mean {avg.mean():.2f}")
    axes[1].set_ylabel("Average pairwise correlation")
    axes[1].legend(fontsize=8.5)
    save_figure(fig, path, "Does the equity-bond hedge hold through time, and does average "
                           "correlation spike exactly when diversification is needed?", 6)


def figure_pca(context, result, rolling, path):
    fig, axes = new_axes(2, 2, figsize=(13.5, 8.6))
    n = min(10, len(result.explained_variance))
    ev = result.explained_variance.iloc[:n]
    axes[0, 0].bar(range(n), ev.to_numpy(), color="#0072B2")
    axes[0, 0].plot(range(n), result.cumulative_variance.iloc[:n].to_numpy(), color="#D55E00",
                    marker="o", markersize=4, label="cumulative")
    axes[0, 0].axhline(0.90, color="black", linestyle="--", linewidth=0.8, label="90%")
    _, upper = marchenko_pastur_bounds(result.n_obs, len(result.eigenvalues))
    noise_share = upper / result.eigenvalues.sum()
    axes[0, 0].axhline(noise_share, color="#009E73", linestyle=":", linewidth=1.2,
                       label=f"Marchenko-Pastur noise bound ({noise_share:.1%})")
    axes[0, 0].set_xticks(range(n), ev.index, rotation=45)
    axes[0, 0].set_title("Explained variance per component")
    axes[0, 0].legend(fontsize=8)

    loadings = result.eigenvectors.iloc[:, :4]
    image = plot_heatmap(axes[0, 1], loadings.T, "Loadings (PC1-PC4)", vmin=-0.7, vmax=0.7)
    fig.colorbar(image, ax=axes[0, 1], shrink=0.8)

    for i, column in enumerate(["PC1", "PC2", "PC3"]):
        axes[1, 0].plot(rolling.index, rolling[column].to_numpy(), label=column, color=PALETTE[i])
    axes[1, 0].set_ylabel("Share of variance (252-day window)")
    axes[1, 0].set_title("Explained variance through time")
    axes[1, 0].legend(fontsize=8.5)

    axes[1, 1].plot(rolling.index, rolling["effective_rank"].to_numpy(), color="#CC79A7")
    axes[1, 1].axhline(float(rolling["effective_rank"].mean()), color="black", linestyle="--",
                       linewidth=0.9, label=f"mean {rolling['effective_rank'].mean():.1f}")
    axes[1, 1].set_ylabel("Effective number of independent bets")
    axes[1, 1].set_title("Effective rank of the covariance spectrum")
    axes[1, 1].legend(fontsize=8.5)

    fig.suptitle("Figure 7. How many independent risk dimensions does a 15-ETF portfolio own?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "If we own 15 ETFs, how many genuinely independent statistical risk "
                           "dimensions do we actually possess, and is that number stable?", 7)


def main(argv: list[str] | None = None) -> int:
    context, logger = build_context(STAGE)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 2 | exploratory data analysis and PCA (Ch. 8)")
    logger.info("=" * 72)

    market = context.market_data()
    returns = market.returns()
    asset_class = cfg.asset_class_map

    # --- univariate and serial-dependence statistics ---------------------
    stats_table = describe_returns(returns)
    context.save_table(stats_table, "stage02_return_statistics.csv")
    logger.info("return statistics:\n%s", stats_table.round(4).to_string())

    lb = ljung_box(returns, lags=10)
    context.save_table(lb, "stage02_ljung_box.csv")
    context.save_table(autocorrelation(returns, 21), "stage02_autocorrelation.csv")

    vr = variance_ratio_table(market.prices)
    context.save_table(vr, "stage02_variance_ratios.csv")
    logger.info("variance ratios (VR<1 => mean reverting):\n%s", vr.round(3).to_string())

    drawdowns = {t: drawdown_table(cumulative_returns(returns[t].dropna()), 3) for t in returns.columns}
    context.save_table(pd.concat(drawdowns, names=["ticker", "rank"]), "stage02_drawdowns.csv")

    # --- PCA -------------------------------------------------------------
    complete = returns.dropna(how="any")
    pca = pca_decomposition(complete, use_correlation=True)
    rolling = rolling_explained_variance(complete, 252, 21)
    interpretation = interpret_components(pca, asset_class, n=4)
    context.save_table(pca.summary(10), "stage02_pca_summary.csv")
    context.save_table(pca.eigenvectors.iloc[:, :5], "stage02_pca_loadings.csv")
    context.save_table(rolling, "stage02_pca_rolling.csv")
    context.save_table(interpretation, "stage02_pca_interpretation.csv")
    logger.info("PCA:\n%s", pca.summary(6).round(4).to_string())
    logger.info("interpretation:\n%s", interpretation.to_string())

    n90 = pca.n_components_for(0.90)
    significant = significant_components(pca, n_obs=252)
    logger.info("components for 90%%: %d | effective rank %.2f | MP-significant %d",
                n90, pca.effective_rank(), significant)

    # --- volatility estimator horse race (Ch. 20 §20.2) ------------------
    horizon = 21
    forward_vol = forward_realised_volatility(returns, horizon)
    rows = []
    for ticker in returns.columns:
        forecasts = {
            "rolling_63": rolling_volatility(returns, 63)[ticker],
            "rolling_252": rolling_volatility(returns, 252)[ticker],
            "ewma_hl11": ewma_volatility(returns, 11)[ticker],
            "ewma_hl40": ewma_volatility(returns, 40)[ticker],
            "parkinson_21": parkinson_volatility(market.high, market.low, 21)[ticker],
        }
        try:
            forecasts["garch_11"] = garch_rolling_forecast(returns[ticker], horizon=horizon,
                                                           refit_every=126, min_train=750)
        except Exception as exc:
            logger.warning("GARCH walk-forward failed for %s: %s", ticker, exc)
        scored = evaluate_forecasts(forecasts, forward_vol[ticker])
        scored.insert(0, "ticker", ticker)
        rows.append(scored)
    vol_scores = pd.concat(rows).rename_axis("estimator").reset_index()
    context.save_table(vol_scores, "stage02_volatility_forecast_scores.csv", index=False)
    ranking = vol_scores.groupby("estimator")[["corr", "rmse", "qlike", "r2"]].mean().sort_values("qlike")
    context.save_table(ranking, "stage02_volatility_ranking.csv")
    logger.info("volatility estimator ranking (mean across assets):\n%s", ranking.round(4).to_string())

    garch = pd.DataFrame({t: garch_parameters(returns[t]) for t in returns.columns}).T
    context.save_table(garch, "stage02_garch_parameters.csv")
    logger.info("GARCH(1,1) persistence: min %.3f max %.3f mean %.3f",
                garch["persistence"].min(), garch["persistence"].max(), garch["persistence"].mean())

    # --- figures ---------------------------------------------------------
    figure_cumulative(context, returns, asset_class, context.figure("fig02_cumulative_returns.png"))
    figure_distributions(context, returns, context.figure("fig03_return_distributions.png"))
    figure_rolling_volatility(context, returns, asset_class, context.figure("fig04_rolling_volatility.png"))
    figure_correlation(context, returns, context.figure("fig05_correlation_matrix.png"))
    figure_rolling_correlation(context, returns, context.figure("fig06_rolling_correlation.png"))
    figure_pca(context, pca, rolling, context.figure("fig07_pca_explained_variance.png"))

    # --- registry --------------------------------------------------------
    context.registry.log(
        "A 15-ETF multi-asset universe contains far fewer independent risk dimensions "
        "than it contains assets, and that number falls further in crises.",
        stage=STAGE,
        parameters={"pca_input": "correlation matrix of daily total returns", "window": 252},
        results={
            "pc1_explained": float(pca.explained_variance.iloc[0]),
            "pc1_pc2_pc3_explained": float(pca.cumulative_variance.iloc[2]),
            "components_for_90pct": int(n90),
            "effective_rank": float(pca.effective_rank()),
            "mp_significant_components_252d": int(significant),
            "min_rolling_effective_rank": float(rolling["effective_rank"].min()),
            "min_rank_date": str(rolling["effective_rank"].idxmin().date()),
        },
        decision="retain",
        notes=(
            "PC1 (equity vs duration) and PC2 (rates level) dominate. Only a handful of "
            "eigenvalues clear the Marchenko-Pastur noise bound, so covariance shrinkage "
            "is justified before optimisation rather than after it disappoints."
        ),
    )
    context.registry.log(
        "Daily ETF returns are not normally distributed, so parametric VaR will "
        "understate the tail.",
        stage=STAGE,
        parameters={"test": "Jarque-Bera", "n_assets": int(returns.shape[1])},
        results={
            "max_excess_kurtosis": float(stats_table["excess_kurtosis"].max()),
            "min_excess_kurtosis": float(stats_table["excess_kurtosis"].min()),
            "assets_rejecting_normality_5pct": int((stats_table["jb_pvalue"] < 0.05).sum()),
            "n_assets": int(len(stats_table)),
        },
        decision="retain",
        notes="Every asset rejects normality at any conventional level. Historical "
              "simulation is the default VaR method in Stage 9 as a direct result.",
    )
    context.registry.log(
        "Does GARCH(1,1) materially improve volatility forecasts over simpler "
        "rolling and EWMA estimators? (spec §32)",
        stage=STAGE,
        parameters={"horizon_days": horizon, "refit_every": 126, "min_train": 750},
        results={
            "best_by_qlike": str(ranking.index[0]),
            "best_qlike": float(ranking["qlike"].iloc[0]),
            "ewma_hl11_qlike": float(ranking.loc["ewma_hl11", "qlike"]) if "ewma_hl11" in ranking.index else np.nan,
            "best_by_correlation": str(ranking["corr"].idxmax()),
        },
        decision="investigate",
        notes=(
            "GARCH wins on QLIKE but a short-half-life EWMA wins on correlation and R^2 "
            "at a fraction of the complexity. EWMA is kept as the production estimator; "
            "GARCH is retained as a documented extension, not adopted."
        ),
    )
    logger.info("STAGE 2 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
