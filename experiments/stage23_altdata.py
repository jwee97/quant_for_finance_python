"""Stage 23 - Non-price ("alternative") data (Generation 3, Priority 6).

Credit spreads, options-implied volatility structure, CFTC positioning and weekly
jobless claims, each with an explicit availability date, tested the way Stage 15
tested macro data: nested models, Clark-West, publication lags respected.
Two pre-declared questions (``config/altdata.yaml``): do the features add
forecasting information to price+macro, and does speculative positioning predict
the matching ETF's next month? The configuration also states which of the
roadmap's data could NOT be sourced; nothing here stands in for them.

Figures 46-47.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.altdata import alt_feature_levels, altdata_version, ensure_alt_raw
from src.data.macro import MacroSeriesSpec, ensure_macro_raw, staleness
from src.features.macro import expanding_zscore, macro_feature_panel
from src.features.sleeves import month_end_dates, monthly_compound, sleeve_returns
from src.models.macro_forecast import build_monthly_panel, evaluate_nested, walk_forward_forecasts
from src.models.regression import ols_hac
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg
from experiments.context import build_context

STAGE = "stage23_altdata"


def figure_panel(levels: pd.DataFrame, z: pd.DataFrame, stale: pd.DataFrame, coverage: pd.DataFrame, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    ax = axes[0, 0]
    ax.plot(levels.index, levels["baa10y_level"], color=PALETTE[0], label="Baa minus 10y (pp)")
    ax.set_ylabel("Credit spread, percentage points")
    ax2 = ax.twinx()
    ax2.plot(levels.index, levels["vix_over_vix3m"], color=PALETTE[1], linewidth=0.8, label="VIX / VIX3M")
    ax2.axhline(1.0, color="grey", linewidth=0.6)
    ax2.set_ylabel("VIX over VIX3M (above 1: inverted)")
    ax.set_title("Credit spread and the implied-volatility term structure")
    ax = axes[0, 1]
    for i, c in enumerate(["cftc_es", "cftc_ust10", "cftc_gold", "cftc_crude"]):
        ax.plot(z.index, z[c], color=PALETTE[i], linewidth=0.9, label=c.replace("cftc_", ""))
    ax.set_ylabel("Net speculative positioning, expanding z-score")
    ax.set_title("Speculative positioning (as known on each date)")
    ax.legend(fontsize=8, ncol=2)
    ax = axes[1, 0]
    ax.bar(np.arange(len(stale.columns)), [stale[c].mean() for c in stale.columns], color=PALETTE[2])
    ax.set_xticks(np.arange(len(stale.columns)), stale.columns, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean age of the value in use, calendar days")
    ax.set_title("How stale the information is on a typical trading day")
    ax = axes[1, 1]
    first = z.apply(lambda col: col.first_valid_index()).dropna()
    ax.barh(np.arange(len(first)), (first.dt.year + first.dt.dayofyear / 366.0), color=PALETTE[3])
    ax.set_yticks(np.arange(len(first)), first.index, fontsize=8)
    ax.set_xlim(1996, 2013)
    ax.set_xlabel("First date the standardised feature exists")
    ax.set_title("Standardisation burns in 756 trading days")
    fig.suptitle("Figure 46. Non-price data: what is known, when, and how stale", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Can credit spreads, options-implied volatility, CFTC positioning and jobless claims be used "
                           "without look-ahead, and how fresh is each on a typical trading day?", 46)


def figure_results(nested: pd.DataFrame, matched: pd.DataFrame, positioning: pd.DataFrame, ic: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(16.5, 5.2))
    ax = axes[0]
    x = np.arange(len(nested))
    ax.bar(x - 0.2, nested["oos_r2"] * 100, width=0.4, color=PALETTE[0], label="vs price+macro (declared)")
    ax.bar(x + 0.2, matched["oos_r2"] * 100, width=0.4, color=PALETTE[1], label="matched sample (post-hoc)")
    for xi, (_, r) in zip(x, nested.iterrows()):
        ax.text(xi - 0.2, r["oos_r2"] * 100, f"p={r['p_value']:.2f}", ha="center", va="bottom" if r["oos_r2"] >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, nested["sleeve"], rotation=20)
    ax.set_ylabel("Out-of-sample R², percentage points")
    ax.set_title("Does adding non-price data help? (Clark-West)")
    ax.legend(fontsize=8)
    ax = axes[1]
    y = np.arange(len(positioning))
    ax.errorbar(positioning["t_beta"], y, xerr=1.65, fmt="o", color="black", capsize=3)
    for yi, (_, r) in zip(y, positioning.iterrows()):
        ax.text(r["t_beta"] + 1.8, yi, f"p={r['p_beta']:.2f}" + ("  *" if r["bh_significant"] else ""), va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, positioning["contract"])
    ax.set_xlabel("t-statistic of positioning z-score (Newey-West)")
    ax.set_title("Does positioning predict next month?", fontsize=10)
    ax = axes[2]
    im = ax.imshow(ic.to_numpy(dtype=float), cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(np.arange(ic.shape[1]), ic.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(ic.shape[0]), ic.index, fontsize=8)
    for i in range(ic.shape[0]):
        for j in range(ic.shape[1]):
            if np.isfinite(ic.iat[i, j]):
                ax.text(j, i, f"{ic.iat[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Rank correlation with next month's\nexcess return (not judged)", fontsize=10)
    fig.suptitle("Figure 47. Do non-price data add anything to price and macro?", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Do non-price features add out-of-sample forecasting information to price and macro features, "
                           "and does speculative positioning predict the matching ETF's next-month return?", 47)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 23: non-price data").parse_args(argv)
    context, logger = build_context(STAGE, generation=3)
    cfg = context.config
    node = cfg.get("altdata", {}) or {}
    logger.info("=" * 72)
    logger.info("STAGE 23 | non-price data (Generation 3, Priority 6)")
    logger.info("=" * 72)
    fdr = float((node.get("decisions", {}) or {}).get("fdr", 0.10))
    market = context.market_data()
    returns = market.returns()
    index = pd.DatetimeIndex(market.prices.index)

    macro_specs, macro_raw = ensure_macro_raw(cfg)
    specs, raw, positioning = ensure_alt_raw(cfg)
    cftc_lag = int((node.get("cftc", {}) or {}).get("release_lag_days", 4))
    vix_spec = next(s for s in macro_specs if s.id == "VIX")
    macro_z = macro_feature_panel(macro_raw, macro_specs, index, cfg)["z"]
    start = max(min(s.index.min() for s in raw.values()), pd.Timestamp("1995-01-01"))
    calendar = pd.bdate_range(start, index.max())
    levels = alt_feature_levels(raw, specs, positioning, macro_raw["VIX"], vix_spec, cftc_lag, calendar)
    z = expanding_zscore(levels, min_periods=756, winsorize=4.0)
    logger.info("non-price data version %s; first usable feature dates:\n%s", altdata_version(context.config.path("metadata")),
                z.apply(lambda c: str(c.first_valid_index().date()) if c.first_valid_index() is not None else "never").to_string())

    coverage = pd.DataFrame({s.id: {"source": s.source, "release_lag_days": s.release_lag_days, "first": str(raw[s.id].index.min().date()),
                                    "last": str(raw[s.id].index.max().date()), "observations": int(len(raw[s.id]))} for s in specs}).T
    cf_cov = positioning.groupby("contract").agg(first=("date", "min"), last=("date", "max"), observations=("date", "size"))
    cf_cov["release_lag_days"] = cftc_lag
    context.save_table(coverage, "stage23_coverage.csv")
    context.save_table(cf_cov, "stage23_cftc_coverage.csv")
    context.save_table(z.describe().T, "stage23_feature_summary.csv")
    stale_specs = {s.id: s for s in specs}
    stale = staleness({k: raw[k] for k in ("BAA10Y", "ICSA", "VIX3M", "GVZ", "OVX")},
                      [stale_specs[k] for k in ("BAA10Y", "ICSA", "VIX3M", "GVZ", "OVX")], index)
    from src.data.altdata import cftc_series
    cs, csp = cftc_series(positioning, cftc_lag)
    stale = pd.concat([stale, staleness(cs, csp, index)], axis=1)
    context.save_table(stale.describe().T, "stage23_staleness.csv")
    figure_panel(levels, z, stale, coverage, context.figure("fig46_nonprice_panel.png"))

    # ------------------------------------------------------------ nested test
    pred = cfg.get("macro.predictability", {}) or {}
    sleeves = {k: list(v) for k, v in (pred.get("sleeves", {}) or {}).items()}
    cash = str(pred.get("cash", "SHY"))
    sleeve_daily = sleeve_returns(returns, sleeves, market.investable)
    monthly = build_monthly_panel(sleeve_daily, returns[cash], macro_z)
    dates = monthly.dates
    mask = z.reindex(dates).copy()
    mask = mask.where(mask.isna(), 0.0)                              # zero-information block with the SAME missingness
    kwargs = dict(min_train_months=int(pred.get("min_train_months", 60)), alphas=list(pred.get("ridge_alphas", [1, 10, 100, 1000])),
                  cv_splits=int(pred.get("cv_splits", 5)), first_test_year=int(pred.get("first_test_year", 2010)))
    blocks = {"nonprice": z.reindex(dates), "mask": mask}
    extra = {"all": ("price", "macro", "nonprice"), "both_matched": ("price", "macro", "mask")}
    logger.info("walk-forward: price+macro, price+macro+non-price, matched-sample restricted model")
    result = walk_forward_forecasts(monthly, models=("both", "all", "both_matched"), extra_blocks=blocks, extra_models=extra, **kwargs)
    declared = evaluate_nested(result, [("both", "all")], fdr)
    matched = evaluate_nested(result, [("both_matched", "all")], fdr)
    context.save_table(declared, "stage23_clark_west_declared.csv", index=False)
    context.save_table(matched, "stage23_clark_west_matched_posthoc.csv", index=False)
    cols = ["sleeve", "oos_r2", "statistic", "p_value", "bh_significant", "n_obs"]
    logger.info("Clark-West, price+macro+non-price vs price+macro (declared):\n%s", declared[cols].round(4).to_string(index=False))
    logger.info("Clark-West, matched-sample restricted model (post-hoc):\n%s", matched[cols].round(4).to_string(index=False))

    # post-hoc control: publication lags ignored
    naive_specs = [MacroSeriesSpec(s.id, s.source, s.frequency, 0, s.symbol, s.feature) for s in specs]
    naive_z = expanding_zscore(alt_feature_levels(raw, naive_specs, positioning, macro_raw["VIX"], vix_spec, 0, calendar), 756, 4.0)
    naive_result = walk_forward_forecasts(monthly, models=("both", "all"), extra_blocks={"nonprice": naive_z.reindex(dates)},
                                          extra_models={"all": ("price", "macro", "nonprice")}, **kwargs)
    naive = evaluate_nested(naive_result, [("both", "all")], fdr)
    context.save_table(naive, "stage23_clark_west_lags_ignored_posthoc.csv", index=False)
    logger.info("lags IGNORED (control):\n%s", naive[cols].round(4).to_string(index=False))

    # ------------------------------------------------- positioning, per contract
    excess = monthly_compound(returns).sub(monthly_compound(returns[cash]), axis=0)
    month_ends = month_end_dates(index)
    forward = excess.reindex(month_ends).shift(-1)
    rows = []
    for name, c in (node.get("cftc", {}) or {}).get("contracts", {}).items():
        x = z[f"cftc_{name}"].reindex(month_ends)
        y = forward[c["etf"]]
        try:
            fit = ols_hac(y, x.rename("beta").to_frame(), hac_lags=1)
            row = fit.to_row("beta")
        except ValueError:
            continue
        rows.append({"contract": name, "etf": c["etf"], **row})
    positioning_table = pd.DataFrame(rows)
    positioning_table["bh_significant"] = benjamini_hochberg(positioning_table["p_beta"], fdr)
    context.save_table(positioning_table, "stage23_positioning_tests.csv", index=False)
    logger.info("positioning -> next-month ETF excess return (Newey-West lag 1, BH FDR %.2f):\n%s", fdr,
                positioning_table[["contract", "etf", "beta", "t_beta", "p_beta", "n_obs", "bh_significant"]].round(4).to_string(index=False))

    # reported, not judged: rank correlations of every feature with next-month excess return
    targets = {"SPY": "SPY", "QQQ": "QQQ", "IEF": "IEF", "HYG": "HYG", "GLD": "GLD", "DBC": "DBC"}
    ic = pd.DataFrame(index=z.columns, columns=list(targets), dtype=float)
    ic_p = ic.copy()
    zm = z.reindex(month_ends)
    for feature in z.columns:
        for label, etf in targets.items():
            pair = pd.concat([zm[feature], forward[etf]], axis=1).dropna()
            if len(pair) >= 36:
                rho, pval = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                ic.loc[feature, label], ic_p.loc[feature, label] = rho, pval
    context.save_table(ic, "stage23_feature_rank_correlations.csv")
    context.save_table(ic_p, "stage23_feature_rank_correlation_pvalues_posthoc.csv")
    flat_p = ic_p.stack()
    n_cells, n_raw = int(len(flat_p)), int((flat_p < 0.05).sum())
    n_bh = int(benjamini_hochberg(flat_p.to_numpy(), fdr).sum())
    logger.info("post-hoc: of %d feature-ETF rank correlations, %d have an uncorrected p < 0.05 (%.1f expected by chance); %d survive BH at %.2f",
                n_cells, n_raw, 0.05 * n_cells, n_bh, fdr)
    figure_results(declared, matched, positioning_table, ic, context.figure("fig47_nonprice_results.png"))

    # -------------------------------------------------------------- registry
    reg = context.registry
    adds = bool(declared["bh_significant"].any() and declared["oos_r2"].mean() > 0)
    reg.log(
        "Non-price data (credit spread, implied-volatility structure, CFTC positioning, jobless claims) add out-of-sample "
        "forecasting information to price and macro features for monthly sleeve returns.",
        stage=STAGE, parameters={"features": list(z.columns), "fdr": fdr, "min_train_months": kwargs["min_train_months"],
                                 "nonprice_data_version": altdata_version(cfg.path("metadata")),
                                 "not_obtained": ["ETF flows", "earnings and analyst revisions", "ETF creation and redemption"]},
        results={"sleeves_bh_significant": int(declared["bh_significant"].sum()), "sleeves_raw_significant": int((declared["p_value"] < 0.05).sum()),
                 "mean_oos_r2": float(declared["oos_r2"].mean()), "min_p_value": float(declared["p_value"].min()),
                 "oos_months_min": int(declared["n_obs"].min()), "oos_months_max": int(declared["n_obs"].max()),
                 "posthoc_matched_sleeves_bh_significant": int(matched["bh_significant"].sum()),
                 "posthoc_matched_mean_oos_r2": float(matched["oos_r2"].mean()),
                 "posthoc_lags_ignored_sleeves_bh_significant": int(naive["bh_significant"].sum()),
                 "posthoc_lags_ignored_mean_oos_r2": float(naive["oos_r2"].mean()),
                 "posthoc_rank_correlations_tested": n_cells, "posthoc_rank_correlations_raw_significant_5pct": n_raw,
                 "posthoc_rank_correlations_bh_significant": n_bh},
        decision="retain" if adds else "reject", test_period="walk-forward monthly, first test year 2010 (limited by feature burn-in)",
        notes=("Rule fixed in advance: retain if at least one sleeve passes Clark-West at FDR 10% for price+macro+non-price against "
               "price+macro AND the mean out-of-sample R-squared is positive. The standardised features need 756 trading days and the implied-"
               "volatility series start in 2006-2008, so the usable sample is shorter than Stage 15's. The declared restricted model "
               "uses all of its own history; the matched-sample control (restricted model trained on exactly the same rows) is post-hoc. "
               "ETF flows, earnings/analyst revisions and creation/redemption data could not be sourced and are not proxied."),
    )
    reg.log(
        "Speculative positioning (CFTC) predicts the next month's excess return of the matching ETF.",
        stage=STAGE, parameters={"contracts": list(positioning_table["contract"]), "design": "OLS on month-end z-score, Newey-West lag 1, BH across the family"},
        results={"contracts_tested": int(len(positioning_table)), "bh_significant": int(positioning_table["bh_significant"].sum()),
                 "min_p_value": float(positioning_table["p_beta"].min()),
                 **{f"t_{r.contract}": float(r.t_beta) for r in positioning_table.itertuples()}},
        decision="retain" if bool(positioning_table["bh_significant"].any()) else "reject", test_period="monthly, from the first usable feature date",
        notes="Two-sided: a contrarian (negative) or a trend (positive) relationship would both count. The sign of each coefficient is in the table.",
    )
    logger.info("STAGE 23 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
