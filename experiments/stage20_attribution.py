"""Stage 20 - Portfolio attribution (Generation 2, Priority 8).

A Sharpe ratio says how a book did, not why. This stage takes the long-only
books of the platform and decomposes their results, with every identity checked
to machine precision before anything is reported:

1. ACTIVE RETURN versus equal weight, by asset class: Brinson-Fachler
   allocation, selection and interaction, linked across months with Carino so
   that the linked effects sum exactly to the cumulative active return.
2. RETURN CONTRIBUTION by asset class (the benchmark return is zero in the
   linking), and TRANSACTION COSTS by asset class.
3. RISK CONTRIBUTION by asset class: the Euler decomposition of forecast
   volatility, averaged over month-ends. Risk parity is supposed to equalise
   these across assets; this checks that it does, and what HRP and HERC do.
4. SLEEVE ATTRIBUTION of the Stage 10 inverse-volatility strategy blend: how
   much of the combined return each member strategy supplied.

Every book is attributed over the SAME months (those in which all of them are
fully invested, 2007-05 onward), so cumulative returns and active returns are
comparable across books and across the panels of the figures.

Accounting, not hypothesis testing: the registry records what was found, and the
stage raises if an identity fails. Figures 40-41.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.backtest.attribution import (
    brinson_fachler,
    contribution_by_asset,
    cost_by_asset,
    cumulative,
    euler_risk_contributions,
    link_effects,
    monthly_inputs,
)
from src.backtest.engine import BacktestEngine
from src.portfolio.covariance import estimate_covariance
from src.signals.combine import combine_strategy_returns
from src.utils.plotting import PALETTE, new_axes, save_figure
from experiments.context import build_context
from experiments.strategies import cached_ladder

STAGE = "stage20_attribution"
BENCHMARK = "M0_equal_weight"
ATTRIBUTED = ["M1_inverse_vol", "M2_risk_parity", "M11_hrp", "M12_herc"]
SHORT = {"M0_equal_weight": "equal weight", "M1_inverse_vol": "inverse vol", "M2_risk_parity": "risk parity",
         "M11_hrp": "HRP", "M12_herc": "HERC", "M3_momentum": "momentum", "M4_mean_reversion": "mean reversion",
         "M9_mean_cvar": "mean-CVaR"}
CLASS_COLOURS = {"equity": "#0072B2", "rates": "#56B4E9", "fixed_income": "#009E73", "credit": "#E69F00",
                 "commodity": "#D55E00", "real_estate": "#CC79A7"}
TOLERANCE = 1e-9
SLEEVES = ["M1_inverse_vol", "M2_risk_parity", "M3_momentum", "M4_mean_reversion", "M9_mean_cvar"]


# ---------------------------------------------------------------------------
# Brinson-Fachler against the benchmark, with every identity checked
# ---------------------------------------------------------------------------
def fully_invested_months(run, returns: pd.DataFrame) -> pd.PeriodIndex:
    """Months whose starting book is fully invested (weights sum to one)."""
    weights = monthly_inputs(run.weights, returns)["weights"]
    return weights.index[weights.sum(axis=1).sub(1.0).abs().lt(1e-6)]


def common_window(runs: dict, returns: pd.DataFrame) -> pd.PeriodIndex:
    """Months in which EVERY book is fully invested, so that all books are attributed over the same months.

    Without this each book would be measured over its own life (the benchmark's starts in 2006, before HYG
    lists, and the risk-based books need a year of history), and cumulative returns would not be comparable.
    """
    window = None
    for run in runs.values():
        months = fully_invested_months(run, returns)
        window = months if window is None else window.intersection(months)
    return window.sort_values()


def attribute_against(benchmark_run, run, returns: pd.DataFrame, sector_of: dict, window: pd.PeriodIndex) -> dict:
    """Monthly BF effects for one book, Carino-linked, with reconciliation to the engine's own returns."""
    mine = monthly_inputs(run.weights, returns, run.gross_returns)
    base = monthly_inputs(benchmark_run.weights, returns, benchmark_run.gross_returns)
    both = (mine["weights"].sum(axis=1).sub(1.0).abs().lt(1e-6) & base["weights"].sum(axis=1).sub(1.0).abs().lt(1e-6)
            & mine["weights"].index.isin(window))
    wp, wb = mine["weights"][both], base["weights"][both]
    asset = mine["asset_returns"].loc[both[both].index]
    result = brinson_fachler(wp, wb, asset, sector_of)

    reconciliation = max(
        float((result.portfolio_return - mine["portfolio_return"].loc[result.portfolio_return.index]).abs().max()),
        float((result.benchmark_return - base["portfolio_return"].loc[result.benchmark_return.index]).abs().max()))
    linked = result.linked()
    engine_active = (cumulative(mine["portfolio_return"].loc[result.portfolio_return.index])
                     - cumulative(base["portfolio_return"].loc[result.benchmark_return.index]))
    checks = {"period_identity": result.identity_error(), "reconciliation_to_engine": reconciliation,
              "linked_total_vs_cumulative_active": abs(float(linked["total"].sum()) - engine_active)}
    months = result.portfolio_return.index
    return {"result": result, "linked": linked, "checks": checks, "cumulative_active_return": engine_active,
            "months": int(len(months)),
            "volatility": float(mine["portfolio_return"].loc[months].std(ddof=1) * np.sqrt(12)),
            "benchmark_volatility": float(base["portfolio_return"].loc[months].std(ddof=1) * np.sqrt(12)),
            "portfolio_cumulative": cumulative(mine["portfolio_return"].loc[result.portfolio_return.index]),
            "benchmark_cumulative": cumulative(base["portfolio_return"].loc[result.benchmark_return.index])}


def contributions_by_class(run, returns: pd.DataFrame, sector_of: dict, window: pd.PeriodIndex) -> dict:
    """Linked contribution to the total gross return by asset class over ``window``; sums to the cumulative return."""
    inputs = monthly_inputs(run.weights, returns, run.gross_returns)
    valid = inputs["weights"].sum(axis=1).sub(1.0).abs().lt(1e-6) & inputs["weights"].index.isin(window)
    contributions = contribution_by_asset(inputs["weights"][valid], inputs["asset_returns"][valid])
    by_class = contributions.T.groupby(lambda c: sector_of[c]).sum().T
    total = inputs["portfolio_return"][valid]
    linked = link_effects(by_class, total)
    error = abs(float(linked.sum()) - cumulative(total))
    return {"linked": linked, "error": error, "cumulative": cumulative(total), "monthly": by_class}


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def cost_attribution(run, engine, sector_of: dict, window: pd.PeriodIndex) -> pd.Series:
    """Annual transaction cost, in basis points, by asset class, over the days of ``window``."""
    per_asset = cost_by_asset(run.trades, engine.cost_model.rates(run.trades.columns))
    per_asset = per_asset[pd.DatetimeIndex(per_asset.index).to_period("M").isin(window)]
    by_class = per_asset.T.groupby(lambda c: sector_of[c]).sum().T
    years = max((per_asset.index[-1] - per_asset.index[0]).days / 365.25, 1e-9)
    return 1e4 * by_class.sum() / years


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
def risk_attribution(run, returns: pd.DataFrame, sector_of: dict, months: pd.PeriodIndex, lookback: int = 252) -> dict:
    """Euler risk shares by class and by asset, averaged over month-ends (shrinkage covariance, annualised)."""
    index = pd.DatetimeIndex(returns.index)
    earning = run.weights.shift(1)
    month_start = earning.groupby(index.to_period("M")).head(1)
    rows, class_rows, identity = [], [], 0.0
    for date, weights in month_start.iterrows():
        if weights.isna().all() or abs(weights.sum() - 1.0) > 1e-6 or date.to_period("M") not in months:
            continue
        position = index.get_loc(date)
        if position < lookback:
            continue
        window = returns.iloc[position - lookback:position]
        live = [c for c in window.columns if window[c].notna().all()]
        if abs(weights.drop(labels=live, errors="ignore").abs().sum()) > 1e-9 or len(live) < 5:
            continue                                       # the book holds an asset with no history in the window
        cov = estimate_covariance(window[live], "shrinkage", lookback, annualise=True)
        table = euler_risk_contributions(weights, cov)
        sigma = float(np.sqrt(weights.reindex(cov.columns).to_numpy() @ cov.to_numpy() @ weights.reindex(cov.columns).to_numpy()))
        identity = max(identity, abs(float(table["component"].sum()) - sigma))
        rows.append(table["share"].rename(date))
        class_rows.append(table["share"].groupby(lambda c: sector_of[c]).sum().rename(date))
    shares = pd.DataFrame(rows)
    class_shares = pd.DataFrame(class_rows)
    return {"asset_shares": shares, "class_shares": class_shares, "identity_error": identity,
            "mean_asset_share": shares.mean(), "mean_class_share": class_shares.mean(),
            "asset_share_dispersion": float(shares.std(axis=1).mean())}


# ---------------------------------------------------------------------------
# Strategy sleeves
# ---------------------------------------------------------------------------
def sleeve_attribution(runs: dict, members: list[str]) -> dict:
    """How much of the Stage 10 inverse-volatility blend each member strategy supplied (Carino, daily)."""
    streams = pd.DataFrame({m: runs[m].net_returns for m in members})
    combined, weights = combine_strategy_returns(streams, "inverse_vol", 252)
    valid = combined.notna()
    contributions = (weights * streams).fillna(0.0)[valid]
    linked = link_effects(contributions, combined[valid])
    error = abs(float(linked.sum()) - cumulative(combined[valid]))
    return {"linked": linked, "error": error, "cumulative": cumulative(combined[valid]),
            "mean_weight": weights[valid].mean(), "combined": combined[valid]}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_brinson(attributions: dict, class_weights: dict, benchmark_class_weight: pd.Series, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    books = list(attributions)
    ax = axes[0, 0]
    x = np.arange(len(books))
    parts = [("allocation", "#0072B2"), ("selection", "#D55E00"), ("interaction", "#999999")]
    bottoms_pos, bottoms_neg = np.zeros(len(books)), np.zeros(len(books))
    for name, colour in parts:
        values = np.array([attributions[b]["linked"][name].sum() for b in books]) * 100
        bottoms = np.where(values >= 0, bottoms_pos, bottoms_neg)
        ax.bar(x, values, bottom=bottoms, color=colour, label=name, width=0.6)
        bottoms_pos += np.where(values >= 0, values, 0)
        bottoms_neg += np.where(values < 0, values, 0)
    totals = np.array([attributions[b]["cumulative_active_return"] for b in books]) * 100
    ax.scatter(x, totals, color="black", zorder=5, s=45, label="cumulative active return")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [SHORT[b] for b in books])
    ax.set_ylabel("Linked effect, percentage points")
    ax.set_title("Active return vs equal weight, by effect")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.16 * (high - low))   # head-room for the legend above the bars
    ax.legend(fontsize=8, loc="upper center", ncol=4, frameon=False)

    ax = axes[0, 1]
    classes = list(benchmark_class_weight.index)
    width = 0.8 / len(books)
    for j, book in enumerate(books):
        values = attributions[book]["linked"]["allocation"].reindex(classes).fillna(0.0) * 100
        ax.bar(np.arange(len(classes)) + (j - (len(books) - 1) / 2) * width, values, width=width,
               color=PALETTE[j], label=SHORT[book])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(classes)), classes, rotation=15)
    ax.set_ylabel("Linked allocation effect, pp")
    ax.set_title("Which asset-class bets paid or cost")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for j, book in enumerate(books):
        active = (class_weights[book] - benchmark_class_weight).reindex(classes) * 100
        ax.bar(np.arange(len(classes)) + (j - (len(books) - 1) / 2) * width, active, width=width,
               color=PALETTE[j], label=SHORT[book])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(classes)), classes, rotation=15)
    ax.set_ylabel("Average weight minus equal weight, pp")
    ax.set_title("What each allocator holds differently")

    ax = axes[1, 1]
    for j, book in enumerate(books):
        result = attributions[book]["result"]
        active = result.active_return()
        ax.plot(active.index.to_timestamp(), (active.cumsum() * 100).to_numpy(), color=PALETTE[j], linewidth=1.3,
                label=SHORT[book])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Sum of monthly active returns, pp (unlinked)")
    ax.set_title("When the active return arrived")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 40. Where the active return came from", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, path, "Is the difference between risk-based allocators and equal weight explained by "
                           "asset-class allocation or by selection within classes, and which bets paid?", 40)


def figure_risk_cost(risk: dict, contributions: dict, costs: pd.DataFrame, sleeves: dict, window, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    classes = list(CLASS_COLOURS)
    ax = axes[0, 0]
    books = list(risk)
    bottoms = np.zeros(len(books))
    for cls in classes:
        values = np.array([float(risk[b]["mean_class_share"].get(cls, 0.0)) for b in books]) * 100
        ax.barh(np.arange(len(books)), values, left=bottoms, color=CLASS_COLOURS[cls], label=cls)
        bottoms += values
    ax.set_yticks(np.arange(len(books)), [SHORT[b] for b in books])
    ax.invert_yaxis()
    ax.set_xlabel("Share of forecast volatility, % (average over month-ends)")
    ax.set_title("Where the risk sits, by asset class (Euler components)")
    
    ax = axes[0, 1]
    books = list(contributions)
    pos, neg = np.zeros(len(books)), np.zeros(len(books))
    for cls in classes:
        values = np.array([float(contributions[b]["linked"].get(cls, 0.0)) for b in books]) * 100
        bottoms = np.where(values >= 0, pos, neg)
        ax.bar(np.arange(len(books)), values, bottom=bottoms, color=CLASS_COLOURS[cls], label=cls, width=0.6)
        pos += np.where(values >= 0, values, 0)
        neg += np.where(values < 0, values, 0)
    ax.scatter(np.arange(len(books)), [contributions[b]["cumulative"] * 100 for b in books], color="black", zorder=5,
               s=40, label="total return")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(books)), [SHORT[b] for b in books])
    ax.set_ylabel("Linked contribution, pp")
    ax.set_title(f"Where the return came from (linked, gross, {len(window)} months from {window[0]})")

    ax = axes[1, 0]
    bottoms = np.zeros(len(costs))
    for cls in classes:
        if cls in costs.columns:
            values = costs[cls].to_numpy()
            ax.bar(np.arange(len(costs)), values, bottom=bottoms, color=CLASS_COLOURS[cls], label=cls, width=0.6)
            bottoms += values
    ax.set_xticks(np.arange(len(costs)), [SHORT[b] for b in costs.index], rotation=15)
    ax.set_ylabel("Transaction cost, basis points a year")
    ax.set_title("What trading costs, by asset class")

    ax = axes[1, 1]
    linked = sleeves["linked"]
    ax.bar(np.arange(len(linked)), linked.to_numpy() * 100, color=[PALETTE[i] for i in range(len(linked))])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(linked)), [SHORT[s] for s in linked.index], rotation=15)
    for i, (weight, value) in enumerate(zip(sleeves["mean_weight"].reindex(linked.index), linked)):
        ax.text(i, value * 100, f"avg weight {100 * weight:.0f}%", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=8)
    ax.set_ylabel("Linked contribution, pp")
    ax.set_title(f"Sleeves of the inverse-volatility blend (total {100 * sleeves['cumulative']:.1f}%)")
    fig.suptitle("Figure 41. Where the risk, the return and the cost came from", fontsize=13, fontweight="bold")
    handles = [Patch(facecolor=CLASS_COLOURS[c], label=c) for c in classes]
    handles.append(Line2D([], [], marker="o", color="black", linestyle="", markersize=6, label="total return"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    save_figure(fig, path, "Which asset classes carry the risk, the return and the transaction cost of each "
                           "allocator, and which sleeve supplied the combined strategy's return?", 41)


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 20: portfolio attribution").parse_args(argv)
    context, logger = build_context(STAGE, generation=2)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 20 | portfolio attribution (Generation 2, Priority 8)")
    logger.info("=" * 72)
    node = cfg.get("gen2_portfolio.attribution", {}) or {}
    benchmark = str(node.get("benchmark", BENCHMARK))
    market = context.market_data()
    returns = market.returns()
    sector_of = dict(cfg.asset_class_map)
    engine = BacktestEngine.from_config(cfg)

    include = [benchmark] + ATTRIBUTED + ["M3_momentum", "M4_mean_reversion", "M9_mean_cvar"]
    books = cached_ladder(market, cfg, context.processed, list(dict.fromkeys(include + SLEEVES)))
    runs = {name: engine.run(w, returns, name, market.investable, apply_vol_target=name.startswith(("M3", "M4", "M5")))
            for name, w in books.items()}

    # ---------------------------------------------------------------- Brinson-Fachler
    window = common_window({b: runs[b] for b in [benchmark] + ATTRIBUTED}, returns)
    logger.info("common window: %d months, %s to %s (every book fully invested)", len(window), window[0], window[-1])
    attributions = {b: attribute_against(runs[benchmark], runs[b], returns, sector_of, window) for b in ATTRIBUTED}
    rows, summary = [], []
    for book, item in attributions.items():
        linked = item["linked"]
        for cls, row in linked.iterrows():
            rows.append({"book": book, "class": cls, **row.to_dict()})
        summary.append({"book": book, "months": item["months"], "cumulative_return": item["portfolio_cumulative"],
                        "benchmark_cumulative_return": item["benchmark_cumulative"],
                        "cumulative_active_return": item["cumulative_active_return"],
                        "annual_volatility": item["volatility"],
                        "benchmark_annual_volatility": item["benchmark_volatility"],
                        "allocation": float(linked["allocation"].sum()), "selection": float(linked["selection"].sum()),
                        "interaction": float(linked["interaction"].sum()),
                        **{f"check_{k}": v for k, v in item["checks"].items()}})
    context.save_table(pd.DataFrame(rows), "stage20_brinson_linked_by_class.csv", index=False, float_format="%.8f")
    summary = pd.DataFrame(summary).set_index("book")
    context.save_table(summary, "stage20_active_return_summary.csv", float_format="%.8f")
    logger.info("Brinson-Fachler against %s, Carino-linked (gross):\n%s", benchmark,
                summary[["cumulative_active_return", "allocation", "selection", "interaction"]].round(4).to_string())
    class_weights = {benchmark: attributions[ATTRIBUTED[0]]["result"].benchmark_sector_weight.mean()}
    class_weights.update({b: attributions[b]["result"].portfolio_sector_weight.mean() for b in ATTRIBUTED})
    context.save_table(pd.DataFrame(class_weights), "stage20_average_class_weights.csv")

    # ---------------------------------------------------------------- contributions, costs, risk
    contributions = {b: contributions_by_class(runs[b], returns, sector_of, window) for b in [benchmark] + ATTRIBUTED}
    context.save_table(pd.DataFrame({b: c["linked"] for b, c in contributions.items()}),
                       "stage20_return_contribution_by_class.csv", float_format="%.8f")
    cost_books = [benchmark] + ATTRIBUTED + ["M3_momentum", "M4_mean_reversion"]
    costs = pd.DataFrame({b: cost_attribution(runs[b], engine, sector_of, window) for b in cost_books}).T.fillna(0.0)
    context.save_table(costs, "stage20_cost_by_class_bp_per_year.csv")
    risk = {b: risk_attribution(runs[b], returns, sector_of, window) for b in [benchmark] + ATTRIBUTED}
    context.save_table(pd.DataFrame({b: r["mean_class_share"] for b, r in risk.items()}), "stage20_risk_share_by_class.csv")
    context.save_table(pd.DataFrame({b: r["mean_asset_share"] for b, r in risk.items()}), "stage20_risk_share_by_asset.csv")
    logger.info("risk share by class (mean over month-ends):\n%s",
                pd.DataFrame({SHORT[b]: r["mean_class_share"] for b, r in risk.items()}).round(3).to_string())
    logger.info("dispersion of asset risk shares (std across assets, mean over months): %s",
                {SHORT[b]: round(r["asset_share_dispersion"], 4) for b, r in risk.items()})

    sleeves = sleeve_attribution(runs, SLEEVES)
    context.save_table(pd.DataFrame({"linked_contribution": sleeves["linked"], "mean_weight": sleeves["mean_weight"]}),
                       "stage20_sleeve_attribution.csv", float_format="%.8f")
    logger.info("sleeve attribution of the blend (cumulative %.4f):\n%s", sleeves["cumulative"], sleeves["linked"].round(4).to_string())

    # ---------------------------------------------------------------- identities
    checks = {f"BF {b} {k}": v for b, item in attributions.items() for k, v in item["checks"].items()}
    checks.update({f"contribution {b}": c["error"] for b, c in contributions.items()})
    # the two analyses cover the same months, so a book's total return must agree between them
    checks.update({f"window agreement {b}": abs(contributions[b]["cumulative"] - attributions[b]["portfolio_cumulative"])
                   for b in ATTRIBUTED})
    checks["window agreement benchmark"] = abs(contributions[benchmark]["cumulative"]
                                               - attributions[ATTRIBUTED[0]]["benchmark_cumulative"])
    checks.update({f"risk {b} Euler sum": r["identity_error"] for b, r in risk.items()})
    checks["sleeve linked sum"] = sleeves["error"]
    context.save_table(pd.Series(checks, name="max_abs_error").to_frame(), "stage20_identity_checks.csv", float_format="%.3e")
    worst = max(checks.values())
    logger.info("identity checks: %d, worst error %.2e", len(checks), worst)
    if worst > TOLERANCE:
        offenders = {k: v for k, v in checks.items() if v > TOLERANCE}
        raise AssertionError(f"attribution identities failed: {offenders}")

    # ---------------------------------------------------------------- figures and registry
    benchmark_weights = class_weights[benchmark]
    figure_brinson(attributions, class_weights, benchmark_weights, context.figure("fig40_brinson_attribution.png"))
    figure_risk_cost(risk, contributions, costs, sleeves, window, context.figure("fig41_risk_cost_sleeves.png"))
    log_registry(context, cfg, attributions, summary, class_weights, benchmark_weights, risk, costs, sleeves, checks, benchmark)
    logger.info("STAGE 20 complete")
    return 0


def log_registry(context, cfg, attributions, summary, class_weights, benchmark_weights, risk, costs, sleeves, checks,
                 benchmark) -> None:
    registry = context.registry
    registry.log(
        "Every attribution identity (Brinson-Fachler per period, Carino-linked totals, return contributions, Euler risk "
        "components, sleeve contributions, reconciliation to the engine's own returns) holds to machine precision.",
        stage=STAGE, parameters={"tolerance": TOLERANCE, "checks": len(checks)},
        results={"worst_error": float(max(checks.values())), "n_checks": len(checks)},
        decision="record",
        notes=("The stage raises if any identity exceeds 1e-9, so a recorded entry means they all held. The reconciliation "
               "check compares start-of-month weights times monthly asset returns with the engine's compounded monthly "
               "return; it only closes because the engine's weight drift is timed correctly (see the Generation 2 report)."),
    )
    for book, row in summary.iterrows():
        linked = attributions[book]["linked"]
        biggest = linked["allocation"].abs().idxmax()
        active = (class_weights[book] - benchmark_weights)
        registry.log(
            f"Descriptive: where the gross active return of {SHORT[book]} against equal weight came from.",
            stage=STAGE, parameters={"benchmark": benchmark, "classes": list(linked.index), "linking": "Carino"},
            results={"cumulative_active_return": float(row["cumulative_active_return"]),
                     "allocation": float(row["allocation"]), "selection": float(row["selection"]),
                     "interaction": float(row["interaction"]),
                     "largest_allocation_class": str(biggest),
                     "largest_allocation_effect": float(linked.loc[biggest, "allocation"]),
                     "average_active_weight_by_class": {k: float(v) for k, v in active.items()}},
            decision="record",
            notes=(f"Over {int(row['months'])} months the linked effects sum to the cumulative active return "
                   f"({100 * float(row['cumulative_active_return']):+.2f} percentage points): allocation "
                   f"{100 * float(row['allocation']):+.2f}, selection {100 * float(row['selection']):+.2f}, interaction "
                   f"{100 * float(row['interaction']):+.2f}. The class with the largest allocation effect is {biggest}. "
                   f"Gross and NOT risk-adjusted: over these months the book's annualised volatility was "
                   f"{100 * float(row['annual_volatility']):.1f}% and the benchmark's "
                   f"{100 * float(row['benchmark_annual_volatility']):.1f}%."),
        )
    rp = risk["M2_risk_parity"]
    registry.log(
        "Descriptive: how equally do risk parity, HRP and HERC spread the forecast risk across assets?",
        stage=STAGE, parameters={"covariance": "shrinkage, 252-day", "averaged": "month-ends"},
        results={f"asset_risk_share_dispersion_{b}": float(r["asset_share_dispersion"]) for b, r in risk.items()}
        | {f"largest_class_risk_share_{b}": float(r["mean_class_share"].max()) for b, r in risk.items()},
        decision="record",
        notes=("Risk parity's dispersion of asset risk shares is the number that should be near zero if the optimiser "
               f"did its job ({float(rp['asset_share_dispersion']):.4f} here, up to the difference between the estimation "
               "window and the holding month and the long-only caps). HRP and HERC do not target equal risk contributions "
               "and their dispersion shows how far they are from it."),
    )
    registry.log(
        "Descriptive: which sleeve supplied the return of the Stage 10 inverse-volatility blend?",
        stage=STAGE, parameters={"sleeves": SLEEVES, "linking": "Carino, daily"},
        results={"cumulative_return": float(sleeves["cumulative"]),
                 "linked_contribution": {k: float(v) for k, v in sleeves["linked"].items()},
                 "mean_weight": {k: float(v) for k, v in sleeves["mean_weight"].items()}},
        decision="record",
        notes="Linked contributions sum exactly to the blend's cumulative net return.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
