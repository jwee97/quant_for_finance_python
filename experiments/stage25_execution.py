"""Stage 25 - Execution model: market impact, capacity and no-trade bands (Generation 3, Priority 7).

Generation 1 charged a linear per-asset rate and said so. Here every book is re-costed with spread and
commission PLUS square-root market impact sized by the fund's own assets against each ETF's average daily
dollar volume (``config/execution.yaml``). The model reduces exactly to Generation 1 as AUM -> 0. Two
questions were declared in advance:

  h_capacity  at $1bn, is every allocator's net Sharpe within 0.05 of its linear-cost Sharpe?
  h_bands     does a 1% no-trade band improve the net Sharpe of each of the three alpha books at $1bn?

Figures 50-51.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.impact import (
    ImpactSettings, banded_execution, banded_gross_returns, capacity_curve, capacity_from_curve, fill_counts,
    impact_net_returns, market_state, scheduling_table, sharpe)
from src.backtest.metrics import performance_summary
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg
from src.validation.robustness import paired_sharpe_test
from experiments.context import build_context
from experiments.strategies import cached_ladder

STAGE = "stage25_execution"
ALLOCATORS = ["M0_equal_weight", "M1_inverse_vol", "M2_risk_parity", "M9_mean_cvar", "M11_hrp", "M12_herc"]
ALPHAS = ["M3_momentum", "M4_mean_reversion", "M5_momentum_plus_mr"]
COMBOS = ["equal", "cost_aware", "hedge"]
SHORT = {"M0_equal_weight": "M0 equal", "M1_inverse_vol": "M1 inv-vol", "M2_risk_parity": "M2 RP", "M9_mean_cvar": "M9 CVaR",
         "M11_hrp": "M11 HRP", "M12_herc": "M12 HERC", "M3_momentum": "M3 momentum", "M4_mean_reversion": "M4 reversion",
         "M5_momentum_plus_mr": "M5 mom+rev", "equal": "combo equal", "cost_aware": "combo cost-aware", "hedge": "combo Hedge"}
COLOUR = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(SHORT)}


def figure_capacity(curves: pd.DataFrame, linear: pd.Series, change: pd.Series, capacity: pd.DataFrame, drag: pd.DataFrame, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    ax = axes[0, 0]
    for name in curves.columns:
        ax.plot(curves.index, curves[name].to_numpy(), color=COLOUR[name], linewidth=1.4, label=SHORT[name])
    ax.axvline(1e9, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Assets under management, USD (log)")
    ax.set_ylabel("Net Sharpe, impact-inclusive")
    ax.set_title("What size does to every book")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[0, 1]
    ax.bar(range(len(change)), change.to_numpy(), color=[COLOUR[n] for n in change.index])
    ax.axhline(-0.05, color="#CC0000", linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(change)), [SHORT[n] for n in change.index], rotation=20, ha="right")
    for i, v in enumerate(change.to_numpy()):
        ax.text(i, v, f"{v:+.3f}", ha="center", va="top" if v < 0 else "bottom", fontsize=8)
    ax.set_ylabel("Net Sharpe at $1bn minus linear-cost Sharpe")
    ax.set_title("Pre-declared test: all six allocators within 0.05 (dashed)")
    ax = axes[1, 0]
    numeric = capacity["capacity_usd"].apply(lambda v: v if isinstance(v, (int, float)) else np.nan)
    top = float(np.nanmax(list(numeric.dropna()) + [1e9]))
    for i, (name, v) in enumerate(capacity["capacity_usd"].items()):
        if isinstance(v, (int, float)):
            ax.bar(i, v, color=COLOUR[name])
            ax.text(i, v, f"${v:.1e}", ha="center", va="bottom", fontsize=7, rotation=90)
        else:
            ax.bar(i, 2e8, color="none", edgecolor=COLOUR[name], hatch="//")
            ax.text(i, 2e8, str(v), ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(1e6, 4e9)
    ax.set_xticks(range(len(capacity)), [SHORT[n] for n in capacity.index], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Capacity, USD (Sharpe falls to half)")
    ax.set_title("Capacity by book (hatched = not reached or n/a)")
    ax = axes[1, 1]
    for name in drag.columns:
        ax.plot(drag.index, drag[name].to_numpy(), color=COLOUR[name], linewidth=1.3, label=SHORT[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Assets under management, USD (log)")
    ax.set_ylabel("Annual cost drag, bps")
    ax.set_title("Cost drag by size")
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Figure 50. Size and cost: impact-inclusive Sharpe, capacity and drag", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Once market impact is charged on the fund's own size, how fast does each book's net Sharpe decay with "
                           "assets under management, and at what size does it lose half of its linear-cost Sharpe?", 50)


def figure_bands(sweep: pd.DataFrame, tests: pd.DataFrame, schedule: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(16.5, 5.2))
    bands = sorted(sweep["band"].unique())
    for key, ax, label in (("sharpe", axes[0], "Net Sharpe at $1bn"), ("ann_turnover", axes[1], "Annual turnover (x)")):
        for name in ALPHAS:
            s = sweep[sweep["book"] == name].sort_values("band")
            ax.plot(100 * s["band"], s[key].to_numpy(), marker="o", color=COLOUR[name], label=SHORT[name])
        ax.axvline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("No-trade band, % of capital (dotted = declared 1%)")
        ax.set_ylabel(label)
        ax.set_title(label + " by band")
        ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(schedule.index, schedule["cost_bps_of_trade"].to_numpy(), marker="o", color=PALETTE[0], label="cost (bps of trade)")
    ax.plot(schedule.index, schedule["timing_risk_bps_of_trade"].to_numpy(), marker="s", color=PALETTE[1], label="timing risk, 1 s.d. (bps)")
    ax.set_xlabel("Days over which the trade is spread")
    ax.set_ylabel("bps of the trade")
    ax.set_title("Scheduling a representative trade (descriptive)")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 51. No-trade bands and trade scheduling", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does holding positions inside a no-trade band save more in impact and spread than it costs in tracking, "
                           "and how much does spreading a trade over several days trade cost for timing risk?", 51)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 25: execution model").parse_args(argv)
    context, logger = build_context(STAGE, generation=3)
    cfg = context.config
    node = cfg.get("execution", {}) or {}
    logger.info("=" * 72)
    logger.info("STAGE 25 | execution model (Generation 3, Priority 7)")
    logger.info("=" * 72)
    impact = node.get("impact", {}) or {}
    settings = ImpactSettings(float(impact.get("coefficient_Y", 1.0)), float(impact.get("max_participation", 1.0)))
    decisions = node.get("decisions", {}) or {}
    fdr = float(decisions.get("fdr", 0.10))
    boot = decisions.get("bootstrap", {}) or {}
    aum_grid = [float(a) for a in node.get("aum_grid_usd", [1e7, 1e8, 1e9, 1e10])]
    cap = node.get("capacity_grid", {}) or {}
    cap_grid = np.logspace(np.log10(float(cap.get("low", 1e6))), np.log10(float(cap.get("high", 1e11))), int(cap.get("points", 21)))
    band_node = node.get("no_trade_band", {}) or {}
    declared_band = float(band_node.get("declared_band", 0.01))
    swept = [float(b) for b in band_node.get("swept", [0.0, 0.0025, 0.005, 0.01, 0.02])]
    band_aum = float(band_node.get("evaluated_at_aum_usd", 1e9))

    market = context.market_data()
    returns = market.returns()
    engine = BacktestEngine.from_config(cfg)
    rates = engine.cost_model.rates(returns.columns)
    sigma, adv = market_state(market.close, market.volume, returns, int(impact.get("volatility_window_days", 21)),
                              int(impact.get("adv_window_days", 63)))
    logger.info("median ADV through the previous close, USD millions:\n%s", (adv.median() / 1e6).round(0).to_string())

    # ------------------------------------------------------------------ books
    books = cached_ladder(market, cfg, context.processed, ALLOCATORS + ALPHAS)
    runs = {n: engine.run(w, returns, n, market.investable, apply_vol_target=n.startswith(("M3", "M4", "M5"))) for n, w in books.items()}
    saved = context.processed / "stage24_runs"
    for name in COMBOS:
        gross_path, trades_path = saved / f"{name}_gross.csv", saved / f"{name}_trades.csv"
        if gross_path.exists() and trades_path.exists():
            gross = pd.read_csv(gross_path, index_col=0, parse_dates=[0])["gross"]
            trades = pd.read_csv(trades_path, index_col=0, parse_dates=[0])
            linear = (gross - (trades.abs() * rates.reindex(trades.columns).to_numpy()[None, :]).sum(axis=1).reindex(gross.index).fillna(0.0))
            runs[name] = type("Run", (), {"gross_returns": gross, "trades": trades, "net_returns": linear, "weights": None})()
        else:
            logger.warning("Stage 24 books for %s not found; the combination row is skipped", name)
    names = [n for n in ALLOCATORS + ALPHAS + COMBOS if n in runs]

    # ---------------------------------------------- the identity at zero AUM
    identity = {}
    for n in ALLOCATORS + ALPHAS:
        net0 = impact_net_returns(runs[n].gross_returns, runs[n].trades, 0.0, sigma, adv, rates, settings)[0]
        identity[n] = float((net0.reindex(runs[n].net_returns.index) - runs[n].net_returns).abs().max())
    logger.info("max |impact(AUM=0) - Generation 1 net| by book: %s", {k: f"{v:.1e}" for k, v in identity.items()})
    assert max(identity.values()) < 1e-12, "the model must reduce to Generation 1 at zero AUM"

    # ------------------------------------------------- every book at every AUM
    rows, part_rows, drag_rows = [], [], []
    for n in names:
        run = runs[n]
        for aum in [0.0] + aum_grid:
            net, daily, stats = impact_net_returns(run.gross_returns, run.trades, aum, sigma, adv, rates, settings)
            perf = performance_summary(net.dropna())
            rows.append({"book": n, "aum_usd": aum, "sharpe": perf.get("sharpe", np.nan), "cagr": perf.get("cagr", np.nan),
                         "max_drawdown": perf.get("max_drawdown", np.nan)})
            drag_rows.append({"book": n, "aum_usd": aum, "annual_cost_bps": 1e4 * float(daily.mean()) * 252})
            if aum > 0:
                part_rows.append({"book": n, "aum_usd": aum, **stats, "filled_trade_days": fill_counts(run.trades, sigma, adv)})
    by_aum = pd.DataFrame(rows)
    context.save_table(by_aum.pivot(index="book", columns="aum_usd", values="sharpe").reindex(names), "stage25_sharpe_by_aum.csv")
    context.save_table(by_aum.pivot(index="book", columns="aum_usd", values="cagr").reindex(names), "stage25_cagr_by_aum.csv")
    context.save_table(pd.DataFrame(drag_rows).pivot(index="book", columns="aum_usd", values="annual_cost_bps").reindex(names),
                       "stage25_cost_drag_bps_by_aum.csv")
    participation = pd.DataFrame(part_rows)
    context.save_table(participation, "stage25_participation.csv", index=False)
    logger.info("net Sharpe by AUM (0 = Generation 1):\n%s",
                by_aum.pivot(index="book", columns="aum_usd", values="sharpe").reindex(names).round(3).to_string())
    at_band = participation[participation["aum_usd"] == band_aum].set_index("book")
    logger.info("participation at $%.0e:\n%s", band_aum,
                at_band[["mean_participation", "p95_participation", "max_participation", "share_capped"]].round(4).to_string())

    # --------------------------------------------------------------- capacity
    linear_sharpe = pd.Series({n: sharpe(runs[n].net_returns) for n in names})
    curves = pd.DataFrame({n: capacity_curve(runs[n].gross_returns, runs[n].trades, cap_grid, sigma, adv, rates, settings) for n in names})
    context.save_table(curves, "stage25_capacity_curves.csv")
    capacity = pd.DataFrame({"linear_sharpe": linear_sharpe,
                             "capacity_usd": pd.Series({n: capacity_from_curve(curves[n], linear_sharpe[n]) for n in names}, dtype=object)})
    for label, y in (("capacity_usd_Y0.5", 0.5), ("capacity_usd_Y2.0", 2.0)):
        alt = ImpactSettings(y, settings.max_participation)
        capacity[label] = pd.Series({n: capacity_from_curve(
            capacity_curve(runs[n].gross_returns, runs[n].trades, cap_grid, sigma, adv, rates, alt), linear_sharpe[n]) for n in names}, dtype=object)
    context.save_table(capacity, "stage25_capacity.csv", float_format="%.6g")
    logger.info("capacity (AUM at half the linear-cost Sharpe):\n%s", capacity.to_string())
    sens_rows = []
    for y in (0.5, 1.0, 2.0):
        alt = ImpactSettings(y, settings.max_participation)
        for n in names:
            sens_rows.append({"book": n, "Y": y, "aum_usd": 1e9,
                              "sharpe": sharpe(impact_net_returns(runs[n].gross_returns, runs[n].trades, 1e9, sigma, adv, rates, alt)[0])})
    context.save_table(pd.DataFrame(sens_rows).pivot(index="book", columns="Y", values="sharpe").reindex(names), "stage25_sharpe_Y_sensitivity_1e9.csv")

    # ------------------------------------------------------ h_capacity (decision)
    at_1e9 = by_aum[by_aum["aum_usd"] == 1e9].set_index("book")["sharpe"]
    change = (at_1e9.reindex(ALLOCATORS) - linear_sharpe.reindex(ALLOCATORS))
    largest = float(change.abs().max())
    h_capacity = bool(largest <= 0.05)
    context.save_table(pd.DataFrame({"linear_sharpe": linear_sharpe.reindex(ALLOCATORS), "sharpe_1e9": at_1e9.reindex(ALLOCATORS),
                                     "change": change}), "stage25_allocator_change.csv")
    logger.info("h_capacity: largest |change| among allocators at $1bn = %.4f (rule: <= 0.05) -> %s", largest, "RETAIN" if h_capacity else "REJECT")

    # ---------------------------------------------------------------- h_bands
    sweep_rows, band_net = [], {}
    for n in ALPHAS:
        for band in swept:
            held, traded = banded_execution(runs[n].weights, runs[n].trades, returns, band)
            gross = banded_gross_returns(held, returns)
            net, daily, stats = impact_net_returns(gross.dropna(), traded, band_aum, sigma, adv, rates, settings)
            band_net[(n, band)] = net.dropna()
            lin = impact_net_returns(gross.dropna(), traded, 0.0, sigma, adv, rates, settings)[0].dropna()
            sweep_rows.append({"book": n, "band": band, "sharpe": sharpe(net), "sharpe_linear_cost": sharpe(lin),
                               "ann_turnover": float(traded.abs().sum(axis=1).mean() * 252),
                               "annual_cost_bps": 1e4 * float(daily.mean()) * 252, "gross_sharpe": sharpe(gross.dropna())})
    sweep = pd.DataFrame(sweep_rows)
    context.save_table(sweep, "stage25_band_sweep.csv", index=False)
    logger.info("band sweep at $%.0e:\n%s", band_aum, sweep.round(4).to_string(index=False))
    zero = sweep[sweep["band"] == 0.0].set_index("book")["sharpe"]
    unbanded = {n: impact_net_returns(runs[n].gross_returns, runs[n].trades, band_aum, sigma, adv, rates, settings)[0].dropna() for n in ALPHAS}
    for n in ALPHAS:                                          # band 0 must equal the unbanded engine book
        assert abs(zero[n] - sharpe(unbanded[n])) < 1e-9, "band 0 must reproduce the unbanded book"
    rows = []
    for n in ALPHAS:
        a, b = band_net[(n, declared_band)], unbanded[n]
        common = a.index.intersection(b.index)
        rows.append({"book": n, **paired_sharpe_test(a.loc[common], b.loc[common], n_samples=int(boot.get("n_samples", 2000)),
                                                     block_length=int(boot.get("block_length", 21)), seed=int(boot.get("seed", 7)))})
    band_tests = pd.DataFrame(rows)
    band_tests["bh_significant"] = benjamini_hochberg(band_tests["p_value"], fdr)
    band_tests["passes"] = band_tests["bh_significant"] & (band_tests["difference"] > 0)
    context.save_table(band_tests, "stage25_band_tests.csv", index=False)
    h_bands = bool(band_tests["passes"].all())
    logger.info("h_bands (banded - unbanded at $%.0e, BH FDR %.2f):\n%s", band_aum, fdr,
                band_tests[["book", "sharpe_a", "sharpe_b", "difference", "p_value", "bh_significant", "passes"]].round(4).to_string(index=False))

    # ------------------------------------------------------------- scheduling
    last_year = slice(returns.index[-252], None)
    sig_med, adv_med = float(sigma.loc[last_year].stack().median()), float(adv.loc[last_year].stack().median())
    nonzero = runs["M3_momentum"].trades.abs().stack()
    trade_size = float(nonzero[nonzero > 1e-9].median())
    schedule = scheduling_table(trade_size, band_aum, sig_med, adv_med, tuple(node.get("scheduling", {}).get("days", [1, 2, 5, 10])),
                                settings.coefficient, float(rates.median()))
    context.save_table(schedule, "stage25_scheduling.csv")
    logger.info("scheduling: median M3 trade %.3f of capital, median sigma %.4f, median ADV $%.0fm over the last year\n%s",
                trade_size, sig_med, adv_med / 1e6, schedule.round(6).to_string())

    # --------------------------------------------------------------- figures
    curve_cols = [n for n in names]
    drag_wide = pd.DataFrame(drag_rows).pivot(index="aum_usd", columns="book", values="annual_cost_bps").drop(index=0.0)
    figure_capacity(curves[curve_cols], linear_sharpe, change, capacity, drag_wide[[n for n in ALPHAS + ALLOCATORS if n in drag_wide]],
                    context.figure("fig50_capacity.png"))
    figure_bands(sweep, band_tests, schedule, context.figure("fig51_bands_scheduling.png"))

    # -------------------------------------------------------------- registry
    context.registry.log(
        "At $1bn of assets the impact-inclusive net Sharpe of every allocator (M0, M1, M2, M9, M11, M12) lies within 0.05 of its "
        "Generation 1 linear-cost net Sharpe.",
        stage=STAGE, parameters={"Y": settings.coefficient, "aum_usd": 1e9, "vol_window": 21, "adv_window": 63, "tolerance": 0.05},
        results={"largest_abs_change": largest, **{f"change_{n}": float(v) for n, v in change.items()},
                 **{f"sharpe_1e9_{n}": float(at_1e9[n]) for n in ALLOCATORS}},
        decision="retain" if h_capacity else "reject", test_period="full sample of each book, net of costs",
        notes="Rule is a magnitude bound, not a significance test: it asks whether size matters for the allocators at $1bn. The model "
              "reduces to Generation 1 at zero AUM (asserted to 1e-12 in the run and pinned by tests). Y=1 is an order-one coefficient from the "
              "square-root law, not an estimate; Y=0.5 and 2.0 are reported.",
    )
    context.registry.log(
        "A 1% no-trade band improves the impact-inclusive net Sharpe at $1bn of EACH of the momentum, mean-reversion and combined books.",
        stage=STAGE, parameters={"band": declared_band, "aum_usd": band_aum, "test": "paired stationary bootstrap, BH across three"},
        results={**{f"diff_{r.book}": float(r.difference) for r in band_tests.itertuples()},
                 **{f"p_{r.book}": float(r.p_value) for r in band_tests.itertuples()}},
        decision="retain" if h_bands else "reject", test_period="full sample of each book, net of impact-inclusive costs",
        notes="Retained only if all three pass after BH control at 10% with a positive difference. The band sweep and every other AUM are "
              "reported, not judged.",
    )
    context.registry.log(
        "Descriptive: capacity by book (AUM at half the linear-cost Sharpe), participation, and the Stage 24 combinations re-costed.",
        stage=STAGE, parameters={"capacity_grid_points": len(cap_grid)},
        results={f"linear_sharpe_{n}": float(v) for n, v in linear_sharpe.items()}
                | {f"sharpe_1e9_{n}": float(at_1e9[n]) for n in names if n in at_1e9.index},
        decision="record", notes="Reported, not judged; the capacity and the combination rows are the Stage 25 deliverable alongside the two decisions.",
    )
    logger.info("STAGE 25 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
