"""Stage 22 - Online learning (Generation 3, Priority 10).

Train, freeze, test becomes train, trade, update, trade, update. Two questions
with rules in ``config/online.yaml``, committed before any result:

1. FORECASTING. Do monthly-updating models (recursive least squares with
   forgetting, normalised LMS, a random-walk Kalman filter) forecast 21-day ETF
   returns better than Stage 19's annually refitted ridge? Same features, same
   volatility forecast, same origins: only the update scheme differs.
2. AGGREGATION. Does Hedge over the strategy sleeves beat the Stage 10
   inverse-volatility blend, and what is its regret against the best sleeve?

Figures 44-45.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import performance_summary
from src.features.sleeves import month_end_dates
from src.models.online import KalmanRW, NLMS, OnlineRidge, hedge_aggregate, run_online_forecasts
from src.models.probabilistic import build_panel, crps_gaussian, design_matrix
from src.signals.combine import combine_strategy_returns
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg, diebold_mariano
from src.validation.robustness import paired_sharpe_test
from experiments.context import build_context
from experiments.strategies import cached_ladder

STAGE = "stage22_online"
EXPERTS = ["M1_inverse_vol", "M2_risk_parity", "M3_momentum", "M4_mean_reversion", "M9_mean_cvar"]
SHORT = {"M1_inverse_vol": "inverse vol", "M2_risk_parity": "risk parity", "M3_momentum": "momentum",
         "M4_mean_reversion": "mean reversion", "M9_mean_cvar": "mean-CVaR"}


def online_predictions(panel, settings: dict, models_cfg: dict, horizon: int, min_train: int, forgetting_override=None):
    """Predictions of every online model at every origin from the first test origin on, keyed by (origin, asset)."""
    frame = panel.frame.reset_index(drop=True)
    positions = np.array([panel.positions[o] for o in frame["origin"]])
    first_test = min(p for p in set(positions.tolist()) if p >= min_train)
    warm_mask = (positions + horizon <= first_test) & frame["target"].notna()
    warm = frame[warm_mask].dropna(subset=panel.price_columns)
    _, stats_ = design_matrix(warm, panel, "price_only")
    design, _ = design_matrix(frame, panel, "price_only", stats_)
    target = frame["target"].to_numpy(dtype=float)
    # residual scale of the warm start, for the Kalman observation variance
    x_warm, _ = design_matrix(warm, panel, "price_only", stats_)
    y_warm = warm["target"].to_numpy(dtype=float)
    beta0 = np.linalg.solve(x_warm.T @ x_warm + 100.0 * np.eye(x_warm.shape[1]), x_warm.T @ y_warm)
    obs_var = float(np.var(y_warm - x_warm @ beta0))
    dim = design.shape[1]
    rls = models_cfg.get("rls", {})
    forgetting = float(forgetting_override if forgetting_override is not None else rls.get("forgetting_per_month", 0.99))
    factories = {
        "rls": lambda: OnlineRidge(dim, forgetting, float(rls.get("ridge", 100.0))),
        "nlms": lambda: NLMS(dim, float((models_cfg.get("nlms", {}) or {}).get("step", 0.1))),
        "kalman": lambda: KalmanRW(dim, float((models_cfg.get("kalman", {}) or {}).get("process_to_observation_noise", 1e-3)),
                                   observation_variance=obs_var, prior_variance=1e-4),
    }
    codes, uniques = pd.factorize(frame["origin"])                    # origins are in chronological order
    test_origins = [(int(c), panel.positions[o]) for c, o in enumerate(uniques) if panel.positions[o] >= first_test]
    out = run_online_forecasts(design, target, codes.astype(int), positions, first_test, horizon, factories, test_origins)
    frames = {}
    for name, pred in out.items():
        merged = pred.drop(columns="origin").merge(frame[["origin", "asset", "sigma", "target"]], left_on="row", right_index=True)
        frames[name] = merged[["origin", "asset", "mu", "sigma", "target"]]
    return frames


def monthly_scores(pred: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    """Cross-asset monthly mean CRPS and squared error of an online model and the frozen benchmark on identical rows."""
    merged = pred.merge(frozen[["origin", "asset", "mu"]].rename(columns={"mu": "mu_frozen"}), on=["origin", "asset"]).dropna(
        subset=["mu", "mu_frozen", "sigma", "target"])
    y, s = merged["target"].to_numpy(), merged["sigma"].to_numpy()
    scored = pd.DataFrame({
        "origin": merged["origin"],
        "crps_model": crps_gaussian(y, merged["mu"].to_numpy(), s),
        "crps_frozen": crps_gaussian(y, merged["mu_frozen"].to_numpy(), s),
        "se_model": (y - merged["mu"].to_numpy()) ** 2, "se_frozen": (y - merged["mu_frozen"].to_numpy()) ** 2,
    })
    return scored.groupby("origin").mean(), merged


def mean_ic(merged: pd.DataFrame, column: str) -> float:
    ics = []
    for _, g in merged.groupby("origin"):
        if len(g) >= 5 and g[column].nunique() > 1:
            ics.append(spearmanr(g[column], g["target"])[0])
    return float(np.nanmean(ics))


def figure_forecasting(scores: dict, table: pd.DataFrame, sweep: pd.DataFrame, path):
    fig, axes = new_axes(1, 3, figsize=(16.5, 5.2))
    ax = axes[0]
    for i, (name, s) in enumerate(scores.items()):
        diff = (s["crps_model"] - s["crps_frozen"]).cumsum()
        ax.plot(pd.to_datetime(diff.index), diff.to_numpy(), color=PALETTE[i], label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative CRPS difference (online minus frozen)")
    ax.set_title("Below zero: the online model is ahead")
    ax.legend(fontsize=8)
    ax = axes[1]
    x = np.arange(len(table))
    ax.bar(x, table["mean_crps_difference"] * 1e4, color=PALETTE[: len(table)])
    for xi, (_, r) in zip(x, table.iterrows()):
        ax.text(xi, r["mean_crps_difference"] * 1e4, f"p={r['p_value']:.2f}" + ((" *worse" if r["mean_crps_difference"] > 0 else " *better") if r["bh_significant"] else ""), ha="center",
                va="bottom" if r["mean_crps_difference"] >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, table.index)
    ax.set_ylabel("Mean CRPS difference, basis points of return")
    ax.set_title("Pre-declared family of three (BH)")
    ax = axes[2]
    ax.plot(sweep.index.astype(str), sweep["mean_crps_difference"] * 1e4, marker="o", color=PALETTE[0])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("RLS forgetting factor per month (post-hoc)")
    ax.set_ylabel("Mean CRPS difference, bp")
    ax.set_title("How much does forgetting matter? (not judged)")
    fig.suptitle("Figure 44. Do models that update every month forecast better than a model refit once a year?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Do online-updating models (recursive least squares, normalised LMS, a Kalman filter) forecast "
                           "21-day ETF returns better than the annually refitted ridge, with the same features and volatility?", 44)


def figure_aggregation(hedge, curves: dict, path):
    fig, axes = new_axes(1, 3, figsize=(16.5, 5.2))
    ax = axes[0]
    w = hedge.weights.resample("ME").last()
    ax.stackplot(w.index, [w[c].to_numpy() for c in w.columns], labels=[SHORT[c] for c in w.columns], colors=PALETTE[: w.shape[1]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hedge weight")
    ax.set_title("What the online rule trusts, through time")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    ax = axes[1]
    for i, (name, s) in enumerate(curves.items()):
        ax.plot(s.index, s.to_numpy(), color=PALETTE[i], label=name)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (net, log scale)")
    ax.set_title("Aggregates of the same five sleeves")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(hedge.regret.index, hedge.regret.to_numpy(), color=PALETTE[0], label="regret vs best sleeve")
    ax.plot(hedge.regret_bound.index, hedge.regret_bound.to_numpy(), color="black", linestyle="--", label="theoretical envelope")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative regret (gain units)")
    ax.set_title("Regret against the best sleeve in hindsight")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 45. Does learning how much to trust each strategy beat a fixed rule?", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path, "Does an online expert-aggregation rule over the strategy sleeves beat the inverse-volatility blend, "
                           "and how large is its regret against the best sleeve in hindsight?", 45)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 22: online learning").parse_args(argv)
    context, logger = build_context(STAGE, generation=3)
    cfg = context.config
    node = cfg.get("online", {}) or {}
    logger.info("=" * 72)
    logger.info("STAGE 22 | online learning (Generation 3, Priority 10)")
    logger.info("=" * 72)
    decisions = node.get("decisions", {}) or {}
    fdr = float(decisions.get("fdr", 0.10))
    boot = decisions.get("bootstrap", {}) or {}
    market = context.market_data()
    prices, returns = market.prices, market.returns()
    index = pd.DatetimeIndex(prices.index)
    horizon = int(cfg.get("forecasting.horizon_days", 21))
    min_train = int(((node.get("forecasting", {}) or {}).get("warm_start_days", 1260)))
    sleeve_of = {a: s for s, members in (cfg.get("forecasting.sleeves", {}) or {}).items() for a in members}
    origins = month_end_dates(index)
    panel = build_panel(prices, origins, horizon, float(cfg.get("forecasting.gaussian.vol_halflife", 40)),
                        pd.DataFrame(index=index), sleeve_of)

    frozen_path = context.tables / "stage19_predictions_price_only.csv"
    if not frozen_path.exists():
        raise FileNotFoundError("Stage 19 must run first: its price-only predictions are this stage's benchmark")
    frozen = pd.read_csv(frozen_path, parse_dates=["origin"])

    # ---------------------------------------------------------------- 1. forecasting
    models_cfg = (node.get("forecasting", {}) or {}).get("models", {}) or {}
    preds = online_predictions(panel, node, models_cfg, horizon, min_train)
    scores, rows = {}, []
    for name, pred in preds.items():
        s, merged = monthly_scores(pred, frozen)
        scores[name] = s
        test = diebold_mariano(s["crps_model"], s["crps_frozen"], lag=0, alternative="two-sided")
        rows.append({"model": name, "n_origins": len(s), "mean_crps_model": float(s["crps_model"].mean()),
                     "mean_crps_frozen": float(s["crps_frozen"].mean()),
                     "mean_crps_difference": float(test["mean_loss_difference"]), "statistic": float(test["statistic"]),
                     "p_value": float(test["p_value"]), "mse_ratio": float(s["se_model"].mean() / s["se_frozen"].mean()),
                     "mean_ic_model": mean_ic(merged, "mu"), "mean_ic_frozen": mean_ic(merged, "mu_frozen")})
    table = pd.DataFrame(rows).set_index("model")
    table["bh_significant"] = benjamini_hochberg(table["p_value"], fdr)
    table["passes"] = table["bh_significant"] & (table["mean_crps_difference"] < 0)
    context.save_table(table, "stage22_forecast_tests.csv")
    for name, s in scores.items():
        context.save_table(s, f"stage22_scores_by_origin_{name}.csv")
    logger.info("online against the annually refitted ridge (CRPS, lower is better):\n%s",
                table[["mean_crps_model", "mean_crps_frozen", "mean_crps_difference", "p_value", "bh_significant", "mse_ratio",
                       "mean_ic_model", "mean_ic_frozen"]].round(6).to_string())

    # post-hoc: the forgetting factor
    sweep_rows = {}
    for lam in (0.95, 0.97, 0.99, 0.995, 1.0):
        p = online_predictions(panel, node, models_cfg, horizon, min_train, forgetting_override=lam)["rls"]
        s, _ = monthly_scores(p, frozen)
        t = diebold_mariano(s["crps_model"], s["crps_frozen"], lag=0, alternative="two-sided")
        sweep_rows[lam] = {"mean_crps_difference": float(t["mean_loss_difference"]), "p_value": float(t["p_value"])}
    sweep = pd.DataFrame(sweep_rows).T
    context.save_table(sweep, "stage22_posthoc_forgetting_sweep.csv")
    logger.info("post-hoc forgetting sweep:\n%s", sweep.round(6).to_string())

    # ---------------------------------------------------------------- 2. aggregation
    engine = BacktestEngine.from_config(cfg)
    books = cached_ladder(market, cfg, context.processed, EXPERTS)
    streams = pd.DataFrame({n: engine.run(w, returns, n, market.investable,
                                          apply_vol_target=n.startswith(("M3", "M4", "M5"))).net_returns
                            for n, w in books.items()}).dropna(how="any")
    agg = node.get("aggregation", {}) or {}
    hedge = hedge_aggregate(streams, float((agg.get("hedge", {}) or {}).get("gain_scale", 0.01)))
    ivol, _ = combine_strategy_returns(streams, "inverse_vol", int(cfg.get("strategies.combination.vol_lookback", 252)))
    equal, _ = combine_strategy_returns(streams, "equal")
    common_index = hedge.returns.index.intersection(ivol.dropna().index).intersection(equal.dropna().index)
    series = {"Hedge": hedge.returns.loc[common_index], "inverse-vol blend": ivol.loc[common_index], "equal blend": equal.loc[common_index]}
    for name in EXPERTS:
        series[SHORT[name]] = streams[name].loc[common_index]
    perf = pd.DataFrame({n: performance_summary(s) for n, s in series.items()}).T
    context.save_table(perf, "stage22_aggregation_performance.csv")
    logger.info("aggregation, net, common window from %s:\n%s", common_index[0].date(),
                perf[["cagr", "ann_vol", "sharpe", "max_drawdown"]].astype(float).round(4).to_string())
    tests = []
    for other in ("inverse-vol blend", "equal blend"):
        t = paired_sharpe_test(series["Hedge"], series[other], n_samples=int(boot.get("n_samples", 2000)),
                               block_length=int(boot.get("block_length", 21)), seed=int(boot.get("seed", 7)))
        tests.append({"a": "Hedge", "b": other, **t})
    best = max(EXPERTS, key=lambda n: performance_summary(streams[n].loc[common_index])["sharpe"])
    t = paired_sharpe_test(series["Hedge"], streams[best].loc[common_index], n_samples=int(boot.get("n_samples", 2000)),
                           block_length=int(boot.get("block_length", 21)), seed=int(boot.get("seed", 7)))
    tests.append({"a": "Hedge", "b": SHORT[best] + " (best sleeve in hindsight)", **t})
    tests = pd.DataFrame(tests)
    context.save_table(tests, "stage22_aggregation_tests.csv", index=False)
    logger.info("paired Sharpe tests:\n%s", tests[["a", "b", "sharpe_a", "sharpe_b", "difference", "p_value"]].round(4).to_string(index=False))
    context.save_table(hedge.weights.resample("ME").last(), "stage22_hedge_weights_monthly.csv")
    regret = pd.DataFrame({"regret": hedge.regret, "bound": hedge.regret_bound})
    context.save_table(regret.resample("ME").last(), "stage22_regret.csv")
    logger.info("Hedge learning rate %.4f; final regret %.2f against envelope %.2f", hedge.learning_rate, hedge.regret.iloc[-1],
                hedge.regret_bound.iloc[-1])

    # --------------------------------------------------------------- figures
    curves = {n: (1.0 + s).cumprod() for n, s in list(series.items())[:3]}
    figure_forecasting(scores, table, sweep, context.figure("fig44_online_forecasting.png"))
    figure_aggregation(hedge, curves, context.figure("fig45_online_aggregation.png"))

    # -------------------------------------------------------------- registry
    reg = context.registry
    any_better = bool(table["passes"].any())
    reg.log(
        "Models that update every month forecast 21-day ETF returns better (lower CRPS) than the annually refitted ridge.",
        stage=STAGE, parameters={"models": models_cfg, "test": "Diebold-Mariano on monthly mean CRPS differences, BH across three"},
        results={**{f"crps_difference_{m}": float(table.loc[m, "mean_crps_difference"]) for m in table.index},
                 **{f"p_value_{m}": float(table.loc[m, "p_value"]) for m in table.index},
                 "mean_crps_frozen": float(table["mean_crps_frozen"].iloc[0]), "n_origins": int(table["n_origins"].iloc[0])},
        decision="retain" if any_better else "reject", test_period="2011-01 onward, monthly origins",
        notes=("Retained only if at least one online model has a significantly lower CRPS after BH control. Same features, same "
               "volatility forecast and same origins as Stage 19, so only the update scheme differs. The forgetting sweep is post-hoc. "
               "Implementation choices not in the declaration: the Kalman filter's observation variance is the warm-start residual "
               "variance and its prior coefficient variance is 1e-4."),
    )
    h = tests.iloc[0]
    reg.log(
        "A Hedge aggregate of the strategy sleeves has a higher net Sharpe than the Stage 10 inverse-volatility blend.",
        stage=STAGE, parameters={"experts": EXPERTS, "learning_rate": hedge.learning_rate, "test": "paired stationary bootstrap"},
        results={"sharpe_hedge": float(h["sharpe_a"]), "sharpe_inverse_vol_blend": float(h["sharpe_b"]),
                 "difference": float(h["difference"]), "p_value": float(h["p_value"]),
                 "sharpe_equal_blend": float(tests.iloc[1]["sharpe_b"]), "final_regret": float(hedge.regret.iloc[-1]),
                 "regret_envelope": float(hedge.regret_bound.iloc[-1])},
        decision="retain" if (h["difference"] > 0 and h["p_value"] < 0.10) else "reject",
        test_period=f"{common_index[0].date()} onward, net of costs",
        notes=("Sleeve returns are the net streams of Stage 10; re-weighting between sleeves is costless, as in Stage 10. "
               "Regret is against the best single sleeve in hindsight, in units of the 1% gain scale."),
    )
    logger.info("STAGE 22 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
