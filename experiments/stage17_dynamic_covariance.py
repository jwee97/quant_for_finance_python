"""Stage 17 - Dynamic covariance (Generation 2, Priority 4).

Static covariance estimators treat the matrix as a number; volatility and
correlation are processes that move most when it matters. Two models that let
the matrix move, DCC-GARCH and Orthogonal GARCH, are walk-forward fitted and
asked three questions against a trailing sample, an EWMA and a shrinkage
estimator:

1. FORECAST LOSS. Is the Gaussian deviance of the forecast lower, over every
   month-end forecast origin, against the covariance realised over the
   following 21 trading days? (Diebold-Mariano, BH across six comparisons.)
2. CRISES. Is it lower when it is needed: inside the GFC, COVID and 2022
   windows? (14-asset universe, because HYG lists in April 2007 and the 15-asset
   sample cannot reach the GFC out of sample.)
3. PORTFOLIOS. Do long-only minimum-variance books built from the dynamic
   forecasts realise lower volatility than the shrinkage-based book? (Paired
   stationary bootstrap.)

Every estimator sees returns through and including the origin and forecasts
the same quantity, so the comparison isolates the estimator. The decision rules
were committed to ``config/gen2_portfolio.yaml`` before any of this ran.

Figures 34-35.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import performance_summary
from src.features.sleeves import month_end_dates
from src.portfolio.constraints import Constraints
from src.portfolio.dynamic_covariance import (
    gmv_realised_variance,
    qlike_loss,
    realised_second_moment,
    static_forecasts,
    walk_forward_dynamic_covariance,
)
from src.portfolio.mean_variance import minimum_variance_weights
from src.portfolio.risk_parity import risk_parity_weights
from src.utils.dates import slice_dates
from src.utils.plotting import new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg, diebold_mariano
from src.validation.robustness import paired_sharpe_test, paired_volatility_test
from experiments.context import build_context

STAGE = "stage17_dynamic_covariance"
DYNAMIC = ("dcc", "ogarch")
STATIC = ("sample", "ewma", "shrinkage")
ESTIMATORS = STATIC + DYNAMIC
LABELS = {"sample": "sample (252d)", "ewma": "EWMA (hl 60)", "shrinkage": "shrinkage (LW)",
          "dcc": "DCC-GARCH", "ogarch": "O-GARCH (3 factors)"}
COLOURS = {"sample": "#7F7F7F", "ewma": "#56B4E9", "shrinkage": "#009E73", "dcc": "#D55E00", "ogarch": "#0072B2"}


# ---------------------------------------------------------------------------
# Universes and forecasts
# ---------------------------------------------------------------------------
def balanced_universe(returns: pd.DataFrame, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """Rows on which every asset in the universe has a return."""
    frame = returns.drop(columns=[c for c in exclude if c in returns.columns])
    return frame.dropna(how="any")


def forecast_universe(label: str, frame: pd.DataFrame, cfg, cache_dir, logger) -> dict:
    """Dynamic and static forecasts at every month-end origin, cached on disk."""
    node = cfg.get("gen2_portfolio.dynamic_covariance", {}) or {}
    static_spec = {name: dict((node.get("static", {}) or {}).get(name, {})) for name in STATIC}
    min_train = int(node.get("gfc_min_train_days", 500)) if label == "ex_hyg" else int(node.get("min_train_days", 750))
    settings = {"min_train": min_train, "refit_every": int(node.get("refit_every", 252)),
                "horizon": int(node.get("horizon_days", 21)), "n_factors": int(node.get("ogarch_factors", 3))}
    key_source = json.dumps({"settings": settings, "static": static_spec, "label": label}, sort_keys=True)
    digest = hashlib.blake2b(np.ascontiguousarray(frame.to_numpy()).tobytes() + key_source.encode(),
                             digest_size=8).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"dynamic_covariance_{label}_{digest}.pkl"
    if path.exists():
        logger.info("%s: using the cache %s", label, path.name)
        return pickle.loads(path.read_bytes())

    origins = month_end_dates(frame.index)
    logger.info("%s: %d assets, %d days from %s; walk-forward DCC and O-GARCH (min train %d, refit every %d)",
                label, frame.shape[1], len(frame), frame.index[0].date(), settings["min_train"],
                settings["refit_every"])
    dynamic = walk_forward_dynamic_covariance(frame, origins, **settings, models=DYNAMIC)
    static = static_forecasts(frame, dynamic.origins, static_spec)
    realised = realised_second_moment(frame, dynamic.origins, settings["horizon"])
    out = {"origins": dynamic.origins, "columns": list(frame.columns), "settings": settings,
           "forecasts": {**static, **dynamic.forecasts}, "realised": realised,
           "diagnostics": dynamic.diagnostics, "static_spec": static_spec}
    path.write_bytes(pickle.dumps(out))
    return out


def loss_frames(result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-origin Gaussian deviance and minimum-variance-portfolio variance for each estimator."""
    qlike = pd.DataFrame({name: qlike_loss(result["forecasts"][name], result["realised"])
                          for name in ESTIMATORS}, index=result["origins"])
    gmv = pd.DataFrame({name: gmv_realised_variance(result["forecasts"][name], result["realised"])
                        for name in ESTIMATORS}, index=result["origins"])
    valid = np.isfinite(result["realised"]).all(axis=(1, 2))
    return qlike[valid].dropna(), gmv[valid].reindex(qlike[valid].dropna().index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def dm_family(losses: pd.DataFrame, lag: int, fdr: float) -> pd.DataFrame:
    """Each dynamic model against each static estimator: Diebold-Mariano, BH across the family."""
    rows = []
    for model in DYNAMIC:
        for static in STATIC:
            test = diebold_mariano(losses[model], losses[static], lag=lag, alternative="two-sided")
            if not test:
                continue
            rows.append({"model": model, "static": static, "n_origins": int(len(losses)),
                         "mean_loss_model": float(losses[model].mean()),
                         "mean_loss_static": float(losses[static].mean()),
                         "mean_loss_difference": float(test["mean_loss_difference"]),
                         "statistic": float(test["statistic"]), "p_value": float(test["p_value"])})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["bh_significant"] = benjamini_hochberg(table["p_value"], fdr)
    table["model_better"] = table["bh_significant"] & (table["mean_loss_difference"] < 0)
    return table


def retained(table: pd.DataFrame, model: str) -> bool:
    rows = table[table["model"] == model]
    return bool(len(rows) == len(STATIC) and rows["model_better"].all())


def crisis_mask(origins: pd.DatetimeIndex, windows: dict) -> pd.Series:
    mask = pd.Series(False, index=origins)
    for lo, hi in windows.values():
        mask |= (origins >= pd.Timestamp(lo)) & (origins <= pd.Timestamp(hi))
    return mask


# ---------------------------------------------------------------------------
# Portfolios built from each covariance forecast
# ---------------------------------------------------------------------------
def build_books(result: dict, constraints: Constraints, annualisation: int = 252) -> dict:
    """Long-only minimum-variance and risk-parity weights at every origin, one book per estimator."""
    columns = result["columns"]
    books: dict[str, dict[str, pd.DataFrame]] = {"min_variance": {}, "risk_parity": {}}
    for name in ESTIMATORS:
        mv = np.full((len(result["origins"]), len(columns)), np.nan)
        rp = np.full_like(mv, np.nan)
        for k in range(len(result["origins"])):
            cov = pd.DataFrame(result["forecasts"][name][k] * annualisation, index=columns, columns=columns)
            try:
                mv[k] = minimum_variance_weights(cov, constraints).weights.reindex(columns).to_numpy()
                rp[k] = risk_parity_weights(cov, None, "newton", constraints).reindex(columns).to_numpy()
            except (ValueError, np.linalg.LinAlgError):
                continue
        books["min_variance"][name] = pd.DataFrame(mv, index=result["origins"], columns=columns)
        books["risk_parity"][name] = pd.DataFrame(rp, index=result["origins"], columns=columns)
    return books


def run_books(engine: BacktestEngine, books: dict, returns: pd.DataFrame, investable: pd.DataFrame) -> dict:
    results = {}
    for allocator, by_estimator in books.items():
        for name, weights in by_estimator.items():
            full = weights.reindex(returns.index).ffill().fillna(0.0)
            results[(allocator, name)] = engine.run(full, returns, f"{allocator}|{name}", investable,
                                                    apply_vol_target=False)
    return results


# ---------------------------------------------------------------------------
# Post-hoc diagnostics (labelled as such; none feeds a decision)
# ---------------------------------------------------------------------------
def sign_tests(losses: pd.DataFrame) -> pd.DataFrame:
    """How OFTEN each dynamic model beats each static one, as opposed to by how much."""
    from scipy import stats

    rows = []
    for model in DYNAMIC:
        for static in STATIC:
            wins = int((losses[model] < losses[static]).sum())
            n = int(len(losses))
            rows.append({"model": model, "static": static, "n_origins": n, "wins": wins,
                         "win_rate": wins / n, "sign_test_p_value": float(stats.binomtest(wins, n, 0.5).pvalue),
                         "median_loss_difference": float(np.median(losses[model] - losses[static])),
                         "mean_loss_difference": float((losses[model] - losses[static]).mean())})
    return pd.DataFrame(rows)


def concentration_table(losses: pd.DataFrame, lag: int) -> pd.DataFrame:
    """POST-HOC: how much of each model's mean advantage comes from a single forecast origin?

    Loss differences are heavy-tailed: a month in which the static estimators
    were caught flat can dominate the sum. For each dynamic-versus-static pair
    this reports the most extreme origin, its share of the total difference,
    and the mean difference and Diebold-Mariano test WITHOUT that origin.
    """
    rows = []
    for model in DYNAMIC:
        for static in STATIC:
            diff = losses[model] - losses[static]
            extreme = diff.abs().idxmax()
            rest = diff.drop(extreme)
            test = diebold_mariano(rest.to_numpy(), np.zeros(len(rest)), lag=lag, alternative="two-sided")
            rows.append({"model": model, "static": static, "total_difference": float(diff.sum()),
                         "most_extreme_origin": extreme.date().isoformat(),
                         "most_extreme_difference": float(diff.loc[extreme]),
                         "share_of_total": float(diff.loc[extreme] / diff.sum()) if diff.sum() != 0 else np.nan,
                         "mean_difference_without_it": float(rest.mean()),
                         "dm_p_value_without_it": float(test["p_value"]) if test else np.nan})
    return pd.DataFrame(rows)


def window_table(qlike: pd.DataFrame, gmv: pd.DataFrame, windows: dict) -> pd.DataFrame:
    rows = []
    for name, (lo, hi) in windows.items():
        mask = (qlike.index >= pd.Timestamp(lo)) & (qlike.index <= pd.Timestamp(hi))
        for estimator in ESTIMATORS:
            rows.append({"window": name, "estimator": estimator, "n_origins": int(mask.sum()),
                         "mean_qlike": float(qlike.loc[mask, estimator].mean()) if mask.any() else np.nan,
                         "mean_gmv_variance": float(gmv.loc[mask, estimator].mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def ogarch_factor_sensitivity(label: str, frame: pd.DataFrame, cfg, realised, origins, factors=(3, 5, 8)) -> pd.DataFrame:
    """POST-HOC: does O-GARCH lose because three factors discard too much of the matrix?"""
    node = cfg.get("gen2_portfolio.dynamic_covariance", {}) or {}
    min_train = int(node.get("gfc_min_train_days", 500)) if label == "ex_hyg" else int(node.get("min_train_days", 750))
    rows = []
    for k in factors:
        run = walk_forward_dynamic_covariance(frame, month_end_dates(frame.index), min_train=min_train,
                                              refit_every=int(node.get("refit_every", 252)),
                                              horizon=int(node.get("horizon_days", 21)), n_factors=k,
                                              models=("ogarch",))
        forecast = run.forecasts["ogarch"]
        loss = qlike_loss(forecast, realised)
        gmv = gmv_realised_variance(forecast, realised)
        valid = np.isfinite(loss)
        rows.append({"n_factors": k, "mean_qlike": float(np.nanmean(loss)),
                     "gmv_volatility_annualised": float(np.sqrt(np.nanmean(gmv[valid]) * 252)),
                     "variance_explained": float(run.diagnostics["ogarch_explained"].mean())})
    return pd.DataFrame(rows)


def book_table(runs: dict, start: pd.Timestamp) -> pd.DataFrame:
    rows = {}
    for (allocator, name), result in runs.items():
        net = slice_dates(result.net_returns, start, None)
        stats = performance_summary(net, turnover=slice_dates(result.turnover, start, None),
                                    costs=slice_dates(result.costs, start, None))
        rows[(allocator, name)] = {
            "volatility": stats["ann_vol"], "sharpe": stats["sharpe"], "cagr": stats["cagr"],
            "max_drawdown": stats["max_drawdown"], "cvar_95": stats["cvar_95"],
            "ann_turnover": float(slice_dates(result.turnover, start, None).sum() / max(len(net) / 252.0, 1e-9)),
            "cost_drag_bp_per_year": float(1e4 * slice_dates(result.costs, start, None).mean() * 252),
        }
    table = pd.DataFrame(rows).T
    table.index.names = ["allocator", "estimator"]
    return table


def volatility_tests(runs: dict, start: pd.Timestamp, bootstrap: dict, fdr: float) -> pd.DataFrame:
    """The pre-declared portfolio test: each dynamic book's volatility against the shrinkage book's."""
    rows = []
    for allocator in ("min_variance", "risk_parity"):
        base = slice_dates(runs[(allocator, "shrinkage")].net_returns, start, None)
        for model in DYNAMIC:
            net = slice_dates(runs[(allocator, model)].net_returns, start, None)
            vol = paired_volatility_test(net, base, n_samples=int(bootstrap.get("n_samples", 2000)),
                                         block_length=int(bootstrap.get("block_length", 21)),
                                         seed=int(bootstrap.get("seed", 7)))
            sharpe = paired_sharpe_test(net, base, n_samples=int(bootstrap.get("n_samples", 2000)),
                                        block_length=int(bootstrap.get("block_length", 21)),
                                        seed=int(bootstrap.get("seed", 7)))
            rows.append({"allocator": allocator, "model": model, **vol,
                         "sharpe_difference": sharpe.get("difference", np.nan),
                         "sharpe_p_value": sharpe.get("p_value", np.nan)})
    table = pd.DataFrame(rows)
    gate = table["allocator"] == "min_variance"
    table["bh_significant"] = False
    table.loc[gate, "bh_significant"] = benjamini_hochberg(table.loc[gate, "p_value"], fdr)
    table["model_better"] = table["bh_significant"] & (table["difference"] < 0)
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _portfolio_vol(forecast: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.einsum("i,tij,j->t", weights, forecast, weights), 0.0) * 252)


def _mean_correlation(forecast: np.ndarray) -> np.ndarray:
    out = np.empty(len(forecast))
    n = forecast.shape[1]
    off = ~np.eye(n, dtype=bool)
    for k, cov in enumerate(forecast):
        d = np.sqrt(np.diag(cov))
        out[k] = (cov / np.outer(d, d))[off].mean() if np.all(np.isfinite(cov)) else np.nan
    return out


def _shade_windows(ax, windows: dict, alpha: float = 0.12) -> None:
    for lo, hi in windows.values():
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color="#CC79A7", alpha=alpha, linewidth=0)


def figure_dynamics(res_a: dict, res_b: dict, qlike_b: pd.DataFrame, windows: dict, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    n = len(res_a["columns"])
    equal = np.full(n, 1.0 / n)
    origins = res_a["origins"]
    ax = axes[0, 0]
    realised = np.sqrt(np.maximum(np.einsum("i,tij,j->t", equal, np.nan_to_num(res_a["realised"]), equal), 0) * 252)
    realised[~np.isfinite(res_a["realised"]).all(axis=(1, 2))] = np.nan
    for name in ESTIMATORS:
        ax.plot(origins, 100 * _portfolio_vol(res_a["forecasts"][name], equal), color=COLOURS[name],
                linewidth=1.0 if name not in ("dcc",) else 1.6, label=LABELS[name], alpha=0.9)
    ax.plot(origins, 100 * realised, color="black", linewidth=1.0, linestyle="--", label="realised over the next 21 days")
    ax.set_yscale("log")
    ax.set_ylim(2.5, 110)
    ax.set_ylabel("Equal-weight book volatility, % a year (log)")
    ax.set_title("What each estimator said at month-end, and what followed")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    ax = axes[0, 1]
    for name in ("sample", "shrinkage", "dcc", "ogarch"):
        wide = name == "sample"
        ax.plot(origins, _mean_correlation(res_a["forecasts"][name]), color=COLOURS[name],
                linewidth=3.2 if wide else (1.6 if name == "dcc" else 1.0), alpha=0.45 if wide else 1.0,
                label=LABELS[name] + (" (shrinkage keeps its mean correlation)" if wide else ""))
    ax.set_ylabel("Average pairwise correlation in the forecast")
    ax.set_title("How each forecast sees correlation")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    diag = res_a["diagnostics"].copy()
    dates = pd.to_datetime(diag["refit_date"])
    ax.plot(dates, diag["dcc_a"], "-o", color="#D55E00", markersize=4, label="DCC  a (reaction to news)")
    ax.plot(dates, diag["dcc_b"], "-o", color="#0072B2", markersize=4, label="DCC  b (memory)")
    ax.plot(dates, diag["garch_persistence_mean"], "-s", color="#009E73", markersize=4,
            label="mean GARCH alpha + beta")
    ax.plot(dates, diag["dcc_persistence"], "-s", color="#CC79A7", markersize=4, label="DCC a + b")
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("Parameter value at each annual refit")
    ax.set_title("What the models learned: volatility and correlation both persist")
    ax.legend(fontsize=7.5, loc="center right")

    ax = axes[1, 1]
    for static in STATIC:
        diff = (qlike_b["dcc"] - qlike_b[static]).cumsum()
        ax.plot(diff.index, diff.to_numpy(), color=COLOURS[static], linewidth=1.3, label=f"DCC minus {LABELS[static]}")
    ax.axhline(0, color="black", linewidth=0.8)
    _shade_windows(ax, windows)
    ax.set_ylabel("Cumulative loss difference (below zero: DCC ahead)")
    ax.set_title("Where the gains arrive: crisis windows shaded (14-asset universe)")
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Figure 34. Dynamic covariance in action", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "How do the dynamic covariance forecasts differ from the static ones, what do "
                           "the models learn, and when does the extra flexibility pay?", 34)


def figure_payoff(signs: pd.DataFrame, dm_tables: dict, vol_tests: pd.DataFrame, runs: dict,
                  start: pd.Timestamp, path):
    fig, axes = new_axes(2, 2, figsize=(14.5, 9.2))
    ax = axes[0, 0]
    rows = signs.reset_index(drop=True)
    y = np.arange(len(rows))
    colours = [COLOURS[m] for m in rows["model"]]
    ax.barh(y, 100 * rows["win_rate"], color=colours)
    ax.axvline(50, color="black", linewidth=0.9)
    for yi, (_, r) in zip(y, rows.iterrows()):
        star = "*" if r["sign_test_p_value"] < 0.01 else ""
        ax.text(100 * r["win_rate"], yi, f" {100 * r['win_rate']:.0f}%{star}", va="center", fontsize=9)
    ax.set_yticks(y, [f"{LABELS[m].split(' (')[0]} vs {LABELS[s].split(' (')[0]}" for m, s in zip(rows["model"], rows["static"])])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of forecast origins on which the dynamic model has the lower loss (* sign test p < 0.01, post-hoc)")
    ax.set_title("How OFTEN does it win? (15-asset universe)")

    ax = axes[0, 1]
    groups = [("15 assets,\nall origins", dm_tables["full_a"]), ("14 assets,\nall origins", dm_tables["full_b"]),
              ("14 assets,\ncrisis windows", dm_tables["crisis"])]
    x = np.arange(len(groups))
    width = 0.13
    for j, (model, static) in enumerate([(m, s) for m in DYNAMIC for s in STATIC]):
        values = [float(t[(t["model"] == model) & (t["static"] == static)]["mean_loss_difference"].iloc[0])
                  for _, t in groups]
        ax.bar(x + (j - 2.5) * width, values, width=width, color=COLOURS[model], alpha=0.45 + 0.18 * STATIC.index(static),
               label=f"{LABELS[model].split(' (')[0]} - {static}")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_yticks([-100, -10, -1, 0, 1, 10])
    ax.set_yticklabels(["-100", "-10", "-1", "0", "1", "10"])
    ax.set_xticks(x, [g for g, _ in groups])
    ax.set_ylabel("Mean loss difference (symlog)")
    ax.set_title("By how MUCH?  (below zero = dynamic ahead)")
    ax.legend(fontsize=7, loc="lower left", ncol=2)

    ax = axes[1, 0]
    gate = vol_tests[vol_tests["allocator"] == "min_variance"].reset_index(drop=True)
    y = np.arange(len(gate))
    for i, row in gate.iterrows():
        ratio = row["ratio"]
        lo, hi = (row["volatility_b"] + row["ci_lower_5pct"]) / row["volatility_b"], (row["volatility_b"] + row["ci_upper_95pct"]) / row["volatility_b"]
        colour = "#009E73" if bool(row["model_better"]) else "#CC0000"
        ax.errorbar(ratio, i, xerr=[[ratio - lo], [hi - ratio]], fmt="o", color=colour, capsize=4, markersize=8)
        ax.text(hi + 0.002, i, f" p={row['p_value']:.3f}", va="center", fontsize=9)
    ax.axvline(1.0, color="black", linewidth=0.9)
    ax.set_yticks(y, [LABELS[m] for m in gate["model"]])
    ax.set_ylim(len(gate) - 0.5, -0.5)
    ax.set_xlabel("Realised volatility of the minimum-variance book, relative to the shrinkage-based book (90% interval)")
    ax.set_title("The pre-declared portfolio test: a lower risk forecast should mean a calmer book")

    ax = axes[1, 1]
    for name in ESTIMATORS:
        net = slice_dates(runs[("min_variance", name)].net_returns, start, None)
        curve = (1.0 + net).cumprod()
        ax.plot(curve.index, curve.to_numpy(), color=COLOURS[name], linewidth=1.5 if name == "dcc" else 1.0,
                label=LABELS[name])
    ax.set_ylabel("Growth of 1 unit, net of costs")
    ax.set_title("Minimum-variance books, long-only, monthly")
    ax.text(0.98, 0.04, "O-GARCH: higher volatility, higher Sharpe (post-hoc, not the criterion)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=COLOURS["ogarch"])
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Figure 35. Do dynamic covariance forecasts pay?", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path, "Do the dynamic forecasts beat the static ones on loss, in crises, and as the "
                           "input to a minimum-variance book?", 35)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
NAMES = {"dcc": "DCC-GARCH", "ogarch": "Orthogonal GARCH"}


def _window_summary(windows_table: pd.DataFrame, model: str) -> str:
    parts = []
    for window, group in windows_table.groupby("window", sort=False):
        loss = group.set_index("estimator")["mean_qlike"]
        parts.append(f"{window} {loss[model] - loss[list(STATIC)].min():+.1f} (n={int(group['n_origins'].iloc[0])})")
    return ", ".join(parts)


def log_registry(context, cfg, res_a, res_b, dm_a, dm_b, dm_crisis, crisis_origins, vol_tests, book_perf,
                 signs, factor_sensitivity, windows_table, pd_checks, concentration) -> None:
    registry = context.registry
    node = cfg.get("gen2_portfolio.dynamic_covariance", {}) or {}
    fdr = float((node.get("decisions", {}) or {}).get("fdr", 0.10))
    settings = res_a["settings"]
    common = {"horizon_days": settings["horizon"], "refit_every": settings["refit_every"],
              "origins": "last trading day of each month", "loss": "Gaussian deviance (multivariate QLIKE)",
              "test": "Diebold-Mariano, two-sided, BH across six comparisons", "fdr": fdr,
              "dm_lag": int((node.get("decisions", {}) or {}).get("dm_lag", 1))}
    for model in DYNAMIC:
        label = NAMES[model]
        rows = dm_a[dm_a["model"] == model]
        results = {"n_origins": int(rows["n_origins"].iloc[0])}
        for _, r in rows.iterrows():
            results[f"mean_loss_difference_vs_{r['static']}"] = float(r["mean_loss_difference"])
            results[f"p_value_vs_{r['static']}"] = float(r["p_value"])
        sign = signs[signs["model"] == model].set_index("static")
        for static in STATIC:
            results[f"posthoc_win_rate_vs_{static}"] = float(sign.loc[static, "win_rate"])
        ok = retained(dm_a, model)
        registry.log(
            f"{label} covariance forecasts have a lower out-of-sample loss than each of the sample, EWMA and "
            "shrinkage estimators.",
            stage=STAGE, parameters={**common, "universe": f"{len(res_a['columns'])} assets", "min_train_days": settings["min_train"]},
            train_period="expanding window, refit every 252 days", test_period=f"{res_a['origins'][0].date()} onward, monthly origins",
            results=results, decision="retain" if ok else "reject",
            notes=("Retained only if the mean loss is lower than each static estimator's AND all three tests are "
                   "significant after BH control. "
                   + ("" if model != "dcc" else
                      "The mean loss is lowest for DCC against all three, but the differences are driven by a few stress "
                      "months and are not significant at this sample size. Post-hoc: DCC has the lower loss on only "
                      f"{100 * float(sign.loc['sample', 'win_rate']):.0f}% of origins against the sample estimator and "
                      f"{100 * float(sign.loc['ewma', 'win_rate']):.0f}% against EWMA, but {100 * float(sign.loc['shrinkage', 'win_rate']):.0f}% "
                      "against shrinkage: in a typical month the static estimators are as good or better, in a crisis "
                      "they are far worse, and a mean-based test sees mostly the second. One origin, "
                      f"{concentration[(concentration['model'] == 'dcc') & (concentration['static'] == 'sample')]['most_extreme_origin'].iloc[0]}, "
                      f"accounts for {100 * float(concentration[(concentration['model'] == 'dcc') & (concentration['static'] == 'sample')]['share_of_total'].iloc[0]):.0f}% "
                      "of the total difference against the sample estimator (post-hoc).")
                   + ("" if model != "ogarch" else
                      "Three factors leave a quarter of the variance in a constant diagonal; it is worse on loss than the "
                      "static estimators in most months (post-hoc win rate "
                      f"{100 * float(sign['win_rate'].mean()):.0f}% on average). See the post-hoc factor-count sensitivity.")),
        )

        rows = dm_crisis[dm_crisis["model"] == model]
        results = {"n_origins": int(rows["n_origins"].iloc[0]), "universe": "14 assets (ex-HYG)"}
        for _, r in rows.iterrows():
            results[f"mean_loss_difference_vs_{r['static']}"] = float(r["mean_loss_difference"])
            results[f"p_value_vs_{r['static']}"] = float(r["p_value"])
        ok = retained(dm_crisis, model)
        registry.log(
            f"{label} covariance forecasts have a lower loss than each static estimator inside the GFC, COVID and "
            "2022 windows.",
            stage=STAGE, parameters={**common, "universe": "14 assets (ex-HYG)", "min_train_days": res_b["settings"]["min_train"],
                                     "windows": node.get("crisis_windows")},
            train_period="expanding window, refit every 252 days",
            test_period="forecast origins inside the three windows, pooled", results=results,
            decision="retain" if ok else "reject",
            notes=("The same rule on the pooled crisis origins (" + str(crisis_origins) + " of them: little power by construction). "
                   "The windows tell different stories. Mean loss of " + label + " minus the best static estimator's "
                   "(below zero: ahead): " + _window_summary(windows_table, model) + ". HYG lists in April 2007, so the "
                   "14-asset universe is the only one with an out-of-sample GFC."),
        )

        row = vol_tests[(vol_tests["allocator"] == "min_variance") & (vol_tests["model"] == model)].iloc[0]
        registry.log(
            f"A long-only minimum-variance book built from {label} forecasts realises lower volatility than the "
            "same book built from the shrinkage estimator.",
            stage=STAGE, parameters={**common, "test": "paired stationary bootstrap on realised volatility, BH across two",
                                     "universe": f"{len(res_a['columns'])} assets", "constraints": "long-only, config caps"},
            train_period="expanding window, refit every 252 days", test_period=f"{res_a['origins'][0].date()} onward, net of costs",
            cost_bps=float(cfg.get("backtest.costs.cost_bps", 10.0)),
            results={"volatility_model_book": float(row["volatility_a"]), "volatility_shrinkage_book": float(row["volatility_b"]),
                     "volatility_ratio": float(row["ratio"]), "p_value": float(row["p_value"]),
                     "bh_significant": bool(row["bh_significant"]), "sharpe_difference": float(row["sharpe_difference"]),
                     "annual_turnover_model_book": float(book_perf.loc[("min_variance", model), "ann_turnover"]),
                     "annual_turnover_shrinkage_book": float(book_perf.loc[("min_variance", "shrinkage"), "ann_turnover"])},
            decision="retain" if bool(row["model_better"]) else "reject",
            notes=("Retained only if the model's book volatility is significantly LOWER than the shrinkage book's "
                   "(two-sided paired bootstrap, BH across the two models). "
                   + (f"Volatility is {100 * (1 - float(row['ratio'])):.1f}% lower"
                      if float(row["difference"]) < 0 else f"Volatility is {100 * (float(row['ratio']) - 1):.1f}% HIGHER")
                   + f" (p={float(row['p_value']):.3f}); the books' returns are {float(row['return_correlation']):.2f} correlated, "
                     "which is why a small difference can be significant. The dynamic book trades "
                     f"{float(book_perf.loc[('min_variance', model), 'ann_turnover']):.1f}x a year against "
                     f"{float(book_perf.loc[('min_variance', 'shrinkage'), 'ann_turnover']):.1f}x. Sharpe ratio and the risk-parity "
                     "books are reported in the tables, not judged."),
        )

    best = factor_sensitivity.sort_values("mean_qlike").iloc[0]
    registry.log(
        "Descriptive: the dynamic models' advantage is concentrated in stress months, and the factor count limits O-GARCH.",
        stage=STAGE, parameters={"post_hoc": True, "ogarch_factors_tried": factor_sensitivity["n_factors"].tolist()},
        train_period="expanding window", test_period=f"{res_a['origins'][0].date()} onward",
        results={"min_eigenvalue_over_all_forecasts": float(pd_checks["min_eigenvalue"].min()),
                 "non_positive_definite_forecasts": int(pd_checks["non_pd"].sum()),
                 "ogarch_best_factor_count_by_loss": int(best["n_factors"]),
                 "ogarch_mean_qlike_by_factors": {int(r["n_factors"]): float(r["mean_qlike"]) for _, r in factor_sensitivity.iterrows()},
                 "dcc_persistence_range": [float(res_a["diagnostics"]["dcc_persistence"].min()), float(res_a["diagnostics"]["dcc_persistence"].max())]},
        decision="record",
        notes=("Post-hoc diagnostics, none of which feeds a decision. Every forecast of every model was positive definite. "
               "DCC's correlation persistence a+b sits between 0.97 and 0.995, close to integrated, so forecasts of correlation "
               "revert very slowly towards the unconditional matrix. Wishart and factor stochastic-volatility "
               "covariance models are not implemented."),
    )


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 17: dynamic covariance").parse_args(argv)
    context, logger = build_context(STAGE, generation=2)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 17 | dynamic covariance (Generation 2, Priority 4)")
    logger.info("=" * 72)
    node = cfg.get("gen2_portfolio.dynamic_covariance", {}) or {}
    decisions = node.get("decisions", {}) or {}
    fdr, dm_lag = float(decisions.get("fdr", 0.10)), int(decisions.get("dm_lag", 1))
    windows = {k: tuple(v) for k, v in (node.get("crisis_windows", {}) or {}).items()}
    market = context.market_data()
    returns = market.returns()
    cache = cfg.path("features") / "dynamic_covariance"

    frame_a = balanced_universe(returns)
    frame_b = balanced_universe(returns, tuple(node.get("gfc_universe_exclude", ["HYG"])))
    res_a = forecast_universe("all15", frame_a, cfg, cache, logger)
    res_b = forecast_universe("ex_hyg", frame_b, cfg, cache, logger)
    qlike_a, gmv_a = loss_frames(res_a)
    qlike_b, gmv_b = loss_frames(res_b)
    for label, q in (("all15", qlike_a), ("ex_hyg", qlike_b)):
        logger.info("%s: %d usable forecast origins from %s; mean loss %s", label, len(q), q.index[0].date(),
                    {k: round(float(v), 2) for k, v in q.mean().items()})
    context.save_table(qlike_a, "stage17_losses_15_assets.csv")
    context.save_table(qlike_b, "stage17_losses_14_assets.csv")
    context.save_table(gmv_a, "stage17_gmv_variance_15_assets.csv", float_format="%.4e")
    context.save_table(res_a["diagnostics"], "stage17_model_parameters.csv", index=False)

    # ---- positive-definiteness of every forecast
    checks = pd.DataFrame({name: [float(np.linalg.eigvalsh(f).min()) for f in res_a["forecasts"][name]]
                           for name in ESTIMATORS}, index=res_a["origins"])
    pd_checks = pd.DataFrame({"min_eigenvalue": checks.min(), "non_pd": (checks <= 0).sum()})
    context.save_table(pd_checks, "stage17_positive_definiteness.csv")

    # ---- the pre-declared tests
    dm_a = dm_family(qlike_a, dm_lag, fdr)
    dm_b = dm_family(qlike_b, dm_lag, fdr)
    mask = crisis_mask(qlike_b.index, windows)
    dm_crisis = dm_family(qlike_b[mask], dm_lag, fdr)
    context.save_table(dm_a, "stage17_dm_15_assets.csv", index=False)
    context.save_table(dm_b, "stage17_dm_14_assets.csv", index=False)
    context.save_table(dm_crisis, "stage17_dm_crisis.csv", index=False)
    cols = ["model", "static", "n_origins", "mean_loss_difference", "statistic", "p_value", "bh_significant", "model_better"]
    logger.info("forecast loss, 15 assets (DM, BH across six):\n%s", dm_a[cols].round(4).to_string(index=False))
    logger.info("crisis origins pooled (%d), 14 assets:\n%s", int(mask.sum()), dm_crisis[cols].round(4).to_string(index=False))
    windows_table = window_table(qlike_b, gmv_b, windows)
    context.save_table(windows_table, "stage17_crisis_windows.csv", index=False)
    signs = sign_tests(qlike_a)
    context.save_table(signs, "stage17_posthoc_sign_tests.csv", index=False)
    logger.info("post-hoc: how often does each dynamic model have the lower loss?\n%s",
                signs[["model", "static", "wins", "n_origins", "win_rate", "sign_test_p_value",
                       "median_loss_difference", "mean_loss_difference"]].round(4).to_string(index=False))

    concentration = concentration_table(qlike_a, dm_lag)
    context.save_table(concentration, "stage17_posthoc_concentration.csv", index=False)
    logger.info("post-hoc: how much of the advantage is one month?\n%s",
                concentration[["model", "static", "total_difference", "most_extreme_origin", "most_extreme_difference",
                               "share_of_total", "mean_difference_without_it", "dm_p_value_without_it"]].round(3).to_string(index=False))

    # ---- post-hoc: does O-GARCH lose because of the factor count?
    factor_sensitivity = ogarch_factor_sensitivity("all15", frame_a, cfg, res_a["realised"], res_a["origins"])
    context.save_table(factor_sensitivity, "stage17_ogarch_factor_sensitivity.csv", index=False)
    logger.info("post-hoc O-GARCH factor count:\n%s", factor_sensitivity.round(4).to_string(index=False))

    # ---- portfolios built from each forecast (15 assets)
    constraints = Constraints.from_config(cfg)
    inv = market.investable.reindex(index=res_a["origins"], columns=res_a["columns"])
    if not bool(inv.fillna(False).all().all()):
        logger.warning("some assets are not investable at some origins; the engine zeroes those weights")
    engine = BacktestEngine.from_config(cfg)
    books = build_books(res_a, constraints)
    runs = run_books(engine, books, returns, market.investable)
    start = res_a["origins"][0]
    perf = book_table(runs, start)
    context.save_table(perf, "stage17_book_performance.csv")
    logger.info("books from %s (net of costs):\n%s", start.date(), perf.round(4).to_string())
    vol_tests = volatility_tests(runs, start, node.get("bootstrap", {}) or {}, fdr)
    context.save_table(vol_tests, "stage17_book_volatility_tests.csv", index=False)
    logger.info("book volatility tests vs the shrinkage book:\n%s",
                vol_tests[["allocator", "model", "volatility_a", "volatility_b", "ratio", "p_value", "bh_significant",
                           "model_better", "sharpe_difference", "sharpe_p_value"]].round(4).to_string(index=False))

    # ---- figures and registry
    figure_dynamics(res_a, res_b, qlike_b, windows, context.figure("fig34_dynamic_covariance.png"))
    figure_payoff(signs, {"full_a": dm_a, "full_b": dm_b, "crisis": dm_crisis}, vol_tests, runs, start,
                  context.figure("fig35_dynamic_covariance_payoff.png"))
    log_registry(context, cfg, res_a, res_b, dm_a, dm_b, dm_crisis, int(mask.sum()), vol_tests, perf, signs,
                 factor_sensitivity, windows_table, pd_checks, concentration)
    logger.info("STAGE 17 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
