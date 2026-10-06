"""Stage 35 - Explainability and calibration of the forecasts (Generation 5).

Part A (descriptive): two models, a small MLP and a gradient-boosted tree model, are fitted on the twelve Stage 19 price features to the 21-day
z-score target on the Stage 19 annual refit schedule, then explained by permutation importance, exact interventional Shapley values and
integrated gradients. The point is not skill. It is that each method's own identity is TESTED on every explained row (Shapley efficiency,
integrated-gradients completeness) and that the methods' global rankings are compared and their disagreement reported.

Part B (one declared hypothesis, ``h_isotonic``): at each monthly origin from 2016-01-31, Platt scaling and isotonic regression are fitted on
the Stage 19 ``p_raw`` and outcomes of the origins whose 21-day outcome had matured, and applied to the current ``p_raw``. Is isotonic's mean log
loss lower than Platt's? All arms are scored with the same probability clip (1e-6), as in Stage 19. An isotonic fit can output exactly 0 or 1, which the
clip turns into a very large penalty when wrong; a variant clipped to [0.02, 0.98] is reported as POST-HOC and cannot overturn the decision.

Rules: ``config/frontier.yaml`` (section ``explain_calibrate``). Figures 69-70.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.context import build_context
from src.models.deep_forecast import TrainSettings
from src.models.explain import integrated_gradients, permutation_importance, shapley_exact
from src.models.probabilistic import (PlattCalibrator, brier_terms, clip_probability, expected_calibration_error, ewma_sigma, log_loss_terms,
                                      price_features, reliability_table)
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import diebold_mariano

STAGE = "stage35_explain"
METHODS = ["perm_mlp", "perm_gbm", "shap_mlp", "shap_gbm", "ig_mlp"]


def train_mlp(Xtr, ytr, Xva, yva, seed: int, settings: TrainSettings, hidden: int = 16):
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    f = lambda z: net(z).squeeze(-1)
    opt = torch.optim.Adam(net.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    tx, ty = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr, dtype=torch.float32)
    vx, vy = torch.as_tensor(Xva, dtype=torch.float32), torch.as_tensor(yva, dtype=torch.float32)
    best, state, best_epoch = np.inf, None, 0
    for epoch in range(settings.max_epochs):
        net.train()
        order = torch.randperm(len(tx), generator=gen)
        for i in range(0, len(order), settings.batch_size):
            idx = order[i:i + settings.batch_size]
            opt.zero_grad()
            torch.mean((f(tx[idx]) - ty[idx]) ** 2).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.mean((f(vx) - vy) ** 2))
        if v < best - 1e-9:
            best, best_epoch, state = v, epoch, {k: t.clone() for k, t in net.state_dict().items()}
        elif epoch - best_epoch >= settings.patience:
            break
    net.load_state_dict(state)
    net.eval()

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, z):
            return self.net(z).squeeze(-1)

    wrapped = Wrapped().eval()

    def predict(X):
        with torch.no_grad():
            return wrapped(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(float)

    return wrapped, predict


def part_a(context, cfg, node, logger, frozen, prices):
    from sklearn.ensemble import HistGradientBoostingRegressor

    horizon = int(cfg.get("forecasting.horizon_days", 21))
    embargo = int(((cfg.get("deeplearning", {}) or {})["task"]["walk_forward"])["embargo_days"])
    returns = prices.pct_change()
    index = pd.DatetimeIndex(prices.index)
    assets = list(prices.columns)
    sigma_h = ewma_sigma(returns, float(cfg.get("forecasting.gaussian.vol_halflife", 40)), horizon)
    forward = prices.shift(-horizon) / prices - 1.0
    feats = price_features(prices)
    names = list(feats)
    cube = np.stack([f.to_numpy(dtype=float) for f in feats.values()], axis=2)
    z_cube = np.clip(forward.to_numpy() / sigma_h.to_numpy(), -5.0, 5.0)
    days = np.arange(252, len(index) - horizon, 5)
    day_grid, asset_grid = np.repeat(days, len(assets)), np.tile(np.arange(len(assets)), len(days))
    Xall, zall = cube[day_grid, asset_grid], z_cube[day_grid, asset_grid]
    good = np.isfinite(Xall).all(axis=1) & np.isfinite(zall)
    settings = TrainSettings(1e-3, 1e-2, 256, 30, 5)
    rng = np.random.default_rng(7)
    explain_rows = int(node["explain"].get("rows", 200)) if isinstance(node["explain"].get("rows"), int) else 200
    perm_parts = {"mlp": [], "gbm": []}
    shap_rows, sizes = [], []
    refits = sorted(frozen["refit_day"].unique())
    pos_of = {o: index.get_loc(o) for o in frozen["origin"].unique()}
    total_test = 0
    plan = {}
    for start in refits:
        g = frozen[frozen["refit_day"] == start].dropna(subset=["target", "sigma"])
        pos = np.array([pos_of[o] for o in g["origin"]])
        a = np.array([assets.index(x) for x in g["asset"]])
        Xt, yt = cube[pos, a], z_cube[pos, a]
        keep = np.isfinite(Xt).all(axis=1) & np.isfinite(yt)
        plan[start] = (pos[keep], a[keep], Xt[keep], yt[keep])
        total_test += int(keep.sum())
    chosen = {s: rng.choice(len(p[0]), size=max(0, round(explain_rows * len(p[0]) / total_test)), replace=False) for s, p in plan.items()}
    checks = []
    for start in refits:
        pos_t, a_t, Xt, yt = plan[start]
        usable = (day_grid + horizon + embargo <= start) & good
        ds = np.unique(day_grid[usable])
        cut = ds[int(np.floor(0.8 * len(ds)))]
        tr, va = usable & (day_grid <= cut - 42), usable & (day_grid > cut)
        mean, std = Xall[tr].mean(axis=0), Xall[tr].std(axis=0, ddof=1)
        std[std == 0] = 1.0
        S = lambda X: (X - mean) / std
        net, predict_mlp = train_mlp(S(Xall[tr]), zall[tr], S(Xall[va]), zall[va], 11, settings)
        gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.05, random_state=0).fit(S(Xall[tr | va]), zall[tr | va])
        predict_gbm = lambda X, m=gbm: m.predict(X)
        St = S(Xt)
        for key, fn in (("mlp", predict_mlp), ("gbm", predict_gbm)):
            perm_parts[key].append((permutation_importance(fn, St, yt, 20, 7), len(yt)))
        bg_idx = rng.choice(np.flatnonzero(tr), size=min(50, int(tr.sum())), replace=False)
        background = S(Xall[bg_idx])
        for i in chosen[start]:
            row = {"refit_day": start, "origin": index[pos_t[i]], "asset": assets[a_t[i]]}
            for key, fn in (("mlp", predict_mlp), ("gbm", predict_gbm)):
                phi, base = shapley_exact(fn, St[i], background)
                checks.append({"model": key, "kind": "shapley_efficiency", "gap": abs(phi.sum() - (fn(St[i][None])[0] - base)), "scale": float(np.ptp(fn(background)) or 1.0)})
                row.update({f"shap_{key}_{n}": v for n, v in zip(names, phi)})
            ig, gap = integrated_gradients(net, St[i], np.zeros(len(names)), steps=100)
            checks.append({"model": "mlp", "kind": "ig_completeness", "gap": gap, "scale": float(np.ptp(predict_mlp(background)) or 1.0)})
            row.update({f"ig_mlp_{n}": v for n, v in zip(names, ig)})
            row.update({f"x_{n}": v for n, v in zip(names, St[i])})
            shap_rows.append(row)
        logger.info("refit %d: %d test rows, %d explained", start, len(yt), len(chosen[start]))
    local = pd.DataFrame(shap_rows)
    checks = pd.DataFrame(checks)
    checks["relative_gap"] = checks["gap"] / checks["scale"]
    imp = {}
    for key in ("mlp", "gbm"):
        w = np.array([n for _, n in perm_parts[key]], dtype=float)
        imp[f"perm_{key}"] = np.average(np.stack([p for p, _ in perm_parts[key]]), axis=0, weights=w)
        imp[f"shap_{key}"] = np.array([local[f"shap_{key}_{n}"].abs().mean() for n in names])
    imp["ig_mlp"] = np.array([local[f"ig_mlp_{n}"].abs().mean() for n in names])
    importance = pd.DataFrame(imp, index=names)[METHODS]
    rank_corr = importance.corr(method="spearman")
    return local, checks, importance, rank_corr, names


def part_b(context, node, logger, frozen, prices):
    index = pd.DatetimeIndex(prices.index)
    horizon = 21
    f = frozen.dropna(subset=["p_raw", "up"]).copy()
    f["pos"] = [index.get_loc(o) for o in f["origin"]]
    origins = sorted(f["origin"].unique())
    start = pd.Timestamp("2016-01-31")
    rows = []
    from sklearn.isotonic import IsotonicRegression

    for o in origins:
        if pd.Timestamp(o) < start:
            continue
        pos_o = index.get_loc(o)
        hist = f[f["pos"] + horizon <= pos_o]
        cur = f[f["origin"] == o]
        if len(hist) < 500 or hist["up"].nunique() < 2:
            continue
        platt = PlattCalibrator().fit(hist["p_raw"].to_numpy(), hist["up"].to_numpy())
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(hist["p_raw"].to_numpy(), hist["up"].to_numpy())
        p = cur["p_raw"].to_numpy()
        out = pd.DataFrame({"origin": o, "asset": cur["asset"].to_numpy(), "up": cur["up"].to_numpy(), "raw": clip_probability(p), "platt": clip_probability(platt.transform(p)),
                            "isotonic": clip_probability(iso.predict(p)), "isotonic_clipped_posthoc": np.clip(iso.predict(p), 0.02, 0.98),
                            "stage19": clip_probability(cur["p_cal"].to_numpy())})
        rows.append(out)
    panel = pd.concat(rows, ignore_index=True)
    context.save_table(panel, "stage35_calibration_panel.csv", index=False)
    arms = ["raw", "platt", "isotonic", "stage19", "isotonic_clipped_posthoc"]
    summary = pd.DataFrame({a: {"log_loss": float(log_loss_terms(panel[a], panel["up"]).mean()), "brier": float(brier_terms(panel[a], panel["up"]).mean()),
                                "ece": expected_calibration_error(panel[a].to_numpy(), panel["up"].to_numpy(), 10)} for a in arms}).T
    logger.info("calibration arms (%d rows, %d origins):\n%s", len(panel), panel["origin"].nunique(), summary.round(5).to_string())
    per_origin = {a: pd.Series(log_loss_terms(panel[a], panel["up"]), index=panel.index).groupby(panel["origin"]).mean() for a in arms}
    tests = []
    for a, b, tag in (("isotonic", "platt", "declared"), ("platt", "raw", "reported"), ("isotonic", "raw", "reported"), ("isotonic_clipped_posthoc", "platt", "post-hoc")):
        t = diebold_mariano(per_origin[a], per_origin[b], lag=0, alternative="two-sided")
        tests.append({"a": a, "b": b, "status": tag, "mean_log_loss_difference": float(t["mean_loss_difference"]), "p_value": float(t["p_value"]), "n_origins": len(per_origin[a])})
    tests = pd.DataFrame(tests)
    context.save_table(summary, "stage35_calibration_summary.csv")
    context.save_table(tests, "stage35_calibration_tests.csv", index=False)
    logger.info("calibration tests:\n%s", tests.round(6).to_string(index=False))
    return panel, summary, tests, arms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 35: explainability and calibration")
    parser.parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    node = (cfg.get("frontier", {}) or {})["explain_calibrate"]
    fdr = float(node["calibration"]["decisions"]["fdr"])
    logger.info("=" * 72)
    logger.info("STAGE 35 | explainability and calibration (Generation 5)")
    logger.info("=" * 72)
    frozen = pd.read_csv(context.tables / "stage19_predictions_price_only.csv", parse_dates=["origin"])
    prices = context.market_data().prices

    local, checks, importance, rank_corr, names = part_a(context, cfg, node, logger, frozen, prices)
    context.save_table(local, "stage35_local_attributions.csv", index=False)
    context.save_table(importance, "stage35_global_importance.csv")
    context.save_table(rank_corr, "stage35_importance_rank_correlation.csv")
    check_summary = checks.groupby(["model", "kind"]).agg(max_gap=("gap", "max"), max_relative_gap=("relative_gap", "max"), n=("gap", "size"))
    context.save_table(check_summary, "stage35_identity_checks.csv")
    logger.info("identity checks:\n%s", check_summary.to_string())
    logger.info("global importance:\n%s", importance.round(5).to_string())
    logger.info("rank correlation between methods:\n%s", rank_corr.round(2).to_string())
    shapley_ok = bool((checks[checks["kind"] == "shapley_efficiency"]["gap"] < 1e-6).all())
    ig_ok = bool((checks[checks["kind"] == "ig_completeness"]["relative_gap"] < 1e-2).all())

    panel, summary, tests, arms = part_b(context, node, logger, frozen, prices)
    row = tests[tests["status"] == "declared"].iloc[0]
    h_isotonic = bool(row["p_value"] <= fdr and row["mean_log_loss_difference"] < 0)

    # --------------------------------------------------------------- figures
    fig, axes = new_axes(1, 3, figsize=(17, 5.4))
    ax = axes[0]
    order = importance.mean(axis=1).sort_values().index
    scaled = importance.div(importance.abs().max(axis=0), axis=1).loc[order]
    y = np.arange(len(order))
    for j, (col, colour) in enumerate(zip(METHODS, PALETTE)):
        ax.barh(y + (j - 2) * 0.16, scaled[col].to_numpy(), height=0.16, color=colour, label=col)
    ax.set_yticks(y, order, fontsize=8)
    ax.set_xlabel("Importance, each method scaled to its own maximum")
    ax.set_title("Global importance by five method-model pairs")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.imshow(rank_corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(METHODS)), METHODS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8)
    for i in range(len(METHODS)):
        for j in range(len(METHODS)):
            ax.text(j, i, f"{rank_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Do the methods agree? Spearman of the rankings")
    ax = axes[2]
    top = importance["shap_gbm"].idxmax()
    ax.scatter(local[f"x_{top}"], local[f"shap_gbm_{top}"], s=10, color=PALETTE[0], label="GBM Shapley")
    ax.scatter(local[f"x_{top}"], local[f"shap_mlp_{top}"], s=10, color=PALETTE[1], label="MLP Shapley")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"{top} (standardised)")
    ax.set_ylabel("Shapley value (z-score units)")
    ax.set_title(f"Local effect of the top GBM feature, {top}")
    ax.legend(fontsize=8)
    fig.suptitle(f"Figure 69. What the models use (Shapley efficiency held on every row: {shapley_ok}; IG completeness: {ig_ok})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, context.figure("fig69_explain_importance.png"), "Which price features do small nonlinear models use to forecast the 21-day return, and do three attribution methods agree?", 69)

    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for a, colour in zip(["raw", "platt", "isotonic"], [PALETTE[2], PALETTE[0], PALETTE[3]]):
        t = reliability_table(panel[a].to_numpy(), panel["up"].to_numpy(), 10).dropna()
        ax.plot(t["mean_forecast"], t["observed_frequency"], marker="o", color=colour, label=a)
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency of an up month")
    ax.set_title("Reliability, 2016 onward")
    ax.legend(fontsize=8)
    ax = axes[1]
    names_ = ["raw", "platt", "isotonic", "stage19"]
    ax.bar(range(4), summary.loc[names_, "log_loss"].to_numpy(), color=[PALETTE[2], PALETTE[0], PALETTE[3], "#777777"])
    ax.set_xticks(range(4), names_)
    ax.set_ylim(summary.loc[names_, "log_loss"].min() - 0.01, summary.loc[names_, "log_loss"].max() + 0.01)
    ax.set_ylabel("Mean log loss (lower is better)")
    ax.set_title(f"Isotonic minus Platt: {row['mean_log_loss_difference']:+.4f}, p = {row['p_value']:.3f}")
    ax = axes[2]
    ax.bar(range(4), summary.loc[names_, "ece"].to_numpy(), color=[PALETTE[2], PALETTE[0], PALETTE[3], "#777777"])
    ax.set_xticks(range(4), names_)
    ax.set_ylabel("Expected calibration error")
    ax.set_title("Calibration error")
    fig.suptitle("Figure 70. Platt scaling against isotonic regression, refitted on matured outcomes", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, context.figure("fig70_calibration_isotonic.png"), "Does the more flexible isotonic recalibration improve on Platt scaling for monthly up/down probabilities?", 70)

    context.registry.log(
        "Descriptive: the Shapley efficiency and integrated-gradients completeness identities hold on every explained row, and the attribution methods agree on the global ranking of price features.",
        stage=STAGE, parameters={"explained_rows": int(len(local)), "models": ["mlp", "gbm"]},
        results={"shapley_efficiency_holds": shapley_ok, "ig_completeness_holds": ig_ok, "max_shapley_gap": float(checks[checks["kind"] == "shapley_efficiency"]["gap"].max()),
                 "max_ig_relative_gap": float(checks[checks["kind"] == "ig_completeness"]["relative_gap"].max()),
                 "rank_corr_shap_mlp_vs_ig": float(rank_corr.loc["shap_mlp", "ig_mlp"]), "rank_corr_shap_gbm_vs_perm_gbm": float(rank_corr.loc["shap_gbm", "perm_gbm"]),
                 "rank_corr_shap_mlp_vs_shap_gbm": float(rank_corr.loc["shap_mlp", "shap_gbm"])},
        decision="record", test_period="2011-01 onward", notes="Teaching and auditing, no hypothesis. Disagreement between methods is reported, not resolved.")
    context.registry.log(
        "Isotonic recalibration of the Stage 19 monthly up-probability has a lower mean log loss than Platt scaling (both refitted on matured outcomes, 2016 onward).",
        stage=STAGE, parameters={"start": "2016-01-31", "min_history_rows": 500, "fdr": fdr},
        results={**{f"log_loss_{a}": float(summary.loc[a, "log_loss"]) for a in arms}, **{f"ece_{a}": float(summary.loc[a, "ece"]) for a in arms},
                 "difference_isotonic_minus_platt": float(row["mean_log_loss_difference"]), "p_value": float(row["p_value"]), "n_origins": int(row["n_origins"])},
        decision="retain" if h_isotonic else "reject", test_period="2016-01 onward, monthly origins, 15 ETFs",
        notes="Retained only with a negative log-loss difference and p <= 0.10. The clipped-isotonic variant is post-hoc and cannot change the decision.")
    logger.info("STAGE 35 complete: shapley_ok=%s ig_ok=%s h_isotonic=%s", shapley_ok, ig_ok, h_isotonic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
