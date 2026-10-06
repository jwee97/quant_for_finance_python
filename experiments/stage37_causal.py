"""Stage 37 - Causal inference (Generation 5).

Part A validates four estimators (DML, R-learner, 2SLS, difference in differences) on simulated data with a KNOWN effect, beside the naive estimate
that ignores the problem. Part B applies DML to one real question: does a more hawkish FOMC statement move TLT or SPY on the day of release?
IV is NOT applied (no credible instrument; stated in the declaration) and DiD is applied only to the announcement-day design declared in
``config/frontier.yaml`` (section ``causal``). Figures 73-74.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments.context import build_context
from src.causal import did_2x2, dml_from_residuals, dml_plr, iv_2sls, naive_ols, r_learner
from src.causal.simulate import confounded_plr, did_panel, endogenous_iv, heterogeneous_effect
from src.models.probabilistic import ewma_sigma
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import benjamini_hochberg

STAGE = "stage37_causal"
REPS, N = 300, 500


def one_rep(kind: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if kind == "plr_oracle_posthoc":                      # POST-HOC: true nuisance functions, isolates the estimator's final step from learner error
        s = confounded_plr(N, rng)
        return {"DML final step, true nuisance": {**dml_from_residuals(s["y"] - s["m_y"], s["d"] - s["m_d"]), "truth": s["theta"]}}
    if kind == "plr_n5000_posthoc":                       # POST-HOC: the same world with ten times the data
        s = confounded_plr(5000, rng)
        return {"naive OLS (y on d)": {**naive_ols(s["y"], s["d"]), "truth": s["theta"]}, "DML, GBM nuisance": {**dml_plr(s["y"], s["d"], s["X"], "gbm", seed=seed), "truth": s["theta"]}}
    if kind == "plr_strong_gbm_posthoc":                  # POST-HOC: a higher-capacity GBM at n = 5000
        s = confounded_plr(5000, rng)
        return {"DML, strong GBM nuisance": {**dml_plr(s["y"], s["d"], s["X"], "gbm_strong", seed=seed), "truth": s["theta"]}}
    if kind == "confounded_plr":
        s = confounded_plr(N, rng)
        ests = {"naive OLS (y on d)": naive_ols(s["y"], s["d"]), "OLS with linear controls": naive_ols(s["y"], s["d"], s["X"]), "DML, ridge nuisance": dml_plr(s["y"], s["d"], s["X"], "ridge", seed=seed),
                "DML, GBM nuisance": dml_plr(s["y"], s["d"], s["X"], "gbm", seed=seed)}
    elif kind == "endogenous_iv":
        s = endogenous_iv(N, rng)
        ests = {"naive OLS (y on d)": naive_ols(s["y"], s["d"]), "2SLS": iv_2sls(s["y"], s["d"], s["z"])}
    elif kind == "did_parallel":
        s = did_panel(N // 2, rng)
        ests = {"naive post-period difference": naive_ols(s["y"][s["post"] == 1], s["treated"][s["post"] == 1]), "DiD": did_2x2(s["y"], s["treated"], s["post"], s["unit"])}
    elif kind == "did_diverging":
        s = did_panel(N // 2, rng, trend_gap=0.8)
        ests = {"naive post-period difference": naive_ols(s["y"][s["post"] == 1], s["treated"][s["post"] == 1]), "DiD": did_2x2(s["y"], s["treated"], s["post"], s["unit"])}
    elif kind == "heterogeneous":
        s = heterogeneous_effect(N, rng)
        W = np.column_stack([np.ones(N), s["X"][:, 0]])
        fit = r_learner(s["y"], s["d"], s["X"], W, "gbm", seed=seed)
        naive = naive_ols(s["y"], s["d"])
        return {"R-learner: corr(tau_hat, tau)": {"theta": float(np.corrcoef(fit["tau"], s["tau"])[0, 1]), "se": np.nan, "truth": np.nan},
                "R-learner: slope on x0": {"theta": float(fit["gamma"][1]), "se": np.nan, "truth": 1.0},
                "R-learner: intercept": {"theta": float(fit["gamma"][0]), "se": np.nan, "truth": 0.5},
                "naive OLS (average effect)": {**naive, "truth": 0.5}}
    else:
        raise ValueError(kind)
    return {k: {**v, "truth": s["theta"]} for k, v in ests.items()}


def validation(logger) -> pd.DataFrame:
    rows = []
    for kind in ("confounded_plr", "endogenous_iv", "did_parallel", "did_diverging", "heterogeneous", "plr_oracle_posthoc", "plr_n5000_posthoc", "plr_strong_gbm_posthoc"):
        t0 = time.perf_counter()
        n_reps = 40 if kind == "plr_strong_gbm_posthoc" else (100 if kind == "plr_n5000_posthoc" else REPS)
        reps = Parallel(n_jobs=-1)(delayed(one_rep)(kind, 1000 + i) for i in range(n_reps))
        for name in reps[0]:
            theta = np.array([r[name]["theta"] for r in reps])
            se = np.array([r[name]["se"] for r in reps])
            truth = reps[0][name]["truth"]
            row = {"world": kind, "estimator": name, "truth": truth, "mean_estimate": float(theta.mean()), "bias": float(theta.mean() - truth) if np.isfinite(truth) else np.nan,
                   "rmse": float(np.sqrt(np.mean((theta - truth) ** 2))) if np.isfinite(truth) else np.nan,
                   "n_reps": n_reps, "status": "post-hoc" if kind.endswith("posthoc") else "declared", "coverage_95": float(np.mean(np.abs(theta - truth) <= 1.96 * se)) if np.isfinite(se).all() else np.nan}
            rows.append(row)
        logger.info("%s: %d replications in %.0fs", kind, n_reps, time.perf_counter() - t0)
    return pd.DataFrame(rows)


def application(context, cfg, logger, fdr: float):
    feats = pd.read_csv(context.tables / "stage28_features_by_statement.csv", parse_dates=["date"])
    prices = context.market_data().prices
    returns = prices.pct_change()
    index = pd.DatetimeIndex(prices.index)
    sigma_d = ewma_sigma(returns, float(cfg.get("forecasting.gaussian.vol_halflife", 40)), 1)
    feats = feats.sort_values("date").reset_index(drop=True)
    feats["tone_change"] = feats["tone"].diff()
    feats["tone_prev"] = feats["tone"].shift(1)
    feats["tone_change_std"] = (feats["tone_change"] - feats["tone_change"].mean()) / feats["tone_change"].std()
    rows, did_rows = [], []
    for _, r in feats.dropna(subset=["tone_change"]).iterrows():
        if r["date"] not in index:
            continue
        p = index.get_loc(r["date"])
        if p < 25:
            continue
        for asset in ("TLT", "SPY"):
            if asset not in prices.columns or not np.isfinite(returns[asset].iloc[p - 21:p + 1]).all():
                continue
            s_prev = sigma_d[asset].iloc[p - 1]
            rows.append({"date": r["date"], "asset": asset, "y": returns[asset].iloc[p] / s_prev, "d": r["tone_change_std"], "action": r["action"], "change": r["change"],
                         "tone_prev": r["tone_prev"], "ret5": float(prices[asset].iloc[p - 1] / prices[asset].iloc[p - 6] - 1.0) / (s_prev * np.sqrt(5)), "vol": s_prev * np.sqrt(252)})
        for asset in ("TLT", "IEF", "GLD"):
            if asset in prices.columns and np.isfinite(returns[asset].iloc[p - 5:p + 1]).all():
                for post, q in ((1, p), (0, p - 5)):
                    did_rows.append({"event": str(r["date"].date()), "asset": asset, "treated": float(asset in ("TLT", "IEF")), "post": float(post), "y": abs(returns[asset].iloc[q] / sigma_d[asset].iloc[q - 1])})
    panel = pd.DataFrame(rows)
    context.save_table(panel, "stage37_fomc_panel.csv", index=False)
    out, cate = [], []
    for asset, g in panel.groupby("asset"):
        g = g.dropna()
        X = g[["action", "change", "tone_prev", "ret5", "vol"]].to_numpy()
        X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
        res = {"asset": asset, "n": len(g), "naive_theta": naive_ols(g["y"], g["d"])["theta"], "naive_p": naive_ols(g["y"], g["d"])["p_value"],
               "ols_controls_theta": naive_ols(g["y"], g["d"], X)["theta"], "ols_controls_p": naive_ols(g["y"], g["d"], X)["p_value"]}
        for learner in ("ridge", "gbm"):
            e = dml_plr(g["y"], g["d"], X, learner, seed=7)
            res.update({f"dml_{learner}_theta": e["theta"], f"dml_{learner}_se": e["se"], f"dml_{learner}_p": e["p_value"]})
        out.append(res)
        terc = pd.qcut(g["vol"], 3, labels=False)
        W = np.column_stack([(terc == k).astype(float) for k in range(3)])
        fit = r_learner(g["y"], g["d"], X, W, "ridge", seed=7, ridge=1e-2)
        cate.append({"asset": asset, **{f"effect_vol_tercile_{k + 1}": float(fit["gamma"][k]) for k in range(3)}})
    table = pd.DataFrame(out).set_index("asset")
    table["declared_estimate"] = table["dml_gbm_theta"]
    table["declared_p"] = table["dml_gbm_p"]
    table["bh_significant"] = benjamini_hochberg(table["declared_p"], fdr)
    context.save_table(table, "stage37_fomc_effects.csv")
    context.save_table(pd.DataFrame(cate), "stage37_fomc_effect_by_volatility.csv", index=False)
    logger.info("FOMC tone change on same-day standardised return:\n%s", table.round(4).T.to_string())
    did = pd.DataFrame(did_rows)
    d = did_2x2(did["y"], did["treated"], did["post"], did["event"])
    context.save_table(pd.DataFrame([{**d, "treated_assets": "TLT, IEF", "control_asset": "GLD"}]), "stage37_fomc_did.csv", index=False)
    logger.info("DiD, announcement-day |standardised return|, rates-sensitive vs gold: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()})
    return panel, table, pd.DataFrame(cate), d


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 37: causal inference").parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    node = (cfg.get("frontier", {}) or {})["causal"]
    fdr = float(node["decisions"]["fdr"])
    logger.info("=" * 72)
    logger.info("STAGE 37 | causal inference (Generation 5)")
    logger.info("=" * 72)
    val = validation(logger)
    context.save_table(val, "stage37_validation.csv", index=False)
    logger.info("validation:\n%s", val.round(3).to_string(index=False))
    panel, table, cate, did = application(context, cfg, logger, fdr)
    h_tone = bool(table["bh_significant"].any())

    fig, axes = new_axes(1, 3, figsize=(17, 5.4))
    ax = axes[0]
    sel = val[val["world"].isin(["confounded_plr", "endogenous_iv", "did_parallel", "did_diverging", "plr_oracle_posthoc", "plr_n5000_posthoc", "plr_strong_gbm_posthoc"])].reset_index(drop=True)
    colours = [PALETTE[3] if ("naive" in e or e.startswith("OLS")) else PALETTE[0] for e in sel["estimator"]]
    ax.barh(range(len(sel)), sel["bias"].to_numpy(), color=colours)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(sel)), [f"{w}: {e}" for w, e in zip(sel["world"], sel["estimator"])], fontsize=7)
    ax.set_xlabel("Bias of the estimate (truth known), pink = ignores the problem")
    ax.set_title("Validation: bias in simulated worlds")
    ax = axes[1]
    cov = sel.dropna(subset=["coverage_95"])
    ax.barh(range(len(cov)), cov["coverage_95"].to_numpy(), color=[PALETTE[3] if ("naive" in e or e.startswith("OLS")) else PALETTE[0] for e in cov["estimator"]])
    ax.axvline(0.95, color="black", linestyle=":")
    ax.set_yticks(range(len(cov)), [f"{w}: {e}" for w, e in zip(cov["world"], cov["estimator"])], fontsize=7)
    ax.set_xlabel("Share of simulations whose 95% interval contains the truth")
    ax.set_title("Validation: coverage")
    ax = axes[2]
    names = ["naive", "ols_controls", "dml_ridge", "dml_gbm"]
    width = 0.2
    for j, n in enumerate(names):
        vals = [table.loc[a, f"{n}_theta"] for a in table.index]
        ax.bar(np.arange(len(table)) + (j - 1.5) * width, vals, width, label=n, color=PALETTE[j])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(table)), [f"{a} (n={int(table.loc[a, 'n'])})" for a in table.index])
    ax.set_ylabel("Effect of a 1-s.d. hawkish tone change on the\nstandardised same-day return")
    ax.set_title("FOMC application, four estimators")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 73. Causal estimators: validated on known truth, then applied to FOMC statements", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, context.figure("fig73_causal_validation.png"), "Do the causal estimators recover a known effect where the naive regression fails, and what do they say about FOMC tone?", 73)

    fig, axes = new_axes(1, 2, figsize=(12, 5))
    ax = axes[0]
    for a, colour in zip(panel["asset"].unique(), PALETTE):
        g = panel[panel["asset"] == a]
        ax.scatter(g["d"], g["y"], s=14, alpha=0.6, color=colour, label=a)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Tone change from the previous statement (standardised)")
    ax.set_ylabel("Same-day return / prior-day daily sigma")
    ax.set_title("The raw data behind the estimates")
    ax.legend(fontsize=8)
    ax = axes[1]
    for k, a in enumerate(cate["asset"]):
        ax.bar(np.arange(3) + k * 0.35, [cate.loc[k, f"effect_vol_tercile_{j + 1}"] for j in range(3)], 0.35, label=a, color=PALETTE[k])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(3) + 0.175, ["low prior volatility", "middle", "high"])
    ax.set_ylabel("R-learner effect of tone change")
    ax.set_title("Does the effect vary with prior volatility?")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 74. FOMC tone and same-day returns: data and heterogeneity", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, context.figure("fig74_causal_fomc.png"), "What does the sample of statement days look like, and is there any sign that volatility changes the response to tone?", 74)

    v = val.set_index(["world", "estimator"])
    context.registry.log(
        "A more hawkish FOMC statement (a standardised increase in the Stage 28 tone score) changes the same-day standardised return of TLT or SPY (double machine learning, Benjamini-Hochberg across the two).",
        stage=STAGE, parameters={"estimator": "DML partially linear, GBM nuisance, 5-fold cross-fitting", "fdr": fdr, "n_statements": int(table["n"].max())},
        results={**{f"dml_theta_{a}": float(table.loc[a, "declared_estimate"]) for a in table.index}, **{f"dml_p_{a}": float(table.loc[a, "declared_p"]) for a in table.index},
                 "validation_dml_gbm_bias": float(v.loc[("confounded_plr", "DML, GBM nuisance"), "bias"]), "validation_naive_bias": float(v.loc[("confounded_plr", "naive OLS (y on d)"), "bias"]),
                 "validation_2sls_bias": float(v.loc[("endogenous_iv", "2SLS"), "bias"]), "validation_ols_iv_bias": float(v.loc[("endogenous_iv", "naive OLS (y on d)"), "bias"])},
        decision="retain" if h_tone else "reject", test_period="2006 statements onward (n about 170)",
        notes="Decided by the declared rule, but fragile: the validation shows DML with these nuisance learners is biased at n = 500 when the nuisance functions are hard to learn, "
              "tone change is not randomised, controls are few and n is small. IV is not applied (no instrument). The DiD estimate and R-learner heterogeneity are reported, not judged.")
    logger.info("STAGE 37 complete: h_tone=%s", h_tone)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
