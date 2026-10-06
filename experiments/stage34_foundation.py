"""Stage 34 - A zero-shot time-series foundation model (Generation 5, descriptive).

Chronos-Bolt (small) is run with NO fitting on the 252 normalised daily returns before each Stage 19 origin. The median path for the next
21 days, summed and scaled by the origin's daily EWMA volatility, is a 21-day return forecast; the Stage 19 EWMA sigma is the spread. The stage
is DESCRIPTIVE by declaration (``config/frontier.yaml``, section ``foundation``): the pre-training corpus is only partly documented and its dates
overlap this sample, so a good score could not be taken as clean evidence while a bad one is. No hypothesis is retained or rejected here.

The model downloads from the Hugging Face hub at run time (optional dependency ``chronos-forecasting``); if it is unavailable the stage
writes nothing new and says so.

Figure 68.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.context import build_context
from src.models.deep_forecast import daily_sigma, normalised_windows
from src.models.probabilistic import crps_gaussian
from src.utils.plotting import PALETTE, new_axes, save_figure
from src.validation.forecast_tests import diebold_mariano

STAGE = "stage34_foundation"
KEYS = ["origin", "asset"]


def load_pipeline(name: str):
    import torch
    from chronos import ChronosBoltPipeline

    return ChronosBoltPipeline.from_pretrained(name, device_map="cpu", dtype=torch.float32)


def median_sum(pipeline, X: np.ndarray, horizon: int, batch: int = 256) -> np.ndarray:
    import torch

    out = []
    for i in range(0, len(X), batch):
        q, _ = pipeline.predict_quantiles(torch.as_tensor(X[i:i + batch], dtype=torch.float32), prediction_length=horizon, quantile_levels=[0.5])
        out.append(q[:, :, 0].sum(dim=1).numpy())
    return np.concatenate(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 34: zero-shot foundation model")
    parser.parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    logger.info("=" * 72)
    logger.info("STAGE 34 | zero-shot foundation model (Generation 5, descriptive)")
    logger.info("=" * 72)
    try:
        import chronos  # noqa: F401
    except ImportError:
        logger.warning("chronos-forecasting is not installed (pip install chronos-forecasting einops); stage 34 skipped, no outputs written")
        return 0
    name = "amazon/chronos-bolt-small"
    try:
        pipeline = load_pipeline(name)
    except Exception as exc:                                   # offline, hub down: say so, do not invent a result
        logger.warning("could not load %s (%s); stage 34 skipped, no outputs written", name, exc)
        return 0
    revision = None
    try:
        from huggingface_hub import model_info

        revision = model_info(name).sha
    except Exception:
        pass
    logger.info("model %s, hub revision %s", name, revision)

    horizon = int(cfg.get("forecasting.horizon_days", 21))
    window = int((cfg.get("deeplearning", {}) or {})["task"]["window"])
    frozen = pd.read_csv(context.tables / "stage19_predictions_price_only.csv", parse_dates=["origin"])
    market = context.market_data()
    returns = market.prices.pct_change()
    index = pd.DatetimeIndex(market.prices.index)
    assets = list(market.prices.columns)
    sigma_d = daily_sigma(returns, float(cfg.get("forecasting.gaussian.vol_halflife", 40)))
    scored = frozen.dropna(subset=["target", "sigma", "mu_benchmark"]).reset_index(drop=True)      # rows whose outcome has matured
    pos = np.array([index.get_loc(o) for o in scored["origin"]])
    a = np.array([assets.index(x) for x in scored["asset"]])
    X, ok = normalised_windows(returns, sigma_d, pos, a, window)
    logger.info("%d Stage 19 rows with a realised outcome, %d with a full window", len(scored), int(ok.sum()))
    f = scored[ok].copy()
    f["mu_ridge"] = f["mu"]
    f["mu_z"] = median_sum(pipeline, X[ok], horizon)
    f["mu"] = f["mu_z"] * sigma_d.to_numpy()[pos[ok], a[ok]]
    context.save_table(f[KEYS + ["mu_z", "mu", "sigma", "target", "mu_benchmark"]], "stage34_predictions_chronos.csv", index=False)

    def by_origin(mu_col: str) -> pd.Series:
        return pd.Series(crps_gaussian(f["target"].to_numpy(), f[mu_col].to_numpy(), f["sigma"].to_numpy()), index=f.index).groupby(f["origin"]).mean()

    c_model, c_hist, c_ridge = by_origin("mu"), by_origin("mu_benchmark"), by_origin("mu_ridge")
    ics = [spearmanr(g["mu"], g["target"])[0] for _, g in f.groupby("origin") if len(g) >= 5 and g["mu"].nunique() > 1]
    rows = []
    for label, other in (("historical_mean", c_hist), ("stage19_ridge", c_ridge)):
        t = diebold_mariano(c_model, other, lag=0, alternative="two-sided")
        rows.append({"against": label, "n_origins": len(c_model), "mean_crps_chronos": float(c_model.mean()), "mean_crps_other": float(other.mean()),
                     "mean_crps_difference": float(t["mean_loss_difference"]), "p_value": float(t["p_value"])})
    # Stage 33 comparison on the common rows (reported, not judged)
    for kind in ("nbeats", "timemixer"):
        p = pd.read_csv(context.tables / f"stage33_predictions_{kind}.csv", parse_dates=["origin"])[KEYS + ["mu"]].rename(columns={"mu": "mu_other"})
        g = f.merge(p, on=KEYS)
        d = (pd.Series(crps_gaussian(g["target"].to_numpy(), g["mu"].to_numpy(), g["sigma"].to_numpy()) - crps_gaussian(g["target"].to_numpy(), g["mu_other"].to_numpy(), g["sigma"].to_numpy()),
                       index=g.index).groupby(g["origin"]).mean())
        t = diebold_mariano(d, pd.Series(0.0, index=d.index), lag=0, alternative="two-sided")
        rows.append({"against": kind, "n_origins": len(d), "mean_crps_chronos": float("nan"), "mean_crps_other": float("nan"),
                     "mean_crps_difference": float(t["mean_loss_difference"]), "p_value": float(t["p_value"])})
    table = pd.DataFrame(rows).set_index("against")
    summary = pd.DataFrame([{"mean_rank_ic": float(np.nanmean(ics)), "mean_abs_mu": float(f["mu"].abs().mean()), "mean_abs_mu_hist_mean": float(f["mu_benchmark"].abs().mean()),
                             "share_positive_forecasts": float((f["mu"] > 0).mean()), "n_rows": len(f), "model": name, "hub_revision": revision}])
    context.save_table(table, "stage34_tests.csv")
    context.save_table(summary, "stage34_summary.csv", index=False)
    (context.tables / "stage34_model.json").write_text(json.dumps({"model": name, "hub_revision": revision}, indent=2))
    logger.info("Chronos-Bolt zero-shot against the benchmarks (CRPS difference, negative favours Chronos):\n%s", table.round(6).to_string())
    logger.info("summary:\n%s", summary.round(5).to_string(index=False))

    adv_hist, adv_ridge = c_hist - c_model, c_ridge - c_model
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ax.plot(adv_hist.cumsum().index, adv_hist.cumsum().to_numpy(), color=PALETTE[0], label="against the historical mean")
    ax.plot(adv_ridge.cumsum().index, adv_ridge.cumsum().to_numpy(), color=PALETTE[3], label="against the Stage 19 ridge")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative CRPS advantage (above zero = better)")
    ax.set_title("Zero-shot Chronos-Bolt, month by month")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(f["mu_z"], bins=50, color=PALETTE[0])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Summed median path (normalised units)")
    ax.set_ylabel("Rows")
    ax.set_title("What the model forecasts")
    ax = axes[2]
    ax.hist(ics, bins=30, color=PALETTE[2])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(float(np.nanmean(ics)), color=PALETTE[3], linestyle="--")
    ax.set_xlabel("Rank IC across the 15 ETFs, one value per month")
    ax.set_title(f"Monthly rank IC (mean {np.nanmean(ics):+.3f})")
    fig.suptitle("Figure 68. A zero-shot foundation model (descriptive; pre-training overlap is unknown)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, context.figure("fig68_foundation_zero_shot.png"), "Does a pre-trained time-series model, used without any fitting, say anything about next month's ETF returns?", 68)

    context.registry.log(
        "Descriptive: how does zero-shot Chronos-Bolt (small) forecast 21-day ETF returns relative to the historical mean and the Stage 19 ridge?",
        stage=STAGE, parameters={"model": name, "hub_revision": revision, "context_days": window, "horizon": horizon},
        results={"crps_difference_vs_hist_mean": float(table.loc["historical_mean", "mean_crps_difference"]), "p_value_vs_hist_mean": float(table.loc["historical_mean", "p_value"]),
                 "crps_difference_vs_ridge": float(table.loc["stage19_ridge", "mean_crps_difference"]), "p_value_vs_ridge": float(table.loc["stage19_ridge", "p_value"]),
                 "mean_rank_ic": float(np.nanmean(ics)), "n_rows": len(f)},
        decision="record", test_period="2011-01 onward, monthly origins, 15 ETFs",
        notes="No decision by declaration. The pre-training corpus overlaps the sample and may contain these prices; a favourable result would not be clean evidence.",
    )
    logger.info("STAGE 34 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
