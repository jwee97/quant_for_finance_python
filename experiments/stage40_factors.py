"""Stage 40 - Factor attribution of the platform's books (Generation 5, descriptive).

Each Generation 1 book's net daily excess return is regressed on the Fama-French five factors plus momentum (Kenneth French's data library, cached under
``data/raw/factors``), with Newey-West (lag 5) standard errors. The factors are EQUITY factors; half the universe is bonds, gold and commodities they do not span,
so the intercept is not unexplained skill. That limitation was declared before the result (``config/diagnostics.yaml``, section ``factor_attribution``).
Figure 79.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from experiments.context import build_context
from experiments.strategies import cached_ladder
from src.backtest.engine import BacktestEngine
from src.data.factors import load_factors
from src.utils.plotting import PALETTE, new_axes, save_figure

STAGE = "stage40_factors"
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def regress(excess: pd.Series, factors: pd.DataFrame, lag: int = 5) -> dict:
    X = sm.add_constant(factors[FACTORS])
    fit = sm.OLS(excess.loc[X.index], X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    out = {"n_days": int(fit.nobs), "alpha_annual": float(fit.params["const"] * 252), "alpha_t": float(fit.tvalues["const"]), "alpha_p": float(fit.pvalues["const"]),
           "r_squared": float(fit.rsquared)}
    for f in FACTORS:
        out[f"beta_{f}"] = float(fit.params[f])
        out[f"t_{f}"] = float(fit.tvalues[f])
    return out


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Stage 40: factor attribution").parse_args(argv)
    context, logger = build_context(STAGE, generation=5)
    cfg = context.config
    node = (cfg.get("diagnostics", {}) or {})["factor_attribution"]
    logger.info("=" * 72)
    logger.info("STAGE 40 | factor attribution (Generation 5, descriptive)")
    logger.info("=" * 72)
    factors = load_factors(cfg.path("raw") / "factors")
    market = context.market_data()
    engine = BacktestEngine.from_config(cfg)
    names = list(node["books"])
    books = cached_ladder(market, cfg, context.processed, names)
    returns = market.returns()
    series = {n: engine.run(w, returns, n, market.investable, apply_vol_target=n.startswith(("M3", "M4", "M5"))).net_returns for n, w in books.items()}
    rows = {}
    for name, r in series.items():
        live = r.loc[r.ne(0).idxmax():].dropna()
        joined = pd.concat([(live - factors["RF"].reindex(live.index)).rename("excess"), factors[FACTORS]], axis=1, sort=False).dropna()
        res = regress(joined["excess"], joined[FACTORS])
        res["first_date"], res["last_date"] = str(joined.index[0].date()), str(joined.index[-1].date())
        rows[name] = res
    table = pd.DataFrame(rows).T
    context.save_table(table, "stage40_factor_regressions.csv")
    logger.info("factor regressions:\n%s", table[["n_days", "alpha_annual", "alpha_t", "r_squared", "beta_Mkt-RF", "t_Mkt-RF", "beta_Mom", "t_Mom"]].astype(float).round(3).to_string())

    ew = series["M0_equal_weight"]
    excess = (ew - factors["RF"].reindex(ew.index)).dropna()
    joined = pd.concat([excess.rename("y"), factors[FACTORS]], axis=1, sort=False).dropna()
    betas = {}
    for end in range(756, len(joined), 21):
        w = joined.iloc[end - 756:end]
        fit = sm.OLS(w["y"], sm.add_constant(w[["Mkt-RF"]])).fit()
        betas[w.index[-1]] = float(fit.params["Mkt-RF"])
    rolling = pd.Series(betas)
    context.save_table(rolling.rename("rolling_3y_market_beta").to_frame(), "stage40_rolling_market_beta.csv")

    explained = table["r_squared"].astype(float)
    fig, axes = new_axes(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    betas_m = table[[f"beta_{f}" for f in FACTORS]].astype(float)
    x = np.arange(len(FACTORS))
    for j, n in enumerate(betas_m.index):
        ax.bar(x + (j - 2) * 0.16, betas_m.loc[n].to_numpy(), 0.16, color=PALETTE[j % len(PALETTE)], label=n.split("_")[0])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, FACTORS)
    ax.set_ylabel("Factor loading")
    ax.set_title("Loadings on equity factors")
    ax.legend(fontsize=8)
    ax = axes[1]
    a = table["alpha_annual"].astype(float) * 100
    ax.bar(range(len(a)), a.to_numpy(), color=PALETTE[0])
    for i, (n, row) in enumerate(table.iterrows()):
        ax.text(i, a.iloc[i], f"t={float(row['alpha_t']):.1f}", ha="center", va="bottom" if a.iloc[i] >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(a)), [n.split("_")[0] for n in a.index])
    ax.set_ylabel("Intercept, % per year (includes bond, gold and commodity premia)")
    ax.set_title("What equity factors leave unexplained")
    ax = axes[2]
    ax.plot(rolling.index, rolling.to_numpy(), color=PALETTE[0])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Beta to the market factor, trailing 3 years")
    ax.set_title(f"Equal-weight book; R-squared by book {explained.min():.2f} to {explained.max():.2f}")
    fig.suptitle("Figure 79. Factor attribution of the Generation 1 books (equity factors only)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, context.figure("fig79_factor_attribution.png"), "How much of each book is exposure to known equity factors, and what is left over?", 79)

    context.registry.log(
        "Descriptive: how much of the Generation 1 books' returns do the Fama-French five factors plus momentum explain, and what intercept remains?",
        stage=STAGE, parameters={"books": names, "factors": FACTORS, "hac_lag": 5},
        results={**{f"alpha_annual_{n.split('_')[0]}": float(table.loc[n, "alpha_annual"]) for n in table.index}, **{f"alpha_t_{n.split('_')[0]}": float(table.loc[n, "alpha_t"]) for n in table.index},
                 **{f"r_squared_{n.split('_')[0]}": float(table.loc[n, "r_squared"]) for n in table.index}},
        decision="record", notes="Equity factors only: bonds, gold and commodities are unspanned, so the intercept is not unexplained skill.")
    logger.info("STAGE 40 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
