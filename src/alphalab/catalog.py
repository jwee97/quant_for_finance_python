"""Where every item of the five-book roadmap lives in this repository, and what is not built.

The roadmap (Grinold and Kahn, *Active Portfolio Management* and *Advances in Active Portfolio Management*; Cajas, *Advanced Portfolio Optimization*; Paleologo, *Advanced Portfolio Management*; López de Prado,
*Advances in Financial Machine Learning*) lists capabilities with priorities. This file is the audit of it: for each item the ``status``

* ``built``: written for this item; * ``existing``: the repository already had it (the entry points to it); * ``partial``: some of it exists, the note says what is missing; * ``not built``: nothing exists, the note says why.

``where`` is ``py:module.object`` or ``allocator:name``; ``tests/test_alphalab_catalog.py`` imports every one of them and checks that ``docs/roadmap_audit.md`` is this file written out, so neither can drift. Chapter
numbers are as the roadmap gave them (the books' own numbering was not available to check against).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..books.catalog import resolve


@dataclass(frozen=True)
class Item:
    priority: str
    area: str
    item: str
    source: str
    status: str
    where: tuple
    note: str = ""


A, B, D, C, E, V = ("Alpha engine", "Risk engine", "Portfolio optimisation", "Dynamic trading and capacity", "Fundamental-quant and attribution", "Validation (López de Prado)")
ITEMS: list[Item] = [
    # ---------------------------------------------------------------------------------------------------------------- alpha
    Item("P0", A, "Signal registry with versions and information timestamps", "Active PM", "built", ("py:src.alphalab.registry.SignalRegistry", "py:src.alphalab.registry.pit_violations"), "a content hash decides the version; each carries its information lag; a point-in-time check counts values known after their date"),
    Item("P0", A, "IC analysis: Pearson/rank, rolling, stability, by regime", "Active PM ch. 10-13", "built", ("py:src.alphalab.ic.ic_summary", "py:src.alphalab.ic.rolling_ic", "py:src.alphalab.ic.ic_by_regime", "py:src.alphalab.ic.stability"), "Newey-West t-statistic for overlapping forward windows; sign stability over blocks"),
    Item("P0", A, "IC decay and information horizon", "Active PM; Advances ch. 5-11", "built", ("py:src.alphalab.ic.lagged_ic", "py:src.alphalab.ic.fit_decay", "py:src.alphalab.ic.delay_cost"), "single-period IC at each lag, exponential fit weighted by sampling error; no half-life is reported for a signal whose first-lag IC is noise"),
    Item("P0", A, "Alpha calibration: alpha = IC x omega x z", "Active PM ch. 10", "built", ("py:src.alphalab.calibrate.calibrate", "py:src.alphalab.calibrate.reliability"), "matured ICs only, shrunk by their own signal-to-noise; a reliability slope says whether the alphas are the right size"),
    Item("P0", A, "Signal combination weighted by IC covariance", "Active PM ch. 11", "existing", ("py:src.equity.alpha_model.optimal_alpha", "py:src.equity.alpha_model.max_ir_weights", "py:src.framework.ic_combination.optimal_ic_weights"), "walk-forward max-IR weights on orthogonalised, standardised factors"),
    Item("P0", A, "Effective breadth", "Active PM ch. 6", "built", ("py:src.alphalab.breadth.effective_breadth", "py:src.alphalab.breadth.effective_cross_section"), "participation ratio of the residual-return correlation matrix x non-overlapping horizons x the share of the signal that is new; an estimate, labelled as one"),
    Item("P0", A, "Fundamental law and implied skill", "Active PM ch. 6", "built", ("py:src.alphalab.breadth.fundamental_law", "py:src.alphalab.breadth.implied_ic"), "IR = TC x IC x sqrt(BR)"),
    Item("P0", A, "Transfer coefficient", "Active PM ch. 14; Clarke et al.", "built", ("py:src.alphalab.breadth.transfer_coefficient", "py:src.alphalab.breadth.constrained_transfer"), "the cosine of risk-adjusted alpha and risk-adjusted holdings: exactly the ratio of information ratios"),
    # ---------------------------------------------------------------------------------------------------------------- risk
    Item("P0", B, "Time-series factor model", "Paleologo ch. 3-5", "built", ("py:src.riskengine.factor_model.fit_time_series",), "betas to given factors; factor covariance and specific variance, optionally exponentially weighted"),
    Item("P0", B, "Cross-sectional (fundamental) factor model", "Paleologo ch. 3-5", "built", ("py:src.riskengine.factor_model.fit_cross_sectional",), "per-date regression on lagged exposures; sector returns constrained to net to zero; needs exposures you supply (Fundamentals data in the library, not in the page)"),
    Item("P0", B, "Statistical factor model", "Paleologo ch. 3-5", "built", ("py:src.riskengine.factor_model.fit_statistical",), "principal components; the number of factors from the Marchenko-Pastur bound"),
    Item("P0", B, "Risk decomposition by factor, group, asset; marginal contributions", "Paleologo ch. 5", "built", ("py:src.riskengine.decompose.risk_decomposition", "py:src.riskengine.decompose.active_risk"), "contributions add exactly to the volatility"),
    Item("P1", B, "Factor risk budgeting", "Paleologo ch. 7; Cajas ch. 10", "built", ("py:src.riskengine.decompose.factor_risk_budget_weights",), "weights whose risk shares by factor group follow a budget"),
    Item("P1", B, "Factor-neutral and benchmark-relative holdings; hedging", "Paleologo ch. 7", "built", ("py:src.riskengine.hedge.neutralise", "py:src.riskengine.hedge.min_variance_hedge"), "smallest change in risk (or in weights) that zeroes chosen exposures; minimum-variance hedge with named instruments"),
    Item("P0", B, "Forecast versus realised risk; factor drift", "Paleologo ch. 4", "built", ("py:src.riskengine.monitor.bias_statistic", "py:src.riskengine.monitor.portfolio_forecast_check", "py:src.riskengine.monitor.exposure_drift"), "the bias statistic with its confidence band"),
    Item("P0", B, "Ledoit-Wolf shrinkage and random-matrix denoising", "Cajas ch. 3-4", "existing", ("py:src.portfolio.covariance.ledoit_wolf_shrinkage", "py:src.portfolio.covariance.pca_denoised_covariance", "py:src.portfolio.covariance.ewma_covariance"), "also a PCA factor covariance (src.ml.risk_models)"),
    Item("P0", B, "Detoning and graphical-lasso covariance", "Cajas ch. 3-4", "built", ("py:src.riskengine.covariance.detone_correlation", "py:src.riskengine.covariance.graphical_lasso_covariance"), "both reach estimate_covariance (graphical_lasso, factor); detoned correlation is for clustering, not inversion"),
    # ---------------------------------------------------------------------------------------------------------------- portfolios
    Item("P0", D, "Cost-aware constrained optimisation", "Active PM ch. 14-16; Cajas ch. 9", "existing", ("allocator:tca_mvo", "py:src.equity.portfolio.optimise_book", "py:src.portfolio.constraints.Constraints"), "costs inside the optimiser; long/short books with sector, beta and turnover limits"),
    Item("P1", D, "Black-Litterman", "Cajas ch. 5", "existing", ("py:src.portfolio.black_litterman.black_litterman_posterior", "allocator:black_litterman"), "equilibrium prior and relative views; Idzorek confidence in the allocator"),
    Item("P1", D, "Augmented and Bayesian Black-Litterman", "Cajas ch. 5", "built", ("py:src.portfolio.bayesian.bayesian_weights", "py:src.portfolio.augmented_bl.augmented_posterior"), "Bayesian mean-variance (normal-inverse-Wishart) existed; views on factors as well as assets are the augmented-universe construction, which reduces to ordinary BL with asset views only. The book's own formulas for its augmented model were not available to check against"),
    Item("P1", D, "Drawdown and tail-risk optimisation: CVaR", "Cajas ch. 7-8", "existing", ("py:src.portfolio.cvar_optimize.mean_cvar_weights",), "linear programme on scenarios"),
    Item("P1", D, "Drawdown and tail-risk optimisation: CDaR, maximum drawdown, EVaR", "Cajas ch. 7-8", "built", ("py:src.portfolio.drawdown.cdar_weights", "py:src.portfolio.drawdown.max_drawdown_weights", "py:src.portfolio.drawdown.evar_weights"), "linear programmes on the scenario path (uncompounded) and a smooth convex programme for EVaR"),
    Item("P1", D, "HERC", "Cajas ch. 12", "existing", ("py:src.portfolio.hierarchical.herc_weights", "py:src.portfolio.hierarchical.hrp_weights"), "Ward clusters chosen by silhouette"),
    Item("P1", D, "Nested clustered optimisation", "Cajas ch. 12", "built", ("py:src.portfolio.nco.nco_weights",), "minimum variance, maximum Sharpe or risk parity inside and between clusters; optional detoning for the clustering"),
    Item("P1", D, "Risk budgeting / risk parity", "Cajas ch. 10", "existing", ("py:src.portfolio.risk_parity.risk_parity_weights", "allocator:risk_budget"), "Newton and SLSQP, budgets by asset class"),
    Item("P2", D, "Graph-based portfolios", "Cajas ch. 13", "built", ("py:src.portfolio.graph_portfolio.hsp_weights", "py:src.portfolio.graph_portfolio.centrality_weights", "py:src.portfolio.graph_constrained.graph_capped_weights"), "MST/TMFG, centrality and hierarchical sensitivity parity existed; a cap on the weight inside every asset's neighbourhood of the graph is new, and refuses a hub that makes the cap impossible"),
    Item("P2", D, "Robust optimisation", "Cajas ch. 11", "built", ("py:src.portfolio.mean_variance.robust_weights", "py:src.portfolio.mean_variance.resampled_weights", "py:src.portfolio.robust_wasserstein.wasserstein_cvar_weights"), "ellipsoidal uncertainty and Michaud resampling existed; the worst-case CVaR over a Wasserstein ball (CVaR + eps ||w|| / beta, attained by moving the worst scenarios) is new, solved exactly by cutting planes"),
    Item("P2", D, "Near-optimal centering", "Cajas ch. 11", "built", ("py:src.portfolio.near_optimal.near_optimal_center", "py:src.portfolio.near_optimal.weight_dispersion"), "the analytic centre of the portfolios within a tolerance of the best utility; resampling stability measured"),
    Item("P1", D, "Optimiser comparison on identical windows, costs and caps", "Cajas ch. 1", "built", ("py:src.portfolio.lab.compare",), "19 methods, six covariance estimators; fallbacks counted; cap and weight stability reported; one method can be kept in a research session"),
    Item("P2", D, "Higher-moment optimisation", "Cajas ch. 3, 7", "not built", (), "deliberately: skewness and kurtosis are poorly estimated for large portfolios; add only against a simpler benchmark that it beats out of sample"),
    Item("P2", D, "Copula scenarios and tail dependence", "Cajas ch. 6, 14", "existing", ("py:src.probability.copulas.simulate_joint", "py:src.scenarios.generators.copula", "py:src.probability.copulas.empirical_tail_dependence"), "Gaussian, Student, Clayton, Gumbel, Frank; empirical or EVT marginals"),
    # ---------------------------------------------------------------------------------------------------------------- dynamic
    Item("P0", C, "Multi-period optimisation, trade rate, no-trade region", "Advances ch. 6-11", "existing", ("py:src.equity.multiperiod.lq_solve", "py:src.equity.multiperiod.mpc_plan", "py:src.equity.multiperiod.no_trade_region", "allocator:multi_period"), "Garleanu-Pedersen closed form; model-predictive control with constraints"),
    Item("P0", C, "Alpha-decay paths and trading-rule comparison", "Advances ch. 5-11", "built", ("py:src.alphalab.dynamic.decay_path", "py:src.alphalab.dynamic.simulate", "py:src.alphalab.dynamic.compare_policies"), "full, partial adjustment, no-trade band and threshold on the same forecasts, covariance and costs"),
    Item("P0", C, "Implementation loss: paper to traded", "Advances ch. 9-11", "built", ("py:src.alphalab.dynamic.implementation_waterfall",), "realised Sharpe as position limits, costs, a cost-aware policy and delay are added in turn"),
    Item("P1", C, "Capacity from market impact", "Advances ch. 31", "built", ("py:src.alphalab.capacity.capacity_curve", "py:src.alphalab.capacity.capacity", "py:src.backtest.impact.capacity_curve"), "square-root law; capacity is the capital at which the Sharpe halves; the backtest module has a fuller version with ADV-banded execution"),
    Item("P1", C, "Crowding and common exposures", "Advances ch. 30", "built", ("py:src.alphalab.capacity.crowding_report",), "position cosine, factor-exposure cosine and return correlation of books; effective independent books"),
    Item("P1", C, "Leverage and optimal gearing", "Advances ch. 29", "built", ("py:src.alphalab.capacity.kelly_leverage", "py:src.alphalab.capacity.leverage_for_drawdown", "py:src.alphalab.capacity.growth_rate"), "Kelly with its estimation error; the stationary drawdown distribution of Brownian motion"),
    # ---------------------------------------------------------------------------------------------------------------- fundamental
    Item("P1", E, "Selection, sizing and timing attribution", "Paleologo ch. 8", "built", ("py:src.alphalab.attribution.selection_sizing_timing",), "an exact three-way split of profit against an equal-size, average-gross version of the same book"),
    Item("P1", E, "Brinson attribution", "Active PM", "existing", ("py:src.backtest.attribution.brinson_fachler",), "allocation, selection and interaction against a benchmark"),
    Item("P1", E, "Stop-loss and leverage policy simulation", "Paleologo ch. 8", "built", ("py:src.alphalab.attribution.stop_loss", "py:src.alphalab.attribution.volatility_target"), "both causal: yesterday's equity or volatility only"),
    Item("P1", E, "Financial-statement factors", "Paleologo ch. 5; Active PM", "existing", ("py:src.equity.factors.compute_all", "py:src.equity.factors.compute"), "value, quality and revision factors from point-in-time statements (needs your fundamentals file)"),
    Item("P1", E, "Event studies: earnings and announcements", "Paleologo ch. 8", "existing", ("py:src.stats.event_study.event_study", "py:src.stats.event_study.buy_and_hold_abnormal_return"), "market, factor and constant-mean models"),
    Item("P1", E, "Consensus-relative valuation signals", "Paleologo ch. 5", "not built", (), "needs analyst-estimate data the platform does not have"),
    Item("P2", E, "Factor-mimicking portfolios", "Paleologo appendix 11.5", "built", ("py:src.riskengine.factor_model.fit_cross_sectional",), "the per-date regression slopes are the returns of factor-mimicking portfolios; custom exposures can be passed in"),
    Item("P2", E, "Smart-beta decomposition: factor return versus skill", "Advances ch. 21-25", "not built", (), "needs a manager's holdings history; the building blocks (factor model, attribution) are here"),
    # ---------------------------------------------------------------------------------------------------------------- AFML
    Item("P0", V, "Triple-barrier labels, meta-labelling, sample uniqueness, purged CV, CPCV, deflated Sharpe", "AFML", "existing", ("py:src.finml.pipeline.run_pipeline", "py:src.finml.cv.purged_folds", "py:src.finml.events.triple_barrier_events"), "the Financial ML Lab"),
]
STATUSES = {"built", "existing", "partial", "not built"}


def table():
    import pandas as pd

    return pd.DataFrame([{"priority": i.priority, "area": i.area, "item": i.item, "source": i.source, "status": i.status, "where": ", ".join(w.split(":", 1)[1] for w in i.where), "note": i.note} for i in ITEMS])


def check() -> list[str]:
    """The locations that do not import, as messages (empty when the catalogue is true)."""
    bad = []
    for i in ITEMS:
        if i.status != "not built" and not i.where:
            bad.append(f"{i.item}: no location")
        for w in i.where:
            try:
                resolve(w)
            except Exception as error:                                       # noqa: BLE001 - report every failure
                bad.append(f"{i.item}: {w}: {error}")
    return bad


INTRO = """# The five-book roadmap: what is built, what already existed, what is not

This page is the audit of the roadmap for Quant Lab: López de Prado's *Advances in Financial Machine Learning*, Grinold and Kahn's *Active Portfolio Management* and *Advances in Active Portfolio Management*, Cajas's *Advanced Portfolio Optimization*, and Paleologo's *Advanced Portfolio Management*. It is written from `src/alphalab/catalog.py`, and a test imports every location named here, so it cannot describe something that is not there.

**Status.** *built*: written for this item. *existing*: the repository already had it, and the entry points to it. *partial*: some of it exists, and the note says what is missing. *not built*: nothing exists, and the note says why. **Priority** is the roadmap's proposed development order, not a statement of what was absent. Chapter numbers are as the roadmap gave them.

**Where to run things.** The *Alpha & Portfolio Lab* tab runs four studies (alpha, risk, portfolio, trading) on a basket of tickers; the Python modules are `src.alphalab`, `src.riskengine` and `src.portfolio`; see [the lab guide](alphalab.md).
"""


def markdown() -> str:
    lines = [INTRO]
    for area in dict.fromkeys(i.area for i in ITEMS):
        lines += [f"## {area}", "", "| Priority | Item | Source | Status | Where | Notes |", "|---|---|---|---|---|---|"]
        for i in (x for x in ITEMS if x.area == area):
            places = ", ".join(f"`{w.split(':', 1)[1]}`" + ("" if w.startswith("py:") else f" ({w.split(':', 1)[0]})") for w in i.where) or "none"
            lines.append(f"| {i.priority} | {i.item} | {i.source} | {i.status} | {places} | {i.note} |")
        lines.append("")
    counts = {s: sum(1 for i in ITEMS if i.status == s) for s in ("built", "existing", "partial", "not built")}
    lines += ["## Summary", "", f"{len(ITEMS)} items: " + ", ".join(f"{n} {s}" for s, n in counts.items()) + ".", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parents[2] / "docs" / "roadmap_audit.md"
    out.write_text(markdown(), encoding="utf-8")
    print(f"wrote {out}")
