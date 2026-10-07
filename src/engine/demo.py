"""The mixed-asset example: one engine, one ledger and one strategy interface running equity-index and commodity futures, currencies, crypto spot and perpetuals, listed options and
interest-rate swaps in a single portfolio.

:func:`run_mixed_asset_demo` builds the synthetic multi-asset market (``synthetic``), a regime-aware ENSEMBLE strategy (trend, carry, a variance-premium options sleeve, a crypto basis
sleeve and a commodity calendar spread, blended by a volatility-regime detector with a risk overlay, volatility target and drawdown brake) and a swap curve strategy, runs them through
one :class:`~src.engine.engine.Engine` with first-class costs, financing, margin, constraints and a portfolio risk model, and returns a :class:`MixedAssetDemo` with the result and the
tables the success criteria ask for: asset-class P&L attribution, the cost summary, futures roll costs, funding accrual, the reconciliation report, a risk report with scenarios and the
replay digest (running the demo twice must give the same digest).

The data are synthetic with a known generating process, so the numbers show that the machinery works end to end, not that the strategies have an edge in real markets.
"""

from __future__ import annotations

from dataclasses import dataclass


from .analysis import md_table
from .constraints import ConstraintSet
from .costs import CommissionModel, CostSchedule, FinancingModel, ImpactModel, LiquidityModel, SlippageModel
from .engine import Engine, EngineConfig
from .risk import PortfolioRisk
from .strategies import (BasisStrategy, CarryStrategy, EnsembleStrategy, RelativeValueStrategy, TrendStrategy, VolatilityPremiumStrategy, VolRegime)
from .synthetic import SyntheticMarket, synthetic_multi_asset_market


@dataclass
class MixedAssetDemo:
    market: SyntheticMarket
    engine: Engine
    result: object
    ensemble: EnsembleStrategy

    def report(self) -> str:
        r, s = self.result, self.result.summary(ex_interest=True)
        pm = r.pnl_matrix("asset_class", "category").round(0)
        lines = ["# Mixed-asset demo", "", f"Period {r.config['start']} to {r.config['end']}; digest `{r.digest[:16]}`; reconciled: {r.reconciliation.ok}.", "",
                 f"Final equity {s['final_equity']:,.0f} (start {r.start_equity:,.0f}); strategy Sharpe ex cash interest {s.get('sharpe', float('nan')):.2f}; max drawdown {s.get('max_drawdown', float('nan')):.1%}.", "",
                 "## P&L by asset class and category", "", md_table(pm, 0), "", "## Costs", "", md_table(r.cost_summary().to_frame("amount"), 0), "",
                 "## Diagnostics", ""] + [f"- {k}: {v}" for k, v in sorted(r.diagnostics.items())]
        if r.last_risk is not None:
            lines += ["", "## Risk at the end", "", md_table(r.last_risk.summary().to_frame("value")), "", "## Scenarios", "", md_table(r.last_risk.scenarios.to_frame("pnl"), 0)]
        return "\n".join(lines)


def default_costs() -> CostSchedule:
    return (CostSchedule()
            .set("future", commission=CommissionModel(per_contract=1.5), impact=ImpactModel("sqrt", 0.3, default_adv=200_000, default_sigma=0.01))
            .set("fx", commission=CommissionModel(bps=0.2))
            .set("crypto", commission=CommissionModel(bps=4.0), slippage=SlippageModel(1.0))
            .set("equity", commission=CommissionModel(per_order=1.0, bps=0.5))
            .set("option", commission=CommissionModel(per_contract=0.65))
            .set("swap", commission=CommissionModel(bps=0.05)))


def build_ensemble(market: SyntheticMarket, capital_share: float = 0.7) -> EnsembleStrategy:
    ids = market.ids
    spot, perp = ids["crypto_spot"], ids["crypto_perp"]
    markets = ["ES", "CL", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"] + spot
    members = {"trend": TrendStrategy(markets, name="trend"),
               "carry": CarryStrategy(futures_chains=["ES", "CL"], fx_pairs=["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"], perps=perp, name="carry"),
               "vrp": VolatilityPremiumStrategy("SPY", mode="vrp", vega_budget=6000.0, name="vrp"),
               "basis": BasisStrategy(spot[0], perp[0], weight=0.3, entry=0.03, name="basis"),
               "calendar": RelativeValueStrategy(kind="calendar", chain_id="CL", weight=0.25, name="calendar")}
    base = {"trend": 0.30, "carry": 0.25, "vrp": 0.15, "basis": 0.15, "calendar": 0.15}
    mult = {"stress": {"trend": 1.3, "carry": 0.5, "vrp": 0.0, "basis": 0.8, "calendar": 0.8}}
    ens = EnsembleStrategy(members, VolRegime("ES"), mult, base, rebalance_days={"vrp": 1, "basis": 1, "calendar": 1, "trend": 5, "carry": 5}, target_vol=0.10, name="ensemble")
    ens.capital_share = capital_share
    return ens


def run_mixed_asset_demo(n_days: int = 300, seed: int = 0, with_swaps: bool = True, initial_cash: float = 10_000_000.0, risk: bool = True) -> MixedAssetDemo:
    market = synthetic_multi_asset_market(n_days=n_days, seed=seed, with_swaps=with_swaps)
    ens = build_ensemble(market, 0.7 if with_swaps else 1.0)
    strategies: list = [ens]
    if with_swaps:
        from ..swaps.strategies import CurveTradeStrategy

        sw = CurveTradeStrategy(short_tenor=2.0, long_tenor=10.0, dv01_budget=3000.0, z_in=1.0)
        sw.capital_share = 0.3
        strategies.append(sw)
    cfg = EngineConfig(**market.config_kwargs(initial_cash={"USD": initial_cash}, min_trade_fraction=0.003, risk_every_snapshot=False, session_days=market.dates))
    constraints = ConstraintSet(max_gross=4.0, max_instrument_weight=1.5, asset_class_caps={"crypto": 1.0}, max_margin_utilization=0.8)
    fin = FinancingModel(deposit_rates={"USD": 0.045, "EUR": 0.03, "GBP": 0.047, "AUD": 0.04, "JPY": 0.002}, default_short_borrow=0.003)
    engine = Engine(market.registry, market.events, strategies, cfg, default_costs(), fin, liquidity=LiquidityModel(), constraints=constraints,
                    risk_model=PortfolioRisk(lookback=120) if risk else None)
    result = engine.run()
    if risk and engine.risk_model is not None and result.last_risk is None:
        result.last_risk = engine.risk_model.report(engine)
    return MixedAssetDemo(market, engine, result, ens)
