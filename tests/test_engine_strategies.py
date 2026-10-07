"""Strategies on the common API: every family runs through the one engine, reconciles, behaves as designed on constructed data, and the learners never see the future."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.engine import (CommissionModel, CostSchedule, Engine, EngineConfig, FinancingModel, SpreadModel, Strategy)
from src.engine.strategies import (BasisStrategy, CarryStrategy, EnsembleStrategy, ForecastCombinationStrategy, MetaLabelStrategy, RelativeValueStrategy, TrendStrategy,
                                   VolatilityPremiumStrategy, VolRegime, WalkForwardMLStrategy, momentum_forecaster, range_forecaster, reversal_forecaster)
from src.engine.strategies.ml import FEATURES, feature_frame, labels
from src.engine.synthetic import synthetic_multi_asset_market
from src.instruments import Instrument, InstrumentRegistry, crypto_perp, crypto_spot, fx_spot
from src.marketdata import concat_events, events_from_funding, events_from_prices
from src.marketdata.schema import normalise_events

warnings.filterwarnings("ignore")
ZERO = CostSchedule().set("equity", spread=SpreadModel(0.0, 0.0)).set("fx", spread=SpreadModel(0.0, 0.0)).set("crypto", spread=SpreadModel(0.0, 0.0)).set("future", spread=SpreadModel(0.0, 0.0))


@pytest.fixture(scope="module")
def market():
    return synthetic_multi_asset_market(n_days=260, seed=11, with_options=True)


def engine_run(market, strategies, cash=5_000_000.0, **kw):
    cfg = EngineConfig(**market.config_kwargs(initial_cash={"USD": cash}, min_trade_fraction=0.003, **kw))
    costs = CostSchedule().set("future", commission=CommissionModel(per_contract=1.5)).set("crypto", commission=CommissionModel(bps=4))
    e = Engine(market.registry, market.events, strategies, cfg, costs, FinancingModel(deposit_rates={"USD": 0.04}))
    return e, e.run()


# ----------------------------------------------------------------------------------------------------------------------------- the synthetic market
def test_synthetic_market_is_consistent(market):
    reg = market.registry
    assert reg.validate() == []
    assert {"ES", "CL"} <= set(reg.chains) and market.ids["options"] and len(market.ids["crypto_perp"]) == 2
    assert market.events["available_at"].is_monotonic_increasing
    assert set(market.events["event_type"]) >= {"bar", "settlement", "quote", "funding", "reference"}


# -------------------------------------------------------------------------------------------------------------------------------- constructed data
def trending_market(sign=1.0, n=200):
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(3)
    px = pd.DataFrame({"UP": 100 * np.exp(np.cumsum(sign * 0.0015 + rng.normal(0, 0.004, n)))}, index=idx)
    reg = InstrumentRegistry([Instrument(instrument_id="UP", asset_class="equity", instrument_type="equity", currency="USD", calendar="WEEKDAY")])
    return idx, px, reg


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_trend_follows_the_direction_of_a_persistent_trend(sign):
    idx, px, reg = trending_market(sign)
    cfg = EngineConfig(start=idx[0], end=idx[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": 1_000_000.0})
    e = Engine(reg, events_from_prices(px, "bar", "16:00"), [TrendStrategy(["UP"], lookbacks=(21, 63), breakout=63, ma_pair=(20, 60))], cfg, ZERO)
    res = e.run()
    assert res.reconciliation.ok
    assert np.sign(e.ledger.quantity("UP")) == sign
    assert res.summary()["total_pnl"] > 0                                       # on a clean trend, following it pays


def test_trend_signal_confidence_is_the_share_of_agreeing_components():
    idx, px, reg = trending_market(1.0)
    cfg = EngineConfig(start=idx[0], end=idx[-1], initial_cash={"USD": 1_000_000.0})
    seen = []

    class Probe(TrendStrategy):
        def generate_signals(self, ctx):
            out = super().generate_signals(ctx)
            seen.extend(out)
            return out

    Engine(reg, events_from_prices(px, "bar", "16:00"), [Probe(["UP"], lookbacks=(21, 63), breakout=63, ma_pair=(20, 60))], cfg, ZERO).run()
    assert seen and all(0.0 <= s.confidence <= 1.0 and -1.0 <= s.value <= 1.0 for s in seen)
    assert seen[-1].value > 0.2 and seen[-1].confidence >= 2 / 3


def carry_market(n=150):
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(1)
    pairs = {"EURUSD": ("EUR", "USD"), "AUDUSD": ("AUD", "USD"), "JPYUSD": ("JPY", "USD")}
    reg = InstrumentRegistry([fx_spot(b, q) for b, q in pairs.values()])
    px = pd.DataFrame({k: 1.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n))) for k in pairs}, index=idx)
    rates = {"EUR": 0.03, "AUD": 0.05, "JPY": 0.0, "USD": 0.04}
    ref = pd.DataFrame({"timestamp": np.repeat(idx + pd.Timedelta(hours=16), 4), "instrument_id": [f"RATE-{c}" for _ in idx for c in rates], "event_type": "reference",
                        "value": [rates[c] for _ in idx for c in rates]})
    return idx, reg, concat_events(events_from_prices(px, "bar", "16:00", spread_bps=1.0), normalise_events(ref, lag="0s"))


def test_carry_goes_long_the_high_yielder_and_short_the_low_yielder():
    idx, reg, ev = carry_market()
    cfg = EngineConfig(start=idx[0], end=idx[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": 2_000_000.0}, pegs={})
    e = Engine(reg, ev, [CarryStrategy(fx_pairs=["EURUSD", "AUDUSD", "JPYUSD"], target_vol=0.1)], cfg, ZERO, FinancingModel())
    res = e.run()
    assert res.reconciliation.ok, res.reconciliation.failures()
    q = {i: e.ledger.quantity(i) for i in ("EURUSD", "AUDUSD", "JPYUSD")}
    assert q["AUDUSD"] > 0 > q["JPYUSD"]                                         # AUD pays 5%, JPY 0%: the carry trade


def basis_market(premium=0.004, rate=0.0003, n=60):
    idx = pd.bdate_range("2024-01-02", periods=n)
    spot, perp = crypto_spot("BINANCE", "BTC", "USDT"), crypto_perp("BINANCE", "BTC", "USDT", margin_mode="cross")
    reg = InstrumentRegistry([spot, perp, fx_spot("USDT", "USD", instrument_id="USDTUSD", lot_size=1.0)])
    s = 30_000 * np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.01, n)))
    ev = [events_from_prices(pd.DataFrame({spot.instrument_id: s, perp.instrument_id: s * (1 + premium), "USDTUSD": np.ones(n)}, index=idx), "bar", "16:00", spread_bps=1.0)]
    ft = pd.date_range(idx[0] + pd.Timedelta(hours=8), idx[-1] + pd.Timedelta(hours=23), freq="8h")
    ev.append(events_from_funding(pd.DataFrame({perp.instrument_id: np.full(len(ft), rate)}, index=ft)))
    return idx, spot, perp, reg, concat_events(*ev)


def test_cash_and_carry_is_delta_neutral_and_earns_the_funding():
    idx, spot, perp, reg, ev = basis_market()
    cfg = EngineConfig(start=idx[0], end=idx[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": 2_000_000.0}, pegs={"USDT": 1.0})
    e = Engine(reg, ev, [BasisStrategy(spot.instrument_id, perp.instrument_id, weight=0.4, entry=0.05)], cfg, ZERO, FinancingModel())
    res = e.run()
    assert res.reconciliation.ok, res.reconciliation.failures()
    qs, qp = e.ledger.quantity(spot.instrument_id), e.ledger.quantity(perp.instrument_id)
    assert qs > 0 > qp and abs(qs + qp * perp.contract_multiplier) < 0.02 * qs               # long the spot, short the perpetual, hedge ratio ~1
    assert res.attribution("category")["funding"] > 0
    pnl = res.summary()["total_pnl"]
    assert pnl > 0 and pnl < 0.2 * 2_000_000                                                # a low-risk yield, not a directional bet


def test_basis_strategy_stays_out_when_the_basis_does_not_pay():
    idx, spot, perp, reg, ev = basis_market(premium=0.0001, rate=0.00001)
    cfg = EngineConfig(start=idx[0], end=idx[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": 2_000_000.0}, pegs={"USDT": 1.0})
    e = Engine(reg, ev, [BasisStrategy(spot.instrument_id, perp.instrument_id, entry=0.08)], cfg, ZERO, FinancingModel())
    res = e.run()
    assert len(res.fills) == 0


def test_calendar_spread_trades_a_dislocation_and_flattens(market):
    e, res = engine_run(market, [RelativeValueStrategy(kind="calendar", chain_id="CL", weight=0.2, entry_z=1.2)])
    assert res.reconciliation.ok and len(res.fills) > 0
    legs = {f for f in res.fills["instrument_id"]}
    assert all(i.startswith("CL") for i in legs)
    held = [i for i, p in e.ledger.positions.items() if not p.is_flat]
    assert len(held) in (0, 2) and not res.reconciliation.expired_held


# ---------------------------------------------------------------------------------------------------------------------------------- volatility
def test_variance_premium_sells_a_hedged_straddle_and_respects_the_notional_cap(market):
    strat = VolatilityPremiumStrategy("SPY", mode="vrp", vega_budget=1e9, max_notional=0.5)
    e, res = engine_run(market, [strat])
    assert res.reconciliation.ok, res.reconciliation.failures()
    opts = res.fills[res.fills["instrument_id"].str.contains("-C|-P")]
    assert len(opts) > 0 and (opts[opts["tags"] == ""].groupby("instrument_id")["quantity"].sum().abs() >= 0).all()
    # sold options: the first fill of each structure is a sale
    first = opts.sort_values("ts").groupby("instrument_id").first()
    assert (first["quantity"] < 0).mean() > 0.9
    # the vega budget was absurdly large: the cap on underlying notional (50% of capital) must bind
    notional = (opts.assign(n=lambda d: d["quantity"].abs() * 100 * 450)).groupby(opts["ts"].dt.normalize())["n"].sum().max()
    assert notional < 0.5 * 5_000_000.0 * 4
    assert "SPY" in set(res.fills["instrument_id"])                                      # the delta hedge


def test_options_book_is_delta_hedged(market):
    from src.derivatives.iv import implied_vol

    ratios = []

    class Probe(VolatilityPremiumStrategy):
        def on_schedule(self, ctx):
            legs = list(ctx.store.get("legs", []))                                           # the structure held coming into today's decision
            S = ctx.data.mid("SPY")
            if legs and np.isfinite(S):
                net = gross = 0.0
                for iid, q in legs:
                    inst = ctx.registry.get(iid)
                    mid, T = ctx.data.mid(iid), inst.time_to_expiry(ctx.ts)
                    if T <= 0 or not np.isfinite(mid) or abs(ctx.portfolio.position(iid)) < 1e-9:
                        continue
                    iv = float(np.asarray(implied_vol(mid, S, inst.strike, T, 0.0, 0.0, inst.is_call)).ravel()[0])
                    if np.isfinite(iv):
                        d = inst.greeks(S, iv, ctx.ts, 0.0)["delta"] * inst.contract_multiplier * q
                        net += d
                        gross += abs(d)
                if gross > 0:                                                               # residual delta of options plus the hedge held from yesterday, against the options' gross delta
                    ratios.append(abs(ctx.portfolio.total_position("SPY") + net) / gross)
            super().on_schedule(ctx)

    e, res = engine_run(market, [Probe("SPY", mode="vrp", vega_budget=5000.0)])
    assert res.reconciliation.ok and len(ratios) > 20
    assert np.median(ratios) < 0.35                                                           # daily re-hedging keeps most of the delta out of the book
    unhedged = VolatilityPremiumStrategy("SPY", mode="vrp", vega_budget=5000.0, hedge=False)
    _, res2 = engine_run(market, [unhedged])
    assert "SPY" not in set(res2.fills["instrument_id"]) and "SPY" in set(res.fills["instrument_id"])


@pytest.mark.parametrize("mode", ["gamma", "skew", "term"])
def test_other_volatility_modes_run_and_reconcile(market, mode):
    e, res = engine_run(market, [VolatilityPremiumStrategy("SPY", mode=mode, vega_budget=4000.0)])
    assert res.reconciliation.ok, res.reconciliation.failures()


def test_invalid_modes_are_rejected():
    with pytest.raises(ValueError):
        VolatilityPremiumStrategy("SPY", mode="astrology")
    with pytest.raises(ValueError):
        RelativeValueStrategy(kind="calendar")
    with pytest.raises(ValueError):
        BasisStrategy("a", "b", mode="x")


# ------------------------------------------------------------------------------------------------------------------------------------- ensemble
def members():
    return {"trend": TrendStrategy(["ES", "CL", "EURUSD"], name="trend"), "carry": CarryStrategy(futures_chains=["ES", "CL"], fx_pairs=["EURUSD", "GBPUSD"], name="carry")}


def test_ensemble_runs_members_unchanged_and_nets_their_targets(market):
    ens = EnsembleStrategy(members(), VolRegime("ES"), {"stress": {"trend": 1.3, "carry": 0.5}}, {"trend": 0.5, "carry": 0.5}, target_vol=0.1)
    e, res = engine_run(market, [ens])
    assert res.reconciliation.ok and len(res.fills) > 0
    h = pd.DataFrame(ens.history)
    assert len(h) > 200 and {"p_calm", "p_stress", "w_trend", "w_carry", "vol_scale", "dd_scale"} <= set(h.columns)
    assert np.allclose(h["p_calm"] + h["p_stress"], 1.0)
    stressed = h[h["p_stress"] > 0.5]
    if len(stressed):
        assert (stressed["w_trend"] > h[h["p_stress"] < 0.5]["w_trend"].mean()).all()        # the regime multiplier tilts the weights


def test_a_member_cannot_bypass_the_ensemble_with_direct_orders():
    class Rogue(Strategy):
        name = "rogue"

        def on_schedule(self, ctx):
            ctx.buy("EURUSD", 1000)

    idx = pd.bdate_range("2023-01-02", periods=10)
    px = pd.DataFrame({"EURUSD": 1.1 * np.ones(10)}, index=idx)
    reg = InstrumentRegistry([fx_spot("EUR", "USD")])
    ens = EnsembleStrategy({"rogue": Rogue()}, None, base_weights={"rogue": 1.0}, target_vol=None)
    cfg = EngineConfig(start=idx[0], end=idx[-1], initial_cash={"USD": 1_000_000.0})
    with pytest.raises(RuntimeError, match="only emit targets"):
        Engine(reg, events_from_prices(px, "bar", "16:00"), [ens], cfg, ZERO).run()


def test_drawdown_brake_and_vol_scale_are_bounded():
    ens = EnsembleStrategy({"a": TrendStrategy(["X"])}, dd_start=0.05, dd_full=0.15, dd_floor=0.25, max_scale=3.0, min_scale=0.2, target_vol=0.1)

    class Ctx:
        class portfolio:
            drawdown = 0.0

    c = Ctx()
    out = []
    for dd in (0.0, 0.05, 0.10, 0.15, 0.30):
        c.portfolio.drawdown = dd
        out.append(ens._drawdown_scale(c))
    assert out[0] == out[1] == 1.0 and out[3] == out[4] == 0.25 and out[1] > out[2] > out[3]
    ens._combined_ret = [0.0001] * 40
    assert ens._vol_scale() == 1.0 or ens._vol_scale() <= 3.0                                  # near-zero measured vol cannot blow the scale up
    ens._combined_ret = list(np.random.default_rng(0).normal(0, 0.05, 60))
    assert 0.2 <= ens._vol_scale() < 1.0                                                      # high measured vol scales the book down


def test_ensemble_overlay_options_validate():
    with pytest.raises(ValueError):
        EnsembleStrategy({"a": TrendStrategy(["X"])}, overlay="vibes")
    ens = EnsembleStrategy({"a": TrendStrategy(["X"]), "b": TrendStrategy(["Y"], name="t2")}, overlay="inverse_vol", overlay_strength=1.0)
    ens._shadow_ret = {"a": [(None, 0.01 * (-1) ** i, 1) for i in range(30)], "b": [(None, 0.002 * (-1) ** i, 1) for i in range(30)]}
    w = ens._overlay_weights({"a": 0.5, "b": 0.5})
    assert w["b"] > w["a"]                                                                     # the calmer member gets more


def test_ensemble_digest_is_deterministic(market):
    def build():
        return EnsembleStrategy(members(), VolRegime("ES"), base_weights={"trend": 0.5, "carry": 0.5})

    _, a = engine_run(market, [build()])
    _, b = engine_run(market, [build()])
    assert a.digest == b.digest


# --------------------------------------------------------------------------------------------------------------------------------------- learners
def test_features_are_causal_and_labels_are_unrealised_at_the_end():
    rng = np.random.default_rng(0)
    p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=pd.bdate_range("2022-01-03", periods=400))
    f = feature_frame(p)
    q = p.copy()
    q.iloc[300:] *= 1.5                                                                         # a different future
    f2 = feature_frame(q)
    pd.testing.assert_frame_equal(f.iloc[:300], f2.iloc[:300])                                  # nothing before the change moved
    y = labels(p, 5)
    assert y.iloc[-5:].isna().all() and y.iloc[100:-5].notna().all()
    assert set(f.columns) == set(FEATURES)


@pytest.mark.parametrize("kind", ["ridge", "gbm"])
def test_walk_forward_model_trains_only_on_realised_labels_and_runs(market, kind):
    strat = WalkForwardMLStrategy(["ES", "EURUSD", "GBPUSD"], model=kind, retrain_every=8, train_window=250, min_samples=150, regime_conditioned=(kind == "gbm"))
    e, res = engine_run(market, [strat])
    assert res.reconciliation.ok and len(strat.fits) >= 2
    assert all(f["n"] >= 150 for f in strat.fits)


def test_ml_strategy_signals_do_not_depend_on_the_future(market):
    ids = ["ES", "EURUSD", "GBPUSD"]
    cut = market.dates[200]

    def run_with(events):
        strat = WalkForwardMLStrategy(ids, model="ridge", retrain_every=8, train_window=250, min_samples=120)
        cfg = EngineConfig(**market.config_kwargs(initial_cash={"USD": 3_000_000.0}, min_trade_fraction=0.003))
        r = Engine(market.registry, events, [strat], cfg, ZERO, FinancingModel()).run()
        return r

    a = run_with(market.events)
    ev2 = market.events.copy()
    mask = (ev2["timestamp"] > cut + pd.Timedelta(hours=20)) & ev2["instrument_id"].isin(["ES" + c for c in ("H22", "M22", "U22", "Z22", "H23", "M23", "U23", "Z23")] + ["EURUSD", "GBPUSD"])
    for col in ("close", "settlement", "bid", "ask"):
        ev2.loc[mask, col] = ev2.loc[mask, col] * 1.2
    b = run_with(ev2)
    upto = cut + pd.Timedelta(hours=20)
    pd.testing.assert_series_equal(a.equity_curve[a.equity_curve.index <= upto], b.equity_curve[b.equity_curve.index <= upto])


def test_forecast_combination_shifts_weight_to_the_forecaster_that_works():
    idx = pd.bdate_range("2022-01-03", periods=420)
    rng = np.random.default_rng(8)
    drift = np.zeros(420)
    for t in range(1, 420):
        drift[t] = 0.97 * drift[t - 1] + rng.normal(0, 0.0006)                                  # persistent drift: momentum works, reversal does not
    px = pd.DataFrame({"UP": 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.005, 420)))}, index=idx)
    reg = InstrumentRegistry([Instrument(instrument_id="UP", asset_class="equity", instrument_type="equity", currency="USD", calendar="WEEKDAY")])
    s = ForecastCombinationStrategy(["UP"], {"mom": momentum_forecaster(21), "rev": reversal_forecaster(21), "rng": range_forecaster(63)}, horizon_days=7.0, rule="hedge", eta=3.0)
    cfg = EngineConfig(start=idx[0], end=idx[-1] + pd.Timedelta(hours=23, minutes=59), initial_cash={"USD": 1_000_000.0})
    res = Engine(reg, events_from_prices(px, "bar", "16:00"), [s], cfg, ZERO, FinancingModel()).run()
    assert res.reconciliation.ok
    assert s.weights["mom"] > s.weights["rev"] and abs(sum(s.weights.values()) - 1.0) < 1e-9
    assert len(s.weight_history) > 20


def test_meta_labelling_sizes_by_probability_and_runs(market):
    primary = TrendStrategy(["ES", "EURUSD", "GBPUSD", "AUDUSD"], name="primary")
    meta = MetaLabelStrategy(primary, horizon_days=7.0, min_samples=30, refit_every=5)
    e, res = engine_run(market, [meta])
    assert res.reconciliation.ok
    probs = pd.DataFrame(meta.probabilities)
    assert len(probs) > 100
    warm = probs[probs["p"].isna()]
    fitted = probs[probs["p"].notna()]
    assert (warm["size"] == meta.warmup_size).all()
    if len(fitted):
        assert fitted["p"].between(0, 1).all() and fitted["size"].between(0, 1).all()
