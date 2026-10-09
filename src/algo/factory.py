"""Building algorithms from short text specifications, and running an order through the simulator from plain parameters: what the command line and the dashboard's execution lab both use.

An algorithm is written ``[tactic+]name[:key=value,key=value]``: ``vwap``, ``is:risk_aversion=0.01``, ``pov:rate=0.15``, ``aim+vwap``, ``target_cost+is:risk_aversion=0.003``. Names are those of
:data:`ALL_ALGORITHMS`; tactics are ``aim``, ``pim`` and ``target_cost``.
"""

from __future__ import annotations

import numpy as np

from .algos import ALGORITHMS
from .liquidity import LiquiditySeeking
from .market import SCENARIOS, Market, Order, Scenario
from .simulate import STYLES, Algo, simulate
from .tactics import AIM, PIM, TargetCost

ALL_ALGORITHMS = {**ALGORITHMS, LiquiditySeeking.name: LiquiditySeeking}
TACTICS = {"aim": AIM, "pim": PIM, "target_cost": TargetCost}


def _value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return {"true": True, "false": False}.get(text.lower(), text)


def build_algo(spec: str) -> Algo:
    """The algorithm a specification names (see the module docstring); ``ValueError`` says what is wrong with it."""
    text = spec.strip()
    tactic = None
    if "+" in text:
        tactic, text = text.split("+", 1)
        if tactic not in TACTICS:
            raise ValueError(f"unknown tactic '{tactic}'; choose from {sorted(TACTICS)}")
    name, _, rest = text.partition(":")
    if name not in ALL_ALGORITHMS:
        raise ValueError(f"unknown algorithm '{name}'; choose from {sorted(ALL_ALGORITHMS)}")
    params = {}
    for part in filter(None, (p.strip() for p in rest.split(","))):
        if "=" not in part:
            raise ValueError(f"'{part}' is not key=value")
        key, value = part.split("=", 1)
        params[key.strip()] = _value(value.strip())
    try:
        algo = ALL_ALGORITHMS[name](**params)
    except TypeError:
        raise ValueError(f"{name} does not accept those settings: {', '.join(params)}") from None
    return TACTICS[tactic](algo) if tactic else algo


def run_order(algos, side: str = "buy", shares: float = 200_000.0, adv: float = 2_000_000.0, sigma: float = 0.02, price: float = 50.0, spread_bps: float = 4.0, intervals: int = 26,
              days: int = 1, max_participation: float = 0.35, style: str = "aggressive", scenario: str = "normal", paths: int = 400, seed: int = 0, target_bps: float | None = None,
              horizon: int | None = None, alpha_bps: float = 0.0) -> dict:
    """Run one or more algorithm specifications on one order and market and return their summaries, mean schedules and the market profile.

    ``alpha_bps`` is the expected price drift the market really has over the order (it overrides the scenario's), which the algorithms do not know unless their own settings say so."""
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy or sell")
    if style not in STYLES:
        raise ValueError(f"unknown style '{style}'; choose from {sorted(STYLES)}")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}'; choose from {sorted(SCENARIOS)}")
    if not 10 <= paths <= 5000:
        raise ValueError("paths must be between 10 and 5000")
    if isinstance(algos, str):
        algos = [algos]
    if not algos or len(algos) > 8:
        raise ValueError("choose between one and eight algorithms")
    market = Market(price=price, adv=adv, sigma=sigma, spread=spread_bps * 1e-4, n=intervals, days=days)
    order = Order(1 if side == "buy" else -1, shares, max_participation=max_participation, horizon=horizon)
    sc = SCENARIOS[scenario]
    if alpha_bps:
        sc = Scenario(sc.name, alpha_bps, sc.persistence, sc.stress, sc.vol_mult, sc.spread_mult, sc.impact_mult, sc.volume_mult, sc.description)
    rows, schedules, labels = [], {}, []
    for spec in algos:
        algo = build_algo(spec)
        result = simulate(order, market, algo, style, sc, paths, seed, target_bps)
        summary = result.summary()
        summary["algo"] = spec
        rows.append(summary)
        schedules[spec] = (result.mean_schedule() / shares).tolist()
        labels.append(spec)
    window = order.window(market)
    return {"rows": rows, "schedules": schedules, "volume_profile": (market.expected_volume()[window] / market.adv).tolist(), "intervals": order.length(market),
            "scenario": {"name": sc.name, "description": sc.description}, "style": {"name": style, "description": STYLES[style].description},
            "order": {"side": side, "shares": shares, "value": shares * price, "participation": shares / (adv * days)}, "labels": labels}


def efficient_frontier(side: str = "buy", shares: float = 200_000.0, adv: float = 2_000_000.0, sigma: float = 0.02, price: float = 50.0, spread_bps: float = 4.0, intervals: int = 26,
                       points: int = 21) -> dict:
    """The cost-risk frontier of an order: for risk aversions from patient to urgent, the expected cost and risk in bps, with the first slice each schedule takes."""
    from . import optimize as opt

    market = Market(price=price, adv=adv, sigma=sigma, spread=spread_bps * 1e-4, n=intervals)
    order = Order(1 if side == "buy" else -1, shares)
    rows = opt.frontier(order, market, np.logspace(-6, 0, points))
    vwap = opt.cost_and_risk(order.shares * market.volume_profile(), order, market)
    twap = opt.cost_and_risk(np.full(intervals, shares / intervals), order, market)
    return {"frontier": [{"risk_aversion": r["risk_aversion"], "cost_bps": r["cost_bps"], "risk_bps": r["risk_bps"], "first_slice": float(r["schedule"][0] / shares)} for r in rows],
            "vwap": {"cost_bps": vwap["cost_bps"], "risk_bps": vwap["risk_bps"]}, "twap": {"cost_bps": twap["cost_bps"], "risk_bps": twap["risk_bps"]}}
