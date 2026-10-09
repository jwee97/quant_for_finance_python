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


# ---------------------------------------------------------------------------------------------------------------------------------- reports for the command line and the dashboard
def _listed(a) -> list:
    return np.asarray(a, float).tolist()


def basket_report(size: int = 8, seed: int = 0, risk_aversion: float = 1e-3, block_threshold: float = 0.005, share: float = 0.5) -> dict:
    """A demonstration basket (random sizes, volumes and volatilities, a one-factor correlation, about half sells) worked three ways: the joint schedule against stock by stock, the minimum trading
    risk quantity for ``share`` of its value, the maximum trading opportunity when only the buys are on offer, and the program-block split. All numbers are JSON-friendly."""
    from . import basket as bk

    if not 2 <= size <= 30 or risk_aversion <= 0 or not 0 <= share <= 1 or block_threshold <= 0:
        raise ValueError("2 <= size <= 30, risk_aversion > 0, 0 <= share <= 1, block_threshold > 0")
    b = bk.random_basket(size, seed=seed)
    Q = {"joint": bk.basket_schedule(b, risk_aversion, exact=True), "independent": bk.independent_schedule(b, risk_aversion, exact=True)}
    result = {k: bk.basket_cost_risk(b, q) for k, q in Q.items()}
    for r in result.values():
        r["objective"] = r["cost_bps"] + risk_aversion * r["risk_bps"] ** 2
    cov = b.covariance()

    def risk_left(q):
        left = (b.side * b.prices)[:, None] * (b.quantity[:, None] - np.cumsum(q, axis=1))
        return [1e4 * float(np.sqrt(max(left[:, i] @ cov @ left[:, i], 0.0))) / b.gross for i in range(b.n)]

    m = bk.minimum_trading_risk_quantity(b, share=share)
    t = bk.maximum_trading_opportunity(b, available=b.quantity * (b.side > 0))
    pb = bk.program_block(b, block_threshold=block_threshold)
    return {"size": b.size, "seed": seed, "risk_aversion": risk_aversion, "gross": b.gross, "net_exposure": float(b.exposure.sum()), "names": list(b.names), "value": _listed(b.value), "side": _listed(b.side),
            "joint": result["joint"], "independent": result["independent"], "intervals": b.n,
            "executed": {k: _listed((b.prices[:, None] * q).sum(axis=0) / b.gross) for k, q in Q.items()}, "risk_left_bps": {k: risk_left(q) for k, q in Q.items()},
            "mtrq": {"share": m["share"], "executed_fraction": _listed(m["executed_fraction"]), "residual_risk": m["residual_risk"], "naive_risk": m["naive_risk"], "original_risk": m["original_risk"]},
            "mto": {"share_of_list": t["share_of_list"], "executed_fraction": _listed(t["executed_fraction"]), "available_value": t["available_value"], "residual_risk": t["residual_risk"],
                    "original_risk": t["original_risk"], "binding": t["binding"], "feasible": t["feasible"]},
            "program_block": {k: pb[k] for k in ("block", "program", "dark", "lit", "dark_share", "original_risk", "worst_case_risk", "exact")}}


HFT_SIMS = ("pairs", "etf", "rebate", "amm")
HFT_CAVEAT = " A research simulator on a stylised market: it sizes an edge against costs and delay, it does not show that one exists."


def hft_report(sim: str = "etf", seed: int = 0) -> dict:
    """One of the high-frequency research simulations as a table: ``rows`` (a label and the figures), the name of the figure to chart, and a plain statement of what it shows."""
    from . import blackbox as bb, hft

    if sim not in HFT_SIMS:
        raise ValueError(f"sim must be one of {HFT_SIMS}")
    if sim == "pairs":
        r = bb.simulate_pair_trading(seed=seed)
        keep = ("days", "mean_daily_bps", "std_daily_bps", "sharpe", "round_trips", "win_rate", "average_hold_bars", "average_trip_bps", "gross_bps", "cost_bps_total")
        cumulative = np.cumsum(np.asarray(r["daily_bps"], float))
        return {"sim": sim, "title": "Pair trading on a mean-reverting spread", "columns": list(keep), "rows": [{"label": "pair trading", **{k: float(r[k]) for k in keep}}], "chart": "mean_daily_bps",
                "series": {"name": "Cumulative profit, bps", "x": list(range(1, len(cumulative) + 1)), "y": cumulative.tolist(), "x_label": "day"},
                "note": "A z-score rule on the spread of two prices, with the spread an Ornstein-Uhlenbeck process, orders arriving after a delay and a cost on both legs. The gap between gross and cost is the edge." + HFT_CAVEAT}
    if sim == "etf":
        table, title = bb.latency_table(seed=seed), "ETF against its basket, by the delay of the order"
        note = "An ETF whose premium to its net asset value decays with a half-life of a few bars: profit per day by how late the order lands (bars of delay)."
    elif sim == "rebate":
        table, title = hft.pressure_table(seed=seed), "Rebate and liquidity trading from order-flow pressure"
        note = "A maker living on rebates and spread. The naive one quotes both sides always and pays adverse selection; the aware one withdraws the side about to be hit, and its edge fades as its view of the flow gets older."
    else:
        table, title = hft.auto_market_making(seed=seed), "Auto market making: inventory-shaded quotes against symmetric quotes"
        note = "The Avellaneda-Stoikov market maker (quotes shaded by inventory) against symmetric quotes with the same average spread: the mean is similar, the risk is not."
    chart = {"etf": "mean_daily_bps", "rebate": "mean_wealth", "amm": "sharpe"}[sim]
    names = {"as": "inventory-shaded (Avellaneda-Stoikov)", "symmetric": "symmetric quotes"}
    rows = [{"label": names.get(str(i), str(i)), **{c: float(v) for c, v in row.items()}} for i, row in table.iterrows()]
    if sim == "etf":
        rows = [{"label": f"{int(r['label'])} bar{'s' if int(r['label']) != 1 else ''} late", **{k: v for k, v in r.items() if k != "label"}} for r in rows]
    return {"sim": sim, "title": title, "columns": [str(c) for c in table.columns], "rows": rows, "chart": chart, "series": None, "note": note + HFT_CAVEAT}
