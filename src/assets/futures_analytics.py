"""The futures workspace: the shape of a curve, what rolling costs, what carry and basis say, and how much margin a book needs. Pure functions over a long contract table
(``date, contract, expiry, price`` and optionally ``open_interest``, ``volume``, ``spot``).

**Where the curves come from.** The repository holds no futures contract prices. The default curves are therefore *simulated*: a Schwartz-Smith (2000) two-factor model per root (``assets.futures``),
with an exchange-style expiry calendar, a simulated open-interest profile (it peaks about half a contract cycle before expiry and crosses from one contract to the next ``roll_oi_days`` before expiry)
and contract specifications and margins that are ILLUSTRATIVE round numbers, not current exchange values. A curve you load yourself (``data/user/futures/<ROOT>.csv``) is analysed as it is. Every payload
carries its ``source``; nothing here may be read as a live quote.

Conventions. Prices are in the contract's quote units; ``multiplier`` is the dollar value of one price unit per contract. Carry (roll yield) is annualised: ``ln(F_near / F_far) / (T_far - T_near)``,
positive in backwardation (the curve slopes down; a long earns the roll) and negative in contango. The roll date is the expiry less ``roll_days`` calendar days, as in ``assets.futures.continuous_series``.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import AssetSnapshot, add_flag, last_zscore, liquidity_score, percentile_rank, realized_vol
from .futures import SchwartzSmithParams, basis_momentum, carry, contract_returns, excess_return_series, synthetic_term_structure, term_structure


@dataclass(frozen=True)
class RootSpec:
    """A futures root: its contract terms, expiry calendar, the model that simulates it, and illustrative margins and costs."""

    root: str
    name: str
    exchange: str
    asset_class: str
    multiplier: float
    tick_size: float
    months: tuple
    rule: str
    params: SchwartzSmithParams
    initial_margin: float                  # per contract, ILLUSTRATIVE
    maintenance_margin: float              # per contract, ILLUSTRATIVE
    roll_oi_days: int = 8                  # days before expiry at which the simulated open interest moves to the next contract
    spread_ticks: float = 1.0              # typical cost, in ticks, of trading the calendar spread
    commission: float = 1.5                # per contract per side

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.multiplier


ROOTS: dict[str, RootSpec] = {
    "CL": RootSpec("CL", "WTI crude oil", "NYMEX", "commodity", 1000.0, 0.01, tuple(range(1, 13)), "business_day_before_25th",
                   SchwartzSmithParams(kappa=1.2, sigma_chi=0.30, mu_xi=0.02, sigma_xi=0.22, rho=0.3, lambda_chi=0.08, mu_xi_star=0.01, s0=70.0), 6500.0, 5900.0, 8, 1.0, 1.5),
    "NG": RootSpec("NG", "Henry Hub natural gas", "NYMEX", "commodity", 10000.0, 0.001, tuple(range(1, 13)), "business_day_before_25th",
                   SchwartzSmithParams(kappa=2.0, sigma_chi=0.55, mu_xi=0.0, sigma_xi=0.25, rho=0.2, lambda_chi=-0.10, mu_xi_star=0.0, seasonal_amp=0.10, seasonal_peak_month=1, s0=3.0), 3500.0, 3200.0, 7, 2.0, 1.5),
    "GC": RootSpec("GC", "Gold", "COMEX", "metal", 100.0, 0.10, (2, 4, 6, 8, 10, 12), "last_business_day",
                   SchwartzSmithParams(kappa=0.5, sigma_chi=0.02, mu_xi=0.05, sigma_xi=0.14, rho=0.0, lambda_chi=0.0, mu_xi_star=0.04, s0=2000.0), 9000.0, 8200.0, 12, 1.0, 1.5),
    "HG": RootSpec("HG", "Copper", "COMEX", "commodity", 25000.0, 0.0005, tuple(range(1, 13)), "last_business_day",
                   SchwartzSmithParams(kappa=1.0, sigma_chi=0.12, mu_xi=0.03, sigma_xi=0.20, rho=0.2, lambda_chi=0.03, mu_xi_star=0.015, s0=4.0), 5500.0, 5000.0, 10, 2.0, 1.5),
    "ES": RootSpec("ES", "E-mini S&P 500", "CME", "equity_index", 50.0, 0.25, (3, 6, 9, 12), "third_friday",
                   SchwartzSmithParams(kappa=1.0, sigma_chi=0.03, mu_xi=0.08, sigma_xi=0.16, rho=0.0, lambda_chi=0.0, mu_xi_star=0.02, s0=5000.0), 12500.0, 11400.0, 8, 1.0, 1.25),
}
HISTORY_DAYS = 1000
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_LOCK = threading.Lock()


@dataclass
class CurveData:
    """A root's contract table with its provenance."""

    root: str
    spec: RootSpec
    long: pd.DataFrame                    # date, contract, expiry, price, ttm, open_interest, volume
    spot: pd.Series | None                # the model's spot (simulated curves) or a spot column from a file; None if unknown
    source: str                           # "simulated" | "user_file"
    notes: list[str] = field(default_factory=list)

    @property
    def as_of(self) -> pd.Timestamp:
        return pd.Timestamp(self.long["date"].max())


def _seed(text: str) -> int:
    return int.from_bytes(text.encode()[:4].ljust(4, b"_"), "little") % (2 ** 31)


def simulate_open_interest(long: pd.DataFrame, spacing_days: float, crossover_days: float, base: float = 250_000.0, seed: int = 0) -> pd.DataFrame:
    """SIMULATED open interest and volume for each contract on each date. Open interest peaks ``crossover_days + spacing / 2`` days before a contract expires and falls to nothing as it expires, so two
    neighbouring contracts cross ``crossover_days`` before the nearer one expires; far contracts keep a small floor. Volume is a noisy fraction of open interest."""
    rng = np.random.default_rng(seed)
    days = (pd.to_datetime(long["expiry"]) - pd.to_datetime(long["date"])).dt.days.to_numpy(dtype=float)
    peak, sigma = crossover_days + spacing_days / 2.0, 0.45 * spacing_days
    shape = np.exp(-0.5 * ((days - peak) / sigma) ** 2) + 0.03 * np.exp(-days / 365.0)
    oi = np.round(base * shape * np.exp(0.03 * rng.standard_normal(len(long))))
    out = long.copy()
    out["open_interest"] = oi
    out["volume"] = np.round(oi * 0.18 * np.exp(0.25 * rng.standard_normal(len(long))))
    return out


def simulated_curve(root: str, as_of=None, n_days: int = HISTORY_DAYS) -> CurveData:
    """The simulated history of a root, ending on ``as_of`` (default: the last weekday), cached because building it takes a second."""
    root = root.upper()
    if root not in ROOTS:
        raise ValueError(f"unknown futures root '{root}'; the simulated roots are {sorted(ROOTS)}")
    spec = ROOTS[root]
    end = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    end = end if end.weekday() < 5 else end - pd.offsets.BDay(1)
    key = (root, str(end.date()), int(n_days))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    start = end - pd.offsets.BDay(n_days - 1)
    out = synthetic_term_structure(n_days=n_days, start=str(start.date()), params=spec.params, months=spec.months, rule=spec.rule, seed=_seed(root))
    long = out["long"][["date", "contract", "expiry", "price", "ttm"]].copy()
    expiries = np.sort(pd.to_datetime(long["expiry"]).unique())
    spacing = float(np.median(np.diff(expiries).astype("timedelta64[D]").astype(float))) if len(expiries) > 1 else 30.0
    long = simulate_open_interest(long, spacing, spec.roll_oi_days, seed=_seed(root + "oi"))
    spot = pd.Series(np.exp(out["chi"].to_numpy() + out["xi"].to_numpy()), index=out["chi"].index, name="spot")
    data = CurveData(root, spec, long, spot, "simulated",
                     ["Prices are a Schwartz-Smith model simulation, not market data.", "Open interest and volume are simulated (a profile that moves to the next contract before expiry).",
                      "Contract terms and margins are illustrative round numbers, not current exchange values."])
    with _LOCK:
        _CACHE[key] = data
        while len(_CACHE) > 8:
            _CACHE.popitem(last=False)
    return data


def load_curve_file(path, root: str) -> CurveData:
    """A contract table from a CSV with columns ``date, contract, expiry, price`` and optionally ``open_interest``, ``volume`` and ``spot``. Contract terms come from ``ROOTS`` when the root is known; otherwise
    a generic specification (multiplier 1, no margin) is used and margin figures are not meaningful."""
    raw = pd.read_csv(path)
    missing = [c for c in ("date", "contract", "expiry", "price") if c not in raw.columns]
    if missing:
        raise ValueError(f"the futures file needs the columns date, contract, expiry, price (missing {missing})")
    raw["date"], raw["expiry"] = pd.to_datetime(raw["date"]), pd.to_datetime(raw["expiry"])
    raw = raw.dropna(subset=["price"]).sort_values(["date", "expiry"])
    if raw.duplicated(["date", "contract"]).any():
        raise ValueError("duplicate (date, contract) rows in the futures file")
    raw = raw[raw["expiry"] >= raw["date"]].copy()
    raw["ttm"] = (raw["expiry"] - raw["date"]).dt.days / 365.0
    notes = ["Prices are from your file."]
    for col in ("open_interest", "volume"):
        if col not in raw.columns:
            raw[col] = np.nan
            notes.append(f"The file has no {col.replace('_', ' ')}; anything that needs it is unavailable.")
    spec = ROOTS.get(root.upper()) or RootSpec(root.upper(), root.upper(), "?", "unknown", 1.0, 0.01, tuple(range(1, 13)), "last_business_day", SchwartzSmithParams(), 0.0, 0.0)
    if root.upper() not in ROOTS:
        notes.append("The contract terms of this root are unknown (multiplier 1, no margin): dollar figures are in price units.")
    spot = raw.drop_duplicates("date").set_index("date")["spot"] if "spot" in raw.columns and raw["spot"].notna().any() else None
    return CurveData(root.upper(), spec, raw[["date", "contract", "expiry", "price", "ttm", "open_interest", "volume"]].reset_index(drop=True), spot, "user_file", notes)


# ------------------------------------------------------------------------------------------------------------------ the curve
def _panel(data: CurveData, depth: int = 6) -> dict:
    return term_structure(data.long[["date", "contract", "expiry", "price"]], depth)


def curve_state(roll_yields: np.ndarray, log_slope: float, tol: float = 0.005) -> str:
    """``backwardation`` when the front roll yield is clearly positive and the curve falls overall, ``contango`` when clearly negative and it rises, otherwise ``mixed`` (a humped or kinked curve)."""
    front = float(roll_yields[0]) if len(roll_yields) else 0.0
    if front > tol and log_slope < 0:
        return "backwardation"
    if front < -tol and log_slope > 0:
        return "contango"
    return "mixed"


def curve_snapshot(data: CurveData, depth: int = 12, date=None) -> dict:
    """The curve on one date (default the latest): each contract's expiry, days to expiry, price, open interest, annualised roll yield against the next contract, and the curve's state, slope and curvature."""
    d = pd.Timestamp(date) if date is not None else data.as_of
    g = data.long[pd.to_datetime(data.long["date"]) == d].sort_values("expiry").head(depth)
    if len(g) < 2:
        raise ValueError(f"fewer than two contracts on {d.date()}")
    price, ttm = g["price"].to_numpy(dtype=float), g["ttm"].to_numpy(dtype=float)
    ry = np.log(price[:-1] / price[1:]) / np.maximum(np.diff(ttm), 1e-6)
    ry = np.r_[ry, np.nan]
    k = min(6, len(g))
    slope = float(np.polyfit(ttm[:k], np.log(price[:k]), 1)[0])                  # d ln F / d T over the nearest contracts, per year
    curv = float((price[0] + price[2] - 2.0 * price[1]) / price[1]) if len(g) >= 3 else float("nan")
    state = curve_state(ry[:-1][:k - 1], slope)
    rows = []
    for i, r in enumerate(g.itertuples()):
        rows.append({"rank": i + 1, "contract": str(r.contract), "expiry": str(pd.Timestamp(r.expiry).date()), "dte": int(round(float(r.ttm) * 365)), "price": float(r.price),
                     "change_vs_front": float(r.price / price[0] - 1.0), "roll_yield": None if not np.isfinite(ry[i]) else float(ry[i]),
                     "open_interest": None if not np.isfinite(r.open_interest) else float(r.open_interest), "volume": None if not np.isfinite(r.volume) else float(r.volume)})
    spec = data.spec
    return {"date": str(d.date()), "root": data.root, "name": spec.name, "rows": rows, "state": state, "annualised_carry": float(ry[0]), "log_slope": slope, "curvature": curv,
            "front": rows[0]["contract"], "front_price": float(price[0]), "multiplier": spec.multiplier, "tick_size": spec.tick_size, "tick_value": spec.tick_value,
            "notional": float(price[0] * spec.multiplier)}


def curve_history(data: CurveData, dates: int = 3, spacing: int = 21) -> list[dict]:
    """The curve now and ``spacing`` trading days ago, twice more, for the chart that shows how it moved."""
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(data.long["date"]).unique()))
    out = []
    for i in range(dates):
        pos = len(idx) - 1 - i * spacing
        if pos < 0:
            break
        snap = curve_snapshot(data, 12, idx[pos])
        out.append({"date": snap["date"], "points": [{"dte": r["dte"], "price": r["price"], "contract": r["contract"]} for r in snap["rows"]]})
    return out


# ------------------------------------------------------------------------------------------------------------------ roll
def calendar_roll_date(expiry, roll_days: int) -> pd.Timestamp:
    return pd.Timestamp(expiry) - pd.Timedelta(days=int(roll_days))


def oi_crossover(data: CurveData, front: str, nxt: str) -> pd.Timestamp | None:
    """The date from which the next contract's open interest stayed above the front's through the latest date (``None`` if the front still leads, or open interest is unavailable). Months before the roll
    the later contract may lead for a spell while both are thin; only the crossing that holds to the end counts."""
    long = data.long
    a = long[long["contract"] == front].set_index("date")["open_interest"]
    b = long[long["contract"] == nxt].set_index("date")["open_interest"]
    joint = pd.concat([a, b], axis=1, keys=["front", "next"]).dropna()
    if joint.empty:
        return None
    ahead = (joint["next"] > joint["front"]).to_numpy()
    if not ahead[-1]:
        return None
    stays = np.logical_and.accumulate(ahead[::-1])[::-1]                      # True from the last date at which the front led, onwards
    return pd.Timestamp(joint.index[int(np.argmax(stays))])


def roll_analysis(data: CurveData, roll_days: int = 5, window: int = 120) -> dict:
    """The roll decision for the contract now in front: the contract held and the one rolled into, days to expiry, the calendar roll date (``roll_days`` before expiry), the date open interest moved to the next
    contract (or has not yet), the suggested roll date (the earlier of the two), the cost of the roll (the calendar spread in dollars per contract plus the spread you cross and commission), and the histories
    that draw it."""
    snap = curve_snapshot(data, 3)
    front, nxt = snap["rows"][0], snap["rows"][1]
    spec, as_of = data.spec, data.as_of
    cal = calendar_roll_date(front["expiry"], roll_days)
    cross = oi_crossover(data, front["contract"], nxt["contract"])
    suggested = min([x for x in (cal, cross) if x is not None]) if cross is not None else cal
    spread_points = nxt["price"] - front["price"]
    exec_cost = spec.spread_ticks * spec.tick_value + 2.0 * spec.commission
    long = data.long
    f = long[long["contract"] == front["contract"]].set_index("date")
    n = long[long["contract"] == nxt["contract"]].set_index("date")
    both = pd.concat([f["price"], n["price"], f["open_interest"], n["open_interest"]], axis=1, keys=["front", "next", "oi_front", "oi_next"]).dropna(subset=["front", "next"]).tail(window)
    hist = {"dates": [str(d.date()) for d in both.index], "front": [round(float(v), 4) for v in both["front"]], "next": [round(float(v), 4) for v in both["next"]],
            "spread": [round(float(v), 4) for v in (both["next"] - both["front"])],
            "oi_front": [None if not np.isfinite(v) else float(v) for v in both["oi_front"]], "oi_next": [None if not np.isfinite(v) else float(v) for v in both["oi_next"]]}
    past = []
    contracts = long.drop_duplicates("contract").set_index("contract")["expiry"].sort_values()
    contracts = contracts[contracts < as_of].tail(12)
    for c, e in contracts.items():
        later = long[long["expiry"] > e]["contract"].drop_duplicates()
        nxt_c = long.loc[long["expiry"] > e].sort_values("expiry")["contract"].iloc[0] if len(later) else None
        d = calendar_roll_date(e, roll_days)
        px = long[(long["contract"] == c) & (pd.to_datetime(long["date"]) <= d)].sort_values("date")
        px2 = long[(long["contract"] == nxt_c) & (pd.to_datetime(long["date"]) <= d)].sort_values("date") if nxt_c else px.iloc[0:0]
        if len(px) and len(px2):
            past.append({"contract": c, "next": nxt_c, "expiry": str(pd.Timestamp(e).date()), "roll_date": str(d.date()), "spread": float(px2["price"].iloc[-1] - px["price"].iloc[-1]),
                         "oi_crossover": None if (x := oi_crossover(data, c, nxt_c)) is None else str(x.date())})
    flags: list[str] = []
    add_flag(flags, front["dte"] <= roll_days + 2, f"the {front['contract']} contract expires in {front['dte']} days: roll now")
    add_flag(flags, cross is not None and as_of >= cross, "open interest has already moved to the next contract: the front is thinning")
    add_flag(flags, np.isfinite(spread_points) and spread_points > 0, f"the curve is in contango at the front: rolling a long position costs about {abs(spread_points) * spec.multiplier:,.0f} per contract")
    return {"root": data.root, "as_of": str(as_of.date()), "current_contract": front["contract"], "next_contract": nxt["contract"], "front_expiry": front["expiry"], "days_to_expiry": front["dte"],
            "roll_days": int(roll_days), "calendar_roll_date": str(cal.date()), "oi_crossover": None if cross is None else str(cross.date()),
            "suggested_roll_date": str(pd.Timestamp(suggested).date()), "days_to_roll": int((pd.Timestamp(suggested) - as_of).days),
            "calendar_spread_points": float(spread_points), "calendar_spread_dollars": float(spread_points * spec.multiplier), "execution_cost_dollars": float(exec_cost),
            "roll_cost_dollars": float(spread_points * spec.multiplier + exec_cost), "roll_cost_pct": float(spread_points / front["price"]), "annualised_roll": front["roll_yield"],
            "history": hist, "past_rolls": past, "risk_flags": flags}


# ------------------------------------------------------------------------------------------------------------------ carry and basis
def carry_dashboard(data: CurveData) -> dict:
    """Carry (annualised roll yield of the first two contracts) through time and today: its level, one-year percentile and z-score, the realised volatility of the roll strategy's excess return, and carry
    per unit of volatility (the carry 'Sharpe'). Also the carry earned over the last year against the excess return actually earned."""
    ts = _panel(data, 4)
    cy = carry(ts["price"], ts["ttm"]).dropna()
    ex = excess_return_series(data.long[["date", "contract", "expiry", "price"]], 5)
    vol = float(ex.std(ddof=1) * np.sqrt(252.0)) if len(ex) > 20 else float("nan")
    snap = curve_snapshot(data, 6)
    year = cy.tail(252)
    now = float(cy.iloc[-1])
    earned = float(cy.tail(252).sum() / 252.0)
    realised = float(np.expm1(np.log1p(ex.tail(252)).sum())) if len(ex) else float("nan")
    return {"root": data.root, "as_of": str(cy.index[-1].date()), "state": snap["state"], "carry": now, "carry_ma_21": float(cy.tail(21).mean()), "carry_percentile_1y": percentile_rank(year, now),
            "carry_zscore": last_zscore(cy, 252), "excess_return_vol": vol, "carry_to_vol": now / vol if np.isfinite(vol) and vol > 0 else None,
            "carry_earned_1y": earned, "excess_return_1y": realised, "curvature": snap["curvature"],
            "series": {"dates": [str(d.date()) for d in cy.index[-400:]], "carry": [round(float(v), 5) for v in cy.tail(400)], "carry_ma": [None if not np.isfinite(v) else round(float(v), 5) for v in cy.rolling(21).mean().tail(400)]}}


def basis_dashboard(data: CurveData) -> dict:
    """The basis of the front contract to spot (when the data has a spot series: simulated curves do, a file may), its annualised value and z-score; the calendar basis between the first two contracts; and basis
    momentum (Boons and Prado 2019: the one-year average return of the front contract minus the second)."""
    ts = _panel(data, 3)
    f1, f2 = ts["price"]["F1"], ts["price"]["F2"]
    t1, t2 = ts["ttm"]["F1"], ts["ttm"]["F2"]
    out = {"root": data.root, "as_of": str(f1.index[-1].date()), "has_spot": data.spot is not None, "front_contract": str(ts["contract"]["F1"].iloc[-1])}
    cal = (f2 / f1 - 1.0).dropna()
    out["calendar_basis"] = float(cal.iloc[-1])
    out["calendar_basis_annualised"] = float(np.log(f2 / f1).iloc[-1] / max(float(t2.iloc[-1] - t1.iloc[-1]), 1e-6))
    out["calendar_series"] = {"dates": [str(d.date()) for d in cal.index[-300:]], "values": [round(float(v), 5) for v in cal.tail(300)]}
    if data.spot is not None:
        s = data.spot.reindex(f1.index)
        basis = (f1 / s - 1.0).dropna()
        ann = (np.log(f1 / s) / t1.where(t1 > 7.0 / 365.0)).dropna()                # a week or less to expiry: the annualised basis is noise
        out.update({"spot": float(s.iloc[-1]), "basis": float(basis.iloc[-1]), "basis_points": float(f1.iloc[-1] - s.iloc[-1]), "basis_annualised": float(ann.iloc[-1]) if len(ann) and ann.index[-1] == basis.index[-1] else None,
                    "basis_zscore": last_zscore(basis, 252), "dte": int(round(float(t1.iloc[-1]) * 365)),
                    "series": {"dates": [str(d.date()) for d in basis.index[-300:]], "basis": [round(float(v), 5) for v in basis.tail(300)], "dte": [round(float(v) * 365, 1) for v in t1.reindex(basis.index).tail(300)]}})
    try:
        bm = basis_momentum(contract_returns(ts["price"], ts["contract"])).dropna()
        out["basis_momentum"] = float(bm.iloc[-1]) if len(bm) else None
        out["basis_momentum_percentile"] = percentile_rank(bm.tail(504), float(bm.iloc[-1])) if len(bm) else None
        out["basis_momentum_series"] = {"dates": [str(d.date()) for d in bm.index[-300:]], "values": [round(float(v), 5) for v in bm.tail(300)]}
    except (KeyError, ValueError):
        out["basis_momentum"] = None
    return out


# ------------------------------------------------------------------------------------------------------------------ margin
def margin_dashboard(positions: list[dict], curves: dict[str, CurveData], equity: float, margin_scale: float = 1.0, near_expiry_days: int = 7) -> dict:
    """Margin and leverage for a book of futures. ``positions`` are ``{root, contract (optional: default the front), quantity}``. Notional is ``quantity x price x multiplier``; initial and maintenance margin are
    the roots' ILLUSTRATIVE per-contract figures times ``margin_scale``; leverage is gross notional over ``equity``. Warnings: margin use above 70% of equity, a single root above half of the margin,
    a contract within ``near_expiry_days`` of expiry, equity below maintenance margin."""
    if equity <= 0:
        raise ValueError("equity must be positive")
    if not positions:
        raise ValueError("add at least one position")
    rows, flags = [], []
    for pos in positions:
        root = str(pos.get("root", "")).upper()
        if root not in curves:
            raise ValueError(f"no curve for root '{root}'")
        data, qty = curves[root], float(pos.get("quantity", 0.0))
        snap = curve_snapshot(data, 12)
        contract = str(pos.get("contract") or snap["front"])
        row = next((r for r in snap["rows"] if r["contract"] == contract), None)
        if row is None:
            raise ValueError(f"contract {contract} is not on the {root} curve")
        spec = data.spec
        notional = qty * row["price"] * spec.multiplier
        vol = float(realized_vol(data.long[data.long["contract"] == contract].set_index("date")["price"], 21).dropna().iloc[-1]) if (data.long["contract"] == contract).sum() > 30 else float("nan")
        rows.append({"root": root, "contract": contract, "quantity": qty, "price": row["price"], "dte": row["dte"], "multiplier": spec.multiplier, "notional": notional, "gross_notional": abs(notional),
                     "initial_margin": abs(qty) * spec.initial_margin * margin_scale, "maintenance_margin": abs(qty) * spec.maintenance_margin * margin_scale, "volatility": vol,
                     "daily_1sd": abs(notional) * vol / np.sqrt(252.0) if np.isfinite(vol) else None})
    gross = sum(r["gross_notional"] for r in rows)
    im, mm = sum(r["initial_margin"] for r in rows), sum(r["maintenance_margin"] for r in rows)
    by_root: dict[str, float] = {}
    for r in rows:
        by_root[r["root"]] = by_root.get(r["root"], 0.0) + r["initial_margin"]
    top_root = max(by_root, key=by_root.get)
    usage = im / equity
    add_flag(flags, usage > 0.70, f"margin use is {usage:.0%} of equity: little room for a loss before a margin call")
    add_flag(flags, len(by_root) > 1 and im > 0 and by_root[top_root] / im > 0.5, f"{top_root} is {by_root[top_root] / im:.0%} of the margin: concentrated")
    add_flag(flags, len(by_root) == 1 and len(rows) >= 1, f"the whole book is in {top_root}")
    for r in rows:
        add_flag(flags, r["dte"] <= near_expiry_days, f"{r['contract']} ({r['root']}) expires in {r['dte']} days: roll or close")
    add_flag(flags, equity < mm, "equity is below the maintenance margin: a margin call")
    add_flag(flags, gross / equity > 10.0, f"leverage is {gross / equity:.1f}x")
    return {"equity": float(equity), "positions": rows, "gross_notional": float(gross), "net_notional": float(sum(r["notional"] for r in rows)), "initial_margin": float(im),
            "maintenance_margin": float(mm), "margin_usage": float(usage), "leverage": float(gross / equity), "excess_liquidity": float(equity - im), "buffer_to_call": float(equity - mm),
            "by_root": {k: float(v) for k, v in by_root.items()}, "margin_scale": float(margin_scale), "risk_flags": flags,
            "notes": ["Margins are illustrative per-contract figures, scaled by margin_scale; use your broker's current numbers."]}


def futures_snapshot(data: CurveData) -> AssetSnapshot:
    """The one-line summary: front price, annualised volatility of the front contract, a liquidity score from the calendar-spread cost and the dollar value of the front's open interest, and the flags."""
    snap = curve_snapshot(data, 6)
    front = data.long[data.long["contract"] == snap["front"]].set_index("date")
    vol = float(realized_vol(front["price"], 21).dropna().iloc[-1]) if len(front) > 30 else float("nan")
    spec = data.spec
    spread_bps = spec.spread_ticks * spec.tick_size / snap["front_price"] * 1e4
    oi = snap["rows"][0]["open_interest"]
    liquid = liquidity_score(spread_bps, (oi or 0.0) * snap["notional"] * 0.18) if oi else 0.0
    flags: list[str] = []
    add_flag(flags, snap["rows"][0]["dte"] <= 7, "the front contract is close to expiry")
    add_flag(flags, snap["state"] == "backwardation", "backwardation: the curve pays holders to wait (positive carry for longs)")
    add_flag(flags, snap["state"] == "contango", "contango: rolling a long position costs the carry")
    return AssetSnapshot(data.root, "future", snap["front_price"], vol, liquid, flags, str(data.as_of.date()), data.source)


# ------------------------------------------------------------------------------------------------------------------ strategies
FUTURES_STRATEGIES = {
    "carry_ts": {"label": "Carry (time-series)", "model": "carry_ts", "allocator": "sleeves", "lookback": "smooth", "text": "Long where the roll yield is positive and short where negative, sized by carry over volatility."},
    "carry_xs": {"label": "Carry (cross-sectional)", "model": "carry_xs", "allocator": "score_stack", "lookback": None, "text": "Long the high-carry roots against the low-carry ones within each asset class."},
    "basis_momentum": {"label": "Basis momentum", "model": "basis_momentum", "allocator": "score_stack", "lookback": None, "text": "Favour commodities whose front contract has outperformed the second over the last year."},
    "tsmom": {"label": "Trend", "model": "tsmom", "allocator": "score_stack", "lookback": "lookback", "text": "Long what has risen over the lookback, short what has fallen, sized by volatility."},
    "donchian": {"label": "Breakout", "model": "donchian", "allocator": "score_stack", "lookback": "entry", "text": "Long a new high of the lookback window, short a new low."},
    "xs_momentum": {"label": "Cross-sectional momentum", "model": "xs_momentum", "allocator": "score_stack", "lookback": "lookback", "text": "Long the strongest roots against the weakest."},
}
_BUNDLES: "OrderedDict[tuple, object]" = OrderedDict()


def futures_bundle_for(roots: list[str], as_of=None, n_days: int = HISTORY_DAYS):
    """The MarketBundle of the simulated roots (excess-return indices with CARRY_, BASISMOM_ and SLOPE_ signals), cached: building it takes seconds."""
    from .bundles import futures_bundle

    end = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    key = (tuple(sorted(r.upper() for r in roots)), str(end.date()), int(n_days))
    with _LOCK:
        if key in _BUNDLES:
            _BUNDLES.move_to_end(key)
            return _BUNDLES[key]
    curves = {r: simulated_curve(r, as_of, n_days).long[["date", "contract", "expiry", "price"]] for r in key[0]}
    classes = {r: ROOTS[r].asset_class for r in key[0]}
    bundle = futures_bundle(curves, classes, name="futures-simulated")
    with _LOCK:
        _BUNDLES[key] = bundle
        while len(_BUNDLES) > 3:
            _BUNDLES.popitem(last=False)
    return bundle


def run_futures_strategy(config, strategy: str, roots: list[str] | None = None, frequency: str | None = None, lookback: int | None = None, target_vol: float | None = None, cost_bps: float = 2.0,
                         as_of=None) -> dict:
    """Run one of ``FUTURES_STRATEGIES`` on the SIMULATED roots through the platform pipeline. The signal is traded as written (the calibrated forecast needs more history than the simulation has);
    ``cost_bps`` is the one-way cost of trading the whole book. Prices are a model, so the result checks the machinery, not the strategy."""
    from .strategy_runs import run_strategy, subset_bundle

    if strategy not in FUTURES_STRATEGIES:
        raise ValueError(f"unknown futures strategy '{strategy}'; choose from {sorted(FUTURES_STRATEGIES)}")
    spec = FUTURES_STRATEGIES[strategy]
    chosen = [r.upper() for r in (roots or list(ROOTS))]
    bad = [r for r in chosen if r not in ROOTS]
    if bad:
        raise ValueError(f"unknown futures roots {bad}; the simulated roots are {sorted(ROOTS)}")
    if len(chosen) < 2 and strategy in ("carry_xs", "xs_momentum"):
        raise ValueError("a cross-sectional strategy needs at least two roots")
    if strategy == "basis_momentum" and sum(ROOTS[r].asset_class == "commodity" for r in chosen) < 2:
        raise ValueError("basis momentum ranks commodities: choose at least two commodity roots (CL, NG, HG)")
    bundle = futures_bundle_for(chosen, as_of)
    params = {spec["lookback"]: int(lookback)} if (lookback is not None and spec["lookback"]) else {}
    out = run_strategy(subset_bundle(bundle, sorted(chosen)), config, spec["model"], params, rebalance=frequency, target_vol=target_vol, cost_bps=float(cost_bps), label=spec["label"],
                       allocator=spec["allocator"])
    out.update({"strategy": strategy, "label": spec["label"], "text": spec["text"], "roots": sorted(chosen), "cost_bps": float(cost_bps), "source": "simulated",
                "notes": ["The futures are a Schwartz-Smith model simulation: the result checks that carry, roll and costs are booked correctly, not whether the strategy earns money.",
                          "Returns are the excess return of holding the front contract and rolling it; margin and funding are not modelled."]})
    return out
