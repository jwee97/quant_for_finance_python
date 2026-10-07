"""The option-chain data model, its validation and the static no-arbitrage checks.

An ``OptionChain`` is a tidy table, one row per quote, with the columns

``date, expiry, strike, right ('C'/'P'), bid, ask, underlying, rate`` (required) and ``dividend, volume, open_interest, iv`` (optional).

Anything that can produce this table (a vendor CSV, ``derivatives.synthetic``) can be fed to the surface fitters and the options backtester. ``validate_chain`` reports
data problems (crossed quotes, expired contracts, duplicates) instead of silently repairing them, and ``arbitrage_report`` tests the quotes against the static no-arbitrage
conditions (Merton 1973; Carr & Madan 2005):

* call prices must be non-increasing and put prices non-decreasing in strike (no vertical-spread arbitrage);
* prices must be convex in strike (no butterfly arbitrage);
* total implied variance must be non-decreasing in expiry at a fixed log-moneyness (no calendar-spread arbitrage);
* ``C - P = S e^{-qT} - K e^{-rT}`` within the bid-ask spread (put-call parity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED = ["date", "expiry", "strike", "right", "bid", "ask", "underlying", "rate"]
OPTIONAL = {"dividend": 0.0, "volume": 0.0, "open_interest": 0.0}


@dataclass(frozen=True)
class OptionContract:
    underlying: str
    expiry: pd.Timestamp
    strike: float
    right: str                     # "C" or "P"
    multiplier: float = 100.0
    style: str = "european"

    @property
    def key(self) -> tuple:
        return (self.underlying, pd.Timestamp(self.expiry), float(self.strike), self.right)


def normalise_chain(df: pd.DataFrame, underlying: str = "UND") -> pd.DataFrame:
    """Coerce types, add optional columns, derive ``mid``, ``T`` (years to expiry, actual/365), the forward ``F`` and the log-moneyness ``k = ln(K / F)``."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"option chain is missing columns {missing}")
    out = df.copy()
    out["date"], out["expiry"] = pd.to_datetime(out["date"]), pd.to_datetime(out["expiry"])
    out["right"] = out["right"].astype(str).str.upper().str[0]
    for col, default in OPTIONAL.items():
        if col not in out:
            out[col] = default
    if "underlying_name" not in out:
        out["underlying_name"] = underlying
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["T"] = (out["expiry"] - out["date"]).dt.days / 365.0
    out["F"] = out["underlying"] * np.exp((out["rate"] - out["dividend"]) * out["T"])
    out["k"] = np.log(out["strike"] / out["F"])
    return out.sort_values(["date", "expiry", "strike", "right"]).reset_index(drop=True)


def validate_chain(df: pd.DataFrame) -> pd.DataFrame:
    """One row per problem: ``(check, count, example_index)``. An empty frame means the chain passed every check."""
    problems = []

    def add(check, mask):
        if int(mask.sum()):
            problems.append({"check": check, "count": int(mask.sum()), "example_index": int(np.flatnonzero(mask.to_numpy())[0])})

    add("bid > ask (crossed quote)", df["bid"] > df["ask"])
    add("negative bid or ask", (df["bid"] < 0) | (df["ask"] < 0))
    add("non-positive strike", df["strike"] <= 0)
    add("expiry on or before the quote date", df["expiry"] <= df["date"])
    add("right not C or P", ~df["right"].isin(["C", "P"]))
    add("duplicate (date, expiry, strike, right)", df.duplicated(["date", "expiry", "strike", "right"], keep=False))
    add("missing price or underlying", df[["bid", "ask", "underlying"]].isna().any(axis=1))
    call = df["right"] == "C"
    intrinsic = np.where(call, np.maximum(df["underlying"] - df["strike"], 0.0) * np.exp(-df["dividend"] * df["T"]) if "T" in df else 0.0, 0.0)
    if "T" in df:
        add("ask below intrinsic value", df["ask"] < intrinsic - 1e-9)
    return pd.DataFrame(problems, columns=["check", "count", "example_index"])


def otm_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """The out-of-the-money option at each strike (puts below the forward, calls above): the liquid, informative side used to fit surfaces."""
    return df[((df["right"] == "P") & (df["strike"] <= df["F"])) | ((df["right"] == "C") & (df["strike"] > df["F"]))]


def parity_forward(df_expiry: pd.DataFrame) -> dict:
    """Implied forward and discount factor from put-call parity at ONE date and expiry: regress ``C - P`` on ``K``: ``C - P = e^{-rT} (F - K)``; slope ``-e^{-rT}``,
    intercept ``e^{-rT} F``. Returns the implied ``F``, the implied rate and the rmse; uses mids at strikes with both a call and a put."""
    pivot = df_expiry.pivot_table(index="strike", columns="right", values="mid").dropna()
    if len(pivot) < 3:
        return {"forward": float("nan"), "rate": float("nan"), "rmse": float("nan")}
    y = (pivot["C"] - pivot["P"]).to_numpy()
    K = pivot.index.to_numpy()
    slope, intercept = np.polyfit(K, y, 1)
    disc = -slope
    T = float(df_expiry["T"].iloc[0])
    F = intercept / disc
    resid = y - (intercept + slope * K)
    return {"forward": float(F), "rate": float(-np.log(disc) / T) if disc > 0 and T > 0 else float("nan"), "rmse": float(np.sqrt(np.mean(resid ** 2)))}


def arbitrage_report(chain: pd.DataFrame, tol: float = 1e-8) -> pd.DataFrame:
    """Static-arbitrage violations in the MID prices of ONE quote date. One row per violation type with the count tested and the number violated (beyond ``tol``).
    ``chain`` must have ``mid``, ``T``, ``F``, ``k``, ``strike``, ``right``, ``expiry``, ``rate`` (use ``normalise_chain``) and, for the calendar test, an ``iv`` column."""
    rows = []
    v_tests = v_bad = b_tests = b_bad = p_tests = p_bad = 0
    for expiry, g in chain.groupby("expiry"):
        for right, h in g.groupby("right"):
            h = h.sort_values("strike")
            price, K = h["mid"].to_numpy(), h["strike"].to_numpy()
            df_ = np.exp(-h["rate"].iloc[0] * h["T"].iloc[0])
            d = np.diff(price) / np.diff(K)
            v_tests += len(d)
            v_bad += int(((d > tol) if right == "C" else (d < -tol)).sum() + ((d < -df_ - tol) if right == "C" else (d > df_ + tol)).sum())
            if len(h) >= 3:
                slope = np.diff(price) / np.diff(K)
                b_tests += len(slope) - 1
                b_bad += int((np.diff(slope) < -tol).sum())
        pivot = g.pivot_table(index="strike", columns="right", values="mid").dropna()
        if len(pivot):
            T, F, r = g["T"].iloc[0], g["F"].iloc[0], g["rate"].iloc[0]
            gap = pivot["C"] - pivot["P"] - np.exp(-r * T) * (F - pivot.index.to_numpy())
            half_spread = g.groupby("strike").apply(lambda x: (x["ask"] - x["bid"]).sum() / 2.0, include_groups=False).reindex(pivot.index)
            p_tests += len(pivot)
            p_bad += int((gap.abs() > half_spread + tol).sum())
    rows += [{"check": "vertical spread (price monotone in strike)", "tested": v_tests, "violations": v_bad},
             {"check": "butterfly (price convex in strike)", "tested": b_tests, "violations": b_bad},
             {"check": "put-call parity within the spread", "tested": p_tests, "violations": p_bad}]
    if "iv" in chain:
        cal_tests = cal_bad = 0
        otm = otm_quotes(chain)
        expiries = sorted(otm["expiry"].unique())
        for e1, e2 in zip(expiries[:-1], expiries[1:]):
            a, b = otm[otm["expiry"] == e1].sort_values("k"), otm[otm["expiry"] == e2].sort_values("k")
            if len(a) < 3 or len(b) < 3:
                continue
            ks = np.linspace(max(a["k"].min(), b["k"].min()), min(a["k"].max(), b["k"].max()), 15)
            if ks[0] >= ks[-1]:
                continue
            w1 = np.interp(ks, a["k"], a["iv"] ** 2 * a["T"])
            w2 = np.interp(ks, b["k"], b["iv"] ** 2 * b["T"])
            cal_tests += len(ks)
            cal_bad += int((w2 < w1 - 1e-9).sum())
        rows.append({"check": "calendar spread (total variance rises with expiry)", "tested": cal_tests, "violations": cal_bad})
    return pd.DataFrame(rows)
