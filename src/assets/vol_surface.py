"""The implied-volatility surface as a research object: what was observed, what is interpolated from it, what is fitted to it, and nothing else.

**Construction** (everything here is done in the data, not in the picture):

1. *Quotes.* The chain's quotes are cleaned and every rejection is counted by reason: no positive bid or a crossed market (bid above ask), no usable implied volatility, a volatility beyond ``max_iv``
   (a stale or erroneous print), a price worth almost nothing (``min_mid``: its volatility is rounding to the tick), a duplicate contract (the one with more open interest stays) and an expired option.
   ``selection`` picks the side: ``otm`` (the out-of-the-money option on each side of the forward: puts below it, calls above), ``call`` or ``put``.
2. *Coordinates.* ``k = ln(K / F_T)`` with the forward of THAT expiry (each expiry has its own carry), or the Black-Scholes call delta of the quote at its own volatility (a put is placed on the
   call-delta scale, 0 to 1). Days to expiry run along the other axis.
3. *One grid for every maturity.* The nodes are the same for all expiries. A node of an expiry is **observed-supported** only if it lies between two of that expiry's own quotes that are close enough
   (no further apart than ``max(4 x the median spacing, 0.10)`` in ``k``, ``0.15`` in delta): there the **total variance** ``w = sigma^2 T`` is interpolated linearly in the coordinate (the usual
   choice: it is the quantity that must increase with maturity) and converted back to volatility. Every other node is **masked** (``null``): outside the quoted strikes, inside a gap, and in an expiry with
   fewer than ``min_points`` quotes. Nothing is extrapolated, nothing is smoothed, and masked regions are never bridged.
4. *Fitted surface* (coordinate ``k`` only). An SVI slice is fitted to each expiry's quotes (weighted by their inverse total variance) and evaluated ONLY at the nodes that are observed-supported, so a fitted
   value never appears where there is no quote. Diagnostics: the error in volatility points, Gatheral's butterfly condition ``g(k) >= 0`` over the quoted range and over a wider range (outside the quotes is
   not trusted, and said so), and the calendar condition (fitted total variance must not fall with expiry at a node both slices support), counted and located. Raw and fitted are returned apart.
5. *Not enough data.* With fewer than two usable expiries there is no surface to draw: the answer says so (``status = "smiles_only"``) and carries the discrete smiles instead.

``term`` is the at-the-money volatility of each expiry (read at ``k = 0`` inside the quoted range only, raw and, where fitted, fitted) and, if dated chains were kept (``options_history``), the same
for an earlier date. Both ``vol`` (percent) and ``w`` (total variance) versions of every grid are returned so that the page switches between them without a second request.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..derivatives import surface as sv
from ..derivatives.greeks import bsm_greeks

SELECTIONS = ("otm", "call", "put")
COORDINATES = ("k", "delta")
MIN_POINTS = 4
MAX_IV = 3.0
MIN_MID = 0.05


def _prepare(chain: pd.DataFrame) -> pd.DataFrame:
    g = chain.copy()
    if "T" not in g:
        g["T"] = (pd.to_datetime(g["expiry"]) - pd.to_datetime(g["date"])).dt.days / 365.0
    if "F" not in g:
        g["F"] = g["underlying"] * np.exp((g["rate"] - g.get("dividend", 0.0)) * g["T"])
    if "mid" not in g:
        g["mid"] = 0.5 * (g["bid"] + g["ask"])
    for c in ("volume", "open_interest"):
        if c not in g:
            g[c] = 0.0
    g["k"] = np.log(g["strike"] / g["F"])
    return g


def clean_quotes(chain: pd.DataFrame, selection: str = "otm", max_iv: float = MAX_IV, min_mid: float = MIN_MID) -> tuple[pd.DataFrame, dict]:
    """The usable quotes and the count of every rejection (a quote is counted once, under the first reason that applies)."""
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {list(SELECTIONS)}")
    g = _prepare(chain)
    n = len(g)
    rej = {"expired": 0, "crossed_or_no_bid": 0, "no_implied_vol": 0, "extreme_vol": 0, "negligible_price": 0, "other_side": 0, "duplicate": 0}
    keep = pd.Series(True, index=g.index)

    def drop(mask, name):
        nonlocal keep
        m = keep & mask
        rej[name] += int(m.sum())
        keep = keep & ~mask
    drop(~(g["T"] > 0), "expired")
    drop(~(g["bid"] > 0) | (g["bid"] > g["ask"]), "crossed_or_no_bid")
    drop(~np.isfinite(g["iv"]) | ~(g["iv"] > 0), "no_implied_vol")
    drop(g["iv"] > max_iv, "extreme_vol")
    drop(g["mid"] < min_mid, "negligible_price")
    if selection == "otm":
        drop(~(((g["right"] == "P") & (g["strike"] <= g["F"])) | ((g["right"] == "C") & (g["strike"] > g["F"]))), "other_side")
    else:
        drop(g["right"] != ("C" if selection == "call" else "P"), "other_side")
    g = g[keep].sort_values("open_interest", ascending=False)
    dup = g.duplicated(["expiry", "strike", "right"])
    rej["duplicate"] = int(dup.sum())
    g = g[~dup].sort_values(["expiry", "strike"]).copy()
    if len(g):
        d = np.empty(len(g))
        for call in (True, False):
            m = (g["right"] == ("C" if call else "P")).to_numpy()
            if m.any():
                s = g[m]
                d[m] = np.asarray(bsm_greeks(s["underlying"].to_numpy(float), s["strike"].to_numpy(float), s["T"].to_numpy(float), s["rate"].to_numpy(float), s["dividend"].to_numpy(float) if "dividend" in s else 0.0,
                                             s["iv"].to_numpy(float), True)["delta"], dtype=float)                       # a put sits on the call-delta scale at its own volatility
        g["delta_call"] = d
    return g, {"received": int(n), "kept": int(len(g)), "rejected": rej}


def _gap_limit(x: np.ndarray, coordinate: str) -> float:
    base = 0.10 if coordinate == "k" else 0.15
    return float(max(4.0 * np.median(np.diff(x)), base)) if len(x) > 1 else base


def _support(x: np.ndarray, nodes: np.ndarray, limit: float) -> np.ndarray:
    """True where a node lies between two consecutive quotes no further apart than ``limit`` (or on a quote)."""
    ok = np.zeros(len(nodes), dtype=bool)
    if len(x) < 2:
        return ok
    i = np.searchsorted(x, nodes, side="right") - 1
    inside = (nodes >= x[0]) & (nodes <= x[-1])
    i = np.clip(i, 0, len(x) - 2)
    ok = inside & ((x[i + 1] - x[i]) <= limit)
    return ok


def _atm(x_k: np.ndarray, w: np.ndarray, limit: float) -> float | None:
    if len(x_k) < 2 or not (x_k[0] <= 0.0 <= x_k[-1]) or not _support(x_k, np.array([0.0]), limit)[0]:
        return None
    return float(np.interp(0.0, x_k, w))


def build(chain: pd.DataFrame, selection: str = "otm", coordinate: str = "k", min_dte: float = 0.0, max_dte: float = 10_000.0, nx: int = 41, fit: bool = True, max_iv: float = MAX_IV,
          min_points: int = MIN_POINTS) -> dict:
    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {list(COORDINATES)}")
    if not 11 <= nx <= 161:
        raise ValueError("nx must be between 11 and 161")
    g, report = clean_quotes(chain, selection, max_iv)
    g = g[(g["T"] * 365.0 >= min_dte) & (g["T"] * 365.0 <= max_dte)]
    xcol = "k" if coordinate == "k" else "delta_call"
    slices, observations, usable = [], [], []
    for e, s in g.groupby("expiry"):
        s = s.sort_values(xcol).drop_duplicates(xcol, keep="first")
        T = float(s["T"].iloc[0])
        info = {"expiry": str(pd.Timestamp(e).date()), "dte": float(round(T * 365.0, 2)), "T": T, "forward": float(s["F"].iloc[0]), "n": int(len(s)), "usable": False, "reason": "",
                "x_min": float(s[xcol].min()) if len(s) else None, "x_max": float(s[xcol].max()) if len(s) else None}
        for _, r in s.iterrows():
            observations.append({"expiry": info["expiry"], "dte": info["dte"], "strike": float(r["strike"]), "right": str(r["right"]), "k": float(r["k"]), "delta": float(r["delta_call"]), "x": float(r[xcol]),
                                 "iv": float(r["iv"]), "w": float(r["iv"] ** 2 * T), "open_interest": float(r["open_interest"]), "volume": float(r["volume"]), "forward": info["forward"]})
        if len(s) < min_points:
            info["reason"] = f"{len(s)} quotes (at least {min_points} are needed to interpolate a smile)"
        else:
            info["usable"] = True
            info["gap_limit"] = _gap_limit(s[xcol].to_numpy(), coordinate)
            usable.append((info, s))
        slices.append(info)
    slices.sort(key=lambda r: r["dte"])
    meta = {"selection": selection, "coordinate": coordinate, "quotes": report, "expiries_listed": len(slices), "expiries_usable": len(usable), "min_points": min_points, "extrapolated_nodes": 0}
    out = {"meta": meta, "slices": slices, "observations": observations, "x_label": "ln(K/F)" if coordinate == "k" else "call delta", "limits": [
        "Masked (empty) regions have no quote behind them: nothing is extrapolated and gaps are not bridged.", "Raw values are interpolated linearly in total variance between neighbouring quotes of one expiry; they are market mid-implied, not a model.",
        "Quotes are delayed or end-of-day mid prices; a volatility is only known to within its bid-ask spread."]}
    if len(usable) < 2:
        meta.update(status="smiles_only" if usable or observations else "empty",
                    reason=("only " + (f"{len(usable)} expiry has" if len(usable) == 1 else "no expiry has") + f" at least {min_points} usable quotes: a surface needs two maturities, so the discrete smiles are shown instead") if observations else
                           "no usable quote survived the cleaning (see the rejections)")
        out.update(x=[], y=[], z={"raw": {"vol": [], "w": []}, "fit": None}, fit=None, term=_term([], []), coverage={"nodes": 0, "supported": 0, "share": 0.0})
        return out
    lo, hi = min(i["x_min"] for i, _ in usable), max(i["x_max"] for i, _ in usable)
    nodes = np.linspace(lo, hi, nx)
    rows_raw_w, rows_fit_w, y, fits = [], [], [], []
    for info, s in usable:
        x, iv, T = s[xcol].to_numpy(float), s["iv"].to_numpy(float), info["T"]
        w = iv ** 2 * T
        sup = _support(x, nodes, info["gap_limit"])
        wr = np.where(sup, np.interp(nodes, x, w), np.nan)
        rows_raw_w.append(wr)
        y.append(info["dte"])
        info["supported_nodes"] = int(sup.sum())
        gaps = [[float(a), float(b)] for a, b in zip(x[:-1], x[1:]) if (b - a) > info["gap_limit"]]
        info["gaps"] = gaps
        kx = s["k"].to_numpy(float)
        o = np.argsort(kx)
        info["atm_iv"] = (lambda v: None if v is None else float(math.sqrt(max(v, 0.0) / T)))(_atm(kx[o], w[o], _gap_limit(kx[o], "k")))
        if fit and coordinate == "k":
            try:
                f = sv.fit_svi_slice(x, iv, T, weights=1.0 / (iv ** 2 * T), n_starts=8)
                wf = np.where(sup, f.w(nodes), np.nan)
                rows_fit_w.append(wf)
                rng = np.linspace(x[0], x[-1], 121)
                wide = np.linspace(x[0] - 0.5, x[-1] + 0.5, 241)
                info["fit"] = {"params": {"a": f.a, "b": f.b, "rho": f.rho, "m": f.m, "s": f.s}, "rmse_vol_points": float(f.rmse * 100.0), "min_g_in_quotes": float(f.density_g(rng).min()),
                               "min_g_wider": float(f.density_g(wide).min()), "butterfly_ok_in_quotes": bool(f.density_g(rng).min() >= -1e-8 and np.all(f.w(rng) > 0)),
                               "butterfly_ok_wider": bool(f.density_g(wide).min() >= -1e-8 and np.all(f.w(wide) > 0)), "quality": "good" if f.rmse * 100.0 <= 1.0 else "fair" if f.rmse * 100.0 <= 2.5 else "poor"}
                fits.append((info, f))
            except Exception as exc:                                             # noqa: BLE001 - one failed slice must not hide the others
                rows_fit_w.append(np.full(len(nodes), np.nan))
                info["fit"] = {"error": f"{type(exc).__name__}: {exc}"}
    raw_w = np.array(rows_raw_w)
    T_col = np.array([i["T"] for i, _ in usable])[:, None]
    z = {"raw": {"vol": _nan_to_none(np.sqrt(raw_w / T_col) * 100.0), "w": _nan_to_none(raw_w)}, "fit": None}
    diag = None
    if rows_fit_w:
        fw = np.array(rows_fit_w)
        z["fit"] = {"vol": _nan_to_none(np.sqrt(np.maximum(fw, 0) / T_col) * 100.0), "w": _nan_to_none(fw)}
        diag = {"method": "SVI per expiry, evaluated only at observed-supported nodes", "calendar": _calendar(fw, nodes, y), "butterfly_failing_slices": int(sum(1 for i, _ in fits if "fit" in i and i["fit"].get("butterfly_ok_in_quotes") is False)),
                "poor_fits": int(sum(1 for i, _ in fits if i["fit"].get("quality") == "poor")), "mean_rmse_vol_points": float(np.mean([i["fit"]["rmse_vol_points"] for i, _ in fits])),
                "calendar_raw": _calendar(raw_w, nodes, y), "reading": "free of the checked static arbitrage on the supported nodes" if False else None}
        bad = diag["calendar"]["violations"] or diag["butterfly_failing_slices"]
        diag["arbitrage_status"] = "violations_detected" if bad else "none_detected_on_tested_grid"
        diag["globally_arbitrage_free"] = False                                          # independent per-expiry SVI fits carry no joint no-arbitrage constraint: absence of detected violations is not a guarantee
        diag["status_text"] = ("Violations detected on the tested grid." if bad else "No violation detected on the tested grid. The slices are fitted independently, so this is NOT a globally arbitrage-free parameterisation (that needs a joint constraint, as in SSVI with its parameter conditions).")
        diag["reading"] = ("static arbitrage found in the fitted slices (see the counts): treat the fitted surface as a description of the quotes, not a tradable model" if bad else
                           "no butterfly or calendar violation on the supported nodes (finite grid; not a proof, and nothing outside the quotes is checked)")
    elif coordinate == "delta" and fit:
        out["limits"].append("The SVI fit is in log-moneyness: switch the coordinate to ln(K/F) to see the fitted surface.")
    sup_cells = int(np.isfinite(raw_w).sum())
    share = sup_cells / raw_w.size
    meta["recommended_view"] = "smile" if (len(usable) < 3 or share < 0.4) else "surface"
    meta["recommend_reason"] = (f"{len(usable)} usable expiries and {share:.0%} of the grid supported by quotes: read the smiles and the quote markers rather than a continuous sheet" if meta["recommended_view"] == "smile" else "")
    meta.update(status="surface", reason="", x_range=[float(lo), float(hi)], dte_range=[min(y), max(y)])
    out.update(x=[float(v) for v in nodes], y=[float(v) for v in y], z=z, fit=diag, coverage={"nodes": int(raw_w.size), "supported": sup_cells, "share": float(sup_cells / raw_w.size)},
               term=_term(slices, usable, fits))
    return out


def _nan_to_none(a: np.ndarray) -> list:
    return [[None if not np.isfinite(v) else round(float(v), 6) for v in row] for row in a]


def _calendar(w: np.ndarray, nodes: np.ndarray, dte: list) -> dict:
    """Total variance must not fall with maturity at a node two neighbouring expiries both support."""
    viol = []
    for i in range(len(w) - 1):
        both = np.isfinite(w[i]) & np.isfinite(w[i + 1])
        bad = both & (w[i + 1] < w[i] - 1e-9)
        for j in np.flatnonzero(bad):
            viol.append({"x": float(nodes[j]), "dte_from": dte[i], "dte_to": dte[i + 1], "w_from": float(w[i, j]), "w_to": float(w[i + 1, j])})
    return {"checked_nodes": int(sum(int((np.isfinite(w[i]) & np.isfinite(w[i + 1])).sum()) for i in range(len(w) - 1))), "violations": len(viol), "worst": max(viol, key=lambda v: v["w_from"] - v["w_to"]) if viol else None}


def _term(slices: list, usable: list, fits: list | None = None) -> dict:
    """At-the-money volatility by expiry from the quotes (raw) and, where fitted, from the fit; ``null`` where ``k = 0`` is outside the quoted range or inside a gap."""
    fitted = {i["expiry"]: f for i, f in (fits or [])}
    rows = []
    for info, s in usable:
        f = fitted.get(info["expiry"])
        atm_fit = None
        if f is not None and info.get("atm_iv") is not None:
            atm_fit = float(f.iv(0.0)) * 100.0
        rows.append({"expiry": info["expiry"], "dte": info["dte"], "atm_iv": None if info.get("atm_iv") is None else info["atm_iv"] * 100.0, "atm_iv_fit": atm_fit})
    return {"rows": rows, "method": "interpolated total variance at k = 0 inside the quoted range of each expiry; null where there is no quote around the money"}


def term_from_chain(chain: pd.DataFrame, selection: str = "otm") -> list[dict]:
    """The at-the-money term structure of another (earlier) chain, by the same rules, for the comparison line."""
    try:
        r = build(chain, selection, "k", fit=False)
    except ValueError:
        return []
    return [{"dte": row["dte"], "atm_iv": row["atm_iv"]} for row in r["term"]["rows"] if row["atm_iv"] is not None]
