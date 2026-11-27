"""Keeping a forecast-driven book from amplifying a weak forecast: blending with a risk-based baseline, shrinking uncertain forecasts, demanding evidence, limiting how far a weight may move, and diagnostics that show how
sensitive the book is.

Blending. ``w_final = (1 - alpha) * w_baseline + alpha * w_ml`` where the baseline is equal weight, inverse volatility or equal risk contribution (a book that reads no forecast) and ``w_ml`` is the configured allocator's
book. Both are feasible under the configured limits and the limits are convex (box, sum, gross, sector), so the blend is feasible; if a later non-convex step (a turnover limit, a volatility target) or a weight-change limit
leaves it infeasible, the book is projected onto the feasible set (Euclidean projection, ``portfolio_alloc.project``) and the deviation is reported with the decision.

alpha comes from one of three rules, none of which reads anything after the decision date:

``fixed``       ``blend_alpha``.
``confidence``  ``blend_max`` times the mean, over the assets held, of the probability that the asset's calibrated slope is positive (``Phi(t)``; 0 where no edge is demonstrated).
``validation``  the alpha in a fixed grid that maximised the mean-variance utility ``mean(r) - lambda/2 var(r)`` of the realised window returns of the last ``blend_window`` *resolved* past decisions (a decision is
                resolved once its holding window has ended before the decision date), using the books both policies actually proposed then. With fewer than ``blend_min_obs`` resolved decisions the baseline is used
                (alpha = 0). Nothing here is fitted on a protected segment: it only reads outcomes that had already happened; the choice of *whether* to use the rule is made on the validation segment in the comparison.

Presets are documented sets of settings, not claims of optimality: ``conservative`` raises the risk aversion, demands dependence-aware significance, shrinks forecasts by one standard error, blends with inverse
volatility at half weight at most, and caps the move of any weight at 10 points a decision. The historical default risk aversion of 5 is unchanged unless a preset is chosen.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

BASELINES = ("none", "equal_weight", "inverse_vol", "erc")
BLENDS = ("none", "fixed", "confidence", "validation")
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

PRESETS = {
    "conservative": {
        "description": "Risk aversion 20, dependence-aware (HAC) calibration with a minimum t of 2, forecasts shrunk by one standard error of the slope, a 50% cap on the weight of the ML book in a confidence-weighted blend with inverse "
                       "volatility, no weight moving more than 10 points per decision, a 60% position cap and a small turnover penalty. A starting point that errs toward the risk-based book, not an optimum.",
        "allocation": {"risk_aversion": 20.0, "calib_inference": "hac", "calib_min_t": 2.0, "uncertainty_shrink": 1.0, "baseline": "inverse_vol", "blend": "confidence", "blend_max": 0.5, "max_weight_change": 0.10,
                       "turnover_penalty": 1.0},
        "limits": {"max_weight": 0.60},
    },
}


def concentration(w: pd.Series) -> dict:
    """How concentrated a book is: the Herfindahl index of its absolute weights, the effective number of positions (its inverse), the largest weight and the largest single share of gross exposure."""
    a = w.abs().astype(float)
    g = float(a.sum())
    if g <= 1e-12:
        return {"hhi": None, "effective_n": None, "max_weight": 0.0, "top_share": None, "gross": 0.0}
    share = a / g
    hhi = float((share ** 2).sum())
    return {"hhi": hhi, "effective_n": 1.0 / hhi, "max_weight": float(a.max()), "top_share": float(share.max()), "gross": g}


def confidence_alpha(conf: pd.Series | None, names: list, blend_max: float) -> tuple[float, str]:
    if conf is None:
        return 0.0, "the forecast carries no confidence estimate: the baseline is used"
    c = conf.reindex(names).fillna(0.0)
    a = float(blend_max * c.mean())
    return a, f"mean confidence {float(c.mean()):.2f} over the assets held, times the cap {blend_max:g}"


def validation_alpha(history: list, real: pd.DataFrame, i: int, lag: int, L: int, window: int, min_obs: int, risk_aversion: float) -> tuple[float, dict]:
    """The alpha that would have done best, by mean-variance utility, over the last ``window`` decisions whose holding window had ended before row ``i``. ``history`` holds (row, baseline book, ML book) for each past decision."""
    resolved = [(s, wb, wm) for (s, wb, wm) in history if s <= i - lag - L]
    resolved = resolved[-int(window):]
    rows = []
    for s, wb, wm in resolved:
        y = real.iloc[s].reindex(wb.index)
        if y.isna().any():
            continue
        rows.append((float(wb @ y), float(wm @ y)))
    if len(rows) < min_obs:
        return 0.0, {"resolved": len(rows), "needed": int(min_obs), "note": f"{len(rows)} resolved decisions; {int(min_obs)} are needed, so the baseline is used"}
    rb, rm = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    best, best_u, utils = 0.0, -math.inf, {}
    for a in ALPHA_GRID:
        r = (1.0 - a) * rb + a * rm
        u = float(r.mean() - 0.5 * risk_aversion * r.var(ddof=1))
        utils[str(a)] = u
        if u > best_u + 1e-15:
            best, best_u = a, u
    return float(best), {"resolved": len(rows), "utility_by_alpha": utils, "note": f"the best of {list(ALPHA_GRID)} over {len(rows)} resolved past decisions"}


def weight_change_limited(w: pd.Series, w0: pd.Series, delta: float, lo: float, hi: float, total: float | None) -> tuple[pd.Series, bool, bool]:
    """Move ``w`` no further than ``delta`` per asset from ``w0``. For a fully invested long-only book the answer is the Euclidean projection of ``w`` onto {sum = ``total``, max(lo, w0 - delta) <= w <= min(hi, w0 + delta)};
    if that set is empty the move cannot be limited without breaking the sum and ``feasible`` is False. Returns the book, whether the limit changed it and whether it could be honoured."""
    lo_v = np.maximum(lo, w0.to_numpy() - delta)
    hi_v = np.minimum(hi, w0.to_numpy() + delta)
    v = w.to_numpy(dtype=float)
    if (np.abs(v - w0.to_numpy()) <= delta + 1e-12).all():
        return w, False, True
    if total is None:                                                                   # no sum constraint to keep: clip each asset
        return pd.Series(np.clip(v, lo_v, hi_v), index=w.index), True, True
    if lo_v.sum() > total + 1e-12 or hi_v.sum() < total - 1e-12:
        return w, False, False
    from .portfolio_alloc import _box_sum_projection

    return pd.Series(_box_sum_projection(v, lo_v, hi_v, float(total)), index=w.index), True, True


def uncertainty_shrunk_slopes(slopes: pd.Series, se: pd.Series | float | None, kappa: float) -> pd.Series:
    """The slope reduced by ``kappa`` standard errors (a lower confidence bound), floored at zero: a forecast whose slope is imprecise loses more of its size."""
    if kappa <= 0 or se is None:
        return slopes
    se_v = se if isinstance(se, pd.Series) else pd.Series(float(se), index=slopes.index)
    return (slopes - kappa * se_v.reindex(slopes.index).fillna(0.0)).clip(lower=0.0)


def narrative(rec: dict, cfg_content: dict) -> dict:
    """Plain-language account of one decision, per stock and for the book: what the model forecast and how far to believe it, how risky the stock is and how it moves with the others, what the baseline and the ML book
    each wanted, and which limits held the result back. Built only from the record, so it says what happened, not what should have."""
    names = rec.get("assets") or list(rec.get("weights", {}))
    w, ex = rec.get("weights", {}), rec.get("explain") or {}
    fc = rec.get("forecast") or {}
    bl = rec.get("blend") or {}
    pct = lambda v: "n/a" if v is None else f"{100 * float(v):+.2f}%"
    pc = lambda v: "n/a" if v is None else f"{100 * float(v):.1f}%"
    per = {}
    corr = rec.get("correlation") or {}
    for a in names:
        parts = []
        mu_raw, mu, sd = (rec.get("mu_raw") or {}).get(a), (rec.get("mu") or {}).get(a), (rec.get("sd") or {}).get(a)
        parts.append(f"the model forecast {pct(mu_raw)} over the window; after calibration the expected return used is {pct(mu)}" + (f" (forecast error sd {pc(sd)})" if sd is not None else ""))
        slopes = fc.get("slopes") or {}
        if fc.get("status") == "calibrated":
            sl = slopes.get(a, fc.get("slope"))
            conf = (fc.get("confidence") or {}).get(a)
            parts.append(f"calibration slope {sl:.2f}" + (f" ± {fc['slope_se']:.2f}" if fc.get("slope_se") else "") + (f", confidence {conf:.2f}" if conf is not None else ""))
        elif fc.get("status"):
            parts.append(f"no calibrated edge ({str(fc['status']).replace('_', ' ')}): the forecast did not move the weight")
        vol = (rec.get("volatility_daily") or {}).get(a)
        if vol is not None:
            parts.append(f"volatility {pc(vol * math.sqrt(252.0))} a year")
        rc = ((rec.get("risk_contributions") or {}).get(a) or {}).get("share")
        if rc is not None:
            parts.append(f"{pc(rc)} of the book's risk")
        others = [b for b in names if b != a and (corr.get(a) or {}).get(b) is not None]
        if others:
            parts.append("correlation with " + ", ".join(f"{b} {corr[a][b]:+.2f}" for b in others[:4]))
        eq, re, fe = (ex.get("equal") or {}).get(a), (ex.get("risk_effect") or {}).get(a), (ex.get("forecast_effect") or {}).get(a)
        if eq is not None:
            pts = lambda v: "n/a" if v is None else f"{100 * v:+.1f} pts"
            parts.append(f"starting from {pc(eq)} equal weight, risk and correlation moved it {pts(re)} and the forecast {pts(fe)}")
        if bl.get("status") == "blended":
            parts.append(f"baseline {bl['baseline']} wanted {pc((bl.get('baseline_weights') or {}).get(a))}, the ML book {pc((bl.get('ml_weights') or {}).get(a))}, blended with alpha {bl['alpha']:.2f}")
        per[a] = f"{a}: " + "; ".join(parts) + f". Final weight {pc(w.get(a))}."
    book = []
    driver = ex.get("driver")
    book.append({"forecast": "The weights are mainly set by the forecast.", "risk": "The weights are mainly set by risk and correlation, not by the forecast.", "constraints": "The weights are mainly held where the limits put them.",
                 "equal": "The book is the equal-weight book."}.get(driver, ""))
    if ex.get("binding"):
        book.append("Binding limits: " + "; ".join(ex["binding"]) + ".")
    if rec.get("weight_change_limited"):
        book.append(f"The weight-change limit of {rec['weight_change_limited']['limit']:g} restricted the move.")
    if rec.get("turnover_limited"):
        book.append("The turnover limit restricted the move.")
    if (rec.get("evidence") or {}):
        book.append("Evidence controls applied: " + "; ".join(k.replace("_", " ") for k in rec["evidence"]) + ".")
    if fc.get("fallback"):
        book.append("Calibration fell back to the pooled slope: " + fc["fallback"] + ".")
    conc = rec.get("concentration") or {}
    if conc.get("effective_n") is not None:
        book.append(f"Effective number of positions {conc['effective_n']:.2f}; largest weight {pc(conc.get('max_weight'))}.")
    return {"per_asset": per, "book": " ".join(x for x in book if x)}
