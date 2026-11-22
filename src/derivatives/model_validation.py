"""Validating a calibrated option model: not only how well it reproduces the quotes it was fitted to.

An in-sample RMSE rewards a model for having enough parameters to bend through every quote, so a model is judged on several things together:

``out_of_sample``      fit on part of the chain (a share of the strikes, or whole expiries held out), test on the rest: a model that only memorised the fitted points does worse here, and one that
                       extrapolates to a longer expiry shows how it does where it was not told the answer.
``stability``          calibrate on several dates (each warm-started from the one before): parameters that jump around from day to day, or fits that need a different shape every day, are not describing a
                       stable market.
``residual_grid``      the signed volatility error by expiry and log-moneyness: structure in it (all the wings wrong, one expiry biased) is what an RMSE hides.
``scorecard``          the models side by side on fit, out-of-sample error, complexity (BIC and the number of parameters), numerical health (parameters at their bounds, ill-conditioning, failed evaluations)
                       and computation time, with the rank by each criterion and the rank by the first three together: an in-sample winner that loses out of sample is reported as such.
"""

from __future__ import annotations

import numpy as np

from . import calibration as cal


def split(quotes: cal.Quotes, mode: str = "strikes", fraction: float = 0.25, seed: int = 0):
    """``(train, test)`` quotes. ``strikes``: a random ``fraction`` of the quotes of every expiry is held out (interior ones only: the wings stay in the fit so that the test is interpolation);
    ``expiry``: the longest expiry is held out (extrapolation in time)."""
    n = len(quotes.T)
    if mode == "expiry":
        mats = quotes.maturities
        if len(mats) < 3:
            raise ValueError("holding out an expiry needs at least three expiries")
        test = quotes.T == mats[-1]
    elif mode == "strikes":
        rng = np.random.default_rng(seed)
        test = np.zeros(n, dtype=bool)
        for T in quotes.maturities:
            idx = np.flatnonzero(quotes.T == T)
            idx = idx[np.argsort(quotes.K[idx])]
            inner = idx[1:-1]
            k = int(round(len(inner) * fraction))
            if k:
                test[rng.choice(inner, size=k, replace=False)] = True
    else:
        raise ValueError("mode is strikes or expiry")
    if test.sum() < 2 or (~test).sum() < 4:
        raise ValueError("too few quotes to hold any out")
    return quotes.subset(~test), quotes.subset(test)


def out_of_sample(model: str, quotes: cal.Quotes, mode: str = "strikes", fraction: float = 0.25, seed: int = 0, **kw) -> dict:
    train, test = split(quotes, mode, fraction, seed)
    fit = cal.calibrate(model, train, **kw)
    iv = cal.model_ivs(fit.model, test)
    err = iv - test.iv
    ok = np.isfinite(err)
    if ok.sum() < 2:
        raise ValueError("the fitted model prices fewer than two held-out quotes to a usable volatility")
    oos = float(np.sqrt(np.mean(err[ok] ** 2)))
    return {"model": fit.model.name, "mode": mode, "n_train": len(train.T), "n_test": int(len(test.T)), "in_sample_rmse": fit.rmse, "out_of_sample_rmse": oos, "ratio": oos / fit.rmse if fit.rmse > 0 else float("inf"),
            "out_of_sample_bias": float(np.mean(err[ok])), "params": dict(fit.model.params), "flat_smile_rmse_test": cal.flat_smile_rmse(test)}


def stability(model: str, series: list[cal.Quotes], dates: list | None = None, **kw) -> dict:
    """Calibrate on each of ``series`` in order (a dated list of quote sets), each started from the previous fit. Returns the parameters through time, the fit through time and a stability measure:
    the mean absolute day-to-day change of each parameter relative to its range over the sample (0 stable, 1 as large as the whole range)."""
    if len(series) < 3:
        raise ValueError("stability needs at least three dated calibrations")
    rows, start = [], None
    for i, q in enumerate(series):
        fit = cal.calibrate(model, q, start=start, **kw)
        start = dict(fit.model.params)
        rows.append({"date": None if dates is None else str(dates[i]), "params": dict(fit.model.params), "rmse": fit.rmse, "at_bounds": list(fit.at_bounds), "success": fit.success})
    names = list(rows[0]["params"])
    arr = np.array([[r["params"][n] for n in names] for r in rows])
    rng = np.maximum(arr.max(axis=0) - arr.min(axis=0), 1e-12)
    jump = np.mean(np.abs(np.diff(arr, axis=0)), axis=0) / rng
    return {"model": model, "rows": rows, "parameter_instability": {n: float(j) for n, j in zip(names, jump)}, "mean_instability": float(jump.mean()), "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
            "reading": "stable" if jump.mean() < 0.35 else "unstable: the parameters move about as much as they range from one day to the next"}


def residual_grid(fit: cal.Calibration, quotes: cal.Quotes, bins: int = 7) -> dict:
    """Signed volatility error (model less market) by expiry (rows) and log-moneyness bin (columns), the mean of the quotes in each cell."""
    err = fit.residuals if fit.residuals is not None else (fit.model_iv - quotes.iv)
    k = np.log(quotes.K / quotes.forward)
    edges = np.linspace(k.min(), k.max() + 1e-12, bins + 1)
    cols = np.clip(np.digitize(k, edges) - 1, 0, bins - 1)
    mats = quotes.maturities
    grid = np.full((len(mats), bins), np.nan)
    count = np.zeros((len(mats), bins), dtype=int)
    for i, T in enumerate(mats):
        for j in range(bins):
            sel = (quotes.T == T) & (cols == j) & np.isfinite(err)
            if sel.any():
                grid[i, j] = float(err[sel].mean())
                count[i, j] = int(sel.sum())
    centres = 0.5 * (edges[:-1] + edges[1:])
    return {"expiries_days": [float(t * 365.0) for t in mats], "log_moneyness": [float(c) for c in centres], "error": [[None if not np.isfinite(v) else float(v) for v in row] for row in grid], "count": count.tolist(),
            "unit": "volatility (0.01 = one point), model less market"}


def scorecard(quotes: cal.Quotes, models=("merton", "kou", "vg", "nig", "heston"), mode: str = "strikes", seed: int = 0, loss: str = "iv", **kw) -> dict:
    """Fit every model, test each out of sample, and rank on several criteria rather than one."""
    rows = []
    for m in models:
        try:
            fit = cal.calibrate(m, quotes, loss=loss, **kw)
        except ValueError as error:
            rows.append({"model": m, "failed": str(error)})
            continue
        row = {"model": m, "label": fit.model.label, "in_sample_rmse": fit.rmse, "bic": fit.bic, "n_params": fit.n_params, "seconds": fit.seconds, "at_bounds": fit.at_bounds, "condition": fit.condition,
               "failed_evaluations": fit.n_failed, "success": fit.success, "within_band": fit.within_band, "params": dict(fit.model.params)}
        try:
            oos = out_of_sample(m, quotes, mode=mode, seed=seed, loss=loss, **kw)
            row.update(out_of_sample_rmse=oos["out_of_sample_rmse"], oos_ratio=oos["ratio"], oos_bias=oos["out_of_sample_bias"])
        except ValueError as error:
            row.update(out_of_sample_rmse=None, oos_ratio=None, oos_note=str(error))
        rows.append(row)
    ok = [r for r in rows if "failed" not in r]

    def rank(key):
        vals = sorted((r[key], i) for i, r in enumerate(ok) if r.get(key) is not None and np.isfinite(r[key]))
        return {ok[i]["model"]: n + 1 for n, (_, i) in enumerate(vals)}

    ranks = {"in_sample_rmse": rank("in_sample_rmse"), "out_of_sample_rmse": rank("out_of_sample_rmse"), "bic": rank("bic"), "seconds": rank("seconds")}
    combined = {r["model"]: np.mean([ranks[k][r["model"]] for k in ("in_sample_rmse", "out_of_sample_rmse", "bic") if r["model"] in ranks[k]]) for r in ok}
    best_in = min(ranks["in_sample_rmse"], key=ranks["in_sample_rmse"].get) if ranks["in_sample_rmse"] else None
    best_oos = min(ranks["out_of_sample_rmse"], key=ranks["out_of_sample_rmse"].get) if ranks["out_of_sample_rmse"] else None
    notes = []
    if best_in and best_oos and best_in != best_oos:
        notes.append(f"{best_in} fits the quotes best in sample but {best_oos} does best on the quotes held out: the in-sample ranking alone would mislead.")
    for r in ok:
        if r["at_bounds"]:
            notes.append(f"{r['model']}: parameters at their bounds ({', '.join(r['at_bounds'])}): the model is straining to fit.")
        if r.get("condition") and np.isfinite(r["condition"]) and r["condition"] > 1e8:
            notes.append(f"{r['model']}: the quotes barely pin down some combination of its parameters (condition {r['condition']:.1e}).")
    return {"models": rows, "ranks": ranks, "combined_rank_score": {k: float(v) for k, v in combined.items()}, "preferred_by_combined_rank": min(combined, key=combined.get) if combined else None,
            "validation": {"mode": mode, "loss": loss, "seed": seed}, "notes": notes,
            "caveat": "A single chain on a single date: out-of-sample here means held-out quotes of the same day, not a later date. Parameter stability needs several dated chains."}
