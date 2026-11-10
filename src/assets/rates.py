"""Fixed income: yield-curve models, bond analytics and carry / roll-down.

* ``nelson_siegel`` / ``svensson``: smooth parametric curves. ``fit_nelson_siegel`` fits ``(beta0, beta1, beta2)`` by least squares for each decay ``lambda`` on a grid; the three betas are the
  level, slope and curvature of the curve (Diebold & Li 2006).
* ``dynamic_nelson_siegel``: fit the curve every day, treat the betas as a VAR(1) and forecast the whole curve (Diebold-Li); ``synthetic_curve_panel`` simulates such a world.
* ``bond_price`` / ``yield_to_maturity`` / ``duration`` / ``convexity`` / ``dv01`` / ``key_rate_durations``: the analytics of a fixed-coupon bond.
* ``bootstrap_zero_curve``: zero rates from par yields; ``forward_rate`` between two maturities.
* ``carry_rolldown``: the expected return of holding a bond for ``horizon`` years if the curve does not move: the coupon accrual (carry) plus the price gain from rolling down a
  steep curve to a lower yield.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy import optimize


def nelson_ns_loadings(tau, lam):
    tau = np.asarray(tau, dtype=float)
    x = lam * tau
    f1 = (1.0 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(tau), f1, f1 - np.exp(-x)])


def nelson_siegel(tau, beta0, beta1, beta2, lam):
    """Zero yield at maturity ``tau`` (years): ``b0 + b1 (1 - e^{-lam t})/(lam t) + b2 [(1 - e^{-lam t})/(lam t) - e^{-lam t}]``."""
    return nelson_ns_loadings(tau, lam) @ np.array([beta0, beta1, beta2])


def svensson(tau, beta0, beta1, beta2, beta3, lam1, lam2):
    tau = np.asarray(tau, dtype=float)
    x1, x2 = lam1 * tau, lam2 * tau
    f1 = (1 - np.exp(-x1)) / x1
    f2 = (1 - np.exp(-x2)) / x2
    return beta0 + beta1 * f1 + beta2 * (f1 - np.exp(-x1)) + beta3 * (f2 - np.exp(-x2))


def fit_nelson_siegel(maturities, yields, lam_grid=None) -> dict:
    """Least-squares fit: for each ``lambda`` on the grid the betas solve a linear regression; the best ``lambda`` minimises the residual sum of squares."""
    tau, y = np.asarray(maturities, float), np.asarray(yields, float)
    grid = np.linspace(0.2, 3.0, 57) if lam_grid is None else np.asarray(lam_grid)
    best = None
    for lam in grid:
        X = nelson_ns_loadings(tau, lam)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(((y - X @ beta) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, lam, beta)
    rss, lam, beta = best
    return {"beta0": float(beta[0]), "beta1": float(beta[1]), "beta2": float(beta[2]), "lambda": float(lam), "rmse": float(np.sqrt(rss / len(y)))}


def dynamic_nelson_siegel(yields: pd.DataFrame, lam: float = 0.5) -> dict:
    """Diebold-Li: with ``lambda`` fixed, the daily betas are an OLS regression of the cross-section of yields on the loadings. ``yields`` has maturities (years) as columns. Returns the
    factor panel ``level, slope, curvature`` (sign conventions as the loadings: ``beta1 = -slope``) and the fitted curves' RMSE."""
    tau = np.asarray(yields.columns, dtype=float)
    X = nelson_ns_loadings(tau, lam)
    pinv = np.linalg.pinv(X)
    betas = yields.to_numpy() @ pinv.T
    fitted = betas @ X.T
    return {"factors": pd.DataFrame(betas, index=yields.index, columns=["level", "slope", "curvature"]), "rmse": float(np.sqrt(np.mean((yields.to_numpy() - fitted) ** 2))), "lambda": lam, "loadings": X}


def forecast_curve(factors: pd.DataFrame, maturities, lam: float = 0.5, horizon: int = 21) -> pd.Series:
    """Diebold-Li forecast: a VAR(1) on the three factors (OLS), iterated ``horizon`` steps, mapped back to a curve."""
    from ..econometrics.var import fit_var

    var = fit_var(factors, 1)
    f = var.forecast(horizon).iloc[-1].to_numpy()
    return pd.Series(nelson_ns_loadings(np.asarray(maturities, float), lam) @ f, index=maturities)


def synthetic_curve_panel(n_days: int = 2000, maturities=(0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30), lam: float = 0.5, seed: int = 0, start: str = "2010-01-04") -> pd.DataFrame:
    """Daily zero yields from VAR(1) dynamics of Nelson-Siegel factors (level persistent, slope and curvature faster) plus measurement noise of about one basis point."""
    rng = np.random.default_rng(seed)
    A = np.diag([0.9995, 0.995, 0.99])
    mean = np.array([0.04, -0.015, -0.01])
    sd = np.array([0.0004, 0.0006, 0.0010])
    f = np.empty((n_days, 3))
    f[0] = mean
    for t in range(1, n_days):
        f[t] = mean + A @ (f[t - 1] - mean) + sd * rng.standard_normal(3) + np.array([-0.2, 0.3, 0.0]) * 0.0
    X = nelson_ns_loadings(np.asarray(maturities, float), lam)
    y = f @ X.T + 0.0001 * rng.standard_normal((n_days, len(maturities)))
    return pd.DataFrame(y, index=pd.bdate_range(start, periods=n_days), columns=list(maturities))


# ------------------------------------------------------------------------------------------------------------------ the yield-curve tool (the Rates tab, ``quant curve``)
MODELS = ("nelson_siegel", "svensson")
INPUTS = ("zero", "par")
MIN_POINTS = {"nelson_siegel": 4, "svensson": 6}                         # parameters to find: three betas and a decay; four betas and two decays
MAX_POINTS = 60
YIELD_LIMITS = (-5.0, 60.0)                                              # percent
MATURITY_LIMITS = (1.0 / 365.0, 100.0)                                   # years
HUMP = 1.7933                                                            # the maturity (in units of 1/lambda) at which the curvature loading peaks
TENOR_UNITS = {"D": 1.0 / 365.0, "W": 7.0 / 365.0, "M": 1.0 / 12.0, "Y": 1.0}
KEY_MATURITIES = (0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30)


def parse_tenor(text) -> float:
    """A maturity in years from ``3M``, ``10Y``, ``90D``, ``2W`` or a plain number of years (``0.25``)."""
    if isinstance(text, (int, float, np.integer, np.floating)) and not isinstance(text, bool):
        value = float(text)
    else:
        t = str(text).strip().upper().replace("YR", "Y").replace("YRS", "Y").replace("YEARS", "Y").replace("YEAR", "Y").replace("MO", "M").replace("MTH", "M")
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([DWMY]?)", t)
        if not match:
            raise ValueError(f"{text!r} is not a maturity: write it in years (0.25) or as 3M, 2Y, 30Y")
        value = float(match.group(1)) * TENOR_UNITS[match.group(2) or "Y"]
    if not math.isfinite(value) or not MATURITY_LIMITS[0] <= value <= MATURITY_LIMITS[1]:
        raise ValueError(f"a maturity must be between one day and {MATURITY_LIMITS[1]:g} years, not {text!r}")
    return value


def parse_curve_text(text) -> tuple[np.ndarray, np.ndarray]:
    """Maturities (years) and yields (as typed, in percent unless the caller says otherwise) from lines of ``maturity yield``: ``2Y 4.25``, ``0.5, 4.8``, ``10 4.31``. A first line that is not a pair of
    numbers (a header) is skipped; blank lines are ignored."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("type or paste the curve: one maturity and one yield on each line (2Y 4.25)")
    if len(text) > 20_000:
        raise ValueError("a curve has at most 60 points: this text is far longer")
    mats, ys = [], []
    for i, line in enumerate(l for l in text.splitlines() if l.strip()):
        parts = [x for x in re.split(r"[\s,;\t]+", line.strip()) if x]
        if len(parts) < 2:
            raise ValueError(f"line {i + 1} ({line.strip()!r}) needs a maturity and a yield")
        try:
            m, y = parse_tenor(parts[0]), float(parts[1].replace("%", ""))
        except ValueError:
            if i == 0:
                continue                                                                           # a header line
            raise
        mats.append(m)
        ys.append(y)
    return np.array(mats), np.array(ys)


def check_curve(maturities, yields, model: str, percent: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The points of a curve as arrays, sorted by maturity and checked: finite, in range, no maturity twice, enough of them for the model."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {', '.join(MODELS)}")
    tau, y = np.asarray(maturities, dtype=float), np.asarray(yields, dtype=float)
    if tau.ndim != 1 or tau.shape != y.shape:
        raise ValueError("give one yield for every maturity")
    if len(tau) < MIN_POINTS[model]:
        raise ValueError(f"the {model.replace('_', '-')} model has {MIN_POINTS[model]} parameters to find: give at least {MIN_POINTS[model]} points (preferably more than {MIN_POINTS[model] + 2})")
    if len(tau) > MAX_POINTS:
        raise ValueError(f"at most {MAX_POINTS} points")
    if not (np.all(np.isfinite(tau)) and np.all(np.isfinite(y))):
        raise ValueError("every maturity and yield must be a finite number")
    if tau.min() < MATURITY_LIMITS[0] or tau.max() > MATURITY_LIMITS[1]:
        raise ValueError(f"maturities must be between one day and {MATURITY_LIMITS[1]:g} years")
    lo, hi = YIELD_LIMITS if percent else (YIELD_LIMITS[0] / 100, YIELD_LIMITS[1] / 100)
    if y.min() < lo or y.max() > hi:
        raise ValueError(f"yields must be between {lo:g} and {hi:g}{' percent' if percent else ''}: check the unit")
    order = np.argsort(tau)
    tau, y = tau[order], y[order]
    if np.any(np.diff(tau) <= 0):
        raise ValueError("a maturity appears twice")
    return tau, y


def svensson_loadings(tau, lam1: float, lam2: float) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    x1, x2 = lam1 * tau, lam2 * tau
    f1, f2 = (1.0 - np.exp(-x1)) / x1, (1.0 - np.exp(-x2)) / x2
    return np.column_stack([np.ones_like(tau), f1, f1 - np.exp(-x1), f2 - np.exp(-x2)])


def fit_svensson(maturities, yields, lam1_grid=None, lam2_grid=None) -> dict:
    """Least-squares Svensson fit: for each pair of decays ``(lambda1, lambda2)`` on a grid the four betas solve a linear regression; the best pair starts a local search (Nelder-Mead on the decays, the
    betas solved exactly at each step). The two decays are kept apart (the two curvature loadings are nearly the same function when they are close, and the betas then cannot be told apart)."""
    tau, y = np.asarray(maturities, float), np.asarray(yields, float)
    g1 = np.linspace(0.15, 3.0, 20) if lam1_grid is None else np.asarray(lam1_grid, float)
    g2 = np.linspace(0.03, 1.5, 22) if lam2_grid is None else np.asarray(lam2_grid, float)

    def solve(l1: float, l2: float):
        X = svensson_loadings(tau, l1, l2)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(((y - X @ beta) ** 2).sum()), beta

    best = None
    for l1 in g1:
        for l2 in g2:
            if abs(l1 - l2) < 0.05:
                continue
            rss, beta = solve(l1, l2)
            if best is None or rss < best[0]:
                best = (rss, l1, l2, beta)
    if best is None:
        raise ValueError("the grids of decays leave no pair that is apart")
    rss, l1, l2, beta = best

    def objective(v):
        a, b = float(v[0]), float(v[1])
        if a <= 0.02 or b <= 0.02 or a > 6.0 or b > 6.0 or abs(a - b) < 0.05:
            return 1e6
        return solve(a, b)[0]

    if lam1_grid is None and lam2_grid is None:
        res = optimize.minimize(objective, [l1, l2], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-14, "maxiter": 300})
        if res.fun < rss:
            l1, l2 = float(res.x[0]), float(res.x[1])
            rss, beta = solve(l1, l2)
    return {"beta0": float(beta[0]), "beta1": float(beta[1]), "beta2": float(beta[2]), "beta3": float(beta[3]), "lambda1": float(l1), "lambda2": float(l2), "rmse": float(np.sqrt(rss / len(y)))}


def _curve_values(model: str, par: dict, tau) -> dict:
    """The zero (spot) yield, the instantaneous forward rate and the discount factor of a fitted curve at maturities ``tau`` (continuous compounding; rates as decimals)."""
    tau = np.asarray(tau, dtype=float)
    if model == "nelson_siegel":
        zero = nelson_siegel(tau, par["beta0"], par["beta1"], par["beta2"], par["lambda"])
        e = np.exp(-par["lambda"] * tau)
        fwd = par["beta0"] + par["beta1"] * e + par["beta2"] * par["lambda"] * tau * e
    else:
        zero = svensson(tau, par["beta0"], par["beta1"], par["beta2"], par["beta3"], par["lambda1"], par["lambda2"])
        e1, e2 = np.exp(-par["lambda1"] * tau), np.exp(-par["lambda2"] * tau)
        fwd = par["beta0"] + par["beta1"] * e1 + par["beta2"] * par["lambda1"] * tau * e1 + par["beta3"] * par["lambda2"] * tau * e2
    return {"zero": zero, "forward": fwd, "discount": np.exp(-zero * tau)}


def curve_report(maturities, yields, model: str = "nelson_siegel", lam: float | None = None, kind: str = "zero", percent: bool = True) -> dict:
    """Fit a Nelson-Siegel or Svensson curve to observed yields and describe it: the parameters and what they mean (level, slope, curvature), how closely the curve passes through each observed yield, and
    the fitted zero curve, forward curve and discount factors, in percent. ``kind`` says whether the yields are zero (spot) rates or par yields of coupon bonds (these are first bootstrapped to zero
    rates; par yields are close to zero rates only on a flat curve). ``lam`` fixes the decay of a Nelson-Siegel fit (per year); otherwise it is found. Nothing here is a forecast."""
    if kind not in INPUTS:
        raise ValueError(f"the yields are either zero rates or par yields: kind must be one of {', '.join(INPUTS)}")
    tau, y = check_curve(maturities, yields, model, percent)
    y = y / 100.0 if percent else y
    notes, warnings = [], []
    fit_tau, fit_y = tau, y
    if kind == "par":
        zero = bootstrap_zero_curve(tau, y)
        fit_tau, fit_y = zero.index.to_numpy(float), zero.to_numpy(float)
        notes.append("The par yields were turned into zero (spot) rates by bootstrapping at half-year maturities before the curve was fitted; the points below are those zero rates.")
    if model == "nelson_siegel":
        if lam is not None and not 0.05 <= float(lam) <= 6.0:
            raise ValueError("the decay lambda is per year, between 0.05 and 6 (0.5 puts the curvature hump at about 3.6 years)")
        fit = fit_nelson_siegel(fit_tau, fit_y, None if lam is None else [float(lam)])
        par = {k: fit[k] for k in ("beta0", "beta1", "beta2", "lambda", "rmse")}
        if lam is None and (par["lambda"] <= 0.21 or par["lambda"] >= 2.99):
            warnings.append(f"The decay hit the edge of the range searched ({par['lambda']:.2f} a year): the curve has little curvature to place, so the hump maturity is not well determined.")
        factors = {"level": par["beta0"] * 100, "short_rate": (par["beta0"] + par["beta1"]) * 100, "slope": -par["beta1"] * 100, "curvature": par["beta2"] * 100, "hump_maturity": HUMP / par["lambda"]}
        shape = [("Level (β0)", par["beta0"] * 100, "the yield at very long maturities"), ("Slope (−β1)", -par["beta1"] * 100, "long rate less short rate; negative: an inverted curve"),
                 ("Curvature (β2)", par["beta2"] * 100, "positive: a hump at medium maturities; negative: a trough"), ("Decay (λ)", par["lambda"], "per year; the curvature factor peaks at 1.79/λ years")]
    else:
        if fit_tau.size < 8:
            warnings.append("Six or seven points leave the Svensson fit almost nothing to check it against: its six parameters can pass through them without the curve being right in between.")
        par = fit_svensson(fit_tau, fit_y)
        if par["lambda1"] <= 0.16 or par["lambda1"] >= 2.99 or par["lambda2"] <= 0.04 or par["lambda2"] >= 1.49:
            warnings.append("A decay hit the edge of the range searched: that hump is not well determined by these maturities.")
        factors = {"level": par["beta0"] * 100, "short_rate": (par["beta0"] + par["beta1"]) * 100, "slope": -par["beta1"] * 100, "curvature": par["beta2"] * 100, "second_curvature": par["beta3"] * 100,
                   "hump_maturity": HUMP / par["lambda1"], "second_hump_maturity": HUMP / par["lambda2"]}
        shape = [("Level (β0)", par["beta0"] * 100, "the yield at very long maturities"), ("Slope (−β1)", -par["beta1"] * 100, "long rate less short rate; negative: an inverted curve"),
                 ("First curvature (β2)", par["beta2"] * 100, f"a hump or trough near {HUMP / par['lambda1']:.1f} years"), ("Second curvature (β3)", par["beta3"] * 100, f"a second hump or trough near {HUMP / par['lambda2']:.1f} years"),
                 ("Decays (λ1, λ2)", None, f"{par['lambda1']:.2f} and {par['lambda2']:.2f} a year")]
    fitted = _curve_values(model, par, fit_tau)["zero"]
    resid_bp = (fit_y - fitted) * 1e4
    rmse_bp = float(np.sqrt(np.mean(resid_bp ** 2)))
    if rmse_bp > 5.0:
        warnings.append(f"The curve misses the observed yields by {rmse_bp:.1f} basis points on average: the curve has more shape than this model can draw" + (" (try the Svensson model)." if model == "nelson_siegel" else "."))
    top = max(30.0, float(tau.max()))
    grid = np.unique(np.concatenate([np.linspace(0.1, top, 120), fit_tau]))
    vals = _curve_values(model, par, grid)
    if float(np.min(vals["forward"])) < 0:
        warnings.append("The fitted forward rate is negative at some maturities: the curve implies a fall in rates there.")
    lo, hi = float(tau.min()), float(tau.max())
    key = [m for m in KEY_MATURITIES if lo * 0.999 <= m <= hi * 1.001]
    kv = _curve_values(model, par, key) if key else {"zero": [], "forward": [], "discount": []}
    if hi < 30.0:
        notes.append(f"The observed maturities run from {lo:g} to {hi:g} years: the curve drawn beyond {hi:g} years is the model's extrapolation, not an observation.")
    return {"model": model, "kind": kind, "parameters": {k: float(v) for k, v in par.items()}, "shape": [{"name": n, "value": v, "text": t} for n, v, t in shape], "factors": factors,
            "fit": {"rmse_bp": rmse_bp, "max_abs_bp": float(np.max(np.abs(resid_bp))), "points": [{"maturity": float(t), "observed": float(o * 100), "fitted": float(f * 100), "residual_bp": float(r)}
                                                                                                    for t, o, f, r in zip(fit_tau, fit_y, fitted, resid_bp)]},
            "curve": {"maturities": grid.tolist(), "zero": (vals["zero"] * 100).tolist(), "forward": (vals["forward"] * 100).tolist(), "discount": vals["discount"].tolist()},
            "key_rates": [{"maturity": float(m), "zero": float(z * 100), "forward": float(f * 100), "discount": float(d)} for m, z, f, d in zip(key, kv["zero"], kv["forward"], kv["discount"])],
            "observed_range": [lo, hi], "warnings": warnings, "notes": notes}


def parse_panel_text(text, percent: bool = True) -> pd.DataFrame:
    """A panel of yield curves from CSV text: a header whose first cell is the date column and whose other cells are maturities (``0.25``, ``3M``, ``10Y``), then one row a day. Returns decimals with
    maturities in years as the columns; a row with a missing yield is dropped."""
    import io

    if not isinstance(text, str) or not text.strip():
        raise ValueError("paste the history: a header of maturities (date, 3M, 2Y, 10Y…) and one row of yields a day")
    if len(text) > 150_000:
        raise ValueError("the pasted history is larger than the page reads (about 150 KB)")
    try:
        frame = pd.read_csv(io.StringIO(text), sep=None, engine="python", index_col=0)
    except Exception as error:                                                                      # noqa: BLE001 - the parser's own message says what is wrong
        raise ValueError(f"the history could not be read as a table: {error}") from None
    try:
        frame.index = pd.to_datetime(frame.index)
    except (ValueError, TypeError):
        raise ValueError("the first column must hold dates (2024-03-28)") from None
    mats = [parse_tenor(c) for c in frame.columns]
    frame.columns = mats
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna().sort_index()
    frame = frame[~frame.index.duplicated()]
    frame = frame.loc[:, sorted(frame.columns)]
    return frame / 100.0 if percent else frame


def _var1_path(factors: np.ndarray, steps: int) -> np.ndarray:
    """OLS VAR(1) with a constant on the rows of ``factors``, iterated ``steps`` days from the last row: the forecast of the factors."""
    X = np.column_stack([np.ones(len(factors) - 1), factors[:-1]])
    B, *_ = np.linalg.lstsq(X, factors[1:], rcond=None)
    f = factors[-1].copy()
    for _ in range(steps):
        f = B[0] + f @ B[1:]
    return f


def dynamic_report(panel: pd.DataFrame, lam: float = 0.5, horizon: int = 21, test_days: int = 750, step: int = 5, max_points: int = 400) -> dict:
    """The Diebold-Li dynamic Nelson-Siegel analysis of a panel of daily yield curves (decimals, maturities in years as columns): the daily level, slope and curvature, how persistent each is, the curve
    ``horizon`` days ahead from a VAR(1) on the factors, and a walk-forward check of that forecast: at origins ``step`` days apart over the last ``test_days`` the curve forecast with only the data up to
    the origin is compared with the curve that followed, and with the random walk (tomorrow's curve is today's)."""
    if not 0.05 <= float(lam) <= 6.0:
        raise ValueError("the decay lambda is per year, between 0.05 and 6")
    if not 1 <= int(horizon) <= 250 or not 100 <= int(test_days) <= 5000 or not 1 <= int(step) <= 60:
        raise ValueError("the horizon is 1 to 250 days, the test window 100 to 5,000 days and the step 1 to 60 days")
    if panel.shape[1] < 4:
        raise ValueError("a curve needs at least four maturities")
    if len(panel) < 250:
        raise ValueError("the history needs at least 250 days")
    if len(panel) > 20_000:
        raise ValueError("the history is longer than 20,000 days")
    tau = np.asarray(panel.columns, dtype=float)
    X = nelson_ns_loadings(tau, lam)
    Y = panel.to_numpy(float)
    betas = Y @ np.linalg.pinv(X).T
    fitted = betas @ X.T
    rmse = float(np.sqrt(np.mean((Y - fitted) ** 2)) * 1e4)
    last = len(panel) - 1
    f_hat = _var1_path(betas, int(horizon))
    now_curve = X @ betas[last]
    fcst_curve = X @ f_hat
    first = max(250, len(panel) - int(test_days))
    err_m, err_r, origins = [], [], []
    for t in range(first, len(panel) - int(horizon), int(step)):
        pred = X @ _var1_path(betas[: t + 1], int(horizon))
        actual = Y[t + int(horizon)]
        err_m.append(pred - actual)
        err_r.append(Y[t] - actual)
        origins.append(t)
    walk = None
    if origins:
        em, er = np.asarray(err_m), np.asarray(err_r)
        walk = {"origins": len(origins), "first": str(panel.index[origins[0]].date()), "last": str(panel.index[origins[-1]].date()), "horizon": int(horizon),
                "rmse_bp": {"dynamic_ns": float(np.sqrt(np.mean(em ** 2)) * 1e4), "random_walk": float(np.sqrt(np.mean(er ** 2)) * 1e4)},
                "by_maturity": [{"maturity": float(m), "dynamic_ns": float(np.sqrt(np.mean(em[:, j] ** 2)) * 1e4), "random_walk": float(np.sqrt(np.mean(er[:, j] ** 2)) * 1e4)} for j, m in enumerate(tau)],
                "better_share": float(np.mean(np.abs(em).mean(axis=1) < np.abs(er).mean(axis=1)))}
    persistence = []
    for j, name in enumerate(("level", "slope", "curvature")):
        b = betas[:, j]
        if len(b) > 3 and np.std(b[:-1]) > 0:
            persistence.append({"factor": name, "ar1": float(np.corrcoef(b[:-1], b[1:])[0, 1]), "mean": float(b.mean() * 100), "std": float(b.std(ddof=1) * 100)})
    stride = max(1, len(panel) // int(max_points))
    pick = np.arange(0, len(panel), stride)
    if pick[-1] != last:
        pick = np.append(pick, last)
    return {"lambda": float(lam), "maturities": tau.tolist(), "dates": [str(panel.index[i].date()) for i in pick], "days": int(len(panel)),
            "factors": {"level": (betas[pick, 0] * 100).tolist(), "slope": (-betas[pick, 1] * 100).tolist(), "curvature": (betas[pick, 2] * 100).tolist()},
            "fit_rmse_bp": rmse, "persistence": persistence, "hump_maturity": HUMP / float(lam),
            "latest": {"date": str(panel.index[last].date()), "observed": (Y[last] * 100).tolist(), "fitted": (now_curve * 100).tolist(), "forecast": (fcst_curve * 100).tolist(),
                       "factors": {"level": float(betas[last, 0] * 100), "slope": float(-betas[last, 1] * 100), "curvature": float(betas[last, 2] * 100)},
                       "forecast_factors": {"level": float(f_hat[0] * 100), "slope": float(-f_hat[1] * 100), "curvature": float(f_hat[2] * 100)}},
            "horizon": int(horizon), "walk_forward": walk}


# ------------------------------------------------------------------------------------------------------------------ bond analytics
def cashflow_schedule(maturity: float, coupon: float, freq: int = 2, face: float = 100.0):
    n = int(round(maturity * freq))
    times = np.arange(1, n + 1) / freq
    flows = np.full(n, face * coupon / freq)
    flows[-1] += face
    return times, flows


def bond_price(maturity, coupon, ytm, freq: int = 2, face: float = 100.0) -> float:
    """Clean price (at a coupon date) of a fixed-coupon bond for a yield quoted with ``freq`` compounding."""
    t, cf = cashflow_schedule(maturity, coupon, freq, face)
    return float((cf / (1.0 + ytm / freq) ** (t * freq)).sum())


def bond_price_from_curve(maturity, coupon, zero_curve, freq: int = 2, face: float = 100.0) -> float:
    """Price from a zero curve ``zero_curve(t)`` (continuously compounded zero rates)."""
    t, cf = cashflow_schedule(maturity, coupon, freq, face)
    return float((cf * np.exp(-np.asarray([zero_curve(x) for x in t]) * t)).sum())


def yield_to_maturity(price, maturity, coupon, freq: int = 2, face: float = 100.0) -> float:
    return float(optimize.brentq(lambda y: bond_price(maturity, coupon, y, freq, face) - price, -0.5, 2.0, xtol=1e-14))


def duration(maturity, coupon, ytm, freq: int = 2, kind: str = "modified") -> float:
    """Macaulay duration (PV-weighted average time of the cash flows) or modified duration ``D_mac / (1 + y/freq)`` (the percentage price change per unit yield)."""
    t, cf = cashflow_schedule(maturity, coupon, freq)
    pv = cf / (1.0 + ytm / freq) ** (t * freq)
    mac = float((t * pv).sum() / pv.sum())
    return mac if kind == "macaulay" else mac / (1.0 + ytm / freq)


def convexity(maturity, coupon, ytm, freq: int = 2) -> float:
    t, cf = cashflow_schedule(maturity, coupon, freq)
    n = t * freq
    pv = cf / (1.0 + ytm / freq) ** n
    return float((pv * n * (n + 1)).sum() / (pv.sum() * freq ** 2 * (1.0 + ytm / freq) ** 2))


def dv01(maturity, coupon, ytm, freq: int = 2, face: float = 100.0) -> float:
    """Dollar value of a basis point: the price change for a one-basis-point fall in yield, by full revaluation (so it contains convexity)."""
    return float(bond_price(maturity, coupon, ytm - 1e-4, freq, face) - bond_price(maturity, coupon, ytm, freq, face))


def key_rate_durations(maturity, coupon, zero_curve, key_maturities=(1, 2, 3, 5, 7, 10, 20, 30), bump: float = 1e-4, freq: int = 2) -> pd.Series:
    """Price sensitivity to a triangular bump of each key zero rate (the bump is spread linearly between neighbouring key maturities). The sum is the effective duration."""
    keys = np.asarray(key_maturities, float)
    base = bond_price_from_curve(maturity, coupon, zero_curve, freq)
    out = {}
    for i, k in enumerate(keys):
        lo = keys[i - 1] if i else 0.0
        hi = keys[i + 1] if i + 1 < len(keys) else keys[-1] + 1.0

        def shock(t, lo=lo, k=k, hi=hi):
            return bump * (np.clip((t - lo) / (k - lo), 0, 1) if t <= k else np.clip((hi - t) / (hi - k), 0, 1))

        bumped = bond_price_from_curve(maturity, coupon, lambda t: zero_curve(t) + shock(t), freq)
        out[k] = -(bumped - base) / (base * bump)
    return pd.Series(out, name="krd")


def bootstrap_zero_curve(par_maturities, par_yields, freq: int = 2) -> pd.Series:
    """Zero (spot) rates, continuously compounded, from par-bond yields at semi-annual maturities by forward substitution: each par bond prices at 100, which pins the discount factor
    of its final cash flow given the earlier ones. Maturities must be a multiple of ``1/freq``; intermediate par yields are linearly interpolated."""
    mats = np.asarray(par_maturities, float)
    grid = np.arange(1, int(round(mats.max() * freq)) + 1) / freq
    py = np.interp(grid, mats, np.asarray(par_yields, float))
    df = np.empty(len(grid))
    for i, (t, y) in enumerate(zip(grid, py)):
        c = y / freq
        df[i] = (1.0 - c * df[:i].sum()) / (1.0 + c)
    return pd.Series(-np.log(df) / grid, index=grid, name="zero")


def forward_rate(zero_curve, t1: float, t2: float) -> float:
    """Continuously compounded forward rate between ``t1`` and ``t2``: ``(z2 t2 - z1 t1) / (t2 - t1)``."""
    return float((zero_curve(t2) * t2 - zero_curve(t1) * t1) / (t2 - t1))


def carry_rolldown(maturity: float, coupon: float, par_curve, horizon: float = 0.25, freq: int = 2, financing: float = 0.0) -> dict:
    """Expected return over ``horizon`` years if the PAR curve does not move. ``par_curve(t)`` gives the par yield. Buy at the yield for ``maturity`` (priced at par if ``coupon`` equals
    it), hold, and revalue at the yield for the SHORTER maturity ``maturity - horizon``: ``carry`` is the coupon accrual minus ``financing``, ``rolldown`` the price gain from the
    lower yield. Returns both and the total, in percent of the starting price."""
    y0 = float(par_curve(maturity))
    y1 = float(par_curve(max(maturity - horizon, 1.0 / freq)))
    p0 = bond_price(maturity, coupon, y0, freq)
    p1 = bond_price(max(maturity - horizon, 1.0 / freq), coupon, y1, freq)
    accrual = coupon * horizon * 100.0
    carry_ = (accrual - financing * horizon * p0) / p0
    roll = (p1 - p0) / p0
    return {"carry": float(carry_), "rolldown": float(roll), "total": float(carry_ + roll), "yield_start": y0, "yield_end": y1}
