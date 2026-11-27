"""Adapters that put optimisers already in ``src/portfolio`` behind the ML allocation interface.

Each adapter declares what it needs and what it can do, so that a request it cannot honour is refused before anything runs and a result is never described as more than it is:

``uses_forecast``   whether the calibrated expected returns reach the optimiser at all. HERC, NCO (minimum variance) and the Wasserstein-robust CVaR book are *risk-only*: they read a covariance (or scenarios) and
                    nothing else, and respond to the ML forecast not at all. NCO's maximum-Sharpe objective and the near-optimal centre do read the expected returns.
``min_assets``      fewer assets than this and the method degenerates (HERC and HRP are inverse variance for two assets; NCO needs clusters).
``long_short``      every adapter here builds long-only, fully invested books. A long-short or cash-holding request is refused for them, not approximated.
``native``          the limits the optimiser honours itself. None of the hierarchical methods has a weight cap; the near-optimal centre and the Wasserstein CVaR take a box.
``requires``        the inputs it reads; ``window`` the history it needs.
``projection``      every limit the optimiser cannot honour itself (position caps and floors, sector caps) is enforced afterwards by the nearest feasible book (the Euclidean projection ``portfolio_alloc.project``)
                    and the size of that move is reported with the decision (``projection_l1``). A move is a modification of the optimiser's answer, so it is shown, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Adapter:
    key: str
    label: str
    objective: str
    uses_forecast: bool
    min_assets: int
    requires: tuple
    window: str
    native: tuple                    # limits the optimiser itself honours: any of box, sum
    failure: str
    numerical: str
    assumptions: str


ADAPTERS: dict[str, Adapter] = {a.key: a for a in (
    Adapter("herc", "Hierarchical equal risk contribution (Raffinot 2018)", "ward clusters of the correlation distance, equal risk contribution within a cluster, inverse risk between clusters", False, 3, ("covariance",),
            "the covariance window (cov_min_obs..cov_window)", (), "refused for fewer than three assets (it is HRP for two); a degenerate tree gives the HRP book",
            "the number of clusters is chosen by silhouette and can change between dates, which changes the book discontinuously; ERC inside a cluster needs a positive-definite block",
            "correlation carries the structure; the ML forecast is NOT used: HERC is a risk-only allocator and does not respond to it"),
    Adapter("nco", "Nested clustered optimisation, minimum variance (Lopez de Prado 2019)", "minimum variance within correlation clusters, then between them", False, 4, ("covariance",),
            "the covariance window", (), "refused for fewer than four assets (three or fewer is a single-stage minimum variance); a failed stage falls back to equal weight inside it",
            "cluster count by silhouette can change between dates; long-only quadratic programmes per stage",
            "the ML forecast is NOT used: this objective is risk-only"),
    Adapter("nco_max_sharpe", "Nested clustered optimisation, maximum Sharpe", "maximum Sharpe within clusters and between them, long-only, with the calibrated forecast as the expected return", True, 4, ("covariance", "forecast"),
            "the covariance window and a calibrated forecast", (), "with no asset forecast to be positive the stage is minimum variance; refused for fewer than four assets",
            "maximum Sharpe is highly sensitive to the forecast: it is exactly as stable as the calibration is, which is why the stabilisers (blend, shrink, evidence) apply to it",
            "the calibrated expected return IS used, per stage, long-only"),
    Adapter("near_optimal", "Near-optimal centre (Lopez de Prado, Lipton and Zoonekynd style)", "the centre of the set of books within 10% of the optimal utility advantage over equal weight, so that a small change in the forecast does not move it far", True, 2,
            ("covariance", "forecast"), "the covariance window and a calibrated forecast", ("box", "sum"), "falls back to the optimum when the barrier solve does not converge; refused when the bounds leave no fully invested book",
            "a barrier method on a small problem; the tolerance is fixed at 10% of the utility gap",
            "the calibrated expected return IS used (with the configured risk aversion); the centring makes the book less concentrated than the optimum"),
    Adapter("wasserstein_cvar", "Wasserstein-robust CVaR", "minimise the worst-case CVaR over a Wasserstein ball around the past window's de-meaned scenarios of window returns", False, 2, ("scenarios",),
            "at least 60 overlapping window-return scenarios (an unweighted past window)", ("box", "sum"), "fewer than 60 scenarios, or a failed linear programme, is reported as a failed decision",
            "overlapping scenarios are not independent; the ball radius is 10% of the mean scenario standard deviation (an assumption, not a calibrated value)",
            "the ML forecast is NOT used: this is a risk-only tail allocator"),
)}
FORECAST_ADAPTERS = {k for k, a in ADAPTERS.items() if a.uses_forecast}
NEEDS_SCENARIOS = {"wasserstein_cvar"}


def docs() -> dict:
    return {k: {"label": a.label, "objective": a.objective, "assumptions": a.assumptions, "needs": f"{', '.join(a.requires)}; window: {a.window}", "failure": f"{a.failure}. {a.numerical}", "uses_forecast": a.uses_forecast,
                "long_short": False, "min_assets": a.min_assets, "native_constraints": list(a.native), "projection": "limits the optimiser cannot honour are enforced by the nearest feasible book and the move is reported"}
            for k, a in ADAPTERS.items()}


def raw_book(key: str, mu: pd.Series, cov_w: pd.DataFrame, scenarios: np.ndarray | None, risk_aversion: float, lb: float, ub: float, names: list) -> pd.Series:
    """The optimiser's own long-only, fully invested answer, before any limit it does not honour itself is applied. Raises ValueError with the reason when it cannot answer."""
    n = len(names)
    if key == "herc":
        from ..portfolio.hierarchical import herc_weights

        return herc_weights(cov_w).reindex(names)
    if key == "nco":
        from ..portfolio.nco import nco_weights

        return nco_weights(cov_w, None, "min_variance").reindex(names)
    if key == "nco_max_sharpe":
        from ..portfolio.nco import nco_weights

        return nco_weights(cov_w, mu, "max_sharpe").reindex(names)
    if key == "near_optimal":
        from ..portfolio.near_optimal import near_optimal_center

        out = near_optimal_center(mu, cov_w, risk_aversion=risk_aversion, tolerance=0.1, lb=lb, ub=ub)
        return out["weights"].reindex(names)
    if key == "wasserstein_cvar":
        from ..portfolio.robust_wasserstein import wasserstein_cvar_weights

        if scenarios is None or len(scenarios) < 60:
            raise ValueError(f"the Wasserstein-robust CVaR needs at least 60 scenarios of window returns; {0 if scenarios is None else len(scenarios)} available")
        eps = 0.1 * float(np.mean(np.std(scenarios, axis=0, ddof=1)))
        out = wasserstein_cvar_weights(pd.DataFrame(scenarios, columns=names), eps=eps, alpha=0.95, lb=lb, ub=ub)
        if not out["ok"]:
            raise ValueError(f"the robust CVaR programme failed: {out['message']}")
        return out["weights"].reindex(names)
    raise ValueError(f"unknown adapter '{key}'")
