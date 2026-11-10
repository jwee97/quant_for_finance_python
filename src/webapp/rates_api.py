"""The Rates tab's endpoints: fit a Nelson-Siegel or Svensson curve to a yield curve, and follow the level, slope and curvature of a history of curves.

``GET /api/rates/plan``       the models (what each assumes and where it fails), the units, the limits, and a made-up example curve to try the form with.
``POST /api/rates/curve``     fit one curve: ``text`` (lines of ``maturity yield``) or ``maturities`` and ``yields``, ``model``, ``kind`` (zero rates or par yields), ``lambda`` (the decay, fixed or found).
``POST /api/rates/dynamic``   the Diebold-Li analysis of a history of curves: ``text`` (a CSV of dates by maturities) or ``source: simulated``, ``lambda``, ``horizon`` and the walk-forward window.

Nothing here reads a data provider: the yields are what the person typed or pasted, or a simulated panel that says it is simulated. The computations are in ``assets.rates`` (``quant curve`` calls the same
functions). A fitted curve describes the yields it was given; it is not a forecast, and the dynamic analysis says how its own forecast has done against the random walk.
"""

from __future__ import annotations

import numpy as np

from ..assets import rates
from ..fundamentals.jsonutil import clean
from .api import ApiError, _number

KINDS = {"zero": "Zero (spot) rates", "par": "Par yields of coupon bonds"}
MODEL_INFO = {
    "nelson_siegel": {"label": "Nelson-Siegel", "text": "Three factors and a decay: a level, a slope and a curvature, with the curvature hump at 1.79/λ years.",
                      "assumes": "The curve is smooth with at most one hump or trough.", "fails": "A curve with two humps, or a sharp kink at one maturity, is only approximated."},
    "svensson": {"label": "Svensson", "text": "Nelson-Siegel with a second curvature term and a second decay, so the curve can have two humps.",
                 "assumes": "At most two humps; enough maturities (eight or more) to tell six parameters apart.", "fails": "With few points it fits them exactly and says nothing between them."},
}
EXAMPLE = {"label": "A made-up curve for trying the form (not market data)", "text": "3M 5.10\n6M 4.95\n1Y 4.70\n2Y 4.35\n3Y 4.20\n5Y 4.15\n7Y 4.25\n10Y 4.35\n20Y 4.65\n30Y 4.55"}


def _wrap(error: ValueError) -> ApiError:
    return ApiError(str(error))


class RatesApi:
    def __init__(self, app):
        self.app = app

    def get_routes(self) -> dict:
        return {"/api/rates/plan": self.plan}

    def post_routes(self) -> dict:
        return {"/api/rates/curve": lambda body=None: clean(self.curve(body or {})), "/api/rates/dynamic": lambda body=None: clean(self.dynamic(body or {}))}

    def plan(self, q: dict | None = None) -> dict:
        """``GET /api/rates/plan``: what the forms choose among."""
        return clean({"models": [{"id": k, **v, "min_points": rates.MIN_POINTS[k]} for k, v in MODEL_INFO.items()], "kinds": [{"id": k, "label": v} for k, v in KINDS.items()],
                      "example": EXAMPLE, "limits": {"max_points": rates.MAX_POINTS, "yield_percent": list(rates.YIELD_LIMITS), "maturity_years": list(rates.MATURITY_LIMITS)},
                      "defaults": {"lambda": 0.5, "horizon": 21, "test_days": 750, "step": 5}, "hump": rates.HUMP})

    def curve(self, body: dict) -> dict:
        """``POST /api/rates/curve``: fit a curve to yields typed as lines of ``maturity yield`` (``text``) or given as two lists, and describe it."""
        model = str(body.get("model") or "nelson_siegel")
        kind = str(body.get("kind") or "zero")
        unit = str(body.get("unit") or "percent")
        if unit not in ("percent", "decimal"):
            raise ApiError("unit must be percent (4.25) or decimal (0.0425)")
        try:
            if body.get("text") not in (None, ""):
                tau, y = rates.parse_curve_text(body["text"])
            else:
                if not isinstance(body.get("maturities"), list) or not isinstance(body.get("yields"), list):
                    raise ApiError("give the curve as text (2Y 4.25 on each line) or as two lists, maturities and yields")
                tau = np.array([rates.parse_tenor(m) for m in body["maturities"]])
                y = np.array([float(v) for v in body["yields"]])
            lam = None if body.get("lambda") in (None, "") else _number(body, "lambda", 0.5, 0.05, 6.0)
            return rates.curve_report(tau, y, model, lam, kind, percent=unit == "percent")
        except (ValueError, TypeError) as error:
            raise _wrap(ValueError(str(error))) from None

    def dynamic(self, body: dict) -> dict:
        """``POST /api/rates/dynamic``: the dynamic Nelson-Siegel analysis of a history of curves (``text`` as CSV) or of the simulated one (``source: simulated``)."""
        source = str(body.get("source") or ("text" if body.get("text") else "simulated"))
        if source not in ("text", "simulated"):
            raise ApiError("source must be text (a pasted history) or simulated")
        lam = _number(body, "lambda", 0.5, 0.05, 6.0)
        horizon = _number(body, "horizon", 21, 1, 250, integer=True)
        test_days = _number(body, "test_days", 750, 100, 5000, integer=True)
        step = _number(body, "step", 5, 1, 60, integer=True)
        unit = str(body.get("unit") or "percent")
        if unit not in ("percent", "decimal"):
            raise ApiError("unit must be percent (4.25) or decimal (0.0425)")
        try:
            if source == "simulated":
                seed = _number(body, "seed", 0, 0, 1_000_000, integer=True)
                panel = rates.synthetic_curve_panel(n_days=1500, seed=seed)
                note = "This history is simulated: Nelson-Siegel factors moving as a VAR(1) with one basis point of noise. It shows how the analysis reads; it says nothing about real rates."
            else:
                panel = rates.parse_panel_text(body.get("text"), percent=unit == "percent")
                note = ""
            out = rates.dynamic_report(panel, lam, horizon, test_days, step)
        except ValueError as error:
            raise _wrap(error) from None
        out.update(source=source, note=note)
        return out
