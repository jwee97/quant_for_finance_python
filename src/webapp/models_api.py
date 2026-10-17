"""The Options tab's model endpoints: fit the option-pricing models of ``src/derivatives`` to a chain, price a product under every fitted model, and cost the hedging of an option.

``GET /api/models/plan``     the models (what each adds and where it fails), the products and the choices the three forms offer.
``POST /api/models/fit``     (a job) calibrate the chosen models to the out-of-the-money quotes of a chain: the parameters with their standard errors and the edges they hit, the error against the best flat
                             volatility, the smile each model draws beside the market's, the density of the price at expiry each implies and its odds of a fall.
``POST /api/models/price``   (a job) one product (a European, digital, barrier, average-price or American option) under every fitted model and under Black-Scholes. On a vanilla the models differ by how
                             well each fits the quotes; on another product also by what the quotes do not determine: the spread between them, set beside their spread on the vanilla, is its model risk.
``POST /api/models/hedge``   (a job) what it costs to hedge an option when every trade costs money: Leland's price at several rebalancing frequencies, the Whalley-Wilmott no-trade band, a simulated
                             comparison of hedging rules and the utility-based bid and ask of a binomial market.

The computations are in ``assets.option_models`` (the ``quant models`` command calls the same functions); this module resolves the chain from the Options tab's underlying, keeps the fits for the session
(keyed by the chain's content and the settings, so pricing and the density need not repeat half a minute of fitting), runs the work as a job and labels where every number comes from. A calibration is a
*risk-neutral* description of the quotes. When the chain is the platform's simulated one the models are fitted to another model: the payload says so.
"""

from __future__ import annotations

from ..assets import option_models as om
from ..derivatives import levy
from .api import ApiError, _number
from .asset_api import _wrap


class ModelsApi:
    def __init__(self, app, assets):
        self.app = app
        self.assets = assets
        self._cache = om.FitCache()

    def get_routes(self) -> dict:
        return {"/api/models/plan": self.plan}

    def post_routes(self) -> dict:
        return {"/api/models/fit": self.fit, "/api/models/price": self.price, "/api/models/hedge": self.hedge}

    def plan(self, q: dict | None = None) -> dict:
        """``GET /api/models/plan``: what the three forms choose among."""
        return om.plan()

    # ------------------------------------------------------------------------------------------------------------ the market
    def _market(self, body: dict) -> dict:
        underlying, rate, dividend, premium = self.assets._option_params(body)
        chain, ctx, source, notes = self.assets._option_chain(underlying, rate, dividend, premium)
        return {"underlying": underlying, "chain": chain, "ctx": ctx, "source": source, "notes": notes}

    @staticmethod
    def _settings(body: dict) -> dict:
        limits, defaults = om.LIMITS, om.DEFAULTS
        return {"max_expiries": _number(body, "max_expiries", defaults["max_expiries"], *limits["max_expiries"], integer=True), "min_dte": _number(body, "min_dte", defaults["min_dte"], *limits["min_dte"], integer=True),
                "window": _number(body, "window", defaults["window"], *limits["window"])}

    @staticmethod
    def _ids(body: dict) -> list[str]:
        try:
            return om.check_models(body.get("models"))
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    @staticmethod
    def _quotes(mkt: dict, settings: dict):
        try:
            return om.select_quotes(mkt["chain"], **settings)
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    @staticmethod
    def _spec(body: dict, mkt: dict) -> dict:
        try:
            return om.product_spec(mkt["chain"], body.get("product") or "vanilla", expiry=body.get("expiry"), dte=body.get("dte", 90), strike_pct=body.get("strike_pct", 100.0), call=body.get("call"),
                                   barrier_kind=body.get("barrier_kind") or "down-out", barrier_pct=body.get("barrier_pct") if body.get("barrier_pct") != "" else None, fixings=body.get("fixings", 12),
                                   paths=body.get("paths", om.DEFAULTS["paths"]), seed=body.get("seed", 0))
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    def _calibrate(self, mkt: dict, settings: dict, quotes, ids: list[str], progress):
        """The calibrations of ``ids`` and the models whose search failed: from the session's cache where this chain's content and these settings were fitted already."""
        key = (om.fingerprint(mkt["chain"]), settings["max_expiries"], settings["min_dte"], round(settings["window"], 4))
        results, failed = self._cache.calibrate(quotes, ids, key, progress)
        if not results:
            raise ApiError("no model could be fitted to this chain: " + "; ".join(f"{levy.MODELS[m].label}: {why}" for m, why in failed.items()))
        return results, failed

    # ------------------------------------------------------------------------------------------------------------ fit
    def fit(self, body: dict) -> dict:
        """``POST /api/models/fit`` (a job). ``body``: the chain's ``underlying`` (and ``rate``, ``dividend``, ``premium`` as the other Options endpoints take them), ``models`` (default all five),
        ``max_expiries``, ``min_dte`` and ``window`` (which quotes: out of the money, within ``window`` of the forward in log strike, from expiries at least ``min_dte`` days out). Returns ``{"job": id}``."""
        ids, settings, mkt = self._ids(body), self._settings(body), self._market(body)
        quotes = self._quotes(mkt, settings)

        def work(progress) -> dict:
            results, failed = self._calibrate(mkt, settings, quotes, ids, progress)
            progress("drawing the smiles and the implied densities")
            data, warnings, notes = om.fit_report(mkt["underlying"], quotes, results, failed, settings, mkt["source"] == "simulated")
            return _wrap("option_model_fit", data, {"quotes": mkt["source"], "fitted_models": "simulated"}, warnings, mkt["ctx"].as_of, notes + list(mkt["notes"]))

        return self.app.submit_task(f"option models: fit {', '.join(ids)} to {mkt['underlying']}", work)

    # ------------------------------------------------------------------------------------------------------------ price
    def price(self, body: dict) -> dict:
        """``POST /api/models/price`` (a job). ``body``: the chain (``underlying`` ...), ``product`` (``vanilla``, ``digital``, ``barrier``, ``asian`` or ``american``), ``call``, ``expiry`` (or ``dte``: the listed expiry
        nearest), ``strike_pct`` (the strike as a per cent of the price), ``models`` (default all five), ``barrier_kind`` and ``barrier_pct``, ``fixings``, ``paths`` and ``seed`` for the simulated products,
        and the quote settings of ``fit``. The models are fitted first if this chain and these settings have not been fitted in this session. Returns ``{"job": id}``."""
        ids, settings, mkt = self._ids(body), self._settings(body), self._market(body)
        spec = self._spec(body, mkt)
        quotes = self._quotes(mkt, settings)

        def work(progress) -> dict:
            results, failed = self._calibrate(mkt, settings, quotes, ids, progress)
            data, warnings, notes = om.price_report(mkt["underlying"], mkt["chain"], spec, quotes, results, failed, mkt["source"] == "simulated", progress)
            return _wrap("option_model_price", data, {"quotes": mkt["source"], "prices": "simulated"}, warnings, mkt["ctx"].as_of, notes + list(mkt["notes"]))

        label = {"vanilla": "European", "digital": "digital", "barrier": "barrier", "asian": "average-price", "american": "American"}[spec["product"]]
        return self.app.submit_task(f"option models: {label} {'call' if spec['call'] else 'put'} on {mkt['underlying']} under {len(ids)} model{'s' if len(ids) != 1 else ''}", work)

    # ------------------------------------------------------------------------------------------------------------ hedge
    def hedge(self, body: dict) -> dict:
        """``POST /api/models/hedge`` (a job). ``body``: the chain (``underlying`` ...) for the price, the rate and the dividend yield, ``strike_pct``, ``expiry`` (or ``dte``), ``call``, ``vol`` (a per cent;
        default the at-the-money volatility of that expiry), ``cost_bp`` (the proportional cost of one trade, in basis points of the value traded), ``steps`` (rebalancing dates, default daily),
        ``risk_aversion`` (in units of the option's price: a loss of its whole premium is worth this many units of utility), ``paths`` and ``seed``. Returns ``{"job": id}``."""
        mkt = self._market(body)
        spec = self._spec({"product": "vanilla", "expiry": body.get("expiry"), "dte": body.get("dte", 90), "strike_pct": body.get("strike_pct", 100.0), "call": body.get("call")}, mkt)
        try:
            inputs = om.hedge_inputs(mkt["chain"], spec, body.get("vol"), body.get("cost_bp", om.DEFAULTS["cost_bp"]), body.get("steps"), body.get("risk_aversion", om.DEFAULTS["risk_aversion"]),
                                     body.get("hedge_paths", body.get("paths", om.DEFAULTS["hedge_paths"])), body.get("seed", 0))
        except ValueError as exc:
            raise ApiError(str(exc)) from None

        def work(progress) -> dict:
            data, warnings, notes = om.hedge_report(mkt["underlying"], spec, inputs, progress)
            return _wrap("option_model_hedge", data, {"underlying_price": mkt["source"], "simulation": "simulated"}, warnings, mkt["ctx"].as_of, notes + list(mkt["notes"]))

        return self.app.submit_task(f"option models: hedging a {'call' if spec['call'] else 'put'} on {mkt['underlying']} at {inputs['cost'] * 1e4:.1f} bp a trade", work)
