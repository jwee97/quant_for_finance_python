"""HTTP routes of the Options Strategy Studio (``assets.options_studio``): opportunity scanner, real-contract builder, scenario laboratory, trade validation and the hypothesis ranking. Chains follow the
request's data mode: a simulated chain is refused in observed-only mode."""

from __future__ import annotations

import pandas as pd

from ..assets import options_analytics as oa
from ..assets import options_studio as st
from .api import ApiError, _number


class OptionsStudioApi:
    def __init__(self, api):
        self.api = api

    def get_routes(self) -> dict:
        return {"/api/options/scan": self.scan}

    def post_routes(self) -> dict:
        return {"/api/options/studio/build": self.build, "/api/options/studio/scenarios": self.scenarios, "/api/options/studio/recommend": self.recommend, "/api/options/studio/validate": self.validate}

    @staticmethod
    def _wrap(*args, **kwargs):
        from .asset_api import _wrap

        return _wrap(*args, **kwargs)

    def _chain(self, q: dict):
        underlying, rate, dividend, premium = self.api._option_params(q)
        chain, ctx, source, notes = self.api._option_chain(underlying, rate, dividend, premium)
        return underlying, chain, ctx, source, notes

    def _parts(self, source: str) -> dict:
        return {"prices": "platform" if source == "simulated" else source, "quotes": source}

    def _iv_history(self, underlying: str) -> pd.Series | None:
        from ..assets.option_history import History

        try:
            hist = History(self.api.root)
            dates = hist.dates(underlying)[-250:]
        except (ValueError, OSError):
            return None
        out = {}
        for d in dates:
            try:
                term = oa.term_structure(oa.prepare_chain(hist.load(underlying, d)))
                iv = oa.interpolate_iv(term, 30.0)
            except (ValueError, KeyError):
                continue
            if iv == iv:
                out[pd.Timestamp(d)] = float(iv)
        return pd.Series(out, dtype=float) if out else None

    def scan(self, q: dict) -> dict:
        """``GET /api/options/scan?underlying=SPY``: where the chain looks unusual (implied against its own history and realised volatility, skew dislocations, term-structure kinks, event volatility, calendar
        candidates), each with quote freshness and liquidity."""
        underlying, chain, ctx, source, notes = self._chain(q)
        try:
            out = st.scan(chain, ctx.spot, ctx.prices if ctx.prices is not None and len(ctx.prices) else None, self._iv_history(underlying), int(_number(q, "min_dte", 3, 1, 60, integer=True)))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return self._wrap("options_scan", {"underlying": underlying, **out}, self._parts(source), [], ctx.as_of, notes)

    def _legs(self, body: dict) -> list[dict]:
        raw = body.get("legs")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 12 or not all(isinstance(x, dict) for x in raw):
            raise ApiError("legs must be a list of 1 to 12 contracts: {expiry, strike, right, quantity, limit}")
        out = []
        for x in raw:
            try:
                leg = {"type": str(x.get("type", "option")), "quantity": float(x["quantity"])}
                if leg["type"] == "stock":
                    leg["price"] = None if x.get("price") in (None, "") else float(x["price"])
                else:
                    leg.update({"expiry": str(x["expiry"]), "strike": float(x["strike"]), "right": str(x["right"]).upper()[:1]})
                    if leg["right"] not in ("C", "P"):
                        raise ValueError("right must be C or P")
                if x.get("limit") not in (None, ""):
                    leg["limit"] = float(x["limit"])
                    if leg["limit"] < 0:
                        raise ValueError("a limit price cannot be negative")
            except (KeyError, TypeError, ValueError) as exc:
                raise ApiError(f"leg {len(out) + 1}: {exc}") from None
            out.append(leg)
        return out

    def build(self, body: dict) -> dict:
        """``POST /api/options/studio/build``: a position of real listed contracts priced from the chain's bid and ask and your limits, with its payoff and risk at that entry."""
        underlying, chain, ctx, source, notes = self._chain(body)
        try:
            out = st.build_real(chain, self._legs(body), ctx.spot, commission=_number(body, "commission", 0.65, 0.0, 20.0),
                                horizon_days=None if body.get("horizon_days") in (None, "") else _number(body, "horizon_days", 30, 1, 800, integer=True))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return self._wrap("options_studio_build", {"underlying": underlying, "spot": ctx.spot, **out}, self._parts(source), out["risk_flags"], ctx.as_of,
                          notes + ["Entry prices are the chain's quotes and your limits; the payoff curves are shifted to that entry. Probabilities assume a lognormal spot at the market's own volatility."])

    def scenarios(self, body: dict) -> dict:
        """``POST /api/options/studio/scenarios``: the position's P&L under spot, volatility, skew and time shocks, and its attribution to the underlying, volatility, skew, time, costs and a residual."""
        underlying, chain, ctx, source, notes = self._chain(body)
        try:
            built = st.build_real(chain, self._legs(body), ctx.spot, commission=_number(body, "commission", 0.65, 0.0, 20.0))
            legs = built["analysis"]["legs"]
            raw_scen = body.get("scenarios")
            if raw_scen is not None and (not isinstance(raw_scen, list) or not all(isinstance(x, dict) for x in raw_scen)):
                raise ApiError("scenarios must be a list of {name, spot, vol, skew, days}")
            first = chain[chain["expiry"] == pd.Timestamp(next(l["expiry"] for l in self._legs(body) if l.get("type") != "stock"))]
            out = st.scenario_lab(legs, ctx.spot, float(chain["rate"].median()), float(chain["dividend"].median()), ctx.as_of, built["totals"]["at_limits"], built["totals"]["spread_cost"] + built["totals"]["commission"],
                                  scenarios=raw_scen, forward=float(first["F"].iloc[0]))
        except (ValueError, StopIteration) as exc:
            raise ApiError(str(exc)) from None
        return self._wrap("options_studio_scenarios", {"underlying": underlying, "entry": built["totals"], **out}, self._parts(source), built["risk_flags"], ctx.as_of, notes)

    def recommend(self, body: dict) -> dict:
        """``POST /api/options/studio/recommend``: the structures that fit a stated view (direction, volatility, horizon, maximum loss, margin), ranked by their payoff if the view is right."""
        underlying, chain, ctx, source, notes = self._chain(body)
        direction = str(body.get("direction") or "neutral")
        vol_view = str(body.get("vol_view") or "neutral")
        try:
            out = st.recommend(chain, ctx.spot, direction, _number(body, "move", 0.05, 0.005, 0.5), vol_view, _number(body, "vol_move", 0.03, 0.0, 0.5), int(_number(body, "horizon_days", 30, 3, 400, integer=True)),
                               None if body.get("max_loss") in (None, "") else _number(body, "max_loss", 1000.0, 1.0, 1e9), None if body.get("margin_budget") in (None, "") else _number(body, "margin_budget", 1000.0, 1.0, 1e9),
                               int(_number(body, "quantity", 1, 1, 1000, integer=True)))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return self._wrap("options_studio_recommend", {"underlying": underlying, **out}, self._parts(source), [], ctx.as_of, notes)

    def validate(self, body: dict) -> dict:
        """``POST /api/options/studio/validate``: the checks a trade must pass before it is placed: real contracts, two-sided quotes, fresh quotes, limits that can fill, a bounded loss within your maximum, margin."""
        underlying, chain, ctx, source, notes = self._chain(body)
        try:
            built = st.build_real(chain, self._legs(body), ctx.spot, commission=_number(body, "commission", 0.65, 0.0, 20.0))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        a = built["analysis"]
        max_loss = None if body.get("max_loss") in (None, "") else _number(body, "max_loss", 1000.0, 1.0, 1e9)
        margin_budget = None if body.get("margin_budget") in (None, "") else _number(body, "margin_budget", 1000.0, 1.0, 1e9)
        option_rows = [r for r in built["legs"] if r["type"] == "option"]
        checks = [
            {"id": "contracts", "label": "every contract is listed on this chain", "ok": True, "detail": f"{len(option_rows)} contracts found"},
            {"id": "two_sided", "label": "every option has a bid and an ask", "ok": all(r["bid"] > 0 and r["ask"] > 0 for r in option_rows), "detail": ""},
            {"id": "fresh", "label": "no quote is stale", "ok": not any(r["stale"] for r in option_rows), "detail": f"{sum(1 for r in option_rows if r['stale'])} stale"},
            {"id": "spread", "label": "no spread wider than 10% of its mid", "ok": all((r["spread_pct"] or 0) <= 0.10 for r in option_rows), "detail": "widest " + f"{max((r['spread_pct'] or 0) for r in option_rows):.0%}" if option_rows else ""},
            {"id": "fills", "label": "every limit can fill (marketable)", "ok": built["fill"]["legs_resting"] == 0, "detail": built["fill"]["reading"]},
            {"id": "liquidity", "label": "every contract has open interest", "ok": all(r["open_interest"] > 0 for r in option_rows), "detail": ""},
            {"id": "bounded", "label": "the loss is bounded", "ok": not a["unlimited_loss"], "detail": "unlimited loss" if a["unlimited_loss"] else f"maximum loss {-a['max_loss']:,.0f}"},
        ]
        if max_loss is not None:
            checks.append({"id": "max_loss", "label": f"the maximum loss is within {max_loss:,.0f}", "ok": (not a["unlimited_loss"]) and -a["max_loss"] <= max_loss, "detail": ""})
        if margin_budget is not None:
            checks.append({"id": "margin", "label": f"the indicative margin is within {margin_budget:,.0f}", "ok": (not a["unlimited_loss"]) and a["margin"]["estimate"] <= margin_budget, "detail": f"indicative {a['margin']['estimate']:,.0f}"})
        passed = all(c["ok"] for c in checks)
        return self._wrap("options_studio_validate", {"underlying": underlying, "checks": checks, "passed": passed, "totals": built["totals"], "margin": a["margin"], "max_loss": a["max_loss"], "max_profit": a["max_profit"],
                                                       "reading": "All checks pass: nothing the platform can see stops this order. It is not a statement that the trade is a good one." if passed else "At least one check fails: see the list."},
                          self._parts(source), built["risk_flags"], ctx.as_of, notes)
