"""Portfolio-rebalance styles: policy portfolios with rebalancing discipline, the month-end rebalancing flow, flight to quality and a top-down market outlook.

    policy_portfolio       asset allocation: strategic weights, restored on a calendar and/or when an asset drifts outside a tolerance band (the rebalancing rule is the strategy)
    rebalancing_flow       a balanced fund that has seen risky assets beat bonds all month must sell them at the month-end: lean against that predictable flow in the last days of the month
    flight_to_quality      a price-only stress gauge (drawdown of the risky assets, their volatility, an unusually strong bid for bonds) that moves the book from risky assets to quality assets
    market_outlook         top-down: trend, twelve-month momentum, breadth and calm of the risky assets add up to an outlook that tilts the book between risky and defensive assets

The other rebalance styles are provided elsewhere: index reconstitution is ``event_study_drift`` reading an index-change file (``alpha_styles``), market-neutral books are the ``beta_neutral``
allocator (``framework/allocators_overlay``), and every ``ml`` / ``deep`` / ``expression`` model is model-driven.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import in_classes, month_end_flags, zero_except
from .taa import PRESETS, RISKY_CLASSES, SAFE_CLASSES, _TAA

CALENDARS = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12, "never": 0}
HAVENS = ("GLD", "IAU", "GLDM", "SGOL")                                                                       # gold ETFs: classed as commodities, but what investors run to in a flight to quality


def _risky_safe(data) -> tuple[list[str], list[str]]:
    """The risky and the safe assets of a universe by asset class (equity, real estate, commodity and credit against rates and fixed income). Gold ETFs are neither: see ``HAVENS``."""
    risky = [a for a in in_classes(data, *RISKY_CLASSES) if a not in HAVENS]
    return risky, in_classes(data, *SAFE_CLASSES)


def _need_both(name: str, risky: list[str], safe: list[str]) -> None:
    if not risky or not safe:
        raise KeyError(f"{name} needs at least one risky asset (an asset of class equity, real_estate, commodity or credit) and one safe asset (an asset of class rates or fixed_income); label your tickers' classes")


def _basket_return(returns: pd.DataFrame, members: list[str], investable: pd.DataFrame) -> pd.Series:
    """The equal-weight daily return of ``members`` (the ones investable that day); NaN when none is."""
    return returns[members].where(investable[members]).mean(axis=1)


def _expanding_z(x: pd.Series, min_periods: int) -> pd.Series:
    return ((x - x.expanding(min_periods).mean()) / x.expanding(min_periods).std()).clip(-4.0, 4.0)


# ------------------------------------------------------------------------------------------------------------------------------- asset allocation
@register_model("policy_portfolio", "allocation", "Policy portfolio with rebalancing discipline: strategic weights restored on a calendar (monthly to annual) and/or whenever an asset drifts outside a tolerance band")
class PolicyPortfolio(_TAA):
    """Most of a strategic allocation's behaviour comes from how it is rebalanced. Between rebalances the weights drift with returns (winners grow); restoring them every month sells winners and
    buys losers (a mild contrarian bet that earns when assets mean-revert and loses when they trend), while never rebalancing lets the riskiest asset take over. A tolerance band restores the policy
    only when it is needed (Leland 1999: under proportional costs the optimal rule is a no-trade region around the target). ``preset`` picks a mix from ``model_portfolio`` (60_40, permanent,
    all_weather, bogleheads) or ``targets`` gives your own ``{ticker: weight}``. ``calendar`` is how often the policy is restored whatever the drift (``never`` for the band alone); ``band`` is the
    drift in weight (0.05 = five points) that restores it. Decisions are taken on the last trading day of each month, and the weights stated when nothing is restored are the drifted ones, so the
    engine's own drift and trading cost agree with the rule. Run it with the engine's monthly rebalance."""

    name = "policy_portfolio"

    def __init__(self, preset: str = "60_40", targets: dict | None = None, calendar: str = "quarterly", band: float = 0.05):
        if targets is None and preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)} or targets must be given")
        if calendar not in CALENDARS:
            raise ValueError(f"calendar must be one of {sorted(CALENDARS)}")
        if band <= 0:
            raise ValueError("band must be positive (use a large band for 'calendar only')")
        if targets is not None and (not targets or any(float(w) < 0 for w in targets.values()) or sum(float(w) for w in targets.values()) <= 0):
            raise ValueError("targets must be non-negative weights that do not all vanish")
        self.preset, self.targets, self.calendar, self.band = preset, dict(targets) if targets else None, calendar, band

    def policy(self, data) -> pd.Series:
        """The strategic weights over the universe (tickers; for a preset, the bucket's ticker or else its asset class), summing to one."""
        out = pd.Series(0.0, index=data.assets)
        if self.targets is not None:
            for ticker, w in self.targets.items():
                if ticker in out.index:
                    out[ticker] = float(w)
        else:
            for weight, tickers, classes in PRESETS[self.preset]:
                members = [a for a in tickers if a in data.assets] or in_classes(data, *classes)
                if members:
                    out[members] += weight / len(members)
        total = out.sum()
        return out / total if total > 0 else out

    def decide(self, data, prices, investable):
        policy = self.policy(data).reindex(prices.columns).fillna(0.0)
        if policy.sum() <= 0:
            return None
        growth = (prices / prices.shift(1)).fillna(1.0)                                                         # the book's asset-by-asset growth from one month-end to the next
        every = CALENDARS[self.calendar]
        held, last_restore, rows = None, 0, {}
        for k, date in enumerate(prices.index):
            live = investable.loc[date] & prices.loc[date].notna()
            target = policy.where(live, 0.0)
            if target.sum() <= 0:
                continue
            target = target / target.sum()
            if held is None:
                held, last_restore = target.copy(), k
            else:
                grown = (held * growth.loc[date]).where(live, 0.0)
                held = grown / grown.sum() if grown.sum() > 0 else target.copy()
                due = every > 0 and k - last_restore >= every
                if due or bool(((held - target).abs() > self.band).any()):
                    held, last_restore = target.copy(), k
            rows[date] = held.copy()
        return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------------------------------- the month-end flow
@register_model("rebalancing_flow", "seasonal", "Month-end rebalancing flow: when risky assets beat safe ones over the month, balanced funds must sell risky assets at the month-end, so lean the other way in its last days")
class RebalancingFlow(ForecastModel):
    """A fund that targets a mix (60/40 is the archetype) finds its risky share too high after risky assets outperform and sells them near the month-end, and the reverse after they underperform:
    the size of the flow is the drift in the mix times the fund's assets, and the drift is visible to everyone from prices. Harvey, Mazzoleni and Melone (NBER 33554, 2025) find that when pension funds are overweight
    stocks, equity returns fall by about 17 basis points over the next day, and that the pressure fades within about two weeks; funds that rebalance on a calendar do so at month- and quarter-ends, which is why this strategy looks at the last days of the month. It measures
    the relative return of the equal-weight risky basket over the safe basket since the last month-end, in units of its own history of months, and in the last days of the month holds the opposite
    of the expected flow: short the risky assets and long the safe ones after a strong month, the reverse after a weak one. It is flat the rest of the month. A position stamped on day s earns the
    return of day s+2 under the engine's one-day signal lag, so ``lead`` (default 2) places the stamps so that the book earns the last ``days`` trading days of the month. Whether the flow is large
    enough to matter after costs is an empirical question: run it before believing it, and count it as a trial."""

    name, family, position_mode = "rebalancing_flow", "seasonal", "time_series"
    book, rebalance, horizon = "sleeves", "daily", 3
    min_assets = 2                                                                                             # a risky and a safe asset

    def __init__(self, days: int = 3, lead: int = 2, scale: float = 1.0, min_months: int = 36, safe_leg: bool = True):
        if not 1 <= days <= 10 or not 0 <= lead <= 5 or scale <= 0 or min_months < 12:
            raise ValueError("1 <= days <= 10, 0 <= lead <= 5, scale > 0, min_months >= 12")
        self.days, self.lead, self.scale, self.min_months, self.safe_leg = days, lead, scale, min_months, bool(safe_leg)

    def score(self, data):
        risky, safe = _risky_safe(data)
        _need_both("rebalancing_flow", risky, safe)
        idx = data.index
        gap = (_basket_return(data.returns, risky, data.investable) - _basket_return(data.returns, safe, data.investable)).fillna(0.0)
        month_id = pd.Series(idx.year * 12 + idx.month, index=idx)
        run = gap.groupby(month_id).cumsum()                                                                    # the month's relative return so far, known at each close
        finished = run[month_end_flags(idx)]                                                                    # one value per completed month
        prior = finished.expanding(self.min_months).std().shift(1)                                              # the spread of EARLIER months only
        scale = month_id.map(pd.Series(prior.to_numpy(), index=finished.index.year * 12 + finished.index.month))
        z = (run / scale).clip(-3.0, 3.0)
        left = month_id.groupby(month_id).cumcount(ascending=False) + 1                                         # trading days left in the month, today included: the calendar is public in advance
        left[month_id == month_id.iloc[-1]] = 10 ** 6                                                           # the last month of the data may be unfinished: its end is unknown
        in_window = (left > self.lead) & (left <= self.lead + self.days)
        flow = (-np.tanh(z / (2.0 * self.scale))).where(in_window, 0.0)                                         # the expected flow is the opposite of the drift
        values = {a: flow for a in risky}
        if self.safe_leg:
            values.update({a: -flow for a in safe})
        return zero_except(idx, data.assets, values).where(data.investable)


# ------------------------------------------------------------------------------------------------------------------------------- flight to quality
def stress_gauge(data, high_window: int = 126, vol_window: int = 21, corr_window: int = 63, min_periods: int = 252) -> pd.DataFrame:
    """Components of a price-only flight-to-quality gauge, all causal: ``drawdown`` of the risky basket from its trailing high (<= 0), ``vol`` (the z-score of its short volatility against its own
    history), ``bid`` (the z-score of minus the trailing correlation of risky and safe returns: positive when bonds are rising as stocks fall MORE than usual) and ``stress`` (the average of their
    squashed positive parts, in [0, 1]). A persistently negative stock-bond correlation is a feature of whole decades, not of stress, which is why the bid is measured against its own history."""
    risky, safe = _risky_safe(data)
    _need_both("flight_to_quality", risky, safe)
    r_risky, r_safe = _basket_return(data.returns, risky, data.investable), _basket_return(data.returns, safe, data.investable)
    nav = (1.0 + r_risky.fillna(0.0)).cumprod()
    drawdown = nav / nav.rolling(high_window, min_periods=20).max() - 1.0
    vol = _expanding_z(r_risky.rolling(vol_window, min_periods=vol_window // 2).std(), min_periods)
    bid = _expanding_z(-r_risky.rolling(corr_window, min_periods=corr_window // 2).corr(r_safe), min_periods)
    parts = pd.concat([np.tanh(-drawdown / 0.10), np.tanh(vol.clip(lower=0.0) / 2.0), np.tanh(bid.clip(lower=0.0) / 2.0)], axis=1)
    return pd.DataFrame({"drawdown": drawdown, "vol": vol, "bid": bid, "stress": parts.mean(axis=1).where(parts.notna().all(axis=1))})


@register_model("flight_to_quality", "macro", "Flight to quality: a price-only stress gauge (drawdown of the risky assets, their volatility, an unusually strong bid for bonds) moves the book from risky assets to rates and gold when stress is high")
class FlightToQuality(ForecastModel):
    """Investors run from risky to safe assets together: Baele, Bekaert, Inghelbrecht and Wei (2020) find flights to safety on the rare days when bonds beat stocks by a wide margin, with money moving from equity funds into government bond and money market funds. Three symptoms measured
    from prices alone are averaged into a stress level in [0, 1]: the drawdown of the equal-weight risky basket from its half-year high, its volatility against its own history, and how unusually
    negative the stock-bond correlation is. In calm markets the book is long the risky assets; as stress passes ``threshold`` it moves to rates, fixed income and gold (gold ETFs count as havens), and
    it shorts the risky assets unless ``short_risky`` is off. It is the version of ``risk_on_off`` that needs only the prices you already have."""

    name, family, position_mode = "flight_to_quality", "macro", "time_series"
    book, rebalance, horizon = "sleeves", "daily", 10
    min_assets = 2

    def __init__(self, threshold: float = 0.5, high_window: int = 126, vol_window: int = 21, corr_window: int = 63, short_risky: bool = True):
        if not 0 < threshold < 1 or high_window < 20 or vol_window < 5 or corr_window < 20:
            raise ValueError("0 < threshold < 1, high_window >= 20, vol_window >= 5, corr_window >= 20")
        self.threshold, self.high_window, self.vol_window, self.corr_window, self.short_risky = threshold, high_window, vol_window, corr_window, bool(short_risky)

    def score(self, data):
        s = stress_gauge(data, self.high_window, self.vol_window, self.corr_window)["stress"]
        flight = ((s - self.threshold) / (1.0 - self.threshold)).clip(0.0, 1.0)                                  # 0 below the threshold, 1 at full stress
        calm = (1.0 - s / self.threshold).clip(0.0, 1.0)                                                         # 1 when perfectly calm, 0 at the threshold
        risky, safe = _risky_safe(data)
        values = {a: calm - (flight if self.short_risky else 0.0) for a in risky}
        values.update({a: flight for a in safe})
        values.update({a: 0.5 * flight for a in HAVENS if a in data.assets})
        out = zero_except(data.index, data.assets, values).where(data.investable)
        out.loc[s.isna()] = np.nan                                                                             # no view until the gauge has its history
        return out


# ------------------------------------------------------------------------------------------------------------------------------- market outlook
@register_model("market_outlook", "allocation", "Top-down market outlook: trend, twelve-month momentum, breadth and calm of the risky assets add up to an outlook in [-1, 1] that tilts the book between risky and defensive assets")
class MarketOutlook(ForecastModel):
    """A manager's market view, made mechanical. Four readings of the equal-weight risky basket are each squashed to [-1, 1] and averaged: how far its price is from its ``trend``-day average (in units of
    its own monthly volatility), its ``momentum``-day return, the share of risky assets above their own averages (breadth: a rally carried by few assets is fragile) and the calm of its volatility against
    its own history. The outlook times ``tilt`` is the score of every risky asset (long when bullish, short when bearish) and, when it is negative, the score of every safe asset (bearish views move
    into safety; a bullish view does not short bonds, which would pay their carry for nothing). ``bias`` adds a constant opinion of your own (positive is bullish) to the computed outlook."""

    name, family, position_mode = "market_outlook", "allocation", "time_series"
    book, rebalance, horizon = "sleeves", "weekly", 21
    min_assets = 2

    def __init__(self, trend: int = 200, momentum: int = 252, tilt: float = 1.0, bias: float = 0.0, min_periods: int = 252):
        if trend < 20 or momentum < 21 or tilt <= 0 or not -1.0 <= bias <= 1.0 or min_periods < 60:
            raise ValueError("trend >= 20, momentum >= 21, tilt > 0, -1 <= bias <= 1, min_periods >= 60")
        self.trend, self.momentum, self.tilt, self.bias, self.min_periods = trend, momentum, tilt, bias, min_periods

    def outlook(self, data) -> pd.Series:
        """The outlook in [-1, 1] on every date (NaN until the readings have their history)."""
        risky, safe = _risky_safe(data)
        _need_both("market_outlook", risky, safe)
        r = _basket_return(data.returns, risky, data.investable)
        nav = (1.0 + r.fillna(0.0)).cumprod()
        sigma = r.rolling(63, min_periods=30).std().replace(0.0, np.nan)
        prices = data.prices[risky]
        breadth = (prices > prices.rolling(self.trend, min_periods=self.trend).mean()).astype(float).where(prices.notna()).mean(axis=1)
        parts = pd.concat([np.tanh((nav / nav.rolling(self.trend, min_periods=self.trend).mean() - 1.0) / (3.0 * sigma * np.sqrt(21.0))),
                           np.tanh((nav / nav.shift(self.momentum) - 1.0) / (3.0 * sigma * np.sqrt(252.0))),
                           2.0 * breadth - 1.0,
                           np.tanh(-_expanding_z(sigma, self.min_periods) / 2.0)], axis=1)
        return parts.mean(axis=1).where(parts.notna().all(axis=1))

    def score(self, data):
        view = ((self.outlook(data) + self.bias).clip(-1.0, 1.0) * self.tilt)
        risky, safe = _risky_safe(data)
        out = zero_except(data.index, data.assets, {a: view for a in risky} | {a: (-view).clip(lower=0.0) for a in safe}).where(data.investable)
        out.loc[view.isna()] = np.nan
        return out
