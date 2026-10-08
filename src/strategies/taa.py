"""Tactical and strategic asset allocation: portfolio rules that decide weights across asset classes once a month.

Faber's GTAA / Ivy portfolio (2007), Keller and Keuning's Protective, Vigilant and Defensive Asset Allocation (2016-2018), ReSolve's adaptive asset allocation, and the
classic static mixes (60/40, Permanent, All Weather, Bogleheads) as benchmarks. They are structured models: ``weights`` are the strategy, decided on the last trading day of each
month from data through that day, and held until the next one (the engine then applies its one-day lag). Assets the rule does not hold are cash (zero weight).

The defaults use the platform's ETF tickers where they exist and fall back to asset classes (``equity``, ``real_estate``, ``commodity`` and ``credit`` as risky; ``rates`` and
``fixed_income`` as safe) so the rules also run on your own tickers once you label them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ._common import in_classes, month_end_flags, monthly_weights

RISKY_CLASSES = ("equity", "real_estate", "commodity", "credit")
SAFE_CLASSES = ("rates", "fixed_income")


def _pick(data, names, classes) -> list[str]:
    chosen = [a for a in (names or ()) if a in data.assets]
    return chosen or in_classes(data, *classes)


class _TAA(ForecastModel):
    """Month-end rules: subclasses implement ``decide`` on the monthly price panel."""

    family, position_mode, structured = "allocation", "time_series", True

    def month_panel(self, data) -> tuple[pd.DataFrame, pd.DataFrame]:
        flags = month_end_flags(data.index).to_numpy()
        return data.prices.loc[flags], data.investable.loc[flags]

    def decide(self, data, prices: pd.DataFrame, investable: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def weights(self, data):
        prices, investable = self.month_panel(data)
        rows = self.decide(data, prices, investable)
        if rows is None or rows.empty:
            return pd.DataFrame(np.nan, index=data.index, columns=data.assets)
        return monthly_weights(rows.reindex(columns=data.assets).fillna(0.0), data.index)

    def score(self, data):
        return self.weights(data).where(data.investable)


def _momentum_13612w(prices: pd.DataFrame) -> pd.DataFrame:
    r = lambda k: prices / prices.shift(k) - 1.0           # noqa: E731
    return (12 * r(1) + 4 * r(3) + 2 * r(6) + r(12)) / 19.0


@register_model("faber_gtaa", "allocation", "Faber's GTAA / Ivy portfolio: hold each asset class with an equal weight only while its price is above its 10-month average, otherwise hold cash")
class FaberGTAA(_TAA):
    """A simple trend filter on each asset class keeps most of the return of buy-and-hold with far shallower drawdowns (Faber 2007)."""

    name = "faber_gtaa"

    def __init__(self, months: int = 10):
        if months < 3:
            raise ValueError("months must be at least 3")
        self.months = months

    def decide(self, data, prices, investable):
        sma = prices.rolling(self.months, min_periods=self.months).mean()
        valid = sma.notna() & investable
        hold = ((prices > sma) & valid).astype(float)
        weights = hold.div(investable.sum(axis=1).replace(0, np.nan), axis=0)
        return weights.where(valid.any(axis=1)).dropna(how="all")


@register_model("paa", "allocation", "Protective Asset Allocation (Keller-Keuning): hold the top risky assets by momentum, moving a growing fraction into the best safe asset as fewer risky assets trend up")
class ProtectiveAssetAllocation(_TAA):
    """Breadth of trend across many markets is an early warning of a crash: when most risky assets are falling, step aside into bonds (a canary built from the whole risky set)."""

    name = "paa"
    min_assets = 2                                              # risky assets to rank and a safe one to hide in

    def __init__(self, lookback: int = 12, protection: int = 1, top: int = 6, risky: tuple = ("SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "GLD", "DBC", "HYG"),
                 safe: tuple = ("IEF", "SHY", "AGG", "TLT")):
        if lookback < 3 or protection not in (0, 1, 2) or top < 1:
            raise ValueError("lookback >= 3, protection in (0, 1, 2), top >= 1")
        self.lookback, self.protection, self.top, self.risky, self.safe = lookback, protection, top, tuple(risky), tuple(safe)

    def decide(self, data, prices, investable):
        mom = prices / prices.rolling(self.lookback + 1, min_periods=self.lookback + 1).mean() - 1.0
        risky, safe = _pick(data, self.risky, RISKY_CLASSES), _pick(data, self.safe, SAFE_CLASSES)
        rows = {}
        for date in mom.index:
            r = [a for a in risky if investable.loc[date, a] and np.isfinite(mom.loc[date, a])]
            if len(r) < 2:
                continue
            n_total, n_good = len(r), int((mom.loc[date, r] > 0).sum())
            n1 = self.protection * n_total / 4.0
            bond_fraction = 1.0 if n_total <= n1 else min(1.0, (n_total - n_good) / (n_total - n1))
            top = mom.loc[date, r].nlargest(min(self.top, n_total)).index
            w = pd.Series(0.0, index=prices.columns)
            w[top] = (1.0 - bond_fraction) / len(top)
            s = [a for a in safe if investable.loc[date, a] and np.isfinite(mom.loc[date, a])]
            if s and bond_fraction > 0:
                best = mom.loc[date, s].idxmax()
                if mom.loc[date, best] > 0:
                    w[best] += bond_fraction                   # a safe asset that is itself falling is not safe: that fraction stays in cash
            rows[date] = w
        return pd.DataFrame(rows).T


@register_model("vaa", "allocation", "Vigilant Asset Allocation (Keller-Keuning): all-in on the best offensive asset while every offensive asset has positive 13612W momentum, else all-in on the best defensive one")
class VigilantAssetAllocation(_TAA):
    """Requiring ALL offensive assets to trend up makes the switch to safety fast, at the cost of being aggressive and concentrated when it is on."""

    name = "vaa"
    min_assets = 2                                              # an offensive and a defensive asset

    def __init__(self, offensive: tuple = ("SPY", "EFA", "EEM", "AGG"), defensive: tuple = ("LQD", "IEF", "SHY")):
        self.offensive, self.defensive = tuple(offensive), tuple(defensive)

    def decide(self, data, prices, investable):
        mom = _momentum_13612w(prices)
        offensive = _pick(data, self.offensive, RISKY_CLASSES)[:4] if not [a for a in self.offensive if a in data.assets] else [a for a in self.offensive if a in data.assets]
        defensive = _pick(data, self.defensive, SAFE_CLASSES)
        rows = {}
        for date in mom.index:
            off = [a for a in offensive if investable.loc[date, a] and np.isfinite(mom.loc[date, a])]
            dfn = [a for a in defensive if investable.loc[date, a] and np.isfinite(mom.loc[date, a])]
            if not off or len(off) < len(offensive) or not dfn:
                continue
            w = pd.Series(0.0, index=prices.columns)
            w[(mom.loc[date, off].idxmax() if (mom.loc[date, off] > 0).all() else mom.loc[date, dfn].idxmax())] = 1.0
            rows[date] = w
        return pd.DataFrame(rows).T


@register_model("daa", "allocation", "Defensive Asset Allocation (Keller-Keuning): canary assets with negative momentum move the portfolio from the top offensive assets into the best defensive one")
class DefensiveAssetAllocation(_TAA):
    """Two 'canary' markets (emerging equities and aggregate bonds) are the first to signal stress; each bad canary moves a share of the portfolio to safety."""

    name = "daa"
    min_assets = 2                                              # an offensive and a defensive asset (the canaries are among them)

    def __init__(self, canary: tuple = ("EEM", "AGG"), offensive: tuple = ("SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "GLD", "DBC", "HYG"),
                 defensive: tuple = ("IEF", "SHY", "LQD", "TLT"), top: int = 6, breadth: int = 2):
        if top < 1 or breadth < 1:
            raise ValueError("top >= 1 and breadth >= 1")
        self.canary, self.offensive, self.defensive, self.top, self.breadth = tuple(canary), tuple(offensive), tuple(defensive), top, breadth

    def decide(self, data, prices, investable):
        mom = _momentum_13612w(prices)
        offensive, defensive = _pick(data, self.offensive, RISKY_CLASSES), _pick(data, self.defensive, SAFE_CLASSES)
        canary = [a for a in self.canary if a in data.assets] or (offensive[:1] + defensive[:1])
        rows = {}
        for date in mom.index:
            c = [a for a in canary if np.isfinite(mom.loc[date, a])]
            off = [a for a in offensive if investable.loc[date, a] and np.isfinite(mom.loc[date, a])]
            dfn = [a for a in defensive if investable.loc[date, a] and np.isfinite(mom.loc[date, a])]
            if len(c) < len(canary) or not off or not dfn:
                continue
            cash_fraction = min(1.0, int((mom.loc[date, c] < 0).sum()) / self.breadth)
            top = mom.loc[date, off].nlargest(min(self.top, len(off))).index
            w = pd.Series(0.0, index=prices.columns)
            w[top] = (1.0 - cash_fraction) / len(top)
            w[mom.loc[date, dfn].idxmax()] += cash_fraction
            rows[date] = w
        return pd.DataFrame(rows).T


@register_model("adaptive_asset_allocation", "allocation", "Adaptive asset allocation (ReSolve): the top-k assets by 6-month momentum, weighted by inverse volatility (the original minimises variance)")
class AdaptiveAssetAllocation(_TAA):
    """Momentum picks what to own and risk weighting sizes it: strong assets in proportion to how calm they are. Inverse volatility stands in for the original's minimum variance."""

    name = "adaptive_asset_allocation"
    min_assets = 2                                              # it picks the top assets, so it needs something to pick from

    def __init__(self, top_k: int = 5, months: int = 6, vol_window: int = 63):
        if top_k < 1 or months < 1 or vol_window < 10:
            raise ValueError("top_k >= 1, months >= 1, vol_window >= 10")
        self.top_k, self.months, self.vol_window = top_k, months, vol_window

    def decide(self, data, prices, investable):
        mom = prices / prices.shift(self.months) - 1.0
        vol = data.returns.rolling(self.vol_window, min_periods=self.vol_window).std().reindex(prices.index)
        rows = {}
        for date in mom.index:
            ok = [a for a in prices.columns if investable.loc[date, a] and np.isfinite(mom.loc[date, a]) and np.isfinite(vol.loc[date, a]) and vol.loc[date, a] > 0]
            if len(ok) < 2:
                continue
            top = mom.loc[date, ok].nlargest(min(self.top_k, len(ok))).index
            inv = 1.0 / vol.loc[date, top]
            w = pd.Series(0.0, index=prices.columns)
            w[top] = inv / inv.sum()
            rows[date] = w
        return pd.DataFrame(rows).T


# name -> list of (weight, preferred tickers, fall-back asset classes)
PRESETS = {
    "60_40": [(0.60, ("SPY",), ("equity",)), (0.40, ("IEF",), SAFE_CLASSES)],
    "permanent": [(0.25, ("SPY",), ("equity",)), (0.25, ("TLT",), ("rates",)), (0.25, ("GLD",), ("commodity",)), (0.25, ("SHY",), ("rates",))],
    "all_weather": [(0.30, ("SPY",), ("equity",)), (0.40, ("TLT",), ("rates",)), (0.15, ("IEF",), ("rates",)), (0.075, ("GLD",), ("commodity",)), (0.075, ("DBC",), ("commodity",))],
    "bogleheads": [(0.40, ("SPY",), ("equity",)), (0.20, ("EFA",), ("equity",)), (0.40, ("AGG",), SAFE_CLASSES)],
}


@register_model("model_portfolio", "allocation", "Static model portfolios as benchmarks: 60/40, Permanent, All Weather, Bogleheads three-fund, or equal weight across asset classes, rebalanced monthly")
class ModelPortfolio(_TAA):
    """The mixes most investors compare themselves with: any active strategy should be judged against them, not only against cash."""

    name = "model_portfolio"

    def __init__(self, preset: str = "60_40"):
        if preset not in PRESETS and preset != "equal_class":
            raise ValueError(f"preset must be one of {sorted(PRESETS) + ['equal_class']}")
        self.preset = preset

    def buckets(self, data) -> list[tuple[float, list[str]]]:
        if self.preset == "equal_class":
            groups = [in_classes(data, c) for c in sorted({data.asset_class.get(a, "unknown") for a in data.assets} - {"unknown"})]
            groups = [g for g in groups if g]
            return [(1.0 / len(groups), g) for g in groups] if groups else []
        out = []
        for weight, tickers, classes in PRESETS[self.preset]:
            members = [a for a in tickers if a in data.assets] or in_classes(data, *classes)
            if members:
                out.append((weight, members))
        return out

    def decide(self, data, prices, investable):
        buckets = self.buckets(data)
        if not buckets:
            return None
        nominal = sum(w for w, _ in buckets)
        rows = {}
        for date in prices.index:
            live = [(w, [a for a in m if investable.loc[date, a] and np.isfinite(prices.loc[date, a])]) for w, m in buckets]
            live = [(w, m) for w, m in live if m]
            if not live:
                continue
            scale = nominal / sum(w for w, _ in live)
            row = pd.Series(0.0, index=prices.columns)
            for w, members in live:
                row[members] = scale * w / len(members)
            rows[date] = row
        return pd.DataFrame(rows).T
