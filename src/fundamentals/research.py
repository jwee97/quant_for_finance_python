"""The research context: one object that holds everything a factor study reads, all of it as it was known on each rebalance date.

It is built from the local database only (never a provider) for a **universe** (countries, regions, sectors, markets, a minimum size and liquidity) and a **rebalance grid** (weekly, monthly or
quarterly), and offers the panels the factor engine needs as dates-by-``security_id`` tables:

* statement figures and derived metrics *as known on the date* (the point-in-time tables of :mod:`pit`: a figure appears on the first rebalance date on or after the day it was public);
* forward returns over any horizon, entered a day after the signal date (``entry_lag``) so that a signal is never traded at the very close it was computed from; a company that delists before the
  horizon ends is carried to its last price, so the losers stay in the sample;
* backward returns (momentum), volatility and beta;
* market capitalisation and enterprise value, from the unadjusted close times the share count known on the date, converted between the reporting and the trading currency with the exchange rate of
  the date (ratios of one company's own figures are never converted; only an amount set against a price is);
* the investable mask: priced on the date, large enough, liquid enough.

``survivorship()`` says honestly what the universe is: it counts the companies in the database that are no longer listed and never claims the sample is free of survivorship bias unless the user has
said the securities are the historical constituents.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import metrics
from .errors import FundamentalsError
from .pit import PeriodTables, PitSettings
from .schema import DEVELOPED
from .store import FundamentalsStore

HORIZONS = {"1d": 1, "1w": 5, "2w": 10, "1m": 21, "2m": 42, "3m": 63, "6m": 126, "9m": 189, "12m": 252}
FREQ_DAYS = {"W": 5, "M": 21, "Q": 63}
GROUPS = ("sector", "industry", "country", "region", "market")


@dataclass(frozen=True)
class UniverseSpec:
    """Which companies a study is about. Empty tuples mean "no restriction"."""

    name: str = "all"
    countries: tuple = ()
    regions: tuple = ()
    sectors: tuple = ()
    markets: tuple = ()
    developed_only: bool = False
    min_market_cap_usd: float = 0.0
    min_dollar_volume_usd: float = 0.0
    include_inactive: bool = True              # keep delisted companies: dropping them is how survivorship bias gets in

    def describe(self) -> dict:
        return asdict(self)

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:12]

    @staticmethod
    def from_dict(d: dict | None) -> "UniverseSpec":
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in UniverseSpec.__dataclass_fields__}
        for k in ("countries", "regions", "sectors", "markets"):
            if k in known:
                known[k] = tuple(known[k])
        return UniverseSpec(**known)


def horizon_days(horizon) -> int:
    if isinstance(horizon, str):
        if horizon not in HORIZONS:
            raise FundamentalsError(f"horizon must be one of {sorted(HORIZONS, key=HORIZONS.get)} or a number of trading days")
        return HORIZONS[horizon]
    return int(horizon)


class ResearchContext:
    def __init__(self, store: FundamentalsStore, spec: UniverseSpec | None = None, start: str | None = None, end: str | None = None, freq: str = "M", entry_lag: int = 1,
                 settings: PitSettings | None = None, security_ids: list[int] | None = None):
        if freq not in FREQ_DAYS:
            raise FundamentalsError("freq must be W (weekly), M (monthly) or Q (quarterly)")
        self.store, self.spec, self.freq, self.entry_lag = store, spec or UniverseSpec(), freq, int(entry_lag)
        self.settings = settings or PitSettings()
        self._cache: dict = {}
        info = self._select(security_ids)
        adj = store.prices(list(info["security_id"]), start, end)
        if adj.empty:
            raise FundamentalsError("the database holds no prices for this universe: load prices first (the Data Manager can load the simulated world, or prices from the market-data page)")
        self.ids = [int(i) for i in info["security_id"] if int(i) in adj.columns]
        self.info = info.set_index("security_id").loc[self.ids]
        self.adj = adj.reindex(columns=self.ids).sort_index()
        self.close = store.closes(self.ids, start, end).reindex(index=self.adj.index, columns=self.ids)
        self.volume = store.volumes(self.ids, start, end).reindex(index=self.adj.index, columns=self.ids)
        self.notes: list[str] = []
        if self.close.isna().all().all():
            self.close = self.adj
            self.notes.append("only adjusted prices are stored, so market capitalisation is approximate (a split after a date changes the price it is built from)")
        self.dates, self.pos = self._grid()
        self.tables = PeriodTables(store, self.ids, self.settings)
        for basis in ("fy", "ttm"):
            metrics.compute(self.tables, basis)
        self.notes.extend(self.tables.notes)

    # ------------------------------------------------------------------------------------------------------------ the universe and the grid
    def _select(self, security_ids: list[int] | None) -> pd.DataFrame:
        s = self.spec
        frame = self.store.securities(active_only=not s.include_inactive)
        if security_ids is not None:
            frame = frame[frame["security_id"].isin(security_ids)]
        for column, values in (("country", s.countries), ("region", s.regions), ("sector", s.sectors), ("market", s.markets)):
            if values:
                frame = frame[frame[column].isin(values)]
        if s.developed_only:
            frame = frame[frame["country"].isin(DEVELOPED)]
        if frame.empty:
            raise FundamentalsError("no companies match this universe")
        return frame.reset_index(drop=True)

    def _grid(self) -> tuple[pd.DatetimeIndex, np.ndarray]:
        idx = self.adj.index
        code = {"W": "W-FRI", "M": "M", "Q": "Q"}[self.freq]
        last = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period(code)).last()
        pos = last.to_numpy(int)
        return idx[pos], pos

    @property
    def n(self) -> int:
        return len(self.ids)

    def group(self, kind: str) -> pd.Series:
        """``security_id -> label`` for ``sector``, ``industry``, ``country``, ``region`` or ``market``; a company with no label gets ``unknown`` so neutralising never drops it."""
        if kind not in GROUPS:
            raise FundamentalsError(f"group must be one of {GROUPS}")
        col = {"market": "market"}.get(kind, kind)
        return self.info[col].astype(object).where(self.info[col].notna(), "unknown").astype(str)

    # ------------------------------------------------------------------------------------------------------------ returns
    def forward(self, horizon="1m") -> pd.DataFrame:
        """Dates by securities: the return from one day after the date (``entry_lag``) over the next ``horizon`` trading days. A company that delists in between is carried to its last price."""
        h = horizon_days(horizon)
        key = ("forward", h)
        if key not in self._cache:
            n = len(self.adj)
            entry, exit_ = self.pos + self.entry_lag, self.pos + self.entry_lag + h
            values = np.full((len(self.pos), self.n), np.nan)
            ok = exit_ < n
            if ok.any():
                held = self.adj.to_numpy(float)
                carried = self.adj.ffill().to_numpy(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    values[ok] = carried[exit_[ok]] / held[entry[ok]] - 1.0
            self._cache[key] = pd.DataFrame(np.where(np.isfinite(values), values, np.nan), index=self.dates, columns=self.ids)
        return self._cache[key]

    def settled_lag(self, horizon="1m") -> int:
        """How many grid periods must pass before a date's forward window has ended: the smallest ``L`` such that, from the day after any date over ``horizon`` trading days, the return is over by the
        date ``L`` periods later. Whatever is learned from realised returns (an IC, a model fitted to them) may be used on date ``t`` only if it comes from dates at least ``L`` periods before ``t``."""
        h = horizon_days(horizon)
        key = ("settled", h)
        if key not in self._cache:
            ends = self.pos + self.entry_lag + h
            first = np.searchsorted(self.pos, ends, side="left")
            reached = first < len(self.pos)
            gaps = (first - np.arange(len(self.pos)))[reached]
            self._cache[key] = int(max(gaps.max(), 1)) if len(gaps) else len(self.pos)
        return self._cache[key]

    def past(self, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
        """The return from ``lookback`` to ``skip`` trading days before each date, known on the date (momentum: 252 and 21 give the classic twelve-minus-one-month)."""
        key = ("past", lookback, skip)
        if key not in self._cache:
            a = self.adj.to_numpy(float)
            late, early = self.pos - skip, self.pos - lookback
            ok = early >= 0
            values = np.full((len(self.pos), self.n), np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                values[ok] = a[late[ok]] / a[early[ok]] - 1.0
            self._cache[key] = pd.DataFrame(np.where(np.isfinite(values), values, np.nan), index=self.dates, columns=self.ids)
        return self._cache[key]

    def _daily_returns(self) -> pd.DataFrame:
        if "daily_returns" not in self._cache:
            self._cache["daily_returns"] = self.adj.pct_change(fill_method=None)
        return self._cache["daily_returns"]

    def volatility(self, window: int = 63) -> pd.DataFrame:
        """Annualised standard deviation of daily returns over the last ``window`` trading days, as known on each date."""
        key = ("vol", window)
        if key not in self._cache:
            vol = self._daily_returns().rolling(window, min_periods=max(window // 2, 20)).std() * np.sqrt(252.0)
            self._cache[key] = vol.iloc[self.pos].set_axis(self.dates)
        return self._cache[key]

    def market_return(self) -> pd.Series:
        """The equal-weighted return of the universe each day, the market the betas are measured against."""
        if "market" not in self._cache:
            self._cache["market"] = self._daily_returns().mean(axis=1)
        return self._cache["market"]

    def beta(self, window: int = 252) -> pd.DataFrame:
        key = ("beta", window)
        if key not in self._cache:
            r, m = self._daily_returns(), self.market_return()
            minimum = max(window // 2, 40)
            cov = r.rolling(window, min_periods=minimum).cov(m)
            var = m.rolling(window, min_periods=minimum).var()
            self._cache[key] = cov.div(var, axis=0).iloc[self.pos].set_axis(self.dates)
        return self._cache[key]

    # ------------------------------------------------------------------------------------------------------------ prices, size, currency
    def _as_of(self, daily: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
        return daily.ffill(limit=limit).iloc[self.pos].set_axis(self.dates)

    def price(self) -> pd.DataFrame:
        """The unadjusted close on each date (the last close within five trading days), in the trading currency."""
        if "price" not in self._cache:
            self._cache["price"] = self._as_of(self.close)
        return self._cache["price"]

    def _fx_per_usd(self) -> pd.DataFrame:
        """Dates by currency: units of the currency per US dollar as of each rebalance date (USD is 1)."""
        if "fx" not in self._cache:
            frame = self.store.fx()
            frame = frame[frame["base"] == "USD"] if not frame.empty else frame
            if frame.empty:
                wide = pd.DataFrame(index=self.dates)
            else:
                wide = frame.pivot_table(index="date", columns="quote", values="rate", aggfunc="last").sort_index()
                wide = wide.reindex(wide.index.union(self.dates)).ffill().reindex(self.dates)
            wide["USD"] = 1.0
            self._cache["fx"] = wide
        return self._cache["fx"]

    def _currency_panel(self, labels: pd.Series) -> pd.DataFrame:
        fx = self._fx_per_usd()
        out = pd.DataFrame(np.nan, index=self.dates, columns=self.ids)
        for cur in labels.dropna().unique():
            if cur in fx.columns:
                cols = labels.index[labels == cur]
                out[cols] = np.repeat(fx[cur].to_numpy()[:, None], len(cols), axis=1)
        return out

    def trading_per_usd(self) -> pd.DataFrame:
        if "trade_fx" not in self._cache:
            self._cache["trade_fx"] = self._currency_panel(self.info["currency"].astype(object).where(self.info["currency"].notna(), None))
        return self._cache["trade_fx"]

    def reporting_currency(self) -> pd.Series:
        """The currency each company reports in (its latest period's), falling back on its trading currency."""
        if "rep_cur" not in self._cache:
            latest = self.tables.ttm.dropna(subset=["reporting_currency"]).groupby("security_id")["reporting_currency"].last() if not self.tables.ttm.empty else pd.Series(dtype=object)
            cur = self.info["currency"].astype(object).copy()                     # a copy: updating a view would overwrite the trading currencies on older pandas
            known = latest.reindex(self.ids).dropna()
            cur.loc[known.index] = known.to_numpy(object)
            self._cache["rep_cur"] = cur
        return self._cache["rep_cur"]

    def to_trading(self, amount: pd.DataFrame) -> pd.DataFrame:
        """An amount in each company's reporting currency, in its trading currency at the exchange rate of the date. Where the two currencies are the same it is returned as it is; where a rate
        is missing it is missing (never converted at 1)."""
        rep = self.reporting_currency()
        trade = self.info["currency"].astype(object)
        same = (rep == trade) | trade.isna()
        if same.all():
            return amount
        reporting = self._currency_panel(rep)
        trading = self.trading_per_usd().copy()
        cross = trading / reporting
        cross.loc[:, same.index[same]] = 1.0
        return amount * cross.reindex(index=amount.index, columns=amount.columns)

    def shares(self) -> pd.DataFrame:
        if "shares" not in self._cache:
            have = self.panel("shares_outstanding")
            self._cache["shares"] = have.fillna(self.panel("weighted_avg_shares_diluted"))
        return self._cache["shares"]

    def market_cap(self) -> pd.DataFrame:
        """Unadjusted close times the share count known on the date, in the trading currency."""
        if "mcap" not in self._cache:
            self._cache["mcap"] = self.price() * self.shares()
        return self._cache["mcap"]

    def market_cap_usd(self) -> pd.DataFrame:
        if "mcap_usd" not in self._cache:
            self._cache["mcap_usd"] = self.market_cap() / self.trading_per_usd()
        return self._cache["mcap_usd"]

    def enterprise_value(self) -> pd.DataFrame:
        """Market capitalisation plus debt, minority interest and preferred equity, less cash, in the trading currency; missing when the balance sheet has no debt and no cash figure at all."""
        if "ev" not in self._cache:
            debt, cash = self.panel("total_debt"), self.panel("cash_and_equivalents").fillna(self.panel("cash"))
            other = self.panel("minority_interest").fillna(0.0) + self.panel("preferred_equity").fillna(0.0)
            net = self.to_trading(debt.fillna(0.0) - cash.fillna(0.0) + other)
            known = debt.notna() | cash.notna()
            self._cache["ev"] = (self.market_cap() + net).where(known)
        return self._cache["ev"]

    def dollar_volume_usd(self, window: int = 63) -> pd.DataFrame:
        key = ("dollar_volume", window)
        if key not in self._cache:
            dv = (self.close * self.volume).rolling(window, min_periods=max(window // 2, 20)).mean()
            self._cache[key] = dv.iloc[self.pos].set_axis(self.dates) / self.trading_per_usd()
        return self._cache[key]

    def investable(self) -> pd.DataFrame:
        """True where a company could have been bought on the date: priced, and (if the universe asks) large and liquid enough."""
        if "investable" not in self._cache:
            mask = self.price().notna()
            if self.spec.min_market_cap_usd > 0:
                mask &= self.market_cap_usd() >= self.spec.min_market_cap_usd
            if self.spec.min_dollar_volume_usd > 0:
                mask &= self.dollar_volume_usd() >= self.spec.min_dollar_volume_usd
            self._cache["investable"] = mask.fillna(False)
        return self._cache["investable"]

    # ------------------------------------------------------------------------------------------------------------ the statements
    def panel(self, name: str, basis: str = "ttm") -> pd.DataFrame:
        """A statement figure or derived metric as known on each date."""
        key = ("panel", name, basis)
        if key not in self._cache:
            self._cache[key] = self.tables.panel(name, self.dates, basis=basis, security_ids=self.ids)
        return self._cache[key]

    def has(self, name: str, basis: str = "ttm") -> bool:
        return name in self.tables.table(basis).columns

    # ------------------------------------------------------------------------------------------------------------ what this universe is
    def survivorship(self) -> dict:
        """What the sample can and cannot say about survivorship. It counts the companies in the universe that are no longer listed; it calls the sample unbiased only if the user declared the
        securities to be the historical constituents (the ``universe_kind`` setting is ``point_in_time``)."""
        declared = self.store.get_meta("universe_kind", "")
        inactive = int((~self.info["active"].astype(bool)).sum())
        share = inactive / max(self.n, 1)
        if declared == "point_in_time":
            verdict = "the securities were declared to be the historical constituents; survivorship bias depends on how complete that list is"
        elif inactive == 0:
            verdict = "every company in the database is still listed, so the sample leaves out the ones that failed and the results are probably flattering (survivorship bias)"
        else:
            verdict = f"{inactive} of {self.n} companies have been delisted and are kept with their losses, but the database is not a complete history of who was listed, so some survivorship bias may remain"
        return {"companies": self.n, "inactive": inactive, "inactive_share": share, "declared": declared or "not declared", "verdict": verdict, "unbiased": declared == "point_in_time"}

    def coverage(self, names: list[str] | None = None, basis: str = "ttm") -> pd.DataFrame:
        """For each figure, the share of (date, company) cells that have a value among the investable ones."""
        names = names or ["revenue", "net_income", "total_assets", "shareholders_equity", "operating_cash_flow", "gross_profit", "total_debt", "shares_outstanding"]
        mask = self.investable()
        out = {n: float(self.panel(n, basis).where(mask).notna().sum().sum() / max(mask.sum().sum(), 1)) for n in names if self.has(n, basis)}
        return pd.DataFrame({"coverage": out})

    def describe(self) -> dict:
        s = self.survivorship()
        pit_estimated = self.tables.estimated_share()
        return {"universe": self.spec.describe(), "companies": self.n, "dates": len(self.dates), "first": str(self.dates[0].date()), "last": str(self.dates[-1].date()), "freq": self.freq,
                "entry_lag_days": self.entry_lag, "countries": int(self.info["country"].nunique()), "sectors": int(self.info["sector"].nunique()),
                "survivorship": s, "pit_estimated_share": None if not np.isfinite(pit_estimated) else float(pit_estimated), "pit": self.settings.describe(), "notes": list(self.notes)}
