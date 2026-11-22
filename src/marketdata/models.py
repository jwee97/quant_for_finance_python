"""Typed requests and answers of the market-data layer. Plain dataclasses and DataFrames: no framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import pandas as pd

AssetClass = Literal["stocks", "options", "futures", "crypto", "indices"]
ASSET_CLASSES = ("stocks", "options", "futures", "crypto", "indices")
Timespan = Literal["second", "minute", "hour", "day", "week", "month", "quarter", "year"]
TIMESPANS = ("second", "minute", "hour", "day", "week", "month", "quarter", "year")

BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "transactions"]


class MarketDataError(Exception):
    """A request that could not be served. ``kind`` is one of: not_configured, auth, entitlement, rate_limit, not_found, bad_request, server, network, cancelled, unsupported."""

    def __init__(self, message: str, kind: str = "server", status: int = 0, endpoint: str = ""):
        super().__init__(message)
        self.kind, self.status, self.endpoint = kind, status, endpoint


@dataclass(frozen=True)
class Provenance:
    """Where an answer came from. ``adjusted`` is the provider's own adjustment flag for prices (None: not applicable)."""

    provider: str
    endpoint: str
    fetched_at: str
    asset_class: str = ""
    adjusted: bool | None = None
    pages: int = 1
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"provider": self.provider, "endpoint": self.endpoint, "fetched_at": self.fetched_at, "asset_class": self.asset_class, "adjusted": self.adjusted, "pages": self.pages, "notes": list(self.notes)}


@dataclass
class Capability:
    """What the key can reach. ``status`` is ``available`` (a probe succeeded), ``not_entitled`` (the provider refused the plan), ``no_data`` (answered, nothing returned), ``documented`` (a documented endpoint not
    probed yet) or ``error``. Nothing is ``available`` until it was actually asked."""

    asset_class: str
    dataset: str
    endpoint: str
    status: str = "documented"
    detail: str = ""
    checked_at: str = ""
    plan_note: str = ""

    def as_dict(self) -> dict:
        return {"asset_class": self.asset_class, "dataset": self.dataset, "endpoint": self.endpoint, "status": self.status, "detail": self.detail, "checked_at": self.checked_at, "plan_note": self.plan_note}


@dataclass
class BarsRequest:
    symbol: str
    asset_class: str = "stocks"
    start: str | None = None
    end: str | None = None
    multiplier: int = 1
    timespan: str = "day"
    adjusted: bool = True
    limit: int = 50000
    max_pages: int = 200


@dataclass
class Bars:
    """Bars with a DatetimeIndex (UTC for intraday, a plain date for daily) and the columns ``open high low close volume`` (+ ``vwap``, ``transactions``, and ``settlement_price`` for futures sessions)."""

    request: BarsRequest
    frame: pd.DataFrame
    provenance: Provenance
    warnings: list[str] = field(default_factory=list)


@dataclass
class Table:
    """Any tabular answer (instruments, contracts, quotes, trades, corporate actions, chains) with its provenance."""

    frame: pd.DataFrame
    provenance: Provenance
    warnings: list[str] = field(default_factory=list)


class MarketDataProvider(Protocol):
    """What the service needs from a provider. Massive is the only implementation; the shape is here so that a second provider (or a test double) can stand in without the service changing."""

    name: str

    def get_instruments(self, asset_class: str = "stocks", **kw) -> Table: ...
    def get_contracts(self, asset_class: str, underlying: str | None = None, **kw) -> Table: ...
    def get_historical_bars(self, req: BarsRequest, cancel=None, on_page=None) -> Bars: ...
    def get_quotes(self, symbol: str, asset_class: str = "stocks", **kw) -> Table: ...
    def get_trades(self, symbol: str, asset_class: str = "stocks", **kw) -> Table: ...
    def get_corporate_actions(self, ticker: str, **kw) -> Table: ...
    def get_option_chain(self, underlying: str, **kw) -> Table: ...
    def get_futures_curve(self, product_code: str, as_of: str | None = None, **kw) -> Table: ...
    def get_provider_capabilities(self) -> list[Capability]: ...
