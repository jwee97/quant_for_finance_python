"""Company statements as the market could have known them: a point-in-time table, and the market value and enterprise value built from it.

This repository carries prices, not financial statements, so the value, quality and estimate factors read a file you supply (``data/user/fundamentals.csv``). The file is a wide table with one row per
company and filing::

    ticker, period_end, available, sales, cogs, ebitda, net_income, cfo, capex, total_assets, ...

``period_end`` is the end of the fiscal period the figures describe. ``available`` is the first date they could have been used (the filing or announcement date); if the column is missing it is
``period_end + lag_days`` (60 days by default, the usual 10-Q and 10-K deadline), so a figure is never used before it was public. A figure holds from its ``available`` date until the next filing
supersedes it and goes stale after ``expiry`` trading days. Flows (sales, earnings, cash flow, capital expenditure, dividends, issuance) are for the trailing twelve months; balance-sheet items are as of
the period end; analyst fields (``eps_fy1``, ``ltg``, ``n_up``, ``n_down``, ``n_estimates``) are as of the date they were published. Use one currency and consistent units: the market value is
``price * shares_outstanding``, so ``shares_outstanding`` must be on the same split-adjusted basis as the prices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

USER_DATA = Path(__file__).resolve().parents[2] / "data" / "user"
MISSING = ["", "NaN", "nan", "N/A", "n/a", "#N/A"]                                                       # NOT "NA": that is a ticker
YEAR, MONTH = 252, 21
DEBT = ("total_debt", "short_term_debt", "long_term_debt")                                                  # enterprise value needs debt: either the total or its parts

FIELDS = {
    # flows: trailing twelve months
    "sales": "revenue", "cogs": "cost of goods sold", "sga": "selling, general and administrative expense", "ebitda": "earnings before interest, taxes, depreciation and amortisation",
    "operating_income": "operating income (EBIT)", "net_income": "net income", "cfo": "cash flow from operations", "capex": "capital expenditure (positive)", "depreciation": "depreciation and amortisation",
    "interest_expense": "interest expense", "dividends": "cash dividends paid (positive)", "equity_issuance": "proceeds from issuing shares", "equity_repurchase": "cash spent buying back shares (positive)",
    "debt_issuance": "proceeds from issuing debt", "debt_repayment": "debt repaid (positive)", "tax_rate": "the effective tax rate, 0 to 1",
    # balance sheet: at the period end
    "total_assets": "total assets", "current_assets": "current assets", "current_liabilities": "current liabilities", "cash": "cash and short-term investments", "short_term_debt": "debt due within a year",
    "long_term_debt": "long-term debt", "total_debt": "total debt (only if short- and long-term debt are not given separately)", "total_liabilities": "total liabilities", "book_equity": "common shareholders' equity", "preferred": "preferred stock", "minority_interest": "minority interest",
    "long_term_investments": "long-term investments", "gross_plant": "gross property, plant and equipment", "net_plant": "net property, plant and equipment",
    "shares_outstanding": "shares outstanding, split-adjusted",
    # analysts: as published
    "eps_fy1": "consensus earnings per share for the current fiscal year", "ltg": "consensus long-term growth forecast", "n_up": "estimates revised up over the last month",
    "n_down": "estimates revised down over the last month", "n_estimates": "number of analysts",
}


class FundamentalsError(ValueError):
    """The statements file is missing, unreadable or inconsistent."""


class Fundamentals:
    """A table of filings (one row per ticker and filing) and the point-in-time panels made from it."""

    def __init__(self, table: pd.DataFrame, expiry: int = 400):
        required = {"ticker", "available"}
        if not required <= set(table.columns):
            raise FundamentalsError("the table needs the columns ticker and available")
        if expiry < 1:
            raise FundamentalsError("expiry must be at least one trading day")
        table = table.copy()
        table["ticker"] = table["ticker"].astype(str).str.strip().str.upper()
        table["available"] = pd.to_datetime(table["available"])
        if table["available"].isna().any():
            raise FundamentalsError("every filing needs an available date")
        self.fields = [c for c in table.columns if c in FIELDS]
        self.extra = [c for c in table.columns if c not in FIELDS and c not in ("ticker", "available", "period_end")]
        for c in self.fields:
            try:
                table[c] = pd.to_numeric(table[c], errors="raise")
            except (ValueError, TypeError) as error:
                raise FundamentalsError(f"column {c} has values that are not numbers ({error})") from None
        self.table = table.sort_values(["available", "ticker"]).reset_index(drop=True)
        self.expiry = int(expiry)

    # ------------------------------------------------------------------------------------------------------------ construction
    @classmethod
    def from_csv(cls, path: str | Path = "fundamentals.csv", lag_days: int = 60, expiry: int = 400) -> "Fundamentals":
        p = Path(path)
        p = p if p.is_absolute() else USER_DATA / p
        if not p.exists():
            raise KeyError(f"this fundamental model needs the file {p}. Columns: ticker, period_end (and available, the first date the figures were public), then any of: "
                           f"{', '.join(FIELDS)}. Put it there (data/user/ is not part of the repository) or pass its path.")
        table = pd.read_csv(p, keep_default_na=False, na_values=MISSING)
        return cls.from_table(table, lag_days, expiry)

    @classmethod
    def from_table(cls, table: pd.DataFrame, lag_days: int = 60, expiry: int = 400) -> "Fundamentals":
        table = table.copy()
        if "ticker" not in table.columns:
            raise FundamentalsError("the table needs a ticker column")
        if lag_days < 0:
            raise FundamentalsError("lag_days must not be negative")
        if "available" not in table.columns:
            if "period_end" not in table.columns:
                raise FundamentalsError("the table needs period_end (the fiscal period it describes) or available (the first date it was public)")
            table["available"] = pd.to_datetime(table["period_end"]) + pd.Timedelta(days=int(lag_days))
        return cls(table, expiry)

    # ------------------------------------------------------------------------------------------------------------ panels
    def has(self, *fields: str) -> bool:
        return all(f in self.fields for f in fields)

    def missing(self, *fields) -> list:
        """The fields in ``fields`` the file lacks. An entry may be a tuple, meaning any one of them will do; it is reported as the tuple when none is present."""
        return [f for f in fields if (not any(g in self.fields for g in f) if isinstance(f, tuple) else f not in self.fields)]

    @property
    def tickers(self) -> list[str]:
        return sorted(self.table["ticker"].unique())

    def panel(self, field: str, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
        """The value of ``field`` that was known on each date for each ticker: the last filing with ``available <= date``, for at most ``expiry`` trading days. Tickers without data are NaN.

        A filing dated on a non-trading day is first used on the next trading day. Of several filings on one day the last wins, but a filing that lacks this field does not erase an earlier value."""
        if field not in self.fields:
            raise KeyError(f"the fundamentals file has no column '{field}'")
        index = pd.DatetimeIndex(index)
        n = len(index)
        names = [str(c).upper() for c in columns]
        wide = self.table.pivot_table(index="available", columns="ticker", values=field, aggfunc="last").sort_index().reindex(columns=names)
        pos = index.searchsorted(wide.index.to_numpy(), side="left")
        keep = pos < n
        grid = pd.DataFrame(np.nan, index=np.arange(n), columns=names)
        if keep.any():
            folded = wide[keep].groupby(pos[keep]).last()
            grid.loc[folded.index, :] = folded.to_numpy()
        source = pd.DataFrame(np.where(grid.notna(), np.arange(n)[:, None], np.nan), columns=names).ffill()
        out = grid.ffill().where((np.arange(n)[:, None] - source) <= self.expiry)
        out.index, out.columns = index, list(columns)
        return out


class FactorInputs:
    """Everything the factor formulas read: prices, the point-in-time statements aligned to the price grid, and market value and enterprise value."""

    def __init__(self, prices: pd.DataFrame, fundamentals: Fundamentals):
        self.prices = prices
        self.returns = prices.pct_change(fill_method=None)
        self.fund = fundamentals
        self.index, self.columns = prices.index, list(prices.columns)
        self._cache: dict[str, pd.DataFrame] = {}

    def field(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = self.fund.panel(name, self.index, self.columns)
        return self._cache[name]

    def lag(self, name: str, years: float = 1.0) -> pd.DataFrame:
        """The value the same field had ``years`` ago (as known then)."""
        return self.field(name).shift(int(round(years * YEAR)))

    def average(self, name: str) -> pd.DataFrame:
        """The mean of the current and the year-ago value: the base that flows are scaled by so growth in the denominator does not flatter a ratio."""
        a, b = self.field(name), self.lag(name)
        return (a + b) / 2.0

    @property
    def market_cap(self) -> pd.DataFrame:
        return self.prices * self.field("shares_outstanding")

    @property
    def total_debt(self) -> pd.DataFrame:
        """``total_debt`` if the file has it, otherwise short-term plus long-term debt (a part the file lacks counts as zero)."""
        if self.fund.has("total_debt"):
            return self.field("total_debt").fillna(0.0)
        parts = [self.field(c).fillna(0.0) for c in ("short_term_debt", "long_term_debt") if self.fund.has(c)]
        return sum(parts) if parts else pd.DataFrame(0.0, index=self.index, columns=self.columns)

    @property
    def enterprise_value(self) -> pd.DataFrame:
        """Market value plus debt plus preferred plus minority interest minus cash; a non-positive value is not meaningful and is left missing."""
        extra = sum(self.field(c).fillna(0.0) for c in ("preferred", "minority_interest") if self.fund.has(c))
        cash = self.field("cash").fillna(0.0) if self.fund.has("cash") else 0.0
        ev = self.market_cap + self.total_debt + extra - cash
        return ev.where(ev > 0)


def ratio(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    """Numerator over denominator with zero or missing denominators left missing (never infinite)."""
    out = numerator / denominator.where(denominator.abs() > 1e-12)
    return out.replace([np.inf, -np.inf], np.nan)
