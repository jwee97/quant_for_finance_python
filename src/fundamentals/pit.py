"""The point-in-time engine: what was known on a given day, enforced in one place.

A figure is *available* on the day it became public, never on the last day of the period it describes. For every stored period that day is

    available = (the earliest filing, acceptance or publication date a provider gave) + ``info_lag_days``       (a restatement: the day we first saw it, not the original filing date)
    available = period end + a conservative lag for that kind of period, flagged ``pit_estimated``              (when no date is known: never silently the period end)

and a research date ``t`` may use a period only if ``available <= t``. Every panel the factor engine reads is built here, so no strategy has to remember the rule.

Two views of a company's statements are built:

``fy``   the annual statements, one row per financial year;
``ttm``  trailing twelve months at each quarter end: the flows of the last four quarters (or two half-years) added, the balance-sheet items as at the latest quarter. A year-end row from the annual
         statements stands in where no quarterly history exists. Quarterly data that do not add up to the annual figures (year-to-date numbers mistaken for quarters) are not used.

Where two providers report the same period the first in ``provider_priority`` leads and the other fills only the gaps, and only with values that were public no later than the leading row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import canonical
from .quality import DEFAULT_EXCLUDE
from .store import FundamentalsStore

QUARTERS = ("Q1", "Q2", "Q3", "Q4")
HALVES = ("H1", "H2")


@dataclass(frozen=True)
class PitSettings:
    info_lag_days: int = 1                      # a filing is usable from the next day: it may have come after the close
    estimated_lag_days: tuple = (("FY", 120), ("H1", 90), ("H2", 90), ("Q1", 60), ("Q2", 60), ("Q3", 60), ("Q4", 60))
    use_estimated: bool = True                  # False: periods without a real filing date are left out altogether
    exclude_flags: tuple = DEFAULT_EXCLUDE      # observations carrying these quality flags are treated as missing
    max_age_days: tuple = (("fy", 600), ("ttm", 420))
    provider_priority: tuple = ("openfilings", "simulated")

    def lag_for(self, fiscal_period: str) -> int:
        return dict(self.estimated_lag_days).get(fiscal_period, 120)

    def age_for(self, basis: str) -> int:
        return dict(self.max_age_days)[basis]

    def rank(self, provider: str) -> int:
        order = list(self.provider_priority)
        return order.index(provider) if provider in order else len(order)

    def describe(self) -> dict:
        return {"info_lag_days": self.info_lag_days, "estimated_lag_days": dict(self.estimated_lag_days), "use_estimated": self.use_estimated, "exclude_flags": list(self.exclude_flags),
                "max_age_days": dict(self.max_age_days), "provider_priority": list(self.provider_priority)}


def availability(periods: pd.DataFrame, settings: PitSettings) -> tuple[pd.Series, pd.Series]:
    """``(available_date, estimated)`` for each row of a periods frame (see the module docstring for the rule)."""
    known = periods[["filing_date", "accepted_date", "publication_date"]].min(axis=1)
    restated = periods["pit_quality"].eq("restated")
    first = known.where(~restated, periods["publication_date"])
    estimated = first.isna()
    lag = pd.to_timedelta(periods["fiscal_period"].map(settings.lag_for).astype(float), unit="D")
    available = first.fillna(periods["period_end"] + lag) + pd.Timedelta(days=settings.info_lag_days)
    return available, estimated


def step_panel(events: pd.DataFrame, dates: pd.DatetimeIndex, columns: list[int], value: str = "value", max_age_days: int | None = None) -> pd.DataFrame:
    """Dates by securities: the latest period's ``value`` that was public on each date.

    ``events`` has ``security_id``, ``available_date``, ``period_end`` and ``value``. On a date, the value is that of the event with the latest period end among those already public (so a late
    restatement of an old year never replaces a newer year), the newest vintage of a period winning; it is kept until the next event and dropped when the period is older than ``max_age_days``.
    """
    dates = pd.DatetimeIndex(dates)
    out = pd.DataFrame(np.nan, index=dates, columns=columns)
    ev = events.dropna(subset=[value]).sort_values(["security_id", "available_date", "period_end"])
    if ev.empty:
        return out
    ev = ev[ev["period_end"] >= ev.groupby("security_id")["period_end"].cummax()]
    col_of = {c: j for j, c in enumerate(columns)}
    ev = ev[ev["security_id"].isin(col_of)]
    day_ns = 86_400 * 1_000_000_000
    grid_days = np.asarray(dates, dtype="datetime64[ns]").astype("int64") / day_ns            # days since the epoch, whatever resolution the index carries
    pos = np.searchsorted(grid_days, np.asarray(ev["available_date"], dtype="datetime64[ns]").astype("int64") / day_ns, side="left")
    keep = pos < len(dates)
    ev, pos = ev[keep], pos[keep]
    cols = ev["security_id"].map(col_of).to_numpy()
    vals = np.full((len(dates), len(columns)), np.nan)
    ends = np.full((len(dates), len(columns)), np.nan)
    vals[pos, cols] = ev[value].to_numpy(float)                                  # later events overwrite earlier ones on the same date: they are sorted
    ends[pos, cols] = np.asarray(ev["period_end"], dtype="datetime64[ns]").astype("int64") / day_ns
    grid = pd.DataFrame(vals, index=dates, columns=columns).ffill()
    if max_age_days is not None:
        e = pd.DataFrame(ends, index=dates, columns=columns).ffill()
        age = pd.DataFrame(grid_days[:, None] - e.to_numpy(), index=dates, columns=columns)
        grid = grid.where(age <= max_age_days)
    return grid


@dataclass(eq=False, repr=False)
class PeriodTables:
    """The annual and trailing-twelve-month tables of a set of companies, built once from the database."""

    store: FundamentalsStore
    security_ids: list[int] | None = None
    settings: PitSettings = field(default_factory=PitSettings)
    fy: pd.DataFrame = field(init=False)
    ttm: pd.DataFrame = field(init=False)
    notes: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        self._prior_cache: dict[tuple, np.ndarray] = {}
        rows = self._rows()
        self.notes = []
        self.fy = self._finish(rows[rows["fiscal_period"] == "FY"], "FY")
        self.ttm = self._build_ttm(rows)

    # ------------------------------------------------------------------------------------------------------------ assembling the rows
    def _rows(self) -> pd.DataFrame:
        s = self.settings
        periods = self.store.periods(self.security_ids)
        periods = periods[periods["vintage"] == 0]                                  # figures as first filed; a restatement changes the raw panels only
        periods["available_date"], periods["pit_estimated"] = availability(periods, s)
        if not s.use_estimated:
            periods = periods[~periods["pit_estimated"]]
        obs = self.store.observations(self.security_ids)
        obs = obs[obs["period_id"].isin(periods["period_id"]) & ~obs["quality_flag"].isin(s.exclude_flags)]
        wide = obs.pivot_table(index="period_id", columns="metric_code", values="value", aggfunc="last")
        for code in canonical.METRICS:
            if code not in wide.columns:
                wide[code] = np.nan
        rows = periods.merge(wide, left_on="period_id", right_index=True, how="inner")
        rows["rank"] = rows["provider"].map(s.rank)
        return self._merge_providers(rows)

    @staticmethod
    def _merge_providers(rows: pd.DataFrame) -> pd.DataFrame:
        """One row per ``(security, fiscal period, period end)``: the leading provider's, gaps filled from the others only with values public no later than the leading row."""
        if rows.groupby(["security_id", "fiscal_period", "period_end"]).ngroups == len(rows):
            return rows.sort_values(["security_id", "period_end"]).reset_index(drop=True)
        rows = rows.sort_values(["security_id", "fiscal_period", "period_end", "rank"]).copy()
        key = ["security_id", "fiscal_period", "period_end"]
        lead_available = rows.groupby(key)["available_date"].transform("first")
        late = rows["available_date"] > lead_available
        metric_cols = list(canonical.METRICS)
        rows.loc[late, metric_cols] = np.nan
        merged = rows.groupby(key, sort=False).first().reset_index()                 # first() takes the first non-missing value of each column, in provider order
        return merged.sort_values(["security_id", "period_end"]).reset_index(drop=True)

    def _finish(self, rows: pd.DataFrame, basis: str) -> pd.DataFrame:
        out = rows.copy()
        out["basis"] = basis
        return out.sort_values(["security_id", "period_end"]).reset_index(drop=True)

    # ------------------------------------------------------------------------------------------------------------ trailing twelve months
    def _build_ttm(self, rows: pd.DataFrame) -> pd.DataFrame:
        flows = [c for c in canonical.flows()]
        stocks = [c for c in canonical.stocks()]
        parts = []
        q = rows[rows["fiscal_period"].isin(QUARTERS + HALVES)].sort_values(["security_id", "period_end"]).copy()
        if not q.empty:
            q["step"] = np.where(q["fiscal_period"].isin(QUARTERS), 91, 182)
            gap = q.groupby("security_id")["period_end"].diff().dt.days
            consecutive = gap.sub(q["step"]).abs().le(30)
            q["run"] = (~consecutive).groupby(q["security_id"]).cumsum()
            for window, mask in ((4, q["fiscal_period"].isin(QUARTERS)), (2, q["fiscal_period"].isin(HALVES))):
                sub = q[mask]
                if sub.empty:
                    continue
                keys = [sub["security_id"], sub["run"]]
                pos = sub.groupby(keys).cumcount()
                values = sub[flows]
                bad = values.isna().astype(int).groupby(keys).cumsum()
                csum = values.fillna(0.0).groupby(keys).cumsum()
                total = csum - csum.groupby(keys).shift(window).fillna(0.0)
                missing = (bad - bad.groupby(keys).shift(window).fillna(0)) > 0
                ttm = total.where(~missing)
                out = sub.copy()
                out[flows] = ttm
                out = out[pos >= window - 1]
                out["fiscal_period"] = "TTM"
                parts.append(out.drop(columns=["step", "run"]))
        quarter_based = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=rows.columns)
        quarter_based = self._drop_inconsistent(quarter_based, rows)
        fy = rows[rows["fiscal_period"] == "FY"].copy()
        fy["fiscal_period"] = "TTM"
        have = set(zip(quarter_based["security_id"], quarter_based["period_end"])) if len(quarter_based) else set()
        fy = fy[[(s, e) not in have for s, e in zip(fy["security_id"], fy["period_end"])]]
        both = pd.concat([quarter_based, fy], ignore_index=True)
        for c in stocks:
            if c not in both.columns:
                both[c] = np.nan
        return self._finish(both, "TTM") if len(both) else both

    def _drop_inconsistent(self, ttm: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Trailing sums at a fiscal year end must equal the annual figure. Where revenue differs by more than ten per cent the 'quarters' were probably year-to-date totals: that company's
        quarterly history is not used."""
        if ttm.empty:
            return ttm
        fy = rows[rows["fiscal_period"] == "FY"][["security_id", "period_end", "revenue"]].rename(columns={"revenue": "fy_revenue"})
        joined = ttm.merge(fy, on=["security_id", "period_end"], how="inner")
        joined = joined[joined["revenue"].notna() & joined["fy_revenue"].notna() & (joined["fy_revenue"] != 0)]
        if joined.empty:
            return ttm
        miss = ((joined["revenue"] / joined["fy_revenue"]) - 1.0).abs().groupby(joined["security_id"]).median()
        bad = miss.index[miss > 0.10]
        if len(bad):
            self.notes.append(f"{len(bad)} companies' quarterly figures do not add up to their annual ones and were not used")
        return ttm[~ttm["security_id"].isin(bad)]

    # ------------------------------------------------------------------------------------------------------------ the year before
    def table(self, basis: str) -> pd.DataFrame:
        return self.fy if basis == "fy" else self.ttm

    def prior_positions(self, basis: str, years: int = 1) -> np.ndarray:
        """For each row of the table, the position of the same company's row ``years`` years earlier (period ends about a year apart), or -1 where there is none."""
        key = (basis, years)
        if key not in self._prior_cache:
            table = self.table(basis)
            if table.empty:
                self._prior_cache[key] = np.zeros(0, dtype=int)
            else:
                left = table[["security_id"]].assign(target=table["period_end"] - pd.DateOffset(years=years), row=np.arange(len(table))).sort_values("target")
                right = table[["security_id", "period_end"]].assign(pos=np.arange(len(table))).rename(columns={"period_end": "prior_end"}).sort_values("prior_end")
                match = pd.merge_asof(left, right, left_on="target", right_on="prior_end", by="security_id", tolerance=pd.Timedelta(days=45), direction="nearest").sort_values("row")
                self._prior_cache[key] = match["pos"].fillna(-1).to_numpy(int)
        return self._prior_cache[key]

    def prior_col(self, basis: str, name: str, years: int = 1) -> pd.Series:
        """The column ``name`` as it was ``years`` years earlier for each row (NaN when there is no such period), index-aligned with the table."""
        table = self.table(basis)
        pos = self.prior_positions(basis, years)
        values = table[name].to_numpy(float)
        out = np.where(pos >= 0, values[np.clip(pos, 0, max(len(values) - 1, 0))] if len(values) else np.nan, np.nan)
        return pd.Series(out, index=table.index, name=f"{name}_{years}y_ago")

    def prior(self, basis: str, years: int = 1) -> pd.DataFrame:
        """Every canonical figure of the row ``years`` years earlier, index-aligned with the table."""
        table = self.table(basis)
        cols = [c for c in canonical.METRICS if c in table.columns]
        return pd.DataFrame({c: self.prior_col(basis, c, years) for c in cols}, index=table.index)

    def add_columns(self, basis: str, columns: dict[str, pd.Series]) -> None:
        """Attach derived columns (index-aligned) to a table; the next call to :meth:`prior_col` can look them up a year back too."""
        table = self.table(basis)
        for name, series in columns.items():
            table[name] = series.reindex(table.index).to_numpy(float)

    def columns(self, basis: str) -> list[str]:
        table = self.table(basis)
        return [c for c in table.columns if c not in ("period_id", "security_id", "fiscal_year", "fiscal_period", "period_start", "period_end", "filing_date", "accepted_date", "publication_date",
                                                      "reporting_currency", "provider", "vintage", "pit_quality", "available_date", "pit_estimated", "rank", "basis")]

    # ------------------------------------------------------------------------------------------------------------ panels
    def panel(self, name: str, dates: pd.DatetimeIndex, basis: str = "ttm", security_ids: list[int] | None = None, lag_years: int = 0) -> pd.DataFrame:
        """Dates by securities: ``name`` as known on each date. ``lag_years=1`` gives the figure for the period a year before the latest one (period-aligned, as known now)."""
        table = self.table(basis)
        if name not in table.columns:
            raise KeyError(f"no figure called {name!r} in the {basis} table")
        cols = list(security_ids if security_ids is not None else (self.security_ids or sorted(table["security_id"].unique())))
        frame = table[["security_id", "available_date", "period_end"]].copy()
        frame["value"] = table[name] if lag_years == 0 else self.prior_col(basis, name, lag_years)
        return step_panel(frame, dates, cols, "value", self.settings.age_for(basis))

    def estimated_share(self) -> float:
        """The share of annual periods whose availability date had to be assumed."""
        return float(self.fy["pit_estimated"].mean()) if len(self.fy) else float("nan")
