"""A restricted, read-only view of a dataset for models that read history (P0-18, P0-20, section 3.2 of the spec).

A model that is bound to a dataset may use the feature panel, the returns, the investable flags, the groups and the calendar up to a cutoff, and nothing else: the view physically holds no row after the cutoff, and
the labels are not in it at all. A model cannot read the future or an outcome by mistake, or by trusting a comment, because there is nothing there to read. The trainer binds a fit to the information available when the
model is fitted (the close before the first day it will be asked about) and rebinds it to the block it predicts, so a perturbation of any later observation cannot change what the model fitted or what it said earlier.

A bound model is still responsible for each of its predictions using only dates up to the one it scores; the view bounds the damage of getting that wrong to the block, and the perturbation tests check the rest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CausalAccessError(AttributeError):
    """Something a causal view does not hold: a label, an outcome, or a date after its cutoff."""


class CausalView:
    SEALED = ("Y", "labels")

    def __init__(self, ds, until=None):
        if isinstance(ds, CausalView):
            ds = ds._source
        dates = pd.DatetimeIndex(ds.prices.index)
        last = len(dates) - 1 if until is None else (int(until) if isinstance(until, (int, np.integer)) else int(dates.get_indexer([pd.Timestamp(until)])[0]))
        if last < 0 or last > len(dates) - 1:
            raise CausalAccessError(f"the cutoff {until!r} is not a date of the dataset")
        n = last + 1
        self._source, self.cutoff_pos, self.cutoff = ds, last, dates[last]
        pos = ds.positions()
        k = int((pos <= last).sum()) if np.all(np.diff(pos) >= 0) else None
        self.X = ds.X.iloc[:k] if k is not None else ds.X[pos <= last]
        self.prices, self.returns, self.investable = ds.prices.iloc[:n], ds.returns.iloc[:n], ds.investable.iloc[:n]
        self.group, self.asset_class = dict(ds.group), dict(ds.asset_class)
        self._meta = {k_: v for k_, v in ds.meta.items() if k_ in ("lag", "horizon", "target", "name", "universe")}

    @classmethod
    def of(cls, ds, until=None) -> "CausalView":
        """The dataset as a view (a view stays as it is unless a different cutoff is asked for; a longer cutoff than the view's own is refused)."""
        if isinstance(ds, CausalView):
            if until is None:
                return ds
            view = cls(ds._source, until)
            if view.cutoff_pos > ds.cutoff_pos:
                raise CausalAccessError(f"a view that ends on {ds.cutoff.date()} cannot be widened to {view.cutoff.date()}")
            return view
        return cls(ds, until)

    # -- what a model may read
    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.prices.index)

    @property
    def assets(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)

    @property
    def lag(self) -> int:
        return int(self._meta["lag"])

    @property
    def horizon(self) -> int:
        return int(self._meta["horizon"])

    @property
    def target(self) -> str:
        return self._meta["target"]

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    def positions(self) -> np.ndarray:
        return self.dates.get_indexer(self.X.index.get_level_values(0))

    # -- what it does not hold
    def __getattr__(self, name):
        if name in CausalView.SEALED:
            raise CausalAccessError("labels are not part of a causal view: a bound model is given the training labels it may fit to as an argument of fit, and nothing else")
        raise CausalAccessError(f"'{name}' is not available through a causal view of the dataset")
