"""Equal weight -- model M0 (spec §25).

The naive benchmark, and a demanding one. DeMiguel, Garlappi and Uppal (2009)
found 1/N beating a long list of optimisers out of sample, for the reason
Ch. 19 §19.3 gives: sophisticated methods buy a better objective at the price
of estimating inputs they cannot estimate. Every model above M0 in this
project has to justify itself against this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constraints import Constraints, project_frame


def equal_weights(assets, value: float = 1.0) -> pd.Series:
    assets = list(assets)
    if not assets:
        return pd.Series(dtype=float)
    return pd.Series(value / len(assets), index=assets)


def equal_weight_book(investable: pd.DataFrame, constraints: Constraints | None = None) -> pd.DataFrame:
    """Equal weight over the assets investable on each date.

    Using the investability mask rather than a fixed 1/15 is what keeps the
    book honest before HYG lists in 2007: the capital goes to the assets that
    exist, not to a placeholder.
    """
    mask = investable.astype(bool)
    counts = mask.sum(axis=1).replace(0, np.nan)
    weights = mask.div(counts, axis=0).fillna(0.0)
    if constraints is not None:
        weights = project_frame(weights, constraints)
    return weights
