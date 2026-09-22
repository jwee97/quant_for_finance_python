"""Automated look-ahead detection (spec §43-§44).

The test, stated precisely. Take a dataset ``D``. Build ``D'`` identical to
``D`` up to and including time ``t``, but with everything after ``t``
replaced by something radically different. Then require

    w_{<=t}(D) == w_{<=t}(D')

If a historical position changes because *future* data changed, the strategy
is reading the future. This is a property of the code, not of the returns, so
it can be tested automatically, deterministically and without any judgement
call -- which is exactly what makes it worth having.

A subtlety worth stating: the test must actually perturb something the signal
*would* use. Scrambling a column the strategy ignores proves nothing, so
``scramble_future`` shocks prices multiplicatively and the checker asserts
that the post-``t`` weights really did move (otherwise the test is vacuous
and the result is reported as inconclusive rather than as a pass).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class LeakageTestResult:
    passed: bool
    split_date: pd.Timestamp
    max_weight_difference: float
    n_differing_cells: int
    n_cells_checked: int
    future_weights_changed: bool
    message: str
    first_difference_date: pd.Timestamp | None = None

    def to_row(self) -> dict:
        return {
            "split_date": self.split_date.date().isoformat(),
            "passed": self.passed,
            "max_weight_difference": self.max_weight_difference,
            "n_differing_cells": self.n_differing_cells,
            "n_cells_checked": self.n_cells_checked,
            "future_weights_changed": self.future_weights_changed,
            "first_difference_date": (self.first_difference_date.date().isoformat()
                                      if self.first_difference_date is not None else ""),
            "message": self.message,
        }


def scramble_future(market: MarketData, split_date, seed: int = 99,
                    shock: float = 0.5, mode: str = "shock") -> MarketData:
    """Return a copy of ``market`` whose post-``split_date`` history is altered.

    ``shock``   multiply future prices by a large random path
    ``reverse`` reverse the order of future returns
    ``zero``    flatten future prices entirely

    The pre-split history is untouched, byte for byte.
    """
    rng = np.random.default_rng(seed)
    split = pd.Timestamp(split_date)
    future = market.prices.index > split
    if not future.any():
        raise ValueError(f"split date {split.date()} leaves no future observations")

    prices = market.prices.copy()
    block = prices.loc[future]
    if mode == "shock":
        multipliers = pd.DataFrame(
            rng.lognormal(0.0, shock, size=block.shape), index=block.index, columns=block.columns
        ).cumprod()
        prices.loc[future] = block.to_numpy() * multipliers.to_numpy()
    elif mode == "reverse":
        prices.loc[future] = block.to_numpy()[::-1]
    elif mode == "zero":
        prices.loc[future] = float(prices.loc[~future].iloc[-1].mean())
    else:
        raise ValueError(f"unknown scramble mode '{mode}'")

    scale = (prices / market.prices).fillna(1.0)
    return MarketData(
        prices=prices,
        close=market.close * scale,
        adjustment_ratio=market.adjustment_ratio,
        volume=market.volume,
        high=market.high * scale,
        low=market.low * scale,
        open_=market.open_ * scale,
        filled_mask=market.filled_mask,
        investable=market.investable,
        data_version=f"{market.data_version}_scrambled_{mode}",
        asset_class=market.asset_class,
        group=market.group,
    )


def check_no_lookahead(
    market: MarketData,
    build_weights: Callable[[MarketData], pd.DataFrame],
    split_date,
    tolerance: float = 1e-10,
    seed: int = 99,
    mode: str = "shock",
) -> LeakageTestResult:
    """Run the look-ahead test for one weight-building function."""
    split = pd.Timestamp(split_date)
    scrambled = scramble_future(market, split, seed, mode=mode)

    original = build_weights(market)
    altered = build_weights(scrambled)

    common_index = original.index.intersection(altered.index)
    common_columns = original.columns.intersection(altered.columns)
    past = common_index[common_index <= split]
    future = common_index[common_index > split]

    if len(past) == 0:
        return LeakageTestResult(False, split, np.nan, 0, 0, False,
                                 "no pre-split weights to compare")

    left = original.loc[past, common_columns].fillna(0.0)
    right = altered.loc[past, common_columns].fillna(0.0)
    difference = (left - right).abs()
    max_difference = float(difference.to_numpy().max()) if difference.size else 0.0
    differing = int((difference > tolerance).to_numpy().sum())

    first_difference = None
    if differing:
        rows = difference.max(axis=1)
        offending = rows[rows > tolerance]
        if len(offending):
            first_difference = pd.Timestamp(offending.index[0])

    # Guard against a vacuous pass: the perturbation must actually change the
    # post-split book, otherwise the function ignores the data we scrambled.
    future_changed = False
    if len(future):
        future_difference = (original.loc[future, common_columns].fillna(0.0)
                             - altered.loc[future, common_columns].fillna(0.0)).abs()
        future_changed = bool((future_difference > tolerance).to_numpy().any())

    if differing:
        message = (f"LEAKAGE: {differing} pre-split weights changed when future data changed "
                   f"(max {max_difference:.3e}, first at {first_difference.date() if first_difference is not None else '?'})")
        passed = False
    elif not future_changed:
        message = ("INCONCLUSIVE: the perturbation did not change any post-split weight either, "
                   "so the test did not exercise this strategy's inputs")
        passed = False
    else:
        message = "PASS: no pre-split weight responded to future data"
        passed = True

    LOGGER.info("look-ahead test @ %s: %s", split.date(), message)
    return LeakageTestResult(
        passed=passed, split_date=split, max_weight_difference=max_difference,
        n_differing_cells=differing, n_cells_checked=int(left.size),
        future_weights_changed=future_changed, message=message,
        first_difference_date=first_difference,
    )


def run_leakage_suite(market: MarketData, builders: dict[str, Callable[[MarketData], pd.DataFrame]],
                      split_dates=("2012-06-29", "2017-06-30", "2021-06-30"),
                      modes=("shock", "reverse"), seed: int = 99) -> pd.DataFrame:
    """Run the test for every strategy at several split dates and modes."""
    rows = []
    for name, builder in builders.items():
        for split in split_dates:
            for mode in modes:
                try:
                    result = check_no_lookahead(market, builder, split, seed=seed, mode=mode)
                    row = result.to_row()
                except Exception as exc:
                    row = {"split_date": str(split), "passed": False, "message": f"error: {exc}",
                           "max_weight_difference": np.nan, "n_differing_cells": -1,
                           "n_cells_checked": 0, "future_weights_changed": False,
                           "first_difference_date": ""}
                row.update({"strategy": name, "mode": mode})
                rows.append(row)
    frame = pd.DataFrame(rows)
    front = ["strategy", "mode", "split_date", "passed", "max_weight_difference",
             "n_differing_cells", "future_weights_changed", "message"]
    return frame.loc[:, front + [c for c in frame.columns if c not in front]]


def check_signal_causality(signal_fn: Callable[[MarketData], pd.DataFrame], market: MarketData,
                           split_date, tolerance: float = 1e-10, seed: int = 99) -> LeakageTestResult:
    """The same test applied to a raw signal rather than to weights.

    Running it at the signal level as well as the portfolio level localises a
    failure: if the signal passes and the book fails, the bug is in execution.
    """
    return check_no_lookahead(market, signal_fn, split_date, tolerance, seed)


def detect_suspicious_alignment(signal: pd.DataFrame, returns: pd.DataFrame,
                                max_lead: int = 3, implausible: float = 0.30) -> pd.DataFrame:
    """Profile a signal's correlation against returns at leads and lags.

    A complementary smoke test to the perturbation test above. What it is
    looking for is an *implausible magnitude* at a non-negative lead, not
    simply a peak in the past: a momentum signal is a function of past
    returns, so correlating strongly with them is its definition, not a bug.

    The two things that are genuinely suspicious:

    ``lead > 0`` with a large correlation -- no real signal predicts daily
        returns with |rho| above ~0.3; a value near 1 means the future return
        was used to build the signal.
    ``lead == 0`` with a large correlation -- the classic off-by-one, where
        the signal was built from the same day's return it is meant to
        predict.

    Past correlations (``lead < 0``) are reported for context and never
    flagged.
    """
    rows = []
    for lead in range(-max_lead, max_lead + 1):
        shifted = returns.shift(-lead)
        aligned_signal, aligned_returns = signal.align(shifted, join="inner")
        daily = aligned_signal.corrwith(aligned_returns, axis=1)
        value = float(daily.mean())
        flagged = lead >= 0 and abs(value) > implausible
        rows.append(
            {
                "lead": lead,
                "interpretation": ("signal vs FUTURE return" if lead > 0 else
                                   "signal vs SAME-DAY return" if lead == 0 else
                                   "signal vs PAST return (expected for momentum)"),
                "mean_correlation": value,
                "abs_mean_correlation": abs(value),
                "implausible": flagged,
            }
        )
    frame = pd.DataFrame(rows)
    forward = frame[frame["lead"] >= 0]
    strictly_future = frame[frame["lead"] > 0]
    same_day = frame[frame["lead"] == 0]

    # The two flags mean different things and call for different responses.
    if bool(strictly_future["implausible"].any()):
        diagnosis = ("LEAKAGE: the signal is implausibly correlated with a strictly future "
                     "return, so a future observation was used to build it")
    elif bool(same_day["implausible"].any()):
        diagnosis = ("CONTEMPORANEOUS: the signal contains the same day's return, which is "
                     "legitimate for a price-level feature but makes an execution lag of at "
                     "least one day mandatory -- trading the close you measured would capture "
                     "a mechanical, untradable correlation")
    else:
        diagnosis = "CLEAN: no implausible correlation at a non-negative lead"

    frame["suspicious"] = bool(forward["implausible"].any())
    frame["leakage"] = bool(strictly_future["implausible"].any())
    frame["worst_forward_correlation"] = float(forward["abs_mean_correlation"].max())
    frame["threshold"] = implausible
    frame["diagnosis"] = diagnosis
    return frame
