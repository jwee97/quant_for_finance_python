"""Shared fixtures.

Tests run on small synthetic panels by default so the suite is fast and does
not depend on the network or on the downloaded dataset. The few tests that
need the real data skip cleanly when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20240101)


@pytest.fixture(scope="session")
def dates():
    return pd.bdate_range("2015-01-01", periods=1200)


@pytest.fixture(scope="session")
def tickers():
    return ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture(scope="session")
def synthetic_returns(dates, tickers, rng):
    """Correlated returns with deliberately different volatilities."""
    n, k = len(dates), len(tickers)
    common = rng.normal(0.0, 0.008, size=(n, 1))
    idiosyncratic = rng.normal(0.0, 0.006, size=(n, k))
    scale = np.array([0.4, 0.8, 1.0, 1.6, 2.5])
    values = (common * scale + idiosyncratic * scale) + 0.0003
    return pd.DataFrame(values, index=dates, columns=tickers)


@pytest.fixture(scope="session")
def synthetic_prices(synthetic_returns):
    return 100.0 * (1.0 + synthetic_returns).cumprod()


@pytest.fixture(scope="session")
def synthetic_market(synthetic_prices, synthetic_returns, tickers):
    from src.data.loader import MarketData

    prices = synthetic_prices
    investable = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    # One asset lists late, so investability is exercised rather than assumed.
    investable.iloc[:200, investable.columns.get_loc("EEE")] = False
    prices = prices.where(investable)
    return MarketData(
        prices=prices,
        close=prices,
        adjustment_ratio=pd.DataFrame(1.0, index=prices.index, columns=prices.columns),
        volume=pd.DataFrame(1e6, index=prices.index, columns=prices.columns),
        high=prices * 1.005,
        low=prices * 0.995,
        open_=prices,
        filled_mask=pd.DataFrame(False, index=prices.index, columns=prices.columns),
        investable=investable,
        data_version="test",
        asset_class={t: "equity" for t in tickers},
        group={t: "grp" for t in tickers},
    )


@pytest.fixture(scope="session")
def real_market():
    """The real cleaned panel, or a skip when it has not been built."""
    from src.data.loader import MarketData
    from src.utils.config import load_config

    config = load_config()
    if not (config.path("processed") / "prices_adjusted.csv").exists():
        pytest.skip("processed data not built; run `python -m experiments.stage01_data`")
    return MarketData.from_processed(config.path("processed"), config.path("metadata"),
                                     config.asset_class_map, config.group_map)
