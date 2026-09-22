"""The model ladder M0-M10 (spec §55), as weight builders.

Every model is a function ``MarketData -> target weights`` so that all of
them can be run through one engine, one cost model and one leakage test. The
ladder is deliberately incremental: each step adds exactly one idea, which is
what makes the final comparison able to answer "where does the improvement
actually come from?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.momentum import total_return_momentum, volatility_scaled_momentum
from src.features.mean_reversion import price_zscore
from src.features.volatility import ewma_volatility, rolling_volatility
from src.models.regression import signal_to_expected_returns
from src.portfolio.black_litterman import black_litterman_weights
from src.portfolio.constraints import Constraints
from src.portfolio.covariance import estimate_covariance
from src.portfolio.cvar_optimize import mean_cvar_weights
from src.portfolio.equal_weight import equal_weight_book
from src.portfolio.inverse_vol import inverse_vol_book
from src.portfolio.mean_variance import mean_variance_weights
from src.portfolio.risk_parity import risk_parity_book
from src.signals.combine import combine_signals
from src.signals.transform import signal_to_positions
from src.utils.dates import rebalance_dates
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def transform_config(config) -> dict:
    node = config.get("strategies.transform", {}) or {}
    return {
        "winsorize_quantile": float(node.get("winsorize_quantile", 0.02)),
        "cross_sectional": bool(node.get("cross_sectional", True)),
        "scale": str(node.get("scale", "zscore")),
        "clip": node.get("clip", 3.0),
        "target_vol": float(config.get("portfolio.volatility.target_vol", 0.10)),
        "max_abs_weight": float(node.get("max_abs_weight", 0.25)),
        "gross_leverage": float(node.get("gross_leverage", 1.0)),
        "long_only_book": bool(node.get("long_only", False)),
    }


# ---------------------------------------------------------------------------
# Signal books (M3-M5)
# ---------------------------------------------------------------------------
def momentum_signal(market, config) -> pd.DataFrame:
    variant = config.get("strategies.momentum.primary_variant", "vol_scaled")
    lookback = int(config.get("strategies.momentum.primary_lookback", 126))
    skip = int(config.get("strategies.momentum.skip_days", 1))
    vol_lookback = int(config.get("strategies.momentum.vol_lookback", 63))
    if variant == "vol_scaled":
        signal = volatility_scaled_momentum(market.prices, market.returns(), lookback, skip, vol_lookback)
    else:
        signal = total_return_momentum(market.prices, lookback, skip)
    return signal.where(market.investable)


def mean_reversion_signal(market, config) -> pd.DataFrame:
    lookback = int(config.get("strategies.mean_reversion.primary_lookback", 21))
    sign = int(config.get("strategies.mean_reversion.sign", -1))
    basis = config.get("strategies.mean_reversion.price_basis", "log")
    return (sign * price_zscore(market.prices, lookback, basis)).where(market.investable)


def combined_signal(market, config) -> pd.DataFrame:
    return combine_signals(
        {"momentum": momentum_signal(market, config),
         "reversion": mean_reversion_signal(market, config)},
        {"momentum": 0.5, "reversion": 0.5},
    )


def signal_book(market, config, which: str = "momentum") -> pd.DataFrame:
    builders = {"momentum": momentum_signal, "mean_reversion": mean_reversion_signal,
                "combined": combined_signal}
    signal = builders[which](market, config)
    volatility = rolling_volatility(market.returns(), int(config.get("portfolio.volatility.lookback", 63)))
    return signal_to_positions(signal, volatility, investable=market.investable,
                               **transform_config(config))


# ---------------------------------------------------------------------------
# Optimised books (M6-M9)
# ---------------------------------------------------------------------------
def _optimised_book(market, config, objective: str, covariance_method: str = "shrinkage",
                    use_signal: bool = True) -> pd.DataFrame:
    """Re-solve an optimiser on every rebalance date, on trailing data only."""
    returns = market.returns()
    index = pd.DatetimeIndex(returns.index)
    frequency = str(config.get("backtest.engine.rebalance", "monthly"))
    marks = rebalance_dates(index, frequency)
    lookback = int(config.get("portfolio.covariance.lookback", 252))
    min_assets = int(config.get("backtest.engine.min_assets", 5))
    constraints = Constraints.from_config(config)
    risk_aversion = float(config.get("portfolio.mean_variance.risk_aversion", 5.0))
    halflife = float(config.get("portfolio.covariance.ewma_halflife", 60.0))

    signal = combined_signal(market, config) if use_signal else None
    mu_all = None
    if signal is not None:
        mu_all = signal_to_expected_returns(
            signal,
            target_spread=float(config.get("portfolio.mean_variance.mu_scale", 0.05)),
            shrinkage=float(config.get("portfolio.mean_variance.mu_shrinkage", 0.5)),
            periods_per_year=1,      # keep mu annualised for the optimiser
        )

    book = pd.DataFrame(np.nan, index=index, columns=returns.columns)
    failures = 0
    for stamp in marks:
        position = index.get_loc(stamp)
        if position < lookback:
            continue
        window = returns.iloc[position - lookback + 1:position + 1]
        live = [c for c in returns.columns
                if bool(market.investable.loc[stamp, c]) and window[c].notna().sum() > lookback * 0.8]
        if len(live) < min_assets:
            continue
        sample = window[live].dropna(how="any")
        if len(sample) < lookback * 0.6:
            continue
        try:
            cov = estimate_covariance(sample, covariance_method, lookback, halflife,
                                      target=config.get("portfolio.covariance.shrinkage_target",
                                                        "constant_correlation"),
                                      annualise=True)
            mu = (mu_all.loc[stamp, live] if mu_all is not None
                  else pd.Series(0.0, index=live))
            mu = mu.fillna(0.0)

            if objective == "mvo":
                result = mean_variance_weights(mu, cov, risk_aversion, constraints)
                weights = result.weights if result.success else None
            elif objective == "black_litterman":
                market_weights = pd.Series(1.0 / len(live), index=live)
                result = black_litterman_weights(
                    cov, market_weights, signal.loc[stamp, live] if signal is not None else None,
                    tau=float(config.get("portfolio.black_litterman.tau", 0.05)),
                    risk_aversion=float(config.get("portfolio.black_litterman.risk_aversion", 2.5)),
                    view_confidence=float(config.get("portfolio.black_litterman.view_confidence", 0.3)),
                    constraints=constraints,
                )
                weights = result.weights if result.success else None
            elif objective == "mean_cvar":
                scenarios = returns[live].iloc[
                    max(position - int(config.get("portfolio.mean_cvar.scenarios", 750)) + 1, 0):position + 1
                ].dropna(how="any")
                weights = mean_cvar_weights(
                    scenarios, mu, float(config.get("portfolio.mean_cvar.alpha", 0.95)),
                    None, risk_aversion, constraints, mu_is_annualised=True,
                )
            else:
                raise ValueError(f"unknown objective '{objective}'")
        except Exception as exc:
            failures += 1
            if failures <= 3:
                LOGGER.warning("%s optimisation failed at %s: %s", objective, stamp.date(), exc)
            continue
        if weights is not None:
            book.loc[stamp, live] = weights.reindex(live).to_numpy()

    if failures:
        LOGGER.warning("%s: %d of %d rebalances failed to solve", objective, failures, len(marks))
    return book.ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------
def build_ladder(market, config, include: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Return ``{model_name: target weights}`` for the requested models."""
    constraints = Constraints.from_config(config)
    index = pd.DatetimeIndex(market.prices.index)
    frequency = str(config.get("backtest.engine.rebalance", "monthly"))
    marks = rebalance_dates(index, frequency)

    builders = {
        "M0_equal_weight": lambda: equal_weight_book(market.investable, constraints),
        "M1_inverse_vol": lambda: inverse_vol_book(
            market.returns(), market.investable,
            int(config.get("portfolio.volatility.lookback", 63)), "ewma",
            float(config.get("portfolio.volatility.ewma_halflife", 40.0)), constraints,
        ),
        "M2_risk_parity": lambda: risk_parity_book(
            market.returns(), market.investable,
            int(config.get("portfolio.covariance.lookback", 252)), marks,
            config.get("portfolio.covariance.method", "shrinkage"),
            config.get("portfolio.risk_parity.method", "newton"), constraints,
            int(config.get("backtest.engine.min_assets", 5)),
            float(config.get("portfolio.covariance.ewma_halflife", 60.0)),
        ),
        "M3_momentum": lambda: signal_book(market, config, "momentum"),
        "M4_mean_reversion": lambda: signal_book(market, config, "mean_reversion"),
        "M5_momentum_plus_mr": lambda: signal_book(market, config, "combined"),
        "M6_combined_alpha_mvo": lambda: _optimised_book(market, config, "mvo", "sample"),
        "M7_combined_alpha_shrinkage_mvo": lambda: _optimised_book(market, config, "mvo", "shrinkage"),
        "M8_black_litterman": lambda: _optimised_book(market, config, "black_litterman", "shrinkage"),
        "M9_mean_cvar": lambda: _optimised_book(market, config, "mean_cvar", "shrinkage"),
    }
    names = include or list(builders)
    out = {}
    for name in names:
        if name not in builders:
            LOGGER.warning("unknown model '%s', skipping", name)
            continue
        LOGGER.info("building %s", name)
        out[name] = builders[name]()
    return out


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def cached_ladder(market, config, directory, include: list[str] | None = None,
                  rebuild: bool = False) -> dict[str, pd.DataFrame]:
    """Build the ladder once and reuse it across stages.

    The cache key is the config fingerprint and the data version, so a changed
    parameter or a re-downloaded dataset invalidates it automatically rather
    than silently serving stale books.
    """
    from pathlib import Path

    directory = Path(directory) / "books"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = directory / "cache_key.txt"
    key = f"{config.fingerprint()}|{market.data_version}"
    if rebuild or not stamp.exists() or stamp.read_text().strip() != key:
        for path in directory.glob("*.csv"):
            path.unlink()
        stamp.write_text(key)

    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    candidates = include or [
        "M0_equal_weight", "M1_inverse_vol", "M2_risk_parity", "M3_momentum",
        "M4_mean_reversion", "M5_momentum_plus_mr", "M6_combined_alpha_mvo",
        "M7_combined_alpha_shrinkage_mvo", "M8_black_litterman", "M9_mean_cvar",
    ]
    for name in candidates:
        path = directory / f"{name}.csv"
        if path.exists():
            out[name] = pd.read_csv(path, index_col=0, parse_dates=[0])
        else:
            missing.append(name)

    if missing:
        LOGGER.info("building %d uncached books: %s", len(missing), ", ".join(missing))
        built = build_ladder(market, config, missing)
        for name, frame in built.items():
            frame.to_csv(directory / f"{name}.csv", index_label="date")
            out[name] = frame
    return {name: out[name] for name in candidates if name in out}
