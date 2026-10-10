"""Scenario generation: ways to produce plausible futures for a set of assets, and a common yardstick for how plausible they are.

``generate(name, returns, n, horizon, seed, **params)`` returns an array of shape ``(n, horizon, assets)`` of daily simple returns from any of the generators in :mod:`src.scenarios.generators`
(historical windows, bootstrap, copula, risk-factor, ARIMA-GARCH, WGAN-GP, factor VAE, diffusion); :mod:`src.scenarios.quality` scores a set against the history it was made from.
"""

from .generators import GENERATORS, generate  # noqa: F401
from .quality import compare, quality_report  # noqa: F401
