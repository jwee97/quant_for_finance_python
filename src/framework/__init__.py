"""The research framework: regimes, forecasts with confidence, combination, allocation, adaptive risk, validation, experiments.

Importing the package registers the built-in detectors and allocators; ``src.strategies`` registers the strategy library
(imported lazily by ``load_library``) and anything you register yourself is found the same way.
"""

from . import allocation, allocators_portfolio, forecasting, regimes, risk  # noqa: F401  (importing registers the built-ins)
from .data import MarketBundle, bundle_from_market, bundle_from_prices, load_default_bundle, load_prices_csv  # noqa: F401
from .forecasting import ForecastModel, combine_forecasts, score_to_forecast  # noqa: F401
from .pipeline import Pipeline, PipelineResult, PipelineSpec  # noqa: F401
from .registry import ALLOCATORS, DETECTORS, MODELS, register_allocator, register_detector, register_model  # noqa: F401
from .types import Forecast, ForecastPanel, Regime, RegimeSeries  # noqa: F401


def load_library() -> None:
    """Import the strategy library so its models are registered."""
    import importlib

    try:
        importlib.import_module("src.strategies")
    except ModuleNotFoundError as error:                       # the library is optional; anything else is a real error
        if error.name != "src.strategies":
            raise
        return
    from ..strategies.user import load_user_strategies
    from ..utils.config import project_root

    load_user_strategies(project_root())                   # the user_strategies/ folder, if there is one
