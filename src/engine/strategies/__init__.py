"""Modern strategy families on the common strategy API: trend, carry, basis, relative value, volatility (and swap strategies in ``src.swaps``)."""

from .basis import BasisStrategy
from .carry import CarryStrategy
from .ensemble import EnsembleStrategy, HMMRegime, VolRegime
from .ml import ForecastCombinationStrategy, MetaLabelStrategy, WalkForwardMLStrategy, momentum_forecaster, range_forecaster, reversal_forecaster
from .relative_value import RelativeValueStrategy
from .trend import TrendStrategy
from .volatility import VolatilityPremiumStrategy

__all__ = ["BasisStrategy", "CarryStrategy", "EnsembleStrategy", "ForecastCombinationStrategy", "MetaLabelStrategy", "WalkForwardMLStrategy", "momentum_forecaster", "range_forecaster", "reversal_forecaster", "HMMRegime", "VolRegime", "RelativeValueStrategy", "TrendStrategy", "VolatilityPremiumStrategy"]
