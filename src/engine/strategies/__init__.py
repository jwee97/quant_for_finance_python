"""Modern strategy families on the common strategy API: trend, carry, basis, relative value, volatility (and swap strategies in ``src.swaps``)."""

from .basis import BasisStrategy
from .carry import CarryStrategy
from .ensemble import EnsembleStrategy, HMMRegime, VolRegime
from .relative_value import RelativeValueStrategy
from .trend import TrendStrategy
from .volatility import VolatilityPremiumStrategy

__all__ = ["BasisStrategy", "CarryStrategy", "EnsembleStrategy", "HMMRegime", "VolRegime", "RelativeValueStrategy", "TrendStrategy", "VolatilityPremiumStrategy"]
