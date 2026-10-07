"""The event-driven multi-asset engine. See ``engine`` for the architecture, ``strategy`` for the strategy API, ``lifecycle`` for contract lifecycle handling."""

from .analysis import BacktestResult, compare
from .constraints import ConstraintInputs, ConstraintSet
from .costs import CommissionModel, CostBreakdown, CostSchedule, FinancingModel, ImpactModel, LiquidityModel, SlippageModel, SpreadModel
from .data import PITData
from .engine import Engine, EngineConfig
from .events import EventLog, EventQueue, SimulationClock
from .execution import FillResult, Quote, simulate_fill
from .marks import MARK_MODELS, MarkProvider, MarkResult, register_mark_model
from .orders import Order, validate_order
from .strategy import InstrumentEvent, PortfolioView, Schedule, Signal, Strategy, StrategyContext, Target

__all__ = [n for n in dir() if not n.startswith("_")]
