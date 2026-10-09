"""Algorithmic trading and best execution: how to work a parent order, and how to tell whether it was done well.

    market       a stylised day (volume and variance profiles, spread) and the Order and Scenario an algorithm is tried on
    impact       temporary and permanent impact, the expected cost and timing risk of a schedule, and Kissell's I-Star pre-trade estimate
"""

from . import impact, market
from .market import SCENARIOS, Market, Order, Scenario

__all__ = ["Market", "Order", "Scenario", "SCENARIOS", "impact", "market"]
