"""Market microstructure: order flow and liquidity estimators, a limit-order-book simulator, market making, and optimal execution.

    order_flow     trade classification, order-flow imbalance, Kyle lambda, Amihud, Roll, Corwin-Schultz, effective/realized spread, VPIN
    lob            a zero-intelligence Poisson limit order book (Cont-Stoikov-Talreja)
    market_making  Avellaneda-Stoikov quotes and a Monte Carlo market maker with inventory limits and adverse selection
    execution      Almgren-Chriss optimal schedule, efficient frontier, TWAP/VWAP, implementation shortfall
"""

from . import execution, lob, market_making, order_flow
