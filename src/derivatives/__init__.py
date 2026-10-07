"""Derivatives: option pricing, Greeks, implied volatility, surfaces, a synthetic option market, an options backtester and volatility strategies.

    pricing      BSM, Black-76, Bachelier, binomial (American), Longstaff-Schwartz, Heston, Merton jump diffusion, SABR, calibration
    greeks       closed-form and numerical Greeks, portfolio aggregation, P&L explain
    iv           implied volatility (vectorised safeguarded Newton, Black-76, Bachelier, American)
    schema       option-chain data model, validation, static no-arbitrage checks, put-call parity forward
    surface      SVI / SSVI surfaces, arbitrage conditions, Dupire local volatility, risk-neutral density, variance swaps, VIX-style index
    synthetic    a reproducible stochastic-volatility-with-jumps option market (the stand-in for unavailable chain data) and a vendor CSV loader
    backtest     the options backtester: fills at bid/ask with a lag, hedging, settlement, Greek attribution
    strategies   short put, covered call, strangle, iron condor, delta-hedged straddle, risk reversal, calendar spread, VRP-timed wrapper
    volatility   VIX series, variance risk premium, term structure, smile summary, dispersion trading simulation
"""

from . import backtest, greeks, iv, pricing, schema, strategies, surface, synthetic, volatility
from .backtest import Hedge, Order, OptionBacktester
from .greeks import bsm_greeks, numerical_greeks
from .iv import implied_vol
from .pricing import bsm_price
from .synthetic import SyntheticMarketSpec, generate_market
