"""Probability tools for research and risk.

    resampling   block / stationary / wild bootstraps, Politis-White block length, subsampling
    montecarlo   path simulators (GBM, OU, jump diffusion, GARCH), variance reduction, quasi-Monte Carlo, importance sampling, cross-entropy
    evt          generalised Pareto (peaks over threshold), GEV, Hill, conditional EVT VaR/ES, threshold diagnostics
    copulas      Gaussian, Student, Clayton, Gumbel, Frank copulas: fit, simulate, tail dependence, copula-marginal scenarios
    ruin         drawdown distribution, ruin and Kelly sizing
    markov       Markov chains, transition-matrix estimation and tests, Markov-switching regression
    bayes        conjugate regression, Bayesian Sharpe, Metropolis-Hastings with diagnostics
"""

from . import bayes, copulas, evt, markov, montecarlo, resampling, ruin
from .resampling import (bootstrap_indices, bootstrap_statistic, circular_block_indices, iid_indices, moving_block_indices, optimal_block_length,
                         stationary_indices, subsample_ci, subsample_statistics, wild_bootstrap_multipliers)
