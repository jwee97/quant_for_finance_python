"""Numerical building blocks that the pricing, PDE and state-space code share.

    linsolve    solvers for the linear systems that finite-difference schemes produce: the Thomas algorithm and LAPACK's tridiagonal LU, Jacobi, Gauss-Seidel, successive
                over-relaxation (SOR), and projected SOR and the Brennan-Schwartz algorithm for the linear complementarity problem behind American options
"""

from . import linsolve
from .linsolve import (brennan_schwartz, gauss_seidel, jacobi, optimal_omega, psor_tridiagonal, sor, sor_tridiagonal, thomas, tridiagonal_factor, tridiagonal_solve)

__all__ = ["linsolve", "thomas", "tridiagonal_factor", "tridiagonal_solve", "jacobi", "gauss_seidel", "sor", "sor_tridiagonal", "psor_tridiagonal", "brennan_schwartz", "optimal_omega"]
