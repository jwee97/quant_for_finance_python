---
title: "Finite-difference and integro-differential solvers, and the linear systems under them"
slug: finite-difference-solvers-and-linear-systems
difficulty: 3
chapter: Ch. 18
prerequisites: [option-pricing-and-greeks]
stages: []
files: [src/derivatives/pde.py, src/derivatives/pide.py, src/numerics/linsolve.py]
figures: []
tests: [tests/test_pde.py, tests/test_pide.py, tests/test_numerics.py]
models: []
---

# Finite-difference and integro-differential solvers, and the linear systems under them

## In one sentence

An option's value solves a differential equation on a grid of prices, and how you step it in time, where you put the payoff on the grid and how you solve the system at each step decide whether you get second-order accuracy, a price that wiggles, or a wrong answer.

## The idea

With `x = ln S` and `tau = T - t` the Black-Scholes value solves `V_tau = (sigma^2/2) V_xx + (r - q - sigma^2/2) V_x - r V`, a constant-coefficient diffusion: the spatial operator is the same tridiagonal matrix at every step, so it is factored once. The **theta method** steps `(I - theta dt L) V_new = (I + (1 - theta) dt L) V_old`: `theta = 1` (implicit Euler) is first order in time and never oscillates, `theta = 1/2` (Crank-Nicolson) is second order but rings on a payoff with a kink or a jump. Two cures give second order for the payoffs that matter: start with a few implicit half-steps (**Rannacher smoothing**), and set the initial condition to the **average of the payoff over each cell**, not its value at the node (Pooley, Vetzal and Forsyth 2003), which is exact for a kink (a call, a put) and for a jump (a digital).

A model with jumps adds a **non-local term**: the value at one price depends on the values at all others, `Int [V(x + z) - V(x) - z V_x] nu(dz)`. The **PIDE** scheme of Cont and Voltchkova (2005) treats the jumps smaller than half a cell as extra diffusion (which is what makes the infinite-activity variance gamma and NIG computable), sums the larger ones on the grid as a correlation with a fixed kernel, and steps the local part implicitly and the jump sum explicitly (Crank-Nicolson / Adams-Bashforth: second order in time and stable for the implicit part). The drift is not taken from a formula but **solved from the martingale condition on the grid itself**, so the discounted price is a martingale for the scheme however coarse the grid.

Each implicit step is a linear system. For a tridiagonal matrix the **Thomas algorithm** is Gaussian elimination in `O(n)`; LAPACK's pivoting factorisation is the same and robust. For a general matrix there are the classical iterations, **Jacobi**, **Gauss-Seidel** and **successive over-relaxation (SOR)**. For an **American** option the step is a *linear complementarity problem*, `x >= g`, `M x >= d`, `(M x - d)'(x - g) = 0`: solved iteratively by **projected SOR** (Cryer 1971), and directly in one pass by the **Brennan-Schwartz** sweep when the exercise region is one interval at one end of the grid.

## Why it matters

A grid prices what a formula cannot (early exercise, barriers, payoffs with no transform), delivers delta, gamma and theta at no extra cost and exposes the numerical choices that a library call hides. The same few ideas (implicit steps, smoothing a kink, a martingale-preserving drift, an M-matrix) also explain why some solvers return garbage on perfectly reasonable inputs.

## How this repo uses it

`derivatives/pde.py` solves vanillas (European and American), knock-out barriers and digitals; `derivatives/pide.py` solves vanillas, with early exercise, under Merton, Kou, variance gamma and NIG; `numerics/linsolve.py` has `thomas`, `tridiagonal_factor`, `jacobi`, `gauss_seidel`, `sor` (with `optimal_omega`), `sor_tridiagonal`, `psor_tridiagonal` and `brennan_schwartz`. Every iterative routine returns the residual after each sweep so that convergence can be plotted. The Options tab's *Model prices* view uses the American solvers for the early-exercise price and boundary.

## What we found

- **Second order is measured, not claimed.** The PDE error falls by four when the grid is refined by two; implicit Euler is first order in time; Crank-Nicolson without the Rannacher start rings at a kink and with it does not; cell averaging is exact for a kink and a jump.
- **The PIDE agrees with the Fourier price,** which is exact to 1e-10: to about 1.5e-4 for Black-Scholes and 4e-4 for Merton and Kou at the default grid, 1.5e-3 for the pure-jump models, whose small jumps are only approximated. Two things were needed for the infinite-activity models: widening the band of jumps treated as diffusion until the explicit jump term is not stiff, and absorbing into the diffusion the variance the midpoint rule overstates in each cell (about `nu_j dx^2 / 12`), which restores second order for the NIG.
- **Brennan-Schwartz is exact, and projected SOR agrees with it.** Both match an exhaustive search over the active sets; the sweep handles an exercise region at the high end (a call on a dividend payer) by mirroring the grid; with an obstacle below the solution the constrained solution is the unconstrained one.
- **Gauss-Seidel needs about half the sweeps of Jacobi and the optimal SOR parameter far fewer;** SOR on a tridiagonal matrix matches the dense one.
- **Early exercise needs an M-matrix.** The tridiagonal matrix is one when the cell Peclet number `b dx / sigma^2` is at most one; the American solvers refine the grid until it is (the tests use a small volatility with a large drift, where twenty cells are not enough, and check that the refined solution is a valid American value). The PIDE's early-exercise price agrees with an independent Longstaff-Schwartz simulation.

## Going deeper

```
theta step:          (I - theta dt L) V^{n+1} = (I + (1 - theta) dt L) V^n               theta = 1/2: Crank-Nicolson
Rannacher:           the first two steps are four implicit half-steps
cell average:        V^0_i = (1/dx) Int_{x_i - dx/2}^{x_i + dx/2} payoff(e^x) dx           split at a kink or jump inside the cell
PIDE drift:          b = ( (r - q) - sigma_a^2 g2 / 2 - Sum_j nu_j (e^{z_j} - 1) ) / g1       g1 = sinh(dx)/dx, g2 = 2 (cosh(dx) - 1) / dx^2
Thomas:              forward sweep  w = a_i / b_{i-1};  b_i -= w c_{i-1};  d_i -= w d_{i-1};   then back substitution
Brennan-Schwartz:    the back substitution clipped at the obstacle:  x_i = max(g_i, (d_i - c_i x_{i+1}) / b_i)
M-matrix (central differences):   sigma^2 >= |b| dx
```

## Pitfalls

- **Crank-Nicolson alone is not safe for a kink.** It is second order and it rings; delta and gamma near the strike show it first.
- **A grid too coarse for the cell Peclet number breaks the guarantee of the Brennan-Schwartz sweep.** The solvers here refine the grid themselves (the result says how many cells it used), a grid you build by hand does not; and a price that has not converged under grid doubling is not a price.
- **The PIDE's accuracy for infinite-activity models is limited by the small jumps,** and the jump term is explicit: a time step that is too long makes it stiff.
- Gauss-Seidel and Jacobi need diagonal dominance (SOR a positive-definite matrix with `0 < omega < 2`); they are here to read and to check the direct solvers, not for production grids.
- The projected iteration makes no assumption about where the exercise region is; Brennan-Schwartz does. For an American option with an exercise region in the middle of the grid, use the iteration.

## Try it

```bash
python -m pytest tests/test_pde.py tests/test_pide.py tests/test_numerics.py -q
python - <<'EOF'
from src.derivatives import pde, pide, levy
print(pde.vanilla(100, 100, 1.0, 0.04, 0.01, 0.25, call=False, american=True).price)
print(pide.american(levy.Merton(sigma=0.15, lam=1.0, mu_j=-0.1, sigma_j=0.15), 100, 100, 1.0, 0.04, 0.01).price)
EOF
```
