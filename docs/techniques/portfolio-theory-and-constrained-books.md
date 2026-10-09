---
title: "Portfolio theory and constrained books: frontier, CAPM and APT, 130/30 and neutral books, CVaR limits and QUBO selection"
slug: portfolio-theory-and-constrained-books
difficulty: 3
chapter: Platform
prerequisites: [mean-variance-and-shrinkage, mean-cvar, costs-constraints-and-capacity, alpha-model-construction]
stages: []
files: [src/equity/qp.py, src/equity/theory.py, src/equity/portfolio.py, src/equity/cvar_portfolio.py, src/equity/qubo.py, src/framework/allocators_book.py, src/framework/allocators_theory.py, experiments/portfolio_world.py]
figures: []
tests: [tests/test_equity_qp.py, tests/test_equity_theory.py, tests/test_equity_portfolio.py, tests/test_equity_cvar_portfolio.py, tests/test_equity_qubo.py, tests/test_allocators_book.py, tests/test_allocators_theory.py]
models: []
---

# Portfolio theory and constrained books: frontier, CAPM and APT, 130/30 and neutral books, CVaR limits and QUBO selection

## In one sentence

The closed-form theory of portfolios (the efficient frontier, the tangency portfolio, the capital market line, CAPM with and without a risk-free asset, the APT) next to the machinery a manager uses to turn a ranking into a book (130/30, market, dollar, sector and beta neutral, turnover limits and trading costs), a tail-risk limit instead of a variance penalty, and best-K selection written as the QUBO a quantum annealer takes.

## The idea

**Mean-variance in closed form.** With expected returns `mu` and covariance `Sigma` and no constraints, everything is linear algebra. Four numbers describe the whole frontier: `A = 1'Sigma^-1 1`, `B = 1'Sigma^-1 mu`, `C = mu'Sigma^-1 mu`, `D = AC - B^2`. The minimum-variance portfolio is `Sigma^-1 1 / A`; the frontier portfolio with expected return `m` has variance `(A m^2 - 2 B m + C) / D`; any two frontier portfolios span it (two-fund separation). Add a risk-free rate and the best risky portfolio is the *tangency* portfolio `Sigma^-1 (mu - rf)`, the same for every investor; the *capital market line* joins cash to it, and risk aversion enters only in how much of the tangency portfolio to hold, `(mu_T - rf) / (gamma sigma_T^2)` of wealth, the rest in cash or borrowed. The weights are the most error-sensitive portfolio there is, which is what the mean-variance guide is about.

**CAPM and its zero-beta version.** If everyone holds the tangency portfolio it is the market, and `E[R_i] - rf = beta_i (E[R_m] - rf)`: the *security market line*. `capm_regression` estimates each asset's alpha and beta with Newey-West errors; `security_market_line` is the cross-sectional test (betas first, then a regression of mean returns on betas, with Shanken's correction for the betas being estimated). Black (1972) showed the model survives without a risk-free asset: the line's intercept is then the expected return of the *zero-beta portfolio*, the minimum-variance portfolio uncorrelated with the market (`zero_beta_portfolio`), and its slope is the market premium *less* that intercept.

**APT.** Ross's arbitrage pricing theory needs no market portfolio: if returns follow a factor model and nothing earns a premium for free, `E[R] = rf + B lambda`. `apt_two_pass` estimates the loadings and premia and tests that the pricing errors are jointly zero (`alpha_chi2`); `apt_arbitrage_portfolio` is how a mispricing would be traded, the cheapest portfolio with zero net investment and zero exposure to every factor whose expected return is the pricing errors it holds. With no named factors, `statistical_factors` takes the first K principal components as portfolios (their *raw* returns, whose means are the premia).

**Constrained books.** A forecast ranks stocks; a book decides how to use the ranking. Splitting the weight into a long part and a short part, `w = w+ - w-` with both non-negative, turns what a portfolio manager specifies into linear constraints on a quadratic objective: gross long and gross short (`130_30` is 130% and 30%, `market_neutral` 100% and 100%, `dollar_neutral` 50% and 50%), position limits, the net weight of each sector within a tolerance of its target, `|beta'w|` within a tolerance of its target, any other exposure inside bounds, and, with one extra variable per name for `|w - w0|`, a turnover limit and a proportional trading cost in the objective. The programme maximises `alpha'w - (gamma/2) w'Sigma w - cost'|w - w0|`. A tiny penalty on gross exposure makes the optimum unique, and a second pass gives every name one side, because a name held long and short at once meets a gross target on paper and not in fact.

**Tail risk.** Variance does not see the tail. CVaR at level `a` is the average loss in the worst `1 - a` of the scenarios. Rockafellar and Uryasev (2000) showed it is `min over zeta of zeta + E[(loss - zeta)+] / (1 - a)`, so a CVaR limit is linear once each scenario's shortfall is a variable. The dual form is cheaper: `CVaR(w)` is the maximum over weightings `q` of the scenarios of `-q'Rw`, each `q` gives a valid linear cut, and Kelley's cutting-plane method adds the most violated cut until none is. As the limit tightens the portfolio leaves the assets with fat left tails and goes to cash.

**QUBO.** A quantum annealer finds low-energy states of `x'Qx` over bits. A portfolio becomes one in two standard ways: weights on a grid of `2^b` values per asset with the budget as a penalty `P (sum w - 1)^2`, or *cardinality* (hold exactly `K` of `n` assets equally, penalty `P (sum x - K)^2`). Picking the best `K` of `n` is combinatorial, which is why it is the textbook case.

## Why it matters

Each of these is a decision made before any forecast: which risky portfolio, how much of it, what the book is allowed to be exposed to, and how much trading a forecast is worth. The same forecast produces a different strategy under each, and the differences are mostly structural (exposure, beta, turnover, cost) rather than a matter of returns, which is why they can be tested.

## How this repo uses it

The numerical core is `src/equity/qp.py`: a quadratic-programme solver in the style of OSQP (ADMM with Ruiz scaling and a polish step, for the large sparse programmes of the books) and a Mehrotra interior-point method (for the small, degenerate programmes the CVaR cuts produce, where a first-order method creeps); no new dependency. `src/equity/theory.py` holds the closed forms; `src/equity/portfolio.py` the book optimiser (`BookSpec`, `optimise_book`, the named `PRESETS`); `src/equity/cvar_portfolio.py` mean-variance under a CVaR limit; `src/equity/qubo.py` the QUBO encodings, a simulated-annealing solver with single and pair flips and a final quench, and an exhaustive search to check it against. Four allocators put them on the platform:

- `constrained_long_short`: each month-end, the combined forecast of any models in the spec, a trailing shrinkage covariance, the named `book` and the limits (`sector_neutral`, `beta_neutral`, `max_turnover`, `cost_bps`, `max_weight`). A month with no solution leaves the book as it was; a universe too small for a position limit gets it widened just enough.
- `tangency_cml`: the tangency portfolio on the forecast, scaled along the capital market line to `risk_aversion` (long-only by default, borrowing only up to `max_leverage`).
- `mv_cvar`: mean-variance under a `cvar_limit` on the trailing scenarios, cash absorbing what diversification cannot.
- `qubo_select`: the best `k` assets for the mean-variance trade-off by annealing, held equally.

They use data through the month-end only. No quantum hardware is involved: the QUBO matrices are the object an annealer would be handed; here they are solved classically.

## What we found

All simulated (`python -m experiments.portfolio_world`, about four minutes, nothing tuned), so these are checks of what the tools recover and of how methods compare, not evidence about markets.

*Estimation error.* Twelve assets, five years of monthly data, 300 repetitions, each portfolio scored on the *true* means and covariance (annualised Sharpe ratio): the best achievable 1.25; equal weights 1.18; minimum variance 1.02; the long-only tangency portfolio 0.91; the tangency portfolio with means shrunk halfway to their average 0.90; the plug-in tangency portfolio 0.70 (quartiles 0.57 and 0.85), worse than not optimising at all. Constraints and shrinkage both recover most of the loss, for the same reason: they cut the weight the estimate puts on its own errors.

*CAPM.* In a world built on the standard CAPM the cross-sectional intercept is -0.0002 per period (t -1.1) and the slope 0.0041 against a sample market premium of 0.0039. In a world with a 0.4% per period zero-beta rate, the intercept is 0.0040 (t 19) and the slope 0.0016, which is the market premium (0.0055) less that rate, as Black's theory says.

*APT.* Two planted factors with premia 0.003 and -0.001 are recovered as 0.0031 and -0.0016; twenty of fifty assets carry a planted pricing error of 0.15% per period and the pricing-error test rejects zero errors (chi-square 153 on 48 degrees of freedom). The arbitrage portfolio built from one sample has net investment 0 and factor exposure 0, and earns 0.16% per period in a second sample (t 8.6).

*Books.* The same alpha (a monthly cross-sectional regression on momentum and reversal, both planted) through six books in four simulated worlds, 40 stocks each, about 6.8 evaluated years per world, net of the platform's cost model. Sharpe ratios are the mean over worlds with the paired difference against long-only (its standard error across worlds):

| book | net Sharpe | minus long-only | gross | net | beta | turnover a year | cost, bp a year |
|---|---|---|---|---|---|---|---|
| long only | 1.06 | 0 | 1.0 | 1.00 | 0.96 | 7.5x | 75 |
| 130/30 | 1.23 | +0.17 (0.05) | 1.6 | 1.00 | 0.96 | 12.1x | 121 |
| 130/30, sector neutral | 1.11 | +0.05 (0.05) | 1.6 | 1.00 | 0.96 | 12.2x | 122 |
| market neutral and beta neutral | 1.52 | +0.46 (0.21) | 2.0 | 0.01 | 0.01 | 16.0x | 160 |
| 130/30, at most 30% turnover a month | 1.22 | +0.17 (0.04) | 1.6 | 1.00 | 0.95 | 4.1x | 41 |
| 130/30, 20 bp cost in the optimiser | 1.26 | +0.20 (0.05) | 1.6 | 1.00 | 0.96 | 7.5x | 75 |

The structure is exact and the Sharpe ratios are not: a single world's Sharpe over six or seven years has a standard error near 0.5, so only the paired differences and the structural columns mean anything. What they say: the 130/30 book beats long-only because it spends more gross on a signal that has a short side; removing the market (beta 0.96 to 0.01) raises the Sharpe ratio because the market is variance without alpha in this world; sector neutrality gave up about 0.12 of Sharpe relative to the plain 130/30 (inside the noise, but with an obvious reason: the sectors here are arbitrary labels, so there was no sector risk to remove and the constraint only took freedom away); and a turnover limit of 30% a month cut turnover and cost to a third with no loss of Sharpe, because the alpha (monthly momentum and reversal) is slow, which is not something to assume of a faster signal.

*CVaR.* Six assets, 800 scenarios, one with a 3% chance of a 15% crash. With no limit the portfolio holds 59% in the fat-tailed asset, CVaR(95%) 9.0% and expected return 1.10% per period. Limits of 8%, 5%, 3% and 2% all bind exactly (CVaR equals the limit), the weight on the fat-tailed asset falls to 50%, 26%, 14% and 5%, expected return to 1.07%, 0.92%, 0.75% and 0.57%, and at 2% the portfolio holds about 5% cash.

*QUBO.* On 14 assets with K = 4, annealing found the exhaustive-search optimum in 12 of 12 problems. At 40 assets (K = 10) and 80 (K = 15), where exhaustive search is out of reach, annealing beat a greedy forward selection by 0.0005 and tied it: for covariance matrices with a few factors the combinatorial problem is not hard, and the case for the QUBO form is that it carries other constraints in the same object, not that it beats greedy here.

## Going deeper

```
frontier       : w(m) = [ (C - B m) Sigma^-1 1 + (A m - B) Sigma^-1 mu ] / D       var(m) = (A m^2 - 2 B m + C) / D          minimum variance: Sigma^-1 1 / A
tangency / CML : w_T = Sigma^-1 (mu - rf) / 1'Sigma^-1 (mu - rf)                    E[R] = rf + Sharpe * sd                   invest y = (mu_T - rf) / (gamma sd_T^2) in w_T
zero-beta      : min w'Sigma w  s.t. 1'w = 1,  w'Sigma b = 0                         Black: E[R_i] = z + beta_i (E[R_m] - z),  z = E[R_zero-beta]
APT arbitrage  : min w'Sigma w  s.t. 1'w = 0,  B'w = 0,  alpha'w = 1                 w = Sigma^-1 C' (C Sigma^-1 C')^-1 d,  C = [1; B'; alpha']
book           : max  alpha'(w+ - w-) - (gamma/2)(w+ - w-)'Sigma(w+ - w-) - cost'(t) - tiny*(1'w+ + 1'w-)
                 s.t. 1'w+ = G_L, 1'w- = G_S, 0 <= w+ <= cap, 0 <= w- <= cap, |sector net - target| <= tol, |beta'w| <= tol, t >= |w - w0|, 1't <= turnover
CVaR           : CVaR_a(w) = max_q -q'Rw,  q_s = 1/((1-a)S) on the worst (1-a)S scenarios;   cut_k:  -(R'q_k)'w <= limit;   repeat until the most violated cut is satisfied
QUBO           : weights  w_i = w_max sum_k 2^k x_ik / (2^b - 1),   energy = (gamma/2) w'Sigma w - mu'w + P (1'w - 1)^2
                 select   : energy = (gamma/(2K^2)) x'Sigma x - mu'x/K + P (1'x - K)^2        P = 10 * the objective's scale
```

## Pitfalls

- The tangency portfolio is a function of the *differences* between expected returns, which are the least well measured numbers in finance. Use it with a forecast you trust, a long-only constraint or shrinkage, never with sample means.
- Sharpe differences between books are small next to their sampling error. Compare books on the same data, report paired differences, and lean on the structural columns (exposure, beta, turnover, cost), which the optimiser controls.
- A constraint has a price when it hedges something the world does not charge for: in the table above, sector neutrality gave up Sharpe. It is worth having when sectors carry risk you cannot forecast, which real sectors do.
- A 130/30 book needs short availability and pays a borrow fee; the platform's cost model charges trades and does not charge borrow. A market-neutral book here is 100% long and 100% short, so its gross is 2.0.
- The CVaR of a few years of history rests on a handful of tail scenarios (the worst 5% of 750 days is 37). A limit set from them is only as good as that tail is representative; the historical scenarios cannot contain a crash that has not happened.
- Annealing gives no proof of optimality. The tests compare it with exhaustive search on small problems; on large ones it returns the best state found, and a penalty that is too small lets the constraint be broken while one that is too large makes the search harder.
- The zero-beta intercept is the return of a *portfolio*, not a rate you can borrow at: a significant intercept in a CAPM test is evidence for the zero-beta version or against the model, and the test alone cannot tell which.

## Try it

```bash
python -m experiments.portfolio_world
quant backtest --model characteristic_regression --allocator constrained_long_short --alloc-param book=130_30 --alloc-param max_turnover=0.3 --tearsheet
quant backtest --model characteristic_regression --allocator constrained_long_short --alloc-param book=market_neutral --alloc-param beta_neutral=true --alloc-param sector_neutral=true
quant backtest --model characteristic_regression --allocator tangency_cml --alloc-param risk_aversion=4
quant backtest --model characteristic_regression --allocator mv_cvar --alloc-param cvar_limit=0.03
quant backtest --model characteristic_regression --allocator qubo_select --alloc-param k=5
```
