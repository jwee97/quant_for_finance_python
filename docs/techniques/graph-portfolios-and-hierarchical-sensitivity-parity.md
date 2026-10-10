---
title: "Graph portfolios: spanning trees, TMFG, centrality and hierarchical sensitivity parity"
slug: graph-portfolios-and-hierarchical-sensitivity-parity
difficulty: 3
chapter: Platform
prerequisites: [hierarchical-risk-parity, risk-parity, pca-effective-rank]
stages: []
files: [src/portfolio/graph_portfolio.py, src/framework/allocators_graph.py, experiments/deep_world.py]
figures: []
tests: [tests/test_graph_portfolio.py]
models: []
---

# Graph portfolios: spanning trees, TMFG, centrality and hierarchical sensitivity parity

## In one sentence

A correlation matrix of `N` assets is a complete graph of `N(N-1)/2` noisy links; filtering it down to its strongest `N-1` (a minimum spanning tree) or `3(N-2)` (a triangulated maximally filtered graph) links shows which assets are hubs and which are peripheral, and portfolios can be built from that position (more of the peripheral, the diversifiers) or from the hierarchy of clusters (hierarchical sensitivity parity).

## The idea

**Filtered graphs.** The *minimum spanning tree* (MST) of the distance `sqrt(2(1 - rho))` is the cheapest connected skeleton of the market: `N-1` links, no loops. The *triangulated maximally filtered graph* (TMFG; Massara, Di Matteo and Aste 2016) starts from the four assets that are most correlated with the others and repeatedly inserts the remaining asset into the triangular face where its total correlation with the three corners is largest, keeping a planar graph of `3(N-2)` links that is a chain of four-cliques glued along triangles. It contains the loops the tree must drop, so it keeps more of the structure.

**Centrality and degeneracy.** *Degree* counts links, *strength* sums their correlation, *eigenvector* centrality is the Perron vector of the weighted graph, and *clique centrality* sums the correlation over every four-clique an asset belongs to (on a tree, its strength). Removing a vertex of smallest degree again and again gives a *degeneracy ordering* and a *core number* for each asset; the TMFG is 3-degenerate, and the reverse of its insertion order is such an ordering. A *peripheral* portfolio holds assets in inverse proportion to centrality, a *central* one in proportion.

**Hierarchical sensitivity parity (HSP).** The source this repository follows names the method but its definition was not available, so this is the author's reading, built on the skeleton of hierarchical risk parity: a dendrogram on the correlation distance, and at every split the two branches get weights `a` and `1 - a` such that the portfolio's *sensitivity* to each is equal. Two sensitivities: `risk`, the branch's contribution to the variance of the two-branch portfolio, covariance included (`a (a s_A + b s_AB) = b (b s_B + a s_AB)`), where HRP's split ignores the covariance between branches and uses inverse variance; and `factor`, the norm of the branch's exposure to the top principal-component shocks. Uncorrelated branches give inverse-*volatility* splits; identical branches give one half. It is a documented generalisation of HRP, not a reproduction of a published algorithm.

## Why it matters

Equal weighting and inverse-variance weighting ignore the structure of dependence; mean-variance uses all of it and is hostage to its estimation error. Graph filters sit between: they use only the dependence that stands out, in a form that is stable (a tree changes a few links when the sample changes) and interpretable (a picture of who drives whom).

## How this repo uses it

`src/portfolio/graph_portfolio.py`: `mst`, `tmfg` (both return a `Graph` with edges, weights, cliques, separators and insertion order), `degeneracy_ordering`, `centrality` (`degree`, `strength`, `eigenvector`, `clique`), `centrality_weights`, `hsp_weights`, and `graph_book`, which re-solves through time from the trailing window only. The allocators `graph_centrality` (`graph=tmfg|mst`, `centrality=...`, `tilt=peripheral|central`) and `hierarchical_sensitivity_parity` (`sensitivity=risk|factor`) put them on the platform with its position limits. Tests check the MST against a brute-force search over all spanning trees, the TMFG for its edge and clique counts, completeness of cliques and 3-degeneracy, the centrality orderings on a constructed hub, and HSP for the inverse-volatility pair, the equal risk contribution at the top split, and the absence of look-ahead.

## What we found

On the platform's 15 ETFs from 2010 (trailing 252-day correlations, monthly rebalances, shrunk covariance, the platform's position limits; `python -m experiments.deep_world`): equal weight returns 8.0% at 9.7% volatility, Sharpe 0.82, drawdown -20.4%. Inverse volatility 0.91, risk parity 0.92 and HRP 0.89 cut volatility to 6 to 7% and drawdown to -17%. HSP with risk sensitivity gets 0.83 (return 5.7%, volatility 6.9%, effective number of positions 9.0, turnover 0.55 a year) and with factor sensitivity 0.81: close to equal weight on risk-adjusted return and below HRP, with more turnover than HRP's 0.22. The graph portfolios are weaker still: the TMFG with clique centrality tilted to the *periphery* has Sharpe 0.74 (return 7.1%, volatility 9.7%, drawdown -23.1%, turnover 1.7 a year), tilted to the *centre* 0.81, the MST with strength centrality 0.81 and the TMFG with eigenvector centrality 0.76. The graph portfolios use only correlations, not volatilities, so they hold volatile assets as readily as calm ones, which on this universe (stocks, bonds, gold, commodities) is what costs them; the finding that peripheral portfolios do better was reported for stock universes, and nothing here supports it for this one.

## Pitfalls

- The graph is estimated: a different sample changes a few links, and the centrality of an asset at the edge of two clusters can flip. The monthly rebalance turnover (0.5 to 1.7 times a year, against 0.2 for risk parity) shows it.
- Centrality weights ignore volatility. On a mixed universe combine them with a volatility measure, or use them for equities of similar risk.
- HSP is the author's definition. Its `risk` form is the equal-risk-contribution split between branches and shares the weaknesses of ERC: it does not use expected returns, and its weights depend on the dendrogram's linkage (ward here).
- Degeneracy and core numbers describe the graph, not the market: in a planar TMFG they are 3 by construction.

## Try it

```bash
python -m experiments.deep_world
quant backtest --model characteristic_regression --allocator graph_centrality --alloc-param graph=tmfg --alloc-param tilt=peripheral
quant backtest --model characteristic_regression --allocator hierarchical_sensitivity_parity --alloc-param sensitivity=risk
```
