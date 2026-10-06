---
title: "Graph attention across assets"
slug: graph-neural-networks
difficulty: 3
chapter: Ch. 23
prerequisites: [deep-time-series-models, pca-effective-rank]
stages: [33]
files: [src/models/graph_forecast.py, experiments/stage33_frontier.py]
figures: [66]
tests: [tests/test_frontier_models.py]
models: []
---

# Graph attention across assets

## In one sentence

Treat each ETF as a node, connect it to its most correlated peers, and let attention decide how much each neighbour's features matter for its forecast.

## The idea

On each day a correlation graph over the previous 252 days joins every ETF to its four nearest neighbours. A graph-attention layer computes, for each node, a weighted average of its neighbours' hidden
features, where the weights (attention) are learned. The idea is that information propagates along relationships between assets.

## Why it matters

Cross-asset structure is real (the first principal component explains about 41% of variance), but a 15-node graph is tiny compared with the stock universes where this is usually studied.

## How this repo uses it

`src/models/graph_forecast.py` is a pure-torch implementation with the graph recomputed from returns up to each origin. It is one of four models in Stage 33.

## What we found

The graph model had the lowest information coefficient (about 0.05) and was the only one whose CRPS was worse than the historical mean (by 0.0001, not significant).

## Going deeper

```
edges: for each ETF i, neighbours N(i) = 4 most correlated ETFs over the previous 252 days (recomputed at every origin)
h_i = W [features_i, embedding_i]
e_ij = LeakyReLU( a_src . h_j + a_dst . h_i );   alpha_ij = softmax_j over N(i) + {i} of e_ij
h_i' = ELU( sum_j alpha_ij h_j );   output_i = MLP([h_i', h_i])
```
Attention lets a node weigh a neighbour more or less depending on both nodes' features, but with 4 neighbours per node there is little room for the weights to matter.

## Pitfalls

- A correlation graph is dense and changes constantly.
- With 15 nodes and monthly rows there is little to learn.
- Attention weights are not explanations.

## Try it

```bash
python -m experiments.stage33_frontier
```
