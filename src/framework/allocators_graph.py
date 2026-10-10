"""Allocators from graph theory: weights from the centrality of each asset in a filtered correlation network, and hierarchical sensitivity parity (:mod:`src.portfolio.graph_portfolio`).

    graph_centrality                  assets held in inverse proportion to their centrality (peripheral, the default) or in proportion to it (central) in the minimum spanning tree or the TMFG
    hierarchical_sensitivity_parity   at every split of a correlation dendrogram the two branches are weighted so that their sensitivity (risk contribution, or exposure to principal-component shocks) is equal

Both use the trailing window ending at each month-end only, a shrunk covariance, and the platform's position limits.
"""

from __future__ import annotations

import pandas as pd

from ..features.sleeves import month_end_dates
from ..portfolio.graph_portfolio import graph_book
from .allocation import Allocator, Context
from .registry import register_allocator


class _GraphAllocator(Allocator):
    min_assets = 4
    method = "graph"

    def _params(self) -> dict:
        return {}

    def build(self, ctx: Context) -> pd.DataFrame:
        bundle = ctx.bundle
        return graph_book(bundle.returns, bundle.investable, method=self.method, lookback=self.lookback, rebalance_index=month_end_dates(bundle.index), constraints=ctx.constraints(),
                          min_assets=self.min_assets, **self._params())


@register_allocator("graph_centrality", "Weights from centrality in a filtered correlation network (minimum spanning tree or TMFG): peripheral assets, the diversifiers, get more; clique, eigenvector, strength or degree centrality")
class GraphCentrality(_GraphAllocator):
    """Build the filtered graph of the trailing ``lookback`` days' correlations, measure each asset's centrality, and hold assets in inverse proportion to it (``tilt='peripheral'``) or in proportion
    (``'central'``). It uses only correlations (no volatilities): combine it with a volatility-aware overlay if volatilities differ much."""

    def __init__(self, graph: str = "tmfg", centrality: str = "clique", tilt: str = "peripheral", lookback: int = 252):
        if graph not in ("tmfg", "mst") or centrality not in ("degree", "strength", "eigenvector", "clique") or tilt not in ("peripheral", "central") or lookback < 60:
            raise ValueError("graph in (tmfg, mst), centrality in (degree, strength, eigenvector, clique), tilt in (peripheral, central), lookback >= 60")
        self.graph, self.centrality, self.tilt, self.lookback = graph, centrality, tilt, int(lookback)

    def _params(self) -> dict:
        return {"graph": self.graph, "centrality_kind": self.centrality, "tilt": self.tilt}


@register_allocator("hierarchical_sensitivity_parity", "A generalisation of hierarchical risk parity: at every split of the dendrogram the two branches are weighted so their risk contributions (or factor sensitivities) are equal")
class HierarchicalSensitivityParity(_GraphAllocator):
    """The author's generalisation of HRP (the published definition of HSP was not available): a ward dendrogram on the correlation distance, and at each node the two branches' weights chosen to equalise their
    sensitivity, ``risk`` (contribution to the variance of the two-branch portfolio, covariance included) or ``factor`` (exposure to the top principal-component shocks)."""

    method = "hsp"
    min_assets = 3

    def __init__(self, sensitivity: str = "risk", lookback: int = 252):
        if sensitivity not in ("risk", "factor") or lookback < 60:
            raise ValueError("sensitivity in (risk, factor), lookback >= 60")
        self.sensitivity, self.lookback = sensitivity, int(lookback)

    def _params(self) -> dict:
        return {"sensitivity": self.sensitivity}
