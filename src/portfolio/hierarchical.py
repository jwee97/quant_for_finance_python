"""Hierarchical risk parity -- Generation 2, Priority 11.

Mean-variance optimisation inverts the covariance matrix. Inverting an
ill-conditioned matrix amplifies estimation noise, which is the instability
Stage 7 measured. Hierarchical methods never invert anything: they cluster the
assets by correlation, then allocate top-down through the tree, so a noisy
entry in the matrix can only misdirect capital *within* a cluster, never
across the whole book.

``hrp_weights``   Lopez de Prado (2016). Single-linkage clustering, quasi-
                  diagonalisation, then recursive bisection of the *ordered
                  list* with capital split inversely to cluster variance.
``herc_weights``  Raffinot (2017). Ward clustering, a cut of the dendrogram
                  into k clusters, equal risk contribution between and within
                  clusters. Follows the tree itself rather than halving a list.

Two deliberate departures from the HERC paper, both stated rather than hidden:
k is chosen by the silhouette score instead of the gap statistic (cheap and
deterministic, no reference-distribution simulation), and the within-cluster
allocation is exact equal risk contribution from ``risk_parity``.

One property worth knowing: the allocation between clusters is a sequence of
*binary* splits down the dendrogram, each inversely proportional to the
volatility of the two subtrees. Cluster risk is therefore equalised exactly
only when the tree is balanced. With three identical independent clusters the
tree is lopsided and the weights come out 0.29 / 0.29 / 0.41, not thirds,
because the merged pair is diversified. That is the algorithm, not a bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, to_tree
from scipy.spatial.distance import squareform

from .constraints import Constraints, project_frame
from .covariance import estimate_covariance, nearest_positive_definite
from .risk_parity import risk_parity_weights


# ---------------------------------------------------------------------------
# Distance, clustering, ordering
# ---------------------------------------------------------------------------
def correlation_matrix(covariance: pd.DataFrame) -> pd.DataFrame:
    """Correlation from a covariance matrix, clipped to [-1, 1]."""
    vol = np.sqrt(np.clip(np.diag(covariance.to_numpy(dtype=float)), 1e-18, None))
    corr = covariance.to_numpy(dtype=float) / np.outer(vol, vol)
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=covariance.index, columns=covariance.columns)


def correlation_distance(correlation: pd.DataFrame) -> pd.DataFrame:
    """``d_ij = sqrt(0.5 (1 - rho_ij))``.

    This is a proper metric (it is the Euclidean distance between standardised
    return vectors, up to a constant), which is what makes Ward linkage valid
    on it. ``1 - rho`` alone is not a metric.
    """
    distance = np.sqrt(0.5 * (1.0 - correlation.to_numpy(dtype=float)))
    np.fill_diagonal(distance, 0.0)
    return pd.DataFrame(distance, index=correlation.index, columns=correlation.columns)


def hierarchical_linkage(covariance: pd.DataFrame, method: str = "single") -> np.ndarray:
    """scipy linkage matrix on the correlation distance."""
    distance = correlation_distance(correlation_matrix(covariance))
    return linkage(squareform(distance.to_numpy(dtype=float), checks=False), method=method)


def quasi_diagonal_order(link: np.ndarray) -> list[int]:
    """Leaf order that places similar assets next to each other.

    Replace each cluster by its two children, left then right, until only
    leaves remain. Written out explicitly (rather than calling
    ``leaves_list``) because the ordering *is* the algorithm and the tests
    check the two agree.
    """
    n = link.shape[0] + 1
    order: list[int] = []
    stack = [2 * n - 2]                      # the root
    while stack:
        node = stack.pop()
        if node < n:
            order.append(int(node))
        else:
            left, right = int(link[node - n, 0]), int(link[node - n, 1])
            stack.append(right)              # popped after left: preserves order
            stack.append(left)
    return order


# ---------------------------------------------------------------------------
# Hierarchical risk parity
# ---------------------------------------------------------------------------
def cluster_variance(covariance: np.ndarray, items: list[int]) -> float:
    """Variance of a cluster held at inverse-variance weights."""
    block = covariance[np.ix_(items, items)]
    inverse = 1.0 / np.clip(np.diag(block), 1e-18, None)
    weights = inverse / inverse.sum()
    return float(weights @ block @ weights)


def _bisect(items: list[int], covariance: np.ndarray, weights: np.ndarray) -> None:
    if len(items) <= 1:
        return
    middle = len(items) // 2
    left, right = items[:middle], items[middle:]
    variance_left = cluster_variance(covariance, left)
    variance_right = cluster_variance(covariance, right)
    share_left = 1.0 - variance_left / (variance_left + variance_right)
    weights[left] *= share_left
    weights[right] *= 1.0 - share_left
    _bisect(left, covariance, weights)
    _bisect(right, covariance, weights)


def hrp_weights(covariance: pd.DataFrame, linkage_method: str = "single") -> pd.Series:
    """Hierarchical risk parity weights (long only, fully invested)."""
    assets = list(covariance.columns)
    n = len(assets)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series(1.0, index=assets)
    cov = nearest_positive_definite(covariance).to_numpy(dtype=float)
    if n == 2:
        inverse = 1.0 / np.diag(cov)
        return pd.Series(inverse / inverse.sum(), index=assets)

    link = hierarchical_linkage(pd.DataFrame(cov, index=assets, columns=assets), linkage_method)
    order = quasi_diagonal_order(link)
    weights = np.ones(n)
    _bisect(order, cov, weights)
    return pd.Series(weights / weights.sum(), index=assets)


# ---------------------------------------------------------------------------
# Hierarchical equal risk contribution
# ---------------------------------------------------------------------------
def choose_number_of_clusters(link: np.ndarray, distance: np.ndarray,
                              k_range: tuple[int, int] = (2, 6)) -> int:
    """k that maximises the silhouette score of the dendrogram cut."""
    from sklearn.metrics import silhouette_score

    n = distance.shape[0]
    low, high = max(2, k_range[0]), min(k_range[1], n - 1)
    if high < low:
        return max(1, min(2, n))
    best_k, best_score = low, -np.inf
    for k in range(low, high + 1):
        labels = fcluster(link, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= n:
            continue
        score = silhouette_score(distance, labels, metric="precomputed")
        if score > best_score:
            best_k, best_score = k, float(score)
    return best_k


def _erc_within(covariance: pd.DataFrame, members: list[str]) -> pd.Series:
    if len(members) == 1:
        return pd.Series(1.0, index=members)
    block = covariance.loc[members, members]
    return risk_parity_weights(block)


def herc_weights(covariance: pd.DataFrame, linkage_method: str = "ward",
                 k: int | None = None, k_range: tuple[int, int] = (2, 6)) -> pd.Series:
    """Hierarchical equal risk contribution weights."""
    assets = list(covariance.columns)
    n = len(assets)
    if n <= 2:
        return hrp_weights(covariance)
    cov_frame = nearest_positive_definite(covariance)
    distance = correlation_distance(correlation_matrix(cov_frame)).to_numpy(dtype=float)
    link = linkage(squareform(distance, checks=False), method=linkage_method)

    k = k or choose_number_of_clusters(link, distance, k_range)
    labels = fcluster(link, t=k, criterion="maxclust")
    cluster_of = {assets[i]: int(labels[i]) for i in range(n)}

    # Equal-risk-contribution weights and volatility inside each cluster.
    within: dict[int, pd.Series] = {}
    risk: dict[int, float] = {}
    for label in np.unique(labels):
        members = [a for a in assets if cluster_of[a] == label]
        within[int(label)] = _erc_within(cov_frame, members)
        w = within[int(label)].to_numpy()
        block = cov_frame.loc[members, members].to_numpy()
        risk[int(label)] = float(np.sqrt(max(w @ block @ w, 1e-18)))

    # Walk the dendrogram top-down, stopping where a node is exactly one cluster.
    cluster_weight: dict[int, float] = {}

    def leaves_of(node) -> list[str]:
        return [assets[i] for i in node.pre_order()]

    def cluster_risk(node) -> float:
        labels_in = {cluster_of[a] for a in leaves_of(node)}
        if len(labels_in) == 1:
            return risk[next(iter(labels_in))]
        # A node containing several clusters is treated as one asset whose risk
        # is the volatility of its own equal-risk-contribution portfolio.
        members = leaves_of(node)
        w = _erc_within(cov_frame, members)
        block = cov_frame.loc[members, members].to_numpy()
        return float(np.sqrt(max(w.to_numpy() @ block @ w.to_numpy(), 1e-18)))

    def allocate(node, capital: float) -> None:
        labels_in = {cluster_of[a] for a in leaves_of(node)}
        if len(labels_in) == 1 or node.is_leaf():
            cluster_weight[next(iter(labels_in))] = capital
            return
        risk_left, risk_right = cluster_risk(node.get_left()), cluster_risk(node.get_right())
        share_left = (1.0 / risk_left) / (1.0 / risk_left + 1.0 / risk_right)
        allocate(node.get_left(), capital * share_left)
        allocate(node.get_right(), capital * (1.0 - share_left))

    allocate(to_tree(link), 1.0)

    weights = pd.Series(0.0, index=assets)
    for label, capital in cluster_weight.items():
        weights.loc[within[label].index] = capital * within[label].to_numpy()
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Through time
# ---------------------------------------------------------------------------
def hierarchical_book(returns: pd.DataFrame, investable: pd.DataFrame, kind: str = "hrp",
                      lookback: int = 252, rebalance_index: pd.DatetimeIndex | None = None,
                      covariance_method: str = "shrinkage", linkage_method: str | None = None,
                      k_range: tuple[int, int] = (2, 6),
                      constraints: Constraints | None = None, min_assets: int = 5) -> pd.DataFrame:
    """HRP or HERC weights through time, re-solved on each rebalance date.

    Uses only the trailing ``lookback`` window ending at the rebalance date.
    Constraints are applied by projection afterwards so the book obeys the same
    position and group limits as every other model in the ladder; HRP has no
    notion of a cap and will otherwise happily put 40% in the short end of the
    curve.
    """
    index = pd.DatetimeIndex(returns.index)
    dates = rebalance_index if rebalance_index is not None else index
    book = pd.DataFrame(np.nan, index=index, columns=returns.columns)
    method = linkage_method or ("single" if kind == "hrp" else "ward")

    for stamp in dates:
        if stamp not in index:
            continue
        position = index.get_loc(stamp)
        if position < lookback:
            continue
        window = returns.iloc[position - lookback + 1:position + 1]
        live = [c for c in returns.columns
                if bool(investable.loc[stamp, c]) and window[c].notna().sum() > lookback * 0.8]
        if len(live) < min_assets:
            continue
        sample = window[live].dropna(how="any")
        if len(sample) < lookback * 0.6:
            continue
        try:
            cov = estimate_covariance(sample, covariance_method, lookback, annualise=True)
            weights = (hrp_weights(cov, method) if kind == "hrp"
                       else herc_weights(cov, method, k_range=k_range))
        except (ValueError, np.linalg.LinAlgError):
            continue
        book.loc[stamp, live] = weights.reindex(live).to_numpy()

    book = book.ffill().fillna(0.0)
    if constraints is not None:
        book = project_frame(book, constraints)
    return book


# ---------------------------------------------------------------------------
# Stability under estimation noise
# ---------------------------------------------------------------------------
def weight_stability(returns: pd.DataFrame, methods: dict, n_bootstrap: int = 200,
                     block_length: int = 21, seed: int = 5,
                     covariance_method: str = "shrinkage") -> pd.DataFrame:
    """How much do the weights move when the sample is resampled?

    Resamples the return window with a stationary block bootstrap (so the
    resamples keep volatility clustering), re-estimates the covariance, and
    recomputes every method's weights. ``methods`` maps a name to a function
    ``covariance -> weights``.

    Reported per method: the mean pairwise L1 distance between bootstrap
    weight vectors (the dispersion of the answer), the mean turnover from the
    full-sample weights, the average maximum weight and the effective number
    of positions. Low dispersion with a sensible effective N is the goal; low
    dispersion from sitting in a corner is not (Stage 7's lesson).

    Dispersion alone cannot tell those two apart, so two risk-based columns are
    added: ``mean_oos_vol``, the volatility of the bootstrap weights evaluated
    under the baseline covariance, and ``risk_inflation``, that volatility
    relative to the baseline weights' own. They measure the price of acting on
    a noisy estimate in the units that matter.
    """
    from ..validation.robustness import stationary_bootstrap_indices

    clean = returns.dropna(how="any")
    n_obs = len(clean)
    rng = np.random.default_rng(seed)
    values = clean.to_numpy()
    columns = list(clean.columns)

    base_cov = estimate_covariance(clean, covariance_method, n_obs, annualise=True)
    base_matrix = base_cov.to_numpy(dtype=float)
    baseline = {name: fn(base_cov).reindex(columns).to_numpy() for name, fn in methods.items()}
    baseline_vol = {name: float(np.sqrt(max(w @ base_matrix @ w, 1e-18))) for name, w in baseline.items()}

    draws: dict[str, list[np.ndarray]] = {name: [] for name in methods}
    for _ in range(n_bootstrap):
        sample = pd.DataFrame(values[stationary_bootstrap_indices(n_obs, block_length, rng)],
                              columns=columns)
        try:
            cov = estimate_covariance(sample, covariance_method, n_obs, annualise=True)
        except (ValueError, np.linalg.LinAlgError):
            continue
        for name, fn in methods.items():
            try:
                draws[name].append(fn(cov).reindex(columns).to_numpy())
            except Exception:
                continue

    rows = []
    for name, vectors in draws.items():
        if len(vectors) < 5:
            continue
        stack = np.vstack(vectors)
        pairwise = [np.abs(stack[i] - stack[j]).sum()
                    for i in range(len(stack)) for j in range(i + 1, min(i + 20, len(stack)))]
        # The risk actually taken when the weights are built from a noisy sample
        # but the world is the baseline covariance. This is the cost of
        # instability in the units an investor cares about, and unlike weight
        # dispersion it cannot be flattered by sitting in a corner.
        realised_vol = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", stack, base_matrix, stack), 1e-18, None))
        rows.append(
            {
                "method": name,
                "n_draws": int(len(stack)),
                "mean_pairwise_l1": float(np.mean(pairwise)),
                "mean_oos_vol": float(realised_vol.mean()),
                "risk_inflation": float(realised_vol.mean() / baseline_vol[name] - 1.0),
                "mean_turnover_from_baseline": float(np.abs(stack - baseline[name]).sum(axis=1).mean()),
                "mean_max_weight": float(stack.max(axis=1).mean()),
                "mean_effective_n": float((1.0 / (stack ** 2).sum(axis=1)).mean()),
                "mean_holdings": float((stack > 1e-4).sum(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("method")
