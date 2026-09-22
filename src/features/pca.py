"""Principal component analysis (Ch. 8 §8.2.4, §8.5).

Solves the eigenproblem

    Sigma v_j = lambda_j v_j,        EV_j = lambda_j / sum_k lambda_k

and answers the project's structural question: *if we own 15 ETFs, how many
genuinely independent statistical risk dimensions do we actually own?*

Two implementation notes that matter for the answer:

1. **Correlation, not covariance, by default.** On a universe containing SHY
   (1.5% annualised vol) and SLV (33%), a covariance-based PCA would simply
   rediscover the volatility ranking: PC1 would be "silver". The correlation
   matrix removes the scale so the components describe co-movement.

2. **Sign convention.** Eigenvectors are only defined up to sign, so each
   loading vector is oriented to have a positive sum. Without this the sign of
   a PC flips arbitrarily between rolling windows and any time series of
   loadings is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    eigenvalues: pd.Series           # descending
    eigenvectors: pd.DataFrame       # assets x components (loadings)
    explained_variance: pd.Series    # EV_j
    cumulative_variance: pd.Series
    scores: pd.DataFrame             # time x components (factor returns)
    used_correlation: bool
    n_obs: int

    def n_components_for(self, threshold: float = 0.90) -> int:
        """Smallest number of PCs explaining at least ``threshold`` of variance."""
        cumulative = self.cumulative_variance.to_numpy()
        hits = np.where(cumulative >= threshold)[0]
        return int(hits[0] + 1) if len(hits) else len(cumulative)

    def effective_rank(self) -> float:
        """Exponential of the entropy of the eigenvalue spectrum.

        A single number for "how many independent bets are there really".
        Equals N when all eigenvalues are equal (perfect diversification) and
        1 when one factor explains everything.
        """
        weights = self.explained_variance.to_numpy()
        weights = weights[weights > 0]
        entropy = -np.sum(weights * np.log(weights))
        return float(np.exp(entropy))

    def loadings_table(self, n: int = 5) -> pd.DataFrame:
        return self.eigenvectors.iloc[:, :n]

    def summary(self, n: int = 5) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "eigenvalue": self.eigenvalues.iloc[:n],
                "explained_variance": self.explained_variance.iloc[:n],
                "cumulative_variance": self.cumulative_variance.iloc[:n],
            }
        )


def _orient(vectors: np.ndarray) -> np.ndarray:
    """Fix the arbitrary eigenvector sign so components are comparable."""
    signs = np.where(vectors.sum(axis=0) < 0, -1.0, 1.0)
    return vectors * signs


def pca_decomposition(returns: pd.DataFrame, use_correlation: bool = True,
                      demean: bool = True) -> PCAResult:
    """Eigen-decompose the return covariance (or correlation) matrix."""
    data = returns.dropna(how="any")
    if data.shape[0] < data.shape[1] + 2:
        raise ValueError(f"need more observations ({data.shape[0]}) than assets ({data.shape[1]})")

    centred = data - data.mean() if demean else data
    if use_correlation:
        scale = data.std(ddof=1).replace(0.0, np.nan)
        standardised = centred / scale
        matrix = standardised.cov()
        projection_input = standardised
    else:
        matrix = centred.cov()
        projection_input = centred

    values, vectors = np.linalg.eigh(matrix.to_numpy(dtype=float))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], _orient(vectors[:, order])

    names = [f"PC{i + 1}" for i in range(len(values))]
    eigenvalues = pd.Series(values, index=names, name="eigenvalue")
    loadings = pd.DataFrame(vectors, index=matrix.index, columns=names)
    explained = eigenvalues / eigenvalues.sum()
    scores = pd.DataFrame(projection_input.to_numpy() @ vectors, index=data.index, columns=names)

    return PCAResult(
        eigenvalues=eigenvalues,
        eigenvectors=loadings,
        explained_variance=explained.rename("explained_variance"),
        cumulative_variance=explained.cumsum().rename("cumulative_variance"),
        scores=scores,
        used_correlation=use_correlation,
        n_obs=int(len(data)),
    )


def rolling_explained_variance(returns: pd.DataFrame, window: int = 252, step: int = 21,
                               n_components: int = 3, use_correlation: bool = True) -> pd.DataFrame:
    """Explained variance of the top components through time.

    The interesting result is not the average level but the *spike*: PC1's
    share of variance jumps in every crisis, which is the quantitative
    statement of "correlations go to one exactly when diversification is
    needed".
    """
    data = returns.dropna(how="any")
    rows, index = [], []
    for end in range(window, len(data) + 1, step):
        block = data.iloc[end - window:end]
        try:
            result = pca_decomposition(block, use_correlation=use_correlation)
        except (ValueError, np.linalg.LinAlgError):
            continue
        row = {f"PC{i + 1}": float(result.explained_variance.iloc[i]) for i in range(n_components)}
        row["effective_rank"] = result.effective_rank()
        row["n_for_90pct"] = result.n_components_for(0.90)
        rows.append(row)
        index.append(data.index[end - 1])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def marchenko_pastur_bounds(n_obs: int, n_assets: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Random-matrix bounds on the eigenvalues of a pure-noise correlation matrix.

    Ch. 20 §20.2.11. Eigenvalues inside ``[lambda_min, lambda_max]`` are
    statistically indistinguishable from noise: with 15 assets and 252 days,
    a large part of the spectrum is not information. This is the honest
    counterweight to reading meaning into PC7.
    """
    if n_obs <= 0 or n_assets <= 0:
        return (np.nan, np.nan)
    ratio = n_assets / n_obs
    lambda_min = sigma2 * (1.0 - np.sqrt(ratio)) ** 2
    lambda_max = sigma2 * (1.0 + np.sqrt(ratio)) ** 2
    return float(lambda_min), float(lambda_max)


def significant_components(result: PCAResult, n_obs: int | None = None) -> int:
    """Count eigenvalues above the Marchenko-Pastur noise bound."""
    n_obs = n_obs or result.n_obs
    n_assets = len(result.eigenvalues)
    _, upper = marchenko_pastur_bounds(n_obs, n_assets)
    return int((result.eigenvalues > upper).sum())


def interpret_components(result: PCAResult, asset_class: dict[str, str] | None = None,
                         n: int = 4) -> pd.DataFrame:
    """Label each PC by the assets that load on it.

    PCs are statistical objects with no inherent economic meaning; this table
    is what lets the report say "PC1 is global risk-on" rather than "PC1".
    """
    asset_class = asset_class or {}
    rows = []
    for i in range(min(n, result.eigenvectors.shape[1])):
        name = f"PC{i + 1}"
        loadings = result.eigenvectors[name]
        ordered = loadings.sort_values()
        top = ordered.tail(3)[::-1]
        bottom = ordered.head(3)
        by_class: dict[str, float] = {}
        for ticker, value in loadings.items():
            key = asset_class.get(ticker, "unknown")
            by_class[key] = by_class.get(key, 0.0) + float(value)
        rows.append(
            {
                "component": name,
                "explained_variance": float(result.explained_variance[name]),
                "highest_loadings": ", ".join(f"{t} {v:+.2f}" for t, v in top.items()),
                "lowest_loadings": ", ".join(f"{t} {v:+.2f}" for t, v in bottom.items()),
                "dominant_class": max(by_class, key=lambda k: abs(by_class[k])) if by_class else "",
                "long_short": "directional" if loadings.min() > 0 or loadings.max() < 0 else "spread",
            }
        )
    return pd.DataFrame(rows).set_index("component")


def reconstruct_covariance(result: PCAResult, n_components: int, target_cov: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rebuild a covariance matrix from the top ``n_components`` PCs.

    This is the PCA-based covariance denoiser of Ch. 20 §20.2.10-11: keep the
    components that carry signal, replace the rest with their average
    (preserving the diagonal so total variance is not destroyed).
    """
    values = result.eigenvalues.to_numpy().copy()
    vectors = result.eigenvectors.to_numpy()
    if n_components < len(values):
        noise_mean = values[n_components:].mean()
        values[n_components:] = noise_mean
    rebuilt = vectors @ np.diag(values) @ vectors.T
    frame = pd.DataFrame(rebuilt, index=result.eigenvectors.index, columns=result.eigenvectors.index)

    if result.used_correlation and target_cov is not None:
        # Correlation-space result: rescale back with the target's volatilities.
        vol = np.sqrt(np.diag(target_cov.to_numpy()))
        outer = np.outer(vol, vol)
        corr = frame.to_numpy()
        diagonal = np.sqrt(np.diag(corr))
        corr = corr / np.outer(diagonal, diagonal)
        frame = pd.DataFrame(corr * outer, index=frame.index, columns=frame.columns)
    return frame
