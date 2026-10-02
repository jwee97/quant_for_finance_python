"""Probabilistic forecasting (Generation 2, Priority 2).

A point forecast discards the one thing a risk-aware allocator needs: how sure
the model is. This module forecasts a DISTRIBUTION for each asset's return over
the next ``horizon`` trading days, scores it with proper scoring rules, and
turns it into positions in three ways that differ only in how much they use the
forecast's confidence.

Two probabilistic forecasts of the same quantity:

* a classifier for the DIRECTION, P(return > 0), calibrated with Platt scaling
  on data held out from the fit (after an embargo, because the labels look
  ``horizon`` days ahead);
* a Gaussian for the RETURN, N(mu, sigma^2), with mu from a ridge regression and
  sigma from an exponentially weighted volatility.

Both are scored against benchmarks that carry no feature information: the
asset's own base rate, and the same Gaussian with the asset's historical mean.
A forecast is only skilful if it beats those, which is a lower bar than being
profitable and a much higher one than being plausible.

Proper scoring rules are used because accuracy is not one: a forecaster that
says 51% every month is "right" about as often as one that says 90%, and only a
proper rule rewards the first for honesty and punishes the second for bluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

EPS = 1e-6


# ---------------------------------------------------------------------------
# Proper scoring rules
# ---------------------------------------------------------------------------
def clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def log_loss_terms(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-observation logarithmic loss -[y log p + (1-y) log(1-p)]."""
    p = clip_probability(p)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def brier_terms(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score of N(mu, sigma^2) at the outcome y (closed form).

    CRPS = sigma [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],  z = (y - mu) / sigma.
    It is in the units of the outcome, and reduces to the absolute error when
    sigma -> 0, so it can be read as a probabilistic generalisation of MAE.
    """
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sigma
    return sigma * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def log_score_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Negative log predictive density of N(mu, sigma^2) at y (lower is better)."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    return 0.5 * np.log(2.0 * np.pi * sigma ** 2) + (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) ** 2 / (2.0 * sigma ** 2)


def pit_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Probability integral transform: uniform on [0, 1] when the forecast is calibrated."""
    return stats.norm.cdf((np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / np.maximum(np.asarray(sigma, dtype=float), 1e-12))


def pit_uniformity(pit: np.ndarray) -> dict:
    """Kolmogorov-Smirnov distance from uniformity (a screening statistic: PITs of nearby forecasts are not independent)."""
    clean = np.asarray(pit, dtype=float)
    clean = clean[np.isfinite(clean)]
    test = stats.kstest(clean, "uniform")
    return {"ks_statistic": float(test.statistic), "ks_p_value": float(test.pvalue), "n": int(len(clean)),
            "share_below_10pct": float((clean < 0.1).mean()), "share_above_90pct": float((clean > 0.9).mean())}


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed frequency against forecast probability in ``n_bins`` equal-width bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = which == b
        rows.append({"bin": b, "lower": edges[b], "upper": edges[b + 1], "count": int(mask.sum()),
                     "mean_forecast": float(p[mask].mean()) if mask.any() else np.nan,
                     "observed_frequency": float(y[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean gap between forecast probability and observed frequency across bins."""
    table = reliability_table(p, y, n_bins)
    used = table[table["count"] > 0]
    weights = used["count"] / used["count"].sum()
    return float((weights * (used["observed_frequency"] - used["mean_forecast"]).abs()).sum())


def calibration_slope_intercept(p: np.ndarray, y: np.ndarray) -> dict:
    """Logistic regression of the outcome on logit(p): slope 1 and intercept 0 mean perfect calibration.

    A slope below one is over-confidence (forecasts too extreme), above one is
    under-confidence.
    """
    from sklearn.linear_model import LogisticRegression

    x = logit(clip_probability(p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, np.asarray(y, dtype=int))
    return {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve via the rank statistic."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(p)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------
@dataclass
class PlattCalibrator:
    """sigmoid(a * logit(p) + b), fitted on held-out forecasts and outcomes."""

    a: float = 1.0
    b: float = 0.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            return self
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(
            logit(clip_probability(p)).reshape(-1, 1), y)
        self.a, self.b = float(model.coef_[0, 0]), float(model.intercept_[0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return expit(self.a * logit(clip_probability(p)) + self.b)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's relative strength index, scaled to [0, 1]."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    strength = gain / loss.replace(0.0, np.nan)
    return (1.0 - 1.0 / (1.0 + strength)).fillna(1.0).where(gain.notna())


def price_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The twelve price features, each a daily frame aligned with ``prices``.

    Every value on day t uses prices up to and including day t.
    """
    logp = np.log(prices)
    returns = prices.pct_change()
    out: dict[str, pd.DataFrame] = {}
    for k in (21, 63, 126, 252):
        out[f"mom_{k}"] = prices / prices.shift(k) - 1.0
    for k in (5, 21, 63):
        mean = logp.rolling(k).mean()
        std = logp.rolling(k).std(ddof=1)
        out[f"zscore_{k}"] = (logp - mean) / std.replace(0.0, np.nan)
    out["vol_21"] = returns.rolling(21).std(ddof=1) * np.sqrt(252)
    out["vol_63"] = returns.rolling(63).std(ddof=1) * np.sqrt(252)
    out["vol_ratio"] = out["vol_21"] / out["vol_63"]
    out["drawdown_252"] = prices / prices.rolling(252, min_periods=126).max() - 1.0
    out["rsi_14"] = rsi(prices, 14)
    return out


def ewma_sigma(returns: pd.DataFrame, halflife: float, horizon: int) -> pd.DataFrame:
    """Volatility of the return over ``horizon`` days, from the EWMA of squared daily returns."""
    variance = (returns ** 2).ewm(halflife=halflife, adjust=False, min_periods=21).mean()
    return np.sqrt(variance * horizon)


# ---------------------------------------------------------------------------
# Panel and walk-forward fitting
# ---------------------------------------------------------------------------
@dataclass
class Panel:
    """One row per (origin, asset): features, outcomes and the volatility forecast."""

    frame: pd.DataFrame
    price_columns: list[str]
    common_columns: list[str]
    asset_names: list[str]
    sleeve_of: dict[str, str]
    origins: pd.DatetimeIndex
    positions: dict[pd.Timestamp, int] = field(default_factory=dict)


def build_panel(prices: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int, sigma_halflife: float,
                common: pd.DataFrame, sleeve_of: dict[str, str]) -> Panel:
    """Assemble the panel; ``common`` holds features shared by every asset (macro, regime), indexed by date."""
    features = price_features(prices)
    returns = prices.pct_change()
    sigma = ewma_sigma(returns, sigma_halflife, horizon)
    forward = prices.shift(-horizon) / prices - 1.0
    index = pd.DatetimeIndex(prices.index)
    rows = []
    for origin in origins:
        for asset in prices.columns:
            row = {"origin": origin, "asset": asset, "sleeve": sleeve_of.get(asset, "other"),
                   "sigma": sigma.at[origin, asset], "target": forward.at[origin, asset]}
            for name, frame in features.items():
                row[name] = frame.at[origin, asset]
            for name in common.columns:
                row[name] = common.at[origin, name] if origin in common.index else np.nan
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["up"] = (frame["target"] > 0).astype(float).where(frame["target"].notna())
    return Panel(frame=frame, price_columns=list(features), common_columns=list(common.columns),
                 asset_names=list(prices.columns), sleeve_of=sleeve_of,
                 origins=pd.DatetimeIndex(origins), positions={d: index.get_loc(d) for d in origins})


def design_matrix(frame: pd.DataFrame, panel: Panel, variant: str, stats_: dict | None = None):
    """Standardised feature matrix for a variant ('price_only' or 'price_macro').

    Asset dummies are always included. The macro variant adds the common
    features alone and interacted with each asset-class dummy. Standardisation
    uses the statistics of the TRAINING rows (passed back in ``stats_``).
    """
    columns = list(panel.price_columns)
    blocks = [frame[columns]]
    dummies = pd.get_dummies(frame["asset"]).reindex(columns=panel.asset_names, fill_value=0).astype(float)
    if variant == "price_macro":
        common = frame[panel.common_columns]
        sleeves = pd.get_dummies(frame["sleeve"]).astype(float)
        sleeve_names = sorted(set(panel.sleeve_of.values()))
        sleeves = sleeves.reindex(columns=sleeve_names, fill_value=0.0)
        inter = {f"{c}*{s}": common[c] * sleeves[s] for c in panel.common_columns for s in sleeve_names}
        blocks += [common, pd.DataFrame(inter, index=frame.index)]
    numeric = pd.concat(blocks, axis=1)
    if stats_ is None:
        stats_ = {"mean": numeric.mean(), "std": numeric.std(ddof=1).replace(0.0, 1.0)}
    scaled = (numeric - stats_["mean"]) / stats_["std"]
    return pd.concat([scaled, dummies], axis=1).to_numpy(dtype=float), stats_


def walk_forward_probabilistic(panel: Panel, variant: str, min_train: int = 1260, refit_every: int = 252,
                               horizon: int = 21, embargo: int = 21, calibration_fraction: float = 0.20,
                               classifier_c: float = 0.1, ridge_alpha: float = 100.0,
                               n_total_days: int | None = None) -> pd.DataFrame:
    """Expanding-window forecasts for every (origin, asset) after the first ``min_train`` days.

    At the refit that starts at day ``r`` the training rows are origins whose
    label (horizon days ahead) plus the embargo is fully realised before ``r``.
    The classifier is fitted on the first (1 - calibration_fraction) of those
    origins, Platt-calibrated on the last ``calibration_fraction`` after dropping
    one origin as an embargo, and the ridge mean on all of them. Test rows are
    the origins in ``[r, r + refit_every)``.
    """
    from sklearn.linear_model import LogisticRegression, Ridge

    frame = panel.frame
    origins = panel.origins
    pos = np.array([panel.positions[o] for o in origins])
    last_day = int(n_total_days if n_total_days is not None else pos.max() + horizon + 1)
    outputs = []
    for start in range(int(min_train), last_day, int(refit_every)):
        trainable = [o for o, p in zip(origins, pos) if p + horizon + embargo <= start]
        test_origins = [o for o, p in zip(origins, pos) if start <= p < start + refit_every]
        if len(trainable) < 24 or not test_origins:
            continue
        train_rows = frame[frame["origin"].isin(trainable)].dropna(subset=["target", "up"] + panel.price_columns)
        if variant == "price_macro":
            train_rows = train_rows.dropna(subset=panel.common_columns)
        test_rows = frame[frame["origin"].isin(test_origins)].dropna(subset=panel.price_columns + ["sigma"])
        if variant == "price_macro":
            test_rows = test_rows.dropna(subset=panel.common_columns)
        if train_rows.empty or test_rows.empty:
            continue
        dates = sorted(train_rows["origin"].unique())
        split = int(np.floor((1.0 - calibration_fraction) * len(dates)))
        base_dates, calib_dates = set(dates[:split]), set(dates[split + 1:])          # one origin dropped between them
        base = train_rows[train_rows["origin"].isin(base_dates)]
        calib = train_rows[train_rows["origin"].isin(calib_dates)]
        X_base, stats_ = design_matrix(base, panel, variant)
        X_calib, _ = design_matrix(calib, panel, variant, stats_)
        X_all, stats_all = design_matrix(train_rows, panel, variant)
        X_test_base, _ = design_matrix(test_rows, panel, variant, stats_)
        X_test_all, _ = design_matrix(test_rows, panel, variant, stats_all)

        classifier = LogisticRegression(C=classifier_c, max_iter=2000).fit(X_base, base["up"].to_numpy(dtype=int))
        raw_calib = classifier.predict_proba(X_calib)[:, 1]
        calibrator = PlattCalibrator().fit(raw_calib, calib["up"].to_numpy(dtype=int))
        raw_test = classifier.predict_proba(X_test_base)[:, 1]
        ridge = Ridge(alpha=ridge_alpha).fit(X_all, train_rows["target"].to_numpy(dtype=float))

        base_rate = train_rows.groupby("asset")["up"].mean()
        mean_return = train_rows.groupby("asset")["target"].mean()
        out = test_rows[["origin", "asset", "sleeve", "sigma", "target", "up"]].copy()
        out["p_raw"] = raw_test
        out["p_cal"] = calibrator.transform(raw_test)
        out["mu"] = ridge.predict(X_test_all)
        out["base_rate"] = out["asset"].map(base_rate)
        out["mu_benchmark"] = out["asset"].map(mean_return)
        out["refit_day"] = start
        out["platt_a"], out["platt_b"] = calibrator.a, calibrator.b
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


# ---------------------------------------------------------------------------
# Scores by origin and tests on them
# ---------------------------------------------------------------------------
def scores_by_origin(pred: pd.DataFrame) -> pd.DataFrame:
    """Cross-asset mean of every score at each origin: one observation per month for the tests."""
    valid = pred.dropna(subset=["target", "up", "p_cal", "mu", "sigma", "base_rate", "mu_benchmark"])
    y, up = valid["target"].to_numpy(), valid["up"].to_numpy()
    base = pd.DataFrame({
        "origin": valid["origin"].to_numpy(),
        "logloss_model": log_loss_terms(valid["p_cal"].to_numpy(), up),
        "logloss_raw": log_loss_terms(valid["p_raw"].to_numpy(), up),
        "logloss_benchmark": log_loss_terms(valid["base_rate"].to_numpy(), up),
        "brier_model": brier_terms(valid["p_cal"].to_numpy(), up),
        "brier_benchmark": brier_terms(valid["base_rate"].to_numpy(), up),
        "crps_model": crps_gaussian(y, valid["mu"].to_numpy(), valid["sigma"].to_numpy()),
        "crps_benchmark": crps_gaussian(y, valid["mu_benchmark"].to_numpy(), valid["sigma"].to_numpy()),
        "logscore_model": log_score_gaussian(y, valid["mu"].to_numpy(), valid["sigma"].to_numpy()),
        "logscore_benchmark": log_score_gaussian(y, valid["mu_benchmark"].to_numpy(), valid["sigma"].to_numpy()),
    })
    return base.groupby("origin").mean()


def block_bootstrap_months(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices over ``n`` months."""
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]


def paired_ece_test(pred: pd.DataFrame, n_samples: int = 2000, block_length: int = 3, seed: int = 7,
                    n_bins: int = 10) -> dict:
    """Does Platt scaling lower the expected calibration error? Paired bootstrap over months.

    Resamples MONTHS (the unit of dependence: assets in one month share the
    market), recomputes both ECEs on every draw, and returns the share of draws
    in which calibration did NOT help.
    """
    valid = pred.dropna(subset=["up", "p_raw", "p_cal"])
    months = sorted(valid["origin"].unique())
    by_month = {m: g for m, g in valid.groupby("origin")}
    rng = np.random.default_rng(seed)
    observed_raw = expected_calibration_error(valid["p_raw"].to_numpy(), valid["up"].to_numpy(), n_bins)
    observed_cal = expected_calibration_error(valid["p_cal"].to_numpy(), valid["up"].to_numpy(), n_bins)
    deltas = []
    for _ in range(n_samples):
        idx = block_bootstrap_months(len(months), block_length, rng)
        sample = pd.concat([by_month[months[i]] for i in idx])
        deltas.append(expected_calibration_error(sample["p_cal"].to_numpy(), sample["up"].to_numpy(), n_bins)
                      - expected_calibration_error(sample["p_raw"].to_numpy(), sample["up"].to_numpy(), n_bins))
    deltas = np.array(deltas)
    return {"ece_raw": observed_raw, "ece_calibrated": observed_cal, "difference": observed_cal - observed_raw,
            "ci_lower_5pct": float(np.percentile(deltas, 5)), "ci_upper_95pct": float(np.percentile(deltas, 95)),
            "p_value_not_improved": float((deltas >= 0).mean())}


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def _cap(weights: pd.Series, max_weight: float) -> pd.Series:
    return weights.clip(lower=-max_weight, upper=max_weight)


def edge_book(pred_at_origin: pd.DataFrame, sizing: str, no_trade_edge: float, max_weight: float) -> pd.Series:
    """Direction-only or probability-sized weights for one origin.

    ``edge`` is the calibrated probability minus the asset's base rate. Assets
    whose |edge| is below ``no_trade_edge`` are dropped. Weights are
    proportional to sign(edge)/sigma (direction only) or edge/sigma (probability
    sized), scaled to gross exposure one and then capped, so the two books
    differ ONLY in whether the size of the edge matters.
    """
    edge = (pred_at_origin["p_cal"] - pred_at_origin["base_rate"]).where(
        (pred_at_origin["p_cal"] - pred_at_origin["base_rate"]).abs() >= no_trade_edge, 0.0)
    scale = 1.0 / pred_at_origin["sigma"].replace(0.0, np.nan)
    raw = (np.sign(edge) if sizing == "direction_only" else edge) * scale
    raw = raw.fillna(0.0)
    gross = raw.abs().sum()
    weights = raw / gross if gross > 0 else raw
    return _cap(weights, max_weight)


def kelly_book(pred_at_origin: pd.DataFrame, kelly_fraction: float, max_weight: float, max_gross: float) -> pd.Series:
    """Fractional-Kelly weights w = f * mu / sigma^2 from the Gaussian forecast, capped, gross-limited."""
    variance = pred_at_origin["sigma"] ** 2
    raw = (kelly_fraction * pred_at_origin["mu"] / variance.replace(0.0, np.nan)).fillna(0.0)
    weights = _cap(raw, max_weight)
    gross = weights.abs().sum()
    return weights * (max_gross / gross) if gross > max_gross else weights
