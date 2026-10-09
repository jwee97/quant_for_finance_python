"""Factor models for expected returns: statistical (PCA) APT, macroeconomic, cross-sectional characteristics, and machine learning on the characteristics.

    apt_alpha                    statistical arbitrage pricing: take out the first K principal components of the trailing returns and rank assets by what is left, the appraisal ratio of the unexplained mean
    macro_factor_model           macroeconomic factors (changes in yields, spreads, volatility, the dollar, inflation expectations): each asset's loadings on them times the premium the market has paid per unit
                                 of loading (a Fama-MacBeth estimate from earlier months)
    characteristic_regression    cross-sectional model: this month's returns regressed on last month's characteristics of the assets (momentum, reversal, volatility, beta, distance from the high); the
                                 forecast is the trailing average of those slopes times today's characteristics
    ml_factor_model              the same characteristics fed to a learner: OLS, ridge, lasso, elastic net, principal-components or partial-least-squares regression, a decision tree (CART), a random forest or a small
                                 neural network, refitted every month on returns that have already happened

All are monthly decisions on data through the decision date. Characteristics are the price-based ones, so they run on any bundle; with company statements use the fundamental models.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..equity.alpha_model import standardize
from ..equity.contextual import fama_macbeth_forecast
from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model
from ..utils.dates import rebalance_dates

MONTH, YEAR = 21, 252


# ------------------------------------------------------------------------------------------------------------------ characteristics
def price_characteristics(data) -> dict[str, pd.DataFrame]:
    """The price-based characteristics of every asset on every day, each oriented so that more of it is expected to be better: ``mom`` (the return from a year ago to a month ago), ``rev`` (minus the last
    month's return), ``lowvol`` (minus the 60-day volatility), ``lowbeta`` (minus the one-year beta to the equal-weight market), ``nomax`` (minus the best day of the last month) and ``high`` (the price over
    its one-year high, less one)."""
    px, r = data.prices, data.returns.where(data.investable)
    market = r.mean(axis=1)
    beta = r.rolling(YEAR, min_periods=YEAR // 2).cov(market).div(market.rolling(YEAR, min_periods=YEAR // 2).var(), axis=0)
    return {
        "mom": px.shift(MONTH) / px.shift(YEAR) - 1.0,
        "rev": -(px / px.shift(MONTH) - 1.0),
        "lowvol": -r.rolling(60, min_periods=40).std(),
        "lowbeta": -beta,
        "nomax": -r.rolling(MONTH, min_periods=15).max(),
        "high": px / px.rolling(YEAR, min_periods=YEAR // 2).max() - 1.0,
    }


def _grid(data) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    grid = rebalance_dates(data.index, "monthly")
    px = data.prices.loc[grid]
    return grid, px.shift(-1) / px - 1.0


def _to_daily(grid_frame: pd.DataFrame, data) -> pd.DataFrame:
    return grid_frame.reindex(data.index).ffill().where(data.investable)


CHARACTERISTICS = ("mom", "rev", "lowvol", "lowbeta", "nomax", "high")


def _characteristic_list(spec: str) -> list[str]:
    names = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [n for n in names if n not in CHARACTERISTICS]
    if not names or unknown:
        raise ValueError(f"characteristics must be a comma-separated subset of {', '.join(CHARACTERISTICS)} (got {unknown or spec!r})")
    return names


# ------------------------------------------------------------------------------------------------------------------ statistical APT
def pca_residual_scores(returns: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For a window of returns (days by assets): take the first ``k`` principal components of the demeaned returns as portfolio weights, form their (raw) returns, regress every asset on them with an
    intercept, and return each asset's intercept (what it earned beyond its loadings times the factors' own mean returns: the APT's mispricing) and the standard deviation of its residual."""
    X = returns - returns.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    F = returns @ Vt[:k].T                                                                  # factor-mimicking portfolios: raw returns, so they have the means that price the assets
    design = np.column_stack([np.ones(len(F)), F])
    coef, *_ = np.linalg.lstsq(design, returns, rcond=None)
    resid = returns - design @ coef
    return coef[0], resid.std(axis=0, ddof=k + 1)


@register_model("apt_alpha", "cross-sectional", "Statistical APT: remove the first K principal components from the last two years of returns and rank assets by the appraisal ratio of what they earned beyond them")
class APTAlpha(ForecastModel):
    """The arbitrage pricing theory says only the exposures to a few common factors are paid; anything an asset earned beyond them is a mispricing (or luck). With no named factors the common ones are
    taken to be the first ``factors`` principal components of the trailing ``window`` days of returns, refitted every ``refit`` days, and the score is the intercept of each asset's regression on them over
    its residual volatility (an appraisal ratio), skipping the last ``skip`` days so the short-term reversal does not leak into a long-term signal. The generalisation of ``jensen_alpha`` from one factor
    (the market) to several."""

    name, family, position_mode = "apt_alpha", "cross-sectional", "cross_sectional"
    min_assets = 6

    def __init__(self, factors: int = 3, window: int = 504, skip: int = 21, refit: int = 21):
        if factors < 1 or window < 120 or skip < 0 or refit < 1:
            raise ValueError("factors >= 1, window >= 120, skip >= 0, refit >= 1")
        self.factors, self.window, self.skip, self.refit = int(factors), int(window), int(skip), int(refit)

    def score(self, data):
        r = data.returns.where(data.investable)
        values = np.full(r.shape, np.nan)
        arr = r.to_numpy()
        for end in range(self.window + self.skip, len(r), self.refit):
            block = arr[end - self.skip - self.window:end - self.skip]
            ok = np.isfinite(block).mean(axis=0) > 0.9
            if ok.sum() <= self.factors + 2:
                continue
            clean = np.nan_to_num(block[:, ok])
            alpha, resid_sd = pca_residual_scores(clean, self.factors)
            score = np.where(resid_sd > 1e-12, alpha / resid_sd, np.nan)
            row = np.full(r.shape[1], np.nan)
            row[ok] = score
            values[end:end + self.refit] = row
        return pd.DataFrame(values, index=r.index, columns=r.columns).where(data.investable)


# ------------------------------------------------------------------------------------------------------------------ macroeconomic factors
def rolling_betas(returns: pd.DataFrame, factors: pd.DataFrame, window: int, min_periods: int | None = None) -> np.ndarray:
    """Rolling multivariate OLS loadings of every asset on the factors: an array (days, factors, assets), NaN until ``min_periods`` days exist. Built from rolling sums of cross products, so it is one pass
    over the data however many assets there are."""
    R = returns.to_numpy(dtype=float)
    F = factors.reindex(returns.index).to_numpy(dtype=float)
    R, F = np.nan_to_num(R), np.nan_to_num(F)
    T, N = R.shape
    K = F.shape[1]
    min_periods = min_periods or window // 2

    def rolling_sum(a):
        c = np.cumsum(a, axis=0)
        out = c.copy()
        out[window:] = c[window:] - c[:-window]
        return out

    count = np.minimum(np.arange(1, T + 1), window).astype(float)
    mf, mr = rolling_sum(F) / count[:, None], rolling_sum(R) / count[:, None]
    ff = rolling_sum(np.einsum("ti,tj->tij", F, F).reshape(T, K * K)).reshape(T, K, K) / count[:, None, None] - np.einsum("ti,tj->tij", mf, mf)
    fr = rolling_sum(np.einsum("ti,tj->tij", F, R).reshape(T, K * N)).reshape(T, K, N) / count[:, None, None] - np.einsum("ti,tj->tij", mf, mr)
    out = np.full((T, K, N), np.nan)
    valid = np.flatnonzero(count >= min_periods)
    for t in valid:
        try:
            out[t] = np.linalg.solve(ff[t] + 1e-12 * np.eye(K), fr[t])
        except np.linalg.LinAlgError:
            continue
    return out


def macro_changes(macro: pd.DataFrame, names: list[str], innovations: bool, window: int = 504) -> pd.DataFrame:
    """Daily changes of the chosen macro series (differences for rates and spreads, percent changes for prices and indices), standardised by their trailing volatility. With ``innovations`` each is replaced by
    the residual of an AR(1) fitted on the trailing window, which is the unexpected part."""
    cols = {}
    for n in names:
        s = macro[n].astype(float).ffill()
        change = s.pct_change() if n.upper() in PRICE_LIKE else s.diff()
        if innovations:
            lag = change.shift(1)
            cov = change.rolling(window, min_periods=window // 2).cov(lag)
            var = lag.rolling(window, min_periods=window // 2).var()
            rho = (cov / var.replace(0.0, np.nan)).shift(1)                             # the autoregression known the day before
            change = change - rho * lag
        cols[n] = change / change.rolling(window, min_periods=window // 2).std().shift(1)
    return pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan)


PRICE_LIKE = {"VIX", "VIX3M", "VXN", "GVZ", "OVX", "MOVE", "DTWEXBGS"}                     # series measured in index points: use percent changes; rates and spreads use differences
MACRO_DEFAULT = ("DGS10", "T10Y3M", "BAA10Y", "VIX", "T10YIE", "DTWEXBGS")


@register_model("macro_factor_model", "macro", "Macroeconomic factor model: each asset's sensitivity to changes in yields, spreads, volatility, inflation expectations and the dollar, priced by the premium per unit of sensitivity paid in earlier months")
class MacroFactorModel(ForecastModel):
    """Chen, Roll and Ross: returns move with unexpected changes in a few macroeconomic variables, and assets that are more exposed to a risk that is priced earn more. Here the factors are daily changes of the
    macro series you name (default: the 10-year yield, the term spread, the credit spread, the VIX, breakeven inflation and the dollar), made unexpected by an AR(1) filter if ``innovations``, and
    standardised by their own volatility. Each asset's loadings on all of them are estimated by multivariate regression over the trailing ``window`` days. At each month-end the premium per unit of loading is the average,
    over the last ``premium_window`` months whose returns are known, of the cross-sectional regression slope of the next month's return on the loadings (Fama and MacBeth), and the score is loadings times
    premia. Needs the macro series in the bundle; uses what is there, and at least two."""

    name, family, position_mode = "macro_factor_model", "macro", "cross_sectional"
    min_assets = 6
    requires = MACRO_DEFAULT                  # what the default configuration reads; the model runs on any two of the series it is given (see ``require``)
    requires_at_least = 2

    def __init__(self, series: str = ",".join(MACRO_DEFAULT), window: int = 504, premium_window: int = 60, min_months: int = 24, innovations: bool = True):
        if window < 120 or premium_window < 12 or min_months < 6:
            raise ValueError("window >= 120, premium_window >= 12, min_months >= 6")
        self.series, self.window, self.premium_window, self.min_months, self.innovations = series, int(window), int(premium_window), int(min_months), bool(innovations)

    def usable_series(self, data) -> list[str]:
        wanted = [s.strip() for s in self.series.split(",") if s.strip()]
        return [n for n in wanted if n in data.macro.columns and data.macro[n].dropna().size > 2 * self.window]

    def require(self, data) -> None:
        """Any two of the chosen series with a long enough history are a factor model; fewer are not."""
        if len(self.usable_series(data)) < 2:
            raise KeyError(f"macro_factor_model needs macro series {[s.strip() for s in self.series.split(',') if s.strip()]}, at least two with more than {2 * self.window} days of history; "
                           f"this bundle has {list(data.macro.columns)}")

    def score(self, data):
        self.require(data)
        names = self.usable_series(data)
        factors = macro_changes(data.macro.reindex(data.index), names, self.innovations, self.window)
        returns = data.returns.where(data.investable)
        betas = rolling_betas(returns, factors, self.window)                                # days x factors x assets
        grid, forward = _grid(data)
        positions = data.index.get_indexer(grid)
        B = betas[positions]                                                                # grid x factors x assets
        K = B.shape[1]
        gammas = np.full((len(grid), K), np.nan)
        for t in range(len(grid)):
            y = forward.iloc[t].to_numpy()
            X = B[t].T                                                                      # assets x factors
            ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
            if ok.sum() >= K + 6:
                Z = np.column_stack([np.ones(ok.sum()), X[ok]])
                beta, *_ = np.linalg.lstsq(Z, y[ok], rcond=None)
                gammas[t] = beta[1:]
        scores = np.full((len(grid), returns.shape[1]), np.nan)
        for t in range(len(grid)):
            known = gammas[: max(t, 0)]                                                     # the regression for month s needs the return after s: it is known at s + 1, so rows before t
            known = known[np.isfinite(known).all(axis=1)][-self.premium_window:]
            if len(known) >= self.min_months and np.isfinite(B[t]).all(axis=0).any():
                premia = known.mean(axis=0)
                scores[t] = premia @ B[t]
        out = pd.DataFrame(scores, index=grid, columns=returns.columns)
        return _to_daily(out, data)


# ------------------------------------------------------------------------------------------------------------------ cross-sectional characteristics
@register_model("characteristic_regression", "cross-sectional", "Cross-sectional characteristics model: regress each month's returns on the previous month-end momentum, reversal, volatility, beta and distance from the high; forecast with the trailing average slopes")
class CharacteristicRegression(ForecastModel):
    """Fama and MacBeth's design as a forecaster: every month, regress the following month's returns of all assets on their characteristics at the month-end (momentum, reversal, low volatility, low beta, no
    extreme day, closeness to the one-year high), and forecast with the average slope of the last ``window`` months whose returns are known times today's characteristics. A characteristic that has not paid
    gets a slope near zero and so little weight. The same machinery the fundamental models use, with characteristics that need only prices."""

    name, family, position_mode = "characteristic_regression", "cross-sectional", "cross_sectional"
    min_assets = 8

    def __init__(self, characteristics: str = ",".join(CHARACTERISTICS), window: int = 60, min_obs: int = 24):
        if window < 12 or min_obs < 6:
            raise ValueError("window >= 12 and min_obs >= 6")
        self.characteristics, self.window, self.min_obs = characteristics, int(window), int(min_obs)
        _characteristic_list(characteristics)

    def score(self, data):
        chars = price_characteristics(data)
        names = _characteristic_list(self.characteristics)
        grid, forward = _grid(data)
        investable = data.investable.loc[grid]
        features = {n: standardize(chars[n].loc[grid], investable) for n in names}
        result = fama_macbeth_forecast(features, forward, 1, self.window, self.min_obs, None, min_assets=max(len(names) + 6, 10))
        return _to_daily(result.forecast, data)


# ------------------------------------------------------------------------------------------------------------------ machine learning on the characteristics
KINDS = ("ols", "ridge", "lasso", "enet", "pcr", "pls", "cart", "forest", "mlp")


def fit_learner(kind: str, X: np.ndarray, y: np.ndarray, alpha: float = 1.0, components: int = 3, depth: int = 4, trees: int = 100, seed: int = 0):
    """Fit one of the learners on standardised features ``X`` and target ``y``; returns ``(predict, info)`` where ``info`` holds the coefficients of the linear learners (and the number of selected features
    for the lasso, the components kept for PCR and PLS, the feature importances of the trees)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                                        # convergence warnings of the iterative learners on noisy targets are expected, not actionable
        return _fit(kind, X, y, alpha, components, depth, trees, seed)


def _fit(kind, X, y, alpha, components, depth, trees, seed):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor

    if kind == "pcr":
        pca = PCA(n_components=min(components, X.shape[1])).fit(X)
        reg = LinearRegression().fit(pca.transform(X), y)
        beta = pca.components_.T @ reg.coef_
        return (lambda Z: reg.predict(pca.transform(Z))), {"coef": beta, "components": pca.n_components_}
    if kind == "pls":
        pls = PLSRegression(n_components=min(components, X.shape[1]), scale=False).fit(X, y)
        beta = np.asarray(pls.coef_).ravel()
        return (lambda Z: pls.predict(Z).ravel()), {"coef": beta, "components": pls.n_components}
    if kind in ("ols", "ridge", "lasso", "enet"):
        model = {"ols": LinearRegression(), "ridge": Ridge(alpha=alpha), "lasso": Lasso(alpha=alpha, max_iter=5000), "enet": ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000)}[kind].fit(X, y)
        beta = np.asarray(model.coef_).ravel()
        return model.predict, {"coef": beta, "selected": int((np.abs(beta) > 1e-12).sum())}
    if kind == "cart":
        model = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(20, len(y) // 100), random_state=seed).fit(X, y)
    elif kind == "forest":
        model = RandomForestRegressor(n_estimators=trees, max_depth=depth, min_samples_leaf=max(20, len(y) // 200), max_features=0.7, n_jobs=1, random_state=seed).fit(X, y)
    else:
        model = MLPRegressor(hidden_layer_sizes=(8,), alpha=1e-2, solver="lbfgs", max_iter=200, random_state=seed).fit(X, y)       # lbfgs: far faster than Adam on a few thousand rows
        return model.predict, {}
    return model.predict, {"importances": model.feature_importances_}


@register_model("ml_factor_model", "machine learning", "Machine learning on price characteristics: OLS, ridge, lasso, elastic net, PCR, PLS, a decision tree, a random forest or a small neural network, refitted monthly on returns that have already happened")
class MLFactorModel(ForecastModel):
    """The characteristics of ``characteristic_regression`` fed to a learner chosen by ``kind``, to predict the next month's return in excess of the cross-section's average. Each month-end the learner is fitted
    on every earlier month-end whose following month has ended (the last ``train_months``), with characteristics standardised across assets and the target demeaned across assets, then applied to today's
    characteristics. ``kind`` is ``ols``, ``ridge``, ``lasso`` (sparse: weak characteristics get exactly zero), ``enet`` (between the two), ``pcr`` (regress on the first ``components`` principal
    components of the characteristics) or ``pls`` (the components most correlated with the target), ``cart`` (one decision tree of depth ``depth``), ``forest`` (an average of ``trees`` of them) or ``mlp`` (a small
    neural network). Trees and the network can find nonlinear and interaction effects the linear learners cannot; the signal-to-noise ratio of monthly returns is low, so expect them to need a lot of data."""

    name, family, position_mode = "ml_factor_model", "machine learning", "cross_sectional"
    min_assets = 8

    def __init__(self, kind: str = "lasso", characteristics: str = ",".join(CHARACTERISTICS), alpha: float = 0.05, components: int = 3, depth: int = 3, trees: int = 100, train_months: int = 60,
                 min_months: int = 24, refit_every: int = 0):
        if kind not in KINDS or alpha < 0 or components < 1 or depth < 1 or trees < 5 or train_months < 12 or min_months < 6 or refit_every < 0:
            raise ValueError(f"kind in {KINDS}, alpha >= 0, components >= 1, depth >= 1, trees >= 5, train_months >= 12, min_months >= 6, refit_every >= 0")
        _characteristic_list(characteristics)
        self.kind, self.characteristics, self.alpha, self.components, self.depth, self.trees = kind, characteristics, float(alpha), int(components), int(depth), int(trees)
        self.train_months, self.min_months = int(train_months), int(min_months)
        self.refit_every = int(refit_every) or (3 if kind in ("cart", "forest", "mlp") else 1)         # the slow learners are refitted quarterly, the linear ones monthly

    def score(self, data):
        chars = price_characteristics(data)
        names = _characteristic_list(self.characteristics)
        grid, forward = _grid(data)
        investable = data.investable.loc[grid]
        z = [standardize(chars[n].loc[grid], investable) for n in names]
        X = np.stack([f.fillna(0.0).to_numpy() for f in z], axis=-1)                           # grid x assets x features
        have = np.stack([f.notna().to_numpy() for f in z], axis=-1).all(axis=-1)
        y = forward.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)                                    # the last month-end has no following month
            y = y - np.nanmean(y, axis=1, keepdims=True)                                       # relative performance
        scores = np.full(have.shape, np.nan)
        predict, fitted_at = None, -10 ** 9
        for t in range(len(grid)):
            if predict is None or t - fitted_at >= self.refit_every:
                lo = max(0, t - self.train_months)
                rows = np.arange(lo, t)                                                        # month-ends whose following month has ended by t (the label of row t - 1 ends at t)
                usable = [(s, np.flatnonzero(have[s] & np.isfinite(y[s]))) for s in rows]
                usable = [(s, idx) for s, idx in usable if len(idx) >= 8]
                if len(usable) >= self.min_months:
                    Xtr = np.vstack([X[s][idx] for s, idx in usable])
                    ytr = np.concatenate([y[s][idx] for s, idx in usable])
                    predict, _ = fit_learner(self.kind, Xtr, ytr * 100.0, self.alpha, self.components, self.depth, self.trees)
                    fitted_at = t
            if predict is None or not have[t].any():
                continue
            now = np.flatnonzero(have[t])
            scores[t, now] = predict(X[t][now])                                                # a learner fitted on returns that had ended by its own date, applied to today's characteristics
        return _to_daily(pd.DataFrame(scores, index=grid, columns=data.returns.columns), data)
