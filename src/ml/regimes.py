"""Regime models: a Gaussian hidden Markov model or a Markov-switching regression finds the market's regimes, and one ridge regression per regime predicts the assets, mixed by the regime probabilities.

**No look-ahead.** The probabilities used at a date are FILTERED: ``P(regime at t | market data up to t)``, a forward recursion. They are never smoothed (``hmmlearn``'s ``predict_proba`` is a forward-
backward pass that lets later observations change earlier probabilities, which is fine for describing history and wrong for trading). The parameters are fitted on the training window only, and the recursion
for a later date starts at the first training day and runs forward with those fixed parameters, so a test day is scored with everything known by its close and nothing after it.

The market state is built from the dataset's returns: the equal-weight return, its trailing 21-day volatility and the cross-sectional dispersion (the asset features are cross-sectionally standardised, so
the market-wide level has to come from the returns themselves). States are numbered by volatility, state 0 being the calmest, so that a number means the same thing from one refit to the next.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.linear_model import Ridge

from .models import Model

STATE_COLUMNS = ("mkt_ret", "log_vol", "log_disp")
MIN_DAYS = 250
FIT_ATTEMPTS = 3                                             # a Markov-switching fit starts from random parameters; a start that fails is replaced by the next seed's


def market_state(ds) -> pd.DataFrame:
    """The market's state on every date of the dataset, each row using that date's returns and earlier ones only."""
    r = ds.returns.where(ds.investable.reindex_like(ds.returns).fillna(False).astype(bool))
    mkt = r.mean(axis=1)
    vol = mkt.rolling(21, min_periods=15).std(ddof=1) * np.sqrt(252.0)
    disp = r.std(axis=1, ddof=0).rolling(5, min_periods=3).mean()
    out = pd.DataFrame({"mkt_ret": mkt, "log_vol": np.log(vol.clip(lower=1e-4)), "log_disp": np.log(disp.clip(lower=1e-5))}, index=ds.dates)
    return out.replace([np.inf, -np.inf], np.nan)


class HMMEngine:
    """A Gaussian HMM (hmmlearn) with the states ordered by volatility and a causal ``filter``."""

    name = "hmm"

    def __init__(self, n_states: int, covariance: str = "diag", n_iter: int = 60, seed: int = 0):
        self.k, self.covariance, self.n_iter, self.seed = int(n_states), covariance, int(n_iter), int(seed)

    def fit(self, Z: np.ndarray) -> "HMMEngine":
        from hmmlearn.hmm import GaussianHMM

        best, best_ll = None, -np.inf
        for restart in range(3):
            hmm = GaussianHMM(n_components=self.k, covariance_type=self.covariance, n_iter=self.n_iter, random_state=self.seed + restart, min_covar=1e-3)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hmm.fit(Z)
                ll = hmm.score(Z)
            except (ValueError, FloatingPointError):
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = hmm, ll
        if best is None:
            raise ValueError("the hidden Markov model could not be fitted to the market's history")
        vol_col = STATE_COLUMNS.index("log_vol")
        order = np.argsort(best.means_[:, vol_col])
        self.means_, self.covars_ = best.means_[order], np.asarray(best.covars_)[order]
        self.trans_, self.start_ = best.transmat_[np.ix_(order, order)], best.startprob_[order]
        self.loglik_ = float(best_ll)
        return self

    def filter(self, Z: np.ndarray) -> np.ndarray:
        """P(state at t | rows 0..t) for every row of Z (a row with a missing value carries the previous belief forward through the transition matrix)."""
        k = self.k
        out = np.zeros((len(Z), k))
        belief = self.start_.copy()
        dists = [multivariate_normal(self.means_[i], self.covars_[i], allow_singular=True) for i in range(k)]
        for t in range(len(Z)):
            if t:
                belief = belief @ self.trans_
            if np.all(np.isfinite(Z[t])):
                like = np.array([d.logpdf(Z[t]) for d in dists])
                belief = belief * np.exp(like - like.max())
            total = belief.sum()
            belief = belief / total if total > 0 else np.full(k, 1.0 / k)
            out[t] = belief
        return out

    def facts(self) -> dict:
        return {"transition": self.trans_.tolist(), "log_likelihood": self.loglik_}


class MarkovSwitchingEngine:
    """A Markov-switching regression (statsmodels) of the market return, with the regimes ordered by variance (or by mean when only the mean switches) and a causal ``filter``."""

    name = "markov_switching"

    def __init__(self, n_states: int, switching_variance: bool = True, seed: int = 0):
        self.k, self.switching_variance, self.seed = int(n_states), bool(switching_variance), int(seed)

    def _model(self, y: np.ndarray):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        return MarkovRegression(y, k_regimes=self.k, trend="c", switching_variance=self.switching_variance)

    def _fit_once(self, y: np.ndarray, seed: int):
        model = self._model(y)
        options = {"disp": False, "maxiter": 200, "search_reps": 5, "search_iter": 10}
        if "rng" in inspect.signature(model.fit).parameters:
            options["rng"] = seed                                                       # newer statsmodels draws its random starts from this, not from numpy's global generator
        else:
            np.random.seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(**options)

    def fit(self, Z: np.ndarray) -> "MarkovSwitchingEngine":
        y = Z[:, STATE_COLUMNS.index("mkt_ret")]
        last = None
        for attempt in range(FIT_ATTEMPTS):                                             # a random start can land on a transition matrix with no steady state; the next seed is another start
            try:
                res = self._fit_once(y, self.seed + attempt)
                break
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                last = exc
        else:
            raise ValueError(f"the Markov-switching model could not be fitted: {last}") from None
        self.params_ = np.asarray(res.params, dtype=float)
        names = list(res.model.param_names)
        key = "sigma2" if self.switching_variance else "const"
        spread = np.array([self.params_[names.index(f"{key}[{i}]")] for i in range(self.k)])
        self.order_ = np.argsort(spread)
        trans = np.asarray(res.regime_transition)[:, :, 0].T                           # statsmodels: element (i, j) = P(S_t = i | S_t-1 = j); rows here are "from"
        self.trans_ = trans[np.ix_(self.order_, self.order_)]
        self.loglik_ = float(res.llf)
        return self

    def filter(self, Z: np.ndarray) -> np.ndarray:
        y = Z[:, STATE_COLUMNS.index("mkt_ret")]
        res = self._model(np.where(np.isfinite(y), y, 0.0)).filter(self.params_)
        return np.asarray(res.filtered_marginal_probabilities)[:, self.order_]

    def facts(self) -> dict:
        return {"transition": self.trans_.tolist(), "log_likelihood": self.loglik_}


class RegimeMixture(Model):
    """One ridge per regime, mixed by the filtered regime probabilities of the date."""

    needs_dataset = True

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        k = int(self.params["n_states"])
        if not 2 <= k <= 5:
            raise ValueError("n_states must be between 2 and 5")
        if kind == "hmm_regime" and self.params["covariance"] not in ("diag", "full", "spherical"):
            raise ValueError("covariance must be diag, full or spherical")
        self.k = k
        self._ds = None
        self._state = None
        self._cache: dict = {}

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(_ds=None, _cache={})                                                  # the dataset is not part of the model; the market state it holds is kept
        return d

    def bind(self, ds) -> "RegimeMixture":
        self._ds = ds
        self._state = market_state(ds)
        self._cache = {}
        return self

    # -- the regime probabilities
    def _engine(self):
        p = self.params
        if self.kind == "hmm_regime":
            return HMMEngine(self.k, str(p["covariance"]), int(p["n_iter"]), self.seed)
        return MarkovSwitchingEngine(self.k, bool(int(p["switching_variance"])), self.seed)

    def _standardised(self, lo, hi) -> pd.DataFrame:
        s = self._state.loc[lo:hi]
        return (s - self.z_mu_) / self.z_sd_

    def probabilities(self, upto) -> pd.DataFrame:
        """Filtered regime probabilities for every date from the first training day to ``upto`` (dates with no market state yet get the uniform belief)."""
        key = pd.Timestamp(upto)
        if key not in self._cache:
            Z = self._standardised(self.start_, key)
            P = pd.DataFrame(self.engine_.filter(Z.to_numpy(dtype=float)), index=Z.index, columns=[f"regime_{i}" for i in range(self.k)])
            P[~np.isfinite(Z.to_numpy()).all(axis=1)] = np.nan
            self._cache = {key: P}
        return self._cache[key]

    def _row_probs(self, X: pd.DataFrame) -> np.ndarray:
        dates = X.index.get_level_values(0)
        P = self.probabilities(dates.max()).reindex(dates).to_numpy()
        return np.where(np.isfinite(P), P, 1.0 / self.k)

    # -- fitting
    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "RegimeMixture":
        if self._state is None:
            raise ValueError(f"{self.kind} needs the dataset's market history: it must be bound to a dataset before it is fitted")
        Xs = self._select(X)
        self.features_ = list(Xs.columns)
        self.n_train_ = len(Xs)
        d = X.index.get_level_values(0)
        self.start_, hi = d.min(), d.max()
        window = self._state.loc[self.start_:hi].dropna()
        if len(window) < MIN_DAYS:
            raise ValueError(f"{len(window)} days of market history are too few to find regimes (at least {MIN_DAYS})")
        self.z_mu_, self.z_sd_ = window.mean(), window.std(ddof=1).replace(0.0, 1.0)
        self.engine_ = self._engine().fit(((window - self.z_mu_) / self.z_sd_).to_numpy(dtype=float))
        self._cache = {}
        probs = self.probabilities(hi)
        P = probs.reindex(d).to_numpy()
        P = np.where(np.isfinite(P), P, 1.0 / self.k)
        values = Xs.to_numpy(dtype=np.float64)
        self.medians_ = np.where(np.isfinite(np.nanmedian(values, axis=0)), np.nanmedian(values, axis=0), 0.0)
        values = np.where(np.isnan(values), self.medians_, values)
        self.mu_, self.sd_ = values.mean(axis=0), values.std(axis=0) + 1e-9
        Z = (values - self.mu_) / self.sd_
        target = self._prepare_y(y)
        base = self._weights(dates)
        base = np.ones(len(Z)) if base is None else base
        alpha, floor = float(self.params["alpha"]), float(self.params["min_weight"])
        pooled = Ridge(alpha=alpha).fit(Z, target, sample_weight=base)
        self.models_ = []
        for i in range(self.k):
            w = P[:, i] * base
            self.models_.append(Ridge(alpha=alpha).fit(Z, target, sample_weight=w) if w.sum() >= floor else pooled)
        self.occupancy_ = P.mean(axis=0)
        return self

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.features_].to_numpy(dtype=np.float64)
        Z = (np.where(np.isnan(values), self.medians_, values) - self.mu_) / self.sd_
        P = self._row_probs(X)
        preds = np.column_stack([m.predict(Z) for m in self.models_])
        return (P * preds).sum(axis=1) * self.y_std_ + self.y_mean_

    def predict_extras(self, X: pd.DataFrame) -> dict:
        dates = pd.DatetimeIndex(sorted(set(X.index.get_level_values(0))))
        P = self.probabilities(dates.max()).reindex(dates)
        return {"regimes": P.fillna(1.0 / self.k)}

    def importance(self) -> pd.Series:
        coef = np.array([np.abs(m.coef_) for m in self.models_])
        return pd.Series(self.occupancy_ @ coef, index=self.features_)

    def diagnostics(self) -> dict:
        stats = self._state.loc[self.start_:].dropna()
        P = self.probabilities(stats.index.max()).reindex(stats.index).dropna()
        hard = P.to_numpy().argmax(axis=1)
        rows = []
        for i in range(self.k):
            sel = stats.loc[P.index[hard == i]] if (hard == i).any() else stats.iloc[:0]
            rows.append({"regime": i, "share": float((hard == i).mean()) if len(hard) else 0.0, "ann_return": float(sel["mkt_ret"].mean() * 252.0) if len(sel) else None,
                         "ann_vol": float(np.exp(sel["log_vol"]).mean()) if len(sel) else None})
        return {"engine": self.engine_.name, "n_states": self.k, "regimes": rows, **self.engine_.facts(), "train_days": int(len(stats))}


def build(kind: str, task: str, params: dict, seed: int) -> Model:
    if task != "regression":
        raise ValueError(f"{kind} is a regression")
    return RegimeMixture(kind, task, params, seed)
