"""``model_ensemble``: several models fitted on the same rows and combined into one score and one predictive distribution.

**Fitting.** The last ``val_fraction`` of the training dates (after a purge of ``lag + horizon`` days) is held out. The members are fitted on the rest and score the held-out dates; each member's score is
calibrated to the target's units by a regression on those dates (so a ranker's z-score and a regression's forecast can be averaged), and its rank IC on each held-out date is recorded. With ``refit_full`` the
members are then refitted on the whole training window for the predictions.

**Weights.** ``equal``; ``dynamic``: a member's weight on a day is a softmax (``temperature``, in standard deviations of the members' scores) of its exponentially weighted rank IC (half-life ``decay_days``)
over the days whose outcome is already known by that day, which means the held-out days and the days of the test block at least ``lag + horizon`` days old, never the days still in flight; ``regime``: the same
softmax computed separately for each volatility regime of the market (``regimes`` buckets of the trailing volatility, thresholds from the training window), so the members that do well in calm and in turbulent
markets are selected by the day's regime.

**Distribution.** The mixture of the members' calibrated Normal predictive distributions with those weights: its mean is the score, its variance the weighted members' variances plus their disagreement. A member
that has a predictive distribution of its own (quantile models, NGBoost, the Temporal Fusion Transformer) contributes its row-level standard deviation; the others the standard deviation of their residuals on the
held-out days.

Members are written ``kind`` or ``kind[setting=value;setting=value]``, separated by commas, for example ``ridge,lightgbm[n_estimators=100],lstm[epochs=5;hidden=16]``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from .models import _INFO, Model, make_model
from .regimes import market_state

MEMBER = re.compile(r"\s*([a-z_0-9]+)\s*(?:\[([^\]]*)\])?\s*(?:,|$)")
MAX_MEMBERS = 8
FLOOR_STD = 0.25


def _value(text: str):
    text = text.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_members(text: str) -> list[tuple[str, dict]]:
    """``"ridge,lightgbm[n_estimators=100]"`` -> ``[("ridge", {}), ("lightgbm", {"n_estimators": 100})]``."""
    out, pos = [], 0
    text = str(text).strip()
    while pos < len(text):
        m = MEMBER.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot read the members at '{text[pos:pos + 20]}': write kind or kind[setting=value;setting=value], separated by commas")
        params = {}
        for item in filter(None, (m.group(2) or "").split(";")):
            if "=" not in item:
                raise ValueError(f"'{item}' is not setting=value")
            k, v = item.split("=", 1)
            params[k.strip()] = _value(v)
        out.append((m.group(1), params))
        pos = m.end()
    return out


def _spearman_by_date(scores: np.ndarray, target: np.ndarray, keys: np.ndarray, min_assets: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """``(unique keys, rank IC of each column of scores against target on each key)``; NaN where fewer than ``min_assets`` rows are usable."""
    keys_u = np.unique(keys)
    out = np.full((len(keys_u), scores.shape[1]), np.nan)
    order = np.argsort(keys, kind="stable")
    bounds = np.searchsorted(keys[order], keys_u), np.searchsorted(keys[order], keys_u, side="right")
    for i, (lo, hi) in enumerate(zip(*bounds)):
        rows = order[lo:hi]
        ok = np.isfinite(target[rows])
        if ok.sum() < min_assets:
            continue
        tr = stats.rankdata(target[rows][ok])
        for m in range(scores.shape[1]):
            s = scores[rows][ok, m]
            if np.ptp(s) > 0:
                out[i, m] = np.corrcoef(stats.rankdata(s), tr)[0, 1]
    return keys_u, out


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class MemberEnsemble(Model):
    needs_dataset = True

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        if task != "regression":
            raise ValueError("model_ensemble is a regression")
        p = self.params
        self.specs_ = parse_members(p["members"])
        if not 2 <= len(self.specs_) <= MAX_MEMBERS:
            raise ValueError(f"an ensemble needs between 2 and {MAX_MEMBERS} members")
        for k, mp in self.specs_:
            if k not in _INFO or k == "model_ensemble":
                raise ValueError(f"'{k}' is not a model that can be a member; choose from {sorted(set(_INFO) - {'model_ensemble'})[:6]} ...")
            make_model(k, mp, "regression", seed)                                    # a wrong setting or a missing package is refused here
        if p["weighting"] not in ("equal", "dynamic", "regime"):
            raise ValueError("weighting must be equal, dynamic or regime")
        if not 0.1 <= float(p["val_fraction"]) <= 0.5 or int(p["decay_days"]) < 1 or float(p["temperature"]) < 0 or not 2 <= int(p["regimes"]) <= 4:
            raise ValueError("invalid ensemble setting(s)")
        self._ds = None
        self._fwd = self._state = None
        self.members_: list = []

    # ------------------------------------------------------------------------------------------------------------ data access
    def bind(self, ds) -> "MemberEnsemble":
        self._ds, self._fwd, self._state = ds, None, None
        for m in self.members_:
            if m.needs_dataset:
                m.bind(ds)
        return self

    def _need_ds(self) -> None:
        if self._ds is None and self.params["weighting"] != "equal":
            raise ValueError(f"{self.params['weighting']} weighting follows the market through time: the ensemble must be bound to a dataset")

    def _positions(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return self._ds.dates.get_indexer(dates)

    def _market(self) -> pd.DataFrame:
        if self._state is None:
            self._state = market_state(self._ds)
        return self._state

    def _forward(self) -> pd.DataFrame:
        if self._fwd is None:
            from .targets import forward_return

            self._fwd = forward_return(self._ds.prices, self._ds.horizon, self._ds.lag).reindex(index=self._ds.dates, columns=self._ds.assets)
        return self._fwd

    def _regime_of(self, dates: pd.DatetimeIndex) -> np.ndarray:
        vol = self._market()["log_vol"].reindex(dates).to_numpy()
        return np.where(np.isfinite(vol), np.searchsorted(self.edges_, vol), 0)

    # ------------------------------------------------------------------------------------------------------------ fitting
    def _new(self, i: int) -> Model:
        k, mp = self.specs_[i]
        m = make_model(k, mp, "regression", self.seed + i)
        return m.bind(self._ds) if (m.needs_dataset and self._ds is not None) else m

    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "MemberEnsemble":
        self._need_ds()
        Xs = self._select(X)
        self.features_ = list(Xs.columns)
        self.n_train_ = len(Xs)
        p = self.params
        pos = np.asarray(dates) if dates is not None else pd.Series(Xs.index.get_level_values(0)).rank(method="dense").to_numpy()
        upos = np.unique(pos)
        n_val = max(5, int(round(len(upos) * float(p["val_fraction"]))))
        start = upos[-n_val]
        purge = (self._ds.lag + self._ds.horizon) if self._ds is not None else 0
        fit_rows, val_rows = np.flatnonzero(pos <= start - 1 - purge), np.flatnonzero(pos >= start)
        if len(fit_rows) < 300 or len(val_rows) < 60:
            raise ValueError(f"too few rows to fit an ensemble ({len(fit_rows)} to fit, {len(val_rows)} to weigh the members on)")
        yv = y.to_numpy(dtype=float)
        self.y_mean_, self.y_std_ = float(np.nanmean(yv)), float(np.nanstd(yv)) or 1.0
        first = [self._new(i).fit(Xs.iloc[fit_rows], y.iloc[fit_rows], pos[fit_rows]) for i in range(len(self.specs_))]
        Xv = Xs.iloc[val_rows]
        S = np.column_stack([m.predict(Xv) for m in first])
        yval = yv[val_rows]
        ok = np.isfinite(yval)
        self.a_, self.b_, self.sigma_ = np.zeros(S.shape[1]), np.zeros(S.shape[1]), np.zeros(S.shape[1])
        for j in range(S.shape[1]):
            s, t = S[ok, j], yval[ok]
            b = max(float(np.cov(s, t)[0, 1] / (np.var(s, ddof=1) + 1e-18)), 0.0)
            self.a_[j], self.b_[j] = float(t.mean() - b * s.mean()), b
            self.sigma_[j] = float(np.std(t - self.a_[j] - b * s, ddof=1)) or self.y_std_
        dates_val = Xv.index.get_level_values(0)
        self.val_dates_, ic = _spearman_by_date(S, yval, dates_val.to_numpy())
        self.val_ic_ = ic
        self.edges_ = np.array([])
        if p["weighting"] == "regime":
            lv = self._market()["log_vol"].loc[Xs.index.get_level_values(0).min():Xs.index.get_level_values(0).max()].dropna()
            self.edges_ = np.quantile(lv, np.linspace(0, 1, int(p["regimes"]) + 1)[1:-1])
        self.regime_weights_ = self._regime_weights(pd.DatetimeIndex(self.val_dates_), ic) if p["weighting"] == "regime" else None
        self.members_ = [self._new(i).fit(Xs, y, pos) for i in range(len(self.specs_))] if int(p["refit_full"]) else first
        self.val_ic_mean_ = np.nanmean(ic, axis=0)
        return self

    def _regime_weights(self, dates: pd.DatetimeIndex, ic: np.ndarray) -> np.ndarray:
        """Member weights for each regime from the held-out days in it (the pooled weights where a regime has fewer than 8 days)."""
        reg = self._regime_of(dates)
        pooled = self._softmax_ic(np.nanmean(ic, axis=0))
        out = np.tile(pooled, (int(self.params["regimes"]), 1))
        for r in range(out.shape[0]):
            sel = reg == r
            if sel.sum() >= 8:
                out[r] = self._softmax_ic(np.nanmean(ic[sel], axis=0))
        return out

    def _softmax_ic(self, s: np.ndarray) -> np.ndarray:
        s = np.where(np.isfinite(s), s, np.nanmean(s) if np.isfinite(s).any() else 0.0)
        z = (s - s.mean()) / max(float(s.std()), 0.01)
        return _softmax(float(self.params["temperature"]) * z)

    # ------------------------------------------------------------------------------------------------------------ prediction
    def _weights(self, X: pd.DataFrame, S: np.ndarray) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
        """``(weights per row, the dates, the weights per date)``."""
        M = S.shape[1]
        d_rows = X.index.get_level_values(0)
        dates = pd.DatetimeIndex(sorted(set(d_rows)))
        mode = self.params["weighting"]
        if mode == "equal":
            W = np.full((len(dates), M), 1.0 / M)
        elif mode == "regime":
            W = self.regime_weights_[self._regime_of(dates)]
        else:
            lag_h = self._ds.lag + self._ds.horizon
            pos = self._positions(dates)
            target = self._forward().to_numpy()[pos[dates.get_indexer(d_rows)], pd.Index(self._ds.assets).get_indexer(X.index.get_level_values(1))]
            keys_u, ic_new = _spearman_by_date(S, target, d_rows.to_numpy())
            new_pos = self._positions(pd.DatetimeIndex(keys_u))
            hist_pos = np.concatenate([self._positions(pd.DatetimeIndex(self.val_dates_)), new_pos])
            avail = np.concatenate([np.full(len(self.val_dates_), -10 ** 9), new_pos + lag_h])                     # a held-out day is known; a day of this block only once its outcome is
            hist_ic = np.vstack([self.val_ic_, ic_new])
            W = np.empty((len(dates), M))
            half = float(self.params["decay_days"])
            for i, t in enumerate(pos):
                use = avail <= t
                ages = (t - hist_pos[use]).astype(float)
                w_age = 0.5 ** (ages / half)
                ic = hist_ic[use]
                good = np.isfinite(ic)
                num = (np.where(good, ic, 0.0) * w_age[:, None]).sum(0)
                den = (good * w_age[:, None]).sum(0)
                s = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
                W[i] = self._softmax_ic(s) if np.isfinite(s).any() else np.full(M, 1.0 / M)
        return W[dates.get_indexer(d_rows)], dates, W

    def _core(self, X: pd.DataFrame) -> dict:
        self._need_ds()
        Xp = X[self.features_]
        S = np.column_stack([m.predict(Xp) for m in self.members_])
        cal = self.a_ + self.b_ * S
        sig = np.tile(self.sigma_, (len(X), 1))
        for j, m in enumerate(self.members_):
            dist = m.predict_extras(Xp).get("distribution")
            if dist is not None and self.b_[j] > 0:
                sig[:, j] = np.maximum(self.b_[j] * dist["std"].to_numpy(), FLOOR_STD * self.sigma_[j])
        Wr, dates, Wd = self._weights(X, S)
        mu = (Wr * cal).sum(axis=1)
        var = (Wr * (sig ** 2 + (cal - mu[:, None]) ** 2)).sum(axis=1)
        return {"mu": mu, "std": np.sqrt(var), "disagreement": np.sqrt((Wr * (cal - mu[:, None]) ** 2).sum(axis=1)), "dates": dates, "W": Wd}

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._core(X)["mu"]

    def predict_extras(self, X: pd.DataFrame) -> dict:
        c = self._core(X)
        z = stats.norm.ppf(0.9)
        dist = pd.DataFrame({"mean": c["mu"], "std": c["std"], "lower": c["mu"] - z * c["std"], "upper": c["mu"] + z * c["std"], "disagreement": c["disagreement"]}, index=X.index)
        weights = pd.DataFrame(c["W"], index=c["dates"], columns=[f"{k}" if [s[0] for s in self.specs_].count(k) == 1 else f"{k}#{i}" for i, (k, _) in enumerate(self.specs_)])
        return {"distribution": dist, "weights": weights}

    def importance(self) -> pd.Series:
        total = pd.Series(0.0, index=self.features_)
        for j, m in enumerate(self.members_):
            imp = m.importance().reindex(self.features_).fillna(0.0).abs()
            total += imp / (imp.sum() or 1.0) / len(self.members_)
        return total

    def diagnostics(self) -> dict:
        names = [k for k, _ in self.specs_]
        d = {"members": names, "weighting": self.params["weighting"], "interval_level": 0.8, "calibration": [{"member": n, "intercept": float(a), "slope": float(b), "residual_std": float(s), "heldout_ic": float(ic) if np.isfinite(ic) else None}
                                                                                   for n, a, b, s, ic in zip(names, self.a_, self.b_, self.sigma_, self.val_ic_mean_)],
             "member_diagnostics": {n: m.diagnostics() for n, m in zip(names, self.members_) if m.diagnostics()}}
        if self.regime_weights_ is not None:
            d["regime_weights"] = self.regime_weights_.tolist()
            d["regime_edges"] = [float(v) for v in self.edges_]
        return d

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_ds=None, _fwd=None, _state=None)
        return state


def build(kind: str, task: str, params: dict, seed: int) -> Model:
    return MemberEnsemble(kind, task, params, seed)
