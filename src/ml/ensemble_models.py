"""``model_ensemble``: several models fitted on the same rows and combined into one score and one predictive distribution.

**Fitting.** The last ``val_fraction`` of the training dates (after a purge of ``lag + horizon`` days) is held out. The members are fitted on the rest and score the held-out dates; each member's score is
calibrated to the target's units by a regression on those dates (so a ranker's z-score and a regression's forecast can be averaged), and its rank IC on each held-out date is recorded. With ``refit_full`` the
members are then refitted on the whole training window for the predictions.

**Weights.** ``equal``; ``dynamic``: a member's weight on a day is a softmax (``temperature``, in standard deviations of the members' scores) of its exponentially weighted rank IC (half-life ``decay_days``)
over the days whose outcome is already known by that day, which means the held-out days and the days of the test block at least ``lag + horizon`` days old, never the days still in flight; ``regime``: the same
softmax computed separately for each volatility regime of the market (``regimes`` buckets of the trailing volatility, thresholds from the training window), so the members that do well in calm and in turbulent
markets are selected by the day's regime.

**Regularised stacking (Release 7, P2-08).** ``stacked``: the weights are the non-negative least-squares fit of the held-out target on the members' calibrated held-out scores (both centred on the stage's mean target),
normalised to sum to one, and shrunk toward equal weights by ``shrink`` (0: the fit alone, 1: equal weights): ``w = (1 - shrink) * w_nnls + shrink / M``. ``stacked_regime`` fits them separately for each volatility regime
(the pooled weights where a regime has fewer than 20 held-out dates). The training window is the held-out stage and no other: members were fitted on earlier, purged rows, so the weights are trained on out-of-sample
streams only, and nothing in the block being scored can move them. **Member coverage**: a member that scores fewer than ``min_coverage`` of the held-out rows is left out of every weighting (and the ensemble refuses to
fit with fewer than two left); where a member is missing a row the other members' weights are renormalised for it, and a row no member scores has no prediction. Coverage and exclusions are in the diagnostics.

**Distribution.** The mixture of the members' calibrated Normal predictive distributions with those weights: its mean is the score, its variance the weighted members' variances plus their disagreement. A member
that has a predictive distribution of its own (quantile models, NGBoost, the Temporal Fusion Transformer) contributes its row-level standard deviation; the others the standard deviation of their residuals on the
held-out days.

Members are written ``kind`` or ``kind[setting=value;setting=value]``, separated by commas, for example ``ridge,lightgbm[n_estimators=100],lstm[epochs=5;hidden=16]``.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls

from .causal import CausalView
from .models import _INFO, Model, make_model
from .regimes import market_state

MEMBER = re.compile(r"\s*([a-z_0-9]+)\s*(?:\[([^\]]*)\])?\s*(?:,|$)")
MAX_MEMBERS = 8
FLOOR_STD = 0.25
WEIGHTINGS = ("equal", "dynamic", "regime", "stacked", "stacked_regime")
MIN_REGIME_DATES = 20                                                           # held-out dates a regime needs before its own stacked weights are used
MIN_STACK_ROWS = 30


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


def stack_weights(cal: np.ndarray, y: np.ndarray, shrink: float) -> tuple[np.ndarray, dict]:
    """The regularised stacking weights: non-negative least squares of ``y`` on the columns of ``cal`` (both centred on the mean of ``y``), normalised to sum to one, shrunk toward equal weights by ``shrink``.
    Too few rows, or no positive coefficient (the members add nothing together), give equal weights and say so."""
    m = cal.shape[1]
    eq = np.full(m, 1.0 / m)
    ok = np.isfinite(y) & np.isfinite(cal).all(axis=1)
    if ok.sum() < MIN_STACK_ROWS:
        return eq, {"rule": "equal weights: fewer than %d complete rows" % MIN_STACK_ROWS, "rows": int(ok.sum())}
    ybar = float(y[ok].mean())
    coef, _ = nnls(cal[ok] - ybar, y[ok] - ybar)
    if coef.sum() <= 1e-12:
        return eq, {"rule": "equal weights: no member has a positive coefficient", "rows": int(ok.sum()), "nnls": coef.tolist()}
    raw = coef / coef.sum()
    return (1.0 - shrink) * raw + shrink * eq, {"rule": "non-negative least squares, normalised, shrunk toward equal weights", "rows": int(ok.sum()), "nnls": raw.tolist(), "shrink": float(shrink)}


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
        if p["weighting"] not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {list(WEIGHTINGS)}")
        if not 0.1 <= float(p["val_fraction"]) <= 0.5 or int(p["decay_days"]) < 1 or float(p["temperature"]) < 0 or not 2 <= int(p["regimes"]) <= 4:
            raise ValueError("invalid ensemble setting(s)")
        if not 0.0 <= float(p.get("shrink", 0.5)) <= 1.0 or not 0.0 <= float(p.get("min_coverage", 0.8)) <= 1.0:
            raise ValueError("shrink and min_coverage are between 0 and 1")
        self._ds = None
        self._fwd = self._state = None
        self.members_: list = []
        self.lineage_: dict = {}

    # ------------------------------------------------------------------------------------------------------------ data access
    def bind(self, ds) -> "MemberEnsemble":
        self._ds, self._fwd, self._state = (CausalView.of(ds) if ds is not None else None), None, None
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
        bound = self._ds is not None
        lag_h = (self._ds.lag + self._ds.horizon) if bound else None                                                 # an ensemble that was never bound to a dataset does not know its label horizon: its purge is 0 and the lineage says so
        date_of = (lambda q: str(self._ds.dates[int(q)].date())) if bound else (lambda q: int(q))
        self.lineage_ = {"scheme": "chronological cross-fitting: members are fitted on the earlier training rows, predict the later ones, and those out-of-sample predictions set the calibration and the weights",
                         "members": [k for k, _ in self.specs_],
                         "stage_1": {"fit_rows": int(len(fit_rows)), "first": date_of(pos[fit_rows].min()), "last": date_of(pos[fit_rows].max()), "label_end_max": date_of(pos[fit_rows].max() + lag_h) if bound else None,
                                     "purge": int(purge) if bound else None, "producer": "each member, fitted on these rows only"},
                         "stage_2": {"rows": int(len(val_rows)), "first": date_of(start), "last": date_of(pos[val_rows].max()), "base_predictions": "out of sample: no member was fitted on a row of this stage",
                                     "sets": "calibration (intercept, slope, residual scale) and member weights"},
                         "stage_1_labels_settled_before_stage_2": bool(pos[fit_rows].max() + lag_h < start) if bound else None, "refit_full": bool(int(p["refit_full"])),
                         "deployed_members_fitted_on": "all training rows (information cutoff = the last training day)" if int(p["refit_full"]) else "stage 1 rows only"}
        first = [self._new(i).fit(Xs.iloc[fit_rows], y.iloc[fit_rows], pos[fit_rows]) for i in range(len(self.specs_))]
        Xv = Xs.iloc[val_rows]
        S = np.column_stack([m.predict(Xv) for m in first])
        yval = yv[val_rows]
        ok = np.isfinite(yval)
        self.coverage_ = np.array([float(np.isfinite(S[:, j]).mean()) for j in range(S.shape[1])])
        self.usable_ = self.coverage_ >= float(p.get("min_coverage", 0.8))
        if self.usable_.sum() < 2:
            raise ValueError(f"fewer than two members score at least {float(p.get('min_coverage', 0.8)):.0%} of the held-out rows (coverage {np.round(self.coverage_, 3).tolist()}): an ensemble needs two members it can weigh")
        self.a_, self.b_, self.sigma_ = np.zeros(S.shape[1]), np.zeros(S.shape[1]), np.zeros(S.shape[1])
        for j in range(S.shape[1]):
            okj = ok & np.isfinite(S[:, j])
            if okj.sum() < MIN_STACK_ROWS:                                                                           # a member with no held-out record is not calibrated, and carries no weight
                self.sigma_[j] = self.y_std_
                continue
            s, t = S[okj, j], yval[okj]
            b = max(float(np.cov(s, t)[0, 1] / (np.var(s, ddof=1) + 1e-18)), 0.0)
            self.a_[j], self.b_[j] = float(t.mean() - b * s.mean()), b
            self.sigma_[j] = float(np.std(t - self.a_[j] - b * s, ddof=1)) or self.y_std_
        dates_val = Xv.index.get_level_values(0)
        self.val_dates_, ic = _spearman_by_date(S, yval, dates_val.to_numpy())
        self.val_ic_ = ic
        self.edges_ = np.array([])
        if p["weighting"] in ("regime", "stacked_regime"):
            lv = self._market()["log_vol"].loc[Xs.index.get_level_values(0).min():Xs.index.get_level_values(0).max()].dropna()
            self.edges_ = np.quantile(lv, np.linspace(0, 1, int(p["regimes"]) + 1)[1:-1])
        self.regime_weights_, self.static_weights_ = None, None
        self.weights_info_ = {"rule": p["weighting"], "training_window": {"stage": "held-out (stage 2)", "first": date_of(start), "last": date_of(pos[val_rows].max()), "rows": int(len(val_rows))},
                              "coverage": {k: float(c) for (k, _), c in zip(self.specs_, self.coverage_)}, "min_coverage": float(p.get("min_coverage", 0.8)),
                              "excluded": [{"member": k, "coverage": float(c), "reason": f"scores {c:.0%} of the held-out rows, below the {float(p.get('min_coverage', 0.8)):.0%} the weights need"}
                                           for (k, _), c, u in zip(self.specs_, self.coverage_, self.usable_) if not u]}
        if p["weighting"] == "regime":
            self.regime_weights_ = self._regime_weights(pd.DatetimeIndex(self.val_dates_), ic)
        elif p["weighting"] in ("stacked", "stacked_regime"):
            self._fit_stacked(S, yval, dates_val)
        self.members_ = [self._new(i).fit(Xs, y, pos) for i in range(len(self.specs_))] if int(p["refit_full"]) else first
        with warnings.catch_warnings():                                                                              # a member with no scored held-out day has no IC: NaN, not a warning
            warnings.simplefilter("ignore", RuntimeWarning)
            self.val_ic_mean_ = np.nanmean(ic, axis=0)
        return self

    def _fit_stacked(self, S: np.ndarray, yval: np.ndarray, dates_val: pd.Index) -> None:
        """The stacking weights from the held-out stage: pooled, and (for ``stacked_regime``) per volatility regime. Members without enough coverage are not in the fit and carry weight zero."""
        shrink = float(self.params.get("shrink", 0.5))
        cols = np.flatnonzero(self.usable_)
        cal = (self.a_ + self.b_ * S)[:, cols]

        def full(w: np.ndarray) -> np.ndarray:
            out = np.zeros(S.shape[1])
            out[cols] = w
            return out

        pooled, info = stack_weights(cal, yval, shrink)
        self.static_weights_ = full(pooled)
        self.weights_info_["pooled"] = {**info, "members": [self.specs_[j][0] for j in cols]}
        if self.params["weighting"] == "stacked_regime":
            reg = self._regime_of(pd.DatetimeIndex(dates_val))
            n_reg = int(self.params["regimes"])
            out, per = np.tile(self.static_weights_, (n_reg, 1)), []
            for r in range(n_reg):
                sel = reg == r
                n_dates = int(pd.Index(dates_val[sel]).nunique())
                if n_dates < MIN_REGIME_DATES:
                    per.append({"regime": r, "dates": n_dates, "rule": "pooled weights: fewer than %d held-out dates in this regime" % MIN_REGIME_DATES})
                    continue
                w, info_r = stack_weights(cal[sel], yval[sel], shrink)
                out[r] = full(w)
                per.append({"regime": r, "dates": n_dates, **info_r})
            self.regime_weights_ = out
            self.weights_info_["regimes"] = per

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
        elif mode == "stacked":
            W = np.tile(self.static_weights_, (len(dates), 1))
        elif mode in ("regime", "stacked_regime"):
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
        usable = getattr(self, "usable_", np.ones(M, dtype=bool))                                                     # a member below the coverage floor carries no weight in any weighting
        if not usable.all():
            W = W * usable
            W = W / W.sum(axis=1, keepdims=True)
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
        fin = np.isfinite(cal)
        if not fin.all():                                                                                            # where a member has no score for a row, the others' weights are renormalised for that row
            Wr = np.where(fin, Wr, 0.0)
            tot = Wr.sum(axis=1)
            Wr = Wr / np.where(tot > 1e-12, tot, np.nan)[:, None]                                                    # no member scores the row: no prediction
            cal = np.where(fin, cal, 0.0)
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
        d = {"members": names, "lineage": self.lineage_, "weighting": self.params["weighting"], "interval_level": 0.8, "calibration": [{"member": n, "intercept": float(a), "slope": float(b), "residual_std": float(s), "heldout_ic": float(ic) if np.isfinite(ic) else None}
                                                                                   for n, a, b, s, ic in zip(names, self.a_, self.b_, self.sigma_, self.val_ic_mean_)],
             "member_diagnostics": {n: m.diagnostics() for n, m in zip(names, self.members_) if m.diagnostics()}}
        if getattr(self, "regime_weights_", None) is not None:
            d["regime_weights"] = self.regime_weights_.tolist()
            d["regime_edges"] = [float(v) for v in self.edges_]
        if getattr(self, "weights_info_", None) is not None:
            d["weights"] = self.weights_info_
            if getattr(self, "static_weights_", None) is not None:
                d["stacked_weights"] = [float(v) for v in self.static_weights_]
        return d

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_ds=None, _fwd=None, _state=None)
        return state


def build(kind: str, task: str, params: dict, seed: int) -> Model:
    return MemberEnsemble(kind, task, params, seed)
