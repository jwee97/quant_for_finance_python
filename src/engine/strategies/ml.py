"""Machine-learning strategies on the common API: walk-forward models, online forecast combination and meta-labelling.

Every learner here is trained ONLY on what the engine could have known at the decision: features are computed from point-in-time price histories and a training label (the
volatility-scaled return over the next ``horizon`` days) is used only after that horizon has elapsed, so the last ``horizon`` rows of any history are never in a training set.

* :class:`WalkForwardMLStrategy` pools the instruments into one panel of features (volatility-scaled returns over several lookbacks, volatility ratio, range position, trailing
  Sharpe) and refits a regularised linear model (``ridge``, ``lasso``, ``elastic``) or a shallow tree ensemble (``gbm``, ``forest``) every ``retrain_every`` decisions on a rolling
  window. With ``regime_conditioned=True`` separate models are fit for calm and stressed states (volatility ratio above ``stress_ratio``) and the one matching today is used. The signal
  is ``tanh`` of the forecast, its confidence the trailing hit rate of resolved forecasts (``0`` below chance).
* :class:`ForecastCombinationStrategy` combines named forecasters (any function ``(ctx, instrument_id) -> float``) with ONLINE weights: after each horizon the realised return scores every
  forecaster and the weights follow an exponentially-weighted (``hedge``) or IC-proportional update, so a forecaster that stops working loses weight without a refit.
* :class:`MetaLabelStrategy` wraps a primary strategy that proposes sides; a logistic model, fit on the outcomes of the primary's past calls, estimates the probability that the next
  call will be profitable from its features (signal strength, volatility ratio, recent hit rate) and the size follows ``2p - 1`` above ``threshold`` (López de Prado, meta-labelling).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..strategy import Schedule, Signal, Strategy, Target
from .common import ann_vol, continuous_history, risk_weights

FEATURES = ("r5", "r21", "r63", "r126", "vol_ratio", "range_pos", "sharpe63")


def _prices(ctx, iid: str, n: int) -> pd.Series:
    p = continuous_history(ctx, iid, n) if ctx.registry.is_chain(iid) else ctx.data.history(iid, n)
    p = p[p > 0]
    return p.groupby(p.index.normalize()).last() if len(p) else p


def feature_frame(p: pd.Series) -> pd.DataFrame:
    """Features for every date of a price series, each computed from data up to and including that date."""
    lp = np.log(p)
    r = lp.diff()
    vol = r.rolling(60, min_periods=30).std()
    out = pd.DataFrame(index=p.index)
    for k, name in ((5, "r5"), (21, "r21"), (63, "r63"), (126, "r126")):
        out[name] = (lp - lp.shift(k)) / (vol * np.sqrt(k))
    out["vol_ratio"] = r.rolling(21, min_periods=15).std() / r.rolling(126, min_periods=60).std()
    hi, lo = p.rolling(126, min_periods=60).max(), p.rolling(126, min_periods=60).min()
    out["range_pos"] = 2.0 * (p - lo) / (hi - lo).replace(0, np.nan) - 1.0
    out["sharpe63"] = r.rolling(63, min_periods=40).mean() / r.rolling(63, min_periods=40).std()
    return out.clip(-6, 6)


def labels(p: pd.Series, horizon: int) -> pd.Series:
    """The volatility-scaled return over the next ``horizon`` observations (NaN for the last ``horizon``: not yet realised)."""
    lp = np.log(p)
    vol = lp.diff().rolling(60, min_periods=30).std()
    return ((lp.shift(-horizon) - lp) / (vol * np.sqrt(horizon))).clip(-6, 6)


def _make_model(kind: str, seed: int = 0):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    if kind == "ridge":
        return Ridge(alpha=50.0)
    if kind == "lasso":
        return Lasso(alpha=0.02)
    if kind == "elastic":
        return ElasticNet(alpha=0.02, l1_ratio=0.5)
    if kind == "gbm":
        return GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7, random_state=seed)
    if kind == "forest":
        return RandomForestRegressor(n_estimators=60, max_depth=4, min_samples_leaf=40, random_state=seed, n_jobs=1)
    raise ValueError("model must be ridge, lasso, elastic, gbm or forest")


class _TrackRecord:
    """Pending forecasts and the trailing hit rate of the resolved ones."""

    def __init__(self, horizon_days: float, window: int = 100):
        self.horizon, self.window = pd.Timedelta(days=horizon_days), window
        self.pending: list[tuple[pd.Timestamp, str, float, float]] = []
        self.hits: list[float] = []

    def add(self, ts, iid, forecast, price):
        self.pending.append((ts, iid, float(forecast), float(price)))

    def resolve(self, ctx, price_fn):
        keep = []
        for ts, iid, f, p0 in self.pending:
            if ctx.ts - ts >= self.horizon:
                p1 = price_fn(iid)
                if np.isfinite(p1) and p0 > 0 and f != 0:
                    self.hits.append(float(np.sign(np.log(p1 / p0)) == np.sign(f)))
            else:
                keep.append((ts, iid, f, p0))
        self.pending = keep

    def hit_rate(self) -> float:
        h = self.hits[-self.window:]
        return float(np.mean(h)) if len(h) >= 20 else float("nan")


def _price_at(ctx, iid):
    if ctx.registry.is_chain(iid):
        s = continuous_history(ctx, iid, 3)
        return float(s.iloc[-1]) if len(s) else float("nan")
    return ctx.data.mid(iid)


# ------------------------------------------------------------------------------------------------------------------------- walk-forward model
class WalkForwardMLStrategy(Strategy):
    name = "ml_walkforward"
    schedule = Schedule("weekly", "16:30", 4, "US")

    def __init__(self, instruments, model: str = "ridge", horizon: int = 5, retrain_every: int = 13, train_window: int = 750, min_samples: int = 400, regime_conditioned: bool = False,
                 stress_ratio: float = 1.2, target_vol: float = 0.10, vol_window: int = 60, max_leverage: float = 3.0, forecast_scale: float = 0.15, seed: int = 0, name: str | None = None):
        self.instruments = list(instruments)
        self.model_kind, self.horizon, self.retrain_every, self.train_window, self.min_samples = model, horizon, retrain_every, train_window, min_samples
        self.regime_conditioned, self.stress_ratio, self.forecast_scale = regime_conditioned, stress_ratio, forecast_scale
        self.target_vol, self.vol_window, self.max_leverage, self.seed = target_vol, vol_window, max_leverage, seed
        if name:
            self.name = name
        self._models: dict = {}
        self._since = None
        self._track = _TrackRecord(horizon * 7.0 / 5.0)
        self.fits: list[dict] = []

    def _panel(self, ctx):
        xs, ys, regs = [], [], []
        for iid in self.instruments:
            p = _prices(ctx, iid, self.train_window + 300)
            if len(p) < 200:
                continue
            f, y = feature_frame(p), labels(p, self.horizon)
            frame = f.assign(y=y).dropna()
            frame = frame.iloc[: max(len(frame) - 0, 0)].tail(self.train_window)
            xs.append(frame[list(FEATURES)])
            ys.append(frame["y"])
        if not xs:
            return None, None
        return pd.concat(xs), pd.concat(ys)

    def _fit(self, ctx):
        X, y = self._panel(ctx)
        if X is None or len(X) < self.min_samples:
            return
        models = {}
        groups = {"all": np.ones(len(X), bool)}
        if self.regime_conditioned:
            stress = (X["vol_ratio"] > self.stress_ratio).to_numpy()
            groups = {"calm": ~stress, "stress": stress, "all": np.ones(len(X), bool)}
        for g, mask in groups.items():
            if mask.sum() < max(self.min_samples // 2, 50):
                continue
            m = _make_model(self.model_kind, self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X[mask].to_numpy(), y[mask].to_numpy())
            models[g] = m
        if models:
            self._models = models
            self.fits.append({"ts": ctx.ts, "n": int(len(X)), "groups": sorted(models)})

    def _predict(self, row: pd.Series) -> float:
        if not self._models:
            return float("nan")
        g = "all"
        if self.regime_conditioned:
            g = "stress" if row["vol_ratio"] > self.stress_ratio else "calm"
            g = g if g in self._models else "all"
        m = self._models.get(g) or next(iter(self._models.values()))
        return float(m.predict(row[list(FEATURES)].to_numpy(dtype=float).reshape(1, -1))[0])

    def generate_signals(self, ctx):
        self._since = (self._since or 0) + 1
        if not self._models or self._since >= self.retrain_every:
            self._fit(ctx)
            self._since = 0
        self._track.resolve(ctx, lambda i: _price_at(ctx, i))
        out = []
        hit = self._track.hit_rate()
        conf = 1.0 if not np.isfinite(hit) else float(np.clip(2.0 * hit - 1.0 + 0.5, 0.0, 1.0))
        for iid in self.instruments:
            p = _prices(ctx, iid, 400)
            if len(p) < 150:
                continue
            row = feature_frame(p).iloc[-1]
            if row[list(FEATURES)].isna().any():
                continue
            f = self._predict(row)
            if not np.isfinite(f):
                continue
            self._track.add(ctx.ts, iid, f, float(p.iloc[-1]))
            out.append(Signal(iid, float(np.tanh(f / self.forecast_scale)), conf, meta={"forecast": f}))
        return out

    def map_to_targets(self, ctx, signals):
        sig = {s.instrument_id: s.value * s.confidence for s in signals}
        vols = {i: ann_vol(ctx, i, self.vol_window) for i in sig}
        return [Target(i, weight=w) for i, w in risk_weights(sig, vols, self.target_vol, self.max_leverage).items()]


# ---------------------------------------------------------------------------------------------------------------------- online forecast combination
def momentum_forecaster(lookback: int = 63):
    def f(ctx, iid):
        p = _prices(ctx, iid, lookback + 80)
        if len(p) <= lookback + 5:
            return float("nan")
        lp = np.log(p)
        vol = lp.diff().tail(60).std()
        return float(np.tanh((lp.iloc[-1] - lp.iloc[-1 - lookback]) / (vol * np.sqrt(lookback)))) if vol > 0 else float("nan")
    return f


def reversal_forecaster(lookback: int = 5):
    base = momentum_forecaster(lookback)
    return lambda ctx, iid: -base(ctx, iid)


def range_forecaster(window: int = 126):
    def f(ctx, iid):
        p = _prices(ctx, iid, window + 5)
        if len(p) < window // 2:
            return float("nan")
        w = p.tail(window)
        return float(2.0 * (p.iloc[-1] - w.min()) / (w.max() - w.min()) - 1.0) if w.max() > w.min() else 0.0
    return f


class ForecastCombinationStrategy(Strategy):
    name = "forecast_combination"
    schedule = Schedule("weekly", "16:30", 4, "US")

    def __init__(self, instruments, forecasters: dict, horizon_days: float = 7.0, rule: str = "hedge", eta: float = 2.0, decay: float = 0.97, target_vol: float = 0.10, vol_window: int = 60,
                 max_leverage: float = 3.0, name: str | None = None):
        if rule not in ("hedge", "ic"):
            raise ValueError("rule must be hedge or ic")
        self.instruments, self.forecasters = list(instruments), dict(forecasters)
        self.horizon, self.rule, self.eta, self.decay = pd.Timedelta(days=horizon_days), rule, eta, decay
        self.target_vol, self.vol_window, self.max_leverage = target_vol, vol_window, max_leverage
        if name:
            self.name = name
        self.weights = {k: 1.0 / len(self.forecasters) for k in self.forecasters}
        self._scores = {k: 0.0 for k in self.forecasters}
        self._pending: list[tuple[pd.Timestamp, str, dict, float]] = []
        self.weight_history: list[dict] = []

    def _update(self, ctx):
        keep, rows = [], {k: [] for k in self.forecasters}
        for ts, iid, fc, p0 in self._pending:
            if ctx.ts - ts >= self.horizon:
                p1 = _price_at(ctx, iid)
                if np.isfinite(p1) and p0 > 0:
                    r = float(np.log(p1 / p0))
                    for k, f in fc.items():
                        if np.isfinite(f):
                            rows[k].append((f, r))
            else:
                keep.append((ts, iid, fc, p0))
        self._pending = keep
        for k, obs in rows.items():
            if len(obs) < 2:
                continue
            f, r = np.array(obs).T
            if self.rule == "hedge":                                       # reward = average of sign(f) x standardised return: in [-1, 1]
                s = float(np.mean(np.sign(f) * np.tanh(r / (np.std(r) + 1e-12))))
            else:
                s = float(np.corrcoef(f, r)[0, 1]) if np.std(f) > 0 and np.std(r) > 0 else 0.0
            self._scores[k] = self.decay * self._scores[k] + s
        if self.rule == "hedge":
            raw = {k: float(np.exp(self.eta * v)) for k, v in self._scores.items()}
        else:
            raw = {k: max(v, 0.0) + 1e-3 for k, v in self._scores.items()}
        z = sum(raw.values())
        self.weights = {k: v / z for k, v in raw.items()}
        self.weight_history.append({"ts": ctx.ts, **self.weights})

    def generate_signals(self, ctx):
        self._update(ctx)
        out = []
        for iid in self.instruments:
            fc = {k: fn(ctx, iid) for k, fn in self.forecasters.items()}
            ok = {k: v for k, v in fc.items() if np.isfinite(v)}
            if not ok:
                continue
            w = np.array([self.weights[k] for k in ok])
            v = float(np.dot(w / w.sum(), list(ok.values())))
            conf = float(abs(np.dot(w / w.sum(), np.sign(list(ok.values())))))             # share of weight that agrees on the direction
            p0 = _price_at(ctx, iid)
            if np.isfinite(p0):
                self._pending.append((ctx.ts, iid, fc, p0))
            out.append(Signal(iid, v, conf, meta=fc))
        return out

    def map_to_targets(self, ctx, signals):
        sig = {s.instrument_id: s.value * s.confidence for s in signals}
        vols = {i: ann_vol(ctx, i, self.vol_window) for i in sig}
        return [Target(i, weight=w) for i, w in risk_weights(sig, vols, self.target_vol, self.max_leverage).items()]


# ------------------------------------------------------------------------------------------------------------------------------- meta-labelling
class MetaLabelStrategy(Strategy):
    name = "meta_label"

    def __init__(self, primary: Strategy, horizon_days: float = 7.0, threshold: float = 0.52, min_samples: int = 60, warmup_size: float = 0.5, refit_every: int = 10, c: float = 1.0,
                 name: str | None = None):
        self.primary = primary
        self.schedule = primary.schedule
        self.horizon, self.threshold, self.min_samples, self.warmup_size, self.refit_every, self.c = pd.Timedelta(days=horizon_days), threshold, min_samples, warmup_size, refit_every, c
        if name:
            self.name = name
        self._pending: list[tuple[pd.Timestamp, str, float, np.ndarray, float]] = []
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._model = None
        self._n = 0
        self._recent: dict[str, list[float]] = {}
        self.probabilities: list[dict] = []

    def on_start(self, ctx):
        self.primary.on_start(ctx)

    def on_instrument_event(self, ctx, event):
        self.primary.on_instrument_event(ctx, event)

    def _features(self, ctx, signal: Signal) -> np.ndarray:
        p = _prices(ctx, signal.instrument_id, 300)
        if len(p) < 130:
            return np.full(5, np.nan)
        f = feature_frame(p).iloc[-1]
        recent = self._recent.get(signal.instrument_id, [])
        return np.array([abs(signal.value), signal.confidence, float(f["vol_ratio"]), float(f["sharpe63"]) * np.sign(signal.value), float(np.mean(recent[-10:])) if recent else 0.5])

    def _resolve(self, ctx):
        keep = []
        for ts, iid, side, x, p0 in self._pending:
            if ctx.ts - ts >= self.horizon:
                p1 = _price_at(ctx, iid)
                if np.isfinite(p1) and p0 > 0 and np.all(np.isfinite(x)):
                    won = float(np.sign(np.log(p1 / p0)) == np.sign(side))
                    self._X.append(x)
                    self._y.append(won)
                    self._recent.setdefault(iid, []).append(won)
            else:
                keep.append((ts, iid, side, x, p0))
        self._pending = keep

    def _refit(self):
        from sklearn.linear_model import LogisticRegression

        y = np.array(self._y)
        if len(y) < self.min_samples or y.min() == y.max():
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = LogisticRegression(C=self.c, max_iter=200).fit(np.array(self._X), y)

    def on_schedule(self, ctx):
        self._n += 1
        self._resolve(ctx)
        if self._n % self.refit_every == 0:
            self._refit()
        signals = self.primary.generate_signals(ctx)
        sized: list[Signal] = []
        for s in signals:
            x = self._features(ctx, s)
            if self._model is not None and np.all(np.isfinite(x)):
                prob = float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
                size = max(2.0 * prob - 1.0, 0.0) * 2.0 if prob >= self.threshold else 0.0       # 2p - 1 scaled so that p = 0.75 gives full size
                size = min(size, 1.0)
            else:
                prob, size = float("nan"), self.warmup_size
            p0 = _price_at(ctx, s.instrument_id)
            if np.isfinite(p0) and s.value != 0:
                self._pending.append((ctx.ts, s.instrument_id, s.value, x, p0))
            self.probabilities.append({"ts": ctx.ts, "instrument_id": s.instrument_id, "p": prob, "size": size})
            sized.append(Signal(s.instrument_id, s.value, s.confidence * size, s.horizon, {**s.meta, "meta_p": prob, "meta_size": size}))
        targets = self.primary.map_to_targets(ctx, sized)
        ctx.set_targets(targets)
