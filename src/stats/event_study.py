"""Event studies (MacKinlay 1997; Campbell, Lo & MacKinlay ch. 4).

An event study asks whether an asset's return around a dated event differs from what it would have earned without the event. The procedure:

1. **Estimation window** (``estimation=(-250, -30)`` trading days relative to the event): fit the normal-return model on data strictly BEFORE the event window.
2. **Event window** (``window=(-5, 5)``): abnormal return ``AR = R - R_normal`` for each day relative to the event.
3. **Aggregate**: cumulative abnormal returns ``CAR(a, b) = sum AR`` per event, averaged over events (``CAAR``), and tested.

Normal-return models: ``market`` (``R = a + b R_m``), ``market_adjusted`` (``R - R_m``), ``mean`` (constant mean) or ``factor`` (several factor returns).
Tests of the cross-sectional mean CAR: the plain cross-sectional t, the Patell (1976) standardised test, the Boehmer-Musumeci-Poulsen (1991) standardised cross-sectional
test, which is robust to event-induced variance, the Corrado (1989) rank test and the sign test. Overlapping events on the same date make the cross-sectional
tests too liberal; ``cluster_by_date=True`` collapses events that share a date into one observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass
class EventStudyResult:
    ar: pd.DataFrame                  # relative day x event: abnormal returns
    car: pd.DataFrame                 # relative day x event: cumulative abnormal returns from the start of the window
    sigma: pd.Series                  # event: residual standard deviation in the estimation window
    n_estimation: pd.Series
    events: pd.DataFrame
    window: tuple
    summary: pd.DataFrame             # per relative day: AAR, CAAR, tests

    def car_window(self, a: int, b: int) -> pd.Series:
        """Per-event cumulative abnormal return from day ``a`` to day ``b`` inclusive."""
        return self.ar.loc[a:b].sum()

    def test_car(self, a: int, b: int) -> dict:
        car = self.car_window(a, b)
        L = b - a + 1
        n = car.notna().sum()
        sd = self.sigma.reindex(car.index)
        scar = car / (sd * np.sqrt(L))
        t_cs = car.mean() / (car.std(ddof=1) / np.sqrt(n))
        patell_z = scar.sum() / np.sqrt(n * (self.n_estimation.mean() - 2) / (self.n_estimation.mean() - 4)) if self.n_estimation.mean() > 4 else np.nan
        bmp = scar.mean() / (scar.std(ddof=1) / np.sqrt(n))
        sign = (car > 0).sum()
        z_sign = (sign - 0.5 * n) / np.sqrt(0.25 * n)
        return {"caar": float(car.mean()), "n_events": int(n), "t_cross_sectional": float(t_cs), "p_cross_sectional": float(2 * sps.t.sf(abs(t_cs), n - 1)),
                "z_patell": float(patell_z), "p_patell": float(2 * sps.norm.sf(abs(patell_z))), "t_bmp": float(bmp), "p_bmp": float(2 * sps.t.sf(abs(bmp), n - 1)),
                "z_sign": float(z_sign), "p_sign": float(2 * sps.norm.sf(abs(z_sign))), "share_positive": float(sign / n)}


def _fit_normal(model: str, r: np.ndarray, factors: np.ndarray | None):
    if model == "mean":
        return np.array([r.mean()]), None
    if model == "market_adjusted":
        return np.array([0.0, 1.0]), None
    Z = np.column_stack([np.ones(len(r)), factors])
    beta = np.linalg.lstsq(Z, r, rcond=None)[0]
    return beta, Z


def event_study(returns: pd.DataFrame, events: pd.DataFrame, market: pd.Series | pd.DataFrame | None = None, model: str = "market", estimation=(-250, -30),
                window=(-5, 5), min_estimation: int = 60, cluster_by_date: bool = False) -> EventStudyResult:
    """Run an event study. ``returns`` is dates x assets; ``events`` has columns ``asset`` and ``date`` (the event day, mapped to the next trading day on
    or after it). ``market`` is a return series (or a frame of factor returns for ``model='factor'``). Events without enough data are dropped and reported
    in ``events`` with a ``used`` flag."""
    if model not in ("market", "market_adjusted", "mean", "factor"):
        raise ValueError("model must be market, market_adjusted, mean or factor")
    if model != "mean" and market is None:
        raise ValueError(f"model '{model}' needs market returns")
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    if cluster_by_date:
        ev = ev.drop_duplicates("date", keep="first")
    index = returns.index
    mkt = None
    if market is not None:
        mkt = (market.to_frame("m") if isinstance(market, pd.Series) else market).reindex(index)
    rel = np.arange(window[0], window[1] + 1)
    ar_cols, full_cols, sigmas, n_est, used = {}, {}, {}, {}, []
    all_rel = np.arange(estimation[0], window[1] + 1)
    for i, row in ev.reset_index(drop=True).iterrows():
        asset = row["asset"]
        if asset not in returns.columns:
            used.append(False)
            continue
        pos = index.searchsorted(row["date"])
        if pos >= len(index):
            used.append(False)
            continue
        e0, e1 = pos + estimation[0], pos + estimation[1]
        w0, w1 = pos + window[0], pos + window[1]
        if e0 < 0 or w1 >= len(index) or w0 <= e1:
            used.append(False)
            continue
        r_est = returns[asset].iloc[e0: e1 + 1]
        f_est = mkt.iloc[e0: e1 + 1] if mkt is not None else None
        ok = r_est.notna() & (f_est.notna().all(axis=1) if f_est is not None else True)
        if ok.sum() < min_estimation:
            used.append(False)
            continue
        r_e = r_est[ok].to_numpy()
        beta, Z = _fit_normal(model, r_e, f_est[ok].to_numpy() if f_est is not None else None)
        if model == "mean":
            resid = r_e - beta[0]
            k = 1
        elif model == "market_adjusted":
            resid = r_e - f_est[ok].iloc[:, 0].to_numpy()
            k = 0
        else:
            resid = r_e - Z @ beta
            k = Z.shape[1]
        sd = float(np.sqrt(resid @ resid / (len(resid) - k)))
        r_w = returns[asset].iloc[w0: w1 + 1].to_numpy()
        if model == "mean":
            normal = np.full(len(r_w), beta[0])
        elif model == "market_adjusted":
            normal = mkt.iloc[w0: w1 + 1].iloc[:, 0].to_numpy()
        else:
            normal = np.column_stack([np.ones(len(r_w)), mkt.iloc[w0: w1 + 1].to_numpy()]) @ beta
        name = f"{asset}@{row['date'].date()}#{i}"
        ar_cols[name] = pd.Series(r_w - normal, index=rel)
        full = pd.Series(np.nan, index=all_rel)
        resid_full = (r_est - (beta[0] if model == "mean" else (f_est.iloc[:, 0] if model == "market_adjusted" else 0.0)))
        if model in ("market", "factor"):
            resid_full = r_est - np.column_stack([np.ones(len(r_est)), f_est.to_numpy()]) @ beta
        full.loc[np.arange(estimation[0], estimation[1] + 1)] = np.asarray(resid_full, dtype=float)
        full.loc[rel] = r_w - normal
        full_cols[name] = full
        sigmas[name], n_est[name] = sd, int(ok.sum())
        used.append(True)
    ev = ev.reset_index(drop=True)
    ev["used"] = used
    ar = pd.DataFrame(ar_cols)
    if ar.empty:
        raise ValueError("no event had enough data; widen the data or shrink the estimation window")
    car = ar.cumsum()
    sigma, nest = pd.Series(sigmas), pd.Series(n_est)
    n = ar.notna().sum(axis=1)
    aar = ar.mean(axis=1)
    sar = ar / sigma
    t_aar = aar / (ar.std(axis=1, ddof=1) / np.sqrt(n))
    bmp_aar = sar.mean(axis=1) / (sar.std(axis=1, ddof=1) / np.sqrt(n))
    # Corrado (1989) rank test: rank each event's abnormal returns over its estimation AND event window, K = rank / (L + 1)
    full = pd.DataFrame(full_cols)
    K = full.rank(axis=0) / (full.notna().sum(axis=0) + 1.0)
    kbar = K.mean(axis=1)
    s_k = float(np.sqrt(np.nanmean((kbar.dropna() - 0.5) ** 2)))
    corrado = (kbar.loc[rel] - 0.5) / s_k
    summary = pd.DataFrame({"AAR": aar, "CAAR": aar.cumsum(), "t": t_aar, "t_bmp": bmp_aar, "z_corrado": corrado, "n": n})
    return EventStudyResult(ar, car, sigma, nest, ev, window, summary)


def buy_and_hold_abnormal_return(returns: pd.DataFrame, benchmark: pd.Series, events: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """BHAR: the compounded return of the asset minus the compounded return of the benchmark over ``horizon`` days AFTER the event (day +1 onwards)."""
    out = {}
    for i, row in events.reset_index(drop=True).iterrows():
        pos = returns.index.searchsorted(pd.Timestamp(row["date"]))
        a, b = pos + 1, pos + 1 + horizon
        if row["asset"] not in returns.columns or b > len(returns):
            continue
        asset = (1.0 + returns[row["asset"]].iloc[a:b]).prod() - 1.0
        bench = (1.0 + benchmark.reindex(returns.index).iloc[a:b]).prod() - 1.0
        out[f"{row['asset']}@{pd.Timestamp(row['date']).date()}#{i}"] = asset - bench
    return pd.Series(out)
