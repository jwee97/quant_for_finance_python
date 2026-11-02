"""Cross-asset transfer with explicit source and target domains and no leakage between them (spec v3, Release 8, P3-06). Optional: it needs only scikit-learn and is off unless asked for.

The question is whether what was learned on one set of assets helps on another. The design makes the ways the answer can be flattered impossible, and reports what remains.

* **Disjoint domains.** The source and the target share no security: a model that has seen a name cannot be said to transfer to it.
* **A cutoff in time.** The source model is fitted on source rows whose labels had **ended by** ``source_end`` (a row's label ends ``lag + horizon`` sessions after its decision), and the target is evaluated
  only on decisions from ``eval_start``, which is not before ``source_end``: market-wide moves in the target's evaluation period were not in the source's training data.
* **Preprocessing at the cutoff.** The scaler (and, for the ``pca`` representation, the components) is fitted on the source training rows only and applied unchanged to the target. The from-scratch model
  scales with its own target training rows. Nothing is fitted on a row that is evaluated.
* **The same output semantics.** Both domains must carry labels of the same horizon and lag; rows whose label interval differs from the declared one are refused.
* **The comparison.** The transferred model and a from-scratch model (trained on the target's own history before the evaluation, which may be short: that is the case transfer is for) are scored on
  the same target-domain rows by the daily rank IC, with a Newey-West interval on their difference. **Negative transfer is a result, not an error**: if the transferred model is worse than scratch with an
  interval that excludes zero, or is itself harmful, the verdict says ``negative_transfer``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from . import alpha as al
from . import inference as inf

VERSION = "transfer-v1"
MIN_ROWS = 60


class TransferError(ValueError):
    """A configuration that would leak between the domains or compare unlike things."""


def check_split(source_assets, target_assets) -> None:
    shared = sorted(set(source_assets) & set(target_assets))
    if shared:
        raise TransferError(f"the source and the target domains share {shared}: a model that has seen a security cannot be said to transfer to it")
    if not source_assets or not target_assets:
        raise TransferError("both domains need at least one asset")


def label_end(dates: pd.Index, trading_dates: pd.DatetimeIndex, lag: int, horizon: int) -> pd.Series:
    """The date each decision's label ends: ``lag + horizon`` sessions after it, on the real date index (the last decisions have no label end yet and get NaT)."""
    pos = trading_dates.get_indexer(pd.DatetimeIndex(dates))
    end = pos + int(lag) + int(horizon)
    vals = [trading_dates[e] if 0 <= p and e < len(trading_dates) else pd.NaT for p, e in zip(pos, end)]
    return pd.Series(vals, index=dates)


class Scaler:
    """Column means and standard deviations fitted on the rows it is given and applied unchanged to any others."""

    def fit(self, X: np.ndarray) -> "Scaler":
        self.mean_, self.std_ = np.nanmean(X, axis=0), np.nanstd(X, axis=0, ddof=1)
        self.std_ = np.where(self.std_ > 0, self.std_, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


def _wide(s: pd.Series) -> pd.DataFrame:
    return s.unstack("asset") if isinstance(s.index, pd.MultiIndex) else s.to_frame()


def run(X: pd.DataFrame, y: pd.Series, trading_dates: pd.DatetimeIndex, source_assets: list[str], target_assets: list[str], *, source_end, eval_start, target_history_end=None, lag: int = 1, horizon: int = 5,
        representation: str = "model", n_components: int = 3, alpha: float = 10.0, embargo: int = 0, horizon_check: pd.Series | None = None, level: float = 0.95, min_assets: int = 4) -> dict:
    """Fit on the source, apply to the target, and compare with a model trained on the target's own earlier rows. ``X`` and ``y`` have a (date, asset) index; ``y`` is the same label in both domains.
    ``target_history_end`` (default: the day before ``eval_start``) bounds the target rows the from-scratch model may train on, by their label ends."""
    check_split(list(source_assets), list(target_assets))
    if representation not in ("model", "pca"):
        raise TransferError("the representation is 'model' (the fitted regression) or 'pca' (components learned on the source, then a regression on them)")
    src_end, ev0 = pd.Timestamp(source_end), pd.Timestamp(eval_start)
    if ev0 < src_end:
        raise TransferError(f"the evaluation starts {ev0.date()}, before the source cutoff {src_end.date()}: the target's evaluation period would overlap the period the source model learned from")
    hist_end = pd.Timestamp(target_history_end) if target_history_end is not None else ev0 - pd.Timedelta(days=1)
    if hist_end >= ev0:
        raise TransferError("the target history the scratch model trains on must end before the evaluation starts")
    dates = X.index.get_level_values(0)
    assets = X.index.get_level_values(1)
    ends = label_end(dates, trading_dates, lag, horizon)
    if horizon_check is not None:                                                                  # labels built with another horizon or lag are not the same quantity
        declared = (horizon_check.reindex(X.index) if isinstance(horizon_check.index, pd.MultiIndex) else horizon_check)
        if declared.notna().any() and not (declared.dropna().astype(int) == int(lag) + int(horizon)).all():
            raise TransferError("the labels of the two domains do not share one lag and horizon: the output would mean different things in each")
    ok = y.reindex(X.index).notna().to_numpy() & X.notna().all(axis=1).to_numpy()
    in_src = np.isin(assets, list(source_assets))
    in_tgt = np.isin(assets, list(target_assets))
    end_arr = pd.to_datetime(ends.to_numpy())
    src_rows = ok & in_src & (end_arr <= np.datetime64(src_end)) & ~pd.isna(end_arr)
    tgt_hist = ok & in_tgt & (end_arr <= np.datetime64(hist_end)) & ~pd.isna(end_arr)
    tgt_eval = ok & in_tgt & (np.asarray(dates) >= np.datetime64(ev0))
    n_src, n_hist, n_eval = int(src_rows.sum()), int(tgt_hist.sum()), int(tgt_eval.sum())
    out = {"version": VERSION, "source_assets": sorted(source_assets), "target_assets": sorted(target_assets), "source_end": str(src_end.date()), "eval_start": str(ev0.date()), "target_history_end": str(hist_end.date()),
           "lag": int(lag), "horizon": int(horizon), "representation": representation, "rows": {"source_train": n_src, "target_history": n_hist, "target_eval": n_eval},
           "label_rule": f"a row's label ends {int(lag) + int(horizon)} sessions after its decision; source rows need that end on or before the source cutoff"}
    if n_src < MIN_ROWS or n_eval < MIN_ROWS:
        return {**out, "eligibility": "insufficient", "reason": f"{n_src} source training rows and {n_eval} target evaluation rows; at least {MIN_ROWS} of each are needed"}
    Xs, ys = X.to_numpy(dtype=float)[src_rows], y.reindex(X.index).to_numpy(dtype=float)[src_rows]
    scaler_s = Scaler().fit(Xs)
    pca_detail = None
    if representation == "pca":
        pca = PCA(n_components=min(int(n_components), Xs.shape[1]), random_state=0).fit(scaler_s.transform(Xs))
        rep_s = lambda A: pca.transform(scaler_s.transform(A))                                        # noqa: E731
        pca_detail = {"components": int(pca.n_components_), "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_], "fitted_on": "source training rows only"}
    else:
        rep_s = scaler_s.transform
    src_model = Ridge(alpha=alpha).fit(rep_s(Xs), ys)
    Xe = X.to_numpy(dtype=float)[tgt_eval]
    pred_transfer = src_model.predict(rep_s(Xe))
    pred_scratch = None
    if n_hist >= MIN_ROWS:
        Xh, yh = X.to_numpy(dtype=float)[tgt_hist], y.reindex(X.index).to_numpy(dtype=float)[tgt_hist]
        scaler_t = Scaler().fit(Xh)
        scratch = Ridge(alpha=alpha).fit(scaler_t.transform(Xh), yh)
        pred_scratch = scratch.predict(scaler_t.transform(Xe))
    idx = X.index[tgt_eval]
    truth = _wide(y.reindex(X.index)[tgt_eval])
    ics = {}
    for name, p in (("transfer", pred_transfer), ("scratch", pred_scratch)):
        if p is None:
            continue
        tab = al.ic_table(_wide(pd.Series(p, index=idx)).reindex(truth.index), truth, min_assets)
        ics[name] = tab["ic"]
    res = {k: inf.hac_mean(v.dropna(), horizon=int(horizon), level=level, label=f"rank IC of the {k} model on the target domain").as_dict() for k, v in ics.items()}
    out.update(eligibility="applied", ic=res, scaler={"fitted_on": "source training rows only", "mean": [float(m) for m in scaler_s.mean_]})
    if pca_detail is not None:
        out["pca"] = pca_detail
    if "scratch" not in ics:
        return {**out, "verdict": "no_baseline", "reason": f"the target has only {n_hist} usable rows before the evaluation: no from-scratch model could be trained, so transfer cannot be compared with it (the transfer IC stands alone)"}
    diff = (ics["transfer"] - ics["scratch"]).dropna()
    d = inf.hac_mean(diff, horizon=int(horizon), level=level, label="transfer minus scratch rank IC on the target domain").as_dict()
    out["difference"] = d
    t_ic, s_ic = res["transfer"].get("value"), res["scratch"].get("value")
    if d["eligibility"] != "applied" or d.get("interval") is None or t_ic is None:
        verdict, why = "inconclusive", "the difference could not be estimated"
    elif d["interval"][1] < 0 or (t_ic < 0 and res["transfer"].get("interval") and res["transfer"]["interval"][1] < 0):
        verdict, why = "negative_transfer", f"the transferred model is worse than training on the target alone (difference {d['value']:+.4f}, interval {d['interval'][0]:+.4f} to {d['interval'][1]:+.4f}) or is itself harmful on the target"
    elif d["interval"][0] > 0:
        verdict, why = "positive_transfer", f"the transferred model beats training on the target alone (difference {d['value']:+.4f}, interval {d['interval'][0]:+.4f} to {d['interval'][1]:+.4f})"
    else:
        verdict, why = "inconclusive", f"the interval ({d['interval'][0]:+.4f} to {d['interval'][1]:+.4f}) includes zero: transfer is not distinguishable from training on the target"
    out.update(verdict=verdict, reason=why, transfer_ic=t_ic, scratch_ic=s_ic)
    out["limitations"] = ["one fit at the cutoff, no refits: the models age through the evaluation period", "the IC is the daily rank IC across the target assets; a small target universe makes each day noisy",
                          "cross-domain correlation in time is excluded by the cutoff, not modelled"]
    return out
