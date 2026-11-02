"""Off-policy evaluation of a target policy from logged decisions, only where the logged data can support it (spec v3, Release 8, P3-08). Optional: nothing here runs unless asked.

The question is what a different policy *would* have earned, from data collected under another one. That is answerable only under conditions, and the module refuses to produce a number when they do
not hold, instead of producing one that looks like an answer:

* **Logged behaviour is required.** The probability the behaviour policy gave to *every* action in every context (``behavior_prob_all``) must be recorded, and so must the action that was taken. A
  propensity that is missing, zero for the action taken, or a row that does not sum to one makes the evaluation ``not_applicable``.
* **Support is required.** Wherever the target policy puts probability on an action the behaviour policy could not have taken (probability zero), nothing can be said about that action: the evaluation is
  ``blocked``, with the count of the contexts that violate support. No estimate is returned.
* **Enough effective sample is required.** The importance weights ``w = pi_target(a | x) / pi_behavior(a | x)`` have an effective sample size ``(sum w)^2 / sum w^2``; below ``min_ess`` the estimate is
  dominated by a few rows and the evaluation is ``blocked``.
* **What is estimated and how.** IPS (the mean of ``w r``; unbiased under the assumptions), SNIPS (``sum w r / sum w``; biased, lower variance) and the doubly robust estimate (a reward model plus
  importance-weighted residuals; unbiased if either the weights or the model are right), each with a bootstrap interval. Clipping the weights, if asked for, is disclosed with the share clipped and the bias it
  introduces.
* **Assumptions, stated.** Positivity (support), no unobserved factor that drove the logged action beyond the recorded context (guaranteed only when the propensities are the true ones), stationarity of
  rewards given context and action, and no interference between decisions. They cannot be tested from the data; the result lists them.
"""

from __future__ import annotations

import numpy as np

VERSION = "offpolicy-v1"
ASSUMPTIONS = ["positivity: every action the target policy might take had a positive chance under the behaviour policy (checked on the logged probabilities)",
               "ignorability: the logged action depended on the recorded context only (true if the logged propensities are the true ones; not testable from the data)",
               "stationarity: the reward given context and action did not change between the logged period and the one the target policy would act in",
               "no interference: one decision does not change the rewards of others (or of its own later decisions) beyond what the context records"]


def _na(status: str, reason: str, **extra) -> dict:
    return {"version": VERSION, "eligibility": status, "estimates": None, "reason": reason, "assumptions": ASSUMPTIONS, **extra}


def effective_sample_size(w) -> float:
    w = np.asarray(w, dtype=float)
    s2 = float((w ** 2).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def evaluate(action, reward, behavior_prob_all, target_prob, *, q_hat=None, min_ess: float = 30.0, clip: float | None = None, n_boot: int = 500, block: int = 1, seed: int = 0, level: float = 0.95) -> dict:
    """Evaluate ``target_prob`` (rows: contexts, columns: actions) on logged ``(action, reward)`` with the behaviour policy's full ``behavior_prob_all``. ``q_hat`` is an optional reward model
    (rows: contexts, columns: actions) for the doubly robust estimate; it must be fitted on data the evaluation does not use for the residuals (cross-fitted by the caller)."""
    a = None if action is None else np.asarray(action)
    r = None if reward is None else np.asarray(reward, dtype=float)
    if behavior_prob_all is None or target_prob is None or a is None or r is None:
        return _na("not_applicable", "the logged data lack the behaviour policy's action probabilities, the action taken, the reward or the target policy: without the logging policy there is no off-policy estimate")
    B, T = np.asarray(behavior_prob_all, dtype=float), np.asarray(target_prob, dtype=float)
    n = len(a)
    if B.ndim != 2 or T.shape != B.shape or B.shape[0] != n or len(r) != n:
        return _na("not_applicable", "the action, reward and the two probability tables must have one row per logged decision and one column per action")
    if not (np.isfinite(B).all() and np.isfinite(T).all() and np.isfinite(r).all()):
        return _na("not_applicable", "missing values in the logged probabilities or rewards: they are not filled in")
    if not (np.allclose(B.sum(axis=1), 1.0, atol=1e-6) and np.allclose(T.sum(axis=1), 1.0, atol=1e-6)) or (B < 0).any() or (T < 0).any():
        return _na("not_applicable", "each row of the behaviour and target probabilities is a distribution over the actions: non-negative and summing to one")
    ai = a.astype(int)
    if ai.min() < 0 or ai.max() >= B.shape[1]:
        return _na("not_applicable", "an action index is outside the columns of the probability tables")
    pb_taken = B[np.arange(n), ai]
    if (pb_taken <= 0).any():
        return _na("not_applicable", f"{int((pb_taken <= 0).sum())} logged actions had zero behaviour probability: the logged propensities are not the ones that generated the data", n=n)
    violating = (T > 0) & (B <= 0)
    if violating.any():
        rows = int(violating.any(axis=1).sum())
        return _na("blocked", f"the target policy takes actions the behaviour policy could not take in {rows} of {n} contexts: nothing can be learned about them from this log, and no estimate is made",
                   support_violations={"contexts": rows, "share": rows / n, "actions": sorted(int(x) for x in np.where(violating.any(axis=0))[0])}, n=n)
    w = T[np.arange(n), ai] / pb_taken
    ess = effective_sample_size(w)
    diag = {"n": n, "effective_sample_size": ess, "ess_share": ess / n, "max_weight": float(w.max()), "max_weight_share": float(w.max() / w.sum()) if w.sum() > 0 else None, "mean_weight": float(w.mean()),
            "clipped_share": 0.0}
    if ess < min_ess:
        return _na("blocked", f"the effective sample size is {ess:.1f} of {n} logged decisions (below {min_ess:g}): a few rows would decide the answer, so none is given", diagnostics=diag)
    wc = w
    if clip is not None:
        wc = np.minimum(w, float(clip))
        diag["clipped_share"] = float((w > clip).mean())
        diag["clip"] = float(clip)
    qa = None if q_hat is None else np.asarray(q_hat, dtype=float)
    if qa is not None and qa.shape != B.shape:
        return _na("not_applicable", "q_hat has one row per decision and one column per action")

    def estimates(idx) -> dict:
        ww, rr = wc[idx], r[idx]
        out = {"ips": float(np.mean(ww * rr)), "snips": float((ww * rr).sum() / ww.sum()) if ww.sum() > 0 else float("nan")}
        if qa is not None:
            qi = qa[idx]
            v_hat = (T[idx] * qi).sum(axis=1)
            out["dr"] = float(np.mean(v_hat + ww * (rr - qi[np.arange(len(idx)), ai[idx]])))
        return out

    point = estimates(np.arange(n))
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for _ in range(int(n_boot)):
        if block > 1:
            starts = rng.integers(0, n, size=int(np.ceil(n / block)))
            idx = ((starts[:, None] + np.arange(block)[None, :]) % n).reshape(-1)[:n]
        else:
            idx = rng.integers(0, n, size=n)
        for k, v in estimates(idx).items():
            draws[k].append(v)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    est = {k: {"value": v, "interval": [float(np.nanquantile(draws[k], lo)), float(np.nanquantile(draws[k], hi))], "bootstrap_se": float(np.nanstd(draws[k], ddof=1))} for k, v in point.items()}
    behavior_value = float(r.mean())
    return {"version": VERSION, "eligibility": "applied", "estimates": est, "behavior_value": behavior_value, "diagnostics": diag, "assumptions": ASSUMPTIONS, "level": level, "bootstrap": {"n_boot": int(n_boot), "block": int(block)},
            "notes": ["IPS is unbiased under the assumptions; SNIPS trades a small bias for lower variance; the doubly robust estimate is unbiased if either the weights or the reward model are right",
                      "the interval is a bootstrap over logged decisions: it carries sampling error, not the error of an assumption that fails"] + (["weights above the clip were cut: the estimates are biased toward the behaviour policy by an amount the data cannot show"] if clip is not None else [])}
