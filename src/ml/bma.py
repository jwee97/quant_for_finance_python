"""Bayesian model averaging with an explicit likelihood, prior and approximation (spec v3, Release 8, P3-03). Optional: the base workflow does not use it.

Each member's forecast ``s_m`` is a candidate explanation of the target: the model ``M_m`` is the Gaussian linear regression ``y = a + b s_m + e`` with a flat prior on ``a``, Jeffreys' prior on the
variance of ``e`` and **Zellner's g-prior** on ``b`` with ``g = n`` (the unit-information prior). Under these priors the marginal likelihood is available in closed form (Liang, Paulo, Molina, Clyde and Berger
2008): the Bayes factor of ``M_m`` against the intercept-only model is

    BF = (1 + g)^((n - 1 - 1) / 2) / (1 + g (1 - R^2))^((n - 1) / 2)

with ``R^2`` the squared correlation of ``s_m`` with ``y`` on the fitting sample. **No approximation is made in the weights.** The posterior probability of a model is its prior times its Bayes factor, normalised
over the candidate set (the intercept-only model is a candidate by default, so that "no member explains the target" is an answer the method can give). The averaged prediction is the posterior-probability
weighted mean, and its variance is split into the **within-model** variance (the plug-in posterior mean of ``sigma^2``, an approximation that ignores the uncertainty of ``a`` and ``b``) and the **between-model**
variance (the weighted disagreement of the members).

The repository's ``probability.bayes.bayes_linear_regression`` is a conjugate regression with a fixed, diffuse prior precision on every coefficient: its marginal likelihoods are not comparable across
models of different size and are dominated by the arbitrary prior scale (the Lindley-Bartlett paradox), which is why model probabilities here come from the g-prior, whose scale is tied to the data.

What this is not: weights that follow performance. A softmax of an IC or a Sharpe ratio is a weighting scheme, not a posterior probability, and ``performance_weights`` says so on its result
(``is_posterior = False``). A member whose output is not an expected return (a rank or a probability) has no Gaussian likelihood for the target and is ``not_applicable``. The comparison with the simpler
regularised ensembles is made on the same held-out rows, for the weights fitted before them only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import inference as inf
from .ensemble_models import stack_weights

VERSION = "bma-v1"
MIN_ROWS = 30
LIKELIHOOD = "Gaussian linear model y = a + b*s + e; flat prior on a, Jeffreys prior on sigma^2, Zellner g-prior on b with g = n (unit information)"
APPROXIMATION = "none in the model probabilities (the Bayes factor is exact under these priors); the predictive variance within a model is the plug-in posterior mean of sigma^2 and ignores the uncertainty of a and b"


class BMAError(ValueError):
    """A request the averaging procedure cannot support."""


def log_bayes_factor(r2: float, n: int, g: float | None = None) -> float:
    """Log Bayes factor of a one-predictor g-prior regression against the intercept-only model."""
    g = float(n) if g is None else float(g)
    return float(0.5 * (n - 2) * np.log1p(g) - 0.5 * (n - 1) * np.log1p(g * (1.0 - r2)))


def _fit_member(s: np.ndarray, y: np.ndarray, g: float) -> dict:
    n = len(y)
    sc, yc = s - s.mean(), y - y.mean()
    sxx, tss = float((sc ** 2).sum()), float((yc ** 2).sum())
    if sxx <= 1e-18 or tss <= 1e-18:
        return {"usable": False, "reason": "the member's forecast or the target is constant on the fitting sample"}
    beta_ols = float((sc * yc).sum() / sxx)
    r2 = float(min(max(beta_ols ** 2 * sxx / tss, 0.0), 1.0))
    shrink = g / (1.0 + g)
    beta = shrink * beta_ols                                                                       # posterior mean of b under the g-prior
    alpha = float(y.mean() - beta * s.mean())
    sigma2 = float(tss * (1.0 - shrink * r2) / max(n - 3, 1))                                      # posterior mean of sigma^2
    return {"usable": True, "r2": r2, "alpha": alpha, "beta": float(beta), "sigma2": sigma2, "beta_ols": beta_ols, "log_bf": log_bayes_factor(r2, n, g)}


def fit(S, y, names: list[str] | None = None, *, prior: dict | None = None, include_null: bool = True, output_kinds: list[str] | None = None, g: float | None = None) -> dict:
    """The posterior model probabilities of the members on the fitting rows ``(S, y)``. ``output_kinds`` (one per member) must all be ``expected_return``: anything else has no Gaussian likelihood for the target."""
    S, y = np.asarray(S, dtype=float), np.asarray(y, dtype=float)
    if S.ndim != 2:
        raise BMAError("S is a matrix with one column per member")
    names = list(names) if names is not None else [f"m{j}" for j in range(S.shape[1])]
    if output_kinds is not None:
        bad = [n for n, k in zip(names, output_kinds) if k != "expected_return"]
        if bad:
            return {"eligibility": "not_applicable", "is_posterior": False, "reason": f"{bad} do not output an expected return: a rank or a probability has no Gaussian likelihood for a return target",
                    "likelihood": LIKELIHOOD}
    ok = np.isfinite(y) & np.isfinite(S).all(axis=1)
    n = int(ok.sum())
    if n < MIN_ROWS:
        return {"eligibility": "insufficient", "is_posterior": False, "reason": f"{n} complete rows; at least {MIN_ROWS} are needed to weigh models", "likelihood": LIKELIHOOD, "n": n}
    S, y = S[ok], y[ok]
    gg = float(n) if g is None else float(g)
    fits = [_fit_member(S[:, j], y, gg) for j in range(S.shape[1])]
    usable = [j for j, f in enumerate(fits) if f["usable"]]
    if not usable:
        return {"eligibility": "insufficient", "is_posterior": False, "reason": "no member varies on the fitting sample", "likelihood": LIKELIHOOD, "n": n}
    cand = [(j, names[j]) for j in usable] + ([(-1, "intercept_only")] if include_null else [])
    pri = np.array([1.0 if prior is None else float(prior.get(nm, 1.0)) for _, nm in cand])                    # a mapping name -> weight; a model not named gets 1
    if (pri < 0).any() or pri.sum() <= 0:
        raise BMAError("the prior over models is non-negative and not all zero")
    pri = pri / pri.sum()
    logbf = np.array([fits[j]["log_bf"] if j >= 0 else 0.0 for j, _ in cand])
    logpost = np.log(np.maximum(pri, 1e-300)) + logbf
    post = np.exp(logpost - logpost.max())
    post /= post.sum()
    tss = float(((y - y.mean()) ** 2).sum())
    models = [{"name": nm, "prior": float(p), "log_bayes_factor_vs_null": float(lb), "posterior": float(q), "r2": fits[j]["r2"] if j >= 0 else 0.0,
               "alpha": fits[j]["alpha"] if j >= 0 else float(y.mean()), "beta": fits[j]["beta"] if j >= 0 else 0.0, "sigma2": fits[j]["sigma2"] if j >= 0 else tss / max(n - 3, 1)}
              for (j, nm), p, lb, q in zip(cand, pri, logbf, post)]
    unusable = [{"name": names[j], "reason": fits[j]["reason"]} for j, f in enumerate(fits) if not f["usable"]]
    return {"eligibility": "applied", "version": VERSION, "is_posterior": True, "label": "posterior model probabilities", "likelihood": LIKELIHOOD, "approximation": APPROXIMATION, "g": gg, "n": n,
            "models": models, "member_index": {nm: j for j, nm in cand if j >= 0}, "unusable_members": unusable, "effective_models": float(1.0 / np.sum(post ** 2)),
            "note": "a posterior probability is conditional on the candidate set, the priors and the Gaussian likelihood; with many observations it concentrates on one model, which is what the posterior says, not a calibrated confidence"}


def predict(fitted: dict, S_new) -> dict:
    """The averaged prediction for new rows ``S_new`` (columns as in the fit): mean, within-model variance and between-model variance. Rows with a missing member forecast get no prediction."""
    if fitted.get("eligibility") != "applied":
        raise BMAError("there is no fitted average: " + str(fitted.get("reason")))
    S = np.asarray(S_new, dtype=float)
    idx = fitted["member_index"]
    mus, sig2, w = [], [], []
    for m in fitted["models"]:
        if m["name"] == "intercept_only":
            mus.append(np.full(len(S), m["alpha"]))
        else:
            mus.append(m["alpha"] + m["beta"] * S[:, idx[m["name"]]])
        sig2.append(m["sigma2"])
        w.append(m["posterior"])
    mus, sig2, w = np.vstack(mus), np.array(sig2)[:, None], np.array(w)[:, None]
    mean = (w * mus).sum(axis=0)
    within = (w * sig2).sum(axis=0)
    between = (w * (mus - mean) ** 2).sum(axis=0)
    bad = ~np.isfinite(S[:, [idx[m["name"]] for m in fitted["models"] if m["name"] != "intercept_only"]]).all(axis=1) if idx else np.zeros(len(S), dtype=bool)
    mean, within, between = (np.where(bad, np.nan, v) for v in (mean, within, between))
    return {"mean": mean, "within_variance": within, "between_variance": between, "total_variance": within + between}


def performance_weights(S, y, kind: str = "ic", temperature: float = 1.0) -> dict:
    """Weights that follow a performance statistic (a softmax of each member's correlation with the target). They are a weighting rule. They are **not** posterior probabilities and say so."""
    S, y = np.asarray(S, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(S).all(axis=1)
    if kind != "ic":
        raise BMAError("the performance statistic is 'ic'")
    stat = np.array([np.corrcoef(S[ok, j], y[ok])[0, 1] if np.std(S[ok, j]) > 0 else 0.0 for j in range(S.shape[1])])
    z = temperature * (stat - stat.mean()) / max(float(stat.std()), 1e-9)
    w = np.exp(z - z.max())
    w /= w.sum()
    return {"weights": w, "statistic": stat, "is_posterior": False, "label": "performance weights",
            "note": "a softmax of each member's in-sample correlation with the target: a weighting rule with a temperature, not a probability that a member is the true model"}


def compare(S_fit, y_fit, S_test, y_test, names: list[str], *, horizon: int = 1, shrink: float = 0.5, seed: int = 0) -> dict:
    """The same held-out rows scored for: the equal-weight average of calibrated members, the best single member by fit R^2, regularised stacked weights, performance weights and the Bayesian average. All
    weights come from ``(S_fit, y_fit)`` only. Returns the root mean square error, out-of-sample R^2 (against the fitting mean), the correlation with the target, the mean Gaussian log score and the paired
    squared-error difference against the equal-weight average with its Newey-West interval."""
    S_fit, y_fit, S_test, y_test = (np.asarray(v, dtype=float) for v in (S_fit, y_fit, S_test, y_test))
    okf = np.isfinite(y_fit) & np.isfinite(S_fit).all(axis=1)
    okt = np.isfinite(y_test) & np.isfinite(S_test).all(axis=1)
    if okf.sum() < MIN_ROWS or okt.sum() < MIN_ROWS:
        return {"eligibility": "insufficient", "reason": f"{int(okf.sum())} fitting and {int(okt.sum())} held-out rows; {MIN_ROWS} are needed in each"}
    Sf, yf, St, yt = S_fit[okf], y_fit[okf], S_test[okt], y_test[okt]
    cal = [np.polyfit(Sf[:, j], yf, 1) if np.std(Sf[:, j]) > 0 else np.array([0.0, yf.mean()]) for j in range(Sf.shape[1])]
    F = np.column_stack([cal[j][1] + cal[j][0] * Sf[:, j] for j in range(Sf.shape[1])])               # calibrated member forecasts on the fitting rows
    T = np.column_stack([cal[j][1] + cal[j][0] * St[:, j] for j in range(St.shape[1])])
    resid_var = (yf[:, None] - F).var(axis=0, ddof=2)
    preds, var = {}, {}
    m = F.shape[1]
    preds["equal"], var["equal"] = T.mean(axis=1), np.full(len(yt), float(resid_var.mean()))
    r2 = [np.corrcoef(Sf[:, j], yf)[0, 1] ** 2 if np.std(Sf[:, j]) > 0 else 0.0 for j in range(m)]
    best = int(np.argmax(r2))
    preds["best_single"], var["best_single"] = T[:, best], np.full(len(yt), float(resid_var[best]))
    ws, _ = stack_weights(F, yf, shrink)
    preds["stacked"], var["stacked"] = T @ ws, np.full(len(yt), float(((yf - F @ ws) ** 2).sum() / max(len(yf) - 2, 1)))
    pw = performance_weights(Sf, yf)
    preds["performance_weights"], var["performance_weights"] = T @ pw["weights"], np.full(len(yt), float(((yf - F @ pw["weights"]) ** 2).sum() / max(len(yf) - 2, 1)))
    fb = fit(Sf, yf, names)
    if fb["eligibility"] == "applied":
        pb = predict(fb, St)
        preds["bma"], var["bma"] = pb["mean"], pb["total_variance"]
    rows, base_loss = [], (yt - preds["equal"]) ** 2
    for k, p in preds.items():
        loss = (yt - p) ** 2
        ls = float(np.mean(-0.5 * np.log(2 * np.pi * var[k]) - (yt - p) ** 2 / (2 * var[k])))
        d = inf.hac_mean(pd.Series(loss - base_loss), horizon=int(horizon), label=f"squared error of {k} minus equal weights") if k != "equal" else None
        rows.append({"method": k, "rmse": float(np.sqrt(loss.mean())), "r2_oos": float(1.0 - loss.sum() / ((yt - yf.mean()) ** 2).sum()), "correlation": float(np.corrcoef(p, yt)[0, 1]) if np.std(p) > 0 else None,
                     "log_score": ls, "squared_error_minus_equal": None if d is None else d.as_dict(), "is_posterior": k == "bma"})
    return {"eligibility": "applied", "version": VERSION, "n_fit": int(okf.sum()), "n_test": int(okt.sum()), "rows": rows, "bma": fb if fb["eligibility"] == "applied" else {"eligibility": fb["eligibility"], "reason": fb.get("reason")},
            "weights": {"stacked": dict(zip(names, map(float, ws))), "performance": dict(zip(names, map(float, pw["weights"]))),
                        "bma": {mm["name"]: mm["posterior"] for mm in fb["models"]} if fb["eligibility"] == "applied" else None},
            "best_single_member": names[best], "note": "every method is scored on the same held-out rows and every weight comes from the fitting rows only; a negative squared-error difference means the method beat the equal-weight average"}
