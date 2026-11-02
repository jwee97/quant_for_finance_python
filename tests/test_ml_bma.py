"""Release 8, P3-03: Bayesian model averaging with an explicit likelihood, prior and approximation.

The Bayes factor is checked against a numerical double integral over the coefficient and the variance (a reference that shares no algebra with the closed form); the posterior probabilities against cases
whose answer is known (identical members, informative and noise members, no informative member); the uncertainty split against its definition; and the comparison with the simpler ensembles against the
requirement that every method sees the same held-out rows and that no weight reads them."""

import numpy as np
import pytest
from scipy import integrate

from src.ml import bma

rng0 = np.random.default_rng(0)


def data(n=400, signal=(1.0, 0.0, 0.0), noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, len(signal)))
    y = S @ np.array(signal) + rng.normal(0, noise, n)
    return S, y


# ------------------------------------------------------------------------------------------------------------------ the marginal likelihood
def test_the_bayes_factor_matches_a_numerical_integral_over_the_coefficient_and_the_variance():
    rng = np.random.default_rng(1)
    n = 12
    s = rng.normal(size=n)
    y = 0.6 * s + rng.normal(0, 1, n)
    sc, yc = s - s.mean(), y - y.mean()
    sxx = float((sc ** 2).sum())
    r2 = float((sc @ yc) ** 2 / (sxx * (yc @ yc)))
    g = float(n)

    def lik(beta, sig2):                                                                              # the centred data, a ~ flat: N(yc; beta * sc, sig2 I_{n-1})-type kernel
        resid = yc - beta * sc
        return (2 * np.pi * sig2) ** (-(n - 1) / 2) * np.exp(-0.5 * (resid @ resid) / sig2)

    def alt(sig2):
        inner = integrate.quad(lambda b: lik(b, sig2) * np.exp(-0.5 * b * b * sxx / (g * sig2)) / np.sqrt(2 * np.pi * g * sig2 / sxx), -np.inf, np.inf, epsabs=1e-14, epsrel=1e-10)[0]
        return inner / sig2                                                                           # Jeffreys prior on sigma^2

    num = integrate.quad(lambda ls: alt(np.exp(ls)) * np.exp(ls), -6, 6, epsabs=1e-14, epsrel=1e-9, limit=200)[0]                            # integrate over log sigma^2
    den = integrate.quad(lambda ls: lik(0.0, np.exp(ls)) / np.exp(ls) * np.exp(ls), -6, 6, epsabs=1e-14, epsrel=1e-9, limit=200)[0]
    assert np.log(num / den) == pytest.approx(bma.log_bayes_factor(r2, n, g), rel=2e-3, abs=2e-3)
    assert bma.log_bayes_factor(0.0, 100) < 0 < bma.log_bayes_factor(0.5, 100), "a member that explains nothing is penalised for its extra parameter; one that explains something is rewarded"


# ------------------------------------------------------------------------------------------------------------------ posterior probabilities
def test_posterior_probabilities_for_cases_whose_answer_is_known():
    S, y = data()
    f = bma.fit(S, y, ["good", "noise1", "noise2"])
    p = {m["name"]: m["posterior"] for m in f["models"]}
    assert f["eligibility"] == "applied" and f["is_posterior"] is True and p["good"] > 0.99 and sum(p.values()) == pytest.approx(1.0) and "Zellner g-prior" in f["likelihood"] and f["approximation"].startswith("none in the model")
    twin = bma.fit(np.column_stack([S[:, 0], S[:, 0]]), y, ["a", "b"], include_null=False)
    assert [m["posterior"] for m in twin["models"]] == pytest.approx([0.5, 0.5]), "identical members are equally probable"
    tilted = bma.fit(np.column_stack([S[:, 0], S[:, 0]]), y, ["a", "b"], include_null=False, prior={"a": 2.0, "b": 1.0})
    assert [m["posterior"] for m in tilted["models"]] == pytest.approx([2 / 3, 1 / 3]), "with equal evidence the posterior is the prior"
    Sn, yn = data(signal=(0.0, 0.0, 0.0), seed=4)
    none = bma.fit(Sn, yn, ["a", "b", "c"])
    pn = {m["name"]: m["posterior"] for m in none["models"]}
    assert pn["intercept_only"] > 0.5 and pn["intercept_only"] > sum(v for k, v in pn.items() if k != "intercept_only"), (
        "when no member explains the target the intercept-only model carries more probability than all the members together: 'none of them' is an answer the method can give")
    assert bma.fit(Sn, yn, ["a", "b", "c"], include_null=False)["models"][0]["posterior"] < 0.9, "without the null in the set the probabilities spread over the members"
    with pytest.raises(bma.BMAError, match="non-negative"):
        bma.fit(S, y, ["a", "b", "c"], prior={"a": -1.0})


def test_members_that_have_no_likelihood_or_no_variation_are_refused_or_excluded_with_a_reason():
    S, y = data()
    out = bma.fit(S, y, ["a", "b", "c"], output_kinds=["expected_return", "rank", "probability"])
    assert out["eligibility"] == "not_applicable" and out["is_posterior"] is False and "rank" not in out["reason"].split("[")[0] and "'b', 'c'" in out["reason"] and "no Gaussian likelihood" in out["reason"]
    const = S.copy()
    const[:, 2] = 3.0
    f = bma.fit(const, y, ["a", "b", "c"])
    assert f["unusable_members"] == [{"name": "c", "reason": "the member's forecast or the target is constant on the fitting sample"}] and "c" not in {m["name"] for m in f["models"]}
    short = bma.fit(S[:10], y[:10], ["a", "b", "c"])
    assert short["eligibility"] == "insufficient" and "at least 30" in short["reason"]
    allc = bma.fit(np.full_like(S, 1.0), y, ["a", "b", "c"])
    assert allc["eligibility"] == "insufficient" and "no member varies" in allc["reason"]


# ------------------------------------------------------------------------------------------------------------------ the averaged prediction and its uncertainty
def test_the_variance_of_the_average_is_the_weighted_within_model_variance_plus_the_weighted_disagreement():
    S, y = data(signal=(1.0, 0.8, 0.0), seed=2)
    f = bma.fit(S, y, ["a", "b", "c"], include_null=False)
    new = np.random.default_rng(9).normal(size=(50, 3))
    out = bma.predict(f, new)
    w = np.array([m["posterior"] for m in f["models"]])
    mus = np.vstack([m["alpha"] + m["beta"] * new[:, j] for j, m in enumerate(f["models"])])
    assert out["mean"] == pytest.approx(w @ mus) and out["within_variance"] == pytest.approx(np.full(50, w @ np.array([m["sigma2"] for m in f["models"]])))
    assert out["between_variance"] == pytest.approx((w[:, None] * (mus - (w @ mus)) ** 2).sum(axis=0)) and np.all(out["total_variance"] == out["within_variance"] + out["between_variance"])
    twin = bma.fit(np.column_stack([S[:, 0], S[:, 0]]), y, ["a", "b"], include_null=False)
    assert np.allclose(bma.predict(twin, np.column_stack([new[:, 0], new[:, 0]]))["between_variance"], 0.0), "identical members cannot disagree"
    holes = new.copy()
    holes[3, 1] = np.nan
    assert np.isnan(bma.predict(f, holes)["mean"][3]) and np.isfinite(bma.predict(f, holes)["mean"][4]), "a missing member forecast gives no prediction for that row"
    with pytest.raises(bma.BMAError, match="no fitted average"):
        bma.predict({"eligibility": "insufficient", "reason": "x"}, new)


def test_performance_weights_are_a_rule_not_probabilities_and_say_so():
    S, y = data(signal=(1.0, 0.5, 0.0), seed=3)
    pw = bma.performance_weights(S, y)
    assert pw["is_posterior"] is False and pw["label"] == "performance weights" and "not a probability" in pw["note"] and pw["weights"].sum() == pytest.approx(1.0) and pw["weights"][0] > pw["weights"][2]
    f = bma.fit(S, y, ["a", "b", "c"])
    assert f["is_posterior"] is True and f["label"] == "posterior model probabilities"
    assert not np.allclose(pw["weights"], [m["posterior"] for m in f["models"] if m["name"] != "intercept_only"], atol=0.05), "the two are different objects with different values"
    with pytest.raises(bma.BMAError):
        bma.performance_weights(S, y, kind="sharpe")


# ------------------------------------------------------------------------------------------------------------------ the comparison on identical held-out rows
def test_every_method_is_scored_on_the_same_held_out_rows_and_no_weight_reads_them():
    S, y = data(n=900, signal=(1.0, 0.0, 0.0), noise=1.5, seed=5)
    names = ["good", "noise1", "noise2"]
    out = bma.compare(S[:500], y[:500], S[500:], y[500:], names)
    assert out["eligibility"] == "applied" and out["n_fit"] == 500 and out["n_test"] == 400
    by = {r["method"]: r for r in out["rows"]}
    assert set(by) == {"equal", "best_single", "stacked", "performance_weights", "bma"} and by["bma"]["is_posterior"] is True and not any(by[k]["is_posterior"] for k in by if k != "bma")
    assert out["best_single_member"] == "good" and by["best_single"]["rmse"] < by["equal"]["rmse"] and by["bma"]["rmse"] < by["equal"]["rmse"] and by["stacked"]["rmse"] < by["equal"]["rmse"]
    assert by["bma"]["squared_error_minus_equal"]["value"] < 0 and by["equal"]["squared_error_minus_equal"] is None and by["bma"]["squared_error_minus_equal"]["settings"]["lags"] >= 0
    assert out["weights"]["bma"]["good"] > 0.99 and out["weights"]["stacked"]["good"] > out["weights"]["stacked"]["noise1"]
    y_test2 = y[500:].copy()
    y_test2[:200] = 1e3                                                                                # a different held-out outcome cannot change a weight fitted before it
    other = bma.compare(S[:500], y[:500], S[500:], y_test2, names)
    assert other["weights"] == out["weights"], "weights come from the fitting rows only"
    assert bma.compare(S[:10], y[:10], S[500:], y[500:], names)["eligibility"] == "insufficient"
    assert all(r["log_score"] is not None and np.isfinite(r["log_score"]) for r in out["rows"])
