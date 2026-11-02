"""Release 8, P3-08: off-policy evaluation only with logged behaviour, support and enough effective sample.

The estimators are worked by hand on four decisions; on a simulated bandit whose target-policy value is known, the estimates and their intervals are checked against it; and every condition that makes an
estimate unsupportable returns a status and a reason instead of a number."""

import json

import numpy as np
import pytest

from src.ml import offpolicy as op

A = 3


def bandit(n=3000, seed=0, temperature=1.0):
    """Contexts x ~ N(0, 1); the reward of action a is mu_a(x) + noise; the behaviour policy is a softmax of noisy scores; the target takes the action with the largest mu."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mu = np.column_stack([0.5 * x, -0.5 * x, np.zeros(n)])
    logits = temperature * np.column_stack([0.2 * x, -0.2 * x, np.zeros(n)])
    B = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    a = np.array([rng.choice(A, p=p) for p in B])
    r = mu[np.arange(n), a] + rng.normal(0, 1.0, n)
    T = np.zeros((n, A))
    T[np.arange(n), mu.argmax(axis=1)] = 1.0
    true_value = float(np.maximum(0.5 * x, np.maximum(-0.5 * x, 0)).mean())                          # E[max_a mu_a(x)] on these contexts
    return {"x": x, "mu": mu, "B": B, "a": a, "r": r, "T": T, "truth": true_value}


# ------------------------------------------------------------------------------------------------------------------ by hand
def test_ips_snips_and_the_effective_sample_size_by_hand_on_four_decisions():
    B = np.array([[0.5, 0.5], [0.25, 0.75], [0.5, 0.5], [0.1, 0.9]])
    T = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    a = np.array([0, 0, 1, 1])
    r = np.array([2.0, 1.0, 3.0, -1.0])
    w = np.array([1.0 / 0.5, 1.0 / 0.25, 1.0 / 0.5, 0.5 / 0.9])
    assert op.effective_sample_size(w) == pytest.approx(w.sum() ** 2 / (w ** 2).sum())
    out = op.evaluate(a, r, B, T, min_ess=1.0, n_boot=50)
    assert out["eligibility"] == "applied"
    assert out["estimates"]["ips"]["value"] == pytest.approx(np.mean(w * r)) and out["estimates"]["snips"]["value"] == pytest.approx((w * r).sum() / w.sum())
    assert out["diagnostics"]["effective_sample_size"] == pytest.approx(op.effective_sample_size(w)) and out["diagnostics"]["max_weight"] == pytest.approx(4.0) and out["behavior_value"] == pytest.approx(1.25)
    q = np.array([[1.5, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    dr = op.evaluate(a, r, B, T, q_hat=q, min_ess=1.0, n_boot=50)["estimates"]["dr"]["value"]
    v_hat = (T * q).sum(axis=1)
    taken = q[np.arange(4), a]
    assert dr == pytest.approx(np.mean(v_hat + w * (r - taken)))


# ------------------------------------------------------------------------------------------------------------------ against a known value
def test_the_estimates_recover_the_known_value_of_the_target_policy_and_their_intervals_cover_it():
    d = bandit(n=4000, seed=1)
    out = op.evaluate(d["a"], d["r"], d["B"], d["T"], q_hat=d["mu"], min_ess=100, n_boot=400, seed=2)
    assert out["eligibility"] == "applied" and out["diagnostics"]["ess_share"] > 0.3
    for k in ("ips", "snips", "dr"):
        e = out["estimates"][k]
        assert e["interval"][0] <= d["truth"] <= e["interval"][1], (k, e, d["truth"])
        assert abs(e["value"] - d["truth"]) < 0.08
    assert out["estimates"]["dr"]["bootstrap_se"] < out["estimates"]["ips"]["bootstrap_se"], "a good reward model cuts the variance of the importance-weighted estimate"
    assert out["behavior_value"] < d["truth"] - 0.05, "the target policy is better than the one that logged the data"
    json.dumps(out, allow_nan=False)


def test_ips_is_unbiased_over_repeated_logs_and_a_ninety_five_percent_interval_covers_about_that_often():
    est, cover = [], 0
    reps = 60
    for s in range(reps):
        d = bandit(n=1500, seed=100 + s)
        out = op.evaluate(d["a"], d["r"], d["B"], d["T"], min_ess=50, n_boot=150, seed=s)
        e = out["estimates"]["ips"]
        est.append(e["value"] - d["truth"])
        cover += int(e["interval"][0] <= d["truth"] <= e["interval"][1])
    assert abs(np.mean(est)) < 0.03 and cover / reps >= 0.85, (np.mean(est), cover / reps)


# ------------------------------------------------------------------------------------------------------------------ what cannot be supported
def test_missing_or_unusable_logged_behaviour_is_not_applicable():
    d = bandit(n=500, seed=2)
    assert op.evaluate(d["a"], d["r"], None, d["T"])["eligibility"] == "not_applicable" and "lack the behaviour policy" in op.evaluate(d["a"], d["r"], None, d["T"])["reason"]
    bad = d["B"].copy()
    bad[0] = [0.5, 0.2, 0.1]
    assert "summing to one" in op.evaluate(d["a"], d["r"], bad, d["T"])["reason"]
    nan = d["B"].copy()
    nan[3, 1] = np.nan
    assert "missing values" in op.evaluate(d["a"], d["r"], nan, d["T"])["reason"]
    zero_taken = d["B"].copy()
    i = 7
    zero_taken[i] = [0.0, 0.5, 0.5]
    a = d["a"].copy()
    a[i] = 0
    z = op.evaluate(a, d["r"], zero_taken, d["T"])
    assert z["eligibility"] == "not_applicable" and "zero behaviour probability" in z["reason"] and z["estimates"] is None
    assert "one row per logged decision" in op.evaluate(d["a"][:10], d["r"], d["B"], d["T"])["reason"]
    assert "outside the columns" in op.evaluate(d["a"] + 5, d["r"], d["B"], d["T"])["reason"]


def test_a_target_policy_outside_the_logged_support_is_blocked_and_no_estimate_is_made():
    d = bandit(n=800, seed=3)
    B = d["B"].copy()
    B[:, 2] = 0.0                                                                                    # the behaviour policy never could take action 2
    B = B / B.sum(axis=1, keepdims=True)
    a = np.where(d["a"] == 2, 0, d["a"])
    T = d["T"].copy()
    T[:100] = [0.0, 0.0, 1.0]                                                                        # the target takes it in 100 contexts
    out = op.evaluate(a, d["r"], B, T, min_ess=10)
    assert out["eligibility"] == "blocked" and out["estimates"] is None and out["support_violations"]["contexts"] == 100 and out["support_violations"]["actions"] == [2] and "could not take" in out["reason"]
    assert out["support_violations"]["share"] == pytest.approx(100 / 800)


def test_too_small_an_effective_sample_is_blocked_and_clipping_is_disclosed():
    d = bandit(n=2000, seed=4, temperature=-8.0)                                                      # a sharp behaviour policy that prefers what the target does not: the few rows where they agree carry huge weights
    out = op.evaluate(d["a"], d["r"], d["B"], d["T"], min_ess=500, n_boot=50)
    assert out["eligibility"] == "blocked" and out["estimates"] is None and "effective sample size" in out["reason"] and out["diagnostics"]["ess_share"] < 0.25
    soft = bandit(n=3000, seed=5)
    clipped = op.evaluate(soft["a"], soft["r"], soft["B"], soft["T"], min_ess=50, clip=2.0, n_boot=100)
    assert clipped["eligibility"] == "applied" and clipped["diagnostics"]["clipped_share"] > 0 and clipped["diagnostics"]["clip"] == 2.0 and any("biased toward the behaviour policy" in n for n in clipped["notes"])
    plain = op.evaluate(soft["a"], soft["r"], soft["B"], soft["T"], min_ess=50, n_boot=100)
    assert plain["diagnostics"]["clipped_share"] == 0.0 and not any("clip" in n for n in plain["notes"])
    assert len(plain["assumptions"]) == 4 and all(isinstance(x, str) for x in plain["assumptions"]) and plain["assumptions"][0].startswith("positivity")
