"""Dynamic programming, optimal control and reinforcement learning on problems with a known answer: how close do the numerical methods get, and what do the learning methods cost in data?

Everything here is either exact mathematics checked against a closed form or a simulated world, so the numbers say how well each method does its job, not whether it makes money. The one real-data run (the
exposure allocator on the platform's 15 ETFs) is reported as it came out.

    python -m experiments.control_world          # about six minutes
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from src.control import hjb, merton, mdp as md, pontryagin as pmp, schrodinger as sb
from src.derivatives.pricing import binomial_price
from src.rl import deep, envs, glearning as gl, tabular as tb


# ------------------------------------------------------------------------------------------------------------------ 1. MDP and POMDP
def tiger():
    P = np.array([np.eye(2), np.full((2, 2), .5), np.full((2, 2), .5)])
    O = np.array([[[.85, .15], [.15, .85]], [[.5, .5]] * 2, [[.5, .5]] * 2])
    R = np.array([[-1, -100, 10], [-1, 10, -100]], float)
    return md.POMDP(P, O, R, 0.95)


def pomdp_table() -> pd.DataFrame:
    pm = tiger()
    b = np.array([.5, .5])
    grid = np.array([[x, 1 - x] for x in np.linspace(0, 1, 41)])
    rows = {}
    for h in (1, 2, 3, 4, 5):
        rows[f"exact tree search, {h} steps"] = {"value at 50/50": md.expectimax(pm, b, h)}
    for it in (10, 60, 200):
        v, a = md.alpha_value(md.pbvi(pm, grid, it), b)
        rows[f"point-based value iteration, {it} sweeps"] = {"value at 50/50": v, "action": ["listen", "open left", "open right"][a]}
    sure = md.alpha_value(md.pbvi(pm, grid, 200), np.array([.97, .03]))
    rows["... when 97% sure the tiger is on the left"] = {"value at 50/50": sure[0], "action": ["listen", "open left", "open right"][sure[1]]}
    return pd.DataFrame(rows).T


def risk_sensitive_table() -> pd.DataFrame:
    P = np.zeros((2, 4, 4))
    for a in range(2):
        for s in (1, 2, 3):
            P[a, s, s] = 1.0
    P[0, 0, 3] = 1.0
    P[1, 0, 1] = P[1, 0, 2] = 0.5
    R = np.array([[0.0, 0.0], [3.2, 3.2], [-1.0, -1.0], [1.0, 1.0]])
    mdp = md.MDP(P, R, 0.5)
    rows = {}
    for theta in (-2.0, -0.5, 0.0, 0.5, 2.0):
        V, pi = md.risk_sensitive_value_iteration(mdp, theta)
        rows[f"theta = {theta:+.1f}"] = {"value of the decision": V[0], "choice": "gamble (mean 1.1)" if pi[0] == 1 else "sure thing (1.0)"}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------------------ 2. Merton, HJB
def merton_table() -> pd.DataFrame:
    rows = {}
    for gamma in (3.0, 0.5):
        for nx, nt in ((101, 25), (201, 50), (401, 100)):
            res = hjb.merton_hjb(0.08, 0.02, 0.2, gamma, 1.0, nx, nt)
            rows[(f"gamma = {gamma}", f"{nx} x {nt} grid")] = {"max relative error of V": res.max_relative_error, "risky fraction": res.fraction[nx // 2],
                                                               "Merton's": min(merton.merton_fraction(0.08, 0.02, 0.2, gamma), 3.0)}
    return pd.DataFrame(rows).T


def cost_band_table() -> pd.DataFrame:
    rows = {}
    for kappa in (0.0, 0.0005, 0.002, 0.008, 0.032):
        sol = merton.CostDP(0.08, 0.02, 0.2, 3.0, kappa, periods=240, grid=301).solve()
        rows[f"cost {10000 * kappa:.0f} bp"] = {"no-trade region lower": sol.lower, "upper": sol.upper, "half-width": sol.half_width,
                                                  "small-cost formula": merton.davis_norman_half_width(0.08, 0.02, 0.2, 3.0, kappa)}
    return pd.DataFrame(rows).T


def obstacle_table() -> pd.DataFrame:
    ref = binomial_price(100, 100, 1.0, 0.05, 0.0, 0.3, call=False, american=True, steps=3000)
    rows = {}
    for n in (100, 200, 400, 800):
        price, sol = hjb.american_put_hjb(100, 100, 1.0, 0.05, 0.0, 0.3, nx=n, nt=n)
        rows[f"{n} x {n} grid"] = {"price": price, "error against a 3000-step tree": price - ref, "exercise boundary at t=0": sol.exercise_boundary}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------------------ 3. Pontryagin
def pontryagin_table() -> pd.DataFrame:
    rows = {}
    s = pmp.lqr_pmp(-0.5, 1.0, 1.0, 0.5, 2.0, 1.0, 5.0)
    _, x, u = pmp.lqr_riccati(-0.5, 1.0, 1.0, 0.5, 2.0, 1.0, 5.0)
    rows["linear-quadratic regulator vs Riccati"] = {"max |difference| in the state": np.abs(s.x[0] - x).max(), "max |difference| in the control": np.abs(s.u[0] - u).max()}
    a = pmp.liquidation_pmp(1e5, 1.0, 0.01, 5.0)
    rows["liquidation vs sinh schedule (shares)"] = {"max |difference| in the state": np.abs(a.x[0] - pmp.almgren_chriss_closed_form(1e5, 1.0, 0.01, 5.0, a.t)).max()}
    k, c = pmp.ramsey_steady_state()
    r = pmp.ramsey_pmp(1.0, 80.0)
    rows["Ramsey growth: k(T) and c(T) vs the steady state"] = {"max |difference| in the state": abs(r.x[0, -1] - k), "max |difference| in the control": abs(r.u[0, -1] - c)}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------------------ 4. Schroedinger
def bridge_table(seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        S = 8
        Q = 0.95 * np.eye(S) + 0.05 * rng.random((S, S))                                    # sticky: the chain remembers where it started, so the endpoints matter
        Q /= Q.sum(axis=1, keepdims=True)
        p0, pT = rng.random(S), rng.random(S)
        p0, pT = p0 / p0.sum(), pT / pT.sum()
        br = sb.markov_bridge(Q, p0, pT, 6)
        K = np.linalg.matrix_power(Q, 6)
        ref = p0[:, None] * K
        indep = p0[:, None] * pT[None, :]
        rows.append({"bridge KL": sb.path_kl(br.transitions, Q, p0), "independent coupling KL": float((indep * np.log(indep / ref)).sum()), "Sinkhorn sweeps": br.iterations})
    return pd.DataFrame(rows).mean().to_frame("mean over 5 random chains").T


# ------------------------------------------------------------------------------------------------------------------ 5. reinforcement learning
def td_table() -> pd.DataFrame:
    env = envs.random_mdp_env(4, 2, 0.7, seed=3)
    mdp = env.to_mdp()
    V, pi, _ = md.value_iteration(mdp)
    rows = {}
    for episodes in (25, 100, 400, 1600):
        for method in tb.METHODS:
            res = tb.train(env, method, episodes=episodes, max_steps=100, gamma=0.7, alpha=0.5, omega=0.6, epsilon=0.1, seed=0)
            tgt = mdp.q_values(V)
            if method in ("sarsa", "expected_sarsa"):
                pol = np.full((4, 2), 0.05)
                pol[np.arange(4), tgt.argmax(1)] += 0.9
                Vp = np.linalg.solve(np.eye(4) - 0.7 * np.einsum("sa,ast->st", pol, mdp.P), (pol * mdp.R).sum(1))
                tgt = mdp.R + 0.7 * np.einsum("ast,t->sa", mdp.P, Vp)
            rows[(method, episodes)] = {"max |Q - target|": np.abs(res.Q - tgt).max(), "greedy policy optimal": float((res.policy == pi).all())}
    return pd.DataFrame(rows).T


def cliff_table() -> pd.DataFrame:
    cw = envs.CliffWalking()
    rows = {}
    for method in ("sarsa", "q_learning"):
        res = [tb.train(cw, method, episodes=1000, gamma=1.0, alpha=0.5, omega=0.5, epsilon=0.1, seed=s) for s in range(5)]
        rows[method] = {"mean return while exploring (last 100)": np.mean([np.mean(r.returns[-100:]) for r in res]),
                        "return of the greedy policy": np.mean([tb.greedy_return(cw, r.policy) for r in res])}
    return pd.DataFrame(rows).T


def g_learning_table() -> pd.DataFrame:
    env = envs.random_mdp_env(4, 3, 0.8, seed=2)
    mdp = env.to_mdp()
    V, pi, _ = md.value_iteration(mdp)
    rows = {}
    for beta in (0.1, 1.0, 5.0, 50.0):
        sol = gl.soft_value_iteration(mdp, None, beta)
        rows[f"beta = {beta:g}"] = {"soft value minus optimal value (mean)": float((sol.F - V).mean()), "largest action probability (mean)": float(sol.policy.max(axis=1).mean()),
                                    "agrees with the greedy policy": float((sol.policy.argmax(1) == pi).mean())}
    return pd.DataFrame(rows).T


def g_learning_sample_table() -> pd.DataFrame:
    env = envs.random_mdp_env(4, 3, 0.8, seed=2)
    exact = gl.soft_value_iteration(env.to_mdp(), None, 1.0)
    rows = {}
    for episodes in (50, 150, 300, 1000):
        learned = gl.g_learning(env, None, 1.0, episodes=episodes, gamma=0.8, max_steps=100, omega=0.6)
        rows[f"{episodes} episodes"] = {"max |F - F exact|": float(np.abs(learned.F - exact.F).max()), "max |policy - policy exact|": float(np.abs(learned.policy - exact.policy).max())}
    return pd.DataFrame(rows).T


def girl_table(seeds=(0, 1, 2)) -> pd.DataFrame:
    rows = {}
    for episodes in (10, 30, 100, 300):
        out = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            S, A, K = 6, 3, 4
            P = rng.random((A, S, S)) ** 2
            P /= P.sum(axis=2, keepdims=True)
            X = rng.normal(size=(S, A, K))
            theta = np.array([1.0, -0.5, 0.8, 0.3])
            mdp = md.MDP(P, X @ theta, 0.9)
            policy = gl.soft_value_iteration(mdp, None, 2.0).policy
            env = envs.MDPEnv(mdp, seed=seed)
            trajs = []
            for _ in range(episodes):
                s = env.reset()
                tr = []
                for _ in range(30):
                    a = int(rng.choice(A, p=policy[s]))
                    tr.append((s, a))
                    s, _, _ = env.step(a)
                trajs.append(tr)
            res = gl.girl(P, X, gl.demonstration_counts(trajs, S, A), 0.9, 2.0)
            out.append({"correlation with the planted reward weights": float(np.corrcoef(res.theta, theta)[0, 1]), "policy KL (learned vs true)": gl.mean_policy_kl(policy, res.policy),
                        "policy KL of 'no information'": gl.mean_policy_kl(policy, np.full((S, A), 1 / A))})
        rows[f"{episodes} demonstrations x 30 steps"] = pd.DataFrame(out).mean()
    return pd.DataFrame(rows).T


def deep_table(seeds=(0, 1, 2)) -> pd.DataFrame:
    env = envs.RegimeFractionEnv(horizon=50)
    opt = np.array([env.optimal_fraction(0), env.optimal_fraction(1)])
    rows = {}
    for name, steps in (("dqn", 6000), ("double_dqn", 6000), ("ddpg", 5000), ("td3", 5000), ("sac", 4000), ("ppo", 16000)):
        fr, reg, t0 = [], [], time.time()
        for seed in seeds:
            table = deep.AGENTS[name]().train(env, steps, seed=seed).policy_table(env)
            fr.append(table)
            reg.append(np.mean([env.expected_reward(i, opt[i]) - env.expected_reward(i, table[i]) for i in range(2)]))
        mean = np.mean(fr, axis=0)
        rows[name] = {"fraction, regime 1 (optimum %.2f)" % opt[0]: mean[0], "fraction, regime 2 (optimum %.2f)" % opt[1]: mean[1], "regret (1e-4 log growth / period)": 1e4 * np.mean(reg),
                      "worst seed regret": 1e4 * np.max(reg), "seconds per run": (time.time() - t0) / len(seeds)}
    return pd.DataFrame(rows).T


def exposure_table() -> pd.DataFrame:
    from src.framework import ALLOCATORS, load_library
    from src.framework.allocation import Context
    from src.framework.data import load_default_bundle
    from src.utils.config import load_config

    load_library()
    cfg = load_config()
    bundle = load_default_bundle(cfg)
    r = bundle.returns.fillna(0.0)

    def stats(w):
        x = (w.shift(1).fillna(0.0) * r).sum(axis=1)
        x = x[x.index >= "2010-01-01"]
        eq = (1 + x).cumprod()
        return {"return": 252 * x.mean(), "volatility": x.std() * np.sqrt(252), "Sharpe": x.mean() / x.std() * np.sqrt(252), "max drawdown": float((eq / eq.cummax()).min() - 1)}

    base = bundle.investable.astype(float)
    base = base.div(base.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    rows = {"equal weight (fully invested)": stats(base)}
    for cost in (10.0, 0.0):
        w = ALLOCATORS.create("q_learning_exposure", cost_bps=cost).build(Context(bundle, cfg))
        rows[f"Q-learning exposure, cost {cost:g} bp"] = {**stats(w), "mean exposure": float(w.sum(axis=1)[w.index >= "2010-01-01"].mean())}
    return pd.DataFrame(rows).T


def run() -> dict:
    warnings.simplefilter("ignore")
    return {"pomdp": pomdp_table(), "risk_sensitive": risk_sensitive_table(), "merton": merton_table(), "cost_band": cost_band_table(), "obstacle": obstacle_table(),
            "pontryagin": pontryagin_table(), "bridge": bridge_table(), "td": td_table(), "cliff": cliff_table(), "g_learning": g_learning_table(), "g_learning_sample": g_learning_sample_table(), "girl": girl_table(), "deep": deep_table(),
            "exposure": exposure_table()}


def main() -> None:
    pd.options.display.float_format = "{:.4g}".format
    pd.options.display.width = 220
    t0 = time.time()
    for name, table in run().items():
        print(f"== {name} ==")
        print(table.to_string(), "\n")
    print(f"{time.time() - t0:.0f} seconds")


if __name__ == "__main__":
    main()
