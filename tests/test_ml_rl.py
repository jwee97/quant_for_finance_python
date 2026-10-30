"""Phase 5: the Gymnasium environment (timing, constraints, costs, rewards) and the PPO, SAC and recurrent-PPO agents behind the model interface."""

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from gymnasium.utils.env_checker import check_env                                  # noqa: E402

from src.ml import config as mc, datasets as dsm, models as md, training as tr   # noqa: E402
from src.ml.rl.env import PortfolioEnv, forward_returns, project_weights, rollout   # noqa: E402
from tests.ml_helpers import make_bundle                                          # noqa: E402

warnings.filterwarnings("ignore")
SPLIT = {"min_train_days": 300, "refit_every": 400}
TINY = {"total_timesteps": 600, "n_steps": 128, "n_epochs": 2, "batch_size": 32, "hidden": 16, "episode_days": 80}


# ------------------------------------------------------------------------------------------------------------ constraints and timing
def test_the_book_respects_exposure_caps_neutrality_the_universe_and_the_turnover_limit():
    prev = np.zeros(6)
    a = np.array([0.9, -0.2, 0.4, -0.8, 0.1, 0.3])
    w = project_weights(a, prev, gross=1.0, max_weight=0.3)
    assert abs(w.sum()) < 1e-9 and np.abs(w).sum() <= 1.0 + 1e-9 and np.abs(w).max() <= 0.3 + 1e-9
    lo = project_weights(np.append(a, 0.0), prev, gross=0.8, max_weight=0.3, long_only=True)
    assert (lo >= 0).all() and lo.sum() <= 0.8 + 1e-9 and lo.max() <= 0.3 + 1e-9
    off = project_weights(a, prev, gross=1.0, max_weight=0.5, mask=np.array([1, 1, 0, 1, 1, 1], bool))
    assert off[2] == 0.0 and abs(off.sum()) < 1e-9
    held = np.array([0.2, -0.2, 0.0, 0.0, 0.0, 0.0])
    step = project_weights(a, held, gross=1.0, max_weight=0.5, max_turnover=0.3)
    assert np.abs(step - held).sum() == pytest.approx(0.3, abs=1e-9)
    assert np.all(project_weights(np.zeros(6), prev) == 0)


def test_a_book_decided_on_day_t_earns_the_return_of_day_t_plus_lag_plus_one():
    r = np.arange(10, dtype=float).reshape(10, 1)
    f = forward_returns(r, lag=1)
    assert list(f[:8, 0]) == [2, 3, 4, 5, 6, 7, 8, 9] and np.isnan(f[8:, 0]).all()


# ------------------------------------------------------------------------------------------------------------ the environment
def _env(**kw):
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(300, 5, 4)).astype(np.float32)
    fwd = 0.01 * rng.normal(size=(300, 5))
    return PortfolioEnv(feats, fwd, None, **{"episode_days": 50, "seed": 1, **kw})


def test_the_environment_follows_the_gymnasium_api():
    check_env(_env(), skip_render_check=True)
    check_env(_env(long_only=True, reward="dsr"), skip_render_check=True)
    env = _env()
    obs, _ = env.reset(seed=3)
    again, _ = _env().reset(seed=3)
    assert obs.shape == env.observation_space.shape == (5 * 4 + 5 + 1,) and np.allclose(obs, again)
    steps, done = 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        done = terminated or truncated
        assert np.isfinite(reward) and {"net_return", "cost", "turnover", "drawdown"} <= set(info)
    assert steps == 50


def test_costs_are_the_turnover_times_the_rate_and_a_held_book_costs_nothing():
    env = _env(cost_bps=10.0, reward="return", gross=1.0, max_weight=0.5)
    env.reset(seed=0)
    a = np.array([1.0, -1.0, 0.5, -0.5, 0.0], dtype=np.float32)
    _, _, _, _, first = env.step(a)
    assert first["cost"] == pytest.approx(first["turnover"] * 1e-3) and first["turnover"] == pytest.approx(np.abs(env.w).sum())
    _, _, _, _, again = env.step(a)
    assert again["turnover"] == pytest.approx(0.0, abs=1e-9) and again["cost"] == pytest.approx(0.0, abs=1e-12)
    impact = _env(cost_bps=0.0, impact=2.0, reward="return")
    impact.reset(seed=0)
    _, _, _, _, info = impact.step(a)
    assert info["cost"] == pytest.approx(2.0 * float((impact.w ** 2).sum()))


def test_each_reward_is_finite_and_the_penalties_bite():
    base = {}
    for reward in ("return", "risk_adjusted", "dsr"):
        env = _env(reward=reward)
        env.reset(seed=0)
        rs = [env.step(env.action_space.sample())[1] for _ in range(40)]
        base[reward] = np.mean(rs)
        assert np.isfinite(rs).all()
    a = np.array([1.0, -1.0, 1.0, -1.0, 0.0], dtype=np.float32)
    plain, risky = _env(reward="return"), _env(reward="risk_adjusted", risk_aversion=5.0, turnover_penalty=0.5)
    plain.reset(seed=0)
    risky.reset(seed=0)
    assert risky.step(a)[1] < plain.step(a)[1]


def test_a_rollout_is_causal_and_starts_flat():
    feats = np.random.default_rng(0).normal(size=(60, 4, 3)).astype(np.float32)
    seen = []

    def policy(obs, state, first):
        seen.append((first, obs[:12].copy()))
        return np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32), state

    books = rollout(policy, feats, np.ones((60, 4), bool), {"gross": 1.0, "max_weight": 0.5, "cost_bps": 0.0})
    assert seen[0][0] is True and not any(f for f, _ in seen[1:]) and np.allclose(seen[10][1], feats[10].ravel()) and books.shape == (60, 4)
    assert np.allclose(books[0], books[1])                                                       # the same action asked every day gives the same book


# ------------------------------------------------------------------------------------------------------------ the agents
@pytest.fixture(scope="module")
def cfg():
    return mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": SPLIT}, defaults=mc.DEFAULTS)


@pytest.fixture(scope="module")
def ds(cfg):
    return dsm.build_dataset(make_bundle(6, 1000, seed=51, alpha=1.0), cfg["dataset"], "rl")


def _params(kind, **over):
    base = {**TINY, **({"learning_starts": 100, "buffer_size": 2000} if kind == "sac" else {}), **({"lstm_hidden": 8} if kind == "recurrent_ppo" else {})}
    return {k: v for k, v in {**base, **over}.items() if k in md._INFO[kind].params}


def _fit(ds, kind, **over):
    pos = ds.positions()
    rows = np.flatnonzero((pos >= 60) & (pos < 600) & ds.Y[ds.target].notna().to_numpy())
    model = md.make_model(kind, _params(kind, **over), "regression", 0).bind(ds)
    return model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos[rows])


@pytest.mark.parametrize("kind", ["ppo", "sac", "recurrent_ppo"])
def test_an_agent_trains_plays_within_its_constraints_and_survives_serialization(ds, kind):
    if kind == "recurrent_ppo":
        pytest.importorskip("sb3_contrib")
    model = _fit(ds, kind, max_weight=0.4, gross=0.9)
    test = ds.X.iloc[np.flatnonzero((ds.positions() >= 620) & (ds.positions() < 700))]
    w = pd.Series(model.predict(test), index=test.index).unstack("asset")
    assert w.shape[1] == len(ds.assets) and (w.abs().sum(axis=1) <= 0.9 + 1e-9).all() and w.abs().max().max() <= 0.4 + 1e-9 and np.allclose(w.sum(axis=1), 0.0, atol=1e-9)
    diag = model.diagnostics()
    assert diag["algo"] == kind and diag["n_params"] > 0 and diag["train_curve"]["equity"] and diag["constraints"]["max_weight"] == 0.4 and diag["rewards"]["timesteps"] is not None
    assert model.importance().sum() == pytest.approx(1.0, abs=1e-3) or model.importance().sum() == 0.0
    again = md.Model.from_bytes(model.to_bytes()).bind(ds)
    assert np.allclose(again.predict(test), model.predict(test), atol=1e-6)


def test_an_agent_scores_a_day_from_the_rows_up_to_that_day_only(ds):
    model = _fit(ds, "ppo")
    pos = ds.positions()
    rows = ds.X.iloc[np.flatnonzero((pos >= 620) & (pos < 700))]
    before = model.predict(rows)
    cut = ds.dates[660]
    X2 = rows.astype("float64")                                                                          # pandas 3 refuses to put float64 noise into float32 columns
    later = rows.index.get_level_values(0) > cut
    X2.loc[later] = np.random.default_rng(0).normal(size=(int(later.sum()), X2.shape[1]))
    after = model.predict(X2)
    early = ~later
    assert np.allclose(before[early], after[early]) and not np.allclose(before[later], after[later])


def test_a_long_only_agent_never_shorts_and_a_walk_forward_run_collects_its_curves(ds, cfg):
    res = tr.walk_forward_train(ds, {"kind": "ppo", "params": _params("ppo", long_only=1, gross=1.0, max_weight=0.5)}, cfg["split"], 0)
    w = res.scores_wide(ds).dropna(how="all")
    assert (w >= -1e-12).all().all() and (w.sum(axis=1) <= 1.0 + 1e-9).all()
    assert res.diagnostics[0]["rewards"]["mean_reward"] is not None and res.diagnostics[0]["train_curve"]["sharpe"] == pytest.approx(res.diagnostics[0]["train_curve"]["sharpe"])


def test_an_agent_needs_the_dataset_enough_days_and_sane_settings(ds):
    pos = ds.positions()
    rows = np.flatnonzero((pos >= 60) & (pos < 600))
    with pytest.raises(ValueError, match="bound to a dataset"):
        md.make_model("ppo", _params("ppo")).fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows])
    with pytest.raises(ValueError, match="too few"):
        md.make_model("ppo", _params("ppo")).bind(ds).fit(ds.X.iloc[:300], ds.Y[ds.target].iloc[:300])
    with pytest.raises(ValueError, match="reward"):
        md.make_model("ppo", _params("ppo", reward="profit"))
    with pytest.raises(ValueError, match="invalid"):
        md.make_model("ppo", _params("ppo", max_weight=0.0))


def test_the_dataset_object_is_untouched_by_an_agents_training(ds):
    before = ds.X.copy(), ds.returns.copy()
    _fit(ds, "ppo")
    assert before[0].equals(ds.X) and before[1].equals(ds.returns)
    assert dataclasses.is_dataclass(ds)
