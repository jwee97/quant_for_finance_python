"""PPO, SAC and recurrent PPO (Stable-Baselines3, sb3-contrib) as models of the zoo.

The agent is fitted inside the walk-forward like every other model: on a refit date it trains in ``PortfolioEnv`` on the training window's rows only (the labels are not used; the reward is the realised
net return, and the purge of the walk-forward already guarantees that those returns were known), and it then predicts the next block by playing the environment forward one day at a time from a flat book,
with its own previous book in the observation. The "score" it returns for an asset is the agent's target weight, to be traded as sized with the signal rule ``target_weights``; the experiment's backtest
(costs, rebalancing, volatility target) then treats it like any other strategy. A new fold starts flat, so a fold boundary costs one round of turnover.

The agent never sees a day after the one it is acting on: the observation is that day's features, its own current weights and its drawdown.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

from ..deep.data import FeatureScaler, SeqAccessor
from ..models import Model
from .env import PortfolioEnv, forward_returns, rollout

RECURRENT = "recurrent_ppo"
MIN_DAYS = 120


def _classes(kind: str):
    if kind == "ppo":
        from stable_baselines3 import PPO
        return PPO
    if kind == "sac":
        from stable_baselines3 import SAC
        return SAC
    from sb3_contrib import RecurrentPPO
    return RecurrentPPO


class RLAgent(Model):
    needs_dataset = True

    def __init__(self, kind: str, task: str, params: dict, seed: int):
        super().__init__(kind, task, dict(params), seed)
        p = self.params
        if p["reward"] not in ("return", "risk_adjusted", "dsr"):
            raise ValueError("reward must be return, risk_adjusted or dsr")
        if int(p["total_timesteps"]) < 64 or float(p["gross"]) <= 0 or not 0.0 < float(p["max_weight"]) <= 1.0 or int(p["hidden"]) < 4 or int(p["episode_days"]) < 20:
            raise ValueError(f"invalid setting(s) for {kind}")
        if not 0.0 < float(p["gamma"]) <= 1.0 or float(p["learning_rate"]) <= 0:
            raise ValueError("gamma must be in (0, 1] and learning_rate positive")
        self._ds = self._base = None
        self.agent_ = None

    def bind(self, ds) -> "RLAgent":
        self._ds, self._base = ds, None
        return self

    # ------------------------------------------------------------------------------------------------------------ the environment
    def _env_kwargs(self) -> dict:
        p = self.params
        return {"cost_bps": float(p["cost_bps"]), "impact": float(p["impact"]), "gross": float(p["gross"]), "max_weight": float(p["max_weight"]),
                "long_only": bool(int(p["long_only"])), "max_turnover": float(p["max_turnover"]), "reward": str(p["reward"]), "risk_aversion": float(p["risk_aversion"]),
                "turnover_penalty": float(p["turnover_penalty"]), "drawdown_penalty": float(p["drawdown_penalty"]), "episode_days": int(p["episode_days"]), "seed": self.seed}

    def _arrays(self, dates: pd.DatetimeIndex, X: pd.DataFrame):
        """Scaled features ``[days, assets, features]`` for ``dates`` (the rows of ``X`` over the dataset's own history), and which assets are present."""
        ds, cols = self._ds, self.features_
        if self._base is None:
            self._base = SeqAccessor.from_frame(ds.X[cols], ds.dates, ds.assets)
        t = ds.dates.get_indexer(dates)
        arr = self._base.arr[t].copy()
        own = SeqAccessor.from_frame(X[cols], ds.dates, ds.assets)
        mine = np.isfinite(own.arr[t]).any(axis=2)
        arr[mine] = own.arr[t][mine]
        present = np.isfinite(arr).any(axis=2)
        return self.scaler_.transform(arr), present

    # ------------------------------------------------------------------------------------------------------------ fitting
    def fit(self, X: pd.DataFrame, y: pd.Series, dates: np.ndarray | None = None) -> "RLAgent":
        if self._ds is None:
            raise ValueError(f"{self.kind} needs the dataset's returns: it must be bound to a dataset before it is fitted")
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import safe_mean

        Xs = self._select(X)
        self.features_ = list(Xs.columns)
        self.n_train_ = len(Xs)
        ds, p = self._ds, self.params
        d = Xs.index.get_level_values(0)
        lo, hi = ds.dates.get_indexer([d.min()])[0], ds.dates.get_indexer([d.max()])[0]
        days = ds.dates[lo:hi + 1]
        if len(days) < MIN_DAYS:
            raise ValueError(f"{len(days)} training days are too few to train an agent (at least {MIN_DAYS})")
        self.scaler_ = FeatureScaler().fit(Xs.to_numpy(dtype=np.float32))
        feats, present = self._arrays(days, Xs)
        fwd = forward_returns(ds.returns[ds.assets].to_numpy(dtype=np.float64), ds.lag)[lo:hi + 1]
        mask = present & ds.investable[ds.assets].to_numpy()[lo:hi + 1]
        env = PortfolioEnv(feats, fwd, mask, **self._env_kwargs())
        algo = _classes(self.kind)
        steps, seed = int(p["total_timesteps"]), self.seed
        common = {"learning_rate": float(p["learning_rate"]), "gamma": float(p["gamma"]), "batch_size": int(p["batch_size"]), "seed": seed, "device": str(p["device"]), "verbose": 0}
        arch = [int(p["hidden"]), int(p["hidden"])]
        if self.kind == "sac":
            kw = dict(buffer_size=int(p["buffer_size"]), learning_starts=int(p["learning_starts"]), tau=float(p["tau"]), policy_kwargs={"net_arch": arch})
        elif self.kind == "ppo":
            kw = dict(n_steps=int(p["n_steps"]), n_epochs=int(p["n_epochs"]), ent_coef=float(p["ent_coef"]), clip_range=float(p["clip_range"]), policy_kwargs={"net_arch": arch})
        else:
            kw = dict(n_steps=int(p["n_steps"]), n_epochs=int(p["n_epochs"]), ent_coef=float(p["ent_coef"]), clip_range=float(p["clip_range"]),
                      policy_kwargs={"lstm_hidden_size": int(p["lstm_hidden"]), "n_lstm_layers": 1, "net_arch": [int(p["hidden"])]})
            common["batch_size"] = int(p["n_steps"])                                     # a recurrent policy trains on whole rollouts
        history = {"timesteps": [], "mean_reward": []}
        every = max(steps // 40, 64)

        class Log(BaseCallback):
            def _on_step(self) -> bool:
                if self.num_timesteps % every < 1 or self.num_timesteps == steps:
                    buf = list(self.model.ep_info_buffer)
                    if buf:
                        history["timesteps"].append(int(self.num_timesteps))
                        history["mean_reward"].append(float(safe_mean([e["r"] for e in buf])))
                return True

        t0 = time.perf_counter()
        self.agent_ = algo("MlpLstmPolicy" if self.kind == RECURRENT else "MlpPolicy", Monitor(env), **common, **kw)
        self.agent_.learn(total_timesteps=steps, callback=Log())
        self.train_seconds_, self.rewards_ = time.perf_counter() - t0, history
        self.n_params_ = int(sum(q.numel() for q in self.agent_.policy.parameters()))
        self.train_days_ = int(len(days))
        self.obs_dim_ = int(env.observation_space.shape[0])
        self.n_assets_ = env.n_assets
        books = self._play(feats, mask)
        net = (books * np.nan_to_num(fwd)).sum(axis=1) - self._env_kwargs()["cost_bps"] / 1e4 * np.abs(np.diff(books, axis=0, prepend=0.0)).sum(axis=1)
        self.train_curve_ = {"dates": [str(x.date()) for x in days[::max(1, len(days) // 200)]], "equity": [float(v) for v in np.cumprod(1.0 + net)[::max(1, len(days) // 200)]],
                             "sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(252.0)) if net.std(ddof=1) > 0 else 0.0}
        return self

    # ------------------------------------------------------------------------------------------------------------ playing
    def _policy(self):
        agent = self.agent_

        def act(obs, state, first):
            if self.kind == RECURRENT:
                a, state = agent.predict(obs, state=state, episode_start=np.array([first]), deterministic=True)
                return a, state
            a, _ = agent.predict(obs, deterministic=True)
            return a, state
        return act

    def _play(self, feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return rollout(self._policy(), feats, mask, self._env_kwargs(), self.kind == RECURRENT)

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        ds = self._ds
        if ds is None:
            raise ValueError("the agent must be bound to a dataset before it predicts")
        days = pd.DatetimeIndex(sorted(set(X.index.get_level_values(0))))
        feats, present = self._arrays(days, X)
        mask = present & ds.investable[ds.assets].to_numpy()[ds.dates.get_indexer(days)]
        books = self._play(feats, mask)
        t = days.get_indexer(X.index.get_level_values(0))
        a = pd.Index(ds.assets).get_indexer(X.index.get_level_values(1))
        return books[t, a]

    def importance(self) -> pd.Series:
        """The size of the first layer's weights on each feature (all assets together): a rough guide to what the policy reads, not an attribution."""
        try:

            policy = self.agent_.policy
            layer = (policy.mlp_extractor.policy_net[0].weight if self.kind == "ppo" else policy.actor.latent_pi[0].weight if self.kind == "sac" else policy.lstm_actor.weight_ih_l0)
            w = layer.detach().abs().mean(dim=0).cpu().numpy()[: self.n_assets_ * len(self.features_)].reshape(self.n_assets_, len(self.features_)).sum(axis=0)
            return pd.Series(w / (w.sum() + 1e-12), index=self.features_)
        except Exception:                                                                 # an unfamiliar policy layout must not lose the experiment
            return pd.Series(0.0, index=self.features_)

    def diagnostics(self) -> dict:
        return {"algo": self.kind, "timesteps": int(self.params["total_timesteps"]), "rewards": self.rewards_, "reward": self.params["reward"], "n_params": self.n_params_, "device": str(self.params["device"]),
                "seconds": self.train_seconds_, "train_days": self.train_days_, "train_curve": self.train_curve_, "constraints": {"gross": float(self.params["gross"]), "max_weight": float(self.params["max_weight"]),
                "long_only": bool(int(self.params["long_only"])), "max_turnover": float(self.params["max_turnover"]), "cost_bps": float(self.params["cost_bps"])}}

    # ------------------------------------------------------------------------------------------------------------ serialization
    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_ds=None, _base=None)
        if self.agent_ is not None:
            buf = io.BytesIO()
            self.agent_.save(buf)
            state.update(agent_=None, _agent_bytes=buf.getvalue())
        return state

    def __setstate__(self, state):
        blob = state.pop("_agent_bytes", None)
        self.__dict__.update(state)
        if blob is not None:
            self.agent_ = _classes(self.kind).load(io.BytesIO(blob), device="cpu")


def build(kind: str, task: str, params: dict, seed: int) -> Model:
    if task != "regression":
        raise ValueError(f"{kind} is trained on returns, not on a target; use a regression target")
    return RLAgent(kind, task, params, seed)
