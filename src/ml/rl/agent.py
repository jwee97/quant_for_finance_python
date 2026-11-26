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
from ..causal import CausalView
from ..models import Model
from .env import STATE_SCHEMA_VERSION, PortfolioEnv, event_kind, rollout

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
        self._ds, self._base = CausalView.of(ds), None
        return self

    # ------------------------------------------------------------------------------------------------------------ the environment
    def _env_kwargs(self) -> dict:
        p = self.params
        return {"cost_bps": float(p["cost_bps"]), "impact": float(p["impact"]), "gross": float(p["gross"]), "max_weight": float(p["max_weight"]),
                "long_only": bool(int(p["long_only"])), "max_turnover": float(p["max_turnover"]), "reward": str(p["reward"]), "risk_aversion": float(p["risk_aversion"]),
                "turnover_penalty": float(p["turnover_penalty"]), "drawdown_penalty": float(p["drawdown_penalty"]), "episode_days": int(p["episode_days"]), "seed": self.seed,
                "lag": int(self._ds.lag) if self._ds is not None else 1}

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
        rets = ds.returns[ds.assets].to_numpy(dtype=np.float64)[lo:hi + 1]                 # the returns of the training window, day by day: the environment advances its clock through them
        mask = present & ds.investable[ds.assets].to_numpy()[lo:hi + 1]
        env = PortfolioEnv(feats, rets, mask, **self._env_kwargs())
        algo = _classes(self.kind)
        steps, seed = int(p["total_timesteps"]), self.seed
        from . import sb3_state

        common = {"learning_rate": sb3_state.make_schedule(p["learning_rate"], p.get("lr_schedule", "constant"), steps), "gamma": float(p["gamma"]), "batch_size": int(p["batch_size"]), "seed": seed, "device": str(p["device"]), "verbose": 0}
        arch = [int(p["hidden"]), int(p["hidden"])]
        if self.kind == "sac":
            kw = dict(buffer_size=int(p["buffer_size"]), learning_starts=int(p["learning_starts"]), tau=float(p["tau"]), policy_kwargs={"net_arch": arch})
        elif self.kind == "ppo":
            kw = dict(n_steps=int(p["n_steps"]), n_epochs=int(p["n_epochs"]), ent_coef=float(p["ent_coef"]), clip_range=sb3_state.make_schedule(p["clip_range"], p.get("clip_schedule", "constant"), steps), policy_kwargs={"net_arch": arch})
        else:
            kw = dict(n_steps=int(p["n_steps"]), n_epochs=int(p["n_epochs"]), ent_coef=float(p["ent_coef"]), clip_range=sb3_state.make_schedule(p["clip_range"], p.get("clip_schedule", "constant"), steps),
                      policy_kwargs={"lstm_hidden_size": int(p["lstm_hidden"]), "n_lstm_layers": 1, "net_arch": [int(p["hidden"])]})
            common["batch_size"] = int(p["n_steps"])                                     # a recurrent policy trains on whole rollouts
        history = {"timesteps": [], "mean_reward": []}
        every = max(steps // 40, 64)

        class Log(BaseCallback):
            def _on_training_start(self) -> None:
                sb3_state.set_clock(self.model)                                                  # a non-constant schedule reads the agent's own timestep count, so blocks and one plain call train alike

            def _on_step(self) -> bool:
                sb3_state.set_clock(self.model)
                if self.num_timesteps % every < 1 or self.num_timesteps == steps:
                    buf = list(self.model.ep_info_buffer)
                    if buf:
                        history["timesteps"].append(int(self.num_timesteps))
                        history["mean_reward"].append(float(safe_mean([e["r"] for e in buf])))
                return True

        t0 = time.perf_counter()
        policy_name = "MlpLstmPolicy" if self.kind == RECURRENT else "MlpPolicy"
        agent = algo(policy_name, Monitor(env), **common, **kw)
        self.agent_ = self._learn(agent, algo, Log, history, steps, common, kw, feats, rets, mask)
        self.train_seconds_, self.rewards_ = time.perf_counter() - t0, history
        self.n_params_ = int(sum(q.numel() for q in self.agent_.policy.parameters()))
        self.train_days_ = int(len(days))
        self.obs_dim_ = int(env.observation_space.shape[0])
        self.n_assets_ = env.n_assets
        self.train_lo_, self.train_hi_, self.schema_version_ = int(lo), int(hi), STATE_SCHEMA_VERSION
        _, trace = rollout(self._policy(), feats, rets, mask, self._env_kwargs(), self.kind == RECURRENT)
        eq = trace["equity"]
        net = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)
        step = max(1, len(days) // 200)
        self.train_curve_ = {"dates": [str(x.date()) for x in days[::step]], "equity": [float(v) for v in eq[::step]],
                             "sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(252.0)) if len(net) > 1 and net.std(ddof=1) > 0 else 0.0}
        return self

    def _learn(self, agent, algo, Log, history: dict, steps: int, common: dict, kw: dict, feats, rets, mask):
        """Train ``agent`` for ``steps`` timesteps. Outside a training run this is SB3's single ``learn`` call, exactly as before. Inside one (a job, the Training monitor) the same training is cut into blocks of
        ``checkpoint_timesteps`` (a whole number of rollouts for PPO), and after each block a checkpoint is committed: SB3's own serialisation of the model and replay buffer plus the application manifest of what SB3 does not
        keep (``sb3_state``). A resumed run rebuilds the agent from the newest valid checkpoint and continues; the guarantee it reports is the one the saved state and the platform's limits (``sb3_state.describe_limits``) support."""
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
        from stable_baselines3.common.utils import safe_mean

        from ..deep import context as train_context
        from ..deep.context import CheckpointIncompatible
        from ..deep.session import TrainingSession, data_fingerprint
        from .. import live
        from . import sb3_state

        p, kind = self.params, self.kind
        if train_context.current() is None:
            agent.learn(total_timesteps=steps, callback=Log())
            return agent
        on_policy = kind in ("ppo", RECURRENT)
        n_steps = int(p.get("n_steps", 1)) if on_policy else 1
        asked = int(p.get("checkpoint_timesteps", 0) or 0)
        chunk = asked if asked > 0 else max(steps // 8, 256)
        chunk = max(-(-chunk // n_steps) * n_steps, n_steps)                                 # PPO collects whole rollouts: a block is a whole number of them
        units = max(-(-steps // chunk), 1)
        holder = {"agent": agent}
        skip = {"checkpoint_timesteps", "grad_monitor", "total_timesteps", "device"}
        config = {"kind": kind, "params": {k: v for k, v in sorted(p.items()) if k not in skip}, "seed": self.seed, "env": self._env_kwargs(), "versions": sb3_state.versions()}
        reasons = sb3_state.ceiling_reasons(agent, kind, str(p.get("device") or "auto"), chunk)
        device = str(agent.device.type)

        def apply(extra):
            """Rebuild the agent from the checkpoint and check it against the digests recorded with it. A mismatch that would change what is trained (timesteps, policy, optimizer, replay buffer, next observation, environment) stops the
            continuation; the others lower the guarantee."""
            saved = extra["sb3"]
            new, _, verdict = sb3_state.restore(algo, saved, device)
            holder["agent"], holder["verdict"] = new, verdict
            history.update(extra["history"])
            if verdict["blocking"]:
                raise CheckpointIncompatible([f"the restored agent does not match the checkpoint: {m}" for m in verdict["blocking"]])
            return verdict["downgrade"]

        sess = TrainingSession("sb3_" + kind, config=config, data={"arrays": data_fingerprint(feats, rets, mask), "features": list(self.features_)}, modules={}, optimizers={},
                               capture=lambda: {"sb3": sb3_state.capture(holder["agent"], kind), "history": {k: list(v) for k, v in history.items()}}, apply=apply, units=units, unit_name="timestep block",
                               family="rl", ceiling="epoch_boundary" if reasons else "exact", device=device, grad_level=str(p.get("grad_monitor") or "auto"), steps_per_unit=chunk,
                               n_params=int(sum(q.numel() for q in agent.policy.parameters())), lr=float(p["learning_rate"]))
        with sess:
            resumed = sess.restore()
            first_chunk = True
            if resumed is not None and resumed.level != "warm_start":
                agent = holder["agent"]
                verdict = holder["verdict"]
                sess.log({"type": "recovery_checks", "at": time.time(), "checks": verdict["checks"], "blocking": verdict["blocking"], "downgrade": verdict["downgrade"]})
                first_chunk = False
                if not hasattr(agent, "_last_lstm_states") and kind == RECURRENT:
                    sess.log({"type": "recovery_note", "message": "the recurrent state could not be restored", "at": time.time()})
            monitors = ({"actor": sess.monitor("actor", agent.actor), "critic": sess.monitor("critic", agent.critic)} if kind == "sac" else {"policy": sess.monitor("policy", agent.policy)})
            if resumed is not None:
                for name, mon in monitors.items():
                    mon.load_state_dict(((resumed.state.get("grad") or {}).get(name)))
            if resumed is None:
                sess.log({"type": "diagnostics_note", "at": time.time(), "message": "gradient diagnostics for Stable-Baselines3 agents are limited to the public interface: the gradient the last update left on the networks is read between updates (PPO's after its own clipping, SAC's unclipped); per-step pre-clip norms and update-to-weight ratios are not available"})
                sess.log({"type": "recovery_limits", "at": time.time(), "limits": sb3_state.describe_limits(kind, str(agent.device.type)), "exactness_blocked_by": reasons,
                          "private_sb3_members_used": sorted(sb3_state.PRIVATE_USED)})
            state = {"step": 0}

            class Lifecycle(BaseCallback):
                """Reads, never changes: the gradients the last update left on the networks (public ``.grad``: PPO's are after its own clipping, SAC's are unclipped) and the cancellation and resource checks of a job."""

                def _observe(self) -> None:
                    for mon in monitors.values():
                        mon.observe(state["step"])
                    state["step"] += 1

                def _on_rollout_start(self) -> None:
                    if on_policy and getattr(self.model, "_n_updates", 0) > 0:
                        self._observe()

                def _on_step(self) -> bool:
                    live.batch(int(self.num_timesteps), steps, 0.0, int(self.num_timesteps))                       # a cancelled job or a memory stop ends the training here
                    if not on_policy and self.model._n_updates > 0 and self.num_timesteps % 8 == 0:
                        self._observe()
                    return True

            callbacks = CallbackList([Log(), Lifecycle()])
            for u in range(resumed.start if resumed is not None and resumed.level != "warm_start" else 0, units):
                sess.begin_unit(u, chunk)
                for mon in monitors.values():
                    mon.begin_epoch(u, max(chunk // max(n_steps, 1), 1), None)
                state["step"] = 0
                target = min((u + 1) * chunk, steps)
                agent.learn(total_timesteps=max(target - agent.num_timesteps, 1), callback=callbacks, reset_num_timesteps=first_chunk)
                first_chunk = False
                buf = list(agent.ep_info_buffer)
                values = {k.replace("train/", ""): float(v) for k, v in getattr(agent.logger, "name_to_value", {}).items() if k.startswith("train/") and np.isfinite(float(v))}
                loss = values.get("loss", values.get("critic_loss"))
                metrics = {"train_loss": loss, "val_loss": None, "timesteps": float(agent.num_timesteps), "episodes": float(getattr(agent, "_episode_num", len(buf))), "updates": float(getattr(agent, "_n_updates", 0)),
                           "mean_reward": float(safe_mean([e["r"] for e in buf])) if buf else None, "mean_episode_length": float(safe_mean([e["l"] for e in buf])) if buf else None, **values}
                if sess.end_unit(u, metrics, last=agent.num_timesteps >= steps) == "stop":
                    break
            sess.finish()
        return agent

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

    def _chain(self, start: int, end: int, X: pd.DataFrame) -> tuple[np.ndarray, list]:
        """The books decided on days ``start .. end`` by playing the policy forward from a flat account on day ``start``, one session after another, with the ledger the policy was trained in, and the events
        the environment reported on each day. The result for a day does not depend on how many days are asked for after it, or in how many calls: every call replays from the start of the chain."""
        ds = self._ds
        days = ds.dates[start:end + 1]
        feats, present = self._arrays(days, X)
        mask = present & ds.investable[ds.assets].to_numpy()[start:end + 1]
        rets = ds.returns[ds.assets].to_numpy(dtype=np.float64)[start:end + 1]
        books, trace = rollout(self._policy(), feats, rets, mask, self._env_kwargs(), self.kind == RECURRENT)
        return books, trace["events"]

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Each day's target weights. Two chains, both starting flat: the training window (in-sample) from its first day, and everything after it from the first day after it (a refit starts a new account)."""
        ds = self._ds
        if ds is None:
            raise ValueError("the agent must be bound to a dataset before it predicts")
        if getattr(self, "schema_version_", 1) != STATE_SCHEMA_VERSION:
            raise ValueError(f"this {self.kind} agent was trained with state schema {getattr(self, 'schema_version_', 1)}, the environment is at {STATE_SCHEMA_VERSION}: retrain it")
        t_row = ds.dates.get_indexer(X.index.get_level_values(0))
        a_row = pd.Index(ds.assets).get_indexer(X.index.get_level_values(1))
        if (t_row < 0).any() or (a_row < 0).any():
            raise ValueError("some rows are on a date or for an asset that is not in the dataset the agent is bound to")
        out = np.full(len(X), np.nan)
        seen = {}
        for first, sel in ((self.train_lo_, t_row <= self.train_hi_), (self.train_hi_ + 1, t_row > self.train_hi_)):
            if not sel.any():
                continue
            start, end = min(first, int(t_row[sel].min())), int(t_row[sel].max())
            books, events = self._chain(start, end, X)
            out[sel] = books[t_row[sel] - start, a_row[sel]]
            seen.update({start + rel: list(events[rel]) for rel in range(len(events))})                      # the same day replays to the same events, so this is the same whatever the call
        self.__dict__.setdefault("_events", {}).update(seen)
        return out

    def predict_extras(self, X: pd.DataFrame) -> dict:
        """The events of the days in ``X``, one row for each day on which the environment clipped an action, liquidated what was no longer eligible, trimmed a limit or shortened a trade: ``n_events``,
        their ``kinds`` (stable names, one for each event) and the ``detail`` (the environment's own sentences). A day with none has no row. The days are those the policy was replayed over; the frame is saved with the
        experiment and summarised in the registry, so a constraint that overrode the policy leaves a record."""
        ds = self._ds
        if ds is None:
            raise ValueError("the agent must be bound to a dataset before it predicts")
        days = sorted({int(t) for t in ds.dates.get_indexer(X.index.get_level_values(0)) if t >= 0})
        if any(d not in self.__dict__.get("_events", {}) for d in days):
            self.raw_predict(X)                                                                           # a model that has not replayed these days (a reloaded one) replays them now
        seen = self.__dict__["_events"]
        rows = [(ds.dates[d], len(seen[d]), ",".join(event_kind(e) for e in seen[d]), " | ".join(seen[d])) for d in days if seen.get(d)]
        frame = pd.DataFrame(rows, columns=["date", "n_events", "kinds", "detail"]).set_index("date")
        return {"events": frame}

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
                "seconds": self.train_seconds_, "train_days": self.train_days_, "train_curve": self.train_curve_, "state_schema": getattr(self, "schema_version_", 1), "constraints": {"gross": float(self.params["gross"]), "max_weight": float(self.params["max_weight"]),
                "long_only": bool(int(self.params["long_only"])), "max_turnover": float(self.params["max_turnover"]), "cost_bps": float(self.params["cost_bps"])}}

    # ------------------------------------------------------------------------------------------------------------ serialization
    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_ds=None, _base=None)
        state.pop("_events", None)                                                               # a replay's events are re-derived, never stored with the model
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
