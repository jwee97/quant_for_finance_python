"""Training scenarios for the fresh-process recovery tests, one small deterministic run per model family.

    python -m tests.fresh_process_scenarios <scenario> <run folder> reference|interrupted|resume <result.json> [--kill-after N]

``reference`` trains without interruption. ``interrupted`` trains until a watcher thread has seen N finished units in the run's metrics files and then kills its own process with SIGKILL (no exception, no cleanup:
what a killed worker looks like). ``resume`` is a NEW process: it starts the same scenario with ``resume`` set, restores the newest checkpoint and finishes. Each run writes the same JSON: the loss history, the
trained outputs and the final parameters, so that the test can compare the resumed run with the reference.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

FORECAST_KINDS = ("patchtst", "tsmixer", "nbeats", "nhits", "timemixer", "tft", "fnn", "cnn", "lstm", "gru", "transformer")     # every network `models.deep_forecast.build_network` can build
KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)                                  # Windows has no SIGKILL: SIGTERM there is TerminateProcess, which is as abrupt


def record(**row) -> None:
    """Append one result line to the file named by $QUANT_RECORD_RECOVERY (used to produce docs/generated/*.jsonl); a no-op otherwise."""
    path = os.environ.get("QUANT_RECORD_RECOVERY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")


def maxdiff(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def die() -> None:
    """End this process at once, with no exception, no cleanup and no flushed buffers (a killed worker)."""
    os.kill(os.getpid(), KILL_SIGNAL)


def was_killed(returncode: int) -> bool:
    """Whether a subprocess's exit status is that of a process killed by ``die`` (-9 on POSIX; a non-zero status on Windows, where TerminateProcess reports the signal number)."""
    return returncode == -int(KILL_SIGNAL) if sys.platform != "win32" else returncode not in (0,)


SLOW = {"default": 0.002, "rl": 0.0}                      # seconds added to every optimizer step so that a unit lasts long enough for the kill to land inside the run


def flat(*modules) -> list:
    import torch

    return [float(x) for x in torch.cat([p.detach().flatten() for m in modules for p in m.parameters()]).tolist()]


# ----------------------------------------------------------------------------------------------------------------------------------- scenarios
def _model_scenario(kind: str, extra: dict | None = None):
    def run():
        import torch

        from src.ml import config as mc, datasets as dsm, models as md
        from tests.ml_helpers import make_bundle

        cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 400, "refit_every": 250}}, defaults=mc.DEFAULTS)
        ds = dsm.build_dataset(make_bundle(8, 1100, seed=31, alpha=1.0), cfg["dataset"], "deep")
        spec = {"patchtst": {"d_model": 8, "n_heads": 2, "ff_dim": 16, "patch_len": 4, "stride": 2, "channels": 4}, "nbeats": {"width": 16, "blocks": 2, "layers": 2}, "nhits": {"width": 16, "blocks": 2, "layers": 2},
                "tft": {"n_heads": 2}, "tcn": {"layers": 2}, "autoencoder": {"latent_dim": 4, "layers": 1}, "vae": {"latent_dim": 4, "layers": 1}}
        params = {"epochs": 8, "patience": 50, "max_rows": 1500, "hidden": 8, "batch_size": 128, "lookback": 8, "num_threads": 1, "device": "cpu", "grad_monitor": "standard", **spec.get(kind, {}), **(extra or {})}
        params = {k: v for k, v in params.items() if k in md._INFO[kind].params}
        pos = ds.positions()
        rows = np.flatnonzero((pos >= 80) & (pos < 700) & ds.Y[ds.target].notna().to_numpy())
        model = md.make_model(kind, params, "regression", 3)
        model.bind(ds)
        model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos[rows])
        h = model.history_
        return {"loss": h.val_loss, "train_loss": h.train_loss, "params": flat(model.net_), "extra": {"recon_error": float(getattr(model, "recon_error_", float("nan")))}}
    return run


def _graph_model_scenario(kind: str):
    return _model_scenario(kind, {"hidden": 8, "layers": 2, "epochs": 8, "patience": 50, "batch_size": 16})


def _forecast(kind: str):
    def run():
        from src.models.deep_forecast import TrainSettings, train_and_predict

        rng = np.random.default_rng(1)
        n = 700
        X = rng.normal(size=(n, 252))
        y = 0.8 * X[:, -21:].mean(axis=1) * np.sqrt(21) + 0.3 * rng.normal(size=n)
        a = rng.integers(0, 3, n)
        out = train_and_predict(kind, X[:500], a[:500], y[:500], X[500:600], a[500:600], y[500:600], X[600:], a[600:], 5, 3, None, TrainSettings(1e-3, 1e-2, 64, 10, 50))
        return {"loss": out["val_loss"], "train_loss": out["train_loss"], "outputs": out["pred"].tolist()}
    return run


def _graph_forecast():
    from src.models.deep_forecast import TrainSettings
    from src.models.graph_forecast import train_and_predict_graph

    n, f, days = 6, 4, 260
    rng = np.random.default_rng(4)
    F = rng.normal(size=(days, n, f))
    nbr = np.stack([np.stack([[(a + 1) % n, (a + 2) % n] for a in range(n)]) for _ in range(days)])
    y = 0.9 * np.tanh(F[:, :, 0]) + 0.3 * rng.normal(size=(days, n))
    y[:20, 0] = np.nan
    out = train_and_predict_graph(F[:160], nbr[:160], y[:160], F[160:200], nbr[160:200], y[160:200], F[200:], nbr[200:], 2, n, TrainSettings(1e-3, 1e-2, 24, 10, 50), {"hidden": 8, "heads": 2})
    return {"loss": out["val_loss"], "train_loss": out["train_loss"], "outputs": np.asarray(out["pred"]).reshape(-1).tolist()}


def _encoder(kind: str):
    def run():
        from src.models.representation import train_encoder

        X = np.random.default_rng(2).normal(size=(220, 32))
        out = train_encoder(X, kind, embed_dim=4, hidden=16, epochs=10, batch_size=32, seed=1)
        return {"loss": out["loss"], "train_loss": out["loss"], "outputs": out["encode"](X[:20]).reshape(-1).tolist()}
    return run


def _wgan():
    from src.models.generative import WGANGP

    X = np.random.default_rng(3).normal(size=(240, 3)) * [1.0, 2.0, 0.5] + [0.0, 1.0, -1.0]
    m = WGANGP(latent=4, hidden=16, layers=2, steps=250, n_critic=2, batch=32, seed=1).fit(X)
    return {"loss": m.critic_gap, "train_loss": m.critic_gap, "outputs": m.sample(20, seed=5).reshape(-1).tolist(), "params": flat(m.G, m.D)}


def _factor_vae():
    from src.models.generative import FactorVAE

    X = np.random.default_rng(3).normal(size=(240, 4)) @ np.random.default_rng(4).normal(size=(4, 4))
    m = FactorVAE(factors=2, hidden=16, epochs=10, batch=64, seed=1).fit(X)
    return {"loss": m.losses, "train_loss": m.losses, "outputs": m.sample(20, seed=5).reshape(-1).tolist(), "params": flat(m.enc, m.g) + [float(v) for v in m.B.detach().flatten()] + [float(m.log_nu)],
            "extra": {"loadings": m.loadings().reshape(-1).tolist(), "nu": m.degrees_of_freedom}}


def _diffusion():
    from src.models.diffusion import Diffusion

    X = np.random.default_rng(3).normal(size=(240, 3)) * [1.0, 2.0, 0.5]
    m = Diffusion(steps=20, hidden=16, layers=2, epochs=10, batch=64, seed=1).fit(X)
    return {"loss": m.losses, "train_loss": m.losses, "outputs": m.sample(10, seed=5).reshape(-1).tolist(), "params": flat(m.net)}


def _hedging():
    from src.derivatives import deep_hedging as dh

    policy, curve = dh.train(100.0, 100.0, 0.5, 0.02, 0.0, 0.2, 0.001, 6, True, 1.0, 5.0, iterations=50, batch=128, seed=1)
    return {"loss": curve, "train_loss": curve, "params": flat(policy)}


def _pinn():
    from src.models import pinn

    m = pinn.BSMPINN(0.03, 0.01, 0.25, 1.0, steps=1000, batch=64, hidden=8, layers=2, lbfgs_steps=15).fit()
    return {"loss": m.history, "train_loss": m.history, "outputs": [float(x) for x in m.price(np.array([90.0, 100.0, 110.0]), 100.0, 0.5)], "params": flat(m.net)}


def _pinn_full(kind: str):
    """The configurations the accuracy tests of ``test_generative_pinn.py`` use, so that the recovered network can be held to the same pricing tolerance."""
    def run():
        from src.models import pinn

        if kind == "bsm":
            m = pinn.BSMPINN(0.03, 0.01, 0.25, 1.0, steps=1200, lbfgs_steps=300).fit()
            out = [float(x) for tau in (0.5, 1.0) for x in m.price(np.array([80.0, 90.0, 100.0, 110.0, 120.0]), 100.0, tau)]
        elif kind == "vasicek":
            m = pinn.VasicekPINN(0.8, 0.05, 0.02, maturity=5.0, steps=1500, lbfgs_steps=100).fit()
            out = [float(x) for x in m.price(np.array([0.0, 0.03, 0.05, 0.08, 0.10]), 3.0)]
        else:
            kw = dict(lam=0.5, mu_j=-0.1, sigma_j=0.15) if kind == "bates" else {}
            m = pinn.HestonPINN(0.03, 0.0, 2.0, 0.04, 0.3, -0.7, steps=900, lbfgs_steps=150, **kw).fit()
            out = [float(x) for x in m.price(np.array([90.0, 100.0, 110.0]), 100, 1.0, np.array([0.03, 0.04, 0.05]))]
        return {"loss": m.history, "train_loss": m.history, "outputs": out, "params": flat(m.net)}
    return run


def _sb3(kind: str):
    """A Stable-Baselines3 agent (PPO, SAC or recurrent PPO) fitted by the platform's own model class on a small synthetic market; run it with a Python that has stable-baselines3, gymnasium and sb3-contrib."""
    def run():
        from src.ml import config as mc, datasets as dsm, models as md
        from tests.ml_helpers import make_bundle

        cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 400, "refit_every": 250}}, defaults=mc.DEFAULTS)
        ds = dsm.build_dataset(make_bundle(8, 1100, seed=31, alpha=1.0), cfg["dataset"], "rl")
        base = {"hidden": 16, "episode_days": 80, "batch_size": 32, "device": "cpu", "grad_monitor": "standard"}
        params = {"ppo": {**base, "total_timesteps": 1280, "n_steps": 128, "n_epochs": 2, "checkpoint_timesteps": 128},
                  "sac": {**base, "total_timesteps": 1000, "learning_starts": 100, "buffer_size": 2000, "checkpoint_timesteps": 100},
                  "recurrent_ppo": {**base, "total_timesteps": 640, "n_steps": 64, "n_epochs": 2, "lstm_hidden": 8, "checkpoint_timesteps": 64}}[kind]
        pos = ds.positions()
        rows = np.flatnonzero((pos >= 80) & (pos < 700) & ds.Y[ds.target].notna().to_numpy())
        model = md.make_model(kind, params, "regression", 3)
        model.bind(ds)
        model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos[rows])
        net = model.agent_.policy
        return {"loss": model.rewards_["mean_reward"], "train_loss": model.rewards_["mean_reward"], "outputs": model.train_curve_["equity"], "params": flat(net),
                "extra": {"timesteps": [float(x) for x in model.rewards_["timesteps"]], "num_timesteps": float(model.agent_.num_timesteps)}}
    return run


def _rl(name: str):
    def run():
        from src.rl import deep
        from src.rl.envs import RegimeFractionEnv

        env = RegimeFractionEnv(seed=0)
        agent = {"dqn": lambda: deep.DQN(hidden=16, warmup=100), "double_dqn": lambda: deep.DQN(double=True, hidden=16, warmup=100, target_every=50), "ddpg": lambda: deep.DDPG(hidden=16, warmup=100), "td3": lambda: deep.TD3(hidden=16, warmup=100), "sac": lambda: deep.SAC(hidden=16, warmup=100),
                 "ppo": lambda: deep.PPO(hidden=16, rollout=200, epochs=2, minibatch=50)}[name]()
        steps = 1200 if name == "ppo" else 2000
        agent.train(env, steps=steps, seed=3)
        table = agent.policy_table(env)
        nets = [m for k, m in vars(agent).items() if hasattr(m, "parameters") and k in ("q", "actor", "critic")]
        return {"loss": [], "train_loss": [], "outputs": [float(x) for x in table], "params": flat(*nets) + ([float(agent.log_std)] if name == "ppo" else [])}
    return run


SCENARIOS = {
    "ae": _model_scenario("autoencoder"), "vae": _model_scenario("vae"), "nbeats": _model_scenario("nbeats"), "nhits": _model_scenario("nhits"), "lstm": _model_scenario("lstm"), "gru": _model_scenario("gru"), "tcn": _model_scenario("tcn"),
    "patchtst": _model_scenario("patchtst"), "itransformer": _model_scenario("itransformer"), "tft": _model_scenario("tft"), "dlinear": _model_scenario("dlinear"), "gcn": _graph_model_scenario("gcn"), "gat": _graph_model_scenario("gat"),
    **{f"forecast_{k}": _forecast(k) for k in FORECAST_KINDS}, "graph_forecast": _graph_forecast, "encoder_ae": _encoder("autoencoder"), "encoder_contrastive": _encoder("contrastive"),
    "wgan_gp": _wgan, "factor_vae": _factor_vae, "diffusion": _diffusion, "deep_hedging": _hedging, "pinn": _pinn,
    "pinn_bsm": _pinn_full("bsm"), "pinn_vasicek": _pinn_full("vasicek"), "pinn_heston": _pinn_full("heston"), "pinn_bates": _pinn_full("bates"),
    "sb3_ppo": _sb3("ppo"), "sb3_sac": _sb3("sac"), "sb3_recurrent_ppo": _sb3("recurrent_ppo"),
    "rl_double_dqn": _rl("double_dqn"), "rl_dqn": _rl("dqn"), "rl_ddpg": _rl("ddpg"), "rl_td3": _rl("td3"), "rl_sac": _rl("sac"), "rl_ppo": _rl("ppo"),
}


# ----------------------------------------------------------------------------------------------------------------------------------- harness
def _kill_when(folder: Path, units: int) -> None:
    def watch():
        while True:
            seen = 0
            for f in folder.rglob("metrics.jsonl"):
                try:
                    seen = max(seen, sum(1 for line in f.read_text().splitlines() if '"type": "epoch"' in line))
                except OSError:
                    pass
            if seen >= units:
                die()
            time.sleep(0.004)
    threading.Thread(target=watch, daemon=True).start()


def _kill_in_closure(k: int) -> None:
    import torch

    real = torch.optim.LBFGS.step

    def step(self, closure):
        return real(self, lambda: _count(closure, k))
    torch.optim.LBFGS.step = step


_COUNT = {"n": 0}


def _count(closure, k):
    _COUNT["n"] += 1
    if _COUNT["n"] == k:
        die()
    return closure()


def _kill_at_finish() -> None:
    from src.ml.deep.session import TrainingSession

    real = TrainingSession.finish

    def finish(self, **kw):
        die()
        return real(self, **kw)
    TrainingSession.finish = finish


def main(argv: list[str]) -> None:
    name, folder, mode, out = argv[0], Path(argv[1]), argv[2], Path(argv[3])
    kill_after = int(argv[argv.index("--kill-after") + 1]) if "--kill-after" in argv else 0
    import torch

    torch.set_num_threads(1)
    delay = SLOW["rl"] if name.startswith("rl_") else SLOW["default"]
    if delay:
        from torch.optim import optimizer as _o

        _o.register_optimizer_step_post_hook(lambda opt, a, k: time.sleep(delay))
    from src.ml.deep import context

    kill_at = argv[argv.index("--kill-at") + 1] if "--kill-at" in argv else ""
    if mode == "interrupted" and kill_at.startswith("closure:"):
        _kill_in_closure(int(kill_at.split(":")[1]))                                          # SIGKILL inside the k-th loss evaluation of an L-BFGS step
    elif mode == "interrupted" and kill_at == "finish":
        _kill_at_finish()                                                                      # SIGKILL after the last unit is committed, before the closing row
    elif mode == "interrupted":
        _kill_when(folder, kill_after)
    if mode == "plain":                                                                    # no training run at all: the code path of a plain library call
        t0 = time.time()
        result = SCENARIOS[name]()
        result["seconds"] = time.time() - t0
        out.write_text(json.dumps(result, default=float))
        return
    with context.use(str(folder), resume=(mode == "resume"), grad_level="standard"):
        t0 = time.time()
        result = SCENARIOS[name]()
    result["seconds"] = time.time() - t0
    out.write_text(json.dumps(result, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
