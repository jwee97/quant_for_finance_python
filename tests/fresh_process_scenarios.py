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


def _rl(name: str):
    def run():
        from src.rl import deep
        from src.rl.envs import RegimeFractionEnv

        env = RegimeFractionEnv(seed=0)
        agent = {"dqn": lambda: deep.DQN(hidden=16, warmup=100), "ddpg": lambda: deep.DDPG(hidden=16, warmup=100), "td3": lambda: deep.TD3(hidden=16, warmup=100), "sac": lambda: deep.SAC(hidden=16, warmup=100),
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
    "forecast_patchtst": _forecast("patchtst"), "forecast_tsmixer": _forecast("tsmixer"), "graph_forecast": _graph_forecast, "encoder_ae": _encoder("autoencoder"), "encoder_contrastive": _encoder("contrastive"),
    "wgan_gp": _wgan, "factor_vae": _factor_vae, "diffusion": _diffusion, "deep_hedging": _hedging, "pinn": _pinn,
    "rl_dqn": _rl("dqn"), "rl_ddpg": _rl("ddpg"), "rl_td3": _rl("td3"), "rl_sac": _rl("sac"), "rl_ppo": _rl("ppo"),
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
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(0.004)
    threading.Thread(target=watch, daemon=True).start()


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

    if mode == "interrupted":
        _kill_when(folder, kill_after)
    with context.use(str(folder), resume=(mode == "resume"), grad_level="standard"):
        t0 = time.time()
        result = SCENARIOS[name]()
    result["seconds"] = time.time() - t0
    out.write_text(json.dumps(result, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
