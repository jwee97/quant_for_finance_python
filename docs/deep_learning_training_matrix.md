# Deep-learning training: checkpoints, recovery, gradient diagnostics and the Training monitor

This page is the model-by-model account of what every PyTorch training loop in the platform can do after the second reliability upgrade. Nothing is called done because a class or a button exists: each row names the test that backs it, and what is not verified is said so. The first page of this subject is `training_reliability.md` (the checkpoint format, recovery levels, resource profiles); this one adds the shared lifecycle, the other loops, the gradient diagnostics and the monitor.

## 1. The audit

Every `.backward()` and optimizer step in `src/` was found by search (`src/ml/`, `src/models/`, `src/rl/`, `src/derivatives/`, `src/strategies/`); `src/forecast/`, `src/scenarios/`, `src/equity/` and `src/engine/` contain no PyTorch training. Calls that are not training: `ml/deep/model.py` (saliency score), `models/explain.py` (attribution) and the closure of the L-BFGS polish in `models/pinn.py`.

Inference-only pretrained models (nothing to train, checkpoint or diagnose): `strategies/deep.py` `ChronosZeroShot` and `TimesFMZeroShot`.

## 2. Shared interfaces

* `ml/deep/checkpoint.py`: atomic, verified checkpoint writes with a previous generation kept; compatibility checks against configuration and data fingerprints, schema, software versions and device (`assess`); recovery levels.
* `ml/deep/trainer.py` `fit_network`: the shared trainer of the feature-based networks (N-BEATS, N-HiTS, LSTM, GRU, TCN, PatchTST, iTransformer, DLinear, TFT, autoencoder, VAE and the graph networks). It has the lifecycle built in.
* `ml/deep/session.py` `TrainingSession`: the same lifecycle for every other loop, which keeps its own mathematics: `restore()`, `begin_unit()`, `end_unit()`, `finish()`, optional `Callback` hooks (start, restore, unit start/end, checkpoint, event, resource warning, stop, failure, finish). A checkpoint holds every module, optimizer, scheduler and scaler state, the global and named random-number generators (NumPy `Generator`, `RandomState`, `torch.Generator`), the loop's own state through `capture`/`apply` (histories, best weights, counters, replay buffer, environment state) and the gradient detector's state. `ceiling` is the loop's honest limit: `exact` only when everything that shapes the next unit is restored.
* `ml/deep/gradients.py` `GradientMonitor`: the diagnostics (section 4). `ml/deep/context.py`: the run a job belongs to. Every job runs inside a training context (`<root>/data/user/training/<run id>/`), so a deep loop inside ANY job, not only `/api/ml/train`, is checkpointed, monitored and resumable through `POST /api/training/resume`.
* A loop with no active run (a direct library call) does nothing extra and costs nothing.

Logs and checkpoints stay consistent after a hard kill: the epoch row, the gradient summary and the unit's events are stored inside the checkpoint (`last`), and on resume the logs are reconciled with it (`runs.reconcile_logs`): rows of a unit that will be redone are dropped; a row the crash prevented is restored and marked `recovered`.

## 3. Model-by-model matrix

Status words: **I+T** implemented and tested, **I** implemented, not verified, **P** partial, **U** unsupported, **N/A** not applicable. "Fresh-process" means a worker killed with SIGKILL after unit 3 (no exception, no cleanup), then a NEW process restored the newest checkpoint and finished (`tests/test_fresh_process_recovery.py`); "exact" means weights, losses and outputs equal the uninterrupted run to 1e-6 (the observed difference was 0.0 in every scenario; CPU, one thread, same software).

| Model | Loop | Checkpoint | Fresh-process resume | Guarantee | Metrics in monitor | Global gradient | Layer gradient | Histogram | Resources | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| N-BEATS, N-HiTS | `fit_network` | I+T | I+T (`nbeats`, `nhits`) | exact | I+T | I+T | I+T | I+T | I+T | stacks and blocks are the layers |
| LSTM, GRU, TCN | `fit_network` | I+T | I+T | exact | I+T | I+T | I+T | I+T | I+T | recurrent thresholds for clipping |
| PatchTST, iTransformer, DLinear, TFT | `fit_network` | I+T | I+T | exact | I+T | I+T | I+T | I+T | I+T | attention thresholds |
| Autoencoder | `fit_network` | I+T | I+T (`ae`) | exact | I+T (reconstruction) | I+T | I+T | I+T | I+T | encoder/decoder layers listed by name |
| VAE | `fit_network` | I+T | I+T (`vae`) | exact | I+T (reconstruction, KL, KL weight) | I+T | I+T | I+T | I+T | the reparameterisation noise comes from the saved global generator; **there is no KL annealing schedule in this codebase**: `beta` is a constant in the fingerprint, so there is nothing to restore |
| GCN, GAT (`ml/graph.py`) | `fit_network` | I+T | I+T (`gcn`, `gat`) | exact | I+T | I+T | I+T | I+T | I+T | node order, adjacency hash, fit window and rows are in the data fingerprint |
| Forecast PatchTST, TSMixer (`models/deep_forecast`) | session | I+T | I+T | exact | I+T | I+T | I+T | I+T | I+T | the other kinds of `deep_family`/`deep_sequence` (TFT, residual, multiscale, FNN, TCN, recurrent, transformer) use the same loop; **only two kinds are exercised** |
| Graph-attention forecaster (`models/graph_forecast`) | session | I+T | I+T (`graph_forecast`) | exact | I+T | I+T | I+T | I+T | I+T | |
| Encoder: autoencoder, contrastive (`models/representation`) | session | I+T | I+T | exact | I+T | I+T | I+T | I+T | I+T | |
| WGAN-GP | session, two optimizers | I+T | I+T | exact | I+T (critic gap, penalty) | I+T per optimizer | I+T | I+T | I+T | generator and critic are monitored separately; a unit is 25 generator updates |
| Factor VAE | session | I+T | I+T | exact | I+T (reconstruction log-likelihood, KL, beta, t degrees of freedom) | I+T | I+T | I+T | I+T | the loadings, noise scale and degrees of freedom are saved with the networks and compared after resume |
| Diffusion | session | I+T | I+T | exact | I+T | I+T | I+T | I+T | I+T | the noise schedule is in the fingerprint; **this model keeps no EMA copy of its weights** |
| Deep hedging | session | I+T | I+T | exact | I+T (certainty equivalent) | I+T with clipping | I+T | I+T | I+T | cosine scheduler state restored |
| PINN (BSM) | session | I+T | I+T (`pinn`) | exact | I+T | I+T | I+T | I+T | I+T | the Adam phase is checkpointed; the L-BFGS polish re-runs deterministically from the last checkpoint (**a kill inside the polish was not tested**). Vasicek and Heston PINNs use the same `_train`, **not run individually** |
| DQN, Double DQN | session | I+T | I+T (`rl_dqn`) | exact (replay buffer, exploration step and environment state restored) | I+T (reward, epsilon, replay size) | I+T | I+T | I+T | I+T | `double_dqn` shares the code, **not run separately** |
| DDPG, TD3 | session, two optimizers | I+T | I+T | exact | I+T (critic and actor loss, reward) | I+T per optimizer | I+T | I+T | I+T | |
| SAC | session, three optimizers | I+T | I+T | exact | I+T (temperature) | I+T per optimizer | I+T | I+T | I+T | |
| PPO (`rl/deep.py`) | session | I+T | I+T | exact | I+T | I+T | I+T | I+T | I+T | a unit is one rollout; the policy standard deviation is saved |
| Stable-Baselines3 PPO/SAC/RecurrentPPO (`ml/rl/agent.py`) | the library's own loop | **U** | **U** | none | P (existing live telemetry only) | U | U | U | P | the library is not installed here and owns its loop; nothing was written or tested for it |
| Chronos and TimesFM zero-shot | none | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | pretrained, inference only |
| Linear, tree and boosting models | single fit | N/A | N/A | restart | N/A | N/A | N/A | N/A | N/A | fits take seconds |

Environment state: the reinforcement-learning environment (`RegimeFractionEnv`) now exposes `get_state()`/`set_state()` (generator, regime, clock), which is what makes the RL guarantee exact. An environment without them would cap the level at `epoch_boundary`, and the session says so (`ceiling`).

## 4. Gradient diagnostics

`GradientMonitor` is read-only: it never changes a gradient, a parameter, a hyperparameter or the control flow of the loop, and a warning never stops or alters training (`tests/test_training_diagnostics_integration.py` shows the weights, losses and histories are identical with the monitor at every level).

**Levels.** `off`; `lightweight` (global L2 norm of the unclipped gradient, non-finite detection, clipping frequency, count of parameters with no gradient); `standard` (about four sampled optimizer steps per epoch: mean, median, RMS, standard deviation, signed minimum and maximum, L-infinity norm, P50/P90/P95/P99 of |g|, zero and near-zero fractions, and per layer: L2, RMS, mean and maximum |g|, near-zero fraction, parameter norm, gradient-to-weight ratio, the update-to-weight ratio the optimizer actually applied, non-finite counts); `detailed` (about eight samples, plus per-layer medians and a histogram of log10 |g| with exact zeros and non-finite values counted apart). `auto`: lightweight on Replit, standard elsewhere, off when there is no run folder. The chosen profile is recorded in `diagnostics.json`, in the `start` row of `metrics.jsonl` and in the run's request.

**Measurement timing** (each verified in `tests/test_gradient_diagnostics.py` against numbers worked out independently): the completed gradient after all accumulation steps; after `GradScaler.unscale_` (a scaled gradient is read back as the true one); before clipping, with the post-clip norm measured separately on sampled steps and clipping frequency counted on every observed step; the update measured around the real `optimizer.step()` (a step skipped by the scaler moves nothing and is counted as skipped); `None` gradients counted apart from exact zeros; frozen parameters excluded from every denominator; sparse gradients measured on their stored values with the implicit zeros counted; non-finite entries counted and excluded from the other statistics (never replaced); one monitor per optimizer.

**Warnings** (`gradient_event` rows, with model, epoch, step, layers, measured values, thresholds, severity, learning rate and what the training did, which is always "continued" or "skipped step" by the scaler): non-finite gradient (immediate, critical); gradient spike (information only: one large step is never called an explosion); exploding gradient (the median norm stays far above where training began, or an absolute ceiling, for several epochs in a row); repeated clipping (more than half the steps clipped for several epochs); vanishing gradient (a layer's RMS below an absolute floor AND a thousandth of the strongest layer's, for several epochs); gradient decay with depth; dead layer (exactly zero gradient at every sampled step; a layer with sparse but non-zero gradients is not dead; frozen layers are never named); no parameter update; large update-to-weight ratio (annotated when the learning rate changed in the unit); unstable optimization (coefficient of variation of the step norms, annotated with learning-rate changes); loss spike; potential overfitting (training loss falling while validation loss rises for three epochs); distribution shift. Thresholds live in `gradients.THRESHOLDS`, with overrides per model family (transformer, recurrent, GAN, RL), and every event carries the thresholds it used. **The thresholds are defensible defaults, not laws: they were exercised on synthetic gradients with known behaviour, not tuned on real training failures.**

## 5. The Training monitor

`GET /api/training/runs|run|gradients|profiles`, `POST /api/training/control|resume|archive|preflight|prune`. Five views (Machine learning workspace, Training monitor, ECharts via the existing chart kit; every chart has a Table view):

* **Overview**: training and validation loss with the best epoch marked, learning rate, memory, CPU and GPU, model-specific curves (reconstruction, KL, critic gap, reward ...), checkpoints.
* **Gradient health**: global norm (median, P90, largest; log scale; warnings marked), percentiles of |g|, zero and near-zero fractions, clipping and non-finite steps.
* **Layer diagnostics**: the gradient-flow chart in depth order (mean, median, maximum |g|, or gradient-to-weight, or update-to-weight; log scale), the layers with zero, missing, non-finite or negligible gradients listed instead of being hidden, and a comparison of chosen layers over epochs.
* **Distributions**: histograms of up to three chosen epochs.
* **Recovery & events**: recovery status and level, a timeline of checkpoints, interruptions, resumes, stops and warnings, the gradient warnings with their evidence, resource warnings.

Everything is read from files (`metrics.jsonl`, `gradients.jsonl`, checkpoints), so a browser refresh, a second browser or a restarted app shows the same history (`tests/test_webapp_training_browser.py`). A run with diagnostics off says so instead of drawing empty charts.

## 6. Results

### 6.1 Fresh-process interruption and recovery

`python -m pytest tests/test_fresh_process_recovery.py` (28 scenarios, 3 processes each; recorded in `docs/generated/fresh_process_results.jsonl`). Besides the numbers, every scenario checks that the resumed run's metrics file has one row per unit in order, the gradient file likewise, the resume row says `exact`, and the run ends with a `finish` row.

| Scenario | Killed after unit | Resumed from unit | Units | Recovery level | Max abs difference vs uninterrupted |
|---|---|---|---|---|---|
| `ae` | 3 | 3 | 8 | exact | 0.0e+00 |
| `deep_hedging` | 3 | 3 | 10 | exact | 0.0e+00 |
| `diffusion` | 3 | 3 | 10 | exact | 0.0e+00 |
| `dlinear` | 3 | 3 | 8 | exact | 0.0e+00 |
| `encoder_ae` | 3 | 3 | 10 | exact | 0.0e+00 |
| `encoder_contrastive` | 3 | 3 | 10 | exact | 0.0e+00 |
| `factor_vae` | 3 | 3 | 10 | exact | 0.0e+00 |
| `forecast_patchtst` | 3 | 3 | 10 | exact | 0.0e+00 |
| `forecast_tsmixer` | 3 | 3 | 10 | exact | 0.0e+00 |
| `gat` | 3 | 3 | 8 | exact | 0.0e+00 |
| `gcn` | 3 | 3 | 8 | exact | 0.0e+00 |
| `graph_forecast` | 3 | 3 | 10 | exact | 0.0e+00 |
| `gru` | 3 | 3 | 8 | exact | 0.0e+00 |
| `itransformer` | 3 | 3 | 8 | exact | 0.0e+00 |
| `lstm` | 3 | 3 | 8 | exact | 0.0e+00 |
| `nbeats` | 3 | 3 | 8 | exact | 0.0e+00 |
| `nhits` | 3 | 3 | 8 | exact | 0.0e+00 |
| `patchtst` | 3 | 3 | 8 | exact | 0.0e+00 |
| `pinn` | 3 | 3 | 5 | exact | 0.0e+00 |
| `rl_ddpg` | 3 | 3 | 8 | exact | 0.0e+00 |
| `rl_dqn` | 3 | 3 | 8 | exact | 0.0e+00 |
| `rl_ppo` | 2 | 2 | 6 | exact | 0.0e+00 |
| `rl_sac` | 3 | 3 | 8 | exact | 0.0e+00 |
| `rl_td3` | 3 | 3 | 8 | exact | 0.0e+00 |
| `tcn` | 3 | 3 | 8 | exact | 0.0e+00 |
| `tft` | 3 | 3 | 8 | exact | 0.0e+00 |
| `vae` | 3 | 3 | 8 | exact | 0.0e+00 |
| `wgan_gp` | 3 | 3 | 10 | exact | 0.0e+00 |

Also tested: a corrupted newest checkpoint falls back to the previous one; an interrupted write never damages the only good checkpoint; a changed model or data window is refused for the whole job; a crash between checkpoint and log is repaired (`tests/test_training_recovery.py`, `tests/test_training_diagnostics_integration.py`); a whole **application** killed mid-job, a new application started on the same folder shows the run as orphaned and the user resumes it through the API (`tests/test_worker_crash_recovery.py`); memory exhaustion stops a run with a valid checkpoint and the resumed run matches the uninterrupted one bit for bit (`tests/test_resource_constrained_training.py`).

### 6.2 Monitoring overhead

`python scripts/benchmark_gradient_monitoring.py --threads 1` (one thread, CPU, 20,000 training rows, batch 128 so 156 optimizer steps per epoch, gradient clipping on, checkpoints on at every level, median of three fresh processes per cell; `docs/generated/gradient_monitoring_benchmark.json`).

| Network | Parameters | Level | Epoch (s) | Overhead vs off | Extra peak memory (MB) | Gradient log (kB) |
|---|---|---|---|---|---|---|
| small | 40,210 | off | 0.413 | +0.0% | +0.0 | 0.0 |
| small | 40,210 | lightweight | 0.416 | +0.9% | +1.4 | 2.8 |
| small | 40,210 | standard | 0.437 | +5.9% | +2.5 | 14.8 |
| small | 40,210 | detailed | 0.500 | +21.0% | +3.8 | 17.1 |
| medium | 257,547 | off | 1.038 | +0.0% | +0.0 | 0.0 |
| medium | 257,547 | lightweight | 1.039 | +0.1% | +1.9 | 2.9 |
| medium | 257,547 | standard | 1.107 | +6.6% | +2.3 | 25.2 |
| medium | 257,547 | detailed | 1.094 | +5.4% | +3.7 | 28.2 |
| large | 2,022,756 | off | 4.937 | +0.0% | +0.0 | 0.0 |
| large | 2,022,756 | lightweight | 5.086 | +3.0% | +3.5 | 2.9 |
| large | 2,022,756 | standard | 5.213 | +5.6% | +8.8 | 38.5 |
| large | 2,022,756 | detailed | 5.463 | +10.7% | +9.4 | 42.3 |

The full-sample cost is paid a fixed number of times per epoch, so the relative overhead falls as epochs get longer and rises when they are very short: in an earlier run with only 47 steps per epoch, before the sampling counts and the percentile subsample were reduced, standard cost 52 to 83 percent. Lightweight is practical anywhere (3 percent or less here); standard stays near 6 percent; detailed is for investigations. Memory: at most about 9 MB extra, because nothing keeps a tensor. These are measurements from one machine and one run, not a guarantee.

## 7. Windows and Replit

Replit: monitoring defaults to lightweight; memory pressure above 85 percent of the container limit is reported once every five epochs as a warning with the numbers; `ResourceLimit` stops a run between batches with the last checkpoint intact; the monitor shows the advice (smaller batch with accumulation, restart or warm start); `data/user/training` retention is `POST /api/training/prune` plus automatic removal of the fallback checkpoints of a run that ended normally. The Replit-specific behaviour was tested by switching `resources.on_replit` and by simulating the container's memory limit, **not on a Replit machine**. Windows: the checkpoint replace retries on sharing violations (simulated); the fresh-process tests use `SIGKILL`, which Windows lacks, so **they were not run on Windows**.

## 8. What is still not done

* Stable-Baselines3 agents (`ml/rl/agent.py`): unsupported, see the matrix.
* Exactness on a GPU, across library versions or with non-deterministic kernels is not claimed; the saved CUDA generator states and mixed-precision scaler are restored in code but **no GPU was available to test them**. Mixed-precision unscaling was verified on CPU with `GradScaler("cpu")`.
* A kill inside the PINN's L-BFGS polish, the Vasicek and Heston PINNs, Double DQN as a separate run and the other forecast-network kinds were not run in a fresh process.
* Gradient thresholds were not tuned on real failures. The update-to-weight ratio is measured on sampled steps only.
* Microbatch gradients inside an accumulation are not stored separately: the diagnostics describe the completed accumulated gradient by design.
* No server-sent events: the page polls every two seconds.
