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
| Forecast networks (`models/deep_forecast`): PatchTST, TSMixer, N-BEATS, N-HiTS, TimeMixer, TFT, transformer, FNN, CNN, LSTM, GRU | session | I+T | I+T (every kind, `forecast_<kind>`) | exact | I+T | I+T | I+T | I+T | I+T | second upgrade: each of the 11 architectures is killed and resumed in its own fresh process (section 6.3) |
| Graph-attention forecaster (`models/graph_forecast`) | session | I+T | I+T (`graph_forecast`) | exact | I+T | I+T | I+T | I+T | I+T | |
| Encoder: autoencoder, contrastive (`models/representation`) | session | I+T | I+T | exact | I+T | I+T | I+T | I+T | I+T | |
| WGAN-GP | session, two optimizers | I+T | I+T | exact | I+T (critic gap, penalty) | I+T per optimizer | I+T | I+T | I+T | generator and critic are monitored separately; a unit is 25 generator updates |
| Factor VAE | session | I+T | I+T | exact | I+T (reconstruction log-likelihood, KL, beta, t degrees of freedom) | I+T | I+T | I+T | I+T | the loadings, noise scale and degrees of freedom are saved with the networks and compared after resume |
| Diffusion | session | I+T | I+T | exact | I+T | I+T | I+T | I+T | I+T | the noise schedule is in the fingerprint; **this model keeps no EMA copy of its weights** |
| Deep hedging | session | I+T | I+T | exact | I+T (certainty equivalent) | I+T with clipping | I+T | I+T | I+T | cosine scheduler state restored |
| PINN: BSM, Vasicek, Heston, Bates (`models/pinn._train`) | session, two optimizers (Adam, L-BFGS) | I+T (Adam block, then each `LBFGS_CHUNK` iterations) | I+T (12 kill scenarios, section 6.4) | exact | I+T (phase-aware: no epochs shown) | I+T per optimizer | I+T | I+T | I+T | second upgrade: a kill during Adam, just before the switch, inside an L-BFGS closure, just after a finished L-BFGS step and after the polish is tested in new processes; each model is held to its pricing-accuracy tolerance against an independent reference |
| DQN | session | I+T | I+T (`rl_dqn`) | exact (replay buffer, exploration step and environment state restored) | I+T (reward, epsilon, replay size, updates, target syncs) | I+T | I+T | I+T | I+T | |
| Double DQN | session | I+T | I+T (`rl_double_dqn`, own test file) | exact | I+T | I+T | I+T | I+T | I+T | second upgrade: online and target networks, optimizer, replay buffer, epsilon, environment steps, update count, target-sync count, RNG and environment state are compared field by field after a kill in a new process |
| DDPG, TD3 | session, two optimizers | I+T | I+T | exact | I+T (critic and actor loss, reward) | I+T per optimizer | I+T | I+T | I+T | |
| SAC | session, three optimizers | I+T | I+T | exact | I+T (temperature) | I+T per optimizer | I+T | I+T | I+T | |
| PPO (`rl/deep.py`) | session | I+T | I+T | exact | I+T | I+T | I+T | I+T | I+T | a unit is one rollout; the policy standard deviation is saved |
| Stable-Baselines3 PPO, SAC, RecurrentPPO (`ml/rl/agent.py`, `ml/rl/sb3_state.py`) | the library's own `learn`, run in chunks | I+T (isolated environment) | I+T (isolated environment) | exact (single environment, CPU, constant learning-rate and clip schedules); approximate after a fallback to the previous checkpoint | I+T (timesteps, episodes, reward) | P: policy or actor/critic gradients from the public `.grad` | P | P | I+T | dependency-blocked in the main environment (not installed); see section 6.5 for what is and is not claimed |
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

### 6.3 Every forecast-network architecture, one fresh process each

`python -m pytest tests/test_fresh_process_forecast.py` (11 kinds, plus one scenario for a damaged and a missing checkpoint). Each kind is killed after unit 3 of 10 and resumed in a new process; the test also checks that the gradient log has one row per epoch with no gap, and that the monitor's run detail (status, epoch series, `resume` event at level `exact`, last training loss) and the gradient API (one optimizer, one row per epoch, layer flow) describe the resumed run.

| Kind | Killed after | Resumed from | Level | Max abs diff validation loss | Max abs diff outputs |
|---|---|---|---|---|---|
| cnn | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |
| fnn | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |
| gru | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |
| lstm | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |
| nbeats | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |
| nhits | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |
| patchtst | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |
| tft | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |
| timemixer | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |
| transformer | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |
| tsmixer | 3 | 3 | exact | 0.0e+00 | 0.0e+00 |

### 6.3b Double DQN

`python -m pytest tests/test_fresh_process_double_dqn.py`: killed after unit 4 of 8, resumed in a new process at level `exact`. Compared with the uninterrupted run: online and target network weights, optimizer state, replay buffer contents and write position, epsilon and exploration step, environment steps, update count (1901), target-sync count (38), RNG states and environment state (max abs difference of the Q-network parameters 0.0e+00). The resumed run never shares an in-memory object with the reference. A corrupted newest checkpoint falls back to the previous one and is reported as `epoch_boundary` (approximate, RNG not restored); with no checkpoint a resume trains from the start; a changed environment is refused.

### 6.4 PINN recovery in new processes, including the L-BFGS polish

`python -m pytest tests/test_fresh_process_pinn.py` (12 kill scenarios plus one for a damaged and a missing checkpoint; 3 processes each: uninterrupted reference, killed worker, resuming worker; recorded in `docs/generated/fresh_process_round2_results.jsonl`). The worker dies by `SIGKILL` (a watcher thread, or a wrapper of `torch.optim.LBFGS.step` for the closure case); each scenario then checks that the resumed run's metrics file has one row per unit in order, a `resume` row at level `exact`, a `recomputed_work` event when a unit had begun and not finished, and a `finish` row. The recovered network is also held to the accuracy test of its model **without loosening any tolerance**: BSM against the closed form (0.7), Vasicek against the analytic bond price (0.005), Heston and Bates against the Fourier price (1.5).

Units: Adam blocks of `report` steps, then one L-BFGS `step()` of `LBFGS_CHUNK` (10) iterations each. A kill inside a step loses at most that chunk; the step is redone from the checkpoint taken at the previous boundary (L-BFGS history, `d`, `t`, `old_dirs`, `old_stps`, `H_diag`, `prev_flat_grad`, `prev_loss` and the iteration and function-evaluation counters are in the optimizer state), and the redone work is recorded (`recomputed_work`).

| Model | Kill point | Killed after unit | Units | Level | Recomputed work recorded | Max abs diff loss / prices / parameters | Pricing error | Tolerance |
|---|---|---|---|---|---|---|---|---|
| bsm | during Adam | 3 | 36 | exact | yes | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.4488 | 0.7 |
| bsm | immediately before the L-BFGS switch | 6 | 36 | exact | yes | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.4488 | 0.7 |
| bsm | inside an L-BFGS closure (line search running) | 9 | 36 | exact | yes | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.4488 | 0.7 |
| bsm | just after a completed L-BFGS step | 8 | 36 | exact | yes | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.4488 | 0.7 |
| bsm | after the polish finished, before the run was closed | 36 | 36 | exact | no (nothing in flight) | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.4488 | 0.7 |
| vasicek | during Adam | 4 | 18 | exact | yes | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0016 | 0.005 |
| vasicek | inside an L-BFGS closure (line search running) | 10 | 18 | exact | yes | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0016 | 0.005 |
| vasicek | after the polish finished, before the run was closed | 18 | 18 | exact | no (nothing in flight) | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.0016 | 0.005 |
| heston | during Adam | 2 | 20 | exact | yes | 0.0e+00 / 0.0e+00 / 0.0e+00 | 1.2291 | 1.5 |
| heston | inside an L-BFGS closure (line search running) | 7 | 20 | exact | yes | 0.0e+00 / 0.0e+00 / 0.0e+00 | 1.2291 | 1.5 |
| heston | after the polish finished, before the run was closed | 20 | 20 | exact | no (nothing in flight) | 0.0e+00 / 0.0e+00 / 0.0e+00 | 1.2291 | 1.5 |
| bates | inside an L-BFGS closure (line search running) | 7 | 20 | exact | yes | 0.0e+00 / 0.0e+00 / 0.0e+00 | 0.7990 | 1.5 |

**Equivalence with the previous code.** The polish used to be one `step()` call; it now runs in chunks, with `max_iter` and `max_eval` of each call set from what the whole polish has consumed so far. Compared in separate processes against the code before this upgrade (a git worktree of the previous commit), for BSM, Vasicek and Heston, with `LBFGS_CHUNK` 3 and 10: the prices and the loss history are **identical** (difference 0.0), so the mathematics did not change. This comparison was a one-off script, not a test in the suite; the suite compares each resumed run with an uninterrupted run of the new code.

Also tested: with the newest polish checkpoint damaged, the previous one is used and the level is reported as `epoch_boundary` (not `exact`); with no checkpoint a resume starts from the beginning and says nothing was restored (`test_a_damaged_newest_checkpoint_inside_the_polish_falls_back_and_a_missing_one_restarts`). Gradients: the Adam and L-BFGS optimizers are monitored separately, the residual-loss gradient is checked against an independent computation, and a parameter that receives no gradient from the residual (the last bias) is reported as missing, not as zero (`tests/test_gradient_diagnostics_families.py`).

### 6.5 Stable-Baselines3 agents (isolated environment)

Stable-Baselines3 owns its training loop and is not installed in the main environment, so the agents there are **dependency-blocked**: the SB3 tests skip. `scripts/make_sb3_env.sh` builds a venv that sees the main packages and adds `stable-baselines3` 2.9.2, `sb3-contrib` 2.9.2 and `gymnasium` 1.4.0 (PyTorch 2.15.0, CPU); the tests find it through `$QUANT_SB3_PYTHON` or `/tmp/sb3env/bin/python`.

**How it works.** Inside a training run, `RLAgent._learn` calls `learn(chunk, reset_num_timesteps=False)` in chunks of `checkpoint_timesteps` (aligned to `n_steps` so a PPO rollout is never split); at each boundary it writes one checkpoint through the shared `TrainingSession`: the library's own `save` (policy, optimizer states, counters), `save_replay_buffer` for SAC, and an application manifest (`ml/rl/sb3_state.py`) for what SB3 does not save: NumPy and PyTorch RNG states (SB3's `load` re-seeds), the environment (pickled), the last observation and episode-start flags (`load` leaves them unset), the recurrent state of RecurrentPPO, counters. Without a run (a direct library call) `learn` is called once, unchanged. Timesteps, episodes and reward are the units the monitor shows (no epochs).

**Equivalence.** Chunked training equals one plain `learn` call (tests `test_chunked_checkpointed_training_equals_a_plain_learn_call`, per algorithm) for constant learning-rate and clip schedules.

| Agent | Killed after unit | Resumed from | Units | Level | Max abs diff policy | Max abs diff rewards | Timesteps | Replay buffer saved | Recurrent state saved |
|---|---|---|---|---|---|---|---|---|---|
| ppo | 3 | 3 | 10 | exact | 0.0e+00 | 0.0e+00 | 1280 | n/a | n/a |
| recurrent_ppo | 3 | 3 | 10 | exact | 0.0e+00 | 0.0e+00 | 640 | n/a | yes |
| sac | 3 | 3 | 10 | exact | 0.0e+00 | 0.0e+00 | 1000 | yes | n/a |

**Recovery guarantees, stated apart.**

* *Exact* (verified above): single environment, CPU, constant learning rate and clip range, same library versions; policy, optimizer, replay buffer, RNG, environment and recurrent state all restored.
* *Approximate* (verified: `test_a_damaged_newest_checkpoint_is_reported_as_approximate_and_another_configuration_is_refused`): the newest checkpoint is damaged and the previous generation is used; training continues but the RNG state is not restored, so the curve will not match the uninterrupted run.
* *Refused*: a different algorithm, policy, hyperparameters or data window is rejected (`CheckpointIncompatible`), never resumed.
* *Restart*: no readable checkpoint; training starts at timestep 0 (`test_with_no_checkpoint_a_resume_trains_from_the_start`).
* *Not claimed*: GPU runs, vectorised environments, `VecNormalize`, learning-rate or clip schedules that depend on the remaining progress, other SB3 versions. The manifest reads three private attributes of SB3 (`_last_obs`, `_last_episode_starts`, `_last_lstm_states`) because `load` leaves them unset; `check_counters` compares the timestep, update and episode counters of the rebuilt agent with the checkpoint's after every restore and writes a `recovery_note` event for each difference (it warns; it does not stop training), so a library change shows up in the monitor's events, not as a silent drift.
* *Gradient diagnostics*: through public parameters only: the policy network (PPO) or actor and critic (SAC) `.grad` read between updates. PPO's are after SB3's own clipping, and the update-to-weight ratio is not measured: marked limited.

### 6.6 Real process termination, corruption and missing checkpoints

All fresh-process scenarios use a real `SIGKILL` of the training process (`tests/fresh_process_scenarios.die()` and `was_killed()`, which accept the Windows termination code instead of `-9`; **not run on Windows**), never an exception raised in-process. A kill after the last unit but before the run is closed (PINN `after_polish`) and inside an L-BFGS closure leave logs that the resume run completes with one row per unit; the crash between a checkpoint and its log row is repaired by `reconcile_logs` (`tests/test_training_recovery.py`). Corrupted-newest-checkpoint and missing-checkpoint behaviour is tested for PINN, Double DQN, SB3 and the forecast networks, as in 6.1 for the others.

### 6.7 Training monitor and job registry after recovery

A run's unit is stored with its rows, so the monitor shows epochs for the feature networks, optimizer-phase blocks for PINNs (distinct `adam` and `lbfgs` phases, a Phase tile, a phase-aware loss chart, no epoch label for L-BFGS steps) and timesteps with episodes and reward for the Stable-Baselines3 agents and the other RL loops. The listing carries `unit`, `phase`, `kind` and `ceiling`; events include `recomputed_work`, `diagnostics_note` and `recovery_limits`. Browser tests: `tests/test_webapp_training_recovery_browser.py`, with the existing `tests/test_webapp_training_browser.py` unchanged and passing.

## 7. Windows and Replit

Replit: monitoring defaults to lightweight; memory pressure above 85 percent of the container limit is reported once every five epochs as a warning with the numbers; `ResourceLimit` stops a run between batches with the last checkpoint intact; the monitor shows the advice (smaller batch with accumulation, restart or warm start); `data/user/training` retention is `POST /api/training/prune` plus automatic removal of the fallback checkpoints of a run that ended normally. The Replit-specific behaviour was tested by switching `resources.on_replit` and by simulating the container's memory limit, **not on a Replit machine**. Windows: the checkpoint replace retries on sharing violations (simulated); the fresh-process tests use `SIGKILL`, which Windows lacks, so **they were not run on Windows**.

## 8. What is still not done

* Stable-Baselines3 agents: integrated and fresh-process tested, but only in an isolated environment (`scripts/make_sb3_env.sh`); in the main environment they are dependency-blocked and their tests skip. See 6.5 for the limits (single environment, CPU, constant schedules, no `VecNormalize`).
* Exactness on a GPU, across library versions or with non-deterministic kernels is not claimed; the saved CUDA generator states and mixed-precision scaler are restored in code but **no GPU was available to test them**. Mixed-precision unscaling was verified on CPU with `GradScaler("cpu")`.
* Closed in the second upgrade: a kill inside the PINN's L-BFGS polish, the Vasicek, Heston and Bates PINNs, Double DQN as a separate run and every forecast-network kind are now fresh-process tested (6.3 to 6.5). What remains for them is the GPU and Windows caveat above.
* Gradient thresholds were not tuned on real failures. The update-to-weight ratio is measured on sampled steps only.
* Microbatch gradients inside an accumulation are not stored separately: the diagnostics describe the completed accumulated gradient by design.
* The gradient diagnostics of the Stable-Baselines3 agents are limited: PPO gradients are read after the library's own clipping, and the update-to-weight ratio is not measured, because SB3 offers no public hook between its optimizer step and the weights. These are read through public parameters only; no private SB3 internals are patched.
* No server-sent events: the page polls every two seconds.

## 9. Completion matrix (second upgrade)

Statuses, used and nothing else: **Implemented and fresh-process tested**, **Implemented but fresh-process unverified**, **Dependency-blocked**, **Partially implemented**, **Unsupported**, **Not applicable**. "Fresh-process tested" means SIGKILL of the training process and recovery in a new process, compared with an uninterrupted run. Tolerances: weights, losses and outputs equal to 1e-6 where stated "exact"; in every recorded scenario the difference was 0.0. Results are in `docs/generated/fresh_process_results.jsonl` (first upgrade, 28 scenarios) and `docs/generated/fresh_process_round2_results.jsonl` (this upgrade, 27 scenarios). The SB3 rows come from the isolated environment of `scripts/make_sb3_env.sh`.

| Model family | Source (file: function) | Checkpoint contents | Checkpointing | Fresh-process resume | Gradient diagnostics | Training monitor | Guarantee | Tests | Limitations |
|---|---|---|---|---|---|---|---|---|---|
| PINN Black-Scholes-Merton (Adam + L-BFGS) | `models/pinn.py`: `_train` | both optimizers (L-BFGS history and counters), networks, RNG, loss history, polish progress, collocation seed | Implemented and fresh-process tested | Implemented and fresh-process tested (5 kill points) | Implemented and fresh-process tested (Adam and L-BFGS separately) | Implemented and fresh-process tested (phases, no epochs) | exact | `test_fresh_process_pinn.py`, `test_gradient_diagnostics_families.py` | a kill inside an L-BFGS chunk redoes at most that chunk (recorded as `recomputed_work`); CPU only |
| PINN Vasicek | same | same | Implemented and fresh-process tested | Implemented and fresh-process tested (3 kill points) | Implemented and fresh-process tested | Implemented and fresh-process tested | exact | same; pricing within 0.005 of the analytic bond price (error 0.0016) | as above |
| PINN Heston | same | same | Implemented and fresh-process tested | Implemented and fresh-process tested (3 kill points) | Implemented and fresh-process tested | Implemented and fresh-process tested | exact | same; within 1.5 of the Fourier price (error 1.23) | as above |
| PINN Bates | same | same | Implemented and fresh-process tested | Implemented and fresh-process tested (1 kill point, inside the polish) | Implemented and fresh-process tested | Implemented and fresh-process tested | exact | same; within 1.5 (error 0.80) | fewer kill points than the others |
| Double DQN | `rl/deep.py`: DQN trainer with `double=True` | online and target networks, Adam, replay buffer, epsilon, env steps, update and sync counts, RNG, environment state | Implemented and fresh-process tested | Implemented and fresh-process tested | Implemented and fresh-process tested | Implemented and fresh-process tested | exact; approximate after a fallback | `test_fresh_process_double_dqn.py` | single environment |
| DQN, DDPG, TD3, SAC, PPO (`rl/deep.py`) | `rl/deep.py` | as above, plus actors, critics, temperature, policy std | Implemented and fresh-process tested | Implemented and fresh-process tested | Implemented and fresh-process tested | Implemented and fresh-process tested | exact | `test_fresh_process_recovery.py` | |
| Forecast networks (11 kinds) | `models/deep_forecast.py`: `deep_family`/`deep_sequence` training loop | network, Adam, scheduler, RNG, history, best weights | Implemented and fresh-process tested | Implemented and fresh-process tested, one process per kind | Implemented and fresh-process tested | Implemented and fresh-process tested | exact | `test_fresh_process_forecast.py` | |
| Stable-Baselines3 PPO | `ml/rl/agent.py`: `RLAgent._learn`; `ml/rl/sb3_state.py` | `save()` zip, manifest (RNG, env, last observation, counters) | Implemented and fresh-process tested (isolated env); Dependency-blocked in the main environment | Implemented and fresh-process tested (isolated env) | Partially implemented (post-clip gradients from public `.grad`; no update-to-weight ratio) | Implemented and fresh-process tested (timesteps, episodes, reward) | exact; approximate after a fallback; refused if configuration differs; restart if none | `test_fresh_process_sb3.py` | single environment, CPU, constant schedules, no `VecNormalize`, three private SB3 attributes read |
| Stable-Baselines3 SAC | same | plus the replay buffer (`save_replay_buffer`) | same | same | Partially implemented (actor and critic) | same | same | same | same |
| Stable-Baselines3 RecurrentPPO (`sb3-contrib`) | same | plus the LSTM state | same | same | Partially implemented | same | same | same | same |
| Feature networks, autoencoders, VAEs, graph networks, GANs, diffusion, deep hedging, encoders, graph forecaster | as in section 3 | as in section 2 | Implemented and fresh-process tested | Implemented and fresh-process tested | Implemented and fresh-process tested | Implemented and fresh-process tested | exact | `test_fresh_process_recovery.py` | unchanged by this upgrade |
| Chronos, TimesFM zero-shot | `strategies/deep.py` | none (pretrained) | Not applicable | Not applicable | Not applicable | Not applicable | not applicable | | inference only |
| Linear, tree, boosting | single fit | none | Not applicable | Not applicable | Not applicable | Not applicable | restart | | fits take seconds |
| Anything on a GPU, vectorised SB3 environments | | | Implemented but fresh-process unverified (GPU RNG states are saved; no GPU available) / Unsupported (vectorised SB3) | | | | | | not claimed |

### 9.1 Gaps of this upgrade: before and after

| Gap | Before | After |
|---|---|---|
| Stable-Baselines3 agents | Unsupported (dependency unavailable, library owns the loop) | integrated through chunked `learn` plus a manifest; fresh-process tested in an isolated environment; Dependency-blocked in the main environment |
| Kill during PINN L-BFGS polish | polish re-ran from the last Adam checkpoint, untested | boundary checkpoints every `LBFGS_CHUNK` iterations, interrupted step detected and recorded, tested at five points |
| Vasicek and Heston PINNs | shared code, not run individually | individually tested against analytic and Fourier references; Bates added |
| Double DQN | shared code, not run separately | dedicated scenario with a field-by-field state comparison |
| Forecast architectures | 2 of 11 tested | all 11, one process each |
| Gradient diagnostics | none for PINN phases, SB3 | per-optimizer for PINN phases; SB3 limited (public hooks only) |
| Monitor | epochs only | timesteps and phases for the new loops, `recomputed_work` events |

### 9.2 Regression results

Main environment (Python 3.11, PyTorch 2.15.0 CPU): the ML, training, deep-learning, web-app and algorithm-catalogue suites ran with `-n 4`; the only failures were `tests/test_training_api.py` `test_finished_runs_are_trimmed_and_old_ones_pruned_but_a_resumable_one_never` and `test_archive_removes_a_run_from_the_list_without_deleting_its_files` (they depend on a previous test's run folder and fail under `-n 4` on the code from before this upgrade as well; both pass when the file is run serially) and one kill-timing race in `tests/test_worker_crash_recovery.py` (the process can be killed between an epoch row and the gradient warning written right after it; the test now ignores warning rows when it looks for the last epoch). The SB3-environment suites of `test_ml_rl*.py`, `test_ml_param_schema.py` and the fresh-process SB3 tests passed. Browser suites: training, training recovery, studio, workspaces and main ran; two `test_webapp_browser.py` tests failed once because the source files were changed under the running server (it shows a restart banner), and pass when run alone.

