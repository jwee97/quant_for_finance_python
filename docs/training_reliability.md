# Training reliability: checkpoints, resume, the Training monitor, resource profiles

This page says what the deep-network training of the Machine learning workspace can and cannot recover from, what the Training monitor shows, and which models are covered. Every claim below names the test that backs it; anything without one is listed under *Not implemented or not verified*.

## 1. What a checkpoint holds and how it is written

`src/ml/deep/checkpoint.py`, used by `fit_network` (`src/ml/deep/trainer.py`). After each epoch (every `checkpoint_every` epochs, always on improvement, on the last epoch and on a stop) the loop commits one file `last.pt` containing:

| Part | Why |
|---|---|
| model weights, optimizer state, scheduler state, AMP scaler state | the exact continuation of the optimisation |
| RNG states: PyTorch (CPU and every CUDA device), NumPy global, the shuffling `Generator`'s bit-generator state | the batches and dropout masks after the resume are the ones the uninterrupted run would have drawn |
| metric history (loss, validation loss, learning rate, gradient norm, epoch seconds), best score and epoch, early-stopping counter, events | the curves and the stopping decision continue instead of restarting |
| `config_fp` (architecture and hyperparameters, minus execution settings), `data_fp` (the fit's training window), schema version, software versions, device | a checkpoint is refused when it does not belong to this model or this data |

Write protocol: serialise to memory, write to `last.next.pt` in the same folder, flush and `fsync`, read it back and verify the SHA-256 and the epoch, then `os.replace` it over `last.pt` (the previous `last.pt` becomes `last.prev.pt` first). `best.pt` (the weights early stopping will return) and optional `epoch-<n>.pt` milestones (`keep_milestones`) are written the same way. `max_checkpoint_bytes` removes the oldest milestones first. A checkpoint write that fails (full disk) is logged as an event and training continues; it never ends the run.

Reading: `last.pt`, else `last.prev.pt`, else `best.pt` (weights only). A file whose checksum or contents do not verify is skipped and named in the result.

`checkpoint_every`, `keep_milestones`, `accumulation_steps`, `amp` and `profile` are execution settings: they are not part of `config_fp`, so changing them never makes a run incompatible with its own checkpoint. `epochs` is also an execution setting (extend or shorten a run when resuming). **`patience` is not**: the learning-rate scheduler's patience derives from it, so changing it changes the trajectory and the resume is refused.

## 2. Recovery levels, as reported

| Level | Meaning |
|---|---|
| `exact` | model, optimizer, scheduler, every RNG, history and early-stopping state restored on the same library versions and device: the continued run follows the uninterrupted one. Verified on CPU against a run that was never interrupted (weights equal to 1e-6, losses to 1e-5). Not claimed across devices, library versions or non-deterministic GPU kernels; then the level drops to `epoch_boundary` with the reason. |
| `epoch_boundary` | continues from the last finished epoch, but something needed for bit-for-bit continuation differs or is missing (device or version change, damaged newest file so `last.prev.pt` was used, no RNG state). |
| `warm_start` | only weights restored (only `best.pt` readable, or the user chose a warm start because the data window changed). Optimizer, scheduler and counters start fresh; the curve will not match an uninterrupted run. |
| `restart_required` | no valid checkpoint. |

A checkpoint whose `config_fp` or `data_fp` differs is **refused** (`CheckpointIncompatible`) unless a warm start was requested. The refusal fails the whole job with the reasons; a walk-forward never skips the fit and carries on (that exception is deliberately not a `ValueError`). Nothing is ever resumed automatically: an interrupted run waits for the user.

## 3. Where a run lives and what the monitor reads

`data/user/training/<run id>/` holds `request.json` (the request that started it), an optional `control.json` (stop or pause), and one sub-folder per fit (a walk-forward refits on every window) containing the checkpoints and `metrics.jsonl`: one row per epoch (train and validation loss, learning rate, gradient norm, epoch seconds, best score, checkpoint written, peak and process memory) plus `start`, `resume`, `user_stop`/`user_pause`, `early_stopping`, `checkpoint_failed`, `interrupted` and `finish` rows. The monitor derives a run's status from those rows and the file's age (`running`, `completed`, `stopped_early`, `stopped`, `paused`, `interrupted`, `orphaned`), never from memory in the web process, so a reloaded page, a second browser or a restarted app shows the same history.

Only training and validation loss are written. The test and forward segments are never read while a network trains (`tests/test_training_api.py::test_the_monitor_never_shows_test_segment_metrics`).

Routes (`src/webapp/training_api.py`): `GET /api/training/runs`, `GET /api/training/run?id=&fit=&since=`, `GET /api/training/profiles`, `POST /api/training/control` (`stop`, `pause`, `clear`), `POST /api/training/resume` (`mode`: `resume`, `warm`, `restart`), `POST /api/training/archive`, `POST /api/training/preflight`.

* **Stop and pause** are cooperative: the loop reads `control.json` after an epoch's checkpoint, records the event, commits the checkpoint and raises `TrainingHalted`. The whole experiment ends (the job shows *cancelled* with the reason) so that the walk-forward does not go on to the next fit with untrained networks and save a result that looks complete; nothing is registered as an experiment. A very long epoch delays the stop to its end.
* **Resume** submits the saved request again with the same run id: every fit whose checkpoint verifies continues from it (finished fits return their saved best weights), others start fresh. **Warm start** is for a changed data window. **Restart** makes a new run.
* **Archive** moves the run to `data/user/training/_archive/`; files are kept.

## 4. Resource profiles and the memory estimate

`src/ml/deep/profiles.py`: `conservative` (batch cap 128, fewer loader threads), `balanced` (cap 512), `performance` (cap 2048, mixed precision when CUDA is present). A profile only caps execution settings and reports what it changed; `comparable_with_unprofiled_run` is false whenever it changed the batch size, because a different batch size is a different optimisation. `estimate()` prices a configuration with `ml/resources.deep_mb` and reports a range (x0.7 to x1.8, the model's own rule of thumb is not exact) against the free memory and the cgroup limit. A request whose upper bound exceeds what is free is refused before it is queued, with a smaller batch size suggested; `force: true` overrides.

**Out-of-memory.** A Python `MemoryError` or the resource watchdog ends the job at an epoch boundary after the last checkpoint, and the run is then resumable. A kernel OOM kill (SIGKILL) gives the process no chance to write anything: the run shows as `orphaned`, resumable from the last committed checkpoint, and the monitor says so. The platform cannot prevent the kill, only make it cheap.

## 5. Model-by-model compatibility matrix

"Exact" is claimed only where `tests/test_training_recovery_architectures.py` (or the N-BEATS/N-HiTS tests in `tests/test_training_recovery.py`) resumed after an interruption in the middle of an epoch and compared with the uninterrupted run.

| Model | Trainer | Checkpointed | Exact resume verified | Notes |
|---|---|---|---|---|
| N-BEATS, N-HiTS | `fit_network` via `DeepModel` | yes | yes (dropout 0.2, scheduler, early stopping) | |
| LSTM, GRU, TCN | same | yes | yes | |
| PatchTST, iTransformer, DLinear, TFT | same | yes | yes | |
| Autoencoder, VAE | same | yes | **no** (checkpointed by the same code, resume not exercised; the VAE draws from the RNG inside the forward pass, which the saved RNG states cover in principle) | |
| Graph models (`ml/graph.py`) | `fit_network` without a folder | **no** | no | calls the trainer with no checkpoint folder |
| `models/deep_sequence`, `deep_family`, `deep_forecast`, `generative`, `diffusion`, `pinn`, `representation`, `graph_forecast`; `strategies/deep.py`; `derivatives/deep_hedging.py`; `rl/deep.py` | their own loops | **no** | no | interrupted runs restart from the beginning |
| Linear, tree and boosting models | single fit | not applicable | n/a | fits take seconds; an interrupted job restarts |

## 6. Small universes

`ml/datasets.py`: a cross-sectional z-score or rank needs at least three assets on a date. With one or two tickers every cell used to become missing and the dataset failed with "no row survives". Now each feature is scaled by the asset's own **expanding** history (`time_series_z`, first 60 days missing), which uses no later observation (`tests/test_ml_small_universe.py::test_time_series_scaling_uses_only_the_past` perturbs the future and compares). The substitution is recorded in the dataset's `meta["normalization"]` (`requested`, `applied`, `note`). When nothing can be built the error lists priced days per asset, the binding rule and the required history. `time_series_z` can also be chosen explicitly for any universe.

## 7. Not implemented or not verified

* Exact resume of the autoencoder and VAE; checkpointing of the graph models and of the loops listed as "own loops" above.
* Resuming a **search** (random, Bayesian, genetic) or an Optuna study: the search keeps its trials in the registry and counts every attempt, but an interrupted search is restarted, not continued, and its interrupted trial is not resumed from a checkpoint.
* A worker **process** separate from the web process: training still runs on the app's worker threads, so an app restart ends the job (the job log marks it `interrupted`, the run is resumable from the monitor).
* GPU out-of-memory recovery was not exercised (no GPU in the build environment); AMP and `accumulation_steps` are tested for consistency on CPU only.
* Gradient histograms and per-layer diagnostics: only the mean gradient norm per epoch is recorded.
