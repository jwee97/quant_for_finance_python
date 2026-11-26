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

## 3a. When is a unit finished? The completion protocol

`src/ml/deep/commit.py`. An epoch (or an optimizer block, or a block of timesteps) is *finished* when its checkpoint has replaced `last.pt`, and for no other reason. The logs, the monitor and the job's progress message are copies and announcements of that fact, written afterwards in a fixed order:

| Step | What happens | Durable? | If the process dies right after |
|---|---|---|---|
| 1 begin | `inflight.json` names the unit | no (one small write) | the unit is reported as recomputed work |
| 2 commit | checkpoint written, `fsync`ed, read back, checksummed, then `os.replace`d over `last.pt` (the commit point). The checkpoint also carries the unit's own record (`last`: metrics row, gradient summaries, warnings). Then `commit.json` is replaced atomically with `state: committed` | yes | the unit is finished; the logs are repaired from the checkpoint on restart |
| 3 apply | gradient summary, metrics row (`committed: true`) and warnings are appended; `commit.json` becomes `state: applied` | the row is flushed | whatever is missing is restored from the checkpoint |
| 4 announce | job progress, callbacks | no | nothing: recovery never reads them |

A unit that is not checkpointed (`checkpoint_every` above 1, no improvement, or a checkpoint that failed for lack of space) still logs its row so that the monitor moves, marked `committed: false`. It is provisional.

On a resume, `commit.reconcile` compares the checkpoint that was read (the truth), `commit.json` (what the dead process believed) and the logs, and: cuts a half-written last line; drops the rows of every unit the checkpoint does not contain; restores the row, the gradient summaries and the warnings of the checkpoint's last unit when the kill fell between steps 2 and 3 (also when only some were applied); and writes one `recomputed_work` row naming the units that will be run again (in flight, logged but not committed, or committed according to `commit.json` but unreadable in the checkpoint that survived), with their number and phase. `CheckpointManager.repair` completes a commit that died between its two renames (the staged `last.next.pt` is read as the newest checkpoint when it is complete and newer) and removes the temporary files of writes that never finished.

The monitor shows `committed_epoch` (from `commit.json`, never more than the checkpoint holds) next to the last epoch logged, and for an interrupted run how many logged epochs a resume will recompute.

**Not covered:** a power loss between `os.replace` and the directory's own flush is handled with `fsync` of the directory on POSIX only; on Windows NTFS journals the rename. A log line that could not be written because the disk was full (as opposed to a checkpoint) is not rebuilt: the monitor shows a gap, training and the checkpoint's own history are intact. Tests: `tests/test_epoch_commit_protocol.py` (kill at each of eight steps, both loops), `tests/test_checkpoint_hardening.py`, `tests/test_recovery_faults.py`.

## 4. Resource profiles and the memory estimate

`src/ml/deep/profiles.py`: `conservative` (batch cap 128, fewer loader threads), `balanced` (cap 512), `performance` (cap 2048, mixed precision when CUDA is present). A profile only caps execution settings and reports what it changed; `comparable_with_unprofiled_run` is false whenever it changed the batch size, because a different batch size is a different optimisation. `estimate()` prices a configuration with `ml/resources.deep_mb` and reports a range (x0.7 to x1.8, the model's own rule of thumb is not exact) against the free memory and the cgroup limit. A request whose upper bound exceeds what is free is refused before it is queued, with a smaller batch size suggested; `force: true` overrides.

**Out-of-memory.** A Python `MemoryError` or the resource watchdog stops the job in the middle of an epoch; the last committed checkpoint (the previous epoch's) is intact and the run is then resumable. A kernel OOM kill (SIGKILL) gives the process no chance to write anything: the run shows as `orphaned`, resumable from the last committed checkpoint, and the monitor says so. The platform cannot prevent the kill, only make it cheap.

## 5. Model-by-model compatibility matrix

Superseded by `deep_learning_training_matrix.md`, which covers every deep-training loop (feature-based networks, autoencoders and VAEs, graph networks, forecast networks, GANs, diffusion, deep hedging, physics-informed networks and the reinforcement-learning agents) with fresh-process recovery results and the gradient diagnostics.

## 6. Small universes

`ml/datasets.py`: a cross-sectional z-score or rank needs at least three assets on a date. With one or two tickers every cell used to become missing and the dataset failed with "no row survives". Now each feature is scaled by the asset's own **expanding** history (`time_series_z`, first 60 days missing), which uses no later observation (`tests/test_ml_small_universe.py::test_time_series_scaling_uses_only_the_past` perturbs the future and compares). The substitution is recorded in the dataset's `meta["normalization"]` (`requested`, `applied`, `note`). When nothing can be built the error lists priced days per asset, the binding rule and the required history. `time_series_z` can also be chosen explicitly for any universe.

## 7. Searches and research operations

A search trial (random, Bayesian, genetic, Optuna) is registered in the study's lineage BEFORE it runs and finished as done, failed or cancelled; every operation counts as a look whatever its end. What was missing: a trial whose process was killed stayed "running" for ever. `research_controls.reconcile_interrupted` (called when the trials or lineage are read) ends operations that have had no event for 30 minutes and are not live in this process as `failed` with `interrupted: true`. They are never read as completed trials and they keep counting (`raw_looks` is unchanged; `tests/test_research_interrupted_operations.py`). The 30 minute threshold is a trade: a trial legitimately running longer in a second process would be closed early (its own `done` still lands afterwards).

An interrupted search is **restarted, not continued**: its completed trials are in the registry and are counted, but the sampler does not pick up from them and the trial that was running is not resumed from a checkpoint.

## 8. Benchmarks

`python scripts/benchmark_training.py` (results in `docs/generated/training_benchmark.json`). Measured on the build machine (CPU only, 4,000 training and 1,000 validation windows of 20 days x 30 features, batch 256, 4 epochs). `commit` is one complete checkpoint write (serialise, fsync, read back and verify, replace) of `last.pt` and `best.pt`; `read` is finding and verifying the newest checkpoint.

| Model | Size | Parameters | Epoch (s) | Checkpoint (MB) | Commit (s) | Read and verify (s) |
|---|---|---|---|---|---|---|
| N-BEATS | small | 40 k | 0.08 | 0.43 | 0.027 | 0.004 |
| N-BEATS | medium | 258 k | 0.15 | 2.8 | 0.039 | 0.007 |
| N-BEATS | large | 2.0 M | 0.46 | 23.1 | 0.229 | 0.042 |
| N-HiTS | small | 36 k | 0.07 | 0.39 | 0.017 | 0.003 |
| N-HiTS | medium | 161 k | 0.14 | 1.7 | 0.032 | 0.006 |
| N-HiTS | large | 1.1 M | 0.35 | 12.4 | 0.154 | 0.018 |

A checkpoint is the weights plus two Adam moment buffers plus the best weights, about 11 bytes per parameter on disk. For the large networks a commit costs about half an epoch: raise `checkpoint_every` (a checkpoint is still written on every improvement, on the last epoch and on a stop). Process memory is dominated by PyTorch and the data, not by the checkpoint (about 350 to 440 MB for every row). Timings from one run on a shared machine vary by tens of percent; they show orders of magnitude, not a guarantee.

## 9. Windows and Replit

* Checkpoint files are replaced with `os.replace` after an `fsync`; on Windows a virus scanner or the monitor reading the file at that instant can raise a sharing violation, so the replace is retried up to eight times with a growing pause (`tests/test_training_recovery_architectures.py::test_a_windows_sharing_violation_during_the_replace_is_retried` simulates the error; it was not run on Windows).
* Run folders use only letters, digits, `.`, `_` and `-`, so no name is invalid on Windows; checkpoint files are written under `data/user/training` relative to the app root, never to a temporary directory on another volume.
* On Replit the monitor prints a warning that a deployment's disk can be replaced on republish; copy `data/user/training` to durable storage for long runs. A Replit workspace that goes idle ends the job like any restart: the run shows as interrupted or orphaned and can be resumed.
* The training runs on the app's worker threads, with PyTorch threads capped by the container's CPU quota (`resources.default_threads`).

## 10. Not implemented or not verified

See section 8 of `deep_learning_training_matrix.md` for the current list; Stable-Baselines3 checkpointing (chunked `learn` plus an application manifest, exact only for a single CPU environment with constant schedules) and the PINN's checkpointing at L-BFGS boundaries are described in its sections 6.4 and 6.5. Still open from this page: continuing (not restarting) an interrupted search or Optuna study; a separate worker process with its own heartbeat (a killed app now leaves a run that a new app can resume, but training still runs on the app's worker threads); GPU behaviour; Windows was reasoned about and simulated, not run.
