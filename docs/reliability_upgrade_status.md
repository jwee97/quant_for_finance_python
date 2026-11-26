# Reliability, ML training and research-integration upgrade: status by part

Status words: **implemented and tested** (a test exercises it and passes), **implemented, not live-verified** (the code exists and was exercised only against mocks, synthetic data or one platform), **partial**, **not implemented**. Nothing is called complete because a class, route or button exists.

| Part | Requirement | Status | Evidence and limits |
|---|---|---|---|
| A | Resumable training: atomic epoch checkpoints with model, optimizer, scheduler, scaler, RNG, early-stopping state, fingerprints, schema version | **implemented and tested** (CPU) | `tests/test_training_recovery.py`, `tests/test_training_recovery_architectures.py`: interrupted mid-epoch, resumed, weights equal to the uninterrupted run (1e-6) for N-BEATS, N-HiTS, LSTM, GRU, TCN, PatchTST, iTransformer, DLinear, TFT |
| A | Honest recovery levels (exact, epoch boundary, warm start, restart required) | **implemented and tested** | `docs/training_reliability.md` section 2 |
| A | Resume, restart or archive interrupted jobs; no automatic restart | **implemented and tested** | `/api/training/resume|archive`, `tests/test_training_api.py`, `tests/test_webapp_training_browser.py` |
| A | Autoencoder / VAE exact resume; graph and other deep models' own loops | **implemented and tested** (second upgrade) | 28 fresh-process kill-and-resume scenarios, `docs/deep_learning_training_matrix.md`; Stable-Baselines3 agents remain unsupported |
| A | Resumable searches; interrupted trials never counted as completed, trial counts preserved | **partial** | interrupted operations are closed as failed-with-interrupted and still counted (`tests/test_research_interrupted_operations.py`); an interrupted search is restarted, not continued |
| B | Training monitor: epoch metrics, loss chart, learning rate, memory, checkpoints, controls, persistence across reloads | **implemented and tested** (backend and browser) | reads `metrics.jsonl`; controls: stop, pause, resume, warm start, restart as new, archive |
| B | Test-set metrics never shown per epoch | **implemented and tested** | only train and validation loss are written |
| B | Server-sent events | **not implemented** | the page polls every two seconds while a run is going |
| B | Gradient diagnostics beyond the mean norm | **implemented and tested** (second upgrade) | `ml/deep/gradients.py`, five-view monitor; "save checkpoint now" and a log viewer are still **not implemented** |
| C | Preflight memory range, profiles, gradient accumulation, AMP, clipping | **implemented and tested** on CPU | `profiles.py`; AMP is a no-op on CPU, **not live-verified on GPU** |
| C | OOM guidance | **partial** | a stopped-for-memory run says so and suggests a smaller batch with accumulation; a kernel OOM kill cannot be intercepted (the run shows as orphaned and is resumable from the last checkpoint) |
| D | Durable job records, heartbeat, stale detection, interrupted marking | pre-existing (`webapp/jobs.py`), unchanged | jobs now carry `training_run` and link to the monitor |
| D | Worker **process** separation | **not implemented** | training still runs on the app's worker threads; a killed app's run is resumable by a new app (`tests/test_worker_crash_recovery.py`) |
| D | No duplicate jobs | **partial** | a second start of a run that already has checkpoints is refused; other job kinds are not de-duplicated |
| E | Small-universe datasets: no silent loss of every row | **implemented and tested** | `time_series_z` fallback for fewer than three assets, recorded in the dataset's metadata; per-asset diagnostics in the "no row survives" error |
| E | Leakage audits, fair model comparison | pre-existing audits unchanged; **no new work** | |
| F | Unified market-data layer, provenance contracts, calendars, corporate actions | **not implemented** in this pass | |
| G | Fundamentals / Alpha Lab improvements | **not implemented** in this pass | |
| H | Portfolio and risk integration | **not implemented** in this pass | |
| I | Unified research workflow | **not implemented** in this pass | |
| J | Windows and Replit reliability | **partial** | replace-retry for sharing violations (simulated, not run on Windows); Replit storage warning in the monitor; run ids are valid Windows names |
| K | Tests and benchmarks | **implemented** | `scripts/benchmark_training.py`, `docs/generated/training_benchmark.json`; structured logging not added |
| N | Model-by-model checkpoint compatibility matrix | **implemented** | `docs/training_reliability.md` section 5 |
