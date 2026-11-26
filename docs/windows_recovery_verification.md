# Windows recovery verification

**Status: unverified on Windows.** No Windows machine was available to this work (the session ran on Linux and no computer was attached). Nothing below is claimed for Windows except that the code takes the Windows-specific precautions listed, and that a verification script exists to find out. A Linux result is not Windows evidence, and the script's report says which platform produced it.

## How to verify on a Windows machine

```
python scripts/verify_windows_recovery.py --report docs/generated/windows_recovery_report.json
```

The repository calls Windows verified only when `docs/generated/windows_recovery_report.json` exists, says `"platform": "win32"`, and `"verified_on_windows": true` (every check ran, none failed, none skipped). `python -m pytest tests/test_platform_recovery_checks.py tests/test_epoch_commit_protocol.py tests/test_checkpoint_hardening.py` runs the same ground as tests.

| Check | What it does for real | Windows-specific because |
|---|---|---|
| `spawn_worker_kill` | five workers started with the `spawn` start method (Windows has no `fork`) commit checkpoints in a loop and are terminated with `Process.kill()` (`TerminateProcess` on Windows) at different moments; after each the newest checkpoint must load and no temporary file may remain after `repair()` | spawn re-imports the module in the child; `TerminateProcess` gives no chance to clean up |
| `atomic_replace_contention` | a reader thread holds `last.pt` open while the writer replaces it 300 times through `checkpoint._replace`; every replace must complete and no read may be torn | Windows refuses to rename over a file another handle has open (`PermissionError`); `_replace` retries up to eight times |
| `sqlite_lock_after_kill` | a spawned process takes a SQLite write lock (rollback journal and WAL) and is terminated; the next process must be able to write, the dead writer's rows must be rolled back, `integrity_check` must be `ok`; while the holder lives a second writer must be refused | Windows file locking semantics for `-journal` and `-wal`/`-shm` files |
| `training_kill_resume` | a training worker is terminated after a committed epoch; a new process resumes to the weights and losses of an uninterrupted run; one metrics row per epoch in order; `commit.json` ends `applied` | exit status of a terminated process, file handles released at termination |
| `commit_protocol_windows` | runs `tests/test_epoch_commit_protocol.py`: a kill at each of eight steps of a unit's completion, both training loops | |
| `browser_reconnection` | the app server is terminated and restarted on the same port and folder with the Training monitor open in a real browser; the page must reconnect and show the interrupted run, resumable | `CREATE_NEW_PROCESS_GROUP`, port reuse after `TerminateProcess` |
| `task_scheduler_restart` | `schtasks /Create` and `/Run` start the resume of a terminated run in a Task Scheduler session; the run must end complete with a `resume` row at level `exact`; the task is deleted afterwards | Task Scheduler exists only on Windows (skipped elsewhere, never "passed") |

Nothing resumes by itself: the platform never restarts a training run automatically (an interrupted run waits for the user, who resumes it in the Training monitor). The Task Scheduler check therefore verifies that a *scheduled command that resumes a run* works, which is how an unattended Windows machine would be set up; it is not a built-in auto-restart.

## What exists for Windows in the code (reasoned, not run on Windows)

* `checkpoint._replace` retries `os.replace` on `PermissionError`; `commit.atomic_write_text` uses it for `commit.json` and for rewritten logs; the directory `fsync` is attempted and ignored where the system refuses it.
* Run folder names contain only letters, digits, `.`, `_` and `-`; checkpoint temporary names include the process id.
* `tests/fresh_process_scenarios.die()` ends a process with `SIGKILL`, or `SIGTERM` (`TerminateProcess`) where there is no `SIGKILL`; `was_killed()` accepts the non-zero status Windows reports.
* The POSIX-only fault test (`RLIMIT_FSIZE`) is skipped on Windows.

## Linux results (not Windows)

`docs/generated/linux_recovery_report.json`, written by the same script on Linux-6.18.44-fc-v880-x86_64-with-glibc2.39, Python 3.11.18, torch 2.15.0+cpu:

| Check | Status | Detail |
|---|---|---|
| `spawn_worker_kill` | passed | 5 forced terminations of spawned workers; newest valid checkpoint epochs [76, 90, 126, 110, 196]; no temporary file left |
| `atomic_replace_contention` | passed | 300 replaces with a reader holding the file open: 0 sharing violations retried, 0 replaces failed, 0 torn reads in 25 reads |
| `sqlite_lock_after_kill` | passed | DELETE: lock held and honoured, killed holder's transaction rolled back, new writer succeeded, integrity ok; WAL: lock held and honoured, killed holder's transaction rolled back, new writer succeeded, integrity ok |
| `training_kill_resume` | passed | terminated worker rc=-9; resumed run equals the uninterrupted one; 8 epoch rows in order; commit.json applied/8 |
| `commit_protocol_windows` | passed | ....................                                                     [100%] |
| `browser_reconnection` | passed | before: lstm-run forecast_lstm Orphaned 3 / 10 0.6718 8 s ago 185 kB Close; while the server was dead: unreachable; after restart on the same port and folder: lstm-run forecast_lstm Orphaned 3 / 10 0.6718 14 s ago 185 kB Close |
| `task_scheduler_restart` | skipped | skipped: Task Scheduler exists only on Windows |

Summary: 6 passed, 0 failed, 1 skipped; `verified_on_windows`: **false**.

## Windows-specific failures

None observed, because nothing was run on Windows. If the script fails there, record the failures in this section, separately from the Linux results above.
