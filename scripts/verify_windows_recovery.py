"""Does checkpoint recovery survive real interruptions on THIS operating system? Run it on the machine you care about:

    python scripts/verify_windows_recovery.py [--report docs/generated/windows_recovery_report.json] [--only name,name] [--skip-browser]

Each check really starts and really terminates processes (``TerminateProcess`` on Windows, ``SIGKILL`` elsewhere) and looks at what is left on disk. The report records the platform: **a report written on Linux is
not evidence for Windows**, and the repository only calls Windows verified when ``docs/generated/windows_recovery_report.json`` says ``"platform": "win32"`` and no check failed. Checks that exist only for one system
say ``skipped`` elsewhere, never ``passed``.

Checks
  spawn_worker_kill       a worker started with the ``spawn`` start method (the only one Windows has) is terminated while it commits checkpoints; the newest valid checkpoint is loadable, no half file is left
  atomic_replace_contention  a reader keeps ``last.pt`` open while the writer replaces it hundreds of times (a Windows sharing violation is a ``PermissionError``): every replace completes, the file is never torn
  sqlite_lock_after_kill  a process killed while it holds a SQLite write lock (rollback journal and WAL) leaves a database the next process can write to and that passes ``integrity_check``
  training_kill_resume    a training worker (new process, not a fork) is terminated after a committed epoch and a new process resumes it to the uninterrupted result; the completion record and logs are consistent
  commit_protocol_windows the kill-at-each-step scenarios of ``tests/test_epoch_commit_protocol.py`` (run through pytest)
  browser_reconnection    the app server is terminated and restarted on the same port and folder while the Training monitor is open in a real browser: the page reconnects and shows the run, resumable
  task_scheduler_restart  (Windows only) Task Scheduler runs the resume of a terminated run in its own session (``schtasks /Create`` then ``/Run``), and the run ends complete
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WIN = sys.platform == "win32"


# ---------------------------------------------------------------------------------------------------------------------------------------- helpers
def _kill(proc) -> None:
    """Terminate abruptly: ``Process.kill``/``Popen.kill`` is SIGKILL on POSIX and TerminateProcess on Windows."""
    proc.kill()
    proc.join(10) if hasattr(proc, "join") else proc.wait(10)


def _say(*a) -> None:
    if os.environ.get("QUANT_VERIFY_VERBOSE"):
        print("  ..", *a, flush=True)


def _py(*args, **kw):
    return subprocess.run([sys.executable, "-W", "ignore", *args], cwd=ROOT, capture_output=True, text=True, timeout=kw.pop("timeout", 900), **kw)


# ---------------------------------------------------------------------------------------------------------------------------------------- spawn worker
def _worker_commit_forever(folder: str) -> None:                                  # runs in the spawned child (must be importable at module level for ``spawn``)
    import torch

    from src.ml.deep.checkpoint import CheckpointManager

    mgr = CheckpointManager(folder)
    net = torch.nn.Linear(64, 64)
    k = 0
    while True:
        k += 1
        with torch.no_grad():
            net.weight += 1e-3
        mgr.save({"model": net.state_dict(), "optimizer": {}, "epoch": k, "config_fp": "x", "data_fp": "y"})
        Path(folder, "alive").write_text(str(k))


def check_spawn_worker_kill(tmp: Path) -> dict:
    from src.ml.deep.checkpoint import CheckpointManager

    ctx = mp.get_context("spawn")
    folder = tmp / "spawn"
    folder.mkdir()
    killed_at = []
    for round_ in range(5):                                                          # several kills at different moments: a single lucky kill proves little
        p = ctx.Process(target=_worker_commit_forever, args=(str(folder),))
        p.start()
        t0 = time.time()
        while time.time() - t0 < 60 and not (folder / "alive").exists():
            time.sleep(0.05)
        time.sleep(0.15 + 0.07 * round_)
        _kill(p)
        mgr = CheckpointManager(folder)
        notes = mgr.repair()
        blob, which, problems = mgr.latest_valid()
        if blob is None:
            return {"ok": False, "detail": f"round {round_}: no valid checkpoint after the kill ({problems})"}
        left = [f.name for f in folder.glob("*.tmp*")]
        if left:
            return {"ok": False, "detail": f"round {round_}: temporary files left after repair: {left}"}
        killed_at.append(int(blob["epoch"]))
        (folder / "alive").unlink(missing_ok=True)
        if round_ and killed_at[-1] <= 0:
            return {"ok": False, "detail": "the checkpoint epoch did not advance"}
    return {"ok": True, "detail": f"5 forced terminations of spawned workers; newest valid checkpoint epochs {killed_at}; no temporary file left"}


# ---------------------------------------------------------------------------------------------------------------------------------------- atomic replace
def check_atomic_replace_contention(tmp: Path) -> dict:
    from src.ml.deep import checkpoint

    folder = tmp / "contend"
    folder.mkdir()
    target = folder / "last.pt"
    target.write_bytes(b"0" * 4096)
    stop = threading.Event()
    seen = {"reads": 0, "torn": 0}

    def reader():
        while not stop.is_set():
            try:
                with open(target, "rb") as fh:                                       # holding the file open is what makes Windows refuse the rename
                    data = fh.read()
                    time.sleep(0.002)
                seen["reads"] += 1
                if data and len(set(data)) != 1:
                    seen["torn"] += 1                                                 # a payload is one repeated byte: anything else is a torn read
            except OSError:
                pass

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    retries, errors = 0, []
    real = os.replace

    def counting(src, dst):
        nonlocal retries
        try:
            return real(src, dst)
        except PermissionError:
            retries += 1
            raise
    checkpoint.os.replace = counting
    try:
        for k in range(1, 301):
            tmp_file = folder / "last.next.pt"
            tmp_file.write_bytes(bytes([k % 250 + 1]) * 4096)
            try:
                checkpoint._replace(tmp_file, target)
            except Exception as exc:                                                 # noqa: BLE001
                errors.append(f"{k}: {exc!r}")
                break
    finally:
        checkpoint.os.replace = real
        stop.set()
        th.join(5)
    ok = not errors and seen["torn"] == 0
    return {"ok": ok, "detail": f"300 replaces with a reader holding the file open: {retries} sharing violations retried, {len(errors)} replaces failed, {seen['torn']} torn reads in {seen['reads']} reads" + (f"; {errors[:2]}" if errors else "")}


# ---------------------------------------------------------------------------------------------------------------------------------------- sqlite
def _sqlite_hold(path: str, mode: str) -> None:
    con = sqlite3.connect(path, isolation_level=None)
    con.execute(f"PRAGMA journal_mode={mode}")
    con.execute("BEGIN IMMEDIATE")
    con.executemany("INSERT INTO t(v) VALUES (?)", [(i,) for i in range(2000)])
    Path(path + ".locked").write_text("1")
    time.sleep(600)                                                                    # holds the write lock until it is terminated


def check_sqlite_lock_after_kill(tmp: Path) -> dict:
    ctx = mp.get_context("spawn")
    details = []
    for mode in ("DELETE", "WAL"):
        path = str(tmp / f"lock-{mode}.db")
        con = sqlite3.connect(path)
        con.execute(f"PRAGMA journal_mode={mode}")
        con.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v INTEGER)")
        con.executemany("INSERT INTO t(v) VALUES (?)", [(i,) for i in range(10)])
        con.commit()
        con.close()
        p = ctx.Process(target=_sqlite_hold, args=(path, mode))
        p.start()
        t0 = time.time()
        while time.time() - t0 < 60 and not Path(path + ".locked").exists():
            time.sleep(0.05)
        try:
            c2 = sqlite3.connect(path, timeout=0.3)
            c2.execute("INSERT INTO t(v) VALUES (-1)")
            c2.commit()
            blocked = False
        except sqlite3.OperationalError:
            blocked = True                                                             # while the holder lives the writer must wait: that is the lock
        _kill(p)
        c3 = sqlite3.connect(path, timeout=10)
        c3.execute("INSERT INTO t(v) VALUES (99)")
        c3.commit()
        n = c3.execute("SELECT count(*) FROM t").fetchone()[0]
        ok = c3.execute("PRAGMA integrity_check").fetchone()[0]
        c3.close()
        if not blocked or ok != "ok" or n != 11:
            return {"ok": False, "detail": f"{mode}: blocked_while_held={blocked}, integrity={ok}, rows={n} (expected 11: the killed writer's 2000 rows rolled back, one new row)"}
        details.append(f"{mode}: lock held and honoured, killed holder's transaction rolled back, new writer succeeded, integrity ok")
    return {"ok": True, "detail": "; ".join(details)}


# ---------------------------------------------------------------------------------------------------------------------------------------- training
def _scenario(name, folder, mode, out, *extra):
    return subprocess.run([sys.executable, "-W", "ignore", "-m", "tests.fresh_process_scenarios", name, str(folder), mode, str(out), *extra], cwd=ROOT, capture_output=True, text=True, timeout=900)


def check_training_kill_resume(tmp: Path) -> dict:
    from tests.fresh_process_scenarios import was_killed

    ref = _scenario("nbeats", tmp / "ref", "reference", tmp / "ref.json")
    dead = _scenario("nbeats", tmp / "run", "interrupted", tmp / "dead.json", "--kill-after", "3")
    if ref.returncode or not was_killed(dead.returncode):
        return {"ok": False, "detail": f"reference rc={ref.returncode}; interrupted rc={dead.returncode}; {dead.stderr[-300:]}"}
    res = _scenario("nbeats", tmp / "run", "resume", tmp / "res.json")
    if res.returncode:
        return {"ok": False, "detail": f"resume failed: {res.stderr[-600:]}"}
    a, b = json.loads((tmp / "ref.json").read_text()), json.loads((tmp / "res.json").read_text())
    same = a["loss"] == b["loss"] and a["params"] == b["params"]
    fit = next((tmp / "run").rglob("metrics.jsonl")).parent
    log = [json.loads(x) for x in (fit / "metrics.jsonl").read_text().splitlines()]
    epochs = [r["epoch"] for r in log if r["type"] == "epoch"]
    cm = json.loads((fit / "commit.json").read_text())
    ok = same and epochs == list(range(1, len(epochs) + 1)) and log[-1]["type"] == "finish" and cm["state"] == "applied" and cm["committed_unit"] == len(epochs)
    return {"ok": ok, "detail": f"terminated worker rc={dead.returncode}; resumed run {'equals' if same else 'DIFFERS FROM'} the uninterrupted one; {len(epochs)} epoch rows in order; commit.json {cm['state']}/{cm['committed_unit']}"}


def check_commit_protocol(tmp: Path) -> dict:
    r = _py("-m", "pytest", "tests/test_epoch_commit_protocol.py", "-q", "-p", "no:cacheprovider", "-x", timeout=3000)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return {"ok": r.returncode == 0, "detail": tail or r.stderr[-300:]}


# ---------------------------------------------------------------------------------------------------------------------------------------- browser
def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def check_browser_reconnection(tmp: Path) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"ok": None, "detail": "skipped: playwright is not installed"}
    from src.ml.deep import runs as run_files
    from tests.app_roots import isolated_root
    from src.utils.config import load_config
    from tests.fresh_process_scenarios import was_killed

    root = isolated_root(tmp / "root", load_config())
    base = run_files.base(root)
    dead = _scenario("forecast_lstm", base / "lstm-run", "interrupted", tmp / "d.json", "--kill-after", "3")
    if not was_killed(dead.returncode):
        return {"ok": False, "detail": f"could not create the interrupted run: rc={dead.returncode}"}
    for f in (base / "lstm-run").rglob("*"):
        if f.is_file():
            os.utime(f, (time.time() - 7200,) * 2)                                      # nothing has written to it for two hours: orphaned
    port = _free_port()
    cmd = [sys.executable, "-W", "ignore", "-m", "tests.app_server_scenario", str(root), str(port)]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    srv = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, creationflags=flags)
    try:
        srv.stdout.readline()
        with sync_playwright() as pw:
            from tests.test_webapp_browser import _chromium                           # the Chromium the browser tests use (system Chrome/Edge on Windows)

            browser = pw.chromium.launch(executable_path=_chromium(), args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            from tests.test_webapp_training_browser import _monitor                   # opens the Machine learning workspace's Training monitor

            def listing() -> str:
                _monitor(page)
                page.wait_for_selector("#ml-results tr[data-run=lstm-run]", timeout=30_000)
                return page.inner_text("#ml-results tr[data-run=lstm-run]").replace("\n", " ").replace("\t", " ")

            _say("opening the page")
            page.goto(f"http://127.0.0.1:{port}/")
            page.wait_for_selector(".chip", timeout=60_000)
            _say("page open; listing runs")
            first = listing()
            _say("first listing:", first[:80])
            srv.kill()                                                                  # the app dies (TerminateProcess on Windows)
            srv.wait(10)
            _say("server killed")
            import urllib.request

            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3)
                gone = "still answering"
            except OSError:
                gone = "unreachable"
            _say("while dead:", gone)
            srv = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, creationflags=flags)
            _say("second server starting")
            srv.stdout.readline()
            _say("second server up")
            token = None
            for _ in range(6):                                                          # a reload reconnects to the new server (it issues a new session token)
                time.sleep(0.5)
                try:
                    page.goto(f"http://127.0.0.1:{port}/")
                    page.wait_for_load_state("load", timeout=15_000)                  # the page restores the tab it was on, so the first chip need not be visible
                    _say("reloaded")
                    token = listing()
                    break
                except Exception as exc:                                                # noqa: BLE001 - the server is still starting
                    _say("reload attempt failed:", repr(exc)[:200])
                    continue
            browser.close()
        ok = gone == "unreachable" and token is not None and "Orphaned" in token and "Orphaned" in first
        return {"ok": ok, "detail": f"before: {first}; while the server was dead: {gone}; after restart on the same port and folder: {token}"}
    finally:
        srv.kill()


# ---------------------------------------------------------------------------------------------------------------------------------------- task scheduler
def check_task_scheduler_restart(tmp: Path) -> dict:
    if not WIN:
        return {"ok": None, "detail": "skipped: Task Scheduler exists only on Windows"}
    from tests.fresh_process_scenarios import was_killed

    name = f"QuantRecoveryCheck{os.getpid()}"
    folder = tmp / "sched"
    dead = _scenario("nbeats", folder, "interrupted", tmp / "d.json", "--kill-after", "3")
    if not was_killed(dead.returncode):
        return {"ok": False, "detail": f"could not create the interrupted run: rc={dead.returncode}"}
    out = tmp / "sched-res.json"
    action = f'"{sys.executable}" -W ignore -m tests.fresh_process_scenarios nbeats "{folder}" resume "{out}"'
    start = time.strftime("%H:%M", time.localtime(time.time() + 120))
    try:
        mk = subprocess.run(["schtasks", "/Create", "/TN", name, "/TR", action, "/SC", "ONCE", "/ST", start, "/F"], capture_output=True, text=True, cwd=ROOT)
        if mk.returncode:
            return {"ok": False, "detail": f"schtasks /Create failed: {mk.stderr.strip() or mk.stdout.strip()}"}
        run = subprocess.run(["schtasks", "/Run", "/TN", name], capture_output=True, text=True)
        if run.returncode:
            return {"ok": False, "detail": f"schtasks /Run failed: {run.stderr.strip() or run.stdout.strip()}"}
        t0 = time.time()
        while time.time() - t0 < 600 and not out.exists():
            time.sleep(2)
        fit = next(folder.rglob("metrics.jsonl")).parent
        log = [json.loads(x) for x in (fit / "metrics.jsonl").read_text().splitlines()]
        epochs = [r["epoch"] for r in log if r["type"] == "epoch"]
        ok = out.exists() and epochs == list(range(1, len(epochs) + 1)) and log[-1]["type"] == "finish" and any(r["type"] == "resume" and r["level"] == "exact" for r in log)
        return {"ok": ok, "detail": f"the scheduled task resumed the terminated run: result file {'written' if out.exists() else 'MISSING'}, {len(epochs)} epochs in order, resume level exact"}
    finally:
        subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], capture_output=True, text=True)


CHECKS = {"spawn_worker_kill": check_spawn_worker_kill, "atomic_replace_contention": check_atomic_replace_contention, "sqlite_lock_after_kill": check_sqlite_lock_after_kill,
          "training_kill_resume": check_training_kill_resume, "commit_protocol_windows": check_commit_protocol, "browser_reconnection": check_browser_reconnection,
          "task_scheduler_restart": check_task_scheduler_restart}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-browser", action="store_true")
    args = ap.parse_args(argv)
    names = [n for n in (args.only.split(",") if args.only else CHECKS) if n]
    if args.skip_browser:
        names = [n for n in names if n != "browser_reconnection"]
    import torch

    report = {"platform": sys.platform, "os": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__, "start_method_default": mp.get_start_method(allow_none=True) or mp.get_context().get_start_method(),
              "at": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    for name in names:
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="quant-recovery-") as d:
            try:
                res = CHECKS[name](Path(d))
            except Exception:                                                              # noqa: BLE001
                res = {"ok": False, "detail": traceback.format_exc()[-800:]}
        status = "skipped" if res["ok"] is None else "passed" if res["ok"] else "failed"
        report["checks"][name] = {"status": status, "detail": res["detail"], "seconds": round(time.time() - t0, 1)}
        print(f"{status.upper():8} {name}: {res['detail']}", flush=True)
    failed = [n for n, c in report["checks"].items() if c["status"] == "failed"]
    ran = [n for n, c in report["checks"].items() if c["status"] == "passed"]
    report["summary"] = {"passed": len(ran), "failed": len(failed), "skipped": len(report["checks"]) - len(ran) - len(failed),
                         "verified_on_windows": bool(WIN and not failed and len(ran) == len(CHECKS))}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"]))
    return 1 if failed else 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
