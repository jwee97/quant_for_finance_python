"""Plant one small bug at a time in a module and see whether the tests notice.

A test suite that passes says only that the code agrees with it. This script asks the opposite question: change one operator, comparison, sign, boolean or number in the module, run the tests you name,
and put the change back. A change that makes a test fail was *noticed* (killed); one that does not is a *survivor*. A survivor is either harmless (a bound that no data touches, a number in a message,
code that repeats a check made elsewhere: an equivalent change) or a place where the tests do not check what the line does, and only a person can tell which, so the survivors are listed with their lines.

    python -m experiments.plant_bugs src/fundamentals/metrics.py tests/test_fund_metrics.py --n 30 --seed 12 --out metrics_bugs.json
    python -m experiments.plant_bugs src/fundamentals/metrics.py tests/test_fund_metrics.py tests/test_fund_exact_values.py --again metrics_bugs.json --out metrics_again.json

The first draws ``--n`` places at random (a fixed ``--seed``) from everywhere a change can be made and tests each. The second puts back the survivors of an earlier file, found again by the text of their
line (line numbers move), and runs the tests as they are now: the way to see that a test written for a survivor does kill it. The work is done in a copy of ``src``, ``tests``, ``config`` and
``pyproject.toml`` made for the purpose, so the working tree is never changed. It takes a test run per change: choose the test files that exercise the module, not the whole suite."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPIED = ("src", "tests", "config", "pyproject.toml")
CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}
ARITH = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
KINDS = ("compare", "arith", "bool", "unary", "flag", "number")


def _docstrings(tree) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str):
            ids.add(id(node.body[0].value))
    return ids


def sites(tree) -> list[tuple[str, int, int]]:
    """``(kind, order, line)`` for every place a change can be made; ``order`` is the index of the node in a fixed walk of the tree. Docstrings, 0, 1 and -1 are left alone."""
    skip = _docstrings(tree)
    out = []
    for order, node in enumerate(ast.walk(tree)):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP:
            out.append(("compare", order, line))
        elif isinstance(node, ast.BinOp) and type(node.op) in ARITH:
            out.append(("arith", order, line))
        elif isinstance(node, ast.BoolOp):
            out.append(("bool", order, line))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
            out.append(("unary", order, line))
        elif isinstance(node, ast.Constant) and id(node) not in skip:
            if isinstance(node.value, bool):
                out.append(("flag", order, line))
            elif isinstance(node.value, (int, float)) and node.value not in (0, 1, -1):
                out.append(("number", order, line))
    return out


def mutate(tree, kind: str, order: int):
    """A copy of ``tree`` with the one node at ``order`` changed: a comparison turned over (``<`` to ``<=``, ``in`` to ``not in``), ``+`` and ``*`` swapped for ``-`` and ``/``, ``and`` for ``or``, a
    boolean flipped, a number made 1 more (an integer) or half as large again (a float), or a ``not`` or a minus taken off."""
    tree = copy.deepcopy(tree)
    nodes = list(ast.walk(tree))
    target = nodes[order]
    if kind == "compare":
        target.ops[0] = CMP[type(target.ops[0])]()
    elif kind == "arith":
        target.op = ARITH[type(target.op)]()
    elif kind == "bool":
        target.op = ast.Or() if isinstance(target.op, ast.And) else ast.And()
    elif kind == "flag":
        target.value = not target.value
    elif kind == "number":
        target.value = target.value + 1 if isinstance(target.value, int) else target.value * 1.5
    elif kind == "unary":
        for parent in nodes:
            for field, value in ast.iter_fields(parent):
                if value is target:
                    setattr(parent, field, target.operand)
                elif isinstance(value, list):
                    for j, item in enumerate(value):
                        if item is target:
                            value[j] = target.operand
    ast.fix_missing_locations(tree)
    return tree


def run_tests(repo: Path, tests: list[str], timeout: int) -> tuple[bool, str]:
    """``(passed, last line)``: whether the named tests all pass on the code as it now is on disk. A change that makes the code hang counts as noticed."""
    try:                                                                       # -B: a compiled copy of the file from a moment ago (the same size, the same second) must never stand in for it
        done = subprocess.run([sys.executable, "-B", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--no-header", *tests], cwd=repo, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout (the changed code hangs)"
    tail = [ln for ln in done.stdout.strip().splitlines() if ln.strip()]
    return done.returncode == 0, (tail[-1] if tail else "")[:160]


def audit(repo: Path, module: str, tests: list[str], chosen: list[tuple[str, int, int]], timeout: int, out: Path | None = None, say=print) -> list[dict]:
    """Make each change in turn in ``repo/module``, run ``tests`` and put the file back; the record of every change, written to ``out`` as it goes."""
    path = repo / module
    original = path.read_text()
    tree = ast.parse(original)
    lines = original.splitlines()
    results = []
    try:
        for n, (kind, order, line) in enumerate(chosen, 1):
            started = time.time()
            try:
                path.write_text(ast.unparse(mutate(tree, kind, order)))
                passed, why = run_tests(repo, tests, timeout)
            finally:
                path.write_text(original)
            rec = {"line": line, "kind": kind, "source": lines[line - 1].strip()[:200] if line else "", "killed": not passed, "how": why, "seconds": round(time.time() - started, 1)}
            results.append(rec)
            say(f"[{n}/{len(chosen)}] {module}:{line} {kind}: {'noticed' if rec['killed'] else 'SURVIVED'} ({rec['seconds']} s)  {rec['source'][:90]}")
            if out:
                out.write_text(json.dumps({"module": module, "tests": tests, "results": results}, indent=1))
    finally:
        path.write_text(original)
    return results


def survivors_again(tree, lines: list[str], earlier: dict) -> list[tuple[str, int, int]]:
    """Every place in the module as it is now whose line and kind are those of a survivor in ``earlier``."""
    wanted = {(r["kind"], r["source"]) for r in earlier["results"] if not r["killed"]}
    return [(kind, order, line) for kind, order, line in sites(tree) if (kind, lines[line - 1].strip()[:200] if line else "") in wanted]


def prepare(repo: Path) -> Path:
    """A copy of what the tests need, in a temporary folder."""
    work = Path(tempfile.mkdtemp(prefix="plant_bugs_"))
    for name in COPIED:
        src = repo / name
        if src.is_dir():
            shutil.copytree(src, work / name, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        elif src.exists():
            shutil.copy2(src, work / name)
    return work


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plant one small bug at a time in a module and see whether the tests notice.")
    ap.add_argument("module", help="the file to change, relative to the repository (src/fundamentals/metrics.py)")
    ap.add_argument("tests", nargs="+", help="the test files (or test ids) that exercise it")
    ap.add_argument("--n", type=int, default=30, help="how many places to draw at random")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kinds", default=",".join(KINDS), help=f"which kinds of change ({', '.join(KINDS)})")
    ap.add_argument("--again", metavar="FILE", help="put back the survivors of an earlier --out file instead of drawing new places")
    ap.add_argument("--timeout", type=int, default=420, help="seconds the tests may take for one change")
    ap.add_argument("--repo", default=str(ROOT), help="the repository to copy (default: this one)")
    ap.add_argument("--out", required=True, help="where to write the record (JSON)")
    a = ap.parse_args(argv)
    work = prepare(Path(a.repo))
    try:
        ok, why = run_tests(work, a.tests, a.timeout)
        if not ok:
            print(f"The tests do not pass on the code as it is, so nothing can be said: {why}")
            Path(a.out).write_text(json.dumps({"module": a.module, "baseline_fails": why}))
            return 2
        text = (work / a.module).read_text()
        tree, lines = ast.parse(text), text.splitlines()
        if a.again:
            chosen = survivors_again(tree, lines, json.loads(Path(a.again).read_text()))
        else:
            pool = [s for s in sites(tree) if s[0] in a.kinds.split(",")]
            random.Random(a.seed).shuffle(pool)
            chosen = pool[:a.n]
        results = audit(work, a.module, a.tests, chosen, a.timeout, Path(a.out))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    noticed = sum(r["killed"] for r in results)
    print(f"{a.module}: {noticed} of {len(results)} changes noticed; the {len(results) - noticed} survivors are in {a.out} for a person to read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
