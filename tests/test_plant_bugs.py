"""The bug-planting tool (``experiments/plant_bugs.py``): each kind of change is the change it says, a change the tests do not check is a survivor, one they do check is noticed, and the survivors of an
earlier run are found again by the text of their line after the file has changed."""

from __future__ import annotations

import ast
import json
import textwrap

from experiments import plant_bugs as P

SOURCE = textwrap.dedent('''\
    """A docstring with the number 42 in it."""


    def classify(x):
        if x > 10 and not x == 99:
            return "big"
        return "small"


    def area(w, h):
        return w * h - 0.5


    def untested(a, b):
        return a - b + 7
    ''')


def _after(kind: str, line_text: str) -> list[str]:
    """The module with each change of this kind on the lines that contain ``line_text``, one version for each."""
    tree = ast.parse(SOURCE)
    lines = SOURCE.splitlines()
    return [ast.unparse(P.mutate(tree, k, order)) for k, order, line in P.sites(tree) if k == kind and line_text in lines[line - 1]]


def test_each_kind_of_change_is_the_change_it_says_and_a_docstring_is_left_alone():
    assert any("x >= 10" in v for v in _after("compare", "if x > 10")) and any("x != 99" in v for v in _after("compare", "if x > 10"))
    assert any("w / h" in v for v in _after("arith", "w * h")) and any("w * h + 0.5" in v for v in _after("arith", "w * h"))
    assert any("x > 10 or" in v for v in _after("bool", "if x > 10"))
    assert any("if x > 10 and x == 99" in v for v in _after("unary", "if x > 10"))                         # ``not`` is taken off
    assert any("w * h - 0.75" in v for v in _after("number", "w * h"))                                    # a float is made half as large again
    assert any("a - b + 8" in v for v in _after("number", "a - b"))                                       # an integer is made one more
    found = P.sites(ast.parse(SOURCE))
    assert not any(line == 1 for _, _, line in found) and [k for k, _, _ in found].count("number") == 4    # 10, 99, 0.5 and 7: not the 42 in the docstring


def test_a_change_the_tests_check_is_noticed_one_they_do_not_is_a_survivor_and_the_survivors_are_found_again(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "toy.py").write_text(SOURCE)
    (repo / "tests" / "test_toy.py").write_text(textwrap.dedent('''\
        from src.toy import area, classify


        def test_classify():
            assert classify(11) == "big" and classify(5) == "small" and classify(99) == "small"


        def test_area():
            assert area(3, 4) == 11.5
        '''))
    first = tmp_path / "first.json"
    assert P.main(["src/toy.py", "tests/test_toy.py", "--n", "100", "--repo", str(repo), "--out", str(first), "--timeout", "120"]) == 0
    record = json.loads(first.read_text())["results"]
    by_line = {}
    for r in record:
        by_line.setdefault(r["source"], []).append(r["killed"])
    assert all(by_line["return w * h - 0.5"])                                                              # every change to the area is noticed
    assert not any(by_line["return a - b + 7"])                                                            # nothing checks the function nobody tests
    assert sum(not r["killed"] for r in record) >= 3
    (repo / "tests" / "test_toy.py").write_text((repo / "tests" / "test_toy.py").read_text() + "\n\ndef test_untested():\n    from src.toy import untested\n    assert untested(5, 2) == 10\n")
    second = tmp_path / "second.json"
    assert P.main(["src/toy.py", "tests/test_toy.py", "--again", str(first), "--repo", str(repo), "--out", str(second), "--timeout", "120"]) == 0
    again = json.loads(second.read_text())["results"]
    assert again and {r["source"] for r in again if r["killed"]} >= {"return a - b + 7"}                    # the new test notices what the old ones did not
    assert (repo / "src" / "toy.py").read_text() == SOURCE                                                 # the working tree is never changed


def test_a_suite_that_fails_on_the_code_as_it_is_says_nothing(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "toy.py").write_text("def f():\n    return 1 + 2\n")
    (repo / "tests" / "test_toy.py").write_text("from src.toy import f\n\n\ndef test_f():\n    assert f() == 4\n")
    out = tmp_path / "out.json"
    assert P.main(["src/toy.py", "tests/test_toy.py", "--repo", str(repo), "--out", str(out), "--timeout", "120"]) == 2
    assert "baseline_fails" in json.loads(out.read_text())
