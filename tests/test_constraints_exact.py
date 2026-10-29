"""Position, group, leverage and turnover limits, checked against numbers worked out by hand.

A planted-bug audit (``experiments/plant_bugs.py``) found ``portfolio/constraints.py`` weakly pinned: of the places where one operator, comparison or number was changed the tests noticed 24% (34 of 141).
Every optimiser and every heuristic book is held to these limits, so what counts as feasible, what a breach is, and where a projection lands are stated here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.constraints import Constraints, apply_turnover_limit, project_frame


class Config:
    def __init__(self, node=None, group_map=None):
        self.node, self.group_map = node, group_map or {}

    def get(self, key, default=None):
        return self.node if key == "portfolio.constraints" and self.node is not None else default


def series(**w) -> pd.Series:
    return pd.Series(w, dtype=float)


# ------------------------------------------------------------------------------------------------------------------ what the set is
def test_the_default_limits_and_the_ones_read_from_configuration():
    c = Constraints()
    assert (c.min_weight, c.max_weight, c.gross_leverage, c.net_exposure, c.group_limits, c.group_map, c.max_turnover, c.long_only) == (0.0, 0.25, 1.0, 1.0, {}, {}, None, True)
    bare = Constraints.from_config(Config(None, {"A": "eq"}))
    assert (bare.min_weight, bare.max_weight, bare.gross_leverage, bare.net_exposure, bare.group_limits, bare.group_map, bare.max_turnover, bare.long_only) == (0.0, 0.25, 1.0, 1.0, {}, {"A": "eq"}, None, True)
    node = {"min_weight": -0.1, "max_weight": 0.3, "gross_leverage": 1.5, "group_limits": {"eq": 0.6}, "max_turnover": 0.2}
    given = Constraints.from_config(Config(node, {"A": "eq"}), {"B": "bonds"})
    assert (given.min_weight, given.max_weight, given.gross_leverage, given.net_exposure, given.group_limits, given.group_map, given.max_turnover, given.long_only) == (-0.1, 0.3, 1.5, 1.0, {"eq": 0.6}, {"B": "bonds"}, 0.2, False)
    assert Constraints.from_config(Config(node, {"A": "eq"})).group_map == {"A": "eq"}, "the configuration's own groups unless some are given"
    assert Constraints.from_config(Config({"min_weight": 0}, {})).long_only is True and Constraints.from_config(Config({"min_weight": -0.0001}, {})).long_only is False
    assert Constraints.from_config(Config({"gross_leverage": None}, {})).gross_leverage is None


def test_bounds_and_the_groups_that_have_a_limit():
    c = Constraints(min_weight=0.05, max_weight=0.3, group_limits={"eq": 0.5, "bonds": 0.4}, group_map={"A": "eq", "B": "eq", "C": "bonds", "D": "other", "E": "eq"})
    assert c.bounds(["A", "B", "C"]) == [(0.05, 0.3)] * 3 and c.bounds([]) == []
    assert c.groups(["A", "B", "C", "D", "Z"]) == {"eq": [0, 1], "bonds": [2]}, "positions in the order given; D's group has no limit and Z has no group"
    assert c.groups(["E", "A"]) == {"eq": [0, 1]} and Constraints().groups(["A"]) == {}


# ------------------------------------------------------------------------------------------------------------------ can any book satisfy them
def test_a_set_of_limits_is_feasible_only_if_a_book_can_reach_the_net_exposure_within_them():
    assert Constraints().is_feasible([]) == (False, "no assets")
    assert Constraints().is_feasible(["A", "B", "C"]) == (False, "3 assets capped at 25% cannot reach a net exposure of 100%")
    assert Constraints().is_feasible(list("ABCD")) == (True, "feasible"), "four at a quarter make exactly a whole"
    assert Constraints(net_exposure=None).is_feasible(["A", "B", "C"])[0] is False, "no target means a target of one"
    assert Constraints(max_weight=0.5, net_exposure=2.0).is_feasible(["A", "B", "C"]) == (False, "3 assets capped at 50% cannot reach a net exposure of 200%")
    assert Constraints(max_weight=1.0, net_exposure=0.5).is_feasible(["A"]) == (True, "feasible")
    assert Constraints(min_weight=0.4, max_weight=1.0).is_feasible(["A", "B", "C"]) == (False, "minimum weights already exceed the net exposure target")
    assert Constraints(min_weight=0.25, max_weight=0.25).is_feasible(list("ABCD")) == (True, "feasible"), "four at exactly a quarter"
    assert Constraints(min_weight=0.2, max_weight=0.3, net_exposure=0.5).is_feasible(list("ABC"))[0] is False


def test_group_limits_can_make_a_book_unreachable():
    groups = {"A": "eq", "B": "eq", "C": "eq"}
    c = Constraints(group_limits={"eq": 0.3}, group_map=groups)
    assert c.is_feasible(list("ABCDE")) == (False, "group limits cap total investable exposure at 80%, below the 100% target"), "0.3 in three names and a quarter in each of two more"
    assert Constraints(group_limits={"eq": 0.5}, group_map=groups).is_feasible(list("ABCDE")) == (True, "feasible"), "0.5 and 0.5 is exactly a whole"
    assert Constraints(group_limits={"eq": 0.9}, group_map=groups).is_feasible(list("ABCDE")) == (True, "feasible"), "a limit above what the members can hold at the cap changes nothing"
    assert Constraints(group_limits={"eq": 0.3, "bonds": 0.1}, group_map={**groups, "D": "bonds"}).is_feasible(list("ABCDE"))[1] == "group limits cap total investable exposure at 65%, below the 100% target"
    assert c.is_feasible(list("ABCDEF")) == (True, "feasible"), "a sixth name makes up the difference"
    assert Constraints(group_limits={"eq": 0.3}, group_map={"X": "eq"}).is_feasible(list("ABCD")) == (True, "feasible"), "a group with no member among the assets is not counted"
    mixed = Constraints(group_limits={"eq": 0.1, "bonds": 0.9}, group_map={"A": "eq", "B": "eq", "C": "bonds", "D": "bonds"})
    assert mixed.is_feasible(list("ABCD")) == (False, "group limits cap total investable exposure at 60%, below the 100% target"), "a tenth for the first pair; the second pair holds only what the cap lets it, a half, however high its limit"


# ------------------------------------------------------------------------------------------------------------------ the form optimisers read
def test_the_constraints_in_scipys_form_evaluate_to_the_slack_of_each_limit():
    c = Constraints(gross_leverage=1.5, net_exposure=1.0, group_limits={"eq": 0.4}, group_map={"A": "eq", "B": "eq"}, max_turnover=0.3, long_only=False)
    previous = np.array([0.2, 0.2, 0.6])
    cons = c.scipy_constraints(["A", "B", "C"], previous)
    assert [k["type"] for k in cons] == ["eq", "ineq", "ineq", "ineq"]
    w = np.array([0.3, 0.2, 0.7])
    assert cons[0]["fun"](w) == pytest.approx(0.2) and cons[1]["fun"](w) == pytest.approx(1.5 - 1.2) and cons[2]["fun"](w) == pytest.approx(0.4 - 0.5), "net 1.2 against 1, gross 1.2 against 1.5, group 0.5 against 0.4"
    assert cons[3]["fun"](w) == pytest.approx(0.3 - (0.1 + 0.0 + 0.1)), "a turnover of 0.2 against 0.3"
    short = np.array([0.6, -0.3, 0.9])
    assert cons[1]["fun"](short) == pytest.approx(1.5 - 1.8), "gross counts shorts"
    assert [k["type"] for k in c.scipy_constraints(["A", "B", "C"])] == ["eq", "ineq", "ineq"], "no turnover limit without a previous book"
    assert [k["type"] for k in Constraints(max_turnover=0.3).scipy_constraints(["A"], previous)] == ["eq", "ineq"], "long only: no gross constraint (the net does it)"
    assert Constraints(net_exposure=None, gross_leverage=None, long_only=False).scipy_constraints(["A"]) == []
    assert [k["type"] for k in Constraints(gross_leverage=None, long_only=False).scipy_constraints(["A"])] == ["eq"]
    assert Constraints(long_only=False, net_exposure=0.5).scipy_constraints(["A"])[0]["fun"](np.array([0.5, 0.25])) == pytest.approx(0.25), "the target of the equality is the net exposure"


# ------------------------------------------------------------------------------------------------------------------ what a breach is
def test_violations_report_by_how_much_each_limit_is_broken():
    c = Constraints(min_weight=0.0, max_weight=0.25, gross_leverage=1.0, net_exposure=1.0, group_limits={"eq": 0.4, "bonds": 0.5}, group_map={"A": "eq", "B": "eq", "C": "bonds"})
    out = c.violations(series(A=0.35, B=0.3, C=-0.05, D=0.4))
    assert out == pytest.approx({"max_weight": 0.1 + 0.05 + 0.15, "min_weight": 0.05, "gross_leverage": 0.1, "group_eq": 0.25}), "net is 1.0 and the bonds group is under its limit, so neither is reported"
    assert c.violations(series(A=0.15, B=0.15, C=0.25, D=0.25, E=0.1, F=0.1)) == {}
    net = c.violations(series(A=0.25, B=0.15, C=0.25, D=0.2))
    assert net == pytest.approx({"net_exposure": -0.15}), "a book 15 points short of its net exposure"
    assert Constraints(net_exposure=None, gross_leverage=None).violations(series(A=0.2, B=0.1)) == {}, "limits that are not set are not broken"


def test_a_breach_smaller_than_the_tolerance_is_rounding():
    """Each tolerance is probed between its value and half as much again, where a loosened tolerance would give a different answer."""
    c = Constraints(min_weight=0.0, max_weight=0.25, gross_leverage=1.0, net_exposure=1.0, group_limits={"eq": 0.5}, group_map={"A": "eq", "B": "eq"})
    base = dict(A=0.25, B=0.25, C=0.25, D=0.25)
    assert c.violations(series(**base)) == {}
    assert c.violations(series(**{**base, "A": 0.25 + 0.9e-9, "C": 0.25 - 0.9e-9})) == {}, "just under a billionth over the cap is rounding"
    over = c.violations(series(**{**base, "A": 0.25 + 1.2e-9, "C": 0.25 - 1.2e-9}))
    assert list(over) == ["max_weight"] and over["max_weight"] == pytest.approx(1.2e-9), "and just over it is a breach"
    assert "min_weight" not in c.violations(series(**{**base, "A": -0.9e-9, "B": 0.5 + 0.9e-9})) and c.violations(series(**{**base, "A": -1.2e-9, "B": 0.5 + 1.2e-9}))["min_weight"] == pytest.approx(1.2e-9)
    assert "net_exposure" not in c.violations(series(**{**base, "C": 0.25 + 0.9e-6})) and c.violations(series(**{**base, "C": 0.25 + 1.2e-6}))["net_exposure"] == pytest.approx(1.2e-6), "a millionth of net exposure is the tolerance"
    assert "gross_leverage" not in c.violations(series(**{**base, "C": 0.25 + 0.9e-6, "D": 0.25 - 0.9e-6 + 0.9e-6}))
    long_short = Constraints(min_weight=-1.0, max_weight=1.0, gross_leverage=1.0, net_exposure=None)
    assert long_short.violations(series(A=0.5, B=0.5 + 0.9e-6)) == {} and long_short.violations(series(A=0.5, B=0.5 + 1.2e-6))["gross_leverage"] == pytest.approx(1.2e-6)
    tight = Constraints(min_weight=0.0, max_weight=1.0, gross_leverage=None, net_exposure=None, group_limits={"eq": 0.5}, group_map={"A": "eq", "B": "eq"})
    assert tight.violations(series(A=0.25, B=0.25 + 0.9e-6)) == {} and tight.violations(series(A=0.25, B=0.25 + 1.2e-6))["group_eq"] == pytest.approx(1.2e-6)
    assert Constraints(group_limits={"cash": 0.1}, group_map={}).violations(series(A=1.0)).get("group_cash") is None, "a group with no members holds nothing"


def test_feasibility_is_judged_to_a_billionth():
    four = list("ABCD")
    assert Constraints(max_weight=(1 - 0.9e-9) / 4).is_feasible(four)[0] is True and Constraints(max_weight=(1 - 1.2e-9) / 4).is_feasible(four)[0] is False
    assert Constraints(min_weight=(1 + 0.9e-9) / 4, max_weight=1.0).is_feasible(four)[0] is True and Constraints(min_weight=(1 + 1.2e-9) / 4, max_weight=1.0).is_feasible(four)[0] is False
    groups = {"A": "eq", "B": "eq", "C": "eq"}
    five = list("ABCDE")
    assert Constraints(group_limits={"eq": 0.5 - 0.9e-9}, group_map=groups).is_feasible(five)[0] is True
    assert Constraints(group_limits={"eq": 0.5 - 1.2e-9}, group_map=groups).is_feasible(five)[0] is False


# ------------------------------------------------------------------------------------------------------------------ projecting onto the set
def test_a_book_over_the_cap_is_clipped_and_the_excess_goes_to_the_names_with_room():
    c = Constraints()
    assert c.project(series(A=0.6, B=0.2, C=0.1, D=0.1)).tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert c.project(series(A=0.25, B=0.25, C=0.25, D=0.25, E=0.25)).tolist() == pytest.approx([0.2] * 5), "a book that is too big is trimmed in proportion to what it holds above the minimum"
    assert c.project(series(A=0.4, B=0.4, C=0.1, D=0.1)).tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25]) and c.project(series(A=0.4, B=0.4, C=0.1, D=0.1)).sum() == pytest.approx(1.0)
    short = c.project(series(A=0.1, B=0.1, C=0.1, D=0.1))
    assert short.tolist() == pytest.approx([0.25] * 4), "a book that is too small is topped up in proportion to the room under the cap"
    assert c.project(series(A=0.0, B=0.0, C=0.0)).tolist() == [0.0, 0.0, 0.0], "nothing is not scaled up from nothing"
    uneven = c.project(series(A=0.1, B=0.2, C=0.1, D=0.2, E=0.1, F=0.1))
    assert uneven.sum() == pytest.approx(1.0) and uneven.max() <= 0.25 + 1e-12 and uneven.min() >= 0.0
    assert uneven["A"] == pytest.approx(0.1 + (0.25 - 0.1) / (0.15 * 4 + 0.05 * 2) * 0.2), "the 0.2 short is shared by headroom: 0.15 for the 0.1 names, 0.05 for the 0.2 names"
    assert uneven["B"] == pytest.approx(0.2 + 0.05 / 0.7 * 0.2)
    holdings = series(A=0.5, B=0.5, C=0.5)
    assert Constraints(min_weight=0.1, max_weight=0.6).project(holdings).tolist() == pytest.approx([1 / 3] * 3), "three halves become a third each: trimmed in proportion to what each holds above the 0.1 minimum"
    assert holdings.tolist() == [0.5, 0.5, 0.5], "the input is not changed"
    assert Constraints().project(series(A=0.6, B=0.2, C=0.1, D=0.1), iterations=1).tolist() == pytest.approx([0.25] * 4)


def test_without_a_net_exposure_target_a_book_is_only_clipped_and_a_net_of_zero_is_left_alone():
    clipped = Constraints(net_exposure=None, max_weight=0.5).project(series(A=0.6, B=0.2))
    assert clipped.tolist() == pytest.approx([0.5, 0.2]), "no target to restore"
    neutral = Constraints(min_weight=-0.3, max_weight=0.3, net_exposure=0.0, gross_leverage=None, long_only=False).project(series(A=0.5, B=-0.5, C=0.0))
    assert neutral.tolist() == pytest.approx([0.3, -0.3, 0.0])
    assert Constraints(net_exposure=1.0).project(series(A=0.0, B=0.0)).tolist() == [0.0, 0.0], "a total of nothing is left as nothing, even with a target of one"


def test_group_caps_scale_the_members_and_the_projection_ends_inside_every_limit():
    groups = {"A": "eq", "B": "eq"}
    c = Constraints(group_limits={"eq": 0.4}, group_map=groups)
    out = c.project(series(A=0.25, B=0.25, C=0.25, D=0.25, E=0.0, F=0.0))
    assert c.violations(out) == {} and out.sum() == pytest.approx(1.0) and out["A"] + out["B"] <= 0.4 + 1e-9 and out.max() <= 0.25 + 1e-12 and out.min() >= -1e-12
    assert out["E"] > 0.0 and out["F"] > 0.0, "the exposure taken from the group goes to names with room"
    ok = series(A=0.15, B=0.15, C=0.25, D=0.25, E=0.1, F=0.1)
    assert c.project(ok).tolist() == pytest.approx(ok.tolist()), "a book that already satisfies everything stays where it is"
    assert Constraints(group_limits={"cash": 0.1}, group_map={}).project(series(A=0.5, B=0.5), iterations=1).tolist() == pytest.approx([0.25, 0.25]), "a group without members is skipped"


def test_the_projection_corrects_a_shortfall_or_an_excess_of_a_trillionth_and_ignores_a_total_of_nothing():
    quarter = 0.25
    short = Constraints().project(series(A=quarter, B=quarter, C=quarter, D=quarter - 1.2e-12))
    assert abs(short.sum() - 1.0) < 1e-13 and short["D"] == pytest.approx(quarter, abs=1e-14), "a shortfall of 1.2e-12 is made up"
    over = Constraints(max_weight=0.3).project(series(A=0.3, B=0.3, C=0.2, D=0.2 + 1.2e-12))
    assert abs(over.sum() - 1.0) < 1e-13, "and an excess of it taken off"
    floor = Constraints(min_weight=0.25, max_weight=0.3).project(series(A=quarter, B=quarter, C=quarter, D=quarter + 1.2e-12))
    assert abs(floor.sum() - 1.0) < 1e-13 and floor["D"] == pytest.approx(quarter, abs=1e-14), "taken off the room above the minimum, however little there is"
    assert Constraints().project(series(A=1.2e-12, B=0.0, C=0.0, D=0.0)).sum() == pytest.approx(1.0), "a total of more than a trillionth is a book"
    assert Constraints().project(series(A=0.9e-12, B=0.0, C=0.0, D=0.0)).sum() == pytest.approx(0.9e-12), "and one of less is nothing"
    boxed = Constraints(max_weight=1.0, group_limits={"eq": 0.5}, group_map={"A": "eq", "B": "eq"}, net_exposure=None).project(series(A=0.25, B=0.25 + 1.2e-12, C=0.1))
    assert boxed["A"] + boxed["B"] == pytest.approx(0.5, abs=1e-14), "a group a trillionth over its limit is brought back"
    unmoved = Constraints(max_weight=1.0, group_limits={"eq": 0.5}, group_map={"A": "eq", "B": "eq"}, net_exposure=None).project(series(A=0.25, B=0.25 + 0.9e-12, C=0.1))
    assert unmoved["B"] == pytest.approx(0.25 + 0.9e-12, abs=1e-16), "and one under a trillionth over is left alone"


def test_the_table_projection_has_the_same_trillionth_edges_as_the_row_projection():
    idx = pd.bdate_range("2021-01-04", periods=1)
    cols = list("ABCD")

    def table(*row):
        return pd.DataFrame([list(row)], index=idx, columns=cols)

    quarter = 0.25
    assert abs(project_frame(table(quarter, quarter, quarter, quarter - 1.2e-12), Constraints()).iloc[0].sum() - 1.0) < 1e-13
    assert abs(project_frame(table(0.3, 0.3, 0.2, 0.2 + 1.2e-12), Constraints(max_weight=0.3)).iloc[0].sum() - 1.0) < 1e-13
    assert abs(project_frame(table(quarter, quarter, quarter, quarter + 1.2e-12), Constraints(min_weight=0.25, max_weight=0.3)).iloc[0].sum() - 1.0) < 1e-13
    assert project_frame(table(1.2e-12, 0.0, 0.0, 0.0), Constraints()).iloc[0].sum() == pytest.approx(1.0) and project_frame(table(0.9e-12, 0.0, 0.0, 0.0), Constraints()).iloc[0].sum() == pytest.approx(0.9e-12)
    grouped = Constraints(max_weight=1.0, group_limits={"eq": 0.5}, group_map={"A": "eq", "B": "eq"}, net_exposure=None)
    assert project_frame(table(0.25, 0.25 + 1.2e-12, 0.1, 0.0), grouped).iloc[0, :2].sum() == pytest.approx(0.5, abs=1e-14)
    assert project_frame(table(0.25, 0.25 + 0.9e-12, 0.1, 0.0), grouped).iloc[0, 1] == pytest.approx(0.25 + 0.9e-12, abs=1e-16)


# ------------------------------------------------------------------------------------------------------------------ projecting a whole table
def test_the_table_projection_agrees_with_the_row_by_row_one_and_leaves_what_is_not_a_book_alone():
    rng = np.random.default_rng(4)
    cols = list("ABCDEF")
    c = Constraints(group_limits={"eq": 0.45}, group_map={"A": "eq", "B": "eq"})
    book = pd.DataFrame(rng.dirichlet(np.ones(6), 25) * rng.uniform(0.6, 1.4, (25, 1)), index=pd.bdate_range("2021-01-04", periods=25), columns=cols)
    out = project_frame(book, c)
    for i in (0, 7, 24):
        assert out.iloc[i].tolist() == pytest.approx(c.project(book.iloc[i]).tolist(), abs=1e-7)
    assert all(c.violations(out.iloc[i]) == {} for i in range(25)) and np.allclose(out.sum(axis=1), 1.0, atol=1e-8)
    flat = book.copy()
    flat.iloc[3] = 0.0
    flat.iloc[5, 2] = np.nan
    out = project_frame(flat, c)
    assert (out.iloc[3] == 0.0).all(), "a date with nothing investable stays flat"
    assert np.isnan(out.iloc[5, 2]) and np.isfinite(out.iloc[5].drop("C").to_numpy()).all(), "a missing weight stays missing"
    assert project_frame(book * 0.0, c).equals(book * 0.0)
    tiny = pd.DataFrame([[1.2e-12, 0, 0, 0, 0, 0], [1e-13, 0, 0, 0, 0, 0]], index=book.index[:2], columns=cols)
    projected = project_frame(tiny, Constraints())
    assert projected.iloc[0].sum() == pytest.approx(1.0) and projected.iloc[1].sum() == pytest.approx(1e-13), "a row that sums to more than a trillionth is a book and is scaled; one that does not is nothing"


def test_the_table_projection_without_a_net_target_only_clips_and_caps_groups_and_respects_iterations():
    book = pd.DataFrame([[0.6, 0.2, 0.1], [0.3, 0.3, 0.3]], index=pd.bdate_range("2021-01-04", periods=2), columns=list("ABC"))
    clipped = project_frame(book, Constraints(net_exposure=None, max_weight=0.5))
    assert clipped.iloc[0].tolist() == pytest.approx([0.5, 0.2, 0.1]) and clipped.iloc[1].tolist() == pytest.approx([0.3, 0.3, 0.3])
    grouped = project_frame(book, Constraints(net_exposure=None, max_weight=1.0, group_limits={"eq": 0.4}, group_map={"A": "eq", "B": "eq"}))
    assert grouped.iloc[0].tolist() == pytest.approx([0.6 * 0.4 / 0.8, 0.2 * 0.4 / 0.8, 0.1]) and grouped.iloc[1].tolist() == pytest.approx([0.2, 0.2, 0.3]), "the group is scaled to its limit and the rest left"
    low = pd.DataFrame([[0.5, 0.5, 0.5]], index=book.index[:1], columns=list("ABC"))
    assert project_frame(low, Constraints(min_weight=0.1, max_weight=0.6)).iloc[0].tolist() == pytest.approx([1 / 3] * 3), "trimmed towards the minimum when the book is too big"
    uneven = pd.DataFrame([[0.3, 0.3, 0.3, 0.2, 0.2]], index=book.index[:1], columns=list("ABCDE"))
    floored = Constraints(min_weight=0.1, max_weight=0.6)
    assert project_frame(uneven, floored).iloc[0].tolist() == pytest.approx([0.225] * 3 + [0.1625] * 2), "0.3 too much, taken from each name in proportion to what it holds above the floor of a tenth: 0.2 and 0.1"
    assert floored.project(uneven.iloc[0]).tolist() == pytest.approx([0.225] * 3 + [0.1625] * 2), "the same as the row at a time"
    one_pass = project_frame(book.iloc[:1], Constraints(max_weight=0.5), iterations=1)
    assert one_pass.iloc[0].tolist() == pytest.approx([0.5, 0.2 + 0.3 * 0.2 / 0.7, 0.1 + 0.4 * 0.2 / 0.7]), "one pass: clip the first to a half, then share the missing 0.2 by headroom (0.3 and 0.4)"
    assert Constraints(max_weight=0.5).project(book.iloc[0], iterations=1).tolist() == pytest.approx(one_pass.iloc[0].tolist()), "the same as the row at a time"
    unreachable = project_frame(book.iloc[:1], Constraints(max_weight=0.25))
    assert unreachable.iloc[0].tolist() == pytest.approx([0.25, 0.25, 0.25]), "three names at a quarter: capped, and short of the target, not over the cap"


# ------------------------------------------------------------------------------------------------------------------ turnover
def test_a_trade_a_trillionth_over_the_limit_is_cut_and_nothing_of_a_trade_is_left_alone():
    previous = series(A=0.5, B=0.5)
    target = series(A=0.5 + 0.3, B=0.5 - 0.3)                                                   # a turnover of 0.6
    over = apply_turnover_limit(target, previous, 0.6 - 1.2e-12)
    assert over is not target and float((over - previous).abs().sum()) == pytest.approx(0.6 - 1.2e-12, abs=1e-14), "a trillionth over the limit is over it"
    assert apply_turnover_limit(target, previous, 0.6 - 0.9e-12) is target, "and a little under that is within it"
    nudge = series(A=0.5 + 1.2e-12, B=0.5)
    assert apply_turnover_limit(nudge, previous, 0.0).tolist() == [0.5, 0.5], "a trade of more than a trillionth is cut to the limit, here nothing at all"
    assert apply_turnover_limit(series(A=0.5 + 0.9e-12, B=0.5), previous, 0.0).tolist() == [0.5 + 0.9e-12, 0.5], "and one of less is no trade to limit"


def test_a_trade_that_is_too_large_is_cut_back_along_the_line_to_the_target():
    previous = series(A=0.5, B=0.5, C=0.0)
    target = series(A=0.2, B=0.5, C=0.3)                                                       # a turnover of 0.6
    out = apply_turnover_limit(target, previous, 0.3)
    assert out.tolist() == pytest.approx([0.5 - 0.15, 0.5, 0.15]) and float((out - previous).abs().sum()) == pytest.approx(0.3), "half the way there"
    assert apply_turnover_limit(target, previous, 0.6) is target and apply_turnover_limit(target, previous, 0.7) is target, "a trade within the limit, or exactly at it, is made in full"
    assert apply_turnover_limit(target, previous, None) is target
    assert apply_turnover_limit(target, previous, 0.6 - 1e-6).tolist() == pytest.approx((previous + (target - previous) * ((0.6 - 1e-6) / 0.6)).tolist())
    unchanged = apply_turnover_limit(previous, previous, 0.0)
    assert unchanged is previous, "nothing to trade"
    partial = apply_turnover_limit(series(A=0.2, B=0.5, C=0.3), series(A=0.5), 0.55)
    assert partial.tolist() == pytest.approx([0.35, 0.25, 0.15]), "earlier weights missing for B and C are zero: a turnover of 1.1, half of it allowed"
    missing = apply_turnover_limit(series(A=0.6, B=0.4), series(A=0.0), 0.5)
    assert missing.tolist() == pytest.approx([0.3, 0.2]), "a book with no earlier weight in B had none: a turnover of 1.0, half of it allowed"
    assert list(missing.index) == ["A", "B"]
