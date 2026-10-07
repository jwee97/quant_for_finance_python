"""Formula strategies and the user_strategies folder: correct values, safe parsing, causality, clear errors, templates that actually run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, Pipeline, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.strategies.expression import EXAMPLES, FUNCTIONS, FormulaError, evaluate, parse
from src.strategies.user import load_user_strategies, new_strategy
from src.utils.config import load_config

load_library()


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(3)
    n = 1800
    idx = pd.bdate_range("2010-01-04", periods=n)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (n, 6)), axis=0), index=idx, columns=list("ABCDEF"))
    macro = pd.DataFrame({"VIX": 20 + np.cumsum(rng.normal(0, 0.3, n))}, index=idx)
    return bundle_from_prices(prices, macro=macro, name="formula")


def test_functions_match_their_textbook_definitions(bundle):
    p = bundle.prices
    assert np.allclose(evaluate("ret(21)", bundle).dropna(), (p / p.shift(21) - 1).dropna())
    assert np.allclose(evaluate("mom(252, 21)", bundle).dropna(), (p.shift(21) / p.shift(252) - 1).dropna())
    assert np.allclose(evaluate("sma(50)", bundle).dropna(), p.rolling(50).mean().dropna())
    assert np.allclose(evaluate("vol(20)", bundle).dropna(), (p.pct_change().rolling(20).std() * np.sqrt(252)).dropna())
    r = evaluate("rank(ret(10))", bundle).dropna(how="all")
    assert r.min().min() > 0 and r.max().max() <= 1.0
    z = evaluate("zs(ret(10))", bundle).dropna(how="all")
    assert np.allclose(z.mean(axis=1), 0, atol=1e-9)
    rsi = evaluate("rsi(14)", bundle).dropna(how="all")
    assert rsi.min().min() >= 0 and rsi.max().max() <= 100


def test_conditions_and_macro_series_work(bundle):
    trend = evaluate("where(close > sma(100), 1, -1)", bundle).dropna(how="all")
    assert set(np.unique(trend.to_numpy()[~np.isnan(trend.to_numpy())])) == {-1.0, 1.0}
    both = evaluate("where((close > sma(50)) & (close > sma(200)), 1, 0)", bundle).dropna(how="all")
    assert set(np.unique(both.to_numpy()[~np.isnan(both.to_numpy())])) <= {0.0, 1.0}
    vix = evaluate('macro("VIX") * 1.0', bundle)
    assert np.allclose(vix["A"].dropna(), bundle.macro["VIX"].reindex(vix.index).dropna()) and (vix["A"] == vix["B"]).all()


@pytest.mark.parametrize("name,formula,mode", EXAMPLES)
def test_every_shipped_example_runs_and_is_causal(name, formula, mode, bundle):
    model = MODELS.create("expression", expr=formula, mode=mode)
    assert check_causality(model, bundle, cutoff=bundle.index[1300])["ok"], name
    assert model.score(bundle).stack().notna().sum() > 500


@pytest.mark.parametrize("formula,message", [
    ("__import__('os').system('x')", "positional"),
    ("close.sum()", "positional"),
    ("close[0]", "Subscript"),
    ("(lambda: 1)()", "positional"),
    ("[x for x in close]", "ListComp"),
    ("open('f')", "unknown name"),
    ("mom(5, 10)", "larger than skip"),
    ("ret(-1)", "whole number"),
    ("ret(0.5)", "whole number"),
    ("lag(close, -3)", "whole number"),
    ("sma(5000)", "whole number"),
    ("sma(close)", "whole number"),
    ("1 + 1", "plain number"),
    ("close // 2", "not allowed"),
    ("close % 2", "not allowed"),
    ("close ** close", "exponent"),
    ("close ** 9", "exponent"),
    ("close > 1 > 0", "chain"),
    ("not close", "unary"),
    ("close if 1 else 2", "IfExp"),
    ("sma(n=5)", "positional"),
    ("macro('NOPE')", "no series"),
    ("close & 1", "conditions"),
    ("where(1, 2, 3)", "condition"),
    ("lag(5, 2)", "table"),
    ("", "empty"),
    ("mom(", "syntax"),
    ("x" * 500, "longer than"),
    ("close.__class__", "Attribute"),
])
def test_unsafe_or_wrong_formulas_are_refused_with_a_reason(formula, message, bundle):
    with pytest.raises(FormulaError, match=message):
        evaluate(formula, bundle)


def test_bad_parameters_are_rejected_when_the_model_is_built():
    with pytest.raises(FormulaError):
        MODELS.create("expression", expr="eval('1')")
    with pytest.raises(ValueError, match="mode must be"):
        MODELS.create("expression", expr="mom(63)", mode="sideways")


def test_a_formula_runs_through_the_pipeline_like_any_strategy(bundle):
    spec = {"name": "f", "models": [{"name": "expression", "params": {"expr": "rank(mom(126, 21)) - rank(vol(63))"}}], "evaluation": {"causality": True, "benchmarks": []}}
    out = Pipeline(spec, load_config(), bundle).run()
    assert out.validation["causality"]["expression"]["ok"] and np.isfinite(out.metrics["sharpe"])


def test_every_function_in_the_reference_table_is_implemented(bundle):
    examples = {"close": "close", "macro": 'macro("VIX")'}
    for name, (signature, _) in FUNCTIONS.items():
        if name in examples:
            parse(examples[name])
        else:
            assert name in signature


# ----------------------------------------------------------------------------------------------------- user strategy files
def test_the_template_is_valid_python_that_registers_and_runs(tmp_path, bundle):
    path = new_strategy(tmp_path, "my_test_idea_one")
    assert path.exists()
    result = load_user_strategies(tmp_path)
    try:
        assert result == {"my_test_idea_one.py": ""} and "my_test_idea_one" in MODELS
        model = MODELS.create("my_test_idea_one", window=21)
        assert check_causality(model, bundle, cutoff=bundle.index[1300])["ok"]
    finally:
        MODELS.unregister("my_test_idea_one")


def test_reloading_after_an_edit_replaces_the_model_instead_of_colliding(tmp_path):
    path = new_strategy(tmp_path, "my_test_idea_two")
    try:
        load_user_strategies(tmp_path)
        path.write_text(path.read_text().replace("window: int = 63", "window: int = 10"))
        assert load_user_strategies(tmp_path, reload=True) == {"my_test_idea_two.py": ""}
        assert MODELS.create("my_test_idea_two").window == 10
    finally:
        MODELS.unregister("my_test_idea_two")


def test_a_broken_file_is_reported_and_does_not_stop_the_others(tmp_path):
    new_strategy(tmp_path, "my_test_idea_three")
    (tmp_path / "user_strategies" / "broken.py").write_text("this is not python (")
    (tmp_path / "user_strategies" / "_skipped.py").write_text("raise RuntimeError('never imported')")
    try:
        result = load_user_strategies(tmp_path)
        assert result["my_test_idea_three.py"] == "" and "SyntaxError" in result["broken.py"] and "_skipped.py" not in result
    finally:
        MODELS.unregister("my_test_idea_three")


@pytest.mark.parametrize("name", ["Bad Name", "x", "1abc", "has-dash", "UPPER", "a" * 60, ""])
def test_new_strategy_rejects_bad_names(name, tmp_path):
    with pytest.raises(ValueError):
        new_strategy(tmp_path, name)


def test_new_strategy_never_overwrites(tmp_path):
    new_strategy(tmp_path, "my_test_idea_four")
    with pytest.raises(FileExistsError):
        new_strategy(tmp_path, "my_test_idea_four")


def test_no_folder_means_nothing_to_load(tmp_path):
    assert load_user_strategies(tmp_path) == {}
