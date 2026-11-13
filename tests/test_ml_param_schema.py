"""The settings of every model as a schema (src/ml/param_schema.py): what the web API validates a request's model.params against, in place of a count of the settings.

For every model of the catalogue, whichever packages are installed (the schema reads none): its settings have a type and a range and its defaults lie inside them; the full default object (twenty-three
settings for PatchTST) and any valid customisation are accepted and laid over the defaults; every kind of wrong value is refused with a message that names the setting; the settings are in one canonical
form whoever sent them; and where a model's constructor or network enforces a limit itself, the schema agrees with it at the limit. The values come from the schema (tests/ml_param_cases.py), so a model
added to the catalogue is covered without being listed here."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from src.ml import models as md
from src.ml import param_schema as ps
from src.ml.config import config_hash, experiment_config
from tests.ml_param_cases import invalid_values, request_params, valid_values

INFO = {m.kind: m for m in md.MODEL_INFO}
KINDS = list(INFO)
RELATION = ("must be a multiple of", "must be at least")
DEEP = ("lstm", "gru", "tcn", "patchtst", "nbeats", "nhits", "itransformer", "dlinear", "tft", "autoencoder", "vae")
AGENTS = ("ppo", "sac", "recurrent_ppo")


def _task(kind: str) -> str:
    return INFO[kind].tasks[0]


# ---------------------------------------------------------------------------------------------------------------------- the schema covers the catalogue
@pytest.mark.parametrize("kind", KINDS)
def test_every_setting_of_a_model_has_a_type_a_range_and_a_default_inside_it(kind):
    own = {n: s for n, s in ps.schema(kind).items() if not s.extra}
    assert list(own) == list(INFO[kind].params), "the schema covers exactly the settings the catalogue lists, in its order"
    for name, s in own.items():
        where = f"{kind}.{name}"
        assert s.type in ps.TYPES and s.name == name, where
        if s.type in (ps.INT, ps.FLOAT):
            assert s.low is not None, f"{where} is a number with no lower bound"
        declared = INFO[kind].specs.get(name, {})
        if declared.get("type") == "choice":
            assert s.type == ps.CHOICE and list(s.choices) == list(declared["choices"]), where
        if declared.get("type") == "bool":
            assert s.type == ps.BOOL, where
        assert s.check(INFO[kind].params[name], where) == s.default, f"the default of {where} is not a valid value of the setting"


def test_no_range_is_declared_for_a_setting_that_does_not_exist():
    used = {"halflife_days"}
    for kind, info in INFO.items():
        used |= set(info.params) | {f"{kind}.{n}" for n in info.params}
    assert sorted(set(ps.RANGES) - used) == [], "a range for a setting no model has is a leftover"


def test_an_interval_is_read_with_its_open_and_closed_ends_and_a_range_is_described_in_words():
    assert ps.interval("[0,1)") == (0.0, 1.0, False, True) and ps.interval("(0,)") == (0.0, None, True, False) and ps.interval("[1, 64]") == (1.0, 64.0, False, False)
    with pytest.raises(ValueError):
        ps.interval("0,1")
    assert ps.schema("lstm")["dropout"].range_text() == "at least 0 and below 1" and ps.schema("lstm")["learning_rate"].range_text() == "above 0"
    assert ps.schema("tft")["alpha_high"].range_text() == "above 0.5 and below 1" and ps.schema("lstm")["epochs"].range_text() == "at least 1 and at most 10000"


# ---------------------------------------------------------------------------------------------------------------------- the defaults, in full and in part
@pytest.mark.parametrize("kind", KINDS)
def test_the_full_default_object_the_form_used_to_send_is_accepted_and_is_the_defaults(kind):
    sent = request_params(kind)
    assert set(ps.validate_params(kind, sent)) == set(INFO[kind].params), "every setting sent is a setting of the model"
    assert ps.effective_params(kind, sent) == ps.effective_params(kind, {}) == ps.effective_params(kind, None)
    assert list(ps.effective_params(kind, {})) == list(INFO[kind].params)
    assert ps.effective_params(kind, {}) == sent, "the defaults, laid over nothing, are the object the form used to send: an experiment made then and one made now have one configuration hash"


def test_the_models_the_limit_of_twenty_settings_refused_are_accepted_with_all_of_their_settings():
    over = [k for k in KINDS if len(INFO[k].params) > 20]
    assert {"patchtst", "tft", "ppo", "recurrent_ppo"} <= set(over), "these are the models the old limit refused"
    for kind in over:
        merged = ps.effective_params(kind, request_params(kind))
        assert len(merged) == len(INFO[kind].params) > 20


@pytest.mark.parametrize("kind", KINDS)
def test_a_value_the_schema_accepts_is_laid_over_the_defaults_and_nothing_else_changes(kind):
    base = ps.effective_params(kind, {})
    for name, s in ps.schema(kind).items():
        if s.extra:
            continue
        accepted = 0
        for value in valid_values(s):
            given = ps.validate_params(kind, {name: value})
            assert list(given) == [name], (kind, name, value)
            try:
                merged = ps.effective_params(kind, {name: value})
            except ps.ParamError as exc:
                assert kind in ps.RELATIONS and any(w in str(exc) for w in RELATION), f"{kind}.{name}={value!r}: {exc}"          # only a relation between two settings may refuse a value of the range
                continue
            accepted += 1
            assert merged[name] == given[name] and {k: v for k, v in merged.items() if k != name} == {k: v for k, v in base.items() if k != name}, (kind, name, value)
            if s.type in (ps.INT, ps.FLOAT):
                assert merged[name] == value, (kind, name, value)
            if s.type == ps.BOOL:
                assert merged[name] == int(value) and type(merged[name]) is int
        assert accepted, f"no valid value of {kind}.{name} could be laid over the defaults"


def test_a_partial_object_keeps_the_catalogue_order_and_the_other_defaults():
    merged = ps.effective_params("patchtst", {"d_model": 64, "dropout": 0.3, "patch_len": 8})
    assert list(merged) == list(INFO["patchtst"].params) and merged["d_model"] == 64 and merged["dropout"] == 0.3 and merged["patch_len"] == 8
    assert {k: v for k, v in merged.items() if k not in ("d_model", "dropout", "patch_len")} == {k: v for k, v in ps.effective_params("patchtst", {}).items() if k not in ("d_model", "dropout", "patch_len")}
    assert ps.validate_params("patchtst", {"d_model": 64}) == {"d_model": 64}, "validation alone adds nothing"


# ---------------------------------------------------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize("kind", KINDS)
def test_every_wrong_value_of_every_setting_is_refused_with_a_message_that_names_the_setting(kind):
    for name, s in ps.schema(kind).items():
        if s.extra:
            continue
        for value in invalid_values(s):
            with pytest.raises(ps.ParamError) as err:
                ps.validate_params(kind, {name: value})
            assert f"model.params.{name}" in str(err.value), (kind, name, value, str(err.value))


def test_the_settings_every_model_takes_are_checked_like_the_rest():
    for name, s in ps.EXTRAS.items():
        for value in valid_values(s) if s.type != ps.FLOAT else [0, 63, 252.5]:
            assert name in ps.validate_params("ridge", {name: value})
        for value in invalid_values(s) if s.type != ps.FLOAT else [-1, "abc", True, float("nan")]:
            with pytest.raises(ps.ParamError, match=f"model.params.{name}"):
                ps.validate_params("ridge", {name: value})
    assert ps.effective_params("ridge", {"columns": "ret_1, ret_lag_2", "halflife_days": 63}) == {"alpha": 10, "columns": ["ret_1", "ret_lag_2"], "halflife_days": 63}


def test_unknown_settings_and_malformed_collections_are_refused():
    with pytest.raises(ps.ParamError, match=r"model\.params\.depth is not a setting of the Ridge model; its settings are \['alpha'\]"):
        ps.validate_params("ridge", {"depth": 3})
    with pytest.raises(ps.ParamError, match=r"model\.params\.'alpha; drop' is not a setting"):
        ps.validate_params("ridge", {"alpha; drop": 1})
    for bad in ([1, 2], "alpha=1", 5, True, ("alpha", 1)):
        with pytest.raises(ps.ParamError, match="must be an object of settings"):
            ps.validate_params("ridge", bad)
    with pytest.raises(ps.ParamError, match="is not a setting"):
        ps.validate_params("ridge", {3: 1})
    with pytest.raises(ps.ParamError, match="unknown model 'oracle'"):
        ps.schema("oracle")
    with pytest.raises(ps.ParamError, match=r"is not a setting of the Linear regression model; its settings are none"):
        ps.validate_params("linear", {"alpha": 1})
    assert ps.validate_params("linear", {}) == {} and ps.effective_params("linear", {}) == {}
    with pytest.raises(ps.ParamError, match=r"model\.params\.patch_len is not a setting of the LSTM model"):
        ps.validate_params("lstm", {"patch_len": 4})                                          # a setting of another network is not this one's


def test_a_null_or_blank_value_leaves_the_default_but_a_blank_list_means_every_feature():
    assert ps.validate_params("ridge", {"alpha": None}) == {} and ps.validate_params("ridge", {"alpha": ""}) == {} and ps.validate_params("ridge", {"alpha": "  "}) == {}
    assert ps.validate_params("lstm", {"device": "", "resume": "", "dropout": None}) == {} and ps.effective_params("lstm", {"device": None})["device"] == "auto"
    assert ps.validate_params("lstm", {"horizons": ""}) == {"horizons": ""}, "a blank text is the empty text where the default is empty"
    with pytest.raises(ps.ParamError, match="must not be empty"):
        ps.validate_params("model_ensemble", {"members": ""})
    assert ps.effective_params("ar", {})["columns"] == list(md.AR_FEATURES) and ps.effective_params("ar", {"columns": ""})["columns"] == []
    assert ps.effective_params("ar", {"columns": "ret_1, ret_lag_2 ,ret_1"})["columns"] == ["ret_1", "ret_lag_2"]


def test_a_number_is_never_a_switch_a_switch_never_a_number_and_a_string_is_never_either():
    for kind, name in (("ridge", "alpha"), ("lstm", "epochs"), ("lstm", "dropout")):
        for bad in (True, False, "5", "1e-3", [1]):
            with pytest.raises(ps.ParamError):
                ps.validate_params(kind, {name: bad})
    for bad in ("true", "1", "yes", 2, 0.5):
        with pytest.raises(ps.ParamError, match="true or false"):
            ps.validate_params("lstm", {"resume": bad})
    assert ps.validate_params("lstm", {"resume": True, "num_threads": 2.0}) == {"resume": 1, "num_threads": 2}
    with pytest.raises(ps.ParamError, match="whole number"):
        ps.validate_params("lstm", {"epochs": 2.5})
    with pytest.raises(ps.ParamError, match="finite"):
        ps.validate_params("ridge", {"alpha": float("inf")})
    with pytest.raises(ps.ParamError, match="between -1e18 and 1e18"):
        ps.validate_params("ridge", {"alpha": 10 ** 400})                                      # an integer too large for a float must not crash the check


# ---------------------------------------------------------------------------------------------------------------------- one spelling per value
def test_one_value_has_one_spelling_whichever_client_sent_it():
    spellings = [{"alpha": 10}, {"alpha": 10.0}, {}, None, {"alpha": None}]
    assert {json.dumps(ps.effective_params("ridge", s)) for s in spellings} == {'{"alpha": 10}'}
    assert ps.validate_params("lstm", {"resume": True, "num_threads": 2.0, "device": " cpu ", "horizons": " 1, 3 "}) == {"resume": 1, "num_threads": 2, "device": "cpu", "horizons": "1, 3"}
    assert ps.canonical_number(3.0) == 3 and type(ps.canonical_number(3.0)) is int and ps.canonical_number(0.25) == 0.25 and ps.canonical_number(1e300) == 1e300


@pytest.mark.parametrize("kind", KINDS)
def test_the_full_default_object_and_no_settings_at_all_have_one_configuration_hash(kind):
    over = {"dataset": {"regime_detector": "vol_state"}}                                          # regime_ridge needs a detector to be a configuration at all
    full = experiment_config({**over, "model": {"kind": kind, "params": ps.effective_params(kind, request_params(kind))}})
    none = experiment_config({**over, "model": {"kind": kind, "params": ps.effective_params(kind, {})}})
    assert config_hash(full) == config_hash(none)


# ---------------------------------------------------------------------------------------------------------------------- the constructors agree
def _base(kind: str) -> dict:
    return ps.effective_params(kind, {"members": "ridge,lasso"} if kind == "model_ensemble" else {})            # an ensemble's default members need packages the checks below are not about


def _make(kind: str, params: dict):
    return md.make_model(kind, params, _task(kind), 0)


@pytest.mark.parametrize("kind", KINDS)
def test_every_model_is_built_from_its_validated_defaults_and_from_each_valid_value(kind):
    try:
        _make(kind, _base(kind))
    except md.ModelUnavailable:
        assert not {m["kind"]: m for m in md.model_catalog()}[kind]["available"], "a model that cannot be built here is one the catalogue lists as unavailable"
        return                                                                                    # the catalogue says so, and the API refuses it with the reason (tests/test_ml_model_params_api.py)
    for name, s in ps.schema(kind).items():
        if s.extra:
            continue
        for value in valid_values(s):
            try:
                merged = ps.effective_params(kind, {**({"members": "ridge,lasso"} if kind == "model_ensemble" else {}), name: value})
            except ps.ParamError:
                continue                                                                          # a relation between settings, tested below against the networks
            try:
                _make(kind, merged)
            except ValueError as exc:
                pytest.fail(f"{kind}.{name}={value!r} passes the schema but the model refuses it: {exc}")


def _edges(s: ps.Setting, side: str):
    """(outside, inside) pairs at the lower and the upper end of the setting's range: the value just beyond the limit and the value just within it."""
    whole = s.type == ps.INT
    step, eps = (1, 1) if whole else (0.01, 1e-3)
    out = []
    if side in ("low", "both"):
        out.append((s.low if s.low_open else s.low - step, s.low + eps if s.low_open else s.low))
    if side in ("high", "both") and s.high is not None:
        out.append((s.high if s.high_open else s.high + step, s.high - eps if s.high_open else s.high))
    return [(int(a) if whole else float(a), int(b) if whole else float(b)) for a, b in out]


LIMITS: dict[str, list] = {k: [("lookback", "low"), ("hidden", "low"), ("layers", "low"), ("epochs", "low"), ("batch_size", "low"), ("learning_rate", "low"), ("patience", "low"), ("dropout", "both"),
                               ("val_fraction", "both")] for k in DEEP}
LIMITS["tft"] += [("alpha_low", "both"), ("alpha_high", "both")]
LIMITS.update({k: [("hidden", "low"), ("layers", "low"), ("neighbors", "low"), ("epochs", "low"), ("dropout", "both")] for k in ("gcn", "gat")})
LIMITS.update({k: [("total_timesteps", "low"), ("hidden", "low"), ("episode_days", "low"), ("gross", "low"), ("gamma", "both"), ("learning_rate", "low"), ("max_weight", "both")] for k in AGENTS})
LIMITS["model_ensemble"] = [("val_fraction", "both"), ("decay_days", "low"), ("temperature", "low"), ("regimes", "both"), ("shrink", "both"), ("min_coverage", "both")]
LIMITS["hmm_regime"] = LIMITS["markov_switching"] = [("n_states", "both")]


@pytest.mark.parametrize("kind", sorted(LIMITS))
def test_where_a_models_constructor_refuses_a_value_the_schema_refuses_it_too_and_where_it_accepts_one_so_does_the_schema(kind):
    base = _base(kind)
    try:
        _make(kind, base)
    except md.ModelUnavailable as exc:
        pytest.skip(f"{kind} needs a package that is not installed here: {exc}")
    for name, side in LIMITS[kind]:
        for outside, inside in _edges(ps.schema(kind)[name], side):
            with pytest.raises(ps.ParamError):
                ps.validate_params(kind, {name: outside})
            with pytest.raises(ValueError):
                _make(kind, {**base, name: outside})
            ps.validate_params(kind, {name: inside})
            _make(kind, {**base, name: inside})


def test_the_relations_between_settings_are_the_ones_the_networks_enforce():
    pytest.importorskip("torch")
    from src.ml.deep.nets import build_net
    from src.ml.graph import GraphNet

    own = {k: dict(INFO[k].params) for k in ("patchtst", "tft")}
    for kind, change, ok, n_out in (("patchtst", {}, True, 1), ("patchtst", {"d_model": 36, "n_heads": 4}, True, 1), ("patchtst", {"d_model": 30, "n_heads": 4}, False, 1),
                                    ("patchtst", {"lookback": 8, "patch_len": 8}, True, 1), ("patchtst", {"lookback": 6, "patch_len": 8}, False, 1),
                                    ("tft", {}, True, 3), ("tft", {"hidden": 32, "n_heads": 4}, True, 3), ("tft", {"hidden": 30, "n_heads": 4}, False, 3)):
        cfg = {**own[kind], **change}
        if ok:
            ps.effective_params(kind, change)
            build_net(kind, 5, int(cfg["lookback"]), n_out, cfg)
        else:
            with pytest.raises(ps.ParamError, match="must be a multiple of|must be at least"):
                ps.effective_params(kind, change)
            with pytest.raises(ValueError):
                build_net(kind, 5, int(cfg["lookback"]), n_out, cfg)
    adjacency = np.zeros((4, 4))
    for hidden, heads, ok in ((32, 2, True), (32, 4, True), (30, 4, False)):
        if ok:
            ps.effective_params("gat", {"hidden": hidden, "heads": heads})
            GraphNet("gat", 5, adjacency, hidden, 2, 0.1, heads)
        else:
            with pytest.raises(ps.ParamError, match="must be a multiple of"):
                ps.effective_params("gat", {"hidden": hidden, "heads": heads})
            with pytest.raises(ValueError):
                GraphNet("gat", 5, adjacency, hidden, 2, 0.1, heads)


def test_an_ensembles_members_are_checked_by_the_grammar_the_count_the_kind_and_each_members_own_schema():
    ok = ps.validate_params("model_ensemble", {"members": "ridge[alpha=3], lasso ,elasticnet[alpha=0.1;l1_ratio=0.3]"})
    assert ok == {"members": "ridge[alpha=3], lasso ,elasticnet[alpha=0.1;l1_ratio=0.3]"}
    for text, why in (("ridge,lasso[alpha=-1]", r"members\[lasso\]\.alpha"), ("ridge,lstm[epochs=0]", r"members\[lstm\]\.epochs"), ("ridge,ppo", "an agent"), ("ridge,nope", "not a model that can be a member"),
                      ("ridge", "between 2 and"), ("ridge,model_ensemble", "not a model that can be a member"), ("ridge[bogus=1],lasso", r"members\[ridge\]\.bogus is not a setting"), ("ridge[alpha],lasso", "not setting=value")):
        with pytest.raises(ps.ParamError, match=why):
            ps.validate_params("model_ensemble", {"members": text})


# ---------------------------------------------------------------------------------------------------------------------- the tuner and the catalogue agree
@pytest.mark.parametrize("kind", KINDS)
def test_every_value_the_tuner_can_pick_is_a_valid_setting_and_a_valid_combination(kind):
    try:
        space = md.search_space(kind)
    except ValueError:
        return                                                                                    # a model with no tuning space (CatBoost) has nothing the tuner could pick
    sch = ps.schema(kind)
    ends = {}
    for name, p in space.items():
        assert name in sch and not sch[name].extra, f"the tuner varies {kind}.{name}, which is not a setting of the model"
        values = list(p.choices) if p.kind == "categorical" else [p.low, p.high]
        for v in values:
            sch[name].check(int(v) if sch[name].type == ps.INT else v, f"{kind}.{name}")
        if p.kind == "int":
            assert sch[name].type == ps.INT
        ends[name] = [int(v) if sch[name].type == ps.INT else v for v in values]
    if kind in ps.RELATIONS:
        names = list(ends)
        for combo in itertools.product(*(ends[n] for n in names)):
            ps.effective_params(kind, dict(zip(names, combo)))                                    # the tuner never proposes a network that cannot be built


def test_the_catalogue_carries_each_models_schema_as_plain_data():
    catalogue = md.model_catalog()
    json.dumps(catalogue, allow_nan=False)
    for m in catalogue:
        assert list(m["schema"]) == list(m["params"]), m["kind"]
        for name, entry in m["schema"].items():
            assert entry["type"] in ps.TYPES and entry["default"] == ps.schema(m["kind"])[name].default, (m["kind"], name)
            if entry["type"] == "choice":
                assert entry["choices"] == m["specs"][name]["choices"]
            if entry["type"] in ("int", "float"):
                assert entry["min"] is not None and entry["range"], (m["kind"], name)
    by = {m["kind"]: m for m in catalogue}
    assert by["patchtst"]["schema"]["dropout"] == {"type": "float", "min": 0.0, "max": 1.0, "min_open": False, "max_open": True, "choices": [], "range": "at least 0 and below 1", "format": "", "default": 0.1}
    assert by["nhits"]["schema"]["pool_sizes"]["format"].startswith("whole numbers separated by commas") and by["ar"]["schema"]["columns"]["type"] == "list"
