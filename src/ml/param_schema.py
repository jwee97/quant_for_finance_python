"""The settings of each model as a schema: what a request may set, of what type, and within which range.

``models.ModelInfo`` lists a model's settings with their defaults. A default cannot say that ``min_child_weight`` takes fractions, that ``dropout`` may not reach 1, that ``device`` is a menu or that ``members``
is a list of models, so the web API used to count the settings instead (no more than 20) and leave every other mistake to the job that failed minutes later. This module states those facts once, for every model
of the catalogue, and the API validates a request against them.

Three functions are the interface:

* ``validate_params(kind, params)``: the settings a request names, each checked for its type and range and written in one canonical form. An unknown name, a wrong type, a number out of range, a choice outside
  the menu or malformed text is refused with a message that names the setting. Nothing is added: this is only what the person set.
* ``effective_params(kind, params)``: the model's defaults with those settings laid over them, and the relations between settings checked (``d_model`` a multiple of ``n_heads``). This is what is trained and what
  the experiment's configuration records, so a saved experiment states every setting it ran with, whichever client made the request and whether it sent the defaults or only the changes.
* ``describe(kind)``: the schema as plain data, for the form.

**Ranges.** A lower bound that mirrors a check in the model's constructor or library is a fact about the model (a layer count of 0 builds nothing). The natural bounds of fractions, probabilities and levels
are kept. For counts that set the cost of a fit (trees, epochs, widths, timesteps) the upper bound is a wide sanity limit that stops a runaway request and binds no real use; it is not a property of the model.
Where a model accepts a value, so does the schema: the tests build every model with each setting at its bounds.

**One spelling per value.** A browser sends ``10.0`` as ``10``; a script sends ``10.0``. Both are stored as ``10``, so a setting has one configuration hash whichever client sent it. A switch is stored as 1 or 0
(as the catalogue lists it), and text and names are stripped. Nothing here changes how a model is built or fitted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

INT, FLOAT, BOOL, CHOICE, TEXT, LIST = "int", "float", "bool", "choice", "text", "list"
TYPES = (INT, FLOAT, BOOL, CHOICE, TEXT, LIST)
MAX_MAGNITUDE = 1e18                                                           # no setting of any model is larger, and a float of this size still converts
MAX_TEXT = 400
MAX_NAMES = 500
FEATURE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,59}$")
WHOLE_NUMBERS = re.compile(r"^\d+(\s*,\s*\d+)*$")
_INTERVAL = re.compile(r"^([\[(])([^,]*),([^,]*)([\])])$")
_UNSET = object()


class ParamError(ValueError):
    """A setting a model cannot take. The message names the setting and says what is allowed."""


def interval(text: str) -> tuple[float | None, float | None, bool, bool]:
    """``"[0,1)"`` -> ``(0.0, 1.0, False, True)``: the lower end, the upper end and whether each end is excluded. An empty end is unbounded: ``"(0,)"`` is every number above zero."""
    m = _INTERVAL.match(text.replace(" ", ""))
    if not m:
        raise ValueError(f"not an interval: {text!r}")
    lo, hi = m.group(2), m.group(3)
    return (float(lo) if lo else None), (float(hi) if hi else None), bool(lo) and m.group(1) == "(", bool(hi) and m.group(4) == ")"


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def _short(value: Any, n: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= n else text[: n - 1] + "…"


def _show(x: int | float) -> str:
    """A number as the message shows it: a whole number in full, anything else to six figures."""
    return str(x) if isinstance(x, int) else f"{x:g}"


def _kind_of(value: Any) -> str:
    return {bool: "true/false", int: "a number", float: "a number", str: "text", list: "a list", dict: "an object", type(None): "null"}.get(type(value), type(value).__name__)


def canonical_number(x: int | float) -> int | float:
    """An integral float is the int (the browser's ``10`` and a script's ``10.0`` are one value); anything else is as it was."""
    if isinstance(x, float) and x.is_integer() and abs(x) < 2 ** 53:
        return int(x)
    return x


@dataclass(frozen=True)
class Setting:
    """One setting of one model: its type, its range or its choices, and the default the catalogue gives it."""

    name: str
    type: str
    default: Any = None
    low: float | None = None
    high: float | None = None
    low_open: bool = False
    high_open: bool = False
    choices: tuple = ()
    grammar: str | None = None            # text only: "ints" (whole numbers separated by commas) or "members" (the models of an ensemble)
    nonempty: bool = False
    extra: bool = False                   # accepted by every model although it is not one of the model's own settings

    # ------------------------------------------------------------------------------------------------------------ what is allowed, in words
    def range_text(self) -> str:
        parts = []
        if self.low is not None:
            parts.append(("above " if self.low_open else "at least ") + _fmt(self.low))
        if self.high is not None:
            parts.append(("below " if self.high_open else "at most ") + _fmt(self.high))
        return " and ".join(parts)

    def format_text(self) -> str:
        return {"ints": "whole numbers separated by commas, for example 4,2,1", "members": "models separated by commas, each as kind or kind[setting=value;setting=value]"}.get(self.grammar or "", "")

    def to_json(self) -> dict:
        return {"type": self.type, "min": self.low, "max": self.high, "min_open": self.low_open, "max_open": self.high_open, "choices": list(self.choices), "range": self.range_text(),
                "format": self.format_text(), "default": list(self.default) if isinstance(self.default, list) else self.default}

    # ------------------------------------------------------------------------------------------------------------ checking one value
    def check(self, value: Any, path: str) -> Any:
        """The value in its canonical form; ``_UNSET`` where nothing was set (null, or a blank where a blank means the default); ``ParamError`` for anything the model cannot take."""
        if value is None:
            return _UNSET
        if self.type in (INT, FLOAT):
            return self._number(value, path)
        if self.type == BOOL:
            return self._switch(value, path)
        if self.type == CHOICE:
            return self._choice(value, path)
        if self.type == LIST:
            return self._names(value, path)
        return self._text(value, path)

    def _number(self, value: Any, path: str):
        if isinstance(value, str) and not value.strip():
            return _UNSET
        what = "a whole number" if self.type == INT else "a number"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParamError(f"{path} must be {what}, not {_kind_of(value)}")
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ParamError(f"{path} must be a finite number")
            if self.type == INT:
                if not value.is_integer():
                    raise ParamError(f"{path} must be a whole number (got {_show(value)})")
                value = int(value)
        if abs(value) > MAX_MAGNITUDE:
            raise ParamError(f"{path} must be between -1e18 and 1e18")
        too_low = self.low is not None and (value < self.low or (self.low_open and value == self.low))
        too_high = self.high is not None and (value > self.high or (self.high_open and value == self.high))
        if too_low or too_high:
            raise ParamError(f"{path} must be {what}, {self.range_text()} (got {_show(value)})")
        return canonical_number(value)

    def _switch(self, value: Any, path: str):
        if isinstance(value, str) and not value.strip():
            return _UNSET
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        raise ParamError(f"{path} must be true or false (or 1 or 0), not {_short(value)}")

    def _choice(self, value: Any, path: str):
        if not isinstance(value, str):
            raise ParamError(f"{path} must be one of {list(self.choices)}, not {_kind_of(value)}")
        text = value.strip()
        if not text:
            return _UNSET
        if text not in self.choices:
            raise ParamError(f"{path} must be one of {list(self.choices)} (got {_short(text)})")
        return text

    def _text(self, value: Any, path: str):
        if not isinstance(value, str):
            raise ParamError(f"{path} must be text, not {_kind_of(value)}")
        text = value.strip()
        if len(text) > MAX_TEXT:
            raise ParamError(f"{path} must be at most {MAX_TEXT} characters")
        if any(ord(c) < 32 or ord(c) == 127 for c in text):
            raise ParamError(f"{path} must not contain control characters")
        if not text:
            if self.nonempty:
                raise ParamError(f"{path} must not be empty")
            return ""
        if self.grammar == "ints":
            _whole_numbers(text, path)
        elif self.grammar == "members":
            _members(text, path)
        return text

    def _names(self, value: Any, path: str):
        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, (list, tuple)):
            if not all(isinstance(s, str) for s in value):
                raise ParamError(f"{path} must be a list of feature names")
            items = list(value)
        else:
            raise ParamError(f"{path} must be a list of feature names (or names separated by commas), not {_kind_of(value)}")
        names = [s for s in dict.fromkeys(s.strip() for s in items) if s]
        if len(names) > MAX_NAMES:
            raise ParamError(f"{path} names at most {MAX_NAMES} features")
        bad = [s for s in names if not FEATURE_NAME.match(s)]
        if bad:
            raise ParamError(f"{path} has {len(bad)} entr{'y' if len(bad) == 1 else 'ies'} that cannot be a feature name, for example {_short(bad[0])}")
        return names


def _whole_numbers(text: str, path: str) -> None:
    if not WHOLE_NUMBERS.match(text):
        raise ParamError(f"{path} must be whole numbers separated by commas, for example 4,2,1")
    if any(not 1 <= int(x) <= 1000 for x in text.split(",")):
        raise ParamError(f"{path} must hold whole numbers from 1 to 1000")


def _members(text: str, path: str) -> None:
    """An ensemble's members: the grammar, the count, that each is a model that can be one, and each member's own settings against that model's schema."""
    from .contracts import AGENTS
    from .ensemble_models import MAX_MEMBERS, parse_members
    from .models import _INFO

    try:
        members = parse_members(text)
    except ValueError as exc:
        raise ParamError(f"{path}: {exc}") from None
    if not 2 <= len(members) <= MAX_MEMBERS:
        raise ParamError(f"{path} needs between 2 and {MAX_MEMBERS} members (got {len(members)})")
    for kind, given in members:
        if kind in AGENTS:
            raise ParamError(f"{path}: '{kind}' is an agent whose output is a set of portfolio weights, not a forecast, so it cannot be a member")
        if kind not in _INFO or kind == "model_ensemble":
            raise ParamError(f"{path}: '{kind}' is not a model that can be a member")
        validate_params(kind, given, f"{path}[{kind}]")


# ---------------------------------------------------------------------------------------------------------------------- the ranges
DEEP_KINDS = ("lstm", "gru", "tcn", "patchtst", "nbeats", "nhits", "tft", "autoencoder", "vae")
GRAPH_KINDS = ("gcn", "gat")
AGENT_KINDS = ("ppo", "sac", "recurrent_ppo")

# A setting's range by its name. "kind.name" overrides it where the same name means something else for one model. Every numeric setting of the catalogue needs an entry here (a test enforces it).
RANGES: dict[str, str] = {
    "alpha": "[0,)", "alpha_high": "(0.5,1)", "alpha_low": "(0,0.5)", "batch_size": "[1,1000000]", "beta": "[0,)", "block_days": "[1,1000]", "blocks": "[1,64]", "buffer_size": "[1,100000000]",
    "channels": "[1,1024]", "clip_range": "(0,1]", "col_sample": "(0,1]", "colsample_bytree": "(0,1]", "corr_weight": "[0,)", "cost_bps": "[0,500]", "d_model": "[1,8192]", "decay_days": "[1,100000]",
    "depth": "[1,16]", "drawdown_penalty": "[0,)", "dropout": "[0,1)", "ent_coef": "[0,)", "episode_days": "[20,100000]", "epochs": "[1,10000]", "eta": "(0,)", "ff_dim": "[1,65536]", "forgetting": "(0,1]",
    "fraction": "(0,1)", "gamma": "(0,1]", "grad_clip": "[0,)", "gross": "(0,100]", "heads": "[1,64]", "hidden": "[1,8192]", "impact": "[0,)", "iterations": "[1,100000]", "kernel_size": "[1,512]",
    "l1_ratio": "[0,1]", "l2": "[0,)", "l2_leaf_reg": "[0,)", "l2_regularization": "[0,)", "latent_dim": "[1,8192]", "layers": "[1,64]", "learning_rate": "(0,)", "learning_starts": "[0,100000000]",
    "level": "(0,1)", "lookback": "[2,2000]", "lstm_hidden": "[1,8192]", "lstm_layers": "[1,64]", "max_depth": "[1,64]", "max_iter": "[1,100000]", "max_rows": "[0,100000000]", "max_turnover": "[0,)",
    "max_weight": "(0,1]", "min_child_samples": "[1,)", "min_child_weight": "[0,)", "min_coverage": "[0,1]", "min_features": "[1,1000]", "min_rows": "[1,)", "min_samples_leaf": "[1,)", "min_weight": "[0,)",
    "minibatch_frac": "(0,1]", "n_epochs": "[1,10000]", "n_estimators": "[1,100000]", "n_heads": "[1,64]", "n_iter": "[1,10000]", "n_states": "[2,5]", "n_steps": "[2,1000000]", "neighbors": "[1,1000]",
    "num_leaves": "[2,131072]", "num_threads": "[0,1024]", "patch_len": "[1,2000]", "patience": "[1,10000]", "pi": "(0,1]", "q": "[0,)", "r": "(0,)", "reg_lambda": "[0,)", "regimes": "[2,4]",
    "relevance_levels": "[2,10]", "ridge_alpha": "[0,)", "risk_aversion": "[0,)", "sector_weight": "[0,)", "selector_alpha": "[0,)", "shrink": "[0,1]", "stride": "[1,2000]", "subsample": "(0,1]",
    "subsamples": "[10,10000]", "tau": "(0,1]", "temperature": "[0,)", "top_k": "[0,1000]", "total_timesteps": "[64,100000000]", "turnover_penalty": "[0,)", "val_fraction": "[0.05,0.5]",
    "weight_decay": "[0,)", "width": "[1,8192]", "halflife_days": "[0,)",
    "random_forest.max_features": "(0,1]", "kalman.max_features": "[1,1000]", "model_ensemble.val_fraction": "[0.1,0.5]",
}
RANGES.update({f"{k}.batch_size": "[8,1000000]" for k in DEEP_KINDS})                  # the networks' own check: a batch of at least 8
RANGES.update({f"{k}.hidden": "[4,8192]" for k in AGENT_KINDS})                          # an agent's network is at least 4 wide
TYPE_FIXES = {("xgboost", "min_child_weight"): FLOAT, ("xgb_ranker", "min_child_weight"): FLOAT, ("hmm_regime", "min_weight"): FLOAT, ("markov_switching", "min_weight"): FLOAT}     # an int default of a setting that takes fractions
GRAMMARS = {"horizons": "ints", "pool_sizes": "ints", "members": "members"}
NONEMPTY = {"members"}

# Settings every model accepts although they are not among its own (``Model._select`` and ``Model._weights`` read them). ``columns`` is also an own setting of the autoregression, with a default.
EXTRAS = {
    "columns": Setting("columns", LIST, [], extra=True),
    "halflife_days": Setting("halflife_days", FLOAT, 0, *interval(RANGES["halflife_days"]), extra=True),
}


def _relation_multiple(big: str, small: str):
    def rule(p: dict, path: str) -> None:
        if p[big] % p[small]:
            raise ParamError(f"{path}.{big} ({p[big]}) must be a multiple of {path}.{small} ({p[small]})")
    return rule


def _relation_at_least(big: str, small: str):
    def rule(p: dict, path: str) -> None:
        if p[big] < p[small]:
            raise ParamError(f"{path}.{big} ({p[big]}) must be at least {path}.{small} ({p[small]})")
    return rule


# The relations the networks themselves enforce when they are built, stated here so that a request that could not build is refused before it is queued.
RELATIONS = {
    "patchtst": (_relation_at_least("lookback", "patch_len"), _relation_multiple("d_model", "n_heads")),
    "tft": (_relation_multiple("hidden", "n_heads"),),
    "gat": (_relation_multiple("hidden", "heads"),),
}


# ---------------------------------------------------------------------------------------------------------------------- building the schema of a model
def _canonical_default(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return canonical_number(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _build(kind: str, name: str, default: Any, declared: Mapping) -> Setting:
    given = declared.get("type")
    choices: tuple = ()
    if given == "choice":
        typ, choices = CHOICE, tuple(declared.get("choices") or ())
    elif given in (INT, FLOAT, BOOL, TEXT):
        typ = given
    elif isinstance(default, bool):
        typ = BOOL
    elif isinstance(default, int):
        typ = INT
    elif isinstance(default, float):
        typ = FLOAT
    elif isinstance(default, str):
        typ = TEXT
    elif isinstance(default, (list, tuple)):
        typ = LIST
    else:
        raise ParamError(f"the type of {kind}.{name} cannot be told from its default {_short(default)}")
    typ = TYPE_FIXES.get((kind, name), typ)
    low = high = None
    low_open = high_open = False
    if typ in (INT, FLOAT):
        text = RANGES.get(f"{kind}.{name}") or RANGES.get(name)
        if text is None:
            raise ParamError(f"no range is declared for the setting {kind}.{name}")
        low, high, low_open, high_open = interval(text)
    return Setting(name, typ, _canonical_default(default), low, high, low_open, high_open, choices, GRAMMARS.get(name) if typ == TEXT else None, name in NONEMPTY)


@lru_cache(maxsize=None)
def schema(kind: str) -> Mapping[str, Setting]:
    """Every setting of a model kind: its own, in the catalogue's order, then the ones every model accepts. Read-only."""
    from .models import _INFO

    info = _INFO.get(kind)
    if info is None:
        raise ParamError(f"unknown model '{kind}'; choose from {sorted(_INFO)}")
    out = {name: _build(kind, name, default, (info.specs or {}).get(name) or {}) for name, default in info.params.items()}
    for name, extra in EXTRAS.items():
        out.setdefault(name, replace(extra))
    return MappingProxyType(out)


def _label(kind: str) -> str:
    from .models import _INFO

    return _INFO[kind].label if kind in _INFO else kind


def validate_params(kind: str, params: Mapping | None, path: str = "model.params") -> dict:
    """The settings ``params`` names, each checked against the model's schema and in canonical form. A null or blank value is not a setting (the default stands). Unknown names are refused."""
    sch = schema(kind)
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ParamError(f"{path} must be an object of settings (name: value), not {_kind_of(params)}")
    out: dict = {}
    for name, value in params.items():
        spec = sch.get(name) if isinstance(name, str) else None
        if spec is None:
            listed = sorted(n for n, s in sch.items() if not s.extra)
            shown = name if isinstance(name, str) and re.fullmatch(r"[\w.:-]{1,60}", name) else _short(name, 60)                      # an ordinary name as it is, anything else quoted
            raise ParamError(f"{path}.{shown} is not a setting of the {_label(kind)} model; its settings are {listed or 'none'}")
        checked = spec.check(value, f"{path}.{name}")
        if checked is not _UNSET:
            out[name] = checked
    return out


def effective_params(kind: str, params: Mapping | None = None, path: str = "model.params") -> dict:
    """The model's defaults with the validated ``params`` laid over them, and the relations between settings checked: every setting the model will run with."""
    sch = schema(kind)
    given = validate_params(kind, params, path)
    merged = {name: (list(s.default) if isinstance(s.default, list) else s.default) for name, s in sch.items() if not s.extra}
    merged.update(given)
    for rule in RELATIONS.get(kind, ()):
        rule(merged, path)
    return merged


def describe(kind: str) -> dict:
    """The schema of a model's own settings as plain data (type, range, choices, default), for the form."""
    return {name: s.to_json() for name, s in schema(kind).items() if not s.extra}


__all__ = ["EXTRAS", "ParamError", "RANGES", "RELATIONS", "Setting", "canonical_number", "describe", "effective_params", "interval", "schema", "validate_params"]
