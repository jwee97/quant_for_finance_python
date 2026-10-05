"""Structured feature extraction from text, behind a swappable backend (Generation 4, Priority 16).

The roadmap's use of language models in finance is not price prediction but turning documents into
STRUCTURED FEATURES. Three disciplines make that usable in a backtest:

1. **A schema.** Every backend returns exactly the declared fields, validated and clipped; anything else is an error.
2. **Point-in-time stamps.** A feature is attached to the date the document became available, never earlier.
3. **A content-addressed cache.** The key is the hash of (text, schema, backend identity), so the same
   document is never extracted twice, a changed schema or model cannot return a stale answer, and a published
   result can say exactly which model and prompt produced each number.

Backends: ``LexiconBackend`` (offline and deterministic: the baseline every model-based feature must beat) and
``AnthropicBackend`` (calls a hosted model through an injected client; tested here against a fake client only).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class ExtractionError(ValueError):
    """The backend's answer did not match the schema."""


@dataclass(frozen=True)
class Field:
    name: str
    kind: str                      # "float" | "int"
    low: float | None = None
    high: float | None = None
    description: str = ""


SCHEMA = (
    Field("tone", "float", None, None, "(hawkish term count - dovish term count) per 1,000 words"),
    Field("change", "float", 0.0, 1.0, "1 - cosine similarity of the statement's tf-idf vector with the previous statement's"),
    Field("action", "int", -1, 1, "target-rate decision: -1 lower, 0 maintain, +1 raise"),
)


def schema_hash(schema: tuple[Field, ...] = SCHEMA) -> str:
    return hashlib.sha256(json.dumps([vars(f) for f in schema], sort_keys=True).encode()).hexdigest()[:12]


def validate(record: dict, schema: tuple[Field, ...] = SCHEMA) -> dict:
    """Exactly the schema's fields, coerced to their types and clipped to range. Anything else raises."""
    if set(record) != {f.name for f in schema}:
        raise ExtractionError(f"fields {sorted(record)} do not match the schema {[f.name for f in schema]}")
    out = {}
    for f in schema:
        value = record[f.name]
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ExtractionError(f"{f.name}={value!r} is not numeric") from error
        if not np.isfinite(number):
            raise ExtractionError(f"{f.name} is not finite")
        if f.low is not None:
            number = max(number, f.low)
        if f.high is not None:
            number = min(number, f.high)
        out[f.name] = int(round(number)) if f.kind == "int" else number
    return out


class Backend(Protocol):
    name: str

    def identity(self) -> str: ...
    def extract(self, text: str, history: list[str]) -> dict: ...


@dataclass
class LexiconBackend:
    """Offline baseline: term counts, a tf-idf change measure and a regular-expression decision reader."""

    hawkish: tuple[str, ...]
    dovish: tuple[str, ...]
    raise_pattern: str = r"decided[^.]{0,80}\b(raise|increase)\b"
    lower_pattern: str = r"decided[^.]{0,80}\b(lower|reduce|decrease)\b"
    name: str = "lexicon"

    @classmethod
    def from_config(cls, node: dict) -> "LexiconBackend":
        patterns = node.get("action_patterns", {}) or {}
        return cls(tuple(node["hawkish"]), tuple(node["dovish"]), patterns.get("raise", cls.raise_pattern), patterns.get("lower", cls.lower_pattern))

    def identity(self) -> str:
        payload = json.dumps({"h": self.hawkish, "d": self.dovish, "r": self.raise_pattern, "l": self.lower_pattern}, sort_keys=True)
        return f"lexicon:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"

    def extract(self, text: str, history: list[str]) -> dict:
        lower = text.lower()
        words = max(len(lower.split()), 1)
        hawk = sum(lower.count(t) for t in self.hawkish)
        dove = sum(lower.count(t) for t in self.dovish)
        if re.search(self.raise_pattern, lower):
            action = 1
        elif re.search(self.lower_pattern, lower):
            action = -1
        else:
            action = 0
        return validate({"tone": 1000.0 * (hawk - dove) / words, "change": self._change(text, history), "action": action})

    @staticmethod
    def _change(text: str, history: list[str]) -> float:
        """1 - cosine similarity of tf-idf vectors with the previous statement; idf from the documents up to and including this one."""
        if not history:
            return 0.0
        from sklearn.feature_extraction.text import TfidfVectorizer

        matrix = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(history + [text])
        a, b = matrix[-1], matrix[-2]
        denom = np.sqrt(a.multiply(a).sum()) * np.sqrt(b.multiply(b).sum())
        return float(1.0 - (a.multiply(b).sum() / denom if denom > 0 else 0.0))


@dataclass
class AnthropicBackend:
    """Hosted-model extraction with temperature 0 and JSON output.

    ``client`` is anything with ``messages.create(...)`` (the ``anthropic`` SDK client, or a fake in tests). It is
    never constructed here without credentials: pass one in, or install the SDK and set ``ANTHROPIC_API_KEY``.
    The model identifier and a hash of the prompt are part of ``identity()``, so a cached answer can never be
    served for a different model or prompt.
    """

    model: str
    client: Any = None
    max_tokens: int = 300
    name: str = "anthropic"
    system: str = field(default=(
        "You extract structured features from a central-bank statement. Answer with a single JSON object and nothing else. "
        "Judge ONLY the text you are given; do not use knowledge of what happened afterwards."))

    def prompt(self, text: str, history: list[str]) -> str:
        fields = "\n".join(f"- {f.name} ({f.kind}{'' if f.low is None else f', {f.low} to {f.high}'}): {f.description}" for f in SCHEMA if f.name != "change")
        previous = history[-1] if history else "(none)"
        return (f"Fields to return (JSON keys exactly: tone, action):\n{fields}\n\nPREVIOUS STATEMENT:\n{previous}\n\nSTATEMENT:\n{text}\n")

    def identity(self) -> str:
        template_hash = hashlib.sha256((self.system + self.prompt("{text}", ["{previous}"])).encode()).hexdigest()[:12]
        return f"anthropic:{self.model}:{template_hash}"

    def _client(self):
        if self.client is not None:
            return self.client
        try:
            import anthropic
        except ImportError as error:
            raise RuntimeError("the Anthropic backend needs `pip install anthropic` and ANTHROPIC_API_KEY") from error
        return anthropic.Anthropic()

    def extract(self, text: str, history: list[str]) -> dict:
        reply = self._client().messages.create(model=self.model, max_tokens=self.max_tokens, temperature=0.0, system=self.system,
                                               messages=[{"role": "user", "content": self.prompt(text, history)}])
        raw = "".join(getattr(block, "text", "") for block in reply.content).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match is None:
            raise ExtractionError("no JSON object in the model's reply")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise ExtractionError("the model's reply is not valid JSON") from error
        parsed = {"tone": parsed.get("tone"), "action": parsed.get("action")}
        parsed["change"] = LexiconBackend._change(text, history)          # a text-similarity measure, computed locally, not by the model
        return validate(parsed)


class ExtractionCache:
    """JSON-lines cache keyed by sha256(text) + schema hash + backend identity."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._data[row["key"]] = row["record"]

    @staticmethod
    def key(text: str, backend: Backend, history: list[str]) -> str:
        context = hashlib.sha256(history[-1].encode()).hexdigest()[:12] if history else "none"
        return f"{hashlib.sha256(text.encode()).hexdigest()}|{context}|{schema_hash()}|{backend.identity()}"

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, record: dict) -> None:
        self._data[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "record": record}) + "\n")

    def __len__(self) -> int:
        return len(self._data)


def extract_corpus(corpus: pd.DataFrame, backend: Backend, cache: ExtractionCache | None = None) -> pd.DataFrame:
    """Features for every document, in date order, each with the date it became available."""
    texts = list(corpus["text"])
    rows = []
    for i, (date, available) in enumerate(zip(corpus["date"], corpus["available_at"])):
        history = texts[:i]
        key = ExtractionCache.key(texts[i], backend, history)
        record = cache.get(key) if cache is not None else None
        if record is None:
            record = backend.extract(texts[i], history)
            if cache is not None:
                cache.put(key, record)
        rows.append({"date": date, "available_at": available, **record})
    return pd.DataFrame(rows)
