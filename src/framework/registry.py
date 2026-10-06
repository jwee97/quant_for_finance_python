"""Plug-ins register themselves; the pipeline finds them by name.

Adding a strategy is writing a ``ForecastModel`` and decorating it::

    @register_model("my_signal", family="time-series", description="what it does and why it might work")
    class MySignal(ForecastModel):
        def score(self, data): ...

and it immediately runs through the same costs, validation, attribution and experiment database as every other model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Entry:
    name: str
    factory: Callable[..., Any]
    kind: str
    family: str
    description: str


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, Entry] = {}

    def register(self, name: str, family: str = "", description: str = ""):
        def decorate(factory):
            if name in self._entries and self._entries[name].factory is not factory:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._entries[name] = Entry(name, factory, self.kind, family, description or (factory.__doc__ or "").strip().split("\n")[0])
            return factory
        return decorate

    def create(self, name: str, **params):
        if name not in self._entries:
            raise KeyError(f"unknown {self.kind} '{name}'; available: {sorted(self._entries)}")
        return self._entries[name].factory(**params)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def entries(self) -> list[Entry]:
        return [self._entries[n] for n in self.names()]

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


MODELS = Registry("forecast model")
DETECTORS = Registry("regime detector")
ALLOCATORS = Registry("allocator")
COMBINERS = Registry("combiner")


def register_model(name: str, family: str = "", description: str = ""):
    return MODELS.register(name, family, description)


def register_detector(name: str, description: str = ""):
    return DETECTORS.register(name, "regime", description)


def register_allocator(name: str, description: str = ""):
    return ALLOCATORS.register(name, "allocation", description)
