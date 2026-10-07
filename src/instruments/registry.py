"""The instrument registry: one place that knows every contract the engine can see, with the lookups the engine, the ledger and the strategies need."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .base import Instrument, instrument_from_dict
from .futures import Future, FutureChain


class InstrumentRegistry:
    """A keyed collection of instruments and future chains, JSON-serialisable.

    ``add`` rejects a duplicate id with a different specification (the same specification is accepted, so loading is idempotent); ``validate`` checks that every derivative's underlying exists
    and that expiries are sensible. OTC contracts (swaps, forwards) are added when they are traded, so the registry also grows during a run."""

    def __init__(self, instruments: list[Instrument] | None = None, chains: list[FutureChain] | None = None):
        self._items: dict[str, Instrument] = {}
        self._chains: dict[str, FutureChain] = {}
        for i in instruments or []:
            self.add(i)
        for c in chains or []:
            self.add_chain(c)

    # ------------------------------------------------------------------------------------------------------------------------------- content
    def add(self, instrument: Instrument) -> Instrument:
        existing = self._items.get(instrument.instrument_id)
        if existing is not None and existing != instrument:
            raise ValueError(f"instrument '{instrument.instrument_id}' is already registered with a different specification")
        self._items[instrument.instrument_id] = instrument
        return instrument

    def add_chain(self, chain: FutureChain) -> FutureChain:
        for c in chain.contracts:
            self.add(c)
        self._chains[chain.chain_id] = chain
        return chain

    def __contains__(self, instrument_id: str) -> bool:
        return instrument_id in self._items or instrument_id in self._chains

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def is_chain(self, identifier: str) -> bool:
        """True if ``identifier`` names a future chain (and not a single instrument)."""
        return identifier in self._chains and identifier not in self._items

    def has_instrument(self, instrument_id: str) -> bool:
        return instrument_id in self._items

    def get(self, instrument_id: str) -> Instrument:
        try:
            return self._items[instrument_id]
        except KeyError:
            raise KeyError(f"unknown instrument '{instrument_id}'") from None

    def chain(self, chain_id: str) -> FutureChain:
        try:
            return self._chains[chain_id]
        except KeyError:
            raise KeyError(f"unknown future chain '{chain_id}'") from None

    @property
    def chains(self) -> dict[str, FutureChain]:
        return dict(self._chains)

    def ids(self) -> list[str]:
        return sorted(self._items)

    # ------------------------------------------------------------------------------------------------------------------------------ queries
    def by_asset_class(self, asset_class: str) -> list[Instrument]:
        return [i for i in self if i.asset_class == asset_class]

    def by_type(self, instrument_type: str) -> list[Instrument]:
        return [i for i in self if i.instrument_type == instrument_type]

    def by_underlying(self, underlying_id: str) -> list[Instrument]:
        return [i for i in self if i.underlying_id == underlying_id]

    def active(self, ts) -> list[Instrument]:
        """Instruments that exist at ``ts``: not yet expired and, for futures, already listed."""
        ts = pd.Timestamp(ts)
        out = []
        for i in self:
            if i.expiry is not None and ts > i.expiry + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1):
                continue
            if isinstance(i, Future) and i.first_trade is not None and ts < i.first_trade:
                continue
            out.append(i)
        return out

    def expiring(self, start, end) -> list[Instrument]:
        """Instruments with an expiry in ``(start, end]``, soonest first."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return sorted((i for i in self if i.expiry is not None and s < i.expiry <= e), key=lambda i: (i.expiry, i.instrument_id))

    def validate(self) -> list[str]:
        """Human-readable problems (an empty list means consistent): a derivative whose underlying is not registered, an expiry before a listing."""
        problems = []
        for i in self:
            if i.underlying_id is not None and i.underlying_id not in self._items and i.underlying_id not in self._chains:
                problems.append(f"{i.instrument_id}: underlying '{i.underlying_id}' is not registered")
            if isinstance(i, Future) and i.first_trade is not None and i.first_trade > i.expiry:
                problems.append(f"{i.instrument_id}: first_trade after expiry")
        return problems

    def to_frame(self) -> pd.DataFrame:
        rows = [{"instrument_id": i.instrument_id, "asset_class": i.asset_class, "type": i.instrument_type, "currency": i.currency, "multiplier": i.contract_multiplier,
                 "tick_size": i.tick_size, "lot_size": i.lot_size, "expiry": i.expiry, "underlying": i.underlying_id, "cash_style": i.cash_style} for i in self]
        return pd.DataFrame(rows).set_index("instrument_id") if rows else pd.DataFrame()

    # ----------------------------------------------------------------------------------------------------------------------- serialisation
    def to_json(self, path: str | Path | None = None) -> str:
        chain_members = {c.instrument_id for ch in self._chains.values() for c in ch.contracts}
        payload = {"instruments": [i.to_dict() for i in self if i.instrument_id not in chain_members], "chains": [c.to_dict() for c in self._chains.values()]}
        text = json.dumps(payload, indent=1, sort_keys=True, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> "InstrumentRegistry":
        text = Path(source).read_text(encoding="utf-8") if not str(source).lstrip().startswith("{") else str(source)
        payload = json.loads(text)
        reg = cls([instrument_from_dict(d) for d in payload.get("instruments", [])])
        for c in payload.get("chains", []):
            contracts = tuple(instrument_from_dict(d) for d in c["contracts"])
            from .futures import RollSpec
            reg.add_chain(FutureChain(c["chain_id"], c["root"], contracts, RollSpec(**c["roll"])))
        return reg
