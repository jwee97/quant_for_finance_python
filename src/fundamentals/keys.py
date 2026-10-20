"""Several API keys for one provider, used as one allowance.

Keys live only in the environment (on Replit: **Tools > Secrets**) or in the git-ignored ``.env`` file, never in the database, a page or a log. For a provider called ``fmp`` the pool reads, in this order

* ``FMP_API_KEYS``: a list separated by commas, semicolons, spaces or line breaks;
* ``FMP_API_KEY``: one key;
* ``FMP_API_KEY_1`` ... ``FMP_API_KEY_20``: one key each (the easiest on Replit, whose Secrets are one name and one value apiece).

The same key listed twice counts once. Each key is a *slot* ("key 1", "key 2", ...) and is identified in the usage log by a salted one-way hash, so its count follows it if the list is reordered and the
log never holds anything from which the key could be read back.

:meth:`KeyPool.call` runs a request with the key that has the most of today's allowance left. A provider answer of "rate limited" ends that key's day and the request moves to the next key; "invalid
key" disables that key; a plan error (the provider does not offer the endpoint) is raised at once, because every key on the same plan would say the same and trying them all would only spend requests.

Using several free accounts to multiply a free allowance may be against a provider's terms of service. The pool does not check that; read the provider's terms before relying on it.
"""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

from ..webapp.providers import read_secret, tidy_key
from .errors import InvalidKey, MissingKey, PlanError, ProviderError, QuotaExhausted, RateLimited
from .usage import UsageLog, daily_limit

_SEPARATORS = re.compile(r"[,;\s]+")
MAX_NUMBERED = 20


def discover(provider: str, root: Path | None = None) -> tuple[list[tuple[str, str]], list[str]]:
    """``([(variable, key), ...], notes)`` for ``provider``: every distinct key found, with the name of the variable it came from. The notes never contain a key."""
    name = provider.upper()
    raw: list[tuple[str, str]] = []
    for part in _SEPARATORS.split(read_secret(f"{name}_API_KEYS", root)):
        if part:
            raw.append((f"{name}_API_KEYS", part))
    for variable in [f"{name}_API_KEY"] + [f"{name}_API_KEY_{i}" for i in range(1, MAX_NUMBERED + 1)]:
        text = read_secret(variable, root)
        if text:
            raw.append((variable, text))
    keys: list[tuple[str, str]] = []
    notes: list[str] = []
    for variable, text in raw:
        key, tidy = tidy_key(text, variable)
        notes.extend(f"{variable}: {n}" for n in tidy)
        if not key or any("placeholder" in n for n in tidy):
            continue
        if any(key == k for _, k in keys):
            notes.append(f"{variable}: the same key is already listed and is counted once")
            continue
        keys.append((variable, key))
    return keys, notes


class KeyPool:
    def __init__(self, provider: str, keys: list[tuple[str, str]], usage: UsageLog | None = None, salt: str = "", limit: int | None = None, notes: list[str] | None = None,
                 keyless: bool = False):
        self.provider, self.usage, self.notes, self.keyless = provider, usage, list(notes or []), keyless
        self.limit = daily_limit(provider) if limit is None else (limit or None)
        if keyless:                                                           # a provider that needs no key still has a daily budget, counted the same way
            keys = [("", "")]
        self._keys = [("(no key needed)" if keyless else f"key {i + 1}", variable, key, hashlib.sha256(f"{salt}:{provider}:{key}".encode()).hexdigest()[:16])
                      for i, (variable, key) in enumerate(keys)]
        self._lock = threading.Lock()
        self._invalid: set[str] = set()

    @classmethod
    def from_environment(cls, provider: str, usage: UsageLog | None = None, salt: str = "", root: Path | None = None, limit: int | None = None) -> "KeyPool":
        keys, notes = discover(provider, root)
        return cls(provider, keys, usage, salt, limit, notes)

    @classmethod
    def without_key(cls, provider: str, usage: UsageLog | None = None, limit: int | None = None) -> "KeyPool":
        """The same accounting for a provider that needs no key (a local library, a regulator's public site, the simulator): one anonymous slot with the provider's daily budget."""
        return cls(provider, [], usage, "", limit, keyless=True)

    # ------------------------------------------------------------------------------------------------------------ state
    def __len__(self) -> int:
        return len(self._keys)

    def secrets(self) -> list[str]:
        """The keys themselves, for redacting them from messages. Nothing else should call this."""
        return [key for _, _, key, _ in self._keys]

    def _state(self, key_id: str) -> str:
        if key_id in self._invalid:
            return "invalid"
        if self.usage is not None:
            why = self.usage.blocked_today(self.provider, key_id)
            if why:
                return "invalid" if why.startswith("auth") else "exhausted"
            if self.limit is not None and self.usage.used_today(self.provider, key_id) >= self.limit:
                return "exhausted"
        return "ready"

    def slots(self) -> list[dict]:
        """One line per key for the Data Manager: its slot, the variable it came from, how many requests it has made today, and whether it can be used. No key material."""
        out = []
        for slot, variable, _, key_id in self._keys:
            used = self.usage.used_today(self.provider, key_id) if self.usage is not None else 0
            out.append({"slot": slot, "variable": variable, "key_id": key_id, "used": used, "limit": self.limit, "state": self._state(key_id)})
        return out

    def remaining_today(self) -> int | None:
        """Requests left today across every usable key (``None`` when there is no limit)."""
        if self.limit is None:
            return None
        return sum(max(self.limit - s["used"], 0) for s in self.slots() if s["state"] == "ready")

    @property
    def configured(self) -> bool:
        """Whether the pool can make a request at all: it holds a key, or the provider needs none."""
        return self.keyless or len(self._keys) > 0

    # ------------------------------------------------------------------------------------------------------------ use
    def _acquire(self, exclude: set[str]) -> tuple[str, str, str]:
        if not self.keyless and not self._keys:
            raise MissingKey(f"no {self.provider.upper()} API key is set: add {self.provider.upper()}_API_KEY (or {self.provider.upper()}_API_KEY_1, _2, ... one per account) "
                             "as a secret (on Replit: Tools > Secrets) and restart the app")
        with self._lock:
            best, room = None, -1.0
            for slot, _, key, key_id in self._keys:
                if key_id in exclude or self._state(key_id) != "ready":
                    continue
                used = self.usage.used_today(self.provider, key_id) if self.usage is not None else 0
                left = float("inf") if self.limit is None else self.limit - used
                if left > room:
                    best, room = (slot, key, key_id), left
        if best is None:
            states = ", ".join(f"{s['slot']} {s['state']}" for s in self.slots())
            raise QuotaExhausted(f"every {self.provider.upper()} key is out for today ({states}); the allowance resets at 00:00 UTC")
        return best

    def call(self, fn, endpoint: str = "", symbol: str = "", market: str = ""):
        """``fn(key)`` with the best available key, logging the request. ``fn`` returns the answer or raises one of the provider errors (never with the key in its message)."""
        tried: set[str] = set()
        while True:
            slot, key, key_id = self._acquire(tried)
            tried.add(key_id)
            try:
                result = fn(key)
            except RateLimited as error:
                self._log(endpoint, symbol, market, slot, key_id, False, f"limit: {error}")
                continue
            except InvalidKey as error:
                self._log(endpoint, symbol, market, slot, key_id, False, f"auth: {error}")
                with self._lock:
                    self._invalid.add(key_id)
                continue
            except PlanError as error:
                self._log(endpoint, symbol, market, slot, key_id, False, f"plan: {error}")
                raise
            except ProviderError as error:
                self._log(endpoint, symbol, market, slot, key_id, False, str(error))
                raise
            self._log(endpoint, symbol, market, slot, key_id, True, "")
            return result

    def _log(self, endpoint, symbol, market, slot, key_id, success, error) -> None:
        if self.usage is not None:
            self.usage.record(self.provider, endpoint, success, symbol, market, slot, key_id, error=error)
