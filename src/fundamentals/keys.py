"""API keys for one provider: finding them, keeping them secret, counting what each has used, and choosing which one makes the next request.

Keys live only in the environment (on Replit: **Tools > Secrets**) or in the git-ignored ``.env`` file, never in the database, a page or a log. For a provider called ``openfilings`` the pool reads, in
this order

* ``OPENFILINGS_API_KEYS``: a list separated by commas, semicolons, spaces or line breaks;
* ``OPENFILINGS_API_KEY``: one key;
* ``OPENFILINGS_API_KEY_1`` ... ``OPENFILINGS_API_KEY_20``: one key each (the easiest on Replit, whose Secrets are one name and one value apiece).

The same key listed twice counts once. Each key is a *slot* ("key 1", "key 2", ...) and is identified in the usage log by a salted one-way hash, so its count follows it if the list is reordered and
the log never holds anything from which the key could be read back.

**Which key makes a request.** The key the user chose, if it is still usable, and otherwise the first key in the list that is not invalid. A later key is a *standby*: it takes over by itself only if
the earlier one is rejected as invalid or revoked (HTTP 401), which is what several keys are for in practice (replacing a key without downtime, a spare, a key per machine). The user can also *choose*
the key in use (:meth:`KeyPool.choose`, the *Use this key* button): the choice is kept, and nothing but another choice changes it. When a key's allowance is spent the pool stops and says when it
resets; it does **not** move on to the next key, and a choice is refused while the key in use has spent its allowance. OpenFilings' usage policy forbids "rotating accounts or API keys" to get past its
quotas, and several keys of one account share one allowance anyway, so rotation would be against the rules where it works and pointless where it does not. More credits come from a bigger plan or from
prepaid top-ups, not from more keys.
"""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

from ..webapp.providers import read_secret, tidy_key
from .errors import CreditsExhausted, FundamentalsError, InvalidKey, MissingKey, PlanError, ProviderError, QuotaExhausted, RateLimited, SyncPending
from .usage import Allowance, UsageLog, allowance

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
    def __init__(self, provider: str, keys: list[tuple[str, str]], usage: UsageLog | None = None, salt: str = "", allow: Allowance | None = None, notes: list[str] | None = None,
                 keyless: bool = False, plan: str | None = None):
        self.provider, self.usage, self.notes, self.keyless, self.plan = provider, usage, list(notes or []), keyless, plan
        self._fixed = allow
        if keyless:                                                           # a provider that needs no key still has a budget, counted the same way
            keys = [("", "")]
        self._keys = [("(no key needed)" if keyless else f"key {i + 1}", variable, key, hashlib.sha256(f"{salt}:{provider}:{key}".encode()).hexdigest()[:16])
                      for i, (variable, key) in enumerate(keys)]
        self._lock = threading.Lock()
        self._invalid: set[str] = set()
        self._chosen: str | None = None                                       # the key the user chose to be the one in use (its salted hash), if any

    @classmethod
    def from_environment(cls, provider: str, usage: UsageLog | None = None, salt: str = "", root: Path | None = None, allow: Allowance | None = None, plan: str | None = None) -> "KeyPool":
        keys, notes = discover(provider, root)
        return cls(provider, keys, usage, salt, allow, notes, plan=plan)

    @classmethod
    def without_key(cls, provider: str, usage: UsageLog | None = None, allow: Allowance | None = None) -> "KeyPool":
        """The same accounting for a provider that needs no key (the simulator): one anonymous slot with the provider's budget."""
        return cls(provider, [], usage, "", allow, keyless=True)

    # ------------------------------------------------------------------------------------------------------------ state
    def __len__(self) -> int:
        return len(self._keys)

    @property
    def allow(self) -> Allowance:
        return self._fixed or allowance(self.provider, self.plan)

    def secrets(self) -> list[str]:
        """The keys themselves, for redacting them from messages. Nothing else should call this."""
        return [key for _, _, key, _ in self._keys]

    def _block(self, key_id: str):
        if key_id in self._invalid:
            return None
        return self.usage.blocked(self.provider, key_id) if self.usage is not None else None

    def _used(self, key_id: str) -> tuple[int, int]:
        if self.usage is None:
            return 0, 0
        return self.usage.used_today(self.provider, key_id), self.usage.used_month(self.provider, key_id)

    def _state(self, key_id: str) -> str:
        """``ready``, ``spent_day``, ``spent_month``, ``no_credits`` (the provider says the prepaid balance cannot cover a call) or ``invalid``."""
        if key_id in self._invalid:
            return "invalid"
        block = self._block(key_id)
        if block is not None:
            return {"day": "spent_day", "month": "spent_month", "balance": "no_credits", "auth": "invalid"}[block.kind]
        daily = self.allow.daily
        if daily is not None and self._used(key_id)[0] >= daily:
            return "spent_day"
        return "ready"

    def _active(self) -> tuple[str, str, str, str] | None:
        """The key in use: the one the user chose while it is not invalid, otherwise the first key that is not invalid."""
        if self._chosen is not None:
            for entry in self._keys:
                if entry[3] == self._chosen and self._state(entry[3]) != "invalid":
                    return entry
        for entry in self._keys:
            if self._state(entry[3]) != "invalid":
                return entry
        return None

    # ------------------------------------------------------------------------------------------------------------ choosing the key in use
    def prefer(self, key_id: str | None) -> None:
        """Take up a choice that was saved earlier (the service reads it from the database when the pool is made). A key id that matches no key is ignored: the key was replaced or removed."""
        self._chosen = key_id if any(e[3] == key_id for e in self._keys) else None

    def _out_of_credits(self, key_id: str) -> str:
        """In words, why ``key_id`` cannot make a request now for want of credits, or an empty string. Both allowances count: the daily one and the monthly one."""
        state, allow = self._state(key_id), self.allow
        if state == "spent_day":
            return "its daily allowance is spent (it resets at 00:00 UTC)"
        if state == "spent_month" or (allow.monthly is not None and self._used(key_id)[1] >= allow.monthly):
            return f"its monthly allowance is spent (it resets on {self.usage.next_month() if self.usage else 'the 1st'})"
        if state == "no_credits":
            return "its included credits and prepaid balance cannot cover a call (top up credits to carry on)"
        return ""

    def _choice_problem(self, key_id: str) -> str:
        """Why ``key_id`` cannot be made the key in use now, in words, or an empty string when it can. A choice is refused while the key in use has spent its allowance: moving to another key then would be
        rotating keys to get past a quota, which OpenFilings' usage policy forbids."""
        slot = next(e[0] for e in self._keys if e[3] == key_id)
        active = self._active()
        if active is not None and active[3] == key_id:
            return ""
        if self._state(key_id) == "invalid":
            return f"{slot} was rejected as invalid or revoked"
        spent = self._out_of_credits(key_id)
        if spent:
            return f"{slot} cannot be used now: {spent}"
        held = self._out_of_credits(active[3]) if active is not None else ""
        if held:
            return (f"{active[0]} is in use and {held}. Another key is not chosen to get past an allowance: OpenFilings' usage policy forbids rotating accounts or API keys to get past its quotas")
        return ""

    @property
    def chosen_id(self) -> str | None:
        """The salted hash of the key the user chose (what the service keeps between runs), or ``None`` when nothing was chosen."""
        return self._chosen

    def choose(self, slot) -> str:
        """Make the key in ``slot`` (``key 2``, or just ``2``) the key in use, because the user asked, and return the slot. Nothing else ever changes the choice, it is kept until the user chooses again, and
        no request is made. It is refused when there is only one key, when there is no such slot, when the key is rejected as invalid or out of credits itself, and while the key in use has spent its
        allowance."""
        with self._lock:
            if self.keyless or len(self._keys) < 2:
                raise FundamentalsError("there is only one key, so there is nothing to choose")
            text = str(slot).strip().lower()
            text = f"key {int(text)}" if text.isdecimal() and len(text) < 4 else text
            entry = next((e for e in self._keys if e[0] == text), None)
            problem = "there is no such key" if entry is None else self._choice_problem(entry[3])
            if problem:
                raise FundamentalsError(problem[0].upper() + problem[1:] + ".")
            self._chosen = entry[3]
            return entry[0]

    def slots(self) -> list[dict]:
        """One line per key for the Data Manager: its slot, the variable it came from, what it has used today and this month, whether it can be used and whether it is the one in use. No key material."""
        allow = self.allow
        active = self._active()
        several = len(self._keys) > 1 and not self.keyless
        out = []
        for slot, variable, _, key_id in self._keys:
            today, month = self._used(key_id)
            state = self._state(key_id)
            role = "unusable" if state == "invalid" else ("active" if active is not None and active[3] == key_id else "standby")
            block = self._block(key_id)
            problem = self._choice_problem(key_id) if several and role != "active" else ""
            out.append({"slot": slot, "variable": variable, "key_id": key_id, "used_today": today, "used_month": month, "daily_limit": allow.daily, "monthly_limit": allow.monthly,
                        "unit": allow.unit, "state": state, "role": role, "resets": block.until if block else None, "chosen": key_id == self._chosen and role == "active",
                        "can_use": several and role != "active" and not problem, "why_not": (problem[0].upper() + problem[1:] + ".") if problem else ""})
        return out

    def remaining_today(self) -> int | None:
        """What the key in use can still spend today (``None`` when there is no limit, ``0`` when it cannot be used)."""
        active = self._active()
        daily = self.allow.daily
        if active is None or self._state(active[3]) != "ready":
            return 0
        return None if daily is None else max(daily - self._used(active[3])[0], 0)

    def remaining_month(self) -> int | None:
        """What is left of the key in use's included monthly allowance (prepaid top-up credits come on top and are not counted here)."""
        active = self._active()
        monthly = self.allow.monthly
        if active is None or monthly is None:
            return None
        return max(monthly - self._used(active[3])[1], 0)

    @property
    def configured(self) -> bool:
        """Whether the pool can make a request at all: it holds a key, or the provider needs none."""
        return self.keyless or len(self._keys) > 0

    # ------------------------------------------------------------------------------------------------------------ use
    def _why(self, entry: tuple[str, str, str, str], cost: int) -> str:
        slot, _, _, key_id = entry
        allow, name = self.allow, self.provider.upper()
        used_day, used_month = self._used(key_id)
        state = self._state(key_id)
        standby = len(self._keys) > 1
        tail = ("; the other keys are standbys, used only if this one is rejected as invalid (the pool does not rotate keys to get past a quota, which the provider's usage policy forbids)"
                if standby else "")
        if state == "no_credits":
            return f"{name} says its included credits and your prepaid balance cannot cover this call ({slot}; {used_month} {allow.unit} used this month): top up credits, or wait for the monthly allowance to reset" + tail
        if state == "spent_month":
            return f"{name} monthly allowance is spent for {slot} ({used_month} {allow.unit} this month); it resets on {self.usage.next_month() if self.usage else 'the 1st'}" + tail
        if state == "spent_day" or (allow.daily is not None and used_day + cost > allow.daily):
            left = max((allow.daily or 0) - used_day, 0)
            need = f" (this call needs {cost}, {left} left)" if state == "ready" else ""
            return f"{name} daily allowance is spent for {slot} ({used_day} of {allow.daily} {allow.unit} today){need}; it resets at 00:00 UTC" + tail
        return f"{name} {slot} cannot be used right now"

    def _acquire(self, cost: int, force: bool = False) -> tuple[str, str, str]:
        if not self.keyless and not self._keys:
            raise MissingKey(f"no {self.provider.upper()} API key is set: add {self.provider.upper()}_API_KEY (or {self.provider.upper()}_API_KEY_1, _2, ... for spares) "
                             "as a secret (on Replit: Tools > Secrets) and restart the app")
        with self._lock:
            entry = self._active()
            if entry is None:
                raise InvalidKey(f"every {self.provider.upper()} key was rejected as invalid or revoked; create a new key in the provider's dashboard and replace it")
            slot, _, key, key_id = entry
            daily = self.allow.daily
            if not force and (self._state(key_id) != "ready" or (daily is not None and self._used(key_id)[0] + cost > daily)):
                raise QuotaExhausted(self._why(entry, cost))
        return slot, key, key_id

    def ensure(self, cost: int) -> None:
        """Raise :class:`QuotaExhausted` unless the key in use can still spend ``cost`` today: a company's lookup and statements are started together or not at all."""
        self._acquire(cost)

    def call(self, fn, endpoint: str = "", symbol: str = "", market: str = "", credits: int = 1, force: bool = False):
        """``fn(key)`` with the key in use, logging the call and what it cost. ``fn`` returns the answer or raises one of the provider errors (never with the key in its message).

        An invalid key is set aside and the next one takes over; an exhausted allowance stops the pool (:class:`QuotaExhausted`) and is remembered, so nothing more is sent until it resets.
        ``force`` makes the call even though the key is remembered as spent: the user pressed Test after topping up, and the provider's answer decides.
        """
        while True:
            slot, key, key_id = self._acquire(credits, force)
            entry = next(e for e in self._keys if e[3] == key_id)
            try:
                result = fn(key)
            except RateLimited as error:
                if error.dimension == "burst":
                    self._log(endpoint, symbol, market, slot, key_id, False, "burst: too many requests at once", 0)
                    raise
                kind = "day" if error.dimension == "daily" else "month"
                self._log(endpoint, symbol, market, slot, key_id, False, f"limit:{kind} {error}", 0)
                raise QuotaExhausted(self._why(entry, credits)) from None
            except CreditsExhausted as error:
                self._log(endpoint, symbol, market, slot, key_id, False, f"balance: {error}", 0)
                raise QuotaExhausted(self._why(entry, credits)) from None
            except InvalidKey as error:
                self._log(endpoint, symbol, market, slot, key_id, False, f"auth: {error}", 0)
                with self._lock:
                    self._invalid.add(key_id)
                continue
            except PlanError as error:
                self._log(endpoint, symbol, market, slot, key_id, False, f"plan: {error}", 0)
                raise
            except SyncPending as error:
                self._log(endpoint, symbol, market, slot, key_id, False, f"pending: {error}", 0)
                raise
            except ProviderError as error:
                self._log(endpoint, symbol, market, slot, key_id, False, str(error), 0)
                raise
            self._log(endpoint, symbol, market, slot, key_id, True, "", credits)
            return result

    def _log(self, endpoint, symbol, market, slot, key_id, success, error, credits) -> None:
        if self.usage is not None:
            self.usage.record(self.provider, endpoint, success, symbol, market, slot, key_id, credits=credits, error=error)
