"""Authentication of checkpoint files.

A checkpoint is a pickle-based file. Loading one with ``torch.load(weights_only=False)`` runs whatever code the file contains, so a file must never reach that call unless this installation wrote it. A SHA-256 stored in a
file next to the checkpoint proves nothing about who wrote it (whoever can replace the checkpoint can replace its sidecar), so the unrestricted load is gated by an HMAC under a secret that stays on this machine:

* the secret is ``QUANT_LAB_INTEGRITY_KEY`` (hex) if set, else 32 random bytes kept in ``data/user/.integrity_key`` (created on first use, owner-only permissions where the system has them);
* every checkpoint written here gets a ``<name>.hmac`` sidecar, the HMAC-SHA256 of the file's bytes;
* a checkpoint is read with ``weights_only=True`` first (tensors, numbers, strings, lists and dicts only, which is all a checkpoint of this platform holds); the unrestricted load is used only when that fails AND the HMAC
  verifies. A file that needs the unrestricted load and has no valid HMAC (copied in from elsewhere, written by an older version, or planted) is refused and reported, never executed.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path

ENV = "QUANT_LAB_INTEGRITY_KEY"
_cache: dict[str, bytes] = {}


def key_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "user" / ".integrity_key"


def key() -> bytes:
    env = os.environ.get(ENV)
    if env:
        try:
            return bytes.fromhex(env.strip())
        except ValueError:
            return env.strip().encode()
    p = key_path()
    k = _cache.get(str(p))
    if k:
        return k
    try:
        if p.is_file():
            k = bytes.fromhex(p.read_text().strip())
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            k = secrets.token_bytes(32)
            tmp = p.with_name(p.name + f".{os.getpid()}")
            tmp.write_text(k.hex())
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            try:
                os.replace(tmp, p)                                                                           # two processes starting together: the first key wins, the other reads it
            except OSError:
                tmp.unlink(missing_ok=True)
            k = bytes.fromhex(p.read_text().strip())
    except (OSError, ValueError):
        k = hashlib.sha256((str(p) + str(os.getpid())).encode()).digest()                                    # an unwritable disk: a key valid for this process only (nothing written earlier verifies)
    _cache[str(p)] = k
    return k


def sign(data: bytes) -> str:
    return hmac.new(key(), data, hashlib.sha256).hexdigest()


def verify(data: bytes, signature: str) -> bool:
    return bool(signature) and hmac.compare_digest(sign(data), signature.strip())
