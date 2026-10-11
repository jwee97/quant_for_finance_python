"""Optional packages: missing, or installed but unable to load.

A package can be "installed" as far as pip is concerned and still not work, because it needs a system library the machine does not have. Importing LightGBM or XGBoost on a machine without the
OpenMP runtime raises ``OSError: libgomp.so.1: cannot open shared object file`` (XGBoost: its own ``XGBoostError``), not ``ImportError``. Code that treats only ``ImportError`` as "not available"
lets that error escape and takes the whole page down with it. Everything optional goes through ``import_optional`` or ``unavailable_reason`` so that both cases mean the same thing: this feature is
unavailable here, and here is why and what to do about it.
"""

from __future__ import annotations

import sys
from types import ModuleType


class OptionalPackageError(ImportError):
    """An optional package is not installed or cannot load. The message says which, and what to do."""


def _first_line(exc: BaseException) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return lines[0] if lines else type(exc).__name__


def _openmp_hint() -> str:
    if sys.platform == "darwin":
        return "it needs the OpenMP runtime, which this Mac does not have: run `brew install libomp`, then restart the app"
    if sys.platform.startswith("win"):
        return "it needs the Microsoft Visual C++ runtime: install the latest Visual C++ Redistributable, then restart the app"
    return "it needs the OpenMP runtime (libgomp), which this machine does not have: on Debian or Ubuntu run `apt-get install -y libgomp1`, then restart the app"


def describe_failure(name: str, exc: BaseException) -> str:
    """One sentence on why ``import name`` failed."""
    if isinstance(exc, ModuleNotFoundError):
        missing = (exc.name or name).partition(".")[0]
        if missing == name.partition(".")[0]:
            return f"the '{name}' package is not installed (pip install {name})"
        return f"the '{name}' package is installed but needs the '{missing}' package, which is not (pip install {missing})"
    text = str(exc).lower()
    if "libgomp" in text or "openmp" in text or "libomp" in text or "vcomp" in text:
        return f"the '{name}' package is installed but cannot load ({_first_line(exc)}); {_openmp_hint()}"
    return f"the '{name}' package is installed but cannot load ({_first_line(exc)})"


def import_optional(name: str) -> ModuleType:
    """The module, or an ``OptionalPackageError`` (an ``ImportError``) saying why it cannot be used. Any exception raised while the package loads counts: ``ImportError`` when it is absent,
    ``OSError`` when a system library is missing, the package's own error type when it reports that itself."""
    try:
        return __import__(name)
    except Exception as exc:                                                          # noqa: BLE001 - whatever a half-installed package raises, it is unavailable
        raise OptionalPackageError(describe_failure(name, exc)) from exc


def unavailable_reason(name: str) -> str | None:
    """``None`` when the package imports, otherwise the reason it cannot be used."""
    try:
        import_optional(name)
    except OptionalPackageError as exc:
        return str(exc)
    return None
