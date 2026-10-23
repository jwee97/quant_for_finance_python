"""Data providers: the only code that talks to the outside world for fundamentals. Each returns the application's own record types and nothing else leaves its module."""

from __future__ import annotations

from .base import FinancialsResult, FundamentalDataProvider

__all__ = ["FundamentalDataProvider", "FinancialsResult", "make_provider", "PROVIDER_NAMES"]

PROVIDER_NAMES = ("openfilings", "sec_edgar", "simulated")


def make_provider(name: str, **kwargs):
    """The provider called ``name`` (``openfilings``, ``sec_edgar`` or ``simulated``); keyword arguments go to its constructor."""
    if name == "openfilings":
        from .openfilings import OpenFilingsProvider
        return OpenFilingsProvider(**kwargs)
    if name == "sec_edgar":
        from .sec_edgar import SecEdgarProvider
        return SecEdgarProvider(**kwargs)
    if name == "simulated":
        from .simulated import SimulatedProvider
        return SimulatedProvider(**kwargs)
    raise ValueError(f"unknown fundamentals provider {name!r}; choose from {', '.join(PROVIDER_NAMES)}")
