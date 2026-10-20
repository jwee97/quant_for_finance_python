"""The ways a data provider can fail, told apart because each needs a different response from the sync: rotate to another key, stop, or tell the user what to change."""

from __future__ import annotations


class FundamentalsError(RuntimeError):
    """Anything the fundamentals module refuses or cannot do; the message is for the user."""


class ProviderError(FundamentalsError):
    """A provider call failed. The message never contains an API key."""


class MissingKey(ProviderError):
    """The provider needs an API key and none is set."""


class InvalidKey(ProviderError):
    """The provider said this key is not valid (HTTP 401). Another key may still work."""


class RateLimited(ProviderError):
    """The provider said this key's allowance is spent (HTTP 429). Another key may still have some."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class PlanError(ProviderError):
    """The provider's plan does not include what was asked (HTTP 402 or 403): every key on the same plan will say the same, so keys are not rotated."""


class NotFound(ProviderError):
    """The provider has nothing for this symbol."""


class QuotaExhausted(FundamentalsError):
    """Every key of a provider has used its allowance for the day, or the provider's budget is spent."""
