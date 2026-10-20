"""The ways a data provider can fail, told apart because each needs a different response from the sync: stop for the day, stop until the user acts, wait a moment, or tell the user what to change."""

from __future__ import annotations


class FundamentalsError(RuntimeError):
    """Anything the fundamentals module refuses or cannot do; the message is for the user."""


class ProviderError(FundamentalsError):
    """A provider call failed. The message never contains an API key."""


class MissingKey(ProviderError):
    """The provider needs an API key and none is set."""


class InvalidKey(ProviderError):
    """The provider said this key is not valid, or has been revoked (HTTP 401). The next key in the list takes over."""


class RateLimited(ProviderError):
    """The provider said this key's allowance is spent (HTTP 429).

    ``dimension`` says which allowance: ``daily`` (resets at 00:00 UTC), ``monthly`` (resets on the 1st) or ``burst`` (a short guard against too many requests at once: wait a moment).
    ``daily_used`` ... ``monthly_remaining`` carry the provider's own count when it gave one.
    """

    def __init__(self, message: str, retry_after: float | None = None, dimension: str = "daily", detail: dict | None = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.dimension = dimension
        self.detail = dict(detail or {})


class CreditsExhausted(ProviderError):
    """The provider's included credits are spent and the prepaid balance cannot cover the call (HTTP 402). Topping up the balance is the only remedy; waiting for tomorrow does not help."""

    def __init__(self, message: str, balance: float | None = None, required: float | None = None):
        super().__init__(message)
        self.balance = balance
        self.required = required


class PlanError(ProviderError):
    """The provider's plan does not include what was asked (HTTP 403): every key on the same plan would say the same."""

    def __init__(self, message: str, feature: str | None = None, min_plan: str | None = None):
        super().__init__(message)
        self.feature = feature
        self.min_plan = min_plan


class NotFound(ProviderError):
    """The provider has nothing for this symbol."""


class SyncPending(ProviderError):
    """The provider is still collecting this company's filings (HTTP 202) and asks to be asked again after ``retry_after`` seconds. Not a failure and not charged."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExhausted(FundamentalsError):
    """Every key of a provider has used its allowance, or the provider's budget is spent. The sync stops cleanly and the rest waits."""
