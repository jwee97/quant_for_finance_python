"""The research workspace: sessions, the shared research context, saved artifacts and the checks that say whether two results can be compared.

``context``  the typed, validated context a session carries (assets, dates, data source, costs, the strategy, the model, the regime model, the option assumptions) and the compatibility rules
``store``    sessions and artifacts as small versioned JSON files under ``data/user/research``, written atomically, with a migration path for older records
``regime_lab``  the Market Regime Lab: filtered (causal) and smoothed (retrospective) regimes and the performance of a return stream by regime
``compare``  the Research Comparison: an honest common window, the differences in assumptions that stop a ranking, synchronised series and a reproducible report

Nothing here computes a financial result of its own that an engine elsewhere already computes: a backtest, a model or a hedge is run where it always was and only its outcome is kept.
"""

from .context import ContextError, compatibility, validate_context  # noqa: F401
from .store import ResearchStore, StoreError  # noqa: F401
