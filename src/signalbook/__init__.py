"""The signal book: follow rules on stocks, futures, crypto and options, read them every day, see what changed and keep a record of every signal issued (see :mod:`src.signalbook.engine`)."""

from .engine import EQUITY_BOARD, HORIZONS, MARKETS, OPTION_RULES, board_rows, changes, consensus, freshness, option_rows, side, snapshot, track_record  # noqa: F401
from .store import Book, BookError, check_webhook  # noqa: F401
