"""One market-data layer: Dashboard -> API -> MarketDataService -> provider adapter (Massive) -> storage -> analytics.

``client`` is the shared HTTP client (key, pooling, pagination, rate limiting, retries, usage); ``models`` the typed requests and answers; ``massive`` the adapter that knows Massive's endpoints and
nothing about options maths or curves; ``storage`` the normalised, versioned, provenance-keeping store; ``service`` the incremental download and gap logic; ``legacy`` the one-off import of the
iTick downloads that earlier versions of the app made. Asset-specific calculations (Greeks, curves, carry) stay in ``assets``.
"""

# The vendor-neutral layer (normalised events with availability timestamps, data contracts, the point-in-time store, loaders and adapters; see ``events`` for the schema). The Massive modules above are imported by
# name where they are used (``from src.marketdata.client import MassiveClient``): importing them here would pull the web layer into every consumer of the events.
from .adapters import (concat_events, events_from_bundle, events_from_curve_panel, events_from_depth, events_from_funding, events_from_futures_table, events_from_fx_market, events_from_lob,   # noqa: E402
                       events_from_option_chains, events_from_prices)
from .contracts import DataContract, QualityReport, UniverseHistory, apply_contract, fill_missing_bars, stale_flags   # noqa: E402
from .events import COLUMNS, EVENT_TYPES                                                                             # noqa: E402
from .loaders import (RestPollingAdapter, ReplayStream, StreamAdapter, WebSocketAdapter, http_fetcher, load_arrow, load_csv, load_events, load_jsonl, load_parquet, load_sql, write_arrow,   # noqa: E402
                      write_csv, write_jsonl, write_parquet, write_sql)
from .schema import assert_valid, normalise_events, validate_events                                                  # noqa: E402
from .store import Obs, PointInTimeStore, PriceSnapshot                                                              # noqa: E402

__all__ = [n for n in dir() if not n.startswith("_")]
