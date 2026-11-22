"""One market-data layer: Dashboard -> API -> MarketDataService -> provider adapter (Massive) -> storage -> analytics.

``client`` is the shared HTTP client (key, pooling, pagination, rate limiting, retries, usage); ``models`` the typed requests and answers; ``massive`` the adapter that knows Massive's endpoints and
nothing about options maths or curves; ``storage`` the normalised, versioned, provenance-keeping store; ``service`` the incremental download and gap logic; ``legacy`` the one-off import of the
iTick downloads that earlier versions of the app made. Asset-specific calculations (Greeks, curves, carry) stay in ``assets``.
"""
