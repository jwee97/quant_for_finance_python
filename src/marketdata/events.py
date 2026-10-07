"""The normalised market-data event: one schema for every asset class and every vendor.

Every observation (a quote, a trade, a daily bar, a futures settlement, a perpetual's funding rate, a yield curve, a reference value) is one row with the same columns. Two timestamps
are mandatory in meaning: ``timestamp`` is WHEN THE THING HAPPENED (the observation time) and ``available_at`` is WHEN A TRADER COULD FIRST HAVE KNOWN IT. The engine delivers an event to
strategies at ``available_at``, never earlier, which is what makes a backtest free of look-ahead by construction: a settlement price stamped 16:00 and published at 17:30 is invisible
to a 16:30 decision, and a revised number is a NEW row with a later ``available_at`` and the same ``timestamp``.

==================  ===============================================================================================================================
column              meaning
==================  ===============================================================================================================================
``timestamp``       observation time, UTC (tz-naive; the loaders convert from the vendor's zone)
``available_at``    time the value became known, UTC; must be >= ``timestamp``
``vendor_timestamp``  the vendor's own stamp (kept for audit; never used for ordering)
``instrument_id``   an instrument id, a curve id (``USD-OIS``), an index id (``SOFR-3M``) or a reference series id
``event_type``      one of ``EVENT_TYPES``
``bid`` ``ask`` ``bid_size`` ``ask_size``  top-of-book quote; ``depth`` (in ``reference_values``) may hold deeper levels
``trade`` ``volume``     last trade price and the volume traded in the interval (``bar``) or on the trade
``open`` ``high`` ``low`` ``close``  bar prices (``close`` is the bar's last price)
``open_interest``   contracts outstanding
``funding``         a funding rate for the interval ending at ``timestamp`` (perpetuals), as a fraction
``mark_price``      the exchange mark used for margin and liquidation
``settlement``      an official settlement price (futures, options)
``value``           a generic numeric value (a fixing, an index level, a macro release)
``curve_values``    ``{tenor_in_years: rate}`` for a curve event
``reference_values``  any other structured payload (corporate-action terms, listing status, depth levels)
``source`` ``revision``  provenance and the revision number (0 = first release)
==================  ===============================================================================================================================

Event types: ``quote``, ``trade``, ``bar``, ``settlement``, ``mark``, ``funding``, ``funding_predicted``, ``open_interest``, ``curve``, ``fixing``, ``reference``,
``corporate_action``, ``listing``.
"""

from __future__ import annotations

EVENT_TYPES = ("quote", "trade", "bar", "settlement", "mark", "funding", "funding_predicted", "open_interest", "curve", "fixing", "reference", "corporate_action", "listing")

NUMERIC_COLUMNS = ["bid", "ask", "bid_size", "ask_size", "trade", "volume", "open", "high", "low", "close", "open_interest", "funding", "mark_price", "settlement", "value"]
OBJECT_COLUMNS = ["curve_values", "reference_values", "source"]
COLUMNS = ["timestamp", "available_at", "vendor_timestamp", "instrument_id", "event_type"] + NUMERIC_COLUMNS + OBJECT_COLUMNS + ["revision"]

# the fields each event type must carry (any of the alternatives in a tuple)
REQUIRED = {"quote": [("bid", "ask")], "trade": [("trade",)], "bar": [("close",)], "settlement": [("settlement",)], "mark": [("mark_price",)], "funding": [("funding",)],
            "funding_predicted": [("funding",)], "open_interest": [("open_interest",)], "curve": [("curve_values",)], "fixing": [("value",)], "reference": [("value",), ("reference_values",)],
            "corporate_action": [("reference_values",)], "listing": [("reference_values",)]}
PRICE_COLUMNS = ["bid", "ask", "trade", "open", "high", "low", "close", "mark_price", "settlement"]
