"""The Fundamentals research workbench: global company fundamentals kept in a local database and turned, step by step, into tested alpha signals for the existing backtester.

    providers -> normalisation -> database -> point-in-time engine -> metrics -> factors -> factor tests -> alpha discovery -> library -> composite and ML -> backtest

The rules that hold throughout (see ``docs/fundamentals.md``): the database, not a provider, is the source of truth; a security is a ``security_id``, never a bare ticker; a figure is used only from the day
it was public; a missing figure is never a zero; and the number of ideas tried is counted, because a result is only as good as the search that found it.
"""
