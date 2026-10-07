"""Currency conversion from observed pair prices: the rate of any currency or coin into the ledger's base currency.

The ledger holds balances in many currencies (dollars, euros, a stablecoin, bitcoin). Their values in the base currency come from the marks of the pairs the portfolio trades or
observes (``EURUSD``, ``BTCUSDT``, ``USDTUSD``): :class:`CurrencyConverter` stores each pair as ``units of quote per unit of base`` and finds a conversion path by breadth-first
search, multiplying along it (inverting an edge when walking it backwards). A currency with no path to the base has NO rate; asking for it raises, because guessing a rate would hide
exactly the exposure a multi-currency book needs to see.
"""

from __future__ import annotations

from collections import deque


class CurrencyConverter:
    def __init__(self, base: str = "USD", pegs: dict[str, float] | None = None):
        self.base = base
        self.pairs: dict[tuple[str, str], float] = {}
        for ccy, rate in (pegs or {}).items():                       # fixed relationships, e.g. {"USDT": 1.0} against the base
            self.pairs[(ccy, base)] = rate

    def set_pair(self, base_ccy: str, quote_ccy: str, rate: float) -> None:
        """Record that one unit of ``base_ccy`` is worth ``rate`` units of ``quote_ccy``."""
        if not rate > 0:
            raise ValueError(f"rate for {base_ccy}{quote_ccy} must be positive, got {rate}")
        self.pairs[(base_ccy, quote_ccy)] = float(rate)

    def has_rate(self, ccy: str) -> bool:
        try:
            self.rate(ccy)
            return True
        except KeyError:
            return False

    def rate(self, ccy: str) -> float:
        """Value of one unit of ``ccy`` in the base currency."""
        if ccy == self.base:
            return 1.0
        graph: dict[str, list[tuple[str, float]]] = {}
        for (a, b), r in self.pairs.items():
            graph.setdefault(a, []).append((b, r))                   # 1 a = r b
            graph.setdefault(b, []).append((a, 1.0 / r))
        seen = {ccy: 1.0}                                            # value of 1 unit of ccy in each reached currency
        queue = deque([ccy])
        while queue:
            cur = queue.popleft()
            for nxt, r in graph.get(cur, []):
                if nxt not in seen:
                    seen[nxt] = seen[cur] * r
                    if nxt == self.base:
                        return seen[nxt]
                    queue.append(nxt)
        raise KeyError(f"no conversion path from {ccy} to {self.base}: observe a pair that links them (known pairs: {sorted(f'{a}{b}' for a, b in self.pairs)})")

    def convert(self, amount: float, ccy: str) -> float:
        return amount * self.rate(ccy)
