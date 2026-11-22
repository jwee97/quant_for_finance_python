import pandas as pd


class FakeClock:
    """A clock that only moves when something sleeps, so rate-limit waits cost no real time."""

    def __init__(self, start: float = 1_700_000_000.0):
        self.t, self.slept = start, []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds
