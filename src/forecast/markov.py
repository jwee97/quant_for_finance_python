"""Markov forecasters: the market is in one of a few states, the state changes with fixed probabilities that depend only on the state it is in now, and each state has its own kind of day.

``markov_returns``     the states are the size and sign of the day's move (five equal-sized groups from a big fall to a big rise), the transition matrix is counted from the last five years, and a day's
                       return is drawn from the days that were in its state. Whether today's move helps predict tomorrow's at all is itself testable (``test_markov_order``): for daily moves it barely does.
``markov_volatility``  the states are the volatility of the last month (calm, ordinary, stormy). These states last for weeks, so the chain carries real information: the next days are drawn from the days
                       that followed calm, ordinary or stormy months, and the chain decides how long the storm lasts.
``hmm``                a hidden Markov model (Hamilton 1989; Rabiner 1989): the states are not observed but inferred. Two regimes, each with its own average return and volatility, fitted by the
                       Baum-Welch algorithm; the forecast starts from the probability of being in each regime today (the filtered probability: it uses today and earlier, never later days) and runs the chain forward.
"""

from __future__ import annotations

import numpy as np

from ..models.regimes import fit_hmm, forward_filter, log_emission
from ..probability.markov import MarkovChain, estimate_transition_matrix, test_markov_order
from .base import Forecast, ForecastError, Forecaster, History, cumulate
from .features import rolling_rms

TRADING_DAYS = 252


def _next_state(rng, cum_rows: np.ndarray, state: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Move every path one step: the new state is where ``u`` falls in the cumulative row of its current state."""
    nxt = (u[:, None] > cum_rows[state]).sum(axis=1)
    return np.minimum(nxt, cum_rows.shape[1] - 1)


def _pool_matrix(pools: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The days of each state as a padded matrix, and how many there are, so that a random day of every path's state can be drawn at once."""
    counts = np.array([len(p) for p in pools])
    if counts.min() < 5:
        raise ForecastError("a state has too few days to draw from")
    M = np.zeros((len(pools), counts.max()))
    for k, p in enumerate(pools):
        M[k, :len(p)] = p
    return M, counts


def _walk(rng, P: np.ndarray, start: np.ndarray, pools: list[np.ndarray], steps: int, emit_after_move: bool) -> np.ndarray:
    """Simulate paths of a chain whose states emit returns drawn from ``pools``. With ``emit_after_move`` the return of a day is drawn from the state it moves into (the state *is* the day's move);
    otherwise from the state it leaves (the state is the volatility the day starts in)."""
    n = len(start)
    cum = np.cumsum(P, axis=1)
    M, counts = _pool_matrix(pools)
    state = start.copy()
    out = np.empty((n, steps))
    for k in range(steps):
        if emit_after_move:
            state = _next_state(rng, cum, state, rng.random(n))
            out[:, k] = M[state, (rng.random(n) * counts[state]).astype(int)]
        else:
            out[:, k] = M[state, (rng.random(n) * counts[state]).astype(int)]
            state = _next_state(rng, cum, state, rng.random(n))
    return out


class MarkovReturns(Forecaster):
    name = "markov_returns"
    short = "Markov, daily move"
    label = "Markov chain on the day's move"
    family = "Markov"
    text = "Five states from a big fall to a big rise, transition probabilities counted from the last five years, returns drawn from the days in each state."
    assumes = "Tomorrow's kind of day depends on today's kind of day, and on nothing before it."
    fails = "For daily moves the dependence is weak (a test is reported), so it is close to a reshuffling of past days; the sign carries almost no memory, only the size does."
    min_history = 500
    mean_signal = True
    K = 5

    def fit(self, h: History, horizons):
        r = h.r[-self.window:]
        if len(r) < 250:
            raise ForecastError("too few days for a Markov chain")
        edges = np.quantile(r, np.linspace(0, 1, self.K + 1)[1:-1])
        labels = np.searchsorted(edges, r)
        chain = estimate_transition_matrix(labels, list(range(self.K)), smoothing=1.0)
        pools = [r[labels == k] for k in range(self.K)]
        order = test_markov_order(labels)
        return {"edges": edges, "P": chain.P, "pools": pools, "order_p": order["pvalue"], "chain": chain}

    def predict(self, state, h: History, horizons, n, rng):
        s = state
        start = np.full(n, int(np.searchsorted(s["edges"], h.r[-1])))
        sim = _walk(rng, s["P"], start, s["pools"], max(horizons), emit_after_move=True)
        chain: MarkovChain = s["chain"]
        stat = chain.stationary().to_numpy()
        return Forecast(cumulate(sim), {"state_today": int(start[0]), "states": ["big fall", "fall", "flat", "rise", "big rise"], "transition_matrix": s["P"].tolist(), "stationary": stat.tolist(),
                                        "order_test_p": float(s["order_p"]), "state_mean_daily": [float(p.mean()) for p in s["pools"]]})


class MarkovVolatility(Forecaster):
    name = "markov_volatility"
    short = "Markov, volatility"
    label = "Markov chain on volatility (calm, ordinary, stormy)"
    family = "Markov"
    text = "Three states by the volatility of the last month. They last for weeks, so the chain says how long a storm tends to last, and the days are drawn from those that followed such months."
    assumes = "What the next days look like depends on how volatile the last month was, and the state changes with fixed probabilities."
    fails = "The three levels are crude (a storm is more than one state), and the states cannot see a change that has not yet shown up in a month of returns."
    min_history = 500
    mean_signal = True
    K = 3
    DAYS = 21

    def fit(self, h: History, horizons):
        r = h.r[-self.window:]
        if len(r) < 250:
            raise ForecastError("too few days for a Markov chain")
        vol = rolling_rms(r, self.DAYS)
        ok = np.isfinite(vol)
        edges = np.quantile(vol[ok], np.linspace(0, 1, self.K + 1)[1:-1])
        s = np.searchsorted(edges, vol[ok])
        r_ok = r[ok]
        chain = estimate_transition_matrix(s, list(range(self.K)), smoothing=1.0)
        pools = [r_ok[1:][s[:-1] == k] for k in range(self.K)]              # the return of the day AFTER a day in state k
        levels = [float(np.sqrt(np.mean(r_ok[s == k] ** 2)) * np.sqrt(TRADING_DAYS)) for k in range(self.K)]
        return {"edges": edges, "P": chain.P, "pools": pools, "levels": levels, "chain": chain}

    def predict(self, state, h: History, horizons, n, rng):
        s = state
        vol = float(np.sqrt(np.mean(h.r[-self.DAYS:] ** 2)))
        now = int(np.searchsorted(s["edges"], vol))
        sim = _walk(rng, s["P"], np.full(n, now), s["pools"], max(horizons), emit_after_move=False)
        chain: MarkovChain = s["chain"]
        return Forecast(cumulate(sim), {"state_today": now, "states": ["calm", "ordinary", "stormy"], "transition_matrix": s["P"].tolist(), "stationary": chain.stationary().to_numpy().tolist(),
                                        "expected_days_in_state": chain.expected_duration().to_numpy().tolist(), "state_annual_volatility": s["levels"], "volatility_month_annual": vol * np.sqrt(TRADING_DAYS)})


class HiddenMarkov(Forecaster):
    name = "hmm"
    short = "Hidden Markov"
    label = "Hidden Markov model (two regimes)"
    family = "Markov"
    text = "Two unobserved regimes, each with its own average return and volatility, fitted to the last five years; the forecast starts from today's probability of each regime."
    assumes = "The market switches between a calm and a turbulent regime with fixed probabilities, and returns inside a regime are normal."
    fails = "A regime is only recognised after it has been under way for a while; the two regimes are a simplification; the average return of each regime is poorly estimated."
    min_history = 600
    mean_signal = True
    heavy = True
    N_STATES = 2

    def fit(self, h: History, horizons):
        r = h.r[-self.window:] * 100.0                                     # percent: a well-scaled input for the optimiser
        if len(r) < 500:
            raise ForecastError("too few days for a hidden Markov model")
        try:
            fit = fit_hmm(r.reshape(-1, 1), self.N_STATES, n_init=3, n_iter=200, seed=7)
        except RuntimeError as error:
            raise ForecastError(str(error)) from None
        if not np.isfinite(fit.loglik) or fit.is_degenerate:
            raise ForecastError("the hidden Markov model did not give two usable regimes")
        return fit

    def predict(self, state, h: History, horizons, n, rng):
        fit = state
        x = (h.r[-self.lookback:] * 100.0).reshape(-1, 1)
        alpha, _ = forward_filter(log_emission(x, fit), fit.startprob, fit.transmat)
        prob_now = alpha[-1]
        A = fit.transmat
        mean = fit.means[:, 0] / 100.0
        sd = np.sqrt(fit.covars[:, 0, 0]) / 100.0
        steps = max(horizons)
        cum = np.cumsum(A, axis=1)
        reg = (rng.random(n)[:, None] > np.cumsum(prob_now @ A)).sum(axis=1)             # the regime tomorrow
        reg = np.minimum(reg, fit.n_states - 1)
        out = np.empty((n, steps))
        for k in range(steps):
            out[:, k] = mean[reg] + sd[reg] * rng.standard_normal(n)
            reg = _next_state(rng, cum, reg, rng.random(n))
        stay = np.diag(A)
        return Forecast(cumulate(out), {"probability_of_each_regime_today": prob_now.tolist(), "regime_annual_mean": (mean * TRADING_DAYS).tolist(), "regime_annual_volatility": (sd * np.sqrt(TRADING_DAYS)).tolist(),
                                        "transition_matrix": A.tolist(), "expected_days_in_regime": (1.0 / np.maximum(1.0 - stay, 1e-9)).tolist(), "stationary": fit.stationary().tolist()})
