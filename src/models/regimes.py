"""Regime detection models (Generation 2, Priority 1).

Three model families, chosen because they fail in different ways.

**Hidden Markov model** (Hamilton 1989). A latent Markov chain with Gaussian
emissions. The temporal structure is what makes the labels persistent, and it
is also the source of the one trap this module is organised around. The
*smoothed* posterior, P(S_t | x_1..x_T), conditions on the whole sample and so
on the future of day t; the *filtered* posterior, P(S_t | x_1..x_t), does not.
A backtest that trades on smoothed probabilities is look-ahead, and it looks
spectacular, which is why it is so tempting. Everything tradable here is
filtered, with parameters estimated only on data before each refit date.

**Gaussian mixture.** The same emissions with no Markov chain: each day is
classified on its own. Included as the control that shows what the temporal
structure buys (persistence) and costs (nothing here, but it needs the chain
to be true).

**Bayesian online change-point detection** (Adams and MacKay 2007). A posterior
over the run length, the time since the last change. With a constant hazard
the prior probability of a change on any day is the same, so P(r_t = 0) is
uninformative by construction (it equals the hazard). What carries the
information is the mass on SHORT run lengths: when the data stop looking like
the current regime, the posterior moves to runs that began recently.

State labels are ordered by the variance of one emission column, so "state K-1"
always means the highest-variance state. The ordering is computed from the
parameters of each fit, never from the outcome of a backtest.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Gaussian HMM
# ---------------------------------------------------------------------------
@dataclass
class HMMFit:
    """Parameters of a fitted Gaussian HMM."""

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray                 # (K, d, d)
    loglik: float
    n_iter: int = 0
    converged: bool = True

    @property
    def n_states(self) -> int:
        return len(self.startprob)

    @property
    def is_degenerate(self) -> bool:
        """A state whose covariance has collapsed onto a few observations.

        The likelihood of a Gaussian mixture is unbounded: a state that sits on
        one observation and shrinks its covariance toward a point scores ever
        higher. Real return data never needs to, but perturbed data (a price
        series spliced to a time-reversed copy of itself has a 300% day) do.
        """
        for cov in self.covars:
            eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
            if eig.max() <= 0 or eig.min() <= 1e-8 * eig.max():
                return True
        return False

    def ordered(self, column: int = 0) -> "HMMFit":
        """Relabel the states by increasing variance of emission ``column``."""
        order = np.argsort([float(c[column, column]) for c in self.covars], kind="stable")
        return HMMFit(self.startprob[order], self.transmat[np.ix_(order, order)],
                      self.means[order], self.covars[order], self.loglik,
                      self.n_iter, self.converged)

    def stationary(self) -> np.ndarray:
        values, vectors = np.linalg.eig(self.transmat.T)
        vector = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
        vector = np.abs(vector)
        return vector / vector.sum()


def _fit_hmmlearn(X: np.ndarray, n_states: int, n_iter: int, tol: float, seed: int,
                  start: HMMFit | None) -> HMMFit | None:
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter, tol=tol,
                        random_state=seed, init_params="" if start is not None else "stmc",
                        params="stmc")
    if start is not None:
        model.startprob_ = start.startprob.copy()
        model.transmat_ = start.transmat.copy()
        model.means_ = start.means.copy()
        model.covars_ = start.covars.copy()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
            loglik = float(model.score(X))
    except Exception:                                    # a degenerate start; try the next one
        return None
    parameters = (model.startprob_, model.transmat_, model.means_, np.asarray(model.covars_))
    if not np.isfinite(loglik) or not all(np.isfinite(p).all() for p in parameters):
        return None
    return HMMFit(model.startprob_.copy(), model.transmat_.copy(), model.means_.copy(),
                  np.asarray(model.covars_).copy(), loglik,
                  int(model.monitor_.iter), bool(model.monitor_.converged))


def fit_hmm(X, n_states: int = 2, n_init: int = 5, n_iter: int = 300, tol: float = 1e-4,
            seed: int = 0, init_from: HMMFit | None = None, label_column: int = 0) -> HMMFit:
    """Maximum-likelihood Gaussian HMM, best of several starts.

    ``init_from`` adds a warm start (the previous solution) to the fresh random
    starts; the highest likelihood wins. The returned states are ordered by the
    variance of ``label_column``.
    """
    data = np.ascontiguousarray(np.asarray(X, dtype=float))
    candidates: list[HMMFit] = []
    if init_from is not None:
        candidates.append(_fit_hmmlearn(data, n_states, n_iter, tol, seed, init_from))
    for i in range(int(n_init)):
        candidates.append(_fit_hmmlearn(data, n_states, n_iter, tol, seed + 17 * (i + 1), None))
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        raise RuntimeError("no HMM start produced a finite likelihood")
    sound = [c for c in candidates if not c.is_degenerate]
    return max(sound or candidates, key=lambda f: f.loglik).ordered(label_column)


def log_emission(X, fit: HMMFit) -> np.ndarray:
    """log p(x_t | S_t = k) for every day and state."""
    data = np.asarray(X, dtype=float)
    out = np.empty((len(data), fit.n_states))
    for k in range(fit.n_states):
        cov = 0.5 * (fit.covars[k] + fit.covars[k].T)
        try:
            out[:, k] = multivariate_normal.logpdf(data, mean=fit.means[k], cov=cov)
        except np.linalg.LinAlgError:                    # same jitter the library applies in its own likelihood
            out[:, k] = multivariate_normal.logpdf(data, mean=fit.means[k], cov=cov + 1e-7 * np.eye(len(cov)))
    return out


def forward_filter(log_b: np.ndarray, startprob: np.ndarray, transmat: np.ndarray):
    """Filtered state probabilities P(S_t | x_1..x_t) and the log-likelihood.

    Day t uses the data of day t and earlier and nothing else; there is no
    backward pass. The sum of the log normalisers is log P(x_1..x_T), which
    must equal the library's own score (the tests check it).
    """
    T, K = log_b.shape
    alpha = np.empty((T, K))
    loglik = 0.0
    previous = None
    for t in range(T):
        prior = startprob if previous is None else previous @ transmat
        peak = log_b[t].max()
        weight = np.exp(log_b[t] - peak) * prior
        total = weight.sum()
        alpha[t] = weight / total
        loglik += float(np.log(total) + peak)
        previous = alpha[t]
    return alpha, loglik


def backward_smooth(log_b: np.ndarray, transmat: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Smoothed posterior P(S_t | x_1..x_T): the look-ahead version.

    Provided for the control that shows how much a backtest on it would
    overstate. Nothing tradable may use it.
    """
    T, K = alpha.shape
    gamma = np.empty((T, K))
    gamma[-1] = alpha[-1]
    beta = np.ones(K)
    for t in range(T - 2, -1, -1):
        emission = np.exp(log_b[t + 1] - log_b[t + 1].max())
        beta = transmat @ (emission * beta)
        beta = beta / beta.sum()
        posterior = alpha[t] * beta
        gamma[t] = posterior / posterior.sum()
    return gamma


def hmm_posteriors(X, fit: HMMFit) -> dict:
    """Filtered and smoothed posteriors of ``X`` under fixed parameters."""
    log_b = log_emission(X, fit)
    alpha, loglik = forward_filter(log_b, fit.startprob, fit.transmat)
    return {"filtered": alpha, "smoothed": backward_smooth(log_b, fit.transmat, alpha),
            "loglik": loglik}


def _digest(array: np.ndarray, *parts) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(array).tobytes())
    h.update(repr(parts).encode())
    return h.hexdigest()


@dataclass
class WalkForwardHMM:
    """Out-of-sample filtered probabilities from an expanding-window HMM.

    ``windows`` keeps, for every refit, the fitted parameters and the filtered
    probabilities of the WHOLE history up to the end of that window under those
    parameters. Rows before ``start`` are in-sample for that fit (legitimate
    training information at the refit date); rows from ``start`` on are the
    out-of-sample probabilities.
    """

    index: pd.DatetimeIndex
    prob: pd.DataFrame
    windows: list[dict] = field(default_factory=list)

    @property
    def n_states(self) -> int:
        return self.prob.shape[1]

    @property
    def p_high(self) -> pd.Series:
        return self.prob.iloc[:, -1].rename("p_high")

    @property
    def labels(self) -> pd.Series:
        valid = self.prob.notna().all(axis=1)
        out = pd.Series(np.nan, index=self.index, name="state")
        out[valid] = self.prob[valid].to_numpy().argmax(axis=1)
        return out

    @property
    def first_date(self):
        valid = self.prob.dropna(how="any")
        return valid.index[0] if len(valid) else None


def walk_forward_hmm(X: pd.DataFrame, n_states: int = 2, min_train: int = 750,
                     refit_every: int = 126, n_init: int = 5, n_init_refit: int = 1,
                     n_iter: int = 300, tol: float = 1e-4, seed: int = 11,
                     label_column: int = 0, cache: dict | None = None) -> WalkForwardHMM:
    """Refit on data strictly before each window, filter through the window.

    At the refit that starts at row ``r`` the model is estimated on rows
    ``0..r-1`` only. The filter then runs over rows ``0..end-1`` with those
    frozen parameters, so the probability on day t uses the parameters of a fit
    that never saw day t or later and the data of days up to t.

    ``cache`` maps a digest of (training data, settings) to a fit. The digest
    is of the data the function actually passes to the fit, so a leak of future
    rows into training would change the digest and force a refit: the cache can
    skip repeated work, never hide a dependence on the future.
    """
    values = np.ascontiguousarray(X.to_numpy(dtype=float))
    index = pd.DatetimeIndex(X.index)
    T = len(values)
    prob = pd.DataFrame(np.nan, index=index, columns=[f"state_{k}" for k in range(n_states)])
    windows: list[dict] = []
    previous: HMMFit | None = None
    for j, start in enumerate(range(int(min_train), T, int(refit_every))):
        end = min(start + int(refit_every), T)
        train = values[:start]
        window_seed = seed + 1000 * j
        key = _digest(train, n_states, n_init if previous is None else n_init_refit,
                      n_iter, tol, window_seed, label_column)
        fit = cache.get(key) if cache is not None else None
        if fit is None:
            fit = fit_hmm(train, n_states, n_init if previous is None else n_init_refit,
                          n_iter, tol, window_seed, init_from=previous, label_column=label_column)
            if cache is not None:
                cache[key] = fit
        previous = fit
        log_b = log_emission(values[:end], fit)
        alpha, _ = forward_filter(log_b, fit.startprob, fit.transmat)
        prob.iloc[start:end] = alpha[start:end]
        windows.append({"start": start, "end": end, "date": index[start], "fit": fit,
                        "alpha": alpha})
    return WalkForwardHMM(index=index, prob=prob, windows=windows)


def full_sample_hmm(X: pd.DataFrame, n_states: int = 2, n_init: int = 5, n_iter: int = 300,
                    tol: float = 1e-4, seed: int = 11, label_column: int = 0) -> dict:
    """One fit on the whole sample: the look-ahead controls.

    ``filtered_full_theta`` removes the smoothing look-ahead but keeps the
    parameter look-ahead; ``smoothed`` has both. Separating them says which of
    the two does the damage.
    """
    fit = fit_hmm(X, n_states, n_init, n_iter, tol, seed, label_column=label_column)
    posteriors = hmm_posteriors(X, fit)
    columns = [f"state_{k}" for k in range(n_states)]
    return {
        "fit": fit,
        "filtered_full_theta": pd.DataFrame(posteriors["filtered"], index=X.index, columns=columns),
        "smoothed": pd.DataFrame(posteriors["smoothed"], index=X.index, columns=columns),
    }


# ---------------------------------------------------------------------------
# Gaussian mixture: the same emissions, no Markov chain
# ---------------------------------------------------------------------------
def walk_forward_gmm(X: pd.DataFrame, n_components: int = 2, min_train: int = 750,
                     refit_every: int = 126, n_init: int = 5, seed: int = 11,
                     label_column: int = 0) -> pd.DataFrame:
    """Out-of-sample component probabilities, each day classified on its own."""
    from sklearn.mixture import GaussianMixture

    values = X.to_numpy(dtype=float)
    out = pd.DataFrame(np.nan, index=X.index, columns=[f"state_{k}" for k in range(n_components)])
    for j, start in enumerate(range(int(min_train), len(values), int(refit_every))):
        end = min(start + int(refit_every), len(values))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianMixture(n_components, covariance_type="full", n_init=int(n_init),
                                    random_state=seed + 1000 * j).fit(values[:start])
        order = np.argsort([float(c[label_column, label_column]) for c in model.covariances_],
                           kind="stable")
        out.iloc[start:end] = model.predict_proba(values[start:end])[:, order]
    return out


# ---------------------------------------------------------------------------
# Bayesian online change-point detection
# ---------------------------------------------------------------------------
@dataclass
class BOCPDResult:
    index: pd.DatetimeIndex
    short_mass: pd.DataFrame           # P(run length <= s) for each requested s
    expected_run_length: pd.Series
    map_run_length: pd.Series
    run_length_bins: pd.DataFrame      # probability mass in coarse run-length bins
    prior: dict
    start: pd.Timestamp                # first date with a posterior (after burn-in)


def _student_t_logpdf(x: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray,
                      beta: np.ndarray) -> np.ndarray:
    """Posterior-predictive of a Normal with unknown mean and variance (Normal-Inverse-Gamma)."""
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    z = (x - mu) ** 2 / (nu * scale2)
    return (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi * scale2) - ((nu + 1.0) / 2.0) * np.log1p(z))


def bocpd(x: pd.Series, hazard_lambda: float = 250.0, burn_in: int = 250, kappa0: float = 1.0,
          alpha0: float = 1.0, short_runs: tuple[int, ...] = (10,),
          bins: tuple[int, ...] = (0, 1, 5, 10, 20, 60, 120, 250)) -> BOCPDResult:
    """Adams-MacKay online change-point detection on one series.

    The prior location and scale come from the first ``burn_in`` observations
    only, and the recursion starts on the next day, so at every date the prior
    depends on data that were already observed. The observation model is a
    Normal with unknown mean and variance, so a change in either is a change.
    The hazard is constant, 1/``hazard_lambda`` per day.
    """
    series = x.dropna()
    values = series.to_numpy(dtype=float)
    if len(values) <= burn_in + 10:
        raise ValueError("series shorter than the burn-in period")
    head = values[:burn_in]
    mu0 = float(head.mean())
    beta0 = float(alpha0 * head.var(ddof=1))
    hazard = 1.0 / float(hazard_lambda)
    log_h, log_1mh = np.log(hazard), np.log1p(-hazard)

    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    log_r = np.array([0.0])                       # P(r_{t-1} = 0) = 1 before the first observation

    n = len(values) - burn_in
    short = np.zeros((n, len(short_runs)))
    expected = np.zeros(n)
    map_run = np.zeros(n, dtype=int)
    edges = np.array(list(bins) + [np.inf])
    binned = np.zeros((n, len(bins)))

    for step, value in enumerate(values[burn_in:]):
        log_pi = _student_t_logpdf(value, mu, kappa, alpha, beta)
        log_joint = log_r + log_pi
        log_growth = log_joint + log_1mh
        log_change = logsumexp(log_joint) + log_h
        log_new = np.concatenate(([log_change], log_growth))
        log_new -= logsumexp(log_new)
        # update the sufficient statistics: every run absorbs x, a new run restarts at the prior
        new_mu = (kappa * mu + value) / (kappa + 1.0)
        new_beta = beta + kappa * (value - mu) ** 2 / (2.0 * (kappa + 1.0))
        mu = np.concatenate(([mu0], new_mu))
        kappa = np.concatenate(([kappa0], kappa + 1.0))
        alpha = np.concatenate(([alpha0], alpha + 0.5))
        beta = np.concatenate(([beta0], new_beta))
        log_r = log_new

        probs = np.exp(log_r)
        lengths = np.arange(len(probs))
        for column, s in enumerate(short_runs):
            short[step, column] = probs[: s + 1].sum()
        expected[step] = float(probs @ lengths)
        map_run[step] = int(np.argmax(probs))
        binned[step] = np.add.reduceat(probs, np.searchsorted(lengths, bins)) if len(probs) > bins[-1] \
            else [probs[(lengths >= lo) & (lengths < hi)].sum() for lo, hi in zip(bins, edges[1:])]

    dates = series.index[burn_in:]
    short_frame = pd.DataFrame(short, index=dates, columns=[f"le_{s}" for s in short_runs])
    labels = [f"{lo}" if hi - lo == 1 else (f"{lo}-{int(hi) - 1}" if np.isfinite(hi) else f"{lo}+")
              for lo, hi in zip(bins, edges[1:])]
    return BOCPDResult(
        index=dates, short_mass=short_frame,
        expected_run_length=pd.Series(expected, index=dates, name="expected_run_length"),
        map_run_length=pd.Series(map_run, index=dates, name="map_run_length"),
        run_length_bins=pd.DataFrame(binned, index=dates, columns=labels),
        prior={"mu0": mu0, "beta0": beta0, "kappa0": kappa0, "alpha0": alpha0,
               "hazard": hazard, "burn_in": burn_in},
        start=dates[0],
    )


def alarm_episodes(alarm: pd.Series, merge_gap: int = 5) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group consecutive alarm days into episodes (first day, last day).

    Alarm days separated by fewer than ``merge_gap`` quiet trading days are one
    episode, so a single shock that flickers around the threshold is counted
    once and false alarms are not inflated by flicker.
    """
    flags = alarm.fillna(False).to_numpy(dtype=bool)
    index = alarm.index
    episodes: list[list[int]] = []
    for i in np.flatnonzero(flags):
        if episodes and i - episodes[-1][1] <= merge_gap:
            episodes[-1][1] = i
        else:
            episodes.append([i, i])
    return [(index[a], index[b]) for a, b in episodes]


def evaluate_detector(alarm: pd.Series, events: dict[str, str], search_days: int = 60,
                      merge_gap: int = 5, years: float | None = None) -> dict:
    """Detection delay and false-alarm rate against dated events.

    An event is DETECTED when an alarm episode starts on or after its date and
    within ``search_days`` trading days. The delay is the number of trading
    days between the two. An episode that starts outside every event window is
    a FALSE alarm (an episode that starts before an event's date is false even
    if it runs into the event: it was not caused by it).
    """
    index = alarm.index
    episodes = alarm_episodes(alarm, merge_gap)
    rows = []
    windows = []
    for name, date in events.items():
        when = pd.Timestamp(date)
        position = int(index.searchsorted(when))
        if position >= len(index):
            rows.append({"event": name, "date": when, "detected": False, "alarm_date": pd.NaT,
                         "delay_days": np.nan})
            continue
        window_end = min(position + int(search_days), len(index) - 1)
        windows.append((position, window_end))
        hit = next((s for s, _ in episodes
                    if position <= int(index.get_loc(s)) <= window_end), None)
        rows.append({
            "event": name, "date": when, "detected": hit is not None,
            "alarm_date": hit if hit is not None else pd.NaT,
            "delay_days": float(index.get_loc(hit) - position) if hit is not None else np.nan,
        })
    false = [s for s, _ in episodes
             if not any(a <= int(index.get_loc(s)) <= b for a, b in windows)]
    span_years = years if years is not None else max((index[-1] - index[0]).days / 365.25, 1e-9)
    table = pd.DataFrame(rows)
    return {
        "events": table,
        "n_detected": int(table["detected"].sum()),
        "n_events": int(len(table)),
        "mean_delay_days": float(table["delay_days"].mean()) if table["detected"].any() else float("nan"),
        "n_episodes": len(episodes),
        "n_false_alarms": len(false),
        "false_alarms_per_year": len(false) / span_years,
        "false_alarm_dates": false,
    }


def naive_shock_detector(returns: pd.Series, short_window: int = 5, long_window: int = 250,
                         ratio: float = 2.0) -> pd.Series:
    """The simple rule BOCPD has to beat: short-window volatility versus its own norm.

    Alarm when the realised volatility of the last ``short_window`` days is at
    least ``ratio`` times the median of that quantity over the trailing
    ``long_window`` days. Uses data up to t only.
    """
    short = np.sqrt((returns ** 2).rolling(short_window).mean())
    norm = short.rolling(long_window, min_periods=long_window // 2).median()
    return (short >= ratio * norm).fillna(False).rename("naive_alarm")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def persistence(labels: pd.Series) -> dict:
    """How long do labels last? Mean run length and switches per year."""
    clean = labels.dropna().astype(int)
    if len(clean) < 2:
        return {"mean_run_days": float("nan"), "switches_per_year": float("nan")}
    change = (clean != clean.shift(1)).to_numpy(copy=True)
    change[0] = True
    run_ids = np.cumsum(change)
    run_lengths = pd.Series(run_ids).value_counts()
    years = (clean.index[-1] - clean.index[0]).days / 365.25
    return {"mean_run_days": float(run_lengths.mean()),
            "switches_per_year": float((run_ids.max() - 1) / max(years, 1e-9))}
