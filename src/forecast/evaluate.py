"""Finding out how good each way of forecasting really is, by doing it the way it would have been done.

Walk forward over a stretch of recent history. At each origin (every ``step`` trading days) every method is given the prices up to that day and nothing later, forecasts the distribution of the return over the next
1, 5, 21 days (and the horizon you chose), and is scored against what then happened. A method is refitted only every ``refit_days`` (the fit is the slow part); in between it updates its state with each new day
and keeps the fitted numbers, which is what a live system would do. The scores (``forecast.scoring``) say how sharp and how right the distributions were, how often the outcome fell inside the ranges they gave, and
whether the means and directions they implied beat the random walk, which says nothing about direction.

A forecast is not a strategy: a method can describe the spread of returns well and make no money, and the other way round. This only asks how good the *forecast* is.
"""

from __future__ import annotations

import math
import time
import warnings
import zlib
from typing import Callable

import numpy as np

from . import scoring as S
from .base import Forecast, ForecastError, History
from .registry import BASELINE, BLEND, BLEND_WEIGHTED, BLENDS, METHODS, VIEWS

LEVELS = S.LEVELS
COVERAGES = S.COVERAGES
MAX_HORIZON = 63
MIN_ORIGINS = 30
TRADING_DAYS = 252
HORIZONS = (1, 5, 21)
EFFORTS = {
    "quick": {"label": "Quick", "years": 4, "step": 21, "refit_days": 126, "n_paths": 500, "text": "Four years of forecasts, one every month, 500 simulated futures each: under a minute."},
    "standard": {"label": "Standard", "years": 8, "step": 10, "refit_days": 63, "n_paths": 1000, "text": "Eight years, a forecast every two weeks, 1,000 simulated futures each: a minute or two."},
    "thorough": {"label": "Thorough", "years": 12, "step": 5, "refit_days": 63, "n_paths": 2000, "text": "Twelve years, a forecast every week, 2,000 simulated futures each: several minutes."},
}
CONTROLS = {"gp_returns": "hull_white", "analog": "hull_white", "fhs": "garch", "hmm": "garch"}     # method -> the simpler method it must beat to have added anything
BENCH_DAYS = 1260                                  # the historical average each method's mean is compared with: five years of returns
FAN_PATHS = 4000
FAN_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99)
HIST_BINS = 41


def scored_horizons(horizon: int) -> tuple[int, ...]:
    if not 1 <= int(horizon) <= MAX_HORIZON:
        raise ForecastError(f"the horizon must be between 1 and {MAX_HORIZON} trading days")
    return tuple(sorted(set(HORIZONS) | {int(horizon)}))


def _rng(seed: int, name: str, i: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(name.encode()), int(i)])


def _f(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


class _Tape:
    """What one method scored at every origin and horizon (NaN where it could not forecast, or the outcome is not known yet)."""

    def __init__(self, n_origins: int, n_h: int):
        shape = (n_origins, n_h)
        self.crps, self.hit50, self.hit90, self.below5, self.pit, self.mean, self.p_up = (np.full(shape, np.nan) for _ in range(7))
        self.pinball = np.full(shape + (len(LEVELS),), np.nan)
        self.failures, self.fits, self.seconds = 0, 0, 0.0
        self.last_info: dict = {}
        self.last_error = ""


def combine(members: list[np.ndarray], rng: np.random.Generator, n: int) -> np.ndarray:
    """An equal-weight pool: the same number of draws from every member (a linear opinion pool)."""
    take = max(2, n // len(members))
    return np.concatenate([m[rng.choice(len(m), take, replace=len(m) < take)] for m in members])


def combine_weighted(members: dict[str, np.ndarray], weights: dict[str, float], rng: np.random.Generator, n: int) -> np.ndarray:
    """A linear opinion pool with unequal weights: from each member a number of draws proportional to its weight (at least one)."""
    parts = []
    for name, draws in members.items():
        take = max(1, int(round(n * weights[name])))
        parts.append(draws[rng.choice(len(draws), take, replace=len(draws) < take)])
    return np.concatenate(parts)


def inverse_loss_weights(losses: dict[str, np.ndarray], min_obs: int = 10) -> dict[str, float] | None:
    """Bates and Granger's (1969) weights from the average losses of each member: proportional to the inverse of the mean loss. ``None`` (so: equal weights) when a member has too few scored forecasts or a
    mean loss of zero."""
    means = {}
    for name, v in losses.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < min_obs or v.mean() <= 0:
            return None
        means[name] = float(v.mean())
    inverse = {n: 1.0 / m for n, m in means.items()}
    total = sum(inverse.values())
    return {n: w / total for n, w in inverse.items()}


def settings_for(effort: str, history: History, names: list[str], horizon: int, *, years: float | None = None, step: int | None = None, refit_days: int | None = None, n_paths: int | None = None) -> dict:
    if effort not in EFFORTS:
        raise ForecastError(f"effort must be one of {', '.join(EFFORTS)}")
    e = EFFORTS[effort]
    hs = scored_horizons(horizon)
    need = max([METHODS[n].min_history for n in names if n in METHODS] + [260]) + 5
    last = history.n - 2
    usable = last - need
    span = min(int((years or e["years"]) * 252), usable)
    st = int(step or e["step"])
    if span < MIN_ORIGINS:
        raise ForecastError(f"the history is too short to evaluate these methods: they need {need} days of prices to learn from and at least {MIN_ORIGINS} more days to be tested on (there are {history.n}). Choose a longer history, or leave out the methods that need most.")
    st = max(1, min(st, span // (MIN_ORIGINS + 10)))
    return {"effort": effort, "horizons": hs, "years": span / 252.0, "span_days": span, "step": st, "refit_days": int(refit_days or e["refit_days"]), "n_paths": int(n_paths or e["n_paths"]), "need": need}


def _origins(history: History, cfg: dict) -> list[int]:
    last = history.n - 2
    lo = last - cfg["span_days"]
    return list(range(last, lo - 1, -cfg["step"]))[::-1]


def evaluate(history: History, names: list[str], horizon: int = 21, effort: str = "standard", seed: int = 0, progress: Callable[[str], None] | None = None,
             stop: Callable[[], bool] | None = None, **overrides) -> dict:
    """Walk-forward evaluation of ``names`` (method ids, and ``blend``) on ``history``. ``progress(text)`` is told where it is; ``stop()`` ends it after the origin it is on (what is done is scored and returned)."""
    names = [n for n in dict.fromkeys(names) if n != BASELINE]
    for n in names:
        if n not in METHODS and n not in BLENDS:
            raise ForecastError(f"unknown forecasting method '{n}'")
    run = [BASELINE] + names
    members = [n for n in run if n in METHODS]
    cfg = settings_for(effort, history, members, horizon, **overrides)
    hs, origins, n_paths = cfg["horizons"], _origins(history, cfg), cfg["n_paths"]
    hmax = max(hs)
    n_h = len(hs)
    j_head = hs.index(int(horizon))
    close = history.close
    realised = np.full((len(origins), n_h), np.nan)
    for oi, i in enumerate(origins):
        for j, hz in enumerate(hs):
            if i + hz < history.n:
                realised[oi, j] = math.log(close[i + hz] / close[i])
    bench = np.full((len(origins), n_h), np.nan)                          # the benchmark forecast of the average return: the mean of the last five years' returns, carried over the horizon
    logp = np.log(close)
    for oi, i in enumerate(origins):
        r = np.diff(logp[max(0, i - BENCH_DAYS):i + 1])
        bench[oi] = r.mean() * np.array(hs) if len(r) >= 250 else np.nan
    tapes = {n: _Tape(len(origins), n_h) for n in run}
    states: dict[str, object] = {}
    fitted_at: dict[str, int] = {}
    done = 0
    t_start = time.perf_counter()

    def score(name: str, oi: int, fc: Forecast) -> None:
        t = tapes[name]
        for j, hz in enumerate(hs):
            y = realised[oi, j]
            if not math.isfinite(y):
                continue
            x = fc.at(hz)
            t.crps[oi, j] = S.crps(x, y)[0]
            t.pinball[oi, j] = S.pinball(x, y)[0]
            t.hit50[oi, j] = float(S.interval_hits(x, y, 0.5)[0])
            t.hit90[oi, j] = float(S.interval_hits(x, y, 0.9)[0])
            t.below5[oi, j] = float(S.below_quantile(x, y, 0.05)[0])
            t.pit[oi, j] = S.pit(x, y)[0]
        for j, hz in enumerate(hs):                                          # the predictive mean and the chance of a rise do not need the outcome
            x = fc.at(hz)
            t.mean[oi, j], t.p_up[oi, j] = float(x.mean()), float((x > 0).mean())

    for oi, i in enumerate(origins):
        if stop is not None and stop():
            break
        past = history.upto(i)
        got: dict[str, np.ndarray] = {}
        for name in run:
            if name in BLENDS:
                continue
            m, t = METHODS[name], tapes[name]
            t0 = time.perf_counter()
            try:
                with np.errstate(all="ignore"), warnings.catch_warnings():              # a degenerate stretch makes a method refuse; the refusal is counted, the numpy warning behind it is noise
                    warnings.simplefilter("ignore")
                    refit = name not in states or i - fitted_at[name] >= cfg["refit_days"] * (2 if m.heavy else 1)
                    if refit:
                        states[name] = m.fit(past.tail(m.window), hs)
                        fitted_at[name] = i
                        t.fits += 1
                    fc = m.predict(states[name], past.tail(m.lookback), hs, n_paths, _rng(seed, name, i))
                score(name, oi, fc)
                got[name] = fc.samples[:, :hmax]
                t.last_info = fc.info
            except (ForecastError, np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
                t.failures += 1
                t.last_error = str(error)
                states.pop(name, None)                                       # try a fresh fit at the next origin
            t.seconds += time.perf_counter() - t0
        pool = {k: v for k, v in got.items() if k != BASELINE}
        for blend in (b for b in BLENDS if b in run):
            if len(pool) < 2:
                tapes[blend].failures += 1
                continue
            t0 = time.perf_counter()
            if blend == BLEND:
                draws = combine(list(pool.values()), _rng(seed, blend, i), n_paths)
            else:                                                            # weights from the forecasts whose outcome was already known at this origin
                known = [k for k in range(oi) if origins[k] + int(horizon) <= i]
                w = inverse_loss_weights({k: tapes[k].crps[known, j_head] for k in pool}) if known else None
                draws = combine_weighted(pool, w or {k: 1.0 / len(pool) for k in pool}, _rng(seed, blend, i), n_paths)
            score(blend, oi, Forecast(draws))
            tapes[blend].seconds += time.perf_counter() - t0
        done = oi + 1
        if progress is not None:
            progress(f"forecast {done} of {len(origins)} ({history.index[i].date() if history.index is not None else i}) · {time.perf_counter() - t_start:.0f} s")
    if done < MIN_ORIGINS:
        raise ForecastError(f"stopped after {done} forecasts: at least {MIN_ORIGINS} are needed to say anything")
    return _summarize(history, run, tapes, realised[:done], bench[:done], origins[:done], cfg, hs, horizon, done)


def _summarize(history: History, run: list[str], tapes: dict[str, _Tape], realised: np.ndarray, bench: np.ndarray, origins: list[int], cfg: dict, hs: tuple[int, ...], horizon: int, done: int) -> dict:
    base = tapes[BASELINE]
    step = cfg["step"]
    rows = {n: {"scores": {}} for n in run}
    for j, hz in enumerate(hs):
        stride = S.stride_for(hz, step)
        lag = stride - 1
        entries = {}
        for name in run:
            t = tapes[name]
            ok = np.isfinite(t.crps[:done, j]) & np.isfinite(base.crps[:done, j])
            n_ok = int(ok.sum())
            e = {"n": n_ok, "n_independent": int(math.ceil(n_ok / stride)) if n_ok else 0}
            if n_ok >= 10:
                e["crps"] = _f(t.crps[:done, j][ok].mean())
                e["crps_baseline"] = _f(base.crps[:done, j][ok].mean())
                e["skill"] = _f(S.skill(t.crps[:done, j][ok], base.crps[:done, j][ok])) if name != BASELINE else 0.0
                cmp = S.better_than(t.crps[:done, j][ok], base.crps[:done, j][ok], lag) if name != BASELINE else {"p": float("nan")}
                e["p"] = _f(cmp["p"])
                e["coverage"] = {"50": _f(np.nanmean(t.hit50[:done, j][ok])), "90": _f(np.nanmean(t.hit90[:done, j][ok]))}
                e["coverage_p"] = {"50": _f(S.binomial_p(t.hit50[:done, j][ok], 0.5, stride)), "90": _f(S.binomial_p(t.hit90[:done, j][ok], 0.9, stride))}
                e["var5"] = _f(np.nanmean(t.below5[:done, j][ok]))
                e["var5_p"] = _f(S.binomial_p(t.below5[:done, j][ok], 0.05, stride))
                e["pit"] = [_f(v) for v in S.pit_histogram(t.pit[:done, j][ok])]
                e["pit_p"] = _f(S.pit_p(t.pit[:done, j][ok], stride))
                e["pinball"] = {str(q): _f(np.nanmean(t.pinball[:done, j, k][ok])) for k, q in enumerate(LEVELS)}
                if name in METHODS and METHODS[name].mean_signal or name in BLENDS:
                    y = realised[:, j][ok]
                    ms = S.mean_skill(y, t.mean[:done, j][ok], bench[:, j][ok], lag)
                    ds = S.direction_skill(y, t.p_up[:done, j][ok], lag)
                    e["mean"] = {"r2": _f(ms["r2"]), "p": _f(ms["p"]), "average": _f(np.nanmean(t.mean[:done, j][ok]))}
                    e["direction"] = {k: _f(v) for k, v in ds.items()}
            entries[name] = e
        ps = [entries[n].get("p", float("nan")) if n != BASELINE else float("nan") for n in run]
        adj = S.holm_adjust(ps)
        for n, a in zip(run, adj):
            if n != BASELINE and entries[n].get("p") is not None:
                entries[n]["p_adjusted"] = _f(a)
                entries[n]["beats_baseline"] = bool(entries[n].get("skill") is not None and entries[n]["skill"] > 0 and a < 0.05)
            rows[n]["scores"][str(hz)] = entries[n]
    controls = []
    for name, control in CONTROLS.items():
        if name in tapes and control in tapes:
            for j, hz in enumerate(hs):
                a, b = tapes[name].crps[:done, j], tapes[control].crps[:done, j]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 10:
                    cmp = S.better_than(a[ok], b[ok], S.stride_for(hz, step) - 1)
                    controls.append({"method": name, "control": control, "horizon": hz, "skill": _f(S.skill(a[ok], b[ok])), "p": _f(cmp["p"]), "n": int(ok.sum())})
    out_methods = []
    for name in run:
        t = tapes[name]
        view = METHODS[name].view() if name in METHODS else VIEWS[name]
        out_methods.append({**view, "scores": rows[name]["scores"], "failures": t.failures, "last_error": t.last_error, "fits": t.fits, "seconds": round(t.seconds, 1), "info": _plain(t.last_info)})
    dates = history.index
    j = hs.index(int(horizon))
    members = [n for n in run if n in METHODS and n != BASELINE]
    weights = inverse_loss_weights({n: tapes[n].crps[:done, j] for n in members}) if len(members) >= 2 else None
    gain, gain_scale = _cumulative_gain(run, tapes, done, j)
    return {"horizons": list(hs), "headline": int(horizon), "baseline": BASELINE, "controls": controls, "methods": out_methods, "blend_weights": weights,
            "pairwise": _pairwise(run, tapes, done, j, S.stride_for(int(horizon), step) - 1, int(horizon)), "gain": gain, "gain_scale": gain_scale,
            "origin_dates": [str(dates[i].date()) if dates is not None else i for i in origins],
            "origins": {"count": done, "first": str(dates[origins[0]].date()) if dates is not None else origins[0], "last": str(dates[origins[-1]].date()) if dates is not None else origins[-1], "step_days": step,
                        "refit_days": cfg["refit_days"], "years": round(cfg["span_days"] / 252.0, 1), "paths": cfg["n_paths"], "history_days": history.n, "effort": cfg["effort"]}}


def _pairwise(run: list[str], tapes: dict[str, _Tape], done: int, j: int, lag: int, horizon: int) -> dict:
    """Every method against every other at the headline horizon: ``skill[a][b]`` is how far below ``b``'s average CRPS ``a``'s was (positive: ``a`` was better), ``p[a][b]`` the one-sided Diebold-Mariano
    p-value that ``a`` is better (Newey-West lag for the overlap of the forecasts). Unadjusted for the number of pairs: the page says so."""
    skill = [[None] * len(run) for _ in run]
    p = [[None] * len(run) for _ in run]
    for a, na in enumerate(run):
        for b, nb in enumerate(run):
            if a == b:
                continue
            x, y = tapes[na].crps[:done, j], tapes[nb].crps[:done, j]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 10:
                skill[a][b] = _f(S.skill(x[ok], y[ok]))
                p[a][b] = _f(S.better_than(x[ok], y[ok], lag)["p"])
    return {"ids": list(run), "horizon": horizon, "skill": skill, "p": p}


def _cumulative_gain(run: list[str], tapes: dict[str, _Tape], done: int, j: int) -> tuple[dict, float | None]:
    """For each method the running sum, over the forecast origins, of how far its CRPS was below the random walk's, as a share of the random walk's average CRPS (the cumulative loss differential of
    Goyal and Welch 2008): a line that climbs is a method that kept beating the random walk, one that falls has been worse, and a line that rises and then goes flat stopped helping. An origin whose outcome
    is not yet known adds nothing. Returns the lines and the random walk's average CRPS they are measured in."""
    base = tapes[BASELINE].crps[:done, j]
    scale = float(np.nanmean(base)) if np.isfinite(base).any() else float("nan")
    out = {}
    for name in run:
        if name == BASELINE or not math.isfinite(scale) or scale <= 0:
            continue
        diff = base - tapes[name].crps[:done, j]
        out[name] = [_f(v) for v in np.cumsum(np.where(np.isfinite(diff), diff, 0.0)) / scale]
    return out, (scale if math.isfinite(scale) and scale > 0 else None)


def _plain(obj):
    """Fitted numbers as plain Python (the report is JSON)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return _f(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def _terminal_histogram(samples: dict[str, np.ndarray], horizon: int) -> dict:
    """The distribution of the ``horizon``-day return of every method on common bins (the central 99.8% of all the draws pooled), as counts, so that the methods can be drawn on one axis."""
    ends = {name: x[:, horizon - 1] for name, x in samples.items()}
    pooled = np.concatenate(list(ends.values()))
    lo, hi = np.quantile(pooled, [0.001, 0.999])
    edges = np.linspace(lo, hi, HIST_BINS + 1)
    return {"horizon": int(horizon), "edges": [_f(e) for e in edges], "counts": {name: [int(c) for c in np.histogram(v, bins=edges)[0]] for name, v in ends.items()},
            "draws": {name: int(len(v)) for name, v in ends.items()}}


def forecast_now(history: History, names: list[str], horizon: int, n_paths: int = FAN_PATHS, seed: int = 0, progress: Callable[[str], None] | None = None, blend_weights: dict | None = None) -> dict:
    """Fit every method on the latest data and forecast from the last day: the fan of the price for each step up to ``horizon``, and a summary at each scored horizon."""
    names = [n for n in dict.fromkeys(names)]
    run = names if BASELINE in names else [BASELINE] + names
    hs = scored_horizons(horizon)
    last = float(history.close[-1])
    steps = max(hs)
    samples: dict[str, np.ndarray] = {}
    infos: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for k, name in enumerate(n for n in run if n not in BLENDS):
        m = METHODS[name]
        if progress is not None:
            progress(f"forecasting from {history.index[-1].date() if history.index is not None else 'the last day'}: {m.label} ({k + 1} of {len([n for n in run if n not in BLENDS])})")
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state = m.fit(history.tail(m.window), hs)
                fc = m.predict(state, history.tail(m.lookback), hs, n_paths, _rng(seed, name, history.n))
            samples[name], infos[name] = fc.samples[:, :steps], _plain(fc.info)
        except (ForecastError, np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
            errors[name] = str(error)
    pool = {k: v for k, v in samples.items() if k != BASELINE}
    for blend in (b for b in BLENDS if b in run):
        if len(pool) < 2:
            errors[blend] = "needs at least two other methods that could forecast"
            continue
        w = {k: blend_weights[k] for k in pool if blend_weights and k in blend_weights} if blend == BLEND_WEIGHTED else {}
        weighted = len(w) == len(pool) and abs(sum(w.values()) - 1.0) < 1e-6
        if blend == BLEND or not weighted:
            samples[blend] = combine(list(pool.values()), _rng(seed, blend, history.n), n_paths)
            infos[blend] = {"members": list(pool)} if blend == BLEND else {"members": list(pool), "weights": "equal: the evaluation had too few scored forecasts to weight by"}
        else:
            samples[blend] = combine_weighted(pool, w, _rng(seed, blend, history.n), n_paths)
            infos[blend] = {"members": list(pool), "weights": {k: round(v, 4) for k, v in w.items()}}
    out = {}
    for name, x in samples.items():
        q = np.quantile(x[:, :horizon], FAN_LEVELS, axis=0)
        view = METHODS[name].view() if name in METHODS else VIEWS[name]
        summary = {}
        for hz in hs:
            r = x[:, hz - 1]
            lo, md, hi = np.quantile(r, [0.05, 0.5, 0.95])
            tail = r[r <= lo]
            summary[str(hz)] = {"mean": _f(r.mean()), "median": _f(md), "q05": _f(lo), "q95": _f(hi), "p_up": _f((r > 0).mean()), "p_loss_5": _f((r < math.log(0.95)).mean()), "p_gain_5": _f((r > math.log(1.05)).mean()),
                                "sd": _f(r.std()), "vol_annual": _f(r.std() * math.sqrt(TRADING_DAYS / hz)), "es_05": _f(tail.mean()) if len(tail) else None}
        out[name] = {"id": name, "label": view["label"], "fan": {str(lv): [_f(last * math.exp(v)) for v in q[k]] for k, lv in enumerate(FAN_LEVELS)}, "summary": summary, "info": infos.get(name, {})}
    return {"last_price": last, "horizons": list(hs), "horizon": int(horizon), "methods": out, "errors": errors, "hist": _terminal_histogram(samples, horizon)}
