"""From a forecast to an expected return a portfolio may use, and nothing else.

A forecast is not an expected return until it says what it forecasts, in what units, over what horizon, when it could first be used, and how sure it is. :class:`ForecastRecord` is that statement. The conversions in
this module are the only way the platform turns one into an expected return, and each one is explicit:

* a **volatility** forecast is refused (``ForecastConversionError``): a spread is not a drift;
* a **price** forecast becomes ``(predicted price x split factor + expected dividends) / last price - 1``; on an adjusted-price basis dividends are already inside the prices and a non-zero dividend is refused rather than
  counted twice; a raw-price forecast needs its split factor; a forecast in another currency needs the forecast of the exchange rate over the same horizon (``(1 + local)(1 + fx) - 1``);
* a **return** forecast in log units is converted with ``exp(mu + sigma^2 / 2) - 1`` when its spread is known (the mean of a log-normal) and ``exp(mu) - 1`` otherwise, which is the median and is labelled so;
* a **distribution** (draws of the log return) becomes ``mean(exp(draw)) - 1``, and its spread is the standard deviation of the simple return.

**Horizons.** A forecast is for one horizon. It feeds a portfolio only at that holding period; a different horizon is refused unless scaling is asked for, in which case the expected *log* return is scaled
linearly in time (a constant-drift assumption, stated in the result) and the uncertainty by the square root of the ratio, and a ratio beyond three is refused. Annualising by multiplying a short-horizon number by a
constant is never done on a return: :func:`annualise` compounds. The only multiplication by ``252 / h`` is :func:`to_optimiser_units`, which puts a horizon-``h`` expected return in the same units as the
annualised covariance the optimisers use (that covariance is the daily one times 252, so the ratio of the two is unchanged); it is not an annual return forecast and is not reported as one.

**Availability.** A record can be used only on a date strictly after its ``available_at``: a forecast made on the close of day ``t`` informs decisions from day ``t + 1`` (:func:`usable`), and a record older than
``max_age_days`` is stale and not used.
"""

from __future__ import annotations

import hashlib
import math
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .base import ForecastError, History
from .registry import METHODS

KINDS = ("price", "return", "volatility", "distribution")
UNITS = ("price", "log_return", "simple_return", "annualised_volatility")
MAX_SCALING = 3.0
ANN = 252.0
MEAN_METHODS = ("historical", "gbm", "hull_white", "kalman", "theta", "arma", "markov_returns", "gp_drift")      # cheap methods whose forecast carries a mean (``mean_signal``)


class ForecastConversionError(ValueError):
    """A forecast cannot become an expected return (and why)."""


@dataclass(frozen=True)
class ForecastRecord:
    instrument: str
    origin: pd.Timestamp                    # the close the forecast was made on
    available_at: pd.Timestamp             # the first moment the forecast could be acted on is strictly after this
    horizon_days: int
    kind: str                               # price | return | volatility | distribution
    units: str                              # price | log_return | simple_return | annualised_volatility
    value: float | None = None              # the point forecast, in ``units``
    samples: tuple | None = None            # draws of the horizon log return (a distribution forecast)
    uncertainty: float | None = None        # the standard deviation of the horizon log return, when known
    last_price: float | None = None         # the price at the origin, for a price forecast
    price_basis: str | None = None          # adjusted | raw
    dividends: float = 0.0                  # expected cash per share over the horizon (raw basis only)
    split_factor: float = 1.0               # new shares per old share between the origin and the horizon (raw basis)
    currency: str | None = None
    annualisation: str | None = None        # how a volatility is annualised, e.g. "sqrt(252)"
    model_id: str = ""
    model_version: str = ""
    training_fingerprint: str = ""
    evaluation: dict = field(default_factory=lambda: {"status": "unevaluated"})
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ForecastConversionError(f"kind must be one of {', '.join(KINDS)}")
        if self.units not in UNITS:
            raise ForecastConversionError(f"units must be one of {', '.join(UNITS)}")
        if int(self.horizon_days) < 1:
            raise ForecastConversionError("the horizon is at least one day")
        if pd.Timestamp(self.available_at) < pd.Timestamp(self.origin):
            raise ForecastConversionError("a forecast cannot be available before the day it was made from")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["origin"], d["available_at"] = str(pd.Timestamp(self.origin).date()), str(pd.Timestamp(self.available_at).date())
        d["samples"] = None if self.samples is None else len(self.samples)
        return d


@dataclass(frozen=True)
class ExpectedReturn:
    value: float                            # the expected simple return over ``horizon_days``
    horizon_days: int
    statistic: str                          # what the number is: mean | median | point
    uncertainty: float | None               # the standard deviation of the simple return, when known
    assumptions: tuple = ()


# ------------------------------------------------------------------------------------------------------------------------ conversions
def expected_simple_return(rec: ForecastRecord, base_currency: str | None = None, fx_forecast: float | None = None) -> ExpectedReturn:
    if rec.kind == "volatility" or rec.units == "annualised_volatility":
        raise ForecastConversionError("a volatility forecast is not an expected return: it says how far the price may move, not which way")
    notes: list[str] = []
    if rec.kind == "price":
        if rec.value is None or rec.last_price is None or rec.last_price <= 0 or rec.value <= 0:
            raise ForecastConversionError("a price forecast needs a positive predicted price and the price at the origin")
        if rec.price_basis not in ("adjusted", "raw"):
            raise ForecastConversionError("say whether the prices are adjusted or raw: dividends and splits are handled differently")
        if rec.price_basis == "adjusted":
            if rec.dividends:
                raise ForecastConversionError("adjusted prices already include dividends: a dividend must not be added again")
            if rec.split_factor != 1.0:
                raise ForecastConversionError("adjusted prices are already adjusted for splits: the split factor must be 1")
            local = rec.value / rec.last_price - 1.0
            notes.append("adjusted prices: dividends and splits are inside both prices")
        else:
            local = (rec.value * rec.split_factor + rec.dividends) / rec.last_price - 1.0
            notes.append(f"raw prices: predicted price x split factor {rec.split_factor:g} plus expected dividends {rec.dividends:g}")
        stat = "point"
        sd = None if rec.uncertainty is None else abs(rec.uncertainty) * (1.0 + local)
    elif rec.kind == "return":
        if rec.value is None:
            raise ForecastConversionError("a return forecast needs a value")
        if rec.units == "simple_return":
            local, stat, sd = float(rec.value), "point", rec.uncertainty
        elif rec.units == "log_return":
            s = rec.uncertainty
            if s is not None and s > 0:
                local, stat = math.exp(rec.value + 0.5 * s * s) - 1.0, "mean"
                notes.append("log-normal mean: exp(mu + sigma^2/2) - 1")
            else:
                local, stat = math.exp(rec.value) - 1.0, "median"
                notes.append("no spread given: exp(mu) - 1 is the median of the simple return, not its mean")
            sd = None if s is None else abs((1.0 + local) * s)
        else:
            raise ForecastConversionError("a return forecast is in log or simple return units")
    else:                                                                                  # distribution
        if rec.samples is None or len(rec.samples) < 30:
            raise ForecastConversionError("a distribution forecast needs at least 30 draws")
        x = np.asarray(rec.samples, dtype=float)
        simple = np.expm1(x)
        local, stat, sd = float(simple.mean()), "mean", float(simple.std(ddof=1))
        notes.append("mean of exp(draw) - 1 over the draws")
    if rec.currency and base_currency and rec.currency != base_currency:
        if fx_forecast is None:
            raise ForecastConversionError(f"the forecast is in {rec.currency} and the portfolio in {base_currency}: give the forecast change of the exchange rate over the same horizon")
        local = (1.0 + local) * (1.0 + fx_forecast) - 1.0
        notes.append(f"converted from {rec.currency} to {base_currency} with a forecast exchange-rate change of {fx_forecast:+.4%}")
    return ExpectedReturn(float(local), int(rec.horizon_days), stat, None if sd is None else float(sd), tuple(notes))


def align_horizon(er: ExpectedReturn, target_days: int, allow_scaling: bool = False) -> ExpectedReturn:
    """The expected return over ``target_days``. An unequal horizon is refused unless scaling is allowed, and then the log return scales linearly (a constant drift) and the spread by the square root."""
    if er.horizon_days == int(target_days):
        return er
    if not allow_scaling:
        raise ForecastConversionError(f"the forecast is for {er.horizon_days} days and the portfolio holds for {int(target_days)}: a forecast is used only at its own horizon (or allow scaling, a constant-drift assumption)")
    ratio = target_days / er.horizon_days
    if ratio > MAX_SCALING or ratio < 1.0 / MAX_SCALING:
        raise ForecastConversionError(f"scaling a {er.horizon_days}-day forecast to {int(target_days)} days is more than {MAX_SCALING:g} times: refused")
    value = math.expm1(math.log1p(er.value) * ratio)
    sd = None if er.uncertainty is None else er.uncertainty * math.sqrt(ratio)
    return ExpectedReturn(value, int(target_days), er.statistic, sd, er.assumptions + (f"scaled from {er.horizon_days} to {int(target_days)} days assuming a constant drift of the log return",))


def annualise(simple_return: float, horizon_days: int) -> float:
    """The compounded equivalent over a year: ``(1 + r)^(252 / h) - 1``. Not ``r x 252 / h``."""
    return (1.0 + simple_return) ** (ANN / horizon_days) - 1.0


def to_optimiser_units(simple_return: float, horizon_days: int) -> float:
    """A horizon-``h`` expected return in the units of the annualised covariance the optimisers use (daily covariance x 252): ``r x 252 / h``. The same scaling is applied to every source of expected returns, so the
    comparison between them is unaffected; the number is not an annual return forecast."""
    return float(simple_return) * ANN / horizon_days


# ------------------------------------------------------------------------------------------------------------------------ availability
def usable(rec: ForecastRecord, on: pd.Timestamp, max_age_days: int | None = None) -> bool:
    """A record may be used on ``on`` only if it was available strictly before that day and is not stale."""
    if not pd.Timestamp(rec.available_at) < pd.Timestamp(on):
        return False
    return max_age_days is None or (pd.Timestamp(on) - pd.Timestamp(rec.origin)).days <= max_age_days


def latest_usable(records: list[ForecastRecord], instrument: str, on: pd.Timestamp, horizon_days: int | None = None, max_age_days: int | None = None) -> ForecastRecord | None:
    pool = [r for r in records if r.instrument == instrument and usable(r, on, max_age_days) and (horizon_days is None or r.horizon_days == horizon_days)]
    return max(pool, key=lambda r: pd.Timestamp(r.origin), default=None)


# ------------------------------------------------------------------------------------------------------------------------ generation
def _fingerprint(asset: str, close: np.ndarray, method: str, n_paths: int, seed: int) -> str:
    h = hashlib.sha1()
    h.update(f"{asset}|{method}|{n_paths}|{seed}|{len(close)}".encode())
    h.update(np.ascontiguousarray(close[-min(len(close), 504):]).tobytes())
    return h.hexdigest()[:12]


def generate(prices: pd.DataFrame, method_id: str, origins: list[pd.Timestamp], horizon: int, n_paths: int = 300, seed: int = 0, version: str = "1") -> list[ForecastRecord]:
    """One distribution forecast per asset and origin from the existing forecasting method ``method_id``, each from the prices up to and including its origin and nothing later. ``available_at`` is the origin."""
    if method_id not in METHODS or method_id not in MEAN_METHODS:
        raise ForecastConversionError(f"choose a forecasting method with a mean from: {', '.join(MEAN_METHODS)}")
    m = METHODS[method_id]
    out: list[ForecastRecord] = []
    for origin in origins:
        past = prices.loc[:origin]
        for a in prices.columns:
            col = past[a].dropna()
            if len(col) < m.min_history:
                continue
            hist = History.from_frame(col)
            rng = np.random.default_rng([int(seed), pd.Timestamp(origin).toordinal(), zlib.crc32(str(a).encode())])
            try:
                state = m.fit(hist.tail(m.window), (horizon,))
                fc = m.predict(state, hist.tail(m.lookback), (horizon,), n_paths, rng)
            except (ForecastError, np.linalg.LinAlgError, FloatingPointError, ValueError):
                continue
            x = fc.at(horizon)
            out.append(ForecastRecord(instrument=str(a), origin=pd.Timestamp(origin), available_at=pd.Timestamp(origin), horizon_days=int(horizon), kind="distribution", units="log_return", samples=tuple(x.tolist()),
                                      uncertainty=float(x.std(ddof=1)), model_id=method_id, model_version=version, training_fingerprint=_fingerprint(str(a), col.to_numpy(float), method_id, n_paths, seed),
                                      provenance={"history_days": int(len(col)), "n_paths": int(n_paths), "seed": int(seed)}))
    return out


# ------------------------------------------------------------------------------------------------------------------------ reliability, from outcomes that were already known
def reliability(records: list[ForecastRecord], prices: pd.DataFrame, as_of: pd.Timestamp, base_currency: str | None = None) -> dict:
    """How the forecasts have done, using only records whose outcome had been observed by ``as_of`` (the origin plus the horizon on or before it): their information coefficient (rank correlation of the expected
    return with the realised one across assets, averaged over origins), hit rate for the sign, the slope of realised on forecast (a past-only calibration: below one the forecasts overstate), and the sample size."""
    pairs = []
    for r in records:
        pos = prices.index.get_indexer([pd.Timestamp(r.origin)])[0]
        if pos < 0 or pos + r.horizon_days >= len(prices) or prices.index[pos + r.horizon_days] > pd.Timestamp(as_of):
            continue
        try:
            er = expected_simple_return(r, base_currency).value
        except ForecastConversionError:
            continue
        y = float(prices[r.instrument].iloc[pos + r.horizon_days] / prices[r.instrument].iloc[pos] - 1.0)
        pairs.append((pd.Timestamp(r.origin), r.instrument, er, y))
    if len(pairs) < 20:
        return {"n": len(pairs), "status": "too few resolved forecasts to say", "ic": None, "hit_rate": None, "slope": None, "as_of": str(pd.Timestamp(as_of).date())}
    df = pd.DataFrame(pairs, columns=["origin", "asset", "forecast", "realised"])
    ics = [g["forecast"].rank().corr(g["realised"].rank()) for _, g in df.groupby("origin") if len(g) >= 3]
    ics = [v for v in ics if np.isfinite(v)]
    x, y = df["forecast"].to_numpy(), df["realised"].to_numpy()
    slope = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1)) if np.var(x) > 0 else None
    return {"n": int(len(df)), "origins": int(df["origin"].nunique()), "ic": float(np.mean(ics)) if ics else None, "hit_rate": float(np.mean(np.sign(x) == np.sign(y))), "slope": slope,
            "status": "past-only", "as_of": str(pd.Timestamp(as_of).date())}
