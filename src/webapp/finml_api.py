"""The Financial ML Lab's endpoints (``src/finml``): event sampling, triple-barrier labelling, meta-labelling and the evidence that goes with them, on one ticker's daily prices.

``GET  /api/finml/plan``       the settings and what each means, the defaults, and what the lab does not do
``POST /api/finml/run``        run the whole pipeline as a job (``ticker``, ``start``, ``source``, ``refresh``, ``config``)
``POST /api/finml/structure``  structural-break and entropy statistics of the same ticker as a job

Information-driven bars are not offered here: they need individual trades, and the platform's daily data has none (``src.finml.bars`` takes a frame of trades). A run is not saved to a research session yet.
"""

from __future__ import annotations

import numpy as np

from ..finml import events as ev
from ..finml import structural as st
from ..finml.pipeline import BAGGING, DEFAULTS, MODELS, PRIMARIES, WEIGHTINGS, PipelineError, run_pipeline
from .api import ApiError
from .universe import UniverseError, normalise

# setting -> (type, low, high) for numbers, or the allowed values
LIMITS = {"lookback": (int, 2, 400), "fast": (int, 2, 200), "cusum_k": (float, 0.1, 10.0), "vol_span": (int, 5, 500), "pt": (float, 0.0, 10.0), "sl": (float, 0.0, 10.0), "vertical_bars": (int, 1, 120),
          "entry_lag": (int, 0, 5), "cost_bps": (float, 0.0, 200.0), "decay": (float, -1.0, 1.0), "n_bags": (int, 1, 30), "cv_splits": (int, 3, 10), "embargo": (int, 0, 60), "holdout": (float, 0.0, 0.45),
          "threshold": (float, 0.5, 0.95), "step": (float, 0.0, 1.0), "cpcv_groups": (int, 4, 10), "cpcv_test": (int, 1, 4), "n_trials": (int, 1, 100000), "seed": (int, 0, 2**31 - 1)}
CHOICES = {"primary": PRIMARIES, "weighting": WEIGHTINGS, "model": MODELS, "bagging": BAGGING, "same_bar": ev.SAME_BAR}

SETTINGS = [
    ("primary", "Primary signal", "The rule that proposes each trade's direction: momentum (the sign of the past return), mean reversion (the opposite), a moving-average cross, or always long. The meta-model only decides whether to take it."),
    ("lookback", "Lookback (days)", "The window of the momentum or mean-reversion signal, and the slow average of the cross."),
    ("fast", "Fast average (days)", "The fast average of the moving-average cross."),
    ("cusum_k", "CUSUM threshold (x daily volatility)", "An event is cut when the price has moved this many daily volatilities since the last one. Lower gives more events, more overlap and more redundancy."),
    ("vol_span", "Volatility span (days)", "The span of the exponentially weighted volatility that scales the barriers."),
    ("pt", "Profit barrier (x volatility)", "Distance of the profit-taking barrier from the entry, in multiples of the volatility known at the event. 0 switches it off."),
    ("sl", "Stop barrier (x volatility)", "Distance of the stop-loss barrier. 0 switches it off."),
    ("vertical_bars", "Holding limit (days)", "The vertical barrier: the trade is closed here if neither horizontal barrier was touched."),
    ("entry_lag", "Entry lag (days)", "0 fills at the close that produced the signal (optimistic); 1 fills at the next close, which is what a signal read at the close could actually get."),
    ("same_bar", "Both barriers in one bar", "Only with the High/Low option. When one day's range reaches both barriers the order cannot be known: assume the stop was hit first (default), the profit, or drop the event."),
    ("cost_bps", "Cost (bp per unit traded)", "Charged on every change of the position, and twice over on a label's round trip."),
    ("weighting", "Sample weights", "Uniqueness: each label counts for the share of its days it does not share. Attribution: by the return it is responsible for. None: every label counts equally."),
    ("decay", "Time decay", "1 keeps all weights; 0 fades the oldest linearly to zero; between 0 and 1 the oldest keeps that share; below 0 drops the oldest share of the sample."),
    ("model", "Meta-model", "Logistic regression (linear, steady) or a shallow random forest. It predicts whether the primary signal will pay."),
    ("bagging", "Bagging", "Sequential bootstrap: fit on several samples drawn so that overlapping labels are less often drawn together, and average the probabilities."),
    ("n_bags", "Bags", "How many sequential-bootstrap samples."),
    ("cv_splits", "Cross-validation folds", "Purged and embargoed folds of the development events."),
    ("embargo", "Embargo (days)", "Rows dropped from training after each test block."),
    ("holdout", "Untouched holdout (share of history)", "The last part of the history. Nothing is fitted, tuned or chosen on it; it is scored once. 0 leaves none, and the comparison is then development evidence only."),
    ("threshold", "Take the signal when P(pays) is at least", "Below 0.5 the meta-model would take signals it thinks more likely to lose than win."),
    ("step", "Size step", "0 keeps sizes continuous; 0.1 rounds the probability-based sizes to tenths."),
    ("cpcv_groups", "CPCV groups", "The development events are cut into this many ordered groups."),
    ("cpcv_test", "CPCV test groups", "Groups tested together in each split; the splits are recombined into out-of-sample paths."),
    ("n_trials", "Configurations tried", "How many settings combinations you looked at before this one. The deflated Sharpe ratio corrects for it; 1 means this is the only one."),
    ("importance", "Feature importance", "MDI, MDA (by cluster of correlated features) and SFI on the development events."),
]


def _config(raw: dict) -> dict:
    cfg = {}
    for key, value in (raw or {}).items():
        if key in CHOICES:
            if value not in CHOICES[key]:
                raise ApiError(f"{key} must be one of {', '.join(map(str, CHOICES[key]))}")
            cfg[key] = value
        elif key in LIMITS:
            kind, lo, hi = LIMITS[key]
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ApiError(f"{key} must be a number") from None
            if v != v or not lo <= v <= hi or (kind is int and v != int(v)):
                raise ApiError(f"{key} must be {'a whole number ' if kind is int else 'a number '}between {lo} and {hi}")
            cfg[key] = kind(v)
        elif key == "importance":
            cfg[key] = bool(value)
        elif key in DEFAULTS:
            continue
        else:
            raise ApiError(f"unknown setting {key!r}")
    return cfg


class FinmlApi:
    def __init__(self, app):
        self.app = app

    def get_routes(self) -> dict:
        return {"/api/finml/plan": self.plan}

    def post_routes(self) -> dict:
        return {"/api/finml/run": self.run, "/api/finml/structure": self.structure}

    def plan(self, q: dict | None = None) -> dict:
        return {"defaults": {k: v for k, v in DEFAULTS.items()}, "settings": [{"key": k, "label": label, "help": help_} for k, label, help_ in SETTINGS],
                "choices": {k: list(v) for k, v in CHOICES.items()}, "limits": {k: [v[1], v[2]] for k, v in LIMITS.items()},
                "steps": ["CUSUM event sampling", "triple-barrier labels with a primary side", "uniqueness-aware weights", "purged and combinatorial cross-validation", "meta-labelling", "probability-based sizing", "cost-aware backtest",
                          "evidence: PSR, deflated Sharpe, CPCV distribution, untouched holdout"],
                "notes": ["One ticker's daily prices. The last share of the history is never used to fit or choose anything.",
                          "The meta-labelled strategy is compared with the primary strategy it filters, with a random filter that skips as many signals, and with buy-and-hold. A negative result is reported as one.",
                          "Information-driven bars (tick, volume, dollar, imbalance, runs) need individual trades: they are in the library (src.finml.bars) and not offered here, because the platform's daily data has no trades.",
                          "CPCV is a research evaluation. It is not a prospective test: only forward, chronological testing is."]}

    def _prices(self, body: dict, progress):
        try:
            tickers = normalise(body.get("ticker") or [])
        except UniverseError as error:
            raise ApiError(str(error)) from None
        if len(tickers) != 1:
            raise ApiError("choose one ticker")
        try:
            self.app.builder.preflight(tickers, body.get("start"), body.get("source"))
        except UniverseError as error:
            raise ApiError(str(error)) from None
        return tickers[0]

    def run(self, body: dict) -> dict:
        ticker = self._prices(body, None)
        cfg = _config(body.get("config") or {})
        use_range = bool(body.get("use_range"))
        start, source, refresh = body.get("start") or None, body.get("source"), bool(body.get("refresh"))

        def work(progress):
            progress("loading the prices")
            try:
                bundle, universe = self.app.builder.resolve([ticker], None, start, refresh, source, progress)
            except UniverseError as error:
                raise ApiError(str(error)) from None
            close = bundle.prices[ticker].dropna()
            high = low = None
            if use_range:
                ratio = bundle.adjustment_ratio[ticker]
                high, low = (bundle.high[ticker] * ratio).reindex(close.index), (bundle.low[ticker] * ratio).reindex(close.index)
            try:
                result = run_pipeline(close, cfg, high=high, low=low, progress=lambda text, frac=None: progress(f"{text}" + ("" if frac is None else f" ({int(frac * 100)}%)")))
            except PipelineError as error:
                raise ApiError(str(error)) from None
            result["ticker"] = ticker
            result["universe"] = {"first": str(close.index[0].date()), "last": str(close.index[-1].date()), "days": int(len(close)), "source": universe.get("source"), "range_used": use_range}
            return result

        return self.app.submit_task(f"Financial ML: {ticker}", work, kind="finml")

    def structure(self, body: dict) -> dict:
        ticker = self._prices(body, None)
        start, source, refresh = body.get("start") or None, body.get("source"), bool(body.get("refresh"))
        window = int(body.get("window") or 750)
        if not 250 <= window <= 4000:
            raise ApiError("window must be 250 to 4000 days")

        def work(progress):
            progress("loading the prices")
            try:
                bundle, _ = self.app.builder.resolve([ticker], None, start, refresh, source, progress)
            except UniverseError as error:
                raise ApiError(str(error)) from None
            px = bundle.prices[ticker].dropna().iloc[-window:]
            logp = np.log(px)
            progress("supremum ADF (explosiveness)")
            s = st.sadf(logp, min_window=60, max_starts=20)
            key = f"{len(logp)}|60|40|10"                                                   # the simulated critical value depends on nothing but these numbers: asked twice, it is read, not simulated again
            crit = self.app.appdb.cache_get("sadf_critical", key, "1")
            if crit is None:
                crit = st.sadf_critical(len(logp), 60, reps=40, max_starts=10)
                self.app.appdb.cache_put("sadf_critical", key, "1", crit)
            progress("CUSUM test on levels")
            cs = st.chu_stinchcombe_white(logp, 30)
            progress("entropy")
            r = logp.diff().dropna()
            enc = {"sign": st.encode_returns(r, kind="sign"), "quartiles": st.encode_returns(r, 4)}
            ent = {k: {"plug_in_bits_per_symbol": st.plug_in_entropy(v, 2), "lempel_ziv": st.lempel_ziv(v), "kontoyiannis": st.kontoyiannis_entropy(v)} for k, v in enc.items()}
            recent = {k: {"plug_in_bits_per_symbol": st.plug_in_entropy(v[-250:], 2), "lempel_ziv": st.lempel_ziv(v[-250:])} for k, v in enc.items()}
            step = max(1, len(px) // 400)
            return {"ticker": ticker, "days": int(len(px)), "first": str(px.index[0].date()), "last": str(px.index[-1].date()),
                    "sadf": {"dates": [str(d.date()) for d in s.index[::step]], "values": [None if v != v else float(v) for v in s.to_numpy()[::step]], "critical_95": float(crit), "last": None if s.iloc[-1] != s.iloc[-1] else float(s.iloc[-1]),
                             "exceeds_share": float((s.dropna() > crit).mean()) if s.notna().any() else None, "note": "Simulated critical value under a random walk of the same length (40 repetitions): a guide, not a table value. A reading above it says the price was explosive over some window ending that day."},
                    "cusum": {"dates": [str(d.date()) for d in cs.index[::step]], "stat": [None if v != v else float(v) for v in cs["stat"].to_numpy()[::step]], "critical": [None if v != v else float(v) for v in cs["critical"].to_numpy()[::step]],
                              "break_share": float(cs["break_"].mean()), "last_break": None if not cs["break_"].any() else str(cs.index[cs["break_"].to_numpy()][-1].date())},
                    "entropy": {"whole_sample": ent, "last_250_days": recent,
                                "note": "Lower than a random sequence's value means more repeated structure in the encoded returns. Encoding discards information, and the quartile edges use the whole window."}}

        return self.app.submit_task(f"Structure and entropy: {ticker}", work, kind="finml")
