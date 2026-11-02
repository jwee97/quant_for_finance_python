"""Optional tax-lot and after-tax accounting under a supplied, versioned policy (spec v3, Release 8, P3-09). Off unless asked for; the pre-tax results it is given are never changed.

What this module is, and is not.

* **It carries no jurisdiction's rules.** A policy (rates, the holding period that makes a gain long-term, the lot-relief method, how fees count, loss netting and carry-forward, the payment lag) is
  *supplied* by the user, with its provenance: who supplied it, the document it comes from and, optionally, who verified it and when. No policy, an incomplete one, or one that needs a rule this
  module does not implement (wash sales, short sales, dividend or interest tax, withholding, currency conversion, brackets, carry-back, other tax years, ...) leaves the result ``unavailable``: the
  jurisdiction is never approximated by a nearby one. The module's only claim is arithmetic: given these rules, these lots and these fills, this is the tax.
* **Supported events.** Realised capital gains and losses on sales matched to lots by FIFO, LIFO or HIFO; transaction fees as an adjustment to basis and proceeds, or as not deductible, as the policy says;
  a short/long-term split at the policy's threshold; loss netting (none, within a term, across terms) and carry-forward of the unused losses, only as the policy says; a calendar-year tax period.
* **Unsupported ledger content blocks the run.** A sale larger than the lots held (a short sale), a non-positive price, a missing field: ``blocked`` with the reason, not a guess.
* **Pre-tax numbers stay intact.** The return series is read and never written; its hash before and after is reported and the pre-tax block is computed from the series alone.
* **After-tax wealth.** Tax is paid out of the portfolio on the policy's payment date: ``A_t = A_{t-1}(1 + r_t) - tax_t``. The tax is computed on a ledger sized at the pre-tax scale, which slightly overstates it
  once the base has shrunk, and the tax paid stops earning: both lean conservative. Tax not yet due at the end of the sample is deducted as an accrued liability. The tax that selling every open lot at the
  last prices would trigger is reported separately as a deferred liability, so a low-turnover book is not flattered by deferral; a negative figure (a benefit from realising losses) is shown but never counted
  as wealth.
* **Not tax advice.** The ledger and the return series are supplied together and are not reconciled with each other; quantities and prices are as traded (not split-adjusted); one currency throughout.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

VERSION = "tax-lots-v1"
POLICY_SCHEMA = "tax-policy-v1"
RULE_FAMILY = "lot_gain_loss_v1"
LOT_METHODS = ("fifo", "lifo", "hifo")
FEE_TREATMENTS = ("capitalise", "not_deductible")
LOSS_OFFSETS = ("none", "within_term", "across_terms")
EVENTS_SUPPORTED = ["realised capital gains and losses on sales, matched to lots (FIFO, LIFO or HIFO)",
                    "transaction fees as an adjustment to basis and proceeds, or as not deductible, per the policy",
                    "a short/long-term split at the policy's holding-period threshold",
                    "loss netting (none, within a term, across terms) and carry-forward, only as the policy declares"]
EVENTS_NOT_SUPPORTED = ["wash sales", "short sales", "dividend and interest tax", "withholding tax", "currency conversion", "progressive brackets or surtaxes", "loss carry-back or expiry of carried losses",
                        "ordinary-income offsets", "corporate actions (splits, mergers)", "derivatives", "fiscal years other than the calendar year"]
_ALLOWED = {"schema", "jurisdiction", "policy_version", "investor", "rule_family", "currency", "provenance", "lot_method", "fee_treatment", "long_term_min_days", "rates", "period", "loss_offset",
            "loss_carryforward", "payment_lag_days", "features"}
_REQUIRED = ("schema", "jurisdiction", "policy_version", "rule_family", "currency", "provenance", "lot_method", "fee_treatment", "long_term_min_days", "rates", "loss_offset", "loss_carryforward")
_EPS = 1e-9


# ---------------------------------------------------------------------------------------------------------------------------------- policy
@dataclass(frozen=True)
class TaxPolicy:
    jurisdiction: str
    version: str
    investor: str
    currency: str
    lot_method: str
    fee_treatment: str
    long_term_min_days: int
    short_rate: float
    long_rate: float
    loss_offset: str
    loss_carryforward: bool
    payment_lag_days: int
    supplied_by: str
    source: str
    verified_by: str | None = None
    verified_on: str | None = None

    @property
    def verification(self) -> str:
        return "verified" if self.verified_by and self.verified_on else "supplied_unverified"

    def content(self) -> dict:
        return {"schema": POLICY_SCHEMA, "rule_family": RULE_FAMILY, "jurisdiction": self.jurisdiction, "policy_version": self.version, "investor": self.investor, "currency": self.currency,
                "lot_method": self.lot_method, "fee_treatment": self.fee_treatment, "long_term_min_days": self.long_term_min_days, "rates": {"short_term": self.short_rate, "long_term": self.long_rate},
                "period": "calendar_year", "loss_offset": self.loss_offset, "loss_carryforward": self.loss_carryforward, "payment_lag_days": self.payment_lag_days,
                "provenance": {"supplied_by": self.supplied_by, "source": self.source, "verified_by": self.verified_by, "verified_on": self.verified_on}}

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.content(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def summary(self) -> dict:
        return {"jurisdiction": self.jurisdiction, "policy_version": self.version, "investor": self.investor, "hash": self.hash(), "verification": self.verification, "supplied_by": self.supplied_by,
                "source": self.source, "verified_by": self.verified_by, "verified_on": self.verified_on, "content": self.content()}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _text(x) -> bool:
    return isinstance(x, str) and bool(x.strip())


def parse_policy(raw) -> dict:
    """Validate a supplied policy. ``status`` is ``applied`` with the parsed ``policy`` or ``unavailable`` with the ``problems`` (malformed or missing parts) and the ``unsupported`` rules named."""
    if not isinstance(raw, dict) or not raw:
        return {"status": "unavailable", "policy": None, "problems": [], "unsupported": [],
                "reason": "no tax policy was supplied: this module carries no jurisdiction's rules and assumes none"}
    problems, unsupported = [], []
    unknown = sorted(set(raw) - _ALLOWED)
    if unknown:
        problems.append(f"unknown policy field(s) {unknown}: a misspelt rule would otherwise be ignored without a trace")
    problems += [f"the policy lacks '{k}'" for k in _REQUIRED if k not in raw]
    if raw.get("schema", POLICY_SCHEMA) != POLICY_SCHEMA:
        problems.append(f"the policy schema is '{raw.get('schema')}'; this module reads '{POLICY_SCHEMA}'")
    for k in ("jurisdiction", "policy_version", "currency"):
        if k in raw and not _text(raw[k]):
            problems.append(f"'{k}' is a non-empty text")
    if "investor" in raw and not _text(raw["investor"]):
        problems.append("'investor' is a non-empty text when given")
    fam = raw.get("rule_family")
    if fam is not None and fam != RULE_FAMILY:
        unsupported.append(f"rule family '{fam}' (only '{RULE_FAMILY}' is implemented)")
    if raw.get("period", "calendar_year") != "calendar_year":
        unsupported.append(f"tax period '{raw.get('period')}' (only the calendar year is implemented)")
    feats = raw.get("features", [])
    if not isinstance(feats, list) or not all(_text(f) for f in feats):
        problems.append("'features' is a list of the extra rules the policy needs, each a text")
    else:
        unsupported += [f"rule '{f}'" for f in feats]
    prov = raw.get("provenance")
    if "provenance" in raw:
        if not isinstance(prov, dict):
            problems.append("'provenance' is a mapping with who supplied the policy and its source")
        else:
            for k in ("supplied_by", "source"):
                if not _text(prov.get(k)):
                    problems.append(f"provenance.{k} is required: a policy with no named supplier and source cannot be audited")
            vb, vo = prov.get("verified_by"), prov.get("verified_on")
            if (vb is None) != (vo is None) or (vb is not None and not (_text(vb) and _text(vo))):
                problems.append("provenance.verified_by and provenance.verified_on go together (or neither is given)")
            extra = sorted(set(prov) - {"supplied_by", "source", "verified_by", "verified_on"})
            if extra:
                problems.append(f"unknown provenance field(s) {extra}")
    if "lot_method" in raw and raw["lot_method"] not in LOT_METHODS:
        problems.append(f"lot_method is one of {list(LOT_METHODS)}")
    if "fee_treatment" in raw and raw["fee_treatment"] not in FEE_TREATMENTS:
        problems.append(f"fee_treatment is one of {list(FEE_TREATMENTS)}")
    if "loss_offset" in raw and raw["loss_offset"] not in LOSS_OFFSETS:
        problems.append(f"loss_offset is one of {list(LOSS_OFFSETS)}")
    if "long_term_min_days" in raw and not (_int(raw["long_term_min_days"]) and raw["long_term_min_days"] >= 1):
        problems.append("long_term_min_days is a whole number of calendar days, at least 1: a gain is long-term once sale date minus acquisition date reaches it")
    if "loss_carryforward" in raw and not isinstance(raw["loss_carryforward"], bool):
        problems.append("loss_carryforward is true or false")
    if raw.get("loss_carryforward") is True and raw.get("loss_offset") == "none":
        problems.append("a loss carry-forward needs loss_offset to allow losses to be used (within_term or across_terms)")
    if "payment_lag_days" in raw and not (_int(raw["payment_lag_days"]) and raw["payment_lag_days"] >= 0):
        problems.append("payment_lag_days is a whole number of days, not negative")
    rates = raw.get("rates")
    if "rates" in raw:
        if not isinstance(rates, dict) or set(rates) != {"short_term", "long_term"}:
            problems.append("rates has exactly 'short_term' and 'long_term'")
        else:
            for k, v in rates.items():
                if not (_num(v) and 0.0 <= float(v) <= 1.0):
                    problems.append(f"rates.{k} is a rate between 0 and 1")
    if problems or unsupported:
        why = []
        if problems:
            why.append("the supplied policy is incomplete or malformed: " + "; ".join(problems))
        if unsupported:
            why.append("the policy needs rules this module does not implement: " + "; ".join(unsupported) + "; the jurisdiction stays unavailable rather than being approximated")
        return {"status": "unavailable", "policy": None, "problems": problems, "unsupported": unsupported, "reason": ". ".join(why)}
    prov = raw["provenance"]
    pol = TaxPolicy(jurisdiction=raw["jurisdiction"], version=raw["policy_version"], investor=raw.get("investor", "unspecified"), currency=raw["currency"], lot_method=raw["lot_method"],
                    fee_treatment=raw["fee_treatment"], long_term_min_days=int(raw["long_term_min_days"]), short_rate=float(rates["short_term"]), long_rate=float(rates["long_term"]),
                    loss_offset=raw["loss_offset"], loss_carryforward=bool(raw["loss_carryforward"]), payment_lag_days=int(raw.get("payment_lag_days", 0)), supplied_by=prov["supplied_by"],
                    source=prov["source"], verified_by=prov.get("verified_by"), verified_on=prov.get("verified_on"))
    return {"status": "applied", "policy": pol, "problems": [], "unsupported": [], "reason": None}


# ---------------------------------------------------------------------------------------------------------------------------------- ledger and lots
def check_ledger(ledger, opening=None) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    """The normalised fills (date, asset, quantity, price, fee, seq) and opening lots, or the problems that make them unusable. Positive quantity buys, negative sells; within a date the given order is kept."""
    problems: list[str] = []
    if not isinstance(ledger, pd.DataFrame):
        return None, None, ["the ledger is a table of fills with date, asset, quantity, price and (optionally) fee"]
    missing = [c for c in ("date", "asset", "quantity", "price") if c not in ledger.columns]
    if missing:
        return None, None, [f"the ledger lacks the column(s) {missing}"]
    df = ledger[[c for c in ("date", "asset", "quantity", "price", "fee") if c in ledger.columns]].copy().reset_index(drop=True)
    if "fee" not in df:
        df["fee"] = 0.0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("quantity", "price", "fee"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["asset"] = df["asset"].astype(str)
    if df["date"].isna().any():
        problems.append(f"{int(df['date'].isna().sum())} fills have no usable date")
    if not np.isfinite(df[["quantity", "price", "fee"]].to_numpy(dtype=float)).all():
        problems.append("missing or non-finite quantity, price or fee in the ledger: they are not filled in")
    else:
        if (df["quantity"] == 0).any():
            problems.append("a fill has zero quantity")
        if (df["price"] <= 0).any():
            problems.append("a fill has a price that is not positive")
        if (df["fee"] < 0).any():
            problems.append("a fill has a negative fee")
    if (df["asset"].str.strip() == "").any():
        problems.append("a fill has no asset")
    op = None
    if opening is not None:
        if not isinstance(opening, pd.DataFrame) or any(c not in opening.columns for c in ("asset", "quantity", "unit_price", "acquired")):
            problems.append("opening lots are a table with asset, quantity, unit_price and acquired")
        else:
            op = opening[["asset", "quantity", "unit_price", "acquired"]].copy().reset_index(drop=True)
            op["acquired"] = pd.to_datetime(op["acquired"], errors="coerce")
            op["asset"] = op["asset"].astype(str)
            for c in ("quantity", "unit_price"):
                op[c] = pd.to_numeric(op[c], errors="coerce")
            if op["acquired"].isna().any() or not np.isfinite(op[["quantity", "unit_price"]].to_numpy(dtype=float)).all() or (op["quantity"] <= 0).any() or (op["unit_price"] <= 0).any():
                problems.append("opening lots need a date, a positive quantity and a positive unit price")
    if problems:
        return None, None, problems
    n_open = 0 if op is None else len(op)
    df["seq"] = np.arange(len(df)) + n_open
    ordered = bool(df["date"].is_monotonic_increasing)
    df = df.sort_values(["date", "seq"], kind="mergesort").reset_index(drop=True)
    df.attrs["input_was_date_ordered"] = ordered
    return df, op, []


@dataclass
class _Lot:
    asset: str
    acquired: pd.Timestamp
    seq: int
    qty: float
    unit_price: float
    unit_fee: float
    opening: bool = False


def _basis(lot: _Lot, policy: TaxPolicy) -> float:
    return lot.unit_price + (lot.unit_fee if policy.fee_treatment == "capitalise" else 0.0)


def _order(lots: list[_Lot], policy: TaxPolicy) -> list[_Lot]:
    if policy.lot_method == "fifo":
        return sorted(lots, key=lambda x: (x.acquired, x.seq))
    if policy.lot_method == "lifo":
        return sorted(lots, key=lambda x: (x.acquired, x.seq), reverse=True)
    return sorted(lots, key=lambda x: (-_basis(x, policy), x.acquired, x.seq))                       # hifo: highest unit basis first, then the oldest


def relieve_lots(ledger, policy: TaxPolicy, opening=None) -> dict:
    """Match every sale to lots by the policy's method. ``applied`` with the realised ``slices`` (one per lot touched), the ``open_lots`` and the fees; ``blocked`` with the reason otherwise."""
    df, op, problems = check_ledger(ledger, opening)
    if problems:
        return {"status": "blocked", "reason": "the ledger cannot be used: " + "; ".join(problems), "problems": problems}
    lots: dict[str, list[_Lot]] = {}
    seq = 0
    if op is not None:
        for r in op.itertuples(index=False):
            lots.setdefault(r.asset, []).append(_Lot(r.asset, r.acquired, seq, float(r.quantity), float(r.unit_price), 0.0, True))
            seq += 1
    slices: list[dict] = []
    fees = {"buy": 0.0, "sell": 0.0}
    for r in df.itertuples(index=False):
        q = float(r.quantity)
        if q > 0:
            lots.setdefault(r.asset, []).append(_Lot(r.asset, r.date, int(r.seq), q, float(r.price), float(r.fee) / q))
            fees["buy"] += float(r.fee)
            continue
        sell = -q
        held = sum(x.qty for x in lots.get(r.asset, []))
        if sell > held * (1.0 + _EPS) + 1e-12:
            return {"status": "blocked", "problems": ["short sale"], "reason": f"{r.date.date()}: the sale of {sell:g} {r.asset} exceeds the {held:g} held; short sales are not supported (they are tax events this module does not implement)"}
        fees["sell"] += float(r.fee)
        remaining = sell
        for lot in _order(lots.get(r.asset, []), policy):
            if remaining <= 1e-12:
                break
            take = min(lot.qty, remaining)
            fee_part = float(r.fee) * take / sell
            proceeds = float(r.price) * take - (fee_part if policy.fee_treatment == "capitalise" else 0.0)
            cost = _basis(lot, policy) * take
            days = int((r.date - lot.acquired).days)
            slices.append({"asset": r.asset, "sale_date": r.date, "acquired": lot.acquired, "quantity": take, "proceeds": proceeds, "cost": cost, "gain": proceeds - cost, "days_held": days,
                           "term": "long" if days >= policy.long_term_min_days else "short", "period": int(r.date.year), "opening_lot": lot.opening})
            lot.qty -= take
            remaining -= take
        lots[r.asset] = [x for x in lots.get(r.asset, []) if x.qty > 1e-12]
    open_lots = [{"asset": x.asset, "acquired": x.acquired, "quantity": x.qty, "unit_basis": _basis(x, policy), "cost": _basis(x, policy) * x.qty, "opening_lot": x.opening}
                 for a in sorted(lots) for x in sorted(lots[a], key=lambda y: (y.acquired, y.seq))]
    return {"status": "applied", "slices": pd.DataFrame(slices, columns=["asset", "sale_date", "acquired", "quantity", "proceeds", "cost", "gain", "days_held", "term", "period", "opening_lot"]),
            "open_lots": pd.DataFrame(open_lots, columns=["asset", "acquired", "quantity", "unit_basis", "cost", "opening_lot"]), "fees": fees, "ledger_rows": int(len(df)),
            "input_was_date_ordered": bool(df.attrs.get("input_was_date_ordered", True))}


# ---------------------------------------------------------------------------------------------------------------------------------- periods and tax
def settle_periods(slices: pd.DataFrame, policy: TaxPolicy, first_year: int, last_year: int) -> pd.DataFrame:
    """The tax of every calendar year from ``first_year`` to ``last_year``: gains and losses by term, netting and carry-forward exactly as the policy says, and the tax at the policy's rates. A year with
    no sales still passes any carried loss along (nothing expires: expiry is not implemented)."""
    rows = []
    carry_st = carry_lt = 0.0
    for y in range(int(first_year), int(last_year) + 1):
        s = slices[slices["period"] == y] if len(slices) else slices
        g = lambda term, sign: float(s.loc[(s["term"] == term) & (np.sign(s["gain"]) == sign), "gain"].abs().sum()) if len(s) else 0.0                 # noqa: E731
        g_st, l_st, g_lt, l_lt = g("short", 1), g("short", -1), g("long", 1), g("long", -1)
        cross = 0.0
        in_st, in_lt = carry_st, carry_lt
        if policy.loss_offset == "none":
            net_st, net_lt = g_st, g_lt
            taxable_st, taxable_lt = g_st, g_lt
            carry_st = carry_lt = 0.0
        else:
            net_st, net_lt = g_st - l_st - carry_st, g_lt - l_lt - carry_lt
            if policy.loss_offset == "across_terms":
                if net_st < 0.0 < net_lt:
                    cross = min(-net_st, net_lt)
                    net_st, net_lt = net_st + cross, net_lt - cross
                elif net_lt < 0.0 < net_st:
                    cross = min(-net_lt, net_st)
                    net_lt, net_st = net_lt + cross, net_st - cross
            taxable_st, taxable_lt = max(net_st, 0.0), max(net_lt, 0.0)
            carry_st, carry_lt = (max(-net_st, 0.0), max(-net_lt, 0.0)) if policy.loss_carryforward else (0.0, 0.0)
        rows.append({"period": y, "gains_short": g_st, "losses_short": l_st, "gains_long": g_lt, "losses_long": l_lt, "carry_in_short": in_st, "carry_in_long": in_lt, "offset_across_terms": cross,
                     "taxable_short": taxable_st, "taxable_long": taxable_lt, "carry_out_short": carry_st, "carry_out_long": carry_lt, "tax": taxable_st * policy.short_rate + taxable_lt * policy.long_rate})
    return pd.DataFrame(rows)


def ledger_from_positions(positions: pd.DataFrame, prices: pd.DataFrame, equity: pd.Series, *, fee_bps: float = 0.0, min_trade_value: float = 0.0) -> pd.DataFrame:
    """An illustrative ledger from a book of weights: the shares held after each day are ``weight x equity / price``; a fill is the day's change in shares at that day's price, with a fee of ``fee_bps`` of
    the traded value. It is a stand-in for real fills, not a reconstruction of the engine's own trades; negative weights become sales the accounting will refuse (short sales)."""
    w = positions.sort_index()
    px = prices.reindex(index=w.index, columns=w.columns)
    eq = equity.reindex(w.index)
    if eq.isna().any() or px.isna().any().any():
        raise ValueError("prices and equity are needed on every date of the positions")
    shares = w.mul(eq, axis=0).div(px)
    change = shares.diff().fillna(shares)
    tol = max(float(min_trade_value), 1e-9 * float(eq.abs().max()))
    out = []
    for d in change.index:
        for a in change.columns:
            q = float(change.at[d, a])
            value = abs(q) * float(px.at[d, a])
            if value > tol:
                out.append({"date": d, "asset": str(a), "quantity": q, "price": float(px.at[d, a]), "fee": value * float(fee_bps) / 1e4})
    return pd.DataFrame(out, columns=["date", "asset", "quantity", "price", "fee"])


# ---------------------------------------------------------------------------------------------------------------------------------- the report
def _hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="float64").tobytes()).hexdigest()


def _perf(final: float, c0: float, n_days: int) -> dict:
    years = n_days / 252.0
    return {"final_wealth": float(final), "total_return": float(final / c0 - 1.0), "cagr": float((final / c0) ** (1.0 / years) - 1.0) if final > 0 and years > 0 else None}


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    out = []
    for r in (df if limit is None else df.head(limit)).to_dict("records"):
        out.append({k: (str(v.date()) if isinstance(v, pd.Timestamp) else (bool(v) if isinstance(v, (np.bool_, bool)) else (float(v) if isinstance(v, (np.floating,)) else (int(v) if isinstance(v, np.integer) else v))))
                    for k, v in r.items()})
    return out


def run(policy_raw, ledger, pre_tax_returns: pd.Series, initial_capital: float, *, opening=None, last_prices: dict | pd.Series | None = None, liquidation_fee_bps: float = 0.0, max_slices: int = 200) -> dict:
    """Account for the lots, the tax and the after-tax wealth of a supplied ledger under a supplied policy, next to the pre-tax figures of ``pre_tax_returns`` (daily simple net returns, left untouched).
    ``status`` is ``applied``, ``unavailable`` (no usable policy) or ``blocked`` (a ledger or return series the supported events cannot represent)."""
    parsed = parse_policy(policy_raw)
    head = {"version": VERSION, "events_supported": EVENTS_SUPPORTED, "events_not_supported": EVENTS_NOT_SUPPORTED, "not_tax_advice": True}
    if parsed["status"] != "applied":
        return {**head, "status": "unavailable", "reason": parsed["reason"], "problems": parsed["problems"], "unsupported": parsed["unsupported"], "policy": None, "after_tax": None}
    pol: TaxPolicy = parsed["policy"]
    head["policy"] = pol.summary()

    def blocked(reason: str, **kw) -> dict:
        return {**head, "status": "blocked", "reason": reason, "after_tax": None, **kw}

    r = pd.Series(pre_tax_returns, dtype=float)
    if not isinstance(r.index, pd.DatetimeIndex) or len(r) < 2 or not r.index.is_monotonic_increasing or r.index.has_duplicates:
        return blocked("the pre-tax returns are a daily series on a sorted, unique date index with at least two days")
    if not np.isfinite(r.to_numpy()).all() or (r <= -1.0).any():
        return blocked("the pre-tax returns hold missing values or a loss of the whole portfolio")
    if not (_num(initial_capital) and initial_capital > 0):
        return blocked("the initial capital is a positive number")
    before = _hash(np.asarray(pre_tax_returns, dtype=float))                                       # the caller's own object, hashed now and again at the end
    c0 = float(initial_capital)
    n = len(r)
    pre_final = c0 * float(np.prod(1.0 + r.to_numpy()))
    pre_tax = _perf(pre_final, c0, n)
    acc = relieve_lots(ledger, pol, opening)
    if acc["status"] != "applied":
        return blocked(acc["reason"], problems=acc.get("problems"))
    slices, open_lots = acc["slices"], acc["open_lots"]
    ld = check_ledger(ledger, opening)[0]
    if ld["date"].min() < r.index[0] or ld["date"].max() > r.index[-1]:
        return blocked(f"the ledger runs {ld['date'].min().date()} to {ld['date'].max().date()}, outside the return series ({r.index[0].date()} to {r.index[-1].date()}): taxes could not be placed in the wealth path (holdings bought earlier belong in the opening lots)")
    y0, y1 = int(r.index[0].year), int(r.index[-1].year)
    periods = settle_periods(slices, pol, y0, y1)
    paid = np.zeros(n)
    accrued = 0.0
    schedule = []
    for row in periods.itertuples(index=False):
        if row.tax <= 0:
            continue
        due = pd.Timestamp(int(row.period), 12, 31) + pd.Timedelta(days=pol.payment_lag_days)
        pos = int(r.index.searchsorted(due, side="left"))
        if pos < n:
            paid[pos] += float(row.tax)
        else:
            accrued += float(row.tax)
        schedule.append({"period": int(row.period), "tax": float(row.tax), "due": str(due.date()), "paid_on": str(r.index[pos].date()) if pos < n else None})
    wealth = c0
    for i, x in enumerate(r.to_numpy()):
        wealth = wealth * (1.0 + float(x)) - paid[i]
        if wealth <= 0:
            return blocked(f"the tax due on {r.index[i].date()} exceeds the portfolio's value: the ledger and the return series are not consistent with each other")
    realised_wealth = wealth - accrued
    out = {**head, "status": "applied", "reason": None,
           "ledger": {"rows": acc["ledger_rows"], "assets": sorted(set(ld["asset"])), "first": str(ld["date"].min().date()) if len(ld) else None, "last": str(ld["date"].max().date()) if len(ld) else None, "opening_lots": 0 if opening is None else int(len(opening)),
                      "input_was_date_ordered": acc["input_was_date_ordered"]},
           "fees": {"buy": acc["fees"]["buy"], "sell": acc["fees"]["sell"], "treatment": pol.fee_treatment},
           "realised": {"slices_total": int(len(slices)), "slices": _records(slices, max_slices), "net_gain": float(slices["gain"].sum()) if len(slices) else 0.0,
                        "net_gain_short": float(slices.loc[slices["term"] == "short", "gain"].sum()) if len(slices) else 0.0, "net_gain_long": float(slices.loc[slices["term"] == "long", "gain"].sum()) if len(slices) else 0.0},
           "periods": _records(periods), "payments": schedule, "open_lots": _records(open_lots),
           "pre_tax": pre_tax, "pre_tax_returns_hash": before}
    total_tax = float(periods["tax"].sum())
    net_gain = out["realised"]["net_gain"]
    taxes = {"total": total_tax, "paid": float(paid.sum()), "accrued_unpaid": float(accrued), "effective_rate_on_net_realised_gains": float(total_tax / net_gain) if net_gain > 0 else None}
    deferred = None
    if len(open_lots) == 0:
        deferred = {"status": "applied", "value": 0.0, "counted_in_wealth": 0.0, "reason": "no open lots"}
    else:
        assets = sorted(set(open_lots["asset"]))
        px = {} if last_prices is None else {str(k): float(v) for k, v in dict(last_prices).items()}
        lacking = [a for a in assets if not (a in px and math.isfinite(px[a]) and px[a] > 0)]
        if lacking:
            deferred = {"status": "unavailable", "value": None, "counted_in_wealth": None, "reason": f"no closing price for {lacking}: the tax on the unrealised gains cannot be estimated, and the after-tax figures exclude it"}
        else:
            end = r.index[-1]
            sales = [{"date": end, "asset": a, "quantity": -float(open_lots.loc[open_lots["asset"] == a, "quantity"].sum()), "price": px[a],
                      "fee": abs(float(open_lots.loc[open_lots["asset"] == a, "quantity"].sum())) * px[a] * float(liquidation_fee_bps) / 1e4} for a in assets]
            aug = pd.concat([ld[["date", "asset", "quantity", "price", "fee"]], pd.DataFrame(sales)], ignore_index=True)
            again = relieve_lots(aug, pol, opening)
            tot_aug = float(settle_periods(again["slices"], pol, y0, y1)["tax"].sum()) if again["status"] == "applied" else None
            if tot_aug is None:
                deferred = {"status": "unavailable", "value": None, "counted_in_wealth": None, "reason": again.get("reason")}
            else:
                val = tot_aug - total_tax
                deferred = {"status": "applied", "value": val, "counted_in_wealth": max(val, 0.0), "last_prices": px, "liquidation_fee_bps": float(liquidation_fee_bps),
                            "reason": "the extra tax if every open lot were sold on the last date at the last prices; a negative value is a benefit and is not counted as wealth"}
    taxes["deferred_if_liquidated"] = deferred
    out["taxes"] = taxes
    after = {"realised": _perf(realised_wealth, c0, n)}
    after["if_liquidated"] = (_perf(realised_wealth - deferred["counted_in_wealth"], c0, n) if deferred["status"] == "applied"
                              else {"status": "unavailable", "reason": deferred["reason"]})
    out["after_tax"] = after
    out["tax_drag"] = {"total_return_points": pre_tax["total_return"] - after["realised"]["total_return"],
                       "cagr_points": (pre_tax["cagr"] - after["realised"]["cagr"]) if pre_tax["cagr"] is not None and after["realised"]["cagr"] is not None else None}
    out["pre_tax_unchanged"] = _hash(np.asarray(pre_tax_returns, dtype=float)) == before
    out["limitations"] = ["the module carries no jurisdiction's rules: the policy is the supplier's, and its provenance is reported with every result",
                          "tax is paid out of the portfolio on the policy's payment date; it is computed on a ledger sized at the pre-tax scale and the tax paid stops earning (both lean conservative)",
                          "the ledger and the return series are supplied together and are not reconciled; quantities and prices are as traded, in one currency",
                          "carried losses never expire and no loss is carried back; only the supported events above are taxed"] + (
                         ["the policy has not been verified by a named person: treat the figures as arithmetic under an unverified rule set"] if pol.verification != "verified" else [])
    return out
