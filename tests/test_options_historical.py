"""The empirical options backtest replays stored chains at their own quotes and refuses what it would have to model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets import options_analytics as oa
from src.assets import options_historical as oh
from src.assets.option_history import History


def store(tmp_path, n=45, source="downloaded", drift=0.0):
    h = History(tmp_path)
    rng = np.random.default_rng(3)
    spot = 100.0
    for i in range(n):
        day = pd.Timestamp("2025-01-02") + pd.Timedelta(days=i)
        spot *= 1 + rng.normal(drift, 0.01)
        h.save("XYZ", oa.prepare_chain(oa.build_model_chain(spot, day, 0.2)), day, source)
    return h


def test_a_short_put_is_opened_at_the_bid_marked_and_settled_from_stored_chains(tmp_path):
    h = store(tmp_path)
    out = oh.run(h, "XYZ", "cash_secured_put", dte=10, every=5, min_dates=20)
    assert out["state"] == "ok" and out["mode"]["id"] == "empirical" and out["trades"] and out["summary"]["trades"] == len(out["trades"])
    t = out["trades"][0]
    leg = t["legs"][0]
    chain = h.load("XYZ", t["open"])
    row = chain[(chain["expiry"] == pd.Timestamp(leg["expiry"])) & (chain["strike"] == leg["strike"]) & (chain["right"] == "P")].iloc[0]
    assert leg["entry"] == row["bid"] and t["entry_debit"] == pytest.approx(-100 * row["bid"])
    end = h.load("XYZ", t["settle_date"])
    intrinsic = max(leg["strike"] - float(end["underlying"].iloc[0]), 0.0)
    assert t["pnl"] == pytest.approx(-100 * intrinsic - t["entry_debit"] - 2 * t["fees"])
    assert out["in_sample"]["trades"] + out["out_of_sample"]["trades"] == out["summary"]["trades"] and "American early exercise" in " ".join(out["not_modelled"])


def test_it_refuses_simulated_snapshots_and_too_short_histories(tmp_path):
    assert oh.run(store(tmp_path / "a", source="simulated"), "XYZ")["state"] == "unavailable"
    short = oh.run(store(tmp_path / "b", n=8), "XYZ")
    assert short["state"] == "unavailable" and "at least 20" in short["reason"] and short["needs"]
    with pytest.raises(ValueError):
        oh.run(store(tmp_path / "c"), "XYZ", "nope")
