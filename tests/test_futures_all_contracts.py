"""Can the Futures tab see every contract a file holds? The curve used to list the nearest twelve whatever the file had; it now says how many there are and can list them all."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assets import futures_analytics as fa
from src.webapp.api import ApiError
from tests.test_webapp_assets_ml import _app
from tests.test_webapp_data import _safe
from tests.test_webapp_labs import _rich

N_CONTRACTS = 40
LAST = pd.Timestamp("2026-10-09")


def write_curve(path, n=N_CONTRACTS, days=200):
    """A crude-oil-like table: ``n`` monthly contracts alive on the last day, 20th-of-the-month expiries, a gently rising curve."""
    dates = pd.bdate_range(end=LAST, periods=days)
    rows = []
    for i in range(n):
        expiry = pd.Timestamp(2026, 11, 20) + pd.DateOffset(months=i)
        label = f"{'FGHJKMNQUVXZ'[expiry.month - 1]}{expiry.year % 100:02d}"
        for k, d in enumerate(dates):
            rows.append({"date": d, "contract": label, "expiry": expiry, "price": 70.0 + 0.25 * i + 0.01 * k + 0.3 * np.sin(k / 9.0), "open_interest": 100_000 - 1_500 * i, "volume": 5_000 - 80 * i})
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def curve_file(tmp_path):
    path = tmp_path / "ZZ.csv"
    write_curve(path)
    return path


def test_the_snapshot_lists_the_nearest_twelve_by_default_says_how_many_there_are_and_can_list_them_all(curve_file):
    data = fa.load_curve_file(curve_file, "ZZ")
    twelve, every, five = fa.curve_snapshot(data), fa.curve_snapshot(data, None), fa.curve_snapshot(data, 5)
    assert (len(twelve["rows"]), twelve["listed"]) == (12, N_CONTRACTS) and (len(every["rows"]), every["listed"]) == (N_CONTRACTS, N_CONTRACTS) and len(five["rows"]) == 5
    assert [r["contract"] for r in every["rows"][:12]] == [r["contract"] for r in twelve["rows"]]                      # the same contracts, the nearest first
    assert [r["rank"] for r in every["rows"]] == list(range(1, N_CONTRACTS + 1)) and every["rows"][-1]["roll_yield"] is None
    assert every["rows"][-1]["contract"] == "G30" and every["rows"][-1]["dte"] > 365 * 3
    assert twelve["state"] == every["state"] and twelve["annualised_carry"] == every["annualised_carry"] and twelve["curvature"] == every["curvature"]      # what is read off the front does not move


def test_the_earlier_curves_follow_the_same_depth(curve_file):
    data = fa.load_curve_file(curve_file, "ZZ")
    assert [len(h["points"]) for h in fa.curve_history(data, 3, 5)] == [12, 12, 12]
    assert [len(h["points"]) for h in fa.curve_history(data, 3, 5, None)] == [N_CONTRACTS] * 3


def test_the_endpoint_takes_a_number_of_contracts_or_all_of_them(tmp_path):
    app = _app(tmp_path, _rich, root=tmp_path)
    (tmp_path / "data" / "user" / "futures").mkdir(parents=True)
    write_curve(tmp_path / "data" / "user" / "futures" / "ZZ.csv")
    get = lambda **q: _safe(app.extra_get["/api/futures/curve"]({"root": "ZZ", **q}))                                    # noqa: E731
    default = get()
    assert len(default["curve"]["rows"]) == 12 and default["curve"]["listed"] == N_CONTRACTS and len(default["history"][0]["points"]) == 12
    assert len(default["shape"]["ranks"]) == 7 and len(default["pca"]["ranks"]) == 8                                    # M2 to M8; the components on the nearest eight
    for every in ("all", "0", "ALL"):
        full = get(contracts=every)
        assert len(full["curve"]["rows"]) == N_CONTRACTS and len(full["history"][0]["points"]) == N_CONTRACTS, every
    full = get(contracts="all")
    assert len(full["shape"]["ranks"]) == 23 and len(full["pca"]["ranks"]) == 8                                           # the shape follows up to 24 contract months, the components do not
    assert len(get(contracts=24)["curve"]["rows"]) == 24 and len(get(contracts=3)["curve"]["rows"]) == 3
    assert len(get(contracts=24)["shape"]["ranks"]) == 23 and len(get(contracts=60)["shape"]["ranks"]) == 23 and len(get(contracts=6)["shape"]["ranks"]) == 7 and len(get(contracts=12)["shape"]["ranks"]) == 7
    assert full["curve"]["rows"][-1]["contract"] == "G30" and full["carry"] == get()["carry"]
    for bad, text in (("1", "between 2 and 400"), ("401", "between 2 and 400"), ("lots", "must be a number")):
        with pytest.raises(ApiError, match=text):
            get(contracts=bad)
