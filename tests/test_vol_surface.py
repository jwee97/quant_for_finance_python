"""The volatility surface is built from the quotes: cleaning, one grid, masked gaps, no extrapolation, raw apart from fitted, and honest when the data are too thin."""
import numpy as np
import pandas as pd
import pytest

from src.assets import options_analytics as oa
from src.assets import vol_surface as vs


@pytest.fixture(scope="module")
def chain():
    return oa.prepare_chain(oa.build_model_chain(100.0, "2026-01-02", 0.2, 0.25))


def _slice(chain, n_expiries=3):
    exps = sorted(chain["expiry"].unique())[:n_expiries]
    return chain[chain["expiry"].isin(exps)].copy()


def test_a_normal_chain_gives_a_surface_with_one_grid_and_counted_rejections(chain):
    r = vs.build(chain)
    m = r["meta"]
    assert m["status"] == "surface" and m["extrapolated_nodes"] == 0 and sum(m["quotes"]["rejected"].values()) + m["quotes"]["kept"] == m["quotes"]["received"]
    z = np.array(r["z"]["raw"]["vol"], dtype=float)
    assert z.shape == (len(r["y"]), len(r["x"])) and 0.2 < r["coverage"]["share"] < 1.0 and np.isnan(z).any()
    assert r["y"] == sorted(r["y"]) and all(len(row) == len(r["x"]) for row in r["z"]["raw"]["w"])


def test_nothing_is_extrapolated_every_filled_node_lies_inside_its_own_expiry_quotes(chain):
    r = vs.build(chain, nx=61)
    x = np.array(r["x"])
    for row, info in zip(r["z"]["raw"]["vol"], [s for s in r["slices"] if s["usable"]]):
        filled = x[[v is not None for v in row]]
        if len(filled):
            assert filled.min() >= info["x_min"] - 1e-9 and filled.max() <= info["x_max"] + 1e-9


def test_a_gap_in_the_strikes_is_masked_and_not_bridged(chain):
    c = _slice(chain, 8)
    first = sorted(c["expiry"].unique())[6]
    kk = np.log(c["strike"] / c["F"])
    hole = (c["expiry"] == first) & (kk > 0.01) & (kk < 0.16)
    r = vs.build(c[~hole], nx=101)
    info = [s for s in r["slices"] if s["expiry"] == str(pd.Timestamp(first).date())][0]
    assert info["gaps"], "the removed strikes leave a gap that is reported"
    row = r["z"]["raw"]["vol"][[s["expiry"] for s in r["slices"] if s["usable"]].index(info["expiry"])]
    x = np.array(r["x"])
    g0, g1 = info["gaps"][0]
    inside = (x > g0) & (x < g1)
    assert inside.any() and all(row[i] is None for i in np.flatnonzero(inside))


def test_a_single_expiry_gives_smiles_not_a_surface(chain):
    r = vs.build(_slice(chain, 1))
    assert r["meta"]["status"] == "smiles_only" and "two maturities" in r["meta"]["reason"] and r["z"]["raw"]["vol"] == [] and len(r["observations"]) > 5


def test_a_sparse_expiry_is_listed_but_not_drawn(chain):
    c = _slice(chain, 3)
    e = sorted(c["expiry"].unique())[1]
    keep = c[c["expiry"] != e]
    few = c[c["expiry"] == e]
    few = few[(few["right"] == "P") & (few["strike"] <= few["F"])].head(2)
    r = vs.build(pd.concat([keep, few]))
    thin = [s for s in r["slices"] if s["expiry"] == str(pd.Timestamp(e).date())][0]
    assert not thin["usable"] and "at least 4" in thin["reason"] and len(r["y"]) == 2


def test_duplicates_crossed_quotes_and_extreme_vols_are_rejected_and_counted(chain):
    c = _slice(chain, 3).reset_index(drop=True)
    otm = c[((c["right"] == "P") & (c["strike"] <= c["F"])) | ((c["right"] == "C") & (c["strike"] > c["F"]))].reset_index(drop=True)
    dup = otm.iloc[[0]].copy()
    dup["open_interest"] = -5.0
    bad = otm.iloc[[10]].copy()
    bad[["bid", "ask"]] = [5.0, 1.0]
    wild = otm.iloc[[20]].copy()
    wild["iv"] = 9.0
    r = vs.build(pd.concat([otm, dup, bad, wild], ignore_index=True))
    rej = r["meta"]["quotes"]["rejected"]
    assert rej["duplicate"] >= 1 and rej["crossed_or_no_bid"] >= 1 and rej["extreme_vol"] >= 1
    assert max(o["iv"] for o in r["observations"]) <= 3.0


def test_total_variance_and_volatility_grids_agree(chain):
    r = vs.build(chain)
    T = [s["T"] for s in r["slices"] if s["usable"]]
    for i, (rv, rw) in enumerate(zip(r["z"]["raw"]["vol"], r["z"]["raw"]["w"])):
        for v, w in zip(rv, rw):
            assert (v is None) == (w is None)
            if v is not None:
                assert w == pytest.approx((v / 100.0) ** 2 * T[i], rel=1e-3)


def test_the_fitted_surface_is_separate_and_exists_only_where_quotes_exist(chain):
    r = vs.build(chain)
    raw, fit = np.array(r["z"]["raw"]["w"], dtype=float), np.array(r["z"]["fit"]["w"], dtype=float)
    assert (np.isfinite(raw) == np.isfinite(fit)).all()
    assert r["fit"]["mean_rmse_vol_points"] < 2.5 and r["fit"]["calendar"]["checked_nodes"] > 0 and r["fit"]["reading"]
    assert not np.allclose(raw[np.isfinite(raw)], fit[np.isfinite(fit)])
    assert all("min_g_in_quotes" in s["fit"] for s in r["slices"] if s["usable"] and "error" not in s["fit"])


def test_the_delta_coordinate_has_no_fit_and_says_how_to_get_one(chain):
    r = vs.build(chain, coordinate="delta")
    assert r["z"]["fit"] is None and 0.0 <= r["x"][0] and r["x"][-1] <= 1.0 and any("log-moneyness" in m for m in r["limits"])


def test_call_and_put_selections_use_only_that_side(chain):
    for sel, right in (("call", "C"), ("put", "P")):
        r = vs.build(chain, selection=sel)
        assert {o["right"] for o in r["observations"]} == {right}


def test_expiry_range_filters_expiries(chain):
    r = vs.build(chain, min_dte=30, max_dte=120)
    assert r["slices"] and all(30 <= s["dte"] <= 120 for s in r["slices"])


def test_a_calendar_violation_in_the_quotes_is_found():
    k = np.linspace(-0.2, 0.2, 9)
    rows = []
    for e, T, base in (("2026-02-01", 30 / 365, 0.40), ("2026-03-01", 60 / 365, 0.20)):          # the later expiry has the LOWER volatility: total variance falls
        for ki in k:
            K = 100 * np.exp(ki)
            rows.append({"date": pd.Timestamp("2026-01-02"), "expiry": pd.Timestamp(e), "strike": K, "right": "P" if ki <= 0 else "C", "underlying": 100.0, "rate": 0.0, "dividend": 0.0, "bid": 1.0, "ask": 1.2, "mid": 1.1,
                         "iv": base + 0.3 * ki ** 2, "T": T, "F": 100.0, "open_interest": 10.0, "volume": 1.0})
    r = vs.build(pd.DataFrame(rows))
    assert r["fit"]["calendar_raw"]["violations"] > 0 and r["fit"]["calendar_raw"]["worst"] is not None


def test_the_term_structure_is_null_where_the_money_is_not_quoted(chain):
    c = _slice(chain, 8)
    e = sorted(c["expiry"].unique())[6]
    far = (c["expiry"] == e) & (np.abs(np.log(c["strike"] / c["F"])) < 0.12)
    r = vs.build(c[~far])
    row = [t for t in r["term"]["rows"] if t["expiry"] == str(pd.Timestamp(e).date())]
    assert row and row[0]["atm_iv"] is None


def test_invalid_settings_are_refused(chain):
    for kw in ({"selection": "x"}, {"coordinate": "z"}, {"nx": 3}):
        with pytest.raises(ValueError):
            vs.build(chain, **kw)


def test_a_per_expiry_svi_fit_is_never_called_globally_arbitrage_free():
    import pandas as pd
    from src.assets import options_analytics as oa
    from src.assets import vol_surface as vs

    chain = oa.prepare_chain(oa.build_model_chain(100.0, pd.Timestamp("2025-01-02"), 0.2))
    out = vs.build(chain, "otm", "k", 7, 400, 41, True)
    fit = out["fit"]
    assert fit["globally_arbitrage_free"] is False and fit["arbitrage_status"] in ("none_detected_on_tested_grid", "violations_detected") and "NOT a globally arbitrage-free" in fit["status_text"] or fit["arbitrage_status"] == "violations_detected"
    assert out["meta"]["recommended_view"] in ("surface", "smile")
