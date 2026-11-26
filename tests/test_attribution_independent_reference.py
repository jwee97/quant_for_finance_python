"""Attribution against references that share no code with it: the 1985 Brinson-Hood-Beebower decomposition written out again here, and Carino linking recomputed with its published formula on a multi-period example.

What these add to ``test_attribution_exact.py`` (one period, worked by hand): a different decomposition of the same active return that must agree on what they share, many periods, and the failure modes that matter
(a residual that must stay visible, an unclassified asset that must not vanish).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.attribution import brinson_fachler, carino_factor, cumulative, link_effects

COLS = ["A1", "A2", "B1", "B2", "C1"]
SECTOR = {"A1": "S1", "A2": "S1", "B1": "S2", "B2": "S2", "C1": "S3"}


def world(n=12, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-31", periods=n, freq="ME")
    wp = pd.DataFrame(rng.dirichlet(np.ones(5), n), index=idx, columns=COLS)
    wb = pd.DataFrame(rng.dirichlet(np.ones(5) * 2, n), index=idx, columns=COLS)
    r = pd.DataFrame(rng.normal(0.01, 0.04, (n, 5)), index=idx, columns=COLS)
    return wp, wb, r


def bhb_reference(wp, wb, r):
    """Brinson-Hood-Beebower 1985, per period and sector: allocation (wp-wb) Rb_s, selection wb (Rp_s - Rb_s), interaction (wp-wb)(Rp_s - Rb_s); written without reference to the code under test."""
    out = {}
    for s in sorted(set(SECTOR.values())):
        m = [c for c in COLS if SECTOR[c] == s]
        w_p, w_b = wp[m].sum(axis=1), wb[m].sum(axis=1)
        rp_s = (wp[m] * r[m]).sum(axis=1) / w_p
        rb_s = (wb[m] * r[m]).sum(axis=1) / w_b
        out[s] = {"alloc": (w_p - w_b) * rb_s, "sel": w_b * (rp_s - rb_s), "inter": (w_p - w_b) * (rp_s - rb_s)}
    return out


def test_brinson_fachler_and_the_1985_decomposition_agree_on_selection_and_interaction_and_on_total_allocation():
    wp, wb, r = world()
    bf = brinson_fachler(wp, wb, r, SECTOR)
    ref = bhb_reference(wp, wb, r)
    for s in ref:
        assert np.allclose(bf.selection[s], ref[s]["sel"], atol=1e-14) and np.allclose(bf.interaction[s], ref[s]["inter"], atol=1e-14)
    # the two differ by what they say about each sector's allocation, but both sum, over sectors, to the same allocation because the active sector weights sum to zero
    bhb_alloc = sum(ref[s]["alloc"] for s in ref)
    assert np.allclose(bf.allocation.sum(axis=1), bhb_alloc, atol=1e-14)
    active = (wp * r).sum(axis=1) - (wb * r).sum(axis=1)
    assert np.allclose(bf.total.sum(axis=1), active, atol=1e-14), "the three effects add to the active return in every period"


def carino_reference(p, b):
    """Carino (1999): k_t = (ln(1+Rp_t) - ln(1+Rb_t)) / (Rp_t - Rb_t), K = (ln(1+Rp) - ln(1+Rb)) / (Rp - Rb) over the whole period, and each period's effect is scaled by k_t / K."""
    k = np.array([(np.log(1 + x) - np.log(1 + y)) / (x - y) if abs(x - y) > 1e-12 else 1 / (1 + x) for x, y in zip(p, b)])
    Rp, Rb = np.prod(1 + p) - 1, np.prod(1 + b) - 1
    return k / ((np.log(1 + Rp) - np.log(1 + Rb)) / (Rp - Rb))


def test_carino_linked_effects_add_to_the_cumulative_active_return_and_the_plain_sum_does_not():
    wp, wb, r = world(24, seed=3)
    bf = brinson_fachler(wp, wb, r, SECTOR)
    p, b = bf.portfolio_return.to_numpy(), bf.benchmark_return.to_numpy()
    effects = pd.DataFrame({"allocation": bf.allocation.sum(axis=1), "selection": bf.selection.sum(axis=1), "interaction": bf.interaction.sum(axis=1)})
    linked = link_effects(effects, bf.portfolio_return, bf.benchmark_return)
    scale = carino_reference(p, b)
    want = (effects.to_numpy() * scale[:, None]).sum(axis=0)
    assert np.allclose(linked.to_numpy(), want, atol=1e-14)
    cum_active = cumulative(p) - cumulative(b)
    assert linked.sum() == pytest.approx(cum_active, abs=1e-12)
    assert abs(effects.to_numpy().sum() - cum_active) > 1e-4, "adding the periods' effects without linking misses the compounding"


def test_a_book_identical_to_its_benchmark_has_no_effects_and_the_limit_of_the_linking_factor_is_used():
    wp, wb, r = world()
    bf = brinson_fachler(wb, wb, r, SECTOR)
    assert np.allclose(bf.total.to_numpy(), 0.0, atol=1e-15)
    k = carino_factor(bf.portfolio_return.to_numpy(), bf.benchmark_return.to_numpy())
    assert np.allclose(k, 1.0 / (1.0 + bf.portfolio_return.to_numpy()))


def test_weights_that_do_not_sum_to_one_are_refused_rather_than_absorbed_into_a_residual():
    wp, wb, r = world()
    with pytest.raises(ValueError, match="sum to one"):
        brinson_fachler(wp * 0.9, wb, r, SECTOR)


def test_an_asset_with_no_sector_is_an_error_not_a_silent_omission():
    wp, wb, r = world()
    partial = {k: v for k, v in SECTOR.items() if k != "C1"}
    with pytest.raises(KeyError):
        brinson_fachler(wp, wb, r, partial)
