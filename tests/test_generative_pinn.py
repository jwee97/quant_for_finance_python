"""Neural scenario generators (WGAN-GP, factor VAE), the classical network families on the return window, and physics-informed networks for pricing equations. Torch only, except the Fourier references."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.derivatives.pricing import bsm_price, heston_price
from src.models import pinn
from src.scenarios import generate, quality_report

torch = pytest.importorskip("torch")


# ------------------------------------------------------------------------------------------------------------------ references
def test_bates_without_jumps_is_heston_and_jumps_move_the_price():
    args = (100, 100, 1.0, 0.03, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7)
    assert pinn.bates_price(*args, 0.0, -0.1, 0.15) == pytest.approx(heston_price(*args), abs=1e-6)
    with_jumps = pinn.bates_price(*args, 0.5, -0.1, 0.15)
    assert abs(with_jumps - heston_price(*args)) > 0.05
    otm = pinn.bates_price(100, 80, 0.5, 0.03, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.5, -0.1, 0.15)
    assert otm > heston_price(100, 80, 0.5, 0.03, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7)                       # negative jumps fatten the left tail: dearer OTM puts' call-side parity partner


def test_vasicek_bond_formula_limits():
    assert pinn.vasicek_bond(0.03, 0.0, 0.8, 0.05, 0.02) == pytest.approx(1.0)
    assert pinn.vasicek_bond(0.03, 5.0, 0.8, 0.05, 0.0) == pytest.approx(np.exp(-(0.05 * 5 + (0.03 - 0.05) * (1 - np.exp(-4.0)) / 0.8)))     # no volatility: deterministic mean reversion
    assert pinn.vasicek_bond(0.10, 3.0, 0.8, 0.05, 0.02) < pinn.vasicek_bond(0.02, 3.0, 0.8, 0.05, 0.02)


# ------------------------------------------------------------------------------------------------------------------ PINNs
@pytest.fixture(scope="module")
def bsm_net():
    return pinn.BSMPINN(0.03, 0.01, 0.25, 1.0, steps=1200, lbfgs_steps=300).fit()


def test_bsm_pinn_prices_a_call_within_half_a_percent_of_the_strike(bsm_net):
    S = np.array([80, 90, 100, 110, 120.0])
    for tau in (0.5, 1.0):
        ref = np.array([bsm_price(s, 100, tau, 0.03, 0.01, 0.25) for s in S])
        assert np.abs(bsm_net.price(S, 100, tau) - ref).max() < 0.7
    assert bsm_net.history[-1] < bsm_net.history[0]


def test_bsm_pinn_price_is_increasing_convex_and_above_intrinsic_in_the_middle(bsm_net):
    S = np.linspace(85, 115, 13)
    p = bsm_net.price(S, 100, 1.0)
    assert (np.diff(p) > 0).all() and (np.diff(p, 2) > -0.05).all()
    assert (p >= np.maximum(S - 100 * np.exp(-0.03), 0) - 0.8).all()
    assert bsm_net.price(100.0, 200.0, 1.0) == pytest.approx(2 * bsm_net.price(50.0, 100.0, 1.0), rel=1e-9)   # homogeneity: the price is K times a function of S/K


def test_vasicek_pinn_matches_the_affine_bond_price():
    net = pinn.VasicekPINN(0.8, 0.05, 0.02, maturity=5.0, steps=1500).fit()
    r = np.array([0.0, 0.03, 0.05, 0.08, 0.10])
    assert np.abs(net.price(r, 3.0) - pinn.vasicek_bond(r, 3.0, 0.8, 0.05, 0.02)).max() < 0.005
    assert (np.diff(net.price(r, 3.0)) < 0).all()                                                       # bonds fall when rates rise


@pytest.mark.parametrize("jumps", [False, True])
def test_heston_and_bates_pinns_are_close_to_the_fourier_price(jumps):
    kw = dict(lam=0.5, mu_j=-0.1, sigma_j=0.15) if jumps else {}
    net = pinn.HestonPINN(0.03, 0.0, 2.0, 0.04, 0.3, -0.7, steps=900, lbfgs_steps=150, **kw).fit()
    S, v = np.array([90.0, 100.0, 110.0]), np.array([0.03, 0.04, 0.05])
    if jumps:
        ref = [pinn.bates_price(s, 100, 1.0, 0.03, 0.0, vv, 2.0, 0.04, 0.3, -0.7, 0.5, -0.1, 0.15) for s, vv in zip(S, v)]
    else:
        ref = [heston_price(s, 100, 1.0, 0.03, 0.0, vv, 2.0, 0.04, 0.3, -0.7) for s, vv in zip(S, v)]
    assert np.abs(net.price(S, 100, 1.0, v) - np.array(ref)).max() < 1.5                              # 1.5% of the strike: a few thousand steps do not do better
    assert net.price(100.0, 100.0, 1.0, 0.09) > net.price(100.0, 100.0, 1.0, 0.01)                    # more variance, dearer option


# ------------------------------------------------------------------------------------------------------------------ generative models
def factor_data(seed=0, n=2500, N=6):
    rng = np.random.default_rng(seed)
    f = rng.standard_t(5, size=(n, 2)) * 0.01
    B = rng.normal(size=(N, 2))
    return pd.DataFrame(f @ B.T + rng.standard_t(4, size=(n, N)) * 0.006, columns=list("ABCDEF")[:N])


def test_factor_vae_recovers_dependence_and_has_fat_tails():
    from src.models.generative import FactorVAE

    df = factor_data()
    vae = FactorVAE(factors=2, epochs=80, seed=0).fit(df.to_numpy())
    S = vae.sample(4000, seed=1)
    assert S.shape == (4000, 6) and np.allclose(vae.sample(10, 3), vae.sample(10, 3))
    assert np.abs(np.corrcoef(S.T) - np.corrcoef(df.to_numpy().T)).max() < 0.12
    assert np.abs(S.std(axis=0) / df.std().to_numpy() - 1).max() < 0.2
    assert vae.degrees_of_freedom > 2 and vae.loadings().shape == (6, 2) and vae.losses[-1] < vae.losses[0]
    q = quality_report(df, S.reshape(4000, 1, 6))
    assert q["kurtosis_ratio"] > 0.4                                                                    # fat tails, if thinner than the data's


def test_wgan_gp_learns_marginals_and_correlations_roughly():
    from src.models.generative import WGANGP

    df = factor_data(1, 2000, 4)
    gan = WGANGP(steps=1500, hidden=128, seed=0).fit(df.to_numpy())
    S = gan.sample(3000, seed=2)
    assert S.shape == (3000, 4) and np.isfinite(S).all()
    assert np.abs(S.std(axis=0) / df.std().to_numpy() - 1).max() < 0.4
    assert np.abs(np.corrcoef(S.T) - np.corrcoef(df.to_numpy().T)).max() < 0.4
    uninformed = np.random.default_rng(0).normal(size=(3000, 4)) * df.std().to_numpy()
    assert np.abs(np.corrcoef(S.T) - np.corrcoef(df.to_numpy().T)).max() < np.abs(np.corrcoef(uninformed.T) - np.corrcoef(df.to_numpy().T)).max()   # better than ignoring dependence


@pytest.mark.parametrize("name,kw", [("factor_vae", {"factors": 2, "epochs": 20}), ("wgan_gp", {"steps": 60, "hidden": 32}), ("diffusion", {"epochs": 15, "hidden": 32})])
def test_learned_generators_run_through_the_common_interface_with_paths(name, kw):
    df = factor_data(2, 600, 3)
    S = generate(name, df, 20, 7, seed=0, window=3, **kw)
    assert S.shape == (20, 7, 3) and np.isfinite(S).all()


# ------------------------------------------------------------------------------------------------------------------ classical networks
@pytest.mark.parametrize("kind", ["fnn", "cnn", "lstm", "gru", "transformer"])
def test_classical_networks_map_a_window_to_one_forecast_per_row(kind):
    from src.models.deep_forecast import build_network, parameter_count

    net = build_network(kind, 5)
    x = torch.randn(7, 252)
    a = torch.randint(0, 5, (7,))
    y = net(x, a)
    assert y.shape == (7,) and torch.isfinite(y).all() and parameter_count(kind, 5) > 100
    y.sum().backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)


def test_networks_depend_on_the_asset_and_reject_a_stride_that_does_not_divide_the_window():
    from src.models.deep_forecast import build_network

    net = build_network("lstm", 4).eval()
    x = torch.randn(1, 252).repeat(2, 1)
    assert not torch.isclose(net(x, torch.tensor([0, 1]))[0], net(x, torch.tensor([0, 1]))[1])
    with pytest.raises(ValueError):
        build_network("lstm", 4, family={"stride": 5}).eval()(x, torch.tensor([0, 1]))
