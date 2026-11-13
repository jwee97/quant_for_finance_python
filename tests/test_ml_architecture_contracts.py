"""P0-08: every network matches its manifest, and the architecture-specific properties (patching, residual decomposition, multi-rate pooling, causal convolution, quantile layout) hold."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml import models as md                                                      # noqa: E402
from src.ml.deep.manifest import ARCHITECTURES, architecture                         # noqa: E402
from src.ml.deep.nets import FORECASTERS, build_net                                  # noqa: E402

L, F_, OUT = 12, 5, 2
CFG = {"hidden": 8, "layers": 2, "d_model": 8, "n_heads": 2, "ff_dim": 16, "patch_len": 4, "stride": 2, "channels": 4, "blocks": 2, "width": 16, "dropout": 0.0, "latent_dim": 4}


def net(arch, **over):
    torch.manual_seed(0)
    return build_net(arch, F_, L, OUT, {**CFG, **over}).eval()


def test_the_manifest_covers_every_network_and_labels_the_reduced_ones():
    assert set(FORECASTERS) <= set(ARCHITECTURES) and {"autoencoder", "vae"} <= set(ARCHITECTURES)
    assert {k for k, v in ARCHITECTURES.items() if v["reduced"]} == {"patchtst", "nbeats", "nhits", "tft", "itransformer"}
    for kind, entry in ARCHITECTURES.items():
        assert {"variant", "reference", "input_axes", "output_axes", "masking", "normalization", "residuals", "attention", "horizon_mapping"} <= set(entry), kind
        assert (not entry["reduced"]) or entry.get("differences"), f"{kind} is a reduction and must say what it leaves out"
        assert (entry["variant"].startswith(("REDUCED", "COMPACT"))) == entry["reduced"], f"{kind}: the variant name must say so when it is reduced"
    assert architecture("tft")["manifest_version"] == 1
    with pytest.raises(KeyError):
        architecture("transformer-xl")


@pytest.mark.parametrize("arch", list(FORECASTERS))
def test_each_forecaster_maps_a_batch_of_windows_to_the_declared_output_axes(arch):
    m = net(arch)
    y = m(torch.randn(6, L, F_))
    assert y.shape == (6, OUT * 3 if arch == "tft" else OUT) and torch.isfinite(y).all(), "one score per horizon; three quantiles per horizon for the tft"
    assert md._INFO[arch].family in ("deep sequence", "forecasting")


def test_patchtst_cuts_the_window_into_overlapping_patches_of_the_stated_count_and_content():
    m = net("patchtst")
    seen = {}
    m.encoder.register_forward_hook(lambda mod, inp, out: seen.update(shape=tuple(inp[0].shape)))
    x = torch.randn(3, L, F_)
    m(x)
    n_patches = (L - 4) // 2 + 1
    assert seen["shape"] == (3 * 4, n_patches, 8), "channels fold into the batch axis; patches along the sequence axis"
    series = torch.arange(L, dtype=torch.float32).repeat(1, 1).T.repeat(1, 4)[None]       # a window whose channel is its own time index
    patches = series[0, :, 0].unfold(0, 4, 2)
    assert [list(p.int()) for p in patches[:2]] == [[0, 1, 2, 3], [2, 3, 4, 5]] and len(patches) == n_patches


def test_patchtst_channels_are_independent_after_the_reduction():
    m = net("patchtst", channels=0)                                                      # every feature its own channel, as in the paper
    x = torch.randn(2, L, F_)
    base = m.encoder(m.embed(x[:, :, 0].unfold(1, 4, 2)) + m.pos)
    x2 = x.clone()
    x2[:, :, 3] += 5.0                                                                   # another channel changes
    assert torch.allclose(base, m.encoder(m.embed(x2[:, :, 0].unfold(1, 4, 2)) + m.pos)), "channel 0's encoding must not see channel 3"


def test_nbeats_with_no_backcast_sums_independent_block_forecasts_and_with_one_subtracts_it_from_the_next_input():
    m = net("nbeats")
    x = torch.randn(4, L, F_)
    flat = x.flatten(1)
    h1, h2 = m.body[0](flat), None
    with torch.no_grad():
        for b in m.back:
            b.weight.zero_(), b.bias.zero_()
        no_back = m(x)
        by_hand = sum(f(body(flat)) for body, f in zip(m.body, m.fore))
        assert torch.allclose(no_back, by_hand, atol=1e-6), "with a zero backcast every block reads the same window"
        torch.nn.init.normal_(m.back[0].weight, std=0.3)
        h1 = m.body[0](flat)
        second_input = flat - m.back[0](h1)
        h2 = m.body[1](second_input)
        want = m.fore[0](h1) + m.fore[1](h2)                                              # unrolled by hand for two blocks
        assert torch.allclose(m(x), want, atol=1e-6)


def test_nhits_pools_at_each_blocks_own_rate_and_subtracts_a_backcast_interpolated_to_the_full_window():
    m = net("nhits", blocks=3)
    assert m.pools == [4, 2, 1] and m.knots == [3, 6, 12] and m.length == L
    x = torch.randn(4, L, F_)
    with torch.no_grad():
        for b in m.back:
            b.weight.zero_(), b.bias.zero_()
        r = x.transpose(1, 2)
        want = sum(f(body(torch.nn.functional.max_pool1d(r, k, k, ceil_mode=True).flatten(1) if k > 1 else r.flatten(1))) for k, body, f in zip(m.pools, m.body, m.fore))
        assert torch.allclose(m(x), want, atol=1e-6)
        torch.nn.init.normal_(m.back[0].weight, std=0.3)
        pooled = torch.nn.functional.max_pool1d(r, 4, 4, ceil_mode=True)
        h0 = m.body[0](pooled.flatten(1))
        coarse = m.back[0](h0).view(4, F_, 3)
        interpolated = torch.nn.functional.interpolate(coarse, size=L, mode="linear", align_corners=False)
        assert interpolated.shape == (4, F_, L), "the coarse backcast covers the whole window"
        r2 = r - interpolated
        h1 = m.body[1](torch.nn.functional.max_pool1d(r2, 2, 2, ceil_mode=True).flatten(1))
        h2 = m.body[2](r2.flatten(1))
        assert torch.allclose(m(x), m.fore[0](h0) + m.fore[1](h1) + m.fore[2](h2), atol=1e-6)


def test_a_tcn_output_at_a_step_does_not_see_later_steps_and_the_receptive_field_is_as_stated():
    m = net("tcn", layers=2, kernel_size=3)
    blocks = [b for b in m.modules() if b.__class__.__name__ == "_CausalBlock"]
    assert len(blocks) == 2
    x = torch.randn(2, L, F_)
    full = m.net(x.transpose(1, 2)) if hasattr(m, "net") else None
    if full is not None:
        x2 = x.clone()
        x2[:, 7:] = torch.randn(2, L - 7, F_)                                             # a different future from step 7 on
        assert torch.allclose(full[:, :, :7], m.net(x2.transpose(1, 2))[:, :, :7], atol=1e-6)
    assert 1 + 2 * (3 - 1) * (2 ** 2 - 1) == 13, "the receptive field the docstring states for two layers and kernel 3"


def test_the_tft_lays_out_three_quantiles_per_horizon_and_keeps_normalised_variable_and_attention_weights():
    m = net("tft")
    out = m(torch.randn(5, L, F_))
    assert out.shape == (5, 3 * OUT), "quantile i of horizon h is output 3h + i"
    assert m.last_variables.shape == (F_,) and m.last_attention.shape == (L,)
    assert float(m.last_variables.sum()) == pytest.approx(1.0, abs=1e-5) and float(m.last_attention.sum()) == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------------------------------------------------------ LSTM and GRU against their equations, written out in numpy
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _recurrent_layer(kind, x, w_ih, w_hh, b_ih, b_hh):
    """One layer of the textbook equations in PyTorch's parametrisation (gate order i, f, g, o for the LSTM and r, z, n for the GRU), for a batch of windows, zero initial state."""
    batch, steps, _ = x.shape
    h, c, out = np.zeros((batch, w_hh.shape[1])), np.zeros((batch, w_hh.shape[1])), []
    for t in range(steps):
        gi, gh = x[:, t] @ w_ih.T + b_ih, h @ w_hh.T + b_hh
        if kind == "lstm":
            i, f, g, o = np.split(gi + gh, 4, axis=1)
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
            h = _sigmoid(o) * np.tanh(c)
        else:
            i_r, i_z, i_n = np.split(gi, 3, axis=1)
            h_r, h_z, h_n = np.split(gh, 3, axis=1)
            r, z = _sigmoid(i_r + h_r), _sigmoid(i_z + h_z)
            h = (1.0 - z) * np.tanh(i_n + r * h_n) + z * h                                  # the reset gate multiplies the hidden-to-hidden part of the candidate, as PyTorch defines it
        out.append(h)
    return np.stack(out, axis=1)


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_lstm_and_gru_match_an_independent_numpy_implementation_of_their_equations(kind):
    torch.manual_seed(1)
    model = build_net(kind, F_, L, OUT, {**CFG, "hidden": 6, "layers": 2}).eval()
    x = torch.randn(4, L, F_)
    want = model(x).detach().numpy()
    seq = x.numpy().astype(np.float64)
    for k in range(2):
        p = {n: getattr(model.rnn, f"{n}_l{k}").detach().numpy().astype(np.float64) for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")}
        seq = _recurrent_layer(kind, seq, p["weight_ih"], p["weight_hh"], p["bias_ih"], p["bias_hh"])
    last = seq[:, -1]
    norm, linear = model.head[0], model.head[2]
    z = (last - last.mean(axis=1, keepdims=True)) / np.sqrt(last.var(axis=1, keepdims=True) + norm.eps) * norm.weight.detach().numpy() + norm.bias.detach().numpy()
    got = z @ linear.weight.detach().numpy().T + linear.bias.detach().numpy()
    np.testing.assert_allclose(got, want, atol=2e-5, err_msg=f"the {kind.upper()} network is not the {kind.upper()} equations")
    # the window is read in time order: reversing it changes the answer, and the first step does not see the last
    assert not np.allclose(model(torch.flip(x, dims=[1])).detach().numpy(), want, atol=1e-4)
