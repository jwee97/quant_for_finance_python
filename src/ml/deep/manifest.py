"""What each network IS, stated once and checked by tests (P0-08).

Every entry says which published architecture the module follows, what it simplifies, the axes of its input and output, how it treats absent values, and how forecast horizons map onto outputs. A
simplification is named here; it is not left to a docstring nobody reads. ``reduced`` is True where the module is a deliberate reduction of the published architecture and not a port of it.
Nothing here imports PyTorch.
"""

from __future__ import annotations

MANIFEST_VERSION = 1
_INPUT = "[batch, lookback, features]: the asset's own window of standardised features ending on the prediction day"
_OUTPUT = "[batch, heads]: one score per forecast horizon, the dataset's own horizon last"

ARCHITECTURES = {
    "lstm": {"variant": "stacked LSTM, last hidden state, normalised linear head", "reference": "Hochreiter and Schmidhuber (1997)", "reduced": False,
             "input_axes": _INPUT, "output_axes": _OUTPUT, "masking": "a missing feature is filled by the training-only scaler before the window is built", "normalization": "LayerNorm on the last hidden state",
             "residuals": "none", "attention": "none", "horizon_mapping": "one linear output per horizon, trained jointly with the dataset's horizon last"},
    "gru": {"variant": "stacked GRU, last hidden state, normalised linear head", "reference": "Cho et al. (2014)", "reduced": False, "input_axes": _INPUT, "output_axes": _OUTPUT,
            "masking": "as lstm", "normalization": "LayerNorm on the last hidden state", "residuals": "none", "attention": "none", "horizon_mapping": "as lstm"},
    "tcn": {"variant": "causal dilated residual convolutions, output at the last step", "reference": "Bai, Kolter and Koltun (2018)", "reduced": False, "input_axes": _INPUT, "output_axes": _OUTPUT,
            "masking": "left padding only: a convolution output at step t sees steps <= t", "normalization": "none beyond dropout", "residuals": "one skip per block",
            "attention": "none", "horizon_mapping": "as lstm"},
    "patchtst": {"variant": "REDUCED PatchTST: a learned channel reduction, patching, a pre-norm transformer encoder per channel, mean pooling over patch tokens, a linear mix of channels",
                 "reference": "Nie et al. (2023)", "reduced": True, "input_axes": _INPUT, "output_axes": _OUTPUT, "masking": "none inside the window; no padding mask is needed because windows are full length",
                 "normalization": "none (the paper's reversible instance normalisation is omitted: features are already standardised on the training rows)", "residuals": "inside the encoder layers",
                 "attention": "full self-attention among the patches of one channel, channels independent after the reduction", "horizon_mapping": "as lstm",
                 "differences": ["a linear layer reduces the features to `channels` channels (the paper keeps every feature as a channel)", "patch tokens are mean pooled and summarised, not flattened into a head",
                                 "no reversible instance normalisation"]},
    "nbeats": {"variant": "REDUCED N-BEATS: the generic architecture over the flattened multivariate window", "reference": "Oreshkin et al. (2020)", "reduced": True, "input_axes": _INPUT + ", flattened",
               "output_axes": _OUTPUT, "masking": "none", "normalization": "none", "residuals": "doubly residual: each block's backcast is subtracted from its input and its forecast added to the output",
               "attention": "none", "horizon_mapping": "each block's forecast has one coefficient per horizon; no trend or seasonality stack",
               "differences": ["multivariate: the paper's series are univariate", "the generic stack only (no interpretable trend/seasonality basis)"]},
    "nhits": {"variant": "REDUCED N-HiTS: multi-rate max-pooled input, coarse backcast interpolated to the window, per-horizon forecast coefficients", "reference": "Challu et al. (2023)", "reduced": True,
              "input_axes": _INPUT + ", transposed to [batch, features, time] for pooling", "output_axes": _OUTPUT, "masking": "none; ceil-mode pooling keeps a partial last pool",
              "normalization": "none", "residuals": "each block's interpolated backcast is subtracted from the residual window; forecasts are summed", "attention": "none",
              "horizon_mapping": "one coefficient per horizon per block, with no hierarchical interpolation of the forecast (there are only a handful of horizons)",
              "differences": ["no hierarchical interpolation of the forecast", "multivariate window"]},
    "tft": {"variant": "COMPACT Temporal Fusion Transformer for a panel without static inputs", "reference": "Lim et al. (2021)", "reduced": True, "input_axes": _INPUT,
            "output_axes": "[batch, 3 * heads]: quantile i of horizon h is output 3h + i; the score is the median of the last horizon", "masking": "none inside the window",
            "normalization": "layer normalisation inside the gated residual networks", "residuals": "gated skip connections after the LSTM and the attention",
            "attention": "standard multi-head attention from the last step over the window (not the paper's interpretable variant)", "horizon_mapping": "three quantiles per horizon",
            "differences": ["no static covariates", "no known-future inputs", "standard multi-head attention instead of the interpretable variant", "one LSTM encoder, no decoder"]},
    "autoencoder": {"variant": "fully connected autoencoder over the flattened window; the code feeds a ridge read-out", "reference": "Hinton and Salakhutdinov (2006)", "reduced": False,
                    "input_axes": _INPUT + ", flattened", "output_axes": "[batch, latent_dim] code; the score is a ridge read-out of the code", "masking": "missing inputs are masked out of the reconstruction loss",
                    "normalization": "none", "residuals": "none", "attention": "none", "horizon_mapping": "the read-out is fitted to the dataset's horizon only"},
    "vae": {"variant": "variational autoencoder over the flattened window, reparameterised code, beta-weighted KL", "reference": "Kingma and Welling (2014)", "reduced": False,
            "input_axes": _INPUT + ", flattened", "output_axes": "[batch, latent_dim] posterior mean; the score is a ridge read-out of it", "masking": "as autoencoder", "normalization": "none",
            "residuals": "none", "attention": "none", "horizon_mapping": "as autoencoder"},
}


def architecture(kind: str) -> dict:
    """The manifest entry of a network kind, with the manifest version."""
    if kind not in ARCHITECTURES:
        raise KeyError(f"no architecture manifest for '{kind}'")
    return {"manifest_version": MANIFEST_VERSION, "kind": kind, **ARCHITECTURES[kind]}
