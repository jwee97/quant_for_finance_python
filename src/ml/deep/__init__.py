"""Reusable PyTorch infrastructure for the sequence, forecasting, representation and graph models.

``data``     feature windows gathered from a (date, asset) panel without look-ahead, and the training-only scaler;
``nets``     the networks (LSTM, GRU, TCN, PatchTST, N-BEATS, N-HiTS, a compact Temporal Fusion Transformer, autoencoders);
``trainer``  device choice, early stopping on a chronological tail, checkpoints, gradient clipping, history and resource facts;
``model``    the ``Model`` wrappers that make every network a model of the platform's zoo, with serialization.

Nothing in this package imports PyTorch until a network is built, so the rest of the platform loads without it.
"""
