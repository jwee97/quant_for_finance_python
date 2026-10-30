"""The model families added after the first zoo: rankers and probabilistic boosters, regime models, sequence networks, forecasters, representation learners, graph networks, member ensembles and
reinforcement-learning agents.

This file is only the catalogue (names, defaults, form hints, search spaces and which module builds each kind). The builders are imported on demand, so a package that is not installed (PyTorch,
NGBoost, Stable-Baselines3) makes its models unavailable with a reason and never breaks the rest of the zoo.
"""

from __future__ import annotations

from ..ops.hpo import Categorical, Float, Int, LogFloat
from .models import ModelInfo

DEVICE = {"type": "choice", "choices": ["auto", "cpu", "cuda"], "help": "auto uses a GPU when PyTorch can see one, otherwise the CPU"}
HORIZONS = {"type": "text", "help": "Extra forecast horizons in days, comma separated, each shorter than the dataset's horizon (for example 1,5). They train extra output heads of the same network"}


def _deep(**over) -> dict:
    """Settings every PyTorch model shares (the trainer in ``ml.deep.trainer``), with a model's own on top."""
    base = {"lookback": 20, "hidden": 32, "layers": 2, "dropout": 0.1, "epochs": 25, "batch_size": 512, "learning_rate": 0.001, "weight_decay": 0.0001, "patience": 5, "grad_clip": 1.0,
            "val_fraction": 0.15, "max_rows": 60000, "device": "auto", "horizons": "", "num_threads": 0, "checkpoint_dir": "", "resume": 0}
    base.update(over)
    return base


DEEP_SPECS = {"device": DEVICE, "horizons": HORIZONS, "checkpoint_dir": {"type": "text", "help": "A folder for checkpoints (empty: none). Each refit writes best.pt and last.pt to a sub-folder of its own, named for its training window"},
              "resume": {"type": "bool", "help": "Continue an interrupted fit from its last.pt (the same training window) when it exists"}}
BOOST = {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 100, "reg_lambda": 5.0}
RL_COMMON = {"total_timesteps": 20000, "learning_rate": 0.0003, "gamma": 0.99, "batch_size": 64, "hidden": 64, "reward": "risk_adjusted", "risk_aversion": 0.5, "turnover_penalty": 0.0,
             "drawdown_penalty": 0.0, "max_weight": 0.3, "gross": 1.0, "long_only": 0, "device": "cpu", "episode_days": 252, "cost_bps": 7.0, "impact": 0.0, "max_turnover": 0.0}
RL_SPECS = {"cost_bps": {"type": "float", "help": "One-way cost per unit of turnover, commission plus slippage, in basis points. Match the Backtest section's costs (the form fills it in from them)"},
            "impact": {"type": "float", "help": "Quadratic market-impact cost: this times the sum of squared weight changes"},
            "max_turnover": {"type": "float", "help": "Largest one-way turnover per day (0: no limit); a larger trade is shortened to it"},
            "reward": {"type": "choice", "choices": ["risk_adjusted", "dsr", "return"], "help": "risk_adjusted: return less a variance and drawdown penalty; dsr: the differential Sharpe ratio of Moody and Saffell; return: the plain net return"},
            "long_only": {"type": "bool", "help": "Long-only weights that sum to the gross exposure; otherwise long-short with zero net exposure"}, "device": DEVICE}

EXTRA_INFO = [
    ModelInfo("lgbm_ranker", "LightGBM LambdaMART ranker", "ranking", ("regression", "classification"),
              "LambdaMART (LightGBM lambdarank): learns to order the assets of each day by their forward return, optimising NDCG. Each day is one query; the labels are the day's return quantile buckets. Scores are standardised with the training scores' mean and spread",
              {**BOOST, "relevance_levels": 5}, 63, "lightgbm", {}, ("lightgbm",), ("ranking",)),
    ModelInfo("xgb_ranker", "XGBoost ranker", "ranking", ("regression", "classification"),
              "XGBoost's learning-to-rank objectives (rank:ndcg, rank:pairwise, rank:map) with one query per day; scores are standardised with the training scores' mean and spread",
              {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 50, "reg_lambda": 5.0, "relevance_levels": 5, "objective": "rank:ndcg"},
              63, "xgboost", {"objective": {"type": "choice", "choices": ["rank:ndcg", "rank:pairwise", "rank:map"], "help": "The ranking loss"}}, ("xgboost",), ("ranking",)),
    ModelInfo("lgbm_quantile", "Quantile LightGBM", "probabilistic", ("regression",),
              "Three LightGBM quantile regressions (low, median, high). The median is the score; the band between the outer two is a prediction interval, and its width is the model's uncertainty",
              {**BOOST, "alpha_low": 0.1, "alpha_high": 0.9}, 63, "lightgbm", {}, ("lightgbm",), ("distribution",)),
    ModelInfo("ngboost", "NGBoost", "probabilistic", ("regression",),
              "Natural-gradient boosting of a Normal predictive distribution (NGBoost): a mean and a standard deviation per asset and day, hence intervals; trained on a random sample of at most max_rows rows",
              {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "minibatch_frac": 0.5, "col_sample": 0.5, "max_rows": 40000, "level": 0.8}, 126, None, {}, ("ngboost",), ("distribution",)),
    ModelInfo("hmm_regime", "Gaussian HMM regimes", "regime", ("regression",),
              "A Gaussian hidden Markov model on the market (equal-weight return, volatility, dispersion) finds the regimes; one ridge regression per regime, mixed by the regime probabilities FILTERED up to each day (never smoothed with later days)",
              {"n_states": 3, "alpha": 10.0, "covariance": "diag", "n_iter": 60, "min_weight": 200}, 63, None,
              {"covariance": {"type": "choice", "choices": ["diag", "full", "spherical"], "help": "Shape of each regime's covariance"}}, ("hmmlearn",), ("regimes",)),
    ModelInfo("markov_switching", "Markov-switching regression", "regime", ("regression",),
              "A Markov-switching regression (statsmodels) of the market return with regime-dependent mean and variance; one ridge per regime mixed by the filtered probabilities up to each day",
              {"n_states": 2, "alpha": 10.0, "switching_variance": 1, "min_weight": 200}, 63, None,
              {"switching_variance": {"type": "bool", "help": "Let the variance switch with the regime as well as the mean"}}, ("statsmodels",), ("regimes",)),
    ModelInfo("lstm", "LSTM", "deep sequence", ("regression",),
              "A long short-term memory network over the last `lookback` days of each asset's features (PyTorch), trained with early stopping on a chronological validation tail and gradient clipping",
              _deep(), 63, None, DEEP_SPECS, ("torch",), ("training", "heads")),
    ModelInfo("gru", "GRU", "deep sequence", ("regression",), "A gated recurrent unit network over the last `lookback` days of each asset's features (PyTorch)", _deep(), 63, None, DEEP_SPECS, ("torch",), ("training", "heads")),
    ModelInfo("tcn", "Temporal convolutional network", "deep sequence", ("regression",),
              "Stacked dilated causal convolutions with residual connections (Bai, Kolter and Koltun): a receptive field that grows with depth, no recurrence", _deep(kernel_size=3, layers=3), 63, None, DEEP_SPECS,
              ("torch",), ("training", "heads")),
    ModelInfo("patchtst", "PatchTST", "forecasting", ("regression",),
              "Patch time-series transformer (Nie et al. 2023): the window is cut into patches, each channel is embedded on its own, and a transformer encoder reads the patches",
              _deep(patch_len=4, stride=2, d_model=32, n_heads=4, ff_dim=64, channels=8, layers=2, lookback=24), 63, None, DEEP_SPECS, ("torch",), ("training", "heads")),
    ModelInfo("nbeats", "N-BEATS", "forecasting", ("regression",),
              "Neural basis expansion (Oreshkin et al. 2020): residual blocks of fully connected layers, each subtracting its backcast from the window and adding a forecast; one forecast per horizon",
              _deep(blocks=3, width=64, layers=3, lookback=24), 63, None, DEEP_SPECS, ("torch",), ("training", "heads")),
    ModelInfo("nhits", "N-HiTS", "forecasting", ("regression",),
              "N-HiTS (Challu et al. 2023): N-BEATS blocks that read the window at several resolutions (max-pooling) and forecast with interpolated coefficients",
              _deep(blocks=3, width=64, layers=2, pool_sizes="4,2,1", lookback=24), 63, None, {**DEEP_SPECS, "pool_sizes": {"type": "text", "help": "Pooling kernel of each block, coarse to fine, comma separated (the number of blocks follows)"}},
              ("torch",), ("training", "heads")),
    ModelInfo("tft", "Temporal Fusion Transformer", "forecasting", ("regression",),
              "A compact Temporal Fusion Transformer (Lim et al. 2021): gated residual networks, variable selection, an LSTM encoder, interpretable multi-head attention and quantile outputs (low, median, high)",
              _deep(hidden=32, n_heads=2, lstm_layers=1, alpha_low=0.1, alpha_high=0.9, lookback=24), 63, None, DEEP_SPECS, ("torch",), ("training", "heads", "distribution")),
    ModelInfo("autoencoder", "Autoencoder features", "representation", ("regression",),
              "An autoencoder learns a low-dimensional code of each asset's recent feature window without looking at returns; a ridge regression on the code (fitted on the training rows only) makes the score",
              _deep(latent_dim=8, ridge_alpha=10.0, lookback=10, layers=1), 63, None, DEEP_SPECS, ("torch",), ("training",)),
    ModelInfo("vae", "Variational autoencoder features", "representation", ("regression",),
              "A variational autoencoder (reconstruction plus beta times the KL divergence) learns a smooth latent code; a ridge on the latent means makes the score",
              _deep(latent_dim=8, ridge_alpha=10.0, beta=1.0, lookback=10, layers=1), 63, None, DEEP_SPECS, ("torch",), ("training",)),
    ModelInfo("gcn", "Graph convolutional network", "graph", ("regression",),
              "A graph convolutional network (Kipf and Welling) over the assets of each day: edges from return correlation (training data only) and from shared sector or class; scores come from neighbours' features as well as the asset's own",
              {"hidden": 32, "layers": 2, "dropout": 0.1, "graph": "both", "neighbors": 4, "corr_weight": 1.0, "sector_weight": 1.0, "epochs": 40, "batch_size": 32, "learning_rate": 0.003, "weight_decay": 0.0001,
               "patience": 6, "grad_clip": 1.0, "val_fraction": 0.15, "device": "auto"}, 126, None,
              {"graph": {"type": "choice", "choices": ["both", "correlation", "sector"], "help": "Which relationships make the edges"}, "device": DEVICE}, ("torch",), ("training",)),
    ModelInfo("gat", "Graph attention network", "graph", ("regression",),
              "A graph attention network (Velickovic et al.): like the GCN, but each asset learns how much attention to give each neighbour, so the weights of its neighbours depend on the day",
              {"hidden": 32, "layers": 2, "dropout": 0.1, "heads": 2, "graph": "both", "neighbors": 4, "corr_weight": 1.0, "sector_weight": 1.0, "epochs": 40, "batch_size": 32, "learning_rate": 0.003,
               "weight_decay": 0.0001, "patience": 6, "grad_clip": 1.0, "val_fraction": 0.15, "device": "auto"}, 126, None,
              {"graph": {"type": "choice", "choices": ["both", "correlation", "sector"], "help": "Which relationships make the edges"}, "device": DEVICE}, ("torch",), ("training", "attention")),
    ModelInfo("model_ensemble", "Model ensemble", "ensemble", ("regression",),
              "Several models fitted on the same rows and combined: equal weights, weights that follow each member's recent rank IC (using only outcomes already realised), or weights chosen per regime. It also returns a predictive "
              "distribution (the members' disagreement plus their own uncertainty), which uncertainty-weighted positions can use",
              {"members": "ridge,lightgbm,lgbm_quantile", "weighting": "dynamic", "val_fraction": 0.25, "decay_days": 63, "temperature": 1.0, "refit_full": 1, "regimes": 2}, 63, None,
              {"members": {"type": "text", "help": "Model kinds, comma separated, each with its default settings (for example ridge,lightgbm,lstm)"},
               "weighting": {"type": "choice", "choices": ["equal", "dynamic", "regime"], "help": "equal; dynamic: weights follow trailing rank IC; regime: the best members of each volatility regime"},
               "refit_full": {"type": "bool", "help": "Refit the members on the whole training window after the weights are set (otherwise they keep the earlier fit)"}}, (), ("distribution", "weights")),
    ModelInfo("ppo", "PPO agent", "reinforcement", ("regression",),
              "Proximal policy optimisation (Stable-Baselines3) trading the assets in a Gymnasium environment with costs, slippage, exposure and turnover limits; the policy's weights are the scores",
              {**RL_COMMON, "n_steps": 512, "ent_coef": 0.0, "n_epochs": 5, "clip_range": 0.2}, 252, None, RL_SPECS, ("gymnasium", "stable_baselines3"), ("training", "rewards")),
    ModelInfo("sac", "SAC agent", "reinforcement", ("regression",),
              "Soft actor-critic (Stable-Baselines3): an off-policy maximum-entropy agent in the same environment", {**RL_COMMON, "total_timesteps": 10000, "buffer_size": 50000, "learning_starts": 500, "tau": 0.005}, 252, None, RL_SPECS,
              ("gymnasium", "stable_baselines3"), ("training", "rewards")),
    ModelInfo("recurrent_ppo", "Recurrent PPO agent", "reinforcement", ("regression",),
              "PPO with an LSTM policy (sb3-contrib), so the agent keeps a memory of the days it has seen", {**RL_COMMON, "n_steps": 512, "ent_coef": 0.0, "n_epochs": 5, "clip_range": 0.2, "lstm_hidden": 32}, 252, None,
              RL_SPECS, ("gymnasium", "stable_baselines3", "sb3_contrib"), ("training", "rewards")),
]

EXTRA_MODULE = {"lgbm_ranker": "boosting", "xgb_ranker": "boosting", "lgbm_quantile": "boosting", "ngboost": "boosting", "hmm_regime": "regimes", "markov_switching": "regimes", "lstm": "deep.model",
                "gru": "deep.model", "tcn": "deep.model", "patchtst": "deep.model", "nbeats": "deep.model", "nhits": "deep.model", "tft": "deep.model", "autoencoder": "deep.model", "vae": "deep.model",
                "gcn": "graph", "gat": "graph", "model_ensemble": "ensemble_models", "ppo": "rl.agent", "sac": "rl.agent", "recurrent_ppo": "rl.agent"}

_BOOST_SPACE = {"num_leaves": Int(4, 31), "learning_rate": LogFloat(0.01, 0.1), "n_estimators": Int(100, 400), "reg_lambda": LogFloat(1.0, 50.0)}
_DEEP_SPACE = {"hidden": Categorical(16, 32, 64), "dropout": Float(0.0, 0.4), "learning_rate": LogFloat(0.0003, 0.01), "weight_decay": LogFloat(1e-6, 1e-2), "lookback": Categorical(10, 20, 40)}
EXTRA_SPACES = {
    "lgbm_ranker": {**_BOOST_SPACE, "relevance_levels": Int(3, 8)},
    "xgb_ranker": {"max_depth": Int(2, 5), "learning_rate": LogFloat(0.01, 0.1), "n_estimators": Int(100, 400), "reg_lambda": LogFloat(1.0, 50.0), "relevance_levels": Int(3, 8)},
    "lgbm_quantile": _BOOST_SPACE, "ngboost": {"n_estimators": Int(60, 300), "learning_rate": LogFloat(0.01, 0.1), "max_depth": Int(2, 5)},
    "hmm_regime": {"n_states": Int(2, 4), "alpha": LogFloat(0.1, 1000.0)}, "markov_switching": {"alpha": LogFloat(0.1, 1000.0)},
    "lstm": _DEEP_SPACE, "gru": _DEEP_SPACE, "tcn": {**_DEEP_SPACE, "kernel_size": Categorical(2, 3, 5)},
    "patchtst": {"d_model": Categorical(16, 32, 64), "dropout": Float(0.0, 0.4), "learning_rate": LogFloat(0.0003, 0.01), "patch_len": Categorical(4, 6, 8)},
    "nbeats": {"width": Categorical(32, 64, 128), "blocks": Int(2, 4), "learning_rate": LogFloat(0.0003, 0.01), "dropout": Float(0.0, 0.4)},
    "nhits": {"width": Categorical(32, 64, 128), "blocks": Int(2, 4), "learning_rate": LogFloat(0.0003, 0.01), "dropout": Float(0.0, 0.4)},
    "tft": {"hidden": Categorical(16, 32, 64), "dropout": Float(0.0, 0.4), "learning_rate": LogFloat(0.0003, 0.01), "n_heads": Categorical(1, 2, 4)},
    "autoencoder": {"latent_dim": Int(4, 16), "ridge_alpha": LogFloat(0.1, 1000.0), "learning_rate": LogFloat(0.0003, 0.01)},
    "vae": {"latent_dim": Int(4, 16), "ridge_alpha": LogFloat(0.1, 1000.0), "beta": LogFloat(0.1, 4.0), "learning_rate": LogFloat(0.0003, 0.01)},
    "gcn": {"hidden": Categorical(16, 32, 64), "dropout": Float(0.0, 0.4), "learning_rate": LogFloat(0.0003, 0.01), "neighbors": Int(2, 6)},
    "gat": {"hidden": Categorical(16, 32, 64), "dropout": Float(0.0, 0.4), "learning_rate": LogFloat(0.0003, 0.01), "neighbors": Int(2, 6), "heads": Categorical(1, 2, 4)},
    "model_ensemble": {"val_fraction": Float(0.15, 0.4), "decay_days": Int(21, 252), "temperature": LogFloat(0.25, 5.0)},
    "ppo": {"learning_rate": LogFloat(0.0001, 0.003), "gamma": Float(0.9, 0.999), "risk_aversion": Float(0.0, 2.0), "ent_coef": Float(0.0, 0.02)},
    "sac": {"learning_rate": LogFloat(0.0001, 0.003), "gamma": Float(0.9, 0.999), "risk_aversion": Float(0.0, 2.0)},
    "recurrent_ppo": {"learning_rate": LogFloat(0.0001, 0.003), "gamma": Float(0.9, 0.999), "risk_aversion": Float(0.0, 2.0)},
}
