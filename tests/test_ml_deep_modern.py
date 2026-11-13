"""The 2023-2024 additions to the networks: iTransformer and DLinear, and the losses that compare the assets of one day (daily correlation, pairwise ranking) with batches made of whole days."""

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy import stats                                                        # noqa: E402

from src.ml import config as mc, datasets as dsm, models as md                 # noqa: E402
from src.ml.deep import losses as ls, trainer as tn                            # noqa: E402
from src.ml.deep.nets import DLinear, ITransformer, build_net                  # noqa: E402
from src.ml.param_schema import ParamError, effective_params                   # noqa: E402
from tests.ml_helpers import make_bundle                                       # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
TINY = {"epochs": 3, "max_rows": 1500, "batch_size": 128, "lookback": 8, "patience": 2}
SPECIFIC = {"itransformer": {"d_model": 8, "n_heads": 2, "ff_dim": 16, "layers": 1}, "dlinear": {"kernel_size": 3}, "lstm": {"hidden": 8}}


@pytest.fixture(scope="module")
def ds():
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 400, "refit_every": 250}}, defaults=mc.DEFAULTS)
    return dsm.build_dataset(make_bundle(8, 1100, seed=31, alpha=1.0), cfg["dataset"], "deep_modern")


def fit_on(ds, kind, **over):
    pos = ds.positions()
    rows = np.flatnonzero((pos >= 80) & (pos < 700) & ds.Y[ds.target].notna().to_numpy())
    model = md.make_model(kind, {**TINY, **SPECIFIC.get(kind, {}), **over}, "regression", 3).bind(ds)
    return model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos[rows]), rows


# ------------------------------------------------------------------------------------------------------------ the networks
def test_itransformer_makes_one_token_per_feature_and_attends_among_features_only():
    torch.manual_seed(0)
    net = ITransformer(n_features=5, lookback=12, n_out=2, d_model=8, n_heads=2, ff_dim=16, layers=1, dropout=0.0).eval()
    x = torch.randn(4, 12, 5)
    assert net(x).shape == (4, 2) and net.embed.in_features == 12 and net.identity.shape == (1, 5, 8)
    seen = {}
    net.encoder.register_forward_hook(lambda m, i, o: seen.update(tokens=i[0].shape))
    net(x)
    assert tuple(seen["tokens"]) == (4, 5, 8)                                           # the sequence the encoder reads is the FEATURES, whatever the lookback
    longer = ITransformer(n_features=5, lookback=48, n_out=2, d_model=8, n_heads=2, ff_dim=16, layers=1)
    count = lambda m: sum(p.numel() for n, p in m.named_parameters() if not n.startswith("embed"))      # noqa: E731
    assert count(longer) == count(net)                                                  # a longer window only widens the embedding


def test_itransformer_refuses_a_width_the_heads_do_not_divide():
    with pytest.raises(ValueError, match="multiple of n_heads"):
        build_net("itransformer", 5, 12, 1, {"d_model": 10, "n_heads": 4})
    with pytest.raises(ParamError, match="multiple"):
        effective_params("itransformer", {"d_model": 10, "n_heads": 4})


def test_dlinear_splits_the_window_into_a_moving_average_and_the_rest_and_is_linear():
    torch.manual_seed(1)
    net = DLinear(n_features=3, lookback=10, n_out=1, kernel_size=5).eval()
    x = torch.randn(6, 10, 3)
    trend, rest = net.decompose(x)
    assert torch.allclose(trend + rest, x.transpose(1, 2), atol=1e-6)
    assert torch.allclose(trend[:, :, 4], x[:, 2:7, :].mean(dim=1), atol=1e-6)            # away from the ends: the centred mean of five days
    assert torch.allclose(trend[:, :, 0], (3 * x[:, 0, :] + x[:, 1, :] + x[:, 2, :]) / 5, atol=1e-6)      # at the start the first value is repeated
    flat = torch.full((2, 10, 3), 0.7)
    t2, r2 = net.decompose(flat)
    assert torch.allclose(t2, torch.full_like(t2, 0.7)) and torch.allclose(r2, torch.zeros_like(r2), atol=1e-6)
    y = torch.randn(6, 10, 3)
    zero = net(torch.zeros(1, 10, 3))
    assert torch.allclose(net(2 * x + 3 * y) - zero, 2 * (net(x) - zero) + 3 * (net(y) - zero), atol=1e-5)      # one linear map of the window (plus a bias)
    assert not any(isinstance(m, (torch.nn.ReLU, torch.nn.GELU, torch.nn.MultiheadAttention)) for m in net.modules())


# ------------------------------------------------------------------------------------------------------------ batches of whole days
def test_group_batches_keep_every_day_whole_and_use_every_row_once():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 7)[rng.permutation(280)]
    out = tn.batches(280, 30, np.random.default_rng(1), groups)
    assert sorted(np.concatenate(out).tolist()) == list(range(280))
    homes = {}
    for b, rows in enumerate(out):
        for g in np.unique(groups[rows]):
            assert homes.setdefault(int(g), b) == b                                       # no day is spread over two batches
        assert len(rows) >= 30 or b == len(out) - 1
    assert [len(b) for b in tn.batches(10, 4, None)] == [4, 4, 2] and tn.batches(10, 4, None)[0].tolist() == [0, 1, 2, 3]
    ordered = tn.batches(280, 30, None, groups)
    assert [int(groups[b[0]]) for b in ordered] == sorted(int(groups[b[0]]) for b in ordered)       # without a generator the days stay in order (validation)


def test_the_default_batches_shuffle_rows_exactly_as_before():
    a = tn.batches(11, 4, np.random.default_rng(5))
    order = np.random.default_rng(5).permutation(11)
    assert [x.tolist() for x in a] == [order[0:4].tolist(), order[4:8].tolist(), order[8:11].tolist()]


# ------------------------------------------------------------------------------------------------------------ the losses
def panel(days=6, assets=9, seed=0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(days), assets)
    pred, target = rng.normal(size=(days * assets, 2)), rng.normal(size=(days * assets, 2))
    mask = np.ones_like(pred)
    mask[rng.choice(days * assets, 8, replace=False), 0] = 0.0
    return pred, target, mask, day


def test_daily_ic_is_the_mean_of_each_days_pearson_correlation_with_missing_labels_left_out():
    pred, target, mask, day = panel()
    want = []
    for k in range(2):
        for d in np.unique(day):
            rows = (day == d) & (mask[:, k] > 0)
            want.append(stats.pearsonr(pred[rows, k], target[rows, k])[0])
    shuffled = np.random.default_rng(3).permutation(len(day))                             # the rows of a day need not sit together
    got, used = ls.daily_ic(*(torch.tensor(a[shuffled], dtype=torch.float64) for a in (pred, target, mask)), torch.tensor(day[shuffled]))
    assert float(got) == pytest.approx(np.mean(want), abs=1e-6) and int(used) == 12


def test_daily_ic_skips_a_day_with_too_few_assets_or_a_flat_target():
    pred, target, mask, day = panel(days=4, assets=6, seed=2)
    mask[:] = 1.0
    mask[day == 0, :] = 0.0
    mask[np.flatnonzero(day == 0)[:2], :] = 1.0                                           # two labels on day 0: not a cross-section
    target[day == 1, :] = 0.3                                                              # a day whose target does not vary
    got, used = ls.daily_ic(*(torch.tensor(a, dtype=torch.float64) for a in (pred, target, mask)), torch.tensor(day))
    want = np.mean([stats.pearsonr(pred[day == d, k], target[day == d, k])[0] for k in range(2) for d in (2, 3)])
    assert int(used) == 4 and float(got) == pytest.approx(want, abs=1e-6)


def test_pairwise_rank_is_zero_for_a_perfect_order_and_matches_the_pairs_by_hand():
    pred, target, mask, day = panel(days=3, assets=5, seed=4)
    t = [torch.tensor(a, dtype=torch.float64) for a in (pred, target, mask)]
    total, count = 0.0, 0
    for k in range(2):
        for i in range(len(day)):
            for j in range(len(day)):
                if day[i] == day[j] and mask[i, k] and mask[j, k]:
                    total += max(0.0, -(pred[i, k] - pred[j, k]) * (target[i, k] - target[j, k]))
                    count += 1
    assert float(ls.pairwise_rank(*t, torch.tensor(day))) == pytest.approx(total / count, abs=1e-9)
    monotone = torch.tensor(np.tanh(target) * 5.0)                                        # the same order as the target on every day
    assert float(ls.pairwise_rank(monotone, t[1], t[2], torch.tensor(day))) == 0.0
    assert float(ls.pairwise_rank(-monotone, t[1], t[2], torch.tensor(day))) > 0.0


def test_the_ic_loss_does_not_reward_getting_the_days_level_right_and_the_squared_error_does():
    pred, target, mask, day = panel(days=5, assets=8, seed=6)
    mask[:] = 1.0
    level = np.repeat(np.random.default_rng(1).normal(size=5) * 4.0, 8)[:, None]         # a market move common to the day's assets
    t = lambda a: torch.tensor(a, dtype=torch.float64)                                    # noqa: E731
    d = torch.tensor(day)
    ic_a, _ = ls.daily_ic(t(pred), t(target + level), t(mask), d)
    ic_b, _ = ls.daily_ic(t(pred), t(target), t(mask), d)
    assert float(ic_a) == pytest.approx(float(ic_b), abs=1e-9)
    assert float(ls.mse(t(pred), t(target + level), t(mask))) > float(ls.mse(t(pred), t(target), t(mask))) + 5.0
    with pytest.raises(ValueError, match="unknown cross-sectional loss"):
        ls.cross_sectional("huber", t(pred), t(target), t(mask), d)


# ------------------------------------------------------------------------------------------------------------ end to end
@pytest.mark.parametrize("kind", ["itransformer", "dlinear"])
def test_the_new_networks_fit_predict_and_survive_a_round_trip(ds, kind):
    model, rows = fit_on(ds, kind)
    later = np.flatnonzero((ds.positions() >= 720) & (ds.positions() < 760))
    scores = model.predict(ds.X.iloc[later])
    assert scores.shape == (len(later),) and np.isfinite(scores).all() and scores.std() > 0
    d = model.diagnostics()
    assert d["arch"] == kind and d["loss"] == "mse" and d["epochs_run"] >= 1 and len(model.importance()) == ds.X.shape[1]
    again = md.Model.from_bytes(model.to_bytes()).bind(ds)
    assert np.allclose(again.predict(ds.X.iloc[later]), scores, atol=1e-6)


@pytest.mark.parametrize("loss", ["ic", "mse_rank"])
def test_a_cross_sectional_loss_trains_on_whole_days_and_reports_itself(ds, loss):
    model, rows = fit_on(ds, "lstm", loss=loss, max_rows=1200)
    d = model.diagnostics()
    assert d["loss"] == loss and np.isfinite(d["val_loss"]).all()
    n_train, n_val = d["windows"]
    assert n_train % 8 == 0 and n_val % 8 == 0                                            # eight assets a day: the thinning to max_rows kept or dropped whole days
    assert n_train <= 1200 + 8
    later = np.flatnonzero((ds.positions() >= 720) & (ds.positions() < 760))
    assert np.isfinite(model.predict(ds.X.iloc[later])).all()


@pytest.mark.parametrize("loss", ["ic", "mse_rank"])
def test_the_training_loop_with_a_cross_sectional_loss_recovers_a_planted_ordering_under_a_large_common_move(loss):
    """Each day's target is ``x0 - x1`` plus a market move five times larger that every asset of the day shares. The loop, fed whole days, must learn the ordering: a daily correlation above 0.8 on later days."""
    rng = np.random.default_rng(0)
    days, assets = 260, 10
    x = rng.normal(size=(days * assets, 4)).astype(np.float32)
    day = np.repeat(np.arange(days), assets)
    y = (x[:, 0] - x[:, 1] + 0.3 * rng.normal(size=len(x)) + np.repeat(rng.normal(size=days) * 5.0, assets)).astype(np.float32)[:, None]
    y = (y - y.mean()) / y.std()
    split = {"train": np.flatnonzero(day < 180), "val": np.flatnonzero((day >= 190) & (day < 220))}
    test = np.flatnonzero(day >= 230)
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 1)

    def batch_fn(name, idx):
        r = split[name][idx]
        return torch.from_numpy(x[r]), torch.from_numpy(y[r]), torch.ones(len(r), 1), torch.from_numpy(day[r])

    def loss_fn(model, batch):
        xb, yb, mb, db = batch
        return ls.cross_sectional(loss, model(xb), yb, mb, db)

    hist = tn.fit_network(net, batch_fn, loss_fn, len(split["train"]), len(split["val"]), tn.TrainConfig(epochs=40, batch_size=100, learning_rate=0.05, weight_decay=0.0, patience=40, device="cpu"),
                          {"train": day[split["train"]], "val": day[split["val"]]})
    assert hist.val_loss[hist.best_epoch] < hist.val_loss[0]
    with torch.no_grad():
        score = net(torch.from_numpy(x[test])).numpy()[:, 0]
    ics = [stats.pearsonr(score[day[test] == d], y[test][day[test] == d, 0])[0] for d in np.unique(day[test])]
    assert np.mean(ics) > 0.8
    w = net.weight.detach().numpy()[0]
    assert w[0] > 0 > w[1] and abs(w[2]) < 0.5 * w[0] and abs(w[3]) < 0.5 * w[0]


def test_a_loss_setting_is_refused_where_the_network_has_its_own(ds):
    assert "loss" not in md._INFO["tft"].params and "loss" not in md._INFO["vae"].params
    with pytest.raises(ParamError):
        effective_params("tft", {"loss": "ic"})
    with pytest.raises(ParamError):
        effective_params("lstm", {"loss": "sharpe"})
    with pytest.raises(ParamError):
        effective_params("lstm", {"rank_weight": -1})
    assert effective_params("gru", {})["loss"] == "mse" and effective_params("itransformer", {"loss": "ic"})["loss"] == "ic"
