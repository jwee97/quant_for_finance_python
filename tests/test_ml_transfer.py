"""Release 8, P3-06: cross-asset transfer with explicit source and target domains and no leakage between them.

A panel is built in which two groups of assets either share the structure that links features to the label or have opposite structures. The guards are tested one at a time (shared securities, an evaluation
that starts before the cutoff, labels of a different horizon); the source rows are counted by hand; the scaler and the model are shown to depend on source rows before the cutoff and on nothing else; and the
verdicts are checked on a case of positive transfer and one of negative transfer."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from src.ml import transfer as tf

LAG, H = 1, 5
SRC = [f"S{i}" for i in range(20)]
TGT = [f"T{i}" for i in range(20)]
DATES = pd.bdate_range("2018-01-01", periods=700)
P = 30


def panel(sign=1.0, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    beta = np.zeros(P)
    beta[:8] = rng.normal(0, 0.4, 8)
    cols = [f"F{k}" for k in range(P)]
    mi = pd.MultiIndex.from_product([DATES, SRC + TGT], names=["date", "asset"])
    X = pd.DataFrame(rng.normal(size=(len(mi), P)), index=mi, columns=cols)
    b = np.where(np.isin(mi.get_level_values(1), SRC), 1.0, sign)[:, None] * beta[None, :]
    y = pd.Series((X.to_numpy() * b).sum(axis=1) + rng.normal(0, noise, len(mi)), index=mi)
    return X, y


def run(X, y, **kw):
    kw.setdefault("source_end", DATES[450])
    kw.setdefault("eval_start", DATES[460])
    kw.setdefault("target_history_end", DATES[14])
    return tf.run(X, y, DATES, SRC, TGT, lag=LAG, horizon=H, **kw)


# ------------------------------------------------------------------------------------------------------------------ the guards
def test_the_domains_must_be_disjoint_the_evaluation_must_follow_the_cutoff_and_the_labels_must_mean_the_same_thing():
    X, y = panel()
    with pytest.raises(tf.TransferError, match=r"share \['S0'\]"):
        tf.run(X, y, DATES, SRC, ["S0", "T1"], source_end=DATES[450], eval_start=DATES[460])
    with pytest.raises(tf.TransferError, match="at least one asset"):
        tf.check_split([], TGT)
    with pytest.raises(tf.TransferError, match="before the source cutoff"):
        run(X, y, eval_start=DATES[400])
    with pytest.raises(tf.TransferError, match="must end before the evaluation starts"):
        run(X, y, target_history_end=DATES[470])
    with pytest.raises(tf.TransferError, match="representation is"):
        run(X, y, representation="vibes")
    wrong = pd.Series(LAG + H + 3, index=X.index)                                                    # a label interval of another length
    with pytest.raises(tf.TransferError, match="do not share one lag and horizon"):
        run(X, y, horizon_check=wrong)
    run(X, y, horizon_check=pd.Series(LAG + H, index=X.index))


def test_source_rows_are_those_whose_labels_ended_by_the_cutoff_and_the_label_end_is_on_the_real_calendar():
    X, y = panel()
    out = run(X, y)
    cut = 450
    assert out["rows"]["source_train"] == (cut - (LAG + H) + 1) * len(SRC), "decisions up to position cutoff - lag - horizon: later ones have labels that end after the cutoff"
    assert out["rows"]["target_eval"] == (len(DATES) - 460) * len(TGT) and out["label_rule"].startswith("a row's label ends 6 sessions after")
    ends = tf.label_end(pd.Index([DATES[0], DATES[-2]]), DATES, LAG, H)
    assert ends.iloc[0] == DATES[6] and pd.isna(ends.iloc[1]), "the last decisions have no label end yet"


# ------------------------------------------------------------------------------------------------------------------ nothing is fitted on a row that is evaluated or comes after the cutoff
def test_the_scaler_and_the_transferred_model_depend_on_source_rows_before_the_cutoff_and_on_nothing_else():
    X, y = panel()
    out = run(X, y)
    rows = (np.isin(X.index.get_level_values(1), SRC)) & (X.index.get_level_values(0) <= DATES[450 - (LAG + H)])
    assert out["scaler"]["fitted_on"] == "source training rows only" and out["scaler"]["mean"] == pytest.approx(X[rows].mean(axis=0).to_numpy(), abs=1e-12)
    X2, y2 = X.copy(), y.copy()
    late = np.isin(X.index.get_level_values(1), SRC) & (X.index.get_level_values(0) > DATES[450 - (LAG + H)])
    X2[late] = 50.0                                                                                  # the source domain after the cutoff is replaced
    y2[late] = -1e3
    again = run(X2, y2)
    assert again["transfer_ic"] == pytest.approx(out["transfer_ic"], abs=1e-12) and again["scaler"]["mean"] == pytest.approx(out["scaler"]["mean"], abs=1e-12), "nothing after the cutoff reached the source model"
    X3, y3 = X.copy(), y.copy()
    early_t = np.isin(X.index.get_level_values(1), TGT) & (X.index.get_level_values(0) <= DATES[14 - (LAG + H)])
    y3[early_t] = np.random.default_rng(9).normal(0, 3, int(early_t.sum()))                          # the target's own early history is scrambled
    moved = run(X3, y3)
    assert moved["transfer_ic"] == pytest.approx(out["transfer_ic"], abs=1e-12) and moved["scratch_ic"] != pytest.approx(out["scratch_ic"], abs=1e-6), "the transferred model never saw the target; the scratch model did"


# ------------------------------------------------------------------------------------------------------------------ verdicts
def test_a_shared_structure_transfers_and_an_opposite_one_is_called_negative_transfer():
    X, y = panel(sign=1.0, seed=1)
    pos = run(X, y)
    assert pos["eligibility"] == "applied" and pos["verdict"] == "positive_transfer" and pos["transfer_ic"] > pos["scratch_ic"] > 0
    assert pos["difference"]["interval"][0] > 0 and "beats training on the target alone" in pos["reason"] and pos["rows"]["target_history"] < 200 and "one fit at the cutoff" in pos["limitations"][0]
    Xn, yn = panel(sign=-1.0, seed=1)
    neg = run(Xn, yn)
    assert neg["verdict"] == "negative_transfer" and neg["transfer_ic"] < 0 < neg["scratch_ic"] and "worse than training on the target alone" in neg["reason"]
    same_target_data = run(X, y, target_history_end=DATES[300])
    assert same_target_data["verdict"] in ("inconclusive", "negative_transfer", "positive_transfer") and same_target_data["scratch_ic"] > pos["scratch_ic"], "with a long target history the from-scratch model catches up"


def test_a_pca_representation_is_learned_on_the_source_alone_and_matches_a_second_fit():
    X, y = panel(seed=2)
    out = run(X, y, representation="pca", n_components=5)
    rows = (np.isin(X.index.get_level_values(1), SRC)) & (X.index.get_level_values(0) <= DATES[450 - (LAG + H)])
    Z = (X[rows] - X[rows].mean()) / X[rows].std(ddof=1)
    ref = PCA(n_components=5, random_state=0).fit(Z.to_numpy())
    assert out["pca"]["explained_variance_ratio"] == pytest.approx(list(ref.explained_variance_ratio_), abs=1e-9) and out["pca"]["fitted_on"] == "source training rows only" and out["representation"] == "pca"
    assert out["eligibility"] == "applied" and out["verdict"] in ("positive_transfer", "inconclusive", "negative_transfer")


def test_too_little_data_or_no_target_history_is_said_and_not_filled_in():
    X, y = panel()
    short = tf.run(X, y, DATES, SRC, TGT, source_end=DATES[6], eval_start=DATES[20], lag=LAG, horizon=H)
    assert short["eligibility"] == "insufficient" and "at least 60" in short["reason"]
    no_hist = run(X, y, target_history_end=DATES[3])
    assert no_hist["verdict"] == "no_baseline" and "no from-scratch model could be trained" in no_hist["reason"] and "scratch" not in no_hist["ic"] and no_hist["ic"]["transfer"]["value"] is not None
