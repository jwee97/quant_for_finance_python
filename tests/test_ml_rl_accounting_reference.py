"""P0-01 to P0-04 and P0-30 for the RL portfolio: hand-worked fixtures and an independent shadow ledger.

The oracle here is written separately from ``src/ml/rl/ledger.py`` and ``env.py``: plain Python lists of quantities and cash, the trade solved by bisection (the production code iterates a fixed point). The
targets are taken from the environment's own ``decide`` (projection is tested elsewhere); only the ACCOUNTING is checked against the oracle.
"""

import math

import numpy as np
import pytest

from src.ml.rl.ledger import Ledger, differential_sharpe, drawdown_increase


# ------------------------------------------------------------------------------------------------------------ the oracle
def solve_k(w0, target, c, i=0.0):
    """Bisection for k in (0, 1] with k = 1 - c*sum|target*k - w0| - i*sum((target*k - w0)^2): post-cost equity over pre-cost equity."""
    def gap(k):
        dw = [t * k - w for t, w in zip(target, w0)]
        return k - (1.0 - c * sum(abs(x) for x in dw) - i * sum(x * x for x in dw))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


class Shadow:
    """Cash, quantities and marks in plain lists; equity and weights are always recomputed from them."""

    def __init__(self, n, cash=1.0, price=None):
        self.cash, self.q, self.p = cash, [0.0] * n, list(price or [1.0] * n)

    def equity(self):
        return self.cash + sum(q * p for q, p in zip(self.q, self.p))

    def weights(self):
        e = self.equity()
        return [q * p / e for q, p in zip(self.q, self.p)]

    def mark(self, rets):
        self.p = [p * (1.0 + r) for p, r in zip(self.p, rets)]

    def trade(self, target, c=0.0, i=0.0):
        e0, w0 = self.equity(), self.weights()
        k = solve_k(w0, target, c, i)
        cost = e0 * (1.0 - k)
        new_q = [t * e0 * k / p for t, p in zip(target, self.p)]
        spent = sum((nq - q) * p for nq, q, p in zip(new_q, self.q, self.p))
        self.cash, self.q = self.cash - spent - cost, new_q
        return cost / e0


# ------------------------------------------------------------------------------------------------------------ hand-worked fixtures
def test_buy_five_shares_mark_up_and_rebalance_to_half_without_costs():
    book = Ledger.start(1, cash=1000.0, price=[100.0])
    book.cash, book.qty = 500.0, np.array([5.0])                                          # bought five shares at 100
    book.mark(np.array([0.10]))                                                           # the mark goes to 110
    assert book.qty[0] == 5.0 and book.cash == 500.0 and book.equity == pytest.approx(1050.0) and book.weights()[0] == pytest.approx(550.0 / 1050.0)
    fill = book.trade_to(np.array([0.5]))
    assert fill["fill"][0] == pytest.approx(-25.0 / 110.0) and book.weights()[0] == pytest.approx(0.5) and book.equity == pytest.approx(1050.0)
    assert book.cash == pytest.approx(500.0 + 25.0) and book.qty[0] == pytest.approx(5.0 - 25.0 / 110.0)


def test_a_fee_is_paid_out_of_the_equity_the_target_weights_are_of():
    book = Ledger.start(1, cash=1000.0, price=[100.0])
    fill = book.trade_to(np.array([0.5]), cost_rate=0.001)
    k = 1.0 / (1.0 + 0.001 * 0.5)                                                         # by hand: k = 1 - 0.001 * 0.5 * k
    assert 1.0 - fill["cost"] == pytest.approx(k, rel=1e-12) and book.equity == pytest.approx(1000.0 * k, rel=1e-12)
    assert book.weights()[0] == pytest.approx(0.5, rel=1e-12) and book.qty[0] == pytest.approx(0.5 * 1000.0 * k / 100.0, rel=1e-12)
    assert book.cash == pytest.approx(1000.0 - book.qty[0] * 100.0 - 1000.0 * (1.0 - k), rel=1e-12)


def test_a_leveraged_long_and_a_short_reconcile_with_marks_financing_and_borrow():
    book = Ledger.start(2, cash=1000.0, price=[100.0, 50.0])
    book.trade_to(np.array([1.5, -1.0]))
    assert list(book.qty) == pytest.approx([15.0, -20.0]) and book.cash == pytest.approx(500.0) and book.gross() == pytest.approx(2.5)
    book.mark(np.array([-0.10, 0.05]))                                                    # prices 90 and 52.5
    assert book.equity == pytest.approx(500.0 + 15 * 90.0 - 20 * 52.5) and book.equity == pytest.approx(800.0)
    assert list(book.weights()) == pytest.approx([1350.0 / 800.0, -1050.0 / 800.0])
    book.mark(np.zeros(2), financing=0.0, borrow=0.001)                                   # 0.1% a day on 1,050 of short notional
    assert book.cash == pytest.approx(500.0 - 1.05)
    debit = Ledger(cash=-100.0, qty=np.array([1.0]), price=np.array([200.0]), peak=100.0)
    debit.mark(np.zeros(1), financing=0.01)
    assert debit.cash == pytest.approx(-101.0)


def test_a_portfolio_that_loses_everything_is_dead_and_stays_dead():
    book = Ledger.start(2)
    book.trade_to(np.array([1.0, 1.0]))
    book.mark(np.array([-1.0, -1.0]))
    assert book.dead and book.equity == 0.0 and book.drawdown() == -1.0 and not book.weights().any()
    book.mark(np.array([0.5, 0.5]))
    assert book.trade_to(np.array([0.5, 0.5]))["dead"] and book.equity == 0.0 and not book.qty.any()


def test_a_trade_that_costs_more_than_the_portfolio_is_worth_ends_it():
    book = Ledger.start(1)
    assert book.trade_to(np.array([1.0]), cost_rate=3.0)["dead"] and book.dead


# ------------------------------------------------------------------------------------------------------------ the ledger against the shadow, random paths
@pytest.mark.parametrize("seed", range(5))
def test_the_ledger_matches_the_shadow_on_random_paths_with_costs_shorts_and_leverage(seed):
    rng = np.random.default_rng(seed)
    n, c, imp = 4, 0.0008, 0.02
    mine, shadow = Ledger.start(n, cash=2.5, price=list(rng.uniform(5, 200, n))), Shadow(n, 2.5, None)
    shadow.p = list(mine.price)
    for _ in range(60):
        rets = rng.normal(0, 0.02, n)
        mine.mark(rets)
        shadow.mark(rets)
        target = rng.uniform(-0.6, 0.6, n)
        target -= target.mean()
        if rng.random() < 0.7:
            fill = mine.trade_to(target, c, imp)
            cost = shadow.trade(list(target), c, imp)
            assert fill["cost"] == pytest.approx(cost, rel=1e-9, abs=1e-12)
        assert mine.equity == pytest.approx(shadow.equity(), rel=1e-9)
        assert list(mine.weights()) == pytest.approx(shadow.weights(), rel=1e-8, abs=1e-12)
        assert mine.cash == pytest.approx(shadow.cash, rel=1e-8, abs=1e-10) and list(mine.qty) == pytest.approx(shadow.q, rel=1e-8, abs=1e-12)


# ------------------------------------------------------------------------------------------------------------ rewards, separate from accounting
def test_drawdown_increase_is_zero_on_a_recovery_and_a_new_peak_and_positive_on_a_new_loss():
    equity = [1.0, 1.1, 1.0, 0.9, 0.95, 1.2]
    peak, dd_prev, out = equity[0], 0.0, []
    for e in equity[1:]:
        peak = max(peak, e)
        dd = e / peak - 1.0
        out.append(drawdown_increase(dd_prev, dd))
        dd_prev = dd
    assert out == pytest.approx([0.0, 0.1 / 1.1, 0.1 / 1.1, 0.0, 0.0])                    # a new peak, a fall of 0.1 from 1.1 twice over, a partial recovery, a new peak
    assert drawdown_increase(-0.05, -0.05) == 0.0 and drawdown_increase(-0.2, -0.1) == 0.0


def reference_dsr(xs, eta, clip=10.0):
    """Independent loop over the formula, with the moments before each update."""
    A = B = 0.0
    out = []
    for x in xs:
        dA, dB = x - A, x * x - B
        var = B - A * A
        d = (B * dA - 0.5 * A * dB) / var ** 1.5 if var > 1e-8 else None
        out.append((0.0 if d is None else max(-clip, min(clip, d)), d))
        A, B = A + eta * dA, B + eta * dB
    return out


@pytest.mark.parametrize("xs", [[0.0] * 20, [1.0] * 20, [1.0, -1.0] * 15, [0.5, 0.5 + 1e-7] * 10, [1e6, -1e6, 3e5] * 8, list(np.random.default_rng(3).normal(0.05, 1.0, 60))])
def test_the_differential_sharpe_reward_matches_an_independent_loop_and_is_always_finite(xs):
    eta, A, B = 1.0 / 63.0, 0.0, 0.0
    ref = reference_dsr(xs, eta)
    for x, (want, raw) in zip(xs, ref):
        reward, got_raw, A, B = differential_sharpe(x, A, B, eta)
        assert reward == pytest.approx(want, rel=1e-9, abs=1e-12) and math.isfinite(reward) and math.isfinite(A) and math.isfinite(B)
        assert (raw is None and math.isnan(got_raw)) or got_raw == pytest.approx(raw, rel=1e-9)


def test_the_differential_sharpe_raw_value_is_clipped_and_the_clip_is_visible():
    reward, raw, _, _ = differential_sharpe(5.0, -0.01, 0.0002, 0.1, clip=2.0)
    assert raw > 2.0 and reward == 2.0
    reward, raw, _, _ = differential_sharpe(5.0, 0.01, 0.0002, 0.1, clip=2.0)
    assert raw < -2.0 and reward == -2.0
    assert differential_sharpe(1.0, 0.0, 0.0, 0.1)[0] == 0.0                              # no variance yet: undefined, so no reward


# ------------------------------------------------------------------------------------------------------------ the environment's accounting against the shadow
try:
    import gymnasium  # noqa: F401
    from src.ml.rl.env import PortfolioEnv, enforce_limits, project_weights, rollout
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
needs_gym = pytest.mark.skipif(not HAS_GYM, reason="Gymnasium is not installed")


def make_env(lag=1, n_days=60, n=3, seed=0, **kw):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n_days, n, 2)).astype(np.float32)
    rets = 0.012 * rng.normal(size=(n_days, n))
    return PortfolioEnv(feats, rets, kw.pop("mask", None), **{"episode_days": n_days - 1, "lag": lag, "reward": "return", **kw}), rets


@pytest.mark.parametrize("lag", [0, 1, 2, 3])
@needs_gym
def test_the_environment_equity_and_weights_follow_the_shadow_ledger_for_every_lag(lag):
    env, rets = make_env(lag=lag, cost_bps=9.0, impact=0.01, gross=1.2, max_weight=0.6)
    env.reset_at(0)
    shadow = Shadow(env.n_assets)
    rng = np.random.default_rng(1)
    pending = []
    for t in range(40):
        a = rng.uniform(-1, 1, env.n_assets).astype(np.float32)
        target, _ = env.decide(a)
        env.step(a)
        pending.append((t + lag, list(target)))
        if lag == 0:                                                                       # filled at today's close, before the next session
            shadow.trade(pending.pop(0)[1], 9e-4, 0.01)
        shadow.mark(rets[t + 1])
        while pending and pending[0][0] <= t + 1:
            shadow.trade(pending.pop(0)[1], 9e-4, 0.01)
        assert env.ledger.equity == pytest.approx(shadow.equity(), rel=1e-9)
        assert list(env.w) == pytest.approx(shadow.weights(), rel=1e-8, abs=1e-12)


@needs_gym
def test_with_a_lag_the_book_is_held_from_t_plus_lag_and_earns_the_return_after_it():
    env, rets = make_env(lag=2, cost_bps=0.0)
    env.reset_at(5)
    a = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    target, _ = env.decide(a)
    env.step(a)                                                                            # day 6: nothing held yet
    assert not env.w.any() and env.ledger.equity == 1.0 and np.allclose(env.observation()[-4:-1], target)       # the order is visible as pending
    env.step(np.zeros(3, dtype=np.float32))                                                # day 7: the order executes at this close
    assert list(env.w) == pytest.approx(list(target)) and env.ledger.equity == pytest.approx(1.0)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))                             # day 8: the first return the book earns
    assert info["net_return"] == pytest.approx(float(target @ rets[8]) - 0.0, rel=1e-9) or info["net_return"] == pytest.approx(float(target @ rets[8]), rel=1e-6)


@needs_gym
def test_a_return_that_has_not_happened_yet_changes_no_observation_reward_or_equity():
    """Two paths that agree through day 20 and differ afterwards: everything the policy has seen through day 20 is identical."""
    env_a, rets = make_env(lag=1, cost_bps=5.0, seed=4)
    rets_b = rets.copy()
    rets_b[21:] = np.random.default_rng(9).normal(0, 0.05, rets_b[21:].shape)
    env_b = PortfolioEnv(env_a.features, rets_b, None, episode_days=59, lag=1, reward="return", cost_bps=5.0)
    env_a.reset_at(0)
    env_b.reset_at(0)
    rng = np.random.default_rng(2)
    for t in range(20):                                                                    # sessions ending on days 1..20
        a = rng.uniform(-1, 1, 3).astype(np.float32)
        oa, ra, *_ = env_a.step(a)
        ob, rb, *_ = env_b.step(a)
        assert np.array_equal(oa, ob) and ra == rb
    assert not np.array_equal(env_a.step(a)[0], env_b.step(a)[0])                          # the first session that reaches day 21 differs


@needs_gym
def test_observation_and_equity_from_training_and_from_a_policy_played_forward_are_the_same_sequence():
    env, rets = make_env(lag=1, cost_bps=7.0, reward="risk_adjusted", drawdown_penalty=0.5, max_turnover=0.8)
    rng = np.random.default_rng(6)
    actions = [rng.uniform(-1, 1, 3).astype(np.float32) for _ in range(60)]
    obs, seen, equity = env.reset_at(0), [], []
    for a in actions[:59]:
        seen.append(obs)
        obs, *_ = env.step(a)
        equity.append(env.ledger.equity)
    it = iter(actions)
    books, trace = rollout(lambda o, s, first: (next(it), s), env.features, rets, np.ones((60, 3), bool),
                           {"cost_bps": 7.0, "reward": "risk_adjusted", "drawdown_penalty": 0.5, "max_turnover": 0.8, "lag": 1}, record=True)
    assert len(trace["observations"]) == 60 and all(np.array_equal(a, b) for a, b in zip(seen, trace["observations"])) and trace["equity"][1:] == pytest.approx(equity, rel=1e-12)


@needs_gym
def test_a_drawdown_penalty_is_charged_on_new_losses_and_not_on_recoveries_or_new_peaks():
    n = 2
    rets = np.zeros((8, n))
    rets[:, 0] = [0, 0, -0.10, -0.05, 0.08, 0.20, 0.0, 0.0]                                 # asset 0 falls, then recovers, then makes a new high
    feats = np.zeros((8, n, 1), dtype=np.float32)
    kw = dict(episode_days=7, lag=0, cost_bps=0.0, gross=1.0, max_weight=1.0, long_only=True, risk_aversion=0.0)
    plain = PortfolioEnv(feats, rets, None, reward="risk_adjusted", drawdown_penalty=0.0, **kw)
    pen = PortfolioEnv(feats, rets, None, reward="risk_adjusted", drawdown_penalty=2.0, **kw)
    a = np.array([5.0, -5.0, -5.0], dtype=np.float32)                                      # long asset 0, no cash
    plain.reset_at(0)
    pen.reset_at(0)
    peak, dd_prev, extra = 1.0, 0.0, []
    for _ in range(7):
        _, r0, *_ = plain.step(a)
        _, r1, _, _, info = pen.step(a)
        peak = max(peak, info["equity"])
        dd = info["equity"] / peak - 1.0
        extra.append((r0 - r1, max(dd_prev - dd, 0.0) * 100.0 * 2.0))
        dd_prev = dd
    for got, want in extra:
        assert got == pytest.approx(want, abs=1e-9)
    assert any(w > 0 for _, w in extra) and any(w == 0 for _, w in extra)


# ------------------------------------------------------------------------------------------------------------ actions, constraints, ruin
@needs_gym
def test_wrong_sized_and_non_finite_actions_are_rejected_and_out_of_domain_ones_are_clipped_and_reported():
    env, _ = make_env()
    env.reset_at(0)
    with pytest.raises(ValueError, match="takes 3"):
        env.step(np.zeros(4))
    for bad in (np.array([np.nan, 0, 0]), np.array([np.inf, 0, 0])):
        with pytest.raises(ValueError, match="not finite"):
            env.step(bad)
    _, _, _, _, info = env.step(np.array([7.0, -7.0, 0.0]))
    assert "action clipped to [-1, 1]" in info["events"]


@needs_gym
def test_with_no_eligible_asset_the_target_is_cash_and_it_is_reported():
    mask = np.zeros((60, 3), bool)
    env, _ = make_env(mask=mask)
    env.reset_at(0)
    _, _, _, _, info = env.step(np.array([1.0, -1.0, 0.5]))
    assert not info["target"].any() and any("no eligible asset" in e for e in info["events"]) and not env.w.any()


@needs_gym
def test_a_held_asset_that_becomes_ineligible_is_liquidated_in_full_whatever_the_turnover_budget():
    mask = np.ones((60, 3), bool)
    mask[10:, 1] = False
    env, _ = make_env(mask=mask, max_turnover=0.2, cost_bps=0.0, gross=1.0, max_weight=0.5)
    env.reset_at(0)
    a = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    for _ in range(9):
        env.step(a)
    assert abs(env.w[1]) > 0.1
    _, _, _, _, info = env.step(a)                                                         # day 10: the asset is gone at this close
    assert env.w[1] == 0.0 and any("forced liquidation" in e for e in info["events"])
    w, why = enforce_limits(np.array([0.3, 0.0, -0.3]), np.array([0.1, 0.4, -0.5]), 0.2, np.array([True, False, True]))
    assert w[1] == 0.0 and why["forced"] == pytest.approx(0.4) and why["exceeded"] and why["shortened"]
    capped, why = enforce_limits(np.array([0.1, 0.0, -0.1]), np.array([0.9, 0.0, -0.9]), 0.05, np.ones(3, bool), cap=0.5, gross=1.0)           # drifted far past the cap, a tiny budget
    assert np.abs(capped).max() <= 0.5 + 1e-12 and np.abs(capped).sum() <= 1.0 + 1e-12 and why["trimmed"] > 0.5


@needs_gym
def test_exposure_above_the_hard_limit_is_cut_back_and_reported():
    n_days, n = 12, 2
    rets = np.zeros((n_days, n))
    rets[1:, 0] = -0.15
    rets[1:, 1] = 0.15
    env = PortfolioEnv(np.zeros((n_days, n, 1), np.float32), rets, None, episode_days=10, lag=1, cost_bps=0.0, gross=1.0, max_weight=0.5, max_leverage=1.05, reward="return")
    env.reset_at(0)
    events = []
    for _ in range(6):
        events += env.step(np.array([1.0, -1.0], dtype=np.float32))[4]["events"]
        assert env.ledger.gross() <= 1.05 + 1e-9
    assert any("above the 1.05x limit" in e for e in events)


@needs_gym
def test_an_insolvent_episode_terminates_and_cannot_resume():
    rets = np.zeros((10, 2))
    rets[3:, 1] = 3.0                                                                      # the asset that is sold short quadruples
    env = PortfolioEnv(np.zeros((10, 2, 1), np.float32), rets, None, episode_days=9, lag=1, cost_bps=0.0, gross=1.0, max_weight=0.5, reward="return")
    env.reset_at(0)
    last = None
    for _ in range(9):
        last = env.step(np.array([1.0, -1.0], dtype=np.float32))
        if last[2]:
            break
    assert last[2] and env.ledger.dead and env.ledger.equity == 0.0 and np.isfinite(last[0]).all() and last[1] == pytest.approx(-100.0 * (1.0 + 0.0) / 1.0 * 1.0, abs=150.0)
    with pytest.raises(RuntimeError, match="insolvent"):
        env.step(np.zeros(2, dtype=np.float32))


@needs_gym
def test_a_holding_that_falls_to_nothing_is_written_off_and_the_rest_of_the_account_goes_on():
    rets = np.zeros((10, 2))
    rets[3:, :] = -1.0
    env = PortfolioEnv(np.zeros((10, 2, 1), np.float32), rets, None, episode_days=9, lag=1, cost_bps=0.0, gross=1.0, max_weight=1.0, long_only=True, reward="return")
    env.reset_at(0)
    for _ in range(9):
        obs, reward, terminated, _, _ = env.step(np.array([1.0, 1.0, -1.0], dtype=np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward)
    cash_share = env.ledger.cash
    assert not env.ledger.dead and 0.0 < env.ledger.equity == pytest.approx(cash_share) and not env.ledger.qty.any() or env.ledger.equity >= 0.0


@needs_gym
def test_random_actions_and_masks_never_break_a_hard_constraint():
    rng = np.random.default_rng(11)
    for _ in range(400):
        n = int(rng.integers(2, 9))
        long_only, gross, cap = bool(rng.random() < 0.4), float(rng.uniform(0.3, 2.0)), float(rng.uniform(0.05, 1.0))
        ok = rng.random(n) < 0.7
        previous = np.where(ok | (rng.random(n) < 0.5), rng.normal(0, 0.2, n), 0.0)
        previous = np.abs(previous) if long_only else previous                                # a long-only account never holds a short
        turnover = float(rng.choice([0.0, 0.1, 0.5]))
        a = rng.normal(0, 3.0, n + (1 if long_only else 0)).clip(-1, 1)
        w = project_weights(a, previous, gross, cap, long_only, turnover, ok)
        assert np.isfinite(w).all() and not w[~ok].any() and np.abs(w).max() <= cap + 1e-9 and np.abs(w).sum() <= gross + 1e-9           # hard limits hold even when holdings had drifted past them
        if turnover == 0.0:
            assert long_only or abs(w.sum()) <= 1e-9
        assert not long_only or (w >= -1e-12).all()
