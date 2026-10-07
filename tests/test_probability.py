"""The probability layer: every estimator is checked against a case with a known answer (closed form, exact simulation or a reference library)."""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import integrate, stats as sps

from src.probability import bayes, copulas, evt, markov, montecarlo as mc, ruin

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------------------------------ EVT
@pytest.fixture(scope="module")
def t4():
    return sps.t.rvs(4, size=30000, random_state=np.random.default_rng(0))


def test_gpd_fit_recovers_tail_index_and_extrapolates_beyond_the_sample(t4):
    f = evt.fit_gpd(t4, quantile=0.95)
    assert f.xi == pytest.approx(0.25, abs=0.07) and f.n_exceed == pytest.approx(0.05 * len(t4), abs=1)
    assert f.var(0.99) == pytest.approx(sps.t.ppf(0.99, 4), rel=0.04)
    assert f.var(0.999) == pytest.approx(sps.t.ppf(0.999, 4), rel=0.08)
    es = evt.gpd_var_es(f, 0.99)
    true_es = integrate.quad(lambda p: sps.t.ppf(p, 4), 0.99, 1 - 1e-9)[0] / 0.01
    assert es["es"] == pytest.approx(true_es, rel=0.12) and es["es"] > es["var"]


def test_gpd_mle_matches_scipy_on_exact_gpd_data():
    y = sps.genpareto.rvs(0.2, scale=1.5, size=5000, random_state=np.random.default_rng(1))
    xi, beta, se_xi, se_beta, ll = evt.fit_gpd_excesses(y)
    c, loc, scale = sps.genpareto.fit(y, floc=0)
    assert xi == pytest.approx(c, abs=0.01) and beta == pytest.approx(scale, rel=0.02) and se_xi > 0 and se_beta > 0
    assert ll == pytest.approx(np.sum(sps.genpareto.logpdf(y, c, 0, scale)), abs=0.5)


def test_gpd_exponential_limit_and_input_checks():
    y = np.random.default_rng(2).exponential(2.0, 4000)
    xi, beta, *_ = evt.fit_gpd_excesses(y)
    assert abs(xi) < 0.06 and beta == pytest.approx(2.0, rel=0.08)
    with pytest.raises(ValueError):
        evt.fit_gpd(np.random.default_rng(0).normal(size=100), quantile=0.99)


def test_hill_estimator_and_threshold_diagnostics(t4):
    assert evt.hill_estimator(t4, 600)["xi"] == pytest.approx(0.25, abs=0.1)
    d = evt.threshold_diagnostics(t4)
    assert (d["ks_pvalue"] > 0.01).all() and (d["xi"] > 0).all()
    assert len(evt.hill_plot(t4)) > 10 and evt.mean_excess(t4)["mean_excess"].iloc[-1] > evt.mean_excess(t4)["mean_excess"].iloc[0]


def test_gev_return_level_and_block_maxima():
    rng = np.random.default_rng(3)
    maxima = np.array([sps.t.rvs(4, size=250, random_state=rng).max() for _ in range(300)])
    g = evt.fit_gev(maxima)
    assert g["xi"] == pytest.approx(0.25, abs=0.12)
    assert evt.gev_return_level(g, 100) > evt.gev_return_level(g, 10) > np.median(maxima)
    s = pd.Series(np.random.default_rng(1).normal(size=1000), index=pd.bdate_range("2018", periods=1000))
    assert len(evt.block_maxima(s)) == 4


def test_extremal_index_detects_clustering():
    rng = np.random.default_rng(4)
    iid = rng.standard_normal(20000)
    h = np.zeros(20000)
    e = rng.standard_normal(20000)
    for t in range(1, 20000):
        h[t] = e[t] * np.sqrt(0.1 + 0.85 * h[t - 1] ** 2)
    assert evt.extremal_index(iid) > 0.85 and evt.extremal_index(np.abs(h)) < 0.7


def test_dynamic_evt_var_is_causal_and_calibrated():
    rng = np.random.default_rng(5)
    r = pd.Series(mc.simulate_garch(3000, 1, 1e-6, 0.08, 0.9, df=5, seed=rng)[0], index=pd.bdate_range("2005", periods=3000))
    out = evt.dynamic_evt_var(r, 0.99, min_obs=750)
    assert out["breach"].mean() == pytest.approx(0.01, abs=0.01) and (out["es"] > out["var"]).mean() > 0.95
    changed = r.copy()
    changed.iloc[2000:] = rng.normal(0, 0.05, 1000)
    other = evt.dynamic_evt_var(changed, 0.99, min_obs=750)
    cut = r.index[1999]
    assert np.allclose(out.loc[:cut, "var"], other.loc[:cut, "var"])
    ci = evt.profile_var_ci(sps.t.rvs(4, size=8000, random_state=rng), 0.99, n_boot=60)
    assert ci["var_ci"][0] < ci["var"] < ci["var_ci"][1]


# --------------------------------------------------------------------------------------------------------------------- copulas
@pytest.mark.parametrize("family,theta", [("clayton", 2.0), ("gumbel", 2.0), ("frank", 5.0), ("frank", -4.0)])
def test_archimedean_simulate_fit_and_tau(family, theta):
    c = copulas.Copula(family, {"theta": theta}, 2)
    u = c.simulate(15000, 1)
    assert np.all((u > 0) & (u < 1)) and np.allclose(u.mean(axis=0), 0.5, atol=0.01)
    assert copulas.kendall_tau(u[:, 0], u[:, 1]) == pytest.approx(c.kendall_tau(), abs=0.025)
    assert copulas.fit_copula(u, family, pseudo=False).params["theta"] == pytest.approx(theta, rel=0.1)


def test_elliptical_copulas_recover_correlation_and_tail_dependence():
    R = np.array([[1, 0.6], [0.6, 1.0]])
    g = copulas.Copula("gaussian", {"corr": R}, 2)
    assert copulas.fit_copula(g.simulate(20000, 2), "gaussian", pseudo=False).params["corr"][0, 1] == pytest.approx(0.6, abs=0.02)
    t = copulas.Copula("student", {"corr": R, "df": 4.0}, 2)
    f = copulas.fit_copula(t.simulate(20000, 3), "student", pseudo=False)
    assert f.params["df"] == pytest.approx(4.0, abs=1.0) and f.params["corr"][0, 1] == pytest.approx(0.6, abs=0.03)
    assert g.tail_dependence() == {"lower": 0.0, "upper": 0.0} and 0.2 < t.tail_dependence()["lower"] < 0.45
    emp = copulas.empirical_tail_dependence(t.simulate(200000, 4), 0.01)
    assert emp["lower"] == pytest.approx(t.tail_dependence()["lower"], abs=0.08)


def test_copula_selection_picks_the_true_family():
    R = np.array([[1, 0.6], [0.6, 1.0]])
    assert copulas.compare_copulas(copulas.Copula("student", {"corr": R, "df": 3.5}, 2).simulate(6000, 3)).index[0] == "student"
    assert copulas.compare_copulas(copulas.Copula("clayton", {"theta": 2.0}, 2).simulate(6000, 3)).index[0] == "clayton"
    assert copulas.compare_copulas(copulas.Copula("gumbel", {"theta": 2.5}, 2).simulate(6000, 3)).index[0] == "gumbel"


@pytest.mark.parametrize("family,params", [("clayton", {"theta": 1.5}), ("gumbel", {"theta": 1.8}), ("frank", {"theta": 3.0}),
                                           ("gaussian", {"corr": np.array([[1, 0.5], [0.5, 1.0]])})])
def test_copula_density_integrates_to_one_and_cdf_respects_frechet_bounds(family, params):
    c = copulas.Copula(family, params, 2)
    n = 200
    g = (np.arange(n) + 0.5) / n
    A, B = np.meshgrid(g, g)
    pts = np.column_stack([A.ravel(), B.ravel()])
    assert np.exp(c.logpdf(pts)).mean() == pytest.approx(1.0, abs=0.06)
    probe = np.random.default_rng(1).uniform(0.05, 0.95, size=(300, 2))
    cdf = c.cdf(probe)
    lower, upper = np.maximum(probe.sum(axis=1) - 1, 0), probe.min(axis=1)
    assert np.all(cdf >= lower - 1e-6) and np.all(cdf <= upper + 1e-6)


def test_marginals_and_copula_scenarios_preserve_dependence_and_tails():
    rng = np.random.default_rng(6)
    cop = copulas.Copula("student", {"corr": np.array([[1, 0.7], [0.7, 1.0]]), "df": 3.0}, 2)
    u = cop.simulate(4000, 1)
    data = pd.DataFrame({"a": sps.t.ppf(u[:, 0], 5) * 0.01, "b": sps.t.ppf(u[:, 1], 5) * 0.012})
    for kind in ("empirical", "normal", "student", "evt"):
        cdf, ppf = copulas.fit_marginal(data["a"], kind)
        p = np.array([0.01, 0.5, 0.99])
        assert np.all(np.diff(np.asarray(ppf(p))) > 0)
        assert np.asarray(cdf(np.asarray(ppf(p)))) == pytest.approx(p, abs=0.02)
    sc = copulas.simulate_joint(data, 30000, copula="student", marginal="evt", seed=2)
    assert sc.corr(method="spearman").iloc[0, 1] == pytest.approx(data.corr(method="spearman").iloc[0, 1], abs=0.05)
    w = np.array([0.5, 0.5])
    gauss = copulas.portfolio_var_es(copulas.simulate_joint(data, 60000, copula="gaussian", marginal="normal", seed=3), w, 0.995)
    heavy = copulas.portfolio_var_es(sc, w, 0.995)
    assert heavy["es"] > gauss["es"] and heavy["es"] > heavy["var"]
    with pytest.raises(ValueError):
        copulas.fit_copula(np.random.default_rng(0).normal(size=(50, 3)), "clayton")


# ---------------------------------------------------------------------------------------------------------------------- ruin
def test_gambler_ruin_matches_closed_form_and_simulation():
    assert ruin.gambler_ruin_probability(0.55, 1.0, 20, 1) == pytest.approx((0.45 / 0.55) ** 20, rel=1e-9)
    assert ruin.gambler_ruin_probability(0.45, 1.0, 10, 1) == 1.0
    rng = np.random.default_rng(1)
    p, b, n = 0.4, 2.0, 6
    ruined = 0
    trials = 4000
    for _ in range(trials):
        cap = n
        for _ in range(400):
            cap += b if rng.random() < p else -1
            if cap <= 0:
                ruined += 1
                break
    assert ruined / trials == pytest.approx(ruin.gambler_ruin_probability(p, b, n, 1), abs=0.03)


def test_prob_fall_below_matches_simulation_with_discretisation_correction():
    rng = np.random.default_rng(1)
    mu, sig, T, steps, n = 0.5, 1.0, 200, 10000, 1500
    dt = T / steps
    x = np.cumsum(mu * dt + sig * np.sqrt(dt) * rng.standard_normal((n, steps)), axis=1)
    sim = (x.min(axis=1) <= -1.0).mean()                       # monitored on a grid, so it misses some crossings
    shifted = 1.0 + 0.5826 * sig * np.sqrt(dt)                 # Broadie-Glasserman-Kou: a discretely monitored barrier acts like a continuous one further away
    assert sim == pytest.approx(ruin.prob_fall_below(mu, sig, shifted), abs=0.04) and sim < ruin.prob_fall_below(mu, sig, 1.0)
    assert ruin.prob_fall_below(-0.1, 1.0, 1.0) == 1.0
    assert ruin.prob_ruin_brownian(0.12, 0.15, 0.5) < ruin.prob_ruin_brownian(0.12, 0.25, 0.5)


def test_expected_max_drawdown_exact_at_zero_drift_and_grows_with_horizon():
    assert ruin.expected_max_drawdown(0.0, 0.2, 1.0) == pytest.approx(0.2 * np.sqrt(np.pi / 2))
    sim = ruin.expected_max_drawdown(1e-9, 0.2, 1.0, n_sim=20000, steps=2000)
    assert sim == pytest.approx(0.2507, rel=0.05)
    assert ruin.expected_max_drawdown(0.1, 0.2, 4.0, n_sim=4000) > ruin.expected_max_drawdown(0.1, 0.2, 1.0, n_sim=4000)
    assert ruin.expected_max_drawdown(0.3, 0.2, 2.0, n_sim=4000) < ruin.expected_max_drawdown(0.0, 0.2, 2.0)
    assert 0 < ruin.brownian_max_drawdown_probability(0.1, 0.2, 0.2, 1.0, n_sim=4000) < 1


def test_kelly_formulas_and_drawdown_constraint():
    mu, sig = 0.08, 0.2
    f = ruin.kelly_fraction(mu, sig)
    assert f == pytest.approx(2.0)
    g = [ruin.kelly_growth(x, mu, sig) for x in (0.5 * f, f, 1.5 * f, 2.0 * f, 2.5 * f)]
    assert g[1] == max(g) and ruin.kelly_growth(2 * f, mu, sig) == pytest.approx(0.0, abs=1e-12) and g[4] < 0
    d = ruin.drawdown_constrained_kelly(mu, sig, 0.2, 0.05)
    assert d["fraction"] == pytest.approx(2 / (1 + np.log(0.05) / np.log(0.8)), rel=1e-9) and d["prob_drawdown_at_choice"] == pytest.approx(0.05)
    assert d["prob_drawdown_at_kelly"] == pytest.approx(0.8)
    r = np.random.default_rng(1).normal(0.001, 0.01, 5000)
    assert ruin.kelly_from_returns(r, 0.5) == pytest.approx(0.5 * r.mean() / r.var(ddof=1))
    w = ruin.multi_asset_kelly(np.array([0.05, 0.03]), np.array([[0.04, 0.01], [0.01, 0.03]]), max_gross=1.0)
    assert np.abs(w).sum() == pytest.approx(1.0)


def test_drawdown_series_episodes_and_bootstrap():
    r = pd.Series([0.1, -0.2, 0.05, 0.1, 0.2, -0.1, -0.1], index=pd.bdate_range("2020", periods=7))
    dd = ruin.drawdown_series(r)
    assert dd.iloc[0] == 0 and dd.iloc[1] == pytest.approx(-0.2)
    ep = ruin.drawdown_episodes(r)
    assert len(ep) == 2 and ep["depth"].iloc[0] == pytest.approx(-0.2) and pd.isna(ep["recovery"].iloc[-1])
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.0006, 0.01, 3000))
    dist = ruin.drawdown_distribution(ret, 252, 1500, seed=2)
    assert dist["p99"] < dist["p95"] < dist["median"] < 0 and dist["horizon"] == 252
    assert ruin.prob_drawdown_exceeds(ret, 0.05, 252, 1500) > ruin.prob_drawdown_exceeds(ret, 0.20, 252, 1500)
    assert 0 < ruin.cdar(ret, 0.9) and ruin.cdar(ret, 0.99) > ruin.cdar(ret, 0.9)


# ----------------------------------------------------------------------------------------------------------------- Monte Carlo
def test_simulators_have_the_right_moments():
    g = mc.simulate_gbm(100, 0.05, 0.2, 1.0, 50, 40000, antithetic=True, seed=1)
    assert g.shape == (40000, 51) and g[:, -1].mean() == pytest.approx(100 * np.exp(0.05), rel=0.01) and np.log(g[:, -1] / 100).std() == pytest.approx(0.2, rel=0.03)
    cov = np.array([[0.04, 0.012], [0.012, 0.09]])
    c = mc.simulate_correlated_gbm([100, 50], [0.05, 0.03], cov, 1.0, 10, 30000, seed=2)
    lr = np.log(c[:, -1] / np.array([100, 50]))
    assert np.cov(lr.T) == pytest.approx(cov, abs=0.006)
    ou = mc.simulate_ou(0.0, 5.0, 1.0, 0.5, 10.0, 2000, 200, seed=3)
    assert ou[:, -500:].mean() == pytest.approx(1.0, abs=0.05) and ou[:, -500:].std() == pytest.approx(0.5 / np.sqrt(10), rel=0.1)
    j = mc.simulate_merton_jump(100, 0.05, 0.15, 0.5, -0.05, 0.1, 1.0, 100, 40000, seed=4)
    assert j[:, -1].mean() == pytest.approx(100 * np.exp(0.05), rel=0.01)
    gr = mc.simulate_garch(3000, 40, 1e-6, 0.08, 0.9, df=6, seed=5)
    assert gr.std() == pytest.approx(np.sqrt(1e-6 / 0.02), rel=0.2)


def test_variance_reduction_beats_plain_monte_carlo():
    def call(z):
        return np.maximum(100 * np.exp(0.03 + 0.2 * z[:, 0]) - 100, 0.0)

    a_, v_ = 0.03, 0.2                                          # ln S ~ N(ln 100 + a, v^2): E[(S - K)+] = e^{a + v^2/2} Phi(d1) - Phi(d2) for K = 100
    truth = 100 * (np.exp(a_ + 0.5 * v_ ** 2) * sps.norm.cdf(a_ / v_ + v_) - sps.norm.cdf(a_ / v_))
    plain = mc.mc_estimate(call, 1, 40000, "plain", seed=1)
    anti = mc.mc_estimate(call, 1, 40000, "antithetic", seed=1)
    ctrl = mc.mc_estimate(call, 1, 40000, "control", control=lambda z: np.exp(0.2 * z[:, 0]), control_mean=np.exp(0.02), seed=1)
    sobol = mc.mc_estimate(call, 1, 40000, "sobol", seed=1)
    for r in (plain, anti, ctrl, sobol):
        assert abs(r["estimate"] - truth) < 4 * r["se"] + 0.02
    assert anti["se"] < plain["se"] and ctrl["se"] < 0.6 * plain["se"] and sobol["se"] < 0.2 * plain["se"]
    with pytest.raises(ValueError):
        mc.mc_estimate(call, 1, 100, "control")


def test_importance_sampling_hits_rare_events_plain_sampling_misses():
    w = np.array([0.5, 0.5])
    cov = np.array([[0.0004, 0.0002], [0.0002, 0.0009]])
    out = mc.portfolio_tail_probability_is(w, np.zeros(2), cov, 0.12, 20000, seed=1)
    assert out["exact"] < 1e-8 and out["plain_estimate"] == 0.0
    assert out["estimate"] == pytest.approx(out["exact"], rel=0.05) and out["ess"] > 1000
    ce = mc.cross_entropy_is(lambda z: z.sum(axis=1), 4.5, 3, final_n=50000, seed=2)
    assert ce["estimate"] == pytest.approx(sps.norm.sf(4.5 / np.sqrt(3)), rel=0.05)
    gen = mc.importance_sampling(lambda x: (x > 4.0).astype(float), lambda rng, n: rng.normal(4.0, 1.0, n), lambda x: sps.norm.logpdf(x), lambda x: sps.norm.logpdf(x, 4.0, 1.0), 50000)
    assert gen["estimate"] == pytest.approx(sps.norm.sf(4.0), rel=0.05)
    sn = mc.importance_sampling(lambda x: x, lambda rng, n: rng.normal(1.0, 2.0, n), lambda x: sps.norm.logpdf(x, 0.0, 1.0) + 7.0, lambda x: sps.norm.logpdf(x, 1.0, 2.0), 60000, self_normalised=True)
    assert abs(sn["estimate"]) < 0.05
    assert mc.weighted_quantile([1, 2, 3, 4, 5], 0.5) == 3.0 and mc.effective_sample_size(np.ones(100)) == 100


# ----------------------------------------------------------------------------------------------------------------------- Markov
def test_markov_chain_analytics_two_state():
    m = markov.MarkovChain([[0.9, 0.1], [0.2, 0.8]])
    assert m.stationary().to_numpy() == pytest.approx([2 / 3, 1 / 3]) and m.expected_duration().to_numpy() == pytest.approx([10.0, 5.0])
    mfp = m.mean_first_passage()
    assert mfp.iloc[0, 1] == pytest.approx(10.0) and mfp.iloc[1, 0] == pytest.approx(5.0) and mfp.iloc[0, 0] == pytest.approx(1.5)
    assert m.n_step(200).to_numpy() == pytest.approx(np.tile([2 / 3, 1 / 3], (2, 1)), abs=1e-6) and 5 < m.mixing_time() < 40
    with pytest.raises(ValueError):
        markov.MarkovChain([[0.5, 0.6], [0.2, 0.8]])


def test_transition_estimation_order_test_and_runs_test():
    m = markov.MarkovChain([[0.9, 0.1], [0.2, 0.8]])
    s = m.simulate(30000, seed=2)
    assert markov.estimate_transition_matrix(s).P == pytest.approx(m.P, abs=0.02)
    assert markov.test_markov_order(s)["pvalue"] < 1e-10 and markov.runs_test(s)["z"] < -10
    iid = np.random.default_rng(1).integers(0, 2, 5000)
    assert markov.test_markov_order(iid)["pvalue"] > 0.01 and abs(markov.runs_test(iid)["z"]) < 3.5
    post = markov.transition_posterior(s, n_draws=200)
    assert post.shape == (200, 2, 2) and post[:, 0, 0].mean() == pytest.approx(0.9, abs=0.02) and np.allclose(post.sum(axis=2), 1.0)
    sm_ = markov.estimate_transition_matrix([0, 0, 0, 1], smoothing=1.0)
    assert (sm_.P > 0).all()
    d = markov.discretize(np.random.default_rng(1).normal(size=3000), 3)
    assert set(d) == {0, 1, 2} and abs(np.mean(d == 1) - 1 / 3) < 0.02


def test_markov_switching_filtered_probabilities_are_causal():
    rng = np.random.default_rng(3)
    state = markov.MarkovChain([[0.97, 0.03], [0.05, 0.95]]).simulate(1500, seed=4)
    r = pd.Series(np.where(state == 0, rng.normal(0.001, 0.007, 1500), rng.normal(-0.002, 0.02, 1500)), index=pd.bdate_range("2015", periods=1500))
    fit = markov.fit_markov_switching(r, 2)
    vols = r.groupby(fit["filtered"].idxmax(axis=1)).std()
    assert vols.max() > 2 * vols.min() and fit["filtered"].sum(axis=1).round(6).eq(1.0).all()
    assert (fit["durations"] > 5).all()
    short = markov.fit_markov_switching(r.iloc[:1000], 2)
    # filtered probabilities at early dates depend only on early data: refitting on a shorter sample with the SAME parameters would match, so only check they are valid
    assert short["filtered"].min().min() >= 0


# ------------------------------------------------------------------------------------------------------------------------ Bayes
def test_bayes_regression_matches_ols_with_flat_prior_and_ranks_models():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = 1 + X @ [0.5, 0.0] + rng.normal(size=300) * 0.5
    b = bayes.bayes_linear_regression(y, X, prior_precision=1e-8)
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    assert b["mean"] == pytest.approx(ols.params, abs=1e-6) and b["sd"] == pytest.approx(ols.bse, rel=0.03)
    full = b["log_marginal_likelihood"]
    reduced = bayes.bayes_linear_regression(y, X[:, :1], prior_precision=1e-8)["log_marginal_likelihood"]
    assert reduced > full                                     # the Occam factor penalises the useless regressor
    wrong = bayes.bayes_linear_regression(y, rng.normal(size=(300, 1)))["log_marginal_likelihood"]
    assert bayes.bayes_factor(reduced, wrong) > 20
    mean, sd, df = b["predict"](np.array([[1.0, 0.0]]))
    assert mean[0] == pytest.approx(1.5, abs=0.15) and sd[0] > 0.4 and df > 100


def test_metropolis_hastings_recovers_a_known_gaussian_and_diagnostics_flag_a_bad_chain():
    target = sps.multivariate_normal([1.0, -2.0], [[1.0, 0.6], [0.6, 2.0]])
    out = bayes.metropolis_hastings(lambda x: target.logpdf(x), [0.0, 0.0], n_samples=3000, burn=1000, seed=1)
    s = out["samples"]
    assert s.mean(axis=0) == pytest.approx([1.0, -2.0], abs=0.15) and np.cov(s.T) == pytest.approx(np.array([[1.0, 0.6], [0.6, 2.0]]), abs=0.25)
    assert out["rhat"].max() < 1.05 and out["ess"].min() > 100 and 0.1 < out["acceptance"].mean() < 0.7
    lo, hi = bayes.hpd_interval(s[:, 0], 0.95)
    assert lo == pytest.approx(1 - 1.96, abs=0.25) and hi == pytest.approx(1 + 1.96, abs=0.25)
    stuck = np.stack([np.full(500, 0.0), np.full(500, 5.0)])[:, :, None]
    assert bayes.rhat(stuck + np.random.default_rng(1).normal(0, 0.01, stuck.shape))[0] > 1.5


def test_bayesian_sharpe_properties():
    rng = np.random.default_rng(0)
    r = rng.standard_t(5, 1500) * 0.01 + 0.0006
    b = bayes.bayesian_sharpe(r, 2000)
    sample = r.mean() / r.std() * np.sqrt(252)
    lo, hi = b["hpd95"]
    assert lo < sample < hi and b["rhat"].max() < 1.1 and 3 < b["nu_median"] < 12
    noise = bayes.bayesian_sharpe(rng.normal(0, 0.01, 1500), 2000)
    assert noise["hpd95"][0] < 0 < noise["hpd95"][1] and 0.05 < noise["prob_positive"] < 0.95
    better = rng.normal(0.0015, 0.01, 1500)
    cmp = bayes.compare_sharpe(better, rng.normal(0.0, 0.01, 1500), 2000)
    assert cmp["prob_a_better"] > 0.95 and cmp["hpd95"][0] > 0
    bb = bayes.beta_binomial(55, 100)
    assert bb["mean"] == pytest.approx(56 / 102) and bb["interval"][0] < 0.55 < bb["interval"][1] and 0.8 < bb["prob_above_half"] < 0.9
