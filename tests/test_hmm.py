import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm, t

from regime_risk.data import simulate_regimes
from regime_risk.hmm import HiddenMarkovModel, Parameters


@pytest.fixture
def model():
    return HiddenMarkovModel.from_parameters(
        [0.65, 0.35], [[0.9, 0.1], [0.2, 0.8]], [-0.5, 1.2], [0.6, 1.5]
    )


def enumerate_paths(model, x):
    p = model.params
    paths = list(itertools.product(range(2), repeat=len(x)))
    weights = []
    for path in paths:
        w = p.initial[path[0]]
        for i, state in enumerate(path):
            w *= norm.pdf(x[i], p.means[state], p.scales[state])
            if i:
                w *= p.transition[path[i - 1], state]
        weights.append(w)
    weights = np.array(weights)
    smooth = np.zeros((len(x), 2))
    for path, w in zip(paths, weights / weights.sum(), strict=True):
        for i, state in enumerate(path):
            smooth[i, state] += w
    return np.log(weights.sum()), smooth, paths[int(weights.argmax())]


def test_forward_matches_exhaustive_path_sum(model):
    x = np.array([0.1, -0.4, 2.0, 0.3])
    score, _, _ = enumerate_paths(model, x)
    assert_allclose(model.score(x), score, atol=1e-12)


def test_smoother_matches_exhaustive_marginals(model):
    x = np.array([0.1, -0.4, 2.0, 0.3])
    _, smooth, _ = enumerate_paths(model, x)
    assert_allclose(model.smooth(x), smooth, atol=1e-12)


def test_viterbi_matches_exhaustive_map(model):
    x = np.array([0.1, -0.4, 2.0, 0.3])
    _, _, best = enumerate_paths(model, x)
    assert tuple(model.viterbi(x)) == best


def test_filter_prefix_invariance(model):
    x = np.array([0.1, -0.4, 2.0, 0.3, 10.0, -20.0])
    assert_allclose(model.filter(x)[0][:4], model.filter(x[:4])[0])


def test_segment_bridge_exact(model):
    x = np.array([0.1, -0.4, 2.0, 0.3, 10.0, -20.0])
    first = model.filter(x[:3])
    next_prior = first[0][-1] @ model.params.transition
    second = model.filter(x[3:], initial=next_prior)
    full = model.filter(x)
    assert_allclose(second[0], full[0][3:])
    assert_allclose(first[2].sum() + second[2].sum(), full[2].sum())


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_emission_density_matches_scipy(family):
    m = HiddenMarkovModel.from_parameters([1.0], [[1.0]], [1.2], [0.7], emission=family)
    x = np.linspace(-10, 10, 101)
    ref = norm.logpdf(x, 1.2, 0.7) if family == "gaussian" else t.logpdf(x, 5, 1.2, 0.7)
    assert_allclose(m.log_emissions(x)[:, 0], ref, atol=1e-12)


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_em_is_monotonic_and_reproducible(family):
    x, _ = simulate_regimes(700, 74, family)
    m = HiddenMarkovModel(emission=family, n_init=2, max_iter=100).fit(x / 0.01)
    other = HiddenMarkovModel(emission=family, n_init=2, max_iter=100).fit(x / 0.01)
    assert np.min(np.diff(m.history_)) > -1e-7
    assert_allclose(m.params.transition.sum(1), 1)
    assert np.all(np.diff(m.params.scales) >= 0)
    assert_allclose(m.params.scales, other.params.scales)
    assert_allclose(m.log_likelihood_, m.score(x / 0.01), atol=1e-9)


def test_one_state_gaussian_recovers_closed_form():
    x = np.random.default_rng(24).normal(2, 3, 500)
    m = HiddenMarkovModel(n_states=1, n_init=1).fit(x)
    assert_allclose(m.params.means, [x.mean()], atol=1e-12)
    assert_allclose(m.params.scales, [x.std()], atol=1e-12)


def test_variance_student_is_not_scale_squared():
    m = HiddenMarkovModel.from_parameters([1.0], [[1.0]], [0.0], [2.0], emission="student")
    assert_allclose(m.variances, [20 / 3])


def test_forecast_includes_between_state_variance(model):
    weights, mu, variance = model.forecast([0.5, 0.5])
    assert_allclose(weights, [0.55, 0.45])
    assert_allclose(variance, weights @ (model.params.scales**2 + (model.params.means - mu) ** 2))


@pytest.mark.parametrize("family", ["gaussian", "student"])
@pytest.mark.parametrize("level", [0.01, 0.05, 0.5, 0.95])
def test_quantile_inverts_mixture(family, level):
    m = HiddenMarkovModel.from_parameters(
        [0.5, 0.5], [[0.9, 0.1], [0.1, 0.9]], [-0.2, 0.4], [0.5, 1.7], emission=family
    )
    w = np.array([0.7, 0.3])
    q = m.mixture_quantile(level, w)
    assert_allclose(m.mixture_cdf(q, w), level, atol=1e-10)
    assert_allclose(m.mixture_quantile(level, np.tile(w, (3, 1))), q)


def test_state_permutation_invariance(model):
    p = model.params
    flipped = HiddenMarkovModel.from_parameters(
        p.initial[::-1], p.transition[::-1, ::-1], p.means[::-1], p.scales[::-1]
    )
    assert_allclose(model.score([0.1, -0.2, 3.0]), flipped.score([0.1, -0.2, 3.0]))


def test_extreme_observations_do_not_underflow(model):
    post, pred, logs = model.filter([1e4, -1e4, 1e3])
    assert np.isfinite(logs).all()
    assert_allclose(post.sum(1), 1)
    assert_allclose(pred.sum(1), 1)


def test_hmmlearn_independent_oracle(model):
    hmm = pytest.importorskip("hmmlearn.hmm")
    external = hmm.GaussianHMM(n_components=2, covariance_type="diag", init_params="")
    external.startprob_ = model.params.initial
    external.transmat_ = model.params.transition
    external.means_ = model.params.means[:, None]
    external.covars_ = model.params.scales[:, None] ** 2
    x = np.random.default_rng(8).normal(size=200)
    assert_allclose(model.score(x), external.score(x[:, None]), atol=1e-10)
    assert_allclose(model.smooth(x), external.predict_proba(x[:, None]), atol=1e-10)


@pytest.mark.parametrize("bad", [[], [np.nan], [[1, 2]], [np.inf]])
def test_bad_observations_rejected(model, bad):
    with pytest.raises(ValueError):
        model.filter(bad)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_states": 0}, {"emission": "poisson"}, {"df": 2}, {"n_init": 0}, {"variance_floor": 0}],
)
def test_bad_model_config(kwargs):
    with pytest.raises(ValueError):
        HiddenMarkovModel(**kwargs)


def test_invalid_probability_matrix_rejected():
    with pytest.raises(ValueError):
        Parameters([1.0], [[0.5]], [0.0], [1.0]).validate()


def test_unreachable_high_density_state_cannot_cause_underflow():
    model = HiddenMarkovModel.from_parameters([1, 0], [[1, 0], [0, 1]], [0, 100], [1, 1])
    assert_allclose(model.score([100]), norm.logpdf(100), atol=1e-10)
    assert_allclose(model.filter([100, 100])[0], [[1, 0], [1, 0]])


@pytest.mark.parametrize("weights", [[-0.1, 1.1], [np.nan, 0.5], [0.2, 0.2], [1.0], [[0.5, 0.5]]])
def test_forecast_rejects_invalid_weights(model, weights):
    with pytest.raises(ValueError):
        model.forecast(weights)


def test_quantile_and_density_reject_misaligned_weights(model):
    with pytest.raises(ValueError):
        model.mixture_quantile(0.05, [[0.2, 0.2]])
    with pytest.raises(ValueError):
        model.predictive_log_density([1, 2], [[0.5, 0.5]])
