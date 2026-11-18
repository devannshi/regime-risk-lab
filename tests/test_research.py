import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from regime_risk.data import load_returns, simulate_regimes
from regime_risk.evaluation import (
    adaptive_quantiles,
    coverage_tests,
    moving_block_interval,
    quantile_loss,
)
from regime_risk.walkforward import WalkForwardConfig, walk_forward


def test_walk_forward_future_perturbation_does_not_change_past():
    x, _ = simulate_regimes(200, 20)
    cfg = WalkForwardConfig(train_size=100, refit_every=50, n_init=1, max_iter=40)
    first, fits = walk_forward(x, config=cfg, families=("gaussian",))
    changed = x.copy()
    changed[170:] += 10
    second, _ = walk_forward(changed, config=cfg, families=("gaussian",))
    a = first[first["index"] < 170].reset_index(drop=True)
    b = second[second["index"] < 170].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert all(row["train_end_exclusive"] == row["test_start"] for row in fits)


def test_current_outcome_does_not_change_current_forecast():
    x, _ = simulate_regimes(120, 4)
    cfg = WalkForwardConfig(train_size=100, refit_every=20, n_init=1, max_iter=40)
    a, _ = walk_forward(x, config=cfg, families=("gaussian",))
    x[100] = 10
    b, _ = walk_forward(x, config=cfg, families=("gaussian",))
    assert_allclose(
        a[a["index"] == 100][["mean", "variance", "q05"]],
        b[b["index"] == 100][["mean", "variance", "q05"]],
    )


def test_online_controller_has_no_current_or_future_leakage():
    y = np.zeros(100)
    q, sd = np.full(100, -0.1), np.ones(100)
    a = adaptive_quantiles(y, q, sd)
    y[50:] = -10
    b = adaptive_quantiles(y, q, sd)
    assert_allclose(a[:51], b[:51])
    assert b[-1] < a[-1]


def test_quantile_loss_known_values():
    assert_allclose(quantile_loss([1.0, -1.0], [0.0, 0.0], 0.05), [0.05, 0.95])


def test_coverage_perfect_nominal_frequency():
    y = np.r_[np.full(5, -2.0), np.zeros(95)]
    result = coverage_tests(y, np.full(100, -1.0), 0.05)
    assert result["breaches"] == 5
    assert result["kupiec_p"] == pytest.approx(1)
    assert result["independence_p"] < 0.01  # clustered violations are caught


def test_coverage_zero_violations_finite():
    result = coverage_tests(np.zeros(100), np.full(100, -1.0))
    assert np.isfinite(result["kupiec_p"])
    assert np.isnan(result["independence_p"])


def test_block_bootstrap_is_seeded_and_preserves_constant():
    result = moving_block_interval(np.ones(60), n_boot=50)
    assert result["lower"] == result["upper"] == 1
    x = np.random.default_rng(2).normal(size=200)
    assert moving_block_interval(x, n_boot=50) == moving_block_interval(x, n_boot=50)


def test_loader_does_not_fill_missing_prices(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4),
            "EURUSD": [1.0, np.nan, 2.0, 4.0],
            "GBPUSD": [1.0, 2.0, 3.0, 4.0],
            "JPYUSD": [0.1, 0.2, 0.3, 0.4],
        }
    )
    path = tmp_path / "prices.csv"
    df.to_csv(path, index=False)
    result = load_returns(path)
    assert len(result["EURUSD"]) == 2
    assert_allclose(result["EURUSD"]["return"], np.log(2))


def test_loader_rejects_reordered_dates(tmp_path):
    df = pd.DataFrame(
        {"date": ["2020-01-02", "2020-01-01"], "EURUSD": [1, 2], "GBPUSD": [1, 2], "JPYUSD": [1, 2]}
    )
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_returns(p)


def test_simulator_is_reproducible():
    a, z1 = simulate_regimes(200, 5, "contaminated")
    b, z2 = simulate_regimes(200, 5, "contaminated")
    assert_allclose(a, b)
    assert_allclose(z1, z2)
