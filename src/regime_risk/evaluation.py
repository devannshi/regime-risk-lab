"""Forecast scores, tail calibration and dependence-aware uncertainty."""

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2


def quantile_loss(y, q, level):
    residual = np.asarray(y) - np.asarray(q)
    return np.maximum(level * residual, (level - 1) * residual)


def coverage_tests(y, q, level=0.05):
    """Kupiec unconditional and Christoffersen independence LR tests.

    These are asymptotic diagnostics, not guarantees, and can be weak with few
    tail events. Degenerate transition tables return NaN for independence.
    """
    hit = np.asarray(y) < np.asarray(q)
    n, count = len(hit), int(hit.sum())
    if n == 0 or not 0 < level < 1:
        raise ValueError("Need observations and 0 < level < 1")
    empirical = count / n
    ll_null = xlogy(count, level) + xlogy(n - count, 1 - level)
    ll_fit = xlogy(count, empirical) + xlogy(n - count, 1 - empirical)
    lr_uc = max(0.0, 2 * float(ll_fit - ll_null))
    pairs = np.zeros((2, 2), int)
    for a, b in zip(hit[:-1], hit[1:], strict=True):
        pairs[int(a), int(b)] += 1
    total = pairs.sum()
    if total == 0 or np.any(pairs.sum(1) == 0):
        lr_ind, p_ind, p_cc = np.nan, np.nan, np.nan
    else:
        pooled = pairs[:, 1].sum() / total
        ll_iid = xlogy(pairs[:, 1].sum(), pooled) + xlogy(pairs[:, 0].sum(), 1 - pooled)
        p01, p11 = pairs[:, 1] / pairs.sum(1)
        ll_markov = (
            xlogy(pairs[0, 1], p01)
            + xlogy(pairs[0, 0], 1 - p01)
            + xlogy(pairs[1, 1], p11)
            + xlogy(pairs[1, 0], 1 - p11)
        )
        lr_ind = max(0.0, 2 * float(ll_markov - ll_iid))
        p_ind, p_cc = float(chi2.sf(lr_ind, 1)), float(chi2.sf(lr_uc + lr_ind, 2))
    return {
        "n": n,
        "breaches": count,
        "breach_rate": empirical,
        "kupiec_p": float(chi2.sf(lr_uc, 1)),
        "independence_p": p_ind,
        "conditional_coverage_p": p_cc,
        "lr_uc": lr_uc,
        "lr_ind": lr_ind,
    }


def moving_block_interval(values, block=20, n_boot=2000, seed=931):
    """Circular block bootstrap interval for a mean (paired differences preferred)."""
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or block < 1 or n_boot < 2:
        raise ValueError("Need finite values, block >= 1, and n_boot >= 2")
    rng = np.random.default_rng(seed)
    n = len(x)
    block = min(int(block), n)
    means = np.empty(n_boot)
    offsets = np.arange(block)
    for b in range(n_boot):
        starts = rng.integers(0, n, int(np.ceil(n / block)))
        idx = ((starts[:, None] + offsets) % n).ravel()[:n]
        means[b] = x[idx].mean()
    return {
        "mean": float(x.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "block": block,
        "n_boot": n_boot,
    }


def adaptive_quantiles(y, base_quantile, predicted_sd, level=0.05, learning_rate=0.02):
    """Past-only stochastic quantile calibration in standardized residual units.

    q_t = base_q_t + c_t sigma_t;
    c_{t+1} = clip(c_t + eta*(alpha - 1[y_t < q_t]), -4, 4).
    A pragmatic online controller, NOT distribution-free conditional coverage.
    """
    y, q, sd = map(lambda a: np.asarray(a, float), (y, base_quantile, predicted_sd))
    if not (y.shape == q.shape == sd.shape) or np.any(sd <= 0):
        raise ValueError("Shapes must agree and standard deviations must be positive")
    out = np.empty_like(q)
    correction = 0.0
    for i in range(len(y)):
        out[i] = q[i] + correction * sd[i]
        correction = float(np.clip(correction + learning_rate * (level - (y[i] < out[i])), -4, 4))
    return out
