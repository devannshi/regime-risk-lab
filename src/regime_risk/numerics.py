"""Scaled forward/backward recursions. No external HMM implementation is used."""

import numpy as np
from numba import njit


@njit(cache=True)
def forward_scaled(log_emissions, transition, initial):
    """Return filtered probabilities, predictive probabilities and log increments.

    ``initial`` describes the state at the FIRST supplied observation. It is not
    a posterior for a preceding observation; callers bridging segments must apply A.
    """
    n, k = log_emissions.shape
    filtered = np.empty((n, k))
    predicted = np.empty((n, k))
    increments = np.empty(n)
    for t in range(n):
        for j in range(k):
            predicted[t, j] = initial[j] if t == 0 else 0.0
            if t > 0:
                for i in range(k):
                    predicted[t, j] += filtered[t - 1, i] * transition[i, j]
        # Include predictive support in the shift: an unreachable state with a
        # large emission density must not cause every reachable weight to vanish.
        log_joint = np.full(k, -np.inf)
        for j in range(k):
            if predicted[t, j] > 0:
                log_joint[j] = np.log(predicted[t, j]) + log_emissions[t, j]
        offset = np.max(log_joint)
        denom = 0.0
        for j in range(k):
            filtered[t, j] = np.exp(log_joint[j] - offset)
            denom += filtered[t, j]
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError("Predictive density underflow: invalid or degenerate parameters")
        increments[t] = np.log(denom) + offset
        filtered[t] /= denom
    return filtered, predicted, increments


@njit(cache=True)
def smooth_scaled(filtered, predicted, transition):
    """Stable backward smoothing and expected transition counts.

    Uses p(z_t=i | z_{t+1}=j, y_0:t) = alpha_t(i) A_ij / pred_{t+1}(j).
    No future-aware probabilities escape into the forecasting API.
    """
    n, k = filtered.shape
    smoothed = filtered.copy()
    counts = np.zeros((k, k))
    for t in range(n - 2, -1, -1):
        for i in range(k):
            smoothed[t, i] = 0.0
            for j in range(k):
                if predicted[t + 1, j] > 0:
                    pair = (
                        filtered[t, i] * transition[i, j] * smoothed[t + 1, j] / predicted[t + 1, j]
                    )
                    smoothed[t, i] += pair
                    counts[i, j] += pair
        smoothed[t] /= smoothed[t].sum()
    return smoothed, counts
