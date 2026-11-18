"""Univariate Gaussian and fixed-degrees-of-freedom Student-t HMMs.

All inference and EM updates are original implementations of standard formulas.
State numbers are ordered by unconditional emission variance, not economic labels.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import norm
from scipy.stats import t as student_t

from .numerics import forward_scaled, smooth_scaled


def _observations(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite one-dimensional observation sequence")
    return np.ascontiguousarray(values)


@dataclass
class Parameters:
    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def validate(self):
        self.initial = np.asarray(self.initial, float)
        self.transition = np.asarray(self.transition, float)
        self.means = np.asarray(self.means, float)
        self.scales = np.asarray(self.scales, float)
        k = len(self.initial)
        if (
            self.initial.shape != (k,)
            or self.transition.shape != (k, k)
            or self.means.shape != (k,)
            or self.scales.shape != (k,)
        ):
            raise ValueError("Parameter shapes disagree")
        if not all(
            np.isfinite(x).all() for x in [self.initial, self.transition, self.means, self.scales]
        ):
            raise ValueError("Parameters must be finite")
        if np.any(self.scales <= 0) or np.any(self.initial < 0) or np.any(self.transition < 0):
            raise ValueError("Scales must be positive and probabilities nonnegative")
        if not np.isclose(self.initial.sum(), 1) or not np.allclose(self.transition.sum(1), 1):
            raise ValueError("Initial and transition probabilities must sum to one")
        return self


class HiddenMarkovModel:
    """Maximum-likelihood HMM with multiple starts and numerical safeguards.

    Student-t ``scales`` are scale parameters, NOT standard deviations. Their
    variance is scale**2 * df/(df-2). df is fixed before evaluating holdout data.
    """

    def __init__(
        self,
        n_states=2,
        emission="gaussian",
        df=5.0,
        n_init=4,
        max_iter=150,
        tol=1e-6,
        variance_floor=1e-6,
        random_state=2026,
    ):
        if n_states < 1 or int(n_states) != n_states:
            raise ValueError("n_states must be a positive integer")
        if emission not in {"gaussian", "student"}:
            raise ValueError("emission must be gaussian or student")
        if df <= 2 or n_init < 1 or max_iter < 1 or tol <= 0 or variance_floor <= 0:
            raise ValueError("Invalid optimization configuration (df must exceed 2)")
        self.n_states = int(n_states)
        self.emission = emission
        self.df = float(df)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.variance_floor = float(variance_floor)
        self.random_state = random_state
        self.params = None

    @classmethod
    def from_parameters(cls, initial, transition, means, scales, emission="gaussian", df=5):
        model = cls(len(means), emission=emission, df=df)
        model.params = Parameters(initial, transition, means, scales).validate()
        return model

    def _check(self):
        if self.params is None:
            raise RuntimeError("Fit the model or supply parameters before inference")

    def _weights(self, weights, allow_matrix=False):
        weights = np.asarray(weights, dtype=float)
        allowed = (1, 2) if allow_matrix else (1,)
        if (
            weights.ndim not in allowed
            or weights.shape[-1] != self.n_states
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=-1), 1.0)
        ):
            raise ValueError("Weights must be finite, nonnegative, normalized state probabilities")
        return weights

    def log_emissions(self, values):
        self._check()
        values = _observations(values)
        z = (values[:, None] - self.params.means) / self.params.scales
        if self.emission == "gaussian":
            return -0.5 * (np.log(2 * np.pi) + z * z) - np.log(self.params.scales)
        nu = self.df
        return (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi)
            - np.log(self.params.scales)
            - 0.5 * (nu + 1) * np.log1p(z * z / nu)
        )

    def filter(self, values, initial=None):
        self._check()
        initial = self.params.initial if initial is None else np.asarray(initial, float)
        if (
            initial.shape != (self.n_states,)
            or np.any(initial < 0)
            or not np.isfinite(initial).all()
            or not np.isclose(initial.sum(), 1)
        ):
            raise ValueError("Invalid initial probability vector")
        return forward_scaled(self.log_emissions(values), self.params.transition, initial)

    def smooth(self, values):
        filtered, predicted, _ = self.filter(values)
        return smooth_scaled(filtered, predicted, self.params.transition)[0]

    def score(self, values, initial=None):
        return float(self.filter(values, initial)[2].sum())

    def viterbi(self, values):
        """Offline MAP path, deliberately excluded from walk-forward forecasts."""
        logs = self.log_emissions(values)
        with np.errstate(divide="ignore"):
            log_a = np.log(self.params.transition)
            delta = np.log(self.params.initial) + logs[0]
        pointers = np.zeros_like(logs, dtype=int)
        for t in range(1, len(logs)):
            scores = delta[:, None] + log_a
            pointers[t] = scores.argmax(0)
            delta = scores.max(0) + logs[t]
        path = np.empty(len(logs), dtype=int)
        path[-1] = delta.argmax()
        for t in range(len(logs) - 2, -1, -1):
            path[t] = pointers[t + 1, path[t + 1]]
        return path

    def fit(self, values):
        x = _observations(values)
        if len(x) < max(10, 4 * self.n_states):
            raise ValueError("Too few observations to estimate the model")
        rng = np.random.default_rng(self.random_state)
        best_score = -np.inf
        self.restart_diagnostics_ = []
        k = self.n_states
        base_var = max(float(np.var(x)), self.variance_floor)
        floor = self.variance_floor * base_var
        for restart in range(self.n_init):
            mean = np.repeat(np.mean(x), k) + rng.normal(0, np.sqrt(base_var) * 0.1, k)
            scales = np.sqrt(base_var) * np.geomspace(0.45, 1.8, k)
            if restart:
                scales *= np.exp(rng.normal(0, 0.3, k))
            transition = np.full((k, k), 0.04 / k) + np.eye(k) * 0.96
            params = Parameters(np.full(k, 1 / k), transition, mean, scales).validate()
            history = []
            converged = False
            stop_reason = "max_iter"
            for iteration in range(self.max_iter):  # noqa: B007 - used in restart diagnostics below
                self.params = params
                filtered, predicted, increments = self.filter(x)
                score = float(increments.sum())
                history.append(score)
                gamma, counts = smooth_scaled(filtered, predicted, params.transition)
                weights = np.maximum(gamma.sum(0), 1e-12)
                if self.emission == "student":
                    standardized = (x[:, None] - params.means) / params.scales
                    u = (self.df + 1) / (self.df + standardized**2)
                    mu = (gamma * u * x[:, None]).sum(0) / (gamma * u).sum(0)
                    var = (gamma * u * (x[:, None] - mu) ** 2).sum(0) / weights
                else:
                    mu = (gamma * x[:, None]).sum(0) / weights
                    var = (gamma * (x[:, None] - mu) ** 2).sum(0) / weights
                # Floor avoids zero-density transitions; acceptance below guards EM monotonicity.
                new_a = np.maximum(counts, 1e-12)
                new_a /= new_a.sum(1, keepdims=True)
                new_pi = np.maximum(gamma[0], 1e-12)
                new_pi /= new_pi.sum()
                candidate = Parameters(new_pi, new_a, mu, np.sqrt(np.maximum(var, floor)))
                self.params = candidate
                candidate_score = self.score(x)
                if candidate_score < score - 1e-7:
                    self.params = params
                    stop_reason = "safeguard_rejected_step"
                    break
                params = candidate
                if abs(candidate_score - score) <= self.tol * (1 + abs(score)):
                    history.append(candidate_score)
                    converged = True
                    stop_reason = "relative_likelihood_tolerance"
                    break
            self.params = params
            final_score = self.score(x)
            self.restart_diagnostics_.append(
                {
                    "restart": restart,
                    "log_likelihood": final_score,
                    "iterations": iteration + 1,
                    "converged": converged,
                    "stop_reason": stop_reason,
                }
            )
            if final_score > best_score:
                best_score = final_score
                best_params = params
                best_history = history
                best_converged = converged
                best_iterations = iteration + 1
        order = np.argsort(best_params.scales)
        self.params = Parameters(
            best_params.initial[order],
            best_params.transition[np.ix_(order, order)],
            best_params.means[order],
            best_params.scales[order],
        ).validate()
        self.log_likelihood_ = best_score
        self.history_ = best_history
        self.converged_ = best_converged
        self.n_iter_ = best_iterations
        return self

    @property
    def variances(self):
        self._check()
        factor = self.df / (self.df - 2) if self.emission == "student" else 1.0
        return self.params.scales**2 * factor

    def forecast(self, posterior):
        """One-step predictive state weights and moments before seeing the outcome."""
        self._check()
        weights = self._weights(posterior) @ self.params.transition
        mean = float(weights @ self.params.means)
        variance = float(weights @ (self.variances + self.params.means**2) - mean**2)
        return weights, mean, variance

    def mixture_cdf(self, values, weights):
        self._check()
        weights = self._weights(weights)
        values = np.asarray(values)
        z = (values[..., None] - self.params.means) / self.params.scales
        cdf = student_t.cdf(z, self.df) if self.emission == "student" else norm.cdf(z)
        return cdf @ weights

    def mixture_quantile(self, level, weights):
        """Vectorized bisection of mixture CDF for one or many weight rows."""
        if not 0 < level < 1:
            raise ValueError("Quantile level must be strictly between zero and one")
        self._check()
        weights = self._weights(weights, allow_matrix=True)
        is_single = weights.ndim == 1
        weights = np.atleast_2d(weights)
        dist_q = student_t.ppf(level, self.df) if self.emission == "student" else norm.ppf(level)
        component_q = self.params.means + self.params.scales * dist_q
        lo = np.full(len(weights), np.min(component_q) - 1e-12)
        hi = np.full(len(weights), np.max(component_q) + 1e-12)
        for _ in range(45):
            mid = (lo + hi) / 2
            z = (mid[:, None] - self.params.means) / self.params.scales
            vals = student_t.cdf(z, self.df) if self.emission == "student" else norm.cdf(z)
            below = np.sum(vals * weights, axis=1) < level
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        result = (lo + hi) / 2
        return float(result[0]) if is_single else result

    def predictive_log_density(self, value, weights):
        self._check()
        weights = self._weights(weights, allow_matrix=True)
        observations = np.atleast_1d(value)
        if weights.ndim == 2 and weights.shape[0] != len(observations):
            raise ValueError("A weight matrix must have one row per observation")
        with np.errstate(divide="ignore"):
            return logsumexp(self.log_emissions(observations) + np.log(weights), axis=1)

    def to_dict(self):
        self._check()
        return {
            "emission": self.emission,
            "df": self.df,
            "n_states": self.n_states,
            "initial": self.params.initial.tolist(),
            "transition": self.params.transition.tolist(),
            "means": self.params.means.tolist(),
            "scales": self.params.scales.tolist(),
        }
