"""Chronological rolling refits with one-step-ahead, forward-only forecasts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from .hmm import HiddenMarkovModel


@dataclass(frozen=True)
class WalkForwardConfig:
    train_size: int = 1250
    refit_every: int = 252
    n_states: int = 2
    n_init: int = 4
    max_iter: int = 150
    seed: int = 20261117


def walk_forward(
    values, dates=None, config=None, families=("gaussian_iid", "student_iid", "gaussian", "student")
):
    """Return one forecast per observed return and model, plus model-fit audit rows.

    Every fold trains on [start-train_size, start), then forecasts [start, end).
    Parameters remain fixed inside a fold; state beliefs update after each outcome.
    Standardization uses training data only and density scores undo the Jacobian.
    """
    cfg = config or WalkForwardConfig()
    x = np.asarray(values, float)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) <= cfg.train_size:
        raise ValueError("Need a finite sequence longer than the training window")
    if cfg.train_size < 20 or cfg.refit_every < 1:
        raise ValueError("Invalid chronological split configuration")
    dates = np.asarray(dates if dates is not None else np.arange(len(x)))
    if len(dates) != len(x):
        raise ValueError("Dates and observations differ in length")
    predictions, fits = [], []
    for fold, start in enumerate(range(cfg.train_size, len(x), cfg.refit_every)):
        end = min(start + cfg.refit_every, len(x))
        train = x[start - cfg.train_size : start]
        center, scale = train.mean(), max(train.std(ddof=0), 1e-8)
        z_train, z_test = (train - center) / scale, (x[start:end] - center) / scale
        for family in families:
            is_iid = family.endswith("_iid")
            emission = family.removesuffix("_iid")
            name = family if is_iid else f"{family}_hmm"
            model = HiddenMarkovModel(
                1 if is_iid else cfg.n_states,
                emission=emission,
                n_init=1 if is_iid else cfg.n_init,
                max_iter=cfg.max_iter,
                random_state=cfg.seed + fold,
            ).fit(z_train)
            train_post = model.filter(z_train)[0][-1]
            post, pred, logs = model.filter(z_test, initial=train_post @ model.params.transition)
            means_z = pred @ model.params.means
            var_z = pred @ (model.variances + model.params.means**2) - means_z**2
            q05 = center + scale * model.mixture_quantile(0.05, pred)
            q01 = center + scale * model.mixture_quantile(0.01, pred)
            cdf_components = (
                student_t.cdf(
                    (z_test[:, None] - model.params.means) / model.params.scales, model.df
                )
                if emission == "student"
                else norm.cdf((z_test[:, None] - model.params.means) / model.params.scales)
            )
            pits = (cdf_components * pred).sum(1)
            for i in range(end - start):
                predictions.append(
                    {
                        "index": start + i,
                        "date": str(dates[start + i]),
                        "model": name,
                        "fold": fold,
                        "return": x[start + i],
                        "mean": center + scale * means_z[i],
                        "variance": scale**2 * var_z[i],
                        "q05": q05[i],
                        "q01": q01[i],
                        "log_score": logs[i] - np.log(scale),
                        "pit": pits[i],
                        "predicted_stress": np.nan if is_iid else pred[i, -1],
                        "filtered_stress": np.nan if is_iid else post[i, -1],
                    }
                )
            fits.append(
                {
                    "fold": fold,
                    "model": name,
                    "train_start": start - cfg.train_size,
                    "train_end_exclusive": start,
                    "test_start": start,
                    "test_end_exclusive": end,
                    "train_mean": center,
                    "train_sd": scale,
                    "converged": model.converged_,
                    "iterations": model.n_iter_,
                    "train_ll": model.log_likelihood_,
                    "params": model.to_dict(),
                    "restarts": model.restart_diagnostics_,
                }
            )
        # Two transparent volatility baselines, recalculated causally every observation.
        ewma = float(np.var(train))
        for past_return in train:
            ewma = 0.94 * ewma + 0.06 * past_return**2
        for i in range(start, end):
            trailing = x[max(0, i - 252) : i]
            baseline_specs = [
                ("rolling_gaussian", float(trailing.mean()), float(trailing.var())),
                ("ewma_gaussian", 0.0, ewma),
            ]
            for name, mean, variance in baseline_specs:
                sd = np.sqrt(max(variance, 1e-12))
                predictions.append(
                    {
                        "index": i,
                        "date": str(dates[i]),
                        "model": name,
                        "fold": fold,
                        "return": x[i],
                        "mean": mean,
                        "variance": sd**2,
                        "q05": mean + norm.ppf(0.05) * sd,
                        "q01": mean + norm.ppf(0.01) * sd,
                        "log_score": norm.logpdf(x[i], mean, sd),
                        "pit": norm.cdf(x[i], mean, sd),
                        "predicted_stress": np.nan,
                        "filtered_stress": np.nan,
                    }
                )
            # Follow-up benchmark: keep EWMA's variance and add Student-t tails.
            # Its scale must be reduced so df=5 still implies variance=ewma.
            t_scale = np.sqrt(max(ewma, 1e-12) * 3 / 5)
            predictions.append(
                {
                    "index": i,
                    "date": str(dates[i]),
                    "model": "ewma_student",
                    "fold": fold,
                    "return": x[i],
                    "mean": 0.0,
                    "variance": max(ewma, 1e-12),
                    "q05": student_t.ppf(0.05, 5) * t_scale,
                    "q01": student_t.ppf(0.01, 5) * t_scale,
                    "log_score": student_t.logpdf(x[i], 5, scale=t_scale),
                    "pit": student_t.cdf(x[i], 5, scale=t_scale),
                    "predicted_stress": np.nan,
                    "filtered_stress": np.nan,
                }
            )
            ewma = 0.94 * ewma + 0.06 * x[i] ** 2
    return pd.DataFrame(predictions), fits
