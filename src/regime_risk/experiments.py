"""Executable research protocol. Tables and figures are computed, never hand-entered."""

import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_returns, simulate_regimes
from .evaluation import adaptive_quantiles, coverage_tests, moving_block_interval, quantile_loss
from .hmm import HiddenMarkovModel
from .walkforward import WalkForwardConfig, walk_forward

COLORS = {
    "gaussian_iid": "#90a3b0",
    "student_iid": "#8e72a7",
    "gaussian_hmm": "#277da8",
    "student_hmm": "#008779",
    "ewma_gaussian": "#d27d2d",
    "rolling_gaussian": "#b95364",
    "ewma_student": "#6b5b35",
}


def plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfcfe",
            "axes.facecolor": "#fbfcfe",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#213547",
            "text.color": "#213547",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "font.size": 10,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
        }
    )


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _table(df):
    """Dependency-free markdown table."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in df.itertuples(index=False, name=None):
        cells = [f"{v:.5f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def synthetic_study(trials, out, n_train=1000, n_test=400):
    rows = []
    for kind_idx, kind in enumerate(["gaussian", "student", "contaminated", "break"]):
        for trial in range(trials):
            seed = 400000 + kind_idx * 10000 + trial
            x, states = simulate_regimes(n_train + n_test, seed, kind)
            train, test = x[:n_train], x[n_train:]
            scale = train.std()
            for family in ["gaussian_iid", "student_iid", "gaussian_hmm", "student_hmm"]:
                emission = family.split("_")[0]
                k = 1 if family.endswith("iid") else 2
                tick = time.perf_counter()
                model = HiddenMarkovModel(
                    k, emission, n_init=1 if k == 1 else 3, max_iter=150, random_state=seed
                ).fit(train / scale)
                past = model.filter(train / scale)[0][-1]
                posterior, pred, logs = model.filter(test / scale, past @ model.params.transition)
                q = scale * model.mixture_quantile(0.05, pred)
                means = pred @ model.params.means
                variance = scale**2 * (pred @ (model.variances + model.params.means**2) - means**2)
                adjusted = adaptive_quantiles(test, q, np.sqrt(variance))
                rows.append(
                    {
                        "dgp": kind,
                        "trial": trial,
                        "seed": seed,
                        "model": family,
                        "mean_log_score": float(np.mean(logs - np.log(scale))),
                        "quantile_loss": float(quantile_loss(test, q, 0.05).mean()),
                        "breach_rate": float(np.mean(test < q)),
                        "adaptive_breach_rate": float(np.mean(test < adjusted)),
                        "adaptive_quantile_loss": float(quantile_loss(test, adjusted, 0.05).mean()),
                        "brier_state": float(np.mean((posterior[:, -1] - states[n_train:]) ** 2))
                        if k == 2
                        else np.nan,
                        "state_accuracy": float(np.mean(posterior.argmax(1) == states[n_train:]))
                        if k == 2
                        else np.nan,
                        "converged": model.converged_,
                        "iterations": model.n_iter_,
                        "seconds": time.perf_counter() - tick,
                    }
                )
        print(f"Synthetic {kind}: {trials} paired seeds completed", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "synthetic_trials.csv", index=False)
    summaries = []
    for (kind, model), frame in result.groupby(["dgp", "model"]):
        rec = {"dgp": kind, "model": model, "trials": len(frame)}
        for metric in [
            "mean_log_score",
            "quantile_loss",
            "breach_rate",
            "adaptive_breach_rate",
            "adaptive_quantile_loss",
            "state_accuracy",
            "brier_state",
        ]:
            v = frame[metric].dropna().to_numpy()
            rec[metric] = float(v.mean()) if len(v) else np.nan
            rec[metric + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        rec["convergence_rate"] = frame.converged.mean()
        summaries.append(rec)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "synthetic_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    kinds = ["gaussian", "student", "contaminated", "break"]
    for ax, metric, title in zip(
        axes,
        ["mean_log_score", "breach_rate"],
        ["Predictive density: higher is better", "5% lower-tail breach rate"],
        strict=True,
    ):
        for i, name in enumerate(COLORS.keys()):
            if name not in result.model.unique():
                continue
            sub = summary[summary.model == name].set_index("dgp").loc[kinds]
            ax.errorbar(
                np.arange(4) + (i - 1.5) * 0.13,
                sub[metric],
                yerr=1.96 * sub[metric + "_se"],
                fmt="o",
                capsize=3,
                color=COLORS[name],
                label=name.replace("_", " "),
            )
        ax.set_xticks(range(4), ["Gaussian", "Heavy tails", "Contamination", "Scale break"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[1].axhline(0.05, color="#222", ls="--", lw=1, label="nominal 5%")
    axes[0].set_ylabel("Mean log density per observation")
    axes[1].set_ylabel("Violation fraction")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Known-state simulation · {trials} independent seeds per scenario", fontsize=15)
    _save(fig, out / "figures" / "synthetic_ablation.png")
    return result, summary


def real_study(data_path, out, config, quick=False):
    datasets = load_returns(data_path)
    if quick:
        datasets = {"EURUSD": datasets["EURUSD"].iloc[-1000:].reset_index(drop=True)}
    summaries, all_fits, paired, all_predictions = [], [], [], []
    for asset, data in datasets.items():
        predictions, fits = walk_forward(
            data["return"].to_numpy(), data.date.dt.strftime("%Y-%m-%d"), config
        )
        predictions["asset"] = asset
        for model, group in predictions.groupby("model", sort=False):
            idx = group.index
            q_adapt = adaptive_quantiles(group["return"], group.q05, np.sqrt(group.variance))
            predictions.loc[idx, "q05_adaptive"] = q_adapt
            rec = {
                "asset": asset,
                "model": model,
                "mean_log_score": group.log_score.mean(),
                "qlike": float(
                    np.mean(
                        np.log(group.variance)
                        + (group["return"] - group["mean"]) ** 2 / group.variance
                    )
                ),
                "q05_loss": float(quantile_loss(group["return"], group.q05, 0.05).mean()),
                "adaptive_q05_loss": float(quantile_loss(group["return"], q_adapt, 0.05).mean()),
                "adaptive_breach_rate": float(np.mean(group["return"] < q_adapt)),
            }
            rec.update(coverage_tests(group["return"], group.q05))
            rec["q01_breach_rate"] = float(np.mean(group["return"] < group.q01))
            summaries.append(rec)
        scores = predictions.pivot(index="index", columns="model", values="log_score")
        for first, second in [
            ("student_hmm", "gaussian_hmm"),
            ("student_hmm", "student_iid"),
            ("student_hmm", "ewma_gaussian"),
            ("gaussian_hmm", "gaussian_iid"),
            ("student_hmm", "ewma_student"),
        ]:
            interval = moving_block_interval(
                (scores[first] - scores[second]).to_numpy(), n_boot=300 if quick else 2000
            )
            paired.append({"asset": asset, "comparison": first + " minus " + second, **interval})
        for fit in fits:
            fit["asset"] = asset
        all_fits.extend(fits)
        all_predictions.append(predictions)
        print(f"Real data {asset}: {len(data)} observations; {len(fits)} model fits", flush=True)
    all_predictions = pd.concat(all_predictions, ignore_index=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "real_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_logscore_intervals.csv", index=False)
    all_predictions.to_csv(out / "forecasts.csv.gz", index=False, compression="gzip")
    (out / "fit_audit.json").write_text(json.dumps(all_fits, indent=2) + "\n")
    real_figures(all_predictions, summary, pd.DataFrame(paired), out)
    return all_predictions, summary, all_fits


def real_figures(predictions, summary, paired, out):
    figures = out / "figures"
    main = predictions[
        (predictions.asset == "EURUSD") & (predictions.model == "student_hmm")
    ].copy()
    dates = pd.to_datetime(main.date)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(dates, main["return"] * 100, color="#72869a", lw=0.7, label="Observed log return")
    axes[0].plot(dates, main.q05 * 100, color="#008779", lw=0.8, label="One-step 5% lower quantile")
    axes[0].set_ylabel("Log return (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].fill_between(dates, main.predicted_stress, color="#008779", alpha=0.7)
    axes[1].set_ylabel("P(high-variance state)\nbefore outcome")
    axes[1].set_ylim(0, 1)
    axes[2].plot(
        dates,
        (main["return"] < main.q05).rolling(252).mean(),
        color="#277da8",
        label="Raw Student-t HMM",
    )
    axes[2].plot(
        dates,
        (main["return"] < main.q05_adaptive).rolling(252).mean(),
        color="#d27d2d",
        label="With past-only calibration",
    )
    axes[2].axhline(0.05, color="#222", ls="--", lw=1)
    axes[2].set_ylabel("Rolling 252-observation\nbreach fraction")
    axes[2].legend(fontsize=8)
    fig.suptitle("EUR/USD · causal regime beliefs and tail-risk calibration", fontsize=16)
    _save(fig, figures / "regime_calibration.png")

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    rows = paired.reset_index(drop=True)
    for i, row in rows.iterrows():
        ax.errorbar(
            row["mean"],
            i,
            xerr=[[row["mean"] - row.lower], [row.upper - row["mean"]]],
            fmt="o",
            capsize=3,
            color="#008779" if row.lower > 0 else "#b95364",
        )
    ax.axvline(0, color="#444", ls="--", lw=1)
    ax.set_yticks(range(len(rows)), rows.asset + " · " + rows.comparison.str.replace("_", " "))
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired log-score difference (nats per observation)")
    ax.set_title("Does extra complexity help? 95% moving-block bootstrap intervals")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, figures / "paired_forecast_scores.png")

    assets = list(summary.asset.unique())
    fig, axes = plt.subplots(
        1, len(assets), figsize=(12, 4.3), squeeze=False, constrained_layout=True
    )
    for ax, asset in zip(axes[0], assets, strict=True):
        s = summary[summary.asset == asset]
        positions = np.arange(len(s))
        ax.bar(positions - 0.18, s.breach_rate, width=0.35, label="Raw", color="#277da8")
        ax.bar(
            positions + 0.18, s.adaptive_breach_rate, width=0.35, label="Adaptive", color="#d27d2d"
        )
        ax.axhline(0.05, color="#222", ls="--", lw=1)
        ax.set_xticks(positions, s.model.str.replace("_", " "), rotation=60, ha="right", fontsize=8)
        ax.set_title(asset)
        ax.set_ylim(0, max(0.075, s.breach_rate.max() * 1.2))
        ax.set_ylabel("5% lower-tail breach fraction")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Calibration is an empirical result, not an automatic HMM property", fontsize=14)
    _save(fig, figures / "coverage_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, model in zip(axes, ["gaussian_iid", "student_hmm", "ewma_gaussian"], strict=True):
        values = predictions[(predictions.asset == "EURUSD") & (predictions.model == model)].pit
        ax.hist(
            values,
            bins=np.linspace(0, 1, 21),
            density=True,
            color=COLORS[model],
            alpha=0.8,
            edgecolor="white",
        )
        ax.axhline(1, color="#222", ls="--", lw=1)
        ax.set_title(model.replace("_", " "))
        ax.set_xlabel("Predictive CDF at the realized return")
        ax.set_ylabel("Density")
    fig.suptitle("EUR/USD probability-integral-transform diagnostic", fontsize=15)
    _save(fig, figures / "pit_calibration.png")


def leakage_demo(out):
    x, states = simulate_regimes(500, 887, "gaussian")
    model = HiddenMarkovModel.from_parameters(
        [0.75, 0.25], [[0.975, 0.025], [0.075, 0.925]], [0.0002, -0.0005], [0.006, 0.02]
    )
    filtered, predicted, _ = model.filter(x)
    smooth = model.smooth(x)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    sl = slice(100, 250)
    ax.fill_between(
        np.arange(500)[sl], states[sl], color="#d4dae1", alpha=0.7, label="True synthetic state"
    )
    ax.plot(
        np.arange(500)[sl],
        predicted[sl, 1],
        color="#008779",
        label="Predicted: before current outcome",
    )
    ax.plot(
        np.arange(500)[sl],
        smooth[sl, 1],
        color="#b95364",
        ls="--",
        label="Smoothed: sees future (offline only)",
    )
    ax.set_xlabel("Observation")
    ax.set_ylabel("High-variance state probability")
    ax.set_title("An attractive historical regime chart can contain future information")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, out / "figures" / "causal_vs_smoothed.png")
    pd.DataFrame(
        {
            "state": states,
            "return": x,
            "predicted": predicted[:, 1],
            "filtered": filtered[:, 1],
            "smoothed": smooth[:, 1],
        }
    ).to_csv(out / "leakage_demo.csv", index=False)


def run_study(
    output="results/full", data_path="data/fx_spot.csv", quick=False, trials=None, overwrite=False
):
    started = time.perf_counter()
    out = Path(output)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError("Output directory is not empty; choose a fresh path or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    plot_style()
    trials = trials or (5 if quick else 100)
    cfg = WalkForwardConfig(
        train_size=500 if quick else 1250,
        refit_every=250 if quick else 252,
        n_init=2 if quick else 4,
    )
    _, synthetic = synthetic_study(
        trials, out, n_train=500 if quick else 1000, n_test=200 if quick else 400
    )
    _, real, fits = real_study(data_path, out, cfg, quick)
    leakage_demo(out)
    manifest = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "seed": 20261117,
        "mode": "quick" if quick else "full",
        "synthetic_trials_per_dgp": trials,
        "synthetic_scenarios": 4,
        "walk_forward": asdict(cfg),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "converged_fits": sum(f["converged"] for f in fits),
        "total_fits": len(fits),
        "selection": "Six initial models plus a labeled follow-up Student-t EWMA; no winner selection.",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = "# Executed research results\n\n"
    report += (
        f"Generated from actual runs: {trials} independent seeds per synthetic scenario; "
        f"{len(fits)} chronological real-data fits. Runtime {manifest['elapsed_seconds']:.1f} seconds.\n\n"
    )
    report += "## Real-data forecasts\n\n" + _table(
        real[
            [
                "asset",
                "model",
                "n",
                "mean_log_score",
                "breach_rate",
                "adaptive_breach_rate",
                "q05_loss",
            ]
        ]
    )
    report += "\n\n## Known-state simulation\n\n" + _table(
        synthetic[
            [
                "dgp",
                "model",
                "trials",
                "mean_log_score",
                "breach_rate",
                "state_accuracy",
                "convergence_rate",
            ]
        ]
    )
    report += (
        "\n\n## Interpretation rules\n\nHigher log score is better; 5% breaches should be near 0.05; "
        "lower quantile loss is better. Confidence intervals compare paired dates using circular "
        "20-observation blocks. They are marginal descriptive intervals, not simultaneous claims "
        "across all tested comparisons. State accuracy is available only in simulations and "
        "states are aligned by emission variance. Calibration feedback uses past errors only. "
        "Neither observed calibration nor an asymptotic p-value establishes future conditional coverage.\n\n"
        "FX observations are Federal Reserve noon buying rates, not executable quotes. H.10 publication "
        "lag and historical revisions are not reconstructed. This is a retrospective forecasting "
        "benchmark, not a real-time trading P&L claim. Missing dates are dropped per asset; a return "
        "across a holiday represents an observation interval rather than exactly 24 hours.\n"
    )
    (out / "REPORT.md").write_text(report)
    manifest["artifact_sha256"] = {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest
