"""Build and execute the self-contained research walkthrough in the active Python environment."""

import sys
from pathlib import Path

import nbformat as nbf
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(
            "# Do Hidden Market Regimes Improve Tail-Risk Forecasts?\n\n"
            "This notebook is an executable tour. It distinguishes predicted, filtered and "
            "smoothed states, fits only on a training segment, and reads the complete research audit. "
            "The package implements inference itself; `hmmlearn` is only an independent test oracle."
        ),
        code(
            "from pathlib import Path\nimport sys\nimport numpy as np\nimport pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "root = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n"
            "sys.path.insert(0, str(root / 'src'))\n"
            "from regime_risk.hmm import HiddenMarkovModel\n"
            "from regime_risk.data import simulate_regimes\n"
            "from regime_risk.evaluation import coverage_tests, quantile_loss\n"
            "print('Imports ready; all examples use explicit random seeds.')"
        ),
        md(
            "## 1. Fit a heavy-tailed regime model\n\nThe synthetic path has known states. "
            "Only the first 1,000 observations estimate parameters. State 1 means higher "
            "emission variance, not a factual economic label."
        ),
        code(
            "returns, states = simulate_regimes(1400, seed=1729, kind='student')\n"
            "train, test = returns[:1000], returns[1000:]\n"
            "scale = train.std()\n"
            "model = HiddenMarkovModel(emission='student', n_init=4, random_state=1729).fit(train / scale)\n"
            "print('Converged:', model.converged_, '| selected iterations:', model.n_iter_)\n"
            "print('Transition matrix:')\nprint(np.round(model.params.transition, 3))\n"
            "print('Emission standard deviations:', np.round(np.sqrt(model.variances) * scale, 5))"
        ),
        md(
            "## 2. Issue forecasts before the outcome\n\nThe last training posterior is advanced "
            "through the transition matrix before the first test observation. Quantiles come from "
            "the predictive mixture; a weighted mean of component quantiles would be incorrect."
        ),
        code(
            "last_train = model.filter(train / scale)[0][-1]\n"
            "filtered, predicted, log_scores = model.filter(test / scale, last_train @ model.params.transition)\n"
            "q05 = scale * model.mixture_quantile(.05, predicted)\n"
            "diagnostics = coverage_tests(test, q05, .05)\n"
            "print('Mean held-out log density:', round(float(np.mean(log_scores - np.log(scale))), 4))\n"
            "print('5% breach rate:', round(diagnostics['breach_rate'], 4))\n"
            "print('Pinball loss:', round(float(quantile_loss(test, q05, .05).mean()), 6))\n"
            "assert np.allclose(predicted.sum(axis=1), 1)"
        ),
        code(
            "fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)\n"
            "axes[0].plot(test * 100, color='#71869a', lw=.8, label='Observed return')\n"
            "axes[0].plot(q05 * 100, color='#008779', label='Previously issued 5% quantile')\n"
            "axes[0].set_ylabel('Log return (%)'); axes[0].legend(fontsize=8)\n"
            "axes[1].fill_between(np.arange(len(test)), states[1000:], alpha=.2, color='gray', label='True simulated state')\n"
            "axes[1].plot(predicted[:, 1], color='#008779', label='Predicted high-variance probability')\n"
            "axes[1].set_ylabel('Probability'); axes[1].set_xlabel('Test observation'); axes[1].legend(fontsize=8)\nplt.show()"
        ),
        md(
            "## 3. A direct test for future leakage\n\nChange later observations dramatically. "
            "Earlier causal forecasts must remain unchanged; offline smoothed probabilities are "
            "allowed to change because they deliberately use future information."
        ),
        code(
            "changed = test.copy(); changed[200:] += .2\n"
            "changed_post, changed_pred, _ = model.filter(changed / scale, last_train @ model.params.transition)\n"
            "assert np.allclose(predicted[:201], changed_pred[:201])\n"
            "assert np.allclose(filtered[:200], changed_post[:200])\n"
            "print('Passed: future outcomes cannot alter already-issued predictions.')"
        ),
        md(
            "## 4. Inspect every model, including stronger baselines\n\nThe committed real-data "
            "experiment evaluates three currency series. A follow-up Student-t EWMA control is "
            "explicitly labeled in the research log. These noon rates do not support an execution "
            "or live-trading claim."
        ),
        code(
            "summary = pd.read_csv(root / 'results/full/real_summary.csv')\n"
            "display(summary[['asset', 'model', 'mean_log_score', 'breach_rate', 'adaptive_breach_rate']].round(4))"
        ),
        code(
            "comparisons = pd.read_csv(root / 'results/full/paired_logscore_intervals.csv')\n"
            "display(comparisons[['asset', 'comparison', 'mean', 'lower', 'upper']].round(4))\n"
            "import json\n"
            "audit = json.loads((root / 'results/full/fit_audit.json').read_text())\n"
            "assert all(fit['train_end_exclusive'] == fit['test_start'] for fit in audit)\n"
            "print('Audited chronological fit boundaries:', len(audit))"
        ),
        md(
            "## 5. Interpret a failure\n\nUnder a persistent scale break, all fitted model "
            "families can under-cover the lower tail. Better average log score does not promise "
            "a correct 5% risk threshold. The online controller is a practical adjustment, not "
            "a new finite-sample coverage theorem."
        ),
        code(
            "synthetic = pd.read_csv(root / 'results/full/synthetic_summary.csv')\n"
            "display(synthetic[synthetic.dgp == 'break'][['model', 'mean_log_score', 'breach_rate', 'adaptive_breach_rate']].round(4))"
        ),
        md(
            "For derivations, assumptions, data provenance, and reproduction commands, see "
            "the repository README and `docs/`. A quantitative explanation should include the "
            "cases where a simple benchmark wins, not just the best-looking model output."
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=180, resources={"metadata": {"path": str(root)}}
    )
    client.execute()
    nbf.validate(notebook)
    destination = root / "notebooks/research_walkthrough.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbf.write(notebook, destination)
    print(
        f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} notebook code cells."
    )


if __name__ == "__main__":
    main()
