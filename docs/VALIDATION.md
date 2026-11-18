# Executed validation record

The published full experiment completed on 2026-11-18 UTC in Python 3.12.6.

- **50 tests pass**, including exhaustive path enumeration, an independent `hmmlearn` likelihood/smoothing oracle, direct causal perturbations, mixture quantile checks, and impossible-state numerical stability.
- Ruff checks pass on source, tests and scripts.
- The full study executes 1,600 synthetic fits and 252 chronological real-data fits, preserving all four nonconverged fits. The ledger contains 110,607 forecasts across seven models.
- A separate numerical review compared the Student-t M-step with direct SciPy optimization; the objective difference was below 1e-8 in the audited example. This checks a numerical component, not a universal guarantee of global EM optimality.
- All eight tutorial code cells were executed in a fresh Jupyter kernel, with their outputs retained. Notebook schema validation passes.
- `scripts/verify_results.py` checks the frozen data hash, every model source hash, every listed output hash, forecast uniqueness, model count and train/test boundaries.

Reproduce the complete suite:

```bash
pip install -e ".[dev]"
ruff check src tests scripts
pytest -q
regime-risk run --output results/reproduced
python scripts/verify_results.py results/reproduced
python scripts/build_notebook.py
```

The tested direct-dependency versions are recorded in `requirements-reproduction.txt`. Floating-point paths can differ across libraries and platforms, so the source/data identity checks are exact while independently rerun numerical outputs should be assessed with suitable tolerances. The runtime manifest records the measured run, including its platform-dependent elapsed time.

The Student-t EWMA control was added after initial results and is labeled as exploratory in the research log. No live data or financial transactions were used. FX observations are retrospective H.10 records; this is not an audit of real-time publication availability or executable trading returns.
