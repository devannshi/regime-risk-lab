"""Verify source/data/artifact identity and forecast cardinality for a completed run."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results/full"
    manifest = json.loads((out / "manifest.json").read_text())
    assert (
        hashlib.sha256((root / "data/fx_spot.csv").read_bytes()).hexdigest()
        == manifest["data_sha256"]
    ), "Frozen data changed"
    for name, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((root / "src/regime_risk" / name).read_bytes()).hexdigest()
        assert actual == expected, f"Source changed: {name}"
    for name, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected, name
    synthetic = pd.read_csv(out / "synthetic_trials.csv")
    assert len(synthetic) == 4 * 4 * manifest["synthetic_trials_per_dgp"]
    forecasts = pd.read_csv(out / "forecasts.csv.gz")
    assert not forecasts.duplicated(["asset", "model", "index"]).any()
    assert forecasts.groupby(["asset", "index"]).model.nunique().eq(7).all()
    audit = json.loads((out / "fit_audit.json").read_text())
    assert all(row["train_end_exclusive"] == row["test_start"] for row in audit)
    print(
        f"Verified {len(synthetic)} synthetic fits, {len(forecasts)} forecasts, source and artifact hashes."
    )


if __name__ == "__main__":
    main()
