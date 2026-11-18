"""Public-domain Federal Reserve FX data and independent synthetic regimes."""

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SERIES = {"EURUSD": "DEXUSEU", "GBPUSD": "DEXUSUK", "JPYUSD": "DEXJPUS"}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU,DEXUSUK,DEXJPUS"


def fetch_fx(output="data", start="2000-01-01", end="2025-12-31"):
    """Freeze a public FX snapshot; invert JPY so every column is USD per FX unit."""
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    response = requests.get(FRED_URL, timeout=60)
    response.raise_for_status()
    raw = pd.read_csv(io.BytesIO(response.content), na_values=["."])
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw.date)
    raw = raw.loc[raw.date.between(start, end)].copy()
    raw = raw.rename(columns={v: k for k, v in SERIES.items()})
    raw["JPYUSD"] = 1 / raw.JPYUSD
    raw = raw[["date", *SERIES]]
    if len(raw) < 1000 or (raw[list(SERIES)] <= 0).any().any():
        raise ValueError("Unexpected provider payload or invalid exchange rates")
    path = target / "fx_spot.csv"
    raw.to_csv(path, index=False, float_format="%.10g")
    manifest = {
        "source_url": FRED_URL,
        "series": SERIES,
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "start": start,
        "end": end,
        "rows": len(raw),
        "provider_payload_sha256": hashlib.sha256(response.content).hexdigest(),
        "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "license": "Public domain: citation requested (Federal Reserve H.10; see FRED series pages)",
        "transform": "JPY series reciprocated; missing releases preserved; no interpolation",
        "vintage_note": "Latest historical snapshot, not real-time publication vintages",
    }
    (target / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def load_returns(path):
    """Each asset uses its own observed quote calendar; gaps are not forward-filled."""
    df = pd.read_csv(path, parse_dates=["date"])
    if df.date.duplicated().any() or not df.date.is_monotonic_increasing:
        raise ValueError("Data dates must be unique and increasing")
    result = {}
    for asset in SERIES:
        clean = df[["date", asset]].dropna()
        if (clean[asset] <= 0).any():
            raise ValueError("Prices must be positive")
        result[asset] = pd.DataFrame(
            {
                "date": clean.date.iloc[1:].to_numpy(),
                "return": np.diff(np.log(clean[asset].to_numpy())),
            }
        )
    return result


def simulate_regimes(n=1500, seed=0, kind="gaussian"):
    """Two-state process; stress scales exceed calm scales by construction."""
    if n < 2 or kind not in {"gaussian", "student", "contaminated", "break"}:
        raise ValueError("Invalid simulation configuration")
    rng = np.random.default_rng(seed)
    a = np.array([[0.975, 0.025], [0.075, 0.925]])
    sd = np.array([0.006, 0.02])
    means = np.array([0.0002, -0.0005])
    states = np.zeros(n, dtype=int)
    x = np.empty(n)
    for t in range(n):
        if t:
            states[t] = rng.choice(2, p=a[states[t - 1]])
        noise = rng.standard_t(5) * np.sqrt(3 / 5) if kind == "student" else rng.normal()
        multiplier = 1.8 if kind == "break" and t >= int(0.75 * n) else 1.0
        x[t] = means[states[t]] + multiplier * sd[states[t]] * noise
        if kind == "contaminated" and rng.random() < 0.015:
            x[t] += rng.normal(0, 0.07)
    return x, states
