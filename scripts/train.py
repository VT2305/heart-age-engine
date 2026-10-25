#!/usr/bin/env python
"""Train and persist the reproducible model bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from heart_age_engine.data import load_and_prepare_data
from heart_age_engine.pipeline import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to cardio_train.csv or .xlsx")
    parser.add_argument("--output", default="artifacts/model_bundle.joblib")
    args = parser.parse_args()

    df = load_and_prepare_data(args.data)
    _, bundle = train_pipeline(df)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)

    metrics = {
        "rows_after_cleaning": int(len(df)),
        "heart_age_mae_years": bundle.heart_age_model.mae,
        "heart_age_r2": bundle.heart_age_model.r2,
        "model_features": bundle.heart_age_model.features,
    }
    results_path = Path("results/training_metrics.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model bundle to {output}")


if __name__ == "__main__":
    main()
