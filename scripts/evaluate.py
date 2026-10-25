#!/usr/bin/env python
"""Reproduce aggregate model, anomaly, and clustering results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heart_age_engine.anomaly_detection import find_label_model_mismatches, run_lof
from heart_age_engine.clustering import cluster_summary, fit_clusters
from heart_age_engine.data import load_and_prepare_data
from heart_age_engine.explainability import global_importance
from heart_age_engine.modeling import (
    add_heart_age_predictions,
    compare_feature_sets,
    train_heart_age_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to cardio_train.csv or .xlsx")
    args = parser.parse_args()

    results = Path("results")
    results.mkdir(exist_ok=True)

    df = load_and_prepare_data(args.data)
    fitted = train_heart_age_model(df)
    scored = add_heart_age_predictions(df, fitted)

    feature_comparison = compare_feature_sets(df)
    feature_comparison.to_csv(results / "feature_set_comparison.csv", index=False)

    lof = run_lof(scored)
    high_gap_healthy, low_gap_disease = find_label_model_mismatches(lof.frame)

    clustered, _ = fit_clusters(scored)
    summary = cluster_summary(clustered)
    summary.to_csv(results / "cluster_summary.csv", index=False)

    shap_table = global_importance(fitted, scored)
    shap_table.to_csv(results / "global_shap_importance.csv", index=False)

    aggregate = {
        "rows_after_cleaning": int(len(df)),
        "heart_age_mae_years": fitted.mae,
        "heart_age_r2": fitted.r2,
        "lof_contamination": lof.contamination,
        "lof_flagged": int((lof.frame["lof_label"] == -1).sum()),
        "high_gap_healthy_label_mismatches": int(len(high_gap_healthy)),
        "low_gap_disease_label_mismatches": int(len(low_gap_disease)),
    }
    (results / "evaluation_summary.json").write_text(json.dumps(aggregate, indent=2))

    print(json.dumps(aggregate, indent=2))
    print("\nFeature-set comparison:")
    print(feature_comparison.to_string(index=False))
    print("\nCluster summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
