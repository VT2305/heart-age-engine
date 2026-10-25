"""Local Outlier Factor and label/model mismatch analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .config import LOF_FEATURES


@dataclass
class LOFResult:
    frame: pd.DataFrame
    scaler: StandardScaler
    model: LocalOutlierFactor
    contamination: float



def run_lof(
    df: pd.DataFrame,
    features: list[str] | None = None,
    *,
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> LOFResult:
    """Run distance-based LOF after standardising the input features."""
    features = features or LOF_FEATURES
    x = df[features].copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    labels = model.fit_predict(x_scaled)

    out = df.copy()
    out["lof_label"] = labels
    out["lof_score"] = model.negative_outlier_factor_
    return LOFResult(out, scaler, model, contamination)


def find_label_model_mismatches(
    df: pd.DataFrame,
    *,
    upper_quantile: float = 0.80,
    lower_quantile: float = 0.20,
    require_lof_anomaly: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find high-gap healthy-label and low-gap disease-label cases.

    These are exploratory disagreements between the dataset label and the
    model-derived age gap. They are not diagnostic categories.
    """
    required = {"age_gap", "cardio"}
    if require_lof_anomaly:
        required.add("lof_label")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for mismatch analysis: {sorted(missing)}")

    upper = df["age_gap"].quantile(upper_quantile)
    lower = df["age_gap"].quantile(lower_quantile)
    lof_mask = df["lof_label"].eq(-1) if require_lof_anomaly else True

    high_gap_healthy = df.loc[
        lof_mask & df["cardio"].eq(0) & df["age_gap"].gt(upper)
    ].copy()
    low_gap_disease = df.loc[
        lof_mask & df["cardio"].eq(1) & df["age_gap"].lt(lower)
    ].copy()
    return high_gap_healthy, low_gap_disease
