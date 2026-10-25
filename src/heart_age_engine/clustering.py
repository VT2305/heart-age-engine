"""Exploratory K-Means patient profiling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import CLUSTER_FEATURES, RANDOM_STATE


@dataclass
class ClusterModel:
    scaler: StandardScaler
    model: KMeans
    features: list[str]

    def predict(self, frame: pd.DataFrame):
        return self.model.predict(self.scaler.transform(frame[self.features]))


def fit_clusters(df: pd.DataFrame, n_clusters: int = 8) -> tuple[pd.DataFrame, ClusterModel]:
    x = df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    labels = model.fit_predict(x_scaled)

    out = df.copy()
    out["cluster"] = labels
    return out, ClusterModel(scaler, model, list(CLUSTER_FEATURES))


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {"cluster", "cardio", "age_gap", "bmi", "ap_hi", "cholesterol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing cluster-summary columns: {sorted(missing)}")

    return (
        df.groupby("cluster")
        .agg(
            count=("cardio", "size"),
            mean_age_gap=("age_gap", "mean"),
            mean_bmi=("bmi", "mean"),
            mean_systolic_bp=("ap_hi", "mean"),
            mean_cholesterol=("cholesterol", "mean"),
            disease_label_rate=("cardio", "mean"),
        )
        .round(3)
        .reset_index()
    )
