"""End-to-end training and inference bundle."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .clustering import ClusterModel, fit_clusters
from .modeling import HeartAgeModel, add_heart_age_predictions, train_heart_age_model


@dataclass
class ModelBundle:
    heart_age_model: HeartAgeModel
    cluster_model: ClusterModel


def train_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, ModelBundle]:
    heart_age_model = train_heart_age_model(df)
    scored = add_heart_age_predictions(df, heart_age_model)
    clustered, cluster_model = fit_clusters(scored)
    return clustered, ModelBundle(heart_age_model, cluster_model)
