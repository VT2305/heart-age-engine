"""Leakage-safe heart-age proxy modelling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .config import (
    CLINICAL_FEATURES,
    LIFESTYLE_FEATURES,
    MODEL_FEATURES,
    RANDOM_STATE,
)


@dataclass
class HeartAgeModel:
    model: GradientBoostingRegressor
    features: list[str]
    mae: float
    r2: float

    def predict(self, frame: pd.DataFrame):
        return self.model.predict(frame[self.features])


def _fit_regressor(df: pd.DataFrame, features: list[str]) -> HeartAgeModel:
    if "age_years" in features:
        raise ValueError("Target leakage: age_years cannot be used as a predictor.")

    healthy = df.loc[df["cardio"] == 0].copy()
    if len(healthy) < 50:
        raise ValueError("Not enough healthy-reference rows to train the model.")

    x_train, x_test, y_train, y_test = train_test_split(
        healthy[features],
        healthy["age_years"],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    return HeartAgeModel(
        model=model,
        features=list(features),
        mae=float(mean_absolute_error(y_test, pred)),
        r2=float(r2_score(y_test, pred)),
    )


def train_heart_age_model(df: pd.DataFrame) -> HeartAgeModel:
    return _fit_regressor(df, MODEL_FEATURES)


def add_heart_age_predictions(df: pd.DataFrame, fitted: HeartAgeModel) -> pd.DataFrame:
    out = df.copy()
    out["heart_age"] = fitted.predict(out)
    out["age_gap"] = out["heart_age"] - out["age_years"]
    return out


def compare_feature_sets(df: pd.DataFrame) -> pd.DataFrame:
    """Compare full, clinical-only, and lifestyle-only leakage-safe models."""
    rows = []
    for name, features in {
        "full": MODEL_FEATURES,
        "clinical_only": CLINICAL_FEATURES,
        "lifestyle_only": LIFESTYLE_FEATURES,
    }.items():
        fitted = _fit_regressor(df, features)
        rows.append({"model": name, "mae_years": fitted.mae, "r2": fitted.r2})
    return pd.DataFrame(rows).sort_values("mae_years").reset_index(drop=True)
