"""SHAP utilities for the Gradient Boosting heart-age proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .config import PRETTY_NAMES
from .modeling import HeartAgeModel


def _tree_explainer(fitted: HeartAgeModel):
    return shap.TreeExplainer(fitted.model)


def local_contributions(fitted: HeartAgeModel, row: pd.DataFrame) -> pd.DataFrame:
    row = row[fitted.features]
    values = _tree_explainer(fitted).shap_values(row)
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[0]

    return (
        pd.DataFrame(
            {
                "feature": [PRETTY_NAMES.get(f, f) for f in fitted.features],
                "contribution_years": values,
            }
        )
        .assign(abs_contribution=lambda x: x["contribution_years"].abs())
        .sort_values("abs_contribution", ascending=False)
        .drop(columns="abs_contribution")
        .reset_index(drop=True)
    )


def global_importance(
    fitted: HeartAgeModel,
    frame: pd.DataFrame,
    sample_size: int = 1000,
) -> pd.DataFrame:
    sample = frame.sample(min(sample_size, len(frame)), random_state=42)
    values = np.asarray(_tree_explainer(fitted).shap_values(sample[fitted.features]))
    mean_abs = np.abs(values).mean(axis=0)

    return (
        pd.DataFrame(
            {
                "feature": [PRETTY_NAMES.get(f, f) for f in fitted.features],
                "mean_abs_shap": mean_abs,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
