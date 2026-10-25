"""Data loading, validation, cleaning, and feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_REQUIRED_COLUMNS


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".csv":
        frame = pd.read_csv(path, sep=";")
        if frame.shape[1] == 1:
            frame = pd.read_csv(path)
        return frame
    raise ValueError(f"Unsupported file type: {suffix}. Use .csv or .xlsx")


def validate_raw_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create age in years, BMI, and pulse pressure."""
    out = df.copy()
    out["age_years"] = out["age"] / 365.25
    out["bmi"] = out["weight"] / (out["height"] / 100.0) ** 2
    out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the physiological plausibility filters used in the coursework."""
    validate_raw_columns(df)
    out = engineer_features(df)

    mask = (
        out["ap_hi"].between(80, 220)
        & out["ap_lo"].between(40, 150)
        & (out["ap_hi"] > out["ap_lo"])
        & out["bmi"].between(14, 60)
        & out["height"].between(130, 210)
        & out["weight"].between(30, 180)
    )

    out = out.loc[mask].copy()
    out = out.drop_duplicates(subset="id", keep="first")
    out = out.dropna(subset=RAW_REQUIRED_COLUMNS)
    return out.reset_index(drop=True)


def load_and_prepare_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV/XLSX source and return the cleaned modelling table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return clean_data(_read_table(path))
