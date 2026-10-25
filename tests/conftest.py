import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_raw_data():
    rng = np.random.default_rng(42)
    n = 240
    age_years = rng.uniform(35, 65, n)
    height = rng.normal(168, 8, n).clip(145, 195)
    bmi = rng.normal(26, 4, n).clip(18, 40)
    weight = bmi * (height / 100) ** 2
    ap_hi = (100 + 0.55 * age_years + rng.normal(0, 8, n)).clip(90, 190)
    ap_lo = (55 + 0.35 * age_years + rng.normal(0, 6, n)).clip(50, 120)
    ap_hi = np.maximum(ap_hi, ap_lo + 10)

    risk = (ap_hi > 140).astype(int) + (bmi > 30).astype(int)
    cardio = ((risk + rng.integers(0, 2, n)) >= 2).astype(int)
    cardio[:80] = 0

    return pd.DataFrame(
        {
            "id": np.arange(n),
            "age": age_years * 365.25,
            "gender": rng.integers(1, 3, n),
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": rng.integers(1, 4, n),
            "gluc": rng.integers(1, 4, n),
            "smoke": rng.integers(0, 2, n),
            "alco": rng.integers(0, 2, n),
            "active": rng.integers(0, 2, n),
            "cardio": cardio,
        }
    )
