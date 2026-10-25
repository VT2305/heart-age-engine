from heart_age_engine.data import clean_data


def test_clean_data_engineers_features(synthetic_raw_data):
    df = clean_data(synthetic_raw_data)
    assert {"age_years", "bmi", "pulse_pressure"}.issubset(df.columns)
    assert (df["ap_hi"] > df["ap_lo"]).all()
    assert df["id"].is_unique
