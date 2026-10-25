from heart_age_engine.config import MODEL_FEATURES
from heart_age_engine.data import clean_data
from heart_age_engine.modeling import add_heart_age_predictions, train_heart_age_model


def test_age_target_is_not_a_predictor():
    assert "age_years" not in MODEL_FEATURES


def test_model_trains_and_scores(synthetic_raw_data):
    df = clean_data(synthetic_raw_data)
    fitted = train_heart_age_model(df)
    scored = add_heart_age_predictions(df, fitted)
    assert fitted.mae >= 0
    assert "heart_age" in scored.columns
    assert "age_gap" in scored.columns
    assert len(scored) == len(df)
