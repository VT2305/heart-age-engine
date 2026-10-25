from heart_age_engine.anomaly_detection import run_lof
from heart_age_engine.clustering import fit_clusters
from heart_age_engine.data import clean_data
from heart_age_engine.modeling import add_heart_age_predictions, train_heart_age_model


def test_lof_and_clustering_shapes(synthetic_raw_data):
    df = clean_data(synthetic_raw_data)
    fitted = train_heart_age_model(df)
    scored = add_heart_age_predictions(df, fitted)

    lof = run_lof(scored, contamination=0.05)
    assert set(lof.frame["lof_label"].unique()).issubset({-1, 1})
    assert len(lof.frame) == len(scored)

    clustered, model = fit_clusters(scored, n_clusters=4)
    assert clustered["cluster"].nunique() <= 4
    assert len(model.features) > 0
