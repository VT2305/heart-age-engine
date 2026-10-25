# Model Card — Heart Age Engine

## Summary

Heart Age Engine is an **educational machine-learning project** built on a public cardiovascular dataset. It produces a model-derived age estimate from health/lifestyle variables, calculates an age-gap proxy, explains predictions using SHAP, screens unusual records using Local Outlier Factor, and groups records using K-Means.

It is **not a medical device**, is not clinically validated, and must not be used for diagnosis, treatment, or individual health decisions.

## Data

Source: Svetlana Ulianova, *Cardiovascular Disease Dataset*, Kaggle.  
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

The original coursework used roughly 70,000 raw records and applied physiological plausibility filtering and deduplication before modelling.

## Heart-age proxy

### Target

`age_years`

### Predictors

- gender
- BMI
- systolic blood pressure
- diastolic blood pressure
- pulse pressure
- cholesterol category
- glucose category
- smoking indicator
- alcohol-use indicator
- physical-activity indicator

### Leakage guardrail

Chronological age is the target and is deliberately excluded from the predictor set. `tests/test_modeling.py` enforces this constraint.

### Reference population

The regressor is fitted only to records where `cardio == 0`, following the coursework idea of learning an age relationship from the healthy-labelled subset.

### Algorithm

`GradientBoostingRegressor`

Tree-based gradient boosting was retained from the original project because it can model nonlinear relationships and interactions without imposing a linear functional form.

## Explainability

SHAP TreeExplainer is used for both global and local model explanations. SHAP values describe the behaviour of the fitted model; they do not establish causal effects.

## Anomaly detection

Local Outlier Factor is applied to 11 health/lifestyle variables after standardisation.

Default settings:

- `n_neighbors=20`
- `contamination=0.05`

The contamination setting is an analyst-chosen threshold. A flagged point is statistically unusual under the fitted neighbourhood model, not necessarily clinically abnormal.

## Clustering

K-Means is used as an exploratory profiling tool after scaling. The default public implementation keeps `k=8` to reproduce the original project configuration. Cluster IDs are arbitrary and should not be interpreted as medical phenotypes.

## Intended use

Appropriate uses:

- demonstrating an end-to-end applied ML workflow;
- studying model explainability and anomaly detection;
- experimenting with preprocessing and feature subsets;
- discussing ML evaluation and limitations in an interview or classroom setting.

Inappropriate uses:

- clinical diagnosis;
- treatment recommendations;
- estimating an individual's true biological heart age;
- medical triage or screening;
- inferring causal effects from SHAP or anomaly scores.

## Known limitations

1. Public observational data can contain measurement and entry errors.
2. The dataset is not a representative clinical cohort for every population.
3. `cardio` is treated as a dataset label, not a verified ground-truth diagnosis.
4. The healthy-labelled subset can still contain undetected risk or disease.
5. The heart-age output is a regression proxy and has no clinical calibration.
6. K-Means results depend on feature selection, scaling, and chosen `k`.
7. LOF results depend on neighbourhood size and contamination threshold.
8. SHAP explains the fitted model, not biological causality.

## Reproducibility

Run:

```bash
python scripts/train.py --data data/cardio_train.xlsx
python scripts/evaluate.py --data data/cardio_train.xlsx
pytest
```

Metrics are generated locally under `results/` rather than hard-coded into the repository.
