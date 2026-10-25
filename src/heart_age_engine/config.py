"""Central configuration for features and reproducibility."""

RANDOM_STATE = 42

RAW_REQUIRED_COLUMNS = [
    "id",
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
]

# IMPORTANT: age_years is the regression target and must never appear here.
MODEL_FEATURES = [
    "gender",
    "bmi",
    "ap_hi",
    "ap_lo",
    "pulse_pressure",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

CLINICAL_FEATURES = [
    "gender",
    "bmi",
    "ap_hi",
    "ap_lo",
    "pulse_pressure",
    "cholesterol",
    "gluc",
]

LIFESTYLE_FEATURES = [
    "gender",
    "bmi",
    "smoke",
    "alco",
    "active",
]

LOF_FEATURES = [
    "age_years",
    "gender",
    "bmi",
    "ap_hi",
    "ap_lo",
    "pulse_pressure",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

CLUSTER_FEATURES = [
    "age_gap",
    "age_years",
    "bmi",
    "ap_hi",
    "pulse_pressure",
    "cholesterol",
    "gluc",
    "active",
    "smoke",
]

PRETTY_NAMES = {
    "gender": "Gender",
    "bmi": "BMI",
    "ap_hi": "Systolic BP",
    "ap_lo": "Diastolic BP",
    "pulse_pressure": "Pulse pressure",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoking",
    "alco": "Alcohol",
    "active": "Physical activity",
}
