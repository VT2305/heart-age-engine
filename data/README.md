# Data setup

The raw dataset is intentionally not committed to this repository.

Download the **Cardiovascular Disease Dataset** from:

https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Place one of the following files here:

```text
data/cardio_train.csv
```

or:

```text
data/cardio_train.xlsx
```

The loader supports the original semicolon-delimited CSV as well as an XLSX copy.

Expected raw columns:

```text
id, age, gender, height, weight, ap_hi, ap_lo,
cholesterol, gluc, smoke, alco, active, cardio
```

Do not commit patient-level generated outputs or modified copies of the raw dataset unless you have confirmed the dataset licence permits redistribution.
