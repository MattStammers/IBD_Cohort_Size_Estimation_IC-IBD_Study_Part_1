# IBD Cohort Size Estimation

### Subtitle: IBD_NLP_Cohort_Identification_Models_IC-IBD_Part_1

## By Dr Matt Stammers

### Originally first uploaded: 07/01/2025. Readme updated: 29/05/2026.

## Overview

This repository contains the analytical code used to estimate inflammatory bowel disease (IBD) cohort size from linked patient-level data sources. The main workflow combines structured features derived from routine data with a regularized and calibrated logistic regression model, then applies additional adjustment using clinic letter signals to produce a final cohort size estimate.

The repository is intended to support methodological transparency for academic reporting. It documents the implemented analytical steps, the order in which they are run, and the intermediate artifacts that are created during model development, evaluation, and final estimation.

## Associated Publications

1. Stammers M, Sartain S, Cummings JF, Kipps C, Nouraei R, Gwiggner M, Metcalf C, Batchelor J. Identification of cohorts with inflammatory bowel disease amidst fragmented clinical databases via machine learning. Digestive Diseases and Sciences. 2025;70(10):3309-22. [Digestive Diseases and Sciences](https://link.springer.com/article/10.1007/s10620-025-09323-1)
2. Stammers M, Ramgopal B, Owusu Nimako A, Vyas A, Nouraei R, Metcalf C, Batchelor J, Shepherd J, Gwiggner M. A foundation systematic review of natural language processing applied to gastroenterology and hepatology. BMC Gastroenterology. 2025;25(1):58. [BMC Gastroenterology](https://bmcgastroenterol.biomedcentral.com/articles/10.1186/s12876-025-03608-5)

## Repository Contents

```text
.
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── final_model/
│   └── bias_metrics.json
├── jaccard_index_size_estimator/
│   └── jaccard_index_cohort_size_estimator.py
├── log_reg_size_estimator/
│   ├── 1_split_data.py
│   ├── 2_preprocess.py
│   ├── 3_feature_selection.py
│   ├── 4_train_model.py
│   ├── 5_evaluate_model.py
│   ├── 6_predict_total_ibd.py
│   ├── 7_bias_analysis.py
│   ├── 8_final_estimate.py
│   └── run_pipeline.py
└── sample_size_estimation/
    ├── original_incorrect_pate_and_riley_sample_size_estimator.py
    └── riley_shrinkage_based_sample_size_estimator.py
```

## Main Components

### Logistic Regression Size Estimator

The primary analytical pipeline is in `log_reg_size_estimator/`. It performs:

1. train/validation splitting;
2. preprocessing of numeric and categorical features;
3. L1-regularized feature selection using `statsmodels`;
4. elastic net logistic regression training with probability calibration;
5. validation-set performance evaluation and threshold selection;
6. whole-cohort prediction;
7. subgroup bias analysis; and
8. final cohort-size estimation incorporating clinic letter information.

### Jaccard Index Estimator

`jaccard_index_size_estimator/` contains an alternative estimator based on Jaccard index logic.

### Sample Size Estimation Utilities

`sample_size_estimation/` contains supporting scripts related to sample size estimation methodology.

## Reproducibility

### Environment

The repository is configured for Python `>=3.10,<3.11`.

Dependencies are pinned in both `pyproject.toml` and `requirements.txt`. Either of the following approaches can be used:

```bash
pip install -r requirements.txt
```

or

```bash
poetry install
```

### Expected Data Layout

This repository does not include patient-level input data. To run the full pipeline, a local `data/` directory must be created alongside the repository with the following structure:

```text
data/
├── final_dataframe.csv
├── merged_dataframe.csv
├── processed/
│   └── clinic_letters.csv
└── log_reg/
```

The scripts write their intermediate and final outputs into `data/log_reg/`. The directory may be created in advance or created by the user before execution.

### Required Input Files

#### 1. `data/final_dataframe.csv`

Used by:

- `1_split_data.py`
- `6_predict_total_ibd.py`
- `7_bias_analysis.py`

Minimum required columns for the training and prediction path:

- `study_id`: unique patient or study identifier.
- `IBD`: binary reference label used where available for supervised training and evaluation. Rows with missing `IBD` are excluded from training and validation steps.

Expected demographic columns excluded from the main predictive feature set in `1_split_data.py`:

- `age_at_referral`
- `sex`
- `ethnicity`
- `imd_decile`

Expected demographic columns used specifically by `7_bias_analysis.py`:

- `age_at_referral_gradings_deduplicated`
- `sex_gradings_deduplicated`
- `ethnicity_gradings_deduplicated`
- `imd_decile_gradings_deduplicated`

All remaining columns are treated as candidate predictor variables unless explicitly excluded by script logic.

#### 2. `data/merged_dataframe.csv`

Used by:

- `2_preprocess.py`

Minimum required columns:

- `study_id`
- `IBD`

Expected demographic columns excluded from model features during preprocessing:

- `age_at_referral`
- `sex`
- `ethnicity`
- `imd_decile`

All remaining columns are interpreted as raw predictor variables and are divided into numeric versus categorical features using pandas data types.

#### 3. `data/processed/clinic_letters.csv`

Used by:

- `8_final_estimate.py`

Required columns:

- `study_id`
- `IBD_Suggestive`: binary indicator identifying patients flagged as suggestive of IBD from clinic letter review.

### Pipeline Order

The full logistic regression workflow is orchestrated by `log_reg_size_estimator/run_pipeline.py`, which executes the following scripts in order:

1. `1_split_data.py`
2. `2_preprocess.py`
3. `3_feature_selection.py`
4. `4_train_model.py`
5. `5_evaluate_model.py`
6. `6_predict_total_ibd.py`
7. `7_bias_analysis.py`
8. `8_final_estimate.py`

This order matters because each stage depends on artifacts written by earlier stages.

### Step-by-Step Workflow

#### 1. Data Splitting

`1_split_data.py` reads `data/final_dataframe.csv`, drops rows with missing `IBD`, excludes `study_id`, `IBD`, and the four demographic columns from the candidate feature matrix, and produces a stratified 70/30 train/validation split.

Outputs written to `data/log_reg/`:

- `X_train.csv`
- `X_val.csv`
- `y_train.csv`
- `y_val.csv`

#### 2. Preprocessing

`2_preprocess.py` reads `data/merged_dataframe.csv` and the previously saved split files. Numeric variables are imputed with `0` and standardized. Categorical variables are imputed with the literal value `missing` and one-hot encoded with unseen categories ignored at transform time.

Outputs written to `data/log_reg/`:

- `preprocessed_train.csv`
- `preprocessed_val.csv`
- `preprocessor.pkl`

#### 3. Feature Selection

`3_feature_selection.py` fits an L1-regularized logistic regression model using `statsmodels.Logit.fit_regularized`. Demographic features matching the following patterns are removed before selection:

- `^age_at_referral_gradings_deduplicated`
- `^sex_gradings_deduplicated`
- `^ethnicity_gradings_deduplicated`
- `^imd_decile_gradings_deduplicated`

Features with absolute coefficient magnitude greater than `1e-4` are retained.

Outputs written to `data/log_reg/`:

- `selected_features.txt`
- `selected_train.csv`
- `selected_val.csv`
- `selected_indices.json`

#### 4. Model Training

`4_train_model.py` trains a pipeline consisting of:

- `StandardScaler`
- `LogisticRegression` with elastic net penalty, `solver='saga'`, `l1_ratio=0.5`, and `max_iter=10000`
- `CalibratedClassifierCV` with sigmoid calibration and 5-fold internal calibration

Hyperparameter selection is performed over `C` values on a log scale using repeated nested cross-validation. The final model is refit on all selected training data using the most frequently chosen `C` value across outer folds.

Output written to `data/log_reg/`:

- `final_model.pkl`

#### 5. Model Evaluation

`5_evaluate_model.py` applies the trained model to the validation set and computes:

- AUROC with bootstrap confidence intervals
- Brier score with bootstrap confidence intervals
- threshold-specific sensitivity, specificity, precision, and accuracy
- an optimal classification threshold using Youden's $J$
- calibration and ROC plots

Outputs written to `data/log_reg/`:

- `evaluation_metrics.json`
- `optimal_threshold.txt`
- `calibration_plot.png`
- `calibration_loess.png`
- `calibration_logistic.png`
- `roc_curve.png`

#### 6. Whole-Cohort Prediction

`6_predict_total_ibd.py` applies the saved preprocessor and final model to the full cohort in `data/final_dataframe.csv`. Missing expected raw columns are created with a default value of `0`, and columns not expected by the preprocessor are dropped before transformation.

Outputs written to `data/log_reg/`:

- `ibd_predictions.csv`
- `total_ibd_estimate.txt`
- `threshold_analysis.json`
- `threshold_analysis.csv`

`ibd_predictions.csv` contains:

- `study_id`
- `IBD_Predicted_Probability`
- `IBD_Predicted`

#### 7. Bias Analysis

`7_bias_analysis.py` evaluates subgroup performance using `fairlearn.metrics.MetricFrame`. The script derives grouped analyses for:

- age band
- IMD band
- sex
- ethnicity

Outputs written to `data/log_reg/`:

- `bias_metrics.json`
- `auc_age_10_100.png`
- `sensitivity_age_10_100.png`
- `auc_imd_bin.png`
- `sensitivity_imd_bin.png`
- `auc_sex.png`
- `auc_ethnicity.png`

#### 8. Final Cohort Size Estimate

`8_final_estimate.py` combines model predictions with clinic letter evidence. In the current implementation:

- patients predicted positive by the model are counted from `ibd_predictions.csv` and adjusted by a precision factor of `0.84`;
- patients flagged as `IBD_Suggestive == 1` in `clinic_letters.csv` but not already predicted positive are identified; and
- that incremental group is adjusted by a precision factor of `0.79` before being added to the model-based estimate.

Outputs written to `data/log_reg/`:

- `final_ibd_size_estimate.txt`
- `model_coefficients.csv`

`model_coefficients.csv` contains averaged coefficients and odds ratios extracted from the fitted calibrated logistic regression model.

### Running the Pipeline

From the repository root:

```bash
python log_reg_size_estimator/run_pipeline.py
```

To run only the final estimation step after upstream artifacts already exist:

```bash
python log_reg_size_estimator/8_final_estimate.py
```

## Important Notes

### Data Availability

The code is public, but the source data are not distributed in this repository. Reproduction therefore requires the user to construct local input files that conform to the column contracts described above.

### Current Column-Naming Assumptions

The scripts currently expect both unsuffixed demographic column names and `_gradings_deduplicated` demographic column names in different parts of the workflow. That reflects the code as implemented. Anyone adapting the repository to a new environment should verify these naming conventions carefully before execution.

### Model Status

The supplied model artifacts and code are provided for research transparency. They are not presented as a medical device and should not be interpreted as carrying regulatory approval or clinical assurance.

## Corrections

- 23/08/2025: sample size estimators corrected.
- 29/05/2026: conservative standard deviation-based function naming corrected and canonical package versions pinned along with complete README re-write.
- 30/05/2026: fixed hardcoded precision adjustment error missed in initial final cohort size estimator.

## Licence

This project and the associated model artifacts are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. The copyright holders are Matt Stammers and University Hospital Southampton NHS Foundation Trust.

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
