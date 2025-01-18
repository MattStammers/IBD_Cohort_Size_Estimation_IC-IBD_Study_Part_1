#!/usr/bin/env python
import pandas as pd
import joblib
import json
import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from fairlearn.metrics import MetricFrame
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix

############################
# Custom Metric Definitions
############################

def specificity(y_true, y_pred):
    """
    Calculate specificity as the recall of the negative class (i.e. true negative rate).

    Specificity is defined as:
        TN / (TN + FP)
    If y_true does not contain at least two unique classes, returns NaN.

    Parameters:
    - y_true: array-like of true binary labels.
    - y_pred: array-like of predicted binary labels.

    Returns:
    - specificity value (float) or NaN if not computable.
    """
    if len(set(y_true)) < 2:
        return float('nan')
    return recall_score(y_true, y_pred, pos_label=0)

def custom_false_positive_rate(y_true, y_pred):
    """
    Calculate the false positive rate (FPR).

    FPR is defined as:
        FP / (FP + TN)
    If y_true or y_pred do not contain at least two unique classes, returns NaN.

    Parameters:
    - y_true: array-like of true binary labels.
    - y_pred: array-like of predicted binary labels.

    Returns:
    - false positive rate (float) or NaN if not computable.
    """
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return float('nan')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)

def custom_false_negative_rate(y_true, y_pred):
    """
    Calculate the false negative rate (FNR).

    FNR is defined as:
        FN / (FN + TP)
    If y_true or y_pred do not contain at least two unique classes, returns NaN.

    Parameters:
    - y_true: array-like of true binary labels.
    - y_pred: array-like of predicted binary labels.

    Returns:
    - false negative rate (float) or NaN if not computable.
    """
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return float('nan')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp)

def custom_auc_score(y_true, y_pred=None, y_prob=None, **kwargs):
    """
    Compute the AUC (Area Under the Receiver Operating Characteristic Curve)
    using predicted probabilities.

    Parameters:
    - y_true: array-like of true binary labels.
    - y_pred: (ignored) placeholder to support MetricFrame.
    - y_prob: array-like of predicted probabilities for the positive class.
    - **kwargs: Additional keyword arguments (unused).

    Returns:
    - AUC value (float). If y_true contains less than two unique classes, returns NaN.
    """
    if len(set(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_prob)

##############################
# Binning / Categorization
##############################

def bin_age_10_100(age):
    """
    Bin age into 10‐year intervals from 10 to 100.

    Bins:
      - [0,10): "<10"
      - [10,20): "10-19"
      - [20,30): "20-29"
      - [30,40): "30-39"
      - [40,50): "40-49"
      - [50,60): "50-59"
      - [60,70): "60-69"
      - [70,80): "70-79"
      - [80,90): "80-89"
      - [90,100): "90-99"
      - [100,∞): "100+"
      
    If age is missing, returns "Unknown".

    Parameters:
    - age: numeric value representing age.

    Returns:
    - A string indicating the age bin.
    """
    if pd.isna(age):
        return "Unknown"
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
    labels = ["<10", "10-19", "20-29", "30-39", "40-49", "50-59",
              "60-69", "70-79", "80-89", "90-99", "100+"]
    return str(pd.cut([age], bins=bins, labels=labels, right=False)[0])

def bin_imd(imd_value):
    """
    Bin IMD (Index of Multiple Deprivation) values into predefined categories.

    Bins:
      - "1-2" for values between 1 and 2,
      - "3-4" for values between 3 and 4,
      - "5-6" for values between 5 and 6,
      - "7-8" for values between 7 and 8,
      - "9-10" for values between 9 and 10,
      - "Unknown" for missing values,
      - "OutOfRange" for any values outside the 1-10 range.

    Parameters:
    - imd_value: numeric value representing the IMD.

    Returns:
    - A string indicating the IMD bin.
    """
    if pd.isna(imd_value):
        return "Unknown"
    elif 1 <= imd_value <= 2:
        return "1-2"
    elif 3 <= imd_value <= 4:
        return "3-4"
    elif 5 <= imd_value <= 6:
        return "5-6"
    elif 7 <= imd_value <= 8:
        return "7-8"
    elif 9 <= imd_value <= 10:
        return "9-10"
    else:
        return "OutOfRange"

##########################
# Evaluation & Plotting
##########################

def evaluate_bias(y_true, y_pred, y_pred_proba, sensitive_feature):
    """
    Evaluate multiple metrics by groups defined by a sensitive feature using Fairlearn's MetricFrame.

    The function calculates several performance metrics for each subgroup in the sensitive feature.
    Metrics include AUC, sensitivity (recall), specificity, precision, false positive rate,
    and false negative rate.

    Parameters:
    - y_true: array-like of true binary labels.
    - y_pred: array-like of predicted binary labels.
    - y_pred_proba: array-like of predicted probabilities for the positive class.
    - sensitive_feature: array-like of sensitive feature values (must be convertible to strings).

    Returns:
    - results: A dictionary mapping each metric name to a dictionary of subgroup metric values.
    - metric_frame: The Fairlearn MetricFrame containing the grouped metrics.
    """
    metrics_dict = {
        'AUC': custom_auc_score,
        'Sensitivity (Recall)': recall_score,
        'Specificity': specificity,
        'Precision': precision_score,
        'False Positive Rate': custom_false_positive_rate,
        'False Negative Rate': custom_false_negative_rate
    }
    # For AUC, provide y_pred_proba as an extra parameter.
    sample_params = {'AUC': {'y_prob': y_pred_proba}}
    
    # Convert sensitive feature values to strings
    sensitive_feature = np.array(sensitive_feature, dtype=str)
    
    metric_frame = MetricFrame(
        metrics=metrics_dict,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature,
        sample_params=sample_params
    )
    
    # Extract the metric results by group into a dictionary
    results = {}
    for name in metrics_dict.keys():
        results[name] = metric_frame.by_group[name].to_dict()
    
    return results, metric_frame

def plot_metric(metric_frame, metric_name, group_name, output_path):
    """
    Plot a single metric by group as a bar chart.

    Uses a default ordering for known sensitive features if available. If not,
    the order follows the index order from the MetricFrame.

    Parameters:
    - metric_frame: Fairlearn MetricFrame with grouped metric values.
    - metric_name: Name of the metric to plot (string).
    - group_name: Name of the sensitive feature group (string) used to determine ordering.
    - output_path: File path to save the generated plot.

    Returns:
    - None; saves the plot to the specified output path.
    """
    plt.figure(figsize=(10, 6))
    data = metric_frame.by_group[metric_name]
    
    # Define a default ordering for known sensitive features
    default_order = {
        "age_decile_10_100": ["<10", "10-19", "20-29", "30-39", "40-49",
                               "50-59", "60-69", "70-79", "80-89", "90-99", "100+"],
        "imd_bin": ["1-2", "3-4", "5-6", "7-8", "9-10", "Unknown", "OutOfRange"],
        "sex": ["Male", "Female", "Unknown"],
        "ethnicity": []  # Add explicit ordering if needed.
    }
    
    if group_name in default_order and default_order[group_name]:
        order = default_order[group_name]
        # Reindex data based on the default order
        data = data.reindex(order)
        groups = order
        values = data.values
    else:
        groups = list(data.index)
        values = data.values
    
    sns.barplot(x=groups, y=values, order=groups)
    plt.title(f'{metric_name} by {group_name.capitalize()}')
    plt.ylabel(metric_name)
    plt.xlabel(group_name.capitalize())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

################
# Main Function
################

def main():
    """
    Main function for fairness evaluation and plotting using a trained model.

    The process includes:
      1) Defining file paths and loading the full dataset.
      2) Loading the preprocessor and final model, ensuring the dataset contains the expected raw columns.
      3) Dropping extra columns (except demographics) and applying the preprocessor.
      4) Selecting final features based on indices from a JSON file.
      5) Generating predictions (probabilities and classes) using the model.
      6) Binning sensitive demographic features (age, IMD, sex, ethnicity) for subgroup analysis.
      7) Evaluating bias (performance metrics) for each demographic group using Fairlearn's MetricFrame.
      8) Plotting metrics by subgroup and saving the resulting plots.
      9) Saving all bias metric results to a JSON file.

    Returns:
    - None; all outputs are saved to specified files.
    """
    # Define paths relative to this script's location.
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(script_dir, '../data'))
    log_reg_dir = os.path.join(data_dir, 'log_reg')
    
    merged_dataframe_file = os.path.join(data_dir, 'final_dataframe.csv')
    
    # Load the full dataset for bias analysis.
    full_df = pd.read_csv(merged_dataframe_file)
    
    # Load the preprocessor and the final model.
    preprocessor_file = os.path.join(log_reg_dir, 'preprocessor.pkl')
    final_model_file = os.path.join(log_reg_dir, 'final_model.pkl')
    preprocessor = joblib.load(preprocessor_file)
    model = joblib.load(final_model_file)
    
    # Ensure full_df contains the expected raw columns.
    if hasattr(preprocessor, 'feature_names_in_'):
        expected_raw_cols = list(preprocessor.feature_names_in_)
        for col in expected_raw_cols:
            if col not in full_df.columns:
                full_df[col] = 0
    else:
        raise ValueError("Preprocessor does not have 'feature_names_in_'.")
    
    # Drop extra columns that are not expected by the pipeline (but keep demographic columns).
    demographic_cols = ['study_id', 'IBD', 'age_at_referral_gradings_deduplicated', 
                        'imd_decile_gradings_deduplicated', 'sex_gradings_deduplicated', 
                        'ethnicity_gradings_deduplicated']
    extra_cols = [col for col in full_df.columns if col not in expected_raw_cols + demographic_cols]
    if extra_cols:
        full_df.drop(columns=extra_cols, inplace=True)
    
    # Transform the full dataset using the preprocessor.
    X_full = preprocessor.transform(full_df[expected_raw_cols])
    
    # Use get_feature_names_out if available, else fallback to expected_raw_cols.
    if hasattr(preprocessor, "get_feature_names_out"):
        new_feature_names = preprocessor.get_feature_names_out()
    else:
        new_feature_names = expected_raw_cols
    
    # Convert the transformed numpy array into a DataFrame with proper feature names.
    X_full = pd.DataFrame(X_full, columns=new_feature_names)
    
    # Load the JSON file containing the indices of the selected features.
    selected_indices_file = os.path.join(log_reg_dir, 'selected_indices.json')
    with open(selected_indices_file, 'r') as f_json:
        feature_to_index = json.load(f_json)
    selected_indices = list(feature_to_index.values())
    
    # Select the final features from the transformed data.
    X_final = X_full.iloc[:, selected_indices].values  # Use .values for model prediction.
    
    # Generate predictions on the full dataset.
    y_pred_proba = model.predict_proba(X_final)[:, 1]
    y_pred = model.predict(X_final)
    
    # Use ground truth labels from full_df if available.
    y_true = full_df['IBD'] if 'IBD' in full_df.columns else None

    # --- Subset rows with valid ground truth ---
    valid_idx = y_true.notna()
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]
    y_pred_proba = y_pred_proba[valid_idx]
    full_df = full_df.loc[valid_idx].copy()
    
    # --- Demographic Processing ---
    # Create a copy for demographic analysis.
    demographic_df = full_df.copy()
    
    # For age, bin the ages using the binning function.
    demographic_df['age_decile_10_100'] = demographic_df['age_at_referral_gradings_deduplicated'].apply(bin_age_10_100)
    demographic_df['age_decile_10_100'] = demographic_df['age_decile_10_100'].astype(str)
    
    # Process IMD by applying the bin_imd function and converting to string.
    demographic_df['imd_bin'] = demographic_df['imd_decile_gradings_deduplicated'].apply(bin_imd).astype(str)
    
    # Process sex and ethnicity, filling missing values with "Unknown" and converting to string.
    demographic_df['sex'] = demographic_df['sex_gradings_deduplicated'].fillna("Unknown").astype(str)
    demographic_df['ethnicity'] = demographic_df['ethnicity_gradings_deduplicated'].fillna("Unknown").astype(str)
    
    # Initialize a dictionary to store all bias metrics.
    all_bias_metrics = {}
    
    #############################
    # Evaluate bias by Age (10-100)
    #############################
    age_sensitive = demographic_df['age_decile_10_100'].astype(str)
    age_metrics, age_metric_frame = evaluate_bias(
        y_true, y_pred, y_pred_proba, sensitive_feature=age_sensitive
    )
    all_bias_metrics['age_10_100'] = age_metrics
    out_auc_age = os.path.join(log_reg_dir, 'auc_age_10_100.png')
    plot_metric(age_metric_frame, 'AUC', 'age_decile_10_100', out_auc_age)
    out_sens_age = os.path.join(log_reg_dir, 'sensitivity_age_10_100.png')
    plot_metric(age_metric_frame, 'Sensitivity (Recall)', 'age_decile_10_100', out_sens_age)
    
    ######################
    # Evaluate bias by IMD
    ######################
    imd_sensitive = demographic_df['imd_bin'].astype(str)
    imd_metrics, imd_metric_frame = evaluate_bias(
        y_true, y_pred, y_pred_proba, sensitive_feature=imd_sensitive
    )
    all_bias_metrics['imd_bin'] = imd_metrics
    out_auc_imd = os.path.join(log_reg_dir, 'auc_imd_bin.png')
    plot_metric(imd_metric_frame, 'AUC', 'imd_bin', out_auc_imd)
    out_sens_imd = os.path.join(log_reg_dir, 'sensitivity_imd_bin.png')
    plot_metric(imd_metric_frame, 'Sensitivity (Recall)', 'imd_bin', out_sens_imd)
    
    #######################
    # Evaluate bias by Sex
    #######################
    sex_sensitive = demographic_df['sex'].astype(str)
    sex_metrics, sex_metric_frame = evaluate_bias(
        y_true, y_pred, y_pred_proba, sensitive_feature=sex_sensitive
    )
    all_bias_metrics['sex'] = sex_metrics
    out_auc_sex = os.path.join(log_reg_dir, 'auc_sex.png')
    plot_metric(sex_metric_frame, 'AUC', 'sex', out_auc_sex)
    
    #############################
    # Evaluate bias by Ethnicity
    #############################
    ethnicity_sensitive = demographic_df['ethnicity'].astype(str)
    eth_metrics, eth_metric_frame = evaluate_bias(
        y_true, y_pred, y_pred_proba, sensitive_feature=ethnicity_sensitive
    )
    all_bias_metrics['ethnicity'] = eth_metrics
    out_auc_eth = os.path.join(log_reg_dir, 'auc_ethnicity.png')
    plot_metric(eth_metric_frame, 'AUC', 'ethnicity', out_auc_eth)
    
    # Save all bias metrics to a JSON file.
    bias_metrics_file = os.path.join(log_reg_dir, 'bias_metrics.json')
    with open(bias_metrics_file, 'w') as f:
        json.dump(all_bias_metrics, f, indent=4)

if __name__ == "__main__":
    # Execute main if this script is run directly.
    main()
