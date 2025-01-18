import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve, confusion_matrix
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Imported for LOESS/logistic calibration plots
import seaborn as sns

def compute_classification_metrics(y_true, y_pred_proba, threshold):
    """
    Compute classification metrics at a specified probability threshold.

    This function calculates the following metrics:
      - Precision (Positive Predictive Value)
      - Recall (Sensitivity)
      - Specificity
      - Accuracy

    Parameters:
    - y_true: Array-like, true binary labels.
    - y_pred_proba: Array-like, predicted probabilities for the positive class.
    - threshold: Float, probability threshold to convert probabilities to class predictions.

    Returns:
    - precision: Computed precision (PPV) value.
    - recall: Computed recall (sensitivity) value.
    - specificity: Computed specificity value.
    - accuracy: Computed overall accuracy.
    """
    # Convert predicted probabilities to binary class predictions based on the threshold
    y_pred_class = (y_pred_proba >= threshold).astype(int)
    # Obtain confusion matrix components: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()

    # Calculate precision (PPV) as TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    # Calculate recall (sensitivity) as TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    # Calculate specificity as TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # Calculate accuracy as (TP + TN) divided by total samples
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return precision, recall, specificity, accuracy

def main():
    """
    Main function to evaluate a trained model using various classification and calibration metrics.

    The process includes:
    1) Defining file paths for the validation data, model, and various outputs.
    2) Loading the validation features, true labels, and the trained model.
    3) Generating predicted probabilities for the positive class.
    4) Computing overall performance metrics such as ROC AUC and Brier score.
    5) Determining the optimal classification threshold using Youden's J statistic.
    6) Computing classification metrics (sensitivity, specificity, precision, accuracy)
       for a range of thresholds.
    7) Saving evaluation metrics as JSON.
    8) Generating and saving calibration plots:
         - Standard calibration curve.
         - LOESS calibration plot.
         - Logistic regression calibration plot.
    9) Generating and saving the ROC curve plot.
    """
    # Define paths relative to the script's directory
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(script_dir, '../data'))
    log_reg_dir = os.path.join(data_dir, 'log_reg')

    # Define file paths for data inputs and outputs
    selected_val_file = os.path.join(log_reg_dir, 'selected_val.csv')
    y_val_file = os.path.join(log_reg_dir, 'y_val.csv')
    final_model_file = os.path.join(log_reg_dir, 'final_model.pkl')
    evaluation_metrics_file = os.path.join(log_reg_dir, 'evaluation_metrics.json')
    calibration_plot_file = os.path.join(log_reg_dir, 'calibration_plot.png')
    roc_curve_file = os.path.join(log_reg_dir, 'roc_curve.png')
    optimal_threshold_file = os.path.join(log_reg_dir, 'optimal_threshold.txt')
    
    # Additional calibration plots:
    calibration_loess_file = os.path.join(log_reg_dir, 'calibration_loess.png')
    calibration_logistic_file = os.path.join(log_reg_dir, 'calibration_logistic.png')

    # -------------------------
    # 1) Load the validation data and model
    # -------------------------
    X_val = pd.read_csv(selected_val_file)
    y_val = pd.read_csv(y_val_file).squeeze()
    model = joblib.load(final_model_file)

    # -------------------------
    # 2) Generate predictions (probability of the positive class)
    # -------------------------
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # -------------------------
    # 3) Compute overall performance metrics: AUC and Brier Score
    # -------------------------
    auc = roc_auc_score(y_val, y_pred_proba)
    brier = brier_score_loss(y_val, y_pred_proba)
    print(f"Validation AUC: {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    # -------------------------
    # 4) Determine the optimal classification threshold using Youden's J statistic
    # -------------------------
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    J = tpr - fpr  # Youden's J statistic for each threshold
    ix = np.argmax(J)
    optimal_threshold = thresholds[ix]
    print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")

    # Save the optimal threshold for future use
    with open(optimal_threshold_file, 'w') as f:
        f.write(str(optimal_threshold))

    # -------------------------
    # 5) Compute classification metrics across a range of thresholds
    # -------------------------
    threshold_range = np.linspace(0.25, 0.75, 9)  # Define a range of thresholds from 0.25 to 0.75
    threshold_metrics = {}

    # Compute metrics for each threshold in the range
    for t in threshold_range:
        precision, recall, specificity, accuracy = compute_classification_metrics(
            y_val, y_pred_proba, t
        )
        threshold_metrics[round(t, 2)] = {
            'Sensitivity (Recall)': recall,
            'Specificity': specificity,
            'PPV (Precision)': precision,
            'Accuracy': accuracy
        }

    print("Threshold analysis metrics (Sensitivity, Specificity, PPV, Accuracy):")
    for t, m in threshold_metrics.items():
        print(f"Threshold {t}: {m}")

    # -------------------------
    # 6) Save evaluation metrics to a JSON file
    # -------------------------
    metrics = {
        'AUC': auc,
        'Brier Score': brier,
        'Optimal Threshold (Youden\'s J)': float(optimal_threshold),
        'Threshold Analysis': threshold_metrics
    }

    with open(evaluation_metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    # -------------------------
    # 7) Generate Calibration Plots
    # -------------------------
    # (a) Standard calibration curve (using scikit-learn's calibration_curve)
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.savefig(calibration_plot_file)
    plt.close()
    print(f"Calibration curve saved to {calibration_plot_file}")

    # (b) LOESS calibration plot using seaborn.regplot with lowess=True
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_pred_proba, y=y_val, 
                scatter_kws={"s": 20, "alpha": 0.5},
                lowess=True, 
                ci=None)
    plt.title('Calibration with LOESS')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Outcome')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(calibration_loess_file)
    plt.close()
    print(f"LOESS calibration plot saved to {calibration_loess_file}")

    # (c) Logistic regression calibration plot using seaborn.regplot with logistic=True
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_pred_proba, y=y_val, 
                scatter_kws={"s": 20, "alpha": 0.5},
                logistic=True, 
                ci=None)
    plt.title('Calibration with Logistic Regression Fit')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Outcome')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(calibration_logistic_file)
    plt.close()
    print(f"Logistic calibration plot saved to {calibration_logistic_file}")

    # -------------------------
    # 8) Generate and save the ROC Curve plot
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(roc_curve_file)
    plt.close()
    print(f"ROC curve saved to {roc_curve_file}")

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()
