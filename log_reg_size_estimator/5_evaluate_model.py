import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve

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

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, specificity, accuracy

def bootstrap_auc(y_true, y_pred_proba, n_boot=1000, alpha=0.05):
    """
    Compute bootstrap confidence intervals for ROC AUC.
    """
    aucs = []
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    n = len(y_true)
    for i in range(n_boot):
        indices = np.random.choice(n, size=n, replace=True)
        try:
            auc_val = roc_auc_score(y_true[indices], y_pred_proba[indices])
            aucs.append(auc_val)
        except Exception:
            # Skip bootstrap samples that cause errors (e.g., only one class present)
            continue
    lower = np.percentile(aucs, 100 * alpha / 2)
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    return lower, upper

def bootstrap_brier(y_true, y_pred_proba, n_boot=1000, alpha=0.05):
    """
    Compute bootstrap confidence intervals for the Brier score.
    """
    briers = []
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    n = len(y_true)
    for i in range(n_boot):
        indices = np.random.choice(n, size=n, replace=True)
        try:
            brier_val = brier_score_loss(y_true[indices], y_pred_proba[indices])
            briers.append(brier_val)
        except Exception:
            continue
    lower = np.percentile(briers, 100 * alpha / 2)
    upper = np.percentile(briers, 100 * (1 - alpha / 2))
    return lower, upper

def bootstrap_classification_metrics(y_true, y_pred_proba, threshold, n_boot=1000, alpha=0.05):
    """
    Bootstrap confidence intervals for classification metrics at a given threshold.
    Returns a dictionary with 95% CI tuples for precision, recall, specificity, and accuracy.
    """
    metrics = {"precision": [], "recall": [], "specificity": [], "accuracy": []}
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    n = len(y_true)
    for i in range(n_boot):
        indices = np.random.choice(n, size=n, replace=True)
        prec, rec, spec, acc = compute_classification_metrics(y_true[indices], y_pred_proba[indices], threshold)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["specificity"].append(spec)
        metrics["accuracy"].append(acc)
    ci = {}
    for key, values in metrics.items():
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        ci[key] = (lower, upper)
    return ci

def main():
    """
    Main function to evaluate a trained model using various classification and calibration metrics.
    Now also computes bootstrapped 95% confidence intervals for overall and threshold-specific metrics.
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
    auc_ci = bootstrap_auc(y_val, y_pred_proba)
    brier_ci = bootstrap_brier(y_val, y_pred_proba)
    print(f"Validation AUC: {auc:.4f} (95% CI: {auc_ci[0]:.4f}-{auc_ci[1]:.4f})")
    print(f"Brier Score: {brier:.4f} (95% CI: {brier_ci[0]:.4f}-{brier_ci[1]:.4f})")

    # -------------------------
    # 4) Determine the optimal classification threshold using Youden's J statistic
    # -------------------------
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    J = tpr - fpr  # Youden's J statistic for each threshold
    ix = np.argmax(J)
    optimal_threshold = thresholds[ix]
    print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")

    with open(optimal_threshold_file, 'w') as f:
        f.write(str(optimal_threshold))

    # -------------------------
    # 5) Compute classification metrics and their confidence intervals across a range of thresholds
    # -------------------------
    threshold_range = np.linspace(0.25, 0.75, 9)  # e.g. thresholds from 0.25 to 0.75
    threshold_metrics = {}

    for t in threshold_range:
        precision, recall, specificity, accuracy = compute_classification_metrics(y_val, y_pred_proba, t)
        ci = bootstrap_classification_metrics(y_val, y_pred_proba, t)
        threshold_metrics[round(t, 2)] = {
            'Sensitivity (Recall)': {'value': recall, '95% CI': ci['recall']},
            'Specificity': {'value': specificity, '95% CI': ci['specificity']},
            'PPV (Precision)': {'value': precision, '95% CI': ci['precision']},
            'Accuracy': {'value': accuracy, '95% CI': ci['accuracy']}
        }

    print("Threshold analysis metrics (Sensitivity, Specificity, PPV, Accuracy with 95% CIs):")
    for t, m in threshold_metrics.items():
        print(f"Threshold {t}: {m}")

    # -------------------------
    # 6) Save evaluation metrics (with CIs) to a JSON file
    # -------------------------
    metrics = {
        'AUC': auc,
        'AUC 95% CI': {'lower': auc_ci[0], 'upper': auc_ci[1]},
        'Brier Score': brier,
        'Brier Score 95% CI': {'lower': brier_ci[0], 'upper': brier_ci[1]},
        'Optimal Threshold (Youden\'s J)': float(optimal_threshold),
        'Threshold Analysis': threshold_metrics
    }

    with open(evaluation_metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    # -------------------------
    # 7) Generate Calibration Plots
    # -------------------------
    # (a) Standard calibration curve using scikit-learn's calibration_curve
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

    # (b) LOESS calibration plot using seaborn.regplot (with CI)
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_pred_proba, y=y_val, 
                scatter_kws={"s": 20, "alpha": 0.5},
                lowess=True, 
                ci=95)
    plt.title('Calibration with LOESS')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Outcome')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(calibration_loess_file)
    plt.close()
    print(f"LOESS calibration plot saved to {calibration_loess_file}")

    # (c) Logistic regression calibration plot using seaborn.regplot (with CI)
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_pred_proba, y=y_val, 
                scatter_kws={"s": 20, "alpha": 0.5},
                logistic=True, 
                ci=95)
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
    main()
