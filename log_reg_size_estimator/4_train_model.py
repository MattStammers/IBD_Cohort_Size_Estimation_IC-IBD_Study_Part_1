import os
import numpy as np
import pandas as pd
import joblib

# Import core machine learning libraries
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

# Import calibration module for platt regression calibration
from sklearn.calibration import CalibratedClassifierCV

# Optional additional imports (for visualization and fairness analysis)
import seaborn as sns
import matplotlib.pyplot as plt
import fairlearn  # Include fairness metrics if needed; otherwise, remove

def one_standard_error_rule_search(
    X, 
    y, 
    model, 
    param_name, 
    param_values, 
    scoring, 
    inner_cv
):
    """
    Perform a custom hyperparameter search using the one-standard-error rule.
    """
    cv_results = []
    
    # Loop over each candidate hyperparameter value
    for val in param_values:
        # Set the hyperparameter value in the model
        model.set_params(**{param_name: val})
        
        scores = []
        # Perform inner cross-validation
        for train_index, valid_index in inner_cv.split(X):
            X_train_fold, X_valid_fold = X[train_index], X[valid_index]
            y_train_fold, y_valid_fold = y[train_index], y[valid_index]
            
            # Fit the model on the training fold
            model.fit(X_train_fold, y_train_fold)
            
            # Predict probabilities for the positive class
            y_prob = model.predict_proba(X_valid_fold)[:, 1]
            
            # Compute the scoring metric (currently only 'roc_auc' is supported)
            if scoring == 'roc_auc':
                fold_score = roc_auc_score(y_valid_fold, y_prob)
            else:
                raise ValueError("Only 'roc_auc' is implemented here.")
            
            scores.append(fold_score)
        
        # Store the cross-validation results for this hyperparameter value
        cv_results.append({
            'param_value': val,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'all_scores': scores
        })
    
    # Determine the best mean score achieved among all hyperparameter values
    best_mean = max(r['mean_score'] for r in cv_results)
    best_std = max(r['std_score'] for r in cv_results if r['mean_score'] == best_mean)
    
    # Define the threshold: one standard error below the best mean score
    one_se_threshold = best_mean - best_std
    
    # Among parameters with a mean score above the threshold, choose the simplest model.
    candidates = [
        (r['param_value'], r['mean_score']) 
        for r in cv_results 
        if r['mean_score'] >= one_se_threshold
    ]
    
    # Sort candidates: primary sort by descending mean score; secondary by ascending parameter value.
    candidates.sort(key=lambda x: (-x[1], x[0]))
    chosen_param = candidates[0][0]
    
    return chosen_param, cv_results


def main():
    """
    Main function to perform nested cross-validation for a pipeline that calibrates a 
    logistic regression model using platt scaling.
    """
    # -------------------------------------------------------------------------
    # 1) Load the data
    # -------------------------------------------------------------------------
    # Define paths relative to the script location
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(script_dir, '../data'))
    log_reg_dir = os.path.join(data_dir, 'log_reg')

    # File paths for training features, targets, and the final model
    selected_train_file = os.path.join(log_reg_dir, 'selected_train.csv')
    y_train_file = os.path.join(log_reg_dir, 'y_train.csv')
    final_model_file = os.path.join(log_reg_dir, 'final_model.pkl')

    # Load training data and target variable; convert to NumPy arrays for CV splits
    X_train = pd.read_csv(selected_train_file).values
    y_train = pd.read_csv(y_train_file).squeeze().values

    # -------------------------------------------------------------------------
    # 2) Build a pipeline for scaling, base logistic regression, and platt calibration
    # -------------------------------------------------------------------------
    # Define the base logistic regression model with elasticnet penalty.
    base_lr = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5, 
        max_iter=10000,
        random_state=42
    )
    
    # Wrap the base logistic regression in a calibrated classifier using platt regression.
    calibrated_clf = CalibratedClassifierCV(estimator=base_lr, method='sigmoid', cv=5)

    # Build the pipeline with a scaler and the calibrated classifier.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('calibrated', calibrated_clf)
    ])

    # -------------------------------------------------------------------------
    # 3) Perform nested cross-validation
    # -------------------------------------------------------------------------
    outer_cv = RepeatedKFold(n_splits=10, n_repeats=100, random_state=42)

    # Define a range of candidate values for the inverse regularization parameter C.
    # (This parameter belongs to the base logistic regression inside the calibrated classifier.)
    param_values = np.logspace(-4, 4, 20)

    outer_auc_scores = []
    outer_brier_scores = []
    chosen_params_across_folds = []

    # Outer cross-validation loop
    for outer_train_idx, outer_test_idx in outer_cv.split(X_train):
        X_tr, X_te = X_train[outer_train_idx], X_train[outer_test_idx]
        y_tr, y_te = y_train[outer_train_idx], y_train[outer_test_idx]
        
        # Set up inner cross-validation
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)

        # ---------------------------------------------------------------------
        # (A) Use the one-standard-error rule to search for the best hyperparameter (C)
        #     Since the parameter is inside the calibrated classifier, use 'calibrated__estimator__C'
        # ---------------------------------------------------------------------
        chosen_param, cv_results = one_standard_error_rule_search(
            X_tr, 
            y_tr, 
            model=pipeline, 
            param_name='calibrated__estimator__C', 
            param_values=param_values, 
            scoring='roc_auc', 
            inner_cv=inner_cv
        )
        chosen_params_across_folds.append(chosen_param)
        
        # ---------------------------------------------------------------------
        # (B) Retrain the pipeline on the outer training data using the chosen parameter.
        # ---------------------------------------------------------------------
        pipeline.set_params(calibrated__estimator__C=chosen_param)
        pipeline.fit(X_tr, y_tr)
        
        # Evaluate the trained model on the outer test set
        y_prob_test = pipeline.predict_proba(X_te)[:, 1]
        
        fold_auc = roc_auc_score(y_te, y_prob_test)
        fold_brier = brier_score_loss(y_te, y_prob_test)

        outer_auc_scores.append(fold_auc)
        outer_brier_scores.append(fold_brier)

    # -------------------------------------------------------------------------
    # 4) Report nested cross-validation results.
    # -------------------------------------------------------------------------
    mean_auc = np.mean(outer_auc_scores)
    std_auc = np.std(outer_auc_scores)

    mean_brier = np.mean(outer_brier_scores)
    std_brier = np.std(outer_brier_scores)

    print("Nested CV Results (platt Calibration)")
    print(f"  AUC:   {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Brier: {mean_brier:.4f} ± {std_brier:.4f}")
    print("Chosen C parameters in each outer fold:")
    print(chosen_params_across_folds)

    # -------------------------------------------------------------------------
    # 5) Fit the final model on all the data.
    # -------------------------------------------------------------------------
    final_chosen_param = max(set(chosen_params_across_folds),
                             key=chosen_params_across_folds.count)

    pipeline.set_params(calibrated__estimator__C=final_chosen_param)
    pipeline.fit(X_train, y_train)

    print(f"\nFinal chosen C after nested CV voting: {final_chosen_param:.6f}")

    # -------------------------------------------------------------------------
    # 6) Save the final pipeline.
    # -------------------------------------------------------------------------
    joblib.dump(pipeline, final_model_file)
    print(f"Trained pipeline (scaler + calibrated logistic regression with platt regression) saved to {final_model_file}")

if __name__ == "__main__":
    main()
