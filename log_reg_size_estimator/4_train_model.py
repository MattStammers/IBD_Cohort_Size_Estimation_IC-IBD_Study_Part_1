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

# Import standard libraries for statistical calculations (for the one-standard-error rule)
from statistics import mean, stdev

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

    For each parameter value in param_values, performs cross-validation with inner_cv and
    computes the performance metric (using the specified scoring, currently supports 'roc_auc').
    The function then selects the simplest model whose mean performance is within one standard
    error of the best (highest mean) performance found.

    Parameters:
    - X: Feature array for training.
    - y: Target array for training.
    - model: A scikit-learn pipeline or estimator on which the hyperparameter is set.
    - param_name: Name of the hyperparameter to be tuned (e.g., 'lasso_lr__C').
    - param_values: List or array of potential values for the hyperparameter.
    - scoring: Scoring metric to optimize ('roc_auc' is implemented).
    - inner_cv: Cross-validation splitter (an instance of KFold or similar) for inner CV.

    Returns:
    - chosen_param: The hyperparameter value selected by the one-standard-error rule.
    - cv_results: A list of dictionaries containing for each parameter value:
        'param_value', 'mean_score', 'std_score', and 'all_scores'.
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
    # For all parameter values with best mean, take the maximum standard deviation as reference
    best_std = max(r['std_score'] for r in cv_results if r['mean_score'] == best_mean)
    
    # Define the threshold: one standard error below the best mean score
    one_se_threshold = best_mean - best_std
    
    # Among parameters with a mean score above the threshold, choose the simplest model.
    # For L1 logistic regression, smaller C implies more regularization (thus, "simpler").
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
    Main function to perform nested cross-validation for logistic regression with L1 penalty
    using the one-standard-error rule for hyperparameter selection.

    The process includes:
    1) Loading the training data and target.
    2) Building a pipeline that includes feature scaling and L1-regularized logistic regression.
    3) Running nested cross-validation:
       - Outer CV: Repeated K-Fold to estimate model performance.
       - Inner CV: K-Fold CV to select the hyperparameter (regularization parameter C) using the
         one-standard-error rule.
    4) Reporting cross-validation performance metrics (ROC AUC and Brier score).
    5) Fitting a final model on the entire training set using the most frequently chosen parameter.
    6) Saving the final pipeline to disk.
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
    # 2) Build a pipeline for scaling and logistic regression (L1 penalty, saga solver)
    # -------------------------------------------------------------------------
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso_lr', LogisticRegression(
            penalty='l1',
            solver='saga',
            max_iter=10000,
            random_state=42
        ))
    ])

    # -------------------------------------------------------------------------
    # 3) Perform nested cross-validation
    #    Outer CV: Repeated K-Fold (10 splits, 2 repeats).
    #    Inner CV: K-Fold (10 splits) for hyperparameter (C) selection using one-standard-error rule.
    # -------------------------------------------------------------------------
    outer_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    # Define a range of candidate values for the inverse regularization parameter C
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
        # ---------------------------------------------------------------------
        chosen_param, cv_results = one_standard_error_rule_search(
            X_tr, 
            y_tr, 
            model=pipeline, 
            param_name='lasso_lr__C', 
            param_values=param_values, 
            scoring='roc_auc', 
            inner_cv=inner_cv
        )
        chosen_params_across_folds.append(chosen_param)
        
        # ---------------------------------------------------------------------
        # (B) Retrain the pipeline on the outer training data using the chosen parameter.
        # ---------------------------------------------------------------------
        pipeline.set_params(lasso_lr__C=chosen_param)
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

    print("Nested CV Results (One-Standard-Error Rule)")
    print(f"  AUC:   {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Brier: {mean_brier:.4f} ± {std_brier:.4f}")
    print("Chosen C parameters in each outer fold:")
    print(chosen_params_across_folds)

    # -------------------------------------------------------------------------
    # 5) Fit the final model on all the data.
    #    (Here, the parameter that appears most frequently in outer CV is selected.)
    # -------------------------------------------------------------------------
    final_chosen_param = max(set(chosen_params_across_folds),
                             key=chosen_params_across_folds.count)

    pipeline.set_params(lasso_lr__C=final_chosen_param)
    pipeline.fit(X_train, y_train)

    print(f"\nFinal chosen C after nested CV voting: {final_chosen_param:.6f}")

    # -------------------------------------------------------------------------
    # 6) Save the final pipeline.
    # -------------------------------------------------------------------------
    joblib.dump(pipeline, final_model_file)
    print(f"Trained pipeline (scaler + L1 logistic) saved to {final_model_file}")

if __name__ == "__main__":
    # Execute the main function when running this script directly
    main()
