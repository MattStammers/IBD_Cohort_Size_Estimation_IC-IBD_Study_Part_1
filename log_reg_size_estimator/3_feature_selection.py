import pandas as pd
import statsmodels.api as sm
import json
import os

def main():
    """
    Main function to perform regularized logistic regression feature selection.

    The process includes:
    1) Defining file paths for preprocessed training/validation data, target variable, and output files.
    2) Loading the preprocessed data and the target variable.
    3) Excluding demographic features using a combined regular expression.
    4) Fitting a regularized logistic regression model (using L1 penalty) to the training data.
    5) Selecting features based on a coefficient threshold (features with absolute coefficient greater than the threshold are retained).
    6) Saving the selected features to a text file and creating subset data for training and validation.
    7) Creating a mapping from feature names to their column indices and saving it as a JSON file.
    """
    # Define paths relative to the script's directory
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(script_dir, '../data'))
    log_reg_dir = os.path.join(data_dir, 'log_reg')

    # Define file paths for preprocessed data and outputs
    preprocessed_train_file = os.path.join(log_reg_dir, 'preprocessed_train.csv')
    y_train_file = os.path.join(log_reg_dir, 'y_train.csv')
    preprocessed_val_file = os.path.join(log_reg_dir, 'preprocessed_val.csv')
    
    selected_features_file = os.path.join(log_reg_dir, 'selected_features.txt')
    selected_train_file = os.path.join(log_reg_dir, 'selected_train.csv')
    selected_val_file = os.path.join(log_reg_dir, 'selected_val.csv')
    selected_indices_file = os.path.join(log_reg_dir, 'selected_indices.json')

    # Define demographic patterns (base names for demographic features, intended for exclusion via regex)
    demo_patterns = [
        '^age_at_referral_gradings_deduplicated',
        '^sex_gradings_deduplicated',
        '^ethnicity_gradings_deduplicated',
        '^imd_decile_gradings_deduplicated'
    ]
    # Combine multiple patterns into a single regex pattern using the OR operator
    regex_pattern = '|'.join(demo_patterns)

    # ------------------------------------------------------------------------
    # 1) Load preprocessed data
    # ------------------------------------------------------------------------
    # Read training features and target variable from CSV files
    X_train = pd.read_csv(preprocessed_train_file)
    y_train = pd.read_csv(y_train_file).squeeze()
    X_val = pd.read_csv(preprocessed_val_file)

    # ------------------------------------------------------------------------
    # 2) Exclude demographic features using regex
    # ------------------------------------------------------------------------
    # Identify columns in training data that match the demographic regex pattern
    cols_to_drop = X_train.filter(regex=regex_pattern).columns
    # Drop the identified demographic columns from the training dataset to build the model inputs
    X_train_model = X_train.drop(columns=cols_to_drop)
    
    # For validation data, identify and drop columns matching the demographic pattern;
    # make sure to drop only those columns that exist in the dataframe.
    cols_to_drop_val = X_val.filter(regex=regex_pattern).columns
    X_val_model = X_val.drop(columns=cols_to_drop_val)

    # ------------------------------------------------------------------------
    # 3) Fit the regularized logistic regression model
    # ------------------------------------------------------------------------
    # Add a constant column to the training features (intercept term for the model)
    X_train_const = sm.add_constant(X_train_model)
    # Initialize the logistic regression model with the training target variable and features
    model = sm.Logit(y_train, X_train_const)
    # Fit the model with L1 regularization. The disp=0 parameter suppresses convergence messages.
    result = model.fit_regularized(method='l1', disp=0)
    
    # ------------------------------------------------------------------------
    # 4) Feature Selection
    # ------------------------------------------------------------------------
    # Define a threshold to determine which features to keep. Features with coefficients whose absolute
    # value is greater than the threshold will be considered significant.
    threshold = 1e-4
    # Drop the constant and then select features based on the threshold
    coef = result.params.drop('const')
    selected_features = coef.index[abs(coef) > threshold].tolist()

    # Save the list of selected features to a text file, one feature per line.
    with open(selected_features_file, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print(f"Selected {len(selected_features)} features (demographic features excluded).")

    # ------------------------------------------------------------------------
    # 5) Create subset data and save
    # ------------------------------------------------------------------------
    # Create a subset of the training and validation data with only the selected features
    selected_train = X_train_model[selected_features]
    selected_val = X_val_model[selected_features]

    # Save the subset data to CSV files for later use
    selected_train.to_csv(selected_train_file, index=False)
    selected_val.to_csv(selected_val_file, index=False)

    print("Feature selection completed successfully.")
    print(f"Selected features saved to {selected_features_file}")
    print(f"Selected training data saved to {selected_train_file}")
    print(f"Selected validation data saved to {selected_val_file}")

    # ------------------------------------------------------------------------
    # 6) Create feature-to-index mapping
    # ------------------------------------------------------------------------
    # Retrieve all column names (features) from the training data used in the model
    all_cols = X_train_model.columns.tolist()
    # Create a mapping from each selected feature to its index in the full training set
    feature_to_index = {feat: all_cols.index(feat) for feat in selected_features}

    # Save the feature-to-index mapping as a JSON file for future reference
    with open(selected_indices_file, 'w') as f_json:
        json.dump(feature_to_index, f_json, indent=4)

    print(f"Feature-to-index mapping saved to {selected_indices_file}")

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()
