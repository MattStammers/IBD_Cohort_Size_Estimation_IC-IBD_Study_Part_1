import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import os
import joblib

def main():
    """
    Main function to load data, preprocess features, and save the transformed data and preprocessor.

    The process includes:
    - Defining file paths for input, intermediate, and output data.
    - Loading the merged dataframe and dropping rows missing the target 'IBD'.
    - Defining the feature set by excluding 'study_id', target, and demographic columns.
    - Loading previously defined training and validation splits.
    - Identifying numerical and categorical feature columns.
    - Constructing preprocessing pipelines for numerical and categorical data.
    - Fitting the preprocessor on the training data and applying the transformation to both training and validation sets.
    - Reconstructing the feature names, converting processed arrays to DataFrames, and saving them as CSV files.
    - Saving the fitted preprocessor as a pickle file using joblib.
    """
    try:
        # Define paths relative to the script's directory
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.abspath(os.path.join(script_dir, '../data'))
        log_reg_dir = os.path.join(data_dir, 'log_reg')

        # File paths for the input merged dataframe and preprocessed outputs
        preprocessed_train_file = os.path.join(log_reg_dir, 'preprocessed_train.csv')
        preprocessed_val_file = os.path.join(log_reg_dir, 'preprocessed_val.csv')
        merged_dataframe_file = os.path.join(data_dir, 'merged_dataframe.csv')

        # Load merged dataframe from CSV file
        df = pd.read_csv(merged_dataframe_file)

        # Drop rows with missing 'IBD' values to ensure the target variable is complete
        initial_size = len(df)
        df_clean = df.dropna(subset=['IBD'])
        cleaned_size = len(df_clean)
        dropped = initial_size - cleaned_size
        print(f"Dropped {dropped} rows due to missing 'IBD' values.")
        print(f"Cleaned dataset size: {cleaned_size} rows.")

        # Define target column and columns to exclude (demographics)
        target = 'IBD'
        demographic_cols = ['age_at_referral', 'sex', 'ethnicity', 'imd_decile']
        # Create feature column list by excluding 'study_id', target, and demographic columns
        feature_cols = [col for col in df_clean.columns if col not in ['study_id', target] + demographic_cols]

        # Ensure that there are feature columns to work with
        if not feature_cols:
            print("Error: No feature columns found after excluding demographics and target.")
            return

        # Extract features (X) and the target (y) from the cleaned dataframe
        X = df_clean[feature_cols]
        y = df_clean[target]

        # Load the train/validation split files from a previous script
        X_train_file = os.path.join(log_reg_dir, 'X_train.csv')
        X_val_file = os.path.join(log_reg_dir, 'X_val.csv')
        y_train_file = os.path.join(log_reg_dir, 'y_train.csv')
        y_val_file = os.path.join(log_reg_dir, 'y_val.csv')

        X_train = pd.read_csv(X_train_file)
        X_val = pd.read_csv(X_val_file)
        # Squeeze is used to convert the DataFrame to a Series for the target variables
        y_train = pd.read_csv(y_train_file).squeeze()
        y_val = pd.read_csv(y_val_file).squeeze()

        # Identify numerical and categorical feature columns in the training data
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"Numerical columns: {numerical_cols}")
        print(f"Categorical columns: {categorical_cols}")

        # Define the preprocessing pipeline for numerical features:
        # 1. Replace missing values with constant 0.
        # 2. Scale numerical features using StandardScaler.
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])

        # If categorical columns exist, define their pipeline:
        # 1. Replace missing values with the string 'missing'.
        # 2. One-hot encode categorical variables, ignoring unknown categories at transform time.
        if categorical_cols:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ])
        else:
            # If no categorical features, only apply the numerical pipeline
            preprocessor = ColumnTransformer([
                ('num', numerical_pipeline, numerical_cols)
            ])

        # Fit the preprocessor on the training data and transform both training and validation sets
        preprocessed_train = preprocessor.fit_transform(X_train)
        preprocessed_val = preprocessor.transform(X_val)

        # Rebuild the feature name list starting with the numerical columns
        all_features = numerical_cols.copy()

        if categorical_cols:
            # Retrieve the fitted one-hot encoder from the ColumnTransformer
            onehot_encoder = preprocessor.named_transformers_['cat']['onehot']
            # Get feature names for the one-hot encoded features
            if hasattr(onehot_encoder, 'get_feature_names_out'):
                onehot_features = onehot_encoder.get_feature_names_out(categorical_cols)
            else:
                onehot_features = onehot_encoder.get_feature_names(categorical_cols)
            # Append one-hot encoded feature names to the overall feature list
            all_features += list(onehot_features)

        # Convert the numpy arrays to pandas DataFrames with the constructed feature names
        preprocessed_train_df = pd.DataFrame(preprocessed_train, columns=all_features)
        preprocessed_val_df = pd.DataFrame(preprocessed_val, columns=all_features)

        # Save the preprocessed training and validation datasets to CSV files
        preprocessed_train_df.to_csv(preprocessed_train_file, index=False)
        preprocessed_val_df.to_csv(preprocessed_val_file, index=False)

        # Save the fitted preprocessor pipeline using joblib
        preprocessor_file = os.path.join(log_reg_dir, 'preprocessor.pkl')
        joblib.dump(preprocessor, preprocessor_file)
        print(f"Preprocessor saved to {preprocessor_file}")

        print("Preprocessing completed successfully.")
        print(f"Preprocessed training data saved to {preprocessed_train_file}")
        print(f"Preprocessed validation data saved to {preprocessed_val_file}")

    except Exception as e:
        # Catch and print any exception that occurs during the preprocessing pipeline
        print(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()
