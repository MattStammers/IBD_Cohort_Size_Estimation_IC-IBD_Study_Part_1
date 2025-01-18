import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    """
    Main function to load data, clean it by dropping missing target values, split it into training
    and validation sets for logistic regression, and save the resulting splits as CSV files.

    The process includes:
    - Defining paths to the input data and output directories.
    - Loading the final DataFrame from CSV.
    - Dropping rows with missing values in the 'IBD' target column.
    - Defining features by excluding the target and demographic columns.
    - Splitting the data into training and validation sets with stratification based on the target.
    - Saving the train and validation sets for both features and the target variable to CSV files.
    """
    # Define directory paths relative to this script's location
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(script_dir, '../data'))
    log_reg_dir = os.path.join(data_dir, 'log_reg')

    # Define full file paths for input and output files
    input_file = os.path.join(data_dir, 'final_dataframe.csv')
    X_train_file = os.path.join(log_reg_dir, 'X_train.csv')
    X_val_file = os.path.join(log_reg_dir, 'X_val.csv')
    y_train_file = os.path.join(log_reg_dir, 'y_train.csv')
    y_val_file = os.path.join(log_reg_dir, 'y_val.csv')

    # Load the data from the final dataframe CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Display the initial number of rows in the dataset
    initial_size = len(df)
    print(f"Initial dataset size: {initial_size} rows.")

    # Exclude rows where the target column 'IBD' has missing values
    df_clean = df.dropna(subset=['IBD'])
    cleaned_size = len(df_clean)
    dropped = initial_size - cleaned_size
    print(f"Dropped {dropped} rows due to missing 'IBD' values.")
    print(f"Cleaned dataset size: {cleaned_size} rows.")

    # Define the target column and demographic columns to exclude from features
    target = 'IBD'  # Gold standard column for classification
    demographic_cols = ['age_at_referral', 'sex', 'ethnicity', 'imd_decile']

    # Define feature columns by excluding 'study_id', target and demographic columns
    feature_cols = [
        col for col in df_clean.columns
        if col not in ['study_id', target] + demographic_cols
    ]

    # Check if any features remain after excluding undesired columns
    if not feature_cols:
        print("Error: No feature columns found after excluding demographics and target.")
        return

    # Separate the features (X) and the target variable (y)
    X = df_clean[feature_cols]
    y = df_clean[target]

    # Split the data into training and validation sets (70% train, 30% validation)
    # Stratify is used to maintain the same distribution of the target variable 'IBD' in both sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Save the training and validation sets to CSV files without the index column
    X_train.to_csv(X_train_file, index=False)
    X_val.to_csv(X_val_file, index=False)
    y_train.to_csv(y_train_file, index=False)
    y_val.to_csv(y_val_file, index=False)

    # Print messages indicating successful data splitting and display sizes of each split
    print("Data splitting completed successfully.")
    print(f"Training set size: {len(X_train)} rows.")
    print(f"Validation set size: {len(X_val)} rows.")

if __name__ == "__main__":
    # Execute the main function if this script is run as the main program
    main()
