import pandas as pd
import os
import joblib
import json
import numpy as np

def main():
    """
    Main function to estimate the final IBD cohort size based on model predictions
    and additional data from clinic letters.

    The process includes:
      1) Defining file paths for IBD predictions, clinic letters, and the output file.
      2) Loading the datasets (predictions and clinic letters) into DataFrames.
      3) Identifying patients with a positive IBD prediction from the predictions dataset 
         counting only true positive values from here.
      4) Finding patients flagged as IBD-suggestive in the clinic letters dataset 
         that are NOT present in the predictions.
      5) Adjusting the count of suggestive patients by a defined precision factor.
      6) Calculating the final cohort size by summing the number of positive predictions 
         and the precision-adjusted suggestive patients.
      7) Writing the final results to an output text file.
    """
    try:
        # ---------------------------------------------------------------------
        # 1) Define paths
        # ---------------------------------------------------------------------
        # Get the directory of this script.
        script_dir = os.path.dirname(__file__)
        # Define directory paths for log_reg outputs and processed data.
        log_reg_dir = os.path.abspath(os.path.join(script_dir, '../data/log_reg'))
        processed_dir = os.path.abspath(os.path.join(script_dir, '../data/processed'))

        # Define the file paths for input predictions, clinic letters, and the output estimate.
        ibd_predictions_file = os.path.join(log_reg_dir, 'ibd_predictions.csv')
        clinic_letters_file = os.path.join(processed_dir, 'clinic_letters.csv')
        output_file = os.path.join(log_reg_dir, 'final_ibd_size_estimate.txt')

        # ---------------------------------------------------------------------
        # 2) Load the datasets
        # ---------------------------------------------------------------------
        # Load the IBD predictions and clinic letters CSVs into DataFrames.
        ibd_predictions = pd.read_csv(ibd_predictions_file)
        clinic_letters = pd.read_csv(clinic_letters_file)

        # Ensure the required columns are present in both datasets.
        if 'study_id' not in ibd_predictions.columns or 'IBD_Predicted' not in ibd_predictions.columns:
            raise ValueError("'ibd_predictions.csv' must contain 'study_id' and 'IBD_Predicted' columns.")

        if 'study_id' not in clinic_letters.columns or 'IBD_Suggestive' not in clinic_letters.columns:
            raise ValueError("'clinic_letters.csv' must contain 'study_id' and 'IBD_Suggestive' columns.")

        # ---------------------------------------------------------------------
        # 3) Identify unique IBD-positive predictions
        # ---------------------------------------------------------------------
        # Extract study IDs where IBD_Predicted equals 1.
        positive_predictions_list = ibd_predictions.loc[ibd_predictions['IBD_Predicted'] == 1, 'study_id']
        # Adjust positive predictions by the precision of this algorithm
        positive_predictions = int(round(len(positive_predictions_list) * 0.86))
        print(f"Number of True IBD-positive predictions: {positive_predictions}")

        # ---------------------------------------------------------------------
        # 4) Identify patients with IBD_Suggestive == 1 not in ibd_predictions
        # ---------------------------------------------------------------------
        # Extract study IDs from clinic letters where IBD_Suggestive is 1.
        suggestive_patients = clinic_letters.loc[clinic_letters['IBD_Suggestive'] == 1, 'study_id']
        print(f"Number of IBD-suggestive patients in clinic_letters: {len(suggestive_patients)}")

        # Identify patients that are flagged as suggestive but were not already predicted as positive.
        suggestive_only = suggestive_patients[~suggestive_patients.isin(positive_predictions_list)]
        print(f"Number of IBD-suggestive patients not in ibd_predictions: {len(suggestive_only)}")

        # Apply a precision adjustment factor (83% precision) to the suggestive patients count.
        suggestive_precision_adjusted = int(round(len(suggestive_only) * 0.83))
        print(f"Number of True IBD-suggestive patients not in ibd_predictions after precision adjustment: {suggestive_precision_adjusted}")

        # ---------------------------------------------------------------------
        # 5) Calculate the final cohort size
        # ---------------------------------------------------------------------
        # Sum the number of positive predictions and the precision-adjusted suggestive count.
        final_cohort_size = positive_predictions + suggestive_precision_adjusted
        print(f"Final IBD cohort size estimate: {final_cohort_size}")

        # ---------------------------------------------------------------------
        # 6) Write the final size estimate to file
        # ---------------------------------------------------------------------
        # Save the results to a text file.
        with open(output_file, 'w') as f:
            f.write(f"Number of IBD-positive predictions (IBD_Predicted == 1): {positive_predictions}\n")
            f.write(f"Number of IBD-suggestive patients not in predictions: {len(suggestive_only)}\n")
            f.write(f"Final IBD cohort size estimate: {final_cohort_size}\n")

        print(f"Final cohort size estimate saved to {output_file}")

        # ---------------------------------------------------------------------
        # 7) Save model coefficients + odds ratios (handling a Pipeline)
        # ---------------------------------------------------------------------
        # Load your trained pipeline or model
        model_path = os.path.abspath(os.path.join(script_dir, '../data/log_reg/final_model.pkl'))
        model = joblib.load(model_path)

        # If it's a Pipeline, grab the final estimator; otherwise use the model directly
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            # assume the last step is your LogisticRegression
            lr = model.steps[-1][1]
        else:
            lr = model

        # Verify we have coefficients
        if not hasattr(lr, 'coef_'):
            raise AttributeError(f"Model object {lr} has no 'coef_' attribute")

        # Load the feature–index map
        selected_indices_path = os.path.join(log_reg_dir, 'selected_indices.json')
        with open(selected_indices_path, 'r') as f:
            selected_indices = json.load(f)

        # Reconstruct feature list in coefficient order
        features = [None] * len(selected_indices)
        for feat, idx in selected_indices.items():
            features[idx] = feat

        # Extract coefficients and intercept
        coef_array = lr.coef_[0]        # shape = (n_features,)
        intercept = lr.intercept_[0]

        # Build DataFrame
        coeff_df = pd.DataFrame({
            'feature': ['(intercept)'] + features,
            'coefficient': [intercept] + coef_array.tolist()
        })

        # Compute odds ratios
        coeff_df['odds_ratio'] = np.exp(coeff_df['coefficient'])

        # Save to CSV in log_reg folder
        coeffs_output = os.path.join(log_reg_dir, 'model_coefficients.csv')
        coeff_df.to_csv(coeffs_output, index=False)
        print(f"Model coefficients and odds ratios saved to {coeffs_output}")


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()