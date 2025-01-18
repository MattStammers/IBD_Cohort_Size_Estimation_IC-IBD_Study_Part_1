import pandas as pd
import joblib
import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

def compute_classification_metrics(y_true, y_pred_proba, threshold):
    """
    Compute precision, recall, and accuracy at a given probability threshold.

    This function converts predicted probabilities into binary class predictions
    using the specified threshold and then calculates:
      - Precision: TP / (TP + FP)
      - Recall: TP / (TP + FN)
      - Accuracy: (TP + TN) / Total samples

    Parameters:
    - y_true: Array-like of true binary labels.
    - y_pred_proba: Array-like of predicted probabilities for the positive class.
    - threshold: Float; probability threshold to convert predicted probabilities into class predictions.

    Returns:
    - precision: Precision (PPV) value.
    - recall: Recall (sensitivity) value.
    - accuracy: Overall accuracy.
    """
    # Convert probabilities into binary class predictions based on the threshold
    y_pred_class = (y_pred_proba >= threshold).astype(int)
    # Compute the confusion matrix components (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()

    # Calculate precision (TP / (TP + FP))
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    # Calculate recall (TP / (TP + FN))
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    # Calculate accuracy ((TP + TN) / total samples)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, accuracy

def main():
    """
    Main function to apply the trained model to the entire dataset and generate predictions,
    perform threshold analysis, and save all relevant outputs.

    The process includes:
      1) Defining file paths for the dataset, model, preprocessor, and outputs.
      2) Loading the merged dataset and preserving the study IDs and ground truth (if available).
      3) Loading the preprocessor and final model.
      4) Ensuring the dataset contains exactly the raw columns expected by the preprocessor;
         missing columns are created with default zeros and extra columns are dropped.
      5) Applying the preprocessor to transform the entire dataset.
      6) Loading the JSON file containing the indices of the final selected features and selecting
         the corresponding columns.
      7) Loading the optimal probability threshold (or using a default of 0.5 if not found).
      8) Generating predictions (probabilities and binary predictions) on the entire dataset.
      9) Saving the predictions to a CSV file.
     10) Estimating the total number of IBD cases using the selected threshold.
     11) Performing threshold analysis (computing metrics like precision, recall, accuracy, predicted counts,
         and estimated true IBD by threshold) over a specified threshold range.
     12) Saving the threshold analysis results to both JSON and CSV files.
    """
    try:
        # ---------------------------------------------------------------------
        # 1) Define paths
        # ---------------------------------------------------------------------
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.abspath(os.path.join(script_dir, '../data'))
        log_reg_dir = os.path.join(data_dir, 'log_reg')

        merged_dataframe_file = os.path.join(data_dir, 'final_dataframe.csv')
        preprocessor_file = os.path.join(log_reg_dir, 'preprocessor.pkl')
        final_model_file = os.path.join(log_reg_dir, 'final_model.pkl')

        selected_indices_file = os.path.join(log_reg_dir, 'selected_indices.json')
        prediction_output_file = os.path.join(log_reg_dir, 'ibd_predictions.csv')
        total_ibd_estimate_file = os.path.join(log_reg_dir, 'total_ibd_estimate.txt')
        threshold_file = os.path.join(log_reg_dir, 'optimal_threshold.txt')
        threshold_analysis_json_file = os.path.join(log_reg_dir, 'threshold_analysis.json')
        threshold_analysis_csv_file = os.path.join(log_reg_dir, 'threshold_analysis.csv')

        # ---------------------------------------------------------------------
        # 2) Load the entire dataset and preserve study_id and ground truth (if available)
        # ---------------------------------------------------------------------
        df = pd.read_csv(merged_dataframe_file)
        print(f"Merged dataframe has {len(df)} rows.")

        # Preserve study_id for later output (if available)
        study_ids = df['study_id'].copy() if 'study_id' in df.columns else None

        # Preserve ground truth 'IBD' for threshold analysis if available
        y_true = None
        if 'IBD' in df.columns:
            y_true = df['IBD'].copy()
            print("Ground-truth column 'IBD' found and preserved for threshold analysis.")

        # ---------------------------------------------------------------------
        # 3) Load the preprocessor and final model
        # ---------------------------------------------------------------------
        preprocessor = joblib.load(preprocessor_file)
        model = joblib.load(final_model_file)
        print("Preprocessor and model loaded successfully.")

        # Check if the preprocessor provides expected raw column names
        if hasattr(preprocessor, 'feature_names_in_'):
            expected_raw_cols = list(preprocessor.feature_names_in_)
            print(f"Pipeline expects {len(expected_raw_cols)} raw columns:\n{expected_raw_cols}")
        else:
            raise ValueError("Your preprocessor doesn't have .feature_names_in_. Please define the raw columns manually.")

        # ---------------------------------------------------------------------
        # 4) Ensure df has exactly those expected raw columns
        # ---------------------------------------------------------------------
        # Create missing columns with default zero
        missing_raw = [c for c in expected_raw_cols if c not in df.columns]
        for col in missing_raw:
            df[col] = 0
            print(f"Created missing raw column '{col}' with default=0.")

        # Identify and drop extra columns that are not expected by the pipeline
        extra_in_df = [c for c in df.columns if c not in expected_raw_cols]
        if extra_in_df:
            print(f"Dropping extra columns not expected by the pipeline: {extra_in_df}")
            df.drop(columns=extra_in_df, inplace=True)

        # ---------------------------------------------------------------------
        # 5) Apply the preprocessor to the entire dataset
        # ---------------------------------------------------------------------
        X_full_preprocessed = preprocessor.transform(df[expected_raw_cols])
        print("Applied the preprocessor to the entire dataset.")
        print(f"Shape after pipeline transform: {X_full_preprocessed.shape}")

        # ---------------------------------------------------------------------
        # 6) Load the final selected feature indices from JSON and select final columns
        # ---------------------------------------------------------------------
        if not os.path.exists(selected_indices_file):
            raise FileNotFoundError(
                f"Could not find {selected_indices_file}, which should contain "
                "the indices used by the final model in JSON format."
            )
        with open(selected_indices_file, 'r') as f_json:
            feature_to_index = json.load(f_json)
        # Get the list of indices corresponding to the final features
        selected_indices = list(feature_to_index.values())
        print(f"Selecting columns {selected_indices} from the pipeline output.")
        X_final = X_full_preprocessed[:, selected_indices]
        print(f"Shape after selecting final columns: {X_final.shape}")

        # ---------------------------------------------------------------------
        # 7) Load the threshold from file or use default (0.5)
        # ---------------------------------------------------------------------
        if os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                threshold_str = f.read().strip()
            threshold = float(threshold_str)
            print(f"Using optimal threshold from evaluation: {threshold}")
        else:
            threshold = 0.5
            print(f"No optimal threshold file found; using default threshold {threshold}")

        # ---------------------------------------------------------------------
        # 8) Predict probabilities on the entire dataset
        # ---------------------------------------------------------------------
        y_proba = model.predict_proba(X_final)[:, 1]

        # ---------------------------------------------------------------------
        # 9) Convert probabilities to binary predictions using the selected threshold
        # ---------------------------------------------------------------------
        y_pred = (y_proba >= threshold).astype(int)
        print("Converted probabilities to predictions using threshold.")

        # ---------------------------------------------------------------------
        # 10) Build final output DataFrame and save predictions to CSV
        # ---------------------------------------------------------------------
        out_df = pd.DataFrame({
            'IBD_Predicted_Probability': y_proba,
            'IBD_Predicted': y_pred
        })
        # Round predicted probabilities for better readability
        out_df['IBD_Predicted_Probability'] = out_df['IBD_Predicted_Probability'].round(2)
        
        if study_ids is not None:
            out_df['study_id'] = study_ids
            out_df = out_df[['study_id', 'IBD_Predicted_Probability', 'IBD_Predicted']]
        out_df.to_csv(prediction_output_file, index=False)
        print(f"IBD predictions saved to {prediction_output_file}")

        # ---------------------------------------------------------------------
        # 11) Estimate total IBD cases on the entire dataset
        # ---------------------------------------------------------------------
        total_patients = len(out_df)
        total_ibd = int(y_pred.sum())
        print(f"Total patients: {total_patients}")
        print(f"Estimated IBD (at threshold {threshold}): {total_ibd}")
        with open(total_ibd_estimate_file, 'w') as f:
            f.write(f"Total patients: {total_patients}\n")
            f.write(f"Estimated IBD at threshold {threshold}: {total_ibd}\n")
        print(f"Total IBD estimate saved to {total_ibd_estimate_file}")

        # ---------------------------------------------------------------------
        # 12) Perform threshold analysis on the entire dataset
        #     Compute metrics (precision, recall, accuracy) and counts at various thresholds,
        #     and estimate the true number of IBD patients based on precision.
        # ---------------------------------------------------------------------
        if y_true is not None:
            # Use only rows with available ground truth for metric computations
            valid_mask = y_true.notna()
            y_true_valid = y_true[valid_mask].values
            y_proba_valid = y_proba[valid_mask]

            # Compute global AUROC and PRAUC for the valid rows
            auc_roc = roc_auc_score(y_true_valid, y_proba_valid)
            auc_pr = average_precision_score(y_true_valid, y_proba_valid)
            global_auroc = round(auc_roc, 2)
            global_prauc = round(auc_pr, 2)

            # Define a range of thresholds from 0.25 to 0.75
            threshold_range = np.linspace(0.25, 0.75, 9)
            threshold_metrics = {}
            for t in threshold_range:
                t_rounded = round(t, 2)
                precision, recall, accuracy = compute_classification_metrics(y_true_valid, y_proba_valid, t)
                # Round metrics for clarity
                precision_rounded = round(precision, 2) if not np.isnan(precision) else np.nan
                recall_rounded = round(recall, 2) if not np.isnan(recall) else np.nan
                accuracy_rounded = round(accuracy, 2) if not np.isnan(accuracy) else np.nan
                
                # Compute the count of predicted IBD cases for valid rows and the entire dataset
                y_pred_valid = (y_proba_valid >= t).astype(int)
                predicted_ibd_valid = int(y_pred_valid.sum())
                predicted_ibd_all = int((y_proba >= t).astype(int).sum())

                # Estimate the true number of IBD cases in the entire dataset using the precision at this threshold
                if not np.isnan(precision_rounded):
                    estimated_true_ibd_all = int(round(precision_rounded * predicted_ibd_all))
                else:
                    estimated_true_ibd_all = np.nan

                threshold_metrics[t_rounded] = {
                    'Precision': precision_rounded,
                    'Recall': recall_rounded,
                    'Accuracy': accuracy_rounded,
                    'Predicted_IBD_Valid': predicted_ibd_valid,
                    'Predicted_IBD_All': predicted_ibd_all,
                    'Estimated_True_IBD_All': estimated_true_ibd_all,
                    'Global_AUROC': global_auroc,
                    'Global_PRAUC': global_prauc
                }
            
            # Compile the threshold analysis results
            results = {
                "Threshold_Metrics": threshold_metrics
            }
            
            # Save the threshold analysis results to a JSON file
            with open(threshold_analysis_json_file, 'w') as f_json:
                json.dump(results, f_json, indent=4)

            # Convert threshold metrics to a DataFrame and save as CSV
            df_threshold = pd.DataFrame.from_dict(threshold_metrics, orient='index')
            df_threshold.index.name = 'Threshold'
            df_threshold = df_threshold.reset_index()
            df_threshold.to_csv(threshold_analysis_csv_file, index=False)

            print("Threshold analysis metrics:")
            print(df_threshold)
            print(f"Threshold analysis saved to {threshold_analysis_json_file} and {threshold_analysis_csv_file}")
        else:
            print("Ground truth (IBD) not available in the merged dataframe. Skipping threshold and AUC analysis.")
            
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    # Execute the main function when running the script directly
    main()
