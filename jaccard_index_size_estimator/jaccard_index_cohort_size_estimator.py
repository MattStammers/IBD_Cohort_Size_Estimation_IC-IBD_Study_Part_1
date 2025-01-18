import warnings
import pandas as pd
import logging
import os
import json
from itertools import combinations
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Suppress unnecessary warnings from libraries or deprecated functions
warnings.filterwarnings("ignore")

# Configure logging with INFO level and a simple message format
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define directory paths for data and configuration files
DATA_DIR = os.path.join('..', 'data', 'processed')
FINAL_DIR = os.path.join('..', 'data', 'final')
CONFIG_DIR = os.path.join('config')

# Mapping of dataset names to their corresponding CSV filenames
DATASET_FILES = {
    'icd10_codes': 'icd10_codes.csv',
    'opcs4_codes': 'opcs4_codes.csv',
    'clinician_entered_ibd_patient_registry': 'clinician_entered_ibd_patient_registry.csv',
    'patient_ibd_portal_registration': 'patient_ibd_portal_registration.csv',
    'endoscopy_reports': 'endoscopy_reports.csv',
    'clinic_appointments': 'clinic_appointments.csv',
    'calprotectins': 'calprotectins.csv',
    'clinic_letters': 'clinic_letters.csv',
    'cytokine_modulator_prescriptions': 'cytokine_modulator_prescriptions.csv',
    'flare_line_calls': 'flare_line_calls.csv',
    'histopathology': 'histopathology.csv',
}


def load_and_clean_data(data_dir: str, dataset_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load and clean all dataframes from CSV files.

    Each CSV file is read into a pandas DataFrame, duplicated columns are removed,
    and key columns such as 'study_id' and 'IBD_Suggestive' are cleaned.

    Parameters:
    - data_dir: Directory containing the CSV files.
    - dataset_files: Dictionary mapping dataset keys to their CSV filenames.

    Returns:
    - A dictionary mapping dataset keys to their cleaned DataFrames.
    """
    dataframes = {}
    for key, filename in dataset_files.items():
        file_path = os.path.join(data_dir, filename)  # Construct full path to file
        if os.path.exists(file_path):
            try:
                # Read CSV and remove duplicated columns by reading the CSV and filtering columns
                df = pd.read_csv(file_path).loc[:, ~pd.read_csv(file_path).columns.duplicated()]
                # Clean 'study_id' column if present by converting to string and stripping whitespace
                if 'study_id' in df.columns:
                    df['study_id'] = df['study_id'].astype(str).str.strip()
                else:
                    logger.warning(f"'study_id' not found in {filename}.")
                # Clean 'IBD_Suggestive' by converting values to numeric and filling missing values with 0
                if 'IBD_Suggestive' in df.columns:
                    df['IBD_Suggestive'] = pd.to_numeric(df['IBD_Suggestive'], errors='coerce').fillna(0).astype(int)
                else:
                    logger.warning(f"'IBD_Suggestive' not found in {filename}.")
                dataframes[key] = df
                logger.info(f"Loaded and cleaned '{key}' from '{filename}'.")
            except Exception as e:
                logger.error(f"Error loading '{filename}': {e}")
                dataframes[key] = pd.DataFrame()  # Use empty DataFrame on error
        else:
            logger.warning(f"File '{filename}' for key '{key}' not found. Using empty DataFrame.")
            dataframes[key] = pd.DataFrame()  # Use empty DataFrame if file does not exist
    return dataframes


def get_ibd_patients(df: pd.DataFrame, include_all: bool = False) -> Set[str]:
    """
    Retrieve patient identifiers from a DataFrame based on IBD criteria.

    If the DataFrame contains an 'IBD_Suggestive' column, only patients marked with a value
    of 1 are returned. Optionally, if include_all is True and 'IBD_Suggestive' is not used,
    all patient IDs from the 'study_id' column are returned.

    Parameters:
    - df: The DataFrame containing patient data.
    - include_all: Flag indicating whether to return all patients if 'IBD_Suggestive' is absent.

    Returns:
    - A set of patient identifiers (strings).
    """
    if 'study_id' not in df.columns:
        logger.warning("'study_id' column not found.")
        return set()
    if 'IBD_Suggestive' in df.columns:
        return set(df[df['IBD_Suggestive'] == 1]['study_id'].dropna().unique())
    elif include_all:
        return set(df['study_id'].dropna().unique())
    else:
        return set()


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate the Jaccard similarity between two sets.

    Jaccard similarity is computed as the size of the intersection divided by the size of the union.

    Parameters:
    - set1: First set of items.
    - set2: Second set of items.

    Returns:
    - Jaccard similarity as a float value.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def recalc_jaccard_matrix(sets_dict: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Calculate the Jaccard similarity matrix for a dictionary of patient ID sets.

    Each cell (i, j) in the resulting DataFrame represents the Jaccard similarity between the
    patient sets of datasets i and j.

    Parameters:
    - sets_dict: Dictionary mapping dataset names to sets of patient identifiers.

    Returns:
    - A pandas DataFrame representing the Jaccard similarity matrix.
    """
    keys = list(sets_dict.keys())
    similarity_matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for k1 in keys:
        for k2 in keys:
            similarity_matrix.loc[k1, k2] = jaccard_similarity(sets_dict[k1], sets_dict[k2])
    return similarity_matrix


def compute_statistics(sets: Dict[str, Set[str]]):
    """
    Compute and display various statistics about patient overlaps in datasets.

    This function prints:
    - Unique patient counts per dataset.
    - Pairwise intersections (number of overlapping patient IDs) between each dataset pair.
    - The overall intersection across all datasets.
    - The count of patient IDs that appear in at least two datasets.
    - The top 10 overlapping patient IDs by frequency.

    Parameters:
    - sets: Dictionary mapping dataset names to sets of patient identifiers.

    Returns:
    - A tuple of DataFrames: (unique_counts_df, pairwise_df)
    """
    # Calculate unique counts per dataset
    unique_counts = {name: len(ids) for name, ids in sets.items()}
    unique_counts_df = pd.DataFrame(list(unique_counts.items()), columns=['Dataset', 'Unique study_ids'])
    print("\nUnique study_ids in each dataset:")
    print(unique_counts_df.to_string(index=False))

    # Calculate pairwise intersections for each combination of datasets
    pairwise = [(ds1, ds2, len(sets[ds1] & sets[ds2]))
                for ds1, ds2 in combinations(sets, 2)]
    pairwise_df = pd.DataFrame(pairwise, columns=['Dataset 1', 'Dataset 2', 'Intersection Count'])
    print("\nPairwise Intersections between datasets:")
    print(pairwise_df.to_string(index=False))

    # Calculate the overall intersection (common study_ids across all datasets)
    if sets:
        overall_intersection = set.intersection(*sets.values()) if all(sets.values()) else set()
        print(f"\nNumber of study_ids present in all {len(sets)} datasets: {len(overall_intersection)}")
    else:
        print("\nNo datasets available to compute overall intersection.")

    # Count the frequency each patient ID appears across datasets
    count = defaultdict(int)
    for ids in sets.values():
        for id_ in ids:
            count[id_] += 1
    # Count study_ids that are present in at least two datasets
    at_least_two = sum(1 for v in count.values() if v >= 2)
    print(f"Number of study_ids present in at least two datasets: {at_least_two}")

    # Identify top 10 patient IDs with highest appearances (overlaps)
    top_overlaps = sorted(count.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 study_ids with highest overlaps across datasets:")
    for id_, cnt in top_overlaps:
        print(f"study_id: {id_}, present in {cnt} datasets")

    return unique_counts_df, pairwise_df


def iterative_ibd_estimate(
    sets_dict: Dict[str, Set[str]],
    data: List[Tuple[str, int, float]],
    iterations: int = 11  # default is 11 (primary dataset + 10 others)
) -> pd.DataFrame:
    """
    Iteratively integrate datasets to estimate cumulative true positives.

    At each iteration, the function calculates the intersection, union, Jaccard similarity,
    and unique new patients when integrating the current dataset into an overall combined set.
    It then estimates the incremental true positives based on the dataset precision and updates
    cumulative true positives.

    Parameters:
    - sets_dict: Dictionary of patient ID sets keyed by dataset name.
    - data: List of tuples containing configuration data in the format (Name, Total Patients, Precision).
    - iterations: Number of datasets (after the primary one) to integrate.

    Returns:
    - A pandas DataFrame detailing integration metrics at each step.
    """
    # Convert the configuration data into a DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=["Name", "Total Patients", "Precision"])
    
    # The primary dataset is assumed to be the first item in the configuration list
    primary = df.iloc[0]
    primary_name = primary["Name"]
    primary_precision = primary["Precision"]

    if primary_name not in sets_dict:
        raise ValueError(f"Primary dataset '{primary_name}' not found in sets_dict.")

    # Initialize the combined patient set with the primary dataset's patients
    combined_set = sets_dict[primary_name].copy()
    total_estimated_true_positives = len(combined_set) * primary_precision

    logger.info(f"Primary dataset '{primary_name}' integrated with {len(combined_set)} patients.")
    logger.info(f"Initial estimated true positives: {total_estimated_true_positives:.2f}")

    # Store the results for the primary dataset integration step
    results = [{
        "Step": 0,
        "Integrated Dataset": primary_name,
        "Total Patients": len(combined_set),
        "Jaccard with Combined": None,
        "Intersection": None,
        "Union": len(combined_set),
        "Unique New Patients": len(combined_set),
        "Precision": primary_precision,
        "Incremental TPs": int(round(total_estimated_true_positives)),
        "Cumulative TPs": int(round(total_estimated_true_positives))
    }]

    # Exclude the primary dataset and sort remaining datasets by descending precision
    others = df[df["Name"] != primary_name].copy().sort_values("Precision", ascending=False)
    integrated = {primary_name}  # Keep track of already integrated datasets
    max_iterations = min(iterations, len(others))

    # Iterate through each dataset to be integrated
    for i, row in enumerate(others.itertuples(index=False), 1):
        if i > iterations:
            break  # Stop after reaching the specified number of iterations
        name, _, precision = row
        if name in integrated:
            logger.warning(f"Dataset '{name}' already integrated. Skipping.")
            continue
        next_set = sets_dict.get(name, set())
        if len(next_set) == 0:
            logger.warning(f"Dataset '{name}' is empty. Skipping integration.")
            continue

        # Calculate intersection, union, and Jaccard similarity with current combined set
        intersection = len(next_set & combined_set)
        union = len(next_set | combined_set)
        jaccard = jaccard_similarity(combined_set, next_set)
        unique_new = len(next_set - combined_set)

        if unique_new == 0:
            logger.info(f"Dataset '{name}' has no unique new patients. All its patients are already integrated.")

        # Estimate incremental true positives based on the precision and unique new patients
        incremental_tp = unique_new * precision
        total_estimated_true_positives += incremental_tp
        # Update the combined set with the new dataset's patients
        combined_set.update(next_set)
        integrated.add(name)

        logger.info(f"Step {i}: Integrated '{name}' with {len(next_set)} patients (Unique new: {unique_new}).")
        results.append({
            "Step": i,
            "Integrated Dataset": name,
            "Total Patients": len(next_set),
            "Jaccard with Combined": round(jaccard, 3),
            "Intersection": intersection,
            "Union": union,
            "Unique New Patients": unique_new,
            "Precision": precision,
            "Incremental TPs": int(round(incremental_tp)),
            "Cumulative TPs": int(round(total_estimated_true_positives))
        })

    # Append a final summary step with cumulative metrics
    results.append({
        "Step": "Final",
        "Integrated Dataset": "All Integrated",
        "Total Patients": len(combined_set),
        "Jaccard with Combined": None,
        "Intersection": None,
        "Union": None,
        "Unique New Patients": None,
        "Precision": None,
        "Incremental TPs": None,
        "Cumulative TPs": int(round(total_estimated_true_positives))
    })

    logger.info("Iterative IBD estimation completed.")
    return pd.DataFrame(results)


def load_ibd_config(config_path: str) -> List[Tuple[str, int, float]]:
    """
    Load IBD dataset configuration from a JSON file.

    The JSON file should contain a list of objects with the following keys:
        - Name (str): Identifier for the dataset.
        - Total Patients (int): (Optional) Provided count of patients (actual counts are recalculated later).
        - Precision (float): The precision value associated with the dataset.

    Example JSON structure:
    [
        {"Name": "icd10_codes", "Total Patients": 8212, "Precision": 0.96},
        {"Name": "opcs4_codes", "Total Patients": 1180, "Precision": 0.91},
        ...
    ]

    Parameters:
    - config_path: The file path to the JSON configuration file.

    Returns:
    - A list of tuples in the format (Name, Total Patients, Precision). Returns an empty list on error.
    """
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        # Convert each dictionary to a tuple; the total patient count will be updated later based on actual data
        return [(item["Name"], item.get("Total Patients", 0), item["Precision"]) for item in config_data]
    except Exception as e:
        logger.error(f"Failed to load IBD configuration from {config_path}: {e}")
        return []


def main():
    """
    Main function to orchestrate data loading, cleaning, statistics computation, and iterative estimation.

    The process includes:
    - Loading and cleaning datasets from CSV files.
    - Extracting patient sets from each dataset.
    - Computing and displaying patient overlap statistics.
    - Loading external configuration for precision values.
    - Running the iterative integration estimation.
    - Saving the estimation results and computed statistics to CSV files.
    """
    # Load and clean the datasets from the specified DATA_DIR and dataset file mappings
    dataframes = load_and_clean_data(DATA_DIR, DATASET_FILES)

    # Prepare the sets of patients for each dataset using 'get_ibd_patients'
    sets = {name: get_ibd_patients(df) for name, df in dataframes.items()}

    # Compute and display various statistics about study_id overlaps
    unique_counts_df, pairwise_df = compute_statistics(sets)

    # Load external IBD configuration (precision data) from JSON file in the config directory
    config_file = os.path.join(CONFIG_DIR, 'ibd_config.json')
    ibd_config = load_ibd_config(config_file)

    # Update configuration tuples with the actual patient counts from the sets
    ibd_corrected = [
        (name, len(sets.get(name, set())), precision)
        for name, _, precision in ibd_config
        if name in sets
    ]

    # Run iterative estimation with enhanced metrics, specifying iterations if needed (here, 10)
    estimator = iterative_ibd_estimate(sets, ibd_corrected, iterations=10)

    # Display the iterative estimation results on the console
    print("\nIterative IBD Estimation Results:")
    print(estimator.to_string(index=False))

    # Ensure the final output directory exists
    os.makedirs(FINAL_DIR, exist_ok=True)
    # Save the iterative estimation results to a CSV file in the final data directory
    estimator.to_csv(os.path.join(FINAL_DIR, 'ibd_jaccard_index_cohort_size_estimation_results.csv'), index=False)
    logger.info("Estimation results saved.")

    # Save additional computed statistics (unique counts and pairwise intersections) to CSV files
    unique_counts_df.to_csv(os.path.join(FINAL_DIR, 'unique_study_ids_per_dataset.csv'), index=False)
    pairwise_df.to_csv(os.path.join(FINAL_DIR, 'pairwise_intersections.csv'), index=False)
    logger.info("Unique counts and pairwise intersections saved.")


if __name__ == "__main__":
    # Run the main function if the script is executed as the main program
    main()
