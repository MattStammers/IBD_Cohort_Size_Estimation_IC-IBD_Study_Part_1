import subprocess
import sys
import os

def run_script(script_path):
    """
    Run a Python script using the current interpreter.

    This function uses subprocess.run to execute the specified Python script.
    It captures and prints the script's standard output. If the script fails,
    it prints the error message and exits the process.

    Parameters:
    - script_path: The file path of the Python script to run.

    Returns:
    - None; exits with code 1 if the script fails.
    """
    try:
        # Run the script with the current Python interpreter, capturing output and errors.
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        # Print the standard output of the executed script.
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        # If the script fails, print an error message along with the stderr output.
        print(f"Error running {script_path}:")
        print(e.stderr)
        sys.exit(1)

def main():
    """
    Main function to sequentially execute a series of Python scripts.

    The process includes:
      1) Determining the directory of this runner script.
      2) Defining an ordered list of script filenames to be executed.
      3) Running each script in sequence using the run_script() function.
      4) Printing a final success message once all scripts have executed.

    Returns:
    - None; exits with an error message if any script fails.
    """
    # Get the directory where this runner script is located.
    script_dir = os.path.dirname(__file__)
    # Define the list of script filenames to be executed in order.
    scripts = [
        '1_split_data.py',
        '2_preprocess.py',
        '3_feature_selection.py',
        '4_train_model.py',
        '5_evaluate_model.py',
        '6_predict_total_ibd.py',
        '7_bias_analysis.py',
        '8_final_estimate.py'
    ]

    # Iterate over each script name.
    for script in scripts:
        # Build the full file path for the current script.
        script_path = os.path.join(script_dir, script)
        print(f"Running {script}...")
        # Execute the script and handle possible errors within run_script().
        run_script(script_path)

    # If all scripts run successfully, print a final confirmation message.
    print("All scripts executed successfully.")

if __name__ == "__main__":
    # Execute the main function if this script is run directly.
    main()
