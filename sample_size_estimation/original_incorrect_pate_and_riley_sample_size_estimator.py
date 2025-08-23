'''
============
ARCHIVED DUE TO BUGS
Date: 23/08/2025

Don't use this sample size calculator please as it contains two bugs I discovered only today. I will leave it up for reference as it calculates correctly for the paper but it is not reliable if others would like to replicate the study.

Bugs:
1. It is written such that num_predictors can become negative - this should not be possible but can happen with this formula.
2. The formula is not quite right and r2 should be divided by shrinkage. I didn't know this at the start as I now realise I misread the original paper. If you refer to the newer version you will see how this has been rectified.


#########################
Key Finding:
============
This all brings up a far more important point about whether such formulas are useful or not for database studies. I would argue that they are not very useful based on this experience because attrition is incredibly high. 

If you are going to use one of these formulas for a database study please create a substantial margin of error in your sample size.

I recommend multiplying the number you get from the formula by the number of databases you are querying (unless you know beforehand the number of patients in each one which is doubtful).

Good luck!
########################
'''

import math

def calculate_sample_size(r2, num_predictors, shrinkage, prevalence):
    """
    Calculate the required sample size for a prediction model.
    
    Parameters:
    r2 (float): Expected Cox-Snell R-squared of the new model
    num_predictors (int): Number of candidate predictor parameters
    shrinkage (float): Desired shrinkage factor
    prevalence (float): Overall outcome prevalence
    
    Returns:
    float: Required sample size
    """
    # Calculate the log term
    log_term = math.log(1 - r2)
    
    # Calculate the effective sample size before adjusting for prevalence
    effective_sample_size = num_predictors / (shrinkage * -log_term)
    
    # Adjust for prevalence
    sample_size = effective_sample_size / (prevalence * (1 - prevalence))
    
    return sample_size

# Parameters
r2 = 0.05  # Expected R-squared
num_predictors = 11  # Number of candidate predictors
shrinkage = 0.9  # Desired shrinkage
prevalence = 0.165  # Outcome prevalence

# Calculate the sample size
required_sample_size = calculate_sample_size(r2, num_predictors, shrinkage, prevalence)
print(f'The sample size should be at least {round(required_sample_size,1)} to guarantee sufficient power for the regression classifier in this study')
