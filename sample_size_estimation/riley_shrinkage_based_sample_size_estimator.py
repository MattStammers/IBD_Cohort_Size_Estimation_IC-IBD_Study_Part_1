"""
Binary clinical prediction — minimum sample size by Riley et al. (criterion i, shrinkage)

Author: Matt Stammers 
Date: 23/08/2025

This is a fixed (patched) version of the prior code which contained several errors which I only discovered post publication. The other code works but is easy to crash and slightly under-estimates the total cohort size (by in my case 195 patients). It still however, obtained values above 20 EPV for me.
If you are not sure and don't have access to a medical statistician I recommend sticking to the 20 EPV rule of thumb as it is far easier to get right. The revised formula is below:

Revised Formula (binary logistic):
    n = K / ((S - 1) * ln(1 - R2_CS / S))

Where
  - K       : number of predictor *parameters* (degrees of freedom) to be estimated
  - S       : target global shrinkage factor (e.g., 0.9)
  - R2_CS   : anticipated Cox-Snell R^2 for the new model (optimism-adjusted if available)

Notes
  - Requirements: 0 < R2_CS < S < 1  → ln(1 - R2_CS/S) < 0 and (S-1) < 0, so denominator > 0.
  - Prevalence π is not part of the formula; it is used AFTER n is computed to report expected events E = n·π and EPP = E/K.
  - Keep K as the number of *parameters* (e.g., categorical levels, spline terms, interactions each add parameters).

This file intentionally contains only the binary shrinkage criterion — no other criteria or add-ons and is consisent with section 2.1 of this paper: (https://pmc.ncbi.nlm.nih.gov/articles/PMC10012398/pdf/10.1177_09622802231151220.pdf
"""

from __future__ import annotations
import math
from typing import Dict

def riley_shrinkage_n(r2_cs: float, K: int, S: float) -> float:
    """Compute sample size n using Riley's binary shrinkage criterion.

    Args:
        r2_cs (float): Anticipated Cox-Snell R^2 (0 < r2_cs < 1). Use optimism-adjusted if available.
        K (int):      Number of predictor parameters (>0).
        S (float):    Target shrinkage factor (0 < S < 1), e.g., 0.9.

    Returns:
        float: Required sample size n.
    """
    if not (0 < r2_cs < 1):
        raise ValueError("r2_cs must be in (0, 1).")
    if K <= 0:
        raise ValueError("K must be positive.")
    if not (0 < S < 1):
        raise ValueError("S (shrinkage) must be in (0, 1).")
    if not (r2_cs < S):
        # Ensures ln(1 - r2_cs/S) is defined and that n is positive and finite
        raise ValueError("Require r2_cs < S. If not, reduce r2_cs or raise S (e.g., S=0.95).")

    denom = (S - 1.0) * math.log(1.0 - (r2_cs / S))  # > 0 under the constraints above
    return K / denom

def binary_sample_size(r2_cs: float, K: int, S: float, prevalence: float) -> Dict[str, float]:
    """Convenience wrapper: compute n, then expected events and EPP.

    Args:
        r2_cs (float): Anticipated Cox-Snell R^2 (optimism-adjusted if possible).
        K (int):      Number of predictor parameters.
        S (float):    Target shrinkage factor.
        prevalence (float): Outcome prevalence π in the target population (0 < π < 1).

    Returns:
        dict: { 'n': n, 'events': E, 'epp': EPP }
    """
    if not (0 < prevalence < 1):
        raise ValueError("prevalence must be in (0, 1).")

    n = riley_shrinkage_n(r2_cs, K, S)
    events = n * prevalence
    epp = events / K
    return {"n": n, "events": events, "epp": epp}


if __name__ == "__main__":
    r2_cs = 0.05   # anticipated Cox–Snell R^2
    K = 11         # number of predictor parameters
    S = 0.9        # target shrinkage
    prevalence = 0.165

    res = binary_sample_size(r2_cs, K, S, prevalence)
    print(f"Sample size (Riley shrinkage): n = {res['n']:.1f}")
    print(f"Expected events: {res['events']:.1f}")
    print(f"Events per predictor (EPP): {res['epp']:.2f}")
