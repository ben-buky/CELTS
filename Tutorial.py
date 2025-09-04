# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:41:14 2025

@author: bbuky
"""

from Truth import Truth
import numpy as np
from matplotlib import pyplot as plt

#%% Investigate the truth class

# Load the standard MOONS truth
truth = Truth()

# Load a cropped version of the MOONS truth
truth2 = Truth(wav_min=1450, wav_max=1700)

# Take a cropped bit of the MOONS truth and use this as an example user inputted truth
sample_truth = [truth.wav[:2000],truth.pix[:2000]]

user_truth = Truth(truth_data=sample_truth,fit_quality=0.1) # mean residual of fit must be less than 10% of a pixel


