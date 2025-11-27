# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:41:14 2025

@author: bbuky
"""

from Truth import Truth
from Spectrum import Spectrum
from Calibration import Calibration
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

#%% Investigate the spectrum class

# Use a Th-Ar lamp and a Ne lamp on the standard MOONS truth, with a low resolution so you see linewidth

spectrum_test = Spectrum(truth=truth, resolution=1000, sampling=3)

lamp_ne = spectrum_test.lamp_builder(lamp='Ne')
lamp_thar = spectrum_test.lamp_builder(lamp='ThAr')

# plot the combined lines from the two lamps
spectrum_test.line_plotter([lamp_ne,lamp_thar])

# Generate your sample spectra
spectrum_test.generate_spectra(lines=[lamp_ne,lamp_thar], photon_noise=True, readout_noise=10)

# Use a Th-Ar lamp on the standard MOONS truth with a high readout noise
spectrum_thar = Spectrum(truth=truth, resolution=4000, sampling=3)
lamp_thar = spectrum_thar.lamp_builder(lamp='ThAr')
spectrum_thar.generate_spectra(lines=lamp_thar, photon_noise=True, readout_noise=15)

#%% Investigate Calibration class

# Use standard MOONS truth and Th-Ar spectrum

calibration = Calibration(truth=truth,spectrum=spectrum_thar,orders=[3,4,5],amp_cutoff=50,sttdev_cutoff=100)
