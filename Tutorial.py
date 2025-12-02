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

spectrum_test = Spectrum(truth=truth, resolution=1000, sampling=2)

lamp_ne = spectrum_test.lamp_builder(lamp='Ne')
lamp_thar = spectrum_test.lamp_builder(lamp='ThAr')

# plot the combined lines from the two lamps
spectrum_test.line_plotter([lamp_ne,lamp_thar])

# Generate your sample spectra
spectrum_test.generate_spectra(lines=[lamp_ne,lamp_thar], photon_noise=True, readout_noise=10)

# Use a Th-Ar lamp on the standard MOONS truth with a high readout noise
spectrum_thar = Spectrum(truth=truth, resolution=4000, sampling=2)
lamp_thar = spectrum_thar.lamp_builder(lamp='ThAr',max_counts=50000)
spectrum_thar.generate_spectra(lines=lamp_thar, photon_noise=True, readout_noise=15)

#%% Investigate Calibration class

# Use standard MOONS truth and Th-Ar spectrum

calibration = Calibration(truth=truth,spectrum=spectrum_thar,orders=[3,4],amp_cutoff=100,sttdev_cutoff=20)

#%% Investigate and verify pixel residuals method

plt.figure()
plt.plot(spectrum_thar.pix, calibration.upd_truth_wav, label='Truth', c='orangered', marker='.')
plt.scatter(calibration.points_pix, calibration.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
plt.plot(spectrum_thar.pix, calibration.calib_fit, label='Fit, order = 4', c='tab:green',marker='.')
plt.scatter(calibration.truth_4_pix_resids, calibration.calib_fit, label = 'Pixel resid points', marker='.')
plt.legend()
plt.xlabel('Pixel')
plt.ylabel('Wavelength')


#%% Testing with Lawrence

# Load the standard MOONS truth
truth = Truth()

# Initiate spectrum class
spec_lawrence = Spectrum(truth            = truth,          # Provide truth
                         resolution       = 3000,
                         sampling         = 2,              # Nyquist sampling by default
                         global_scaling   = 50000,          # 50,000 counts is the maximum amplitude of a line
                         rel_ints         = None,           # use the saved lamp conversions by default
                         scaling_unit     = 'max_counts')   # the global scaling unit is the amplitude/maximum number of counts by default

# Choose your lamp

lamp_1 = spec_lawrence.lamp_builder(lamp = 'ThAr',
                                    plot = True) # we produce a plot of your lamp lines by default

# If using multiple lamps, plot their combined lines here:
#spec_lawrence.line_plotter([lamp_1,lamp_2])

# Generate your calibration spectrum

spec_lawrence.generate_spectra(lines          = lamp_1,      # state your lamps
                               photon_noise   = True,
                               readout_noise  = 2,
                               seed           = None,
                               plot           = True) # a plot of the spectrum is produced by default

# Conduct calibration

cal_lawrence = Calibration(truth          = truth,           # provide truth
                           spectrum       = spec_lawrence,   # provide calibration spectrum
                           orders         = [4,7,2],                 # legendre polynomial orders
                           amp_cutoff     = 50,             # minimum amplitude for line to be used for calibration, default is SNR=10
                           sttdev_cutoff  = 50)              # maximum acceptable variation in standard deviation of line fit, in %