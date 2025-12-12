# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:41:14 2025

@author: bbuky
"""
# Import classes
from Truth import Truth
from Spectrum import Spectrum
from Calibration import Calibration
import numpy as np
from matplotlib import pyplot as plt

#%% Investigate the truth class

# Load the stored truth
truth = Truth()

# Load a cropped version of the stored truth
truth2 = Truth(wav_min=1450, wav_max=1700)

#%% Investigate the spectrum class

# Use a Th-Ar lamp and a Ne lamp on the stored truth, with a low resolution so you see linewidth

spectrum_test = Spectrum(truth=truth, resolution=1000, sampling=2,rel_ints=None,scaling_unit='peak_counts') # initialise class by specifying truth to use plus resolution and sampling

# build desired lamps, can be pencil or hollow cathode lamp
lamp_ne = spectrum_test.lamp_builder(lamp='Ne',max_counts=50000,user_ints=None,plot=True)
lamp_thar = spectrum_test.lamp_builder(lamp='ThAr',max_counts=50000,user_ints=None,plot=True)

# plot the combined lines from the two lamps
spectrum_test.line_plotter([lamp_ne,lamp_thar])

# Generate your sample spectra
spectrum_test.generate_spectra(lines=[lamp_ne,lamp_thar], photon_noise=True, readout_noise=10, seed=None, plot=True) # set noise in spectrum when you generate it


# Use a Th-Ar lamp on the stored truth with a higher readout noise and resolution
spectrum_thar = Spectrum(truth=truth, resolution=4000, sampling=2)
lamp_thar = spectrum_thar.lamp_builder(lamp='ThAr',max_counts=50000)
spectrum_thar.generate_spectra(lines=lamp_thar, photon_noise=True, readout_noise=15)

#%% Investigate Calibration class

# Using the stored truth and Th-Ar spectrum

calibration = Calibration(truth=truth,spectrum=spectrum_thar,orders=[3,4],amp_cutoff=100,sttdev_cutoff=20,plot=True)

#%% Investigate using a user inputted truth

# Take a cropped bit of the stored truth and use this as an example user inputted truth
sample_truth = [truth.wav[:2000],truth.pix[:2000]]

user_truth = Truth(truth_data=sample_truth,fit_quality=0.1) # mean residual of fit must be less than 10% of a pixel

# plot truth to check things are the same

plt.figure()
plt.plot(user_truth.wav, user_truth.pix)
plt.plot(sample_truth[0],sample_truth[1])
plt.show()

# generate spectrum

spectrum = Spectrum(truth=user_truth, sampling=2)

lamp_thar = spectrum.lamp_builder(lamp='ThAr')

spectrum.generate_spectra(lines=lamp_thar, photon_noise=True, readout_noise=15)

calibration = Calibration(truth=user_truth, spectrum=spectrum, orders=[3,4,7])

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

# Load the stored truth
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