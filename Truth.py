# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 09:56:45 2025

@author: bbuky
"""

import numpy as np
from numpy.polynomial.legendre import Legendre
from matplotlib import pyplot as plt
import pickle

class Truth:
    
    def __init__(self, wav_min=1418, wav_max=1836, truth_data=None, fit_quality=0.01, plot=True):
        
        """ Truth Object
        
        Class for storing the chosen 'truth' solution for the wavelength calibration, defined over a given wavelength range. 
        The pre-loaded truth is based on the VLT MOONS solution between 1418 and 1835 nm, but should hold for all MOS instruments.
        Users can also input their own truths using truth_data. A Legendre polynomial fit will be found for this data, and the order of this polynomial is determined by fit_quality.
        
        Parameters
        ------------
        wav_min : float
            The minimum wavelength, in nanometres, which you want your calibration to cover. The default is for the MOONS data.
        wav_max : float
            The maximum wavelength, in nanometres, which you want your calibration to cover. The default is for the MOONS data.
        truth_data : 2D array
            The true wavelength calibration solution in the form [lambda (nm), pix]. If None, the pre-loaded solution is used.
        fit_quality : float
            The desired mean of the absolute residuals for the fit to a user defined truth, in pixels. The order of the polynomial will be increased until this is met. The default is 0.01 (1%).
        plot : bool
            The user can set if they want to automatically receive a plot of their truth. The default is True.
            
        Returns
        ------------
        None
        
        """
        
        if truth_data is None:
            
            # Load the MOONS Legendre fits. The wav2pix fit has a mean absolute residual <1% of a pixel, and maximum residual <5% of a pixel. 
            
            with open('Truths/MOONS_wav2pix.pkl', 'rb') as f:
                self.wav2pix = pickle.load(f)
                
            with open('Truths/MOONS_pix2wav.pkl', 'rb') as f:
                pix2wav = pickle.load(f)
            
            # this is the hardcoded original number of pixels for the MOONS data
            self.pix = np.arange(0,4096)
            wav = pix2wav(self.pix)
            
            wav = wav[wav_min < wav]
            self.wav = wav[wav < wav_max]
        
            
            
            
            
            
            
            