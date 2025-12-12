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
        
        """ Truth Class
        
        Class for storing the chosen 'truth' solution for the wavelength calibration, defined over a given wavelength range. 
        The pre-loaded truth is based on MOS data between 1418 and 1835 nm.
        Users can also input their own truths using truth_data. A Legendre polynomial fit will be found for this data, and the order of this polynomial is determined by fit_quality.
        When using a user defined truth, the inputs wav_min and wav_max are ignored. The code instead finds these limits from the truth data provided.
        
        Parameters
        ------------
        wav_min : float
            The minimum wavelength, in nanometres, which you want your calibration to cover. The default is for the pre-loaded data.
        wav_max : float
            The maximum wavelength, in nanometres, which you want your calibration to cover. The default is for the pre-loaded data.
        truth_data : 2D array
            The user defined true wavelength calibration solution in the form [lambda (nm), pix]. The default is None, meaning the pre-loaded solution is used.
        fit_quality : float
            The desired mean of the absolute residuals for the fit to a user defined truth, in pixels. The order of the polynomial will be increased until this is met. The default is 0.01 (1%).
        plot : bool
            The user can set if they want to automatically receive a plot of their truth. The default is True.
            
        Returns
        ------------
        None
        
        """
        
        self.tag = 'truth'
        
        if truth_data is None:
            
            self.user_truth = False
            
            # Load the stored Legendre fits. The wav2pix fit has a mean absolute residual <1% of a pixel, and maximum residual <5% of a pixel. 
            
            # load wav2pix mapping - this is used later for spectrum creation
            with open('Truths/MOS_wav2pix.pkl', 'rb') as f:
                self.wav2pix = pickle.load(f)
            
            # load pix2wav mapping - this is used to generate our truth data
            with open('Truths/MOS_pix2wav.pkl', 'rb') as f:
                pix2wav = pickle.load(f)
            
            # this is the hardcoded original number of pixels for the stored data
            pix = np.arange(0,4096)
            # generate wavelength 'truth' data
            wav = pix2wav(pix)
            
            # crop pix and wav according to the min and max wavelengths
            self.wav = wav[(wav_min < wav) & (wav < wav_max)]
            inds = np.where((wav_min < wav) & (wav < wav_max))
            self.pix = pix[inds]
            
            self.wav_min = wav_min
            self.wav_max = wav_max
            
        else:
            
            self.user_truth = True
            
            # if user specifies their own truth, save this in the same format and generate wav2pix mapping
            
            self.wav = truth_data[0]
            self.pix = truth_data[1]
            
            # we assume user has desired wavelengths already if using their own truth
            self.wav_min = self.wav[0]
            self.wav_max = self.wav[-1]
            
            # generate legendre fit to truth, with order and quality specified by fit_quality
            i = 1
            while True:
                
                fit = Legendre.fit(self.wav,self.pix,deg=i)
                y = fit(self.wav)
                resids_mean = np.mean(abs(y-self.pix))
                
                if resids_mean < fit_quality:
                    break
                else:
                    i += 1
            
            print('Fit condition met with polynomial of degree ' + str(i))
            
            self.wav2pix = fit
            
            print('Wavelength to pixel truth fit = ' + str(self.wav2pix))
            
        if plot is True:
            
            fig = plt.figure()
            gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[4,1])
            axs = gs.subplots(sharex='col')
            axs[1].set(xlabel='Wavelength (nm)', ylabel='Residual (pix)')
            axs[0].set(ylabel = 'Pixel')
            plt.suptitle('Truth Data')
            axs[0].plot(self.wav,self.pix,label='Absolute Truth')
            axs[0].plot(self.wav,self.wav2pix(self.wav),label='Truth Fit')
            axs[0].legend()
            resids =  self.wav2pix(self.wav)-self.pix
            axs[1].plot(self.wav,resids,c='tab:green')
            fig.tight_layout()
            plt.show()
            
        
            
            
            
            
            
            
            