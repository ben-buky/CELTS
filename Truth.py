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
    
    def __init__(self, wav_min=1418, wav_max=1836, truth_data=None, fit_quality=0.01, plot=True, interp_truth=True):
        
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
            The desired maximum absolute residual for the fit to a user defined truth, in pixels. The order of the polynomial will be increased until this is met. The default is 0.01 (1%).
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
            with open('Truths\\MOS_wav2pix.pkl', 'rb') as f:
                self.wav2pix = pickle.load(f)
            
            # load pix2wav mapping - this is used to generate our truth data
            with open('Truths\\MOS_pix2wav.pkl', 'rb') as f:
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
            
            wav = truth_data[0]
            pix = truth_data[1]
            
            # generate legendre fit to truth, with order and quality specified by fit_quality
            i = 1
            while True:
                
                fit = Legendre.fit(wav,pix,deg=i)
                y = fit(wav)
                resids_max = np.max(abs(y-pix))
                
                if resids_max < fit_quality:
                    break
                else:
                    i += 1
            
            print('Fit condition met with polynomial of degree ' + str(i))
            
            self.wav2pix = fit
            
            print('Wavelength to pixel truth fit = ' + str(self.wav2pix))
            
            # Find a pix2wav fit and use this to create an integer pixel grid of truth data
            if interp_truth:
                
                # use the same order as found above
                fit = Legendre.fit(pix,wav,deg=i)
                self.pix2wav = fit
                
                # test the fit by converting pixels to wavelength and using wav2pix to convert back to pixels
                w = self.pix2wav(pix)
                p = self.wav2pix(w)
                res_max = np.max(abs(p-pix))
                
                if res_max > fit_quality:
                    print('Max residual exceeds target fit quality value: ' + str(res_max) + ' > ' + str(fit_quality))
                    raise ValueError('Pixel to wavelength function failed quality check! Consider having a finer sampling in your truth input or relax your fit quality requirement.')
                else:
                    print('Pix2wav quality check passed! Max residual within target fit quality value: ' + str(res_max) + ' < ' + str(fit_quality))
                 
                # calculate the total number of pixels across the detector
                n_pix = round(abs(pix[0]) + abs(pix[-1]))
            
                # generate pixel and wavelength truth arrays using pix2wav function
                self.pix = np.arange(round(pix[0]), round(pix[0])+n_pix)
                self.wav = self.pix2wav(self.pix)
            
            # option for not interpolating truth input
            else:
                self.wav = truth_data[0]
                self.pix = truth_data[1]
                
            # we assume user has desired wavelengths already if using their own truth
            self.wav_min = self.wav[0]
            self.wav_max = self.wav[-1]
            
            # put flag here
            # find pix2wav fit of order i 
            # do quality check (convert from pix to wav, then back to pix, check against original error limit), have print statement if there's an issue
            # find n_pix by doing int(abs(self.pix[0]) + abs(self.pix[-1])), this should give total number of pixels needed irrespective of whether dispersion relation pixel array starts from 0 or a negative number
            # create new pixel array running from 0 to n_pix
            # use pix2wav to gather corresponding wavelength values
            
            
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
            
        
            
            
            
            
            
            
            