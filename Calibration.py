# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:13:53 2025

@author: bbuky
"""

from astropy.modeling.functional_models import Gaussian1D
from astropy.modeling import fitting
from numpy.polynomial.legendre import Legendre
from matplotlib import pyplot as plt
import numpy as np

class Calibration:
    
    def __init__(self,truth,spectrum,orders=5,line_fit='gaussian',plot=True):
        
        """ Calibration Class
        
        Class for conducting line fitting to the calibration points and generating a Legendre calibration model. 
        This class requires CELTS.Truth and CELTS.Spectrum objects to work.
        Calibration.calib_fit contains the final pix -> wavelength fit points. Calibration.calib_fit_func is the Legendre polynomial fit itself. 
        
        Parameters
        ------------
        truth : class
            
            FILL THIS IN
  
            
        Returns
        ------------
        None
        
        """
        
        if truth.tag != 'truth':
            print('Incorrect truth object has been used as an input. Please use a CELTS Truth object.')
            
        if spectrum.tag != 'spectrum':
            print('Incorrect spectrum object has been used as an input. Please use a CELTS Spectrum object.')
        
        
        # -------- Calculate centre of each expected line in the noisy data --------------

        noisy_points = []

        for line in spectrum.lines:
            
            # can choose type of fitting to do using line_fit - ADD THIS CAPABILITY
            
            if line_fit == 'gaussian':
                
                amp = line['Intensity']*spectrum.global_scaling
                mean = truth.wav2pix(line['Wavelength']/10) # using truth fit to estimate where the lines will be in pixels
                print('Centre of line = ' + str(round(mean,1)) + ' pix')
                stddev = 1 # setting this as one pixel in all cases for now
                
                g_init = Gaussian1D(amplitude=amp,mean=mean,stddev=stddev)
                fit_g = fitting.TRFLSQFitter()
                g = fit_g(g_init,truth.pix,spectrum.calib_spec)
                
                # record the estimated positions of each line
                noisy_points.append(g.mean.value)
            
            if plot is True:
                plt.figure()
                plt.plot([g.mean.value,g.mean.value],[0,np.max(g(truth.pix))+5],'--',c='tab:blue',alpha=0.7)
                plt.plot([mean,mean],[0,np.max(g(truth.pix))+5],'--',c='r',alpha=0.7)
                plt.plot(spectrum.line_pix,spectrum.ideal_spectrum,label='Noiseless data', c='r')
                plt.plot(truth.pix,spectrum.calib_spec,label='Noisy data',c='tab:orange')
                plt.plot(truth.pix,g(truth.pix),'--',label='Fit',c='tab:blue')
                plt.legend()
                plt.xlabel('Pixel')
                plt.ylabel('Intensity')
                plt.xlim((mean-3*g.stddev.value,mean+3*g.stddev.value)) # restrict plot to +/- 3 sigma
                plt.show()
            
        self.points_pix = np.asarray(noisy_points)
        
        
        # --------- Perform fitting on points obtained ---------------------------
        
        # record the known wavelengths for the lines fitted above
        self.points_wav = spectrum.lines['Wavelength']/10
        
        # plot the calculated points onto our truth
        
        if plot is True:
            
            plt.figure()
            plt.plot(truth.pix,truth.wav,label='Truth', c='orangered')
            plt.scatter(self.points_pix,self.points_wav,label='Spectrum points', c='deepskyblue', marker='x')
            plt.xlabel('Pixel')
            plt.ylabel('Wavelength (nm)')
            plt.legend()
            plt.title('Calibration Points obtained from Spectrum')
            plt.show()
        
        # for doing just one fit:
        
        if type(orders) is int:
            
            # compute and print functional form of fit
            self.calib_fit_func = Legendre.fit(self.points_pix, self.points_wav, deg=orders)
            print('Legendre Calibration Fit: ' + str(self.calib_fit_func))
            
            # use fit to compute corresponding wavelength values
            self.calib_fit = self.calib_fit_func(truth.pix)
            
            # compute residuals
            self.resids_wav = self.calib_fit - truth.wav
            
            # interpolate to compute pixel residuals
            
            if plot is True:
                
                fig = plt.figure()
                gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[4,1])
                axs = gs.subplots(sharex='col')
                axs[1].set_xlabel('Pixel')
                axs[1].set_ylabel('Residual (nm)', labelpad=20)
                axs[0].set(ylabel = 'Wavelength (nm)')
                plt.suptitle('Calibration Fit')
                axs[0].plot(truth.pix, truth.wav, label='Truth', c='orangered')
                axs[0].scatter(self.points_pix, self.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
                axs[0].plot(truth.pix, self.calib_fit, label='Fit, order = ' + str(orders), c='tab:green')
                axs[1].plot(truth.pix, self.resids_wav, c='tab:green')
                axs[0].legend()
                fig.tight_layout()
                plt.show()
             
        # perform fitting for different orders:
            
        else:
            
            self.calib_fit = np.zeros((len(orders),len(truth.pix)))
            self.resids_wav = np.zeros((len(orders),len(truth.pix)))
            self.calib_fit_func = []
            self.resids_wav_means = np.zeros(len(orders))
            
            # do universal bits of plot
            if plot is True:
                
                fig = plt.figure()
                gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[4,1])
                axs = gs.subplots(sharex='col')
                axs[1].set_xlabel('Pixel')
                axs[1].set_ylabel('Residual (nm)', labelpad=20)
                axs[0].set(ylabel = 'Wavelength (nm)')
                plt.suptitle('Calibration Fit')
            
            for i in range(len(orders)):
                
                # compute and print functional form of fit
                legendre_fit = Legendre.fit(self.points_pix, self.points_wav, deg=orders[i])
                self.calib_fit_func.append(legendre_fit)
                print('Legendre Calibration Fit: ' + str(legendre_fit))
                
                # use fit to compute corresponding wavelength values
                self.calib_fit[i] = legendre_fit(truth.pix)
                
                # compute residuals
                self.resids_wav[i] = self.calib_fit[i] - truth.wav
                self.resids_wav_means[i] = np.mean(abs(self.resids_wav[i]))
                
                if plot is True:
                    
                    axs[0].plot(truth.pix, self.calib_fit[i], label='Fit, order = ' + str(orders[i]))
                    axs[1].plot(truth.pix, abs(self.resids_wav[i]))
                 
            # leave loop
            
            if plot is True:
                
                # complete plot
                axs[0].plot(truth.pix, truth.wav, label='Truth', c='orangered')
                axs[0].scatter(self.points_pix, self.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
                axs[0].legend()
                #axs[1].set_yscale('log')
                plt.show()      
                
                # plot absolute mean of residuals
                
                plt.figure()
                plt.plot(orders,self.resids_wav_means)
                plt.xlabel('Order of Fit')
                plt.ylabel('Mean of absolute residuals (nm)')
                plt.title('Mean of Residuals for different fits')
                plt.show()
            
