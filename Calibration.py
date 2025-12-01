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
    
    def __init__(self,truth,spectrum,orders=5,line_fit='gaussian',amp_cutoff=100,sttdev_cutoff=20,plot=True):
        
        """ Calibration Class
        
        Class for conducting line fitting to the calibration points and generating a Legendre calibration model. 
        Lines are kept and used to produce the final Legendre calibration model if they meet the criteria set by amp_cutoff and stddev_cutoff. 
        This class requires CELTS.Truth and CELTS.Spectrum objects to work.
        Calibration.calib_fit contains the final pix -> wavelength fit points. Calibration.calib_fit_func is the Legendre polynomial fit itself. 
        
        Parameters
        ------------
        truth : class
            The CELTS truth object containing the truth solution for your calibration and the wavelength range of interest.
        spectrum : class
            The CELTS spectrum object containing the calibration spectrum.
        orders : int, list
            Specifies the order of the Legendre calibration fit you wish to generate. This can be a single order or a list of orders you would like to test. 
        line_fit : str
            The type of fit you would like to conduct on the lines in your spectrum. The default is 'gaussian'.
        amp_cutoff : float
            The minum amplitude (in counts) you require from a line fit for it to be considered a robust measurement and worth using for calibration. Lines which don't meet this criteria are discarded.
        sttdev_cutoff : float
            The acceptable tolerance (as a percentage) in the standard deviation of a line fit for it to be considered a robust measurement and worth using for calibration. 
            The default is 50 % meaning if the calculated standard deviation is within +/- 50% of the expected value, the line is kept. 
        plot : bool
            The user can set if they want to automatically receive plots and analysis of their line fitting and final calibration. The default is True.
        
        Returns
        ------------
        None
        
        """
        
        if truth.tag != 'truth':
            print('Incorrect truth object has been used as an input. Please use a CELTS Truth object.')
            
        if spectrum.tag != 'spectrum':
            print('Incorrect spectrum object has been used as an input. Please use a CELTS Spectrum object with spectra generated.')
            
        # ------ Convert sttdev_cutoff from percentage into multiplicative factors -------
        
        s = sttdev_cutoff/100
        s_min = 1 - s
        s_max = 1 + s
        
        # -------- Carry out line fitting to determine centres ---------------
        # only lines which meet the amplitude and stddev criteria are carried forward for the calibration fitting
        
        true_points = []
        noisy_points = []
        wavelengths = []

        for line in spectrum.lines:
            
            # can choose type of fitting to do using line_fit - ADD THIS CAPABILITY
            
            if line_fit == 'gaussian':
                
                amp = line['Intensity']*np.max(spectrum.calib_spec) # amplitude estimate based on brightest line in calibration spectrum
                mean = truth.wav2pix(line['Wavelength(Ã…)']/10)*(len(spectrum.pix)/len(truth.pix)) # using truth fit to estimate where the lines will be in pixels
                print('Centre of line = ' + str(round(mean,1)) + ' pix')
                stddev = spectrum.sampling/2.355 
                
                g_init = Gaussian1D(amplitude=amp,mean=mean,stddev=stddev)
                fit_g = fitting.TRFLSQFitter()
                g = fit_g(g_init,spectrum.pix,spectrum.calib_spec)
                
                print([g.amplitude.value,g.stddev.value])
                
                # apply amplitude and sttdev cutoffs to determine which lines to use
                if g.amplitude.value > amp_cutoff and stddev*s_min < g.stddev.value < stddev*s_max:
                    
                    # record the truth centre of each line in pixels, based on truth.wav2pix (accuracy is tied to this)
                    true_points.append(mean)
                    # record the estimated positions of each line
                    noisy_points.append(g.mean.value)
                    # record the known wavelength of that line
                    wavelengths.append(line['Wavelength(Ã…)']/10)
                
                    if plot is True:
                        plt.figure()
                        plt.plot([g.mean.value,g.mean.value],[0,np.max(g(truth.pix))+5],'--',c='tab:blue',alpha=0.7)
                        plt.plot([mean,mean],[0,np.max(g(truth.pix))+5],'--',c='r',alpha=0.7)
                        plt.plot(spectrum.pix,spectrum.ideal_spectrum,label='Noiseless data', c='r')
                        plt.plot(spectrum.pix,spectrum.calib_spec,label='Noisy data',c='tab:orange')
                        plt.plot(truth.pix,g(truth.pix),'--',label='Fit',c='tab:blue')
                        plt.legend()
                        plt.xlabel('Pixel')
                        plt.ylabel('Intensity')
                        plt.xlim((mean-3*g.stddev.value,mean+3*g.stddev.value)) # restrict plot to +/- 3 sigma
                        plt.show()
         
        self.true_points_pix = np.asarray(true_points)
        self.points_pix = np.asarray(noisy_points)
        self.points_wav = np.asarray(wavelengths)
        
        
        # --------- Perform final wavelength calibration fitting on points obtained --------------
        
        # plot the calculated points onto our truth
        
        # NEED TO DECIDE WHETHER TO SCALE UP OUR FITTING OR REDUCE THE TRUTH TO THE CORRECT PIXEL SIZES
        
        scale_truth = truth.pix*(len(spectrum.pix)-1)/(len(truth.pix)-1)
        self.upd_truth_wav = np.interp(spectrum.pix,scale_truth,truth.wav)
        
        if plot is True:
            
            plt.figure()
            #plt.plot(truth.pix,truth.wav,label='Truth', c='orangered')
            plt.plot(spectrum.pix,self.upd_truth_wav,label='Truth', c='orangered')
            plt.scatter(self.points_pix,self.points_wav,label='Spectrum points', c='deepskyblue', marker='x')
            plt.xlabel('Pixel')
            plt.ylabel('Wavelength (nm)')
            plt.legend()
            plt.title('Calibration Points obtained from Spectrum')
            plt.show()
        
        # for doing just one calibration fit:
        
        if type(orders) is int:
            
            # compute and print functional form of fit
            self.calib_fit_func = Legendre.fit(self.points_pix, self.points_wav, deg=orders)
            print('Legendre Calibration Fit: ' + str(self.calib_fit_func))
            
            # use fit to compute corresponding wavelength values
            self.calib_fit = self.calib_fit_func(spectrum.pix)
            
            # compute residuals
            self.resids_wav = self.calib_fit - self.upd_truth_wav
            
            # interpolate to compute pixel residuals
            # NEED TO ADD THESE ######################################################################################################################
            
            if plot is True:
                
                fig = plt.figure()
                gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[4,1])
                axs = gs.subplots(sharex='col')
                axs[1].set_xlabel('Pixel')
                axs[1].set_ylabel('Residual (nm)', labelpad=20)
                axs[0].set(ylabel = 'Wavelength (nm)')
                plt.suptitle('Calibration Fit')
                axs[0].plot(spectrum.pix, self.upd_truth_wav, label='Truth', c='orangered')
                axs[0].scatter(self.points_pix, self.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
                axs[0].plot(spectrum.pix, self.calib_fit, label='Fit, order = ' + str(orders), c='tab:green')
                axs[1].plot(spectrum.pix, self.resids_wav, c='tab:green')
                axs[0].legend()
                fig.tight_layout()
                plt.show()
             
        # perform fitting for different orders:
            
        else:
            
            orders = np.sort(orders)
            
            self.calib_fit = np.zeros((len(orders),len(spectrum.pix)))
            self.resids_wav = np.zeros((len(orders),len(spectrum.pix)))
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
                calib_fit_func = Legendre.fit(self.points_pix, self.points_wav, deg=orders[i])
                self.calib_fit_func.append(calib_fit_func)
                print('Legendre Calibration Fit: ' + str(calib_fit_func))
                
                # use fit to compute corresponding wavelength values
                self.calib_fit[i] = calib_fit_func(spectrum.pix)
                
                # compute residuals
                self.resids_wav[i] = self.calib_fit[i] - self.upd_truth_wav
                self.resids_wav_means[i] = np.mean(abs(self.resids_wav[i]))
                
                if plot is True:
                    
                    axs[0].plot(spectrum.pix, self.calib_fit[i], label='Fit, order = ' + str(orders[i]))
                    axs[1].plot(spectrum.pix, abs(self.resids_wav[i]))
                 
            # leave loop
            
            if plot is True:
                
                # complete plot
                axs[0].plot(spectrum.pix, self.upd_truth_wav, label='Truth', c='orangered')
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
            
