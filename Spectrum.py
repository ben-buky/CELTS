# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:03:19 2025

@author: bbuky
"""

from astropy import io
from astropy import table
import numpy as np
from astropy.modeling.functional_models import Gaussian1D
from matplotlib import pyplot as plt

class Spectrum:
    
    def __init__(self,lines,truth,resolution,sampling,global_scaling=2,photon_noise=True,readout_noise=1,plot=True,int_cutoff=1):
        
        """ Spectrum Class
        
        Class for creating the spectrum produced on a detector during wavelength calibration. 
        This class requires a CELTS.Truth object to work.
        The line lists used are taken from NIST.
        
        Parameters
        ------------
        lines : str, list
            The elements you will be using for your calibration. Options are...
            
            FILL THIS IN
  
            
        Returns
        ------------
        None
        
        """
        
        if truth.tag != 'truth':
            print('Incorrect truth object has been used as an input. Please use a CELTS Truth object.')
            
        self.tag = 'spectrum'
        
        # Create full line list object
        loc = 'Line_lists/'
        kr = io.ascii.read(loc + 'Kr.ascii')
        ar = io.ascii.read(loc + 'Ar.ascii')
        ne = io.ascii.read(loc + 'Ne.ascii')
        xe = io.ascii.read(loc + 'Xe.ascii')
        hg = io.ascii.read(loc + 'Hg.ascii')
        u  = io.ascii.read(loc + 'U.ascii')
        th = io.ascii.read(loc + 'Th.ascii')
        
        all_line_data = [
            {"name": "kr", "data": kr},
            {"name": "ar", "data": ar},
            {"name": "ne", "data": ne},
            {"name": "xe", "data": xe},
            {"name": "hg", "data": hg},
            {"name": "u", "data": u},
            {"name": "th", "data": th}]
        
        # Select all the lines for the desired elements
        selected = [all_line_data["data"] for all_line_data in all_line_data if all_line_data["name"] in lines]
                
        selected_lines=table.vstack(selected)
        
        # Need to work out how to scale lines to account for differences in relative intensity normalisation?
        
        # Filter the lines by intensity
        relevant_lines=selected_lines[(selected_lines['Wavelength']/10>truth.wav_min) & (selected_lines['Wavelength']/10<truth.wav_max)] # /10 to account for angstroms
        self.lines=relevant_lines[relevant_lines['Intensity']>int_cutoff]
        
        # Set up variables for creating idealised spectrum
        lambda_range=[truth.wav_min,truth.wav_max]
        pix_delta_wl=np.median(lambda_range)/resolution/sampling
        pix_wl=np.arange(lambda_range[0],lambda_range[1],pix_delta_wl)
        pix_x=np.arange(len(pix_wl))
        y_lines=np.zeros_like(pix_x)
        
        # Create idealised spectrum
        
        for line in self.lines:
            
            gaussian=Gaussian1D(line['Intensity'],np.interp(line['Wavelength']/10,pix_wl,pix_x),sampling/2.355) # 2.355 converts between gaussian fwhm and std dev
            y_lines=y_lines+gaussian(pix_x)
            
        self.ideal_spectrum = y_lines
            
        # Create plots of lines and idealised spectrum
        
        if plot is True:
            
            # Create plot of lines, coloured by element
        
            plt.figure()
            plt.xlim(lambda_range[0],lambda_range[1])
            plt.xlabel('WL (nm)')
            plt.ylabel('Relative Intensity')
            plt.title('Chosen Lines')
            for line in self.lines:
               if line['(Ã…)'] == 'Kr':
                  color='blue'
                  lab='Kr'
               elif line['(Ã…)'] == 'Ar':
                  color='green'
                  lab='Ar'
               elif line['(Ã…)'] == 'Xe':
                  color='yellow' 
                  lab='Xe'
               elif line['(Ã…)'] == 'Ne':
                  color='orange' 
                  lab='Ne'
               elif line['(Ã…)'] == 'Hg':
                  color='red' 
                  lab='Hg'
               elif line['(Ã…)'] == 'U':
                  color='purple' 
                  lab='U'
               elif line['(Ã…)'] == 'Th':
                  color='pink' 
                  lab='Th'
                  
               plt.plot([line['Wavelength']/10,line['Wavelength']/10],[0,line['Intensity']],color=color,label=lab)
               
            handles, labels = plt.gca().get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            plt.legend(unique.values(), unique.keys())
            
            #Plot idealised spectrum with representative linewidth
            plt.figure(figsize=(10,5))
            plt.plot(pix_wl,self.ideal_spectrum)
            plt.xlabel('WL (nm)')
            plt.ylabel('Relative Intensity')
            plt.title('Chosen Lines Idealised Spectrum')
            
        # Convert spectrum into realistic form
        
        # Rescale y axis
        y_spec = global_scaling*self.ideal_spectrum

        # Spectrum from instrument will have pixels as x axis, so use wav2pix fit to convert
        line_pix = truth.wav2pix(pix_wl)

        # Add noise
        
        # Photon noise
        if photon_noise is True:
            phot = np.random.poisson(lam=y_spec) # SHOULD THIS BE ADDED TO SELF.SPECTRUM??
        else:
            phot = np.zeros(len(y_spec))
            
        # Readout noise
        readout = np.random.normal(0,readout_noise,y_spec.shape)
        
        # New spectrum
        noisy_data_full = phot+readout

        # Interpolate so we have correct number of data points for pixels
        self.calib_spec = np.interp(truth.pix,line_pix,noisy_data_full)
        
        if plot is True:
            
            plt.figure(figsize=(10,5))
            plt.plot(line_pix,y_spec,'--',label='Noiseless data', c='r')
            plt.plot(truth.pix,self.calib_spec,label='Noisy data')
            plt.legend()
            plt.xlabel('Pixel')
            plt.ylabel('Intensity')
            plt.title('Instrument Calibration Spectrum')
            plt.show()
        
        
        