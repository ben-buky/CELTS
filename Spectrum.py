# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:03:19 2025

@author: bbuky
"""

from astropy import io
from astropy import table
from astropy.table import vstack
import re
import numpy as np
from astropy.modeling.functional_models import Gaussian1D
from matplotlib import pyplot as plt

class Spectrum:
    
    def __init__(self,truth,resolution,sampling,global_scaling=50000,rel_ints=None,scaling_unit='max_counts'):
        
        """ Spectrum Class
        
        Class for creating the spectrum produced on a detector during wavelength calibration. 
        This class requires a CELTS.Truth object to work.
        The line lists used are taken from NIST.
        Initialization is used to set the parameters which will be consistent across any lamp later specified for use.
        
        Parameters
        ------------
        lamp : str, list
            The elements you will be using for your calibration. Options are...
            
            FILL THIS IN
  
            
        Returns
        ------------
        None
        
        """
        ############### Initial Configuring: ###############
        
        if truth.tag != 'truth':
            print('Incorrect truth object has been used as an input. Please use a CELTS Truth object.')
        else:
            self.truth = truth
        
        if scaling_unit == 'max_counts':
            self.global_scaling = global_scaling
            
        if scaling_unit == 'photons':       
            # need to put eqn in here to convert from max counts to flux under gaussian
            self.global_scaling = global_scaling
            
        self.resolution = resolution
        self.sampling   = sampling
        
        ########## Set the relative intensities dictionary to use ###########
        
        # case where we are using the stored conversions between different lamps
        if rel_ints == None:
            
            self.rel_ints = dict()
            
            ###################     Pencil Lamps    #########################

            self.rel_ints['Ar']   = [1]
            self.rel_ints['Ne']   = [1]
            self.rel_ints['Kr']   = [1]
            self.rel_ints['Xe']   = [1]
            
            ###############   Hollow Cathode Lamps   ########################
            
            # the values refer to the relative intensities of the elements in the order they are written
            
            self.rel_ints['ThAr'] = [0.1,1]
            self.rel_ints['ThNe'] = [0.01,0.2]
            self.rel_ints['UAr']  = [0.01,0.2]
            self.rel_ints['UNe']  = [0.01,0.2]
        
        # case where user sets the relative intensities themselves
        else:
            self.rel_ints = rel_ints
            
        
    def lamp_builder(self,lamp,plot=True):
        
        # convert lamp string into list of elements in that lamp
        elements = re.findall(r'[A-Z][^A-Z]*', lamp)
        #print(elements)  
        
        ############## Create full line list object #############
        
        loc = 'Line_lists/'
        Kr = io.ascii.read(loc + 'Kr.ascii')
        Ar = io.ascii.read(loc + 'Ar.ascii')
        Ne = io.ascii.read(loc + 'Ne.ascii')
        Xe = io.ascii.read(loc + 'Xe.ascii')
        Hg = io.ascii.read(loc + 'Hg.ascii')
        U  = io.ascii.read(loc + 'U.ascii')
        Th = io.ascii.read(loc + 'Th.ascii')
        
        all_line_data = [
            {"name": "Kr", "data": Kr},
            {"name": "Ar", "data": Ar},
            {"name": "Ne", "data": Ne},
            {"name": "Xe", "data": Xe},
            {"name": "Hg", "data": Hg},
            {"name": "U", "data": U},
            {"name": "Th", "data": Th}]
        
        # Select all the lines for the desired elements
        selected = [all_line_data["data"] for all_line_data in all_line_data if all_line_data["name"] in elements]
                
        selected_lines=table.vstack(selected)
        
        # set the scaling factors you're going to use based on the lamp provided
        rel_scaling = self.rel_ints[lamp]
        
        # apply the relative scaling factors
        for i in range(len(selected_lines)):
            for j in range(len(rel_scaling)):
                if selected_lines['Element'][i] == elements[j]:
                    selected_lines['Intensity'][i] = selected_lines['Intensity'][i]*rel_scaling[j]*self.global_scaling/1000
        
        # Filter the lines by wavelength
        lines=selected_lines[(selected_lines['Wavelength(Ã…)']/10>self.truth.wav_min) & (selected_lines['Wavelength(Ã…)']/10<self.truth.wav_max)] # /10 to account for angstroms
        
        # Plot the chosen lines
        
        if plot is True:
            
            # Create plot of lines, coloured by element
        
            plt.figure()
            plt.xlim(self.truth.wav_min,self.truth.wav_max)
            plt.xlabel('WL (nm)')
            plt.ylabel('Relative Intensity')
            plt.title('Chosen Lines')
            for line in lines:
               if line['Element'] == 'Kr':
                  color='blue'
                  lab='Kr'
               elif line['Element'] == 'Ar':
                  color='green'
                  lab='Ar'
               elif line['Element'] == 'Xe':
                  color='yellow' 
                  lab='Xe'
               elif line['Element'] == 'Ne':
                  color='orange' 
                  lab='Ne'
               elif line['Element'] == 'Hg':
                  color='red' 
                  lab='Hg'
               elif line['Element'] == 'U':
                  color='purple' 
                  lab='U'
               elif line['Element'] == 'Th':
                  color='pink' 
                  lab='Th'
                  
               plt.plot([line['Wavelength(Ã…)']/10,line['Wavelength(Ã…)']/10],[0,line['Intensity']],color=color,label=lab)
               
            handles, labels = plt.gca().get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            plt.legend(unique.values(), unique.keys())
            
        return lines
          
           
    def generate_spectra(self,lines,photon_noise,readout_noise,plot=True):
        
        self.tag = 'spectrum' # used to ensure this function has been run before a calibration object is initiated
        
        # combine lines from different lamps
        if type(lines) is list:   
            self.lines = vstack(lines)
            
        else:
            self.lines = lines
        
        # Set up variables for creating idealised spectrum
        lambda_range=[self.truth.wav_min,self.truth.wav_max]
        pix_delta_wl=np.median(lambda_range)/self.resolution/self.sampling
        pix_wl=np.arange(lambda_range[0],lambda_range[1],pix_delta_wl)
        pix_x=np.arange(len(pix_wl))
        y_lines=np.zeros_like(pix_x)
        
        # Create idealised spectrum
        
        for line in self.lines:
            
            gaussian=Gaussian1D(line['Intensity'],np.interp(line['Wavelength(Ã…)']/10,pix_wl,pix_x),self.sampling/2.355) # 2.355 converts between gaussian fwhm and std dev
            y_lines=y_lines+gaussian(pix_x)
            
        self.ideal_spectrum = y_lines
            
        # Create plot of idealised spectrum
        
        if plot is True:
            
            #Plot idealised spectrum with representative linewidth
            plt.figure(figsize=(10,5))
            plt.plot(pix_wl,self.ideal_spectrum)
            plt.xlabel('WL (nm)')
            plt.ylabel('Relative Intensity')
            plt.title('Chosen Lines Idealised Spectrum')
            
        # Convert spectrum into realistic form

        # Spectrum from instrument will have pixels as x axis, so use wav2pix fit to convert
        self.line_pix = self.truth.wav2pix(pix_wl)

        ###############   Add noise   ###############
        
        # Photon noise
        if photon_noise is True:
            phot = np.random.poisson(lam=self.ideal_spectrum) 
        else:
            phot = self.ideal_spectrum
            
        # Readout noise
        readout = np.random.normal(0,readout_noise,self.ideal_spectrum.shape)
        
        # New spectrum
        noisy_data_full = phot + readout

        # Interpolate so we have correct number of data points for pixels
        self.calib_spec = np.interp(self.truth.pix,self.line_pix,noisy_data_full)
        
        if plot is True:
            
            plt.figure(figsize=(10,5))
            plt.plot(self.line_pix,self.ideal_spectrum,'--',label='Noiseless data', c='r')
            plt.plot(self.truth.pix,self.calib_spec,label='Noisy data')
            plt.legend()
            plt.xlabel('Pixel')
            plt.ylabel('Intensity')
            plt.title('Instrument Calibration Spectrum')
            plt.show()
        
    
    def line_plotter(self,lines):
        
        # combine lines from different sources if provided with a list of lamps
        if type(lines) is list:   
            lines = vstack(lines)
        
        plt.figure()
        plt.xlim(self.truth.wav_min,self.truth.wav_max)
        plt.xlabel('WL (nm)')
        plt.ylabel('Relative Intensity')
        plt.title('Lines')
        for line in lines:
           if line['Element'] == 'Kr':
              color='blue'
              lab='Kr'
           elif line['Element'] == 'Ar':
              color='green'
              lab='Ar'
           elif line['Element'] == 'Xe':
              color='yellow' 
              lab='Xe'
           elif line['Element'] == 'Ne':
              color='orange' 
              lab='Ne'
           elif line['Element'] == 'Hg':
              color='red' 
              lab='Hg'
           elif line['Element'] == 'U':
              color='purple' 
              lab='U'
           elif line['Element'] == 'Th':
              color='pink' 
              lab='Th'
              
           plt.plot([line['Wavelength(Ã…)']/10,line['Wavelength(Ã…)']/10],[0,line['Intensity']],color=color,label=lab)
           
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys())
        
        
        