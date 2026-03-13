# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:03:19 2025

@author: bbuky
"""

from astropy import io
from astropy import table
from astropy.table import vstack, Column, Table
import re
import numpy as np
from astropy.modeling.functional_models import Gaussian1D
from matplotlib import pyplot as plt

class Spectrum:
    
    def __init__(self,truth,resolution=None,sampling=2,rel_ints=None):
        
        """ Spectrum Class
        
        Class for creating the spectrum produced on a detector during wavelength calibration. 
        This class requires a CELTS.Truth object to work.
        The line lists used are taken from NIST.
        Initialization is used to set the parameters which will be consistent across any lamp later specified for use.
        
        Parameters
        ------------
        truth : class
            The CELTS truth object containing the truth solution for your calibration and the wavelength range of interest.
        resolution : int
            The spectral resolution of your instrument. The default is none - used when a user truth is supplied.
        sampling : int
            The sampling on your detector in pixels. The default is 2 - Nyquist sampling. 
        rel_ints : dict
            The user has the ability to set the relative intensities of different lamps using this dictionary. 
            The default is None meaning the default relative intensities will be used for converting between lamps.

        Returns
        ------------
        None
        
        """
        ############### Initial Configuring: ###############
        
        if truth.tag != 'truth':
            print('Incorrect truth object has been used as an input. Please use a CELTS Truth object.')
        else:
            self.truth = truth
            
        self.resolution = resolution
        self.sampling   = sampling
        
        ########## Set the relative intensities dictionary to use ###########
        
        # case where we are using the stored conversions between different lamps
        if rel_ints == None:
            
            self.rel_ints = dict()
            
            ###################     Pencil Lamps    #########################

            #self.rel_ints['Ar']   = [1]
            #self.rel_ints['Ne']   = [1]
            #self.rel_ints['Kr']   = [1]
            #self.rel_ints['Xe']   = [1]
            
            ###############   Hollow Cathode Lamps   ########################
            
            # the values refer to the relative intensities of the elements in the order they are written
            
            self.rel_ints['ThAr'] = [0.093,1]
            self.rel_ints['ThNe'] = [0.1,1]
            self.rel_ints['UAr']  = [0.1,1]
            self.rel_ints['UNe']  = [0.1,1]
        
        # case where user sets the relative intensities themselves
        else:
            self.rel_ints = rel_ints
            
    
    def load_stored_lamp(self, lamp, max_counts=50000,plot=True):
        
        """ load stored lamp function
        
        Function for calling the data from a lamp stored within the package. A ThAr lamp and a CeNe lamp are the options.
        This function will scale the data for this lamp and apply wavelength bounds.
        
        Parameters
        ------------
        lamp : str
            The type of lamp you wish to generate lines for. The available options are 'ThAr' and 'CeNe'.
        max_counts : int
            The peak number of counts/intensity for the brightest line in your wavelength range for this lamp. The default is 50000. 
        plot : bool
            The user can set if they want to automatically receive a plot of their lines, coloured by element and scaled by rel_ints and max_counts. The default is True.
            
        Returns
        ------------
        lines : astropy table
            A table of all possible lines for your lamp and wavelength range. This includes their scaled intensities and wavelength.
        
        """
        
        all_lines = Table.read('Line_lists\\' + lamp + '.csv')
        
        # Filter the lines by wavelength
        lines=all_lines[(all_lines['Wavelength (nm)']>self.truth.wav_min) & (all_lines['Wavelength (nm)']<self.truth.wav_max)]
        
        lines['Intensity'] = Column(lines['Intensity'].astype(np.float64), name='Intensity')
        
        # apply the global scaling based on max_counts
        global_scaling = max_counts/np.max(lines['Intensity'])
        lines['Intensity'] *= global_scaling
                    
        #order the lines in the table by wavelength
        lines.sort('Wavelength (nm)')
        
        # Plot the chosen lines
        if plot is True:
            
            # Create plot of lines, coloured by element
        
            plt.figure()
            plt.xlim(self.truth.wav_min,self.truth.wav_max)
            plt.xlabel('Wavelength (nm)')
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
               elif line['Element'] == 'Ce':
                   color='deepskyblue'
                   lab='Ce'
                  
               plt.plot([line['Wavelength (nm)'],line['Wavelength (nm)']],[0,line['Intensity']],color=color,label=lab)
               
            handles, labels = plt.gca().get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            plt.legend(unique.values(), unique.keys())
            plt.show()
            
        return lines
        
        
    def lamp_builder(self,lamp,max_counts=50000,user_ints=None,plot=True):
        
        """ lamp_builder function
        
        Function for generating all the possible lines produced by a given lamp in your wavelength range.
        All intensity scaling factors (rel_ints and max_counts) are applied here.
        
        Parameters
        ------------
        lamp : str
            The type of lamp you wish to generate lines for. The available options are defined by the entries in the stored rel_ints dictionary.
        max_counts : int
            The peak number of counts/intensity for the brightest line in your wavelength range for this lamp. The default is 50000. 
        user_ints : list
            The relative intensities you would like to use in this lamp (overwrites stored values). The default is None.
        plot : bool
            The user can set if they want to automatically receive a plot of their lines, coloured by element and scaled by rel_ints and max_counts. The default is True.
            
        Returns
        ------------
        lines : astropy table
            A table of all possible lines for your lamp and wavelength range. This includes their scaled intensities and wavelength.
        
        """
        
        # convert lamp string into list of elements in that lamp
        elements = re.findall(r'[A-Z][^A-Z]*', lamp)
        #print(elements)  
        
        # overwrite stored relative intensities if user has specified their own
        if user_ints is not None:
            self.rel_ints[lamp] = user_ints
        
        ############## Create full line list object #############
        
        loc = 'Line_lists\\'
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
        
        # Filter the lines by wavelength
        lines=selected_lines[(selected_lines['Wavelength(Ã…)']/10>self.truth.wav_min) & (selected_lines['Wavelength(Ã…)']/10<self.truth.wav_max)] # /10 to account for angstroms
        
        lines['Intensity'] = Column(lines['Intensity'].astype(np.float64), name='Intensity')
        
        # set the relative scaling factors
        if len(elements) > 1:
            rel_scaling = self.rel_ints[lamp]
        else:
            rel_scaling = [1]
        
        # apply the relative scaling factors
        for i in range(len(lines)):
            for j in range(len(rel_scaling)):
                if lines['Element'][i] == elements[j]:
                    lines['Intensity'][i] = lines['Intensity'][i]*rel_scaling[j]
                    
        # apply the global scaling based on max_counts
        global_scaling = max_counts/np.max(lines['Intensity'])
        lines['Intensity'] *= global_scaling
                    
        #order the lines in the table by wavelength
        lines.sort('Wavelength(Ã…)')
        
        # round intensity values to the nearest integer
        lines['Intensity'] = np.rint(lines['Intensity'])

        # make simplified table of lines, and change wavelength unit to nanometres
        lines = lines[['Intensity', 'Wavelength(Ã…)', 'Element']].copy()
        lines['Wavelength(Ã…)'] = lines['Wavelength(Ã…)']/10
        lines.rename_column('Wavelength(Ã…)', 'Wavelength (nm)')
        
        # Plot the chosen lines
        
        if plot is True:
            
            # Create plot of lines, coloured by element
        
            plt.figure()
            plt.xlim(self.truth.wav_min,self.truth.wav_max)
            plt.xlabel('Wavelength (nm)')
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
                  
               plt.plot([line['Wavelength (nm)'],line['Wavelength (nm)']],[0,line['Intensity']],color=color,label=lab)
               
            handles, labels = plt.gca().get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            plt.legend(unique.values(), unique.keys())
            plt.show()
            
        return lines
          
           
    def generate_spectra(self,lines,photon_noise=False,readout_noise=0,seed=None,plot=True):
        
        """ generate_spectra function
        
        This function produces a predicted calibration spectrum that you would receive from your instrument.
        
        Parameters
        ------------
        lines : list, astropy table
            The lines for the lamp or lamps you wish to use. These are astropy tables produced using the lamp_builder function or provided by the user. 
            A list of astropy tables for different lamps can be provided (to enable the use of multiple lamps at once) or a single table for one lamp.
        photon_noise : bool
            The user can set if they want poisson photon noise to be applied to their spectrum. The default is False.
        readout_noise : float
            The user can set the level of readout noise they expect during their calibration in e-. The default is 0. 
        seed : int
            The user can provide a seed if they want their noise profiles to be reproducible. The default is None.
        plot : bool
            The user can set if they want to automatically receive a plot of their predicted calibration spectrum. The default is True.
            
        Returns
        ------------
        None
        
        """
        
        self.tag = 'spectrum' # used to ensure this function has been run before a calibration object is initiated
        
        # combine lines from different lamps
        if type(lines) is list:   
            self.lines = vstack(lines)
            self.lines.sort('Wavelength (nm)')
            
        else:
            self.lines = lines
            self.lines.sort('Wavelength (nm)')
            
        # check the filtering of lines in the event of a user inputted lamp being used
        self.lines=self.lines[(self.lines['Wavelength (nm)']>self.truth.wav_min) & (self.lines['Wavelength (nm)']<self.truth.wav_max)]
            
        # Create the pixel grid for your spectrum
        
        if self.truth.user_truth is True:
            self.pix = self.truth.pix # if the user supplies a truth we use that as our pixel grid
            y_lines=np.zeros_like(self.pix)
            #pix_wl = self.truth.wav
            
        else:
            # Set up variables for creating idealised spectrum
            lambda_range=[self.truth.wav_min,self.truth.wav_max]
            pix_delta_wl=np.median(lambda_range)/self.resolution/self.sampling # we're approximating delta lambda as being constant per pixel
            pix_wl=np.arange(lambda_range[0],lambda_range[1],pix_delta_wl)
            self.pix=np.arange(len(pix_wl))
            y_lines=np.zeros_like(self.pix)
        
        # Create idealised spectrum
        
        for line in self.lines:
            
            # generate gaussian for each line
            gaussian=Gaussian1D(amplitude = line['Intensity'],
                                mean      = self.truth.wav2pix(line['Wavelength (nm)'])*(len(self.pix)-1)/(len(self.truth.pix)-1), #np.interp(line['Wavelength (nm)'],pix_wl,self.pix),
                                stddev    = self.sampling/2.355) # 2.355 converts between gaussian fwhm and std dev
            
            y_lines=y_lines+gaussian(self.pix)
            
        self.ideal_spectrum = y_lines
            
        # Create plot of idealised spectrum
        
        #if plot is True:
            
            #Plot idealised spectrum with representative linewidth
         #   plt.figure(figsize=(10,5))
          #  plt.plot(pix_wl,self.ideal_spectrum)
           # plt.xlabel('Wavelength (nm)')
            #plt.ylabel('Relative Intensity')
            #plt.title('Chosen Lines Idealised Spectrum')
            
        # Convert spectrum into realistic form

        # Spectrum from instrument will have pixels as x axis, so use wav2pix fit to convert
        #self.line_pix = self.truth.wav2pix(pix_wl)

        ###############   Add noise   ###############
         
        rng = np.random.default_rng(seed)
            
        # Photon noise
        if photon_noise is True:
            phot = rng.poisson(lam=self.ideal_spectrum) 
        else:
            phot = self.ideal_spectrum
            
        # Readout noise
        readout = rng.normal(0,readout_noise,self.ideal_spectrum.shape)
        
        # New spectrum
        self.calib_spec = phot + abs(readout)

        # Interpolate so we have correct number of data points for pixels
        #self.calib_spec = np.interp(self.truth.pix,self.line_pix,noisy_data_full)
        
        if plot is True:
            
            plt.figure(figsize=(10,5))
            plt.plot(self.pix,self.ideal_spectrum,'--',label='Noiseless data', c='r')
            plt.plot(self.pix,self.calib_spec,label='Noisy data')
            plt.legend()
            plt.xlabel('Pixel')
            plt.ylabel('Intensity')
            plt.title('Instrument Calibration Spectrum')
            plt.show()
        
    
    def line_plotter(self,lines):
        
        """ line_plotter function
        
        This function produces a plot of lamp lines, coloured by element and scaled by rel_ints and max_counts. 
        It enables you to combine multiple lamps and view a plot of their lines.
        
        Parameters
        ------------
        lines : list, astropy table
            The lines for the lamp or lamps you wish to use. These are astropy tables produced using the lamp_builder function. 
            A list of astropy tables for different lamps can be provided (to enable the use of multiple lamps at once) or a single table for one lamp.
            
        Returns
        ------------
        None
        
        """
        
        # combine lines from different sources if provided with a list of lamps
        if type(lines) is list:   
            lines = vstack(lines)
            
        # check the filtering of lines in the event of a user inputted lamp being used
        lines=lines[(lines['Wavelength (nm)']>self.truth.wav_min) & (lines['Wavelength (nm)']<self.truth.wav_max)]
        
        plt.figure()
        plt.xlim(self.truth.wav_min,self.truth.wav_max)
        plt.xlabel('Wavelength (nm)')
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
           elif line['Element'] == 'Ce':
               color='deepskyblue'
               lab='Ce'
              
           plt.plot([line['Wavelength (nm)'],line['Wavelength (nm)']],[0,line['Intensity']],color=color,label=lab)
           
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys())
        plt.show()
        
        
        