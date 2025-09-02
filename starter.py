# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:23:25 2025

@author: bbuky

Adapted version of Jay's initial script'
"""

import scipy
import numpy as np
from matplotlib import pyplot as plt
from astropy.modeling.functional_models import Gaussian1D
from astropy import io
from astropy import table
from astropy.modeling import models, fitting
from scipy.optimize import curve_fit

#%% MOONS Truth: maps wavelength to pixels

loc = "C:/Users/tsl29789/OneDrive - Science and Technology Facilities Council/Documents/WaveCal_CFI/Code/"

data = scipy.io.readsav(loc + 'Ciaran_Frame1_Moons_v52_01mar2019_H_Solutions.sav')

#dict entry 'all_lamda_x_continuum' is what we need. The index of which is the trace number.
plt.figure(figsize=(8,7))
# pick a trace to be our truth
tr = data['all_lambda_x_continuum'][200] # why 200 - could pick any trace
pix = np.arange(0,len(tr))
plt.plot(tr,pix,label='MOONS Truth') 
# plot a straight line to show the truth is non linear
plt.plot([np.min(tr),np.max(tr)],[0,4095],label='Straight line')
plt.ylabel('Pixel')
plt.xlabel('Wavelength (um)')
plt.title('Truth for MOONS')
plt.legend()
plt.show()

#%% Legendre Fit to MOONS truth

from numpy.polynomial.legendre import Legendre

inds = [1,2,3,4,5,6]
fits = np.zeros((len(inds),len(tr)))
resids = np.zeros((len(inds),len(tr)))
resids_means = np.zeros(len(inds))

fig = plt.figure()
gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[4,1])
axs = gs.subplots(sharex='col')
axs[1].set(xlabel='Wavelength (um)', ylabel='Residual (pix)')
axs[0].set(ylabel = 'Pixel')
plt.suptitle('Legendre Fitting to MOONS truth')
axs[0].plot(tr,pix,label='MOONS Truth',c='m')

for i in range(len(inds)):
    
    legendre_fit = Legendre.fit(tr,pix,deg=inds[i])
    print(legendre_fit)

    y_fit = legendre_fit(tr)
    fits[i] = y_fit

    axs[0].plot(tr,y_fit,label='Order = ' + str(inds[i]))
    axs[1].plot(tr,y_fit-pix)
    
    resids[i] = y_fit-pix
    resids_means[i] = np.mean(abs(y_fit-pix))

axs[0].legend()
plt.show()

true_fit = Legendre.fit(tr,pix,deg=6)
print('True Fit = ' + str(legendre_fit))

#%% Plot Residuals

plt.figure()
plt.plot(inds,resids_means)
plt.xlabel('Order of Fit')
plt.ylabel('Mean of absolute residuals (pix)')
plt.title('Mean of Residuals for different Fits')
plt.show()


#%% Find the relevant lines from our choice of elements

#Noble gas + mercury linelists. Note: the wavelengths are in AA.
kr=io.ascii.read(loc + 'Kr.ascii')
ar=io.ascii.read(loc + 'Ar.ascii')
ne=io.ascii.read(loc + 'Ne.ascii')
xe=io.ascii.read(loc + 'Xe.ascii')
hg = io.ascii.read(loc + 'Hg.ascii')
u = io.ascii.read(loc + 'U.ascii')
th = io.ascii.read(loc + 'Th.ascii')
all_lines=table.vstack([th,ar]) # CHOOSE ELEMENTS HERE

#Instrument params
lambda_truth=[tr[0],tr[-1]] # interval of MOONS truth
resolution=4000
sampling=3

pix_delta_wl=np.median(lambda_truth)/resolution/sampling
pix_wl=np.arange(lambda_truth[0],lambda_truth[1],pix_delta_wl)
pix_x=np.arange(len(pix_wl))
y_lines=np.zeros_like(pix_x)

#Select the lines we care about
intensity_limit=1
relevant_lines=all_lines[(all_lines['Wavelength']>lambda_truth[0]*10000) & (all_lines['Wavelength']<lambda_truth[-1]*10000)] # *10 to account for angstroms
relevant_lines=relevant_lines[relevant_lines['Intensity']>intensity_limit]

#Plot lines
plt.figure()
plt.xlim(lambda_truth[0]*10000,lambda_truth[1]*10000)
plt.xlabel('WL (AA)')
plt.ylabel('Intensity')
plt.title('Relevant lines')
for line in relevant_lines:
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
      
   gaussian=Gaussian1D(line['Intensity'],np.interp(line['Wavelength']/10000,pix_wl,pix_x),sampling/2.355) # why 2.355 - converts between gaussian fwhm and std dev
   y_lines=y_lines+gaussian(pix_x)
      
   plt.plot([line['Wavelength'],line['Wavelength']],[0,line['Intensity']],color=color,label=lab)
   
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys())

#Plot 'actual' spectrum, representitive of linewidth etc.
plt.figure(figsize=(10,5))
plt.plot(pix_wl,y_lines)
plt.xlabel('WL (um)')
plt.ylabel('Intensity')
plt.title('Relevant lines spectrum')

#%% Convert relevant lines spectrum into realistic form which might be produced by instrument

# spectrum from instrument will have pixels as x axis, so use fit to truth to convert from wavelength to pixels
    
line_pix = true_fit(pix_wl)

# Add noise
shot = np.random.poisson(lam=y_lines) # SHOULD THIS BE ADDED TO Y_LINES??
readout = np.random.normal(0,4.5,y_lines.shape) 
noisy_data_full = shot+readout

# Interpolate so we have correct number of data points for pixels

noisy_data = np.interp(pix,line_pix,noisy_data_full)

plt.figure(figsize=(10,5))
plt.plot(line_pix,y_lines,'--',label='Noiseless data', c='r')
plt.plot(pix,noisy_data,label='Noisy data')
plt.legend()
plt.xlabel('Pixel')
plt.ylabel('Intensity')
plt.title('Realistic instrument calibration spectrum')
plt.show()

#%% Calculate centre of each expected line in the noisy data

noisy_lines = []

for line in relevant_lines:
    amp = line['Intensity']
    mean = true_fit(line['Wavelength']/10000) # using truth fit to estimate where the lines will be
    print(mean)
    stddev = 1 # setting this as one pixel in all cases for now
    
    g_init = Gaussian1D(amplitude=amp,mean=mean,stddev=stddev)
    #popt, pcov = curve_fit(g_init,pix,noisy_data)
    fit_g = fitting.TRFLSQFitter()
    g = fit_g(g_init,pix,noisy_data)
    
    c = int(mean)
    
    # record the estimated positions of each line
    noisy_lines.append(g.mean.value)

    plt.figure()
    plt.plot([g.mean.value,g.mean.value],[0,np.max(g(pix))+5],'--',c='tab:blue',alpha=0.7)
    plt.plot([mean,mean],[0,np.max(g(pix))+5],'--',c='r',alpha=0.7)
    plt.plot(line_pix,y_lines,label='Noiseless data', c='r')
    plt.plot(pix,noisy_data,label='Noisy data',c='tab:orange')
    plt.plot(pix,g(pix),'--',label='Fit',c='tab:blue')
    plt.legend()
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.xlim((c-10,c+10))
    plt.show()
    
noisy_lines = np.asarray(noisy_lines)

#%% Fit to relevant lines for chosen elements in an idealised case

line_wvlgths = relevant_lines['Wavelength']/10000
lines = np.interp(line_wvlgths,tr,pix)

plt.figure()
plt.plot(tr,pix,label='MOONS Truth') 
plt.scatter(line_wvlgths,lines,label='Idealised Lamp lines',c='tab:orange',marker='x')
plt.scatter(line_wvlgths,noisy_lines,label='Noisy Lamp Lines',c='tab:green',marker='x')
plt.ylabel('Pixel')
plt.xlabel('Wavelength (um)')
plt.title('Simulated Lines found from Lamp')
plt.legend()
plt.show()

# perform fitting for different orders

inds2 = [1,2,3,4,5,6]
fits2 = np.zeros((len(inds2),len(tr)))
fits3 = np.zeros((len(inds2),len(tr)))
resids2 = np.zeros((len(inds2),len(tr)))
resids3 = np.zeros((len(inds2),len(tr)))
resids_means2 = np.zeros(len(inds2))
resids_means3 = np.zeros(len(inds2))

fig = plt.figure()
gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[4,1])
axs = gs.subplots(sharex='col')
axs[1].set(xlabel='Wavelength (um)', ylabel='Residual (pix)')
axs[0].set(ylabel = 'Pixel')
plt.suptitle('Legendre Fitting to line points')
axs[0].plot(tr,pix,label='MOONS Truth',c='m')
axs[0].scatter(line_wvlgths,lines,label='Idealised Lamp lines',c='tab:orange',marker='x')
axs[0].scatter(line_wvlgths,noisy_lines,label='Noisy Lamp Lines',c='tab:green',marker='x')

for i in range(len(inds2)):
    
    legendre_fit2 = Legendre.fit(line_wvlgths,lines,deg=inds2[i])
    print(legendre_fit2)
    
    legendre_fit3 = Legendre.fit(line_wvlgths,noisy_lines,deg=inds2[i])
    print(legendre_fit3)

    y_fit2 = legendre_fit2(tr)
    fits2[i] = y_fit2
    
    y_fit3 = legendre_fit3(tr)
    fits3[i] = y_fit3

    axs[0].plot(tr,y_fit2,label='Order = ' + str(inds2[i]))
    axs[1].plot(tr,y_fit2-pix)
    
    axs[0].plot(tr,y_fit3,label='Order = ' + str(inds2[i]))
    axs[1].plot(tr,y_fit3-pix)
    
    resids2[i] = y_fit2-pix
    resids_means2[i] = np.mean(abs(y_fit2-pix))
    
    resids3[i] = y_fit3-pix
    resids_means3[i] = np.mean(abs(y_fit3-pix))

axs[0].legend()
plt.show()

#%% Plot Residuals

plt.figure()
plt.plot(inds,resids_means,label='Truth fit')
plt.plot(inds2,resids_means2,label='Idealised Lines fit')
plt.plot(inds2,resids_means3,label='Noisy Lines fit')
plt.xlabel('Order of Fit')
plt.ylabel('Mean of absolute residuals (pix)')
plt.title('Mean of Residuals for different Fits')
plt.legend()
plt.show()

