# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:09:15 2026

@author: tsl29789
"""

# Import classes
from Truth import Truth
from Spectrum import Spectrum
from Calibration import Calibration
import numpy as np
from matplotlib import pyplot as plt
import scipy
from matplotlib.gridspec import GridSpec

#%% Load dispersion relation to use as truth

dispersion = np.loadtxt('Tutorials\\Model_Dispersion_LR.txt',skiprows=11,encoding='UTF-16 LE')

# find delta wav
deltaL = dispersion[1,0] - dispersion[0,0]
L = dispersion[0,0]
R_req = 3500
N_pix = 2.5
deltaL_pix = L/(R_req*N_pix)
pixel_size = deltaL_pix/deltaL*(dispersion[1,1]-dispersion[0,1])

dispersion[:,0] = dispersion[:,0]*1000
dispersion[:,1] = dispersion[:,1]/pixel_size
dispersion = [dispersion[:,0],dispersion[:,1]]

# dispersion now in form [wav (nm), pix]

#%% Create truth class

truth = Truth(truth_data=dispersion, fit_quality=0.005, interp_truth=True)

#%% Create spectrum class

spec1 = Spectrum(truth=truth, sampling=2.5)

lamp_ne = spec1.lamp_builder(lamp='Ne',max_counts=50000,user_ints=None,plot=True)
lamp_ar = spec1.lamp_builder(lamp='Ar',max_counts=50000,user_ints=None,plot=True)
lamp_kr = spec1.lamp_builder(lamp='Kr',max_counts=50000,user_ints=None,plot=True)

spec1.line_plotter([lamp_ne,lamp_ar,lamp_kr])

#%% Set orders to test for calibration fitting

orders = np.arange(1,21)

#%% Generate instrument calibration spectrum and conduct calibration for Ne lamp

# Generate your sample spectra
spec1.generate_spectra(lines=lamp_ne, photon_noise=True, readout_noise=10, seed=10, plot=True) # set noise in spectrum when you generate it

cal_ne = Calibration(truth=truth,spectrum=spec1,orders=orders,amp_cutoff=1000,sttdev_cutoff=20,plot=False,filter_lines=True,print_line_fit_data=False)

max_resids_ne = np.max(cal_ne.resids_wav, axis=1)

best_ind_ne = np.argmin(max_resids_ne)
best_order_ne = orders[best_ind_ne]

plt.figure()
plt.plot(orders,max_resids_ne)
plt.xlabel('Order of Fit')
plt.ylabel('Max residual (nm)')
plt.title('Max Residuals for different fits')
plt.show()


fig = plt.figure()
gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[5, 1], hspace=0, wspace=0)

ax_main = fig.add_subplot(gs[0, 0])
ax_yres = fig.add_subplot(gs[0, 1], sharey=ax_main)
ax_xres = fig.add_subplot(gs[1, 0], sharex=ax_main)

# Main plot
ax_main.plot(spec1.pix, cal_ne.upd_truth_wav, label='Truth', c='orangered')
ax_main.scatter(cal_ne.points_pix, cal_ne.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
ax_main.plot(spec1.pix, cal_ne.calib_fit[best_ind_ne], label='Fit, order = ' + str(best_order_ne), c='tab:green')
ax_main.set_ylabel('Wavelength (nm)')
ax_main.set_xticks([-6000,-4000,-2000,0,2000,4000,6000])
ax_main.legend()

# Y-residuals (horizontal residual plot)
ax_yres.plot(cal_ne.resids_pix[best_ind_ne], cal_ne.calib_fit[best_ind_ne], c='tab:green')
ax_yres.set_xlabel("Residual (pix)")
ax_yres.set_xticks([0,0.03])
ax_yres.xaxis.label.set_x(0.56)
ax_yres.tick_params(labelleft=False)

# X-residuals (vertical residual plot)
ax_xres.plot(spec1.pix, cal_ne.resids_wav[best_ind_ne], c='tab:green')
ax_xres.set_ylabel("Residual (nm)")
ax_xres.set_xlabel("Pixel")

plt.tight_layout()
plt.show()

#%% Generate instrument calibration spectrum and conduct calibration for Ar lamp

# Generate your sample spectra
spec1.generate_spectra(lines=lamp_ar, photon_noise=True, readout_noise=10, seed=10, plot=True) # set noise in spectrum when you generate it

cal_ar = Calibration(truth=truth,spectrum=spec1,orders=orders,amp_cutoff=1000,sttdev_cutoff=20,plot=False,filter_lines=True,print_line_fit_data=False)

max_resids_ar = np.max(cal_ar.resids_wav, axis=1)

best_ind_ar = np.argmin(max_resids_ar)
best_order_ar = orders[best_ind_ar]

plt.figure()
plt.plot(orders,max_resids_ar)
plt.xlabel('Order of Fit')
plt.ylabel('Max residual (nm)')
plt.title('Max Residuals for different fits')
plt.show()


fig = plt.figure()
gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[5, 1], hspace=0, wspace=0)

ax_main = fig.add_subplot(gs[0, 0])
ax_yres = fig.add_subplot(gs[0, 1], sharey=ax_main)
ax_xres = fig.add_subplot(gs[1, 0], sharex=ax_main)

# Main plot
ax_main.plot(spec1.pix, cal_ar.upd_truth_wav, label='Truth', c='orangered')
ax_main.scatter(cal_ar.points_pix, cal_ar.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
ax_main.plot(spec1.pix, cal_ar.calib_fit[best_ind_ar], label='Fit, order = ' + str(best_order_ar), c='tab:green')
ax_main.set_ylabel('Wavelength (nm)')
ax_main.set_xticks([-6000,-4000,-2000,0,2000,4000,6000])
ax_main.legend()

# Y-residuals (horizontal residual plot)
ax_yres.plot(cal_ar.resids_pix[best_ind_ar], cal_ar.calib_fit[best_ind_ar], c='tab:green')
ax_yres.set_xlabel("Residual (pix)")
ax_yres.set_xticks([0,10])
ax_yres.xaxis.label.set_x(0.56)
ax_yres.tick_params(labelleft=False)

# X-residuals (vertical residual plot)
ax_xres.plot(spec1.pix, cal_ar.resids_wav[best_ind_ar], c='tab:green')
ax_xres.set_ylabel("Residual (nm)")
ax_xres.set_xlabel("Pixel")

plt.tight_layout()
plt.show()

#%% Generate instrument calibration spectrum and conduct calibration for Kr lamp

# Generate your sample spectra
spec1.generate_spectra(lines=lamp_kr, photon_noise=True, readout_noise=10, seed=10, plot=True) # set noise in spectrum when you generate it

cal_kr = Calibration(truth=truth,spectrum=spec1,orders=orders,amp_cutoff=1000,sttdev_cutoff=20,plot=False,filter_lines=True,print_line_fit_data=False)

max_resids_kr = np.max(cal_kr.resids_wav, axis=1)

best_ind_kr = np.argmin(max_resids_kr)
best_order_kr = orders[best_ind_kr]

plt.figure()
plt.plot(orders,max_resids_kr)
plt.xlabel('Order of Fit')
plt.ylabel('Max residual (nm)')
plt.title('Max Residuals for different fits')
plt.show()


fig = plt.figure()
gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[5, 1], hspace=0, wspace=0)

ax_main = fig.add_subplot(gs[0, 0])
ax_yres = fig.add_subplot(gs[0, 1], sharey=ax_main)
ax_xres = fig.add_subplot(gs[1, 0], sharex=ax_main)

# Main plot
ax_main.plot(spec1.pix, cal_kr.upd_truth_wav, label='Truth', c='orangered')
ax_main.scatter(cal_kr.points_pix, cal_kr.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
ax_main.plot(spec1.pix, cal_kr.calib_fit[best_ind_kr], label='Fit, order = ' + str(best_order_kr), c='tab:green')
ax_main.set_ylabel('Wavelength (nm)')
ax_main.set_xticks([-6000,-4000,-2000,0,2000,4000,6000])
ax_main.legend()

# Y-residuals (horizontal residual plot)
ax_yres.plot(cal_kr.resids_pix[best_ind_kr], cal_kr.calib_fit[best_ind_kr], c='tab:green')
ax_yres.set_xlabel("Residual (pix)")
ax_yres.set_xticks([0,0.03])
ax_yres.xaxis.label.set_x(0.56)
ax_yres.tick_params(labelleft=False)

# X-residuals (vertical residual plot)
ax_xres.plot(spec1.pix, cal_kr.resids_wav[best_ind_kr], c='tab:green')
ax_xres.set_ylabel("Residual (nm)")
ax_xres.set_xlabel("Pixel")

plt.tight_layout()
plt.show()

#%% Generate instrument calibration spectrum and conduct calibration for all lamps combined

# Generate your sample spectra
spec1.generate_spectra(lines=[lamp_ne, lamp_ar, lamp_kr], photon_noise=True, readout_noise=10, seed=10, plot=True) # set noise in spectrum when you generate it

cal_all = Calibration(truth=truth,spectrum=spec1,orders=orders,amp_cutoff=1000,sttdev_cutoff=20,plot=False,filter_lines=True,print_line_fit_data=False)

max_resids_all = np.max(cal_all.resids_wav, axis=1)

best_ind_all = np.argmin(max_resids_all)
best_order_all = orders[best_ind_all]

plt.figure()
plt.plot(orders,max_resids_all)
plt.xlabel('Order of Fit')
plt.ylabel('Max residual (nm)')
plt.title('Max Residuals for different fits')
plt.show()


fig = plt.figure()
gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[5, 1], hspace=0, wspace=0)

ax_main = fig.add_subplot(gs[0, 0])
ax_yres = fig.add_subplot(gs[0, 1], sharey=ax_main)
ax_xres = fig.add_subplot(gs[1, 0], sharex=ax_main)

# Main plot
ax_main.plot(spec1.pix, cal_all.upd_truth_wav, label='Truth', c='orangered')
ax_main.scatter(cal_all.points_pix, cal_all.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
ax_main.plot(spec1.pix, cal_all.calib_fit[best_ind_all], label='Fit, order = ' + str(best_order_all), c='tab:green')
ax_main.set_ylabel('Wavelength (nm)')
ax_main.set_xticks([-6000,-4000,-2000,0,2000,4000,6000])
ax_main.legend()

# Y-residuals (horizontal residual plot)
ax_yres.plot(cal_all.resids_pix[best_ind_all], cal_all.calib_fit[best_ind_all], c='tab:green')
ax_yres.set_xlabel("Residual (pix)")
ax_yres.set_xticks([-0.04,0.04])
ax_yres.xaxis.label.set_x(0.56)
ax_yres.tick_params(labelleft=False)

# X-residuals (vertical residual plot)
ax_xres.plot(spec1.pix, cal_all.resids_wav[best_ind_all], c='tab:green')
ax_xres.set_ylabel("Residual (nm)")
ax_xres.set_xlabel("Pixel")

plt.tight_layout()
plt.show()