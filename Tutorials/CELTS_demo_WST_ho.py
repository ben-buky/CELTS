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
from matplotlib.gridspec import GridSpec
import scipy

#%% Load dispersion relation to use as truth

dispersion = np.loadtxt('Tutorials\\Model_Dispersion_LR.txt',skiprows=11,encoding='UTF-16 LE')

# find delta wav
L = 0.476 #min 452, max 524
BW = L/10
dispersion[:,0] = np.linspace(L-BW/2,L+BW/2,len(dispersion[:,0]))

# find delta wav
deltaL = dispersion[1,0] - dispersion[0,0]
L = dispersion[0,0]
R_req = 40000
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

lamp_thar = spec1.lamp_builder(lamp='ThAr',max_counts=50000,user_ints=None,plot=True)
lamp_une = spec1.lamp_builder(lamp='UNe',max_counts=50000,user_ints=None,plot=True)

spec1.line_plotter([lamp_thar,lamp_une])

#%% Set orders to test for calibration fitting

orders = np.arange(1,21)

#%% Generate instrument calibration spectrum and conduct calibration for ThAr lamp

# Generate your sample spectra
spec1.generate_spectra(lines=lamp_thar, photon_noise=True, readout_noise=10, seed=10, plot=True) # set noise in spectrum when you generate it

cal_thar = Calibration(truth=truth,spectrum=spec1,orders=orders,amp_cutoff=1000,sttdev_cutoff=20,plot=False,filter_lines=True,print_line_fit_data=False)

max_resids_thar = np.max(cal_thar.resids_wav, axis=1) 

best_ind_thar = np.argmin(max_resids_thar)
best_order_thar = orders[best_ind_thar]

plt.figure()
plt.plot(orders,max_resids_thar)
plt.xlabel('Order of Fit')
plt.ylabel('Max residual (nm)')
plt.title('Max Residuals for different fits')
plt.show()

# fig = plt.figure()
# gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[4,1])
# axs = gs.subplots(sharex='col')
# axs[1].set_xlabel('Pixel')
# axs[1].set_ylabel('Residual (nm)', labelpad=20)
# axs[0].set(ylabel = 'Wavelength (nm)')
# plt.suptitle('Calibration Fit')
# axs[0].plot(spec1.pix, cal_thar.upd_truth_wav, label='Truth', c='orangered')
# axs[0].scatter(cal_thar.points_pix, cal_thar.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
# axs[0].plot(spec1.pix, cal_thar.calib_fit[best_ind_thar], label='Fit, order = ' + str(best_order_thar), c='tab:green')
# axs[1].plot(spec1.pix, cal_thar.resids_wav[best_ind_thar], c='tab:green')
# axs[0].legend()
# fig.tight_layout()
# plt.show()

fig = plt.figure()
gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[5, 1], hspace=0, wspace=0)

ax_main = fig.add_subplot(gs[0, 0])
ax_yres = fig.add_subplot(gs[0, 1], sharey=ax_main)
ax_xres = fig.add_subplot(gs[1, 0], sharex=ax_main)

# Main plot
ax_main.plot(spec1.pix, cal_thar.upd_truth_wav, label='Truth', c='orangered')
ax_main.scatter(cal_thar.points_pix, cal_thar.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
ax_main.plot(spec1.pix, cal_thar.calib_fit[best_ind_thar], label='Fit, order = ' + str(best_order_thar), c='tab:green')
ax_main.set_ylabel('Wavelength (nm)')
ax_main.legend()

# Y-residuals (horizontal residual plot)
ax_yres.plot(cal_thar.resids_pix[best_ind_thar], cal_thar.calib_fit[best_ind_thar], c='tab:green')
ax_yres.set_xlabel("Residual (pix)")
ax_yres.xaxis.label.set_x(0.56)
ax_yres.tick_params(labelleft=False)

# X-residuals (vertical residual plot)
ax_xres.plot(spec1.pix, cal_thar.resids_wav[best_ind_thar], c='tab:green')
ax_xres.set_ylabel("Residual (nm)")
ax_xres.set_xlabel("Pixel")

plt.tight_layout()
plt.show()

#%% Generate instrument calibration spectrum and conduct calibration for UNe lamp

# Generate your sample spectra
spec1.generate_spectra(lines=lamp_une, photon_noise=True, readout_noise=10, seed=10, plot=True) # set noise in spectrum when you generate it

cal_une = Calibration(truth=truth,spectrum=spec1,orders=orders,amp_cutoff=1000,sttdev_cutoff=20,plot=False,filter_lines=True,print_line_fit_data=False)

max_resids_une = np.max(cal_une.resids_wav, axis=1)

best_ind_une = np.argmin(max_resids_une)
best_order_une = orders[best_ind_une]

plt.figure()
plt.plot(orders,max_resids_une)
plt.xlabel('Order of Fit')
plt.ylabel('Max residual (nm)')
plt.title('Max Residuals for different fits')
plt.show()

# fig = plt.figure()
# gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[4,1])
# axs = gs.subplots(sharex='col')
# axs[1].set_xlabel('Pixel')
# axs[1].set_ylabel('Residual (nm)', labelpad=20)
# axs[0].set(ylabel = 'Wavelength (nm)')
# plt.suptitle('Calibration Fit')
# axs[0].plot(spec1.pix, cal_une.upd_truth_wav, label='Truth', c='orangered')
# axs[0].scatter(cal_une.points_pix, cal_une.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
# axs[0].plot(spec1.pix, cal_une.calib_fit[best_ind_une], label='Fit, order = ' + str(best_order_une), c='tab:green')
# axs[1].plot(spec1.pix, cal_une.resids_wav[best_ind_une], c='tab:green')
# axs[0].legend()
# fig.tight_layout()
# plt.show()


fig = plt.figure()
gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[5, 1], hspace=0, wspace=0)

ax_main = fig.add_subplot(gs[0, 0])
ax_yres = fig.add_subplot(gs[0, 1], sharey=ax_main)
ax_xres = fig.add_subplot(gs[1, 0], sharex=ax_main)

# Main plot
ax_main.plot(spec1.pix, cal_une.upd_truth_wav, label='Truth', c='orangered')
ax_main.scatter(cal_une.points_pix, cal_une.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
ax_main.plot(spec1.pix, cal_une.calib_fit[best_ind_une], label='Fit, order = ' + str(best_order_une), c='tab:green')
ax_main.set_ylabel('Wavelength (nm)')
ax_main.set_yticks([460,470,480,490,500])
ax_main.legend()

# Y-residuals (horizontal residual plot)
ax_yres.plot(cal_une.resids_pix[best_ind_une], cal_une.calib_fit[best_ind_une], c='tab:green')
ax_yres.set_xlabel("Residual (pix)")
ax_yres.xaxis.label.set_x(0.56)
ax_yres.tick_params(labelleft=False)

# X-residuals (vertical residual plot)
ax_xres.plot(spec1.pix, cal_une.resids_wav[best_ind_une], c='tab:green')
ax_xres.set_ylabel("Residual (nm)")
ax_xres.set_xlabel("Pixel")

plt.tight_layout()
plt.show()

#%% Generate instrument calibration spectrum and conduct calibration for all lamps combined

# Generate your sample spectra
spec1.generate_spectra(lines=[lamp_thar, lamp_une], photon_noise=True, readout_noise=10, seed=10, plot=True) # set noise in spectrum when you generate it

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

# fig = plt.figure()
# gs = fig.add_gridspec(2,1, hspace=0, height_ratios=[4,1])
# axs = gs.subplots(sharex='col')
# axs[1].set_xlabel('Pixel')
# axs[1].set_ylabel('Residual (nm)', labelpad=20)
# axs[0].set(ylabel = 'Wavelength (nm)')
# plt.suptitle('Calibration Fit')
# axs[0].plot(spec1.pix, cal_all.upd_truth_wav, label='Truth', c='orangered')
# axs[0].scatter(cal_all.points_pix, cal_all.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
# axs[0].plot(spec1.pix, cal_all.calib_fit[best_ind_all], label='Fit, order = ' + str(best_order_all), c='tab:green')
# axs[1].plot(spec1.pix, cal_all.resids_wav[best_ind_all], c='tab:green')
# axs[0].legend()
# fig.tight_layout()
# plt.show()

fig = plt.figure(figsize=(7,5))
gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[5, 1], hspace=0, wspace=0)

ax_main = fig.add_subplot(gs[0, 0])
ax_yres = fig.add_subplot(gs[0, 1], sharey=ax_main)
ax_xres = fig.add_subplot(gs[1, 0], sharex=ax_main)

# Main plot
ax_main.plot(spec1.pix, cal_all.upd_truth_wav, label='Truth', c='orangered')
ax_main.scatter(cal_all.points_pix, cal_all.points_wav, label='Spectrum points', c='deepskyblue', marker='x')
ax_main.plot(spec1.pix, cal_all.calib_fit[best_ind_all], label='Fit, order = ' + str(best_order_all), c='tab:green')
ax_main.set_ylabel('Wavelength (nm)')
ax_main.set_yticks([460,470,480,490,500])
ax_main.legend()

# Y-residuals (horizontal residual plot)
ax_yres.plot(cal_all.resids_pix[best_ind_all], cal_all.calib_fit[best_ind_all], c='tab:green')
ax_yres.set_xlabel("Residual (pix)")
ax_yres.set_xticks([0.1,0.2])
ax_yres.xaxis.label.set_x(0.57)
ax_yres.tick_params(labelleft=False)

# X-residuals (vertical residual plot)
ax_xres.plot(spec1.pix, cal_all.resids_wav[best_ind_all], c='tab:green')
ax_xres.set_ylabel("Residual (nm)")
ax_xres.set_xlabel("Pixel")

#plt.tight_layout()
plt.show()