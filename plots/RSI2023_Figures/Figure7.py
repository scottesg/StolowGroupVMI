#%% Imports

import os
os.chdir(r'../..')

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from VMI3D_Functions import xydetmap

#%% Figure 7: Spatial Correction and Correlation Comparison

# Load
path = r"C:\Users\Scott\Python\VMI\data\20221003_VMI3DPaperData/"
pointspath1 = path + "xenon_9ns_uc.pt.npy"
pointspath2 = path + "xenon_9ns.pt.npy"
points1 = np.load(pointspath1)
points2 = np.load(pointspath2)

# Rotate Points
cen = [248, 257]
rotangle = 25*(np.pi/180)
xc = points1[:,0] - cen[0]
yc = points1[:,1] - cen[1]
x2 = xc*np.cos(rotangle) + yc*np.sin(rotangle)
y2 = yc*np.cos(rotangle) - xc*np.sin(rotangle)
points1[:,0] = x2 + cen[0]
points1[:,1] = y2 + cen[0]

gsize = 20
tilelim = 10
dim = 512
binpoints, detmap, num_map = xydetmap(points1, gsize, tilelim,
                                      [0, dim], [0, dim], plot=False)
var = detmap['var']

xmax1 = 3.8*(0.65/0.55)
ymax1 = 0.55*(0.65/0.55)
points1 = points1[points1[:,3]<xmax1]
points1 = points1[points1[:,4]<ymax1]

xmax2 = 0.65
ymax2 = 0.65
points2 = points2[points2[:,3]<xmax2]
points2 = points2[points2[:,4]<ymax2]

ci1 = points1[:,3]/xmax1
pi1 = points1[:,4]/ymax1

ci2 = points2[:,3]/xmax2
pi2 = points2[:,4]/ymax2
nbins = 200
    
zmin = 0
zmax = 150

bins = [np.linspace(0, 1, nbins), np.linspace(0, 1, nbins)]

x0 = np.array([1, 0])
def f(x, t, y):
        return x[1] + x[0]*t - y 

re_lsq1 = least_squares(f, x0, loss='cauchy', f_scale=0.001, args=(ci1, pi1))
slope1, yint1 = re_lsq1.x
xul1 = max(ci1)

re_lsq2 = least_squares(f, x0, loss='cauchy', f_scale=0.001, args=(ci2, pi2))
slope2, yint2 = re_lsq2.x
xul2 = max(ci2)

pred1 = yint1 + slope1*ci1
corr_matrix1 = np.corrcoef(pi1, pred1)
corr1 = corr_matrix1[0,1]
R_sq1 = corr1**2
print(R_sq1)

pred2 = yint2 + slope2*ci2
corr_matrix2 = np.corrcoef(pi2, pred2)
corr2 = corr_matrix2[0,1]
R_sq2 = corr2**2
print(R_sq2)

#%% plots

fig, ax = plt.subplots(1,3)
ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]

im = ax1.hist2d(ci1, pi1, bins, cmap='nipy_spectral_r', vmin=zmin, vmax=zmax)
ax1.plot([0, xul1], [yint1, yint1+slope1*xul1], color='black',
        lw=4, ls='--', label='Fit ($R^2$ = {:1.2f})'.format(R_sq1))
ax1.set_ylabel("Time Amplitude", fontsize=20)
ax1.set_xlabel("Position Amplitude", fontsize=20)
ax1.tick_params(labelsize=20, width=2, length=5)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim((0, 1))
ax1.set_ylim((0, 1))
leg = ax1.legend(fontsize=18)
leg.set_draggable(True)
cbar = plt.colorbar(im[3], ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20)

vmax=0.18
im = ax2.imshow(var/vmax, vmin=0.45, vmax=1.1, cmap='nipy_spectral_r')
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20)

im = ax3.hist2d(ci2, pi2, bins, cmap='nipy_spectral_r', vmin=zmin, vmax=zmax)
ax3.plot([0, xul2], [yint2, yint2+slope2*xul2], color='black',
        lw=4, ls='--', label='Fit ($R^2$ = {:1.2f})'.format(R_sq2))
ax3.set_ylabel("Time Amplitude", fontsize=20)
ax3.set_xlabel("Position Amplitude", fontsize=20)
ax3.tick_params(labelsize=20, width=2, length=5)
ax3.set_aspect('equal', adjustable='box')
ax3.set_xlim((0, 1))
ax3.set_ylim((0, 1))
leg = ax3.legend(fontsize=18)
leg.set_draggable(True)
cbar = plt.colorbar(im[3], ax=ax3, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20)

#%% For Poster Gordon 2022
fig, ax3 = plt.subplots()

im = ax3.hist2d(ci2, pi2, bins, cmap='nipy_spectral_r', vmin=zmin, vmax=zmax)
ax3.plot([0, xul2], [yint2, yint2+slope2*xul2], color='black',
        lw=4, ls='--', label='Fit ($R^2$ = {:1.2f})'.format(R_sq2))
ax3.set_ylabel("Time Amplitude", fontsize=30)
ax3.set_xlabel("Position Amplitude", fontsize=30)
ax3.tick_params(labelsize=20, width=2, length=5)
ax3.set_aspect('equal', adjustable='box')
ax3.set_xlim((0, 1))
ax3.set_ylim((0, 1))
leg = ax3.legend(fontsize=30)
leg.set_draggable(True)
cbar = plt.colorbar(im[3])#, ax=ax3, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=25)