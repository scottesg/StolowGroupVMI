#%% Imports

import os
os.chdir(r'C:\Users\Scott\Python\VMI\src')

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy import optimize
import sgolay2
from matplotlib.colors import LogNorm

#%% load data

datapath = r"C:\Users\Scott\Python\VMI\data\20240418_VMI3DPaperData2\213/213_20231213_170156/"
path1 = "mpolSsu000.npz"

data1 = np.load(datapath + path1)

ring = 2
n0 = 5 # width of momentum slice
nth = 5 # width of theta slices
smooth = 0
plot = False

FSt = 25
FSlb = 20
FSlg = 15
FStk = 15

ke = [0.054, 0.88, 2.05][ring] # eV, energy of ring
keau = ke/27.211 # in a.u.

mom = data1['mom']
theta = data1['theta']
phi = data1['phi']
DCSSP = data1['mpol']

argp = np.argmin(abs(np.sqrt(2*keau)-mom))

PHI, THETA, PP = np.meshgrid(phi, theta, mom, indexing="ij")

if plot:
    plt.figure()
    plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], np.sum(DCSSP[:,:,argp-4:argp+5], axis=2)/9, cmap="seismic", norm=LogNorm())
    plt.xlabel("Phi (deg)")
    plt.ylabel("Theta (deg)")
    cm = plt.colorbar()
    cm.set_label("Yield (arb)")

#%% smooth the collected data

sm = np.sum(DCSSP[:,:,argp-n0:argp+n0+1], axis=2) / (2*n0+1)

if smooth > 0:
    # smooth the edges in phi
    sm[0] = ss.savgol_filter(sm[0], 3, 1)
    sm[1] = ss.savgol_filter(sm[1], 3, 1)
    sm[2] = ss.savgol_filter(sm[2], 3, 1)
    sm[-1] = ss.savgol_filter(sm[-1], 3, 1)
    sm[-2] = ss.savgol_filter(sm[-2], 3, 1)
    sm[-3] = ss.savgol_filter(sm[-3], 3, 1)
    
    # full smoothing
    sphere = sgolay2.SGolayFilter2(smooth, 1)(sm)
else:
    sphere = sm

if plot:
    fig = plt.figure()
    plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], 2*sphere/sphere.max(), cmap="seismic", vmax=1, vmin=0)
    plt.xlabel("Phi (deg)")
    plt.ylabel("Theta (deg)")
    cm = plt.colorbar()
    cm.set_label("Normalized Yield (arb)")

#%% fit phi distribution to Legendre polynomials

arg90 = np.argmin(abs(theta-90))
arg180 = np.argmin(abs(theta-180))
arg270 = np.argmin(abs(theta-270))

# average over theta
phi = np.hstack((phi, phi+180))
ABExy = np.hstack((np.average(sphere[:, arg90-nth:arg90+nth+1], axis=1),
                   np.average(sphere[:, arg270-nth:arg270+nth+1], axis=1)[::-1]))
ABExz = np.hstack(((np.average(sphere[:, :nth+1], axis=1) + np.average(sphere[:, -nth:], axis=1))/2,
                   np.average(sphere[:, arg180-nth:arg180+nth+1], axis=1)[::-1]))

def legendre(x, beta2, beta4, a, ang):
    return a * (1 + beta2*0.5*(3*np.cos(np.radians(x-ang))**2 - 1) +
                beta4*(1/8)*(35*np.cos(np.radians(x-ang))**4 - 30*np.cos(np.radians(x-ang))**2 + 3))

fig = plt.figure()
fig.set_tight_layout(True)

ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

# fit xy

optim = optimize.curve_fit(legendre, phi, ABExy/ABExy.max(), p0=[1.6, 0, 1, 28])
b2 = optim[0][0]
b4 = optim[0][1]
afit = optim[0][2]
angfit = optim[0][3]
errs = np.sqrt(np.diag(optim[1]))
db2 = errs[0]
db4 = errs[1]
dang = errs[3]

ax1.plot(phi, legendre(phi, *optim[0]), ls="--", c="b",
         label=r"Fit")
ax1.plot(phi, ABExy/ABExy.max(), lw=0, marker='x', color='red', label='Data')
ax1.vlines(180, 0, 1, ls='--', lw=1, color='black', label="180\N{DEGREE SIGN}")
ax1.fill_between([180+angfit-dang, 180+angfit+dang], 0, 1, color='black', alpha=0.5,
                 label="Angular Shift = %2.1f$\N{DEGREE SIGN}\pm$%2.1f\N{DEGREE SIGN}"%(angfit, dang))
ax1.legend(loc=2, fontsize=FSlg)
ax1.set_ylabel("Yield (normalized)", fontsize=FSlb)
ax1.set_ylim([0, 1.2])
ax1.set_xlabel("$\phi$ (\N{DEGREE SIGN})", fontsize=FSlb)
ax1.set_title("XY Slice", fontsize=FSt)
ax1.tick_params(labelsize=FStk)

# fit xz

optim = optimize.curve_fit(legendre, phi, ABExz/ABExz.max(), p0=[1.6, 0, 1, 5])
b2 = optim[0][0]
b4 = optim[0][1]
afit = optim[0][2]
angfit = optim[0][3]
errs = np.sqrt(np.diag(optim[1]))
db2 = errs[0]
db4 = errs[1]
dang = errs[3]

ax2.plot(phi, legendre(phi, *optim[0]), ls="--", c="b",
         label=r"Fit")
ax2.plot(phi, ABExz/ABExz.max(), lw=0, marker='x', color='red', label='Data')
ax2.vlines(180, 0, 1, ls='--', lw=1, color='black', label="180\N{DEGREE SIGN}")
ax2.fill_between([180+angfit-dang, 180+angfit+dang], 0, 1, color='black', alpha=0.5,
                 label="Angular Shift = %2.1f$\N{DEGREE SIGN}\pm$%2.1f\N{DEGREE SIGN}"%(angfit, dang))
ax2.legend(loc=2, fontsize=FSlg)
ax2.set_ylabel("Yield (normalized)", fontsize=FSlb)
ax2.set_ylim([0, 1.2])
ax2.set_xlabel("$\phi'$ (\N{DEGREE SIGN})", fontsize=FSlb)
ax2.set_title("XZ Slice", fontsize=FSt)
ax2.tick_params(labelsize=FStk)
