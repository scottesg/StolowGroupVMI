#%% Imports

import os
os.chdir(r'../..')

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from VMI3D_Functions import (histxy, pts2img)
import imutils
from VMI3D_Fitting import fit_gauss as fg
from matplotlib.ticker import FormatStrFormatter

#%% Figure 10: NO Results
 
path = r"C:\Users\Scott\Python\VMI\data\20221003_VMI3DPaperData/NOFig/"
pts45 = np.load(path + "NO_45L.pt.npy")
ptsh = np.load(path + "NO_0L.pt.npy")
ptsv = np.load(path + "NO_90L.pt.npy")

# Params
ang = 68 # rotation angle
trange = np.array([66, 77]) # Time histogram range
nlf = 0.04 # nonlinear factor for time points
tbins = 1000 # Time bins for time histogram
tsm = 2 # Time smoothing
imsm = 2 # Image smoothing
dim = 512 # Image dimension
cen = [258, 247] # Image centre
dslice = 0.16 # Width of slices

# Time points of slices
tn = np.linspace(3, 8, 5)
tn = tn + tn*np.linspace(0, nlf, 5)

# Prepare time histograms
th, yh = histxy(ptsh[:,2], trange, tbins)
tv, yv = histxy(ptsv[:,2], trange, tbins)
t45, y45 = histxy(pts45[:,2], trange, tbins)

# Smooth histograms
yh = gaussian_filter(yh, tsm)
yv = gaussian_filter(yv, tsm)
y45 = gaussian_filter(y45, tsm)
ys = [yh, y45, yv]

# Prepare slices:
ptsh[:,2] -= trange[0]
ptsv[:,2] -= trange[0]
pts45[:,2] -= trange[0]
pts = [ptsh, pts45, ptsv]
    
# Prepare slices:
frmsh = np.zeros((5, dim, dim))
frmsv = np.zeros((5, dim, dim))
frms45 = np.zeros((5, dim, dim))
frms = [frmsh, frms45, frmsv]

for i in range(0, 3):
    for j in range(0, 5):
        frmpts = pts[i][pts[i][:,2]>tn[j]]
        frmpts = frmpts[frmpts[:,2]<(tn[j]+dslice)]
        if len(frmpts) == 0: continue
        frm = pts2img(frmpts, dim, imsm)
        frms[i][j] = imutils.rotate(frm, ang)
    
# Start time at 0
t = th - trange[0]

#%% Setup figure

angles = [90, 45, 0]

fig, ax = plt.subplot_mosaic("ABCDEF\nGHIJKL\nMNOPQR")#"\nYYYZZZ\nYYYZZZ")
def indi(s):
    if s in "ABCDEF": return 0
    if s in "GHIJKL": return 1
    if s in "MNOPQR": return 2
def indj(s):
    if s in "AGM": return 0
    if s in "BHN": return 1
    if s in "CIO": return 2
    if s in "DJP": return 3
    if s in "EKQ": return 4
    if s in "FLR": return 5

d = 140
for i in "ABCDEFGHIJKLMNOPQR":
    ax[i].axes.get_xaxis().set_visible(False)
    ax[i].axes.get_yaxis().set_ticks([])
    if not i in "AGM":
        ax[i].axes.get_yaxis().set_visible(False)
        ax[i].set_xlim([cen[0]-d, cen[0]+d])
        ax[i].set_ylim([cen[1]-d, cen[1]+d])
        
frmsh = np.array(frmsh)
frms45 = np.array(frms45)
frmsv = np.array(frmsv)

gvmin = 95
gvmax = 99.9
gmins = np.zeros(3)
gmaxs = np.zeros(3)

for i in range(0, 3):
    gmaxs[i] = np.percentile(frms[i], gvmax)
    gmins[i] = np.percentile(frms[i], gvmin)

# Histograms
for i in "AGM":
    ax[i].plot(t, ys[indi(i)], lw=3, color='black')
    ax[i].set_ylabel("\u03B8 = {}\N{DEGREE SIGN}".format(angles[indi(i)]), fontsize=30)
    ax[i].vlines(tn, 0, max(ys[indi(i)]), lw=5, color='red')
    ax[i].set_xticks([0, tn[0], tn[2], tn[4]])
    ax[i].tick_params(labelsize=20, width=2, length=6)
    #ax[i].set_aspect('1./ax[i].get_data_ratio()')
    ax[i].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax['A'].set_title("Time (ns)", fontsize=30)
ax["M"].axes.get_xaxis().set_visible(True)

# Slices
for i in "BCDEFHIJKLNOPQR":
    ax[i].imshow(frms[indi(i)][indj(i)-1], cmap='nipy_spectral_r', 
                 vmin=gmins[indi(i)], vmax=gmaxs[indi(i)])
    if indi(i) == 0:
        ax[i].set_title("{:.2f} ns".format(tn[indj(i)-1]), fontsize=30)

#%%

fig, ax = plt.subplot_mosaic("YZ")

# PES

energy = 0.84

# pesI = np.load(path+"../PES/noPESinv_DAVIS3.npy")
# eVI = pesI[0]
# IrI = pesI[1]
# #IrI = gaussian_filter(IrI, 2)
# IrI = IrI/max(IrI)

pesI = np.load(path+"../PES/noPESinv_Slice.npy")
pesI = np.reshape(pesI, (4, 497))
eVI = pesI[2]
IrI = pesI[3]
IrI = IrI/max(IrI[2:])

pesXY = np.load(path+"../PES/noPESsliceXY.npy")
eVXY = pesXY[0]
IrXY = pesXY[1]
IrXY = IrXY/max(IrXY)

# pesXY = np.load(path+"../PES/noPESsliceXY_A.npy")
# pesXY = np.reshape(pesXY, (4, 203))
# eVXY = pesXY[2]
# IrXY = pesXY[3]
# IrXY = IrXY/max(IrXY[2:])

pesXZ = np.load(path+"../PES/noPESsliceXZ.npy")
eVXZ = pesXZ[0]
IrXZ = pesXZ[1]
IrXZ = IrXZ/max(IrXZ)

#  [\u0394E/E = 3.8%]
#  [\u0394E/E = 2.2%]
ax['Y'].plot(eVI, IrI+0.3, label="Inverted", color="red", lw=3)
ax['Y'].plot(eVXY, IrXY, label="XY Slice", color="blue", lw=3)
#ax['Y'].plot(eVXZ, IrXZ, label="XZ Slice", lw=3)
ax['Y'].set_xlabel("Energy (eV)", fontsize=20)
ax['Y'].set_ylabel("Intensity (arb)", fontsize=20)
#ax['Y'].set_title("Photoelectron Spectrum", fontsize=25)
ax['Y'].set_xlim([0, 1.5])
ax['Y'].tick_params(labelsize=20, width=2, length=6)
leg = ax['Y'].legend(fontsize=20)
leg.set_draggable(1)

# PAD

padI = np.load(path+"../PAD/noPADinv_2.npy")
angI = padI[:,0]
intI = padI[:,1]
fitI = padI[:,2]
s = fitI[180]
h = fitI[0]
intI -= s
fitI -= s
intI /= h
fitI /= h

# impolarI = np.load(path+"../PAD/noPADinv_ImPolar2.npy")
# r1 = 423
# r2 = 430
# impolarI = impolarI[:,r1:r2]
# angI = np.arange(0, 180, 1)
# intI = np.mean(impolarI, 1)
# s = intI[90]
# h = intI[0]
# intI -= s
# intI /= h

padXY = np.load(path+"../PAD/noPADsliceXY.npy")
angXY = padXY[:,0]
intXY = padXY[:,1]
fitXY = padXY[:,2]
s = fitXY[180]
h = fitXY[0]
intXY -= s
fitXY -= s
intXY /= h
fitXY /= h

padXZ = np.load(path+"../PAD/noPADsliceXZ.npy")
angXZ = padXZ[:,0]
intXZ = padXZ[:,1]
fitXZ = padXZ[:,2]
s = fitXZ[180]
h = fitXZ[0]
intXZ -= s
fitXZ -= s
intXZ /= h
fitXZ /= h

padYZ = np.load(path+"../PAD/noPADsliceYZ.npy")
angYZ = padYZ[:,0]
intYZ = padYZ[:,1]
fitYZ = padYZ[:,2]
s = fitYZ[180]
h = fitYZ[0]*2
intYZ -= s
fitYZ -= s
intYZ /= h
fitYZ /= h

beta = "\u03B2\u2082\u2080/\u03B2\u2080\u2080"

# For Andrey inversion: 1.67\u00B10.04
# For DAVIS Inversion: 1.70\u00B10.07
ax['Z'].plot(angI, intI+1.5, label="Inverted ["+beta+" = 1.67\u00B10.12]", color="red", marker='+', lw=0)
ax['Z'].plot(angI, fitI+1.5, color="black", lw=4)

ax['Z'].plot(angXY, intXY+1, label="XY Slice ["+beta+" = 1.62\u00B10.06]", marker="x", color="blue", lw=0)
ax['Z'].plot(angXY, fitXY+1, color="black", lw=4)

ax['Z'].plot(angXZ, intXZ+0.5, label="XZ Slice ["+beta+" = 1.68\u00B10.05]", marker="1", color="green", lw=0)
ax['Z'].plot(angXZ, fitXZ+0.5, color="black", lw=4)

ax['Z'].plot(angYZ, intYZ, label="YZ Slice ["+beta+" = 0.13\u00B10.06]", marker="^", markerfacecolor='none', color="purple", lw=0)
ax['Z'].plot(angYZ, fitYZ, color="black", lw=4)

ax['Z'].set_xlabel("Photoemission Angle (deg)", fontsize=20)
ax['Z'].set_ylabel("Intensity (arb)", fontsize=20)
#ax['Z'].set_title("Photoelectron Angular Distribution", fontsize=25)
ax['Z'].set_xlim([0, 180])
ax['Z'].set_ylim([-0.5, 3.1])
ax['Z'].tick_params(labelsize=20, width=2, length=6)
leg = ax['Z'].legend(fontsize=20)
leg.set_draggable(1)

#%% PES

energy = 0.84

# Fitting
d = 0.08
i0a = np.searchsorted(eVI, energy-6*d)
i0b = np.searchsorted(eVI, energy-2*d)
bk = np.mean(IrI[i0a:i0b])
IrI -= bk
i1 = np.searchsorted(eVI, energy-d)
i2 = np.searchsorted(eVI, energy+d)
xsI = eVI[i1:i2]
ys = IrI[i1:i2]
fityI, params, pcov = fg(xsI, ys, 0, np.max(IrI[2:]), energy, 0.04, czero=True)
wI = params[2]*2.355

d = 0.06
i0a = np.searchsorted(eVXY, energy-6*d)
i0b = np.searchsorted(eVXY, energy-2*d)
bk = np.mean(IrXY[i0a:i0b])
IrXY -= bk
i1 = np.searchsorted(eVXY, energy-d)
i2 = np.searchsorted(eVXY, energy+d)
xsXY = eVXY[i1:i2]
ys = IrXY[i1:i2]
fityXY, params, pcov = fg(xsXY, ys, 0, np.max(IrXY), energy, 0.04, czero=True)
wXY = params[2]*2.355

d = 0.12
i0a = np.searchsorted(eVXZ, energy-6*d)
i0b = np.searchsorted(eVXZ, energy-2*d)
bk = np.mean(IrXZ[i0a:i0b])
IrXZ -= bk
i1 = np.searchsorted(eVXZ, energy-d)
i2 = np.searchsorted(eVXZ, energy+d)
xsXZ = eVXZ[i1:i2]
ys = IrXZ[i1:i2]
fityXZ, params, pcov = fg(xsXZ, ys, 0, np.max(IrXZ), energy, 0.04, czero=True)
wXZ = params[2]*2.355

plt.figure()
plt.plot(eVI, IrI+0.3, label="Inverted", lw=3)
plt.plot(eVXY, IrXY, label="XY Slice", lw=3)
plt.plot(eVXZ, IrXZ+0.6, label="XZ Slice", lw=3)
plt.plot(xsI, fityI+0.3, ls=":", lw=3)
plt.plot(xsXY, fityXY, ls=":", lw=3)
plt.plot(xsXZ, fityXZ+0.6, ls=":", lw=3)
plt.xlabel("Energy (eV)", fontsize=20)
plt.ylabel("Intensity (Normalized)", fontsize=20)
plt.title("Photoelectron Spectrum, NO 6.2 ns", fontsize=25)
plt.xlim([0, 1.5])
plt.gca().tick_params(labelsize=20, width=2, length=6)

leg = plt.gca().legend(fontsize=20)
plt.gca().tick_params(labelsize=15)
leg.set_draggable(1)

print("[I] DE/E: {}".format(np.abs(wI)/energy))
print("[XY] DE/E: {}".format(np.abs(wXY)/energy))
print("[XZ] DE/E: {}".format(np.abs(wXZ)/energy))