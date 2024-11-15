#%% Imports

import os
os.chdir(r'../..')

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from VMI3D_Functions import (histxy, pts2img)
from VMI3D_Fitting import fit_gauss as fg
from matplotlib.ticker import FormatStrFormatter
import imutils

#%% Figure 9: Xe Results

path = r"C:\Users\Scott\Python\VMI\data\20221003_VMI3DPaperData/"
pts = np.load(path + "xenon_30ns_full.pt.npy")

# Params
ang = -25 # rotation angle
trange = np.array([243, 283])-0.2 # Time histogram range
nlf = 0.04 # nonlinear factor for time points
tbins = 2000 # Time bins for time histogram
tsm = 2 # Time smoothing
imsm = 2 # Image smoothing
dim = 512 # Image dimension
cen = [253, 260] # Image centre
dslice = 0.16 # Width of slices

# Time points of slices
tn = np.linspace(4, 34, 7)
tn = tn + tn*np.linspace(0, nlf, 7)

# Prepare time histogram
t, y = histxy(pts[:,2], trange, tbins)

# Smooth histogram
y = gaussian_filter(y, tsm)

# Prepare slices:
pts[:,2] -= trange[0]
frms = np.zeros((7, dim, dim))
for i in range(0, 7):
    frmpts = pts[pts[:,2]>tn[i]]
    frmpts = frmpts[frmpts[:,2]<(tn[i]+dslice)]
    if len(frmpts) == 0: continue
    frm = pts2img(frmpts, dim, imsm)
    frms[i] = imutils.rotate(frm, ang)
    
# Start time at 0
t = t - trange[0]

# Setup figure
fig, ax = plt.subplots(2,4)
d = 140
for i in range(0, len(ax)):
    for j in range(0, len(ax[i])):
        ax[i,j].axes.get_xaxis().set_visible(False)
        ax[i,j].axes.get_yaxis().set_ticks([])
        if not [i,j]==[0,0]:
            ax[i,j].set_xlim([cen[0]-d, cen[0]+d])
            ax[i,j].set_ylim([cen[1]-d, cen[1]+d])
    
ax = [ax[0,0], ax[0,1], ax[0,2], ax[0,3],
      ax[1,0], ax[1,1], ax[1,2], ax[1,3]]

frms = np.array(frms)
globalvmax = np.percentile(frms, 99.9)
globalvmin = np.percentile(frms, 95)

# Plot Histogram
ax[0].plot(t, y, lw=3, color='black')
ax[0].vlines(tn, 0, max(y), lw=5, color='red')
ax[0].axes.get_xaxis().set_visible(True)
ax[0].tick_params(labelsize=20, width=2, length=6)
ax[0].set_title("Time (ns)", fontsize=30)
ax[0].set_xticks([0, tn[1], tn[3], tn[5], 40])
ax[0].set_aspect(1./ax[0].get_data_ratio())
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Plot Slices
for i in range(0, len(tn)):
    ax[i+1].imshow(frms[i], cmap='nipy_spectral_r', 
                 vmin=globalvmin, vmax=globalvmax)
    ax[i+1].set_title("{:.2f} ns".format(tn[i]), fontsize=30)
    
#%% PES

energy = 0.27

pesI = np.load(path+"PES/xe30PESinv.npy")
eVI = pesI[0]
IrI = pesI[1]
IrI = IrI/max(IrI)

pesXY = np.load(path+"PES/xe30PESsliceXY.npy")
eVXY = pesXY[0]
IrXY = pesXY[1]
IrXY = IrXY/max(IrXY)

d = 0.03
# Fitting
i1 = np.searchsorted(eVI, energy-d)
i2 = np.searchsorted(eVI, energy+d)
xsI = eVI[i1:i2]
ys = IrI[i1:i2]
fityI, params, pcov = fg(xsI, ys, 0, np.max(IrI), energy, 0.04, czero=True)
wI = params[2]*2.355

i1 = np.searchsorted(eVXY, energy-d)
i2 = np.searchsorted(eVXY, energy+d)
xsXY = eVXY[i1:i2]
ys = IrXY[i1:i2]
fityXY, params, pcov = fg(xsXY, ys, 0, np.max(IrXY), energy, 0.04, czero=True)
wXY = params[2]*2.355

plt.figure()
plt.plot(eVI, IrI, label="Inverted, {:2.1f}%".format(100*np.abs(wI)/energy), lw=3)
plt.plot(eVXY, IrXY, label="XY Slice, {:2.1f}%".format(100*np.abs(wXY)/energy), lw=3)
plt.plot(xsI, fityI, ls=":", lw=3)
plt.plot(xsXY, fityXY, ls=":", lw=3)
plt.xlabel("Energy (eV)", fontsize=20)
plt.ylabel("Intensity (Normalized)", fontsize=20)
plt.title("Photoelectron Spectrum, Xenon 31 ns", fontsize=25)
plt.xlim([0, 0.5])
plt.gca().tick_params(labelsize=20, width=2, length=6)

leg = plt.gca().legend(fontsize=20)
plt.gca().tick_params(labelsize=15)
leg.set_draggable(1)

print("[I] DE/E: {}".format(np.abs(wI)/energy))
print("[XY] DE/E: {}".format(np.abs(wXY)/energy))
