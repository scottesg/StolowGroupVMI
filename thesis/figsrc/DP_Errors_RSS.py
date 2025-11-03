#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% Load data

datapath = "DP_Errors_RSS_3day_"
name = 'As'

ke = 2.05  # eV, energy of ring
E = ke/27.211
pmax = np.sqrt(2*E)

fitpol = np.poly1d(np.load(datapath + "calpoly{}.npy".format(name)))
pt = np.load(datapath + "data{}P.npy".format(name))
K = np.load(datapath + "K{}.npy".format(name))
A = np.load(datapath + "A.npy")[::-1,:]
resid = np.load(datapath + "resid.npz")
tofp, pzp, err = np.load(datapath + "caldata{}.npy".format(name))
peaks = np.load(datapath + "calpeaks{}.npy".format(name))

B = resid['B']
dtof = resid['dtof']
dX = resid['dX']
chi2 = resid['chi2']

dim = len(A)

#%% Prepare data

x = pt[:, 0]
y = pt[:, 1]
tof = pt[:, 2]

px = x*K
py = y*K
pz = fitpol(tof) # flip sign

ptrans = np.sqrt(px**2 + py**2)

H, PZ, PT = np.histogram2d(pz, ptrans, dim)
PZ = PZ[:-1]+(PZ[1]-PZ[0])/2
PT = PT[:-1]+(PT[1]-PT[0])/2

#%% Masks

# dPT is actually ptrans
HD, dPZ, dPT = np.histogram2d(fitpol.deriv()(tof), ptrans, dim)
dPZ = dPZ[:-1]+(dPZ[1]-dPZ[0])/2
dPT = dPT[:-1]+(dPT[1]-dPT[0])/2

DPt, DPz = np.meshgrid(dPT, dPZ)
Ptrans, Pz = np.meshgrid(PT, PZ)

# Sample mask
smask1 = np.tile(0, (dim, dim))
smask2 = np.copy(smask1)

sample_dt = 0.1 # ns
sample_dx = 1 # pixel

errterm = np.sqrt((Pz*DPz*sample_dt)**2 + (DPt*K*sample_dx)**2)
smask1[np.sqrt(Ptrans**2+Pz**2) < pmax + 1/pmax * errterm] = 1
smask2[np.sqrt(Ptrans**2+Pz**2) > pmax - 1/pmax * errterm] = 1

# Fit mask
mask1 = np.tile(0, (dim, dim))
mask2 = np.copy(mask1)

errterm = np.sqrt((Pz*DPz*dtof[B[0]])**2 + (DPt*K*dX[B[1]])**2)
mask1[np.sqrt(Ptrans**2+Pz**2) < pmax + 1/pmax * errterm] = 1
mask2[np.sqrt(Ptrans**2+Pz**2) > pmax - 1/pmax * errterm] = 1

Amask = A.copy()
Amask = np.ma.masked_where(Amask<0.5, Amask)

EmptyMask = np.ma.masked_where(Amask<5, Amask)

#%% Plots

pos1 = 0.03
pos2 = 0.96
letsize = 15
col1 = 'black'
col2 = 'white'

FSt = 25
FSlb = 20
FSlg = 15
FStk = 15

prmin = 0
prmax = 0.45
pzmin = -0.41
pzmax = 0.40

fig = plt.figure()
fig.set_tight_layout(True)

ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0))
ax4 = plt.subplot2grid((2, 2), (1, 1))

im1 = ax1.pcolor(PZ, PT, (gaussian_filter(H,5)/PT).T, cmap="Grays", vmax=8)
ax1.set_xticklabels([])
ax1.set_ylabel(r"P$_{\perp}$ (a.u.)", fontsize=FSlb)
ax1.set_ylim(prmin, prmax)
ax1.set_xlim(pzmin, pzmax)
eb1 = ax1.errorbar(pzp[2:-3], peaks[2:-3, 0], yerr=peaks[2:-3, 1], fmt="ro", markersize=0, lw=2, alpha=0.5,
                    label=r"Error Bars (1$\sigma$)")
ax1.tick_params(labelsize=FStk)
ax1.set_title("Measured Momentum Distribution", fontsize=FSt)
ax1.legend(fontsize=FSlg, loc='upper right')

cbaxes = inset_axes(ax1, width="80%", height="8%", loc='lower center') 
cbar = plt.colorbar(im1, cax=cbaxes, orientation='horizontal')
cbar.ax.xaxis.set_ticks_position('top')  

# cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.10, pad=-0.8, location='top')
# cbar1.ax.tick_params(labelsize=FSlg)

t1 = ax1.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=letsize)
t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

ax2.pcolor(PZ, PT, A.T, cmap="Blues")
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_ylim(prmin, prmax)
ax2.set_xlim(pzmin, pzmax)
ax2.tick_params(labelsize=FStk)
ax2.set_title("Experimental Error Mask", fontsize=FSt)

t2 = ax2.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=letsize)
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

ax3.pcolormesh(-1*Pz, Ptrans, smask1*smask2, cmap="hot")
ax3.contourf(Pz, Ptrans, Amask, cmap="Blues_r", alpha=0.8)
ax3.pcolor(Pz, Ptrans, EmptyMask, cmap="Blues_r", alpha=0.8, label="Experimental Error Mask")
ax3.set_ylabel(r"P$_{\perp}$ (a.u.)", fontsize=FSlb)
ax3.set_xlabel(r"P$_{z}$ (a.u.)", fontsize=FSlb)
ax3.set_ylim(prmin, prmax)
ax3.set_xlim(pzmin, pzmax)
ax3.tick_params(labelsize=FStk)
ax3.set_title(r"Sample Mask ($\sigma_t$ = 100 ps, $\sigma_r$ = 1 pixel)", fontsize=FSt)
ax3.legend(fontsize=FSlg, loc='upper right')

t3 = ax3.text(pos1, pos2, '(c)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=letsize)
t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

ax4.pcolormesh(-1*Pz, Ptrans, mask1*mask2, cmap="hot")
ax4.contourf(Pz, Ptrans, Amask, cmap="Blues_r", alpha=0.8)
ax4.pcolor(Pz, Ptrans, EmptyMask, cmap="Blues_r", alpha=0.8, label="Experimental Error Mask")
ax4.set_xlabel(r"P$_{z}$ (a.u.)", fontsize=FSlb)
ax4.set_yticklabels([])
ax4.set_ylim(prmin, prmax)
ax4.set_xlim(pzmin, pzmax)
ax4.tick_params(labelsize=FStk)
ax4.set_title(r"Fitted Mask ($\sigma_t$ = %.0f ps, $\sigma_r$ = %.1f pixels)"%(1000*dtof[B[0]], dX[B[1]]), fontsize=FSt)
ax4.legend(fontsize=FSlg, loc='upper right')

t4 = ax4.text(pos1, pos2, '(d)', ha='left', va='top', color=col1, transform=ax4.transAxes, fontsize=letsize)
t4.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

#%% Residual plot

# plt.figure()
# plt.pcolor(1000*dtof, dX, np.log(chi2).T, cmap="hot")
# plt.xlabel(r"$\sigma_t$ (ps)", fontsize=FSlb)
# plt.ylabel("$\sigma_r$ (pixels)", fontsize=FSlb)
# plt.axvline(x=1000*dtof[B[0]], color="white", ls="--", lw=2)
# plt.axhline(y=dX[B[1]], color="white", ls="--", lw=2)
# plt.text(1000*(dtof[B[0]]+0.004), dX[B[1]]-0.45, r"$\sigma_t$ = %.0f ps"%(1000*dtof[B[0]])+"\n"+"$\sigma_r$ = %.1f pixels"%(dX[B[1]]),
#          color="white", fontsize=FSlg*2, fontweight="black")
# plt.gca().tick_params(labelsize=FStk)

# cb = plt.colorbar()
# cb.set_label("log(RSS)", fontsize=FSlb)
# cb.ax.tick_params(labelsize=FStk)
