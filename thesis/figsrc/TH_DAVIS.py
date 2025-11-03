#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from VMI3D_Functions import pts2img
import DAVIS

#%% Load Data

dim = 512
size = 1500
dtdeg = 1
dr = 0.5
amax = 180
cen = 256

pt = np.load("DP_PXY_NO213266_ALL_dataSsP.npy")
pt[:,:2] += 256
image = pts2img(pt, dim, 2)

imagep, cen = DAVIS.prepimage(image, rotangle=0, cen=[cen,cen], imsm=0)
rmax = len(imagep)
r = np.arange(0, rmax/2, dr)

LM, ms = DAVIS.loadmats(size, dtdeg, dr, amax, rmax)

#%% Invert

# DAVIS:
f_values, deltas, impol, impolinv, imout = DAVIS.transform(imagep, LM, ms, dtdeg, dr)

f0 = f_values[0]*r
b2 = f_values[1]/f_values[0]
b4 = f_values[2]/f_values[0]

#%% Plot

FSt = 15
FSlb = 15
FStk = 12
FSlg = 12
beta2 = r'$\beta_2$'
beta4 = r'$\beta_4$'
frac = 0.046

col1 = 'black'
col2 = 'white'
pos1 = 0.05
pos2 = 0.95
fslett = 14

pldim = (2, 3)
fig = plt.figure(figsize=(12,8))
ax1 = plt.subplot2grid(pldim, (0, 0))
ax2 = plt.subplot2grid(pldim, (0, 1))
ax3 = plt.subplot2grid(pldim, (0, 2))
ax4 = plt.subplot2grid(pldim, (1, 0), colspan=3)

im1 = ax1.imshow(image/np.max(image), cmap='nipy_spectral_r', vmin=0.00)
ax1.set_title("Original Image", fontsize=FSt)
ax1.set_xlim(0, dim)
ax1.set_ylim(0, dim)
ax1.set_xticks((0, dim))
ax1.set_yticks((0, dim))
ax1.tick_params(labelsize=FStk)

cbar1 = plt.colorbar(im1, ax=ax1, fraction=frac, pad=0.04)
cbar1.ax.tick_params(labelsize=FStk)

im2 = ax2.imshow(imagep/np.max(imagep), cmap='nipy_spectral_r', vmin=0.00)
ax2.set_title("Prepared Image", fontsize=FSt)
ax2.set_xlim(0, dim)
ax2.set_ylim(0, dim)
ax2.set_xticks((0, dim))
ax2.set_yticks((0, dim))
ax2.tick_params(labelsize=FStk)

cbar2 = plt.colorbar(im2, ax=ax2, fraction=frac, pad=0.04)
cbar2.ax.tick_params(labelsize=FStk)

im3 = ax3.imshow(imout/np.max(imout), cmap='nipy_spectral_r', vmin=0.00)
ax3.set_title("Inverted Image", fontsize=FSt)
ax3.set_xlim(0, dim)
ax3.set_ylim(0, dim)
ax3.set_xticks((0, dim))
ax3.set_yticks((0, dim))
ax3.tick_params(labelsize=FStk)

cbar3 = plt.colorbar(im3, ax=ax3, fraction=frac, pad=0.04)
cbar3.ax.tick_params(labelsize=FStk)

xmax = 140
ax4.plot(r, 2*f0/max(f0), color='black', lw=2, label="Spectrum")
ax4.plot(r, b2, color='blue', lw=2, ls='--', label=beta2)
ax4.plot(r, b4, color='green', lw=2, ls=':', label=beta4)
ax4.grid()
ax4.set_title("Radial Spectrum and Asymmetry Parameters", fontsize=FSt)
ax4.set_xlim([0, xmax])
ax4.set_ylim([-1, 2.2])
ax4.set_xlabel("Radius (pix)", fontsize=FSlb)
ax4.set_ylabel("Asymmetry Parameter", fontsize=FSlb)
ax4.tick_params(labelsize=FStk)
ax4.legend(fontsize=FSlg)

t1 = ax1.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=fslett)
t2 = ax2.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)
t3 = ax3.text(pos1, pos2, '(c)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=fslett)
t4 = ax4.text(pos1-0.04, pos2+0.01, '(d)', ha='left', va='top', color=col1, transform=ax4.transAxes, fontsize=fslett)

t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t4.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

fig.set_tight_layout(True)
