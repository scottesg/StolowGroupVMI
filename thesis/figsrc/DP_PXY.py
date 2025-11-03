#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from VMI3D_Fitting import fit_gauss as fg

#%% Processing

pt = np.load("DP_PXY_NO213266_ALL_dataSsP.npy")

ke = 0.88  # eV, energy of ring
dslicexy = 0.2 # width of centre time slice
T0 = 29.5 # ns, put centre here
subpix = 2
fitwidth = 5*subpix

rs = np.sqrt(pt[:,0]**2 + pt[:,1]**2)
c1 = pt[:,2] > -(dslicexy/2) + T0
c2 = pt[:,2] < +(dslicexy/2) + T0
cond = c1 & c2
cenxy = pt[cond]
rc = rs[cond]

ri, bins = np.histogram(rc, 256*subpix, (0, 256), density=True)
d = bins[1] - bins[0]
rbins = bins[:-1] + d/2
ri /= rbins

rmax = np.where(ri[40*subpix:]==max(ri[40*subpix:]))[0][0]
rmax += 40*subpix

fitrs = rbins[rmax-fitwidth: rmax+fitwidth+1]
fitri, params, pcov = fg(fitrs, ri[rmax-fitwidth:rmax+fitwidth+1],
                          0, np.max(ri), rbins[rmax], 2, czero=True)

rmaxval = params[1]
E = ke/27.211
K = np.sqrt(2*E)/rmaxval

norm = max(ri[40*subpix:])

#%% Plot

fstitle = 20
fslabel = 15
fstick = 12
fsleg = 12
cbshrink = 0.74

fig = plt.figure(figsize=(19,6.5))
fig.set_tight_layout(True)
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1))
ax3 = plt.subplot2grid((1, 3), (0, 2))

im1 = ax1.hist2d(pt[:,0], pt[:,1], bins=300, range=([-200,200], [-200,200]))
ax1.grid(ls='--')
ax1.set_xlabel("X Position (Pixels)", fontsize=fslabel)
ax1.set_ylabel("Y Position (Pixels)", fontsize=fslabel)
ax1.set_title("Integrated VMI Image", fontsize=fstitle)

cbar1 = plt.colorbar(im1[3], ax=ax1, fraction=0.10, pad=0.02, shrink=cbshrink)
cbar1.ax.tick_params(labelsize=fsleg)

im2 = ax2.hist2d(cenxy[:,0], cenxy[:,1], bins=200, range=([-200,200], [-200,200]))
ax2.grid(ls='--')
ax2.set_xlabel("X Position (Pixels)", fontsize=fslabel)
ax2.set_ylabel("Y Position (Pixels)", fontsize=fslabel)
ax2.set_title("Centre XY Slice", fontsize=fstitle)

cbar2 = plt.colorbar(im2[3], ax=ax2, fraction=0.10, pad=0.02, shrink=cbshrink)
cbar2.ax.tick_params(labelsize=fsleg)

ax3.plot(rbins, ri/norm, color='black', lw=3, label="Data")
ax3.plot(fitrs, fitri/norm, color='red', lw=2, ls='--', marker='x', markersize=12,
         label="Fit " + "($r_{max}$ =" + "\n" + "        {:1.2f} pixels)".format(rmaxval))
ax3.grid()
ax3.set_xlabel("Radius (Pixels)", fontsize=fslabel)
ax3.set_ylabel("Amplitude (arb)", fontsize=fslabel)
ax3.set_title("Radial Yield (Centre Slice)", fontsize=fstitle)
ax3.set_xlim(rmaxval-6, rmaxval+6)
ax3.set_ylim(-0.1, 1.1)

ax3.legend(fontsize=fsleg)

ax1.set_aspect(1./ax1.get_data_ratio())
ax2.set_aspect(1./ax2.get_data_ratio())
ax3.set_aspect(1./ax3.get_data_ratio())

ax1.tick_params(labelsize=fstick)
ax2.tick_params(labelsize=fstick)
ax3.tick_params(labelsize=fstick)

# Letters

col1 = 'black'
col2 = 'white'
pos1 = 0.03
pos2 = 0.97
fslett = 15

t1 = ax1.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=fslett)
t2 = ax2.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)
t3 = ax3.text(pos1, pos2, '(c)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=fslett)

t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
