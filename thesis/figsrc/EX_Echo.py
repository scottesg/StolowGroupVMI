#%% Imports

import numpy as np
import matplotlib.pyplot as plt

#%% Load data

nosub = np.load("EX_Echo_nosub.npz")
Hns = nosub['H']
t = nosub['t']
r = nosub['r']

sub = np.load("EX_Echo.npz")
Hs = sub['H']

#%% Plot

FSlb = 15
FStk = 12

col1 = 'black'
col2 = 'white'
pos1 = 0.02
pos2 = 0.98
fslett = 15

pldim = (1, 2)
fig = plt.figure(figsize=(12,6))
ax1 = plt.subplot2grid(pldim, (0, 0))
ax2 = plt.subplot2grid(pldim, (0, 1))

im1 = ax1.pcolor(t, r, Hns, vmax=10, cmap="nipy_spectral_r")
ax1.set_xlabel("Time (ns)", fontsize=FSlb)
ax1.set_ylabel("Radius (pixels)", fontsize=FSlb)
ax1.tick_params(labelsize=FStk, width=2, length=4)
ax1.set_xlim([22.5, 38.5])
ax1.set_ylim([0, 120])
cbar1 = fig.colorbar(im1, ax=ax1, pad=0.02)
cbar1.ax.tick_params(labelsize=FStk)
t1 = ax1.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=fslett)
t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

im2 = ax2.pcolor(t, r, Hs, vmax=10, cmap="nipy_spectral_r")
ax2.set_xlabel("Time (ns)", fontsize=FSlb)
ax2.axes.get_yaxis().set_visible(False)
ax2.tick_params(labelsize=FStk, width=2, length=4)
ax2.set_xlim([22.5, 38.5])
ax2.set_ylim([0, 120])
cbar2 = fig.colorbar(im2, ax=ax2, pad=0.02)
cbar2.ax.tick_params(labelsize=FStk)
t2 = ax2.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

fig.set_tight_layout(True)