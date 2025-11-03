#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from VMI3D_IO import readwfm, genT

#%% Load

bk = np.load("DP_BKRPR_135554_bk.npy")
rpr = np.load("DP_BKRPR_135554_rpr.npy")

dt = 0.5 # 2 channel
dqdim = 4096
t = genT(dqdim, dt)

wfm = readwfm("DP_BKRPR_135554_wfm.uint16", dqdim, groupstks=False, ch2=True)

ti = 237 #101
trace = wfm[0][ti]
#plt.plot(trace)

blpts = 200
bk = bk - np.mean(bk[:blpts])

norm = max(rpr)
bk = bk / norm
rpr = rpr / norm
trace = -1*(trace-np.mean(trace[:500])) / norm

#%% Plot

bkscale = 80

fig = plt.figure()
ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 3), (0, 2))

FSt = 30
FSlb = 25
FSlg = 18
FStk = 18

ax1.plot(t[:800], np.roll(rpr, -900)[:800], lw=2, color='black', label="Average Reference Pulse")
ax1.plot(t[800:1200], trace[2000:2400], lw=2, ls='-', color='red', label="Typical Single Hit")
ax1.plot(t[1200:], bk[1200:], lw=2, color='blue', ls='-', label="Average Background")
ax1.plot(t[1200:], 0.5+bkscale*bk[1200:], lw=1, color='blue', ls='-', label="Average Background (x80)")
ax1.set_xlabel("Time (ns)", fontsize=FSlb)
ax1.set_ylabel("Amplitude (arb)", fontsize=FSlb)
ax1.minorticks_on()
ax1.tick_params(which='both', width=2, length=4)
ax1.tick_params(labelsize=FStk)
ax1.legend(loc=1, fontsize=FSlg)

ax2.plot(t-475, rpr, lw=3, color='black', label="Average Reference Pulse")
ax2.set_xlim([0, 100])
ax2.set_ylim([-0.2, 1.1])
ax2.set_xlabel("Time (ns)", fontsize=FSlb)
ax2.axes.get_yaxis().set_visible(False)
ax2.minorticks_on()
ax2.tick_params(which='both', width=2, length=4)
ax2.tick_params(labelsize=FStk)
ax2.legend(loc=1, fontsize=FSlg)

col1 = 'black'
col2 = 'white'
pos1 = 0.04
pos2 = 0.98
fslett = 20

t1 = ax1.text(pos1-0.025, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=fslett)
t2 = ax2.text(pos1-0.01, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)

t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

fig.set_tight_layout(True)
