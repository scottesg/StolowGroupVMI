#%% Imports

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from VMI3D_IO import readwfm, genT
from VMI3D_Functions import wiener_deconvolution, genBK, genPR

#%% Trace Deconvolution

tis = 8
tim = 218
dqdim = 2048
dt = 0.25
blpts = 200
snr = 500
sgn = -1
gsm = 1.7 # sigma

wfmpath = "DP_Decon_wfmstk.uint16"
traces = readwfm(wfmpath, dqdim)

t = genT(dqdim, dt)
bk = genBK(wfmpath, 200, blpts)
pr = genPR(wfmpath, dqdim, bk, sgn, 600, 4000, 12000, 2, 0,
             None, None, None, None, 20, shreturn=True)[0]

trs = sgn*(traces[tis]-bk)
dcs = wiener_deconvolution(trs, pr, snr, relativeposition=True)
dcs = gaussian_filter(dcs, gsm)

trm = sgn*(traces[tim]-bk)
dcm = wiener_deconvolution(trm, pr, snr, relativeposition=True)
dcm = gaussian_filter(dcm, gsm)

trs = trs/max(trs)
trm = trm/max(trm)
dcs = dcs/max(dcs)
dcm = dcm/max(dcm)
resplt = pr/max(pr)

#%% Plot

t0 = 248
t1 = 38
t2 = 23

plotGSR = False

fig = plt.figure()

if plotGSR:
    ax1 = plt.subplot2grid((1, 3), (0, 0))
    ax2 = plt.subplot2grid((1, 3), (0, 1))
    ax3 = plt.subplot2grid((1, 3), (0, 2))
else:
    ax2 = plt.subplot2grid((1, 2), (0, 0))
    ax3 = plt.subplot2grid((1, 2), (0, 1))

FSt = 25
FSlb = 20
FSlg = 15
FStk = 15

xlim = [-2, 30]
ylim = [-0.30, 1.1]

if plotGSR:
    pi1, pi2 = np.searchsorted(t, xlim)
    ax1.plot(t-t0, resplt, lw=4)
    ax1.set_title("Global Single-Hit Response", fontsize=FSt)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel("Time (ns)", fontsize=FSlb)
    ax1.set_ylabel("Amplitude (arb)", fontsize=FSlb)
    ax1.minorticks_on()
    ax1.tick_params(which='both', width=2, length=4)
    ax1.tick_params(labelsize=FStk)
    ax2.axes.get_yaxis().set_visible(False)
else:
    ax2.set_ylabel("Amplitude (arb)", fontsize=FSlb)

ax2.plot(t-t0-t1, trs, label="Experimental Trace", lw=4)
ax2.plot(t-t0-t1, dcs-0.1, label="Deconvolution", ls="--", color="black", lw=2)
ax2.set_title("Typical Experimental Single-Hit Trace", fontsize=FSt)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_xlabel("Time (ns)", fontsize=FSlb)
ax2.minorticks_on()
ax2.tick_params(labelsize=FStk, which='both', width=2, length=4)
ax2.legend(fontsize=FSlg)

ax3.plot(t-t0-t2, trm, label="Experimental Trace", lw=4)
ax3.plot(t-t0-t2, dcm-0.1, label="Deconvolution", ls="--", color="black", lw=2)
ax3.set_title("Typical Experimental Multi-Hit Trace", fontsize=FSt)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_xlabel("Time (ns)", fontsize=FSlb)
ax3.axes.get_yaxis().set_visible(False)
ax3.minorticks_on()
ax3.tick_params(labelsize=FStk, which='both', width=2, length=4)
ax3.legend(fontsize=FSlg)

col1 = 'black'
col2 = 'white'
pos1 = 0.04
if not plotGSR: pos1 -= 0.01
pos2 = 0.98
fslett = 20

if plotGSR:
    t1 = ax1.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=fslett)
    t2 = ax2.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)
    t3 = ax3.text(pos1, pos2, '(c)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=fslett)
    t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
else:
    t2 = ax2.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)
    t3 = ax3.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=fslett)

t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

fig.set_tight_layout(True)
