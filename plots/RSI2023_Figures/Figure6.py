#%% Imports

import os
os.chdir(r'../..')

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from VMI3D_IO import readwfm, genT
from VMI3D_Functions import wiener_deconvolution

#%% Figure 5: Trace Deconvolution and Single-Hit Response

path = r"C:\Users\Scott\Python\VMI\data\20221003_VMI3DPaperData/"
traces = readwfm(path + "wfmstk.uint16", 2048)
response = np.load(path + "pr.npy")
bk = np.load(path + "bk.npy")
t = genT(2048, 0.25) - 270
tis = 8
tim = 218

trs = -1*traces[tis]
trs = trs - bk
trs = trs - np.mean(trs[:200])
dcs = wiener_deconvolution(trs, response, 500, relativeposition=True)
dcs = gaussian_filter(dcs, 1.2)

trm = -1*traces[tim]
trm = trm - bk
trm = trm - np.mean(trm[:200])
dcm = wiener_deconvolution(trm, response, 500, relativeposition=True)
dcm = gaussian_filter(dcm, 1.2)

trs = trs/max(trs)
trm = trm/max(trm)
dcs = dcs/max(dcs)
dcm = dcm/max(dcm)
resplt = 0.8*response/max(response)

fig, ax = plt.subplots(1,3)
ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]
xlim = [-2, 30]
ylim = [-0.05, 1.5]

prp = 1080
prw = 170
ax1.plot(t[prp:prp+prw], resplt[980:1150]+0.1, lw=4)
ax1.set_title("Global Single-Hit Response", fontsize=22)
ax1.set_xlim([0, 30])
ax1.set_ylim([-0.25, 1])
ax1.set_xlabel("Time (ns)", fontsize=35)
ax1.axes.get_yaxis().set_visible(False)
ax1.minorticks_on()
ax1.tick_params(which='both', width=2, length=4)
ax1.tick_params(labelsize=25)

ax2.plot(t-15, trs+0.4, label="Raw", lw=4)
ax2.plot(t-15, dcs+0.1, label="Deconvolution", lw=4)
ax2.set_title("Typical Single-Hit Pickoff Trace", fontsize=22)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_xlabel("Time (ns)", fontsize=35)
ax2.axes.get_yaxis().set_visible(False)
ax2.minorticks_on()
ax2.tick_params(labelsize=25)
ax2.tick_params(which='both', width=2, length=4)
leg = ax2.legend(fontsize=16)
leg.set_draggable(True)

ax3.plot(t, trm+0.4, label="Raw", lw=4)
ax3.plot(t, dcm+0.1, label="Deconvolution", lw=4)
ax3.set_title("Typical Multi-Hit Pickoff Trace", fontsize=22)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_xlabel("Time (ns)", fontsize=35)
ax3.axes.get_yaxis().set_visible(False)
ax3.minorticks_on()
ax3.tick_params(labelsize=25)
ax3.tick_params(which='both', width=2, length=4)
leg = ax3.legend(fontsize=16)
leg.set_draggable(True)

#%% For Poster Gordon 2022

fig, ax3 = plt.subplots()

prp = 1080
pp = 980
prw = 110
ax3.plot(0.5*t[prp:prp+prw]+16, 0.5*resplt[pp:pp+prw]+0.6, label="Deconvoltion Kernel\n(Half Scale)", lw=4)
ax3.plot(t, trm+0.4, label="Raw Trace", lw=4)
ax3.plot(t, dcm+0.1, label="Deconvolution\n(Smoothed)", lw=4)
#ax3.set_title("Typical Three-Hit Pickoff Trace", fontsize=30)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_xlabel("Time (ns)", fontsize=35)
ax3.axes.get_yaxis().set_visible(False)
ax3.minorticks_on()
ax3.tick_params(labelsize=25)
ax3.tick_params(which='both', width=2, length=4)
leg = ax3.legend(fontsize=20)
leg.set_draggable(True)
ax3.set_aspect(1./ax3.get_data_ratio())
