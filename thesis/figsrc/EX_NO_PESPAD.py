#%% Imports

import numpy as np
import matplotlib.pyplot as plt

#%% Load data

pesI = np.load("EX_NO_PESPAD_noPESinv_Slice.npy")
pesI = np.reshape(pesI, (4, 497))
eVI = pesI[2]
IrI = pesI[3]
IrI = IrI/max(IrI[2:])

pesXY = np.load("EX_NO_PESPAD_noPESsliceXY.npy")
eVXY = pesXY[0]
IrXY = pesXY[1]
IrXY = IrXY/max(IrXY)

pesXZ = np.load("EX_NO_PESPAD_noPESsliceXZ.npy")
eVXZ = pesXZ[0]
IrXZ = pesXZ[1]
IrXZ = IrXZ/max(IrXZ)

padI = np.load("EX_NO_PESPAD_noPADinv_2.npy")
angI = padI[:,0]
intI = padI[:,1]
fitI = padI[:,2]
s = fitI[180]
h = fitI[0]
intI -= s
fitI -= s
intI /= h
fitI /= h

padXY = np.load("EX_NO_PESPAD_noPADsliceXY.npy")
angXY = padXY[:,0]
intXY = padXY[:,1]
fitXY = padXY[:,2]
s = fitXY[180]
h = fitXY[0]
intXY -= s
fitXY -= s
intXY /= h
fitXY /= h

padXZ = np.load("EX_NO_PESPAD_noPADsliceXZ.npy")
angXZ = padXZ[:,0]
intXZ = padXZ[:,1]
fitXZ = padXZ[:,2]
s = fitXZ[180]
h = fitXZ[0]
intXZ -= s
fitXZ -= s
intXZ /= h
fitXZ /= h

padYZ = np.load("EX_NO_PESPAD_noPADsliceYZ.npy")
angYZ = padYZ[:,0]
intYZ = padYZ[:,1]
fitYZ = padYZ[:,2]
s = fitYZ[180]
h = fitYZ[0]*2
intYZ -= s
fitYZ -= s
intYZ /= h
fitYZ /= h

#%% Plot

FStk = 12
FSlb = 15
FSlg = 12
beta = "\u03B2\u2082\u2080/\u03B2\u2080\u2080"

col1 = 'black'
col2 = 'white'
pos1 = 0.02
pos2 = 0.98
fslett = 15

fig = plt.figure(figsize=(14,8))
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

ax1.plot(eVI, IrI+0.3, label="Inverted", color="red", lw=3)
ax1.plot(eVXY, IrXY, label="XY Slice", color="blue", lw=3)
ax1.set_xlabel("Energy (eV)", fontsize=FSlb)
ax1.set_ylabel("Intensity (arb)", fontsize=FSlb)
ax1.set_xlim([0, 1.4])
ax1.set_ylim([-0.1, 1.4])
ax1.tick_params(labelsize=FStk, width=2, length=4)
ax1.legend(fontsize=FSlg)

ax2.plot(angI, intI+1.5, label="Inverted ["+beta+" = 1.67\u00B10.12]", color="red", marker='+', lw=0)
ax2.plot(angI, fitI+1.5, color="black", lw=3, ls='--')

ax2.plot(angXY, intXY+1, label="XY Slice ["+beta+" = 1.62\u00B10.06]", marker="x", color="blue", lw=0)
ax2.plot(angXY, fitXY+1, color="black", lw=3, ls='--')

ax2.plot(angXZ, intXZ+0.5, label="XZ Slice ["+beta+" = 1.68\u00B10.05]", marker="1", color="green", lw=0)
ax2.plot(angXZ, fitXZ+0.5, color="black", lw=3, ls='--')

ax2.plot(angYZ, intYZ, label="YZ Slice ["+beta+" = 0.13\u00B10.06]", marker="^", markerfacecolor='none', color="purple", lw=0)
ax2.plot(angYZ, fitYZ, color="black", lw=3, ls='--')

ax2.set_xlabel("Photoemission Angle (deg)", fontsize=FSlb)
ax2.set_ylabel("Intensity (arb)", fontsize=FSlb)
ax2.set_xlim([0, 180])
ax2.set_ylim([-0.5, 3.0])
ax2.tick_params(labelsize=FStk, width=2, length=4)
ax2.legend(fontsize=FSlg)

fig.subplots_adjust(wspace=0.25, hspace=0, left=0.08, right=0.95, top=0.95, bottom=0.1)

txt = ax1.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=fslett)
txt.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

txt = ax2.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)
txt.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

#%% PES

# energy = 0.84

# # Fitting
# d = 0.08
# i0a = np.searchsorted(eVI, energy-6*d)
# i0b = np.searchsorted(eVI, energy-2*d)
# bk = np.mean(IrI[i0a:i0b])
# IrI -= bk
# i1 = np.searchsorted(eVI, energy-d)
# i2 = np.searchsorted(eVI, energy+d)
# xsI = eVI[i1:i2]
# ys = IrI[i1:i2]
# fityI, params, pcov = fg(xsI, ys, 0, np.max(IrI[2:]), energy, 0.04, czero=True)
# wI = params[2]*2.355

# d = 0.06
# i0a = np.searchsorted(eVXY, energy-6*d)
# i0b = np.searchsorted(eVXY, energy-2*d)
# bk = np.mean(IrXY[i0a:i0b])
# IrXY -= bk
# i1 = np.searchsorted(eVXY, energy-d)
# i2 = np.searchsorted(eVXY, energy+d)
# xsXY = eVXY[i1:i2]
# ys = IrXY[i1:i2]
# fityXY, params, pcov = fg(xsXY, ys, 0, np.max(IrXY), energy, 0.04, czero=True)
# wXY = params[2]*2.355

# d = 0.12
# i0a = np.searchsorted(eVXZ, energy-6*d)
# i0b = np.searchsorted(eVXZ, energy-2*d)
# bk = np.mean(IrXZ[i0a:i0b])
# IrXZ -= bk
# i1 = np.searchsorted(eVXZ, energy-d)
# i2 = np.searchsorted(eVXZ, energy+d)
# xsXZ = eVXZ[i1:i2]
# ys = IrXZ[i1:i2]
# fityXZ, params, pcov = fg(xsXZ, ys, 0, np.max(IrXZ), energy, 0.04, czero=True)
# wXZ = params[2]*2.355

# plt.figure()
# plt.plot(eVI, IrI+0.3, label="Inverted", lw=3)
# plt.plot(eVXY, IrXY, label="XY Slice", lw=3)
# plt.plot(eVXZ, IrXZ+0.6, label="XZ Slice", lw=3)
# plt.plot(xsI, fityI+0.3, ls=":", lw=3)
# plt.plot(xsXY, fityXY, ls=":", lw=3)
# plt.plot(xsXZ, fityXZ+0.6, ls=":", lw=3)
# plt.xlabel("Energy (eV)", fontsize=20)
# plt.ylabel("Intensity (Normalized)", fontsize=20)
# plt.title("Photoelectron Spectrum, NO 6.2 ns", fontsize=25)
# plt.xlim([0, 1.5])
# plt.gca().tick_params(labelsize=20, width=2, length=6)

# leg = plt.gca().legend(fontsize=20)
# plt.gca().tick_params(labelsize=15)
# leg.set_draggable(1)

# print("[I] DE/E: {}".format(np.abs(wI)/energy))
# print("[XY] DE/E: {}".format(np.abs(wXY)/energy))
# print("[XZ] DE/E: {}".format(np.abs(wXZ)/energy))