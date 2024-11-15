#%% Imports

import os
os.chdir(r'../..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from VMI3D_Fitting import fit_gauss
from scipy.io import loadmat

#%% Figure 6: Spatial Resolution

path = r"C:\Users\Scott\Python\VMI\data\20221003_VMI3DPaperData/"
data = loadmat(path + "lineoutfig.mat")

image = data['a1s']
x1 = data['tx1']
x2 = data['tx2']
y1 = data['ty1']
y2 = data['ty2']

scale = max(y2)

x1p = 215
x2p = 240
y1p = 178
y2p = 158
zoomregion = 100
xr1 = x1p - zoomregion
xr2 = x2p + zoomregion
yr1 = y2p - zoomregion
yr2 = y1p + zoomregion
r = 15
circ1 = Circle((x1p, y1p), r, lw=2, ls='-', fill=False, color='blue', zorder=2)
circ2 = Circle((x2p, y2p), r, lw=2, ls='-', fill=False, color='orange', zorder=2)

fig = plt.figure()
ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
ax4a = plt.subplot2grid((2, 4), (1, 2))
ax4b = plt.subplot2grid((2, 4), (1, 3))

ax1.imshow(image, cmap='gray_r')
ax1.vlines(x1p, 0, y1p-r, color='blue', lw=2, label="Hit 1")
ax1.vlines(x1p, y1p+r, 512, color='blue', lw=2)
ax1.vlines(x2p, 0, y2p-r, color='orange', lw=2, label = "Hit 2")
ax1.vlines(x2p, y2p+r, 512, color='orange', lw=2)
ax1.hlines(y1p, 0, x1p-r, color='blue', lw=2)
ax1.hlines(y1p, x1p+r, 512, color='blue', lw=2)
ax1.hlines(y2p, 0, x2p-r, color='orange', lw=2)
ax1.hlines(y2p, x2p+r, 512, color='orange', lw=2)
ax1.add_patch(circ1)
ax1.add_patch(circ2)
ax1.set_xticks([0, 512])
ax1.set_yticks([0, 512])
ax1.set_xlabel("X", fontsize=25)
ax1.xaxis.set_label_coords(0.5, -0.03)
ax1.set_ylabel("Y", fontsize=25)
ax1.yaxis.set_label_coords(-0.03, 0.5)
ax1.tick_params(labelsize=25, width=2, length=5)
leg = ax1.legend(fontsize=20)
ax1.set_xlim([xr1, xr2])
ax1.set_ylim([yr1, yr2])

ax2.plot(x2.T/scale, label='Hit 1', color='blue', lw=3)
ax2.plot(x1.T/scale, label='Hit 2', color='orange', lw=3)
leg = ax2.legend(fontsize=20)
ax2.tick_params(which='both', labelsize=25, width=2, length=5)
ax2.set_xticks(np.linspace(xr1, xr2, 6).astype(int))
ax2.set_xlabel("X Position (pixels)", fontsize=25)
ax2.set_ylabel("Normalized Amplitude", fontsize=25)
ax2.xaxis.set_label_coords(0.5, -0.12)
ax2.set_xlim([xr1, xr2])
ax2.text(xr1+15, 0.5, "X", fontsize=70)
ax2.minorticks_on()
ax2.tick_params(axis='y', which='minor', left=False)
#minor_locator = AutoMinorLocator(10)
#ax2.xaxis.set_minor_locator(minor_locator)
#ax2.set_aspect(1./ax2.get_data_ratio())

ax3.plot(y2/scale, label='Hit 1', color='blue', lw=3)
ax3.plot(y1/scale, label='Hit 2', color='orange', lw=3)
leg = ax3.legend(fontsize=20)
ax3.tick_params(which='both', labelsize=25, width=2, length=5)
ax3.set_xticks(np.linspace(yr1, yr2, 6).astype(int))
ax3.set_xlabel("Y Position (pixels)", fontsize=25)
ax3.set_ylabel("Normalized Amplitude", fontsize=25)
ax3.xaxis.set_label_coords(0.5, -0.12)
ax3.set_xlim([yr1, yr2])
ax3.text(yr1+15, 0.5, "Y", fontsize=70)
ax3.minorticks_on()
ax3.tick_params(axis='y', which='minor', left=False)
#ax3.xaxis.set_minor_locator(minor_locator)
#ax3.set_aspect(1./ax3.get_data_ratio())

d = 50
x2sub = (x2.T[x1p-d:x1p+d]/scale)[:,0]
xax = np.arange(0, len(x2sub))
fitx, params, pcov = fit_gauss(xax, x2sub, 0, 1, 50, 10)

y2sub = (y2[y1p-d:y1p+d]/scale)[:,0]
yax = np.arange(0, len(y2sub))
fity, params, pcov = fit_gauss(xax, y2sub, 0, 1, 50, 10)

ax4a.plot(xax+x1p-d, x2sub, label='Hit 1, X Lineout', color='blue', lw=3)
ax4a.plot(xax+x1p-d, fitx, label='Gaussian Fit', color='green', lw=0, marker='o')
ax4a.set_xticks(np.linspace(x1p-d, x1p+d, 4).astype(int))
ax4a.set_ylim(-0.1, 1.5)
#ax4.set_xlim(yr1, yr2)
ax4a.set_xlabel("X", fontsize=25)
#ax4.xaxis.set_label_coords(0.5, -0.15)
#ax4.minorticks_on()
#ax4.xaxis.set_minor_locator(minor_locator)
ax4a.tick_params(which='both', labelsize=25, width=2, length=5)
leg = ax4a.legend(fontsize=18)
ax4a.tick_params(axis='y', which='minor', left=False)
ax4a.set_aspect(1./ax4a.get_data_ratio())

ax4b.plot(yax+y1p-d, y2sub, label='Hit 1, Y Lineout', color='blue', lw=3)
ax4b.plot(yax+y1p-d, fity, label='Gaussian Fit', color='green', lw=0, marker='o')
ax4b.set_xticks(np.linspace(y1p-d, y1p+d, 4).astype(int))
ax4b.set_ylim(-0.1, 1.5)
#ax4.set_xlim(yr1, yr2)
ax4b.set_xlabel("Y", fontsize=25)
#ax4.xaxis.set_label_coords(0.5, -0.15)
#ax4.minorticks_on()
#ax4.xaxis.set_minor_locator(minor_locator)
ax4b.tick_params(which='both', labelsize=25, width=2, length=5)
leg = ax4b.legend(fontsize=18)
ax4b.tick_params(axis='y', which='minor', left=False)
#ax4b.axes.get_yaxis().set_visible(False)
ax4b.set_aspect(1./ax4b.get_data_ratio())