#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from VMI3D_Fitting import fit_gauss
from scipy.io import loadmat
from matplotlib.ticker import MultipleLocator

lw = 2
lw2 = 3
fslabel = 18
fsleg = 13
fsbig = 50
fstitle = 18

ticks = 15
tickw = 2
tickl = 5

#%% Figure 6: Spatial Resolution

data = loadmat("DP_ImageProcessing.mat")

image = data['a1s']
x1 = data['tx1']
x2 = data['tx2']
y1 = data['ty1']
y2 = data['ty2']

scale = max(y2)

cbshrink = 0.75

x1p = 215
x2p = 240
y1p = 178
y2p = 158
zoomregion = 100
xr1 = int(0.5*(x1p+x2p)) - zoomregion
xr2 = int(0.5*(x1p+x2p)) + zoomregion
yr1 = int(0.5*(y1p+y2p)) - zoomregion
yr2 = int(0.5*(y1p+y2p)) + zoomregion
r = 15
circ1 = Circle((x1p, y1p), r, lw=2, ls='-', fill=False, color='blue', zorder=2)
circ2 = Circle((x2p, y2p), r, lw=2, ls='--', fill=False, color='orange', zorder=2)

roi = image[y1p-15:y1p+16, x1p-15:x1p+16]

fig = plt.figure()
ax0 = plt.subplot2grid((2, 4), (0, 0))
ax1 = plt.subplot2grid((2, 4), (0, 1))
ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
ax4a = plt.subplot2grid((2, 4), (1, 2))
ax4b = plt.subplot2grid((2, 4), (1, 3))

im0 = ax0.imshow(roi/scale, cmap='gray_r')
ax0.set_title("Typical ROI (P-Hit #1)", fontsize=fstitle)
ax0.set_xticks([0, 30], [1, 31])
ax0.set_yticks([0, 30], [1, 31])
ax0.set_xlim([-0.5, 30.5])
ax0.set_ylim([-0.5, 30.5])
ax0.set_xlabel("X (pixels)", fontsize=fslabel)
ax0.xaxis.set_label_coords(0.5, -0.03)
ax0.set_ylabel("Y (pixels)", fontsize=fslabel)
ax0.yaxis.set_label_coords(-0.03, 0.5)
ax0.tick_params(labelsize=ticks, width=tickw, length=tickl)

cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.10, pad=0.04, shrink=cbshrink)
cbar0.ax.tick_params(labelsize=fsleg)

im1 = ax1.imshow(image/scale, cmap='gray_r')
ax1.set_title("Region Containing Two P-Hits", fontsize=fstitle)
ax1.vlines(x1p, 0, y1p-r, color='blue', lw=lw, label="p-hit #1")
ax1.vlines(x1p, y1p+r, 512, color='blue', lw=lw)
ax1.vlines(x2p, 0, y2p-r, color='orange', lw=lw, ls='--', label = "p-hit #2")
ax1.vlines(x2p, y2p+r, 512, color='orange', lw=lw, ls='--')
ax1.hlines(y1p, 0, x1p-r, color='blue', lw=lw)
ax1.hlines(y1p, x1p+r, 512, color='blue', lw=lw)
ax1.hlines(y2p, 0, x2p-r, color='orange', lw=lw, ls='--')
ax1.hlines(y2p, x2p+r, 512, color='orange', lw=lw, ls='--')
ax1.add_patch(circ1)
ax1.add_patch(circ2)
ax1.set_xticks([xr1, xr2-1], [1, xr2-xr1])
ax1.set_yticks([yr1, yr2-1], [1, yr2-yr1])
ax1.set_xlabel("X (pixels)", fontsize=fslabel)
ax1.xaxis.set_label_coords(0.5, -0.03)
ax1.set_ylabel("Y (pixels)", fontsize=fslabel)
ax1.yaxis.set_label_coords(-0.03, 0.5)
ax1.tick_params(labelsize=ticks, width=tickw, length=tickl)
leg = ax1.legend(fontsize=fsleg)
ax1.set_xlim([xr1-0.5, xr2-0.5])
ax1.set_ylim([yr1-0.5, yr2-0.5])

cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.10, pad=0.04, shrink=cbshrink)
cbar1.ax.tick_params(labelsize=fsleg)

xaxis = np.arange(1, 513)-xr1
ax2.plot(xaxis, x2.T/scale, label='p-hit #1', color='blue', lw=lw2)
ax2.plot(xaxis, x1.T/scale, label='p-hit #2', color='orange', lw=lw2, ls='--')
ax2.set_title("X-Lineouts", fontsize=fstitle)
leg = ax2.legend(fontsize=fsleg)
ax2.tick_params(which='both', labelsize=ticks, width=tickw, length=tickl)
ax2.set_xticks(np.linspace(1, xr2-xr1, 5).astype(int))
ax2.set_xlabel("X Position (pixels)", fontsize=fslabel)
ax2.set_ylabel("Normalized Amplitude", fontsize=fslabel)
ax2.xaxis.set_label_coords(0.5, -0.12)
ax2.set_xlim([1, xr2-xr1])
#ax2.text(xr1+15, 0.5, "X", fontsize=fsbig)
ax2.minorticks_on()
ax2.tick_params(axis='y', which='minor', left=False)
ax2.xaxis.set_minor_locator(MultipleLocator(5))
#ax2.set_aspect(1./ax2.get_data_ratio())

yaxis=np.arange(1, 513)-yr1
ax3.plot(yaxis, y2/scale, label='p-hit #1', color='blue', lw=lw2)
ax3.plot(yaxis, y1/scale, label='p-hit #2', color='orange', lw=lw2, ls='--')
ax3.set_title("Y-Lineouts", fontsize=fstitle)
leg = ax3.legend(fontsize=fsleg)
ax3.tick_params(which='both', labelsize=ticks, width=tickw, length=tickl)
ax3.set_xticks(np.linspace(1, yr2-yr1, 5).astype(int))
ax3.set_xlabel("Y Position (pixels)", fontsize=fslabel)
ax3.set_ylabel("Normalized Amplitude", fontsize=fslabel)
ax3.xaxis.set_label_coords(0.5, -0.12)
ax3.set_xlim([1, yr2-yr1])
#ax3.text(yr1+15, 0.5, "Y", fontsize=fsbig)
ax3.minorticks_on()
ax3.tick_params(axis='y', which='minor', left=False)
ax3.xaxis.set_minor_locator(MultipleLocator(5))
#ax3.set_aspect(1./ax3.get_data_ratio())

fwhmconv = 2.355
d = 26
x2sub = (x2.T[x1p-d:x1p+d]/scale)[:,0]
xax = np.arange(1, len(x2sub)+1)
fitx, paramsx, pcovx = fit_gauss(xax, x2sub, 0, 1, d/2, 10)
xp = paramsx[2]
dxp = np.sqrt(pcovx[2,2])
wx = fwhmconv*np.abs(paramsx[3])
dwx = fwhmconv*np.sqrt(pcovx[3,3])

y2sub = (y2[y1p-d:y1p+d]/scale)[:,0]
yax = np.arange(1, len(y2sub)+1)
fity, paramsy, pcovy = fit_gauss(xax, y2sub, 0, 1, d/2, 10)
yp = paramsy[2]
dyp = np.sqrt(pcovy[2,2])
wy = fwhmconv*np.abs(paramsy[3])
dwy = fwhmconv*np.sqrt(pcovy[3,3])

ax4a.plot(xax+x1p-d-xr1, x2sub, label='Lineout', color='blue', lw=lw2)
ax4a.plot(xax+x1p-d-xr1, fitx, label='Gaussian Fit' + '\n' r'Width = {:2.2f}$\pm${:2.2f} FWHM'.format(wx, dwx)
          #+ '\n' + r'x = {:2.2f}$\pm${:2.2f}'.format(x1p-xr1-d+xp, dxp)
          ,color='green', lw=0, marker='o')
ax4a.set_title("X-Lineout Fit (P-Hit #1)", fontsize=fstitle)
ax4a.set_xticks(np.linspace(x1p-xr1-d, x1p-xr1+d, 5).astype(int))
ax4a.set_ylim(-0.1, 1.35)
#ax4a.set_xlim(x1p-xr1-d, x1p-xr1+d)
ax4a.set_xlabel("X", fontsize=fslabel)
#ax4.xaxis.set_label_coords(0.5, -0.15)
#ax4.minorticks_on()
#ax4.xaxis.set_minor_locator(minor_locator)
ax4a.tick_params(which='both', labelsize=ticks, width=tickw, length=tickl)
leg = ax4a.legend(fontsize=fsleg-2, loc='upper right')
ax4a.tick_params(axis='y', which='minor', left=False)
ax4a.set_aspect(1./ax4a.get_data_ratio())

ax4b.plot(yax+y1p-d-yr1, y2sub, label='Lineout', color='blue', lw=lw2)
ax4b.plot(yax+y1p-d-yr1, fity, label='Gaussian Fit' + '\n' + r'Width = {:2.2f}$\pm${:2.2f} FWHM'.format(wy, dwy)
          #+ '\n' + r'y = {:2.2f}$\pm${:2.2f}'.format(y1p-yr1-d+yp, dyp)
          ,color='green', lw=0, marker='o')
ax4b.set_title("Y-Lineout Fit (P-Hit #1)", fontsize=fstitle)
ax4b.set_xticks(np.linspace(y1p-yr1-d, y1p-yr1+d, 5).astype(int))
ax4b.set_ylim(-0.1, 1.35)
#ax4.set_xlim(yr1, yr2)
ax4b.set_xlabel("Y", fontsize=fslabel)
#ax4.xaxis.set_label_coords(0.5, -0.15)
#ax4.minorticks_on()
#ax4.xaxis.set_minor_locator(minor_locator)
ax4b.tick_params(which='both', labelsize=ticks, width=tickw, length=tickl)
leg = ax4b.legend(fontsize=fsleg-2, loc='upper right')
ax4b.tick_params(axis='y', which='minor', left=False)
#ax4b.axes.get_yaxis().set_visible(False)
ax4b.set_aspect(1./ax4b.get_data_ratio())

col1 = 'black'
col2 = 'white'
pos1 = 0.04
pos2 = 0.96
fslett = 20

t0 = ax0.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax0.transAxes, fontsize=fslett)
t1 = ax1.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=fslett)
t2 = ax2.text(pos1-0.02, pos2, '(d)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)
t3 = ax3.text(pos1-0.02, pos2, '(c)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=fslett)
t4a = ax4a.text(pos1, pos2, '(e)', ha='left', va='top', color=col1, transform=ax4a.transAxes, fontsize=fslett)
t4b = ax4b.text(pos1, pos2, '(f)', ha='left', va='top', color=col1, transform=ax4b.transAxes, fontsize=fslett)

t0.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t4a.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t4b.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))