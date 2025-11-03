#%% Imports

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from VMI3D_Functions import (histxy, pts2img, timebin)
from VMI3D_Fitting import fit_gauss as fg
from matplotlib.ticker import FormatStrFormatter
import imutils
import matplotlib.cm as cm
import matplotlib.animation as animation

#%% Load data

pts = np.load("EX_Xenon31ns_20220831_1823_T1.pt.npy")
coords = plt.imread("EX_Xenon31ns_Axes.png")

# Params
ang = -25 # rotation angle
trange = np.array([243, 283])-0.2 # Time histogram range
nlf = 0.04 # nonlinear factor for time points
tbins = 2000 # Time bins for time histogram
tsm = 1 # Time smoothing
imsm = 1 # Image smoothing
dim = 512 # Image dimension
cen = [253, 260] # Image centre
dslice = 0.16 # Width of slices

#%% Slices for figure

# Time points of slices
tn = np.linspace(4, 34, 7)
tn = tn + tn*np.linspace(0, nlf, 7)

# Prepare time histogram
t, y = histxy(pts[:,2], trange, tbins)

# Smooth histogram
y = gaussian_filter(y, tsm)

# Prepare slices:
pts[:,2] -= trange[0]
frms = np.zeros((7, dim, dim))
for i in range(0, 7):
    frmpts = pts[pts[:,2]>tn[i]]
    frmpts = frmpts[frmpts[:,2]<(tn[i]+dslice)]
    if len(frmpts) == 0: continue
    frm = pts2img(frmpts, dim, imsm)
    frms[i] = imutils.rotate(frm, ang)
    
# Start time at 0
t = t - trange[0]

#%% Figure

FSt = 20
FStk = 15

col1 = 'black'
col2 = 'white'
pos1 = 0.04
pos2 = 0.96
fslett = 15
lett = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

cmap = 'nipy_spectral_r' 

pldim = (2,6)
csp = 2
fig = plt.figure(figsize=(16,7))
ax1 = plt.subplot2grid(pldim, (0, 0), colspan=csp, rowspan=2)
ax2 = plt.subplot2grid(pldim, (0, csp))
ax3 = plt.subplot2grid(pldim, (0, csp+1))
ax4 = plt.subplot2grid(pldim, (0, csp+2))
ax5 = plt.subplot2grid(pldim, (0, csp+3))
ax6 = plt.subplot2grid(pldim, (1, csp))
ax7 = plt.subplot2grid(pldim, (1, csp+1))
ax8 = plt.subplot2grid(pldim, (1, csp+2))
ax9 = plt.subplot2grid(pldim, (1, csp+3))
ax = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

ax1.imshow(coords)
ax1.axis('off')

d = 130 # Radius around centre to show
r = 80 # Radius of ring
for i in range(0, len(ax)):
    ax[i].axes.get_xaxis().set_visible(False)
    ax[i].axes.get_yaxis().set_ticks([])
    if i>0:
        ax[i].set_xlim([cen[0]-d, cen[0]+d])
        ax[i].set_ylim([cen[1]-d, cen[1]+d])
        # ax[i].vlines([cen[0]-r, cen[0]+r], cen[1]-d, cen[1]+d)
        # ax[i].hlines([cen[1]-r, cen[1]+r], cen[0]-d, cen[0]+d)
    
    # Letters
    txt = ax[i].text(pos1, pos2, '({})'.format(lett[i]), ha='left', va='top',
                     color=col1, transform=ax[i].transAxes, fontsize=fslett)
    txt.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
    

frms = np.array(frms)
globalvmax = np.percentile(frms, 99.9)
globalvmin = np.percentile(frms, 95)/globalvmax

frms = frms / globalvmax

# Plot Histogram
ax[0].plot(t, y, lw=3, color='black')
ax[0].vlines(tn, 0, 0.95*max(y), lw=3, color='red', ls='--')
ax[0].axes.get_xaxis().set_visible(True)
ax[0].tick_params(labelsize=FStk, width=2, length=4)
ax[0].set_title("Time (ns)", fontsize=FSt)
ax[0].set_xlim([-5, 42])
ax[0].set_xticks([tn[0], tn[3], tn[6]])
ax[0].xaxis.tick_top()
ax[0].set_aspect(1./ax[0].get_data_ratio())
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Plot Slices
for i in range(0, len(tn)):
    imcb = ax[i+1].imshow(frms[i], cmap=cmap, 
                 vmin=globalvmin, vmax=1)
    ax[i+1].set_title("{:.2f} ns".format(tn[i]), fontsize=FSt)
    
cbar5 = plt.colorbar(imcb, ax=ax1, fraction=0.056, pad=0.06)
cbar5.ax.tick_params(labelsize=FStk)
cbar5.ax.yaxis.set_ticks_position('left')

fig.set_tight_layout(True)

#%% Save movie

nbin = 250 # number of frames
trange = [2, 40]
smth = 0
framedelay = 80 # time between frames (ms)
trim = [0, 99.5] # show between these percentiles

tbinpts, bins = timebin(pts, nbin, trange[0], trange[1], discrete=False)

img = np.zeros((len(tbinpts), dim, dim))
for i in range(0, len(tbinpts)):
    #print("Frame: {}".format(i))
    if len(tbinpts[i]) == 0:
        continue
    img[i] = pts2img(tbinpts[i], dim, 0)

for i in range(0, len(img)):
    img[i] = gaussian_filter(img[i], smth)

frames = []
fig, ax = plt.subplots()
vmax = np.percentile(img, trim[1])
vmin = np.percentile(img, trim[0])

for i in range(0, len(img)):
    title = ax.text(0.2, 0.92, '{:1.2f} ns'.format(bins[i]-trange[0]), ha="center",color='white',
                     transform=ax.transAxes, size=20)#plt.rcParams["axes.titlesize"])
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frames.append([plt.imshow(imutils.rotate(img[i], ang), cmap=cm.grey, animated=True, vmin=vmin, vmax=vmax), title])

ani = animation.ArtistAnimation(fig, frames, interval=framedelay, blit=True)
plt.show()

#%% Save movie

Writer = animation.writers['ffmpeg']
writer = Writer(metadata=dict(artist='Me'))
ani.save("EX_Xe31ns_Movie.mp4", bitrate=20000)

#%% PES

# energy = 0.27

# pesI = np.load(path+"PES/xe30PESinv.npy")
# eVI = pesI[0]
# IrI = pesI[1]
# IrI = IrI/max(IrI)

# pesXY = np.load(path+"PES/xe30PESsliceXY.npy")
# eVXY = pesXY[0]
# IrXY = pesXY[1]
# IrXY = IrXY/max(IrXY)

# d = 0.03
# # Fitting
# i1 = np.searchsorted(eVI, energy-d)
# i2 = np.searchsorted(eVI, energy+d)
# xsI = eVI[i1:i2]
# ys = IrI[i1:i2]
# fityI, params, pcov = fg(xsI, ys, 0, np.max(IrI), energy, 0.04, czero=True)
# wI = params[2]*2.355

# i1 = np.searchsorted(eVXY, energy-d)
# i2 = np.searchsorted(eVXY, energy+d)
# xsXY = eVXY[i1:i2]
# ys = IrXY[i1:i2]
# fityXY, params, pcov = fg(xsXY, ys, 0, np.max(IrXY), energy, 0.04, czero=True)
# wXY = params[2]*2.355

# plt.figure()
# plt.plot(eVI, IrI, label="Inverted, {:2.1f}%".format(100*np.abs(wI)/energy), lw=3)
# plt.plot(eVXY, IrXY, label="XY Slice, {:2.1f}%".format(100*np.abs(wXY)/energy), lw=3)
# plt.plot(xsI, fityI, ls=":", lw=3)
# plt.plot(xsXY, fityXY, ls=":", lw=3)
# plt.xlabel("Energy (eV)", fontsize=20)
# plt.ylabel("Intensity (Normalized)", fontsize=20)
# plt.title("Photoelectron Spectrum, Xenon 31 ns", fontsize=25)
# plt.xlim([0, 0.5])
# plt.gca().tick_params(labelsize=20, width=2, length=6)

# leg = plt.gca().legend(fontsize=20)
# plt.gca().tick_params(labelsize=15)
# leg.set_draggable(1)

# print("[I] DE/E: {}".format(np.abs(wI)/energy))
# print("[XY] DE/E: {}".format(np.abs(wXY)/energy))
