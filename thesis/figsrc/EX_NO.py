#%% Imports

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from VMI3D_Functions import (histxy, pts2img, timebin)
import imutils
from VMI3D_Fitting import fit_gauss as fg
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm
import matplotlib.animation as animation

#%% Load data
 
ptsh = np.load("EX_NOh.pt.npy")
pts45 = np.load("EX_NO45.pt.npy")
ptsv = np.load("EX_NOv.pt.npy")
coords = plt.imread("EX_NO_Axes.png")

# Params
ang = 68 # rotation angle
trange = np.array([66, 77]) # Time histogram range
nlf = 0.04 # nonlinear factor for time points
tbins = 1000 # Time bins for time histogram
tsm = 1 # Time smoothing
imsm = 1 # Image smoothing
dim = 512 # Image dimension
cen = [258, 247] # Image centre
dslice = 0.16 # Width of slices

#%% Slices for figure

# Time points of slices
tn = np.linspace(3, 8, 5)
tn = tn + tn*np.linspace(0, nlf, 5)

# Prepare time histograms
th, yh = histxy(ptsh[:,2], trange, tbins)
tv, yv = histxy(ptsv[:,2], trange, tbins)
t45, y45 = histxy(pts45[:,2], trange, tbins)

# Smooth histograms
yh = gaussian_filter(yh, tsm)
yv = gaussian_filter(yv, tsm)
y45 = gaussian_filter(y45, tsm)
ys = [yh, y45, yv]

# Prepare slices:
ptsh[:,2] -= trange[0]
ptsv[:,2] -= trange[0]
pts45[:,2] -= trange[0]
pts = [ptsh, pts45, ptsv]
    
# Prepare slices:
frmsh = np.zeros((5, dim, dim))
frmsv = np.zeros((5, dim, dim))
frms45 = np.zeros((5, dim, dim))
frms = [frmsh, frms45, frmsv]

for i in range(0, 3):
    for j in range(0, 5):
        frmpts = pts[i][pts[i][:,2]>tn[j]]
        frmpts = frmpts[frmpts[:,2]<(tn[j]+dslice)]
        if len(frmpts) == 0: continue
        frm = pts2img(frmpts, dim, imsm)
        frms[i][j] = imutils.rotate(frm, ang)
    
# Start time at 0
t = th - trange[0]

#%% Figure

angles = [90, 45, 0]
frmsh = np.array(frmsh)
frms45 = np.array(frms45)
frmsv = np.array(frmsv)

FSt = 20
FStk = 12

col1 = 'black'
col2 = 'white'
pos1 = 0.04
pos2 = 0.96
fslett = 12
lett = ['a', 'b', 'c', 'd', 'e', 'f',
        'g', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'q', 'r']

cmap = 'nipy_spectral_r' 

pldim = (3,8)
csp = 2
fig = plt.figure(figsize=(14,6))
ax1 = plt.subplot2grid(pldim, (0, 0), colspan=csp, rowspan=3)

ax2 = plt.subplot2grid(pldim, (0, csp))
ax3 = plt.subplot2grid(pldim, (0, csp+1))
ax4 = plt.subplot2grid(pldim, (0, csp+2))
ax5 = plt.subplot2grid(pldim, (0, csp+3))
ax6 = plt.subplot2grid(pldim, (0, csp+4))
ax7 = plt.subplot2grid(pldim, (0, csp+5))

ax8 = plt.subplot2grid(pldim, (1, csp))
ax9 = plt.subplot2grid(pldim, (1, csp+1))
ax10 = plt.subplot2grid(pldim, (1, csp+2))
ax11 = plt.subplot2grid(pldim, (1, csp+3))
ax12 = plt.subplot2grid(pldim, (1, csp+4))
ax13 = plt.subplot2grid(pldim, (1, csp+5))

ax14 = plt.subplot2grid(pldim, (2, csp))
ax15 = plt.subplot2grid(pldim, (2, csp+1))
ax16 = plt.subplot2grid(pldim, (2, csp+2))
ax17 = plt.subplot2grid(pldim, (2, csp+3))
ax18 = plt.subplot2grid(pldim, (2, csp+4))
ax19 = plt.subplot2grid(pldim, (2, csp+5))

ax = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11,
      ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19]

ax1.imshow(coords)
ax1.axis('off')

d = 140 # Radius around centre to show
r = 80 # Radius of ring
for i in range(0, len(ax)):
    ax[i].axes.get_xaxis().set_visible(False)
    ax[i].axes.get_yaxis().set_ticks([])
    if not i in (0, 6, 12):
        ax[i].set_xlim([cen[0]-d, cen[0]+d])
        ax[i].set_ylim([cen[1]-d, cen[1]+d])
        # ax[i].vlines([cen[0]-r, cen[0]+r], cen[1]-d, cen[1]+d)
        # ax[i].hlines([cen[1]-r, cen[1]+r], cen[0]-d, cen[0]+d)
    
    # Letters
    txt = ax[i].text(pos1, pos2, '({})'.format(lett[i]), ha='left', va='top',
                     color=col1, transform=ax[i].transAxes, fontsize=fslett)
    #txt.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
    

gvmin = 95
gvmax = 99.9
gmins = np.zeros(3)
gmaxs = np.zeros(3)

for i in range(0, 3):
    gmaxs[i] = np.percentile(frms[i], gvmax)
    gmins[i] = np.percentile(frms[i], gvmin)
    frms[i] = (frms[i]-gmins[i]) / gmaxs[i]
    
# Histograms
for i in range(3):
    ind = [0, 6, 12][i]
    ax[ind].plot(t, ys[i], lw=3, color='black')
    ax[ind].set_ylabel("\u03B8 = {}\N{DEGREE SIGN}".format(angles[i]), fontsize=FSt)
    ax[ind].vlines(tn, 0, 0.94*max(ys[i]), lw=2, color='red', ls='--')
    ax[ind].set_xticks([tn[0], tn[2], tn[4]])
    ax[ind].tick_params(labelsize=FStk, width=2, length=4)
    ax[ind].set_aspect(1./ax[ind].get_data_ratio())
    ax[ind].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].set_title("Time (ns)", fontsize=FSt)
ax[12].axes.get_xaxis().set_visible(True)

# Plot Slices
for i in range(3):
    for j in range(0, 5):
        ind = [[1, 2, 3, 4, 5],
               [7, 8, 9, 10, 11],
               [13, 14, 15, 16, 17]][i][j]
        imcb = ax[ind].imshow(frms[i][j], cmap=cmap, 
                     vmin=0, vmax=1)
        if i == 0: ax[ind].set_title("{:.2f} ns".format(tn[j]), fontsize=FSt)
        
cbar5 = plt.colorbar(imcb, ax=ax1, fraction=0.20, pad=0.06)
cbar5.ax.tick_params(labelsize=FStk)
cbar5.ax.yaxis.set_ticks_position('left')        

fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=0.99, top=0.92, bottom=0.05)

#%% Save movie

nbin = 55 # number of frames
trange = [2, 10]
smth = 0
framedelay = 80 # time between frames (ms)
trim = [0, 99.5] # show between these percentiles

tbinpts, bins = timebin(pts[2], nbin, trange[0], trange[1], discrete=False)

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
ani.save("EX_NO6_Movie.mp4", bitrate=20000)
