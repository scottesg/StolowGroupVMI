#%% Imports

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from VMI3D_Functions import timebin, pts2img
import imutils

#%% Load data and trim

# Params
dim = 512
smth = 4
r = 180
slicewidth = 1.0 # ns
ang = -28 # rotation angle

ptXe40 = np.load("EX_Square.pt.npy")
t0Xe40 = 270
dtXe40 = 75
cenXe40 = [260, 252]
nsliceXe40 = int(dtXe40/slicewidth)

ptXe31 = np.load("EX_Xenon31ns_20220831_1823_T1.pt.npy")

titles = ["Xenon [40 ns]"]
pts = [ptXe40]
cens = [cenXe40]
t0s = [t0Xe40]
dts = [dtXe40]
nsl = [nsliceXe40]
xrs = []
yrs = []

for i in range(len(pts)):
    xrs.append([cens[i][0]-r, cens[i][0]+r])
    yrs.append([cens[i][1]-r, cens[i][1]+r])
    pts[i] = pts[i][pts[i][:,2].argsort()]
    pts[i][:,2] -= t0s[i]
    pts[i] = pts[i][np.searchsorted(pts[i][:,2], 0):
                    np.searchsorted(pts[i][:,2], dts[i]), :]

# make colormap
ncolors = 256
color_array = plt.get_cmap('gray_r')(range(ncolors))
color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
custmap = LinearSegmentedColormap.from_list(name='custom',colors=color_array)
        
#%% Slices

slices = []
for ds in range(len(pts)):
    ptt, bins = timebin(pts[ds], nsl[ds], 0, dts[ds])
    sl = np.zeros((len(ptt), 2*r, 2*r))
    for i in range(len(ptt)):
        if len(ptt[i]) == 0: continue
        sl[i] = pts2img(ptt[i], dim, 0)[xrs[ds][0]:xrs[ds][1], yrs[ds][0]:yrs[ds][1]]
        sl[i] = gaussian_filter(sl[i], smth)
        sl[i] = imutils.rotate(sl[i], ang)
    slices.append(sl[::-1])    

X = np.arange(0, 2*r, 1)
Y = np.arange(0, 2*r, 1)
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(X)

full2d = pts2img(pts[0], dim, 0)[xrs[ds][0]:xrs[ds][1], yrs[ds][0]:yrs[ds][1]]
full2d = gaussian_filter(full2d, 2)
full2d = imutils.rotate(full2d, ang)

full31 = pts2img(ptXe31, dim, 0)[xrs[ds][0]:xrs[ds][1], yrs[ds][0]:yrs[ds][1]]
full31 = gaussian_filter(full31, 2)
full31 = imutils.rotate(full31, ang)

#%% Plot

col1 = 'black'
fst = 15
fsl = 10

fig = plt.figure(figsize=(8,5))
ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2, projection="3d")
ax2 = plt.subplot2grid((1, 3), (0, 2))
axs = [ax1]

ax2.imshow(full2d, cmap='gray_r')
#ax2.imshow(full31, cmap='Blues_r', alpha=0.5)
ax2.axis('off')
ax2.vlines(174, 0, 55, color='red', ls='--', lw=2)
ax2.text(0.5, 1.0, "E", ha='left', va='top',
         color=col1, transform=ax2.transAxes, fontsize=fst)
ax2.hlines(178, 276, 350, color='red', ls='--', lw=2)
ax2.text(0.92, 0.48, "k", ha='left', va='top',
         color=col1, transform=ax2.transAxes, fontsize=fst)
ax2.text(0.05, 0.95, "(b)", ha='left', va='top',
         color=col1, transform=ax2.transAxes, fontsize=fst)

for ds in range(len(pts)):
    ax = axs[ds]
    sl = slices[ds]
    
    ax.axis('off')
    ax.view_init(elev=8, azim=112)
    ax.set_box_aspect((2,2,5))
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([None, nsliceXe40*slicewidth])
    ax.text2D(0.05, 0.95, '(a)', ha='left', va='top',
              color=col1, transform=ax.transAxes, fontsize=fst)
    
    vmax = np.percentile(sl, 99)
    vmin = np.percentile(sl, 10)
    
    for i in range(len(sl)):
        sli = sl[i]
        sli[sli<vmin] = vmin
        sli[sli>vmax] = vmax
        sli = sli - vmin
        sli = sli/(vmax-vmin)
        ax.plot_surface(X, Y, slicewidth*i+Z, facecolors=custmap(sli), linewidth=0)

fig.subplots_adjust(wspace=-0.4, hspace=0, left=0.1, right=0.9, top=1, bottom=0)
