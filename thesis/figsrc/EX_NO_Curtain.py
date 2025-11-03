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
smth = 2
r = 180
slicewidth = 0.2 # ns

ptNO266 = np.load("EX_NO_Curtain_266_1CH_154004_154649_SsP.npy")
ptNO266[:,0] += 256
ptNO266[:,1] += 256
angNO266 = -23
t0NO266 = 25
dtNO266 = 10
cenNO266 = [256, 256]
nsliceNO266 = int(dtNO266/slicewidth)

ptNO213266 = np.load("EX_NO_Curtain_213266_ALL_SsP.npy")
ptNO213266[:,0] += 256
ptNO213266[:,1] += 256
angNO213266 = -23
t0NO213266 = 25
dtNO213266 = 10
cenNO213266 = [256, 256]
nsliceNO213266 = int(dtNO213266/slicewidth)

ptNO213 = np.load("EX_NO_Curtain_213_150711_151615_141739_SsP.npy")
ptNO213[:,0] += 256
ptNO213[:,1] += 256
angNO213 = -23
t0NO213 = 25
dtNO213 = 10
cenNO213 = [256, 256]
nsliceNO213 = int(dtNO213/slicewidth)

titles = ["266 nm only", "213 nm + 266 nm", "213 nm only"]
pts = [ptNO266, ptNO213266, ptNO213]
angs = [angNO266, angNO213266, angNO213]
cens = [cenNO266, cenNO213266, cenNO213]
t0s = [t0NO266, t0NO213266, t0NO213]
dts = [dtNO266, dtNO213266, dtNO213]
nsl = [nsliceNO266, nsliceNO213266, nsliceNO213]
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
        sl[i] = imutils.rotate(sl[i], angs[ds])
    slices.append(sl[::-1])    

X = np.arange(0, 2*r, 1)
Y = np.arange(0, 2*r, 1)
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(X)

#%% Plot

pos1 = 0.13
pos2 = 1.0
posl1 = 0.04
posl2 = 0.85
col1 = 'black'
fst = 12
fsl = 10
lett = ['a', 'b', 'c']

fig = plt.figure(figsize=(8,4.8))
ax1 = plt.subplot2grid((1, 3), (0, 0), projection="3d")
ax2 = plt.subplot2grid((1, 3), (0, 1), projection="3d")
ax3 = plt.subplot2grid((1, 3), (0, 2), projection="3d")
axs = [ax1, ax2, ax3]

for ds in range(len(pts)):
    ax = axs[ds]
    sl = slices[ds]
    
    ax.axis('off')
    ax.view_init(elev=8, azim=112)
    ax.set_box_aspect((2,2,5))
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([None, nsliceNO213266*slicewidth])
    ax.text2D(pos1, pos2, titles[ds], ha='left', va='top',
              color=col1, transform=ax.transAxes, fontsize=fst)
    ax.text2D(posl1, posl2, '({})'.format(lett[ds]), ha='left', va='top',
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

fig.subplots_adjust(wspace=-0.4, hspace=0, left=0, right=1.25, top=1, bottom=0)
