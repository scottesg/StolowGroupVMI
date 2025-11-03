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
slicewidth = 0.5 # ns

ptXe9 = np.load("EX_Xenon9ns_20210923_T6.pt.npy")
angXe9 = -50
t0Xe9 = 270
dtXe9 = 15
cenXe9 = [247, 258]
nsliceXe9 = int(dtXe9/slicewidth)

ptXe31 = np.load("EX_Xenon31ns_20220831_1823_T1.pt.npy")
angXe31 = -50
t0Xe31 = 243
dtXe31 = 36
cenXe31 = [260, 253]
nsliceXe31 = int(dtXe31/slicewidth)

ptCH3I = np.load("EX_CH3I_20220831_2126_T1.pt.npy")
angCH3I = -50
t0CH3I = 197
dtCH3I = 13
cenCH3I = [260, 253]
nsliceCH3I = int(dtCH3I/slicewidth)

ptNOh = np.load("EX_NOh.pt.npy")
angNOh = -42
t0NOh = 66
dtNOh = 11
cenNOh = [247, 258]
nsliceNOh = int(dtNOh/slicewidth)

ptNO45 = np.load("EX_NO45.pt.npy")
angNO45 = -42
t0NO45 = 66
dtNO45 = 11
cenNO45 = [247, 258]
nsliceNO45 = int(dtNO45/slicewidth)

ptNOv = np.load("EX_NOv.pt.npy")
angNOv = -42
t0NOv = 66
dtNOv = 11
cenNOv = [247, 258]
nsliceNOv = int(dtNOv/slicewidth)

titles = ["Xenon [31 ns]", "Xenon [8.8 ns]", r"CH$_3$I [6.9 ns]",
         "NO, 90\N{DEGREE SIGN} [6.2 ns]", "NO, 45\N{DEGREE SIGN} [6.2 ns]", "NO, 0\N{DEGREE SIGN} [6.2 ns]"]
pts = [ptXe31, ptXe9, ptCH3I, ptNOh, ptNO45, ptNOv]
angs = [angXe31, angXe9, angCH3I, angNOh, angNO45, angNOv]
cens = [cenXe31, cenXe9, cenCH3I, cenNOh, cenNO45, cenNOv]
t0s = [t0Xe31, t0Xe9, t0CH3I, t0NOh, t0NO45, t0NOv]
dts = [dtXe31, dtXe9, dtCH3I, dtNOh, dtNO45, dtNOv]
nsl = [nsliceXe31, nsliceXe9, nsliceCH3I, nsliceNOh, nsliceNO45, nsliceNOv]
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
posl2a = 0.90
posl2b = 0.65
col1 = 'black'
fst = 16
fsl = 10
lett = ['a', 'b', 'c', 'd', 'e', 'f']

fig = plt.figure(figsize=(14,4))
ax1 = plt.subplot2grid((1, 6), (0, 0), projection="3d")
ax2 = plt.subplot2grid((1, 6), (0, 1), projection="3d")
ax3 = plt.subplot2grid((1, 6), (0, 2), projection="3d")
ax4 = plt.subplot2grid((1, 6), (0, 3), projection="3d")
ax5 = plt.subplot2grid((1, 6), (0, 4), projection="3d")
ax6 = plt.subplot2grid((1, 6), (0, 5), projection="3d")
axs = [ax1, ax2, ax3, ax4, ax5, ax6]

for ds in range(len(pts)):
    ax = axs[ds]
    sl = slices[ds]
    
    ofs = 0
    posl2 = posl2a
    if ds>0:
        ofs = 12
        posl2 = posl2b
    
    ax.axis('off')
    ax.view_init(elev=8, azim=112)
    ax.set_box_aspect((2,2,5))
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([None, nsliceXe31*0.5])
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
        ax.plot_surface(X, Y, slicewidth*i+ofs+Z, facecolors=custmap(sli), linewidth=0)

fig.subplots_adjust(wspace=-0.4, hspace=0, left=0, right=1.1, top=1, bottom=0)
