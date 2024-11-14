#%% Imports

import os
os.chdir("..")

import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from VMI3D_Functions import (genOffsets, genshad, xydetmap, timebin, fitcorr)
from VMI3D_Plotting import vmi3dplots, pts2img
from VMI3D_Load3D import load3D

#%% Path and parameters

# Path should indicate directory with wfm and ctr data
# as indicated by the paths below
path = r"C:\Users\Scott\Python\VMI\data\20240125/144323/"

name = "T1" # base name and directory for output
outpath = path + name + "/" # full path for output

stksize = 64000 # number of images per data set (file)
dim = 512 # resolution of images (square)
wdsnr = 500 # signal-to-noise ratio for deconvolutions

dt = 0.5 # time step for digitizer traces in ns

# Raw data paths (* indicates multiple numbered files)
ctrpath = 'ctr/ctrs*.mat'

# Output paths
hitspath = 'hits.npy' # Extraced hits from pickoff traces
offsetspath = 'offsets.npy' # shot offsets between camera and digitizer stacks

# Path for ffmpeg (for movies)
# plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

# Create the output directory
if os.path.exists(path + name):
    print("Warning: Directory already exists!")
else:
    os.mkdir(path + name)
    
#%% Combine hit files

def ahits2hits(ahits, stknum):
    hts = []
    
    for i in range(len(ahits)):
        nhits = ahits[i,0]
        if np.isnan(nhits): continue
        if nhits>9: continue
        for j in range(0, int(nhits)):
            pos = ahits[i, 2+j*2]
            amp = ahits[i, 3+j*2]
            hit = [pos, amp, 0, 0, stknum, i, nhits]
            hts.append(hit)
    return np.array(hts)

n = 5 # number of stacks to combine

hits = np.zeros((0, 7))
for i in range(1, n+1):
    print("Stack #{}".format(i))
    ahits = np.loadtxt(path + "hits/THits_{}.txt".format(i))
    h = ahits2hits(ahits, i-1)
    hits = np.vstack((hits,h))

if os.path.exists(path + hitspath):
    print("Warning: File already exists!")
else:
    np.save(path + hitspath, hits)

#%% Generate camera/pickoff offsets

plot = True # for debugging

if not os.path.exists(path + offsetspath):
    cpath = path + ctrpath
    hpath = path + hitspath
    offsets = genOffsets(cpath, hpath, stksize, wdsnr, plot=plot)
    np.save(path + offsetspath, offsets)
else:
    offsets = np.load(path + offsetspath)
    
plt.figure()
plt.plot(offsets[0], label="Offset")
plt.plot(offsets[1]/max(offsets[1]), label="Relative Amplitude")
plt.xlabel("Stack Number")
leg = plt.gca().legend(fontsize=20)
leg.set_draggable(1)

#%% Load3D first run - single hits only (to produce correlation)
# Points structure looks like: [x, y, t, cameraheight, pickoffheight, stackindex, shotindex]

points = load3D(path+ctrpath, path+hitspath, path+offsetspath, outpath,
                stksize, nctrmax=1, matchcountnumber=True, name=name)

vmi3dplots(points, 512, smooth=0, cbins=500, phdbins=200,
           save=path+name, name="(Uncorrected)")

#%% Detector variation correction

gsize = 30 # dimension of grid used for correction
tilelim = 6 # minimum number of hits per tile

scaledpts, detmap, nmap = xydetmap(points, gsize, tilelim, xrange=[0,dim], yrange=[0,dim])

vmi3dplots(scaledpts, 512, smooth=0, cbins=500, phdbins=200,
            save=path+name, name="(Corrected)")

#%% Save detector variation correction

points = scaledpts
np.save(path + name + "/" + name + "_all.pt.npy", points)
np.save(path + name + "/detmap.npy", detmap)

#%% Single-Hit movie

nbin = 80 # number of frames
trange = [2080, 2100]#[1040, 1070] # ns
smth = 4
framedelay = 200 # time between frames (ms)
trim = [90, 99.9] # show between these percentiles

tbinpts, bins = timebin(points, nbin, trange[0], trange[1], discrete=False)

img = np.zeros((len(tbinpts), dim, dim))
for i in range(0, len(tbinpts)):
    print("Frame: {}".format(i))
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
    title = ax.text(0.5, 0.95, '{:1.2f} ns'.format(bins[i]), ha="center",color='white',
                     transform=ax.transAxes, size=plt.rcParams["axes.titlesize"])
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frames.append([plt.imshow(img[i], cmap=cm.plasma, animated=True, vmin=vmin, vmax=vmax), title])

ani = animation.ArtistAnimation(fig, frames, interval=framedelay, blit=True)
plt.show()

#%% Save movie

Writer = animation.writers['ffmpeg']
writer = Writer(metadata=dict(artist='Me'))
ani.save(path + name + '/VMI3Dmovie_singlehits_{}.mp4'.format(name))

#%% Obtain SHAD and correlation slope/y-int

corrdivs = 8 # number of divisions along the correlation line to fit widths
fscale = 0.001 # for linear fit of correlation
nbinshad = 400 # number of bins for 2D histogram of SHAD 
smshad = 8 # smoothing applied to SHAD
plot = True

corfit = fitcorr(points, path+name, corrdivs, fscale=fscale, plot=plot)
genshad(points, path+name, nbinshad, smooth=smshad, plot=plot)

#%% Load3D with correlation
# Points structure looks like: [x, y, t, cameraheight, pickoffheight, ctrsinshot, hitsinshot, stackindex, shotindex]

# setting ACT to 0 will process pairs with ~0 probability (takes longer), use 1e-20 instead
act = 1e-20 # absolute correlation threshold, <0 turns off multihit treatment
rct = -1 # relative correlation threshold, <0 takes best assignment with no comparison
nprobreturn = 1 # number of best assignments to compare, must be >=1
useshad = True # use the full 2D SHAD to compare assignments [false uses just the fit line]
nctrmax = 20 # maximum number of hits per shot
maxstks = -1 # only analyze this many stacks
save = True

debug = False # print cscorr log [limit data volume if using]

if debug:
    f = open('output.txt','w')
    s = sys.stdout
    sys.stdout = f
    save = False

points = load3D(path+ctrpath, path+hitspath, path+offsetspath, outpath,
                stksize, nctrmax=nctrmax, matchcountnumber=False,
                act=act, rct=rct, useshad=useshad, nprobreturn=nprobreturn,
                save=save, name=name, maxstks=maxstks, v=True, debug=debug)

if debug:
    sys.stdout = s
    f.close()
else:
    np.save(path + name + "/" + name + "_all_cor.pt.npy", points)

vmi3dplots(points, 512, smooth=0, cbins=500, phdbins=200,
            save=path+name, name="(Correlated)")

#%% Shift and rotate data for movie

# centre and rotation angle
cen = [248.5, 257.4]
rotangle = 110*np.pi/180

points[:,0] -= cen[0]
points[:,1] -= cen[1]

x2 = points[:,0]*np.cos(rotangle) + points[:,1]*np.sin(rotangle)
y2 = points[:,1]*np.cos(rotangle) - points[:,0]*np.sin(rotangle)

points[:,0] = x2
points[:,1] = y2

points[:,0] += 256
points[:,1] += 256

#%% Time-bin points and prepare images for movie

nbin = 200 # number of frames
trange = [2080, 2100]#[1040, 1070]
smth = 1
framedelay = 100 # time between frames (ms)
trim = [0, 99.99] # show between these percentiles

tbinpts, bins = timebin(points, nbin, trange[0], trange[1], discrete=False)

img = np.zeros((len(tbinpts), dim, dim))
for i in range(0, len(tbinpts)):
    print("Frame: {}".format(i))
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
    title = ax.text(0.2, 0.92, '{:1.2f} ns {}'.format(bins[i]-trange[0], i), ha="center",color='black',
                     transform=ax.transAxes, size=20)#plt.rcParams["axes.titlesize"])
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frames.append([plt.imshow(img[i], cmap=cm.nipy_spectral_r, animated=True, vmin=vmin, vmax=vmax), title])

ani = animation.ArtistAnimation(fig, frames, interval=framedelay, blit=True)
plt.show()

#%% Save movie

Writer = animation.writers['ffmpeg']
writer = Writer(metadata=dict(artist='Me'))
ani.save(path + name + '/VMI3Dmovie_full_{}.mp4'.format(name), bitrate=20000)
