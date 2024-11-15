#%% Imports

from FunctionsForPlots import (getpeaks, pts2pr)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from VMI3D_IO import readwfm, genT
from VMI3D_Functions import (ccornorm, pts2img, xybin)

#%% Figure 4: Pickoff Time Response Uniformity
# Using 'points' output of VMI3D 1st run

nx = 20
ny = 20
nsamples = 50
dim = 512
bsi = 6 # xenon9ns: 18
cen = [253, 260]

# Load
path = r"C:\Users\Scott\Python\VMI\data\20221003_VMI3DPaperData/"
pointspath = path + "CH3I.pt.npy"
wfmpath = path + "CH3I_wfm/daqdata*.uint16"
bkpath = path + "CH3I_bk.npy"
points = np.load(pointspath)
wfms = readwfm(wfmpath, 2048, groupstks=True)
bk = np.load(bkpath)
tracelen = len(wfms[0][0])

# Rotate Points
rotangle = 32*(np.pi/180)
xc = points[:,0] - cen[0]
yc = points[:,1] - cen[1]
x2 = xc*np.cos(rotangle) + yc*np.sin(rotangle)
y2 = yc*np.cos(rotangle) - xc*np.sin(rotangle)
points[:,0] = x2 + cen[0]
points[:,1] = y2 + cen[0]

# Bin
xval = np.linspace(0, dim, nx+1)
yval = np.linspace(0, dim, ny+1)
binned = xybin(points, nx, ny, dim)

# Generate response for each bin
prs = []
for i in range(0, len(binned)):
    print("Bin Number: {}".format(i))
    if len(binned[i]>1):
        pr = pts2pr(binned[i], wfms, 200, 1500, ntrace=nsamples)
        pr = np.hstack((pr[500:], pr[:500]))
        prs.append(pr)
    else: prs.append(np.zeros(2048))

# Normalized average response
prav = np.mean(np.array(prs), 0)
prav = prav / max(prav)

# Artificial test traces
nohits = -1*wfms[0][bsi] - bk
trsingle = nohits + prav*5000
secondpeak = np.hstack((prav[40:], prav[:40]))
trdouble = nohits + (prav + secondpeak)*5000

# Time (ns)
t = genT(tracelen, 0.25)
ef = 1/np.linspace(1000, 1, 20)

# Comparison metrics
ccor = []
rms = []
shpos = []
shint = []
dhpos = []
dhint= []

#%%    
# Evaluate comparisons for each bin

for i in range(0, len(prs)):
    pri = prs[i]
    
    # If bin did not have enough samples
    if max(pri)==0:
        rms.append(0)
        ccor.append(0)
        shpos.append(0)
        shint.append(0)
        dhpos.append(0)
        dhint.append(0)
        continue
    
    # Normalize bin pr
    pri = pri / max(pri)
    
    # Cross correlation
    cc = ccornorm(pri, prav)
    ccor.append(max(cc))
    
    # RMS Difference
    diff = np.sqrt(np.sum((pri - prav)**2))
    rms.append(diff)
    
    # Test deconvolution of trial single-hit trace 
    shpeaks = getpeaks(trsingle, pri, 600, 500, [0.6, 2], 200, 6, t, ef, gfit=True)
    shpos.append(shpeaks[0][0][0])
    shint.append(shpeaks[0][0][1])
    
    # Test deconvolution of trial double-hit trace
    dhpeaks = getpeaks(trdouble, pri, 600, 500, [0.6, 2], 200, 6, t, ef, gfit=True)
    if not len(dhpeaks[0])==2:
        print("Wrong Number of Peaks! [{}]".format(i))
    dhpos.append(dhpeaks[0][1][0] - dhpeaks[0][0][0])
    dhint.append(dhpeaks[0][1][1]/dhpeaks[0][0][1])

# Reshape arrays
rms = np.reshape(rms, (nx, ny))
ccor = np.reshape(ccor, (nx, ny))
shpos = np.reshape(shpos, (nx, ny))
shint = np.reshape(shint, (nx, ny))
dhpos = np.reshape(dhpos, (nx, ny))
dhint = np.reshape(dhint, (nx, ny))

#%% Figures

plt.figure()
plt.imshow(rms, cmap='nipy_spectral_r')
plt.title("RMS Difference Between Average and Localized SH Response")
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)

plt.figure()
plt.imshow(ccor, vmin=0.995, vmax=1, cmap='nipy_spectral_r')
plt.title("Cross-correlation values of average and localized responses")
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)

plt.figure()
plt.imshow(shpos, vmin=374.95, cmap='nipy_spectral_r')
plt.title("Time (ns) of a Sample SH Trace by Deconvolution Using Localized Responses")
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)

plt.figure()
plt.imshow(dhpos, vmin=9.99, vmax=10.04, cmap='nipy_spectral_r')
plt.title("Time Difference (ns) of an Artificial Double-Hit Trace (d=10ns) by Deconvolution Using Localized Responses")
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)

plt.figure()
plt.imshow(dhint, vmin=0.96, vmax=1.08, cmap='nipy_spectral_r')
plt.title("Intensity Ratio of a Sample Double-Hit Trace by Deconvolution Using Localized Responses")
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)

# Show regions
#cen = [258, 247]
img = pts2img(points, dim, 5)
plt.figure()
plt.imshow(img)
plt.vlines(yval, 0, dim, color='red')
plt.hlines(xval, 0, dim, color='red')
plt.xlim(0, dim)
plt.ylim(0, dim)
dxt = (xval[1] - xval[0]) / 2
dyt = (yval[1] - yval[0]) / 2
xtk = np.arange(1, len(xval), dtype=int)
ytk = np.arange(1, len(yval), dtype=int)
plt.xticks(yval[1:] - dyt, ytk)
plt.yticks(xval[1:] - dxt, xtk)
circ0 = Circle((cen[0], cen[1]), 80, lw=3, ls='--', fill=False, color='black', zorder=2)
circ1 = Circle((cen[0], cen[1]), 210, lw=3, ls='--', fill=False, color='black', zorder=2)
plt.gca().add_patch(circ0)
plt.gca().add_patch(circ1)

t = t - 270

#%% Paper Figure

#cen = [255, 265] # xenon: [258, 247]

fig = plt.figure()
fig.set_tight_layout(True)
ax1 = plt.subplot2grid((2, 3), (0, 0))
ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
ax3 = plt.subplot2grid((2, 3), (1, 0))
ax4 = plt.subplot2grid((2, 3), (1, 1))
ax5 = plt.subplot2grid((2, 3), (1, 2))

# Single-Hit Response
prp = 1080
prw = 170
stp = 480 # xenon: 980
resplt = 0.8*prav/max(prav)
ax1.plot(t[prp:prp+prw], prav[stp:stp+prw], lw=4)
ax1.set_title("Global Single-Hit Response", fontsize=20)
ax1.set_xlim([0, 30])
#ax1.set_ylim([-0.25, 1])
ax1.set_xlabel("Time (ns)", fontsize=20)
ax1.axes.get_yaxis().set_visible(False)
ax1.tick_params(labelsize=18)

# Test Double-Hit Trace
dtp = 430
dtw = 250
ax2.plot(t[prp:prp+dtw], trdouble[dtp:dtp+dtw], lw=4)
#ax2.hlines(max(trdouble), 0, 17.7, color='red', lw=4)
#ax2.vlines([7.6, 17.6], min(trdouble), max(trdouble), color='red', lw=2)
ax2.set_title("Double-Hit Test Trace", fontsize=20)
ax2.set_xlim([0, 60])
ax2.set_xlabel("Time (ns)", fontsize=20)
ax2.tick_params(labelsize=18)
ax2.set_xticks([0, 10, 20, 30, 40, 50, 60])#, 7.7, 17.7])
ax2.set_xticklabels([0, 10, 20, 30, 40, 50, 60])#, "t1", "t2"])
ax2.set_yticks([])
#ax2.set_yticklabels(["A1=A2"], fontname="Cambria")
#ax2.get_xticklabels()[-1].set_color('red') 
#ax2.get_xticklabels()[-2].set_color('red') 
#ax2.get_yticklabels()[0].set_color('red')

# Regions
img = pts2img(points, dim, 5)
img = img/np.max(img)
#circ0 = Circle((cen[0], cen[1]), 100, lw=3, ls='--', fill=False, color='black', zorder=2)
im0 = ax3.imshow(img, cmap='nipy_spectral_r', vmin=0, vmax=1)
ax3.set_title("Spatial Bins", fontsize=20)
ax3.vlines(yval, 0, dim, color='gray', ls=':')
ax3.hlines(xval, 0, dim, color='gray', ls=':')
#ax3.add_patch(circ0)
ax3.set_xlim(0, dim)
ax3.set_ylim(0, dim)
dxt = (xval[1] - xval[0]) / 2
dyt = (yval[1] - yval[0]) / 2
xtk = np.arange(1, len(xval), dtype=int)
ytk = np.arange(1, len(yval), dtype=int)
ax3.set_xticks(yval[1::2] - dyt)
ax3.set_xticklabels(xtk[::2])
ax3.set_yticks(xval[1::2] - dxt)
ax3.set_yticklabels(ytk[::2])
ax3.tick_params(labelsize=15)
cbar = fig.colorbar(im0, ax=ax3)
cbar.ax.tick_params(labelsize=25)

# Time difference
vmin1 = 9.995 #9.978
vmax1 = 10.04 #10.015
#circ1 = Circle((9.9, 9.3), 4, lw=6, fill=False, color='black')
im1 = ax4.imshow(dhpos, vmin=vmin1, vmax=vmax1, cmap='nipy_spectral_r')
#ax[1,0].add_patch(circ1)
ax4.set_title("Time Difference, t2-t1 (ns)", fontsize=20)
ax4.axes.get_xaxis().set_visible(False)
ax4.axes.get_yaxis().set_visible(False)
cbar = fig.colorbar(im1, ax=ax4)
cbar.ax.tick_params(labelsize=25)

# Intensity Ratio
vmin2 = 0.85
vmax2 = 1 #1.08
#circ2 = Circle((9.9, 9.3), 4, lw=6, fill=False, color='black')
im2 = ax5.imshow(dhint, vmin=vmin2, vmax=vmax2, cmap='nipy_spectral_r')
#ax[1,1].add_patch(circ2)
ax5.set_title("Amplitude Ratio, A2/A1", fontsize=20)
ax5.axes.get_xaxis().set_visible(False)
ax5.axes.get_yaxis().set_visible(False)
cbar = fig.colorbar(im2, ax=ax5)
cbar.ax.tick_params(labelsize=25)

figman = plt.get_current_fig_manager()
figman.window.showMaximized()