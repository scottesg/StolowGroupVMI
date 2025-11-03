#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from VMI3D_IO import readwfm, genT
from VMI3D_Functions import (ccornorm, pts2img, xybin, pts2pr, genBK, getpeaks)

#%% Figure 4: Pickoff Time Response Uniformity
# Using 'points' output of VMI3D 1st run
# xenon 31ns H:\DataBackup\20220831\20220831_1823\OldAnalysis\X_20221011\T2

nx = 20
ny = 20
nsamples = 100
dim = 512
dqdim = 2048
angl = 32
blpts = 200
sgn = -1
extraplots = False
fw = 20

# Load
bsi = 6 # background index
pointspath = "DP_SHTT_DHTT_CH3I_pt.npy"
bkpath = "DP_SHTT_DHTT_CH3I_bk.npy"
wfmpath = "DP_SHTT_DHTT_CH3I_wfm/daqdata*.uint16"
cen = [264, 255]

points = np.load(pointspath)

wfms = readwfm(wfmpath, dqdim, groupstks=True)
bk = np.load(bkpath)
tracelen = len(wfms[0][0])

# Rotate Points
rotangle = angl*(np.pi/180)
xc = points[:,0] - cen[0]
yc = points[:,1] - cen[1]
x2 = xc*np.cos(rotangle) + yc*np.sin(rotangle)
y2 = yc*np.cos(rotangle) - xc*np.sin(rotangle)

circen = [256, 256]
points[:,0] = x2 + circen[0]
points[:,1] = y2 + circen[1]

# Bin
xval = np.linspace(0, dim, nx+1)
yval = np.linspace(0, dim, ny+1)
binned = xybin(points, nx, ny, dim)

if extraplots:
    # Show regions
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
    circ0 = Circle((circen[1], circen[0]), 80, lw=3, ls='--', fill=False, color='black', zorder=2)
    circ1 = Circle((circen[1], circen[0]), 210, lw=3, ls='--', fill=False, color='black', zorder=2)
    plt.gca().add_patch(circ0)
    plt.gca().add_patch(circ1)

#%% Generate response for each bin

# bk = genBK(wfmpath, 200, blpts)
# np.save("DP_WFTest_CH3I_bk.npy", bk)

prs = []
shtraces = []
for i in range(0, len(binned)):
    print("Bin Number: {}".format(i))
    if len(binned[i]>1):
        pri = pts2pr(binned[i], wfms, dqdim, bk, sgn, 600, 4000, 12000, 2, 0,
                     None, None, None, None, fw, ntrace=nsamples, shreturn=True)
        if len(pri) == 2:
            pr = np.hstack((pri[0][500:], pri[0][:500]))
            prs.append(pr)
            shtraces.append(pri[1])
        else: 
            prs.append(np.zeros(2048))
            shtraces.append([])
    else:
        prs.append(np.zeros(2048))
        shtraces.append([])

# Normalized average response
gsr = np.mean(np.array(prs), 0)
h = max(gsr)
gsr = gsr / max(gsr)

# Time (ns)
t = genT(tracelen, 0.25)
ef = 1/np.linspace(1000, 1, 20)

#%% Evaluate comparisons for each bin

# Comparison metrics
shpos = []
shint = []
dhpos = []
dhint = []
shwd = []
dhwd = []

filt = [0.6, 2]
gsm = 1.7 # Gaussian filter [data points], overrides butter filter if not 0
thresh = 0.1

gsrpeaks = getpeaks(gsr*h, gsr*h, thresh, 500, filt, 2, t, ef, gsm=gsm, gfit=True)
gsrtime = gsrpeaks[0][0][0]
gsrampl = gsrpeaks[0][0][1]

for i in range(0, len(prs)):
    pri = prs[i]
    
    print("LSR #", i)
    
    # If bin did not have enough samples
    if max(pri)<1:
        shpos.append(0)
        shint.append(0)
        dhpos.append(0)
        dhint.append(0)
        shwd.append(0)
        dhwd.append(0)
        continue
    
    # Normalize bin pr
    pri = pri / max(pri)
    
    # Artificial test traces
    trsingle = pri*h
    secondpeak = np.hstack((pri[40:], pri[:40]))
    trdouble = (pri + secondpeak)*h
    
    # Test deconvolution of trial single-hit trace 
    shpeaks = getpeaks(trsingle, gsr*h, thresh, 500, filt, 2, t, ef, gsm=gsm, gfit=True)
    if shpeaks is None:
        print("Wrong Number of Peaks! [{}]".format(i))
        shwd.append(0)
        shpos.append(0)
        shint.append(0)
    elif not len(shpeaks[0])==1:
        print("Wrong Number of Peaks! [{}]".format(i))
        shwd.append(0)
        shpos.append(0)
        shint.append(0)
    else:
        shwd.append(shpeaks[3])
        shpos.append(shpeaks[0][0][0])
        shint.append(shpeaks[0][0][1])
    
    # Test deconvolution of trial double-hit trace
    dhpeaks = getpeaks(trdouble, gsr*h, thresh, 500, filt, 3, t, ef, gsm=gsm, gfit=True)
    if shpeaks is None:
        print("Wrong Number of Peaks! [{}]".format(i))
        dhwd.append(0)
        dhpos.append(0)
        dhint.append(0)
    elif not len(dhpeaks[0])==2:
        print("Wrong Number of Peaks! [{}]".format(i))
        dhwd.append(0)
        dhpos.append(0)
        dhint.append(0)
    else:
        dhwd.append(dhpeaks[3])
        dhpos.append(dhpeaks[0][1][0] - dhpeaks[0][0][0])
        dhint.append(dhpeaks[0][1][1]/dhpeaks[0][0][1])

#%% Reshape arrays

shpos = np.reshape(shpos, (nx, ny))
shint = np.reshape(shint, (nx, ny))
dhpos = np.reshape(dhpos, (nx, ny))
dhint = np.reshape(dhint, (nx, ny))
prs = [prs[x:x+nx] for x in range(0, len(prs), ny)] 
shwd = [shwd[x:x+nx] for x in range(0, len(shwd), ny)] 
dhwd = [dhwd[x:x+nx] for x in range(0, len(dhwd), ny)] 

#%% Figures

if extraplots:
    
    plt.figure()
    plt.imshow(shpos, vmin=124.85, cmap='nipy_spectral_r')
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
    plt.imshow(dhint, vmin=0.90, vmax=1.08, cmap='nipy_spectral_r')
    plt.title("Intensity Ratio of a Sample Double-Hit Trace by Deconvolution Using Localized Responses")
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=30)

#%% SH Figure

pri = prs[10][10]
pri = pri / max(pri)
trsingle = pri*h
secondpeak = np.hstack((pri[40:], pri[:40]))
trdouble = (pri + secondpeak)*h
    
trsingle -= np.mean(trsingle[:300])
deconorm = shwd[10][10]/max(shwd[10][10])
shttnorm = trsingle/max(trsingle)

img = pts2img(points, dim, 5)
img = img/np.max(img)
dxt = (xval[1] - xval[0]) / 2
dyt = (yval[1] - yval[0]) / 2
xtk = np.arange(1, len(xval), dtype=int)
ytk = np.arange(1, len(yval), dtype=int)

FSt = 20
FSlb = 18
FSlg = 15
FStk = 15

fig = plt.figure()
fig.set_tight_layout(True)
ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=3)
ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=3)
ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)

# GSR
tstart = 120
twidth = 30
ax1.plot(t-tstart, gsr, color='blue', lw=4)
ax1.set_title("Global Single-Hit Response (GSR)", fontsize=FSt)
ax1.set_xlim([0, twidth])
ax1.set_ylim([-0.25, 1.1])
ax1.set_xlabel("Time (ns)", fontsize=FSlb)
ax1.set_ylabel("Normalized Amplitude (arb)", fontsize=FSlb)
ax1.set_yticks([])
ax1.tick_params(labelsize=FStk)

# SH Deconvolution and SHTT
tstart = 120
twidth = 30
ax2.plot(t-tstart, shttnorm, lw=4, color='blue', label="LSR")
ax2.plot(t-tstart, deconorm, lw=2, color='black', ls="--", label="Deconvolution")
ax2.set_title("Local Single-Hit Response (LSR)", fontsize=FSt)
ax2.set_xlim([0, twidth])
ax2.set_ylim([-0.25, 1.1])
ax2.set_xlabel("Time (ns)", fontsize=FSlb)
ax2.set_ylabel("Normalized Amplitude (arb)", fontsize=FSlb)
ax2.set_yticks([])
ax2.tick_params(labelsize=FStk)
ax2.legend(loc=1, fontsize=FSlg)

rscale = 1.6
circen = (9.5, 9.5)
circr = 10
circ1 = Circle((256, 256), 256/rscale, lw=4, ls="--", fill=False, color='black')
circ2 = Circle(circen, circr/rscale, lw=4, ls="--", fill=False, color='black')
circ3 = Circle(circen, circr/rscale, lw=4, ls="--", fill=False, color='black')

# Regions
im0 = ax3.imshow(img, cmap='nipy_spectral_r', vmin=0, vmax=1)
#ax3.add_patch(circ1) # Use to test circle radius
ax3.set_title("Spatial Bins", fontsize=FSt)
ax3.vlines(yval, 0, dim, color='gray', ls=':')
ax3.hlines(xval, 0, dim, color='gray', ls=':')
ax3.set_xlim(0, dim)
ax3.set_ylim(0, dim)
ax3.set_xticks(yval[1::2] - dyt)
ax3.set_xticklabels(xtk[::2])
ax3.set_yticks(xval[1::2] - dxt)
ax3.set_yticklabels(ytk[::2])
ax3.tick_params(labelsize=FStk)
cbar = fig.colorbar(im0, ax=ax3)
cbar.ax.tick_params(labelsize=FStk)

# Time
vmin1 = -0.04
vmax1 = 0.06
im1 = ax4.imshow(shpos-gsrtime, vmin=vmin1, vmax=vmax1, cmap='nipy_spectral_r')
ax4.add_patch(circ2)
ax4.set_title(r"Time Difference, $t_{LSR}$ - $t_{GSR}$ (ns)", fontsize=FSt)
ax4.vlines(np.arange(-0.5, 20, 1), -0.5, 19.5, color='gray', ls=':')
ax4.hlines(np.arange(-0.5, 20, 1), -0.5, 19.5, color='gray', ls=':')
ax4.set_xlim(-0.5, 19.5)
ax4.set_ylim(-0.5, 19.5)
ax4.set_xticks(np.arange(0, 20, 2))
ax4.set_xticklabels(np.arange(1, 21, 2))
ax4.set_yticks(np.arange(0, 20, 2))
ax4.set_yticklabels(np.arange(1, 21, 2))
ax4.tick_params(labelsize=FStk)
cbar = fig.colorbar(im1, ax=ax4)
cbar.ax.tick_params(labelsize=FStk)

# Intensity Ratio
vmin2 = 0.85
vmax2 = 1.25
im2 = ax5.imshow(shint/gsrampl, vmin=vmin2, vmax=vmax2, cmap='nipy_spectral_r')
ax5.add_patch(circ3)
ax5.set_title(r"Amplitude Ratio, $A_{LSR}$ / $A_{GSR}$", fontsize=FSt)
ax5.vlines(np.arange(-0.5, 20, 1), -0.5, 19.5, color='gray', ls=':')
ax5.hlines(np.arange(-0.5, 20, 1), -0.5, 19.5, color='gray', ls=':')
ax5.set_xlim(-0.5, 19.5)
ax5.set_ylim(-0.5, 19.5)
ax5.set_xticks(np.arange(0, 20, 2))
ax5.set_xticklabels(np.arange(1, 21, 2))
ax5.set_yticks(np.arange(0, 20, 2))
ax5.set_yticklabels(np.arange(1, 21, 2))
ax5.tick_params(labelsize=FStk)
cbar = fig.colorbar(im2, ax=ax5)
cbar.ax.tick_params(labelsize=FStk)

figman = plt.get_current_fig_manager()
figman.window.showMaximized()

col1 = 'black'
col2 = 'white'
pos1 = 0.04
pos2 = 0.96
fslett = 20

t1 = ax1.text(pos1-0.02, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=fslett)
t2 = ax2.text(pos1-0.03, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)
t3 = ax3.text(pos1, pos2, '(c)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=fslett)
t4 = ax4.text(pos1, pos2, '(d)', ha='left', va='top', color=col1, transform=ax4.transAxes, fontsize=fslett)
t5 = ax5.text(pos1, pos2, '(e)', ha='left', va='top', color=col1, transform=ax5.transAxes, fontsize=fslett)

t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t4.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t5.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

#%% DH Figure

trdouble -= np.mean(trdouble[:300])
deconormdh = dhwd[10][10]/max(dhwd[10][10])
dhttnorm = trdouble/max(trdouble)

FSt = 18
FSlb = 15
FSlg = 12
FStk = 12

fig = plt.figure(figsize=(10.5,8.5))
fig.set_tight_layout(True)
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (1, 1))

# SH Deconvolution and DHTT
tstart = 110
twidth = 30
ax1.plot(t-tstart, dhttnorm, lw=4, color='blue', label="Test Trace")
ax1.plot(t-tstart, deconormdh, lw=2, color='black', ls="--", label="Deconvolution")
ax1.set_title("Double-Hit Test Trace", fontsize=FSt)
ax1.set_xlim([0, twidth])
ax1.set_ylim([-0.30, 1.1])
ax1.set_xlabel("Time (ns)", fontsize=FSlb)
ax1.set_ylabel("Normalized Amplitude (arb)", fontsize=FSlb)
ax1.set_yticks([])
ax1.tick_params(labelsize=FStk)
ax1.legend(loc=1, fontsize=FSlg)
ax1.text(0.2, 0.95, 'Peak #1', ha='left', va='top', color='black', transform=ax1.transAxes, fontsize=FSlb)
ax1.text(0.54, 0.95, 'Peak #2', ha='left', va='top', color='black', transform=ax1.transAxes, fontsize=FSlb)

circ4 = Circle(circen, circr/rscale, lw=4, ls="--", fill=False, color='black')
circ5 = Circle(circen, circr/rscale, lw=4, ls="--", fill=False, color='black')

# Time
vmin1 = 9.985
vmax1 = 10.015
im1 = ax2.imshow(dhpos, vmin=vmin1, vmax=vmax1, cmap='nipy_spectral_r')
ax2.add_patch(circ4)
ax2.set_title(r"Time Difference, $t_2$ - $t_1$ (ns)", fontsize=FSt)
ax2.vlines(np.arange(-0.5, 20, 1), -0.5, 19.5, color='gray', ls=':')
ax2.hlines(np.arange(-0.5, 20, 1), -0.5, 19.5, color='gray', ls=':')
ax2.set_xlim(-0.5, 19.5)
ax2.set_ylim(-0.5, 19.5)
ax2.set_xticks(np.arange(0, 20, 2))
ax2.set_xticklabels(np.arange(1, 21, 2))
ax2.set_yticks(np.arange(0, 20, 2))
ax2.set_yticklabels(np.arange(1, 21, 2))
ax2.tick_params(labelsize=FStk)
cbar = fig.colorbar(im1, ax=ax2)
cbar.ax.tick_params(labelsize=FStk)

# Intensity Ratio
vmin2 = 0.97
vmax2 = 1.03
im2 = ax3.imshow(dhint, vmin=vmin2, vmax=vmax2, cmap='nipy_spectral_r')
ax3.add_patch(circ5)
ax3.set_title(r"Amplitude Ratio, $A_2$ / $A_1$", fontsize=FSt)
ax3.vlines(np.arange(-0.5, 20, 1), -0.5, 19.5, color='gray', ls=':')
ax3.hlines(np.arange(-0.5, 20, 1), -0.5, 19.5, color='gray', ls=':')
ax3.set_xlim(-0.5, 19.5)
ax3.set_ylim(-0.5, 19.5)
ax3.set_xticks(np.arange(0, 20, 2))
ax3.set_xticklabels(np.arange(1, 21, 2))
ax3.set_yticks(np.arange(0, 20, 2))
ax3.set_yticklabels(np.arange(1, 21, 2))
ax3.tick_params(labelsize=FStk)
cbar = fig.colorbar(im2, ax=ax3)
cbar.ax.tick_params(labelsize=FStk)

col1 = 'black'
col2 = 'white'
pos1 = 0.04
pos2 = 0.96
fslett = 20

t1 = ax1.text(pos1-0.03, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=fslett)
t2 = ax2.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)
t3 = ax3.text(pos1, pos2, '(c)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=fslett)

t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
