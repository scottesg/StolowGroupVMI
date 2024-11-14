#%% Imports

import os
os.chdir(r'C:\Users\Scott\Python\VMI\src')

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy import optimize
import sgolay2
from matplotlib.colors import LogNorm

#%% load data

datapath = r"C:\Users\Scott\Python\VMI\data\20240418_VMI3DPaperData2\213266/213266_20240125_144323/"

name = "Ss"

path = "mpol{}9.300M.npz".format(name)

data = np.load(datapath + path)
K = np.load(datapath + "K{}.npy".format(name))

ring = 1
n0 = 5 # width of momentum slice
nth = 5 # width of theta slices
smooth = 0
plot = False
bksub = False
b4fit = True
plotslices = False

FSst = 20
FSt = 15
FSlb = 12
FSlg = 12
FStk = 12

ke = [0.054, 0.88, 2.05][ring] # eV, energy of ring
keau = ke/27.211 # in a.u.

mom = data['mom']
theta = data['theta']
phi = data['phi']
DCSSP = data['mpol']
dmom = np.round((mom[1]-mom[0])*n0/K)
dthet = (theta[1]-theta[0])*nth

argp = np.argmin(abs(np.sqrt(2*keau)-mom))

PHI, THETA, PP = np.meshgrid(phi, theta, mom, indexing="ij")

if plot:
    plt.figure()
    plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], np.sum(DCSSP[:,:,argp-4:argp+5], axis=2)/9, cmap="seismic", norm=LogNorm())
    plt.xlabel("Phi (deg)")
    plt.ylabel("Theta (deg)")
    cm = plt.colorbar()
    cm.set_label("Yield (arb)")

#%%

if bksub:

    bk = np.load(datapath + "mpolA800_BKNC.npz")
    DCSSP_BK = bk['mpol']
    
    if plot:
        plt.figure()
        plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], np.sum(DCSSP_BK[:,:,argp-4:argp+5], axis=2)/9, cmap="seismic", norm=LogNorm())
        plt.xlabel("Phi (deg)")
        plt.ylabel("Theta (deg)")
        cm = plt.colorbar()
        cm.set_label("Yield (arb)")
    
    DCSSP = DCSSP - 0.2*DCSSP_BK
    
    DCSSP[DCSSP<0] = 0

#%% smooth the collected data

sm = np.sum(DCSSP[:,:,argp-n0:argp+n0+1], axis=2) / (2*n0+1)

if smooth > 0:
    # smooth the edges in phi
    sm[0] = ss.savgol_filter(sm[0], 3, 1)
    sm[1] = ss.savgol_filter(sm[1], 3, 1)
    sm[2] = ss.savgol_filter(sm[2], 3, 1)
    sm[-1] = ss.savgol_filter(sm[-1], 3, 1)
    sm[-2] = ss.savgol_filter(sm[-2], 3, 1)
    sm[-3] = ss.savgol_filter(sm[-3], 3, 1)
    
    # full smoothing
    sphere = sgolay2.SGolayFilter2(smooth, 1)(sm)
else:
    sphere = sm

if plot:
    fig = plt.figure()
    plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], 2*sphere/sphere.max(), cmap="seismic", vmax=1, vmin=0)
    plt.xlabel("Phi (deg)")
    plt.ylabel("Theta (deg)")
    cm = plt.colorbar()
    cm.set_label("Normalized Yield (arb)")

#%% fit phi distribution to Legendre polynomials

fig = plt.figure()
fig.set_tight_layout(True)

ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1))
ax3 = plt.subplot2grid((3, 3), (0, 2))
ax4 = plt.subplot2grid((3, 3), (1, 0), rowspan=2)
ax5 = plt.subplot2grid((3, 3), (1, 1), colspan=2)
ax6 = plt.subplot2grid((3, 3), (2, 1), colspan=2)

pos1 = 0.03
pos2 = 0.96
letsize = 15
col1 = 'black'
col2 = 'white'

if b4fit:
    
    arg90 = np.argmin(abs(theta-90))
    arg180 = np.argmin(abs(theta-180))
    arg270 = np.argmin(abs(theta-270))
    
    # average over theta
    ABE = np.average(sphere, axis=1)
    ABExy = (np.average(sphere[:, arg90-nth:arg90+nth+1], axis=1)
              + np.average(sphere[:, arg270-nth:arg270+nth+1], axis=1))
    ABExz = (np.average(sphere[:, :nth+1], axis=1)
              + np.average(sphere[:, arg180-nth:arg180+nth+1], axis=1)
              + np.average(sphere[:, -nth:], axis=1))
    
    phi2 = np.hstack((phi, phi+180))
    ABExy2 = np.hstack((np.average(sphere[:, arg90-nth:arg90+nth+1], axis=1),
                        np.average(sphere[:, arg270-nth:arg270+nth+1], axis=1)[::-1]))
    ABExz2 = np.hstack(((np.average(sphere[:, :nth+1], axis=1) + np.average(sphere[:, -nth:], axis=1))/2,
                        np.average(sphere[:, arg180-nth:arg180+nth+1], axis=1)[::-1]))
    
    #plt.suptitle("Legendre Fit: {} eV Ring [{:1.0f}*K width] [{}]".format(ke, dmom, path[:-4]), fontsize=FSst)
    
    norm = 0.9*ABE.max()
    
    def legendre(x, beta2, beta4, a):
        return a * (1 + beta2*0.5*(3*x**2 - 1) + beta4*(1/8)*(35*x**4 - 30*x**2 + 3))
    
    # fit
    optim = optimize.curve_fit(legendre, np.cos(np.radians(phi)), ABE/norm, p0=[1.6, 0, 1])
    b2 = optim[0][0]
    b4 = optim[0][1]
    afit = optim[0][2]
    errs = np.sqrt(np.diag(optim[1]))
    db2 = errs[0]
    db4 = errs[1]
    
    # plot beta and upper and lower limits from fit
    ax1.plot(phi, legendre(np.cos(np.radians(phi)), *optim[0]), ls="--", c="b",
              label=r"Fit: $\beta_{2}=$%.3f$\pm$%.3f,"%(b2, db2) + "\n      " + r"$\beta_{4}=$%.3f$\pm$%.3f"%(b4, db4))
    ax1.plot(phi, ABE/norm, lw=0, marker='x', color='red', label='Data')
    ax1.legend(loc=9, fontsize=FSlg)
    ax1.set_ylabel("Yield (normalized)", fontsize=FSlb)
    ax1.set_ylim([0, 1.35])
    ax1.set_xlim([-2, 182])
    ax1.set_xlabel("$\phi$' (\N{DEGREE SIGN})", fontsize=FSlb)
    ax1.set_title("Integrated", fontsize=FSt)
    ax1.tick_params(labelsize=FStk)
    
    t1 = ax1.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=letsize)
    t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
    
    norm = 1.2*ABE.max()
    
    # fit xy
    
    optim = optimize.curve_fit(legendre, np.cos(np.radians(phi2)), ABExy2/norm, p0=[1.6, 0, 1])
    b2 = optim[0][0]
    b4 = optim[0][1]
    afit = optim[0][2]
    errs = np.sqrt(np.diag(optim[1]))
    db2 = errs[0]
    db4 = errs[1]
    
    ax2.plot(phi2, legendre(np.cos(np.radians(phi2)), *optim[0]), ls="--", c="b",
              label=r"Fit: $\beta_{2}=$%.2f$\pm$%.2f, "%(b2, db2)
              + r"$\beta_{4}=$%.2f$\pm$%.2f"%(b4, db4))
    ax2.plot(phi2, ABExy2/norm, lw=0, marker='x', color='red', label='Data')
    leg = ax2.legend(loc=9, fontsize=FSlg, framealpha=0.2)
    leg.set_draggable(1)
    ax2.set_ylabel("Yield (normalized)", fontsize=FSlb)
    ax2.set_ylim([0, 1.35])
    ax2.set_xlim([-2, 362])
    ax2.set_xlabel("$\phi$' (\N{DEGREE SIGN})", fontsize=FSlb)
    ax2.set_title("XY Slice", fontsize=FSt)
    ax2.tick_params(labelsize=FStk)
    
    t2 = ax2.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=letsize)
    t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
    
    # fit xz
    
    optim = optimize.curve_fit(legendre, np.cos(np.radians(phi2)), ABExz2/norm, p0=[1.6, 0, 1])
    b2 = optim[0][0]
    b4 = optim[0][1]
    afit = optim[0][2]
    errs = np.sqrt(np.diag(optim[1]))
    db2 = errs[0]
    db4 = errs[1]
    
    ax3.plot(phi2, legendre(np.cos(np.radians(phi2)), *optim[0]), ls="--", c="b",
              label=r"Fit: $\beta_{2}=$%.2f$\pm$%.2f, "%(b2, db2)
              + r"$\beta_{4}=$%.2f$\pm$%.2f"%(b4, db4))
    ax3.plot(phi2, ABExz2/norm, lw=0, marker='x', color='red', label='Data')
    leg = ax3.legend(loc=9, fontsize=FSlg, framealpha=0.2)
    leg.set_draggable(1)
    ax3.set_ylabel("Yield (normalized)", fontsize=FSlb)
    ax3.set_ylim([0, 1.35])
    ax3.set_xlim([-2, 362])
    ax3.set_xlabel("$\phi$' (\N{DEGREE SIGN})", fontsize=FSlb)
    ax3.set_title("XZ Slice", fontsize=FSt)
    ax3.tick_params(labelsize=FStk)
    
    t3 = ax3.text(pos1, pos2, '(c)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=letsize)
    t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

# fit phi distribution to Legendre polynomials [b4=0]
else:

    arg90 = np.argmin(abs(theta-90))
    arg180 = np.argmin(abs(theta-180))
    arg270 = np.argmin(abs(theta-270))
    
    # average over theta
    ABE = np.average(sphere, axis=1)
    ABExy = (np.average(sphere[:, arg90-nth:arg90+nth+1], axis=1)
             + np.average(sphere[:, arg270-nth:arg270+nth+1], axis=1))
    ABExz = (np.average(sphere[:, :nth+1], axis=1)
             + np.average(sphere[:, arg180-nth:arg180+nth+1], axis=1)
             + np.average(sphere[:, -nth:], axis=1))
    
    phi2 = np.hstack((phi, phi+180))
    ABExy2 = np.hstack((np.average(sphere[:, arg90-nth:arg90+nth+1], axis=1),
                       np.average(sphere[:, arg270-nth:arg270+nth+1], axis=1)[::-1]))
    ABExz2 = np.hstack(((np.average(sphere[:, :nth+1], axis=1) + np.average(sphere[:, -nth:], axis=1))/2,
                        np.average(sphere[:, arg180-nth:arg180+nth+1], axis=1)[::-1]))
    
    plt.suptitle("Legendre Fit: {} eV Ring [{:1.0f}*K width] [{}]".format(ke, dmom, path[:-4]), fontsize=FSst)
    
    def legendre(x, beta2, a):
        return a * (1 + beta2*0.5*(3*x**2 - 1))
    
    # fit
    optim = optimize.curve_fit(legendre, np.cos(np.radians(phi)), ABE/ABE.max(), p0=[1.6, 1])
    b2 = optim[0][0]
    afit = optim[0][1]
    errs = np.sqrt(np.diag(optim[1]))
    db2 = errs[0]
    
    # plot beta and upper and lower limits from fit
    ax1.plot(phi, legendre(np.cos(np.radians(phi)), *optim[0]), ls="--", c="b",
             label=r"Fit: $\beta_{2}=$%.2f$\pm$%.2f"%(b2, db2))
    ax1.plot(phi, ABE/ABE.max(), lw=0, marker='x', color='red', label='Data')
    ax1.legend(loc=9, fontsize=FSlg)
    ax1.set_ylabel("Yield (normalized)", fontsize=FSlb)
    ax1.set_xlabel("Phi (deg)", fontsize=FSlb)
    ax1.set_title("Integrated", fontsize=FSt)
    ax1.tick_params(labelsize=FStk)
    
    # fit xy
    
    optim = optimize.curve_fit(legendre, np.cos(np.radians(phi2)), ABExy2/ABExy2.max(), p0=[1.85, 1])
    b2 = optim[0][0]
    afit = optim[0][1]
    errs = np.sqrt(np.diag(optim[1]))
    db2 = errs[0]
    
    ax2.plot(phi2, legendre(np.cos(np.radians(phi2)), *optim[0]), ls="--", c="b",
             label=r"Fit_xy: $\beta_{2}=$%.2f$\pm$%.2f"%(b2, db2))
    ax2.plot(phi2, ABExy2/ABExy2.max(), lw=0, marker='x', color='red', label='Data')
    ax2.legend(loc=9, fontsize=FSlg)
    ax2.set_ylabel("Yield (normalized)", fontsize=FSlb)
    ax2.set_xlabel("Phi (deg)", fontsize=FSlb)
    ax2.set_title("XY Slice [{:2.0f} deg width]".format(dthet), fontsize=FSt)
    ax2.tick_params(labelsize=FStk)
    
    # fit xz
    
    optim = optimize.curve_fit(legendre, np.cos(np.radians(phi2)), ABExz2/ABExz2.max(), p0=[1.85, 1])
    b2 = optim[0][0]
    afit = optim[0][1]
    errs = np.sqrt(np.diag(optim[1]))
    db2 = errs[0]
    
    ax3.plot(phi2, legendre(np.cos(np.radians(phi2)), *optim[0]), ls="--", c="b",
             label=r"Fit_xz: $\beta_{2}=$%.2f$\pm$%.2f"%(b2, db2))
    ax3.plot(phi2, ABExz2/ABExz2.max(), lw=0, marker='x', color='red', label='Data')
    ax3.legend(loc=9, fontsize=FSlg)
    ax3.set_ylabel("Yield (normalized)", fontsize=FSlb)
    ax3.set_xlabel("Phi (deg)", fontsize=FSlb)
    ax3.set_title("XZ Slice [{:2.0f} deg width]".format(dthet), fontsize=FSt)
    ax3.tick_params(labelsize=FStk)

#%% fit phi distribution to Legendre polynomials [theta slices]

plotall = False

thet = np.arange(0, 361, 10)
b2s = np.zeros(len(thet)-1)
b4s = np.zeros(len(thet)-1)
db2s = np.zeros(len(thet)-1)
db4s = np.zeros(len(thet)-1)

def legendre(x, beta2, beta4, a):
    return a * (1 + beta2*0.5*(3*x**2 - 1) + beta4*(1/8)*(35*x**4 - 30*x**2 + 3))

for i in range(len(thet)-1):

    arg1 = np.argmin(abs(theta-thet[i]))
    arg2 = np.argmin(abs(theta-thet[i+1]))
    
    thslice = np.average(sphere[:, arg1:arg2+1], axis=1)
    
    optim = optimize.curve_fit(legendre, np.cos(np.radians(phi)), thslice/thslice.max(), p0=[1.6, 0, 1])
    b2 = optim[0][0]
    b4 = optim[0][1]
    afit = optim[0][2]
    errs = np.sqrt(np.diag(optim[1]))
    db2 = errs[0]
    db4 = errs[1]
    
    b2s[i] = b2
    b4s[i] = b4
    db2s[i] = db2
    db4s[i] = db4

    if plotall:
        fig, ax = plt.subplots()
        ax.plot(phi, legendre(np.cos(np.radians(phi)), *optim[0]), ls="--", c="b",
                 label=r"Fit: $\beta_{2}=$%.2f"%(b2))
        ax.plot(phi, thslice/thslice.max(), lw=0, marker='x', color='red', label='Data')
        ax.legend(loc=9, fontsize=FSlg)
        ax.set_ylabel("Yield (normalized)", fontsize=FSlb)
        ax.set_xlabel("Phi (deg)", fontsize=FSlb)
        ax.set_title("Angle {}:{}".format(thet[i], thet[i+1]), fontsize=FSt)
        ax.tick_params(labelsize=FStk)
        
#%%

ax5.plot(thet[:-1]+5, b2s, marker='o', ls='--')
ax5.fill_between(thet[:-1]+5, y1=b2s-db2s, y2=b2s+db2s, color="k", alpha=0.3, label=r"$\beta_{2}\pm1\sigma$")
ax5.set_xlabel("$\\theta$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax5.set_ylabel(r"$\beta_{2}$", fontsize=FSlb)
ax5.set_title(r"Angular Variation of $\beta_{2}$", fontsize=FSt)
ax5.set_xlim([0, 360])
ax5.tick_params(labelsize=FStk)
ax5.legend(loc=1, fontsize=FSlg)

t5 = ax5.text(pos1-0.02, pos2, '(e)', ha='left', va='top', color=col1, transform=ax5.transAxes, fontsize=letsize)
t5.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

ax6.plot(thet[:-1]+5, b4s, marker='o', ls='--')
ax6.fill_between(thet[:-1]+5, y1=b4s-db4s, y2=b4s+db4s, color="k", alpha=0.3, label=r"$\beta_{4}\pm1\sigma$")
ax6.set_xlabel("$\\theta$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax6.set_ylabel(r"$\beta_{4}$", fontsize=FSlb)
ax6.set_title(r"Angular Variation of $\beta_{4}$", fontsize=FSt)
ax6.set_xlim([0, 360])
ax6.tick_params(labelsize=FStk)
ax6.legend(loc=1, fontsize=FSlg)

t6 = ax6.text(pos1-0.02, pos2, '(f)', ha='left', va='top', color=col1, transform=ax6.transAxes, fontsize=letsize)
t6.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

#%% fit phi distribution to Legendre polynomials [theta slices]

if plotslices:

    plotall = False
    
    thet = np.arange(0, 361, 30)
    b2s = np.zeros(len(thet)-1)
    b4s = np.zeros(len(thet)-1)
    db2s = np.zeros(len(thet)-1)
    db4s = np.zeros(len(thet)-1)
    
    thparams = []
    thslices = []
    
    def legendre(x, beta2, beta4, a):
        return a * (1 + beta2*0.5*(3*x**2 - 1) + beta4*(1/8)*(35*x**4 - 30*x**2 + 3))
    
    for i in range(len(thet)-1):
    
        arg1 = np.argmin(abs(theta-thet[i]))
        arg2 = np.argmin(abs(theta-thet[i+1]))
        
        thslice = np.average(sphere[:, arg1:arg2+1], axis=1)
        
        optim = optimize.curve_fit(legendre, np.cos(np.radians(phi)), thslice/thslice.max(), p0=[1.6, 0, 1])
        thparams.append(optim)
        thslices.append(thslice)
        
        b2 = optim[0][0]
        b4 = optim[0][1]
        afit = optim[0][2]
        errs = np.sqrt(np.diag(optim[1]))
        db2 = errs[0]
        db4 = errs[1]
        
        b2s[i] = b2
        b4s[i] = b4
        db2s[i] = db2
        db4s[i] = db4
    
        if plotall:
            fig, ax = plt.subplots()
            ax.plot(phi, legendre(np.cos(np.radians(phi)), *optim[0]), ls="--", c="b",
                     label=r"Fit: $\beta_{2}=$%.2f"%(b2))
            ax.plot(phi, thslice/thslice.max(), lw=0, marker='x', color='red', label='Data')
            ax.legend(loc=9, fontsize=FSlg)
            ax.set_ylabel("Yield (normalized)", fontsize=FSlb)
            ax.set_xlabel("Phi (deg)", fontsize=FSlb)
            ax.set_title("Angle {}:{}".format(thet[i], thet[i+1]), fontsize=FSt)
            ax.tick_params(labelsize=FStk)

#%%

# smooth the edges in phi
sm[0] = ss.savgol_filter(sm[0], 3, 1)
sm[1] = ss.savgol_filter(sm[1], 3, 1)
sm[2] = ss.savgol_filter(sm[2], 3, 1)
sm[-1] = ss.savgol_filter(sm[-1], 3, 1)
sm[-2] = ss.savgol_filter(sm[-2], 3, 1)
sm[-3] = ss.savgol_filter(sm[-3], 3, 1)

# full smoothing
sphere = sgolay2.SGolayFilter2(5, 1)(sm)

ax4.pcolor(PHI[:,:,argp], THETA[:,:,argp], 2*sphere/sphere.max(), cmap="seismic", vmax=1, vmin=0)
ax4.set_xlabel("$\phi$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax4.set_ylabel("$\\theta$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax4.set_title("Normalized Yield (smoothed)", fontsize=FSt)

t4 = ax4.text(pos1, pos2, '(d)', ha='left', va='top', color=col1, transform=ax4.transAxes, fontsize=letsize)
t4.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

#%%

if plotslices:

    fig, ax = plt.subplots(3, 4)
    fig.set_tight_layout(True)
    ax = ax.flatten()
    
    plt.suptitle("Beta 2 Fits For Slices (30 deg width) [{}]".format(path[:-4]), fontsize=FSst)
    
    for i in range(len(ax)):
        b2 = thparams[i][0]
        ax[i].plot(phi, legendre(np.cos(np.radians(phi)), *thparams[i]), ls="--", c="b")
        ax[i].plot(phi, thslices[i], lw=0, marker='x', color='red', label="B2={:1.2f}".format(b2))
        ax[i].set_title("Theta {}:{} deg".format(thet[i], thet[i+1]))
        ax[i].set_xlabel("Phi (deg)")
        ax[i].set_ylabel("Yield (norm.)")
        ax[i].legend(loc=9)