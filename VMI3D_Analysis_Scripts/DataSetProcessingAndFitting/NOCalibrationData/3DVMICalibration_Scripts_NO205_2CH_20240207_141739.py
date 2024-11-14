# %% Imports

import os
os.chdir(r'C:\Users\Scott\Python\VMI\src')

import random as rd
from matplotlib.colors import LogNorm
from VMI3D_Fitting import fit_gauss as fg
from VMI3D_Functions import xycorrectpoints
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np

# Constants

datapath = r"C:\Users\Scott\Python\VMI\data\20240418_VMI3DPaperData2\213/213_20240207_141739/"
ke = 2.05  # eV, energy of ring
t0 = 2084.95  # ns, centre tof
T0 = 29.5  # ns, put centre here
cen = [255.75, 238]  # xy centre
rotangle = 26.7*np.pi/180  # xy rotation angle (camera not aligned)
dt = 5.5 #32  # ns, width/2 of time range to look at
plotdim = (1280, 400)
# dt and plotdim have to be the same value as in the 0.88 eV analysis for use in Fig2.py

covshuffle = True  # shuffle based covariance method
undogridscale = False

name = 'S'
if covshuffle: name = name + 's'
if undogridscale: name = name + 'u'

tl = [T0-dt, T0+dt]  # time range
tshift = t0 - T0

fs = 7  # fontsize
plt.rcParams['figure.dpi'] = 150

loadpath = r"C:\Users\Scott\Python\VMI\data\20240418_VMI3DPaperData2\213/"
paths = [loadpath + "213_20240207_141739/"]

# %% Load data, shift and trim

for i in range(len(paths)):
    a = np.array(np.load(paths[i] + "dataS.npy"))
    if undogridscale:
        b = np.load(paths[i] + "detmap.npy", allow_pickle=True).item()
        a = xycorrectpoints(a, b, inverse=True)
    if i == 0: pt = a
    else: pt = np.vstack((pt, a))

# shift tof to T0
pt[:, 2] -= tshift

# trim to time range
print("Before trim:", len(pt))
pt = pt[pt[:, 2] > tl[0]]
pt = pt[pt[:, 2] < tl[1]]
print("After trim:", len(pt))

# centre
pt[:, 0] -= cen[0]
pt[:, 1] -= cen[1]

# rotate
x2 = pt[:, 0]*np.cos(rotangle) + pt[:, 1]*np.sin(rotangle)
y2 = pt[:, 1]*np.cos(rotangle) - pt[:, 0]*np.sin(rotangle)
pt[:, 0] = x2
pt[:, 1] = y2

# save processed data
np.save(datapath + "data{}P_NC.npy".format(name), pt)

# %% Plots

# r vs ToF histogram
rs = np.sqrt(pt[:, 0]**2 + pt[:, 1]**2)
H, tb, rb = np.histogram2d(
    pt[:, 2], rs, bins=plotdim, range=((tl[0], tl[1]), (0.1, 215)))

fig = plt.figure()
plt.pcolor(tb[:-1], rb[:-1], H.T/rb[:-1, np.newaxis], vmax=0.3, cmap="Greys")
plt.xlabel("ToF (ns)")
plt.ylabel("Radius (pixels)")
plt.title("213+266 nm, 2-channel data")
plt.colorbar()

# %% r vs ToF covariance

tt = np.arange(pt[:, 2].min(), pt[:, 2].max(), 0.1)
rr = np.arange(0.3, np.sqrt(pt[:, 0]**2+pt[:, 1]**2).max(), 0.6)

# Scott's 3D background
if covshuffle:

    H, tb, rb = np.histogram2d(pt[:, 2], rs, bins=(
        len(tt), len(rr)), range=((tl[0], tl[1]), (0.1, 215)))
    ptuc = np.copy(pt)
    rd.shuffle(ptuc[:, 2])
    pthistU, tU, rU = np.histogram2d(ptuc[:, 2], rs, bins=(
        len(tt), len(rr)), range=((tl[0], tl[1]), (0.1, 215)))
    tmid = tU[:-1] + (tU[1]-tU[0])/2
    rmid = rU[:-1] + (rU[1]-rU[0])/2

    # for each hit, check (r, ToF) bin and store (correlated - uncorrelated) for that bin
    cov = np.zeros(len(pt))
    for i in range(len(pt)):
        if i % 10000 == 0:
            print("pt #{} / {}".format(i, len(pt)))
        argr = np.argmin(abs(np.sqrt(pt[i, 0]**2+pt[i, 1]**2)-rmid))
        argt = np.argmin(abs(pt[i, 2]-tmid))
        cov[i] = H[argt, argr] - pthistU[argt, argr]

    fig, axU = plt.subplots(1, 3)

    axU[0].pcolormesh(tmid, rmid, H.T/H.max(), cmap="seismic",
                      norm=LogNorm(vmax=1, vmin=0.001))
    axU[0].set_ylabel("Radius (pixels)", labelpad=0, fontsize=fs+2)
    axU[0].set_xlabel("Time of Flight (ns)", labelpad=0, fontsize=fs+2)
    axU[0].tick_params(labelsize=fs)
    axU[0].text(10, 220, "Correlated", color="w")
    axU[0].set_xlim([20, 40])
    axU[0].set_ylim([0, 190])

    axU[1].pcolormesh(tmid, rmid, pthistU.T/pthistU.max(),
                      cmap="seismic", norm=LogNorm(vmax=1, vmin=0.001))
    axU[1].set_xlabel("Time of Flight (ns)", labelpad=0, fontsize=fs+2)
    axU[1].tick_params(labelsize=fs)
    axU[1].text(10, 220, "Uncorrelated", color="w")
    axU[1].set_xlim([20, 40])
    axU[1].set_ylim([0, 190])

    axU[2].pcolormesh(tmid, rmid, (H-pthistU).T/(H-pthistU).max(),
                      cmap="seismic", norm=LogNorm(vmax=1, vmin=0.001))
    axU[2].set_xlabel("Time of Flight (ns)", labelpad=0, fontsize=fs+2)
    axU[2].tick_params(labelsize=fs)
    axU[2].text(10, 220, "Covariance", color="w")
    axU[2].set_xlim([20, 40])
    axU[2].set_ylim([0, 190])

else:

    Ytt = []
    Yrr = []

    m = pt[0, -1]  # first shot number
    l = pt[0, -2]  # first stack number
    T = np.zeros(len(tt))
    R = np.zeros(len(rr))
    for i in range(len(pt)):
        if i % 100000 == 0:
            print("pt #{} / {}".format(i, len(pt)))

        # if new shot:
        if pt[i, -1] != m or pt[i, -2] != l:
            # update Ytt and Yrr
            Ytt.append(T)
            Yrr.append(R)
            # reset T and R
            T = np.zeros(len(tt))
            R = np.zeros(len(rr))
            # advance shot/stack
            m = pt[i, -1]
            l = pt[i, -2]

        # indices of hit
        argt = np.argmin(abs(pt[i, 2]-tt))
        argr = np.argmin(abs(np.sqrt(pt[i, 0]**2+pt[i, 1]**2)-rr))

        # increment bins
        T[argt] += 1
        R[argr] += 1/(rr[argr])  # Jacobian

    # store last shot
    Yrr.append(R)
    Ytt.append(T)

    Yrr = np.array(Yrr)
    Ytt = np.array(Ytt)

    # correlated and uncorrelated data
    corrRT = np.zeros((len(rr), len(tt)))
    uncorrRT = np.zeros((len(rr), len(tt)))
    SxT = np.sum(Ytt, axis=0)
    SxR = np.sum(Yrr, axis=0)
    for i in range(len(rr)):
        if i % 10 == 0:
            print("rr #{} / {}".format(i, len(rr)))
        for j in range(len(tt)):
            uncorrRT[i, j] = SxR[i] * SxT[j] / (len(Ytt)*(len(Ytt)-1))
            corrRT[i, j] = np.sum(Ytt[:, j]*Yrr[:, i]) / (len(Ytt)-1)

    # for each hit, check (r, ToF) bin and store (correlated - uncorrelated) for that bin
    cov = []
    for i in range(len(pt)):
        if i % 10000 == 0:
            print("pt #{} / {}".format(i, len(pt)))
        argr = np.argmin(abs(np.sqrt(pt[i, 0]**2+pt[i, 1]**2)-rr))
        argt = np.argmin(abs(pt[i, 2]-tt))
        cov.append(corrRT[argr, argt] - uncorrRT[argr, argt])

    fig, axs = plt.subplots(2, 6, gridspec_kw={"hspace": 0.35, "left": 0.01, "right": 0.98, "bottom": 0.02,
                                               "height_ratios": [6, 1], "wspace": 0.42, "width_ratios": [2, 12, 12, 1, 12, 1]})

    (ax_a, ax_c, ax_e, ax_g, ax_i, ax_k), (ax_b, ax_d, ax_f, ax_h, ax_j, ax_l) = axs
    plt.gcf().set_size_inches(8.5, 2.5)

    ax_b.set_visible(False)
    ax_l.set_visible(False)
    ax_h.set_visible(False)
    ax_g.set_visible(False)
    ax_k.set_visible(False)

    ax_a.plot(np.average(Yrr, axis=0) / np.average(Yrr, axis=0).max(), rr)
    ax_a.set_yticklabels([])
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel("<Y>", labelpad=0, fontsize=fs+2)
    ax_a.tick_params(direction="in", labelsize=fs)
    ax_a.invert_xaxis()

    ax_d.plot(tt, np.average(Ytt, axis=0) / np.average(Ytt, axis=0).max())
    ax_d.set_ylim(0, 1)
    ax_d.set_xticklabels([])
    ax_d.set_ylabel("<X>", labelpad=0, fontsize=fs+2)
    ax_d.tick_params(direction="in", labelsize=fs)

    ax_f.plot(tt, np.average(Ytt, axis=0) / np.average(Ytt, axis=0).max())
    ax_f.set_ylim(0, 1)
    ax_f.set_xticklabels([])
    ax_f.tick_params(direction="in", labelsize=fs)

    ax_j.plot(tt, np.average(Ytt, axis=0) / np.average(Ytt, axis=0).max())
    ax_j.set_ylim(0, 1)
    ax_j.set_ylabel("<X>", labelpad=0, fontsize=fs+2)
    ax_j.set_xticklabels([])
    ax_j.tick_params(direction="in", labelsize=fs)

    c = ax_c.pcolormesh(tt, rr, corrRT/corrRT.max(),
                        cmap="seismic", norm=LogNorm(vmax=1, vmin=0.001))
    ax_c.set_ylabel("Radius (pixels)", labelpad=0, fontsize=fs+2)
    ax_c.set_xlabel("Time of Flight (ns)", labelpad=0, fontsize=fs+2)
    ax_c.tick_params(labelsize=fs)
    ax_c.text(10, 220, "Correlated", color="w")

    e = ax_e.pcolormesh(tt, rr, uncorrRT/uncorrRT.max(),
                        cmap="seismic", norm=LogNorm(vmax=1, vmin=0.001))
    ax_e.set_xlabel("Time of Flight (ns)", labelpad=0, fontsize=fs+2)
    ax_e.tick_params(labelsize=fs)
    ax_e.text(10, 220, "uncorrelated", color="w")

    cax1 = fig.add_axes([0.58, 0.255, 0.01, 0.625])
    cb1 = plt.colorbar(e, cax=cax1, orientation="vertical")
    cb1.ax.set_yticks([0, 1])
    cb1.ax.tick_params(labelsize=fs)
    cb1.ax.set_ylabel("<XY> and <X><Y> (norm.)", fontsize=fs+1, labelpad=0)

    i = ax_i.pcolormesh(tt, rr, corrRT-uncorrRT, cmap="seismic",
                        norm=LogNorm(vmax=0.001, vmin=0.00001))
    ax_i.set_xlabel("Time of Flight (ns)", labelpad=0, fontsize=fs+2)
    ax_i.set_ylabel("Radius (pixels)", labelpad=5, fontsize=fs+2)
    ax_i.tick_params(labelsize=fs)
    ax_i.text(10, 220, "Covariance", color="w")

    cax2 = fig.add_axes([0.917, 0.255, 0.01, 0.625])
    cb2 = plt.colorbar(i, cax=cax2, orientation="vertical")
    cb2.ax.tick_params(labelsize=fs)
    cb2.ax.set_ylabel("cov(X,Y) = <XY> - <X><Y>", fontsize=fs+1, labelpad=0)

# %% Covariance plots

plt.figure()
plt.subplot(121)
H, tb, rb = np.histogram2d(
    pt[:, 2], rs, bins=plotdim, range=((tl[0], tl[1]), (0.1, 215)))
plt.pcolor(tb[:-1], rb[:-1], H.T/rb[:-1, np.newaxis], vmax=0.15, cmap="Greys")
plt.title("Before 2D Covariance")
plt.xlabel("ToF (ns)")
plt.ylabel("Radius (pixels)")

# remove points in bins where cov<=0
cov = np.array(cov)
pt = pt[cov > 0]
rs = np.sqrt(pt[:, 0]**2 + pt[:, 1]**2)

plt.subplot(122)
H, tb, rb = np.histogram2d(
    pt[:, 2], rs, bins=plotdim, range=((tl[0], tl[1]), (0.1, 215)))
plt.pcolor(tb[:-1], rb[:-1], H.T/rb[:-1, np.newaxis], vmax=0.15, cmap="Greys")
plt.title("After 2D Covariance")
plt.xlabel("ToF (ns)")
plt.ylabel("Radius (pixels)")

# save covariance results
np.save(datapath + "data{}P.npy".format(name), pt)
np.savez(datapath + "data{}P.npz".format(name), H=H.T /
         rb[:-1, np.newaxis], t=tb[:-1], r=rb[:-1])

# %% Plots

dslicexy = 0.16  # width of centre time slice

plt.figure()
plt.hist2d(pt[:, 0], pt[:, 1], bins=300, range=([-200, 200], [-200, 200]))
plt.axhline(y=-171, ls="--", lw=0.75, c="w")
plt.axhline(y=171, ls="--", lw=0.75, c="w")
plt.axvline(x=0, ls="--", lw=0.75, c="w")
plt.axvline(x=-171, ls="--", lw=0.75, c="w")
plt.axvline(x=171, ls="--", lw=0.75, c="w")
plt.title("XY")

c1 = pt[:, 2] > -(dslicexy/2) + T0
c2 = pt[:, 2] < +(dslicexy/2) + T0
cond = c1 & c2
cenxy = pt[cond]
rc = rs[cond]

plt.figure()
plt.hist2d(cenxy[:, 0], cenxy[:, 1], 200, norm=LogNorm())
plt.axhline(y=-171, ls="--", lw=0.75, c="k")
plt.axhline(y=171, ls="--", lw=0.75, c="k")
plt.axvline(x=0, ls="--", lw=0.75, c="k")
plt.axvline(x=-171, ls="--", lw=0.75, c="k")
plt.axvline(x=171, ls="--", lw=0.75, c="k")
plt.title("XY Slice at T0")

# %% Find 2D radius

fitwidth = 15

# r histogram
ri, bins = np.histogram(rc, 256, (0, 256), density=True)
d = bins[1] - bins[0]
rbins = bins[:-1] + d/2
ri /= rbins

rmax = np.where(ri[50:] == max(ri[40:]))[0][0]
rmax += 50

fitrs = rbins[rmax-fitwidth: rmax+fitwidth]
fitri, params, pcov = fg(fitrs, ri[rmax-fitwidth:rmax+fitwidth],
                         0, np.max(ri), rbins[rmax], 5, czero=True)

plt.figure()
plt.plot(rbins, ri)
plt.plot(fitrs, fitri)
plt.plot(rbins[rmax], ri[rmax], "ko")
plt.plot(params[1], params[0], "bo")

rmax = params[1]

# %% Get K

E = ke/27.211
K = np.sqrt(2*E)/rmax
print("K:", K)

np.save(datapath + "K{}.npy".format(name), K)

# %% Trim radius, convert to pr, pz

print("Before trim:", len(pt))
pt = pt[rs < rmax*1.1]
rs = rs[rs < rmax*1.1]
print("After trim:", len(pt))

pr = rs*K

pz = np.sqrt(2*E - pr**2)
pz[np.isnan(pz)] = 0

tof = tb[:-2]

# trim rb to rmax
argr = np.argmin(abs(1.1*rmax-rb[:-1]))
R = rb[:-1][:argr]

pr = R*K  # pr axis
H = H[0:-1, :argr] / pr[:, np.newaxis].T

# %% Prepare for calibration fit

plotall = False
average = True
smooth = False

# errors
dfitt = 0.06  # a.u. 20% p0
dfitp = 80  # 10 points corresponds to 0.5 ns

nt = 50  # number of fits

# define time points to fit
tofmax = 4.5 + T0
tofmin = -3.83 + T0
tofslice = np.linspace(tofmin, tofmax, 2*nt)

# only consider p above this value
prmin = 0.05
pmax = prmin
llimin = np.argmin(abs(pr-prmin))

# fit the momentum for each tof
peaks = np.zeros((len(tofslice), 3))
for i in range(0, len(tofslice)):

    # extract time slice
    argt = np.argmin(abs(tofslice[i]-tof))
    
    if average==True:
        ai = np.average(H[argt-2:argt+3],axis=0)
    else:
        ai = H[argt]
    
    if smooth==True or pmax<0.25:
        rism = gaussian_filter(ai, 2)
    else:
        rism = ai

    # guess for maximum
    pmaxi = np.where(rism[llimin:] == max(rism[llimin:]))[0][0]
    pmax = pr[llimin:][pmaxi]

    # set fit limits
    if tofslice[i] > T0:  # for positive end, use previous fit as guess
        pmax = peaks[i-1, 0]
        pmaxi = np.argmin(abs(pmax-pr[llimin:]))
        uli = np.searchsorted(pr[llimin:], pmax+dfitt)
        #pmaxi = np.argmin(abs(pmax-pr))
        #uli = np.searchsorted(pr, pmax+0.2*dfitt)
    else:
        uli = np.searchsorted(pr[llimin:], pmax+dfitt)
        #uli = np.searchsorted(pr, pmax+dfitt)
    lli = np.searchsorted(pr[llimin:], pmax-dfitt)
    #lli = np.searchsorted(pr, pmax-dfitt)

    lli += llimin
    uli += llimin

    # redefine using limits
    pmaxi = np.where(rism[lli:uli] == max(rism[lli:uli]))[0][0]
    pmax = pr[lli:uli][pmaxi]

    # fitting
    fiti, params, pcov = fg(pr[lli:uli], rism[lli:uli],
                            0, rism[lli:uli][pmaxi], pmax, 0.01, czero=False)
    peaks[i, :2] = params[2], abs(params[3])

    if plotall:
        plt.figure()
        plt.plot(pr, ai, label='data')
        plt.plot(pr, rism, label='smoothed')
        plt.plot(peaks[i, 0], params[1]+params[0], "ko", label='peak')
        plt.plot(pr[lli:uli], fiti, label='fit')
        plt.title("time {}, vertical fit".format(tofslice[i]))
        plt.legend()

    # fit horizontal slice for error (gaussian width)
    argr = np.argmin(abs(peaks[i, 0]-pr))
    argmin = argt-dfitp
    argmax = argt+dfitp
    if argmin < 0:
        argmin = 0
    if argmax > len(tof)-1:
        argmax = -1
    
    if average==True:
        ai = np.average(H[argmin:argmax, argr-2:argr+3], axis=1)
        #ai = np.average(H[:, argr-1:argr+2], axis=1)
    else:
        ai = H[argmin:argmax, argr]

    if smooth==True or pr[argr]<0.25:
        rism = gaussian_filter(ai, 2)
    else:
        rism = ai

    tofi = tof[argmin:argmax]
    #rismi = rism[argmin:argmax]

    #fiti, params, pcov = fg(tofi, rismi, 0, rism[argr], tofslice[i], 0.03, czero=False)
    fiti, params, pcov = fg(tofi, rism, 0, H[argt,argr], tof[argt], 0.03, czero=False)
    peaks[i, 2] = abs(params[3])

    if plotall:
        plt.figure()
        plt.plot(tof, ai, label='data')
        plt.plot(tof, rism, label='smoothed')
        #plt.plot(tofi, rismi, label='fit region')
        plt.plot(params[2], params[1]+params[0], "ko", label='peak')
        plt.plot(tofi, fiti, label='fit')
        plt.title("time {}, horizontal fit".format(tofslice[i]))
        plt.legend()

# plot fit results and errors (gaussian width)
plt.figure()
plt.pcolor(tof, pr, H.T, norm=LogNorm(), cmap="Greys")
plt.errorbar(tofslice, peaks[:, 0], yerr=peaks[:, 1],
             xerr=peaks[:, 2], color='k', fmt="o", markersize=1, alpha=0.3)

# tof vs pr for fitting
pfit = np.vstack(([tofslice, peaks[:, 0]]))
np.savez(datapath + "prtof{}.npz".format(name),
         pfit=pfit, errx=peaks[:, 1], erry=peaks[:, 2])

# %% Calibration fit, tof vs pz

fitpolorder = 2

tofp = pfit[0]
prp = pfit[1]

# convert to pz
pzp = -1*np.sign(tofp-T0)*np.sqrt(2*E - prp**2)

# dpz = |pr/pz|*dpr or |pr/pz|*K*dr
err = abs(prp/pzp) * peaks[:, 1]

err[np.isnan(pzp)] = 0
err[err > 1] = 0
pzp[np.isnan(pzp)] = 0

fitpol = np.poly1d(np.polyfit(tofp, pzp, fitpolorder))

plt.figure()
plt.errorbar(tofp, pzp, yerr=err, xerr=peaks[:, 2], lw=1, c="r", label="Data")
plt.plot(tofp, fitpol(tofp), c="k", label="Fit")
plt.xlabel("ToF (ns)")
plt.ylabel("Pz (a.u.)")
leg = plt.gca().legend(fontsize=20)
plt.gca().tick_params(labelsize=15)
leg.set_draggable(1)

np.save(datapath + "caldata{}.npy".format(name), [tofp, pzp, err])
np.save(datapath + "calpoly{}.npy".format(name), fitpol)
np.save(datapath + "calpeaks{}.npy".format(name), peaks)
np.save(datapath + "calpfit{}.npy".format(name), pfit)

