#%% Imports

import os
os.chdir(r'../..')

import numpy as np
import matplotlib.pyplot as plt

# Plot constants
fs = 6
lw = 1
plt.rcParams['figure.dpi'] = 300

K = 0.002265 # read this in instead
dToF = 0.1 # 100 ps
T0 = 29.5 # time centre

datapath = r"C:\Users\Scott\Python\VMI\data\20240418_VMI3DPaperData2/"

#%% Load data

path1 = "213266/213266_20231213_160902_161613_162638/dataAsP.npz"  
path2 = "213/213_20240207_141739_old/dataAsP.npz"

pt = np.load(datapath + path1)
H1 = pt["H"]
t = pt["t"]
r = pt["r"]

H2 = np.load(datapath + path2)["H"]

fitdata1 = np.load(datapath + "213266/213266_20231213_160902_161613_162638/prtofAs05.npz")
fitdata2 = np.load(datapath + "213266/213266_20231213_160902_161613_162638/prtofAs88.npz")
fitdata3 = np.load(datapath + "213/213_20240207_141739_old/prtofAs.npz")

# combine data sets
H = 0.1*H1 + H2

# load calibration 0.88 eV
calpath = "213266/213266_20231213_160902_161613_162638/calpolyAs88.npy"
fitpol = np.poly1d(np.load(datapath + calpath))

calexpath = "213266/213266_20231213_160902_161613_162638/caldataAs88.npy"
tofp_ex, pzp_ex, err_ex = np.load(datapath + calexpath)

# filter out bad fits
errlim = 1

c1 = fitdata1["errx"]<errlim
c2 = fitdata1["erry"]<errlim
c = c1&c2
pfit1 = fitdata1["pfit"][:,c]
errx1 = fitdata1["errx"][c]
erry1 = fitdata1["erry"][c]

c1 = fitdata2["errx"]<errlim
c2 = fitdata2["erry"]<errlim
c = c1&c2
pfit2 = fitdata2["pfit"][:,c]
errx2 = fitdata2["errx"][c]
erry2 = fitdata2["erry"][c]

c1 = fitdata3["errx"][1:-3]<errlim
c2 = fitdata3["erry"][1:-3]<errlim
c = c1&c2
pfit3 = fitdata3["pfit"][:, 1:-3][:,c]
errx3 = fitdata3["errx"][1:-3][c]
erry3 = fitdata3["erry"][1:-3][c]

#%% Figure 2, Panel 1

fig = plt.figure()

ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 4), (1, 3))
ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=3)
ax4 = plt.subplot2grid((3, 4), (2, 0), colspan=3)

fig.subplots_adjust(hspace=0.2, wspace=0.3)

ax1.pcolor(t-T0, r*K, 4*H, vmax=1, cmap="Greys")

ax1.plot(pfit1[0]-T0, pfit1[1], color='r', marker='o', markersize=3, ls="none", alpha=0.1, label="0.05 eV")
ax1.plot(pfit2[0]-T0, pfit2[1], color='g', marker='o', markersize=3, ls="none", alpha=0.1, label="0.88 eV")
ax1.plot(pfit3[0]-T0, pfit3[1], color='b', marker='o', markersize=3, ls="none", alpha=0.1, label="2.05 eV")

ax1.text(.02, .95, '(a)', ha='left', va='top', transform=ax1.transAxes, fontsize=fs+2)

ax1.set_ylabel(r"$P_{\perp}$ (a.u.)", fontsize=fs, labelpad=1)
ax1.set_ylim(0, 0.42)
ax1.set_xlim(-4.2, 4.4)
ax1.legend(fontsize=fs-2, loc=4).draw_frame(False)

# ax1b = ax1.secondary_xaxis('top', functions=(lambda x: x-T0, lambda x: x+T0), zorder=3)
# ax1b.set_xlabel("Centered Time, t (ns)", fontsize=fs, labelpad=1)
# ax1b.tick_params(direction='in', labelsize=fs, pad=1)

#%% Figure 2, Panel 2

ToF = np.linspace(t[0], t[-1], 512)
dPzdT = abs(fitpol.deriv()(ToF))

# 0.054 eV
E1 = 0.054/27.211
tofp = pfit1[0] + 0.1 # 100 ps delay
prp = pfit1[1]
pzp1 = -1*np.sign(tofp-T0)*np.sqrt(2*E1 - prp**2)
err1 = abs(prp/pzp1) * errx1
err1[np.isnan(pzp1)] = 0
err1[err1>1] = 0
pzp1[np.isnan(pzp1)] = 0

# 0.878 eV
E2 = 0.878/27.211
tofp = pfit2[0]
prp = pfit2[1]
pzp2 = -1*np.sign(tofp-T0)*np.sqrt(2*E2 - prp**2)
err2 = abs(prp/pzp2) * errx2
err2[np.isnan(pzp2)] = 0
err2[err2>1] = 0
pzp2[np.isnan(pzp2)] = 0

# 2.045 eV
E3 = 2.045/27.211
tofp = pfit3[0]
prp = pfit3[1]
pzp3 = -1*np.sign(tofp-T0)*np.sqrt(2*E3 - prp**2)
err3 = abs(prp/pzp3) * errx3
err3[np.isnan(pzp3)] = 0
err3[err3>1] = 0
pzp3[np.isnan(pzp3)] = 0

# 0.88 eV calibration, extrapolated
ax3.errorbar(pfit3[0]-T0, pzp3, yerr=err3, xerr=erry3, color='b', elinewidth=0.5, ls="none", label="2.05 eV")
ax3.errorbar(pfit2[0]-T0, pzp2, yerr=err2, xerr=erry2, color='g', elinewidth=0.5, ls="none", label="0.88 eV")
ax3.errorbar(pfit1[0]-T0+0.1, pzp1, yerr=err1, xerr=erry1, color='r', elinewidth=0.5, ls="none", label="0.05 eV")
ax3.plot(ToF-T0, fitpol(ToF), lw=lw, c="gray", label="Fit to 0.88 eV")

ax3.text(.02, .83, '(b)', ha='left', va='top', transform=ax3.transAxes, fontsize=fs+2)

ax3.legend(fontsize=fs-2, loc=3).draw_frame(False)
ax3.set_ylabel(r"$P_{z}$ (a.u.)", fontsize=fs, labelpad=0)
ax3.set_xlim(-4.2, 4.4)
ax3.set_ylim(-0.4, 0.4)

# ax3b = ax3.secondary_xaxis('top', functions=(lambda x: x-T0, lambda x: x+T0))
# ax3b.tick_params(direction='in', labelsize=fs, pad=1)

#%% Figure 2, Panel 3

ax2.plot(dPzdT, fitpol(ToF), lw=2, c="black")

ax2.text(.05, .95, '(d)', ha='left', va='top', transform=ax2.transAxes, fontsize=fs+2)

udel = "\u2202"

ax2.set_xlabel(udel + r"$p_{z}/$" + udel + r"$t$ (a.u./ns)", fontsize=fs, labelpad=1) 
#ax2.set_ylabel(r"$P_{z}$ (a.u.)", fontsize=fs, labelpad=0)   
ax2.set_ylim(-0.4, 0.4)
ax2.set_xlim(0.055, 0.115)
ax2.tick_params(direction='in', length=2, width=0.5, labelsize=fs, pad=1)

for AX in [ax1, ax2, ax3, ax4]:
    AX.tick_params(direction='in', labelsize=fs, pad=1)

#%% Residuals

res1 = pzp1 - fitpol(pfit1[0]+0.1)
res2 = pzp2 - fitpol(pfit2[0])
res3 = pzp3 - fitpol(pfit3[0])

ax4.plot(pfit1[0]-T0+0.1, res1, color='r', lw=1, marker='.', markersize=2, label="0.05 eV")
ax4.plot(pfit2[0]-T0, res2, color='g', lw=1, marker='.', markersize=2, label="0.88 eV")
ax4.plot(pfit3[0]-T0, res3, color='b', lw=1, marker='.', markersize=2, label="2.05 eV")

ax4.set_xlim(-4.2, 4.4)
ax4.text(.02, .95, '(c)', ha='left', va='top', transform=ax4.transAxes, fontsize=fs+2)
ax4.legend(fontsize=fs-2, loc=3).draw_frame(False)
ax4.set_ylabel("Deviations (a.u.)", fontsize=fs, labelpad=0)
ax4.set_xlabel("Time, t (ns)", fontsize=fs, labelpad=0)

