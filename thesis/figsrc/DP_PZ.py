#%% Imports

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

K = 0.002265 # read this in instead
dToF = 0.1 # 100 ps
T0 = 29.5 # time centre

#%% Load data

path2 = "213/213_20240207_141739_old/dataAsP.npz"

pt = np.load("DP_PZ_213266_1213_902613638_dataAsP.npz")
H1 = pt["H"]
t = pt["t"]
r = pt["r"]

H2 = np.load("DP_PZ_213_0207_141739_old_dataAsP.npz")["H"]

fitdata1 = np.load("DP_PZ_213266_1213_902613638_prtofAs05.npz")
fitdata2 = np.load("DP_PZ_213266_1213_902613638_prtofAs88.npz")
fitdata3 = np.load("DP_PZ_213_0207_141739_old_prtofAs.npz")

# combine data sets
H = 0.1*H1 + H2

# load calibration 0.88 eV
fitpol = np.poly1d(np.load("DP_PZ_213266_1213_902613638_calpolyAs88.npy"))
tofp_ex, pzp_ex, err_ex = np.load("DP_PZ_213266_1213_902613638_caldataAs88.npy")

#%% filter out bad fits

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

#%% Processing

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

res1 = pzp1 - fitpol(pfit1[0]+0.1)
res2 = pzp2 - fitpol(pfit2[0])
res3 = pzp3 - fitpol(pfit3[0])

#%% Figure

mpl.rcParams['axes.linewidth'] = 2
udel = "\u2202"

fstitle = 20
fslabel = 15
fstick = 15
fsleg = 12
fslett = 20

fig = plt.figure(figsize=(16, 9), layout='tight')
ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 4), (1, 0), colspan=3)
ax3 = plt.subplot2grid((3, 4), (2, 0), colspan=3)
ax4 = plt.subplot2grid((3, 4), (1, 3))

fig.subplots_adjust(hspace=0, wspace=0.3)

ax1.pcolor(t-T0, r*K, 4*H, vmax=1, cmap="Greys")
ax1.plot(pfit1[0]-T0, pfit1[1], color='r', marker='o', markersize=8, ls="none", alpha=0.1, label="0.05 eV")
ax1.plot(pfit2[0]-T0, pfit2[1], color='g', marker='o', markersize=8, ls="none", alpha=0.1, label="0.88 eV")
ax1.plot(pfit3[0]-T0, pfit3[1], color='b', marker='o', markersize=8, ls="none", alpha=0.1, label="2.05 eV")
ax1.set_ylabel(r"$P_{\perp}$ (a.u.)", fontsize=fslabel, labelpad=1)
ax1.set_ylim(0, 0.42)
ax1.set_xlim(-4.2, 4.4)
ax1.legend(fontsize=fsleg, loc=4)

ax2.errorbar(pfit3[0]-T0, pzp3, yerr=err3, xerr=erry3, color='b', elinewidth=1, ls="none", label="2.05 eV")
ax2.errorbar(pfit2[0]-T0, pzp2, yerr=err2, xerr=erry2, color='g', elinewidth=1, ls="none", label="0.88 eV")
ax2.errorbar(pfit1[0]-T0+0.1, pzp1, yerr=err1, xerr=erry1, color='r', elinewidth=1, ls="none", label="0.05 eV")
ax2.plot(ToF-T0, fitpol(ToF), lw=3, c="gray", label="Fit to 0.88 eV")
ax2.set_ylabel(r"$P_{z}$ (a.u.)", fontsize=fslabel, labelpad=0)
ax2.set_xlim(-4.2, 4.4)
ax2.set_ylim(-0.4, 0.4)
ax2.legend(fontsize=fsleg, loc=3).draw_frame(False)

ax3.plot(pfit1[0]-T0+0.1, res1, color='r', lw=2, marker='.', markersize=6, label="0.05 eV")
ax3.plot(pfit2[0]-T0, res2, color='g', lw=2, marker='.', markersize=6, label="0.88 eV")
ax3.plot(pfit3[0]-T0, res3, color='b', lw=2, marker='.', markersize=6, label="2.05 eV")
ax3.set_xlim(-4.2, 4.4)
ax3.set_yticks([-0.01, 0, 0.01])
ax3.set_ylabel("Deviations (a.u.)", fontsize=fslabel, labelpad=0)
ax3.set_xlabel("Time, t (ns)", fontsize=fslabel, labelpad=0)
ax3.legend(fontsize=fsleg, loc=3).draw_frame(False)

ax4.plot(dPzdT, fitpol(ToF), lw=4, c="black")
ax4.set_xlabel(udel + r"$p_{z}/$" + udel + r"$t$ (a.u./ns)", fontsize=fslabel, labelpad=1)  
ax4.set_ylim(-0.4, 0.4)
ax4.set_xlim(0.055, 0.115)
ax4.tick_params(direction='in',  labelsize=fstick, pad=1)

col1 = 'black'
col2 = 'white'
pos1 = 0.01
pos2 = 0.95

for AX in [ax1, ax2, ax3, ax4]:
    AX.tick_params(direction='in', labelsize=fstick, length=6, width=1.5, pad=4)
t1 = ax1.text(pos1, pos2, '(a)', ha='left', va='top', transform=ax1.transAxes, fontsize=fslett)
t2 = ax2.text(pos1, pos2-0.08, '(b)', ha='left', va='top', transform=ax2.transAxes, fontsize=fslett)
t3 = ax3.text(pos1, pos2, '(c)', ha='left', va='top', transform=ax3.transAxes, fontsize=fslett)
t4 = ax4.text(pos1+0.04, pos2, '(d)', ha='left', va='top', transform=ax4.transAxes, fontsize=fslett)

t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t4.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

