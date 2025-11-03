#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy import optimize
import sgolay2

def legendre(x, beta2, beta4, a, ang):
    return a * (1 + beta2*0.5*(3*np.cos(np.radians(x-ang))**2 - 1) +
                beta4*(1/8)*(35*np.cos(np.radians(x-ang))**4 - 30*np.cos(np.radians(x-ang))**2 + 3))

#%% load data

data = np.load("DP_AngCorrect_213266_902613638_mpolAs000_NoRot.npz")
mom = data['mom']
theta = data['theta']
phi = data['phi']
DCSSP = data['mpol']

ke = 0.88
keau = ke/27.211 # in a.u.

n0 = 5 # width of momentum slice
nth = 5 # width of theta slices
smooth = 0
plot = False

#%% Processing

argp = np.argmin(abs(np.sqrt(2*keau)-mom))
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

#%% fit phi distribution to Legendre polynomials

arg90 = np.argmin(abs(theta-90))
arg180 = np.argmin(abs(theta-180))
arg270 = np.argmin(abs(theta-270))

# average over theta
phi = np.hstack((phi, phi+180))
ABExy = np.hstack((np.average(sphere[:, arg90-nth:arg90+nth+1], axis=1),
                   np.average(sphere[:, arg270-nth:arg270+nth+1], axis=1)[::-1]))
ABExz = np.hstack(((np.average(sphere[:, :nth+1], axis=1) + np.average(sphere[:, -nth:], axis=1))/2,
                   np.average(sphere[:, arg180-nth:arg180+nth+1], axis=1)[::-1]))

# fit xy
optimxy = optimize.curve_fit(legendre, phi, ABExy/ABExy.max(), p0=[1.6, 0, 1, 28])
b2 = optimxy[0][0]
b4 = optimxy[0][1]
afit = optimxy[0][2]
angfitxy = optimxy[0][3]
errs = np.sqrt(np.diag(optimxy[1]))
db2 = errs[0]
db4 = errs[1]
dangxy = errs[3]

# fit xz
optimxz = optimize.curve_fit(legendre, phi, ABExz/ABExz.max(), p0=[1.6, 0, 1, 5])
b2 = optimxz[0][0]
b4 = optimxz[0][1]
afit = optimxz[0][2]
angfitxz = optimxz[0][3]
errs = np.sqrt(np.diag(optimxz[1]))
db2 = errs[0]
db4 = errs[1]
dangxz = errs[3]

#%% Figure

FSt = 25
FSlb = 20
FSlg = 15
FStk = 15

fig = plt.figure()
fig.set_tight_layout(True)
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

ax1.plot(phi, legendre(phi, *optimxy[0]), ls="--", c="b", label=r"Fit")
ax1.plot(phi, ABExy/ABExy.max(), lw=0, marker='x', color='red', label='Data')
ax1.vlines(180, 0, 1, ls='--', lw=1, color='black', label="180\N{DEGREE SIGN}")
ax1.fill_between([180+angfitxy-dangxy, 180+angfitxy+dangxy], 0, 1, color='black', alpha=0.5,
                 label="$\chi_{z}$ = %2.1f$\N{DEGREE SIGN}\pm$%2.1f\N{DEGREE SIGN}"%(angfitxy, dangxy))
ax1.set_ylabel("Yield (normalized)", fontsize=FSlb)
ax1.set_ylim([0, 1.1])
ax1.set_xlabel("$\\theta_{xy}$ (\N{DEGREE SIGN})", fontsize=FSlb)
ax1.set_title("z-Axis Angular Correction (XY Slice)", fontsize=FSt)
ax1.legend(loc=1, fontsize=FSlg)
ax1.tick_params(labelsize=FStk)

ax2.plot(phi, legendre(phi, *optimxz[0]), ls="--", c="b", label=r"Fit")
ax2.plot(phi, ABExz/ABExz.max(), lw=0, marker='x', color='red', label='Data')
ax2.vlines(180, 0, 1, ls='--', lw=1, color='black', label="180\N{DEGREE SIGN}")
ax2.fill_between([180+angfitxz-dangxy, 180+angfitxz+dangxy], 0, 1, color='black', alpha=0.5,
                 label="$\chi_{y}$ = %2.1f$\N{DEGREE SIGN}\pm$%2.1f\N{DEGREE SIGN}"%(angfitxz, dangxy))
ax2.set_ylabel("Yield (normalized)", fontsize=FSlb)
ax2.set_ylim([0, 1.1])
ax2.set_xlabel("$\\theta_{xz}$ (\N{DEGREE SIGN})", fontsize=FSlb)
ax2.set_title("y-Axis Angular Correction (XZ Slice)", fontsize=FSt)
ax2.legend(loc=1, fontsize=FSlg)
ax2.tick_params(labelsize=FStk)

col1 = 'black'
col2 = 'white'
pos1 = 0.02
pos2 = 0.98
fslett = 20

t1 = ax1.text(pos1, pos2, '(a)', ha='left', va='top', transform=ax1.transAxes, fontsize=fslett)
t2 = ax2.text(pos1, pos2, '(b)', ha='left', va='top', transform=ax2.transAxes, fontsize=fslett)

t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))