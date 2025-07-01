#%% Imports

import os
os.chdir('..')

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.signal as ss
import scipy.interpolate as interpolate
import sgolay2

#%% Load

datapath = r"C:\Users\Scott\Python\VMI\data\20240418_VMI3DPaperData2\213\213_20240207_141739_old/"

a = 4 # rotation angle
path = "mpolAs000M.npz"
ring = 2

plot = True
smooth = 9
n0 = 5 # width of momentum slice
b2rough = 12 # (x10)

ke = [0.054, 0.88, 2.05][ring] # eV, energy of ring
keau = ke/27.211 # in a.u.

data = np.load(datapath + path)
K = np.load(datapath + "KAs.npy")
DCSSP = data['mpol']
phi = data['phi']
theta = data['theta']
mom = data['mom']

argp = np.argmin(abs(np.sqrt(2*keau)-mom))
PHI, THETA, PP = np.meshgrid(phi, theta, mom, indexing="ij")

#%% Slice

sm = np.sum(DCSSP[:,:,argp-n0:argp+n0+1], axis=2) / (2*n0+1)

if smooth > 0:
    # smooth the edges in phi
    sm[0] = ss.savgol_filter(sm[0], 11, 1)
    sm[1] = ss.savgol_filter(sm[1], 11, 1)
    sm[2] = ss.savgol_filter(sm[2], 11, 1)
    sm[-1] = ss.savgol_filter(sm[-1], 11, 1)
    sm[-2] = ss.savgol_filter(sm[-2], 11, 1)
    sm[-3] = ss.savgol_filter(sm[-3], 11, 1)
    
    # full smoothing
    sphere = sgolay2.SGolayFilter2(smooth, 1)(sm)
else:
    sphere = sm

if plot:
    fig = plt.figure()
    plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], 1.5*sphere/sphere.max(), cmap="seismic", vmax=1, vmin=0)
    plt.xlabel("Phi (deg)")
    plt.ylabel("Theta (deg)")
    cm = plt.colorbar()
    cm.set_label("Normalized Yield (arb)")

#%% Divide by the line theta = 90 deg

argt = np.argmin(abs(90-theta))
sphere = sphere/np.average(sphere[:,argt-2:argt+3], axis=1)[:,np.newaxis]

if plot:
    fig = plt.figure()
    plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], sphere, cmap="bone", vmax=3.0, vmin=0.1)
    plt.xlabel("Phi (deg)")
    plt.ylabel("Theta (deg)")
    cm = plt.colorbar()
    cm.set_label("Normalized Yield (arb)")

#%% load simulated distribution with the same tilt angle

corr = np.load(datapath + "../../mod_{}_{}.npz".format(a, b2rough))["corr"]

fig = plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], corr, cmap="bone")
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Normalized Yield (arb)")

# divide by simulation and smooth
if smooth > 0:
    corr = sgolay2.SGolayFilter2(11,1)(sphere/corr)
else:
    corr = (sphere/corr)

fig = plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], corr, cmap="ocean")
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Ratio")

argphi = np.argmin(np.absolute(105-phi))
argphi2 = np.argmin(np.absolute(75-phi))
argt = np.argmin(np.absolute(160-theta))
argt2 = np.argmin(np.absolute(200-theta))

plt.axvline(x=phi[argphi], c="C3", lw=2, ls="--")
plt.axvline(x=phi[argphi2], c="C3", lw=2, ls="--")
plt.axhline(y=theta[argt], c="C1", lw=2, ls="--")
plt.axhline(y=theta[argt2], c="C1", lw=2, ls="--")

np.savez(datapath + "correction.npz", corr=corr, phi=phi, theta=theta)

#%% extract the detector efficiency curve (theta, phi)

argtp1 = np.argmin(np.absolute(270-theta))
argtp2 = np.argmin(np.absolute(90-theta))

# theta from 160 to 200
phid = np.average(corr[:,argt:argt2+1], axis=1)

# phi from 75 to 105, theta 90 to 270
td = np.average(corr[argphi2:argphi+1,argtp2:argtp1+1], axis=0)

fig = plt.figure()
plt.subplot(211)
plt.plot(theta[argtp2:argtp1+1], td, c="C3", label="[Y] Theta: 90:270 [Averaged Phi: 75:105]")
plt.plot(phi+91, phid, c="C1", label="[X] Phi: 0:180 [Averaged Theta: 160:200]")
plt.xlabel("Angle (deg)")
plt.ylabel("Efficiency")
ax1 = plt.gca()
ax1.legend(loc=9).draw_frame(False)

# interpolate td and phid
angle = np.arange(phi.min()+90, phi.max()+90.5, 0.5)   
f1 = interpolate.interp1d(phi+90, phid) # now phi is 90:270?
y1 = f1(angle)
f2 = interpolate.interp1d(theta[argtp2:argtp1+1], td)
y2 = f2(angle)

# fit response to damping function
def damp(r, a, a0):
    return a0*np.exp(-r**(3/2)*a)
func0 = lambda x,r,y: (damp(r, x[0], x[1]) - y)**2

guess0 = [0.5, 1] # a, a0
ang2r = np.sqrt(2*keau)/K
rpix = abs(np.sin(np.radians(angle))) * ang2r # convert angle into radius (pixels)
args = (rpix, 0.5*(y1+y2)) # fit to average of both x and y response [why not flip to symmetrize as well?]
p0 = optimize.least_squares(func0, x0=guess0, args=args)

fig = plt.figure()
plt.subplot(211)
ang180 = np.argmin(abs(180-angle))
ang1 = abs(np.sin(np.radians(angle[ang180:])))
ang2 = abs(np.sin(np.radians(angle[:ang180+1])))
plt.plot(ang1*ang2r, damp(ang1*ang2r, *p0.x), c="k", ls="--", label="Damping Function 1")
plt.plot(ang2*ang2r, 0.5*(y1+y2)[:ang180+1], c="C0", label="X,Y > 0")
plt.plot(ang1*ang2r, 0.5*(y1+y2)[ang180:], c="C4", label="X,Y < 0")
plt.xlabel("Radius (pixels)")
plt.ylabel("Efficiency")
plt.legend().draw_frame(False)
plt.xlim(0, 172)

# save this one
rint = np.linspace(0.1, 174, 10000)
np.savez(datapath + "../../radial_correction_4_12_205.npz", Rcorr=damp(rint, *p0.x), R=rint)

# try a different damping function?
def damp(r, a, a0):
    return a0*np.exp(-r*a)     
func0 = lambda x,r,y: (damp(r, x[0], x[1]) - y)**2

guess0 = [0.5, 1]
p0 = optimize.least_squares(func0, x0=guess0, args=(rpix, 0.5*(y1+y2)))

plt.subplot(212)
plt.plot(ang2*ang2r, damp(ang2*ang2r, *p0.x), c="k", ls="--", label="Damping Function 2")
plt.plot(ang2*ang2r, 0.5*(y1+y2)[:ang180+1], c="C0", label="X,Y > 0")
plt.plot(ang1*ang2r, 0.5*(y1+y2)[ang180:], c="C4", label="X,Y < 0")
plt.xlabel("Radius (pixels)")
plt.ylabel("Efficiency")
plt.legend().draw_frame(False)
plt.xlim(0, 172)

#%% convert to xy coordinates and plot (non essential)

X0 = np.linspace(-200, 200, 100)
Y0 = np.linspace(-200, 200, 100)

XY = np.zeros((len(X0), len(Y0)))
for i in range(len(phi)):
    for j in range(len(theta)):
        x0 = np.argmin(abs(X0 - ang2r*np.sin(np.radians(phi[i]))*np.cos(np.radians(theta[j]))))
        y0 = np.argmin(abs(Y0 - ang2r*np.cos(np.radians(phi[i]))))
        XY[x0,y0] += abs(np.sin(np.radians(phi[i]))) * corr[i,j]

plt.figure()
plt.pcolor(X0, Y0, XY, cmap="ocean")
plt.xlabel("X (pix.)")
plt.ylabel("Y (pix.)")

XY = sgolay2.SGolayFilter2(15, 2)(XY)

plt.figure()
plt.pcolor(X0, Y0, XY,cmap="ocean")
plt.xlabel("X (pix.)")
plt.ylabel("Y (pix.)")

arg0 = np.argmin(abs(X0))
plt.figure()
plt.plot(X0, XY[arg0], label="X=0")
plt.plot(X0, XY[:,arg0], label="X=0")