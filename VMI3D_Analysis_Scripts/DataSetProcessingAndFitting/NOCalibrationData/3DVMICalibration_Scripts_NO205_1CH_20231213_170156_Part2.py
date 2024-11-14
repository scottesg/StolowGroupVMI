#%% Imports

import os
os.chdir(r'C:\Users\Scott\Python\VMI\src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import optimize
import scipy.signal as ss
import sgolay2

# Constants

datapath = r"C:\Users\Scott\Python\VMI\data\20240418_VMI3DPaperData2\213/213_20231213_170156/"
ke = 2.05 # eV, energy of ring
keau = ke/27.211 # in a.u.
pol = 'H' # polarization direction (H: along x axis)
dToF = 0.100 # 50 ps time resolution

name = "Ssu"

fs = 7 # fontsize
plt.rcParams['figure.dpi'] = 100

path = "data{}P.npy".format(name)
calpath = "../../calpolyNO88.npy"
Kpath = "K{}.npy".format(name)

#%% Load

pt = np.load(datapath + path)
calpoly = np.poly1d(np.load(datapath + calpath))
K = np.load(datapath + Kpath)

dPx = K/2 # momentum resolution (1 pixel * K)

#%% 3D momentum map and slices

# note x and y are reversed

# define momentum axes
Px = np.linspace(-np.sqrt(2*1.1*keau), np.sqrt(2*1.1*keau), int(np.sqrt(2*1.1*keau)/dPx))
if len(Px)%2==0: # make length odd
    Px = np.linspace(-np.sqrt(2*1.1*keau), np.sqrt(2*1.1*keau), int(np.sqrt(2*1.1*keau)/dPx)+1)
Py = np.copy(Px)
Pz = np.copy(Px)    

n = len(Px)

# momentum arrays
# transverse is the plane through the node of the p-wave (depends on polarization)
Pxyz = np.zeros((n,n,n))
Pt = np.linspace(0, abs(Px).max(), int(0.5*n)+1)

# errors
dPl = np.zeros(n)
dPt = np.zeros(len(Pt))

# counters
Cz = np.zeros(n)
Ct = np.zeros(len(Pt))

for i in range(len(pt)):
    if i % 10000 == 0: print("pt #{} / {}".format(i, len(pt)))
    
    # conver to momentum
    px = K*pt[i,0]
    py = K*pt[i,1]
    pz = calpoly(pt[i,2])
    
    # check if it falls inside the grid
    if (abs(pz) < abs(Pz).max() + dPx*0.5) and (abs(px) < abs(Px).max() + dPx*0.5) and (abs(py) < abs(Px).max() + dPx*0.5):
        
        argx = np.argmin(np.absolute(px-Px))
        argy = np.argmin(np.absolute(py-Py))
        argz = np.argmin(np.absolute(pz-Pz))

        if pol=='H':
            
            # transverse momentum
            argt = np.argmin(np.absolute(np.sqrt(px**2+pz**2) - Pt))
            
            # (Px*dPx + Pz*dPz) / (Pt+[half step])
            # *Q* What is the 0.5*Pt[1] for?
            dPt[argt] += (abs(Px[argx])*dPx + abs(Pz[argz]*calpoly.deriv()(pt[i,2]))*dToF) / (Pt[argt]+0.5*Pt[1])
            
        elif pol=='V':
            
            # transverse momentum
            argt = np.argmin(np.absolute(np.sqrt(px**2+py**2) - Pt))
            
            # (Px*dPx + Py*dPy) / (Pt+[half step])?
            # *Q* What is the 0.5*Pt[1] for?
            dPt[argt] += (abs(Px[argx]) + abs(Py[argy]))*dPx / (Pt[argt]+0.5*Pt[1])
            
        else:
            print("Not Supported!")
        
        # longitudinal (tof axis) error
        dPl[argz] += abs(calpoly.deriv()(pt[i,2]))*dToF 
        Cz[argz] += 1
        Ct[argt] += 1
        
        Pxyz[argx,argy,argz] += 1

# average errors
dPl/=Cz   
dPt/=Ct

# dPl = f(Pz), for later
f = np.poly1d(np.polyfit(Pz, dPl, 3))

# slice around Px=0
argp = np.argmin(np.absolute(0-Px))
pslice = np.average(Pxyz[argp-4:argp+5], axis=0)

plt.figure()
plt.subplot(131)
plt.pcolor(Pz, Py, pslice, cmap="Greys", vmax=pslice.max()/2)
plt.axhline(y=0, c="m", lw=1)
plt.axvline(x=0, c="m", lw=1)
plt.xlabel("Pz (a.u.)")
plt.ylabel("Px (a.u.)") # x and y are reversed

# slice around Py=0
pslice = np.average(Pxyz[:,argp-4:argp+5], axis=1)

plt.subplot(132)
plt.pcolor(Pz, Px, pslice, cmap="Greys", vmax=pslice.max()/2)
plt.axhline(y=0, c="m", lw=1)
plt.axvline(x=0, c="m", lw=1)
plt.xlabel("Pz (a.u.)")
plt.ylabel("Py (a.u.)") # x and y are reversed

# slice around Pz=0
pslice = np.average(Pxyz[:,:,argp-4:argp+5],axis=2)

plt.subplot(133)
plt.pcolor(Px, Py, pslice.T, cmap="Greys", vmax=pslice.max()/2)
plt.axhline(y=0, c="m", lw=1)
plt.axvline(x=0, c="m", lw=1)
plt.xlabel("Py (a.u.)") # x and y are reversed
plt.ylabel("Px (a.u.)")

np.savez(datapath + "mcart{}.npz".format(name), Pxyz=Pxyz, Px=Px, Py=Py, Pz=Pz)

#%% Polar plots of energy and momentum, without r correction

mom = np.linspace(dPx, abs(Pz).max(), int(0.5*n)+1)
E = np.linspace(0.5*dPx**2, 0.5*mom.max()**2, int(0.5*n)+1)  
width = mom[1] - mom[0]

# for H polarization
# defining phi [0, pi], x-r angle
# defining theta [0, 2pi], (-z)-ryz angle
# note x and y are reversed

theta = np.linspace(0, 360, 181)
phi = np.linspace(1, 179, 90) # not including 0 and 180 to avoid discontinuity

DCSSE = np.zeros((len(phi), len(theta), len(E))) # polar energy
DCSSP = np.zeros((len(phi), len(theta), len(mom))) # polar momentum

PHI, THETA, PP = np.meshgrid(phi, theta, mom, indexing="ij")
PHI, THETA, EE = np.meshgrid(phi, theta, E, indexing="ij")

# rotation to correct for laser pointing angle
# a(Rx), b(Ry), c(Rz)
a, b, c = 1.3, 0, 0
ca = np.cos(a*np.pi/180)
sa = np.sin(a*np.pi/180)
cb = np.cos(b*np.pi/180)
sb = np.sin(b*np.pi/180)
cc = np.cos(c*np.pi/180)
sc = np.sin(c*np.pi/180)

for i in range(len(pt)):
    if i % 10000 == 0: print("pt #{} / {}".format(i, len(pt)))
    
    px = K*pt[i,0]
    py = K*pt[i,1]
    pz = calpoly(pt[i,2])
    pf = np.sqrt(pz**2 + px**2 + py**2)
    
    # check if it falls inside the grid
    if pf < mom.max() + 0.5*width:
        
        # apply rotation
        px2 = px*cb*cc + py*(sa*sb*cc-ca*sc) + pz*(ca*sb*cc+sa*sc)
        py2 = px*cb*sc + py*(sa*sb*sc+ca*cc) + pz*(ca*sb*sc-sa*cc)
        pz2 = -px*sb + py*sa*cb + pz*ca*cb
        
        if pz2==0: pz2 = 0.5*mom[1]
        if py2==0: py2 = 0.5*mom[1]
        if px2==0: px2 = 0.5*mom[1]
       
        px = px2
        py = py2
        pz = pz2 
       
        if pol=='H':
            
            thetai = np.rad2deg(np.sign(px) * np.arccos(pz/np.sqrt(pz**2+px**2))) + 180
            phii = np.rad2deg(np.arccos(py/pf))

        elif pol=='V':

            thetai = np.rad2deg(np.sign(py) * np.arccos(px/np.sqrt(py**2+px**2))) + 180
            phii = np.rad2deg(np.arccos(pz/pf))               
            
        else:
            print("Not Supported!")
            
        argz = np.argmin(np.absolute(pf-mom))
        arge = np.argmin(np.absolute(0.5*pf**2-E))
        argth = np.argmin(np.absolute(thetai-theta))
        argph = np.argmin(np.absolute(phii-phi))
 
        DCSSP[argph, argth, argz] += 1
        DCSSE[argph, argth, arge] += 1

# apply Jacobian from 3D cartesian to spherical
DCSSP /= abs(PP*PP*np.sin(np.radians(PHI)))
DCSSE /= np.sqrt(2*EE)*abs(np.sin(np.radians(PHI)))

# since the domain of arcos is [0, Pi] it has half chance to fall in first or last theta bin
# DCSSP[:,0,:] += DCSSP[:,-1,:]
# DCSSP[:,-1,:] = DCSSP[:,0,:]
# DCSSE[:,0,:] += DCSSE[:,-1,:]
# DCSSE[:,-1,:] = DCSSE[:,0,:]

arge = np.argmin(abs(keau-E))
argp = np.argmin(abs(np.sqrt(2*keau)-mom))

# energy slice
plt.figure()
plt.pcolor(PHI[:,:,arge], THETA[:,:,arge], np.sum(DCSSE[:,:,arge-4:arge+5], axis=2)/9, cmap="seismic", norm=LogNorm())
cm = plt.colorbar()

# momentum slice
plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], np.sum(DCSSP[:,:,argp-4:argp+5], axis=2)/9, cmap="seismic", norm=LogNorm())
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

abc = "{}{}{}{}".format(name, a, b, c)
np.savez(datapath + "mpol{}.npz".format(abc), mpol=DCSSP, phi=phi, theta=theta, mom=mom)

#%% Mirroring (using only front end - x deg on the +y side)

# theta indices
argt = np.argmin(abs(theta-90))
argtm = np.argmin(abs(theta-180))
argtp = np.argmin(abs(theta-250))
argtp2 = np.argmin(abs(theta-110))

DCSSP1 = DCSSP.copy()
DCSSE1 = DCSSE.copy()

DCSSP1[:,:argt+1] = DCSSP1[:,argt:argtm+1][::-1,::-1]
DCSSP1[:,argtp:] = DCSSP1[:,:argtp2+1][:,::-1]

DCSSE1[:,:argt+1] = DCSSE1[:,argt:argtm+1][::-1,::-1]
DCSSE1[:,argtp:] = DCSSE1[:,:argtp2+1][:,::-1]

DCSSP = DCSSP1.copy()
DCSSE = DCSSE1.copy()

# energy slice
plt.figure()
plt.pcolor(PHI[:,:,arge], THETA[:,:,arge], np.sum(DCSSE[:,:,arge-3:arge+4], axis=2)/7, cmap="seismic", norm=LogNorm())
cm = plt.colorbar()

# momentum slice
plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], np.sum(DCSSP[:,:,argp-3:argp+4], axis=2)/7, cmap="seismic", norm=LogNorm())
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

np.savez(datapath + "mpol{}M.npz".format(abc), mpol=DCSSP, phi=phi, theta=theta, mom=mom)

#%% smooth the collected data

n0 = 2 # width of momentum slice
sm = np.sum(DCSSP[:,:,argp-n0:argp+n0+1], axis=2) / (2*n0+1)

# smooth the edges in phi
sm[0] = ss.savgol_filter(sm[0], 11, 1)
sm[1] = ss.savgol_filter(sm[1], 11, 1)
sm[2] = ss.savgol_filter(sm[2], 11, 1)
sm[-1] = ss.savgol_filter(sm[-1], 11, 1)
sm[-2] = ss.savgol_filter(sm[-2], 11, 1)
sm[-3] = ss.savgol_filter(sm[-3], 11, 1)

# full smoothing
sphere = sgolay2.SGolayFilter2(7, 1)(sm)

fig = plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], 1.5*sphere/sphere.max(), cmap="seismic", vmax=1, vmin=0)
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Normalized Yield (arb)")

np.savez(datapath + "mpol{}MInt.npz".format(abc), sphere=sphere, phi=phi, theta=theta)

#%% fit phi distribution to Legendre polynomials (theta averaged)

# average over theta and smooth
ABE = np.average(sphere, axis=1)
f1 = ss.savgol_filter(ABE, 3, 1)

fig = plt.figure()
plt.fill_between(phi, y1=(f1-np.sqrt(f1))/f1.max(), y2=(f1+np.sqrt(f1))/f1.max(), color="k", alpha=0.3)
plt.ylabel("Yield (normalized)")
plt.xlabel("Phi (deg)")

def legendre(x, beta2, beta4, a):
    return a * (1 + beta2*0.5*(3*x**2 - 1) + beta4*(1/8)*(35*x**4 - 30*x**2 + 3))

# fit
optim = optimize.curve_fit(legendre, np.cos(np.radians(phi)), f1/f1.max(), p0=[1.6, 0, 1], sigma=np.sqrt(f1)/f1.max())
b2 = optim[0][0]
b4 = optim[0][1]
afit = optim[0][2]

# plot beta and upper and lower limits from fit
plt.plot(phi, legendre(np.cos(np.radians(phi)), *optim[0]), ls="--", c="b",
         label=r"$\beta_{2;2.05 eV}=$%.2f, $\beta_{4;2.05 eV}=$%.2f"%(b2, b4))
plt.legend(loc=9)

#%% Gain Correction (Scott Version)

# correction = np.load(datapath + "correction.npz")
# corr = correction['corr']
# phicorr = correction['phi']
# thetacorr = correction['theta']

corrS = False

#%% apply phosphor gain and tilt corrections

# load phosphor correction
if not corrS:
    f = np.load(datapath + "../../radial_correction_13.npz")
    eff = f["Rcorr"]
    r = f["R"]

DCSSE = np.zeros((len(phi), len(theta), len(E))) # polar energy
DCSSP = np.zeros((len(phi), len(theta), len(mom))) # polar momentum

PHI, THETA, PP = np.meshgrid(phi, theta, mom, indexing="ij")
PHI, THETA, EE = np.meshgrid(phi, theta, E, indexing="ij")

# rotation to correct for laser pointing angle
# a(Rx), b(Ry), c(Rz)
#a, b, c = 10, 0, 0
ca = np.cos(a*np.pi/180)
sa = np.sin(a*np.pi/180)
cb = np.cos(b*np.pi/180)
sb = np.sin(b*np.pi/180)
cc = np.cos(c*np.pi/180)
sc = np.sin(c*np.pi/180)

for i in range(len(pt)):
    if i % 10000 == 0: print("pt #{} / {}".format(i, len(pt)))
    
    px = K*pt[i,0]
    py = K*pt[i,1]
    ri = np.sqrt(pt[i,0]**2 + pt[i,1]**2)
    
    if not corrS:
        argy = np.argmin(abs(r - abs(pt[i,1])))
        argx = np.argmin(abs(r - abs(pt[i,0])))
    
    pz = calpoly(pt[i,2])
    pf = np.sqrt(pz**2 + px**2 + py**2)

    # check if it falls inside the grid
    if pf < mom.max() + 0.5*width:
        
        # apply rotation
        px2 = px*cb*cc + py*(sa*sb*cc-ca*sc) + pz*(ca*sb*cc+sa*sc)
        py2 = px*cb*sc + py*(sa*sb*sc+ca*cc) + pz*(ca*sb*sc-sa*cc)
        pz2 = -px*sb + py*sa*cb + pz*ca*cb
        
        if pz2==0: pz2 = 0.5*mom[1]
        if py2==0: py2 = 0.5*mom[1]
        if px2==0: px2 = 0.5*mom[1]
        
        px = px2
        py = py2
        pz = pz2
        
        if pol=='H':
            
            thetai = np.rad2deg(np.sign(px) * np.arccos(pz/np.sqrt(pz**2+px**2))) + 180
            phii = np.rad2deg(np.arccos(py/pf))

        elif pol=='V':

            thetai = np.rad2deg(np.sign(py) * np.arccos(px/np.sqrt(py**2+px**2))) + 180
            phii = np.rad2deg(np.arccos(pz/pf))               
            
        else:
            print("Not Supported!")
            
        argz = np.argmin(np.absolute(pf-mom))
        arge = np.argmin(np.absolute(0.5*pf**2-E))
        argth = np.argmin(np.absolute(thetai-theta))
        argph = np.argmin(np.absolute(phii-phi))
 
        # if corrS:
        #     DCSSP[argph, argth, argz] += 1/corr[argph, argth]
        #     DCSSE[argph, argth, arge] += 1/corr[argph, argth]
        # else:
        DCSSP[argph, argth, argz] += 1/(eff[argx]*eff[argy])
        DCSSE[argph, argth, arge] += 1/(eff[argx]*eff[argy])

# apply Jacobian from 3D cartesian to spherical
DCSSP /= abs(PP*PP*np.sin(np.radians(PHI)))
DCSSE /= np.sqrt(2*EE)*abs(np.sin(np.radians(PHI)))

# DCSSP[:,0,:] += DCSSP[:,-1,:]
# DCSSP[:,-1,:] = DCSSP[:,0,:]
                                  
# DCSSE[:,0,:] += DCSSE[:,-1,:]
# DCSSE[:,-1,:] = DCSSE[:,0,:]

arge = np.argmin(abs(keau-E))
argp = np.argmin(abs(np.sqrt(2*keau)-mom))

plt.figure()
plt.pcolor(PHI[:,:,arge], THETA[:,:,arge], np.sum(DCSSE[:,:,arge-4:arge+5], axis=2)/9, cmap="seismic", norm=LogNorm())
cm = plt.colorbar()

plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], np.sum(DCSSP[:,:,argp-4:argp+5], axis=2)/9, cmap="seismic", norm=LogNorm())
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

abc = "{}{}{}{}".format(name, a, b, c)
np.savez(datapath + "mpol{}rc.npz".format(abc), mpol=DCSSP, phi=phi, theta=theta, mom=mom)

# rcode = "rc"
# if corrS: rcode = "rcS"
# np.savez(datapath + "mpolA{}{}.npz".format(abc, rcode), mpol=DCSSP, phi=phi, theta=theta, mom=mom)

#%% Mirroring (using only front end - x deg on the +y side)

# theta indices
argt = np.argmin(abs(theta-90))
argtm = np.argmin(abs(theta-180))
argtp = np.argmin(abs(theta-250))
argtp2 = np.argmin(abs(theta-110))

DCSSP1 = DCSSP.copy()
DCSSE1 = DCSSE.copy()

DCSSP1[:,:argt+1] = DCSSP1[:,argt:argtm+1][::-1,::-1]
DCSSP1[:,argtp:] = DCSSP1[:,:argtp2+1][:,::-1]

DCSSE1[:,:argt+1] = DCSSE1[:,argt:argtm+1][::-1,::-1]
DCSSE1[:,argtp:] = DCSSE1[:,:argtp2+1][:,::-1]

DCSSP = DCSSP1.copy()
DCSSE = DCSSE1.copy()
             
# energy slice
plt.figure()
plt.pcolor(PHI[:,:,arge], THETA[:,:,arge], np.sum(DCSSE[:,:,arge-3:arge+4], axis=2)/7, cmap="seismic", norm=LogNorm())
cm = plt.colorbar()

# momentum slice
plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], np.sum(DCSSP[:,:,argp-3:argp+4], axis=2)/7, cmap="seismic", norm=LogNorm())
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

np.savez(datapath + "mpol{}rcM.npz".format(abc), mpol=DCSSP, phi=phi, theta=theta, mom=mom)
#np.savez(datapath + "mpolA{}{}M.npz".format(abc, rcode), mpol=DCSSP, phi=phi, theta=theta, mom=mom)

#%% smooth the collected data

n0 = 2 # width of momentum slice
sm = np.sum(DCSSP[:,:,argp-n0:argp+n0+1], axis=2) / (2*n0+1)

# smooth the edges in phi
sm[0] = ss.savgol_filter(sm[0], 5, 1)
sm[1] = ss.savgol_filter(sm[1], 5, 1)
sm[2] = ss.savgol_filter(sm[2], 5, 1)
sm[-1] = ss.savgol_filter(sm[-1], 5, 1)
sm[-2] = ss.savgol_filter(sm[-2], 5, 1)
sm[-3] = ss.savgol_filter(sm[-3], 5, 1)

# full smoothing
sphere = sgolay2.SGolayFilter2(7, 1)(sm)

fig = plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], 2*sphere/sphere.max(), cmap="seismic", vmax=1, vmin=0)
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Normalized Yield (arb)")

np.savez(datapath + "mpol{}rcMInt.npz".format(abc), sphere=sphere, phi=phi, theta=theta)
#np.savez(datapath + "mpolA{}{}MInt.npz".format(abc, rcode), sphere=sphere, phi=phi, theta=theta)
