#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy import optimize
import sgolay2
from matplotlib.colors import LogNorm
from VMI3D_PADSIM import padsim

# simulation parameters

pol = "H" # H (horizontal) or V (vertical), for file name
eref = 2.05 # energy of signal (eV)
eau = eref/27.211
pref = np.sqrt(2*eau) # equivalent momentum (a.u.)
dp = 0.01 * pref # expected bandwith
beta = 1.2 # beta to simulate
load = True # loads pre-generated simulated distribution
name = "sim_12_205.npz"

a = -4 # rotation angle (deg)
n0 = 5 # width of momentum slice

datapath = r"../thesis/data/"

#%% Step 1: Generate or load simulated momentum distribution

#np.random.seed(123) # for consistant results

N = 1e7 # number of points to simulate

if load:
    P = np.load(datapath + name)["P"]
else:
    P = padsim(N, pref, dp, beta)
    np.savez(datapath + name, P=P)

plt.figure()
plt.hist2d(P[0], P[1], bins=2*int(pref/dp),range=[[-pref*1.2, pref*1.2], [-pref*1.2, pref*1.2]])
plt.xlabel("Px (a.u.)")
plt.ylabel("Py (a.u.)")

#%% Step 2: Polar plots of energy and momentum, without pointing correction

# for H polarization
# defining phi [0, pi], x-r angle
# defining theta [0, 2pi], (-z)-ryz angle
# note x and y are reversed

mom = np.linspace(dp, 1.1*pref, int(1.1*pref/dp)+1)  
E = np.linspace(0.5*dp**2, 0.5*(1.1*pref)**2, int(1.1*pref/dp)+1)  
width = mom[1] - mom[0]

theta = np.linspace(0, 360, 181)
phi = np.linspace(1, 179, 90) # not including 0 and 180 to avoid discontinuity

DCSSE = np.zeros((len(phi), len(theta), len(E))) # polar energy
DCSSP = np.zeros((len(phi), len(theta), len(mom))) # polar momentum

PHI, THETA, PP = np.meshgrid(phi, theta, mom, indexing="ij")
PHI, THETA, EE = np.meshgrid(phi, theta, E, indexing="ij")

for i in range(len(P[0])):
    if i % 10000 == 0: print("pt #{} / {}".format(i, len(P[0])))
    
    py = P[0][i]
    pz = P[1][i]
    px = P[2][i]
    pf = np.sqrt(px**2 + py**2 + pz**2)
    
    # check if it falls inside the grid
    if pf < mom.max() + 0.5*width:
       
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
DCSSP[:,0,:] += DCSSP[:,-1,:]
DCSSP[:,-1,:] = DCSSP[:,0,:]
DCSSE[:,0,:] += DCSSE[:,-1,:]
DCSSE[:,-1,:] = DCSSE[:,0,:]

arge = np.argmin(abs(eau-E))
argp = np.argmin(abs(np.sqrt(2*eau)-mom))

# energy slice
plt.figure()
plt.pcolor(PHI[:,:,arge], THETA[:,:,arge], np.sum(DCSSE[:,:,arge-1:arge+2], axis=2)/3, cmap="seismic", norm=LogNorm())
cm = plt.colorbar()

# momentum slice
plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], np.sum(DCSSP[:,:,argp-1:argp+2], axis=2)/3, cmap="seismic")
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

# theta slice (energy)
plt.figure()
argt = np.argmin(abs(180-theta))
plt.pcolor(PHI[:,argt,:], EE[:,argt,:]*27.211, DCSSE[:,argt], cmap="seismic")
plt.xlabel("Phi (deg)")
plt.ylabel("Energy (eV)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

# theta slice (momentum)
plt.figure()
plt.pcolor(PHI[:,argt,:], PP[:,argt,:], DCSSP[:,argt], cmap="seismic")
plt.xlabel("Phi (deg)")
plt.ylabel("Momentum (a.u.)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

#%% Step 3: Pointing correction, polar plots of energy and momentum

DCSSE = np.zeros((len(phi), len(theta), len(E))) # polar energy
DCSSP = np.zeros((len(phi), len(theta), len(mom))) # polar momentum

# rotation to correct for laser pointing angle
# a(Rx), b(Ry), c(Rz)
b, c = 0, 0
ca = np.cos(a*np.pi/180)
sa = np.sin(a*np.pi/180)
cb = np.cos(b*np.pi/180)
sb = np.sin(b*np.pi/180)
cc = np.cos(c*np.pi/180)
sc = np.sin(c*np.pi/180)

for i in range(len(P[0])):
    if i % 10000 == 0: print("pt #{} / {}".format(i, len(P[0])))
    
    py = P[0][i]
    pz = P[1][i]
    px = P[2][i]
    pf = np.sqrt(px**2 + py**2 + pz**2)
    
    # check if it falls inside the grid
    if pf < mom.max() + 0.5*width:
        
        # apply rotation
        px2 = px*cb*cc + py*(sa*sb*cc-ca*sc) + pz*(ca*sb*cc+sa*sc)
        py2 = px*cb*sc + py*(sa*sb*sc+ca*cc) + pz*(ca*sb*sc-sa*cc)
        pz2 = -px*sb + py*sa*cb + pz*ca*cb
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
DCSSP[:,0,:] += DCSSP[:,-1,:]
DCSSP[:,-1,:] = DCSSP[:,0,:]
DCSSE[:,0,:] += DCSSE[:,-1,:]
DCSSE[:,-1,:] = DCSSE[:,0,:]

arge = np.argmin(abs(eau-E))
argp = np.argmin(abs(np.sqrt(2*eau)-mom))

# energy slice
plt.figure()
plt.pcolor(PHI[:,:,arge], THETA[:,:,arge], np.sum(DCSSE[:,:,arge-1:arge+2], axis=2)/3, cmap="seismic", norm=LogNorm())
cm = plt.colorbar()

# momentum slice
plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], np.sum(DCSSP[:,:,argp-1:argp+2], axis=2)/3, cmap="seismic")
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

# theta slice (energy)
plt.figure()
argt = np.argmin(abs(180-theta))
plt.pcolor(PHI[:,argt,:], EE[:,argt,:]*27.211, DCSSE[:,argt], cmap="seismic")
plt.xlabel("Phi (deg)")
plt.ylabel("Energy (eV)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

# theta slice (momentum)
plt.figure()
plt.pcolor(PHI[:,argt,:], PP[:,argt,:], DCSSP[:,argt], cmap="seismic")
plt.xlabel("Phi (deg)")
plt.ylabel("Momentum (a.u.)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

np.savez(datapath + "sim_mpol_{}_{}.npz".format(np.abs(a), int(beta*100)), mpol=DCSSP, phi=phi, theta=theta, mom=mom)

#%% Step 4: divide by the line theta = 90 deg, and save

sm = np.sum(DCSSP[:,:,argp-n0:argp+n0+1], axis=2) / (2*n0+1)

# smooth the edges in phi
sm[0] = ss.savgol_filter(sm[0], 11, 1)
sm[1] = ss.savgol_filter(sm[1], 11, 1)
sm[-1] = ss.savgol_filter(sm[-1], 11, 1)
sm[-2] = ss.savgol_filter(sm[-2], 11, 1)

# full smoothing
sphere = sgolay2.SGolayFilter2(11, 1)(sm)

# momentum slice (smoothed)
plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], sphere, cmap="seismic")
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Yield (arb)")

# normalization
argt = np.argmin(abs(theta-90))
corr = sphere / np.average(sphere[:,argt-1:argt+2], axis=1)[:,np.newaxis]

# momentum slice (normalized)
plt.figure()
plt.pcolor(PHI[:,:,argp], THETA[:,:,argp], corr, cmap="bone")
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Normalized Yield (arb)")

# save simulated results
np.savez(datapath + "mod_{}_{}.npz".format(np.abs(a), int(beta*100)), corr=corr)

#%% Extra stuff

# momentum slice
plt.figure()
plt.pcolor(PHI[:,:,arge], THETA[:,:,arge], sphere, cmap="seismic")
plt.plot(90 + np.cos(theta*np.pi/180)*4, theta, c="w", ls="--", lw=1.5)
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")

bet = np.zeros(sphere.shape)
betsim = np.zeros(sphere.shape)

for i in range(len(theta)):
    
    offset = np.cos(np.radians(theta[i])) * (4*np.pi/180)

    def legendre(x, a):
        return a * (1 + beta*0.5*(3*np.cos(x-offset)**2 - 1))
    
    bet[:,i] = legendre(np.radians(phi), 1)
    
    #then we fit the simulation which is not perfect, so it has to be fitted
    optim = optimize.curve_fit(legendre, np.radians(phi),
                               sphere[:,i] / sphere[:,i].max(), p0=[1],
                               sigma = np.sqrt(sphere[:,i]) / sphere[:,i].max())
    betsim[:,i] = legendre(np.radians(phi), optim[0])

plt.figure()
plt.pcolor(PHI[:,:,arge], THETA[:,:,arge], bet, cmap="seismic")
plt.plot(90 + np.cos(theta*np.pi/180)*4, theta, c="w", ls="--", lw=1.5)
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Yield")

plt.figure()
plt.pcolor(PHI[:,:,arge], THETA[:,:,arge], betsim, cmap="seismic")
plt.plot(90 + np.cos(theta*np.pi/180)*4, theta, c="w", ls="--", lw=1.5)
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Yield")

plt.figure()
plt.pcolor(PHI[:,:,arge], THETA[:,:,arge], (betsim-betsim/bet)/(bet-1), cmap="bone")
plt.xlabel("Phi (deg)")
plt.ylabel("Theta (deg)")
cm = plt.colorbar()
cm.set_label("Modulation")

argt = np.argmin(abs(phi-10))

plt.figure()
plt.plot(theta, (betsim/bet)[argt])
plt.xlabel("Phi (deg)")
plt.ylabel("Yield")

argt = np.argmin(abs(phi-90))
plt.plot(theta, (betsim/bet)[argt])
argt = np.argmin(abs(phi-150))
plt.plot(theta, (betsim/bet)[argt])

