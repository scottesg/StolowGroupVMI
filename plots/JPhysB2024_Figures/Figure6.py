#%% Imports

import os
os.chdir(r'../..')

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150

from VMI3D_IO import genT
from VMI3D_Fitting import fit_gauss, gauss
from VMI3D_Functions import histxy

dt = 0.5
dqdim = 4096
stksize = 64000
rpos = 1684

#%% Load

tx = genT(dqdim, 1)
t = genT(dqdim, dt)

rpath = r"C:\Users\Scott\Python\VMI\documents\Calibration2024Paper\FX1/"

ch2pos = np.load(rpath + 'tc.npy')
tca = np.load(rpath + 'tca.npy')
hits = np.load(rpath + "hits.npy")

ch2pos = rpos - ch2pos

#%% Filter Hits

pll = 810
pul = 900

xp, yp = histxy(hits[:,0], [0, 2000], nbin=20000)

plt.figure()
plt.plot(xp, yp)
plt.vlines([pll, pul], 0, np.max(yp), color='red')
plt.xlim([0.9*pll, 1.1*pul])

hll = 0.1#0.01#0.09
hul = 0.4#0.04#0.5

xh, yh = histxy(hits[:,1], [0, 1], nbin=2000)

plt.figure()
plt.plot(xh, yh)
plt.vlines([hll, hul], 0, np.max(yh), color='red')
plt.xlim([0.9*hll, 1.1*hul])

wll = 1.06#1.06#1.0
wul = 1.38#1.38#1.45

xw, yw = histxy(hits[:,2], [0, 2], nbin=2000)

plt.figure()
plt.plot(xw, yw)
plt.vlines([wll, wul], 0, np.max(yw), color='red')
plt.xlim([0.9*wll, 1.1*wul])

#%% Ref Filters

# define acceptable time range
macc = 0.05
dacc = 0.8
acc = [macc-dacc, macc+dacc]

# define acceptable amplitude range
macca = 0.11766
dacca = 0.0006
acca = [macca-dacca, macca+dacca]

plt.figure()
xx, yy = histxy(ch2pos, xrange=[-5, 5], nbin=500)
plt.plot(xx, yy)
plt.vlines(acc, 0, np.max(yy), color='red')
plt.title("Reference Peak Time Range to Accept")
plt.xlabel("Relative Reference Peak Time ({} ns)".format(dt))

plt.figure()
xx, yy = histxy(tca, xrange=[0.115, 0.12], nbin=500)
plt.plot(xx, yy)
plt.vlines(acca, 0, np.max(yy), color='red')
plt.title("Reference Peak Amplitude Range to Accept")
plt.xlabel("Reference Peak Amplitude")

#%% Apply

c1 = hits[:,0]>pll
c2 = hits[:,0]<pul
c3 = hits[:,1]>hll
c4 = hits[:,1]<hul
c5 = hits[:,2]>wll
c6 = hits[:,2]<wul

a1 = ch2pos>acc[0]
a2 = ch2pos<acc[1]
a = a1 & a2

b1 = tca>acca[0]
b2 = tca<acca[1]
b = b1 & b2

ab = a & b

tcinds = np.argwhere(ab)
hitinds = hits[:,5]

c7 = [hitinds[i] in tcinds for i in range(len(hitinds))]

c = c1 & c2 & c3 & c4 & c5 & c6 & c7

#c = np.ones(len(hits)).astype(bool)

hitsf = hits[c]
np.save(rpath + "hitsf.npy", hitsf)

#%% Apply correction

hcf = []
nstk = int(len(ch2pos)/stksize)
for i in range(0, nstk):
    hitsfi = hitsf[hitsf[:,4]==i]
    for j in range(0, stksize):
        hj = hitsfi[hitsfi[:,5]==j]
        if j % 5000 == 0: print("Shot #{} / {}".format(j, stksize))
        n = i*stksize + j
        tci = ch2pos[n]
        tci = tci*dt
        hj[:,0] += tci
        hcf.extend(hj)

hcf = np.array(hcf)
np.save(rpath + "hitsc.npy", hcf)

#%% load

huf = np.load(rpath + "hitsf.npy")
hcf = np.load(rpath + "hitsc.npy")

#%%

p = 820
d = 20
plt.figure()
xu, yu = histxy(huf[:,0], [p-d, p+d], nbin=500)
xc, yc = histxy(hcf[:,0], [p-d, p+d], nbin=500)
plt.plot(xu, yu)
plt.plot(xc, yc)
plt.xlabel("Time (ns)")

#%%

p1 = 820
dd = 5

xc, yc = histxy(hcf[:,0], [p1-dd, p1+dd], nbin=400)
ymax = np.max(yc)

pp1 = 195
d1 = 6
xf1 = xc[pp1-d1:pp1+d1]
yf1 = yc[pp1-d1:pp1+d1]
fit1, params1, cov1 = fit_gauss(xf1, yf1, 0, 350, 820, 0.1, czero=True)
cen = params1[1]

xf2 = xc[pp1-d1*3:pp1+d1*3]
fit2 = gauss(xf2, 0, *params1)

#%%

plt.figure()
plt.plot(1000*(xc-cen), yc/ymax, lw=4, color='black')
plt.xlabel("Photon Arrival Time (ps)", fontsize=20)
plt.ylabel("Counts (Normalized)", fontsize=20)
plt.plot(1000*(xf2-cen), fit2/ymax, lw=2, color='red', ls=':', marker='x', markersize=0)
plt.plot(1000*(xf1-cen), fit1/ymax, lw=2, color='red', marker='x', markersize=10, label="\u03C3 = {:1.1f} ps".format(np.abs(1000*params1[2])))
plt.gca().tick_params(labelsize=20, width=2, length=6)
#plt.gca().get_yaxis().set_visible(False)
plt.xlim([-1200, 1200])
leg = plt.gca().legend(fontsize=20)
plt.gca().tick_params(labelsize=15)
leg.set_draggable(1)

