#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from VMI3D_Fitting import fit_gauss, gauss
from VMI3D_Functions import histxy

#%% Load

hcf = np.load("DP_PhotonScatter_hitsc.npy")

#%% Processing

p = 820
d = 5

xc, yc = histxy(hcf[:,0], [p-d, p+d], nbin=400)
ymax = np.max(yc)

pp = 195
dd = 6
xf1 = xc[pp-dd:pp+dd]
yf1 = yc[pp-dd:pp+dd]
fit1, params1, cov1 = fit_gauss(xf1, yf1, 0, 350, 820, 0.1, czero=True)
cen = params1[1]

xf2 = xc[pp-dd*3:pp+dd*3]
fit2 = gauss(xf2, 0, *params1)

#%% Plot

fslabel = 20
fstick = 20
fsleg = 20

plt.figure(figsize=(12, 8), layout='tight')
plt.plot(1000*(xc-cen), yc/ymax, lw=4, color='black')
plt.xlabel("Photon Arrival Time (ps)", fontsize=fslabel)
plt.ylabel("Counts (Normalized)", fontsize=fslabel)
plt.plot(1000*(xf2-cen), fit2/ymax, lw=2, color='red', ls=':', marker='x', markersize=0)
plt.plot(1000*(xf1-cen), fit1/ymax, lw=2, color='red', marker='x', markersize=10, label="\u03C3 = {:1.1f} ps".format(np.abs(1000*params1[2])))
plt.gca().tick_params(labelsize=fstick, width=2, length=6)
plt.xlim([-1200, 1200])
plt.gca().legend(fontsize=fsleg)
plt.gca().tick_params(labelsize=fstick)

