#%% Setup
import os
os.chdir("..")

import numpy as np
import matplotlib.pyplot as plt
from VMI3D_IO import readcsv

pypath = r"C:\Users\Scott\Python\VMI\reduceddata\20241010\20241010_cs2test3/"
mlpath = r"C:\Users\Scott\Python\VMI\reduceddataml\data\20241010\20241010_cs2test3/"

trpesPYr = readcsv(pypath + "20241010_cs2test3.trpes.ke.csv")
trpesMLr = readcsv(mlpath + "20241010_cs2test3.trpes.ke.csv")

#%%

bkrange = -2
r1 = 0.0
r2 = 0.6

tPY = -1*trpesPYr[1:,0]
ePY = trpesPYr[0,1:]
trpesPY = trpesPYr[1:,1:]

tML = trpesMLr[1:,0]
eML = trpesMLr[0,1:]
trpesML = trpesMLr[1:,1:]

bkPY = np.mean(trpesPY[bkrange:,:], axis=0)
bkML = np.mean(trpesML[bkrange:,:], axis=0)

trpesPY = trpesPY - bkPY
trpesML = trpesML - bkML

trpesPY = trpesPY / np.percentile(trpesPY, 99.5)
trpesML = trpesML / np.percentile(trpesML, 99.5)

p1 = np.searchsorted(ePY, r1)
p2 = np.searchsorted(ePY, r2)
subPY = np.mean(trpesPY[:,p1:p2], axis=1)

p1 = np.searchsorted(eML, r1)
p2 = np.searchsorted(eML, r2)
subML = np.mean(trpesML[:,p1:p2], axis=1)

#%% PY plot

fig, ax = plt.subplots()
X, Y = np.meshgrid(ePY, tPY);
pcm = ax.pcolormesh(X, Y, trpesPY, cmap='nipy_spectral', vmax=np.percentile(trpesPY, 100))
ax.set_ylabel("delays, [ps]")
ax.set_xlabel("energy, [eV]")
ax.set_title("Python TRPES")

fig.colorbar(pcm, ax=ax)

#%% ML plot

fig, ax = plt.subplots()
X, Y = np.meshgrid(eML, tML);
pcm = ax.pcolormesh(X, Y, trpesML, cmap='nipy_spectral', vmax=np.percentile(trpesML, 100))
ax.set_ylabel("delays, [ps]")
ax.set_xlabel("energy, [eV]")
ax.set_title("MATLAB TRPES")

fig.colorbar(pcm, ax=ax)

#%% Range TRPES

fig, ax = plt.subplots()
ax.plot(tPY, subPY, label="{}:{} eV (Python)".format(r1, r2))
ax.plot(tML, subML, label="{}:{} eV (MATLAB)".format(r1, r2))

ax.set_title("TRPES between {} and {} eV".format(r1, r2), fontsize=25)
ax.set_xlabel("Time (ps)", fontsize=15)
ax.set_ylabel("Amplitude (arb)", fontsize=15)
ax.tick_params(labelsize=12)

plt.grid()
leg = ax.legend(fontsize=20)
leg.set_draggable(1)

