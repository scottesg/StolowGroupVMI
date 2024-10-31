#%% Imports
import os
os.chdir("..")

import numpy as np
import matplotlib.pyplot as plt
from Loadmat import loadmat

#%% Load data [vmistats structure from both python and matlab scripts]

path = r"C:\Users\Scott\Python\VMI\data\20241010/"

dataML = loadmat(path + "MLTest.mat")['vmistats']
dataPY = np.load(path + "PYTest.npy", allow_pickle=True).item()

#%% Prepare

imgn = 20

IrPY = dataPY['Ir']
delaysPY = dataPY['delays']
rPY = dataPY['r']

IrML = dataML['Ir'][:,:156]
delaysML = dataML['delays']
rML = dataML['r'][:156]

#%% plot TR-PES (r[pix],t[ps]), PY

plot3d = True

upperlimit_percent_2d = 100

fig = plt.figure()
if len(delaysPY)>1:
    if plot3d:
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(rPY, delaysPY);
        ax.plot_surface(X, Y, IrPY, cmap='nipy_spectral')
        ax.set_zlabel("Intensity")
        ax.view_init(elev=30, azim=-125, roll=0)
    else:
        ax = plt.axes()
        X, Y = np.meshgrid(rPY, delaysPY);
        ax.pcolormesh(X, Y, IrPY, cmap='nipy_spectral', vmax=np.percentile(IrPY,upperlimit_percent_2d))
    ax.set_ylabel("delays, [ps]")
else:
    plt.plot(rPY, IrPY)
ax.set_xlabel("ring radius, [pix]")
ax.set_title("TRPES (Python)")

#%% plot TR-PES (r[pix],t[ps]), ML

upperlimit_percent_2d = 100

fig = plt.figure()
if len(delaysML)>1:
    if plot3d:
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(rML, delaysML);
        ax.plot_surface(X, Y, IrML, cmap='nipy_spectral')
        ax.set_zlabel("Intensity")
        ax.view_init(elev=30, azim=-125, roll=0)
    else:
        ax = plt.axes()
        X, Y = np.meshgrid(rML, delaysML);
        ax.pcolormesh(X, Y, IrML, cmap='nipy_spectral', vmax=np.percentile(IrML,upperlimit_percent_2d))
    ax.set_ylabel("delays, [ps]")
else:
    plt.plot(rML, IrML)
ax.set_xlabel("ring radius, [pix]")
ax.set_title("TRPES (Matlab)")

