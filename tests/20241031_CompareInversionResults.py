#%% Imports

import os
os.chdir("..")

import numpy as np
import matplotlib.pyplot as plt
from Loadmat import loadmat
import polarTransform as pt
from DAVIS import create_circular_mask

#%% Load data [vmistats structure from both python and matlab scripts]

path = r"C:\Users\Scott\Python\VMI\data\20241010/"

dataML = loadmat(path + "MLTest.mat")['vmistats']
dataPY = np.load(path + "PYTest.npy", allow_pickle=True).item()

#%% Prepare

imgn = 20

IrPY = dataPY['Ir']
delaysPY = dataPY['delays']
rPY = dataPY['r']

imrawPY = dataPY['imstks'][7]
iminPY = dataPY['imstks']['difstk'] # not smoothed, symmetrized
iminpolPY = dataPY['imstks']['iminpol']
imoutpolPY = dataPY['imstks']['imoutpol']
imoutPY = dataPY['imstks']['idifstk']


IrML = dataML['Ir']
delaysML = dataML['delays']
rML = dataML['r']

imrawML = dataML['imstks']['img']
iminML = dataML['imstks']['difstk']
imoutML = dataML['imstks']['idifstk']

MLDim = dataML['difstk_imsize'][0]
iminpolML = pt.convertToPolarImage(iminML[imgn], finalAngle = np.pi,
                                   finalRadius = int(MLDim/2),
                                   angleSize=180, radiusSize=int(MLDim/0.4))[0]

imoutpolML = pt.convertToPolarImage(imoutML[imgn], finalAngle = np.pi,
                                    finalRadius = int(MLDim/2),
                                    angleSize=180, radiusSize=int(MLDim/0.4))[0]

#%% Images

fig, axs = plt.subplots(2,2)
fig.suptitle("Python Results")

axs[0,0].imshow(iminPY[imgn])
axs[0,0].set_title("Input Image")
axs[0,1].imshow(iminpolPY[imgn])
axs[0,1].set_title("Input Image (Polar)")
axs[1,1].imshow(imoutpolPY[imgn])
axs[1,1].set_title("Output Image (Polar)")
axs[1,0].imshow(imoutPY[imgn])
axs[1,0].set_title("Output Image")

# plt.figure()
# plt.title("Python Raw Image")
# plt.imshow(imrawPY[imgn])

fig, axs = plt.subplots(2,2)
fig.suptitle("Matlab Results")

axs[0,0].imshow(iminML[imgn])
axs[0,0].set_title("Input Image")
axs[0,1].imshow(iminpolML)
axs[0,1].set_title("Input Image (Polar)")
axs[1,1].imshow(imoutpolML)
axs[1,1].set_title("Output Image (Polar)")
axs[1,0].imshow(imoutML[imgn])
axs[1,0].set_title("Output Image")

mask = create_circular_mask(MLDim, MLDim, rin=147, rout=192)
normPY = np.mean(mask*imoutPY[imgn])
normML = np.mean(mask*imoutML[imgn])
vmax = 50

fig, axs = plt.subplots(1,2)
fig.suptitle("Inverted Images")

axs[0].imshow(imoutPY[imgn]/normPY, vmax=vmax)
axs[0].set_title("Python")
axs[1].imshow(imoutML[imgn]/normML, vmax=vmax)
axs[1].set_title("Matlab")

#%% Spectra

maxPY = np.mean(IrPY[imgn][810:896])
maxML = np.mean(IrML[imgn][115:128])

fig, ax = plt.subplots()
plt.plot(rPY, IrPY[imgn]/maxPY, label="Python")
plt.plot(rML, IrML[imgn]/maxML, label="Matlab")
plt.xlabel("Radius (pix)")
plt.ylabel("Amplitude")

leg = ax.legend()
leg.set_draggable(1)

