#%% Imports

import os
os.chdir(r'../..')

import matplotlib.pyplot as plt
from VMI3D_IO import readwfm, readimg

#%% Figure 1: Trace and Image for VMI Chamber

path = r"C:\Users\Scott\Python\VMI\data\20221003_VMI3DPaperData/"
wfmpath = path + "wfmstk.uint16"
wfms = readwfm(wfmpath, 2048, groupstks=True)

plt.figure()
plt.plot(-1*wfms[218], lw=5)
plt.xlim([1050, 1230])
plt.ylim([-35000, -20000])

plt.figure()
ims = readimg(path + "imstk.uint8", 512)
im = ims[10]
plt.imshow(im, cmap='Greys', vmin=5, vmax=250)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
