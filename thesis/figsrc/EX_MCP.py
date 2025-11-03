#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from VMI3D_Functions import (histxy)

#%% Load data

pts = np.load("EX_Xenon31ns_20220831_1823_T1.pt.npy")

#%% Histograms

x, phd = histxy(pts[:,3], [0, 0.29], 1000)
x, phd2 = histxy(pts[:,4], [0, 0.29], 1000)

phd = phd/max(phd)
phd2 = phd2/max(phd2)
x = x/0.29

#%% Figure

FSlb = 20
FStk = 15

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(x, phd, lw=3, color='black', label="PHD (Camera)")
ax.plot(x, phd2, lw=3, color='red', label="PHD (Pickoff)")
ax.set_xlim([0, 1])
ax.set_ylim([-0.05, 1.1])
ax.tick_params(labelsize=FStk, width=2, length=4)
ax.set_xlabel("Amplitude (arb)", fontsize=FSlb)
ax.set_ylabel("Counts (normalized)", fontsize=FSlb)
ax.legend(fontsize=FStk)

fig.set_tight_layout(True)
