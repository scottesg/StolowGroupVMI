#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from VMI3D_PeakFinding import fitting_routine
from numpy.fft import fft, ifft

#%% Load and Process

S0 = np.load("DP_Fitting_S0.npy")
S1pad = np.load("DP_Fitting_S1.npy")
trace = np.load("DP_Fitting_Trace.npy")

# set time axis
tracelen = 4096
dt = 0.25 # ns, sampling interval
t = np.arange(0, tracelen)*dt
tpad = np.hstack((-t[::-1]-1, t, t[-1]+1+t))
F = interpolate.interp1d(tpad, S1pad.real) # interpolation function on S1, for fitting

t0 = 529 # position of reference peak
refwidth = 1.7 # ns, reference width fwhm
gausstresh = 0.3 # sets height threshold for reference+background peak
nmax = 9 # maximum number of peaks to fit (up to 11)
SNR = 15 # sets lower amplitude limit on peaks
ind = 38
nsize = 5 # number of runs to process
 
hits, P = fitting_routine(nmax, trace, t, t0, refwidth, F, S0, SNR, dt, gausstresh)
decon = ifft(fft(trace) / fft(S0)).real
targ = int(np.round(t0/dt))
decon = np.roll(decon, targ)
fullfit = np.sum(P, axis=0)

trace = trace / max(trace)

#%% Plot

fig = plt.figure()
ax1 = plt.subplot2grid((1, 1), (0, 0))

r1 = 4*555
r2 = 4*625
markers = ['o', 's', '^', 'P', '*']
colors = ['C1', 'C2', 'C3', 'C4', 'C5']

FSt = 25
FSlb = 25
FSlg = 18
FStk = 18

#ax1.plot(t, trace, lw=2, color='red', ls=':', label="Average Reference Pulse")
ax1.plot(t, decon, lw=2, color='blue', label="Deconvolved Trace")
ax1.plot(t, fullfit, lw=2, ls='--', color='black', label="Total Fit")
for i in range(1, len(P)):
    ax1.plot(t[r1:r2], 0.04*i+0.5+P[i][r1:r2], lw=1, color=colors[i-1], ls='--')
    ax1.plot(r1/4 - 2, 0.04*i+0.5, marker=markers[i-1], markersize=8, ls='--', color=colors[i-1], label="Fit Function #{}".format(i))
ax1.set_xlabel("Time (ns)", fontsize=FSlb)
ax1.set_ylabel("Amplitude (arb)", fontsize=FSlb)
ax1.set_xlim([525, 650])
ax1.set_ylim([-0.15, 1.08])
ax1.minorticks_on()
ax1.tick_params(labelsize=FStk, which='both', width=2, length=4)
ax1.legend(loc=1, fontsize=FSlg)

