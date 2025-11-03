#%% Imports

import scipy.fft as fft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import tekwfm

#%% Load data

sgn = -1
trnum = 125

traces, tstart, tscale, tfrac, tdatefrac, tdate = tekwfm.read_wfm("EX_Bandwidth.wfm")
traces = traces.T
trlen = len(traces[0])

t = np.linspace(0, (trlen-1)*tscale*1e9, trlen)

trace = traces[trnum]*sgn
trace = trace/max(trace)

#%% Processing

sr = 1/tscale # samples/second
dur = trlen*tscale # time window

# Define filter 1
freq = 0.25 * 1e9 # GHz
order = 4
bfilter = signal.butter(order, freq, 'low', analog=False, output='sos', fs=sr)

# Perform FFT
yf = fft.rfft(trace)
yf = np.abs(yf)
yf = yf/max(yf)
tf = fft.rfftfreq(trlen, 1/sr) * 1e-9

# Apply Filter
y_butter = signal.sosfiltfilt(bfilter, trace)
b, a = signal.butter(order, freq, 'low', analog=False, fs=sr)
w, h = signal.freqz(b, a, fs=sr)
w = w * 1e-9
h = np.abs(h)
h = h/max(h)

# FFT After Filter
yf_butter = fft.rfft(y_butter)
yf_butter = np.abs(yf_butter)
yf_butter = yf_butter/max(yf_butter)

#######

# Define filter 2
freq = 0.5 * 1e9 # GHz
order = 4
bfilter = signal.butter(order, freq, 'low', analog=False, output='sos', fs=sr)

# Perform FFT
yf2 = fft.rfft(trace)
yf2 = np.abs(yf2)
yf2 = yf2/max(yf2)

# Apply Filter
y_butter2 = signal.sosfiltfilt(bfilter, trace)
b, a = signal.butter(order, freq, 'low', analog=False, fs=sr)
w2, h2 = signal.freqz(b, a, fs=sr)
w2 = w2 * 1e-9
h2 = np.abs(h2)
h2 = h2/max(h2)

# FFT After Filter
yf_butter2 = fft.rfft(y_butter2)
yf_butter2 = np.abs(yf_butter2)
yf_butter2 = yf_butter2/max(yf_butter2)

#%% Plot

FSt = 15
FSlb = 12
FStk = 10
FSlg = 10

col1 = 'black'
col2 = 'white'
pos1 = 0.015
pos2 = 0.95
fslett = 14

pldim = (2, 2)
fig = plt.figure(figsize=(12,6))
ax1 = plt.subplot2grid(pldim, (0, 0))
ax2 = plt.subplot2grid(pldim, (0, 1))
ax3 = plt.subplot2grid(pldim, (1, 0))
ax4 = plt.subplot2grid(pldim, (1, 1))

ax1.plot(t, trace, lw=1, color='black')
ax1.set_xlabel("Time (ns)", fontsize=FSlb)
ax1.set_title("(a) 10 GS/s Trace", fontsize=FSt)
ax1.set_ylabel("Amplitude (arb)", fontsize=FSlb)
ax1.tick_params(labelsize=FStk, width=2, length=4)
ax1.set_xlim([0, 500])
ax1.set_ylim([-0.25, 1.1])

ax2.plot(tf, yf, lw=1, color='black', label="FFT")
ax2.plot(w, h, lw=1, color='red', label="0.25 GHz filter")
ax2.plot(w2, h2, lw=1, color='blue', label="0.50 GHz filter")
ax2.set_xlabel("Frequency (GHz)", fontsize=FSlb)
ax2.set_title("(b) FFT and Filters", fontsize=FSt)
ax2.set_ylabel("Amplitude (arb)", fontsize=FSlb)
ax2.tick_params(labelsize=FStk, width=2, length=4)
ax2.set_xlim([-0.02, 2])
ax2.set_ylim([-0.05, 1.1])
ax2.legend(fontsize=FSlg)

ax3.plot(t, trace, lw=1, color='black', label="Unfiltered")
ax3.plot(t, y_butter/max(y_butter), lw=2, ls="--", color='red', label="Filtered (0.25 GHz)")
ax3.plot(t, y_butter2/max(y_butter2), lw=2, ls="--", color='blue', label="Filtered (0.50 GHz)")
ax3.set_title("(c) Filtered and Unfiltered Trace", fontsize=FSt)
ax3.set_xlabel("Time (ns)", fontsize=FSlb)
ax3.set_ylabel("Amplitude (arb)", fontsize=FSlb)
ax3.tick_params(labelsize=FStk, width=2, length=4)
ax3.set_xlim([135, 210])
ax3.set_ylim([-0.25, 1.1])
ax3.legend(fontsize=FSlg)

ax4.plot(tf, yf, lw=1, color='black', label="FFT")
ax4.plot(tf, np.abs(yf_butter), lw=1, ls="--", color='red', label="Filtered (0.25 GHz)")
ax4.plot(tf, np.abs(yf_butter2), lw=1, ls="--", color='blue', label="Filtered (0.50 GHz)")
ax4.set_title("(d) FFT of Filtered Trace", fontsize=FSt)
ax4.set_xlabel("Frequency (GHz)", fontsize=FSlb)
ax4.set_ylabel("Amplitude (arb)", fontsize=FSlb)
ax4.tick_params(labelsize=FStk, width=2, length=4)
ax4.set_xlim([-0.02, 0.8])
ax4.set_ylim([-0.02, 0.4])
ax4.legend(fontsize=FSlg)

# t1 = ax1.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=fslett)
# t2 = ax2.text(pos1+0.02, pos2-0.1, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=fslett)
# t3 = ax3.text(pos1, pos2, '(c)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=fslett)
# t4 = ax4.text(pos1+0.023, pos2, '(d)', ha='left', va='top', color=col1, transform=ax4.transAxes, fontsize=fslett)

# t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
# t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
# t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))
# t4.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

fig.set_tight_layout(True)
